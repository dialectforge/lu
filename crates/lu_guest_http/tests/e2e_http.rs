//! OP-007 / Port U end-to-end HTTP test for the guest surface.
//!
//! Spawns the REAL `lu_guest_http` binary against a REAL v5 brain fixture
//! (built through the writer path, then opened read-only) and drives it
//! over a REAL TCP socket. Nothing here is mocked: the login gate, the
//! guest MCP trust gate, the narrate/ask pipeline and its idk-gate
//! refusal are all the production code paths.
//!
//! Why this exists: the canonical 235 GiB brain cannot be opened for
//! serving while an ingest run holds the box's RAM (docs/TRAINING.md
//! golden rule 1). This test proves every link of the chain EXCEPT the
//! size-dependent ones (real-brain recall quality and real-brain
//! latency), so the auth + privsep + refusal properties stay verifiable
//! at any time, on any machine, without touching the canonical brain.
//!
//! The HTTP client is hand-rolled over `std::net::TcpStream` on purpose:
//! the guest crate's dependency set is a privilege-separation surface, and
//! a test-only HTTP client is not worth widening it.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::Path;
use std::process::{Child, Command};
use std::time::{Duration, Instant};

use serde_json::{json, Value};

const OWNER_PASSPHRASE: &str = "e2e-owner-passphrase-not-a-real-secret";
const INVITE_A: &str = "E2E-INVITE-AAAA";
const INVITE_B: &str = "E2E-INVITE-BBBB";

/// Kill the child on drop so a failed assertion never leaks a server.
struct Server {
    child: Child,
    port: u16,
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Ask the OS for a free port, then release it for the server to bind.
fn free_port() -> u16 {
    let l = TcpListener::bind("127.0.0.1:0").expect("bind ephemeral");
    l.local_addr().expect("local_addr").port()
}

/// Minimal HTTP/1.1 request. Returns (status_code, body).
fn http(port: u16, method: &str, path: &str, bearer: Option<&str>, body: Option<&Value>) -> (u16, String) {
    let payload = body.map(|b| b.to_string()).unwrap_or_default();
    let mut req = format!(
        "{} {} HTTP/1.1\r\nHost: 127.0.0.1:{}\r\nConnection: close\r\n",
        method, path, port
    );
    if let Some(t) = bearer {
        req.push_str(&format!("Authorization: Bearer {}\r\n", t));
    }
    if body.is_some() {
        req.push_str("Content-Type: application/json\r\n");
        req.push_str(&format!("Content-Length: {}\r\n", payload.len()));
    }
    req.push_str("\r\n");
    req.push_str(&payload);

    let mut s = TcpStream::connect(("127.0.0.1", port)).expect("connect to guest server");
    s.set_read_timeout(Some(Duration::from_secs(120))).unwrap();
    s.write_all(req.as_bytes()).expect("write request");
    s.flush().unwrap();

    let mut raw = Vec::new();
    s.read_to_end(&mut raw).expect("read response");
    let text = String::from_utf8_lossy(&raw).to_string();

    let status: u16 = text
        .split_whitespace()
        .nth(1)
        .and_then(|c| c.parse().ok())
        .unwrap_or_else(|| panic!("no status line in response: {:?}", text));

    let body = match text.find("\r\n\r\n") {
        Some(i) => text[i + 4..].to_string(),
        None => String::new(),
    };
    (status, body)
}

fn http_json(port: u16, method: &str, path: &str, bearer: Option<&str>, body: Option<&Value>) -> (u16, Value) {
    let (status, body) = http(port, method, path, bearer, body);
    let v = serde_json::from_str(&body).unwrap_or(Value::Null);
    (status, v)
}

/// Build a real v5 brain DIRECTORY through the WRITER path, then drop the
/// writer so only committed SSTs remain — exactly the on-disk shape the
/// read-only reader is designed to open.
fn build_v5_fixture(dir: &Path) {
    use lu4d::bond::BondType;
    use lu4d::node::Confidence;
    use lu4d::Brain;

    std::fs::create_dir_all(dir.join("node_store")).unwrap();
    std::fs::create_dir_all(dir.join("bond_store")).unwrap();
    std::fs::create_dir_all(dir.join("wal")).unwrap();
    let fp = lu4d::manifest::KeyFingerprint::none();
    let mut m = lu4d::manifest::Manifest::new(fp);
    m.set_brain_format_v5();
    m.save_atomic(&dir.join(lu4d::manifest::MANIFEST_FILE_NAME)).unwrap();

    let mut brain = Brain::load_v5(dir, None).expect("v5 writer must boot");
    brain.teach("rust", BondType::IsA, "programming language", 0.95, Confidence::Verified);
    brain.teach("rust", BondType::RelatedTo, "memory safety", 0.9, Confidence::Verified);
    brain.teach("rust", BondType::RelatedTo, "borrow checker", 0.9, Confidence::Verified);
    brain.teach("python", BondType::IsA, "programming language", 0.95, Confidence::Verified);
    brain.flush_now().expect("flush_now must seal an L0 SST");
    // Build the keyword sidecar with the WRITER, as production does. The
    // read-only reader never builds one (that would be a write); it can
    // only install a sidecar whose stamp still matches the committed
    // frontier. A fixture without this would exercise the degraded-recall
    // path instead of the real one.
    brain
        .build_keyword_index_v5()
        .expect("writer must build the v5 keyword sidecar");
    drop(brain);
}

/// Snapshot every file under `dir` as path -> (len, mtime_nanos). Used to
/// prove the served workload mutated nothing on disk.
fn snapshot_dir(dir: &Path) -> std::collections::BTreeMap<std::path::PathBuf, (u64, i128)> {
    fn walk(dir: &Path, out: &mut std::collections::BTreeMap<std::path::PathBuf, (u64, i128)>) {
        let rd = match std::fs::read_dir(dir) {
            Ok(r) => r,
            Err(_) => return,
        };
        for ent in rd.flatten() {
            let p = ent.path();
            let md = match std::fs::metadata(&p) {
                Ok(m) => m,
                Err(_) => continue,
            };
            if md.is_dir() {
                walk(&p, out);
            } else {
                let mtime = md
                    .modified()
                    .ok()
                    .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                    .map(|d| d.as_nanos() as i128)
                    .unwrap_or(-1);
                out.insert(p, (md.len(), mtime));
            }
        }
    }
    let mut out = std::collections::BTreeMap::new();
    walk(dir, &mut out);
    out
}

fn start_server(brain_dir: &Path) -> Server {
    let port = free_port();
    let child = Command::new(env!("CARGO_BIN_EXE_lu_guest_http"))
        .arg("--brain")
        .arg(brain_dir)
        .arg("--port")
        .arg(port.to_string())
        .env("LU_GUEST_PASSPHRASE", OWNER_PASSPHRASE)
        .env("LU_GUEST_INVITES", format!("{},{}", INVITE_A, INVITE_B))
        .spawn()
        .expect("spawn lu_guest_http");
    let server = Server { child, port };

    // Wait for /health. The read-only open is fast on a fixture brain.
    let deadline = Instant::now() + Duration::from_secs(60);
    while Instant::now() < deadline {
        if TcpStream::connect(("127.0.0.1", port)).is_ok() {
            let (status, body) = http(port, "GET", "/health", None, None);
            if status == 200 && body.trim() == "ok" {
                return server;
            }
        }
        std::thread::sleep(Duration::from_millis(200));
    }
    panic!("guest server did not become healthy on port {}", port);
}

fn auth_token(port: u16, body: Value) -> Option<String> {
    let (status, v) = http_json(port, "POST", "/auth", None, Some(&body));
    if status != 200 {
        return None;
    }
    v.get("token").and_then(|t| t.as_str()).map(|s| s.to_string())
}

fn call_tool(port: u16, token: &str, name: &str, args: Value) -> Value {
    let (status, v) = http_json(
        port,
        "POST",
        "/mcp",
        Some(token),
        Some(&json!({"method": "tools/call", "params": {"name": name, "arguments": args}})),
    );
    assert_eq!(status, 200, "tools/call {} returned {}: {}", name, status, v);
    v
}

#[test]
fn guest_surface_end_to_end() {
    let dir = tempfile::tempdir().expect("tempdir");
    build_v5_fixture(dir.path());

    // Snapshot the committed brain BEFORE any serving happens.
    let before = snapshot_dir(dir.path());
    assert!(!before.is_empty(), "the fixture must have committed some files");

    let srv = start_server(dir.path());
    let port = srv.port;

    // ---- 1. The login gate actually gates ---------------------------
    let (status, _) = http_json(port, "POST", "/mcp", None, Some(&json!({"method": "tools/list"})));
    assert_eq!(status, 401, "/mcp must reject an unauthenticated request");

    let (status, _) = http_json(port, "POST", "/mcp", Some("not-a-real-token"), Some(&json!({"method": "tools/list"})));
    assert_eq!(status, 401, "/mcp must reject a forged bearer token");

    assert!(
        auth_token(port, json!({"passphrase": "wrong-passphrase"})).is_none(),
        "a wrong owner passphrase must not yield a token"
    );
    assert!(
        auth_token(port, json!({"invite_code": "NOT-A-SEEDED-CODE"})).is_none(),
        "an unseeded invite code must not yield a token"
    );

    // ---- 2. Owner login ---------------------------------------------
    let (status, v) = http_json(port, "POST", "/auth", None, Some(&json!({"passphrase": OWNER_PASSPHRASE})));
    assert_eq!(status, 200, "owner passphrase must authenticate");
    assert_eq!(v["principal"], "owner", "owner principal label");
    let owner_token = v["token"].as_str().expect("owner token").to_string();

    // ---- 3. Guest invite login, single-use --------------------------
    let (status, v) = http_json(port, "POST", "/auth", None, Some(&json!({"invite_code": INVITE_A})));
    assert_eq!(status, 200, "a seeded invite code must authenticate a guest");
    assert_eq!(v["principal"], format!("guest:{}", INVITE_A));
    let guest_token = v["token"].as_str().expect("guest token").to_string();

    assert!(
        auth_token(port, json!({"invite_code": INVITE_A})).is_none(),
        "invite codes must be SINGLE-USE — a redeemed code must not authenticate again"
    );
    assert!(
        auth_token(port, json!({"invite_code": INVITE_B})).is_some(),
        "an unredeemed seeded code must still work"
    );

    // ---- 4. The guest surface is exactly three tools ----------------
    let (status, v) = http_json(port, "POST", "/mcp", Some(&guest_token), Some(&json!({"method": "tools/list"})));
    assert_eq!(status, 200);
    let names: Vec<String> = v["tools"]
        .as_array()
        .expect("tools array")
        .iter()
        .map(|t| t["name"].as_str().unwrap_or_default().to_string())
        .collect();
    // NOTE: lu4d_image_describe / lu4d_file_read / lu4d_code were added to the
    // guest surface after this test was first written; assert the ask+session
    // core is present and (critically) that NO admin tool leaks, rather than a
    // brittle exact-vector match.
    for required in ["lu4d_ask", "lu4d_session_start", "lu4d_session_end"] {
        assert!(names.iter().any(|n| n == required),
            "guest surface must expose {}; got {:?}", required, names);
    }
    for admin in ["lu4d_teach", "lu4d_user_statement", "lu4d_save", "lu4d_scar"] {
        assert!(!names.iter().any(|n| n == admin),
            "admin tool {} must never appear on the guest surface; got {:?}", admin, names);
    }

    // ---- 5. Admin tools are unreachable, for BOTH principals --------
    // (They are also physically absent from the binary — tests/privsep.rs.)
    for tok in [&owner_token, &guest_token] {
        for admin in ["lu4d_teach", "lu4d_configure", "lu4d_revoke_by_source", "lu4d_trust_set"] {
            let v = call_tool(port, tok, admin, json!({}));
            assert_eq!(
                v["error"]["code"], -32601,
                "admin tool {} must be refused on the guest surface (got {})",
                admin, v
            );
        }
    }

    // ---- 6. Recall actually resolves against the committed frontier --
    // The assertion is on RECALL MECHANICS, not on narration wording: a
    // 4-bond fixture cannot produce a representative sentence, but it can
    // and must produce query seeds. Empty seeds is the exact symptom of
    // the reader failing to install the writer-built keyword sidecar
    // (NodeId divergence), which silently degrades every answer to the
    // idk gate. Narration quality against the canonical brain is covered
    // by scripts/verify_port.sh, which is the only valid rig for it.
    let v = call_tool(port, &owner_token, "lu4d_ask", json!({"query": "what is rust"}));
    let answer = v["result"]["content"][0]["text"].as_str().unwrap_or_default().to_lowercase();
    let seeds = &v["result"]["structuredContent"]["ask"]["recall"]["query_seeds"];
    assert!(
        seeds.as_array().map(|a| !a.is_empty()).unwrap_or(false),
        "a taught term must resolve to query seeds through the read-only reader \
         (empty seeds means the keyword sidecar was rejected); got seeds={} answer={:?}",
        seeds,
        answer
    );
    assert!(
        !answer.contains("nothing in my brain matched"),
        "a taught term must not hit the no-match path; got: {:?}",
        answer
    );

    // ---- 7. The idk gate refuses what Lu does not know --------------
    let v = call_tool(
        port,
        &guest_token,
        "lu4d_ask",
        json!({"query": "what is the flurgle capacitance of a zorbnak manifold"}),
    );
    let refusal = v["result"]["content"][0]["text"].as_str().unwrap_or_default().to_lowercase();
    assert!(
        refusal.contains("don't have") || refusal.contains("do not have") || refusal.contains("meaningful information"),
        "the idk gate must refuse an unknown topic rather than fabricate; got: {:?}",
        refusal
    );

    // ---- 8. Session tools work on the guest surface -----------------
    let v = call_tool(port, &guest_token, "lu4d_session_start", json!({}));
    assert!(
        v["result"]["structuredContent"]["session_id"].is_number(),
        "session_start must return an id; got {}",
        v
    );
    let v = call_tool(port, &guest_token, "lu4d_session_end", json!({}));
    assert!(v["result"].is_object(), "session_end must return a result; got {}", v);

    // ---- 9. Serving must not have written the brain -----------------
    // The structural guarantee is asserted in src/read_only_reader.rs; here
    // we assert the operational consequence over a REAL served workload:
    // not one byte of the brain directory changed while the server booted,
    // authenticated users, answered, refused, and ran sessions.
    let after = snapshot_dir(dir.path());
    assert_eq!(
        before, after,
        "serving must not create, delete, or modify ANY file in the brain dir"
    );
    assert!(
        !dir.path().join(".kw_rebuild_runs").exists(),
        "serving must not create a keyword-rebuild scratch dir in the brain"
    );
}


/// OP-ROUTE-GUEST regression test.
///
/// `QueryRouter::route` has always set `RoutingDecision::needs_math` for
/// these queries, but narrate() consumes that flag only as a NEGATIVE
/// suppressor and the sole positive math handler lives in `lu4d::mcp`,
/// which `default-features = false` compiles OUT of the guest binary.
/// The result: "2+2" fell through to graph recall, matched nothing, and
/// hit the idk floor. This test pins the short-circuit at the guest MCP
/// boundary -- and, just as importantly, pins that the honesty floor did
/// NOT regress into a math-shaped catch-all.
#[test]
fn guest_math_short_circuit() {
    let dir = tempfile::tempdir().expect("tempdir");
    build_v5_fixture(dir.path());
    let srv = start_server(dir.path());
    let port = srv.port;

    let token = auth_token(port, json!({"passphrase": OWNER_PASSPHRASE}))
        .expect("owner token");

    // ---- arithmetic is COMPUTED, never recalled ---------------------
    for (q, expect) in [
        ("2+2", "4"),
        ("what is 17*3", "51"),
        ("what is 7 plus 5", "12"),
        ("what is 25 percent of 80", "20"),
        ("step by step, what is 47 times 13", "611"),
    ] {
        let v = call_tool(port, &token, "lu4d_ask", json!({"query": q}));
        let src = v["result"]["structuredContent"]["source"].as_str().unwrap_or_default();
        assert_eq!(
            src, "arithmetic_primitive",
            "query {:?} did not reach the arithmetic primitive; got source={:?} body={}",
            q, src, v
        );
        let text = v["result"]["content"][0]["text"].as_str().unwrap_or_default();
        assert!(
            text.contains(expect),
            "query {:?} should compute to {:?}; got {:?}",
            q, expect, text
        );
        assert!(
            text.contains("not recalled"),
            "query {:?} lost its provenance disclosure; got {:?}",
            q, text
        );
    }

    // ---- the honesty floor must NOT be swallowed by math ------------
    let v = call_tool(
        port,
        &token,
        "lu4d_ask",
        json!({"query": "what is the flurgle capacitance of a zorbnak manifold"}),
    );
    let src = v["result"]["structuredContent"]["source"].as_str().unwrap_or_default();
    assert_ne!(
        src, "arithmetic_primitive",
        "a nonsense query must not be misrouted into the math primitive; got {}",
        v
    );
    let refusal = v["result"]["content"][0]["text"].as_str().unwrap_or_default().to_lowercase();
    assert!(
        refusal.contains("don't have")
            || refusal.contains("do not have")
            || refusal.contains("meaningful information")
            || refusal.contains("won't fabricate")
            || refusal.contains("doesn't match"),
        "the idk gate must still refuse an unknown topic; got {:?}",
        refusal
    );

    // ---- a plain definition query still goes through narrate --------
    let v = call_tool(port, &token, "lu4d_ask", json!({"query": "what is rust"}));
    let src = v["result"]["structuredContent"]["source"].as_str().unwrap_or_default();
    assert_ne!(
        src, "arithmetic_primitive",
        "a definition query must not be misrouted into the math primitive; got {}",
        v
    );

    // ---- a greeting still hits the greet guard, not math -------------
    let v = call_tool(port, &token, "lu4d_ask", json!({"query": "hello"}));
    let src = v["result"]["structuredContent"]["source"].as_str().unwrap_or_default();
    assert_eq!(
        src, "greet_refusal",
        "a greeting must still reach greet_refusal; got {}",
        v
    );
}
