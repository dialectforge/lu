//! lu_codeexec -- privilege-separated code-exec broker (OP-CODE-EXEC Phase 6/7).
//!
//! The internet-facing guest (`lu_guest_http`) is hardened with
//! NoNewPrivileges/ProtectControlGroups/ProtectSystem=strict, so it CANNOT and
//! MUST NOT hold the X1 privilege (sudo -> systemd-run scope -> bwrap). This
//! broker owns that privilege in a SEPARATE local process. It:
//!   * listens on a localhost Unix socket (0600, owner-only), never the network;
//!   * accepts ONE JSON request per connection from the OWNER-gated guest tool;
//!   * two modes:
//!       - planner mode (default): {code, inputs} -> native X1 + bounded
//!         run->fix loop -> {ok, code, stdout, stderr, exit_code, ...};
//!       - raw mode ({raw:true} or {want_file}): single native X1 run returning
//!         the FULL sandbox result incl seccomp state and a produced file
//!         (base64) -- used for the G-X1 battery and file-download output.
//! It never loads the brain or the master key; it only executes candidate code
//! inside the fully isolated X1 sandbox. Privilege separation, not relaxation.
use lu4d::code_exec::planner::run_fix;
use lu4d::code_exec::{run, X1Input, X1Limits, X1Request};
use std::io::{Read, Write};
use std::os::unix::fs::PermissionsExt;
use std::os::unix::net::UnixListener;

fn b64(data: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut s = String::new();
    for c in data.chunks(3) {
        let b = [c[0], *c.get(1).unwrap_or(&0), *c.get(2).unwrap_or(&0)];
        let n = ((b[0] as u32) << 16) | ((b[1] as u32) << 8) | (b[2] as u32);
        s.push(T[((n >> 18) & 63) as usize] as char);
        s.push(T[((n >> 12) & 63) as usize] as char);
        s.push(if c.len() > 1 { T[((n >> 6) & 63) as usize] as char } else { '=' });
        s.push(if c.len() > 2 { T[(n & 63) as usize] as char } else { '=' });
    }
    s
}

fn parse_inputs(v: &serde_json::Value) -> Vec<X1Input> {
    v.get("inputs")
        .and_then(|x| x.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|i| {
                    Some(X1Input {
                        name: i.get("name")?.as_str()?.to_string(),
                        text: i.get("text")?.as_str()?.to_string(),
                    })
                })
                .collect()
        })
        .unwrap_or_default()
}

fn parse_limits(v: &serde_json::Value) -> X1Limits {
    let d = X1Limits::default();
    let l = v.get("limits");
    let g = |k: &str, dv: u64| l.and_then(|x| x.get(k)).and_then(|x| x.as_u64()).unwrap_or(dv);
    X1Limits {
        mem_mib: g("mem_mib", d.mem_mib as u64) as u32,
        pids: g("pids", d.pids as u64) as u32,
        cpu_cores: g("cpu_cores", d.cpu_cores as u64) as u32,
        wall_ms: g("wall_ms", d.wall_ms),
    }
}

fn main() {
    let sock = std::env::args().nth(1).unwrap_or_else(|| "/run/lu-codeexec/sock".to_string());
    let _ = std::fs::remove_file(&sock);
    let listener = UnixListener::bind(&sock).unwrap_or_else(|e| {
        eprintln!("[lu_codeexec] bind {}: {}", sock, e);
        std::process::exit(1);
    });
    let _ = std::fs::set_permissions(&sock, std::fs::Permissions::from_mode(0o600));
    eprintln!("[lu_codeexec] listening on {} (localhost socket, owner-only)", sock);
    for conn in listener.incoming() {
        let mut s = match conn {
            Ok(s) => s,
            Err(_) => continue,
        };
        let mut buf = String::new();
        if s.read_to_string(&mut buf).is_err() {
            continue;
        }
        let v: serde_json::Value = serde_json::from_str(&buf).unwrap_or(serde_json::Value::Null);
        let code = v.get("code").and_then(|x| x.as_str()).unwrap_or("");
        let raw = v.get("raw").and_then(|x| x.as_bool()).unwrap_or(false);
        let want_file = v.get("want_file").and_then(|x| x.as_str());
        let resp = if code.is_empty() {
            serde_json::json!({"ok": false, "error": "empty code"})
        } else if raw || want_file.is_some() {
            // Raw single-run: full sandbox result (G-X1 battery + file download).
            let req = X1Request {
                code: code.to_string(),
                inputs: parse_inputs(&v),
                want_file: want_file.map(|s| s.to_string()),
                limits: parse_limits(&v),
            };
            let r = run(&req);
            let file = r.file.as_ref().map(|f| serde_json::json!({
                "name": f.name, "size": f.size, "sha256": f.sha256, "bytes_b64": b64(&f.bytes),
            }));
            serde_json::json!({
                "ok": r.exit_code == 0,
                "exit_code": r.exit_code,
                "stdout": r.stdout,
                "stderr": r.stderr,
                "killed_by": r.killed_by,
                "timed_out": r.timed_out,
                "wall_ms": r.wall_ms,
                "seccomp": r.seccomp,
                "file": file,
            })
        } else {
            // Planner mode: native X1 + bounded run->fix.
            let rf = run_fix(code, &parse_inputs(&v), None);
            serde_json::json!({
                "ok": rf.ok,
                "code": rf.code,
                "stdout": rf.result.stdout,
                "stderr": rf.result.stderr,
                "exit_code": rf.result.exit_code,
                "killed_by": rf.result.killed_by,
                "timed_out": rf.result.timed_out,
                "attempts": rf.attempts,
                "fixes": rf.fixes,
            })
        };
        let _ = s.write_all(resp.to_string().as_bytes());
    }
}
