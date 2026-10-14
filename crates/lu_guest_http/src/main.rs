//! `lu_guest_http` — the PUBLIC-facing guest binary of the OP-007 / Port U
//! two-binary privilege split.
//!
//! Committed design (repo OPEN_PORTS.md OP-007; internal notes:
//!   PWA frontend -> HTTP shim -> guest MCP trust gate -> Lu narrate/ask
//!   pipeline. Users talk to Lu directly, not to Claude-calling-Lu.
//!
//! THIS scaffold is deliberately LOCALHOST-ONLY. No public :443, no Caddy,
//! no DNS — those are later OP-007 steps and are out of scope here. The
//! binary refuses to bind anything but a loopback address.
//!
//! Privilege separation: this crate builds `lu4d` with
//! `default-features = false`, so the admin `mcp` module is compiled out.
//! The only tools reachable here are `lu4d_ask` + session tools, gated by
//! `guest_mcp::dispatch`. See `tests/privsep.rs`.

mod auth;
mod guest_mcp;

use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, RwLock};

use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{json, Value};

use auth::{AuthState, Principal};
use lu4d::encrypt::{MasterKey, DEFAULT_KEYFILE_PATH};
use lu4d::Brain;

/// How often the refresh task polls the committed MANIFEST frontier.
const REFRESH_POLL_SECS: u64 = 10;

/// Read the committed-frontier epoch from the on-disk MANIFEST. Returns 0
/// if the manifest can't be read (transient during the writer's atomic
/// rename); a 0 simply defers the swap to the next poll.
fn current_frontier_epoch(manifest_path: &std::path::Path) -> u64 {
    match lu4d::manifest::Manifest::load(manifest_path) {
        Ok(Some(m)) => m.peek_next_epoch(),
        _ => 0,
    }
}

struct AppState {
    /// OP-007 live-serving: the current read-only snapshot. Held as
    /// RwLock<Arc<Mutex<Brain>>> so the background refresh task can swap
    /// in a freshly re-opened reader without blocking in-flight requests
    /// (the RwLock is taken only to clone/replace the Arc; the heavy
    /// narrate work runs under the per-snapshot Mutex after release).
    brain: RwLock<Arc<Mutex<Brain>>>,
    auth: AuthState,
    brain_label: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ---- args -------------------------------------------------------
    let mut brain_path: Option<PathBuf> = None;
    let mut keyfile = PathBuf::from(DEFAULT_KEYFILE_PATH);
    let mut port: u16 = 8787;
    // OP-CURIOSITY-1: directory for the durable wonder journal. This is
    // the ONLY writable path this process is given, and it is outside
    // the brain directory on purpose — see the privilege note in
    // `lu4d::wonder_sidecar`.
    let mut wonder_queue_dir: Option<PathBuf> = None;

    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--brain" => brain_path = args.next().map(PathBuf::from),
            "--keyfile" => {
                if let Some(p) = args.next() {
                    keyfile = PathBuf::from(p);
                }
            }
            "--port" => {
                if let Some(p) = args.next() {
                    port = p.parse().map_err(|_| "invalid --port")?;
                }
            }
            "--wonder-queue" => wonder_queue_dir = args.next().map(PathBuf::from),
            "-h" | "--help" => {
                eprintln!("lu_guest_http --brain <path> [--keyfile <path>] [--port <n>] [--wonder-queue <dir>]");
                eprintln!("  --wonder-queue <dir>  append detected curiosity gaps to <dir>/wonder_queue.jsonl");
                eprintln!("                        (append-only; this process never writes the brain)");
                eprintln!("  Binds 127.0.0.1 ONLY. Exposes lu4d_ask + session tools via the guest MCP gate.");
                eprintln!("  Env: LU_GUEST_PASSPHRASE (owner), LU_GUEST_INVITES (comma-separated seeded codes).");
                return Ok(());
            }
            other => return Err(format!("unknown arg {:?}", other).into()),
        }
    }
    let brain_path = brain_path.ok_or("--brain <path> is required")?;

    // ---- load brain READ-ONLY (never saved by this process) ---------
    // OP-007: open the v5 brain DIRECTORY as a non-writing reader. This
    // process holds NO flush worker and NO WAL append handle, so it
    // physically cannot mutate the writer's on-disk brain.
    let key = MasterKey::load_from_keyfile(&keyfile).ok();
    let brain = Brain::load_readonly(&brain_path, key)
        .map_err(|e| format!("failed to load brain {}: {}", brain_path.display(), e))?;
    if !brain.is_read_only_reader() {
        return Err("loaded brain is not a read-only reader (refusing to serve a writable brain)".into());
    }
    let brain_label = brain.brain_format_label().to_string();

    // ---- OP-CURIOSITY-1: PRODUCER wiring ----------------------------
    // Detection already runs here — it fires on essentially every guest
    // query. What was missing is anywhere to PUT the result: the wonder
    // queue was a RAM heap that died with the process, and the half of
    // the engine that fills gaps lives in a different process entirely.
    //
    // This attaches an append-only journal. It does NOT make this
    // process a writer to the brain — `Brain::load_readonly` still
    // refuses `save` and `flush_now`, and systemd still mounts
    // /var/lib/lu read-only for this unit. The journal directory is the
    // single narrow write capability, and the consumer treats every
    // line in it as untrusted.
    //
    // A configured-but-unusable journal directory is a STARTUP FAILURE,
    // not a warning: a guest that silently cannot record what it wonders
    // about looks identical to a guest that has stopped wondering.
    let mut brain = brain;
    brain.narration_config.expose_curiosity_voice = false; // BUG1: never leak internal curiosity voice to users
    if let Some(dir) = wonder_queue_dir.as_ref() {
        let sidecar = lu4d::wonder_sidecar::WonderSidecar::in_dir(dir).map_err(|e| {
            format!("--wonder-queue {}: cannot create/open journal dir: {}", dir.display(), e)
        })?;
        eprintln!(
            "[lu_guest_http] curiosity PRODUCER active -> {}",
            sidecar.path().display()
        );
        brain.attach_wonder_sidecar(sidecar);
    } else {
        eprintln!(
            "[lu_guest_http] no --wonder-queue: gaps are detected but NOT persisted \
             (the fill worker will have nothing to drain)"
        );
    }
    let brain = brain;

    let auth = AuthState::from_env();
    eprintln!(
        "[lu_guest_http] brain={} format={} | owner_passphrase={} seeded_invites={}",
        brain_path.display(),
        brain_label,
        auth.owner_configured(),
        auth.seeded_invite_count()
    );

    let state = Arc::new(AppState {
        brain: RwLock::new(Arc::new(Mutex::new(brain))),
        auth,
        brain_label,
    });

    // ---- OP-007 live-serving refresh task -----------------------------
    // Poll the committed MANIFEST frontier; when the writer commits new
    // SSTs (epoch advances), re-open a fresh read-only reader and swap it
    // in atomically. This is the "moving live view, never frozen"
    // mechanism: the served view tracks the writer's committed frontier,
    // trailing the in-memory MemTable tail by at most one flush interval.
    {
        let st = state.clone();
        let brain_dir = brain_path.clone();
        let keyfile2 = keyfile.clone();
        let wq2: Option<lu4d::wonder_sidecar::WonderSidecar> = wonder_queue_dir
            .as_ref()
            .and_then(|d| lu4d::wonder_sidecar::WonderSidecar::in_dir(d).ok());
        std::thread::spawn(move || {
            let manifest_path = brain_dir.join("MANIFEST");
            let mut last_epoch = current_frontier_epoch(&manifest_path);
            loop {
                std::thread::sleep(std::time::Duration::from_secs(REFRESH_POLL_SECS));
                let ep = current_frontier_epoch(&manifest_path);
                if ep == last_epoch {
                    continue;
                }
                let key = MasterKey::load_from_keyfile(&keyfile2).ok();
                match Brain::load_readonly(&brain_dir, key) {
                    Ok(fresh) => {
                        // OP-CURIOSITY-1: a swapped-in reader is a NEW
                        // Brain, so it needs the journal re-attached or
                        // the producer goes silent after the first
                        // frontier advance.
                        let mut fresh = fresh;
                        fresh.narration_config.expose_curiosity_voice = false; // BUG1
                        if let Some(sc) = wq2.as_ref() {
                            fresh.attach_wonder_sidecar(sc.clone());
                        }
                        let fresh = fresh;
                        // Heavy reopen done OUTSIDE the lock; only the
                        // pointer swap is under the brief write lock.
                        *st.brain.write().unwrap() = Arc::new(Mutex::new(fresh));
                        last_epoch = ep;
                        eprintln!(
                            "[lu_guest_http] refreshed reader to committed frontier (epoch {})",
                            ep
                        );
                    }
                    Err(e) => eprintln!(
                        "[lu_guest_http] frontier refresh failed (keeping current snapshot): {}",
                        e
                    ),
                }
            }
        });
    }

    let app = Router::new()
        .route("/health", get(health))
        .route("/auth", post(do_auth))
        .route("/mcp", post(do_mcp))
        .with_state(state);

    // LOCALHOST ONLY — hard invariant for this scaffold.
    let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), port);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    eprintln!(
        "[lu_guest_http] listening on http://{}  (LOCALHOST ONLY — no public exposure)",
        addr
    );
    axum::serve(listener, app).await?;
    Ok(())
}

async fn health() -> &'static str {
    "ok"
}

/// POST /auth  {"passphrase": "..."}  or  {"invite_code": "..."}
/// -> {"token": "...", "principal": "owner"|"guest:<code>"}
async fn do_auth(
    State(st): State<Arc<AppState>>,
    Json(body): Json<Value>,
) -> Result<Json<Value>, (StatusCode, String)> {
    let principal: Option<Principal> = if let Some(p) = body.get("passphrase").and_then(|v| v.as_str())
    {
        if st.auth.verify_passphrase(p) {
            Some(Principal::Owner)
        } else {
            None
        }
    } else if let Some(code) = body.get("invite_code").and_then(|v| v.as_str()) {
        st.auth.redeem_invite(code)
    } else {
        return Err((
            StatusCode::BAD_REQUEST,
            "provide 'passphrase' or 'invite_code'".to_string(),
        ));
    };

    match principal {
        Some(pr) => {
            let token = st.auth.issue_token(pr.clone());
            Ok(Json(json!({"token": token, "principal": pr.label()})))
        }
        None => Err((StatusCode::UNAUTHORIZED, "invalid credentials".to_string())),
    }
}

/// POST /mcp  (Authorization: Bearer <token>)
/// Body: {"method": "tools/list"}  or
///       {"method": "tools/call", "params": {"name": "lu4d_ask", "arguments": {..}}}
async fn do_mcp(
    State(st): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Result<Json<Value>, (StatusCode, String)> {
    // Identity injection: every /mcp call must carry a valid bearer token.
    let principal = st
        .auth
        .principal_for_bearer(&headers)
        .ok_or((StatusCode::UNAUTHORIZED, "missing or invalid bearer token".to_string()))?;

    let method = body.get("method").and_then(|v| v.as_str()).unwrap_or("");
    match method {
        "tools/list" => Ok(Json(json!({"tools": guest_mcp::tools_list()}))),
        "tools/call" => {
            let params = body.get("params").cloned().unwrap_or(Value::Null);
            let name = params
                .get("name")
                .and_then(|v| v.as_str())
                .ok_or((StatusCode::BAD_REQUEST, "params.name is required".to_string()))?;
            let arguments = params.get("arguments").cloned().unwrap_or(json!({}));

            // Route through the guest MCP trust gate under a brief lock.
            // The guard is dropped before we return; it is never held
            // across an await.
            let result = {
                // Clone the current snapshot Arc under a brief read lock
                // (never held across the dispatch), then serve under the
                // per-snapshot Mutex. A concurrent frontier swap replaces
                // the Arc; this in-flight request keeps its clone.
                let snapshot = st.brain.read().unwrap().clone();
                let mut brain = snapshot.lock().unwrap();
                guest_mcp::dispatch(&mut brain, &principal, name, &arguments)
            };
            match result {
                Ok(v) => Ok(Json(json!({"result": v}))),
                Err((code, message)) => Ok(Json(json!({"error": {"code": code, "message": message}}))),
            }
        }
        other => Err((
            StatusCode::BAD_REQUEST,
            format!("unsupported method '{}'", other),
        )),
    }
}
