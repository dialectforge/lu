//! `lu_admin_mcp` — the ADMIN half of the OP-007 / Port U two-binary
//! privilege split.
//!
//! This binary exposes the FULL Lu MCP tool surface (teach, revoke,
//! configure, trust, packs, approvals, …) over JSON-RPC on stdio. It is
//! intended to run bound to localhost / a trusted admin channel ONLY
//! (never public :443). Its public-facing counterpart is `lu_guest_http`,
//! which is compiled WITHOUT the admin `mcp` module (see the `admin_mcp`
//! feature gate) so the admin handlers are physically absent from the
//! guest binary rather than merely permission-checked at runtime.
//!
//! Scaffold scope: minimal arg handling sufficient to load an encrypted
//! v3/v5 brain and serve the admin MCP loop. It deliberately mirrors the
//! load path of the main `lu4d --mcp` entry point; the canonical daemon
//! continues to run `lu4d --mcp` and is NOT replaced by this binary.

#[cfg(feature = "admin_mcp")]
fn main() -> std::process::ExitCode {
    use lu4d::encrypt::{MasterKey, DEFAULT_KEYFILE_PATH};
    use lu4d::{Brain, BrainLoadMode};
    use std::path::PathBuf;
    use std::process::ExitCode;

    let mut brain_path: Option<PathBuf> = None;
    let mut keyfile: PathBuf = PathBuf::from(DEFAULT_KEYFILE_PATH);
    let mut save_on_write = false; // admin scaffold defaults to read-only
    let mut require_encrypted = false;

    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--brain" => brain_path = args.next().map(PathBuf::from),
            "--keyfile" => {
                if let Some(p) = args.next() {
                    keyfile = PathBuf::from(p);
                }
            }
            "--allow-save" => save_on_write = true,
            "--require-encrypted" => require_encrypted = true,
            "--mcp" => {} // implied; accepted for CLI parity
            "-h" | "--help" => {
                eprintln!("lu_admin_mcp --brain <path> [--keyfile <path>] [--allow-save] [--require-encrypted]");
                return ExitCode::SUCCESS;
            }
            other => {
                eprintln!("lu_admin_mcp: unknown arg {:?}", other);
                return ExitCode::from(2);
            }
        }
    }

    let brain_path = match brain_path {
        Some(p) => p,
        None => {
            eprintln!("lu_admin_mcp: --brain <path> is required");
            return ExitCode::from(2);
        }
    };

    // Load the master key if present; a plaintext brain loads with None.
    let key = MasterKey::load_from_keyfile(&keyfile).ok();
    let mut brain = match Brain::load_or_new_with_mode_and_key(&brain_path, BrainLoadMode::Auto, key) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("lu_admin_mcp: failed to load brain {}: {}", brain_path.display(), e);
            return ExitCode::from(1);
        }
    };
    brain.require_encrypted = require_encrypted;

    eprintln!(
        "[lu_admin_mcp] admin MCP surface (ALL tools) over stdio | brain={} format={} save_on_write={}",
        brain_path.display(),
        brain.brain_format_label(),
        save_on_write
    );

    // The admin trust gate: the full tool_definitions()/handle_tools_call
    // dispatch lives in `lu4d::mcp`, present here because this binary is
    // built with the default `admin_mcp` feature.
    if let Err(e) = lu4d::mcp::run_mcp_server_opts(&mut brain, save_on_write) {
        eprintln!("lu_admin_mcp: mcp server error: {}", e);
        return ExitCode::from(1);
    }
    ExitCode::SUCCESS
}

#[cfg(not(feature = "admin_mcp"))]
fn main() {
    eprintln!("lu_admin_mcp was built without the `admin_mcp` feature; nothing to serve.");
}
