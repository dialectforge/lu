//! Port Cat.5 — MCP tool surface for the schema-only signed
//! capability pack format.
//!
//! Six tools (mirror Cat.4's surface):
//!   * `lu4d_capability_pack_install`   — verify + record install (NO
//!     execution; runtime loading is Cat.5.1's job).
//!   * `lu4d_capability_pack_uninstall` — remove from ledger,
//!     optionally revoke pack_id to prevent re-install.
//!   * `lu4d_capability_pack_list`      — list trusted publishers,
//!     installed packs, revoked pack_ids.
//!   * `lu4d_capability_pack_publisher_add`    — add a trusted publisher.
//!   * `lu4d_capability_pack_publisher_remove` — remove a trusted
//!     publisher (existing installs continue; no new installs from
//!     that publisher).
//!   * `lu4d_capability_pack_create`           — admin/producer tool.
//!     Sign a `.lupack5` file from a manifest JSON + opaque content
//!     payload via a 32-byte Ed25519 secret keyfile.
//!
//! Anti-hallucination invariant — STRUCTURAL:
//! Tool bodies in this file MUST mutate ONLY `brain.cap_pack_ledger.*`
//! state. They MUST NOT call `brain.teach`, `add_bond`, `pending.push`,
//! or any other bond-touching path. The grep guard in the tests
//! module (`capability_pack::tests::antimutation_grep_guard`) asserts
//! this at test time.

use serde_json::{json, Value};

use crate::brain::Brain;
use crate::capability_pack::{
    self, build_capability_pack, generate_keypair, install_capability_pack,
    read_capability_pack_file, signing_key_from_bytes, write_capability_pack_file,
    CapabilityPackBuildSpec, CapabilityPackManifestSeed, PackPermissions,
    PackRuntimeKind, PackToolDecl,
};

fn wrap_content(text: String, structured: Value) -> Value {
    json!({
        "content": [{ "type": "text", "text": text }],
        "structuredContent": structured,
    })
}

fn now_ms_from(args: &Value) -> u64 {
    if let Some(v) = args.get("now_ms").and_then(|v| v.as_u64()) {
        return v;
    }
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn err_to_jsonrpc(e: capability_pack::CapabilityPackError) -> (i64, String) {
    (-32603, format!("capability pack error: {}", e))
}

fn parse_runtime_kind(s: &str) -> PackRuntimeKind {
    match s.trim().to_ascii_lowercase().as_str() {
        "wasm" => PackRuntimeKind::Wasm,
        "sharedlib" | "shared_lib" | "shared-lib" | "so" => PackRuntimeKind::SharedLib,
        "subprocess" | "subproc" => PackRuntimeKind::Subprocess,
        "reserved" => PackRuntimeKind::Reserved,
        _ => PackRuntimeKind::Wasm,
    }
}

fn runtime_kind_str(k: PackRuntimeKind) -> &'static str {
    match k {
        PackRuntimeKind::Wasm => "Wasm",
        PackRuntimeKind::SharedLib => "SharedLib",
        PackRuntimeKind::Subprocess => "Subprocess",
        PackRuntimeKind::Reserved => "Reserved",
    }
}

// ----------------------------------------------------------------
// lu4d_capability_pack_install
// ----------------------------------------------------------------
pub(crate) fn tool_capability_pack_install(
    brain: &mut Brain,
    args: Value,
) -> Result<Value, (i64, String)> {
    let path = args
        .get("path")
        .and_then(|v| v.as_str())
        .ok_or((-32602, "missing required arg: path".to_string()))?
        .to_string();
    let force = args.get("force").and_then(|v| v.as_bool()).unwrap_or(false);
    let now_ms = now_ms_from(&args);
    let (pack, raw_bytes) = read_capability_pack_file(&path).map_err(err_to_jsonrpc)?;
    let res = install_capability_pack(
        &mut brain.cap_pack_ledger,
        &pack,
        &raw_bytes,
        &path,
        force,
        now_ms,
    )
    .map_err(err_to_jsonrpc)?;
    let text = format!(
        "capability_pack_install ok: pack_id={} pack_version={} runtime={} publisher={}\u{2026} tools={} overwrite={}",
        res.pack_id,
        res.pack_version,
        runtime_kind_str(res.runtime_kind),
        &res.publisher_pubkey_hex[..16],
        res.tool_count,
        res.already_installed_overwrite,
    );
    Ok(wrap_content(text, json!({
        "pack_id": res.pack_id,
        "pack_version": res.pack_version,
        "publisher_pubkey_hex": res.publisher_pubkey_hex,
        "runtime_kind": runtime_kind_str(res.runtime_kind),
        "tool_count": res.tool_count,
        "manifest_hash_hex": res.manifest_hash_hex,
        "already_installed_overwrite": res.already_installed_overwrite,
        "note": "Cat.5 schema-only: pack recorded in ledger but NOT executed. Runtime loading lands in Cat.5.1.",
    })))
}

// ----------------------------------------------------------------
// lu4d_capability_pack_uninstall
// ----------------------------------------------------------------
pub(crate) fn tool_capability_pack_uninstall(
    brain: &mut Brain,
    args: Value,
) -> Result<Value, (i64, String)> {
    let pack_id = args
        .get("pack_id")
        .and_then(|v| v.as_str())
        .ok_or((-32602, "missing required arg: pack_id".to_string()))?
        .to_string();
    let revoke = args.get("revoke").and_then(|v| v.as_bool()).unwrap_or(false);
    let removed = brain.cap_pack_ledger.remove_install(&pack_id);
    if revoke {
        brain.cap_pack_ledger.revoke(&pack_id);
    }
    let text = match &removed {
        Some(p) => format!(
            "capability_pack_uninstall ok: pack_id={} pack_version={} revoke={}",
            p.pack_id, p.pack_version, revoke
        ),
        None => format!(
            "capability_pack_uninstall: no installed pack with id {} (revoke={})",
            pack_id, revoke
        ),
    };
    Ok(wrap_content(text, json!({
        "pack_id": pack_id,
        "removed": removed.is_some(),
        "revoked": revoke,
        "installed_count": brain.cap_pack_ledger.installed_packs.len(),
        "revoked_count": brain.cap_pack_ledger.revoked_packs.len(),
    })))
}

// ----------------------------------------------------------------
// lu4d_capability_pack_list
// ----------------------------------------------------------------
pub(crate) fn tool_capability_pack_list(
    brain: &mut Brain,
    _args: Value,
) -> Result<Value, (i64, String)> {
    let pubs: Vec<Value> = brain.cap_pack_ledger.trusted_publishers
        .values()
        .map(|p| json!({
            "pubkey_hex": p.pubkey_hex,
            "label": p.label,
            "added_at_ms": p.added_at_ms,
            "notes": p.notes,
        }))
        .collect();
    let installs: Vec<Value> = brain.cap_pack_ledger.installed_packs
        .values()
        .map(|i| json!({
            "pack_id": i.pack_id,
            "pack_version": i.pack_version,
            "publisher_pubkey_hex": i.publisher_pubkey_hex,
            "runtime_kind": runtime_kind_str(i.runtime_kind),
            "installed_at_ms": i.installed_at_ms,
            "signed_at_ms": i.signed_at_ms,
            "manifest_hash_hex": i.manifest_hash_hex,
            "tool_count": i.tool_count,
            "source_path": i.source_path,
        }))
        .collect();
    let revokes: Vec<Value> = brain.cap_pack_ledger.revoked_packs
        .iter()
        .map(|p| json!(p))
        .collect();
    let text = format!(
        "capability_pack_list: trusted_publishers={} installed_packs={} revoked_packs={}",
        pubs.len(), installs.len(), revokes.len()
    );
    Ok(wrap_content(text, json!({
        "trusted_publishers": pubs,
        "installed_packs": installs,
        "revoked_packs": revokes,
    })))
}

// ----------------------------------------------------------------
// lu4d_capability_pack_publisher_add
// ----------------------------------------------------------------
pub(crate) fn tool_capability_pack_publisher_add(
    brain: &mut Brain,
    args: Value,
) -> Result<Value, (i64, String)> {
    let pubkey_hex = args
        .get("pubkey_hex")
        .and_then(|v| v.as_str())
        .ok_or((-32602, "missing required arg: pubkey_hex".to_string()))?
        .to_string();
    let label = args
        .get("label")
        .and_then(|v| v.as_str())
        .ok_or((-32602, "missing required arg: label".to_string()))?
        .to_string();
    let notes = args.get("notes").and_then(|v| v.as_str()).map(String::from);
    let now_ms = now_ms_from(&args);
    let added = brain.cap_pack_ledger.add_publisher(&pubkey_hex, &label, notes.clone(), now_ms)
        .map_err(err_to_jsonrpc)?;
    let text = format!(
        "capability_pack_publisher_add ok: label=\"{}\" pubkey={}\u{2026}",
        added.label,
        &added.pubkey_hex[..16],
    );
    Ok(wrap_content(text, json!({
        "pubkey_hex": added.pubkey_hex,
        "label": added.label,
        "added_at_ms": added.added_at_ms,
        "notes": added.notes,
        "trusted_publishers_count": brain.cap_pack_ledger.trusted_publishers.len(),
    })))
}

// ----------------------------------------------------------------
// lu4d_capability_pack_publisher_remove
// ----------------------------------------------------------------
pub(crate) fn tool_capability_pack_publisher_remove(
    brain: &mut Brain,
    args: Value,
) -> Result<Value, (i64, String)> {
    let pubkey_hex = args
        .get("pubkey_hex")
        .and_then(|v| v.as_str())
        .ok_or((-32602, "missing required arg: pubkey_hex".to_string()))?
        .to_string();
    let removed = brain.cap_pack_ledger.remove_publisher(&pubkey_hex);
    let text = match &removed {
        Some(p) => format!(
            "capability_pack_publisher_remove ok: label=\"{}\" pubkey={}\u{2026}",
            p.label, &p.pubkey_hex[..16],
        ),
        None => format!(
            "capability_pack_publisher_remove: no publisher with pubkey={}\u{2026}",
            &pubkey_hex[..pubkey_hex.len().min(16)]
        ),
    };
    Ok(wrap_content(text, json!({
        "pubkey_hex": pubkey_hex,
        "removed": removed.is_some(),
        "trusted_publishers_count": brain.cap_pack_ledger.trusted_publishers.len(),
        "note": "Existing installs from this publisher remain; future installs are refused.",
    })))
}

// ----------------------------------------------------------------
// lu4d_capability_pack_create  (admin / producer)
// ----------------------------------------------------------------
//
// Reads:
//   * `manifest_path`  — JSON file describing the manifest seed:
//                         { pack_id, pack_version, runtime_kind,
//                           tools: [...], permissions: {...},
//                           description }
//   * `content_path`   — opaque bytes (can be a placeholder for Cat.5
//                         schema-only; e.g. the literal string
//                         "RUNTIME_TBD").
//   * `signing_keyfile`— 32-byte Ed25519 secret key.
//   * `output_path`    — `.lupack5` destination.
pub(crate) fn tool_capability_pack_create(
    _brain: &mut Brain,
    args: Value,
) -> Result<Value, (i64, String)> {
    let manifest_path = args
        .get("manifest_path")
        .and_then(|v| v.as_str())
        .ok_or((-32602, "missing required arg: manifest_path".to_string()))?
        .to_string();
    let content_path = args
        .get("content_path")
        .and_then(|v| v.as_str())
        .ok_or((-32602, "missing required arg: content_path".to_string()))?
        .to_string();
    let output_path = args
        .get("output_path")
        .and_then(|v| v.as_str())
        .ok_or((-32602, "missing required arg: output_path".to_string()))?
        .to_string();
    let signing_keyfile = args
        .get("signing_keyfile")
        .and_then(|v| v.as_str())
        .ok_or((-32602, "missing required arg: signing_keyfile (32-byte ed25519 secret key path)".to_string()))?
        .to_string();
    let now_ms = now_ms_from(&args);

    // Load manifest JSON.
    let manifest_json_bytes = std::fs::read(&manifest_path)
        .map_err(|e| (-32603, format!("manifest_path read {}: {}", manifest_path, e)))?;
    let manifest_json: Value = serde_json::from_slice(&manifest_json_bytes)
        .map_err(|e| (-32603, format!("manifest_path parse {}: {}", manifest_path, e)))?;
    let pack_id = manifest_json
        .get("pack_id")
        .and_then(|v| v.as_str())
        .ok_or((-32603, "manifest missing pack_id".to_string()))?
        .to_string();
    let pack_version = manifest_json
        .get("pack_version")
        .and_then(|v| v.as_str())
        .unwrap_or("0.0.1")
        .to_string();
    let runtime_kind = parse_runtime_kind(
        manifest_json
            .get("runtime_kind")
            .and_then(|v| v.as_str())
            .unwrap_or("Wasm"),
    );
    let description = manifest_json
        .get("description")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    // Tools array — pass through verbatim (each entry is
    // {tool_name, description, args_schema}).
    let tools: Vec<PackToolDecl> = if let Some(arr) = manifest_json.get("tools").and_then(|v| v.as_array()) {
        arr.iter().map(|t| PackToolDecl {
            tool_name: t.get("tool_name").and_then(|v| v.as_str()).unwrap_or("").to_string(),
            description: t.get("description").and_then(|v| v.as_str()).unwrap_or("").to_string(),
            args_schema: t.get("args_schema").cloned().unwrap_or(json!({"type": "object"})),
        }).collect()
    } else { Vec::new() };
    // Permissions block.
    let permissions: PackPermissions = if let Some(p) = manifest_json.get("permissions") {
        PackPermissions {
            filesystem_read: p.get("filesystem_read").and_then(|v| v.as_array())
                .map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect())
                .unwrap_or_default(),
            filesystem_write: p.get("filesystem_write").and_then(|v| v.as_array())
                .map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect())
                .unwrap_or_default(),
            network_egress: p.get("network_egress").and_then(|v| v.as_array())
                .map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect())
                .unwrap_or_default(),
            brain_read: p.get("brain_read").and_then(|v| v.as_bool()).unwrap_or(false),
            brain_write: p.get("brain_write").and_then(|v| v.as_bool()).unwrap_or(false),
        }
    } else { PackPermissions::default() };

    // Load opaque content bytes.
    let content = std::fs::read(&content_path)
        .map_err(|e| (-32603, format!("content_path read {}: {}", content_path, e)))?;

    // Load signing key.
    let raw = std::fs::read(&signing_keyfile)
        .map_err(|e| (-32603, format!("signing_keyfile read {}: {}", signing_keyfile, e)))?;
    if raw.len() != 32 {
        return Err((-32603, format!(
            "signing_keyfile {} wrong size: got {} expected 32", signing_keyfile, raw.len()
        )));
    }
    let mut sk_bytes = [0u8; 32];
    sk_bytes.copy_from_slice(&raw);
    let sk = signing_key_from_bytes(&sk_bytes);

    let seed = CapabilityPackManifestSeed {
        pack_id: pack_id.clone(),
        pack_version: pack_version.clone(),
        runtime_kind,
        tools,
        permissions,
        description,
    };
    let pack = build_capability_pack(CapabilityPackBuildSpec {
        manifest_seed: seed,
        content,
        signing_key: &sk,
        now_ms,
    }).map_err(err_to_jsonrpc)?;
    write_capability_pack_file(&pack, &output_path).map_err(err_to_jsonrpc)?;
    let pubkey_hex = capability_pack::hex_encode(sk.verifying_key().as_bytes());
    let text = format!(
        "capability_pack_create ok: pack_id={} pack_version={} runtime={} content_len={} output={} publisher={}\u{2026}",
        pack.manifest.pack_id, pack.manifest.pack_version,
        runtime_kind_str(pack.manifest.runtime_kind),
        pack.content.len(), output_path,
        &pubkey_hex[..16]
    );
    Ok(wrap_content(text, json!({
        "pack_id": pack.manifest.pack_id,
        "pack_version": pack.manifest.pack_version,
        "publisher_pubkey_hex": pubkey_hex,
        "runtime_kind": runtime_kind_str(pack.manifest.runtime_kind),
        "content_len": pack.content.len(),
        "output_path": output_path,
        "signed_at_ms": pack.manifest.signed_at_ms,
        "tool_count": pack.manifest.tools.len(),
    })))
}

// ----------------------------------------------------------------
// CLI helper: --gen-capability-pack-keypair <path>
// ----------------------------------------------------------------
/// Used by main.rs `--gen-capability-pack-keypair`. Mirrors
/// `mcp_pack::gen_pack_keypair_cli`. Writes raw 32 bytes (mode 0600)
/// and prints the public-key hex on stdout. Returns Err if the file
/// already exists (refuses to overwrite).
pub fn gen_capability_pack_keypair_cli(path: &str) -> Result<String, String> {
    let p = std::path::Path::new(path);
    if p.exists() {
        return Err(format!("keyfile {} already exists", path));
    }
    let (sk_bytes, pk_hex) = generate_keypair();
    std::fs::write(p, &sk_bytes).map_err(|e| format!("write {}: {}", path, e))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(p, std::fs::Permissions::from_mode(0o600));
    }
    Ok(pk_hex)
}
