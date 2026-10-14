//! Port Cat.4 — MCP tool surface for the signed safety pack format.
//!
//! Six tools:
//!   * Consumer: `lu4d_pack_install`, `lu4d_pack_uninstall`, `lu4d_pack_list`.
//!   * Admin:    `lu4d_pack_publisher_add`, `lu4d_pack_publisher_remove`,
//!               `lu4d_pack_create`.
//!
//! The install path is the only one that mutates `bond_count` —
//! and even then ONLY indirectly: pack bonds become `PendingChange`
//! entries that the user must approve through the existing
//! `lu4d_approve_pending` flow. The anti-hallucination invariant
//! lives in `crate::pack::ingest_pack`.

use serde_json::{json, Value};

use crate::brain::Brain;
use crate::pack::{
    self, build_pack, generate_keypair, hex_decode, ingest_pack, read_pack_file,
    sha256_hex, signing_key_from_bytes, write_pack_file, InstalledPack,
    PackBond, PackBuildSpec, PackManifestSeed, PACK_HEADER_LEN,
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

fn err_to_jsonrpc(e: pack::PackError) -> (i64, String) {
    (-32603, format!("pack error: {}", e))
}

// ----------------------------------------------------------------
// lu4d_pack_install
// ----------------------------------------------------------------
pub(crate) fn tool_pack_install(
    brain: &mut Brain,
    args: Value,
) -> Result<Value, (i64, String)> {
    let path = args
        .get("pack_path")
        .and_then(|v| v.as_str())
        .ok_or((-32602, "missing required arg: pack_path".to_string()))?
        .to_string();
    let now_ms = now_ms_from(&args);
    let (pack, raw_bytes) = read_pack_file(&path).map_err(err_to_jsonrpc)?;
    let res = ingest_pack(brain, &pack, &raw_bytes, now_ms).map_err(err_to_jsonrpc)?;
    // Record the install in the ledger immediately. The bonds are
    // staged in PendingLog; the install_count in the ledger reflects
    // the queued bond count, not the applied count. (A future
    // refinement could update this on apply.)
    brain.pack_ledger.record_install(InstalledPack {
        pack_id: res.pack_id.clone(),
        publisher_id: res.publisher_id.clone(),
        pack_version: res.pack_version.clone(),
        bond_count_added: res.queued_bond_count,
        node_count_added: res.queued_node_count,
        installed_at_ms: now_ms,
        manifest_hash_hex: res.manifest_hash_hex.clone(),
    });
    let text = format!(
        "pack_install ok: pack_id={} publisher_id={} pack_version={} queued={} pending_ids={:?}",
        res.pack_id,
        &res.publisher_id[..16],
        res.pack_version,
        res.queued_bond_count,
        res.queued_pending_ids,
    );
    Ok(wrap_content(text, json!({
        "pack_id": res.pack_id,
        "publisher_id": res.publisher_id,
        "pack_version": res.pack_version,
        "queued_bond_count": res.queued_bond_count,
        "queued_node_count": res.queued_node_count,
        "queued_pending_ids": res.queued_pending_ids,
        "manifest_hash_hex": res.manifest_hash_hex,
        "note": "Pack bonds queued in PendingLog. Run lu4d_approve_pending to merge into the brain.",
    })))
}

// ----------------------------------------------------------------
// lu4d_pack_uninstall
// ----------------------------------------------------------------
pub(crate) fn tool_pack_uninstall(
    brain: &mut Brain,
    args: Value,
) -> Result<Value, (i64, String)> {
    let pack_id = args
        .get("pack_id")
        .and_then(|v| v.as_str())
        .ok_or((-32602, "missing required arg: pack_id".to_string()))?
        .to_string();
    let reason = args
        .get("reason")
        .and_then(|v| v.as_str())
        .unwrap_or("uninstalled by user")
        .to_string();
    let dry_run = args.get("dry_run").and_then(|v| v.as_bool()).unwrap_or(false);
    let now_ms = now_ms_from(&args);
    // Port SourceKind-Variants — match BOTH the new first-class
    // SourceKind::PackInstall { pack_id, .. } AND the legacy
    // SourceKind::Teach { actor: "pack_install:<pack_id>" } shape.
    // The legacy shape can still appear on canonical brains saved by
    // pre-port binaries; the migrate tool (off by default) is the
    // operator-driven upgrade path — until it runs, both shapes coexist.
    let legacy_actor = format!("pack_install:{}", pack_id);
    let want_pack_id = pack_id.clone();
    let want_legacy_actor = legacy_actor.clone();
    let pack_match = move |attr: &crate::source::SourceAttribution| -> bool {
        match &attr.kind {
            crate::source::SourceKind::PackInstall { pack_id: p, .. } => p == &want_pack_id,
            crate::source::SourceKind::Teach { actor: a, .. } => a == &want_legacy_actor,
            _ => false,
        }
    };
    // Walk the source overlay first to count what we'll touch.
    let mut matched: usize = 0;
    {
        let source_overlay = brain.source_overlay.read().unwrap();
        for (_k, attr) in &source_overlay.entries {
            if pack_match(attr) { matched += 1; }
        }
    }
    let revoked = if dry_run {
        (0usize, 0usize)
    } else {
        // Re-clone for the move into revoke_bonds_by_source.
        let want_pack_id2 = pack_id.clone();
        let want_legacy_actor2 = legacy_actor.clone();
        brain.revoke_bonds_by_source(move |attr| {
            match &attr.kind {
                crate::source::SourceKind::PackInstall { pack_id: p, .. } => p == &want_pack_id2,
                crate::source::SourceKind::Teach { actor: a, .. } => a == &want_legacy_actor2,
                _ => false,
            }
        })
    };
    if !dry_run {
        brain.pack_ledger.record_revoke(&pack_id, &reason, now_ms);
    }
    let text = format!(
        "pack_uninstall pack_id={} matched={} dry_run={} overlay_revoked={} bonds_revoked={}",
        pack_id, matched, dry_run, revoked.0, revoked.1
    );
    Ok(wrap_content(text, json!({
        "pack_id": pack_id,
        "reason": reason,
        "dry_run": dry_run,
        "matched": matched,
        "overlay_revoked": revoked.0,
        "bonds_revoked": revoked.1,
    })))
}

// ----------------------------------------------------------------
// lu4d_pack_list
// ----------------------------------------------------------------
pub(crate) fn tool_pack_list(
    brain: &mut Brain,
    _args: Value,
) -> Result<Value, (i64, String)> {
    let pubs: Vec<Value> = brain.pack_ledger.trusted_publishers
        .values()
        .map(|p| json!({
            "pubkey_hex": p.pubkey_hex,
            "name": p.name,
            "added_at_ms": p.added_at_ms,
            "notes": p.notes,
        }))
        .collect();
    let installs: Vec<Value> = brain.pack_ledger.installed_packs
        .values()
        .map(|i| json!({
            "pack_id": i.pack_id,
            "publisher_id": i.publisher_id,
            "pack_version": i.pack_version,
            "bond_count_added": i.bond_count_added,
            "node_count_added": i.node_count_added,
            "installed_at_ms": i.installed_at_ms,
            "manifest_hash_hex": i.manifest_hash_hex,
        }))
        .collect();
    let revokes: Vec<Value> = brain.pack_ledger.revoked_packs
        .values()
        .map(|r| json!({
            "pack_id": r.pack_id,
            "revoked_at_ms": r.revoked_at_ms,
            "reason": r.reason,
        }))
        .collect();
    let text = format!(
        "pack_list: trusted_publishers={} installed_packs={} revoked_packs={}",
        pubs.len(), installs.len(), revokes.len()
    );
    Ok(wrap_content(text, json!({
        "trusted_publishers": pubs,
        "installed_packs": installs,
        "revoked_packs": revokes,
    })))
}

// ----------------------------------------------------------------
// lu4d_pack_publisher_add
// ----------------------------------------------------------------
pub(crate) fn tool_pack_publisher_add(
    brain: &mut Brain,
    args: Value,
) -> Result<Value, (i64, String)> {
    let pubkey_hex = args
        .get("pubkey_hex")
        .and_then(|v| v.as_str())
        .ok_or((-32602, "missing required arg: pubkey_hex".to_string()))?
        .to_string();
    let name = args
        .get("name")
        .and_then(|v| v.as_str())
        .ok_or((-32602, "missing required arg: name".to_string()))?
        .to_string();
    let notes = args.get("notes").and_then(|v| v.as_str()).map(String::from);
    let now_ms = now_ms_from(&args);
    let added = brain.pack_ledger.add_publisher(&pubkey_hex, &name, notes.clone(), now_ms)
        .map_err(err_to_jsonrpc)?;
    let text = format!(
        "pack_publisher_add ok: name=\"{}\" pubkey={}…",
        added.name,
        &added.pubkey_hex[..16],
    );
    Ok(wrap_content(text, json!({
        "pubkey_hex": added.pubkey_hex,
        "name": added.name,
        "added_at_ms": added.added_at_ms,
        "notes": added.notes,
        "trusted_publishers_count": brain.pack_ledger.trusted_publishers.len(),
    })))
}

// ----------------------------------------------------------------
// lu4d_pack_publisher_remove
// ----------------------------------------------------------------
pub(crate) fn tool_pack_publisher_remove(
    brain: &mut Brain,
    args: Value,
) -> Result<Value, (i64, String)> {
    let pubkey_hex = args
        .get("pubkey_hex")
        .and_then(|v| v.as_str())
        .ok_or((-32602, "missing required arg: pubkey_hex".to_string()))?
        .to_string();
    let removed = brain.pack_ledger.remove_publisher(&pubkey_hex);
    let text = match &removed {
        Some(p) => format!("pack_publisher_remove ok: name=\"{}\" pubkey={}…", p.name, &p.pubkey_hex[..16]),
        None => format!("pack_publisher_remove: no publisher with pubkey={}…", &pubkey_hex[..pubkey_hex.len().min(16)]),
    };
    Ok(wrap_content(text, json!({
        "pubkey_hex": pubkey_hex,
        "removed": removed.is_some(),
        "trusted_publishers_count": brain.pack_ledger.trusted_publishers.len(),
    })))
}

// ----------------------------------------------------------------
// lu4d_pack_create  (admin / producer)
// ----------------------------------------------------------------
pub(crate) fn tool_pack_create(
    brain: &mut Brain,
    args: Value,
) -> Result<Value, (i64, String)> {
    let pack_id = args
        .get("pack_id")
        .and_then(|v| v.as_str())
        .ok_or((-32602, "missing required arg: pack_id".to_string()))?
        .to_string();
    let output_path = args
        .get("output_path")
        .and_then(|v| v.as_str())
        .ok_or((-32602, "missing required arg: output_path".to_string()))?
        .to_string();
    let keyfile = args
        .get("keyfile")
        .and_then(|v| v.as_str())
        .ok_or((-32602, "missing required arg: keyfile (32-byte ed25519 secret key path)".to_string()))?
        .to_string();
    let pack_version = args
        .get("pack_version")
        .and_then(|v| v.as_str())
        .unwrap_or("0.0.1")
        .to_string();
    let summary = args
        .get("content_summary")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let license = args
        .get("license")
        .and_then(|v| v.as_str())
        .unwrap_or("proprietary")
        .to_string();
    let minimum_lu_version = args
        .get("minimum_lu_version")
        .and_then(|v| v.as_str())
        .unwrap_or("0.1.0")
        .to_string();
    let bond_limit = args.get("bond_limit").and_then(|v| v.as_u64()).unwrap_or(1000) as usize;
    let bond_type_filter = args.get("bond_type").and_then(|v| v.as_str()).map(String::from);
    let now_ms = now_ms_from(&args);

    // Load the secret key.
    let raw = std::fs::read(&keyfile).map_err(|e| (-32603,
        format!("keyfile read {}: {}", keyfile, e)))?;
    if raw.len() != 32 {
        return Err((-32603, format!(
            "keyfile {} wrong size: got {} expected 32", keyfile, raw.len()
        )));
    }
    let mut sk_bytes = [0u8; 32];
    sk_bytes.copy_from_slice(&raw);
    let sk = signing_key_from_bytes(&sk_bytes);
    let mk = brain.encryption_key.clone().ok_or((-32603,
        "brain has no master_key — pack_create requires --keyfile".to_string()))?;

    // Walk the brain's bonds and select up to bond_limit matching
    // the optional type filter. This is the simple v1 selector;
    // Port V will add richer query DSL.
    let mut selected: Vec<PackBond> = Vec::new();
    let bonds_view = brain.network.bonds();
    for (key, bond) in bonds_view.iter() {
        if selected.len() >= bond_limit { break; }
        let bt = bond.bond_type();
        if let Some(filter) = &bond_type_filter {
            if format!("{:?}", bt).to_ascii_lowercase() != filter.to_ascii_lowercase() {
                continue;
            }
        }
        // Resolve back to text via the node store; skip if either end
        // can't be resolved.
        let from_text = brain.network.get_node(&key.0).map(|n| n.content().to_string());
        let to_text = brain.network.get_node(&key.1).map(|n| n.content().to_string());
        if let (Some(s), Some(o)) = (from_text, to_text) {
            selected.push(PackBond {
                subject_text: s,
                object_text: o,
                bond_type: bt,
                weight: 0.7,
                negated: false,
                confidence: 0.7,
                reason: format!("pack:{}", pack_id),
            });
        }
    }

    let seed = PackManifestSeed {
        pack_id: pack_id.clone(),
        pack_version: pack_version.clone(),
        content_summary: if summary.is_empty() {
            format!("{} bonds from brain", selected.len())
        } else {
            summary
        },
        minimum_lu_version,
        depends_on_pack_ids: Vec::new(),
        license,
    };
    let pack = build_pack(PackBuildSpec {
        manifest_seed: seed,
        bonds: selected,
        signing_key: &sk,
        master_key: &mk,
        now_ms,
    }).map_err(err_to_jsonrpc)?;
    write_pack_file(&pack, &output_path).map_err(err_to_jsonrpc)?;
    let pubkey_hex = pack::hex_encode(sk.verifying_key().as_bytes());
    let text = format!(
        "pack_create ok: pack_id={} pack_version={} bonds={} output={} publisher={}…",
        pack.manifest.pack_id, pack.manifest.pack_version,
        pack.manifest.bond_count, output_path,
        &pubkey_hex[..16]
    );
    Ok(wrap_content(text, json!({
        "pack_id": pack.manifest.pack_id,
        "pack_version": pack.manifest.pack_version,
        "publisher_id": pubkey_hex,
        "bond_count": pack.manifest.bond_count,
        "node_count": pack.manifest.node_count,
        "output_path": output_path,
        "signed_at_ms": pack.manifest.signed_at_ms,
    })))
}

// ----------------------------------------------------------------
// CLI helper: --gen-pack-keypair <path>
// ----------------------------------------------------------------
/// Used by main.rs `--gen-pack-keypair`. Writes raw 32 bytes (mode
/// 0600) and prints the public-key hex on stdout. Returns Err if the
/// file already exists (refuses to overwrite).
pub fn gen_pack_keypair_cli(path: &str) -> Result<String, String> {
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

// silence unused-import warning in --release if hex_decode is
// only used by tests in the same file.
#[allow(dead_code)]
fn _hex_decode_keep_import(s: &str) -> Result<[u8; 32], pack::PackError> {
    hex_decode::<32>(s)
}

#[allow(dead_code)]
fn _sha256_hex_keep_import(b: &[u8]) -> String { sha256_hex(b) }

#[allow(dead_code)]
fn _pack_header_len_keep() -> usize { PACK_HEADER_LEN }
