//! Port Cat.5.1 — MCP surface for the capability-pack EXECUTION
//! runtime.
//!
//! Admin/control tools (mirror the Cat.5 ledger tools' shape):
//!   * `lu4d_capability_pack_runtime_configure` — master switch +
//!     unsandboxed opt-in + resource caps.
//!   * `lu4d_capability_pack_runtime_status`    — enabled state, caps,
//!     loaded packs, registered tools.
//!   * `lu4d_capability_pack_load`              — load an installed
//!     (or explicitly-pathed) signed pack into the runtime registry.
//!     Re-verifies trust + revocation + signature before registering.
//!   * `lu4d_capability_pack_unload`            — drop a loaded pack
//!     (its tools + extracted executable).
//!
//! Dynamic dispatch: `dispatch_pack_tool` is the fallback in
//! `mcp.rs::handle_tools_call_inner` for any `pack__*` tool. It
//! re-verifies on execute, runs the subprocess under resource caps,
//! and — this is the ONLY place Cat.5.1 touches the graph — routes a
//! granted pack's proposed brain_writes through the runtime PendingLog
//! as teach-class proposals. They land as `Pending`, never as a direct
//! bond and never as a retrain, so `bond_count` is unchanged.

use serde_json::{json, Value};

use crate::brain::Brain;
use crate::capability_pack::read_capability_pack_file;
use crate::capability_runtime::PackRuntimeError;
use crate::pending::{PendingChange, PendingChangeKind, PendingStatus};
use crate::source::SourceAttribution;

fn wrap_content(text: String, structured: Value) -> Value {
    json!({
        "content": [{ "type": "text", "text": text }],
        "structuredContent": structured,
    })
}

fn now_ms(brain: &Brain) -> u64 {
    match brain.clock.fetch() {
        Ok(a) => a.utc_ms,
        Err(_) => crate::brain::now_ms() as u64,
    }
}

pub fn map_runtime_err(e: PackRuntimeError) -> (i64, String) {
    let code = match e {
        PackRuntimeError::ToolNotFound(_) => -32601,
        PackRuntimeError::NotLoaded(_) => -32602,
        _ => -32603,
    };
    (code, format!("capability runtime: {}", e))
}

// ----------------------------------------------------------------
// lu4d_capability_pack_runtime_configure
// ----------------------------------------------------------------
pub fn tool_runtime_configure(brain: &mut Brain, args: Value) -> Result<Value, (i64, String)> {
    let reg = &mut brain.pack_registry;
    if let Some(b) = args.get("enabled").and_then(|v| v.as_bool()) {
        reg.set_enabled(b);
    }
    if let Some(b) = args.get("allow_unsandboxed").and_then(|v| v.as_bool()) {
        reg.set_allow_unsandboxed(b);
    }
    let mut caps = reg.caps;
    if let Some(v) = args.get("cpu_ms").and_then(|v| v.as_u64()) {
        caps.cpu_ms = v;
    }
    if let Some(v) = args.get("mem_bytes").and_then(|v| v.as_u64()) {
        caps.mem_bytes = v;
    }
    if let Some(v) = args.get("wall_ms").and_then(|v| v.as_u64()) {
        caps.wall_ms = v;
    }
    if let Some(v) = args.get("output_bytes").and_then(|v| v.as_u64()) {
        caps.output_bytes = v as usize;
    }
    reg.set_caps(caps);
    let text = format!(
        "capability_runtime configured: enabled={} allow_unsandboxed={} caps(cpu_ms={} mem_bytes={} wall_ms={} output_bytes={})",
        reg.enabled, reg.allow_unsandboxed, caps.cpu_ms, caps.mem_bytes, caps.wall_ms, caps.output_bytes
    );
    Ok(wrap_content(
        text,
        json!({
            "enabled": reg.enabled,
            "allow_unsandboxed": reg.allow_unsandboxed,
            "caps": {
                "cpu_ms": caps.cpu_ms,
                "mem_bytes": caps.mem_bytes,
                "wall_ms": caps.wall_ms,
                "output_bytes": caps.output_bytes,
            }
        }),
    ))
}

// ----------------------------------------------------------------
// lu4d_capability_pack_runtime_status
// ----------------------------------------------------------------
pub fn tool_runtime_status(brain: &mut Brain, _args: Value) -> Result<Value, (i64, String)> {
    let reg = &brain.pack_registry;
    let tools: Vec<Value> = reg.tool_decls();
    let packs: Vec<Value> = reg
        .loaded_pack_ids()
        .iter()
        .filter_map(|id| reg.loaded_pack(id))
        .map(|p| {
            json!({
                "pack_id": p.pack_id,
                "pack_version": p.pack_version,
                "publisher_pubkey_hex": p.publisher_pubkey_hex,
                "source_path": p.source_path,
                "manifest_hash_hex": p.manifest_hash_hex,
                "tool_names": p.tool_names,
                "brain_write": p.permissions.brain_write,
                "brain_read": p.permissions.brain_read,
            })
        })
        .collect();
    let text = format!(
        "capability_runtime status: enabled={} loaded_packs={} registered_tools={}",
        reg.enabled,
        packs.len(),
        tools.len()
    );
    Ok(wrap_content(
        text,
        json!({
            "enabled": reg.enabled,
            "allow_unsandboxed": reg.allow_unsandboxed,
            "caps": {
                "cpu_ms": reg.caps.cpu_ms,
                "mem_bytes": reg.caps.mem_bytes,
                "wall_ms": reg.caps.wall_ms,
                "output_bytes": reg.caps.output_bytes,
            },
            "loaded_packs": packs,
            "registered_tools": tools,
        }),
    ))
}

// ----------------------------------------------------------------
// lu4d_capability_pack_load
// ----------------------------------------------------------------
pub fn tool_pack_load(brain: &mut Brain, args: Value) -> Result<Value, (i64, String)> {
    // Resolve the source path: explicit `path`, else look up the
    // installed pack's recorded source_path by `pack_id`.
    let src = if let Some(p) = args.get("path").and_then(|v| v.as_str()) {
        p.to_string()
    } else if let Some(id) = args.get("pack_id").and_then(|v| v.as_str()) {
        brain
            .cap_pack_ledger
            .installed_packs
            .get(id)
            .map(|i| i.source_path.clone())
            .ok_or((
                -32602,
                format!("pack_id {} not found in install ledger (install it first, or pass path)", id),
            ))?
    } else {
        return Err((-32602, "missing required arg: pack_id or path".to_string()));
    };

    let (pack, bytes) = read_capability_pack_file(&src)
        .map_err(|e| (-32603, format!("read pack {}: {}", src, e)))?;
    let clk = brain.clock.clone();
    let summary = brain
        .pack_registry
        .load(&pack, &bytes, &src, &brain.cap_pack_ledger, clk)
        .map_err(map_runtime_err)?;
    let text = format!(
        "capability_pack_load ok: pack_id={} pack_version={} runtime={:?} tools={:?}",
        summary.pack_id, summary.pack_version, summary.runtime_kind, summary.tool_names
    );
    Ok(wrap_content(
        text,
        json!({
            "pack_id": summary.pack_id,
            "pack_version": summary.pack_version,
            "runtime_kind": format!("{:?}", summary.runtime_kind),
            "registered_tools": summary.tool_names,
            "source_path": src,
        }),
    ))
}

// ----------------------------------------------------------------
// lu4d_capability_pack_unload
// ----------------------------------------------------------------
pub fn tool_pack_unload(brain: &mut Brain, args: Value) -> Result<Value, (i64, String)> {
    let pack_id = args
        .get("pack_id")
        .and_then(|v| v.as_str())
        .ok_or((-32602, "missing required arg: pack_id".to_string()))?
        .to_string();
    let n = brain.pack_registry.unload(&pack_id).map_err(map_runtime_err)?;
    Ok(wrap_content(
        format!("capability_pack_unload ok: pack_id={} tools_removed={}", pack_id, n),
        json!({ "pack_id": pack_id, "tools_removed": n }),
    ))
}

// ----------------------------------------------------------------
// Dynamic dispatch for pack__* tools
// ----------------------------------------------------------------
pub fn dispatch_pack_tool(
    brain: &mut Brain,
    name: &str,
    args: Value,
) -> Result<Value, (i64, String)> {
    let pack_id = brain.pack_registry.owning_pack_id(name).unwrap_or_default();
    let granted_brain_write = brain.pack_registry.tool_brain_write_granted(name);

    // execute() re-verifies (trust + revocation + signature + manifest
    // hash) BEFORE running the subprocess, under the configured caps.
    let out = brain
        .pack_registry
        .execute(name, &args, &brain.cap_pack_ledger)
        .map_err(map_runtime_err)?;

    let mut queued = 0u64;
    if !out.brain_writes.is_empty() {
        // DEFAULT-DENY: a pack proposing graph writes without the
        // brain_write grant is refused outright. Load-bearing — this is
        // the "action outside granted permissions is DENIED" gate.
        if !granted_brain_write {
            return Err((
                -32603,
                format!(
                    "permission denied: pack tool {} proposed {} brain_write(s) but lacks brain_write permission",
                    name,
                    out.brain_writes.len()
                ),
            ));
        }
        // Granted: route through the runtime PendingLog as teach-class
        // proposals. Status Pending => bonds are NOT mutated and no
        // retrain occurs. Operator approval (lu4d_approve) is required
        // before anything reaches the graph.
        let now = now_ms(brain);
        for w in &out.brain_writes {
            let change = PendingChange {
                pending_id: 0, // assigned by the log
                proposed_at_ms: now,
                proposed_by: SourceAttribution::teach(format!("capability_pack:{}", pack_id), now),
                change_kind: PendingChangeKind::AddBond {
                    subject_text: w.subject_text.clone(),
                    object_text: w.object_text.clone(),
                    bond_type: w.bond_type,
                    weight: w.weight,
                    negated: w.negated,
                },
                confidence: w.weight,
                reason: format!(
                    "capability pack runtime brain_write via PendingLog (tool {})",
                    name
                ),
                status: PendingStatus::Pending,
            };
            brain
                .pending
                .push(change)
                .map_err(|e| (-32603, format!("pending push: {}", e)))?;
            queued += 1;
        }
    }

    let text = format!(
        "pack tool {} ok (pack_id={}, queued_brain_writes={})",
        name, pack_id, queued
    );
    Ok(wrap_content(
        text,
        json!({
            "tool": name,
            "pack_id": pack_id,
            "result": out.result,
            "queued_brain_writes": queued,
        }),
    ))
}

// ----------------------------------------------------------------
// tools/list definitions for the four control tools
// ----------------------------------------------------------------
pub fn static_tool_defs() -> Vec<Value> {
    vec![
        json!({
            "name": "lu4d_capability_pack_runtime_configure",
            "description": "Cat.5.1 - configure the capability execution runtime: enabled (master switch, default false), allow_unsandboxed (permit packs requesting fs/network without OS sandbox; default false), and resource caps cpu_ms/mem_bytes/wall_ms/output_bytes.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "enabled": {"type": "boolean"},
                    "allow_unsandboxed": {"type": "boolean"},
                    "cpu_ms": {"type": "integer"},
                    "mem_bytes": {"type": "integer"},
                    "wall_ms": {"type": "integer"},
                    "output_bytes": {"type": "integer"}
                }
            }
        }),
        json!({
            "name": "lu4d_capability_pack_runtime_status",
            "description": "Cat.5.1 - report capability runtime state: enabled, caps, loaded packs and registered pack tools.",
            "inputSchema": {"type": "object", "properties": {}}
        }),
        json!({
            "name": "lu4d_capability_pack_load",
            "description": "Cat.5.1 - load a signed capability pack into the execution runtime by pack_id (uses install-ledger source_path) or explicit path. Re-verifies trust + revocation + signature, then registers its tools (callable as pack__<pack_id>__<tool>).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "pack_id": {"type": "string"},
                    "path": {"type": "string"}
                }
            }
        }),
        json!({
            "name": "lu4d_capability_pack_unload",
            "description": "Cat.5.1 - unload a capability pack from the runtime, removing its registered tools and its extracted executable.",
            "inputSchema": {
                "type": "object",
                "properties": {"pack_id": {"type": "string"}},
                "required": ["pack_id"]
            }
        }),
    ]
}
