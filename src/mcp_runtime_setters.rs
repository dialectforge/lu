//! Port BR.2 -- per-port RuntimeConfig setters and a multi-call
//! batch dispatcher.
//!
//! Background: as of internal notes / BR.0 preflight audit, the
//! `lu4d_configure` omnibus tool only exposed 10 of the 30 fields
//! in `RuntimeConfigSnapshot`. The other 20 fields (sense_*,
//! chained_research_*, ingest_* / tier2_prompt_*, file_io_*, image_*)
//! were populated at compile / load time only and unreachable at
//! runtime via MCP.
//!
//! BR.2 closes that gap by adding five per-port setters
//! (`lu4d_sense_set`, `lu4d_chained_research_set`, `lu4d_image_set`,
//! `lu4d_file_io_set`, `lu4d_ingest_set`) plus an ergonomic
//! `lu4d_configure_batch` that dispatches a heterogeneous list of
//! `{tool, args}` calls in best-effort sequential order.
//!
//! Persistence: each setter mutates the relevant `Brain` runtime
//! field directly (e.g. `brain.sense_config.context_window`). The
//! existing `RuntimeConfigSnapshot` encoder in `Brain::save_v3`
//! re-derives section #15 from these fields, so the next save
//! picks up the change with no extra plumbing.
//!
//! Architectural note: the per-port surface is the canonical one.
//! `lu4d_configure` is kept (so existing callers still work) but is
//! marked deprecated in its tool description. Removal is deferred
//! to a future port.

use crate::brain::Brain;
use crate::runtime_config::{IngestRunIdStrategy, Tier2PromptSessionStrategy};
use crate::mcp_vision::VisionExtractorKind;
use serde_json::{json, Value};

fn br2_wrap(text: String, structured: Value) -> Value {
    json!({
        "content": [{"type": "text", "text": text}],
        "structuredContent": structured,
    })
}

// =====================================================================
// Port Sense -- runtime sense disambiguation knobs.
// =====================================================================

/// `lu4d_sense_set` -- mutate any subset of the three Sense knobs.
/// Unspecified keys are left unchanged. Pass with no args to read
/// the current effective config.
pub(crate) fn tool_sense_set(
    brain: &mut Brain,
    args: Value,
) -> Result<Value, (i64, String)> {
    let mut changes: Vec<String> = Vec::new();
    if let Some(v) = args.get("sense_disambiguation_enabled").and_then(|v| v.as_bool()) {
        let prev = brain.sense_config.disambiguation_enabled;
        brain.sense_config.disambiguation_enabled = v;
        changes.push(format!("sense_disambiguation_enabled {} -> {}", prev, v));
    }
    if let Some(v) = args.get("sense_context_window").and_then(|v| v.as_u64()) {
        // Bound: 1..=4096. 0 would silence the spread-path consumer
        // entirely; >4096 would be a degenerate workload.
        if v < 1 || v > 4096 {
            return Err((-32602, format!(
                "sense_context_window out of bounds [1, 4096]: {}", v
            )));
        }
        let prev = brain.sense_config.context_window;
        brain.sense_config.context_window = v as usize;
        changes.push(format!("sense_context_window {} -> {}", prev, v));
    }
    if let Some(v) = args.get("sense_loser_gain").and_then(|v| v.as_f64()) {
        if !(0.0..=1.0).contains(&v) {
            return Err((-32602, format!(
                "sense_loser_gain out of bounds [0.0, 1.0]: {}", v
            )));
        }
        let prev = brain.sense_config.loser_gain;
        brain.sense_config.loser_gain = v as f32;
        changes.push(format!("sense_loser_gain {:.3} -> {:.3}", prev, v));
    }

    let text = if changes.is_empty() {
        format!(
            "sense_set (read-only): disambiguation_enabled={} context_window={} loser_gain={:.3}",
            brain.sense_config.disambiguation_enabled,
            brain.sense_config.context_window,
            brain.sense_config.loser_gain,
        )
    } else {
        format!("sense_set: {}", changes.join("; "))
    };
    Ok(br2_wrap(text, json!({
        "sense_disambiguation_enabled": brain.sense_config.disambiguation_enabled,
        "sense_context_window": brain.sense_config.context_window,
        "sense_loser_gain": brain.sense_config.loser_gain,
        "applied_changes": changes,
    })))
}

// =====================================================================
// Port J.2 -- chained curiosity research knobs.
// =====================================================================

/// `lu4d_chained_research_set` -- mutate any subset of the four
/// chained-research knobs. Bound checks: depth in 1..=64, cycles in
/// 1..=1024, gap_threshold in 0.0..=1.0.
pub(crate) fn tool_chained_research_set(
    brain: &mut Brain,
    args: Value,
) -> Result<Value, (i64, String)> {
    let mut changes: Vec<String> = Vec::new();
    if let Some(v) = args.get("chained_research_enabled").and_then(|v| v.as_bool()) {
        let prev = brain.chain_research_config.enabled;
        brain.chain_research_config.enabled = v;
        changes.push(format!("chained_research_enabled {} -> {}", prev, v));
    }
    if let Some(v) = args.get("max_chain_depth").and_then(|v| v.as_u64()) {
        if v < 1 || v > 64 {
            return Err((-32602, format!(
                "max_chain_depth out of bounds [1, 64]: {}", v
            )));
        }
        let prev = brain.chain_research_config.max_chain_depth;
        brain.chain_research_config.max_chain_depth = v as u32;
        changes.push(format!("max_chain_depth {} -> {}", prev, v));
    }
    if let Some(v) = args.get("max_chain_cycles").and_then(|v| v.as_u64()) {
        if v < 1 || v > 1024 {
            return Err((-32602, format!(
                "max_chain_cycles out of bounds [1, 1024]: {}", v
            )));
        }
        let prev = brain.chain_research_config.max_chain_cycles;
        brain.chain_research_config.max_chain_cycles = v as u32;
        changes.push(format!("max_chain_cycles {} -> {}", prev, v));
    }
    if let Some(v) = args.get("chain_gap_threshold").and_then(|v| v.as_f64()) {
        if !(0.0..=1.0).contains(&v) {
            return Err((-32602, format!(
                "chain_gap_threshold out of bounds [0.0, 1.0]: {}", v
            )));
        }
        let prev = brain.chain_research_config.gap_threshold;
        brain.chain_research_config.gap_threshold = v as f32;
        changes.push(format!("chain_gap_threshold {:.3} -> {:.3}", prev, v));
    }

    let text = if changes.is_empty() {
        format!(
            "chained_research_set (read-only): enabled={} max_depth={} max_cycles={} gap_threshold={:.3}",
            brain.chain_research_config.enabled,
            brain.chain_research_config.max_chain_depth,
            brain.chain_research_config.max_chain_cycles,
            brain.chain_research_config.gap_threshold,
        )
    } else {
        format!("chained_research_set: {}", changes.join("; "))
    };
    Ok(br2_wrap(text, json!({
        "chained_research_enabled": brain.chain_research_config.enabled,
        "max_chain_depth": brain.chain_research_config.max_chain_depth,
        "max_chain_cycles": brain.chain_research_config.max_chain_cycles,
        "chain_gap_threshold": brain.chain_research_config.gap_threshold,
        "applied_changes": changes,
    })))
}

// =====================================================================
// Port R input -- user-facing image / vision ingest knobs.
// =====================================================================

fn parse_extractor_kind(v: &Value) -> Result<VisionExtractorKind, String> {
    if let Some(s) = v.as_str() {
        match s {
            "stub" | "Stub" => return Ok(VisionExtractorKind::Stub),
            "subprocess" | "Subprocess" => return Ok(VisionExtractorKind::Subprocess),
            other => return Err(format!(
                "image_extractor_kind: unknown variant '{}' (expected 'stub' or 'subprocess')",
                other
            )),
        }
    }
    if let Some(o) = v.as_object() {
        if let Some(pid) = o.get("capability_pack").and_then(|x| x.as_str()) {
            return Ok(VisionExtractorKind::CapabilityPack { pack_id: pid.to_string() });
        }
    }
    Err("image_extractor_kind: must be 'stub' | 'subprocess' | {capability_pack: <id>}".to_string())
}

/// `lu4d_image_set` -- mutate any subset of the five vision knobs.
pub(crate) fn tool_image_set(
    brain: &mut Brain,
    args: Value,
) -> Result<Value, (i64, String)> {
    let mut changes: Vec<String> = Vec::new();

    if let Some(v) = args.get("image_extractor_kind") {
        let kind = parse_extractor_kind(v).map_err(|e| (-32602, e))?;
        let prev = brain.vision_config.extractor_kind.clone();
        brain.vision_config.extractor_kind = kind.clone();
        changes.push(format!("image_extractor_kind {:?} -> {:?}", prev, kind));
    }
    if let Some(v) = args.get("image_extractor_subprocess_path") {
        let prev = brain.vision_config.subprocess_path.clone();
        let new_val: Option<String> = if v.is_null() {
            None
        } else if let Some(s) = v.as_str() {
            if s.is_empty() { None } else { Some(s.to_string()) }
        } else {
            return Err((-32602, "image_extractor_subprocess_path must be string or null".to_string()));
        };
        brain.vision_config.subprocess_path = new_val.clone();
        changes.push(format!("image_extractor_subprocess_path {:?} -> {:?}", prev, new_val));
    }
    if let Some(v) = args.get("image_extractor_subprocess_timeout_ms").and_then(|v| v.as_u64()) {
        // Bound: 100ms..=10 minutes. <100ms is a misconfiguration; >10min
        // would let a stuck subprocess wedge the MCP loop indefinitely.
        if v < 100 || v > 600_000 {
            return Err((-32602, format!(
                "image_extractor_subprocess_timeout_ms out of bounds [100, 600000]: {}", v
            )));
        }
        let prev = brain.vision_config.subprocess_timeout_ms;
        brain.vision_config.subprocess_timeout_ms = v;
        changes.push(format!("image_extractor_subprocess_timeout_ms {} -> {}", prev, v));
    }
    if let Some(v) = args.get("image_max_size_bytes").and_then(|v| v.as_u64()) {
        // Bound: 1KB..=1GB. >1GB is unrealistic for image ingest.
        if v < 1024 || v > 1024 * 1024 * 1024 {
            return Err((-32602, format!(
                "image_max_size_bytes out of bounds [1024, 1073741824]: {}", v
            )));
        }
        let prev = brain.vision_config.image_max_size_bytes;
        brain.vision_config.image_max_size_bytes = v as usize;
        changes.push(format!("image_max_size_bytes {} -> {}", prev, v));
    }
    if let Some(v) = args.get("image_supported_mime_types") {
        let arr = v.as_array().ok_or((
            -32602,
            "image_supported_mime_types must be an array of strings".to_string(),
        ))?;
        let mut new_list: Vec<String> = Vec::with_capacity(arr.len());
        for entry in arr {
            let s = entry.as_str().ok_or((
                -32602,
                "image_supported_mime_types entries must be strings".to_string(),
            ))?;
            new_list.push(s.to_string());
        }
        let prev_len = brain.vision_config.image_supported_mime_types.len();
        brain.vision_config.image_supported_mime_types = new_list.clone();
        changes.push(format!(
            "image_supported_mime_types ({} entries) -> ({} entries)",
            prev_len, new_list.len()
        ));
    }

    let text = if changes.is_empty() {
        format!(
            "image_set (read-only): kind={:?} timeout_ms={} max_size_bytes={} mime_types={}",
            brain.vision_config.extractor_kind,
            brain.vision_config.subprocess_timeout_ms,
            brain.vision_config.image_max_size_bytes,
            brain.vision_config.image_supported_mime_types.len(),
        )
    } else {
        format!("image_set: {}", changes.join("; "))
    };
    Ok(br2_wrap(text, json!({
        "image_extractor_kind": format!("{:?}", brain.vision_config.extractor_kind),
        "image_extractor_subprocess_path": brain.vision_config.subprocess_path,
        "image_extractor_subprocess_timeout_ms": brain.vision_config.subprocess_timeout_ms,
        "image_max_size_bytes": brain.vision_config.image_max_size_bytes,
        "image_supported_mime_types": brain.vision_config.image_supported_mime_types,
        "applied_changes": changes,
    })))
}

// =====================================================================
// Port File-IO -- file ingest / emission knobs.
// =====================================================================

fn coerce_string_array(v: &Value, key: &str) -> Result<Vec<String>, (i64, String)> {
    let arr = v.as_array().ok_or((
        -32602,
        format!("{} must be an array of strings", key),
    ))?;
    let mut out: Vec<String> = Vec::with_capacity(arr.len());
    for entry in arr {
        let s = entry.as_str().ok_or((
            -32602,
            format!("{} entries must be strings", key),
        ))?;
        out.push(s.to_string());
    }
    Ok(out)
}

/// `lu4d_file_io_set` -- mutate any subset of the three file-io knobs.
pub(crate) fn tool_file_io_set(
    brain: &mut Brain,
    args: Value,
) -> Result<Value, (i64, String)> {
    let mut changes: Vec<String> = Vec::new();
    if let Some(v) = args.get("file_io_max_size_bytes").and_then(|v| v.as_u64()) {
        // Bound: 1KB..=2GB. >2GB exceeds typical usize on 32-bit but
        // we run 64-bit; cap conservatively to keep memory pressure sane.
        if v < 1024 || v > 2_u64 * 1024 * 1024 * 1024 {
            return Err((-32602, format!(
                "file_io_max_size_bytes out of bounds [1024, 2147483648]: {}", v
            )));
        }
        let prev = brain.file_io_config.max_size_bytes;
        brain.file_io_config.max_size_bytes = v as usize;
        changes.push(format!("file_io_max_size_bytes {} -> {}", prev, v));
    }
    if let Some(v) = args.get("file_io_allowed_read_paths") {
        let new_list = coerce_string_array(v, "file_io_allowed_read_paths")?;
        let prev_len = brain.file_io_config.allowed_read_paths.len();
        brain.file_io_config.allowed_read_paths = new_list.clone();
        changes.push(format!(
            "file_io_allowed_read_paths ({} entries) -> ({} entries)",
            prev_len, new_list.len()
        ));
    }
    if let Some(v) = args.get("file_io_allowed_write_paths") {
        let new_list = coerce_string_array(v, "file_io_allowed_write_paths")?;
        let prev_len = brain.file_io_config.allowed_write_paths.len();
        brain.file_io_config.allowed_write_paths = new_list.clone();
        changes.push(format!(
            "file_io_allowed_write_paths ({} entries) -> ({} entries)",
            prev_len, new_list.len()
        ));
    }

    let text = if changes.is_empty() {
        format!(
            "file_io_set (read-only): max_size_bytes={} read_paths={} write_paths={}",
            brain.file_io_config.max_size_bytes,
            brain.file_io_config.allowed_read_paths.len(),
            brain.file_io_config.allowed_write_paths.len(),
        )
    } else {
        format!("file_io_set: {}", changes.join("; "))
    };
    Ok(br2_wrap(text, json!({
        "file_io_max_size_bytes": brain.file_io_config.max_size_bytes,
        "file_io_allowed_read_paths": brain.file_io_config.allowed_read_paths,
        "file_io_allowed_write_paths": brain.file_io_config.allowed_write_paths,
        "applied_changes": changes,
    })))
}

// =====================================================================
// Port T-Ingest / T-Ingest.2 -- training-corpus + tier-2-prompt knobs.
//
// Grouped together because tier2_prompt_* fields are documented in
// runtime_config.rs as Port T-Ingest.2 (a follow-up to T-Ingest), and
// keeping them in one setter avoids fragmenting the ingest surface.
// =====================================================================

fn parse_run_id_strategy(s: &str) -> Result<IngestRunIdStrategy, String> {
    match s {
        "caller" | "Caller" => Ok(IngestRunIdStrategy::Caller),
        "server_startup" | "ServerStartup" => Ok(IngestRunIdStrategy::ServerStartup),
        "per_session" | "PerSession" => Ok(IngestRunIdStrategy::PerSession),
        other => Err(format!(
            "ingest_run_id_strategy: unknown variant '{}' (expected 'caller' | 'server_startup' | 'per_session')",
            other
        )),
    }
}

fn parse_tier2_session_strategy(s: &str) -> Result<Tier2PromptSessionStrategy, String> {
    match s {
        "ephemeral" | "Ephemeral" => Ok(Tier2PromptSessionStrategy::Ephemeral),
        "reuse" | "Reuse" => Ok(Tier2PromptSessionStrategy::Reuse),
        other => Err(format!(
            "tier2_prompt_default_session_strategy: unknown variant '{}' (expected 'ephemeral' | 'reuse')",
            other
        )),
    }
}

/// `lu4d_ingest_set` -- mutate any subset of the five ingest knobs
/// (three T-Ingest, two T-Ingest.2). Pass `null` to clear an
/// `Option<>` field. Pass with no args to read the current config.
pub(crate) fn tool_ingest_set(
    brain: &mut Brain,
    args: Value,
) -> Result<Value, (i64, String)> {
    let mut changes: Vec<String> = Vec::new();

    if let Some(v) = args.get("ingest_corpus_default") {
        let prev = brain.ingest_config.corpus_default.clone();
        let new_val: Option<String> = if v.is_null() {
            None
        } else if let Some(s) = v.as_str() {
            if s.is_empty() { None } else { Some(s.to_string()) }
        } else {
            return Err((-32602, "ingest_corpus_default must be string or null".to_string()));
        };
        brain.ingest_config.corpus_default = new_val.clone();
        changes.push(format!("ingest_corpus_default {:?} -> {:?}", prev, new_val));
    }
    if let Some(v) = args.get("ingest_run_id_strategy").and_then(|v| v.as_str()) {
        let strat = parse_run_id_strategy(v).map_err(|e| (-32602, e))?;
        let prev = brain.ingest_config.run_id_strategy.clone();
        brain.ingest_config.run_id_strategy = strat.clone();
        changes.push(format!("ingest_run_id_strategy {:?} -> {:?}", prev, strat));
    }
    if let Some(v) = args.get("ingest_auto_approve_threshold") {
        let prev = brain.ingest_config.auto_approve_threshold;
        let new_val: Option<f32> = if v.is_null() {
            None
        } else if let Some(f) = v.as_f64() {
            if !(0.0..=1.0).contains(&f) {
                return Err((-32602, format!(
                    "ingest_auto_approve_threshold out of bounds [0.0, 1.0]: {}", f
                )));
            }
            Some(f as f32)
        } else {
            return Err((-32602,
                "ingest_auto_approve_threshold must be number or null".to_string()));
        };
        brain.ingest_config.auto_approve_threshold = new_val;
        changes.push(format!("ingest_auto_approve_threshold {:?} -> {:?}", prev, new_val));
    }
    if let Some(v) = args.get("tier2_prompt_default_session_strategy").and_then(|v| v.as_str()) {
        let strat = parse_tier2_session_strategy(v).map_err(|e| (-32602, e))?;
        let prev = brain.tier2_prompt_config.default_session_strategy.clone();
        brain.tier2_prompt_config.default_session_strategy = strat.clone();
        changes.push(format!("tier2_prompt_default_session_strategy {:?} -> {:?}", prev, strat));
    }
    if let Some(v) = args.get("tier2_prompt_max_text_bytes").and_then(|v| v.as_u64()) {
        // Bound: 256B..=16MB. <256B is too small for most prompts;
        // >16MB would let a single prompt blow up LB.
        if v < 256 || v > 16 * 1024 * 1024 {
            return Err((-32602, format!(
                "tier2_prompt_max_text_bytes out of bounds [256, 16777216]: {}", v
            )));
        }
        let prev = brain.tier2_prompt_config.max_text_bytes;
        brain.tier2_prompt_config.max_text_bytes = v as usize;
        changes.push(format!("tier2_prompt_max_text_bytes {} -> {}", prev, v));
    }

    let text = if changes.is_empty() {
        format!(
            "ingest_set (read-only): corpus={:?} run_id_strategy={:?} auto_approve={:?} tier2_strategy={:?} tier2_max_bytes={}",
            brain.ingest_config.corpus_default,
            brain.ingest_config.run_id_strategy,
            brain.ingest_config.auto_approve_threshold,
            brain.tier2_prompt_config.default_session_strategy,
            brain.tier2_prompt_config.max_text_bytes,
        )
    } else {
        format!("ingest_set: {}", changes.join("; "))
    };
    Ok(br2_wrap(text, json!({
        "ingest_corpus_default": brain.ingest_config.corpus_default,
        "ingest_run_id_strategy": format!("{:?}", brain.ingest_config.run_id_strategy),
        "ingest_auto_approve_threshold": brain.ingest_config.auto_approve_threshold,
        "tier2_prompt_default_session_strategy": format!("{:?}", brain.tier2_prompt_config.default_session_strategy),
        "tier2_prompt_max_text_bytes": brain.tier2_prompt_config.max_text_bytes,
        "applied_changes": changes,
    })))
}

// =====================================================================
// Port BR.4 -- recall-time typed-KU runtime kill switch.
// =====================================================================

/// `lu4d_recall_set` -- mutate the recall-time typed-KU flag.
///
/// The flag is a kill switch on Q.1 typed-KU activation in the recall
/// pipeline (`reasoning.rs` chain composer + procedure assembler).
/// `true` (default) routes endpoint `KnowledgeContent` kinds into
/// `articulate_bond_typed`, which appends a `(kind)` tag in prose.
/// `false` falls back to the pre-Q.1 untyped baseline.
///
/// This setter currently exposes a single field; it is named
/// `lu4d_recall_set` (not `lu4d_recall_typed_ku_set`) so future
/// recall-shaped knobs can land here without inventing a new tool.
///
/// Pass with no args to read the current effective value.
pub(crate) fn tool_recall_set(
    brain: &mut Brain,
    args: Value,
) -> Result<Value, (i64, String)> {
    let mut changes: Vec<String> = Vec::new();
    if let Some(v) = args.get("recall_typed_ku_enabled").and_then(|v| v.as_bool()) {
        let prev = brain.recall_config.typed_ku_enabled;
        brain.recall_config.typed_ku_enabled = v;
        changes.push(format!("recall_typed_ku_enabled {} -> {}", prev, v));
    }

    let text = if changes.is_empty() {
        format!(
            "recall_set (read-only): typed_ku_enabled={}",
            brain.recall_config.typed_ku_enabled,
        )
    } else {
        format!("recall_set: {}", changes.join("; "))
    };
    Ok(br2_wrap(text, json!({
        "recall_typed_ku_enabled": brain.recall_config.typed_ku_enabled,
        "applied_changes": changes,
    })))
}

// =====================================================================
// Port BR.5 -- `lu4d_admin_set` setter for admin-tier runtime gates.
//
// Currently exposes a single field: `admin_revoke_enabled` (bool,
// default false). When false, the destructive lu4d_revoke_by_source
// path returns -32603 with a directive to call this setter first.
//
// Default-deny posture: every brain ships with admin_revoke_enabled
// = false. Operators must explicitly call this setter before any
// production rollback. Persisted via runtime_config section #15 so
// the gate state survives restarts.
//
// Pass with no args to read the current effective value.
//
// This setter is named `lu4d_admin_set` (not `lu4d_revoke_admin_set`)
// so future admin-tier knobs can land here without inventing a new
// tool -- e.g. a future `admin_pack_unsign_enabled` gate, or
// per-actor allow-lists if a true RBAC layer is ever built.
// =====================================================================
pub(crate) fn tool_admin_set(
    brain: &mut Brain,
    args: Value,
) -> Result<Value, (i64, String)> {
    let mut changes: Vec<String> = Vec::new();
    if let Some(v) = args.get("admin_revoke_enabled").and_then(|v| v.as_bool()) {
        let prev = brain.admin_revoke_enabled;
        brain.admin_revoke_enabled = v;
        changes.push(format!("admin_revoke_enabled {} -> {}", prev, v));
    }

    let text = if changes.is_empty() {
        format!(
            "admin_set (read-only): admin_revoke_enabled={}",
            brain.admin_revoke_enabled,
        )
    } else {
        format!("admin_set: {}", changes.join("; "))
    };
    Ok(br2_wrap(text, json!({
        "admin_revoke_enabled": brain.admin_revoke_enabled,
        "applied_changes": changes,
    })))
}

// =====================================================================
// `lu4d_configure_batch` -- best-effort sequential dispatch over a
// heterogeneous list of {tool, args} entries.
//
// Semantics: NOT transactional. Each entry is dispatched in order,
// independently mutating brain state. If entry K fails, entries
// 0..K-1 remain applied; entries K+1..N continue executing.
// Return shape: an array of per-operation results, each carrying
// either { ok: structuredContent } or { error: { code, message } }.
//
// Limited to the BR.2 setter surface plus the legacy
// `lu4d_configure` so callers can mix old + new shapes in one batch.
// Adding more dispatch targets later is a one-line change in the
// match below.
// =====================================================================

pub(crate) fn tool_configure_batch(
    brain: &mut Brain,
    args: Value,
) -> Result<Value, (i64, String)> {
    let calls = args
        .get("calls")
        .ok_or((-32602, "missing required 'calls' array".to_string()))?
        .as_array()
        .ok_or((-32602, "'calls' must be an array of {tool, args} objects".to_string()))?
        .clone();
    if calls.is_empty() {
        return Err((-32602, "'calls' must be non-empty".to_string()));
    }

    let mut results: Vec<Value> = Vec::with_capacity(calls.len());
    let mut applied = 0usize;
    let mut failed = 0usize;

    for (idx, entry) in calls.into_iter().enumerate() {
        let tool = entry.get("tool").and_then(|v| v.as_str()).unwrap_or("").to_string();
        let inner_args = entry.get("args").cloned().unwrap_or_else(|| json!({}));
        let outcome = match tool.as_str() {
            "lu4d_sense_set"             => tool_sense_set(brain, inner_args.clone()),
            "lu4d_chained_research_set"  => tool_chained_research_set(brain, inner_args.clone()),
            "lu4d_image_set"             => tool_image_set(brain, inner_args.clone()),
            "lu4d_file_io_set"           => tool_file_io_set(brain, inner_args.clone()),
            "lu4d_ingest_set"            => tool_ingest_set(brain, inner_args.clone()),
            "lu4d_recall_set"            => tool_recall_set(brain, inner_args.clone()),
            "lu4d_admin_set"             => tool_admin_set(brain, inner_args.clone()),
            "lu4d_configure"             => crate::mcp_t5::tool_configure(brain, inner_args.clone()),
            other => Err((-32602, format!(
                "unsupported batch tool: '{}' (allowed: lu4d_sense_set, lu4d_chained_research_set, lu4d_image_set, lu4d_file_io_set, lu4d_ingest_set, lu4d_recall_set, lu4d_admin_set, lu4d_configure)",
                other
            ))),
        };
        let entry_result = match outcome {
            Ok(v) => {
                applied += 1;
                json!({
                    "index": idx,
                    "tool": tool,
                    "ok": true,
                    "structuredContent": v.get("structuredContent").cloned().unwrap_or(json!({})),
                })
            }
            Err((code, msg)) => {
                failed += 1;
                json!({
                    "index": idx,
                    "tool": tool,
                    "ok": false,
                    "error": { "code": code, "message": msg },
                })
            }
        };
        results.push(entry_result);
    }

    let total = results.len();
    let text = format!(
        "configure_batch: {} call(s), {} applied, {} failed",
        total, applied, failed
    );
    Ok(br2_wrap(text, json!({
        "total": total,
        "applied": applied,
        "failed": failed,
        "results": results,
    })))
}

// =====================================================================
// Tests -- round-trip every setter + the batch dispatcher.
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn fresh_brain() -> Brain {
        Brain::new()
    }

    #[test]
    fn sense_set_roundtrip_all_three_fields() {
        let mut b = fresh_brain();
        let v = tool_sense_set(&mut b, json!({
            "sense_disambiguation_enabled": false,
            "sense_context_window": 32,
            "sense_loser_gain": 0.75,
        })).unwrap();
        let s = &v["structuredContent"];
        assert_eq!(s["sense_disambiguation_enabled"], json!(false));
        assert_eq!(s["sense_context_window"], json!(32));
        assert!((s["sense_loser_gain"].as_f64().unwrap() - 0.75).abs() < 1e-6);
        // Read-back from brain field is the source of truth.
        assert!(!b.sense_config.disambiguation_enabled);
        assert_eq!(b.sense_config.context_window, 32);
        assert!((b.sense_config.loser_gain - 0.75).abs() < 1e-6);
    }

    #[test]
    fn sense_set_rejects_out_of_bounds() {
        let mut b = fresh_brain();
        assert!(tool_sense_set(&mut b, json!({"sense_context_window": 0})).is_err());
        assert!(tool_sense_set(&mut b, json!({"sense_context_window": 99999})).is_err());
        assert!(tool_sense_set(&mut b, json!({"sense_loser_gain": -0.1})).is_err());
        assert!(tool_sense_set(&mut b, json!({"sense_loser_gain": 1.5})).is_err());
    }

    #[test]
    fn chained_research_set_roundtrip_all_four_fields() {
        let mut b = fresh_brain();
        let v = tool_chained_research_set(&mut b, json!({
            "chained_research_enabled": false,
            "max_chain_depth": 7,
            "max_chain_cycles": 13,
            "chain_gap_threshold": 0.6,
        })).unwrap();
        let s = &v["structuredContent"];
        assert_eq!(s["chained_research_enabled"], json!(false));
        assert_eq!(s["max_chain_depth"], json!(7));
        assert_eq!(s["max_chain_cycles"], json!(13));
        assert!((s["chain_gap_threshold"].as_f64().unwrap() - 0.6).abs() < 1e-6);
        assert!(!b.chain_research_config.enabled);
        assert_eq!(b.chain_research_config.max_chain_depth, 7);
        assert_eq!(b.chain_research_config.max_chain_cycles, 13);
        assert!((b.chain_research_config.gap_threshold - 0.6).abs() < 1e-6);
    }

    #[test]
    fn chained_research_set_rejects_out_of_bounds() {
        let mut b = fresh_brain();
        assert!(tool_chained_research_set(&mut b, json!({"max_chain_depth": 0})).is_err());
        assert!(tool_chained_research_set(&mut b, json!({"max_chain_cycles": 0})).is_err());
        assert!(tool_chained_research_set(&mut b, json!({"chain_gap_threshold": 1.5})).is_err());
    }

    #[test]
    fn image_set_roundtrip_all_five_fields() {
        let mut b = fresh_brain();
        let v = tool_image_set(&mut b, json!({
            "image_extractor_kind": "subprocess",
            "image_extractor_subprocess_path": "/usr/local/bin/extractor",
            "image_extractor_subprocess_timeout_ms": 5000,
            "image_max_size_bytes": 10_000_000,
            "image_supported_mime_types": ["image/jpeg", "image/png"],
        })).unwrap();
        let s = &v["structuredContent"];
        assert_eq!(s["image_extractor_subprocess_path"], json!("/usr/local/bin/extractor"));
        assert_eq!(s["image_extractor_subprocess_timeout_ms"], json!(5000));
        assert_eq!(s["image_max_size_bytes"], json!(10_000_000));
        let mimes = s["image_supported_mime_types"].as_array().unwrap();
        assert_eq!(mimes.len(), 2);
        // Read-back from brain field.
        assert!(matches!(b.vision_config.extractor_kind, VisionExtractorKind::Subprocess));
        assert_eq!(b.vision_config.subprocess_timeout_ms, 5000);
        assert_eq!(b.vision_config.image_max_size_bytes, 10_000_000);
        assert_eq!(b.vision_config.image_supported_mime_types.len(), 2);
    }

    #[test]
    fn image_set_rejects_unknown_kind() {
        let mut b = fresh_brain();
        let r = tool_image_set(&mut b, json!({"image_extractor_kind": "banana"}));
        assert!(r.is_err());
    }

    #[test]
    fn file_io_set_roundtrip_all_three_fields() {
        let mut b = fresh_brain();
        let v = tool_file_io_set(&mut b, json!({
            "file_io_max_size_bytes": 12_345_678,
            "file_io_allowed_read_paths": ["/srv/in/"],
            "file_io_allowed_write_paths": ["/srv/out/", "/tmp/x/"],
        })).unwrap();
        let s = &v["structuredContent"];
        assert_eq!(s["file_io_max_size_bytes"], json!(12_345_678));
        assert_eq!(b.file_io_config.max_size_bytes, 12_345_678);
        assert_eq!(b.file_io_config.allowed_read_paths, vec!["/srv/in/".to_string()]);
        assert_eq!(b.file_io_config.allowed_write_paths, vec![
            "/srv/out/".to_string(),
            "/tmp/x/".to_string(),
        ]);
    }

    #[test]
    fn file_io_set_rejects_non_string_array_entries() {
        let mut b = fresh_brain();
        let r = tool_file_io_set(&mut b, json!({"file_io_allowed_read_paths": [42]}));
        assert!(r.is_err());
    }

    #[test]
    fn ingest_set_roundtrip_all_five_fields() {
        let mut b = fresh_brain();
        let v = tool_ingest_set(&mut b, json!({
            "ingest_corpus_default": "wikimedia/wikipedia",
            "ingest_run_id_strategy": "per_session",
            "ingest_auto_approve_threshold": 0.8,
            "tier2_prompt_default_session_strategy": "reuse",
            "tier2_prompt_max_text_bytes": 131072,
        })).unwrap();
        let s = &v["structuredContent"];
        assert_eq!(s["ingest_corpus_default"], json!("wikimedia/wikipedia"));
        assert_eq!(s["tier2_prompt_max_text_bytes"], json!(131072));
        assert_eq!(b.ingest_config.corpus_default.as_deref(), Some("wikimedia/wikipedia"));
        assert!(matches!(b.ingest_config.run_id_strategy, IngestRunIdStrategy::PerSession));
        assert_eq!(b.ingest_config.auto_approve_threshold, Some(0.8));
        assert!(matches!(
            b.tier2_prompt_config.default_session_strategy,
            Tier2PromptSessionStrategy::Reuse
        ));
        assert_eq!(b.tier2_prompt_config.max_text_bytes, 131072);
    }

    #[test]
    fn ingest_set_clears_optional_fields_with_null() {
        let mut b = fresh_brain();
        b.ingest_config.corpus_default = Some("x".to_string());
        b.ingest_config.auto_approve_threshold = Some(0.5);
        let _ = tool_ingest_set(&mut b, json!({
            "ingest_corpus_default": null,
            "ingest_auto_approve_threshold": null,
        })).unwrap();
        assert!(b.ingest_config.corpus_default.is_none());
        assert!(b.ingest_config.auto_approve_threshold.is_none());
    }

    #[test]
    fn ingest_set_rejects_unknown_strategy() {
        let mut b = fresh_brain();
        let r = tool_ingest_set(&mut b, json!({"ingest_run_id_strategy": "random"}));
        assert!(r.is_err());
    }

    #[test]
    fn configure_batch_applies_all_when_all_succeed() {
        let mut b = fresh_brain();
        let v = tool_configure_batch(&mut b, json!({
            "calls": [
                {"tool": "lu4d_sense_set",            "args": {"sense_context_window": 16}},
                {"tool": "lu4d_chained_research_set", "args": {"max_chain_depth": 5}},
                {"tool": "lu4d_file_io_set",          "args": {"file_io_max_size_bytes": 1048576}},
            ]
        })).unwrap();
        let s = &v["structuredContent"];
        assert_eq!(s["total"], json!(3));
        assert_eq!(s["applied"], json!(3));
        assert_eq!(s["failed"], json!(0));
        assert_eq!(b.sense_config.context_window, 16);
        assert_eq!(b.chain_research_config.max_chain_depth, 5);
        assert_eq!(b.file_io_config.max_size_bytes, 1048576);
    }

    #[test]
    fn configure_batch_continues_after_failure_best_effort() {
        let mut b = fresh_brain();
        let v = tool_configure_batch(&mut b, json!({
            "calls": [
                {"tool": "lu4d_sense_set",            "args": {"sense_context_window": 16}},
                {"tool": "lu4d_sense_set",            "args": {"sense_context_window": 0}},
                {"tool": "lu4d_chained_research_set", "args": {"max_chain_depth": 9}},
            ]
        })).unwrap();
        let s = &v["structuredContent"];
        assert_eq!(s["total"], json!(3));
        assert_eq!(s["applied"], json!(2));
        assert_eq!(s["failed"], json!(1));
        // First entry applied, second errored, third still applied.
        assert_eq!(b.sense_config.context_window, 16);
        assert_eq!(b.chain_research_config.max_chain_depth, 9);
        let results = s["results"].as_array().unwrap();
        assert_eq!(results[0]["ok"], json!(true));
        assert_eq!(results[1]["ok"], json!(false));
        assert_eq!(results[2]["ok"], json!(true));
    }

    #[test]
    fn configure_batch_rejects_empty_calls() {
        let mut b = fresh_brain();
        let r = tool_configure_batch(&mut b, json!({"calls": []}));
        assert!(r.is_err());
    }

    #[test]
    fn configure_batch_records_unknown_tool_as_per_call_error() {
        let mut b = fresh_brain();
        let v = tool_configure_batch(&mut b, json!({
            "calls": [
                {"tool": "lu4d_made_up_tool", "args": {}},
                {"tool": "lu4d_sense_set",   "args": {"sense_context_window": 24}},
            ]
        })).unwrap();
        let s = &v["structuredContent"];
        assert_eq!(s["applied"], json!(1));
        assert_eq!(s["failed"], json!(1));
        assert_eq!(b.sense_config.context_window, 24);
    }

    // -----------------------------------------------------------------
    // Port BR.4: recall_set round-trip + batch + read-only echo.
    // -----------------------------------------------------------------

    #[test]
    fn br4_recall_set_default_is_true() {
        let b = fresh_brain();
        // Default for the runtime flag must be `true` so behavior is
        // unchanged after BR.4 lands.
        assert!(b.recall_config.typed_ku_enabled);
    }

    #[test]
    fn br4_recall_set_round_trip_flag_off_then_on() {
        let mut b = fresh_brain();
        let v = tool_recall_set(&mut b, json!({
            "recall_typed_ku_enabled": false,
        })).unwrap();
        assert_eq!(v["structuredContent"]["recall_typed_ku_enabled"], json!(false));
        assert_eq!(b.recall_config.typed_ku_enabled, false);

        let v = tool_recall_set(&mut b, json!({
            "recall_typed_ku_enabled": true,
        })).unwrap();
        assert_eq!(v["structuredContent"]["recall_typed_ku_enabled"], json!(true));
        assert_eq!(b.recall_config.typed_ku_enabled, true);
    }

    #[test]
    fn br4_recall_set_no_args_is_read_only_echo() {
        let mut b = fresh_brain();
        let v = tool_recall_set(&mut b, json!({})).unwrap();
        assert_eq!(v["structuredContent"]["recall_typed_ku_enabled"], json!(true));
        let changes = v["structuredContent"]["applied_changes"].as_array().unwrap();
        assert_eq!(changes.len(), 0);
        assert!(b.recall_config.typed_ku_enabled);
    }

    #[test]
    fn br4_recall_set_reachable_from_configure_batch() {
        let mut b = fresh_brain();
        let v = tool_configure_batch(&mut b, json!({
            "calls": [
                {"tool": "lu4d_recall_set", "args": {"recall_typed_ku_enabled": false}},
                {"tool": "lu4d_sense_set",  "args": {"sense_context_window": 16}},
            ]
        })).unwrap();
        let s = &v["structuredContent"];
        assert_eq!(s["total"], json!(2));
        assert_eq!(s["applied"], json!(2));
        assert_eq!(s["failed"], json!(0));
        assert_eq!(b.recall_config.typed_ku_enabled, false);
        assert_eq!(b.sense_config.context_window, 16);
    }

    #[test]
    fn br4_recall_flag_persists_into_section_15_runtime_config() {
        let mut b = fresh_brain();
        tool_recall_set(&mut b, json!({"recall_typed_ku_enabled": false})).unwrap();
        let file = b.to_brain_file(crate::brain_io_v3::FORMAT_VERSION_V3);
        assert_eq!(file.runtime_config.recall_typed_ku_enabled, false);
    }
}
