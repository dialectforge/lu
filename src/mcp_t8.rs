// =====================================================================
// Port T.8 — rejection-pattern MCP tools.
//
//   1. `lu4d_rejection_patterns` — list learned rejection patterns.
//      Inputs: { include_expired: bool (default false), limit: u32 (default 20) }
//      Output: { patterns: [{ pattern_id, bond_type, source_domain,
//                             sample_features, rejection_count,
//                             last_rejection_at_ms, status }],
//                total }
//
// Companion to `crate::rejection::RejectionPatternStore`. The store
// itself is owned by the Brain (`brain.rejection_store`), persisted
// via the new v3 section (`SectionIdV3::RejectionPatterns`, id 10)
// added in this port.
//
// `lu4d_pending_status` (T.3, lives in mcp_t3.rs) is also extended
// to surface `proposals_filtered_total` in its structured payload —
// kept in mcp_t3.rs so the pending-status surface stays single-call.
// =====================================================================

use crate::brain::Brain;
use crate::rejection::RejectionPattern;
use serde_json::{json, Value};

fn t8_wrap(text: String, structured: Value) -> Value {
    json!({
        "content": [{"type": "text", "text": text}],
        "structuredContent": structured,
    })
}

fn pattern_to_json(p: &RejectionPattern) -> Value {
    json!({
        "pattern_id": p.pattern_id,
        "bond_type": format!("{:?}", p.key.bond_type),
        "source_domain": p.key.source_domain,
        "sample_features": {
            "subject_starts_with_preposition": p.key.subject_starts_with_preposition,
            "subject_contains_citation_marker": p.key.subject_contains_citation_marker,
            "subject_contains_generic_pronoun": p.key.subject_contains_generic_pronoun,
            "object_ends_with_preposition": p.key.object_ends_with_preposition,
        },
        "rejection_count": p.rejection_count,
        "last_rejection_at_ms": p.last_rejection_at_ms,
        "first_learned_at_ms": p.first_learned_at_ms,
        "status": p.status.as_str(),
    })
}

/// `lu4d_rejection_patterns` — read-only listing of learned rejection
/// patterns. The proposal pre-filter consults the same data; this
/// tool surfaces what the filter is enforcing.
pub(crate) fn tool_rejection_patterns(
    brain: &Brain,
    args: Value,
) -> Result<Value, (i64, String)> {
    let include_expired = args
        .get("include_expired")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let limit = args
        .get("limit")
        .and_then(|v| v.as_u64())
        .unwrap_or(20)
        .min(1000) as usize;

    let all = brain.rejection_store.learned_patterns(include_expired);
    let total = all.len();
    let limited: Vec<Value> = all.iter().take(limit).map(|p| pattern_to_json(p)).collect();
    let text = format!(
        "rejection_patterns: total={} returned={} include_expired={} \
         proposals_filtered_total={}",
        total,
        limited.len(),
        include_expired,
        brain.rejection_store.proposals_filtered_total(),
    );
    Ok(t8_wrap(
        text,
        json!({
            "patterns": limited,
            "total": total,
            "include_expired": include_expired,
            "proposals_filtered_total": brain.rejection_store.proposals_filtered_total(),
        }),
    ))
}
