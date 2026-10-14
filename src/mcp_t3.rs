// =====================================================================
// Port T.3 — Pending changes + approval workflow MCP tools.
//
// Five tools implement the structural answer to AI safety concerns
// about autonomous self-modification: idle research, future Cat. 3
// self-rewriting passes, and any other autonomous mutation route
// their proposals through `Brain::pending` and wait for explicit
// human approve/reject/modify before committing to the brain.
//
//   1. `lu4d_pending_changes`   — list pending entries (with filters)
//   2. `lu4d_approve_pending`   — approve + apply (atomic batch)
//   3. `lu4d_reject_pending`    — reject one or many
//   4. `lu4d_modify_pending`    — edit a still-Pending entry
//   5. `lu4d_pending_status`    — summary + thinking_indicator
//
// Predecessor: T.2 (idle work loop, internal notes. Successor:
// Port U (approval UI) consumes this MCP API; T.6 (daemon driver)
// drives idle_tick on an interval and the resulting pending
// entries land here.
// =====================================================================

use crate::brain::{now_ms, Brain};
use serde_json::{json, Value};

/// Small helper that re-uses the same wrap_content shape used by
/// every other tool in this module. We can't import it across the
/// patch boundary, so we re-implement here. (Simple two-field
/// JSON object — easy to keep in sync.)
fn t3_wrap(text: String, structured: Value) -> Value {
    json!({
        "content": [{"type": "text", "text": text}],
        "structuredContent": structured,
    })
}

fn render_change(entry: &crate::pending::change::PendingChange) -> Value {
    json!({
        "pending_id": entry.pending_id,
        "proposed_at_ms": entry.proposed_at_ms,
        "source_kind": entry.proposed_by.kind.tag(),
        "source_label": entry.proposed_by.kind.short_label(),
        "change_kind_tag": entry.change_kind.tag(),
        "change_summary": entry.change_kind.summary(),
        "confidence": entry.confidence,
        "reason": entry.reason,
        "status": entry.status.tag(),
    })
}

fn collect_filtered<'a>(
    log: &'a crate::pending::PendingLog,
    status_filter: Option<&str>,
    kind_filter: Option<&str>,
    min_confidence: Option<f32>,
    max_age_ms: Option<u64>,
    now: u64,
) -> Vec<&'a crate::pending::change::PendingChange> {
    log.iter()
        .map(|(_, e)| e)
        .filter(|e| {
            if let Some(s) = status_filter {
                if !s.eq_ignore_ascii_case("all") && !s.eq_ignore_ascii_case(e.status.tag()) {
                    return false;
                }
            }
            if let Some(k) = kind_filter {
                if !k.eq_ignore_ascii_case("all")
                    && !k.eq_ignore_ascii_case(e.proposed_by.kind.tag())
                {
                    return false;
                }
            }
            if let Some(c) = min_confidence {
                if e.confidence < c {
                    return false;
                }
            }
            if let Some(age) = max_age_ms {
                if now.saturating_sub(e.proposed_at_ms) > age {
                    return false;
                }
            }
            true
        })
        .collect()
}

// =====================================================================
// lu4d_pending_changes — list pending entries.
// =====================================================================
pub(crate) fn tool_pending_changes(
    brain: &mut Brain,
    args: Value,
) -> Result<Value, (i64, String)> {
    let status_filter = args.get("status").and_then(|v| v.as_str()).map(String::from);
    let kind_filter = args
        .get("source_kind")
        .and_then(|v| v.as_str())
        .map(String::from);
    let min_conf = args
        .get("min_confidence")
        .and_then(|v| v.as_f64())
        .map(|f| f as f32);
    let max_age = args.get("max_age_ms").and_then(|v| v.as_u64());
    let limit = args
        .get("limit")
        .and_then(|v| v.as_u64())
        .unwrap_or(50) as usize;
    // Port T.5 -- `stale_only` keeps entries older than 30 days.
    let stale_only = args
        .get("stale_only")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let now = now_ms() as u64;
    let mut filtered = collect_filtered(
        &brain.pending,
        status_filter.as_deref(),
        kind_filter.as_deref(),
        min_conf,
        max_age,
        now,
    );
    if stale_only {
        filtered.retain(|e| {
            now.saturating_sub(e.proposed_at_ms) > crate::decay::STALE_PENDING_THRESHOLD_MS
        });
    }
    // Port T.5 -- newest-first ordering.
    filtered.sort_by(|a, b| {
        b.proposed_at_ms
            .cmp(&a.proposed_at_ms)
            .then_with(|| b.pending_id.cmp(&a.pending_id))
    });
    // Port T.5 -- stale_count over the still-Pending entries in the
    // full log, so callers get a filter-independent rot signal.
    let stale_count: u32 = brain
        .pending
        .iter()
        .filter(|(_, e)| e.status.is_pending())
        .filter(|(_, e)| {
            now.saturating_sub(e.proposed_at_ms) > crate::decay::STALE_PENDING_THRESHOLD_MS
        })
        .count() as u32;
    let total = filtered.len();
    let limited: Vec<&crate::pending::change::PendingChange> =
        filtered.into_iter().take(limit).collect();
    let entries: Vec<Value> = limited.iter().map(|e| render_change(e)).collect();
    let text = format!(
        "pending_changes: {} entries shown (of {} matched, stale_count={})",
        entries.len(), total, stale_count
    );
    Ok(t3_wrap(
        text,
        json!({
            "entries": entries,
            "total_count": total,
            "showing": limited.len(),
            "limit": limit,
            "stale_count": stale_count,
            "stale_threshold_days": 30,
        }),
    ))
}

// =====================================================================
// lu4d_approve_pending — approve + apply (atomic batch).
// =====================================================================
pub(crate) fn tool_approve_pending(
    brain: &mut Brain,
    args: Value,
) -> Result<Value, (i64, String)> {
    let approver = args
        .get("approved_by")
        .and_then(|v| v.as_str())
        .unwrap_or("john")
        .to_string();
    let now = now_ms() as u64;

    // Resolve the target id list.
    let ids = resolve_ids(brain, &args, /*default_status=*/ Some("Pending"));
    if ids.is_empty() {
        return Ok(t3_wrap(
            "approve_pending: no matching entries".into(),
            json!({ "approved_count": 0, "applied_count": 0, "failures": [] }),
        ));
    }

    // Atomic-ish: we approve all first (no brain mutation yet), then
    // apply each. If apply fails for one, we keep the approval but
    // surface the failure in `failures` — the caller can re-try or
    // reject manually.
    let mut approved = 0u32;
    let mut applied = 0u32;
    let mut failures: Vec<Value> = Vec::new();

    // Phase 1: approve.
    for id in &ids {
        match brain.pending.approve(*id, now, approver.clone()) {
            Ok(()) => approved += 1,
            Err(e) => failures.push(json!({
                "pending_id": id, "phase": "approve", "error": format!("{}", e),
            })),
        }
    }
    // Phase 2: apply each Approved entry.
    for id in &ids {
        if !brain
            .pending
            .get(*id)
            .map(|e| e.status.is_approved())
            .unwrap_or(false)
        {
            continue;
        }
        match crate::pending::apply_change(brain, *id) {
            Ok(()) => applied += 1,
            Err(e) => failures.push(json!({
                "pending_id": id, "phase": "apply", "error": format!("{}", e),
            })),
        }
    }

    let text = format!(
        "approve_pending: approved={} applied={} failures={}",
        approved,
        applied,
        failures.len()
    );
    Ok(t3_wrap(
        text,
        json!({
            "approved_count": approved,
            "applied_count": applied,
            "failures": failures,
            "approved_by": approver,
        }),
    ))
}

// =====================================================================
// lu4d_reject_pending — reject one or many Pending entries.
// =====================================================================
pub(crate) fn tool_reject_pending(
    brain: &mut Brain,
    args: Value,
) -> Result<Value, (i64, String)> {
    let rejector = args
        .get("rejected_by")
        .and_then(|v| v.as_str())
        .unwrap_or("john")
        .to_string();
    let reason = args.get("reason").and_then(|v| v.as_str()).map(String::from);
    let now = now_ms() as u64;

    let ids = resolve_ids(brain, &args, Some("Pending"));
    let mut rejected = 0u32;
    let mut failures: Vec<Value> = Vec::new();
    for id in &ids {
        match brain
            .pending
            .reject(*id, now, rejector.clone(), reason.clone())
        {
            Ok(()) => {
                // Port T.8: record a rejection fingerprint so Lu can
                // learn the shape of bad proposals.
                brain.record_rejection_fingerprint(*id, now);
                rejected += 1;
            }
            Err(e) => failures.push(json!({
                "pending_id": id, "error": format!("{}", e),
            })),
        }
    }
    let text = format!(
        "reject_pending: rejected={} failures={}",
        rejected,
        failures.len()
    );
    Ok(t3_wrap(
        text,
        json!({
            "rejected_count": rejected,
            "failures": failures,
            "rejected_by": rejector,
        }),
    ))
}

// =====================================================================
// lu4d_modify_pending — edit a Pending entry's confidence/weight/reason.
// =====================================================================
pub(crate) fn tool_modify_pending(
    brain: &mut Brain,
    args: Value,
) -> Result<Value, (i64, String)> {
    let id = args
        .get("pending_id")
        .and_then(|v| v.as_u64())
        .ok_or((-32602, "missing required arg: pending_id".into()))?;
    let m = args.get("modifications").cloned().unwrap_or_else(|| json!({}));
    let new_confidence = m.get("confidence").and_then(|v| v.as_f64()).map(|f| f as f32);
    let new_weight = m.get("weight").and_then(|v| v.as_f64()).map(|f| f as f32);
    let new_reason = m.get("reason").and_then(|v| v.as_str()).map(String::from);
    brain
        .pending
        .modify(id, new_confidence, new_weight, new_reason)
        .map_err(|e| (-32603, format!("{}", e)))?;
    let updated = brain
        .pending
        .get(id)
        .map(render_change)
        .unwrap_or(Value::Null);
    let text = format!("modify_pending: pending_id={} updated", id);
    Ok(t3_wrap(text, json!({ "updated": updated })))
}

// =====================================================================
// lu4d_pending_status — summary + thinking_indicator.
// =====================================================================
pub(crate) fn tool_pending_status(brain: &mut Brain) -> Result<Value, (i64, String)> {
    let pending_count = brain.pending.pending_count();
    let approved = brain.pending.approved_unapplied_count();
    let applied = brain.pending.applied_count();
    let rejected = brain.pending.rejected_count();
    let last_commit = brain.pending.last_commit_at_ms();
    let active_queries = brain.idle_state.active_queries();
    // "thinking" iff there are signals in flight (active queries OR
    // pending entries growing); otherwise "settled".
    let thinking_indicator = if pending_count > 0 || active_queries > 0 {
        "thinking"
    } else {
        "settled"
    };
    let text = format!(
        "pending_status: pending={} approved_unapplied={} applied_total={} rejected_total={} indicator={}",
        pending_count, approved, applied, rejected, thinking_indicator
    );
    Ok(t3_wrap(
        text,
        json!({
            "pending_count": pending_count,
            "approved_unapplied_count": approved,
            "applied_total": applied,
            "rejected_total": rejected,
            "last_commit_at_ms": last_commit,
            "thinking_indicator": thinking_indicator,
            "active_queries": active_queries,
            "proposals_filtered_total": brain.rejection_store.proposals_filtered_total(),
        }),
    ))
}

// =====================================================================
// Helper — resolve a list of pending ids from args.
//
// Supports three inputs (in priority order):
//   * `pending_ids: [u64]` — explicit list.
//   * `all: true` — every entry matching `default_status` (if Some)
//                   or every entry (if None).
//   * filter: `status`, `source_kind`, `min_confidence`, `max_age_ms`
//             — same shape as `lu4d_pending_changes`.
// =====================================================================
fn resolve_ids(
    brain: &Brain,
    args: &Value,
    default_status: Option<&str>,
) -> Vec<u64> {
    if let Some(arr) = args.get("pending_ids").and_then(|v| v.as_array()) {
        return arr
            .iter()
            .filter_map(|v| v.as_u64())
            .collect();
    }
    if args.get("all").and_then(|v| v.as_bool()).unwrap_or(false) {
        return brain
            .pending
            .iter()
            .filter(|(_, e)| {
                default_status
                    .map(|s| s.eq_ignore_ascii_case(e.status.tag()))
                    .unwrap_or(true)
            })
            .map(|(id, _)| *id)
            .collect();
    }
    let status_filter = args.get("status").and_then(|v| v.as_str()).or(default_status);
    let kind_filter = args.get("source_kind").and_then(|v| v.as_str());
    let min_conf = args
        .get("min_confidence")
        .and_then(|v| v.as_f64())
        .map(|f| f as f32);
    let max_age = args.get("max_age_ms").and_then(|v| v.as_u64());
    let now = now_ms() as u64;
    let filtered = collect_filtered(
        &brain.pending,
        status_filter,
        kind_filter,
        min_conf,
        max_age,
        now,
    );
    filtered.iter().map(|e| e.pending_id).collect()
}
