// =====================================================================
// Port LB — Left Brain session memory MCP tools.
//
//   1. `lu4d_session_memory_status` — counts by session_id and kind,
//      total entries, active session, archived sessions.
//   2. `lu4d_session_memory_inspect` — list entries; filters by
//      session_id, kind, limit.
//   3. `lu4d_session_memory_propose` — convert an entry to a
//      PendingChange (T.3 path). Routes through
//      `crate::lb_session::propose_to_pending`.
//   4. `lu4d_session_memory_archive` — mark a session as archived
//      (immutable from then on).
//
// The store itself is owned by the Brain (`brain.lb_session`),
// persisted via the v3 section `SectionIdV3::SessionMemory` (id 11)
// added in this port. Companion to `crate::lb_session`.
// =====================================================================

use crate::brain::Brain;
use crate::lb_session::{
    propose_to_pending, PromoteOptions, SessionEntry, SessionEntryKind,
};
use serde_json::{json, Value};

fn lb_wrap(text: String, structured: Value) -> Value {
    json!({
        "content": [{"type": "text", "text": text}],
        "structuredContent": structured,
    })
}

fn parse_kind(s: &str) -> Option<SessionEntryKind> {
    SessionEntryKind::all()
        .iter()
        .find(|k| k.tag() == s)
        .cloned()
}

fn entry_to_json(e: &SessionEntry) -> Value {
    json!({
        "entry_id": e.entry_id,
        "session_id": e.session_id,
        "ts_ms": e.ts_ms,
        "kind": e.kind.tag(),
        "subject": e.subject,
        "content": e.content,
        "confidence_at_creation": e.confidence_at_creation,
        "promoted_at_ms": e.promoted_at_ms,
        "promoted_pending_id": e.promoted_pending_id,
    })
}

/// `lu4d_session_memory_status` — counts grouped by session_id and
/// kind, total entries, active session.
pub(crate) fn tool_session_memory_status(
    brain: &Brain,
    _args: Value,
) -> Result<Value, (i64, String)> {
    let store = &brain.lb_session;
    let counts_by_session = store.counts_by_session();
    let counts_by_kind = store.counts_by_kind();
    let archived = store.archived_sessions();
    let total = store.total_count();
    let active = store.active_session();
    let text = format!(
        "lb_session: total={} active_session={:?} sessions={} archived={} kinds={}",
        total,
        active,
        counts_by_session.len(),
        archived.len(),
        counts_by_kind.len(),
    );
    Ok(lb_wrap(
        text,
        json!({
            "total": total,
            "active_session": active,
            "counts_by_session": counts_by_session
                .into_iter()
                .map(|(s, c)| (s.to_string(), c))
                .collect::<std::collections::BTreeMap<_, _>>(),
            "counts_by_kind": counts_by_kind,
            "archived_sessions": archived,
            "next_entry_id": store.next_id(),
        }),
    ))
}

/// `lu4d_session_memory_inspect` — list entries matching
/// session_id/kind filters.
pub(crate) fn tool_session_memory_inspect(
    brain: &Brain,
    args: Value,
) -> Result<Value, (i64, String)> {
    let session_id = args.get("session_id").and_then(|v| v.as_u64());
    let kind = match args.get("kind").and_then(|v| v.as_str()) {
        Some(s) => match parse_kind(s) {
            Some(k) => Some(k),
            None => return Err((-32602, format!("unknown kind tag `{}`", s))),
        },
        None => None,
    };
    let limit = args
        .get("limit")
        .and_then(|v| v.as_u64())
        .unwrap_or(50)
        .min(1000) as usize;
    let entries = brain
        .lb_session
        .filter(session_id, kind.as_ref(), limit);
    let total = brain.lb_session.total_count();
    let json_entries: Vec<Value> = entries.iter().map(entry_to_json).collect();
    let text = format!(
        "lb_session_inspect: returned={} of total={} (session_id={:?} kind={:?} limit={})",
        json_entries.len(),
        total,
        session_id,
        args.get("kind").and_then(|v| v.as_str()),
        limit,
    );
    Ok(lb_wrap(
        text,
        json!({
            "entries": json_entries,
            "total": total,
            "returned": json_entries.len(),
        }),
    ))
}

/// `lu4d_session_memory_propose` — convert a session entry to a T.3
/// `PendingChange::AddBond`. The propose flow's hard rule: the
/// canonical brain is NOT mutated until a human approves the pending
/// change via `lu4d_approve_pending`.
pub(crate) fn tool_session_memory_propose(
    brain: &mut Brain,
    args: Value,
) -> Result<Value, (i64, String)> {
    let entry_id = args
        .get("entry_id")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| (-32602, "missing required arg `entry_id`".to_string()))?;
    let bond_type_str = args
        .get("bond_type")
        .and_then(|v| v.as_str())
        .unwrap_or("RelatedTo");
    let bond_type = crate::brain::parse_bond_type(bond_type_str)
        .map_err(|e| (-32602, format!("bad bond_type: {}", e)))?;
    let weight = args
        .get("weight")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.6) as f32;
    let subject_text = args
        .get("subject_text")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let object_text = args
        .get("object_text")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let ts_ms = crate::brain::now_ms() as u64;
    let pending_id = propose_to_pending(
        brain,
        entry_id,
        PromoteOptions {
            subject_text,
            object_text,
            bond_type,
            weight,
        },
        ts_ms,
    )
    .map_err(|e| (-32603, format!("propose_to_pending: {}", e)))?;
    let pc = brain.pending.get(pending_id);
    let summary = pc.map(|c| c.change_kind.summary()).unwrap_or_default();
    let text = format!(
        "lb_session_propose: entry_id={} -> pending_id={} ({})",
        entry_id, pending_id, summary,
    );
    Ok(lb_wrap(
        text,
        json!({
            "entry_id": entry_id,
            "pending_id": pending_id,
            "summary": summary,
            "note": "canonical brain unchanged until lu4d_approve_pending fires",
        }),
    ))
}

/// `lu4d_session_memory_archive` — mark a session immutable.
pub(crate) fn tool_session_memory_archive(
    brain: &mut Brain,
    args: Value,
) -> Result<Value, (i64, String)> {
    let session_id = args
        .get("session_id")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| (-32602, "missing required arg `session_id`".to_string()))?;
    brain.lb_session.archive_session(session_id);
    let archived_count = brain.lb_session.archived_sessions().len();
    let text = format!(
        "lb_session_archive: session_id={} archived (total_archived={})",
        session_id, archived_count,
    );
    Ok(lb_wrap(
        text,
        json!({
            "session_id": session_id,
            "archived": true,
            "total_archived": archived_count,
        }),
    ))
}
