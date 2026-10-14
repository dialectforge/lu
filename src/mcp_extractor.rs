//! MCP tool `lu4d_learn_text` — accepts free-form article text and
//! queues `PendingChangeKind::AddBond` entries via
//! `Brain::pending.push` for each extractor-emitted bond.
//!
//! STRUCTURAL ANTI-MUTATION INVARIANT: the only allowed bond-creation
//! path inside this function is `brain.pending.push`. Direct
//! `brain.teach*` / `brain.network.add_bond` calls are forbidden so
//! the T.3 approve / reject / revoke gates remain the single mutation
//! choke point.
//!
//! Knowledge-aware filter: a candidate bond survives if EITHER the
//! subject OR the object normalises to a known existing node (per
//! `network.keyword_index_bucket`). If both are unknown, the bond is
//! still kept when its shape is small (< 50 chars) so the graph can
//! grow. Pure NP-leading-verb fragments with both endpoints unknown
//! are dropped — the same shape-floor `_truncate_to_np` enforces on
//! the LM tail.

use serde_json::{json, Value};

use crate::brain::Brain;
use crate::extractor::{extract_from_article, Bond};
use crate::pending::{PendingChange, PendingChangeKind, PendingStatus};
use crate::source::SourceAttribution;

fn arg_str<'a>(args: &'a Value, key: &str) -> Result<&'a str, (i64, String)> {
    match args.get(key) {
        Some(Value::String(s)) => Ok(s),
        Some(other) => Err((-32602, format!("arg `{}` must be a string, got {:?}", key, other))),
        None => Err((-32602, format!("missing arg `{}`", key))),
    }
}

fn opt_str<'a>(args: &'a Value, key: &str) -> Option<&'a str> {
    match args.get(key) {
        Some(Value::String(s)) if !s.is_empty() => Some(s.as_str()),
        _ => None,
    }
}

fn normalize_token(s: &str) -> String {
    s.to_lowercase()
        .split_whitespace()
        .next()
        .unwrap_or("")
        .trim_matches(|c: char| !c.is_alphanumeric())
        .to_string()
}

fn known_in_index(brain: &Brain, label: &str) -> bool {
    let tok = normalize_token(label);
    if tok.is_empty() {
        return false;
    }
    match brain.network.keyword_index_bucket(&tok) {
        Some(b) => !b.is_empty(),
        None => false,
    }
}

fn np_leading_verb(s: &str) -> bool {
    crate::extractor::patterns::np_leading_verb_re().is_match(s)
}

/// Knowledge-aware filter. Returns true when the bond should be kept.
fn keep_bond(brain: &Brain, b: &Bond) -> bool {
    let subj_known = known_in_index(brain, &b.subject);
    let obj_known = known_in_index(brain, &b.object);
    if subj_known || obj_known {
        return true;
    }
    // Both unknown. Keep small clean shapes so the graph grows;
    // drop suspicious fragments (verb-led NP, oversize).
    if b.object.chars().count() >= 50 {
        return false;
    }
    if np_leading_verb(&b.object) {
        return false;
    }
    if np_leading_verb(&b.subject) {
        return false;
    }
    true
}

pub fn tool_learn_text(brain: &mut Brain, args: Value) -> Result<Value, (i64, String)> {
    let text = arg_str(&args, "text")?.to_string();
    let title = opt_str(&args, "title").map(|s| s.to_string());
    let source_url = opt_str(&args, "source_url").map(|s| s.to_string());
    let corpus = arg_str(&args, "corpus")?.to_string();
    if corpus.trim().is_empty() {
        return Err((-32602, "`corpus` must be a non-empty string".to_string()));
    }

    let candidates = extract_from_article(&text, title.as_deref(), source_url.as_deref());
    let total = candidates.len();

    let mut bonds_emitted: usize = 0;
    let mut bonds_dropped_knowledge: usize = 0;
    let mut bonds_dropped_validation: usize = 0;
    let mut samples: Vec<Value> = Vec::new();
    let mut layer_breakdown: std::collections::BTreeMap<String, usize> =
        std::collections::BTreeMap::new();
    let mut pending_ids: Vec<u64> = Vec::new();
    let now = crate::brain::now_ms() as u64;

    for bond in candidates {
        // Validation: extractor should already produce non-empty
        // labels, but guard once more so a malformed match never
        // pushes a junk pending entry.
        if bond.subject.trim().is_empty() || bond.object.trim().is_empty() {
            bonds_dropped_validation += 1;
            continue;
        }
        if !keep_bond(brain, &bond) {
            bonds_dropped_knowledge += 1;
            continue;
        }
        let proposed_by = if let Some(url) = source_url.clone() {
            SourceAttribution::training_with_url(
                corpus.clone(),
                Some(brain.ingest_run_id.clone()),
                Some(url),
            )
        } else {
            SourceAttribution::training(corpus.clone(), Some(brain.ingest_run_id.clone()))
        };
        let reason = format!(
            "learn_text layer={} subject_origin={:?} freq={:?}",
            bond.layer, bond.subject_origin, bond.frequency
        );
        let change = PendingChange {
            pending_id: 0,
            proposed_at_ms: now,
            proposed_by,
            change_kind: PendingChangeKind::AddBond {
                subject_text: bond.subject.clone(),
                object_text: bond.object.clone(),
                bond_type: bond.relation,
                weight: bond.weight,
                negated: false,
            },
            confidence: bond.confidence.score(),
            reason,
            status: PendingStatus::Pending,
        };
        let pid = brain
            .pending
            .push(change)
            .map_err(|e| (-32603, e.to_string()))?;
        pending_ids.push(pid);
        bonds_emitted += 1;
        *layer_breakdown.entry(bond.layer.to_string()).or_insert(0) += 1;
        if samples.len() < 5 {
            samples.push(json!({
                "subject": bond.subject,
                "relation": format!("{:?}", bond.relation),
                "object": bond.object,
                "kind": bond.kind,
                "weight": bond.weight,
                "confidence": bond.confidence.as_str(),
                "layer": bond.layer,
                "subject_origin": bond.subject_origin,
                "frequency": bond.frequency,
                "pending_id": pid,
            }));
        }
    }

    let text_msg = format!(
        "lu4d_learn_text: corpus={} title={} bonds_emitted={} dropped_knowledge={} dropped_validation={} total_candidates={}",
        corpus,
        title.as_deref().unwrap_or("(none)"),
        bonds_emitted,
        bonds_dropped_knowledge,
        bonds_dropped_validation,
        total,
    );
    let structured = json!({
        "corpus": corpus,
        "title": title,
        "source_url": source_url,
        "bonds_emitted": bonds_emitted,
        "bonds_dropped_knowledge_filter": bonds_dropped_knowledge,
        "bonds_dropped_validation": bonds_dropped_validation,
        "total_candidates": total,
        "layer_breakdown": layer_breakdown,
        "pending_ids": pending_ids,
        "sample": samples,
        "routed_via": "T.3 PendingLog (brain.pending.push)",
    });
    Ok(json!({
        "content": [{"type": "text", "text": text_msg}],
        "structuredContent": structured,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn permissive_brain() -> Brain {
        Brain::new()
    }

    #[test]
    fn learn_text_queues_pending_for_real_article() {
        let mut b = permissive_brain();
        let pending_before = b.pending.total_count();
        let bond_before = b.network.bond_count();
        let resp = tool_learn_text(
            &mut b,
            json!({
                "text": "Rust is a systems programming language. Rust is fast.\n\nSee also\n\nGo\nC++\n",
                "title": "Rust",
                "corpus": "unit_test/learn_text",
            }),
        )
        .unwrap();
        let s = &resp["structuredContent"];
        let emitted = s["bonds_emitted"].as_u64().unwrap();
        assert!(emitted >= 1, "expected bonds, got {:?}", resp);
        // Must NOT mutate bond_count directly.
        assert_eq!(b.network.bond_count(), bond_before);
        assert!(b.pending.total_count() > pending_before);
    }

    #[test]
    fn learn_text_requires_corpus() {
        let mut b = permissive_brain();
        let res = tool_learn_text(
            &mut b,
            json!({
                "text": "Some text.",
                "title": "Title",
                "corpus": "",
            }),
        );
        assert!(res.is_err());
    }
}
