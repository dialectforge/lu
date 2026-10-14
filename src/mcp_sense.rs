//! Port Sense — MCP tool wiring.
//!
//! Exposes `lu4d_sense_status`, which reports the sense-cluster
//! decomposition and per-cluster scores for a polysemous seed under
//! a caller-supplied context window. Useful for debugging and as a
//! substrate for future Port U.6 (curiosity panel sense visualization).
//!
//! Tool input shape (single JSON object):
//!  - `query` (string, required): the seed term to inspect. Resolved
//!    via the same TF-IDF-weighted seed selector that `lu4d_ask` uses,
//!    so the canonical "bank" / "spring" / "bark" anchors of the live
//!    encrypted brain reach the same Seed they would on a real ask.
//!  - `context` (array of strings, optional): the context window.
//!    Each term is resolved to a Seed; unknown terms are dropped.
//!
//! Output (structured):
//!  - `seed`, `is_polysemous`, `cluster_count`, `clusters[]`
//!    `{label, member_count, members[], typed_bond_count, score, is_winner}`,
//!    `winner_elected`, `context_window[]`, `loser_gain`,
//!    `sense_disambiguation_enabled`.

use crate::brain::Brain;
use crate::seed::Seed;
use serde_json::{json, Value};

/// Resolve a free-form term to the same Seed the ask path would use.
/// `select_seeds_for_query` returns up to 3 TF-IDF-weighted seeds for
/// the term — we take the highest-weight one. Falls back to a
/// keyword-search hit if the TF-IDF path returns nothing. Returns
/// `None` when the brain has never seen the term.
fn resolve_term(brain: &Brain, term: &str) -> Option<Seed> {
    let trimmed = term.trim();
    if trimmed.is_empty() {
        return None;
    }
    let weighted = brain.network.select_seeds_for_query(trimmed);
    if let Some((s, _)) = weighted.into_iter().next() {
        return Some(s);
    }
    let kw = brain.network.keyword_search(trimmed);
    kw.into_iter().next().map(|(s, _)| s)
}

/// Resolve a list of term strings to Seeds. Unknown terms are
/// dropped quietly — the MCP caller is exploratory.
fn resolve_terms(brain: &Brain, terms: &[String]) -> Vec<Seed> {
    terms.iter().filter_map(|t| resolve_term(brain, t)).collect()
}

/// Build the structured payload for `lu4d_sense_status`. Pulled out
/// as a free fn so unit tests can drive it without the dispatcher.
///
/// BR.5 audit P1 #4 internal notes: READ-ONLY. The signature is
/// deliberately `&Brain` (not `&mut Brain`) so the type system
/// enforces the no-mutation invariant. A future maintainer who
/// tries to add a mutation to a `_status` tool will be stopped at
/// compile time. Use `lu4d_sense_set` (`tool_sense_set` in
/// mcp_runtime_setters.rs) for any actual mutation.
pub fn sense_status_payload(brain: &Brain, args: Value) -> Result<Value, String> {
    let query = args
        .get("query")
        .and_then(|v| v.as_str())
        .map(|s| s.trim().to_string())
        .unwrap_or_default();
    if query.is_empty() {
        return Err("missing required argument: query (the seed term)".to_string());
    }

    let enabled = brain.sense_config.disambiguation_enabled;
    let loser_gain = brain.sense_config.loser_gain;

    let seed = match resolve_term(brain, &query) {
        Some(s) => s,
        None => {
            return Ok(json!({
                "seed": query,
                "is_polysemous": false,
                "cluster_count": 0,
                "clusters": [],
                "winner_elected": false,
                "context_window": [],
                "sense_disambiguation_enabled": enabled,
                "loser_gain": loser_gain,
                "reason": "seed not found in brain",
            }));
        }
    };

    let provided_context: Vec<String> = args
        .get("context")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|x| x.as_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default();

    let context_seeds: Vec<Seed> = resolve_terms(brain, &provided_context);

    let context_terms: Vec<String> = context_seeds
        .iter()
        .filter_map(|s| brain.network.get_node(s).map(|n| n.content().to_string()))
        .collect();

    let result = brain.network.score_sense_clusters(&seed, &context_seeds);
    let (clusters_json, cluster_count, is_polysemous, winner_elected) = match result {
        None => (Vec::<Value>::new(), 0usize, false, false),
        Some(scores) => {
            let cluster_count = scores.len();
            let any_winner = scores.iter().any(|c| c.is_winner);
            let mut arr: Vec<Value> = Vec::with_capacity(cluster_count);
            for c in &scores {
                let members_str: Vec<String> = c
                    .members
                    .iter()
                    .filter_map(|m| brain.network.get_node(m).map(|n| n.content().to_string()))
                    .collect();
                arr.push(json!({
                    "label": c.label,
                    "member_count": c.members.len(),
                    "members": members_str,
                    "typed_bond_count": c.typed_bond_count,
                    "score": c.score,
                    "is_winner": c.is_winner,
                }));
            }
            (arr, cluster_count, true, any_winner)
        }
    };

    Ok(json!({
        "seed": query,
        "is_polysemous": is_polysemous,
        "cluster_count": cluster_count,
        "clusters": clusters_json,
        "winner_elected": winner_elected,
        "context_window": context_terms,
        "sense_disambiguation_enabled": enabled,
        "loser_gain": loser_gain,
    }))
}

/// Wire-level MCP tool entry. Wraps `sense_status_payload` in the
/// `(content, structuredContent)` envelope.
///
/// BR.5 audit P1 #4: READ-ONLY -- accepts `&Brain` so any future
/// attempt to mutate state from a `_status` MCP tool is blocked at
/// compile time.
pub(crate) fn tool_sense_status(brain: &Brain, args: Value) -> Result<Value, (i64, String)> {
    let payload = sense_status_payload(brain, args).map_err(|e| (-32602, e))?;
    let seed = payload["seed"].as_str().unwrap_or("?").to_string();
    let cluster_count = payload["cluster_count"].as_u64().unwrap_or(0);
    let is_poly = payload["is_polysemous"].as_bool().unwrap_or(false);
    let winner = payload["winner_elected"].as_bool().unwrap_or(false);
    let enabled = payload["sense_disambiguation_enabled"].as_bool().unwrap_or(true);
    let ctx_n = payload["context_window"]
        .as_array()
        .map(|a| a.len())
        .unwrap_or(0);
    let text = format!(
        "Sense status for '{}':\n  enabled: {}\n  is_polysemous: {}\n  cluster_count: {}\n  winner_elected: {}\n  context_window_size: {}",
        seed, enabled, is_poly, cluster_count, winner, ctx_n,
    );
    Ok(crate::mcp_envelope::wrap_content(text, payload))
}

#[cfg(test)]
mod tests {
    //! Anti-hallucination grep guard. STRUCTURAL: this test fails
    //! at compile time of the test binary if mutating-bond-API
    //! identifiers ever appear in the Sense source. Mirrors the
    //! Cat.3a / Cat.3b read-side guards.

    #[test]
    fn sense_module_never_mutates_bond_storage() {
        // Split tokens at compile time so the literal substrings
        // don't appear in this source file (which would self-trigger
        // the grep). Each `concat!` produces a single static &str.
        const FORBIDDEN: &[&str] = &[
            concat!("add", "_bond"),
            concat!("remove", "_bond"),
            concat!("get", "_bond_", "mut"),
            concat!("set", "_weight"),
            concat!("set", "_historical", "_weight"),
            concat!("add", "_keyframe"),
            concat!("set", "_bond_", "type"),
        ];
        let sense_rs = include_str!("sense.rs");
        let mcp_sense_rs = include_str!("mcp_sense.rs");
        for f in FORBIDDEN {
            assert!(
                !sense_rs.contains(f),
                "sense.rs must NEVER reference mutating bond API '{}': anti-hallucination invariant",
                f
            );
            assert!(
                !mcp_sense_rs.contains(f),
                "mcp_sense.rs must NEVER reference mutating bond API '{}': anti-hallucination invariant",
                f
            );
        }
    }

    #[test]
    fn sense_status_payload_returns_seed_not_found_for_unknown_term() {
        use crate::brain::Brain;
        use serde_json::json;
        let mut brain = Brain::new();
        let payload = super::sense_status_payload(
            &mut brain,
            json!({"query": "this-seed-does-not-exist-anywhere-zzqq"}),
        )
        .expect("payload must be Ok even on unknown seed");
        assert_eq!(payload["is_polysemous"], json!(false));
        assert_eq!(payload["cluster_count"], json!(0));
        assert_eq!(payload["reason"], json!("seed not found in brain"));
    }

    #[test]
    fn sense_status_payload_rejects_empty_query() {
        use crate::brain::Brain;
        use serde_json::json;
        let mut brain = Brain::new();
        let res = super::sense_status_payload(&mut brain, json!({"query": ""}));
        assert!(res.is_err(), "empty query must error");
    }

    #[test]
    fn sense_status_payload_default_runtime_knobs_visible() {
        use crate::brain::Brain;
        use serde_json::json;
        let mut brain = Brain::new();
        let payload = super::sense_status_payload(&mut brain, json!({"query": "x"})).unwrap();
        assert_eq!(payload["sense_disambiguation_enabled"], json!(true));
        // Default loser_gain is 0.25 per spec.
        let g = payload["loser_gain"].as_f64().unwrap();
        assert!((g - 0.25).abs() < 1e-6, "loser_gain default must be 0.25, got {}", g);
    }
}
