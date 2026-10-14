//! Port Cat.3a — source-trust MCP tools.
//!
//!   1. `lu4d_trust_status`            — read-only view of the
//!                                       current TrustTable.
//!   2. `lu4d_trust_set`               — admin-only: add/update an
//!                                       entry. Validates 0.0..=2.0.
//!   3. `lu4d_trust_remove`            — admin-only: remove an
//!                                       entry; default still applies.
//!   4. `lu4d_trust_reset_to_defaults` — admin-only: restore the
//!                                       C2-aligned default table.
//!
//! Mutations save the JSON sidecar at `<brain>.trust.json` (NOT a v3
//! schema section). Existing bonds are NEVER mutated; trust applies
//! only at bond formation in the research path (see
//! `crate::research::loop_::apply_proposal`).

use crate::brain::Brain;
use crate::trust::{TrustEntry, MAX_TRUST_MULTIPLIER, MIN_TRUST_MULTIPLIER};
use serde_json::{json, Value};

fn cat3a_wrap(text: String, structured: Value) -> Value {
    json!({
        "content": [{"type": "text", "text": text}],
        "structuredContent": structured,
    })
}

fn entry_to_json(e: &TrustEntry) -> Value {
    json!({
        "domain_pattern": e.domain_pattern,
        "multiplier": e.multiplier,
        "note": e.note,
    })
}

/// `lu4d_trust_status` — read-only listing of the current trust
/// table. Returns entries (in insertion order), the
/// default_multiplier, and the master `learning_enabled` switch.
pub(crate) fn tool_trust_status(
    brain: &mut Brain,
    _args: Value,
) -> Result<Value, (i64, String)> {
    let entries: Vec<Value> = brain
        .trust_table
        .entries
        .iter()
        .map(entry_to_json)
        .collect();
    // Port T-Ingest.1: expose per-corpus multipliers alongside the
    // existing url-host entries. Sort by key for deterministic output
    // so test assertions are stable.
    let mut corpus_pairs: Vec<(&String, &f32)> =
        brain.trust_table.corpus_multipliers.iter().collect();
    corpus_pairs.sort_by(|a, b| a.0.cmp(b.0));
    let corpus_map: serde_json::Map<String, Value> = corpus_pairs
        .iter()
        .map(|(k, v)| ((*k).clone(), serde_json::json!(*v)))
        .collect();
    let text = format!(
        "trust_status: {} url entries, {} corpus entries, default={:.3}, learning_enabled={}",
        entries.len(),
        brain.trust_table.corpus_multipliers.len(),
        brain.trust_table.default_multiplier,
        brain.trust_table.learning_enabled,
    );
    Ok(cat3a_wrap(
        text,
        json!({
            "entries": entries,
            "corpus_multipliers": corpus_map,
            "default_multiplier": brain.trust_table.default_multiplier,
            "learning_enabled": brain.trust_table.learning_enabled,
            "bounds": {
                "min_multiplier": MIN_TRUST_MULTIPLIER,
                "max_multiplier": MAX_TRUST_MULTIPLIER,
            }
        }),
    ))
}

/// `lu4d_trust_set` — add or update a domain → multiplier entry.
/// Required: `domain` (string), `multiplier` (number 0.0..=2.0).
/// Optional: `note` (string).
pub(crate) fn tool_trust_set(
    brain: &mut Brain,
    args: Value,
) -> Result<Value, (i64, String)> {
    let domain = args
        .get("domain")
        .and_then(|v| v.as_str())
        .ok_or((-32602, "missing required 'domain' string".to_string()))?
        .to_string();
    if domain.trim().is_empty() {
        return Err((-32602, "'domain' must be a non-empty string".to_string()));
    }
    let multiplier = args
        .get("multiplier")
        .and_then(|v| v.as_f64())
        .ok_or((-32602, "missing required 'multiplier' number".to_string()))?
        as f32;
    let note = args
        .get("note")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let prev = brain
        .trust_table
        .set(&domain, multiplier, note.clone())
        .map_err(|e| (-32602, e))?;
    let text = match prev {
        Some(p) => format!(
            "trust_set: {} multiplier {:.3} -> {:.3}",
            domain, p, multiplier
        ),
        None => format!(
            "trust_set: added {} -> {:.3}",
            domain, multiplier
        ),
    };
    Ok(cat3a_wrap(
        text,
        json!({
            "domain": domain,
            "new_multiplier": multiplier,
            "previous_multiplier": prev,
            "note": note,
        }),
    ))
}

/// `lu4d_trust_remove` — remove an entry. Returns whether it was
/// removed; the default multiplier still applies for that domain.
pub(crate) fn tool_trust_remove(
    brain: &mut Brain,
    args: Value,
) -> Result<Value, (i64, String)> {
    let domain = args
        .get("domain")
        .and_then(|v| v.as_str())
        .ok_or((-32602, "missing required 'domain' string".to_string()))?
        .to_string();
    let removed = brain.trust_table.remove(&domain);
    if removed {
    }
    let text = if removed {
        format!("trust_remove: removed {}", domain)
    } else {
        format!("trust_remove: no entry for {} (no-op)", domain)
    };
    Ok(cat3a_wrap(
        text,
        json!({
            "domain": domain,
            "removed": removed,
            "default_multiplier_now_applies": removed,
        }),
    ))
}

/// `lu4d_trust_reset_to_defaults` — restore the C2-aligned default
/// table. Idempotent.
pub(crate) fn tool_trust_reset_to_defaults(
    brain: &mut Brain,
    args: Value,
) -> Result<Value, (i64, String)> {
    // Port T-Ingest.1: optional seed_curriculum_corpora flag pre-populates
    // the corpus_multipliers map with the locked curriculum hierarchy.
    // Defaults to false so existing operator behavior is preserved.
    let seed_curriculum = args
        .get("seed_curriculum_corpora")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    brain.trust_table.reset_to_defaults();
    if seed_curriculum {
        brain.trust_table.seed_curriculum_corpora();
    }
    let text = format!(
        "trust_reset_to_defaults: {} url entries, {} corpus entries, default={:.3} (seed_curriculum_corpora={})",
        brain.trust_table.entries.len(),
        brain.trust_table.corpus_multipliers.len(),
        brain.trust_table.default_multiplier,
        seed_curriculum,
    );
    let entries: Vec<Value> = brain.trust_table.entries.iter().map(entry_to_json).collect();
    let mut corpus_pairs: Vec<(&String, &f32)> =
        brain.trust_table.corpus_multipliers.iter().collect();
    corpus_pairs.sort_by(|a, b| a.0.cmp(b.0));
    let corpus_map: serde_json::Map<String, Value> = corpus_pairs
        .iter()
        .map(|(k, v)| ((*k).clone(), serde_json::json!(*v)))
        .collect();
    Ok(cat3a_wrap(
        text,
        json!({
            "entries": entries,
            "corpus_multipliers": corpus_map,
            "default_multiplier": brain.trust_table.default_multiplier,
            "learning_enabled": brain.trust_table.learning_enabled,
            "seeded_curriculum": seed_curriculum,
        }),
    ))
}

/// Port T-Ingest.1: `lu4d_trust_set_corpus` — add or update a
/// per-corpus multiplier. Required: `corpus` (string), `multiplier`
/// (number 0.0..=2.0). Mutations affect ONLY new bond formation;
/// existing bonds are not retroactively reweighted.
pub(crate) fn tool_trust_set_corpus(
    brain: &mut Brain,
    args: Value,
) -> Result<Value, (i64, String)> {
    let corpus = args
        .get("corpus")
        .and_then(|v| v.as_str())
        .ok_or((-32602, "missing required 'corpus' string".to_string()))?
        .to_string();
    if corpus.trim().is_empty() {
        return Err((-32602, "'corpus' must be a non-empty string".to_string()));
    }
    let multiplier = args
        .get("multiplier")
        .and_then(|v| v.as_f64())
        .ok_or((-32602, "missing required 'multiplier' number".to_string()))?
        as f32;
    let prev = brain
        .trust_table
        .set_corpus(&corpus, multiplier)
        .map_err(|e| (-32602, e))?;
    let text = match prev {
        Some(p) => format!(
            "trust_set_corpus: {} multiplier {:.3} -> {:.3}",
            corpus, p, multiplier
        ),
        None => format!(
            "trust_set_corpus: added {} -> {:.3}",
            corpus, multiplier
        ),
    };
    Ok(cat3a_wrap(
        text,
        json!({
            "corpus": corpus,
            "new_multiplier": multiplier,
            "previous_multiplier": prev,
        }),
    ))
}

/// Port T-Ingest.1: `lu4d_trust_remove_corpus` — remove a per-corpus
/// entry. Returns whether it existed; the `default_multiplier` then
/// applies for that corpus on subsequent teach calls.
pub(crate) fn tool_trust_remove_corpus(
    brain: &mut Brain,
    args: Value,
) -> Result<Value, (i64, String)> {
    let corpus = args
        .get("corpus")
        .and_then(|v| v.as_str())
        .ok_or((-32602, "missing required 'corpus' string".to_string()))?
        .to_string();
    let removed = brain.trust_table.remove_corpus(&corpus);
    let text = if removed {
        format!("trust_remove_corpus: removed {}", corpus)
    } else {
        format!("trust_remove_corpus: no entry for {} (no-op)", corpus)
    };
    Ok(cat3a_wrap(
        text,
        json!({
            "corpus": corpus,
            "removed": removed,
            "default_multiplier_now_applies": removed,
        }),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fresh_brain() -> Brain {
        Brain::new()
    }

    #[test]
    fn tool_trust_status_returns_default_table() {
        let mut b = fresh_brain();
        let v = tool_trust_status(&mut b, json!({})).unwrap();
        let entries = v["structuredContent"]["entries"].as_array().unwrap();
        assert_eq!(entries.len(), 3);
        let domains: Vec<&str> = entries
            .iter()
            .map(|e| e["domain_pattern"].as_str().unwrap())
            .collect();
        assert!(domains.contains(&"grokipedia.com"));
        assert!(domains.contains(&"en.wikipedia.org"));
        assert!(domains.contains(&"duckduckgo.com"));
        assert!(
            (v["structuredContent"]["default_multiplier"]
                .as_f64()
                .unwrap()
                - 0.4)
                .abs()
                < 1e-6
        );
    }

    #[test]
    fn tool_trust_set_updates_existing_entry() {
        let mut b = fresh_brain();
        let v = tool_trust_set(
            &mut b,
            json!({"domain":"en.wikipedia.org","multiplier":0.3,"note":"demoting"}),
        )
        .unwrap();
        let prev = v["structuredContent"]["previous_multiplier"]
            .as_f64()
            .unwrap();
        assert!((prev - 0.7).abs() < 1e-6);
        let m = b.trust_table.lookup("https://en.wikipedia.org/x");
        assert!((m - 0.3).abs() < 1e-6);
    }

    #[test]
    fn tool_trust_set_rejects_out_of_bounds_multiplier() {
        let mut b = fresh_brain();
        let r = tool_trust_set(
            &mut b,
            json!({"domain":"x.example","multiplier":5.0}),
        );
        assert!(r.is_err());
        let r2 = tool_trust_set(
            &mut b,
            json!({"domain":"x.example","multiplier":-1.0}),
        );
        assert!(r2.is_err());
    }

    #[test]
    fn tool_trust_remove_then_reset_round_trip() {
        let mut b = fresh_brain();
        // Remove wikipedia, confirm default takes over.
        let _ =
            tool_trust_remove(&mut b, json!({"domain":"en.wikipedia.org"})).unwrap();
        let m = b.trust_table.lookup("https://en.wikipedia.org/x");
        assert!((m - 0.4).abs() < 1e-6);
        // Reset.
        let _ = tool_trust_reset_to_defaults(&mut b, json!({})).unwrap();
        let m2 = b.trust_table.lookup("https://en.wikipedia.org/x");
        assert!((m2 - 0.7).abs() < 1e-6);
    }

    #[test]
    fn tool_trust_set_missing_required_args_returns_error() {
        let mut b = fresh_brain();
        // Missing multiplier.
        let r = tool_trust_set(&mut b, json!({"domain":"x.example"}));
        assert!(r.is_err());
        // Missing domain.
        let r = tool_trust_set(&mut b, json!({"multiplier":0.5}));
        assert!(r.is_err());
    }
}
