//! Top-level extractor dispatch — runs all 8 layers, dedups bonds
//! on `(subject.lower(), relation, object.lower())`, returns combined
//! list.
//!
//! Mirrors `extract_bonds_from_wikipedia_article` in
//! `smart_extractor.py` plus the union+dedup wiring in
//! `ingest_curriculum_full.ingest_wikipedia` that folds in the LM
//! multi-sentence pass.

use super::layers::{
    l1_see_also, l2_section_headings, l2b_category_tail, l3_lead_patterns,
    l4_body_comentions, l5_numerical_attrs, lm_multi_sentence,
};
use super::types::Bond;

pub const L4_MAX_EMIT: usize = 80;

pub fn extract_from_article(
    text: &str,
    title: Option<&str>,
    source_url: Option<&str>,
) -> Vec<Bond> {
    let title = match title {
        Some(t) if !t.trim().is_empty() => t.trim(),
        _ => return Vec::new(),
    };
    if text.trim().is_empty() {
        return Vec::new();
    }
    let mut all: Vec<Bond> = Vec::new();
    all.extend(l1_see_also::extract(title, text, source_url));
    all.extend(l2_section_headings::extract(title, text, source_url));
    all.extend(l2b_category_tail::extract(title, text, source_url));
    all.extend(l3_lead_patterns::extract(title, text, source_url));
    all.extend(l4_body_comentions::extract(title, text, source_url, L4_MAX_EMIT));
    all.extend(l5_numerical_attrs::extract(title, text, source_url));
    all.extend(lm_multi_sentence::extract(title, text, source_url));

    // OP-EXTRACT-WIDEN: collapse same-(subject, object) pairs to the single
    // richest relation. The brain dedups by (from, to) PAIR ONLY, keeping the
    // first-seen bond_type; without this, generic RelatedTo (L1/L4) and IsA
    // (L2b) shadow the typed L3/L5 relations internal notes. Keep, per pair,
    // the bond whose relation has the highest `relation_priority`; ties keep
    // the earliest-seen. The set of (subject, object) PAIRS is unchanged --
    // only the surviving relation per pair can upgrade generic -> typed.
    use super::types::relation_priority;
    let mut best_idx: std::collections::HashMap<(String, String), usize> =
        std::collections::HashMap::new();
    let mut out: Vec<Bond> = Vec::with_capacity(all.len());
    for b in all {
        let key = (b.subject.to_lowercase(), b.object.to_lowercase());
        match best_idx.get(&key) {
            None => {
                best_idx.insert(key, out.len());
                out.push(b);
            }
            Some(&i) => {
                if relation_priority(b.relation) > relation_priority(out[i].relation) {
                    out[i] = b;
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispatch_empty_inputs_return_empty() {
        assert!(extract_from_article("", Some("Title"), None).is_empty());
        assert!(extract_from_article("text", None, None).is_empty());
        assert!(extract_from_article("text", Some(""), None).is_empty());
    }

    #[test]
    fn dispatch_runs_all_layers_and_dedups() {
        // Title-prefix IsA fires on "Rust is a fast language."
        // because the [.;,] lookahead matches the period.
        let article = "Rust is a fast language.\n\nIt is used widely.\n\nSee also\n\n Garbage collection\n Memory safety\n\nReferences\n";
        let bonds = extract_from_article(article, Some("Rust"), None);
        assert!(bonds.iter().any(|b| b.layer == "L1"));
        assert!(
            bonds.iter().any(|b| b.layer.starts_with("L3")),
            "expected an L3 bond, got {:?}",
            bonds.iter().map(|b| b.layer).collect::<Vec<_>>()
        );
    }
}
