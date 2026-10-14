//! L5 — Typed numerical / attribute extractor. Emits subject-anchored
//! typed bonds for the 8 attribute patterns.
//!
//! Mirrors `extract_numerical_attrs` in `smart_extractor.py`.

use regex::Regex;
use std::collections::HashSet;

use crate::bond::BondType;
use crate::extractor::aliases::{build_subject_aliases, sentence_anchors_subject};
use crate::extractor::labels::{clean_label, ok_term, trim_object};
use crate::extractor::patterns::{
    rx_altitude, rx_area_km2, rx_body_founded_in, rx_born_date, rx_capital,
    rx_died_date, rx_lang_official, rx_population,
    rx_population2, rx_population3, rx_area2, rx_capital2, rx_founded_year2,
    rx_numeric_ok,
};
use crate::extractor::sentence::split_sentences;
use crate::extractor::tail_strip::strip_tail_sections;
use crate::extractor::types::{kind_for, Bond, ConfidenceLabel};

struct L5Entry {
    getter: fn() -> &'static Regex,
    relation: BondType,
    weight: f32,
    layer: &'static str,
}

fn l5_entries() -> &'static [L5Entry] {
    use std::sync::OnceLock;
    static V: OnceLock<Vec<L5Entry>> = OnceLock::new();
    V.get_or_init(|| {
        vec![
            L5Entry { getter: rx_body_founded_in, relation: BondType::FoundedIn, weight: 0.6, layer: "L5-foundyear" },
            L5Entry { getter: rx_population, relation: BondType::HasPopulation, weight: 0.55, layer: "L5-pop" },
            L5Entry { getter: rx_area_km2, relation: BondType::HasArea, weight: 0.55, layer: "L5-area" },
            L5Entry { getter: rx_altitude, relation: BondType::HasAltitude, weight: 0.55, layer: "L5-alt" },
            L5Entry { getter: rx_lang_official, relation: BondType::HasOfficialLanguage, weight: 0.6, layer: "L5-lang" },
            L5Entry { getter: rx_capital, relation: BondType::HasCapital, weight: 0.6, layer: "L5-cap" },
            L5Entry { getter: rx_born_date, relation: BondType::BornOn, weight: 0.6, layer: "L5-borndt" },
            L5Entry { getter: rx_died_date, relation: BondType::DiedOn, weight: 0.6, layer: "L5-dieddt" },
            L5Entry { getter: rx_population2, relation: BondType::HasPopulation, weight: 0.55, layer: "L5-pop" },
            L5Entry { getter: rx_population3, relation: BondType::HasPopulation, weight: 0.55, layer: "L5-pop" },
            L5Entry { getter: rx_area2, relation: BondType::HasArea, weight: 0.55, layer: "L5-area" },
            L5Entry { getter: rx_capital2, relation: BondType::HasCapital, weight: 0.6, layer: "L5-cap" },
            L5Entry { getter: rx_founded_year2, relation: BondType::FoundedIn, weight: 0.55, layer: "L5-foundyear" },
        ]
    })
}

// OP-EXTRACT-BREADTH B2 — relation-aware object validator. Mirrors
// `_ok_attr_term` in smart_extractor.py: numeric attribute relations accept
// a bare digit+unit shape (the old ok_term rejected pure-numeric objects, so
// bare-year FoundedIn and bare-number HasPopulation never landed); alpha
// relations keep the strict ok_term.
fn ok_attr_term(s: &str, relation: BondType) -> bool {
    if s.is_empty() || s.chars().count() > 80 {
        return false;
    }
    match relation {
        BondType::HasPopulation
        | BondType::HasArea
        | BondType::HasAltitude
        | BondType::FoundedIn => rx_numeric_ok().is_match(s.trim()),
        _ => ok_term(s, 2, 80),
    }
}

pub fn extract(title: &str, text: &str, source_url: Option<&str>) -> Vec<Bond> {
    let mut bonds = Vec::new();
    if title.is_empty() || text.is_empty() {
        return bonds;
    }
    let body = strip_tail_sections(text);
    if body.is_empty() {
        return bonds;
    }
    let aliases = build_subject_aliases(title);
    let sentences = split_sentences(&body);
    let title_lower = title.to_lowercase();
    let mut seen: HashSet<(BondType, String)> = HashSet::new();

    for sentence in &sentences {
        if !sentence_anchors_subject(sentence, &aliases) {
            continue;
        }
        for entry in l5_entries() {
            let re = (entry.getter)();
            for cap in re.captures_iter(sentence) {
                let raw = match cap.get(1) {
                    Some(g) => g.as_str(),
                    None => continue,
                };
                let obj = trim_object(&clean_label(raw));
                if obj.is_empty() {
                    continue;
                }
                if !ok_attr_term(&obj, entry.relation) {
                    continue;
                }
                let ol = obj.to_lowercase();
                if ol == title_lower {
                    continue;
                }
                let key = (entry.relation, ol.clone());
                if seen.contains(&key) {
                    continue;
                }
                seen.insert(key);
                bonds.push(Bond {
                    subject: title.to_string(),
                    relation: entry.relation,
                    object: obj,
                    kind: kind_for(entry.relation),
                    weight: entry.weight,
                    confidence: ConfidenceLabel::Inferred,
                    layer: entry.layer,
                    source_url: source_url.map(|s| s.to_string()),
                    frequency: None,
                    subject_origin: None,
                });
            }
        }
    }
    bonds
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn l5_founded_in_and_population() {
        // OP-EXTRACT-BREADTH B2: the relation-aware ok_attr_term now accepts
        // bare-numeric objects for numeric relations, so a bare founded year
        // ("1801") lands as FoundedIn (previously silently dropped by ok_term).
        let text = "Springfield was founded in 1801 and has a population of 50 thousand people. The mayor lives there.";
        let bonds = extract("Springfield", text, None);
        let pop = bonds.iter().any(|b| b.relation == BondType::HasPopulation);
        let founded = bonds
            .iter()
            .any(|b| b.relation == BondType::FoundedIn && b.object.contains("1801"));
        assert!(pop, "HasPopulation missing, got {:?}", bonds);
        assert!(founded, "FoundedIn bare-year missing, got {:?}", bonds);
    }

    #[test]
    fn l5_subject_anchored() {
        // Population statement is about Mexico, not Lincoln — must be
        // dropped by the subject-anchor gate.
        let text =
            "Abraham Lincoln served as president. Mexico has a population of 130 million people.";
        let bonds = extract("Abraham Lincoln", text, None);
        assert!(
            bonds.iter().all(|b| b.relation != BondType::HasPopulation),
            "unexpected pop bond {:?}",
            bonds
        );
    }

    #[test]
    fn l5_born_date() {
        let text = "Abraham Lincoln was born on February 12, 1809 in Kentucky. He served as president.";
        let bonds = extract("Abraham Lincoln", text, None);
        assert!(bonds.iter().any(|b| b.relation == BondType::BornOn));
    }
}
