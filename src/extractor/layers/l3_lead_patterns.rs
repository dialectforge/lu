//! L3 — Lead-paragraph subject-anchored typed patterns. Emits typed
//! bonds from the article's first three paragraphs.
//!
//! Mirrors `extract_lead_patterns` + helpers in `smart_extractor.py`.
//! Two-stage:
//!   (a) title-prefix IsA — anchored by construction.
//!   (b) all other patterns + IsA fallback — only on sentences that
//!       contain the article subject (title or alias).

use fancy_regex::Regex as FRegex;
use regex::Regex;
use std::collections::HashSet;
use std::sync::OnceLock;

use crate::bond::BondType;
use crate::extractor::aliases::{build_subject_aliases, sentence_anchors_subject};
use crate::extractor::labels::{clean_label, ok_term, trim_object};
use crate::extractor::patterns::{
    rx_born_in, rx_composed_of, rx_died_in, rx_discovered_by, rx_founded_by,
    rx_founded_in, rx_in_place, rx_isa_fallback, rx_known_for, rx_located,
    rx_measured_in, rx_named_after, rx_part_of, rx_type_of, rx_used_for,
    rx_written_by, stop_isa_nouns, stop_objects,
    rx_geo_in, rx_subsidiary, rx_causes, rx_requires, rx_produces,
    rx_person_name, rx_one_date, rx_date_range, category_hints_re,
};
use crate::extractor::sentence::{first_n_paragraphs, split_sentences};
use crate::extractor::types::{kind_for, Bond, ConfidenceLabel};

enum L3Pattern {
    Fancy(fn() -> &'static FRegex, usize),
    Plain(fn() -> &'static Regex, usize),
}

struct L3Entry {
    pattern: L3Pattern,
    relation: BondType,
    weight: f32,
    layer: &'static str,
}

fn l3_patterns() -> &'static Vec<L3Entry> {
    static V: OnceLock<Vec<L3Entry>> = OnceLock::new();
    V.get_or_init(|| {
        vec![
            // _RX_LOCATED captures group 2 (group 1 is the optional "the\s+").
            L3Entry {
                pattern: L3Pattern::Fancy(rx_located, 2),
                relation: BondType::LocatedIn,
                weight: 0.6,
                layer: "L3-loc",
            },
            L3Entry {
                pattern: L3Pattern::Fancy(rx_in_place, 1),
                relation: BondType::LocatedIn,
                weight: 0.6,
                layer: "L3-loc",
            },
            L3Entry {
                pattern: L3Pattern::Fancy(rx_part_of, 1),
                relation: BondType::PartOf,
                weight: 0.55,
                layer: "L3-part",
            },
            L3Entry {
                pattern: L3Pattern::Fancy(rx_founded_by, 1),
                relation: BondType::FoundedBy,
                weight: 0.6,
                layer: "L3-foundby",
            },
            L3Entry {
                pattern: L3Pattern::Plain(rx_founded_in, 1),
                relation: BondType::FoundedIn,
                weight: 0.6,
                layer: "L3-foundin",
            },
            L3Entry {
                pattern: L3Pattern::Fancy(rx_discovered_by, 1),
                relation: BondType::DiscoveredBy,
                weight: 0.6,
                layer: "L3-discby",
            },
            L3Entry {
                pattern: L3Pattern::Fancy(rx_named_after, 1),
                relation: BondType::NamedAfter,
                weight: 0.6,
                layer: "L3-named",
            },
            L3Entry {
                pattern: L3Pattern::Fancy(rx_measured_in, 1),
                relation: BondType::MeasuredIn,
                weight: 0.5,
                layer: "L3-meas",
            },
            L3Entry {
                pattern: L3Pattern::Fancy(rx_born_in, 1),
                relation: BondType::BornIn,
                weight: 0.55,
                layer: "L3-born",
            },
            L3Entry {
                pattern: L3Pattern::Fancy(rx_died_in, 1),
                relation: BondType::DiedIn,
                weight: 0.55,
                layer: "L3-died",
            },
            L3Entry {
                pattern: L3Pattern::Fancy(rx_written_by, 1),
                relation: BondType::WrittenBy,
                weight: 0.6,
                layer: "L3-writby",
            },
            L3Entry {
                pattern: L3Pattern::Fancy(rx_composed_of, 1),
                relation: BondType::PartOf,
                weight: 0.5,
                layer: "L3-comp",
            },
            L3Entry {
                pattern: L3Pattern::Fancy(rx_known_for, 1),
                relation: BondType::KnownFor,
                weight: 0.5,
                layer: "L3-known",
            },
            L3Entry {
                pattern: L3Pattern::Fancy(rx_used_for, 1),
                relation: BondType::UsedFor,
                weight: 0.5,
                layer: "L3-used",
            },
            L3Entry {
                pattern: L3Pattern::Fancy(rx_type_of, 1),
                relation: BondType::IsA,
                weight: 0.55,
                layer: "L3-type",
            },
            L3Entry { pattern: L3Pattern::Fancy(rx_geo_in, 2), relation: BondType::LocatedIn, weight: 0.55, layer: "L3-geoin" },
            L3Entry { pattern: L3Pattern::Fancy(rx_subsidiary, 1), relation: BondType::PartOf, weight: 0.55, layer: "L3-subsid" },
            L3Entry { pattern: L3Pattern::Fancy(rx_causes, 1), relation: BondType::Causes, weight: 0.5, layer: "L3-causes" },
            L3Entry { pattern: L3Pattern::Fancy(rx_requires, 1), relation: BondType::Requires, weight: 0.5, layer: "L3-req" },
            L3Entry { pattern: L3Pattern::Fancy(rx_produces, 1), relation: BondType::Produces, weight: 0.5, layer: "L3-prod" },
        ]
    })
}

/// Title-prefix IsA matcher. Anchored by construction — title must
/// immediately precede "is/was/are/were".
///
/// Returns a `Vec` of `(category, weight)` because a coordinator-joined
/// NP can yield two candidates. Mirrors Python `_isa_lead_title_prefix`
/// after the 2026-05-31 widening.
fn isa_lead_title_prefix(title: &str, lead: &str) -> Vec<(String, f32)> {
    if title.is_empty() {
        return Vec::new();
    }
    let pat = format!(
        r"(?i){title_esc}\s+(?:is|was|are|were)\s+(?:an?|the)\s+([a-z][a-zA-Z\-]{{2,30}}(?:\s+[a-z][a-zA-Z\-]{{0,30}}){{0,5}})(?=[.;,]|$|\s+(?:that|which|who|where|and|or|used|of|in|for|from)\b)",
        title_esc = regex::escape(title),
    );
    let re = match Regex::new(&pat) {
        Ok(r) => r,
        Err(_) => return Vec::new(),
    };
    let m = match re.captures(lead) {
        Some(c) => c,
        None => return Vec::new(),
    };
    let cat_raw = match m.get(1) {
        Some(g) => g.as_str().to_lowercase(),
        None => return Vec::new(),
    };
    let cat = trim_object(&clean_label(&cat_raw)).to_lowercase();
    emit_isa_candidates(&cat, 0.65, 70)
}

/// Take a lowercased NP, optionally split on a coordinator (" and " /
/// " or ") into 2 candidates, validate each via `ok_term` and the
/// `stop_isa_nouns` head-word check, return list of `(cat, weight)`
/// survivors. Shared by title-prefix and fallback paths.
fn emit_isa_candidates(np_lower: &str, weight: f32, max_len: usize) -> Vec<(String, f32)> {
    if np_lower.is_empty() {
        return Vec::new();
    }
    static SPLIT_RE: OnceLock<Regex> = OnceLock::new();
    let split_re = SPLIT_RE.get_or_init(|| {
        Regex::new(r"^(.+?)\s+(?:and|or)\s+(?:an?\s+|the\s+)?(.+)$").unwrap()
    });
    let mut candidates: Vec<String> = vec![np_lower.to_string()];
    if let Some(caps) = split_re.captures(np_lower) {
        let left = clean_label(caps.get(1).map(|m| m.as_str()).unwrap_or("")).to_lowercase();
        let right = clean_label(caps.get(2).map(|m| m.as_str()).unwrap_or("")).to_lowercase();
        if !left.is_empty() && !right.is_empty() && left != right {
            candidates = vec![left, right];
        }
    }
    let mut out = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    for cat in candidates {
        let cat = trim_object(&cat).to_lowercase();
        let head = cat.split_whitespace().next().unwrap_or("");
        if stop_isa_nouns().contains(head) {
            continue;
        }
        if !ok_term(&cat, 3, max_len) {
            continue;
        }
        if seen.contains(&cat) {
            continue;
        }
        seen.insert(cat.clone());
        out.push((cat, weight));
    }
    out
}

fn isa_lead_fallback_in_sentence(sentence: &str) -> Vec<(String, f32)> {
    let m = match rx_isa_fallback().captures(sentence).ok().flatten() {
        Some(c) => c,
        None => return Vec::new(),
    };
    let cat_raw = match m.get(1) {
        Some(g) => g.as_str().to_lowercase(),
        None => return Vec::new(),
    };
    let cat = trim_object(&clean_label(&cat_raw)).to_lowercase();
    emit_isa_candidates(&cat, 0.5, 60)
}

fn capture_obj(entry: &L3Entry, sentence: &str) -> Vec<String> {
    let mut out = Vec::new();
    match &entry.pattern {
        L3Pattern::Fancy(getter, idx) => {
            let re = getter();
            for m in re.captures_iter(sentence) {
                if let Ok(c) = m {
                    if let Some(g) = c.get(*idx) {
                        out.push(g.as_str().to_string());
                    }
                }
            }
        }
        L3Pattern::Plain(getter, idx) => {
            let re = getter();
            for c in re.captures_iter(sentence) {
                if let Some(g) = c.get(*idx) {
                    out.push(g.as_str().to_string());
                }
            }
        }
    }
    out
}

/// OP-EXTRACT-BREADTH B1 — construction-anchored lead parenthetical
/// birth/death dates. Mirrors `_extract_lead_parenthetical_dates` in
/// `smart_extractor.py`. Person-name title + month-bearing date only.
fn looks_like_person_name(s: &str) -> bool {
    rx_person_name().is_match(s) && !category_hints_re().is_match(s)
}

fn extract_lead_parenthetical_dates(
    title: &str,
    lead_text: &str,
    source_url: Option<&str>,
) -> Vec<Bond> {
    let mut out = Vec::new();
    if title.is_empty() || lead_text.is_empty() {
        return out;
    }
    if !looks_like_person_name(title) {
        return out;
    }
    let pat = format!(r"^\s*{}\s*\(([^)]{{4,80}})\)", regex::escape(title));
    let re = match Regex::new(&pat) {
        Ok(r) => r,
        Err(_) => return out,
    };
    let caps = match re.captures(lead_text) {
        Some(c) => c,
        None => return out,
    };
    let paren = caps.get(1).map(|m| m.as_str()).unwrap_or("");
    let mut born: Option<String> = None;
    let mut died: Option<String> = None;
    if let Some(r) = rx_date_range().captures(paren) {
        born = r.get(1).map(|m| m.as_str().trim().to_string());
        died = r.get(2).map(|m| m.as_str().trim().to_string());
    } else if let Some(one) = rx_one_date().find(paren) {
        let token = one.as_str().trim().to_string();
        let before = &paren[..one.start()];
        static DIED_RE: OnceLock<Regex> = OnceLock::new();
        let died_re = DIED_RE.get_or_init(|| Regex::new(r"(?i)\b(?:died|d\.)").unwrap());
        if died_re.is_match(before) {
            died = Some(token);
        } else {
            born = Some(token);
        }
    }
    for (rel, val) in [(BondType::BornOn, born), (BondType::DiedOn, died)] {
        if let Some(v) = val {
            if !ok_term(&v, 4, 40) {
                continue;
            }
            out.push(Bond {
                subject: title.to_string(),
                relation: rel,
                object: v,
                kind: kind_for(rel),
                weight: 0.62,
                confidence: ConfidenceLabel::Inferred,
                layer: "L3-parendt",
                source_url: source_url.map(|s| s.to_string()),
                frequency: None,
                subject_origin: None,
            });
        }
    }
    out
}

pub fn extract(title: &str, text: &str, source_url: Option<&str>) -> Vec<Bond> {
    let mut bonds = Vec::new();
    if title.is_empty() || text.is_empty() {
        return bonds;
    }
    let paragraphs = first_n_paragraphs(text, 3);
    if paragraphs.is_empty() {
        return bonds;
    }
    let lead_text = paragraphs.join(" ");
    // OP-EXTRACT-BREADTH B1: construction-anchored parenthetical dates.
    bonds.extend(extract_lead_parenthetical_dates(title, &lead_text, source_url));
    let aliases = build_subject_aliases(title);
    let sentences = split_sentences(&lead_text);
    let title_lower = title.to_lowercase();

    let mut seen_relobj: HashSet<(BondType, String)> = HashSet::new();
    let mut isa_fired = false;

    // Title-prefix IsA: anchored by construction.
    // May return >1 bond (coordinator split).
    let isa_results = isa_lead_title_prefix(title, &lead_text);
    for (cat, w) in isa_results {
        let key = (BondType::IsA, cat.clone());
        if seen_relobj.contains(&key) {
            continue;
        }
        seen_relobj.insert(key);
        bonds.push(Bond {
            subject: title.to_string(),
            relation: BondType::IsA,
            object: cat,
            kind: kind_for(BondType::IsA),
            weight: w,
            confidence: ConfidenceLabel::Inferred,
            layer: "L3-isa",
            source_url: source_url.map(|s| s.to_string()),
            frequency: None,
            subject_origin: None,
        });
        isa_fired = true;
    }

    for sentence in &sentences {
        if !sentence_anchors_subject(sentence, &aliases) {
            continue;
        }

        // Fallback IsA — only if title-prefix hasn't fired.
        if !isa_fired {
            for (cat, w) in isa_lead_fallback_in_sentence(sentence) {
                let key = (BondType::IsA, cat.clone());
                if seen_relobj.contains(&key) {
                    continue;
                }
                seen_relobj.insert(key);
                bonds.push(Bond {
                    subject: title.to_string(),
                    relation: BondType::IsA,
                    object: cat,
                    kind: kind_for(BondType::IsA),
                    weight: w,
                    confidence: ConfidenceLabel::Inferred,
                    layer: "L3-isa",
                    source_url: source_url.map(|s| s.to_string()),
                    frequency: None,
                    subject_origin: None,
                });
                isa_fired = true;
            }
        }

        for entry in l3_patterns() {
            for raw in capture_obj(entry, sentence) {
                let obj = trim_object(&clean_label(&raw));
                if !ok_term(&obj, 3, 80) {
                    continue;
                }
                let ol = obj.to_lowercase();
                if stop_objects().contains(ol.as_str()) {
                    continue;
                }
                if ol == title_lower {
                    continue;
                }
                if entry.relation == BondType::IsA {
                    let head = ol.split_whitespace().next().unwrap_or("");
                    if stop_isa_nouns().contains(head) {
                        continue;
                    }
                }
                let key = (entry.relation, ol.clone());
                if seen_relobj.contains(&key) {
                    continue;
                }
                seen_relobj.insert(key);
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
    fn l3_title_prefix_isa() {
        let text = "Rust is a systems programming language. It was designed by Mozilla.";
        let bonds = extract("Rust", text, None);
        let has_isa = bonds
            .iter()
            .any(|b| b.relation == BondType::IsA && b.object.contains("systems"));
        assert!(has_isa, "expected IsA bond, got {:?}", bonds);
    }

    #[test]
    fn l3_born_in() {
        let text = "Abraham Lincoln was born in Hodgenville, Kentucky in 1809. He served as president.";
        let bonds = extract("Abraham Lincoln", text, None);
        let has_born = bonds.iter().any(|b| b.relation == BondType::BornIn);
        assert!(has_born, "expected BornIn bond, got {:?}", bonds);
    }

    #[test]
    fn l3_located_in() {
        let text =
            "Acme Corp is headquartered in Springfield, which is in Illinois. The company makes things.";
        let bonds = extract("Acme Corp", text, None);
        let has_loc = bonds.iter().any(|b| b.relation == BondType::LocatedIn);
        assert!(has_loc, "expected LocatedIn bond, got {:?}", bonds);
    }

    #[test]
    fn l3_parenthetical_dates() {
        let text = "Ada Lovelace (10 December 1815 - 27 November 1852) was an English mathematician.";
        let bonds = extract("Ada Lovelace", text, None);
        assert!(bonds.iter().any(|b| b.relation == BondType::BornOn && b.object.contains("1815")), "BornOn missing: {:?}", bonds);
        assert!(bonds.iter().any(|b| b.relation == BondType::DiedOn && b.object.contains("1852")), "DiedOn missing: {:?}", bonds);
    }

    #[test]
    fn l3_geo_in_and_causes() {
        let loc = extract("Lyon", "Lyon is a city in France. It is large.", None);
        assert!(loc.iter().any(|b| b.relation == BondType::LocatedIn && b.object == "France"), "LocatedIn missing: {:?}", loc);
        let cause = extract("Smoking", "Smoking causes lung cancer.", None);
        assert!(cause.iter().any(|b| b.relation == BondType::Causes), "Causes missing: {:?}", cause);
        let neg = extract("Cancer", "The causes of cancer are complex.", None);
        assert!(neg.iter().all(|b| b.relation != BondType::Causes), "unexpected Causes: {:?}", neg);
    }
}
