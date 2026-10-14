//! L4 — Body proper-noun co-mention walker. Frequency-weighted
//! RelatedTo bonds from the article title to each capitalised noun
//! phrase mentioned ≥ 2 times in the body.
//!
//! Mirrors `extract_body_comentions` in `smart_extractor.py`.

use std::collections::HashMap;
use std::collections::HashSet;

use crate::bond::BondType;
use crate::extractor::labels::{clean_label, ok_term};
use crate::extractor::patterns::{
    alpha_word_re, l4_cap_phrase_re, l4_generic_adjectives, l4_single_word_stop, stop_objects,
};
use crate::extractor::tail_strip::strip_tail_sections;
use crate::extractor::types::{kind_for, Bond, ConfidenceLabel};

fn normalize_phrase(phrase: &str) -> (String, Vec<String>) {
    let mut words: Vec<String> = phrase.split_whitespace().map(|s| s.to_string()).collect();
    while !words.is_empty() {
        let head = words[0].to_lowercase();
        if matches!(head.as_str(), "the" | "a" | "an") {
            words.remove(0);
        } else {
            break;
        }
    }
    while !words.is_empty() {
        let last = words.last().unwrap().to_lowercase();
        if matches!(
            last.as_str(),
            "of" | "the" | "de" | "van" | "von" | "den" | "du" | "la" | "and" | "or"
        ) {
            words.pop();
        } else {
            break;
        }
    }
    // Strip apostrophe-s possessive.
    if let Some(last) = words.last_mut() {
        if last.ends_with("'s") {
            let trimmed = last.trim_end_matches("'s").to_string();
            if !trimmed.is_empty() {
                *last = trimmed;
            } else {
                words.pop();
            }
        } else if last.ends_with("\u{2019}s") {
            let trimmed = last.trim_end_matches("\u{2019}s").to_string();
            if !trimmed.is_empty() {
                *last = trimmed;
            } else {
                words.pop();
            }
        }
    }
    let joined = words.join(" ");
    (joined, words)
}

fn title_word_prefixes(title: &str, prefix_len: usize) -> HashSet<String> {
    let mut out = HashSet::new();
    for m in alpha_word_re().find_iter(title) {
        let w = m.as_str();
        if w.chars().count() >= prefix_len {
            let lower = w.to_lowercase();
            let pref: String = lower.chars().take(prefix_len).collect();
            out.insert(pref);
        }
    }
    out
}

pub fn extract(title: &str, text: &str, source_url: Option<&str>, max_emit: usize) -> Vec<Bond> {
    let mut bonds = Vec::new();
    if title.is_empty() || text.is_empty() {
        return bonds;
    }
    let body = strip_tail_sections(text);
    if body.is_empty() {
        return bonds;
    }
    let title_lower = title.to_lowercase();
    let title_words: HashSet<String> = alpha_word_re()
        .find_iter(title)
        .map(|m| m.as_str().to_lowercase())
        .collect();
    let title_prefixes = title_word_prefixes(title, 5);

    let digits_only = regex::Regex::new(r"^\d+$").unwrap();

    let mut counts: HashMap<String, u32> = HashMap::new();
    for cap in l4_cap_phrase_re().captures_iter(&body) {
        let raw_m = match cap.get(1) {
            Some(g) => g.as_str(),
            None => continue,
        };
        let raw = clean_label(raw_m);
        if raw.is_empty() {
            continue;
        }
        let (phrase, words) = normalize_phrase(&raw);
        if phrase.is_empty() || words.is_empty() {
            continue;
        }

        if words.len() == 1 {
            let wl = phrase.to_lowercase();
            if l4_single_word_stop().contains(wl.as_str()) {
                continue;
            }
            if l4_generic_adjectives().contains(wl.as_str()) {
                continue;
            }
            if stop_objects().contains(wl.as_str()) {
                continue;
            }
            if phrase.chars().count() < 3 {
                continue;
            }
            if title_words.contains(&wl) {
                continue;
            }
            if wl.chars().count() >= 5 {
                let pref: String = wl.chars().take(5).collect();
                if title_prefixes.contains(&pref) {
                    continue;
                }
            }
        }
        if phrase.to_lowercase() == title_lower {
            continue;
        }
        if digits_only.is_match(&phrase) {
            continue;
        }
        if !ok_term(&phrase, 3, 80) {
            continue;
        }
        *counts.entry(phrase).or_insert(0) += 1;
    }

    let mut candidates: Vec<(String, u32)> =
        counts.into_iter().filter(|(_, c)| *c >= 2).collect();
    // Sort by freq desc, then alphabetic asc — matches Python
    // `(-x[1], x[0])`.
    candidates.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    candidates.truncate(max_emit);

    for (phrase, freq) in candidates {
        let weight = if freq >= 5 {
            0.55
        } else if freq >= 3 {
            0.5
        } else {
            0.4
        };
        bonds.push(Bond {
            subject: title.to_string(),
            relation: BondType::RelatedTo,
            object: phrase,
            kind: kind_for(BondType::RelatedTo),
            weight,
            confidence: ConfidenceLabel::Unverified,
            layer: "L4",
            source_url: source_url.map(|s| s.to_string()),
            frequency: Some(freq),
            subject_origin: None,
        });
    }
    bonds
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn l4_picks_up_repeated_proper_nouns() {
        let article = "Acme Corp makes widgets. Acme Corp was founded in 1990. The CEO of Acme Corp lives in Springfield. Springfield is a town. Acme Corp expanded to Springfield.";
        let bonds = extract("Widgets Inc", article, None, 80);
        let acme: Vec<&Bond> = bonds.iter().filter(|b| b.object == "Acme Corp").collect();
        assert!(!acme.is_empty(), "expected Acme Corp bond, got {:?}", bonds);
        for b in &bonds {
            assert_eq!(b.relation, BondType::RelatedTo);
            assert_eq!(b.layer, "L4");
        }
    }

    #[test]
    fn l4_skips_title_words() {
        let article = "Lincoln was a president. Lincoln moved to Illinois. Lincoln stayed in Illinois.";
        let bonds = extract("Abraham Lincoln", article, None, 80);
        // Lincoln is part of the title — should be filtered out.
        assert!(!bonds.iter().any(|b| b.object == "Lincoln"));
    }
}
