//! L1 — See also section walker. Emits `(title, RelatedTo, item)`
//! bonds for each bullet item in the article's See-also section.
//!
//! Python source of truth: `extract_see_also` in `smart_extractor.py`.

use fancy_regex::Regex as FRegex;
use std::sync::OnceLock;

use crate::bond::BondType;
use crate::extractor::labels::{clean_label, ok_term};
use crate::extractor::patterns::drop_headings;
use crate::extractor::types::{kind_for, Bond, ConfidenceLabel};

fn see_also_block_re() -> &'static FRegex {
    static R: OnceLock<FRegex> = OnceLock::new();
    R.get_or_init(|| {
        // Mirrors Python `r"\nSee also\s*\n(.*?)(?=\n[A-Z][A-Za-z][^\n]{0,60}\s*\n|$)"`.
        // The DOTALL behaviour is selected with `(?s)`. The lookahead
        // anchors on the next ALL-CAPS-prefixed heading line or EOF.
        FRegex::new(
            r"(?s)\nSee also\s*\n(.*?)(?=\n[A-Z][A-Za-z][^\n]{0,60}\s*\n|$)",
        )
        .unwrap()
    })
}

pub fn extract(title: &str, text: &str, source_url: Option<&str>) -> Vec<Bond> {
    let mut bonds = Vec::new();
    if title.is_empty() || text.is_empty() {
        return bonds;
    }
    let m = match see_also_block_re().captures(text) {
        Ok(Some(c)) => c,
        _ => return bonds,
    };
    let block = match m.get(1) {
        Some(g) => g.as_str(),
        None => return bonds,
    };
    for raw in block.split('\n') {
        // Strip bullets + leading whitespace, then clean.
        let trimmed: String = raw
            .trim_start_matches(|c: char| {
                c == ' '
                    || c == '\t'
                    || c == '*'
                    || c == '•'
                    || c == '·'
                    || c == '–'
                    || c == '—'
                    || c == '‐'
                    || c == '-'
            })
            .to_string();
        let item = clean_label(&trimmed);
        if !ok_term(&item, 3, 80) {
            continue;
        }
        let il = item.to_lowercase();
        if drop_headings().contains(il.as_str()) {
            continue;
        }
        if item.starts_with("http") || item.starts_with("www.") {
            continue;
        }
        bonds.push(Bond {
            subject: title.to_string(),
            relation: BondType::RelatedTo,
            object: item,
            kind: kind_for(BondType::RelatedTo),
            weight: 0.55,
            confidence: ConfidenceLabel::Unverified,
            layer: "L1",
            source_url: source_url.map(|s| s.to_string()),
            frequency: None,
            subject_origin: None,
        });
    }
    bonds
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn l1_extracts_see_also_bullets() {
        let article = "Albedo is the fraction of sunlight reflected.\n\n\
See also\n\n Cool roof\n Daisyworld\n Emissivity\n Global dimming\n\nReferences\n";
        let bonds = extract("Albedo", article, Some("https://example.com"));
        assert!(bonds.len() >= 3, "expected >=3 bonds, got {}", bonds.len());
        let objects: Vec<&str> = bonds.iter().map(|b| b.object.as_str()).collect();
        assert!(objects.contains(&"Cool roof"));
        assert!(objects.contains(&"Daisyworld"));
        for b in &bonds {
            assert_eq!(b.relation, BondType::RelatedTo);
            assert_eq!(b.subject, "Albedo");
            assert_eq!(b.layer, "L1");
            assert!((b.weight - 0.55).abs() < 1e-6);
        }
    }

    #[test]
    fn l1_no_see_also_section_returns_empty() {
        let article = "Body without that section.";
        let bonds = extract("Topic", article, None);
        assert!(bonds.is_empty());
    }

    #[test]
    fn l1_drops_url_lines() {
        let article = "Body content here.\n\nSee also\n\n http://example.com\n Real Topic\n\nReferences\n";
        let bonds = extract("Title", article, None);
        let objects: Vec<&str> = bonds.iter().map(|b| b.object.as_str()).collect();
        assert!(objects.contains(&"Real Topic"));
        assert!(!objects.iter().any(|o| o.starts_with("http")));
    }
}
