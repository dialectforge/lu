//! L2b — Tightened category-tail walker. Emits `(title, IsA, cat)`
//! bonds for the contiguous run of one-line heading-shaped category
//! lines at the very end of an article.
//!
//! Heuristic (Python `extract_category_tail`, tightened 2026-05-30):
//!   * Take the last 60 non-empty lines.
//!   * Find the LAST contiguous run of heading-shaped lines of size
//!     ≥ 3. That's the Wikipedia category footer.
//!   * Emit each line as an IsA bond.
//!
//! Noise filter (added 2026-05-31 — text-extractor v2 Fix 1):
//! drops reference-subsection labels even if they pass the
//! heading-shape gate. The earlier port omitted the canonical
//! `_L2B_REFERENCE_NOISE` set on the assumption that the
//! longest-run gate alone carried the load; on articles where the
//! References section's subheadings form a long-enough run that
//! sits between the body and the category footer, the noise still
//! leaks. Shared with L2 via `extractor/layers/shared_noise.rs`.

use crate::bond::BondType;
use crate::extractor::labels::{clean_label, ok_term};
use crate::extractor::layers::shared_noise::reference_noise;
use crate::extractor::patterns::{
    digits_only_re, drop_headings, heading_re, person_name_re, category_hints_re,
};
use crate::extractor::types::{kind_for, Bond, ConfidenceLabel};

fn looks_like_person_name(s: &str) -> bool {
    person_name_re().is_match(s) && !category_hints_re().is_match(s)
}

fn is_heading_shape(line: &str) -> bool {
    if !heading_re().is_match(line) {
        return false;
    }
    let last = line.chars().last();
    if matches!(last, Some('.') | Some('!') | Some('?') | Some(':') | Some(';')) {
        return false;
    }
    if line.starts_with("http") || line.starts_with("www.") {
        return false;
    }
    true
}

pub fn extract(title: &str, text: &str, source_url: Option<&str>) -> Vec<Bond> {
    let mut bonds = Vec::new();
    if title.is_empty() || text.is_empty() {
        return bonds;
    }
    let nonempty_lines: Vec<&str> = text
        .split('\n')
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .collect();
    if nonempty_lines.is_empty() {
        return bonds;
    }
    let tail_start = nonempty_lines.len().saturating_sub(60);
    let tail_lines: &[&str] = &nonempty_lines[tail_start..];

    // Walk the tail, accumulating contiguous heading-shape runs.
    let mut runs: Vec<Vec<&str>> = Vec::new();
    let mut current: Vec<&str> = Vec::new();
    for line in tail_lines {
        if is_heading_shape(line) {
            current.push(*line);
        } else if !current.is_empty() {
            runs.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        runs.push(std::mem::take(&mut current));
    }

    // Pick the LAST run of length ≥ 3.
    let category_run = runs
        .into_iter()
        .rev()
        .find(|r| r.len() >= 3);
    let category_run = match category_run {
        Some(r) => r,
        None => return bonds,
    };

    let title_lower = title.to_lowercase();
    for line in category_run {
        let cat = clean_label(line);
        if !ok_term(&cat, 4, 70) {
            continue;
        }
        let cl = cat.to_lowercase();
        if drop_headings().contains(cl.as_str()) {
            continue;
        }
        if reference_noise().contains(cl.as_str()) {
            continue;
        }
        if cl == title_lower {
            continue;
        }
        if digits_only_re().is_match(&cat) {
            continue;
        }
        if looks_like_person_name(&cat) {
            continue;
        }
        bonds.push(Bond {
            subject: title.to_string(),
            relation: BondType::IsA,
            object: cat,
            kind: kind_for(BondType::IsA),
            weight: 0.5,
            confidence: ConfidenceLabel::Unverified,
            layer: "L2b",
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

    /// Article with a real External Links section ABOVE the category
    /// footer. The longest-contiguous-run-of-3+ heuristic should pick
    /// up ONLY the trailing category run, ignoring the broken-up
    /// link captions interleaved with URLs.
    #[test]
    fn l2b_picks_categories_not_external_link_captions() {
        let article = "\
Body content here in some paragraphs.

Introduction text.

External links

NASA MODIS BRDF albedo product site
http://modis.gsfc.nasa.gov/data
A discussion of Lunar albedos
http://example.com/lunar
Goddard Earth Sciences Data
http://disc.gsfc.nasa.gov

Anti-capitalism
Far-left politics
Political philosophy
Social movements
Anarchism
";
        let bonds = extract("Some Topic", article, None);
        let objs: Vec<&str> = bonds.iter().map(|b| b.object.as_str()).collect();
        // Categories
        assert!(objs.contains(&"Anti-capitalism"), "got {:?}", objs);
        assert!(objs.contains(&"Far-left politics"));
        assert!(objs.contains(&"Political philosophy"));
        // Link captions (broken up by URLs into runs of length 1)
        // must NOT be emitted.
        assert!(!objs.iter().any(|o| o.contains("MODIS")));
        assert!(!objs.iter().any(|o| o.contains("discussion")));
        assert!(!objs.iter().any(|o| o.contains("Goddard")));
        for b in &bonds {
            assert_eq!(b.relation, BondType::IsA);
            assert_eq!(b.subject, "Some Topic");
            assert_eq!(b.layer, "L2b");
        }
    }

    #[test]
    fn l2b_no_run_returns_empty() {
        let article = "Plain prose. Nothing heading-shaped here.";
        let bonds = extract("Topic", article, None);
        assert!(bonds.is_empty());
    }

    /// Regression for text-extractor v2 Fix 1 (2026-05-31): even if
    /// a category-run contains a sub-heading like `Primary sources`
    /// it must be dropped via the shared reference-noise set.
    #[test]
    fn l2b_drops_reference_noise_within_category_run() {
        let article = "\
Body text.

Anti-capitalism
Far-left politics
Political philosophy
Primary sources
Tertiary sources
Social movements
";
        let bonds = extract("Topic", article, None);
        let objs: Vec<&str> = bonds.iter().map(|b| b.object.as_str()).collect();
        assert!(objs.contains(&"Anti-capitalism"));
        assert!(objs.contains(&"Far-left politics"));
        assert!(!objs.iter().any(|o| *o == "Primary sources"), "got {:?}", objs);
        assert!(!objs.iter().any(|o| *o == "Tertiary sources"), "got {:?}", objs);
    }
}
