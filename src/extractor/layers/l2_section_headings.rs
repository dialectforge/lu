//! L2 — Section heading walker. Emits `(title, HasAspect, heading)`
//! bonds for each heading-shaped line that follows an empty line.
//!
//! Python source of truth: `extract_section_headings` in
//! `smart_extractor.py`.
//!
//! Noise filter (added 2026-05-31 — text-extractor v2 Fix 1):
//! drops reference-subsection labels (`Explanatory notes`,
//! `Primary sources`, etc.) that pass the heading-shape gate but
//! are sub-headings of References/Notes/External-Links sections.
//! Same `_L2B_REFERENCE_NOISE` set used by L2b; sourced from
//! `extractor/layers/shared_noise.rs` so the two layers stay in
//! lockstep.

use crate::bond::BondType;
use crate::extractor::labels::{clean_label, ok_term};
use crate::extractor::layers::shared_noise::reference_noise;
use crate::extractor::patterns::{drop_headings, heading_re};
use crate::extractor::types::{kind_for, Bond, ConfidenceLabel};

pub fn extract(title: &str, text: &str, source_url: Option<&str>) -> Vec<Bond> {
    let mut bonds = Vec::new();
    if title.is_empty() || text.is_empty() {
        return bonds;
    }
    let lines: Vec<&str> = text.split('\n').collect();
    let title_lower = title.to_lowercase();
    for (i, line) in lines.iter().enumerate() {
        let line = line.trim_end();
        if !heading_re().is_match(line) {
            continue;
        }
        let head = clean_label(line);
        if !ok_term(&head, 4, 79) {
            continue;
        }
        let hl = head.to_lowercase();
        if drop_headings().contains(hl.as_str()) {
            continue;
        }
        if reference_noise().contains(hl.as_str()) {
            continue;
        }
        // Heading must be preceded by a blank line (or be the first line).
        let prev_empty = i == 0 || lines[i - 1].trim().is_empty();
        if !prev_empty {
            continue;
        }
        if hl == title_lower {
            continue;
        }
        bonds.push(Bond {
            subject: title.to_string(),
            relation: BondType::HasAspect,
            object: head,
            kind: kind_for(BondType::HasAspect),
            weight: 0.5,
            confidence: ConfidenceLabel::Unverified,
            layer: "L2",
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
    fn l2_extracts_section_headings() {
        let article = "\
Introduction text here.

History

The history starts here.

Culture

Culture text.

See also

Foo
";
        let bonds = extract("Topic", article, None);
        let objects: Vec<&str> = bonds.iter().map(|b| b.object.as_str()).collect();
        assert!(objects.contains(&"History"));
        assert!(objects.contains(&"Culture"));
        // See also is in the drop set
        assert!(!objects.contains(&"See also"));
        for b in &bonds {
            assert_eq!(b.relation, BondType::HasAspect);
            assert_eq!(b.layer, "L2");
        }
    }

    #[test]
    fn l2_requires_preceding_blank() {
        // "Inline Heading" appears mid-paragraph with NO preceding
        // blank line. It must NOT be emitted as a section heading.
        let article = "First sentence here.\nInline Heading\nMore body line";
        let bonds = extract("X", article, None);
        assert!(
            !bonds.iter().any(|b| b.object == "Inline Heading"),
            "unexpected inline heading bond: {:?}",
            bonds,
        );
    }

    /// Regression for text-extractor v2 Fix 1 (2026-05-31): the
    /// reference-subsection noise tokens MUST be dropped even when
    /// they pass the heading-shape detector. These appeared as
    /// `HasAspect` bonds on the 46KB Anarchism article before the
    /// shared noise set was applied to L2.
    #[test]
    fn l2_drops_reference_subsection_noise() {
        let article = "\
Lead paragraph.

History

Body.

References

Explanatory notes

The notes.

Citations

The cites.

Primary sources

Foo.

Secondary sources

Bar.

Tertiary sources

Baz.

General and cited sources

Qux.

Further reading

More.
";
        let bonds = extract("Topic", article, None);
        let objs: Vec<&str> = bonds.iter().map(|b| b.object.as_str()).collect();
        // History is a real section, must survive.
        assert!(objs.contains(&"History"), "got {:?}", objs);
        // Reference noise — all must be dropped.
        for noise in [
            "Explanatory notes",
            "Citations",
            "Primary sources",
            "Secondary sources",
            "Tertiary sources",
            "General and cited sources",
            "Further reading",
        ] {
            assert!(
                !objs.contains(&noise),
                "reference-noise leaked through L2: {:?} in {:?}",
                noise, objs,
            );
        }
    }
}
