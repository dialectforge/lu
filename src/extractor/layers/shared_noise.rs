//! Reference-section subheadings shared between L2 and L2b.
//!
//! These are sub-headings of External Links / References / Notes
//! sections that look like categories/aspects but aren't real
//! section headings or category bonds. They're heading-shaped
//! (capitalised, no terminal punctuation, short) so the L2 and L2b
//! shape detectors pick them up unless explicitly filtered here.
//!
//! Mirrors `_L2B_REFERENCE_NOISE` in `scripts/smart_extractor.py`.
//! Originally lived inside L2b only; extracted to a shared module
//! 2026-05-31 (text-extractor v2 Fix 1) so L2 can drop the same
//! noise from its `HasAspect` emissions — Anarchism's `Explanatory
//! notes`, `Primary sources`, `Tertiary sources` etc were leaking
//! through L2 even though L2b dedups them post-emit.

use std::collections::HashSet;
use std::sync::OnceLock;

pub fn reference_noise() -> &'static HashSet<&'static str> {
    static S: OnceLock<HashSet<&'static str>> = OnceLock::new();
    S.get_or_init(|| {
        [
            "explanatory notes",
            "general and cited sources",
            "primary sources",
            "secondary sources",
            "tertiary sources",
            "citations",
            "bibliography",
            "further reading",
            "external links",
            "references",
            "notes",
            "footnotes",
            "sources",
            "works cited",
            "see also",
            "gallery",
            "general sources",
            "specific sources",
            "general bibliography",
        ]
        .into_iter()
        .collect()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noise_set_contains_canonical_python_entries() {
        // Spot-check exact strings that appeared as L2 noise on the
        // 46KB Anarchism article (2026-05-31 v2 audit).
        let n = reference_noise();
        for s in [
            "explanatory notes",
            "primary sources",
            "secondary sources",
            "tertiary sources",
            "general and cited sources",
            "citations",
            "bibliography",
            "further reading",
        ] {
            assert!(n.contains(s), "missing canonical entry {:?}", s);
        }
    }
}
