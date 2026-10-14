//! Tail-section stripper — removes See Also / References / External
//! Links / Bibliography / Further Reading / Notes / Footnotes /
//! Citations blocks before scanning body text.
//!
//! Anchored on the EARLIEST tail marker (Python `_strip_tail_sections`).

use super::patterns::TAIL_SECTION_MARKERS;

pub fn strip_tail_sections(text: &str) -> String {
    let mut earliest: Option<usize> = None;
    for marker in TAIL_SECTION_MARKERS {
        if let Some(idx) = text.find(marker) {
            if idx > 0 {
                match earliest {
                    None => earliest = Some(idx),
                    Some(e) if idx < e => earliest = Some(idx),
                    _ => {}
                }
            }
        }
    }
    match earliest {
        Some(e) => text[..e].to_string(),
        None => text.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_external_links() {
        let t = "body content here\nExternal links\nlink 1\n";
        let stripped = strip_tail_sections(t);
        assert!(stripped.contains("body content here"));
        assert!(!stripped.contains("link 1"));
    }

    #[test]
    fn picks_earliest_marker() {
        let t = "body\nReferences\nthing\nSee also\nmore\n";
        let stripped = strip_tail_sections(t);
        // References comes first, so the See-also block is also gone.
        assert!(stripped.contains("body"));
        assert!(!stripped.contains("See also"));
        assert!(!stripped.contains("thing"));
    }

    #[test]
    fn no_marker_passthrough() {
        let t = "plain body, no tails";
        assert_eq!(strip_tail_sections(t), t);
    }
}
