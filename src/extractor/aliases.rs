//! Subject alias helpers — Python `_build_subject_aliases`,
//! `_sentence_anchors_subject`.

use std::collections::HashSet;

use super::patterns::capital_word_re;

/// Build the set of acceptable subject anchor strings for an article
/// title. Includes the full title plus every capital word of length
/// ≥ 4. Matches Python `_build_subject_aliases`.
pub fn build_subject_aliases(title: &str) -> HashSet<String> {
    let mut out: HashSet<String> = HashSet::new();
    if title.is_empty() {
        return out;
    }
    out.insert(title.to_string());
    for m in capital_word_re().find_iter(title) {
        let w = m.as_str();
        if w.chars().count() >= 4 {
            out.insert(w.to_string());
        }
    }
    out
}

/// True if `sentence` contains any alias from `aliases`. Case-
/// sensitive substring match (aliases are already capitalised).
/// Matches Python `_sentence_anchors_subject`.
pub fn sentence_anchors_subject(sentence: &str, aliases: &HashSet<String>) -> bool {
    if aliases.is_empty() {
        return false;
    }
    for alias in aliases {
        if !alias.is_empty() && sentence.contains(alias.as_str()) {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aliases_two_word_title() {
        let a = build_subject_aliases("Abraham Lincoln");
        assert!(a.contains("Abraham Lincoln"));
        assert!(a.contains("Abraham"));
        assert!(a.contains("Lincoln"));
    }

    #[test]
    fn aliases_short_title() {
        let a = build_subject_aliases("ABC");
        assert!(a.contains("ABC"));
        // "ABC" is 3 chars < 4 — not added as an alias from the
        // capital_word pass.
        assert_eq!(a.len(), 1);
    }

    #[test]
    fn sentence_anchors_works() {
        let a = build_subject_aliases("Abraham Lincoln");
        assert!(sentence_anchors_subject("Abraham Lincoln was born.", &a));
        assert!(sentence_anchors_subject("Lincoln moved to Springfield.", &a));
        assert!(!sentence_anchors_subject("His mother was born in Virginia.", &a));
    }
}
