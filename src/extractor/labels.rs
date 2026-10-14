//! Label cleaning and noun-phrase shaping helpers.
//!
//! Mirrors the Python `_clean`, `_ok_term`, `_trim_object`,
//! `_truncate_to_np`, and the `_NP_*` regex constants from
//! `scripts/smart_extractor.py` + `scripts/ingest_curriculum_full.py`.

use regex::Regex;
use std::sync::OnceLock;

use super::patterns::{
    np_leading_verb_re, np_predicate_verb_re, np_stop_conjunction_re,
    np_stop_punct_re, stop_objects, trailing_junk_re,
};

fn punct_tail_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"[\s\.,;:!?\-]+$").unwrap())
}

fn ws_collapse_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"\s+").unwrap())
}

/// Strip whitespace, trailing punctuation, and collapse internal
/// whitespace runs. Matches Python `_clean`.
pub fn clean_label(s: &str) -> String {
    let s = s.trim();
    if s.is_empty() {
        return String::new();
    }
    let s = punct_tail_re().replace(s, "").into_owned();
    let s = ws_collapse_re().replace_all(&s, " ").into_owned();
    s
}

/// Accept-list filter on a single label. Matches Python `_ok_term`.
pub fn ok_term(s: &str, min_len: usize, max_len: usize) -> bool {
    if s.is_empty() {
        return false;
    }
    let n = s.chars().count();
    if n < min_len || n > max_len {
        return false;
    }
    if stop_objects().contains(s.to_lowercase().as_str()) {
        return false;
    }
    let alpha = s.chars().filter(|c| c.is_alphabetic()).count();
    let threshold = std::cmp::max(2usize, n / 2);
    if alpha < threshold {
        return false;
    }
    true
}

/// Strip a trailing dangling connector / preposition until none
/// remains, then clean the result. Mirrors Python `_trim_object`.
pub fn trim_object(s: &str) -> String {
    let mut cur = s.to_string();
    loop {
        let new = trailing_junk_re().replace(&cur, "").trim().to_string();
        if new == cur {
            break;
        }
        cur = new;
    }
    clean_label(&cur)
}

/// Truncate a raw text fragment to a clean noun phrase.
/// Returns None when the result still looks like a clause fragment
/// (contains a predicate verb or leads with a verb form). Mirrors
/// Python `_truncate_to_np` from `ingest_curriculum_full.py`.
pub fn truncate_to_np(text: &str, max_len: usize) -> Option<String> {
    if text.is_empty() {
        return None;
    }
    let trimmed: String = text
        .trim()
        .trim_matches(|c| matches!(c, ',' | ';' | ':' | '—' | '–'))
        .to_string();
    if trimmed.is_empty() {
        return None;
    }
    // 1. Cut at first sentence-internal punctuation.
    let mut cur = if let Some(m) = np_stop_punct_re().find(&trimmed) {
        trimmed[..m.start()].trim().to_string()
    } else {
        trimmed.clone()
    };
    if cur.is_empty() {
        return None;
    }
    // 2. Cut at first subordinating conjunction.
    if let Some(m) = np_stop_conjunction_re().find(&cur) {
        cur = cur[..m.start()].trim().to_string();
    }
    if cur.is_empty() {
        return None;
    }
    // 3. Hard cap with word-boundary alignment.
    if cur.chars().count() > max_len {
        let byte_cap = max_len.min(cur.len());
        let mut cut_end = byte_cap;
        while cut_end > 0 && !cur.is_char_boundary(cut_end) {
            cut_end -= 1;
        }
        let cut = cur[..cut_end].trim_end().to_string();
        if let Some(last_space) = cut.rfind(' ') {
            if last_space > (max_len as f32 * 0.5) as usize {
                cur = cut[..last_space].trim().to_string();
            } else {
                cur = cut;
            }
        } else {
            cur = cut;
        }
    }
    if cur.chars().count() < 2 {
        return None;
    }
    // 4. Reject if still contains a finite predicate verb.
    if np_predicate_verb_re().is_match(&cur) {
        return None;
    }
    // 5. Reject if starts with a verb form (past participle / gerund).
    if np_leading_verb_re().is_match(&cur) {
        return None;
    }
    Some(cur)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clean_label_strips_trailing_punct() {
        assert_eq!(clean_label("  Hello world.  "), "Hello world");
        assert_eq!(clean_label("  spaces   between  "), "spaces between");
        assert_eq!(clean_label(""), "");
    }

    #[test]
    fn ok_term_rejects_stopwords_and_short() {
        assert!(!ok_term("", 2, 80));
        assert!(!ok_term("a", 2, 80));
        assert!(!ok_term("the", 2, 80));
        assert!(!ok_term("123", 2, 80));
        assert!(ok_term("thing", 2, 80));
        assert!(!ok_term(&"x".repeat(100), 2, 80));
    }

    #[test]
    fn trim_object_strips_dangling() {
        assert_eq!(trim_object("a member of the"), "a member");
        assert_eq!(trim_object("a state which"), "a state");
        assert_eq!(trim_object("clean"), "clean");
    }

    /// Sanity port of the 13 hand-rolled cases in
    /// /tmp/test_np_truncation.py on the server.
    #[test]
    fn truncate_to_np_audit_cases() {
        // Case 1 — cuts at first comma.
        assert_eq!(
            truncate_to_np(
                "an incredibly important aspect of the film, as the conversations that Fiodor has between the other characters always seem to mention the Spanish Civil War, the Popular Front and notions of communism",
                60,
            )
            .as_deref(),
            Some("an incredibly important aspect of the film"),
        );

        // Case 2 — verb-led, rejected.
        assert!(truncate_to_np(
            "judged Frederick harshly because of this episode but there were very severe pressures on him at the time",
            60,
        )
        .is_none());

        // Case 3 — gerund-led, rejected.
        assert!(truncate_to_np("moving away from it at astounding speed", 60).is_none());

        // Case 4 — clean NP passes through.
        assert_eq!(
            truncate_to_np("political philosophy", 60).as_deref(),
            Some("political philosophy"),
        );

        // Case 5 — cuts at first comma.
        assert_eq!(
            truncate_to_np(
                "Spanish Civil War, the Popular Front and notions of communism",
                60,
            )
            .as_deref(),
            Some("Spanish Civil War"),
        );

        // Case 6 — "that" subordinator truncates to "a state";
        // the remainder has no predicate verb / verb-led head,
        // so it passes. (Python source-of-truth comment in
        // /tmp/test_np_truncation.py was a hand-written expectation
        // that did not match the actual implementation.)
        assert_eq!(
            truncate_to_np("a state that was established in 1801", 60).as_deref(),
            Some("a state"),
        );

        // Case 7 — empty input.
        assert!(truncate_to_np("", 60).is_none());

        // Case 8 — single char rejected (< 2).
        assert!(truncate_to_np("a", 60).is_none());

        // Case 9 — single-word valid NP.
        assert_eq!(
            truncate_to_np("reflectance", 60).as_deref(),
            Some("reflectance"),
        );

        // Case 10 — proper NP passes.
        assert_eq!(
            truncate_to_np("the Spanish Civil War", 60).as_deref(),
            Some("the Spanish Civil War"),
        );

        // Case 11 — `that` subordinator cuts; remainder has no
        // predicate verb, no verb-led start; should pass.
        let got = truncate_to_np(
            "a member of the political philosophy that rejects all forms of authority",
            60,
        );
        assert_eq!(got.as_deref(), Some("a member of the political philosophy"));
    }
}
