//! Port F.2 — English word-number parsing.
//!
//! Accepts cardinal numbers from "zero" up to (and including) integer
//! multiples of "billion". Out of scope for Round 1: ordinals ("first",
//! "third"), fractions ("one half"), point-decimals ("two point five").
//!
//! Strategy: small lookup table + a state machine that accumulates
//! into a `current` bucket below 1000 and folds into `total` when a
//! scale word ("thousand", "million", "billion") arrives. "Hundred"
//! multiplies the current bucket but does NOT flush.
//!
//! Hyphens ("twenty-three") and the connective "and" ("one hundred
//! and twenty") are tolerated.

pub fn parse(s: &str) -> Option<f64> {
    let s = s.trim().to_lowercase();
    if s.is_empty() {
        return None;
    }
    let cleaned: String = s
        .chars()
        .map(|c| if c == '-' { ' ' } else { c })
        .collect();
    let tokens: Vec<&str> = cleaned
        .split_whitespace()
        .filter(|t| *t != "and")
        .collect();
    if tokens.is_empty() {
        return None;
    }
    let mut total: i64 = 0;
    let mut current: i64 = 0;
    let mut consumed = false;
    for t in &tokens {
        match *t {
            "hundred" => {
                current = if current == 0 { 100 } else { current * 100 };
                consumed = true;
            }
            "thousand" => {
                let bucket = if current == 0 { 1 } else { current };
                total += bucket * 1_000;
                current = 0;
                consumed = true;
            }
            "million" => {
                let bucket = if current == 0 { 1 } else { current };
                total += bucket * 1_000_000;
                current = 0;
                consumed = true;
            }
            "billion" => {
                let bucket = if current == 0 { 1 } else { current };
                total += bucket * 1_000_000_000;
                current = 0;
                consumed = true;
            }
            other => match small_value(other) {
                Some(v) => {
                    current += v;
                    consumed = true;
                }
                None => return None,
            },
        }
    }
    if !consumed {
        return None;
    }
    Some((total + current) as f64)
}

fn small_value(t: &str) -> Option<i64> {
    Some(match t {
        "zero" | "naught" | "nought" => 0,
        "one" => 1, "two" => 2, "three" => 3, "four" => 4, "five" => 5,
        "six" => 6, "seven" => 7, "eight" => 8, "nine" => 9, "ten" => 10,
        "eleven" => 11, "twelve" => 12, "thirteen" => 13, "fourteen" => 14,
        "fifteen" => 15, "sixteen" => 16, "seventeen" => 17, "eighteen" => 18,
        "nineteen" => 19,
        "twenty" => 20, "thirty" => 30, "forty" => 40, "fifty" => 50,
        "sixty" => 60, "seventy" => 70, "eighty" => 80, "ninety" => 90,
        _ => return None,
    })
}

/// Convenience wrapper for callers that already have a numeric string
/// in hand: try `f64::parse` first, then fall through to the word-form
/// parser. Returns None if neither path succeeds.
pub fn parse_number_or_word(s: &str) -> Option<f64> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }
    // Reject scientific notation (Round-1 rule, matches expr.rs).
    if !s.bytes().any(|b| b.is_ascii_alphabetic()) {
        let stripped = s.trim_end_matches(|c: char| matches!(c, '?' | '!' | ','));
        if stripped.bytes().any(|b| b == b'e' || b == b'E') {
            return None;
        }
        if let Ok(v) = stripped.parse::<f64>() {
            if !v.is_nan() && !v.is_infinite() {
                return Some(v);
            }
        }
    }
    parse(s)
}

#[cfg(test)]
mod word_num_tests {
    use super::*;

    #[test]
    fn parse_zero_through_nineteen() {
        for (i, w) in [
            "zero","one","two","three","four","five","six","seven","eight",
            "nine","ten","eleven","twelve","thirteen","fourteen","fifteen",
            "sixteen","seventeen","eighteen","nineteen",
        ].iter().enumerate() {
            assert_eq!(parse(w), Some(i as f64), "failed for {}", w);
        }
    }

    #[test]
    fn parse_compound_tens() {
        assert_eq!(parse("twenty"), Some(20.0));
        assert_eq!(parse("twenty three"), Some(23.0));
        assert_eq!(parse("twenty-three"), Some(23.0));
        assert_eq!(parse("forty seven"), Some(47.0));
    }

    #[test]
    fn parse_hundreds() {
        assert_eq!(parse("one hundred"), Some(100.0));
        assert_eq!(parse("one hundred fifty"), Some(150.0));
        assert_eq!(parse("one hundred and fifty"), Some(150.0));
        assert_eq!(parse("three hundred twenty one"), Some(321.0));
    }

    #[test]
    fn parse_thousands() {
        assert_eq!(parse("one thousand"), Some(1000.0));
        assert_eq!(parse("two thousand five hundred"), Some(2500.0));
        assert_eq!(parse("twelve thousand"), Some(12_000.0));
    }

    #[test]
    fn parse_million_billion() {
        assert_eq!(parse("one million"), Some(1_000_000.0));
        assert_eq!(parse("three billion"), Some(3_000_000_000.0));
    }

    #[test]
    fn parse_rejects_garbage() {
        assert_eq!(parse(""), None);
        assert_eq!(parse("foo bar"), None);
        assert_eq!(parse("one foo"), None);
        // "and" alone is not a number.
        assert_eq!(parse("and"), None);
    }

    #[test]
    fn parse_number_or_word_dispatches() {
        assert_eq!(parse_number_or_word("42"), Some(42.0));
        assert_eq!(parse_number_or_word("forty two"), Some(42.0));
        assert_eq!(parse_number_or_word("twelve"), Some(12.0));
        // Scientific notation rejected.
        assert_eq!(parse_number_or_word("1e3"), None);
        // Empty / nonsense.
        assert_eq!(parse_number_or_word("mitochondria"), None);
    }
}
