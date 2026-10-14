// OP-GREET-REFUSAL — defense-in-depth deterministic refusal for
// social pleasantries.
//
// Why this exists
// ---------------
// The K-gate (`crate::idk_gate`) operates on Mode A predicate
// `matched_tokens.is_empty()`. As the brain has grown from ~10.8M
// to ~16M bonds, incidental Wikipedia / Seinfeld bonds on words
// like "hello", "goodbye", "thank you" mean the keyword index
// returns hits, the predicate fails, the gate returns Pass, and
// the downstream picks a `Wondering` shape instead of refusing.
// See internal notes (2026-05-28).
//
// Per the investigation's recommendation, we DO NOT change the
// K-gate threshold (that would break legitimate single-noun
// queries like "mitochondria"). Instead, this module sits ABOVE
// the K-gate as a pre-pass: deterministic exact-match against a
// small grep-able table of greetings, thanks, and apologies.
//
// Match strategy
// --------------
// 1. Lowercase the query.
// 2. Trim leading + trailing whitespace.
// 3. Strip trailing ASCII punctuation (.!?,;:).
// 4. Exact match against the table. **No substring matching** —
//    "say hello to the world" must NOT trigger on "hello".
//
// Hook sites
// ----------
// * `narration::narrate` (top, before `IntentClassifier::classify`).
//   Covers `lu4d_narrate` automatically.
// * `mcp::tool_ask` (after `try_arithmetic_response_with`, before
//   recall). `lu4d_ask` doesn't go through `narrate`, so it needs
//   its own hook.
//
// Math precedence
// ---------------
// In `tool_ask` and `tool_narrate`, `try_arithmetic_response_with`
// fires BEFORE the greet check, so a query like "hi 2+2" — were
// it ever classified as math — would still take the math path.
// In practice neither path matches "hi" as math, but the ordering
// invariant is preserved either way.
//
// This module ships zero stubs and never calls `Brain::ask`; the
// refusal is computed purely from the query string.

/// Class of social pleasantry that triggered the refusal.
/// Distinct variants drive distinct canned prose so the user sees
/// an acknowledgement appropriate to what they said.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GreetClass {
    /// Hello / hi / good morning / goodbye / see you / etc.
    /// Both salutations and valedictions live here; the canned
    /// response works for both because it acknowledges the
    /// social signal and redirects to questions.
    Greeting,
    /// Thank you / thanks / appreciate it / you're welcome /
    /// no problem. Symmetric: "you're welcome" and "no problem"
    /// are responses to thanks, but treating them as thanks-class
    /// for refusal purposes keeps the response neutral.
    Thanks,
    /// Sorry / excuse me / pardon. Apologies and polite
    /// interruptions.
    Apology,
}

/// Grep-able table of (normalized phrase, class). Phrases are
/// already lowercase and have no trailing punctuation; the
/// `is_greeting` normalizer reduces queries to the same shape
/// before lookup.
///
/// Ordering does not matter — exact match returns first hit.
/// Keep the list intentionally short and conservative: anything
/// that could plausibly be the start of a real question ("how are
/// you" — could be asked of Lu) is OUT.
static GREET_TABLE: &[(&str, GreetClass)] = &[
    // ---- salutations ----
    ("hello", GreetClass::Greeting),
    ("hi", GreetClass::Greeting),
    ("hey", GreetClass::Greeting),
    ("yo", GreetClass::Greeting),
    ("good morning", GreetClass::Greeting),
    ("good afternoon", GreetClass::Greeting),
    ("good evening", GreetClass::Greeting),
    ("good night", GreetClass::Greeting),
    ("howdy", GreetClass::Greeting),
    ("greetings", GreetClass::Greeting),
    // ---- valedictions (folded into Greeting class) ----
    ("goodbye", GreetClass::Greeting),
    ("bye", GreetClass::Greeting),
    ("bye bye", GreetClass::Greeting),
    ("see you", GreetClass::Greeting),
    ("see ya", GreetClass::Greeting),
    ("farewell", GreetClass::Greeting),
    // ---- thanks ----
    ("thank you", GreetClass::Thanks),
    ("thanks", GreetClass::Thanks),
    ("thx", GreetClass::Thanks),
    ("ty", GreetClass::Thanks),
    ("appreciate it", GreetClass::Thanks),
    ("much appreciated", GreetClass::Thanks),
    ("you're welcome", GreetClass::Thanks),
    ("youre welcome", GreetClass::Thanks),
    ("no problem", GreetClass::Thanks),
    ("no worries", GreetClass::Thanks),
    // ---- apologies ----
    ("sorry", GreetClass::Apology),
    ("excuse me", GreetClass::Apology),
    ("pardon", GreetClass::Apology),
    ("pardon me", GreetClass::Apology),
    ("my bad", GreetClass::Apology),
];

/// Normalize a query for greet-table lookup.
///
/// Steps, in order:
/// 1. Lowercase (ASCII-only fast path is fine; greetings are all
///    ASCII).
/// 2. Trim leading + trailing whitespace.
/// 3. Strip trailing ASCII punctuation `. ! ? , ; :` (repeatedly,
///    so "hello!!!" reduces to "hello").
///
/// Pure function. No allocation beyond the lowercase copy.
fn normalize(query: &str) -> String {
    let lower = query.to_lowercase();
    let trimmed = lower.trim();
    // Strip trailing punctuation. Loop because "hello!!!" should
    // reduce to "hello", and `trim_end_matches` with a single char
    // misses repeats of mixed punctuation like "hi.!".
    let stripped = trimmed.trim_end_matches(['.', '!', '?', ',', ';', ':']);
    stripped.to_string()
}

/// Exact-match the query against the greet table.
///
/// Returns `Some(GreetClass)` when the normalized query is exactly
/// one of the listed phrases; returns `None` otherwise.
///
/// **Does not substring-match.** "say hello to the world" returns
/// `None`. This is intentional: substring matching would catch
/// legitimate compositional queries that happen to contain a
/// greeting word.
pub fn is_greeting(query: &str) -> Option<GreetClass> {
    let norm = normalize(query);
    if norm.is_empty() {
        return None;
    }
    for (phrase, class) in GREET_TABLE {
        if norm == *phrase {
            return Some(*class);
        }
    }
    None
}

/// Canonical refusal prose per class. Deterministic, static, never
/// fabricated. The same input produces the same output every time;
/// no brain access, no recall.
///
/// Tone: brief, acknowledges the social signal, redirects to the
/// memory-anchored question interface. Does NOT pretend
/// understanding of anything beyond the greeting itself.
pub fn refusal_prose(class: GreetClass) -> &'static str {
    match class {
        GreetClass::Greeting => "Hello. I respond to questions about things I have memory of \u{2014} what would you like to know?",
        GreetClass::Thanks => "You're welcome.",
        GreetClass::Apology => "No need to apologize. What would you like to know?",
    }
}

/// Stable refusal-reason tag for the `idk_gate` signal slot in
/// `NarrationResult.signals_used`. Format: `greet_refusal:<class>`.
/// Lowercase, no spaces — easy to grep in logs and dashboards.
pub fn refusal_reason(class: GreetClass) -> &'static str {
    match class {
        GreetClass::Greeting => "greet_refusal:greeting",
        GreetClass::Thanks => "greet_refusal:thanks",
        GreetClass::Apology => "greet_refusal:apology",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- core acceptance: exact match, case + punctuation tolerance ----

    #[test]
    fn matches_hello_bare() {
        assert_eq!(is_greeting("hello"), Some(GreetClass::Greeting));
    }

    #[test]
    fn matches_hello_capitalized_with_bang() {
        assert_eq!(is_greeting("Hello!"), Some(GreetClass::Greeting));
    }

    #[test]
    fn matches_hi_period() {
        assert_eq!(is_greeting("Hi."), Some(GreetClass::Greeting));
    }

    #[test]
    fn matches_thank_you_two_words() {
        assert_eq!(is_greeting("thank you"), Some(GreetClass::Thanks));
    }

    #[test]
    fn matches_thanks_mixed_case() {
        assert_eq!(is_greeting("Thanks!"), Some(GreetClass::Thanks));
    }

    #[test]
    fn matches_sorry_apology() {
        assert_eq!(is_greeting("sorry"), Some(GreetClass::Apology));
    }

    #[test]
    fn matches_excuse_me_apology() {
        assert_eq!(is_greeting("Excuse me?"), Some(GreetClass::Apology));
    }

    #[test]
    fn matches_goodbye_as_greeting_class() {
        // Valedictions are folded into Greeting class so a single
        // canned response covers both ends of the conversation.
        assert_eq!(is_greeting("goodbye"), Some(GreetClass::Greeting));
    }

    // ---- crucial NEGATIVE cases: must not catch real queries ----

    #[test]
    fn does_not_match_substring_in_real_sentence() {
        // The whole point of this port: "hello" as a substring of a
        // longer query MUST NOT trigger the refusal. The K-gate
        // (and downstream narration) handle real queries.
        assert_eq!(is_greeting("say hello to the world"), None);
    }

    #[test]
    fn does_not_match_legitimate_single_noun_query() {
        // Single-noun queries like "mitochondria" are exactly the
        // queries the K-gate threshold lowering would have broken.
        // Greet refusal must leave them alone.
        assert_eq!(is_greeting("what is mitochondria"), None);
        assert_eq!(is_greeting("mitochondria"), None);
    }

    #[test]
    fn does_not_match_question_about_greetings() {
        // "what is hello in spanish" is a real query about the
        // greeting word itself, not a greeting. Must pass through.
        assert_eq!(is_greeting("what is hello in spanish"), None);
    }

    #[test]
    fn does_not_match_empty() {
        assert_eq!(is_greeting(""), None);
        assert_eq!(is_greeting("   "), None);
        assert_eq!(is_greeting("!!!"), None);
    }

    #[test]
    fn does_not_match_how_are_you() {
        // "how are you" is a question that a future Lu could
        // plausibly attempt to answer (about herself / her state).
        // Refusing it deterministically here is wrong; leave it to
        // the regular pipeline.
        assert_eq!(is_greeting("how are you"), None);
    }

    // ---- punctuation edge cases ----

    #[test]
    fn repeated_punctuation_strips() {
        assert_eq!(is_greeting("hello!!!"), Some(GreetClass::Greeting));
        assert_eq!(is_greeting("hi???"), Some(GreetClass::Greeting));
        assert_eq!(is_greeting("thanks."), Some(GreetClass::Thanks));
    }

    #[test]
    fn leading_whitespace_strips() {
        assert_eq!(is_greeting("  hello  "), Some(GreetClass::Greeting));
    }

    // ---- prose / reason API stability ----

    #[test]
    fn refusal_prose_is_deterministic() {
        // Same input, same output, every call.
        let a = refusal_prose(GreetClass::Greeting);
        let b = refusal_prose(GreetClass::Greeting);
        assert_eq!(a, b);
        assert!(!a.is_empty());
    }

    #[test]
    fn refusal_prose_distinct_per_class() {
        let g = refusal_prose(GreetClass::Greeting);
        let t = refusal_prose(GreetClass::Thanks);
        let a = refusal_prose(GreetClass::Apology);
        assert_ne!(g, t);
        assert_ne!(g, a);
        assert_ne!(t, a);
    }

    #[test]
    fn refusal_reason_has_stable_prefix() {
        assert!(refusal_reason(GreetClass::Greeting).starts_with("greet_refusal:"));
        assert!(refusal_reason(GreetClass::Thanks).starts_with("greet_refusal:"));
        assert!(refusal_reason(GreetClass::Apology).starts_with("greet_refusal:"));
    }
}
