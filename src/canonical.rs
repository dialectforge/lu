//! OP-CANON-1 -- label canonicalisation for the ingest path.
//!
//! Root cause of the ~52M-node / ~77%-non-concept brain (node-quality
//! census 2026-07-23, internal notes: `Brain::teach` derives a
//! node identity with `Seed::from_label(surface)` -- an XOR-fold over
//! the RAW surface bytes (src/seed.rs). No case-fold, no article strip,
//! no noun-phrase gate. So `Dog`, `dog`, `the dog`, and
//! `"a dark-haired man with a canister"` are all distinct first-class
//! nodes.
//!
//! This module is the "one classifier built once, used three times"
//! foundation from #1331: (1) an ingest-time REJECT/NORMALISE, (2) the
//! read-time frontier down-rank already shipped as
//! `network::non_concept_label`, and (3) the future prune key.
//!
//! NOTE on the relationship to `network::non_concept_label`: that
//! read-time classifier treats an article-led label as junk (it
//! down-ranks "a dark-haired man ..."). The ingest canonicaliser
//! instead STRIPS a leading article and then judges the CONTENT, so
//! "the dog" normalises to "dog" (kept) while "a dark-haired man with a
//! canister" is rejected on its content shape. They share the concept-
//! shape rules but differ on articles by design.
//!
//! It is a PURE function library. It is deliberately NOT wired into
//! `teach()` in this commit: doing so correctly also requires the
//! query-side seed/keyword derivation to canonicalise identically (or
//! new canonical nodes become unrecallable), plus a decision on the
//! ~40M existing non-canonical nodes -- both architectural and coupled
//! to the (gated) prune. See the OP-CANON-1 wiring plan.

/// Leading articles stripped before deriving the canonical key.
const LEADING_ARTICLES: &[&str] = &["a", "an", "the"];

use std::sync::atomic::{AtomicU64, Ordering};

/// OP-RETRAIN-1 -- process-global count of teaches rejected by the concept-
/// admission gate, so rejects stop hiding inside `bonds_landed`.
static CANON_GATE_REJECTIONS: AtomicU64 = AtomicU64::new(0);

/// Record one admission-gate rejection.
pub fn record_canon_gate_rejection() {
    CANON_GATE_REJECTIONS.fetch_add(1, Ordering::Relaxed);
}

/// Total admission-gate rejections since process start.
pub fn canon_gate_rejections_total() -> u64 {
    CANON_GATE_REJECTIONS.load(Ordering::Relaxed)
}

/// Canonical KEY for a raw surface label: the string that should be fed
/// to `Seed::from_label` so surface variants collapse to one node.
///
/// Returns `None` when the label is REJECTED as a non-concept (the bond
/// should be dropped at ingest rather than minted into a junk node).
/// Returns `Some(key)` otherwise: leading article stripped, Unicode-
/// lowercased, whitespace collapsed.
///
/// The DISPLAY surface a caller chooses to store is a separate decision
/// (first-seen surface, title-case, etc.); this function only defines
/// node IDENTITY.
pub fn canonical_key(raw: &str) -> Option<String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }
    let raw_words: Vec<&str> = trimmed.split_whitespace().collect();
    // Strip a single leading article for the CONTENT view (only when
    // something follows it, so a bare "the" is never stripped to empty).
    let start = if raw_words.len() >= 2
        && LEADING_ARTICLES.contains(&raw_words[0].to_lowercase().as_str())
    {
        1
    } else {
        0
    };
    let content = &raw_words[start..];
    if content.is_empty() {
        return None;
    }
    if is_reject_content(content) {
        return None;
    }
    let key = content.join(" ").to_lowercase();
    if key.is_empty() {
        None
    } else {
        Some(key)
    }
}

/// True when a raw label should be REJECTED at ingest (non-concept).
/// Convenience wrapper over the content-shape test after article strip.
pub fn is_reject(raw: &str) -> bool {
    canonical_key(raw).is_none()
}

/// Content-shape reject test, applied to the article-stripped word list.
fn is_reject_content(content: &[&str]) -> bool {
    let n = content.len();
    // Leading-underscore code identifier (e.g. "_dart_getFlipped").
    if n == 1 && content[0].starts_with('_') {
        return true;
    }
    // Bare single function word as a whole label ("For", "Although").
    if n == 1 && is_function_word(&content[0].to_lowercase()) {
        return true;
    }
    // camelCase / hyphenated-code single-token identifiers: reuse the
    // shipped recall-pollution filter so ingest and recall agree -- but
    // SPARE model/unit/product designations (P-40, U-656, LST-23,
    // Videocart-15, J-STD-038): they share the alphanumeric shape but are
    // real entities (OP-CANON-1 FP fix 2026-07-23).
    if n == 1 && crate::network::is_recall_pollution(content[0]) && !is_designation(content[0]) {
        return true;
    }
    // Over-long fragment: real entity titles are rarely >5 content words.
    if n >= 6 {
        return true;
    }
    // >=3-word fragments. OP-RETRAIN-1 over-rejection fix: a lowercase-initial
    // >=3-word phrase is rejected ONLY when it is FRAGMENT-SHAPED. This recovers
    // legit "X of Y" concepts ("theory of relativity", "speed of light", "law of
    // large numbers") that start lowercase but are not sentence fragments, while
    // still rejecting concept-shaped fragments. The dangling function-word tail
    // and the n>=6 / trailing-punct / code-shape rules are unchanged.
    if n >= 3 {
        let case_word = content
            .iter()
            .find(|w| w.chars().next().map(|c| c.is_alphabetic()).unwrap_or(false));
        if let Some(w) = case_word {
            let starts_lower = w.chars().next().map(|c| c.is_lowercase()).unwrap_or(false);
            if starts_lower && looks_like_sentence_fragment(content, &w.to_lowercase()) {
                return true;
            }
        }
        let last = content[n - 1]
            .trim_matches(|c: char| !c.is_alphanumeric())
            .to_lowercase();
        if is_function_word(&last) {
            return true;
        }
    }
    // Trailing sentence punctuation (extractor heading/list leakage).
    let joined = content.join(" ");
    if joined.ends_with(',') || joined.ends_with(':') || joined.ends_with(';') {
        return true;
    }
    false
}

/// A model / unit / product DESIGNATION token such as `P-40`, `U-656`,
/// `LST-23`, `MDC-1200`, `PRS-700`, `IE-11`, `Videocart-15`, `J-STD-038`,
/// `COVID-19`. Shape: an uppercase-led alphabetic block, hyphen-separated
/// groups, and a trailing all-digit group -- a shape real code identifiers
/// do not take (they carry a `.`/`_` or a camelCase transition, none of
/// which a designation has). Used to SPARE such tokens from the
/// `is_recall_pollution` reject so numbered real entities are not killed.
fn is_designation(tok: &str) -> bool {
    if tok.is_empty() || tok.len() > 16 {
        return false;
    }
    if tok.contains('.') || tok.contains('_') {
        return false;
    }
    if !tok
        .chars()
        .next()
        .map(|c| c.is_ascii_uppercase())
        .unwrap_or(false)
    {
        return false;
    }
    // Split on ASCII hyphen and Unicode en-dash.
    let parts: Vec<&str> = tok
        .split(|c| c == '-' || c == '–')
        .filter(|p| !p.is_empty())
        .collect();
    if parts.len() < 2 {
        return false;
    }
    if !parts[parts.len() - 1].chars().all(|c| c.is_ascii_digit()) {
        return false;
    }
    for part in &parts[..parts.len() - 1] {
        if !part.chars().all(|c| c.is_ascii_alphanumeric()) {
            return false;
        }
        if !part
            .chars()
            .next()
            .map(|c| c.is_ascii_uppercase())
            .unwrap_or(false)
        {
            return false;
        }
        let mut prev_lower = false;
        for c in part.chars() {
            if prev_lower && c.is_ascii_uppercase() {
                return false;
            }
            prev_lower = c.is_ascii_lowercase();
        }
    }
    true
}

/// Fragment-shape test for a lowercase-initial >=3-word phrase (OP-RETRAIN-1).
/// `first` is the lowercased effective-first (alphabetic-led) word. True when the
/// phrase looks like a sentence fragment rather than an "X of Y"-style concept:
/// (1) led by a function word or discourse adverb (also/however/to/and/of ...);
/// (2) gerund + determiner ("developing the ..."); (3) an interior finite verb or
/// relativizer (is/was/became/that/which ...); or (4) an indefinite article after
/// the first word.
fn looks_like_sentence_fragment(content: &[&str], first: &str) -> bool {
    if is_function_word(first) || is_discourse_adverb(first) {
        return true;
    }
    if first.ends_with("ing") && content.len() >= 2 && is_determiner(&content[1].to_lowercase()) {
        return true;
    }
    for w in content.iter().skip(1) {
        let wl = w.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase();
        if is_finite_verb_or_relativizer(&wl) || wl == "a" || wl == "an" {
            return true;
        }
    }
    false
}

fn is_discourse_adverb(w: &str) -> bool {
    matches!(
        w,
        "also" | "later" | "therefore" | "thus" | "moreover" | "furthermore"
            | "meanwhile" | "subsequently" | "additionally" | "consequently"
            | "nevertheless" | "nonetheless" | "hence" | "finally" | "initially"
            | "ultimately" | "previously" | "currently" | "recently" | "originally"
            | "eventually" | "similarly" | "likewise" | "instead" | "besides"
            | "otherwise" | "notably" | "arguably" | "reportedly"
    )
}

fn is_determiner(w: &str) -> bool {
    matches!(
        w,
        "the" | "a" | "an" | "this" | "that" | "these" | "those"
            | "its" | "his" | "her" | "their" | "our" | "my" | "your"
    )
}

fn is_finite_verb_or_relativizer(w: &str) -> bool {
    matches!(
        w,
        "is" | "was" | "were" | "are" | "be" | "been" | "being"
            | "became" | "become" | "becomes"
            | "that" | "which" | "who" | "whom" | "whose" | "where" | "when"
            | "has" | "have" | "had"
    )
}

fn is_function_word(w: &str) -> bool {
    matches!(
        w,
        "the" | "a" | "an" | "of" | "to" | "by" | "in" | "on" | "for"
            | "and" | "or" | "with" | "as" | "at" | "from" | "that"
            | "is" | "was" | "were" | "be" | "been" | "are" | "its"
            | "this" | "these" | "those" | "which" | "into" | "than"
            | "then" | "but" | "if" | "so" | "although" | "however"
    )
}

/// FOLD/normalise key for the INGEST path and for merge grouping:
/// case-fold + strip one leading article + collapse whitespace. Unlike
/// [`canonical_key`], this is LOSSLESS -- it never rejects on content
/// shape and returns `None` only for an empty/whitespace label. It
/// decides node IDENTITY at teach time (so `Dog`/`dog`/`the dog`
/// collapse to one node); the content-shape REJECT lives in
/// [`canonical_key`] and is used only by the (gated) offline prune.
pub fn canonical_ingest_key(raw: &str) -> Option<String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }
    let words: Vec<&str> = trimmed.split_whitespace().collect();
    let start = if words.len() >= 2
        && LEADING_ARTICLES.contains(&words[0].to_lowercase().as_str())
    {
        1
    } else {
        0
    };
    let key = words[start..].join(" ").to_lowercase();
    if key.is_empty() {
        None
    } else {
        Some(key)
    }
}

/// The exact string `teach()` feeds to `Seed::from_label` when ingest
/// folding is enabled: the fold key, or the trimmed surface if the label
/// has no fold key (empty after trim -> its own trimmed form, preserving
/// legacy behaviour rather than dropping).
pub fn fold_seed_label(raw: &str) -> String {
    canonical_ingest_key(raw).unwrap_or_else(|| raw.trim().to_string())
}

/// Env gate for the ingest-path canonicalisation wiring in `teach()`.
/// Default OFF: the wiring is present and unit-tested but does not change
/// live ingest identity until `LU_CANON_INGEST=1` (or any value other
/// than `0`/`false`) is set, mirroring the reversible-by-flag pattern of
/// `network::junk_downrank_enabled`. Read per call (uncached) so it is
/// deterministically testable; `teach` is a batch/ingest path, not the
/// serving hot path, so a `getenv` per call is negligible.
pub fn ingest_canon_enabled() -> bool {
    std::env::var("LU_CANON_INGEST")
        .map(|v| v != "0" && !v.eq_ignore_ascii_case("false"))
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {

    #[test]
    fn recovers_legit_multiword_concepts() {
        for s in [
            "theory of relativity", "speed of light", "law of large numbers",
            "conservation of energy", "rule of law", "freedom of speech",
            "balance of power", "theory of everything",
        ] {
            assert!(canonical_key(s).is_some(), "should keep concept: {s}");
        }
    }
    use super::*;

    #[test]
    fn spares_ordinal_led_titles() {
        for s in [
            "518th Fighter-Interceptor Squadron",
            "21st Screen Actors Guild Awards",
            "111th Regiment of Foot",
            "162nd Virginia General Assembly",
            "2nd Maine Cavalry Regiment",
            "10th Independent Spirit Awards",
            "52nd World Science Fiction Convention",
        ] {
            assert!(canonical_key(s).is_some(), "should keep ordinal title: {s}");
        }
    }

    #[test]
    fn spares_numbered_designations() {
        for s in [
            "P-40", "U-656", "LST-23", "MDC-1200", "PRS-700", "IE-11",
            "BGM-075", "J-STD-038", "Videocart-15", "Formula-2000", "COVID-19",
        ] {
            assert!(is_designation(s), "is_designation should hold: {s}");
            assert!(canonical_key(s).is_some(), "should keep designation: {s}");
        }
    }

    #[test]
    fn fp_fix_does_not_reopen_caught_junk() {
        for s in [
            "also a member of the Vietnam national football team",
            "one of the two Transdev operations in the Netherlands",
            "a very suitable platform to be investigated by the",
            "an off-leash dog park located at the township",
            "For",
            "Although",
            "Awards and exhibits,",
            "_dart_getFlipped",
            "getPropType",
            "resolveComponentFactory",
            "handleCrossRibbonClick",
            "goToRecoverPassword",
            "oneMinusBeta1",
        ] {
            assert!(canonical_key(s).is_none(), "should still reject junk: {s}");
        }
    }

    #[test]
    fn is_designation_rejects_code_shapes() {
        for s in [
            "getPropType", "resolveComponentFactory", "RpcLoader.get_asset",
            "index_file.metadata", "_dart_getFlipped", "download_ena_report",
            "SandeshCtrlServerToClient", "surface.free",
        ] {
            assert!(!is_designation(s), "code must not be a designation: {s}");
        }
    }

    #[test]
    fn number_led_lowercase_fragment_still_rejected() {
        assert!(canonical_key("1st to arrive at scene").is_none());
        assert!(canonical_key("3rd and long yardage").is_none());
    }

    #[test]
    fn ingest_key_folds_case_and_article_losslessly() {
        // Same fold key -> same Seed identity for all surface variants.
        assert_eq!(canonical_ingest_key("Dog").as_deref(), Some("dog"));
        assert_eq!(canonical_ingest_key("the dog").as_deref(), Some("dog"));
        assert_eq!(canonical_ingest_key("  The   DOG ").as_deref(), Some("dog"));
        assert_eq!(canonical_ingest_key("a Dog").as_deref(), Some("dog"));
    }

    #[test]
    fn ingest_key_never_rejects_on_shape_only_on_empty() {
        // Content-shape reject is NOT the ingest key's job: a long
        // article-led fragment still yields a (lossless) fold key.
        assert!(canonical_ingest_key("a dark-haired man with a canister").is_some());
        assert!(canonical_key("a dark-haired man with a canister").is_none());
        // Only empty / whitespace yields None.
        assert!(canonical_ingest_key("   ").is_none());
        assert!(canonical_ingest_key("").is_none());
    }

    #[test]
    fn fold_seed_label_collapses_variants() {
        assert_eq!(fold_seed_label("Dog"), fold_seed_label("the dog"));
        assert_eq!(fold_seed_label("Dog"), fold_seed_label("DOG"));
        assert_eq!(fold_seed_label("Dog"), "dog");
    }

    #[test]
    fn collapses_case_and_articles() {
        assert_eq!(canonical_key("Dog").as_deref(), Some("dog"));
        assert_eq!(canonical_key("dog").as_deref(), Some("dog"));
        assert_eq!(canonical_key("the dog").as_deref(), Some("dog"));
        assert_eq!(canonical_key("  The   Dog ").as_deref(), Some("dog"));
        // All four collapse to the same key -> one node.
        let k = canonical_key("Dog").unwrap();
        assert_eq!(canonical_key("the dog").unwrap(), k);
    }

    #[test]
    fn keeps_multiword_entities_but_normalises() {
        assert_eq!(canonical_key("Domestic Dog").as_deref(), Some("domestic dog"));
        assert_eq!(canonical_key("Canis familiaris").as_deref(), Some("canis familiaris"));
        // Article stripped, rest preserved.
        assert_eq!(canonical_key("The United States").as_deref(), Some("united states"));
    }

    #[test]
    fn rejects_the_census_junk_classes() {
        assert!(canonical_key("a dark-haired man with a canister").is_none());
        assert!(canonical_key("an off-leash dog park located at the township").is_none());
        assert!(canonical_key("a very suitable platform to be investigated by the").is_none());
        assert!(canonical_key("_dart_getFlipped").is_none());
        assert!(canonical_key("For").is_none());
        assert!(canonical_key("Although").is_none());
        assert!(canonical_key("Awards and exhibits,").is_none());
    }

    #[test]
    fn spares_real_concepts() {
        assert!(canonical_key("Dog").is_some());
        assert!(canonical_key("Photosynthesis").is_some());
        assert!(canonical_key("Python").is_some());
        assert!(canonical_key("Canis familiaris (domestic dog)").is_some());
    }

    #[test]
    fn never_strips_to_empty() {
        // A bare article is a single-word function word -> rejected,
        // never returned as an empty key.
        assert!(canonical_key("the").is_none());
        assert!(canonical_key("a").is_none());
    }
}
