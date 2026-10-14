//! OP-FLUENT (Phase 4b) -- grounded sentence-planning / discourse
//! composition over the grounded ContentPlan (design section 11,
//! option (b): multi-sentence discourse composition + subject
//! continuity, building on Phase 4a's aggregation/template realizer).
//! ANSWER-SIDE ONLY.
//!
//! It is called ONCE from `narration::narrate_inner`, on the synthesized
//! [`Conclusion`], BEFORE `epistemic::recast`. Ordering is deliberate:
//! this pass rewrites the ASSERTIVE body (a discourse plan over the
//! cited edges -> ordered, multi-sentence prose with connectives,
//! subject continuity, and the tracked cosmetics); the Phase-3 epistemic
//! register calibration then runs LAST and remains the authoritative
//! voice -- fluent layers UNDER it, never overrides it.
//!
//! THE PARAMOUNT CONSTRAINT (honesty, design section 8/11): this planner
//! may only ORDER and PHRASE the closed set of grounded propositions
//! ALREADY in `Conclusion.cited_edges`. It introduces NO claim, entity,
//! relation, or "connective fact" not in that plan. Discourse
//! connectives / transitions ("it", "also", "and") are FUNCTION WORDS
//! only. It rebuilds the body ONLY from cited-edge endpoint labels + a
//! CLOSED, per-BondType lexicalization vocabulary + pure function words.
//! It NEVER gathers new edges or expands the cited set: the grounded-fact
//! set for any query is byte-identical to Phase 4a (only ordering /
//! phrasing changed). A post-realization AUDIT re-tokenizes the rebuilt
//! body: every content token must be a function word OR a lexicalization
//! of a cited-edge proposition. If ANY ungrounded content token appears,
//! the rebuilt body is DISCARDED and the Phase-4a/templated prose is
//! returned (degrade to the safe form). Fluency that could fabricate is
//! worse than templated honesty -- when in doubt, degrade.
//!
//! Subject continuity across sentences NEVER guesses a gendered pronoun:
//! the composer only ever re-mentions the grounded subject label (a
//! surname for entities) or uses the neutral pronoun "it" (concepts).
//! Gender is not carried by any cited edge, so asserting "he"/"she" would
//! be an ungrounded claim -- the composer simply never constructs one.
//! (Such pronouns are in any case dropped by the shared stopword filter,
//! so they are neither introduced nor needed.)
//!
//! HONESTY INVARIANTS enforced here:
//!   * never touches `cited_edges` -- the audit trail is preserved.
//!   * never touches a refusal / single-fact plan (len < 2) -- those keep
//!     their template (and their yes/no opener for epistemic to calibrate).
//!   * aggregation, sectioning, and the property-list cleanup only ever
//!     DROP or REORDER already-grounded material; they never ADD a
//!     proposition.

use std::collections::HashSet;

use crate::bond::BondType;
use crate::synthesis::{CitedEdge, Conclusion, ReasoningKind};

// ---------------------------------------------------------------------
// Tokeniser (identical filters to synthesis::content_tokens / recall):
// hub-stopwords + query-noise dropped, length >= 2. Used by the audit so
// "is", "the", "and" never count as ungrounded operands.
// ---------------------------------------------------------------------
fn content_tokens(s: &str) -> Vec<String> {
    s.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| {
            w.len() >= 2
                && !crate::network::is_hub_stopword(w)
                && !crate::network::is_query_noise(w)
        })
        .map(|w| w.to_string())
        .collect()
}

/// Pure function words the realizer itself may emit (auxiliaries /
/// coordinators / neutral pronouns that carry NO claim). Bond-specific
/// content lexemes are NOT here -- they are scoped to the bond via
/// [`bond_lexemes`] so the audit stays tight (e.g. "born" is only allowed
/// when a BornIn edge is actually cited). NOTE: gendered pronouns are
/// deliberately absent -- guessing "he"/"she" is an ungrounded claim, and
/// their absence here means the audit rejects any such guess.
const REALIZER_FUNCWORDS: &[&str] = &[
    "was", "were", "is", "are", "and", "or", "for", "in", "of", "to",
    "the", "that", "this", "as", "well", "both", "also", "among", "others",
    // neutral subject-continuity pronouns (concepts / things only):
    "it", "they", "them", "its", "their",
];

/// The CLOSED lexicalization vocabulary for a bond: the only content
/// words the realizer may use to VOICE that relation. Everything else in
/// a rebuilt clause must be an endpoint label word. Keeping this closed +
/// bond-scoped is what makes the audit a real fabrication guard.
fn bond_lexemes(bt: BondType) -> &'static [&'static str] {
    use BondType::*;
    match bt {
        IsA | DefinedAs | Inherits | ExampleOf => &["kind"],
        KnownFor => &["known", "recognized", "noted"],
        BornIn | BornOn => &["born"],
        DiedIn | DiedOn => &["died"],
        Causes | LeadsTo => &["causes", "cause", "caused", "contributing", "factor", "factors"],
        PartOf | ComposedOf | DerivedFrom => &["made", "part", "composed", "derived"],
        UsedFor | PurposeOf => &["used", "purpose"],
        LocatedIn => &["located"],
        HasCapital => &["capital"],
        HasOfficialLanguage => &["official", "language", "languages"],
        HasCurrency => &["currency"],
        // HasProperty/HasAspect/HasAttribute: the property IS the endpoint
        // label; no extra verb lexeme needed.
        _ => &[],
    }
}

/// The set of content tokens the rebuilt body is ALLOWED to contain:
/// every cited-edge endpoint label word, plus the bond-scoped lexemes for
/// the bonds actually cited, plus pure function words.
fn grounded_lexicon(edges: &[CitedEdge]) -> HashSet<String> {
    let mut lex: HashSet<String> = HashSet::new();
    for e in edges {
        for t in content_tokens(&e.from) {
            lex.insert(t);
        }
        for t in content_tokens(&e.to) {
            lex.insert(t);
        }
        for w in bond_lexemes(e.bond) {
            lex.insert((*w).to_string());
        }
    }
    for w in REALIZER_FUNCWORDS {
        lex.insert((*w).to_string());
    }
    lex
}

/// THE AUDIT. Every content token of the rebuilt body must be grounded.
/// One ungrounded token -> reject the whole rebuild (caller degrades to
/// the templated prose).
fn audit(body: &str, edges: &[CitedEdge]) -> bool {
    let lex = grounded_lexicon(edges);
    content_tokens(body).iter().all(|t| lex.contains(t))
}

// ---------------------------------------------------------------------
// Cosmetics: a/an article agreement ("an elephant", not "a elephant").
// ---------------------------------------------------------------------
fn indefinite_article(word: &str) -> &'static str {
    let w = word.trim().to_ascii_lowercase();
    // "u"-as-"you" / "eu"-as-"you" words take "a" despite the vowel.
    for p in ["uni", "use", "eu", "one", "ubi"] {
        if w.starts_with(p) {
            return "a";
        }
    }
    // silent-h words take "an".
    for p in ["hour", "honest", "heir", "honor"] {
        if w.starts_with(p) {
            return "an";
        }
    }
    match w.chars().next() {
        Some('a') | Some('e') | Some('i') | Some('o') | Some('u') => "an",
        _ => "a",
    }
}

fn cap_first(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        None => String::new(),
        Some(f) => f.to_uppercase().chain(c).collect(),
    }
}

/// Natural coordinated list: "a", "a and b", "a, b, and c".
fn join_and(items: &[String]) -> String {
    match items.len() {
        0 => String::new(),
        1 => items[0].clone(),
        2 => format!("{} and {}", items[0], items[1]),
        _ => format!("{}, and {}", items[..items.len() - 1].join(", "), items[items.len() - 1]),
    }
}

/// Deterministic small index from a seed string -- connective variety
/// that is stable per subject (so the SAME question always reads the
/// same way, but different subjects vary).
fn variety_pick(seed: &str, n: usize) -> usize {
    if n == 0 {
        return 0;
    }
    let h: u32 = seed.bytes().fold(2166136261u32, |a, b| (a ^ b as u32).wrapping_mul(16777619));
    (h as usize) % n
}

/// Subject reference for a CONTINUATION sentence, chosen so no ungrounded
/// claim is introduced. For a multi-token label whose last token is
/// capitalised (a proper surname, e.g. "Albert Einstein" -> "Einstein")
/// we re-mention the surname; the surname is a content word of the
/// original label, so it stays inside the grounded lexicon and is NOT a
/// gendered-pronoun guess. Everything else re-mentions the full label.
fn continuation_subject(name: &str) -> String {
    let toks: Vec<&str> = name.split_whitespace().collect();
    if toks.len() > 1 {
        let last = toks[toks.len() - 1];
        if last.chars().next().map_or(false, |c| c.is_uppercase()) {
            return last.to_string();
        }
    }
    name.to_string()
}

// ---------------------------------------------------------------------
// Property-list cleanup (design cosmetic (d)). CONSERVATIVE + honesty-
// safe: it only ever DROPS, never adds. Worst case understates.
// ---------------------------------------------------------------------

/// Known low-quality descriptor heads surfaced by the property loader
/// (subjective / non-adjectival noise, e.g. the design-called-out
/// "beautiful", "big eyes"). Matched as whole words, case-insensitive.
const NOISE_WORDS: &[&str] = &[
    "beautiful", "baby", "dead", "eyes", "friend", "colors", "colours",
    "species", "things", "stuff", "thing",
];

fn is_noise(prop: &str) -> bool {
    let low = prop.to_ascii_lowercase();
    let words: Vec<&str> = low.split_whitespace().collect();
    // Drop multi-word noun phrases longer than two tokens ("different
    // colors" is 2 and handled by NOISE_WORDS; ">2" catches longer noise).
    if words.len() > 2 {
        return true;
    }
    words.iter().any(|w| NOISE_WORDS.contains(w))
}

/// Dedupe (case-insensitive), drop word-superstrings (keep the shorter of
/// a prefix family, e.g. "different" over "different species"), drop
/// noise, cap for readability. Returns the surviving property labels in
/// their original order/case.
fn clean_props(names: &[String]) -> Vec<String> {
    const CAP: usize = 6;
    let mut out: Vec<String> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    for n in names {
        let t = n.trim();
        if t.is_empty() || is_noise(t) {
            continue;
        }
        let low = t.to_ascii_lowercase();
        if !seen.insert(low.clone()) {
            continue;
        }
        // Word-superstring test against already-kept entries.
        let lw: Vec<&str> = low.split_whitespace().collect();
        let is_super = out.iter().any(|k| {
            let kl = k.to_ascii_lowercase();
            let kw: Vec<&str> = kl.split_whitespace().collect();
            kw.len() < lw.len() && lw[..kw.len()] == kw[..]
        });
        if is_super {
            continue;
        }
        out.push(t.to_string());
        if out.len() >= CAP {
            break;
        }
    }
    out
}

// ---------------------------------------------------------------------
// Shape realizers -- the discourse planners. Each returns Some(rebuilt
// multi-sentence body) or None (degrade). Composition is strictly over
// the cited edges: sectioning REORDERS, chunking only splits sentence
// boundaries. No new proposition is ever synthesised.
// ---------------------------------------------------------------------

/// EntityFact Who/WhatIs/KnownFor: order the cited edges into a small
/// discourse plan -- IDENTITY (IsA) + NOTABLE (KnownFor) as the opening
/// sentence, then ORIGIN (BornIn / LocatedIn) as a second sentence with
/// subject continuity (surname re-mention, never a gendered pronoun).
/// A single-section entity collapses to one sentence (Phase-4a shape).
fn fluent_entity(c: &Conclusion) -> Option<String> {
    let edges = &c.cited_edges;
    let name = edges.first()?.from.clone();
    if name.is_empty() {
        return None;
    }
    let mut isa: Option<&str> = None;
    let mut kf: Vec<String> = Vec::new();
    // OP-OVERVIEW: birth/death carry a PLACE (BornIn/DiedIn) and, when the
    // source pull provides it, a YEAR (BornOn/DiedOn). Both are cited, so the
    // composer may weave "born in Ulm in 1879" -- the year is the grounded
    // BornOn endpoint label, never a guess.
    let mut born_place: Option<&str> = None;
    let mut born_year: Option<&str> = None;
    let mut died_place: Option<&str> = None;
    let mut died_year: Option<&str> = None;
    // OP-OVERVIEW: place-of-subject attributes, woven ONLY when a genuine
    // place signal (capital / official language / currency) is cited, so a
    // person node never triggers the place clause.
    let mut located: Option<&str> = None;
    let mut capital: Option<&str> = None;
    let mut languages: Vec<String> = Vec::new();
    let mut currency: Option<&str> = None;
    for e in edges {
        match e.bond {
            BondType::IsA | BondType::DefinedAs | BondType::ExampleOf => {
                if isa.is_none() {
                    isa = Some(&e.to);
                }
            }
            BondType::KnownFor => kf.push(e.to.clone()),
            BondType::BornIn => {
                if born_place.is_none() {
                    born_place = Some(&e.to);
                }
            }
            BondType::BornOn => {
                if born_year.is_none() {
                    born_year = Some(&e.to);
                }
            }
            BondType::DiedIn => {
                if died_place.is_none() {
                    died_place = Some(&e.to);
                }
            }
            BondType::DiedOn => {
                if died_year.is_none() {
                    died_year = Some(&e.to);
                }
            }
            BondType::LocatedIn => {
                if located.is_none() {
                    located = Some(&e.to);
                }
            }
            BondType::HasCapital => {
                if capital.is_none() {
                    capital = Some(&e.to);
                }
            }
            BondType::HasOfficialLanguage => languages.push(e.to.clone()),
            BondType::HasCurrency => {
                if currency.is_none() {
                    currency = Some(&e.to);
                }
            }
            _ => {}
        }
    }
    // A DiedIn/DiedOn edge is the ONLY grounded deceased signal (past tense);
    // absent => present tense, never a guess about aliveness.
    let deceased = died_place.is_some() || died_year.is_some();

    // --- Sentence 1: IDENTITY (+ coordinated NOTABLE known-for). ------
    let mut s1parts: Vec<String> = Vec::new();
    let be = if deceased { "was" } else { "is" };
    if let Some(cat) = isa {
        s1parts.push(format!("{} {} {}", be, indefinite_article(cat), cat));
    }
    if !kf.is_empty() {
        // Show up to KF_CAP known-for items; if MORE are cited, trail off
        // honestly with "among others" rather than dumping the whole list.
        const KF_CAP: usize = 3;
        let shown: Vec<String> = kf.iter().take(KF_CAP).cloned().collect();
        let mut kf_phrase = join_and(&shown);
        if kf.len() > KF_CAP {
            kf_phrase.push_str(", among others");
        }
        let verb = ["known for", "recognized for", "noted for"][variety_pick(&name, 3)];
        // If there is no IsA lead, the KnownFor clause needs its own copula.
        let lead = if isa.is_some() { verb.to_string() } else { format!("{} {}", be, verb) };
        s1parts.push(format!("{} {}", lead, kf_phrase));
    }

    // --- Sentence 2: ORIGIN (birth / death: place [+ year]), subject
    // continuity. The subject re-mention leads sentence 2 when sentence 1
    // exists; otherwise it carries the full name.
    let s1_present = !s1parts.is_empty();
    let subj2 = if s1_present {
        continuation_subject(&name)
    } else {
        name.clone()
    };
    let born_clause: Option<String> = match (born_place, born_year) {
        (Some(p), Some(y)) => Some(format!("born in {} in {}", p, y)),
        (Some(p), None) => Some(format!("born in {}", p)),
        (None, Some(y)) => Some(format!("born in {}", y)),
        (None, None) => None,
    };
    let died_clause: Option<String> = match (died_place, died_year) {
        (Some(p), Some(y)) => Some(format!("died in {} in {}", p, y)),
        (Some(p), None) => Some(format!("died in {}", p)),
        (None, Some(y)) => Some(format!("died in {}", y)),
        (None, None) => None,
    };
    let s2 = match (born_clause, died_clause) {
        (Some(b), Some(d)) => Some(format!("{} was {} and {}.", subj2, b, d)),
        (Some(b), None) => Some(format!("{} was {}.", subj2, b)),
        (None, Some(d)) => Some(format!("{} {}.", subj2, d)),
        (None, None) => None,
    };

    // --- PLACE attributes: located / capital / official language(s) /
    // currency. Voiced ONLY when a real place signal is cited (capital,
    // language, or currency), so a person node with an incidental LocatedIn
    // never triggers it. "it"/"its" are neutral function-word pronouns.
    let place_signal = capital.is_some() || !languages.is_empty() || currency.is_some();
    let loc_sentence: Option<String> = if place_signal {
        located.map(|l| format!("{} is located in {}.", subj2, l))
    } else {
        None
    };
    let attr_sentence: Option<String> = if place_signal {
        let mut parts: Vec<String> = Vec::new();
        if let Some(cap) = capital {
            parts.push(format!("its capital is {}", cap));
        }
        if !languages.is_empty() {
            let (noun, verb) = if languages.len() > 1 {
                ("official languages", "are")
            } else {
                ("official language", "is")
            };
            parts.push(format!("its {} {} {}", noun, verb, join_and(&languages)));
        }
        if let Some(cur) = currency {
            parts.push(format!("its currency is {}", cur));
        }
        if parts.is_empty() {
            None
        } else {
            Some(format!("{}.", cap_first(&join_and(&parts))))
        }
    } else {
        None
    };
    // Legacy Phase-4a bare-located fallback: a plain LocatedIn with nothing
    // else still yields "X is located in Y."
    let bare_located: Option<String> =
        if !place_signal && s2.is_none() && !s1_present {
            located.map(|l| format!("{} is located in {}.", name, l))
        } else {
            None
        };

    let mut out = String::new();
    if s1_present {
        // Identity + known-for coordinate as apposition inside one sentence:
        // "X is a physicist, known for A and B".
        out.push_str(&format!("{} {}.", name, s1parts.join(", ")));
    }
    for seg in [s2, loc_sentence, attr_sentence, bare_located]
        .into_iter()
        .flatten()
    {
        if !out.is_empty() {
            out.push(' ');
        }
        out.push_str(&seg);
    }
    if out.is_empty() {
        return None;
    }
    Some(out)
}

/// Descriptive PropertyCheck ("what is X like" / "tell me about X"): the
/// flat colon list becomes flowing, deduped, noise-cleaned prose. A short
/// list (<= 3) stays one sentence; a longer list is chunked into two
/// balanced sentences with a neutral subject-continuity pronoun ("It is
/// also ...") so a six-item run reads as discourse rather than a dump.
fn fluent_property_list(c: &Conclusion) -> Option<String> {
    // Guard: only the "A {subj} is: ..." descriptive template. The yes/no
    // matched form cites a single edge (already excluded by len < 2).
    if !(c.prose.starts_with("A ") || c.prose.starts_with("An ")) {
        return None;
    }
    let subj = c.cited_edges.first()?.from.clone();
    if subj.is_empty() {
        return None;
    }
    let names: Vec<String> = c.cited_edges.iter().map(|e| e.to.clone()).collect();
    let props = clean_props(&names);
    if props.is_empty() {
        return None;
    }
    let art = cap_first(indefinite_article(&subj));
    if props.len() <= 3 {
        return Some(format!("{} {} is {}.", art, subj, join_and(&props)));
    }
    // Balanced split; second sentence continues on the neutral pronoun.
    let mid = (props.len() + 1) / 2;
    let (g1, g2) = props.split_at(mid);
    Some(format!(
        "{} {} is {}. It is also {}.",
        art,
        subj,
        join_and(g1),
        join_and(g2)
    ))
}

// ---------------------------------------------------------------------
// Public entry.
// ---------------------------------------------------------------------

/// Re-voice a synthesized conclusion as composed, multi-sentence prose,
/// phrasing ONLY the grounded propositions it already cites. Refusals and
/// single-fact plans pass through byte-identical. Any rebuild that fails
/// the honesty audit is discarded (degrade to the templated prose).
pub fn realize(mut c: Conclusion) -> Conclusion {
    // Single-fact plans and refusals: leave the template (and any yes/no
    // opener) for epistemic to calibrate.
    if c.cited_edges.len() < 2 {
        return c;
    }
    let rebuilt = match c.kind {
        ReasoningKind::EntityFact => fluent_entity(&c),
        ReasoningKind::PropertyCheck => fluent_property_list(&c),
        _ => None, // other shapes already read fluently / self-hedge; leave them.
    };
    if let Some(body) = rebuilt {
        if audit(&body, &c.cited_edges) {
            c.prose = body;
        }
        // else: ungrounded token slipped in -> keep the safe template.
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::Confidence;

    fn e(from: &str, bond: BondType, to: &str, conf: Confidence) -> CitedEdge {
        CitedEdge { from: from.into(), bond, to: to.into(), confidence: conf }
    }
    fn concl(kind: ReasoningKind, prose: &str, edges: Vec<CitedEdge>) -> Conclusion {
        Conclusion { kind, prose: prose.into(), cited_edges: edges, confidence: 0.6 }
    }

    // --- honesty invariants -------------------------------------------
    #[test]
    fn refusal_untouched() {
        let c = concl(ReasoningKind::CausalWhy, "Not from what I have -- no link.", vec![]);
        assert_eq!(realize(c.clone()).prose, c.prose);
    }
    #[test]
    fn single_fact_untouched() {
        let c = concl(
            ReasoningKind::PropertyCheck,
            "Yes -- a dog is loyal. (dog -HasProperty-> loyal)",
            vec![e("dog", BondType::HasProperty, "loyal", Confidence::Verified)],
        );
        assert_eq!(realize(c.clone()).prose, c.prose);
    }
    #[test]
    fn cited_edges_never_mutated() {
        let c = concl(
            ReasoningKind::EntityFact,
            "Albert Einstein is a physicist. Known for relativity. Born in Ulm.",
            vec![
                e("Albert Einstein", BondType::IsA, "physicist", Confidence::Unverified),
                e("Albert Einstein", BondType::KnownFor, "relativity", Confidence::Unverified),
                e("Albert Einstein", BondType::BornIn, "Ulm", Confidence::Unverified),
            ],
        );
        let out = realize(c);
        assert_eq!(out.cited_edges.len(), 3);
        assert_eq!(out.cited_edges[1].to, "relativity");
    }

    // --- discourse composition: multi-sentence + subject continuity ----
    #[test]
    fn entity_composes_multi_sentence_with_subject_continuity() {
        let c = concl(
            ReasoningKind::EntityFact,
            "Albert Einstein is a physicist. Known for relativity and photoelectric effect. Born in Ulm.",
            vec![
                e("Albert Einstein", BondType::IsA, "physicist", Confidence::Unverified),
                e("Albert Einstein", BondType::KnownFor, "relativity", Confidence::Unverified),
                e("Albert Einstein", BondType::KnownFor, "photoelectric effect", Confidence::Unverified),
                e("Albert Einstein", BondType::BornIn, "Ulm", Confidence::Unverified),
            ],
        );
        let out = realize(c).prose;
        // Two sentences (one internal ". " boundary).
        assert_eq!(out.matches(". ").count(), 1, "two sentences: {}", out);
        // Full name introduced exactly once; surname re-mention continues.
        assert_eq!(out.matches("Albert Einstein").count(), 1, "full name once: {}", out);
        assert!(out.contains("is a physicist, known for relativity and photoelectric effect"), "got: {}", out);
        assert!(out.contains("Einstein was born in Ulm"), "continuity: {}", out);
        // No gendered-pronoun guess -- structurally impossible via audit.
        assert!(!out.to_lowercase().contains(" he ") && !out.to_lowercase().contains(" she "), "no gender guess: {}", out);
    }
    #[test]
    fn entity_single_section_stays_one_sentence() {
        let c = concl(
            ReasoningKind::EntityFact,
            "Marie Curie is a chemist. Known for radioactivity.",
            vec![
                e("Marie Curie", BondType::IsA, "chemist", Confidence::Unverified),
                e("Marie Curie", BondType::KnownFor, "radioactivity", Confidence::Unverified),
            ],
        );
        let out = realize(c).prose;
        assert_eq!(out.matches(". ").count(), 0, "one sentence: {}", out);
        assert!(out.contains("Marie Curie is a chemist, known for radioactivity"), "got: {}", out);
    }

    // --- property list cleanup + a/an + chunking -----------------------
    #[test]
    fn property_list_flows_dedupes_and_drops_noise() {
        let c = concl(
            ReasoningKind::PropertyCheck,
            "A fish is: animals, beautiful, big eyes, Clever, Clever, Cold-blooded, different colors.",
            vec![
                e("fish", BondType::HasProperty, "animals", Confidence::Inferred),
                e("fish", BondType::HasProperty, "beautiful", Confidence::Verified),
                e("fish", BondType::HasProperty, "big eyes", Confidence::Inferred),
                e("fish", BondType::HasProperty, "Clever", Confidence::Verified),
                e("fish", BondType::HasProperty, "Clever", Confidence::Verified),
                e("fish", BondType::HasProperty, "Cold-blooded", Confidence::Verified),
                e("fish", BondType::HasProperty, "different colors", Confidence::Inferred),
            ],
        );
        let out = realize(c).prose;
        assert!(!out.contains("beautiful"), "noise dropped: {}", out);
        assert!(!out.contains("big eyes"), "noise dropped: {}", out);
        assert!(!out.contains("different colors"), "noise dropped: {}", out);
        assert_eq!(out.to_lowercase().matches("clever").count(), 1, "deduped: {}", out);
        assert!(out.contains("Cold-blooded"), "kept adjective: {}", out);
        assert!(!out.contains(" is: "), "no colon list: {}", out);
    }
    #[test]
    fn property_list_an_elephant() {
        let c = concl(
            ReasoningKind::PropertyCheck,
            "A elephant is: big, grey, smart.",
            vec![
                e("elephant", BondType::HasProperty, "big", Confidence::Verified),
                e("elephant", BondType::HasProperty, "grey", Confidence::Verified),
                e("elephant", BondType::HasProperty, "smart", Confidence::Verified),
            ],
        );
        let out = realize(c).prose;
        assert!(out.starts_with("An elephant is "), "got: {}", out);
    }
    #[test]
    fn property_list_chunks_long_into_two_sentences() {
        let c = concl(
            ReasoningKind::PropertyCheck,
            "A dog is: active, aggressive, deaf, different, diurnal, docile.",
            vec![
                e("dog", BondType::HasProperty, "active", Confidence::Verified),
                e("dog", BondType::HasProperty, "aggressive", Confidence::Verified),
                e("dog", BondType::HasProperty, "deaf", Confidence::Verified),
                e("dog", BondType::HasProperty, "different", Confidence::Verified),
                e("dog", BondType::HasProperty, "diurnal", Confidence::Verified),
                e("dog", BondType::HasProperty, "docile", Confidence::Verified),
            ],
        );
        let out = realize(c).prose;
        assert_eq!(out.matches(". ").count(), 1, "two sentences: {}", out);
        assert!(out.starts_with("A dog is active, aggressive, and deaf."), "s1: {}", out);
        assert!(out.contains("It is also different, diurnal, and docile."), "s2: {}", out);
    }

    // --- the audit is a real fabrication guard -------------------------
    #[test]
    fn audit_rejects_ungrounded_token() {
        // "unicorn" is not in any endpoint label or bond lexeme.
        let edges = vec![e("dog", BondType::HasProperty, "loyal", Confidence::Verified)];
        assert!(!audit("A dog is loyal and a unicorn.", &edges));
        assert!(audit("A dog is loyal.", &edges));
    }
    #[test]
    fn entity_composition_never_emits_gendered_pronoun() {
        // Gender is not on any cited edge; continuity re-mentions the
        // grounded surname instead of guessing "he"/"she".
        let c = concl(
            ReasoningKind::EntityFact,
            "seed",
            vec![
                e("Albert Einstein", BondType::IsA, "physicist", Confidence::Unverified),
                e("Albert Einstein", BondType::BornIn, "Ulm", Confidence::Unverified),
            ],
        );
        let out = realize(c).prose.to_lowercase();
        assert!(!out.contains(" he ") && !out.contains(" she "), "no gender guess: {}", out);
        assert!(out.contains("einstein was born in ulm"), "surname continuity: {}", out);
    }
    #[test]
    fn indefinite_article_agreement() {
        assert_eq!(indefinite_article("elephant"), "an");
        assert_eq!(indefinite_article("dog"), "a");
        assert_eq!(indefinite_article("hour"), "an");
        assert_eq!(indefinite_article("university"), "a");
    }

    #[test]
    fn deceased_person_reads_past_tense_with_death_place() {
        // OP-TENSE: a cited DiedIn bond => past tense + a grounded death-place
        // clause. "died"/"princeton" are grounded by the DiedIn edge + lexeme;
        // no birth/death DATE is emitted (BornOn/DiedOn are a source gap).
        let c = concl(
            ReasoningKind::EntityFact,
            "seed",
            vec![
                e("Albert Einstein", BondType::IsA, "physicist", Confidence::Unverified),
                e("Albert Einstein", BondType::KnownFor, "relativity", Confidence::Unverified),
                e("Albert Einstein", BondType::BornIn, "Ulm", Confidence::Unverified),
                e("Albert Einstein", BondType::DiedIn, "Princeton", Confidence::Unverified),
            ],
        );
        let out = realize(c).prose.to_lowercase();
        assert!(out.contains("was a physicist"), "past tense identity: {}", out);
        assert!(!out.contains("is a physicist"), "no present tense for deceased: {}", out);
        assert!(out.contains("died in princeton"), "death place: {}", out);
        assert!(out.contains("born in ulm"), "birth place retained: {}", out);
        assert!(!out.contains("1879") && !out.contains("1955"), "no invented dates: {}", out);
    }

    #[test]
    fn living_person_stays_present_tense() {
        // Guard: no DiedIn/DiedOn bond => present tense, no death clause. Being
        // dead must be grounded in a bond, never assumed from absence.
        let c = concl(
            ReasoningKind::EntityFact,
            "seed",
            vec![
                e("Barack Obama", BondType::IsA, "politician", Confidence::Unverified),
                e("Barack Obama", BondType::KnownFor, "civil rights", Confidence::Unverified),
                e("Barack Obama", BondType::BornIn, "Honolulu", Confidence::Unverified),
            ],
        );
        let out = realize(c).prose.to_lowercase();
        assert!(out.contains("is a politician"), "present tense: {}", out);
        assert!(!out.contains(" was a "), "no false past tense: {}", out);
        assert!(!out.contains("died"), "no death clause: {}", out);
    }

    // --- OP-OVERVIEW: richer subject-overview composition ---------------
    #[test]
    fn entity_person_weaves_birth_and_death_years() {
        // BornOn/DiedOn year edges are now cited -> the composer weaves
        // "born in Ulm in 1879 and died in Princeton in 1955". The years are
        // grounded BornOn/DiedOn endpoints, never a guess; the audit passes.
        let c = concl(
            ReasoningKind::EntityFact,
            "seed",
            vec![
                e("Albert Einstein", BondType::IsA, "physicist", Confidence::Unverified),
                e("Albert Einstein", BondType::KnownFor, "relativity", Confidence::Unverified),
                e("Albert Einstein", BondType::BornIn, "Ulm", Confidence::Unverified),
                e("Albert Einstein", BondType::BornOn, "1879", Confidence::Unverified),
                e("Albert Einstein", BondType::DiedIn, "Princeton", Confidence::Unverified),
                e("Albert Einstein", BondType::DiedOn, "1955", Confidence::Unverified),
            ],
        );
        let out = realize(c).prose;
        let low = out.to_lowercase();
        assert!(low.contains("was a physicist"), "past-tense identity: {}", out);
        assert!(low.contains("born in ulm in 1879"), "birth place+year: {}", out);
        assert!(low.contains("died in princeton in 1955"), "death place+year: {}", out);
    }

    #[test]
    fn entity_person_trails_off_when_many_known_for() {
        // > 3 KnownFor -> show a few, trail off honestly with "among others"
        // (all are cited, so the trailing signal is grounded, not padding).
        let c = concl(
            ReasoningKind::EntityFact,
            "seed",
            vec![
                e("Isaac Newton", BondType::IsA, "physicist", Confidence::Unverified),
                e("Isaac Newton", BondType::KnownFor, "gravity", Confidence::Unverified),
                e("Isaac Newton", BondType::KnownFor, "calculus", Confidence::Unverified),
                e("Isaac Newton", BondType::KnownFor, "optics", Confidence::Unverified),
                e("Isaac Newton", BondType::KnownFor, "mechanics", Confidence::Unverified),
            ],
        );
        let out = realize(c).prose;
        assert!(out.to_lowercase().contains("among others"), "trails off: {}", out);
    }

    #[test]
    fn entity_place_weaves_capital_language_currency() {
        // Place shape: capital / official language / currency woven when a
        // genuine place signal is cited. Not live-reachable today (countries
        // hit the concept-definition path upstream), so this is the
        // composition + honesty proof for the place branch.
        let c = concl(
            ReasoningKind::EntityFact,
            "seed",
            vec![
                e("France", BondType::IsA, "country", Confidence::Unverified),
                e("France", BondType::LocatedIn, "Europe", Confidence::Unverified),
                e("France", BondType::HasCapital, "Paris", Confidence::Unverified),
                e("France", BondType::HasOfficialLanguage, "French", Confidence::Unverified),
                e("France", BondType::HasCurrency, "euro", Confidence::Unverified),
            ],
        );
        let out = realize(c).prose;
        let low = out.to_lowercase();
        assert!(low.contains("is a country"), "identity: {}", out);
        assert!(low.contains("capital is paris"), "capital: {}", out);
        assert!(low.contains("official language is french"), "language: {}", out);
        assert!(low.contains("currency is euro"), "currency: {}", out);
    }

    #[test]
    fn entity_thin_node_stays_short_no_padding() {
        // Degrade honestly: a two-fact node yields a short answer and invents
        // neither a birthplace nor a place attribute.
        let c = concl(
            ReasoningKind::EntityFact,
            "seed",
            vec![
                e("Ada Lovelace", BondType::IsA, "mathematician", Confidence::Unverified),
                e("Ada Lovelace", BondType::KnownFor, "computing", Confidence::Unverified),
            ],
        );
        let out = realize(c).prose;
        let low = out.to_lowercase();
        assert!(low.contains("is a mathematician"), "identity: {}", out);
        assert!(low.contains("known for computing"), "known-for: {}", out);
        assert!(!low.contains("born"), "no invented birth: {}", out);
        assert!(!low.contains("capital"), "no invented place attrs: {}", out);
    }
}
