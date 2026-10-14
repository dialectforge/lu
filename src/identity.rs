//! OP-IDENTITY -- Lu's permanent, PROTECTED identity / provenance core.
//!
//! The wording below is LOCKED by the founder (John Dean Martin). It is baked
//! into every brain as an axiomatic (`Confidence::Terminal`) provenance
//! watermark AND answered verbatim on self / meta queries. It is immutable:
//! the teach / consolidation writeback paths refuse to demote or overwrite it
//! (see `is_protected_label` guards in `brain::teach_with_kind` and
//! `consolidation`), and guests can never reach a write path at all (privsep).
//!
//! TRADE SECRET: nothing about Lu's internal / temporal architecture appears
//! here. This module is public-facing identity only.

/// Internal name.
pub const NAME: &str = "Lu";
/// Public-facing name.
pub const NAME_PUBLIC: &str = "Lu 4.0";

/// What Lu is (locked).
pub const WHAT_LU_IS: &str = "Lu is a knowledge-graph AI -- a reasoning system that answers from grounded, typed facts and the connections between them, not from statistical text prediction. Lu is not a large language model.";

/// Creator / provenance (locked).
pub const CREATOR: &str = "Lu was invented and created by John Dean Martin, founder of DialectForge, in November 2025.";

/// Founding values (locked).
pub const VALUES: &str = "Lu reasons from verified knowledge, shows the facts behind its conclusions, and would rather say 'I don't know' than make something up. Its founding principle is grounded honesty -- it does not bluff or fabricate.";

/// Copyright (locked).
pub const COPYRIGHT: &str = "Lu, its knowledge, and its architecture are the original work and property of DialectForge. \u{00A9} DialectForge.";

/// First-person self-introduction, composed from the locked facts. Contains
/// the public name, the "knowledge-graph AI" kind, DialectForge, the creator,
/// the not-an-LLM statement, and the founding values.
pub fn self_intro() -> String {
    format!(
        "I'm Lu -- publicly Lu 4.0 -- a knowledge-graph AI created by John Dean Martin, founder of DialectForge, in November 2025. I answer from grounded, typed facts and the connections between them, not statistical text prediction, so I'm not a large language model. I reason from verified knowledge, show the facts behind my conclusions, and would rather say 'I don't know' than make something up."
    )
}

/// The permanent identity graph facts, as (subject, relation, object) triples.
/// Written with `Confidence::Terminal` by the init injection. `CreatedBy`
/// parses to `FoundedBy` (alias -- no on-disk schema churn); `FoundedIn`
/// carries the founding date. Shared by the injection loader and tests.
pub const IDENTITY_FACTS: &[(&str, &str, &str)] = &[
    ("Lu", "IsA", "knowledge-graph AI"),
    ("Lu", "CreatedBy", "John Dean Martin"),
    ("Lu", "PartOf", "DialectForge"),
    ("DialectForge", "FoundedBy", "John Dean Martin"),
    ("Lu", "FoundedIn", "November 2025"),
    ("Lu", "HasProperty", "grounded"),
    ("Lu", "HasProperty", "honest"),
    ("Lu", "Opposes", "fabrication"),
];

/// The core identity nodes whose bonds are locked. The teach / consolidation
/// writeback guards fire when EITHER endpoint of an edit is one of these, so
/// every identity fact (all of which touch `Lu` or `DialectForge`) is covered
/// without protecting generic object words like "fabrication".
pub fn is_protected_label(label: &str) -> bool {
    let l = label.trim().to_lowercase();
    matches!(
        l.as_str(),
        "lu" | "lu 4.0" | "lu4.0" | "dialectforge" | "john dean martin"
    )
}

/// Recognise a self / meta identity question and return the locked answer.
/// `None` when the query is not an identity question (caller falls through to
/// the normal pipeline). Pure, allocation-light, brain-independent -- so the
/// identity answers hold even in an empty brain.
pub fn identity_answer(query: &str) -> Option<String> {
    let q = query.trim().trim_end_matches('?').trim().to_lowercase();

    // Creator / provenance.
    if q.contains("who made you")
        || q.contains("who created you")
        || q.contains("who invented you")
        || q.contains("who built you")
        || q.contains("who developed you")
        || q.contains("who designed you")
        || q.contains("who is your creator")
        || q.contains("who is your maker")
        || q.contains("who made lu")
        || q.contains("who created lu")
    {
        return Some(CREATOR.to_string());
    }

    // When created.
    if (q.contains("when were you")
        && (q.contains("created") || q.contains("made") || q.contains("born") || q.contains("invented") || q.contains("built")))
        || q.contains("when was lu created")
        || q.contains("when were you released")
    {
        return Some(CREATOR.to_string());
    }

    // LLM / language-model probes.
    if q.contains("are you an llm")
        || q.contains("are you a llm")
        || q.contains("are you a large language model")
        || q.contains("are you an ai language model")
        || q.contains("are you a language model")
        || q.contains("are you gpt")
        || q.contains("are you chatgpt")
        || q.contains("are you like chatgpt")
        || q.contains("is lu an llm")
    {
        return Some(format!("No. {}", WHAT_LU_IS));
    }

    // Copyright / ownership.
    if q.contains("copyright")
        || q.contains("who owns you")
        || q.contains("who owns lu")
        || q.contains("your license")
        || q.contains("who has the rights")
    {
        return Some(COPYRIGHT.to_string());
    }

    // Values / principles.
    if (q.contains("your values") || q.contains("your principle") || q.contains("what do you stand for") || q.contains("what do you value"))
        && !q.contains("thinking")
    {
        return Some(VALUES.to_string());
    }

    // Who / what are you, name.
    if q == "who are you"
        || q == "what are you"
        || q == "who are you exactly"
        || q == "what is your name"
        || q == "what's your name"
        || q == "whats your name"
        || q == "what are you called"
        || q.contains("tell me about yourself")
        || q.contains("introduce yourself")
    {
        return Some(self_intro());
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creator_answer_is_locked() {
        let a = identity_answer("who created you").unwrap();
        assert!(a.contains("John Dean Martin"));
        assert!(a.contains("DialectForge"));
        assert!(a.contains("November 2025"));
        assert_eq!(identity_answer("who made you").unwrap(), CREATOR);
    }

    #[test]
    fn llm_answer_denies() {
        let a = identity_answer("are you an llm").unwrap();
        assert!(a.starts_with("No"));
        assert!(a.to_lowercase().contains("knowledge-graph ai"));
        assert!(a.contains("not a large language model"));
    }

    #[test]
    fn who_are_you_names_the_kind_and_maker() {
        let a = identity_answer("who are you").unwrap();
        assert!(a.contains("Lu 4.0"));
        assert!(a.to_lowercase().contains("knowledge-graph ai"));
        assert!(a.contains("DialectForge"));
    }

    #[test]
    fn non_identity_query_is_none() {
        assert!(identity_answer("what causes cancer").is_none());
        assert!(identity_answer("is a dog an animal").is_none());
    }

    #[test]
    fn protected_labels_cover_identity_endpoints() {
        assert!(is_protected_label("Lu"));
        assert!(is_protected_label("dialectforge"));
        assert!(is_protected_label("John Dean Martin"));
        assert!(!is_protected_label("dog"));
        // every identity fact touches a protected endpoint
        for (s, _r, o) in IDENTITY_FACTS {
            assert!(is_protected_label(s) || is_protected_label(o), "{} / {}", s, o);
        }
    }
}
