//! Port G — negation handling end-to-end.
//!
//! Lu 4D's TemporalBond rkyv-archive layout is locked (the canonical
//! brain at /var/lib/lu/brain.lbr is 1.26 GB v3enc and cannot be
//! retrained in this port). Port G therefore stores negation as an
//! *overlay* keyed by `(from, to, bond_type)` rather than as a struct
//! field on TemporalBond itself. The conceptual model (Learning #577)
//! is unchanged: every bond is either positive or negated, with
//! parity-XOR propagation through reasoning chains. The implementation
//! detail is that the `negated: bool` lives in `Brain.negation_overlay`
//! instead of inside `TemporalBond`.
//!
//! Backward compat: pre-G brains have an empty overlay → all bonds
//! default to negated=false. Forward compat: post-G brains save the
//! overlay through serde JSON; pre-G binaries that load a post-G brain
//! either skip the field (#[serde(default)] in BrainFile) or fail
//! cleanly. The rkyv-archive layout for TemporalBond stays
//! byte-identical to pre-G, so the production v3enc brain at
//! /var/lib/lu/brain.lbr loads with `negated=false` for every bond
//! without any migration shim.
//!
//! Per Learning #634, this schema work lands pre-retrain. Once the
//! retrain happens the overlay stays as the persistence shape; if a
//! future port introduces the field directly on TemporalBond (post-
//! retrain), the overlay can be folded into it via a one-shot
//! migration.

use crate::bond::BondType;
use crate::seed::Seed;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Composite key for the negation overlay. Carries the bond type so
/// (X IsA Y, negated=true) and (X RelatedTo Y, negated=false) coexist
/// cleanly between the same pair of seeds.
pub type NegationKey = (Seed, Seed, BondType);

/// Side-table that records which bonds carry an explicit negation flag.
/// Absent entries default to negated=false.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NegationOverlay {
    /// Map from (from, to, bond_type) -> negated flag. Only `true`
    /// entries need to be recorded; any missing key is implicitly
    /// positive.
    #[serde(default)]
    pub flags: HashMap<String, bool>,
    /// Port G + J — keys whose flag has been observed BOTH positive
    /// AND negative across teach calls. Drives the
    /// `has_contradiction` signal that surfaces `ContestedAnswer`-style
    /// narration in J's confidence aggregation. Empty by default;
    /// populated by `set` when a flag change flips polarity.
    #[serde(default)]
    pub contested: std::collections::HashSet<String>,
    /// Witness set for keys that have been observed positive at least
    /// once. Needed because `flags` only stores explicit `true`
    /// entries (positive bonds default to absent). Without this
    /// witness, set(false) -> set(true) would not register as a flip.
    #[serde(default)]
    pub seen_positive: std::collections::HashSet<String>,
}

impl NegationOverlay {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn key(from: &Seed, to: &Seed, bond_type: BondType) -> String {
        // Encode seed pair + bond_type as a hex string so the overlay
        // round-trips through JSON (HashMap keys must be strings).
        // Truncate seed to 8 bytes -- 64 bits of identity per side is
        // enough to avoid collisions in any realistic graph and keeps
        // the stored key compact.
        let f8 = &from.0[0..8];
        let t8 = &to.0[0..8];
        format!(
            "{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}|\
             {:02x}{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}|{:?}",
            f8[0], f8[1], f8[2], f8[3], f8[4], f8[5], f8[6], f8[7],
            t8[0], t8[1], t8[2], t8[3], t8[4], t8[5], t8[6], t8[7],
            bond_type
        )
    }

    /// True if this bond was tagged negated (or has its flag explicitly
    /// set to true). Defaults to false for any unknown bond.
    pub fn is_negated(&self, from: &Seed, to: &Seed, bond_type: BondType) -> bool {
        self.flags
            .get(&Self::key(from, to, bond_type))
            .copied()
            .unwrap_or(false)
    }

    /// Set the negation flag for a bond. Setting to `false` removes the
    /// entry from `flags` to keep the storage sparse. The witness
    /// sets `seen_positive` and the existence of an entry in `flags`
    /// detect polarity flips: any flip across teach calls adds the
    /// key to `contested` so callers can surface the contradiction
    /// explicitly.
    pub fn set(&mut self, from: Seed, to: Seed, bond_type: BondType, negated: bool) {
        let k = Self::key(&from, &to, bond_type);
        let was_negated = self.flags.contains_key(&k);
        let was_positive = self.seen_positive.contains(&k);
        if negated {
            // Flip detection: the key was previously witnessed
            // positive, now we're going negative. That's a contradiction.
            if was_positive {
                self.contested.insert(k.clone());
            }
            self.flags.insert(k, true);
        } else {
            // Flip detection: the key was previously stored negated
            // and we're now setting it positive.
            if was_negated {
                self.contested.insert(k.clone());
            }
            self.flags.remove(&k);
            self.seen_positive.insert(k);
        }
    }

    /// Port G + J — true if this bond's flag has flipped polarity
    /// across teach calls (i.e., we have contradictory evidence for
    /// this triple).
    pub fn is_contested(&self, from: &Seed, to: &Seed, bond_type: BondType) -> bool {
        self.contested.contains(&Self::key(from, to, bond_type))
    }

    /// Iterator over the contested keys for diagnostics + tests.
    pub fn contested_keys(&self) -> impl Iterator<Item = &String> {
        self.contested.iter()
    }

    /// Mark a bond as contested without changing the flag. Useful when a
    /// caller observes contradictory evidence from external sources
    /// (e.g. two ingest packs with opposite polarity for the same
    /// triple) and wants to surface the contradiction without
    /// silently letting the later one win.
    pub fn mark_contested(&mut self, from: Seed, to: Seed, bond_type: BondType) {
        let k = Self::key(&from, &to, bond_type);
        self.contested.insert(k);
    }

    /// How many bonds carry an explicit negation flag.
    pub fn len(&self) -> usize {
        self.flags.len()
    }

    pub fn is_empty(&self) -> bool {
        self.flags.is_empty()
    }
}

/// Cue lexicon for the ingest extractor. Kept tiny per Learning #577
/// (~25 entries) and exposed for tuning. Order matters only for
/// determinism in tests.
pub const NEGATION_CUES: &[&str] = &[
    "not",
    "no",
    "never",
    "none",
    "nothing",
    "nobody",
    "nowhere",
    "neither",
    "nor",
    "isn't",
    "isnt",
    "aren't",
    "arent",
    "wasn't",
    "wasnt",
    "weren't",
    "werent",
    "doesn't",
    "doesnt",
    "don't",
    "dont",
    "didn't",
    "didnt",
    "cannot",
    "can't",
    "cant",
    "won't",
    "wont",
    "shouldn't",
    "shouldnt",
    "wouldn't",
    "wouldnt",
];

/// Detect a negation cue in `text`. Punctuation-aware tokenisation:
/// strips trailing/leading punctuation, normalises to lowercase, then
/// matches against `NEGATION_CUES`. False on the empty string.
///
/// Tunable behaviour: handles "not only X but Y" — we still flag the
/// `not` here, but downstream parity logic will compose with whatever
/// other negation cues appear in the same span. Per the spec scope
/// boundary, full quantifier-aware negation ("not all birds fly") is
/// out of scope for Port G; we detect the cue and let the pipeline
/// decide what to do with it.
pub fn detect_negation_cue(text: &str) -> bool {
    if text.is_empty() {
        return false;
    }
    for raw in text.split_whitespace() {
        let t = raw
            .trim_matches(|c: char| !c.is_alphanumeric() && c != '\'')
            .to_lowercase();
        if t.is_empty() {
            continue;
        }
        for cue in NEGATION_CUES {
            if t == *cue {
                return true;
            }
        }
    }
    false
}

/// Same as [`detect_negation_cue`] but returns the count of cues found.
/// Useful for tests asserting that "not unlike" trips the cue twice
/// (parity even → positive) under future quantifier-aware extensions.
/// Today the chain composer's overlay is set/cleared per bond, so this
/// is mostly diagnostic.
pub fn count_negation_cues(text: &str) -> usize {
    let mut n = 0;
    for raw in text.split_whitespace() {
        let t = raw
            .trim_matches(|c: char| !c.is_alphanumeric() && c != '\'')
            .to_lowercase();
        if t.is_empty() {
            continue;
        }
        for cue in NEGATION_CUES {
            if t == *cue {
                n += 1;
                break;
            }
        }
    }
    n
}

/// XOR the negation flags along a chain. Used by the chain composer
/// to decide whether a multi-leg conclusion should articulate as
/// positive or negative.
///
/// Per Learning #577 the parity is *only* meaningful for chains whose
/// pattern propagates negation — Causal and Hierarchical. Other
/// patterns (PropertyChain, Composition, Transformational, Contrastive,
/// Mixed) do NOT propagate negation through composition: a thing's
/// part not having a property doesn't mean the thing doesn't have it.
/// Callers must gate on `pattern_propagates_negation` before using the
/// returned parity for a final-conclusion flip.
pub fn chain_parity(negated_flags: &[bool]) -> bool {
    negated_flags.iter().fold(false, |acc, &b| acc ^ b)
}

/// Whether a reasoning pattern propagates negation through composition.
/// This is the spec-prescribed restriction (#577 LAYER 2 (2b)).
pub fn pattern_propagates_negation(pattern_label: &str) -> bool {
    matches!(pattern_label, "Causal" | "Hierarchical")
}

/// Render the negative form of a bond's articulation. Mirrors
/// `articulate_bond` in reasoning.rs but flips the polarity. The
/// phrasing follows ordinary English negation rules (verb-specific:
/// "is a kind of" → "is not a kind of"; "causes" → "does not cause";
/// "comes from" → "does not come from").
pub fn articulate_bond_negative(bond_type: BondType, from: &str, to: &str) -> String {
    let f = capitalize(from);
    match bond_type {
        BondType::IsA => format!("{} is not a kind of {}.", f, to),
        BondType::DefinedAs => format!("{} is not defined as {}.", f, to),
        BondType::WorksBy => format!("{} does not work by {}.", f, to),
        BondType::ComposedOf => format!("{} is not composed of {}.", f, to),
        BondType::PurposeOf => format!("{} is not for {}.", f, to),
        BondType::DiffersFrom => format!("{} does not differ from {}.", f, to),
        BondType::LeadsTo => format!("{} does not lead to {}.", f, to),
        BondType::ExampleOf => format!("{} is not an example of {}.", f, to),
        BondType::HasProperty => format!("{} does not have the property {}.", f, to),
        BondType::Causes => format!("{} does not cause {}.", f, to),
        BondType::PartOf => format!("{} is not part of {}.", f, to),
        BondType::UsedFor => format!("{} is not used for {}.", f, to),
        BondType::Produces => format!("{} does not produce {}.", f, to),
        BondType::Requires => format!("{} does not require {}.", f, to),
        BondType::Opposes => format!("{} does not oppose {}.", f, to),
        BondType::DerivedFrom => format!("{} does not come from {}.", f, to),
        BondType::ContrastedWith => format!("{} is not contrasted with {}.", f, to),
        BondType::RelatedTo => format!("{} is not associated with {}.", f, to),
        BondType::BornIn => format!("{} was not born in {}.", f, to),
        BondType::BornOn => format!("{} was not born on {}.", f, to),
        BondType::DiedIn => format!("{} did not die in {}.", f, to),
        BondType::DiedOn => format!("{} did not die on {}.", f, to),
        BondType::DiscoveredBy => format!("{} was not discovered by {}.", f, to),
        BondType::FoundedBy => format!("{} was not founded by {}.", f, to),
        BondType::FoundedIn => format!("{} was not founded in {}.", f, to),
        BondType::HasAltitude => format!("{} does not have altitude {}.", f, to),
        BondType::HasAspect => format!("{} does not have aspect {}.", f, to),
        BondType::HasCapital => format!("{} does not have capital {}.", f, to),
        BondType::HasOfficialLanguage => format!("{} does not have official language {}.", f, to),
        BondType::HasPopulation => format!("{} does not have population {}.", f, to),
        BondType::KnownFor => format!("{} is not known for {}.", f, to),
        BondType::LocatedIn => format!("{} is not located in {}.", f, to),
        BondType::MeasuredIn => format!("{} is not measured in {}.", f, to),
        BondType::NamedAfter => format!("{} is not named after {}.", f, to),
        BondType::WrittenBy => format!("{} was not written by {}.", f, to),
        BondType::HasArea => format!("{} does not have area {}.", f, to),
        BondType::HasMethod => format!("{} does not have method {}.", f, to),
        BondType::HasAttribute => format!("{} does not have attribute {}.", f, to),
        BondType::HasParameter => format!("{} does not have parameter {}.", f, to),
        BondType::Returns => format!("{} does not return {}.", f, to),
        BondType::Inherits => format!("{} does not inherit from {}.", f, to),
        BondType::Calls => format!("{} does not call {}.", f, to),
        BondType::HasType => format!("{} does not have type {}.", f, to),
        BondType::Implements => format!("{} does not implement {}.", f, to),
        BondType::HasCurrency => format!("{} does not use the currency {}.", f, to),
        BondType::AuthoredBy => format!("{} was not written by {}.", f, to),
    }
}

fn capitalize(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) => c.to_uppercase().collect::<String>() + chars.as_str(),
        None => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bond::BondType;

    #[test]
    fn cue_detection_basics() {
        assert!(detect_negation_cue("Sharks are not mammals"));
        assert!(detect_negation_cue("birds aren't reptiles"));
        assert!(detect_negation_cue("this medication should not be combined"));
        assert!(detect_negation_cue("Pluto is no longer a planet"));
        assert!(detect_negation_cue("never combine these"));
        assert!(detect_negation_cue("cannot work with alcohol"));
        assert!(detect_negation_cue("can't be true"));
    }

    #[test]
    fn cue_detection_no_false_positives() {
        assert!(!detect_negation_cue("birds are mammals"));
        assert!(!detect_negation_cue("the sun is a star"));
        assert!(!detect_negation_cue(""));
        assert!(!detect_negation_cue("Notable scientists studied this")); // "Notable" ≠ "not"
        assert!(!detect_negation_cue("nothings"));                         // "nothings" ≠ "nothing"
    }

    #[test]
    fn cue_detection_punctuation_aware() {
        assert!(detect_negation_cue("Pluto is not, however, a planet."));
        assert!(detect_negation_cue("\"don't\" do this"));
        assert!(detect_negation_cue("(cannot) be combined"));
    }

    #[test]
    fn count_cues_handles_double_negatives() {
        assert_eq!(count_negation_cues("not unlike X"), 1);
        assert_eq!(count_negation_cues("never not true"), 2);
        assert_eq!(count_negation_cues("birds are mammals"), 0);
    }

    #[test]
    fn chain_parity_xor() {
        assert!(!chain_parity(&[]));            // empty = positive
        assert!(!chain_parity(&[false]));       // 0 negations
        assert!(chain_parity(&[true]));         // 1 negation
        assert!(!chain_parity(&[true, true])); // 2 negations
        assert!(chain_parity(&[true, true, true])); // 3 negations
        assert!(!chain_parity(&[false, true, true, false])); // 2 negations
    }

    #[test]
    fn pattern_propagation_table() {
        assert!(pattern_propagates_negation("Causal"));
        assert!(pattern_propagates_negation("Hierarchical"));
        assert!(!pattern_propagates_negation("PropertyChain"));
        assert!(!pattern_propagates_negation("Transformational"));
        assert!(!pattern_propagates_negation("Contrastive"));
        assert!(!pattern_propagates_negation("Mixed"));
    }

    #[test]
    fn negation_overlay_basics() {
        let mut o = NegationOverlay::new();
        let s1 = Seed::from_label("birds");
        let s2 = Seed::from_label("mammals");
        assert!(!o.is_negated(&s1, &s2, BondType::IsA));
        o.set(s1, s2, BondType::IsA, true);
        assert!(o.is_negated(&s1, &s2, BondType::IsA));
        assert_eq!(o.len(), 1);
        // Different bond type stays positive.
        assert!(!o.is_negated(&s1, &s2, BondType::HasProperty));
        // Clearing.
        o.set(s1, s2, BondType::IsA, false);
        assert!(!o.is_negated(&s1, &s2, BondType::IsA));
        assert!(o.is_empty());
    }

    #[test]
    fn negative_articulation_table() {
        assert_eq!(
            articulate_bond_negative(BondType::IsA, "bird", "mammal"),
            "Bird is not a kind of mammal."
        );
        assert_eq!(
            articulate_bond_negative(BondType::Causes, "rust", "leaks"),
            "Rust does not cause leaks."
        );
        assert_eq!(
            articulate_bond_negative(BondType::HasProperty, "snow", "warm"),
            "Snow does not have the property warm."
        );
        assert_eq!(
            articulate_bond_negative(BondType::PartOf, "moon", "sun"),
            "Moon is not part of sun."
        );
    }

    #[test]
    fn overlay_round_trips_through_serde() {
        let mut o = NegationOverlay::new();
        let s1 = Seed::from_label("sharks");
        let s2 = Seed::from_label("mammals");
        let s3 = Seed::from_label("pluto");
        let s4 = Seed::from_label("planet");
        o.set(s1, s2, BondType::IsA, true);
        o.set(s3, s4, BondType::IsA, true);
        let json = serde_json::to_string(&o).unwrap();
        let back: NegationOverlay = serde_json::from_str(&json).unwrap();
        assert!(back.is_negated(&s1, &s2, BondType::IsA));
        assert!(back.is_negated(&s3, &s4, BondType::IsA));
        assert!(!back.is_negated(&s1, &s4, BondType::IsA));
        assert_eq!(back.len(), 2);
    }

    #[test]
    fn pre_g_brain_compat_default_empty() {
        // A pre-G brain (no negation overlay in the JSON) deserializes
        // as an empty overlay -- every bond defaults to negated=false.
        let json = "{}";
        let o: NegationOverlay = serde_json::from_str(json).unwrap();
        assert!(o.is_empty());
        let s1 = Seed::from_label("any");
        let s2 = Seed::from_label("thing");
        assert!(!o.is_negated(&s1, &s2, BondType::IsA));
    }

    #[test]
    fn flag_flip_records_contestation() {
        let mut o = NegationOverlay::new();
        let s1 = Seed::from_label("X");
        let s2 = Seed::from_label("Y");
        // First observation: positive -> no contestation.
        o.set(s1, s2, BondType::IsA, false);
        assert!(!o.is_contested(&s1, &s2, BondType::IsA));
        // Flip to negative: now contested.
        o.set(s1, s2, BondType::IsA, true);
        assert!(o.is_contested(&s1, &s2, BondType::IsA));
        // Flip back: still contested (history is sticky).
        o.set(s1, s2, BondType::IsA, false);
        assert!(o.is_contested(&s1, &s2, BondType::IsA));
    }

    #[test]
    fn first_negative_is_not_contested() {
        let mut o = NegationOverlay::new();
        let s1 = Seed::from_label("U");
        let s2 = Seed::from_label("V");
        // First-ever observation, even when negated, is NOT a contradiction.
        o.set(s1, s2, BondType::IsA, true);
        assert!(o.is_negated(&s1, &s2, BondType::IsA));
        assert!(!o.is_contested(&s1, &s2, BondType::IsA));
    }
}
