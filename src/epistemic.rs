//! OP-EPISTEMIC (Phase 3) -- confidence -> epistemic-language realizer.
//!
//! A single, shape-AGNOSTIC pass that calibrates the register of a
//! synthesized [`Conclusion`] to the confidence of the evidence it already
//! cites. It is called ONCE from `narration::narrate_inner`, right after
//! `synthesis::try_synthesize` returns, and rewrites `conclusion.prose`
//! in place -- the `ResponseShape::Conclusion` render arm prints that
//! prose verbatim, so there is no Renderer surgery.
//!
//! Design components (EPISTEMIC_CALIBRATION_DESIGN.md):
//!   * Component 2 -- map the weakest cited edge's [`Confidence`] to a
//!     connective register (Terminal/Verified -> assertive; Unverified ->
//!     reported; Inferred -> associative; Contested -> disputed).
//!   * Component 3 -- when a conclusion IS grounded but the recall behind
//!     it is weak (low max_activation / anchored ratio), prepend a
//!     "thinly known" hedge. A Contested cited edge routes to the disputed
//!     register. Neither lowers the honesty floor.
//!   * Component 4 -- one function keyed on register + kind (NOT per
//!     shape), so property / category / definition / causal / comparative
//!     are all calibrated by the same code path.
//!
//! HONESTY INVARIANT (design section 8), enforced here:
//!   * recast NEVER touches `cited_edges` -- the audit trail is preserved.
//!   * recast NEVER touches a conclusion with NO cited edges. That is a
//!     refusal / no-figures integrity answer; the idk floor owns its
//!     wording, and recast returns it byte-identical.
//!   * recast may only ADD hedging over already-grounded material. The
//!     register it applies can only move toward MORE hedging, never toward
//!     more confidence than min(cited edge confidence).
//!   * It cannot manufacture a claim: it only weakens an assertive
//!     connective that is already present, or prepends a hedge frame.

use crate::idk_gate::IdkSignals;
use crate::node::Confidence;
use crate::synthesis::{CitedEdge, Conclusion};

/// Recall-weakness thresholds for the thin-topic hedge (Component 3).
/// Tuned against the canonical brain baseline probe: healthy grounded
/// conclusions cluster at max_activation >= 0.42 (dog/cancer/water/
/// einstein), so a 0.35 floor fires only on genuinely weak recall and
/// never on the healthy set. The anchored ratio is ~0 for single/two
/// concept queries (they resolve the subject directly rather than by
/// anchored spread), so it is a permissive AND-term, not the driver.
const THIN_MAX_ACT: f32 = 0.35;
const THIN_ANCHOR_RATIO: f32 = 0.34;

/// Coarse epistemic register, derived from the WEAKEST cited edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Register {
    /// Terminal / Verified -- corroborated or axiomatic. Assertive voice
    /// ("causes", "is"). Prose left byte-identical.
    Assertive,
    /// Unverified -- single source. "is linked to / reported to / from a
    /// single source on record".
    Reported,
    /// Inferred -- derived. "is associated with / a contributing factor in".
    Associative,
    /// Contested -- contradicted by other evidence. "claimed by some,
    /// though this is disputed".
    Disputed,
}

impl Register {
    fn from_conf(c: Confidence) -> Self {
        match c {
            Confidence::Terminal | Confidence::Verified => Register::Assertive,
            Confidence::Unverified => Register::Reported,
            Confidence::Inferred => Register::Associative,
            Confidence::Contested => Register::Disputed,
        }
    }
}

/// The weakest (lowest-value) confidence across the cited edges. A chain
/// is only as strong as its weakest link (synthesis design section 6/7).
/// Callers only invoke this on a non-empty edge set.
fn weakest_conf(edges: &[CitedEdge]) -> Confidence {
    edges
        .iter()
        .map(|e| e.confidence)
        .fold(Confidence::Terminal, |acc, c| if c.value() < acc.value() { c } else { acc })
}

/// Component 3 signal test, split out from `IdkSignals` so it is unit
/// testable without constructing the full signal struct. Grounded-but
/// weak == low top activation AND a low anchored-token ratio.
fn thin_from(max_activation: f32, anchored: usize, content: usize) -> bool {
    if content == 0 {
        return false;
    }
    let ratio = anchored as f32 / content as f32;
    max_activation < THIN_MAX_ACT && ratio < THIN_ANCHOR_RATIO
}

/// True when the prose already carries its own hedge -- integrity answers
/// ("depends on what you value"), Component-1 multifactorial framing
/// ("Contributing factors"), or a connective this pass already softened
/// ("associated with" / "linked to"). Suppresses a redundant register
/// frame so an already-humble Phase-1/2 shape is never double-hedged or
/// regressed.
fn already_hedged(prose: &str) -> bool {
    const MARKERS: &[&str] = &[
        "multifactorial",
        "Contributing factors",
        "single cause",
        "associated with",
        "linked to",
        "reported",
        "depends on what you value",
        "no grounded basis",
        "claimed by some",
    ];
    MARKERS.iter().any(|m| prose.contains(m))
}

/// Weaken an assertive causal connective to the register's hedged form.
/// Only ever runs for non-assertive registers, and only ever WEAKENS --
/// it rewrites a connective that is already in the prose, introducing no
/// new claim. The Component-1 multifactorial framing contains none of
/// these fragments, so it passes through unchanged.
fn soften_connectives(prose: &str, reg: Register) -> String {
    let repl = match reg {
        Register::Reported => "is linked to",
        Register::Associative => "is associated with",
        Register::Disputed => "is claimed by some to be linked to",
        Register::Assertive => return prose.to_string(),
    };
    let spaced = format!(" {} ", repl);
    prose
        .replace(" is caused by ", &spaced)
        .replace(" can cause ", &spaced)
        .replace(" causes ", &spaced)
        .replace(" leads to ", &spaced)
}

/// Public entry: calibrate a conclusion to the confidence of its evidence
/// and the strength of its recall. `sig` is the ask's idk signals (None
/// degrades to "not thin").
pub fn recast(c: Conclusion, sig: Option<&IdkSignals>) -> Conclusion {
    let thin = sig
        .map(|s| thin_from(s.max_activation, s.anchored_token_count, s.content_token_count))
        .unwrap_or(false);
    recast_calibrated(c, thin)
}

/// Pure calibration core (Component 2/3/4). Split from [`recast`] so the
/// thin/register logic is testable without an `IdkSignals`.
fn recast_calibrated(mut c: Conclusion, thin: bool) -> Conclusion {
    // Honesty floor: an ungrounded conclusion is a refusal / integrity
    // answer. Return it byte-identical -- the floor owns the wording.
    if c.cited_edges.is_empty() {
        return c;
    }

    // A Contested cited edge routes to the disputed register even when it
    // is not the numeric minimum: dispute dominates. (Ships mostly dark
    // today -- curation removed most contested material.)
    let reg = if c.cited_edges.iter().any(|e| e.confidence == Confidence::Contested) {
        Register::Disputed
    } else {
        Register::from_conf(weakest_conf(&c.cited_edges))
    };

    // Body calibration: weaken the assertive connective for non-assertive
    // registers. Assertive prose is left byte-identical.
    if reg != Register::Assertive {
        c.prose = soften_connectives(&c.prose, reg);
    }

    // Frame prefix (additive hedge). Suppressed when the body already
    // hedges. Ordering: Disputed > thin > register register frame.
    let hedged_body = already_hedged(&c.prose);
    let frame: Option<&str> = if reg == Register::Disputed {
        Some("This is contested -- parts of what I hold dispute it, but on record: ")
    } else if thin {
        Some("This isn't fully understood from what I hold, but here's what's on record: ")
    } else if reg == Register::Reported {
        // OP-GLOSS (John directive): single-source answers read as natural
        // prose -- no stiff front-loaded "From a single source on record,"
        // qualifier. Honesty holds structurally (the body is never rewritten
        // to imply corroboration; non-assertive connective softening still
        // applies). Single-source is signalled subtly/elsewhere, not as a
        // lead-in. => no frame.
        None
    } else if reg == Register::Associative && !hedged_body {
        Some("From what I hold this is an association rather than an established fact: ")
    } else {
        None
    };

    if let Some(f) = frame {
        // Drop a now-redundant assertive "Yes -- " opener so the frame
        // carries the stance (a flat "Yes" would overclaim under a hedge).
        let body = c.prose.strip_prefix("Yes -- ").unwrap_or(&c.prose).to_string();
        c.prose = format!("{}{}", f, body);
    }

    c
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bond::BondType;
    use crate::synthesis::{Conclusion, ReasoningKind};

    fn edge(bond: BondType, conf: Confidence) -> CitedEdge {
        CitedEdge { from: "x".into(), bond, to: "y".into(), confidence: conf }
    }

    fn concl(kind: ReasoningKind, prose: &str, edges: Vec<CitedEdge>) -> Conclusion {
        let confidence = edges.iter().map(|e| e.confidence.value()).fold(1.0_f32, f32::min);
        Conclusion { kind, prose: prose.into(), cited_edges: edges, confidence }
    }

    // --- Honesty invariant ---------------------------------------------

    #[test]
    fn refusal_untouched() {
        // Empty cited_edges == refusal; recast must not add or remove text.
        let c = concl(ReasoningKind::CausalWhy, "Not from what I have -- I hold no causal link.", vec![]);
        let out = recast_calibrated(c.clone(), true);
        assert_eq!(out.prose, c.prose, "refusal prose is byte-identical");
        assert!(out.cited_edges.is_empty());
    }

    #[test]
    fn cited_edges_never_mutated() {
        let c = concl(
            ReasoningKind::EntityFact,
            "Einstein is a physicist.",
            vec![edge(BondType::IsA, Confidence::Unverified)],
        );
        let out = recast_calibrated(c, false);
        assert_eq!(out.cited_edges.len(), 1);
        assert_eq!(out.cited_edges[0].confidence, Confidence::Unverified);
    }

    // --- Component 2: register mapping ---------------------------------

    #[test]
    fn verified_stays_assertive() {
        let c = concl(
            ReasoningKind::CategoryMembership,
            "Yes -- a dog is a kind of animal. (dog -IsA-> animal)",
            vec![edge(BondType::IsA, Confidence::Verified)],
        );
        let out = recast_calibrated(c.clone(), false);
        assert_eq!(out.prose, c.prose, "Verified IsA is left crisp");
    }

    #[test]
    fn terminal_stays_assertive() {
        let c = concl(
            ReasoningKind::PropertyCheck,
            "Yes -- a dog is loyal. (dog -HasProperty-> loyal)",
            vec![edge(BondType::HasProperty, Confidence::Terminal)],
        );
        let out = recast_calibrated(c.clone(), false);
        assert_eq!(out.prose, c.prose);
    }

    #[test]
    fn unverified_entity_reads_reported() {
        let c = concl(
            ReasoningKind::EntityFact,
            "Albert Einstein is a physicist. Known for relativity. Born in Ulm.",
            vec![edge(BondType::IsA, Confidence::Unverified)],
        );
        let out = recast_calibrated(c, false);
        // OP-GLOSS: single-source (Reported) no longer front-loads a stiff
        // provenance qualifier; the answer reads as clean natural prose.
        assert!(!out.prose.to_lowercase().contains("single source"), "got: {}", out.prose);
        assert!(out.prose.starts_with("Albert Einstein is a physicist"), "got: {}", out.prose);
    }

    #[test]
    fn inferred_causal_reads_associated_not_causes() {
        // Single-factor causal at Inferred confidence must not overclaim.
        let c = concl(
            ReasoningKind::CausalWhy,
            "Cancer is caused by asbestos. (asbestos -Causes-> cancer)",
            vec![edge(BondType::Causes, Confidence::Inferred)],
        );
        let out = recast_calibrated(c, false);
        assert!(out.prose.contains("is associated with"), "got: {}", out.prose);
        assert!(!out.prose.contains("is caused by"));
    }

    #[test]
    fn unverified_causal_reads_linked() {
        let c = concl(
            ReasoningKind::CausalWhy,
            "Smoking can cause emphysema. (smoking -Causes-> emphysema)",
            vec![edge(BondType::Causes, Confidence::Unverified)],
        );
        let out = recast_calibrated(c, false);
        assert!(out.prose.contains("is linked to"), "got: {}", out.prose);
        assert!(!out.prose.contains("can cause"));
    }

    #[test]
    fn contested_reads_disputed() {
        let c = concl(
            ReasoningKind::CausalWhy,
            "X causes Y. (X -Causes-> Y)",
            vec![edge(BondType::Causes, Confidence::Contested)],
        );
        let out = recast_calibrated(c, false);
        assert!(out.prose.starts_with("This is contested"), "got: {}", out.prose);
    }

    #[test]
    fn contested_edge_dominates_even_with_verified() {
        let c = concl(
            ReasoningKind::CausalWhy,
            "X causes Y.",
            vec![
                edge(BondType::Causes, Confidence::Verified),
                edge(BondType::Causes, Confidence::Contested),
            ],
        );
        let out = recast_calibrated(c, false);
        assert!(out.prose.starts_with("This is contested"), "got: {}", out.prose);
    }

    // --- Component 1 interaction: multifactorial passes through --------

    #[test]
    fn multifactorial_framing_preserved_when_assertive() {
        let prose = "There isn't a single cause of cancer -- it's multifactorial. \
                     Contributing factors on record include alcohol and asbestos.";
        let c = concl(
            ReasoningKind::CausalWhy,
            prose,
            vec![edge(BondType::Causes, Confidence::Verified)],
        );
        let out = recast_calibrated(c.clone(), false);
        assert_eq!(out.prose, c.prose, "Verified multifactorial is left as Component 1 wrote it");
    }

    #[test]
    fn multifactorial_not_double_framed_when_inferred() {
        // Even at Inferred, the multifactorial body self-hedges -> no frame.
        let prose = "There isn't a single cause of cancer -- it's multifactorial. \
                     Contributing factors on record include alcohol.";
        let c = concl(
            ReasoningKind::CausalWhy,
            prose,
            vec![edge(BondType::Causes, Confidence::Inferred)],
        );
        let out = recast_calibrated(c.clone(), false);
        assert_eq!(out.prose, c.prose);
    }

    // --- Component 3: thin hedge ---------------------------------------

    #[test]
    fn thin_grounded_gets_hedge() {
        let c = concl(
            ReasoningKind::PropertyCheck,
            "Yes -- a quark is charged. (quark -HasProperty-> charged)",
            vec![edge(BondType::HasProperty, Confidence::Verified)],
        );
        let out = recast_calibrated(c, true);
        assert!(out.prose.starts_with("This isn't fully understood"), "got: {}", out.prose);
        assert!(!out.prose.contains("Yes -- "), "assertive opener dropped under hedge");
    }

    #[test]
    fn not_thin_no_hedge() {
        let c = concl(
            ReasoningKind::PropertyCheck,
            "Yes -- a dog is loyal.",
            vec![edge(BondType::HasProperty, Confidence::Verified)],
        );
        let out = recast_calibrated(c.clone(), false);
        assert_eq!(out.prose, c.prose);
    }

    #[test]
    fn thin_signal_thresholds() {
        // Healthy recall (>=0.42) is never thin.
        assert!(!thin_from(0.49, 0, 2));
        assert!(!thin_from(0.60, 0, 2));
        // Genuinely weak top activation is thin.
        assert!(thin_from(0.20, 0, 2));
        // No content tokens -> not thin (avoids div-by-zero).
        assert!(!thin_from(0.0, 0, 0));
    }

    // --- Component 4: one path, all shapes -----------------------------

    #[test]
    fn comparative_integrity_answer_not_reframed() {
        // Comparative min-edge is Inferred, but the body self-hedges
        // ("depends on what you value") -> no frame, no regression.
        let prose = "Dog is active. Fish is small. Which is \"better\" depends on what you value -- \
                     I have no grounded basis to pick a winner.";
        let c = concl(
            ReasoningKind::Comparative,
            prose,
            vec![
                edge(BondType::HasProperty, Confidence::Verified),
                edge(BondType::HasProperty, Confidence::Inferred),
            ],
        );
        let out = recast_calibrated(c.clone(), false);
        assert_eq!(out.prose, c.prose);
    }

    #[test]
    fn reported_composition_reads_clean_no_frame() {
        let c = concl(
            ReasoningKind::Composition,
            "Water is made of hydrogen and oxygen.",
            vec![edge(BondType::PartOf, Confidence::Unverified)],
        );
        let out = recast_calibrated(c, false);
        // OP-GLOSS: no stiff single-source lead-in; body stands as-is.
        assert!(!out.prose.to_lowercase().contains("single source"), "got: {}", out.prose);
        assert_eq!(out.prose, "Water is made of hydrogen and oxygen.");
    }
}
