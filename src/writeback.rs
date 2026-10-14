//! OP-WRITEBACK Phase 5 — compounding writeback + the self-learning immune
//! system. This is the first subsystem that lets Lu turn a *reasoned*
//! conclusion into a persisted bond, under strict discipline:
//!
//!   * OWNER-SCOPED. A guest can never trigger writeback (privsep mirror of
//!     the guest binary having no teach tool).
//!   * EPHEMERAL AT SERVING. Conclusions produced while answering a query are
//!     Session/Current-scoped keyframes only — never persisted. Durable
//!     writeback happens only in the owner maintenance window, on a COPY of
//!     the canonical brain, never on the live reader.
//!   * STAGE-0 CALCIFICATION GATE. A durable conclusion is not written
//!     directly; it is queued into the `pending::PendingLog` review queue and
//!     nothing persists until it is explicitly approved + applied.
//!   * HONEST. A derived bond is stamped `Confidence::Inferred` with a
//!     `KeyframeSource::ConcludedFrom { rule, cited_edges }` lineage, so it is
//!     always distinguishable from an ingested fact and never asserted as one.
//!   * SELF-CORRECTING. The immune system promotes Inferred->Verified on
//!     independent corroboration, and demotes ->Contested->retract on
//!     contradiction (an Opposes/negation edge, or a Cattle Dog off-ness
//!     flag). The `cited_edges` lineage makes retraction SURGICAL: exactly the
//!     derived bonds whose premises changed are removed, cascading to anything
//!     derived from them; ingested facts are NEVER touched.

use crate::bond::{
    confidence_from_keyframe_source, deciding_keyframe_source, DerivationRule, KeyframeSource,
    PremiseEdge, TemporalBond, TemporalKeyframe, BondType, DEFAULT_HALF_LIFE_MS,
};
use crate::brain::Brain;
use crate::node::Confidence;
use crate::pending::{PendingChange, PendingChangeKind, PendingStatus};
use crate::seed::Seed;
use crate::source::SourceAttribution;
use crate::synthesis::ReasoningKind;
use std::collections::HashMap;

/// A stable key for a directed typed edge (subject, bond-type byte, object).
pub type EdgeKey = (Seed, u8, Seed);

#[inline]
pub fn ekey(subject: Seed, bt: BondType, object: Seed) -> EdgeKey {
    (subject, bt as u8, object)
}

/// Who is driving the request. Writeback is OWNER-scoped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Actor {
    Owner,
    Guest,
}

/// Where a conclusion is being emitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmitMode {
    /// Serving a query -> EPHEMERAL only (never persisted).
    Serving,
    /// Owner maintenance window -> durable persistence allowed, but only
    /// through the Stage-0 review queue on a COPY of the canonical.
    OwnerWindow,
}

/// The concrete triple a reasoning shape concluded, plus the real premise
/// edges it rests on (its lineage).
#[derive(Debug, Clone)]
pub struct DerivedConclusion {
    pub subject: Seed,
    pub bond_type: BondType,
    pub object: Seed,
    pub rule: DerivationRule,
    pub premises: Vec<PremiseEdge>,
    pub weight: f32,
    pub subject_label: String,
    pub object_label: String,
}

impl DerivedConclusion {
    pub fn key(&self) -> EdgeKey {
        ekey(self.subject, self.bond_type, self.object)
    }
    pub fn keyframe_source(&self) -> KeyframeSource {
        KeyframeSource::ConcludedFrom {
            rule: self.rule,
            cited_edges: self.premises.clone(),
        }
    }
}

/// Map a synthesis shape to the derivation rule that named it.
pub fn rule_for_kind(kind: ReasoningKind) -> DerivationRule {
    match kind {
        ReasoningKind::PropertyInheritance | ReasoningKind::PropertyCheck => {
            DerivationRule::PropertyInheritance
        }
        ReasoningKind::CategoryMembership | ReasoningKind::CommonAncestor => {
            DerivationRule::Transitivity
        }
        ReasoningKind::CausalWhy => DerivationRule::CausalChain,
        ReasoningKind::Composition | ReasoningKind::Meronymy => DerivationRule::Meronymy,
        _ => DerivationRule::Other,
    }
}

/// Outcome of an emit decision.
#[derive(Debug, Clone)]
pub enum WritebackDecision {
    /// Guest actor or no cited premise -> nothing written.
    Skipped(&'static str),
    /// Serving path -> an ephemeral keyframe (NOT persisted). Session/Current
    /// scoped; carries the source it would be stamped with.
    Ephemeral { source: KeyframeSource },
    /// Owner window -> queued into the Stage-0 review log. Must be approved +
    /// applied before anything durable happens.
    Queued { pending_id: u64 },
}

/// A contradiction signal that can catch a bad derived belief.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Contradiction {
    None,
    /// A later authoritative fact marks the derived bond negated.
    Negated,
    /// An Opposes edge exists between the endpoints.
    Opposes,
    /// Cattle Dog off-ness flagged the belief as ungrounded / off.
    OffnessFlag,
}

/// The writeback + immune subsystem state (owner-side, lives on the COPY).
#[derive(Default)]
pub struct Writeback {
    /// pending_id -> derived conclusion awaiting Stage-0 review.
    queued: HashMap<u64, DerivedConclusion>,
    /// derived edge -> its conclusion (only edges WE derived).
    derived: HashMap<EdgeKey, DerivedConclusion>,
    /// reverse lineage: premise edge -> derived edges that cite it.
    dependents: HashMap<EdgeKey, Vec<EdgeKey>>,
}

impl Writeback {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_derived(&self, edge: &EdgeKey) -> bool {
        self.derived.contains_key(edge)
    }
    pub fn derived_count(&self) -> usize {
        self.derived.len()
    }

    /// Emit a reasoned conclusion. Owner-scoped; ephemeral at serving; Stage-0
    /// queued in the owner window. Never persists directly.
    pub fn emit(
        &mut self,
        brain: &mut Brain,
        actor: Actor,
        mode: EmitMode,
        concl: &DerivedConclusion,
        now_ms: i64,
    ) -> WritebackDecision {
        // PRIVSEP: guests never trigger writeback.
        if actor != Actor::Owner {
            return WritebackDecision::Skipped("guest: writeback is owner-scoped");
        }
        // Must rest on >=1 REAL cited premise.
        if concl.premises.is_empty() {
            return WritebackDecision::Skipped("no cited premise");
        }
        match mode {
            EmitMode::Serving => WritebackDecision::Ephemeral {
                source: concl.keyframe_source(),
            },
            EmitMode::OwnerWindow => {
                let change = PendingChange {
                    pending_id: 0,
                    proposed_at_ms: now_ms as u64,
                    proposed_by: SourceAttribution::research(
                        "reasoner:concluded",
                        None,
                        now_ms as u64,
                    ),
                    change_kind: PendingChangeKind::AddBond {
                        subject_text: concl.subject_label.clone(),
                        object_text: concl.object_label.clone(),
                        bond_type: concl.bond_type,
                        weight: concl.weight,
                        negated: false,
                    },
                    confidence: Confidence::Inferred.value(),
                    reason: format!(
                        "Concluded via {:?} from {} premise(s): {}",
                        concl.rule,
                        concl.premises.len(),
                        premise_summary(&concl.premises)
                    ),
                    status: PendingStatus::Pending,
                };
                let id = brain.pending.push(change).expect("push pending");
                self.queued.insert(id, concl.clone());
                WritebackDecision::Queued { pending_id: id }
            }
        }
    }

    /// Stage-0: approve a queued conclusion and persist it durably as an
    /// Inferred bond carrying its ConcludedFrom lineage. Rejects (or an
    /// un-approved id) persist NOTHING.
    pub fn approve_and_persist(
        &mut self,
        brain: &mut Brain,
        pending_id: u64,
        approver: &str,
        now_ms: i64,
    ) -> Result<EdgeKey, String> {
        let concl = self
            .queued
            .get(&pending_id)
            .cloned()
            .ok_or_else(|| "unknown pending id".to_string())?;
        brain
            .pending
            .approve(pending_id, now_ms as u64, approver)
            .map_err(|e| format!("approve: {:?}", e))?;
        let key = self.persist(brain, &concl, &[], now_ms);
        brain
            .pending
            .mark_applied(pending_id, now_ms as u64)
            .map_err(|e| format!("mark_applied: {:?}", e))?;
        self.queued.remove(&pending_id);
        Ok(key)
    }

    /// Persist a derived bond with its ConcludedFrom lineage keyframe, and
    /// index it for corroboration / surgical retraction.
    fn persist(
        &mut self,
        brain: &mut Brain,
        concl: &DerivedConclusion,
        extra_kf: &[TemporalKeyframe],
        now_ms: i64,
    ) -> EdgeKey {
        let mut keyframes = vec![TemporalKeyframe {
            timestamp: now_ms,
            weight: concl.weight,
            source: concl.keyframe_source(),
        }];
        keyframes.extend_from_slice(extra_kf);
        let bond = TemporalBond {
            from: concl.subject,
            to: concl.object,
            bond_type: concl.bond_type,
            keyframes,
            half_life_ms: DEFAULT_HALF_LIFE_MS,
        };
        brain.network.add_bond(bond).expect("add derived bond");
        let key = concl.key();
        self.derived.insert(key, concl.clone());
        for p in &concl.premises {
            self.dependents
                .entry(ekey(p.subject, p.predicate, p.object))
                .or_default()
                .push(key);
        }
        key
    }

    /// IMMUNE SYSTEM — corroboration. An INDEPENDENT re-derivation (a
    /// different rule/path) or a later agreeing ingested fact promotes the
    /// derived bond Inferred->Verified by appending a Corroboration keyframe.
    /// Returns true iff a promotion occurred.
    pub fn corroborate(
        &mut self,
        brain: &mut Brain,
        edge: EdgeKey,
        via: DerivationRule,
        independent_premises: Vec<PremiseEdge>,
        now_ms: i64,
    ) -> bool {
        let concl = match self.derived.get(&edge) {
            Some(c) => c.clone(),
            None => return false,
        };
        // Independence: the corroborating rule must differ from the original.
        if via == concl.rule {
            return false;
        }
        let mut kfs = match brain.network.get_bond(&edge.0, &edge.2) {
            Some(bv) => bv.keyframes_iter().collect::<Vec<_>>(),
            None => return false,
        };
        kfs.push(TemporalKeyframe {
            timestamp: now_ms,
            weight: concl.weight.max(0.6),
            source: KeyframeSource::ConcludedFrom {
                rule: DerivationRule::Corroboration,
                cited_edges: independent_premises,
            },
        });
        brain.network.remove_bond(&edge.0, &edge.2);
        let bond = TemporalBond {
            from: edge.0,
            to: edge.2,
            bond_type: concl.bond_type,
            keyframes: kfs,
            half_life_ms: DEFAULT_HALF_LIFE_MS,
        };
        brain.network.add_bond(bond).expect("re-add corroborated bond");
        true
    }

    /// The belief state of a derived edge, read from its keyframes: Verified
    /// once corroborated, Inferred while single-derivation, or the honest
    /// keyframe-derived fallback.
    pub fn belief_state(&self, brain: &Brain, edge: &EdgeKey) -> Option<Confidence> {
        let bv = brain.network.get_bond(&edge.0, &edge.2)?;
        Some(belief_from_keyframes(&bv.keyframes_iter().collect::<Vec<_>>()))
    }

    /// IMMUNE SYSTEM — detect a contradiction against a derived belief.
    /// `offness_off` lets the caller feed a Cattle Dog off-ness verdict.
    pub fn detect_contradiction(
        &self,
        brain: &Brain,
        edge: &EdgeKey,
        bond_type: BondType,
        offness_off: bool,
    ) -> Contradiction {
        if offness_off {
            return Contradiction::OffnessFlag;
        }
        if brain.bond_negated(&edge.0, &edge.2, bond_type) {
            return Contradiction::Negated;
        }
        // An Opposes edge between the same endpoints.
        if let Some(bv) = brain.network.get_bond(&edge.0, &edge.2) {
            if bv.bond_type() == BondType::Opposes {
                return Contradiction::Opposes;
            }
        }
        Contradiction::None
    }

    /// IMMUNE SYSTEM — surgical retraction. Removes exactly the derived bond
    /// whose premise changed, then CASCADES to any derived bond that cited it
    /// as a premise. Ingested facts (not in `self.derived`) are NEVER removed.
    /// Returns the list of retracted derived edges (in retraction order).
    pub fn retract(&mut self, brain: &mut Brain, edge: EdgeKey) -> Vec<EdgeKey> {
        let mut out = Vec::new();
        self.retract_rec(brain, edge, &mut out);
        out
    }

    fn retract_rec(&mut self, brain: &mut Brain, edge: EdgeKey, out: &mut Vec<EdgeKey>) {
        // Surgical: only ever remove edges WE derived.
        if !self.derived.contains_key(&edge) {
            return;
        }
        brain.network.remove_bond(&edge.0, &edge.2);
        self.derived.remove(&edge);
        out.push(edge);
        if let Some(deps) = self.dependents.remove(&edge) {
            for dep in deps {
                self.retract_rec(brain, dep, out);
            }
        }
    }
}

/// Order-insensitive: is this edge currently a persisted DERIVED bond in the
/// brain (deciding keyframe is ConcludedFrom)? Distinguishes reasoned
/// conclusions from ingested facts at the byte level.
pub fn edge_is_concluded(brain: &Brain, from: &Seed, to: &Seed) -> bool {
    match brain.network.get_bond(from, to) {
        Some(bv) => matches!(
            deciding_keyframe_source(&bv.keyframes_iter().collect::<Vec<_>>()),
            Some(KeyframeSource::ConcludedFrom { .. })
        ),
        None => false,
    }
}

/// Belief state from a keyframe history (immune-aware).
pub fn belief_from_keyframes(keyframes: &[TemporalKeyframe]) -> Confidence {
    let mut has_concluded = false;
    let mut has_corroboration = false;
    for kf in keyframes {
        if let KeyframeSource::ConcludedFrom { rule, .. } = &kf.source {
            has_concluded = true;
            if *rule == DerivationRule::Corroboration {
                has_corroboration = true;
            }
        }
    }
    if has_corroboration {
        Confidence::Verified
    } else if has_concluded {
        Confidence::Inferred
    } else {
        // Non-derived: honest keyframe-source mapping, else Unverified.
        deciding_keyframe_source(keyframes)
            .and_then(confidence_from_keyframe_source)
            .unwrap_or(Confidence::Unverified)
    }
}

/// Phase-3 payoff — per-EDGE read-time confidence. Reads the bond behind a
/// cited edge and derives its confidence from the bond's own keyframe
/// provenance (immune-aware). Falls back to `node_fallback` ONLY when the
/// bond carries no tier-fixing provenance, so existing (Baseline) knowledge
/// reads exactly as before — no SST rewrite, no fabrication.
pub fn edge_confidence(
    brain: &Brain,
    from: &Seed,
    to: &Seed,
    node_fallback: Confidence,
) -> Confidence {
    if let Some(bv) = brain.network.get_bond(from, to) {
        // Corroborated / concluded -> immune-aware tier.
        let kfs: Vec<TemporalKeyframe> = bv.keyframes_iter().collect();
        let mut has_concluded = false;
        let mut has_corroboration = false;
        for kf in &kfs {
            if let KeyframeSource::ConcludedFrom { rule, .. } = &kf.source {
                has_concluded = true;
                if *rule == DerivationRule::Corroboration {
                    has_corroboration = true;
                }
            }
        }
        if has_corroboration {
            return Confidence::Verified;
        }
        if has_concluded {
            return Confidence::Inferred;
        }
        // Otherwise honour a fixed keyframe-source mapping if present.
        if let Some(c) =
            deciding_keyframe_source(&kfs).and_then(confidence_from_keyframe_source)
        {
            return c;
        }
    }
    node_fallback
}

fn premise_summary(premises: &[PremiseEdge]) -> String {
    premises
        .iter()
        .map(|p| format!("{:?}", p.predicate))
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bond::{BondType, PremiseEdge};
    use crate::brain::Brain;
    use crate::embedding::Embedder;
    use crate::node::Confidence;
    use crate::offness::{assess, GroundedEvidence, OffnessConfig};

    // Deterministic mock embedder over a tiny fixed dictionary: disjoint token
    // sets embed to (near-)orthogonal vectors, so cosine cleanly straddles the
    // 0.70 off-ness floor. Exercises the REAL Cattle Dog assess().
    struct MockEmbedder;
    impl MockEmbedder {
        fn slot(tok: &str) -> Option<usize> {
            match tok {
                "penguins" => Some(0), "fly" => Some(1), "can" => Some(2),
                "flightless" => Some(3), "ice" => Some(4), "cold" => Some(5),
                "birds" => Some(6), "are" => Some(7),
                _ => None,
            }
        }
    }
    impl Embedder for MockEmbedder {
        fn embed(&self, text: &str) -> Vec<f32> {
            let mut v = vec![0.0f32; 8];
            for tok in text.to_lowercase().split(|c: char| !c.is_alphanumeric()) {
                if let Some(i) = Self::slot(tok) {
                    v[i] = 1.0;
                }
            }
            v
        }
        fn dim(&self) -> usize { 8 }
        fn name(&self) -> &'static str { "mock" }
    }

    fn base(b: &mut Brain) -> (crate::seed::Seed, crate::seed::Seed, crate::seed::Seed, Vec<PremiseEdge>) {
        // Ingested ground facts.
        let r1 = b.teach("penguin", BondType::IsA, "bird", 0.9, Confidence::Verified);
        let r2 = b.teach("bird", BondType::HasProperty, "flies", 0.9, Confidence::Verified);
        let penguin = r1.subject_seed;
        let bird = r1.object_seed;
        let flies = r2.object_seed;
        let premises = vec![
            PremiseEdge { subject: penguin, predicate: BondType::IsA, object: bird },
            PremiseEdge { subject: r2.subject_seed, predicate: BondType::HasProperty, object: flies },
        ];
        (penguin, bird, flies, premises)
    }

    fn faulty(penguin: crate::seed::Seed, flies: crate::seed::Seed, premises: Vec<PremiseEdge>) -> DerivedConclusion {
        DerivedConclusion {
            subject: penguin, bond_type: BondType::HasProperty, object: flies,
            rule: DerivationRule::PropertyInheritance, premises, weight: 0.7,
            subject_label: "penguin".into(), object_label: "flies".into(),
        }
    }

    #[test]
    fn guest_can_never_writeback() {
        let mut b = Brain::new();
        let mut wb = Writeback::new();
        let (p, f, prem) = { let (p,_,f,prem) = base(&mut b); (p,f,prem) };
        let dc = faulty(p, f, prem);
        let d = wb.emit(&mut b, Actor::Guest, EmitMode::OwnerWindow, &dc, 1000);
        assert!(matches!(d, WritebackDecision::Skipped(_)));
        assert_eq!(b.pending.pending_count(), 0);
        assert_eq!(wb.derived_count(), 0);
    }

    #[test]
    fn serving_conclusions_are_ephemeral() {
        let mut b = Brain::new();
        let mut wb = Writeback::new();
        let (p, _bird, f, prem) = base(&mut b);
        let pending_before = b.pending.pending_count();
        let bonds_before = b.network.bond_count();
        let dc = faulty(p, f, prem);
        let d = wb.emit(&mut b, Actor::Owner, EmitMode::Serving, &dc, 1000);
        match d {
            WritebackDecision::Ephemeral { source } => {
                assert!(matches!(source, KeyframeSource::ConcludedFrom { .. }));
            }
            _ => panic!("serving must be ephemeral"),
        }
        // NOTHING persisted: no pending entry, no new bond, not in derived set.
        assert_eq!(b.pending.pending_count(), pending_before);
        assert_eq!(b.network.bond_count(), bonds_before);
        assert!(b.network.get_bond(&p, &f).is_none());
        assert_eq!(wb.derived_count(), 0);
    }

    #[test]
    fn stage0_gate_blocks_until_approved_then_inferred_and_distinguishable() {
        let mut b = Brain::new();
        let mut wb = Writeback::new();
        let (p, bird, f, prem) = base(&mut b);
        let dc = faulty(p, f, prem);
        let d = wb.emit(&mut b, Actor::Owner, EmitMode::OwnerWindow, &dc, 1000);
        let id = match d { WritebackDecision::Queued { pending_id } => pending_id, _ => panic!() };
        // Stage-0: queued, NOT durable.
        assert_eq!(b.pending.pending_count(), 1);
        assert!(b.network.get_bond(&p, &f).is_none(), "nothing durable before approval");
        // Approve + persist.
        let key = wb.approve_and_persist(&mut b, id, "john", 1001).unwrap();
        assert!(b.network.get_bond(&p, &f).is_some(), "persisted after approval");
        // Derived bond reads Inferred and is byte-distinguishable from an ingested fact.
        assert_eq!(wb.belief_state(&b, &key), Some(Confidence::Inferred));
        assert!(edge_is_concluded(&b, &p, &f), "derived edge is ConcludedFrom");
        assert!(!edge_is_concluded(&b, &p, &bird), "ingested edge is NOT ConcludedFrom");
        assert!(b.pending.get(id).unwrap().status.is_applied());
    }

    #[test]
    fn rejected_conclusion_persists_nothing() {
        let mut b = Brain::new();
        let mut wb = Writeback::new();
        let (p, _bird, f, prem) = base(&mut b);
        let dc = faulty(p, f, prem);
        let d = wb.emit(&mut b, Actor::Owner, EmitMode::OwnerWindow, &dc, 1000);
        let id = match d { WritebackDecision::Queued { pending_id } => pending_id, _ => panic!() };
        b.pending.reject(id, 1001, "john", Some("over-broad inheritance".into())).unwrap();
        // No approve_and_persist -> still nothing durable.
        assert!(b.network.get_bond(&p, &f).is_none());
        assert_eq!(wb.derived_count(), 0);
    }

    #[test]
    fn corroboration_promotes_inferred_to_verified() {
        let mut b = Brain::new();
        let mut wb = Writeback::new();
        let (p, bird, f, prem) = base(&mut b);
        let dc = faulty(p, f, prem);
        let id = match wb.emit(&mut b, Actor::Owner, EmitMode::OwnerWindow, &dc, 1000) {
            WritebackDecision::Queued { pending_id } => pending_id, _ => panic!() };
        let key = wb.approve_and_persist(&mut b, id, "john", 1001).unwrap();
        assert_eq!(wb.belief_state(&b, &key), Some(Confidence::Inferred));
        // Independent re-derivation via a DIFFERENT rule.
        let independent = vec![PremiseEdge { subject: p, predicate: BondType::IsA, object: bird }];
        let promoted = wb.corroborate(&mut b, key, DerivationRule::Transitivity, independent, 1002);
        assert!(promoted, "independent corroboration should promote");
        assert_eq!(wb.belief_state(&b, &key), Some(Confidence::Verified));
        // Same-rule 're-derivation' is NOT independent -> no further change.
        let same = wb.corroborate(&mut b, key, DerivationRule::PropertyInheritance, vec![], 1003);
        assert!(!same);
    }

    #[test]
    fn contradiction_via_negation_retracts_surgically_and_cascades() {
        let mut b = Brain::new();
        let mut wb = Writeback::new();
        let (p, bird, f, prem) = base(&mut b);
        // Create a node for the cascade target.
        let rf = b.teach("bird", BondType::RelatedTo, "flyer", 0.5, Confidence::Unverified);
        let flyer = rf.object_seed;
        // derived1: penguin HasProperty flies (rests on ingested premises)
        let dc1 = faulty(p, f, prem);
        let id1 = match wb.emit(&mut b, Actor::Owner, EmitMode::OwnerWindow, &dc1, 1000) {
            WritebackDecision::Queued { pending_id } => pending_id, _ => panic!() };
        let key1 = wb.approve_and_persist(&mut b, id1, "john", 1001).unwrap();
        // derived2: penguin IsA flyer, resting on derived1 as its premise.
        let dc2 = DerivedConclusion {
            subject: p, bond_type: BondType::IsA, object: flyer,
            rule: DerivationRule::Transitivity,
            premises: vec![PremiseEdge { subject: p, predicate: BondType::HasProperty, object: f }],
            weight: 0.6, subject_label: "penguin".into(), object_label: "flyer".into(),
        };
        let id2 = match wb.emit(&mut b, Actor::Owner, EmitMode::OwnerWindow, &dc2, 1002) {
            WritebackDecision::Queued { pending_id } => pending_id, _ => panic!() };
        let key2 = wb.approve_and_persist(&mut b, id2, "john", 1003).unwrap();
        assert!(b.network.get_bond(&p, &f).is_some());
        assert!(b.network.get_bond(&p, &flyer).is_some());
        // CATCH: owner learns penguins do NOT fly (negation, no positive bond added).
        b.set_bond_negated(p, f, BondType::HasProperty, true);
        let c = wb.detect_contradiction(&b, &key1, BondType::HasProperty, false);
        assert_eq!(c, Contradiction::Negated, "contradiction must be caught");
        // SURGICAL RETRACT + CASCADE.
        let retracted = wb.retract(&mut b, key1);
        assert!(retracted.contains(&key1) && retracted.contains(&key2), "derived1 + cascade derived2");
        assert!(b.network.get_bond(&p, &f).is_none(), "derived1 removed");
        assert!(b.network.get_bond(&p, &flyer).is_none(), "derived2 cascaded");
        // INGESTED FACTS SURVIVE — retraction never touches them.
        assert!(b.network.get_bond(&p, &bird).is_some(), "penguin IsA bird survives");
        assert!(b.network.get_bond(&bird, &f).is_some(), "bird HasProperty flies survives");
        assert_eq!(wb.derived_count(), 0);
    }

    #[test]
    fn contradiction_via_cattledog_offness_flag() {
        // Exercises the REAL offness::assess: an ungrounded derived answer is
        // flagged off, feeding detect_contradiction -> retract.
        let mut b = Brain::new();
        let mut wb = Writeback::new();
        let (p, _bird, f, prem) = base(&mut b);
        let dc = faulty(p, f, prem);
        let id = match wb.emit(&mut b, Actor::Owner, EmitMode::OwnerWindow, &dc, 1000) {
            WritebackDecision::Queued { pending_id } => pending_id, _ => panic!() };
        let key = wb.approve_and_persist(&mut b, id, "john", 1001).unwrap();
        let cfg = OffnessConfig::default(); // floor 0.70
        let embedder = MockEmbedder;
        // Grounded evidence says penguins are flightless; the derived claim is
        // "penguins can fly" -> low similarity -> off.
        let evidence = vec![GroundedEvidence { text: "penguins are flightless birds".into(), weight: 1.0 }];
        let (best, is_off) = assess(&embedder, "penguins can fly", &evidence, &cfg);
        assert!(is_off, "offness must flag the ungrounded conclusion (best={})", best);
        let c = wb.detect_contradiction(&b, &key, BondType::HasProperty, is_off);
        assert_eq!(c, Contradiction::OffnessFlag);
        let retracted = wb.retract(&mut b, key);
        assert_eq!(retracted, vec![key]);
        assert!(b.network.get_bond(&p, &f).is_none());
        // Control: a grounded claim is NOT off.
        let (_b2, is_off2) = assess(&embedder, "penguins are flightless birds", &evidence, &cfg);
        assert!(!is_off2, "grounded claim must not be flagged off");
    }

    #[test]
    fn per_edge_confidence_sharpens_live() {
        let mut b = Brain::new();
        let mut wb = Writeback::new();
        let (p, bird, f, prem) = base(&mut b);
        // Ingested edge falls back to node/stored tier (here: caller-supplied).
        assert_eq!(edge_confidence(&b, &p, &bird, Confidence::Verified), Confidence::Verified);
        // Single-source derived edge reads Inferred (tentative).
        let dc = faulty(p, f, prem);
        let id = match wb.emit(&mut b, Actor::Owner, EmitMode::OwnerWindow, &dc, 1000) {
            WritebackDecision::Queued { pending_id } => pending_id, _ => panic!() };
        let key = wb.approve_and_persist(&mut b, id, "john", 1001).unwrap();
        assert_eq!(edge_confidence(&b, &p, &f, Confidence::Verified), Confidence::Inferred);
        // After corroboration it sharpens to Verified (established).
        let independent = vec![PremiseEdge { subject: p, predicate: BondType::IsA, object: bird }];
        assert!(wb.corroborate(&mut b, key, DerivationRule::Transitivity, independent, 1002));
        assert_eq!(edge_confidence(&b, &p, &f, Confidence::Verified), Confidence::Verified);
    }
}
