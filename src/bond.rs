//! Temporal bonds — the core innovation of Lu 4D.
//!
//! In Lu v2, bond weights are single f32 values. In Lu 4D, every bond
//! carries an append-only history of temporal keyframes. Weight at any
//! point in time is computed by finding the nearest keyframes and
//! applying exponential decay.
//!
//! As of the Scar port, every weight read is type-enforced to consult
//! the `ScarLog`. `weight_at(t, &ScarLog)` and
//! `weight_in_scope(scope, &ScarLog)` both return `ScarAwareReading`,
//! never a bare `f32`. There is no path through the type system to a
//! bond's weight that bypasses the scar check — the compiler refuses
//! it. Action layers consume `effective_weight`; observation layers
//! (time-travel, narration of moral state) consume both the
//! `historical_weight` and the `active_scars` list.

use crate::scar::{self, ScarAwareReading, ScarLog};
use crate::scope::TemporalScope;
use crate::seed::Seed;
use serde::{Deserialize, Serialize};

/// Default half-life for exponential decay, in milliseconds.
/// After this duration, a keyframe's influence is halved.
pub const DEFAULT_HALF_LIFE_MS: f64 = 86_400_000.0; // 24 hours

/// The source that created a keyframe.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
#[archive(check_bytes)]
pub enum KeyframeSource {
    /// Part of the baseline knowledge graph.
    Baseline,
    /// Created during a specific session.
    Session(u64),
    /// Created by Hebbian strengthening.
    Hebbian,
    /// Manually set by the user or system.
    Manual,
    /// Created by the consolidation synthesis pass (triangle closure).
    /// `from_path` records the [A, B, C] triple that justified the bond,
    /// so a later time-travel query can audit why this belief exists.
    Synthesis { from_path: [Seed; 3] },
    /// Created by the Thinking Layer as a candidate ("if A then B is
    /// relevant") at low initial weight. Promoted to ConfirmedHypothesis
    /// on confirmation, otherwise left to decay. `goal` and `step`
    /// identify which think pass / iteration proposed it; `session` is
    /// the active session at the time, if any.
    Hypothesis { goal: Seed, step: usize, session: Option<u64> },
    /// A previously-Hypothesis bond promoted by the Thinking Layer's
    /// reflection step. Higher initial weight than Hypothesis. `goal`
    /// and `step` mirror the original proposal; `session` carries
    /// through.
    ConfirmedHypothesis { goal: Seed, step: usize, session: Option<u64> },
    /// Created inside a Variant Generation arena. Visible ONLY when the
    /// active scope is `TemporalScope::Variant { id: <matching>, .. }`;
    /// `weight_at` filters these out so Baseline / Current / Session
    /// reads stay clean. `goal` is the source goal seed the variant is
    /// reasoning off of; `strategy` is the perturbation strategy as a
    /// small enum tag (see `variants::PerturbationStrategy`).
    Variant { id: u64, goal: Seed, strategy: u8 },
    /// A keyframe written by `Brain::promote_variant` that re-tags a
    /// winning variant's contribution as baseline knowledge. Visible to
    /// every scope (no source filter applies). `variant_id` is the
    /// winner's id; `original_strategy` records which perturbation
    /// strategy produced this bond, for later audit.
    PromotedFromVariant { variant_id: u64, original_strategy: u8 },
    /// Created inside a multi-agent thinking pass. Visible ONLY when the
    /// active scope is `TemporalScope::Agent { multi_think_id: <matching>, .. }`;
    /// `weight_at` filters these out so Baseline / Current / Session
    /// reads stay clean. `role` is `AgentRole::as_u8()` for the agent
    /// that wrote it; `multi_think_id` identifies the orchestrating
    /// call; `step` is the index of the work-item the keyframe records
    /// (e.g. the synthesized belief's rank in the Synthesis output).
    /// Agent-source keyframes age out via the consolidation prune pass
    /// and never auto-promote — multi-think is observation, not
    /// commitment.
    Agent { role: u8, multi_think_id: u64, step: u32 },
    /// OP-WRITEBACK Phase 5 -- a bond CONCLUDED by the reasoner from >=1
    /// real cited edge via a named rule. Distinguishes a derived belief
    /// from an ingested fact (carries Confidence::Inferred on the node,
    /// lineage here). `rule` names the inference; `cited_edges` are the
    /// exact premise edges, so retraction is surgical and cascades.
    /// Appended AFTER Agent to preserve rkyv/bincode discriminant and
    /// archived-size stability (existing on-disk archives decode
    /// byte-identically; proven in STOP-POINT 1).
    ConcludedFrom { rule: DerivationRule, cited_edges: Vec<PremiseEdge> },
}

/// The inference rule that produced a `KeyframeSource::ConcludedFrom` bond.
/// Fieldless (1-byte archived) so appending ConcludedFrom cannot grow the
/// KeyframeSource archived union past its current max variant (Synthesis).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
#[archive(check_bytes)]
pub enum DerivationRule {
    /// A IsA B, B IsA C => A IsA C
    Transitivity,
    /// A IsA B, B HasProperty P => A HasProperty P
    PropertyInheritance,
    /// A Causes B, B Causes C => A LeadsTo C
    CausalChain,
    /// A PartOf B, B PartOf C => A PartOf C
    Meronymy,
    /// Independent re-derivation agreeing (immune-system corroboration).
    Corroboration,
    /// Any reasoner-named rule not yet enumerated.
    Other,
}

/// One premise edge cited by a `KeyframeSource::ConcludedFrom` bond. Names
/// the exact bond (subject, predicate, object) that justified a conclusion,
/// forming the lineage the immune system walks to retract surgically when a
/// premise changes. Behind a `Vec` in the variant, so it contributes only an
/// `ArchivedVec` header (relptr+len) to the inline archived size.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
#[archive(check_bytes)]
pub struct PremiseEdge {
    pub subject: Seed,
    pub predicate: BondType,
    pub object: Seed,
}

/// A single temporal keyframe recording a bond weight at a point in time.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
#[archive(check_bytes)]
pub struct TemporalKeyframe {
    /// Timestamp when this keyframe was recorded (epoch millis).
    pub timestamp: i64,
    /// The bond weight at this point in time.
    pub weight: f32,
    /// What created this keyframe.
    pub source: KeyframeSource,
}

/// The bond type taxonomy — 37 variants across four epochs: 11 original Lu v2 semantic types, 17 added by BR.1 extensions, 1 added by BR.10b (HasArea), 8 Phase-1 code-structure variants. Each carries a base weight multiplier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
#[archive(check_bytes)]
#[archive_attr(derive(Hash, PartialEq, Eq))]
pub enum BondType {
    IsA,
    HasProperty,
    Causes,
    PartOf,
    UsedFor,
    RelatedTo,
    ContrastedWith,
    DerivedFrom,
    Requires,
    Produces,
    Opposes,
    // BR.1 first-class extension variants — appended after
    // existing variants to preserve discriminant stability
    // (rkyv archive on-disk format compat).
    BornIn,
    BornOn,
    DiedIn,
    DiedOn,
    DiscoveredBy,
    FoundedBy,
    FoundedIn,
    HasAltitude,
    HasAspect,
    HasCapital,
    HasOfficialLanguage,
    HasPopulation,
    KnownFor,
    LocatedIn,
    MeasuredIn,
    NamedAfter,
    WrittenBy,
    // BR.10b — geographic area attribute. Emitted by smart_extractor.py
    // L5-area pattern (e.g. 'Geography of Azerbaijan -> 605 km²') and
    // listed in preflight as a known relation. Appended after WrittenBy
    // to preserve discriminant stability for existing rkyv archives.
    HasArea,
    // Phase-1 code-structure extension variants (Python AST). Appended
    // after HasArea to preserve rkyv on-disk discriminant stability.
    HasMethod,
    HasAttribute,
    HasParameter,
    Returns,
    Inherits,
    Calls,
    HasType,
    Implements,
    // OP-EXTRACT-BREADTH A (#587): promote 7 smart_extractor alias-groups to
    // first-class BondTypes. Appended after Implements for rkyv on-disk
    // discriminant stability; v5 full-u8 codec only (v4 5-bit SST cannot hold >=32).
    DefinedAs,
    WorksBy,
    ComposedOf,
    PurposeOf,
    DiffersFrom,
    LeadsTo,
    ExampleOf,
    // OP-SOURCEPULL Batch-C: append-only (full-u8 v5 codec). Never reorder.
    HasCurrency,
    AuthoredBy,
}

impl BondType {
    /// Returns the base weight multiplier for this bond type.
    ///
    /// These multipliers reflect the semantic strength of each
    /// relationship type in the knowledge graph.
    pub fn base_weight(&self) -> f32 {
        match self {
            BondType::IsA => 1.5,
            BondType::DefinedAs => 1.5,
            BondType::WorksBy => 1.0,
            BondType::ComposedOf => 1.1,
            BondType::PurposeOf => 1.0,
            BondType::DiffersFrom => 0.7,
            BondType::LeadsTo => 1.3,
            BondType::ExampleOf => 1.0,
            BondType::HasProperty => 1.2,
            BondType::Causes => 1.3,
            BondType::PartOf => 1.1,
            BondType::UsedFor => 1.0,
            BondType::RelatedTo => 0.8,
            BondType::ContrastedWith => 0.7,
            BondType::DerivedFrom => 1.0,
            BondType::Requires => 1.1,
            BondType::Produces => 1.2,
            BondType::Opposes => 0.6,
            BondType::BornIn => 1.5,
            BondType::BornOn => 1.5,
            BondType::DiedIn => 1.5,
            BondType::DiedOn => 1.5,
            BondType::DiscoveredBy => 1.5,
            BondType::FoundedBy => 1.5,
            BondType::FoundedIn => 1.1,
            BondType::HasAltitude => 1.2,
            BondType::HasAspect => 1.2,
            BondType::HasCapital => 1.2,
            BondType::HasOfficialLanguage => 1.2,
            BondType::HasPopulation => 1.2,
            BondType::KnownFor => 1.2,
            BondType::LocatedIn => 1.1,
            BondType::MeasuredIn => 1.2,
            BondType::NamedAfter => 1.5,
            BondType::WrittenBy => 1.5,
            BondType::HasArea => 1.2,
            BondType::HasMethod => 1.2,
            BondType::HasAttribute => 1.2,
            BondType::HasParameter => 1.2,
            BondType::Returns => 1.2,
            BondType::Inherits => 1.5,
            BondType::Calls => 1.0,
            BondType::HasType => 1.2,
            BondType::Implements => 1.0,
            BondType::HasCurrency => 1.2,
            BondType::AuthoredBy => 1.5,
        }
    }

    /// All bond types as a slice, for iteration.
    pub fn all() -> &'static [BondType] {
        &[
            BondType::IsA,
            BondType::HasProperty,
            BondType::Causes,
            BondType::PartOf,
            BondType::UsedFor,
            BondType::RelatedTo,
            BondType::ContrastedWith,
            BondType::DerivedFrom,
            BondType::Requires,
            BondType::Produces,
            BondType::Opposes,
            BondType::BornIn,
            BondType::BornOn,
            BondType::DiedIn,
            BondType::DiedOn,
            BondType::DiscoveredBy,
            BondType::FoundedBy,
            BondType::FoundedIn,
            BondType::HasAltitude,
            BondType::HasAspect,
            BondType::HasCapital,
            BondType::HasOfficialLanguage,
            BondType::HasPopulation,
            BondType::KnownFor,
            BondType::LocatedIn,
            BondType::MeasuredIn,
            BondType::NamedAfter,
            BondType::WrittenBy,
            BondType::HasArea,
            BondType::HasMethod,
            BondType::HasAttribute,
            BondType::HasParameter,
            BondType::Returns,
            BondType::Inherits,
            BondType::Calls,
            BondType::HasType,
            BondType::Implements,
            BondType::DefinedAs,
            BondType::WorksBy,
            BondType::ComposedOf,
            BondType::PurposeOf,
            BondType::DiffersFrom,
            BondType::LeadsTo,
            BondType::ExampleOf,
            BondType::HasCurrency,
            BondType::AuthoredBy,
        ]
    }
}

/// A temporal bond connecting two nodes in the knowledge graph.
///
/// Unlike Lu v2's single-weight bonds, Lu 4D bonds carry an append-only
/// history of keyframes. The effective weight at any time `t` is computed
/// by interpolation and exponential decay.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
#[archive(check_bytes)]
pub struct TemporalBond {
    /// Source node.
    pub from: Seed,
    /// Target node.
    pub to: Seed,
    /// Semantic type of this bond.
    pub bond_type: BondType,
    /// Append-only temporal weight history. Sorted by timestamp.
    pub keyframes: Vec<TemporalKeyframe>,
    /// Half-life for exponential decay (milliseconds).
    pub half_life_ms: f64,
}

impl TemporalBond {
    /// View-style accessor: source seed. Mirrors `BondView::from`.
    pub fn from(&self) -> Seed { self.from }
    /// View-style accessor: target seed.
    pub fn to(&self) -> Seed { self.to }
    /// View-style accessor: semantic bond type.
    pub fn bond_type(&self) -> crate::bond::BondType { self.bond_type }
    /// View-style accessor: append-only keyframe history slice.
    pub fn keyframes(&self) -> &[TemporalKeyframe] { self.keyframes.as_slice() }
    /// View-style accessor: half-life for exponential decay.
    pub fn half_life_ms(&self) -> f64 { self.half_life_ms }

    /// Create a new temporal bond with an initial baseline keyframe.
    pub fn new(from: Seed, to: Seed, bond_type: BondType, initial_weight: f32, timestamp: i64) -> Self {
        TemporalBond {
            from,
            to,
            bond_type,
            keyframes: vec![TemporalKeyframe {
                timestamp,
                weight: initial_weight,
                source: KeyframeSource::Baseline,
            }],
            half_life_ms: DEFAULT_HALF_LIFE_MS,
        }
    }

    /// Append a new keyframe. Keyframes are never mutated, only added.
    ///
    /// The keyframe is inserted in sorted order by timestamp.
    pub fn add_keyframe(&mut self, timestamp: i64, weight: f32, source: KeyframeSource) {
        let kf = TemporalKeyframe {
            timestamp,
            weight,
            source,
        };
        // Insert in sorted position
        let pos = self
            .keyframes
            .binary_search_by_key(&timestamp, |k| k.timestamp)
            .unwrap_or_else(|pos| pos);
        self.keyframes.insert(pos, kf);
    }

    /// Compute the historical weight at `t`. Pre-scar arithmetic.
    /// Internal — public callers must go through `weight_at` or
    /// `weight_in_scope`, which thread the scar log.
    fn historical_weight_at(&self, t: i64) -> f32 {
        if self.keyframes.is_empty() {
            return 0.0;
        }

        // Variant- and Agent-source keyframes are scope-isolated: they
        // live in the bond's history but only show through their
        // respective TemporalScope::Variant / TemporalScope::Agent
        // queries (handled in `weight_in_scope`). Filter them out for
        // every other read.
        let visible_idx: Vec<usize> = self
            .keyframes
            .iter()
            .enumerate()
            .filter(|(_, kf)| !matches!(&kf.source,
                KeyframeSource::Variant { .. } | KeyframeSource::Agent { .. }))
            .map(|(i, _)| i)
            .collect();
        if visible_idx.is_empty() {
            return 0.0;
        }

        // For Current scope (i64::MAX), just return latest visible weight.
        if t == i64::MAX {
            let last = visible_idx[visible_idx.len() - 1];
            return self.keyframes[last].weight * self.bond_type.base_weight();
        }

        // Find the most recent visible keyframe at or before t. Linear
        // scan over the filtered index list — bond histories stay small
        // in practice, and bypassing binary_search keeps the visible-set
        // logic local to this method.
        let mut idx_opt: Option<usize> = None;
        for &i in &visible_idx {
            if self.keyframes[i].timestamp > t {
                break;
            }
            idx_opt = Some(i);
        }
        let idx = match idx_opt {
            Some(i) => i,
            None => return 0.0, // t is before all visible keyframes
        };

        let keyframe = &self.keyframes[idx];
        let elapsed_ms = (t - keyframe.timestamp) as f64;

        // Exponential decay: w * 2^(-elapsed / half_life)
        let decay = (-elapsed_ms / self.half_life_ms).exp2();
        let decayed_weight = keyframe.weight as f64 * decay;

        (decayed_weight as f32) * self.bond_type.base_weight()
    }

    /// Internal: historical weight under a scope, ignoring scars.
    /// Public callers see this only through `weight_in_scope`, which
    /// adds the scar overlay.
    fn historical_weight_in_scope(&self, scope: &TemporalScope) -> f32 {
        match scope {
            TemporalScope::Session { id, .. } => {
                // For session scope, only consider keyframes from this session or baseline
                let session_keyframes: Vec<&TemporalKeyframe> = self
                    .keyframes
                    .iter()
                    .filter(|kf| match &kf.source {
                        KeyframeSource::Baseline => true,
                        KeyframeSource::Session(sid) => sid == id,
                        _ => false,
                    })
                    .collect();

                if session_keyframes.is_empty() {
                    return self.historical_weight_at(scope.timestamp());
                }

                // Use the latest qualifying keyframe
                let latest = session_keyframes[session_keyframes.len() - 1];
                latest.weight * self.bond_type.base_weight()
            }
            TemporalScope::Variant { id, parent } => {
                // Prefer keyframes tagged with this variant's id; if none
                // exist, fall through to whatever the parent scope sees.
                let variant_keyframes: Vec<&TemporalKeyframe> = self
                    .keyframes
                    .iter()
                    .filter(|kf| matches!(&kf.source,
                        KeyframeSource::Variant { id: vid, .. } if vid == id))
                    .collect();
                if let Some(latest) = variant_keyframes.last() {
                    latest.weight * self.bond_type.base_weight()
                } else {
                    self.historical_weight_in_scope(parent)
                }
            }
            TemporalScope::Agent { role, multi_think_id, parent } => {
                // Prefer Agent-tagged keyframes from this multi_think /
                // role; otherwise fall through to the parent scope. The
                // role match keeps Recall / Synthesis / Critic reads
                // separated even within the same multi_think_id.
                let role_u8 = role.as_u8();
                let agent_keyframes: Vec<&TemporalKeyframe> = self
                    .keyframes
                    .iter()
                    .filter(|kf| matches!(&kf.source,
                        KeyframeSource::Agent { role: r, multi_think_id: m, .. }
                            if *r == role_u8 && m == multi_think_id))
                    .collect();
                if let Some(latest) = agent_keyframes.last() {
                    latest.weight * self.bond_type.base_weight()
                } else {
                    self.historical_weight_in_scope(parent)
                }
            }
            _ => self.historical_weight_at(scope.timestamp()),
        }
    }

    /// Apply the scar overlay to a historical weight. The forward-search
    /// anchor is `scar::now_ms()` — wall-clock NOW, NOT `t`. This is
    /// the architectural invariant the whole port is built on.
    fn apply_scar_overlay(&self, historical: f32, scar_log: &ScarLog) -> ScarAwareReading {
        let active = scar_log.active_scars(self.from, self.to, scar::now_ms());
        let active_owned: Vec<crate::scar::Scar> = active.into_iter().cloned().collect();
        let effective = if active_owned.is_empty() { historical } else { 0.0 };
        ScarAwareReading {
            historical_weight: historical,
            effective_weight: effective,
            active_scars: active_owned,
        }
    }

    /// Scar-aware weight read at time `t`. Returns BOTH the bond's
    /// pre-scar historical weight (intellectually honest about what Lu
    /// used to believe) AND the effective weight (zeroed if any scar
    /// is currently active) AND the active scar set.
    ///
    /// The scar lookup is anchored to wall-clock NOW, never to `t`.
    /// A scar created at t=20 is visible to a query at t=10. There is
    /// no signature path to a bare unscar-aware weight; this is the
    /// type-enforced safety primitive.
    pub fn weight_at(&self, t: i64, scar_log: &ScarLog) -> ScarAwareReading {
        let historical = self.historical_weight_at(t);
        self.apply_scar_overlay(historical, scar_log)
    }

    /// Scar-aware weight read under a temporal scope. Same contract as
    /// `weight_at` — historical weight respects the scope's
    /// scope-isolated keyframes; the scar overlay is anchored to NOW.
    pub fn weight_in_scope(&self, scope: &TemporalScope, scar_log: &ScarLog) -> ScarAwareReading {
        let historical = self.historical_weight_in_scope(scope);
        self.apply_scar_overlay(historical, scar_log)
    }

    /// Return the number of distinct sessions that have strengthened this bond.
    pub fn distinct_session_count(&self) -> usize {
        let mut sessions = std::collections::HashSet::new();
        for kf in &self.keyframes {
            if let KeyframeSource::Session(id) = &kf.source {
                sessions.insert(*id);
            }
        }
        sessions.len()
    }

    /// Return the latest keyframe's weight (without decay or bond-type multiplier).
    pub fn latest_raw_weight(&self) -> f32 {
        self.keyframes
            .last()
            .map(|kf| kf.weight)
            .unwrap_or(0.0)
    }

    /// Prune keyframes older than `cutoff` whose decayed weight at `cutoff`
    /// falls below `threshold`. Returns the number of pruned keyframes.
    pub fn prune_before(&mut self, cutoff: i64, threshold: f32) -> usize {
        let before = self.keyframes.len();
        self.keyframes.retain(|kf| {
            if kf.timestamp >= cutoff {
                return true;
            }
            let elapsed_ms = (cutoff - kf.timestamp) as f64;
            let decay = (-elapsed_ms / self.half_life_ms).exp2();
            let decayed = kf.weight.abs() as f64 * decay;
            decayed as f32 >= threshold
        });
        before - self.keyframes.len()
    }
}

// ---------------------------------------------------------------------
// OP-BRAIN-V4 Phase 5 — confidence re-derivation from keyframe source.
// ---------------------------------------------------------------------
//
// Phase 4's flush worker writes every flushed bond at `Confidence::Verified`
// because `MemBond` does not carry a confidence band. Phase 5's read path
// re-derives the correct confidence band from the bond's keyframe-source
// history at read time. The deciding signal is the source on the LATEST
// (highest-timestamp) keyframe in the bond's history.
//
// The brief originally referenced `Confidence::Hypothesis` and
// `Confidence::SessionLocal` variants, but the actual `Confidence` enum
// (`src/node.rs`) carries `Terminal / Verified / Unverified / Inferred /
// Contested`. Per the spec's "adapt the mapping to whatever's there"
// instruction we map onto the existing variants:
//
// | KeyframeSource          | Mapped Confidence        |
// |-------------------------|--------------------------|
// | Baseline                | stored (preserve)        |
// | Hebbian                 | stored (preserve)        |
// | Manual                  | stored (preserve)        |
// | PromotedFromVariant     | stored (preserve)        |
// | Synthesis               | Inferred                 |
// | Hypothesis              | Inferred                 |
// | ConfirmedHypothesis     | Verified                 |
// | Session(_)              | Unverified               |
// | Variant { .. }          | Inferred                 |
// | Agent { .. }            | Inferred                 |
//
// `Inferred` is the natural mapping for synthesis / hypothesis / variant /
// agent sources because that variant's documented meaning is "Derived
// through inference or reasoning" — matches their content provenance
// exactly. `Unverified` is the natural mapping for single-session-local
// sources because that variant's documented meaning is "Single-source,
// not yet corroborated".
//
// The discipline preserved here is the asymmetry-aware ingest contract
// (`feedback_training_data_asymmetry`): SST-stored bonds are explicitly
// the "post-asymmetry, pre-curriculum-discriminator" tier, and Phase 5
// re-derives confidence so spread activation reads the right multiplier
// at the moment of recall, without baking it into the SST byte layout.
use crate::node::Confidence;

/// Phase 5 helper: derive the read-time confidence band from a bond's
/// keyframe source. Pure function — easy to test in isolation. Caller
/// passes the source from the LATEST (highest-timestamp) keyframe.
///
/// Returns `Some(c)` if the source's mapping is fixed, `None` if the
/// source is one of the "preserve stored" variants (`Baseline /
/// Hebbian / Manual / PromotedFromVariant`). When `None` is returned
/// the caller must keep whatever confidence the underlying store had.
pub fn confidence_from_keyframe_source(source: &KeyframeSource) -> Option<Confidence> {
    match source {
        // Preserve-stored variants — caller keeps the SST/heap value.
        KeyframeSource::Baseline => None,
        KeyframeSource::Hebbian => None,
        KeyframeSource::Manual => None,
        KeyframeSource::PromotedFromVariant { .. } => None,
        // Inference-derived sources: drop to Inferred.
        KeyframeSource::Synthesis { .. } => Some(Confidence::Inferred),
        KeyframeSource::Hypothesis { .. } => Some(Confidence::Inferred),
        KeyframeSource::Variant { .. } => Some(Confidence::Inferred),
        KeyframeSource::Agent { .. } => Some(Confidence::Inferred),
        // OP-WRITEBACK Phase 5: reasoned conclusion -> Inferred
        // (a derived belief is never asserted as an ingested fact).
        KeyframeSource::ConcludedFrom { .. } => Some(Confidence::Inferred),
        // Promoted hypothesis: caller already vetted it.
        KeyframeSource::ConfirmedHypothesis { .. } => Some(Confidence::Verified),
        // Session-local: single-source, awaits corroboration.
        KeyframeSource::Session(_) => Some(Confidence::Unverified),
    }
}

/// Phase 5 helper: pick the deciding keyframe for confidence
/// re-derivation. Returns the source on the highest-timestamp
/// keyframe, or `None` if `keyframes` is empty.
pub fn deciding_keyframe_source(keyframes: &[TemporalKeyframe]) -> Option<&KeyframeSource> {
    keyframes.iter().max_by_key(|kf| kf.timestamp).map(|kf| &kf.source)
}

/// Phase 5 convenience: apply the mapping using `stored` as the
/// fallback when the source's mapping is "preserve stored". When
/// `keyframes` is empty, the stored confidence is returned unchanged.
pub fn derive_confidence(
    keyframes: &[TemporalKeyframe],
    stored: Confidence,
) -> Confidence {
    match deciding_keyframe_source(keyframes) {
        Some(src) => confidence_from_keyframe_source(src).unwrap_or(stored),
        None => stored,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_seeds() -> (Seed, Seed) {
        (Seed::from_label("A"), Seed::from_label("B"))
    }

    #[test]
    fn test_weight_at_current() {
        let (a, b) = test_seeds();
        let bond = TemporalBond::new(a, b, BondType::RelatedTo, 0.5, 1000);
        let log = ScarLog::default();
        // Current (i64::MAX) should return latest weight * base_weight
        let w = bond.weight_at(i64::MAX, &log).effective_weight;
        assert!((w - 0.5 * 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_weight_at_exact_timestamp() {
        let (a, b) = test_seeds();
        let bond = TemporalBond::new(a, b, BondType::IsA, 1.0, 1000);
        let log = ScarLog::default();
        let w = bond.weight_at(1000, &log).effective_weight;
        assert!((w - 1.0 * 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_weight_decays_over_time() {
        let (a, b) = test_seeds();
        let bond = TemporalBond::new(a, b, BondType::UsedFor, 1.0, 0);
        let log = ScarLog::default();
        let w0 = bond.weight_at(0, &log).effective_weight;
        let w1 = bond.weight_at(DEFAULT_HALF_LIFE_MS as i64, &log).effective_weight;
        // After one half-life, weight should be halved
        assert!((w1 - w0 * 0.5).abs() < 0.01);
    }

    #[test]
    fn test_weight_before_all_keyframes() {
        let (a, b) = test_seeds();
        let bond = TemporalBond::new(a, b, BondType::Causes, 1.0, 1000);
        let log = ScarLog::default();
        assert_eq!(bond.weight_at(500, &log).effective_weight, 0.0);
    }

    #[test]
    fn test_add_keyframe_sorted() {
        let (a, b) = test_seeds();
        let mut bond = TemporalBond::new(a, b, BondType::Causes, 0.5, 100);
        bond.add_keyframe(300, 0.9, KeyframeSource::Session(1));
        bond.add_keyframe(200, 0.7, KeyframeSource::Hebbian);
        assert_eq!(bond.keyframes[0].timestamp, 100);
        assert_eq!(bond.keyframes[1].timestamp, 200);
        assert_eq!(bond.keyframes[2].timestamp, 300);
    }

    #[test]
    fn test_distinct_session_count() {
        let (a, b) = test_seeds();
        let mut bond = TemporalBond::new(a, b, BondType::RelatedTo, 0.5, 0);
        bond.add_keyframe(100, 0.6, KeyframeSource::Session(1));
        bond.add_keyframe(200, 0.7, KeyframeSource::Session(2));
        bond.add_keyframe(300, 0.8, KeyframeSource::Session(1));
        assert_eq!(bond.distinct_session_count(), 2);
    }

    #[test]
    fn test_prune_below_threshold() {
        let (a, b) = test_seeds();
        let mut bond = TemporalBond::new(a, b, BondType::Opposes, 0.01, 0);
        bond.add_keyframe(1_000_000_000, 1.0, KeyframeSource::Baseline);
        let pruned = bond.prune_before(1_000_000_000, 0.001);
        assert_eq!(pruned, 1); // The old tiny-weight keyframe decayed away
    }

    // -----------------------------------------------------------------
    // OP-BRAIN-V4 Phase 5 — confidence_from_keyframe_source tests.
    // -----------------------------------------------------------------

    fn kf(ts: i64, source: KeyframeSource) -> TemporalKeyframe {
        TemporalKeyframe { timestamp: ts, weight: 0.5, source }
    }

    #[test]
    fn confidence_mapping_baseline_preserves_stored() {
        assert_eq!(confidence_from_keyframe_source(&KeyframeSource::Baseline), None);
    }

    #[test]
    fn confidence_mapping_hebbian_preserves_stored() {
        assert_eq!(confidence_from_keyframe_source(&KeyframeSource::Hebbian), None);
    }

    #[test]
    fn confidence_mapping_manual_preserves_stored() {
        assert_eq!(confidence_from_keyframe_source(&KeyframeSource::Manual), None);
    }

    #[test]
    fn confidence_mapping_promoted_from_variant_preserves_stored() {
        let src = KeyframeSource::PromotedFromVariant { variant_id: 7, original_strategy: 0 };
        assert_eq!(confidence_from_keyframe_source(&src), None);
    }

    #[test]
    fn confidence_mapping_synthesis_inferred() {
        let (a, b) = test_seeds();
        let src = KeyframeSource::Synthesis { from_path: [a, b, a] };
        assert_eq!(confidence_from_keyframe_source(&src), Some(Confidence::Inferred));
    }

    #[test]
    fn confidence_mapping_hypothesis_inferred() {
        let (a, _) = test_seeds();
        let src = KeyframeSource::Hypothesis { goal: a, step: 1, session: None };
        assert_eq!(confidence_from_keyframe_source(&src), Some(Confidence::Inferred));
    }

    #[test]
    fn confidence_mapping_confirmed_hypothesis_verified() {
        let (a, _) = test_seeds();
        let src = KeyframeSource::ConfirmedHypothesis { goal: a, step: 2, session: Some(9) };
        assert_eq!(confidence_from_keyframe_source(&src), Some(Confidence::Verified));
    }

    #[test]
    fn confidence_mapping_session_unverified() {
        assert_eq!(
            confidence_from_keyframe_source(&KeyframeSource::Session(42)),
            Some(Confidence::Unverified),
        );
    }

    #[test]
    fn confidence_mapping_variant_inferred() {
        let (a, _) = test_seeds();
        let src = KeyframeSource::Variant { id: 1, goal: a, strategy: 0 };
        assert_eq!(confidence_from_keyframe_source(&src), Some(Confidence::Inferred));
    }

    #[test]
    fn confidence_mapping_agent_inferred() {
        let src = KeyframeSource::Agent { role: 0, multi_think_id: 1, step: 0 };
        assert_eq!(confidence_from_keyframe_source(&src), Some(Confidence::Inferred));
    }

    #[test]
    fn deciding_keyframe_picks_highest_timestamp() {
        let kfs = vec![
            kf(100, KeyframeSource::Baseline),
            kf(500, KeyframeSource::Session(7)),
            kf(250, KeyframeSource::Hebbian),
        ];
        let src = deciding_keyframe_source(&kfs).unwrap();
        assert!(matches!(src, KeyframeSource::Session(7)));
    }

    #[test]
    fn deciding_keyframe_empty_history_is_none() {
        let kfs: Vec<TemporalKeyframe> = Vec::new();
        assert!(deciding_keyframe_source(&kfs).is_none());
    }

    #[test]
    fn derive_confidence_preserves_stored_when_baseline_latest() {
        let kfs = vec![
            kf(100, KeyframeSource::Session(1)),
            kf(200, KeyframeSource::Baseline),
        ];
        assert_eq!(derive_confidence(&kfs, Confidence::Verified), Confidence::Verified);
    }

    #[test]
    fn derive_confidence_overrides_when_session_latest() {
        let kfs = vec![
            kf(100, KeyframeSource::Baseline),
            kf(200, KeyframeSource::Session(1)),
        ];
        assert_eq!(derive_confidence(&kfs, Confidence::Verified), Confidence::Unverified);
    }

    #[test]
    fn derive_confidence_overrides_when_synthesis_latest() {
        let (a, b) = test_seeds();
        let kfs = vec![
            kf(100, KeyframeSource::Baseline),
            kf(300, KeyframeSource::Synthesis { from_path: [a, b, a] }),
        ];
        assert_eq!(derive_confidence(&kfs, Confidence::Verified), Confidence::Inferred);
    }

    #[test]
    fn derive_confidence_empty_history_returns_stored() {
        let kfs: Vec<TemporalKeyframe> = Vec::new();
        assert_eq!(derive_confidence(&kfs, Confidence::Unverified), Confidence::Unverified);
    }

}
