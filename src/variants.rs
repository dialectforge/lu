//! Variant Generation Engine — fifth of the seven Lu 4.0 capability ports.
//!
//! For a goal seed (from the Thinking Layer or a curiosity signal),
//! spawn up to 5 alternative hypothesis CLUSTERS — each one a "what if"
//! perspective on the same input. Run each variant in its own
//! `TemporalScope::Variant{id}`, score by coherence x novelty x
//! parsimony, and promote the winner's keyframes to baseline. Losers'
//! keyframes age out via the existing temporal half-life — no cloning,
//! no copy-on-write, the temporal graph IS the variant arena.
//!
//! Architecturally:
//!
//! 1. **Anchoring (deterministic).** Each `PerturbationStrategy` has a
//!    rule for picking an anchor seed from the goal's neighborhood
//!    (highest-degree neighbor, deepest path, weakest-bonded contrarian,
//!    most-recent bridge, baseline = goal itself). Sort orders are
//!    stable so the same brain state always yields the same anchors.
//! 2. **Variant scoping.** When a variant runs a spread, the scope is
//!    `TemporalScope::Variant{id, parent}` where `parent` carries the
//!    surrounding session/current state. `weight_in_scope` prefers
//!    variant-tagged keyframes; if none exist for a bond, it falls
//!    through to `parent`. `weight_at` (used by Baseline / Current /
//!    Session lookups) actively filters out `Variant`-source keyframes,
//!    so non-variant readers stay clean.
//! 3. **Scoring.** After each variant's spread we have an activated
//!    set. Coherence sums in-set bond weights (in variant scope);
//!    novelty is Jaccard distance from the Baseline strategy's
//!    activation set; parsimony rewards small clusters. Composite =
//!    `coherence * 0.6 + novelty * 0.3 + parsimony * 0.1`.
//! 4. **Promotion.** The arena's winner gets its variant-tagged
//!    keyframes re-tagged as `PromotedFromVariant` (visible to baseline
//!    scope) at `now + 1`. Other variants are not actively cleaned up;
//!    their keyframes decay naturally.
//!
//! Bounds: `variant_node_cap` caps activations per variant;
//! `variants_per_goal` caps the strategy count.

use crate::bond::{BondType, KeyframeSource, TemporalBond, DEFAULT_HALF_LIFE_MS};
use crate::brain::Brain;
use crate::network::AssociativeNetwork;
use crate::scope::TemporalScope;
use crate::scar::ScarLog;
use crate::seed::Seed;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// One of five perturbation rules for picking the variant anchor seed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PerturbationStrategy {
    /// Anchor at the goal seed itself. The control variant. Carries
    /// `novelty == 0` by definition; useful as a baseline against which
    /// the others are scored.
    Baseline,
    /// Anchor at the goal's highest-degree non-stopword neighbor.
    DomainShift,
    /// Walk the goal's strongest-bond chain for `max_hops_for_deep`
    /// hops; anchor at the chain's end.
    Deep,
    /// Anchor at the goal's weakest non-zero-weight neighbor — the
    /// "explore the opposite" bet.
    Contrarian,
    /// Anchor at the other endpoint of a recent bridge event involving
    /// the goal seed; cross-domain probe.
    Interdisciplinary,
}

impl PerturbationStrategy {
    /// All five strategies in a fixed, deterministic order.
    pub fn all() -> &'static [PerturbationStrategy] {
        &[
            PerturbationStrategy::Baseline,
            PerturbationStrategy::DomainShift,
            PerturbationStrategy::Deep,
            PerturbationStrategy::Contrarian,
            PerturbationStrategy::Interdisciplinary,
        ]
    }

    /// Compact byte tag (used in `KeyframeSource::Variant.strategy`).
    pub fn as_u8(self) -> u8 {
        match self {
            PerturbationStrategy::Baseline => 0,
            PerturbationStrategy::DomainShift => 1,
            PerturbationStrategy::Deep => 2,
            PerturbationStrategy::Contrarian => 3,
            PerturbationStrategy::Interdisciplinary => 4,
        }
    }

    pub fn from_u8(b: u8) -> Option<Self> {
        match b {
            0 => Some(PerturbationStrategy::Baseline),
            1 => Some(PerturbationStrategy::DomainShift),
            2 => Some(PerturbationStrategy::Deep),
            3 => Some(PerturbationStrategy::Contrarian),
            4 => Some(PerturbationStrategy::Interdisciplinary),
            _ => None,
        }
    }
}

/// One spawned variant's anchor + identity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariantSpec {
    /// Brain-issued u64 used both as the variant id and as
    /// `KeyframeSource::Variant.id` and `Scope::Variant.id`.
    pub id: u64,
    /// Which perturbation rule chose this anchor.
    pub strategy: PerturbationStrategy,
    /// Anchor seed for variant spread.
    pub anchor_seed: Seed,
    /// The goal seed this variant is reasoning off of.
    pub source_goal: Seed,
    /// Wall-clock millis when the spec was created.
    pub created_ms: i64,
}

/// Per-variant arena score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariantScore {
    pub variant_id: u64,
    /// Sum of in-variant bond weights between activated nodes,
    /// normalized to [0, 1].
    pub coherence: f32,
    /// Jaccard distance between this variant's activated set and the
    /// Baseline strategy's activated set. Baseline itself is 0.
    pub novelty: f32,
    /// `1 - (activated_count / max_activated_count_in_arena)`. Smaller
    /// clusters get more parsimony credit.
    pub parsimony: f32,
    /// `coherence * w_c + novelty * w_n + parsimony * w_p` per
    /// `VariantConfig` weights.
    pub composite: f32,
}

/// One full arena run: specs + matched scores + winner (if any).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariantArena {
    pub variants: Vec<VariantSpec>,
    pub scores: Vec<VariantScore>,
    /// Variant id of the highest-composite spec, IFF its composite is
    /// at least `VariantConfig::min_winner_composite`. `None` means no
    /// promotion this round.
    pub winner_id: Option<u64>,
}

/// All knobs for one arena run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariantConfig {
    /// Cap on strategies per goal. Default 5 — one per `PerturbationStrategy`.
    pub variants_per_goal: usize,
    /// Hops for the Deep-strategy chain walk.
    pub max_hops_for_deep: usize,
    /// Composite weight: coherence.
    pub coherence_weight: f32,
    /// Composite weight: novelty.
    pub novelty_weight: f32,
    /// Composite weight: parsimony.
    pub parsimony_weight: f32,
    /// Below this composite, no winner is chosen.
    pub min_winner_composite: f32,
    /// Cap on activated nodes per variant spread.
    pub variant_node_cap: usize,
    /// Half-life for abandoned variant keyframes (ms). Currently
    /// informational — pruning lives in the consolidation engine and
    /// reads `DEFAULT_HALF_LIFE_MS` from `bond.rs`. Surfaced here so
    /// future tuning knobs have a home.
    pub variant_decay_halflife_ms: u64,
    /// Initial weight for a Variant-tagged keyframe.
    pub variant_keyframe_weight: f32,
}

impl Default for VariantConfig {
    fn default() -> Self {
        VariantConfig {
            variants_per_goal: 5,
            max_hops_for_deep: 3,
            coherence_weight: 0.6,
            novelty_weight: 0.3,
            parsimony_weight: 0.1,
            min_winner_composite: 0.4,
            variant_node_cap: 200,
            variant_decay_halflife_ms: 6 * 60 * 60 * 1000, // 6 hours
            variant_keyframe_weight: 0.25,
        }
    }
}

// =====================================================================
// Strategy implementations
// =====================================================================

/// Sort key that breaks ties by `Seed` byte order. All anchor pickers
/// use this so their output is reproducible.
fn seed_tiebreak(s: &Seed) -> [u8; 32] {
    *s.as_bytes()
}

/// Total degree (incoming + outgoing) of a seed in the network.
fn degree(net: &AssociativeNetwork, s: &Seed) -> usize {
    net.seam_total_degree(s)
}

/// Ordered (deduped) neighbor list of a seed: outgoing then incoming,
/// preserving first-seen order over the CANONICAL (seed-byte ascending)
/// seam edge order. Used by every strategy; deterministic in BOTH legacy
/// and cache mode (OP-BRAIN-MapRAM P5-b2: off resident adjacency).
fn neighbors_ordered(net: &AssociativeNetwork, s: &Seed) -> Vec<Seed> {
    let mut seen: HashSet<Seed> = HashSet::new();
    let mut out: Vec<Seed> = Vec::new();
    for e in net.seam_outgoing_edges(s) {
        if seen.insert(e.nbr) {
            out.push(e.nbr);
        }
    }
    for e in net.seam_incoming_edges(s) {
        if seen.insert(e.nbr) {
            out.push(e.nbr);
        }
    }
    out
}

/// Effective bond weight (current scope) between `a -> b` if such a
/// directed bond exists, else 0.
fn bond_weight_directed(net: &AssociativeNetwork, a: &Seed, b: &Seed, scar_log: &ScarLog) -> f32 {
    net.seam_directed_bond(a, b)
        .map(|bond| bond.view().weight_in_scope(&TemporalScope::Current, scar_log).effective_weight)
        .unwrap_or(0.0)
}

/// Highest-degree non-stopword neighbor of `goal_seed`. Tiebreak by
/// seed bytes ASC for determinism.
fn anchor_domain_shift(net: &AssociativeNetwork, goal_seed: Seed) -> Option<Seed> {
    let mut candidates: Vec<(Seed, usize)> = neighbors_ordered(net, &goal_seed)
        .into_iter()
        .filter_map(|s| {
            let n = net.get_node(&s)?;
            if crate::network::is_hub_stopword(n.content()) {
                return None;
            }
            Some((s, degree(net, &s)))
        })
        .collect();
    // Sort: degree DESC, seed bytes ASC.
    candidates.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| seed_tiebreak(&a.0).cmp(&seed_tiebreak(&b.0))));
    candidates.first().map(|(s, _)| *s)
}

/// End of the strongest-weight chain starting at `goal_seed`. At each
/// hop, pick the neighbor with the highest current-scope weight; tie-
/// break by seed bytes ASC. Stop on a cycle (already visited) or when
/// no outgoing neighbor exists. Up to `max_hops` hops.
fn anchor_deep(net: &AssociativeNetwork, goal_seed: Seed, max_hops: usize, scar_log: &ScarLog) -> Option<Seed> {
    if max_hops == 0 {
        return None;
    }
    let mut visited: HashSet<Seed> = HashSet::new();
    visited.insert(goal_seed);
    let mut current = goal_seed;
    let mut hops_taken = 0usize;
    while hops_taken < max_hops {
        let neighbors = neighbors_ordered(net, &current);
        let mut scored: Vec<(Seed, f32)> = neighbors
            .into_iter()
            .filter(|n| !visited.contains(n))
            .map(|n| {
                let w = bond_weight_directed(net, &current, &n, scar_log)
                    .max(bond_weight_directed(net, &n, &current, scar_log));
                (n, w)
            })
            .filter(|(_, w)| *w > 0.0)
            .collect();
        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| seed_tiebreak(&a.0).cmp(&seed_tiebreak(&b.0)))
        });
        match scored.first() {
            Some((next, _)) => {
                visited.insert(*next);
                current = *next;
                hops_taken += 1;
            }
            None => break,
        }
    }
    if hops_taken == 0 {
        None
    } else {
        Some(current)
    }
}

/// Weakest non-zero bond's other endpoint. Tiebreak by seed bytes ASC.
/// Considers both directions.
fn anchor_contrarian(net: &AssociativeNetwork, goal_seed: Seed, scar_log: &ScarLog) -> Option<Seed> {
    let mut candidates: Vec<(Seed, f32)> = Vec::new();
    for n in neighbors_ordered(net, &goal_seed) {
        let w = bond_weight_directed(net, &goal_seed, &n, scar_log)
            .max(bond_weight_directed(net, &n, &goal_seed, scar_log));
        if w > 0.0 {
            candidates.push((n, w));
        }
    }
    candidates.sort_by(|a, b| {
        a.1.partial_cmp(&b.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| seed_tiebreak(&a.0).cmp(&seed_tiebreak(&b.0)))
    });
    candidates.first().map(|(s, _)| *s)
}

/// First recent-bridge event involving `goal_seed`, in insertion order;
/// pick the OTHER endpoint as the anchor.
fn anchor_interdisciplinary(brain: &Brain, goal_seed: Seed) -> Option<Seed> {
    for ev in brain.recent_bridges.iter() {
        if ev.low == goal_seed {
            return Some(ev.high);
        }
        if ev.high == goal_seed {
            return Some(ev.low);
        }
    }
    None
}

/// Resolve a strategy's anchor seed. Returns `None` if no anchor can
/// be derived (e.g., goal has no neighbors for DomainShift).
pub fn resolve_anchor(
    brain: &Brain,
    goal_seed: Seed,
    strategy: PerturbationStrategy,
    cfg: &VariantConfig,
    scar_log: &ScarLog
) -> Option<Seed> {
    match strategy {
        PerturbationStrategy::Baseline => Some(goal_seed),
        PerturbationStrategy::DomainShift => anchor_domain_shift(&brain.network, goal_seed),
        PerturbationStrategy::Deep => anchor_deep(&brain.network, goal_seed, cfg.max_hops_for_deep, scar_log),
        PerturbationStrategy::Contrarian => anchor_contrarian(&brain.network, goal_seed, scar_log),
        PerturbationStrategy::Interdisciplinary => anchor_interdisciplinary(brain, goal_seed),
    }
}

/// Spawn a variant for a single strategy: derive the anchor, write a
/// Variant-tagged keyframe on the (goal -> anchor) bond (creating the
/// bond with a Variant-only constructor if it didn't exist), and
/// return the spec. `None` if the strategy has no candidate anchor.
pub fn generate_variants_for_goal(
    brain: &mut Brain,
    goal_seed: Seed,
    strategy: PerturbationStrategy,
    variant_id: u64,
    now: i64,
    cfg: &VariantConfig,
    scar_log: &crate::scar::ScarLog,
) -> Option<VariantSpec> {
    let anchor = resolve_anchor(brain, goal_seed, strategy, cfg, scar_log)?;
    // Skip the trivial self-loop case for non-Baseline strategies.
    if anchor == goal_seed && strategy != PerturbationStrategy::Baseline {
        return None;
    }

    // Tag the (goal -> anchor) edge with a Variant keyframe. Creates
    // the bond if it didn't already exist (with a Variant-only first
    // keyframe so baseline scope sees weight 0 until promotion).
    if anchor != goal_seed {
        if brain.network.get_bond(&goal_seed, &anchor).is_some() {
            if let Some(mut b) = brain.network.get_bond_mut(&goal_seed, &anchor) {
                b.add_keyframe(
                    now,
                    cfg.variant_keyframe_weight,
                    KeyframeSource::Variant {
                        id: variant_id,
                        goal: goal_seed,
                        strategy: strategy.as_u8(),
                    },
                );
            }
        } else {
            // New bond, variant-only keyframe.
            let bond = TemporalBond {
                from: goal_seed,
                to: anchor,
                bond_type: BondType::RelatedTo,
                keyframes: vec![crate::bond::TemporalKeyframe {
                    timestamp: now,
                    weight: cfg.variant_keyframe_weight,
                    source: KeyframeSource::Variant {
                        id: variant_id,
                        goal: goal_seed,
                        strategy: strategy.as_u8(),
                    },
                }],
                half_life_ms: DEFAULT_HALF_LIFE_MS,
            };
            let _ = brain.network.add_bond(bond);
        }
    }

    Some(VariantSpec {
        id: variant_id,
        strategy,
        anchor_seed: anchor,
        source_goal: goal_seed,
        created_ms: now,
    })
}

// =====================================================================
// Arena scoring
// =====================================================================

/// Run the spread for one variant and return its activated seed set
/// (capped at `variant_node_cap`).
fn variant_activated_set(
    brain: &Brain,
    spec: &VariantSpec,
    parent: &TemporalScope,
    cfg: &VariantConfig,
    scar_log: &ScarLog
) -> Vec<Seed> {
    let recall = brain.network.spread_activation_in_variant_scope(
        spec.id,
        parent.clone(),
        &[(spec.anchor_seed, 1.0)],
        3, // depth: same shape as Brain::ask; not knob-exposed yet.
        cfg.variant_node_cap,
        scar_log,
    );
    recall.activated.into_iter().map(|a| a.seed).collect()
}

/// Sum of in-variant-scope bond weights for every directed edge whose
/// endpoints are both inside `set`. Used as the unnormalized coherence
/// numerator.
fn coherence_raw(net: &AssociativeNetwork, scope: &TemporalScope, set: &[Seed], scar_log: &ScarLog) -> f32 {
    if set.len() < 2 {
        return 0.0;
    }
    let s: HashSet<Seed> = set.iter().copied().collect();
    let mut total = 0.0f32;
    for (&(from, to), bond) in net.bonds().iter() {
        if s.contains(&from) && s.contains(&to) {
            total += bond.weight_in_scope(scope, scar_log).effective_weight;
        }
    }
    total
}

/// Jaccard distance between two seed sets (1 - intersection/union).
/// Returns 0 for two empty sets (no information).
fn jaccard_distance(a: &[Seed], b: &[Seed]) -> f32 {
    let sa: HashSet<Seed> = a.iter().copied().collect();
    let sb: HashSet<Seed> = b.iter().copied().collect();
    let inter = sa.intersection(&sb).count() as f32;
    let union = sa.union(&sb).count() as f32;
    if union == 0.0 {
        0.0
    } else {
        1.0 - (inter / union)
    }
}

/// Score the arena and pick a winner. Pure function over the brain
/// state at call time — no mutation.
pub fn score_arena(
    brain: &Brain,
    _goal_seed: Seed,
    specs: &[VariantSpec],
    parent: &TemporalScope,
    cfg: &VariantConfig,
    scar_log: &crate::scar::ScarLog,
) -> VariantArena {
    if specs.is_empty() {
        return VariantArena {
            variants: Vec::new(),
            scores: Vec::new(),
            winner_id: None,
        };
    }

    // Run each variant's spread once. Hold onto the activated sets so
    // novelty can compare against the Baseline-strategy variant.
    let activated_sets: Vec<Vec<Seed>> = specs
        .iter()
        .map(|s| variant_activated_set(brain, s, parent, cfg, scar_log))
        .collect();

    // Locate the Baseline variant's activated set, if any. Used as the
    // "control" for novelty Jaccard; if absent, novelty falls back to
    // Jaccard distance from the empty set (i.e. 1.0 minus 0 / |v| = 1.0).
    let baseline_set: Vec<Seed> = specs
        .iter()
        .zip(activated_sets.iter())
        .find(|(s, _)| s.strategy == PerturbationStrategy::Baseline)
        .map(|(_, set)| set.clone())
        .unwrap_or_default();

    let max_count = activated_sets
        .iter()
        .map(|v| v.len())
        .max()
        .unwrap_or(1)
        .max(1);

    let mut scores: Vec<VariantScore> = Vec::new();
    for (spec, set) in specs.iter().zip(activated_sets.iter()) {
        // Coherence: in-variant bond weight sum, normalized by
        // (n*(n-1)) which is the maximum possible directed bond count.
        let n = set.len();
        let denom = if n >= 2 { (n * (n - 1)) as f32 } else { 1.0 };
        let scope = TemporalScope::Variant {
            id: spec.id,
            parent: Box::new(parent.clone()),
        };
        let coherence_raw = coherence_raw(&brain.network, &scope, set, scar_log);
        let coherence = (coherence_raw / denom).clamp(0.0, 1.0);

        // Novelty: Jaccard distance from Baseline set. Baseline itself
        // is 0 by definition.
        let novelty = if spec.strategy == PerturbationStrategy::Baseline {
            0.0
        } else {
            jaccard_distance(set, &baseline_set)
        };

        // Parsimony: smaller activations score higher.
        let parsimony = (1.0 - (n as f32 / max_count as f32)).clamp(0.0, 1.0);

        let composite = coherence * cfg.coherence_weight
            + novelty * cfg.novelty_weight
            + parsimony * cfg.parsimony_weight;

        scores.push(VariantScore {
            variant_id: spec.id,
            coherence,
            novelty,
            parsimony,
            composite,
        });
    }

    // Pick winner: highest composite >= min_winner_composite. Tiebreak
    // by id ASC (older variant wins) to keep the choice deterministic.
    let winner_id = scores
        .iter()
        .filter(|s| s.composite >= cfg.min_winner_composite)
        .max_by(|a, b| {
            a.composite
                .partial_cmp(&b.composite)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| b.variant_id.cmp(&a.variant_id))
        })
        .map(|s| s.variant_id);

    VariantArena {
        variants: specs.to_vec(),
        scores,
        winner_id,
    }
}


// =====================================================================
// OP-ROUTE-3b -- compact summary for NarrationResult.variants
// =====================================================================

/// Compact summary of a `VariantArena` that landed on a narrate PASS-path
/// query. Designed for the JSON wire surface of `NarrationResult.variants`
/// where the full arena (5 specs + 5 scores + temporal-keyframe substrate)
/// is too heavy and most of its fields are internal-only.
///
/// `None` outside this struct (i.e. `NarrationResult.variants == None`)
/// means the router did not request variants, or the request was guarded
/// out (meta / math / missing goal seed / below activation threshold).
/// When the engine ran but elected no winner (composite stayed below
/// `min_winner_composite`), the summary is present with
/// `no_winner: true, winner_strategy: None`. This preserves the
/// "engine ran" signal per Section 15 resolved decision #8.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VariantArenaSummary {
    /// Which perturbation strategy produced the winner. `None` when no
    /// winner was elected.
    pub winner_strategy: Option<PerturbationStrategy>,
    /// Winner's composite score. 0.0 when no winner.
    pub winner_composite: f32,
    /// Number of activated nodes in the winner's scope (informational).
    /// NOTE: `VariantArena` does not retain per-spec activated-set sizes
    /// post-scoring, so this is currently always 0 in the
    /// `From<&VariantArena>` builder. A future port can populate it by
    /// adding `activated_count` to `VariantScore` or by re-spreading at
    /// summary-construction time. Section 15 resolved decision #7 dial.
    pub winner_node_count: usize,
    /// Total candidate specs spawned (1-5, capped by `variants_per_goal`).
    pub num_candidates: usize,
    /// `true` when the arena ran but elected no winner (composite below
    /// floor). Distinct from "variants did not run at all" (the outer
    /// `Option<VariantArenaSummary>` is `None` for that).
    pub no_winner: bool,
}

impl From<&VariantArena> for VariantArenaSummary {
    fn from(arena: &VariantArena) -> Self {
        let winner_score = arena
            .winner_id
            .and_then(|wid| arena.scores.iter().find(|s| s.variant_id == wid));
        let winner_strategy = arena
            .winner_id
            .and_then(|wid| arena.variants.iter().find(|v| v.id == wid))
            .map(|spec| spec.strategy);
        VariantArenaSummary {
            winner_strategy,
            winner_composite: winner_score.map(|s| s.composite).unwrap_or(0.0),
            winner_node_count: 0,
            num_candidates: arena.variants.len(),
            no_winner: arena.winner_id.is_none(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::Confidence;

    /// A small brain with two clusters: an alpha hub with 4 leaves and
    /// a beta hub with 3 leaves, plus a weak alpha->beta edge so the
    /// goal has a multi-strategy neighborhood. Returns the alpha seed
    /// (used as the goal in tests).
    fn small_brain() -> (Brain, Seed) {
        let mut brain = Brain::new();
        // Alpha cluster: alpha -> a1..a4 (strong), a1 -> a2 (medium).
        brain.teach("alpha", BondType::IsA,         "a1", 0.9, Confidence::Verified);
        brain.teach("alpha", BondType::HasProperty, "a2", 0.8, Confidence::Verified);
        brain.teach("alpha", BondType::Causes,      "a3", 0.7, Confidence::Verified);
        brain.teach("alpha", BondType::RelatedTo,   "a4", 0.4, Confidence::Verified);
        brain.teach("a1",    BondType::RelatedTo,   "a2", 0.6, Confidence::Verified);
        // Beta cluster: beta -> b1..b3 (strong).
        brain.teach("beta",  BondType::IsA,         "b1", 0.9, Confidence::Verified);
        brain.teach("beta",  BondType::HasProperty, "b2", 0.8, Confidence::Verified);
        brain.teach("beta",  BondType::Causes,      "b3", 0.7, Confidence::Verified);
        // Weak cross-cluster edge for the bridge / interdisciplinary path.
        brain.teach("alpha", BondType::RelatedTo,   "beta", 0.1, Confidence::Verified);
        let alpha = Seed::from_label("alpha");
        (brain, alpha)
    }

    #[test]
    fn test_variant_spawner_creates_5_per_strategy() {
        // With Interdisciplinary requiring a populated recent_bridges
        // ring, default-config call may emit 4 -- but with all 5
        // strategies active in the small brain (alpha has multiple
        // neighbors and we can manually seed a bridge), we expect 5.
        let (mut brain, goal) = small_brain();
        // Seed a synthetic bridge so Interdisciplinary fires.
        let beta = Seed::from_label("beta");
        brain.recent_bridges.push_back(crate::bridge::BridgeEvent {
            low:  if goal.as_bytes() < beta.as_bytes() { goal } else { beta },
            high: if goal.as_bytes() < beta.as_bytes() { beta } else { goal },
            low_content: "alpha".into(),
            high_content: "beta".into(),
            bond_weight: 0.1,
            jaccard: 0.0,
            scope: TemporalScope::Current,
            timestamp_ms: 0,
        });

        let arena = brain.generate_variants(goal);
        assert_eq!(arena.variants.len(), 5, "expected 5 variants, got: {:?}", arena.variants.iter().map(|v| v.strategy).collect::<Vec<_>>());
        let strategies: HashSet<PerturbationStrategy> =
            arena.variants.iter().map(|v| v.strategy).collect();
        assert!(strategies.contains(&PerturbationStrategy::Baseline));
        assert!(strategies.contains(&PerturbationStrategy::DomainShift));
        assert!(strategies.contains(&PerturbationStrategy::Deep));
        assert!(strategies.contains(&PerturbationStrategy::Contrarian));
        assert!(strategies.contains(&PerturbationStrategy::Interdisciplinary));
    }

    #[test]
    fn test_each_strategy_produces_distinct_anchor() {
        // In small_brain alpha has multiple neighbors at different
        // weights, so the four non-Baseline strategies should disagree
        // on the anchor. Baseline anchors at goal itself.
        let (mut brain, goal) = small_brain();
        let beta = Seed::from_label("beta");
        brain.recent_bridges.push_back(crate::bridge::BridgeEvent {
            low:  if goal.as_bytes() < beta.as_bytes() { goal } else { beta },
            high: if goal.as_bytes() < beta.as_bytes() { beta } else { goal },
            low_content: "alpha".into(),
            high_content: "beta".into(),
            bond_weight: 0.1,
            jaccard: 0.0,
            scope: TemporalScope::Current,
            timestamp_ms: 0,
        });
        let arena = brain.generate_variants(goal);
        let baseline = arena.variants.iter()
            .find(|v| v.strategy == PerturbationStrategy::Baseline).unwrap();
        assert_eq!(baseline.anchor_seed, goal);

        // The non-Baseline strategies should not all collapse to the
        // same anchor. Collect their anchors and require >=2 distinct.
        let mut anchors: Vec<Seed> = arena.variants.iter()
            .filter(|v| v.strategy != PerturbationStrategy::Baseline)
            .map(|v| v.anchor_seed).collect();
        anchors.sort_by_key(|s| seed_tiebreak(s));
        anchors.dedup();
        assert!(anchors.len() >= 2,
            "expected >=2 distinct non-Baseline anchors; got {}", anchors.len());
    }

    #[test]
    fn test_variant_scope_isolates_keyframes() {
        // A Variant-only keyframe written via generate_variants must
        // NOT be visible in Current scope. We verify by reading the
        // (goal -> anchor) bond's weight in both scopes.
        let (mut brain, goal) = small_brain();
        let arena = brain.generate_variants(goal);
        let domain = arena.variants.iter()
            .find(|v| v.strategy == PerturbationStrategy::DomainShift)
            .expect("DomainShift should produce a spec");
        let bond = brain.network.get_bond(&goal, &domain.anchor_seed);
        assert!(bond.is_some(), "variant should have created or touched the goal->anchor bond");
        let bond = bond.unwrap();
        let scar_log = ScarLog::default();
        let current = bond.weight_in_scope(&TemporalScope::Current, &scar_log).effective_weight;
        let in_variant = bond.weight_in_scope(&TemporalScope::Variant {
            id: domain.id,
            parent: Box::new(TemporalScope::Current),
        }, &scar_log).effective_weight;
        assert!(in_variant > 0.0, "variant scope should see the variant-tagged keyframe");
        // Current scope sees only baseline-eligible keyframes. If the
        // bond predated the variant (alpha->a1 etc.), `current` will be
        // the baseline weight; if the variant created the bond fresh,
        // `current` should be 0. Either way `current` must be strictly
        // less than `in_variant + epsilon` since the variant keyframe
        // is at variant_keyframe_weight and overrides anything older
        // only inside variant scope.
        assert!(
            (current - in_variant).abs() > 1e-4 || current == 0.0,
            "variant keyframe should not equal current-scope reading; current={}, variant={}",
            current, in_variant,
        );
    }

    #[test]
    fn test_arena_scoring_picks_high_coherence_winner() {
        // Force a clear winner by lowering min_winner_composite so
        // the Baseline strategy (which usually wins on coherence in
        // dense graphs) gets picked.
        let (mut brain, goal) = small_brain();
        let cfg = VariantConfig {
            min_winner_composite: 0.05, // very permissive
            ..VariantConfig::default()
        };
        let arena = brain.generate_variants_with_config(goal, &cfg);
        assert!(arena.winner_id.is_some(),
            "expected a winner with permissive min_winner_composite; arena scores: {:?}", arena.scores);
        // Winner's composite must be the max in the arena.
        let winner_composite = arena.scores.iter()
            .find(|s| Some(s.variant_id) == arena.winner_id)
            .unwrap()
            .composite;
        let max_composite = arena.scores.iter()
            .map(|s| s.composite)
            .fold(f32::MIN, f32::max);
        assert!((winner_composite - max_composite).abs() < 1e-6);
    }

    #[test]
    fn test_promotion_retags_winner_keyframes_to_baseline() {
        // After promote, the goal->anchor bond must carry a
        // PromotedFromVariant keyframe and weight_in_scope(Current)
        // must reflect it.
        let (mut brain, goal) = small_brain();
        let cfg = VariantConfig { min_winner_composite: 0.05, ..VariantConfig::default() };
        let arena = brain.generate_variants_with_config(goal, &cfg);
        let winner_id = arena.winner_id.expect("permissive cfg should yield a winner");
        let promoted = brain.promote_variant(&arena);
        assert!(promoted >= 1, "at least the goal->anchor bond should be promoted");

        // Find the winner spec, check the (goal -> anchor) bond carries
        // a PromotedFromVariant keyframe with that variant id.
        let winner_spec = arena.variants.iter().find(|v| v.id == winner_id).unwrap();
        let bond = brain.network.get_bond(&goal, &winner_spec.anchor_seed).unwrap();
        let has_promoted = bond.keyframes().iter().any(|kf| matches!(
            &kf.source,
            KeyframeSource::PromotedFromVariant { variant_id, .. } if *variant_id == winner_id
        ));
        assert!(has_promoted,
            "promoted keyframe with variant_id={} should be on the bond", winner_id);
        // Current-scope weight should now be > 0 (variant keyframe was
        // hidden, promoted keyframe is visible).
        let cw = bond.weight_in_scope(&TemporalScope::Current, &ScarLog::default()).effective_weight;
        assert!(cw > 0.0,
            "current-scope weight should be > 0 after promotion; was {}", cw);
    }

    #[test]
    fn test_no_promotion_below_min_winner_composite() {
        // Crank min_winner_composite above 1.0 — nothing can win.
        let (mut brain, goal) = small_brain();
        let cfg = VariantConfig { min_winner_composite: 1.5, ..VariantConfig::default() };
        let arena = brain.generate_variants_with_config(goal, &cfg);
        assert_eq!(arena.winner_id, None);
        let promoted = brain.promote_variant(&arena);
        assert_eq!(promoted, 0, "no winner -> no promotion");
    }

    #[test]
    fn test_abandoned_variants_decay_naturally() {
        // No promotion: variant keyframes stay in the bond's history
        // but should NOT be visible in Historical scope at a far-future
        // timestamp -- weight_at filters Variant-source keyframes out
        // for non-Variant scopes, so the read is dominated by whatever
        // baseline keyframes existed (decayed by half-life).
        let (mut brain, goal) = small_brain();
        let cfg = VariantConfig { min_winner_composite: 1.5, ..VariantConfig::default() };
        let arena = brain.generate_variants_with_config(goal, &cfg);
        brain.abandon_variants(arena);

        // Pick any variant-touched bond and read its weight in a
        // far-future Historical scope. The variant keyframe must not
        // surface (filtered) and any baseline keyframe will have
        // decayed.
        let dom_anchor = anchor_domain_shift(&brain.network, goal).unwrap();
        let bond = brain.network.get_bond(&goal, &dom_anchor).unwrap();
        let far_future = crate::brain::now_ms() + 1_000 * 86_400_000; // ~1000 days
        let w = bond.weight_in_scope(&TemporalScope::Historical(far_future), &ScarLog::default()).effective_weight;
        // Either the bond predates the variant (decayed baseline
        // weight, near 0) or it was created fresh by the variant
        // (weight 0). Either way, w must be small.
        assert!(w < 0.1, "abandoned variant bond should decay; got {}", w);
    }

    #[test]
    fn test_interdisciplinary_uses_recent_bridges() {
        let (mut brain, goal) = small_brain();
        // Manually seed a bridge between alpha (goal) and a1 -- not
        // beta -- so we can detect that the strategy followed it.
        let a1 = Seed::from_label("a1");
        let (low, high) = if goal.as_bytes() < a1.as_bytes() { (goal, a1) } else { (a1, goal) };
        brain.recent_bridges.push_back(crate::bridge::BridgeEvent {
            low, high,
            low_content: "alpha".into(),
            high_content: "a1".into(),
            bond_weight: 0.1,
            jaccard: 0.0,
            scope: TemporalScope::Current,
            timestamp_ms: 0,
        });
        let arena = brain.generate_variants(goal);
        let inter = arena.variants.iter()
            .find(|v| v.strategy == PerturbationStrategy::Interdisciplinary)
            .expect("Interdisciplinary should fire when recent_bridges has a goal-touching event");
        assert_eq!(inter.anchor_seed, a1);
    }
}
#[cfg(test)]
mod mapram_p5d_variants_tests {
    //! OP-BRAIN-MapRAM P5-batch2 (sub-batch B.2): byte-equivalence +
    //! resident-independence for the variants private adjacency readers
    //! `neighbors_ordered` (out+in, canonicalised via the seam edge
    //! accessors) and `bond_weight_directed` (single directed bond via
    //! `seam_directed_bond`). Both outputs are content-free (a seed list /
    //! an f32 weight) so they are compared in full -- no content exclusion
    //! needed -- and must be invariant under resident poisoning in cache mode.
    use super::*;
    use crate::bond::{BondType, KeyframeSource, TemporalKeyframe};
    use crate::brain::Brain;
    use crate::manifest::{KeyFingerprint, Manifest, MANIFEST_FILE_NAME};
    use crate::network::AssociativeNetwork;
    use crate::scar::ScarLog;
    use crate::seed::Seed;
    use tempfile::tempdir;

    fn kf(w: f32) -> Vec<TemporalKeyframe> {
        vec![TemporalKeyframe { timestamp: 100, weight: w, source: KeyframeSource::Baseline }]
    }
    fn lbl(n: &str) -> Seed { Seed::from_label(&format!("p5d_v_{}", n)) }
    fn boot_v5(dir: &std::path::Path) -> Brain {
        let mut m = Manifest::new(KeyFingerprint::none());
        m.set_brain_format_v5();
        m.save_atomic(&dir.join(MANIFEST_FILE_NAME)).unwrap();
        Brain::load_v5(dir, None).expect("v5 brain must boot via load_v5")
    }
    const SPINE: &[(&str, &str, f32, BondType)] = &[
        ("q", "a", 0.90, BondType::RelatedTo),
        ("q", "b", 0.70, BondType::IsA),
        ("q", "c", 0.50, BondType::HasProperty),
        ("inc", "q", 0.60, BondType::RelatedTo),
        ("inc2", "q", 0.40, BondType::PartOf),
    ];
    fn teach(brain: &Brain) {
        let intern = brain.flush_intern.as_ref().unwrap().clone();
        let mt = brain.memtable().unwrap().clone();
        let mut labels: std::collections::HashSet<&str> = std::collections::HashSet::new();
        for (f, t, _, _) in SPINE.iter().copied() { labels.insert(f); labels.insert(t); }
        for l in labels { intern.intern_or_get(lbl(l)); }
        for (f, t, w, bt) in SPINE.iter().copied() {
            mt.insert(lbl(f), lbl(t), bt, w, kf(w), &intern);
        }
        brain.flush_now().unwrap();
    }
    fn load_routed(dir: &std::path::Path) -> Brain {
        let mut brain = Brain::load_v5(dir, None).expect("reload via load_v5");
        brain.rehydrate_adjacency()
            .expect("rehydrate resident adjacency for legacy/parity comparison");
        brain.enable_cache_backed_adjacency(4096, 0.0).expect("enable cache");
        brain.enable_adjacency_seam().expect("install seam");
        brain
    }
    fn no(net: &AssociativeNetwork) -> Vec<Vec<u8>> {
        neighbors_ordered(net, &lbl("q")).iter().map(|s| s.as_bytes().to_vec()).collect()
    }
    fn bw(net: &AssociativeNetwork) -> (u32, u32) {
        let sl = ScarLog::default();
        (
            bond_weight_directed(net, &lbl("q"), &lbl("a"), &sl).to_bits(),
            bond_weight_directed(net, &lbl("inc"), &lbl("q"), &sl).to_bits(),
        )
    }

    /// `neighbors_ordered` + `bond_weight_directed` are byte-identical legacy
    /// vs cache (canonical seam order both modes) and non-vacuous.
    #[test]
    fn p5d_variants_byte_equivalent() {
        let dir = tempdir().unwrap();
        { let b = boot_v5(dir.path()); teach(&b); }
        let brain = load_routed(dir.path());

        brain.set_adjacency_cache_backed(false);
        let l_no = no(&brain.network);
        let l_bw = bw(&brain.network);
        brain.set_adjacency_cache_backed(true);
        let c_no = no(&brain.network);
        let c_bw = bw(&brain.network);

        assert_eq!(l_no, c_no, "neighbors_ordered byte-differs legacy vs cache");
        assert_eq!(l_bw, c_bw, "bond_weight_directed byte-differs legacy vs cache");
        assert!(c_no.len() >= 5, "neighbors_ordered vacuous (out+in not gathered)");
        assert!(c_bw.0 != 0, "bond_weight_directed vacuous (q->a weight zero)");
    }

    /// INDEPENDENCE: in cache mode both readers are invariant after poisoning
    /// every resident structure the legacy path reads. Sanity: legacy diverges.
    #[test]
    fn p5d_variants_independent_of_resident() {
        let dir = tempdir().unwrap();
        { let b = boot_v5(dir.path()); teach(&b); }
        let mut brain = load_routed(dir.path());
        brain.set_adjacency_cache_backed(true);
        let base_no = no(&brain.network);
        let base_bw = bw(&brain.network);
        assert!(base_no.len() >= 5 && base_bw.0 != 0, "baseline vacuous");

        brain.network.mapram_test_poison_resident();
        brain.network.mapram_test_corrupt_resident_bond_types();
        brain.network.mapram_test_corrupt_resident_keyframes();

        brain.set_adjacency_cache_backed(true);
        assert_eq!(no(&brain.network), base_no, "cache neighbors_ordered changed under poison");
        assert_eq!(bw(&brain.network), base_bw, "cache bond_weight_directed changed under poison");

        brain.set_adjacency_cache_backed(false);
        assert_ne!(no(&brain.network), base_no, "poison must clear the legacy adjacency (else vacuous proof)");
    }
}
