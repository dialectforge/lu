//! Port Sense — runtime sense disambiguation for polysemous concepts.
//!
//! Builds on Ports #17 / #18 / #18a / #18c / #19 / #22 (anchor-aware
//! polysemy disambiguation, label-propagation cluster build, multi-
//! sense queries) by adding a CONTEXT-WINDOW driven scoring path:
//! when the caller has no raw query anchors but DOES have a window
//! of recently-activated nodes (from the current query, the active
//! session, or both), each sense cluster is scored against that
//! context vector and the winning cluster gets full activation gain
//! while losers get an attenuated `loser_gain`.
//!
//! Constraint compliance:
//!  - C1 (encryption structural): no plaintext sidecars; all knobs
//!    persist via section #15 (RuntimeConfigSnapshot) which already
//!    rides inside the encrypted v3enc container.
//!  - C5 (graph-processing-first): no schema additions, no new bond
//!    types, no new v3 sections, no new SourceKind variants. Cluster
//!    detection reuses `polysemy_label_prop_clusters`. Per-bond
//!    inputs (typed/RelatedTo split, decay, trust) are read via
//!    existing accessors only.
//!  - C12 (frontier dedup): no new traversal — scoring is a flat
//!    pass over already-built cluster member lists.
//!
//! Anti-hallucination invariant (STRUCTURAL): every public function
//! in this module is read-only with respect to bond storage. The
//! grep guard test `sense_module_never_mutates_bond_storage` enforces
//! this at the source level by failing the build if any
//! mutating-bond-API surface name appears in this file or in the
//! sense-disambiguation code path inside `network.rs` Sense block.

use crate::seed::Seed;
use std::collections::{HashMap, HashSet};

/// Score for a single sense cluster of a polysemous seed.
#[derive(Debug, Clone, PartialEq)]
pub struct SenseClusterScore {
    /// Internal label-prop label for this cluster.
    pub label: u32,
    /// Member seeds in this cluster (subset of `seed`'s outgoing
    /// neighbors).
    pub members: Vec<Seed>,
    /// Context score: sum over members of
    ///   (direct_hit_bonus if member ∈ context else 0)
    /// + (typed_weight_for_bond(seed→member))
    /// * (1 + neighborhood_overlap(member, context) * NEIGHBORHOOD_OVERLAP_GAIN).
    /// All three terms read existing per-bond/per-graph state — no new fields.
    pub score: f32,
    /// Number of typed (non-RelatedTo) bonds from the seed into
    /// this cluster. Mirrors the `typed_score` carried by the
    /// existing label-propagation helper, kept here so the MCP
    /// surface and the gain-map producer can break score ties
    /// with the same signal.
    pub typed_bond_count: usize,
    /// Whether this cluster is the single winner. At most one
    /// cluster has `is_winner == true`. None are winners when the
    /// top two scores tie.
    pub is_winner: bool,
}

/// Per-cluster breakdown plus the per-neighbor gain map produced by
/// `disambiguate_polysemous_with_context`. `neighbor_gain[member]`
/// is the multiplier subsequent activation/spread should apply to
/// the bond from `seed` into `member`. Winner cluster members get
/// 1.0; loser cluster members get the configured `loser_gain` (so
/// 0.0 = full prune, 1.0 = no disambiguation effect, default 0.25
/// = strong but not absolute preference).
#[derive(Debug, Clone, PartialEq)]
pub struct SenseDisambiguation {
    pub clusters: Vec<SenseClusterScore>,
    pub neighbor_gain: HashMap<Seed, f32>,
    /// True iff a single winner was elected. False when there were
    /// fewer than 2 qualifying clusters or the top two tied.
    pub winner_elected: bool,
}

/// Brain-resident snapshot of the runtime sense knobs. Mirrors the
/// fields stored in v3 section #15 (RuntimeConfigSnapshot) but lives
/// on `Brain` directly so query-time consumers (network.rs, the
/// MCP layer) read it without going through the serde snapshot.
#[derive(Debug, Clone, PartialEq)]
pub struct SenseConfig {
    pub disambiguation_enabled: bool,
    pub context_window: usize,
    pub loser_gain: f32,
}

impl Default for SenseConfig {
    fn default() -> Self {
        SenseConfig {
            disambiguation_enabled: crate::runtime_config::default_sense_disambiguation_enabled(),
            context_window: crate::runtime_config::default_sense_context_window(),
            loser_gain: crate::runtime_config::default_sense_loser_gain(),
        }
    }
}

/// Constants tuned to the existing label-propagation primitive.
/// Tweaks here should not exceed the conservatism of the existing
/// activation-overlap fallback in `disambiguate_polysemous_with_anchors`
/// — Sense is a SOFTER signal than the hard allow-set used there.
pub mod tuning {
    /// A direct hit (member is itself in the context window) carries
    /// this much score on top of the bond/typed component. Picked
    /// at 1.0 to roughly match a single "strong typed bond" of mass.
    pub const DIRECT_HIT_BONUS: f32 = 1.0;
    /// Neighborhood-overlap multiplicative gain. Each context node
    /// that appears among `member`'s outgoing neighbors raises the
    /// member's contribution by this factor.
    pub const NEIGHBORHOOD_OVERLAP_GAIN: f32 = 0.5;
    /// Mass assigned to a typed (non-RelatedTo) bond from the seed
    /// into the member.
    pub const TYPED_BOND_MASS: f32 = 1.0;
    /// Mass assigned to a RelatedTo bond.
    pub const RELATED_BOND_MASS: f32 = 0.5;
    /// A cluster needs at least this many members to count as a
    /// candidate winner. Mirrors the `MIN_MEMBERS = 2` floor used
    /// by `disambiguate_polysemous_multi_sense`.
    pub const MIN_CLUSTER_SIZE: usize = 2;
    /// The winner must lead the runner-up by at least this absolute
    /// score margin. Without a margin, low-signal noise on near-
    /// empty contexts could flip the winner unpredictably.
    pub const MIN_WINNER_MARGIN: f32 = 0.5;
}

/// Compute a score for one cluster's members against a context set,
/// given the per-member typed/related bond mass. Pure function — no
/// graph reads. The caller (`AssociativeNetwork::score_sense_clusters`)
/// is responsible for resolving bond mass and neighborhood overlap
/// and feeding them in.
///
/// Arguments:
///  - `members_with_mass`: each `(Seed, mass, neighborhood_overlap)`
///    triple. `mass` is the per-bond mass (typed or related). The
///    overlap count is how many context nodes appear in the
///    member's outgoing neighborhood.
///  - `context_set`: the context window (used to detect direct hits).
pub(crate) fn score_cluster(
    members_with_mass: &[(Seed, f32, usize)],
    context_set: &HashSet<Seed>,
) -> f32 {
    let mut total = 0.0f32;
    for (m, mass, overlap_count) in members_with_mass {
        let direct = if context_set.contains(m) {
            tuning::DIRECT_HIT_BONUS
        } else {
            0.0
        };
        let overlap_factor = 1.0 + (*overlap_count as f32) * tuning::NEIGHBORHOOD_OVERLAP_GAIN;
        total += direct + (*mass) * overlap_factor;
    }
    total
}

/// Decide which cluster wins given a sorted (descending) score list.
/// Encapsulated so unit tests can drive it without graph state.
pub(crate) fn elect_winner(
    sorted_scores: &[(u32, f32, usize)],
) -> Option<u32> {
    if sorted_scores.len() < 2 {
        return None;
    }
    let (best_lbl, best_score, _) = sorted_scores[0];
    let (_, second_score, _) = sorted_scores[1];
    if best_score <= 0.0 {
        return None;
    }
    if (best_score - second_score) < tuning::MIN_WINNER_MARGIN {
        return None;
    }
    Some(best_lbl)
}

/// Port Sense.1 — per-query runtime cache for sense disambiguation.
///
/// Built once per query at the same site as T.5.1 builds its
/// `DecayRuntime` (top of `Brain::ask` / `ask_in_scope` /
/// `typed_ask` and the agents/thinking recall paths). Stashed on
/// `AssociativeNetwork::sense_runtime` so the 9 query-path
/// effective_weight reads can multiply by the per-bond gain in O(1)
/// without re-running disambiguation per BFS step.
///
/// Default state: `gain_by_seed` is empty -> `sense_gain_for_bond`
/// returns 1.0 for every bond -> byte-for-byte parity with the
/// pre-Sense.1 spread/typed_recall/subgraph_coherence outputs
/// (regression-equivalent guarantee, mirrors T.5.1's None-runtime
/// behaviour).
///
/// Anti-hallucination invariant (STRUCTURAL): this cache is read-
/// only with respect to bond storage. The grep guards in network.rs
/// (`sense1_no_query_path_calls_effective_weight_directly`,
/// `sense1_uses_clock_source_not_system_time`) enforce that every
/// query-path effective_weight read multiplies by the gain via the
/// `sense_gain_for_bond` accessor and that no SystemTime::now
/// sneaks into network.rs.
#[derive(Debug, Clone, PartialEq)]
pub struct SenseRuntime {
    /// Snapshot of the runtime knobs at the moment this runtime was
    /// built. The query-path read consults `disambiguation_enabled`
    /// (so flipping the knob mid-query still routes through the
    /// next refresh).
    pub config: SenseConfig,
    /// `gain_by_seed[polysemous_seed][neighbor]` = per-bond gain for
    /// the bond between `polysemous_seed` and `neighbor`. Winner-
    /// cluster members get 1.0; loser-cluster members get
    /// `config.loser_gain`. Bonds NOT keyed here implicitly use 1.0
    /// (i.e. either the seed is non-polysemous, no winner could be
    /// elected, or this neighbor isn't in any cluster — fall through
    /// gracefully).
    pub gain_by_seed: std::collections::HashMap<
        crate::seed::Seed,
        std::collections::HashMap<crate::seed::Seed, f32>,
    >,
}

impl SenseRuntime {
    /// Construct an empty runtime (no polysemous seeds). Subsequent
    /// `sense_gain_for_bond` reads return 1.0, so this is
    /// regression-equivalent to `set_sense_runtime(None)`.
    pub fn empty(config: SenseConfig) -> Self {
        SenseRuntime {
            config,
            gain_by_seed: std::collections::HashMap::new(),
        }
    }

    /// Number of polysemous seeds in the gain map. 0 means no
    /// disambiguation will fire for this query.
    pub fn polysemous_seed_count(&self) -> usize {
        self.gain_by_seed.len()
    }

    /// O(1) per-bond gain lookup. Tries the outgoing direction
    /// (`from` is the polysemous seed) first, then the incoming
    /// direction (`to` is the polysemous seed). Returns 1.0 if
    /// neither endpoint is a polysemous seed in this runtime, or
    /// disambiguation is disabled.
    #[inline]
    pub fn gain_for_bond(&self, from: &crate::seed::Seed, to: &crate::seed::Seed) -> f32 {
        if !self.config.disambiguation_enabled || self.gain_by_seed.is_empty() {
            return 1.0;
        }
        if let Some(map) = self.gain_by_seed.get(from) {
            if let Some(&g) = map.get(to) { return g; }
        }
        if let Some(map) = self.gain_by_seed.get(to) {
            if let Some(&g) = map.get(from) { return g; }
        }
        1.0
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::seed::Seed;

    fn s(n: u64) -> Seed {
        // Construct a Seed deterministically for tests. Mirror what
        // the production code uses — Seed is a copy type wrapped
        // around a u64 / hash. Tests in network.rs already build
        // synthetic seeds via teach(); here we only need value
        // equality, so any constructor works.
        Seed::from_label(&format!("sense_test_{}", n))
    }

    #[test]
    fn sense_score_cluster_zero_for_empty_context_and_zero_mass() {
        let members = vec![(s(1), 0.0_f32, 0_usize), (s(2), 0.0, 0)];
        let ctx: HashSet<Seed> = HashSet::new();
        assert_eq!(score_cluster(&members, &ctx), 0.0);
    }

    #[test]
    fn sense_score_cluster_typed_bond_dominates_related() {
        let typed = vec![(s(1), tuning::TYPED_BOND_MASS, 0)];
        let related = vec![(s(1), tuning::RELATED_BOND_MASS, 0)];
        let ctx: HashSet<Seed> = HashSet::new();
        let typed_score = score_cluster(&typed, &ctx);
        let related_score = score_cluster(&related, &ctx);
        assert!(typed_score > related_score, "{} > {}", typed_score, related_score);
    }

    #[test]
    fn sense_score_cluster_direct_hit_outranks_neighborhood_overlap() {
        // Member 1: direct hit, no overlap.
        let direct = vec![(s(1), 0.0, 0)];
        let mut ctx_direct: HashSet<Seed> = HashSet::new();
        ctx_direct.insert(s(1));
        let direct_score = score_cluster(&direct, &ctx_direct);

        // Member 2: no direct hit, single overlap, RelatedTo mass.
        let overlap_only = vec![(s(2), tuning::RELATED_BOND_MASS, 1)];
        let ctx_empty: HashSet<Seed> = HashSet::new();
        let overlap_score = score_cluster(&overlap_only, &ctx_empty);

        assert!(direct_score > overlap_score,
            "direct hit {} should dominate single overlap {}", direct_score, overlap_score);
    }

    #[test]
    fn sense_score_cluster_overlap_factor_multiplies_mass() {
        let zero_overlap = vec![(s(1), tuning::TYPED_BOND_MASS, 0)];
        let one_overlap = vec![(s(1), tuning::TYPED_BOND_MASS, 1)];
        let two_overlap = vec![(s(1), tuning::TYPED_BOND_MASS, 2)];
        let ctx: HashSet<Seed> = HashSet::new();
        let s0 = score_cluster(&zero_overlap, &ctx);
        let s1 = score_cluster(&one_overlap, &ctx);
        let s2 = score_cluster(&two_overlap, &ctx);
        assert!(s2 > s1 && s1 > s0, "score must rise monotonically with overlap: {} {} {}", s0, s1, s2);
    }

    #[test]
    fn sense_elect_winner_requires_margin() {
        let tied = vec![(0u32, 5.0_f32, 3usize), (1, 4.6, 3)];
        assert!(elect_winner(&tied).is_none(), "sub-margin lead must not elect winner");
        let clear = vec![(0u32, 5.0, 3), (1, 3.5, 2)];
        assert_eq!(elect_winner(&clear), Some(0));
    }

    #[test]
    fn sense_elect_winner_requires_two_clusters() {
        let single = vec![(0u32, 9.0_f32, 5usize)];
        assert!(elect_winner(&single).is_none());
    }

    #[test]
    fn sense_elect_winner_rejects_zero_score() {
        let zero = vec![(0u32, 0.0_f32, 0usize), (1, 0.0, 0)];
        assert!(elect_winner(&zero).is_none());
    }

    #[test]
    fn sense_loser_gain_default_is_strong_but_not_absolute() {
        // The runtime config default for sense_loser_gain is 0.25 —
        // this test exists so accidentally bumping the default to
        // 0.0 (full prune) or 1.0 (no effect) trips a unit failure
        // alongside the spec.
        let default = crate::runtime_config::default_sense_loser_gain();
        assert!(default > 0.0 && default < 1.0,
            "loser_gain default must be in (0,1): {}", default);
        assert!((default - 0.25).abs() < 1e-6,
            "loser_gain default must be 0.25 to match the Sense spec: {}", default);
    }

    #[test]
    fn sense_context_window_default_is_eight() {
        let default = crate::runtime_config::default_sense_context_window();
        assert_eq!(default, 8);
    }

    #[test]
    fn sense_disambiguation_enabled_default_is_true() {
        let default = crate::runtime_config::default_sense_disambiguation_enabled();
        assert!(default);
    }
}
