//! Baseline consolidation — promotes session-learned knowledge to
//! the baseline graph and prunes decayed keyframes.
//!
//! Runs during idle time. Bonds that have been strengthened across
//! N or more distinct sessions get promoted: a new Baseline keyframe
//! is written with the averaged weight. Old keyframes below the decay
//! threshold are pruned to keep storage bounded.

use crate::bond::{BondType, KeyframeSource, TemporalBond};
use crate::network::{is_hub_stopword, AssociativeNetwork, DEGREE_NORM_THRESHOLD};
use crate::seed::Seed;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Configuration for the consolidation process.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsolidationConfig {
    /// Minimum number of distinct sessions a bond must be strengthened in
    /// before it's promoted to baseline.
    pub promotion_threshold: usize,
    /// Keyframes whose decayed weight falls below this value are pruned.
    pub prune_threshold: f32,
    /// Maximum age (in milliseconds) for keyframes to be considered for pruning.
    pub max_keyframe_age_ms: i64,

    // --- Synthesis (triangle closure) ---
    /// Master switch for the synthesis pass. When false, the phase is skipped
    /// entirely and all synthesis_* counters remain zero.
    pub synthesis_enabled: bool,
    /// Minimum raw keyframe weight required on BOTH legs (A->B and B->C) for a
    /// triangle to be considered for closure.
    pub synthesis_min_leg_weight: f32,
    /// Upper bound on the current raw weight of A->C. If A->C already exceeds
    /// this, we don't reinforce it (and idempotency falls out of this check).
    pub synthesis_max_target_weight: f32,
    /// Multiplier applied to min(w_AB, w_BC) to derive the proposed A->C
    /// weight. Reflects path-length decay: longer inference chains produce
    /// weaker bonds than direct evidence.
    pub synthesis_path_decay: f32,

    // --- Variant prune pass ---
    /// Half-life for variant keyframes (ms). Variant-source keyframes
    /// older than this are dropped during consolidation. Mirrors
    /// `VariantConfig::variant_decay_halflife_ms` so consolidation can
    /// run independently of the variants module. Default 6 hours.
    pub variant_prune_halflife_ms: u64,

    // --- Agent prune pass ---
    /// Half-life for agent keyframes (ms). Agent-source keyframes
    /// older than this are dropped during consolidation. Mirrors
    /// `MultiAgentConfig::agent_keyframe_decay_halflife_ms` for the
    /// same reason as variants — multi-think is observation, the
    /// scope-isolated traces should age out so they don't accumulate.
    /// Default 6 hours.
    pub agent_prune_halflife_ms: u64,

    // --- Synthesis memory caps (Port #13) ---
    /// Hard cap on triangles examined per `synthesize` call. The synthesis
    /// pass is the worst-case allocator on prod-scale brains (937K bonds ->
    /// ~16M triangle examinations would OOM lu_4_mcp at ~31.5 GB anon-RSS).
    /// This bounds transient memory regardless of graph size; remaining
    /// triangles get picked up on subsequent consolidate calls via the
    /// rotating cursor.
    pub max_triangles_per_pass: usize,
    /// Hard cap on `Proposal` structs accumulated per `synthesize` call.
    /// Once reached, examination continues so stats stay honest, but no
    /// further proposals allocate. The dropped count is reported via
    /// `synthesis_proposals_dropped_for_cap`.
    pub max_proposals_per_pass: usize,
    /// Belt-and-suspenders RSS-growth guard: if the synthesis pass adds
    /// more than this many bytes to the process resident set since the
    /// pass started, the loop breaks early with
    /// `synthesis_truncated_oom_guard = true`.
    ///
    /// Pre-2026-05-30 this was an absolute-RSS check, which fired
    /// immediately on production-scale brains (~32 GB resident at the
    /// default 4 GB guard => break before any synthesis work). The
    /// delta semantics work at any brain size: the guard caps growth of
    /// the pass itself, not the daemon's steady-state size. See the
    /// `let start_rss` capture in `synthesize` for the reference frame.
    ///
    /// Set to 0 to disable. Default 4 GB of growth.
    pub rss_oom_guard_bytes: u64,
}

impl Default for ConsolidationConfig {
    fn default() -> Self {
        ConsolidationConfig {
            promotion_threshold: 3,
            prune_threshold: 0.01,
            max_keyframe_age_ms: 7 * 24 * 60 * 60 * 1000, // 7 days

            // Synthesis defaults — tuned to propose only on strong, uncontaminated
            // triangles. See the port notes for the rationale behind each number.
            synthesis_enabled: true,
            synthesis_min_leg_weight: 0.4,
            synthesis_max_target_weight: 0.3,
            synthesis_path_decay: 0.7,

            // Variant prune default: 6 hours, matching VariantConfig.
            variant_prune_halflife_ms: 6 * 60 * 60 * 1000,
            // Agent prune default: 6 hours, matching MultiAgentConfig.
            agent_prune_halflife_ms: 6 * 60 * 60 * 1000,

            // Synthesis memory caps (Port #13). 100K triangles + 1K
            // proposals + 4 GB RSS guard keep the prod brain (937K bonds)
            // under 5 GB RSS during consolidate.
            max_triangles_per_pass: 100_000,
            max_proposals_per_pass: 1_000,
            rss_oom_guard_bytes: 4 * 1024 * 1024 * 1024,
        }
    }
}

/// Statistics from a consolidation run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsolidationStats {
    /// Number of bonds that were promoted to baseline.
    pub bonds_promoted: usize,
    /// Number of keyframes pruned across all bonds.
    pub keyframes_pruned: usize,
    /// Total number of bonds examined.
    pub bonds_examined: usize,
    /// Timestamp when consolidation ran.
    pub ran_at: i64,

    // --- Synthesis counters ---
    /// Every (A, B, C) triple the synthesis phase walked over.
    pub triangles_examined: usize,
    /// Triangles where a Synthesis keyframe was actually written (or a new
    /// A->C bond was created).
    pub triangles_proposed: usize,
    /// Triangles rejected because A, B, or C is a hub (stopword or super-hub).
    pub triangles_skipped_hub: usize,
    /// Triangles rejected because of weight gating or scope mismatch.
    /// (Scope mismatch is folded here rather than given its own counter.)
    pub triangles_skipped_threshold: usize,

    // --- Variant prune counter ---
    /// Variant-source keyframes dropped by the variant prune phase.
    pub variant_keyframes_pruned: usize,

    // --- Agent prune counter ---
    /// Agent-source keyframes dropped by the agent prune phase.
    pub agent_keyframes_pruned: usize,

    // --- Per-graph hub threshold ---
    /// Value of `degree_p90` AFTER the recompute that runs at the end
    /// of consolidation. Lets callers see how the per-graph hub
    /// threshold has shifted as the graph grows.
    pub degree_p90_after_consolidation: u32,

    // --- Synthesis truncation flags (Port #13) ---
    /// True when `synthesize` hit `max_triangles_per_pass` and exited
    /// early. The next consolidate call will continue from a different
    /// starting point via the rotating cursor.
    pub synthesis_truncated: bool,
    /// Number of proposals dropped because `max_proposals_per_pass` was
    /// hit. Examination still proceeds so `triangles_examined` and the
    /// skip counters remain accurate.
    pub synthesis_proposals_dropped_for_cap: usize,
    /// True when `synthesize` broke early because process RSS exceeded
    /// `rss_oom_guard_bytes`. Independent of `synthesis_truncated`.
    pub synthesis_truncated_oom_guard: bool,
}

/// The consolidation engine that promotes and prunes the knowledge graph.
pub struct ConsolidationEngine {
    /// Configuration for this engine.
    pub config: ConsolidationConfig,
    /// Rotating start index into the (sorted) seed list for `synthesize`.
    /// Persisted across calls so successive consolidate runs walk a
    /// different slice of the graph when triangles are capped. Wraps
    /// modulo `seeds.len()`. (Port #13.)
    synthesis_seed_cursor: usize,
}

impl ConsolidationEngine {
    /// Create a new consolidation engine with default config.
    pub fn new() -> Self {
        ConsolidationEngine {
            config: ConsolidationConfig::default(),
            synthesis_seed_cursor: 0,
        }
    }

    /// Create a consolidation engine with custom config.
    pub fn with_config(config: ConsolidationConfig) -> Self {
        ConsolidationEngine {
            config,
            synthesis_seed_cursor: 0,
        }
    }

    /// Run a full consolidation pass over the network.
    ///
    /// For each bond:
    /// 1. Count distinct sessions that have strengthened it.
    /// 2. If count >= `promotion_threshold`, compute the average session
    ///    weight and write a new Baseline keyframe.
    /// 3. Prune old keyframes below the decay threshold.
    pub fn consolidate(
        &mut self,
        network: &mut AssociativeNetwork,
        now: i64,
    ) -> ConsolidationStats {
        let mut stats = ConsolidationStats {
            bonds_promoted: 0,
            keyframes_pruned: 0,
            bonds_examined: 0,
            ran_at: now,
            triangles_examined: 0,
            triangles_proposed: 0,
            triangles_skipped_hub: 0,
            triangles_skipped_threshold: 0,
            variant_keyframes_pruned: 0,
            agent_keyframes_pruned: 0,
            degree_p90_after_consolidation: 0,
            synthesis_truncated: false,
            synthesis_proposals_dropped_for_cap: 0,
            synthesis_truncated_oom_guard: false,
        };

        // Collect bond keys first to avoid borrow issues
        let bond_keys: Vec<(Seed, Seed)> = network
            .bonds()
            .keys()
            .copied()
            .collect();

        for (from, to) in bond_keys {
            stats.bonds_examined += 1;

            let mut bond = match network.get_bond_mut(&from, &to) {
                Some(b) => b,
                None => continue,
            };

            // Step 1: Count distinct sessions and compute average session weight
            let mut session_weights: std::collections::HashMap<u64, Vec<f32>> =
                std::collections::HashMap::new();

            for kf in bond.keyframes() {
                if let KeyframeSource::Session(sid) = &kf.source {
                    session_weights
                        .entry(*sid)
                        .or_insert_with(Vec::new)
                        .push(kf.weight);
                }
            }

            let distinct_sessions = session_weights.len();

            // Step 2: Promote if threshold met
            if distinct_sessions >= self.config.promotion_threshold {
                // Compute average weight across all session keyframes
                let total_weight: f32 = session_weights
                    .values()
                    .flat_map(|weights| weights.iter())
                    .sum();
                let total_count: usize = session_weights
                    .values()
                    .map(|weights| weights.len())
                    .sum();

                if total_count > 0 {
                    let avg_weight = total_weight / total_count as f32;

                    // Write a new Baseline keyframe with the consolidated weight
                    bond.add_keyframe(now, avg_weight, KeyframeSource::Baseline);
                    stats.bonds_promoted += 1;
                }
            }

            // Step 3: Prune old keyframes below decay threshold
            let cutoff = now - self.config.max_keyframe_age_ms;
            let pruned = bond.prune_before(cutoff, self.config.prune_threshold);
            stats.keyframes_pruned += pruned;

            // Safety: always keep at least one keyframe per bond
            if bond.keyframes().is_empty() {
                bond.add_keyframe(now, 0.0, KeyframeSource::Baseline);
            }
        }

        // Step 4: Knowledge synthesis (triangle closure).
        // Runs AFTER promote+prune so we synthesize on the freshly consolidated
        // graph state, not on pre-promotion data.
        self.synthesize(network, now, &mut stats);

        // Step 5: Variant prune pass. Runs LAST so synthesis still
        // sees any variant-tagged keyframes it might want to consider
        // (currently it doesn't, but keeping the order forward-compatible
        // with future passes that read variant history). Drops every
        // KeyframeSource::Variant{..} keyframe older than
        // `variant_prune_halflife_ms`. Variant keyframes are invisible
        // to non-variant scopes via `weight_at` filtering, so dropping
        // them never changes baseline / current / session reads — this
        // is purely a memory + serialization win.
        self.prune_variants(network, now, &mut stats);

        // Step 6: Agent prune pass. Same logic as the variant prune —
        // Agent-source keyframes are scope-isolated, so dropping aged
        // ones never changes baseline / current / session reads. Multi-
        // think is observation, not commitment.
        self.prune_agents(network, now, &mut stats);

        // Step 7: Recompute the per-graph hub threshold so spread/
        // bridge/curiosity see an up-to-date `degree_p90` as the graph
        // grows. Consolidation is the natural periodic-maintenance
        // hook; doing this per-bond would be overkill.
        stats.degree_p90_after_consolidation =
            network.recompute_degree_p90() as u32;
        // Storage v2 Phase 1 (open issue #1): refresh cached typed_ratio
        // on every node, since consolidation may have promoted, demoted
        // or pruned bonds.
        network.recompute_all_typed_ratios();

        stats
    }

    /// Drop every `KeyframeSource::Variant{..}` keyframe whose age
    /// exceeds `variant_prune_halflife_ms`. Returns nothing — counter
    /// is updated on `stats.variant_keyframes_pruned`.
    fn prune_variants(
        &self,
        network: &mut AssociativeNetwork,
        now: i64,
        stats: &mut ConsolidationStats,
    ) {
        let halflife_ms = self.config.variant_prune_halflife_ms as i64;
        if halflife_ms <= 0 {
            return;
        }
        let cutoff = now.saturating_sub(halflife_ms);

        let bond_keys: Vec<(Seed, Seed)> =
            network.bonds().keys().copied().collect();
        for (from, to) in bond_keys {
            let mut bond = match network.get_bond_mut(&from, &to) {
                Some(b) => b,
                None => continue,
            };
            let before = bond.keyframes().len();
            // Retain everything except aged-out Variant keyframes.
            bond.keyframes.retain(|kf| {
                if matches!(kf.source, KeyframeSource::Variant { .. }) {
                    kf.timestamp >= cutoff
                } else {
                    true
                }
            });
            stats.variant_keyframes_pruned += before - bond.keyframes().len();

            // Same safety as the main prune: never let a bond go empty.
            if bond.keyframes().is_empty() {
                bond.add_keyframe(now, 0.0, KeyframeSource::Baseline);
            }
        }
    }

    /// Drop every `KeyframeSource::Agent{..}` keyframe whose age
    /// exceeds `agent_prune_halflife_ms`. Mirrors `prune_variants`.
    fn prune_agents(
        &self,
        network: &mut AssociativeNetwork,
        now: i64,
        stats: &mut ConsolidationStats,
    ) {
        let halflife_ms = self.config.agent_prune_halflife_ms as i64;
        if halflife_ms <= 0 {
            return;
        }
        let cutoff = now.saturating_sub(halflife_ms);
        let bond_keys: Vec<(Seed, Seed)> =
            network.bonds().keys().copied().collect();
        for (from, to) in bond_keys {
            let mut bond = match network.get_bond_mut(&from, &to) {
                Some(b) => b,
                None => continue,
            };
            let before = bond.keyframes().len();
            bond.keyframes.retain(|kf| {
                if matches!(kf.source, KeyframeSource::Agent { .. }) {
                    kf.timestamp >= cutoff
                } else {
                    true
                }
            });
            stats.agent_keyframes_pruned += before - bond.keyframes().len();
            if bond.keyframes().is_empty() {
                bond.add_keyframe(now, 0.0, KeyframeSource::Baseline);
            }
        }
    }

    /// Triangle-closure pass: for every A->B->C chain whose legs are strong
    /// and scope-compatible, propose strengthening A->C (or creating it if
    /// missing) with a `KeyframeSource::Synthesis` keyframe.
    ///
    /// Algorithm:
    /// 1. Pre-compute a hub set (stopwords OR degree > `DEGREE_NORM_THRESHOLD`).
    /// 2. Walk every (A, B, C) triple using outgoing adjacency.
    /// 3. Skip if any endpoint is a hub, if legs are below threshold, if
    ///    the current A->C is already strong, or if the two legs come from
    ///    incompatible sessions.
    /// 4. Collect proposals immutably, then apply them in a second pass to
    ///    sidestep borrow-checker conflicts.
    fn synthesize(
        &mut self,
        network: &mut AssociativeNetwork,
        now: i64,
        stats: &mut ConsolidationStats,
    ) {
        if !self.config.synthesis_enabled {
            return;
        }

        // Hub set: function-word stopwords (per network.rs) OR super-hubs by
        // degree. Triangle closure through hubs creates spurious bonds, so we
        // reuse the same criteria spread-activation uses for suppression.
        let mut seeds: Vec<Seed> = network.all_seeds();
        // Sort by raw bytes so the rotating cursor walks a deterministic
        // order across calls -- HashMap iteration order is otherwise arbitrary
        // and would defeat the "continue from a different starting point"
        // guarantee the cap relies on.
        seeds.sort_unstable_by(|a, b| a.0.cmp(&b.0));

        let mut hubs: HashSet<Seed> = HashSet::new();
        for s in &seeds {
            let deg = network.seam_total_degree(s);
            let is_stop = network
                .get_node(s)
                .map(|n| is_hub_stopword(n.content()))
                .unwrap_or(false);
            if is_stop || deg > DEGREE_NORM_THRESHOLD {
                hubs.insert(*s);
            }
        }

        // `seen` dedupes (A, C) across multiple intermediate B's. Without it, a
        // highly-connected graph produces combinatorial blowup of proposals.
        let mut proposals: Vec<Proposal> = Vec::new();
        let mut seen: HashSet<(Seed, Seed)> = HashSet::new();

        // Port #13: hard caps. We bound triangles examined and proposals
        // accumulated so consolidate stays under the RSS budget on prod-scale
        // brains. The cursor lets successive calls cover the full graph.
        let triangle_cap = self.config.max_triangles_per_pass;
        let proposal_cap = self.config.max_proposals_per_pass;
        let rss_guard = self.config.rss_oom_guard_bytes;
        // RSS-growth guard baseline: capture process resident bytes at
        // pass entry so the in-loop check fires on PASS GROWTH rather
        // than absolute RSS. Pre-2026-05-30 the guard read absolute RSS
        // and structurally failed on production-sized brains (~32 GB
        // resident at default 4 GB guard => break before any work). See
        // the doc on `rss_oom_guard_bytes` for the full migration note.
        let start_rss = current_rss_bytes().unwrap_or(0);

        let n_seeds = seeds.len();
        // How many seeds we visited as A on this pass. Used to advance the
        // cursor whether we exited cleanly or via cap.
        let mut a_visited: usize = 0;
        // Sample RSS every this many examinations -- frequent enough to react,
        // cheap enough not to dominate the loop.
        const RSS_SAMPLE_EVERY: usize = 8_192;

        let start = if n_seeds == 0 { 0 } else { self.synthesis_seed_cursor % n_seeds };

        'outer: for offset in 0..n_seeds {
            let a_idx = (start + offset) % n_seeds;
            let a = &seeds[a_idx];
            a_visited += 1;
            let a_is_hub = hubs.contains(a);
            // Clone adjacency so we're not holding a borrow on `network` during
            // the inner `get_bond` calls.
            let out_a: Vec<Seed> = network.seam_neighbors_outgoing(a).to_vec();
            for b in out_a {
                if b == *a {
                    continue;
                }
                let b_is_hub = hubs.contains(&b);

                // Fetch A->B once per B (reused across all its C's).
                let (ab_raw, ab_source) = match network.get_bond(a, &b) {
                    Some(bn) => match bn.keyframes_iter().last() {
                        Some(kf) => (kf.weight, kf.source),
                        None => continue,
                    },
                    None => continue,
                };

                let out_b: Vec<Seed> = network.seam_neighbors_outgoing(&b).to_vec();
                for c in out_b {
                    if c == *a || c == b {
                        continue;
                    }

                    // Triangle cap: stop the whole pass once we've examined
                    // enough. This is the primary OOM defence.
                    if triangle_cap > 0 && stats.triangles_examined >= triangle_cap {
                        stats.synthesis_truncated = true;
                        break 'outer;
                    }

                    // RSS guard: belt-and-suspenders against the OOM scenario.
                    // Sample cheaply every RSS_SAMPLE_EVERY examinations.
                    if rss_guard > 0
                        && stats.triangles_examined > 0
                        && stats.triangles_examined % RSS_SAMPLE_EVERY == 0
                    {
                        if let Some(rss) = current_rss_bytes() {
                            // Delta semantics: fire on pass GROWTH, not
                            // absolute RSS. saturating_sub guards against
                            // RSS shrinkage (e.g. malloc_trim mid-pass).
                            if rss.saturating_sub(start_rss) >= rss_guard {
                                stats.synthesis_truncated_oom_guard = true;
                                stats.synthesis_truncated = true;
                                break 'outer;
                            }
                        }
                    }

                    stats.triangles_examined += 1;

                    // Hub guard: any of A, B, C being a hub invalidates it.
                    // Cheap check first -- gate before any allocation.
                    if a_is_hub || b_is_hub || hubs.contains(&c) {
                        stats.triangles_skipped_hub += 1;
                        continue;
                    }

                    // Dedupe so each (A, C) gets at most one proposal even if
                    // multiple B's connect them.
                    if !seen.insert((*a, c)) {
                        continue;
                    }

                    // Fetch B->C leg.
                    let (bc_raw, bc_source) = match network.get_bond(&b, &c) {
                        Some(bn) => match bn.keyframes_iter().last() {
                            Some(kf) => (kf.weight, kf.source),
                            None => continue,
                        },
                        None => continue,
                    };

                    // Scope compatibility: the two legs must not belong to
                    // different sessions. Anything else (Baseline, Hebbian,
                    // Manual, Synthesis) is frame-neutral.
                    if !sources_compatible(&ab_source, &bc_source) {
                        stats.triangles_skipped_threshold += 1;
                        continue;
                    }

                    // Leg-strength gate: both legs must exceed the floor.
                    let leg_min = ab_raw.min(bc_raw);
                    if leg_min < self.config.synthesis_min_leg_weight {
                        stats.triangles_skipped_threshold += 1;
                        continue;
                    }

                    // Target gate: current A->C must be weak (or absent).
                    // Absent => raw = 0.0, which passes the < upper check.
                    let ac_raw = network
                        .get_bond(a, &c)
                        .and_then(|bn| bn.keyframes_iter().last())
                        .map(|k| k.weight)
                        .unwrap_or(0.0);
                    if ac_raw >= self.config.synthesis_max_target_weight {
                        stats.triangles_skipped_threshold += 1;
                        continue;
                    }

                    // Proposal cap: keep examining (so stats stay honest)
                    // but stop allocating Proposal structs.
                    if proposal_cap > 0 && proposals.len() >= proposal_cap {
                        stats.synthesis_proposals_dropped_for_cap += 1;
                        continue;
                    }

                    let proposed_weight = leg_min * self.config.synthesis_path_decay;
                    let create_if_missing = network.get_bond(a, &c).is_none();
                    proposals.push(Proposal {
                        a: *a,
                        c,
                        new_weight: proposed_weight,
                        path: [*a, b, c],
                        create_if_missing,
                    });
                }
            }
        }

        // Advance the cursor so the next consolidate call begins at a
        // different A. Even on a clean (uncapped) pass we still rotate -- on
        // a graph small enough to fit one pass this is harmless, and on a
        // capped pass it guarantees forward progress across the seed list.
        if n_seeds > 0 {
            let advance = a_visited.max(1);
            self.synthesis_seed_cursor = (start + advance) % n_seeds;
        }

        // Apply pass: mutate the network now that the immutable walk is done.
        for p in proposals {
            // OP-IDENTITY: consolidation / self-correction must never touch the
            // protected identity core -- skip any proposal whose endpoint is a
            // locked identity node.
            let la = network.node_label(&p.a).unwrap_or_default();
            let lc = network.node_label(&p.c).unwrap_or_default();
            if crate::identity::is_protected_label(&la)
                || crate::identity::is_protected_label(&lc)
            {
                continue;
            }
            let source = KeyframeSource::Synthesis { from_path: p.path };
            if p.create_if_missing {
                // Create a fresh bond. We don't know the semantic relation, so
                // default to RelatedTo (a generic associative link). The initial
                // keyframe is retagged Synthesis so the bond is recognizably
                // inferred rather than baseline.
                let mut bond = TemporalBond::new(
                    p.a,
                    p.c,
                    BondType::RelatedTo,
                    p.new_weight,
                    now,
                );
                if let Some(first) = bond.keyframes.first_mut() {
                    first.source = source.clone();
                }
                let _ = network.add_bond(bond);
            } else if let Some(mut bond) = network.get_bond_mut(&p.a, &p.c) {
                bond.add_keyframe(now, p.new_weight, source);
            }
            stats.triangles_proposed += 1;
        }
    }

    /// Check if consolidation should run based on time since last run.
    ///
    /// Returns true if enough time has passed since the last consolidation
    /// (default: 1 hour minimum between runs).
    pub fn should_run(&self, last_ran_at: Option<i64>, now: i64) -> bool {
        match last_ran_at {
            None => true,
            Some(last) => {
                let min_interval_ms = 60 * 60 * 1000; // 1 hour
                (now - last) >= min_interval_ms
            }
        }
    }

    /// Identify bonds that are candidates for promotion but haven't
    /// reached the threshold yet. Useful for monitoring.
    pub fn pending_promotions(
        &self,
        network: &AssociativeNetwork,
    ) -> Vec<(Seed, Seed, usize)> {
        let mut pending = Vec::new();

        for (&(from, to), bond) in network.bonds().iter() {
            let count = bond.distinct_session_count();
            if count > 0 && count < self.config.promotion_threshold {
                pending.push((from, to, count));
            }
        }

        // Sort by session count descending (closest to promotion first)
        pending.sort_by(|a, b| b.2.cmp(&a.2));
        pending
    }
}

impl Default for ConsolidationEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// A pending synthesis output: either strengthen an existing A->C bond or
/// create a fresh one. We collect these during the immutable walk and apply
/// them all at once to dodge borrow-checker conflicts.
struct Proposal {
    a: Seed,
    c: Seed,
    new_weight: f32,
    /// The [A, B, C] triple that justified the closure. Stored in the
    /// `KeyframeSource::Synthesis` variant for later time-travel audit.
    path: [Seed; 3],
    create_if_missing: bool,
}

/// True when two keyframe sources belong to the same "temporal frame" —
/// i.e. it's safe to combine them into a synthesis proposal.
///
/// Rule: `Session(x)` and `Session(y)` must match on `x == y`. Everything
/// else (Baseline, Hebbian, Manual, Synthesis, PromotedFromVariant, or
/// mixed with Baseline) is frame-neutral and compatible. This prevents
/// the session-bleed problem the Lu 4.0 temporal rewrite was built to
/// solve. `PromotedFromVariant` is included because once a variant
/// wins it represents settled, baseline-visible knowledge — synthesis
/// triangles must be able to fire on those bonds, otherwise winning a
/// variant arena dead-ends the inference chain.
fn sources_compatible(a: &KeyframeSource, b: &KeyframeSource) -> bool {
    match (a, b) {
        (KeyframeSource::Session(x), KeyframeSource::Session(y)) => x == y,
        _ => true,
    }
}

/// Read resident-set size of the current process from `/proc/self/statm`.
/// Returns `None` on platforms where the file is unavailable. Used by the
/// synthesis pass to break early before the OOM killer fires (Port #13).
fn current_rss_bytes() -> Option<u64> {
    // statm format: "size resident shared text lib data dt" (in pages).
    let raw = std::fs::read_to_string("/proc/self/statm").ok()?;
    let resident_pages: u64 = raw.split_whitespace().nth(1)?.parse().ok()?;
    // page_size: 4 KiB on every Linux target Lu runs on. We don't link
    // libc just for sysconf -- if the page size ever differs we'll under-
    // estimate, which fails safe (guard fires later, not never).
    Some(resident_pages.saturating_mul(4096))
}

// =====================================================================
// Port "internal-first" (2026-05-30) — focused triangle closure for
// a single seed. Used by the idle loop to fill a wonder's gap from
// existing graph knowledge BEFORE attempting external web research.
// Reuses every gate from the synthesize() whole-graph pass:
//   * hub-stopword + degree-threshold exclusion (a hub seed produces
//     spurious closures because everything touches it),
//   * min_leg_weight floor on both A→B and B→C,
//   * max_target_weight ceiling on existing A→C (idempotency),
//   * source compatibility (Session(x) and Session(y) must match;
//     other sources are frame-neutral),
//   * path_decay applied to the proposed weight,
//   * dedupe by (A, C) so multiple B's don't multi-propose.
//
// Scoped to a single A seed (the wonder seed), bounded by
// INTERNAL_FILL_MAX_NEIGHBORS to keep per-call cost predictable on
// high-degree hubs (degree² can be huge otherwise).
// =====================================================================

/// Per-pass cap on B-neighbors examined when filling a single seed.
/// 20² = 400 triangle examinations worst case, which is sub-100ms in
/// practice and meets the "cheap enough to run on every idle tick"
/// bar that the original 100K cap on the whole-graph pass was tuned
/// against. Promote to ConsolidationConfig if runtime tuning is
/// needed later.
pub const INTERNAL_FILL_MAX_NEIGHBORS: usize = 20;

/// A synthesis proposal produced by `internal_fill_at_seed`. Public
/// counterpart to the module-private `Proposal` struct used by the
/// whole-graph synthesize pass. Idle-loop callers push these into
/// `PendingLog` as `PendingChangeKind::AddBond` (or weight-bump)
/// entries — they DO NOT mutate the brain directly.
#[derive(Debug, Clone)]
pub struct InternalFillProposal {
    pub a: Seed,
    pub c: Seed,
    pub new_weight: f32,
    /// The [A, B, C] triple that justified the closure. Surfaces in
    /// the PendingChange.reason so reviewers can audit the path.
    pub path: [Seed; 3],
    /// True when no A→C bond currently exists (create-mode), false
    /// when an existing weak A→C would be reinforced (strengthen-mode).
    pub create_if_missing: bool,
}

/// Local triangle closure around a single seed. Returns proposals
/// without mutating the network — the caller (idle_tick) decides
/// whether to push them to PendingLog. Matches the gates of
/// `ConsolidationEngine::synthesize` so internal-fill bonds and
/// whole-graph synthesis bonds carry identical quality thresholds.
pub fn internal_fill_at_seed(
    network: &AssociativeNetwork,
    seed: &Seed,
    config: &ConsolidationConfig,
    max_neighbors: usize,
) -> Vec<InternalFillProposal> {
    if !config.synthesis_enabled {
        return Vec::new();
    }

    let a = seed;
    // Hub-A gate: a hub seed produces spurious closures because
    // everything touches it. Match the whole-graph pass's check.
    let a_deg = network.seam_total_degree(a);
    let a_is_stop = network.get_node(a).map(|n| is_hub_stopword(n.content())).unwrap_or(false);
    if a_is_stop || a_deg > DEGREE_NORM_THRESHOLD {
        return Vec::new();
    }

    let mut proposals: Vec<InternalFillProposal> = Vec::new();
    let mut seen: std::collections::HashSet<(Seed, Seed)> = std::collections::HashSet::new();

    // Clone adjacency so we're not holding a borrow on `network`
    // during the inner get_bond calls. Bounded by max_neighbors so
    // a high-degree wonder doesn't blow up cost.
    let out_a: Vec<Seed> = network.seam_neighbors_outgoing(a).iter().take(max_neighbors).copied().collect();

    for b in &out_a {
        if b == a {
            continue;
        }
        // Hub-B gate.
        let b_deg = network.seam_total_degree(b);
        let b_is_stop = network.get_node(b).map(|n| is_hub_stopword(n.content())).unwrap_or(false);
        if b_is_stop || b_deg > DEGREE_NORM_THRESHOLD {
            continue;
        }

        // Fetch A→B once per B (reused across all its C's).
        let (ab_raw, ab_source) = match network.get_bond(a, b) {
            Some(bn) => match bn.keyframes_iter().last() {
                Some(kf) => (kf.weight, kf.source),
                None => continue,
            },
            None => continue,
        };

        let out_b: Vec<Seed> = network.seam_neighbors_outgoing(b).iter().take(max_neighbors).copied().collect();
        for c in out_b {
            if c == *a || c == *b {
                continue;
            }
            // Hub-C gate.
            let c_deg = network.seam_total_degree(&c);
            let c_is_stop = network.get_node(&c).map(|n| is_hub_stopword(n.content())).unwrap_or(false);
            if c_is_stop || c_deg > DEGREE_NORM_THRESHOLD {
                continue;
            }
            // Dedupe so each (A, C) gets at most one proposal even if
            // multiple B's connect them.
            if !seen.insert((*a, c)) {
                continue;
            }
            // Fetch B→C.
            let (bc_raw, bc_source) = match network.get_bond(b, &c) {
                Some(bn) => match bn.keyframes_iter().last() {
                    Some(kf) => (kf.weight, kf.source),
                    None => continue,
                },
                None => continue,
            };
            // Scope compatibility — must not cross session boundaries.
            if !sources_compatible(&ab_source, &bc_source) {
                continue;
            }
            // Both legs must clear the floor.
            let leg_min = ab_raw.min(bc_raw);
            if leg_min < config.synthesis_min_leg_weight {
                continue;
            }
            // Existing A→C must be weak/absent.
            let ac_raw = network
                .get_bond(a, &c)
                .and_then(|bn| bn.keyframes_iter().last())
                .map(|k| k.weight)
                .unwrap_or(0.0);
            if ac_raw >= config.synthesis_max_target_weight {
                continue;
            }
            let proposed_weight = leg_min * config.synthesis_path_decay;
            let create_if_missing = network.get_bond(a, &c).is_none();
            proposals.push(InternalFillProposal {
                a: *a,
                c,
                new_weight: proposed_weight,
                path: [*a, *b, c],
                create_if_missing,
            });
        }
    }
    proposals
}

// =====================================================================
// Port "hub-fill" (2026-05-30, follow-on to internal-first) —
// internal fill semantics for HubBoundary and DeadEnd wonders, whose
// SEED is a hub. The port-1 internal_fill_at_seed correctly skips
// hubs (triangle closures THROUGH hubs are spurious), which means the
// most common wonder types benefit from nothing in port 1 alone.
//
// Different semantics here: look at the hub's NEIGHBORS B, C. If both
// A→B and A→C are strong and B, C are themselves non-hubs and existing
// B→C is weak/absent → propose B→C with weight =
// min(w_AB, w_AC) × path_decay² (squared because sideways-through-a-hub
// is weaker evidence than direct transitivity).
// =====================================================================

/// Per-leg weight floor when the connecting node is a hub. Stricter
/// than the synthesis_min_leg_weight used for transitive closures —
/// shared-hub co-occurrence is a weaker inference, so we want both
/// arms strong before proposing the lateral bond.
pub const HUB_LEG_MIN_WEIGHT: f32 = 0.4;  // lowered 2026-05-30 — match synthesis_min_leg_weight after empirical verification

/// Internal fill via shared-hub co-occurrence. For a hub seed A,
/// proposes B↔C bonds where B and C are both A's neighbors. Mirrors
/// `internal_fill_at_seed`'s contract: returns proposals without
/// mutating; caller decides whether to push to PendingLog.
pub fn internal_fill_at_hub(
    network: &AssociativeNetwork,
    hub_seed: &Seed,
    config: &ConsolidationConfig,
    max_neighbors: usize,
    min_hub_leg_weight: f32,
) -> Vec<InternalFillProposal> {
    if !config.synthesis_enabled {
        return Vec::new();
    }

    // Gather hub's outgoing neighbors. We deliberately DO NOT skip
    // when the hub itself is super-hub or stopword — that's the whole
    // point of HubBoundary signals. The protection is the B/C gate
    // below (don't link hub to hub through hub).
    // OP-BRAIN-MapRAM P5-batch2: gather the hub's outgoing neighbours
    // WITH their edge bond from the canonical, uncapped, edge-carrying
    // seam. CACHE reconstructs each bond byte-identically from the
    // carried `BondPayload` (never resident `get_bond`); LEGACY is the
    // prior `get_bond`. Canonical seed-byte order in BOTH modes makes
    // `take(max_neighbors)` deterministic and identical across paths
    // (legacy resident order was `HashMap::drain`-random per load). Each
    // leg carries its last-keyframe (weight, source) -- exactly the
    // `get_bond(hub, nbr).keyframes_iter().last()` the old code read.
    let legs: Vec<(Seed, f32, KeyframeSource)> = network
        .seam_walk_outgoing(hub_seed)
        .into_iter()
        .take(max_neighbors)
        .filter_map(|(nbr, bond)| {
            let bond = bond?;
            let kf = bond.view().keyframes_iter().last()?;
            Some((nbr, kf.weight, kf.source))
        })
        .collect();

    let mut proposals: Vec<InternalFillProposal> = Vec::new();
    let mut seen: std::collections::HashSet<(Seed, Seed)> = std::collections::HashSet::new();

    for (i, leg) in legs.iter().enumerate() {
        let b = leg.0;
        let ab_raw = leg.1;
        let ab_source = &leg.2;
        if b == *hub_seed {
            continue;
        }
        // B must NOT itself be a hub -- otherwise we are linking hub to
        // hub through hub, which is the spurious pattern.
        let b_deg = network.seam_total_degree(&b);
        let b_is_stop = network.get_node(&b).map(|n| is_hub_stopword(n.content())).unwrap_or(false);
        if b_is_stop || b_deg > DEGREE_NORM_THRESHOLD {
            continue;
        }
        if ab_raw < min_hub_leg_weight {
            continue;
        }

        // Iterate over the OTHER neighbors, dedup-by-pair so we only
        // visit each (B, C) once.
        for cleg in legs.iter().skip(i + 1) {
            let c = cleg.0;
            let ac_raw = cleg.1;
            let ac_source = &cleg.2;
            if c == b || c == *hub_seed {
                continue;
            }
            let c_deg = network.seam_total_degree(&c);
            let c_is_stop = network.get_node(&c).map(|n| is_hub_stopword(n.content())).unwrap_or(false);
            if c_is_stop || c_deg > DEGREE_NORM_THRESHOLD {
                continue;
            }
            if !seen.insert((b, c)) {
                continue;
            }
            if ac_raw < min_hub_leg_weight {
                continue;
            }
            if !sources_compatible(ab_source, ac_source) {
                continue;
            }
            // Existing B->C must be weak/absent -- idempotency gate.
            // Sourced from the seam's directed-bond accessor (CACHE:
            // carried payload; LEGACY: `get_bond`), so the B->C leg also
            // avoids resident `get_bond` in cache mode.
            let bc_bond = network.seam_directed_bond(&b, &c);
            let bc_raw = bc_bond
                .as_ref()
                .and_then(|sb| sb.view().keyframes_iter().last())
                .map(|k| k.weight)
                .unwrap_or(0.0);
            if bc_raw >= config.synthesis_max_target_weight {
                continue;
            }
            // Sideways through hub: weaker than direct transitive
            // closure. Square the path_decay to reflect the indirect
            // inference.
            let decay = config.synthesis_path_decay * config.synthesis_path_decay;
            let proposed_weight = ab_raw.min(ac_raw) * decay;
            let create_if_missing = bc_bond.is_none();
            proposals.push(InternalFillProposal {
                a: b,        // B becomes the bond's `from`
                c,           // C is the bond's `to`
                new_weight: proposed_weight,
                // Path attribute records the inference shape:
                // "B and C both connect through hub A."
                path: [b, *hub_seed, c],
                create_if_missing,
            });
        }
    }
    proposals
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bond::{BondType, TemporalBond};
    use crate::node::{Confidence, KnowledgeContent, KnowledgeNode};

    fn setup_network_with_sessions() -> AssociativeNetwork {
        let mut net = AssociativeNetwork::new();
        let sa = Seed::from_label("ConA");
        let sb = Seed::from_label("ConB");

        net.add_node(KnowledgeNode::with_seed(
            sa, "ConA".into(),
            KnowledgeContent::Raw("Node A".into()),
            Confidence::Verified, 0,
        ));
        net.add_node(KnowledgeNode::with_seed(
            sb, "ConB".into(),
            KnowledgeContent::Raw("Node B".into()),
            Confidence::Verified, 0,
        ));

        let mut bond = TemporalBond::new(sa, sb, BondType::RelatedTo, 0.5, 0);
        // Add keyframes from 3 distinct sessions
        bond.add_keyframe(1000, 0.6, KeyframeSource::Session(1));
        bond.add_keyframe(2000, 0.7, KeyframeSource::Session(2));
        bond.add_keyframe(3000, 0.8, KeyframeSource::Session(3));

        net.add_bond(bond).unwrap();
        net
    }

    #[test]
    fn test_consolidation_promotes_bonds() {
        let mut net = setup_network_with_sessions();
        let mut engine = ConsolidationEngine::with_config(ConsolidationConfig {
            promotion_threshold: 3,
            prune_threshold: 0.01,
            max_keyframe_age_ms: 100_000,
            ..Default::default()
        });

        let stats = engine.consolidate(&mut net, 50_000);
        assert_eq!(stats.bonds_promoted, 1);
        assert_eq!(stats.bonds_examined, 1);

        // Check that a new baseline keyframe was added
        let sa = Seed::from_label("ConA");
        let sb = Seed::from_label("ConB");
        let bond = net.get_bond(&sa, &sb).unwrap();
        let baseline_kfs: Vec<_> = bond
            .keyframes()
            .iter()
            .filter(|kf| matches!(kf.source, KeyframeSource::Baseline))
            .collect();
        // Original baseline + new promoted baseline
        assert!(baseline_kfs.len() >= 2);
    }

    #[test]
    fn test_consolidation_prunes_old_keyframes() {
        let mut net = setup_network_with_sessions();
        let mut engine = ConsolidationEngine::with_config(ConsolidationConfig {
            promotion_threshold: 10, // Won't promote
            prune_threshold: 0.5,    // Aggressive threshold
            max_keyframe_age_ms: 2000, // Short window
            ..Default::default()
        });

        let sa = Seed::from_label("ConA");
        let sb = Seed::from_label("ConB");
        let before_count = net.get_bond(&sa, &sb).unwrap().keyframes().len();

        let stats = engine.consolidate(&mut net, 100_000);
        let after_count = net.get_bond(&sa, &sb).unwrap().keyframes().len();

        assert!(stats.keyframes_pruned > 0);
        assert!(after_count <= before_count);
    }

    #[test]
    fn test_pending_promotions() {
        let mut net = AssociativeNetwork::new();
        let sa = Seed::from_label("PendA");
        let sb = Seed::from_label("PendB");

        net.add_node(KnowledgeNode::with_seed(
            sa, "PendA".into(),
            KnowledgeContent::Raw("A".into()),
            Confidence::Verified, 0,
        ));
        net.add_node(KnowledgeNode::with_seed(
            sb, "PendB".into(),
            KnowledgeContent::Raw("B".into()),
            Confidence::Verified, 0,
        ));

        let mut bond = TemporalBond::new(sa, sb, BondType::RelatedTo, 0.5, 0);
        bond.add_keyframe(100, 0.6, KeyframeSource::Session(1));
        bond.add_keyframe(200, 0.7, KeyframeSource::Session(2));
        net.add_bond(bond).unwrap();

        let engine = ConsolidationEngine::new(); // threshold = 3
        let pending = engine.pending_promotions(&net);
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].2, 2); // 2 sessions so far
    }

    #[test]
    fn test_should_run_timing() {
        let engine = ConsolidationEngine::new();
        assert!(engine.should_run(None, 0));
        assert!(!engine.should_run(Some(0), 1000));
        assert!(engine.should_run(Some(0), 4_000_000)); // > 1 hour
    }

    // ----- Synthesis (triangle closure) tests -----

    /// Helper: add a node with the given label as its content.
    fn add_node(net: &mut AssociativeNetwork, label: &str) -> Seed {
        let s = Seed::from_label(label);
        net.add_node(KnowledgeNode::with_seed(
            s,
            label.into(),
            KnowledgeContent::Raw(label.into()),
            Confidence::Verified,
            0,
        ));
        s
    }

    /// Helper: disable promotion and prune so tests only exercise synthesis.
    fn synthesis_only_config() -> ConsolidationConfig {
        ConsolidationConfig {
            promotion_threshold: 1_000_000, // effectively disabled
            prune_threshold: 0.0,           // don't drop anything
            max_keyframe_age_ms: i64::MAX / 2,
            synthesis_enabled: true,
            synthesis_min_leg_weight: 0.4,
            synthesis_max_target_weight: 0.3,
            synthesis_path_decay: 0.7,
            // Variant prune disabled in synthesis-only tests (i64-cast
            // overflow danger if we set MAX; 0 short-circuits the pass).
            variant_prune_halflife_ms: 0,
            agent_prune_halflife_ms: 0,
            // Generous caps so the existing synthesis tests keep their old
            // semantics; the cap-specific tests override these explicitly.
            max_triangles_per_pass: 1_000_000,
            max_proposals_per_pass: 1_000_000,
            rss_oom_guard_bytes: 0,
        }
    }

    /// (a) Basic A->B->C with weak A->C produces a Synthesis keyframe on A->C.
    #[test]
    fn test_synthesis_basic_triangle_closes() {
        let mut net = AssociativeNetwork::new();
        let a = add_node(&mut net, "SynA");
        let b = add_node(&mut net, "SynMid");
        let c = add_node(&mut net, "SynC");

        // Strong legs, weak target.
        net.add_bond(TemporalBond::new(a, b, BondType::RelatedTo, 0.8, 0)).unwrap();
        net.add_bond(TemporalBond::new(b, c, BondType::RelatedTo, 0.8, 0)).unwrap();
        net.add_bond(TemporalBond::new(a, c, BondType::RelatedTo, 0.1, 0)).unwrap();

        let mut engine = ConsolidationEngine::with_config(synthesis_only_config());
        let stats = engine.consolidate(&mut net, 10_000);

        assert_eq!(stats.triangles_examined, 1);
        assert_eq!(stats.triangles_proposed, 1);
        assert_eq!(stats.triangles_skipped_hub, 0);
        assert_eq!(stats.triangles_skipped_threshold, 0);

        // Confirm a Synthesis keyframe landed on A->C.
        let ac = net.get_bond(&a, &c).expect("A->C bond should still exist");
        let synth_count = ac.keyframes().iter().filter(|k| {
            matches!(k.source, KeyframeSource::Synthesis { .. })
        }).count();
        assert_eq!(synth_count, 1, "expected exactly one Synthesis keyframe on A->C");

        // Expected proposed weight: min(0.8, 0.8) * 0.7 = 0.56.
        let latest = ac.keyframes().last().unwrap();
        assert!((latest.weight - 0.56).abs() < 1e-5);
    }

    /// (b) Hub B is correctly skipped: even with strong legs, triangle through
    ///     a stopword hub should NOT produce a Synthesis keyframe.
    #[test]
    fn test_synthesis_skips_hub_b() {
        let mut net = AssociativeNetwork::new();
        let a = add_node(&mut net, "HubA");
        // "the" is in is_hub_stopword — a canonical function-word hub.
        let b = add_node(&mut net, "the");
        let c = add_node(&mut net, "HubC");

        net.add_bond(TemporalBond::new(a, b, BondType::RelatedTo, 0.9, 0)).unwrap();
        net.add_bond(TemporalBond::new(b, c, BondType::RelatedTo, 0.9, 0)).unwrap();
        net.add_bond(TemporalBond::new(a, c, BondType::RelatedTo, 0.1, 0)).unwrap();

        let mut engine = ConsolidationEngine::with_config(synthesis_only_config());
        let stats = engine.consolidate(&mut net, 10_000);

        assert_eq!(stats.triangles_examined, 1);
        assert_eq!(stats.triangles_proposed, 0);
        assert_eq!(stats.triangles_skipped_hub, 1);

        // A->C should have no Synthesis keyframe.
        let ac = net.get_bond(&a, &c).unwrap();
        let has_synth = ac.keyframes().iter().any(|k| {
            matches!(k.source, KeyframeSource::Synthesis { .. })
        });
        assert!(!has_synth, "hub B must block synthesis");
    }

    /// (c) Cross-scope legs: A->B(Session 1) and B->C(Session 2) must NOT
    ///     close into A->C. This is the session-bleed protection.
    #[test]
    fn test_synthesis_cross_session_skipped() {
        let mut net = AssociativeNetwork::new();
        let a = add_node(&mut net, "CrossA");
        let b = add_node(&mut net, "CrossMid");
        let c = add_node(&mut net, "CrossC");

        // Build A->B: initial Baseline(0.0), then strong Session(1) keyframe
        // that becomes the LATEST. latest.source == Session(1).
        let mut ab = TemporalBond::new(a, b, BondType::RelatedTo, 0.0, 0);
        ab.add_keyframe(100, 0.8, KeyframeSource::Session(1));
        net.add_bond(ab).unwrap();

        // B->C: latest is Session(2) — different session.
        let mut bc = TemporalBond::new(b, c, BondType::RelatedTo, 0.0, 0);
        bc.add_keyframe(100, 0.8, KeyframeSource::Session(2));
        net.add_bond(bc).unwrap();

        net.add_bond(TemporalBond::new(a, c, BondType::RelatedTo, 0.1, 0)).unwrap();

        let mut engine = ConsolidationEngine::with_config(synthesis_only_config());
        let stats = engine.consolidate(&mut net, 10_000);

        assert_eq!(stats.triangles_examined, 1);
        assert_eq!(stats.triangles_proposed, 0);
        assert_eq!(stats.triangles_skipped_hub, 0);
        assert_eq!(stats.triangles_skipped_threshold, 1);

        let ac = net.get_bond(&a, &c).unwrap();
        let has_synth = ac.keyframes().iter().any(|k| {
            matches!(k.source, KeyframeSource::Synthesis { .. })
        });
        assert!(!has_synth, "cross-session legs must NOT synthesize");
    }

    /// (d) Idempotency: running synthesis twice doesn't stack keyframes,
    ///     because after the first pass A->C is above the upper bound.
    #[test]
    fn test_synthesis_is_idempotent() {
        let mut net = AssociativeNetwork::new();
        let a = add_node(&mut net, "IdemA");
        let b = add_node(&mut net, "IdemMid");
        let c = add_node(&mut net, "IdemC");

        net.add_bond(TemporalBond::new(a, b, BondType::RelatedTo, 0.8, 0)).unwrap();
        net.add_bond(TemporalBond::new(b, c, BondType::RelatedTo, 0.8, 0)).unwrap();
        net.add_bond(TemporalBond::new(a, c, BondType::RelatedTo, 0.1, 0)).unwrap();

        let mut engine = ConsolidationEngine::with_config(synthesis_only_config());
        let s1 = engine.consolidate(&mut net, 10_000);
        let s2 = engine.consolidate(&mut net, 20_000);

        assert_eq!(s1.triangles_proposed, 1);
        // Second run: A->C latest raw = 0.56 >= 0.3 upper bound, so skipped.
        assert_eq!(s2.triangles_proposed, 0);
        assert_eq!(s2.triangles_skipped_threshold, 1);

        let ac = net.get_bond(&a, &c).unwrap();
        let synth_count = ac.keyframes().iter().filter(|k| {
            matches!(k.source, KeyframeSource::Synthesis { .. })
        }).count();
        assert_eq!(synth_count, 1, "second pass must not add another Synthesis keyframe");
    }

    /// (1a) Variant prune phase: variant-source keyframes older than
    /// `variant_prune_halflife_ms` are dropped during consolidation.
    /// Non-variant keyframes are untouched.
    #[test]
    fn test_consolidation_prunes_old_variant_keyframes() {
        let mut net = AssociativeNetwork::new();
        let a = add_node(&mut net, "VarPruneA");
        let b = add_node(&mut net, "VarPruneB");

        // Bond carries: a Baseline keyframe at t=0, an OLD Variant at
        // t=100 (will be dropped), a RECENT Variant at t=now-1000 (kept),
        // and a Session keyframe at t=now (kept).
        let now = 10 * 60 * 60 * 1000; // 10h in ms — well past the 6h halflife
        let mut bond = TemporalBond::new(a, b, BondType::RelatedTo, 0.5, 0);
        bond.add_keyframe(
            100,
            0.25,
            KeyframeSource::Variant { id: 7, goal: a, strategy: 0 },
        );
        bond.add_keyframe(
            now - 1000,
            0.25,
            KeyframeSource::Variant { id: 8, goal: a, strategy: 1 },
        );
        bond.add_keyframe(now, 0.7, KeyframeSource::Session(42));
        net.add_bond(bond).unwrap();

        let mut engine = ConsolidationEngine::with_config(ConsolidationConfig {
            promotion_threshold: 1_000_000, // disable promote
            prune_threshold: 0.0,           // disable main prune
            max_keyframe_age_ms: i64::MAX / 2,
            synthesis_enabled: false,       // disable synthesis
            variant_prune_halflife_ms: 6 * 60 * 60 * 1000, // 6h
            ..Default::default()
        });

        let stats = engine.consolidate(&mut net, now);
        assert_eq!(
            stats.variant_keyframes_pruned, 1,
            "exactly one variant kf (the t=100 one) must be pruned"
        );

        let bond = net.get_bond(&a, &b).unwrap();
        // Surviving sources: original Baseline, recent Variant, Session.
        let survived: Vec<&KeyframeSource> =
            bond.keyframes().iter().map(|k| &k.source).collect();
        assert!(survived.iter().any(|s| matches!(s, KeyframeSource::Baseline)));
        assert!(survived.iter().any(|s| matches!(s, KeyframeSource::Session(42))));
        // The recent Variant (id=8) must still be there.
        assert!(survived.iter().any(|s| matches!(s,
            KeyframeSource::Variant { id: 8, .. }
        )));
        // The old Variant (id=7) must be gone.
        assert!(!survived.iter().any(|s| matches!(s,
            KeyframeSource::Variant { id: 7, .. }
        )));
    }

    /// (1b) Synthesis triangles must fire on bonds whose latest keyframe
    /// is `PromotedFromVariant`. Without source-class compatibility the
    /// post-arena reasoning chain dead-ends after promotion.
    #[test]
    fn test_synthesis_triangles_fire_on_promoted_variant_bonds() {
        let mut net = AssociativeNetwork::new();
        let a = add_node(&mut net, "PvA");
        let b = add_node(&mut net, "PvMid");
        let c = add_node(&mut net, "PvC");

        // A->B: a strong PromotedFromVariant keyframe is the latest.
        let mut ab = TemporalBond::new(a, b, BondType::RelatedTo, 0.0, 0);
        ab.add_keyframe(
            100,
            0.8,
            KeyframeSource::PromotedFromVariant {
                variant_id: 99,
                original_strategy: 1,
            },
        );
        net.add_bond(ab).unwrap();

        // B->C: a strong Baseline keyframe is the latest.
        let mut bc = TemporalBond::new(b, c, BondType::RelatedTo, 0.0, 0);
        bc.add_keyframe(100, 0.8, KeyframeSource::Baseline);
        net.add_bond(bc).unwrap();

        // A->C exists at low weight (below synthesis_max_target_weight).
        net.add_bond(TemporalBond::new(a, c, BondType::RelatedTo, 0.1, 0))
            .unwrap();

        let mut engine = ConsolidationEngine::with_config(synthesis_only_config());
        let stats = engine.consolidate(&mut net, 10_000);

        assert_eq!(stats.triangles_examined, 1);
        assert_eq!(
            stats.triangles_proposed, 1,
            "PromotedFromVariant must be Baseline-compatible for synthesis"
        );

        let ac = net.get_bond(&a, &c).unwrap();
        let has_synth = ac.keyframes().iter().any(|k| {
            matches!(k.source, KeyframeSource::Synthesis { .. })
        });
        assert!(has_synth, "A->C must carry a Synthesis keyframe");
    }

    /// (Multi-agent) Agent-source keyframes older than
    /// `agent_prune_halflife_ms` are dropped during consolidation.
    /// Non-agent keyframes are untouched. Mirrors the variant prune test.
    #[test]
    fn test_consolidation_prunes_old_agent_keyframes() {
        let mut net = AssociativeNetwork::new();
        let a = add_node(&mut net, "AgPruneA");
        let b = add_node(&mut net, "AgPruneB");

        let now = 10 * 60 * 60 * 1000; // 10h — past the 6h halflife
        let mut bond = TemporalBond::new(a, b, BondType::RelatedTo, 0.5, 0);
        // Old Agent kf at t=100 — should be pruned.
        bond.add_keyframe(
            100, 0.20,
            KeyframeSource::Agent { role: 1, multi_think_id: 7, step: 0 },
        );
        // Recent Agent kf — kept.
        bond.add_keyframe(
            now - 1000, 0.20,
            KeyframeSource::Agent { role: 2, multi_think_id: 8, step: 0 },
        );
        // Session kf — kept.
        bond.add_keyframe(now, 0.7, KeyframeSource::Session(42));
        net.add_bond(bond).unwrap();

        let mut engine = ConsolidationEngine::with_config(ConsolidationConfig {
            promotion_threshold: 1_000_000,
            prune_threshold: 0.0,
            max_keyframe_age_ms: i64::MAX / 2,
            synthesis_enabled: false,
            variant_prune_halflife_ms: 0,                  // skip variant prune
            agent_prune_halflife_ms: 6 * 60 * 60 * 1000,   // 6h
            ..Default::default()
        });
        let stats = engine.consolidate(&mut net, now);
        assert_eq!(
            stats.agent_keyframes_pruned, 1,
            "exactly one agent kf (the t=100 one) must be pruned"
        );

        let bond = net.get_bond(&a, &b).unwrap();
        let survived: Vec<&KeyframeSource> =
            bond.keyframes().iter().map(|k| &k.source).collect();
        // Survivors must include the recent Agent and the Session kfs;
        // the old Agent (mt=7) must be gone.
        assert!(survived.iter().any(|s| matches!(s,
            KeyframeSource::Agent { multi_think_id: 8, .. }
        )));
        assert!(!survived.iter().any(|s| matches!(s,
            KeyframeSource::Agent { multi_think_id: 7, .. }
        )));
        assert!(survived.iter().any(|s| matches!(s, KeyframeSource::Session(42))));
    }

    /// Consolidation should recompute `degree_p90` and surface the new
    /// value via `ConsolidationStats.degree_p90_after_consolidation`.
    /// Adding bonds that shift the degree distribution must change the
    /// reported p90.
    #[test]
    fn test_consolidation_recomputes_degree_p90() {
        let mut net = AssociativeNetwork::new();
        // Build a small star: one hub + 12 leaves, hub connected to
        // each leaf. Initial degree distribution is [1; 12] + [12].
        let hub = add_node(&mut net, "DegHub");
        let mut leaves = Vec::new();
        for i in 0..12 {
            let s = add_node(&mut net, &format!("DegLeaf{}", i));
            leaves.push(s);
            net.add_bond(TemporalBond::new(
                hub, s, BondType::RelatedTo, 0.5, 0,
            )).unwrap();
        }
        let initial = net.recompute_degree_p90();

        // Add bonds between every pair of leaves to densify the graph.
        // Now most leaves carry degree close to the hub's, so the p90
        // should rise above `initial`.
        for i in 0..leaves.len() {
            for j in (i + 1)..leaves.len() {
                let _ = net.add_bond(TemporalBond::new(
                    leaves[i], leaves[j], BondType::RelatedTo, 0.5, 0,
                ));
            }
        }

        let mut engine = ConsolidationEngine::with_config(ConsolidationConfig {
            promotion_threshold: 1_000_000, // disable promote
            prune_threshold: 0.0,
            max_keyframe_age_ms: i64::MAX / 2,
            synthesis_enabled: false,
            variant_prune_halflife_ms: 0,
            agent_prune_halflife_ms: 0,
            ..Default::default()
        });
        let stats = engine.consolidate(&mut net, 10_000);

        // The stats field must be populated (non-zero on a non-empty
        // graph) and must equal what the network now reports.
        assert!(stats.degree_p90_after_consolidation >= 1);
        assert_eq!(
            stats.degree_p90_after_consolidation as usize,
            net.degree_p90(),
            "stats must reflect the post-consolidation cached p90"
        );
        assert!(
            stats.degree_p90_after_consolidation as usize > initial,
            "densifying the graph must shift the p90 upward (initial={}, after={})",
            initial, stats.degree_p90_after_consolidation,
        );
    }


    // ----- Port #13: synthesis cap tests -----

    /// Build a small graph with many triangles between every pair so the
    /// triangle counter has plenty of work to cap. Returns the engine
    /// configured with the requested caps.
    fn cap_graph(n: usize) -> AssociativeNetwork {
        let mut net = AssociativeNetwork::new();
        let mut nodes = Vec::with_capacity(n);
        for i in 0..n {
            nodes.push(add_node(&mut net, &format!("CapNode{:03}", i)));
        }
        // Fully-connect (directed): every i -> j (i != j). That gives
        // n*(n-1) bonds and n*(n-1)*(n-2) potential triangle examinations
        // before dedupe -- ample room to exercise any small cap.
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                let _ = net.add_bond(TemporalBond::new(
                    nodes[i], nodes[j], BondType::RelatedTo, 0.8, 0,
                ));
            }
        }
        net
    }

    /// (Port #13 / cap-1) `max_triangles_per_pass` is honoured: synthesize
    /// stops once the cap is reached and `synthesis_truncated` flips true.
    #[test]
    fn test_synthesize_respects_max_triangles_cap() {
        let mut net = cap_graph(8); // 8*7*6 = 336 potential triangles
        let mut engine = ConsolidationEngine::with_config(ConsolidationConfig {
            promotion_threshold: 1_000_000,
            prune_threshold: 0.0,
            max_keyframe_age_ms: i64::MAX / 2,
            synthesis_enabled: true,
            synthesis_min_leg_weight: 0.4,
            synthesis_max_target_weight: 0.3,
            synthesis_path_decay: 0.7,
            variant_prune_halflife_ms: 0,
            agent_prune_halflife_ms: 0,
            max_triangles_per_pass: 10,
            max_proposals_per_pass: 1_000_000,
            rss_oom_guard_bytes: 0,
        });
        let stats = engine.consolidate(&mut net, 10_000);

        // Examined exactly the cap -- not 9, not 11. The check fires *before*
        // incrementing the counter, so the counter never crosses the cap.
        assert_eq!(
            stats.triangles_examined, 10,
            "triangle cap must bound examined count exactly"
        );
        assert!(
            stats.synthesis_truncated,
            "synthesis_truncated must flip when the cap fires"
        );
        assert!(
            !stats.synthesis_truncated_oom_guard,
            "oom guard must not fire when only the count cap was hit"
        );
    }

    /// Build a 3-layer A->B->C graph with strong legs but NO direct A->C
    /// bond, so every triangle is a fresh proposal candidate. n_per_layer=k
    /// gives k*k*k triangles (de-duped to k*k unique (A,C) proposals).
    fn layered_proposal_graph(n_per_layer: usize) -> AssociativeNetwork {
        let mut net = AssociativeNetwork::new();
        let mut a_nodes = Vec::with_capacity(n_per_layer);
        let mut b_nodes = Vec::with_capacity(n_per_layer);
        let mut c_nodes = Vec::with_capacity(n_per_layer);
        for i in 0..n_per_layer {
            a_nodes.push(add_node(&mut net, &format!("PropA{:02}", i)));
            b_nodes.push(add_node(&mut net, &format!("PropB{:02}", i)));
            c_nodes.push(add_node(&mut net, &format!("PropC{:02}", i)));
        }
        // Strong A->B bonds (every A to every B) and B->C bonds.
        for &a in &a_nodes {
            for &b in &b_nodes {
                let _ = net.add_bond(TemporalBond::new(
                    a, b, BondType::RelatedTo, 0.8, 0,
                ));
            }
        }
        for &b in &b_nodes {
            for &c in &c_nodes {
                let _ = net.add_bond(TemporalBond::new(
                    b, c, BondType::RelatedTo, 0.8, 0,
                ));
            }
        }
        // Deliberately NO A->C bonds -- those are the proposals we want.
        net
    }

    /// (Port #13 / cap-2) `max_proposals_per_pass` is honoured: at most N
    /// proposals are kept and the dropped counter increments for the rest.
    #[test]
    fn test_synthesize_respects_max_proposals_cap() {
        // 4*4*4 = 64 triangle examinations, 4*4 = 16 unique (A,C) proposals
        // before the cap clamps it to 2.
        let mut net = layered_proposal_graph(4);
        let mut engine = ConsolidationEngine::with_config(ConsolidationConfig {
            promotion_threshold: 1_000_000,
            prune_threshold: 0.0,
            max_keyframe_age_ms: i64::MAX / 2,
            synthesis_enabled: true,
            synthesis_min_leg_weight: 0.4,
            synthesis_max_target_weight: 0.3,
            synthesis_path_decay: 0.7,
            variant_prune_halflife_ms: 0,
            agent_prune_halflife_ms: 0,
            // Generous triangle cap so we run to completion of the pass.
            max_triangles_per_pass: 1_000_000,
            max_proposals_per_pass: 2,
            rss_oom_guard_bytes: 0,
        });
        let stats = engine.consolidate(&mut net, 10_000);

        // Exactly 2 proposals applied; everything past that bumps the dropped
        // counter rather than allocating.
        assert_eq!(
            stats.triangles_proposed, 2,
            "proposal cap must bound applied proposals"
        );
        assert!(
            stats.synthesis_proposals_dropped_for_cap > 0,
            "dropped counter must increment when proposal cap is hit"
        );
        // Examination keeps going past the proposal cap so triangle stats
        // stay honest -- assert that explicitly.
        assert!(
            stats.triangles_examined > stats.triangles_proposed,
            "examination must continue past the proposal cap (examined={}, proposed={})",
            stats.triangles_examined, stats.triangles_proposed,
        );
    }

    /// (Port #13 / cap-3) Successive consolidate calls walk a different
    /// slice of the graph -- the rotating cursor must NOT re-examine the
    /// same starting A on the second pass.
    #[test]
    fn test_synthesize_continues_from_different_start() {
        // 12 fully-connected nodes -- 12*11*10 = 1320 triangles, plenty of
        // headroom for a cap of 30 to hit before exhausting any one A.
        let mut net = cap_graph(12);
        let cfg = ConsolidationConfig {
            promotion_threshold: 1_000_000,
            prune_threshold: 0.0,
            max_keyframe_age_ms: i64::MAX / 2,
            synthesis_enabled: true,
            synthesis_min_leg_weight: 0.4,
            synthesis_max_target_weight: 0.3,
            synthesis_path_decay: 0.7,
            variant_prune_halflife_ms: 0,
            agent_prune_halflife_ms: 0,
            max_triangles_per_pass: 30,
            // High proposal cap so we don't accidentally clamp by proposals
            // and miss the cursor-advance signal.
            max_proposals_per_pass: 1_000_000,
            rss_oom_guard_bytes: 0,
        };
        let mut engine = ConsolidationEngine::with_config(cfg);

        let cursor_before = engine.synthesis_seed_cursor;
        let stats1 = engine.consolidate(&mut net, 10_000);
        let cursor_mid = engine.synthesis_seed_cursor;

        assert!(
            stats1.synthesis_truncated,
            "first pass must hit the cap on this graph"
        );
        assert_ne!(
            cursor_before, cursor_mid,
            "cursor must advance after a capped pass (before={}, after={})",
            cursor_before, cursor_mid,
        );

        let stats2 = engine.consolidate(&mut net, 20_000);
        let cursor_after = engine.synthesis_seed_cursor;
        assert_ne!(
            cursor_mid, cursor_after,
            "cursor must advance again on the second pass"
        );
        // Both passes did real work -- the second pass isn't a no-op.
        assert!(
            stats2.triangles_examined > 0,
            "second pass must examine triangles from its new starting A"
        );
    }
}
