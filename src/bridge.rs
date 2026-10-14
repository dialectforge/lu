//! Cross-domain bridge detection — second of the seven Lu 4.0 capability ports.
//!
//! When spread activation crosses a *weak* bond between two *hub* nodes
//! belonging to *otherwise-disjoint clusters*, that's a structurally
//! meaningful weak link bridging distant domains. We notice these — we
//! don't act on them yet. Acting on bridges (problem-space generation,
//! invention spawning) belongs to later capability ports.
//!
//! Two-phase design (constraint: keep the spread hot path cheap):
//!
//! 1. Phase 1 — INLINE in spread (`network.rs`): record every bond
//!    crossing whose weight is below `weak_bond_threshold` AND whose
//!    endpoints are both hubs. Only cheap predicates run inline (a
//!    HashMap-backed degree lookup + a stopword check). No Jaccard,
//!    no cluster analysis here.
//! 2. Phase 2 — POST-spread (`classify_candidates`): for each candidate,
//!    compute Jaccard neighbor-similarity. Only candidates whose hubs
//!    have disjoint neighborhoods (Jaccard < `cluster_disjoint_threshold`)
//!    survive as `BridgeEvent`s.
//!
//! Storage: bridges are NOT persisted in the brain file. They live in
//! `Brain.recent_bridges` as a runtime-only ring buffer, surfaced via
//! `lu4d_status`.

use crate::network::{is_hub_stopword, AssociativeNetwork, DEGREE_NORM_THRESHOLD};
use crate::scope::TemporalScope;
use crate::seed::Seed;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Defaults for bridge detection — tuned conservatively (false-positive
/// averse) so surfaced bridges are likely to be real cross-domain links.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgeConfig {
    /// Master switch. When false, spread skips candidate recording and
    /// the classifier returns empty. `dropped_overflow` stays at 0.
    pub bridges_enabled: bool,
    /// Bond weights (post scope evaluation, post bond-type base multiplier)
    /// below this count as "weak" — the kind of link a frontier crosses
    /// without it being the dominant force in recall.
    pub weak_bond_threshold: f32,
    /// Two hubs are "in disjoint clusters" iff
    /// `Jaccard(neighbors_A, neighbors_B) < cluster_disjoint_threshold`.
    /// 0.15 → clusters share at most ~15% of neighbors.
    pub cluster_disjoint_threshold: f32,
    /// Per-spread cap on the candidate buffer. When full, `record()`
    /// increments `dropped_overflow` instead of allocating more.
    pub candidate_buffer_cap: usize,
    /// Ring-buffer cap for `Brain.recent_bridges`.
    pub recent_bridges_cap: usize,
    /// Degree above which a node counts as a hub. Default matches
    /// `network::DEGREE_NORM_THRESHOLD` (2000) so production behavior is
    /// identical to the rest of the codebase. Exposed primarily so unit
    /// tests can use small graphs. (Deviation from the original 5-knob
    /// spec — flagged in port notes.)
    pub hub_degree_threshold: usize,
}

impl Default for BridgeConfig {
    fn default() -> Self {
        BridgeConfig {
            bridges_enabled: true,
            weak_bond_threshold: 0.25,
            cluster_disjoint_threshold: 0.15,
            candidate_buffer_cap: 1024,
            recent_bridges_cap: 256,
            hub_degree_threshold: DEGREE_NORM_THRESHOLD,
        }
    }
}

/// Phase-1 output: a single (low, high) pair the spread frontier crossed
/// where the bond was weak and both endpoints were hubs. Internal — not
/// serialized; lives only between spread and `classify_candidates`.
#[derive(Debug, Clone)]
pub struct BridgeCandidate {
    /// Lower-ordered seed (canonical pair).
    pub low: Seed,
    /// Higher-ordered seed (canonical pair).
    pub high: Seed,
    /// Bond weight at crossing time (post scope evaluation).
    pub bond_weight: f32,
    /// Epoch millis when the crossing happened.
    pub timestamp_ms: i64,
}

/// Phase-2 output: a confirmed bridge — a structurally meaningful weak
/// link bridging two distant clusters. Surfaced via
/// `Brain.recent_bridges` and `lu4d_status`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgeEvent {
    /// Lower seed (canonical pair).
    pub low: Seed,
    /// Higher seed (canonical pair).
    pub high: Seed,
    /// `low`'s content label, snapshotted at detection time.
    pub low_content: String,
    /// `high`'s content label.
    pub high_content: String,
    /// Bond weight when the bridge was detected.
    pub bond_weight: f32,
    /// Jaccard similarity of the two hubs' neighbor sets (each excluded
    /// from the other's set). Always `< cluster_disjoint_threshold`.
    pub jaccard: f32,
    /// Scope active at detection time. Lets a session bridge be told
    /// apart from a baseline bridge in the surfaced output.
    pub scope: TemporalScope,
    /// Epoch millis when detected.
    pub timestamp_ms: i64,
}

/// Buffer + bookkeeping for a single spread call's bridge recording pass.
/// Created fresh per spread, consumed by `classify_candidates`.
#[derive(Debug)]
pub struct BridgeCollector {
    /// Recorded candidates, deduped by canonical pair.
    pub candidates: Vec<BridgeCandidate>,
    /// Set of (low, high) pairs already recorded. Prevents duplicates
    /// when the same bond is re-evaluated via multiple spread paths.
    pub seen: HashSet<(Seed, Seed)>,
    /// Number of would-be candidates dropped because the cap was hit.
    pub dropped_overflow: usize,
    /// Cap value (mirror of `BridgeConfig::candidate_buffer_cap`).
    pub cap: usize,
}

impl BridgeCollector {
    /// Empty collector with the given cap.
    pub fn new(cap: usize) -> Self {
        BridgeCollector {
            candidates: Vec::new(),
            seen: HashSet::new(),
            dropped_overflow: 0,
            cap,
        }
    }

    /// Record a (a, b) crossing. Canonicalises the pair, dedupes via
    /// `seen`, respects the cap (incrementing `dropped_overflow` instead
    /// of growing on overflow).
    #[inline]
    pub fn record(&mut self, a: Seed, b: Seed, bond_weight: f32, timestamp_ms: i64) {
        let (low, high) = canonical_pair(a, b);
        if !self.seen.insert((low, high)) {
            return; // already recorded — not an overflow
        }
        if self.candidates.len() >= self.cap {
            self.dropped_overflow += 1;
            return;
        }
        self.candidates.push(BridgeCandidate {
            low,
            high,
            bond_weight,
            timestamp_ms,
        });
    }
}

/// Canonical (low, high) pair ordering (compares raw seed bytes). Lets
/// `(a,b)` and `(b,a)` collapse to the same dedupe key.
#[inline]
pub fn canonical_pair(a: Seed, b: Seed) -> (Seed, Seed) {
    if a.as_bytes() <= b.as_bytes() {
        (a, b)
    } else {
        (b, a)
    }
}

/// Hub criterion: stopword content OR degree above the configured
/// threshold. Mirrors `network.rs`'s spread-suppression rule so bridges
/// are detected only between the same kinds of nodes the rest of the
/// system already treats as hubs.
#[inline]
pub fn is_hub(network: &AssociativeNetwork, seed: &Seed, hub_degree_threshold: usize) -> bool {
    let view = network.get_node(seed);
    let cont = view.as_ref().map(|n| n.content()).unwrap_or("");
    if is_hub_stopword(cont) {
        return true;
    }
    let deg = network.seam_total_degree(seed);
    deg > hub_degree_threshold
}

/// Cheap fast-path predicate used inline in spread: is this bond a
/// plausible bridge candidate? Real Jaccard happens later in
/// `classify_candidates`.
#[inline]
pub fn is_candidate(
    network: &AssociativeNetwork,
    a: Seed,
    b: Seed,
    bond_weight: f32,
    cfg: &BridgeConfig,
) -> bool {
    if !cfg.bridges_enabled {
        return false;
    }
    if bond_weight.abs() >= cfg.weak_bond_threshold {
        return false;
    }
    is_hub(network, &a, cfg.hub_degree_threshold)
        && is_hub(network, &b, cfg.hub_degree_threshold)
}

/// OP-SPREAD-1 -- [`is_hub`] with the degree supplied by the caller.
///
/// Byte-identical to `is_hub`; the only difference is that the caller
/// hands over a `seam_total_degree` it has already paid for. `is_hub`
/// re-probed it, and the BFS edge loop called `is_hub` twice per bond,
/// so a single bond cost four `NodeIntern` lock-and-hash lookups on top
/// of the two the spread already did for itself.
#[inline]
pub fn is_hub_with_degree(
    network: &AssociativeNetwork,
    seed: &Seed,
    degree: usize,
    hub_degree_threshold: usize,
) -> bool {
    let view = network.get_node(seed);
    let cont = view.as_ref().map(|n| n.content()).unwrap_or("");
    if is_hub_stopword(cont) {
        return true;
    }
    degree > hub_degree_threshold
}

/// OP-SPREAD-1 -- [`is_candidate`] with both endpoints' expensive parts
/// supplied by the caller: `a_is_hub` hoisted out of the edge loop (the
/// frontier node does not change across it) and `b_degree` taken from
/// the single per-edge seam probe. Evaluation order and short-circuit
/// behaviour are identical to `is_candidate`.
#[inline]
pub fn is_candidate_with_hubs(
    network: &AssociativeNetwork,
    a_is_hub: bool,
    b: &Seed,
    b_degree: usize,
    bond_weight: f32,
    cfg: &BridgeConfig,
) -> bool {
    if !cfg.bridges_enabled {
        return false;
    }
    if bond_weight.abs() >= cfg.weak_bond_threshold {
        return false;
    }
    a_is_hub && is_hub_with_degree(network, b, b_degree, cfg.hub_degree_threshold)
}

/// Phase-2 classifier. For each candidate, compute the Jaccard similarity
/// of its endpoints' neighbor sets and drop those whose clusters overlap
/// too much. Surviving candidates become `BridgeEvent`s tagged with the
/// supplied scope.
pub fn classify_candidates(
    network: &AssociativeNetwork,
    candidates: &[BridgeCandidate],
    scope: &TemporalScope,
    cfg: &BridgeConfig,
) -> Vec<BridgeEvent> {
    if !cfg.bridges_enabled {
        return Vec::new();
    }
    let mut events = Vec::with_capacity(candidates.len());
    for c in candidates {
        let j = jaccard_neighbors(network, &c.low, &c.high);
        if j >= cfg.cluster_disjoint_threshold {
            continue;
        }
        // OP-LABEL-4 -- THE SECOND BLANK-LABEL SITE.
        //
        // `get_node` is the RESIDENT map. On a cache-backed v5 brain it
        // is empty (read-only reader) or holds content-free rehydration
        // stubs (writer), so both endpoints resolved to the empty
        // string and the serendipity line rendered as
        //   "a connection between  and : they sit in different
        //    clusters but a weak link bridges them."
        // -- grammatical, confident, and missing both nouns. Same root
        // cause as OP-LABEL-1 internal notes at a call site that
        // fix did not reach: the seam is the PRESENCE oracle, the node
        // store is the LABEL oracle.
        //
        // Cost is bounded and in practice near-zero: candidates are
        // capped at `candidate_buffer_cap` (1024) and every endpoint is
        // a node the spread just activated, so it is already in the
        // per-network label LRU. An endpoint with genuinely no stored
        // text still yields "", and `narration` now SKIPS the slot
        // rather than rendering a hole (never a placeholder -- that is
        // the OP-LABEL-1 rule).
        let low_content = network.node_label_or_empty(&c.low);
        let high_content = network.node_label_or_empty(&c.high);
        events.push(BridgeEvent {
            low: c.low,
            high: c.high,
            low_content,
            high_content,
            bond_weight: c.bond_weight,
            jaccard: j,
            scope: scope.clone(),
            timestamp_ms: c.timestamp_ms,
        });
    }
    events
}

/// Jaccard similarity of the two seeds' combined (out + in) neighbor
/// sets. Each endpoint is excluded from the OTHER's neighbor set so the
/// bridge bond itself doesn't artificially raise the similarity. Returns
/// 0 when both sets end up empty.
fn jaccard_neighbors(network: &AssociativeNetwork, a: &Seed, b: &Seed) -> f32 {
    let neigh = |s: &Seed| -> HashSet<Seed> {
        let mut set: HashSet<Seed> =
            network.seam_outgoing_set(s).into_iter().collect();
        for n in network.seam_incoming_set(s) {
            set.insert(n);
        }
        set
    };
    let mut na = neigh(a);
    let mut nb = neigh(b);
    na.remove(b);
    nb.remove(a);
    if na.is_empty() && nb.is_empty() {
        return 0.0;
    }
    let inter = na.intersection(&nb).count() as f32;
    let union = na.union(&nb).count() as f32;
    if union == 0.0 {
        0.0
    } else {
        inter / union
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scar::ScarLog;
    use crate::bond::{BondType, TemporalBond};
    use crate::node::{Confidence, KnowledgeContent, KnowledgeNode};

    /// Helper: add a Raw-content node with the given label.
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

    /// A small `BridgeConfig` for unit tests — uses a tiny hub threshold
    /// so we can build hub-degree clusters with a handful of nodes.
    fn small_cfg() -> BridgeConfig {
        BridgeConfig {
            bridges_enabled: true,
            weak_bond_threshold: 0.25,
            cluster_disjoint_threshold: 0.15,
            candidate_buffer_cap: 1024,
            recent_bridges_cap: 256,
            hub_degree_threshold: 3, // > 3 → hub for tests
        }
    }

    #[test]
    fn test_canonical_pair_ordering() {
        let a = Seed::from_label("alpha");
        let b = Seed::from_label("beta");
        let (l1, h1) = canonical_pair(a, b);
        let (l2, h2) = canonical_pair(b, a);
        assert_eq!(l1, l2);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_collector_dedupes_repeated_pairs() {
        let mut bc = BridgeCollector::new(10);
        let a = Seed::from_label("x");
        let b = Seed::from_label("y");
        bc.record(a, b, 0.1, 0);
        bc.record(b, a, 0.1, 0); // same pair, reversed
        bc.record(a, b, 0.1, 0);
        assert_eq!(bc.candidates.len(), 1);
        assert_eq!(bc.dropped_overflow, 0);
    }

    /// Stretch test (per port spec): cap=8, generate 20 unique pairs.
    /// 12 should land in `dropped_overflow`.
    #[test]
    fn test_candidate_buffer_overflow_increments_counter() {
        let mut bc = BridgeCollector::new(8);
        for i in 0..20 {
            let a = Seed::from_label(&format!("a{}", i));
            let b = Seed::from_label(&format!("b{}", i));
            bc.record(a, b, 0.1, 0);
        }
        assert_eq!(bc.candidates.len(), 8);
        assert_eq!(bc.dropped_overflow, 12);
    }

    #[test]
    fn test_jaccard_disjoint_neighbors() {
        let mut net = AssociativeNetwork::new();
        let h1 = add_node(&mut net, "hub1");
        let h2 = add_node(&mut net, "hub2");
        let n1 = add_node(&mut net, "n1");
        let n2 = add_node(&mut net, "n2");
        net.add_bond(TemporalBond::new(h1, n1, BondType::RelatedTo, 0.9, 0))
            .unwrap();
        net.add_bond(TemporalBond::new(h2, n2, BondType::RelatedTo, 0.9, 0))
            .unwrap();
        // Disjoint neighborhoods → Jaccard = 0.
        let j = jaccard_neighbors(&net, &h1, &h2);
        assert!(j < 0.001);
    }

    #[test]
    fn test_jaccard_shared_neighbors() {
        let mut net = AssociativeNetwork::new();
        let h1 = add_node(&mut net, "hub1");
        let h2 = add_node(&mut net, "hub2");
        let shared = add_node(&mut net, "shared");
        net.add_bond(TemporalBond::new(h1, shared, BondType::RelatedTo, 0.9, 0))
            .unwrap();
        net.add_bond(TemporalBond::new(h2, shared, BondType::RelatedTo, 0.9, 0))
            .unwrap();
        // Both share one neighbor and have no others; Jaccard = 1/1 = 1.0.
        let j = jaccard_neighbors(&net, &h1, &h2);
        assert!((j - 1.0).abs() < 1e-5);
    }

    /// Build two clusters of hubs joined by a single bond. Returns
    /// `(hub_a, hub_b, a_leaves, b_leaves)`.
    fn build_two_clusters(
        net: &mut AssociativeNetwork,
        bridge_weight: f32,
    ) -> (Seed, Seed, Vec<Seed>, Vec<Seed>) {
        // Cluster A: hub_a + 5 unique neighbors → degree 5 > 3 → hub.
        let hub_a = add_node(net, "hub_a");
        let mut a_leaves = Vec::new();
        for i in 0..5 {
            let n = add_node(net, &format!("a_leaf_{}", i));
            net.add_bond(TemporalBond::new(hub_a, n, BondType::RelatedTo, 0.9, 0))
                .unwrap();
            a_leaves.push(n);
        }
        // Cluster B: hub_b + 5 unique neighbors → degree 5 > 3 → hub.
        let hub_b = add_node(net, "hub_b");
        let mut b_leaves = Vec::new();
        for i in 0..5 {
            let n = add_node(net, &format!("b_leaf_{}", i));
            net.add_bond(TemporalBond::new(hub_b, n, BondType::RelatedTo, 0.9, 0))
                .unwrap();
            b_leaves.push(n);
        }
        // Bridge bond.
        net.add_bond(TemporalBond::new(
            hub_a,
            hub_b,
            BondType::RelatedTo,
            bridge_weight,
            0,
        ))
        .unwrap();
        (hub_a, hub_b, a_leaves, b_leaves)
    }

    /// (1) Two disjoint hub clusters joined by ONE weak bond → 1 BridgeEvent.
    #[test]
    fn test_bridge_detected_for_weak_hub_to_hub() {
        let mut net = AssociativeNetwork::new();
        // raw 0.1 × RelatedTo base 0.8 = 0.08 effective → weak (< 0.25).
        let (hub_a, hub_b, a_leaves, _b_leaves) = build_two_clusters(&mut net, 0.1);
        let cfg = small_cfg();
        let scope = TemporalScope::Current;

        let mut bc = BridgeCollector::new(cfg.candidate_buffer_cap);
        // Seed at one of cluster A's leaves so spread flows: leaf -> hub_a -> hub_b -> ...
        let _ = net.spread_activation_weighted_with_bridges(
            &[(a_leaves[0], 1.0)],
            &scope,
            3,
            50,
            &mut bc,
            &cfg,
            1000,
            &ScarLog::default(),
        );
        let events = classify_candidates(&net, &bc.candidates, &scope, &cfg);

        assert_eq!(events.len(), 1, "exactly one bridge event expected");
        let e = &events[0];
        let (low, high) = canonical_pair(hub_a, hub_b);
        assert_eq!(e.low, low);
        assert_eq!(e.high, high);
        assert!(e.jaccard < cfg.cluster_disjoint_threshold);
        assert!(e.bond_weight.abs() < cfg.weak_bond_threshold);
    }

    /// (2) Same setup but the bridge bond is STRONG → no event.
    #[test]
    fn test_no_bridge_for_strong_bond() {
        let mut net = AssociativeNetwork::new();
        // raw 0.9 × RelatedTo 0.8 = 0.72 effective → strong (>= 0.25).
        let (_hub_a, _hub_b, a_leaves, _) = build_two_clusters(&mut net, 0.9);
        let cfg = small_cfg();
        let scope = TemporalScope::Current;

        let mut bc = BridgeCollector::new(cfg.candidate_buffer_cap);
        let _ = net.spread_activation_weighted_with_bridges(
            &[(a_leaves[0], 1.0)],
            &scope,
            3,
            50,
            &mut bc,
            &cfg,
            1000,
            &ScarLog::default(),
        );
        let events = classify_candidates(&net, &bc.candidates, &scope, &cfg);
        assert!(events.is_empty(), "strong bond must NOT produce a bridge");
    }

    /// (3) Weak bridge, but the two hubs share many neighbors → Jaccard
    ///     above threshold → no event (filtered in Phase 2).
    #[test]
    fn test_no_bridge_when_clusters_share_neighbors() {
        let mut net = AssociativeNetwork::new();
        let hub_a = add_node(&mut net, "shared_hub_a");
        let hub_b = add_node(&mut net, "shared_hub_b");
        // 4 shared neighbors → both hubs see all 4. Plus 1 unique each →
        // each hub has degree 5, > 3 threshold.
        let mut shared = Vec::new();
        for i in 0..4 {
            let n = add_node(&mut net, &format!("shared_n{}", i));
            net.add_bond(TemporalBond::new(hub_a, n, BondType::RelatedTo, 0.9, 0))
                .unwrap();
            net.add_bond(TemporalBond::new(hub_b, n, BondType::RelatedTo, 0.9, 0))
                .unwrap();
            shared.push(n);
        }
        let unique_a = add_node(&mut net, "unique_a");
        let unique_b = add_node(&mut net, "unique_b");
        net.add_bond(TemporalBond::new(hub_a, unique_a, BondType::RelatedTo, 0.9, 0))
            .unwrap();
        net.add_bond(TemporalBond::new(hub_b, unique_b, BondType::RelatedTo, 0.9, 0))
            .unwrap();
        // Weak bridge.
        net.add_bond(TemporalBond::new(hub_a, hub_b, BondType::RelatedTo, 0.1, 0))
            .unwrap();

        // Sanity: Jaccard = 4 shared / (4 + 2 unique) = 4/6 ≈ 0.67 > 0.15.
        let j = jaccard_neighbors(&net, &hub_a, &hub_b);
        assert!(j > 0.15, "test setup must produce overlap > threshold");

        let cfg = small_cfg();
        let scope = TemporalScope::Current;
        let mut bc = BridgeCollector::new(cfg.candidate_buffer_cap);
        let _ = net.spread_activation_weighted_with_bridges(
            &[(unique_a, 1.0)],
            &scope,
            3,
            50,
            &mut bc,
            &cfg,
            1000,
            &ScarLog::default(),
        );
        // Phase 1 will have recorded at least one candidate (weak hub-to-hub),
        // but Phase 2 must drop it because of the high Jaccard.
        let events = classify_candidates(&net, &bc.candidates, &scope, &cfg);
        assert!(
            events.is_empty(),
            "shared-neighbor hubs must NOT produce a bridge"
        );
    }

    /// (4) Bridge event carries the scope active at detection time.
    #[test]
    fn test_bridge_event_carries_scope() {
        let mut net = AssociativeNetwork::new();
        let (_hub_a, _hub_b, a_leaves, _) = build_two_clusters(&mut net, 0.1);
        let cfg = small_cfg();
        let scope = TemporalScope::Session {
            id: 42,
            start: 0,
        };

        let mut bc = BridgeCollector::new(cfg.candidate_buffer_cap);
        let _ = net.spread_activation_weighted_with_bridges(
            &[(a_leaves[0], 1.0)],
            &scope,
            3,
            50,
            &mut bc,
            &cfg,
            1000,
            &ScarLog::default(),
        );
        let events = classify_candidates(&net, &bc.candidates, &scope, &cfg);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].scope.session_id(), Some(42));
    }
}
