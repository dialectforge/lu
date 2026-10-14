//! Curiosity Engine — fourth of the seven Lu 4.0 capability ports.
//!
//! Detects "interesting unknowns" surfacing during spread activation:
//! dead-ends past a hub, isolated high-activation nodes, knowledge
//! deserts where the top result is weak, and hub boundaries the
//! frontier couldn't traverse. Each detection is a `CuriositySignal`,
//! ranked into a per-brain `WonderQueue` and folded into a
//! per-domain `InterestProfile`.
//!
//! Two-phase design (mirrors Bridge Detection):
//!
//! 1. Phase 1 — INLINE in spread (`network.rs`): record raw
//!    `CuriosityCandidate`s with cheap predicates only. No intensity
//!    math. Visited-set dedupe by `(SignalType, seed)`.
//! 2. Phase 2 — POST-spread (`classify_curiosity_candidates`): score
//!    intensity per signal type, drop signals below `intensity_floor`.
//!
//! Storage: signals NOT persisted in the graph. They live in
//! `Brain.wonder_queue` (capped binary heap) and the per-call
//! `AskResult.curiosity_signals` list. Same operational-memory boundary
//! as `ThoughtTrace` and `BridgeEvent`.

use crate::network::{is_hub_stopword, AssociativeNetwork, DEGREE_NORM_THRESHOLD};
use crate::scope::TemporalScope;
use crate::seed::Seed;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

/// Default half-life for InterestProfile keyframe decay (7 days, ms).
pub const DEFAULT_INTEREST_HALFLIFE_MS: u64 = 7 * 24 * 60 * 60 * 1000;

/// All knobs for the Curiosity Engine. Defaults match the port spec.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CuriosityConfig {
    /// Master switch. When false, spread skips candidate recording and
    /// classification returns empty.
    pub enabled: bool,
    /// Cap on `Brain.wonder_queue`. Lowest-intensity dropped on overflow.
    pub wonder_queue_cap: usize,
    /// Cap on per-domain entries in `Brain.interest_profile`. Oldest
    /// (smallest `last_touched_ms`) dropped on overflow.
    pub interest_profile_cap: usize,
    /// Minimum further activations a hub-frontier expansion must yield;
    /// below this counts as a DeadEnd.
    pub deadend_yield_threshold: usize,
    /// Degree below which a high-activation node counts as IsolatedNode.
    pub isolated_degree_threshold: usize,
    /// Top-activation level below which the query counts as a
    /// KnowledgeDesert (effective post-confidence/decay activation).
    pub knowledge_desert_threshold: f32,
    /// Half-life for InterestProfile keyframe decay (ms).
    pub interest_decay_halflife_ms: u64,
    /// Signals with intensity below this floor are dropped in Phase 2.
    pub intensity_floor: f32,
    /// Port #17 Fix 2: a curiosity seed that has surfaced as the
    /// `wondering_about` slot in narration this many times is
    /// treated as a "noise gap" — its signal still records
    /// in `wonder_queue` (so research can target it), but
    /// `extract_slots` skips it when picking the wondering modulation.
    pub gap_appearance_threshold: u32,

    // -- OP-CURIOSITY-1: what happens to a filled gap --------------
    //
    // Everything the fill half produces (hub-fill, triangle closure,
    // single-hop and chained research) lands in the PendingLog as
    // `PendingStatus::Pending`. Until now the ONLY way for a pending
    // entry to become a bond was `lu4d_approve_changes`, an admin MCP
    // tool. That is a real safety property and it is why the default
    // below is `false`: a process that edits its own long-term memory
    // unsupervised is a decision, not a default.
    //
    // But it also means an unattended Lu proposes into a void. These
    // knobs make that a CHOICE rather than a structural fact.
    /// Master switch for autonomous application of curiosity fills.
    /// `false` = every fill waits for `lu4d_approve_changes` (the
    /// historical, and still default, behaviour).
    pub auto_apply: bool,
    /// Confidence floor for autonomous application. Only consulted when
    /// `auto_apply` is true.
    ///
    /// Calibration: internal triangle-closure synthesis is hard-coded to
    /// 0.5 (`idle::loop_`, "path-decayed inference, moderate
    /// confidence"); research-derived candidates carry the extractor's
    /// own confidence. A floor of 0.75 therefore admits well-sourced
    /// research and excludes bare internal synthesis by construction --
    /// the graph does not get to promote its own guesses into facts
    /// without external corroboration.
    pub auto_apply_min_confidence: f32,
    /// Hard ceiling on autonomous applications per idle tick. A bound on
    /// blast radius, not a tuning parameter: if something goes wrong the
    /// damage is O(this), not O(queue).
    pub auto_apply_max_per_tick: u32,
}

impl Default for CuriosityConfig {
    fn default() -> Self {
        CuriosityConfig {
            enabled: true,
            wonder_queue_cap: 4096,
            interest_profile_cap: 256,
            deadend_yield_threshold: 2,
            isolated_degree_threshold: 3,
            knowledge_desert_threshold: 0.3,
            interest_decay_halflife_ms: DEFAULT_INTEREST_HALFLIFE_MS,
            intensity_floor: 0.1,
            gap_appearance_threshold: 10,
            // OFF. See the field docs: this is a product decision.
            auto_apply: false,
            auto_apply_min_confidence: 0.75,
            auto_apply_max_per_tick: 8,
        }
    }
}

/// Type of curiosity signal — what made a seed "interesting".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SignalType {
    /// Frontier hit a hub but yielded < `deadend_yield_threshold`
    /// further activations.
    DeadEnd,
    /// Node activated above 0.3 but has fewer than
    /// `isolated_degree_threshold` total neighbors.
    IsolatedNode,
    /// Spread exhausted before reaching `knowledge_desert_threshold`
    /// in top activation.
    KnowledgeDesert,
    /// Spread frontier touched a hub it couldn't traverse (no outgoing
    /// activation above ACTIVATION_THRESHOLD).
    HubBoundary,
}

/// Phase-2 output: a confirmed curiosity signal. Surfaced via
/// `AskResult.curiosity_signals` (per-call) and `Brain.wonder_queue`
/// (cross-call, capped).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CuriositySignal {
    /// The seed the signal is *about* (the hub for HubBoundary/DeadEnd,
    /// the isolated node for IsolatedNode, the query top-result for
    /// KnowledgeDesert).
    pub seed: Seed,
    /// What kind of unknown we noticed.
    pub signal_type: SignalType,
    /// 0.0 .. 1.0. Higher = more interesting.
    pub intensity: f32,
    /// Scope active at detection time. Lets a session signal be told
    /// apart from a baseline signal.
    pub scope: TemporalScope,
    /// Epoch millis when detected.
    pub timestamp_ms: u64,
}

/// Phase-1 output: a raw observation made inline during spread. Internal
/// — not serialized; lives only between spread and
/// `classify_curiosity_candidates`.
#[derive(Debug, Clone)]
pub struct CuriosityCandidate {
    pub seed: Seed,
    pub signal_type: SignalType,
    /// Free-form numeric payload interpreted per signal type:
    /// - DeadEnd: number of yields past this hub (usize cast).
    /// - IsolatedNode: this node's activation level.
    /// - KnowledgeDesert: top activation seen in spread.
    /// - HubBoundary: hub's degree (usize cast).
    pub payload: f32,
    pub timestamp_ms: u64,
}

/// Buffer + dedupe set for one spread call's curiosity recording pass.
/// Mirrors `BridgeCollector`. Created fresh per spread, consumed by
/// `classify_curiosity_candidates`.
#[derive(Debug)]
pub struct CuriosityCollector {
    /// Recorded raw candidates, deduped by `(SignalType, seed)`.
    pub candidates: Vec<CuriosityCandidate>,
    /// Set of already-recorded `(SignalType, seed)` pairs.
    pub seen: HashSet<(SignalType, Seed)>,
    /// Number of would-be candidates dropped because the cap was hit.
    pub dropped_overflow: usize,
    /// Per-spread cap (mirror of a config knob — keeps spread bounded
    /// even on pathological graphs).
    pub cap: usize,
}

impl CuriosityCollector {
    /// Empty collector with the given cap.
    pub fn new(cap: usize) -> Self {
        CuriosityCollector {
            candidates: Vec::new(),
            seen: HashSet::new(),
            dropped_overflow: 0,
            cap,
        }
    }

    /// Record a `(signal_type, seed)` candidate. Dedupes via `seen`,
    /// respects the cap.
    #[inline]
    pub fn record(
        &mut self,
        signal_type: SignalType,
        seed: Seed,
        payload: f32,
        timestamp_ms: u64,
    ) {
        if !self.seen.insert((signal_type, seed)) {
            return;
        }
        if self.candidates.len() >= self.cap {
            self.dropped_overflow += 1;
            return;
        }
        self.candidates.push(CuriosityCandidate {
            seed,
            signal_type,
            payload,
            timestamp_ms,
        });
    }
}

/// Hub criterion: stopword content OR degree above the configured
/// threshold. Mirrors `bridge::is_hub` so curiosity and bridge use the
/// same hub definition.
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

/// Total node degree (out + in).
#[inline]
fn degree(network: &AssociativeNetwork, seed: &Seed) -> usize {
    network.seam_total_degree(seed)
}

/// Phase-2 classifier. Takes raw inline candidates plus the final
/// activations map (seed -> activation) and produces scored signals.
///
/// Synthesises three signal types:
///   - `DeadEnd`/`HubBoundary` from inline candidates (intensity from
///     payload).
///   - `IsolatedNode` from the activations map (any seed with
///     activation > 0.3 and degree < `isolated_degree_threshold`).
///   - `KnowledgeDesert` (single signal) when the highest activation
///     in the map falls below `knowledge_desert_threshold`. Attached to
///     the highest-activation seed for traceability; falls back to the
///     first query seed if no node activated.
///
/// Drops signals below `intensity_floor`.
pub fn classify_curiosity_candidates(
    network: &AssociativeNetwork,
    candidates: &[CuriosityCandidate],
    activations: &[(Seed, f32)],
    query_seeds: &[Seed],
    scope: &TemporalScope,
    cfg: &CuriosityConfig,
    timestamp_ms: u64,
) -> Vec<CuriositySignal> {
    if !cfg.enabled {
        return Vec::new();
    }
    let mut events: Vec<CuriositySignal> = Vec::with_capacity(candidates.len() + 4);
    let mut emitted: HashSet<(SignalType, Seed)> = HashSet::new();

    // 1. Inline-collected DeadEnd / HubBoundary.
    for c in candidates {
        let intensity = match c.signal_type {
            SignalType::DeadEnd => {
                // 1.0 - yields / threshold; clamped.
                let yields = c.payload as f32;
                let t = cfg.deadend_yield_threshold.max(1) as f32;
                (1.0 - (yields / t)).clamp(0.0, 1.0)
            }
            SignalType::HubBoundary => {
                // 0.5 + 0.5 * normalized hub size. Normalize against
                // DEGREE_NORM_THRESHOLD (the "true super-hub" line) so
                // bigger hubs blocking the frontier score higher.
                let deg = c.payload as f32;
                let norm =
                    (deg / DEGREE_NORM_THRESHOLD as f32).min(1.0_f32).max(0.0_f32);
                (0.5 + 0.5 * norm).clamp(0.0, 1.0)
            }
            _ => continue, // IsolatedNode / KnowledgeDesert synthesized below
        };
        if intensity < cfg.intensity_floor {
            continue;
        }
        if !emitted.insert((c.signal_type, c.seed)) {
            continue;
        }
        events.push(CuriositySignal {
            seed: c.seed,
            signal_type: c.signal_type,
            intensity,
            scope: scope.clone(),
            timestamp_ms: c.timestamp_ms,
        });
    }

    // 2. IsolatedNode — synthesized post-spread from activations.
    for (seed, act) in activations {
        if *act <= 0.3 {
            continue;
        }
        let deg = degree(network, seed);
        if deg >= cfg.isolated_degree_threshold {
            continue;
        }
        // (activation - 0.3) * (1.0 - degree / threshold), clamped.
        let t = cfg.isolated_degree_threshold.max(1) as f32;
        let intensity = ((*act - 0.3) * (1.0 - deg as f32 / t)).clamp(0.0, 1.0);
        if intensity < cfg.intensity_floor {
            continue;
        }
        if !emitted.insert((SignalType::IsolatedNode, *seed)) {
            continue;
        }
        events.push(CuriositySignal {
            seed: *seed,
            signal_type: SignalType::IsolatedNode,
            intensity,
            scope: scope.clone(),
            timestamp_ms,
        });
    }

    // 3. KnowledgeDesert — single signal when the top activation is
    //    below threshold. Attach to the top-activated seed if any,
    //    otherwise to the first query seed.
    let top = activations
        .iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
    let top_act = top.map(|(_, a)| *a).unwrap_or(0.0);
    if top_act < cfg.knowledge_desert_threshold {
        let anchor = top
            .map(|(s, _)| *s)
            .or_else(|| query_seeds.first().copied());
        if let Some(seed) = anchor {
            let denom = cfg.knowledge_desert_threshold.max(1e-6);
            let intensity = (1.0 - (top_act / denom)).clamp(0.0, 1.0);
            if intensity >= cfg.intensity_floor
                && emitted.insert((SignalType::KnowledgeDesert, seed))
            {
                events.push(CuriositySignal {
                    seed,
                    signal_type: SignalType::KnowledgeDesert,
                    intensity,
                    scope: scope.clone(),
                    timestamp_ms,
                });
            }
        }
    }

    events
}

/// A signal wrapped for ordering inside the wonder queue. `BinaryHeap`
/// is a max-heap; `Ord` returns `intensity` order so peek gives the
/// highest-intensity signal.
#[derive(Debug, Clone)]
pub struct RankedSignal {
    pub signal: CuriositySignal,
}

impl PartialEq for RankedSignal {
    fn eq(&self, other: &Self) -> bool {
        self.signal.intensity == other.signal.intensity
    }
}
impl Eq for RankedSignal {}
impl PartialOrd for RankedSignal {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for RankedSignal {
    fn cmp(&self, other: &Self) -> Ordering {
        self.signal
            .intensity
            .partial_cmp(&other.signal.intensity)
            .unwrap_or(Ordering::Equal)
    }
}

/// Bounded ranked queue of curiosity signals. Insertion past `cap`
/// drops the lowest-intensity signal (if it would lower the cap).
#[derive(Debug)]
pub struct WonderQueue {
    inner: BinaryHeap<RankedSignal>,
    cap: usize,
}

impl WonderQueue {
    pub fn new(cap: usize) -> Self {
        WonderQueue {
            inner: BinaryHeap::new(),
            cap: cap.max(1),
        }
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Push a signal. If pushing would exceed the cap, the
    /// lowest-intensity item in the queue is dropped first (linear
    /// scan — bounded by `cap`, so cheap in practice).
    pub fn push(&mut self, signal: CuriositySignal) {
        if self.inner.len() >= self.cap {
            // Find lowest-intensity element; if the new one is even
            // lower, drop the new one instead.
            let mut min_intensity = f32::INFINITY;
            for r in self.inner.iter() {
                if r.signal.intensity < min_intensity {
                    min_intensity = r.signal.intensity;
                }
            }
            if signal.intensity <= min_intensity {
                return;
            }
            // Rebuild without one occurrence of the min element.
            let mut kept: Vec<RankedSignal> = self.inner.drain().collect();
            if let Some(idx) = kept.iter().position(|r| r.signal.intensity == min_intensity) {
                kept.remove(idx);
            }
            self.inner = BinaryHeap::from(kept);
        }
        self.inner.push(RankedSignal { signal });
    }

    /// Peek at the highest-intensity signal, if any.
    pub fn peek(&self) -> Option<&CuriositySignal> {
        self.inner.peek().map(|r| &r.signal)
    }

    /// Pop the highest-intensity signal.
    pub fn pop(&mut self) -> Option<CuriositySignal> {
        self.inner.pop().map(|r| r.signal)
    }

    /// Top-N signals in descending intensity order. Non-destructive.
    pub fn top_n(&self, n: usize) -> Vec<CuriositySignal> {
        let mut all: Vec<CuriositySignal> = self.inner.iter().map(|r| r.signal.clone()).collect();
        all.sort_by(|a, b| {
            b.intensity.partial_cmp(&a.intensity).unwrap_or(Ordering::Equal)
        });
        all.truncate(n);
        all
    }
}

/// Port J.1 — serializable snapshot of the wonder queue state for
/// brain-file persistence. Brain-file save calls `WonderQueue::snapshot_signals`
/// to dump every queued signal into a `Vec<CuriositySignal>`; load reverses it
/// via `WonderQueue::restore_from`. The companion `gap_label_map` lives on
/// `Brain` directly (not here) so the snapshot stays a pure ranked-queue thing.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WonderState {
    /// Every signal currently in the wonder queue (heap order is not
    /// preserved; restore re-pushes them so the cap-eviction policy is
    /// re-applied at load).
    pub signals: Vec<CuriositySignal>,
    /// Hash-seed -> human-readable label (typically the original
    /// surface query that produced the gap). Looked up by
    /// `tool_wonder` when a synthetic seed has no node in the graph.
    pub gap_label_map: std::collections::HashMap<Seed, String>,
}

impl WonderQueue {
    /// Port J.1 — clone every queued signal for persistence. Order
    /// is heap order (NOT sorted by intensity). Cheap: O(n) clone.
    pub fn snapshot_signals(&self) -> Vec<CuriositySignal> {
        self.inner.iter().map(|r| r.signal.clone()).collect()
    }

    /// Port J.1 — drop current contents and re-push every signal in
    /// `signals`. Cap-eviction policy applies: signals beyond `cap`
    /// fall to the floor.
    pub fn restore_from(&mut self, signals: Vec<CuriositySignal>) {
        self.inner.clear();
        for sig in signals {
            self.push(sig);
        }
    }
}

/// One per-domain interest entry. `last_touched_ms` is the wall time of
/// the last update; `weight` is the un-decayed weight at that moment.
/// Effective weight is computed lazily by `InterestProfile::weight_at`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterestKeyframe {
    pub last_touched_ms: u64,
    pub weight: f32,
}

/// Serde adapter: serialize `HashMap<Seed, InterestKeyframe>` as a list of
/// `(Seed, InterestKeyframe)` pairs. JSON object keys must be strings, but
/// `Seed` serializes as a 32-byte array — so we sidestep map-key encoding
/// by emitting a sequence instead.
mod interest_map_serde {
    use super::{InterestKeyframe, Seed};
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::collections::HashMap;

    pub fn serialize<S: Serializer>(
        map: &HashMap<Seed, InterestKeyframe>,
        s: S,
    ) -> Result<S::Ok, S::Error> {
        let v: Vec<(&Seed, &InterestKeyframe)> = map.iter().collect();
        v.serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        d: D,
    ) -> Result<HashMap<Seed, InterestKeyframe>, D::Error> {
        let v = Vec::<(Seed, InterestKeyframe)>::deserialize(d)?;
        Ok(v.into_iter().collect())
    }
}

/// Per-domain interest tracker keyed by hub seed. Bounded; the oldest
/// entry is evicted on overflow.
///
/// Persisted in the brain file: interest IS Lu's personality and must
/// survive a restart. Loaded with `#[serde(default)]` so older brain
/// files (no `interest_profile` field) come up with an empty profile.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct InterestProfile {
    #[serde(with = "interest_map_serde")]
    pub interests: HashMap<Seed, InterestKeyframe>,
    pub cap: usize,
}

impl InterestProfile {
    pub fn new(cap: usize) -> Self {
        InterestProfile {
            interests: HashMap::new(),
            cap: cap.max(1),
        }
    }

    /// Touch a domain (hub seed). Re-weights against the decayed prior
    /// so repeated touches stack with diminishing return rather than
    /// resetting. Adds 1.0 - prior so weight asymptotically approaches
    /// 1.0 across many touches.
    pub fn touch(&mut self, seed: Seed, now_ms: u64, halflife_ms: u64) {
        let prior = self
            .interests
            .get(&seed)
            .map(|k| decay(k.weight, now_ms.saturating_sub(k.last_touched_ms), halflife_ms))
            .unwrap_or(0.0);
        let new_weight = (prior + (1.0 - prior) * 0.5).clamp(0.0, 1.0);
        self.interests.insert(
            seed,
            InterestKeyframe {
                last_touched_ms: now_ms,
                weight: new_weight,
            },
        );
        // Evict oldest if over cap.
        if self.interests.len() > self.cap {
            if let Some((evict, _)) = self
                .interests
                .iter()
                .min_by_key(|(_, k)| k.last_touched_ms)
                .map(|(s, k)| (*s, k.last_touched_ms))
            {
                self.interests.remove(&evict);
            }
        }
    }

    /// Effective weight at `now_ms`, decayed from `last_touched_ms`.
    pub fn weight_at(&self, seed: &Seed, now_ms: u64, halflife_ms: u64) -> f32 {
        match self.interests.get(seed) {
            Some(k) => decay(
                k.weight,
                now_ms.saturating_sub(k.last_touched_ms),
                halflife_ms,
            ),
            None => 0.0,
        }
    }

    /// Snapshot all interest entries (seed, decayed weight, last_touched_ms),
    /// sorted by decayed weight descending.
    pub fn snapshot(&self, now_ms: u64, halflife_ms: u64) -> Vec<(Seed, f32, u64)> {
        let mut v: Vec<(Seed, f32, u64)> = self
            .interests
            .iter()
            .map(|(s, k)| {
                (
                    *s,
                    decay(k.weight, now_ms.saturating_sub(k.last_touched_ms), halflife_ms),
                    k.last_touched_ms,
                )
            })
            .collect();
        v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        v
    }
}

/// Exponential decay: w * 2^(-elapsed / halflife). Mirrors `TemporalBond::weight_at`.
#[inline]
fn decay(weight: f32, elapsed_ms: u64, halflife_ms: u64) -> f32 {
    if halflife_ms == 0 {
        return 0.0;
    }
    let factor = (-(elapsed_ms as f64) / halflife_ms as f64).exp2();
    (weight as f64 * factor) as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scar::ScarLog;
    use crate::bond::{BondType, TemporalBond};
    use crate::brain::{now_ms, Brain};
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

    /// Curiosity config with a tiny hub threshold so unit tests can
    /// build small graphs with hub-degree nodes.
    fn small_cfg() -> CuriosityConfig {
        CuriosityConfig {
            enabled: true,
            wonder_queue_cap: 16,
            interest_profile_cap: 8,
            deadend_yield_threshold: 2,
            isolated_degree_threshold: 3,
            knowledge_desert_threshold: 0.3,
            interest_decay_halflife_ms: 1_000_000,
            intensity_floor: 0.05,
            gap_appearance_threshold: 10,
            auto_apply: false,
            auto_apply_min_confidence: 0.75,
            auto_apply_max_per_tick: 8,
        }
    }

    /// (1) Spread frontier hits a hub whose expansion yields fewer than
    ///     `deadend_yield_threshold` further activations → DeadEnd recorded.
    ///     Setup: hub has 4 outgoing leaves but only 1 carries strong
    ///     enough weight to propagate. yields_this_step = 1 < threshold (2).
    #[test]
    fn test_deadend_signal_emitted() {
        let mut net = AssociativeNetwork::new();
        let entry = add_node(&mut net, "entry_de");
        let hub = add_node(&mut net, "hub_de");
        net.add_bond(TemporalBond::new(entry, hub, BondType::RelatedTo, 0.9, 0))
            .unwrap();
        // One strong leaf — yields a real propagation.
        let strong_leaf = add_node(&mut net, "strong_leaf");
        net.add_bond(TemporalBond::new(hub, strong_leaf, BondType::RelatedTo, 0.9, 0))
            .unwrap();
        // Three weak leaves — bonds too small to push activation past
        // ACTIVATION_THRESHOLD, so they contribute zero to yields.
        for i in 0..3 {
            let weak_leaf = add_node(&mut net, &format!("weak_leaf_{}", i));
            net.add_bond(TemporalBond::new(hub, weak_leaf, BondType::RelatedTo, 0.001, 0))
                .unwrap();
        }
        // hub now has degree 5 (1 entry + 4 leaves) > 3 → counts as hub.
        let cfg = small_cfg();
        let mut cc = CuriosityCollector::new(64);
        let scope = TemporalScope::Current;
        let _ = net.spread_activation_weighted_with_curiosity(
            &[(entry, 1.0)],
            &scope,
            3,
            50,
            &mut cc,
            &cfg,
            3,
            now_ms() as u64,
            &ScarLog::default(),
        );
        let has_deadend = cc.candidates.iter().any(|c| c.signal_type == SignalType::DeadEnd);
        assert!(
            has_deadend,
            "expected a DeadEnd candidate; got: {:?}",
            cc.candidates,
        );
    }

    /// (2) Isolated high-activation node → IsolatedNode signal.
    #[test]
    fn test_isolated_node_signal_emitted() {
        // A is the seed (activates to ~conf=0.9). It has a single
        // neighbor B (degree 1, well below threshold 3). A's own
        // activation is high (0.9) and degree 1.
        let mut net = AssociativeNetwork::new();
        let a = add_node(&mut net, "alpha_iso");
        let b = add_node(&mut net, "beta_iso");
        net.add_bond(TemporalBond::new(a, b, BondType::RelatedTo, 0.9, 0))
            .unwrap();

        let cfg = small_cfg();
        let scope = TemporalScope::Current;
        // Use spread_activation_weighted (no collector) for the activations map.
        let recall = net.spread_activation_weighted(&[(a, 1.0)], &scope, 2, 10, &ScarLog::default());
        let acts: Vec<(Seed, f32)> = recall
            .activated
            .iter()
            .map(|n| (n.seed, n.activation))
            .collect();
        let signals = classify_curiosity_candidates(
            &net, &[], &acts, &[a], &scope, &cfg, now_ms() as u64,
        );
        assert!(
            signals.iter().any(|s| s.signal_type == SignalType::IsolatedNode),
            "expected IsolatedNode signal for low-degree high-activation seed; got {:?}",
            signals,
        );
    }

    /// (3) Top activation below threshold → KnowledgeDesert signal.
    #[test]
    fn test_knowledge_desert_when_top_activation_low() {
        // No nodes that the query can find → empty activations →
        // top activation 0.0 → KnowledgeDesert.
        let mut net = AssociativeNetwork::new();
        let _ = add_node(&mut net, "unrelated");
        let cfg = small_cfg();
        let scope = TemporalScope::Current;
        let dummy_seed = Seed::from_label("phantom");
        let signals = classify_curiosity_candidates(
            &net,
            &[],
            &[],                 // empty activations
            &[dummy_seed],
            &scope,
            &cfg,
            now_ms() as u64,
        );
        assert!(
            signals.iter().any(|s| s.signal_type == SignalType::KnowledgeDesert),
            "expected KnowledgeDesert; got {:?}",
            signals,
        );
    }

    /// (4) Frontier reaches a hub it can't traverse → HubBoundary.
    #[test]
    fn test_hub_boundary_signal_when_hub_blocks() {
        // Same shape as deadend but the hub bond going IN from `entry`
        // is strong, while neighbor bonds are very weak so spread can't
        // continue past the hub.
        let mut net = AssociativeNetwork::new();
        let entry = add_node(&mut net, "entry_b");
        let hub = add_node(&mut net, "hub_b");
        net.add_bond(TemporalBond::new(entry, hub, BondType::RelatedTo, 0.9, 0))
            .unwrap();
        for i in 0..4 {
            let leaf = add_node(&mut net, &format!("blocked_leaf_{}", i));
            // Tiny weight × RelatedTo 0.8 base = tiny effective weight.
            net.add_bond(TemporalBond::new(hub, leaf, BondType::RelatedTo, 0.001, 0))
                .unwrap();
        }
        let cfg = small_cfg();
        let mut cc = CuriosityCollector::new(64);
        let scope = TemporalScope::Current;
        let _ = net.spread_activation_weighted_with_curiosity(
            &[(entry, 1.0)],
            &scope,
            3,
            50,
            &mut cc,
            &cfg,
            3,
            now_ms() as u64,
            &ScarLog::default(),
        );
        assert!(
            cc.candidates.iter().any(|c| c.signal_type == SignalType::HubBoundary),
            "expected HubBoundary candidate; got: {:?}",
            cc.candidates,
        );
    }

    /// (5) Wonder queue ranks signals by intensity.
    #[test]
    fn test_wonder_queue_ranks_by_intensity() {
        let mut q = WonderQueue::new(8);
        let mk = |intensity: f32, label: &str| CuriositySignal {
            seed: Seed::from_label(label),
            signal_type: SignalType::DeadEnd,
            intensity,
            scope: TemporalScope::Current,
            timestamp_ms: 0,
        };
        q.push(mk(0.3, "low"));
        q.push(mk(0.9, "high"));
        q.push(mk(0.5, "mid"));
        let top = q.peek().expect("non-empty");
        assert!((top.intensity - 0.9).abs() < 1e-6);
        let top3 = q.top_n(3);
        assert_eq!(top3.len(), 3);
        assert!(top3[0].intensity >= top3[1].intensity);
        assert!(top3[1].intensity >= top3[2].intensity);
    }

    /// (6) InterestProfile decays over time.
    #[test]
    fn test_interest_profile_decays_over_time() {
        let halflife = 1000_u64;
        let mut prof = InterestProfile::new(8);
        let s = Seed::from_label("domain");
        prof.touch(s, 0, halflife);
        let initial = prof.weight_at(&s, 0, halflife);
        assert!(initial > 0.0);
        // Fast-forward ~10 half-lives via Scope::Historical-style
        // timestamp increase. Decay is monotonic.
        let later = prof.weight_at(&s, 10_000, halflife);
        assert!(later < initial * 0.01, "weight after 10 halflives must collapse: initial={}, later={}", initial, later);
    }

    /// (7) Signals carry the scope active at detection time.
    #[test]
    fn test_curiosity_signals_carry_scope() {
        let mut brain = Brain::new();
        // Build a tiny graph and force a desert query inside session 99.
        brain.curiosity_config.enabled = true;
        brain.curiosity_config.knowledge_desert_threshold = 0.99; // force desert
        let session_id = brain.session_start();
        let _ = session_id;
        brain.teach(
            "anchor", BondType::RelatedTo, "tip",
            0.9, Confidence::Verified,
        );
        let r = brain.ask("anchor", 1, 5);
        assert!(
            !r.curiosity_signals.is_empty(),
            "high desert threshold should always emit a signal",
        );
        for s in &r.curiosity_signals {
            assert!(
                matches!(s.scope, TemporalScope::Session { .. }),
                "signal must carry session scope; got {:?}",
                s.scope,
            );
        }
    }

    /// (8) Stretch: thinking layer consults the wonder queue. Seeded a
    ///     strong signal, then run think and confirm the wondered seed
    ///     appears in the per-step top_seeds (the candidate hypothesis
    ///     pool — see thinking::propose_hypotheses).
    #[test]
    fn test_thinking_consults_wonder_queue() {
        use crate::thinking::ThinkConfig;
        let mut brain = Brain::new();
        // Seed a graph the planner can pair from.
        brain.teach("alpha", BondType::IsA, "topic", 0.95, Confidence::Verified);
        brain.teach("topic", BondType::RelatedTo, "intermediary", 0.9, Confidence::Verified);
        brain.teach("alpha", BondType::HasProperty, "feature", 0.9, Confidence::Verified);
        // Inject a high-intensity wonder signal pointing at a known
        // node already in the graph (`feature`).
        let wonder_seed = Seed::from_label("feature");
        brain.wonder_queue.push(CuriositySignal {
            seed: wonder_seed,
            signal_type: SignalType::IsolatedNode,
            intensity: 0.9,
            scope: TemporalScope::Current,
            timestamp_ms: now_ms() as u64,
        });
        let r = brain.think("alpha", ThinkConfig::default());
        let mut found = false;
        for step in &r.trace.steps {
            if step.observation.top_seeds.contains(&wonder_seed) {
                found = true;
                break;
            }
        }
        assert!(
            found,
            "thinking must consult the wonder queue and surface the wondered seed",
        );
    }

    /// Defaults are sized for production traffic, not the toy values
    /// used in unit tests. Regression guard for the calibration fix.
    #[test]
    fn test_wonder_queue_default_cap_is_4096() {
        let cfg = CuriosityConfig::default();
        assert_eq!(cfg.wonder_queue_cap, 4096);
        assert_eq!(cfg.interest_profile_cap, 256);
    }

    /// Port J.1 — wonder queue snapshot/restore round-trip preserves
    /// signals AND keeps cap eviction behaviour.
    #[test]
    fn test_wonder_queue_snapshot_restore_round_trip() {
        let mut q = WonderQueue::new(8);
        let mk = |intensity: f32, label: &str| CuriositySignal {
            seed: Seed::from_label(label),
            signal_type: SignalType::KnowledgeDesert,
            intensity,
            scope: TemporalScope::Current,
            timestamp_ms: 100,
        };
        q.push(mk(0.9, "strong"));
        q.push(mk(0.4, "medium"));
        q.push(mk(0.1, "weak"));
        let snap = q.snapshot_signals();
        assert_eq!(snap.len(), 3);
        let mut q2 = WonderQueue::new(8);
        q2.restore_from(snap);
        assert_eq!(q2.len(), 3);
        let top = q2.peek().expect("non-empty after restore");
        assert!((top.intensity - 0.9).abs() < 1e-6, "strongest survives heap rebuild");
        let top3 = q2.top_n(3);
        assert!(top3[0].intensity >= top3[1].intensity);
        assert!(top3[1].intensity >= top3[2].intensity);
    }

    /// Port J.1 — WonderState round-trips through bincode just like
    /// other side tables on the brain file.
    #[test]
    fn test_wonder_state_bincode_round_trip() {
        let mut state = WonderState::default();
        state.signals.push(CuriositySignal {
            seed: Seed::from_label("persistent_seed"),
            signal_type: SignalType::KnowledgeDesert,
            intensity: 0.7,
            scope: TemporalScope::Current,
            timestamp_ms: 999,
        });
        state.gap_label_map.insert(
            Seed::from_label("persistent_seed"),
            "how do you write a for loop in python".to_string(),
        );
        let bytes = bincode::serialize(&state).expect("bincode serialize");
        let decoded: WonderState = bincode::deserialize(&bytes).expect("bincode deserialize");
        assert_eq!(decoded.signals.len(), 1);
        assert_eq!(decoded.signals[0].intensity, 0.7);
        assert_eq!(
            decoded.gap_label_map.get(&Seed::from_label("persistent_seed")).map(String::as_str),
            Some("how do you write a for loop in python"),
        );
    }
}
