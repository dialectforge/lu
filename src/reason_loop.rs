//! OP-REASON-LOOP Phase 1 -- `ContextSet`, the accumulator for Lu's
//! prompt-reasoning loop (retrieve -> load-context -> reason -> re-retrieve).
//!
//! Design: internal notes (base spec) + #1396 (session-scoped revision).
//!
//! `Brain::ask_with_config` returns an `AskResult`, which is transient and
//! single-shot -- it is the product of ONE recall spread and is dropped at the
//! end of the request. The reasoning loop needs an accumulator that (a) can be
//! built from a single recall (the iters==1 free-degrade path, identical to
//! today), (b) can absorb additional recalls from later loop iterations
//! (Phase 3, mid-loop re-retrieval), and (c) can be merged into a
//! session-level hold that survives across turns so follow-ups warm-start
//! from it (Phase 1.5, #1396 SS10).
//!
//! `ContextSet` is that accumulator. It is activation-ranked, keyed by node
//! `Seed`, and bounded by a per-prompt read budget. Every `Entry` carries its
//! `source_turn` / `source_iter` and merge bookkeeping (`hits`,
//! `last_touched_turn`) from the start, so the Phase 1.5 session merge is
//! purely additive -- no schema change is required to reach it.
//!
//! GUEST-SAFE: a `ContextSet` holds only labels / definitions / activations
//! (derived, ephemeral view data). It never holds or writes nodes or bonds.

use crate::brain::AskResult;
use crate::node::Confidence;
use crate::seed::Seed;
use indexmap::IndexMap;

/// Default per-prompt read budget: the maximum number of distinct nodes a
/// single ContextSet holds from one recall. Recall is already activation-
/// descending, so the budget keeps the strongest N and drops the long weak
/// tail. Sized well above the renderer's working set (top_k defaults to 10)
/// so the iters==1 path never truncates anything the render would have seen.
pub const DEFAULT_READ_BUDGET: usize = 64;

/// How a node entered a `ContextSet` -- carried so downstream ranking /
/// eviction can weigh confidence and hop-distance, and so provenance is not
/// lost when the per-prompt set is merged into the session hold.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Provenance {
    /// Confidence of the recalled node at capture time.
    pub confidence: Confidence,
    /// Hops from the nearest query seed (0 == a seed itself).
    pub depth: usize,
}

/// One held context node.
#[derive(Debug, Clone)]
pub struct Entry {
    /// Node content label (from `ActivatedNode::content`).
    pub label: String,
    /// Stored definitional gloss, when resolved. `None` on the iters==1
    /// recall path (definitions are resolved lazily / by later phases); the
    /// field exists from the start so the session hold can carry glosses.
    pub definition: Option<String>,
    /// Best activation seen for this node across the merges that touched it.
    pub activation: f32,
    /// Where this entry came from (confidence + depth of the strongest hit).
    pub provenance: Provenance,
    /// Session turn index that first introduced this node (0 when stateless).
    pub source_turn: u64,
    /// Loop iteration index that first introduced this node (0 == the single
    /// initial recall).
    pub source_iter: u16,
    /// How many merges have touched this node. Starts at 1. Used by the
    /// Phase 1.5 relevance x recency evictor.
    pub hits: u32,
    /// Most recent turn that re-touched this node. Refreshed on every merge.
    pub last_touched_turn: u64,
}

/// Activation-ranked, seed-keyed, budget-bounded accumulator of context
/// nodes for one prompt (Phase 1) and, once merged, one session (Phase 1.5).
#[derive(Debug, Clone)]
pub struct ContextSet {
    /// Insertion-ordered map; ranked views sort by activation on demand.
    nodes: IndexMap<Seed, Entry>,
    /// Maximum distinct nodes this set will hold from recalls.
    read_budget: usize,
}

impl Default for ContextSet {
    fn default() -> Self {
        ContextSet { nodes: IndexMap::new(), read_budget: DEFAULT_READ_BUDGET }
    }
}

impl ContextSet {
    /// Empty set with the default read budget.
    pub fn new() -> Self {
        ContextSet::default()
    }

    /// Empty set with an explicit read budget.
    pub fn with_budget(read_budget: usize) -> Self {
        ContextSet { nodes: IndexMap::new(), read_budget }
    }

    /// Number of held nodes.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// True when no nodes are held.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// The per-prompt read budget.
    pub fn read_budget(&self) -> usize {
        self.read_budget
    }

    /// Borrow a held entry by seed.
    pub fn get(&self, seed: &Seed) -> Option<&Entry> {
        self.nodes.get(seed)
    }

    /// Iterate held entries in insertion order.
    pub fn iter(&self) -> impl Iterator<Item = (&Seed, &Entry)> {
        self.nodes.iter()
    }

    /// Build a ContextSet from a single `ask_with_config` recall. This is the
    /// iters==1 free-degrade constructor: the resulting set holds exactly the
    /// (budget-capped) recall the renderer already consumes, so rendering from
    /// `ask` and rendering from this set are identical by construction.
    pub fn from_ask(ask: &AskResult, turn: u64, iter: u16, read_budget: usize) -> Self {
        let mut cs = ContextSet::with_budget(read_budget);
        cs.merge_ask(ask, turn, iter);
        cs
    }

    /// Merge one recall (one loop iteration / one prompt turn) into this set.
    /// Union by seed. On collision: keep the max activation (and its
    /// provenance), bump `hits`, refresh `last_touched_turn`. Additive --
    /// never removes. Honors the read budget: because recall is activation-
    /// descending, once the budget is full only nodes ALREADY held are
    /// updated; strictly-new weaker nodes are dropped.
    pub fn merge_ask(&mut self, ask: &AskResult, turn: u64, iter: u16) {
        for node in ask.recall.activated.iter() {
            let known = self.nodes.contains_key(&node.seed);
            if !known && self.nodes.len() >= self.read_budget {
                break;
            }
            let prov = Provenance { confidence: node.confidence, depth: node.depth };
            self.merge_node(node.seed, &node.content, node.activation, prov, turn, iter);
        }
    }

    fn merge_node(
        &mut self,
        seed: Seed,
        label: &str,
        activation: f32,
        prov: Provenance,
        turn: u64,
        iter: u16,
    ) {
        match self.nodes.get_mut(&seed) {
            Some(e) => {
                if activation > e.activation {
                    e.activation = activation;
                    e.provenance = prov;
                }
                e.hits = e.hits.saturating_add(1);
                e.last_touched_turn = turn;
            }
            None => {
                self.nodes.insert(
                    seed,
                    Entry {
                        label: label.to_string(),
                        definition: None,
                        activation,
                        provenance: prov,
                        source_turn: turn,
                        source_iter: iter,
                        hits: 1,
                        last_touched_turn: turn,
                    },
                );
            }
        }
    }

    /// Merge another whole ContextSet into this one (the primitive the Phase
    /// 1.5 session hold uses to absorb each prompt's set). Union by seed;
    /// keep max activation, carry a gloss if the incoming entry has one and
    /// this one does not, sum `hits`, and advance `last_touched_turn`.
    /// Budget-bounded: strictly-new nodes past the budget are skipped.
    pub fn merge(&mut self, other: &ContextSet, turn: u64) {
        for (seed, incoming) in other.nodes.iter() {
            match self.nodes.get_mut(seed) {
                Some(cur) => {
                    if incoming.activation > cur.activation {
                        cur.activation = incoming.activation;
                        cur.provenance = incoming.provenance;
                    }
                    if cur.definition.is_none() && incoming.definition.is_some() {
                        cur.definition = incoming.definition.clone();
                    }
                    cur.hits = cur.hits.saturating_add(incoming.hits);
                    cur.last_touched_turn = cur.last_touched_turn.max(turn);
                }
                None => {
                    if self.nodes.len() >= self.read_budget {
                        continue;
                    }
                    let mut ne = incoming.clone();
                    ne.last_touched_turn = ne.last_touched_turn.max(turn);
                    self.nodes.insert(*seed, ne);
                }
            }
        }
    }

    /// Activation-ranked view (highest first). Stable for equal activations
    /// (retains insertion order via a stable sort).
    pub fn ranked(&self) -> Vec<(&Seed, &Entry)> {
        let mut v: Vec<(&Seed, &Entry)> = self.nodes.iter().collect();
        v.sort_by(|a, b| {
            b.1.activation
                .partial_cmp(&a.1.activation)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        v
    }
}

// =====================================================================
// SessionContext (Phase 1.5) -- the session-scoped hold
// =====================================================================

use std::collections::HashSet;

/// Node cap for a session-level held context. Sized in the ~1-2k band the
/// spec (#1396) calls for. Bounds RAM on marathon sessions: eviction runs
/// at commit time (end of each prompt), so the transient ceiling is
/// `MAX_SESSION_CONTEXT_NODES` + one prompt worth of `DEFAULT_READ_BUDGET`.
pub const MAX_SESSION_CONTEXT_NODES: usize = 1500;

impl ContextSet {
    /// True when a node with this seed is held.
    pub fn contains_key(&self, seed: &Seed) -> bool {
        self.nodes.contains_key(seed)
    }

    /// True when any held entry has this label (case-insensitive).
    pub fn contains_label(&self, label: &str) -> bool {
        let want = label.to_lowercase();
        self.nodes.values().any(|e| e.label.to_lowercase() == want)
    }

    /// Sum of `hits` across all held entries (merge-activity total).
    pub fn total_hits(&self) -> u64 {
        self.nodes.values().map(|e| e.hits as u64).sum()
    }

    /// Evict down to `cap` nodes by relevance x recency, keeping the
    /// highest-scoring. Score = activation * per_turn_decay^age, where
    /// age = current_turn - last_touched_turn (the same multiplicative
    /// half-life discipline as `SessionMemory::decay_anchors`). No-op when
    /// already within cap. Intended to run at COMMIT time (end of a
    /// prompt), never mid-answer.
    pub fn evict_to(&mut self, cap: usize, current_turn: u64, per_turn_decay: f32) {
        if self.nodes.len() <= cap {
            return;
        }
        let mut scored: Vec<(Seed, f32)> = self
            .nodes
            .iter()
            .map(|(seed, e)| {
                let age = current_turn.saturating_sub(e.last_touched_turn);
                let recency = per_turn_decay.powi(age.min(i32::MAX as u64) as i32);
                (*seed, e.activation * recency)
            })
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let keep: HashSet<Seed> = scored.into_iter().take(cap).map(|(s, _)| s).collect();
        self.nodes.retain(|seed, _| keep.contains(seed));
    }
}

/// The session-scoped context hold (#1396 SS10, the HOLD-everything-for-
/// follow-ups requirement). Lives on the in-RAM `SessionMemory` (never the
/// WAL-backed lb_session store). Each prompt per-prompt `ContextSet` is
/// merged in and held for the whole session, so a follow-up warm-starts
/// from the union of everything seen so far. Bounded by
/// `MAX_SESSION_CONTEXT_NODES` via a relevance x recency evictor applied at
/// commit time. GUEST-SAFE: labels / definitions / activations only --
/// never nodes or bonds.
#[derive(Debug, Clone)]
pub struct SessionContext {
    set: ContextSet,
    cap: usize,
}

impl Default for SessionContext {
    fn default() -> Self {
        // Unbounded merge budget: the cap is enforced by eviction at commit
        // time, not by refusing fresh nodes (which would freeze stale nodes
        // in and lock relevant ones out).
        SessionContext { set: ContextSet::with_budget(usize::MAX), cap: MAX_SESSION_CONTEXT_NODES }
    }
}

impl SessionContext {
    /// Empty hold with the default cap.
    pub fn new() -> Self {
        SessionContext::default()
    }

    /// Empty hold with an explicit cap (tests / tuning).
    pub fn with_cap(cap: usize) -> Self {
        SessionContext { set: ContextSet::with_budget(usize::MAX), cap }
    }

    /// Merge one prompt per-prompt ContextSet into the session hold. Does
    /// NOT evict -- eviction is a separate commit-time step so a merge
    /// mid-answer never drops a node the current answer might read.
    pub fn merge_prompt(&mut self, prompt: &ContextSet, turn: u64) {
        self.set.merge(prompt, turn);
    }

    /// Commit-time eviction to the cap (relevance x recency, reusing the
    /// `SessionMemory` anchor half-life). Call at the END of a prompt.
    pub fn evict(&mut self, current_turn: u64) {
        self.set.evict_to(self.cap, current_turn, crate::session_memory::ANCHOR_DECAY_FACTOR);
    }

    /// Number of held nodes.
    pub fn len(&self) -> usize {
        self.set.len()
    }

    /// True when nothing is held.
    pub fn is_empty(&self) -> bool {
        self.set.is_empty()
    }

    /// The node cap.
    pub fn cap(&self) -> usize {
        self.cap
    }

    /// True when a node with this label is held (case-insensitive).
    pub fn contains_label(&self, label: &str) -> bool {
        self.set.contains_label(label)
    }

    /// True when a node with this seed is held.
    pub fn contains_seed(&self, seed: &Seed) -> bool {
        self.set.contains_key(seed)
    }

    /// Top-`n` held labels by activation (the warm-start read a follow-up
    /// loop consults). Read-only view; never mutates the hold.
    pub fn warm_start_labels(&self, n: usize) -> Vec<String> {
        self.set.ranked().into_iter().take(n).map(|(_, e)| e.label.clone()).collect()
    }

    /// Borrow the underlying context set (read-only).
    pub fn context(&self) -> &ContextSet {
        &self.set
    }
}

// =====================================================================
// Reason<->retrieve loop (Phase 3): mid-loop re-retrieval / two-pass
// =====================================================================

use crate::brain::{Brain, BrainAskConfig};
use crate::narration::NarrationConfig;
use crate::thinking::{ThinkConfig, ThoughtTrace};

/// Max fresh re-seeds expanded per loop iteration (spec K ~ 3). Hub words are
/// filtered out BEFORE this cap so a single hub can never consume the budget.
pub const REASON_LOOP_MAX_SEEDS: usize = 3;

/// Bounded depth for each seeded re-retrieval spread -- a frontier
/// expansion from a concept, never a deep re-flood of the graph.
const REASON_LOOP_SEED_DEPTH: usize = 2;
/// Hard wall-clock floors used when the config disables the watchdogs (0),
/// so the loop is bounded regardless of config (read-amplification guard).
const REASON_LOOP_ASK_WALL_MS: u64 = 4_000;
const REASON_LOOP_THINK_WALL_MS: u64 = 5_000;
/// Loop-level total wall-clock ceiling (spec section 5): caps the whole
/// reason<->retrieve cycle regardless of per-iteration bounds.
const REASON_LOOP_TOTAL_WALL_MS: u64 = 12_000;

/// Per-iteration think budget for the loop. The loop reasons ONLY to derive
/// re-seeds (not to compose an answer), so one Observe->Plan->Act->Reflect
/// step is enough; more would multiply read cost without improving the seeds.
const REASON_LOOP_THINK_MAX_STEPS: usize = 1;

/// Why the reason<->retrieve loop stopped (spec section 5 termination).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TerminationReason {
    /// The loop was off (max_context_iters <= 1).
    #[default]
    LoopOff,
    /// Accumulated context cleared the idk floor -- answerable, stop.
    Grounded,
    /// No new re-seed available (gaps drained, no curiosity, no fresh topic).
    Converged,
    /// Hit the iteration budget (max_context_iters).
    IterBudget,
    /// The ContextSet reached its node read budget.
    ReadBudget,
    /// The loop-level wall-clock ceiling tripped.
    WallClock,
}

impl TerminationReason {
    pub fn as_str(&self) -> &'static str {
        match self {
            TerminationReason::LoopOff => "loop_off",
            TerminationReason::Grounded => "grounded",
            TerminationReason::Converged => "converged",
            TerminationReason::IterBudget => "iter_budget",
            TerminationReason::ReadBudget => "read_budget",
            TerminationReason::WallClock => "wall_clock",
        }
    }
}

/// Outcome of a reason<->retrieve loop, for observability + tests.
#[derive(Debug, Clone, Default)]
pub struct ReasonLoopStats {
    /// Extra retrieval iterations actually executed (0 when the loop was off
    /// or stopped before doing any retrieval).
    pub iters_ran: usize,
    /// Labels of the concepts reasoning chose to re-seed from, in order.
    pub seeds_used: Vec<String>,
    /// Why the loop stopped.
    pub terminated: TerminationReason,
}

/// Serializable summary carried on NarrationResult so a caller can observe the
/// loop over the wire (enriched ContextSet size, iterations run, and the
/// re-seed concepts). JSON-safe (no Seed map keys, unlike ContextSet itself).
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct ReasonLoopSummary {
    pub context_set_size: usize,
    pub iters_ran: usize,
    pub seeds: Vec<String>,
    /// Why the loop stopped ("grounded" / "iter_budget" / "converged" / ...).
    #[serde(default)]
    pub terminated: String,
    /// True when the loop-EXIT idk gate refused (context could not ground an
    /// answer, so narrate returned a refusal instead of prose).
    #[serde(default)]
    pub refused: bool,
}

/// Post-comprehension topic selector (NEW; spec [[two-pass-recall]]). After a
/// reasoning pass, pick the concepts reasoning converged on -- the confirmed
/// hypothesis endpoints first (strongest signal), then the final-step
/// observation top nodes -- as candidate expansion points for the next
/// retrieval. Returns (seed, label) so the caller can hub-filter by label.
/// Deduped; order = reasoning salience.
pub fn select_topic_seeds(_context: &ContextSet, think: &ThoughtTrace) -> Vec<(Seed, String)> {
    let mut out: Vec<(Seed, String)> = Vec::new();
    let mut seen: HashSet<Seed> = HashSet::new();
    // 1. Confirmed hypothesis endpoints -- what reasoning promoted.
    for step in &think.steps {
        for h in &step.reflection.confirmed {
            if seen.insert(h.to) {
                out.push((h.to, h.to_content.clone()));
            }
        }
    }
    // 2. Final-step observation top nodes -- what reasoning was looking at.
    if let Some(last) = think.steps.last() {
        for (s, c) in last
            .observation
            .top_seeds
            .iter()
            .zip(last.observation.top_contents.iter())
        {
            if seen.insert(*s) {
                out.push((*s, c.clone()));
            }
        }
    }
    out
}

/// Loop grounding check + EXIT idk floor (spec section 6). Builds a synthetic
/// AskResult whose recall is the accumulated ContextSet, then runs the SAME
/// idk_gate signals + classify that the entry floor uses. `Pass` means the
/// context can ground an answer; a non-`Pass` verdict at loop exit means
/// narrate must refuse rather than fabricate. Read-only, and monotone with
/// enrichment: merges only ADD activated nodes, so more iterations can only
/// raise grounding, never lower the floor.
pub fn classify_context(
    brain: &crate::brain::Brain,
    query: &str,
    ask: &AskResult,
    context: &ContextSet,
) -> (crate::idk_gate::IdkVerdict, crate::idk_gate::IdkSignals) {
    let thresholds = crate::idk_gate::IdkThresholds::default();
    // trace=None (no reasoning pass): classify chain-dependent Mode A is
    // UNREACHABLE for entry-passed queries anyway (their tokens already
    // matched), and running reason_with_config here on a hub-dense query cost
    // ~10s -- twice (grounded stop + exit floor) -- blowing the loop latency
    // ceiling. The token / activation / single-anchor-dominance signals (idk
    // Modes B/D), which are what the exit floor actually fires on, need no
    // trace. Keeping this read-only + cheap is what makes the loop safe to run
    // by default (Phase 5).
    let activated: Vec<crate::network::ActivatedNode> = context
        .iter()
        .map(|(seed, e)| crate::network::ActivatedNode {
            seed: *seed,
            content: e.label.clone(),
            activation: e.activation,
            confidence: e.provenance.confidence,
            path: Vec::new(),
            depth: e.provenance.depth,
        })
        .collect();
    let synth = AskResult {
        query: query.to_string(),
        seed_hits: ask.seed_hits.clone(),
        recall: crate::network::RecallResult {
            activated,
            query_seeds: ask.recall.query_seeds.clone(),
            scope: crate::scope::TemporalScope::Current,
            nodes_visited: 0,
            bonds_evaluated: 0,
            activation_truncated: false,
            status: crate::network::SpreadStatus::Complete,
        },
        hebbian_strengthened: 0,
        session_id: None,
        bridges: Vec::new(),
        curiosity_signals: Vec::new(),
        scars: Vec::new(),
        status: crate::brain::AskStatus::Complete,
    };
    let signals = crate::idk_gate::evaluate_signals(brain, &synth, query, None, &thresholds);
    let verdict = crate::idk_gate::classify(&signals, &thresholds);
    (verdict, signals)
}

/// Run the reason<->retrieve loop, enriching `context` in place.
///
/// No-op when `max_context_iters <= 1` (the free degrade path -- byte-identical
/// to Phase 1). READ-ONLY: reasoning runs under Phase-2 `no_write` and
/// re-retrieval uses the lean read-only `ask_with_config_seeded`, so the loop
/// never writes nodes or bonds. BOUNDED: at most `REASON_LOOP_MAX_SEEDS` fresh
/// seeds per iteration (hub / already-seeded / non-existent seeds dropped),
/// each seeded recall is a single bounded spread, and merges are capped by the
/// ContextSet read budget -- so total read work stays ~O(iters * a single pass).
/// Runs only on the narrate PASS path (after the idk_gate refusal floor), so it
/// can only ADD grounding to an already-answerable query, never fabricate.
pub fn run_reason_loop(
    brain: &mut Brain,
    query: &str,
    ask: &AskResult,
    context: &mut ContextSet,
    depth: usize,
    top_k: usize,
    cfg: &NarrationConfig,
) -> ReasonLoopStats {
    let mut stats = ReasonLoopStats::default();
    if cfg.max_context_iters <= 1 {
        return stats;
    }
    // Seeds already used to spread. Seed pass 2+ with something NEW rather than
    // redoing pass 1, so the original query seeds are off-limits as re-seeds.
    let mut seeded: HashSet<Seed> = ask.recall.query_seeds.iter().copied().collect();
    for s in &ask.seed_hits {
        seeded.insert(*s);
    }
    // Curiosity re-seeds from the initial ask, above the intensity floor.
    let floor = cfg.curiosity_min_intensity;
    let curiosity_seeds: Vec<Seed> = ask
        .curiosity_signals
        .iter()
        .filter(|c| c.intensity >= floor)
        .map(|c| c.seed)
        .collect();
    // Hard wall-clock floors: the loop must stay bounded even if the config
    // disabled the ask/think watchdogs (read-amplification guard; gate 3).
    let ask_wall = Some(std::time::Duration::from_millis(
        if cfg.ask_max_wall_clock_ms == 0 { REASON_LOOP_ASK_WALL_MS } else { cfg.ask_max_wall_clock_ms },
    ));
    let think_wall = Some(std::time::Duration::from_millis(
        if cfg.thinking_max_wall_clock_ms == 0 { REASON_LOOP_THINK_WALL_MS } else { cfg.thinking_max_wall_clock_ms },
    ));

    // Loop-level total wall-clock ceiling (spec section 5): even with per-iter
    // bounds, cap the whole reason<->retrieve cycle so it can never run away.
    let loop_start = std::time::Instant::now();
    let total_budget = std::time::Duration::from_millis(REASON_LOOP_TOTAL_WALL_MS);

    for iter in 1..cfg.max_context_iters {
        // STOP (wall-clock): loop-level ceiling, checked before spending an
        // iteration so the whole cycle can never run away.
        if loop_start.elapsed() >= total_budget {
            stats.terminated = TerminationReason::WallClock;
            return stats;
        }
        // 1. REASON (read-only, single step) to derive topic seeds.
        let t_cfg = ThinkConfig {
            no_write: true,
            max_steps: REASON_LOOP_THINK_MAX_STEPS,
            max_wall_clock: think_wall,
            max_ask_wall_clock: ask_wall,
            ..ThinkConfig::default()
        };
        let think = crate::thinking::think(brain, query, t_cfg);

        // 2. GATHER re-seed candidates: gap targets U curiosity U topic seeds.
        let mut cands: Vec<(Seed, Option<String>)> = Vec::new();
        // Post-comprehension topic seeds first (reasoning-derived, highest
        // value for the bounded K slots), then curiosity, then gap targets.
        for (s, label) in select_topic_seeds(context, &think.trace) {
            cands.push((s, Some(label)));
        }
        for s in &curiosity_seeds {
            cands.push((*s, None));
        }
        for g in brain.gap_emitter.drain() {
            cands.push((g.target_seed(), None));
        }

        // 3. FILTER: real node, not a hub stopword, not already seeded, unique.
        let mut chosen: Vec<(Seed, f32)> = Vec::new();
        let mut seen_this: HashSet<Seed> = HashSet::new();
        for (seed, label_hint) in cands {
            if seeded.contains(&seed) || !seen_this.insert(seed) {
                continue;
            }
            // Resolve the node label from the reasoning hint, else the node
            // store (works on v5 SST-resident nodes, unlike a heap get_node).
            // An empty label means the seed resolves to no real node content
            // (e.g. a synthetic gap target_seed) -- drop it.
            let label = label_hint
                .or_else(|| brain.network.node_label(&seed))
                .unwrap_or_default();
            if label.is_empty() {
                continue;
            }
            if crate::network::is_hub_stopword(&label) {
                continue;
            }
            chosen.push((seed, 1.0));
            stats
                .seeds_used
                .push(if label.is_empty() { seed.to_string() } else { label });
            if chosen.len() >= REASON_LOOP_MAX_SEEDS {
                break;
            }
        }

        // STOP (converged): nothing new to expand.
        if chosen.is_empty() {
            stats.terminated = TerminationReason::Converged;
            return stats;
        }
        for (s, _) in &chosen {
            seeded.insert(*s);
        }

        // 5. RE-RETRIEVE (read-only seeded recall) + MERGE into the context.
        // Shallow, bounded re-retrieval: cap the seeded spread depth so a
        // frontier expansion from a concept cannot fan out across the graph.
        let seed_depth = depth.min(REASON_LOOP_SEED_DEPTH);
        let ask_cfg = BrainAskConfig::new(seed_depth, top_k).with_wall_clock(ask_wall);
        let recall2 = brain.ask_with_config_seeded(&chosen, ask_cfg);
        context.merge_ask(&recall2, iter as u64, iter as u16);
        stats.iters_ran += 1;

        // STOP (read budget): the ContextSet reached its node cap; further
        // retrieval would only be dropped by the budget.
        if context.len() >= context.read_budget() {
            stats.terminated = TerminationReason::ReadBudget;
            return stats;
        }

        // STOP (grounded): after enriching, if the accumulated context now
        // clears the idk floor + chain coherence, the query is answerable --
        // stop early rather than burning the rest of the iteration budget.
        if classify_context(brain, query, ask, context).0
            == crate::idk_gate::IdkVerdict::Pass
        {
            stats.terminated = TerminationReason::Grounded;
            return stats;
        }
    }
    // Ran out the iteration budget without grounding.
    stats.terminated = TerminationReason::IterBudget;
    stats
}
impl ContextSet {
    /// Entries FIRST introduced by the reason<->retrieve loop
    /// (`source_iter >= 1`), activation-ranked. These are the
    /// reasoning-derived 2nd-hop nodes a single pass-1 recall did not reach
    /// -- the loop's contribution, surfaced for answer synthesis (#225).
    pub fn loop_added_ranked(&self) -> Vec<(&Seed, &Entry)> {
        let mut v: Vec<(&Seed, &Entry)> =
            self.nodes.iter().filter(|(_, e)| e.source_iter >= 1).collect();
        v.sort_by(|a, b| {
            b.1.activation
                .partial_cmp(&a.1.activation)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brain::{AskResult, AskStatus};
    use crate::network::{ActivatedNode, RecallResult, SpreadStatus};
    use crate::scope::TemporalScope;

    fn seed(b: u8) -> Seed {
        let mut raw = [0u8; 32];
        raw[0] = b;
        Seed::from_bytes(raw)
    }

    fn node(b: u8, label: &str, activation: f32, depth: usize) -> ActivatedNode {
        ActivatedNode {
            seed: seed(b),
            content: label.to_string(),
            activation,
            confidence: Confidence::Unverified,
            path: Vec::new(),
            depth,
        }
    }

    fn ask_with(nodes: Vec<ActivatedNode>) -> AskResult {
        AskResult {
            query: "q".to_string(),
            seed_hits: Vec::new(),
            recall: RecallResult {
                activated: nodes,
                query_seeds: Vec::new(),
                scope: TemporalScope::Current,
                nodes_visited: 0,
                bonds_evaluated: 0,
                activation_truncated: false,
                status: SpreadStatus::Complete,
            },
            hebbian_strengthened: 0,
            session_id: None,
            bridges: Vec::new(),
            curiosity_signals: Vec::new(),
            scars: Vec::new(),
            status: AskStatus::Complete,
        }
    }

    #[test]
    fn from_ask_captures_recall_in_activation_order() {
        let ask = ask_with(vec![
            node(1, "dog", 0.9, 0),
            node(2, "wolf", 0.5, 1),
            node(3, "canine", 0.3, 1),
        ]);
        let cs = ContextSet::from_ask(&ask, 0, 0, DEFAULT_READ_BUDGET);
        assert_eq!(cs.len(), 3);
        let ranked = cs.ranked();
        assert_eq!(ranked[0].1.label, "dog");
        assert_eq!(ranked[1].1.label, "wolf");
        assert_eq!(ranked[2].1.label, "canine");
        assert_eq!(ranked[0].1.hits, 1);
        assert_eq!(ranked[0].1.provenance.depth, 0);
    }

    #[test]
    fn read_budget_keeps_strongest_and_drops_weak_tail() {
        let ask = ask_with(vec![
            node(1, "a", 0.9, 0),
            node(2, "b", 0.8, 1),
            node(3, "c", 0.7, 1),
        ]);
        let cs = ContextSet::from_ask(&ask, 0, 0, 2);
        assert_eq!(cs.len(), 2);
        assert!(cs.get(&seed(1)).is_some());
        assert!(cs.get(&seed(2)).is_some());
        assert!(cs.get(&seed(3)).is_none());
    }

    #[test]
    fn merge_ask_collision_keeps_max_activation_and_bumps_hits() {
        let mut cs = ContextSet::from_ask(&ask_with(vec![node(1, "dog", 0.4, 0)]), 0, 0, DEFAULT_READ_BUDGET);
        // second turn re-hits the same node with a HIGHER activation
        cs.merge_ask(&ask_with(vec![node(1, "dog", 0.7, 0)]), 1, 0);
        assert_eq!(cs.len(), 1);
        let e = cs.get(&seed(1)).unwrap();
        assert_eq!(e.activation, 0.7);
        assert_eq!(e.hits, 2);
        assert_eq!(e.source_turn, 0);
        assert_eq!(e.last_touched_turn, 1);
    }

    #[test]
    fn merge_ask_collision_keeps_prior_max_when_new_is_weaker() {
        let mut cs = ContextSet::from_ask(&ask_with(vec![node(1, "dog", 0.7, 0)]), 0, 0, DEFAULT_READ_BUDGET);
        cs.merge_ask(&ask_with(vec![node(1, "dog", 0.2, 0)]), 1, 0);
        let e = cs.get(&seed(1)).unwrap();
        assert_eq!(e.activation, 0.7);
        assert_eq!(e.hits, 2);
    }

    #[test]
    fn set_to_set_merge_unions_and_advances_recency() {
        let mut a = ContextSet::from_ask(&ask_with(vec![node(1, "dog", 0.9, 0)]), 0, 0, DEFAULT_READ_BUDGET);
        let b = ContextSet::from_ask(&ask_with(vec![node(1, "dog", 0.95, 0), node(2, "wolf", 0.4, 1)]), 3, 0, DEFAULT_READ_BUDGET);
        a.merge(&b, 3);
        assert_eq!(a.len(), 2);
        let dog = a.get(&seed(1)).unwrap();
        assert_eq!(dog.activation, 0.95);
        assert!(dog.hits >= 2);
        assert_eq!(dog.last_touched_turn, 3);
        assert!(a.get(&seed(2)).is_some());
    }

    #[test]
    fn empty_recall_yields_empty_set() {
        let cs = ContextSet::from_ask(&ask_with(vec![]), 0, 0, DEFAULT_READ_BUDGET);
        assert!(cs.is_empty());
    }

    fn seedn(n: u32) -> Seed {
        let mut raw = [0u8; 32];
        raw[..4].copy_from_slice(&n.to_le_bytes());
        Seed::from_bytes(raw)
    }

    fn node_n(n: u32, activation: f32) -> ActivatedNode {
        ActivatedNode {
            seed: seedn(n),
            content: format!("n{}", n),
            activation,
            confidence: Confidence::Unverified,
            path: Vec::new(),
            depth: 1,
        }
    }

    #[test]
    fn session_context_accumulates_across_turns_warm_start() {
        let mut sc = SessionContext::new();
        let t1 = ContextSet::from_ask(&ask_with(vec![node(1, "dog", 0.9, 0), node(2, "wolf", 0.5, 1)]), 1, 0, DEFAULT_READ_BUDGET);
        sc.merge_prompt(&t1, 1);
        sc.evict(1);
        assert!(sc.contains_label("dog"));
        assert!(sc.contains_label("wolf"));
        let len_after_t1 = sc.len();
        // turn 2 warm-starts from held dog+wolf, adds cat and re-hits wolf
        let t2 = ContextSet::from_ask(&ask_with(vec![node(2, "wolf", 0.6, 1), node(3, "cat", 0.4, 1)]), 2, 0, DEFAULT_READ_BUDGET);
        sc.merge_prompt(&t2, 2);
        sc.evict(2);
        assert!(sc.contains_label("dog"));
        assert!(sc.contains_label("wolf"));
        assert!(sc.contains_label("cat"));
        assert!(sc.len() >= len_after_t1);
        let ws = sc.warm_start_labels(3);
        assert!(ws.contains(&"dog".to_string()));
    }

    #[test]
    fn session_context_marathon_stays_bounded() {
        let cap = 50usize;
        let mut sc = SessionContext::with_cap(cap);
        for turn in 1..=200u32 {
            let mut nodes = Vec::new();
            for k in 0..40u32 {
                let id = turn * 1000 + k;
                nodes.push(node_n(id, 0.1 + (k as f32) * 0.01));
            }
            let cs = ContextSet::from_ask(&ask_with(nodes), turn as u64, 0, usize::MAX);
            sc.merge_prompt(&cs, turn as u64);
            sc.evict(turn as u64);
            assert!(sc.len() <= cap, "len {} exceeded cap {} at turn {}", sc.len(), cap, turn);
        }
        assert!(sc.len() <= cap);
    }

    #[test]
    fn session_context_evicts_stale_low_relevance_first() {
        let mut sc = SessionContext::with_cap(3);
        let old = ContextSet::from_ask(&ask_with(vec![node(1, "stale", 0.15, 3)]), 0, 0, usize::MAX);
        sc.merge_prompt(&old, 0);
        let fresh = ContextSet::from_ask(&ask_with(vec![node_n(10, 0.9), node_n(11, 0.85), node_n(12, 0.8), node_n(13, 0.75)]), 20, 0, usize::MAX);
        sc.merge_prompt(&fresh, 20);
        sc.evict(20);
        assert_eq!(sc.len(), 3);
        assert!(!sc.contains_label("stale"));
    }

    #[test]
    fn session_context_empty_by_default() {
        let sc = SessionContext::new();
        assert!(sc.is_empty());
        assert_eq!(sc.cap(), MAX_SESSION_CONTEXT_NODES);
    }
}
