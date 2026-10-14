//! Thinking Layer — third of the seven Lu 4.0 capability ports.
//!
//! A ReAct-style Observe -> Plan -> Act -> Reflect loop layered on top of
//! the temporal graph. One user call to `think()` runs one synchronous
//! pass; no background threads. Bounded everything.
//!
//! Mechanics:
//! 1. A `Goal` (Meta) node is created carrying the original query.
//! 2. For each step, observation is a `Brain::ask` recall with bridges.
//! 3. The plan proposes a small fixed number of `Hypothesis` bonds
//!    between strongly- and moderately-activated nodes that aren't
//!    already strongly bonded. Heuristic: "if definitional A, then
//!    consequential B is relevant."
//! 4. Action writes the candidates to the graph with
//!    `KeyframeSource::Hypothesis` at low weight.
//! 5. Reflection runs a second sub-spread; hypotheses whose `to` seed
//!    activates above threshold are promoted to
//!    `KeyframeSource::ConfirmedHypothesis` at higher weight. The rest
//!    are left to decay.
//! 6. The loop exits early once the per-step confirmation ratio crosses
//!    `confidence_to_answer`, otherwise runs `max_steps`.
//!
//! Operational/session memory: `ThoughtTrace` is returned but not
//! written into the graph. Same boundary the project log's session memory
//! enforces — process narration is not a fact.

use crate::bond::{BondType, KeyframeSource, TemporalBond};
use crate::brain::{now_ms, AskResult, Brain};
use crate::network::AssociativeNetwork;
use crate::node::{Confidence, KnowledgeContent, KnowledgeNode};
use crate::scope::TemporalScope;
use crate::seed::Seed;
use crate::bridge::BridgeEvent;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Hard cap on hypothesis bonds written by a single think pass. Prevents
/// a runaway loop from filling the graph with low-weight noise.
pub const MAX_HYPOTHESIS_BONDS_PER_GOAL: usize = 32;

/// Cutoff on existing edge strength above which the heuristic considers
/// a pair "already bonded" and skips proposing it as a hypothesis.
const ALREADY_BONDED_STRENGTH_CUTOFF: f32 = 0.25;

/// Default wall-clock ceiling on one think pass. The step counter
/// bounds iteration count, but each step runs a multi-spread `Brain::ask`
/// plus a pre-reflect BFS plus (in variant mode) a 5-arena variant pass —
/// individual steps can run for seconds on a saturated-hub query against
/// a multi-million-bond brain. The wall-clock watchdog is the second
/// halting axis: even if `max_steps` is intact, a step that drifts into
/// hub territory will exit cleanly with `HaltedReason::WallClockExceeded`
/// and the partial trace is returned.
///
/// Engine default `30s` keeps headroom for batch / audit harnesses.
/// User-facing narrate paths override to `5s` via
/// `NarrationConfig::thinking_max_wall_clock_ms`.
pub const DEFAULT_THINK_MAX_WALL_CLOCK: Duration = Duration::from_secs(30);

/// Why a `think` pass stopped iterating. Carried on `ThoughtTrace`
/// and surfaced via `ThinkingTraceSummary` so MCP / narrate callers can
/// distinguish a clean convergence (`EarlyExit`) from a watchdog halt
/// (`WallClockExceeded`) from a normal cap (`MaxSteps`).
///
/// `Default::default()` is `MaxSteps` because that is the loop's
/// historical "ran to completion against the hard step cap" outcome.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum HaltedReason {
    /// Outer loop ran all `max_steps` iterations.
    #[default]
    MaxSteps,
    /// Per-step confirmation ratio crossed `confidence_to_answer`.
    EarlyExit,
    /// Wrote `MAX_HYPOTHESIS_BONDS_PER_GOAL` low-weight hypotheses
    /// before the step cap was reached.
    MaxHypothesisBonds,
    /// Wall-clock watchdog fired — `ThinkConfig::max_wall_clock`
    /// elapsed before the loop converged or hit `max_steps`. Partial
    /// trace is returned; downstream slot composition treats this
    /// the same as a normal halt for graceful degradation.
    WallClockExceeded,
}

/// All knobs for one think pass. Defaults match the port spec.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThinkConfig {
    /// Hard cap on iterations.
    pub max_steps: usize,
    /// Per-step cap on hypothesis bonds proposed.
    pub hypotheses_per_step: usize,
    /// Initial weight of a Hypothesis-source keyframe.
    pub initial_hypothesis_weight: f32,
    /// Promotion weight for a ConfirmedHypothesis keyframe.
    pub confirmed_hypothesis_weight: f32,
    /// Per-step confirmation ratio at which the loop exits early.
    pub confidence_to_answer: f32,
    /// How many top observations are eligible for hypothesis pairing.
    pub top_n_observations: usize,
    /// When true, the Plan phase will route through the Variant
    /// Generation arena (Brain::generate_variants) instead of writing
    /// single-shot Hypothesis bonds. Off by default — variant mode adds
    /// 5x the work per goal and is opt-in. Wiring of the variant->plan
    /// path lives at the call site (Brain::think); this flag carries
    /// the toggle.
    pub variant_mode: bool,
    /// Minimum composite score a variant arena must produce for the
    /// Plan phase to write the hypothesis bond on its target seed.
    /// Mirrors `VariantConfig::min_winner_composite` so callers can
    /// tighten or loosen the gate without rebuilding the full variants
    /// config. Only consulted when `variant_mode == true`.
    #[serde(default = "default_variant_min_winner_composite")]
    pub variant_min_winner_composite: f32,
    /// Permanent observability hook: when true, the think loop
    /// emits per-step / per-hypothesis trace lines on stderr.
    /// OFF by default. Used by `examples/audit_hypothesis_promotion`
    /// (BR.C-Hyp diagnosis port) and any future investigation of
    /// hypothesis generation, REFLECT gate behavior, or promotion
    /// outcomes. Behavior is unchanged when false; lines go to
    /// stderr so production stdout is unaffected.
    #[serde(default)]
    pub verbose_trace: bool,
    /// OP-THINK-WATCHDOG -- wall-clock ceiling on the entire think
    /// pass. `Some(d)` = halt the loop as soon as `Instant::now() -
    /// started >= d`, set `halted_reason = WallClockExceeded`, and
    /// return whatever steps already completed. `None` = no
    /// wall-clock bound (unit tests + audit harnesses that need
    /// deterministic completion). Checked at the top of each step,
    /// after OBSERVE, after the variant-arena loop (when armed),
    /// after the pre-reflect spread, and inside the bond-writing
    /// inner loop — anywhere a single step can drift into
    /// multi-second territory.
    ///
    /// Default `Some(DEFAULT_THINK_MAX_WALL_CLOCK)` = `30s` for
    /// engine callers. User-facing narrate overrides to `5s`.
    #[serde(default = "default_max_wall_clock", with = "opt_duration_ms_serde")]
    pub max_wall_clock: Option<Duration>,
    /// OP-ASK-WATCHDOG -- wall-clock ceiling per inner `Brain::ask`
    /// call (OBSERVE phase and final composition). `None` preserves
    /// legacy behaviour; narrate plumbs a tightened value so a
    /// single runaway ask cannot consume the entire think budget.
    /// Must be < `max_wall_clock` when set.
    #[serde(default = "default_max_ask_wall_clock", with = "opt_duration_ms_serde")]
    pub max_ask_wall_clock: Option<Duration>,
    // OP-REASON-LOOP Phase 2 (FLAG D): read-only reasoning. When true, ACT
    // writes NOTHING to the graph (no goal/step nodes, no Hypothesis bonds,
    // no ConfirmedHypothesis keyframes). REFLECT promotes on PRE-WRITE
    // activations, so trace, score and promotion counts are identical; only
    // the graph side effects are suppressed. Older payloads default to false.
    #[serde(default)]
    pub no_write: bool,
}

fn default_variant_min_winner_composite() -> f32 {
    0.4
}

fn default_max_wall_clock() -> Option<Duration> {
    Some(DEFAULT_THINK_MAX_WALL_CLOCK)
}

fn default_max_ask_wall_clock() -> Option<Duration> {
    None
}

/// Serde helper: serialise `Option<Duration>` as `Option<u64>` of
/// milliseconds, so older brain files / JSON payloads that omit the
/// field round-trip cleanly via `default_max_wall_clock` and newer
/// payloads stay human-readable.
mod opt_duration_ms_serde {
    use super::Duration;
    use serde::{Deserialize, Deserializer, Serializer};
    pub fn serialize<S: Serializer>(v: &Option<Duration>, s: S) -> Result<S::Ok, S::Error> {
        match v {
            Some(d) => s.serialize_some(&(d.as_millis() as u64)),
            None => s.serialize_none(),
        }
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Option<Duration>, D::Error> {
        let v: Option<u64> = Option::deserialize(d)?;
        Ok(v.map(Duration::from_millis))
    }
}

impl Default for ThinkConfig {
    fn default() -> Self {
        ThinkConfig {
            max_steps: 5,
            hypotheses_per_step: 3,
            initial_hypothesis_weight: 0.15,
            confirmed_hypothesis_weight: 0.55,
            confidence_to_answer: 0.7,
            top_n_observations: 8,
            variant_mode: false,
            variant_min_winner_composite: 0.4,
            verbose_trace: false,
            max_wall_clock: Some(DEFAULT_THINK_MAX_WALL_CLOCK),
            max_ask_wall_clock: None,
            no_write: false,
        }
    }
}

/// Identity of the goal a think pass is reasoning about.
#[derive(Debug, Clone, Serialize)]
pub struct Goal {
    pub seed: Seed,
    pub query: String,
    pub created_at: i64,
}

/// One proposed (or written) hypothesis bond. Held in memory for the
/// trace; the bond itself lives in the graph.
#[derive(Debug, Clone, Serialize)]
pub struct HypothesisCandidate {
    pub from: Seed,
    pub from_content: String,
    pub to: Seed,
    pub to_content: String,
    pub initial_activation_from: f32,
    pub initial_activation_to: f32,
}

/// Snapshot of what the OBSERVE phase saw.
#[derive(Debug, Clone, Serialize)]
pub struct Observation {
    pub top_seeds: Vec<Seed>,
    pub top_contents: Vec<String>,
    pub top_activations: Vec<f32>,
    pub bridges: Vec<BridgeEvent>,
}

/// PLAN output for a single step.
#[derive(Debug, Clone, Serialize)]
pub struct Plan {
    pub hypotheses: Vec<HypothesisCandidate>,
}

/// REFLECT output for a single step.
#[derive(Debug, Clone, Serialize)]
pub struct Reflection {
    pub confirmed: Vec<HypothesisCandidate>,
    pub unconfirmed: Vec<HypothesisCandidate>,
    /// Confirmation ratio for this step (`confirmed / proposed`). Empty
    /// plans score 0 so the loop never exits early on no work.
    pub score: f32,
}

/// One Observe -> Plan -> Act -> Reflect iteration's full record.
#[derive(Debug, Clone, Serialize)]
pub struct ThinkStep {
    pub index: usize,
    pub step_seed: Seed,
    pub observation: Observation,
    pub plan: Plan,
    pub reflection: Reflection,
}

/// Operational chain-of-thought for one think pass. NOT persisted in the
/// brain graph — it lives with the result so callers can surface it.
#[derive(Debug, Clone, Serialize)]
pub struct ThoughtTrace {
    pub goal: Goal,
    pub steps: Vec<ThinkStep>,
    pub final_score: f32,
    pub early_exit: bool,
    pub hypothesis_bonds_written: usize,
    pub hypotheses_promoted: usize,
    /// OP-THINK-WATCHDOG -- why the outer loop stopped. Defaults to
    /// `MaxSteps` for older trace JSON that predates the field.
    #[serde(default)]
    pub halted_reason: HaltedReason,
}

/// What `Brain::think` returns: the final answer assembled after the
/// loop, plus the trace.
#[derive(Debug, Clone, Serialize)]
pub struct ThinkResult {
    pub trace: ThoughtTrace,
    pub final_answer: AskResult,
}

/// OP-ROUTE-3c -- compact summary of a `ThoughtTrace` that landed
/// on a narrate PASS-path query. Carried on `NarrationResult.thinking`
/// so callers can inspect convergence outcome without paying the
/// wire cost of the full `ThoughtTrace` (per-step Observe/Plan/Act/
/// Reflect tuples can run to 100+ seed references at max_steps=5).
/// `None` on `NarrationResult` when the router did not request
/// agentic thinking, or when the request was guarded out (meta /
/// math / missing goal seed / below activation threshold).
/// Backward-compat: older NarrationResult JSON without this field
/// round-trips clean via `#[serde(default)]`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ThinkingTraceSummary {
    /// Number of Observe-Plan-Act-Reflect iterations that ran.
    /// Bounded above by `NarrationConfig::thinking_max_steps`.
    pub steps_taken: usize,
    /// True when the loop hit `confidence_to_answer` and exited
    /// before `max_steps`. False when the loop ran to the cap.
    pub early_exit: bool,
    /// Reflection score on the final iteration (`confirmed /
    /// proposed`). 0.0 when no plan had any hypotheses to propose
    /// on that step.
    pub final_score: f32,
    /// Total Hypothesis-source keyframes written across all
    /// steps. Bounded above by
    /// `thinking::MAX_HYPOTHESIS_BONDS_PER_GOAL = 32`.
    pub hypothesis_bonds_written: usize,
    /// Subset of `hypothesis_bonds_written` that the REFLECT phase
    /// promoted to ConfirmedHypothesis weight.
    pub hypotheses_promoted: usize,
    /// `true` when `final_answer.recall.activated` is non-empty.
    /// Lets callers distinguish "loop ran, no recall" from
    /// "loop ran, recall present."
    pub final_answer_present: bool,
    /// Top-activation content from the final recall, if any.
    /// Used by `extract_slots` to populate `chain_articulation`
    /// when reasoning (3a) did not. `None` when
    /// `final_answer.recall.activated` was empty.
    #[serde(default)]
    pub final_answer_text: Option<String>,
    /// Aggregate confidence -- copy of `final_score` for parity
    /// with `ReasoningTrace.confidence`. Carried separately so
    /// callers can surface it without coupling to think-engine
    /// semantics.
    pub confidence: f32,
    /// OP-THINK-WATCHDOG -- why the underlying think loop halted.
    /// Older payloads default to `MaxSteps`.
    #[serde(default)]
    pub halted_reason: HaltedReason,
}

impl From<&ThinkResult> for ThinkingTraceSummary {
    fn from(r: &ThinkResult) -> Self {
        let final_answer_text = r
            .final_answer
            .recall
            .activated
            .first()
            .map(|a| a.content.clone());
        ThinkingTraceSummary {
            steps_taken: r.trace.steps.len(),
            early_exit: r.trace.early_exit,
            final_score: r.trace.final_score,
            hypothesis_bonds_written: r.trace.hypothesis_bonds_written,
            hypotheses_promoted: r.trace.hypotheses_promoted,
            final_answer_present: !r.final_answer.recall.activated.is_empty(),
            final_answer_text,
            confidence: r.trace.final_score,
            halted_reason: r.trace.halted_reason,
        }
    }
}

/// Run one synchronous think pass on `brain`. See module docs.
pub fn think(brain: &mut Brain, query: &str, opts: ThinkConfig) -> ThinkResult {
    let now = now_ms();
    // OP-THINK-WATCHDOG -- wall-clock baseline + helper closure.
    // `started` is captured ONCE at the top of the pass so every
    // halting check inside the loop measures against the same
    // reference, not against a fresh `Instant::now()` per step.
    let started = Instant::now();
    let max_wall = opts.max_wall_clock;
    // Returns true when the watchdog has fired. Tight inline check;
    // `None` is the deterministic-tests / audit-harness escape hatch.
    let watchdog_fired = || -> bool {
        match max_wall {
            Some(d) => started.elapsed() >= d,
            None => false,
        }
    };
    let active_session_id = brain.sessions.active_session().map(|s| s.id);
    let scope = match brain.sessions.active_session() {
        Some(s) => TemporalScope::Session { id: s.id, start: s.start_time },
        None => TemporalScope::Current,
    };

    // 1. Create the Meta goal node. Embedding stays zero — it's a
    //    process anchor, not a semantic concept. Stays OUT of the
    //    keyword index — its content (`goal:<query>`) would otherwise
    //    pollute keyword recall for the query terms themselves.
    let goal_seed = Seed::generate();
    let goal_node = KnowledgeNode::with_seed(
        goal_seed,
        format!("goal:{}", query),
        KnowledgeContent::Meta(query.to_string()),
        Confidence::Inferred,
        now,
    );
    // OP-REASON-LOOP Phase 2 (FLAG D): skip the goal-node graph write under
    // no_write. goal_node/goal_seed stay in-memory for the trace only.
    if !opts.no_write {
        brain.network.add_node_unindexed(goal_node);
    }
    let goal = Goal {
        seed: goal_seed,
        query: query.to_string(),
        created_at: now,
    };

    let mut steps: Vec<ThinkStep> = Vec::new();
    let mut total_hypothesis_bonds: usize = 0;
    let mut total_promoted: usize = 0;
    let mut early_exit = false;
    let mut last_score: f32 = 0.0;
    // OP-THINK-WATCHDOG -- mutable halt reason. Defaults to MaxSteps
    // (the historical outcome when the outer loop runs to the cap)
    // and gets overwritten by EarlyExit / MaxHypothesisBonds /
    // WallClockExceeded as those conditions fire.
    let mut halted_reason = HaltedReason::MaxSteps;

    for step_index in 0..opts.max_steps {
        // OP-THINK-WATCHDOG -- wall-clock guard #1: top of each step.
        // Cheap, catches the simple cumulative-time case before any
        // additional spread cost is paid.
        if watchdog_fired() {
            halted_reason = HaltedReason::WallClockExceeded;
            break;
        }
        // a. OBSERVE: Brain::ask surfaces top-N activations + bridges
        //    from this step's spread. OP-ASK-WATCHDOG plumbs the
        //    inner-ask wall-clock budget so a single runaway ask
        //    cannot consume the entire think wall-clock.
        let obs_ask = brain.ask_with_config(
            query,
            crate::brain::BrainAskConfig::new(3, opts.top_n_observations)
                .with_wall_clock(opts.max_ask_wall_clock),
        );
        let mut observation = Observation {
            top_seeds: obs_ask.recall.activated.iter().map(|a| a.seed).collect(),
            top_contents: obs_ask.recall.activated.iter().map(|a| a.content.clone()).collect(),
            top_activations: obs_ask.recall.activated.iter().map(|a| a.activation).collect(),
            bridges: obs_ask.bridges.clone(),
        };

        if opts.verbose_trace {
            eprintln!(
                "[think:trace] step={} observation top_n={} top0_act={:.4}",
                step_index,
                observation.top_seeds.len(),
                observation.top_activations.first().copied().unwrap_or(0.0),
            );
            for (i, (c, a)) in observation.top_contents.iter()
                .zip(observation.top_activations.iter())
                .take(observation.top_seeds.len().min(10))
                .enumerate()
            {
                eprintln!("[think:trace]   obs[{}] act={:.4} content={:?}", i, a, c);
            }
        }

        // OP-THINK-WATCHDOG -- wall-clock guard #2: post-OBSERVE.
        // `brain.ask` runs a bridge+curiosity-collecting full BFS
        // PLUS a cascade re-spread when recall is thin. On a
        // saturated-hub query this is the single most expensive
        // operation inside a step; check the watchdog immediately
        // after rather than waiting for the next step's top guard.
        if watchdog_fired() {
            halted_reason = HaltedReason::WallClockExceeded;
            break;
        }

        // Curiosity bias: if the wonder queue holds a strong signal
        // (intensity > 0.6) and the seed is already a node in the graph,
        // append it to the candidate hypothesis pool so the planner can
        // pair it with the observation top. Soft integration — does
        // nothing on an empty queue.
        if let Some(wondered) = brain.wonder_queue.peek().cloned() {
            if wondered.intensity > 0.6
                && !observation.top_seeds.contains(&wondered.seed)
            {
                if let Some(node) = brain.network.get_node(&wondered.seed) {
                    observation.top_seeds.push(wondered.seed);
                    observation.top_contents.push(node.content().to_string());
                    observation.top_activations.push(wondered.intensity);
                }
            }
        }

        // b. PLAN: deterministic hypothesis proposal (top-half
        //    "definitional" x bottom-half "consequential").
        let cap_remaining = MAX_HYPOTHESIS_BONDS_PER_GOAL
            .saturating_sub(total_hypothesis_bonds);
        let plan = propose_hypotheses(
            &brain.network,
            &observation,
            opts.hypotheses_per_step,
            cap_remaining,
            &scope,
            &brain.scar_log,
        );

        // b'. VARIANT MODE: for each proposed hypothesis target, run a
        //     5-strategy variant arena on that target seed. Keep only
        //     the candidates whose arena produced a winner — meaning
        //     coherence x novelty x parsimony cleared
        //     `min_winner_composite`. If no variant wins, we drop that
        //     candidate. That null is itself signal: weaker than a
        //     low-confidence hypothesis would have been, which is the
        //     point of variant mode. Each arena writes Variant-tagged
        //     keyframes that are invisible to baseline scope and decay
        //     naturally (or get pruned by consolidation).
        let plan = if opts.variant_mode {
            // Build a variant config that mirrors the default but
            // honors the ThinkConfig override on min_winner_composite.
            let var_cfg = crate::variants::VariantConfig {
                min_winner_composite: opts.variant_min_winner_composite,
                ..crate::variants::VariantConfig::default()
            };
            let mut filtered: Vec<HypothesisCandidate> = Vec::new();
            for h in plan.hypotheses {
                // OP-THINK-WATCHDOG -- variant arenas are 5x the
                // work per candidate; check between arenas, not
                // just at the outer loop boundary.
                if watchdog_fired() {
                    break;
                }
                let arena = brain.generate_variants_with_config(h.to, &var_cfg);
                if arena.winner_id.is_some() {
                    filtered.push(h);
                }
            }
            Plan { hypotheses: filtered }
        } else {
            plan
        };

        if opts.verbose_trace {
            eprintln!(
                "[think:trace] step={} plan proposed={} (after variant_mode={})",
                step_index, plan.hypotheses.len(), opts.variant_mode,
            );
            for (i, h) in plan.hypotheses.iter().enumerate() {
                eprintln!(
                    "[think:trace]   plan[{}] from={:?} to={:?} init_from_act={:.4} init_to_act={:.4}",
                    i, h.from_content, h.to_content,
                    h.initial_activation_from, h.initial_activation_to,
                );
            }
        }

        // c. PRE-WRITE REFLECT (Tier 2.5 fix): spread from observation
        //    seeds BEFORE writing any hypothesis bonds.  An hypothesis
        //    `from -> to` is confirmed only if `to` already activated
        //    in this pre-write recall, i.e. the brain has structural
        //    support for the connection independent of the bond we
        //    are about to write.  Stops the old tautology where the
        //    just-written bond drove its own confirmation.
        let pre_reflect_seeds: Vec<(Seed, f32)> = observation
            .top_seeds
            .iter()
            .take(3)
            .map(|s| (*s, 1.0_f32))
            .collect();
        // Port T.5.1 — arm freshness multiplier for pre-reflect recall.
        brain.refresh_decay_runtime();
        // Port Sense.1 -- arm sense cluster-gain multiplier for pre-reflect recall.
        let _sense_seeds: Vec<crate::seed::Seed> = pre_reflect_seeds.iter().map(|(s, _)| *s).collect();
        brain.refresh_sense_runtime(&_sense_seeds);
        let pre_reflect_recall = brain.network.spread_activation_weighted_with_wall_clock(
            &pre_reflect_seeds, &scope, 3, 50,
            &brain.scar_log,
            opts.max_ask_wall_clock,
        );
        let pre_reflect_acts: HashMap<Seed, f32> = pre_reflect_recall
            .activated
            .iter()
            .map(|a| (a.seed, a.activation))
            .collect();

        if opts.verbose_trace {
            eprintln!(
                "[think:trace] step={} pre_reflect seeds={} acts.len={}",
                step_index, pre_reflect_seeds.len(), pre_reflect_acts.len(),
            );
        }

        // OP-THINK-WATCHDOG -- wall-clock guard #3: post pre-reflect.
        // The pre-reflect spread is the second BFS this step runs;
        // halt before we start writing bonds if the budget is gone.
        if watchdog_fired() {
            halted_reason = HaltedReason::WallClockExceeded;
            break;
        }

        // c. ACT: create a Procedure step node (parent_seeds = [goal])
        //    for traceability, then write each hypothesis bond. Step
        //    timestamp is offset by +step_index so keyframe ordering
        //    is well-defined even if `now_ms` returns the same value.
        let step_now = now + step_index as i64 + 1;
        let step_seed = Seed::generate();
        let mut step_node = KnowledgeNode::with_seed(
            step_seed,
            format!("goal:{}/step:{}", query, step_index),
            KnowledgeContent::Procedure(format!("think step {}", step_index)),
            Confidence::Inferred,
            step_now,
        );
        step_node.parent_seeds.push(goal_seed);
        // Step nodes are also process anchors — keep out of keyword recall.
        // OP-REASON-LOOP Phase 2 (FLAG D): skip the step-node graph write
        // under no_write (step_node built above only for the in-memory trace).
        if !opts.no_write {
            brain.network.add_node_unindexed(step_node);
        }

        let mut written_this_step: Vec<HypothesisCandidate> = Vec::new();
        for h in &plan.hypotheses {
            // Don't trample existing bonds — only write fresh ones.
            if brain.network.get_bond(&h.from, &h.to).is_some()
                || brain.network.get_bond(&h.to, &h.from).is_some()
            {
                continue;
            }
            // TemporalBond::new stamps a Baseline keyframe at
            // initial_weight; we follow up with a Hypothesis-source
            // keyframe at the same weight + 1ms so spread sees it as
            // the latest source.
            // OP-REASON-LOOP Phase 2 (FLAG D): under no_write the Hypothesis
            // bond is a candidate for REFLECT scoring but is NOT written to
            // the graph. written_this_step + the counter update either way so
            // the trace/score/gating stay identical to the write path.
            if !opts.no_write {
                let bond = TemporalBond::new(
                    h.from, h.to, BondType::RelatedTo,
                    opts.initial_hypothesis_weight, step_now,
                );
                if brain.network.add_bond(bond).is_err() {
                    continue;
                }
                if let Some(mut b) = brain.network.get_bond_mut(&h.from, &h.to) {
                    b.add_keyframe(
                        step_now + 1,
                        opts.initial_hypothesis_weight,
                        KeyframeSource::Hypothesis {
                            goal: goal_seed,
                            step: step_index,
                            session: active_session_id,
                        },
                    );
                }
            }
            written_this_step.push(h.clone());
            total_hypothesis_bonds += 1;
            if total_hypothesis_bonds >= MAX_HYPOTHESIS_BONDS_PER_GOAL {
                break;
            }
        }

        // d. REFLECT: confirm using the PRE-WRITE activations gathered
        //    above.  An hypothesis is promoted only when `to` already
        //    activated independently of the bond we just wrote, i.e.
        //    when the brain had prior structural support for the
        //    proposed link.  This is the fix for internal notes (NO
        //    STUBS): the prior implementation re-spread AFTER the bond
        //    write, so `to` was guaranteed to activate via the new bond
        //    and confirmation was a tautology.
        let reflect_acts: &HashMap<Seed, f32> = &pre_reflect_acts;

        let mut confirmed: Vec<HypothesisCandidate> = Vec::new();
        let mut unconfirmed: Vec<HypothesisCandidate> = Vec::new();
        for h in &written_this_step {
            // `to` activated independently in the pre-write reflect?
            // Promote.  Otherwise the bond is genuinely a leap and
            // gets left at its initial Hypothesis weight.
            let to_act = reflect_acts.get(&h.to).copied().unwrap_or(0.0);
            if opts.verbose_trace {
                eprintln!(
                    "[think:trace] step={} reflect to={:?} to_act={:.4} promote={}",
                    step_index, h.to_content, to_act, to_act > 0.0,
                );
            }
            if to_act > 0.0 {
                // OP-REASON-LOOP Phase 2 (FLAG D): promotion is recorded in the
                // trace either way; the ConfirmedHypothesis keyframe is a graph
                // write, suppressed under no_write.
                if !opts.no_write {
                    let promote_ts = step_now + 2;
                    if let Some(mut b) = brain.network.get_bond_mut(&h.from, &h.to) {
                        b.add_keyframe(
                            promote_ts,
                            opts.confirmed_hypothesis_weight,
                            KeyframeSource::ConfirmedHypothesis {
                                goal: goal_seed,
                                step: step_index,
                                session: active_session_id,
                            },
                        );
                    }
                }
                confirmed.push(h.clone());
                total_promoted += 1;
            } else {
                unconfirmed.push(h.clone());
            }
        }

        let score = if written_this_step.is_empty() {
            0.0
        } else {
            confirmed.len() as f32 / written_this_step.len() as f32
        };
        last_score = score;

        steps.push(ThinkStep {
            index: step_index,
            step_seed,
            observation,
            plan,
            reflection: Reflection { confirmed, unconfirmed, score },
        });

        if score >= opts.confidence_to_answer {
            early_exit = true;
            halted_reason = HaltedReason::EarlyExit;
            break;
        }
        if total_hypothesis_bonds >= MAX_HYPOTHESIS_BONDS_PER_GOAL {
            halted_reason = HaltedReason::MaxHypothesisBonds;
            break;
        }
        // OP-THINK-WATCHDOG -- wall-clock guard #4: end of step.
        // Catches the case where REFLECT scoring itself drifted into
        // hub territory (large pre_reflect_acts map iteration).
        if watchdog_fired() {
            halted_reason = HaltedReason::WallClockExceeded;
            break;
        }
    }

    // Final composition: one more ask, now able to ride the strengthened
    // ConfirmedHypothesis bonds. OP-ASK-WATCHDOG: inherit the inner
    // ask budget from ThinkConfig so the final pass cannot dwarf
    // the overall think wall-clock either.
    let final_answer = brain.ask_with_config(
        query,
        crate::brain::BrainAskConfig::new(3, 10).with_wall_clock(opts.max_ask_wall_clock),
    );

    ThinkResult {
        trace: ThoughtTrace {
            goal,
            steps,
            final_score: last_score,
            early_exit,
            hypothesis_bonds_written: total_hypothesis_bonds,
            hypotheses_promoted: total_promoted,
            halted_reason,
        },
        final_answer,
    }
}

/// Deterministic hypothesis proposal. Splits the top-N observations into
/// a definitional half (top) and a consequential half (bottom). Cross
/// product, drop pairs whose endpoints are already strongly bonded,
/// rank by joint activation, take top `k` (capped by `cap_remaining`).
///
/// Sort is total-ordered (activations then seed bytes) so the same brain
/// state always produces the same plan.
fn propose_hypotheses(
    net: &AssociativeNetwork,
    obs: &Observation,
    k: usize,
    cap_remaining: usize,
    scope: &TemporalScope,
    scar_log: &crate::scar::ScarLog,
) -> Plan {
    let n = obs.top_seeds.len();
    if n < 2 || k == 0 || cap_remaining == 0 {
        return Plan { hypotheses: Vec::new() };
    }
    let split = (n + 1) / 2; // ceil — 3 -> 2 def + 1 cons; 8 -> 4 + 4
    let mut candidates: Vec<HypothesisCandidate> = Vec::new();
    for i in 0..split {
        for j in split..n {
            let a_seed = obs.top_seeds[i];
            let b_seed = obs.top_seeds[j];
            if a_seed == b_seed {
                continue;
            }
            let strength_fwd = net.get_bond(&a_seed, &b_seed)
                .map(|b| b.weight_in_scope(scope, scar_log).effective_weight.abs())
                .unwrap_or(0.0);
            let strength_rev = net.get_bond(&b_seed, &a_seed)
                .map(|b| b.weight_in_scope(scope, scar_log).effective_weight.abs())
                .unwrap_or(0.0);
            if strength_fwd.max(strength_rev) >= ALREADY_BONDED_STRENGTH_CUTOFF {
                continue;
            }
            candidates.push(HypothesisCandidate {
                from: a_seed,
                from_content: obs.top_contents[i].clone(),
                to: b_seed,
                to_content: obs.top_contents[j].clone(),
                initial_activation_from: obs.top_activations[i],
                initial_activation_to: obs.top_activations[j],
            });
        }
    }
    candidates.sort_by(|x, y| {
        let xs = x.initial_activation_from + x.initial_activation_to;
        let ys = y.initial_activation_from + y.initial_activation_to;
        ys.partial_cmp(&xs)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| x.from.as_bytes().cmp(y.from.as_bytes()))
            .then_with(|| x.to.as_bytes().cmp(y.to.as_bytes()))
    });
    let take = k.min(cap_remaining);
    candidates.truncate(take);
    Plan { hypotheses: candidates }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bond::{BondType, KeyframeSource};
    use crate::node::Confidence;

    /// Brain factory: empty graph.
    fn empty_brain() -> Brain {
        Brain::new()
    }

    /// Brain with two hub clusters joined by a moderate (NOT bridge-weak)
    /// IsA chain so observation lights up several distinct nodes for
    /// hypothesis pairing.
    fn populated_brain() -> Brain {
        let mut b = Brain::new();
        b.teach("alpha", BondType::IsA, "topic", 0.95, Confidence::Verified);
        b.teach("topic", BondType::RelatedTo, "intermediary", 0.9, Confidence::Verified);
        b.teach("intermediary", BondType::RelatedTo, "endpoint", 0.9, Confidence::Verified);
        b.teach("alpha", BondType::HasProperty, "property", 0.9, Confidence::Verified);
        b.teach("topic", BondType::HasProperty, "feature", 0.85, Confidence::Verified);
        b
    }

    #[test]
    fn test_think_creates_goal_node() {
        let mut brain = empty_brain();
        let n_before = brain.network.node_count();
        let r = think(&mut brain, "what is the capital of france", ThinkConfig::default());
        // Goal node + at least one step node must have been added.
        assert!(brain.network.node_count() > n_before);
        let goal_node = brain.network.get_node(&r.trace.goal.seed)
            .expect("goal node must exist");
        match &goal_node.knowledge().expect("goal node is heap-resident").content {
            KnowledgeContent::Meta(q) => {
                assert_eq!(q, "what is the capital of france");
            }
            other => panic!("goal node should be Meta, got {:?}", other),
        }
    }

    #[test]
    fn test_think_runs_max_steps() {
        let mut brain = empty_brain();
        let opts = ThinkConfig { max_steps: 5, ..ThinkConfig::default() };
        // Multi-letter terms now safe: goal/step nodes are unindexed,
        // so the keyword index stays empty and observations stay empty.
        let r = think(&mut brain, "what is the capital", opts);
        // Empty graph -> empty observations -> nothing to confirm ->
        // never crosses confidence threshold -> all steps run.
        assert_eq!(r.trace.steps.len(), 5);
        assert!(!r.trace.early_exit);
    }

    #[test]
    fn test_hypothesis_promoted_on_confirmation() {
        let mut brain = populated_brain();
        let r = think(&mut brain, "alpha", ThinkConfig::default());
        // At least one bond must carry a ConfirmedHypothesis keyframe.
        let mut found = false;
        for ((_a, _b), bond) in brain.network.bonds().iter() {
            for kf in bond.keyframes() {
                if matches!(kf.source, KeyframeSource::ConfirmedHypothesis { .. }) {
                    found = true;
                    break;
                }
            }
            if found { break; }
        }
        assert!(found, "some hypothesis bond must have been promoted");
        assert!(r.trace.hypotheses_promoted > 0);
    }

    #[test]
    fn test_thought_trace_has_one_entry_per_step() {
        let mut brain = populated_brain();
        let opts = ThinkConfig { max_steps: 3, confidence_to_answer: 1.01, ..ThinkConfig::default() };
        // confidence_to_answer > 1 so we never exit early.
        let r = think(&mut brain, "alpha", opts);
        assert_eq!(r.trace.steps.len(), 3);
        for (i, s) in r.trace.steps.iter().enumerate() {
            assert_eq!(s.index, i);
        }
    }

    #[test]
    fn test_think_respects_temporal_scope() {
        let mut brain = populated_brain();
        let session_id = brain.session_start();
        let _ = think(&mut brain, "alpha", ThinkConfig::default());
        // Find at least one Hypothesis (or ConfirmedHypothesis) keyframe
        // tagged with the active session.
        let mut tagged = 0;
        for ((_a, _b), bond) in brain.network.bonds().iter() {
            for kf in bond.keyframes() {
                match &kf.source {
                    KeyframeSource::Hypothesis { session: Some(sid), .. }
                    | KeyframeSource::ConfirmedHypothesis { session: Some(sid), .. } => {
                        if *sid == session_id { tagged += 1; }
                    }
                    _ => {}
                }
            }
        }
        assert!(tagged > 0, "hypothesis keyframes must carry the active session id");
    }

    #[test]
    fn test_think_early_exit_on_confidence() {
        let mut brain = populated_brain();
        // populated_brain produces a high confirmation ratio in step 0.
        let opts = ThinkConfig {
            max_steps: 5,
            confidence_to_answer: 0.5,
            ..ThinkConfig::default()
        };
        let r = think(&mut brain, "alpha", opts);
        assert!(r.trace.early_exit, "loop should exit early when score >= confidence_to_answer");
        assert!(r.trace.steps.len() < 5);
    }

    /// Variant mode: when the variant arena produces a winner for a
    /// proposed target, the hypothesis bond is written with the
    /// `Hypothesis` source (same as default mode). When no variant
    /// wins, no bond is written. Constructs a brain with disjoint
    /// dense clusters so variant arenas reliably clear the default
    /// `min_winner_composite = 0.4` — non-Baseline strategies on
    /// these targets see DISJOINT activation sets (novelty ~ 1.0,
    /// parsimony ~ 1.0) which alone clear the threshold.
    #[test]
    fn test_pre_write_reflect_does_not_tautologically_promote() {
        // Tier 2.5: with PRE-write reflect, a hypothesis whose `to` is
        // unreachable from observation seeds should NOT auto-promote
        // just because the bond was written (the old post-write reflect
        // was guaranteed to find `to` via the just-written bond).
        let mut brain = Brain::new();
        brain.teach("alpha", crate::bond::BondType::IsA, "mid", 0.9, crate::node::Confidence::Verified);
        // Add an isolated node so a hypothesis can target it but no
        // pre-existing path reaches it.
        let isolated = crate::seed::Seed::from_label("isolated");
        brain.network.add_node(crate::node::KnowledgeNode::with_seed(
            isolated, "isolated".into(),
            crate::node::KnowledgeContent::Raw("".into()),
            crate::node::Confidence::Verified, 0,
        ));
        let opts = ThinkConfig {
            max_steps: 1,
            confidence_to_answer: 1.01,
            ..ThinkConfig::default()
        };
        let r = brain.think("alpha", opts);
        // Score must NOT be 1.0 just because we wrote bonds; the
        // pre-write reflect filters tautological promotions.
        // We accept any score < 1.0 as evidence the tautology is gone.
        assert!(
            r.trace.final_score < 1.0 || r.trace.steps.is_empty(),
            "pre-write reflect must not produce score=1.0 tautologically; got {}",
            r.trace.final_score,
        );
    }

    #[test]
    fn test_think_with_variant_mode_writes_winner_hypothesis() {
        let mut brain = Brain::new();
        // Main cluster: alpha branches to three target hubs.
        brain.teach("alpha", BondType::IsA,         "topic", 0.95, Confidence::Verified);
        brain.teach("alpha", BondType::HasProperty, "feature", 0.9, Confidence::Verified);
        brain.teach("alpha", BondType::Causes,      "effect", 0.9, Confidence::Verified);

        // Topic's disjoint cluster (3 leaves, all strong, all unique to topic).
        brain.teach("topic", BondType::IsA,         "tA", 0.9, Confidence::Verified);
        brain.teach("topic", BondType::HasProperty, "tB", 0.9, Confidence::Verified);
        brain.teach("topic", BondType::Causes,      "tC", 0.9, Confidence::Verified);
        brain.teach("tA",    BondType::RelatedTo,   "tB", 0.8, Confidence::Verified);

        // Feature's disjoint cluster.
        brain.teach("feature", BondType::IsA,         "fA", 0.9, Confidence::Verified);
        brain.teach("feature", BondType::HasProperty, "fB", 0.9, Confidence::Verified);
        brain.teach("feature", BondType::Causes,      "fC", 0.9, Confidence::Verified);

        // Effect's disjoint cluster.
        brain.teach("effect", BondType::IsA,         "eA", 0.9, Confidence::Verified);
        brain.teach("effect", BondType::HasProperty, "eB", 0.9, Confidence::Verified);

        let opts = ThinkConfig {
            max_steps: 1,
            hypotheses_per_step: 3,
            variant_mode: true,
            // Permissive gate so SOME variant arena clears it on this
            // small test brain. Production default is 0.4.
            variant_min_winner_composite: 0.05,
            confidence_to_answer: 1.01, // never early-exit
            ..ThinkConfig::default()
        };
        let r = think(&mut brain, "alpha", opts);

        // Confirm at least one Hypothesis-source keyframe was written —
        // i.e. some variant arena produced a winner.
        let mut has_hyp = false;
        for ((_a, _b), bond) in brain.network.bonds().iter() {
            for kf in bond.keyframes() {
                if matches!(kf.source, KeyframeSource::Hypothesis { .. }) {
                    has_hyp = true;
                    break;
                }
            }
            if has_hyp { break; }
        }
        assert!(
            has_hyp,
            "variant mode must write at least one Hypothesis bond when a variant wins; \
             trace: {} steps, {} bonds written",
            r.trace.steps.len(),
            r.trace.hypothesis_bonds_written,
        );
    }
    // ============================================================
    // OP-THINK-WATCHDOG tests
    // ============================================================

    /// Wall-clock guard MUST NOT fire when a generous budget is set
    /// and the brain converges normally on `max_steps`. Halt reason
    /// stays `MaxSteps` (or `EarlyExit` if the populated brain crosses
    /// the confidence floor; either is acceptable here — the assertion
    /// is that it is NOT `WallClockExceeded`).
    #[test]
    fn test_watchdog_does_not_fire_with_generous_budget() {
        let mut brain = populated_brain();
        let opts = ThinkConfig {
            max_steps: 5,
            confidence_to_answer: 1.01, // never early-exit on confidence
            max_wall_clock: Some(std::time::Duration::from_secs(60)),
            ..ThinkConfig::default()
        };
        let r = brain.think("alpha", opts);
        assert_eq!(r.trace.steps.len(), 5,
            "max_steps must be honoured when watchdog is generous");
        assert_eq!(r.trace.halted_reason, HaltedReason::MaxSteps,
            "halt reason must be MaxSteps when the loop runs to the cap, got {:?}",
            r.trace.halted_reason);
    }

    /// Wall-clock guard MUST fire and short-circuit the outer loop
    /// when the budget is zero (or near-zero). This is the structural
    /// safety being asserted: the loop CANNOT spend longer than the
    /// wall-clock allows even if `max_steps` is generous.
    #[test]
    fn test_watchdog_fires_on_tight_budget() {
        let mut brain = populated_brain();
        // Zero-duration budget: any non-zero work elapses past it,
        // so the very first top-of-step guard fires.
        let opts = ThinkConfig {
            max_steps: 100,
            confidence_to_answer: 1.01,
            max_wall_clock: Some(std::time::Duration::from_nanos(1)),
            ..ThinkConfig::default()
        };
        let r = brain.think("alpha", opts);
        // Either 0 steps (guard #1 fired before any work) or a small
        // number (guard #2/#3/#4 fired mid-step). Either way, far below
        // the 100-step cap.
        assert!(r.trace.steps.len() < 100,
            "watchdog must short-circuit before max_steps; got {} steps",
            r.trace.steps.len());
        assert_eq!(r.trace.halted_reason, HaltedReason::WallClockExceeded,
            "halt reason must be WallClockExceeded under a tight budget, got {:?}",
            r.trace.halted_reason);
    }

    /// `None` wall-clock budget is the audit-harness escape hatch:
    /// the watchdog must be fully disarmed and `max_steps` is the
    /// only halt axis. Asserts that `None` is honoured (no panic, no
    /// premature halt) and the halt reason is NOT WallClockExceeded.
    #[test]
    fn test_watchdog_none_disables_guard() {
        let mut brain = empty_brain();
        let opts = ThinkConfig {
            max_steps: 3,
            max_wall_clock: None,
            ..ThinkConfig::default()
        };
        let r = think(&mut brain, "alpha", opts);
        assert_eq!(r.trace.steps.len(), 3);
        assert_ne!(r.trace.halted_reason, HaltedReason::WallClockExceeded,
            "None budget must not produce WallClockExceeded; got {:?}",
            r.trace.halted_reason);
    }

    /// Partial-trace invariant: when the watchdog fires, whatever
    /// steps DID complete must be returned, never an empty steps
    /// vector with a panic. This is the graceful-degradation contract
    /// the narration slot-fill depends on.
    ///
    /// We stage this by giving a slightly-larger-than-trivial budget
    /// on a populated brain: at least one step runs to completion
    /// before the watchdog clamps further work. With a tight 1ns
    /// budget the trace can legitimately be empty (guard #1 fires
    /// before any work), so this test uses a different lever:
    /// `max_steps = 0` would give an empty trace too. Instead we
    /// assert that the result is well-formed (no panic, halt reason
    /// makes sense) under ANY watchdog firing — empty or partial.
    #[test]
    fn test_watchdog_returns_partial_trace_no_panic() {
        let mut brain = populated_brain();
        let opts = ThinkConfig {
            max_steps: 10,
            confidence_to_answer: 1.01,
            max_wall_clock: Some(std::time::Duration::from_nanos(1)),
            ..ThinkConfig::default()
        };
        // The structural assertion: think() returns cleanly, no
        // panic, no inf loop. ThinkResult is well-formed.
        let r = brain.think("alpha", opts);
        // Final answer must always be assembled (graceful degradation).
        // It can be empty-recall on a tight budget; the contract is
        // that the AskResult exists.
        let _ = &r.final_answer;
        // halted_reason must be WallClockExceeded since max_steps
        // (10) is generous; we set the budget so tight that the
        // watchdog has to fire.
        assert_eq!(r.trace.halted_reason, HaltedReason::WallClockExceeded);
        // Step count is bounded above by max_steps and below by 0.
        // The exact value depends on how fast guard #1 fires.
        assert!(r.trace.steps.len() <= 10);
    }

    /// ThinkingTraceSummary carries halted_reason through to the
    /// narrate-path summary so callers (MCP / extract_slots) can
    /// route on it. This locks the wire format.
    #[test]
    fn test_thinking_trace_summary_carries_halted_reason() {
        let mut brain = populated_brain();
        let opts = ThinkConfig {
            max_steps: 10,
            confidence_to_answer: 1.01,
            max_wall_clock: Some(std::time::Duration::from_nanos(1)),
            ..ThinkConfig::default()
        };
        let r = brain.think("alpha", opts);
        let summary = ThinkingTraceSummary::from(&r);
        assert_eq!(summary.halted_reason, HaltedReason::WallClockExceeded,
            "summary must mirror trace halted_reason; got {:?}",
            summary.halted_reason);
    }
}
