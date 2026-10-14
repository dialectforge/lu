//! Multi-agent thinking — seventh of the seven Lu 4.0 capability ports.
//!
//! Three agent roles, run synchronously and sequentially over a single
//! Brain in scope-isolated views:
//!
//! 1. **RecallAgent** — surfaces the top candidate seeds for a query in
//!    `Scope::Agent { role: Recall }`. Read-only spread; bounded by
//!    `MultiAgentConfig::max_recall_candidates`.
//! 2. **SynthesisAgent** — over the recalled candidates, synthesizes
//!    a→c via b inference triples in `Scope::Agent { role: Synthesis }`.
//!    Each synthesized belief gets an Agent-tagged keyframe so the work
//!    is auditable. Bounded by `max_hypotheses_per_synthesis`.
//! 3. **CriticAgent** — filters synthesized beliefs by
//!    `critic_confidence_threshold`. Pass-throughs get a soft Agent
//!    keyframe for provenance.
//!
//! Multi-think is OBSERVATION, not commitment. Vetted beliefs are
//! returned in the trace; they do NOT auto-promote to baseline. Agent
//! keyframes are scope-isolated (filtered out of every non-Agent read
//! by `TemporalBond::weight_at`) and age out via the half-life pruner
//! in `consolidation`.
//!
//! Architectural notes:
//! - `AgentRole` lives in `scope.rs` to break the dependency cycle
//!   between `bond.rs` and the multi-agent module. We re-export it here.
//! - Agent scopes do NOT nest. `multi_think` rejects calls whose
//!   surrounding scope is already an Agent scope. Future work:
//!   tree-of-thought / hierarchical agents.
//! - Per-agent `QueryMetric`s (with `agent_role: Some(role)`) are
//!   pushed into `Brain::performance_monitor` so the
//!   `AgentDriftDetected` analyzer can spot a deteriorating Critic
//!   acceptance rate over time. Aggregates are computed on demand —
//!   we do NOT push a separate aggregate metric per multi_think.

use crate::bond::{BondType, KeyframeSource, TemporalBond, TemporalKeyframe, DEFAULT_HALF_LIFE_MS};
use crate::brain::{now_ms, Brain};
use crate::scope::{AgentRole, TemporalScope};
use crate::seed::Seed;
use crate::reasoning::{reason_from_recall, EvidenceKind, ReasoningTrace};
use crate::self_reflection::QueryMetric;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

// =====================================================================
// Public types
// =====================================================================

/// All knobs for one multi-think run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiAgentConfig {
    /// Cap on candidate seeds RecallAgent surfaces.
    pub max_recall_candidates: usize,
    /// Cap on hypotheses SynthesisAgent emits per pass.
    pub max_hypotheses_per_synthesis: usize,
    /// Beliefs below this confidence are dropped by CriticAgent.
    pub critic_confidence_threshold: f32,
    /// Hard ceiling on wall-clock per multi-think call (ms). The
    /// orchestration loop checks this between phases — we do not
    /// preempt mid-spread, so a runaway recall is still bounded only
    /// by `max_recall_candidates`.
    pub max_multi_think_lifetime_ms: u64,
    /// Half-life for agent-tagged keyframes (ms). Mirrors
    /// `ConsolidationConfig::agent_prune_halflife_ms`. Surfaced on
    /// `MultiAgentConfig` so callers can tune per-call without
    /// touching the consolidation engine.
    pub agent_keyframe_decay_halflife_ms: u64,
    /// Spread depth used inside the Recall agent scope.
    pub agent_recall_depth: usize,
    /// Initial weight for an Agent-tagged keyframe (multiplied by the
    /// belief's confidence on the Synthesis write).
    pub agent_keyframe_weight: f32,
    // OP-REASON-LOOP Phase 2 (FLAG D): when true, multi_think reasons read-
    // only -- the Agent provenance keyframes/bonds are NOT written. The
    // vetted/rejected result is unchanged (it is decided by independent
    // recall, not by the written bonds). Older payloads default to false.
    #[serde(default)]
    pub no_write: bool,
}

impl Default for MultiAgentConfig {
    fn default() -> Self {
        MultiAgentConfig {
            max_recall_candidates: 32,
            max_hypotheses_per_synthesis: 8,
            critic_confidence_threshold: 0.5,
            max_multi_think_lifetime_ms: 30_000,
            agent_keyframe_decay_halflife_ms: 6 * 60 * 60 * 1000, // 6h
            agent_recall_depth: 3,
            agent_keyframe_weight: 0.20,
            no_write: false,
        }
    }
}

/// One seed surfaced by RecallAgent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecalledItem {
    pub seed: Seed,
    pub content: String,
    pub activation: f32,
    pub depth: u32,
}

/// One inferred a→c belief (via intermediate b) emitted by
/// SynthesisAgent. Confidence is the path-aware formula in
/// `compute_confidence` below.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Belief {
    pub from: Seed,
    pub via: Seed,
    pub to: Seed,
    pub from_content: String,
    pub via_content: String,
    pub to_content: String,
    pub w_from_via: f32,
    pub w_via_to: f32,
    pub confidence: f32,
    /// Reasons (one per scar) for refusing to act on this belief's
    /// path. Empty for unscarred paths. Populated by
    /// `run_synthesis` from the active scar set on either leg.
    /// `run_critic` rejects any belief with non-empty
    /// `scar_reasons`, recording the rejection in
    /// `MultiAgentTrace::rejected`.
    #[serde(default)]
    pub scar_reasons: Vec<String>,
    /// Caveats raised by the Critic during its independent recall.
    /// A non-empty list means the belief is accepted-with-caveats
    /// (in `vetted`) but the Critic's independent spread did not
    /// strongly reach `to`. Empty for clean acceptances.
    #[serde(default)]
    pub caveats: Vec<String>,
}

/// One Critic rejection — a synthesized belief plus the reason
/// for refusing to vet it. Surfaces in `MultiAgentTrace::rejected`
/// and feeds the narration layer's CriticVettedAnswer caveat.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RejectedBelief {
    pub belief: Belief,
    pub reason: String,
}

/// One agent's execution record. Lightweight — the heavy artifacts
/// (Recall items, beliefs) live on `MultiAgentTrace`, not here.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentStep {
    pub role: AgentRole,
    /// Wall-clock latency, microseconds.
    pub elapsed_us: u64,
    /// How many items the agent received as input.
    pub items_in: u32,
    /// How many items the agent emitted as output.
    pub items_out: u32,
}

/// Full multi-think output: every agent's trace plus the consolidated
/// answer (the CriticAgent-vetted beliefs) and the per-call telemetry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiAgentTrace {
    pub multi_think_id: u64,
    pub query: String,
    pub recall: Vec<RecalledItem>,
    pub hypotheses: Vec<Belief>,
    pub vetted: Vec<Belief>,
    /// Beliefs the Critic refused to vet, with the articulated
    /// reason. Includes both scar-driven rejections and
    /// below-threshold drops.
    #[serde(default)]
    pub rejected: Vec<RejectedBelief>,
    /// Q.4: ReasoningTrace built alongside Recall and threaded into the
    /// SynthesisAgent step. `None` when reason_from_recall produced no
    /// useful trace (anchors empty / no_chain). `#[serde(default)]` so
    /// pre-Q.4 JSON traces round-trip cleanly into the current shape.
    #[serde(default)]
    pub reasoning_trace: Option<ReasoningTrace>,
    /// Q.4: SynthesisAgent prose. References the reasoning chain when
    /// `reasoning_trace` carries a non-Mixed structured chain;
    /// otherwise falls back to a belief-only summary preserving
    /// pre-Q.4 multi_think behavior. Empty string is the documented
    /// no-prose default for backward compatibility.
    #[serde(default)]
    pub synthesis_prose: String,
    pub steps: [AgentStep; 3],
    pub started_ms: i64,
    pub ended_ms: i64,
    /// True if the orchestrator hit `max_multi_think_lifetime_ms`
    /// between phases. The current phase still completed; downstream
    /// phases were skipped.
    pub timed_out: bool,
}

// =====================================================================
// Confidence formula
// =====================================================================

/// `min(weight(a→b), weight(b→c)) * (1 - scar_penalty)` clamped `[0, 1]`.
///
/// `scar_penalty` is currently always 0 — once we wire belief-rejection
/// history into the brain, this will reflect how often a previous
/// (a, c) pair has been vetoed by the Critic. Deliberately kept simple
/// so the formula is easy to inspect when tuning later.
//
// TODO: tune this once live
pub fn compute_confidence(w_ab: f32, w_bc: f32, scar_penalty: f32) -> f32 {
    let raw = w_ab.min(w_bc) * (1.0 - scar_penalty);
    raw.clamp(0.0, 1.0)
}

// =====================================================================
// Agent runners (private)
// =====================================================================

/// Pre-flight check: a multi_think call cannot run inside another
/// agent's scope (no nesting). Returns the parent scope to use for
/// agent sub-scopes (the caller's session/current).
fn pick_parent_scope(brain: &Brain) -> Result<TemporalScope, String> {
    let scope = match brain.sessions.active_session() {
        Some(s) => TemporalScope::Session { id: s.id, start: s.start_time },
        None => TemporalScope::Current,
    };
    if matches!(scope, TemporalScope::Agent { .. }) {
        // Defensive: `active_session()` returns Session/Current today,
        // never an Agent scope, but guard anyway in case the future
        // surfaces a scoped session API.
        return Err("nested agent scopes are not supported".into());
    }
    Ok(scope)
}

fn run_recall(
    brain: &mut Brain,
    query: &str,
    multi_think_id: u64,
    cfg: &MultiAgentConfig,
    parent: &TemporalScope,
) -> Vec<RecalledItem> {
    // Issue A fix internal notes: defense-in-depth. The MCP startup
    // hoist already builds the keyword index, but multi-agent paths can
    // also be invoked from REPL / tests where startup hoist has not run.
    // Idempotent after the first call. Signature changed from &Brain to
    // &mut Brain so the lazy build can fire.
    brain.network.ensure_keyword_index_built();
    let scope = TemporalScope::Agent {
        role: AgentRole::Recall,
        multi_think_id,
        parent: Box::new(parent.clone()),
    };
    // Same seed-selection cascade as Brain::ask.
    let mut weighted = brain.network.select_seeds_for_query(query);
    if weighted.is_empty() {
        let hits = brain.network.keyword_search(query);
        weighted = hits.iter().take(3).map(|(s, _)| (*s, 0.7_f32)).collect();
    }
    // Port T.5.1 — arm freshness multiplier on this recall path.
    brain.refresh_decay_runtime();
    // Port Sense.1 -- arm sense cluster-gain multiplier on this recall path.
    let _sense_seeds: Vec<crate::seed::Seed> = weighted.iter().map(|(s, _)| *s).collect();
    brain.refresh_sense_runtime(&_sense_seeds);
    let recall = brain.network.spread_activation_weighted(
        &weighted,
        &scope,
        cfg.agent_recall_depth,
        cfg.max_recall_candidates,
        &brain.scar_log,
    );
    let mut out: Vec<RecalledItem> = recall
        .activated
        .iter()
        .map(|a| RecalledItem {
            seed: a.seed,
            content: a.content.clone(),
            activation: a.activation,
            depth: a.depth as u32,
        })
        .take(cfg.max_recall_candidates)
        .collect();
    // Stable: activation DESC, then seed bytes ASC.
    out.sort_by(|x, y| {
        y.activation
            .partial_cmp(&x.activation)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| x.seed.as_bytes().cmp(y.seed.as_bytes()))
    });
    out
}

/// SynthesisAgent: walk two-hop paths from recalled seeds and emit
/// `Belief` candidates. Uses `historical_weight` for the arithmetic so
/// scarred paths surface as candidates that the Critic can reject WITH
/// REASON.
///
/// Architecture (see scar.rs module docs, "Recall enforces, Critic
/// articulates"): Recall already refused to propagate activation
/// through scarred bonds (using `effective_weight`). Synthesis reads
/// `historical_weight` because its job is the opposite — surface the
/// candidate so the Critic can articulate the refusal. Conservative
/// defense-in-depth: the same scar drives both layers, but their
/// contracts are different.
fn run_synthesis(
    brain: &mut Brain,
    multi_think_id: u64,
    recall: &[RecalledItem],
    cfg: &MultiAgentConfig,
    parent: &TemporalScope,
    now: i64,
) -> Vec<Belief> {
    let scope = TemporalScope::Agent {
        role: AgentRole::Synthesis,
        multi_think_id,
        parent: Box::new(parent.clone()),
    };
    let recalled_set: HashSet<Seed> = recall.iter().map(|r| r.seed).collect();
    let recalled_seeds: Vec<Seed> = recall.iter().map(|r| r.seed).collect();

    let mut beliefs: Vec<Belief> = Vec::new();
    // Walk every two-hop path from a recalled `a`. We do NOT
    // require `b` or `c` to be in the recalled set: if Recall
    // pruned them (e.g. because a leg is scarred and so
    // effective_weight=0), Synthesis still surfaces the
    // candidate so the Critic can reject it with the scar
    // reason. Confidence uses historical weight, so scarred
    // paths show up as candidates that the Critic refuses.
    // Port #18c: anchor-based polysemy disambiguation needs the raw query
    // tokens, not the post-spread activation set. Depth-0 RecalledItems are
    // exactly the seeds Recall started from (see run_recall above), so they
    // serve as the SynthesisAgent analogue of the Walker’s seed_hits.
    let anchors: Vec<Seed> = recall.iter().filter(|r| r.depth == 0).map(|r| r.seed).collect();
    // Port #23: seed-cluster gate. The synthesis loop walks two hops
    // outward from every recalled `a`, so b (a's outgoing neighbor)
    // and c (b's outgoing neighbor) are unconstrained graph
    // neighbors — they can leave the query's neighborhood entirely.
    // In a 105k-node brain with hub edges that lets the global
    // confidence sort drag in topologically-distant triples (e.g.
    // "microscope -> laboratory -> workplace" for `what is mitosis`:
    // microscope is a real depth-1 neighbor of mitosis but laboratory
    // and workplace are NOT in mitosis's neighborhood). The gate:
    //   accept the triple iff
    //     (1) at least one of {a, b, c} is in `anchors` (the depth-0
    //         query seeds — option (1)/(2) in the seed-cluster spec),
    //         OR
    //     (2) BOTH b and c are in `recalled_set` (option (3): the
    //         RecallAgent's top-K activations).
    // Branch (1) preserves scarred-belief surfacing: when a leg is
    // scarred, RecallAgent prunes the downstream node, so b or c may
    // be missing from recalled_set; but as long as `a` is the actual
    // query seed (always true for those test-fixture chains) the
    // scarred chain still reaches the Critic with the scar reason.
    // Branch (2) catches the live-brain case where `a` is a recalled
    // hub but the chain wanders OUTSIDE the recalled neighborhood.
    // `a` itself is always in recalled_set by iteration.
    let anchor_set: HashSet<Seed> = anchors.iter().copied().collect();
    for a in &recalled_seeds {
        // OP-BRAIN-MapRAM P5-b2: triangle leg A->B via the canonical edge-
        // carrying seam; bond reconstructed from the carried payload in cache
        // mode (never resident neighbors_*/get_bond). Canonical order in BOTH
        // modes makes the walk deterministic and identical across paths.
        let out_a = brain.network.seam_walk_outgoing(a);
        for (b, walk_bond) in out_a {
            if b == *a {
                continue;
            }
            let r_ab = match walk_bond {
                Some(bn) => bn.view().weight_in_scope(&scope, &brain.scar_log),
                None => continue,
            };
            // Use historical weight for the arithmetic so a scar
            // surfaces a candidate belief that the Critic can
            // explicitly reject with the scar reason. If we used
            // effective_weight the path would silently vanish.
            let w_ab = r_ab.historical_weight;
            if w_ab <= 0.0 {
                continue;
            }
            // Port #18 / #18c: polysemy disambiguation on the triangle bridge.
            // Mirrors the Walker (narration.rs Port #17 Fix 4 + Port #18a): when
            // the bridge node b is polysemous in this graph, restrict the c
            // candidates to the sense cluster best supported by the recalled
            // set AND by the raw query anchors. Without this filter a hub like
            // "cell" (biology + telephony) drags in a telephony neighbor like
            // "radiophone" purely on degree, producing nonsensical "mitosis
            // relates to radiophone via cell" inferences. Anchors are the raw
            // query seeds (depth==0); the with_anchors API itself excludes b.
            // Filter is applied at hop 2 only; b reached a through a bond
            // Recall already vetted.
            let polysemy_allow_c: Option<HashSet<Seed>> = brain
                .network
                .disambiguate_polysemous_with_anchors(&b, &recalled_set, &anchors);
            // OP-BRAIN-MapRAM P5-b2: triangle leg B->C via the canonical seam.
            let out_b = brain.network.seam_walk_outgoing(&b);
            for (c, walk_bond) in out_b {
                if c == *a || c == b {
                    continue;
                }
                if let Some(allow) = &polysemy_allow_c {
                    if !allow.contains(&c) {
                        continue;
                    }
                }
                // Port #23: seed-cluster gate (see comment above).
                let anchored = anchor_set.contains(a)
                    || anchor_set.contains(&b)
                    || anchor_set.contains(&c);
                let in_recall_neighborhood =
                    recalled_set.contains(&b) && recalled_set.contains(&c);
                if !anchored && !in_recall_neighborhood {
                    continue;
                }
                let r_bc = match walk_bond {
                    Some(bn) => bn.view().weight_in_scope(&scope, &brain.scar_log),
                    None => continue,
                };
                let w_bc = r_bc.historical_weight;
                if w_bc <= 0.0 {
                    continue;
                }
                // Collect any scar reasons on either leg of the
                // path. The Critic uses these to reject the
                // belief outright with an articulated reason
                // (rather than the silent below-threshold drop
                // that the historical confidence formula
                // produces).
                let mut leg_scars: Vec<String> = Vec::new();
                for sc in r_ab.active_scars.iter().chain(r_bc.active_scars.iter()) {
                    leg_scars.push(sc.reason.clone());
                }
                let conf = compute_confidence(w_ab, w_bc, 0.0);
                let from_content = brain
                    .network
                    .get_node(a)
                    .map(|n| n.content().to_string())
                    .unwrap_or_default();
                let via_content = brain
                    .network
                    .get_node(&b)
                    .map(|n| n.content().to_string())
                    .unwrap_or_default();
                let to_content = brain
                    .network
                    .get_node(&c)
                    .map(|n| n.content().to_string())
                    .unwrap_or_default();
                beliefs.push(Belief {
                    from: *a,
                    via: b,
                    to: c,
                    from_content,
                    via_content,
                    to_content,
                    w_from_via: w_ab,
                    w_via_to: w_bc,
                    confidence: conf,
                    scar_reasons: leg_scars,
                    caveats: Vec::new(),
                });
            }
        }
    }

    // Sort by confidence DESC; tiebreak by seeds for determinism.
    beliefs.sort_by(|x, y| {
        y.confidence
            .partial_cmp(&x.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| x.from.as_bytes().cmp(y.from.as_bytes()))
            .then_with(|| x.via.as_bytes().cmp(y.via.as_bytes()))
            .then_with(|| x.to.as_bytes().cmp(y.to.as_bytes()))
    });
    beliefs.truncate(cfg.max_hypotheses_per_synthesis);

    // Provenance: mark each belief's (a, c) bond with an Agent-source
    // keyframe. Visible only inside Synthesis-role agent scopes.
    for (step, bl) in beliefs.iter().enumerate() {
        let weight = (cfg.agent_keyframe_weight * bl.confidence).clamp(0.0, 1.0);
        write_agent_keyframe(
            cfg.no_write,
            brain,
            bl.from,
            bl.to,
            AgentRole::Synthesis,
            multi_think_id,
            step as u32,
            weight,
            now,
        );
    }
    beliefs
}

/// Critic acceptance threshold for the independent second spread.
/// `to` must reach at least this activation in the Critic-scope
/// recall to count as a clean acceptance. Below this but above
/// `CRITIC_REJECT_FLOOR` -> AcceptedWithCaveats. At or below the
/// floor -> Rejected.
const CRITIC_ACCEPT_FLOOR: f32 = 0.10;
const CRITIC_REJECT_FLOOR: f32 = 0.01;
/// Spread depth used by the Critic's second-spread vetting pass.
/// Cheap (depth=2 from one seed); avoids re-running the full Recall
/// agent's depth budget.
const CRITIC_SPREAD_DEPTH: usize = 2;
const CRITIC_SPREAD_TOPK: usize = 32;

fn run_critic(
    brain: &mut Brain,
    multi_think_id: u64,
    hypotheses: &[Belief],
    cfg: &MultiAgentConfig,
    parent: &TemporalScope,
    now: i64,
) -> (Vec<Belief>, Vec<RejectedBelief>) {
    // Tier 2.6 fix: the Critic now runs its OWN spread inside the
    // Critic agent_scope (constructed below) and uses the
    // independent reach of `to` from `from` to decide acceptance.
    // Three outcomes:
    //   - independent activation >= CRITIC_ACCEPT_FLOOR: clean accept.
    //   - between floors: AcceptedWithCaveats (still vetted, but
    //     a caveat string is attached and surfaced by narration).
    //   - <= CRITIC_REJECT_FLOOR: Rejected.
    let agent_scope = TemporalScope::Agent {
        role: AgentRole::Critic,
        multi_think_id,
        parent: Box::new(parent.clone()),
    };
    let mut vetted: Vec<Belief> = Vec::new();
    let mut rejected: Vec<RejectedBelief> = Vec::new();
    for b in hypotheses {
        if !b.scar_reasons.is_empty() {
            // Scar trumps every other consideration. Lu has
            // moral knowledge of the path — articulate it.
            let reason = format!("scarred: {}", b.scar_reasons.join("; "));
            rejected.push(RejectedBelief { belief: b.clone(), reason });
            continue;
        }
        if b.confidence < cfg.critic_confidence_threshold {
            rejected.push(RejectedBelief {
                belief: b.clone(),
                reason: "below confidence threshold".to_string(),
            });
            continue;
        }
        // Independent recall: spread from `from` in the Critic
        // scope and read the activation of `to`.  Depth=2 covers
        // the synthesized two-hop path (a -> b -> c) without
        // burning the recall agent's depth budget.
        // Port T.5.1 — arm freshness multiplier for Critic's recall.
        brain.refresh_decay_runtime();
        // Port Sense.1 -- arm sense cluster-gain multiplier for Critic's recall.
        // Critic seeds on a single bond endpoint, so the gain map only
        // fires when `b.from` is itself polysemous (rare); otherwise
        // gain stays 1.0 and behaviour matches pre-Sense.1 byte-for-byte.
        brain.refresh_sense_runtime(&[b.from]);
        let critic_recall = brain.network.spread_activation_weighted(
            &[(b.from, 1.0_f32)],
            &agent_scope,
            CRITIC_SPREAD_DEPTH,
            CRITIC_SPREAD_TOPK,
            &brain.scar_log,
        );
        let to_act = critic_recall
            .activated
            .iter()
            .find(|n| n.seed == b.to)
            .map(|n| n.activation)
            .unwrap_or(0.0);
        if to_act >= CRITIC_ACCEPT_FLOOR {
            vetted.push(b.clone());
        } else if to_act > CRITIC_REJECT_FLOOR {
            let mut bb = b.clone();
            bb.caveats.push(format!(
                "Critic's independent recall reached `{}` only weakly (act={:.3})",
                bb.to_content, to_act,
            ));
            vetted.push(bb);
        } else {
            rejected.push(RejectedBelief {
                belief: b.clone(),
                reason: format!(
                    "Critic's independent recall did not reach `{}` (act={:.3})",
                    b.to_content, to_act,
                ),
            });
        }
    }
    // Soft provenance mark — half the standard agent-keyframe weight.
    for (step, bl) in vetted.iter().enumerate() {
        let weight = (cfg.agent_keyframe_weight * 0.5).clamp(0.0, 1.0);
        write_agent_keyframe(
            cfg.no_write,
            brain,
            bl.from,
            bl.to,
            AgentRole::Critic,
            multi_think_id,
            step as u32,
            weight,
            now,
        );
    }
    (vetted, rejected)
}

/// Write an Agent-tagged keyframe on the (`from` → `to`) bond, creating
/// the bond with a Variant-style fresh constructor if it didn't exist.
/// Mirrors `variants::generate_variants_for_goal`'s keyframe writer.
fn write_agent_keyframe(
    no_write: bool,
    brain: &mut Brain,
    from: Seed,
    to: Seed,
    role: AgentRole,
    multi_think_id: u64,
    step: u32,
    weight: f32,
    now: i64,
) {
    // OP-REASON-LOOP Phase 2 (FLAG D): read-only reasoning suppresses the
    // Agent provenance bond write.
    if no_write {
        return;
    }
    let source = KeyframeSource::Agent {
        role: role.as_u8(),
        multi_think_id,
        step,
    };
    if brain.network.get_bond(&from, &to).is_some() {
        if let Some(mut b) = brain.network.get_bond_mut(&from, &to) {
            b.add_keyframe(now, weight.clamp(0.0, 1.0), source);
        }
    } else {
        let bond = TemporalBond {
            from,
            to,
            bond_type: BondType::RelatedTo,
            keyframes: vec![TemporalKeyframe {
                timestamp: now,
                weight: weight.clamp(0.0, 1.0),
                source,
            }],
            half_life_ms: DEFAULT_HALF_LIFE_MS,
        };
        let _ = brain.network.add_bond(bond);
    }
}

// =====================================================================
// Public orchestration entrypoint
// =====================================================================

/// Q.4: build the SynthesisAgent prose, optionally referencing the
/// reasoning chain articulated by `reason_from_recall`.
///
/// Behavior matrix:
/// - `reasoning` is `Some` and the trace carries a Structured chain
///   (non-Mixed, non-no-chain): prose opens with the chain
///   articulation — "Based on the reasoning chain: <final_answer>".
/// - `reasoning` is `Some` but the trace is Associative (Mixed) or
///   `no_chain == true`: chain is NOT referenced; falls through to
///   the belief-only summary (preserves pre-Q.4 behavior).
/// - `reasoning` is `None`: same belief-only fallback.
///
/// `beliefs` is the SynthesisAgent's post-cap output
/// (`run_synthesis` return value, length ≤ `max_hypotheses_per_synthesis`).
pub(crate) fn build_synthesis_prose(
    query: &str,
    beliefs: &[Belief],
    reasoning: Option<&ReasoningTrace>,
) -> String {
    let chain_clause: Option<String> = reasoning.and_then(|t| {
        if t.no_chain || t.evidence_kind == EvidenceKind::Associative {
            None
        } else if t.final_answer.trim().is_empty() {
            None
        } else {
            Some(t.final_answer.clone())
        }
    });
    // First-person clause works for action verbs ("I synthesized ...")
    // but is ungrammatical for the empty-belief noun phrase, so the
    // empty branch uses an absolute clause ("Found no...") that reads
    // cleanly with or without the chain prefix.
    let (predicate, leading_capital) = if beliefs.is_empty() {
        ("found no two-hop beliefs from recall".to_string(), "Found no two-hop beliefs from recall".to_string())
    } else if beliefs.len() == 1 {
        let b = &beliefs[0];
        let lower = format!(
            "synthesized 1 two-hop belief ({} → {} → {}, conf={:.2})",
            b.from_content, b.via_content, b.to_content, b.confidence,
        );
        let upper = format!(
            "Synthesized 1 two-hop belief ({} → {} → {}, conf={:.2})",
            b.from_content, b.via_content, b.to_content, b.confidence,
        );
        (lower, upper)
    } else {
        let top = &beliefs[0];
        let lower = format!(
            "synthesized {} two-hop beliefs (top: {} → {} → {}, conf={:.2})",
            beliefs.len(), top.from_content, top.via_content, top.to_content, top.confidence,
        );
        let upper = format!(
            "Synthesized {} two-hop beliefs (top: {} → {} → {}, conf={:.2})",
            beliefs.len(), top.from_content, top.via_content, top.to_content, top.confidence,
        );
        (lower, upper)
    };
    match chain_clause {
        Some(chain) => format!(
            "Based on the reasoning chain: {} — {} for query \"{}\".",
            chain, predicate, query,
        ),
        None => format!(
            "{} for query \"{}\".",
            leading_capital, query,
        ),
    }
}

/// Run one full multi-think: Recall → Synthesis → Critic. Synchronous,
/// sequential, single-threaded. Bounded by `cfg`. Returns `Err` if the
/// caller's surrounding scope is already an Agent scope (no nesting).
pub fn multi_think(
    brain: &mut Brain,
    query: &str,
    cfg: &MultiAgentConfig,
) -> Result<MultiAgentTrace, String> {
    let parent = pick_parent_scope(brain)?;
    let multi_think_id = brain.next_multi_think_id;
    brain.next_multi_think_id = brain.next_multi_think_id.saturating_add(1);

    let started_ms = now_ms();
    let mut timed_out = false;

    // ---- Phase 1: Recall ----
    let t0 = now_us();
    let recall = run_recall(brain, query, multi_think_id, cfg, &parent);
    let recall_us = now_us().saturating_sub(t0);
    let recall_step = AgentStep {
        role: AgentRole::Recall,
        elapsed_us: recall_us,
        items_in: 1, // the query itself
        items_out: recall.len() as u32,
    };
    push_agent_metric(
        brain,
        query,
        AgentRole::Recall,
        &parent,
        recall_us,
        1,
        recall.len() as u32,
        recall.first().map(|r| r.activation).unwrap_or(0.0),
    );
    if now_ms().saturating_sub(started_ms) >= cfg.max_multi_think_lifetime_ms as i64 {
        timed_out = true;
    }

    // ---- Phase 1.5 (Q.4): Reasoning trace from Recall ----
    // Compute a ReasoningTrace from the recall items so SynthesisAgent
    // can reference the Port A-E chain articulation when one exists.
    // Ride on the parent scope (NOT an Agent scope) so the trace sees
    // baseline weights, not Recall-agent provenance keyframes. Skipped
    // when timed_out so a starved orchestrator can't double-charge.
    let reasoning_trace: Option<ReasoningTrace> = if timed_out || recall.is_empty() {
        None
    } else {
        Some(reason_from_recall(brain, query, &recall, parent.clone()))
    };

    // ---- Phase 2: Synthesis ----
    let t0 = now_us();
    let hypotheses = if timed_out {
        Vec::new()
    } else {
        run_synthesis(brain, multi_think_id, &recall, cfg, &parent, started_ms)
    };
    let synth_us = now_us().saturating_sub(t0);
    let synth_step = AgentStep {
        role: AgentRole::Synthesis,
        elapsed_us: synth_us,
        items_in: recall.len() as u32,
        items_out: hypotheses.len() as u32,
    };
    push_agent_metric(
        brain,
        query,
        AgentRole::Synthesis,
        &parent,
        synth_us,
        recall.len() as u32,
        hypotheses.len() as u32,
        0.0,
    );
    if now_ms().saturating_sub(started_ms) >= cfg.max_multi_think_lifetime_ms as i64 {
        timed_out = true;
    }

    // ---- Phase 3: Critic ----
    let t0 = now_us();
    let (vetted, rejected) = if timed_out {
        (Vec::new(), Vec::new())
    } else {
        run_critic(brain, multi_think_id, &hypotheses, cfg, &parent, started_ms)
    };
    let crit_us = now_us().saturating_sub(t0);
    let crit_step = AgentStep {
        role: AgentRole::Critic,
        elapsed_us: crit_us,
        items_in: hypotheses.len() as u32,
        items_out: vetted.len() as u32,
    };
    push_agent_metric(
        brain,
        query,
        AgentRole::Critic,
        &parent,
        crit_us,
        hypotheses.len() as u32,
        vetted.len() as u32,
        0.0,
    );

    let ended_ms = now_ms();
    // Q.4: SynthesisAgent prose — references the chain when one was
    // articulated above; falls back to belief-only summary otherwise.
    let synthesis_prose = build_synthesis_prose(query, &hypotheses, reasoning_trace.as_ref());

    // Port LB.1: route each multi-think agent step into the Left Brain
    // as a ThinkingTrace entry. Anti-hallucination invariant: these are
    // operational traces, not facts -- the canonical bond_count is
    // unchanged by these writes.
    let lb_now = now_ms() as u64;
    for step in &[&recall_step, &synth_step, &crit_step] {
        let payload = serde_json::to_string(step).unwrap_or_default();
        let _ = brain.lb_session.push_internal(
            crate::lb_session::SessionEntry::new(
                0,
                lb_now,
                crate::lb_session::SessionEntryKind::ThinkingTrace,
                query,
                format!("multi_think id={} role={:?} {}", multi_think_id, step.role, payload),
                0.0,
            ),
        );
    }

    Ok(MultiAgentTrace {
        multi_think_id,
        query: query.to_string(),
        recall,
        hypotheses,
        vetted,
        rejected,
        reasoning_trace,
        synthesis_prose,
        steps: [recall_step, synth_step, crit_step],
        started_ms,
        ended_ms,
        timed_out,
    })
}

// =====================================================================
// Telemetry helpers
// =====================================================================

fn now_us() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_micros() as u64)
        .unwrap_or(0)
}

fn hash_query(q: &str) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    q.hash(&mut h);
    h.finish()
}

fn push_agent_metric(
    brain: &mut Brain,
    query: &str,
    role: AgentRole,
    scope: &TemporalScope,
    elapsed_us: u64,
    items_in: u32,
    items_out: u32,
    top_activation: f32,
) {
    brain.push_query_metric(QueryMetric {
        timestamp_ms: now_ms() as u64,
        query_text_hash: hash_query(query),
        scope: scope.clone(),
        latency_us: elapsed_us,
        top_activation,
        bridges_fired: 0,
        synthesis_triangles_fired: 0,
        curiosity_signals_emitted: 0,
        confirmed_hypotheses: 0,
        winning_variants: 0,
        agent_role: Some(role),
        agent_items_in: items_in,
        agent_items_out: items_out,
    });
}

// =====================================================================
// Tests
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scar::ScarLog;
    use crate::bond::BondType;
    use crate::node::Confidence;

    /// Three-node line graph (alpha → mid → omega) plus a weak alpha → omega
    /// shortcut. Synthesis should produce one belief alpha → mid → omega
    /// at confidence min(0.8, 0.8) = 0.8.
    fn line_brain() -> Brain {
        let mut brain = Brain::new();
        brain.teach("alpha", BondType::IsA,         "mid",   0.8, Confidence::Verified);
        brain.teach("mid",   BondType::HasProperty, "omega", 0.8, Confidence::Verified);
        brain.teach("alpha", BondType::RelatedTo,   "omega", 0.2, Confidence::Verified);
        brain
    }

    #[test]
    fn test_compute_confidence_clamping() {
        // min picks the smaller leg; scar_penalty 0 leaves it untouched.
        assert!((compute_confidence(0.4, 0.9, 0.0) - 0.4).abs() < 1e-6);
        assert!((compute_confidence(0.9, 0.4, 0.0) - 0.4).abs() < 1e-6);
        // Heavy scar penalty drives below 0 — must clamp to 0.
        assert_eq!(compute_confidence(0.5, 0.5, 1.5), 0.0);
        // Negative leg also clamps.
        assert_eq!(compute_confidence(-0.2, 0.7, 0.0), 0.0);
    }

    #[test]
    fn test_multi_think_returns_three_traces() {
        let mut brain = line_brain();
        let cfg = MultiAgentConfig::default();
        let trace = multi_think(&mut brain, "alpha", &cfg).expect("no nested scope");
        assert_eq!(trace.steps.len(), 3);
        assert_eq!(trace.steps[0].role, AgentRole::Recall);
        assert_eq!(trace.steps[1].role, AgentRole::Synthesis);
        assert_eq!(trace.steps[2].role, AgentRole::Critic);
        // Recall must surface alpha and at least one neighbor.
        assert!(!trace.recall.is_empty());
        // Critic's items_in must equal Synthesis's items_out.
        assert_eq!(trace.steps[1].items_out, trace.steps[2].items_in);
        // multi_think_id must be monotonically issued.
        let trace2 = multi_think(&mut brain, "alpha", &cfg).unwrap();
        assert_eq!(trace2.multi_think_id, trace.multi_think_id + 1);
    }

    #[test]
    fn test_synthesis_agent_proposes_hypotheses() {
        let mut brain = line_brain();
        let cfg = MultiAgentConfig::default();
        let trace = multi_think(&mut brain, "alpha", &cfg).unwrap();
        // Synthesis should have produced at least one belief — the
        // alpha → mid → omega chain.
        let alpha = Seed::from_label("alpha");
        let mid = Seed::from_label("mid");
        let omega = Seed::from_label("omega");
        assert!(
            trace.hypotheses.iter().any(|b| b.from == alpha && b.via == mid && b.to == omega),
            "synthesis should produce alpha→mid→omega; got {:?}",
            trace.hypotheses.iter().map(|b| (b.from_content.clone(), b.via_content.clone(), b.to_content.clone())).collect::<Vec<_>>()
        );
    }

    /// Port #18: SynthesisAgent must filter the c (hop-2) candidate
    /// against the bridge node b polysemy disambiguation. Without
    /// this fix the highest-degree polysemous hub (cell) drags
    /// telephony neighbors into a biology query. Fixture mirrors
    /// network.rs test_disambiguate_polysemous_clusters_by_neighborhood:
    /// a cell hub with two disjoint sense clusters (biology, telephony)
    /// plus an anchor mitosis -> cell so Synthesis has a two-hop walk.
    #[test]
    fn test_synthesis_polysemy_filter_excludes_offsense_bridge_neighbor() {
        let mut brain = Brain::new();
        // Biology cluster: every term shares an outgoing neighbor
        // (biology) so they cluster under union-find Jaccard.
        for t in ["mitosis", "chromosome", "nucleus", "meiosis"] {
            brain.teach(t, BondType::IsA, "biology", 0.9, Confidence::Verified);
        }
        // Telephony cluster: disjoint shared neighbor (telephony).
        for t in ["radiophone", "radiotelephone"] {
            brain.teach(t, BondType::IsA, "telephony", 0.9, Confidence::Verified);
        }
        // cell is the polysemous hub: outgoing into both clusters.
        for t in ["mitosis", "chromosome", "nucleus", "meiosis"] {
            brain.teach("cell", BondType::HasProperty, t, 0.9, Confidence::Verified);
        }
        for t in ["radiophone", "radiotelephone"] {
            brain.teach("cell", BondType::RelatedTo, t, 0.9, Confidence::Verified);
        }
        // Anchor: mitosis -> cell gives Synthesis a two-hop. Without
        // the filter Synthesis would emit mitosis -> cell -> radiophone.
        brain.teach("mitosis", BondType::RelatedTo, "cell", 0.9, Confidence::Verified);
        brain.network.recompute_degree_p90();

        // Biology-anchored query so the recalled set has more biology
        // members of cells outgoing neighborhood than telephony.
        let cfg = MultiAgentConfig::default();
        let trace = multi_think(&mut brain, "mitosis chromosome nucleus", &cfg)
            .expect("no nested scope");

        let cell = Seed::from_label("cell");
        let radiophone = Seed::from_label("radiophone");
        let radiotelephone = Seed::from_label("radiotelephone");
        let leaked: Vec<(String, String, String)> = trace.hypotheses.iter()
            .filter(|b| b.via == cell && (b.to == radiophone || b.to == radiotelephone))
            .map(|b| (b.from_content.clone(), b.via_content.clone(), b.to_content.clone()))
            .collect();
        assert!(
            leaked.is_empty(),
            "telephony-cluster bridge leaked into Synthesis hypotheses: {:?}",
            leaked
        );
        // Sanity: biology-side hop-2 (cell -> chromosome etc.) is still
        // emitted, so the filter pruned the off-sense branch only.
        let chromosome = Seed::from_label("chromosome");
        let nucleus = Seed::from_label("nucleus");
        let meiosis = Seed::from_label("meiosis");
        let kept_bio = trace.hypotheses.iter().any(|b| {
            b.via == cell && (b.to == chromosome || b.to == nucleus || b.to == meiosis)
        });
        assert!(kept_bio, "biology-side hop-2 must still be produced after the filter");
    }

    /// Port #23: seed-cluster gate. The synthesis ranker walks two
    /// hops outward from every recalled seed, but b and c are
    /// unconstrained graph neighbors that can leave the recalled
    /// neighborhood entirely. The gate rejects triples where b or c
    /// is outside `recalled_set` (the RecallAgent's depth-K
    /// activations). Fixture: a relevant (alpha, beta, gamma) chain
    /// inside the recalled set AND a far-out (delta, epsilon, zeta)
    /// chain whose epsilon and zeta lie outside recall.
    #[test]
    fn test_synthesis_seed_cluster_gate_drops_unanchored_triples() {
        let mut brain = Brain::new();
        // Relevant chain — anchored on the query subject. (alpha, beta, gamma)
        // must survive the gate (alpha is depth-0 in seed_cluster).
        brain.teach("alpha", BondType::IsA,         "beta",  0.9, Confidence::Verified);
        brain.teach("beta",  BondType::HasProperty, "gamma", 0.9, Confidence::Verified);
        // Bridge so RecallAgent can still spread to depth-3 nodes.
        brain.teach("gamma", BondType::RelatedTo,   "delta", 0.9, Confidence::Verified);
        // Far-out chain — (delta, epsilon, zeta) is heavy, so without
        // the gate it ranks ABOVE (alpha, beta, gamma) on confidence
        // and crowds in. delta is at depth 3 from alpha; epsilon and
        // zeta are NOT in the recalled set at all — the synthesis
        // walks them as 1-hop / 2-hop neighbors of delta.
        brain.teach("delta",   BondType::HasProperty, "epsilon", 0.95, Confidence::Verified);
        brain.teach("epsilon", BondType::HasProperty, "zeta",    0.95, Confidence::Verified);
        brain.network.recompute_degree_p90();

        let cfg = MultiAgentConfig::default();
        let trace = multi_think(&mut brain, "alpha", &cfg).expect("no nested scope");

        let alpha = Seed::from_label("alpha");
        let beta = Seed::from_label("beta");
        let gamma = Seed::from_label("gamma");
        let delta = Seed::from_label("delta");
        let epsilon = Seed::from_label("epsilon");
        let zeta = Seed::from_label("zeta");

        // The relevant triple must survive the gate.
        let kept_relevant = trace.hypotheses.iter().any(|b| {
            b.from == alpha && b.via == beta && b.to == gamma
        });
        assert!(
            kept_relevant,
            "gate over-pruned: anchored (alpha, beta, gamma) was dropped; got {:?}",
            trace.hypotheses.iter().map(|b| (b.from_content.clone(), b.via_content.clone(), b.to_content.clone())).collect::<Vec<_>>()
        );

        // Unanchored triples (delta, epsilon, zeta) and (gamma, delta, epsilon)
        // must be rejected by the gate. delta is at depth 3 (NOT in cluster);
        // epsilon and zeta were never in the recalled set at all.
        let leaked_far: Vec<(String, String, String)> = trace.hypotheses.iter()
            .filter(|b| (b.from == delta && b.via == epsilon && b.to == zeta)
                || (b.from == gamma && b.via == delta && b.to == epsilon))
            .map(|b| (b.from_content.clone(), b.via_content.clone(), b.to_content.clone()))
            .collect();
        assert!(
            leaked_far.is_empty(),
            "gate failed: unanchored triples leaked into synthesis hypotheses: {:?}",
            leaked_far
        );
    }

    #[test]
    fn test_critic_second_spread_can_reject() {
        // Tier 2.6: synthesize a Belief whose `to` is unreachable from
        // `from` in the Critic's independent recall.  The Critic must
        // either reject it or attach a caveat — it must NOT clean-vet.
        let mut brain = Brain::new();
        // Disconnected pair so independent spread can't reach `to`.
        brain.teach("alpha", BondType::IsA,         "mid",   0.8, Confidence::Verified);
        brain.teach("mid",   BondType::HasProperty, "omega", 0.8, Confidence::Verified);
        // Inject a synthetic high-confidence belief alpha->island that
        // the Synthesis agent could NEVER produce from this graph.
        let alpha = Seed::from_label("alpha");
        let mid = Seed::from_label("mid");
        let island = Seed::from_label("island");
        // Add the island node but no bonds reaching it from alpha.
        brain.network.add_node(crate::node::KnowledgeNode::with_seed(
            island, "island".into(),
            crate::node::KnowledgeContent::Raw("".into()),
            Confidence::Verified, 0,
        ));
        let synthetic = Belief {
            from: alpha, via: mid, to: island,
            from_content: "alpha".into(), via_content: "mid".into(), to_content: "island".into(),
            w_from_via: 0.9, w_via_to: 0.9, confidence: 0.9,
            scar_reasons: Vec::new(), caveats: Vec::new(),
        };
        let cfg = MultiAgentConfig::default();
        let parent = TemporalScope::Current;
        let (vetted, rejected) = run_critic(
            &mut brain, 999, &[synthetic], &cfg, &parent, 0,
        );
        // Either rejected outright OR accepted-with-caveats; never
        // clean-vetted.
        let clean_vetted = vetted.iter().any(|b| b.caveats.is_empty());
        assert!(
            !clean_vetted,
            "Critic must NOT clean-vet a belief whose `to` is unreachable; vetted={} rejected={}",
            vetted.len(), rejected.len(),
        );
        let signaled = !rejected.is_empty()
            || vetted.iter().any(|b| !b.caveats.is_empty());
        assert!(signaled, "Critic must surface SOME signal (rejection or caveat)");
    }

    #[test]
    fn test_critic_agent_filters_below_threshold() {
        let mut brain = line_brain();
        // Force every belief to be vetoed by setting threshold above 1.
        let cfg = MultiAgentConfig {
            critic_confidence_threshold: 1.5,
            ..MultiAgentConfig::default()
        };
        let trace = multi_think(&mut brain, "alpha", &cfg).unwrap();
        assert!(!trace.hypotheses.is_empty(), "synthesis still produced beliefs");
        assert!(trace.vetted.is_empty(), "critic must veto all");
    }

    #[test]
    fn test_agent_keyframes_isolated_from_baseline() {
        // Agent-source keyframes must NOT be visible to Current scope.
        let mut brain = line_brain();
        let cfg = MultiAgentConfig::default();
        let trace = multi_think(&mut brain, "alpha", &cfg).unwrap();
        assert!(!trace.hypotheses.is_empty());
        // Pick the first hypothesis: the (from, to) bond must have an
        // Agent keyframe but its Current-scope weight must come from
        // baseline only (not the agent kf).
        let bl = &trace.hypotheses[0];
        let bond = brain.network.get_bond(&bl.from, &bl.to)
            .expect("bond should exist");
        let has_agent = bond.keyframes().iter().any(|kf| matches!(
            &kf.source, KeyframeSource::Agent { .. }
        ));
        assert!(has_agent, "agent keyframe should be on the bond");
        // Agent scope sees the agent kf (or falls through):
        let scar_log = ScarLog::default();
        let in_agent = bond.weight_in_scope(&TemporalScope::Agent {
            role: AgentRole::Synthesis,
            multi_think_id: trace.multi_think_id,
            parent: Box::new(TemporalScope::Current),
        }, &scar_log).effective_weight;
        assert!(in_agent > 0.0);
        // Current scope must IGNORE the agent kf — weight should be the
        // baseline (alpha→omega = 0.2 raw * 0.8 RelatedTo = 0.16) or
        // 0 if the bond was created fresh by the agent.
        let in_current = bond.weight_in_scope(&TemporalScope::Current, &scar_log).effective_weight;
        // Sanity: pre-existing alpha→omega has Current-scope weight ≈
        // 0.16, the agent kf is 0.2 * confidence ≈ 0.2 * 0.64 = 0.128.
        // The two must NOT be equal; the agent kf must not show in
        // Current. Allow some slop.
        let agent_only_kf: Vec<&TemporalKeyframe> = bond.keyframes().iter()
            .filter(|kf| matches!(&kf.source, KeyframeSource::Agent { .. }))
            .collect();
        let agent_kf_weight = agent_only_kf.last().map(|kf| kf.weight).unwrap_or(0.0);
        assert!(
            (in_current - agent_kf_weight * bond.bond_type().base_weight()).abs() > 1e-4,
            "current scope must not equal agent-only weight"
        );
    }

    #[test]
    fn test_multi_think_does_not_auto_promote() {
        // Even with high-confidence beliefs, vetted beliefs do NOT
        // auto-promote — re-querying via Brain::ask must not pick up
        // any new baseline-visible bonds the multi-think wrote.
        let mut brain = line_brain();
        let alpha = Seed::from_label("alpha");
        let omega = Seed::from_label("omega");
        let pre_weight = brain.network.get_bond(&alpha, &omega)
            .map(|b| b.weight_in_scope(&TemporalScope::Current, &ScarLog::default()).effective_weight)
            .unwrap_or(0.0);
        let cfg = MultiAgentConfig::default();
        let _ = multi_think(&mut brain, "alpha", &cfg).unwrap();
        let post_weight = brain.network.get_bond(&alpha, &omega)
            .map(|b| b.weight_in_scope(&TemporalScope::Current, &ScarLog::default()).effective_weight)
            .unwrap_or(0.0);
        assert!(
            (pre_weight - post_weight).abs() < 1e-5,
            "current-scope weight must not change after multi_think; pre={pre_weight}, post={post_weight}",
        );
    }

    #[test]
    fn test_agent_metric_carries_role() {
        let mut brain = line_brain();
        let cfg = MultiAgentConfig::default();
        let _ = multi_think(&mut brain, "alpha", &cfg).unwrap();
        // The performance monitor must now contain three metrics with
        // agent_role set, one per role.
        let roles: HashSet<AgentRole> = brain.performance_monitor.recent_metrics.iter()
            .filter_map(|m| m.agent_role)
            .collect();
        assert!(roles.contains(&AgentRole::Recall));
        assert!(roles.contains(&AgentRole::Synthesis));
        assert!(roles.contains(&AgentRole::Critic));
    }

    #[test]
    fn test_multi_think_bounded_lifetime_zero() {
        // A 0-ms lifetime forces the orchestrator to skip Synthesis +
        // Critic. Recall always runs at least once.
        let mut brain = line_brain();
        let cfg = MultiAgentConfig {
            max_multi_think_lifetime_ms: 0,
            ..MultiAgentConfig::default()
        };
        let trace = multi_think(&mut brain, "alpha", &cfg).unwrap();
        assert!(trace.timed_out, "0-ms lifetime should trigger timed_out");
        assert!(trace.hypotheses.is_empty());
        assert!(trace.vetted.is_empty());
    }

    #[test]
    fn test_no_nested_agent_scopes_via_helper() {
        // pick_parent_scope is the runtime guard. Smoke-test by
        // checking it returns Ok for plain Current.
        let brain = Brain::new();
        let scope = pick_parent_scope(&brain).unwrap();
        assert!(matches!(scope, TemporalScope::Current));
    }

    #[test]
    fn test_multi_think_id_persists_across_calls() {
        let mut brain = Brain::new();
        brain.teach("x", BondType::IsA, "y", 0.5, Confidence::Verified);
        let cfg = MultiAgentConfig::default();
        let t1 = multi_think(&mut brain, "x", &cfg).unwrap();
        let t2 = multi_think(&mut brain, "x", &cfg).unwrap();
        let t3 = multi_think(&mut brain, "x", &cfg).unwrap();
        assert_eq!(t2.multi_think_id, t1.multi_think_id + 1);
        assert_eq!(t3.multi_think_id, t2.multi_think_id + 1);
    }

    #[test]
    fn test_agent_keyframes_decay_via_consolidation() {
        // After multi-think, run consolidate with an aggressive 1ms
        // agent halflife and a far-future `now`. Every Agent kf should
        // be pruned.
        let mut brain = line_brain();
        let cfg = MultiAgentConfig::default();
        let _ = multi_think(&mut brain, "alpha", &cfg).unwrap();

        let alpha = Seed::from_label("alpha");
        let omega = Seed::from_label("omega");
        let pre_agent_kfs = brain.network.get_bond(&alpha, &omega).unwrap().keyframes()
            .iter().filter(|kf| matches!(&kf.source, KeyframeSource::Agent { .. }))
            .count();
        assert!(pre_agent_kfs > 0, "multi_think should have written agent kfs");

        // Mutate consolidator config in-place to use a tight halflife.
        brain.consolidator.config.agent_prune_halflife_ms = 1;
        // Use a far-future timestamp so the cutoff sweeps everything.
        let far_future = now_ms() + 10 * 86_400_000;
        let stats = brain.consolidator.consolidate(&mut brain.network, far_future);
        assert!(
            stats.agent_keyframes_pruned >= pre_agent_kfs,
            "pruner should drop all aged agent kfs; pruned={}, expected>={}",
            stats.agent_keyframes_pruned, pre_agent_kfs,
        );
    }

    // =================================================================
    // Q.4 — ReasoningTrace into SynthesisAgent
    // =================================================================

    use crate::reasoning::{ReasoningStep, ReasoningTrace as RT};

    fn structured_trace(query: &str, final_answer: &str) -> RT {
        RT {
            query: query.into(),
            scope: TemporalScope::Current,
            steps: vec![ReasoningStep {
                from: Seed::from_label("a"),
                from_content: "a".into(),
                bond_type: BondType::IsA,
                to: Seed::from_label("b"),
                to_content: "b".into(),
                confidence: 0.8,
                articulation: "a is a b".into(),
            }],
            final_answer: final_answer.into(),
            confidence: 0.8,
            bonds_used: vec![(Seed::from_label("a"), BondType::IsA, Seed::from_label("b"))],
            no_chain: false,
            evidence_kind: EvidenceKind::Structured,
        }
    }

    fn associative_trace(query: &str, final_answer: &str) -> RT {
        let mut t = structured_trace(query, final_answer);
        t.evidence_kind = EvidenceKind::Associative;
        t
    }

    fn no_chain_rt(query: &str) -> RT {
        RT {
            query: query.into(),
            scope: TemporalScope::Current,
            steps: Vec::new(),
            final_answer: "I don't have a chain.".into(),
            confidence: 0.0,
            bonds_used: Vec::new(),
            no_chain: true,
            evidence_kind: EvidenceKind::Structured,
        }
    }

    fn one_belief() -> Vec<Belief> {
        vec![Belief {
            from: Seed::from_label("alpha"),
            via:  Seed::from_label("mid"),
            to:   Seed::from_label("omega"),
            from_content: "alpha".into(),
            via_content:  "mid".into(),
            to_content:   "omega".into(),
            w_from_via: 0.8, w_via_to: 0.8, confidence: 0.8,
            scar_reasons: Vec::new(), caveats: Vec::new(),
        }]
    }

    // ---- 10 unit tests on build_synthesis_prose ----

    #[test]
    fn q4_unit_structured_chain_is_referenced_in_prose() {
        let beliefs = one_belief();
        let trace = structured_trace("alpha", "alpha is a mid which has property omega.");
        let prose = build_synthesis_prose("alpha", &beliefs, Some(&trace));
        assert!(prose.contains("Based on the reasoning chain:"),
            "structured chain must be referenced; got: {prose}");
        assert!(prose.contains("alpha is a mid"),
            "final_answer text must appear in prose; got: {prose}");
    }

    #[test]
    fn q4_unit_associative_chain_falls_back_to_belief_summary() {
        let beliefs = one_belief();
        let trace = associative_trace("alpha", "alpha relates to mid which relates to omega.");
        let prose = build_synthesis_prose("alpha", &beliefs, Some(&trace));
        assert!(!prose.contains("Based on the reasoning chain:"),
            "Associative chain must NOT be referenced; got: {prose}");
        assert!((prose.contains("synthesized 1 two-hop belief") || prose.contains("Synthesized 1 two-hop belief")),
            "belief-only fallback must surface; got: {prose}");
    }

    #[test]
    fn q4_unit_no_chain_trace_falls_back() {
        let beliefs = one_belief();
        let trace = no_chain_rt("alpha");
        let prose = build_synthesis_prose("alpha", &beliefs, Some(&trace));
        assert!(!prose.contains("Based on the reasoning chain:"),
            "no_chain trace must NOT be referenced; got: {prose}");
        assert!((prose.contains("synthesized 1 two-hop belief") || prose.contains("Synthesized 1 two-hop belief")));
    }

    #[test]
    fn q4_unit_none_trace_uses_belief_summary() {
        let beliefs = one_belief();
        let prose = build_synthesis_prose("alpha", &beliefs, None);
        assert!(!prose.contains("Based on the reasoning chain:"),
            "None trace must NOT be referenced; got: {prose}");
        assert!(prose.contains("alpha"), "query echo expected; got: {prose}");
    }

    #[test]
    fn q4_unit_empty_beliefs_with_structured_chain_still_references_chain() {
        let trace = structured_trace("alpha", "alpha is a mid which has property omega.");
        let prose = build_synthesis_prose("alpha", &[], Some(&trace));
        assert!(prose.contains("Based on the reasoning chain:"),
            "chain referenced even when beliefs is empty; got: {prose}");
        assert!(
            prose.contains("no two-hop beliefs from recall")
                || prose.contains("Found no two-hop beliefs from recall"),
            "empty-beliefs phrasing expected; got: {prose}",
        );
    }

    #[test]
    fn q4_unit_empty_beliefs_no_trace() {
        let prose = build_synthesis_prose("alpha", &[], None);
        assert!(
            prose.contains("no two-hop beliefs from recall")
                || prose.contains("Found no two-hop beliefs from recall"),
            "empty-beliefs phrasing expected; got: {prose}",
        );
        assert!(!prose.contains("Based on the reasoning chain:"));
    }

    #[test]
    fn q4_unit_multi_belief_uses_top_summary() {
        let mut beliefs = one_belief();
        let mut second = beliefs[0].clone();
        second.from_content = "beta".into();
        second.via_content  = "gamma".into();
        second.to_content   = "delta".into();
        second.confidence   = 0.5;
        beliefs.push(second);
        let prose = build_synthesis_prose("q", &beliefs, None);
        assert!(
            prose.contains("synthesized 2 two-hop beliefs")
                || prose.contains("Synthesized 2 two-hop beliefs"),
            "multi-belief summary expected; got: {prose}",
        );
        assert!(prose.contains("top: alpha"),
            "top belief should be the highest-confidence one; got: {prose}");
    }

    #[test]
    fn q4_unit_empty_final_answer_falls_back() {
        let beliefs = one_belief();
        let mut trace = structured_trace("alpha", "");
        trace.no_chain = false;
        let prose = build_synthesis_prose("alpha", &beliefs, Some(&trace));
        assert!(!prose.contains("Based on the reasoning chain:"),
            "empty final_answer must NOT spawn a chain prefix; got: {prose}");
    }

    #[test]
    fn q4_unit_query_text_is_quoted_in_prose() {
        let beliefs = one_belief();
        let prose = build_synthesis_prose("why is the sky blue", &beliefs, None);
        assert!(prose.contains("\"why is the sky blue\""),
            "query echoed back in quotes; got: {prose}");
    }

    #[test]
    fn q4_unit_structured_prose_carries_query_too() {
        let beliefs = one_belief();
        let trace = structured_trace("alpha", "alpha is mid is omega.");
        let prose = build_synthesis_prose("alpha", &beliefs, Some(&trace));
        assert!(prose.contains("\"alpha\""),
            "structured-chain prose must still echo query; got: {prose}");
    }

    // ---- 4 integration tests: multi_think with chain present ----

    #[test]
    fn q4_int_chain_present_populates_reasoning_trace() {
        let mut brain = line_brain();
        let cfg = MultiAgentConfig::default();
        let trace = multi_think(&mut brain, "alpha", &cfg).unwrap();
        assert!(trace.reasoning_trace.is_some(),
            "reasoning_trace must be Some for connected line graph");
        let rt = trace.reasoning_trace.as_ref().unwrap();
        assert!(!rt.no_chain, "line graph alpha→mid→omega must produce a chain");
    }

    #[test]
    fn q4_int_structured_chain_appears_in_synthesis_prose() {
        let mut brain = line_brain();
        let cfg = MultiAgentConfig::default();
        let trace = multi_think(&mut brain, "alpha", &cfg).unwrap();
        // alpha -[IsA]-> mid -[HasProperty]-> omega is a structured pattern
        // (Hierarchical → PropertyChain). Prose must reference the chain.
        let rt = trace.reasoning_trace.as_ref().expect("trace built");
        if rt.evidence_kind == EvidenceKind::Structured && !rt.no_chain {
            assert!(
                trace.synthesis_prose.contains("Based on the reasoning chain:"),
                "structured chain must surface in synthesis_prose; rt.final_answer={:?}; prose={:?}",
                rt.final_answer, trace.synthesis_prose,
            );
        }
    }

    #[test]
    fn q4_int_synthesis_prose_is_never_empty_when_recall_succeeded() {
        let mut brain = line_brain();
        let cfg = MultiAgentConfig::default();
        let trace = multi_think(&mut brain, "alpha", &cfg).unwrap();
        assert!(!trace.recall.is_empty(), "recall must produce items");
        assert!(!trace.synthesis_prose.is_empty(),
            "synthesis_prose must be populated whenever recall produced anything");
    }

    #[test]
    fn q4_int_chain_in_prose_references_query() {
        let mut brain = line_brain();
        let cfg = MultiAgentConfig::default();
        let trace = multi_think(&mut brain, "alpha", &cfg).unwrap();
        assert!(trace.synthesis_prose.contains("alpha"),
            "prose must reference query; got: {}", trace.synthesis_prose);
    }

    // ---- 4 integration tests: multi_think fallback when no useful chain ----

    #[test]
    fn q4_int_associative_chain_falls_back() {
        // Pure RelatedTo chain classifies as Mixed/Associative. Prose
        // must NOT reference the chain.
        let mut brain = Brain::new();
        brain.teach("a", BondType::RelatedTo, "b", 0.8, Confidence::Verified);
        brain.teach("b", BondType::RelatedTo, "c", 0.8, Confidence::Verified);
        brain.network.recompute_degree_p90();
        let cfg = MultiAgentConfig::default();
        let trace = multi_think(&mut brain, "a", &cfg).unwrap();
        if let Some(rt) = trace.reasoning_trace.as_ref() {
            // If a Mixed chain was produced, prose must NOT reference it.
            if rt.evidence_kind == EvidenceKind::Associative && !rt.no_chain {
                assert!(
                    !trace.synthesis_prose.contains("Based on the reasoning chain:"),
                    "Associative chain leaked into prose; rt={:?}; prose={:?}",
                    rt.final_answer, trace.synthesis_prose,
                );
            }
        }
    }

    #[test]
    fn q4_int_isolated_recall_yields_no_chain_reference() {
        // Single isolated node: recall surfaces it but no chain exists.
        let mut brain = Brain::new();
        let only = Seed::from_label("loner");
        brain.network.add_node(crate::node::KnowledgeNode::with_seed(
            only, "loner".into(),
            crate::node::KnowledgeContent::Raw("".into()),
            Confidence::Verified, 0,
        ));
        let cfg = MultiAgentConfig::default();
        let trace = multi_think(&mut brain, "loner", &cfg).unwrap();
        assert!(
            !trace.synthesis_prose.contains("Based on the reasoning chain:"),
            "isolated recall must not produce a chain reference; got: {}",
            trace.synthesis_prose,
        );
    }

    #[test]
    fn q4_int_empty_recall_has_no_reasoning_trace() {
        // Query with no anchors at all (empty brain): recall is empty,
        // reasoning_trace must be None (we skip the call).
        let mut brain = Brain::new();
        let cfg = MultiAgentConfig::default();
        let trace = multi_think(&mut brain, "unknown_word_xyz", &cfg).unwrap();
        if trace.recall.is_empty() {
            assert!(trace.reasoning_trace.is_none(),
                "empty recall must skip reason_from_recall");
        }
    }

    #[test]
    fn q4_int_synthesis_still_runs_when_reasoning_trace_is_present() {
        // Regression: the Q.4 reasoning_trace pre-step must NOT suppress
        // the existing synthesis loop. Use line_brain (known to produce
        // hypotheses pre-Q.4) and assert hypotheses are still produced
        // with the new pre-step in place.
        let mut brain = line_brain();
        let cfg = MultiAgentConfig::default();
        let trace = multi_think(&mut brain, "alpha", &cfg).unwrap();
        assert!(
            !trace.hypotheses.is_empty(),
            "synthesis loop must still produce hypotheses with reasoning_trace pre-step",
        );
        // Port-A test_synthesis_agent_proposes_hypotheses contract still
        // holds: alpha-mid-omega is the canonical belief.
        let alpha = Seed::from_label("alpha");
        let mid = Seed::from_label("mid");
        let omega = Seed::from_label("omega");
        assert!(
            trace.hypotheses.iter().any(|bl| bl.from == alpha && bl.via == mid && bl.to == omega),
            "canonical alpha-mid-omega belief preserved post-Q.4",
        );
    }

    #[test]
    fn q4_int_reasoning_trace_does_not_perturb_existing_belief_count() {
        // Sanity: the reasoning_trace pre-step must not mutate brain state
        // in a way that changes the synthesis output count for line_brain.
        let mut brain1 = line_brain();
        let mut brain2 = line_brain();
        let cfg = MultiAgentConfig::default();
        let t1 = multi_think(&mut brain1, "alpha", &cfg).unwrap();
        let t2 = multi_think(&mut brain2, "alpha", &cfg).unwrap();
        assert_eq!(t1.hypotheses.len(), t2.hypotheses.len(),
            "deterministic belief count across runs");
        assert_eq!(t1.vetted.len(), t2.vetted.len(),
            "deterministic vetted count across runs");
    }

    #[test]
    fn q4_int_pre_q4_trace_construction_still_valid_via_serde_default() {
        // Round-trip: a JSON MultiAgentTrace WITHOUT the Q.4 fields must
        // deserialize cleanly thanks to #[serde(default)].
        let json = r#"{
            "multi_think_id": 42,
            "query": "x",
            "recall": [],
            "hypotheses": [],
            "vetted": [],
            "steps": [
                {"role": "Recall",    "elapsed_us": 0, "items_in": 0, "items_out": 0},
                {"role": "Synthesis", "elapsed_us": 0, "items_in": 0, "items_out": 0},
                {"role": "Critic",    "elapsed_us": 0, "items_in": 0, "items_out": 0}
            ],
            "started_ms": 0, "ended_ms": 0, "timed_out": false
        }"#;
        let parsed: MultiAgentTrace = serde_json::from_str(json)
            .expect("pre-Q.4 JSON must deserialize via serde defaults");
        assert!(parsed.reasoning_trace.is_none());
        assert!(parsed.synthesis_prose.is_empty());
    }
}
#[cfg(test)]
mod mapram_p5d_agents_tests {
    //! OP-BRAIN-MapRAM P5-batch2 (sub-batch B.2): byte-equivalence +
    //! resident-independence for the agents SYNTHESIS triangle walk
    //! (`run_synthesis`: A->B via `seam_walk_outgoing`, B->C via
    //! `seam_walk_outgoing`, both bond readings reconstructed from the
    //! carried payload). Beliefs are projected to their ADJACENCY/BOND-
    //! derived fields (from/via/to seeds + leg weight bits + confidence
    //! bits); the *_content fields are the separate resident "B" thread and
    //! are EXCLUDED (poison overwrites node content with "POISON"). The
    //! end-of-synthesis Agent-keyframe provenance write is a resident
    //! maintenance write that does not affect the returned beliefs or the
    //! cache, so fresh brains are used per mode for the equivalence check.
    use super::*;
    use crate::bond::{BondType, KeyframeSource, TemporalKeyframe};
    use crate::brain::Brain;
    use crate::manifest::{KeyFingerprint, Manifest, MANIFEST_FILE_NAME};
    use crate::scope::TemporalScope;
    use crate::seed::Seed;
    use tempfile::tempdir;

    fn kf(w: f32) -> Vec<TemporalKeyframe> {
        vec![TemporalKeyframe { timestamp: 100, weight: w, source: KeyframeSource::Baseline }]
    }
    fn lbl(n: &str) -> Seed { Seed::from_label(&format!("p5d_ag_{}", n)) }
    fn boot_v5(dir: &std::path::Path) -> Brain {
        let mut m = Manifest::new(KeyFingerprint::none());
        m.set_brain_format_v5();
        m.save_atomic(&dir.join(MANIFEST_FILE_NAME)).unwrap();
        Brain::load_v5(dir, None).expect("v5 brain must boot via load_v5")
    }
    const SPINE: &[(&str, &str, f32, BondType)] = &[
        ("q", "a", 0.90, BondType::RelatedTo),
        ("a", "b", 0.80, BondType::RelatedTo),
        ("q", "x", 0.70, BondType::RelatedTo),
        ("x", "y", 0.60, BondType::RelatedTo),
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
    fn recall() -> Vec<RecalledItem> {
        vec![
            RecalledItem { seed: lbl("q"), content: String::new(), activation: 1.0, depth: 0 },
            RecalledItem { seed: lbl("a"), content: String::new(), activation: 0.9, depth: 1 },
            RecalledItem { seed: lbl("b"), content: String::new(), activation: 0.8, depth: 2 },
            RecalledItem { seed: lbl("x"), content: String::new(), activation: 0.7, depth: 1 },
            RecalledItem { seed: lbl("y"), content: String::new(), activation: 0.6, depth: 2 },
        ]
    }
    #[allow(clippy::type_complexity)]
    fn proj(b: &[Belief]) -> Vec<(Vec<u8>, Vec<u8>, Vec<u8>, u32, u32, u32)> {
        b.iter().map(|x| (
            x.from.as_bytes().to_vec(), x.via.as_bytes().to_vec(), x.to.as_bytes().to_vec(),
            x.w_from_via.to_bits(), x.w_via_to.to_bits(), x.confidence.to_bits(),
        )).collect()
    }
    fn synth(brain: &mut Brain) -> Vec<Belief> {
        let cfg = MultiAgentConfig::default();
        run_synthesis(brain, 1, &recall(), &cfg, &TemporalScope::Current, 1_000_000)
    }

    /// The synthesis triangle walk is byte-identical legacy vs cache
    /// (canonical seam order both modes) on the bond-derived belief
    /// projection, and non-vacuous. Fresh brains per mode (run_synthesis
    /// performs a resident provenance write).
    #[test]
    fn p5d_agents_triangle_byte_equivalent() {
        let dir = tempdir().unwrap();
        { let b = boot_v5(dir.path()); teach(&b); }
        let mut bl = load_routed(dir.path());
        bl.set_adjacency_cache_backed(false);
        let legacy = proj(&synth(&mut bl));
        let mut bc = load_routed(dir.path());
        bc.set_adjacency_cache_backed(true);
        let cache = proj(&synth(&mut bc));
        assert_eq!(legacy, cache, "run_synthesis triangle byte-differs legacy vs cache");
        assert!(!cache.is_empty(), "run_synthesis produced no beliefs (vacuous)");
    }

    /// INDEPENDENCE: in cache mode the synthesis beliefs are invariant after
    /// poisoning every resident structure the legacy triangle reads. Sanity:
    /// the legacy triangle diverges (adjacency cleared -> no beliefs).
    #[test]
    fn p5d_agents_triangle_independent_of_resident() {
        let dir = tempdir().unwrap();
        { let b = boot_v5(dir.path()); teach(&b); }
        let mut brain = load_routed(dir.path());
        brain.set_adjacency_cache_backed(true);
        let base = proj(&synth(&mut brain));
        assert!(!base.is_empty(), "baseline synthesis vacuous");

        brain.network.mapram_test_poison_resident();
        brain.network.mapram_test_corrupt_resident_bond_types();
        brain.network.mapram_test_corrupt_resident_keyframes();

        brain.set_adjacency_cache_backed(true);
        assert_eq!(proj(&synth(&mut brain)), base, "cache synthesis beliefs changed under poison");

        brain.set_adjacency_cache_backed(false);
        assert_ne!(proj(&synth(&mut brain)), base, "poison must break the legacy triangle (else vacuous proof)");
    }
}
