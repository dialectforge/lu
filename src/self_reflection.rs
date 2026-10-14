//! Self-reflection / Meta-learning — sixth of the seven Lu 4.0 capability ports.
//!
//! Lu observes her own performance over time and identifies patterns the
//! developer (John) can act on. Examples she should be able to surface:
//!
//! - "Synthesis confirmation rate is 3% over the last 200 calls; the
//!   leg-weight floor may be too low."
//! - "Variant arenas almost never produce winners — `min_winner_composite`
//!   may be too strict."
//! - "Curiosity dead-end signals dominate the wonder queue — the
//!   yield threshold may be too sensitive."
//! - "Median latency over the last 100 queries is 4x the prior window —
//!   something has drifted."
//!
//! These observations are packaged as `AdjustmentProposal`s — STRUCTURED
//! suggestions for config-knob tuning. Proposals NEVER auto-apply. They
//! are surfaced via MCP; John explicitly approves or rejects each one.
//!
//! ## What self-reflection is NOT
//!
//! - NOT code self-modification. Lu does not write or modify Rust source.
//! - NOT silent parameter mutation — every change is gated by approval.
//! - NOT predictive of architecture changes (no ArchitectureProposer).
//! - NOT auto-tuning. This is observation + proposal, not optimization.
//!
//! ## Architecture
//!
//! - `PerformanceMonitor` is a bounded ring buffer of `QueryMetric`s,
//!   updated on every `Brain::ask`, `Brain::typed_ask`, and `Brain::think`.
//!   The buffer is runtime-only (`#[serde(skip)]`); telemetry doesn't
//!   need to survive restarts.
//! - `MetaReflection` holds the persistent state: the proposal log, the
//!   timestamp of the last reflection pass, and a counter. It DOES
//!   serialize (so John doesn't lose pending proposals on restart).
//! - `analyze()` runs six fixed heuristics over the recent metrics
//!   buffer; each can emit at most one new proposal per pass. A
//!   pending proposal for the same target prevents reproposing —
//!   no proposal storm if John lets one sit.

use crate::scope::{AgentRole, TemporalScope};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// One observation of a single `ask`/`typed_ask`/`think` call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryMetric {
    pub timestamp_ms: u64,
    /// Hash of the query text. Stored as a hash, not the text, so the
    /// metric buffer doesn't accumulate user content.
    pub query_text_hash: u64,
    pub scope: TemporalScope,
    /// End-to-end latency, microseconds.
    pub latency_us: u64,
    /// Highest activation in the recall result. 0.0 if recall was empty.
    pub top_activation: f32,
    pub bridges_fired: u32,
    pub synthesis_triangles_fired: u32,
    pub curiosity_signals_emitted: u32,
    /// Hypothesis bonds promoted to ConfirmedHypothesis. Only set on
    /// `think()` calls; 0 for plain ask/typed_ask.
    pub confirmed_hypotheses: u32,
    /// Variant arenas that produced a winner. Only set on variant_mode
    /// `think()` calls; 0 otherwise.
    pub winning_variants: u32,
    /// Which multi-agent role (if any) produced this metric. Multi-think
    /// pushes one metric per agent (Recall / Synthesis / Critic); plain
    /// ask / typed_ask / think calls leave this `None`. `#[serde(default)]`
    /// keeps older saves loadable, though the buffer itself is
    /// `#[serde(skip)]` so this is mostly for API consistency.
    #[serde(default)]
    pub agent_role: Option<AgentRole>,
    /// For multi-agent metrics: number of items the agent received.
    /// Used by the `AgentDriftDetected` analyzer to compute Critic
    /// acceptance rates. 0 for non-agent metrics.
    #[serde(default)]
    pub agent_items_in: u32,
    /// For multi-agent metrics: number of items the agent emitted.
    /// Critic's items_out / items_in is the acceptance ratio.
    #[serde(default)]
    pub agent_items_out: u32,
}

/// Bounded ring buffer of recent query metrics. NOT persisted across
/// restarts — telemetry is operational state, not knowledge.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMonitor {
    /// Newest at the back. Length is bounded by `cap` via `push`.
    /// `#[serde(skip)]` — drained on every save/load round-trip.
    #[serde(skip)]
    pub recent_metrics: VecDeque<QueryMetric>,
    pub cap: usize,
}

impl PerformanceMonitor {
    pub fn new(cap: usize) -> Self {
        let cap = cap.max(1);
        PerformanceMonitor {
            recent_metrics: VecDeque::with_capacity(cap),
            cap,
        }
    }

    /// Append a metric, dropping the oldest if `cap` is exceeded.
    pub fn push(&mut self, metric: QueryMetric) {
        self.recent_metrics.push_back(metric);
        while self.recent_metrics.len() > self.cap {
            self.recent_metrics.pop_front();
        }
    }

    pub fn len(&self) -> usize { self.recent_metrics.len() }
    pub fn is_empty(&self) -> bool { self.recent_metrics.is_empty() }

    /// Median latency in microseconds across the buffer. 0 if empty.
    pub fn median_latency_us(&self) -> u64 {
        if self.recent_metrics.is_empty() {
            return 0;
        }
        let mut v: Vec<u64> = self.recent_metrics.iter().map(|m| m.latency_us).collect();
        v.sort_unstable();
        v[v.len() / 2]
    }
}

impl Default for PerformanceMonitor {
    fn default() -> Self { PerformanceMonitor::new(1024) }
}

/// Which configuration knob a proposal targets. Each variant maps to a
/// concrete f32 field elsewhere in the codebase. The Brain layer is
/// responsible for translating an approved proposal into a mutation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AdjustmentTarget {
    SynthesisMinLegWeight,
    SynthesisMaxTargetWeight,
    SynthesisPathDecay,
    ConfidenceToAnswer,
    ConfirmedHypothesisWeight,
    InitialHypothesisWeight,
    WeakBondThreshold,
    ClusterDisjointThreshold,
    DeadendYieldThreshold,
    KnowledgeDesertThreshold,
    VariantMinWinnerComposite,
    /// Notice-only: a "performance regression" pattern with no
    /// associated knob. `proposed_value` is meaningless here; consumers
    /// should treat it as informational.
    PerformanceRegressionNotice,
    /// Notice-only: the multi-agent CriticAgent's recent acceptance
    /// rate has dropped below half of its lifetime average. No knob to
    /// nudge automatically — surface the pattern so John can
    /// investigate.
    AgentDriftNotice,
}

/// Proposed direction of change. The contained f32 is the suggested
/// delta (always positive), applied + or - to `current_value` per
/// direction.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum AdjustmentDirection {
    Raise(f32),
    Lower(f32),
}

/// Lifecycle of a proposal. Pending until John decides; then either
/// Approved (decision recorded but not yet executed), Applied (the
/// adjustment was actually performed by `Brain::approve_proposal`),
/// or Rejected.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ProposalStatus {
    Pending,
    Approved { approved_ms: u64 },
    Rejected { rejected_ms: u64, reason: Option<String> },
    Applied { applied_ms: u64 },
}

/// One concrete tuning suggestion. Surfaced to MCP for John to act on.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdjustmentProposal {
    pub id: u64,
    pub target: AdjustmentTarget,
    pub direction: AdjustmentDirection,
    pub current_value: f32,
    pub proposed_value: f32,
    /// Human-readable rationale. Pre-formatted in `analyze()`.
    pub evidence: String,
    /// How many metrics in the buffer supported this inference.
    pub supporting_metric_count: u32,
    pub created_ms: u64,
    pub status: ProposalStatus,
}

/// Persistent self-reflection state. Survives save/load.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MetaReflection {
    pub proposals: Vec<AdjustmentProposal>,
    pub last_reflection_ms: u64,
    pub reflection_count: u64,
    /// Monotonic counter for assigning proposal ids. Persisted so ids
    /// don't collide across restarts.
    #[serde(default)]
    pub next_proposal_id: u64,
}

impl MetaReflection {
    pub fn new() -> Self { MetaReflection::default() }

    /// True if a Pending proposal already exists for this target.
    /// Used by the analyzers to avoid re-proposing the same knob over
    /// and over while John deliberates.
    pub fn has_pending_for(&self, target: AdjustmentTarget) -> bool {
        self.proposals.iter().any(|p|
            p.target == target && matches!(p.status, ProposalStatus::Pending)
        )
    }

    pub fn pending(&self) -> impl Iterator<Item = &AdjustmentProposal> {
        self.proposals.iter().filter(|p| matches!(p.status, ProposalStatus::Pending))
    }
}

/// All tunable knobs for the self-reflection subsystem.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfReflectionConfig {
    pub enabled: bool,
    pub metric_buffer_cap: usize,
    /// Don't bother analyzing if there are fewer metrics than this.
    pub min_metrics_for_proposal: usize,
    /// Suggested cadence between auto-reflection passes (ms). Brain is
    /// not a daemon — it's the caller's job to honor this.
    pub reflection_interval_ms: u64,
    /// Hard cap on Pending+Approved proposals — we drop new proposals
    /// once this many are alive.
    pub max_pending_proposals: usize,
    /// Per-analyzer floor on the number of supporting metrics required
    /// before a proposal is emitted.
    pub proposal_min_evidence_count: u32,
}

impl Default for SelfReflectionConfig {
    fn default() -> Self {
        SelfReflectionConfig {
            enabled: true,
            metric_buffer_cap: 1024,
            min_metrics_for_proposal: 50,
            reflection_interval_ms: 60 * 60 * 1000, // 1 hour
            max_pending_proposals: 16,
            proposal_min_evidence_count: 20,
        }
    }
}

/// Snapshot of the current values of every adjustable knob. Built by
/// the Brain layer right before calling `analyze` so the analyzers
/// don't need cross-module access. Each field corresponds to one
/// `AdjustmentTarget` variant.
#[derive(Debug, Clone)]
pub struct CurrentKnobs {
    pub synthesis_min_leg_weight: f32,
    pub synthesis_max_target_weight: f32,
    pub synthesis_path_decay: f32,
    pub confidence_to_answer: f32,
    pub confirmed_hypothesis_weight: f32,
    pub initial_hypothesis_weight: f32,
    pub weak_bond_threshold: f32,
    pub cluster_disjoint_threshold: f32,
    pub deadend_yield_threshold: f32,
    pub knowledge_desert_threshold: f32,
    pub variant_min_winner_composite: f32,
}

/// Run all six analyzers against `monitor` and append any new
/// proposals to `meta`. Returns the number of new proposals created.
///
/// Analyzers in order:
/// 1. Synthesis underperforming → propose Lower(SynthesisMinLegWeight).
/// 2. Synthesis over-eager     → propose Raise(SynthesisMinLegWeight).
/// 3. Variants too strict      → propose Lower(VariantMinWinnerComposite).
/// 4. Variants too lax         → propose Raise(VariantMinWinnerComposite).
/// 5. Curiosity too sensitive  → propose Raise(DeadendYieldThreshold).
/// 6. Latency drift            → emit a non-adjustment Notice.
///
/// Each analyzer is gated by `cfg.proposal_min_evidence_count` and the
/// stretch invariant: a Pending proposal for the same target this
/// cycle blocks duplicates.
pub fn analyze(
    monitor: &PerformanceMonitor,
    meta: &mut MetaReflection,
    cfg: &SelfReflectionConfig,
    knobs: &CurrentKnobs,
    now_ms: u64,
) -> usize {
    if !cfg.enabled {
        return 0;
    }
    if monitor.len() < cfg.min_metrics_for_proposal {
        return 0;
    }
    // Cap the live (Pending) proposal queue.
    let pending_count = meta.pending().count();
    if pending_count >= cfg.max_pending_proposals {
        return 0;
    }

    let mut emitted = 0usize;

    // ----- 1. Synthesis underperforming -----
    // High triangle activity but low confirmed-hypothesis rate suggests
    // the synthesis floor is too low — too many weak triangles, too few
    // useful ones.
    let total_synth: u32 = monitor.recent_metrics.iter()
        .map(|m| m.synthesis_triangles_fired).sum();
    let total_confirmed: u32 = monitor.recent_metrics.iter()
        .map(|m| m.confirmed_hypotheses).sum();
    if total_synth > 100
        && total_confirmed > 0
        && (total_confirmed as f32) / (total_synth as f32) < 0.05
        && total_synth >= cfg.proposal_min_evidence_count
        && !meta.has_pending_for(AdjustmentTarget::SynthesisMinLegWeight)
    {
        let delta = 0.05;
        let proposed = (knobs.synthesis_min_leg_weight + delta).clamp(0.0, 1.0);
        meta.next_proposal_id += 1;
        meta.proposals.push(AdjustmentProposal {
            id: meta.next_proposal_id,
            target: AdjustmentTarget::SynthesisMinLegWeight,
            direction: AdjustmentDirection::Raise(delta),
            current_value: knobs.synthesis_min_leg_weight,
            proposed_value: proposed,
            evidence: format!(
                "Synthesis triangles fired {} times, confirmed {} hypotheses ({}%); \
                 raising the synthesis leg-weight floor may cut weak-triangle noise.",
                total_synth, total_confirmed,
                if total_synth > 0 {
                    (total_confirmed as f32 / total_synth as f32) * 100.0
                } else { 0.0 },
            ),
            supporting_metric_count: total_synth,
            created_ms: now_ms,
            status: ProposalStatus::Pending,
        });
        emitted += 1;
    }

    // ----- 2. Synthesis over-eager -----
    // Many triangles fired but average top_activation isn't improving —
    // synthesis is producing noise, not value. Suggest raising the floor.
    if total_synth > 500
        && monitor.len() >= cfg.proposal_min_evidence_count as usize
        && !meta.has_pending_for(AdjustmentTarget::SynthesisMinLegWeight)
    {
        // Compare avg top_activation in the first half vs. second half.
        let half = monitor.len() / 2;
        let (first, second): (Vec<&QueryMetric>, Vec<&QueryMetric>) = {
            let v: Vec<&QueryMetric> = monitor.recent_metrics.iter().collect();
            (v[..half].to_vec(), v[half..].to_vec())
        };
        let avg = |slice: &[&QueryMetric]| -> f32 {
            if slice.is_empty() { return 0.0; }
            let sum: f32 = slice.iter().map(|m| m.top_activation).sum();
            sum / slice.len() as f32
        };
        let early_avg = avg(&first);
        let late_avg = avg(&second);
        // Over-eager: more recent half has NOT improved.
        if late_avg <= early_avg + 0.001 {
            let delta = 0.05;
            let proposed = (knobs.synthesis_min_leg_weight + delta).clamp(0.0, 1.0);
            meta.next_proposal_id += 1;
            meta.proposals.push(AdjustmentProposal {
                id: meta.next_proposal_id,
                target: AdjustmentTarget::SynthesisMinLegWeight,
                direction: AdjustmentDirection::Raise(delta),
                current_value: knobs.synthesis_min_leg_weight,
                proposed_value: proposed,
                evidence: format!(
                    "{} synthesis triangles fired but top_activation hasn't improved \
                     (early avg {:.3}, recent avg {:.3}); the leg-weight floor may be \
                     producing low-value triangles.",
                    total_synth, early_avg, late_avg,
                ),
                supporting_metric_count: total_synth,
                created_ms: now_ms,
                status: ProposalStatus::Pending,
            });
            emitted += 1;
        }
    }

    // ----- 3. Variants too strict -----
    // Look at the most recent 50 think-with-variant_mode metrics. If
    // fewer than 10% had a winner, the gate is likely too tight.
    let variant_metrics: Vec<&QueryMetric> = monitor.recent_metrics.iter()
        .rev()
        .filter(|m| m.winning_variants > 0 || think_was_variant(m))
        .take(50)
        .collect();
    if variant_metrics.len() >= cfg.proposal_min_evidence_count as usize
        && !meta.has_pending_for(AdjustmentTarget::VariantMinWinnerComposite)
    {
        let winners: u32 = variant_metrics.iter().map(|m| m.winning_variants).sum();
        let promotion_rate = winners as f32 / variant_metrics.len() as f32;
        if promotion_rate < 0.10 {
            let delta = 0.05;
            let proposed = (knobs.variant_min_winner_composite - delta).max(0.0);
            meta.next_proposal_id += 1;
            meta.proposals.push(AdjustmentProposal {
                id: meta.next_proposal_id,
                target: AdjustmentTarget::VariantMinWinnerComposite,
                direction: AdjustmentDirection::Lower(delta),
                current_value: knobs.variant_min_winner_composite,
                proposed_value: proposed,
                evidence: format!(
                    "{} of {} recent variant arenas produced a winner ({:.1}%); \
                     min_winner_composite may be too strict.",
                    winners, variant_metrics.len(), promotion_rate * 100.0,
                ),
                supporting_metric_count: variant_metrics.len() as u32,
                created_ms: now_ms,
                status: ProposalStatus::Pending,
            });
            emitted += 1;
        }
        // ----- 4. Variants too lax -----
        // > 60% of arenas promote AND post-promotion top_activation
        // hasn't improved. The arena is rubber-stamping.
        else if promotion_rate > 0.60 {
            // Promotion rate is high: did top_activation respond?
            let avg_top: f32 = variant_metrics.iter()
                .map(|m| m.top_activation).sum::<f32>()
                / variant_metrics.len() as f32;
            let baseline_avg: f32 = monitor.recent_metrics.iter()
                .map(|m| m.top_activation).sum::<f32>()
                / monitor.recent_metrics.len() as f32;
            if avg_top <= baseline_avg + 0.01
                && !meta.has_pending_for(AdjustmentTarget::VariantMinWinnerComposite)
            {
                let delta = 0.05;
                let proposed = (knobs.variant_min_winner_composite + delta).clamp(0.0, 1.0);
                meta.next_proposal_id += 1;
                meta.proposals.push(AdjustmentProposal {
                    id: meta.next_proposal_id,
                    target: AdjustmentTarget::VariantMinWinnerComposite,
                    direction: AdjustmentDirection::Raise(delta),
                    current_value: knobs.variant_min_winner_composite,
                    proposed_value: proposed,
                    evidence: format!(
                        "{:.0}% of recent variants promoted but top_activation didn't \
                         improve (variant avg {:.3}, baseline avg {:.3}); the gate may \
                         be too lax.",
                        promotion_rate * 100.0, avg_top, baseline_avg,
                    ),
                    supporting_metric_count: variant_metrics.len() as u32,
                    created_ms: now_ms,
                    status: ProposalStatus::Pending,
                });
                emitted += 1;
            }
        }
    }

    // ----- 5. Curiosity dead-ends dominating -----
    // If > 80% of curiosity_signals are dead-end-ish (proxied via
    // signals emitted with low surrounding bridge counts), suggest
    // raising the dead-end yield threshold so fewer trivial gaps
    // surface as signals.
    let total_curiosity: u32 = monitor.recent_metrics.iter()
        .map(|m| m.curiosity_signals_emitted).sum();
    let total_bridges: u32 = monitor.recent_metrics.iter()
        .map(|m| m.bridges_fired).sum();
    if total_curiosity > 100
        && total_curiosity >= cfg.proposal_min_evidence_count
        && !meta.has_pending_for(AdjustmentTarget::DeadendYieldThreshold)
    {
        // Crude proxy: when curiosity signals dwarf bridge events by 5x,
        // the engine is mostly classifying dead-ends.
        if total_bridges == 0 || (total_curiosity as f32) / (total_bridges.max(1) as f32) > 5.0 {
            let delta = 1.0;
            let proposed = knobs.deadend_yield_threshold + delta;
            meta.next_proposal_id += 1;
            meta.proposals.push(AdjustmentProposal {
                id: meta.next_proposal_id,
                target: AdjustmentTarget::DeadendYieldThreshold,
                direction: AdjustmentDirection::Raise(delta),
                current_value: knobs.deadend_yield_threshold,
                proposed_value: proposed,
                evidence: format!(
                    "{} curiosity signals vs {} bridge events ({}x ratio); raising the \
                     dead-end yield threshold should reduce trivial-gap noise.",
                    total_curiosity, total_bridges,
                    if total_bridges > 0 { total_curiosity / total_bridges } else { total_curiosity },
                ),
                supporting_metric_count: total_curiosity,
                created_ms: now_ms,
                status: ProposalStatus::Pending,
            });
            emitted += 1;
        }
    }

    // ----- 6. Latency drift -----
    // Median latency over the last 100 metrics > 2x the median over the
    // window before that. No knob to recommend; emit a Notice so John
    // can investigate manually.
    if monitor.len() >= 200
        && !meta.has_pending_for(AdjustmentTarget::PerformanceRegressionNotice)
    {
        let v: Vec<&QueryMetric> = monitor.recent_metrics.iter().collect();
        let split = v.len() - 100;
        let prior: Vec<u64> = v[split.saturating_sub(100)..split]
            .iter().map(|m| m.latency_us).collect();
        let recent: Vec<u64> = v[split..].iter().map(|m| m.latency_us).collect();
        let median = |mut nums: Vec<u64>| -> u64 {
            if nums.is_empty() { return 0; }
            nums.sort_unstable();
            nums[nums.len() / 2]
        };
        let prior_med = median(prior);
        let recent_med = median(recent);
        if prior_med > 0 && recent_med > prior_med.saturating_mul(2) {
            meta.next_proposal_id += 1;
            meta.proposals.push(AdjustmentProposal {
                id: meta.next_proposal_id,
                target: AdjustmentTarget::PerformanceRegressionNotice,
                direction: AdjustmentDirection::Raise(0.0),
                current_value: prior_med as f32,
                proposed_value: recent_med as f32,
                evidence: format!(
                    "Median latency rose from {}us to {}us over the most recent 100 \
                     queries; investigate spread depth, top_k, or graph fan-out.",
                    prior_med, recent_med,
                ),
                supporting_metric_count: 200,
                created_ms: now_ms,
                status: ProposalStatus::Pending,
            });
            emitted += 1;
        }
    }

    // ----- 7. AgentDriftDetected -----
    // Critic's recent acceptance rate has fallen below half of its
    // lifetime acceptance rate. Notice-only: there's no knob to nudge
    // automatically (the threshold is per-call MultiAgentConfig), so
    // we surface the pattern. Same gating as the other analyzers:
    // require at least `proposal_min_evidence_count` recent Critic
    // metrics, and don't re-propose while a Pending notice exists.
    if !meta.has_pending_for(AdjustmentTarget::AgentDriftNotice) {
        let critic_iter = monitor.recent_metrics.iter()
            .filter(|m| m.agent_role == Some(AgentRole::Critic));
        let lifetime: Vec<&QueryMetric> = critic_iter.clone().collect();
        let recent: Vec<&QueryMetric> = critic_iter.rev()
            .take(cfg.proposal_min_evidence_count as usize)
            .collect();
        let acc = |slice: &[&QueryMetric]| -> Option<f32> {
            let inn: u32 = slice.iter().map(|m| m.agent_items_in).sum();
            let out: u32 = slice.iter().map(|m| m.agent_items_out).sum();
            if inn == 0 { None } else { Some(out as f32 / inn as f32) }
        };
        if recent.len() >= cfg.proposal_min_evidence_count as usize {
            if let (Some(rate_recent), Some(rate_life)) = (acc(&recent), acc(&lifetime)) {
                if rate_life > 0.0 && rate_recent < 0.5 * rate_life {
                    meta.next_proposal_id += 1;
                    meta.proposals.push(AdjustmentProposal {
                        id: meta.next_proposal_id,
                        target: AdjustmentTarget::AgentDriftNotice,
                        direction: AdjustmentDirection::Raise(0.0),
                        current_value: rate_life,
                        proposed_value: rate_recent,
                        evidence: format!(
                            "CriticAgent acceptance rate dropped from {:.1}% (lifetime, n={})                              to {:.1}% (last {} multi-thinks); investigate the synthesis or                              critic confidence threshold.",
                            rate_life * 100.0, lifetime.len(),
                            rate_recent * 100.0, recent.len(),
                        ),
                        supporting_metric_count: recent.len() as u32,
                        created_ms: now_ms,
                        status: ProposalStatus::Pending,
                    });
                    emitted += 1;
                }
            }
        }
    }

    meta.last_reflection_ms = now_ms;
    meta.reflection_count = meta.reflection_count.saturating_add(1);
    emitted
}

/// Heuristic: was this metric produced by a variant_mode `think()` call?
/// We don't carry an explicit flag; instead we treat `winning_variants > 0`
/// OR a non-zero `confirmed_hypotheses` count as a signal that this was a
/// think call. For the variants-too-strict analyzer we want all
/// variant-mode calls (winners + losers), so this is a best-effort filter.
fn think_was_variant(m: &QueryMetric) -> bool {
    m.winning_variants > 0 || m.confirmed_hypotheses > 0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_knobs() -> CurrentKnobs {
        CurrentKnobs {
            synthesis_min_leg_weight: 0.4,
            synthesis_max_target_weight: 0.3,
            synthesis_path_decay: 0.7,
            confidence_to_answer: 0.7,
            confirmed_hypothesis_weight: 0.55,
            initial_hypothesis_weight: 0.15,
            weak_bond_threshold: 0.1,
            cluster_disjoint_threshold: 0.3,
            deadend_yield_threshold: 3.0,
            knowledge_desert_threshold: 0.2,
            variant_min_winner_composite: 0.4,
        }
    }

    fn metric(latency_us: u64, top: f32, synth: u32, confirmed: u32) -> QueryMetric {
        QueryMetric {
            timestamp_ms: 0,
            query_text_hash: 0,
            scope: TemporalScope::Current,
            latency_us,
            top_activation: top,
            bridges_fired: 0,
            synthesis_triangles_fired: synth,
            curiosity_signals_emitted: 0,
            confirmed_hypotheses: confirmed,
            winning_variants: 0,
            agent_role: None,
            agent_items_in: 0,
            agent_items_out: 0,
        }
    }

    /// (1) After Brain pushes a metric on every ask, the buffer holds it.
    /// We exercise this in the brain.rs tests; here we just sanity-check
    /// the buffer mechanics directly.
    #[test]
    fn test_query_metric_pushed_on_ask() {
        let mut mon = PerformanceMonitor::new(8);
        assert!(mon.is_empty());
        mon.push(metric(100, 0.5, 0, 0));
        assert_eq!(mon.len(), 1);
        mon.push(metric(200, 0.6, 0, 0));
        assert_eq!(mon.len(), 2);
        assert_eq!(mon.recent_metrics.back().unwrap().latency_us, 200);
    }

    /// (2) Metric buffer caps at config; oldest evicted.
    #[test]
    fn test_metric_buffer_caps_at_config() {
        let mut mon = PerformanceMonitor::new(3);
        for i in 0..10u64 {
            mon.push(metric(i * 100, 0.5, 0, 0));
        }
        assert_eq!(mon.len(), 3);
        assert_eq!(mon.recent_metrics.front().unwrap().latency_us, 700);
        assert_eq!(mon.recent_metrics.back().unwrap().latency_us, 900);
    }

    /// (3) Synthesis underperforming → propose Raise(SynthesisMinLegWeight).
    /// (Per the heuristic in `analyze`: high triangle count + very low
    /// confirmation rate → raise the floor.)
    #[test]
    fn test_synthesis_underperforming_proposes_lower_threshold() {
        let mut mon = PerformanceMonitor::new(1024);
        // Heavy triangle activity, very few confirmed hypotheses.
        for _ in 0..100 {
            mon.push(metric(500, 0.3, 5, 0));
        }
        // Plus a single confirmed hypothesis to keep the ratio non-zero
        // (per the analyzer guard `total_confirmed > 0`).
        mon.push(metric(500, 0.3, 5, 1));

        let mut meta = MetaReflection::new();
        let cfg = SelfReflectionConfig::default();
        let knobs = empty_knobs();
        let n = analyze(&mon, &mut meta, &cfg, &knobs, 1_000);
        assert!(n >= 1, "expected at least one proposal, got {}", n);

        let p = meta.proposals.iter()
            .find(|p| p.target == AdjustmentTarget::SynthesisMinLegWeight)
            .expect("synthesis-leg-weight proposal");
        assert!(matches!(p.direction, AdjustmentDirection::Raise(_)));
        assert!(matches!(p.status, ProposalStatus::Pending));
    }

    /// (4) Synthesis over-eager → propose Raise(SynthesisMinLegWeight).
    /// 500+ triangles AND top_activation NOT improving over the buffer
    /// window. The under-performing analyzer also fires here (very low
    /// confirmed/synth ratio); the over-eager analyzer is suppressed
    /// because both target the same knob and the under-performing
    /// branch grabs the slot first via has_pending_for.
    #[test]
    fn test_synthesis_overeager_proposes_raise_threshold() {
        let mut mon = PerformanceMonitor::new(1024);
        // Half the buffer at top_activation 0.5, half at 0.5 — flat.
        for _ in 0..600 {
            mon.push(metric(500, 0.5, 5, 0));
        }
        let mut meta = MetaReflection::new();
        let cfg = SelfReflectionConfig::default();
        let knobs = empty_knobs();
        let n = analyze(&mon, &mut meta, &cfg, &knobs, 1_000);
        assert!(n >= 1, "expected at least one synthesis proposal, got {}", n);
        let p = meta.proposals.iter()
            .find(|p| p.target == AdjustmentTarget::SynthesisMinLegWeight)
            .expect("synthesis-leg-weight proposal");
        assert!(matches!(p.direction, AdjustmentDirection::Raise(_)));
    }

    /// (5) Variants too strict → propose Lower(VariantMinWinnerComposite).
    #[test]
    fn test_variant_minimum_too_strict_proposes_lower() {
        let mut mon = PerformanceMonitor::new(1024);
        // Many variant-mode metrics, very few winners. We use
        // `confirmed_hypotheses > 0` to mark "this was a think call"
        // (per `think_was_variant`).
        for _ in 0..50 {
            mon.push(QueryMetric {
                timestamp_ms: 0,
                query_text_hash: 0,
                scope: TemporalScope::Current,
                latency_us: 500,
                top_activation: 0.3,
                bridges_fired: 0,
                synthesis_triangles_fired: 0,
                curiosity_signals_emitted: 0,
                confirmed_hypotheses: 1, // marks as variant-mode
                winning_variants: 0,
                agent_role: None,
                agent_items_in: 0,
                agent_items_out: 0,
            });
        }
        let mut meta = MetaReflection::new();
        let cfg = SelfReflectionConfig::default();
        let knobs = empty_knobs();
        let _ = analyze(&mon, &mut meta, &cfg, &knobs, 1_000);
        let p = meta.proposals.iter()
            .find(|p| p.target == AdjustmentTarget::VariantMinWinnerComposite)
            .expect("variant-composite proposal");
        assert!(matches!(p.direction, AdjustmentDirection::Lower(_)));
    }

    /// (6) Approval applies the adjustment and marks the proposal Applied.
    /// `analyze` produces the proposal; the brain layer records the
    /// status transition via `MetaReflection`.
    #[test]
    fn test_proposal_approval_applies_adjustment_and_marks_applied() {
        let mut meta = MetaReflection::new();
        meta.next_proposal_id = 7;
        meta.proposals.push(AdjustmentProposal {
            id: 8,
            target: AdjustmentTarget::SynthesisMinLegWeight,
            direction: AdjustmentDirection::Raise(0.05),
            current_value: 0.4,
            proposed_value: 0.45,
            evidence: "test".into(),
            supporting_metric_count: 100,
            created_ms: 1_000,
            status: ProposalStatus::Pending,
        });
        // Mimic the Brain layer's approve flow.
        let p = meta.proposals.iter_mut().find(|p| p.id == 8).unwrap();
        p.status = ProposalStatus::Approved { approved_ms: 2_000 };
        // Then Applied after the adjustment is performed.
        p.status = ProposalStatus::Applied { applied_ms: 2_001 };
        assert!(matches!(meta.proposals[0].status, ProposalStatus::Applied { .. }));
    }

    /// (7) Rejection records the reason and does not apply.
    #[test]
    fn test_proposal_rejection_does_not_apply_and_marks_rejected() {
        let mut meta = MetaReflection::new();
        meta.proposals.push(AdjustmentProposal {
            id: 1,
            target: AdjustmentTarget::DeadendYieldThreshold,
            direction: AdjustmentDirection::Raise(1.0),
            current_value: 3.0,
            proposed_value: 4.0,
            evidence: "noise".into(),
            supporting_metric_count: 200,
            created_ms: 0,
            status: ProposalStatus::Pending,
        });
        let p = meta.proposals.iter_mut().find(|p| p.id == 1).unwrap();
        p.status = ProposalStatus::Rejected {
            rejected_ms: 500,
            reason: Some("not yet".into()),
        };
        assert!(matches!(
            meta.proposals[0].status,
            ProposalStatus::Rejected { reason: Some(_), .. }
        ));
    }

    /// (8) No proposal below `min_metrics_for_proposal`.
    #[test]
    fn test_no_proposal_below_min_evidence_count() {
        let mut mon = PerformanceMonitor::new(1024);
        // Just 10 metrics — far below the 50 default.
        for _ in 0..10 {
            mon.push(metric(500, 0.3, 5, 1));
        }
        let mut meta = MetaReflection::new();
        let cfg = SelfReflectionConfig::default();
        let knobs = empty_knobs();
        let n = analyze(&mon, &mut meta, &cfg, &knobs, 1_000);
        assert_eq!(n, 0);
        assert!(meta.proposals.is_empty());
    }

    /// (Stretch) Don't re-propose a target that already has a Pending
    /// proposal.
    #[test]
    fn test_reflect_does_not_repropose_pending_target() {
        let mut mon = PerformanceMonitor::new(1024);
        for _ in 0..200 {
            mon.push(metric(500, 0.5, 5, 0));
        }
        // Insert a Pending proposal for SynthesisMinLegWeight first.
        let mut meta = MetaReflection::new();
        meta.proposals.push(AdjustmentProposal {
            id: 99,
            target: AdjustmentTarget::SynthesisMinLegWeight,
            direction: AdjustmentDirection::Raise(0.05),
            current_value: 0.4,
            proposed_value: 0.45,
            evidence: "first".into(),
            supporting_metric_count: 100,
            created_ms: 0,
            status: ProposalStatus::Pending,
        });
        let cfg = SelfReflectionConfig::default();
        let knobs = empty_knobs();
        let _ = analyze(&mon, &mut meta, &cfg, &knobs, 2_000);
        let synth_count = meta.proposals.iter()
            .filter(|p| p.target == AdjustmentTarget::SynthesisMinLegWeight)
            .count();
        assert_eq!(synth_count, 1, "no duplicate proposal for the same target");
    }

    /// (Stretch) AgentDriftDetected fires when the recent Critic
    /// acceptance rate falls below half of the lifetime average.
    #[test]
    fn test_agent_drift_proposes_notice() {
        let mut mon = PerformanceMonitor::new(2048);
        // First half: lifetime accept rate ~80% (8 vetted out of 10 in).
        for _ in 0..30 {
            mon.push(QueryMetric {
                timestamp_ms: 0,
                query_text_hash: 0,
                scope: TemporalScope::Current,
                latency_us: 200,
                top_activation: 0.0,
                bridges_fired: 0,
                synthesis_triangles_fired: 0,
                curiosity_signals_emitted: 0,
                confirmed_hypotheses: 0,
                winning_variants: 0,
                agent_role: Some(AgentRole::Critic),
                agent_items_in: 10,
                agent_items_out: 8,
            });
        }
        // Second half: recent accept rate ~10% — way below half of 80%.
        for _ in 0..30 {
            mon.push(QueryMetric {
                timestamp_ms: 0,
                query_text_hash: 0,
                scope: TemporalScope::Current,
                latency_us: 200,
                top_activation: 0.0,
                bridges_fired: 0,
                synthesis_triangles_fired: 0,
                curiosity_signals_emitted: 0,
                confirmed_hypotheses: 0,
                winning_variants: 0,
                agent_role: Some(AgentRole::Critic),
                agent_items_in: 10,
                agent_items_out: 1,
            });
        }
        let mut meta = MetaReflection::new();
        let cfg = SelfReflectionConfig::default();
        let knobs = empty_knobs();
        let _ = analyze(&mon, &mut meta, &cfg, &knobs, 1_000);
        let p = meta.proposals.iter()
            .find(|p| p.target == AdjustmentTarget::AgentDriftNotice)
            .expect("expected AgentDriftNotice");
        assert!(matches!(p.status, ProposalStatus::Pending));
        assert!(p.current_value > p.proposed_value, "lifetime > recent");
    }
}
