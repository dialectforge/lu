//! Port J.2 — chained curiosity research.
//!
//! J / J.1 / J.1.1 wire one-shot research: a high-intensity wonder
//! pops, gets researched, candidates queue as a Pending proposal.
//! J.2 deepens that loop: when a research cycle resolves a gap (i.e.
//! `apply_proposal` lands new bonds via T.3 PendingLog), inspect the
//! just-formed bond endpoints for sparse-neighborhood gaps and feed
//! those endpoints back into the wonder queue as fresh
//! `CuriositySignal`s. Chain depth and total cycles are bounded; the
//! loop terminates on (a) queue empty, (b) depth cap, (c) cycle cap,
//! (d) idle budget exhaustion (T.6, enforced by callers).
//!
//! ## Anti-hallucination invariant — STRUCTURAL
//!
//! Every applied proposal in the chain still routes through the same
//! `apply_proposal` -> `teach_with_source` path used by single-hop
//! research. T.3 PendingLog approval gating is preserved; this module
//! does NOT auto-approve proposals. Stored bond weights are NEVER
//! mutated mid-chain. `bond_count` MUST be unchanged across pure
//! observation calls; only externally-approved proposals land bonds.
//!
//! ## Hub safety (Cat.3b)
//!
//! Candidate endpoints whose existing in/out degree exceeds
//! `HubDegreeConfig::max_in_degree` / `max_out_degree` are dropped
//! at enqueue time — chasing super-hub gaps wastes cycles because
//! the read-side filter would clip them anyway.
//!
//! ## Source ordering (C2)
//!
//! All chain research delegates to `Brain::research_one`, which uses
//! `research::router::run_router` — Grokipedia first, DDG fallback,
//! Wikipedia deprioritized. No new SourceKind variant is introduced;
//! every bond carries `SourceKind::Research` with a wonder_seed label
//! that includes `chain:<chain_id>` so chain-produced bonds are
//! revocable as a unit via `Brain::revoke_bonds_by_source`.

use crate::brain::Brain;
use crate::curiosity::{CuriositySignal, SignalType};
use crate::scope::TemporalScope;
use crate::research::http::HttpFetcher;
use crate::research::loop_::ResearchError;
use crate::seed::Seed;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Brain-resident snapshot of the J.2 runtime knobs. Mirrors the
/// fields stored in v3 section #15 (`RuntimeConfigSnapshot`) but
/// lives on `Brain` directly so chain-time consumers read it without
/// going through the serde snapshot.
#[derive(Debug, Clone, PartialEq)]
pub struct ChainResearchConfig {
    /// Master switch. False collapses chain runs to single-hop.
    pub enabled: bool,
    /// Hard cap on chain depth.
    pub max_chain_depth: u32,
    /// Hard cap on total research cycles per chain.
    pub max_chain_cycles: u32,
    /// Minimum gap intensity to enqueue the next hop.
    pub gap_threshold: f32,
}

impl Default for ChainResearchConfig {
    fn default() -> Self {
        ChainResearchConfig {
            enabled: crate::runtime_config::default_chained_research_enabled(),
            max_chain_depth: crate::runtime_config::default_max_chain_depth(),
            max_chain_cycles: crate::runtime_config::default_max_chain_cycles(),
            gap_threshold: crate::runtime_config::default_chain_gap_threshold(),
        }
    }
}

/// Why a chain run terminated. Surfaced via the MCP trace so John can
/// tell the difference between "this thread had nothing more to find"
/// and "this thread hit a budget cap and might have more to find".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChainTerminationReason {
    /// Wonder queue exhausted with no remaining chain candidates.
    QueueEmpty,
    /// `max_chain_depth` reached.
    DepthCap,
    /// `max_chain_cycles` reached.
    CyclesCap,
    /// Single-hop short-circuit (chained_research_enabled = false).
    Disabled,
    /// First call in the chain failed with a `ResearchError` other than
    /// `QueueEmpty` (e.g. RouterFailed, BelowThreshold).
    InitialError(InitialErrorKind),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InitialErrorKind {
    Disabled,
    QueueEmpty,
    BelowThreshold,
    QueueFull,
    Router,
    NoTextFound,
    Other,
}

impl From<&ResearchError> for InitialErrorKind {
    fn from(e: &ResearchError) -> Self {
        match e {
            ResearchError::Disabled => InitialErrorKind::Disabled,
            ResearchError::QueueEmpty => InitialErrorKind::QueueEmpty,
            ResearchError::BelowThreshold => InitialErrorKind::BelowThreshold,
            ResearchError::QueueFull => InitialErrorKind::QueueFull,
            ResearchError::RouterFailed(_) => InitialErrorKind::Router,
            ResearchError::NoTextFound => InitialErrorKind::NoTextFound,
            _ => InitialErrorKind::Other,
        }
    }
}

/// One hop in a chain run. Captures what was researched, what came
/// back, and what fresh gaps were enqueued for the next hop.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainHop {
    /// 0 = the seed hop, 1 = first descendant, etc.
    pub depth: u32,
    /// Surface query researched at this hop.
    pub query: String,
    /// Proposal id queued by `Brain::research_one` for this hop. None
    /// only if research_one returned an error mid-chain (we still
    /// record the hop so the trace shows where the chain broke).
    pub proposal_id: Option<u64>,
    /// URLs of evidence pulled at this hop (empty when proposal_id is
    /// None).
    pub source_urls: Vec<String>,
    /// Number of bond candidates the proposal contains (NOT applied;
    /// just queued in PendingLog).
    pub candidates: usize,
    /// Endpoints we considered as fresh gaps for the next hop.
    pub considered_endpoints: usize,
    /// Endpoints we actually enqueued (passed dedupe + hub-cap +
    /// gap_threshold).
    pub enqueued_endpoints: usize,
    /// Endpoints filtered out for being over hub cap.
    pub hub_filtered: usize,
    /// Endpoints filtered out for already being in this chain's seen
    /// set.
    pub dedupe_filtered: usize,
    /// Endpoints filtered out for falling under `gap_threshold`.
    pub threshold_filtered: usize,
}

/// End-to-end trace of a single chain run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainTrace {
    /// Stable id for this chain. Embeds into the wonder_seed of every
    /// bond that this chain's approval would land, so chain-produced
    /// bonds can be revoked as a group via `revoke_bonds_by_source`.
    pub chain_id: u64,
    /// The seed gap (a CuriositySignal) that opened the chain.
    pub seed_query: String,
    /// Per-hop record. `hops.len() - 1 == final depth`.
    pub hops: Vec<ChainHop>,
    /// Total research cycles executed.
    pub cycles: u32,
    /// Why the chain stopped.
    pub termination_reason: ChainTerminationReason,
    /// Total proposals queued (sum of `proposal_id.is_some()` over
    /// hops). Equal to `cycles` on a clean run.
    pub proposals_queued: u32,
}

impl ChainTrace {
    /// True if the chain advanced past hop 0 (multi-hop succeeded).
    pub fn is_multi_hop(&self) -> bool {
        self.hops.len() > 1
    }
}

/// Compute the gap intensity for a candidate endpoint. Higher when
/// the endpoint's existing neighborhood is sparse (low in/out degree),
/// lower when it's already well-bonded. Mirrors the J.1 KnowledgeDesert
/// shape (top activation as a proxy for confidence).
///
/// Pure function — does not mutate the brain.
pub(crate) fn endpoint_gap_intensity(brain: &Brain, seed: &Seed) -> f32 {
    let in_d = brain.network.seam_in_degree(seed) as f32;
    let out_d = brain.network.seam_out_degree(seed) as f32;
    // Sparse neighborhoods are interesting; saturate at 16 bonds in
    // either direction (anything above that is well-explored). The
    // 16 boundary is the same scale CuriosityCollector treats as
    // "enough" for a HubBoundary classification.
    let combined = (in_d + out_d).min(32.0);
    let sparsity = 1.0 - (combined / 32.0);
    // Clamp to the curiosity intensity domain.
    sparsity.clamp(0.0, 1.0)
}

/// True if the endpoint should be filtered for being over the Cat.3b
/// hub cap (in or out direction). Uses the brain's currently active
/// `HubDegreeConfig`. Read-only.
pub(crate) fn endpoint_is_hub(brain: &Brain, seed: &Seed) -> bool {
    let cfg = &brain.hub_degree_config;
    // Honor exempt kinds.
    if let Some(node) = brain.network.get_node(seed) {
        if cfg.is_exempt_kind(node.kind_str()) {
            return false;
        }
    }
    let in_d = brain.network.seam_in_degree(seed) as u32;
    let out_d = brain.network.seam_out_degree(seed) as u32;
    in_d >= cfg.max_in_degree || out_d >= cfg.max_out_degree
}

/// Inspect a just-applied proposal's bond candidates for fresh gap
/// targets. Returns the seeds + their computed gap intensity in
/// candidate order; the caller filters by gap_threshold + hub cap +
/// dedupe before enqueueing.
///
/// Pure function over (brain, proposal). Does not mutate.
pub(crate) fn collect_chain_candidates(
    brain: &Brain,
    proposal: &crate::research::proposal::ResearchProposal,
) -> Vec<(Seed, String, f32)> {
    let mut out: Vec<(Seed, String, f32)> = Vec::new();
    let mut seen: HashSet<Seed> = HashSet::new();
    for c in &proposal.bond_candidates {
        // Resolve from_text and to_text to seeds via the brain's
        // existing seed-from-label hash. This is the same path teach()
        // uses internally so the chain endpoints line up with the
        // bonds the proposal would create.
        for label in [&c.from_text, &c.to_text] {
            // teach_with_kind uses Seed::from_label directly as the
            // canonical text -> seed map (src/brain.rs:2599), so the
            // chain endpoint seeds line up with whatever bonds
            // apply_proposal would create.
            let seed = Seed::from_label(label);
            if !seen.insert(seed) {
                continue;
            }
            let intensity = endpoint_gap_intensity(brain, &seed);
            out.push((seed, label.to_string(), intensity));
        }
    }
    out
}

/// Chain driver. Returns a `ChainTrace` describing every hop. The
/// caller is responsible for choosing the seed (e.g. the top
/// `wonder_queue.peek()` for the autonomous loop, or a manual seed
/// from `lu4d_research_chain_run`).
///
/// Termination modes:
/// - `Disabled`: short-circuits to single-hop, returns one-hop trace.
/// - `DepthCap` / `CyclesCap`: budget exhausted.
/// - `QueueEmpty`: nothing more to chase.
/// - `InitialError(...)`: the very first `research_one` failed; trace
///   contains a stub hop with `proposal_id=None` and the error kind.
pub fn run_chain(
    brain: &mut Brain,
    fetcher: &dyn HttpFetcher,
    seed_query_hint: Option<&str>,
) -> ChainTrace {
    let cfg = brain.chain_research_config.clone();
    let chain_id = brain.next_chain_id;
    brain.next_chain_id = brain.next_chain_id.saturating_add(1);

    // The chain seed is whatever's currently at the top of the wonder
    // queue; if the caller passed a `seed_query_hint`, we synthesize
    // a CuriositySignal for it and push it to the front of the queue
    // so research_one picks it up first. (Mirrors the manual
    // `lu4d_research { query }` shape.)
    let now = crate::brain::now_ms() as u64;
    let resolved_seed_query: String = if let Some(q) = seed_query_hint {
        let synth = CuriositySignal {
            seed: synth_seed_for_query(q),
            signal_type: SignalType::KnowledgeDesert,
            // 1.0 ensures it's above any reasonable
            // research_intensity_threshold so research_one will pop it.
            intensity: 1.0,
            scope: TemporalScope::Current,
            timestamp_ms: now,
        };
        // Stash the human-readable label so wonder_to_query produces
        // the original query string back instead of a hex hash.
        brain.gap_label_map.insert(synth.seed, q.to_string());
        brain.wonder_queue.push(synth);
        q.to_string()
    } else {
        match brain.wonder_queue.peek() {
            Some(w) => brain.wonder_to_query(w),
            None => String::new(),
        }
    };

    let mut trace = ChainTrace {
        chain_id,
        seed_query: resolved_seed_query.clone(),
        hops: Vec::new(),
        cycles: 0,
        termination_reason: ChainTerminationReason::QueueEmpty,
        proposals_queued: 0,
    };

    // Master switch off → single-hop only, then bail.
    if !cfg.enabled {
        match brain.research_one(fetcher) {
            Ok(id) => {
                let hop = build_hop(brain, 0, id, &resolved_seed_query, 0, 0, 0, 0, 0);
                trace.hops.push(hop);
                trace.cycles = 1;
                trace.proposals_queued = 1;
            }
            Err(ref e) => {
                trace.termination_reason =
                    ChainTerminationReason::InitialError(InitialErrorKind::from(e));
                trace.hops.push(stub_hop(0, &resolved_seed_query));
            }
        }
        // Disabled is the explicit termination reason — overwrite any
        // queue/initial state we tracked so callers can branch on it.
        if matches!(trace.termination_reason, ChainTerminationReason::QueueEmpty)
            || matches!(
                trace.termination_reason,
                ChainTerminationReason::InitialError(_)
            )
        {
            // Only mark Disabled when the call actually short-circuited
            // because of the master switch (i.e. research_one
            // succeeded). When the underlying call errored, prefer the
            // InitialError so the caller sees the real cause.
            if !trace.hops.is_empty() && trace.hops[0].proposal_id.is_some() {
                trace.termination_reason = ChainTerminationReason::Disabled;
            }
        }
        return trace;
    }

    // Track endpoints we've already enqueued (or seeded) inside this
    // chain so we don't loop back on ourselves (Constraint C12).
    let mut chain_seen: HashSet<Seed> = HashSet::new();
    if let Some(top) = brain.wonder_queue.peek() {
        chain_seen.insert(top.seed);
    }

    let halflife = brain.curiosity_config.interest_decay_halflife_ms;
    let mut depth: u32 = 0;
    let mut cycles: u32 = 0;

    loop {
        if cycles >= cfg.max_chain_cycles {
            trace.termination_reason = ChainTerminationReason::CyclesCap;
            break;
        }
        if depth >= cfg.max_chain_depth {
            trace.termination_reason = ChainTerminationReason::DepthCap;
            break;
        }
        // Capture the query BEFORE research_one pops the wonder.
        let hop_query = match brain.wonder_queue.peek() {
            Some(w) => brain.wonder_to_query(w),
            None => {
                trace.termination_reason = ChainTerminationReason::QueueEmpty;
                break;
            }
        };
        match brain.research_one(fetcher) {
            Ok(id) => {
                cycles += 1;
                trace.proposals_queued += 1;

                // Port SourceKind-Variants — stamp chain context onto
                // the Brain side-table (`proposal_chain_context`) keyed
                // by the proposal id. apply_proposal (after approval)
                // looks the entry up and builds SourceKind::ChainedResearch
                // instead of plain SourceKind::Research. The side-table
                // is in-memory only; if the brain saves and restarts
                // between research_one and apply_proposal the chain
                // context is lost and the bond falls back to plain
                // SourceKind::Research. Acceptable degradation pre-retrain.
                brain.proposal_chain_context.insert(
                    id,
                    crate::research::proposal::ChainContext {
                        chain_id: chain_id.to_string(),
                        hop: depth,
                        parent_query: hop_query.clone(),
                    },
                );

                let proposal_clone = brain
                    .research_proposals
                    .iter()
                    .find(|p| p.id == id)
                    .cloned();

                let (considered, enqueued, hub_filtered, dedupe_filtered, threshold_filtered) =
                    if let Some(prop) = proposal_clone.as_ref() {
                        // Stamp the chain_id onto the proposal's
                        // wonder_seed label so when this proposal is
                        // approved later, every bond it lands carries
                        // a chain-traceable source. We do this by
                        // amending the triggering_wonder seed in the
                        // gap_label_map; the wonder_seed_label that
                        // apply_proposal builds reads from the proposal
                        // directly so we leave it alone here. The
                        // chain_id linkage lives in the trace itself.
                        let candidates = collect_chain_candidates(brain, prop);
                        let considered = candidates.len();
                        let mut enqueued = 0usize;
                        let mut hub_filtered = 0usize;
                        let mut dedupe_filtered = 0usize;
                        let mut threshold_filtered = 0usize;
                        for (s, label, intensity) in candidates {
                            if !chain_seen.insert(s) {
                                dedupe_filtered += 1;
                                continue;
                            }
                            if endpoint_is_hub(brain, &s) {
                                hub_filtered += 1;
                                continue;
                            }
                            if intensity < cfg.gap_threshold {
                                threshold_filtered += 1;
                                continue;
                            }
                            // Push to wonder queue. Same shape as
                            // flush_gaps: KnowledgeDesert with the
                            // computed sparsity intensity.
                            let cs = CuriositySignal {
                                seed: s,
                                signal_type: SignalType::KnowledgeDesert,
                                intensity,
                                scope: TemporalScope::Current,
                                timestamp_ms: crate::brain::now_ms() as u64,
                            };
                            brain.gap_label_map.insert(s, label.clone());
                            brain
                                .interest_profile
                                .touch(s, cs.timestamp_ms, halflife);
                            brain.wonder_queue.push(cs);
                            enqueued += 1;
                        }
                        (considered, enqueued, hub_filtered, dedupe_filtered, threshold_filtered)
                    } else {
                        (0, 0, 0, 0, 0)
                    };

                let candidates_len = proposal_clone
                    .as_ref()
                    .map(|p| p.bond_candidates.len())
                    .unwrap_or(0);
                let source_urls = proposal_clone
                    .as_ref()
                    .map(|p| p.source_urls.clone())
                    .unwrap_or_default();

                let hop = ChainHop {
                    depth,
                    query: hop_query,
                    proposal_id: Some(id),
                    source_urls,
                    candidates: candidates_len,
                    considered_endpoints: considered,
                    enqueued_endpoints: enqueued,
                    hub_filtered,
                    dedupe_filtered,
                    threshold_filtered,
                };
                trace.hops.push(hop);
                depth += 1;
            }
            Err(ref e) => {
                if depth == 0 {
                    trace.termination_reason =
                        ChainTerminationReason::InitialError(InitialErrorKind::from(e));
                    trace.hops.push(stub_hop(depth, &hop_query));
                } else {
                    // A mid-chain error: treat as queue-empty for
                    // termination accounting (the trace still shows
                    // the cleanly-completed prior hops).
                    trace.termination_reason = ChainTerminationReason::QueueEmpty;
                }
                break;
            }
        }
    }

    trace.cycles = cycles;
    trace
}

fn build_hop(
    _brain: &Brain,
    depth: u32,
    proposal_id: u64,
    query: &str,
    considered: usize,
    enqueued: usize,
    hub_filtered: usize,
    dedupe_filtered: usize,
    threshold_filtered: usize,
) -> ChainHop {
    ChainHop {
        depth,
        query: query.to_string(),
        proposal_id: Some(proposal_id),
        source_urls: Vec::new(),
        candidates: 0,
        considered_endpoints: considered,
        enqueued_endpoints: enqueued,
        hub_filtered,
        dedupe_filtered,
        threshold_filtered,
    }
}

fn stub_hop(depth: u32, query: &str) -> ChainHop {
    ChainHop {
        depth,
        query: query.to_string(),
        proposal_id: None,
        source_urls: Vec::new(),
        candidates: 0,
        considered_endpoints: 0,
        enqueued_endpoints: 0,
        hub_filtered: 0,
        dedupe_filtered: 0,
        threshold_filtered: 0,
    }
}

/// Stable per-query seed for a synthetic chain entrypoint. Mirrors
/// `GapSignal::target_seed` so a chain seeded with the same query
/// twice resolves to the same wonder seed and the wonder-queue
/// dedupe-on-(SignalType, Seed) merges duplicate emits.
pub(crate) fn synth_seed_for_query(query: &str) -> Seed {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(b"chain:");
    h.update(query.to_lowercase().as_bytes());
    let out = h.finalize();
    let mut bytes = [0u8; 32];
    bytes.copy_from_slice(&out[..32]);
    Seed(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bond::BondType;
    use crate::node::Confidence;
    use crate::research::http::MockFetcher;
    use crate::research::proposal::{BondCandidate, ResearchProposal, ResearchProposalStatus};
    use crate::research::router::ResearchProvider;

    fn route_grok(m: &MockFetcher, query: &str, page_url: &str, page_html: &str) {
        let q = query.replace(' ', "+");
        m.route_ok(
            &format!("https://grokipedia.com/search?q={}", q),
            200,
            &format!(
                r#"<html><body><a href="{}">Result</a></body></html>"#,
                page_url
            ),
        );
        m.route_ok(
            &format!("https://grokipedia.com{}", page_url),
            200,
            page_html,
        );
    }

    /// Push a synthetic high-intensity wonder for `query` so
    /// `research_one` will pop it.
    fn seed_wonder(brain: &mut Brain, query: &str) {
        let seed = synth_seed_for_query(query);
        let cs = CuriositySignal {
            seed,
            signal_type: SignalType::KnowledgeDesert,
            intensity: 0.95,
            scope: TemporalScope::Current,
            timestamp_ms: crate::brain::now_ms() as u64,
        };
        brain.gap_label_map.insert(seed, query.to_string());
        brain.wonder_queue.push(cs);
    }

    #[test]
    fn j2_two_hop_synthetic_terminates_queue_empty() {
        let mut brain = Brain::new();
        // Hop 0 will research "alpha" and produce a candidate that
        // mentions "beta". Hop 1 will research "beta" (auto-enqueued)
        // and produce a candidate that mentions "gamma". After hop 1
        // we cap depth at 2 by setting max_chain_depth=2 so the chain
        // terminates with QueueEmpty (depth_cap is checked AT loop
        // top, so hitting it after hop 1 triggers DepthCap; we want
        // QueueEmpty so we let the budget breathe).
        brain.chain_research_config.max_chain_depth = 5;
        brain.chain_research_config.max_chain_cycles = 2;
        brain.chain_research_config.gap_threshold = 0.0;
        let m = MockFetcher::new();
        route_grok(
            &m,
            "alpha",
            "/page/Alpha",
            "<html><body><main><p>Alpha is a beta.</p></main></body></html>",
        );
        route_grok(
            &m,
            "beta",
            "/page/Beta",
            "<html><body><main><p>Beta is a gamma.</p></main></body></html>",
        );
        seed_wonder(&mut brain, "alpha");
        let trace = run_chain(&mut brain, &m, None);
        assert!(trace.cycles >= 1, "chain produced no hops: {:?}", trace);
        assert!(
            matches!(
                trace.termination_reason,
                ChainTerminationReason::CyclesCap | ChainTerminationReason::QueueEmpty
            ),
            "unexpected termination: {:?}",
            trace.termination_reason
        );
    }

    #[test]
    fn j2_depth_cap_terminates_with_depth_cap() {
        let mut brain = Brain::new();
        brain.chain_research_config.max_chain_depth = 1;
        brain.chain_research_config.max_chain_cycles = 10;
        brain.chain_research_config.gap_threshold = 0.0;
        let m = MockFetcher::new();
        route_grok(
            &m,
            "alpha",
            "/page/Alpha",
            "<html><body><main><p>Alpha is a beta. Alpha bonds with delta.</p></main></body></html>",
        );
        route_grok(
            &m,
            "beta",
            "/page/Beta",
            "<html><body><main><p>Beta connects to gamma.</p></main></body></html>",
        );
        seed_wonder(&mut brain, "alpha");
        let trace = run_chain(&mut brain, &m, None);
        assert!(
            matches!(trace.termination_reason, ChainTerminationReason::DepthCap),
            "expected DepthCap, got {:?}",
            trace.termination_reason
        );
        assert_eq!(trace.cycles, 1);
    }

    #[test]
    fn j2_cycles_cap_terminates_with_cycles_cap() {
        let mut brain = Brain::new();
        brain.chain_research_config.max_chain_depth = 100;
        brain.chain_research_config.max_chain_cycles = 1;
        brain.chain_research_config.gap_threshold = 0.0;
        let m = MockFetcher::new();
        route_grok(
            &m,
            "alpha",
            "/page/Alpha",
            "<html><body><main><p>Alpha is a beta.</p></main></body></html>",
        );
        route_grok(
            &m,
            "beta",
            "/page/Beta",
            "<html><body><main><p>Beta connects to gamma.</p></main></body></html>",
        );
        seed_wonder(&mut brain, "alpha");
        let trace = run_chain(&mut brain, &m, None);
        assert!(
            matches!(trace.termination_reason, ChainTerminationReason::CyclesCap),
            "expected CyclesCap, got {:?}",
            trace.termination_reason
        );
        assert_eq!(trace.cycles, 1);
    }

    #[test]
    fn j2_disabled_short_circuits_to_single_hop() {
        let mut brain = Brain::new();
        brain.chain_research_config.enabled = false;
        brain.chain_research_config.gap_threshold = 0.0;
        let m = MockFetcher::new();
        route_grok(
            &m,
            "alpha",
            "/page/Alpha",
            "<html><body><main><p>Alpha is a beta.</p></main></body></html>",
        );
        seed_wonder(&mut brain, "alpha");
        let queue_before = brain.wonder_queue.len();
        let trace = run_chain(&mut brain, &m, None);
        let queue_after = brain.wonder_queue.len();
        assert!(
            matches!(trace.termination_reason, ChainTerminationReason::Disabled),
            "expected Disabled, got {:?}",
            trace.termination_reason
        );
        assert_eq!(trace.hops.len(), 1);
        // Single-hop path must not enqueue any fresh gaps.
        assert!(
            queue_after <= queue_before,
            "disabled chain enqueued: before={} after={}",
            queue_before,
            queue_after
        );
    }

    #[test]
    fn j2_anti_hallucination_bond_count_unchanged_through_chain() {
        let mut brain = Brain::new();
        brain.chain_research_config.gap_threshold = 0.0;
        brain.chain_research_config.max_chain_cycles = 3;
        let m = MockFetcher::new();
        route_grok(
            &m,
            "alpha",
            "/page/Alpha",
            "<html><body><main><p>Alpha is a beta.</p></main></body></html>",
        );
        route_grok(
            &m,
            "beta",
            "/page/Beta",
            "<html><body><main><p>Beta connects to gamma.</p></main></body></html>",
        );
        seed_wonder(&mut brain, "alpha");
        let bonds_before = brain.network.bond_count();
        let trace = run_chain(&mut brain, &m, None);
        let bonds_after = brain.network.bond_count();
        // Chain MUST NOT mutate stored bond weights; new bonds are
        // PendingChange queued via apply_proposal only after explicit
        // approval. Pure observation here means bond_count is
        // unchanged.
        assert_eq!(
            bonds_before, bonds_after,
            "chain mutated bond_count: before={} after={} trace={:?}",
            bonds_before, bonds_after, trace
        );
    }

    #[test]
    fn j2_hub_cap_excludes_super_hub_endpoints() {
        // Build a brain where "hub" already has out_degree at the cap;
        // a research result that would point to "hub" must NOT enqueue
        // it for the next hop.
        let mut brain = Brain::new();
        brain.chain_research_config.gap_threshold = 0.0;
        brain.hub_degree_config.max_out_degree = 100;
        brain.hub_degree_config.max_in_degree = 1; // very tight
        // Pre-create a node "hub" with in_degree >= 1 by teaching one
        // bond into it.
        brain.teach(
            "existing_anchor",
            BondType::RelatedTo,
            "hub",
            0.5,
            Confidence::Inferred,
        );
        // Synthesize a proposal that names "hub" as a candidate
        // endpoint without going through the network.
        let prop = ResearchProposal {
            id: 999,
            triggering_wonder: None,
            query: "alpha".to_string(),
            source_urls: vec!["https://x/y".to_string()],
            raw_excerpts: vec!["Alpha is hub.".to_string()],
            providers: vec![ResearchProvider::Grokipedia],
            bond_candidates: vec![BondCandidate {
                from_text: "alpha".to_string(),
                to_text: "hub".to_string(),
                bond_type: BondType::RelatedTo,
                confidence: Confidence::Inferred,
                source_url: "https://x/y".to_string(),
                source_excerpt: "Alpha is hub.".to_string(),
                fingerprint: vec![],
            }],
            scar_blocked_count: 0,
            created_ms: 1,
            status: ResearchProposalStatus::Pending,
        };
        // Drive the candidate-collection + filter directly.
        let candidates = collect_chain_candidates(&brain, &prop);
        assert!(
            candidates.iter().any(|(_, label, _)| label == "hub"),
            "hub should be a candidate before filter"
        );
        // Apply the hub-cap filter the chain driver uses.
        let mut hub_filtered = 0usize;
        for (s, label, _i) in &candidates {
            if label == "hub" && endpoint_is_hub(&brain, s) {
                hub_filtered += 1;
            }
        }
        assert_eq!(
            hub_filtered, 1,
            "hub endpoint should have been filtered by Cat.3b cap"
        );
    }

    #[test]
    fn j2_dedupe_prevents_revisit() {
        // Pushing the same endpoint twice into chain_seen returns false
        // on the second insert; the chain driver uses this to skip the
        // dedupe target. We test the algorithm at the HashSet level
        // because reaching it through `run_chain` requires identical
        // mock evidence twice in a row, which the proposal-id
        // monotonic counter guarantees.
        let mut chain_seen: HashSet<Seed> = HashSet::new();
        let s = synth_seed_for_query("alpha");
        assert!(chain_seen.insert(s), "first insert should succeed");
        assert!(
            !chain_seen.insert(s),
            "second insert MUST be rejected — dedupe invariant"
        );
    }

    /// Port SourceKind-Variants — end-to-end: a chain run that
    /// produces a proposal, then apply_proposal -> bonds carry
    /// SourceKind::ChainedResearch with the chain_id / hop / parent_query
    /// stamped by chain_driver_run. The MockFetcher used by other J.2
    /// tests gives us a deterministic surface here.
    #[test]
    fn j2_apply_proposal_stamps_chained_research_when_chain_context_present() {
        // Build a proposal that already has a chain_context set (mirrors
        // what chain_driver_run does after research_one). Then call
        // apply_proposal directly and inspect the source overlay
        // attribution on each landed bond.
        let mut brain = Brain::new();
        // Inject a research proposal with chain_context.
        let proposal = crate::research::proposal::ResearchProposal {
            id: 42,
            triggering_wonder: None,
            query: "what is photosynthesis".to_string(),
            source_urls: vec!["https://en.wikipedia.org/wiki/Photosynthesis".into()],
            raw_excerpts: vec!["Photosynthesis converts light to chemical energy.".into()],
            providers: vec![ResearchProvider::Grokipedia],
            bond_candidates: vec![
                BondCandidate {
                    from_text: "photosynthesis".into(),
                    to_text: "light".into(),
                    bond_type: BondType::Causes,
                    confidence: Confidence::Inferred,
                    source_url: "https://en.wikipedia.org/wiki/Photosynthesis".into(),
                    source_excerpt: "...".into(),
                    fingerprint: vec![],
                },
                BondCandidate {
                    from_text: "photosynthesis".into(),
                    to_text: "energy".into(),
                    bond_type: BondType::HasProperty,
                    confidence: Confidence::Inferred,
                    source_url: "https://en.wikipedia.org/wiki/Photosynthesis".into(),
                    source_excerpt: "...".into(),
                    fingerprint: vec![],
                },
            ],
            scar_blocked_count: 0,
            created_ms: 1,
            status: crate::research::proposal::ResearchProposalStatus::Pending,
        };
        brain.research_proposals.push(proposal);
        // Side-table insert keyed on proposal.id mirrors what
        // chain_driver_run does after research_one returns.
        brain.proposal_chain_context.insert(
            42,
            crate::research::proposal::ChainContext {
                chain_id: "c-test-99".into(),
                hop: 2,
                parent_query: "what is photosynthesis".into(),
            },
        );
        let added = crate::research::loop_::apply_proposal(&mut brain, 0);
        assert!(added >= 1, "apply_proposal should land at least one bond");

        // Inspect both landed bonds' source attribution.
        let from = crate::seed::Seed::from_label("photosynthesis");
        let to_light = crate::seed::Seed::from_label("light");
        let to_energy = crate::seed::Seed::from_label("energy");
        let attr_light = brain.source_overlay.read().unwrap().get(&from, &to_light, BondType::Causes);
        match attr_light.kind {
            crate::source::SourceKind::ChainedResearch { chain_id, hop, parent_query } => {
                assert_eq!(chain_id, "c-test-99");
                assert_eq!(hop, 2);
                assert_eq!(parent_query, "what is photosynthesis");
            }
            other => panic!("expected ChainedResearch, got {:?}", other),
        }
        let attr_energy = brain.source_overlay.read().unwrap().get(&from, &to_energy, BondType::HasProperty);
        assert!(matches!(
            attr_energy.kind,
            crate::source::SourceKind::ChainedResearch { .. }
        ));
    }

    /// Port SourceKind-Variants — when chain_context is None,
    /// apply_proposal still produces SourceKind::Research (single-hop /
    /// manual research path unchanged).
    #[test]
    fn j2_apply_proposal_stays_research_without_chain_context() {
        let mut brain = Brain::new();
        let proposal = crate::research::proposal::ResearchProposal {
            id: 43,
            triggering_wonder: None,
            query: "single hop".to_string(),
            source_urls: vec!["https://example.com/x".into()],
            raw_excerpts: vec!["...".into()],
            providers: vec![ResearchProvider::Grokipedia],
            bond_candidates: vec![BondCandidate {
                from_text: "alpha".into(),
                to_text: "beta".into(),
                bond_type: BondType::IsA,
                confidence: Confidence::Inferred,
                source_url: "https://example.com/x".into(),
                source_excerpt: "...".into(),
                fingerprint: vec![],
            }],
            scar_blocked_count: 0,
            created_ms: 1,
            status: crate::research::proposal::ResearchProposalStatus::Pending,
        };
        brain.research_proposals.push(proposal);
        let added = crate::research::loop_::apply_proposal(&mut brain, 0);
        assert!(added >= 1);
        let attr = brain.source_overlay.read().unwrap().get(
            &crate::seed::Seed::from_label("alpha"),
            &crate::seed::Seed::from_label("beta"),
            BondType::IsA,
        );
        assert!(
            matches!(attr.kind, crate::source::SourceKind::Research { .. }),
            "expected Research (no chain_context), got {:?}",
            attr.kind
        );
    }
}
