//! Port T.2 — Idle work loop primitive.
//!
//! [`idle_tick`] does ONE unit of idle work: pops the highest-intensity
//! wonder above threshold, runs the existing research path, returns
//! the outcome. Caller-driven for now (`lu4d_idle_tick` MCP tool); the
//! T.6 daemon will auto-drive it later.
//!
//! IMPORTANT — auto-commit-research caveat (T.2 known gap, closed by
//! T.3): for T.2 the popped wonder runs through the existing
//! `Brain::research_one` path which queues a `Pending` proposal. T.3
//! introduces the pending-changes-with-approval gate so idle research
//! mutations require explicit approval. T.2 deliberately does NOT
//! design that twice — it just makes the existing manual research path
//! drive automatically when idle.

use crate::brain::Brain;
use crate::idle::state::{IdleTickOutcome, IdleTickResult};
use crate::research::http::HttpFetcher;
use crate::research::loop_::ResearchError;

/// Single tick of the idle work loop.
///
/// Gate sequence (each gate, on miss, returns immediately):
///   1. Brain must be idle (`active_queries == 0`).
///   2. `min_idle_ms` must be satisfied — `last_query_age_ms() >=
///      min_idle_ms`. With `min_idle_ms == 0` this gate is a no-op.
///   3. The wonder queue must yield a signal above
///      `research_intensity_threshold` (delegated to
///      `Brain::research_one`).
///
/// On success, returns `ResearchCompleted` (proposal queued as
/// `Pending`). On a router/extractor failure for a popped wonder,
/// returns `ResearchFailed` — the seed is still surfaced via
/// `wonder_seed_processed` so callers can correlate.
///
/// This function does NOT register the tick as a query in
/// `IdleState.active_queries`. The MCP wrapper (`tool_idle_tick`)
/// owns its own [`crate::idle::QueryGuard`] so the gate-1 check sees
/// the brain's actual pre-tick state. Test code that calls this
/// function directly likewise does not need a guard.
pub fn idle_tick(
    brain: &mut Brain,
    fetcher: &dyn HttpFetcher,
    min_idle_ms: u64,
) -> IdleTickResult {
    // ---- Gate 1: brain must be idle. ----
    let active = brain.idle_state.active_queries();
    let idle_age = brain.idle_state.last_query_age_ms().unwrap_or(0);
    if active != 0 {
        brain.idle_state.record_tick(false);
        return IdleTickResult {
            did_work: false,
            wonder_seed_processed: None,
            outcome: IdleTickOutcome::NoWorkNotIdle,
            idle_duration_ms: 0,
            total_idle_ticks: brain.idle_state.total_idle_ticks(),
            pending_added_count: 0,
        };
    }

    // ---- Gate 2: min_idle_ms. ----
    if min_idle_ms > 0 && idle_age < min_idle_ms {
        brain.idle_state.record_tick(false);
        return IdleTickResult {
            did_work: false,
            wonder_seed_processed: None,
            outcome: IdleTickOutcome::NoWorkNotIdle,
            idle_duration_ms: idle_age,
            total_idle_ticks: brain.idle_state.total_idle_ticks(),
            pending_added_count: 0,
        };
    }

    // ---- Gate 3: pop a wonder. We peek first so the seed prefix is
    //              available even if `research_one` later errors. ----
    let pop_preview = brain.wonder_queue.peek().cloned();
    let seed_prefix = pop_preview.as_ref().map(|w| hex_seed_prefix_8(&w.seed));

    // ---- Internal-first gap-fill (Port "internal-first" 2026-05-30) ----
    // Before going to the web, try focused triangle closure around the
    // wonder's seed. If existing graph knowledge can answer the gap,
    // push those candidate bonds as PendingChange::AddBond entries,
    // pop the wonder, and return InternalFillAdded — skipping the
    // 3.2s web round-trip entirely. Falls through to research_one
    // unchanged if internal yields nothing.
    if let Some(wonder) = &pop_preview {
        let mut new_ids: Vec<u64> = Vec::new();
        let cons_config = crate::consolidation::ConsolidationConfig::default();
        // Dispatch on the wonder's signal_type. HubBoundary and DeadEnd
        // both target hubs, so the port-1 transitive-closure path returns
        // empty (hub-A gate). For those we use the shared-hub co-occurrence
        // fill (port hub-fill 2026-05-30). For IsolatedNode and
        // KnowledgeDesert the seed is non-hub so port-1 applies cleanly.
        let proposals = match wonder.signal_type {
            crate::curiosity::SignalType::HubBoundary
            | crate::curiosity::SignalType::DeadEnd => {
                crate::consolidation::internal_fill_at_hub(
                    &brain.network,
                    &wonder.seed,
                    &cons_config,
                    crate::consolidation::INTERNAL_FILL_MAX_NEIGHBORS,
                    crate::consolidation::HUB_LEG_MIN_WEIGHT,
                )
            }
            crate::curiosity::SignalType::IsolatedNode
            | crate::curiosity::SignalType::KnowledgeDesert => {
                crate::consolidation::internal_fill_at_seed(
                    &brain.network,
                    &wonder.seed,
                    &cons_config,
                    crate::consolidation::INTERNAL_FILL_MAX_NEIGHBORS,
                )
            }
        };
        if !proposals.is_empty() {
            let now = crate::brain::now_ms() as u64;
            let wonder_seed_hex = seed_prefix.clone().unwrap_or_else(|| "manual".to_string());
            let mut pending_added: u32 = 0;
            for p in &proposals {
                // Resolve subject/object node labels for the AddBond shape.
                // OP-RECALL-1 (R6). Was `brain.network.get_node(..)`,
                // which reads the RESIDENT node map -- empty by design on
                // a cache-backed v5 brain. Every proposal therefore hit
                // the `None => continue` arm and internal fill emitted
                // nothing at all on the live brain, silently. The
                // OP-LABEL-1 resolver walks resident -> memo -> node
                // store and is the same source narration uses.
                //
                // The `continue` on a missing label is deliberate and
                // unchanged: a bond proposal needs real subject and
                // object text, and inventing a placeholder for an
                // unlabelled node would be a fabricated edge.
                let subj = match brain.network.node_label(&p.a) {
                    Some(n) => n,
                    None => continue,
                };
                let obj = match brain.network.node_label(&p.c) {
                    Some(n) => n,
                    None => continue,
                };
                let path_label = format!(
                    "synthesis(triangle:{}→{}→{})",
                    hex_seed_prefix_8(&p.path[0]),
                    hex_seed_prefix_8(&p.path[1]),
                    hex_seed_prefix_8(&p.path[2]),
                );
                let attr = crate::source::SourceAttribution::teach(
                    format!("synthesis:{}", wonder_seed_hex),
                    now,
                );
                let change = crate::pending::PendingChange {
                    pending_id: 0,
                    proposed_at_ms: now,
                    proposed_by: attr,
                    change_kind: crate::pending::PendingChangeKind::AddBond {
                        subject_text: subj,
                        object_text: obj,
                        bond_type: crate::bond::BondType::RelatedTo,
                        weight: p.new_weight,
                        negated: false,
                    },
                    confidence: 0.5,  // path-decayed inference, moderate confidence
                    reason: format!(
                        "Internal synthesis from gap signal `{}` via {} (w={:.2})",
                        &wonder_seed_hex[..wonder_seed_hex.len().min(8)],
                        path_label,
                        p.new_weight,
                    ),
                    status: crate::pending::PendingStatus::Pending,
                };
                if let Ok(Some(id)) = brain.push_pending_with_filter(change) {
                    pending_added += 1;
                    new_ids.push(id);
                }
            }
            // OP-CURIOSITY-1: hand the just-created entries to the
            // approval switch. No-op unless it is explicitly enabled.
            let (_considered, _applied) = brain.auto_apply_curiosity_pending(&new_ids);
            if pending_added > 0 {
                // Pop the wonder we just filled internally so we don't
                // re-process it next tick. (research_one would have done
                // the pop in its success branch; we own that responsibility
                // when we short-circuit it.)
                let _ = brain.wonder_queue.pop();
                brain.idle_state.record_tick(true);
                return IdleTickResult {
                    did_work: true,
                    wonder_seed_processed: seed_prefix,
                    outcome: IdleTickOutcome::InternalFillAdded,
                    idle_duration_ms: idle_age,
                    total_idle_ticks: brain.idle_state.total_idle_ticks(),
                    pending_added_count: pending_added,
                };
            }
            // proposals existed but none survived the rejection-pattern
            // pre-filter — fall through to external research below.
        }
    }

    // Drive ONE iteration. Delegates wonder pop, threshold check,
    // QueueFull / Disabled handling, and proposal creation to the
    // existing autonomous-research code path.
    match brain.research_one(fetcher) {
        Ok(proposal_id) => {
            // Port T.3: a successful research_one created a Pending
            // ResearchProposal in `brain.research_proposals`. Walk
            // its bond_candidates and push each as a
            // `PendingChangeKind::AddBond` into `brain.pending`,
            // then strip the now-redundant ResearchProposal so the
            // legacy `lu4d_research_proposals_list` view stays
            // clean. Auto-commit gap closed: brain bond count is
            // unchanged until pending entries are approved.
            let mut pending_added: u32 = 0;
            let mut research_ids: Vec<u64> = Vec::new();
            if let Some(idx) = brain
                .research_proposals
                .iter()
                .position(|p| p.id == proposal_id)
            {
                let proposal = brain.research_proposals.remove(idx);
                let now = crate::brain::now_ms() as u64;
                let wonder_seed = proposal
                    .triggering_wonder
                    .as_ref()
                    .map(|w| {
                        let b = &w.seed.0[0..8];
                        format!(
                            "{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
                            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
                        )
                    })
                    .unwrap_or_else(|| "manual".to_string());
                let intensity = proposal
                    .triggering_wonder
                    .as_ref()
                    .map(|w| w.intensity)
                    .unwrap_or(0.0);
                for cand in &proposal.bond_candidates {
                    let attr = crate::source::SourceAttribution::research(
                        wonder_seed.clone(),
                        Some(cand.source_url.clone()),
                        now,
                    );
                    let confidence = cand.confidence.value();
                    let change = crate::pending::PendingChange {
                        pending_id: 0, // allocated by push()
                        proposed_at_ms: now,
                        proposed_by: attr,
                        change_kind: crate::pending::PendingChangeKind::AddBond {
                            subject_text: cand.from_text.clone(),
                            object_text: cand.to_text.clone(),
                            bond_type: cand.bond_type,
                            weight: 0.7,
                            negated: false,
                        },
                        confidence,
                        reason: format!(
                            "Research-derived from gap signal `{}` (intensity {:.2})",
                            &wonder_seed[..wonder_seed.len().min(8)],
                            intensity,
                        ),
                        status: crate::pending::PendingStatus::Pending,
                    };
                    // Port T.8: route through the rejection-pattern
                    // pre-filter so learned bad-shape proposals are dropped.
                    if let Ok(Some(id)) = brain.push_pending_with_filter(change) {
                        pending_added += 1;
                        research_ids.push(id);
                    }
                }
            }
            // OP-CURIOSITY-1: see `auto_apply_curiosity_pending`.
            let (_considered, _applied) = brain.auto_apply_curiosity_pending(&research_ids);
            brain.idle_state.record_tick(true);
            IdleTickResult {
                did_work: true,
                wonder_seed_processed: seed_prefix,
                outcome: IdleTickOutcome::PendingAdded,
                idle_duration_ms: idle_age,
                total_idle_ticks: brain.idle_state.total_idle_ticks(),
                pending_added_count: pending_added,
            }
        }
        Err(ResearchError::QueueEmpty)
        | Err(ResearchError::BelowThreshold)
        | Err(ResearchError::Disabled) => {
            // "Nothing idle-worthy to do" — collapsed into one outcome
            // since they all mean the same thing to a daemon driver.
            brain.idle_state.record_tick(false);
            IdleTickResult {
                did_work: false,
                wonder_seed_processed: None,
                outcome: IdleTickOutcome::NoWorkNoSignals,
                idle_duration_ms: idle_age,
                total_idle_ticks: brain.idle_state.total_idle_ticks(),
                pending_added_count: 0,
            }
        }
        Err(_) => {
            // RouterFailed / NoTextFound / QueueFull / others —
            // counts as "work completed" because we made the attempt.
            brain.idle_state.record_tick(true);
            IdleTickResult {
                did_work: true,
                wonder_seed_processed: seed_prefix,
                outcome: IdleTickOutcome::ResearchFailed,
                idle_duration_ms: idle_age,
                total_idle_ticks: brain.idle_state.total_idle_ticks(),
                pending_added_count: 0,
            }
        }
    }
}

/// Hex-encode the first 8 bytes of a [`crate::seed::Seed`]. Mirrors
/// the prefix length used by `hex_seed_prefix` in
/// `crate::research::loop_` (kept private to each module to avoid
/// pulling that helper into the public surface).
fn hex_seed_prefix_8(seed: &crate::seed::Seed) -> String {
    let b = &seed.0[0..8];
    format!(
        "{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]
    )
}
