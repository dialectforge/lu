//! Research-loop driver. Pure function over Brain + HttpFetcher so
//! tests can plug in a `MockFetcher`. The Brain wires the production
//! path with a `UreqFetcher` in `Brain::default_research_fetcher`.
//!
//! See `Brain::research_one`, `Brain::research_loop_run`,
//! `Brain::approve_research_proposal`, `Brain::reject_research_proposal`.

use crate::brain::{now_ms, Brain};
use crate::curiosity::CuriositySignal;
use crate::node::Confidence;
use crate::research::extract::extract_triples;
use crate::research::http::HttpFetcher;
use crate::research::proposal::{
    build_candidates, ResearchProposal, ResearchProposalStatus,
};
use crate::research::router::{run_router, ResearchEvidence, ResearchProvider};

/// Errors the research loop can surface to the caller. Network errors
/// fold into `RouterFailed`; logic errors (e.g. unknown proposal id)
/// have their own variants.
#[derive(Debug, Clone)]
pub enum ResearchError {
    Disabled,
    QueueEmpty,
    BelowThreshold,
    QueueFull,
    RouterFailed(String),
    UnknownProposal(u64),
    AlreadyDecided(u64),
    NoTextFound,
}

impl std::fmt::Display for ResearchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ResearchError::Disabled => write!(f, "research disabled in config"),
            ResearchError::QueueEmpty => write!(f, "wonder queue is empty"),
            ResearchError::BelowThreshold => {
                write!(f, "top wonder below research_intensity_threshold")
            }
            ResearchError::QueueFull => {
                write!(f, "max_proposals_pending reached; approve / reject some first")
            }
            ResearchError::RouterFailed(s) => write!(f, "router failed: {}", s),
            ResearchError::UnknownProposal(id) => write!(f, "unknown proposal id: {}", id),
            ResearchError::AlreadyDecided(id) => {
                write!(f, "proposal {} is not Pending", id)
            }
            ResearchError::NoTextFound => {
                write!(f, "no extractable text in router evidence")
            }
        }
    }
}

impl std::error::Error for ResearchError {}

/// Build the proposal for a single wonder. Public so the manual
/// `lu4d_research { query }` MCP path can call it without a wonder.
pub fn make_proposal_for_query(
    brain: &mut Brain,
    fetcher: &dyn HttpFetcher,
    query: &str,
    triggering_wonder: Option<CuriositySignal>,
) -> Result<ResearchProposal, ResearchError> {
    if !brain.research_config.enabled {
        return Err(ResearchError::Disabled);
    }
    if brain.pending_proposal_count() >= brain.research_config.max_proposals_pending {
        return Err(ResearchError::QueueFull);
    }
    let cfg = brain.research_config.clone();
    let evidence: Vec<ResearchEvidence> =
        run_router(fetcher, &cfg, query).map_err(|e| ResearchError::RouterFailed(e.to_string()))?;

    // Concatenate all evidence excerpts for SVO extraction; track
    // sentence -> source URL for provenance.
    let combined: String = evidence
        .iter()
        .map(|e| e.excerpt.clone())
        .collect::<Vec<_>>()
        .join(" ");
    if combined.trim().is_empty() {
        return Err(ResearchError::NoTextFound);
    }
    let triples = extract_triples(&combined, cfg.max_bond_candidates_per_proposal * 2);

    // Snapshot every active scar's fingerprint at this instant. Empty
    // fingerprints get filtered out — they can't meaningfully match.
    let now = now_ms();
    let scar_fps: Vec<Vec<f32>> = brain
        .scar_log
        .scars
        .iter()
        .filter(|s| s.created_ms <= now && s.superseded_by.is_none())
        .filter(|s| !s.fingerprint.is_empty())
        .map(|s| s.fingerprint.clone())
        .collect();

    let (candidates, blocked) = build_candidates(
        &triples,
        &evidence,
        &scar_fps,
        cfg.scar_fingerprint_match_threshold,
        Confidence::Inferred,
        cfg.max_bond_candidates_per_proposal,
        query,
    );

    let id = brain.next_research_proposal_id;
    brain.next_research_proposal_id = brain.next_research_proposal_id.saturating_add(1);

    let providers: Vec<ResearchProvider> = evidence.iter().map(|e| e.provider).collect();
    let source_urls: Vec<String> = evidence.iter().map(|e| e.source_url.clone()).collect();
    let raw_excerpts: Vec<String> = evidence
        .iter()
        .take(cfg.max_excerpts_per_proposal)
        .map(|e| e.excerpt.clone())
        .collect();

    Ok(ResearchProposal {
        id,
        triggering_wonder,
        query: query.to_string(),
        source_urls,
        raw_excerpts,
        providers,
        bond_candidates: candidates,
        scar_blocked_count: blocked,
        created_ms: now as u64,
        status: ResearchProposalStatus::Pending,
    })
}

/// Apply an approved proposal: walk every bond candidate and run
/// `Brain::teach`. teach() routes through `add_bond_with_scar_check`
/// so a scarred-seed candidate is STILL refused at ingest. Returns
/// the number of bonds successfully added.
pub fn apply_proposal(brain: &mut Brain, proposal_idx: usize) -> u32 {
    let mut added: u32 = 0;
    let proposal = brain.research_proposals[proposal_idx].clone();
    let candidates = proposal.bond_candidates;
    let bonds_before = brain.network.bond_count();
    // Port S — every bond produced by an approved research call
    // carries `SourceKind::Research` with the wonder seed that
    // triggered the call (when one exists), the source URL, and
    // the wall-clock fetch time. This is what closes the curiosity
    // loop for provenance: given a bond, you can locate the wonder
    // that produced it; given a wonder, you can revoke every bond
    // it produced via `Brain::revoke_bonds_by_source`.
    let wonder_seed_label = match &proposal.triggering_wonder {
        Some(w) => format!("wonder:{}:{}", proposal.id, hex_seed_prefix(&w.seed)),
        None => format!("manual:{}", proposal.id),
    };
    let fetched_at_ms = crate::brain::now_ms() as u64;
    // Port Cat.3a: snapshot the trust table at apply time so the
    // multiplier each candidate receives is deterministic across the
    // batch (an MCP `lu4d_trust_set` call mid-loop cannot perturb a
    // candidate later in the same loop). Mirrors the scar-fingerprint
    // snapshot pattern in make_proposal_for_query.
    let trust_snapshot: crate::trust::TrustTable = brain.trust_table.clone();
    const BASE_RESEARCH_WEIGHT: f32 = 0.7;
    for c in candidates {
        let url_opt = if c.source_url.is_empty() {
            None
        } else {
            Some(c.source_url.clone())
        };
        // Look up the trust multiplier from the candidate's source URL
        // host. Empty URL or unmatched host → default_multiplier.
        // Master switch off → 1.0 pass-through.
        let multiplier = trust_snapshot.lookup(&c.source_url);
        let final_weight = (BASE_RESEARCH_WEIGHT * multiplier)
            .clamp(0.0, 1.0);
        // Port SourceKind-Variants — when this proposal was produced by
        // a chain run (J.2), stamp every landed bond with the first-class
        // SourceKind::ChainedResearch carrying chain_id/hop/parent_query.
        // Otherwise fall back to the pre-port SourceKind::Research shape
        // (single-hop curiosity research and manual research_one calls).
        // Read from the in-memory `proposal_chain_context` side-table on
        // Brain (keyed by proposal.id) instead of a field on the proposal
        // itself — see `ChainContext` doc for the bincode rationale.
        let mut attr = match brain.proposal_chain_context.get(&proposal.id) {
            Some(ctx) => crate::source::SourceAttribution::chained_research(
                ctx.chain_id.clone(),
                ctx.hop,
                ctx.parent_query.clone(),
            ),
            None => crate::source::SourceAttribution::research(
                wonder_seed_label.clone(),
                url_opt,
                fetched_at_ms,
            ),
        };
        // Stash the actual multiplier on the attribution so future
        // chain composers / J's confidence aggregator can read it
        // back without re-doing the host lookup.
        // Cat.3a: stamp the resolved multiplier on the attribution.
        // T-Ingest.1 keeps `trust: f32` (sentinel-based, NOT Option) for
        // bincode wire-format compat with pre-T-Ingest.1 PendingLog
        // entries on disk. A non-negative `trust` tells teach_with_source
        // "already resolved, do not re-multiply" — preserves the C2
        // Grokopedia-first ordering exactly.
        attr.trust = multiplier;
        brain.teach_with_source(
            &c.from_text,
            c.bond_type,
            &c.to_text,
            final_weight,
            c.confidence,
            attr,
        );
    }
    let bonds_after = brain.network.bond_count();
    if bonds_after > bonds_before {
        added = (bonds_after - bonds_before) as u32;
    }
    added
}

/// Hex-encode the first 8 bytes of a Seed for use in a Research
/// source's `wonder_seed` label. Mirrors the prefix length used by
/// `SourceOverlay::key`.
fn hex_seed_prefix(seed: &crate::seed::Seed) -> String {
    let b = &seed.0[0..8];
    format!(
        "{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::research::http::MockFetcher;

    fn route_grok_first(m: &MockFetcher, query: &str, page_url: &str, page_html: &str) {
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

    #[test]
    fn test_make_proposal_disabled_errs() {
        let mut brain = Brain::new();
        brain.research_config.enabled = false;
        let m = MockFetcher::new();
        let err = make_proposal_for_query(&mut brain, &m, "x", None).unwrap_err();
        assert!(matches!(err, ResearchError::Disabled));
    }

    #[test]
    fn test_make_proposal_full_queue_errs() {
        let mut brain = Brain::new();
        brain.research_config.max_proposals_pending = 1;
        // Pre-populate one Pending proposal manually.
        brain.research_proposals.push(ResearchProposal {
            id: 0,
            triggering_wonder: None,
            query: "old".into(),
            source_urls: Vec::new(),
            raw_excerpts: Vec::new(),
            providers: Vec::new(),
            bond_candidates: Vec::new(),
            scar_blocked_count: 0,
            created_ms: 1,
            status: ResearchProposalStatus::Pending,
        });
        let m = MockFetcher::new();
        let err = make_proposal_for_query(&mut brain, &m, "x", None).unwrap_err();
        assert!(matches!(err, ResearchError::QueueFull));
    }

    #[test]
    fn test_make_proposal_round_trip_via_grokipedia() {
        let mut brain = Brain::new();
        let m = MockFetcher::new();
        route_grok_first(
            &m,
            "rust",
            "/page/Rust",
            "<html><body><main><p>Rust is a systems programming language.</p></main></body></html>",
        );
        let prop = make_proposal_for_query(&mut brain, &m, "rust", None).unwrap();
        assert_eq!(prop.query, "rust");
        assert!(matches!(prop.status, ResearchProposalStatus::Pending));
        assert!(!prop.bond_candidates.is_empty());
        assert!(!prop.source_urls.is_empty());
        assert_eq!(prop.providers[0], ResearchProvider::Grokipedia);
    }
}
