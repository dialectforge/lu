//! Port #10 — Autonomous Research Loop.
//!
//! Lu's Wonder Queue accumulates curiosity signals (DeadEnd,
//! IsolatedNode, KnowledgeDesert, HubBoundary). The autonomous
//! research loop closes the gap between "I notice I don't know X" and
//! "here are some candidate facts about X for John to approve".
//!
//! Pipeline (one iteration):
//!   1. Pop the highest-intensity wonder above
//!      `ResearchConfig::research_intensity_threshold`.
//!   2. Construct a search query from the wonder's anchor seed.
//!   3. Route through Grokipedia first; fall back to DDG on miss.
//!   4. Extract clean prose → sentence-level SVO triples.
//!   5. Drop any candidate whose fingerprint matches an active scar.
//!   6. Wrap the survivors in a Pending `ResearchProposal`.
//!
//! Lu can SUGGEST what to learn but cannot DECIDE to learn it.
//! Approval is mandatory and routes back through the scar-aware
//! `Brain::teach` path so a scarred concept STILL refuses the bond at
//! ingest time even after John approved.
//!
//! "NO STUBS ANYWHERE": every component is a real implementation.
//! HTTP hits the real network (UreqFetcher); Grokipedia + DDG return
//! real results from real endpoints; HTML extraction runs on real
//! markup via `scraper`; the scar-fingerprint check uses the same
//! cosine math the brain itself stores. Tests mock at the
//! `HttpFetcher` boundary to keep the suite hermetic, but live
//! `#[ignore]`d tests prove the stack end-to-end.

pub mod chain;
pub mod duckduckgo;
pub mod extract;
pub mod grokipedia;
pub mod http;
pub mod loop_;
pub mod proposal;
pub mod router;

pub use http::{HttpError, HttpFetcher, HttpResponse, MockFetcher, UreqFetcher};
pub use proposal::{
    build_candidates, cosine_similarity, BondCandidate, ResearchProposal,
    ResearchProposalStatus,
};
pub use router::{
    fingerprint_text, run_router, ResearchConfig, ResearchEvidence, ResearchProvider,
    RouterError,
};
pub use chain::{ChainHop, ChainResearchConfig, ChainTerminationReason, ChainTrace};
