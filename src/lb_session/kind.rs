//! Port LB — Left Brain session entry kinds.
//!
//! Categorical tag for a [`crate::lb_session::SessionEntry`]. The set
//! is deliberately open-ended; each variant is a different shape of
//! session-scoped operational state that informs reasoning but does
//! NOT auto-promote to a fact bond. Promotion goes through the T.3
//! pending-change path (see
//! [`crate::lb_session::SessionMemoryStore::propose_to_pending`]).

use serde::{Deserialize, Serialize};

/// Categorical tag for a session memory entry.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SessionEntryKind {
    /// One step of a `lu4d_think` reasoning trace. Lands in the Left
    /// Brain and never directly modifies the canonical brain.
    ThinkingTrace,
    /// A run-time observation about brain state, recall results, or
    /// the world (e.g. "recall returned 0 results for `gpt-5`").
    Observation,
    /// An unverified hypothesis Lu is exploring but has not yet
    /// approved as a fact.
    Hypothesis,
    /// A persistent task or goal that should survive process
    /// restarts (the goal layer of the agent loop).
    Goal,
    /// A wonder-queue item Lu has surfaced but not yet researched.
    /// Migrating wonder state into Left Brain (vs. a denormalized
    /// `WonderQueueState` v3 section) is a follow-up port; for
    /// today this kind exists so research+wonder integrations can
    /// land without another schema bump.
    WonderItem,
    /// Research-loop output (URL, fetched content summary, derived
    /// candidate bond) before any human approval.
    ResearchResult,
    /// A user-supplied statement received during the session
    /// (raw, before parse / classification).
    UserStatement,
}

impl SessionEntryKind {
    /// Stable short tag used by structuredContent and probe scripts.
    pub fn tag(&self) -> &'static str {
        match self {
            SessionEntryKind::ThinkingTrace => "thinking_trace",
            SessionEntryKind::Observation => "observation",
            SessionEntryKind::Hypothesis => "hypothesis",
            SessionEntryKind::Goal => "goal",
            SessionEntryKind::WonderItem => "wonder_item",
            SessionEntryKind::ResearchResult => "research_result",
            SessionEntryKind::UserStatement => "user_statement",
        }
    }

    /// All kinds, for status-counting and exhaustive iteration.
    pub fn all() -> &'static [SessionEntryKind] {
        &[
            SessionEntryKind::ThinkingTrace,
            SessionEntryKind::Observation,
            SessionEntryKind::Hypothesis,
            SessionEntryKind::Goal,
            SessionEntryKind::WonderItem,
            SessionEntryKind::ResearchResult,
            SessionEntryKind::UserStatement,
        ]
    }
}
