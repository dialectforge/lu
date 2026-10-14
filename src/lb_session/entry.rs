//! Port LB — Left Brain session entry data model.
//!
//! A [`SessionEntry`] is one record in the operational/episodic Left
//! Brain layer. Entries are session-scoped, durable across saves
//! (via `SectionIdV3::SessionMemory`), and never auto-promoted to
//! the canonical brain — promotion goes through T.3's PendingLog.

use crate::lb_session::kind::SessionEntryKind;
use serde::{Deserialize, Serialize};

/// One Left Brain session entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionEntry {
    /// Monotonic id allocated by [`crate::lb_session::SessionMemoryStore`].
    pub entry_id: u64,
    /// The session this entry was written within.
    pub session_id: u64,
    /// Wall-clock ms when the entry was created.
    pub ts_ms: u64,
    /// Categorical tag (thinking trace, observation, hypothesis, …).
    pub kind: SessionEntryKind,
    /// Short human-readable subject ("why is the sky blue?",
    /// "recall(gpt-5) returned 0 results", …). Used by
    /// `lu4d_session_memory_inspect` for prose summaries.
    pub subject: String,
    /// Free-form payload. JSON-encoded structured content lands
    /// here when callers want to attach structuredContent shape
    /// without inflating the schema.
    pub content: String,
    /// Confidence at the moment the entry was written. Snapshot
    /// rather than a live signal — entries do NOT participate in
    /// the canonical confidence aggregation, so this field is
    /// purely descriptive.
    pub confidence_at_creation: f32,
    /// Wall-clock ms when this entry was promoted to a `PendingChange`
    /// via [`crate::lb_session::SessionMemoryStore::propose_to_pending`].
    /// `None` for entries that have never been proposed.
    #[serde(default)]
    pub promoted_at_ms: Option<u64>,
    /// `pending_id` of the `PendingChange` this entry was promoted to.
    /// Cross-link from the Left Brain into the T.3 pending log.
    #[serde(default)]
    pub promoted_pending_id: Option<u64>,
}

impl SessionEntry {
    /// Construct a new pending-write entry. The store overwrites
    /// `entry_id` on push so the caller-supplied value is ignored.
    pub fn new(
        session_id: u64,
        ts_ms: u64,
        kind: SessionEntryKind,
        subject: impl Into<String>,
        content: impl Into<String>,
        confidence_at_creation: f32,
    ) -> Self {
        Self {
            entry_id: 0,
            session_id,
            ts_ms,
            kind,
            subject: subject.into(),
            content: content.into(),
            confidence_at_creation,
            promoted_at_ms: None,
            promoted_pending_id: None,
        }
    }

    /// True if this entry has already been promoted to T.3 PendingLog.
    pub fn is_promoted(&self) -> bool {
        self.promoted_at_ms.is_some()
    }
}
