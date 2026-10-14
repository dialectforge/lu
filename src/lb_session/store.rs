//! Port LB - Left Brain session memory store. Port LB.1 adds the
//! optional WAL handle (`<brain>.lb_session.wal`) for between-saves
//! durability and a `push_internal` path so system writers (`think`,
//! `multi_think`, `research_loop_run`, `wonder`) can land entries
//! even when the user has not started an explicit session.
//!
//! Hard rule (matches `agent/memory/project_lu_session_memory.md`):
//! session entries inform reasoning but NEVER auto-promote to fact
//! bonds. Promotion goes through [`crate::lb_session::propose_to_pending`].

use crate::brain::BrainError;
use crate::lb_session::entry::SessionEntry;
use crate::lb_session::kind::SessionEntryKind;
use crate::lb_session::wal::{LbWalRecord, LbWalWriter, read_all as read_lb_wal};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

/// Reserved session id used by `push_internal` when no real session
/// is active. Picked at the very top of the u64 range so it never
/// collides with `Brain::session_start()` ids (which start at 1 and
/// monotonically increase).
pub const SCRATCHPAD_SESSION_ID: u64 = u64::MAX;

/// Snapshot of the session memory store for serialization in v3
/// section #11.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SessionMemorySnapshot {
    #[serde(default)]
    pub entries: Vec<SessionEntry>,
    #[serde(default = "default_next_id")]
    pub next_entry_id: u64,
    #[serde(default)]
    pub archived_sessions: Vec<u64>,
}

fn default_next_id() -> u64 {
    1
}

/// Live session memory store.
#[derive(Default)]
pub struct SessionMemoryStore {
    entries: BTreeMap<u64, SessionEntry>,
    next_entry_id: u64,
    active_session: Option<u64>,
    archived_sessions: BTreeSet<u64>,
    /// Port LB.1 - optional WAL append handle. Populated by
    /// `Brain::attach_lb_session_wal`; absent for `Brain::new`
    /// in-memory brains and for tests that build a store directly.
    wal: Option<LbWalWriter>,
}

impl std::fmt::Debug for SessionMemoryStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SessionMemoryStore")
            .field("entries", &self.entries.len())
            .field("next_entry_id", &self.next_entry_id)
            .field("active_session", &self.active_session)
            .field("archived_sessions", &self.archived_sessions)
            .field("wal_path", &self.wal.as_ref().map(|w| w.path().to_path_buf()))
            .finish()
    }
}

impl SessionMemoryStore {
    pub fn new() -> Self {
        Self {
            entries: BTreeMap::new(),
            next_entry_id: 1,
            active_session: None,
            archived_sessions: BTreeSet::new(),
            wal: None,
        }
    }

    pub fn from_snapshot(snap: SessionMemorySnapshot) -> Self {
        let mut entries = BTreeMap::new();
        let mut max_id = 0u64;
        for e in snap.entries {
            max_id = max_id.max(e.entry_id);
            entries.insert(e.entry_id, e);
        }
        let next_entry_id = snap.next_entry_id.max(max_id + 1).max(1);
        let archived_sessions: BTreeSet<u64> =
            snap.archived_sessions.into_iter().collect();
        Self {
            entries,
            next_entry_id,
            active_session: None,
            archived_sessions,
            wal: None,
        }
    }

    /// Port LB.1 - load a store from a snapshot AND replay the WAL
    /// at `wal_path` on top. Idempotent: replay skips entries that
    /// were already loaded from the snapshot (dedupe by entry_id).
    pub fn from_snapshot_with_wal(
        snap: SessionMemorySnapshot,
        wal_path: PathBuf,
    ) -> Result<Self, BrainError> {
        let mut store = Self::from_snapshot(snap);
        let records = read_lb_wal(&wal_path)?;
        for rec in records {
            store.apply_wal_record(rec);
        }
        store.wal = Some(LbWalWriter::open(wal_path)?);
        Ok(store)
    }

    /// Replay one WAL record into the in-memory state. Idempotent.
    fn apply_wal_record(&mut self, rec: LbWalRecord) {
        match rec {
            LbWalRecord::Push(entry) => {
                if !self.entries.contains_key(&entry.entry_id) {
                    self.next_entry_id =
                        self.next_entry_id.max(entry.entry_id + 1);
                    self.entries.insert(entry.entry_id, entry);
                }
            }
            LbWalRecord::MarkPromoted {
                entry_id,
                ts_ms,
                pending_id,
            } => {
                if let Some(e) = self.entries.get_mut(&entry_id) {
                    if e.promoted_at_ms.is_none() {
                        e.promoted_at_ms = Some(ts_ms);
                        e.promoted_pending_id = Some(pending_id);
                    }
                }
            }
            LbWalRecord::Archive { session_id } => {
                self.archived_sessions.insert(session_id);
                if self.active_session == Some(session_id) {
                    self.active_session = None;
                }
            }
        }
    }

    /// Port LB.1 - attach a freshly-opened WAL writer (used by tests
    /// and `Brain::attach_lb_session_wal`).
    pub fn attach_wal(&mut self, wal: LbWalWriter) {
        self.wal = Some(wal);
    }

    /// Port LB.1 - reset the WAL after a successful brain save.
    /// Mirrors `PendingLog::truncate_wal`. Idempotent on the no-WAL
    /// case (Brain::new in-memory brains skip the WAL entirely).
    pub fn wal_position(&mut self) -> Result<Option<u64>, BrainError> {
        match self.wal.as_mut() {
            Some(w) => Ok(Some(w.position()?)),
            None => Ok(None),
        }
    }

    pub fn truncate_wal(&mut self, watermark: Option<u64>) -> Result<(), BrainError> {
        if let Some(w) = self.wal.as_mut() {
            w.truncate(watermark)?;
        }
        Ok(())
    }

    pub fn snapshot(&self) -> SessionMemorySnapshot {
        SessionMemorySnapshot {
            entries: self.entries.values().cloned().collect(),
            next_entry_id: self.next_entry_id,
            archived_sessions: self.archived_sessions.iter().copied().collect(),
        }
    }

    pub fn total_count(&self) -> usize {
        self.entries.len()
    }

    pub fn next_id(&self) -> u64 {
        self.next_entry_id
    }

    pub fn active_session(&self) -> Option<u64> {
        self.active_session
    }

    pub fn is_archived(&self, session_id: u64) -> bool {
        self.archived_sessions.contains(&session_id)
    }

    pub fn archived_sessions(&self) -> Vec<u64> {
        self.archived_sessions.iter().copied().collect()
    }

    pub fn activate_session(&mut self, session_id: u64) -> Result<(), BrainError> {
        if self.archived_sessions.contains(&session_id) {
            return Err(BrainError::Format(format!(
                "session {} is archived (immutable)",
                session_id
            )));
        }
        self.active_session = Some(session_id);
        Ok(())
    }

    pub fn deactivate_session(&mut self) {
        self.active_session = None;
    }

    /// Push a new entry under the active session (existing LB API).
    pub fn push(&mut self, mut entry: SessionEntry) -> Result<u64, BrainError> {
        let active = self.active_session.ok_or_else(|| {
            BrainError::Format(
                "no active session - call activate_session before writing".into(),
            )
        })?;
        if entry.session_id == 0 {
            entry.session_id = active;
        } else if entry.session_id != active {
            return Err(BrainError::Format(format!(
                "entry session_id {} does not match active session {}",
                entry.session_id, active
            )));
        }
        if self.archived_sessions.contains(&entry.session_id) {
            return Err(BrainError::Format(format!(
                "session {} is archived (immutable)",
                entry.session_id
            )));
        }
        let id = self.next_entry_id;
        entry.entry_id = id;
        if let Some(w) = self.wal.as_mut() {
            w.append(&LbWalRecord::Push(entry.clone()))?;
        }
        self.entries.insert(id, entry);
        self.next_entry_id += 1;
        Ok(id)
    }

    /// Port LB.1 - system-internal push that bypasses the
    /// active-session guard. Used by auto-routing call sites
    /// (`think`, `multi_think`, `research_loop_run`, wonder push)
    /// that may run without an explicit `session_start`. Falls back
    /// to [`SCRATCHPAD_SESSION_ID`] when no session is active.
    /// Archive guard still applies to the chosen session id.
    pub fn push_internal(&mut self, mut entry: SessionEntry) -> Result<u64, BrainError> {
        let sid = if entry.session_id != 0 {
            entry.session_id
        } else {
            self.active_session.unwrap_or(SCRATCHPAD_SESSION_ID)
        };
        if self.archived_sessions.contains(&sid) {
            return Err(BrainError::Format(format!(
                "session {} is archived (immutable)",
                sid
            )));
        }
        entry.session_id = sid;
        let id = self.next_entry_id;
        entry.entry_id = id;
        if let Some(w) = self.wal.as_mut() {
            w.append(&LbWalRecord::Push(entry.clone()))?;
        }
        self.entries.insert(id, entry);
        self.next_entry_id += 1;
        Ok(id)
    }

    pub fn get(&self, id: u64) -> Option<&SessionEntry> {
        self.entries.get(&id)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&u64, &SessionEntry)> {
        self.entries.iter()
    }

    pub fn filter(
        &self,
        session_id: Option<u64>,
        kind: Option<&SessionEntryKind>,
        limit: usize,
    ) -> Vec<SessionEntry> {
        self.entries
            .values()
            .filter(|e| session_id.map_or(true, |s| e.session_id == s))
            .filter(|e| kind.map_or(true, |k| &e.kind == k))
            .take(limit.max(1))
            .cloned()
            .collect()
    }

    pub fn counts_by_session(&self) -> BTreeMap<u64, usize> {
        let mut out = BTreeMap::new();
        for e in self.entries.values() {
            *out.entry(e.session_id).or_insert(0) += 1;
        }
        out
    }

    pub fn counts_by_kind(&self) -> BTreeMap<&'static str, usize> {
        let mut out = BTreeMap::new();
        for e in self.entries.values() {
            *out.entry(e.kind.tag()).or_insert(0) += 1;
        }
        out
    }

    pub fn archive_session(&mut self, session_id: u64) {
        if self.active_session == Some(session_id) {
            self.active_session = None;
        }
        self.archived_sessions.insert(session_id);
        if let Some(w) = self.wal.as_mut() {
            // Best-effort: an Archive WAL append failure should not
            // brick the in-memory mutation. The next save consolidates
            // archived_sessions into the v3 section regardless.
            let _ = w.append(&LbWalRecord::Archive { session_id });
        }
    }

    pub fn mark_promoted(
        &mut self,
        entry_id: u64,
        ts_ms: u64,
        pending_id: u64,
    ) -> Result<(), BrainError> {
        match self.entries.get_mut(&entry_id) {
            Some(e) => {
                if e.promoted_at_ms.is_some() {
                    return Err(BrainError::Format(format!(
                        "entry {} already promoted at {}",
                        entry_id,
                        e.promoted_at_ms.unwrap(),
                    )));
                }
                e.promoted_at_ms = Some(ts_ms);
                e.promoted_pending_id = Some(pending_id);
                if let Some(w) = self.wal.as_mut() {
                    w.append(&LbWalRecord::MarkPromoted {
                        entry_id,
                        ts_ms,
                        pending_id,
                    })?;
                }
                Ok(())
            }
            None => Err(BrainError::Format(format!(
                "unknown entry id {}",
                entry_id
            ))),
        }
    }
}
