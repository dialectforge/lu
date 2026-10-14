//! Port T.3 — In-memory `PendingLog` with WAL persistence.
//!
//! The log is the structural answer to "AI safety concerns about
//! autonomous self-modification": idle research, future Cat. 3
//! self-rewriting passes, and any other autonomous mutation route
//! their proposals through here, where they wait for explicit
//! human approve / reject / modify before being committed to the
//! brain.
//!
//! Storage strategy:
//!   * In-memory: ordered map keyed by `pending_id` (BTreeMap so
//!     iteration is deterministic).
//!   * WAL: append-only newline-delimited JSON next to the brain
//!     file at `<brain_path>.pending.wal`. Source of truth between
//!     brain saves.
//!   * Brain file: a `PendingChanges` section serializes the
//!     consolidated state at save time; backward-compatible with
//!     pre-T.3 brain files via `#[serde(default)]`.
//!
//! Crash recovery: load brain → load + replay WAL → state is
//! restored. Replay is idempotent (dedupes by `pending_id`), so
//! the corner case where a save succeeded but the WAL truncate
//! failed leaves no duplicates.

use crate::brain::BrainError;
use crate::pending::change::{PendingChange, PendingChangeKind, PendingStatus};
use crate::pending::wal::{read_all, WalRecord, WalWriter};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// Snapshot of the pending log for serialization in the brain
/// file's `PendingChanges` section.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PendingLogSnapshot {
    #[serde(default)]
    pub entries: Vec<PendingChange>,
    #[serde(default = "default_next_id")]
    pub next_pending_id: u64,
}

fn default_next_id() -> u64 {
    1
}

/// Live pending log: ordered map + monotonic id allocator + open WAL
/// handle (when persistent).
pub struct PendingLog {
    entries: BTreeMap<u64, PendingChange>,
    next_pending_id: u64,
    wal: Option<WalWriter>,
    /// Wall-clock ms of the most recent Apply event (for the
    /// `lu4d_pending_status` `last_commit_at_ms` field).
    last_commit_ms: Option<u64>,
}

impl Default for PendingLog {
    fn default() -> Self {
        PendingLog {
            entries: BTreeMap::new(),
            next_pending_id: 1,
            wal: None,
            last_commit_ms: None,
        }
    }
}

impl std::fmt::Debug for PendingLog {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PendingLog")
            .field("entries", &self.entries.len())
            .field("next_pending_id", &self.next_pending_id)
            .field("wal_path", &self.wal.as_ref().map(|w| w.path().to_path_buf()))
            .field("last_commit_ms", &self.last_commit_ms)
            .finish()
    }
}

impl PendingLog {
    /// Empty log with no WAL backing. Useful for tests and pre-T.3
    /// brain loads (Brain::new without a path).
    pub fn new_in_memory() -> Self {
        Self::default()
    }

    /// Open a WAL-backed log. Replays the WAL into the in-memory
    /// state, then attaches an append handle for further writes.
    pub fn open(wal_path: PathBuf) -> Result<Self, BrainError> {
        let mut log = Self::default();
        let records = read_all(&wal_path)?;
        for rec in records {
            log.apply_wal_record(rec);
        }
        log.wal = Some(WalWriter::open(wal_path)?);
        Ok(log)
    }

    /// Open a WAL-backed log starting from an explicit snapshot
    /// (loaded from the brain file's `PendingChanges` section), then
    /// replay any newer WAL entries on top. Idempotent — entries
    /// already present in the snapshot are skipped during replay.
    pub fn open_with_snapshot(
        wal_path: PathBuf,
        snapshot: PendingLogSnapshot,
    ) -> Result<Self, BrainError> {
        let mut log = Self::default();
        for entry in snapshot.entries {
            log.next_pending_id = log.next_pending_id.max(entry.pending_id + 1);
            if let PendingStatus::Applied { applied_at_ms } = entry.status {
                log.last_commit_ms =
                    Some(log.last_commit_ms.unwrap_or(0).max(applied_at_ms));
            }
            log.entries.insert(entry.pending_id, entry);
        }
        log.next_pending_id = log.next_pending_id.max(snapshot.next_pending_id);
        let records = read_all(&wal_path)?;
        for rec in records {
            log.apply_wal_record(rec);
        }
        log.wal = Some(WalWriter::open(wal_path)?);
        Ok(log)
    }

    /// Replay a WAL record into the in-memory state. Idempotent —
    /// `Push` is a no-op when the id already exists; lifecycle
    /// transitions overwrite (last write wins, which mirrors the
    /// on-disk order).
    fn apply_wal_record(&mut self, rec: WalRecord) {
        match rec {
            WalRecord::Push(change) => {
                if !self.entries.contains_key(&change.pending_id) {
                    self.next_pending_id =
                        self.next_pending_id.max(change.pending_id + 1);
                    self.entries.insert(change.pending_id, change);
                }
            }
            WalRecord::Approve {
                pending_id,
                approved_at_ms,
                approved_by,
            } => {
                if let Some(e) = self.entries.get_mut(&pending_id) {
                    e.status = PendingStatus::Approved {
                        approved_at_ms,
                        approved_by,
                    };
                }
            }
            WalRecord::Reject {
                pending_id,
                rejected_at_ms,
                rejected_by,
                reason,
            } => {
                if let Some(e) = self.entries.get_mut(&pending_id) {
                    e.status = PendingStatus::Rejected {
                        rejected_at_ms,
                        rejected_by,
                        reason,
                    };
                }
            }
            WalRecord::Apply {
                pending_id,
                applied_at_ms,
            } => {
                if let Some(e) = self.entries.get_mut(&pending_id) {
                    e.status = PendingStatus::Applied { applied_at_ms };
                    self.last_commit_ms =
                        Some(self.last_commit_ms.unwrap_or(0).max(applied_at_ms));
                }
            }
            WalRecord::Modify {
                pending_id,
                new_confidence,
                new_weight,
                new_reason,
            } => {
                if let Some(e) = self.entries.get_mut(&pending_id) {
                    if let Some(c) = new_confidence {
                        e.confidence = c;
                    }
                    if let Some(w) = new_weight {
                        match &mut e.change_kind {
                            PendingChangeKind::AddBond { weight, .. } => *weight = w,
                            PendingChangeKind::ModifyBondWeight { new_weight, .. } => {
                                *new_weight = w
                            }
                            _ => {}
                        }
                    }
                    if let Some(r) = new_reason {
                        e.reason = r;
                    }
                }
            }
        }
    }

    pub fn pending_count(&self) -> usize {
        self.entries
            .values()
            .filter(|e| matches!(e.status, PendingStatus::Pending))
            .count()
    }

    pub fn approved_unapplied_count(&self) -> usize {
        self.entries
            .values()
            .filter(|e| matches!(e.status, PendingStatus::Approved { .. }))
            .count()
    }

    pub fn applied_count(&self) -> usize {
        self.entries
            .values()
            .filter(|e| matches!(e.status, PendingStatus::Applied { .. }))
            .count()
    }

    pub fn rejected_count(&self) -> usize {
        self.entries
            .values()
            .filter(|e| matches!(e.status, PendingStatus::Rejected { .. }))
            .count()
    }

    pub fn total_count(&self) -> usize {
        self.entries.len()
    }

    pub fn last_commit_at_ms(&self) -> Option<u64> {
        self.last_commit_ms
    }

    pub fn next_id(&self) -> u64 {
        self.next_pending_id
    }

    pub fn wal_path(&self) -> Option<&Path> {
        self.wal.as_ref().map(|w| w.path())
    }

    pub fn get(&self, id: u64) -> Option<&PendingChange> {
        self.entries.get(&id)
    }

    /// Iterate entries in ascending pending_id order.
    pub fn iter(&self) -> impl Iterator<Item = (&u64, &PendingChange)> {
        self.entries.iter()
    }

    /// Push a new pending change. Allocates `pending_id` and forces
    /// the status to `Pending` regardless of what the caller
    /// supplied. Returns the allocated id.
    pub fn push(&mut self, mut change: PendingChange) -> Result<u64, BrainError> {
        change.pending_id = self.next_pending_id;
        change.status = PendingStatus::Pending;
        let id = change.pending_id;
        let rec = WalRecord::Push(change.clone());
        if let Some(w) = self.wal.as_mut() {
            w.append(&rec)?;
        }
        self.entries.insert(id, change);
        self.next_pending_id += 1;
        Ok(id)
    }

    /// Pending → Approved.
    pub fn approve(
        &mut self,
        id: u64,
        approved_at_ms: u64,
        approved_by: impl Into<String>,
    ) -> Result<(), BrainError> {
        let approved_by = approved_by.into();
        match self.entries.get_mut(&id) {
            Some(e) => {
                if !e.status.is_pending() {
                    return Err(BrainError::Format(format!(
                        "pending {} already {}",
                        id,
                        e.status.tag()
                    )));
                }
                e.status = PendingStatus::Approved {
                    approved_at_ms,
                    approved_by: approved_by.clone(),
                };
            }
            None => {
                return Err(BrainError::Format(format!("unknown pending id {}", id)))
            }
        }
        if let Some(w) = self.wal.as_mut() {
            w.append(&WalRecord::Approve {
                pending_id: id,
                approved_at_ms,
                approved_by,
            })?;
        }
        Ok(())
    }

    /// Pending → Rejected.
    pub fn reject(
        &mut self,
        id: u64,
        rejected_at_ms: u64,
        rejected_by: impl Into<String>,
        reason: Option<String>,
    ) -> Result<(), BrainError> {
        let rejected_by = rejected_by.into();
        match self.entries.get_mut(&id) {
            Some(e) => {
                if !e.status.is_pending() {
                    return Err(BrainError::Format(format!(
                        "pending {} already {}",
                        id,
                        e.status.tag()
                    )));
                }
                e.status = PendingStatus::Rejected {
                    rejected_at_ms,
                    rejected_by: rejected_by.clone(),
                    reason: reason.clone(),
                };
            }
            None => {
                return Err(BrainError::Format(format!("unknown pending id {}", id)))
            }
        }
        if let Some(w) = self.wal.as_mut() {
            w.append(&WalRecord::Reject {
                pending_id: id,
                rejected_at_ms,
                rejected_by,
                reason,
            })?;
        }
        Ok(())
    }

    /// Approved → Applied. Called by `crate::pending::apply::apply_change`
    /// after the brain mutation has succeeded.
    pub fn mark_applied(&mut self, id: u64, applied_at_ms: u64) -> Result<(), BrainError> {
        match self.entries.get_mut(&id) {
            Some(e) => {
                if !e.status.is_approved() {
                    return Err(BrainError::Format(format!(
                        "pending {} not approved (status={})",
                        id,
                        e.status.tag()
                    )));
                }
                e.status = PendingStatus::Applied { applied_at_ms };
                self.last_commit_ms =
                    Some(self.last_commit_ms.unwrap_or(0).max(applied_at_ms));
            }
            None => {
                return Err(BrainError::Format(format!("unknown pending id {}", id)))
            }
        }
        if let Some(w) = self.wal.as_mut() {
            w.append(&WalRecord::Apply {
                pending_id: id,
                applied_at_ms,
            })?;
        }
        Ok(())
    }

    /// Modify a still-Pending entry's confidence / weight / reason
    /// in place.
    pub fn modify(
        &mut self,
        id: u64,
        new_confidence: Option<f32>,
        new_weight: Option<f32>,
        new_reason: Option<String>,
    ) -> Result<(), BrainError> {
        match self.entries.get_mut(&id) {
            Some(e) => {
                if !e.status.is_pending() {
                    return Err(BrainError::Format(format!(
                        "pending {} already {}",
                        id,
                        e.status.tag()
                    )));
                }
                if let Some(c) = new_confidence {
                    e.confidence = c;
                }
                if let Some(w) = new_weight {
                    match &mut e.change_kind {
                        PendingChangeKind::AddBond { weight, .. } => *weight = w,
                        PendingChangeKind::ModifyBondWeight { new_weight, .. } => {
                            *new_weight = w
                        }
                        _ => {}
                    }
                }
                if let Some(r) = new_reason.as_ref() {
                    e.reason = r.clone();
                }
            }
            None => {
                return Err(BrainError::Format(format!("unknown pending id {}", id)))
            }
        }
        if let Some(w) = self.wal.as_mut() {
            w.append(&WalRecord::Modify {
                pending_id: id,
                new_confidence,
                new_weight,
                new_reason,
            })?;
        }
        Ok(())
    }

    /// Build a snapshot for serialization in the brain file.
    pub fn snapshot(&self) -> PendingLogSnapshot {
        PendingLogSnapshot {
            entries: self.entries.values().cloned().collect(),
            next_pending_id: self.next_pending_id,
        }
    }

    /// OP-SAVE-ASYNC — current WAL byte position, or None when no
    /// WAL is attached (in-memory brain).
    pub fn wal_position(&mut self) -> Result<Option<u64>, BrainError> {
        match self.wal.as_mut() {
            Some(w) => Ok(Some(w.position()?)),
            None => Ok(None),
        }
    }

    /// Truncate the WAL after a successful brain save consolidates
    /// the in-memory state on disk. OP-SAVE-ASYNC: optionally accept
    /// a watermark so entries appended after the snapshot survive the
    /// truncate.
    pub fn truncate_wal(&mut self, watermark: Option<u64>) -> Result<(), BrainError> {
        if let Some(w) = self.wal.as_mut() {
            w.truncate(watermark)?;
        }
        Ok(())
    }
}
