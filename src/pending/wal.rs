//! Port T.3 — Write-Ahead Log for the pending changes log.
//!
//! Append-only newline-delimited JSON. Each line is a [`WalRecord`]
//! capturing one mutation event (push / approve / reject / apply /
//! modify). On startup the log is replayed line-by-line into the
//! in-memory [`crate::pending::PendingLog`].
//!
//! Crash safety: every append `flush`es and `sync_all`s the file. If
//! the process dies mid-write, the partial line at EOF fails to
//! parse and is skipped on replay; earlier lines are durable because
//! the previous record's fsync completed before the new one began.
//!
//! After a successful brain save, [`WalWriter::truncate`] resets the
//! file — the on-disk brain section now holds the consolidated
//! state, so the WAL only needs to capture mutations since that
//! save.

use crate::brain::BrainError;
use crate::pending::change::PendingChange;
use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

/// One mutation event in the pending log.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WalRecord {
    /// Brand-new pending entry.
    Push(PendingChange),
    /// Pending → Approved.
    Approve {
        pending_id: u64,
        approved_at_ms: u64,
        approved_by: String,
    },
    /// Pending → Rejected.
    Reject {
        pending_id: u64,
        rejected_at_ms: u64,
        rejected_by: String,
        reason: Option<String>,
    },
    /// Approved → Applied (after the brain mutation succeeded).
    Apply {
        pending_id: u64,
        applied_at_ms: u64,
    },
    /// In-place edit while still Pending.
    Modify {
        pending_id: u64,
        new_confidence: Option<f32>,
        new_weight: Option<f32>,
        new_reason: Option<String>,
    },
}

/// Open append handle to the WAL file. Optional — in-memory logs
/// (test-only and pre-T.3 brains loaded with no persistence) skip
/// the WAL entirely.
pub struct WalWriter {
    path: PathBuf,
    file: File,
}

impl WalWriter {
    /// Open or create the WAL file in append mode.
    pub fn open(path: PathBuf) -> Result<Self, BrainError> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)?;
        Ok(WalWriter { path, file })
    }

    /// Append one record. Flushes + fsyncs before returning.
    pub fn append(&mut self, rec: &WalRecord) -> Result<(), BrainError> {
        let line = serde_json::to_string(rec)?;
        self.file.write_all(line.as_bytes())?;
        self.file.write_all(b"\n")?;
        self.file.flush()?;
        self.file.sync_all()?;
        Ok(())
    }

    /// Current append position (byte offset) of the WAL. Used by the
    /// OP-SAVE-ASYNC snapshot path to capture a high-water mark so a
    /// post-save truncate preserves entries appended after the
    /// snapshot was taken.
    pub fn position(&mut self) -> Result<u64, BrainError> {
        // Append-mode files report stream_position 0 on Linux; query
        // file length via the path metadata to get the real EOF that
        // subsequent append() calls will start writing at.
        let md = std::fs::metadata(&self.path)?;
        Ok(md.len())
    }

    /// Reset the file to empty. Called by the brain save path after a
    /// successful save consolidates the pending state into the brain
    /// file. OP-SAVE-ASYNC: optionally takes a watermark byte offset;
    /// when Some(W), bytes [W..end) are preserved (entries appended
    /// after the snapshot started); when None, the file is zeroed.
    pub fn truncate(&mut self, watermark: Option<u64>) -> Result<(), BrainError> {
        use std::io::{Read, Write};
        match watermark {
            Some(w) => {
                // Read tail bytes [w..end) BEFORE truncating, then
                // rewrite them at offset 0 so subsequent appends
                // continue past them.
                let mut src = OpenOptions::new().read(true).open(&self.path)?;
                use std::io::Seek;
                let file_len = src.seek(std::io::SeekFrom::End(0))?;
                let tail_len = file_len.saturating_sub(w);
                let tail = if tail_len > 0 {
                    src.seek(std::io::SeekFrom::Start(w))?;
                    let mut buf = Vec::with_capacity(tail_len as usize);
                    src.read_to_end(&mut buf)?;
                    buf
                } else {
                    Vec::new()
                };
                drop(src);
                let mut trunc = OpenOptions::new()
                    .create(true)
                    .write(true)
                    .truncate(true)
                    .open(&self.path)?;
                if !tail.is_empty() {
                    trunc.write_all(&tail)?;
                }
                trunc.sync_all()?;
                drop(trunc);
                self.file = OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&self.path)?;
            }
            None => {
                let trunc = OpenOptions::new()
                    .create(true)
                    .write(true)
                    .truncate(true)
                    .open(&self.path)?;
                trunc.sync_all()?;
                drop(trunc);
                self.file = OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&self.path)?;
            }
        }
        Ok(())
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

/// Read every record in `path` in order. Skips lines that fail to
/// parse (mid-write torn lines) — earlier records are still
/// authoritative because each record fsyncs before the next begins.
pub fn read_all(path: &Path) -> Result<Vec<WalRecord>, BrainError> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let f = File::open(path)?;
    let reader = BufReader::new(f);
    let mut out = Vec::new();
    for (i, line) in reader.lines().enumerate() {
        let line = match line {
            Ok(s) => s,
            Err(_) => continue,
        };
        if line.trim().is_empty() {
            continue;
        }
        match serde_json::from_str::<WalRecord>(&line) {
            Ok(rec) => out.push(rec),
            Err(e) => {
                if std::env::var_os("LU_BRAIN_DEBUG").is_some() {
                    eprintln!(
                        "pending WAL replay: skipping torn line {} ({})",
                        i + 1,
                        e
                    );
                }
            }
        }
    }
    Ok(out)
}
