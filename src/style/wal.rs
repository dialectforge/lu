//! Port CS.1 — Write-Ahead Log for the conversational style profile
//! store.
//!
//! Mirrors the Port T.3 `<brain>.pending.wal` and Port LB.1
//! `<brain>.lb_session.wal` patterns: append-only, newline-delimited
//! JSON; every append is `flush()` + `sync_all()`; the WAL is
//! truncated by the brain save path after the v3 section #12
//! snapshot becomes durable on disk; replay is idempotent (skip-if
//! `record.ts_ms <= existing vector.last_refined_at_ms`).
//!
//! Anti-hallucination invariant: WAL replay applies the same
//! style-store mutations as the live `tool_ask` / `tool_style_set`
//! / `tool_style_reset` paths. Style is presentation-only — replay
//! never touches `bond_count` or any canonical brain state.

use crate::brain::BrainError;
use crate::style::{StyleSample, StyleVector};
use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

/// One mutation event in the conversational style store.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum StyleWalRecord {
    /// EMA observation from `tool_ask` (or any other system call site
    /// that calls `ConversationalStyleStore::observe`).
    Observe {
        user_key: String,
        sample: StyleSample,
        learning_rate: f32,
        ts_ms: u64,
    },
    /// Operator override from `tool_style_set` (replaces the entire
    /// per-user vector).
    Set {
        user_key: String,
        vector: StyleVector,
        ts_ms: u64,
    },
    /// Operator clear from `tool_style_reset` (removes the per-user
    /// profile entirely; subsequent `get` returns the default).
    Reset { user_key: String, ts_ms: u64 },
}

pub struct StyleWalWriter {
    path: PathBuf,
    file: File,
}

impl StyleWalWriter {
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
        Ok(StyleWalWriter { path, file })
    }

    pub fn append(&mut self, rec: &StyleWalRecord) -> Result<(), BrainError> {
        let line = serde_json::to_string(rec)?;
        self.file.write_all(line.as_bytes())?;
        self.file.write_all(b"\n")?;
        self.file.flush()?;
        self.file.sync_all()?;
        Ok(())
    }

    /// OP-SAVE-ASYNC — current append position (byte offset).
    pub fn position(&mut self) -> Result<u64, BrainError> {
        // Append-mode files report stream_position 0 on Linux; query
        // file length via the path metadata to get the real EOF that
        // subsequent append() calls will start writing at.
        let md = std::fs::metadata(&self.path)?;
        Ok(md.len())
    }

    /// OP-SAVE-ASYNC — optionally takes a watermark byte offset; when
    /// Some(W), bytes [W..end) are preserved so entries appended after
    /// the snapshot started survive the truncate; when None, the file
    /// is zeroed.
    pub fn truncate(&mut self, watermark: Option<u64>) -> Result<(), BrainError> {
        use std::io::{Read, Write};
        match watermark {
            Some(w) => {
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

/// Read every well-formed record from the on-disk WAL. Torn-line
/// tolerant: a single half-written final line is silently skipped
/// while every prior valid line is returned. A non-existent file
/// returns an empty vector — pre-CS.1 brains (no `<brain>.style.wal`
/// alongside their `brain.lbr`) load cleanly via this path.
pub fn read_all(path: &Path) -> Result<Vec<StyleWalRecord>, BrainError> {
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
        match serde_json::from_str::<StyleWalRecord>(&line) {
            Ok(rec) => out.push(rec),
            Err(e) => {
                if std::env::var_os("LU_BRAIN_DEBUG").is_some() {
                    eprintln!(
                        "cs_style WAL replay: skipping torn line {} ({})",
                        i + 1,
                        e
                    );
                }
            }
        }
    }
    Ok(out)
}
