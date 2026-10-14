//! Port LB.1 - Write-Ahead Log for the Left Brain session memory store.
//!
//! Mirrors the Port T.3 `<brain>.pending.wal` pattern: append-only,
//! newline-delimited JSON; every append is `flush()` + `sync_all()`;
//! the WAL is truncated by the brain save path after the v3
//! section #11 snapshot becomes durable on disk; replay is
//! idempotent (skip-if-entry_id-already-present).
//!
//! See the LB scope rule in `src/lb_session/mod.rs`: this WAL is
//! a context layer, NOT a truth layer. Entries replayed from this
//! WAL still NEVER auto-promote to fact bonds.

use crate::brain::BrainError;
use crate::lb_session::entry::SessionEntry;
use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

/// One mutation event in the Left Brain session memory store.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LbWalRecord {
    Push(SessionEntry),
    MarkPromoted {
        entry_id: u64,
        ts_ms: u64,
        pending_id: u64,
    },
    Archive { session_id: u64 },
}

pub struct LbWalWriter {
    path: PathBuf,
    file: File,
}

impl LbWalWriter {
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
        Ok(LbWalWriter { path, file })
    }

    pub fn append(&mut self, rec: &LbWalRecord) -> Result<(), BrainError> {
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

pub fn read_all(path: &Path) -> Result<Vec<LbWalRecord>, BrainError> {
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
        match serde_json::from_str::<LbWalRecord>(&line) {
            Ok(rec) => out.push(rec),
            Err(e) => {
                if std::env::var_os("LU_BRAIN_DEBUG").is_some() {
                    eprintln!(
                        "lb_session WAL replay: skipping torn line {} ({})",
                        i + 1,
                        e
                    );
                }
            }
        }
    }
    Ok(out)
}
