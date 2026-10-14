//! OP-CURIOSITY-1 — the DURABLE, CROSS-PROCESS wonder queue.
//!
//! # Why this exists
//!
//! The curiosity engine is two halves that, since the OP-007 privilege
//! split, live in two different processes:
//!
//! * **DETECT** (the producer) runs inside the spread-activation loop.
//!   It is compiled into the public-facing `lu_guest_http` binary and
//!   fires on essentially every real user query.
//! * **FILL** (the consumer) — hub-fill, triangle closure, single-hop
//!   and chained research — needs to WRITE bonds into the canonical
//!   brain. The guest binary is a structurally read-only reader and is
//!   additionally confined by systemd (`ReadOnlyPaths=/var/lib/lu`), so
//!   it cannot and must not do this.
//!
//! Between the two halves the queue was a `BinaryHeap` in RAM belonging
//! to whichever `Brain` happened to be loaded. Detected gaps were
//! therefore discarded on every process exit, and the consumer — living
//! in a different process entirely — never saw a single one of them.
//!
//! This module is the thing that joins them: an append-only,
//! line-delimited JSON journal on a filesystem path that BOTH processes
//! can name, outside the brain directory.
//!
//! # The privilege story (read this before changing anything)
//!
//! The whole point of the split is that the internet-facing process is
//! not a writer to the canonical brain. This sidecar does not weaken
//! that, and must not be extended in a way that does:
//!
//! * The producer's ONLY write capability is `append` — `O_APPEND` on a
//!   single journal file in a directory that has nothing else in it. It
//!   never opens the brain for writing, never truncates, never rewrites,
//!   and never reads back what the consumer wrote.
//! * `rewrite` (compaction after a drain) is a CONSUMER operation. It is
//!   here rather than in the worker crate only because it shares the
//!   record format; the guest never calls it.
//! * A hostile or buggy producer can therefore do exactly one thing to
//!   this system: append junk to a journal. The consumer treats every
//!   line as untrusted input and skips anything it cannot parse.
//!
//! # Format
//!
//! One [`WonderRecord`] per line as JSON, UTF-8, newline-terminated.
//! Chosen over a binary/rkyv encoding on purpose:
//!
//! * `O_APPEND` writes of whole lines do not interleave at these sizes,
//!   so concurrent producers cannot corrupt each other's records.
//! * A torn tail (power loss mid-write) costs exactly one record, and
//!   the reader drops it silently instead of failing the whole file.
//! * It is greppable. This is a demo-critical mechanism whose whole
//!   value is being able to SHOW that a gap was detected, persisted,
//!   and later filled.

use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::curiosity::CuriositySignal;
use crate::seed::Seed;

/// Record format version. Bump on any incompatible field change; the
/// reader skips records whose version it does not recognise rather than
/// guessing.
pub const WONDER_RECORD_V: u32 = 1;

/// Default file name inside the wonder-queue directory.
pub const WONDER_JOURNAL_FILE: &str = "wonder_queue.jsonl";

/// One persisted wonder.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WonderRecord {
    /// Format version. See [`WONDER_RECORD_V`].
    pub v: u32,
    /// Epoch millis at which the PRODUCER appended this record. Distinct
    /// from `signal.timestamp_ms` (detection time) so a slow or batched
    /// producer is visible in the journal.
    pub appended_at_ms: u64,
    /// The signal itself, verbatim.
    pub signal: CuriositySignal,
    /// Human-readable label for `signal.seed` when the producer could
    /// resolve one. Carried across the process boundary because the
    /// consumer's `gap_label_map` is otherwise empty for synthetic gap
    /// seeds that have no node in the graph.
    #[serde(default)]
    pub label: Option<String>,
    /// The surface query that produced the gap, when known. Diagnostic
    /// only — never fed back into the graph.
    #[serde(default)]
    pub query: Option<String>,
}

impl WonderRecord {
    /// Build a record from a signal at the current wall clock.
    pub fn new(signal: CuriositySignal, label: Option<String>, query: Option<String>) -> Self {
        WonderRecord {
            v: WONDER_RECORD_V,
            appended_at_ms: crate::brain::now_ms() as u64,
            signal,
            label,
            query,
        }
    }
}

/// Handle on the journal file.
///
/// Holds no file descriptor between calls: each `append` opens, writes,
/// flushes and closes. That is deliberate — the producer is a long-lived
/// public service and should not hold a writable descriptor open across
/// its whole lifetime for a once-per-query write.
#[derive(Debug, Clone)]
pub struct WonderSidecar {
    path: PathBuf,
}

impl WonderSidecar {
    /// Point at an explicit journal file.
    pub fn new(path: impl Into<PathBuf>) -> Self {
        WonderSidecar { path: path.into() }
    }

    /// Point at the default journal inside `dir`, creating `dir` if it
    /// does not exist. Fails loudly — a producer configured with a
    /// wonder-queue directory it cannot create should say so at startup,
    /// not silently drop every gap it finds for the rest of its life.
    pub fn in_dir(dir: impl AsRef<Path>) -> std::io::Result<Self> {
        let dir = dir.as_ref();
        std::fs::create_dir_all(dir)?;
        Ok(WonderSidecar {
            path: dir.join(WONDER_JOURNAL_FILE),
        })
    }

    /// The journal path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// PRODUCER OPERATION — append records. Returns how many were written.
    ///
    /// Opens `O_APPEND | O_CREAT`. Never truncates. Never seeks. A
    /// record that fails to serialise is skipped rather than aborting
    /// the batch, because losing one gap signal is not worth failing a
    /// user's query over.
    pub fn append(&self, records: &[WonderRecord]) -> std::io::Result<usize> {
        if records.is_empty() {
            return Ok(0);
        }
        let mut buf = String::new();
        let mut n = 0usize;
        for r in records {
            match serde_json::to_string(r) {
                Ok(line) => {
                    // A raw newline would break the one-record-per-line
                    // invariant. serde_json escapes newlines inside
                    // strings so this cannot happen; the assertion
                    // documents the invariant for whoever changes the
                    // encoder next.
                    debug_assert!(!line.contains('\n'));
                    buf.push_str(&line);
                    buf.push('\n');
                    n += 1;
                }
                Err(_) => continue,
            }
        }
        if n == 0 {
            return Ok(0);
        }
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut f = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;
        f.write_all(buf.as_bytes())?;
        f.flush()?;
        Ok(n)
    }

    /// Read every well-formed record. A missing file is EMPTY, not an
    /// error (a producer that has not yet seen a query is normal).
    ///
    /// Malformed and unknown-version lines are SKIPPED and counted, not
    /// fatal: this file is written by another process and may have a
    /// torn tail.
    pub fn load(&self) -> std::io::Result<LoadedJournal> {
        let f = match std::fs::File::open(&self.path) {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Ok(LoadedJournal::default())
            }
            Err(e) => return Err(e),
        };
        let mut out = LoadedJournal::default();
        for line in BufReader::new(f).lines() {
            let line = match line {
                Ok(l) => l,
                Err(_) => {
                    out.skipped += 1;
                    continue;
                }
            };
            if line.trim().is_empty() {
                continue;
            }
            match serde_json::from_str::<WonderRecord>(&line) {
                Ok(r) if r.v == WONDER_RECORD_V => out.records.push(r),
                _ => out.skipped += 1,
            }
        }
        Ok(out)
    }

    /// CONSUMER OPERATION — replace the journal with `keep`.
    ///
    /// Written to a sibling temp file and renamed, so a reader sees
    /// either the whole old journal or the whole new one. Called after a
    /// drain to retire consumed wonders; the guest never calls it.
    ///
    /// A concurrent producer append CAN be lost by this (it may land on
    /// the old inode between our read and our rename). That is an
    /// accepted, bounded loss: on this deployment producer and consumer
    /// do not run concurrently (see `docs/CURIOSITY.md`), and a dropped
    /// gap signal is re-detected the next time that query is asked.
    pub fn rewrite(&self, keep: &[WonderRecord]) -> std::io::Result<()> {
        let tmp = self.path.with_extension("jsonl.tmp");
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        {
            let mut f = OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(&tmp)?;
            for r in keep {
                if let Ok(line) = serde_json::to_string(r) {
                    f.write_all(line.as_bytes())?;
                    f.write_all(b"\n")?;
                }
            }
            f.flush()?;
            f.sync_all()?;
        }
        std::fs::rename(&tmp, &self.path)
    }
}

/// Result of reading the journal.
#[derive(Debug, Default, Clone)]
pub struct LoadedJournal {
    /// Well-formed records, in file (append) order.
    pub records: Vec<WonderRecord>,
    /// Lines that could not be parsed or carried an unknown version.
    /// Surfaced rather than swallowed so a persistently-growing skip
    /// count is visible instead of looking like an empty queue.
    pub skipped: usize,
}

impl LoadedJournal {
    /// Distinct signals, deduplicated by `(seed, signal_type)` keeping
    /// the HIGHEST intensity seen, sorted intensity-descending.
    ///
    /// The same gap is re-detected on every query that touches it, so a
    /// busy day produces thousands of near-duplicate records for a
    /// handful of real gaps. Deduping here is what makes the journal a
    /// queue of DISTINCT open questions rather than a query log.
    pub fn dedup_signals(&self) -> Vec<CuriositySignal> {
        let mut best: HashMap<(Seed, u8), CuriositySignal> = HashMap::new();
        for r in &self.records {
            let key = (r.signal.seed, signal_type_tag(&r.signal.signal_type));
            let replace = match best.get(&key) {
                Some(existing) => r.signal.intensity > existing.intensity,
                None => true,
            };
            if replace {
                best.insert(key, r.signal.clone());
            }
        }
        let mut out: Vec<CuriositySignal> = best.into_values().collect();
        out.sort_by(|a, b| {
            b.intensity
                .partial_cmp(&a.intensity)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.seed.as_bytes().cmp(b.seed.as_bytes()))
        });
        out
    }

    /// `seed -> label` for every record that carried one. Feeds the
    /// consumer's `gap_label_map`, which is how a synthetic gap seed
    /// with no node in the graph still gets a human-readable name.
    pub fn labels(&self) -> HashMap<Seed, String> {
        let mut m = HashMap::new();
        for r in &self.records {
            if let Some(l) = &r.label {
                if !l.is_empty() {
                    m.insert(r.signal.seed, l.clone());
                }
            }
        }
        m
    }
}

/// Stable small tag for a signal type, for use as a dedupe key
/// component. Deliberately not `as u8` on the enum: this must not
/// silently change if variants are reordered.
fn signal_type_tag(t: &crate::curiosity::SignalType) -> u8 {
    match t {
        crate::curiosity::SignalType::DeadEnd => 1,
        crate::curiosity::SignalType::IsolatedNode => 2,
        crate::curiosity::SignalType::KnowledgeDesert => 3,
        crate::curiosity::SignalType::HubBoundary => 4,
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::curiosity::SignalType;
    use crate::scope::TemporalScope;

    fn sig(seed_byte: u8, t: SignalType, intensity: f32) -> CuriositySignal {
        let mut b = [0u8; 32];
        b[0] = seed_byte;
        CuriositySignal {
            seed: Seed::from_bytes(b),
            signal_type: t,
            intensity,
            scope: TemporalScope::Current,
            timestamp_ms: 1_000,
        }
    }

    fn tmpdir(tag: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!(
            "lu_wonder_sidecar_{}_{}",
            tag,
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).expect("tempdir");
        p
    }

    #[test]
    fn append_then_load_roundtrips() {
        let dir = tmpdir("roundtrip");
        let sc = WonderSidecar::in_dir(&dir).expect("sidecar");
        let recs = vec![
            WonderRecord::new(sig(1, SignalType::HubBoundary, 0.9), Some("dog".into()), None),
            WonderRecord::new(sig(2, SignalType::IsolatedNode, 0.4), None, None),
        ];
        assert_eq!(sc.append(&recs).expect("append"), 2);
        let loaded = sc.load().expect("load");
        assert_eq!(loaded.records.len(), 2);
        assert_eq!(loaded.skipped, 0);
        assert_eq!(loaded.labels().len(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn append_is_additive_across_calls() {
        let dir = tmpdir("additive");
        let sc = WonderSidecar::in_dir(&dir).expect("sidecar");
        for i in 0..5u8 {
            sc.append(&[WonderRecord::new(sig(i, SignalType::DeadEnd, 0.5), None, None)])
                .expect("append");
        }
        assert_eq!(sc.load().expect("load").records.len(), 5);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn missing_journal_is_empty_not_an_error() {
        let dir = tmpdir("missing");
        let sc = WonderSidecar::new(dir.join("nope.jsonl"));
        let loaded = sc.load().expect("load must not error on a missing file");
        assert!(loaded.records.is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn torn_tail_costs_one_record_not_the_file() {
        let dir = tmpdir("torn");
        let sc = WonderSidecar::in_dir(&dir).expect("sidecar");
        sc.append(&[WonderRecord::new(sig(1, SignalType::HubBoundary, 0.9), None, None)])
            .expect("append");
        // Simulate a power loss mid-append.
        let mut f = OpenOptions::new().append(true).open(sc.path()).expect("open");
        f.write_all(b"{\"v\":1,\"appended").expect("partial write");
        drop(f);
        let loaded = sc.load().expect("load");
        assert_eq!(loaded.records.len(), 1, "the intact record must survive");
        assert_eq!(loaded.skipped, 1, "the torn line must be counted, not hidden");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn dedup_keeps_highest_intensity_per_seed_and_type() {
        let dir = tmpdir("dedup");
        let sc = WonderSidecar::in_dir(&dir).expect("sidecar");
        sc.append(&[
            WonderRecord::new(sig(7, SignalType::HubBoundary, 0.3), None, None),
            WonderRecord::new(sig(7, SignalType::HubBoundary, 0.8), None, None),
            WonderRecord::new(sig(7, SignalType::DeadEnd, 0.5), None, None),
        ])
        .expect("append");
        let d = sc.load().expect("load").dedup_signals();
        assert_eq!(d.len(), 2, "same seed, two distinct signal types");
        assert!(
            (d[0].intensity - 0.8).abs() < 1e-6,
            "highest intensity must win and sort first, got {:?}",
            d[0].intensity
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn rewrite_replaces_atomically() {
        let dir = tmpdir("rewrite");
        let sc = WonderSidecar::in_dir(&dir).expect("sidecar");
        sc.append(&[
            WonderRecord::new(sig(1, SignalType::DeadEnd, 0.5), None, None),
            WonderRecord::new(sig(2, SignalType::DeadEnd, 0.5), None, None),
        ])
        .expect("append");
        let keep: Vec<WonderRecord> =
            sc.load().expect("load").records.into_iter().take(1).collect();
        sc.rewrite(&keep).expect("rewrite");
        assert_eq!(sc.load().expect("load").records.len(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
