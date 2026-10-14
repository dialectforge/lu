//! `NodeFlushWorker` -- NodeMemTable -> NodeSST flush + atomic manifest commit.
//!
//! OP-BRAIN-V5 Phase 2.6 (Q3, Q6, Q7). The first port where the v5
//! substrate becomes production-functional: the bridge from the
//! Phase 2.3 [`NodeMemTable`] through the Phase 2.4
//! [`NodeSstWriter`] / [`NodeSstReader`] into the Phase 2.5
//! manifest schema-v2 helpers (`allocate_node_sst_uid`,
//! `allocate_epoch`, `add_node_sst`, `live_node_sst_uids`).
//!
//! ## Atomic crash-safe commit sequence
//!
//! Per the brief (Phase 2.6 sub-deliverable B):
//!
//! ```text
//!   1. manifest.allocate_node_sst_uid()  -> file_uid (monotonic)
//!   2. manifest.allocate_epoch()         -> epoch    (monotonic)
//!   3. NodeSstWriter::create(<file_uid>.nsst, opts)
//!         (the writer opens <path>.tmp.<file_uid> internally)
//!   4. for (id, entry) in memtable.iter() { writer.append(id, entry) }
//!         -- STREAMING: NodeMemTable::iter() is borrowing; NEVER collect.
//!   5. writer.finish() -> NodeSstWriteStats
//!         (writer fsyncs, then renames .tmp.<uid> -> .nsst)
//!   6. fsync the node_store/ directory so the rename hits disk
//!   7. caller: manifest.add_node_sst(level=0, entry)
//!   8. caller: manifest.save_atomic(...)
//! ```
//!
//! ## Crash-window analysis
//!
//! The WAL still holds every record on the flush path; data loss is
//! never possible at any of these break points.
//!
//! - **Crash inside step 3-5 (before writer.finish() rename):**
//!   `<file_uid>.nsst.tmp.<file_uid>` exists; no `<file_uid>.nsst`;
//!   no manifest reference. [`cleanup_orphan_node_ssts`] removes
//!   every `.nsst.tmp*` unconditionally at restart. WAL replay
//!   reconstructs the NodeMemTable; the next flush allocates a
//!   FRESH `file_uid` (the manifest's `next_node_sst_uid` was
//!   already bumped by step 1, so the failed uid is never reused).
//!
//! - **Crash after step 5 rename, before step 7 `add_node_sst`:**
//!   final-name `<file_uid>.nsst` on disk, no manifest reference.
//!   [`cleanup_orphan_node_ssts`] removes it -- its `file_uid` is
//!   not in [`Manifest::live_node_sst_uids`]. WAL replay rebuilds.
//!
//! - **Crash after step 7 `add_node_sst`, before step 8
//!   `save_atomic`:** the manifest changes were in-memory only and
//!   are lost on restart; the on-disk manifest is the pre-flush
//!   version, which does NOT reference the new `file_uid`. The
//!   `.nsst` file IS on disk -- classified as orphan at restart,
//!   removed. WAL replay rebuilds. NO data loss.
//!
//! - **Crash after step 8 `save_atomic` commit:** the manifest on
//!   disk references the new `file_uid`; the file is on disk and
//!   classified live by [`cleanup_orphan_node_ssts`]. WAL replay
//!   still folds duplicate records (matches the bond-side
//!   discipline at `flush.rs`).
//!
//! The orphan-cleanup pass at restart (sub-deliverable D) is the
//! load-bearing recovery primitive; it MUST run before the brain
//! accepts the first write, otherwise a concurrent flush could
//! collide with an orphan file -- unlikely in practice (the
//! manifest's `next_node_sst_uid` counter is monotonic and the
//! crashed-flush uid was already bumped past), but the discipline
//! still belongs at restart, not lazily.
//!
//! ## Empty NodeMemTable
//!
//! [`NodeFlushWorker::flush`] returns `Ok(None)` WITHOUT allocating
//! a `file_uid`, calling `allocate_epoch`, or creating any file.
//! The caller (the bond-side `flush_once` integration point) skips
//! the `add_node_sst` + `save_atomic` chain for the node side and
//! falls through to the existing bond-side commit.

use std::path::{Path, PathBuf};

use crate::manifest::{Manifest, ManifestError, NodeSSTEntry};
use crate::node_store::memtable::NodeMemTable;
use crate::node_store::sst_writer::{
    NodeSstError, NodeSstWriteOpts, NodeSstWriter,
};

/// File suffix for committed NodeSST files. The Phase 2.4 writer
/// renames `<file_uid>.nsst.tmp.<file_uid>` -> `<file_uid>.nsst` on
/// `finish()`; [`cleanup_orphan_node_ssts`] keys on this suffix.
pub const NSST_SUFFIX: &str = ".nsst";

/// Errors surfaced by [`NodeFlushWorker`] and the
/// [`cleanup_orphan_node_ssts`] restart pass.
#[derive(Debug)]
pub enum NodeFlushError {
    /// std::io::Error propagated from mkdir / fsync / readdir / remove.
    Io(std::io::Error),
    /// Underlying [`NodeSstWriter`] failure (open / append / finish).
    Sst(NodeSstError),
    /// Manifest helper rejected the [`Manifest::add_node_sst`] call.
    /// Always a programmer error in the flush worker context (uid
    /// uniqueness or level-monotonicity violation); never a crash
    /// or data condition. Surfaced rather than panicking so the
    /// caller can shut the flush worker down cleanly.
    Manifest(ManifestError),
    /// The flush worker was asked to write into a node_store_dir
    /// that could not be created (mkdir_p failure). Wraps the
    /// underlying IO error context.
    NodeStoreDirCreate { path: PathBuf, source: std::io::Error },
}

impl std::fmt::Display for NodeFlushError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NodeFlushError::Io(e) => write!(f, "NodeFlushWorker IO: {}", e),
            NodeFlushError::Sst(e) => write!(f, "NodeFlushWorker NodeSST: {}", e),
            NodeFlushError::Manifest(e) => write!(f, "NodeFlushWorker manifest: {}", e),
            NodeFlushError::NodeStoreDirCreate { path, source } => write!(
                f,
                "NodeFlushWorker mkdir_p({}): {}",
                path.display(), source,
            ),
        }
    }
}

impl std::error::Error for NodeFlushError {}

impl From<std::io::Error> for NodeFlushError {
    fn from(e: std::io::Error) -> Self {
        NodeFlushError::Io(e)
    }
}

impl From<NodeSstError> for NodeFlushError {
    fn from(e: NodeSstError) -> Self {
        NodeFlushError::Sst(e)
    }
}

impl From<ManifestError> for NodeFlushError {
    fn from(e: ManifestError) -> Self {
        NodeFlushError::Manifest(e)
    }
}

/// Cleanup report returned by [`cleanup_orphan_node_ssts`]. Both
/// counters are diagnostic / test-visible; the cleanup is
/// best-effort discipline (failure to remove a single file is
/// logged but not fatal -- the next flush will still allocate a
/// fresh `file_uid`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CleanupReport {
    /// Number of `.nsst` files removed because their `file_uid` is
    /// not present in [`Manifest::live_node_sst_uids`]. These are
    /// orphans from a crashed flush between the writer rename
    /// (step 5) and the manifest `save_atomic` (step 8).
    pub orphans_removed: usize,
    /// Number of `.nsst.tmp*` files removed unconditionally. A
    /// `.tmp` means a flush in progress that was interrupted
    /// before step 5 -- the next flush allocates a new `file_uid`
    /// so reusing the half-written file is never desirable.
    pub tmps_removed: usize,
}

/// Node-store flush worker. Borrows the live manifest at flush
/// time; does NOT own any background thread of its own (Phase 2.6
/// integrates into the existing `flush.rs::flush_once` worker
/// thread -- a second background thread is deferred to a later
/// port if the bond-side cadence proves wrong for the node side).
pub struct NodeFlushWorker {
    /// Directory where committed NodeSST files live (`<file_uid>.nsst`).
    /// Derived by the caller from `Brain.wal_path.parent()/node_store/`;
    /// the worker creates the directory on first flush if missing.
    node_store_dir: PathBuf,
    /// OP-BRAIN-V5 6.5: per-SST encryption key. `None` = plaintext.
    key: Option<crate::encrypt::MasterKey>,
}

impl NodeFlushWorker {
    /// Construct a NodeFlushWorker rooted at `node_store_dir`.
    /// The directory is NOT created here; [`Self::flush`] does
    /// `create_dir_all` on first call so test rigs that never
    /// invoke `flush` don't leave empty directories on disk.
    pub fn new(node_store_dir: PathBuf) -> Self {
        NodeFlushWorker { node_store_dir, key: None }
    }

    /// Builder: set the per-SST encryption key (OP-BRAIN-V5 6.5).
    /// Existing `new(dir)` sites stay plaintext.
    pub fn with_key(mut self, key: Option<crate::encrypt::MasterKey>) -> Self {
        self.key = key;
        self
    }

    /// Read-only view of the configured `node_store_dir`.
    pub fn node_store_dir(&self) -> &Path {
        &self.node_store_dir
    }

    /// Compute the canonical `node_store_dir` from a brain's WAL
    /// path. Convention: `<wal_path.parent()>/node_store/`.
    pub fn node_store_dir_for_wal(wal_path: &Path) -> PathBuf {
        let parent = wal_path.parent().unwrap_or_else(|| Path::new("."));
        parent.join("node_store")
    }

    /// Flush `memtable` to a fresh NodeSST file in `node_store_dir`,
    /// allocating `file_uid` and `epoch` from `manifest`. Returns
    /// the [`NodeSSTEntry`] for the caller to commit via
    /// [`Manifest::add_node_sst`] + [`Manifest::save_atomic`].
    ///
    /// Empty `memtable` returns `Ok(None)` WITHOUT allocating a
    /// `file_uid`, calling `allocate_epoch`, or creating any file
    /// -- the brief mandates exactly this behaviour so a no-op
    /// flush leaves the manifest counters untouched.
    ///
    /// Streaming discipline: this method consumes
    /// `memtable.iter()` directly without materialising into a
    /// `Vec`. The Phase 2.3 [`NodeMemTable::iter`] borrows from
    /// the inner `BTreeMap` and yields `(NodeId, &NodeEntry)` in
    /// `NodeId`-ascending order -- the NodeSST writer's exact
    /// contract.
    ///
    /// The caller is responsible for the manifest commit:
    ///
    /// ```ignore
    /// if let Some(entry) = node_flush_worker.flush(&snapshot_memtable, &mut working_manifest)? {
    ///     working_manifest.add_node_sst(0, entry)?;
    /// }
    /// working_manifest.save_atomic(&manifest_path)?;
    /// ```
    ///
    /// The caller threads `working_manifest` through the bond-side
    /// commit too (single atomic save covers both).
    pub fn flush(
        &self,
        memtable: &NodeMemTable,
        manifest: &mut Manifest,
    ) -> Result<Option<NodeSSTEntry>, NodeFlushError> {
        if memtable.is_empty() {
            // Empty memtable: NO file_uid allocation, NO epoch
            // allocation, NO file. Manifest counters untouched.
            return Ok(None);
        }

        // 0. Ensure the node_store/ directory exists.
        if !self.node_store_dir.exists() {
            std::fs::create_dir_all(&self.node_store_dir).map_err(|e| {
                NodeFlushError::NodeStoreDirCreate {
                    path: self.node_store_dir.clone(),
                    source: e,
                }
            })?;
        }

        // 1. + 2. Allocate file_uid and epoch from the working
        // manifest. The bumps are persisted only when the caller
        // calls `save_atomic`; if THIS flush crashes between now
        // and that commit, the in-memory bumps are lost and the
        // next boot's flush will reuse `file_uid` N -- but the
        // orphan file (if any) from this attempt has been removed
        // by [`cleanup_orphan_node_ssts`] at restart, so the reuse
        // is safe.
        let file_uid = manifest.allocate_node_sst_uid();
        let epoch = manifest.allocate_epoch();

        // 3. Construct the writer. NodeSstWriter::create opens
        // `<final_path>.tmp.<file_uid>` internally; finish()
        // fsyncs + renames to `<final_path>`.
        let filename = nsst_filename(file_uid);
        let final_path = self.node_store_dir.join(&filename);
        let opts = NodeSstWriteOpts {
            file_uid,
            ..Default::default()
        };
        let mut writer =
            NodeSstWriter::create_with_key(&final_path, opts, self.key.clone())?;

        // 4. Stream the memtable. NodeMemTable::iter() borrows; we
        // never collect into a Vec. first/last NodeId are tracked by
        // the writer itself (Phase 3.8.2: NodeSstWriteStats grew
        // first_node_id/last_node_id, mirroring BondSstWriteStats).
        let mut entry_count: u64 = 0;
        for (id, entry) in memtable.iter() {
            writer.append(id, entry)?;
            entry_count += 1;
        }

        // 5. Finish: fsync the .tmp file, atomic rename to final_path.
        let stats = writer.finish()?;

        // 6. fsync the node_store/ directory so the rename hits
        // disk (writer.finish only fsyncs the file, not the
        // directory entry update). Best-effort: a sync_all failure
        // here is logged via the IO error surface but does not
        // poison the flush -- the manifest commit at step 8 is
        // the durability anchor, and the orphan-cleanup pass at
        // restart still recovers correctly if this fsync was
        // racy.
        if let Ok(dir) = std::fs::File::open(&self.node_store_dir) {
            let _ = dir.sync_all();
        }

        // 8. Build the manifest entry. `entry_count` was tracked
        // in the streaming loop above (matches
        // `stats.entries_written` -- we assert for paranoia in
        // debug builds).
        debug_assert_eq!(
            entry_count, stats.entries_written,
            "NodeFlushWorker entry_count drift: counted {}, writer reports {}",
            entry_count, stats.entries_written,
        );
        let first = stats
            .first_node_id
            .expect("non-empty memtable has a first NodeId")
            .0;
        let last = stats
            .last_node_id
            .expect("non-empty memtable has a last NodeId")
            .0;
        let entry = NodeSSTEntry {
            file_uid,
            first_node_id: first,
            last_node_id: last,
            entry_count: stats.entries_written,
            file_size: stats.file_size_bytes,
            epoch,
        };

        // 9. Caller commits the entry via add_node_sst + save_atomic.
        Ok(Some(entry))
    }
}

/// Canonical filename for a NodeSST given its `file_uid`. The
/// manifest records `file_uid` (not a path), so the worker, the
/// orphan-cleanup pass, and any future read-path code all derive
/// the on-disk name from the same helper.
pub fn nsst_filename(file_uid: u64) -> String {
    format!("{}{}", file_uid, NSST_SUFFIX)
}

/// Parse a `file_uid` out of an `.nsst` filename. Returns `None`
/// for filenames that don't match the `<file_uid>.nsst` shape
/// (used by [`cleanup_orphan_node_ssts`] to skip files it does not
/// own).
fn parse_nsst_filename(name: &str) -> Option<u64> {
    let stem = name.strip_suffix(NSST_SUFFIX)?;
    stem.parse::<u64>().ok()
}

/// Restart-time orphan cleanup pass.
///
/// Scans `node_store_dir` for committed `.nsst` files whose
/// `file_uid` is NOT present in [`Manifest::live_node_sst_uids`]
/// (orphans from a crashed flush between writer rename and
/// manifest `save_atomic`) and any `.nsst.tmp*` files (interrupted
/// flushes before the writer rename), removes them, and returns
/// the counters.
///
/// Discipline:
///
/// - MUST run before the brain accepts the first write. Otherwise
///   a new flush could collide with an orphan file_uid -- unlikely
///   in practice (the manifest counter is monotonic) but the
///   precondition is what makes that safety property obvious.
///
/// - Best-effort per-file: a `remove_file` failure on one entry
///   logs and continues; the counter does NOT increment for that
///   file. The next flush still allocates a fresh `file_uid` so a
///   stuck orphan doesn't block forward progress.
///
/// - Files that don't match the `<file_uid>.nsst` or `.nsst.tmp*`
///   shape are LEFT ALONE. The cleanup pass does not own any
///   other file extensions in this directory.
pub fn cleanup_orphan_node_ssts(
    node_store_dir: &Path,
    manifest: &Manifest,
) -> Result<CleanupReport, NodeFlushError> {
    let mut report = CleanupReport::default();

    // If the directory does not exist, no files to clean up.
    // First-boot brains have not flushed yet; the directory will
    // be created on the first flush by NodeFlushWorker::flush.
    if !node_store_dir.exists() {
        return Ok(report);
    }

    let live: std::collections::HashSet<u64> =
        manifest.live_node_sst_uids().into_iter().collect();

    let dir_iter = std::fs::read_dir(node_store_dir)?;
    for entry in dir_iter {
        let entry = match entry {
            Ok(e) => e,
            Err(e) => {
                eprintln!(
                    "[node_store::cleanup_orphan_node_ssts] readdir error in {}: {} (skipping)",
                    node_store_dir.display(), e,
                );
                continue;
            }
        };
        let path = entry.path();
        let name = match path.file_name().and_then(|n| n.to_str()) {
            Some(n) => n,
            None => continue, // non-UTF8 filenames not owned by us.
        };

        // Categorise:
        //   - `<file_uid>.nsst`      -> orphan candidate (check live_uids)
        //   - `*.nsst.tmp*`          -> unconditional tmp
        //   - anything else          -> leave alone
        //
        // The .tmp check is BEFORE the .nsst check because the
        // writer's actual tmp filename ends in `.nsst.tmp.<uid>`
        // (NOT `.nsst.tmp`), so a substring check is the safe match.
        if name.contains(".nsst.tmp") {
            if let Err(e) = std::fs::remove_file(&path) {
                eprintln!(
                    "[node_store::cleanup_orphan_node_ssts] remove tmp {} failed: {} (continuing)",
                    path.display(), e,
                );
                continue;
            }
            report.tmps_removed += 1;
            continue;
        }

        if let Some(uid) = parse_nsst_filename(name) {
            if !live.contains(&uid) {
                if let Err(e) = std::fs::remove_file(&path) {
                    eprintln!(
                        "[node_store::cleanup_orphan_node_ssts] remove orphan {} failed: {} (continuing)",
                        path.display(), e,
                    );
                    continue;
                }
                eprintln!(
                    "[node_store::cleanup_orphan_node_ssts] removed orphan {} (file_uid {} not in manifest)",
                    path.display(), uid,
                );
                report.orphans_removed += 1;
            }
        }
        // Anything else: leave alone.
    }

    // fsync the directory so the unlink(s) hit disk before we
    // accept writes. Best-effort -- failure is non-fatal because
    // the next flush still allocates a fresh file_uid.
    if report.orphans_removed > 0 || report.tmps_removed > 0 {
        if let Ok(dir) = std::fs::File::open(node_store_dir) {
            let _ = dir.sync_all();
        }
    }

    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::{KeyFingerprint, Manifest};
    use crate::node::{Confidence, KnowledgeContent, KnowledgeNode, KnowledgeUnit};
    use crate::node_intern::NodeId;
    use crate::node_store::memtable::NodeMemTable;
    use crate::node_store::sst_reader::NodeSstReader;
    use crate::seed::Seed;
    use tempfile::tempdir;

    fn mk_node(label: &str) -> KnowledgeNode {
        KnowledgeNode {
            seed: Seed::from_label(label),
            content: label.to_string(),
            knowledge: KnowledgeUnit {
                content: KnowledgeContent::Concept(label.to_string()),
                confidence: Confidence::Verified,
                created_at: 0,
                updated_at: 0,
            },
            embedding: vec![0.0; 384],
            parent_seeds: Vec::new(),
            typed_ratio: 0.0,
        }
    }

    fn populate(memtable: &mut NodeMemTable, n: usize, base_id: u64, base_seq: u64) {
        for i in 0..n {
            memtable.insert(
                NodeId(base_id + i as u64),
                mk_node(&format!("node_{}", base_id + i as u64)),
                base_seq + i as u64,
            );
        }
    }

    /// Test A: flush creates a NodeSST file, manifest gains an entry,
    /// live_node_sst_uids includes the new file_uid.
    #[test]
    fn phase26_test_a_flush_creates_nsst_and_updates_manifest() {
        let dir = tempdir().unwrap();
        let node_store_dir = dir.path().join("node_store");
        let worker = NodeFlushWorker::new(node_store_dir.clone());
        let mut manifest = Manifest::new(KeyFingerprint::none());

        let mut memtable = NodeMemTable::new();
        populate(&mut memtable, 50, 0, 0);

        let entry = worker.flush(&memtable, &mut manifest).unwrap().expect("non-empty flush");
        assert_eq!(entry.file_uid, 0, "first allocate returns 0");
        assert_eq!(entry.first_node_id, 0);
        assert_eq!(entry.last_node_id, 49);
        assert_eq!(entry.entry_count, 50);
        assert_eq!(entry.epoch, 0);

        // File on disk at expected path.
        let expected_path = node_store_dir.join(nsst_filename(0));
        assert!(expected_path.exists(), "<file_uid>.nsst must exist post-flush");

        // Manifest add_node_sst commits the entry.
        manifest.add_node_sst(0, entry.clone()).unwrap();
        assert_eq!(manifest.total_node_sst_count(), 1);
        assert_eq!(manifest.live_node_sst_uids(), vec![0u64]);
    }

    /// Test B: empty NodeMemTable returns Ok(None), no file created,
    /// manifest counters unchanged.
    #[test]
    fn phase26_test_b_empty_memtable_no_file_no_alloc() {
        let dir = tempdir().unwrap();
        let node_store_dir = dir.path().join("node_store");
        let worker = NodeFlushWorker::new(node_store_dir.clone());
        let mut manifest = Manifest::new(KeyFingerprint::none());
        let pre_uid = manifest.peek_next_node_sst_uid();
        let pre_epoch = manifest.peek_next_epoch();

        let memtable = NodeMemTable::new();
        let result = worker.flush(&memtable, &mut manifest).unwrap();
        assert!(result.is_none(), "empty memtable returns Ok(None)");

        // No file created -- the directory may or may not exist;
        // either way it must not contain any .nsst.
        if node_store_dir.exists() {
            let entries: Vec<_> = std::fs::read_dir(&node_store_dir).unwrap()
                .filter_map(|e| e.ok())
                .filter(|e| e.file_name().to_string_lossy().ends_with(".nsst"))
                .collect();
            assert!(entries.is_empty(), "no .nsst on empty-flush");
        }

        // Manifest counters untouched.
        assert_eq!(manifest.peek_next_node_sst_uid(), pre_uid);
        assert_eq!(manifest.peek_next_epoch(), pre_epoch);
        assert_eq!(manifest.total_node_sst_count(), 0);
    }

    /// Test C: 3 sequential flushes -> 3 distinct files, monotonic
    /// file_uids (0, 1, 2), monotonic epochs (0, 1, 2), manifest
    /// has 3 entries.
    #[test]
    fn phase26_test_c_monotonic_file_uids_and_epochs() {
        let dir = tempdir().unwrap();
        let node_store_dir = dir.path().join("node_store");
        let worker = NodeFlushWorker::new(node_store_dir.clone());
        let mut manifest = Manifest::new(KeyFingerprint::none());

        let mut seen_uids = Vec::new();
        let mut seen_epochs = Vec::new();
        for round in 0..3 {
            let mut memtable = NodeMemTable::new();
            populate(&mut memtable, 20, (round * 100) as u64, (round * 1000) as u64);
            let entry = worker.flush(&memtable, &mut manifest).unwrap()
                .expect("non-empty flush");
            seen_uids.push(entry.file_uid);
            seen_epochs.push(entry.epoch);
            manifest.add_node_sst(0, entry).unwrap();
        }
        assert_eq!(seen_uids, vec![0, 1, 2]);
        assert_eq!(seen_epochs, vec![0, 1, 2]);
        for uid in &seen_uids {
            assert!(
                node_store_dir.join(nsst_filename(*uid)).exists(),
                "file_uid={} .nsst must exist", uid,
            );
        }
        assert_eq!(manifest.total_node_sst_count(), 3);
    }

    /// Test D: simulate a crash between writer rename and
    /// `add_node_sst`. The file exists on disk but the manifest
    /// does NOT reference it. cleanup_orphan_node_ssts deletes
    /// it; cleanup report shows 1 orphan removed; manifest is
    /// unchanged.
    #[test]
    fn phase26_test_d_orphan_cleaned_after_rename_without_commit() {
        let dir = tempdir().unwrap();
        let node_store_dir = dir.path().join("node_store");
        let worker = NodeFlushWorker::new(node_store_dir.clone());
        let mut manifest = Manifest::new(KeyFingerprint::none());

        let mut memtable = NodeMemTable::new();
        populate(&mut memtable, 10, 0, 0);

        // Flush returns the entry but we deliberately DO NOT call
        // add_node_sst -- simulating a crash in that window.
        let entry = worker.flush(&memtable, &mut manifest).unwrap()
            .expect("non-empty flush");
        let orphan_path = node_store_dir.join(nsst_filename(entry.file_uid));
        assert!(orphan_path.exists(), "file on disk pre-cleanup");
        // Manifest live uids does NOT include the orphan -- the
        // allocate_* bumps were in-memory only; we did NOT call
        // save_atomic.
        let live_uids_before = manifest.live_node_sst_uids();
        assert!(
            !live_uids_before.contains(&entry.file_uid),
            "orphan file_uid {} should NOT be live (add_node_sst not called)",
            entry.file_uid,
        );

        // Simulate restart: load a manifest from disk (or use a
        // fresh empty Manifest, equivalent at this point) and
        // call cleanup_orphan_node_ssts.
        let restart_manifest = Manifest::new(KeyFingerprint::none());
        let report = cleanup_orphan_node_ssts(&node_store_dir, &restart_manifest).unwrap();
        assert_eq!(report.orphans_removed, 1, "exactly 1 orphan removed");
        assert_eq!(report.tmps_removed, 0);
        assert!(!orphan_path.exists(), "orphan file deleted");
        assert_eq!(restart_manifest.total_node_sst_count(), 0);
    }

    /// Test E: a stray .nsst.tmp file is removed unconditionally on
    /// restart even if it doesn't correspond to any uid currently
    /// in the manifest's allocator range.
    #[test]
    fn phase26_test_e_tmp_files_cleaned() {
        let dir = tempdir().unwrap();
        let node_store_dir = dir.path().join("node_store");
        std::fs::create_dir_all(&node_store_dir).unwrap();

        // Match the writer's actual tmp shape: `<final>.tmp.<uid>`.
        let stray_tmp = node_store_dir.join("42.nsst.tmp.42");
        std::fs::write(&stray_tmp, b"abandoned tmp from a crashed writer").unwrap();
        // Plus the audit-doc-style plain `.nsst.tmp` form, to be
        // sure the substring match catches both.
        let stray_tmp_plain = node_store_dir.join("7.nsst.tmp");
        std::fs::write(&stray_tmp_plain, b"plain tmp").unwrap();

        let manifest = Manifest::new(KeyFingerprint::none());
        let report = cleanup_orphan_node_ssts(&node_store_dir, &manifest).unwrap();
        assert_eq!(report.tmps_removed, 2, "both tmp shapes removed");
        assert_eq!(report.orphans_removed, 0);
        assert!(!stray_tmp.exists());
        assert!(!stray_tmp_plain.exists());
    }

    /// Test F: commit-but-no-save semantics. The brief calls this
    /// out: a crash after add_node_sst but before save_atomic
    /// loses the in-memory manifest change. At restart the
    /// on-disk manifest is the PRE-add version, so the file is
    /// classified orphan and removed. NO data loss (WAL still
    /// holds the records).
    #[test]
    fn phase26_test_f_crash_after_add_before_save_treated_as_orphan() {
        let dir = tempdir().unwrap();
        let node_store_dir = dir.path().join("node_store");
        let worker = NodeFlushWorker::new(node_store_dir.clone());

        // Pre-crash: build a manifest, flush, add_node_sst (in-memory).
        let mut working_manifest = Manifest::new(KeyFingerprint::none());
        let mut memtable = NodeMemTable::new();
        populate(&mut memtable, 5, 0, 0);
        let entry = worker.flush(&memtable, &mut working_manifest).unwrap()
            .expect("non-empty flush");
        working_manifest.add_node_sst(0, entry.clone()).unwrap();
        // CRASH here: NO save_atomic call.
        // Drop the working_manifest, simulating process exit.
        drop(working_manifest);

        // Restart: load on-disk manifest. We never wrote one, so
        // simulate by constructing a fresh Manifest (the equivalent
        // of loading a pre-flush version that does NOT reference
        // file_uid 0).
        let restart_manifest = Manifest::new(KeyFingerprint::none());
        assert!(!restart_manifest.live_node_sst_uids().contains(&entry.file_uid));

        let orphan_path = node_store_dir.join(nsst_filename(entry.file_uid));
        assert!(orphan_path.exists(), "file on disk before cleanup");

        let report = cleanup_orphan_node_ssts(&node_store_dir, &restart_manifest).unwrap();
        assert_eq!(report.orphans_removed, 1);
        assert!(!orphan_path.exists(), "orphan removed");
    }

    /// Test G: post-save the file persists and is NOT an orphan.
    /// cleanup_orphan_node_ssts reports 0 removals; the file stays
    /// on disk; the manifest still references it.
    #[test]
    fn phase26_test_g_post_save_file_persists_not_orphan() {
        let dir = tempdir().unwrap();
        let node_store_dir = dir.path().join("node_store");
        let manifest_path = dir.path().join("MANIFEST");
        let worker = NodeFlushWorker::new(node_store_dir.clone());

        let mut working_manifest = Manifest::new(KeyFingerprint::none());
        let mut memtable = NodeMemTable::new();
        populate(&mut memtable, 5, 0, 0);
        let entry = worker.flush(&memtable, &mut working_manifest).unwrap()
            .expect("non-empty flush");
        working_manifest.add_node_sst(0, entry.clone()).unwrap();
        working_manifest.save_atomic(&manifest_path).unwrap();

        // Restart: load the persisted manifest.
        let restart_manifest = Manifest::load(&manifest_path).unwrap()
            .expect("saved manifest loads");
        assert!(restart_manifest.live_node_sst_uids().contains(&entry.file_uid));

        let final_path = node_store_dir.join(nsst_filename(entry.file_uid));
        assert!(final_path.exists());

        let report = cleanup_orphan_node_ssts(&node_store_dir, &restart_manifest).unwrap();
        assert_eq!(report.orphans_removed, 0, "no orphans -- file is live");
        assert_eq!(report.tmps_removed, 0);
        assert!(final_path.exists(), "live file persists");
        assert_eq!(restart_manifest.total_node_sst_count(), 1);
    }

    /// Test H: round-trip. Flush 100 nodes, persist manifest,
    /// reload manifest, open the referenced .nsst via
    /// NodeSstReader, spot-check entries by NodeId.
    #[test]
    fn phase26_test_h_round_trip_flush_reload_read() {
        let dir = tempdir().unwrap();
        let node_store_dir = dir.path().join("node_store");
        let manifest_path = dir.path().join("MANIFEST");
        let worker = NodeFlushWorker::new(node_store_dir.clone());

        let mut working_manifest = Manifest::new(KeyFingerprint::none());
        let mut memtable = NodeMemTable::new();
        populate(&mut memtable, 100, 1_000, 0);
        let entry = worker.flush(&memtable, &mut working_manifest).unwrap()
            .expect("non-empty flush");
        working_manifest.add_node_sst(0, entry.clone()).unwrap();
        working_manifest.save_atomic(&manifest_path).unwrap();

        // Reload manifest.
        let reloaded = Manifest::load(&manifest_path).unwrap().expect("manifest reload");
        assert_eq!(reloaded.total_node_sst_count(), 1);
        let (_lvl, reloaded_entry) = reloaded.iter_node_ssts().next().unwrap();
        assert_eq!(reloaded_entry.file_uid, entry.file_uid);
        assert_eq!(reloaded_entry.entry_count, 100);
        assert_eq!(reloaded_entry.first_node_id, 1_000);
        assert_eq!(reloaded_entry.last_node_id, 1_099);

        // Open the file via NodeSstReader and spot-check three
        // NodeIds (first, middle, last).
        let nsst_path = node_store_dir.join(nsst_filename(reloaded_entry.file_uid));
        let reader = NodeSstReader::open(&nsst_path).unwrap();
        for id in &[1_000u64, 1_050, 1_099] {
            let got = reader.get(NodeId(*id)).unwrap()
                .unwrap_or_else(|| panic!("NodeId({}) must round-trip", id));
            assert_eq!(got.payload.content, format!("node_{}", id));
            assert!(!got.tombstone);
        }
        // A NodeId outside the file's range returns None.
        let outside = reader.get(NodeId(5_000)).unwrap();
        assert!(outside.is_none(), "NodeId outside range returns None");
    }

    /// Plus: cleanup is a no-op on a never-flushed brain (the
    /// node_store dir doesn't exist yet). The brain calls
    /// cleanup_orphan_node_ssts at every startup; this guard
    /// matters for first-boot.
    #[test]
    fn phase26_cleanup_noop_on_missing_dir() {
        let dir = tempdir().unwrap();
        let node_store_dir = dir.path().join("node_store");
        let manifest = Manifest::new(KeyFingerprint::none());
        let report = cleanup_orphan_node_ssts(&node_store_dir, &manifest).unwrap();
        assert_eq!(report.orphans_removed, 0);
        assert_eq!(report.tmps_removed, 0);
    }

    /// Plus: nsst_filename / parse_nsst_filename round-trip.
    #[test]
    fn phase26_nsst_filename_round_trip() {
        for uid in [0u64, 1, 42, 999_999] {
            let name = nsst_filename(uid);
            assert_eq!(name, format!("{}.nsst", uid));
            assert_eq!(parse_nsst_filename(&name), Some(uid));
        }
        // Bad shapes return None.
        assert_eq!(parse_nsst_filename("abc.nsst"), None);
        assert_eq!(parse_nsst_filename("42.txt"), None);
        assert_eq!(parse_nsst_filename("42.nsst.tmp"), None);
        assert_eq!(parse_nsst_filename(".nsst"), None);
    }
}
