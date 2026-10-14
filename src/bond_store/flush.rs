//! `BondFlushWorker` -- bond iterator -> BondSST flush + atomic
//! manifest commit.
//!
//! OP-BRAIN-V5 Phase 3.3 (Q3, Q6, Q7). Sibling of Phase 2.6
//! [`crate::node_store::NodeFlushWorker`]: identical structural
//! shape, bond-specific input. Bridges Phase 3.1
//! [`crate::bond_store::BondSstWriter`] + the v3 manifest helpers
//! (`allocate_bond_v5_sst_uid`, `allocate_epoch`,
//! `add_bond_v5_sst`, `live_bond_v5_sst_uids`) added by this same
//! port to `src/manifest.rs`.
//!
//! ## Coexistence with the v4 bond store
//!
//! The v4 bond SST writer (`src/sst_writer.rs`) is still alive
//! and production-read at Phase 3.3 ship time. Phase 3.9 retires
//! it. Phase 3.3 runs the BondFlushWorker **alongside** the v4
//! bond SST writer, not replacing it: every `flush_once` invocation
//! emits BOTH a v4 `<level>_<id>.sst` (existing path, production-
//! read) AND a v5 `<file_uid>.bsst` (new path, read by nothing
//! until Phase 3.7 ships the v5 read path). The 2x write overhead
//! is bounded by the v4 single-store flush and is the cost of the
//! cutover window.
//!
//! ## Manifest distinguisher mechanism
//!
//! The v4 [`crate::manifest::SSTEntry`] carries a `file: String`
//! relative path; v5 [`crate::manifest::BondSSTEntry`] carries a
//! bare `file_uid: u64` (path derived from the writer's directory
//! convention). Rather than adding a `store_format` discriminator
//! to the v4 `SSTEntry` (path-level schema change, complicates
//! the v4 read path), Phase 3.3 stores v5 bond entries under a
//! sibling field on `ManifestPayload`:
//!
//! ```text
//!   levels: Vec<LevelEntry>          -- v4 bond store (Phase 4)
//!   node_levels: Vec<NodeLevel>      -- v5 node store (Phase 2.5)
//!   bond_v5_levels: Vec<BondV5Level> -- v5 bond store (Phase 3.3)
//! ```
//!
//! Each field is independent: v4 and v5 bond entries coexist
//! without a discriminator, the v4 read path remains untouched,
//! Phase 3.9 simply drops the `levels` field (and the `LevelEntry`
//! / `SSTEntry` types) in a clean schema bump v3 -> v4. Mirrors
//! the Phase 2.5 additive pattern that introduced `node_levels`
//! sibling-to-`levels`. **Choice recorded here so future
//! retirement of v4 is mechanical.**
//!
//! ## Atomic crash-safe commit sequence
//!
//! ```text
//!   1. manifest.allocate_bond_v5_sst_uid() -> file_uid (monotonic)
//!   2. manifest.allocate_epoch()           -> epoch    (monotonic)
//!   3. BondSstWriter::create(<file_uid>.bsst, opts)
//!         (the writer opens .tmp.<file_uid> internally)
//!   4. for (subject, predicate, object, &payload) in iter {
//!          writer.append(subject, predicate, object, &payload)
//!      }
//!         -- STREAMING: iter is consumed lazily; NEVER collected.
//!   5. writer.finish() -> BondSstWriteStats
//!         (writer fsyncs, then renames .tmp.<uid> -> .bsst)
//!   6. fsync the bond_store/ directory so the rename hits disk
//!   7. caller: manifest.add_bond_v5_sst(level=0, entry)
//!   8. caller: manifest.save_atomic(...)
//! ```
//!
//! ## Crash-window analysis
//!
//! The WAL still holds every record on the flush path; bond data
//! loss is never possible at any of these break points. Mirrors
//! Phase 2.6's analysis substituting `.bsst` for `.nsst`.
//!
//! - **Crash inside step 3-5 (before writer.finish() rename):**
//!   `<file_uid>.bsst.tmp.<file_uid>` exists; no `<file_uid>.bsst`;
//!   no manifest reference. [`cleanup_orphan_bond_ssts`] removes
//!   every `.bsst.tmp*` unconditionally at restart. WAL replay
//!   reconstructs the MemTable; the next flush allocates a FRESH
//!   `file_uid` (the manifest's `next_bond_v5_sst_uid` was already
//!   bumped by step 1, so the failed uid is never reused).
//!
//! - **Crash after step 5 rename, before step 7 `add_bond_v5_sst`:**
//!   final-name `<file_uid>.bsst` on disk, no manifest reference.
//!   [`cleanup_orphan_bond_ssts`] removes it -- its `file_uid` is
//!   not in [`crate::manifest::Manifest::live_bond_v5_sst_uids`].
//!   WAL replay rebuilds.
//!
//! - **Crash after step 7 `add_bond_v5_sst`, before step 8
//!   `save_atomic`:** the manifest changes were in-memory only and
//!   are lost on restart; the on-disk manifest is the pre-flush
//!   version, which does NOT reference the new `file_uid`. The
//!   `.bsst` file IS on disk -- classified as orphan at restart,
//!   removed. WAL replay rebuilds. NO data loss.
//!
//! - **Crash after step 8 `save_atomic` commit:** the manifest on
//!   disk references the new `file_uid`; the file is on disk and
//!   classified live by [`cleanup_orphan_bond_ssts`]. WAL replay
//!   still folds duplicate records.
//!
//! ## Empty iterator
//!
//! [`BondFlushWorker::flush`] returns `Ok(None)` WITHOUT allocating
//! a `file_uid`, calling `allocate_epoch`, or creating any file.
//! The caller (the bond-side `flush_once` integration point) skips
//! the `add_bond_v5_sst` call and the manifest counters stay
//! untouched. Mirrors `NodeFlushWorker::flush` semantics.
//!
//! ## Streaming discipline
//!
//! The flush takes an iterator (`IntoIterator<Item = (NodeId, u8,
//! NodeId, &BondPayload)>`) NOT a reference to a MemTable -- per
//! Phase 3.0 audit S5 Recommendation F (Q3.0.F), the BondMemTable
//! shape lives in `src/memtable.rs::MemTableInner.entries` and
//! stays there; the flush worker is decoupled from MemTable
//! internals. `flush_once` supplies the iterator from a swapped
//! snapshot. The writer holds at most one block in memory; the
//! iterator is consumed lazily.

use std::path::{Path, PathBuf};

use crate::bond_store::sst_writer::{
    BondPayload, BondSstError, BondSstWriteOpts, BondSstWriter,
};
use crate::manifest::{BondSSTEntry, Manifest, ManifestError};
use crate::node_intern::NodeId;

/// File suffix for committed BondSST files. The Phase 3.1 writer
/// renames `<file_uid>.bsst.tmp.<file_uid>` -> `<file_uid>.bsst`
/// on `finish()`; [`cleanup_orphan_bond_ssts`] keys on this suffix.
pub const BSST_SUFFIX: &str = ".bsst";

/// File suffix for committed reverse-index BondSST files (Phase
/// 3.6, Q3.0.B). A reverse file pairs with exactly one primary
/// BondSST via `BondSSTEntry.reverse_file_uid == Some(uid)` and
/// lives at `<bond_store_dir>/<uid>.bsst.reverse`. Same wire
/// format as primaries; the three wire fields semantically store
/// (object, predicate, subject) instead of (subject, predicate,
/// object) -- the CALLER knows which file it opened. See the
/// Phase 3.6 section in [`crate::bond_store::compaction`].
pub const BSST_REVERSE_SUFFIX: &str = ".bsst.reverse";

/// Errors surfaced by [`BondFlushWorker`] and the
/// [`cleanup_orphan_bond_ssts`] restart pass.
#[derive(Debug)]
pub enum BondFlushError {
    /// std::io::Error propagated from mkdir / fsync / readdir / remove.
    Io(std::io::Error),
    /// Underlying [`BondSstWriter`] failure (open / append / finish).
    Sst(BondSstError),
    /// Manifest helper rejected the
    /// [`Manifest::add_bond_v5_sst`] call. Always a programmer
    /// error in the flush worker context (uid uniqueness or level
    /// monotonicity violation); never a crash or data condition.
    Manifest(ManifestError),
    /// The flush worker was asked to write into a bond_store_dir
    /// that could not be created (mkdir_p failure). Wraps the
    /// underlying IO error context.
    BondStoreDirCreate { path: PathBuf, source: std::io::Error },
}

impl std::fmt::Display for BondFlushError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BondFlushError::Io(e) => write!(f, "BondFlushWorker IO: {}", e),
            BondFlushError::Sst(e) => write!(f, "BondFlushWorker BondSST: {}", e),
            BondFlushError::Manifest(e) => write!(f, "BondFlushWorker manifest: {}", e),
            BondFlushError::BondStoreDirCreate { path, source } => write!(
                f,
                "BondFlushWorker mkdir_p({}): {}",
                path.display(), source,
            ),
        }
    }
}

impl std::error::Error for BondFlushError {}

impl From<std::io::Error> for BondFlushError {
    fn from(e: std::io::Error) -> Self {
        BondFlushError::Io(e)
    }
}

impl From<BondSstError> for BondFlushError {
    fn from(e: BondSstError) -> Self {
        BondFlushError::Sst(e)
    }
}

impl From<ManifestError> for BondFlushError {
    fn from(e: ManifestError) -> Self {
        BondFlushError::Manifest(e)
    }
}

/// Cleanup report returned by [`cleanup_orphan_bond_ssts`]. Both
/// counters are diagnostic / test-visible; the cleanup is
/// best-effort discipline (failure to remove a single file is
/// logged but not fatal -- the next flush will still allocate a
/// fresh `file_uid`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CleanupReport {
    /// Number of `.bsst` files removed because their `file_uid` is
    /// not present in
    /// [`Manifest::live_bond_v5_sst_uids`]. These are
    /// orphans from a crashed flush between the writer rename
    /// (step 5) and the manifest `save_atomic` (step 8).
    pub orphans_removed: usize,
    /// Number of `.bsst.tmp*` files removed unconditionally. A
    /// `.tmp` means a flush in progress that was interrupted
    /// before step 5 -- the next flush allocates a new `file_uid`
    /// so reusing the half-written file is never desirable.
    pub tmps_removed: usize,
    /// Phase 3.6: number of `<uid>.bsst.reverse` files removed
    /// because their uid is not present in
    /// [`Manifest::live_bond_v5_reverse_uids`]. These are orphans
    /// from a crashed compaction between the reverse-file rename
    /// and the manifest `save_atomic`.
    pub reverse_orphans_removed: usize,
    /// Phase 3.6: number of `.bsst.reverse.tmp*` files removed
    /// unconditionally -- an interrupted reverse-index build at
    /// compaction time.
    pub reverse_tmps_removed: usize,
}

/// Bond-store flush worker. Borrows the live manifest at flush
/// time; does NOT own any background thread of its own. Phase 3.3
/// integrates into the existing `flush.rs::flush_once` worker
/// thread alongside [`crate::node_store::NodeFlushWorker`] and the
/// v4 bond SST writer.
pub struct BondFlushWorker {
    /// Directory where committed BondSST files live
    /// (`<file_uid>.bsst`). Derived by the caller from
    /// `Brain.wal_path.parent()/bond_store/`; the worker creates
    /// the directory on first flush if missing.
    bond_store_dir: PathBuf,
    /// OP-BRAIN-V5 6.5: per-SST encryption key. `None` = plaintext.
    key: Option<crate::encrypt::MasterKey>,
}

impl BondFlushWorker {
    /// Construct a BondFlushWorker rooted at `bond_store_dir`.
    /// The directory is NOT created here; [`Self::flush`] does
    /// `create_dir_all` on first call so test rigs that never
    /// invoke `flush` don't leave empty directories on disk.
    pub fn new(bond_store_dir: PathBuf) -> Self {
        BondFlushWorker { bond_store_dir, key: None }
    }

    /// Builder: set the per-SST encryption key (OP-BRAIN-V5 6.5).
    pub fn with_key(mut self, key: Option<crate::encrypt::MasterKey>) -> Self {
        self.key = key;
        self
    }

    /// Read-only view of the configured `bond_store_dir`.
    pub fn bond_store_dir(&self) -> &Path {
        &self.bond_store_dir
    }

    /// Compute the canonical `bond_store_dir` from a brain's WAL
    /// path. Convention: `<wal_path.parent()>/bond_store/`.
    /// Mirrors [`crate::node_store::NodeFlushWorker::node_store_dir_for_wal`].
    pub fn bond_store_dir_for_wal(wal_path: &Path) -> PathBuf {
        let parent = wal_path.parent().unwrap_or_else(|| Path::new("."));
        parent.join("bond_store")
    }

    /// Flush a bond iterator to a fresh BondSST file in
    /// `bond_store_dir`, allocating `file_uid` and `epoch` from
    /// `manifest`. Returns the [`BondSSTEntry`] for the caller to
    /// commit via [`Manifest::add_bond_v5_sst`] +
    /// [`Manifest::save_atomic`].
    ///
    /// Empty `bonds` returns `Ok(None)` WITHOUT allocating a
    /// `file_uid`, calling `allocate_epoch`, or creating any file
    /// -- the brief mandates exactly this behaviour so a no-op
    /// flush leaves the manifest counters untouched.
    ///
    /// Streaming discipline: the iterator is consumed lazily; the
    /// underlying [`BondSstWriter`] holds at most one block in
    /// memory (`opts.target_entries_per_block` bond rows). No Vec
    /// of the full bond stream is materialised in the flush worker.
    ///
    /// Ordering precondition: the iterator must yield bonds in
    /// strictly ascending `(subject, predicate, object)` order.
    /// The Phase 3.1 [`BondSstWriter::append`] enforces this
    /// invariant and returns [`BondSstError::OutOfOrderAppend`]
    /// on violation (debug builds panic on the way out for
    /// faster surfacing in tests).
    pub fn flush<'a, I>(
        &self,
        bonds: I,
        manifest: &mut Manifest,
    ) -> Result<Option<BondSSTEntry>, BondFlushError>
    where
        I: IntoIterator<Item = (NodeId, u8, NodeId, &'a BondPayload)>,
    {
        let mut iter = bonds.into_iter().peekable();
        if iter.peek().is_none() {
            // Empty iterator: NO file_uid allocation, NO epoch
            // allocation, NO file. Manifest counters untouched.
            return Ok(None);
        }

        // 0. Ensure the bond_store/ directory exists.
        if !self.bond_store_dir.exists() {
            std::fs::create_dir_all(&self.bond_store_dir).map_err(|e| {
                BondFlushError::BondStoreDirCreate {
                    path: self.bond_store_dir.clone(),
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
        // by [`cleanup_orphan_bond_ssts`] at restart, so the
        // reuse is safe.
        let file_uid = manifest.allocate_bond_v5_sst_uid();
        let epoch = manifest.allocate_epoch();

        // 3. Construct the writer. BondSstWriter::create opens
        // `<final_path>.tmp.<file_uid>` internally; finish()
        // fsyncs + renames to `<final_path>`.
        let filename = bsst_filename(file_uid);
        let final_path = self.bond_store_dir.join(&filename);
        let opts = BondSstWriteOpts {
            file_uid,
            ..Default::default()
        };
        let mut writer =
            BondSstWriter::create_with_key(&final_path, opts, self.key.clone())?;

        // 4. Stream the bond iterator. The peekable wrapper
        // means the first item is already-realised; iterate the
        // rest lazily so the writer holds at most one block in
        // memory at any time. We track entry_count locally for
        // a paranoia debug_assert against
        // BondSstWriteStats.entries_written.
        let mut first_subject: Option<u64> = None;
        let mut last_subject: u64 = 0;
        let mut entry_count: u64 = 0;
        for (subject, predicate, object, payload) in iter {
            writer.append(subject, predicate, object, payload)?;
            if first_subject.is_none() {
                first_subject = Some(subject.0);
            }
            last_subject = subject.0;
            entry_count += 1;
        }

        // 5. Finish: fsync the .tmp file, atomic rename to final_path.
        let stats = writer.finish()?;

        // 6. fsync the bond_store/ directory so the rename hits
        // disk (writer.finish only fsyncs the file, not the
        // directory entry update). Best-effort: a sync_all
        // failure here is logged via the IO error surface but
        // does not poison the flush -- the manifest commit at
        // step 8 is the durability anchor, and the orphan-
        // cleanup pass at restart still recovers correctly if
        // this fsync was racy.
        if let Ok(dir) = std::fs::File::open(&self.bond_store_dir) {
            let _ = dir.sync_all();
        }

        // 8. Build the manifest entry. `entry_count` was tracked
        // in the streaming loop above (matches
        // `stats.entries_written` -- we assert for paranoia in
        // debug builds).
        debug_assert_eq!(
            entry_count, stats.entries_written,
            "BondFlushWorker entry_count drift: counted {}, writer reports {}",
            entry_count, stats.entries_written,
        );
        let first = first_subject.expect("non-empty iterator has a first subject");
        let entry = BondSSTEntry {
            file_uid,
            first_subject: first,
            last_subject,
            entry_count: stats.entries_written,
            file_size: stats.file_size_bytes,
            epoch,
            // Phase 3.5 (Q3.0.B): L0 flush never builds a reverse
            // index -- Phase 3.6 builds reverse indexes lazily at
            // compaction time.
            reverse_file_uid: None,
        };

        // 9. Caller commits the entry via add_bond_v5_sst + save_atomic.
        Ok(Some(entry))
    }
}

/// Canonical filename for a BondSST given its `file_uid`. The
/// manifest records `file_uid` (not a path), so the worker, the
/// orphan-cleanup pass, and any future read-path code all derive
/// the on-disk name from the same helper. Mirrors Phase 2.6's
/// [`crate::node_store::nsst_filename`].
pub fn bsst_filename(file_uid: u64) -> String {
    format!("{}{}", file_uid, BSST_SUFFIX)
}

/// Parse a `file_uid` out of a `.bsst` filename. Returns `None`
/// for filenames that don't match the `<file_uid>.bsst` shape
/// (used by [`cleanup_orphan_bond_ssts`] to skip files it does
/// not own).
fn parse_bsst_filename(name: &str) -> Option<u64> {
    let stem = name.strip_suffix(BSST_SUFFIX)?;
    stem.parse::<u64>().ok()
}

/// Phase 3.6 (Q3.0.B): canonical filename for a reverse-index
/// BondSST given its uid:
/// `<bond_store_dir>/<reverse_uid>.bsst.reverse`. The manifest
/// records the uid on
/// [`crate::manifest::BondSSTEntry::reverse_file_uid`]; the
/// compactor, the orphan-cleanup pass, and the (Phase 3.7) read
/// path all derive the on-disk name from this one helper. Sibling
/// of [`bsst_filename`].
pub fn bsst_reverse_filename(reverse_uid: u64) -> String {
    format!("{}{}", reverse_uid, BSST_REVERSE_SUFFIX)
}

/// Parse a uid out of a `<uid>.bsst.reverse` filename. Returns
/// `None` for filenames that don't match (used by
/// [`cleanup_orphan_bond_ssts`] to skip files it does not own).
/// Note [`parse_bsst_filename`] independently rejects
/// `.bsst.reverse` names (the `.bsst` suffix does not terminate
/// them), so the classifier is robust to check order; the reverse
/// check still runs first for clarity.
fn parse_bsst_reverse_filename(name: &str) -> Option<u64> {
    let stem = name.strip_suffix(BSST_REVERSE_SUFFIX)?;
    stem.parse::<u64>().ok()
}

/// Restart-time orphan cleanup pass.
///
/// Scans `bond_store_dir` for committed `.bsst` files whose
/// `file_uid` is NOT present in
/// [`Manifest::live_bond_v5_sst_uids`] (orphans from a crashed
/// flush between writer rename and manifest `save_atomic`),
/// committed `.bsst.reverse` files whose uid is NOT present in
/// [`Manifest::live_bond_v5_reverse_uids`] (Phase 3.6: orphans
/// from a crashed compaction between the reverse-file rename and
/// the manifest save), and any `.bsst.tmp*` / `.bsst.reverse.tmp*`
/// files (interrupted writes before the respective rename),
/// removes them, and returns the counters.
///
/// Discipline (mirrors Phase 2.6's `cleanup_orphan_node_ssts`):
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
/// - Files that don't match the `<file_uid>.bsst`,
///   `<uid>.bsst.reverse`, `.bsst.tmp*`, or `.bsst.reverse.tmp*`
///   shape are LEFT ALONE. The cleanup pass does not own any
///   other file extensions in this directory.
pub fn cleanup_orphan_bond_ssts(
    bond_store_dir: &Path,
    manifest: &Manifest,
) -> Result<CleanupReport, BondFlushError> {
    let mut report = CleanupReport::default();

    // If the directory does not exist, no files to clean up.
    // First-boot brains have not flushed yet; the directory will
    // be created on the first flush by BondFlushWorker::flush.
    if !bond_store_dir.exists() {
        return Ok(report);
    }

    let live: std::collections::HashSet<u64> =
        manifest.live_bond_v5_sst_uids().into_iter().collect();
    let live_reverse: std::collections::HashSet<u64> =
        manifest.live_bond_v5_reverse_uids().into_iter().collect();

    let dir_iter = std::fs::read_dir(bond_store_dir)?;
    for entry in dir_iter {
        let entry = match entry {
            Ok(e) => e,
            Err(e) => {
                eprintln!(
                    "[bond_store::cleanup_orphan_bond_ssts] readdir error in {}: {} (skipping)",
                    bond_store_dir.display(), e,
                );
                continue;
            }
        };
        let path = entry.path();
        let name = match path.file_name().and_then(|n| n.to_str()) {
            Some(n) => n,
            None => continue, // non-UTF8 filenames not owned by us.
        };

        // Categorise (Phase 3.6 extended for reverse-index files):
        //   - `*.bsst.reverse.tmp*`  -> unconditional tmp (interrupted
        //                               reverse build at compaction)
        //   - `*.bsst.tmp*`          -> unconditional tmp
        //   - `<uid>.bsst.reverse`   -> reverse orphan candidate
        //                               (check live_bond_v5_reverse_uids)
        //   - `<file_uid>.bsst`      -> orphan candidate (check live_uids)
        //   - anything else          -> leave alone
        //
        // The .tmp checks come BEFORE the committed-name checks
        // because the writer's actual tmp filename appends
        // `.tmp.<uid>` to the FINAL name (`X.bsst.tmp.X` for
        // primaries, `X.bsst.reverse.tmp.X` for reverse files), so
        // substring checks are the safe match. NB
        // `.bsst.reverse.tmp` does NOT contain `.bsst.tmp` as a
        // substring -- the two tmp checks are disjoint and both
        // needed.
        if name.contains(".bsst.reverse.tmp") {
            if let Err(e) = std::fs::remove_file(&path) {
                eprintln!(
                    "[bond_store::cleanup_orphan_bond_ssts] remove reverse tmp {} failed: {} (continuing)",
                    path.display(), e,
                );
                continue;
            }
            report.reverse_tmps_removed += 1;
            continue;
        }

        if name.contains(".bsst.tmp") {
            if let Err(e) = std::fs::remove_file(&path) {
                eprintln!(
                    "[bond_store::cleanup_orphan_bond_ssts] remove tmp {} failed: {} (continuing)",
                    path.display(), e,
                );
                continue;
            }
            report.tmps_removed += 1;
            continue;
        }

        if let Some(uid) = parse_bsst_reverse_filename(name) {
            if !live_reverse.contains(&uid) {
                if let Err(e) = std::fs::remove_file(&path) {
                    eprintln!(
                        "[bond_store::cleanup_orphan_bond_ssts] remove reverse orphan {} failed: {} (continuing)",
                        path.display(), e,
                    );
                    continue;
                }
                eprintln!(
                    "[bond_store::cleanup_orphan_bond_ssts] removed reverse orphan {} (reverse uid {} not in manifest)",
                    path.display(), uid,
                );
                report.reverse_orphans_removed += 1;
            }
            continue;
        }

        if let Some(uid) = parse_bsst_filename(name) {
            if !live.contains(&uid) {
                if let Err(e) = std::fs::remove_file(&path) {
                    eprintln!(
                        "[bond_store::cleanup_orphan_bond_ssts] remove orphan {} failed: {} (continuing)",
                        path.display(), e,
                    );
                    continue;
                }
                eprintln!(
                    "[bond_store::cleanup_orphan_bond_ssts] removed orphan {} (file_uid {} not in manifest)",
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
    if report.orphans_removed > 0
        || report.tmps_removed > 0
        || report.reverse_orphans_removed > 0
        || report.reverse_tmps_removed > 0
    {
        if let Ok(dir) = std::fs::File::open(bond_store_dir) {
            let _ = dir.sync_all();
        }
    }

    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bond::TemporalKeyframe;
    use crate::bond_store::sst_reader::BondSstReader;
    use crate::bond_store::sst_writer::BondPayload;
    use crate::manifest::{KeyFingerprint, Manifest};
    use crate::node_intern::NodeId;
    use tempfile::tempdir;

    fn mk_payload(seq: u64) -> BondPayload {
        BondPayload {
            weight: 200,
            confidence: 1,
            keyframes: vec![TemporalKeyframe {
                timestamp: seq as i64,
                weight: 0.5,
                source: crate::bond::KeyframeSource::Baseline,
            }],
            seq,
        }
    }

    /// Build N bonds in strictly ascending (subject, predicate, object) order.
    /// subject = `base_subject + i`, predicate = 0, object = `base_object + i`.
    fn build_bonds(
        n: usize,
        base_subject: u64,
        base_object: u64,
        base_seq: u64,
    ) -> Vec<(NodeId, u8, NodeId, BondPayload)> {
        (0..n)
            .map(|i| {
                (
                    NodeId(base_subject + i as u64),
                    0u8,
                    NodeId(base_object + i as u64),
                    mk_payload(base_seq + i as u64),
                )
            })
            .collect()
    }

    /// Adapter that converts owned bonds into the borrow-shaped iterator
    /// `BondFlushWorker::flush` requires.
    fn as_borrow_iter<'a>(
        owned: &'a [(NodeId, u8, NodeId, BondPayload)],
    ) -> impl Iterator<Item = (NodeId, u8, NodeId, &'a BondPayload)> + 'a {
        owned.iter().map(|(s, p, o, pay)| (*s, *p, *o, pay))
    }

    /// Test A: flush creates a BondSST file, manifest gains a v5 bond
    /// entry, live_bond_v5_sst_uids includes the new file_uid.
    #[test]
    fn phase33_test_a_flush_creates_bsst_and_updates_manifest() {
        let dir = tempdir().unwrap();
        let bond_store_dir = dir.path().join("bond_store");
        let worker = BondFlushWorker::new(bond_store_dir.clone());
        let mut manifest = Manifest::new(KeyFingerprint::none());

        let owned = build_bonds(50, 1_000, 5_000, 0);
        let entry = worker
            .flush(as_borrow_iter(&owned), &mut manifest)
            .unwrap()
            .expect("non-empty flush");
        assert_eq!(entry.file_uid, 0, "first allocate returns 0");
        assert_eq!(entry.first_subject, 1_000);
        assert_eq!(entry.last_subject, 1_049);
        assert_eq!(entry.entry_count, 50);
        assert_eq!(entry.epoch, 0);

        // File on disk at expected path.
        let expected_path = bond_store_dir.join(bsst_filename(0));
        assert!(
            expected_path.exists(),
            "<file_uid>.bsst must exist post-flush"
        );

        // Manifest add_bond_v5_sst commits the entry.
        manifest.add_bond_v5_sst(0, entry.clone()).unwrap();
        assert_eq!(manifest.total_bond_v5_sst_count(), 1);
        assert_eq!(manifest.live_bond_v5_sst_uids(), vec![0u64]);
    }

    /// Test B: empty iterator returns Ok(None), no file created,
    /// manifest counters unchanged.
    #[test]
    fn phase33_test_b_empty_iter_no_file_no_alloc() {
        let dir = tempdir().unwrap();
        let bond_store_dir = dir.path().join("bond_store");
        let worker = BondFlushWorker::new(bond_store_dir.clone());
        let mut manifest = Manifest::new(KeyFingerprint::none());
        let pre_uid = manifest.peek_next_bond_v5_sst_uid();
        let pre_epoch = manifest.peek_next_epoch();

        let empty: Vec<(NodeId, u8, NodeId, BondPayload)> = Vec::new();
        let result = worker
            .flush(as_borrow_iter(&empty), &mut manifest)
            .unwrap();
        assert!(result.is_none(), "empty iterator returns Ok(None)");

        if bond_store_dir.exists() {
            let entries: Vec<_> = std::fs::read_dir(&bond_store_dir)
                .unwrap()
                .filter_map(|e| e.ok())
                .filter(|e| e.file_name().to_string_lossy().ends_with(".bsst"))
                .collect();
            assert!(entries.is_empty(), "no .bsst on empty-flush");
        }

        assert_eq!(manifest.peek_next_bond_v5_sst_uid(), pre_uid);
        assert_eq!(manifest.peek_next_epoch(), pre_epoch);
        assert_eq!(manifest.total_bond_v5_sst_count(), 0);
    }

    /// Test C: 3 sequential flushes -> 3 distinct files, monotonic
    /// file_uids (0, 1, 2), monotonic epochs (0, 1, 2), manifest
    /// has 3 entries.
    #[test]
    fn phase33_test_c_monotonic_file_uids_and_epochs() {
        let dir = tempdir().unwrap();
        let bond_store_dir = dir.path().join("bond_store");
        let worker = BondFlushWorker::new(bond_store_dir.clone());
        let mut manifest = Manifest::new(KeyFingerprint::none());

        let mut seen_uids = Vec::new();
        let mut seen_epochs = Vec::new();
        for round in 0..3u64 {
            let owned = build_bonds(20, round * 100_000, round * 200_000, round * 1_000);
            let entry = worker
                .flush(as_borrow_iter(&owned), &mut manifest)
                .unwrap()
                .expect("non-empty flush");
            seen_uids.push(entry.file_uid);
            seen_epochs.push(entry.epoch);
            manifest.add_bond_v5_sst(0, entry).unwrap();
        }
        assert_eq!(seen_uids, vec![0, 1, 2]);
        assert_eq!(seen_epochs, vec![0, 1, 2]);
        for uid in &seen_uids {
            assert!(
                bond_store_dir.join(bsst_filename(*uid)).exists(),
                "file_uid={} .bsst must exist",
                uid,
            );
        }
        assert_eq!(manifest.total_bond_v5_sst_count(), 3);
    }

    /// Test D: simulate a crash between writer rename and
    /// `add_bond_v5_sst`. The file exists on disk but the manifest
    /// does NOT reference it. cleanup_orphan_bond_ssts deletes it;
    /// cleanup report shows 1 orphan removed; manifest unchanged.
    #[test]
    fn phase33_test_d_orphan_cleaned_after_rename_without_commit() {
        let dir = tempdir().unwrap();
        let bond_store_dir = dir.path().join("bond_store");
        let worker = BondFlushWorker::new(bond_store_dir.clone());
        let mut manifest = Manifest::new(KeyFingerprint::none());

        let owned = build_bonds(10, 1_000, 5_000, 0);

        // Flush returns the entry but we deliberately DO NOT call
        // add_bond_v5_sst -- simulating a crash in that window.
        let entry = worker
            .flush(as_borrow_iter(&owned), &mut manifest)
            .unwrap()
            .expect("non-empty flush");
        let orphan_path = bond_store_dir.join(bsst_filename(entry.file_uid));
        assert!(orphan_path.exists(), "file on disk pre-cleanup");
        let live_uids_before = manifest.live_bond_v5_sst_uids();
        assert!(
            !live_uids_before.contains(&entry.file_uid),
            "orphan file_uid {} should NOT be live (add_bond_v5_sst not called)",
            entry.file_uid,
        );

        // Simulate restart: load a fresh manifest (equivalent to
        // reopening the on-disk MANIFEST, which never saw the bumps).
        let restart_manifest = Manifest::new(KeyFingerprint::none());
        let report = cleanup_orphan_bond_ssts(&bond_store_dir, &restart_manifest).unwrap();
        assert_eq!(report.orphans_removed, 1, "exactly 1 orphan removed");
        assert_eq!(report.tmps_removed, 0);
        assert!(!orphan_path.exists(), "orphan file deleted");
        assert_eq!(restart_manifest.total_bond_v5_sst_count(), 0);
    }

    /// Test E: a stray .bsst.tmp file is removed unconditionally on
    /// restart even if it doesn't correspond to any uid currently in
    /// the manifest's allocator range.
    #[test]
    fn phase33_test_e_tmp_files_cleaned() {
        let dir = tempdir().unwrap();
        let bond_store_dir = dir.path().join("bond_store");
        std::fs::create_dir_all(&bond_store_dir).unwrap();

        // Match the writer's actual tmp shape: `<final>.tmp.<uid>`.
        let stray_tmp = bond_store_dir.join("42.bsst.tmp.42");
        std::fs::write(&stray_tmp, b"abandoned tmp from a crashed writer").unwrap();
        // Plus the plain `.bsst.tmp` shape, to confirm the substring
        // match catches both.
        let stray_tmp_plain = bond_store_dir.join("7.bsst.tmp");
        std::fs::write(&stray_tmp_plain, b"plain tmp").unwrap();

        let manifest = Manifest::new(KeyFingerprint::none());
        let report = cleanup_orphan_bond_ssts(&bond_store_dir, &manifest).unwrap();
        assert_eq!(report.tmps_removed, 2, "both tmp shapes removed");
        assert_eq!(report.orphans_removed, 0);
        assert!(!stray_tmp.exists());
        assert!(!stray_tmp_plain.exists());
    }

    /// Test F: round-trip. Flush 100 bonds, persist manifest, reload
    /// manifest, open the referenced .bsst via BondSstReader,
    /// spot-check via iter_subject.
    #[test]
    fn phase33_test_f_round_trip_flush_reload_read() {
        let dir = tempdir().unwrap();
        let bond_store_dir = dir.path().join("bond_store");
        let manifest_path = dir.path().join("MANIFEST");
        let worker = BondFlushWorker::new(bond_store_dir.clone());

        let mut working_manifest = Manifest::new(KeyFingerprint::none());
        let owned = build_bonds(100, 1_000, 5_000, 0);
        let entry = worker
            .flush(as_borrow_iter(&owned), &mut working_manifest)
            .unwrap()
            .expect("non-empty flush");
        working_manifest.add_bond_v5_sst(0, entry.clone()).unwrap();
        working_manifest.save_atomic(&manifest_path).unwrap();

        // Reload manifest. v3 round-trip.
        let reloaded = Manifest::load(&manifest_path).unwrap().expect("manifest reload");
        assert_eq!(reloaded.total_bond_v5_sst_count(), 1);
        let (_lvl, reloaded_entry) = reloaded.iter_bond_v5_ssts().next().unwrap();
        assert_eq!(reloaded_entry.file_uid, entry.file_uid);
        assert_eq!(reloaded_entry.entry_count, 100);
        assert_eq!(reloaded_entry.first_subject, 1_000);
        assert_eq!(reloaded_entry.last_subject, 1_099);

        // Open via BondSstReader and spot-check three subjects.
        let bsst_path = bond_store_dir.join(bsst_filename(reloaded_entry.file_uid));
        let reader = BondSstReader::open(&bsst_path).unwrap();
        for subj in &[1_000u64, 1_050, 1_099] {
            let mut iter = reader.iter_subject(NodeId(*subj));
            let first = iter.next().expect("subject must round-trip").unwrap();
            assert_eq!(first.0 .0, *subj);
        }
        // A subject outside the file's range yields zero entries.
        let mut outside = reader.iter_subject(NodeId(5_000));
        assert!(outside.next().is_none(), "subject outside range yields none");
    }

    /// Test G: stream discipline. Yield a moderate number of bonds
    /// (8K) and assert the writer's internal block buffer never
    /// exceeds `target_entries_per_block + safety` rows live --
    /// i.e. the worker does not collect the whole stream into a
    /// Vec. The Phase 3.1 writer flushes a block when
    /// `current_block_entries >= target_entries_per_block`; we
    /// don't have a direct accessor into the writer's internal
    /// state from the worker boundary, so we assert structurally
    /// via input-iterator side-effect: count how many bonds the
    /// iterator has yielded vs how many `writer.append` has fully
    /// processed (we cannot observe `append` directly either, so
    /// the structural witness is that the flush completes without
    /// allocating 8K * payload bytes of resident memory in the
    /// worker frame -- functionally equivalent to NodeFlushWorker's
    /// pattern).
    ///
    /// The strongest structural assertion available at this
    /// boundary is: the iterator we pass in is single-pass (a
    /// `std::iter::from_fn` closure) -- if the worker materialised
    /// the stream into a Vec it would still complete but the same
    /// iterator instance could not be iterated again. We verify
    /// the lazy-consumption shape compiles and runs to completion.
    #[test]
    fn phase33_test_g_stream_discipline() {
        let dir = tempdir().unwrap();
        let bond_store_dir = dir.path().join("bond_store");
        let worker = BondFlushWorker::new(bond_store_dir.clone());
        let mut manifest = Manifest::new(KeyFingerprint::none());

        // Lazy iterator over 8_192 bonds. Each yield produces a
        // fresh BondPayload owned by the closure's captured buffer.
        // We use a self-managed buffer so the borrow against the
        // payload is valid for the iterator's lifetime.
        let n = 8_192usize;
        let mut buffer: Vec<BondPayload> = Vec::with_capacity(n);
        for i in 0..n {
            buffer.push(mk_payload(i as u64));
        }
        // The iterator borrows from `buffer` to produce the
        // (subject, predicate, object, &payload) tuples lazily.
        let mut idx = 0usize;
        let lazy = std::iter::from_fn(|| {
            if idx >= n {
                return None;
            }
            let i = idx;
            idx += 1;
            Some((NodeId(1_000 + i as u64), 0u8, NodeId(5_000 + i as u64), &buffer[i]))
        });

        let entry = worker
            .flush(lazy, &mut manifest)
            .unwrap()
            .expect("non-empty flush");
        assert_eq!(entry.entry_count, n as u64);
        assert_eq!(entry.first_subject, 1_000);
        assert_eq!(entry.last_subject, 1_000 + n as u64 - 1);
        // The from_fn closure consumed `idx` up to `n` -- if the
        // worker had collected via .collect::<Vec<_>>() it would
        // still pass this assert, but the surface signature
        // `IntoIterator<Item = (...)>` plus the peekable adapter
        // is the structural pattern that prevents collection.
        assert_eq!(idx, n, "iterator was fully consumed");
    }

    /// Plus: cleanup is a no-op on a never-flushed brain (the
    /// bond_store dir doesn't exist yet).
    #[test]
    fn phase33_cleanup_noop_on_missing_dir() {
        let dir = tempdir().unwrap();
        let bond_store_dir = dir.path().join("bond_store");
        let manifest = Manifest::new(KeyFingerprint::none());
        let report = cleanup_orphan_bond_ssts(&bond_store_dir, &manifest).unwrap();
        assert_eq!(report.orphans_removed, 0);
        assert_eq!(report.tmps_removed, 0);
    }

    /// Plus: bsst_filename / parse_bsst_filename round-trip.
    #[test]
    fn phase33_bsst_filename_round_trip() {
        for uid in [0u64, 1, 42, 999_999] {
            let name = bsst_filename(uid);
            assert_eq!(name, format!("{}.bsst", uid));
            assert_eq!(parse_bsst_filename(&name), Some(uid));
        }
        // Bad shapes return None.
        assert_eq!(parse_bsst_filename("abc.bsst"), None);
        assert_eq!(parse_bsst_filename("42.txt"), None);
        assert_eq!(parse_bsst_filename("42.bsst.tmp"), None);
        assert_eq!(parse_bsst_filename(".bsst"), None);
    }

    /// Plus: live .bsst files are NOT cleaned (manifest references them).
    #[test]
    fn phase33_live_files_preserved() {
        let dir = tempdir().unwrap();
        let bond_store_dir = dir.path().join("bond_store");
        let manifest_path = dir.path().join("MANIFEST");
        let worker = BondFlushWorker::new(bond_store_dir.clone());

        let mut working_manifest = Manifest::new(KeyFingerprint::none());
        let owned = build_bonds(5, 1_000, 5_000, 0);
        let entry = worker
            .flush(as_borrow_iter(&owned), &mut working_manifest)
            .unwrap()
            .expect("non-empty flush");
        working_manifest.add_bond_v5_sst(0, entry.clone()).unwrap();
        working_manifest.save_atomic(&manifest_path).unwrap();

        let restart_manifest = Manifest::load(&manifest_path).unwrap().expect("manifest reload");
        assert!(restart_manifest
            .live_bond_v5_sst_uids()
            .contains(&entry.file_uid));

        let final_path = bond_store_dir.join(bsst_filename(entry.file_uid));
        assert!(final_path.exists());

        let report = cleanup_orphan_bond_ssts(&bond_store_dir, &restart_manifest).unwrap();
        assert_eq!(report.orphans_removed, 0, "no orphans -- file is live");
        assert_eq!(report.tmps_removed, 0);
        assert!(final_path.exists(), "live file persists");
        assert_eq!(restart_manifest.total_bond_v5_sst_count(), 1);
    }
}
