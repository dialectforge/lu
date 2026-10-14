//! `BondCompactionWorker` -- streaming k-way merge over BondSST files
//! with epoch-barrier tombstone GC.
//!
//! OP-BRAIN-V5 Phase 3.4 (Q7). The bond-side compactor: takes N
//! input BondSSTs at one LSM level, merges them into a single
//! output BondSST at a higher level, drops tombstones that no
//! older SST could still reference, and commits the swap to the
//! manifest via the Phase 3.3 helpers
//! (`allocate_bond_v5_sst_uid`, `allocate_epoch`,
//! `add_bond_v5_sst`, `retire_bond_v5_ssts`).
//!
//! Direct structural template: Phase 2.7
//! [`crate::node_store::compaction`] (`NodeCompactionWorker`). The
//! cursor shape, the GC barrier semantics, and the atomic commit
//! sequence are ported 1:1 with bond key/payload types
//! (`(subject, predicate, object)` triple key, [`BondPayload`]).
//!
//! ## Scope boundary (NEW v5 BondSST compaction, NOT v4)
//!
//! Phase 3.4 ships compaction for the v5 BondSST store ONLY. The
//! v4 `src/compaction.rs` (bond compaction with the two
//! HashMap-the-world antipatterns -- `rows: Vec<MergeRow>` at line
//! 357 and `node_winners: HashMap` at line 362) is NOT touched
//! here. Q7's directive to replace both with streaming heap merges
//! is fulfilled on the v5 side by THIS module; the v4 compactor is
//! retired wholesale at the Phase 3.9 cutover, not patched.
//!
//! ## Streaming k-way merge -- RSS bound is O(k), NOT O(n)
//!
//! The merge is implemented as a [`BinaryHeap`] (min-heap by the
//! `(subject, predicate, object)` triple key) of [`MergeCursor`]
//! entries -- ONE cursor per input SST. Each `pop` yields the
//! lowest triple across all inputs; the merge then advances THAT
//! cursor by pulling the next item from `iters[source_idx]` and
//! pushes the new head back.
//!
//! The heap NEVER holds more than `plan.source_uids.len()`
//! cursors. Total in-memory state during the merge is bounded by:
//!
//! - `k` open [`BondSstReader`] handles (each carries a small block
//!   cache, capped by `DEFAULT_BLOCK_CACHE_CAP`),
//! - `k` heap cursors (each holds one [`BondPayload`] -- a few
//!   dozen bytes plus inline keyframes),
//! - one in-flight bond row being written to the output.
//!
//! The entire output is streamed through [`BondSstWriter::append`]
//! one entry at a time. NO `Vec<(NodeId, u8, NodeId, BondPayload)>`
//! is ever materialised. This is Q7's CORE INVARIANT -- the exact
//! invariant the v4 bond compactor violates and that this port
//! exists to restore.
//!
//! (Phase 3.6 scope note: the MERGE stays O(k). The reverse-index
//! projection added by Phase 3.6 deliberately materialises one Vec
//! of the compacted OUTPUT -- see the Phase 3.6 section below for
//! the bound and the trade-off.)
//!
//! ## Dedup -- higher `seq` wins
//!
//! When two cursors point at the SAME `(subject, predicate,
//! object)` triple, the merge loop pops them both, keeps the entry
//! with the strictly higher [`BondPayload::seq`], and discards the
//! other. This matches the Phase 2.7 node-side discipline and
//! preserves the latest-write-wins property of the LSM layer.
//!
//! ## Tombstone GC -- cross-store epoch barrier
//!
//! A bond row is a tombstone iff `payload.confidence ==
//! 0xFF` ([`BOND_TOMBSTONE_CONFIDENCE`], Phase 3.0 audit + Phase
//! 3.1 convention, matching v4 `TOMBSTONE_CONFIDENCE` at
//! `src/sst.rs:187`). A tombstone INHERITS the epoch of the source
//! BondSST it came from (the [`BondSSTEntry::epoch`] looked up via
//! the manifest before the merge starts).
//!
//! The barrier is the minimum epoch we must keep tombstones for:
//!
//! ```text
//!   gc_barrier = min(
//!       Manifest::min_bond_v5_sst_epoch_excluding(&plan.source_uids),
//!       Manifest::min_node_sst_epoch_excluding(&[]),
//!   )
//! ```
//!
//! `None` from either query is treated as `u64::MAX` ("no SST in
//! this store could possibly reference the tombstone"). If BOTH
//! are `None`, the effective barrier is `u64::MAX` and every
//! tombstone is safe to drop.
//!
//! The NODE-side minimum participates because the bond store holds
//! tombstones referring to NodeIds: a bond tombstone must not be
//! GC'd before every node-side SST that could still surface the
//! pre-tombstone state has itself been compacted past the barrier.
//! Both stores share one epoch counter
//! ([`Manifest::allocate_epoch`]), so the cross-store comparison
//! is `==` on `epoch: u64`. This is the same cross-store epoch
//! barrier discipline as Phase 2.7 (which symmetrically folds the
//! bond-side minimum into the node compactor's barrier).
//!
//! Drop rule:
//!
//! ```text
//!   payload.confidence == 0xFF && source_epoch < gc_barrier
//! ```
//!
//! Otherwise the tombstone is carried into the output, where it
//! inherits the OUTPUT's new epoch from the manifest's perspective
//! (the [`BondSSTEntry::epoch`] recorded for the output SST). The
//! manifest is the authority on "what epoch is this SST stamped
//! with"; a future compaction that consults `source_epoch` for
//! this entry will find the output's epoch.
//!
//! Mental model: drop a tombstone only when no SST that could
//! reference the value it shadows still exists.
//!
//! ## Phase 3.6 -- paired reverse-index emission (Q3.0.B)
//!
//! Every compaction output gets a paired reverse-index file at
//! `<bond_store_dir>/<reverse_uid>.bsst.reverse`, written AFTER
//! the primary `.bsst` is finished + renamed and BEFORE the
//! manifest commit. The output [`BondSSTEntry::reverse_file_uid`]
//! is `Some(reverse_uid)`. L0 (flush-born) SSTs keep `None`;
//! reverse-index queries against L0 fall through to a scan --
//! Phase 3.7's read-path concern, not Phase 3.6's.
//!
//! ### Same wire format, swapped field semantics
//!
//! A reverse file uses the SAME BondSST wire format as a primary:
//! three NodeId-u64 fields with a u8 predicate between them. For a
//! primary those fields semantically mean (subject, predicate,
//! object); for a reverse file the SAME fields mean (object,
//! predicate, subject). [`BondSstReader`] has no "reverse mode" --
//! it just iterates entries; the CALLER knows which file it
//! opened. A reverse file is therefore sorted, block-indexed, and
//! range-scannable by original-object first -- exactly the
//! reverse-traversal access path Phase 3.7 needs.
//!
//! ### RSS bound -- one-pass-with-buffer, NOT streaming
//!
//! The projection re-reads the just-written primary via
//! `iter_all()`, collects every entry into a `Vec<(object,
//! predicate, subject, payload)>`, sorts it, and streams it
//! through a second [`BondSstWriter`]. The Vec materialises ALL
//! compacted bonds at once: ~54 B/entry plus inline keyframes,
//! i.e. ~1 GB resident peak at 18M bonds; ~2.7 MB at a 50K-entry
//! L0->L1 merge. Acceptable for periodic compaction.
//!
//! Alternative considered + REJECTED: a streaming two-pass
//! re-merge through the source readers would bound RSS at O(k)
//! but doubles input I/O and re-runs dedup + tombstone GC a
//! second time. The Phase 3.0 audit's "lazy at compaction"
//! wording implies one-pass-with-buffer; the buffer is the
//! documented trade-off.
//!
//! ### Crash windows for the reverse file
//!
//! - Crash during the reverse write: `.bsst.reverse.tmp.<uid>`
//!   orphan on disk; cleanup removes every `*.bsst.reverse.tmp*`
//!   unconditionally. The already-renamed primary `.bsst` is
//!   itself unreferenced (no manifest commit yet) and removed as
//!   an orphan. Inputs intact, retry safe.
//! - Crash after the reverse rename, before save_atomic: both the
//!   primary `.bsst` and the `.bsst.reverse` are unreferenced
//!   orphans; cleanup classifies `.bsst.reverse` files against
//!   [`Manifest::live_bond_v5_reverse_uids`] and removes both.
//!   Inputs intact, retry safe.
//! - Crash after save_atomic: primary + reverse are both durably
//!   referenced (the committed entry carries `reverse_file_uid`);
//!   the inputs are orphans and are cleaned. No data loss.
//!
//! ## Atomic commit sequence
//!
//! ```text
//!   1.  Query min_bond_v5_sst_epoch_excluding(&plan.source_uids)
//!       + min_node_sst_epoch_excluding(&[]) -> gc_barrier
//!   2.  manifest.allocate_bond_v5_sst_uid() -> out_uid
//!   3.  manifest.allocate_epoch()           -> out_epoch
//!   4.  Open all source readers; look up each input's BondSSTEntry
//!       to recover its source_epoch (for the GC rule).
//!   5.  Build the streaming merge iterator (RSS-bounded heap).
//!   6.  BondSstWriter::create(<out_uid>.bsst, opts) -- writer opens
//!       the .bsst.tmp.<uid> file internally.
//!   7.  for each (subject, predicate, object, payload, source_idx)
//!       from merge:
//!         if payload.confidence == 0xFF
//!            && source_epochs[source_idx] < gc_barrier {
//!             drop  // GC'd tombstone
//!         } else if dedup -> older seq drops
//!         else writer.append(subject, predicate, object, &payload)
//!   8.  writer.finish() -> BondSstWriteStats (fsync + rename
//!       .bsst.tmp.<uid> -> <out_uid>.bsst).
//!   9.  fsync the bond_store/ directory (rename hits disk).
//!   9b. (Phase 3.6) manifest.allocate_bond_v5_sst_uid() ->
//!       reverse_uid; open the primary via BondSstReader::open;
//!       collect the (object, predicate, subject, payload)
//!       projection into a Vec; sort by (object, predicate,
//!       subject); stream through a second BondSstWriter to
//!       <reverse_uid>.bsst.reverse (writer tmp + fsync + rename);
//!       fsync the directory again.
//!   10. Build BondSSTEntry from stats + out_uid + out_epoch +
//!       reverse_file_uid = Some(reverse_uid).
//!   11. manifest.add_bond_v5_sst(plan.target_level, output_entry).
//!       (ADD FIRST -- see below.)
//!   12. manifest.retire_bond_v5_ssts(&plan.source_uids).
//!       (RETIRE SECOND.)
//!   13. Caller owns manifest.save_atomic.
//!   14. Source BondSST files NOT unlinked here. The
//!       `cleanup_orphan_bond_ssts` pass owns that after save_atomic.
//! ```
//!
//! ## ADD-then-RETIRE order is load-bearing
//!
//! `add_bond_v5_sst` requires `target_level <= bond_v5_levels.len()`
//! and `retire_bond_v5_ssts` drops empty trailing levels. A
//! retire-first order fails when the sources are the only L0
//! entries: retire shrinks `bond_v5_levels.len()` to 0, then
//! `add_bond_v5_sst(target_level = 1)` rejects with `InvalidLevel`.
//! ADD first creates L1, then RETIRE cleans L0. Phase 2.7 caught
//! exactly this bug on the node side; the bond manifest helpers
//! have the same trailing-level-trim semantics, so the same order
//! is the only legal one here.
//!
//! ## Crash-window analysis (no data loss possible)
//!
//! - **Crash before step 8 rename:** `.bsst.tmp.<uid>` orphan on
//!   disk; `cleanup_orphan_bond_ssts` at restart removes it
//!   unconditionally. Inputs intact. The `next_bond_v5_sst_uid`
//!   and `next_epoch` bumps in steps 2/3 are in-memory only --
//!   lost on restart, and the manifest counter is monotonic from
//!   the on-disk value, so the next compaction simply allocates
//!   the same uid; the tmp orphan is gone by then.
//!
//! - **Crash between 8 and 11 (rename done, manifest not
//!   mutated):** `<out_uid>.bsst` on disk, no manifest reference.
//!   Cleanup classifies it as an orphan (file_uid not in
//!   `live_bond_v5_sst_uids`) and removes it. Inputs intact.
//!   Retry safe.
//!
//! - **Crash between 11 and 12 / before save:** the in-memory
//!   manifest mutation never reaches disk because `save_atomic`
//!   is the durability anchor. On restart the on-disk manifest
//!   still references the inputs; the output is an orphan and is
//!   cleaned up. Retry safe.
//!
//! - **Crash after save_atomic:** the manifest on disk references
//!   the new output and NOT the inputs. The input `.bsst` files
//!   are now orphans; the cleanup pass at next restart removes
//!   them. Output is canonical. No data loss.
//!
//! At every break point the input files survive UNTIL the new
//! output is durably referenced by the manifest. Same discipline
//! as the Phase 3.3 flush worker, scaled to N inputs and 1 output.
//! Phase 3.6 adds the reverse-file windows -- see "Crash windows
//! for the reverse file" above; the property is unchanged because
//! the reverse is written BEFORE the manifest commit and cleaned
//! as an orphan in every pre-commit window.
//!
//! ## Not yet wired
//!
//! Phase 3.4 ships the WORKER, not a scheduler. Tests construct
//! the [`CompactionPlan`] directly. No production code path calls
//! [`BondCompactionWorker::compact`] in this commit.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use crate::bond_store::flush::{bsst_filename, bsst_reverse_filename};
use crate::bond_store::sst_reader::{BondSstRangeIter, BondSstReader};
use crate::bond_store::sst_writer::{BondPayload, BondSstError, BondSstWriteOpts, BondSstWriter};
use crate::manifest::{BondSSTEntry, Manifest, ManifestError};
use crate::node_intern::NodeId;

/// Tombstone marker on [`BondPayload::confidence`]. Matches the
/// v4 `TOMBSTONE_CONFIDENCE` convention at `src/sst.rs:187` and
/// the Phase 3.1 writer's doc-comment. A bond row whose
/// `confidence` equals this value shadows (deletes) every older
/// row with the same `(subject, predicate, object)` triple.
pub const BOND_TOMBSTONE_CONFIDENCE: u8 = 0xFF;

// ----------------------------------------------------------------------
// Errors.
// ----------------------------------------------------------------------

/// Errors surfaced by [`BondCompactionWorker::compact`].
#[derive(Debug)]
pub enum BondCompactionError {
    /// `std::io::Error` propagated from open / read / write / fsync
    /// / rename (NB: this worker does NOT unlink inputs -- the
    /// cleanup pass does; an `Io` here means an output-side
    /// failure).
    Io(std::io::Error),
    /// Underlying [`BondSstError`] from reader/writer.
    BondSst(BondSstError),
    /// Plan-shape violation: empty `source_uids`, duplicate uids in
    /// the same plan, etc.
    InvalidPlan(String),
    /// One of the `source_uids` in the plan is not present in any
    /// live v5 bond level. The manifest is the authority on
    /// liveness; compaction against a non-existent input is a
    /// programmer error.
    SourceNotFound(u64),
    /// The sources span more than one level, or a source does not
    /// live at `plan.source_level`. Phase 3.4 ships a single-level
    /// compactor; cross-level merges are a later scheduler concern.
    LevelMismatch(String),
    /// Manifest helper (`add_bond_v5_sst`, `retire_bond_v5_ssts`)
    /// rejected the post-merge commit. Always a programmer error.
    Manifest(ManifestError),
}

impl std::fmt::Display for BondCompactionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BondCompactionError::Io(e) => write!(f, "BondCompactionWorker IO: {}", e),
            BondCompactionError::BondSst(e) => write!(f, "BondCompactionWorker BondSST: {}", e),
            BondCompactionError::InvalidPlan(s) => {
                write!(f, "BondCompactionWorker invalid plan: {}", s)
            }
            BondCompactionError::SourceNotFound(uid) => write!(
                f,
                "BondCompactionWorker source file_uid {} not live in manifest",
                uid
            ),
            BondCompactionError::LevelMismatch(s) => {
                write!(f, "BondCompactionWorker level mismatch: {}", s)
            }
            BondCompactionError::Manifest(e) => write!(f, "BondCompactionWorker manifest: {}", e),
        }
    }
}

impl std::error::Error for BondCompactionError {}

impl From<std::io::Error> for BondCompactionError {
    fn from(e: std::io::Error) -> Self {
        BondCompactionError::Io(e)
    }
}

impl From<BondSstError> for BondCompactionError {
    fn from(e: BondSstError) -> Self {
        BondCompactionError::BondSst(e)
    }
}

impl From<ManifestError> for BondCompactionError {
    fn from(e: ManifestError) -> Self {
        BondCompactionError::Manifest(e)
    }
}

// ----------------------------------------------------------------------
// Plan + result.
// ----------------------------------------------------------------------

/// Compaction job description supplied by the caller.
///
/// `source_uids` lists the [`BondSSTEntry::file_uid`]s to merge.
/// All sources MUST live at `source_level` (Phase 3.4 ships a
/// single-level compactor; the scheduler decides which level
/// participates per round).
///
/// `target_level` is where the merged output is added via
/// [`Manifest::add_bond_v5_sst`]. Conventionally `source_level + 1`
/// but the worker does not enforce that -- a same-level rewrite
/// is also a legal plan (used by future read-amp control).
#[derive(Debug, Clone)]
pub struct CompactionPlan {
    pub source_uids: Vec<u64>,
    pub source_level: usize,
    pub target_level: usize,
}

/// Result returned by a successful [`BondCompactionWorker::compact`].
#[derive(Debug, Clone)]
pub struct CompactionResult {
    /// New manifest entry now live at `plan.target_level`.
    pub output_entry: BondSSTEntry,
    /// The retired input entries, in the order they appeared in
    /// `plan.source_uids`. Returned to the caller for diagnostics
    /// and so tests can verify that retirement matched the intent.
    pub input_entries_retired: Vec<BondSSTEntry>,
    /// Total entries read across all inputs (live + tombstone).
    pub entries_in: u64,
    /// Total entries written to the output.
    pub entries_out: u64,
    /// Tombstones dropped by the GC rule.
    pub entries_dropped_tombstone: u64,
    /// Dedup losers (entries shadowed by a higher-seq entry at the
    /// same `(subject, predicate, object)` triple in a different
    /// input).
    pub entries_dropped_overwritten: u64,
    /// Wall-clock compaction duration (open-to-rename).
    pub elapsed: Duration,
}

// ----------------------------------------------------------------------
// Merge cursor (heap entry).
// ----------------------------------------------------------------------

/// One entry in the min-heap that drives the k-way merge.
///
/// Ordering is reversed because [`BinaryHeap`] is a max-heap by
/// default and we want the smallest `(subject, predicate, object)`
/// triple at the top. The `source_idx` is the index into the
/// parallel `Vec<BondSstReader>` / `Vec<BondSstRangeIter>` /
/// `Vec<u64>` (source_epochs); after `pop` the merge loop advances
/// `iters[source_idx]` to get the next entry from THAT source.
#[derive(Debug)]
struct MergeCursor {
    key: (NodeId, u8, NodeId),
    payload: BondPayload,
    source_idx: usize,
}

impl PartialEq for MergeCursor {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key && self.source_idx == other.source_idx
    }
}
impl Eq for MergeCursor {}

impl PartialOrd for MergeCursor {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MergeCursor {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse for min-heap.
        // Primary key: (subject, predicate, object) ascending.
        // Tie-break: source_idx ascending so the heap order is
        // total and reproducible (tests assert on it).
        other
            .key
            .cmp(&self.key)
            .then_with(|| other.source_idx.cmp(&self.source_idx))
    }
}

// ----------------------------------------------------------------------
// Worker.
// ----------------------------------------------------------------------

/// BondSST compaction worker rooted at a `bond_store_dir`.
///
/// Stateless other than the directory path; safe to construct
/// per-compaction or to share across compactions.
pub struct BondCompactionWorker {
    bond_store_dir: PathBuf,
    /// OP-BRAIN-V5 6.5: per-SST encryption key. `None` = plaintext.
    key: Option<crate::encrypt::MasterKey>,
}

// Debug-only counter for the streaming RSS-bound test (E).
// Compaction never panics on this; the test reads it after
// `compact()` returns. Updated inside the merge loop with every
// heap mutation. Lives in a thread-local so concurrent tests
// don't collide.
#[cfg(test)]
thread_local! {
    static MAX_HEAP_SIZE_OBSERVED: std::cell::Cell<usize> = std::cell::Cell::new(0);
}

#[cfg(test)]
fn _reset_heap_observer() {
    MAX_HEAP_SIZE_OBSERVED.with(|c| c.set(0));
}

#[cfg(test)]
fn _record_heap_size(n: usize) {
    MAX_HEAP_SIZE_OBSERVED.with(|c| {
        if n > c.get() {
            c.set(n);
        }
    });
}

#[cfg(test)]
fn _peek_max_heap_size() -> usize {
    MAX_HEAP_SIZE_OBSERVED.with(|c| c.get())
}

#[cfg(not(test))]
#[inline]
fn _record_heap_size(_: usize) {}

impl BondCompactionWorker {
    /// Construct a worker rooted at `bond_store_dir`. The directory
    /// must already exist (compaction is always preceded by a flush
    /// that created the directory).
    pub fn new(bond_store_dir: PathBuf) -> Self {
        BondCompactionWorker { bond_store_dir, key: None }
    }

    /// Builder: set the per-SST encryption key (OP-BRAIN-V5 6.5).
    pub fn with_key(mut self, key: Option<crate::encrypt::MasterKey>) -> Self {
        self.key = key;
        self
    }

    /// Read-only view of the configured directory (mostly for tests).
    pub fn bond_store_dir(&self) -> &Path {
        &self.bond_store_dir
    }

    /// Compact the inputs named by `plan` into a single output
    /// BondSST at `plan.target_level`.
    ///
    /// See the module docs for the atomic commit sequence and the
    /// tombstone-GC rule. The caller is responsible for
    /// `manifest.save_atomic` after this returns; the worker
    /// mutates the in-memory manifest only. Source BondSST files
    /// are NOT unlinked here -- `cleanup_orphan_bond_ssts` owns
    /// that after the manifest save commits.
    ///
    /// OP-BRAIN-V5 Phase 3.8.1: `reader_cache` is the Phase 3.7
    /// `BondSstReaderCache` to evict retired readers from after the
    /// manifest retire commits -- both the primary `file_uid` AND
    /// the Phase 3.6 `reverse_file_uid` (when present) of every
    /// retired entry. `None` for scaffolds without a cache.
    pub fn compact(
        &self,
        plan: CompactionPlan,
        manifest: &mut Manifest,
        reader_cache: Option<&crate::bond_store::BondSstReaderCache>,
    ) -> Result<CompactionResult, BondCompactionError> {
        let started = Instant::now();

        // ---- 0. Validate plan shape. ----
        if plan.source_uids.is_empty() {
            return Err(BondCompactionError::InvalidPlan(
                "plan.source_uids is empty".to_string(),
            ));
        }
        // Disallow duplicate uids in the same plan.
        {
            let mut sorted = plan.source_uids.clone();
            sorted.sort_unstable();
            sorted.dedup();
            if sorted.len() != plan.source_uids.len() {
                return Err(BondCompactionError::InvalidPlan(
                    "plan.source_uids contains duplicate file_uids".to_string(),
                ));
            }
        }

        // ---- 1. Resolve each source uid to its (level, BondSSTEntry). ----
        let mut source_entries: Vec<BondSSTEntry> = Vec::with_capacity(plan.source_uids.len());
        for &uid in &plan.source_uids {
            let found = manifest.iter_bond_v5_ssts().find(|(_, e)| e.file_uid == uid);
            let (lvl, entry) = match found {
                Some((l, e)) => (l as usize, e.clone()),
                None => return Err(BondCompactionError::SourceNotFound(uid)),
            };
            if lvl != plan.source_level {
                return Err(BondCompactionError::LevelMismatch(format!(
                    "source file_uid {} lives at level {}, plan.source_level = {}",
                    uid, lvl, plan.source_level
                )));
            }
            source_entries.push(entry);
        }
        let source_epochs: Vec<u64> = source_entries.iter().map(|e| e.epoch).collect();

        // ---- 2. Compute the GC barrier BEFORE mutating the manifest. ----
        //
        // gc_barrier = min(bond_v5_min_excluding_inputs, node_store_min).
        // None from either query is treated as u64::MAX (no SST in
        // that store can possibly reference the tombstone). If both
        // are None, the barrier is u64::MAX and every tombstone is
        // safe to drop. The node-side minimum participates because
        // bond tombstones refer to NodeIds -- see module docs
        // (cross-store epoch barrier, same discipline as Phase 2.7).
        let bond_barrier = manifest
            .min_bond_v5_sst_epoch_excluding(&plan.source_uids)
            .unwrap_or(u64::MAX);
        let node_barrier = manifest
            .min_node_sst_epoch_excluding(&[])
            .unwrap_or(u64::MAX);
        let gc_barrier = bond_barrier.min(node_barrier);

        // ---- 3. Allocate output uid + epoch. ----
        //
        // Bumps the in-memory counters. If we crash before
        // save_atomic, the bumps are lost and the next boot reuses
        // the same uid -- but the orphan cleanup pass removes the
        // .bsst file from the failed attempt before the brain
        // accepts writes, so reuse is safe.
        let out_uid = manifest.allocate_bond_v5_sst_uid();
        let out_epoch = manifest.allocate_epoch();

        // ---- 4. Open all source readers. ----
        //
        // Vec<BondSstReader> outlives the iterators we build over
        // them, so the lifetime juggling is clean.
        let mut readers: Vec<BondSstReader> = Vec::with_capacity(plan.source_uids.len());
        for entry in &source_entries {
            let path = self.bond_store_dir.join(bsst_filename(entry.file_uid));
            let reader = BondSstReader::open_with_key(&path, self.key.clone())?;
            readers.push(reader);
        }

        // ---- 5. Build per-source iterators + seed the heap. ----
        let mut iters: Vec<BondSstRangeIter<'_>> = readers.iter().map(|r| r.iter_all()).collect();

        let mut heap: BinaryHeap<MergeCursor> = BinaryHeap::with_capacity(plan.source_uids.len());
        for (idx, it) in iters.iter_mut().enumerate() {
            if let Some(next) = it.next() {
                let (s, p, o, payload) = next?;
                heap.push(MergeCursor {
                    key: (s, p, o),
                    payload,
                    source_idx: idx,
                });
                _record_heap_size(heap.len());
            }
        }

        // ---- 6. Construct the writer. ----
        //
        // BondSstWriter::create opens `<final_path>.tmp.<uid>`
        // internally; finish() fsyncs + renames to `<final_path>`.
        let out_filename = bsst_filename(out_uid);
        let out_path = self.bond_store_dir.join(&out_filename);
        let opts = BondSstWriteOpts {
            file_uid: out_uid,
            ..Default::default()
        };
        let mut writer =
            BondSstWriter::create_with_key(&out_path, opts, self.key.clone())?;

        // ---- 7. Streaming k-way merge loop. ----
        let mut entries_in: u64 = 0;
        let mut entries_out: u64 = 0;
        let mut entries_dropped_tombstone: u64 = 0;
        let mut entries_dropped_overwritten: u64 = 0;

        // We coalesce same-triple duplicates across multiple inputs
        // (dedup -- higher seq wins). Pop the head, then repeatedly
        // check whether the next head is the SAME triple; if so,
        // pop it and compare seqs; the loser is dropped, the winner
        // becomes the "current best". After the duplicate run ends
        // we apply tombstone GC + writer.append to the winner.
        while let Some(first) = heap.pop() {
            entries_in += 1;
            // Advance the source we just popped from.
            advance_source(&mut iters, &mut heap, first.source_idx)?;
            _record_heap_size(heap.len());

            let best_key = first.key;
            let mut best_payload = first.payload;
            let mut best_source_idx = first.source_idx;

            // Coalesce: any same-triple head from another source.
            while let Some(peek) = heap.peek() {
                if peek.key != best_key {
                    break;
                }
                let next = heap.pop().expect("peek succeeded; pop must too");
                entries_in += 1;
                advance_source(&mut iters, &mut heap, next.source_idx)?;
                _record_heap_size(heap.len());

                // Higher seq wins. Equal seq is not expected in
                // production (seq is allocated monotonically by the
                // memtable); on a tie the earlier-listed source
                // wins, which conventionally is the younger input.
                if next.payload.seq > best_payload.seq {
                    entries_dropped_overwritten += 1;
                    best_payload = next.payload;
                    best_source_idx = next.source_idx;
                } else {
                    // next loses (same or lower seq).
                    entries_dropped_overwritten += 1;
                }
            }

            // ---- Tombstone GC decision. ----
            //
            // Drop a tombstone iff (a) the payload carries the
            // 0xFF tombstone confidence marker AND (b) the source
            // SST that produced it has an epoch strictly below the
            // gc_barrier. The "source" is the source the winner
            // came from (post-dedup), because that is the
            // tombstone's actual provenance. Otherwise the
            // tombstone is carried into the output with the
            // output's new epoch (manifest-side).
            if best_payload.confidence == BOND_TOMBSTONE_CONFIDENCE
                && source_epochs[best_source_idx] < gc_barrier
            {
                entries_dropped_tombstone += 1;
                continue;
            }

            // ---- Append to output. ----
            writer.append(best_key.0, best_key.1, best_key.2, &best_payload)?;
            entries_out += 1;
        }

        // ---- 8. Finalise the output (fsync + rename). ----
        let stats = writer.finish()?;

        // ---- 9. fsync directory so the rename hits disk. ----
        if let Ok(dir) = std::fs::File::open(&self.bond_store_dir) {
            let _ = dir.sync_all();
        }

        // ---- 9b. Phase 3.6 (Q3.0.B): build the paired reverse-
        //          index file. ----
        //
        // The primary is final on disk. Re-open it, project every
        // entry to (object, predicate, subject) order, and write
        // the projection to `<reverse_uid>.bsst.reverse` in the
        // SAME BondSST wire format -- field semantics differ by
        // which file the caller opened (see module docs, "Same
        // wire format, swapped field semantics").
        //
        // RSS bound: the projection Vec materialises every
        // compacted bond at once -- ~54 B/entry plus inline
        // keyframes, ~1 GB resident peak at 18M bonds, ~2.7 MB at
        // a 50K-entry merge. One-pass-with-buffer chosen over a
        // two-pass streaming re-merge (O(k) RSS but 2x input I/O)
        // -- see module docs for the trade-off.
        let reverse_uid = manifest.allocate_bond_v5_sst_uid();
        let reverse_path = self
            .bond_store_dir
            .join(bsst_reverse_filename(reverse_uid));
        {
            let primary_reader = BondSstReader::open_with_key(&out_path, self.key.clone())?;
            let mut projection: Vec<(NodeId, u8, NodeId, BondPayload)> =
                Vec::with_capacity(stats.entries_written as usize);
            for item in primary_reader.iter_all() {
                let (subject, predicate, object, payload) = item?;
                // The projection swaps subject <-> object.
                projection.push((object, predicate, subject, payload));
            }
            projection.sort_unstable_by_key(|e| (e.0, e.1, e.2));

            // BondSstWriter::create opens
            // `<reverse_path>.tmp.<reverse_uid>` internally (i.e.
            // `<uid>.bsst.reverse.tmp.<uid>`); finish() fsyncs +
            // renames to the final `.bsst.reverse` name -- the
            // same atomic discipline as the primary. Post-dedup
            // primary triples are unique, so the swapped triples
            // are unique too and the writer's strict-monotonic
            // append invariant holds by construction.
            let reverse_opts = BondSstWriteOpts {
                file_uid: reverse_uid,
                ..Default::default()
            };
            let mut reverse_writer =
                BondSstWriter::create_with_key(&reverse_path, reverse_opts, self.key.clone())?;
            for (object, predicate, subject, payload) in &projection {
                reverse_writer.append(*object, *predicate, *subject, payload)?;
            }
            let reverse_stats = reverse_writer.finish()?;
            debug_assert_eq!(
                reverse_stats.entries_written, stats.entries_written,
                "reverse projection entry-count drift: primary {}, reverse {}",
                stats.entries_written, reverse_stats.entries_written,
            );
        }

        // ---- 9c. fsync directory again so the reverse rename
        //          hits disk. ----
        if let Ok(dir) = std::fs::File::open(&self.bond_store_dir) {
            let _ = dir.sync_all();
        }

        // ---- 10. Build the output BondSSTEntry. ----
        //
        // If the merge produced zero entries (every input row was a
        // tombstone that got GC'd), the writer still finishes with
        // entry_count = 0 and first_key/last_key = None. The
        // manifest accepts a zero-entry SST -- the next compaction
        // can retire it -- but in practice the scheduler should not
        // schedule a compaction that produces an empty output. We
        // document and tolerate it with a 0..0 sentinel range.
        //
        // Unlike the node-side writer, BondSstWriteStats DOES
        // expose first_key/last_key, so the manifest range is
        // exact -- no input-range widening needed. (Phase 2.7 had
        // to widen because NodeSstWriteStats lacked the keys;
        // Phase 3.8.2 closed that gap and the node compactor now
        // records exact stats bounds too.)
        let (first_subject, last_subject) = match (stats.first_key, stats.last_key) {
            (Some(f), Some(l)) => ((f.0).0, (l.0).0),
            _ => (0u64, 0u64),
        };

        let output_entry = BondSSTEntry {
            file_uid: out_uid,
            first_subject,
            last_subject,
            entry_count: stats.entries_written,
            file_size: stats.file_size_bytes,
            epoch: out_epoch,
            // Phase 3.6 (Q3.0.B): every compaction output carries
            // a paired reverse-index file, built in step 9b above.
            // L0 (flush-born) SSTs keep None -- reverse queries
            // against L0 fall through to a scan (Phase 3.7's read
            // path).
            reverse_file_uid: Some(reverse_uid),
        };

        // ---- 11. Add the output FIRST. ----
        //
        // ADD-then-RETIRE: `add_bond_v5_sst` requires
        // `target_level <= bond_v5_levels.len()` and
        // `retire_bond_v5_ssts` drops empty trailing levels. A
        // retire-first order would shrink bond_v5_levels and then
        // reject the add (Phase 2.7 caught this on the node side).
        // Order also matches crash-safety: failing between add and
        // retire leaves the manifest with BOTH inputs and output,
        // which is internally consistent; the disk-side orphan is
        // the cleanup pass's problem.
        if let Err(e) = manifest.add_bond_v5_sst(plan.target_level, output_entry.clone()) {
            return Err(BondCompactionError::Manifest(e));
        }

        // ---- 12. Retire the inputs. ----
        //
        // If retirement fails (programmer error -- the source uids
        // came straight from a manifest lookup above) we roll the
        // output back out so the manifest is not left with both
        // input and output simultaneously. The output uid was
        // freshly allocated; retiring the just-added single entry
        // cannot fail.
        let input_entries_retired = match manifest.retire_bond_v5_ssts(&plan.source_uids) {
            Ok(v) => v,
            Err(e) => {
                let _ = manifest.retire_bond_v5_ssts(&[output_entry.file_uid]);
                return Err(BondCompactionError::Manifest(e));
            }
        };

        // ---- 12.5 OP-BRAIN-V5 Phase 3.8.1: evict retired readers. ----
        //
        // Both halves of each retired entry: the primary `.bsst`
        // reader AND the `.bsst.reverse` reader (Phase 3.6) when
        // the entry carries one. Eviction runs AFTER the retire
        // commits so a failed retire leaves the cache intact.
        if let Some(cache) = reader_cache {
            for entry in &input_entries_retired {
                cache.evict(entry.file_uid);
                if let Some(rev_uid) = entry.reverse_file_uid {
                    cache.evict_reverse(rev_uid);
                }
            }
        }

        let elapsed = started.elapsed();

        Ok(CompactionResult {
            output_entry,
            input_entries_retired,
            entries_in,
            entries_out,
            entries_dropped_tombstone,
            entries_dropped_overwritten,
            elapsed,
        })
    }
}

/// Pull the next item from source `idx` and push it onto the heap
/// if one exists. The merge loop calls this every time it pops a
/// cursor for source `idx`. Iterator errors propagate as
/// [`BondCompactionError::BondSst`].
fn advance_source<'a>(
    iters: &mut [BondSstRangeIter<'a>],
    heap: &mut BinaryHeap<MergeCursor>,
    idx: usize,
) -> Result<(), BondCompactionError> {
    if let Some(next) = iters[idx].next() {
        let (s, p, o, payload) = next?;
        heap.push(MergeCursor {
            key: (s, p, o),
            payload,
            source_idx: idx,
        });
    }
    Ok(())
}
// ----------------------------------------------------------------------
// Tests.
// ----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bond::{KeyframeSource, TemporalKeyframe};
    use crate::bond_store::flush::{cleanup_orphan_bond_ssts, BondFlushWorker};
    use crate::manifest::KeyFingerprint;
    use tempfile::tempdir;

    fn mk_payload(seq: u64, weight: u8, confidence: u8) -> BondPayload {
        BondPayload {
            weight,
            confidence,
            keyframes: vec![TemporalKeyframe {
                timestamp: seq as i64,
                weight: 0.5,
                source: KeyframeSource::Baseline,
            }],
            seq,
        }
    }

    /// Keyframe-free payload for the high-volume RSS test.
    fn mk_payload_lean(seq: u64) -> BondPayload {
        BondPayload {
            weight: 128,
            confidence: 1,
            keyframes: Vec::new(),
            seq,
        }
    }

    /// Build live bonds with subjects drawn from `subjects`,
    /// predicate 0, object = subject + 1_000_000, strictly ascending
    /// (subject, predicate, object) order.
    fn build_bonds(
        subjects: std::ops::Range<u64>,
        base_seq: u64,
    ) -> Vec<(NodeId, u8, NodeId, BondPayload)> {
        subjects
            .enumerate()
            .map(|(i, s)| {
                (
                    NodeId(s),
                    0u8,
                    NodeId(s + 1_000_000),
                    mk_payload(base_seq + i as u64, 200, 1),
                )
            })
            .collect()
    }

    /// Adapter that converts owned bonds into the borrow-shaped
    /// iterator `BondFlushWorker::flush` requires.
    fn as_borrow_iter<'a>(
        owned: &'a [(NodeId, u8, NodeId, BondPayload)],
    ) -> impl Iterator<Item = (NodeId, u8, NodeId, &'a BondPayload)> + 'a {
        owned.iter().map(|(s, p, o, pay)| (*s, *p, *o, pay))
    }

    /// Flush `owned` and commit the entry into the manifest at L0.
    fn flush_and_commit(
        worker: &BondFlushWorker,
        manifest: &mut Manifest,
        owned: &[(NodeId, u8, NodeId, BondPayload)],
    ) -> BondSSTEntry {
        let entry = worker
            .flush(as_borrow_iter(owned), manifest)
            .unwrap()
            .expect("non-empty flush");
        manifest.add_bond_v5_sst(0, entry.clone()).unwrap();
        entry
    }

    /// Synthetic manifest-only v5 BondSSTEntry (no backing file --
    /// compact() only opens files for plan.source_uids, so a
    /// barrier-raising sibling entry never needs disk presence).
    fn mk_bond_v5_entry(file_uid: u64, epoch: u64) -> BondSSTEntry {
        BondSSTEntry {
            file_uid,
            first_subject: 0,
            last_subject: u64::MAX - 1,
            entry_count: 1,
            file_size: 1024,
            epoch,
            reverse_file_uid: None,
        }
    }

    /// Collect every (subject, predicate, object, payload) from an
    /// output BondSST.
    fn read_all(path: &Path) -> Vec<(NodeId, u8, NodeId, BondPayload)> {
        let reader = BondSstReader::open(path).unwrap();
        reader
            .iter_all()
            .collect::<Result<Vec<_>, _>>()
            .expect("output SST readable")
    }

    // ------------------------------------------------------------------
    // A. Two-input compaction: 100 entries total, both inputs retired,
    //    output at target_level, contents readable via BondSstReader.
    // ------------------------------------------------------------------
    #[test]
    fn phase34_test_a_two_input_compaction() {
        let dir = tempdir().unwrap();
        let bond_store_dir = dir.path().join("bond_store");
        let flush = BondFlushWorker::new(bond_store_dir.clone());
        let mut manifest = Manifest::new(KeyFingerprint::none());

        let a = flush_and_commit(&flush, &mut manifest, &build_bonds(0..50, 0));
        let b = flush_and_commit(&flush, &mut manifest, &build_bonds(50..100, 100));

        let worker = BondCompactionWorker::new(bond_store_dir.clone());
        let plan = CompactionPlan {
            source_uids: vec![a.file_uid, b.file_uid],
            source_level: 0,
            target_level: 1,
        };
        let result = worker.compact(plan, &mut manifest, None).unwrap();

        assert_eq!(result.entries_in, 100);
        assert_eq!(result.entries_out, 100);
        assert_eq!(result.entries_dropped_tombstone, 0);
        assert_eq!(result.entries_dropped_overwritten, 0);
        assert_eq!(result.output_entry.entry_count, 100);
        assert_eq!(result.output_entry.first_subject, 0);
        assert_eq!(result.output_entry.last_subject, 99);
        assert_eq!(result.input_entries_retired.len(), 2);

        // Both inputs retired.
        assert!(!manifest.live_bond_v5_sst_uids().contains(&a.file_uid));
        assert!(!manifest.live_bond_v5_sst_uids().contains(&b.file_uid));
        // Output live at L1.
        let out_live = manifest
            .iter_bond_v5_ssts()
            .find(|(_, e)| e.file_uid == result.output_entry.file_uid);
        assert!(out_live.is_some(), "output not live in manifest");
        assert_eq!(out_live.unwrap().0 as usize, 1);

        // Spot check via the reader: every subject 0..100 present
        // exactly once, in ascending order.
        let out_path = bond_store_dir.join(bsst_filename(result.output_entry.file_uid));
        let rows = read_all(&out_path);
        assert_eq!(rows.len(), 100);
        for (i, (s, p, o, _)) in rows.iter().enumerate() {
            assert_eq!(s.0, i as u64, "subject order drift at {}", i);
            assert_eq!(*p, 0u8);
            assert_eq!(o.0, i as u64 + 1_000_000);
        }
    }

    // ------------------------------------------------------------------
    // B. Dedup: same triple in multiple inputs -> higher seq wins.
    // ------------------------------------------------------------------
    #[test]
    fn phase34_test_b_dedup_higher_seq_wins() {
        let dir = tempdir().unwrap();
        let bond_store_dir = dir.path().join("bond_store");
        let flush = BondFlushWorker::new(bond_store_dir.clone());
        let mut manifest = Manifest::new(KeyFingerprint::none());

        // Input A: triple (5, 3, 10) at seq=1, weight 10.
        let owned_a = vec![(NodeId(5), 3u8, NodeId(10), mk_payload(1, 10, 1))];
        let a = flush_and_commit(&flush, &mut manifest, &owned_a);

        // Input B: SAME triple (5, 3, 10) at seq=2, weight 20.
        let owned_b = vec![(NodeId(5), 3u8, NodeId(10), mk_payload(2, 20, 1))];
        let b = flush_and_commit(&flush, &mut manifest, &owned_b);

        let worker = BondCompactionWorker::new(bond_store_dir.clone());
        let plan = CompactionPlan {
            source_uids: vec![a.file_uid, b.file_uid],
            source_level: 0,
            target_level: 1,
        };
        let result = worker.compact(plan, &mut manifest, None).unwrap();

        assert_eq!(result.entries_in, 2);
        assert_eq!(result.entries_out, 1);
        assert_eq!(result.entries_dropped_overwritten, 1);

        let out_path = bond_store_dir.join(bsst_filename(result.output_entry.file_uid));
        let rows = read_all(&out_path);
        assert_eq!(rows.len(), 1, "exactly one entry for the triple");
        let (s, p, o, payload) = &rows[0];
        assert_eq!((s.0, *p, o.0), (5u64, 3u8, 10u64));
        assert_eq!(payload.seq, 2, "higher seq must win");
        assert_eq!(payload.weight, 20, "payload from higher-seq entry");
    }

    // ------------------------------------------------------------------
    // C. Tombstone GC HAPPENS when source_epoch < gc_barrier.
    // ------------------------------------------------------------------
    #[test]
    fn phase34_test_c_tombstone_gc_when_barrier_above() {
        let dir = tempdir().unwrap();
        let bond_store_dir = dir.path().join("bond_store");
        let flush = BondFlushWorker::new(bond_store_dir.clone());
        let mut manifest = Manifest::new(KeyFingerprint::none());

        // Input A at epoch 0 (first allocate_epoch returns 0):
        // one live bond (5, 0, 9) + one tombstone (7, 0, 10).
        let owned = vec![
            (NodeId(5), 0u8, NodeId(9), mk_payload(1, 200, 1)),
            (
                NodeId(7),
                0u8,
                NodeId(10),
                mk_payload(2, 0, BOND_TOMBSTONE_CONFIDENCE),
            ),
        ];
        let a = flush_and_commit(&flush, &mut manifest, &owned);
        assert_eq!(a.epoch, 0);

        // Synthetic sibling v5 bond SST at epoch 5 raises the
        // barrier; the input itself is excluded from the query.
        manifest.add_bond_v5_sst(0, mk_bond_v5_entry(999, 5)).unwrap();
        assert_eq!(
            manifest.min_bond_v5_sst_epoch_excluding(&[a.file_uid]),
            Some(5)
        );

        let worker = BondCompactionWorker::new(bond_store_dir.clone());
        let plan = CompactionPlan {
            source_uids: vec![a.file_uid],
            source_level: 0,
            target_level: 1,
        };
        let result = worker.compact(plan, &mut manifest, None).unwrap();

        assert_eq!(result.entries_in, 2);
        assert_eq!(result.entries_out, 1, "live bond kept, tombstone GC'd");
        assert_eq!(result.entries_dropped_tombstone, 1);

        let out_path = bond_store_dir.join(bsst_filename(result.output_entry.file_uid));
        let rows = read_all(&out_path);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].0, NodeId(5), "only the live bond survives");
        assert!(
            !rows.iter().any(|(s, p, o, _)| (*s, *p, *o) == (NodeId(7), 0u8, NodeId(10))),
            "GC'd tombstone must not be in output"
        );
    }

    // ------------------------------------------------------------------
    // D. Tombstone NOT GC'd when barrier too low.
    // ------------------------------------------------------------------
    #[test]
    fn phase34_test_d_tombstone_preserved_when_barrier_low() {
        let dir = tempdir().unwrap();
        let bond_store_dir = dir.path().join("bond_store");
        let flush = BondFlushWorker::new(bond_store_dir.clone());
        let mut manifest = Manifest::new(KeyFingerprint::none());

        let owned = vec![(
            NodeId(7),
            0u8,
            NodeId(10),
            mk_payload(2, 0, BOND_TOMBSTONE_CONFIDENCE),
        )];
        let a = flush_and_commit(&flush, &mut manifest, &owned);
        assert_eq!(a.epoch, 0);

        // Synthetic sibling at epoch 0 (same as the input's).
        // 0 < 0 is false, so the tombstone is NOT GC'd.
        manifest.add_bond_v5_sst(0, mk_bond_v5_entry(999, 0)).unwrap();
        assert_eq!(
            manifest.min_bond_v5_sst_epoch_excluding(&[a.file_uid]),
            Some(0)
        );

        let worker = BondCompactionWorker::new(bond_store_dir.clone());
        let plan = CompactionPlan {
            source_uids: vec![a.file_uid],
            source_level: 0,
            target_level: 1,
        };
        let result = worker.compact(plan, &mut manifest, None).unwrap();

        assert_eq!(result.entries_in, 1);
        assert_eq!(result.entries_out, 1, "tombstone preserved");
        assert_eq!(result.entries_dropped_tombstone, 0);

        let out_path = bond_store_dir.join(bsst_filename(result.output_entry.file_uid));
        let rows = read_all(&out_path);
        assert_eq!(rows.len(), 1);
        let (s, p, o, payload) = &rows[0];
        assert_eq!((s.0, *p, o.0), (7u64, 0u8, 10u64));
        assert_eq!(
            payload.confidence, BOND_TOMBSTONE_CONFIDENCE,
            "preserved entry must still be a tombstone"
        );
    }

    // ------------------------------------------------------------------
    // E. Streaming RSS bound -- heap size never exceeds
    //    source_uids.len(). 5 inputs * 10_000 entries = 50K total.
    // ------------------------------------------------------------------
    #[test]
    fn phase34_test_e_streaming_heap_bounded() {
        let dir = tempdir().unwrap();
        let bond_store_dir = dir.path().join("bond_store");
        let flush = BondFlushWorker::new(bond_store_dir.clone());
        let mut manifest = Manifest::new(KeyFingerprint::none());

        // 5 inputs * 10_000 entries each. Subject ranges are
        // disjoint so no dedup happens; this exercises the merge
        // loop in its hot path.
        let mut source_uids = Vec::new();
        for s in 0..5u64 {
            let lo = s * 10_000;
            let owned: Vec<(NodeId, u8, NodeId, BondPayload)> = (lo..lo + 10_000)
                .map(|subj| {
                    (
                        NodeId(subj),
                        0u8,
                        NodeId(subj + 1_000_000),
                        mk_payload_lean(s * 100_000 + subj),
                    )
                })
                .collect();
            let e = flush_and_commit(&flush, &mut manifest, &owned);
            source_uids.push(e.file_uid);
        }

        _reset_heap_observer();

        let worker = BondCompactionWorker::new(bond_store_dir.clone());
        let plan = CompactionPlan {
            source_uids: source_uids.clone(),
            source_level: 0,
            target_level: 1,
        };
        let result = worker.compact(plan, &mut manifest, None).unwrap();

        assert_eq!(result.entries_in, 50_000);
        assert_eq!(result.entries_out, 50_000);

        let peak = _peek_max_heap_size();
        assert!(
            peak <= source_uids.len(),
            "heap exceeded O(k) bound: peak={}, k={}",
            peak,
            source_uids.len()
        );
        // Sanity: the heap should have actually been used.
        assert!(peak > 0, "heap observer was never updated");
    }

    // ------------------------------------------------------------------
    // F. Crash BEFORE manifest commit -> cleanup removes the output,
    //    inputs preserved on disk + in manifest.
    // ------------------------------------------------------------------
    #[test]
    fn phase34_test_f_crash_before_commit_output_orphaned() {
        let dir = tempdir().unwrap();
        let bond_store_dir = dir.path().join("bond_store");
        let flush = BondFlushWorker::new(bond_store_dir.clone());
        let mut manifest = Manifest::new(KeyFingerprint::none());

        let a = flush_and_commit(&flush, &mut manifest, &build_bonds(0..20, 0));
        let b = flush_and_commit(&flush, &mut manifest, &build_bonds(20..40, 100));

        // Simulate the pre-commit crash by running the worker on a
        // CLONE of the manifest and then THROWING the clone away.
        // The original `manifest` still references both inputs, no
        // output -- exactly the state the cleanup pass would see at
        // restart (the on-disk MANIFEST never saw the mutation).
        let mut manifest_clone = manifest.clone();
        let worker = BondCompactionWorker::new(bond_store_dir.clone());
        let plan = CompactionPlan {
            source_uids: vec![a.file_uid, b.file_uid],
            source_level: 0,
            target_level: 1,
        };
        let result = worker.compact(plan, &mut manifest_clone, None).unwrap();
        drop(manifest_clone);

        // On-disk: A.bsst, B.bsst, AND the output file. The original
        // manifest does not know about the output. Cleanup should
        // remove the output, leave A+B.
        let out_path = bond_store_dir.join(bsst_filename(result.output_entry.file_uid));
        assert!(out_path.exists(), "output file on disk pre-cleanup");

        let report = cleanup_orphan_bond_ssts(&bond_store_dir, &manifest).unwrap();
        assert_eq!(report.orphans_removed, 1, "output is the only primary orphan");
        // Phase 3.6: the output's paired reverse file is orphaned
        // in the same crash window and removed alongside it.
        assert_eq!(report.reverse_orphans_removed, 1, "paired reverse orphaned too");
        assert!(!out_path.exists(), "orphan output removed");

        // Inputs intact on disk and in the manifest.
        assert!(bond_store_dir.join(bsst_filename(a.file_uid)).exists());
        assert!(bond_store_dir.join(bsst_filename(b.file_uid)).exists());
        assert!(manifest.live_bond_v5_sst_uids().contains(&a.file_uid));
        assert!(manifest.live_bond_v5_sst_uids().contains(&b.file_uid));
    }

    // ------------------------------------------------------------------
    // G. Crash AFTER manifest commit (save_atomic done) -> inputs
    //    become orphans, output preserved.
    // ------------------------------------------------------------------
    #[test]
    fn phase34_test_g_crash_after_commit_inputs_orphaned() {
        let dir = tempdir().unwrap();
        let bond_store_dir = dir.path().join("bond_store");
        let manifest_path = dir.path().join("MANIFEST");
        let flush = BondFlushWorker::new(bond_store_dir.clone());
        let mut manifest = Manifest::new(KeyFingerprint::none());

        let a = flush_and_commit(&flush, &mut manifest, &build_bonds(0..20, 0));
        let b = flush_and_commit(&flush, &mut manifest, &build_bonds(20..40, 100));

        let worker = BondCompactionWorker::new(bond_store_dir.clone());
        let plan = CompactionPlan {
            source_uids: vec![a.file_uid, b.file_uid],
            source_level: 0,
            target_level: 1,
        };
        let result = worker.compact(plan, &mut manifest, None).unwrap();

        // Commit durably, then "crash" (drop the in-memory state)
        // and reload from disk.
        manifest.save_atomic(&manifest_path).unwrap();
        drop(manifest);
        let reloaded = Manifest::load(&manifest_path)
            .unwrap()
            .expect("manifest reload");

        let report = cleanup_orphan_bond_ssts(&bond_store_dir, &reloaded).unwrap();
        assert_eq!(
            report.orphans_removed, 2,
            "both inputs orphaned (== source_uids.len())"
        );

        // Output preserved.
        let out_path = bond_store_dir.join(bsst_filename(result.output_entry.file_uid));
        assert!(out_path.exists(), "output preserved after commit");
        // Phase 3.6: the paired reverse file is durably referenced
        // via reverse_file_uid (round-tripped through save_atomic +
        // load) and preserved too.
        assert_eq!(report.reverse_orphans_removed, 0);
        let reverse_path = bond_store_dir.join(bsst_reverse_filename(
            result.output_entry.reverse_file_uid.unwrap(),
        ));
        assert!(reverse_path.exists(), "reverse preserved after commit");
        // Inputs gone.
        assert!(!bond_store_dir.join(bsst_filename(a.file_uid)).exists());
        assert!(!bond_store_dir.join(bsst_filename(b.file_uid)).exists());
    }

    // ------------------------------------------------------------------
    // H. Round-trip: compact -> save -> load -> read output.
    // ------------------------------------------------------------------
    #[test]
    fn phase34_test_h_round_trip_save_load_read() {
        let dir = tempdir().unwrap();
        let bond_store_dir = dir.path().join("bond_store");
        let manifest_path = dir.path().join("MANIFEST");
        let flush = BondFlushWorker::new(bond_store_dir.clone());
        let mut manifest = Manifest::new(KeyFingerprint::none());

        let a = flush_and_commit(&flush, &mut manifest, &build_bonds(0..30, 0));
        let b = flush_and_commit(&flush, &mut manifest, &build_bonds(30..60, 100));

        let worker = BondCompactionWorker::new(bond_store_dir.clone());
        let plan = CompactionPlan {
            source_uids: vec![a.file_uid, b.file_uid],
            source_level: 0,
            target_level: 1,
        };
        let result = worker.compact(plan, &mut manifest, None).unwrap();
        manifest.save_atomic(&manifest_path).unwrap();

        let reloaded = Manifest::load(&manifest_path)
            .unwrap()
            .expect("manifest loads");
        let out_uid = result.output_entry.file_uid;
        assert!(reloaded.live_bond_v5_sst_uids().contains(&out_uid));
        assert!(!reloaded.live_bond_v5_sst_uids().contains(&a.file_uid));
        assert!(!reloaded.live_bond_v5_sst_uids().contains(&b.file_uid));

        let out_path = bond_store_dir.join(bsst_filename(out_uid));
        let reader = BondSstReader::open(&out_path).unwrap();
        for subj in [0u64, 5, 15, 29, 30, 45, 59] {
            let mut iter = reader.iter_subject(NodeId(subj));
            let row = iter
                .next()
                .unwrap_or_else(|| panic!("subject {} missing post-reload", subj))
                .unwrap();
            assert_eq!(row.0, NodeId(subj));
            assert_eq!(row.2, NodeId(subj + 1_000_000));
        }
    }

    // ------------------------------------------------------------------
    // I. Empty inputs -> InvalidPlan.
    // ------------------------------------------------------------------
    #[test]
    fn phase34_test_i_empty_plan_rejected() {
        let dir = tempdir().unwrap();
        let bond_store_dir = dir.path().join("bond_store");
        std::fs::create_dir_all(&bond_store_dir).unwrap();
        let mut manifest = Manifest::new(KeyFingerprint::none());

        let worker = BondCompactionWorker::new(bond_store_dir.clone());
        let plan = CompactionPlan {
            source_uids: vec![],
            source_level: 0,
            target_level: 1,
        };
        let err = worker.compact(plan, &mut manifest, None).unwrap_err();
        match err {
            BondCompactionError::InvalidPlan(_) => {}
            other => panic!("expected InvalidPlan, got {:?}", other),
        }
    }

    // ------------------------------------------------------------------
    // J. min_bond_v5_sst_epoch_excluding correctness.
    // ------------------------------------------------------------------
    #[test]
    fn phase34_test_j_min_bond_v5_epoch_excluding_correctness() {
        let mut manifest = Manifest::new(KeyFingerprint::none());

        // Zero v5 bond SSTs -> None.
        assert_eq!(manifest.min_bond_v5_sst_epoch_excluding(&[]), None);

        // Three v5 bond SSTs at epochs 5, 8, 12 (two at L0, one at L1
        // -- the query spans levels).
        manifest.add_bond_v5_sst(0, mk_bond_v5_entry(1, 5)).unwrap();
        manifest.add_bond_v5_sst(0, mk_bond_v5_entry(2, 8)).unwrap();
        manifest.add_bond_v5_sst(1, mk_bond_v5_entry(3, 12)).unwrap();
        assert_eq!(manifest.min_bond_v5_sst_epoch_excluding(&[]), Some(5));

        // Exclude the lowest -> next-lowest wins.
        assert_eq!(manifest.min_bond_v5_sst_epoch_excluding(&[1]), Some(8));
        // Exclude two -> highest remains.
        assert_eq!(manifest.min_bond_v5_sst_epoch_excluding(&[1, 2]), Some(12));
        // Exclude all -> None.
        assert_eq!(manifest.min_bond_v5_sst_epoch_excluding(&[1, 2, 3]), None);
    }

    // ------------------------------------------------------------------
    // K. SourceNotFound when uid is not live in the manifest.
    // ------------------------------------------------------------------
    #[test]
    fn phase34_test_k_source_not_found() {
        let dir = tempdir().unwrap();
        let bond_store_dir = dir.path().join("bond_store");
        std::fs::create_dir_all(&bond_store_dir).unwrap();
        let mut manifest = Manifest::new(KeyFingerprint::none());

        let worker = BondCompactionWorker::new(bond_store_dir.clone());
        let plan = CompactionPlan {
            source_uids: vec![999],
            source_level: 0,
            target_level: 1,
        };
        let err = worker.compact(plan, &mut manifest, None).unwrap_err();
        match err {
            BondCompactionError::SourceNotFound(999) => {}
            other => panic!("expected SourceNotFound(999), got {:?}", other),
        }
    }

    // ------------------------------------------------------------------
    // L. Duplicate uids in the same plan -> InvalidPlan.
    // ------------------------------------------------------------------
    #[test]
    fn phase34_test_l_duplicate_plan_uids_rejected() {
        let dir = tempdir().unwrap();
        let bond_store_dir = dir.path().join("bond_store");
        std::fs::create_dir_all(&bond_store_dir).unwrap();
        let mut manifest = Manifest::new(KeyFingerprint::none());

        let worker = BondCompactionWorker::new(bond_store_dir.clone());
        let plan = CompactionPlan {
            source_uids: vec![7, 7],
            source_level: 0,
            target_level: 1,
        };
        let err = worker.compact(plan, &mut manifest, None).unwrap_err();
        match err {
            BondCompactionError::InvalidPlan(_) => {}
            other => panic!("expected InvalidPlan, got {:?}", other),
        }
    }

    // ==================================================================
    // Phase 3.6 (Q3.0.B) -- reverse-index emission at compaction.
    // ==================================================================

    // ------------------------------------------------------------------
    // 3.6-A. Reverse file created on compaction: output entry pairs
    //        Some(reverse_uid), file exists on disk, manifest reports
    //        the uid via live_bond_v5_reverse_uids().
    // ------------------------------------------------------------------
    #[test]
    fn phase36_test_a_reverse_file_created_on_compaction() {
        let dir = tempdir().unwrap();
        let bond_store_dir = dir.path().join("bond_store");
        let flush = BondFlushWorker::new(bond_store_dir.clone());
        let mut manifest = Manifest::new(KeyFingerprint::none());

        let a = flush_and_commit(&flush, &mut manifest, &build_bonds(0..50, 0));
        let b = flush_and_commit(&flush, &mut manifest, &build_bonds(50..100, 100));

        let worker = BondCompactionWorker::new(bond_store_dir.clone());
        let plan = CompactionPlan {
            source_uids: vec![a.file_uid, b.file_uid],
            source_level: 0,
            target_level: 1,
        };
        let result = worker.compact(plan, &mut manifest, None).unwrap();

        let reverse_uid = result
            .output_entry
            .reverse_file_uid
            .expect("compaction output must pair a reverse file");
        let reverse_path = bond_store_dir.join(bsst_reverse_filename(reverse_uid));
        assert!(reverse_path.exists(), "<reverse_uid>.bsst.reverse on disk");
        assert_eq!(manifest.live_bond_v5_reverse_uids(), vec![reverse_uid]);

        // The LIVE manifest entry carries the same pairing (compact
        // committed the entry with reverse_file_uid already set --
        // not patched in after the fact).
        let (lvl, live_entry) = manifest
            .iter_bond_v5_ssts()
            .find(|(_, e)| e.file_uid == result.output_entry.file_uid)
            .expect("output live in manifest");
        assert_eq!(lvl as usize, 1);
        assert_eq!(live_entry.reverse_file_uid, Some(reverse_uid));

        // The reverse uid came off the same allocator as primaries
        // and is distinct from the output's own file_uid.
        assert_ne!(reverse_uid, result.output_entry.file_uid);
    }

    // ------------------------------------------------------------------
    // 3.6-B. Reverse file is sort-by-object correct. Objects DESCEND
    //        as subjects ascend in one input so the projection sort
    //        genuinely reorders; every reverse row maps back to a
    //        primary row with subject/object swapped.
    // ------------------------------------------------------------------
    #[test]
    fn phase36_test_b_reverse_sorted_by_object_and_swap_correct() {
        let dir = tempdir().unwrap();
        let bond_store_dir = dir.path().join("bond_store");
        let flush = BondFlushWorker::new(bond_store_dir.clone());
        let mut manifest = Manifest::new(KeyFingerprint::none());

        // Input A: subjects 0..50, object = 10_000 - s (objects
        // strictly DESCEND while subjects ascend).
        let owned_a: Vec<(NodeId, u8, NodeId, BondPayload)> = (0..50u64)
            .map(|s| (NodeId(s), 0u8, NodeId(10_000 - s), mk_payload(s + 1, 200, 1)))
            .collect();
        let a = flush_and_commit(&flush, &mut manifest, &owned_a);

        // Input B: subjects 100..150, object = 20_000 + s, mixed
        // predicates.
        let owned_b: Vec<(NodeId, u8, NodeId, BondPayload)> = (100..150u64)
            .map(|s| {
                (
                    NodeId(s),
                    (s % 3) as u8,
                    NodeId(20_000 + s),
                    mk_payload(s + 1_000, 200, 1),
                )
            })
            .collect();
        let b = flush_and_commit(&flush, &mut manifest, &owned_b);

        let worker = BondCompactionWorker::new(bond_store_dir.clone());
        let plan = CompactionPlan {
            source_uids: vec![a.file_uid, b.file_uid],
            source_level: 0,
            target_level: 1,
        };
        let result = worker.compact(plan, &mut manifest, None).unwrap();
        let reverse_uid = result.output_entry.reverse_file_uid.unwrap();

        let primary_rows =
            read_all(&bond_store_dir.join(bsst_filename(result.output_entry.file_uid)));
        let reverse_rows = read_all(&bond_store_dir.join(bsst_reverse_filename(reverse_uid)));
        assert_eq!(reverse_rows.len(), primary_rows.len());
        assert_eq!(reverse_rows.len(), 100);

        // (a) Strictly ascending (field_a, predicate, field_b) --
        //     where field_a IS the original object and field_b IS
        //     the original subject.
        for w in reverse_rows.windows(2) {
            let ka = (w[0].0, w[0].1, w[0].2);
            let kb = (w[1].0, w[1].1, w[1].2);
            assert!(
                ka < kb,
                "reverse rows not strictly ascending: {:?} !< {:?}",
                ka,
                kb
            );
        }

        // (b) Swap check: (field_b, predicate, field_a, seq) of
        //     every reverse row must be a (subject, predicate,
        //     object, seq) of exactly one primary row; counts match.
        use std::collections::HashSet;
        let primary_set: HashSet<(u64, u8, u64, u64)> = primary_rows
            .iter()
            .map(|(s, p, o, pay)| (s.0, *p, o.0, pay.seq))
            .collect();
        assert_eq!(primary_set.len(), primary_rows.len(), "primary triples unique");
        for (field_a, p, field_b, pay) in &reverse_rows {
            assert!(
                primary_set.contains(&(field_b.0, *p, field_a.0, pay.seq)),
                "reverse row ({}, {}, {}) seq={} has no swapped primary counterpart",
                field_a.0,
                p,
                field_b.0,
                pay.seq,
            );
        }
    }

    // ------------------------------------------------------------------
    // 3.6-C. Orphan .bsst.reverse cleaned at restart (manifest never
    //        references uid 999).
    // ------------------------------------------------------------------
    #[test]
    fn phase36_test_c_reverse_orphan_cleaned_at_restart() {
        let dir = tempdir().unwrap();
        let bond_store_dir = dir.path().join("bond_store");
        std::fs::create_dir_all(&bond_store_dir).unwrap();

        let stray = bond_store_dir.join(bsst_reverse_filename(999));
        std::fs::write(&stray, b"orphan reverse from a crashed compaction").unwrap();

        let manifest = Manifest::new(KeyFingerprint::none());
        let report = cleanup_orphan_bond_ssts(&bond_store_dir, &manifest).unwrap();
        assert_eq!(report.reverse_orphans_removed, 1);
        assert_eq!(report.orphans_removed, 0, "not counted as a primary orphan");
        assert_eq!(report.tmps_removed, 0);
        assert_eq!(report.reverse_tmps_removed, 0);
        assert!(!stray.exists(), "orphan reverse file deleted");
    }

    // ------------------------------------------------------------------
    // 3.6-D. `.bsst.reverse.tmp*` files cleaned unconditionally at
    //        restart.
    // ------------------------------------------------------------------
    #[test]
    fn phase36_test_d_reverse_tmp_cleaned_at_restart() {
        let dir = tempdir().unwrap();
        let bond_store_dir = dir.path().join("bond_store");
        std::fs::create_dir_all(&bond_store_dir).unwrap();

        // Match the writer's actual tmp shape for a reverse target:
        // `<final>.tmp.<uid>` == `7.bsst.reverse.tmp.7`.
        let stray_tmp = bond_store_dir.join("7.bsst.reverse.tmp.7");
        std::fs::write(&stray_tmp, b"interrupted reverse build").unwrap();

        let manifest = Manifest::new(KeyFingerprint::none());
        let report = cleanup_orphan_bond_ssts(&bond_store_dir, &manifest).unwrap();
        assert_eq!(report.reverse_tmps_removed, 1);
        assert_eq!(report.tmps_removed, 0, "not counted as a primary tmp");
        assert!(!stray_tmp.exists());
    }

    // ------------------------------------------------------------------
    // 3.6-E. live_bond_v5_reverse_uids: Some entries included
    //        (ascending), None entries excluded.
    // ------------------------------------------------------------------
    #[test]
    fn phase36_test_e_live_bond_v5_reverse_uids_correctness() {
        let mut manifest = Manifest::new(KeyFingerprint::none());
        let mut e1 = mk_bond_v5_entry(1, 0);
        e1.reverse_file_uid = Some(102);
        let mut e2 = mk_bond_v5_entry(2, 1);
        e2.reverse_file_uid = Some(101);
        let e3 = mk_bond_v5_entry(3, 2); // reverse_file_uid: None
        manifest.add_bond_v5_sst(0, e1).unwrap();
        manifest.add_bond_v5_sst(0, e3).unwrap();
        manifest.add_bond_v5_sst(1, e2).unwrap();

        // The two Some uids, sorted ascending (insertion order was
        // 102 before 101); the None entry is excluded.
        assert_eq!(manifest.live_bond_v5_reverse_uids(), vec![101, 102]);
    }

    // ------------------------------------------------------------------
    // 3.6-F. compact() returning Err(InvalidPlan) leaves no reverse
    //        artifacts (no .bsst.reverse, no .bsst.reverse.tmp*).
    // ------------------------------------------------------------------
    #[test]
    fn phase36_test_f_invalid_plan_leaves_no_reverse_artifacts() {
        let dir = tempdir().unwrap();
        let bond_store_dir = dir.path().join("bond_store");
        std::fs::create_dir_all(&bond_store_dir).unwrap();
        let mut manifest = Manifest::new(KeyFingerprint::none());

        let worker = BondCompactionWorker::new(bond_store_dir.clone());
        let plan = CompactionPlan {
            source_uids: vec![],
            source_level: 0,
            target_level: 1,
        };
        let err = worker.compact(plan, &mut manifest, None).unwrap_err();
        match err {
            BondCompactionError::InvalidPlan(_) => {}
            other => panic!("expected InvalidPlan, got {:?}", other),
        }

        // No reverse files OR reverse tmps left lying around.
        let leftovers: Vec<String> = std::fs::read_dir(&bond_store_dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.contains(".bsst.reverse"))
            .collect();
        assert!(leftovers.is_empty(), "leaked reverse artifacts: {:?}", leftovers);
    }

    // ------------------------------------------------------------------
    // 3.6-G. Round-trip via reverse query: flush -> compact -> open
    //        the reverse file -> iter_range over one object's key
    //        space -> all bonds with that object, payloads intact.
    // ------------------------------------------------------------------
    #[test]
    fn phase36_test_g_round_trip_reverse_query() {
        let dir = tempdir().unwrap();
        let bond_store_dir = dir.path().join("bond_store");
        let flush = BondFlushWorker::new(bond_store_dir.clone());
        let mut manifest = Manifest::new(KeyFingerprint::none());

        // Input A: ten subjects 0..10 ALL bonded to object 777 --
        // the reverse query target.
        let owned_a: Vec<(NodeId, u8, NodeId, BondPayload)> = (0..10u64)
            .map(|s| (NodeId(s), 0u8, NodeId(777), mk_payload(s + 1, 200, 1)))
            .collect();
        let a = flush_and_commit(&flush, &mut manifest, &owned_a);

        // Input B: noise bonds at unrelated objects.
        let owned_b: Vec<(NodeId, u8, NodeId, BondPayload)> = (100..110u64)
            .map(|s| (NodeId(s), 0u8, NodeId(9_000 + s), mk_payload(s, 200, 1)))
            .collect();
        let b = flush_and_commit(&flush, &mut manifest, &owned_b);

        let worker = BondCompactionWorker::new(bond_store_dir.clone());
        let plan = CompactionPlan {
            source_uids: vec![a.file_uid, b.file_uid],
            source_level: 0,
            target_level: 1,
        };
        let result = worker.compact(plan, &mut manifest, None).unwrap();
        let reverse_uid = result.output_entry.reverse_file_uid.unwrap();

        // Reverse query: all bonds with object 777 == iter_range
        // over (777, 0, 0)..(778, 0, 0) in the reverse file's key
        // space.
        let reverse_path = bond_store_dir.join(bsst_reverse_filename(reverse_uid));
        let reader = BondSstReader::open(&reverse_path).unwrap();
        let rows: Vec<(NodeId, u8, NodeId, BondPayload)> = reader
            .iter_range((NodeId(777), 0, NodeId(0)), (NodeId(778), 0, NodeId(0)))
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(rows.len(), 10, "all ten subjects found via the object");
        for (i, (obj, p, subj, payload)) in rows.iter().enumerate() {
            assert_eq!(obj.0, 777, "field_a is the original object");
            assert_eq!(*p, 0u8);
            assert_eq!(subj.0, i as u64, "field_b is the original subject, ascending");
            assert_eq!(payload.seq, i as u64 + 1, "payload round-trips the projection");
            assert_eq!(payload.weight, 200);
        }

        // Noise objects are reachable too but disjoint -- spot
        // check one.
        let noise: Vec<_> = reader
            .iter_range((NodeId(9_100), 0, NodeId(0)), (NodeId(9_101), 0, NodeId(0)))
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(noise.len(), 1);
        assert_eq!(noise[0].2, NodeId(100));
    }
}
// ======================================================================
// OP-COMPACT: bond-store major-compaction (same-level L1 collapse).
//
// Mirror of the node-store proof: collapse ALL overlapping L1 BondSSTs
// into ONE L1 segment via the proven `compact()` engine
// (source_level == target_level == 1), leaving an already-occupied L2
// untouched. Additionally asserts the Phase 3.6 paired reverse-index
// is re-emitted for the collapsed output (recall via the reverse
// traversal path stays intact) and that (subject,predicate,object)
// dedup keeps the highest-seq copy with ZERO triple loss.
// ======================================================================
#[cfg(test)]
mod op_compact_bond_same_level_tests {
    use super::{BondCompactionWorker, CompactionPlan};
    use crate::bond::{KeyframeSource, TemporalKeyframe};
    use crate::bond_store::BondPayload;
    use crate::bond_store::flush::{bsst_filename, bsst_reverse_filename, BondFlushWorker};
    use crate::bond_store::sst_reader::BondSstReader;
    use crate::bond_store::BondSstReaderCache;
    use crate::manifest::{KeyFingerprint, Manifest};
    use crate::node_intern::NodeId;
    use std::collections::BTreeMap;
    use std::path::Path;
    use tempfile::tempdir;

    fn mk_payload(seq: u64) -> BondPayload {
        BondPayload {
            weight: 200,
            confidence: 1,
            keyframes: vec![TemporalKeyframe { timestamp: seq as i64, weight: 0.5, source: KeyframeSource::Baseline }],
            seq,
        }
    }

    /// subject=s, predicate=0, object=s+1_000_000, seq=base_seq+i.
    fn build_bonds(subjects: std::ops::Range<u64>, base_seq: u64) -> Vec<(NodeId, u8, NodeId, BondPayload)> {
        subjects.enumerate()
            .map(|(i, s)| (NodeId(s), 0u8, NodeId(s + 1_000_000), mk_payload(base_seq + i as u64)))
            .collect()
    }

    fn flush_l0(worker: &BondFlushWorker, m: &mut Manifest, owned: &[(NodeId, u8, NodeId, BondPayload)]) {
        let it = owned.iter().map(|(s, p, o, pay)| (*s, *p, *o, pay));
        let entry = worker.flush(it, m).unwrap().expect("non-empty flush");
        m.add_bond_v5_sst(0, entry).unwrap();
    }

    /// Flush one L0 then drain all L0 into a new L1 segment. Returns L1 uid.
    fn build_one_l1(bond_dir: &Path, m: &mut Manifest, owned: &[(NodeId, u8, NodeId, BondPayload)]) -> u64 {
        let flush = BondFlushWorker::new(bond_dir.to_path_buf());
        flush_l0(&flush, m, owned);
        let l0: Vec<u64> = m.bond_v5_levels.iter().find(|l| l.level == 0)
            .map(|l| l.ssts.iter().map(|s| s.file_uid).collect()).unwrap_or_default();
        let w = BondCompactionWorker::new(bond_dir.to_path_buf());
        w.compact(CompactionPlan { source_uids: l0, source_level: 0, target_level: 1 }, m, None)
            .unwrap().output_entry.file_uid
    }

    fn level_count(m: &Manifest, level: u8) -> usize {
        m.bond_v5_levels.iter().find(|l| l.level == level).map(|l| l.ssts.len()).unwrap_or(0)
    }

    fn read_all(path: &Path) -> BTreeMap<(u64, u8, u64), u64> {
        let r = BondSstReader::open(path).unwrap();
        r.iter_all().collect::<Result<Vec<_>, _>>().unwrap().into_iter()
            .map(|(s, p, o, pay)| ((s.0, p, o.0), pay.seq)).collect()
    }

    #[test]
    fn same_level_l1_collapse_bond_lossless_dedup_reverse_intact() {
        let dir = tempdir().unwrap();
        let bond_dir = dir.path().join("bond_store");
        std::fs::create_dir_all(&bond_dir).unwrap();
        let mut m = Manifest::new(KeyFingerprint::none());
        let worker = BondCompactionWorker::new(bond_dir.clone());

        // Occupied L2: subjects 0..30 (seq 0..29).
        let seg0 = build_one_l1(&bond_dir, &mut m, &build_bonds(0..30, 0));
        let l2 = worker
            .compact(CompactionPlan { source_uids: vec![seg0], source_level: 1, target_level: 2 }, &mut m, None)
            .unwrap().output_entry.file_uid;
        assert_eq!(level_count(&m, 1), 0);
        assert_eq!(level_count(&m, 2), 1);
        let l2_path = bond_dir.join(bsst_filename(l2));
        let l2_before = read_all(&l2_path);

        // Overlapping 3-segment L1: subjects 0..10 re-emitted with rising
        // seq in each segment (duplicate triples) + disjoint unique tails.
        let mut s2 = build_bonds(0..10, 100); s2.extend(build_bonds(30..35, 130));
        let mut s3 = build_bonds(0..10, 200); s3.extend(build_bonds(35..40, 230));
        let mut s4 = build_bonds(0..10, 300); s4.extend(build_bonds(40..45, 330));
        build_one_l1(&bond_dir, &mut m, &s2);
        build_one_l1(&bond_dir, &mut m, &s3);
        build_one_l1(&bond_dir, &mut m, &s4);
        assert_eq!(level_count(&m, 1), 3, "three overlapping L1 bond segments");
        let total_before = m.bond_v5_levels.iter().map(|l| l.ssts.len()).sum::<usize>();

        // COLLAPSE all L1 -> single L1.
        let l1_uids: Vec<u64> = m.bond_v5_levels.iter().find(|l| l.level == 1).unwrap()
            .ssts.iter().map(|s| s.file_uid).collect();
        let cache = BondSstReaderCache::new(bond_dir.clone());
        let res = worker
            .compact(CompactionPlan { source_uids: l1_uids, source_level: 1, target_level: 1 }, &mut m, Some(&cache))
            .expect("bond same-level L1 collapse must succeed");

        // Structure.
        assert_eq!(level_count(&m, 1), 1, "L1 collapsed to one segment");
        assert_eq!(level_count(&m, 2), 1, "L2 untouched");
        let l1_entry = m.bond_v5_levels.iter().find(|l| l.level == 1).unwrap().ssts[0].clone();
        assert_eq!(m.bond_v5_levels.iter().find(|l| l.level == 2).unwrap().ssts[0].file_uid, l2);
        assert!(m.bond_v5_levels.iter().map(|l| l.ssts.len()).sum::<usize>() < total_before);
        // subjects 0..10 duplicated across 3 segments -> >=20 dedup drops.
        assert!(res.entries_dropped_overwritten >= 20,
            "expected >=20 triple dedup drops, got {}", res.entries_dropped_overwritten);

        // Reverse index re-emitted for the collapsed output.
        let rev_uid = l1_entry.reverse_file_uid.expect("collapsed L1 output must carry a reverse index");
        assert!(bond_dir.join(bsst_reverse_filename(rev_uid)).exists(), "reverse file present on disk");

        // Losslessness: distinct (s,p,o) set preserved; hub triples 0..10
        // keep the highest seq (from s4: 300..309); L2-only triples intact.
        let l1_after = read_all(&bond_dir.join(bsst_filename(l1_entry.file_uid)));
        for s in 0..10u64 {
            assert_eq!(l1_after.get(&(s, 0, s + 1_000_000)), Some(&(300 + s)),
                "hub triple {} must keep highest seq", s);
        }
        for s in 30..45u64 {
            assert!(l1_after.contains_key(&(s, 0, s + 1_000_000)), "unique tail {} preserved", s);
        }
        // L2 unchanged on disk -> subjects 10..30 still reachable there.
        assert_eq!(read_all(&l2_path), l2_before, "L2 segment byte-stable");
        for s in 10..30u64 {
            assert!(l2_before.contains_key(&(s, 0, s + 1_000_000)), "L2-only subject {} intact", s);
        }
    }
}
