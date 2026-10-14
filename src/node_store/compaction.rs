//! `NodeCompactionWorker` -- streaming k-way merge over NodeSST files
//! with epoch-barrier tombstone GC.
//!
//! OP-BRAIN-V5 Phase 2.7 (Q7). The node-side compactor: takes N
//! input NodeSSTs at one LSM level, merges them into a single
//! output NodeSST at a higher level, drops tombstones that no
//! older SST could still reference, and commits the swap to the
//! manifest via the Phase 2.5 helpers
//! (`allocate_node_sst_uid`, `allocate_epoch`, `add_node_sst`,
//! `retire_node_ssts`).
//!
//! ## Scope boundary (NEW node-store compaction, NOT v4 bond)
//!
//! Phase 2.7 ships compaction for the NodeSST store ONLY. The
//! v4 `src/compaction.rs` (bond compaction with the two
//! HashMap-the-world antipatterns -- `rows: Vec<MergeRow>` at line
//! 357 and `node_winners: HashMap` at line 362) is NOT touched
//! here. Q7's directive to replace both with streaming heap merges
//! applies to the v5 BOND store, which lands in Phase 3. This
//! module is the load-bearing reference implementation -- the bond
//! port will mirror the cursor shape, the GC barrier semantics,
//! and the atomic commit sequence below.
//!
//! ## Streaming k-way merge -- RSS bound is O(k), NOT O(n)
//!
//! The merge is implemented as a [`BinaryHeap`] (min-heap by
//! [`NodeId`]) of [`MergeCursor`] entries -- ONE cursor per input
//! SST. Each `pop` yields the lowest NodeId across all inputs;
//! the merge then advances THAT cursor by pulling the next item
//! from `iters[source_idx]` and pushes the new head back.
//!
//! The heap NEVER holds more than `plan.source_uids.len()`
//! cursors. Total in-memory state during the merge is bounded by:
//!
//! - `k` open [`NodeSstReader`] handles (each carries a small block
//!   cache, capped by `DEFAULT_BLOCK_CACHE_CAP`),
//! - `k` heap cursors (each holds one [`NodeEntry`] -- ~1.8 KiB),
//! - one in-flight `NodeEntry` being written to the output.
//!
//! The entire output is streamed through [`NodeSstWriter::append`]
//! one entry at a time. NO `Vec<(NodeId, NodeEntry)>` is ever
//! materialised. This is the Q7 invariant that the v4 bond
//! compactor violates and that this port exists to demonstrate.
//!
//! ## Dedup -- higher `seq` wins
//!
//! When two cursors point at the SAME [`NodeId`], the merge loop
//! pops them both, keeps the entry with the strictly higher
//! [`NodeEntry::seq`], and discards the other. This matches the
//! [`NodeMemTable`] insert discipline (see
//! `src/node_store/memtable.rs::NodeMemTable::insert` -- \"higher
//! `seq` wins\") and means the merge output retains the
//! latest-write-wins property of the LSM layer.
//!
//! ## Tombstone GC -- epoch barrier
//!
//! A [`NodeEntry::tombstone`] entry must be preserved across
//! compaction iff some other live SST might still reference the
//! older payload it shadows. "Reference" in the LSM sense means
//! "an older SST whose epoch is `<` the tombstone's source-SST
//! epoch could still surface the pre-tombstone value at read
//! time."
//!
//! The barrier is the minimum epoch we must keep tombstones for:
//!
//! ```text
//!   gc_barrier = min(
//!       Manifest::min_bond_sst_epoch,
//!       Manifest::min_node_sst_epoch_excluding(&plan.source_uids),
//!   )
//!```
//! 
//! `None` from either query is treated as `u64::MAX` ("no SST in
//! this store could possibly reference the tombstone"). If BOTH
//! are `None`, the effective barrier is `u64::MAX` and every
//! tombstone is safe to drop.
//!
//! Drop rule:
//!
//! ```text
//!   tombstone.tombstone == true && source_epoch < gc_barrier
//! ```
//!
//! Otherwise the tombstone is carried into the output. Note the
//! carried tombstone INHERITS the output SST's new epoch from the
//! reader's perspective (the `NodeSSTEntry.epoch` recorded in the
//! manifest), even though the on-disk [`NodeEntry`] payload is
//! unchanged. This is correct because the manifest is the
//! authority on "what epoch is this SST stamped with"; a future
//! compaction that consults `source_epoch` for THIS entry will
//! find the OUTPUT's epoch.
//!
//! Mental model: drop a tombstone only when no SST that could
//! reference the value it shadows still exists.
//!
//! ## Atomic commit sequence
//!
//! ```text
//!   1.  Query min_bond_sst_epoch + min_node_sst_epoch_excluding
//!       -> gc_barrier
//!   2.  manifest.allocate_node_sst_uid() -> out_uid
//!   3.  manifest.allocate_epoch()        -> out_epoch
//!   4.  Open all source readers; look up each input's NodeSSTEntry
//!       to recover its source_epoch (for the GC rule).
//!   5.  Build the streaming merge iterator (RSS-bounded heap).
//!   6.  NodeSstWriter::create(<out_uid>.nsst, opts) -- writer opens
//!       the .tmp.<uid> file.
//!   7.  for each (NodeId, NodeEntry, source_idx) from merge:
//!         if entry.tombstone && source_epochs[source_idx] < gc_barrier {
//!             drop  // GC'd
//!         } else if dedup -> older seq drops
//!         else writer.append(id, &entry)
//!   8.  writer.finish() -> NodeSstWriteStats (fsync + rename).
//!   9.  fsync the node_store/ directory (rename hits disk).
//!   10. Build NodeSSTEntry from stats + out_uid + out_epoch.
//!   11. manifest.add_node_sst(plan.target_level, output_entry).
//!       (add BEFORE retire -- retire shrinks node_levels when
//!       it drops the now-empty source level, which would make
//!       add_node_sst(target_level=1) fail with InvalidLevel
//!       when len() == 0. Add-then-retire is the only legal
//!       order against the Phase 2.5 manifest helpers.)
//!   12. manifest.retire_node_ssts(&plan.source_uids).
//!   13. Caller owns manifest.save_atomic.
//!   14. Source NodeSST files NOT unlinked here. The
//!       `cleanup_orphan_node_ssts` pass owns that after save_atomic.
//! ```
//!
//! ## Crash-window analysis (no data loss possible)
//!
//! - **Crash before step 8 rename:** `.nsst.tmp.<uid>` orphan on
//!   disk; cleanup pass at restart removes it. Inputs intact. The
//!   `next_node_sst_uid` and `next_epoch` bumps in step 2/3 are
//!   in-memory only -- they're lost on restart, but the manifest
//!   counter is monotonic from the on-disk value, so the next
//!   compaction simply allocates the same uid; the orphan is gone
//!   by then.
//!
//! - **Crash between 8 and 11 (rename done, manifest not
//!   mutated):** `<out_uid>.nsst` on disk, no manifest reference.
//!   Cleanup classifies as orphan (file_uid not in
//!   `live_node_sst_uids`) and removes it. Inputs intact. Retry
//!   safe.
//!
//! - **Crash between 11 and 12 / save:** in-memory manifest had
//!   inputs retired but no output added; that state never reaches
//!   disk because `save_atomic` is the durability anchor. On
//!   restart the on-disk manifest still references the inputs;
//!   output is an orphan and gets cleaned up. Retry safe.
//!
//! - **Crash after save_atomic:** the manifest on disk references
//!   the new output and NOT the inputs. The input `.nsst` files
//!   are now orphans; the cleanup pass at next restart removes
//!   them. Output is canonical. No data loss.
//!
//! At every break point the input files survive UNTIL the new
//! output is durably referenced by the manifest. This is the
//! same discipline as the Phase 2.6 flush worker, scaled to
//! N inputs and 1 output instead of 0 inputs and 1 output.
//!
//! ## Not yet wired
//!
//! Phase 2.7 ships the WORKER. Tests construct the
//! [`CompactionPlan`] directly. A scheduler that picks plans by
//! level-fill / size-tier triggers is a later port. No production
//! code path calls [`NodeCompactionWorker::compact`] in this
//! commit.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use crate::manifest::{Manifest, ManifestError, NodeSSTEntry};
use crate::node_intern::NodeId;
use crate::node_store::flush::{nsst_filename, NSST_SUFFIX};
use crate::node_store::memtable::NodeEntry;
use crate::node_store::sst_reader::{NodeSstRangeIter, NodeSstReader};
use crate::node_store::sst_writer::{NodeSstError, NodeSstWriteOpts, NodeSstWriter};

// Re-export so callers do not need a separate `use` of the suffix.
pub use crate::node_store::flush::NSST_SUFFIX as NODE_SST_SUFFIX;
use std::marker::PhantomData;

// Suppress unused-import lints when the helpers above are only
// reached via doc-tests / future ports.
fn _unused() {
    let _ = NSST_SUFFIX;
    let _: PhantomData<NodeSstRangeIter<'static>> = PhantomData;
}

// ----------------------------------------------------------------------
// Errors.
// ----------------------------------------------------------------------

/// Errors surfaced by [`NodeCompactionWorker::compact`].
#[derive(Debug)]
pub enum NodeCompactionError {
    /// `std::io::Error` propagated from open / read / write / fsync /
    /// rename / unlink (NB: this worker does NOT unlink inputs --
    /// the cleanup pass does; an `Io` here means an output-side
    /// failure).
    Io(std::io::Error),
    /// Underlying [`NodeSstError`] from reader/writer.
    NodeSst(NodeSstError),
    /// Plan-shape violation: empty `source_uids`, target_level
    /// implausible against current manifest, etc.
    InvalidPlan(String),
    /// One of the `source_uids` in the plan is not present in any
    /// live node level. The manifest is the authority on liveness;
    /// compaction against a non-existent input is a programmer
    /// error.
    SourceNotFound(u64),
    /// The sources span more than one level, or the target level is
    /// not strictly above the source level. Phase 2.7 ships a
    /// single-level compactor; cross-level merges are a later
    /// scheduler concern.
    LevelMismatch(String),
    /// Manifest helper (`retire_node_ssts`, `add_node_sst`)
    /// rejected the post-merge commit. Always a programmer error.
    Manifest(ManifestError),
}

impl std::fmt::Display for NodeCompactionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NodeCompactionError::Io(e) => write!(f, "NodeCompactionWorker IO: {}", e),
            NodeCompactionError::NodeSst(e) => write!(f, "NodeCompactionWorker NodeSST: {}", e),
            NodeCompactionError::InvalidPlan(s) => write!(f, "NodeCompactionWorker invalid plan: {}", s),
            NodeCompactionError::SourceNotFound(uid) => write!(
                f,
                "NodeCompactionWorker source file_uid {} not live in manifest",
                uid
            ),
            NodeCompactionError::LevelMismatch(s) => write!(
                f,
                "NodeCompactionWorker level mismatch: {}",
                s
            ),
            NodeCompactionError::Manifest(e) => write!(
                f,
                "NodeCompactionWorker manifest: {}",
                e
            ),
        }
    }
}

impl std::error::Error for NodeCompactionError {}

impl From<std::io::Error> for NodeCompactionError {
    fn from(e: std::io::Error) -> Self {
        NodeCompactionError::Io(e)
    }
}

impl From<NodeSstError> for NodeCompactionError {
    fn from(e: NodeSstError) -> Self {
        NodeCompactionError::NodeSst(e)
    }
}

impl From<ManifestError> for NodeCompactionError {
    fn from(e: ManifestError) -> Self {
        NodeCompactionError::Manifest(e)
    }
}

// ----------------------------------------------------------------------
// Plan + result.
// ----------------------------------------------------------------------

/// Compaction job description supplied by the caller.
///
/// `source_uids` lists the [`NodeSSTEntry::file_uid`]s to merge.
/// All sources MUST live at `source_level` (Phase 2.7 ships a
/// single-level compactor; the scheduler decides which level
/// participates per round).
///
/// `target_level` is where the merged output is added via
/// [`Manifest::add_node_sst`]. Conventionally `source_level + 1`
/// but the worker does not enforce that -- a same-level rewrite
/// is also a legal plan (used by future read-amp control).
#[derive(Debug, Clone)]
pub struct CompactionPlan {
    pub source_uids: Vec<u64>,
    pub source_level: usize,
    pub target_level: usize,
}

/// Result returned by a successful [`NodeCompactionWorker::compact`].
#[derive(Debug, Clone)]
pub struct CompactionResult {
    /// New manifest entry now live at `plan.target_level`.
    pub output_entry: NodeSSTEntry,
    /// The retired input entries, in the order they appeared in
    /// `plan.source_uids`. Returned to the caller for diagnostics
    /// and so a future test can verify that retirement matched the
    /// intent.
    pub input_entries_retired: Vec<NodeSSTEntry>,
    /// Total entries read across all inputs (live + tombstone).
    pub entries_in: u64,
    /// Total entries written to the output.
    pub entries_out: u64,
    /// Tombstones dropped by the GC rule.
    pub entries_dropped_tombstone: u64,
    /// Dedup losers (entries shadowed by a higher-seq entry at the
    /// same NodeId in a different input).
    pub entries_dropped_overwritten: u64,
    /// OP-COMPACT-DEFGUARD internal notes: live nodes whose winning
    /// (newest-by-seq) version lacked a Definition but a coalesced older
    /// version of the SAME NodeId carried one, so the gloss was carried
    /// forward onto the winner instead of being silently dropped.
    pub entries_definitions_preserved: u64,
    /// Wall-clock compaction duration (open-to-rename).
    pub elapsed: Duration,
}

// ----------------------------------------------------------------------
// Merge cursor (heap entry).
// ----------------------------------------------------------------------

/// One entry in the min-heap that drives the k-way merge.
///
/// Ordering is reversed because [`BinaryHeap`] is a max-heap by
/// default and we want the smallest [`NodeId`] at the top. The
/// `source_idx` is the index into the parallel `Vec<NodeSstReader>`
/// / `Vec<NodeSstRangeIter>` / `Vec<u64>` (source_epochs); after
/// `pop` the merge loop advances `iters[source_idx]` to get the
/// next entry from THAT source.
#[derive(Debug)]
struct MergeCursor {
    id: NodeId,
    entry: NodeEntry,
    source_idx: usize,
}

impl PartialEq for MergeCursor {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && self.source_idx == other.source_idx
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
        // Primary key: NodeId ascending.
        // Tie-break: source_idx ascending so the heap order is
        // total and reproducible (tests assert on it).
        other
            .id
            .cmp(&self.id)
            .then_with(|| other.source_idx.cmp(&self.source_idx))
    }
}

// ----------------------------------------------------------------------
// Worker.
// ----------------------------------------------------------------------

/// NodeSST compaction worker rooted at a `node_store_dir`.
///
/// Stateless other than the directory path; safe to construct
/// per-compaction or to share across compactions.
pub struct NodeCompactionWorker {
    node_store_dir: PathBuf,
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

impl NodeCompactionWorker {
    /// Construct a worker rooted at `node_store_dir`. The directory
    /// must already exist (compaction is always preceded by a flush
    /// or a load that created the directory).
    pub fn new(node_store_dir: PathBuf) -> Self {
        NodeCompactionWorker { node_store_dir, key: None }
    }

    /// Builder: set the per-SST encryption key (OP-BRAIN-V5 6.5).
    pub fn with_key(mut self, key: Option<crate::encrypt::MasterKey>) -> Self {
        self.key = key;
        self
    }

    /// Read-only view of the configured directory (mostly for tests).
    pub fn node_store_dir(&self) -> &Path {
        &self.node_store_dir
    }

    /// Compact the inputs named by `plan` into a single output
    /// NodeSST at `plan.target_level`.
    ///
    /// See the module docs for the atomic commit sequence and the
    /// tombstone-GC rule. The caller is responsible for
    /// `manifest.save_atomic` after this returns; the worker
    /// mutates the in-memory manifest only.
    ///
    /// OP-BRAIN-V5 Phase 3.8.1: `reader_cache` is the Phase 2.8
    /// `NodeSstReaderCache` to evict retired `file_uid`s from after
    /// the manifest retire commits. `Some` in any wiring where a
    /// live read path shares the store (the Brain's cache);
    /// `None` for scaffolds that never construct a cache. Without
    /// the eviction the cache would keep serving -- and keeping
    /// open -- readers for files the manifest no longer references.
    pub fn compact(
        &self,
        plan: CompactionPlan,
        manifest: &mut Manifest,
        reader_cache: Option<&crate::node_store::NodeSstReaderCache>,
    ) -> Result<CompactionResult, NodeCompactionError> {
        let started = Instant::now();

        // ---- 0. Validate plan shape. ----
        if plan.source_uids.is_empty() {
            return Err(NodeCompactionError::InvalidPlan(
                "plan.source_uids is empty".to_string(),
            ));
        }
        // Disallow duplicate uids in the same plan.
        {
            let mut sorted = plan.source_uids.clone();
            sorted.sort_unstable();
            sorted.dedup();
            if sorted.len() != plan.source_uids.len() {
                return Err(NodeCompactionError::InvalidPlan(
                    "plan.source_uids contains duplicate file_uids".to_string(),
                ));
            }
        }

        // ---- 1. Resolve each source uid to its (level, NodeSSTEntry). ----
        let mut source_entries: Vec<NodeSSTEntry> =
            Vec::with_capacity(plan.source_uids.len());
        for &uid in &plan.source_uids {
            let found = manifest
                .iter_node_ssts()
                .find(|(_, e)| e.file_uid == uid);
            let (lvl, entry) = match found {
                Some((l, e)) => (l as usize, e.clone()),
                None => return Err(NodeCompactionError::SourceNotFound(uid)),
            };
            if lvl != plan.source_level {
                return Err(NodeCompactionError::LevelMismatch(format!(
                    "source file_uid {} lives at level {}, plan.source_level = {}",
                    uid, lvl, plan.source_level
                )));
            }
            source_entries.push(entry);
        }
        let source_epochs: Vec<u64> = source_entries.iter().map(|e| e.epoch).collect();

        // ---- 2. Compute the GC barrier BEFORE mutating the manifest. ----
        //
        // gc_barrier = min(bond_store_min, node_store_min_excluding_inputs).
        // None from either query is treated as u64::MAX (no SST in
        // that store can possibly reference the tombstone). If both
        // are None, the barrier is u64::MAX and every tombstone is
        // safe to drop.
        let bond_min = manifest.min_bond_sst_epoch().unwrap_or(u64::MAX);
        let node_min = manifest
            .min_node_sst_epoch_excluding(&plan.source_uids)
            .unwrap_or(u64::MAX);
        let gc_barrier = bond_min.min(node_min);

        // ---- 3. Allocate output uid + epoch. ----
        //
        // Bumps the in-memory counters. If we crash before
        // save_atomic, the bumps are lost and the next boot reuses
        // the same uid -- but the orphan cleanup pass removes the
        // .nsst file from the failed attempt before we accept
        // writes, so reuse is safe.
        let out_uid = manifest.allocate_node_sst_uid();
        let out_epoch = manifest.allocate_epoch();

        // ---- 4. Open all source readers. ----
        //
        // Vec<NodeSstReader> outlives the iterators we build over
        // them, so the lifetime juggling is clean.
        let mut readers: Vec<NodeSstReader> = Vec::with_capacity(plan.source_uids.len());
        for entry in &source_entries {
            let path = self.node_store_dir.join(nsst_filename(entry.file_uid));
            let reader = NodeSstReader::open_with_key(&path, self.key.clone())?;
            readers.push(reader);
        }

        // ---- 5. Build per-source iterators + seed the heap. ----
        let mut iters: Vec<NodeSstRangeIter<'_>> =
            readers.iter().map(|r| r.iter_all()).collect();

        let mut heap: BinaryHeap<MergeCursor> =
            BinaryHeap::with_capacity(plan.source_uids.len());
        for (idx, it) in iters.iter_mut().enumerate() {
            if let Some(next) = it.next() {
                let (id, entry) = next?;
                heap.push(MergeCursor { id, entry, source_idx: idx });
                _record_heap_size(heap.len());
            }
        }

        // ---- 6. Construct the writer. ----
        let out_filename = nsst_filename(out_uid);
        let out_path = self.node_store_dir.join(&out_filename);
        let opts = NodeSstWriteOpts {
            file_uid: out_uid,
            ..Default::default()
        };
        let mut writer =
            NodeSstWriter::create_with_key(&out_path, opts, self.key.clone())?;

        // ---- 7. Streaming k-way merge loop. ----
        let mut entries_in: u64 = 0;
        let mut entries_out: u64 = 0;
        let mut entries_dropped_tombstone: u64 = 0;
        let mut entries_dropped_overwritten: u64 = 0;
        // OP-COMPACT-DEFGUARD internal notes: Definition glosses carried
        // forward onto a seq-winner that would otherwise have dropped them when
        // a later contentless stub of the same NodeId collapsed in.
        let mut entries_definitions_preserved: u64 = 0;

        // We need to coalesce same-NodeId duplicates across
        // multiple inputs (dedup -- higher seq wins). We do this
        // by popping the head, then repeatedly checking if the
        // next head is the SAME NodeId; if so, pop it and compare
        // seqs; the loser is dropped, the winner becomes the
        // "current best". After the duplicate run ends we apply
        // tombstone GC + writer.append to the winner.
        while let Some(first) = heap.pop() {
            entries_in += 1;
            // Advance the source we just popped from.
            advance_source(&mut iters, &mut heap, first.source_idx, &mut entries_in)?;
            _record_heap_size(heap.len());

            let best_id = first.id;
            let mut best_entry = first.entry;
            let mut best_source_idx = first.source_idx;

            // OP-COMPACT-DEFGUARD internal notes: track the highest-seq
            // LIVE Definition gloss across ALL same-NodeId versions in this
            // coalesce run, so a later contentless / IsA-only stub cannot drop
            // it when we keep newest-by-seq below.
            let mut def_carry: Option<(u64, String)> =
                node_live_definition(&best_entry).map(|d| (best_entry.seq, d.to_string()));

            // Coalesce: any same-NodeId head from another source.
            while let Some(peek) = heap.peek() {
                if peek.id != best_id {
                    break;
                }
                let next = heap.pop().expect("peek succeeded; pop must too");
                entries_in += 1;
                advance_source(&mut iters, &mut heap, next.source_idx, &mut entries_in)?;
                _record_heap_size(heap.len());

                // Higher seq wins. Equal seq: the on-disk LSM layer
                // says the entry from the HIGHER level (older) is
                // shadowed by the LOWER level (newer); we keep the
                // one from the LOWER source_idx because the caller
                // is conventionally expected to list younger inputs
                // first. (For Phase 2.7 ALL inputs are same-level
                // and seq is monotonic per writer, so equal-seq is
                // not expected in production. Tie-break is
                // documented for completeness.)
                // OP-COMPACT-DEFGUARD: remember a LIVE Definition carried by
                // this version before it may be dropped as a dedup loser.
                if let Some(d) = node_live_definition(&next.entry) {
                    let better = def_carry.as_ref().map_or(true, |(sq, _)| next.entry.seq > *sq);
                    if better {
                        def_carry = Some((next.entry.seq, d.to_string()));
                    }
                }
                if next.entry.seq > best_entry.seq {
                    entries_dropped_overwritten += 1;
                    best_entry = next.entry;
                    best_source_idx = next.source_idx;
                } else {
                    // next loses (same or lower seq).
                    entries_dropped_overwritten += 1;
                }
            }

            // OP-COMPACT-DEFGUARD internal notes: if the seq-winner is a
            // LIVE node that does NOT already carry a non-empty Definition but a
            // coalesced version of this same NodeId did, transplant the
            // (highest-seq) gloss onto the winner. The winner keeps its newest
            // metadata (seq / confidence / embedding / timestamps); only the
            // typed content is upgraded from a bare stub back to the stored
            // Definition. Tombstone winners (genuine deletes) are never
            // resurrected.
            if !best_entry.tombstone && node_live_definition(&best_entry).is_none() {
                if let Some((_, gloss)) = def_carry.take() {
                    best_entry.payload.knowledge.content =
                        crate::node::KnowledgeContent::Definition(gloss);
                    entries_definitions_preserved += 1;
                }
            }

            // ---- Tombstone GC decision. ----
            //
            // Drop a tombstone iff the source SST that produced it
            // has an epoch strictly below the gc_barrier. The
            // "source" here is the source the winner came from
            // (post-dedup), because that's the tombstone's actual
            // provenance.
            if best_entry.tombstone && source_epochs[best_source_idx] < gc_barrier {
                entries_dropped_tombstone += 1;
                continue;
            }

            // ---- Append to output. ----
            writer.append(best_id, &best_entry)?;
            entries_out += 1;
        }

        // ---- 8. Finalise the output. ----
        let stats = writer.finish()?;

        // ---- 9. fsync directory so the rename hits disk. ----
        if let Ok(dir) = std::fs::File::open(&self.node_store_dir) {
            let _ = dir.sync_all();
        }

        // ---- 10. Build the output NodeSSTEntry. ----
        //
        // If the merge produced zero entries (every input was
        // tombstoned + GC'd), the writer still finishes with
        // entry_count = 0, blocks = 0, file_size = HEADER+TRAILER.
        // The manifest happily accepts a zero-entry SST -- the
        // next compaction can retire it -- but in practice the
        // scheduler should not schedule a compaction that
        // produces an empty output. We document and tolerate it.
        //
        // first/last NodeId for the manifest entry: EXACT post-GC
        // bounds from the writer's accounting (Phase 3.8.2 --
        // NodeSstWriteStats grew first_node_id/last_node_id,
        // mirroring the bond-side BondSstWriteStats
        // first_key/last_key). Before 3.8.2 this widened to the
        // input hull (min input first / max input last), which was
        // conservative-correct for overlap detection but over-wide
        // whenever tombstone GC dropped entries at the edges.
        // Empty output (every input tombstoned + GC'd): 0..0
        // sentinel, matching the bond-side compactor.
        debug_assert_eq!(
            entries_out == 0,
            stats.first_node_id.is_none(),
            "writer stats first_node_id must be Some iff entries were emitted",
        );
        let (first_id, last_id) = match (stats.first_node_id, stats.last_node_id) {
            (Some(f), Some(l)) => (f.0, l.0),
            _ => (0u64, 0u64),
        };

        let output_entry = NodeSSTEntry {
            file_uid: out_uid,
            first_node_id: first_id,
            last_node_id: last_id,
            entry_count: stats.entries_written,
            file_size: stats.file_size_bytes,
            epoch: out_epoch,
        };

        // ---- 11. Add the output FIRST. ----
        //
        // We add the output before retiring inputs because
        // `add_node_sst` requires `target_level <= node_levels.len()`
        // and `retire_node_ssts` drops empty trailing levels.
        // A retire-first order would shrink node_levels and then
        // reject the add. Order-of-operations also matches
        // crash-safety: if we fail between add and retire, the
        // manifest has BOTH inputs and output -- which on
        // save_atomic + reload looks like "we crashed mid-flight"
        // and the cleanup pass takes care of the orphan output
        // on the disk side; the in-memory manifest stays internally
        // consistent.
        if let Err(e) = manifest.add_node_sst(plan.target_level, output_entry.clone()) {
            return Err(NodeCompactionError::Manifest(e));
        }

        // ---- 12. Retire the inputs. ----
        //
        // If retirement fails (programmer error -- the source uids
        // came straight from a manifest lookup above) we roll the
        // output back out so the manifest is not left with both
        // input and output simultaneously. The output uid was
        // freshly allocated; retire_node_ssts on a single-element
        // slice cannot fail because the entry was just added.
        let input_entries_retired = match manifest.retire_node_ssts(&plan.source_uids) {
            Ok(v) => v,
            Err(e) => {
                let _ = manifest.retire_node_ssts(&[output_entry.file_uid]);
                return Err(NodeCompactionError::Manifest(e));
            }
        };

        // ---- 12.5 OP-BRAIN-V5 Phase 3.8.1: evict retired readers. ----
        //
        // After the retire commits, the reader cache must not keep
        // serving the dead files. Eviction runs AFTER the retire
        // (not before) so a failed retire leaves the cache intact;
        // it is keyed on the same uids the retire consumed.
        if let Some(cache) = reader_cache {
            for uid in &plan.source_uids {
                cache.evict(*uid);
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
            entries_definitions_preserved,
            elapsed,
        })
    }
}

/// OP-COMPACT-DEFGUARD internal notes: if `entry` is a LIVE node whose
/// typed content is a non-empty `Definition`, return the trimmed gloss.
/// Tombstones and non-Definition / empty content return `None`. Used by the
/// compaction coalesce loop so a later contentless / IsA-only stub of the same
/// NodeId cannot silently drop a stored Definition when newest-by-seq wins.
fn node_live_definition(entry: &NodeEntry) -> Option<&str> {
    if entry.tombstone {
        return None;
    }
    match &entry.payload.knowledge.content {
        crate::node::KnowledgeContent::Definition(d) => {
            let t = d.trim();
            if t.is_empty() {
                None
            } else {
                Some(t)
            }
        }
        _ => None,
    }
}

/// Pull the next item from source `idx` and push it onto the heap
/// if one exists. The merge loop calls this every time it pops a
/// cursor for source `idx`. Iterator errors propagate as
/// [`NodeCompactionError::NodeSst`].
///
/// `entries_in` is NOT bumped here -- the merge loop owns that
/// counter since the popped cursor counts as "one in".
fn advance_source<'a>(
    iters: &mut [NodeSstRangeIter<'a>],
    heap: &mut BinaryHeap<MergeCursor>,
    idx: usize,
    _entries_in: &mut u64,
) -> Result<(), NodeCompactionError> {
    if let Some(next) = iters[idx].next() {
        let (id, entry) = next?;
        heap.push(MergeCursor { id, entry, source_idx: idx });
    }
    Ok(())
}

// ----------------------------------------------------------------------
// Tests.
// ----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::{KeyFingerprint, SSTEntry};
    use crate::node::{Confidence, KnowledgeContent, KnowledgeNode, KnowledgeUnit};
    use crate::node_intern::NodeId;
    use crate::node_store::flush::{cleanup_orphan_node_ssts, nsst_filename, NodeFlushWorker};
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

    fn populate(memtable: &mut NodeMemTable, ids: std::ops::Range<u64>, base_seq: u64) {
        for (off, id) in ids.enumerate() {
            memtable.insert(
                NodeId(id),
                mk_node(&format!("node_{}", id)),
                base_seq + off as u64,
            );
        }
    }

    /// Flush `memtable` and commit the entry into the manifest at L0.
    /// Returns the NodeSSTEntry that was added.
    fn flush_and_commit(
        worker: &NodeFlushWorker,
        manifest: &mut Manifest,
        memtable: &NodeMemTable,
    ) -> NodeSSTEntry {
        let entry = worker
            .flush(memtable, manifest)
            .unwrap()
            .expect("non-empty memtable");
        manifest.add_node_sst(0, entry.clone()).unwrap();
        entry
    }

    fn make_bond_sst_entry(id: u64, epoch: u64) -> SSTEntry {
        SSTEntry {
            id,
            file: format!("bond-{}.sst", id),
            subject_min: [0u8; 32],
            subject_max: [0xffu8; 32],
            size: 1024,
            mtime: 1_700_000_000,
            bond_count: 10,
            epoch,
        }
    }

    // ------------------------------------------------------------------
    // A. Two-input compaction: 100 entries total, both inputs retired,
    //    output at target_level, contents readable via NodeSstReader.
    // ------------------------------------------------------------------
    #[test]
    fn phase27_test_a_two_input_compaction() {
        let dir = tempdir().unwrap();
        let node_store_dir = dir.path().join("node_store");
        let flush = NodeFlushWorker::new(node_store_dir.clone());
        let mut manifest = Manifest::new(KeyFingerprint::none());

        let mut mta = NodeMemTable::new();
        populate(&mut mta, 0..50, 0);
        let a = flush_and_commit(&flush, &mut manifest, &mta);

        let mut mtb = NodeMemTable::new();
        populate(&mut mtb, 50..100, 100);
        let b = flush_and_commit(&flush, &mut manifest, &mtb);

        let worker = NodeCompactionWorker::new(node_store_dir.clone());
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
        assert_eq!(result.input_entries_retired.len(), 2);

        // Both inputs retired from L0.
        assert!(!manifest.live_node_sst_uids().contains(&a.file_uid));
        assert!(!manifest.live_node_sst_uids().contains(&b.file_uid));
        // Output live at L1.
        let out_live = manifest
            .iter_node_ssts()
            .find(|(_, e)| e.file_uid == result.output_entry.file_uid);
        assert!(out_live.is_some(), "output not live in manifest");
        assert_eq!(out_live.unwrap().0 as usize, 1);

        // Spot check 10 NodeIds via the reader.
        let out_path = node_store_dir.join(nsst_filename(result.output_entry.file_uid));
        let reader = NodeSstReader::open(&out_path).unwrap();
        for id in [0u64, 1, 17, 25, 33, 49, 50, 72, 88, 99].iter().copied() {
            let got = reader.get(NodeId(id)).unwrap();
            assert!(got.is_some(), "output missing NodeId {}", id);
            assert!(!got.unwrap().tombstone);
        }
    }

    // ------------------------------------------------------------------
    // B. Dedup: same NodeId in multiple inputs -> higher seq wins.
    // ------------------------------------------------------------------
    #[test]
    fn phase27_test_b_dedup_higher_seq_wins() {
        let dir = tempdir().unwrap();
        let node_store_dir = dir.path().join("node_store");
        let flush = NodeFlushWorker::new(node_store_dir.clone());
        let mut manifest = Manifest::new(KeyFingerprint::none());

        // Input A: NodeId 5 with payload "X", seq 1.
        let mut mta = NodeMemTable::new();
        mta.insert(NodeId(5), mk_node("X"), 1);
        let a = flush_and_commit(&flush, &mut manifest, &mta);

        // Input B: NodeId 5 with payload "Y", seq 2.
        let mut mtb = NodeMemTable::new();
        mtb.insert(NodeId(5), mk_node("Y"), 2);
        let b = flush_and_commit(&flush, &mut manifest, &mtb);

        let worker = NodeCompactionWorker::new(node_store_dir.clone());
        let plan = CompactionPlan {
            source_uids: vec![a.file_uid, b.file_uid],
            source_level: 0,
            target_level: 1,
        };
        let result = worker.compact(plan, &mut manifest, None).unwrap();

        assert_eq!(result.entries_in, 2);
        assert_eq!(result.entries_out, 1);
        assert_eq!(result.entries_dropped_overwritten, 1);

        let out_path = node_store_dir.join(nsst_filename(result.output_entry.file_uid));
        let reader = NodeSstReader::open(&out_path).unwrap();
        let got = reader.get(NodeId(5)).unwrap().expect("NodeId 5 in output");
        assert_eq!(got.seq, 2, "higher seq must win");
        assert_eq!(got.payload.content, "Y", "payload from higher-seq entry");
    }

    // ------------------------------------------------------------------
    // C. Tombstone GC HAPPENS when source_epoch < gc_barrier.
    // ------------------------------------------------------------------
    #[test]
    fn phase27_test_c_tombstone_gc_when_barrier_above() {
        let dir = tempdir().unwrap();
        let node_store_dir = dir.path().join("node_store");
        let flush = NodeFlushWorker::new(node_store_dir.clone());
        let mut manifest = Manifest::new(KeyFingerprint::none());

        // Input A at epoch 0 (the first allocate_epoch returns 0)
        // containing a tombstone for NodeId 7.
        let mut mta = NodeMemTable::new();
        mta.insert(NodeId(7), mk_node("to-delete"), 1);
        mta.tombstone(NodeId(7), 2);
        let a = flush_and_commit(&flush, &mut manifest, &mta);
        assert_eq!(a.epoch, 0);

        // Synthetic bond SSTEntry at epoch=5 so the barrier is high.
        manifest.add_sst(0, make_bond_sst_entry(1, 5));
        assert_eq!(manifest.min_bond_sst_epoch(), Some(5));

        let worker = NodeCompactionWorker::new(node_store_dir.clone());
        let plan = CompactionPlan {
            source_uids: vec![a.file_uid],
            source_level: 0,
            target_level: 1,
        };
        let result = worker.compact(plan, &mut manifest, None).unwrap();

        assert_eq!(result.entries_in, 1);
        assert_eq!(result.entries_out, 0, "tombstone GC'd");
        assert_eq!(result.entries_dropped_tombstone, 1);

        let out_path = node_store_dir.join(nsst_filename(result.output_entry.file_uid));
        let reader = NodeSstReader::open(&out_path).unwrap();
        let got = reader.get(NodeId(7)).unwrap();
        assert!(got.is_none(), "GC'd tombstone must not be in output");
    }

    // ------------------------------------------------------------------
    // D. Tombstone NOT GC'd when barrier too low.
    // ------------------------------------------------------------------
    #[test]
    fn phase27_test_d_tombstone_preserved_when_barrier_low() {
        let dir = tempdir().unwrap();
        let node_store_dir = dir.path().join("node_store");
        let flush = NodeFlushWorker::new(node_store_dir.clone());
        let mut manifest = Manifest::new(KeyFingerprint::none());

        // Input A at epoch 0 containing a tombstone for NodeId 7.
        let mut mta = NodeMemTable::new();
        mta.insert(NodeId(7), mk_node("to-delete"), 1);
        mta.tombstone(NodeId(7), 2);
        let a = flush_and_commit(&flush, &mut manifest, &mta);
        assert_eq!(a.epoch, 0);

        // Synthetic bond SSTEntry at epoch=0 (same as input). 0 < 0
        // is false, so the tombstone is NOT GC'd.
        manifest.add_sst(0, make_bond_sst_entry(1, 0));
        assert_eq!(manifest.min_bond_sst_epoch(), Some(0));

        let worker = NodeCompactionWorker::new(node_store_dir.clone());
        let plan = CompactionPlan {
            source_uids: vec![a.file_uid],
            source_level: 0,
            target_level: 1,
        };
        let result = worker.compact(plan, &mut manifest, None).unwrap();

        assert_eq!(result.entries_in, 1);
        assert_eq!(result.entries_out, 1, "tombstone preserved");
        assert_eq!(result.entries_dropped_tombstone, 0);

        let out_path = node_store_dir.join(nsst_filename(result.output_entry.file_uid));
        let reader = NodeSstReader::open(&out_path).unwrap();
        let got = reader.get(NodeId(7)).unwrap().expect("tombstone in output");
        assert!(got.tombstone, "preserved entry must be a tombstone");
    }

    // ------------------------------------------------------------------
    // E. Streaming RSS bound -- heap size never exceeds source_uids.len().
    // ------------------------------------------------------------------
    #[test]
    fn phase27_test_e_streaming_heap_bounded() {
        let dir = tempdir().unwrap();
        let node_store_dir = dir.path().join("node_store");
        let flush = NodeFlushWorker::new(node_store_dir.clone());
        let mut manifest = Manifest::new(KeyFingerprint::none());

        // 5 inputs * 10_000 entries each. NodeId ranges are disjoint
        // so no dedup happens; this exercises the merge loop in its
        // hot path.
        let mut source_uids = Vec::new();
        for s in 0..5u64 {
            let mut mt = NodeMemTable::new();
            let lo = s * 10_000;
            populate(&mut mt, lo..(lo + 10_000), s * 100_000);
            let e = flush_and_commit(&flush, &mut manifest, &mt);
            source_uids.push(e.file_uid);
        }

        _reset_heap_observer();

        let worker = NodeCompactionWorker::new(node_store_dir.clone());
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
            peak, source_uids.len()
        );
        // Sanity: the heap should have actually been used.
        assert!(peak > 0, "heap observer was never updated");
    }

    // ------------------------------------------------------------------
    // F. Crash BEFORE manifest commit -> cleanup removes the output,
    //    inputs preserved on disk + in manifest.
    // ------------------------------------------------------------------
    #[test]
    fn phase27_test_f_crash_before_commit_output_orphaned() {
        let dir = tempdir().unwrap();
        let node_store_dir = dir.path().join("node_store");
        let flush = NodeFlushWorker::new(node_store_dir.clone());
        let mut manifest = Manifest::new(KeyFingerprint::none());

        let mut mta = NodeMemTable::new();
        populate(&mut mta, 0..20, 0);
        let a = flush_and_commit(&flush, &mut manifest, &mta);

        let mut mtb = NodeMemTable::new();
        populate(&mut mtb, 20..40, 100);
        let b = flush_and_commit(&flush, &mut manifest, &mtb);

        // Simulate the pre-commit crash by running the worker on a
        // CLONE of the manifest and then THROWING the clone away.
        // The original `manifest` still references both inputs, no
        // output -- exactly the state the cleanup pass would see
        // at restart.
        let mut manifest_clone = manifest.clone();
        let worker = NodeCompactionWorker::new(node_store_dir.clone());
        let plan = CompactionPlan {
            source_uids: vec![a.file_uid, b.file_uid],
            source_level: 0,
            target_level: 1,
        };
        let result = worker.compact(plan, &mut manifest_clone, None).unwrap();
        drop(manifest_clone);

        // On-disk: A.nsst, B.nsst, AND the output file. Manifest
        // (the original, not the clone) does not know about the
        // output. Cleanup should remove the output, leave A+B.
        let out_path = node_store_dir.join(nsst_filename(result.output_entry.file_uid));
        assert!(out_path.exists(), "output file on disk pre-cleanup");

        let report = cleanup_orphan_node_ssts(&node_store_dir, &manifest).unwrap();
        assert_eq!(report.orphans_removed, 1, "output is the only orphan");
        assert!(!out_path.exists(), "orphan output removed");

        // Inputs intact.
        assert!(node_store_dir.join(nsst_filename(a.file_uid)).exists());
        assert!(node_store_dir.join(nsst_filename(b.file_uid)).exists());
        assert!(manifest.live_node_sst_uids().contains(&a.file_uid));
        assert!(manifest.live_node_sst_uids().contains(&b.file_uid));
    }

    // ------------------------------------------------------------------
    // G. Crash AFTER manifest commit -> inputs become orphans, output
    //    preserved.
    // ------------------------------------------------------------------
    #[test]
    fn phase27_test_g_crash_after_commit_inputs_orphaned() {
        let dir = tempdir().unwrap();
        let node_store_dir = dir.path().join("node_store");
        let flush = NodeFlushWorker::new(node_store_dir.clone());
        let mut manifest = Manifest::new(KeyFingerprint::none());

        let mut mta = NodeMemTable::new();
        populate(&mut mta, 0..20, 0);
        let a = flush_and_commit(&flush, &mut manifest, &mta);

        let mut mtb = NodeMemTable::new();
        populate(&mut mtb, 20..40, 100);
        let b = flush_and_commit(&flush, &mut manifest, &mtb);

        let worker = NodeCompactionWorker::new(node_store_dir.clone());
        let plan = CompactionPlan {
            source_uids: vec![a.file_uid, b.file_uid],
            source_level: 0,
            target_level: 1,
        };
        let result = worker.compact(plan, &mut manifest, None).unwrap();

        // Simulate save_atomic having happened (the in-memory
        // manifest is what would be on disk). Now the inputs are
        // not referenced; cleanup should orphan them.
        let report = cleanup_orphan_node_ssts(&node_store_dir, &manifest).unwrap();
        assert_eq!(report.orphans_removed, 2, "both inputs orphaned");

        // Output preserved.
        let out_path = node_store_dir.join(nsst_filename(result.output_entry.file_uid));
        assert!(out_path.exists(), "output preserved after commit");
        // Inputs gone.
        assert!(!node_store_dir.join(nsst_filename(a.file_uid)).exists());
        assert!(!node_store_dir.join(nsst_filename(b.file_uid)).exists());
    }

    // ------------------------------------------------------------------
    // H. Round-trip: compact -> save -> load -> read output.
    // ------------------------------------------------------------------
    #[test]
    fn phase27_test_h_round_trip_save_load_read() {
        let dir = tempdir().unwrap();
        let node_store_dir = dir.path().join("node_store");
        let flush = NodeFlushWorker::new(node_store_dir.clone());
        let manifest_path = dir.path().join("MANIFEST");
        let mut manifest = Manifest::new(KeyFingerprint::none());

        let mut mta = NodeMemTable::new();
        populate(&mut mta, 0..30, 0);
        let a = flush_and_commit(&flush, &mut manifest, &mta);

        let mut mtb = NodeMemTable::new();
        populate(&mut mtb, 30..60, 100);
        let b = flush_and_commit(&flush, &mut manifest, &mtb);

        let worker = NodeCompactionWorker::new(node_store_dir.clone());
        let plan = CompactionPlan {
            source_uids: vec![a.file_uid, b.file_uid],
            source_level: 0,
            target_level: 1,
        };
        let result = worker.compact(plan, &mut manifest, None).unwrap();
        manifest.save_atomic(&manifest_path).unwrap();

        let reloaded = Manifest::load(&manifest_path).unwrap().expect("manifest loads");
        let out_uid = result.output_entry.file_uid;
        assert!(reloaded.live_node_sst_uids().contains(&out_uid));

        let out_path = node_store_dir.join(nsst_filename(out_uid));
        let reader = NodeSstReader::open(&out_path).unwrap();
        for id in [0u64, 5, 15, 29, 30, 45, 59].iter().copied() {
            let got = reader.get(NodeId(id)).unwrap();
            assert!(got.is_some(), "NodeId {} missing post-reload", id);
        }
    }

    // ------------------------------------------------------------------
    // I. Empty inputs -> InvalidPlan.
    // ------------------------------------------------------------------
    #[test]
    fn phase27_test_i_empty_plan_rejected() {
        let dir = tempdir().unwrap();
        let node_store_dir = dir.path().join("node_store");
        std::fs::create_dir_all(&node_store_dir).unwrap();
        let mut manifest = Manifest::new(KeyFingerprint::none());

        let worker = NodeCompactionWorker::new(node_store_dir.clone());
        let plan = CompactionPlan {
            source_uids: vec![],
            source_level: 0,
            target_level: 1,
        };
        let err = worker.compact(plan, &mut manifest, None).unwrap_err();
        match err {
            NodeCompactionError::InvalidPlan(_) => {}
            other => panic!("expected InvalidPlan, got {:?}", other),
        }
    }

    // ------------------------------------------------------------------
    // J. min_bond_sst_epoch / min_node_sst_epoch_excluding correctness.
    // ------------------------------------------------------------------
    #[test]
    fn phase27_test_j_min_epoch_correctness() {
        let mut manifest = Manifest::new(KeyFingerprint::none());
        assert_eq!(manifest.min_bond_sst_epoch(), None);

        manifest.add_sst(0, make_bond_sst_entry(1, 5));
        manifest.add_sst(0, make_bond_sst_entry(2, 8));
        manifest.add_sst(1, make_bond_sst_entry(3, 12));
        assert_eq!(manifest.min_bond_sst_epoch(), Some(5));

        // No node SSTs -> None.
        assert_eq!(manifest.min_node_sst_epoch_excluding(&[]), None);

        // Add three node SSTs at epochs 3, 7, 11.
        let mut add_node = |uid: u64, epoch: u64| {
            manifest
                .add_node_sst(
                    0,
                    NodeSSTEntry {
                        file_uid: uid,
                        first_node_id: 0,
                        last_node_id: 10,
                        entry_count: 10,
                        file_size: 1024,
                        epoch,
                    },
                )
                .unwrap();
        };
        add_node(100, 3);
        add_node(101, 7);
        add_node(102, 11);
        assert_eq!(manifest.min_node_sst_epoch_excluding(&[]), Some(3));
        // Exclude the lowest -> next-lowest wins.
        assert_eq!(manifest.min_node_sst_epoch_excluding(&[100]), Some(7));
        // Exclude all -> None.
        assert_eq!(
            manifest.min_node_sst_epoch_excluding(&[100, 101, 102]),
            None
        );
    }

    // ------------------------------------------------------------------
    // K. SourceNotFound when uid is not live in the manifest.
    // ------------------------------------------------------------------
    #[test]
    fn phase27_test_k_source_not_found() {
        let dir = tempdir().unwrap();
        let node_store_dir = dir.path().join("node_store");
        std::fs::create_dir_all(&node_store_dir).unwrap();
        let mut manifest = Manifest::new(KeyFingerprint::none());

        let worker = NodeCompactionWorker::new(node_store_dir.clone());
        let plan = CompactionPlan {
            source_uids: vec![999],
            source_level: 0,
            target_level: 1,
        };
        let err = worker.compact(plan, &mut manifest, None).unwrap_err();
        match err {
            NodeCompactionError::SourceNotFound(999) => {}
            other => panic!("expected SourceNotFound(999), got {:?}", other),
        }
    }

    // ------------------------------------------------------------------
    // Phase 3.8.2 (Tier 2B). Output entry carries the EXACT post-GC
    // range from NodeSstWriteStats, not the input hull. Tombstones at
    // both edges are GC'd (source epoch < barrier), so the hull
    // [0, 99] is strictly wider than the exact range [1, 98].
    // ------------------------------------------------------------------
    #[test]
    fn phase382_compaction_output_range_exact_not_input_hull() {
        let dir = tempdir().unwrap();
        let node_store_dir = dir.path().join("node_store");
        let flush = NodeFlushWorker::new(node_store_dir.clone());
        let mut manifest = Manifest::new(KeyFingerprint::none());

        // Input A: tombstone at the LOW edge (NodeId 0) + live 1..=49.
        let mut mta = NodeMemTable::new();
        mta.tombstone(NodeId(0), 0);
        populate(&mut mta, 1..50, 1);
        let a = flush_and_commit(&flush, &mut manifest, &mta);
        assert_eq!(a.first_node_id, 0);
        assert_eq!(a.last_node_id, 49);

        // Input B: live 50..=98 + tombstone at the HIGH edge (NodeId 99).
        let mut mtb = NodeMemTable::new();
        populate(&mut mtb, 50..99, 100);
        mtb.tombstone(NodeId(99), 200);
        let b = flush_and_commit(&flush, &mut manifest, &mtb);
        assert_eq!(b.first_node_id, 50);
        assert_eq!(b.last_node_id, 99);

        // Raise the GC barrier above both input epochs (0 and 1) so
        // both edge tombstones are dropped at compaction.
        manifest.add_sst(0, make_bond_sst_entry(1, 5));
        assert_eq!(manifest.min_bond_sst_epoch(), Some(5));

        let worker = NodeCompactionWorker::new(node_store_dir.clone());
        let plan = CompactionPlan {
            source_uids: vec![a.file_uid, b.file_uid],
            source_level: 0,
            target_level: 1,
        };
        let result = worker.compact(plan, &mut manifest, None).unwrap();

        assert_eq!(result.entries_in, 100);
        assert_eq!(result.entries_dropped_tombstone, 2);
        assert_eq!(result.entries_out, 98);
        assert_eq!(result.output_entry.entry_count, 98);

        // Input hull is [0, 99]; the EXACT emitted range is [1, 98].
        // Pre-3.8.2 the manifest entry recorded the hull.
        assert_eq!(
            result.output_entry.first_node_id, 1,
            "manifest range must be exact (1), not the input hull (0)"
        );
        assert_eq!(
            result.output_entry.last_node_id, 98,
            "manifest range must be exact (98), not the input hull (99)"
        );
    }
}
// ======================================================================
// OP-COMPACT: major-compaction (same-level L1 collapse) losslessness.
//
// Proves the SAFE major-compaction primitive used to flatten resident
// RAM on the canonical v5 brain: collapse ALL overlapping L1 NodeSSTs
// into a SINGLE L1 segment via the PROVEN `compact()` engine
// (source_level == target_level == 1), leaving the older, already-
// populated L2 untouched and correctly ordered underneath. This is
// the second-major-safe path: a naive L1->L2 merge when L2 is already
// occupied would leave TWO overlapping L2 segments, which the read
// path (single binary-search candidate per level >= 1) cannot resolve
// -> silent recall loss. Same-level collapse never creates a second
// segment at any level, so the L1+ non-overlap invariant holds.
// ======================================================================
#[cfg(test)]
mod op_compact_same_level_tests {
    use super::{CompactionPlan, NodeCompactionWorker};
    use crate::manifest::{KeyFingerprint, Manifest};
    use crate::node::{Confidence, KnowledgeContent, KnowledgeNode, KnowledgeUnit};
    use crate::node_intern::NodeId;
    use crate::node_store::flush::NodeFlushWorker;
    use crate::node_store::memtable::NodeMemTable;
    use crate::node_store::sst_reader::NodeSstReader;
    use crate::node_store::NodeSstReaderCache;
    use crate::seed::Seed;
    use std::path::Path;
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

    /// Flush one memtable of (id, seq, label) into a fresh L0 SST, then
    /// minor-compact ALL current L0 into a new L1 segment. Returns the
    /// L1 output file_uid. Each call adds exactly ONE L1 segment.
    fn build_one_l1_segment(
        node_dir: &Path,
        manifest: &mut Manifest,
        entries: &[(u64, u64, &str)],
    ) -> u64 {
        let flush = NodeFlushWorker::new(node_dir.to_path_buf());
        let mut mt = NodeMemTable::new();
        for (id, seq, label) in entries {
            mt.insert(NodeId(*id), mk_node(label), *seq);
        }
        let l0 = flush.flush(&mt, manifest).unwrap().expect("non-empty memtable");
        manifest.add_node_sst(0, l0).unwrap();
        // Drain ALL live L0 into one new L1 segment.
        let l0_uids: Vec<u64> = manifest
            .node_levels
            .iter()
            .find(|l| l.level == 0)
            .map(|l| l.ssts.iter().map(|s| s.file_uid).collect())
            .unwrap_or_default();
        let worker = NodeCompactionWorker::new(node_dir.to_path_buf());
        let plan = CompactionPlan { source_uids: l0_uids, source_level: 0, target_level: 1 };
        worker.compact(plan, manifest, None).unwrap().output_entry.file_uid
    }

    fn level_sst_count(m: &Manifest, level: u8) -> usize {
        m.node_levels.iter().find(|l| l.level == level).map(|l| l.ssts.len()).unwrap_or(0)
    }

    /// Read a node id honouring level order (L0 -> L1 -> L2 ...),
    /// highest-seq winner within a level; first level with a hit wins
    /// (mirrors the production read path's level-walk semantics).
    fn read_id(node_dir: &Path, m: &Manifest, id: u64) -> Option<(String, u64)> {
        for lvl in &m.node_levels {
            let mut best: Option<(String, u64)> = None;
            for s in &lvl.ssts {
                if s.first_node_id > id || s.last_node_id < id {
                    continue;
                }
                let r = NodeSstReader::open_with_key(&node_dir.join(format!("{}.nsst", s.file_uid)), None).unwrap();
                if let Some(e) = r.get(NodeId(id)).unwrap() {
                    let cand = (e.payload.content.clone(), e.seq);
                    best = match best {
                        None => Some(cand),
                        Some(p) if cand.1 > p.1 => Some(cand),
                        Some(p) => Some(p),
                    };
                }
            }
            if let Some(b) = best {
                return Some(b);
            }
        }
        None
    }

    #[test]
    fn same_level_l1_collapse_is_lossless_and_dedups_with_occupied_l2() {
        let dir = tempdir().unwrap();
        let node_dir = dir.path().join("node_store");
        std::fs::create_dir_all(&node_dir).unwrap();
        let mut m = Manifest::new(KeyFingerprint::none());

        // ---- Build occupied L2: one L1 seg {1,2,3} then promote L1->L2.
        let seg0 = build_one_l1_segment(&node_dir, &mut m,
            &[(1, 10, "id1_v10"), (2, 11, "id2"), (3, 12, "id3")]);
        let worker = NodeCompactionWorker::new(node_dir.clone());
        let l2 = worker
            .compact(CompactionPlan { source_uids: vec![seg0], source_level: 1, target_level: 2 }, &mut m, None)
            .unwrap()
            .output_entry
            .file_uid;
        assert_eq!(level_sst_count(&m, 1), 0, "L1 drained into L2");
        assert_eq!(level_sst_count(&m, 2), 1, "L2 occupied");

        // ---- Rebuild an overlapping 3-segment L1 (hub id 1 re-emitted
        //      with rising seq in every segment + a unique id each).
        build_one_l1_segment(&node_dir, &mut m, &[(1, 20, "id1_v20"), (4, 21, "id4")]);
        build_one_l1_segment(&node_dir, &mut m, &[(1, 30, "id1_v30"), (5, 31, "id5")]);
        build_one_l1_segment(&node_dir, &mut m, &[(1, 40, "id1_v40"), (6, 41, "id6")]);
        assert_eq!(level_sst_count(&m, 1), 3, "three overlapping L1 segments");
        let total_before = m.total_node_sst_count();
        assert_eq!(total_before, 4, "3xL1 + 1xL2");

        // Pre-collapse recall snapshot for every distinct id.
        let ids = [1u64, 2, 3, 4, 5, 6];
        let before: Vec<Option<(String, u64)>> =
            ids.iter().map(|&i| read_id(&node_dir, &m, i)).collect();
        assert_eq!(before[0], Some(("id1_v40".to_string(), 40)), "hub newest seq before");

        // ---- COLLAPSE: merge all L1 -> single L1 (same level).
        let l1_uids: Vec<u64> = m
            .node_levels
            .iter()
            .find(|l| l.level == 1)
            .map(|l| l.ssts.iter().map(|s| s.file_uid).collect())
            .unwrap();
        let cache = NodeSstReaderCache::new(node_dir.clone());
        let res = worker
            .compact(CompactionPlan { source_uids: l1_uids, source_level: 1, target_level: 1 }, &mut m, Some(&cache))
            .expect("same-level L1 collapse must succeed");

        // ---- Structural assertions.
        assert_eq!(level_sst_count(&m, 1), 1, "L1 collapsed to a single segment");
        assert_eq!(level_sst_count(&m, 2), 1, "L2 untouched");
        assert_eq!(m.node_levels.iter().find(|l| l.level == 2).unwrap().ssts[0].file_uid, l2,
            "L2 segment uid unchanged");
        assert!(m.total_node_sst_count() < total_before, "segment count reduced");
        assert!(res.entries_dropped_overwritten >= 2,
            "hub id1 appeared in 3 L1 segments -> >=2 dedup drops, got {}",
            res.entries_dropped_overwritten);

        // ---- Losslessness: every distinct id still resolves, hub wins
        //      newest seq, ids only in L2 still reachable, payloads exact.
        let after: Vec<Option<(String, u64)>> =
            ids.iter().map(|&i| read_id(&node_dir, &m, i)).collect();
        assert_eq!(before, after, "recall byte-identical before vs after collapse");
        assert_eq!(after[0], Some(("id1_v40".to_string(), 40)), "hub newest seq preserved");
        assert_eq!(after[1], Some(("id2".to_string(), 11)), "L2-only id2 preserved");
        assert_eq!(after[2], Some(("id3".to_string(), 12)), "L2-only id3 preserved");
        assert_eq!(after[3], Some(("id4".to_string(), 21)), "id4 preserved");
        assert_eq!(after[5], Some(("id6".to_string(), 41)), "id6 preserved");
    }

    /// OP-COMPACT-DEFGUARD / internal notes REPRODUCTION + REGRESSION.
    ///
    /// Several Definition-carrying concept nodes at a LOWER seq, each shadowed
    /// by a later contentless / IsA-only stub of the SAME NodeId at a HIGHER
    /// seq. Before the fix the k-way merge kept newest-by-seq unconditionally,
    /// so every stub won its collapse and the stored gloss was silently
    /// DROPPED (exactly the breadthD-compaction loss of radioactivity /
    /// radioactive decay / nuclear fission / wave-particle duality glosses).
    /// After the fix the surviving collapsed node retains the Definition
    /// (newest seq metadata kept; gloss carried forward).
    #[test]
    fn definitions_survive_seq_dedup_collapse_op1505() {
        fn mk_def(label: &str, gloss: &str) -> KnowledgeNode {
            KnowledgeNode {
                seed: Seed::from_label(label),
                content: label.to_string(),
                knowledge: KnowledgeUnit {
                    content: KnowledgeContent::Definition(gloss.to_string()),
                    confidence: Confidence::Verified,
                    created_at: 0,
                    updated_at: 0,
                },
                embedding: vec![0.0; 384],
                parent_seeds: Vec::new(),
                typed_ratio: 0.0,
            }
        }
        fn mk_stub(label: &str) -> KnowledgeNode {
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
        // Flush one node into a fresh L0 SST then drain ALL L0 -> a new L1 seg.
        fn seg(node_dir: &Path, m: &mut Manifest, id: u64, node: KnowledgeNode, seq: u64) {
            let flush = NodeFlushWorker::new(node_dir.to_path_buf());
            let mut mt = NodeMemTable::new();
            mt.insert(NodeId(id), node, seq);
            let l0 = flush.flush(&mt, m).unwrap().expect("non-empty memtable");
            m.add_node_sst(0, l0).unwrap();
            let l0_uids: Vec<u64> = m.node_levels.iter().find(|l| l.level == 0)
                .map(|l| l.ssts.iter().map(|s| s.file_uid).collect()).unwrap_or_default();
            NodeCompactionWorker::new(node_dir.to_path_buf())
                .compact(CompactionPlan { source_uids: l0_uids, source_level: 0, target_level: 1 }, m, None)
                .unwrap();
        }
        // Highest-seq winner across ALL segments = exactly what the compaction
        // dedup keeps ("newest by seq").
        fn winner(node_dir: &Path, m: &Manifest, id: u64) -> Option<(KnowledgeContent, u64)> {
            let mut best: Option<(KnowledgeContent, u64)> = None;
            for lvl in &m.node_levels {
                for s in &lvl.ssts {
                    if s.first_node_id > id || s.last_node_id < id { continue; }
                    let r = NodeSstReader::open_with_key(&node_dir.join(format!("{}.nsst", s.file_uid)), None).unwrap();
                    if let Some(e) = r.get(NodeId(id)).unwrap() {
                        let cand = (e.payload.knowledge.content.clone(), e.seq);
                        best = match best { None => Some(cand), Some(p) if cand.1 > p.1 => Some(cand), Some(p) => Some(p) };
                    }
                }
            }
            best
        }
        fn def_of(c: &Option<(KnowledgeContent, u64)>) -> Option<String> {
            match c {
                Some((KnowledgeContent::Definition(d), _)) => Some(d.clone()),
                _ => None,
            }
        }

        let concepts: [(u64, &str, &str); 4] = [
            (1, "radioactivity",
             "the spontaneous emission of radiation from an unstable atomic nucleus"),
            (2, "radioactive decay",
             "the process by which an unstable atomic nucleus loses energy by emitting radiation"),
            (3, "nuclear fission",
             "a nuclear reaction in which a heavy nucleus splits into lighter nuclei, releasing energy"),
            (4, "wave-particle duality",
             "the principle that every quantum entity exhibits both wave-like and particle-like behaviour"),
        ];

        let dir = tempdir().unwrap();
        let node_dir = dir.path().join("node_store");
        std::fs::create_dir_all(&node_dir).unwrap();
        let mut m = Manifest::new(KeyFingerprint::none());

        // Build, per concept: a Definition seg (LOW seq) then a later bare stub
        // seg (HIGHER seq) for the SAME NodeId.
        for (i, (id, label, gloss)) in concepts.iter().enumerate() {
            seg(&node_dir, &mut m, *id, mk_def(label, gloss), 100 + i as u64);
            seg(&node_dir, &mut m, *id, mk_stub(label), 200 + i as u64);
        }

        // BEFORE the collapse: newest-by-seq selects the stub for every concept
        // -> the stored gloss is the loser and would be DROPPED.
        for (id, label, _gloss) in concepts.iter() {
            let w = winner(&node_dir, &m, *id);
            eprintln!("[OP-1505] BEFORE collapse  {:<22} newest-by-seq winner = {:?}", label, w);
            assert!(def_of(&w).is_none(),
                "precondition: the higher-seq stub shadows the Definition for {}", label);
        }

        // COLLAPSE all L1 segments into a single L1 (the real seq-dedup merge).
        let l1_uids: Vec<u64> = m.node_levels.iter().find(|l| l.level == 1)
            .map(|l| l.ssts.iter().map(|s| s.file_uid).collect()).unwrap();
        let cache = NodeSstReaderCache::new(node_dir.clone());
        NodeCompactionWorker::new(node_dir.clone())
            .compact(CompactionPlan { source_uids: l1_uids, source_level: 1, target_level: 1 }, &mut m, Some(&cache))
            .expect("same-level L1 collapse must succeed");
        assert_eq!(level_sst_count(&m, 1), 1, "L1 collapsed to a single segment");

        // AFTER the collapse: the surviving node must STILL carry its gloss.
        for (id, label, gloss) in concepts.iter() {
            let w = winner(&node_dir, &m, *id);
            eprintln!("[OP-1505] AFTER  collapse  {:<22} winner = {:?}", label, w);
            match &w {
                Some((KnowledgeContent::Definition(d), seq)) => {
                    assert_eq!(*seq, 200 + *id - 1, "winner keeps newest seq metadata for {}", label);
                    assert_eq!(d, gloss, "gloss must survive the seq-dedup collapse for {}", label);
                }
                other => panic!(
                    "OP-1505 REGRESSION: gloss for {} dropped on collapse -> winner {:?}", label, other),
            }
        }
    }
}
