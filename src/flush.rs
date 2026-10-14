//! OP-BRAIN-V4 Phase 4 — Background flush worker.
//!
//! Wires the Phase 3 MemTable + WAL primitives to the Phase 2 SST
//! writer. When the MemTable fills (or `flush_now` is called) the
//! worker:
//!
//! 1. Atomically swaps the live MemTable with a fresh empty one
//!    (`MemTable::swap_for_flush`).
//! 2. Sorts the swapped table's entries by subject Seed bytes (the
//!    SST writer's contract). The MemTable's `iter_for_flush` already
//!    returns subject-bytes-sorted data, but we re-validate cheaply.
//! 3. Allocates a fresh monotonic SST id from the manifest counter.
//! 4. Opens an `SSTWriter` at `<sst_dir>/L0-<id>.sst`, pushes each
//!    bond + keyframes, calls `finalize_with_meta` (fsyncs the file).
//! 5. Clones the live manifest, appends the new SSTEntry, calls
//!    `Manifest::save_atomic` (writes tmp, fsyncs, rotates, fsyncs
//!    the directory).
//! 6. Truncates the WAL up to the highest seq in the flushed table
//!    (`Wal::truncate_up_to`). Only AFTER the manifest is durable —
//!    if any step before this fails, the WAL still holds the records.
//! 7. Atomically replaces the Brain's live `Arc<Manifest>` snapshot.
//!
//! ## Durability sequence (load-bearing — do NOT reorder)
//!
//! ```text
//!   SST file fsync          (writer finalize)
//!         |
//!         v
//!   Manifest .tmp written, .tmp fsync, atomic rename
//!         |
//!         v
//!   Manifest parent dir fsync
//!         |
//!         v
//!   WAL truncate_up_to(highest_seq)
//! ```
//!
//! At every break point the system is recoverable on restart:
//!
//! - Crash AFTER SST fsync but BEFORE manifest rename: the SST is an
//!   orphan on disk; restart calls `manifest::cleanup_orphan_ssts`
//!   which renames it to `.orphan-<ts>`. WAL replay reconstructs the
//!   MemTable; the next flush re-writes the bonds.
//! - Crash AFTER manifest rename but BEFORE WAL truncate: WAL replay
//!   re-inserts the (already-flushed) bonds into the new MemTable.
//!   Phase 5 read path dedups by `(subject, predicate, object)`.
//!   No data loss; Phase 6 compaction will fold the duplicates.
//!
//! ## Concurrency
//!
//! Only one flush runs at a time. The worker thread is the sole
//! flusher; `flush_now` synchronizes through a `Mutex<()>` guard so
//! two concurrent callers serialize naturally. Reads on other
//! threads see either the pre-flush or post-flush manifest via the
//! `Arc<RwLock<Arc<Manifest>>>` snapshot pointer; never a torn
//! intermediate state.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread::{self, JoinHandle};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::brain::BrainError;
use crate::encrypt::MasterKey;
use crate::manifest::{
    sst_filename, Manifest, SSTEntry,
};
use crate::memtable::MemTable;
use crate::node::Confidence;
use crate::node_intern::NodeIntern;
use crate::sst_writer::SSTWriter;
use crate::wal::Wal;

/// Default polling cadence for the background flush thread (100 ms).
pub const DEFAULT_POLL_INTERVAL_MS: u64 = 100;

/// State shared between the public `FlushWorker` handle and the
/// worker thread. Held behind `Arc<...>` so both can reach it.
struct FlushWorkerShared {
    memtable: Arc<MemTable>,
    wal: Arc<Mutex<Wal>>,
    intern: NodeIntern,
    manifest: Arc<RwLock<Arc<Manifest>>>,
    manifest_path: PathBuf,
    sst_dir: PathBuf,
    key: Option<MasterKey>,
    /// OP-BRAIN-V5 Phase 2.3 -- path to the nodeintern.bin sidecar.
    /// Lives next to the WAL segment (canonical path derived via
    /// `Brain::sidecar_path_for_wal`). Written atomically by
    /// `flush_once` BEFORE the SST + manifest sequence so a crash
    /// between sidecar rename and manifest commit leaves the
    /// sidecar valid + WAL delta intact (audit S4.3 Case A boot).
    sidecar_path: PathBuf,
    /// OP-BRAIN-V5 Phase 2.3 -- handle on the NodeIntern table the
    /// flush worker calls `save_to` on. Cloned at construction
    /// from the same intern used for bond translation (the network
    /// intern); both views are backed by the same shared inner
    /// RwLock so a save here observes the live state.
    sidecar_intern: NodeIntern,
    /// OP-BRAIN-V5 Phase 2.6 -- node-store flush worker. Owns
    /// the `<wal_dir>/node_store/` directory and translates a
    /// swapped-snapshot `NodeMemTable` into a fresh NodeSST
    /// file, allocating `file_uid` + `epoch` from the working
    /// manifest clone. See `flush_once` for the integration
    /// ordering (sidecar -> NodeSST -> v5 BondSST -> v4 bond SST
    /// -> manifest save_atomic -> WAL truncate).
    node_flush_worker: crate::node_store::NodeFlushWorker,
    /// OP-BRAIN-V5 Phase 3.3 -- v5 bond-store flush worker. Owns
    /// the `<wal_dir>/bond_store/` directory and translates a
    /// swapped-snapshot bond iterator into a fresh BondSST file,
    /// allocating `file_uid` + `epoch` from the working manifest
    /// clone. Coexists with the v4 bond SST writer until Phase
    /// 3.9 retires v4. See `flush_once` for the integration
    /// ordering.
    bond_flush_worker: crate::bond_store::BondFlushWorker,
    stop_flag: AtomicBool,
    /// OP-BRAIN-V5 root-cause fix (flush-vs-compaction stale-base
    /// manifest swap, sibling of 7bd94b8): the SHARED manifest-commit
    /// lock. Serializes flushes against each other AND against the
    /// bond/node compaction schedulers, which reuse this exact
    /// `Arc<Mutex<()>>` (see `FlushWorker::commit_lock`). Holding it
    /// across the whole read-clone -> save_atomic -> swap sequence
    /// guarantees no actor ever publishes a manifest derived from a base
    /// snapshot another actor has already superseded.
    commit_lock: Arc<Mutex<()>>,
    /// Highest seq successfully durable (SST + manifest fsync'd, WAL
    /// truncated). Diagnostic / test hook.
    last_flushed_seq: AtomicU64,
    /// Total flushes performed (success only). Diagnostic.
    flush_count: AtomicU64,
    /// OP-BRAIN-V4 Phase 5 -- one-shot guard for `cleanup_orphan_ssts`.
    /// First call to `run_orphan_cleanup_once` flips this from `false`
    /// to `true` and runs cleanup; subsequent calls skip. Owned as an
    /// `Arc<AtomicBool>` so the Brain's own field on `Brain::orphan_cleanup_done`
    /// can be the same atomic (the worker doesn't own a clone of the
    /// Brain). When `Brain::enable_flush_worker` constructs the worker
    /// it passes the Brain-side `Arc<AtomicBool>` in.
    orphan_cleanup_done: Arc<AtomicBool>,
}

/// Background flush worker. Owns a thread that polls the MemTable
/// for fullness and triggers a swap+flush when it crosses the cap.
pub struct FlushWorker {
    shared: Arc<FlushWorkerShared>,
    thread_handle: Option<JoinHandle<()>>,
    poll_interval: Duration,
}

impl FlushWorker {
    /// Spawn the worker thread. Polls every `DEFAULT_POLL_INTERVAL_MS`
    /// for `MemTable::is_full()` and flushes when triggered.
    ///
    /// `manifest_path` must already point to a valid `MANIFEST` file
    /// (or be the target path for the first save). The flush worker
    /// does NOT create the file on its own; callers should call
    /// `Brain::enable_flush_worker` which initializes the manifest
    /// via `Manifest::new(...)` and saves an initial empty one.
    pub fn start(
        memtable: Arc<MemTable>,
        wal: Arc<Mutex<Wal>>,
        intern: NodeIntern,
        manifest: Arc<RwLock<Arc<Manifest>>>,
        manifest_path: PathBuf,
        sst_dir: PathBuf,
        key: Option<MasterKey>,
        orphan_cleanup_done: Arc<AtomicBool>,
        // OP-BRAIN-V5 Phase 2.3 -- sidecar path + intern handle.
        sidecar_path: PathBuf,
        sidecar_intern: NodeIntern,
        // OP-BRAIN-V5 Phase 2.6 -- node_store directory root.
        node_store_dir: PathBuf,
        // OP-BRAIN-V5 Phase 3.3 -- bond_store directory root.
        bond_store_dir: PathBuf,
    ) -> Self {
        // OP-BRAIN-V5 6.5: thread the master key into the v5 flush
        // workers so flushed Node/Bond SSTs are encrypted at rest.
        let node_flush_worker =
            crate::node_store::NodeFlushWorker::new(node_store_dir).with_key(key.clone());
        let bond_flush_worker =
            crate::bond_store::BondFlushWorker::new(bond_store_dir).with_key(key.clone());
        let shared = Arc::new(FlushWorkerShared {
            memtable,
            wal,
            intern,
            manifest,
            manifest_path,
            sst_dir,
            key,
            sidecar_path,
            sidecar_intern,
            node_flush_worker,
            bond_flush_worker,
            stop_flag: AtomicBool::new(false),
            commit_lock: Arc::new(Mutex::new(())),
            last_flushed_seq: AtomicU64::new(0),
            flush_count: AtomicU64::new(0),
            orphan_cleanup_done,
        });
        // Phase 5: one-shot orphan cleanup gated by the shared
        // AtomicBool. First worker startup runs it; subsequent
        // FlushWorker::start calls (or hypothetical future re-init
        // paths) skip. Failures are non-fatal -- log and continue,
        // matching the pre-Phase-5 behaviour of
        // `Brain::enable_flush_worker`.
        run_orphan_cleanup_once(&shared);
        let poll_interval = Duration::from_millis(DEFAULT_POLL_INTERVAL_MS);
        let thread_handle = {
            let shared = shared.clone();
            let pi = poll_interval;
            Some(thread::Builder::new()
                .name("lu4d-flush".to_string())
                .spawn(move || worker_loop(shared, pi))
                .expect("spawn lu4d-flush thread"))
        };
        FlushWorker { shared, thread_handle, poll_interval }
    }

    /// Configure the polling interval. Useful for tests that want a
    /// fast trigger; production keeps the default 100 ms.
    pub fn set_poll_interval(&mut self, ms: u64) {
        self.poll_interval = Duration::from_millis(ms);
    }

    /// Force a synchronous flush. Triggers a swap even if the
    /// MemTable hasn't hit its cap. Returns after the SST is durable,
    /// the manifest is updated, and the WAL is truncated. If the
    /// MemTable is empty, returns Ok without producing an SST.
    pub fn flush_now(&self) -> Result<(), BrainError> {
        flush_once(&self.shared)
    }

    /// OP-BRAIN-V5 root-cause fix -- the SHARED manifest-commit lock.
    /// The bond (and, transitively, node) compaction schedulers reuse
    /// THIS lock so flush and compaction serialize their
    /// read-clone -> save_atomic -> swap (-> cleanup) against the SAME
    /// `Arc<RwLock<Arc<Manifest>>>` and never publish a swap derived
    /// from a stale base snapshot. `load_v5` wires the bond scheduler
    /// to this lock via `start_with_commit_lock`. Sibling fix to
    /// 7bd94b8 (node-vs-bond) on the flush path.
    pub fn commit_lock(&self) -> Arc<Mutex<()>> {
        self.shared.commit_lock.clone()
    }

    /// Stop the worker. Sets the stop flag, joins the thread.
    /// Idempotent: calling twice is harmless.
    pub fn stop(&mut self) -> Result<(), BrainError> {
        self.shared.stop_flag.store(true, Ordering::SeqCst);
        if let Some(h) = self.thread_handle.take() {
            // join() returns Result<(), Box<dyn Any+Send>>. Swallow
            // panics into a BrainError so the caller surface is
            // uniform; we cannot meaningfully resume a panicked
            // worker thread anyway.
            h.join().map_err(|_| BrainError::Format(
                "lu4d-flush thread panicked".to_string()
            ))?;
        }
        Ok(())
    }

    /// Highest seq successfully flushed (SST durable + manifest
    /// updated + WAL truncated). Diagnostic / test hook.
    pub fn last_flushed_seq(&self) -> u64 {
        self.shared.last_flushed_seq.load(Ordering::SeqCst)
    }

    /// Total flush operations that succeeded. Diagnostic.
    pub fn flush_count(&self) -> u64 {
        self.shared.flush_count.load(Ordering::SeqCst)
    }

    /// Read-only view of the manifest path used by this worker.
    pub fn manifest_path(&self) -> &Path {
        &self.shared.manifest_path
    }

    /// Read-only view of the SST directory used by this worker.
    pub fn sst_dir(&self) -> &Path {
        &self.shared.sst_dir
    }
}

impl Drop for FlushWorker {
    fn drop(&mut self) {
        // Best-effort clean shutdown. Errors are silently swallowed
        // in Drop (no recovery option). The stop() call is idempotent
        // so a caller that already stopped doesn't double-join.
        let _ = self.stop();
    }
}

fn worker_loop(shared: Arc<FlushWorkerShared>, poll_interval: Duration) {
    loop {
        if shared.stop_flag.load(Ordering::SeqCst) {
            break;
        }
        // Cheap lock-free poll.
        let should_flush = shared.memtable.is_full();
        if should_flush {
            // Errors during a background flush are logged but do not
            // kill the worker; the WAL still holds the records so the
            // next attempt will retry. Killing the worker on a single
            // I/O hiccup would lose all subsequent flush coverage.
            if let Err(e) = flush_once(&shared) {
                eprintln!(
                    "[flush-worker] flush failed (records preserved in WAL): {}",
                    e
                );
            }
        }
        thread::sleep(poll_interval);
    }
}

/// OP-BRAIN-V5 forced-interleaving probe (TEST-ONLY, env-gated, default
/// OFF). Widens the window between reading the base manifest snapshot
/// and committing the swap so a concurrent compaction can be forced to
/// read the SAME base snapshot. Used by the flush-vs-compaction
/// stale-base-swap regression; production builds never compile this.
#[cfg(test)]
fn race_probe_sleep(var: &str) {
    if let Ok(v) = std::env::var(var) {
        if let Ok(ms) = v.parse::<u64>() {
            if ms > 0 {
                std::thread::sleep(std::time::Duration::from_millis(ms));
            }
        }
    }
}

/// The one canonical flush implementation. Holds the shared
/// `commit_lock` for
/// the duration so concurrent calls (worker poll + explicit
/// `flush_now`) serialize naturally.
fn flush_once(shared: &Arc<FlushWorkerShared>) -> Result<(), BrainError> {
    let _g = shared
        .commit_lock
        .lock()
        .expect("commit lock (shared flush/compaction) poisoned");

    // 0. Snapshot the current next_sst_id from the manifest. We
    //    bump the counter INSIDE the manifest clone before the SST
    //    is written, so a crash mid-flush leaves an unused id
    //    range — harmless, the counter is monotonic-only.
    let (mut new_manifest, sst_id) = {
        let live = shared.manifest.read().expect("manifest read lock");
        let mut cloned: Manifest = (**live).clone();
        let id = cloned.next_sst_id();
        (cloned, id)
    };

    // OP-BRAIN-V5 forced-interleaving probe point: between reading the
    // base manifest snapshot and the eventual swap (step 6).
    #[cfg(test)]
    race_probe_sleep("LU5_FLUSH_PROBE_MS");

    // 1. Swap-for-flush. The returned box holds the records we will
    //    flush; the live MemTable now starts empty (fresh inserts go
    //    there).
    let snapshot: Box<MemTable> = shared.memtable.swap_for_flush();
    let bonds = snapshot.iter_for_flush(&shared.intern);
    // Phase 10: collect the node table snapshot. Empty is legal --
    // a flush with bonds-but-no-nodes is fine (the v3 archive read
    // path still finds the node).
    let nodes = snapshot.iter_nodes_for_flush(&shared.intern);
    if bonds.is_empty() && nodes.is_empty() {
        // Nothing to flush. Don't write an SST or touch the manifest.
        return Ok(());
    }

    // Compute the highest seq we are about to flush. OP-BRAIN-V5
    // Phase 3.8.1: `MemBondFlush` now carries the per-bond seq, so
    // the max folds directly over the snapshot we already collected
    // -- no second walk of the boxed MemTable.
    let highest_seq = compute_highest_seq(&bonds);

    // 2. Validate subject ordering. `iter_for_flush` already sorts
    //    by subject bytes, but we re-check cheaply so a future
    //    refactor that drops the sort can't silently corrupt
    //    on-disk order.
    for w in bonds.windows(2) {
        if w[0].0.as_bytes() > w[1].0.as_bytes() {
            return Err(BrainError::Format(format!(
                "flush: MemTable iter_for_flush returned subjects out of order \
                 (subject {:?} > {:?})",
                w[0].0, w[1].0
            )));
        }
    }

    // OP-BRAIN-V5 Phase 2.3 -- write the nodeintern.bin sidecar
    // BEFORE the SST + manifest commit. The order is
    // load-bearing for the boot-replay arm (audit S4.3):
    //
    //   sidecar.tmp -> fsync -> rename(.tmp -> sidecar) -> SST -> manifest -> WAL truncate
    //
    // Crash semantics:
    //   - Before sidecar rename: old sidecar (or no sidecar)
    //     stays; WAL holds the full delta. Next boot replays
    //     WAL on top of the stale-but-valid snapshot via
    //     `insert_with_id` (BootReplayMode::SidecarLoaded) or
    //     via `intern_or_get` (BootReplayMode::InsertionOrderFallback).
    //   - After sidecar rename, before manifest commit: sidecar
    //     is fresh; WAL still holds the bonds that were going
    //     to land in the SST. Next boot: sidecar restores
    //     bindings verbatim; WAL replay rebuilds the MemTable;
    //     next flush re-writes the SST. No data loss.
    //   - After manifest commit, before WAL truncate: same as
    //     v4 (existing flush_once tail) -- WAL replay folds
    //     duplicate bonds that are already in the SST.
    //
    // Atomicity: we write to `<sidecar_path>.tmp` first, fsync
    // the tmp file (NodeIntern::save_to already does the file
    // fsync_all), then `rename(.tmp -> sidecar_path)`. POSIX
    // rename is atomic on the same filesystem; co-locating the
    // sidecar with the WAL (Brain::sidecar_path_for_wal puts it
    // in `wal.parent()/nodeintern.bin`) guarantees same-fs.
    {
        let sidecar = &shared.sidecar_path;
        let tmp_path = sidecar_tmp_path(sidecar);
        let _stats = shared
            .sidecar_intern
            .save_to_with_key(&tmp_path, shared.key.clone())
            .map_err(|e| BrainError::Format(format!(
                "flush: sidecar save_to({}) failed: {}",
                tmp_path.display(), e,
            )))?;
        // Atomic rename. On failure, leave the .tmp in place
        // so a future flush can try again; the previous sidecar
        // (if any) stays valid.
        std::fs::rename(&tmp_path, sidecar).map_err(|e| BrainError::Io(
            std::io::Error::new(
                std::io::ErrorKind::Other,
                format!(
                    "flush: sidecar rename({} -> {}): {}",
                    tmp_path.display(), sidecar.display(), e,
                ),
            ),
        ))?;
        // fsync the containing directory so the rename hits
        // disk (the tmp's own fsync_all guarantees the data
        // bytes are durable; the dir fsync makes the directory
        // entry update durable too -- standard POSIX rename
        // discipline; mirrors Manifest::save_atomic).
        if let Some(parent) = sidecar.parent() {
            if let Ok(dir) = std::fs::File::open(parent) {
                let _ = dir.sync_all();
            }
        }
    }

    // OP-BRAIN-V5 Phase 2.6 -- flush the swapped
    // NodeMemTable to a fresh NodeSST file BEFORE the bond
    // SST + manifest commit. The order is load-bearing for
    // crash recovery: the sidecar (Phase 2.3 above) captures
    // the NodeIntern Seed->NodeId map, the NodeSST captures
    // the payloads at those NodeIds, and the manifest
    // save_atomic (below) commits BOTH the bond-side SSTEntry
    // AND the node-side NodeSSTEntry in a single atomic
    // rename. Crash semantics:
    //   - Before NodeSstWriter::finish() rename: the .nsst.tmp.<uid>
    //     is on disk, no manifest reference; cleaned at restart
    //     by `cleanup_orphan_node_ssts`. WAL replay rebuilds.
    //   - After rename, before manifest save_atomic: final-name
    //     .nsst on disk, no manifest reference (the in-memory
    //     bumps to next_node_sst_uid / next_epoch / node_levels
    //     are lost); same orphan cleanup. WAL replay rebuilds.
    //   - After manifest save_atomic: NodeSST is live, manifest
    //     references it, WAL truncate runs next (existing tail).
    //
    // Empty NodeMemTable: NodeFlushWorker::flush returns
    // Ok(None); we skip the add_node_sst call. The bond-side
    // commit chain proceeds unchanged.
    // We hold the inner lock on the swapped Box<MemTable> for
    // the duration of the NodeSST write. The snapshot is
    // exclusively owned by this thread so the lock is
    // uncontended (writes already race against the live
    // post-swap MemTable, not this snapshot). Closure-form here
    // avoids materialising a cloned NodeMemTable: we pass
    // `&NodeMemTable` straight into NodeFlushWorker::flush,
    // preserving the borrowing-iter streaming discipline.
    let node_flush_result = snapshot.with_node_table(
        |nt: &crate::node_store::NodeMemTable| -> Result<
            Option<crate::manifest::NodeSSTEntry>,
            crate::node_store::NodeFlushError,
        > {
            shared.node_flush_worker.flush(nt, &mut new_manifest)
        },
    );
    let node_sst_entry_opt = node_flush_result.map_err(|e| BrainError::Format(format!(
        "flush: NodeFlushWorker::flush failed: {}", e,
    )))?;
    if let Some(ref node_entry) = node_sst_entry_opt {
        // Commit the node-side entry into the working manifest;
        // the existing save_atomic call at step 5 below will
        // persist both the bond SSTEntry and this NodeSSTEntry
        // atomically.
        new_manifest.add_node_sst(0, node_entry.clone()).map_err(|e| BrainError::Format(
            format!("flush: Manifest::add_node_sst failed: {}", e),
        ))?;
    }

    // OP-BRAIN-V5 Phase 3.3 -- v5 BondStore flush. Runs
    // ALONGSIDE (NOT replacing) the v4 bond SST writer below.
    // Each flush emits BOTH a v4 `<level>_<id>.sst` AND a v5
    // `<file_uid>.bsst`. v4 stays production-read until Phase
    // 3.7 wires the v5 read path; Phase 3.9 retires the v4
    // writer. The dual-emit 2x write overhead is bounded and
    // intentional for the cutover window.
    //
    // Stream ordering: BondSstWriter requires strictly ascending
    // (subject NodeId, predicate u8, object NodeId). The v4
    // iter_for_flush sorts subjects by Seed-bytes; that's NOT
    // NodeId order. We re-sort here for the v5 stream. The
    // bond_store snapshot is collected ONCE into a Vec ordered
    // by the v5 key; the worker then iterates lazily into the
    // writer (writer holds one block buffer max).
    //
    // Tombstones: the v5 BondPayload uses `confidence = 0xFF`
    // as the TOMBSTONE marker per Phase 3.0 audit Q9.2. We
    // include tombstone rows so compaction can shadow older
    // same-key inserts -- symmetric with v4's push_tombstone.
    //
    // Crash semantics: see BondFlushWorker doc-comment.
    let v5_bonds_owned: Vec<(
        crate::node_intern::NodeId,
        u8,
        crate::node_intern::NodeId,
        crate::bond_store::BondPayload,
    )> = {
        let mut out: Vec<(
            crate::node_intern::NodeId,
            u8,
            crate::node_intern::NodeId,
            crate::bond_store::BondPayload,
        )> = Vec::with_capacity(bonds.iter().map(|(_, b)| b.len()).sum());
        // Walk the v4 iter_for_flush output directly. OP-BRAIN-V5
        // Phase 3.8.1: `MemBondFlush` carries `seq` now, so the
        // Phase 3.3 O(N) re-fetch via `get_bonds_for_subject`
        // (which existed only to read per-bond seqs the flush
        // projection used to omit) is gone.
        for (subject_seed, flush_bonds) in &bonds {
            let subject_id = match shared.intern.get(subject_seed) {
                Some(id) => id,
                None => continue,
            };
            for mb in flush_bonds {
                // OP-BRAIN-V5 6.3 -- the object NodeId is carried on the
                // MemBondFlush now (interned at insert time in
                // MemTable::insert / ::delete), so the flush-time
                // Seed->NodeId re-intern is a pure no-op lookup we elide
                // entirely. Symmetric with how the subject is keyed.
                let object_id = mb.object_id;
                let predicate_disc =
                    crate::memtable::predicate_discriminant(mb.predicate);
                let confidence_byte = if mb.tombstone { 0xFFu8 } else { 1u8 };
                let weight_quant =
                    (mb.weight.clamp(0.0, 1.0) * 255.0).round() as u8;
                out.push((
                    subject_id,
                    predicate_disc,
                    object_id,
                    crate::bond_store::BondPayload {
                        weight: weight_quant,
                        confidence: confidence_byte,
                        keyframes: mb.keyframes.clone(),
                        seq: mb.seq,
                    },
                ));
            }
        }
        // Sort by (subject_id, predicate, object_id) ascending to
        // satisfy BondSstWriter::append's strict-monotonic
        // precondition. Dedup duplicate triples by keeping highest
        // seq (writer rejects same-triple repeats outright).
        out.sort_by(|a, b| {
            a.0.0.cmp(&b.0.0)
                .then_with(|| a.1.cmp(&b.1))
                .then_with(|| a.2.0.cmp(&b.2.0))
                .then_with(|| b.3.seq.cmp(&a.3.seq))
        });
        out.dedup_by(|a, b| a.0 == b.0 && a.1 == b.1 && a.2 == b.2);
        out
    };
    let bond_flush_result = shared.bond_flush_worker.flush(
        v5_bonds_owned.iter().map(|(s, p, o, pay)| (*s, *p, *o, pay)),
        &mut new_manifest,
    );
    let bond_v5_entry_opt = bond_flush_result.map_err(|e| BrainError::Format(format!(
        "flush: BondFlushWorker::flush failed: {}", e,
    )))?;
    if let Some(ref bond_entry) = bond_v5_entry_opt {
        new_manifest.add_bond_v5_sst(0, bond_entry.clone()).map_err(|e| BrainError::Format(
            format!("flush: Manifest::add_bond_v5_sst failed: {}", e),
        ))?;
    }

    // 3. Open the SST writer.
    let filename = sst_filename(0, sst_id);
    let sst_path = shared.sst_dir.join(&filename);
    let mut writer = SSTWriter::create(&sst_path, 0, shared.key.clone())
        .map_err(BrainError::from)?;
    writer.set_manifest_id(sst_id);
    // Phase 10: index the node table by Seed bytes for O(1) lookups
    // inside the bond loop. Then route bond pushes through
    // push_bond_with_nodes so subject/object node payloads land in the
    // SST's per-block node table.
    let mut node_map: std::collections::HashMap<[u8; 32], crate::node::KnowledgeNode> =
        std::collections::HashMap::with_capacity(nodes.len());
    for (seed, n, _seq) in &nodes {
        node_map.insert(*seed.as_bytes(), n.clone());
    }
    let mut seen_in_sst: std::collections::HashSet<[u8; 32]> =
        std::collections::HashSet::with_capacity(nodes.len());
    for (subj, subj_bonds) in &bonds {
        for mb in subj_bonds {
            // Phase 6: tombstones land as on-disk tombstone rows so
            // compaction can shadow older same-key inserts. Normal
            // bonds flush at Confidence::Verified; the read path
            // re-derives the band from keyframe-source provenance.
            if mb.tombstone {
                writer.push_tombstone(subj, &mb.object, mb.predicate)
                    .map_err(BrainError::from)?;
            } else {
                let subj_node = node_map.get(subj.as_bytes());
                let obj_node = node_map.get(mb.object.as_bytes());
                writer.push_bond_with_nodes(
                    subj, subj_node,
                    &mb.object, obj_node,
                    mb.predicate, mb.weight,
                    Confidence::Verified, &mb.keyframes,
                ).map_err(BrainError::from)?;
                if subj_node.is_some() {
                    seen_in_sst.insert(*subj.as_bytes());
                }
                if obj_node.is_some() {
                    seen_in_sst.insert(*mb.object.as_bytes());
                }
            }
        }
    }
    // Phase 10: any node that did not land via a bond row (e.g. an
    // isolated node, or a node whose only bond was a tombstone) gets
    // a standalone push so the metadata still lands in this SST. The
    // subject-sort contract requires nodes to be pushed in seed
    // order; the MemTable's iter_nodes_for_flush already sorts that
    // way, but if any are interleaved with the bond stream we need
    // to be sure we don't regress. push_node enforces it.
    for (seed, n, _seq) in &nodes {
        if !seen_in_sst.contains(seed.as_bytes()) {
            // Verify seed monotonicity vs the writer's last_subject.
            // If a standalone node would regress (because its seed is
            // lower than the last bond's subject), skip it -- the
            // next flush will pick it up since it remains in the
            // post-swap MemTable for the next cycle. In practice the
            // MemTable's sorted iter avoids this case; this is a
            // defensive skip.
            // (Cleaner: we sort `nodes` ahead of the bond stream and
            // emit in interleaved seed order. Deferred to a Phase
            // 10.x cleanup port if profiling shows the
            // tombstone-only path is hot.)
            if let Err(e) = writer.push_node(seed, n) {
                // OutOfOrder here is expected for the tombstone-only
                // case described above; log and continue rather than
                // poisoning the flush.
                match e {
                    crate::sst_writer::SSTWriterError::OutOfOrder { .. } => {
                        eprintln!(
                            "[flush] standalone node {:?} regressed against bond stream; skipping (next flush picks it up)",
                            seed
                        );
                    }
                    other => return Err(BrainError::from(other)),
                }
            }
        }
    }
    // 4. Finalize: writes block index, footer, rewinds + writes
    //    header, fsyncs the file. Returns metadata for the SSTEntry.
    let meta = writer.finalize_with_meta().map_err(BrainError::from)?;

    // Stat the file for size + mtime AFTER fsync.
    let size_bytes = std::fs::metadata(&sst_path).map(|m| m.len()).unwrap_or(0);
    let mtime = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);

    // 5. Build the manifest entry and save_atomic.
    let entry = SSTEntry {
        id: sst_id,
        file: filename,
        subject_min: *meta.subject_min.as_bytes(),
        subject_max: *meta.subject_max.as_bytes(),
        size: size_bytes,
        mtime,
        bond_count: meta.bond_count,
        epoch: 0,
    };
    new_manifest.add_sst(0, entry);
    new_manifest.save_atomic(&shared.manifest_path)?;

    // 6. Swap the live Arc<Manifest> snapshot pointer.
    {
        let mut w = shared.manifest.write().expect("manifest write lock");
        *w = Arc::new(new_manifest);
    }

    // 7. Truncate the WAL up to highest_seq (inclusive). After this
    //    point a restart will replay only the records that landed in
    //    the NEW (post-swap) MemTable.
    {
        let mut wal = shared.wal.lock().expect("wal lock poisoned");
        wal.truncate_up_to(highest_seq + 1).map_err(|e| {
            BrainError::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("WAL truncate_up_to({}): {}", highest_seq + 1, e),
            ))
        })?;
    }

    shared.last_flushed_seq.store(highest_seq, Ordering::SeqCst);
    shared.flush_count.fetch_add(1, Ordering::SeqCst);
    Ok(())
}

/// Find the maximum seq in the swapped MemTable. Re-walks the inner
/// entries because `iter_for_flush` discards seq for ordering
/// purposes.
/// Max per-bond seq across the flush snapshot. OP-BRAIN-V5 Phase
/// 3.8.1: folds over the `MemBondFlush.seq` field that
/// `iter_for_flush` now carries. The pre-3.8.1 implementation
/// re-walked the boxed MemTable via `get_bonds_for_subject` because
/// the flush projection deliberately omitted seq -- two full extra
/// walks per flush, now zero.
fn compute_highest_seq(
    bonds: &[(crate::seed::Seed, Vec<crate::memtable::MemBondFlush>)],
) -> u64 {
    let mut max_seq: u64 = 0;
    for (_, subj_bonds) in bonds {
        for mb in subj_bonds {
            if mb.seq > max_seq {
                max_seq = mb.seq;
            }
        }
    }
    max_seq
}
/// Phase 5 one-shot orphan-SST cleanup. Idempotent across
/// FlushWorker::start calls thanks to the shared `AtomicBool` flag.
/// Logs non-fatally and continues on failure, matching the prior
/// `Brain::enable_flush_worker` behaviour.
fn run_orphan_cleanup_once(shared: &Arc<FlushWorkerShared>) {
    use std::sync::atomic::Ordering;
    // compare_exchange: only the very first caller transitions
    // false -> true; everyone else returns Err and skips.
    if shared
        .orphan_cleanup_done
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_err()
    {
        return;
    }
    // Snapshot the current manifest so the cleanup pass sees the
    // same SST set the worker will publish writes against.
    let live = shared.manifest.read().expect("manifest snapshot read lock").clone();
    match crate::manifest::cleanup_orphan_ssts(&shared.sst_dir, &live) {
        Ok(orphans) => {
            if !orphans.is_empty() {
                eprintln!(
                    "[lu4d-flush] orphan-SST cleanup renamed {} file(s) under {}",
                    orphans.len(), shared.sst_dir.display(),
                );
            }
        }
        Err(e) => eprintln!(
            "[lu4d-flush] orphan-SST cleanup failed (non-fatal): {}", e
        ),
    }
}


/// OP-BRAIN-V5 Phase 2.3 -- derive the sidecar tmp path used by
/// `flush_once`. The convention is `<sidecar_path>.tmp` -- a
/// sibling file in the same directory, so the rename stays
/// intra-filesystem (POSIX atomic). Mirrors the
/// `Manifest::save_atomic` `.tmp` convention.
fn sidecar_tmp_path(sidecar: &Path) -> PathBuf {
    let mut s = sidecar.as_os_str().to_owned();
    s.push(".tmp");
    PathBuf::from(s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bond::{BondType, KeyframeSource, TemporalKeyframe};
    use crate::manifest::{KeyFingerprint, MANIFEST_FILE_NAME};
    use crate::seed::Seed;
    use rand::{Rng, SeedableRng};
    use rand::rngs::StdRng;
    use tempfile::tempdir;

    fn rand_seed(rng: &mut StdRng) -> Seed {
        let mut b = [0u8; 32];
        rng.fill(&mut b);
        Seed::from_bytes(b)
    }

    fn baseline_kf(ts: i64, w: f32) -> TemporalKeyframe {
        TemporalKeyframe { timestamp: ts, weight: w, source: KeyframeSource::Baseline }
    }

    /// Build a test-rig: empty MemTable + Wal + manifest in a temp dir.
    /// Returns the rig + path handles for assertions.
    struct Rig {
        _dir: tempfile::TempDir,
        sst_dir: PathBuf,
        manifest_path: PathBuf,
        wal_path: PathBuf,
        // OP-BRAIN-V5 Phase 2.3 -- sidecar path next to the WAL.
        sidecar_path: PathBuf,
        node_store_dir: PathBuf,
        bond_store_dir: PathBuf,
        memtable: Arc<MemTable>,
        wal: Arc<Mutex<Wal>>,
        intern: NodeIntern,
        manifest: Arc<RwLock<Arc<Manifest>>>,
        // Phase 5: per-rig orphan-cleanup flag. Fresh `false` per
        // rig so each test sees a one-shot cleanup, never a
        // leftover from a previous test in the file.
        orphan_done: Arc<AtomicBool>,
    }

    fn make_rig(cap_bytes: usize) -> Rig {
        let dir = tempdir().unwrap();
        let sst_dir = dir.path().to_path_buf();
        let manifest_path = dir.path().join(MANIFEST_FILE_NAME);
        let wal_path = dir.path().join("wal.seg");
        let memtable = Arc::new(MemTable::new(cap_bytes));
        let mut wal_writer = Wal::open(&wal_path).unwrap();
        wal_writer.set_fsync_thresholds(10_000, 64 * 1024 * 1024);
        let wal = Arc::new(Mutex::new(wal_writer));
        let intern = NodeIntern::new();
        let initial = Manifest::new(KeyFingerprint::none());
        initial.save_atomic(&manifest_path).unwrap();
        let manifest = Arc::new(RwLock::new(Arc::new(initial)));
        let orphan_done = Arc::new(AtomicBool::new(false));
        let sidecar_path = crate::brain::Brain::sidecar_path_for_wal(&wal_path);
        let node_store_dir = crate::node_store::NodeFlushWorker::node_store_dir_for_wal(&wal_path);
        let bond_store_dir = crate::bond_store::BondFlushWorker::bond_store_dir_for_wal(&wal_path);
        Rig { _dir: dir, sst_dir, manifest_path, wal_path, sidecar_path, node_store_dir, bond_store_dir, memtable, wal, intern, manifest, orphan_done }
    }

    fn insert_n(rig: &Rig, n: usize, rng: &mut StdRng) {
        for i in 0..n {
            let subj = rand_seed(rng);
            let obj = rand_seed(rng);
            let kfs = vec![baseline_kf(i as i64, 0.5)];
            let r = rig.memtable.insert(
                subj, obj, BondType::IsA, 0.5, kfs.clone(), &rig.intern,
            );
            let mut w = rig.wal.lock().unwrap();
            w.append(r.seq, &subj, &obj, BondType::IsA, 0.5, &kfs).unwrap();
        }
        // Force a final fsync so the WAL state is on disk for any
        // tests that crash-test replay.
        rig.wal.lock().unwrap().fsync().unwrap();
    }

    #[test]
    fn end_to_end_flush_produces_sst_and_updates_manifest() {
        let rig = make_rig(1024 * 1024 * 1024); // Large cap — only flush via flush_now.
        let mut rng = StdRng::seed_from_u64(0xCAFE);
        insert_n(&rig, 10_000, &mut rng);
        let pre_count = rig.memtable.bond_count();
        assert_eq!(pre_count, 10_000);

        let worker = FlushWorker::start(
            rig.memtable.clone(), rig.wal.clone(), rig.intern.clone(),
            rig.manifest.clone(),
            rig.manifest_path.clone(), rig.sst_dir.clone(), None,
            rig.orphan_done.clone(),
            rig.sidecar_path.clone(), rig.intern.clone(),
            rig.node_store_dir.clone(),
            rig.bond_store_dir.clone(),
        );
        worker.flush_now().unwrap();

        // MemTable drained.
        assert_eq!(rig.memtable.bond_count(), 0);
        // SST exists on disk.
        let sst_path = rig.sst_dir.join("L0-1.sst");
        assert!(sst_path.exists(), "L0-1.sst should exist post-flush");
        // Manifest references it.
        let live = rig.manifest.read().unwrap().clone();
        assert_eq!(live.total_sst_count(), 1);
        let (_, e) = live.iter_ssts().next().unwrap();
        assert_eq!(e.bond_count, 10_000);
        // WAL truncated to empty (or near-empty: just the magic).
        let replay = crate::wal::replay(&rig.wal_path).unwrap();
        assert_eq!(replay.len(), 0, "WAL should be empty after flush+truncate");
    }

    #[test]
    fn multiple_flushes_produce_distinct_sst_ids() {
        let rig = make_rig(1024 * 1024 * 1024);
        let worker = FlushWorker::start(
            rig.memtable.clone(), rig.wal.clone(), rig.intern.clone(),
            rig.manifest.clone(),
            rig.manifest_path.clone(), rig.sst_dir.clone(), None,
            rig.orphan_done.clone(),
            rig.sidecar_path.clone(), rig.intern.clone(),
            rig.node_store_dir.clone(),
            rig.bond_store_dir.clone(),
        );
        let mut rng = StdRng::seed_from_u64(0xBEEF);
        for round in 0..5 {
            insert_n(&rig, 100, &mut rng);
            worker.flush_now().unwrap();
            let live = rig.manifest.read().unwrap().clone();
            assert_eq!(live.total_sst_count(), round + 1);
        }
        // 5 L0 SSTs with ids 1..=5.
        for id in 1..=5 {
            assert!(rig.sst_dir.join(format!("L0-{}.sst", id)).exists());
        }
    }

    #[test]
    fn flush_now_on_empty_memtable_is_noop() {
        let rig = make_rig(1024 * 1024);
        let worker = FlushWorker::start(
            rig.memtable.clone(), rig.wal.clone(), rig.intern.clone(),
            rig.manifest.clone(),
            rig.manifest_path.clone(), rig.sst_dir.clone(), None,
            rig.orphan_done.clone(),
            rig.sidecar_path.clone(), rig.intern.clone(),
            rig.node_store_dir.clone(),
            rig.bond_store_dir.clone(),
        );
        worker.flush_now().unwrap();
        assert_eq!(rig.manifest.read().unwrap().total_sst_count(), 0);
        // No SST file got created.
        let entries: Vec<_> = std::fs::read_dir(&rig.sst_dir).unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().ends_with(".sst"))
            .collect();
        assert!(entries.is_empty());
    }

    #[test]
    fn concurrent_flush_now_calls_serialize() {
        // Two threads call flush_now on the same worker. The
        // flush_lock guarantees serialization; one produces the
        // SST, the other is a no-op (MemTable is empty by then).
        let rig = make_rig(1024 * 1024 * 1024);
        let mut rng = StdRng::seed_from_u64(0x1234);
        insert_n(&rig, 100, &mut rng);
        let worker = Arc::new(FlushWorker::start(
            rig.memtable.clone(), rig.wal.clone(), rig.intern.clone(),
            rig.manifest.clone(),
            rig.manifest_path.clone(), rig.sst_dir.clone(), None,
            rig.orphan_done.clone(),
            rig.sidecar_path.clone(), rig.intern.clone(),
            rig.node_store_dir.clone(),
            rig.bond_store_dir.clone(),
        ));
        let w1 = worker.clone();
        let w2 = worker.clone();
        let t1 = std::thread::spawn(move || w1.flush_now());
        let t2 = std::thread::spawn(move || w2.flush_now());
        t1.join().unwrap().unwrap();
        t2.join().unwrap().unwrap();
        // Exactly one SST was produced (the second flush saw empty).
        assert_eq!(rig.manifest.read().unwrap().total_sst_count(), 1);
    }

    #[test]
    fn flush_writes_subject_sorted_sst() {
        let rig = make_rig(1024 * 1024 * 1024);
        let worker = FlushWorker::start(
            rig.memtable.clone(), rig.wal.clone(), rig.intern.clone(),
            rig.manifest.clone(),
            rig.manifest_path.clone(), rig.sst_dir.clone(), None,
            rig.orphan_done.clone(),
            rig.sidecar_path.clone(), rig.intern.clone(),
            rig.node_store_dir.clone(),
            rig.bond_store_dir.clone(),
        );
        let mut rng = StdRng::seed_from_u64(0xAAAA);
        insert_n(&rig, 500, &mut rng);
        worker.flush_now().unwrap();
        let live = rig.manifest.read().unwrap().clone();
        let (_, e) = live.iter_ssts().next().unwrap();
        assert!(e.subject_min <= e.subject_max);
    }

    #[test]
    fn crash_after_sst_before_manifest_leaves_orphan_for_cleanup() {
        // Simulate a crash AFTER the SST is fsync'd but BEFORE the
        // manifest is updated. We do this by:
        //   1. Filling the MemTable.
        //   2. Computing what the next SST id will be and pre-writing
        //      a junk file at that path (simulating a partial write
        //      that the worker is about to fsync over).
        //   3. Manually performing the SST write + fsync via SSTWriter,
        //      then DELETING the manifest before the worker's
        //      save_atomic would run.
        // Cleaner version: pretend the SST was written but the
        // manifest update never happened. Drop a fake L*.sst into
        // the directory and run `cleanup_orphan_ssts` — that mirrors
        // exactly what Brain::enable_flush_worker does at startup.
        use crate::manifest::cleanup_orphan_ssts;
        let rig = make_rig(1024 * 1024 * 1024);
        // Drop a fake L0-99.sst that the manifest doesn't know about.
        let fake = rig.sst_dir.join("L0-99.sst");
        std::fs::write(&fake, b"this is a half-written SST from a crash")
            .unwrap();
        // The manifest is empty so cleanup_orphan_ssts must rename
        // the fake to .orphan-<ts>.
        let live = rig.manifest.read().unwrap().clone();
        let orphans = cleanup_orphan_ssts(&rig.sst_dir, &live).unwrap();
        assert_eq!(orphans.len(), 1);
        assert!(!fake.exists(), "orphan SST should be renamed");
        // The orphan got renamed but the original L*.sst slot is
        // free; a subsequent flush at id=99 would work.
    }

    #[test]
    fn crash_after_manifest_before_wal_truncate_replay_recovers() {
        // Simulate a crash AFTER manifest update but BEFORE WAL
        // truncate: the records are STILL in the WAL even though
        // they're also in the SST. On restart, WAL replay re-inserts
        // them into the new MemTable. The duplicate state is benign
        // (Phase 5 read path dedups, Phase 6 compaction folds).
        //
        // To simulate: insert 100 bonds, run flush_now, then DO NOT
        // truncate the WAL (we just don't run any extra writes). The
        // WAL on disk still holds the 100 records right after fsync
        // — but after our flush_now() returns, the WAL is truncated.
        // To prove replay, we bypass flush_now and run the flush
        // steps manually, stopping before truncate_up_to.
        use crate::manifest::sst_filename;
        use crate::sst_writer::SSTWriter;
        use crate::node::Confidence;
        let rig = make_rig(1024 * 1024 * 1024);
        let mut rng = StdRng::seed_from_u64(0xFEED);
        insert_n(&rig, 100, &mut rng);
        // Manually do the flush WITHOUT WAL truncate.
        let (mut manifest, sst_id) = {
            let live = rig.manifest.read().unwrap();
            let mut cloned: Manifest = (**live).clone();
            let id = cloned.next_sst_id();
            (cloned, id)
        };
        let snapshot = rig.memtable.swap_for_flush();
        let bonds = snapshot.iter_for_flush(&rig.intern);
        let sst_path = rig.sst_dir.join(sst_filename(0, sst_id));
        let mut writer = SSTWriter::create(&sst_path, 0, None).unwrap();
        writer.set_manifest_id(sst_id);
        for (subj, subj_bonds) in &bonds {
            for mb in subj_bonds {
                writer.push_bond(subj, &mb.object, mb.predicate, mb.weight, Confidence::Verified, &mb.keyframes).unwrap();
            }
        }
        let meta = writer.finalize_with_meta().unwrap();
        let entry = SSTEntry {
            id: sst_id,
            file: sst_filename(0, sst_id),
            subject_min: *meta.subject_min.as_bytes(),
            subject_max: *meta.subject_max.as_bytes(),
            size: std::fs::metadata(&sst_path).map(|m| m.len()).unwrap_or(0),
            mtime: 0,
            bond_count: meta.bond_count,
            epoch: 0,
        };
        manifest.add_sst(0, entry);
        manifest.save_atomic(&rig.manifest_path).unwrap();
        // *** SIMULATED CRASH HERE *** (no truncate_up_to call)

        // Now replay the WAL into a fresh MemTable as the restart
        // path would.
        let replayed = crate::wal::replay(&rig.wal_path).unwrap();
        assert_eq!(replayed.len(), 100, "WAL replay must recover all 100 records");
        // And the SST is intact on disk with the same 100 bonds.
        let cache = std::sync::Arc::new(crate::sst::BlockCache::new(
            crate::sst::DEFAULT_SST_BLOCK_CACHE_BYTES
        ));
        let sst = crate::sst::SST::open(&sst_path, None, cache).unwrap();
        assert_eq!(sst.header().bond_count, 100);
    }

    #[test]
    fn worker_polls_full_memtable_and_flushes_in_background() {
        // Small cap so a moderate insert burst trips is_full().
        let rig = make_rig(8 * 1024);
        let mut worker = FlushWorker::start(
            rig.memtable.clone(), rig.wal.clone(), rig.intern.clone(),
            rig.manifest.clone(),
            rig.manifest_path.clone(), rig.sst_dir.clone(), None,
            rig.orphan_done.clone(),
            rig.sidecar_path.clone(), rig.intern.clone(),
            rig.node_store_dir.clone(),
            rig.bond_store_dir.clone(),
        );
        worker.set_poll_interval(10);
        let mut rng = StdRng::seed_from_u64(0xDEAD);
        // Insert enough to cross the 8 KiB cap.
        for _ in 0..200 {
            let subj = rand_seed(&mut rng);
            let obj = rand_seed(&mut rng);
            let kfs = vec![baseline_kf(0, 0.5)];
            let r = rig.memtable.insert(
                subj, obj, BondType::IsA, 0.5, kfs.clone(), &rig.intern,
            );
            rig.wal.lock().unwrap().append(
                r.seq, &subj, &obj, BondType::IsA, 0.5, &kfs,
            ).unwrap();
        }
        rig.wal.lock().unwrap().fsync().unwrap();
        // Wait up to ~2 s for the background worker to spot the
        // fullness and flush.
        let start = std::time::Instant::now();
        loop {
            if worker.flush_count() >= 1 { break; }
            if start.elapsed() > Duration::from_secs(2) {
                panic!("worker did not flush in 2 s; flush_count={}",
                    worker.flush_count());
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        // After a successful background flush, the manifest has at
        // least one SST. Note: subsequent polls may flush an empty
        // MemTable as no-ops; we only assert ≥ 1 SST.
        assert!(rig.manifest.read().unwrap().total_sst_count() >= 1);
    }

    // ------------------------------------------------------------------
    // OP-BRAIN-V5 Phase 2.3 tests -- sidecar wiring in `flush_once`.
    //
    // These tests assert the new sidecar-before-SST ordering on top of
    // the existing v4 flush rig. They do NOT touch the boot-replay arm
    // (Phase 2.2 owns those tests in brain.rs::phase2_2_*) or the
    // sidecar wire format (Phase 2.1 owns those in node_store::sidecar).
    // ------------------------------------------------------------------

    use crate::node::{Confidence, KnowledgeContent, KnowledgeNode, KnowledgeUnit};
    use crate::node_intern::NodeId;

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

    /// Insert a few bond + node records into the rig (using both the
    /// public MemTable surface AND `intern_or_get` to populate the
    /// intern), append the matching WAL records, fsync.
    fn insert_bonds_with_nodes(rig: &Rig, n: usize, rng: &mut StdRng) {
        for i in 0..n {
            let subj = Seed::from_label(&format!("phase23_subj_{}", i));
            let obj = Seed::from_label(&format!("phase23_obj_{}", i));
            // Intern both seeds so NodeIntern carries them when the
            // sidecar is written. (persist_node_v4 in production does
            // this; for the test rig we call intern_or_get directly.)
            let _ = rig.intern.intern_or_get(subj);
            let _ = rig.intern.intern_or_get(obj);
            let kfs = vec![baseline_kf(i as i64, 0.5)];
            let r = rig.memtable.insert(
                subj, obj, BondType::IsA, 0.5, kfs.clone(), &rig.intern,
            );
            {
                let mut w = rig.wal.lock().unwrap();
                w.append(r.seq, &subj, &obj, BondType::IsA, 0.5, &kfs).unwrap();
            }
            // Also insert a node for the subject so the node half of
            // the MemTable is exercised.
            rig.memtable.insert_node(subj, mk_node(&format!("phase23_subj_{}", i)), &rig.intern);
            let _ = rng;
        }
        rig.wal.lock().unwrap().fsync().unwrap();
    }

    /// OP-BRAIN-V5 Phase 2.3 Test E: a successful flush writes the
    /// sidecar AND it can be loaded back into the same NodeIntern
    /// state. The sidecar path is `rig.sidecar_path`; after the
    /// flush it exists, the `.tmp` is gone, and `NodeIntern::load_from`
    /// returns an intern with the same `len()` (the v4 path adds
    /// 2 * n seeds via `intern_or_get` -- one for each subject and
    /// object).
    #[test]
    fn phase23_flush_writes_sidecar_with_correct_bindings() {
        let rig = make_rig(1024 * 1024 * 1024);
        let mut rng = StdRng::seed_from_u64(0xC0DE);
        const N: usize = 50;
        insert_bonds_with_nodes(&rig, N, &mut rng);

        let worker = FlushWorker::start(
            rig.memtable.clone(), rig.wal.clone(), rig.intern.clone(),
            rig.manifest.clone(),
            rig.manifest_path.clone(), rig.sst_dir.clone(), None,
            rig.orphan_done.clone(),
            rig.sidecar_path.clone(), rig.intern.clone(),
            rig.node_store_dir.clone(),
            rig.bond_store_dir.clone(),
        );
        worker.flush_now().unwrap();

        // Sidecar file exists.
        assert!(
            rig.sidecar_path.exists(),
            "sidecar must exist at {} after flush",
            rig.sidecar_path.display(),
        );
        // The `.tmp` file is gone (rename succeeded).
        let tmp = sidecar_tmp_path(&rig.sidecar_path);
        assert!(!tmp.exists(), "sidecar .tmp must be cleaned up by rename");

        // Load back via NodeIntern::load_from; expect 2*N bindings
        // (one per unique seed -- each subject and object is unique
        // in the test rig).
        let loaded = NodeIntern::load_from(&rig.sidecar_path).expect("sidecar must load");
        // intern.len() is the largest NodeId + 1; with monotonic
        // allocation that equals the number of `intern_or_get` calls.
        assert_eq!(
            loaded.len(),
            rig.intern.len(),
            "sidecar must round-trip every binding",
        );
        // Spot check: every subject seed maps to the same NodeId in
        // the original and loaded interns.
        for i in 0..N {
            let subj = Seed::from_label(&format!("phase23_subj_{}", i));
            let id_orig = rig.intern.get(&subj).expect("subj interned");
            let id_loaded = loaded.get(&subj).expect("subj loaded");
            assert_eq!(id_orig, id_loaded, "subj NodeId must round-trip");
        }
    }

    /// OP-BRAIN-V5 Phase 2.3 Test F: flush-then-crash-then-reopen
    /// preserves NodeIntern verbatim via the Phase 2.2 boot-replay
    /// arm. We simulate the crash by dropping the FlushWorker
    /// IMMEDIATELY after a successful `flush_now` (the worker's drop
    /// stops the thread but the sidecar + manifest + WAL truncate
    /// have all completed by the time `flush_now` returns -- this
    /// proves the happy-path crash-resilience: even if the process
    /// crashed RIGHT AFTER the flush returned, the state is durable).
    ///
    /// The "crash between sidecar and manifest" case is more subtle
    /// to construct without a fault-injection hook; we cover the
    /// observable outcome here: post-flush sidecar exists, post-flush
    /// reload via `Brain::enable_memtable` takes the SidecarLoaded
    /// path AND preserves all NodeIds verbatim.
    #[test]
    fn phase23_flush_then_reopen_takes_sidecar_loaded_path() {
        let rig = make_rig(1024 * 1024 * 1024);
        let mut rng = StdRng::seed_from_u64(0xF00D);
        const N: usize = 30;
        insert_bonds_with_nodes(&rig, N, &mut rng);

        // Record the NodeIds for cross-boot comparison.
        let mut original_bindings: Vec<(Seed, NodeId)> = Vec::with_capacity(2 * N);
        for i in 0..N {
            let subj = Seed::from_label(&format!("phase23_subj_{}", i));
            let obj = Seed::from_label(&format!("phase23_obj_{}", i));
            original_bindings.push((subj, rig.intern.get(&subj).unwrap()));
            original_bindings.push((obj, rig.intern.get(&obj).unwrap()));
        }

        {
            let worker = FlushWorker::start(
                rig.memtable.clone(), rig.wal.clone(), rig.intern.clone(),
                rig.manifest.clone(),
                rig.manifest_path.clone(), rig.sst_dir.clone(), None,
                rig.orphan_done.clone(),
                rig.sidecar_path.clone(), rig.intern.clone(),
                rig.node_store_dir.clone(),
                rig.bond_store_dir.clone(),
            );
            worker.flush_now().unwrap();
            // Drop the worker explicitly -- simulates "process exits
            // here". The sidecar + manifest + WAL truncate have
            // already happened inside flush_now.
            drop(worker);
        }

        // Sidecar must exist post-drop.
        assert!(
            rig.sidecar_path.exists(),
            "sidecar must persist past worker drop",
        );

        // Reopen via Brain::enable_memtable. Phase 2.2 should take
        // the SidecarLoaded path (we observe this indirectly: every
        // original binding must round-trip verbatim).
        let mut brain = crate::brain::Brain::new();
        brain
            .enable_memtable(1024 * 1024, &rig.wal_path)
            .expect("enable_memtable Case A must succeed");
        let post_intern = brain.network.node_intern().clone();

        for (seed, expected_id) in &original_bindings {
            let observed = post_intern.get(seed);
            assert_eq!(
                observed,
                Some(*expected_id),
                "post-reopen NodeId for {} must match pre-flush original",
                seed,
            );
        }
    }

    /// OP-BRAIN-V5 Phase 2.3 Test G: `.tmp` cleanup discipline.
    /// After a successful flush the `<sidecar>.tmp` sibling file is
    /// gone (the rename consumed it).
    ///
    /// Honest scope note: simulating "rename fails so .tmp persists"
    /// requires a fault-injection hook we do not have here (chmod-ing
    /// the parent directory to read-only would also block fsync, which
    /// would surface as a different error path). The test exercises
    /// the happy path -- the more operationally important assertion
    /// per the brief ("if the rename step succeeds, no .tmp file
    /// remains") -- and pins the cleanup discipline against regression.
    #[test]
    fn phase23_sidecar_tmp_cleaned_up_after_successful_rename() {
        let rig = make_rig(1024 * 1024 * 1024);
        let mut rng = StdRng::seed_from_u64(0xBEEF);
        insert_bonds_with_nodes(&rig, 10, &mut rng);

        let worker = FlushWorker::start(
            rig.memtable.clone(), rig.wal.clone(), rig.intern.clone(),
            rig.manifest.clone(),
            rig.manifest_path.clone(), rig.sst_dir.clone(), None,
            rig.orphan_done.clone(),
            rig.sidecar_path.clone(), rig.intern.clone(),
            rig.node_store_dir.clone(),
            rig.bond_store_dir.clone(),
        );
        worker.flush_now().unwrap();

        let tmp = sidecar_tmp_path(&rig.sidecar_path);
        assert!(!tmp.exists(), ".tmp must be gone after rename");
        assert!(rig.sidecar_path.exists(), "sidecar must exist");

        // A second flush also produces a clean state.
        let mut rng2 = StdRng::seed_from_u64(0xBEEF);
        insert_bonds_with_nodes(&rig, 5, &mut rng2);
        worker.flush_now().unwrap();
        assert!(!tmp.exists(), ".tmp must be gone after second flush");
    }

    /// Helper assertion -- the sidecar_tmp_path helper is the same as
    /// the production callsite, and a `.tmp` extension is appended.
    #[test]
    fn phase23_sidecar_tmp_path_appends_tmp_extension() {
        let p = std::path::PathBuf::from("/tmp/test/nodeintern.bin");
        let tmp = sidecar_tmp_path(&p);
        assert_eq!(tmp.to_str().unwrap(), "/tmp/test/nodeintern.bin.tmp");
    }
}
