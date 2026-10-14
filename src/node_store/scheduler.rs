//! OP-BRAIN-V5 -- production node-compaction SCHEDULER.
//!
//! # Why this module exists
//!
//! The v5 boot path (`Brain::load_v5`) wires the Phase 2.6
//! [`crate::node_store::NodeFlushWorker`] (which emits L0 NodeSSTs on
//! every flush) and, since the bond-scheduler commit, the Phase 3.6
//! bond compactor via [`crate::bond_store::BondV5CompactionScheduler`].
//! But the node substrate had the SAME latent-bug class the bond side
//! had: the Phase 2.7 [`crate::node_store::NodeCompactionWorker`] is a
//! one-shot compactor (production code only ever constructed
//! [`CompactionPlan`] by hand in tests) with NO production scheduler.
//! `load_v5` never scheduled it, so during v5 training the node L0
//! NodeSST count grew unbounded: node reads (`Brain::get_node_v5`)
//! had to fall through an ever-growing L0 fan-in, node L0 / RAM
//! bloated, and the level NEVER compacted.
//!
//! This module is the missing scheduler -- the direct analog of
//! [`crate::bond_store::BondV5CompactionScheduler`]. It is a
//! background thread that watches `manifest.node_levels` and, when the
//! L0 NodeSST count reaches `l0_trigger`, runs the REAL Phase 2.7
//! minor compaction (all live L0 -> L1), commits the manifest
//! atomically, swaps the live snapshot, and cleans up the retired
//! source SSTs. It loops as ingest produces more L0.
//!
//! # What it mirrors
//!
//! Structurally identical to
//! [`crate::bond_store::BondV5CompactionScheduler`] (which in turn
//! mirrors the v4 [`crate::compaction::CompactionWorker`]): `start`
//! spawns a poll loop; `worker_loop` cheaply decides whether to act;
//! `compact_once` does the read-clone-merge-save-swap; `stop` (and
//! `Drop`) join the thread. The per-compaction call sequence
//! (`compact` -> `save_atomic` -> `cleanup_orphan_node_ssts`) matches
//! the bond scheduler's flow exactly, driving the real Phase 2.7
//! node compactor rather than a one-shot.
//!
//! # Concurrency model
//!
//! Identical to the shipped bond scheduler. The scheduler shares the
//! Brain's `Arc<RwLock<Arc<Manifest>>>` snapshot pointer with the
//! flush worker and the bond scheduler. `compact_once` reads a
//! snapshot, runs the merge, `save_atomic`s the new manifest, then
//! takes the write lock only briefly to swap the live
//! `Arc<Manifest>`. A `compaction_lock` serializes concurrent
//! `compact_now` calls against the background loop. Node and bond
//! schedulers operate on DISJOINT manifest sub-trees (`node_levels`
//! vs. `bond_v5_levels`) and each only ever swaps in a manifest it
//! derived from the latest snapshot under its own lock, so they
//! coexist on the shared snapshot pointer the same way the flush
//! worker already does; crash and orphan handling is owned by
//! `cleanup_orphan_node_ssts`.
//!
//! # Scope
//!
//! Minor compaction only (L0 -> L1), which keeps node L0 small -- the
//! documented in-scope fix. Major (Ln -> Ln+1, n>=1) compaction is a
//! deliberate follow-up, matching the bond scheduler's scope note.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use crate::brain::BrainError;
use crate::encrypt::MasterKey;
use crate::manifest::Manifest;
use crate::node_store::{
    cleanup_orphan_node_ssts, CompactionPlan, NodeCompactionWorker, NodeSstReaderCache,
};

/// Default L0 NodeSST count at or above which a minor compaction
/// (L0 -> L1) is triggered. Mirrors
/// [`crate::bond_store::DEFAULT_BOND_V5_L0_TRIGGER`] /
/// [`crate::compaction::DEFAULT_L0_TRIGGER`] so the v5 node substrate
/// behaves like the bond/v4 workers the rest of the system was tuned
/// for.
pub const DEFAULT_NODE_V5_L0_TRIGGER: usize = 4;

/// Default poll cadence for the scheduler thread (ms). Mirrors
/// [`crate::bond_store::DEFAULT_BOND_V5_POLL_INTERVAL_MS`].
pub const DEFAULT_NODE_V5_POLL_INTERVAL_MS: u64 = 200;

/// State shared between the public handle and the background thread.
struct NodeV5CompactionShared {
    /// `<brain_dir>/node_store` -- where `.nsst` files live. Taken
    /// from the reader cache so it is guaranteed consistent with the
    /// flush worker's output directory.
    node_store_dir: PathBuf,
    /// Manifest file path for the atomic commit after each compaction.
    manifest_path: PathBuf,
    /// Shared live-manifest snapshot pointer (same Arc the flush
    /// worker, read path, and bond scheduler hold).
    manifest: Arc<RwLock<Arc<Manifest>>>,
    /// Phase 2.8 reader cache; `compact` evicts retired readers from
    /// it after the manifest retire commits.
    reader_cache: Arc<NodeSstReaderCache>,
    /// Per-SST encryption key (`None` = plaintext).
    key: Option<MasterKey>,
    stop_flag: AtomicBool,
    l0_trigger: usize,
    /// Serializes concurrent compactions (background loop vs.
    /// `compact_now`).
    compaction_lock: Arc<Mutex<()>>,
    compactions_run: AtomicU64,
}

/// Background v5 node-compaction scheduler. Spawn via [`start`];
/// stop cleanly via [`stop`] (also handled by `Drop`).
///
/// [`start`]: NodeV5CompactionScheduler::start
/// [`stop`]: NodeV5CompactionScheduler::stop
pub struct NodeV5CompactionScheduler {
    shared: Arc<NodeV5CompactionShared>,
    thread_handle: Option<JoinHandle<()>>,
    poll_interval: Duration,
}

impl NodeV5CompactionScheduler {
    /// Spawn the scheduler. Polls every
    /// [`DEFAULT_NODE_V5_POLL_INTERVAL_MS`] for L0 pile-up and runs a
    /// minor compaction when the trigger is met. `stop()` joins the
    /// thread.
    pub fn start(
        node_store_dir: PathBuf,
        manifest_path: PathBuf,
        manifest: Arc<RwLock<Arc<Manifest>>>,
        reader_cache: Arc<NodeSstReaderCache>,
        key: Option<MasterKey>,
    ) -> Self {
        Self::start_with_trigger(
            node_store_dir,
            manifest_path,
            manifest,
            reader_cache,
            key,
            DEFAULT_NODE_V5_L0_TRIGGER,
        )
    }

    /// Spawn with an explicit L0 trigger. Production uses
    /// [`start`]; tests use a small trigger to trip compaction with
    /// a handful of L0 SSTs.
    ///
    /// [`start`]: NodeV5CompactionScheduler::start
    pub fn start_with_trigger(
        node_store_dir: PathBuf,
        manifest_path: PathBuf,
        manifest: Arc<RwLock<Arc<Manifest>>>,
        reader_cache: Arc<NodeSstReaderCache>,
        key: Option<MasterKey>,
        l0_trigger: usize,
    ) -> Self {
        let shared = Arc::new(NodeV5CompactionShared {
            node_store_dir,
            manifest_path,
            manifest,
            reader_cache,
            key,
            stop_flag: AtomicBool::new(false),
            l0_trigger: l0_trigger.max(1),
            compaction_lock: Arc::new(Mutex::new(())),
            compactions_run: AtomicU64::new(0),
        });
        let poll_interval = Duration::from_millis(DEFAULT_NODE_V5_POLL_INTERVAL_MS);
        let thread_handle = {
            let shared = shared.clone();
            let pi = poll_interval;
            Some(
                thread::Builder::new()
                    .name("lu5d-node-compaction".to_string())
                    .spawn(move || worker_loop(shared, pi))
                    .expect("spawn lu5d-node-compaction thread"),
            )
        };
        NodeV5CompactionScheduler { shared, thread_handle, poll_interval }
    }

    /// OP-BRAIN-V5 -- spawn SHARING an existing manifest-commit lock
    /// (the `Arc<Mutex<()>>` the bond scheduler already holds).
    ///
    /// # Why this exists (root-cause fix for the node-vs-bond race)
    ///
    /// The node and bond schedulers mutate the SAME shared
    /// `Arc<RwLock<Arc<Manifest>>>` via the same sequence: read-clone
    /// the live snapshot, compact, `save_atomic`, swap the WHOLE
    /// `Arc<Manifest>`, then `cleanup_orphan_*` unlinks the retired
    /// source SSTs. Each scheduler previously serialized only against
    /// ITSELF (a private `compaction_lock`). So both could read the
    /// SAME snapshot `S0`, and whichever swapped LAST would publish a
    /// manifest derived from the stale `S0` -- REVERTING the other
    /// scheduler's just-committed sub-tree (`node_levels` vs.
    /// `bond_v5_levels`). Because the reverting swap restores entries
    /// the earlier scheduler had already retired, and that earlier
    /// scheduler's `cleanup_*` had ALREADY unlinked those source
    /// files, the live manifest ends up pointing at deleted files:
    /// the next compaction fails with ENOENT and L0 never drains
    /// (unbounded growth in production; flaky
    /// `node_v5_scheduler_auto_compacts_l0` under test).
    ///
    /// Sharing ONE commit lock across both schedulers forces each to
    /// read its snapshot AFTER the other has fully committed (swap +
    /// cleanup), so no swap is ever derived from a stale base. Uses
    /// the default L0 trigger; this is the production (`load_v5`)
    /// wiring.
    pub fn start_with_commit_lock(
        node_store_dir: PathBuf,
        manifest_path: PathBuf,
        manifest: Arc<RwLock<Arc<Manifest>>>,
        reader_cache: Arc<NodeSstReaderCache>,
        key: Option<MasterKey>,
        commit_lock: Arc<Mutex<()>>,
    ) -> Self {
        let shared = Arc::new(NodeV5CompactionShared {
            node_store_dir,
            manifest_path,
            manifest,
            reader_cache,
            key,
            stop_flag: AtomicBool::new(false),
            l0_trigger: DEFAULT_NODE_V5_L0_TRIGGER.max(1),
            compaction_lock: commit_lock,
            compactions_run: AtomicU64::new(0),
        });
        let poll_interval = Duration::from_millis(DEFAULT_NODE_V5_POLL_INTERVAL_MS);
        let thread_handle = {
            let shared = shared.clone();
            let pi = poll_interval;
            Some(
                thread::Builder::new()
                    .name("lu5d-node-compaction".to_string())
                    .spawn(move || worker_loop(shared, pi))
                    .expect("spawn lu5d-node-compaction thread"),
            )
        };
        NodeV5CompactionScheduler { shared, thread_handle, poll_interval }
    }

    /// Force a synchronous compaction pass. Returns `Ok(())` whether
    /// or not anything actually compacted (L0 below trigger == no-op).
    pub fn compact_now(&self) -> Result<(), BrainError> {
        compact_once(&self.shared)
    }

    /// Stop the worker thread and join it.
    pub fn stop(&mut self) -> Result<(), BrainError> {
        self.shared.stop_flag.store(true, Ordering::SeqCst);
        if let Some(h) = self.thread_handle.take() {
            h.join().map_err(|_| {
                BrainError::Format("lu5d-node-compaction thread panicked".to_string())
            })?;
        }
        Ok(())
    }

    /// Total successful compactions performed by this scheduler.
    pub fn compactions_run(&self) -> u64 {
        self.shared.compactions_run.load(Ordering::SeqCst)
    }

    /// The node-store directory this scheduler operates on.
    pub fn node_store_dir(&self) -> &Path {
        &self.shared.node_store_dir
    }

    /// The configured L0 trigger.
    pub fn l0_trigger(&self) -> usize {
        self.shared.l0_trigger
    }

    /// Poll cadence of the background thread.
    pub fn poll_interval(&self) -> Duration {
        self.poll_interval
    }
}

impl Drop for NodeV5CompactionScheduler {
    fn drop(&mut self) {
        let _ = self.stop();
    }
}

fn worker_loop(shared: Arc<NodeV5CompactionShared>, poll_interval: Duration) {
    loop {
        if shared.stop_flag.load(Ordering::SeqCst) {
            break;
        }
        // Cheap snapshot read to decide whether there is work.
        let should_compact = {
            let live = shared
                .manifest
                .read()
                .expect("manifest snapshot read lock")
                .clone();
            l0_sst_count(&live) >= shared.l0_trigger
        };
        if should_compact {
            if let Err(e) = compact_once(&shared) {
                eprintln!("[lu5d-node-compaction] compact_once failed (non-fatal): {}", e);
            }
        }
        thread::sleep(poll_interval);
    }
}

/// Count of live NodeSSTs at L0 in the given manifest snapshot.
fn l0_sst_count(manifest: &Manifest) -> usize {
    manifest
        .node_levels
        .iter()
        .find(|l| l.level == 0)
        .map(|l| l.ssts.len())
        .unwrap_or(0)
}

/// Live L0 NodeSST `file_uid`s, ascending. The canonical minor-
/// compaction plan drains exactly this set into L1.
fn l0_uids(manifest: &Manifest) -> Vec<u64> {
    let mut uids: Vec<u64> = manifest
        .node_levels
        .iter()
        .find(|l| l.level == 0)
        .map(|l| l.ssts.iter().map(|s| s.file_uid).collect())
        .unwrap_or_default();
    uids.sort_unstable();
    uids
}

/// One canonical minor-compaction step (L0 -> L1). Holds
/// `compaction_lock` for the duration so concurrent `compact_now`
/// calls serialize. No-op (returns `Ok`) when L0 is below trigger.
fn compact_once(shared: &Arc<NodeV5CompactionShared>) -> Result<(), BrainError> {
    let _g = shared
        .compaction_lock
        .lock()
        .expect("node v5 compaction lock poisoned");

    // Read-clone the live snapshot. The node compactor mutates this
    // owned Manifest (retire sources, add output, bump uid counter);
    // we swap it back in after the durable save.
    let live = shared
        .manifest
        .read()
        .expect("manifest snapshot read lock")
        .clone();
    if l0_sst_count(&live) < shared.l0_trigger {
        return Ok(());
    }
    let source_uids = l0_uids(&live);
    if source_uids.is_empty() {
        return Ok(());
    }
    let mut working: Manifest = (*live).clone();

    // Run the REAL Phase 2.7 compactor. It writes the merged
    // `.nsst`, retires the source uids from the manifest, adds the
    // output at L1, and evicts retired readers from the cache.
    let worker = NodeCompactionWorker::new(shared.node_store_dir.clone())
        .with_key(shared.key.clone());
    let plan = CompactionPlan {
        source_uids,
        source_level: 0,
        target_level: 1,
    };
    let result = worker
        .compact(plan, &mut working, Some(&shared.reader_cache))
        .map_err(|e| BrainError::Format(format!("node v5 compaction failed: {}", e)))?;

    // Commit the manifest atomically, then publish the new snapshot
    // so the read path sees the compacted level.
    working.save_atomic(&shared.manifest_path)?;
    {
        let mut w = shared.manifest.write().expect("manifest write lock");
        *w = Arc::new(working);
    }

    // Unlink retired L0 source files (and any orphaned tmp). The
    // manifest no longer references them, so this is safe; open
    // readers keep their fd alive under POSIX until they drop.
    {
        let live_now = shared
            .manifest
            .read()
            .expect("manifest snapshot read lock")
            .clone();
        if let Err(e) = cleanup_orphan_node_ssts(&shared.node_store_dir, &live_now) {
            eprintln!(
                "[lu5d-node-compaction] cleanup_orphan_node_ssts failed (non-fatal): {}",
                e
            );
        }
    }

    shared.compactions_run.fetch_add(1, Ordering::SeqCst);
    eprintln!(
        "[lu5d-node-compaction] minor L0->L1: drained {} L0 SST(s) -> uid {}, {} entries out",
        result.input_entries_retired.len(),
        result.output_entry.file_uid,
        result.entries_out,
    );
    Ok(())
}
