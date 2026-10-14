//! OP-BRAIN-V5 -- production bond-compaction SCHEDULER.
//!
//! # Why this module exists
//!
//! At HEAD 750eb9c the v5 boot path (`Brain::load_v5`) called
//! `enable_compaction`, which starts only the v4
//! [`crate::compaction::CompactionWorker`]. That worker's
//! `decide_compaction` reads `manifest.levels` (the v4 bond store),
//! which is EMPTY for a migrated/v5 brain -- so it never compacts the
//! v5 substrate. Consequence: the Phase 3.6
//! [`crate::bond_store::BondCompactionWorker`] -- the only code that
//! builds the on-disk reverse index (`<uid>.bsst.reverse`) -- was
//! never scheduled in production. During v5 training, L0 BondSSTs
//! piled up unbounded, reverse indexes were never built, and
//! incoming-edge reads degraded toward a full-scan cliff.
//!
//! This module is the missing scheduler. It is a background thread
//! that watches `manifest.bond_v5_levels` and, when the L0 BondSST
//! count reaches `l0_trigger`, runs the REAL Phase 3.6 minor
//! compaction (all live L0 -> L1), which emits the paired reverse
//! index, commits the manifest atomically, swaps the live snapshot,
//! and cleans up the retired source SSTs. It loops as ingest
//! produces more L0.
//!
//! # What it mirrors
//!
//! Structure mirrors the v4 [`crate::compaction::CompactionWorker`]:
//! `start` spawns a poll loop; `worker_loop` cheaply decides whether
//! to act; `compact_once` does the read-clone-merge-save-swap;
//! `stop` (and `Drop`) join the thread. The per-compaction call
//! sequence (`compact` -> `save_atomic` -> `cleanup_orphan_bond_ssts`)
//! is exactly the flow the untracked
//! `examples/derisk_drive_compaction.rs` harness validated by hand
//! against the real production compactor -- here it is promoted to a
//! proper background worker rather than a one-shot.
//!
//! # Concurrency model
//!
//! Identical to the shipped v4 worker. The scheduler shares the
//! Brain's `Arc<RwLock<Arc<Manifest>>>` snapshot pointer with the
//! flush worker. `compact_once` reads a snapshot, runs the merge,
//! `save_atomic`s the new manifest, then takes the write lock only
//! briefly to swap the live `Arc<Manifest>`. A `compaction_lock`
//! serializes concurrent `compact_now` calls against the background
//! loop. Flush/compaction interleaving follows the same property as
//! the v4 worker (separate `flush_lock` / `compaction_lock`); crash
//! and orphan handling is owned by `cleanup_orphan_bond_ssts`.
//!
//! # Scope
//!
//! Minor compaction only (L0 -> L1), which is what builds the reverse
//! index and keeps L0 small -- the documented in-scope fix. Major
//! (Ln -> Ln+1, n>=1) compaction is a deliberate follow-up, matching
//! the node/bond-store module notes that a level-fill scheduler is a
//! later port.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use crate::bond_store::{
    cleanup_orphan_bond_ssts, BondCompactionPlan, BondCompactionWorker, BondSstReaderCache,
};
use crate::brain::BrainError;
use crate::encrypt::MasterKey;
use crate::manifest::Manifest;

/// Default L0 BondSST count at or above which a minor compaction
/// (L0 -> L1) is triggered. Mirrors
/// [`crate::compaction::DEFAULT_L0_TRIGGER`] so the v5 bond substrate
/// behaves like the v4 worker the rest of the system was tuned for.
pub const DEFAULT_BOND_V5_L0_TRIGGER: usize = 4;

/// Default poll cadence for the scheduler thread (ms). Mirrors
/// [`crate::compaction::DEFAULT_POLL_INTERVAL_MS`].
pub const DEFAULT_BOND_V5_POLL_INTERVAL_MS: u64 = 200;

/// State shared between the public handle and the background thread.
struct BondV5CompactionShared {
    /// `<brain_dir>/bond_store` -- where `.bsst` + `.bsst.reverse`
    /// files live. Taken from the reader cache so it is guaranteed
    /// consistent with the flush worker's output directory.
    bond_store_dir: PathBuf,
    /// Manifest file path for the atomic commit after each compaction.
    manifest_path: PathBuf,
    /// Shared live-manifest snapshot pointer (same Arc the flush
    /// worker and read path hold).
    manifest: Arc<RwLock<Arc<Manifest>>>,
    /// Phase 3.7 reader cache; `compact` evicts retired primary +
    /// reverse readers from it after the manifest retire commits.
    reader_cache: Arc<BondSstReaderCache>,
    /// Per-SST encryption key (`None` = plaintext).
    key: Option<MasterKey>,
    stop_flag: AtomicBool,
    l0_trigger: usize,
    /// Serializes concurrent compactions (background loop vs.
    /// `compact_now`).
    compaction_lock: Arc<Mutex<()>>,
    compactions_run: AtomicU64,
}

/// Background v5 bond-compaction scheduler. Spawn via [`start`];
/// stop cleanly via [`stop`] (also handled by `Drop`).
///
/// [`start`]: BondV5CompactionScheduler::start
/// [`stop`]: BondV5CompactionScheduler::stop
pub struct BondV5CompactionScheduler {
    shared: Arc<BondV5CompactionShared>,
    thread_handle: Option<JoinHandle<()>>,
    poll_interval: Duration,
}

impl BondV5CompactionScheduler {
    /// Spawn the scheduler. Polls every
    /// [`DEFAULT_BOND_V5_POLL_INTERVAL_MS`] for L0 pile-up and runs a
    /// minor compaction when the trigger is met. `stop()` joins the
    /// thread.
    pub fn start(
        bond_store_dir: PathBuf,
        manifest_path: PathBuf,
        manifest: Arc<RwLock<Arc<Manifest>>>,
        reader_cache: Arc<BondSstReaderCache>,
        key: Option<MasterKey>,
    ) -> Self {
        Self::start_with_trigger(
            bond_store_dir,
            manifest_path,
            manifest,
            reader_cache,
            key,
            DEFAULT_BOND_V5_L0_TRIGGER,
        )
    }

    /// Spawn with an explicit L0 trigger. Production uses
    /// [`start`]; tests use a small trigger to trip compaction with
    /// a handful of L0 SSTs.
    ///
    /// [`start`]: BondV5CompactionScheduler::start
    pub fn start_with_trigger(
        bond_store_dir: PathBuf,
        manifest_path: PathBuf,
        manifest: Arc<RwLock<Arc<Manifest>>>,
        reader_cache: Arc<BondSstReaderCache>,
        key: Option<MasterKey>,
        l0_trigger: usize,
    ) -> Self {
        // Non-shared wiring: a private commit lock. `load_v5` instead
        // calls `start_with_commit_lock` to share the flush worker's
        // lock (see below).
        Self::start_with_trigger_and_lock(
            bond_store_dir,
            manifest_path,
            manifest,
            reader_cache,
            key,
            l0_trigger,
            Arc::new(Mutex::new(())),
        )
    }

    /// OP-BRAIN-V5 root-cause fix (flush-vs-compaction stale-base swap,
    /// sibling of 7bd94b8): spawn the scheduler reusing an EXISTING
    /// manifest-commit lock instead of allocating a private one. In
    /// `load_v5` this is the flush worker's `commit_lock`, so flush and
    /// bond compaction serialize their read-clone -> save -> swap
    /// (-> cleanup) against the SAME `Arc<RwLock<Arc<Manifest>>>`; the
    /// node scheduler then reuses it via `commit_lock()`. Uses the
    /// DEFAULT L0 trigger.
    pub fn start_with_commit_lock(
        bond_store_dir: PathBuf,
        manifest_path: PathBuf,
        manifest: Arc<RwLock<Arc<Manifest>>>,
        reader_cache: Arc<BondSstReaderCache>,
        key: Option<MasterKey>,
        commit_lock: Arc<Mutex<()>>,
    ) -> Self {
        Self::start_with_trigger_and_lock(
            bond_store_dir,
            manifest_path,
            manifest,
            reader_cache,
            key,
            DEFAULT_BOND_V5_L0_TRIGGER,
            commit_lock,
        )
    }

    /// Shared spawn path. `commit_lock` becomes the scheduler's
    /// `compaction_lock`: pass a fresh one for an isolated scheduler or
    /// a shared one to serialize against flush + the node scheduler.
    fn start_with_trigger_and_lock(
        bond_store_dir: PathBuf,
        manifest_path: PathBuf,
        manifest: Arc<RwLock<Arc<Manifest>>>,
        reader_cache: Arc<BondSstReaderCache>,
        key: Option<MasterKey>,
        l0_trigger: usize,
        commit_lock: Arc<Mutex<()>>,
    ) -> Self {
        let shared = Arc::new(BondV5CompactionShared {
            bond_store_dir,
            manifest_path,
            manifest,
            reader_cache,
            key,
            stop_flag: AtomicBool::new(false),
            l0_trigger: l0_trigger.max(1),
            compaction_lock: commit_lock,
            compactions_run: AtomicU64::new(0),
        });
        let poll_interval = Duration::from_millis(DEFAULT_BOND_V5_POLL_INTERVAL_MS);
        let thread_handle = {
            let shared = shared.clone();
            let pi = poll_interval;
            Some(
                thread::Builder::new()
                    .name("lu5d-bond-compaction".to_string())
                    .spawn(move || worker_loop(shared, pi))
                    .expect("spawn lu5d-bond-compaction thread"),
            )
        };
        BondV5CompactionScheduler { shared, thread_handle, poll_interval }
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
                BrainError::Format("lu5d-bond-compaction thread panicked".to_string())
            })?;
        }
        Ok(())
    }

    /// Total successful compactions performed by this scheduler.
    pub fn compactions_run(&self) -> u64 {
        self.shared.compactions_run.load(Ordering::SeqCst)
    }

    /// OP-BRAIN-V5 -- the shared manifest-commit lock. The node
    /// scheduler reuses THIS lock so node + bond compactions
    /// serialize their read-clone-save-swap-cleanup against the SAME
    /// `Arc<RwLock<Arc<Manifest>>>` and never swap a manifest derived
    /// from a stale snapshot. See
    /// `crate::node_store::NodeV5CompactionScheduler::start_with_commit_lock`.
    pub fn commit_lock(&self) -> Arc<Mutex<()>> {
        self.shared.compaction_lock.clone()
    }

    /// The bond-store directory this scheduler operates on.
    pub fn bond_store_dir(&self) -> &Path {
        &self.shared.bond_store_dir
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

impl Drop for BondV5CompactionScheduler {
    fn drop(&mut self) {
        let _ = self.stop();
    }
}

/// OP-BRAIN-V5 forced-interleaving probe (TEST-ONLY, env-gated, default
/// OFF). Mirror of the flush-side probe: widens the window between
/// reading the base manifest snapshot and the swap so a concurrent
/// flush can be forced to read the SAME base. Production never
/// compiles this.
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

fn worker_loop(shared: Arc<BondV5CompactionShared>, poll_interval: Duration) {
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
                eprintln!("[lu5d-bond-compaction] compact_once failed (non-fatal): {}", e);
            }
        }
        thread::sleep(poll_interval);
    }
}

/// Count of live BondSSTs at L0 in the given manifest snapshot.
fn l0_sst_count(manifest: &Manifest) -> usize {
    manifest
        .bond_v5_levels
        .iter()
        .find(|l| l.level == 0)
        .map(|l| l.ssts.len())
        .unwrap_or(0)
}

/// Live L0 BondSST `file_uid`s, ascending. The canonical minor-
/// compaction plan drains exactly this set into L1.
fn l0_uids(manifest: &Manifest) -> Vec<u64> {
    let mut uids: Vec<u64> = manifest
        .bond_v5_levels
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
fn compact_once(shared: &Arc<BondV5CompactionShared>) -> Result<(), BrainError> {
    let _g = shared
        .compaction_lock
        .lock()
        .expect("bond v5 compaction lock poisoned");

    // Read-clone the live snapshot. The bond compactor mutates this
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

    // OP-BRAIN-V5 forced-interleaving probe point: between reading the
    // base manifest snapshot and compact->save->swap->cleanup.
    #[cfg(test)]
    race_probe_sleep("LU5_BOND_PROBE_MS");

    // Run the REAL Phase 3.6 compactor. It writes the merged primary
    // `.bsst`, builds the paired `<uid>.bsst.reverse`, retires the
    // source uids from the manifest, adds the output at L1, and
    // evicts retired primary + reverse readers from the cache.
    let worker = BondCompactionWorker::new(shared.bond_store_dir.clone())
        .with_key(shared.key.clone());
    let plan = BondCompactionPlan {
        source_uids,
        source_level: 0,
        target_level: 1,
    };
    let result = worker
        .compact(plan, &mut working, Some(&shared.reader_cache))
        .map_err(|e| BrainError::Format(format!("bond v5 compaction failed: {}", e)))?;

    // Commit the manifest atomically, then publish the new snapshot
    // so the read path sees the compacted level + reverse index.
    working.save_atomic(&shared.manifest_path)?;
    {
        let mut w = shared.manifest.write().expect("manifest write lock");
        *w = Arc::new(working);
    }

    // Unlink retired L0 source files (and any orphaned reverse/tmp).
    // The manifest no longer references them, so this is safe; open
    // readers keep their fd alive under POSIX until they drop.
    {
        let live_now = shared
            .manifest
            .read()
            .expect("manifest snapshot read lock")
            .clone();
        if let Err(e) = cleanup_orphan_bond_ssts(&shared.bond_store_dir, &live_now) {
            eprintln!(
                "[lu5d-bond-compaction] cleanup_orphan_bond_ssts failed (non-fatal): {}",
                e
            );
        }
    }

    shared.compactions_run.fetch_add(1, Ordering::SeqCst);
    eprintln!(
        "[lu5d-bond-compaction] minor L0->L1: drained {} L0 SST(s) -> uid {} (reverse {:?}), {} entries out",
        result.input_entries_retired.len(),
        result.output_entry.file_uid,
        result.output_entry.reverse_file_uid,
        result.entries_out,
    );
    Ok(())
}
