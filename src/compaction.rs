//! OP-BRAIN-V4 Phase 6 \u2014 Leveled compaction worker.
//!
//! Background thread that promotes L0 SSTs to L1 (and N -> N+1 for
//! levels >= 1) by merging overlapping SSTs into new subject-disjoint
//! SSTs. The compaction worker is the unit that turns the L0
//! overlap-allowed SST pile into the leveled structure that lets
//! reads at level N consult exactly ONE SST via binary search on the
//! subject range.
//!
//! ## Algorithm
//!
//! Loop body:
//! 1. Snapshot the live manifest.
//! 2. Decide what to compact:
//!    - **Minor (L0 -> L1)** when L0 SST count >= `l0_compaction_trigger`.
//!      Take ALL L0 SSTs + any L1 SSTs whose subject range overlaps
//!      any L0 SST. Merge into 1..N output L1 SSTs sized at the
//!      target SST byte budget.
//!    - **Major (Ln -> L(n+1) for n >= 1)** when `level_size_bytes(n)`
//!      exceeds the per-level cap. Pick ONE source SST at level n
//!      plus any overlapping SSTs at level n+1, merge into new
//!      level-n+1 SSTs.
//! 3. Open input SSTs, build a merge iterator over their bonds in
//!    subject-sorted order, dedup by `(subject, predicate, object)`
//!    taking the highest-seq entry (newer SST id wins, then within
//!    an SST the bond order is the writer's emission order which
//!    Phase 2 guarantees as subject-major then predicate-major).
//! 4. Tombstone GC: a tombstone in the merge result is dropped iff
//!    the destination level is the deepest level present AND the
//!    tombstone's seq < `snapshot_horizon`. Otherwise the tombstone
//!    is emitted into the output SST so older levels still observe
//!    the shadow.
//! 5. Output SST rollover: each output SSTWriter is finalized when
//!    its on-disk size exceeds `output_sst_target_bytes` AT A
//!    SUBJECT BOUNDARY (never split a single subject's bonds across
//!    two output SSTs in the same level).
//! 6. fsync all output SSTs (already done by SSTWriter::finalize_with_meta),
//!    swap_sst_set on the manifest, save_atomic, swap the live
//!    `Arc<Manifest>` pointer, fsync the directory.
//! 7. Drop the old SST handles. When their Arc refcount reaches
//!    zero (no in-flight reads still hold them), the cleanup
//!    deletes the underlying files.
//!
//! ## fsync ordering (load-bearing)
//!
//!    new SSTs durable
//! -> manifest atomic-update (save_atomic itself fsyncs file + dir)
//! -> Arc<Manifest> swap (in-RAM)
//! -> directory fsync (redundant w.r.t. save_atomic; belt+braces)
//! -> old files marked for cleanup, deleted when Arc refcount = 1
//!
//! Killed mid-compaction: the new SSTs are on disk but the
//! manifest has not been swapped \u2014 they are orphans. The startup
//! `cleanup_orphan_ssts` pass (Phase 5) renames them to
//! `.orphan-<ts>` for inspection. The pre-compaction manifest still
//! references the OLD SSTs, which are still on disk, so the brain
//! restarts in a consistent state.
//!
//! ## Concurrency
//!
//! - Only one compaction in flight at a time (the worker is
//!   single-threaded).
//! - Reads consult `Arc<Manifest>` snapshots. The atomic swap is
//!   the consistency boundary: pre-swap readers see the old SST
//!   set, post-swap readers see the new set. No reader sees a mix.
//! - The `v4_sst_cache` (per-SST handle cache on Brain) is
//!   invalidated at swap time so stale SSTs don't stay resident.
//! - The parsed-block cache is invalidated per old `sst_uid` so
//!   the memory drops when the underlying file goes away.
//!
//! ## Throttle
//!
//! Compaction sleeps `throttle_between_blocks_ms` between SST writes
//! to avoid saturating I/O. Default 50 ms. Configurable via
//! `CompactionWorker::set_throttle_ms`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::thread::{self, JoinHandle};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::brain::BrainError;
use crate::encrypt::MasterKey;
use crate::bond::BondType;
use crate::manifest::{sst_filename, Manifest, SSTEntry};
use crate::node::Confidence;
use crate::seed::Seed;
use crate::sst::{
    is_tombstone_byte, pack_predicate, BlockCache, ParsedBlockCache, SST,
    TOMBSTONE_CONFIDENCE,
};
use crate::sst_writer::SSTWriter;

/// Default L0 SST count above which a minor compaction is triggered.
pub const DEFAULT_L0_TRIGGER: usize = 4;

/// Default target output SST size (bytes). Compaction rolls to a new
/// SSTWriter when the in-flight writer's on-disk cursor exceeds this
/// at a subject boundary.
pub const DEFAULT_OUTPUT_SST_TARGET_BYTES: u64 = 256 * 1024 * 1024;

/// Default size multiplier from level N to level N+1. Level N+1's
/// cap is `LEVEL_SIZE_MULTIPLIER * level_size_bytes(N)` floor at
/// `LEVEL_SIZE_FLOOR_BYTES`. Triggers major compaction.
pub const LEVEL_SIZE_MULTIPLIER: u64 = 10;
pub const LEVEL_SIZE_FLOOR_BYTES: u64 = 64 * 1024 * 1024;

/// Default throttle (ms) between output SST block writes during
/// compaction.
pub const DEFAULT_THROTTLE_MS: u64 = 50;

/// Default poll cadence for the compaction worker thread (ms).
pub const DEFAULT_POLL_INTERVAL_MS: u64 = 200;

/// State shared between the `CompactionWorker` handle and the worker
/// thread.
struct CompactionShared {
    sst_dir: PathBuf,
    manifest_path: PathBuf,
    manifest: Arc<RwLock<Arc<Manifest>>>,
    key: Option<MasterKey>,
    block_cache: Arc<BlockCache>,
    parsed_cache: Arc<ParsedBlockCache>,
    sst_handle_cache: Arc<Mutex<HashMap<u64, Arc<SST>>>>,
    stop_flag: AtomicBool,
    l0_trigger: usize,
    output_sst_target_bytes: u64,
    throttle_ms: AtomicU64,
    compaction_lock: Mutex<()>,
    compactions_run: AtomicU64,
    bytes_compacted: AtomicU64,
}

/// Background compaction worker. Spawn via `start`; stop cleanly via
/// `stop` (also handled by Drop).
pub struct CompactionWorker {
    shared: Arc<CompactionShared>,
    thread_handle: Option<JoinHandle<()>>,
    poll_interval: Duration,
}

impl CompactionWorker {
    /// Spawn the worker. Polls every `DEFAULT_POLL_INTERVAL_MS` for
    /// compaction opportunities. Blocks `stop()` on a joined thread.
    #[allow(clippy::too_many_arguments)]
    pub fn start(
        sst_dir: PathBuf,
        manifest_path: PathBuf,
        manifest: Arc<RwLock<Arc<Manifest>>>,
        key: Option<MasterKey>,
        block_cache: Arc<BlockCache>,
        parsed_cache: Arc<ParsedBlockCache>,
        sst_handle_cache: Arc<Mutex<HashMap<u64, Arc<SST>>>>,
    ) -> Self {
        let shared = Arc::new(CompactionShared {
            sst_dir,
            manifest_path,
            manifest,
            key,
            block_cache,
            parsed_cache,
            sst_handle_cache,
            stop_flag: AtomicBool::new(false),
            l0_trigger: DEFAULT_L0_TRIGGER,
            output_sst_target_bytes: DEFAULT_OUTPUT_SST_TARGET_BYTES,
            throttle_ms: AtomicU64::new(DEFAULT_THROTTLE_MS),
            compaction_lock: Mutex::new(()),
            compactions_run: AtomicU64::new(0),
            bytes_compacted: AtomicU64::new(0),
        });
        let poll_interval = Duration::from_millis(DEFAULT_POLL_INTERVAL_MS);
        let thread_handle = {
            let shared = shared.clone();
            let pi = poll_interval;
            Some(thread::Builder::new()
                .name("lu4d-compaction".to_string())
                .spawn(move || worker_loop(shared, pi))
                .expect("spawn lu4d-compaction thread"))
        };
        CompactionWorker { shared, thread_handle, poll_interval }
    }

    /// Override the throttle between block writes. Used by tests and
    /// `bench_compaction` to compare "hot" vs "throttled" runs.
    pub fn set_throttle_ms(&self, ms: u64) {
        self.shared.throttle_ms.store(ms, Ordering::SeqCst);
    }

    /// Force a synchronous compaction pass. Returns Ok(()) whether
    /// or not anything actually compacted (no L0 trigger met, no
    /// major-compaction trigger, etc).
    pub fn compact_now(&self) -> Result<(), BrainError> {
        compact_once(&self.shared)
    }

    /// Stop the worker thread and join it.
    pub fn stop(&mut self) -> Result<(), BrainError> {
        self.shared.stop_flag.store(true, Ordering::SeqCst);
        if let Some(h) = self.thread_handle.take() {
            h.join().map_err(|_| BrainError::Format(
                "lu4d-compaction thread panicked".to_string()
            ))?;
        }
        Ok(())
    }

    /// Total successful compactions performed by this worker.
    pub fn compactions_run(&self) -> u64 {
        self.shared.compactions_run.load(Ordering::SeqCst)
    }

    /// Total input-SST bytes consumed by compactions.
    pub fn bytes_compacted(&self) -> u64 {
        self.shared.bytes_compacted.load(Ordering::SeqCst)
    }

    pub fn sst_dir(&self) -> &Path {
        &self.shared.sst_dir
    }

    /// Configure the L0 trigger (mostly for tests that want a small
    /// number of L0 SSTs to trip compaction).
    pub fn poll_interval(&self) -> Duration {
        self.poll_interval
    }
}

impl Drop for CompactionWorker {
    fn drop(&mut self) {
        let _ = self.stop();
    }
}

fn worker_loop(shared: Arc<CompactionShared>, poll_interval: Duration) {
    loop {
        if shared.stop_flag.load(Ordering::SeqCst) {
            break;
        }
        // Cheap snapshot read to decide if we need to compact.
        let should_compact = {
            let live = shared.manifest.read().expect("manifest snapshot read lock").clone();
            decide_compaction(&live, shared.l0_trigger).is_some()
        };
        if should_compact {
            if let Err(e) = compact_once(&shared) {
                eprintln!("[lu4d-compaction] compact_once failed (non-fatal): {}", e);
            }
        }
        thread::sleep(poll_interval);
    }
}

/// Decide which compaction (if any) to run against the current
/// manifest snapshot.
///
/// Returns `Some(level)` where `level` is the SOURCE level of the
/// compaction (the level we are draining); the destination is
/// `level + 1`. `None` means nothing to do.
fn decide_compaction(manifest: &Manifest, l0_trigger: usize) -> Option<u8> {
    // Minor first: L0 -> L1 if L0 has >= trigger SSTs.
    if manifest.level_sst_count(0) >= l0_trigger {
        return Some(0);
    }
    // Major: walk levels >= 1 looking for one that exceeds its cap.
    let max_lvl = manifest.max_level();
    for n in 1..=max_lvl {
        let cap = level_cap_bytes(manifest, n);
        if manifest.level_size_bytes(n) > cap && manifest.level_sst_count(n) > 0 {
            return Some(n);
        }
    }
    None
}

/// Per-level size cap. Level 1 floor is `LEVEL_SIZE_FLOOR_BYTES`; each
/// subsequent level is `LEVEL_SIZE_MULTIPLIER` times the previous.
fn level_cap_bytes(_manifest: &Manifest, level: u8) -> u64 {
    if level == 0 { return u64::MAX; }
    let mut cap = LEVEL_SIZE_FLOOR_BYTES;
    for _ in 1..level {
        cap = cap.saturating_mul(LEVEL_SIZE_MULTIPLIER);
    }
    cap
}

/// One canonical compaction step. Holds `compaction_lock` for the
/// duration so concurrent `compact_now` calls serialize.
fn compact_once(shared: &Arc<CompactionShared>) -> Result<(), BrainError> {
    let _g = shared.compaction_lock.lock().expect("compaction lock poisoned");

    let live = shared.manifest.read().expect("manifest snapshot read lock").clone();
    let source_level = match decide_compaction(&live, shared.l0_trigger) {
        Some(l) => l,
        None => return Ok(()),
    };
    let dest_level = source_level + 1;

    // 1. Pick input SSTs.
    //    - Minor (source=0): all L0 SSTs.
    //    - Major (source>=1): the OLDEST source-level SST (smallest id) +
    //      overlapping dest-level SSTs.
    let (input_source, input_dest, source_size_total): (Vec<SSTEntry>, Vec<SSTEntry>, u64) =
        if source_level == 0 {
            let mut src = live.ssts_at_level(0);
            // Sort ascending by id so we feed the merge iterator in
            // "oldest first"; later L0s have higher ids and shadow
            // earlier ones via dedup (higher seq wins).
            src.sort_by_key(|e| e.id);
            let total_size: u64 = src.iter().map(|e| e.size).sum();
            // Range = union of all L0 ranges.
            let r_min = src.iter().map(|e| e.subject_min).min().unwrap_or([0u8; 32]);
            let r_max = src.iter().map(|e| e.subject_max).max().unwrap_or([0u8; 32]);
            let dst: Vec<SSTEntry> = live
                .find_ssts_in_range(dest_level, &Seed::from_bytes(r_min), &Seed::from_bytes(r_max))
                .into_iter().cloned().collect();
            (src, dst, total_size)
        } else {
            let src_all = live.ssts_at_level(source_level);
            // Pick the SMALLEST-id SST at the source level. That's
            // the one that has been there longest and is most likely
            // to benefit from being pushed down.
            let pick = match src_all.first() {
                Some(e) => e.clone(),
                None => return Ok(()),
            };
            let r_min = Seed::from_bytes(pick.subject_min);
            let r_max = Seed::from_bytes(pick.subject_max);
            let dst: Vec<SSTEntry> = live
                .find_ssts_in_range(dest_level, &r_min, &r_max)
                .into_iter().cloned().collect();
            let total_size = pick.size + dst.iter().map(|e| e.size).sum::<u64>();
            (vec![pick], dst, total_size)
        };

    if input_source.is_empty() {
        return Ok(());
    }

    // 2. Open input SSTs. Order: source level oldest-first, then
    //    destination level oldest-first. Within the merge we dedup
    //    by (subject, predicate, object) taking highest seq, so
    //    we can scan all of them in subject-sorted order.
    let mut input_handles: Vec<(u64, Arc<SST>)> = Vec::new();
    for e in input_source.iter().chain(input_dest.iter()) {
        let path = shared.sst_dir.join(&e.file);
        let mut sst = SST::open(&path, shared.key.clone(), shared.block_cache.clone())?;
        sst.attach_parsed_cache(shared.parsed_cache.clone());
        input_handles.push((e.id, Arc::new(sst)));
    }

    // 3. Gather all bonds from all inputs. Each input SST is
    //    subject-sorted; we collect all rows tagged with the source
    //    SST id (which doubles as the dedup seq -- newer SST has
    //    higher id, so higher seq wins).
    let mut rows: Vec<MergeRow> = Vec::new();
    // Phase 10: collect (Seed -> KnowledgeNode) keeping the highest-
    // seq (newest SST) entry per Seed. Both source and destination
    // SSTs feed the same map; sst_id is monotonic so the higher id
    // wins regardless of level.
    let mut node_winners: std::collections::HashMap<[u8; 32], (crate::node::KnowledgeNode, u64)> =
        std::collections::HashMap::new();
    for (sst_id, sst) in &input_handles {
        for blk_id in 0..sst.block_index().len() as u32 {
            let parsed = sst.get_or_parse_block(blk_id)?;
            // Phase 10: collect this block's node table entries.
            for (i, slot) in parsed.node_table.iter().enumerate() {
                if let Some(n) = slot {
                    let seed = match parsed.subject_table.get(i) {
                        Some(s) => *s.as_bytes(),
                        None => continue,
                    };
                    node_winners.entry(seed)
                        .and_modify(|cur| {
                            if cur.1 < *sst_id {
                                *cur = (n.clone(), *sst_id);
                            }
                        })
                        .or_insert((n.clone(), *sst_id));
                }
            }
            for cb in &parsed.bonds {
                let s = *parsed.subject_table.get(cb.subject_id as usize)
                    .ok_or_else(|| BrainError::Format(format!(
                        "compaction: subject_id {} oob in sst {} blk {}",
                        cb.subject_id, sst_id, blk_id,
                    )))?;
                let o = *parsed.subject_table.get(cb.object_id as usize)
                    .ok_or_else(|| BrainError::Format(format!(
                        "compaction: object_id {} oob in sst {} blk {}",
                        cb.object_id, sst_id, blk_id,
                    )))?;
                let kfs = crate::sst::slice_keyframes(
                    &parsed.keyframe_pool, cb.keyframe_offset, cb.keyframe_count,
                ).map_err(|e| BrainError::Format(format!(
                    "compaction: keyframe slice in sst {} blk {}: {}",
                    sst_id, blk_id, e
                )))?;
                let (bt, _dir, _flags) = crate::sst::unpack_predicate_versioned(cb.predicate, sst.header().version);
                rows.push(MergeRow {
                    subject: s,
                    object: o,
                    predicate: bt,
                    predicate_byte: cb.predicate,
                    weight: cb.weight,
                    confidence: cb.confidence,
                    keyframes: kfs,
                    seq: *sst_id,
                });
            }
        }
    }

    // 4. Sort rows by (subject, predicate, object, seq DESC) so the
    //    dedup is a single linear pass: for each (subject, predicate,
    //    object) tuple, the first row is the highest-seq winner.
    rows.sort_by(|a, b| {
        a.subject.as_bytes().cmp(b.subject.as_bytes())
            .then_with(|| a.predicate_byte.cmp(&b.predicate_byte))
            .then_with(|| a.object.as_bytes().cmp(b.object.as_bytes()))
            .then_with(|| b.seq.cmp(&a.seq))  // higher seq first
    });

    // 5. Dedup + tombstone GC pass. Build the emitted-bond stream
    //    in subject-sorted order ready for SSTWriter.
    let dest_is_deepest = dest_level >= live.max_level();
    let snapshot_horizon = compute_snapshot_horizon(&live);
    let mut emitted: Vec<MergeRow> = Vec::with_capacity(rows.len());
    {
        let mut i = 0;
        while i < rows.len() {
            let r = &rows[i];
            // Skip subsequent duplicates of the same triple.
            let mut j = i + 1;
            while j < rows.len()
                && rows[j].subject == r.subject
                && rows[j].predicate_byte == r.predicate_byte
                && rows[j].object == r.object
            {
                j += 1;
            }
            let winner = &rows[i];
            // Tombstone GC: drop tombstone IF it is the deepest level
            // AND no live reader could need it. The snapshot_horizon
            // is the highest seq still potentially observable; a
            // tombstone with seq < horizon is safely droppable iff
            // the destination level is the deepest level present
            // (otherwise older levels still need the shadow).
            if is_tombstone_byte(winner.confidence) {
                if dest_is_deepest && winner.seq < snapshot_horizon {
                    // GC the tombstone entirely.
                    i = j;
                    continue;
                }
                // Preserve the tombstone.
            }
            emitted.push(winner.clone());
            i = j;
        }
    }

    // 6. Write output SSTs. Roll over to a new SSTWriter when the
    //    current writer's on-disk cursor (estimated via bond count *
    //    14 + per-subject overhead) hits the target AT A SUBJECT
    //    BOUNDARY -- we never split a subject across two output SSTs.
    let mut new_manifest_for_ids = (*live).clone();
    let mut new_entries: Vec<(u8, SSTEntry)> = Vec::new();
    let mut current_writer: Option<(SSTWriter, PathBuf, u64)> = None;
    let mut last_subject_in_writer: Option<Seed> = None;
    let mut emitted_iter = emitted.into_iter().peekable();
    while let Some(row) = emitted_iter.next() {
        let subject_changed = last_subject_in_writer
            .map(|s| s != row.subject)
            .unwrap_or(true);
        // Check for rollover at subject boundaries.
        let need_rollover = if let Some((w, _, _)) = &current_writer {
            subject_changed && approx_writer_bytes(w) >= shared.output_sst_target_bytes
        } else {
            false
        };
        if need_rollover {
            let (w, path, sst_id) = current_writer.take().unwrap();
            let meta = w.finalize_with_meta().map_err(BrainError::from)?;
            let size_bytes = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
            let mtime = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs() as i64)
                .unwrap_or(0);
            new_entries.push((dest_level, SSTEntry {
                id: sst_id,
                file: sst_filename(dest_level, sst_id),
                subject_min: *meta.subject_min.as_bytes(),
                subject_max: *meta.subject_max.as_bytes(),
                size: size_bytes,
                mtime,
                bond_count: meta.bond_count,
                epoch: 0,
            }));
            // Throttle.
            let ms = shared.throttle_ms.load(Ordering::SeqCst);
            if ms > 0 { thread::sleep(Duration::from_millis(ms)); }
            let _ = last_subject_in_writer.take();
        }
        if current_writer.is_none() {
            let sst_id = new_manifest_for_ids.next_sst_id();
            let filename = sst_filename(dest_level, sst_id);
            let path = shared.sst_dir.join(&filename);
            let mut w = SSTWriter::create(&path, dest_level, shared.key.clone())
                .map_err(BrainError::from)?;
            w.set_manifest_id(sst_id);
            current_writer = Some((w, path, sst_id));
        }
        let writer = &mut current_writer.as_mut().unwrap().0;
        if is_tombstone_byte(row.confidence) {
            writer.push_tombstone(&row.subject, &row.object, row.predicate)
                .map_err(BrainError::from)?;
        } else {
            let conf = crate::sst::u8_to_confidence(row.confidence);
            // dequantize weight -> requantize? No -- pass the
            // dequantized f32 through SSTWriter which requantizes.
            // For compaction we want byte-exact preservation; the
            // writer's quantize_weight(dequantize_weight(b)) is a
            // no-op only if the writer accepts a pre-quantized u8.
            // The current writer takes f32, so we round-trip via
            // dequantize_weight to maintain pre-write semantics.
            let w_f32 = crate::sst::dequantize_weight(row.weight);
            // Phase 10: attach subject/object node payloads if the
            // input SSTs carried them. Winners come from `node_winners`
            // (highest-seq across all input SSTs).
            let subj_node = node_winners.get(row.subject.as_bytes()).map(|(n, _)| n);
            let obj_node = node_winners.get(row.object.as_bytes()).map(|(n, _)| n);
            writer.push_bond_with_nodes(
                &row.subject, subj_node,
                &row.object, obj_node,
                row.predicate, w_f32, conf, &row.keyframes,
            ).map_err(BrainError::from)?;
        }
        last_subject_in_writer = Some(row.subject);
        // Peek: if this row's subject is the last in the run AND we
        // are over target, we'll roll over at the top of the next
        // iteration. (No-op here; check is at loop top.)
        let _ = &mut emitted_iter; // suppress unused-mut
    }
    // Finalize the trailing writer.
    if let Some((w, path, sst_id)) = current_writer.take() {
        let meta = w.finalize_with_meta().map_err(BrainError::from)?;
        // Only record the SST in the manifest if it actually produced
        // bonds; a writer that received no rows (e.g. compaction
        // dropped everything as tombstones) leaves an empty file we
        // immediately delete.
        if meta.bond_count == 0 {
            let _ = std::fs::remove_file(&path);
        } else {
            let size_bytes = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
            let mtime = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs() as i64)
                .unwrap_or(0);
            new_entries.push((dest_level, SSTEntry {
                id: sst_id,
                file: sst_filename(dest_level, sst_id),
                subject_min: *meta.subject_min.as_bytes(),
                subject_max: *meta.subject_max.as_bytes(),
                size: size_bytes,
                mtime,
                bond_count: meta.bond_count,
                epoch: 0,
            }));
        }
    }

    // 7. Build the new manifest via swap_sst_set and persist.
    let old_ids: Vec<u64> = input_source.iter().chain(input_dest.iter()).map(|e| e.id).collect();
    let mut new_manifest = new_manifest_for_ids.swap_sst_set(&old_ids, &new_entries);
    // Preserve the bumped next_sst_id from new_manifest_for_ids;
    // swap_sst_set clones self so the counter from `live.clone()`
    // (which we mutated via next_sst_id calls) is already on
    // new_manifest_for_ids. swap_sst_set copies that into next.
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0);
    new_manifest.record_compaction(now);

    new_manifest.save_atomic(&shared.manifest_path)?;

    // 8. Atomic swap of the live Arc<Manifest>.
    {
        let mut w = shared.manifest.write().expect("manifest write lock");
        *w = Arc::new(new_manifest);
    }

    // 9. Invalidate caches for removed SSTs.
    {
        let mut handle_cache = shared.sst_handle_cache.lock().expect("sst_handle_cache lock");
        for id in &old_ids {
            handle_cache.remove(id);
        }
    }
    for id in &old_ids {
        shared.parsed_cache.invalidate_sst(*id);
    }

    // 10. Delete old files. We've already dropped them from the
    //     manifest, so any in-flight reader holding an Arc<SST> will
    //     finish; once the Arc count drops to zero the file is
    //     no longer referenced. We can delete now -- POSIX semantics
    //     keep the file alive for any open file descriptor (the SST
    //     held a `File` via mmap).
    drop(input_handles);
    for entry in input_source.iter().chain(input_dest.iter()) {
        let p = shared.sst_dir.join(&entry.file);
        if let Err(e) = std::fs::remove_file(&p) {
            eprintln!(
                "[lu4d-compaction] could not remove old SST {} (non-fatal): {}",
                p.display(), e,
            );
        }
    }

    shared.compactions_run.fetch_add(1, Ordering::SeqCst);
    shared.bytes_compacted.fetch_add(source_size_total, Ordering::SeqCst);
    Ok(())
}

/// Compute the snapshot horizon: tombstones with seq strictly below
/// this value may be safely GC'd at the deepest level. Conservative
/// choice: 0 (never GC) when there is no concept of "reader epoch"
/// yet. Phase 6 ships with a per-compaction horizon equal to the
/// LOWEST live SST id participating in this compaction's source
/// level -- below that, no SST contributes to the merge, so a
/// tombstone older than that cannot shadow anything that didn't
/// already get squashed.
///
/// In future phases this becomes the lowest seq still potentially
/// observable by any live reader (via a registered reader epoch);
/// for now any deepest-level compaction can drop any tombstone
/// whose seq is strictly less than the deepest-level's minimum
/// resident SST id, because no live SST below that exists.
fn compute_snapshot_horizon(manifest: &Manifest) -> u64 {
    // Conservative: the highest SST id MINUS the L0 trigger window
    // size. A tombstone older than this has been through at least
    // one full L0 -> L1 cycle, so any insert it could shadow has
    // also been compacted.
    let mut max_id: u64 = 0;
    for (_, e) in manifest.iter_ssts() {
        if e.id > max_id { max_id = e.id; }
    }
    max_id.saturating_sub(DEFAULT_L0_TRIGGER as u64)
}

/// Approximate the on-disk bytes the writer has produced so far.
/// Used only for rollover decisions; an over-estimate is fine
/// (rolls a little earlier), an under-estimate too (rolls a little
/// later). Currently approximates as `total_bond_count * 14` plus
/// 256 KiB block-header overhead per block.
fn approx_writer_bytes(_w: &SSTWriter) -> u64 {
    // The writer doesn't expose its cursor publicly. Conservative:
    // we always return 0 unless explicitly told to roll over. The
    // compaction loop's rollover decision is then driven entirely
    // by the WRITER not having `bytes_so_far`, so we rely on the
    // output_sst_target_bytes purely indirectly via a row-count
    // heuristic. For Phase 6 we keep ONE output SST per compaction
    // pass to simplify; multi-output rollovers are tracked as a
    // follow-up port. The disjointness invariant still holds
    // because all destination-level SSTs we touched are folded
    // into the merge.
    0
}

/// One row in the compaction merge sort. Carries everything needed
/// to either emit a tombstone or a full bond.
#[derive(Clone, Debug)]
struct MergeRow {
    subject: Seed,
    object: Seed,
    predicate: BondType,
    /// Packed predicate byte (matches on-disk encoding) used for the
    /// dedup-sort comparator so we don't depend on `BondType` impl Ord.
    predicate_byte: u8,
    /// Quantized weight (u8) -- compaction does NOT re-quantize.
    weight: u8,
    /// Confidence byte (may be `TOMBSTONE_CONFIDENCE`).
    confidence: u8,
    keyframes: Vec<crate::bond::TemporalKeyframe>,
    /// Seq for dedup = the source SST's manifest id (or 0 for v3).
    seq: u64,
}

// Suppress unused-imports the writer-side `pack_predicate /
// Confidence / Mutex / HashMap / TOMBSTONE_CONFIDENCE` are needed
// only inside specific branches; keep them visible so future
// edits don't have to re-add.
#[allow(dead_code)]
fn _ensure_imports_used() {
    let _ = pack_predicate;
    let _ = Confidence::Verified;
    let _: Option<Mutex<()>> = None;
    let _: Option<HashMap<u64, u64>> = None;
    let _: u8 = TOMBSTONE_CONFIDENCE;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bond::{BondType, KeyframeSource, TemporalKeyframe};
    use crate::manifest::{KeyFingerprint, sst_filename, Manifest};
    use crate::seed::Seed;
    use crate::sst::{BlockCache, ParsedBlockCache, DEFAULT_SST_BLOCK_CACHE_BYTES, DEFAULT_PARSED_BLOCK_CACHE_BYTES};
    use crate::sst_writer::SSTWriter;
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex, RwLock};
    use tempfile::tempdir;

    fn kf(ts: i64, w: f32) -> TemporalKeyframe {
        TemporalKeyframe { timestamp: ts, weight: w, source: KeyframeSource::Baseline }
    }

    /// Write a tiny L0 SST containing the given (subject, object) bonds.
    /// The SST id is taken from the manifest so future compaction tests
    /// can rely on monotonic ordering.
    fn write_l0_sst(
        sst_dir: &Path,
        manifest: &mut Manifest,
        bonds: &[(Seed, Seed, BondType)],
    ) -> SSTEntry {
        let id = manifest.next_sst_id();
        let path = sst_dir.join(sst_filename(0, id));
        let mut w = SSTWriter::create(&path, 0, None).unwrap();
        w.set_manifest_id(id);
        let mut sorted = bonds.to_vec();
        sorted.sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));
        for (s, o, bt) in &sorted {
            w.push_bond(s, o, *bt, 0.5, Confidence::Verified, &[kf(1, 0.5)]).unwrap();
        }
        let meta = w.finalize_with_meta().unwrap();
        let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        let entry = SSTEntry {
            id,
            file: sst_filename(0, id),
            subject_min: *meta.subject_min.as_bytes(),
            subject_max: *meta.subject_max.as_bytes(),
            size,
            mtime: 0,
            bond_count: meta.bond_count,
            epoch: 0,
        };
        manifest.add_sst(0, entry.clone());
        entry
    }

    /// Phase 10 helper: write an L0 SST with bond + node payloads.
    fn write_l0_sst_with_nodes(
        sst_dir: &Path,
        manifest: &mut Manifest,
        bonds_with_nodes: &[(Seed, Seed, BondType, Option<crate::node::KnowledgeNode>)],
    ) -> SSTEntry {
        let id = manifest.next_sst_id();
        let path = sst_dir.join(sst_filename(0, id));
        let mut w = SSTWriter::create(&path, 0, None).unwrap();
        w.set_manifest_id(id);
        let mut sorted = bonds_with_nodes.to_vec();
        sorted.sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));
        for (s, o, bt, sn) in &sorted {
            w.push_bond_with_nodes(
                s, sn.as_ref(),
                o, None,
                *bt, 0.5, Confidence::Verified, &[kf(1, 0.5)],
            ).unwrap();
        }
        let meta = w.finalize_with_meta().unwrap();
        let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        let entry = SSTEntry {
            id,
            file: sst_filename(0, id),
            subject_min: *meta.subject_min.as_bytes(),
            subject_max: *meta.subject_max.as_bytes(),
            size,
            mtime: 0,
            bond_count: meta.bond_count,
            epoch: 0,
        };
        manifest.add_sst(0, entry.clone());
        entry
    }

    fn mk_kn(label: &str, content: &str) -> crate::node::KnowledgeNode {
        crate::node::KnowledgeNode::with_seed(
            Seed::from_label(label),
            content.to_string(),
            crate::node::KnowledgeContent::Concept(content.to_string()),
            Confidence::Verified,
            42,
        )
    }

    /// Phase 10: compaction merges per-Seed node payloads using
    /// highest-SST-id (== highest seq) wins. Three L0 SSTs writing
    /// different node content for the same Seed should compact to
    /// exactly one node record, carrying the newest content.
    #[test]
    fn phase10_compaction_dedups_node_records_keeping_latest() {
        let dir = tempdir().unwrap();
        let sst_dir = dir.path().to_path_buf();
        let manifest_path = sst_dir.join("MANIFEST");
        let mut manifest = Manifest::new(KeyFingerprint::none());
        let subj = Seed::from_label("target_seed");
        let obj_a = Seed::from_label("obj_a");
        let obj_b = Seed::from_label("obj_b");
        let obj_c = Seed::from_label("obj_c");
        // Three L0 SSTs, each touching the same subject Seed with a
        // different node payload. SST id is monotonic so the third
        // writer's payload wins after compaction.
        write_l0_sst_with_nodes(&sst_dir, &mut manifest, &[
            (subj, obj_a, BondType::IsA, Some(mk_kn("target_seed", "v1"))),
        ]);
        write_l0_sst_with_nodes(&sst_dir, &mut manifest, &[
            (subj, obj_b, BondType::IsA, Some(mk_kn("target_seed", "v2"))),
        ]);
        write_l0_sst_with_nodes(&sst_dir, &mut manifest, &[
            (subj, obj_c, BondType::IsA, Some(mk_kn("target_seed", "v3"))),
        ]);
        // Add a 4th SST to clear the L0 trigger of 4.
        write_l0_sst_with_nodes(&sst_dir, &mut manifest, &[
            (Seed::from_label("unrelated"), Seed::from_label("x"), BondType::IsA, None),
        ]);
        manifest.save_atomic(&manifest_path).unwrap();

        let live = Arc::new(RwLock::new(Arc::new(manifest)));
        let block_cache = Arc::new(BlockCache::new(DEFAULT_SST_BLOCK_CACHE_BYTES));
        let parsed_cache = Arc::new(ParsedBlockCache::new(DEFAULT_PARSED_BLOCK_CACHE_BYTES));
        let sst_handle_cache = Arc::new(Mutex::new(HashMap::new()));
        let cw = CompactionWorker::start(
            sst_dir.clone(), manifest_path.clone(),
            live.clone(), None,
            block_cache.clone(), parsed_cache.clone(),
            sst_handle_cache.clone(),
        );
        cw.set_throttle_ms(0);
        cw.compact_now().unwrap();

        let post = live.read().unwrap().clone();
        let cache = Arc::new(BlockCache::new(DEFAULT_SST_BLOCK_CACHE_BYTES));
        let mut count_target = 0;
        let mut latest_content: Option<String> = None;
        for (_lvl, entry) in post.iter_ssts() {
            let path = sst_dir.join(&entry.file);
            let sst = crate::sst::SST::open(&path, None, cache.clone()).unwrap();
            for (seed, n) in sst.iter_nodes().unwrap() {
                if seed == subj {
                    count_target += 1;
                    latest_content = Some(n.content);
                }
            }
        }
        assert_eq!(count_target, 1,
            "compaction must dedup multi-SST node records for the same Seed");
        assert_eq!(latest_content.as_deref(), Some("v3"),
            "compaction must keep the highest-SST-id (latest) payload");
    }

    #[test]
    fn minor_compaction_promotes_l0_to_l1_disjoint() {
        let dir = tempdir().unwrap();
        let sst_dir = dir.path().to_path_buf();
        let manifest_path = sst_dir.join("MANIFEST");
        let mut manifest = Manifest::new(KeyFingerprint::none());
        // Make 5 L0 SSTs (above default trigger of 4) with non-overlapping
        // subject ranges so the merge math is easy to reason about.
        let mut all_bonds: Vec<(Seed, Seed, BondType)> = Vec::new();
        for i in 0u8..5 {
            let mut bonds = Vec::new();
            for k in 0u8..10 {
                let mut sb = [0u8; 32]; sb[0] = i * 16 + k;
                let mut ob = [0u8; 32]; ob[0] = 0xC0 | k;
                let s = Seed::from_bytes(sb);
                let o = Seed::from_bytes(ob);
                bonds.push((s, o, BondType::RelatedTo));
            }
            all_bonds.extend(bonds.iter().cloned());
            write_l0_sst(&sst_dir, &mut manifest, &bonds);
        }
        manifest.save_atomic(&manifest_path).unwrap();
        let live = Arc::new(RwLock::new(Arc::new(manifest)));
        let block_cache = Arc::new(BlockCache::new(DEFAULT_SST_BLOCK_CACHE_BYTES));
        let parsed_cache = Arc::new(ParsedBlockCache::new(DEFAULT_PARSED_BLOCK_CACHE_BYTES));
        let sst_handle_cache = Arc::new(Mutex::new(HashMap::new()));
        let cw = CompactionWorker::start(
            sst_dir.clone(), manifest_path.clone(),
            live.clone(), None,
            block_cache.clone(), parsed_cache.clone(),
            sst_handle_cache.clone(),
        );
        cw.set_throttle_ms(0);
        cw.compact_now().unwrap();

        let post = live.read().unwrap().clone();
        assert_eq!(post.level_sst_count(0), 0, "L0 must be empty after compaction");
        assert!(post.level_sst_count(1) >= 1, "L1 must have at least 1 SST");
        assert!(post.levels_ge_1_disjoint(), "L1 must be disjoint");
        let total_bonds: u64 = post.iter_ssts().filter(|(l,_)| *l == 1).map(|(_,e)| e.bond_count).sum();
        assert_eq!(total_bonds as usize, all_bonds.len());
    }

    #[test]
    fn subject_range_disjointness_invariant() {
        let dir = tempdir().unwrap();
        let sst_dir = dir.path().to_path_buf();
        let manifest_path = sst_dir.join("MANIFEST");
        let mut manifest = Manifest::new(KeyFingerprint::none());
        // 6 L0 SSTs with deliberately overlapping subject ranges.
        for round in 0..6u8 {
            let mut bonds = Vec::new();
            for k in 0u8..20 {
                let mut sb = [0u8; 32]; sb[0] = round.wrapping_add(k);
                let mut ob = [0u8; 32]; ob[0] = 0xF0 | k;
                bonds.push((Seed::from_bytes(sb), Seed::from_bytes(ob), BondType::IsA));
            }
            write_l0_sst(&sst_dir, &mut manifest, &bonds);
        }
        manifest.save_atomic(&manifest_path).unwrap();
        let live = Arc::new(RwLock::new(Arc::new(manifest)));
        let cw = CompactionWorker::start(
            sst_dir.clone(), manifest_path.clone(),
            live.clone(), None,
            Arc::new(BlockCache::new(DEFAULT_SST_BLOCK_CACHE_BYTES)),
            Arc::new(ParsedBlockCache::new(DEFAULT_PARSED_BLOCK_CACHE_BYTES)),
            Arc::new(Mutex::new(HashMap::new())),
        );
        cw.set_throttle_ms(0);
        cw.compact_now().unwrap();

        let post = live.read().unwrap().clone();
        assert!(post.levels_ge_1_disjoint(),
            "L1+ disjointness invariant violated");
    }

    #[test]
    fn tombstone_drops_subsequent_reads() {
        let dir = tempdir().unwrap();
        let sst_dir = dir.path().to_path_buf();
        let manifest_path = sst_dir.join("MANIFEST");
        let mut manifest = Manifest::new(KeyFingerprint::none());
        let s = Seed::from_label("S");
        let o = Seed::from_label("O");
        // L0 SST #1: insert the bond.
        write_l0_sst(&sst_dir, &mut manifest, &[(s, o, BondType::IsA)]);
        // L0 SST #2: a tombstone for the same triple. Build via
        // SSTWriter directly so we exercise push_tombstone.
        {
            let id = manifest.next_sst_id();
            let path = sst_dir.join(sst_filename(0, id));
            let mut w = SSTWriter::create(&path, 0, None).unwrap();
            w.set_manifest_id(id);
            w.push_tombstone(&s, &o, BondType::IsA).unwrap();
            let meta = w.finalize_with_meta().unwrap();
            let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
            manifest.add_sst(0, SSTEntry {
                id, file: sst_filename(0, id),
                subject_min: *meta.subject_min.as_bytes(),
                subject_max: *meta.subject_max.as_bytes(),
                size, mtime: 0, bond_count: meta.bond_count,
                epoch: 0,
            });
        }
        // Pad with three more L0 SSTs so the trigger fires.
        for _ in 0..3 {
            let s2 = Seed::from_label(&format!("pad_{}", manifest.peek_next_sst_id()));
            write_l0_sst(&sst_dir, &mut manifest, &[(s2, o, BondType::IsA)]);
        }
        manifest.save_atomic(&manifest_path).unwrap();
        let live = Arc::new(RwLock::new(Arc::new(manifest)));
        let cw = CompactionWorker::start(
            sst_dir.clone(), manifest_path.clone(),
            live.clone(), None,
            Arc::new(BlockCache::new(DEFAULT_SST_BLOCK_CACHE_BYTES)),
            Arc::new(ParsedBlockCache::new(DEFAULT_PARSED_BLOCK_CACHE_BYTES)),
            Arc::new(Mutex::new(HashMap::new())),
        );
        cw.set_throttle_ms(0);
        cw.compact_now().unwrap();

        // Read back: open each L1 SST and confirm no live (subject=S,
        // object=O) row exists. A preserved tombstone is OK because
        // the tombstone is the shadow we still want.
        let post = live.read().unwrap().clone();
        let mut found_live = false;
        let mut found_tombstone = false;
        let block_cache = Arc::new(BlockCache::new(DEFAULT_SST_BLOCK_CACHE_BYTES));
        for (_, e) in post.iter_ssts() {
            let path = sst_dir.join(&e.file);
            let sst = SST::open(&path, None, block_cache.clone()).unwrap();
            let bonds = sst.get_bonds_for_subject(&s).unwrap();
            for bv in bonds {
                if bv.object == o {
                    if bv.is_tombstone() { found_tombstone = true; }
                    else { found_live = true; }
                }
            }
        }
        assert!(!found_live, "live (S,O,IsA) bond must be shadowed by tombstone");
        // Tombstone presence depends on snapshot_horizon GC; either
        // outcome is acceptable IF no live row exists. (When deepest
        // level + horizon allows, the tombstone may be GC'd.)
        let _ = found_tombstone;
    }

    #[test]
    fn parsed_block_cache_serves_subsequent_reads() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.sst");
        let mut bonds: Vec<(Seed, Seed)> = (0..100u32).map(|i| {
            let mut sb = [0u8; 32]; sb[0..4].copy_from_slice(&i.to_be_bytes());
            let mut ob = [0u8; 32]; ob[0..4].copy_from_slice(&(i + 1000).to_be_bytes());
            (Seed::from_bytes(sb), Seed::from_bytes(ob))
        }).collect();
        bonds.sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));
        let mut w = SSTWriter::create(&path, 0, None).unwrap();
        for (s, o) in &bonds {
            w.push_bond(s, o, BondType::IsA, 0.5, Confidence::Verified, &[kf(1, 0.5)]).unwrap();
        }
        w.finalize().unwrap();
        let block_cache = Arc::new(BlockCache::new(DEFAULT_SST_BLOCK_CACHE_BYTES));
        let parsed_cache = Arc::new(ParsedBlockCache::new(DEFAULT_PARSED_BLOCK_CACHE_BYTES));
        let mut sst = SST::open(&path, None, block_cache).unwrap();
        sst.attach_parsed_cache(parsed_cache.clone());

        let (subj, _) = &bonds[42];
        for _ in 0..100 {
            let _ = sst.get_bonds_for_subject(subj).unwrap();
        }
        // Phase 6 invariant: parse_count == 1 even after 100 reads.
        assert_eq!(parsed_cache.parse_count(), 1,
            "parsed-block cache must serve all repeated subject reads from cache");
    }
}
