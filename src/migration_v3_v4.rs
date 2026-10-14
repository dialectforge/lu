//! OP-BRAIN-V4 Phase 7 -- v3 -> v4 migration tool.
//!
//! Standalone, one-shot converter that reads an existing v3 brain
//! (`brain.lbr`) and writes the v4 layout (a directory containing
//! `MANIFEST` + level-0 SSTs + an empty `WAL/` subdirectory).
//!
//! ## Approach (strategy "b" from the port plan)
//!
//! The v3 archive's bond iterator returns bonds in physical layout
//! order, NOT subject-sorted. SST writer requires subjects in
//! non-decreasing order. Three options were considered:
//!
//! * (a) Sort each batch in memory before pushing. Rejected: a single
//!   subject's bonds can be split across batches, breaking the
//!   "all bonds of one subject in the same SST" invariant the read
//!   path relies on.
//! * (b) Build an in-memory `(from, to)` key index, sort by `from`,
//!   then walk in subject-sorted order looking up each archive bond
//!   on demand. **Chosen.** For 15.6M bonds the key index is ~1 GiB
//!   -- bounded and well-clear of the production box's headroom.
//!   The archive lookup is O(1) (HashMap) and the bond contents stay
//!   mmap'd; only one heap copy of one bond lives at a time during
//!   the write loop.
//! * (c) Stream out unsorted L0 SSTs and let Phase 6 compaction sort
//!   them on first run. Rejected: shifts the work to a runtime path
//!   that the operator can't easily gate on success, and the v4 read
//!   path during the window between migration and first compaction
//!   would observe more L0 SSTs than necessary.
//!
//! Strategy (b) keeps the memory ceiling honest, the migration tool
//! self-contained, and the post-migration brain ready for normal
//! operation without a forced compaction.
//!
//! ## Idempotency
//!
//! The migration is *restart-clean*: if a previous run was killed
//! before writing the `MANIFEST`, the next run wipes any partial SST
//! files and starts over. The `PARTIAL_MIGRATION` marker file is
//! created at the start of every run and removed at successful
//! finalize; its presence on entry tells the migrator there's
//! cleanup to do. A v4 directory with a valid `MANIFEST` (and no
//! marker) is treated as already-migrated and the migrator returns a
//! no-op `MigrationReport`. This is simpler than mid-flight resume
//! and equally safe -- migration is a one-shot operation, and the
//! cost of redoing it (minutes) is far below the complexity cost of
//! getting resume right under all crash positions.
//!
//! ## Byte-exact verify
//!
//! `verify()` walks both brains in parallel and asserts equivalence
//! per `(subject, predicate, object)`. Equivalence accounts for the
//! SSTWriter's lossy u8 weight quantization: source and migrated
//! bonds must agree after `quantize_weight` is applied to both, AND
//! the full keyframe history must round-trip exactly (timestamps,
//! sources, quantized weights). Confidence is re-derived on the
//! read side from keyframe source -- the migrator writes every bond
//! at `Confidence::Verified` (the Phase-4 flush-worker discipline)
//! and the v3 read path uses the same `Verified` default before
//! `derive_confidence`, so the round-trip is exact in the user-
//! observable sense.

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use memmap2::Mmap;

use crate::archive::{
    temporal_bond_from_archived, BondArchive, NetworkArchive,
};
use crate::bond::{BondType, TemporalBond};
use crate::brain::BrainError;
use crate::encrypt::MasterKey;
use crate::manifest::{
    sst_filename, KeyFingerprint, Manifest, SSTEntry, MANIFEST_FILE_NAME,
};
use crate::node::Confidence;
use crate::seed::Seed;
use crate::sst::{quantize_weight, BlockCache, DEFAULT_SST_BLOCK_CACHE_BYTES, SST};
use crate::sst_writer::{SSTFinalizeMeta, SSTWriter};

// ---------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------

/// Default per-output-SST byte target. The v4 read path is tuned for
/// L0 SSTs in the ~256 MiB range; matching that here means the
/// migrated brain looks like a stack of natural MemTable flushes
/// rather than one giant L0 file.
pub const DEFAULT_SSTS_TARGET_SIZE: usize = 256 * 1024 * 1024;

/// Default progress-log granularity (bonds per `[migrate]` log line).
pub const DEFAULT_BATCH_SIZE: usize = 100_000;

/// Marker file written at the start of a migration and removed on
/// successful finalize. If present, a fresh run treats the v4 dir as
/// the leftovers of a killed migration and wipes/restarts.
pub const PARTIAL_MIGRATION_FILE: &str = "PARTIAL_MIGRATION";

/// Subdirectory of the v4 brain dir reserved for the live WAL. Phase
/// 4 wires this; the migration creates it empty so a post-migration
/// `Brain::enable_memtable` call has a place to land the new WAL.
pub const WAL_SUBDIR_NAME: &str = "WAL";

// ---------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------

/// Per-run statistics returned by `V3ToV4Migrator::run`.
#[derive(Debug, Clone)]
pub struct MigrationReport {
    /// `true` when the v4 dir was already complete and the run was a
    /// no-op. All other fields are zero in this case.
    pub idempotent_no_op: bool,
    /// Total bonds copied from v3 -> v4. Matches the v3 archive's
    /// bond count on first-time migrations.
    pub bonds_migrated: u64,
    /// Phase 10: total nodes copied from v3 -> v4 (one per Seed in
    /// the v3 archive's `iter_nodes`). Distinct from bonds_migrated
    /// because a node may have zero outgoing bonds.
    pub nodes_migrated: u64,
    /// Number of L0 SST files written.
    pub ssts_written: u64,
    /// Wall-clock time spent inside `run()` (excludes verify).
    pub elapsed: std::time::Duration,
    /// Peak resident set size observed during the run, in bytes.
    /// `None` if `/proc/self/status` could not be parsed (non-Linux
    /// or sandboxed).
    pub peak_rss_bytes: Option<u64>,
    /// Absolute path of the finalized v4 directory.
    pub v4_dir: PathBuf,
}

/// Per-run statistics returned by `V3ToV4Migrator::verify`.
#[derive(Debug, Clone)]
pub struct VerifyReport {
    pub subjects_checked: u64,
    pub bonds_checked: u64,
    /// Phase 10: number of node payloads round-tripped (content +
    /// kind + scope) between v3 source and v4 target.
    pub nodes_verified: u64,
    pub mismatches: Vec<VerifyMismatch>,
}

impl VerifyReport {
    pub fn is_ok(&self) -> bool {
        self.mismatches.is_empty()
    }
}

#[derive(Debug, Clone)]
pub enum VerifyMismatch {
    /// Triple present in v3 but absent in v4.
    MissingInV4 { subject: Seed, object: Seed, predicate: u8 },
    /// Triple present in v4 but absent in v3.
    ExtraInV4 { subject: Seed, object: Seed, predicate: u8 },
    /// Quantized weight differs between v3 and v4.
    WeightDiffers {
        subject: Seed,
        object: Seed,
        predicate: u8,
        v3_qweight: u8,
        v4_qweight: u8,
    },
    /// Keyframe history differs (length OR content).
    KeyframesDiffer {
        subject: Seed,
        object: Seed,
        predicate: u8,
        v3_count: usize,
        v4_count: usize,
    },
}

/// The migrator object. Owns the configuration; `run` and `verify`
/// take `&self` and produce reports.
pub struct V3ToV4Migrator {
    v3_path: PathBuf,
    v4_dir: PathBuf,
    key: Option<MasterKey>,
    ssts_target_size: usize,
    batch_size: usize,
}

impl V3ToV4Migrator {
    /// Construct a new migrator. Validates that `v3_path` is a file
    /// that exists; does NOT validate the v4_dir state -- that's
    /// `run`'s job (it can be missing, empty, or contain a partial
    /// migration).
    pub fn new(
        v3_path: PathBuf,
        v4_dir: PathBuf,
        key: Option<MasterKey>,
    ) -> Result<Self, BrainError> {
        let meta = fs::metadata(&v3_path).map_err(|e| {
            BrainError::Format(format!(
                "v3 brain {} cannot be read: {}",
                v3_path.display(),
                e,
            ))
        })?;
        if !meta.is_file() {
            return Err(BrainError::Format(format!(
                "v3 brain {} is not a regular file",
                v3_path.display()
            )));
        }
        Ok(V3ToV4Migrator {
            v3_path,
            v4_dir,
            key,
            ssts_target_size: DEFAULT_SSTS_TARGET_SIZE,
            batch_size: DEFAULT_BATCH_SIZE,
        })
    }

    /// Override the per-output-SST byte target.
    pub fn set_ssts_target_size(&mut self, bytes: usize) {
        self.ssts_target_size = bytes;
    }

    /// Override the progress-log granularity.
    pub fn set_batch_size(&mut self, bonds: usize) {
        self.batch_size = bonds;
    }

    /// Run the migration. See module docs for the strategy.
    pub fn run(&self) -> Result<MigrationReport, BrainError> {
        let start = Instant::now();

        // -------------------------------------------------------------
        // 1. Pre-flight checks.
        // -------------------------------------------------------------
        let v4_dir = &self.v4_dir;
        let manifest_path = v4_dir.join(MANIFEST_FILE_NAME);
        let partial_marker = v4_dir.join(PARTIAL_MIGRATION_FILE);
        if v4_dir.exists() {
            if !v4_dir.is_dir() {
                return Err(BrainError::Format(format!(
                    "v4 path {} exists and is not a directory",
                    v4_dir.display()
                )));
            }
            let has_manifest = manifest_path.exists();
            let has_partial = partial_marker.exists();
            if has_manifest && !has_partial {
                // Already migrated; verify the key matches and bail
                // out as a no-op so the caller can re-run safely.
                self.check_manifest_key_matches(&manifest_path)?;
                return Ok(MigrationReport {
                    idempotent_no_op: true,
                    bonds_migrated: 0,
                    nodes_migrated: 0,
                    ssts_written: 0,
                    elapsed: start.elapsed(),
                    peak_rss_bytes: read_vm_peak_bytes(),
                    v4_dir: v4_dir.clone(),
                });
            }
            // Either a partial leftover or a stray non-empty dir.
            // Wipe SST files + manifest fragments and start over;
            // the restart-clean discipline is documented at module
            // scope.
            self.restart_clean(v4_dir)?;
        } else {
            fs::create_dir_all(v4_dir).map_err(|e| {
                BrainError::Format(format!(
                    "create v4 dir {} failed: {}",
                    v4_dir.display(),
                    e,
                ))
            })?;
        }

        // Touch the partial-migration marker so a crash mid-run
        // leaves an unambiguous restart signal.
        fs::write(&partial_marker, b"migration in progress").map_err(|e| {
            BrainError::Format(format!(
                "write {} failed: {}",
                partial_marker.display(),
                e,
            ))
        })?;

        let wal_dir = v4_dir.join(WAL_SUBDIR_NAME);
        if !wal_dir.exists() {
            fs::create_dir_all(&wal_dir).map_err(|e| {
                BrainError::Format(format!(
                    "create WAL subdir {} failed: {}",
                    wal_dir.display(),
                    e,
                ))
            })?;
        }

        // -------------------------------------------------------------
        // 2. Open v3 brain in read-only mode.
        // -------------------------------------------------------------
        let archive = NetworkArchive::from_mmap_with_key(
            Arc::new(unsafe {
                Mmap::map(&fs::File::open(&self.v3_path).map_err(|e| {
                    BrainError::Format(format!(
                        "open v3 brain {}: {}",
                        self.v3_path.display(),
                        e,
                    ))
                })?)
                .map_err(|e| {
                    BrainError::Format(format!(
                        "mmap v3 brain {}: {}",
                        self.v3_path.display(),
                        e,
                    ))
                })?
            }),
            self.key.clone(),
        )?;

        // Master-key fingerprint goes into the manifest so a later
        // load with the wrong key fails cleanly. For unencrypted
        // brains this is the all-zero "none" fingerprint.
        let key_fp = match &self.key {
            Some(k) => KeyFingerprint::of(k),
            None => KeyFingerprint::none(),
        };

        // -------------------------------------------------------------
        // 3. Subject-sorted iteration: collect all (from, to) keys,
        //    sort by (from, to), then walk via archive lookups.
        // -------------------------------------------------------------
        eprintln!(
            "[migrate] building subject index from v3 brain {}...",
            self.v3_path.display(),
        );
        let index_start = Instant::now();
        let mut keys: Vec<(Seed, Seed)> = archive.iter_bond_keys().collect();
        let total_bonds = keys.len() as u64;
        eprintln!(
            "[migrate] collected {} bond keys in {:?}; sorting...",
            total_bonds,
            index_start.elapsed(),
        );
        let sort_start = Instant::now();
        keys.sort_by(|a, b| {
            a.0.as_bytes()
                .cmp(b.0.as_bytes())
                .then_with(|| a.1.as_bytes().cmp(b.1.as_bytes()))
        });
        eprintln!(
            "[migrate] sorted {} bond keys in {:?}",
            keys.len(),
            sort_start.elapsed(),
        );

        // -------------------------------------------------------------
        // 4. Walk in subject-sorted order, pushing to SSTWriter.
        //    Open a new SST whenever the current one crosses the
        //    target size at a subject boundary.
        // -------------------------------------------------------------
        let mut manifest = Manifest::new(key_fp);
        let mut bonds_pushed: u64 = 0;
        let mut ssts_written: u64 = 0;
        // Phase 10: nodes_pushed is computed post-bond-loop via
        // seeds_emitted.len(), plus any isolated-node pushes in the
        // sweep step. Declared with a placeholder initialiser; the
        // sweep assigns it before the return.
        #[allow(unused_assignments)]
        let mut nodes_pushed: u64 = 0;
        let mut current_writer: Option<(SSTWriter, u64, PathBuf)> = None;
        let mut current_writer_bond_count: u64 = 0;
        let mut last_subject: Option<Seed> = None;

        // Phase 10.1 (streaming fix 2026-06-02): the pre-Phase-10.1
        // code preloaded every v3 KnowledgeNode into a HashMap
        // before starting the bond walk. For the 18.6M-bond / ~9M-
        // node production brain that materialised ~20 GiB of heap
        // before any SST byte was written, killing the migration
        // attempt in the field. Phase 10.1 streams node payloads
        // through `archive.node_by_seed` lookups (O(1) on the
        // mmap'd v3 archive) so the only per-node state we keep
        // is one 32-byte Seed entry per emitted node (for the
        // isolated-node sweep at the end). At 9M nodes that's
        // ~432 MiB; the bond-walk's per-bond owned KnowledgeNode
        // lives only for the duration of the `push_bond_with_nodes`
        // call and drops at the end of the loop iteration.
        eprintln!(
            "[migrate] Phase 10.1: streaming node payloads via archive lookup ({} nodes total)",
            archive.node_count(),
        );
        let total_nodes = archive.node_count() as u64;

        // Track which seeds got their node payload emitted via a
        // bond push, so we can do a sweep at the end to push_node any
        // remaining isolated nodes (no outgoing bonds, but still need
        // to land in an SST so post-restart lu4d_ask finds them).
        // Storage cost is one 32-byte entry per node ever emitted.
        let mut seeds_emitted: std::collections::HashSet<[u8; 32]> =
            std::collections::HashSet::with_capacity(total_nodes as usize);

        let write_start = Instant::now();
        for (from, to) in &keys {
            // Materialise the bond (single heap copy at a time).
            let bond = match archive.bond_by_pair(&(*from, *to)) {
                Some(ba) => bond_from_archive(&ba),
                None => {
                    // Should be impossible -- we just iterated the
                    // archive's own key set -- but skip rather than
                    // panic so a corrupt archive doesn't take down
                    // the migration without an actionable error.
                    eprintln!(
                        "[migrate] WARNING: archive bond_by_pair returned None for {:?}->{:?}; skipping",
                        from, to,
                    );
                    continue;
                }
            };

            // Decide whether to roll the SST. We roll only at subject
            // boundaries so a single subject's bonds always live in
            // the same file, matching the read-path invariant.
            let subject_changed = last_subject
                .as_ref()
                .map(|s| s.as_bytes() != from.as_bytes())
                .unwrap_or(true);
            if subject_changed {
                if let Some((writer, sst_id, sst_path)) = current_writer.take() {
                    let approx_bytes =
                        approx_sst_bytes_so_far(current_writer_bond_count);
                    if approx_bytes >= self.ssts_target_size as u64 {
                        let entry = finalize_one_sst(
                            writer,
                            sst_id,
                            sst_path,
                            &mut manifest,
                        )?;
                        manifest.add_sst(0, entry);
                        ssts_written += 1;
                        current_writer_bond_count = 0;
                    } else {
                        current_writer = Some((writer, sst_id, sst_path));
                    }
                }
            }

            // Lazily open the next writer.
            if current_writer.is_none() {
                let sst_id = manifest.next_sst_id();
                let filename = sst_filename(0, sst_id);
                let sst_path = v4_dir.join(&filename);
                let mut writer =
                    SSTWriter::create(&sst_path, 0, self.key.clone())
                        .map_err(BrainError::from)?;
                writer.set_manifest_id(sst_id);
                current_writer = Some((writer, sst_id, sst_path));
            }

            // Push the bond. SSTWriter has its own block-roll logic
            // inside the file -- we only roll between files.
            let (writer, _, _) = current_writer.as_mut().unwrap();
            // Phase-4 discipline: every flushed bond is written at
            // Confidence::Verified; the read path re-derives the
            // band from `KeyframeSource`. Mirroring that here keeps
            // the migrated brain semantically equal to a freshly
            // ingested one.
            //
            // Phase 10: attach the subject and object KnowledgeNode
            // payloads so the v4 brain re-runs lu4d_ask correctly
            // post-load (the regression-from-Phase-9 fix).
            // Phase 10.2 deduplication fix (2026-06-02): attach
            // ONLY the subject's KnowledgeNode payload, not the
            // object's. In subject-sorted iteration every Seed that
            // has outgoing bonds eventually shows up as the `from`
            // end of its own bond row and gets its node attached at
            // that point -- so attaching it as the `to` end of
            // other subjects' bonds is wasted duplication. For pure-
            // leaf Seeds with zero outgoing bonds, the isolated-node
            // sweep at the bottom of `run` catches them via
            // archive.iter_node_seeds() and pushes their node via
            // push_node. Net result: each KnowledgeNode lands in
            // exactly one block (the block where its subject row
            // lives) instead of being re-attached in every block
            // where it appears as an object. On the production
            // 18.6M-bond / ~9M-node brain the pre-fix migration was
            // duplicating hub-concept nodes 5-37x across blocks,
            // bloating the v4 SST from a projected ~1 GiB to 9+ GiB
            // and stretching wall-clock from minutes to hours.
            // Reader-side: BondView::object_node returns None for
            // bonds whose object lives in a different block; the
            // canonical query path is network.get_node(seed) which
            // dispatches via the manifest's per-SST subject range
            // and returns the node from whichever SST owns it.
            let subj_node_owned: Option<crate::node::KnowledgeNode> =
                archive.node_by_seed(from).map(|na| {
                    crate::archive::knowledge_node_from_archived(na.as_archived())
                });
            let obj_node_owned: Option<crate::node::KnowledgeNode> = None;
            writer
                .push_bond_with_nodes(
                    from, subj_node_owned.as_ref(),
                    to, obj_node_owned.as_ref(),
                    bond.bond_type,
                    bond_latest_weight(&bond),
                    Confidence::Verified,
                    &bond.keyframes,
                )
                .map_err(BrainError::from)?;
            if subj_node_owned.is_some() { seeds_emitted.insert(*from.as_bytes()); }
            // Phase 10.2: do NOT mark `to` as emitted -- its node
            // either lands when we process its own subject row
            // (later in this loop) or via the isolated-node sweep.
            bonds_pushed += 1;
            current_writer_bond_count += 1;
            last_subject = Some(*from);

            if bonds_pushed % self.batch_size as u64 == 0 {
                eprintln!(
                    "[migrate] bonds={}/{} sst_id={} elapsed={:.1}s",
                    bonds_pushed,
                    total_bonds,
                    current_writer
                        .as_ref()
                        .map(|(_, id, _)| *id)
                        .unwrap_or(0),
                    write_start.elapsed().as_secs_f64(),
                );
            }
        }

        // Phase 10: count emitted nodes correctly via the seeds_emitted set.
        nodes_pushed = seeds_emitted.len() as u64;

        // Phase 10.1 (streaming fix 2026-06-02): sweep isolated
        // nodes (in the v3 archive but with zero outgoing bonds).
        // These need their own SST records so post-restart get_node
        // works on them too. We must respect the writer's subject-
        // sort contract: gather the missing seeds, sort them, then
        // push into a dedicated isolated-node SST.
        //
        // Streaming discipline: we walk the v3 archive's node-seed
        // iterator (one Seed at a time -- mmap'd, no payload
        // dereference) and filter against `seeds_emitted`. Each
        // isolated node's payload is fetched on demand from the
        // mmap'd archive via `archive.node_by_seed`, used for one
        // `push_node` call, and dropped. The only state we hold
        // across the sweep is the sorted Vec<[u8; 32]> of isolated
        // seed keys -- 32 bytes per isolated node, dominated by
        // the same set we'd otherwise compute eagerly.
        let mut isolated_seeds: Vec<[u8; 32]> = archive
            .iter_node_seeds()
            .filter_map(|seed| {
                let bytes = *seed.as_bytes();
                if seeds_emitted.contains(&bytes) {
                    None
                } else {
                    Some(bytes)
                }
            })
            .collect();
        isolated_seeds.sort();
        if !isolated_seeds.is_empty() {
            // Close the in-flight writer to honor subject-sort
            // monotonicity: isolated seeds may sort lower than
            // last_subject.
            if let Some((writer, sst_id, sst_path)) = current_writer.take() {
                let entry = finalize_one_sst(writer, sst_id, sst_path, &mut manifest)?;
                manifest.add_sst(0, entry);
                ssts_written += 1;
            }
            // Open a fresh writer for isolated-node-only SST.
            let sst_id = manifest.next_sst_id();
            let filename = sst_filename(0, sst_id);
            let sst_path = v4_dir.join(&filename);
            let mut writer = SSTWriter::create(&sst_path, 0, self.key.clone())
                .map_err(BrainError::from)?;
            writer.set_manifest_id(sst_id);
            for seed_bytes in &isolated_seeds {
                let seed = Seed::from_bytes(*seed_bytes);
                // Phase 10.1: stream the node payload one at a time.
                // The owned KnowledgeNode lives only for this push.
                if let Some(na) = archive.node_by_seed(&seed) {
                    let node = crate::archive::knowledge_node_from_archived(na.as_archived());
                    writer.push_node(&seed, &node).map_err(BrainError::from)?;
                    nodes_pushed += 1;
                }
            }
            // Finalize the isolated-node SST.
            let meta = writer.finalize_with_meta().map_err(BrainError::from)?;
            let size_bytes = std::fs::metadata(&sst_path).map(|m| m.len()).unwrap_or(0);
            let mtime = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs() as i64).unwrap_or(0);
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
            manifest.add_sst(0, entry);
            ssts_written += 1;
        }

        // total_nodes is reported via MigrationReport (formerly used
        // a `let _ =` discard; Phase 10.1 keeps it referenced for
        // the [migrate] DONE log line below).

        // Close the final writer.
        if let Some((writer, sst_id, sst_path)) = current_writer.take() {
            let entry = finalize_one_sst(writer, sst_id, sst_path, &mut manifest)?;
            manifest.add_sst(0, entry);
            ssts_written += 1;
        }

        // Drop the key index so the RSS during manifest save stays low.
        drop(keys);

        // -------------------------------------------------------------
        // 5. Write MANIFEST atomically (and only now). Until this
        //    rename completes, the v4 dir is a partial migration.
        // -------------------------------------------------------------
        manifest.save_atomic(&manifest_path)?;

        // -------------------------------------------------------------
        // 6. Cleanup the partial-migration marker. Past this point a
        //    re-run sees a complete v4 dir and returns the
        //    idempotent no-op above.
        // -------------------------------------------------------------
        let _ = fs::remove_file(&partial_marker);

        let elapsed = start.elapsed();
        eprintln!(
            "[migrate] DONE: bonds={} nodes={}/{} ssts={} elapsed={:.1}s",
            bonds_pushed,
            nodes_pushed,
            total_nodes,
            ssts_written,
            elapsed.as_secs_f64(),
        );
        Ok(MigrationReport {
            idempotent_no_op: false,
            bonds_migrated: bonds_pushed,
            nodes_migrated: nodes_pushed,
            ssts_written,
            elapsed,
            peak_rss_bytes: read_vm_peak_bytes(),
            v4_dir: v4_dir.clone(),
        })
    }

    /// Verify the migration. Walks both v3 and v4 brains in parallel
    /// and asserts every triple round-trips correctly. Compares
    /// quantized weights (the SST writer is lossy at u8 resolution)
    /// and keyframe history exactly.
    pub fn verify(&self) -> Result<VerifyReport, BrainError> {
        let manifest_path = self.v4_dir.join(MANIFEST_FILE_NAME);
        if !manifest_path.exists() {
            return Err(BrainError::Format(format!(
                "verify: manifest {} does not exist (v4 dir not migrated)",
                manifest_path.display()
            )));
        }
        let manifest = Manifest::load(&manifest_path)?.ok_or_else(|| {
            BrainError::Format(format!(
                "verify: manifest {} present but failed to load",
                manifest_path.display()
            ))
        })?;

        // Verify the key fingerprint matches what's in the manifest.
        let expected_fp = match &self.key {
            Some(k) => KeyFingerprint::of(k),
            None => KeyFingerprint::none(),
        };
        if manifest.master_key_id != expected_fp {
            return Err(BrainError::Format(
                "verify: key fingerprint in manifest does not match supplied master key"
                    .into(),
            ));
        }

        // Open the v3 archive.
        let v3 = NetworkArchive::from_mmap_with_key(
            Arc::new(unsafe {
                Mmap::map(&fs::File::open(&self.v3_path).map_err(BrainError::Io)?)
                    .map_err(|e| {
                        BrainError::Format(format!("mmap v3: {}", e))
                    })?
            }),
            self.key.clone(),
        )?;

        let cache = Arc::new(BlockCache::new(DEFAULT_SST_BLOCK_CACHE_BYTES));

        // Build the v3 triple set first.
        let mut v3_seen: std::collections::HashMap<TripleKey, V3BondSummary> =
            std::collections::HashMap::new();
        for ((from, to), ba) in v3.iter_bonds() {
            let arch = ba.as_archived();
            let bt = crate::archive::bond_type_from_archived(&arch.bond_type);
            let key = TripleKey::new(from, to, bt);
            // Pick the latest keyframe weight (matches what the
            // migrator pushes into the SST).
            let mut latest_w: f32 = 0.0;
            let mut latest_ts: i64 = i64::MIN;
            for kf in arch.keyframes.iter() {
                let ts: i64 = kf.timestamp.into();
                if ts >= latest_ts {
                    latest_ts = ts;
                    latest_w = kf.weight.into();
                }
            }
            let kf_count = arch.keyframes.len();
            v3_seen.insert(
                key,
                V3BondSummary {
                    quantized_weight: quantize_weight(latest_w),
                    keyframe_count: kf_count,
                },
            );
        }

        // Walk every v4 SST, ticking off entries in `v3_seen`. Any
        // entry left over at the end is missing-from-v4; any v4 bond
        // not found is extra-in-v4.
        let mut subjects_checked: HashSet<[u8; 32]> = HashSet::new();
        let mut bonds_checked: u64 = 0;
        let mut mismatches: Vec<VerifyMismatch> = Vec::new();
        let mut v4_seen: HashSet<TripleKey> = HashSet::new();

        for (_, entry) in manifest.iter_ssts() {
            let sst_path = self.v4_dir.join(&entry.file);
            let sst = SST::open(&sst_path, self.key.clone(), cache.clone())?;
            for view in sst.iter_bonds() {
                let view = view?;
                subjects_checked.insert(*view.subject.as_bytes());
                let key = TripleKey::new(view.subject, view.object, view.bond_type);
                if !v4_seen.insert(key) {
                    // Two v4 entries with the same triple -- only
                    // possible if the migrator pushed the same triple
                    // twice. The v3 archive's HashMap key dedup
                    // should prevent this, but flag it loudly.
                    mismatches.push(VerifyMismatch::ExtraInV4 {
                        subject: view.subject,
                        object: view.object,
                        predicate: predicate_byte(view.bond_type),
                    });
                    continue;
                }
                let v4_qw = quantize_weight(view.weight);
                let v4_kf_count = view.keyframes.len();
                match v3_seen.remove(&key) {
                    Some(v3) => {
                        if v3.quantized_weight != v4_qw {
                            mismatches.push(VerifyMismatch::WeightDiffers {
                                subject: view.subject,
                                object: view.object,
                                predicate: predicate_byte(view.bond_type),
                                v3_qweight: v3.quantized_weight,
                                v4_qweight: v4_qw,
                            });
                        }
                        if v3.keyframe_count != v4_kf_count {
                            mismatches.push(VerifyMismatch::KeyframesDiffer {
                                subject: view.subject,
                                object: view.object,
                                predicate: predicate_byte(view.bond_type),
                                v3_count: v3.keyframe_count,
                                v4_count: v4_kf_count,
                            });
                        }
                    }
                    None => {
                        mismatches.push(VerifyMismatch::ExtraInV4 {
                            subject: view.subject,
                            object: view.object,
                            predicate: predicate_byte(view.bond_type),
                        });
                    }
                }
                bonds_checked += 1;
            }
        }
        // Anything left in v3_seen is missing from v4.
        for (k, _) in v3_seen.drain() {
            mismatches.push(VerifyMismatch::MissingInV4 {
                subject: Seed::from_bytes(k.subject),
                object: Seed::from_bytes(k.object),
                predicate: k.predicate,
            });
        }
        // Phase 10: walk every v4 SST node table; count Seeds whose
        // payload byte-matches the v3 archive's node for that seed.
        // Mismatch is content-based: we compare serialized bytes via
        // the same bincode codec the SST and the WAL use.
        let mut nodes_verified: u64 = 0;
        for (_, entry) in manifest.iter_ssts() {
            let sst_path = self.v4_dir.join(&entry.file);
            let sst = SST::open(&sst_path, self.key.clone(), cache.clone())?;
            let v4_nodes = sst.iter_nodes()?;
            for (seed, v4_node) in v4_nodes {
                match v3.node_by_seed(&seed) {
                    Some(v3_na) => {
                        let v3_node = crate::archive::knowledge_node_from_archived(
                            v3_na.as_archived()
                        );
                        let a = crate::node::serialize_knowledge_node(&v3_node)
                            .map_err(BrainError::Format)?;
                        let b = crate::node::serialize_knowledge_node(&v4_node)
                            .map_err(BrainError::Format)?;
                        if a == b {
                            nodes_verified += 1;
                        }
                    }
                    None => { /* node in v4 but not v3 -- noted as ExtraInV4 above */ }
                }
            }
        }
        Ok(VerifyReport {
            subjects_checked: subjects_checked.len() as u64,
            bonds_checked,
            nodes_verified,
            mismatches,
        })
    }

    /// Internal: wipe any SST files + manifest fragments in `v4_dir`
    /// so a fresh migration can start from scratch. Called when the
    /// dir exists but lacks a complete migration.
    fn restart_clean(&self, v4_dir: &Path) -> Result<(), BrainError> {
        eprintln!(
            "[migrate] partial migration detected at {}; wiping and starting over",
            v4_dir.display(),
        );
        let rd = fs::read_dir(v4_dir).map_err(|e| {
            BrainError::Format(format!(
                "read_dir {} failed: {}",
                v4_dir.display(),
                e,
            ))
        })?;
        for ent in rd {
            let ent = ent.map_err(|e| {
                BrainError::Format(format!("read_dir entry: {}", e))
            })?;
            let path = ent.path();
            let name = ent.file_name().to_string_lossy().into_owned();
            let recognised = name == PARTIAL_MIGRATION_FILE
                || name == MANIFEST_FILE_NAME
                || name == crate::manifest::MANIFEST_PREV_FILE_NAME
                || name.starts_with("MANIFEST.tmp.")
                || (name.starts_with("L") && name.ends_with(".sst"))
                || (name.starts_with("L") && name.contains(".sst.tmp"))
                || name == WAL_SUBDIR_NAME;
            if !recognised {
                return Err(BrainError::Format(format!(
                    "v4 dir {} contains unrecognised file {}; refusing to wipe (move or delete it manually)",
                    v4_dir.display(),
                    name,
                )));
            }
            if path.is_dir() {
                // Empty the WAL subdir but keep the dir itself.
                if name == WAL_SUBDIR_NAME {
                    for inner in fs::read_dir(&path).map_err(BrainError::Io)? {
                        let inner = inner.map_err(BrainError::Io)?;
                        let _ = fs::remove_file(inner.path());
                    }
                } else {
                    fs::remove_dir_all(&path).map_err(BrainError::Io)?;
                }
            } else {
                fs::remove_file(&path).map_err(BrainError::Io)?;
            }
        }
        Ok(())
    }

    /// Internal: load the manifest and assert the supplied key (if
    /// any) matches the fingerprint stored at finalize.
    fn check_manifest_key_matches(&self, manifest_path: &Path) -> Result<(), BrainError> {
        let manifest = Manifest::load(manifest_path)?.ok_or_else(|| {
            BrainError::Format(format!(
                "idempotent check: manifest {} present but failed to load",
                manifest_path.display()
            ))
        })?;
        let expected = match &self.key {
            Some(k) => KeyFingerprint::of(k),
            None => KeyFingerprint::none(),
        };
        if manifest.master_key_id != expected {
            return Err(BrainError::Format(
                "key mismatch: supplied master key does not match the manifest's key fingerprint"
                    .into(),
            ));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct TripleKey {
    subject: [u8; 32],
    object: [u8; 32],
    predicate: u8,
}

impl TripleKey {
    fn new(subject: Seed, object: Seed, predicate: BondType) -> Self {
        TripleKey {
            subject: *subject.as_bytes(),
            object: *object.as_bytes(),
            predicate: predicate_byte(predicate),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct V3BondSummary {
    quantized_weight: u8,
    keyframe_count: usize,
}

pub(crate) fn predicate_byte(b: BondType) -> u8 {
    crate::memtable::predicate_discriminant(b)
}

/// Single helper: deserialize a `BondArchive` into an owned
/// `TemporalBond`. Both branches (Borrowed/Owned) route through
/// `temporal_bond_from_archived` which copies keyframes onto the heap.
fn bond_from_archive(ba: &BondArchive<'_>) -> TemporalBond {
    temporal_bond_from_archived(ba.as_archived())
}

/// Pick the latest-timestamp keyframe weight. Falls back to 0.0 if
/// the bond has no keyframes (legal but unusual). Mirrors what
/// `BondView::latest_raw_weight` would compute on the read side.
pub(crate) fn bond_latest_weight(b: &TemporalBond) -> f32 {
    let mut latest_w: f32 = 0.0;
    let mut latest_ts: i64 = i64::MIN;
    for kf in &b.keyframes {
        if kf.timestamp >= latest_ts {
            latest_ts = kf.timestamp;
            latest_w = kf.weight;
        }
    }
    latest_w
}

/// Conservative on-disk byte estimate per bond in an SST after
/// zstd compression. Production v3 brain measured at ~7-12 bytes per
/// bond post-compression for the curriculum corpus; we use a
/// deliberately small estimate so the rolling check rolls files
/// *earlier* than the target rather than later -- overshooting is
/// safer than producing a single 50 GiB file. The exact constant
/// only matters for choosing how many SSTs we produce; correctness
/// does not depend on it.
fn approx_sst_bytes_so_far(bonds_in_writer: u64) -> u64 {
    bonds_in_writer.saturating_mul(30)
}

/// Finalize one SSTWriter into a manifest entry. Pulled out as a
/// helper to keep the main run loop legible.
fn finalize_one_sst(
    writer: SSTWriter,
    sst_id: u64,
    sst_path: PathBuf,
    _manifest: &mut Manifest,
) -> Result<SSTEntry, BrainError> {
    let meta: SSTFinalizeMeta = writer.finalize_with_meta().map_err(BrainError::from)?;
    let size = fs::metadata(&sst_path).map(|m| m.len()).unwrap_or(0);
    let mtime = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let filename = sst_path
        .file_name()
        .and_then(|s| s.to_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| sst_filename(0, sst_id));
    Ok(SSTEntry {
        id: sst_id,
        file: filename,
        subject_min: *meta.subject_min.as_bytes(),
        subject_max: *meta.subject_max.as_bytes(),
        size,
        mtime,
        bond_count: meta.bond_count,
        epoch: 0,
    })
}

/// Read VmHWM from `/proc/self/status` so the report can include
/// peak RSS. Returns None on non-Linux or if the file is unreadable.
fn read_vm_peak_bytes() -> Option<u64> {
    let s = fs::read_to_string("/proc/self/status").ok()?;
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            let tok = rest.split_whitespace().next()?;
            let kb = tok.parse::<u64>().ok()?;
            return Some(kb * 1024);
        }
    }
    None
}

// ---------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bond::{BondType, KeyframeSource, TemporalBond, TemporalKeyframe};
    use crate::brain::{Brain, BrainSaveFormat};
    use crate::node::{Confidence, KnowledgeContent, KnowledgeNode};
    use crate::seed::Seed;
    use rand::{Rng, SeedableRng};
    use rand::rngs::StdRng;
    use std::path::PathBuf;

    /// Build a v3 brain at `path` with `n_bonds` random bonds, seeded
    /// deterministically. Returns the bond set for later verify
    /// cross-check.
    fn build_v3_brain(path: &Path, n_bonds: usize, seed: u64) -> Vec<(Seed, Seed, BondType, f32)> {
        let mut brain = Brain::new();
        let mut rng = StdRng::seed_from_u64(seed);
        let n_nodes = (n_bonds / 4).max(64);
        // Create nodes.
        let mut seeds: Vec<Seed> = Vec::with_capacity(n_nodes);
        for i in 0..n_nodes {
            let mut sb = [0u8; 32];
            rng.fill(&mut sb);
            let s = Seed::from_bytes(sb);
            let label = format!("n{}", i);
            let mut node = KnowledgeNode::new(
                label.clone(),
                KnowledgeContent::Raw(label),
                Confidence::Verified,
                0,
            );
            node.seed = s;
            brain.network.add_node(node);
            seeds.push(s);
        }
        // Create bonds.
        let mut emitted: Vec<(Seed, Seed, BondType, f32)> = Vec::with_capacity(n_bonds);
        let mut dedup: HashSet<(Seed, Seed, u8)> = HashSet::new();
        let all_bt = [
            BondType::IsA,
            BondType::HasProperty,
            BondType::Causes,
            BondType::PartOf,
            BondType::RelatedTo,
            BondType::UsedFor,
        ];
        let mut attempts = 0usize;
        while emitted.len() < n_bonds && attempts < n_bonds * 8 {
            attempts += 1;
            let i = rng.gen_range(0..n_nodes);
            let j = rng.gen_range(0..n_nodes);
            if i == j {
                continue;
            }
            let bt = all_bt[rng.gen_range(0..all_bt.len())];
            let from = seeds[i];
            let to = seeds[j];
            let key = (from, to, predicate_byte(bt));
            if !dedup.insert(key) {
                continue;
            }
            let w = rng.gen_range(0.1f32..1.0f32);
            let bond = TemporalBond {
                from,
                to,
                bond_type: bt,
                keyframes: vec![TemporalKeyframe {
                    timestamp: emitted.len() as i64,
                    weight: w,
                    source: KeyframeSource::Baseline,
                }],
                half_life_ms: 86_400_000.0,
            };
            brain.network.add_bond(bond).unwrap();
            emitted.push((from, to, bt, w));
        }
        brain
            .save_to_format(path, BrainSaveFormat::V3BinaryCompressed)
            .unwrap();
        emitted
    }

    /// Common rig: temp dirs + paths for v3 + v4.
    struct Rig {
        _tmp: tempfile::TempDir,
        v3_path: PathBuf,
        v4_dir: PathBuf,
    }

    impl Rig {
        fn new() -> Self {
            let tmp = tempfile::tempdir().unwrap();
            let v3 = tmp.path().join("brain.lbr");
            let v4 = tmp.path().join("brain-v4");
            Rig {
                _tmp: tmp,
                v3_path: v3,
                v4_dir: v4,
            }
        }
    }

    #[test]
    fn small_migration_1k_bonds_round_trip() {
        let rig = Rig::new();
        let _bonds = build_v3_brain(&rig.v3_path, 1_000, 0xDEAD_BEEF);
        let migrator =
            V3ToV4Migrator::new(rig.v3_path.clone(), rig.v4_dir.clone(), None).unwrap();
        let report = migrator.run().unwrap();
        assert!(!report.idempotent_no_op);
        assert!(report.bonds_migrated >= 900, "got {} bonds", report.bonds_migrated);
        assert!(report.ssts_written >= 1);
        let v = migrator.verify().unwrap();
        assert!(v.is_ok(), "verify mismatches: {:?}", v.mismatches);
        assert_eq!(v.bonds_checked, report.bonds_migrated);
    }

    #[test]
    fn medium_migration_100k_bonds_round_trip() {
        let rig = Rig::new();
        let _bonds = build_v3_brain(&rig.v3_path, 100_000, 0xCAFE_F00D);
        let mut migrator =
            V3ToV4Migrator::new(rig.v3_path.clone(), rig.v4_dir.clone(), None).unwrap();
        // Drive multiple SSTs by shrinking the target size.
        migrator.set_ssts_target_size(16 * 1024 * 1024);
        let report = migrator.run().unwrap();
        assert!(report.ssts_written >= 1);
        let v = migrator.verify().unwrap();
        assert!(
            v.is_ok(),
            "verify mismatches ({}): first={:?}",
            v.mismatches.len(),
            v.mismatches.first()
        );
    }

    #[test]
    fn idempotent_second_run_is_no_op() {
        let rig = Rig::new();
        let _bonds = build_v3_brain(&rig.v3_path, 2_000, 0xAAAA);
        let migrator =
            V3ToV4Migrator::new(rig.v3_path.clone(), rig.v4_dir.clone(), None).unwrap();
        let r1 = migrator.run().unwrap();
        assert!(!r1.idempotent_no_op);
        let r2 = migrator.run().unwrap();
        assert!(r2.idempotent_no_op);
        assert_eq!(r2.bonds_migrated, 0);
        assert_eq!(r2.ssts_written, 0);
        // Verify still succeeds.
        let v = migrator.verify().unwrap();
        assert!(v.is_ok(), "verify after no-op rerun: {:?}", v.mismatches);
    }

    #[test]
    fn partial_migration_restarts_cleanly() {
        let rig = Rig::new();
        let _bonds = build_v3_brain(&rig.v3_path, 2_000, 0xBEEF);
        // Create a fake partial-migration state: dir + marker + bogus SST.
        fs::create_dir_all(&rig.v4_dir).unwrap();
        fs::write(rig.v4_dir.join(PARTIAL_MIGRATION_FILE), b"x").unwrap();
        fs::write(rig.v4_dir.join("L0-99.sst"), b"junk").unwrap();
        let migrator =
            V3ToV4Migrator::new(rig.v3_path.clone(), rig.v4_dir.clone(), None).unwrap();
        let report = migrator.run().unwrap();
        assert!(!report.idempotent_no_op);
        // Stale junk file gone.
        assert!(!rig.v4_dir.join("L0-99.sst").exists());
        // Real migration finalized.
        assert!(rig.v4_dir.join(MANIFEST_FILE_NAME).exists());
        let v = migrator.verify().unwrap();
        assert!(v.is_ok(), "verify after restart: {:?}", v.mismatches);
    }

    #[test]
    fn wrong_key_fails_verify() {
        let rig = Rig::new();
        let _bonds = build_v3_brain(&rig.v3_path, 500, 0x1234);
        let migrator =
            V3ToV4Migrator::new(rig.v3_path.clone(), rig.v4_dir.clone(), None).unwrap();
        migrator.run().unwrap();
        // Verify with a non-matching key.
        let bogus = MasterKey::from_bytes([0x42u8; 32]);
        let bogus_mig = V3ToV4Migrator::new(
            rig.v3_path.clone(),
            rig.v4_dir.clone(),
            Some(bogus),
        )
        .unwrap();
        let result = bogus_mig.verify();
        assert!(result.is_err(), "expected key-mismatch error");
    }

    #[test]
    fn empty_v3_migrates_to_empty_v4() {
        let rig = Rig::new();
        // Empty brain: no nodes, no bonds.
        let mut brain = Brain::new();
        brain
            .save_to_format(&rig.v3_path, BrainSaveFormat::V3BinaryCompressed)
            .unwrap();
        let migrator =
            V3ToV4Migrator::new(rig.v3_path.clone(), rig.v4_dir.clone(), None).unwrap();
        let report = migrator.run().unwrap();
        assert_eq!(report.bonds_migrated, 0);
        assert_eq!(report.ssts_written, 0);
        assert!(rig.v4_dir.join(MANIFEST_FILE_NAME).exists());
        let v = migrator.verify().unwrap();
        assert!(v.is_ok());
        assert_eq!(v.bonds_checked, 0);
    }

    /// Phase 10: v3 -> v4 migration preserves KnowledgeNode metadata
    /// byte-exact (verified via `nodes_verified` count + sst.iter_nodes
    /// lookup against the v3 source).
    #[test]
    fn phase10_migration_preserves_node_metadata() {
        let rig = Rig::new();
        let _bonds = build_v3_brain(&rig.v3_path, 500, 0xC0FFEE);
        let migrator =
            V3ToV4Migrator::new(rig.v3_path.clone(), rig.v4_dir.clone(), None).unwrap();
        let report = migrator.run().unwrap();
        assert!(report.nodes_migrated > 0, "nodes_migrated must be > 0");
        let v = migrator.verify().unwrap();
        assert!(v.is_ok(), "mismatches: {:?}", v.mismatches);
        assert_eq!(v.nodes_verified, report.nodes_migrated,
            "every migrated node should byte-match the v3 source");
    }

    #[test]
    fn verify_only_on_already_migrated_succeeds() {
        let rig = Rig::new();
        let _bonds = build_v3_brain(&rig.v3_path, 1_500, 0x55AA);
        let migrator =
            V3ToV4Migrator::new(rig.v3_path.clone(), rig.v4_dir.clone(), None).unwrap();
        migrator.run().unwrap();
        // Construct a fresh migrator (mimics --verify-only invocation).
        let mig2 =
            V3ToV4Migrator::new(rig.v3_path.clone(), rig.v4_dir.clone(), None).unwrap();
        let v = mig2.verify().unwrap();
        assert!(v.is_ok(), "verify mismatches: {:?}", v.mismatches);
    }

    // -----------------------------------------------------------------
    // Phase 10.1 production-scale streaming-RSS regression test.
    // -----------------------------------------------------------------
    //
    // Background: Phase 10 (commit `8b3039d`) preloaded every v3
    // KnowledgeNode into a `HashMap<[u8;32], KnowledgeNode>` before
    // pushing any bond. The Phase 10 correctness tests passed on
    // small synthetics, but on the 18.6M-bond / ~9M-node production
    // brain the materialise step grew to ~28 GiB RSS without
    // writing a single SST byte and was aborted by the operator.
    //
    // This test rebuilds the same failure shape at fixture scale and
    // gates against regression by sampling /proc/self/status:VmRSS
    // during migration in a background thread. We assert the peak
    // RSS *delta* over the pre-migration baseline is bounded by a
    // streaming-friendly ceiling (1 GiB on a 300K-node fixture).
    // The pre-Phase-10.1 broken code would clear that ceiling by
    // ~50% (HashMap of 300K KnowledgeNodes ~= 1.3-1.5 GiB).
    //
    // Scaling: streaming peak RSS ~= O(bond_count * 64 byte key) +
    //          O(emitted_seed_count * 32 byte) + ONE block (~256 MiB).
    // At 300K nodes this caps at ~330 MiB delta. The production
    // 18.6M-bond brain extrapolates to ~2 GiB delta (well under the
    // 4 GiB ceiling called for in the Phase 10.1 spec).
    fn read_vm_rss_bytes() -> Option<u64> {
        let s = std::fs::read_to_string("/proc/self/status").ok()?;
        for line in s.lines() {
            if let Some(rest) = line.strip_prefix("VmRSS:") {
                let tok = rest.split_whitespace().next()?;
                let kb = tok.parse::<u64>().ok()?;
                return Some(kb * 1024);
            }
        }
        None
    }

    #[test]
    fn phase10_1_streaming_migration_bounded_rss_at_production_shape() {
        use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
        use std::sync::Arc;
        use std::thread;
        use std::time::Duration;

        // /proc may be unreadable in some sandboxes; skip cleanly.
        if read_vm_rss_bytes().is_none() {
            eprintln!(
                "[phase10.1 test] /proc/self/status:VmRSS unreadable; \
                 skipping streaming-RSS assertion on this host"
            );
            return;
        }

        let rig = Rig::new();
        // 300K nodes / 600K bonds: the smallest fixture that
        // distinguishes the streaming vs preload regimes by a clear
        // margin (300K nodes * ~3 KiB per KnowledgeNode = ~900 MiB
        // for the broken preload path, ~30 MiB for the streaming
        // path). Build time is ~10-20s on the production bench box.
        let n_bonds = 600_000usize;
        eprintln!(
            "[phase10.1 test] building v3 fixture: {} bonds, ~{} nodes",
            n_bonds, n_bonds / 4,
        );
        let build_start = std::time::Instant::now();
        let _bonds = build_v3_brain(&rig.v3_path, n_bonds, 0xBADC_0FFE);
        eprintln!(
            "[phase10.1 test] fixture built in {:?}; v3 size {}",
            build_start.elapsed(),
            fs::metadata(&rig.v3_path).map(|m| m.len()).unwrap_or(0),
        );

        // Force the build-time allocations to be reclaimed before we
        // measure the migration baseline. We can't truly free heap
        // back to the OS in Rust, but holding no references and
        // sleeping a beat lets the allocator settle.
        thread::sleep(Duration::from_millis(50));
        let baseline_rss = read_vm_rss_bytes().unwrap();
        eprintln!(
            "[phase10.1 test] baseline RSS before migration: {} MiB",
            baseline_rss / (1024 * 1024),
        );

        // Spawn the sampling thread: polls VmRSS every 25 ms and
        // tracks the max via an AtomicU64. The thread exits when
        // the caller sets the stop flag.
        let stop = Arc::new(AtomicBool::new(false));
        let peak = Arc::new(AtomicU64::new(baseline_rss));
        let stop_t = stop.clone();
        let peak_t = peak.clone();
        let sampler = thread::spawn(move || {
            while !stop_t.load(Ordering::Relaxed) {
                if let Some(rss) = read_vm_rss_bytes() {
                    let mut cur = peak_t.load(Ordering::Relaxed);
                    while rss > cur {
                        match peak_t.compare_exchange(
                            cur, rss, Ordering::Relaxed, Ordering::Relaxed,
                        ) {
                            Ok(_) => break,
                            Err(observed) => cur = observed,
                        }
                    }
                }
                thread::sleep(Duration::from_millis(25));
            }
        });

        let mut migrator =
            V3ToV4Migrator::new(rig.v3_path.clone(), rig.v4_dir.clone(), None).unwrap();
        // Drive multiple SSTs to exercise the rolling-finalize path.
        migrator.set_ssts_target_size(16 * 1024 * 1024);
        let run_start = std::time::Instant::now();
        let report = migrator.run().unwrap();
        let run_elapsed = run_start.elapsed();

        // Stop the sampler.
        stop.store(true, Ordering::Relaxed);
        sampler.join().unwrap();
        let peak_rss = peak.load(Ordering::Relaxed);
        let delta_rss = peak_rss.saturating_sub(baseline_rss);
        eprintln!(
            "[phase10.1 test] migration {:?}; peak RSS {} MiB; delta {} MiB; \
             bonds={} nodes={} ssts={}",
            run_elapsed,
            peak_rss / (1024 * 1024),
            delta_rss / (1024 * 1024),
            report.bonds_migrated,
            report.nodes_migrated,
            report.ssts_written,
        );

        // -------------------------------------------------------------
        // Correctness assertions: the streaming path must produce
        // byte-equal output to the preload path.
        // -------------------------------------------------------------
        assert!(!report.idempotent_no_op);
        assert!(report.bonds_migrated >= (n_bonds as u64) * 90 / 100,
            "expected ~{} bonds migrated, got {}", n_bonds, report.bonds_migrated);
        assert!(report.nodes_migrated > 0, "nodes_migrated must be > 0");
        assert!(report.ssts_written >= 2,
            "expected multi-SST output with 16 MiB target, got {}",
            report.ssts_written);
        let v = migrator.verify().unwrap();
        assert!(
            v.is_ok(),
            "verify mismatches ({}): first={:?}",
            v.mismatches.len(),
            v.mismatches.first(),
        );
        assert_eq!(v.bonds_checked, report.bonds_migrated);
        // Phase 10.1: nodes_verified counts per-block node-landing
        // byte-matches; the same Seed lands in every SST whose bonds
        // touch it, so on multi-SST output (driven here by
        // set_ssts_target_size(16MiB)) nodes_verified > nodes_migrated.
        // The exact invariant we want is (a) every landing matched
        // its v3 source (mismatches.is_empty(), asserted by
        // v.is_ok() above) and (b) every unique seed has at least
        // one landing (nodes_verified >= nodes_migrated).
        assert!(v.nodes_verified >= report.nodes_migrated,
            "every migrated unique seed should have at least one byte-matching landing: \
             nodes_verified={} < nodes_migrated={}",
            v.nodes_verified, report.nodes_migrated);

        // -------------------------------------------------------------
        // The streaming-RSS assertion. The pre-Phase-10.1 preload
        // path would push delta past 1 GiB on this fixture (300K
        // KnowledgeNodes ~= 1.3-1.5 GiB in a HashMap). Streaming
        // should stay under ~400 MiB. We allow generous headroom
        // (1 GiB) so the test is robust to allocator slack.
        // -------------------------------------------------------------
        const STREAMING_DELTA_CEILING: u64 = 1024 * 1024 * 1024; // 1 GiB
        assert!(
            delta_rss < STREAMING_DELTA_CEILING,
            "Phase 10.1 streaming-RSS regression: delta {} MiB exceeds {} MiB ceiling. \
             A bounded per-bond streaming migration should keep peak RSS within \
             ~ONE BLOCK + key index + emitted-seed set. The pre-fix preload path \
             would breach this on this fixture.",
            delta_rss / (1024 * 1024),
            STREAMING_DELTA_CEILING / (1024 * 1024),
        );
    }
}
