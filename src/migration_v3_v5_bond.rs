//! OP-BRAIN-V5 Phase 3.10 -- v3 -> v5 BOND-SIDE migration.
//!
//! Standalone, one-shot migration that reads an existing v3 brain
//! file (`brain.lbr`, encrypted or plain) and emits the BOND half of
//! a v5 brain directory ALONGSIDE the node half produced by the
//! Phase 2.9 node tool (`lu_brain_migrate_v3_v5_node`):
//!
//! ```text
//! <v5_dir>/                 (created by Phase 2.9; this tool ADDS to it)
//!   MANIFEST                -- updated in place: bond_v5_levels[0] populated
//!   wal/
//!     nodeintern.bin        -- READ ONLY here; never rewritten
//!   node_store/             -- READ ONLY here; never touched
//!     *.nsst
//!   bond_store/             -- NEW: this tool's output
//!     <uid>.bsst            -- L0 BondSST files (Phase 3.1 writer format)
//!     ...
//! ```
//!
//! This mirrors the Phase 2.9 node tool exactly (Q2: standalone bond
//! migrator run AFTER the node tool, NOT a combined migrator). The
//! bond tool REQUIRES the node-migrated dir to already exist so that
//! bonds reference the SAME NodeIds Phase 2.9 assigned -- it loads
//! `wal/nodeintern.bin` and maps every v3 Seed endpoint through it.
//!
//! ## Forward bonds only (Q1)
//!
//! This tool emits FORWARD bonds only -- one `(subject, predicate,
//! object)` row per v3 bond, sorted ascending. The reverse index
//! (`.bsst.reverse`, for fast incoming-bond reads) is intentionally
//! NOT produced here: every L0 `BondSSTEntry.reverse_file_uid` is
//! left `None`. Reverse pairs are emitted later by the existing
//! `BondCompactionWorker` (Phase 3.6) during a post-migration
//! compaction pass. Keeping the migrator single-purpose matches the
//! L0-flush path (Phase 3.3), where flushes also leave reverse
//! unbuilt until compaction.
//!
//! ## External merge sort (Q3)
//!
//! ~39M bonds with inline keyframes is too close to the 20 GB
//! ingest-unit cap to risk an in-memory `Vec` sort (unlike the
//! Phase 2.9 node sort, which is a bounded ~600 MB `Vec<(NodeId,
//! Seed)>`). Instead we EXTERNAL-sort:
//!
//!   1. **Run-generation pass.** Stream `archive.iter_bonds()`,
//!      derive each `(subject, predicate, object) + BondPayload`,
//!      buffer into a bounded run of `sort_run_entries` records,
//!      sort the run by `(subject, predicate, object)`, and spill it
//!      to a temporary run file under
//!      `<v5_dir>/bond_store/.migrate_runs/`. Peak resident is one
//!      run buffer (default ~2M records).
//!   2. **k-way merge pass.** Open every run file, min-heap-merge by
//!      `(subject, predicate, object)`, and stream the globally
//!      sorted record stream straight into [`BondSstWriter`]
//!      (rotating at subject boundaries per `target_entries_per_sst`).
//!      Peak resident is the heap (one record per run) + one writer
//!      block buffer.
//!
//! The run files are deleted on success.
//!
//! ## Byte-equivalent payload mapping
//!
//! Per the brief, the v5 bond rows MUST be byte-equivalent to the
//! already-verified v4 mapping. We REUSE the exact `migration_v3_v4`
//! helpers rather than re-deriving:
//!
//! - `predicate = migration_v3_v4::predicate_byte(bond.bond_type)`
//!   (== `memtable::predicate_discriminant`).
//! - `weight = sst::quantize_weight(migration_v3_v4::bond_latest_weight(&bond))`
//!   -- the v4 SST writer quantises the latest-keyframe weight with
//!   the same `quantize_weight`, so the stored byte is identical.
//! - `confidence = sst::confidence_to_u8(Confidence::Verified)` (== 1).
//!   The v3->v4 migrator writes EVERY migrated bond at
//!   `Confidence::Verified` (`migration_v3_v4.rs` `push_bond_with_nodes`
//!   call) and the SST stores `confidence_to_u8(c)`; the read path
//!   re-derives the band from `KeyframeSource`. We mirror that exactly.
//! - `keyframes = bond.keyframes` -- carried inline (v5 drops v4's
//!   per-block keyframe pool; the bytes per keyframe are identical).
//! - `seq = 0` across every migrated row -- the whole BondStore is one
//!   logical snapshot event; each triple appears exactly once (Q4),
//!   so cross-record dedup is unaffected.
//!
//! ## Q4 -- one predicate per (Seed, Seed) pair
//!
//! VERIFIED against the code+disk: both the live
//! `AssociativeNetwork.bonds` (`network.rs:319`) and the archive
//! `bond_index` (`archive.rs:524`) are `HashMap<(Seed, Seed), _>`.
//! A v3 (Seed, Seed) ordered pair therefore carries EXACTLY ONE
//! `TemporalBond` -- hence exactly one `bond_type`/predicate --
//! by construction; `add_bond` overwrites on the same pair. It is
//! structurally impossible for `iter_bonds()` to yield two entries
//! with the same `(from, to)`. Re-keying to `(subject, predicate,
//! object)` triples is therefore collision-free. We STILL assert
//! this at merge time (a duplicate emitted triple is a hard
//! [`MigrationError::DuplicateTriple`]) as belt-and-suspenders.
//!
//! ## Verification gate (GO/NO-GO)
//!
//! 1. **Bond-count parity.** Sum of `BondSSTEntry.entry_count` across
//!    `manifest.bond_v5_levels[0]` must equal `archive.bond_count()`.
//! 2. **Sampled outgoing spot reads.** For a random sample of
//!    subjects, `get_outgoing_bonds_v5(subject)` must match the v3
//!    archive's outgoing set for that Seed (object, predicate,
//!    quantised weight, keyframe history) exactly.
//!
//! Any mismatch is a hard failure -- partial success is not success.
//!
//! ## Production brain (`/var/lib/lu/brain.lbr`) usage
//!
//! **DO NOT run against the production brain in-place** -- same
//! discipline as Phase 2.9. Snapshot it first (`systemctl stop lu4d;
//! cp; systemctl start lu4d`) and run against the copy. NOTE that a
//! migrated v5 brain is NOT production-consumable yet: there is no
//! `Brain::load_v5` / `V5Only` read mode today (the loader-cutover
//! phase is separate). This tool produces the bond substrate; it
//! does not wire it into the live read path.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use memmap2::Mmap;

use crate::archive::{temporal_bond_from_archived, NetworkArchive};
use crate::bond::TemporalKeyframe;
use crate::bond_store::{
    bsst_filename, get_outgoing_bonds_v5, BondPayload, BondSstReaderCache, BondSstWriteOpts,
    BondSstWriter, DEFAULT_TARGET_ENTRIES_PER_BLOCK, MAX_TARGET_ENTRIES_PER_BLOCK,
};
use crate::encrypt::MasterKey;
use crate::manifest::{BondSSTEntry, Manifest, MANIFEST_FILE_NAME};
use crate::migration_v3_v4::{bond_latest_weight, predicate_byte};
use crate::node::Confidence;
use crate::node_intern::{NodeId, NodeIntern};
use crate::sst::{confidence_to_u8, quantize_weight};
use crate::BrainError;

// ----------------------------------------------------------------------
// Public configuration + result types.
// ----------------------------------------------------------------------

/// Default cap on entries per BondSST file. Beyond this point the
/// migration rotates to a fresh `file_uid` at the next subject
/// boundary. Bonds are far smaller than nodes (no embeddings), so
/// the per-file entry cap is higher than the node tool's 1M.
pub const DEFAULT_TARGET_ENTRIES_PER_SST: u64 = 4_000_000;

/// Default number of records buffered per external-sort run before
/// the run is sorted and spilled to disk. At ~50 B/record this is
/// ~100 MB resident for the run buffer.
pub const DEFAULT_SORT_RUN_ENTRIES: u64 = 2_000_000;

/// Default sample size for verification.
pub const DEFAULT_VERIFY_SAMPLE_SIZE: usize = 10_000;

/// Default progress-log cadence (one line per N bonds processed).
pub const DEFAULT_PROGRESS_EVERY_N: u64 = 100_000;

/// Caller-supplied configuration for a bond migration run.
#[derive(Debug, Clone)]
pub struct MigrationConfig {
    /// v3 brain.lbr to read. Mandatory. Read-only -- never written.
    pub v3_path: PathBuf,
    /// Node-migrated v5 directory (output of Phase 2.9). MUST exist
    /// and already hold `MANIFEST` + `wal/nodeintern.bin` +
    /// `node_store/`. The bond tool ADDS `bond_store/` and updates
    /// the manifest in place; it refuses to overwrite an existing
    /// `bond_store/`.
    pub v5_dir: PathBuf,
    /// Optional master key for encrypted v3 brains. Required iff the
    /// v3 brain is encrypted.
    pub key: Option<MasterKey>,
    /// Target entries per BondSST block (Phase 3.1 writer option).
    /// Defaults to [`DEFAULT_TARGET_ENTRIES_PER_BLOCK`].
    pub target_entries_per_block: usize,
    /// Cap on entries per BondSST file. When exceeded, the writer is
    /// finished at the next subject boundary and a fresh `file_uid`
    /// is allocated. Defaults to [`DEFAULT_TARGET_ENTRIES_PER_SST`].
    pub target_entries_per_sst: u64,
    /// Records buffered per external-sort run before spill. Bounds
    /// the run-generation pass's resident set. Defaults to
    /// [`DEFAULT_SORT_RUN_ENTRIES`].
    pub sort_run_entries: u64,
    /// Skip the verification pass after migration.
    pub skip_verify: bool,
    /// Random-sample size for verification. Ignored if `skip_verify`.
    pub verify_sample_size: usize,
    /// Soft RSS budget for the migration process, in megabytes.
    /// Warning-only; never aborts.
    pub max_rss_mb: Option<u64>,
    /// Progress-log cadence: log one line per N bonds processed.
    pub progress_every_n: u64,
}

impl MigrationConfig {
    /// Sensible defaults given the two mandatory inputs.
    pub fn new(v3_path: PathBuf, v5_dir: PathBuf) -> Self {
        MigrationConfig {
            v3_path,
            v5_dir,
            key: None,
            target_entries_per_block: DEFAULT_TARGET_ENTRIES_PER_BLOCK,
            target_entries_per_sst: DEFAULT_TARGET_ENTRIES_PER_SST,
            sort_run_entries: DEFAULT_SORT_RUN_ENTRIES,
            skip_verify: false,
            verify_sample_size: DEFAULT_VERIFY_SAMPLE_SIZE,
            max_rss_mb: Some(8192),
            progress_every_n: DEFAULT_PROGRESS_EVERY_N,
        }
    }
}

/// Statistics returned by [`migrate`].
#[derive(Debug, Clone)]
pub struct MigrationStats {
    /// v3 archive node count at open time (diagnostic).
    pub v3_node_count: u64,
    /// v3 archive bond count at open time.
    pub v3_bond_count: u64,
    /// NodeIntern entry count loaded from the node-migrated dir.
    pub intern_node_count: u64,
    /// Number of v3 bonds read from the archive.
    pub bonds_read: u64,
    /// Number of bond rows written to v5 BondSSTs. Equals
    /// `bonds_read` on a clean run (one row per v3 bond, Q4).
    pub bonds_written: u64,
    /// Number of external-sort run files spilled.
    pub sort_runs_spilled: u64,
    /// Number of BondSST files written at L0.
    pub bsst_files_written: u64,
    /// Total bytes across every BondSST file.
    pub bsst_total_bytes: u64,
    /// Manifest file size after the bond entries were added.
    pub manifest_bytes: u64,
    /// Peak resident set size, in KB (`/proc/self/status` VmHWM).
    pub peak_rss_kb: Option<u64>,
    /// Elapsed wall-clock for the migration phase (excludes verify).
    pub migration_elapsed: Duration,
    /// Verification report. `None` if `skip_verify == true`.
    pub verification: Option<VerificationReport>,
}

/// Result of the verification pass.
#[derive(Debug, Clone)]
pub struct VerificationReport {
    /// Number of subjects sampled for spot reads.
    pub sample_size: usize,
    /// Number of mismatches found. Zero is the only success state.
    pub mismatch_count: u64,
    /// Up to 10 mismatch details for diagnostic dumping.
    pub mismatch_details: Vec<MismatchDetail>,
    /// v3 archive bond count.
    pub v3_bond_count: u64,
    /// Sum of entry_count across `manifest.bond_v5_levels[0]`.
    pub v5_bond_count: u64,
    /// True iff `v3_bond_count == v5_bond_count`.
    pub count_matches: bool,
    /// Elapsed wall-clock for the verification pass.
    pub elapsed: Duration,
}

impl VerificationReport {
    /// True iff zero mismatches AND total count match.
    pub fn is_ok(&self) -> bool {
        self.mismatch_count == 0 && self.count_matches
    }
}

/// One mismatch caught by the verification sample.
#[derive(Debug, Clone)]
pub struct MismatchDetail {
    pub subject_id: u64,
    pub seed_hex: String,
    pub reason: String,
}

/// Errors surfaced by the bond migration tool.
#[derive(Debug)]
pub enum MigrationError {
    /// `bond_store/` already exists under the v5 dir. The brief
    /// mandates no overwrite -- the operator must clear it explicitly.
    BondStoreExists(PathBuf),
    /// v3 brain path does not exist.
    V3PathMissing(PathBuf),
    /// The v5 dir is not a completed node-migrated directory (missing
    /// MANIFEST, wal/nodeintern.bin, or node_store/). Carries a short
    /// description of what was missing.
    V5DirNotNodeMigrated(String),
    /// A v3 bond endpoint Seed was not present in the loaded
    /// NodeIntern -- i.e. the node migration did not intern it. This
    /// should be impossible (Phase 2.9 interns every bond endpoint
    /// via `intern_from_v3_archive`); surfaced as a hard error rather
    /// than silently dropping the bond.
    SeedNotInterned { seed_hex: String, end: &'static str },
    /// Two emitted bond rows shared the same `(subject, predicate,
    /// object)` triple. Structurally impossible given the v3
    /// `HashMap<(Seed,Seed),_>` keying (Q4); a hit here means the
    /// source archive violated that invariant.
    DuplicateTriple { subject: u64, predicate: u8, object: u64 },
    /// Any wrapped `BrainError` -- archive open, manifest save, etc.
    Brain(BrainError),
    /// BondSstWriter / reader / manifest errors during write.
    BondStore(String),
    /// Verification failed (count or sample mismatch).
    VerificationFailed(VerificationReport),
    /// IO error during dir creation, run spill, or rename.
    Io(std::io::Error),
}

impl std::fmt::Display for MigrationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MigrationError::BondStoreExists(p) => write!(
                f,
                "bond_store directory {} already exists; refusing to overwrite",
                p.display()
            ),
            MigrationError::V3PathMissing(p) => {
                write!(f, "v3 brain path {} does not exist", p.display())
            }
            MigrationError::V5DirNotNodeMigrated(s) => {
                write!(f, "v5 dir is not a completed node-migrated directory: {}", s)
            }
            MigrationError::SeedNotInterned { seed_hex, end } => write!(
                f,
                "v3 bond {} endpoint Seed {} not found in NodeIntern (node migration incomplete?)",
                end, seed_hex
            ),
            MigrationError::DuplicateTriple { subject, predicate, object } => write!(
                f,
                "duplicate bond triple emitted: (subject={}, predicate={}, object={})",
                subject, predicate, object
            ),
            MigrationError::Brain(e) => write!(f, "brain error: {}", e),
            MigrationError::BondStore(s) => write!(f, "bond store error: {}", s),
            MigrationError::VerificationFailed(r) => write!(
                f,
                "verification FAILED: {} mismatches, count_matches={} (v3={} v5={})",
                r.mismatch_count, r.count_matches, r.v3_bond_count, r.v5_bond_count
            ),
            MigrationError::Io(e) => write!(f, "io error: {}", e),
        }
    }
}

impl std::error::Error for MigrationError {}

impl From<BrainError> for MigrationError {
    fn from(e: BrainError) -> Self {
        MigrationError::Brain(e)
    }
}

impl From<std::io::Error> for MigrationError {
    fn from(e: std::io::Error) -> Self {
        MigrationError::Io(e)
    }
}

// ----------------------------------------------------------------------
// Public entry point.
// ----------------------------------------------------------------------

/// Run the full bond-migration pipeline per the module-level doc.
pub fn migrate(cfg: &MigrationConfig) -> Result<MigrationStats, MigrationError> {
    let migration_start = Instant::now();

    // ----- 1. Validate inputs ------------------------------------
    if !cfg.v3_path.exists() {
        return Err(MigrationError::V3PathMissing(cfg.v3_path.clone()));
    }
    let manifest_path = cfg.v5_dir.join(MANIFEST_FILE_NAME);
    let sidecar_path = cfg.v5_dir.join("wal").join("nodeintern.bin");
    let node_store_dir = cfg.v5_dir.join("node_store");
    if !manifest_path.exists() {
        return Err(MigrationError::V5DirNotNodeMigrated(format!(
            "missing manifest at {}",
            manifest_path.display()
        )));
    }
    if !sidecar_path.exists() {
        return Err(MigrationError::V5DirNotNodeMigrated(format!(
            "missing node-intern sidecar at {}",
            sidecar_path.display()
        )));
    }
    if !node_store_dir.is_dir() {
        return Err(MigrationError::V5DirNotNodeMigrated(format!(
            "missing node_store/ at {}",
            node_store_dir.display()
        )));
    }
    let bond_store_dir = cfg.v5_dir.join("bond_store");
    if bond_store_dir.exists() {
        return Err(MigrationError::BondStoreExists(bond_store_dir));
    }

    eprintln!(
        "[migrate-v3-v5-bond] opening v3 brain {} ({})",
        cfg.v3_path.display(),
        if cfg.key.is_some() { "encrypted" } else { "plain" },
    );

    // ----- 2. Open v3 archive read-only --------------------------
    let archive = open_v3_archive(&cfg.v3_path, cfg.key.as_ref())?;
    let v3_node_count = archive.node_count() as u64;
    let v3_bond_count = archive.bond_count() as u64;
    eprintln!(
        "[migrate-v3-v5-bond] v3 opened: {} nodes, {} bonds",
        v3_node_count, v3_bond_count,
    );

    // ----- 3. Load the node-migrated NodeIntern + Manifest -------
    eprintln!(
        "[migrate-v3-v5-bond] loading NodeIntern sidecar {}",
        sidecar_path.display(),
    );
    let intern = NodeIntern::load_from_with_key(&sidecar_path, cfg.key.clone())
        .map_err(|e| MigrationError::BondStore(format!("NodeIntern::load_from: {}", e)))?;
    let intern_node_count = intern.len() as u64;
    let mut manifest = Manifest::load(&manifest_path)?.ok_or_else(|| {
        MigrationError::V5DirNotNodeMigrated(format!(
            "manifest at {} failed to load",
            manifest_path.display()
        ))
    })?;
    if !manifest.bond_v5_levels.is_empty() {
        return Err(MigrationError::BondStore(format!(
            "manifest already carries {} bond_v5 level(s); refusing to append bonds to a \
             non-empty bond store",
            manifest.bond_v5_levels.len()
        )));
    }
    eprintln!(
        "[migrate-v3-v5-bond] NodeIntern: {} nodes; manifest loaded (schema v{})",
        intern_node_count, manifest.schema_version,
    );

    // ----- 4. Build bond_store/ and the scratch run dir ----------
    fs::create_dir_all(&bond_store_dir)?;
    let run_dir = bond_store_dir.join(".migrate_runs");
    fs::create_dir_all(&run_dir)?;

    // ----- 5. Run-generation pass (external sort, phase 1) -------
    eprintln!(
        "[migrate-v3-v5-bond] run-generation pass (sort_run_entries={})",
        cfg.sort_run_entries,
    );
    let gen_start = Instant::now();
    let RunGenSummary {
        bonds_read,
        run_paths,
    } = generate_sorted_runs(&archive, &intern, &run_dir, cfg)?;
    eprintln!(
        "[migrate-v3-v5-bond] run-generation done in {:?}: {} bonds, {} runs",
        gen_start.elapsed(),
        bonds_read,
        run_paths.len(),
    );

    // ----- 6. k-way merge pass into BondSSTs (phase 2) -----------
    eprintln!(
        "[migrate-v3-v5-bond] merge pass: {} runs -> BondSSTs (target entries/sst={})",
        run_paths.len(),
        cfg.target_entries_per_sst,
    );
    let merge_start = Instant::now();
    let MergeSummary {
        bonds_written,
        bsst_files_written,
        bsst_total_bytes,
    } = merge_runs_to_ssts(&run_paths, &bond_store_dir, &mut manifest, cfg)?;
    eprintln!(
        "[migrate-v3-v5-bond] merge done in {:?}: {} bonds, {} BondSSTs, {} bytes",
        merge_start.elapsed(),
        bonds_written,
        bsst_files_written,
        bsst_total_bytes,
    );

    // ----- 7. Save manifest atomically ---------------------------
    manifest.save_atomic(&manifest_path)?;
    let manifest_bytes = fs::metadata(&manifest_path)?.len();

    // ----- 8. Clean up run files ---------------------------------
    for p in &run_paths {
        let _ = fs::remove_file(p);
    }
    let _ = fs::remove_dir(&run_dir);

    let peak_rss_kb = read_self_rss_peak_kb();
    let migration_elapsed = migration_start.elapsed();

    // ----- 9. Verification pass (unless --skip-verify) -----------
    let verification = if cfg.skip_verify {
        eprintln!("[migrate-v3-v5-bond] --skip-verify set; verification SKIPPED");
        None
    } else {
        eprintln!(
            "[migrate-v3-v5-bond] running verification (sample_size={})",
            cfg.verify_sample_size,
        );
        let report = verify_migration(
            &cfg.v5_dir,
            &archive,
            &intern,
            cfg.verify_sample_size,
            cfg.key.as_ref(),
        )?;
        if !report.is_ok() {
            return Err(MigrationError::VerificationFailed(report));
        }
        Some(report)
    };

    Ok(MigrationStats {
        v3_node_count,
        v3_bond_count,
        intern_node_count,
        bonds_read,
        bonds_written,
        sort_runs_spilled: run_paths.len() as u64,
        bsst_files_written,
        bsst_total_bytes,
        manifest_bytes,
        peak_rss_kb,
        migration_elapsed,
        verification,
    })
}

// ----------------------------------------------------------------------
// External sort -- phase 1: run generation.
// ----------------------------------------------------------------------

/// One bond row in the external sort: the triple key plus the
/// bincode-serialised [`BondPayload`] bytes. Stored compactly on
/// disk; re-hydrated into a `BondPayload` only at merge time.
struct SortRecord {
    subject: u64,
    predicate: u8,
    object: u64,
    payload: Vec<u8>,
}

impl SortRecord {
    /// Total triple ordering used by both the in-run sort and the
    /// k-way merge heap.
    fn key(&self) -> (u64, u8, u64) {
        (self.subject, self.predicate, self.object)
    }
}

struct RunGenSummary {
    bonds_read: u64,
    run_paths: Vec<PathBuf>,
}

/// Stream `archive.iter_bonds()`, derive each bond's triple +
/// payload, and spill bounded sorted runs to `run_dir`.
fn generate_sorted_runs(
    archive: &NetworkArchive,
    intern: &NodeIntern,
    run_dir: &Path,
    cfg: &MigrationConfig,
) -> Result<RunGenSummary, MigrationError> {
    let run_capacity = cfg.sort_run_entries.max(1) as usize;
    let progress_every = cfg.progress_every_n.max(1);
    let progress_start = Instant::now();

    let mut buffer: Vec<SortRecord> = Vec::with_capacity(run_capacity.min(1_000_000));
    let mut run_paths: Vec<PathBuf> = Vec::new();
    let mut bonds_read: u64 = 0;

    for ((from, to), ba) in archive.iter_bonds() {
        // Map both endpoints through the node-migrated NodeIntern so
        // bonds reference the SAME NodeIds Phase 2.9 assigned.
        let subject = intern.get(&from).ok_or_else(|| MigrationError::SeedNotInterned {
            seed_hex: hex_short(from.as_bytes()),
            end: "subject",
        })?;
        let object = intern.get(&to).ok_or_else(|| MigrationError::SeedNotInterned {
            seed_hex: hex_short(to.as_bytes()),
            end: "object",
        })?;

        // Materialise the bond (one heap copy at a time).
        let bond = temporal_bond_from_archived(ba.as_archived());

        // Byte-equivalent v4 mapping -- REUSE the migration_v3_v4
        // helpers; do NOT re-derive.
        let predicate = predicate_byte(bond.bond_type);
        let weight = quantize_weight(bond_latest_weight(&bond));
        let confidence = confidence_to_u8(Confidence::Verified);
        let payload = BondPayload {
            weight,
            confidence,
            keyframes: bond.keyframes,
            seq: 0,
        };
        let payload_bytes = bincode::serialize(&payload).map_err(|e| {
            MigrationError::BondStore(format!("serialize BondPayload: {}", e))
        })?;

        buffer.push(SortRecord {
            subject: subject.0,
            predicate,
            object: object.0,
            payload: payload_bytes,
        });
        bonds_read += 1;

        if buffer.len() >= run_capacity {
            run_paths.push(spill_run(&mut buffer, run_dir, run_paths.len())?);
        }

        if bonds_read % progress_every == 0 {
            log_progress("run-gen", bonds_read, progress_start.elapsed(), cfg);
        }
    }

    // Final partial run.
    if !buffer.is_empty() {
        run_paths.push(spill_run(&mut buffer, run_dir, run_paths.len())?);
    }

    Ok(RunGenSummary {
        bonds_read,
        run_paths,
    })
}

/// Sort `buffer` in place by `(subject, predicate, object)` and write
/// it to a fresh run file, clearing the buffer afterwards.
fn spill_run(
    buffer: &mut Vec<SortRecord>,
    run_dir: &Path,
    run_idx: usize,
) -> Result<PathBuf, MigrationError> {
    buffer.sort_unstable_by(|a, b| a.key().cmp(&b.key()));
    let path = run_dir.join(format!("run_{:06}.bin", run_idx));
    let file = File::create(&path)?;
    let mut w = BufWriter::with_capacity(1 << 20, file);
    for rec in buffer.iter() {
        write_record(&mut w, rec)?;
    }
    w.flush()?;
    buffer.clear();
    Ok(path)
}

/// On-disk record framing: subject(u64 LE) predicate(u8) object(u64
/// LE) payload_len(u32 LE) payload[payload_len].
fn write_record<W: Write>(w: &mut W, rec: &SortRecord) -> Result<(), MigrationError> {
    w.write_all(&rec.subject.to_le_bytes())?;
    w.write_all(&[rec.predicate])?;
    w.write_all(&rec.object.to_le_bytes())?;
    let len = u32::try_from(rec.payload.len()).map_err(|_| {
        MigrationError::BondStore(format!(
            "run record payload {} exceeds u32::MAX",
            rec.payload.len()
        ))
    })?;
    w.write_all(&len.to_le_bytes())?;
    w.write_all(&rec.payload)?;
    Ok(())
}

/// Read a single framed record; `Ok(None)` on clean EOF.
fn read_record<R: Read>(r: &mut R) -> Result<Option<SortRecord>, MigrationError> {
    let mut subj = [0u8; 8];
    match r.read_exact(&mut subj) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e.into()),
    }
    let mut pred = [0u8; 1];
    r.read_exact(&mut pred)?;
    let mut obj = [0u8; 8];
    r.read_exact(&mut obj)?;
    let mut len_b = [0u8; 4];
    r.read_exact(&mut len_b)?;
    let len = u32::from_le_bytes(len_b) as usize;
    let mut payload = vec![0u8; len];
    r.read_exact(&mut payload)?;
    Ok(Some(SortRecord {
        subject: u64::from_le_bytes(subj),
        predicate: pred[0],
        object: u64::from_le_bytes(obj),
        payload,
    }))
}

// ----------------------------------------------------------------------
// External sort -- phase 2: k-way merge into BondSSTs.
// ----------------------------------------------------------------------

/// Heap entry for the k-way merge: orders by the triple key, then by
/// run index for stable tie-breaking. Wrapped so the `BinaryHeap`
/// (a max-heap) behaves as a min-heap via reversed `Ord`.
struct HeapItem {
    key: (u64, u8, u64),
    run_idx: usize,
}

impl PartialEq for HeapItem {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key && self.run_idx == other.run_idx
    }
}
impl Eq for HeapItem {}
impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse so the BinaryHeap yields the SMALLEST key first.
        other
            .key
            .cmp(&self.key)
            .then_with(|| other.run_idx.cmp(&self.run_idx))
    }
}
impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

struct MergeSummary {
    bonds_written: u64,
    bsst_files_written: u64,
    bsst_total_bytes: u64,
}

/// Active BondSST writer state threaded across the merge loop so we
/// can rotate at subject boundaries.
struct ActiveBsst {
    writer: BondSstWriter,
    path: PathBuf,
    entries_in_sst: u64,
    file_uid: u64,
    epoch: u64,
    first_subject: u64,
    last_subject: u64,
}

/// k-way merge every run file into globally-sorted order and stream
/// into one or more BondSSTs at L0.
fn merge_runs_to_ssts(
    run_paths: &[PathBuf],
    bond_store_dir: &Path,
    manifest: &mut Manifest,
    cfg: &MigrationConfig,
) -> Result<MergeSummary, MigrationError> {
    let mut bonds_written: u64 = 0;
    let mut bsst_files_written: u64 = 0;
    let mut bsst_total_bytes: u64 = 0;

    if run_paths.is_empty() {
        return Ok(MergeSummary {
            bonds_written,
            bsst_files_written,
            bsst_total_bytes,
        });
    }

    // Open every run as a buffered reader and prime the heap with the
    // first record from each.
    let mut readers: Vec<BufReader<File>> = Vec::with_capacity(run_paths.len());
    let mut heads: Vec<Option<SortRecord>> = Vec::with_capacity(run_paths.len());
    let mut heap: BinaryHeap<HeapItem> = BinaryHeap::with_capacity(run_paths.len());

    for (idx, p) in run_paths.iter().enumerate() {
        let mut r = BufReader::with_capacity(1 << 20, File::open(p)?);
        let first = read_record(&mut r)?;
        if let Some(ref rec) = first {
            heap.push(HeapItem {
                key: rec.key(),
                run_idx: idx,
            });
        }
        readers.push(r);
        heads.push(first);
    }

    let progress_every = cfg.progress_every_n.max(1);
    let progress_start = Instant::now();
    let target_per_sst = cfg.target_entries_per_sst.max(1);

    let mut active: Option<ActiveBsst> = None;
    let mut last_key: Option<(u64, u8, u64)> = None;

    while let Some(top) = heap.pop() {
        let run_idx = top.run_idx;
        // Take the record at the head of this run; refill from disk.
        let rec = heads[run_idx]
            .take()
            .expect("heap referenced an empty run head");
        if let Some(next) = read_record(&mut readers[run_idx])? {
            heap.push(HeapItem {
                key: next.key(),
                run_idx,
            });
            heads[run_idx] = Some(next);
        }

        let key = rec.key();

        // Q4 belt-and-suspenders: a duplicate triple is structurally
        // impossible (v3 keys bonds by (Seed,Seed)); fail loudly.
        if let Some(prev) = last_key {
            if prev == key {
                return Err(MigrationError::DuplicateTriple {
                    subject: key.0,
                    predicate: key.1,
                    object: key.2,
                });
            }
        }

        // Rotate at a subject boundary when the active file is full.
        let subject_changed = active
            .as_ref()
            .map(|a| a.last_subject != key.0)
            .unwrap_or(true);
        if subject_changed {
            if let Some(act) = active.take() {
                if act.entries_in_sst >= target_per_sst {
                    let stats = finish_and_record(act, manifest)?;
                    bsst_files_written += 1;
                    bsst_total_bytes += stats;
                } else {
                    active = Some(act);
                }
            }
        }

        // Open a writer if we don't have one.
        if active.is_none() {
            active = Some(open_bsst(bond_store_dir, manifest, cfg, key.0)?);
        }

        let payload: BondPayload = bincode::deserialize(&rec.payload).map_err(|e| {
            MigrationError::BondStore(format!("deserialize BondPayload: {}", e))
        })?;
        let act = active.as_mut().expect("active set above");
        act.writer
            .append(NodeId(key.0), key.1, NodeId(key.2), &payload)
            .map_err(|e| {
                MigrationError::BondStore(format!(
                    "BondSstWriter::append ({}, {}, {}): {}",
                    key.0, key.1, key.2, e
                ))
            })?;
        act.entries_in_sst += 1;
        act.last_subject = key.0;
        bonds_written += 1;
        last_key = Some(key);

        if bonds_written % progress_every == 0 {
            log_progress("merge", bonds_written, progress_start.elapsed(), cfg);
        }
    }

    // Finish the trailing writer.
    if let Some(act) = active.take() {
        let stats = finish_and_record(act, manifest)?;
        bsst_files_written += 1;
        bsst_total_bytes += stats;
    }

    Ok(MergeSummary {
        bonds_written,
        bsst_files_written,
        bsst_total_bytes,
    })
}

/// Open a fresh BondSST writer + allocate its uid/epoch.
fn open_bsst(
    bond_store_dir: &Path,
    manifest: &mut Manifest,
    cfg: &MigrationConfig,
    first_subject: u64,
) -> Result<ActiveBsst, MigrationError> {
    let file_uid = manifest.allocate_bond_v5_sst_uid();
    let epoch = manifest.allocate_epoch();
    let path = bond_store_dir.join(bsst_filename(file_uid));
    let opts = BondSstWriteOpts {
        target_entries_per_block: cfg
            .target_entries_per_block
            .max(1)
            .min(MAX_TARGET_ENTRIES_PER_BLOCK),
        compression: Default::default(),
        file_uid,
    };
    let writer = BondSstWriter::create_with_key(&path, opts, cfg.key.clone())
        .map_err(|e| MigrationError::BondStore(format!("BondSstWriter::create: {}", e)))?;
    Ok(ActiveBsst {
        writer,
        path,
        entries_in_sst: 0,
        file_uid,
        epoch,
        first_subject,
        last_subject: first_subject,
    })
}

/// Finish a BondSST and record its `BondSSTEntry` at L0. Returns the
/// file size in bytes.
fn finish_and_record(
    act: ActiveBsst,
    manifest: &mut Manifest,
) -> Result<u64, MigrationError> {
    let ActiveBsst {
        writer,
        path,
        entries_in_sst,
        file_uid,
        epoch,
        first_subject,
        last_subject,
    } = act;
    let stats = writer
        .finish()
        .map_err(|e| MigrationError::BondStore(format!("BondSstWriter::finish: {}", e)))?;
    debug_assert_eq!(stats.entries_written, entries_in_sst);
    let _ = path;

    let entry = BondSSTEntry {
        file_uid,
        first_subject,
        last_subject,
        entry_count: stats.entries_written,
        file_size: stats.file_size_bytes,
        epoch,
        // Q1: forward bonds only -- reverse index is emitted later by
        // the BondCompactionWorker, not at migration time.
        reverse_file_uid: None,
    };
    manifest
        .add_bond_v5_sst(0, entry)
        .map_err(|e| MigrationError::BondStore(format!("manifest.add_bond_v5_sst: {}", e)))?;
    Ok(stats.file_size_bytes)
}

// ----------------------------------------------------------------------
// Verification pass.
// ----------------------------------------------------------------------

/// Verify a migrated v5 bond store against its v3 source.
///
/// 1. Bond-count parity: sum of `entry_count` over `bond_v5_levels[0]`
///    must equal `archive.bond_count()`.
/// 2. Sampled outgoing spot reads: for a random sample of subjects,
///    `get_outgoing_bonds_v5(subject)` must match the v3 archive's
///    outgoing set for that Seed exactly.
pub fn verify_migration(
    v5_dir: &Path,
    archive: &NetworkArchive,
    intern: &NodeIntern,
    sample_size: usize,
    key: Option<&MasterKey>,
) -> Result<VerificationReport, MigrationError> {
    let verify_start = Instant::now();

    let manifest_path = v5_dir.join(MANIFEST_FILE_NAME);
    let manifest = Manifest::load(&manifest_path)?.ok_or_else(|| {
        MigrationError::BondStore(format!(
            "manifest at {} not found during verification",
            manifest_path.display()
        ))
    })?;
    let bond_store_dir = v5_dir.join("bond_store");
    let reader_cache = BondSstReaderCache::new(bond_store_dir).with_key(key.cloned());

    // (1) Count match.
    let v5_bond_count: u64 = manifest
        .bond_v5_levels
        .first()
        .map(|l| l.ssts.iter().map(|e| e.entry_count).sum::<u64>())
        .unwrap_or(0);
    let v3_bond_count = archive.bond_count() as u64;
    let count_matches = v3_bond_count == v5_bond_count;

    let total_nodes = intern.len();
    let actual_sample = sample_size.min(total_nodes);

    let mut mismatch_count: u64 = 0;
    let mut mismatch_details: Vec<MismatchDetail> = Vec::with_capacity(16);

    if total_nodes == 0 || actual_sample == 0 {
        return Ok(VerificationReport {
            sample_size: 0,
            mismatch_count,
            mismatch_details,
            v3_bond_count,
            v5_bond_count,
            count_matches,
            elapsed: verify_start.elapsed(),
        });
    }

    // Draw a deterministic random subject sample.
    let mut rng = SplitMix64::new(0xB0D5_F00D_1234_5678);
    let mut sampled_ids: HashSet<u64> = HashSet::with_capacity(actual_sample);
    let mut sampled_seeds: HashMap<[u8; 32], NodeId> = HashMap::with_capacity(actual_sample);
    let mut attempts = 0usize;
    let max_attempts = actual_sample.saturating_mul(8).max(actual_sample + 64);
    while sampled_ids.len() < actual_sample && attempts < max_attempts {
        attempts += 1;
        let raw = rng.next_u64() % (total_nodes as u64);
        if !sampled_ids.insert(raw) {
            continue;
        }
        if let Some(seed) = intern.reverse_lookup(NodeId(raw)) {
            sampled_seeds.insert(*seed.as_bytes(), NodeId(raw));
        }
    }

    // Build the EXPECTED outgoing set for the sampled subjects in a
    // single pass over the v3 archive (memory bounded by the sample,
    // not the whole bond set).
    let mut expected: HashMap<u64, Vec<ExpectedBond>> = HashMap::with_capacity(actual_sample);
    for ((from, to), ba) in archive.iter_bonds() {
        if let Some(&subj_id) = sampled_seeds.get(from.as_bytes()) {
            let object = match intern.get(&to) {
                Some(o) => o,
                None => {
                    return Err(MigrationError::SeedNotInterned {
                        seed_hex: hex_short(to.as_bytes()),
                        end: "object",
                    })
                }
            };
            let bond = temporal_bond_from_archived(ba.as_archived());
            expected.entry(subj_id.0).or_default().push(ExpectedBond {
                predicate: predicate_byte(bond.bond_type),
                object: object.0,
                weight: quantize_weight(bond_latest_weight(&bond)),
                keyframes: bond.keyframes,
            });
        }
    }

    // Compare each sampled subject's v5 outgoing set to the expected.
    for (_seed, &id) in &sampled_seeds {
        match check_outgoing(&manifest, &reader_cache, &expected, id) {
            Ok(()) => {}
            Err(reason) => {
                mismatch_count += 1;
                if mismatch_details.len() < 10 {
                    let seed_hex = intern
                        .reverse_lookup(id)
                        .map(|s| hex_short(s.as_bytes()))
                        .unwrap_or_else(|| "<unknown>".to_string());
                    mismatch_details.push(MismatchDetail {
                        subject_id: id.0,
                        seed_hex,
                        reason,
                    });
                }
            }
        }
    }

    Ok(VerificationReport {
        sample_size: sampled_seeds.len(),
        mismatch_count,
        mismatch_details,
        v3_bond_count,
        v5_bond_count,
        count_matches,
        elapsed: verify_start.elapsed(),
    })
}

/// The expected projection of one v3 outgoing bond.
struct ExpectedBond {
    predicate: u8,
    object: u64,
    weight: u8,
    keyframes: Vec<TemporalKeyframe>,
}

/// Compare one subject's v5 outgoing set against the expected v3 set.
fn check_outgoing(
    manifest: &Manifest,
    reader_cache: &BondSstReaderCache,
    expected: &HashMap<u64, Vec<ExpectedBond>>,
    subject: NodeId,
) -> Result<(), String> {
    // Collect the v5 outgoing rows.
    let mut got: Vec<(u8, u64, u8, Vec<TemporalKeyframe>)> = Vec::new();
    for item in get_outgoing_bonds_v5(manifest, reader_cache, subject) {
        let (subj, predicate, object, payload) =
            item.map_err(|e| format!("get_outgoing_bonds_v5 errored: {}", e))?;
        if subj != subject {
            return Err(format!(
                "read returned foreign subject {} for query {}",
                subj.0, subject.0
            ));
        }
        got.push((predicate, object.0, payload.weight, payload.keyframes));
    }

    let empty = Vec::new();
    let want = expected.get(&subject.0).unwrap_or(&empty);

    if got.len() != want.len() {
        return Err(format!(
            "outgoing-count mismatch: v5 has {}, v3 has {}",
            got.len(),
            want.len()
        ));
    }

    // Sort both by (predicate, object) and compare row-by-row.
    got.sort_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));
    let mut want_sorted: Vec<&ExpectedBond> = want.iter().collect();
    want_sorted.sort_by(|a, b| (a.predicate, a.object).cmp(&(b.predicate, b.object)));

    for (g, w) in got.iter().zip(want_sorted.iter()) {
        if g.0 != w.predicate || g.1 != w.object {
            return Err(format!(
                "triple mismatch: v5=(pred {}, obj {}), v3=(pred {}, obj {})",
                g.0, g.1, w.predicate, w.object
            ));
        }
        if g.2 != w.weight {
            return Err(format!(
                "weight mismatch on (pred {}, obj {}): v5={}, v3={}",
                g.0, g.1, g.2, w.weight
            ));
        }
        if !keyframes_eq(&g.3, &w.keyframes) {
            return Err(format!(
                "keyframe mismatch on (pred {}, obj {}): v5 len={}, v3 len={}",
                g.0,
                g.1,
                g.3.len(),
                w.keyframes.len()
            ));
        }
    }
    Ok(())
}

/// Structural keyframe equality (timestamp, weight, source).
fn keyframes_eq(a: &[TemporalKeyframe], b: &[TemporalKeyframe]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b.iter()).all(|(x, y)| {
        x.timestamp == y.timestamp
            && x.weight.to_bits() == y.weight.to_bits()
            && x.source == y.source
    })
}

// ----------------------------------------------------------------------
// Internal helpers.
// ----------------------------------------------------------------------

/// Open a v3 brain.lbr read-only via mmap.
fn open_v3_archive(
    path: &Path,
    key: Option<&MasterKey>,
) -> Result<NetworkArchive, MigrationError> {
    let f = fs::File::open(path).map_err(|e| {
        MigrationError::Brain(BrainError::Format(format!(
            "open v3 brain {}: {}",
            path.display(),
            e
        )))
    })?;
    // SAFETY: read-only mmap of a file we just opened; the operator
    // snapshotted the brain so it is quiescent during the read window.
    let mmap = unsafe { Mmap::map(&f) }.map_err(|e| {
        MigrationError::Brain(BrainError::Format(format!(
            "mmap v3 brain {}: {}",
            path.display(),
            e
        )))
    })?;
    let archive = NetworkArchive::from_mmap_with_key(Arc::new(mmap), key.cloned())?;
    Ok(archive)
}

/// Log a progress line plus a soft RSS check.
fn log_progress(phase: &str, processed: u64, elapsed: Duration, cfg: &MigrationConfig) {
    let rss = read_self_rss_kb();
    eprintln!(
        "[migrate-v3-v5-bond] {} progress: {} bonds elapsed={:?} rss={}",
        phase,
        processed,
        elapsed,
        rss.map(|kb| format!("{:.1}MB", kb as f64 / 1024.0))
            .unwrap_or_else(|| "<unknown>".to_string()),
    );
    if let (Some(max_mb), Some(current_kb)) = (cfg.max_rss_mb, rss) {
        let current_mb = current_kb / 1024;
        if current_mb > max_mb {
            eprintln!(
                "[migrate-v3-v5-bond] WARNING: RSS {} MB exceeds --max-rss-mb {}; continuing",
                current_mb, max_mb,
            );
        }
    }
}

/// Read VmHWM (peak RSS) from `/proc/self/status`. None on non-Linux.
fn read_self_rss_peak_kb() -> Option<u64> {
    crate::node_store::sst_writer::read_self_rss_peak_kb()
}

/// Read VmRSS (current RSS) from `/proc/self/status`. None on non-Linux.
fn read_self_rss_kb() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            let mut it = rest.split_ascii_whitespace();
            if let Some(num) = it.next() {
                if let Ok(v) = num.parse::<u64>() {
                    return Some(v);
                }
            }
        }
    }
    None
}

/// Format the first 8 bytes of a Seed as hex for diagnostic output.
fn hex_short(bytes: &[u8; 32]) -> String {
    let mut s = String::with_capacity(16);
    for b in &bytes[..8] {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

/// Tiny deterministic PRNG for the verification sample. Mirrors the
/// Phase 2.9 node tool's SplitMix64 -- avoids a `rand` dep in the
/// production migration path.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        SplitMix64 { state: seed }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
}

// ----------------------------------------------------------------------
// Tests.
// ----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bond::{BondType, KeyframeSource, TemporalBond, TemporalKeyframe};
    use crate::brain::{Brain, BrainSaveFormat};
    use crate::migration_v3_v5_node as node_mig;
    use crate::seed::Seed;
    use crate::node::{Confidence, KnowledgeContent, KnowledgeNode};
        use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use std::collections::HashSet as StdHashSet;
    use tempfile::tempdir;

    /// Build a v3 brain with `n_nodes` nodes and ~`n_nodes` random
    /// bonds. Returns the seeds in insertion order.
    fn build_v3_brain(path: &Path, n_nodes: usize, seed: u64) -> Vec<Seed> {
        let mut brain = Brain::new();
        let mut rng = StdRng::seed_from_u64(seed);
        let mut seeds: Vec<Seed> = Vec::with_capacity(n_nodes);
        for i in 0..n_nodes {
            let mut sb = [0u8; 32];
            rng.fill(&mut sb);
            let s = Seed::from_bytes(sb);
            let label = format!("n{:06}", i);
            let mut node = KnowledgeNode::new(
                label.clone(),
                KnowledgeContent::Concept(label),
                Confidence::Verified,
                0,
            );
            node.seed = s;
            brain.network.add_node(node);
            seeds.push(s);
        }
        let connected_n = (n_nodes * 8) / 10;
        let mut dedup: StdHashSet<(Seed, Seed)> = StdHashSet::new();
        let bts = [
            BondType::IsA,
            BondType::HasProperty,
            BondType::RelatedTo,
            BondType::Causes,
        ];
        let target_bonds = n_nodes;
        let mut emitted = 0usize;
        let mut attempts = 0usize;
        while emitted < target_bonds && attempts < target_bonds * 8 {
            attempts += 1;
            let i = rng.gen_range(0..connected_n);
            let j = rng.gen_range(0..connected_n);
            if i == j {
                continue;
            }
            let from = seeds[i];
            let to = seeds[j];
            // v3 keys bonds by (from,to) -- dedup the ORDERED PAIR so
            // we never attempt two predicates on one pair (Q4).
            if !dedup.insert((from, to)) {
                continue;
            }
            let bt = bts[rng.gen_range(0..bts.len())];
            let bond = TemporalBond {
                from,
                to,
                bond_type: bt,
                keyframes: vec![TemporalKeyframe {
                    timestamp: emitted as i64,
                    weight: 0.5,
                    source: KeyframeSource::Baseline,
                }],
                half_life_ms: 86_400_000.0,
            };
            brain.network.add_bond(bond).unwrap();
            emitted += 1;
        }
        brain
            .save_to_format(path, BrainSaveFormat::V3BinaryCompressed)
            .unwrap();
        seeds
    }

    /// Run the Phase 2.9 node migration so the bond tool has a
    /// node-migrated dir to attach to.
    fn run_node_migration(v3_path: &Path, v5_dir: &Path) {
        let cfg = node_mig::MigrationConfig::new(v3_path.to_path_buf(), v5_dir.to_path_buf());
        let stats = node_mig::migrate(&cfg).expect("node migration succeeded");
        let v = stats.verification.expect("node verification ran");
        assert!(v.is_ok(), "node migration verify not OK: {:?}", v);
    }

    /// (i) Full round-trip: node-migrate, bond-migrate, verify GO.
    #[test]
    fn phase310_synthetic_round_trip() {
        let tmp = tempdir().unwrap();
        let v3_path = tmp.path().join("brain.lbr");
        let v5_dir = tmp.path().join("brain-v5");
        let _seeds = build_v3_brain(&v3_path, 1_000, 0xDEAD_BEEF);
        run_node_migration(&v3_path, &v5_dir);

        let cfg = MigrationConfig::new(v3_path.clone(), v5_dir.clone());
        let stats = migrate(&cfg).expect("bond migration succeeded");

        assert!(v5_dir.join("bond_store").exists(), "bond_store/ created");
        assert!(stats.bsst_files_written >= 1);
        assert!(stats.bsst_total_bytes > 0);
        assert_eq!(stats.bonds_read, stats.bonds_written, "one row per v3 bond");

        // Parity: v3 bond count == sum of v5 entry_count.
        let v = stats.verification.expect("verification ran");
        assert!(v.is_ok(), "verification not OK: {:?}", v);
        assert_eq!(v.mismatch_count, 0);
        assert!(v.count_matches);
        assert_eq!(v.v3_bond_count, v.v5_bond_count);
        assert_eq!(v.v3_bond_count, stats.bonds_written);

        // The run scratch dir was cleaned up.
        assert!(
            !v5_dir.join("bond_store").join(".migrate_runs").exists(),
            "run scratch dir cleaned up",
        );
    }

    /// (ii) Force MANY external-sort runs by setting a tiny run size,
    /// proving the k-way merge produces a correctly sorted stream and
    /// the same parity as a single-run migration.
    #[test]
    fn phase310_external_sort_many_runs() {
        let tmp = tempdir().unwrap();
        let v3_path = tmp.path().join("brain.lbr");
        let v5_dir = tmp.path().join("brain-v5");
        let _seeds = build_v3_brain(&v3_path, 2_000, 0x1234_5678);
        run_node_migration(&v3_path, &v5_dir);

        let mut cfg = MigrationConfig::new(v3_path, v5_dir);
        // 50 records/run -> dozens of runs over ~2000 bonds.
        cfg.sort_run_entries = 50;
        // Small per-SST cap -> several BondSST files, exercising
        // subject-boundary rotation.
        cfg.target_entries_per_sst = 200;
        let stats = migrate(&cfg).expect("bond migration succeeded");

        assert!(stats.sort_runs_spilled > 1, "expected multiple runs");
        assert!(stats.bsst_files_written > 1, "expected multiple BondSSTs");
        let v = stats.verification.expect("verification ran");
        assert!(v.is_ok(), "verify not OK: {:?}", v);
        assert_eq!(v.v3_bond_count, v.v5_bond_count);
    }

    /// (iii) Q4 EMPIRICAL: attempt to add two predicates on the SAME
    /// ordered pair in the v3 network; confirm only ONE survives
    /// (the network keys by (from,to)), so the migrated v5 store has
    /// exactly one row for that pair and parity holds.
    #[test]
    fn phase310_q4_one_predicate_per_pair() {
        let tmp = tempdir().unwrap();
        let v3_path = tmp.path().join("brain.lbr");
        let v5_dir = tmp.path().join("brain-v5");

        let mut brain = Brain::new();
        let mut mk = |b: u8| {
            let mut sb = [0u8; 32];
            sb[0] = b;
            let s = Seed::from_bytes(sb);
            let mut node = KnowledgeNode::new(
                format!("c{}", b),
                KnowledgeContent::Concept(format!("c{}", b)),
                Confidence::Verified,
                0,
            );
            node.seed = s;
            brain.network.add_node(node);
            s
        };
        let a = mk(1);
        let b = mk(2);
        // Two bonds on the SAME ordered pair (a,b) with DIFFERENT
        // predicates. The network's HashMap<(Seed,Seed),_> keeps only
        // the LAST -> exactly one bond for (a,b).
        for bt in [BondType::IsA, BondType::Causes] {
            brain
                .network
                .add_bond(TemporalBond {
                    from: a,
                    to: b,
                    bond_type: bt,
                    keyframes: vec![TemporalKeyframe {
                        timestamp: 0,
                        weight: 0.5,
                        source: KeyframeSource::Baseline,
                    }],
                    half_life_ms: 86_400_000.0,
                })
                .unwrap();
        }
        assert_eq!(brain.network.bond_count(), 1, "Q4: one bond per pair");
        brain
            .save_to_format(&v3_path, BrainSaveFormat::V3BinaryCompressed)
            .unwrap();

        run_node_migration(&v3_path, &v5_dir);
        let cfg = MigrationConfig::new(v3_path, v5_dir);
        let stats = migrate(&cfg).expect("bond migration succeeded");
        assert_eq!(stats.bonds_written, 1, "exactly one migrated bond row");
        let v = stats.verification.expect("verification ran");
        assert!(v.is_ok(), "verify not OK: {:?}", v);
        assert_eq!(v.v3_bond_count, 1);
        assert_eq!(v.v5_bond_count, 1);
    }

    /// (iv.a) bond_store/ already exists -> hard error, no overwrite.
    #[test]
    fn phase310_bond_store_exists_errors_out() {
        let tmp = tempdir().unwrap();
        let v3_path = tmp.path().join("brain.lbr");
        let v5_dir = tmp.path().join("brain-v5");
        let _seeds = build_v3_brain(&v3_path, 100, 0xAABB);
        run_node_migration(&v3_path, &v5_dir);
        fs::create_dir_all(v5_dir.join("bond_store")).unwrap();

        let cfg = MigrationConfig::new(v3_path, v5_dir);
        let err = migrate(&cfg).expect_err("existing bond_store must error");
        assert!(
            matches!(err, MigrationError::BondStoreExists(_)),
            "wrong error: {:?}",
            err
        );
    }

    /// (iv.b) v5 dir not node-migrated -> hard error.
    #[test]
    fn phase310_v5_not_node_migrated_errors_out() {
        let tmp = tempdir().unwrap();
        let v3_path = tmp.path().join("brain.lbr");
        let v5_dir = tmp.path().join("brain-v5-empty");
        let _seeds = build_v3_brain(&v3_path, 50, 0xCCDD);
        fs::create_dir_all(&v5_dir).unwrap(); // empty -- no manifest

        let cfg = MigrationConfig::new(v3_path, v5_dir);
        let err = migrate(&cfg).expect_err("non-node-migrated v5 must error");
        assert!(
            matches!(err, MigrationError::V5DirNotNodeMigrated(_)),
            "wrong error: {:?}",
            err
        );
    }

    /// (iv.c) v3 brain missing -> hard error.
    #[test]
    fn phase310_v3_missing_errors_out() {
        let tmp = tempdir().unwrap();
        let v3_path = tmp.path().join("nope.lbr");
        let v5_dir = tmp.path().join("brain-v5");
        let cfg = MigrationConfig::new(v3_path, v5_dir);
        let err = migrate(&cfg).expect_err("missing v3 must error");
        assert!(
            matches!(err, MigrationError::V3PathMissing(_)),
            "wrong error: {:?}",
            err
        );
    }

    /// (v) Empty v3 brain (no bonds) -> trivially-verified, zero
    /// BondSSTs, parity 0 == 0.
    #[test]
    fn phase310_empty_brain_zero_bonds() {
        let tmp = tempdir().unwrap();
        let v3_path = tmp.path().join("brain.lbr");
        let v5_dir = tmp.path().join("brain-v5");

        let mut brain = Brain::new();
        let mut node = KnowledgeNode::new(
            "solo".into(),
            KnowledgeContent::Concept("solo".into()),
            Confidence::Verified,
            0,
        );
        node.seed = Seed::from_bytes([7u8; 32]);
        brain.network.add_node(node);
        brain
            .save_to_format(&v3_path, BrainSaveFormat::V3BinaryCompressed)
            .unwrap();

        run_node_migration(&v3_path, &v5_dir);
        let cfg = MigrationConfig::new(v3_path, v5_dir);
        let stats = migrate(&cfg).expect("bond migration succeeded on bondless brain");
        assert_eq!(stats.bonds_read, 0);
        assert_eq!(stats.bonds_written, 0);
        assert_eq!(stats.bsst_files_written, 0);
        let v = stats.verification.expect("verification ran");
        assert!(v.is_ok(), "verify not OK: {:?}", v);
        assert_eq!(v.v3_bond_count, 0);
        assert_eq!(v.v5_bond_count, 0);
    }

    // ====================================================================
    // OP-BRAIN-V5 6.1/6.2/6.4 -- loader-cutover end-to-end tests.
    // Migrate a small ENCRYPTED fixture v3 brain through the node + bond
    // tools, then stand it up via Brain::load_v5 and verify reads under
    // V5Only match the original v3 brain.
    // ====================================================================

    /// Build a small ENCRYPTED v3 brain with a deterministic node + bond
    /// graph. Returns (subjects, expected) where `expected` maps each
    /// subject Seed to the set of (object Seed, predicate_byte,
    /// quantized_weight) it must have after migration.
    fn build_encrypted_v3_fixture(
        v3_path: &std::path::Path,
        key: &MasterKey,
    ) -> (
        Vec<Seed>,
        std::collections::HashMap<Seed, StdHashSet<(Seed, u8, u8)>>,
    ) {
        let mut brain = Brain::new();
        let n = 60usize;
        let mut seeds = Vec::with_capacity(n);
        for i in 0..n {
            let label = format!("v5cut{:04}", i);
            let mut node = KnowledgeNode::new(
                label.clone(),
                KnowledgeContent::Concept(label.clone()),
                Confidence::Verified,
                0,
            );
            node.seed = Seed::from_label(&label);
            seeds.push(node.seed);
            brain.network.add_node(node);
        }
        let bts = [
            BondType::IsA,
            BondType::HasProperty,
            BondType::RelatedTo,
            BondType::Causes,
        ];
        let weights = [0.25f32, 0.5, 0.75, 0.9];
        let mut expected: std::collections::HashMap<Seed, StdHashSet<(Seed, u8, u8)>> =
            std::collections::HashMap::new();
        // Deterministic graph: node i -> i+1, i+2, i+3 (mod n) with
        // cycling predicate + weight. v3 keys bonds by the ORDERED pair
        // so every (from,to) here is unique.
        for i in 0..n {
            for step in 1..=3usize {
                let j = (i + step) % n;
                if i == j {
                    continue;
                }
                let from = seeds[i];
                let to = seeds[j];
                let bt = bts[(i + step) % bts.len()];
                let w = weights[(i + step) % weights.len()];
                brain
                    .network
                    .add_bond(TemporalBond {
                        from,
                        to,
                        bond_type: bt,
                        keyframes: vec![TemporalKeyframe {
                            timestamp: (i * 8 + step) as i64,
                            weight: w,
                            source: KeyframeSource::Baseline,
                        }],
                        half_life_ms: 86_400_000.0,
                    })
                    .unwrap();
                expected
                    .entry(from)
                    .or_default()
                    .insert((to, predicate_byte(bt), quantize_weight(w)));
            }
        }
        brain.encryption_key = Some(key.clone());
        brain
            .save_to_format(v3_path, BrainSaveFormat::V3BinaryEncrypted)
            .unwrap();
        let subjects: Vec<Seed> = seeds.iter().take(n - 3).cloned().collect();
        (subjects, expected)
    }

    /// Run BOTH migration tools with `key`, producing an encrypted v5
    /// dir. Asserts both verification gates pass.
    fn migrate_v3_to_v5_encrypted(
        v3_path: &std::path::Path,
        v5_dir: &std::path::Path,
        key: &MasterKey,
    ) {
        let mut ncfg =
            node_mig::MigrationConfig::new(v3_path.to_path_buf(), v5_dir.to_path_buf());
        ncfg.key = Some(key.clone());
        let nstats = node_mig::migrate(&ncfg).expect("node migration succeeded");
        assert!(
            nstats.verification.expect("node verify ran").is_ok(),
            "node migration verify must pass"
        );
        let mut bcfg = MigrationConfig::new(v3_path.to_path_buf(), v5_dir.to_path_buf());
        bcfg.key = Some(key.clone());
        let bstats = migrate(&bcfg).expect("bond migration succeeded");
        assert!(
            bstats.verification.expect("bond verify ran").is_ok(),
            "bond migration verify must pass"
        );
    }

    /// 6.1 + 6.4 + 6.2: detect -> V5Dir, the require-encrypted guard
    /// accepts the dir, load_v5 boots in V5Only, and sampled node +
    /// outgoing-bond reads match the source graph.
    #[test]
    fn cutover_v5_e2e_encrypted_load_read_parity() {
        let tmp = tempdir().unwrap();
        let v3_path = tmp.path().join("brain.lbr");
        let v5_dir = tmp.path().join("brain-v5");
        let key = MasterKey::from_bytes([0x5Au8; 32]);

        let (subjects, expected) = build_encrypted_v3_fixture(&v3_path, &key);
        migrate_v3_to_v5_encrypted(&v3_path, &v5_dir, &key);

        // 6.1: detection classifies the dir as V5Dir (NOT V4Dir), via the
        // explicit manifest format tag.
        assert_eq!(
            Brain::detect_brain_layout(&v5_dir).unwrap(),
            crate::BrainLayout::V5Dir,
            "v5 dir must detect as V5Dir, not V4Dir"
        );
        // 6.5 guard: an encrypted v5 dir passes --require-encrypted.
        assert!(
            crate::brain::require_v3enc_or_err(&v5_dir).is_ok(),
            "encrypted v5 dir must satisfy require-encrypted"
        );

        // 6.4: boot via load_v5 with the key.
        let brain = Brain::load_v5(&v5_dir, Some(key.clone()))
            .expect("load_v5 must boot an encrypted v5 brain with the key");
        // 6.2: read path mode is V5Only.
        assert_eq!(
            brain.read_path_mode(),
            crate::read_path::ReadPathMode::V5Only,
            "load_v5 must set V5Only"
        );

        let intern = brain.network.node_intern();
        let mut checked_nodes = 0usize;
        let mut checked_bonds = 0usize;
        for s in &subjects {
            let nid = intern.get(s).expect("subject interned in v5");

            // Node read via the v5 node path.
            let node = brain
                .get_node_v5(nid)
                .expect("get_node_v5 ok")
                .expect("node present in v5 store");
            assert_eq!(node.seed, *s, "get_node_v5 seed parity");
            checked_nodes += 1;

            // Outgoing bonds via the v5 SST read path, mapped back to
            // Seeds, compared against the source-derived expected set.
            let v5 = brain.get_outgoing_bonds_v5(nid).expect("v5 outgoing ok");
            let got: StdHashSet<(Seed, u8, u8)> = v5
                .iter()
                .map(|(_subj, pred, obj, payload)| {
                    let obj_seed =
                        intern.reverse_lookup(*obj).expect("object NodeId -> Seed");
                    (obj_seed, *pred, payload.weight)
                })
                .collect();
            let want = expected.get(s).cloned().unwrap_or_default();
            assert_eq!(got, want, "v5 outgoing-bond parity for subject {:?}", s);
            checked_bonds += got.len();

            // 6.2: the V5Only recall path (Brain::read_outgoing_bonds,
            // served from the rehydrated adjacency) returns the same
            // object set.
            let recall = brain.read_outgoing_bonds(s).expect("V5Only recall ok");
            let recall_objs: StdHashSet<Seed> = recall.iter().map(|m| m.object).collect();
            let want_objs: StdHashSet<Seed> = want.iter().map(|(o, _, _)| *o).collect();
            assert_eq!(
                recall_objs, want_objs,
                "V5Only read_outgoing_bonds object parity for {:?}",
                s
            );
        }
        assert!(checked_nodes >= 50, "sampled enough nodes ({})", checked_nodes);
        assert!(checked_bonds >= 100, "sampled enough bonds ({})", checked_bonds);
    }

    /// 6.4 + 6.5: load_v5 against an encrypted v5 dir REFUSES when the
    /// key is missing or wrong (encrypted SSTs cannot be read -> load
    /// fails rather than booting a half-empty brain).
    #[test]
    fn cutover_v5_load_refuses_wrong_or_missing_key() {
        let tmp = tempdir().unwrap();
        let v3_path = tmp.path().join("brain.lbr");
        let v5_dir = tmp.path().join("brain-v5");
        let key = MasterKey::from_bytes([0x5Au8; 32]);
        let (_subjects, _expected) = build_encrypted_v3_fixture(&v3_path, &key);
        migrate_v3_to_v5_encrypted(&v3_path, &v5_dir, &key);

        // No key: encrypted NodeSST/BondSST cannot be read -> load_v5
        // fails during rehydration. Must NOT silently succeed.
        assert!(
            Brain::load_v5(&v5_dir, None).is_err(),
            "load_v5 with NO key on an encrypted v5 brain must fail"
        );
        // Wrong key: decrypt garbage -> CRC failure -> load fails.
        let wrong = MasterKey::from_bytes([0x01u8; 32]);
        assert!(
            Brain::load_v5(&v5_dir, Some(wrong)).is_err(),
            "load_v5 with the WRONG key must fail"
        );
    }

    /// OP-BRAIN-V5 6.3 -- INGEST-CORRECTNESS end-to-end. The real proof
    /// the Seed-vs-NodeId MemBond gap is closed: on a migrated, encrypted,
    /// keyed v5 brain, a brand-new bond taught AFTER boot is recallable
    /// under V5Only BEFORE any flush; STILL recallable after a forced flush
    /// (served from the SST now); and STILL recallable after a full reload
    /// from disk (persistence across restart). Also proves NO duplicate is
    /// returned when the same triple lives in BOTH the MemTable and an SST.
    #[test]
    fn cutover_v5_6_3_ingest_recall_pre_post_flush_and_restart() {
        let tmp = tempdir().unwrap();
        let v3_path = tmp.path().join("brain.lbr");
        let v5_dir = tmp.path().join("brain-v5");
        let key = MasterKey::from_bytes([0x5Au8; 32]);

        let (subjects, _expected) = build_encrypted_v3_fixture(&v3_path, &key);
        migrate_v3_to_v5_encrypted(&v3_path, &v5_dir, &key);

        let mut brain = Brain::load_v5(&v5_dir, Some(key.clone()))
            .expect("load_v5 boots encrypted v5 brain");
        assert_eq!(brain.read_path_mode(), crate::read_path::ReadPathMode::V5Only);

        // Existing migrated subject + a BRAND-NEW object Seed that exists
        // nowhere in the migrated brain. UsedFor is not used by the fixture
        // graph, so it cannot collide with this subject's existing edges.
        let subj_label = "v5cut0000";
        let subj = Seed::from_label(subj_label);
        assert_eq!(subj, subjects[0], "subj label matches fixture subject 0");
        let new_obj_label = "v5cut_ingest_new_object_6_3";
        let new_obj = Seed::from_label(new_obj_label);
        let pred = BondType::UsedFor;
        let pred_disc = crate::memtable::predicate_discriminant(pred);

        let subj_id = brain.network.node_intern().get(&subj)
            .expect("migrated subject is interned");

        // Occurrences of the new bond in the MERGED V5Only recall path.
        let recall_count = |b: &Brain| -> usize {
            b.read_outgoing_bonds(&subj).expect("V5Only recall ok").iter()
                .filter(|m| m.object == new_obj && m.predicate == pred && !m.tombstone)
                .count()
        };
        // Occurrences in the SST-ONLY surface (no MemTable merge).
        let sst_count = |b: &Brain| -> usize {
            let intern = b.network.node_intern();
            b.get_outgoing_bonds_v5(subj_id).expect("get_outgoing_bonds_v5 ok").iter()
                .filter(|(_s, p, o, _pl)| *p == pred_disc && intern.reverse_lookup(*o) == Some(new_obj))
                .count()
        };

        // Baseline: bond exists in neither surface.
        assert_eq!(recall_count(&brain), 0, "new bond absent before teach (recall)");
        assert_eq!(sst_count(&brain), 0, "new bond absent before teach (sst)");

        // TEACH the new bond. load_v5 set brain_dir_v4, so teach routes
        // through teach_with_kind_v4 -> MemTable + WAL. NOT flushed yet.
        brain.teach(subj_label, pred, new_obj_label, 0.8, Confidence::Verified);
        assert!(brain.memtable.as_ref().unwrap().bond_count() >= 1,
            "taught bond resident in MemTable pre-flush");

        // PROOF 1 (the core gap fix): recallable under V5Only BEFORE any
        // flush, even though the SST-only surface still cannot see it.
        assert_eq!(sst_count(&brain), 0,
            "pre-flush: bond is NOT yet in any SST (MemTable-only)");
        assert_eq!(recall_count(&brain), 1,
            "6.3: just-taught bond MUST be recallable under V5Only before flush, exactly once");

        // Force a synchronous flush -> bond moves to a v5 BondSST.
        brain.flush_now().expect("flush_now ok");
        assert_eq!(brain.memtable.as_ref().unwrap().bond_count(), 0,
            "MemTable bond half empty after flush");

        // PROOF 2: still recallable AFTER flush, now from the SST via the
        // live manifest snapshot get_outgoing_bonds_v5 consults.
        assert_eq!(sst_count(&brain), 1, "post-flush: bond now lives in an SST");
        assert_eq!(recall_count(&brain), 1,
            "6.3: bond stays recallable after flush, exactly once");

        // Re-TEACH the SAME triple: now in BOTH the fresh MemTable AND the
        // just-written SST. The merge must de-dup to ONE (MemTable wins).
        brain.teach(subj_label, pred, new_obj_label, 0.95, Confidence::Verified);
        assert_eq!(sst_count(&brain), 1, "older copy still in the SST");
        assert!(brain.memtable.as_ref().unwrap().bond_count() >= 1,
            "re-taught copy resident in MemTable");
        assert_eq!(recall_count(&brain), 1,
            "6.3: NO duplicate when the bond lives in BOTH the MemTable and an SST");

        // Persist the second copy and reload from disk.
        brain.flush_now().expect("second flush ok");
        drop(brain);
        let brain2 = Brain::load_v5(&v5_dir, Some(key.clone())).expect("reload via load_v5");
        assert_eq!(brain2.read_path_mode(), crate::read_path::ReadPathMode::V5Only);

        // PROOF 3: persistence across restart -- still recallable, once.
        let subj_id2 = brain2.network.node_intern().get(&subj)
            .expect("subject interned after reload");
        assert_eq!(subj_id2, subj_id, "stable NodeId across restart");
        assert_eq!(
            brain2.read_outgoing_bonds(&subj).expect("recall ok").iter()
                .filter(|m| m.object == new_obj && m.predicate == pred && !m.tombstone)
                .count(),
            1,
            "6.3: bond persists across a load_v5 restart, exactly once");
    }

    /// (e) The --require-encrypted guard hard-fails a plaintext WAL AND a
    /// plaintext sidecar (a v5 dir is "encrypted" only if SSTs AND WAL
    /// AND sidecar all are). OP-BRAIN-V5 6.8.
    #[test]
    fn enc68_guard_hardfails_plaintext_wal_and_sidecar() {
        let tmp = tempdir().unwrap();
        let v3_path = tmp.path().join("brain.lbr");
        let v5_dir = tmp.path().join("brain-v5");
        let key = MasterKey::from_bytes([0x5Au8; 32]);
        let (_subjects, _expected) = build_encrypted_v3_fixture(&v3_path, &key);
        migrate_v3_to_v5_encrypted(&v3_path, &v5_dir, &key);

        // Fully encrypted (encrypted SSTs + encrypted sidecar, no WAL yet).
        crate::brain::require_v3enc_or_err(&v5_dir)
            .expect("fully-encrypted v5 dir must pass the guard");

        let sidecar = v5_dir.join("wal").join("nodeintern.bin");
        let wal_path = v5_dir.join("wal").join("current.wal");
        let intern = NodeIntern::load_from_with_key(&sidecar, Some(key.clone())).unwrap();

        // (1) Plaintext sidecar -> hard fail.
        intern.save_to_with_key(&sidecar, None).unwrap();
        assert_eq!(
            crate::node_store::sidecar::sidecar_encryption_status(&sidecar).unwrap(),
            crate::node_store::sidecar::SidecarEncStatus::Plaintext,
        );
        assert!(crate::brain::require_v3enc_or_err(&v5_dir).is_err(),
            "guard must hard-fail a plaintext sidecar");
        // Restore the encrypted sidecar -> guard passes again.
        intern.save_to_with_key(&sidecar, Some(key.clone())).unwrap();
        crate::brain::require_v3enc_or_err(&v5_dir)
            .expect("guard passes once sidecar is re-encrypted");

        // (2) Plaintext WAL -> hard fail.
        {
            let mut wal = crate::wal::Wal::open_with_key(&wal_path, None).unwrap();
            let s = Seed::from_label("plain_subj");
            let o = Seed::from_label("plain_obj");
            wal.append(0, &s, &o, BondType::IsA, 0.5, &[crate::bond::TemporalKeyframe {
                timestamp: 0, weight: 0.5, source: crate::bond::KeyframeSource::Baseline,
            }]).unwrap();
            wal.fsync().unwrap();
        }
        assert_eq!(
            crate::wal::wal_encryption_status(&wal_path).unwrap(),
            crate::wal::WalEncStatus::Plaintext,
        );
        assert!(crate::brain::require_v3enc_or_err(&v5_dir).is_err(),
            "guard must hard-fail a plaintext WAL");
    }

    /// (g) End-to-end: teach -> hard-kill -> restart recovers the taught
    /// bond by replaying the ENCRYPTED WAL (never flushed to an SST).
    /// OP-BRAIN-V5 6.8.
    #[test]
    fn enc68_e2e_teach_hardkill_restart_recovers_via_encrypted_wal() {
        let tmp = tempdir().unwrap();
        let v3_path = tmp.path().join("brain.lbr");
        let v5_dir = tmp.path().join("brain-v5");
        let key = MasterKey::from_bytes([0x5Au8; 32]);
        let (subjects, _expected) = build_encrypted_v3_fixture(&v3_path, &key);
        migrate_v3_to_v5_encrypted(&v3_path, &v5_dir, &key);

        let subj_label = "v5cut0000";
        let subj = Seed::from_label(subj_label);
        assert_eq!(subj, subjects[0]);
        let new_obj_label = "enc68_taught_object";
        let new_obj = Seed::from_label(new_obj_label);
        let pred = BondType::UsedFor;

        let wal_path = v5_dir.join("wal").join("current.wal");
        {
            let mut brain = Brain::load_v5(&v5_dir, Some(key.clone()))
                .expect("load encrypted v5 brain");
            brain.teach(subj_label, pred, new_obj_label, 0.8, Confidence::Verified);
            assert!(brain.memtable.as_ref().unwrap().bond_count() >= 1,
                "taught bond resident in MemTable (not yet flushed)");
            // Force the just-taught record durable in the WAL, then HARD
            // KILL (drop WITHOUT flush_now -> bond is in the WAL ONLY).
            { let w = brain.wal().expect("wal present"); w.lock().unwrap().fsync().unwrap(); }
            assert_eq!(
                crate::wal::wal_encryption_status(&wal_path).unwrap(),
                crate::wal::WalEncStatus::Encrypted,
                "taught bonds land in an ENCRYPTED WAL",
            );
            drop(brain);
        }

        // Restart: load_v5 replays the ENCRYPTED WAL with the key.
        let brain2 = Brain::load_v5(&v5_dir, Some(key.clone()))
            .expect("reload after hard kill");
        let n = brain2.read_outgoing_bonds(&subj).expect("recall ok").iter()
            .filter(|m| m.object == new_obj && m.predicate == pred && !m.tombstone)
            .count();
        assert_eq!(n, 1, "taught bond recovered via encrypted-WAL replay, exactly once");
    }
}
