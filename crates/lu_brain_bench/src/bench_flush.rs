//! OP-BRAIN-V4 Phase 4 — Flush worker bench.
//!
//! Measures the end-to-end flush pipeline:
//!   - SST write throughput (MB/s of payload written, derived from
//!     the on-disk file size produced by one flush).
//!   - Manifest update latency (the wall-clock cost of save_atomic).
//!   - Sustained ingest rate with the worker running in background.
//!
//! Output: a Markdown report at the path passed via `--out`.

use std::env;
use std::fs::{self, File};
use std::io::Write;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, RwLock};
use std::sync::atomic::AtomicBool;
use std::time::Instant;

use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

use lu4d::bond::{BondType, KeyframeSource, TemporalKeyframe};
use lu4d::flush::FlushWorker;
use lu4d::manifest::{KeyFingerprint, Manifest, MANIFEST_FILE_NAME};
use lu4d::memtable::MemTable;
use lu4d::node_intern::NodeIntern;
use lu4d::seed::Seed;
use lu4d::wal::Wal;

fn rand_seed(rng: &mut StdRng) -> Seed {
    let mut b = [0u8; 32];
    rng.fill(&mut b);
    Seed::from_bytes(b)
}

fn baseline_kf(ts: i64, w: f32) -> TemporalKeyframe {
    TemporalKeyframe { timestamp: ts, weight: w, source: KeyframeSource::Baseline }
}

struct Args {
    out: PathBuf,
    fill_bonds: usize,
    sustained_bonds: usize,
    sustained_cap_kib: usize,
}

fn parse_args() -> Args {
    let mut out = PathBuf::from("docs/v4_phase4_bench_2026-06-01.md");
    let mut fill_bonds: usize = 200_000;
    let mut sustained_bonds: usize = 500_000;
    let mut sustained_cap_kib: usize = 64;
    let mut it = env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--out" => { out = PathBuf::from(it.next().expect("--out arg")); }
            "--fill-bonds" => { fill_bonds = it.next().unwrap().parse().unwrap(); }
            "--sustained-bonds" => { sustained_bonds = it.next().unwrap().parse().unwrap(); }
            "--sustained-cap-kib" => { sustained_cap_kib = it.next().unwrap().parse().unwrap(); }
            _ => { eprintln!("unknown arg: {}", a); std::process::exit(1); }
        }
    }
    Args { out, fill_bonds, sustained_bonds, sustained_cap_kib }
}

struct Rig {
    _dir: tempfile::TempDir,
    sst_dir: PathBuf,
    manifest_path: PathBuf,
    sidecar_path: PathBuf,
    node_store_dir: PathBuf,
    bond_store_dir: PathBuf,
    memtable: Arc<MemTable>,
    wal: Arc<Mutex<Wal>>,
    intern: NodeIntern,
    manifest: Arc<RwLock<Arc<Manifest>>>,
}

fn make_rig(cap_bytes: usize) -> Rig {
    let dir = tempfile::tempdir().unwrap();
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
    let sidecar_path = lu4d::brain::Brain::sidecar_path_for_wal(&wal_path);
    let node_store_dir = lu4d::node_store::NodeFlushWorker::node_store_dir_for_wal(&wal_path);
    let bond_store_dir = lu4d::bond_store::BondFlushWorker::bond_store_dir_for_wal(&wal_path);
    Rig { _dir: dir, sst_dir, manifest_path, sidecar_path, node_store_dir, bond_store_dir, memtable, wal, intern, manifest }
}

fn fill_memtable(rig: &Rig, n: usize, rng: &mut StdRng) {
    for i in 0..n {
        let subj = rand_seed(rng);
        let obj = rand_seed(rng);
        let kfs = vec![baseline_kf(i as i64, 0.5)];
        let r = rig.memtable.insert(
            subj, obj, BondType::IsA, 0.5, kfs.clone(), &rig.intern,
        );
        rig.wal.lock().unwrap().append(
            r.seq, &subj, &obj, BondType::IsA, 0.5, &kfs,
        ).unwrap();
    }
    rig.wal.lock().unwrap().fsync().unwrap();
}

fn main() -> std::io::Result<()> {
    let args = parse_args();
    eprintln!("[bench_flush] fill_bonds={}, sustained_bonds={}, cap_kib={}",
        args.fill_bonds, args.sustained_bonds, args.sustained_cap_kib);

    // Scenario A: one big flush_now. Measures SST write throughput +
    // manifest update latency on a populated MemTable.
    let rig_a = make_rig(8 * 1024 * 1024 * 1024); // 8 GiB cap => only flush via flush_now
    let mut rng = StdRng::seed_from_u64(0xCAFE_BABE);
    let t_fill = Instant::now();
    fill_memtable(&rig_a, args.fill_bonds, &mut rng);
    let fill_elapsed = t_fill.elapsed();
    let worker_a = FlushWorker::start(
        rig_a.memtable.clone(), rig_a.wal.clone(), rig_a.intern.clone(),
        rig_a.manifest.clone(),
        rig_a.manifest_path.clone(), rig_a.sst_dir.clone(), None,
        Arc::new(AtomicBool::new(false)),
        rig_a.sidecar_path.clone(), rig_a.intern.clone(),
        rig_a.node_store_dir.clone(),
        rig_a.bond_store_dir.clone(),
    );
    let t_flush = Instant::now();
    worker_a.flush_now().unwrap();
    let flush_elapsed = t_flush.elapsed();
    let sst_path = rig_a.sst_dir.join("L0-1.sst");
    let sst_size = fs::metadata(&sst_path).map(|m| m.len()).unwrap_or(0);
    let sst_mb = sst_size as f64 / (1024.0 * 1024.0);
    let throughput_mb_s = if flush_elapsed.as_secs_f64() > 0.0 {
        sst_mb / flush_elapsed.as_secs_f64()
    } else { 0.0 };
    let bonds_per_s_a = args.fill_bonds as f64 / flush_elapsed.as_secs_f64();
    eprintln!("[bench_flush] A: fill {:.2}s, flush {:.3}s, SST {:.2} MiB, {:.1} MB/s, {:.0} bonds/s",
        fill_elapsed.as_secs_f64(), flush_elapsed.as_secs_f64(),
        sst_mb, throughput_mb_s, bonds_per_s_a);

    // Scenario B: sustained ingest with a small cap so the background
    // worker repeatedly flushes. Measure end-to-end bonds/sec.
    let rig_b = make_rig(args.sustained_cap_kib * 1024);
    let mut worker_b = FlushWorker::start(
        rig_b.memtable.clone(), rig_b.wal.clone(), rig_b.intern.clone(),
        rig_b.manifest.clone(),
        rig_b.manifest_path.clone(), rig_b.sst_dir.clone(), None,
        Arc::new(AtomicBool::new(false)),
        rig_b.sidecar_path.clone(), rig_b.intern.clone(),
        rig_b.node_store_dir.clone(),
        rig_b.bond_store_dir.clone(),
    );
    worker_b.set_poll_interval(10);
    let mut rng = StdRng::seed_from_u64(0xDEAD_BEEF);
    let t_b = Instant::now();
    for i in 0..args.sustained_bonds {
        let subj = rand_seed(&mut rng);
        let obj = rand_seed(&mut rng);
        let kfs = vec![baseline_kf(i as i64, 0.5)];
        let r = rig_b.memtable.insert(
            subj, obj, BondType::IsA, 0.5, kfs.clone(), &rig_b.intern,
        );
        rig_b.wal.lock().unwrap().append(
            r.seq, &subj, &obj, BondType::IsA, 0.5, &kfs,
        ).unwrap();
    }
    rig_b.wal.lock().unwrap().fsync().unwrap();
    // Force one final flush to capture everything.
    worker_b.flush_now().unwrap();
    let elapsed_b = t_b.elapsed();
    let bonds_per_s_b = args.sustained_bonds as f64 / elapsed_b.as_secs_f64();
    let l0_count = rig_b.manifest.read().unwrap().total_sst_count();
    eprintln!("[bench_flush] B: sustained {:.2}s, {} bonds, {:.0} bonds/s, {} L0 SSTs",
        elapsed_b.as_secs_f64(), args.sustained_bonds, bonds_per_s_b, l0_count);

    // Manifest update latency: time 100 save_atomic round-trips on a
    // small manifest. This isolates the directory-fsync cost.
    let mut manifest = Manifest::new(KeyFingerprint::none());
    for _ in 0..10 {
        let id = manifest.next_sst_id();
        manifest.add_sst(0, lu4d::manifest::SSTEntry {
            id,
            file: lu4d::manifest::sst_filename(0, id),
            subject_min: [0; 32],
            subject_max: [0xFF; 32],
            size: 1024,
            mtime: 0,
            bond_count: 1,
            epoch: 0,
        });
    }
    let bench_dir = tempfile::tempdir().unwrap();
    let bench_path = bench_dir.path().join("MANIFEST");
    manifest.save_atomic(&bench_path).unwrap();
    let t_m = Instant::now();
    for _ in 0..100 {
        manifest.save_atomic(&bench_path).unwrap();
    }
    let manifest_latency_us = t_m.elapsed().as_micros() as f64 / 100.0;

    // Write the report.
    let mut out = File::create(&args.out)?;
    writeln!(out, "# OP-BRAIN-V4 Phase 4 — flush bench")?;
    writeln!(out, "")?;
    writeln!(out, "Generated by `bench_flush` 2026-06-01.")?;
    writeln!(out, "")?;
    writeln!(out, "## Scenario A — single large flush_now")?;
    writeln!(out, "")?;
    writeln!(out, "| metric | value |")?;
    writeln!(out, "|--------|-------|")?;
    writeln!(out, "| bonds  | {} |", args.fill_bonds)?;
    writeln!(out, "| fill_elapsed_s | {:.3} |", fill_elapsed.as_secs_f64())?;
    writeln!(out, "| flush_elapsed_s | {:.3} |", flush_elapsed.as_secs_f64())?;
    writeln!(out, "| sst_size_MiB | {:.2} |", sst_mb)?;
    writeln!(out, "| sst_throughput_MB_per_s | {:.1} |", throughput_mb_s)?;
    writeln!(out, "| flush_bonds_per_s | {:.0} |", bonds_per_s_a)?;
    writeln!(out, "")?;
    writeln!(out, "## Scenario B — sustained ingest with background flush")?;
    writeln!(out, "")?;
    writeln!(out, "| metric | value |")?;
    writeln!(out, "|--------|-------|")?;
    writeln!(out, "| total_bonds | {} |", args.sustained_bonds)?;
    writeln!(out, "| memtable_cap_KiB | {} |", args.sustained_cap_kib)?;
    writeln!(out, "| sustained_elapsed_s | {:.3} |", elapsed_b.as_secs_f64())?;
    writeln!(out, "| sustained_bonds_per_s | {:.0} |", bonds_per_s_b)?;
    writeln!(out, "| L0_SSTs_produced | {} |", l0_count)?;
    writeln!(out, "")?;
    writeln!(out, "## Manifest update latency")?;
    writeln!(out, "")?;
    writeln!(out, "| metric | value |")?;
    writeln!(out, "|--------|-------|")?;
    writeln!(out, "| save_atomic_us | {:.1} |", manifest_latency_us)?;
    writeln!(out, "")?;
    writeln!(out, "## Durability sequence (verified by tests)")?;
    writeln!(out, "")?;
    writeln!(out, "1. SST file `finalize_with_meta` -> `sync_all` (fsync).")?;
    writeln!(out, "2. Manifest `save_atomic` writes `.tmp`, fsyncs, rotates `MANIFEST -> .prev`, renames `.tmp -> MANIFEST`, fsyncs the parent directory.")?;
    writeln!(out, "3. WAL `truncate_up_to(seq)` only AFTER (1) and (2) succeed.")?;
    writeln!(out, "")?;
    writeln!(out, "Crash between (1) and (2): SST is an orphan; `manifest::cleanup_orphan_ssts` renames it to `.orphan-<ts>` on restart. WAL replay reconstructs the MemTable. No data loss.")?;
    writeln!(out, "")?;
    writeln!(out, "Crash between (2) and (3): WAL replay re-inserts the (already-flushed) records into the new MemTable. Phase 5 read-path dedup + Phase 6 compaction handle the duplication. No data loss.")?;
    Ok(())
}
