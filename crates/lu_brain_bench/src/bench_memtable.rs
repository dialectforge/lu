//! OP-BRAIN-V4 Phase 3 — MemTable + WAL bench.
//!
//! Measures:
//!   - MemTable insert throughput (bonds/sec) at empty, half-full,
//!     and near-full table states.
//!   - WAL append throughput across three fsync cadences.
//!   - `swap_for_flush` latency on a populated table.
//!
//! Output: a Markdown report at the path passed via `--out`.

use std::env;
use std::fs::{self, File};
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

use lu4d::bond::{BondType, KeyframeSource, TemporalKeyframe};
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
    TemporalKeyframe {
        timestamp: ts, weight: w, source: KeyframeSource::Baseline,
    }
}

struct Args {
    out: PathBuf,
    cap_bytes: usize,
    bonds_per_phase: usize,
    wal_records: usize,
}

fn parse_args() -> Args {
    let mut out = PathBuf::from("/tmp/v4_phase3_bench.md");
    let mut cap_bytes = 64 * 1024 * 1024; // 64 MiB for bench (faster than 500)
    let mut bonds_per_phase = 200_000usize;
    let mut wal_records = 50_000usize;
    let argv: Vec<String> = env::args().collect();
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--out" => { out = PathBuf::from(&argv[i+1]); i += 2; }
            "--cap-bytes" => { cap_bytes = argv[i+1].parse().unwrap(); i += 2; }
            "--bonds" => { bonds_per_phase = argv[i+1].parse().unwrap(); i += 2; }
            "--wal-records" => { wal_records = argv[i+1].parse().unwrap(); i += 2; }
            _ => i += 1,
        }
    }
    Args { out, cap_bytes, bonds_per_phase, wal_records }
}

/// Insert N random bonds into `mt` and return the elapsed wall-clock
/// duration in nanoseconds.
fn time_insert(mt: &MemTable, intern: &NodeIntern, n: usize, rng_seed: u64) -> u128 {
    let mut rng = StdRng::seed_from_u64(rng_seed);
    let t0 = Instant::now();
    for _ in 0..n {
        mt.insert(
            rand_seed(&mut rng), rand_seed(&mut rng),
            BondType::RelatedTo, 0.5,
            vec![baseline_kf(0, 0.5)], intern,
        );
    }
    t0.elapsed().as_nanos()
}

fn bench_insert_empty(cap_bytes: usize, bonds: usize) -> f64 {
    let mt = MemTable::new(cap_bytes);
    let intern = NodeIntern::new();
    let ns = time_insert(&mt, &intern, bonds, 0xAAAA);
    bonds as f64 / (ns as f64 / 1e9)
}

fn bench_insert_half(cap_bytes: usize, bonds: usize) -> f64 {
    // Pre-fill to ~50% of cap, then time `bonds` more inserts.
    let mt = MemTable::new(cap_bytes);
    let intern = NodeIntern::new();
    let mut rng = StdRng::seed_from_u64(0xBBBB);
    while mt.size_bytes() < cap_bytes / 2 {
        for _ in 0..1000 {
            mt.insert(
                rand_seed(&mut rng), rand_seed(&mut rng),
                BondType::RelatedTo, 0.5,
                vec![baseline_kf(0, 0.5)], &intern,
            );
        }
    }
    let ns = time_insert(&mt, &intern, bonds, 0xCCCC);
    bonds as f64 / (ns as f64 / 1e9)
}

fn bench_insert_near_full(cap_bytes: usize, bonds: usize) -> f64 {
    let mt = MemTable::new(cap_bytes);
    let intern = NodeIntern::new();
    let mut rng = StdRng::seed_from_u64(0xDDDD);
    // Fill to ~90% of cap.
    let target = (cap_bytes as f64 * 0.9) as usize;
    while mt.size_bytes() < target {
        for _ in 0..1000 {
            mt.insert(
                rand_seed(&mut rng), rand_seed(&mut rng),
                BondType::RelatedTo, 0.5,
                vec![baseline_kf(0, 0.5)], &intern,
            );
        }
    }
    let ns = time_insert(&mt, &intern, bonds, 0xEEEE);
    bonds as f64 / (ns as f64 / 1e9)
}

fn bench_wal(records: usize, fsync_ms: u64, fsync_bytes: usize) -> (f64, u128) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bench.wal");
    let mut wal = Wal::open(&path).unwrap();
    wal.set_fsync_thresholds(fsync_ms, fsync_bytes);
    let mut rng = StdRng::seed_from_u64(0xFAFA);
    let t0 = Instant::now();
    for i in 0..records {
        wal.append(
            i as u64,
            &rand_seed(&mut rng), &rand_seed(&mut rng),
            BondType::RelatedTo, 0.5,
            &[baseline_kf(0, 0.5)],
        ).unwrap();
    }
    wal.fsync().unwrap();
    let ns = t0.elapsed().as_nanos();
    let rps = records as f64 / (ns as f64 / 1e9);
    (rps, ns)
}

fn bench_swap_latency(cap_bytes: usize) -> u128 {
    // Populate a MemTable to about half-full, time the swap.
    let mt = MemTable::new(cap_bytes);
    let intern = NodeIntern::new();
    let mut rng = StdRng::seed_from_u64(0x5555);
    while mt.size_bytes() < cap_bytes / 2 {
        for _ in 0..1000 {
            mt.insert(
                rand_seed(&mut rng), rand_seed(&mut rng),
                BondType::RelatedTo, 0.5, vec![], &intern,
            );
        }
    }
    let t0 = Instant::now();
    let _ = mt.swap_for_flush();
    t0.elapsed().as_nanos()
}

fn main() {
    let args = parse_args();

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();

    let empty = bench_insert_empty(args.cap_bytes, args.bonds_per_phase);
    let half = bench_insert_half(args.cap_bytes, args.bonds_per_phase);
    let near = bench_insert_near_full(args.cap_bytes, args.bonds_per_phase / 4);

    let (wal_default, _) = bench_wal(args.wal_records, 100, 1024 * 1024);
    let (wal_aggressive, _) = bench_wal(args.wal_records, 1, 4096);
    let (wal_lazy, _) = bench_wal(args.wal_records, 10_000, 64 * 1024 * 1024);

    let swap_ns = bench_swap_latency(args.cap_bytes);

    let mut report = String::new();
    report.push_str("# OP-BRAIN-V4 Phase 3 — MemTable + WAL bench\n\n");
    report.push_str(&format!("Generated: {} (Unix epoch)\n\n", now));
    report.push_str(&format!("Cap bytes: {} ({} MiB)\n", args.cap_bytes, args.cap_bytes / 1024 / 1024));
    report.push_str(&format!("Bonds per phase: {}\n", args.bonds_per_phase));
    report.push_str(&format!("WAL records per phase: {}\n\n", args.wal_records));

    report.push_str("## MemTable insert throughput\n\n");
    report.push_str("| State | Throughput (bonds/sec) |\n");
    report.push_str("|-------|------------------------:|\n");
    report.push_str(&format!("| empty | {:.0} |\n", empty));
    report.push_str(&format!("| half-full | {:.0} |\n", half));
    report.push_str(&format!("| near-full (90%) | {:.0} |\n\n", near));

    report.push_str("## WAL append throughput\n\n");
    report.push_str("| Fsync cadence | Throughput (records/sec) |\n");
    report.push_str("|---------------|--------------------------:|\n");
    report.push_str(&format!("| default (100 ms / 1 MiB) | {:.0} |\n", wal_default));
    report.push_str(&format!("| aggressive (1 ms / 4 KiB) | {:.0} |\n", wal_aggressive));
    report.push_str(&format!("| lazy (10 s / 64 MiB) | {:.0} |\n\n", wal_lazy));

    report.push_str("## Swap-for-flush latency\n\n");
    report.push_str(&format!("- Half-full table: {} ns ({:.3} us)\n", swap_ns, swap_ns as f64 / 1000.0));

    if let Some(parent) = args.out.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let mut f = File::create(&args.out).unwrap();
    f.write_all(report.as_bytes()).unwrap();
    println!("wrote {}", args.out.display());
    println!("\n{}", report);
}
