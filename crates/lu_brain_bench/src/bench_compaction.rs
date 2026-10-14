//! OP-BRAIN-V4 Phase 6 -- compaction bench.
//!
//! Scenarios:
//!
//! - **A. Compaction throughput.** Fill 10 L0 SSTs (50K bonds each),
//!   trigger compaction, measure wall-clock, throughput (MB/s),
//!   post-state level distribution.
//! - **B. Read latency POST-compaction.** Measure
//!   `read_outgoing_bonds` p50/p99/mean in V4Only mode against the
//!   post-compaction brain. Compare to Phase 5's 311 us p50 baseline.
//!
//! The "reads during compaction" concurrency test lives in
//! `compaction::tests` (lib unit tests) instead of the bench because
//! `Brain` is not `Sync` -- the cargo test harness builds the
//! concurrency case via in-module `Mutex<Manifest>` snapshots directly.
//!
//! Output: Markdown report at `--out`.

use std::env;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

use lu4d::bond::{BondType, KeyframeSource, TemporalKeyframe};
use lu4d::brain::Brain;
use lu4d::read_path::ReadPathMode;
use lu4d::seed::Seed;

fn rand_seed(rng: &mut StdRng) -> Seed {
    let mut b = [0u8; 32];
    rng.fill(&mut b);
    Seed::from_bytes(b)
}

fn baseline_kf(ts: i64, w: f32) -> TemporalKeyframe {
    TemporalKeyframe { timestamp: ts, weight: w, source: KeyframeSource::Baseline }
}

fn pxx(sorted_ns: &[u128], p: f64) -> u128 {
    if sorted_ns.is_empty() { return 0; }
    let idx = ((sorted_ns.len() as f64) * p).floor() as usize;
    let idx = idx.min(sorted_ns.len() - 1);
    sorted_ns[idx]
}

fn main() -> std::io::Result<()> {
    let mut args = env::args().skip(1);
    let mut out_path: Option<PathBuf> = None;
    while let Some(a) = args.next() {
        if a == "--out" {
            out_path = args.next().map(PathBuf::from);
        }
    }
    let out_path = out_path.unwrap_or_else(|| PathBuf::from("/tmp/v4_phase6_bench.md"));

    let tmp = tempfile::tempdir().unwrap();
    let wal_path = tmp.path().join("wal.seg");
    let sst_dir = tmp.path().join("ssts");
    std::fs::create_dir_all(&sst_dir).unwrap();
    let manifest_path = tmp.path().join("MANIFEST");

    let mut brain = Brain::new();
    brain.enable_memtable(2 * 1024 * 1024 * 1024, &wal_path).unwrap();
    brain.enable_flush_worker(sst_dir.clone(), manifest_path.clone(), None).unwrap();
    let mt = brain.memtable().unwrap().clone();
    let intern = brain.flush_intern.as_ref().unwrap().clone();

    let mut rng = StdRng::seed_from_u64(0xCAFE_BABE);
    let n_subjects = 5_000usize;
    let bonds_per_l0 = 50_000usize;
    let n_l0 = 10usize;
    let mut subjects: Vec<Seed> = (0..n_subjects).map(|_| rand_seed(&mut rng)).collect();
    subjects.sort_by(|a, b| a.as_bytes().cmp(b.as_bytes()));

    eprintln!("[bench_compaction] filling {} L0 SSTs ({} bonds each)...", n_l0, bonds_per_l0);
    let fill_start = Instant::now();
    for round in 0..n_l0 {
        for i in 0..bonds_per_l0 {
            let sidx = rng.gen_range(0..subjects.len());
            let s = subjects[sidx];
            let o = rand_seed(&mut rng);
            mt.insert(s, o, BondType::IsA, 0.6, vec![baseline_kf((round * bonds_per_l0 + i) as i64, 0.6)], &intern);
        }
        brain.flush_now().unwrap();
    }
    let fill_elapsed = fill_start.elapsed();
    eprintln!("[bench_compaction] L0 fill took {:?}", fill_elapsed);

    let snap_pre = brain.manifest_snapshot().unwrap();
    let l0_count_pre = snap_pre.level_sst_count(0);
    let l0_bytes_pre: u64 = snap_pre.iter_ssts().filter(|(l, _)| *l == 0).map(|(_, e)| e.size).sum();
    let l0_total_bonds_pre: u64 = snap_pre.iter_ssts().filter(|(l, _)| *l == 0).map(|(_, e)| e.bond_count).sum();

    eprintln!("[bench_compaction] L0 pre-compaction: {} SSTs, {} bytes, {} bonds",
        l0_count_pre, l0_bytes_pre, l0_total_bonds_pre);

    // --- pre-compaction baseline: V4Only read latency on raw L0 stack ---
    let sample_size = 1000usize;
    let mut sample_rng = StdRng::seed_from_u64(0x1234_5678);
    let mut sample_subjects: Vec<Seed> = (0..sample_size).map(|_| {
        subjects[sample_rng.gen_range(0..subjects.len())]
    }).collect();
    use rand::seq::SliceRandom;
    sample_subjects.shuffle(&mut sample_rng);

    brain.set_read_path_mode(ReadPathMode::V4Only);
    // Warm.
    for s in &sample_subjects { let _ = brain.read_outgoing_bonds(s).unwrap(); }
    let mut samples_pre: Vec<u128> = Vec::with_capacity(sample_size);
    for s in &sample_subjects {
        let t = Instant::now();
        let _ = brain.read_outgoing_bonds(s).unwrap();
        samples_pre.push(t.elapsed().as_nanos());
    }
    samples_pre.sort_unstable();
    let p50_pre = pxx(&samples_pre, 0.5);
    let p99_pre = pxx(&samples_pre, 0.99);

    // --- Scenario A: compaction throughput ---
    brain.enable_compaction(None).unwrap();
    let comp_start = Instant::now();
    brain.compact_now().unwrap();
    let comp_elapsed = comp_start.elapsed();
    let throughput_mb_s = if comp_elapsed.as_secs_f64() > 0.0 {
        (l0_bytes_pre as f64 / 1024.0 / 1024.0) / comp_elapsed.as_secs_f64()
    } else { 0.0 };

    let snap_post = brain.manifest_snapshot().unwrap();
    let l0_count_post = snap_post.level_sst_count(0);
    let l1_count_post = snap_post.level_sst_count(1);
    let l1_bytes_post: u64 = snap_post.iter_ssts().filter(|(l, _)| *l == 1).map(|(_, e)| e.size).sum();
    let l1_total_bonds_post: u64 = snap_post.iter_ssts().filter(|(l, _)| *l == 1).map(|(_, e)| e.bond_count).sum();

    // --- Scenario B: read latency post-compaction ---
    // Warm again post-compaction.
    for s in &sample_subjects { let _ = brain.read_outgoing_bonds(s).unwrap(); }
    let mut samples_post: Vec<u128> = Vec::with_capacity(sample_size);
    for s in &sample_subjects {
        let t = Instant::now();
        let _ = brain.read_outgoing_bonds(s).unwrap();
        samples_post.push(t.elapsed().as_nanos());
    }
    samples_post.sort_unstable();
    let p50_post = pxx(&samples_post, 0.5);
    let p99_post = pxx(&samples_post, 0.99);
    let mean_post = samples_post.iter().sum::<u128>() / (samples_post.len() as u128).max(1);

    let mut out = File::create(&out_path)?;
    writeln!(out, "# OP-BRAIN-V4 Phase 6 -- compaction bench\n")?;
    writeln!(out, "## Scenario A: compaction throughput\n")?;
    writeln!(out, "| metric | value |")?;
    writeln!(out, "|---|---|")?;
    writeln!(out, "| L0 SSTs (pre) | {} |", l0_count_pre)?;
    writeln!(out, "| L0 bytes (pre) | {} |", l0_bytes_pre)?;
    writeln!(out, "| L0 bonds (pre) | {} |", l0_total_bonds_pre)?;
    writeln!(out, "| compaction wall-clock | {:?} |", comp_elapsed)?;
    writeln!(out, "| compaction throughput (MB/s) | {:.2} |", throughput_mb_s)?;
    writeln!(out, "| L0 SSTs (post) | {} |", l0_count_post)?;
    writeln!(out, "| L1 SSTs (post) | {} |", l1_count_post)?;
    writeln!(out, "| L1 bytes (post) | {} |", l1_bytes_post)?;
    writeln!(out, "| L1 bonds (post) | {} |", l1_total_bonds_post)?;
    writeln!(out)?;

    writeln!(out, "## Scenario B: read latency PRE vs POST compaction (V4Only)\n")?;
    writeln!(out, "| metric | pre (ns) | post (ns) | post/pre |")?;
    writeln!(out, "|---|---|---|---|")?;
    writeln!(out, "| p50 | {} | {} | {:.2}x |", p50_pre, p50_post,
        (p50_post as f64) / (p50_pre.max(1) as f64))?;
    writeln!(out, "| p99 | {} | {} | {:.2}x |", p99_pre, p99_post,
        (p99_post as f64) / (p99_pre.max(1) as f64))?;
    writeln!(out, "| mean (post) | -- | {} | -- |", mean_post)?;
    writeln!(out)?;
    writeln!(out, "Phase 5 baseline (V4Only p50): ~311_000 ns on a similar L0+MemTable stack.")?;
    writeln!(out, "Phase 6 target: p50 <= 10_000 ns after compaction folds L0 into one L1 SST.")?;

    eprintln!("[bench_compaction] wrote {}", out_path.display());
    Ok(())
}
