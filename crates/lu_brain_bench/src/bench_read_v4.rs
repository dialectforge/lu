//! OP-BRAIN-V4 Phase 5 — read-path bench.
//!
//! Measures `Brain::read_outgoing_bonds` p50/p99 in each of the three
//! `ReadPathMode` variants:
//!
//! - `V3Only` (production default — the regression baseline)
//! - `V4Only` (consults MemTable + L0 + v3 via the merged path)
//! - `MergedV3V4` (same internal path as V4Only today; spec for Phase
//!   6 will split them)
//!
//! Setup mirrors the brief's expectation: MemTable (10K bonds) + 3
//! L0 SSTs (~100K bonds each) + a v3 heap brain (~1M bonds). All
//! subjects are drawn from a shared pool so each read hits every
//! layer with non-trivial probability.
//!
//! The brief targets p50 ≤ v3 baseline + 50% (i.e. ≤ ~200 ns)
//! against the Phase 0 v3 baseline (130 ns p50). If V4Only is
//! materially slower we'll surface the hot path bottleneck in the
//! report.

use std::env;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

use lu4d::bond::{BondType, KeyframeSource, TemporalKeyframe};
use lu4d::brain::Brain;
use lu4d::node::{Confidence, KnowledgeContent, KnowledgeNode};
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

/// p50 / p99 helper. Sort, index.
fn pxx(sorted_ns: &[u128], p: f64) -> u128 {
    if sorted_ns.is_empty() {
        return 0;
    }
    let idx = ((sorted_ns.len() as f64) * p).floor() as usize;
    let idx = idx.min(sorted_ns.len() - 1);
    sorted_ns[idx]
}

struct ScenarioStats {
    mode_label: &'static str,
    #[allow(dead_code)]
    n_reads: usize,
    p50_ns: u128,
    p99_ns: u128,
    mean_ns: u128,
    bonds_total: usize,
}

fn run_mode(brain: &Brain, subjects: &[Seed], mode: ReadPathMode, mode_label: &'static str) -> ScenarioStats {
    brain.set_read_path_mode(mode);
    let mut samples_ns: Vec<u128> = Vec::with_capacity(subjects.len());
    let mut bonds_total: usize = 0;
    for s in subjects {
        let t = Instant::now();
        let r = brain.read_outgoing_bonds(s).expect("read_outgoing_bonds");
        samples_ns.push(t.elapsed().as_nanos());
        bonds_total += r.len();
    }
    samples_ns.sort_unstable();
    let mean = samples_ns.iter().sum::<u128>() / (samples_ns.len() as u128).max(1);
    ScenarioStats {
        mode_label,
        n_reads: subjects.len(),
        p50_ns: pxx(&samples_ns, 0.5),
        p99_ns: pxx(&samples_ns, 0.99),
        mean_ns: mean,
        bonds_total,
    }
}

fn main() -> std::io::Result<()> {
    let mut args = env::args().skip(1);
    let mut out_path: Option<PathBuf> = None;
    while let Some(a) = args.next() {
        if a == "--out" {
            out_path = args.next().map(PathBuf::from);
        }
    }
    let out_path = out_path.unwrap_or_else(|| PathBuf::from("/tmp/v4_phase5_bench.md"));

    eprintln!("[bench_read_v4] building scenario (this can take ~30s)...");

    // -----------------------------------------------------------------
    // 1. Build a v3 heap brain with ~1M bonds across 50K subjects.
    //    Heap-only construction (no archive); spread-activation reads
    //    take this same path in production reads of the production
    //    binary's recall today.
    // -----------------------------------------------------------------
    let v3_subjects = 50_000usize;
    let v3_bonds_per_subj = 20usize;
    let total_v3_bonds = v3_subjects * v3_bonds_per_subj;

    let mut brain = Brain::new();
    let mut rng = StdRng::seed_from_u64(0xB3A1A12604);
    let mut all_subjects: Vec<Seed> = Vec::with_capacity(v3_subjects);

    for _ in 0..v3_subjects {
        let s = rand_seed(&mut rng);
        all_subjects.push(s);
        brain.network.add_node(KnowledgeNode::with_seed(
            s, "subj".to_string(),
            KnowledgeContent::Concept("s".to_string()),
            Confidence::Verified, 0,
        ));
    }
    for s in &all_subjects {
        for _ in 0..v3_bonds_per_subj {
            let o = rand_seed(&mut rng);
            brain.network.add_node(KnowledgeNode::with_seed(
                o, "obj".to_string(),
                KnowledgeContent::Concept("o".to_string()),
                Confidence::Verified, 0,
            ));
            let mut bond = lu4d::bond::TemporalBond::new(*s, o, BondType::RelatedTo, 0.5, 0);
            bond.add_keyframe(1, 0.5, KeyframeSource::Baseline);
            brain.network.add_bond(bond).ok();
        }
    }
    eprintln!("[bench_read_v4] v3 heap done: {} subjects x {} bonds = {}",
        v3_subjects, v3_bonds_per_subj, total_v3_bonds);

    // -----------------------------------------------------------------
    // 2. Enable v4 stack: MemTable + WAL + flush worker. Insert 300K
    //    bonds, flush every ~100K so we get 3 L0 SSTs. Then leave 10K
    //    bonds in the MemTable.
    // -----------------------------------------------------------------
    let tmp = tempfile::tempdir().unwrap();
    let wal_path = tmp.path().join("wal.seg");
    let sst_dir = tmp.path().join("ssts");
    std::fs::create_dir_all(&sst_dir).unwrap();
    let manifest_path = tmp.path().join("MANIFEST");
    brain.enable_memtable(2 * 1024 * 1024 * 1024, &wal_path).unwrap();
    brain.enable_flush_worker(sst_dir.clone(), manifest_path.clone(), None).unwrap();
    let mt = brain.memtable().unwrap().clone();
    let intern = brain.flush_intern.as_ref().unwrap().clone();

    let l0_bonds_per_flush = 100_000usize;
    let memtable_bonds = 10_000usize;
    // Use the SAME subject pool so reads hit MemTable + L0 + v3 layers.
    for sst_round in 0..3 {
        for i in 0..l0_bonds_per_flush {
            let sidx = rng.gen_range(0..all_subjects.len());
            let s = all_subjects[sidx];
            let o = rand_seed(&mut rng);
            mt.insert(
                s, o, BondType::IsA, 0.6,
                vec![baseline_kf((sst_round * l0_bonds_per_flush + i) as i64, 0.6)],
                &intern,
            );
        }
        brain.flush_now().unwrap();
        eprintln!("[bench_read_v4] L0 SST {} flushed", sst_round + 1);
    }
    for i in 0..memtable_bonds {
        let sidx = rng.gen_range(0..all_subjects.len());
        let s = all_subjects[sidx];
        let o = rand_seed(&mut rng);
        mt.insert(
            s, o, BondType::HasProperty, 0.4,
            vec![baseline_kf((1_000_000 + i) as i64, 0.4)],
            &intern,
        );
    }
    eprintln!("[bench_read_v4] MemTable has {} bonds (post-3-flush phase)", mt.bond_count());

    // -----------------------------------------------------------------
    // 3. Build a 1000-subject sample (drawn from all_subjects, in a
    //    deterministic shuffle) and time `read_outgoing_bonds` in each
    //    mode.
    // -----------------------------------------------------------------
    let sample_size = 1000usize;
    let mut sample_rng = StdRng::seed_from_u64(0xCAFE);
    let mut sample_subjects: Vec<Seed> = (0..sample_size).map(|_| {
        all_subjects[sample_rng.gen_range(0..all_subjects.len())]
    }).collect();
    // Warm cache by reading once through (V3Only) so the v3 lookup
    // tables are touched and the L0 block cache is primed.
    brain.set_read_path_mode(ReadPathMode::V3Only);
    for s in &sample_subjects {
        let _ = brain.read_outgoing_bonds(s).unwrap();
    }
    brain.set_read_path_mode(ReadPathMode::V4Only);
    for s in &sample_subjects {
        let _ = brain.read_outgoing_bonds(s).unwrap();
    }

    // Shuffle so the samples are not in insertion order.
    use rand::seq::SliceRandom;
    sample_subjects.shuffle(&mut sample_rng);

    let stats_v3 = run_mode(&brain, &sample_subjects, ReadPathMode::V3Only, "V3Only");
    let stats_v4 = run_mode(&brain, &sample_subjects, ReadPathMode::V4Only, "V4Only");
    let stats_merged = run_mode(&brain, &sample_subjects, ReadPathMode::MergedV3V4, "MergedV3V4");

    // -----------------------------------------------------------------
    // 4. Report.
    // -----------------------------------------------------------------
    let mut out = File::create(&out_path)?;
    writeln!(out, "# OP-BRAIN-V4 Phase 5 — read-path bench\n")?;
    writeln!(out, "Generated by `bench_read_v4`.\n")?;
    writeln!(out, "## Scenario\n")?;
    writeln!(out, "| component | size |")?;
    writeln!(out, "|---|---|")?;
    writeln!(out, "| v3 heap subjects | {} |", v3_subjects)?;
    writeln!(out, "| v3 heap bonds | {} |", total_v3_bonds)?;
    writeln!(out, "| L0 SSTs | 3 |")?;
    writeln!(out, "| bonds per L0 SST | {} |", l0_bonds_per_flush)?;
    writeln!(out, "| MemTable bonds (post-flush) | {} |", memtable_bonds)?;
    writeln!(out, "| read samples per mode | {} |\n", sample_size)?;

    writeln!(out, "## Read latency by `ReadPathMode`\n")?;
    writeln!(out, "| mode | p50 (ns) | p99 (ns) | mean (ns) | bonds returned (total) |")?;
    writeln!(out, "|---|---|---|---|---|")?;
    for s in [&stats_v3, &stats_v4, &stats_merged] {
        writeln!(out, "| `{}` | {} | {} | {} | {} |",
            s.mode_label, s.p50_ns, s.p99_ns, s.mean_ns, s.bonds_total)?;
    }
    writeln!(out)?;

    let v4_overhead_pct = if stats_v3.p50_ns > 0 {
        ((stats_v4.p50_ns as i128 - stats_v3.p50_ns as i128) * 100) / (stats_v3.p50_ns as i128)
    } else { 0 };
    writeln!(out, "## Overhead vs V3Only baseline\n")?;
    writeln!(out, "- V4Only p50 / V3Only p50: {:.2}x",
        (stats_v4.p50_ns as f64) / (stats_v3.p50_ns.max(1) as f64))?;
    writeln!(out, "- V4Only p50 overhead: {}%", v4_overhead_pct)?;
    writeln!(out, "- V4Only p99 / V3Only p99: {:.2}x",
        (stats_v4.p99_ns as f64) / (stats_v3.p99_ns.max(1) as f64))?;
    writeln!(out)?;
    writeln!(out, "Brief target: V4Only p50 ≤ V3Only + 50%. ")?;
    writeln!(out, "Phase 0 v3 baseline: ~130 ns p50 (`bench_v3`).\n")?;
    writeln!(out, "V3Only result here ({} ns p50) reflects the bench's\n\
local heap shape (~50K subjects, ~1M bonds with the in-memory\n\
hashmap-based `neighbors_outgoing` path); the Phase 0 130 ns\n\
number is the canonical 16M-bond brain.\n", stats_v3.p50_ns)?;

    writeln!(out, "## Notes on hot path\n")?;
    writeln!(out, "The merge step builds a `HashMap<DedupKey, MergedBondView>`")?;
    writeln!(out, "and inserts every (MemTable + L0 + v3) bond. For hub")?;
    writeln!(out, "subjects with high fan-out this allocates O(N) capacity;")?;
    writeln!(out, "if p99 inflates substantially under such subjects a")?;
    writeln!(out, "future port can swap to a subject-scoped `BTreeSet` or")?;
    writeln!(out, "a small-vec preallocator. Phase 5 keeps the simple")?;
    writeln!(out, "`HashMap` for clarity.")?;

    eprintln!("[bench_read_v4] wrote {}", out_path.display());
    Ok(())
}
