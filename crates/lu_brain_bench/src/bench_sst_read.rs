//! OP-BRAIN-V4 Phase 2 — SST read-path bench.
//!
//! Builds a 100 MiB SST with 10M random bonds in `/tmp/`, then
//! measures:
//!   - p50 / p99 `get_bonds_for_subject` latency on warm cache
//!   - p50 / p99 same on cold cache (BlockCache cleared between calls)
//!   - BlockCache hit rate on a locality-friendly access pattern
//!   - mmap-vs-decompress cost (single-block decompress time)
//!   - MB/s read throughput from `iter_bonds`
//!
//! The output Markdown file is written to the path passed via
//! `--out`. The test SST is removed before exit unless `--keep`.

use std::env;
use std::fs::{self, File};
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

use lu4d::bond::{BondType, KeyframeSource, TemporalKeyframe};
use lu4d::node::Confidence;
use lu4d::seed::Seed;
use lu4d::sst::{BlockCache, SST, DEFAULT_SST_BLOCK_CACHE_BYTES};
use lu4d::sst_writer::SSTWriter;

fn percentile_ns(samples: &mut [u128], pct: f64) -> f64 {
    if samples.is_empty() { return 0.0; }
    samples.sort_unstable();
    let k = ((samples.len() - 1) as f64 * pct).floor() as usize;
    samples[k] as f64
}

fn rand_seed(rng: &mut StdRng) -> Seed {
    let mut b = [0u8; 32];
    rng.fill(&mut b);
    Seed::from_bytes(b)
}

struct Args {
    out: PathBuf,
    keep: bool,
    bonds: usize,
    queries: usize,
}

fn parse_args() -> Args {
    let mut out = PathBuf::from("docs/v4_phase2_bench_2026-06-01.md");
    let mut keep = false;
    let mut bonds: usize = 10_000_000;
    let mut queries: usize = 5_000;
    let mut it = env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--out" => { out = PathBuf::from(it.next().expect("--out needs arg")); }
            "--keep" => { keep = true; }
            "--bonds" => { bonds = it.next().expect("--bonds").parse().expect("bonds u64"); }
            "--queries" => { queries = it.next().expect("--queries").parse().expect("queries u64"); }
            other => panic!("unknown arg: {}", other),
        }
    }
    Args { out, keep, bonds, queries }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args();
    eprintln!("OP-BRAIN-V4 Phase 2 — bench_sst_read");
    eprintln!("  bonds      = {}", args.bonds);
    eprintln!("  queries    = {}", args.queries);
    eprintln!("  out        = {}", args.out.display());

    let sst_path = PathBuf::from(format!("/tmp/lu_bench_phase2_{}.sst", std::process::id()));
    let _ = fs::remove_file(&sst_path);

    // ---------- Write ----------
    let write_start = Instant::now();
    let mut rng = StdRng::seed_from_u64(0xC0DE_C0DE_C0DE_C0DE);
    // Use a bounded subject pool so subjects are revisited across many
    // bonds (matches real-corpus hub-heavy distribution). 1/8 the bond
    // count is ~1.25M distinct subjects for the 10M-bond setting —
    // enough to keep the per-subject bond list shallow yet many subjects
    // per block.
    let subject_pool_size = (args.bonds / 8).max(1);
    let subject_pool: Vec<Seed> = (0..subject_pool_size)
        .map(|_| rand_seed(&mut rng))
        .collect();
    // Build a sorted list of (subject, object, bondtype, weight, kf).
    eprintln!("  generating bond stream...");
    let mut bonds: Vec<(Seed, Seed, BondType, f32, Option<TemporalKeyframe>)>
        = Vec::with_capacity(args.bonds);
    let bt_table = BondType::all();
    for i in 0..args.bonds {
        let s_idx = rng.gen_range(0..subject_pool.len());
        let s = subject_pool[s_idx];
        let o = rand_seed(&mut rng);
        let bt = bt_table[i % bt_table.len()];
        let w = rng.gen::<f32>();
        let kf = if i % 5 == 0 {
            Some(TemporalKeyframe {
                timestamp: i as i64,
                weight: w,
                source: KeyframeSource::Baseline,
            })
        } else { None };
        bonds.push((s, o, bt, w, kf));
    }
    bonds.sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));

    eprintln!("  writing SST...");
    let mut w = SSTWriter::create(&sst_path, 0, None)?;
    for (s, o, bt, wt, kf) in &bonds {
        let kfs: Vec<TemporalKeyframe> = kf.iter().cloned().collect();
        w.push_bond(s, o, *bt, *wt, Confidence::Verified, &kfs)?;
    }
    w.finalize()?;
    let write_elapsed = write_start.elapsed();
    let file_size = fs::metadata(&sst_path)?.len();
    eprintln!("  SST = {} MiB (wrote in {:.2}s)",
        file_size / (1024 * 1024), write_elapsed.as_secs_f64());

    // ---------- Open + measure ----------
    let cache = Arc::new(BlockCache::new(DEFAULT_SST_BLOCK_CACHE_BYTES));
    let open_start = Instant::now();
    let sst = SST::open(&sst_path, None, cache.clone())?;
    let open_ms = open_start.elapsed().as_secs_f64() * 1000.0;
    eprintln!("  opened, header.bond_count={} block_count={}",
        sst.header().bond_count, sst.block_count());
    eprintln!("  open + index load: {:.2} ms", open_ms);

    // ---------- Cold-cache p50/p99 lookups ----------
    eprintln!("  measuring cold-cache lookups...");
    let mut cold_samples: Vec<u128> = Vec::with_capacity(args.queries.min(1000));
    let mut rng2 = StdRng::seed_from_u64(0xFEED_FACE);
    for _ in 0..args.queries.min(1000) {
        let subj = subject_pool[rng2.gen_range(0..subject_pool.len())];
        // Fresh cache each iteration to isolate the cold path. This is
        // expensive — only run 1000 of them.
        let cold_cache = Arc::new(BlockCache::new(DEFAULT_SST_BLOCK_CACHE_BYTES));
        let cold_sst = SST::open(&sst_path, None, cold_cache)?;
        let t = Instant::now();
        let _v = cold_sst.get_bonds_for_subject(&subj)?;
        cold_samples.push(t.elapsed().as_nanos());
    }
    let cold_p50 = percentile_ns(&mut cold_samples, 0.50);
    let cold_p99 = percentile_ns(&mut cold_samples, 0.99);

    // ---------- Warm-cache p50/p99 lookups ----------
    eprintln!("  warming cache with sweep...");
    // Walk a slice of subjects first to populate cache.
    for subj in subject_pool.iter().take(50_000) {
        let _ = sst.get_bonds_for_subject(subj)?;
    }
    // Reset hit counters by reading cache stats before measurement.
    let (h0, m0) = cache.hit_miss_counts();
    eprintln!("  cache pre-warm: hits={} misses={}", h0, m0);

    eprintln!("  measuring warm-cache lookups ({} queries)...", args.queries);
    let mut warm_samples: Vec<u128> = Vec::with_capacity(args.queries);
    let mut rng3 = StdRng::seed_from_u64(0xBA5E_BA11);
    for _ in 0..args.queries {
        let subj = subject_pool[rng3.gen_range(0..subject_pool.len().min(50_000))];
        let t = Instant::now();
        let _v = sst.get_bonds_for_subject(&subj)?;
        warm_samples.push(t.elapsed().as_nanos());
    }
    let warm_p50 = percentile_ns(&mut warm_samples, 0.50);
    let warm_p99 = percentile_ns(&mut warm_samples, 0.99);
    let (h1, m1) = cache.hit_miss_counts();
    let measurement_hits = h1 - h0;
    let measurement_misses = m1 - m0;
    let warm_hit_rate = if measurement_hits + measurement_misses > 0 {
        (measurement_hits as f64) / ((measurement_hits + measurement_misses) as f64)
    } else { 0.0 };

    // ---------- Single-block decompress cost ----------
    eprintln!("  measuring single-block decompress...");
    let cache2 = Arc::new(BlockCache::new(DEFAULT_SST_BLOCK_CACHE_BYTES));
    let sst2 = SST::open(&sst_path, None, cache2)?;
    let mut block_samples: Vec<u128> = Vec::with_capacity(sst2.block_count().min(500));
    for bid in 0..sst2.block_count().min(500) {
        // Each iter uses a fresh cache so the call is truly a miss.
        let fresh = Arc::new(BlockCache::new(DEFAULT_SST_BLOCK_CACHE_BYTES));
        let s = SST::open(&sst_path, None, fresh)?;
        let t = Instant::now();
        let _b = s.decompress_block(bid as u32)?;
        block_samples.push(t.elapsed().as_nanos());
    }
    let blk_p50 = percentile_ns(&mut block_samples, 0.50);
    let blk_p99 = percentile_ns(&mut block_samples, 0.99);

    // ---------- Throughput via iter_bonds ----------
    eprintln!("  measuring iter_bonds throughput...");
    let fresh = Arc::new(BlockCache::new(DEFAULT_SST_BLOCK_CACHE_BYTES));
    let sst3 = SST::open(&sst_path, None, fresh)?;
    let t = Instant::now();
    let mut count = 0usize;
    for b in sst3.iter_bonds() {
        let _ = b?;
        count += 1;
    }
    let iter_elapsed = t.elapsed();
    let bytes_read = file_size as f64;
    let mbps = (bytes_read / (1024.0 * 1024.0)) / iter_elapsed.as_secs_f64();

    eprintln!("--- results ---");
    eprintln!("  cold-cache p50: {:.0} ns | p99: {:.0} ns", cold_p50, cold_p99);
    eprintln!("  warm-cache p50: {:.0} ns | p99: {:.0} ns", warm_p50, warm_p99);
    eprintln!("  warm hit rate: {:.1}%", warm_hit_rate * 100.0);
    eprintln!("  single-block decompress p50: {:.0} ns | p99: {:.0} ns", blk_p50, blk_p99);
    eprintln!("  iter_bonds: {} bonds in {:.2}s → {:.1} MB/s", count, iter_elapsed.as_secs_f64(), mbps);

    // ---------- Write the Markdown report ----------
    let mut out = String::new();
    use std::fmt::Write as _;
    let _ = writeln!(out, "# OP-BRAIN-V4 Phase 2 — `bench_sst_read` 2026-06-01");
    let _ = writeln!(out, "");
    let _ = writeln!(out, "SST read-path bench. Synthetic SST built in `/tmp/`, then queried.");
    let _ = writeln!(out, "");
    let _ = writeln!(out, "## Test corpus");
    let _ = writeln!(out, "");
    let _ = writeln!(out, "| Field | Value |");
    let _ = writeln!(out, "|---|---|");
    let _ = writeln!(out, "| Bonds                  | {} |", args.bonds);
    let _ = writeln!(out, "| Subject pool           | {} |", subject_pool_size);
    let _ = writeln!(out, "| SST size on disk       | {:.1} MiB |", file_size as f64 / (1024.0*1024.0));
    let _ = writeln!(out, "| Block count            | {} |", sst.block_count());
    let _ = writeln!(out, "| Write wall-clock       | {:.2} s |", write_elapsed.as_secs_f64());
    let _ = writeln!(out, "| Open + index load      | {:.2} ms |", open_ms);
    let _ = writeln!(out, "");
    let _ = writeln!(out, "## Lookup latency");
    let _ = writeln!(out, "");
    let _ = writeln!(out, "`SST::get_bonds_for_subject` measured over {} queries against a {}-entry subject pool.", args.queries, subject_pool_size);
    let _ = writeln!(out, "");
    let _ = writeln!(out, "| Scenario | p50 | p99 |");
    let _ = writeln!(out, "|---|---|---|");
    let _ = writeln!(out, "| Cold cache | {:.0} ns ({:.2} µs) | {:.0} ns ({:.2} µs) |",
        cold_p50, cold_p50 / 1000.0, cold_p99, cold_p99 / 1000.0);
    let _ = writeln!(out, "| Warm cache | {:.0} ns ({:.2} µs) | {:.0} ns ({:.2} µs) |",
        warm_p50, warm_p50 / 1000.0, warm_p99, warm_p99 / 1000.0);
    let _ = writeln!(out, "");
    let _ = writeln!(out, "## BlockCache hit rate");
    let _ = writeln!(out, "");
    let _ = writeln!(out, "| Counter | Value |");
    let _ = writeln!(out, "|---|---|");
    let _ = writeln!(out, "| Hits (measurement window)   | {} |", measurement_hits);
    let _ = writeln!(out, "| Misses (measurement window) | {} |", measurement_misses);
    let _ = writeln!(out, "| Hit rate                    | {:.1}% |", warm_hit_rate * 100.0);
    let _ = writeln!(out, "");
    let _ = writeln!(out, "## Cost breakdown");
    let _ = writeln!(out, "");
    let _ = writeln!(out, "| Op | p50 | p99 |");
    let _ = writeln!(out, "|---|---|---|");
    let _ = writeln!(out, "| Single block decompress (cache miss) | {:.0} ns | {:.0} ns |", blk_p50, blk_p99);
    let _ = writeln!(out, "");
    let _ = writeln!(out, "The mmap-vs-decompress ratio (warm vs cold p50) is {:.1}× — warm reads avoid the zstd decompress cost.", cold_p50 / warm_p50.max(1.0));
    let _ = writeln!(out, "");
    let _ = writeln!(out, "## Iteration throughput");
    let _ = writeln!(out, "");
    let _ = writeln!(out, "| Op | Value |");
    let _ = writeln!(out, "|---|---|");
    let _ = writeln!(out, "| `iter_bonds` walked         | {} bonds in {:.2} s |", count, iter_elapsed.as_secs_f64());
    let _ = writeln!(out, "| Throughput                  | {:.1} MB/s |", mbps);
    let _ = writeln!(out, "");

    if let Some(parent) = args.out.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let mut f = File::create(&args.out)?;
    f.write_all(out.as_bytes())?;
    eprintln!("  wrote {}", args.out.display());

    if !args.keep {
        let _ = fs::remove_file(&sst_path);
    }
    Ok(())
}
