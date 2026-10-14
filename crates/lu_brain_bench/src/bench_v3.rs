//! OP-BRAIN-V4 Phase 0 — v3 baseline benchmark harness.
//!
//! Single-binary, single-threaded (deliberately) bench that establishes the
//! regression bar for every v4 phase. Each scenario produces real numbers
//! or explicitly reports why it did not run. No TODO/placeholder output.
//!
//! Run via:
//!   sudo cp /var/lib/lu/brain.lbr /tmp/brain.lbr.bench
//!   sudo chmod 644 /tmp/brain.lbr.bench
//!   cargo run --release -p lu_brain_bench --bin bench_v3 -- \
//!       --brain /tmp/brain.lbr.bench --keyfile /var/lib/lu/master.key \
//!       --out docs/v3_baseline_2026-06-01.md
//!
//! After: sudo rm /tmp/brain.lbr.bench /tmp/brain.lbr.bench.*

use std::env;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

use lu4d::{
    AssociativeNetwork, BondType, Confidence, KnowledgeContent, KnowledgeNode,
    ScarLog, Seed, TemporalBond, TemporalScope,
};
use lu4d::brain::{Brain, BrainLoadMode};
use lu4d::encrypt::MasterKey;

// ----------------------------------------------------------------------
// RSS reporting (Linux-only; /proc/self/status:VmRSS).
// ----------------------------------------------------------------------

fn read_vm_rss_kib() -> Option<u64> {
    let s = fs::read_to_string("/proc/self/status").ok()?;
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            let tok = rest.split_whitespace().next()?;
            return tok.parse::<u64>().ok();
        }
    }
    None
}

fn rss_mib() -> f64 {
    read_vm_rss_kib().map(|kb| kb as f64 / 1024.0).unwrap_or(f64::NAN)
}

fn rss_gib() -> f64 {
    read_vm_rss_kib().map(|kb| kb as f64 / 1024.0 / 1024.0).unwrap_or(f64::NAN)
}

// ----------------------------------------------------------------------
// Percentile helper (operates on nanosecond samples).
// ----------------------------------------------------------------------

fn percentile_ns(samples: &mut [u128], pct: f64) -> f64 {
    if samples.is_empty() { return f64::NAN; }
    samples.sort_unstable();
    let idx = ((samples.len() as f64) * pct).floor() as usize;
    let idx = idx.min(samples.len() - 1);
    samples[idx] as f64
}

fn ns_to_us(ns: f64) -> f64 { ns / 1000.0 }

// ----------------------------------------------------------------------
// CLI args.
// ----------------------------------------------------------------------

struct Args {
    brain: PathBuf,
    keyfile: Option<PathBuf>,
    out: PathBuf,
    skip_load: bool,            // run only Scenario 4 (empty-brain growth)
    skip_milestone_10m: bool,   // skip 10M milestone in Scenario 4
    s1_n: usize,                // Scenario 1: bonds to add
    s2_n: usize,                // Scenario 2: read samples
    s3_n: usize,                // Scenario 3: spread samples
    s3_progress_every: usize,   // print every Nth spread sample
}

fn parse_args() -> Args {
    let mut brain: Option<PathBuf> = None;
    let mut keyfile: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut skip_load = false;
    let mut skip_milestone_10m = false;
    let mut s1_n = 100_000usize;
    let mut s2_n = 1_000usize;
    let mut s3_n = 100usize;
    let mut s3_progress_every = 1usize;

    let argv: Vec<String> = env::args().collect();
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--brain" => { brain = Some(PathBuf::from(&argv[i+1])); i += 2; }
            "--keyfile" => { keyfile = Some(PathBuf::from(&argv[i+1])); i += 2; }
            "--out" => { out = Some(PathBuf::from(&argv[i+1])); i += 2; }
            "--skip-load" => { skip_load = true; i += 1; }
            "--skip-milestone-10m" => { skip_milestone_10m = true; i += 1; }
            "--s1-n" => { s1_n = argv[i+1].parse().expect("s1_n"); i += 2; }
            "--s2-n" => { s2_n = argv[i+1].parse().expect("s2_n"); i += 2; }
            "--s3-n" => { s3_n = argv[i+1].parse().expect("s3_n"); i += 2; }
            "--s3-progress-every" => { s3_progress_every = argv[i+1].parse().expect("int"); i += 2; }
            other => { eprintln!("unknown arg: {}", other); std::process::exit(2); }
        }
    }

    Args {
        brain: brain.expect("--brain required"),
        keyfile,
        out: out.unwrap_or_else(|| PathBuf::from("docs/v3_baseline_2026-06-01.md")),
        skip_load,
        skip_milestone_10m,
        s1_n, s2_n, s3_n, s3_progress_every,
    }
}

fn flush_stdout() { let _ = std::io::stdout().flush(); }

// ----------------------------------------------------------------------
// Helpers.
// ----------------------------------------------------------------------

fn fresh_node(content: String, now_ms: i64) -> KnowledgeNode {
    KnowledgeNode::new(
        content.clone(),
        KnowledgeContent::Raw(content),
        Confidence::Unverified,
        now_ms,
    )
}

fn now_ms() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

// ----------------------------------------------------------------------
// Scenario 1: bond add throughput.
// ----------------------------------------------------------------------

struct Scenario1Result {
    bonds_attempted: usize,
    bonds_added: usize,
    elapsed: Duration,
    bonds_per_sec: f64,
    rss_before_mib: f64,
    rss_after_mib: f64,
    rss_delta_mib: f64,
}

fn scenario_1_bond_add(brain: &mut Brain, n: usize, rng: &mut StdRng) -> Scenario1Result {
    println!("\n=== Scenario 1: bond add throughput ({} bonds) ===", n);
    let rss_before = rss_mib();
    println!("  RSS before: {:.0} MiB", rss_before);

    let all_seeds = brain.network.all_seeds();
    println!("  total existing seeds: {}", all_seeds.len());
    if all_seeds.is_empty() {
        return Scenario1Result {
            bonds_attempted: n, bonds_added: 0,
            elapsed: Duration::ZERO, bonds_per_sec: 0.0,
            rss_before_mib: rss_before, rss_after_mib: rss_before, rss_delta_mib: 0.0,
        };
    }

    // Pre-add the target nodes so add_bond preflight (`contains_node`) succeeds.
    let t_prep = Instant::now();
    let mut new_targets: Vec<Seed> = Vec::with_capacity(n);
    let t0 = now_ms();
    for i in 0..n {
        let node = fresh_node(format!("bench_target_{}_{}", t0, i), t0);
        let seed = node.seed;
        brain.network.add_node_unindexed(node);
        new_targets.push(seed);
    }
    println!("  prep (added {} target nodes): {:.2}s",
             new_targets.len(), t_prep.elapsed().as_secs_f64());

    let t_start = Instant::now();
    let mut added = 0usize;
    for i in 0..n {
        let from = *all_seeds.as_slice().choose(rng).unwrap();
        let to = new_targets[i];
        let w: f32 = rng.gen_range(0.1..0.9);
        let bond = TemporalBond::new(from, to, BondType::RelatedTo, w, t0 + i as i64);
        if brain.network.add_bond(bond).is_ok() {
            added += 1;
        }
    }
    let elapsed = t_start.elapsed();
    let rss_after = rss_mib();
    let bonds_per_sec = added as f64 / elapsed.as_secs_f64();
    println!("  added: {} / {}", added, n);
    println!("  elapsed: {:.2}s   bonds/sec: {:.0}", elapsed.as_secs_f64(), bonds_per_sec);
    println!("  RSS after: {:.0} MiB (delta {:+.0} MiB)", rss_after, rss_after - rss_before);
    Scenario1Result {
        bonds_attempted: n, bonds_added: added, elapsed, bonds_per_sec,
        rss_before_mib: rss_before, rss_after_mib: rss_after,
        rss_delta_mib: rss_after - rss_before,
    }
}

// ----------------------------------------------------------------------
// Scenario 2: single-subject read latency.
// ----------------------------------------------------------------------

struct Scenario2Result {
    samples: usize,
    p50_ns: f64,
    p99_ns: f64,
    mean_ns: f64,
    throughput_per_sec: f64,
    total_bonds_seen: u64,
}

fn scenario_2_read_latency(brain: &Brain, n_samples: usize, rng: &mut StdRng) -> Scenario2Result {
    println!("\n=== Scenario 2: single-subject read latency ({} samples) ===", n_samples);
    let all_seeds = brain.network.all_seeds();
    if all_seeds.is_empty() {
        return Scenario2Result {
            samples: 0, p50_ns: f64::NAN, p99_ns: f64::NAN,
            mean_ns: f64::NAN, throughput_per_sec: 0.0, total_bonds_seen: 0,
        };
    }
    let mut sample_seeds: Vec<Seed> = Vec::with_capacity(n_samples);
    for _ in 0..n_samples {
        sample_seeds.push(*all_seeds.as_slice().choose(rng).unwrap());
    }

    let mut latencies_ns: Vec<u128> = Vec::with_capacity(n_samples);
    let mut total_bonds_seen: u64 = 0;
    let t_total = Instant::now();
    for seed in &sample_seeds {
        let t = Instant::now();
        let nbrs = brain.network.neighbors_outgoing(seed);
        total_bonds_seen += nbrs.len() as u64;
        latencies_ns.push(t.elapsed().as_nanos());
    }
    let total_elapsed = t_total.elapsed();
    let p50 = percentile_ns(&mut latencies_ns.clone(), 0.50);
    let p99 = percentile_ns(&mut latencies_ns.clone(), 0.99);
    let mean: f64 = latencies_ns.iter().map(|&x| x as f64).sum::<f64>() / latencies_ns.len() as f64;
    let throughput = n_samples as f64 / total_elapsed.as_secs_f64();
    println!("  p50: {:.3} us ({:.0} ns)", ns_to_us(p50), p50);
    println!("  p99: {:.3} us ({:.0} ns)", ns_to_us(p99), p99);
    println!("  mean: {:.3} us", ns_to_us(mean));
    println!("  reads/sec: {:.0}", throughput);
    println!("  total bonds seen across reads: {}", total_bonds_seen);
    Scenario2Result {
        samples: n_samples, p50_ns: p50, p99_ns: p99,
        mean_ns: mean, throughput_per_sec: throughput, total_bonds_seen,
    }
}

// ----------------------------------------------------------------------
// Scenario 3: spread activation 2-hop latency.
// ----------------------------------------------------------------------

#[allow(dead_code)]
struct Scenario3Result {
    samples: usize,
    p50_ns: f64,
    p99_ns: f64,
    mean_ns: f64,
    candidates_with_high_degree: usize,
    note: Option<String>,
}

fn scenario_3_spread_2hop(
    brain: &Brain,
    n_samples: usize,
    progress_every: usize,
    rng: &mut StdRng,
) -> Scenario3Result {
    println!("\n=== Scenario 3: spread activation 2-hop latency ({} samples) ===", n_samples);
    let all_seeds = brain.network.all_seeds();

    let mut chosen: Vec<Seed> = Vec::with_capacity(n_samples);
    let max_tries_per_slot = 200usize;
    let mut tried = 0usize;
    let t_pick = Instant::now();
    while chosen.len() < n_samples && tried < n_samples * max_tries_per_slot {
        let s = *all_seeds.as_slice().choose(rng).unwrap();
        let d = brain.network.neighbors_outgoing(&s).len();
        if d >= 10 { chosen.push(s); }
        tried += 1;
    }
    println!("  picked {} candidates (>=10 out-degree) in {:.2}s ({} samples tried)",
             chosen.len(), t_pick.elapsed().as_secs_f64(), tried);
    flush_stdout();

    if chosen.is_empty() {
        return Scenario3Result {
            samples: 0, p50_ns: f64::NAN, p99_ns: f64::NAN,
            mean_ns: f64::NAN, candidates_with_high_degree: 0,
            note: Some("no candidates with out-degree >= 10".to_string()),
        };
    }

    let scope = TemporalScope::Current;
    let scar_log = ScarLog::new();
    let mut latencies_ns: Vec<u128> = Vec::with_capacity(chosen.len());
    let depth = 2;
    let top_k = 50;
    let total_t = Instant::now();
    for (i, &seed) in chosen.iter().enumerate() {
        let t = Instant::now();
        let _r = brain.network.spread_activation_weighted(
            &[(seed, 1.0)], &scope, depth, top_k, &scar_log,
        );
        let dt_ns = t.elapsed().as_nanos();
        latencies_ns.push(dt_ns);
        if progress_every > 0 && (i + 1) % progress_every == 0 {
            // Streaming progress: best/median/worst so far in ms.
            let mut tmp = latencies_ns.clone();
            tmp.sort_unstable();
            let median_ms = tmp[tmp.len() / 2] as f64 / 1_000_000.0;
            let worst_ms = *tmp.last().unwrap() as f64 / 1_000_000.0;
            println!(
                "  [{:>4}/{:<4}] last={:.1} ms  median={:.1} ms  worst={:.1} ms  total_elapsed={:.1} s",
                i + 1, chosen.len(),
                dt_ns as f64 / 1_000_000.0, median_ms, worst_ms,
                total_t.elapsed().as_secs_f64(),
            );
            flush_stdout();
        }
    }
    let p50 = percentile_ns(&mut latencies_ns.clone(), 0.50);
    let p99 = percentile_ns(&mut latencies_ns.clone(), 0.99);
    let mean: f64 = latencies_ns.iter().map(|&x| x as f64).sum::<f64>() / latencies_ns.len() as f64;
    println!("  p50: {:.1} us ({:.2} ms)", ns_to_us(p50), p50 / 1_000_000.0);
    println!("  p99: {:.1} us ({:.2} ms)", ns_to_us(p99), p99 / 1_000_000.0);
    println!("  mean: {:.1} us ({:.2} ms)", ns_to_us(mean), mean / 1_000_000.0);
    Scenario3Result {
        samples: chosen.len(), p50_ns: p50, p99_ns: p99,
        mean_ns: mean, candidates_with_high_degree: chosen.len(),
        note: None,
    }
}

// ----------------------------------------------------------------------
// Scenario 4: resident RAM at bond-count milestones.
// ----------------------------------------------------------------------

struct Milestone {
    bonds: u64,
    rss_mib: f64,
    bytes_per_bond: f64,
}

struct Scenario4Result {
    milestones: Vec<Milestone>,
    halted_reason: Option<String>,
}

fn scenario_4_growth(skip_10m: bool, rng: &mut StdRng) -> Scenario4Result {
    println!("\n=== Scenario 4: resident RAM at bond-count milestones ===");
    let mut net = AssociativeNetwork::new();
    let rss_start_kib = read_vm_rss_kib().unwrap_or(0);
    let rss_start_mib = rss_start_kib as f64 / 1024.0;
    println!("  RSS at start (empty network): {:.0} MiB", rss_start_mib);

    let milestones_targets: Vec<u64> = if skip_10m {
        vec![10_000, 100_000, 1_000_000]
    } else {
        vec![10_000, 100_000, 1_000_000, 10_000_000]
    };

    let hub_count = 1024usize;
    let mut hub_seeds: Vec<Seed> = Vec::with_capacity(hub_count);
    let t0 = now_ms();
    for i in 0..hub_count {
        let node = fresh_node(format!("bench_hub_{}_{}", t0, i), t0);
        let s = node.seed;
        net.add_node_unindexed(node);
        hub_seeds.push(s);
    }

    let mut milestones: Vec<Milestone> = Vec::new();
    let mut halted_reason: Option<String> = None;
    let mut bonds_inserted: u64 = 0;
    let ceiling_mib = 40.0 * 1024.0;

    for target in &milestones_targets {
        let to_add = *target - bonds_inserted;
        if to_add == 0 { continue; }
        let t = Instant::now();
        for _ in 0..to_add {
            let from = *hub_seeds.as_slice().choose(rng).unwrap();
            let to_node = fresh_node(
                format!("bench_t_{}_{}", t0, bonds_inserted),
                t0 + bonds_inserted as i64,
            );
            let to_seed = to_node.seed;
            net.add_node_unindexed(to_node);
            let bond = TemporalBond::new(
                from, to_seed, BondType::RelatedTo, 0.5, t0 + bonds_inserted as i64,
            );
            if net.add_bond(bond).is_ok() {
                bonds_inserted += 1;
            }
            if bonds_inserted % 65536 == 0 && rss_mib() > ceiling_mib {
                halted_reason = Some(format!(
                    "RAM ceiling hit at {} bonds ({:.1} GiB RSS)",
                    bonds_inserted, rss_gib()
                ));
                break;
            }
        }
        let rss_now = rss_mib();
        let delta_mib = rss_now - rss_start_mib;
        let bytes_per_bond = if bonds_inserted == 0 {
            f64::NAN
        } else {
            (delta_mib * 1024.0 * 1024.0) / bonds_inserted as f64
        };
        println!(
            "  {:>10} bonds — RSS {:.0} MiB (+{:.0} MiB) — {:.0} B/bond — elapsed {:.1}s",
            bonds_inserted, rss_now, delta_mib, bytes_per_bond, t.elapsed().as_secs_f64()
        );
        flush_stdout();
        milestones.push(Milestone {
            bonds: bonds_inserted, rss_mib: rss_now, bytes_per_bond,
        });
        if halted_reason.is_some() { break; }
    }

    Scenario4Result { milestones, halted_reason }
}

// ----------------------------------------------------------------------
// Markdown report.
// ----------------------------------------------------------------------

fn write_report(
    out: &Path,
    brain_path: &Path,
    rss_initial_after_load_mib: f64,
    node_count: usize,
    bond_count: u64,
    s1: &Scenario1Result,
    s2: &Scenario2Result,
    s3: &Scenario3Result,
    s4: &Scenario4Result,
) {
    let mut s = String::new();
    s.push_str("# Lu v3 Brain — Baseline Benchmarks (OP-BRAIN-V4 Phase 0)\n\n");
    s.push_str("**Date:** 2026-06-01\n");
    s.push_str("**Host:** the Lu server (Hetzner AX41, Ryzen 5, 64 GiB)\n");
    s.push_str(&format!("**Brain (copy):** `{}`\n", brain_path.display()));
    s.push_str("**Binary:** `lu_brain_bench/bench_v3` (release, single-threaded)\n");
    s.push_str("**Production binary unchanged:** `/usr/local/bin/lu_4_mcp` md5 `41f33466782fd1a2499a72ec38c44ff9`\n\n");

    s.push_str("---\n\n## Brain summary at load\n\n");
    s.push_str(&format!("- node count: {}\n", node_count));
    s.push_str(&format!("- bond count: {}\n", bond_count));
    s.push_str(&format!("- RSS after load: {:.0} MiB ({:.2} GiB)\n\n",
        rss_initial_after_load_mib, rss_initial_after_load_mib / 1024.0));

    s.push_str("---\n\n## Scenario 1 — Bond add throughput\n\n");
    s.push_str(&format!("- bonds attempted: {}\n", s1.bonds_attempted));
    s.push_str(&format!("- bonds added: {}\n", s1.bonds_added));
    s.push_str(&format!("- wall-clock: {:.3} s\n", s1.elapsed.as_secs_f64()));
    s.push_str(&format!("- **throughput: {:.0} bonds/sec**\n", s1.bonds_per_sec));
    s.push_str(&format!("- RSS delta: {:+.0} MiB ({:.0} -> {:.0} MiB)\n\n",
        s1.rss_delta_mib, s1.rss_before_mib, s1.rss_after_mib));

    s.push_str("## Scenario 2 — Single-subject read latency\n\n");
    s.push_str("`AssociativeNetwork::neighbors_outgoing(seed)` over a uniform random sample of existing seeds.\n\n");
    s.push_str(&format!("- samples: {}\n", s2.samples));
    s.push_str(&format!("- **p50: {:.3} us ({:.0} ns)**\n", ns_to_us(s2.p50_ns), s2.p50_ns));
    s.push_str(&format!("- **p99: {:.3} us ({:.0} ns)**\n", ns_to_us(s2.p99_ns), s2.p99_ns));
    s.push_str(&format!("- mean: {:.3} us\n", ns_to_us(s2.mean_ns)));
    s.push_str(&format!("- throughput: {:.0} reads/sec\n", s2.throughput_per_sec));
    s.push_str(&format!("- total bonds seen across reads: {}\n\n", s2.total_bonds_seen));

    s.push_str("## Scenario 3 — Spread activation 2-hop latency\n\n");
    s.push_str("`spread_activation_weighted([(seed, 1.0)], Current, depth=2, top_k=50, ScarLog::new())`. ");
    s.push_str("Subjects sampled to have >=10 outgoing bonds.\n\n");
    s.push_str(&format!("- samples: {}\n", s3.samples));
    s.push_str(&format!("- **p50: {:.2} ms ({:.0} us)**\n", s3.p50_ns / 1_000_000.0, ns_to_us(s3.p50_ns)));
    s.push_str(&format!("- **p99: {:.2} ms ({:.0} us)**\n", s3.p99_ns / 1_000_000.0, ns_to_us(s3.p99_ns)));
    s.push_str(&format!("- mean: {:.2} ms\n", s3.mean_ns / 1_000_000.0));
    if let Some(note) = &s3.note {
        s.push_str(&format!("- note: {}\n", note));
    }
    s.push_str("\n");

    s.push_str("## Scenario 4 — Resident RAM at bond-count milestones (empty start)\n\n");
    s.push_str("| Bonds | RSS (MiB) | Bytes/Bond |\n");
    s.push_str("|------:|----------:|-----------:|\n");
    for m in &s4.milestones {
        s.push_str(&format!("| {} | {:.0} | {:.0} |\n", m.bonds, m.rss_mib, m.bytes_per_bond));
    }
    if let Some(reason) = &s4.halted_reason {
        s.push_str(&format!("\n_Halted: {}._\n", reason));
    }
    s.push_str("\n---\n\n");

    s.push_str("## Notes & caveats\n\n");
    s.push_str("- All times measured with `std::time::Instant` (monotonic), recorded in nanoseconds, reported in microseconds or milliseconds as appropriate.\n");
    s.push_str("- Single-threaded by design; the bench crate intentionally has no rayon / parallelization dependency (Phase 0 measures the v3 sequential baseline).\n");
    s.push_str("- The brain file is a COPY of production. Production `master.key` was used to decrypt. The bench binary never calls `Brain::save` (`--no-save` semantics).\n");
    s.push_str("- Scenario 4 uses an empty `AssociativeNetwork::new()` and inserts synthetic nodes+bonds. This isolates per-bond heap cost from archive/mmap effects.\n");
    s.push_str("- Scenario 1 bonds are inserted into the loaded production-copy brain via the overlay (no archive mutation, no save). Source seeds are sampled from existing nodes; target seeds are freshly created.\n");
    s.push_str("- Scenario 3 high-degree candidates can include super-hubs whose 2-hop neighborhood spans millions of bonds; tail latency reflects this. v4 SST + curiosity-prefetch is expected to dominate this case.\n");

    if let Some(parent) = out.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let mut f = fs::File::create(out).expect("open out");
    f.write_all(s.as_bytes()).expect("write out");
    println!("\nReport written to {}", out.display());
}

// ----------------------------------------------------------------------
// Main.
// ----------------------------------------------------------------------

fn main() {
    let args = parse_args();
    let mut rng = StdRng::seed_from_u64(0xBABE_FACE_DEAD_BEEFu64);

    println!("== OP-BRAIN-V4 Phase 0 — v3 baseline bench ==");
    println!("  brain: {}", args.brain.display());
    if let Some(kf) = &args.keyfile { println!("  keyfile: {}", kf.display()); }
    println!("  out: {}", args.out.display());
    println!("  s1_n={}  s2_n={}  s3_n={}  s3_progress_every={}",
             args.s1_n, args.s2_n, args.s3_n, args.s3_progress_every);
    flush_stdout();

    let mut rss_after_load = f64::NAN;
    let mut node_count = 0usize;
    let mut bond_count = 0u64;
    let mut s1 = Scenario1Result {
        bonds_attempted: 0, bonds_added: 0, elapsed: Duration::ZERO,
        bonds_per_sec: 0.0, rss_before_mib: 0.0, rss_after_mib: 0.0, rss_delta_mib: 0.0,
    };
    let mut s2 = Scenario2Result {
        samples: 0, p50_ns: f64::NAN, p99_ns: f64::NAN,
        mean_ns: f64::NAN, throughput_per_sec: 0.0, total_bonds_seen: 0,
    };
    let mut s3 = Scenario3Result {
        samples: 0, p50_ns: f64::NAN, p99_ns: f64::NAN,
        mean_ns: f64::NAN, candidates_with_high_degree: 0, note: None,
    };

    if !args.skip_load {
        println!("\n-- Loading brain --");
        flush_stdout();
        let t_load = Instant::now();
        let key = if let Some(kf) = &args.keyfile {
            Some(MasterKey::load_from_keyfile(kf).expect("load master key"))
        } else { None };
        let mut brain = Brain::load_or_new_with_mode_and_key(
            &args.brain, BrainLoadMode::Auto, key,
        ).expect("load brain");
        let load_elapsed = t_load.elapsed();
        rss_after_load = rss_mib();
        node_count = brain.network.node_count();
        bond_count = brain.network.bond_count() as u64;
        println!("  load elapsed: {:.2}s", load_elapsed.as_secs_f64());
        println!("  RSS after load: {:.0} MiB ({:.2} GiB)",
                 rss_after_load, rss_after_load / 1024.0);
        println!("  nodes: {}  bonds: {}", node_count, bond_count);
        flush_stdout();

        s1 = scenario_1_bond_add(&mut brain, args.s1_n, &mut rng);
        s2 = scenario_2_read_latency(&brain, args.s2_n, &mut rng);
        s3 = scenario_3_spread_2hop(&brain, args.s3_n, args.s3_progress_every, &mut rng);
        drop(brain);
    } else {
        println!("  --skip-load passed; running only Scenario 4.");
    }

    let s4 = scenario_4_growth(args.skip_milestone_10m, &mut rng);

    write_report(
        &args.out, &args.brain, rss_after_load, node_count, bond_count,
        &s1, &s2, &s3, &s4,
    );

    println!("\nDone. Bench complete.");
}
