//! Port Perf-Ingest — per-bond teach throughput benchmark.
//!
//! Drives `Brain::teach_corpus` in a tight loop and records per-batch
//! elapsed time so we can plot the degradation curve from the project log
//! #672 / 1pct retrain test (22ms -> 80ms/bond over 17K nodes).
//!
//! Usage:
//!     cargo run --release --example perf_ingest_benchmark -- \
//!         <out.csv> <total_bonds> <sample_every> [hub|spread|zipf]
//!
//! Modes:
//!   * hub    — single subject node; objects all distinct.
//!              Pure O(degree(from)) test — exposes any per-add work
//!              that walks neighbors of `from`.
//!   * spread — every triple has a fresh subject AND fresh object.
//!              Constant per-add cost is the floor; should stay flat.
//!   * zipf   — Zipf(s=1.1) sampled subject pool, fresh objects.
//!              Closer to natural-language corpus shape.

use std::io::Write;
use std::time::Instant;
use lu4d::{Brain, BondType, Confidence};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out_path  = args.get(1).cloned().unwrap_or_else(|| "/tmp/perf_ingest.csv".to_string());
    let total_bonds: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(5_000);
    let sample_every: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(50);
    let mode = args.get(4).cloned().unwrap_or_else(|| "zipf".to_string());

    eprintln!("perf_ingest_benchmark: out={} total={} sample_every={} mode={}",
              out_path, total_bonds, sample_every, mode);

    let mut brain = Brain::new();
    // Match production teach path: corpus stamping on, default trust path.
    // teach_corpus already stamps TrainingCorpus; no extra config needed.

    let mut csv = std::fs::File::create(&out_path).expect("create csv");
    writeln!(csv, "bonds_so_far,batch_us,per_bond_us,total_node_count,total_bond_count").unwrap();

    let mut rng_state: u64 = 0xDEAD_BEEF_CAFE_BABE;
    let mut next = || {
        // SplitMix64
        rng_state = rng_state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = rng_state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };

    // Pre-allocate the subject pool for zipf mode.
    // 2000 subjects with Zipf(s=1.1) heavy tail roughly mirrors a
    // wikipedia-ingest distribution — head subjects get 100s of bonds,
    // tail subjects get 1.
    let pool_size: usize = 2000;

    let mut bonds_done: usize = 0;
    let mut batch_start = Instant::now();

    while bonds_done < total_bonds {
        let i = bonds_done;
        let (subj, obj) = match mode.as_str() {
            "hub" => ("hub_subject".to_string(), format!("obj_{:08}", i)),
            "spread" => (format!("subj_{:08}", i), format!("obj_{:08}", i)),
            _ => {
                // Zipf-ish: log-uniform pool index biases toward small indices.
                let r = next();
                let u = (r as f64) / (u64::MAX as f64);
                let idx = ((pool_size as f64).powf(u) - 1.0).max(0.0) as usize;
                let idx = idx.min(pool_size - 1);
                (format!("subj_{:06}", idx), format!("obj_{:08}", i))
            }
        };
        let _ = brain.teach_corpus(
            &subj,
            BondType::IsA,
            &obj,
            0.7,
            Confidence::Unverified,
            "PerfIngestBenchmark",
            Some("bench-run-001".to_string()),
            None,
        );
        bonds_done += 1;
        if bonds_done % sample_every == 0 {
            let dt = batch_start.elapsed();
            let dt_us = dt.as_micros() as u64;
            let per_bond_us = dt_us / (sample_every as u64);
            let n_nodes = brain.network.all_seeds().len();
            let n_bonds = brain.network.bond_count();
            writeln!(csv, "{},{},{},{},{}",
                bonds_done, dt_us, per_bond_us, n_nodes, n_bonds).unwrap();
            if bonds_done % (sample_every * 20) == 0 {
                eprintln!("  bonds={} per_bond_us={} nodes={} stored_bonds={}",
                          bonds_done, per_bond_us, n_nodes, n_bonds);
            }
            batch_start = Instant::now();
        }
    }

    csv.flush().unwrap();
    eprintln!("benchmark complete: {} bonds taught, csv={}", bonds_done, out_path);
}
