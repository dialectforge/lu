//! OP-INGEST-1 Phase E — cached bond_count throughput bench.
//!
//! Measures the fix's centerpiece: per-call wall time of bond_count()
//! versus bond_count_walked() on the production-scale snapshot brain.
//!
//! Also simulates the `tool_teach_corpus_batch` telemetry pattern
//! (initial_bond_count + delta computation) which calls bond_count()
//! twice per batch.
//!
//! Daemon-isolated: stop lu4d-curriculum-ingest before running.
//!
//! Usage:
//!   op_ingest_1_bench <brain.lbr input>
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use lu4d::{Brain, BrainLoadMode};
use std::hint::black_box;

fn load_key() -> Result<lu4d::encrypt::MasterKey, String> {
    let p = std::env::var("LU4D_MASTER_KEY")
        .unwrap_or_else(|_| "/var/lib/lu/master.key".to_string());
    let b = std::fs::read(&p).map_err(|e| format!("read {}: {}", p, e))?;
    if b.len() != 32 { return Err(format!("key {} is {} bytes", p, b.len())); }
    let mut a = [0u8; 32]; a.copy_from_slice(&b);
    Ok(lu4d::encrypt::MasterKey::from_bytes(a))
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: {} <brain.lbr input>", args[0]);
        return ExitCode::from(1);
    }
    let brain_in = PathBuf::from(&args[1]);
    if !brain_in.exists() {
        eprintln!("[bench] FATAL: {} does not exist", brain_in.display());
        return ExitCode::from(1);
    }

    let key = match load_key() {
        Ok(k) => k,
        Err(e) => { eprintln!("[bench] FATAL: {}", e); return ExitCode::from(1); }
    };

    println!("[bench] loading {} (mmap zero-copy)", brain_in.display());
    let t_load = Instant::now();
    let brain = match Brain::load_or_new_with_mode_and_key(
        &brain_in, BrainLoadMode::MmapZeroCopy, Some(key),
    ) {
        Ok(b) => b,
        Err(e) => { eprintln!("[bench] load: {:?}", e); return ExitCode::from(1); }
    };
    println!("[bench] loaded in {:.2}s", t_load.elapsed().as_secs_f64());
    println!("[bench] node_count={}, bond_count={}",
        brain.network.node_count(), brain.network.bond_count());

    // Parity check first: cached == walked at startup.
    match brain.network.verify_bond_count_cache() {
        Ok(()) => println!("[bench] parity check at startup: OK"),
        Err((cached, walked)) => {
            eprintln!("[bench] FATAL: cache drift at startup -- cached={} walked={}",
                cached, walked);
            return ExitCode::from(2);
        }
    }

    // === Single-call latency ===
    println!("\n[bench] === single-call latency ===");
    let n_walked = 5;
    let n_cached = 1_000_000;

    let t_w = Instant::now();
    let mut sink_w: usize = 0;
    for _ in 0..n_walked {
        sink_w = sink_w.wrapping_add(brain.network.bond_count_walked());
    }
    let d_w = t_w.elapsed();
    let per_walked_us = d_w.as_micros() as f64 / n_walked as f64;
    println!("[bench] walked:  {} iters in {:.3}s -> {:.1} us/call (sink={})",
        n_walked, d_w.as_secs_f64(), per_walked_us, sink_w);

    let t_c = Instant::now();
    let mut sink_c: usize = 0;
    for _ in 0..n_cached {
        sink_c = sink_c.wrapping_add(black_box(brain.network.bond_count()));
    }
    let d_c = t_c.elapsed();
    let per_cached_ns = d_c.as_nanos() as f64 / n_cached as f64;
    println!("[bench] cached:  {} iters in {:.3}s -> {:.2} ns/call (sink={})",
        n_cached, d_c.as_secs_f64(), per_cached_ns, sink_c);

    let speedup = per_walked_us * 1000.0 / per_cached_ns;
    println!("[bench] speedup: {:.0}x ({}M-bond brain)", speedup, brain.network.bond_count() / 1_000_000);

    // === Simulated tool_teach_corpus_batch telemetry overhead ===
    // Each batch in production calls bond_count() twice (initial + final).
    // Simulate 100 batches.
    println!("\n[bench] === simulated batch telemetry (100 batches, 2 calls each) ===");
    let n_batches = 100;

    // walked path (pre-OP-INGEST-1)
    let t_w2 = Instant::now();
    let mut delta_sum_w: usize = 0;
    for _ in 0..n_batches {
        let initial = brain.network.bond_count_walked();
        // (imagined ingest work would happen here)
        let final_ = brain.network.bond_count_walked();
        delta_sum_w = delta_sum_w.wrapping_add(final_.saturating_sub(initial));
    }
    let d_w2 = t_w2.elapsed();
    println!("[bench] walked:  {} batches in {:.3}s -> {:.1} ms/batch (delta_sum={})",
        n_batches, d_w2.as_secs_f64(),
        d_w2.as_millis() as f64 / n_batches as f64, delta_sum_w);

    // cached path (OP-INGEST-1)
    let t_c2 = Instant::now();
    let mut delta_sum_c: usize = 0;
    for _ in 0..n_batches {
        let initial = black_box(brain.network.bond_count());
        let final_ = black_box(brain.network.bond_count());
        delta_sum_c = delta_sum_c.wrapping_add(final_.saturating_sub(initial));
    }
    let d_c2 = t_c2.elapsed();
    println!("[bench] cached:  {} batches in {:.3}s -> {:.3} us/batch (delta_sum={})",
        n_batches, d_c2.as_secs_f64(),
        d_c2.as_micros() as f64 / n_batches as f64, delta_sum_c);

    let batch_speedup = d_w2.as_nanos() as f64 / d_c2.as_nanos().max(1) as f64;
    println!("[bench] batch-telemetry speedup: {:.0}x", batch_speedup);

    // Sanity: both should agree.
    if delta_sum_w != delta_sum_c {
        eprintln!("[bench] FATAL: delta_sum mismatch -- walked={} cached={}",
            delta_sum_w, delta_sum_c);
        return ExitCode::from(3);
    }

    println!("\n[bench] === SUMMARY ===");
    println!("[bench] bond_count:        {:.2} ns/call (cached, was {:.1} us walked)",
        per_cached_ns, per_walked_us);
    println!("[bench] per-batch telemetry: {:.3} us (cached) vs {:.1} ms (walked) -> {:.0}x",
        d_c2.as_micros() as f64 / n_batches as f64,
        d_w2.as_millis() as f64 / n_batches as f64,
        batch_speedup);
    ExitCode::SUCCESS
}
