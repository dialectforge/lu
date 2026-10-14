//! OP-022.s.v4 Phase E — production-pattern save bench.
//!
//! Critical lesson from OP-022.s.v3: a cold-load-then-save-once bench
//! ONLY exercises the full-passthrough path (overlay empty). Production
//! training adds ~5000 bonds per save interval, so by Save 2 the
//! overlay is non-empty and v3 falls back to the slow re-encode path.
//!
//! This harness simulates the continuous-training pattern:
//!   1. load brain from snapshot (mmap zero-copy)
//!   2. Save 1: overlay empty -> full passthrough (v3 path)
//!   3. add ~5000 nodes + bonds
//!   4. Save 2: overlay non-empty -> v4 PARTIAL passthrough (gated path)
//!   5. add ~5000 more
//!   6. Save 3: same as Save 2
//!
//! Gate: Save 2 + Save 3 throughput must be >= 25 MB/s.
//!
//! Daemon-isolated: stop lu4d-curriculum-ingest before running.
//!
//! Usage:
//!   op022sv4_bench <brain.lbr input> <output dir>
//!   The harness writes brain-v4bench-{1,2,3}.lbr into output dir.
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use lu4d::{Brain, BrainLoadMode, BrainSaveFormat};
use lu4d::node::{Confidence, KnowledgeContent, KnowledgeNode, KnowledgeUnit, EMBEDDING_DIM};
use lu4d::bond::{BondType, KeyframeSource, TemporalBond, TemporalKeyframe};
use lu4d::seed::Seed;

fn load_key() -> Result<lu4d::encrypt::MasterKey, String> {
    let p = std::env::var("LU4D_MASTER_KEY")
        .unwrap_or_else(|_| "/var/lib/lu/master.key".to_string());
    let b = std::fs::read(&p).map_err(|e| format!("read {}: {}", p, e))?;
    if b.len() != 32 { return Err(format!("key {} is {} bytes", p, b.len())); }
    let mut a = [0u8; 32]; a.copy_from_slice(&b);
    Ok(lu4d::encrypt::MasterKey::from_bytes(a))
}

fn make_bench_node(iter: u32, idx: u32) -> KnowledgeNode {
    let label = format!("bench_v4_iter{}_n{}", iter, idx);
    let seed = Seed::from_label(&label);
    // Tiny embedding stub (zeros are fine — bench is about save throughput).
    let emb = vec![0.0_f32; EMBEDDING_DIM];
    KnowledgeNode {
        seed,
        content: label.clone(),
        knowledge: KnowledgeUnit {
            content: KnowledgeContent::Concept(format!("bench-content {}", label)),
            confidence: Confidence::Unverified,
            created_at: 1_700_000_000_000,
            updated_at: 1_700_000_000_000,
        },
        embedding: emb,
        parent_seeds: Vec::new(),
        typed_ratio: 0.0,
    }
}

fn make_bench_bond(from: Seed, to: Seed) -> TemporalBond {
    TemporalBond {
        from,
        to,
        bond_type: BondType::HasProperty,
        keyframes: vec![TemporalKeyframe {
            timestamp: 1_700_000_000_000,
            weight: 0.5,
            source: KeyframeSource::Baseline,
        }],
        half_life_ms: (30u64 * 24 * 3600 * 1000) as f64,
    }
}

fn add_bench_overlay(brain: &mut Brain, iter: u32, count: u32) -> Result<(), String> {
    // Add `count` nodes then a chain of `count` bonds (each bond from
    // node[i] -> node[(i+1) % count]). Production overlay shape: more
    // bonds than nodes typically; this keeps it 1:1 for predictability.
    let mut seeds: Vec<Seed> = Vec::with_capacity(count as usize);
    for i in 0..count {
        let n = make_bench_node(iter, i);
        seeds.push(n.seed);
        brain.network.add_node(n);
    }
    for i in 0..count {
        let from = seeds[i as usize];
        let to = seeds[((i + 1) % count) as usize];
        let bond = make_bench_bond(from, to);
        brain.network.add_bond(bond).map_err(|e| format!("add_bond: {:?}", e))?;
    }
    Ok(())
}

fn report_save(
    label: &str,
    elapsed: std::time::Duration,
    out_path: &std::path::Path,
) -> Result<f64, String> {
    let bytes = std::fs::metadata(out_path)
        .map_err(|e| format!("stat {}: {}", out_path.display(), e))?
        .len();
    let secs = elapsed.as_secs_f64();
    let mb = (bytes as f64) / 1024.0 / 1024.0;
    let mbps = if secs > 0.0 { mb / secs } else { 0.0 };
    println!(
        "[bench] {:8} elapsed={:.2}s   size={:.1} MiB   throughput={:.1} MiB/s",
        label, secs, mb, mbps
    );
    Ok(mbps)
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: {} <brain.lbr input> <output dir>", args[0]);
        return ExitCode::from(1);
    }
    let brain_in = PathBuf::from(&args[1]);
    let out_dir = PathBuf::from(&args[2]);
    if !brain_in.exists() {
        eprintln!("[bench] FATAL: {} does not exist", brain_in.display());
        return ExitCode::from(1);
    }
    std::fs::create_dir_all(&out_dir).ok();

    let key = match load_key() {
        Ok(k) => k,
        Err(e) => { eprintln!("[bench] FATAL: {}", e); return ExitCode::from(1); }
    };

    println!("[bench] loading {} (mmap zero-copy)", brain_in.display());
    let t_load = Instant::now();
    let mut brain = match Brain::load_or_new_with_mode_and_key(
        &brain_in, BrainLoadMode::MmapZeroCopy, Some(key.clone()),
    ) {
        Ok(b) => b,
        Err(e) => { eprintln!("[bench] load: {:?}", e); return ExitCode::from(1); }
    };
    // Override the require_encrypted guard's symmetric-format check
    // is implicit (we save back to V3BinaryEncrypted).
    println!("[bench] loaded in {:.2}s, nodes~={}, bonds~={}",
        t_load.elapsed().as_secs_f64(),
        brain.network.node_count(),
        brain.network.bond_count(),
    );

    let out1 = out_dir.join("brain-v4bench-1.lbr");
    let out2 = out_dir.join("brain-v4bench-2.lbr");
    let out3 = out_dir.join("brain-v4bench-3.lbr");

    // -------- Save 1: overlay empty -> full passthrough (v3 path) --------
    println!("[bench] SAVE 1 (overlay empty -> v3 full passthrough)");
    let t1 = Instant::now();
    if let Err(e) = brain.save_to_format(&out1, BrainSaveFormat::V3BinaryEncrypted) {
        eprintln!("[bench] save 1 failed: {:?}", e); return ExitCode::from(1);
    }
    let s1 = t1.elapsed();
    let mbps1 = match report_save("Save 1", s1, &out1) {
        Ok(v) => v, Err(e) => { eprintln!("[bench] {}", e); return ExitCode::from(1); }
    };

    // -------- Add ~5000 entities --------
    println!("[bench] adding overlay batch 1 (5000 nodes + 5000 bonds)");
    if let Err(e) = add_bench_overlay(&mut brain, 1, 5000) {
        eprintln!("[bench] overlay add: {}", e); return ExitCode::from(1);
    }

    // -------- Save 2: overlay non-empty -> v4 partial passthrough --------
    println!("[bench] SAVE 2 (overlay non-empty -> v4 partial passthrough)");
    let t2 = Instant::now();
    if let Err(e) = brain.save_to_format(&out2, BrainSaveFormat::V3BinaryEncrypted) {
        eprintln!("[bench] save 2 failed: {:?}", e); return ExitCode::from(1);
    }
    let s2 = t2.elapsed();
    let mbps2 = match report_save("Save 2", s2, &out2) {
        Ok(v) => v, Err(e) => { eprintln!("[bench] {}", e); return ExitCode::from(1); }
    };

    // -------- Add another ~5000 entities --------
    println!("[bench] adding overlay batch 2 (5000 more)");
    if let Err(e) = add_bench_overlay(&mut brain, 2, 5000) {
        eprintln!("[bench] overlay add 2: {}", e); return ExitCode::from(1);
    }

    // -------- Save 3 --------
    println!("[bench] SAVE 3 (overlay larger -> v4 partial passthrough)");
    let t3 = Instant::now();
    if let Err(e) = brain.save_to_format(&out3, BrainSaveFormat::V3BinaryEncrypted) {
        eprintln!("[bench] save 3 failed: {:?}", e); return ExitCode::from(1);
    }
    let s3 = t3.elapsed();
    let mbps3 = match report_save("Save 3", s3, &out3) {
        Ok(v) => v, Err(e) => { eprintln!("[bench] {}", e); return ExitCode::from(1); }
    };

    let total = s1 + s2 + s3;
    println!("\n[bench] === SUMMARY ===");
    println!("[bench] Save 1: {:.2}s  {:.1} MiB/s  (v3 full passthrough)", s1.as_secs_f64(), mbps1);
    println!("[bench] Save 2: {:.2}s  {:.1} MiB/s  (v4 partial passthrough)", s2.as_secs_f64(), mbps2);
    println!("[bench] Save 3: {:.2}s  {:.1} MiB/s  (v4 partial passthrough)", s3.as_secs_f64(), mbps3);
    println!("[bench] total elapsed: {:.2}s", total.as_secs_f64());

    let gate = 25.0_f64;
    let pass = mbps2 >= gate && mbps3 >= gate;
    println!("[bench] gate (Save 2 & 3 >= {} MiB/s): {}",
        gate, if pass { "PASS" } else { "FAIL" });

    if pass { ExitCode::SUCCESS } else { ExitCode::from(2) }
}
