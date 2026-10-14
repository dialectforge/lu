//! Daemon Perf-1 — overlay scaling study.
use std::time::Instant;
use lu4d::{Brain, BondType, Confidence};

fn varied_label(prefix: &str, i: usize) -> String {
    // Splitmix-derived hex avoids the from_label aliasing on prefix-shared
    // labels. We want strings whose first 8 chars vary across i.
    let mut z = (i as u64).wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z = z ^ (z >> 31);
    format!("{:016x}_{}_{}", z, prefix, i)
}

fn main() {
    let mut brain = Brain::new();
    let probe_n: usize = 1000;
    println!("overlay_size,teach_corpus_per_bond_us");
    let checkpoints = [0usize, 50_000, 100_000, 250_000, 500_000, 1_000_000];
    let mut idx: usize = 0;
    for target in checkpoints {
        while idx < target {
            let s = varied_label("prepop_subj", idx);
            let o = varied_label("prepop_obj", idx + 100_000_000);
            let _ = brain.teach_corpus(
                &s, BondType::IsA, &o, 0.7, Confidence::Unverified,
                "PerfBench", Some("run-1".to_string()), None,
            );
            idx += 1;
        }
        let t = Instant::now();
        let probe_base = 1_000_000_000 + target;
        for j in 0..probe_n {
            let s = varied_label("probe_subj", probe_base + j);
            let o = varied_label("probe_obj", probe_base + j + 500_000_000);
            let _ = brain.teach_corpus(
                &s, BondType::IsA, &o, 0.7, Confidence::Unverified,
                "PerfBench", Some("run-1".to_string()), None,
            );
        }
        let dt = t.elapsed();
        let per_bond_us = dt.as_nanos() as f64 / 1000.0 / probe_n as f64;
        let overlay_size = brain.source_overlay.read().unwrap().entries.len();
        let bond_count = brain.network.bond_count();
        eprintln!("target={} overlay_size={} bonds={} per_bond_us={:.2}",
                 target, overlay_size, bond_count, per_bond_us);
        println!("{},{:.2}", overlay_size, per_bond_us);
        idx += probe_n;
    }
}
