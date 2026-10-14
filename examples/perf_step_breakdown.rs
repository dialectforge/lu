//! Daemon Perf-1 — per-step breakdown for `Brain::teach_corpus`.
//!
//! Pure measurement tool. No production code is modified.
//!
//! Each step is timed in isolation by repeating it M times against the
//! exact same data shape teach_corpus uses, so we can attribute the
//! per-bond cost down to: format!-key construction, HashMap insert,
//! Seed::from_label, network add_node / add_bond, hub-cap decision,
//! source/negation overlay set.
//!
//! Usage:  cargo run --release --example perf_step_breakdown

use std::time::Instant;
use lu4d::{Brain, BondType, Confidence, Seed};
use lu4d::source::{SourceAttribution, SourceOverlay};
use lu4d::negation::NegationOverlay;

const N: usize = 100_000;

fn main() {
    eprintln!("perf_step_breakdown: N={} per step\n", N);

    // ----- Step A: Seed::from_label
    let labels: Vec<String> = (0..N).map(|i| format!("label_{:08}", i)).collect();
    let t = Instant::now();
    let mut sink: u64 = 0;
    for s in &labels {
        let seed = Seed::from_label(s);
        sink ^= seed.0[0] as u64;
    }
    let dt = t.elapsed();
    eprintln!("A. Seed::from_label              {:>9.0} ns/op  (sink={:#x})",
              dt.as_nanos() as f64 / N as f64, sink);

    // ----- Step B: SourceOverlay::key (format!-only)
    let seeds: Vec<Seed> = (0..N).map(|i| Seed::from_label(&format!("s_{:08}", i))).collect();
    let seeds2: Vec<Seed> = (0..N).map(|i| Seed::from_label(&format!("o_{:08}", i))).collect();
    let t = Instant::now();
    let mut sink2: usize = 0;
    for i in 0..N {
        let k = SourceOverlay::key(&seeds[i], &seeds2[i], BondType::IsA);
        sink2 = sink2.wrapping_add(k.len());
    }
    let dt = t.elapsed();
    eprintln!("B. SourceOverlay::key (format!)  {:>9.0} ns/op  (sink={})",
              dt.as_nanos() as f64 / N as f64, sink2);

    // ----- Step C: NegationOverlay::key (format!-only)
    let t = Instant::now();
    let mut sink3: usize = 0;
    for i in 0..N {
        let k = NegationOverlay::key(&seeds[i], &seeds2[i], BondType::IsA);
        sink3 = sink3.wrapping_add(k.len());
    }
    let dt = t.elapsed();
    eprintln!("C. NegationOverlay::key (format!) {:>8.0} ns/op  (sink={})",
              dt.as_nanos() as f64 / N as f64, sink3);

    // ----- Step D: SourceOverlay::set (key + insert)
    let mut so = SourceOverlay::new();
    let attr = SourceAttribution::training_with_url(
        "PerfBench".to_string(), Some("run-1".to_string()), None,
    );
    let t = Instant::now();
    for i in 0..N {
        so.set(seeds[i], seeds2[i], BondType::IsA, attr.clone());
    }
    let dt = t.elapsed();
    eprintln!("D. SourceOverlay::set            {:>9.0} ns/op  (size={})",
              dt.as_nanos() as f64 / N as f64, so.entries.len());

    // ----- Step E: NegationOverlay seen_positive insert (mirrors teach_with_source)
    let mut no = NegationOverlay::new();
    let t = Instant::now();
    for i in 0..N {
        let k = NegationOverlay::key(&seeds[i], &seeds2[i], BondType::IsA);
        no.seen_positive.insert(k);
    }
    let dt = t.elapsed();
    eprintln!("E. NegationOverlay seen_positive {:>9.0} ns/op  (size={})",
              dt.as_nanos() as f64 / N as f64, no.seen_positive.len());

    // ----- Step F: full Brain::teach_corpus (baseline reference)
    {
        let mut brain = Brain::new();
        let t = Instant::now();
        for i in 0..N {
            let s = format!("subj_{:08}", i);
            let o = format!("obj_{:08}", i);
            let _ = brain.teach_corpus(
                &s, BondType::IsA, &o, 0.7, Confidence::Unverified,
                "PerfBench", Some("run-1".to_string()), None,
            );
        }
        let dt = t.elapsed();
        let bonds = brain.network.bond_count();
        eprintln!("F. teach_corpus (spread, N={})  {:>9.0} ns/op  (stored_bonds={})",
                  N, dt.as_nanos() as f64 / N as f64, bonds);
    }

    // ----- Step G: full Brain::teach_corpus on populated brain
    {
        let mut brain = Brain::new();
        // Pre-populate with 200K bonds.
        for i in 0..200_000usize {
            let s = format!("prepop_subj_{:08}", i);
            let o = format!("prepop_obj_{:08}", i);
            let _ = brain.teach_corpus(
                &s, BondType::IsA, &o, 0.7, Confidence::Unverified,
                "PerfBench", Some("run-1".to_string()), None,
            );
        }
        let pre_bonds = brain.network.bond_count();
        eprintln!("   [populated brain: {} bonds, {} overlay entries]",
                  pre_bonds, brain.source_overlay.read().unwrap().entries.len());
        let t = Instant::now();
        for i in 0..N {
            let s = format!("newsubj_{:08}", i);
            let o = format!("newobj_{:08}", i);
            let _ = brain.teach_corpus(
                &s, BondType::IsA, &o, 0.7, Confidence::Unverified,
                "PerfBench", Some("run-1".to_string()), None,
            );
        }
        let dt = t.elapsed();
        eprintln!("G. teach_corpus on 200K-bond brain {:>5.0} ns/op",
                  dt.as_nanos() as f64 / N as f64);
    }
}
