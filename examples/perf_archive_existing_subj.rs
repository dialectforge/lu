//! Daemon Perf-1 — measure teach_corpus per-bond cost on an actual
//! archive-resident brain, using subjects that ALREADY EXIST in the
//! brain so get_node lookups exercise the archive-decrypt path.
use std::time::Instant;
use lu4d::{Brain, BondType, Confidence, BrainLoadMode};
use lu4d::encrypt::MasterKey;

fn varied_label(prefix: &str, i: usize) -> String {
    let mut z = (i as u64).wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z = z ^ (z >> 31);
    format!("{:016x}_{}_{}", z, prefix, i)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let brain_path = args.get(1).cloned().unwrap_or_else(|| "/tmp/brain-1pct-bench.lbr".to_string());
    let key_path = args.get(2).cloned().unwrap_or_else(|| "/tmp/master-1pct.key".to_string());
    let n: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(10_000);

    eprintln!("perf_archive_existing_subj: brain={} key={} N={}", brain_path, key_path, n);

    let load_t = Instant::now();
    let key = MasterKey::load_from_keyfile(&key_path).expect("load key");
    let mut brain = Brain::load_or_new_with_mode_and_key(
        &brain_path,
        BrainLoadMode::Auto,
        Some(key),
    ).expect("load brain");
    let load_dt = load_t.elapsed();
    let bond_count = brain.network.bond_count();
    eprintln!("loaded in {:.2}s; bonds={} overlay_entries={}",
              load_dt.as_secs_f64(),
              bond_count,
              brain.source_overlay.read().unwrap().entries.len());

    let all_seeds = brain.network.all_seeds();
    eprintln!("all_seeds: {} nodes available", all_seeds.len());
    if all_seeds.is_empty() {
        eprintln!("no seeds — bench cannot run");
        return;
    }

    // For each iteration, pick an existing seed as subject, get its
    // text content from the network, and teach a NEW (fresh) object.
    // This mirrors what the smart extractor does: "Abraham_Lincoln
    // IsA <fresh_concept>" where Abraham_Lincoln is a node already
    // present in the archive.
    let mut subjects: Vec<String> = Vec::with_capacity(n);
    for i in 0..n {
        let seed = all_seeds[i % all_seeds.len()];
        if let Some(node) = brain.network.get_node(&seed) {
            subjects.push(node.content().to_string());
        } else {
            subjects.push(format!("missing_{}", i));
        }
    }
    eprintln!("subjects ready ({}); first few: {:?}", subjects.len(), &subjects[..3.min(subjects.len())]);

    eprintln!("running {} teach_corpus calls (existing subjects, fresh objects)...", n);
    let t = Instant::now();
    for i in 0..n {
        let s = &subjects[i];
        let o = varied_label("perfbench_obj", i + 1_000_000_000);
        let _ = brain.teach_corpus(
            s, BondType::IsA, &o, 0.7, Confidence::Unverified,
            "PerfArchiveBench", Some("run-1".to_string()), None,
        );
    }
    let dt = t.elapsed();
    let per_bond_us = dt.as_nanos() as f64 / 1000.0 / n as f64;
    eprintln!("--- result ---");
    eprintln!("teach_corpus per bond: {:.2} us", per_bond_us);
    eprintln!("total: {:.2}s for {} bonds", dt.as_secs_f64(), n);
    eprintln!("throughput: {:.1} bonds/sec", n as f64 / dt.as_secs_f64());
    println!("{:.2}", per_bond_us);
}
