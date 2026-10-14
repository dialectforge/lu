//! Daemon Perf-1 — find at-cap hubs in the production brain and
//! measure teach_corpus per-bond cost when the subject is saturated.
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
    let brain_path = args.get(1).cloned().unwrap_or_else(|| "/tmp/brain-prod-bench.lbr".to_string());
    let key_path = args.get(2).cloned().unwrap_or_else(|| "/tmp/master-prod.key".to_string());
    let n_per_subject: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(200);

    eprintln!("perf_hub_saturated: brain={} key={} N_per_subject={}", brain_path, key_path, n_per_subject);
    let load_t = Instant::now();
    let key = MasterKey::load_from_keyfile(&key_path).expect("load key");
    let mut brain = Brain::load_or_new_with_mode_and_key(
        &brain_path, BrainLoadMode::Auto, Some(key),
    ).expect("load brain");
    eprintln!("loaded in {:.2}s; bonds={} cap_max_out={}",
              load_t.elapsed().as_secs_f64(),
              brain.network.bond_count(),
              brain.hub_degree_config.max_out_degree);

    let all_seeds = brain.network.all_seeds();
    // Find the top 5 hubs by out_degree.
    let mut by_deg: Vec<(lu4d::Seed, usize)> = all_seeds.iter()
        .map(|s| (*s, brain.network.out_degree(s)))
        .collect();
    by_deg.sort_by(|a, b| b.1.cmp(&a.1));
    let max_out_cap = brain.hub_degree_config.max_out_degree as usize;
    eprintln!("top 5 hubs by out_degree (cap={}):", max_out_cap);
    for (i, (s, d)) in by_deg.iter().take(5).enumerate() {
        let label = brain.network.get_node(s).map(|n| n.content().to_string()).unwrap_or("?".into());
        eprintln!("  hub#{} out_deg={} {}", i, d, label);
    }

    // Pick the highest-out hub.
    let (hub_seed, hub_deg) = by_deg[0];
    let hub_label = brain.network.get_node(&hub_seed).map(|n| n.content().to_string()).unwrap_or("hub_unknown".into());
    let at_cap = hub_deg >= max_out_cap;
    eprintln!("--> using hub: {} (out_deg={} at_cap={})", hub_label, hub_deg, at_cap);

    // Teach n_per_subject fresh bonds with this hub as subject.
    let t = Instant::now();
    for i in 0..n_per_subject {
        let o = varied_label("hub_obj", i + 9_999_999_999);
        let _ = brain.teach_corpus(
            &hub_label, BondType::IsA, &o, 0.7, Confidence::Unverified,
            "PerfHubSat", Some("run-1".to_string()), None,
        );
    }
    let dt = t.elapsed();
    let per_bond_us = dt.as_nanos() as f64 / 1000.0 / n_per_subject as f64;
    let after_deg = brain.network.out_degree(&hub_seed);
    eprintln!("--- result ---");
    eprintln!("per_bond {:.2} us; after out_deg={}", per_bond_us, after_deg);
    println!("{:.2}", per_bond_us);
}
