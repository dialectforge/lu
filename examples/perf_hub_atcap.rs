//! Daemon Perf-1 — at-cap eviction-scan path.
//! Force max_out_degree=5000 on the loaded brain, then teach from a hub
//! whose out_degree is above that threshold to exercise the slow path.
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
    let n: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(200);
    let cap: u32 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(5000);

    let key = MasterKey::load_from_keyfile(&key_path).expect("load key");
    let mut brain = Brain::load_or_new_with_mode_and_key(
        &brain_path, BrainLoadMode::Auto, Some(key),
    ).expect("load brain");

    // Force the cap to the requested value (for an at-cap test).
    brain.hub_degree_config.max_out_degree = cap;
    brain.hub_degree_config.max_in_degree = cap;
    brain.network.sync_hub_read_config(&brain.hub_degree_config.clone());
    eprintln!("loaded; bonds={} forced max_out_degree={} max_in_degree={}",
              brain.network.bond_count(), cap, cap);

    let all_seeds = brain.network.all_seeds();
    // Find any hub above the forced cap.
    let mut hub: Option<(lu4d::Seed, usize, String)> = None;
    for s in &all_seeds {
        let d = brain.network.out_degree(s);
        if d as u32 >= cap {
            let label = brain.network.get_node(s).map(|n| n.content().to_string()).unwrap_or("".into());
            hub = Some((*s, d, label));
            break;
        }
    }
    let (hub_seed, hub_deg, hub_label) = match hub {
        Some(h) => h,
        None => {
            // Find the highest-out-degree node and use that — even if not at cap, this stresses the get_node path.
            let mut best: Option<(lu4d::Seed, usize, String)> = None;
            for s in &all_seeds {
                let d = brain.network.out_degree(s);
                let label = brain.network.get_node(s).map(|n| n.content().to_string()).unwrap_or("".into());
                match &best {
                    None => best = Some((*s, d, label)),
                    Some((_, bd, _)) if d > *bd => best = Some((*s, d, label)),
                    _ => {}
                }
            }
            best.expect("some hub")
        }
    };
    let at_cap = hub_deg as u32 >= cap;
    eprintln!("hub={} out_deg={} at_cap={} (forced cap={})", hub_label, hub_deg, at_cap, cap);

    let t = Instant::now();
    for i in 0..n {
        let o = varied_label("hub_obj", i + 9_999_999_999);
        let _ = brain.teach_corpus(
            &hub_label, BondType::IsA, &o, 0.7, Confidence::Unverified,
            "PerfHubAtCap", Some("run-1".to_string()), None,
        );
    }
    let dt = t.elapsed();
    let per_bond_us = dt.as_nanos() as f64 / 1000.0 / n as f64;
    let after_deg = brain.network.out_degree(&hub_seed);
    eprintln!("--- result ---");
    eprintln!("per_bond {:.2} us; after out_deg={}", per_bond_us, after_deg);
    println!("{:.2}", per_bond_us);
}
