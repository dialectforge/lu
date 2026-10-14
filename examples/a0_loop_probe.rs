//! OP-A0 reason<->retrieve loop probe (READ-ONLY). Confirms the multi-step
//! loop iterates and terminates within the step cap, and measures loop latency
//! with vs without the A0 spread bound (set via LU_SPREAD_BEAM /
//! LU_SPREAD_MAX_EXPANSIONS in the environment).
//!
//! usage: a0_loop_probe <brain_dir> <keyfile> [max_context_iters]
use std::path::Path;
use lu4d::encrypt::MasterKey;
use lu4d::Brain;

const QUERIES: &[&str] = &[
    "How does photosynthesis relate to cellular respiration",
    "What connects gravity and the orbit of planets",
    "How does a compiler turn source code into a running program",
    "Why does inflation reduce purchasing power over time",
    "How do neurons transmit signals to each other",
    "What is the relationship between energy and mass",
    "How does the immune system fight infection",
    "How does electricity flow through a circuit",
];

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let brain_dir = Path::new(&a[1]);
    let key = MasterKey::load_from_keyfile(Path::new(&a[2])).expect("load key");
    let iters: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(4);
    let mut brain = Brain::load_readonly(brain_dir, Some(key)).expect("load_readonly");
    let mut cfg = brain.narration_config.clone();
    cfg.max_context_iters = iters;
    println!(
        "== reason-loop probe: max_context_iters={} beam={:?} maxexp={:?} ==",
        iters,
        std::env::var("LU_SPREAD_BEAM").ok(),
        std::env::var("LU_SPREAD_MAX_EXPANSIONS").ok()
    );
    let mut lats: Vec<u128> = Vec::new();
    for q in QUERIES {
        let t0 = std::time::Instant::now();
        let nr = brain.narrate_with_config(q, 3, 10, false, &cfg);
        let dt = t0.elapsed().as_millis();
        lats.push(dt);
        match &nr.reason_loop {
            Some(s) => println!(
                "  {:>7}ms iters_ran={} terminated={:<10} refused={} ctx={}  {}",
                dt, s.iters_ran, s.terminated, s.refused, s.context_set_size, q
            ),
            None => println!("  {:>7}ms (no loop summary)  {}", dt, q),
        }
    }
    lats.sort();
    let p50 = lats[lats.len() / 2];
    let p95 = lats[((lats.len() as f64 * 0.95) as usize).min(lats.len() - 1)];
    let max = *lats.last().unwrap();
    println!("LOOP LATENCY: p50={}ms p95={}ms max={}ms n={}", p50, p95, max, lats.len());
}
