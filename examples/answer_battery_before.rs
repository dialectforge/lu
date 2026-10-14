//! OP-LABEL-1 / OP-NEARMISS-1 verification battery -- BEFORE variant.
//!
//! Identical query set and output shape to `answer_battery`, but built
//! against the PRE-FIX code, so it uses no API the fix introduced.
//!
//! usage: answer_battery_before <brain_dir> <keyfile>

use std::path::Path;

use lu4d::encrypt::MasterKey;
use lu4d::Brain;

const QUERIES: &[&str] = &[
    // --- the reported failure ---
    "What is a dog",
    // --- coding corpus ---
    "what does the def keyword do in python",
    "what is a binary search tree",
    "how does recursion use stack memory",
    "what is a hash table",
    // --- BUG 2: near-miss spelling ---
    "Tell me about the python launguage",
    // --- BUG 2 safety: genuine nonsense MUST still refuse ---
    "how do capacitance and flurgle and zorbnak relate",
    "what is a flurgle",
];

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let brain_dir = Path::new(&a[1]);
    let key = MasterKey::load_from_keyfile(Path::new(&a[2])).expect("load key");

    let mut brain = Brain::load_readonly(brain_dir, Some(key)).expect("load_readonly");
    assert!(brain.is_read_only_reader(), "must be a read-only reader");
    println!("== read-only reader up (BEFORE) ==\n");

    for q in QUERIES {
        let t0 = std::time::Instant::now();
        let nr = brain.narrate(q, 3, 10, false);
        let dt = t0.elapsed();
        println!("────────────────────────────────────────────────────────");
        println!("Q: {}", q);
        println!("A: {}", nr.answer);
        println!(
            "   [shape={:?} intent={:?} seeds={} activated={} bonds_evaluated={} {:.2}s]",
            nr.shape,
            nr.intent,
            nr.ask.seed_hits.len(),
            nr.ask.recall.activated.len(),
            nr.ask.recall.bonds_evaluated,
            dt.as_secs_f32(),
        );
        let blank = nr
            .ask
            .recall
            .activated
            .iter()
            .filter(|x| x.content.is_empty())
            .count();
        println!(
            "   [labels: {}/{} resolved, {} blank]",
            nr.ask.recall.activated.len() - blank,
            nr.ask.recall.activated.len(),
            blank,
        );
        for (i, act) in nr.ask.recall.activated.iter().take(4).enumerate() {
            println!("     {}. {:?} (act={:.4} d={})", i + 1, act.content, act.activation, act.depth);
        }
        println!();
    }
}
