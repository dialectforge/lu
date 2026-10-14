//! OP-LABEL-1 / OP-NEARMISS-1 verification battery (READ-ONLY).
//!
//! Opens the canonical brain through the OP-007 read-only reader and
//! prints the VERBATIM narrated answer for a fixed query set, so the
//! before/after comparison is a diff of real output rather than a
//! claim.
//!
//! usage: answer_battery <brain_dir> <keyfile>

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
    println!(
        "== read-only reader up. node_label_source={} ==\n",
        brain.network.has_node_label_source()
    );

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
            "   [labels: {}/{} resolved, {} blank; store_errors={}]",
            nr.ask.recall.activated.len() - blank,
            nr.ask.recall.activated.len(),
            blank,
            brain.network.label_store_error_count(),
        );
        for (i, act) in nr.ask.recall.activated.iter().take(4).enumerate() {
            println!("     {}. {:?} (act={:.4} d={})", i + 1, act.content, act.activation, act.depth);
        }
        println!();
    }
}
