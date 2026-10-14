//! OP-FLUENT Phase-4b discourse-composition probe (READ-ONLY).
//! usage: disc_probe <brain_dir> <keyfile>
use std::path::Path;
use lu4d::encrypt::MasterKey;
use lu4d::Brain;
use lu4d::idk_gate::{evaluate_signals, IdkThresholds};
use lu4d::narration::QueryIntent;

const QUERIES: &[&str] = &[
    "who is Einstein",
    "tell me about photosynthesis",
    "what is photosynthesis",
    "tell me about the sun",
    "what is a dog like",
    "what is a fish like",
    "what is water made of",
    "does a vaccine cause autism",
    "what causes glorptar",
    "who created you",
];

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let brain_dir = Path::new(&a[1]);
    let key = MasterKey::load_from_keyfile(Path::new(&a[2])).expect("load key");
    let mut brain = Brain::load_readonly(brain_dir, Some(key)).expect("load_readonly");
    let th = IdkThresholds::default();
    for q in QUERIES {
        let t0 = std::time::Instant::now();
        let nr = brain.narrate(q, 3, 10, false);
        let dt = t0.elapsed();
        let sig = evaluate_signals(&brain, &nr.ask, q, None, &th);
        println!("--------------------------------------------------------");
        println!("Q: {}", q);
        println!("A: {}", nr.answer);
        println!("   [shape={:?} {:.2}s]", nr.shape, dt.as_secs_f32());
        let _ = sig;
        if let Some(c) = lu4d::synthesis::try_synthesize(&brain, q, QueryIntent::Generic, None) {
            print!("   [synth kind={:?} conf={:.2} edges:", c.kind, c.confidence);
            for e in &c.cited_edges { print!(" ({} -{:?}-> {} {:?})", e.from, e.bond, e.to, e.confidence); }
            println!("]");
        } else {
            println!("   [synth: None]");
        }
    }
}
