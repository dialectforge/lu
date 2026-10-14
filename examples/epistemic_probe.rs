//! OP-EPISTEMIC Phase-3 verification probe (READ-ONLY).
//! Prints the VERBATIM narrated answer + idk signals + per-cited-edge
//! Confidence tier for the validation-gate query set, so before/after is
//! a diff of real output. usage: epistemic_probe <brain_dir> <keyfile>
use std::path::Path;
use lu4d::encrypt::MasterKey;
use lu4d::Brain;
use lu4d::idk_gate::{evaluate_signals, IdkThresholds};
use lu4d::narration::QueryIntent;

const QUERIES: &[&str] = &[
    "is a dog loyal",
    "is a dog an animal",
    "what causes cancer",
    "does smoking cause cancer",
    "what does smoking cause",
    "who is Einstein",
    "what is dark matter",
    "what is water made of",
    "what is a hammer used for",
    "is a dog a better pet than a fish",
    "what do a dog and a cat have in common",
    "how do capacitance and flurgle and zorbnak relate",
    "what causes glorptar",
    "does a vaccine cause autism",
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
        let ratio = if sig.content_token_count > 0 {
            sig.anchored_token_count as f32 / sig.content_token_count as f32
        } else { 0.0 };
        println!("--------------------------------------------------------");
        println!("Q: {}", q);
        println!("A: {}", nr.answer);
        println!("   [shape={:?} {:.2}s max_act={:.3} anchored={}/{} ratio={:.2}]",
            nr.shape, dt.as_secs_f32(), sig.max_activation,
            sig.anchored_token_count, sig.content_token_count, ratio);
        if let Some(c) = lu4d::synthesis::try_synthesize(&brain, q, QueryIntent::Generic, None) {
            print!("   [synth kind={:?} conf={:.2} edges:", c.kind, c.confidence);
            for e in &c.cited_edges { print!(" ({:?} {:?})", e.bond, e.confidence); }
            println!("]");
        } else {
            println!("   [synth: None]");
        }
    }
}
