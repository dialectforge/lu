//! Port C live demo: compare ReasoningAgent chain selection with intent
//! detection ON vs OFF (forced Generic) on the canonical brain, using the
//! same anchors and same query so any difference in chain selection is
//! attributable to the Port C intent bias.
//!
//! Usage: cargo run --release --example intent_demo -- <brain_path> <query> [<query2> ...]

use lu4d::reasoning::{
    chain_score_with_intent, classify_pattern, detect_intent, reason_with_config,
    reason_with_explicit_intent, QueryIntent, ReasoningConfig, ReasoningPattern, ReasoningTrace,
};
use lu4d::scope::TemporalScope;
use lu4d::{BondType, Brain, BrainLoadMode};

fn main() {
    let mut args = std::env::args().skip(1);
    let brain_path = args.next().expect("brain path required");
    let queries: Vec<String> = args.collect();
    if queries.is_empty() {
        eprintln!("give at least one query");
        std::process::exit(2);
    }

    eprintln!("loading {} ...", brain_path);
    let mut brain = Brain::load_or_new_with_mode(&brain_path, BrainLoadMode::Auto)
        .expect("failed to load brain");
    brain.network.ensure_keyword_index_built();
    eprintln!("loaded.");
    let cfg = ReasoningConfig::default();

    for q in &queries {
        println!();
        println!("========== query: {:?}", q);
        let detected = detect_intent(q);
        println!("detected intent: {:?}", detected);

        let off = reason_with_explicit_intent(
            &mut brain, q, TemporalScope::Current, &cfg, QueryIntent::Generic,
        );
        report("intent=Generic (Port B baseline)", &off, QueryIntent::Generic);

        let on = reason_with_config(&mut brain, q, TemporalScope::Current, &cfg);
        report(
            &format!("intent={:?} (Port C shipped)", detected),
            &on,
            detected,
        );

        let same = same_chain(&off, &on);
        if !same {
            println!("  >>> CHAIN FLIPPED by intent bias <<<");
        }
    }
}

fn same_chain(a: &ReasoningTrace, b: &ReasoningTrace) -> bool {
    if a.no_chain != b.no_chain {
        return false;
    }
    if a.bonds_used.len() != b.bonds_used.len() {
        return false;
    }
    a.bonds_used
        .iter()
        .zip(b.bonds_used.iter())
        .all(|(x, y)| x == y)
}

fn report(tag: &str, t: &ReasoningTrace, intent: QueryIntent) {
    if t.no_chain {
        println!("  {}: NO CHAIN -- {}", tag, t.final_answer);
        return;
    }
    let seq: Vec<BondType> = t.steps.iter().map(|s| s.bond_type).collect();
    let pattern: ReasoningPattern = classify_pattern(&seq);
    let min_w = t
        .steps
        .iter()
        .map(|s| s.confidence)
        .fold(f32::INFINITY, f32::min);
    let score = chain_score_with_intent(min_w, &seq, intent);
    println!(
        "  {}: {} steps  conf={:.3}  pattern={:?}  score={:.4}",
        tag, t.steps.len(), t.confidence, pattern, score
    );
    for (i, s) in t.steps.iter().enumerate() {
        println!(
            "    {} ({:?}, w={:.2}) {} -> {}",
            i, s.bond_type, s.confidence, s.from_content, s.to_content
        );
    }
}
