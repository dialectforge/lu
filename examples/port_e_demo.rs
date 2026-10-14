//! Port E live demo: compare single-anchor (Port D) vs multi-anchor
//! (Port E) ReasoningAgent output on the canonical brain. The "before"
//! column calls `reason_legacy_single_anchor`, which mirrors the
//! pre-Port-E control flow exactly (`pick_anchors` -> `best_chain` over
//! every anchor at once -> `assemble_trace`).
//!
//! Usage: cargo run --release --example port_e_demo -- <brain_path> <query> [<query2> ...]

use lu4d::reasoning::{
    detect_intent, pick_query_anchors, reason_legacy_single_anchor, reason_with_config,
    try_multi_anchor_trace, ReasoningConfig, ReasoningTrace,
};
use lu4d::scope::TemporalScope;
use lu4d::{Brain, BrainLoadMode};

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
        let intent = detect_intent(q);
        let anchors = pick_query_anchors(&mut brain, q);
        println!("detected intent: {:?}", intent);
        println!("anchors: {} found", anchors.len());

        let port_e_fires = try_multi_anchor_trace(
            &brain,
            q,
            &TemporalScope::Current,
            &cfg,
            intent,
            &anchors,
        )
        .is_some();

        let before = reason_legacy_single_anchor(
            &mut brain,
            q,
            TemporalScope::Current,
            &cfg,
            intent,
        );
        let after = reason_with_config(&mut brain, q, TemporalScope::Current, &cfg);

        report("PORT D (single-chain, before)", &before);
        report("PORT E (multi-anchor, after)", &after);
        if port_e_fires {
            println!("  >>> Port E FIRED <<<");
        } else {
            println!("  (Port E did NOT fire — single-anchor or all-Mixed; outputs identical)");
        }
    }
}

fn report(label: &str, t: &ReasoningTrace) {
    println!("--- {} ---", label);
    if t.no_chain {
        println!("  (no chain): {}", t.final_answer);
        return;
    }
    let bond_seq: Vec<String> = t
        .steps
        .iter()
        .map(|s| format!("{:?}", s.bond_type))
        .collect();
    println!("  evidence_kind: {:?}", t.evidence_kind);
    println!("  bonds:        [{}]", bond_seq.join(", "));
    println!("  confidence:   {:.4}", t.confidence);
    println!("  final_answer: {}", t.final_answer);
}
