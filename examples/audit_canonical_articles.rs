//! Identify nodes that should have canonical Wikipedia articles ingested.
//!
//! Selects nodes with high incoming-bond count but low outgoing-bond count
//! (popular as objects but with no outgoing structure yet). Outputs labels
//! one per line, ordered by total_degree desc.
//!
//! Usage: audit_canonical_articles <brain_path> <keyfile> [out_threshold] [limit]
//!
//! Read-only.

use lu4d::encrypt::MasterKey;
use lu4d::seed::Seed;
use lu4d::{Brain, BrainLoadMode};
use std::collections::HashMap;

fn main() {
    let mut args = std::env::args().skip(1);
    let brain_path = args.next().expect("brain path required");
    let keyfile = args.next().expect("keyfile required");
    let out_threshold: usize = args
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    let limit: usize = args
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or(500);

    eprintln!("loading {} ...", brain_path);
    let key = MasterKey::load_from_keyfile(&keyfile).expect("load keyfile");
    let brain = Brain::load_or_new_with_mode_and_key(&brain_path, BrainLoadMode::Auto, Some(key))
        .expect("failed to load brain");
    eprintln!("loaded.");

    let mut out_count: HashMap<Seed, usize> = HashMap::new();
    let mut in_count: HashMap<Seed, usize> = HashMap::new();
    for ((from, to), _bv) in brain.network.bonds().iter() {
        *out_count.entry(*from).or_insert(0) += 1;
        *in_count.entry(*to).or_insert(0) += 1;
    }

    let mut candidates: Vec<(Seed, usize, usize)> = Vec::new();
    for s in brain.network.all_seeds() {
        let oc = *out_count.get(&s).unwrap_or(&0);
        let ic = *in_count.get(&s).unwrap_or(&0);
        if oc < out_threshold && ic >= 2 {
            candidates.push((s, ic, oc));
        }
    }
    // sort by incoming desc
    candidates.sort_by(|a, b| b.1.cmp(&a.1));

    eprintln!("candidates: {}", candidates.len());
    eprintln!("writing top {} ...", limit);

    let mut emitted = 0usize;
    for (s, ic, oc) in candidates.iter() {
        if emitted >= limit { break; }
        if let Some(view) = brain.network.get_node(s) {
            let label = view.content().to_string();
            // Skip technical / pollution-looking labels: must be alphabetic at start,
            // not contain newlines or weird shell chars, length sensible.
            if label.is_empty() { continue; }
            if label.len() > 120 { continue; }
            if !label.chars().next().unwrap().is_alphabetic() { continue; }
            if label.contains("\\n") || label.contains("\\t") { continue; }
            // Print: label\\tin\\tout
            println!("{}\\t{}\\t{}", label, ic, oc);
            emitted += 1;
        }
    }
    eprintln!("emitted: {}", emitted);
}
