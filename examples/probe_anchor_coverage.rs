//! K.1 probe v2: dumps neighborhood-reach signals per token.
//! For each token, for each keyword-hit seed, counts how many of
//! the seed's 1-hop neighbors appear in the activated set, and the
//! max activation among those neighbor activations. This lets us
//! distinguish a rich-neighborhood anchor (`quantum`) from a poor
//! one (`jellyfish`) on the canonical brain.

use lu4d::encrypt::MasterKey;
use lu4d::idk_gate::extract_query_tokens;
use lu4d::{Brain, BrainLoadMode};
use std::collections::{HashMap, HashSet};

fn main() {
    let mut args = std::env::args().skip(1);
    let brain_path = args.next().expect("brain path required");
    let keyfile = args.next().expect("keyfile required");
    let queries: Vec<String> = args.collect();
    if queries.is_empty() {
        eprintln!("need at least one query");
        std::process::exit(2);
    }

    eprintln!("loading {} (key={}) ...", brain_path, keyfile);
    let key = MasterKey::load_from_keyfile(&keyfile).expect("load keyfile");
    let mut brain = Brain::load_or_new_with_mode_and_key(
        &brain_path,
        BrainLoadMode::Auto,
        Some(key),
    )
    .expect("failed to load brain");
    brain.network.ensure_keyword_index_built();
    eprintln!("loaded.");

    for q in &queries {
        let result = brain.ask(q, 3, 10);
        let tokens = extract_query_tokens(q);

        // Build the activated map for fast lookup.
        let act_by_seed: HashMap<_, f32> = result
            .recall
            .activated
            .iter()
            .map(|a| (a.seed, a.activation))
            .collect();
        let activated_set: HashSet<_> = act_by_seed.keys().copied().collect();

        println!("==== query: {:?}", q);
        println!("  tokens                    : {:?}", tokens);
        println!("  total activated           : {}", result.recall.activated.len());

        for tok in &tokens {
            let hits = brain.network.keyword_search(tok);
            if hits.is_empty() {
                println!("  [{:24}] no keyword hits", tok);
                continue;
            }
            // Aggregate across all keyword-hit seeds for this token.
            let mut total_neighbors = 0usize;
            let mut activated_neighbors = 0usize;
            let mut max_neighbor_act = 0.0_f32;
            let mut sum_neighbor_act = 0.0_f32;
            let mut seed_activation = 0.0_f32;
            // Take only the top hit (highest hit_count) to keep the
            // signal simple. Could broaden later.
            let top_seed = hits[0].0;
            seed_activation = *act_by_seed.get(&top_seed).unwrap_or(&0.0_f32);
            let mut neighbor_set: HashSet<_> = HashSet::new();
            for n in brain.network.neighbors_outgoing(&top_seed) {
                neighbor_set.insert(n);
            }
            for n in brain.network.neighbors_incoming(&top_seed) {
                neighbor_set.insert(n);
            }
            total_neighbors = neighbor_set.len();
            for n in &neighbor_set {
                if let Some(a) = act_by_seed.get(n) {
                    activated_neighbors += 1;
                    max_neighbor_act = max_neighbor_act.max(*a);
                    sum_neighbor_act += *a;
                }
            }
            println!(
                "  [{:24}] seed_act={:.3} neighbors={} activated_neighbors={} max_nbr_act={:.3} sum_nbr_act={:.3}",
                tok, seed_activation, total_neighbors, activated_neighbors, max_neighbor_act, sum_neighbor_act,
            );
        }
        println!();
    }
}
