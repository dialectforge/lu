//! Probe: dump sample node contents from the brain.

use lu4d::{Brain, BrainLoadMode};

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| "/opt/lu/brain.training.lbr".to_string());
    eprintln!("loading {}...", path);
    let brain = Brain::load_or_new_with_mode(&path, BrainLoadMode::Auto).unwrap();
    let seeds = brain.network.all_seeds();
    eprintln!("total seeds = {}", seeds.len());
    for s in seeds.iter().take(60) {
        if let Some(n) = brain.network.get_node(s) {
            println!("{}", n.content());
        }
    }
}
