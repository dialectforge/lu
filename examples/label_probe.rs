//! BUG-1 diagnostic (READ-ONLY): does the node store actually hold the
//! content for the seeds that recall activates, and does the recall
//! path return it?
//!
//! usage: label_probe <brain_dir> <keyfile> "<query>"

use std::path::Path;

use lu4d::encrypt::MasterKey;
use lu4d::Brain;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let brain_dir = Path::new(&a[1]);
    let key = MasterKey::load_from_keyfile(Path::new(&a[2])).expect("load key");
    let query = a.get(3).cloned().unwrap_or_else(|| "What is a dog".to_string());

    let mut brain = Brain::load_readonly(brain_dir, Some(key)).expect("load_readonly");
    assert!(brain.is_read_only_reader());
    println!("loaded read-only. query = {:?}", query);

    // --- 1. seed selection ---
    let seeds = brain.network.select_seeds_for_query_ask(&query);
    println!("seeds selected: {}", seeds.len());

    let manifest = brain.manifest_snapshot().expect("manifest");
    let cache = brain.node_reader_cache.clone().expect("node reader cache");
    let empty_mt = lu4d::node_store::memtable::NodeMemTable::new();

    for (i, (seed, w)) in seeds.iter().take(10).enumerate() {
        let bytes = seed.as_bytes();
        let printable = bytes.iter().all(|b| *b == 0 || (*b >= 32 && *b < 127));
        let inline: String = bytes.iter().take_while(|b| **b != 0)
            .map(|b| *b as char).collect();
        let id = brain.network.node_intern().get(seed);

        // direct node-store fetch, bypassing the seam
        let store_content = match id {
            Some(nid) => match lu4d::node_store::read_path::get_node_v5(
                &empty_mt, &manifest, &cache, nid,
            ) {
                Ok(Some(n)) => format!("{:?}", n.content()),
                Ok(None) => "<store MISS>".to_string(),
                Err(e) => format!("<store ERR {:?}>", e),
            },
            None => "<no intern binding>".to_string(),
        };

        println!(
            "  seed[{}] w={:.3} id={:?} inline_printable={} inline={:?}\n      store_content={}",
            i, w, id.map(|n| n.0), printable, inline, store_content
        );
    }

    // --- 2. what recall returns ---
    let res = brain.ask(&query, 3, 8);
    println!(
        "\nask: seed_hits={} activated={} bonds_evaluated={}",
        res.seed_hits.len(),
        res.recall.activated.len(),
        res.recall.bonds_evaluated
    );
    for (i, act) in res.recall.activated.iter().take(8).enumerate() {
        let id = brain.network.node_intern().get(&act.seed);
        let store_content = match id {
            Some(nid) => match lu4d::node_store::read_path::get_node_v5(
                &empty_mt, &manifest, &cache, nid,
            ) {
                Ok(Some(n)) => format!("{:?}", n.content()),
                Ok(None) => "<store MISS>".to_string(),
                Err(e) => format!("<store ERR {:?}>", e),
            },
            None => "<no intern binding>".to_string(),
        };
        println!(
            "  act[{}] recall_content={:?} act={:.4} depth={} | id={:?} store_content={}",
            i, act.content, act.activation, act.depth, id.map(|n| n.0), store_content
        );
    }
}
