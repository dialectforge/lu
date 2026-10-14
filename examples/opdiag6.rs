//! v6 (read-only): full node-label substring scan (existence + degree).
use lu4d::encrypt::MasterKey;
use lu4d::Brain;
use std::path::Path;

fn main() {
    let mut a = std::env::args().skip(1);
    let bp = a.next().unwrap();
    let kf = a.next().unwrap();
    let needles: Vec<String> = a.map(|s| s.to_lowercase()).collect();
    let key = MasterKey::load_from_keyfile(&kf).unwrap();
    let mut brain = Brain::load_readonly(Path::new(&bp), Some(key)).unwrap();
    brain.network.ensure_keyword_index_built();
    let net = &brain.network;
    let seeds = net.all_seeds();
    eprintln!("scanning {} seeds", seeds.len());
    let mut n = 0;
    for s in &seeds {
        let lbl = net.node_label(s).unwrap_or_default();
        let low = lbl.to_lowercase();
        if needles.iter().any(|nd| low.contains(nd.as_str())) {
            println!("deg={:<6} outc={:<4} {:?}", net.seam_total_degree(s), net.neighbors_outgoing_capped(s).len(), lbl);
            n += 1;
            if n >= 600 { println!("...cap"); break; }
        }
    }
    eprintln!("matches={}", n);
}
