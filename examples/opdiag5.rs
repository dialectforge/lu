//! v5 (read-only): dump e_<token> entity buckets + csense shelves.
use lu4d::encrypt::MasterKey;
use lu4d::Brain;
use std::path::Path;

fn main() {
    let mut args = std::env::args().skip(1);
    let bp = args.next().unwrap();
    let kf = args.next().unwrap();
    let tokens: Vec<String> = args.collect();
    let key = MasterKey::load_from_keyfile(&kf).unwrap();
    let mut brain = Brain::load_readonly(Path::new(&bp), Some(key)).unwrap();
    brain.network.ensure_keyword_index_built();
    let net = &brain.network;
    for t in &tokens {
        let bkey = format!("e_{}", t.to_lowercase());
        println!("\n########## {:?}", bkey);
        match net.keyword_index_bucket(&bkey) {
            Some(b) => {
                println!("  bucket_len={}", b.len());
                for s in b.iter().take(50) {
                    let lbl = net.node_label(s).unwrap_or_default();
                    let deg = net.seam_total_degree(s);
                    let outs = net.neighbors_outgoing_capped(s);
                    let names: Vec<String> = outs.iter().take(14).map(|n| net.node_label_or_empty(n)).collect();
                    println!("   deg={:<5} {:?}  OUT[{}]: {}", deg, lbl, outs.len(), names.join(" | "));
                }
            }
            None => println!("  NO e-bucket"),
        }
        let handle = format!("csense{}", t.to_lowercase());
        let raw = lu4d::seed::Seed::from_label(&handle);
        let hs = if net.node_intern().get(&raw).is_some() { Some(raw) } else {
            let f = lu4d::seed::Seed::from_label(&lu4d::canonical::fold_seed_label(&handle));
            if net.node_intern().get(&f).is_some() { Some(f) } else { None }
        };
        match hs {
            Some(h) => {
                let outs = net.neighbors_outgoing_capped(&h);
                println!("  @@ {:?} deg={} shelfOUT[{}]: {}", handle, net.seam_total_degree(&h), outs.len(),
                    outs.iter().take(6).map(|n| net.node_label_or_empty(n)).collect::<Vec<_>>().join(" | "));
                for sh in outs.iter().take(3) {
                    let so = net.neighbors_outgoing_capped(sh);
                    println!("       shelf {:?} OUT[{}]: {}", net.node_label_or_empty(sh), so.len(),
                        so.iter().take(16).map(|n| net.node_label_or_empty(n)).collect::<Vec<_>>().join(" | "));
                }
            }
            None => println!("  @@ {:?}: MISSING handle", handle),
        }
    }
}
