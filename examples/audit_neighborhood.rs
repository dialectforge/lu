//! Stage-2 deep neighborhood audit (BR.10c+ retrain).
//!
//! Sections:
//!   A. Top 20 hubs neighborhood dump (out + in bonds grouped by relation)
//!   B. 10 middle-tier samples (degree in 10..=50)
//!   C. Path probes between named hub pairs (BFS with relation labels)
//!   D. 10 multi-hop chains A --R1--> B --R2--> C where R1 != R2
//!   E. Themed 2-hop subgraph snapshots (capped at 50 bonds each)
//!
//! Read-only.

use lu4d::bond::BondType;
use lu4d::encrypt::MasterKey;
use lu4d::seed::Seed;
use lu4d::{Brain, BrainLoadMode};

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};

fn main() {
    let mut args = std::env::args().skip(1);
    let brain_path = args.next().expect("brain path required");
    let keyfile = args.next().expect("keyfile required");
    let seed_arg: u64 = args
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20_260_509);

    eprintln!("loading {} ...", brain_path);
    let key = MasterKey::load_from_keyfile(&keyfile).expect("load keyfile");
    let brain = Brain::load_or_new_with_mode_and_key(&brain_path, BrainLoadMode::Auto, Some(key))
        .expect("failed to load brain");
    eprintln!("loaded.");

    let nodes = brain.network.all_seeds();
    let n_nodes = nodes.len();
    let n_bonds = brain.network.bonds().len();

    println!("=== NEIGHBORHOOD AUDIT: {} ===", brain_path);
    println!("nodes={} bonds={} seed={}", n_nodes, n_bonds, seed_arg);

    // Adjacency: directed (out / in) + undirected-with-rel (for path probes / chains).
    let mut out_adj: HashMap<Seed, Vec<(Seed, BondType)>> = HashMap::new();
    let mut in_adj: HashMap<Seed, Vec<(Seed, BondType)>> = HashMap::new();
    let mut adj_undirected: HashMap<Seed, Vec<(Seed, BondType)>> = HashMap::new();
    for ((from, to), bv) in brain.network.bonds().iter() {
        let bt = bv.bond_type();
        out_adj.entry(*from).or_default().push((*to, bt));
        in_adj.entry(*to).or_default().push((*from, bt));
        adj_undirected.entry(*from).or_default().push((*to, bt));
        if from != to {
            adj_undirected.entry(*to).or_default().push((*from, bt));
        }
    }

    // Degree (undirected, deduped) — same convention as audit_depth.
    let mut undirected: HashSet<(Seed, Seed)> = HashSet::new();
    for ((from, to), _) in brain.network.bonds().iter() {
        let pair = if from.0 <= to.0 { (*from, *to) } else { (*to, *from) };
        undirected.insert(pair);
    }
    let mut deg: HashMap<Seed, usize> = HashMap::new();
    for (a, b) in &undirected {
        *deg.entry(*a).or_insert(0) += 1;
        if a != b {
            *deg.entry(*b).or_insert(0) += 1;
        }
    }
    let mut deg_pairs: Vec<(Seed, usize)> = deg.iter().map(|(s, d)| (*s, *d)).collect();
    deg_pairs.sort_by(|a, b| b.1.cmp(&a.1));

    // ---- Section A ----
    println!("\n========== A. TOP 20 HUBS NEIGHBORHOOD ==========");
    for (rank, (s, d)) in deg_pairs.iter().take(20).enumerate() {
        dump_neighborhood(s, *d, rank + 1, &brain, &out_adj, &in_adj, 5);
    }

    // ---- Section B ----
    println!("\n========== B. MIDDLE-TIER SAMPLES (deg 10..=50) ==========");
    let mut mid: Vec<(Seed, usize)> = deg_pairs
        .iter()
        .filter(|(_, d)| *d >= 10 && *d <= 50)
        .cloned()
        .collect();
    let mid_n = mid.len();
    println!("middle_tier_pool_size={}", mid_n);
    let mut rng_state: u64 = seed_arg;
    let mut rng = move || -> usize {
        rng_state = rng_state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (rng_state >> 33) as usize
    };
    let take = 10.min(mid_n);
    for i in 0..take {
        let j = i + (rng() % (mid_n - i));
        mid.swap(i, j);
    }
    for (rank, (s, d)) in mid.iter().take(take).enumerate() {
        dump_neighborhood(s, *d, rank + 1, &brain, &out_adj, &in_adj, 4);
    }

    // ---- Section C ----
    println!("\n========== C. PATH PROBES (named hub pairs) ==========");
    let mut by_label: HashMap<String, Seed> = HashMap::new();
    for s in &nodes {
        if let Some(view) = brain.network.get_node(s) {
            by_label.entry(view.content().to_string()).or_insert(*s);
        }
    }
    let pairs: &[(&str, &str)] = &[
        ("Albert Einstein", "United States"),
        ("History", "China"),
        ("United Kingdom", "England"),
        ("Bertrand Russell", "Aristotle"),
        ("Berlin", "France"),
    ];
    for (a_lbl, b_lbl) in pairs {
        match (by_label.get(*a_lbl), by_label.get(*b_lbl)) {
            (Some(&sa), Some(&sb)) => {
                println!("\n--- Path: {:?} -> {:?} ---", a_lbl, b_lbl);
                match bfs_path(&adj_undirected, sa, sb, 10) {
                    Some(path) => {
                        let chain = format_path(&path, &brain);
                        println!("path_len={} chain: {}", path.len() - 1, chain);
                    }
                    None => println!("UNREACHABLE within 10 hops"),
                }
            }
            (None, _) => println!("\n--- Path: {:?} -> {:?} --- A NOT IN GRAPH", a_lbl, b_lbl),
            (_, None) => println!("\n--- Path: {:?} -> {:?} --- B NOT IN GRAPH", a_lbl, b_lbl),
        }
    }

    // ---- Section D ----
    println!("\n========== D. MULTI-HOP RELATION CHAINS (R1 != R2) ==========");
    let mut chains: Vec<(Seed, BondType, Seed, BondType, Seed)> = Vec::new();
    let mut tries = 0usize;
    let mut rng_state2: u64 = seed_arg.wrapping_add(0xC0FF_EE12_3456_7890);
    let mut rng2 = move || -> usize {
        rng_state2 = rng_state2
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (rng_state2 >> 33) as usize
    };
    let nodes_with_out: Vec<Seed> = out_adj.keys().copied().collect();
    while chains.len() < 10 && tries < 20000 && !nodes_with_out.is_empty() {
        tries += 1;
        let a = nodes_with_out[rng2() % nodes_with_out.len()];
        let outs_a = match out_adj.get(&a) {
            Some(v) if !v.is_empty() => v,
            _ => continue,
        };
        let (b, r1) = outs_a[rng2() % outs_a.len()];
        let outs_b = match out_adj.get(&b) {
            Some(v) if !v.is_empty() => v,
            _ => continue,
        };
        let (c, r2) = outs_b[rng2() % outs_b.len()];
        if r1 == r2 || a == b || b == c || a == c {
            continue;
        }
        chains.push((a, r1, b, r2, c));
    }
    println!("chains_found={}/10 tries={}", chains.len(), tries);
    for (i, (a, r1, b, r2, c)) in chains.iter().enumerate() {
        let la = lookup(a, &brain);
        let lb = lookup(b, &brain);
        let lc = lookup(c, &brain);
        println!(
            "  #{}: {:?} --[{:?}]--> {:?} --[{:?}]--> {:?}",
            i + 1,
            la,
            r1,
            lb,
            r2,
            lc
        );
    }

    // ---- Section E ----
    println!("\n========== E. THEMED 2-HOP SUBGRAPHS ==========");
    let themes = ["American Revolution", "Photosynthesis", "World War II"];
    for theme in &themes {
        println!("\n--- Theme: {:?} ---", theme);
        match by_label.get(*theme) {
            Some(&seed) => {
                println!("seed: {:?} (deg={})", lookup(&seed, &brain), deg.get(&seed).copied().unwrap_or(0));
                dump_subgraph(seed, &brain, &adj_undirected, 2, 50);
            }
            None => {
                println!("seed not found, listing fuzzy alternatives (substring, lowercased):");
                let lower = theme.to_lowercase();
                let mut alt = 0;
                for s in &nodes {
                    if let Some(view) = brain.network.get_node(s) {
                        let name = view.content().to_string();
                        if name.to_lowercase().contains(&lower) && alt < 8 {
                            println!(
                                "  alt: {:?} (deg={})",
                                name,
                                deg.get(s).copied().unwrap_or(0)
                            );
                            alt += 1;
                        }
                    }
                }
                if alt == 0 {
                    println!("  no fuzzy matches found");
                }
            }
        }
    }

    println!("\n=== END NEIGHBORHOOD AUDIT ===");
}

fn lookup(s: &Seed, brain: &Brain) -> String {
    brain
        .network
        .get_node(s)
        .map(|n| n.content().to_string())
        .unwrap_or_else(|| "<missing>".to_string())
}

fn dump_neighborhood(
    s: &Seed,
    deg: usize,
    rank: usize,
    brain: &Brain,
    out_adj: &HashMap<Seed, Vec<(Seed, BondType)>>,
    in_adj: &HashMap<Seed, Vec<(Seed, BondType)>>,
    sample_per_rel: usize,
) {
    let label = lookup(s, brain);
    println!("\n=== #{} {:?} (deg={}) ===", rank, label, deg);

    let outs = out_adj.get(s).cloned().unwrap_or_default();
    let mut out_grouped: BTreeMap<String, Vec<Seed>> = BTreeMap::new();
    for (to, bt) in &outs {
        out_grouped.entry(format!("{:?}", bt)).or_default().push(*to);
    }
    println!(
        "  out_total={} out_distinct_rels={}",
        outs.len(),
        out_grouped.len()
    );
    for (rel, tos) in &out_grouped {
        let count = tos.len();
        let sample: Vec<String> = tos.iter().take(sample_per_rel).map(|t| lookup(t, brain)).collect();
        let ellipsis = if count > sample_per_rel { " ..." } else { "" };
        println!("    [out:{}] x{}: {:?}{}", rel, count, sample, ellipsis);
    }

    let ins = in_adj.get(s).cloned().unwrap_or_default();
    let mut in_grouped: BTreeMap<String, Vec<Seed>> = BTreeMap::new();
    for (from, bt) in &ins {
        in_grouped.entry(format!("{:?}", bt)).or_default().push(*from);
    }
    println!(
        "  in_total={} in_distinct_rels={}",
        ins.len(),
        in_grouped.len()
    );
    for (rel, froms) in &in_grouped {
        let count = froms.len();
        let sample: Vec<String> = froms.iter().take(sample_per_rel).map(|f| lookup(f, brain)).collect();
        let ellipsis = if count > sample_per_rel { " ..." } else { "" };
        println!("    [in:{}] x{}: {:?}{}", rel, count, sample, ellipsis);
    }

    let mut combined: HashSet<String> = HashSet::new();
    for k in out_grouped.keys() {
        combined.insert(k.clone());
    }
    for k in in_grouped.keys() {
        combined.insert(k.clone());
    }
    println!("  combined_distinct_rels_touching={}", combined.len());
}

fn bfs_path(
    adj: &HashMap<Seed, Vec<(Seed, BondType)>>,
    a: Seed,
    b: Seed,
    max_depth: usize,
) -> Option<Vec<(Seed, Option<BondType>)>> {
    if a == b {
        return Some(vec![(a, None)]);
    }
    let mut prev: HashMap<Seed, (Seed, BondType)> = HashMap::new();
    let mut visited: HashSet<Seed> = HashSet::new();
    let mut q: VecDeque<(Seed, usize)> = VecDeque::new();
    q.push_back((a, 0));
    visited.insert(a);
    while let Some((cur, d)) = q.pop_front() {
        if d >= max_depth {
            continue;
        }
        if let Some(neis) = adj.get(&cur) {
            for (n, bt) in neis {
                if visited.insert(*n) {
                    prev.insert(*n, (cur, *bt));
                    if *n == b {
                        let mut path: Vec<(Seed, Option<BondType>)> = Vec::new();
                        let mut node = b;
                        while node != a {
                            let (p, bt) = prev[&node];
                            path.push((node, Some(bt)));
                            node = p;
                        }
                        path.push((a, None));
                        path.reverse();
                        return Some(path);
                    }
                    q.push_back((*n, d + 1));
                }
            }
        }
    }
    None
}

fn format_path(path: &[(Seed, Option<BondType>)], brain: &Brain) -> String {
    if path.is_empty() {
        return String::new();
    }
    let mut s = format!("{:?}", lookup(&path[0].0, brain));
    for hop in path.iter().skip(1) {
        let bt = hop.1.expect("non-start hop must have relation");
        s.push_str(&format!(" --[{:?}]--> {:?}", bt, lookup(&hop.0, brain)));
    }
    s
}

fn dump_subgraph(
    seed: Seed,
    brain: &Brain,
    adj: &HashMap<Seed, Vec<(Seed, BondType)>>,
    depth: usize,
    max_bonds: usize,
) {
    let mut visited: HashSet<Seed> = HashSet::new();
    let mut q: VecDeque<(Seed, usize)> = VecDeque::new();
    q.push_back((seed, 0));
    visited.insert(seed);
    let mut printed = 0usize;
    while let Some((cur, d)) = q.pop_front() {
        if d >= depth {
            continue;
        }
        if let Some(neis) = adj.get(&cur) {
            for (n, bt) in neis {
                if printed >= max_bonds {
                    println!("  (cap reached at {} bonds; visited {} nodes)", printed, visited.len());
                    return;
                }
                println!(
                    "  d{} {:?} --[{:?}]--> {:?}",
                    d + 1,
                    lookup(&cur, brain),
                    bt,
                    lookup(n, brain)
                );
                printed += 1;
                if visited.insert(*n) {
                    q.push_back((*n, d + 1));
                }
            }
        }
    }
    println!("  (printed {} bonds, visited {} nodes)", printed, visited.len());
}
