//! Stage-1 retrain depth audit.
//!
//! Replicates the format of /var/lib/lu/audit_5k_validation_2026-05-08.txt:
//! - counts (nodes, bonds, ratio)
//! - BondType distribution (sorted desc, with %)
//! - KnowledgeContent.kind distribution
//! - Confidence distribution
//! - SourceKind distribution
//! - Trust bucket distribution
//! - Multi-source nodes (TrainingCorpus.url) — total + >=2 distinct urls
//! - Degree distribution (undirected, deduped) with min/p10/.../p99/max/mean
//!   + histogram buckets
//! - Connected components (undirected) — total + giant + isolated + top-5 sizes
//! - Top 50 hubs by degree (label, degree, distinct_relation_types)
//! - Path-length probe (20 random pairs in giant component, BFS)
//!
//! Usage:
//!   audit_depth <brain> <keyfile> [seed=20260509]
//!
//! Read-only — never mutates the brain.

use lu4d::bond::BondType;
use lu4d::encrypt::MasterKey;
use lu4d::node::Confidence;
use lu4d::seed::Seed;
use lu4d::source::SourceKind;
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

    eprintln!("loading {} (key={}) ...", brain_path, keyfile);
    let key = MasterKey::load_from_keyfile(&keyfile).expect("load keyfile");
    let brain = Brain::load_or_new_with_mode_and_key(
        &brain_path,
        BrainLoadMode::Auto,
        Some(key),
    )
    .expect("failed to load brain");
    eprintln!("loaded.");

    let nodes = brain.network.all_seeds();
    let n_nodes = nodes.len();
    let n_bonds = brain.network.bonds().len();

    println!("=== AUDIT: {} ===", brain_path);
    println!("nodes={}", n_nodes);
    println!("bonds={}", n_bonds);
    if n_nodes > 0 {
        let r = n_bonds as f64 / n_nodes as f64;
        println!("ratio_nodes_to_bonds=1:{:.4}", r);
    }

    // ---- BondType distribution ----
    let mut bt_hist: HashMap<String, usize> = HashMap::new();
    let mut bt_total = 0usize;
    for ((_from, _to), bv) in brain.network.bonds().iter() {
        let bt = bv.bond_type();
        *bt_hist.entry(format!("{:?}", bt)).or_insert(0) += 1;
        bt_total += 1;
    }
    let mut bt_sorted: Vec<(&String, &usize)> = bt_hist.iter().collect();
    bt_sorted.sort_by(|a, b| b.1.cmp(a.1));
    println!("\n--- BondType distribution ---");
    for (name, cnt) in &bt_sorted {
        let pct = if bt_total > 0 { (**cnt as f64 / bt_total as f64) * 100.0 } else { 0.0 };
        println!("  {:<26} {:>8} {:.2}%", name, cnt, pct);
    }
    let related_to_share = bt_hist.get("RelatedTo").copied().unwrap_or(0);
    let rt_pct = if bt_total > 0 { related_to_share as f64 / bt_total as f64 * 100.0 } else { 0.0 };
    println!("RelatedTo_share={} ({:.2}%)", related_to_share, rt_pct);
    println!("BondType_variants_seen={}/28 ({:.1}%)", bt_hist.len(), bt_hist.len() as f64 / 28.0 * 100.0);

    // ---- KnowledgeContent.kind distribution ----
    let mut kind_hist: HashMap<&str, usize> = HashMap::new();
    let mut kind_total = 0usize;
    let mut conf_hist: HashMap<&'static str, usize> = HashMap::new();
    for s in &nodes {
        if let Some(view) = brain.network.get_node(s) {
            *kind_hist.entry(view.kind_str()).or_insert(0) += 1;
            let c = match view.confidence() {
                Confidence::Terminal => "Terminal",
                Confidence::Verified => "Verified",
                Confidence::Unverified => "Unverified",
                Confidence::Inferred => "Inferred",
                Confidence::Contested => "Contested",
            };
            *conf_hist.entry(c).or_insert(0) += 1;
            kind_total += 1;
        }
    }
    println!("\n--- KnowledgeContent.kind distribution (nodes) ---");
    let mut kv: Vec<(&&str, &usize)> = kind_hist.iter().collect();
    kv.sort_by(|a, b| b.1.cmp(a.1));
    for (k, c) in &kv {
        let pct = if kind_total > 0 { **c as f64 / kind_total as f64 * 100.0 } else { 0.0 };
        println!("  {:<16} {:>8} {:.2}%", k, c, pct);
    }

    println!("\n--- Confidence distribution (nodes) ---");
    let mut cv: Vec<(&&str, &usize)> = conf_hist.iter().collect();
    cv.sort_by(|a, b| b.1.cmp(a.1));
    for (k, c) in &cv {
        let pct = if kind_total > 0 { **c as f64 / kind_total as f64 * 100.0 } else { 0.0 };
        println!("  {:<16} {:>8} {:.2}%", k, c, pct);
    }

    // ---- SourceKind + Trust bucket ----
    let mut sk_hist: HashMap<&'static str, usize> = HashMap::new();
    let mut trust_buckets: BTreeMap<&'static str, usize> = BTreeMap::new();
    let mut node_urls: HashMap<Seed, HashSet<String>> = HashMap::new();
    let mut bond_keys: Vec<(Seed, Seed, BondType)> = Vec::with_capacity(n_bonds);
    let mut tier1_corpus_hist: HashMap<String, usize> = HashMap::new();
    let mut all_corpus_hist: HashMap<String, usize> = HashMap::new();

    for ((from, to), bv) in brain.network.bonds().iter() {
        let bt = bv.bond_type();
        bond_keys.push((*from, *to, bt));
        let attribution = brain.source_overlay.read().unwrap().get(from, to, bt);
        let sk_label: &'static str = match &attribution.kind {
            SourceKind::TrainingCorpus { corpus, url, .. } => {
                *all_corpus_hist.entry(corpus.clone()).or_insert(0) += 1;
                if let Some(u) = url {
                    node_urls.entry(*from).or_default().insert(u.clone());
                    node_urls.entry(*to).or_default().insert(u.clone());
                }
                if matches!(
                    corpus.as_str(),
                    "wikimedia/wikipedia"
                        | "wikimedia/structured-wikipedia"
                        | "ranjaykrishna/visual_genome"
                        | "grokipedia"
                ) {
                    *tier1_corpus_hist.entry(corpus.clone()).or_insert(0) += 1;
                }
                "training"
            }
            SourceKind::Research { .. } => "research",
            SourceKind::Teach { .. } => "teach",
            SourceKind::Document { .. } => "document",
            SourceKind::ChainedResearch { .. } => "chained_research",
            SourceKind::SessionPromote { .. } => "session_promote",
            SourceKind::PackInstall { .. } => "pack_install",
            SourceKind::Unknown => "unknown",
        };
        *sk_hist.entry(sk_label).or_insert(0) += 1;
        let t = attribution.trust;
        let bucket = if t < 0.5 { "low(0..0.5)" }
                     else if t < 0.7 { "mid(0.5..0.7)" }
                     else if t < 0.9 { "high(0.7..0.9)" }
                     else { "top(0.9..)" };
        *trust_buckets.entry(bucket).or_insert(0) += 1;
    }

    println!("\n--- SourceKind distribution (bonds) ---");
    let mut skv: Vec<(&&str, &usize)> = sk_hist.iter().collect();
    skv.sort_by(|a, b| b.1.cmp(a.1));
    for (k, c) in &skv {
        let pct = if n_bonds > 0 { **c as f64 / n_bonds as f64 * 100.0 } else { 0.0 };
        println!("  {:<26} {:>8} {:.2}%", k, c, pct);
    }

    println!("\n--- Corpus distribution (TrainingCorpus bonds) ---");
    let mut cph: Vec<(&String, &usize)> = all_corpus_hist.iter().collect();
    cph.sort_by(|a, b| b.1.cmp(a.1));
    for (k, c) in &cph {
        let pct = if n_bonds > 0 { **c as f64 / n_bonds as f64 * 100.0 } else { 0.0 };
        println!("  {:<40} {:>8} {:.2}%", k, c, pct);
    }

    println!("\n--- Trust bucket distribution (bonds) ---");
    for (k, c) in &trust_buckets {
        let pct = if n_bonds > 0 { *c as f64 / n_bonds as f64 * 100.0 } else { 0.0 };
        println!("  {:<26} {:>8} {:.2}%", k, c, pct);
    }

    // ---- Multi-source nodes ----
    let with_any: usize = node_urls.values().filter(|s| !s.is_empty()).count();
    let with_two_plus: usize = node_urls.values().filter(|s| s.len() >= 2).count();
    let pct_any = if n_nodes > 0 { with_any as f64 / n_nodes as f64 * 100.0 } else { 0.0 };
    let pct_two = if n_nodes > 0 { with_two_plus as f64 / n_nodes as f64 * 100.0 } else { 0.0 };
    println!("\n--- Multi-source nodes (TrainingCorpus.url) ---");
    println!("nodes_with_any_url={} ({:.2}%)", with_any, pct_any);
    println!("nodes_with_>=2_urls={} ({:.2}%) [direct depth signal]", with_two_plus, pct_two);

    // ---- Degree distribution (undirected, deduped) ----
    // Build undirected edge set canonicalized as (min,max) Seed pairs
    let mut undirected: HashSet<(Seed, Seed)> = HashSet::new();
    for (from, to, _bt) in &bond_keys {
        let pair = if from.0 <= to.0 { (*from, *to) } else { (*to, *from) };
        undirected.insert(pair);
    }
    let mut deg: HashMap<Seed, usize> = HashMap::new();
    let mut node_rels: HashMap<Seed, HashSet<BondType>> = HashMap::new();
    for (a, b) in &undirected {
        *deg.entry(*a).or_insert(0) += 1;
        if a != b { *deg.entry(*b).or_insert(0) += 1; }
    }
    for (from, to, bt) in &bond_keys {
        node_rels.entry(*from).or_default().insert(*bt);
        node_rels.entry(*to).or_default().insert(*bt);
    }
    // Include zero-degree nodes
    for s in &nodes {
        deg.entry(*s).or_insert(0);
    }
    let mut degs: Vec<usize> = deg.values().copied().collect();
    degs.sort_unstable();
    let pct = |p: f64| -> usize {
        if degs.is_empty() { return 0; }
        let idx = ((degs.len() as f64 - 1.0) * p / 100.0).round() as usize;
        degs[idx.min(degs.len() - 1)]
    };
    let min = degs.first().copied().unwrap_or(0);
    let max = degs.last().copied().unwrap_or(0);
    let sum: usize = degs.iter().sum();
    let mean = if degs.is_empty() { 0.0 } else { sum as f64 / degs.len() as f64 };
    println!("\n--- Degree distribution (undirected, deduped) ---");
    println!("min={} max={} mean={:.3}", min, max, mean);
    println!("p  10={}", pct(10.0));
    println!("p  25={}", pct(25.0));
    println!("p  50={}", pct(50.0));
    println!("p  75={}", pct(75.0));
    println!("p  90={}", pct(90.0));
    println!("p  95={}", pct(95.0));
    println!("p  99={}", pct(99.0));
    println!("histogram (degree bucket -> node count):");
    let buckets: &[(usize, usize, &str)] = &[
        (0, 0, "     0-0      "),
        (1, 1, "     1-1      "),
        (2, 2, "     2-2      "),
        (3, 3, "     3-3      "),
        (4, 4, "     4-4      "),
        (5, 5, "     5-5      "),
        (6, 7, "     6-7      "),
        (8, 9, "     8-9      "),
        (10, 14, "    10-14     "),
        (15, 19, "    15-19     "),
        (20, 29, "    20-29     "),
        (30, 49, "    30-49     "),
        (50, 99, "    50-99     "),
        (100, 199, "   100-199    "),
        (200, 499, "   200-499    "),
        (500, 999, "   500-999    "),
        (1000, 4999, "  1000-4999   "),
        (5000, usize::MAX, "    5000+     "),
    ];
    let max_bucket_count: usize = buckets.iter().map(|(lo, hi, _)| degs.iter().filter(|&&d| d >= *lo && d <= *hi).count()).max().unwrap_or(0).max(1);
    for (lo, hi, label) in buckets {
        let c = degs.iter().filter(|&&d| d >= *lo && d <= *hi).count();
        let bar_len = c * 60 / max_bucket_count;
        let bar = "#".repeat(bar_len);
        println!("{}  {:>8}  {}", label, c, bar);
    }

    // ---- Connected components (undirected, union-find) ----
    let mut idx_of: HashMap<Seed, usize> = HashMap::new();
    for (i, s) in nodes.iter().enumerate() { idx_of.insert(*s, i); }
    let mut parent: Vec<usize> = (0..n_nodes).collect();
    fn find(parent: &mut Vec<usize>, mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }
    fn union(parent: &mut Vec<usize>, a: usize, b: usize) {
        let ra = find(parent, a);
        let rb = find(parent, b);
        if ra != rb { parent[ra] = rb; }
    }
    for (a, b) in &undirected {
        if let (Some(&ia), Some(&ib)) = (idx_of.get(a), idx_of.get(b)) {
            union(&mut parent, ia, ib);
        }
    }
    let mut comp_size: HashMap<usize, usize> = HashMap::new();
    for i in 0..n_nodes {
        let r = find(&mut parent, i);
        *comp_size.entry(r).or_insert(0) += 1;
    }
    let mut comps: Vec<(usize, usize)> = comp_size.iter().map(|(r, s)| (*r, *s)).collect();
    comps.sort_by(|a, b| b.1.cmp(&a.1));
    let total_comps = comps.len();
    let giant = comps.first().map(|(_, s)| *s).unwrap_or(0);
    let giant_pct = if n_nodes > 0 { giant as f64 / n_nodes as f64 * 100.0 } else { 0.0 };
    let isolated = comps.iter().filter(|(_, s)| *s == 1).count();
    let giant_root = comps.first().map(|(r, _)| *r).unwrap_or(0);
    println!("\n--- Connected components (undirected) ---");
    println!("total_components={}", total_comps);
    println!("giant_component_size={} ({:.2}% of nodes)", giant, giant_pct);
    println!("isolated_singletons={}", isolated);
    println!("top-5 component sizes:");
    for (i, (_, s)) in comps.iter().take(5).enumerate() {
        let p = if n_nodes > 0 { *s as f64 / n_nodes as f64 * 100.0 } else { 0.0 };
        println!("  #{}: {} ({:.2}% of nodes)", i + 1, s, p);
    }

    // ---- Top 50 hubs by degree ----
    let mut deg_pairs: Vec<(Seed, usize)> = deg.iter().map(|(s, d)| (*s, *d)).collect();
    deg_pairs.sort_by(|a, b| b.1.cmp(&a.1));
    println!("\n--- Top 50 hubs by degree (rank, label, degree, distinct_relation_types) ---");
    for (i, (s, d)) in deg_pairs.iter().take(50).enumerate() {
        let label = brain.network.get_node(s).map(|n| n.content().to_string()).unwrap_or_else(|| "<missing>".to_string());
        let rels = node_rels.get(s).map(|r| r.len()).unwrap_or(0);
        println!("  {:>3}. deg={:>5}  rels={:>2}  {:?}", i + 1, d, rels, label);
    }

    // ---- Path-length probe: 20 random pairs in giant component ----
    let mut giant_seeds: Vec<Seed> = Vec::new();
    for s in &nodes {
        if let Some(&i) = idx_of.get(s) {
            if find(&mut parent, i) == giant_root {
                giant_seeds.push(*s);
            }
        }
    }
    let n_giant = giant_seeds.len();
    println!("\n--- Path-length probe (20 random pairs in giant component) ---");
    println!("giant_component_count={} (sampling from this set)", n_giant);
    if n_giant >= 2 {
        // adjacency for BFS — undirected
        let mut adj: HashMap<Seed, Vec<Seed>> = HashMap::new();
        for (a, b) in &undirected {
            adj.entry(*a).or_default().push(*b);
            if a != b { adj.entry(*b).or_default().push(*a); }
        }
        let mut rng_state: u64 = seed_arg.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut next = || {
            rng_state = rng_state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            (rng_state >> 33) as usize
        };
        let mut lens: Vec<i64> = Vec::new();
        let mut unreachable: usize = 0;
        for _ in 0..20 {
            let a = giant_seeds[next() % n_giant];
            let b = giant_seeds[next() % n_giant];
            if a == b { lens.push(0); continue; }
            // BFS from a to b
            let mut dist: HashMap<Seed, i64> = HashMap::new();
            let mut q: VecDeque<Seed> = VecDeque::new();
            dist.insert(a, 0); q.push_back(a);
            let mut found = -1i64;
            while let Some(cur) = q.pop_front() {
                if cur == b { found = dist[&cur]; break; }
                let cd = dist[&cur];
                if cd > 30 { break; }
                if let Some(neis) = adj.get(&cur) {
                    for &n in neis {
                        if !dist.contains_key(&n) {
                            dist.insert(n, cd + 1);
                            q.push_back(n);
                        }
                    }
                }
            }
            if found < 0 { unreachable += 1; } else { lens.push(found); }
        }
        let succ = lens.len();
        let mn: i64 = *lens.iter().min().unwrap_or(&0);
        let mx: i64 = *lens.iter().max().unwrap_or(&0);
        let mean = if lens.is_empty() { 0.0 } else { lens.iter().sum::<i64>() as f64 / lens.len() as f64 };
        println!("trials=20 successful={} unreachable={} mean={:.2} min={} max={}", succ, unreachable, mean, mn, mx);
        println!("raw lengths: {:?}", lens);
    }

    // ---- Random sample 10 bonds ----
    println!("\n--- 10 random bonds (subject, relation, object, corpus, run_id, trust) ---");
    let mut rng_state: u64 = seed_arg.wrapping_add(0xDEAD_BEEF_FEED_FACE);
    for i in 0..10 {
        rng_state = rng_state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        let idx = ((rng_state >> 33) as usize) % bond_keys.len().max(1);
        let (from, to, bt) = bond_keys[idx];
        let subj = brain.network.get_node(&from).map(|n| n.content().to_string()).unwrap_or_else(|| "<missing>".to_string());
        let obj = brain.network.get_node(&to).map(|n| n.content().to_string()).unwrap_or_else(|| "<missing>".to_string());
        let attribution = brain.source_overlay.read().unwrap().get(&from, &to, bt);
        let (corpus_s, run_id_s, url_s) = match &attribution.kind {
            SourceKind::TrainingCorpus { corpus, run_id, url } => (
                corpus.clone(),
                run_id.clone().unwrap_or_else(|| "<none>".to_string()),
                url.clone().unwrap_or_else(|| "<none>".to_string()),
            ),
            other => (format!("{:?}", other), "<n/a>".to_string(), "<n/a>".to_string()),
        };
        println!("--- sample #{} ---\n  subject  : {:?}\n  relation : {:?}\n  object   : {:?}\n  corpus   : {}\n  run_id   : {}\n  url      : {}\n  trust    : {:.4}", i+1, subj, bt, obj, corpus_s, run_id_s, url_s, attribution.trust);
    }

    println!("\n=== END AUDIT ===");
}
