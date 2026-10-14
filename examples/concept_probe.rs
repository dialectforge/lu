//! OP-SEED diagnostic (READ-ONLY). For each probe token:
//!   - keyword-index bucket size
//!   - what the per-token take-2 picker actually sees (byte order)
//!   - the top-degree nodes in that bucket (the "canonical concept" candidates)
//!   - whether a clean concept node exists at all
//!   - fragment-vs-concept shape stats over a sample
//!   - what select_seeds_for_query_ask actually returns for "What is a X"
//!
//! usage: concept_probe <brain_dir> <keyfile> <token> [<token> ...]

use std::path::Path;

use lu4d::encrypt::MasterKey;
use lu4d::network::{is_recall_pollution, AssociativeNetwork};
use lu4d::Brain;

fn label(net: &AssociativeNetwork, seed: &lu4d::Seed) -> String {
    net.node_label(seed).unwrap_or_else(|| "<NO LABEL>".to_string())
}

/// Crude shape classifier for a stored label.
/// "concept" = short, title-ish. "fragment" = sentence-shaped.
fn shape(l: &str) -> &'static str {
    if l == "<NO LABEL>" {
        return "nolabel";
    }
    let words = l.split_whitespace().count();
    let chars = l.chars().count();
    if chars > 80 || words > 10 {
        "fragment"
    } else if words >= 5 {
        "phrase"
    } else {
        "concept"
    }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let brain_dir = Path::new(&a[1]);
    let key = MasterKey::load_from_keyfile(Path::new(&a[2])).expect("load key");
    let all: Vec<String> = a[3..].to_vec();
    let tokens: Vec<String> = all.iter().filter(|x| !x.contains(char::is_whitespace)).cloned().collect();
    let queries: Vec<String> = all.iter().filter(|x| x.contains(char::is_whitespace)).cloned().collect();

    let mut brain = Brain::load_readonly(brain_dir, Some(key)).expect("load_readonly");
    assert!(brain.is_read_only_reader(), "MUST be read-only");
    eprintln!("loaded read-only. nodes={}", brain.network.node_count());

    for token in &tokens {
        println!("\n================ TOKEN: {:?} ================", token);

        let bucket = match brain.network.keyword_index_bucket(token) {
            Some(b) => b,
            None => {
                println!("  <no keyword-index bucket>");
                continue;
            }
        };
        let n = bucket.len();
        println!("  bucket_len = {}", n);
        println!("  term_idf   = {:.4}", brain.network.term_idf(token));

        // --- degrees over the whole bucket (cheap: mapram degree lookup) ---
        let mut by_deg: Vec<(usize, lu4d::Seed)> = Vec::with_capacity(n);
        for seed in bucket.iter() {
            by_deg.push((brain.network.seam_total_degree(seed), *seed));
        }
        let max_deg = by_deg.iter().map(|x| x.0).max().unwrap_or(0);
        let sum_deg: usize = by_deg.iter().map(|x| x.0).sum();
        println!(
            "  degree: max={} mean={:.2} zero_degree_entries={}",
            max_deg,
            sum_deg as f64 / n.max(1) as f64,
            by_deg.iter().filter(|x| x.0 == 0).count()
        );

        // --- A. what the per-token take-2 picker sees: BYTE ORDER, first 12 ---
        println!("\n  --- A. BYTE-ORDER HEAD (what the take-2 picker actually grabs) ---");
        for (i, seed) in bucket.iter().take(12).enumerate() {
            let d = brain.network.seam_total_degree(seed);
            let l = label(&brain.network, seed);
            let id = brain.network.node_intern().get(seed).map(|x| x.0);
            println!(
                "   [{:2}] deg={:6} id={:?} pollution={} shape={:8} label={:?}",
                i,
                d,
                id,
                is_recall_pollution(&l),
                shape(&l),
                if l.chars().count() > 140 {
                    format!("{}…", l.chars().take(140).collect::<String>())
                } else {
                    l.clone()
                }
            );
        }

        // --- B. top by degree: the canonical-concept candidates ---
        by_deg.sort_by(|x, y| y.0.cmp(&x.0).then(x.1.as_bytes().cmp(y.1.as_bytes())));
        println!("\n  --- B. TOP-15 BY DEGREE (the canonical-concept candidates) ---");
        for (i, (d, seed)) in by_deg.iter().take(15).enumerate() {
            let l = label(&brain.network, seed);
            let id = brain.network.node_intern().get(seed).map(|x| x.0);
            println!(
                "   [{:2}] deg={:6} id={:?} shape={:8} label={:?}",
                i,
                d,
                id,
                shape(&l),
                if l.chars().count() > 140 {
                    format!("{}…", l.chars().take(140).collect::<String>())
                } else {
                    l.clone()
                }
            );
        }

        // --- B2. where does the top-degree node sit in byte order? ---
        if let Some((topd, topseed)) = by_deg.first() {
            let pos = bucket
                .binary_search_by(|s| s.as_bytes().cmp(topseed.as_bytes()))
                .map(|p| p as i64)
                .unwrap_or(-1);
            println!(
                "\n  top-degree node (deg={}) sits at BYTE-ORDER RANK {} of {} \
                 -> take-2 would need rank <2",
                topd, pos, n
            );
        }

        // --- C. exact-label match: is there a node literally labelled the token? ---
        println!("\n  --- C. EXACT / NEAR-EXACT LABEL MATCH ---");
        let mut exact: Vec<(usize, lu4d::Seed, String)> = Vec::new();
        for (d, seed) in by_deg.iter().take(4000) {
            let l = label(&brain.network, seed);
            let ll = l.to_lowercase();
            if ll == *token || ll == format!("{}s", token) || ll.trim() == token.as_str() {
                exact.push((*d, *seed, l));
            }
        }
        if exact.is_empty() {
            println!("   none in top-4000-by-degree");
        } else {
            for (d, seed, l) in exact.iter().take(8) {
                let id = brain.network.node_intern().get(seed).map(|x| x.0);
                println!("   deg={:6} id={:?} label={:?}", d, id, l);
            }
        }

        // --- D. shape distribution over a sample ---
        println!("\n  --- D. LABEL SHAPE SAMPLE (stride sample, up to 300) ---");
        let stride = (n / 300).max(1);
        let (mut c_concept, mut c_phrase, mut c_frag, mut c_none) = (0, 0, 0, 0);
        let mut sampled = 0;
        for (i, seed) in bucket.iter().enumerate() {
            if i % stride != 0 {
                continue;
            }
            if sampled >= 300 {
                break;
            }
            sampled += 1;
            match shape(&label(&brain.network, seed)) {
                "concept" => c_concept += 1,
                "phrase" => c_phrase += 1,
                "fragment" => c_frag += 1,
                _ => c_none += 1,
            }
        }
        println!(
            "   sampled={} concept={} phrase={} fragment={} nolabel={}",
            sampled, c_concept, c_phrase, c_frag, c_none
        );

        // --- E. what seed selection actually chooses ---
        for q in [
            format!("What is a {}", token),
            format!("What is {}", token),
            token.clone(),
        ] {
            let seeds = brain.network.select_seeds_for_query_ask(&q);
            println!("\n  --- E. select_seeds_for_query_ask({:?}) -> {} seeds ---", q, seeds.len());
            for (i, (s, w)) in seeds.iter().enumerate() {
                let d = brain.network.seam_total_degree(s);
                let l = label(&brain.network, s);
                let id = brain.network.node_intern().get(s).map(|x| x.0);
                println!(
                    "   seed[{}] w={:.4} deg={:6} id={:?} shape={:8} label={:?}",
                    i,
                    w,
                    d,
                    id,
                    shape(&l),
                    if l.chars().count() > 140 {
                        format!("{}…", l.chars().take(140).collect::<String>())
                    } else {
                        l
                    }
                );
            }
        }
    }
    // ---- RAW QUERY TRACE ----
    for q in &queries {
        println!("\n################ QUERY: {:?} ################", q);
        let raw: Vec<String> = q
            .to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| !w.is_empty() && w.len() >= 2)
            .map(|s| s.to_string())
            .collect();
        println!("  raw tokens: {:?}", raw);
        for t in &raw {
            println!(
                "    token {:?}: hub_stopword={} query_noise={}",
                t,
                lu4d::network::is_hub_stopword(t),
                lu4d::network::is_query_noise(t)
            );
        }
        let kept: Vec<&String> = raw
            .iter()
            .filter(|t| !lu4d::network::is_hub_stopword(t) && !lu4d::network::is_query_noise(t))
            .collect();
        let kept_relaxed: Vec<&String> = raw
            .iter()
            .filter(|t| !lu4d::network::is_hub_stopword(t))
            .collect();
        println!("  STRICT content tokens (hub+noise filtered): {:?} -> len {}", kept, kept.len());
        println!("  RELAXED content tokens (hub only):          {:?} -> len {}", kept_relaxed, kept_relaxed.len());
        println!("  => majority-scan / coverage-promotion ACTIVE? {}", kept.len() >= 2);

        let seeds = brain.network.select_seeds_for_query_ask(q);
        println!("  select_seeds_for_query_ask -> {} seeds", seeds.len());
        for (i, (s, w)) in seeds.iter().enumerate() {
            let d = brain.network.seam_total_degree(s);
            let l = label(&brain.network, s);
            let id = brain.network.node_intern().get(s).map(|x| x.0);
            println!(
                "   seed[{}] w={:.4} deg={:6} id={:?} shape={:8} label={:?}",
                i,
                w,
                d,
                id,
                shape(&l),
                if l.chars().count() > 140 {
                    format!("{}...", l.chars().take(140).collect::<String>())
                } else {
                    l
                }
            );
        }

        let res = brain.ask(q, 3, 8);
        println!(
            "  ask(): seed_hits={} activated={} bonds_evaluated={}",
            res.seed_hits.len(),
            res.recall.activated.len(),
            res.recall.bonds_evaluated
        );
        for (i, act) in res.recall.activated.iter().take(6).enumerate() {
            let d = brain.network.seam_total_degree(&act.seed);
            println!(
                "   act[{}] a={:.4} depth={} deg={:6} content={:?}",
                i,
                act.activation,
                act.depth,
                d,
                if act.content.chars().count() > 120 {
                    format!("{}...", act.content.chars().take(120).collect::<String>())
                } else {
                    act.content.clone()
                }
            );
        }
    }
}
