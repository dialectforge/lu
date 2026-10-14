//! BR.9 audit: BondType distribution over Tier-2 bonds.
//!
//! After BR.8b the subjects were clean but every Tier-2 bond was RelatedTo
//! (the `extract_response_bond` hardcode). BR.9 routes the BondType from
//! the predicate verb that fired Pattern A/B + phrase-anchored cues. This
//! audit reports the histogram so we can confirm the monoshape is gone.
//!
//! Pass criteria (per BR.9 directive):
//!   * >= 5 distinct BondType variants firing  (HARD pass)
//!   * No single variant >70% of total          (quality bar)
//!   * >=10% of bonds in IsA/HasProperty/etc.   (quality bar)
//!
//! Usage:
//!   audit_relation_distribution <brain> <keyfile> [sample_n=30] [seed=20260509]
//!
//! Read-only -- never mutates the brain.

use lu4d::bond::BondType;
use lu4d::encrypt::MasterKey;
use lu4d::seed::Seed;
use lu4d::source::SourceKind;
use lu4d::{Brain, BrainLoadMode};

use std::collections::BTreeMap;

fn is_tier2_corpus(corpus: &str) -> bool {
    matches!(
        corpus,
        "HuggingFaceH4/no_robots"
            | "HuggingFaceH4/instruction-dataset"
            | "no_robots"
            | "instruction-dataset"
    )
}

fn main() {
    let mut args = std::env::args().skip(1);
    let brain_path = args.next().expect("brain path required");
    let keyfile = args.next().expect("keyfile required");
    let sample_n: usize = args
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or(30);
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

    // Walk every bond, keep only TrainingCorpus stamps from Tier 2.
    let mut tier2: Vec<(Seed, Seed, BondType, String, Option<String>, Option<String>)> =
        Vec::new();
    for ((from, to), bv) in brain.network.bonds().iter() {
        let bond_type = bv.bond_type();
        let attribution = brain.source_overlay.read().unwrap().get(from, to, bond_type);
        if let SourceKind::TrainingCorpus { corpus, run_id, url } = attribution.kind {
            if is_tier2_corpus(&corpus) {
                tier2.push((*from, *to, bond_type, corpus, run_id, url));
            }
        }
    }

    let total = tier2.len();
    eprintln!("found {} Tier-2 bonds", total);
    if total == 0 {
        eprintln!("no Tier-2 bonds in this brain; nothing to audit.");
        std::process::exit(2);
    }

    // Histogram by BondType (Debug-formatted variant name for stable display).
    let mut hist: BTreeMap<String, usize> = BTreeMap::new();
    for (_, _, bt, _, _, _) in &tier2 {
        *hist.entry(format!("{:?}", bt)).or_insert(0) += 1;
    }

    println!(
        "== BR.9 Tier-2 BondType distribution ==\n\
         brain     : {}\n\
         total t2  : {}\n",
        brain_path, total
    );

    // Sorted by count descending.
    let mut hv: Vec<(&String, &usize)> = hist.iter().collect();
    hv.sort_by(|a, b| b.1.cmp(a.1));
    let max_share = hv.first().map(|(_, c)| **c as f64 / total as f64 * 100.0).unwrap_or(0.0);
    let typed_count: usize = hv
        .iter()
        .filter(|(name, _)| name.as_str() != "RelatedTo")
        .map(|(_, c)| **c)
        .sum();
    let typed_share = typed_count as f64 / total as f64 * 100.0;

    println!("BondType histogram (sorted by count):");
    for (name, count) in &hv {
        let pct = **count as f64 / total as f64 * 100.0;
        println!("  {:<20}  {:>6}  ({:>5.1}%)", name, count, pct);
    }
    println!("\ndistinct variants : {}", hist.len());
    println!("max single share : {:.1}%", max_share);
    println!("typed share      : {:.1}%  (RelatedTo excluded)", typed_share);

    // Pass criteria
    let pass_distinct = hist.len() >= 5;
    let pass_no_dominator = max_share <= 70.0;
    let pass_typed_min = typed_share >= 10.0;
    println!("\n-- pass criteria --");
    println!("  >= 5 distinct variants      : {}",
        if pass_distinct { "PASS" } else { "FAIL" });
    println!("  no single variant > 70%     : {}",
        if pass_no_dominator { "PASS" } else { "FAIL" });
    println!("  >= 10% typed (non-RelatedTo): {}",
        if pass_typed_min { "PASS" } else { "FAIL" });

    // Now sample N bonds with their (subject, relation, object, source).
    // Same Fisher-Yates LCG as audit_subject_provenance for parity.
    let mut indices: Vec<usize> = (0..total).collect();
    let mut rng_state: u64 = seed_arg.wrapping_add(0x9E37_79B9_7F4A_7C15);
    for i in (1..total).rev() {
        rng_state = rng_state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let j = (rng_state >> 33) as usize % (i + 1);
        indices.swap(i, j);
    }
    let take = sample_n.min(total);

    println!(
        "\n== {}-bond sample (rng_seed = {}) ==",
        take, seed_arg
    );
    for (i, sample_idx) in indices.iter().take(take).enumerate() {
        let (from, to, bond_type, corpus, run_id, url) = &tier2[*sample_idx];
        let subj = brain
            .network
            .get_node(from)
            .map(|n| n.content().to_string())
            .unwrap_or_else(|| format!("<no_node:{:?}>", from));
        let obj = brain
            .network
            .get_node(to)
            .map(|n| n.content().to_string())
            .unwrap_or_else(|| format!("<no_node:{:?}>", to));
        let run_id_s = run_id.as_deref().unwrap_or("<none>");
        let url_s = url.as_deref().unwrap_or("<none>");
        println!(
            "--- sample #{} ---\n\
             subject  : {:?}\n\
             relation : {:?}\n\
             object   : {:?}\n\
             corpus   : {}\n\
             run_id   : {}\n\
             url      : {}\n",
            i + 1, subj, bond_type, obj, corpus, run_id_s, url_s
        );
    }

    let all_pass = pass_distinct && pass_no_dominator && pass_typed_min;
    if !all_pass {
        std::process::exit(3);
    }
}
