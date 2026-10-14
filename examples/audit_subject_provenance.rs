//! BR.8 audit: sample N random Tier-2 bonds and dump
//! (subject, predicate, object, source) so an operator can manually
//! classify each subject as declarative / interrogative / ambiguous.
//!
//! Pass criteria (per BR.8 directive):
//!   - 0 (b) interrogative subjects out of 30  -- HARD pass.
//!   - >= 25 (a) declarative subjects.
//!   - <=  5 (c) ambiguous (lossy fragments).
//!
//! Usage:
//!   audit_subject_provenance <brain> <keyfile> [N=30] [seed=20260509]
//!
//! The bond filter accepts the canonical Tier-2 corpora stamped by
//! `_ingest_tier2` in scripts/ingest_curriculum_full.py:
//!   - HuggingFaceH4/no_robots
//!   - HuggingFaceH4/instruction-dataset
//!
//! Read-only -- never mutates the brain.

use lu4d::bond::BondType;
use lu4d::encrypt::MasterKey;
use lu4d::seed::Seed;
use lu4d::source::SourceKind;
use lu4d::{Brain, BrainLoadMode};

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
    let n: usize = args
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

    // Deterministic Fisher-Yates shuffle using a simple LCG seeded from
    // the user-supplied seed. Stable across runs for the same seed +
    // bond population, so an audit run is reproducible from the report.
    let mut indices: Vec<usize> = (0..total).collect();
    let mut rng_state: u64 = seed_arg.wrapping_add(0x9E37_79B9_7F4A_7C15);
    for i in (1..total).rev() {
        rng_state = rng_state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let j = (rng_state >> 33) as usize % (i + 1);
        indices.swap(i, j);
    }
    let take = n.min(total);

    println!(
        "== BR.8 Tier-2 subject-provenance audit ==\n\
         brain     : {}\n\
         total t2  : {}\n\
         sample n  : {} (rng_seed = {})\n",
        brain_path, total, take, seed_arg
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
             predicate: {:?}\n\
             object   : {:?}\n\
             corpus   : {}\n\
             run_id   : {}\n\
             url      : {}\n",
            i + 1, subj, bond_type, obj, corpus, run_id_s, url_s
        );
    }
}
