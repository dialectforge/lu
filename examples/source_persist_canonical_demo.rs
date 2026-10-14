//! Port Source-Persist empirical demo. Two invocations:
//!   1. `bake <brain> <key>` — capture bond_count baseline, teach a unique
//!      corpus bond tagged `source-persist-demo/<run-id>`, save.
//!   2. `verify <brain> <key> <corpus> <baseline>` — fresh process, load,
//!      dry_run revoke (expect matched_count=1), real revoke, confirm
//!      bond_count returns to baseline.
//!
//! The cross-process boundary is the load — phase 2 runs `Brain::load_*`
//! against the file written by phase 1, which is the production save→reload
//! cycle exactly. Run sequentially against a /tmp copy of the canonical
//! v3enc brain to prove Source-Persist works on the real data shape.

use lu4d::brain::{Brain, BrainLoadMode, BrainSaveFormat};
use lu4d::bond::BondType;
use lu4d::Confidence;
use lu4d::encrypt::MasterKey;
use std::path::Path;

fn read_master_key(p: &str) -> MasterKey {
    let bytes = std::fs::read(p).expect("read master key");
    assert_eq!(bytes.len(), 32, "master key must be 32 bytes");
    let mut arr = [0u8; 32];
    arr.copy_from_slice(&bytes);
    MasterKey::from_bytes(arr)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("usage:");
        eprintln!("  {} bake <brain.lbr> <key>", args[0]);
        eprintln!("  {} verify <brain.lbr> <key> <corpus-tag> <baseline-bond-count>", args[0]);
        std::process::exit(2);
    }
    let cmd = args[1].as_str();
    let brain_path = Path::new(&args[2]);
    let key = read_master_key(&args[3]);

    match cmd {
        "bake" => {
            let mut brain = Brain::load_or_new_with_mode_and_key(
                brain_path, BrainLoadMode::Auto, Some(key),
            ).expect("load brain");
            let baseline = brain.network.bond_count();
            let run_id = format!("sp-demo-{}", std::process::id());
            let corpus = format!("source-persist-demo/{}", run_id);
            println!("BAKE: baseline_bond_count={}", baseline);
            println!("BAKE: corpus_tag={}", corpus);
            let r = brain.teach_corpus(
                &format!("sp_demo_subj_{}", run_id),
                BondType::IsA,
                &format!("sp_demo_obj_{}", run_id),
                0.5, Confidence::Verified,
                &corpus, Some(run_id.clone()), None,
            );
            println!("BAKE: teach_corpus -> created_subj={} created_obj={}",
                r.created_subject, r.created_object);
            let post_teach = brain.network.bond_count();
            assert_eq!(post_teach, baseline + 1,
                "bond_count must increment by exactly 1");
            println!("BAKE: post_teach_bond_count={}", post_teach);
            // Force v3enc save explicitly; the loader path with Some(key)
            // already configured save_format=V3BinaryEncrypted but we are
            // explicit for clarity.
            brain.save_to_format(brain_path, BrainSaveFormat::V3BinaryEncrypted)
                .expect("save v3enc");
            println!("BAKE: saved v3enc");
            println!("BAKE_OK baseline={} corpus={}", baseline, corpus);
        }
        "verify" => {
            let corpus_tag = args[4].clone();
            let baseline: usize = args[5].parse().expect("baseline usize");
            let mut brain = Brain::load_or_new_with_mode_and_key(
                brain_path, BrainLoadMode::Auto, Some(key),
            ).expect("load brain");
            let pre = brain.network.bond_count();
            println!("VERIFY: post_load_bond_count={} (expected baseline+1={})", pre, baseline + 1);
            assert_eq!(pre, baseline + 1,
                "bond from BAKE phase must survive reload");

            // Dry-run revoke first
            let predicate = |attr: &lu4d::source::SourceAttribution| -> bool {
                matches!(
                    &attr.kind,
                    lu4d::source::SourceKind::TrainingCorpus { corpus, .. }
                        if corpus == &corpus_tag
                )
            };
            let dry_match = brain.source_overlay.read().unwrap().entries.iter()
                .filter(|(_, attr)| predicate(attr))
                .count();
            println!("VERIFY: dry_run match_count={} (expected 1)", dry_match);
            assert_eq!(dry_match, 1,
                "overlay must contain the BAKE corpus tag after reload");

            // Real revoke
            let (overlay_removed, bonds_removed) = brain.revoke_bonds_by_source(predicate);
            println!("VERIFY: revoke overlay_removed={} bonds_removed={}",
                overlay_removed, bonds_removed);
            assert_eq!(overlay_removed, 1, "overlay entry removed");
            assert_eq!(bonds_removed, 1, "underlying bond removed");

            let post = brain.network.bond_count();
            println!("VERIFY: post_revoke_bond_count={} (expected baseline={})", post, baseline);
            assert_eq!(post, baseline,
                "bond_count must return to the BAKE baseline post-revoke");
            brain.save_to_format(brain_path, BrainSaveFormat::V3BinaryEncrypted)
                .expect("save v3enc post-revoke");
            println!("VERIFY_OK net_delta_bonds=0 net_delta_overlay=0");
        }
        _ => {
            eprintln!("unknown command: {}", cmd);
            std::process::exit(2);
        }
    }
}
