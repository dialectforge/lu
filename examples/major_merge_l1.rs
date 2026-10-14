//! One-shot MAJOR compaction proof: merge ALL live L1 SSTs -> L2 for
//! both the bond store and the node store, with cross-segment dedup.
//!
//! v5 ships only MINOR compaction (L0->L1). L1 accumulates unbounded:
//! the same hub node is re-emitted into every L1 NodeSST, bloating
//! node_store on disk and keeping per-segment index/bloom metadata
//! resident -> non-flat RAM. This driver invokes the SAME production
//! compactor (BondCompactionWorker / NodeCompactionWorker) the minor
//! scheduler uses, but over source_level=1 -> target_level=2, merging
//! every L1 segment into one L2 segment and dropping cross-segment
//! duplicates. No fabricated files, no stubs; manifest committed
//! atomically, retired sources unlinked only after the commit.
//!
//! Usage: major_merge_l1 <brain_dir> <keyfile>

use std::path::PathBuf;
use std::time::Instant;

use lu4d::bond_store::{
    cleanup_orphan_bond_ssts, BondCompactionPlan, BondCompactionWorker, BondSstReaderCache,
};
use lu4d::node_store::{
    cleanup_orphan_node_ssts, CompactionPlan, NodeCompactionWorker, NodeSstReaderCache,
};
use lu4d::encrypt::MasterKey;
use lu4d::manifest::{Manifest, MANIFEST_FILE_NAME};

fn print_bond(manifest: &Manifest, tag: &str) {
    println!("--- bond_v5_levels {} ---", tag);
    for lvl in &manifest.bond_v5_levels {
        let bytes: u64 = lvl.ssts.iter().map(|s| s.file_size).sum();
        println!(
            "  L{:<2} ssts={:<3} entries={} bytes={:.2} GiB",
            lvl.level,
            lvl.ssts.len(),
            lvl.ssts.iter().map(|s| s.entry_count).sum::<u64>(),
            bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        );
    }
}

fn print_node(manifest: &Manifest, tag: &str) {
    println!("--- node_levels {} ---", tag);
    for lvl in &manifest.node_levels {
        let bytes: u64 = lvl.ssts.iter().map(|s| s.file_size).sum();
        println!(
            "  L{:<2} ssts={:<3} entries={} bytes={:.2} GiB",
            lvl.level,
            lvl.ssts.len(),
            lvl.ssts.iter().map(|s| s.entry_count).sum::<u64>(),
            bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        );
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: major_merge_l1 <brain_dir> <keyfile>");
        std::process::exit(2);
    }
    let brain_dir = PathBuf::from(&args[1]);
    let keyfile = PathBuf::from(&args[2]);
    let manifest_path = brain_dir.join(MANIFEST_FILE_NAME);
    let bond_dir = brain_dir.join("bond_store");
    let node_dir = brain_dir.join("node_store");

    let mut manifest = Manifest::load(&manifest_path)
        .expect("manifest load io")
        .expect("manifest present");
    let key = MasterKey::load_from_keyfile(&keyfile).expect("load master key");

    print_bond(&manifest, "BEFORE");
    print_node(&manifest, "BEFORE");

    // ---- BOND: merge all live L1 -> L2 ----
    let l1_bond: Vec<u64> = manifest
        .bond_v5_levels
        .iter()
        .find(|l| l.level == 1)
        .map(|l| l.ssts.iter().map(|s| s.file_uid).collect())
        .unwrap_or_default();
    if l1_bond.len() >= 2 {
        println!("\nBOND major: merging {} L1 SSTs -> L2", l1_bond.len());
        let cache = BondSstReaderCache::new(bond_dir.clone()).with_key(Some(key.clone()));
        let worker = BondCompactionWorker::new(bond_dir.clone()).with_key(Some(key.clone()));
        let plan = BondCompactionPlan {
            source_uids: l1_bond,
            source_level: 1,
            target_level: 2,
        };
        let t = Instant::now();
        let res = worker
            .compact(plan, &mut manifest, Some(&cache))
            .expect("bond major compaction failed");
        println!("BOND major done in {:.1}s -> {:?}", t.elapsed().as_secs_f64(), res);
    } else {
        println!("BOND: <2 L1 SSTs, skipping");
    }

    // ---- NODE: merge all live L1 -> L2 ----
    let l1_node: Vec<u64> = manifest
        .node_levels
        .iter()
        .find(|l| l.level == 1)
        .map(|l| l.ssts.iter().map(|s| s.file_uid).collect())
        .unwrap_or_default();
    if l1_node.len() >= 2 {
        println!("\nNODE major: merging {} L1 SSTs -> L2", l1_node.len());
        let cache = NodeSstReaderCache::new(node_dir.clone()).with_key(Some(key.clone()));
        let worker = NodeCompactionWorker::new(node_dir.clone()).with_key(Some(key.clone()));
        let plan = CompactionPlan {
            source_uids: l1_node,
            source_level: 1,
            target_level: 2,
        };
        let t = Instant::now();
        let res = worker
            .compact(plan, &mut manifest, Some(&cache))
            .expect("node major compaction failed");
        println!("NODE major done in {:.1}s -> {:?}", t.elapsed().as_secs_f64(), res);
    } else {
        println!("NODE: <2 L1 SSTs, skipping");
    }

    // ---- Commit + cleanup ----
    manifest.save_atomic(&manifest_path).expect("manifest save_atomic");
    let cb = cleanup_orphan_bond_ssts(&bond_dir, &manifest).expect("cleanup bond");
    let cn = cleanup_orphan_node_ssts(&node_dir, &manifest).expect("cleanup node");
    println!("\ncommitted. cleaned bond_orphans={:?} node_orphans={:?}", cb, cn);

    print_bond(&manifest, "AFTER");
    print_node(&manifest, "AFTER");
}
