//! OP-COMPACT: SAFE major compaction driver -- flattens resident RAM by
//! collapsing the accumulated, mutually-overlapping L1 segments of BOTH
//! the node store and the bond store into a SINGLE L1 segment each,
//! deduping cross-segment duplicates (higher seq wins) and GC'ing
//! tombstones, via the PROVEN production `compact()` engine.
//!
//! Why same-level (L1 -> L1) and not L1 -> L2: the canonical brain
//! ALREADY has an occupied L2 (a prior major merge). The read path
//! resolves a single binary-search candidate per level >= 1, so writing
//! a second L2 segment (what a naive L1 -> L2 merge would do while L2 is
//! occupied) would leave two OVERLAPPING L2 segments -> silent recall
//! loss. Collapsing L1 into one L1 segment never creates a second
//! segment at any level, keeps the L1+ non-overlap invariant, preserves
//! recency (L1 stays newer than the untouched L2), and still removes the
//! duplicate-segment RAM/disk bloat. Proven lossless by the
//! `op_compact_*_same_level_tests` unit tests.
//!
//! LSM-safe + verify-before-reclaim: writes NEW segments, commits the
//! MANIFEST atomically (save_atomic), and by DEFAULT does NOT unlink the
//! retired source segments -- they stay on disk so a simple MANIFEST
//! restore fully rolls back with zero data loss. Pass --reclaim ONLY
//! after recall has been verified against the new manifest.
//!
//! Usage: major_compact_collapse <brain_dir> <keyfile> [--reclaim] [--node-only|--bond-only]

use std::path::PathBuf;
use std::time::Instant;

use lu4d::bond_store::{
    cleanup_orphan_bond_ssts, BondCompactionPlan, BondCompactionWorker, BondSstReaderCache,
};
use lu4d::encrypt::MasterKey;
use lu4d::manifest::{Manifest, MANIFEST_FILE_NAME};
use lu4d::node_store::{
    cleanup_orphan_node_ssts, CompactionPlan, NodeCompactionWorker, NodeSstReaderCache,
};

fn node_census(m: &Manifest, tag: &str) {
    println!("--- node_levels {} ---", tag);
    for l in &m.node_levels {
        let b: u64 = l.ssts.iter().map(|s| s.file_size).sum();
        println!("  L{:<2} ssts={:<3} entries={:<12} {:.2} GiB", l.level, l.ssts.len(),
            l.ssts.iter().map(|s| s.entry_count).sum::<u64>(), b as f64 / 1073741824.0);
    }
}
fn bond_census(m: &Manifest, tag: &str) {
    println!("--- bond_v5_levels {} ---", tag);
    for l in &m.bond_v5_levels {
        let b: u64 = l.ssts.iter().map(|s| s.file_size).sum();
        println!("  L{:<2} ssts={:<3} entries={:<12} {:.2} GiB", l.level, l.ssts.len(),
            l.ssts.iter().map(|s| s.entry_count).sum::<u64>(), b as f64 / 1073741824.0);
    }
}
fn uids_at<F: Fn(&Manifest) -> Vec<(u8, Vec<u64>)>>(m: &Manifest, level: u8, f: F) -> Vec<u64> {
    f(m).into_iter().find(|(l, _)| *l == level).map(|(_, u)| u).unwrap_or_default()
}
fn node_level_uids(m: &Manifest) -> Vec<(u8, Vec<u64>)> {
    m.node_levels.iter().map(|l| (l.level, l.ssts.iter().map(|s| s.file_uid).collect())).collect()
}
fn bond_level_uids(m: &Manifest) -> Vec<(u8, Vec<u64>)> {
    m.bond_v5_levels.iter().map(|l| (l.level, l.ssts.iter().map(|s| s.file_uid).collect())).collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: major_compact_collapse <brain_dir> <keyfile> [--reclaim] [--node-only|--bond-only]");
        std::process::exit(2);
    }
    let brain_dir = PathBuf::from(&args[1]);
    let keyfile = PathBuf::from(&args[2]);
    let reclaim = args.iter().any(|a| a == "--reclaim");
    let node_only = args.iter().any(|a| a == "--node-only");
    let bond_only = args.iter().any(|a| a == "--bond-only");
    let do_node = !bond_only;
    let do_bond = !node_only;

    let manifest_path = brain_dir.join(MANIFEST_FILE_NAME);
    let node_dir = brain_dir.join("node_store");
    let bond_dir = brain_dir.join("bond_store");

    let mut m = Manifest::load(&manifest_path).expect("manifest io").expect("manifest present");
    let key = MasterKey::load_from_keyfile(&keyfile).expect("load master key");

    node_census(&m, "BEFORE");
    bond_census(&m, "BEFORE");

    if do_node {
        let ncache = NodeSstReaderCache::new(node_dir.clone()).with_key(Some(key.clone()));
        let nw = NodeCompactionWorker::new(node_dir.clone()).with_key(Some(key.clone()));
        // (1) minor: drain any L0 into L1 so all recent data joins the collapse.
        let l0 = uids_at(&m, 0, node_level_uids);
        if !l0.is_empty() {
            let t = Instant::now();
            let r = nw.compact(CompactionPlan { source_uids: l0.clone(), source_level: 0, target_level: 1 }, &mut m, Some(&ncache)).expect("node minor L0->L1");
            println!("NODE minor L0->L1: {} in -> {} out ({:.1}s)", l0.len(), r.entries_out, t.elapsed().as_secs_f64());
        }
        // (2) collapse: merge ALL L1 -> single L1 (dedup).
        let l1 = uids_at(&m, 1, node_level_uids);
        if l1.len() >= 2 {
            let t = Instant::now();
            let r = nw.compact(CompactionPlan { source_uids: l1.clone(), source_level: 1, target_level: 1 }, &mut m, Some(&ncache)).expect("node collapse L1->L1");
            println!("NODE collapse L1->L1: {} segs in -> uid {}, {} entries out, {} dedup drops, {} tomb GC ({:.1}s)",
                l1.len(), r.output_entry.file_uid, r.entries_out, r.entries_dropped_overwritten, r.entries_dropped_tombstone, t.elapsed().as_secs_f64());
        } else {
            println!("NODE collapse: L1 has {} seg(s), nothing to collapse", l1.len());
        }
    }

    if do_bond {
        let bcache = BondSstReaderCache::new(bond_dir.clone()).with_key(Some(key.clone()));
        let bw = BondCompactionWorker::new(bond_dir.clone()).with_key(Some(key.clone()));
        let l0 = uids_at(&m, 0, bond_level_uids);
        if !l0.is_empty() {
            let t = Instant::now();
            let r = bw.compact(BondCompactionPlan { source_uids: l0.clone(), source_level: 0, target_level: 1 }, &mut m, Some(&bcache)).expect("bond minor L0->L1");
            println!("BOND minor L0->L1: {} in -> {} out ({:.1}s)", l0.len(), r.entries_out, t.elapsed().as_secs_f64());
        }
        let l1 = uids_at(&m, 1, bond_level_uids);
        if l1.len() >= 2 {
            let t = Instant::now();
            let r = bw.compact(BondCompactionPlan { source_uids: l1.clone(), source_level: 1, target_level: 1 }, &mut m, Some(&bcache)).expect("bond collapse L1->L1");
            println!("BOND collapse L1->L1: {} segs in -> uid {}, {} entries out, {} dedup drops, {} tomb GC ({:.1}s)",
                l1.len(), r.output_entry.file_uid, r.entries_out, r.entries_dropped_overwritten, r.entries_dropped_tombstone, t.elapsed().as_secs_f64());
        } else {
            println!("BOND collapse: L1 has {} seg(s), nothing to collapse", l1.len());
        }
    }

    m.save_atomic(&manifest_path).expect("manifest save_atomic");
    println!("\nMANIFEST committed atomically.");

    if reclaim {
        let cn = cleanup_orphan_node_ssts(&node_dir, &m).expect("cleanup node");
        let cb = cleanup_orphan_bond_ssts(&bond_dir, &m).expect("cleanup bond");
        println!("RECLAIMED retired segments: node_orphans={:?} bond_orphans={:?}", cn, cb);
    } else {
        println!("reclaim SKIPPED (default): retired source segments left on disk.");
        println!("  -> verify recall against the new MANIFEST, then re-run with --reclaim,");
        println!("  -> or restore the pre-run MANIFEST backup to roll back with zero loss.");
    }

    node_census(&m, "AFTER");
    bond_census(&m, "AFTER");
}
