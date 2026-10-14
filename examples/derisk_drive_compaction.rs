//! Drive the REAL Phase 3.6 bond compactor (`BondCompactionWorker`)
//! over the v5 bond store to build the on-disk reverse index.
//!
//! At HEAD 750eb9c the production boot path (`load_v5` ->
//! `enable_compaction`) starts only the v4 `CompactionWorker`
//! (operates on `manifest.levels`, which is empty for a migrated v5
//! brain), so the v5 `BondCompactionWorker` is never scheduled at
//! boot. This driver invokes that SAME production compactor with the
//! canonical minor-compaction plan a scheduler would build:
//! source = every live L0 BondSST, source_level 0, target_level 1.
//! It writes a REAL primary `.bsst` + paired `.bsst.reverse` and
//! commits the manifest atomically. No fabricated files, no stubs.
//!
//! Usage: derisk_drive_compaction <brain_dir> <keyfile>

use std::path::PathBuf;
use std::time::Instant;

use lu4d::bond_store::{
    cleanup_orphan_bond_ssts, BondCompactionPlan, BondCompactionWorker, BondSstReaderCache,
};
use lu4d::encrypt::MasterKey;
use lu4d::manifest::{Manifest, MANIFEST_FILE_NAME};

fn print_structure(manifest: &Manifest, tag: &str) {
    println!("--- bond_v5_levels {} ---", tag);
    for lvl in &manifest.bond_v5_levels {
        let rev = lvl.ssts.iter().filter(|s| s.reverse_file_uid.is_some()).count();
        let bytes: u64 = lvl.ssts.iter().map(|s| s.file_size).sum();
        println!(
            "  L{:<2} ssts={:<3} with_reverse={:<3} entries={} bytes={:.2} GiB uids={:?} rev_uids={:?}",
            lvl.level,
            lvl.ssts.len(),
            rev,
            lvl.ssts.iter().map(|s| s.entry_count).sum::<u64>(),
            bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            lvl.ssts.iter().map(|s| s.file_uid).collect::<Vec<_>>(),
            lvl.ssts.iter().map(|s| s.reverse_file_uid).collect::<Vec<_>>(),
        );
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: derisk_drive_compaction <brain_dir> <keyfile>");
        std::process::exit(2);
    }
    let brain_dir = PathBuf::from(&args[1]);
    let keyfile = PathBuf::from(&args[2]);
    let manifest_path = brain_dir.join(MANIFEST_FILE_NAME);
    let bond_dir = brain_dir.join("bond_store");

    let mut manifest = Manifest::load(&manifest_path)
        .expect("manifest load io")
        .expect("manifest present");
    let key = MasterKey::load_from_keyfile(&keyfile).expect("load master key");

    print_structure(&manifest, "BEFORE");

    // Canonical minor-compaction plan: drain ALL live L0 SSTs -> L1.
    let l0_uids: Vec<u64> = manifest
        .bond_v5_levels
        .iter()
        .find(|l| l.level == 0)
        .map(|l| l.ssts.iter().map(|s| s.file_uid).collect())
        .unwrap_or_default();
    if l0_uids.is_empty() {
        eprintln!("no L0 SSTs to compact; nothing to do");
        std::process::exit(1);
    }
    println!("plan: source_uids={:?} source_level=0 target_level=1", l0_uids);

    let cache = BondSstReaderCache::new(bond_dir.clone()).with_key(Some(key.clone()));
    let worker = BondCompactionWorker::new(bond_dir.clone()).with_key(Some(key.clone()));
    let plan = BondCompactionPlan {
        source_uids: l0_uids,
        source_level: 0,
        target_level: 1,
    };

    let wall = Instant::now();
    let result = worker
        .compact(plan, &mut manifest, Some(&cache))
        .expect("compaction failed");
    let compact_wall = wall.elapsed();

    // Commit the manifest atomically, then unlink retired L0 orphans.
    manifest.save_atomic(&manifest_path).expect("manifest save_atomic");
    let cleanup = cleanup_orphan_bond_ssts(&bond_dir, &manifest).expect("cleanup");

    println!();
    println!("=== COMPACTION DONE ===");
    println!("wall_clock           = {:.2}s", compact_wall.as_secs_f64());
    println!("worker.elapsed       = {:.2}s", result.elapsed.as_secs_f64());
    println!("entries_in           = {}", result.entries_in);
    println!("entries_out          = {}", result.entries_out);
    println!("dropped_tombstone    = {}", result.entries_dropped_tombstone);
    println!("dropped_overwritten  = {}", result.entries_dropped_overwritten);
    println!(
        "output_entry         = uid {} reverse_uid {:?} entries {} size {:.2} GiB",
        result.output_entry.file_uid,
        result.output_entry.reverse_file_uid,
        result.output_entry.entry_count,
        result.output_entry.file_size as f64 / (1024.0 * 1024.0 * 1024.0),
    );
    println!("retired_inputs       = {} L0 ssts", result.input_entries_retired.len());
    println!("cleanup_report       = {:?}", cleanup);
    println!();
    print_structure(&manifest, "AFTER");
}
