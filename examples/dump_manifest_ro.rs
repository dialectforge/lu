//! Read-only manifest dump: per-level node/bond SST entry counts + ranges.
use lu4d::manifest::Manifest;
use std::path::Path;
fn main() {
    let p = std::env::args().nth(1).expect("manifest path");
    let m = Manifest::load(Path::new(&p)).expect("load").expect("some");
    println!("schema_version={} format_tag={}", m.schema_version, m.format_tag);
    let mut node_total=0u64;
    println!("--- NODE levels ---");
    for lvl in &m.node_levels {
        let lt: u64 = lvl.ssts.iter().map(|s| s.entry_count).sum();
        node_total += lt;
        println!("L{} ssts={} entries={}", lvl.level, lvl.ssts.len(), lt);
    }
    println!("node_entries_total(all levels)={}", node_total);
    let mut bond_total=0u64;
    println!("--- BOND v5 levels ---");
    for lvl in &m.bond_v5_levels {
        let lt: u64 = lvl.ssts.iter().map(|s| s.entry_count).sum();
        bond_total += lt;
        println!("L{} ssts={} entries={}", lvl.level, lvl.ssts.len(), lt);
        for s in &lvl.ssts {
            println!("   uid={} entries={} subj=[{}..{}] size={}", s.file_uid, s.entry_count, s.first_subject, s.last_subject, s.file_size);
        }
    }
    println!("bond_entries_total(all levels)={}", bond_total);
}
