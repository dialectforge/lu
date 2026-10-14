//! TRUE read-only BondType census straight off the v5 bond_store SSTs.
//! Opens each forward *.bsst with BondSstReader (no Brain load, no
//! flush/compaction workers -> never mutates the brain) and tallies
//! the predicate byte (== BondType discriminant, memtable::predicate_discriminant).
//! Usage: audit_bondtype_sst_ro <bond_store_dir> <keyfile>
use lu4d::bond_store::sst_reader::BondSstReader;
use lu4d::encrypt::MasterKey;
use std::path::Path;

const NAMES: [&str; 29] = [
    "IsA","HasProperty","Causes","PartOf","UsedFor","RelatedTo","ContrastedWith",
    "DerivedFrom","Requires","Produces","Opposes","BornIn","BornOn","DiedIn","DiedOn",
    "DiscoveredBy","FoundedBy","FoundedIn","HasAltitude","HasAspect","HasCapital",
    "HasOfficialLanguage","HasPopulation","KnownFor","LocatedIn","MeasuredIn",
    "NamedAfter","WrittenBy","HasArea",
];

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = args.next().expect("bond_store dir required");
    let keyfile = args.next().expect("keyfile required");
    let key = MasterKey::load_from_keyfile(&keyfile).expect("load keyfile");

    let mut files: Vec<_> = std::fs::read_dir(&dir).expect("read_dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map(|x| x == "bsst").unwrap_or(false))
        .collect();
    files.sort();
    eprintln!("forward .bsst files: {}", files.len());

    let mut hist = [0u64; 29];
    let mut other = 0u64;
    let mut total = 0u64;
    for (i, f) in files.iter().enumerate() {
        let r = BondSstReader::open_with_key(f, Some(key.clone())).expect("open bsst");
        let mut n = 0u64;
        for item in r.iter_all() {
            let (_s, pred, _o, _payload) = item.expect("iter");
            if (pred as usize) < 29 { hist[pred as usize] += 1; } else { other += 1; }
            n += 1; total += 1;
        }
        eprintln!("  [{}/{}] {} -> {} bonds", i+1, files.len(), f.file_name().unwrap().to_string_lossy(), n);
    }

    println!("=== BondType census (forward SSTs, read-only) ===");
    println!("total_bonds={}", total);
    let mut idx: Vec<usize> = (0..29).collect();
    idx.sort_by(|a,b| hist[*b].cmp(&hist[*a]));
    for i in idx {
        if hist[i]==0 { continue; }
        println!("  {:<22} {:>12} {:.3}%", NAMES[i], hist[i], hist[i] as f64/total as f64*100.0);
    }
    if other>0 { println!("  {:<22} {:>12}", "<out-of-range>", other); }
    let isa = hist[0]; let related = hist[5];
    let generic = isa + related;
    let typed = total - generic - other;
    println!("--- summary ---");
    println!("RelatedTo={} ({:.3}%)", related, related as f64/total as f64*100.0);
    println!("IsA={} ({:.3}%)", isa, isa as f64/total as f64*100.0);
    println!("generic(RelatedTo+IsA)={} ({:.3}%)", generic, generic as f64/total as f64*100.0);
    println!("typed_specific={} ({:.3}%)", typed, typed as f64/total as f64*100.0);
    let variants = hist.iter().filter(|c| **c>0).count();
    println!("variants_present={}/29", variants);
}
