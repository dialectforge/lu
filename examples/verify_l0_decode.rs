//! Belt-and-suspenders read-only decode digest over the top-level L0-*.sst
//! (LSO4) files of a v5 brain. Opens each SST via the production SST reader
//! (mmap, read-only), iterates every BondView, and folds an order-sensitive
//! FNV-1a digest over (subject, bond_type, dir/flags/raw_conf, object, weight,
//! keyframes{timestamp,weight,source Debug}). Deliberately does NOT name the
//! ConcludedFrom variant, so the SAME source compiles on the parent commit
//! (no ConcludedFrom) and on the branch (ConcludedFrom present); identical
//! digest across the two builds == existing L0 bonds decode byte-identically.
use lu4d::bond::KeyframeSource as _;
use lu4d::encrypt::MasterKey;
use lu4d::sst::{BlockCache, SST};
use std::collections::BTreeMap;
use std::sync::Arc;

#[inline]
fn fnv(mut h: u64, b: &[u8]) -> u64 {
    for &x in b {
        h ^= x as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn main() {
    let dir = std::env::args().nth(1).expect("brain dir");
    let keyfile = std::env::args().nth(2).unwrap_or_else(|| "/var/lib/lu/master.key".into());
    let key = MasterKey::load_from_keyfile(&keyfile).expect("key");
    let mut files: Vec<_> = std::fs::read_dir(&dir)
        .expect("rd")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.file_name()
                .map(|n| {
                    let s = n.to_string_lossy();
                    s.starts_with("L0-") && s.ends_with(".sst")
                })
                .unwrap_or(false)
        })
        .collect();
    files.sort();
    eprintln!("L0 sst files: {}", files.len());
    let cache = Arc::new(BlockCache::new(256 * 1024 * 1024));
    let mut digest: u64 = 0xcbf29ce484222325;
    let mut bonds: u64 = 0;
    let mut kfs: u64 = 0;
    let mut errs: u64 = 0;
    let mut census: BTreeMap<String, u64> = BTreeMap::new();
    for f in &files {
        let sst = match SST::open(f, Some(key.clone()), cache.clone()) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("open {} err {}", f.display(), e);
                errs += 1;
                continue;
            }
        };
        for item in sst.iter_bonds() {
            match item {
                Ok(bv) => {
                    bonds += 1;
                    digest = fnv(digest, bv.subject.as_bytes());
                    digest = fnv(digest, format!("{:?}", bv.bond_type).as_bytes());
                    digest = fnv(digest, &[bv.direction as u8, bv.flags, bv.raw_confidence]);
                    digest = fnv(digest, bv.object.as_bytes());
                    digest = fnv(digest, &bv.weight.to_bits().to_le_bytes());
                    for kf in &bv.keyframes {
                        kfs += 1;
                        digest = fnv(digest, &kf.timestamp.to_le_bytes());
                        digest = fnv(digest, &kf.weight.to_bits().to_le_bytes());
                        let sb = format!("{:?}", kf.source).into_bytes();
                        digest = fnv(digest, &sb);
                        let name = String::from_utf8_lossy(&sb)
                            .split(|c: char| !c.is_alphanumeric())
                            .next()
                            .unwrap_or("")
                            .to_string();
                        *census.entry(name).or_insert(0) += 1;
                    }
                }
                Err(e) => {
                    errs += 1;
                    if errs < 5 {
                        eprintln!("iter err: {}", e);
                    }
                }
            }
        }
    }
    println!("=== L0-*.sst decode digest (read-only) ===");
    println!("l0_files          = {}", files.len());
    println!("bonds_decoded     = {}", bonds);
    println!("keyframes_decoded = {}", kfs);
    println!("decode_errors     = {}", errs);
    println!("decode_digest_fnv = {:#018x}", digest);
    println!("=== keyframe source census ===");
    for (k, v) in &census {
        println!("  {:<22} {}", k, v);
    }
}
