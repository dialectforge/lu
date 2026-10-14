//! STOP-POINT 1 read-only verifier for OP-WRITEBACK Phase 5.
//! Proves appending KeyframeSource::ConcludedFrom is backward-compatible:
//!  (1) rkyv archived layout invariant: prints size_of/align_of of
//!      ArchivedKeyframeSource + ArchivedTemporalKeyframe so before/after compare.
//!  (2) real-brain decode digest: decodes every bond in the canonical bond_store
//!      (bincode BondPayload path) into an order-sensitive FNV-1a digest over
//!      (subject,pred,object,weight,confidence,seq) + each keyframe
//!      (timestamp, weight bits, source Debug). Identical digest before/after
//!      == existing bonds decode byte-identically.
//! READ-ONLY: opens forward *.bsst via BondSstReader; never builds a Brain,
//! never flushes/compacts, never writes.
use lu4d::bond::KeyframeSource;
use lu4d::bond::{ArchivedKeyframeSource, ArchivedTemporalKeyframe};
use lu4d::bond_store::sst_reader::BondSstReader;
use lu4d::encrypt::MasterKey;
use std::collections::BTreeMap;
use std::mem::{align_of, size_of};

#[inline]
fn fnv(mut h: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn main() {
    let dir = std::env::args().nth(1).expect("bond_store dir");
    let keyfile = std::env::args().nth(2).unwrap_or_else(|| "/var/lib/lu/master.key".to_string());
    let key = MasterKey::load_from_keyfile(&keyfile).expect("load keyfile");

    println!("=== rkyv archived layout invariant ===");
    println!("size_of ArchivedKeyframeSource    = {}", size_of::<ArchivedKeyframeSource>());
    println!("align_of ArchivedKeyframeSource   = {}", align_of::<ArchivedKeyframeSource>());
    println!("size_of ArchivedTemporalKeyframe  = {}", size_of::<ArchivedTemporalKeyframe>());
    println!("align_of ArchivedTemporalKeyframe = {}", align_of::<ArchivedTemporalKeyframe>());
    println!("size_of KeyframeSource(heap)       = {}", size_of::<KeyframeSource>());

    let mut files: Vec<_> = std::fs::read_dir(&dir).expect("read_dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map(|x| x == "bsst").unwrap_or(false))
        .collect();
    files.sort();
    eprintln!("forward .bsst files: {}", files.len());

    let mut digest: u64 = 0xcbf29ce484222325;
    let mut bonds: u64 = 0;
    let mut kfs: u64 = 0;
    let mut census: BTreeMap<String, u64> = BTreeMap::new();

    for f in &files {
        let r = BondSstReader::open_with_key(f, Some(key.clone())).expect("open bsst");
        for item in r.iter_all() {
            let (s, pred, o, payload) = item.expect("iter");
            bonds += 1;
            digest = fnv(digest, &s.0.to_le_bytes());
            digest = fnv(digest, &[pred]);
            digest = fnv(digest, &o.0.to_le_bytes());
            digest = fnv(digest, &[payload.weight, payload.confidence]);
            digest = fnv(digest, &payload.seq.to_le_bytes());
            for kf in &payload.keyframes {
                kfs += 1;
                digest = fnv(digest, &kf.timestamp.to_le_bytes());
                digest = fnv(digest, &kf.weight.to_bits().to_le_bytes());
                let sb = format!("{:?}", kf.source).into_bytes();
                digest = fnv(digest, &sb);
                let name = String::from_utf8_lossy(&sb)
                    .split(|c: char| !c.is_alphanumeric())
                    .next().unwrap_or("").to_string();
                *census.entry(name).or_insert(0) += 1;
            }
        }
    }

    println!("=== canonical bond_store decode digest (read-only) ===");
    println!("bonds_decoded     = {}", bonds);
    println!("keyframes_decoded = {}", kfs);
    println!("decode_digest_fnv = {:#018x}", digest);
    println!("=== keyframe source census ===");
    for (k, v) in &census {
        println!("  {:<22} {}", k, v);
    }
}
