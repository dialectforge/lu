//! De-risk harness: incoming-vs-outgoing v5 bond read benchmark.
//!
//! Read-only. Constructs a `Manifest` + keyed `BondSstReaderCache`
//! directly (NO `Brain`, so NO compaction/flush worker is started and
//! nothing on disk is mutated) and times the SAME public read-path
//! functions the prod recall path uses.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use lu4d::bond_store::{get_incoming_bonds_v5, get_outgoing_bonds_v5, BondSstReaderCache};
use lu4d::encrypt::MasterKey;
use lu4d::manifest::{Manifest, MANIFEST_FILE_NAME};
use lu4d::node_intern::NodeId;

fn pct(sorted_us: &[f64], p: f64) -> f64 {
    if sorted_us.is_empty() {
        return f64::NAN;
    }
    let rank = (p / 100.0) * ((sorted_us.len() - 1) as f64);
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    if lo == hi {
        sorted_us[lo]
    } else {
        let frac = rank - lo as f64;
        sorted_us[lo] * (1.0 - frac) + sorted_us[hi] * frac
    }
}

fn fmt_us(us: f64) -> String {
    if us >= 1_000_000.0 {
        format!("{:.3} s", us / 1_000_000.0)
    } else if us >= 1_000.0 {
        format!("{:.3} ms", us / 1_000.0)
    } else {
        format!("{:.1} us", us)
    }
}

fn report(label: &str, n_random: usize, n_hub: usize, mut times_us: Vec<f64>) {
    times_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let sum: f64 = times_us.iter().sum();
    let mean = sum / times_us.len() as f64;
    println!(
        "{:<28} n={} (rand={}, hub={})  mean={:<11} p50={:<11} p90={:<11} p99={:<11} max={}",
        label,
        times_us.len(),
        n_random,
        n_hub,
        fmt_us(mean),
        fmt_us(pct(&times_us, 50.0)),
        fmt_us(pct(&times_us, 90.0)),
        fmt_us(pct(&times_us, 99.0)),
        fmt_us(pct(&times_us, 100.0)),
    );
}

fn top_k(deg: &HashMap<u64, u32>, k: usize) -> Vec<u64> {
    let mut v: Vec<(u64, u32)> = deg.iter().map(|(a, b)| (*a, *b)).collect();
    v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    v.into_iter().take(k).map(|x| x.0).collect()
}

fn sample(keys: &[u64], hubs: &[u64], n_total: usize, state: &mut u64) -> (Vec<u64>, usize, usize) {
    let n_hub = hubs.len().min(n_total);
    let n_rand = n_total.saturating_sub(n_hub);
    let mut out: Vec<u64> = Vec::with_capacity(n_total);
    let mut seen = std::collections::HashSet::new();
    let mut tries = 0;
    while out.len() < n_rand && tries < n_rand * 50 && !keys.is_empty() {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        let idx = (z % keys.len() as u64) as usize;
        let id = keys[idx];
        if seen.insert(id) {
            out.push(id);
        }
        tries += 1;
    }
    let rand_count = out.len();
    for &h in hubs.iter().take(n_hub) {
        out.push(h);
    }
    (out, rand_count, n_hub)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: derisk_reverse_bench <brain_dir> <keyfile> [incoming_n] [outgoing_n] [seed]");
        std::process::exit(2);
    }
    let brain_dir = PathBuf::from(&args[1]);
    let keyfile = PathBuf::from(&args[2]);
    let inc_n: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(30);
    let out_n: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(220);
    let seed: u64 = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(0xDF_DF_DF_DF);

    let manifest_path = brain_dir.join(MANIFEST_FILE_NAME);
    let manifest = Manifest::load(&manifest_path)
        .expect("manifest load io")
        .expect("manifest present");
    let key = MasterKey::load_from_keyfile(&keyfile).expect("load master key");
    let bond_dir = brain_dir.join("bond_store");
    let cache = BondSstReaderCache::new(bond_dir).with_key(Some(key));

    println!("=== bond_v5_levels structure: {} ===", brain_dir.display());
    let mut total_ssts = 0usize;
    let mut total_reverse = 0usize;
    for lvl in &manifest.bond_v5_levels {
        let n = lvl.ssts.len();
        let rev = lvl.ssts.iter().filter(|s| s.reverse_file_uid.is_some()).count();
        let bytes: u64 = lvl.ssts.iter().map(|s| s.file_size).sum();
        total_ssts += n;
        total_reverse += rev;
        println!(
            "  L{:<2} ssts={:<3} with_reverse={:<3} bytes={:.2} GiB",
            lvl.level,
            n,
            rev,
            bytes as f64 / (1024.0 * 1024.0 * 1024.0)
        );
    }
    println!(
        "  TOTAL ssts={} with_reverse={} ({} reverse-less)",
        total_ssts,
        total_reverse,
        total_ssts - total_reverse
    );

    let mut out_deg: HashMap<u64, u32> = HashMap::new();
    let mut in_deg: HashMap<u64, u32> = HashMap::new();
    let scan_start = Instant::now();
    let mut total_rows: u64 = 0;
    for lvl in &manifest.bond_v5_levels {
        for sst in &lvl.ssts {
            let reader = cache
                .get_or_open_primary(sst.file_uid)
                .expect("open primary");
            for item in reader.iter_all() {
                let (subj, _pred, obj, _payload) = item.expect("primary row decode");
                *out_deg.entry(subj.0).or_insert(0) += 1;
                *in_deg.entry(obj.0).or_insert(0) += 1;
                total_rows += 1;
            }
        }
    }
    println!(
        "scan: {} bond rows in {:.2}s; distinct subjects={}, distinct objects={}",
        total_rows,
        scan_start.elapsed().as_secs_f64(),
        out_deg.len(),
        in_deg.len()
    );

    let out_keys: Vec<u64> = out_deg.keys().copied().collect();
    let in_keys: Vec<u64> = in_deg.keys().copied().collect();
    let out_hubs = top_k(&out_deg, 20);
    let in_hubs = top_k(&in_deg, 5.min(inc_n));

    let mut state = seed;
    let (out_sample, out_rand, out_hub) = sample(&out_keys, &out_hubs, out_n, &mut state);
    let (in_sample, in_rand, in_hub) = sample(&in_keys, &in_hubs, inc_n, &mut state);

    for &id in &out_sample {
        let _ = get_outgoing_bonds_v5(&manifest, &cache, NodeId(id))
            .collect::<Result<Vec<_>, _>>()
            .expect("outgoing ok");
    }
    let mut out_times = Vec::with_capacity(out_sample.len());
    for &id in &out_sample {
        let t = Instant::now();
        let r = get_outgoing_bonds_v5(&manifest, &cache, NodeId(id))
            .collect::<Result<Vec<_>, _>>()
            .expect("outgoing ok");
        out_times.push(t.elapsed().as_secs_f64() * 1e6);
        std::hint::black_box(r.len());
    }

    let mut in_times = Vec::with_capacity(in_sample.len());
    let mut hub_counts: Vec<(u64, usize)> = Vec::new();
    for (i, &id) in in_sample.iter().enumerate() {
        let t = Instant::now();
        let r = get_incoming_bonds_v5(&manifest, &cache, NodeId(id))
            .collect::<Result<Vec<_>, _>>()
            .expect("incoming ok");
        let dt = t.elapsed().as_secs_f64() * 1e6;
        in_times.push(dt);
        if i >= in_sample.len() - in_hub {
            hub_counts.push((id, r.len()));
        }
        std::hint::black_box(r.len());
    }

    println!();
    println!("=== RESULTS ({}) ===", brain_dir.display());
    report("get_outgoing_bonds_v5", out_rand, out_hub, out_times);
    report("get_incoming_bonds_v5", in_rand, in_hub, in_times);
    println!("incoming hub result-counts (node:bonds): {:?}", hub_counts);
}
