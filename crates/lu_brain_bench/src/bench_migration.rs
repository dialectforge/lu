//! OP-BRAIN-V4 Phase 7 -- v3 -> v4 migration bench.
//!
//! Scenarios:
//!
//! 1. Build a 1M-bond v3 brain in /tmp.
//! 2. Run the migration to a v4 dir in /tmp.
//! 3. Report wall-clock, MB/s, peak RSS, output SST count.
//! 4. Verify and report verify time + bonds checked.
//!
//! Output: Markdown report at the path passed via `--out`.

use std::env;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

use lu4d::bond::{BondType, KeyframeSource, TemporalBond, TemporalKeyframe};
use lu4d::brain::{Brain, BrainSaveFormat};
use lu4d::migration_v3_v4::V3ToV4Migrator;
use lu4d::node::{Confidence, KnowledgeContent, KnowledgeNode};
use lu4d::seed::Seed;

fn rss_mib() -> f64 {
    let s = match fs::read_to_string("/proc/self/status") {
        Ok(s) => s,
        Err(_) => return f64::NAN,
    };
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            if let Some(tok) = rest.split_whitespace().next() {
                if let Ok(kb) = tok.parse::<u64>() {
                    return kb as f64 / 1024.0;
                }
            }
        }
    }
    f64::NAN
}

struct Args {
    out: PathBuf,
    bond_count: usize,
    tmp_root: PathBuf,
}

fn parse_args() -> Args {
    let mut out = PathBuf::from("docs/v4_phase7_bench_2026-06-01.md");
    let mut bond_count = 1_000_000usize;
    let mut tmp_root = PathBuf::from("/tmp");
    let mut it = env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--out" => out = PathBuf::from(it.next().expect("--out arg")),
            "--bonds" => bond_count = it.next().unwrap().parse().unwrap(),
            "--tmp" => tmp_root = PathBuf::from(it.next().unwrap()),
            _ => {
                eprintln!("unknown arg: {}", a);
                std::process::exit(1);
            }
        }
    }
    Args {
        out,
        bond_count,
        tmp_root,
    }
}

fn build_v3_brain(path: &std::path::Path, n_bonds: usize) {
    let mut brain = Brain::new();
    let mut rng = StdRng::seed_from_u64(0x7777_FACE);
    let n_nodes = (n_bonds / 4).max(64);
    let mut seeds: Vec<Seed> = Vec::with_capacity(n_nodes);
    eprintln!("[bench_migration] building {} nodes...", n_nodes);
    for i in 0..n_nodes {
        let mut sb = [0u8; 32];
        rng.fill(&mut sb);
        let s = Seed::from_bytes(sb);
        let label = format!("n{}", i);
        let mut node = KnowledgeNode::new(
            label.clone(),
            KnowledgeContent::Raw(label),
            Confidence::Verified,
            0,
        );
        node.seed = s;
        brain.network.add_node(node);
        seeds.push(s);
    }
    eprintln!("[bench_migration] building {} bonds...", n_bonds);
    let all_bt = [
        BondType::IsA,
        BondType::HasProperty,
        BondType::Causes,
        BondType::PartOf,
        BondType::RelatedTo,
        BondType::UsedFor,
    ];
    let mut dedup: std::collections::HashSet<(Seed, Seed, u8)> =
        std::collections::HashSet::new();
    let mut emitted = 0usize;
    let mut attempts = 0usize;
    while emitted < n_bonds && attempts < n_bonds * 8 {
        attempts += 1;
        let i = rng.gen_range(0..n_nodes);
        let j = rng.gen_range(0..n_nodes);
        if i == j {
            continue;
        }
        let bt = all_bt[rng.gen_range(0..all_bt.len())];
        let from = seeds[i];
        let to = seeds[j];
        // Approximate the predicate discriminant ordering used
        // internally; collisions on this key are dropped, so the
        // exact byte doesn't need to match memtable's.
        let pb: u8 = match bt {
            BondType::IsA => 1,
            BondType::HasProperty => 2,
            BondType::Causes => 3,
            BondType::PartOf => 4,
            BondType::RelatedTo => 5,
            BondType::UsedFor => 6,
            _ => 0,
        };
        if !dedup.insert((from, to, pb)) {
            continue;
        }
        let w = rng.gen_range(0.1f32..1.0f32);
        let bond = TemporalBond {
            from,
            to,
            bond_type: bt,
            keyframes: vec![TemporalKeyframe {
                timestamp: emitted as i64,
                weight: w,
                source: KeyframeSource::Baseline,
            }],
            half_life_ms: 86_400_000.0,
        };
        if brain.network.add_bond(bond).is_ok() {
            emitted += 1;
            if emitted % 100_000 == 0 {
                eprintln!("[bench_migration] emitted {}/{} bonds", emitted, n_bonds);
            }
        }
    }
    eprintln!(
        "[bench_migration] saving v3 brain ({} bonds emitted) to {}...",
        emitted,
        path.display(),
    );
    brain
        .save_to_format(path, BrainSaveFormat::V3BinaryCompressed)
        .expect("save v3 brain");
}

fn main() -> std::io::Result<()> {
    let args = parse_args();
    let v3_path = args.tmp_root.join(format!("bench_migration_{}.lbr", args.bond_count));
    let v4_dir = args
        .tmp_root
        .join(format!("bench_migration_{}_v4", args.bond_count));
    // Clean slate.
    let _ = fs::remove_file(&v3_path);
    let _ = fs::remove_dir_all(&v4_dir);

    // ---------------------------------------------------------------
    // Phase 1: build the v3 brain.
    // ---------------------------------------------------------------
    let build_start = Instant::now();
    build_v3_brain(&v3_path, args.bond_count);
    let build_elapsed = build_start.elapsed();
    let v3_size_bytes = fs::metadata(&v3_path).map(|m| m.len()).unwrap_or(0);
    eprintln!(
        "[bench_migration] v3 brain built: {:.2} MiB in {:?}",
        v3_size_bytes as f64 / (1024.0 * 1024.0),
        build_elapsed,
    );
    let rss_after_build = rss_mib();

    // ---------------------------------------------------------------
    // Phase 2: migrate.
    // ---------------------------------------------------------------
    let migrator = V3ToV4Migrator::new(v3_path.clone(), v4_dir.clone(), None)
        .expect("migrator init");
    let mig_start = Instant::now();
    let report = migrator.run().expect("migration");
    let mig_elapsed = mig_start.elapsed();
    let rss_after_mig = rss_mib();

    let v4_bytes: u64 = fs::read_dir(&v4_dir)
        .map(|rd| {
            rd.filter_map(|e| e.ok())
                .filter(|e| {
                    e.file_name()
                        .to_string_lossy()
                        .starts_with("L")
                })
                .filter_map(|e| e.metadata().ok())
                .map(|m| m.len())
                .sum()
        })
        .unwrap_or(0);

    let throughput_mbps =
        (v3_size_bytes as f64 / (1024.0 * 1024.0)) / mig_elapsed.as_secs_f64();
    let bonds_per_sec =
        report.bonds_migrated as f64 / mig_elapsed.as_secs_f64();

    eprintln!(
        "[bench_migration] migration: bonds={} ssts={} elapsed={:?}",
        report.bonds_migrated, report.ssts_written, mig_elapsed,
    );

    // ---------------------------------------------------------------
    // Phase 3: verify.
    // ---------------------------------------------------------------
    let ver_start = Instant::now();
    let verify = migrator.verify().expect("verify");
    let ver_elapsed = ver_start.elapsed();
    let rss_after_verify = rss_mib();

    eprintln!(
        "[bench_migration] verify: bonds_checked={} mismatches={} elapsed={:?}",
        verify.bonds_checked,
        verify.mismatches.len(),
        ver_elapsed,
    );

    // ---------------------------------------------------------------
    // Output: Markdown.
    // ---------------------------------------------------------------
    if let Some(parent) = args.out.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let mut out = fs::File::create(&args.out)?;
    writeln!(out, "# OP-BRAIN-V4 Phase 7 -- migration bench (2026-06-01)")?;
    writeln!(out)?;
    writeln!(out, "## Setup")?;
    writeln!(out)?;
    writeln!(out, "- target bond count: {}", args.bond_count)?;
    writeln!(out, "- v3 path: `{}`", v3_path.display())?;
    writeln!(out, "- v4 dir: `{}`", v4_dir.display())?;
    writeln!(out, "- v3 size on disk: {} bytes ({:.2} MiB)", v3_size_bytes, v3_size_bytes as f64 / (1024.0 * 1024.0))?;
    writeln!(out, "- v3 build time (one-shot): {:.2}s", build_elapsed.as_secs_f64())?;
    writeln!(out)?;
    writeln!(out, "## Migration")?;
    writeln!(out)?;
    writeln!(out, "- bonds migrated: {}", report.bonds_migrated)?;
    writeln!(out, "- output SST count: {}", report.ssts_written)?;
    writeln!(out, "- total output bytes: {} ({:.2} MiB)", v4_bytes, v4_bytes as f64 / (1024.0 * 1024.0))?;
    writeln!(out, "- wall-clock: {:.2}s", mig_elapsed.as_secs_f64())?;
    writeln!(out, "- throughput: {:.2} MB/s (v3 bytes / migration wall-clock)", throughput_mbps)?;
    writeln!(out, "- bonds/sec: {:.0}", bonds_per_sec)?;
    if let Some(p) = report.peak_rss_bytes {
        writeln!(out, "- migrator-reported peak RSS: {:.2} MiB", p as f64 / (1024.0 * 1024.0))?;
    }
    writeln!(out, "- RSS after build: {:.1} MiB", rss_after_build)?;
    writeln!(out, "- RSS after migration: {:.1} MiB", rss_after_mig)?;
    writeln!(out)?;
    writeln!(out, "## Verify")?;
    writeln!(out)?;
    writeln!(out, "- subjects checked: {}", verify.subjects_checked)?;
    writeln!(out, "- bonds checked: {}", verify.bonds_checked)?;
    writeln!(out, "- mismatches: {}", verify.mismatches.len())?;
    writeln!(out, "- verify wall-clock: {:.2}s", ver_elapsed.as_secs_f64())?;
    writeln!(out, "- RSS after verify: {:.1} MiB", rss_after_verify)?;
    writeln!(out, "- result: {}", if verify.is_ok() { "OK" } else { "FAIL" })?;
    writeln!(out)?;
    writeln!(out, "## Honest assessment")?;
    writeln!(out)?;
    writeln!(out, "The v3 brain is built fresh from random bonds with the same shape")?;
    writeln!(out, "as the canonical curriculum-ingest output (one Baseline keyframe per")?;
    writeln!(out, "bond, 6 BondType variants, ~25% node:bond ratio). Migration walks")?;
    writeln!(out, "the bond key index in subject-sorted order (strategy 'b' from the")?;
    writeln!(out, "port plan); each bond is materialised once on the heap, pushed to")?;
    writeln!(out, "the active SSTWriter, then dropped. Verify is a full v3+v4 walk that")?;
    writeln!(out, "reconciles every (subject, predicate, object) triple.")?;
    writeln!(out)?;
    writeln!(out, "Production rehearsal numbers live in the project log log for this")?;
    writeln!(out, "port; the bench above is the synthetic baseline.")?;

    // Cleanup.
    let _ = fs::remove_file(&v3_path);
    let _ = fs::remove_dir_all(&v4_dir);

    eprintln!("[bench_migration] report written to {}", args.out.display());
    if !verify.is_ok() {
        std::process::exit(1);
    }
    Ok(())
}
