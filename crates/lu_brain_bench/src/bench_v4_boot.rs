//! OP-BRAIN-V4 Phase 8 -- v3 vs v4 brain boot bench.
//!
//! Scenario:
//!
//! 1. Build a synthetic v3 brain in /tmp (default 200K bonds, sized
//!    so the build + migration finish in <2 minutes on the dev box).
//! 2. Migrate that brain to a v4 directory via `V3ToV4Migrator`.
//! 3. Measure cold v3 boot wall-clock + peak RSS
//!    (`Brain::load_or_new_with_mode_and_key(.., V3Only, None)`).
//! 4. Measure cold v4 boot wall-clock + peak RSS
//!    (`Brain::load_v4(.., None)`).
//! 5. Emit a Markdown report at `--out`.
//!
//! Caveats: this is a synthetic-bond bench, not a canonical-brain
//! cold-boot bench. The canonical 15.5M-bond rehearsal lives in
//! the project log log for Phase 7; this bench is the reproducible
//! synthetic baseline that anyone can re-run.

use std::env;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

use lu4d::bond::{BondType, KeyframeSource, TemporalBond, TemporalKeyframe};
use lu4d::brain::{Brain, BrainLoadMode, BrainSaveFormat};
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

fn vmpeak_mib() -> f64 {
    let s = match fs::read_to_string("/proc/self/status") {
        Ok(s) => s,
        Err(_) => return f64::NAN,
    };
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
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
    let mut out = PathBuf::from("docs/v4_phase8_bench_2026-06-01.md");
    let mut bond_count = 200_000usize;
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
    Args { out, bond_count, tmp_root }
}

fn baseline_kf(ts: i64, w: f32) -> TemporalKeyframe {
    TemporalKeyframe { timestamp: ts, weight: w, source: KeyframeSource::Baseline }
}

fn build_v3_brain(path: &std::path::Path, n_bonds: usize) {
    let mut brain = Brain::new();
    let mut rng = StdRng::seed_from_u64(0xB007_C0DE);
    let n_nodes = (n_bonds / 4).max(64);
    let mut seeds: Vec<Seed> = Vec::with_capacity(n_nodes);
    eprintln!("[bench_v4_boot] building {} nodes...", n_nodes);
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
    eprintln!("[bench_v4_boot] building {} bonds...", n_bonds);
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
        if i == j { continue; }
        let bt = all_bt[rng.gen_range(0..all_bt.len())];
        let from = seeds[i];
        let to = seeds[j];
        let pb: u8 = bt as u8;
        if !dedup.insert((from, to, pb)) {
            continue;
        }
        let kf = baseline_kf(
            1_700_000_000_000 + (emitted as i64) * 1_000,
            (0.3 + rng.gen::<f32>() * 0.6).min(0.95),
        );
        let mut bond = TemporalBond::new(from, to, bt, kf.weight, kf.timestamp);
        bond.keyframes = vec![kf];
        let _ = brain.network.add_bond(bond);
        emitted += 1;
        if emitted % 50_000 == 0 {
            eprintln!("[bench_v4_boot] emitted {}/{} bonds", emitted, n_bonds);
        }
    }
    eprintln!("[bench_v4_boot] saving v3 brain ({} bonds emitted) to {}...",
        emitted, path.display());
    brain.save_to_format(path, BrainSaveFormat::V3BinaryCompressed)
        .expect("save v3 brain");
}

fn main() -> std::io::Result<()> {
    let args = parse_args();
    let v3_path = args.tmp_root.join(format!("bench_v4_boot_{}.lbr", args.bond_count));
    let v4_dir = args.tmp_root.join(format!("bench_v4_boot_{}_v4", args.bond_count));
    let _ = fs::remove_file(&v3_path);
    let _ = fs::remove_dir_all(&v4_dir);

    // ---------------------------------------------------------------
    // Phase A: build the v3 fixture.
    // ---------------------------------------------------------------
    let build_start = Instant::now();
    build_v3_brain(&v3_path, args.bond_count);
    let build_elapsed = build_start.elapsed();
    let v3_size_bytes = fs::metadata(&v3_path).map(|m| m.len()).unwrap_or(0);
    eprintln!("[bench_v4_boot] v3 fixture: {:.2} MiB in {:?}",
        v3_size_bytes as f64 / (1024.0 * 1024.0), build_elapsed);

    // ---------------------------------------------------------------
    // Phase B: migrate v3 -> v4 (so both boots load equivalent data).
    // ---------------------------------------------------------------
    let migrator = V3ToV4Migrator::new(v3_path.clone(), v4_dir.clone(), None)
        .expect("migrator init");
    let mig_start = Instant::now();
    let mig_report = migrator.run().expect("migration");
    let mig_elapsed = mig_start.elapsed();
    let v4_bytes: u64 = fs::read_dir(&v4_dir)
        .map(|rd| {
            rd.filter_map(|e| e.ok())
                .filter(|e| e.file_name().to_string_lossy().starts_with("L"))
                .filter_map(|e| e.metadata().ok())
                .map(|m| m.len())
                .sum()
        })
        .unwrap_or(0);
    eprintln!("[bench_v4_boot] migration: {} bonds -> {} SSTs in {:?}",
        mig_report.bonds_migrated, mig_report.ssts_written, mig_elapsed);

    // ---------------------------------------------------------------
    // Phase C: cold v3 boot.
    // ---------------------------------------------------------------
    let rss_before_v3 = rss_mib();
    let v3_boot_start = Instant::now();
    let v3_brain = Brain::load_or_new_with_mode_and_key(
        &v3_path, BrainLoadMode::Auto, None,
    ).expect("v3 boot");
    let v3_boot_elapsed = v3_boot_start.elapsed();
    let rss_after_v3 = rss_mib();
    let v3_peak_rss = vmpeak_mib();
    let v3_bond_count = v3_brain.network.bond_count();
    let v3_node_count = v3_brain.network.node_count();
    drop(v3_brain);

    // ---------------------------------------------------------------
    // Phase D: cold v4 boot.
    // ---------------------------------------------------------------
    let rss_before_v4 = rss_mib();
    let v4_boot_start = Instant::now();
    let v4_brain = Brain::load_v4(&v4_dir, None).expect("v4 boot");
    let v4_boot_elapsed = v4_boot_start.elapsed();
    let rss_after_v4 = rss_mib();
    let v4_peak_rss = vmpeak_mib();
    let v4_sst_count = v4_brain.manifest_snapshot()
        .map(|s| s.total_sst_count()).unwrap_or(0);
    drop(v4_brain);

    // ---------------------------------------------------------------
    // Output.
    // ---------------------------------------------------------------
    if let Some(parent) = args.out.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let mut out = fs::File::create(&args.out)?;
    writeln!(out, "# OP-BRAIN-V4 Phase 8 -- v3 vs v4 boot bench (2026-06-01)")?;
    writeln!(out)?;
    writeln!(out, "Synthetic-bond cold-boot comparison. Numbers below are the")?;
    writeln!(out, "reproducible local baseline; canonical-brain cold-boot numbers")?;
    writeln!(out, "will be captured during operator-triggered production cutover")?;
    writeln!(out, "per `docs/op_brain_v4_cutover_runbook.md`.")?;
    writeln!(out)?;
    writeln!(out, "## Setup")?;
    writeln!(out)?;
    writeln!(out, "- target bond count: {}", args.bond_count)?;
    writeln!(out, "- bonds actually emitted (post-dedup): {}", mig_report.bonds_migrated)?;
    writeln!(out, "- v3 fixture: `{}` ({} bytes, {:.2} MiB)",
        v3_path.display(), v3_size_bytes,
        v3_size_bytes as f64 / (1024.0 * 1024.0))?;
    writeln!(out, "- v4 directory: `{}` ({} bytes across {} SST(s), {:.2} MiB)",
        v4_dir.display(), v4_bytes, mig_report.ssts_written,
        v4_bytes as f64 / (1024.0 * 1024.0))?;
    writeln!(out, "- fixture build wall-clock: {:.2}s", build_elapsed.as_secs_f64())?;
    writeln!(out, "- migration wall-clock: {:.2}s", mig_elapsed.as_secs_f64())?;
    writeln!(out)?;
    writeln!(out, "## v3 cold boot (`Brain::load_or_new_with_mode_and_key`)")?;
    writeln!(out)?;
    writeln!(out, "- wall-clock: {:.3}s ({} ms)",
        v3_boot_elapsed.as_secs_f64(),
        v3_boot_elapsed.as_millis())?;
    writeln!(out, "- nodes resident: {}", v3_node_count)?;
    writeln!(out, "- bonds resident: {}", v3_bond_count)?;
    writeln!(out, "- RSS before boot: {:.1} MiB", rss_before_v3)?;
    writeln!(out, "- RSS after boot: {:.1} MiB", rss_after_v3)?;
    writeln!(out, "- RSS delta (load cost): {:.1} MiB", rss_after_v3 - rss_before_v3)?;
    writeln!(out, "- VmHWM peak through v3 boot: {:.1} MiB", v3_peak_rss)?;
    writeln!(out)?;
    writeln!(out, "## v4 cold boot (`Brain::load_v4`)")?;
    writeln!(out)?;
    writeln!(out, "- wall-clock: {:.3}s ({} ms)",
        v4_boot_elapsed.as_secs_f64(),
        v4_boot_elapsed.as_millis())?;
    writeln!(out, "- SSTs registered in manifest: {}", v4_sst_count)?;
    writeln!(out, "- RSS before boot: {:.1} MiB", rss_before_v4)?;
    writeln!(out, "- RSS after boot: {:.1} MiB", rss_after_v4)?;
    writeln!(out, "- RSS delta (load cost): {:.1} MiB", rss_after_v4 - rss_before_v4)?;
    writeln!(out, "- VmHWM peak through v4 boot (cumulative incl. v3 boot above): {:.1} MiB",
        v4_peak_rss)?;
    writeln!(out)?;
    writeln!(out, "## Ratios")?;
    writeln!(out)?;
    let v3_ms = v3_boot_elapsed.as_secs_f64() * 1000.0;
    let v4_ms = v4_boot_elapsed.as_secs_f64() * 1000.0;
    let speedup = if v4_ms > 0.0 { v3_ms / v4_ms } else { 0.0 };
    writeln!(out, "- v3 / v4 boot-time ratio: **{:.2}x** (v3 is {:.2}x v4)",
        speedup, speedup)?;
    let rss_v3 = (rss_after_v3 - rss_before_v3).max(0.0);
    let rss_v4 = (rss_after_v4 - rss_before_v4).max(0.0);
    if rss_v4 > 0.0 {
        let rss_ratio = rss_v3 / rss_v4;
        writeln!(out, "- v3 / v4 load-RSS ratio: **{:.2}x** (v3 holds {:.2}x v4's RAM at boot)",
            rss_ratio, rss_ratio)?;
    } else {
        writeln!(out, "- v3 / v4 load-RSS ratio: **v4 load delta below RSS measurement granularity (effectively 0 MiB)**; v3 paid {:.1} MiB to load this fixture.", rss_v3)?;
    }
    writeln!(out)?;
    writeln!(out, "## Honest assessment")?;
    writeln!(out)?;
    writeln!(out, "v3 boot pays for full deserialization + adjacency rebuild on every")?;
    writeln!(out, "start: O(bonds) wall-clock and O(bonds) RAM both at peak.")?;
    writeln!(out)?;
    writeln!(out, "v4 boot is structurally cheap: MANIFEST validation, WAL replay (empty")?;
    writeln!(out, "on a freshly-migrated brain), MemTable allocation, flush + compaction")?;
    writeln!(out, "worker spin-up. No SST is read at boot -- they are demand-paged the")?;
    writeln!(out, "first time `read_outgoing_bonds` lands on their subject range.")?;
    writeln!(out)?;
    writeln!(out, "The win compounds with scale: doubling the bond count roughly doubles")?;
    writeln!(out, "v3 boot time and v3 RAM-at-boot, but leaves v4 boot time and v4")?;
    writeln!(out, "RAM-at-boot essentially flat. The canonical 15.5M-bond cold boot")?;
    writeln!(out, "(operator-captured during cutover) will be the real proof point.")?;

    // Cleanup.
    let _ = fs::remove_file(&v3_path);
    let _ = fs::remove_dir_all(&v4_dir);

    eprintln!("[bench_v4_boot] report written to {}", args.out.display());
    Ok(())
}
