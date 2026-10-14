//! OP-BRAIN-V5 Phase 3.10 -- CLI front-end for v3 -> v5 BOND-SIDE migration.
//!
//! Thin shell over [`lu4d::migration_v3_v5_bond::migrate`]. All
//! pipeline logic + tests live in the library module so they ride
//! `cargo test -p lu4d --lib`. Mirrors the Phase 2.9 node CLI
//! (`lu_brain_migrate_v3_v5_node`).
//!
//! ## Ordering
//!
//! Run the Phase 2.9 NODE tool FIRST. This bond tool requires the
//! node-migrated `<v5>` dir to already hold `MANIFEST` +
//! `wal/nodeintern.bin` + `node_store/`; it loads the NodeIntern so
//! bonds reference the same NodeIds, then ADDS `bond_store/` and
//! updates the manifest's `bond_v5_levels[0]` in place.
//!
//! ## CLI
//!
//! ```text
//! lu_brain_migrate_v3_v5_bond --v3 <path> --v5 <node_migrated_dir>
//!                             [--keyfile <path>]
//!                             [--target-entries-per-block N]
//!                             [--target-entries-per-sst N]
//!                             [--sort-run-entries N]
//!                             [--skip-verify]
//!                             [--verify-sample-size N]
//!                             [--max-rss-mb N]
//!                             [--progress-every-n N]
//! ```
//!
//! ## Exit codes
//!
//! - `0` -- migration ran and verification passed (or `--skip-verify`).
//! - `1` -- migration ran but verification failed (data mismatch).
//! - `2` -- usage error / missing args / keyfile load error /
//!   v3 missing / bond_store already exists / v5 not node-migrated.
//! - `3` -- migration failed for any other reason (archive open,
//!   file IO, BondSST write, etc.).
//!
//! ## Production-brain procedure (NOT run by the agent)
//!
//! Snapshot `/var/lib/lu/brain.lbr` to a scratch path before running
//! (same discipline as Phase 2.9). NOTE: a migrated v5 brain is not
//! production-consumable until the separate loader-cutover phase adds
//! `Brain::load_v5`. The agent that built this binary did NOT execute
//! the real-brain migration.

use std::path::PathBuf;
use std::process::ExitCode;

use lu4d::encrypt::MasterKey;
use lu4d::migration_v3_v5_bond::{
    migrate, MigrationConfig, MigrationError, MigrationStats, VerificationReport,
    DEFAULT_PROGRESS_EVERY_N, DEFAULT_VERIFY_SAMPLE_SIZE,
};

struct Args {
    v3: PathBuf,
    v5: PathBuf,
    keyfile: Option<PathBuf>,
    target_entries_per_block: Option<usize>,
    target_entries_per_sst: Option<u64>,
    sort_run_entries: Option<u64>,
    skip_verify: bool,
    verify_sample_size: Option<usize>,
    max_rss_mb: Option<u64>,
    progress_every_n: Option<u64>,
}

fn print_usage_and_exit(code: u8) -> ! {
    eprintln!(
        "usage: lu_brain_migrate_v3_v5_bond --v3 <path> --v5 <node_migrated_dir>\n  \
         [--keyfile <path>] [--target-entries-per-block N] \
         [--target-entries-per-sst N] [--sort-run-entries N] [--skip-verify] \
         [--verify-sample-size N] [--max-rss-mb N] [--progress-every-n N]"
    );
    std::process::exit(code as i32);
}

fn parse_args() -> Args {
    let argv: Vec<String> = std::env::args().collect();
    let mut v3: Option<PathBuf> = None;
    let mut v5: Option<PathBuf> = None;
    let mut keyfile: Option<PathBuf> = None;
    let mut tepb: Option<usize> = None;
    let mut teps: Option<u64> = None;
    let mut sre: Option<u64> = None;
    let mut skip_verify = false;
    let mut vss: Option<usize> = None;
    let mut max_rss_mb: Option<u64> = None;
    let mut progress_every_n: Option<u64> = None;

    let mut i = 1usize;
    while i < argv.len() {
        match argv[i].as_str() {
            "--v3" => {
                require_next(&argv, i, "--v3");
                v3 = Some(PathBuf::from(&argv[i + 1]));
                i += 2;
            }
            "--v5" => {
                require_next(&argv, i, "--v5");
                v5 = Some(PathBuf::from(&argv[i + 1]));
                i += 2;
            }
            "--keyfile" => {
                require_next(&argv, i, "--keyfile");
                keyfile = Some(PathBuf::from(&argv[i + 1]));
                i += 2;
            }
            "--target-entries-per-block" => {
                require_next(&argv, i, "--target-entries-per-block");
                tepb = Some(parse_or_die::<usize>(&argv[i + 1], "--target-entries-per-block"));
                i += 2;
            }
            "--target-entries-per-sst" => {
                require_next(&argv, i, "--target-entries-per-sst");
                teps = Some(parse_or_die::<u64>(&argv[i + 1], "--target-entries-per-sst"));
                i += 2;
            }
            "--sort-run-entries" => {
                require_next(&argv, i, "--sort-run-entries");
                sre = Some(parse_or_die::<u64>(&argv[i + 1], "--sort-run-entries"));
                i += 2;
            }
            "--skip-verify" => {
                skip_verify = true;
                i += 1;
            }
            "--verify-sample-size" => {
                require_next(&argv, i, "--verify-sample-size");
                vss = Some(parse_or_die::<usize>(&argv[i + 1], "--verify-sample-size"));
                i += 2;
            }
            "--max-rss-mb" => {
                require_next(&argv, i, "--max-rss-mb");
                max_rss_mb = Some(parse_or_die::<u64>(&argv[i + 1], "--max-rss-mb"));
                i += 2;
            }
            "--progress-every-n" => {
                require_next(&argv, i, "--progress-every-n");
                progress_every_n = Some(parse_or_die::<u64>(&argv[i + 1], "--progress-every-n"));
                i += 2;
            }
            "-h" | "--help" => print_usage_and_exit(0),
            other => {
                eprintln!("unknown argument: {}", other);
                print_usage_and_exit(2);
            }
        }
    }

    let v3 = v3.unwrap_or_else(|| {
        eprintln!("--v3 is required");
        print_usage_and_exit(2);
    });
    let v5 = v5.unwrap_or_else(|| {
        eprintln!("--v5 is required");
        print_usage_and_exit(2);
    });
    Args {
        v3,
        v5,
        keyfile,
        target_entries_per_block: tepb,
        target_entries_per_sst: teps,
        sort_run_entries: sre,
        skip_verify,
        verify_sample_size: vss,
        max_rss_mb,
        progress_every_n,
    }
}

fn require_next(argv: &[String], i: usize, flag: &str) {
    if i + 1 >= argv.len() {
        eprintln!("{} requires an argument", flag);
        print_usage_and_exit(2);
    }
}

fn parse_or_die<T: std::str::FromStr>(s: &str, flag: &str) -> T
where
    <T as std::str::FromStr>::Err: std::fmt::Display,
{
    s.parse::<T>().unwrap_or_else(|e| {
        eprintln!("{}: cannot parse {:?}: {}", flag, s, e);
        std::process::exit(2);
    })
}

// JSON-emitter -- minimal, no serde dep. Numeric + string fields only.
fn emit_json_summary(stats: &MigrationStats) {
    let mut s = String::new();
    s.push('{');
    s.push_str(&kv_num("v3_node_count", stats.v3_node_count));
    s.push_str(&kv_num("v3_bond_count", stats.v3_bond_count));
    s.push_str(&kv_num("intern_node_count", stats.intern_node_count));
    s.push_str(&kv_num("bonds_read", stats.bonds_read));
    s.push_str(&kv_num("bonds_written", stats.bonds_written));
    s.push_str(&kv_num("sort_runs_spilled", stats.sort_runs_spilled));
    s.push_str(&kv_num("bsst_files_written", stats.bsst_files_written));
    s.push_str(&kv_num("bsst_total_bytes", stats.bsst_total_bytes));
    s.push_str(&kv_num("manifest_bytes", stats.manifest_bytes));
    if let Some(peak_kb) = stats.peak_rss_kb {
        s.push_str(&kv_num("peak_rss_kb", peak_kb));
    } else {
        s.push_str("\"peak_rss_kb\":null,");
    }
    s.push_str(&format!(
        "\"migration_elapsed_secs\":{:.3},",
        stats.migration_elapsed.as_secs_f64()
    ));
    match &stats.verification {
        Some(v) => {
            s.push_str("\"verification\":");
            s.push_str(&verification_to_json(v));
        }
        None => {
            s.push_str("\"verification\":null");
        }
    }
    s.push('}');
    println!("{}", s);
}

fn verification_to_json(v: &VerificationReport) -> String {
    let mut s = String::new();
    s.push('{');
    s.push_str(&kv_num("sample_size", v.sample_size as u64));
    s.push_str(&kv_num("mismatch_count", v.mismatch_count));
    s.push_str(&kv_num("v3_bond_count", v.v3_bond_count));
    s.push_str(&kv_num("v5_bond_count", v.v5_bond_count));
    s.push_str(&format!("\"count_matches\":{},", v.count_matches));
    s.push_str(&format!("\"is_ok\":{},", v.is_ok()));
    s.push_str(&format!("\"elapsed_secs\":{:.3}", v.elapsed.as_secs_f64()));
    s.push('}');
    s
}

fn kv_num<T: std::fmt::Display>(k: &str, v: T) -> String {
    format!("\"{}\":{},", k, v)
}

fn main() -> ExitCode {
    let args = parse_args();

    let key = match &args.keyfile {
        Some(p) => match MasterKey::load_from_keyfile(p) {
            Ok(k) => Some(k),
            Err(e) => {
                eprintln!("failed to load keyfile {}: {}", p.display(), e);
                return ExitCode::from(2);
            }
        },
        None => None,
    };

    let mut cfg = MigrationConfig::new(args.v3.clone(), args.v5.clone());
    cfg.key = key;
    if let Some(n) = args.target_entries_per_block {
        cfg.target_entries_per_block = n;
    }
    if let Some(n) = args.target_entries_per_sst {
        cfg.target_entries_per_sst = n;
    }
    if let Some(n) = args.sort_run_entries {
        cfg.sort_run_entries = n;
    }
    cfg.skip_verify = args.skip_verify;
    cfg.verify_sample_size = args.verify_sample_size.unwrap_or(DEFAULT_VERIFY_SAMPLE_SIZE);
    cfg.max_rss_mb = args.max_rss_mb;
    cfg.progress_every_n = args.progress_every_n.unwrap_or(DEFAULT_PROGRESS_EVERY_N);

    match migrate(&cfg) {
        Ok(stats) => {
            emit_json_summary(&stats);
            ExitCode::from(0)
        }
        Err(MigrationError::VerificationFailed(report)) => {
            let json = format!(
                "{{\"error\":\"verification_failed\",\"verification\":{}}}",
                verification_to_json(&report),
            );
            println!("{}", json);
            eprintln!("verification mismatches (up to 10 shown):");
            for (i, d) in report.mismatch_details.iter().enumerate() {
                eprintln!(
                    "  [{}] subject_id={} seed_hex={} reason={}",
                    i, d.subject_id, d.seed_hex, d.reason
                );
            }
            ExitCode::from(1)
        }
        Err(MigrationError::V3PathMissing(_))
        | Err(MigrationError::BondStoreExists(_))
        | Err(MigrationError::V5DirNotNodeMigrated(_)) => ExitCode::from(2),
        Err(e) => {
            eprintln!("migration failed: {}", e);
            ExitCode::from(3)
        }
    }
}
