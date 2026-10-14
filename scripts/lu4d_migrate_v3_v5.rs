//! OP-BRAIN-V4 Phase 7 -- v3 -> v4 migration CLI.
//!
//! Standalone, one-shot binary that converts an existing v3 brain
//! file (`brain.lbr`) into the v4 layout (a directory containing
//! `MANIFEST` + L0 SSTs + an empty `WAL/` subdirectory).
//!
//! Usage:
//!   lu4d_migrate_v3_to_v4 --v3 <path> --v4 <dir> [--keyfile <path>]
//!                         [--ssts-target-size <bytes>] [--verify-only]
//!
//! Exit codes:
//!   0 - migration ran and verify passed (or `--verify-only` passed)
//!   1 - migration ran but verify failed (data mismatch)
//!   2 - usage / IO / format error before migration started
//!   3 - the v4 dir is in an unexpected state and the tool refuses to
//!       touch it (e.g. contains unrecognised files)

use std::path::PathBuf;
use std::process::ExitCode;

use lu4d::encrypt::MasterKey;
use lu4d::migration_v3_v4::{MigrationReport, V3ToV4Migrator, VerifyReport};

struct Args {
    v3_path: PathBuf,
    v4_dir: PathBuf,
    keyfile: Option<PathBuf>,
    ssts_target_size: Option<usize>,
    verify_only: bool,
}

fn print_usage_and_exit(code: u8) -> ! {
    eprintln!(
        "usage: lu4d_migrate_v3_to_v4 --v3 <path> --v4 <dir>            [--keyfile <path>] [--ssts-target-size <bytes>] [--verify-only]"
    );
    std::process::exit(code as i32);
}

fn parse_args() -> Args {
    let mut v3: Option<PathBuf> = None;
    let mut v4: Option<PathBuf> = None;
    let mut keyfile: Option<PathBuf> = None;
    let mut ssts_target_size: Option<usize> = None;
    let mut verify_only = false;

    let argv: Vec<String> = std::env::args().collect();
    let mut i = 1usize;
    while i < argv.len() {
        match argv[i].as_str() {
            "--v3" => {
                if i + 1 >= argv.len() {
                    eprintln!("--v3 requires a path argument");
                    print_usage_and_exit(2);
                }
                v3 = Some(PathBuf::from(&argv[i + 1]));
                i += 2;
            }
            "--v4" => {
                if i + 1 >= argv.len() {
                    eprintln!("--v4 requires a directory argument");
                    print_usage_and_exit(2);
                }
                v4 = Some(PathBuf::from(&argv[i + 1]));
                i += 2;
            }
            "--keyfile" => {
                if i + 1 >= argv.len() {
                    eprintln!("--keyfile requires a path argument");
                    print_usage_and_exit(2);
                }
                keyfile = Some(PathBuf::from(&argv[i + 1]));
                i += 2;
            }
            "--ssts-target-size" => {
                if i + 1 >= argv.len() {
                    eprintln!("--ssts-target-size requires a byte count");
                    print_usage_and_exit(2);
                }
                ssts_target_size = Some(argv[i + 1].parse().unwrap_or_else(|e| {
                    eprintln!(
                        "--ssts-target-size: cannot parse {} as bytes: {}",
                        argv[i + 1],
                        e,
                    );
                    std::process::exit(2);
                }));
                i += 2;
            }
            "--verify-only" => {
                verify_only = true;
                i += 1;
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
    let v4 = v4.unwrap_or_else(|| {
        eprintln!("--v4 is required");
        print_usage_and_exit(2);
    });
    Args {
        v3_path: v3,
        v4_dir: v4,
        keyfile,
        ssts_target_size,
        verify_only,
    }
}

fn print_migration_report(r: &MigrationReport) {
    println!("--- migration report ---");
    println!("v4_dir: {}", r.v4_dir.display());
    println!("idempotent_no_op: {}", r.idempotent_no_op);
    println!("bonds_migrated: {}", r.bonds_migrated);
    println!("ssts_written: {}", r.ssts_written);
    println!("elapsed: {:.1}s", r.elapsed.as_secs_f64());
    if let Some(p) = r.peak_rss_bytes {
        println!(
            "peak_rss: {:.2} MiB",
            p as f64 / (1024.0 * 1024.0)
        );
    } else {
        println!("peak_rss: <unknown>");
    }
}

fn print_verify_report(v: &VerifyReport) {
    println!("--- verify report ---");
    println!("subjects_checked: {}", v.subjects_checked);
    println!("bonds_checked: {}", v.bonds_checked);
    if v.is_ok() {
        println!("result: OK");
    } else {
        println!("result: FAIL ({} mismatches)", v.mismatches.len());
        for (i, m) in v.mismatches.iter().take(10).enumerate() {
            println!("  [{}] {:?}", i, m);
        }
        if v.mismatches.len() > 10 {
            println!("  ...and {} more", v.mismatches.len() - 10);
        }
    }
}

fn main() -> ExitCode {
    let args = parse_args();

    // Load key if a keyfile was supplied.
    let key = match &args.keyfile {
        Some(p) => match MasterKey::load_from_keyfile(p) {
            Ok(k) => Some(k),
            Err(e) => {
                eprintln!(
                    "failed to load keyfile {}: {}",
                    p.display(),
                    e,
                );
                return ExitCode::from(2);
            }
        },
        None => None,
    };

    let mut migrator = match V3ToV4Migrator::new(
        args.v3_path.clone(),
        args.v4_dir.clone(),
        key,
    ) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("migrator init failed: {}", e);
            return ExitCode::from(2);
        }
    };
    if let Some(sz) = args.ssts_target_size {
        migrator.set_ssts_target_size(sz);
    }

    if args.verify_only {
        match migrator.verify() {
            Ok(v) => {
                print_verify_report(&v);
                return if v.is_ok() {
                    ExitCode::from(0)
                } else {
                    ExitCode::from(1)
                };
            }
            Err(e) => {
                eprintln!("verify failed: {}", e);
                return ExitCode::from(2);
            }
        }
    }

    let report = match migrator.run() {
        Ok(r) => r,
        Err(e) => {
            eprintln!("migration failed: {}", e);
            return ExitCode::from(3);
        }
    };
    print_migration_report(&report);

    // Always verify after a successful run; the spec's discipline
    // says "don't claim success without verify passing."
    match migrator.verify() {
        Ok(v) => {
            print_verify_report(&v);
            if v.is_ok() {
                ExitCode::from(0)
            } else {
                ExitCode::from(1)
            }
        }
        Err(e) => {
            eprintln!("post-migration verify failed: {}", e);
            ExitCode::from(1)
        }
    }
}
