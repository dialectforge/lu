//! OP-BRAIN-V5 6.7 -- v3 vs v5 migration verifier CLI.
//!
//! Offline GO/NO-GO gate for the loader cutover (design doc §7 step-5
//! runbook). Loads the source v3 brain and the migrated v5 directory
//! READ-ONLY, runs recall parity + encryption-at-rest + node/bond count
//! parity, prints a report, and exits 0 on GO / 1 on NO-GO. It never
//! opens the live brain for write -- run it against snapshots/copies.
//!
//! Usage:
//!   lu_brain_verify_v3_v5 --v3 <brain.lbr> --v5 <v5dir> [--keyfile <p>] [--sample N]

use std::path::PathBuf;
use std::process::ExitCode;

use lu4d::encrypt::MasterKey;
use lu4d::migration_verify_v5::{verify, VerifyConfig};

fn usage() {
    eprintln!("usage: lu_brain_verify_v3_v5 --v3 <brain.lbr> --v5 <v5dir> [--keyfile <path>] [--sample <N>]");
    eprintln!("  --v3 <path>       source v3 brain (.lbr), opened read-only");
    eprintln!("  --v5 <dir>        migrated v5 directory (node_store/ + bond_store/ + MANIFEST)");
    eprintln!("  --keyfile <path>  master key (required for encrypted brains)");
    eprintln!("  --sample <N>      deep-compare at most N subjects for recall parity (0 = all; default 0)");
}

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let mut v3: Option<PathBuf> = None;
    let mut v5: Option<PathBuf> = None;
    let mut keyfile: Option<PathBuf> = None;
    let mut sample: usize = 0;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--v3" => v3 = args.next().map(PathBuf::from),
            "--v5" => v5 = args.next().map(PathBuf::from),
            "--keyfile" => keyfile = args.next().map(PathBuf::from),
            "--sample" => match args.next().and_then(|s| s.parse::<usize>().ok()) {
                Some(n) => sample = n,
                None => {
                    eprintln!("error: --sample requires a non-negative integer");
                    return ExitCode::from(2);
                }
            },
            "-h" | "--help" => {
                usage();
                return ExitCode::SUCCESS;
            }
            other => {
                eprintln!("error: unknown argument {}", other);
                usage();
                return ExitCode::from(2);
            }
        }
    }
    let (v3, v5) = match (v3, v5) {
        (Some(a), Some(b)) => (a, b),
        _ => {
            usage();
            return ExitCode::from(2);
        }
    };
    let key = match keyfile {
        Some(p) => match MasterKey::load_from_keyfile(&p) {
            Ok(k) => Some(k),
            Err(e) => {
                eprintln!("error: keyfile {}: {}", p.display(), e);
                return ExitCode::from(1);
            }
        },
        None => None,
    };
    let mut cfg = VerifyConfig::new(v3, v5);
    cfg.key = key;
    cfg.sample_limit = sample;
    match verify(&cfg) {
        Ok(report) => {
            print!("{}", report.render());
            if report.passed() {
                println!("\nRESULT: GO");
                ExitCode::SUCCESS
            } else {
                println!("\nRESULT: NO-GO");
                ExitCode::from(1)
            }
        }
        Err(e) => {
            eprintln!("error: verification could not run: {}", e);
            ExitCode::from(1)
        }
    }
}
