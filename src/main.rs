//! Lu 4D CLI entry point.
//!
//! Two modes:
//!   - REPL (default): interactive prompt for humans
//!   - MCP  (--mcp):   JSON-RPC over stdio for Claude and other LLM clients
//!
//! Brain file path is controlled via `--brain <path>` (default: `lu4d.lbr`).

use lu4d::{
    parse_bond_type, parse_confidence, now_ms,
    Brain, BondType, BrainFormatOverride, BrainLayout, BrainLoadMode, BrainSaveFormat,
};
use lu4d::encrypt::{MasterKey, DEFAULT_KEYFILE_PATH};
use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use std::process::ExitCode;

fn print_usage() {
    eprintln!("lu4d {} \u{2014} temporal knowledge graph", env!("CARGO_PKG_VERSION"));
    eprintln!();
    eprintln!("USAGE:");
    eprintln!("  lu4d [--brain <path>] [--brain-format-out v3|v3c|v3enc]");
    eprintln!("                                       start the interactive REPL");
    eprintln!("  lu4d --mcp [--brain <path>] [--brain-format-out v3|v3c|v3enc]");
    eprintln!("                                       run as an MCP server over stdio");
    eprintln!("  lu4d --migrate-from-v1 <src> <dst> [--keyfile <p>] [--out-format v3enc|v3]");
    eprintln!("                                       one-step migrate a v1 JSON brain to v3enc (default) or v3.");
    eprintln!("  lu4d --migrate-v1-to-v2 <src> <dst>");
    eprintln!("                                       convert a v1 JSON brain file to v2 binary");
    eprintln!("  lu4d --migrate-v2-to-v3 <src> <dst>");
    eprintln!("                                       convert an uncompressed v3 brain file to per-block zstd-compressed v3 (Phase 3)");
    eprintln!("  lu4d --migrate-v3-to-v3-compressed <src> <dst>");
    eprintln!("                                       convert a v2 binary brain file to v3 (rkyv-archived)");
    eprintln!("  lu4d --migrate-v3-compressed-to-encrypted <src> <dst> [--keyfile <p>]");
    eprintln!("                                       convert compressed v3 to Phase 4 encrypted (DF XOR + zstd)");
    eprintln!("  lu4d --encrypt-init [--keyfile <p>]  generate a new master key, print recovery phrase");
    eprintln!("  lu4d --from-recovery-phrase [--keyfile <p>]");
    eprintln!("                                       read a 24-word phrase from stdin, write the keyfile");
    eprintln!("  lu4d --help                          show this help");
    eprintln!();
    eprintln!("OPTIONS:");
    eprintln!("  --brain <path>             path to a .lbr brain file (default: lu4d.lbr)");
    eprintln!("  --brain-format <FMT>       v3-file vs v4-directory vs v5-directory detection override");
    eprintln!("                             auto (default) | v3 | v4");
    eprintln!("  --mcp                      run as an MCP server instead of the REPL");
    eprintln!("  --no-save                  do not persist the brain to disk");
    eprintln!("  --time-source SRC          'system' (default, unauthenticated) or 'df' (DF Time Service; stub in T.1, real in T.1.1)");
    eprintln!("  --df-time-endpoint URL     endpoint for --time-source df (T.1.1+; ignored when stub)");
    eprintln!("  --no-noise-filter          disable Port #20 ingestion-noise filter (debug A/B)");
    eprintln!("  --brain-format-out FMT     override on-save format. Accepts v3, v3c, v3enc. v1 (plaintext JSON)");
    eprintln!("  --allow-decrypt-on-save    allow saving an encrypted-loaded brain in a plaintext format (default: refused).");
    eprintln!("                             is no longer a valid save target -- use --migrate-from-v1 to upgrade.");
    eprintln!("  --brain-format-in FMT      assert that the brain file is v1, v2, or v3 — exits");
    eprintln!("                             with an error if the on-disk format does not match.");
    eprintln!("                             FMT may also be v3-mmap-zerocopy (default for v3) or");
    eprintln!("                             v3-eager to force the heap-deserialize debug fallback.");
    eprintln!("                             with an error if the on-disk format doesn't match.");
    eprintln!("                             Auto-detect (the default) handles all three transparently.");
    eprintln!("  --migrate-from-v1 SRC DST  one-step migrate a v1 JSON brain to v3enc (default) or v3, then exit.");
    eprintln!("                             Honors --keyfile and --out-format. Reads the v1 file in-place; SRC unchanged.");
    eprintln!("  --migrate-v1-to-v2 SRC DST migrate a v1 JSON file to v2 binary, then exit");
    eprintln!("  --migrate-v2-to-v3 SRC DST migrate a v2 binary file to v3 (rkyv-archived), then exit");
    eprintln!("  --migrate-v3-to-v3-compressed SRC DST   migrate v3 to per-block zstd-compressed v3, then exit");
    eprintln!("  --migrate-v3-compressed-to-encrypted SRC DST   migrate compressed v3 to encrypted v3 (Phase 4), then exit");
    eprintln!("  --encrypt-init             generate a new master key + recovery phrase, then exit");
    eprintln!("  --from-recovery-phrase     read a 24-word recovery phrase from stdin into the keyfile, then exit");
    eprintln!("  --require-encrypted        Port T.7: refuse to start unless the brain is v3enc, and refuse non-v3enc saves.");
    eprintln!("                             Production gate. Default OFF for dev/test compatibility.");
    eprintln!("  --keyfile <path>           override the default keyfile path ({})", DEFAULT_KEYFILE_PATH);
}

/// Direction for a migration subcommand.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MigrationDir {
    V1ToV2,
    V2ToV3,
    V3ToV3Compressed,
    V3CompressedToEncrypted,
}

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let mut brain_path: Option<PathBuf> = None;
    let mut mcp_mode = false;
    let mut no_save = false;
    let mut no_noise_filter = false;
    // Issue B internal notes: None means "preserve loaded format" so a
    // v3enc brain round-trips intact unless --brain-format-out explicitly
    // overrides. Combined with the v1 JSON removal there is no longer any
    // path that silently downgrades a brain on save.
    let mut save_format: Option<BrainSaveFormat> = None;
    let mut migrate_from_v1: Option<(PathBuf, PathBuf)> = None;
    // Audit fix internal notes: explicit opt-in to downgrade an
    // encrypted-loaded brain to a plaintext format on save.
    let mut allow_decrypt_on_save: bool = false;
    // Port T.7: structural encryption guard. When set, the
    // brain loader pre-checks that the on-disk file is v3enc
    // before loading; the save guard refuses any non-v3enc
    // write. Default OFF preserves dev/test workflows that
    // use plaintext brains; production lu.service sets it.
    let mut require_encrypted: bool = false;
    let mut migrate_from_v1_out_format: Option<BrainSaveFormat> = None;
    let mut expected_in_format: Option<BrainSaveFormat> = None;
    let mut load_mode: BrainLoadMode = BrainLoadMode::Auto;
    let mut migrate_paths: Option<(MigrationDir, PathBuf, PathBuf)> = None;
    let mut keyfile_path: Option<PathBuf> = None;
    // OP-BRAIN-V4 Phase 8: --brain-format <auto|v3|v4> overrides the
    // automatic v3-file vs v4-directory detection. Default Auto.
    let mut brain_format_override: BrainFormatOverride = BrainFormatOverride::Auto;
    let mut encrypt_init = false;
    let mut from_recovery_phrase = false;
    // Port T.1: clock source backend. "system" = SystemClockSource
    // (default, unauthenticated, matches pre-T.1 behaviour). "df" =
    // DFTimeServiceSource (stub in T.1; real signed fetch lands in T.1.1).
    let mut time_source: String = "system".to_string();
    let mut df_time_endpoint: Option<String> = None;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--help" | "-h" => { print_usage(); return ExitCode::SUCCESS; }
            "--mcp" => { mcp_mode = true; }
            "--no-save" => { no_save = true; }
            "--time-source" => {
                match args.next().as_deref() {
                    Some(s) if s == "system" || s == "df" => time_source = s.to_string(),
                    Some(other) => {
                        eprintln!("error: --time-source must be 'system' or 'df', got: {}", other);
                        return ExitCode::from(2);
                    }
                    None => { eprintln!("error: --time-source requires system|df"); return ExitCode::from(2); }
                }
            }
            "--df-time-endpoint" => {
                match args.next() {
                    Some(u) => df_time_endpoint = Some(u),
                    None => { eprintln!("error: --df-time-endpoint requires a URL"); return ExitCode::from(2); }
                }
            }
            "--no-noise-filter" => { no_noise_filter = true; }
            "--brain" => {
                match args.next() {
                    Some(p) => brain_path = Some(PathBuf::from(p)),
                    None => { eprintln!("error: --brain requires a path"); return ExitCode::from(2); }
                }
            }
            "--brain-format" => {
                match args.next().as_deref().and_then(BrainFormatOverride::parse) {
                    Some(o) => brain_format_override = o,
                    None => {
                        eprintln!("error: --brain-format requires auto, v3, v4, or v5");
                        return ExitCode::from(2);
                    }
                }
            }
            "--brain-format-out" => {
                match args.next().as_deref().and_then(BrainSaveFormat::parse) {
                    Some(fmt) => save_format = Some(fmt),
                    None => {
                        eprintln!("error: --brain-format-out requires v3, v3c, or v3enc (v1/v2 are no longer valid save targets)");
                        return ExitCode::from(2);
                    }
                }
            }
            "--allow-decrypt-on-save" => {
                allow_decrypt_on_save = true;
            }
            "--require-encrypted" => {
                require_encrypted = true;
            }
            "--migrate-from-v1" => {
                let src = match args.next() {
                    Some(p) => PathBuf::from(p),
                    None => { eprintln!("error: --migrate-from-v1 requires SRC and DST paths"); return ExitCode::from(2); }
                };
                let dst = match args.next() {
                    Some(p) => PathBuf::from(p),
                    None => { eprintln!("error: --migrate-from-v1 requires SRC and DST paths"); return ExitCode::from(2); }
                };
                migrate_from_v1 = Some((src, dst));
            }
            "--out-format" => {
                match args.next().as_deref().and_then(BrainSaveFormat::parse) {
                    Some(fmt) => migrate_from_v1_out_format = Some(fmt),
                    None => {
                        eprintln!("error: --out-format requires v3, v3c, or v3enc");
                        return ExitCode::from(2);
                    }
                }
            }
            "--load-mode" => {
                // OP-INGEST-LEAN: explicit load-mode selector that does
                // NOT imply a v3 format (unlike --brain-format-in). Used
                // by the ingest harness to request the lean write-only
                // v5 load: `--load-mode ingest-only`.
                let arg = match args.next() {
                    Some(s) => s,
                    None => {
                        eprintln!("error: --load-mode requires a value (auto|ingest-only|read-only|v3-eager|v3-mmap-zerocopy)");
                        return ExitCode::from(2);
                    }
                };
                match BrainLoadMode::parse(&arg) {
                    Some(m) => { load_mode = m; }
                    None => {
                        eprintln!("error: --load-mode unknown value: {}", arg);
                        return ExitCode::from(2);
                    }
                }
            }
            "--brain-format-in" => {
                let arg = match args.next() {
                    Some(s) => s,
                    None => {
                        eprintln!("error: --brain-format-in requires a value (v1|v2|v3|v3-eager|v3-mmap-zerocopy)");
                        return ExitCode::from(2);
                    }
                };
                // Storage v2 Slice 2c — v3-mmap-zerocopy / v3-eager are
                // not save-format names; they pick the v3 LOAD path.
                if let Some(mode) = BrainLoadMode::parse(&arg) {
                    if mode != BrainLoadMode::Auto {
                        load_mode = mode;
                        // Treat the file as v3 for the format-in assertion.
                        expected_in_format = Some(BrainSaveFormat::V3Binary);
                        continue;
                    }
                }
                match BrainSaveFormat::parse(&arg) {
                    Some(fmt) => expected_in_format = Some(fmt),
                    None => {
                        eprintln!("error: --brain-format-in expected v1, v2, v3, v3-eager, or v3-mmap-zerocopy (got {})", arg);
                        return ExitCode::from(2);
                    }
                }
            }
            "--migrate-v1-to-v2" => {
                let src = match args.next() {
                    Some(p) => PathBuf::from(p),
                    None => { eprintln!("error: --migrate-v1-to-v2 requires SRC and DST paths"); return ExitCode::from(2); }
                };
                let dst = match args.next() {
                    Some(p) => PathBuf::from(p),
                    None => { eprintln!("error: --migrate-v1-to-v2 requires SRC and DST paths"); return ExitCode::from(2); }
                };
                migrate_paths = Some((MigrationDir::V1ToV2, src, dst));
            }
            "--migrate-v3-to-v3-compressed" => {
                let src = match args.next() {
                    Some(p) => PathBuf::from(p),
                    None => { eprintln!("error: --migrate-v3-to-v3-compressed requires SRC and DST paths"); return ExitCode::from(2); }
                };
                let dst = match args.next() {
                    Some(p) => PathBuf::from(p),
                    None => { eprintln!("error: --migrate-v3-to-v3-compressed requires SRC and DST paths"); return ExitCode::from(2); }
                };
                migrate_paths = Some((MigrationDir::V3ToV3Compressed, src, dst));
            }
            "--keyfile" => {
                match args.next() {
                    Some(p) => keyfile_path = Some(PathBuf::from(p)),
                    None => { eprintln!("error: --keyfile requires a path"); return ExitCode::from(2); }
                }
            }
            "--encrypt-init" => {
                encrypt_init = true;
            }
            "--gen-capability-pack-keypair" => {
                // Port Cat.5: generate a fresh Ed25519 keypair
                // for capability-pack signing. Mirrors --gen-pack-keypair
                // (Cat.4) but isolated from content-pack trust.
                let path = match args.next() {
                    Some(p) => p,
                    None => { eprintln!("error: --gen-capability-pack-keypair requires a path"); return ExitCode::from(2); }
                };
                match lu4d::mcp_cat5::gen_capability_pack_keypair_cli(&path) {
                    Ok(pk_hex) => {
                        println!("{}", pk_hex);
                        return ExitCode::SUCCESS;
                    }
                    Err(e) => {
                        eprintln!("error: --gen-capability-pack-keypair: {}", e);
                        return ExitCode::from(1);
                    }
                }
            }
            "--gen-pack-keypair" => {
                // Port Cat.4: generate a fresh Ed25519 keypair, write
                // the 32-byte secret key to <path> (mode 0600 on Unix),
                // and print the public-key hex on stdout. Exit immediately.
                let path = match args.next() {
                    Some(p) => p,
                    None => { eprintln!("error: --gen-pack-keypair requires a path"); return ExitCode::from(2); }
                };
                match lu4d::mcp_pack::gen_pack_keypair_cli(&path) {
                    Ok(pk_hex) => {
                        println!("{}", pk_hex);
                        return ExitCode::SUCCESS;
                    }
                    Err(e) => {
                        eprintln!("error: --gen-pack-keypair: {}", e);
                        return ExitCode::from(1);
                    }
                }
            }
            "--from-recovery-phrase" => {
                from_recovery_phrase = true;
            }
            "--migrate-v3-compressed-to-encrypted" => {
                let src = match args.next() {
                    Some(p) => PathBuf::from(p),
                    None => { eprintln!("error: --migrate-v3-compressed-to-encrypted requires SRC and DST paths"); return ExitCode::from(2); }
                };
                let dst = match args.next() {
                    Some(p) => PathBuf::from(p),
                    None => { eprintln!("error: --migrate-v3-compressed-to-encrypted requires SRC and DST paths"); return ExitCode::from(2); }
                };
                migrate_paths = Some((MigrationDir::V3CompressedToEncrypted, src, dst));
            }
            "--migrate-v2-to-v3" => {
                let src = match args.next() {
                    Some(p) => PathBuf::from(p),
                    None => { eprintln!("error: --migrate-v2-to-v3 requires SRC and DST paths"); return ExitCode::from(2); }
                };
                let dst = match args.next() {
                    Some(p) => PathBuf::from(p),
                    None => { eprintln!("error: --migrate-v2-to-v3 requires SRC and DST paths"); return ExitCode::from(2); }
                };
                migrate_paths = Some((MigrationDir::V2ToV3, src, dst));
            }
            other => {
                eprintln!("error: unknown argument {}", other);
                print_usage();
                return ExitCode::from(2);
            }
        }
    }

    if let Some((src, dst)) = migrate_from_v1.take() {
        return run_migrate_from_v1(&src, &dst, migrate_from_v1_out_format, keyfile_path.as_deref());
    }

    if encrypt_init {
        let kf = keyfile_path.clone().unwrap_or_else(|| PathBuf::from(DEFAULT_KEYFILE_PATH));
        return run_encrypt_init(&kf);
    }
    if from_recovery_phrase {
        let kf = keyfile_path.clone().unwrap_or_else(|| PathBuf::from(DEFAULT_KEYFILE_PATH));
        return run_from_recovery_phrase(&kf);
    }

    if let Some((dir, src, dst)) = migrate_paths {
        return match dir {
            MigrationDir::V1ToV2 => run_migration(&src, &dst),
            MigrationDir::V2ToV3 => run_migration_v2_to_v3(&src, &dst),
            MigrationDir::V3ToV3Compressed => run_migration_v3_to_v3_compressed(&src, &dst),
            MigrationDir::V3CompressedToEncrypted => {
                let kf = keyfile_path.clone().unwrap_or_else(|| PathBuf::from(DEFAULT_KEYFILE_PATH));
                run_migration_v3_compressed_to_encrypted(&src, &dst, &kf)
            }
        };
    }

    // If --brain-format-in was passed, verify the file on disk has the
    // expected format before we touch it. Auto-detect always wins for
    // actual loading; this flag is purely an assertion to catch
    // operator errors (e.g., pointing at the wrong file).
    if let Some(expected) = expected_in_format {
        let candidate_path = brain_path.clone().unwrap_or_else(|| PathBuf::from("lu4d.lbr"));
        if candidate_path.exists() {
            let detected = match lu4d::brain_io::detect_format_path(&candidate_path) {
                Ok(d) => d,
                Err(e) => {
                    eprintln!("error: cannot read {}: {}", candidate_path.display(), e);
                    return ExitCode::from(1);
                }
            };
            let detected_save = match detected {
                lu4d::brain_io::BrainFormat::V1Json => BrainSaveFormat::V1Json,
                lu4d::brain_io::BrainFormat::V2Binary => BrainSaveFormat::V2Binary,
                lu4d::brain_io::BrainFormat::V3Binary => BrainSaveFormat::V3Binary,
            };
            if detected_save != expected {
                eprintln!(
                    "error: --brain-format-in {} but {} is {}",
                    expected.as_str(),
                    candidate_path.display(),
                    detected_save.as_str(),
                );
                return ExitCode::from(1);
            }
        }
    }

    let path = Some(brain_path.unwrap_or_else(|| PathBuf::from("lu4d.lbr")));

    // Phase 4: load keyfile bytes only if the file exists (encrypted
    // brains require it; unencrypted brains tolerate a missing
    // keyfile). The actual encryption-required check happens inside
    // NetworkArchive::from_mmap_with_key.
    let key_for_load: Option<MasterKey> = match keyfile_path.as_ref() {
        Some(p) => match MasterKey::load_from_keyfile(p) {
            Ok(k) => Some(k),
            Err(e) => { eprintln!("error: keyfile {}: {}", p.display(), e); return ExitCode::from(1); }
        },
        None => {
            // Try the default keyfile -- only useful if the brain
            // turns out to be encrypted. Missing-default is fine.
            let default = std::path::Path::new(DEFAULT_KEYFILE_PATH);
            if default.exists() {
                match MasterKey::load_from_keyfile(default) {
                    Ok(k) => Some(k),
                    Err(_) => None,
                }
            } else {
                None
            }
        }
    };

    // Port T.7: pre-load encryption guard. If --require-encrypted
    // is set, refuse to start unless the brain on disk is v3enc.
    // The error message points the operator at
    // --migrate-v3-compressed-to-encrypted so they can recover.
    if require_encrypted {
        if let Some(p) = path.as_ref() {
            if let Err(e) = lu4d::brain::require_v3enc_or_err(p) {
                eprintln!("error: {}", e);
                return ExitCode::from(1);
            }
            // Port T-Ingest.3: empty-creation guard. require_v3enc_or_err
            // returns Ok when the path does not exist (nothing on disk to
            // validate), so add the companion check that refuses to start
            // when --require-encrypted is set, the brain file does not exist,
            // and no keyfile was provided. Without this guard the empty-brain
            // path would create a plaintext brain that the very next save
            // would refuse to write -- but only after ingest tools were wired
            // up. Refusing here keeps the failure at startup.
            if let Err(e) = lu4d::brain::require_encrypted_create_or_err(p, &key_for_load) {
                eprintln!("error: {}", e);
                return ExitCode::from(1);
            }
        }
    }

    // OP-BRAIN-V4 Phase 8: brain-path detection.
    //   --brain-format auto (default) -> detect_brain_layout decides.
    //   --brain-format v3              -> force the v3 .lbr path.
    //   --brain-format v4              -> force the v4 directory path.
    // The v4 directory path uses Brain::load_v4 (MemTable + WAL replay
    // + flush worker + compaction worker auto-enabled; read_path_mode
    // = V4Only by default). The v3 path stays on the legacy loader so
    // existing brains are byte-equivalent.
    let mut brain = match &path {
        Some(p) => {
            let layout = match brain_format_override {
                BrainFormatOverride::Auto => match Brain::detect_brain_layout(p) {
                    Ok(l) => l,
                    Err(e) => { eprintln!("error: {}", e); return ExitCode::from(1); }
                },
                BrainFormatOverride::V3 => {
                    if p.is_dir() {
                        eprintln!("error: --brain-format v3 but {} is a directory; remove the override or point at a .lbr file", p.display());
                        return ExitCode::from(2);
                    }
                    if p.exists() { BrainLayout::V3File } else { BrainLayout::Missing }
                }
                BrainFormatOverride::V4 => {
                    if !p.exists() {
                        eprintln!("error: --brain-format v4 but {} does not exist; run `lu4d_migrate_v3_to_v4` first", p.display());
                        return ExitCode::from(2);
                    }
                    if !p.is_dir() {
                        eprintln!("error: --brain-format v4 but {} is not a directory; v4 brains live in a directory containing MANIFEST + L*-N.sst + wal/", p.display());
                        return ExitCode::from(2);
                    }
                    BrainLayout::V4Dir
                }
                BrainFormatOverride::V5 => {
                    if !p.exists() {
                        eprintln!("error: --brain-format v5 but {} does not exist; run the v3->v5 migration tools first", p.display());
                        return ExitCode::from(2);
                    }
                    if !p.is_dir() {
                        eprintln!("error: --brain-format v5 but {} is not a directory; v5 brains live in a directory containing MANIFEST + node_store/ + bond_store/ + wal/", p.display());
                        return ExitCode::from(2);
                    }
                    BrainLayout::V5Dir
                }
            };
            match layout {
                BrainLayout::V3File | BrainLayout::Missing => {
                    match Brain::load_or_new_with_mode_and_key(p, load_mode, key_for_load) {
                        Ok(b) => b,
                        Err(e) => { eprintln!("error: failed to load v3 brain from {}: {}", p.display(), e); return ExitCode::from(1); }
                    }
                }
                BrainLayout::V4Dir => {
                    match Brain::load_v4(p, key_for_load) {
                        Ok(b) => {
                            eprintln!("[lu4d] OP-BRAIN-V4: loaded v4 brain at {} (flush + compaction workers enabled, read_path = V4Only)", p.display());
                            b
                        }
                        Err(e) => { eprintln!("error: failed to load v4 brain from {}: {}", p.display(), e); return ExitCode::from(1); }
                    }
                }
                BrainLayout::V5Dir => {
                    match Brain::load_v5_with_mode(p, key_for_load, load_mode) {
                        Ok(b) => {
                            eprintln!("[lu4d] OP-BRAIN-V5: loaded v5 brain at {} (node_store + bond_store, flush + compaction workers enabled, read_path = V5Only)", p.display());
                            b
                        }
                        Err(e) => { eprintln!("error: failed to load v5 brain from {}: {}", p.display(), e); return ExitCode::from(1); }
                    }
                }
            }
        }
        None => Brain::new(),
    };
    // Issue B internal notes: only override save_format when the user
    // explicitly passed --brain-format-out. Otherwise preserve the format
    // the brain was loaded with so v3enc round-trips intact.
    if let Some(fmt) = save_format {
        if let Some(loaded) = brain.loaded_format {
            if loaded != fmt {
                eprintln!(
                    "[lu4d] WARNING: --brain-format-out {} differs from on-disk format {}; brain will be re-saved as {} on exit",
                    fmt.as_str(), loaded.as_str(), fmt.as_str()
                );
            }
        }
        brain.save_format = fmt;
    }
    brain.allow_decrypt_on_save = allow_decrypt_on_save;
    // Port T.7: propagate to the save-side guard.
    brain.require_encrypted = require_encrypted;
    // Port T.1.1: apply the loaded brain's time_source_config (system
    // by default; df_service if the brain was previously configured
    // via lu4d_configure). The --time-source CLI flag below still
    // wins as an explicit override for ad-hoc testing.
    match brain.apply_time_source_config() {
        Ok(name) if name != "system" => {
            eprintln!("[lu4d] T.1.1: time_source = {} (from runtime config)", name);
        }
        Ok(_) => {} // system mode — no log spam.
        Err(e) => {
            eprintln!("[lu4d] T.1.1: apply_time_source_config failed: {}; staying on system", e);
        }
    }
    // Port T.1: swap clock source if --time-source df was passed.
    if time_source == "df" {
        let src: std::sync::Arc<dyn lu4d::time::ClockSource> = match &df_time_endpoint {
            Some(ep) => std::sync::Arc::new(lu4d::time::DFTimeServiceSource::new(ep.clone())),
            None => std::sync::Arc::new(lu4d::time::DFTimeServiceSource::stub()),
        };
        match brain.set_clock_source(src) {
            Ok(()) => eprintln!("[lu4d] T.1: time-source = {} (authenticated)", brain.clock.name()),
            Err(e) => {
                eprintln!("[lu4d] T.1: --time-source df failed: {}; falling back to system", e);
            }
        }
    }
    // else: preserve brain.save_format set by load(). Brain::new()
    // (no file backing) defaults to V3Binary post-v1-removal.
    if no_noise_filter {
        brain.narration_config.noise_filter_enabled = false;
    }

    // OP-BRAIN-V5 6.6 -- operator-facing startup banner (both --mcp and
    // REPL paths): report loaded brain FORMAT (v3/v4/v5) + at-rest
    // ENCRYPTION so an operator can confirm at a glance what is running.
    // Capability only -- does not change which brain the service loads.
    eprintln!("{}", brain.operator_status_line(require_encrypted));

    if mcp_mode {
        if let Err(e) = lu4d::mcp::run_mcp_server_opts(&mut brain, !no_save) {
            eprintln!("mcp server error: {}", e);
            return ExitCode::from(1);
        }
        if !no_save {
            if let Err(e) = brain.save() {
                eprintln!("warning: failed to save brain on exit: {}", e);
            }
        }
        return ExitCode::SUCCESS;
    }

    match run_repl(&mut brain) {
        Ok(()) => {
            if !no_save {
                if let Err(e) = brain.save() {
                    eprintln!("warning: failed to save brain on exit: {}", e);
                }
            }
            ExitCode::SUCCESS
        }
        Err(e) => { eprintln!("repl error: {}", e); ExitCode::from(1) }
    }
}

fn run_repl(brain: &mut Brain) -> io::Result<()> {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = stdout.lock();

    let banner = format!(
        "Lu 4D v{} — temporal knowledge graph\n\
         loaded: {} nodes, {} bonds (file: {})\n\
         type `help` for commands, `quit` to exit\n",
        env!("CARGO_PKG_VERSION"),
        brain.network.node_count(),
        brain.network.bond_count(),
        brain.path.as_ref().map(|p| p.display().to_string()).unwrap_or_else(|| "<ephemeral>".into()),
    );
    out.write_all(banner.as_bytes())?;
    out.flush()?;

    let mut input = stdin.lock();
    loop {
        out.write_all(b"lu4d> ")?;
        out.flush()?;

        let mut line = String::new();
        let n = input.read_line(&mut line)?;
        if n == 0 { // EOF
            out.write_all(b"\n")?;
            break;
        }

        let line = line.trim();
        if line.is_empty() { continue; }

        let response = handle_command(brain, line);
        out.write_all(response.as_bytes())?;
        if !response.ends_with('\n') { out.write_all(b"\n")?; }
        out.flush()?;

        if line == "quit" || line == "exit" { break; }
    }
    Ok(())
}

fn handle_command(brain: &mut Brain, line: &str) -> String {
    let mut parts = line.splitn(2, char::is_whitespace);
    let cmd = parts.next().unwrap_or("").to_lowercase();
    let rest = parts.next().unwrap_or("").trim();

    match cmd.as_str() {
        "help" | "?" => help_text(),
        "quit" | "exit" => "bye!".to_string(),
        "teach" => cmd_teach(brain, rest),
        "ask" => cmd_ask(brain, rest),
        "status" => cmd_status(brain),
        "session" => cmd_session(brain, rest),
        "timetravel" | "tt" => cmd_timetravel(brain, rest),
        "consolidate" => cmd_consolidate(brain),
        "save" => match brain.save() {
            Ok(()) => format!("saved brain to {}", brain.path.as_ref().map(|p| p.display().to_string()).unwrap_or_else(|| "<ephemeral>".into())),
            Err(e) => format!("save failed: {}", e),
        },
        "bonds" => cmd_list_bond_types(),
        other => format!("unknown command: `{}` (try `help`)", other),
    }
}

fn help_text() -> String {
    r#"Commands:
  teach <subject> | <relation> | <object> [| <weight> [| <confidence>]]
      Add a bond. `weight` default 0.8, `confidence` default Verified.
      Example: teach Rust | IsA | ProgrammingLanguage | 0.95 | Verified
  ask <query> [| <depth> [| <top_k>]]
      Run temporal spread activation. Defaults: depth=3, top_k=10.
      Hebbian-strengthens co-activated bonds if a session is active.
  status
      Print node/bond/session counts.
  session start | session end
      Manage temporal sessions. Active session isolates Hebbian learning.
  timetravel <query> | <ms_ago>
      Diff activation at <now - ms_ago> vs current state.
      Example: timetravel Rust | 60000   (one minute ago)
  consolidate
      Run baseline consolidation — promotes multi-session bonds, prunes decay.
  bonds
      List the 11 bond types.
  save
      Persist the brain to the .lbr file.
  help | quit
"#.to_string()
}

fn cmd_list_bond_types() -> String {
    let mut s = String::from("Bond types:\n");
    for bt in BondType::all() {
        s.push_str(&format!("  {:?} (base weight {:.2})\n", bt, bt.base_weight()));
    }
    s
}

fn cmd_teach(brain: &mut Brain, rest: &str) -> String {
    let parts: Vec<&str> = rest.split('|').map(|p| p.trim()).collect();
    if parts.len() < 3 {
        return "usage: teach <subject> | <relation> | <object> [| <weight> [| <confidence>]]".into();
    }
    let subject = parts[0];
    let relation_str = parts[1];
    let object = parts[2];
    let weight: f32 = parts.get(3).and_then(|s| s.parse().ok()).unwrap_or(0.8);
    let confidence_str = parts.get(4).copied().unwrap_or("Verified");

    let relation = match parse_bond_type(relation_str) {
        Ok(r) => r,
        Err(e) => return format!("error: {}", e),
    };
    let confidence = match parse_confidence(confidence_str) {
        Ok(c) => c,
        Err(e) => return format!("error: {}", e),
    };

    let r = brain.teach(subject, relation, object, weight, confidence);
    format!(
        "taught: {} -[{:?}]-> {} (w={:.2}, conf={:?}, t={}){}{}{}{}",
        subject, r.bond_type, object, r.weight, confidence, r.timestamp,
        if r.created_subject { "\n  + created subject node" } else { "" },
        if r.created_object { "\n  + created object node" } else { "" },
        if r.reinforced_existing_bond { "\n  + reinforced existing bond (new keyframe)" } else { "\n  + new bond" },
        match r.session_id {
            Some(id) => format!("\n  session: {}", id),
            None => String::new(),
        }
    )
}

fn cmd_ask(brain: &mut Brain, rest: &str) -> String {
    if rest.is_empty() {
        return "usage: ask <query> [| <depth> [| <top_k>]]".into();
    }
    let parts: Vec<&str> = rest.split('|').map(|p| p.trim()).collect();
    let query = parts[0];
    let depth: usize = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(3);
    let top_k: usize = parts.get(2).and_then(|s| s.parse().ok()).unwrap_or(10);

    let r = brain.ask(query, depth, top_k);
    let mut s = format!(
        "query: {}\n  seeds: {}\n  activated: {}\n  nodes_visited: {}\n  bonds_evaluated: {}\n",
        r.query, r.seed_hits.len(), r.recall.activated.len(),
        r.recall.nodes_visited, r.recall.bonds_evaluated,
    );
    if r.recall.activated.is_empty() {
        s.push_str("  (no matches — try teaching some facts first, or broaden your query)\n");
    } else {
        for (i, a) in r.recall.activated.iter().enumerate() {
            s.push_str(&format!(
                "  {:>2}. {} [act={:.4} depth={} conf={:?}]\n",
                i + 1, a.content, a.activation, a.depth, a.confidence,
            ));
        }
    }
    if r.hebbian_strengthened > 0 {
        s.push_str(&format!("  hebbian: {} bond keyframes written\n", r.hebbian_strengthened));
    }
    if let Some(sid) = r.session_id {
        s.push_str(&format!("  session: {}\n", sid));
    }
    s
}

fn cmd_status(brain: &Brain) -> String {
    let s = brain.status();
    format!(
        "nodes:              {}\n\
         bonds:              {}\n\
         total sessions:     {}\n\
         active session:     {}\n\
         active activations: {}\n\
         active keyframes:   {}\n\
         brain file:         {}\n\
         format version:     {}",
        s.node_count, s.bond_count, s.total_sessions,
        s.active_session_id.map(|id| id.to_string()).unwrap_or_else(|| "<none>".into()),
        s.active_session_activations, s.active_session_keyframes,
        s.brain_file.as_deref().unwrap_or("<ephemeral>"),
        s.format_version,
    )
}

fn cmd_session(brain: &mut Brain, rest: &str) -> String {
    match rest.trim().to_lowercase().as_str() {
        "start" => {
            let id = brain.session_start();
            format!("started session {}", id)
        }
        "end" => match brain.session_end() {
            Some(id) => format!("ended session {}", id),
            None => "no active session".into(),
        },
        "" => "usage: session start | session end".into(),
        other => format!("unknown session subcommand: {}", other),
    }
}

fn cmd_timetravel(brain: &Brain, rest: &str) -> String {
    let parts: Vec<&str> = rest.split('|').map(|p| p.trim()).collect();
    if parts.len() < 2 || parts[0].is_empty() {
        return "usage: timetravel <query> | <ms_ago>".into();
    }
    let query = parts[0];
    let ms_ago: i64 = match parts[1].parse() {
        Ok(n) => n,
        Err(_) => return "error: ms_ago must be an integer".into(),
    };
    let at = now_ms() - ms_ago;
    let diff = brain.timetravel(query, at);

    let mut s = format!(
        "time travel: query=\"{}\" at={} ({}ms ago)\n  summary: emerged={}, faded={}, strengthened={}, weakened={}, stable={}, avg|Δ|={:.4}\n",
        query, at, ms_ago,
        diff.summary.emerged, diff.summary.faded,
        diff.summary.strengthened, diff.summary.weakened,
        diff.summary.stable, diff.summary.avg_abs_delta,
    );
    s.push_str(&format!(
        "  historical recall: {} activated, {} bonds evaluated\n  current recall:    {} activated, {} bonds evaluated\n",
        diff.historical_recall.activated.len(), diff.historical_recall.bonds_evaluated,
        diff.current_recall.activated.len(), diff.current_recall.bonds_evaluated,
    ));
    for (i, d) in diff.diffs.iter().take(10).enumerate() {
        s.push_str(&format!(
            "  {:>2}. {} [{:?}] hist={:.4} cur={:.4} Δ={:+.4}\n",
            i + 1, d.content, d.change_type, d.historical_activation, d.current_activation, d.delta,
        ));
    }
    s
}

fn cmd_consolidate(brain: &mut Brain) -> String {
    let stats = brain.consolidate();
    format!(
        "consolidation done:\n  promoted: {}\n  pruned:   {}\n  examined: {}\n  at:       {}",
        stats.bonds_promoted, stats.keyframes_pruned, stats.bonds_examined, stats.ran_at,
    )
}


/// Migrate a v1 JSON brain file to a v2 binary brain file.
///
/// Reads `src` (refusing if it is not v1), writes `dst` as v2 binary,
/// then reloads `dst` and verifies counts match. Source is never modified.
fn run_migration(src: &PathBuf, dst: &PathBuf) -> ExitCode {
    use lu4d::brain_io;

    if !src.exists() {
        eprintln!("error: source file does not exist: {}", src.display());
        return ExitCode::from(1);
    }
    if src == dst {
        eprintln!("error: SRC and DST must differ -- refusing to overwrite source");
        return ExitCode::from(1);
    }

    let src_format = match brain_io::detect_format_path(src) {
        Ok(f) => f,
        Err(e) => { eprintln!("error: cannot read {}: {}", src.display(), e); return ExitCode::from(1); }
    };
    if src_format != brain_io::BrainFormat::V1Json {
        eprintln!(
            "error: source {} is not v1 JSON (detected {:?}); refusing migration",
            src.display(), src_format,
        );
        return ExitCode::from(1);
    }

    eprintln!("loading v1: {}", src.display());
    let t0 = std::time::Instant::now();
    // Audit fix internal notes / Problem #111 follow-up): the migration
    // utility must use the isolated v1 reader, not the production
    // load_or_new (which now rejects v1 outright).
    let mut brain = match Brain::load_v1_for_migration_only(src) {
        Ok(b) => b,
        Err(e) => { eprintln!("error: load failed: {}", e); return ExitCode::from(1); }
    };
    let v1_load_ms = t0.elapsed().as_millis();
    let v1_node = brain.network.node_count();
    let v1_bond = brain.network.bond_count();
    let v1_kf: usize = brain.network.bonds().values().map(|b| b.keyframes().len()).sum();
    let v1_scar = brain.scar_log.scars.len();
    let v1_size = std::fs::metadata(src).map(|m| m.len()).unwrap_or(0);
    eprintln!(
        "  v1: {} nodes, {} bonds, {} keyframes, {} scars ({} bytes, load {} ms)",
        v1_node, v1_bond, v1_kf, v1_scar, v1_size, v1_load_ms,
    );

    eprintln!("writing v2: {}", dst.display());
    let t1 = std::time::Instant::now();
    brain.save_format = BrainSaveFormat::V2Binary;
    if let Err(e) = brain.save_to_format(dst, BrainSaveFormat::V2Binary) {
        eprintln!("error: write failed: {}", e); return ExitCode::from(1);
    }
    let v2_write_ms = t1.elapsed().as_millis();
    let v2_size = std::fs::metadata(dst).map(|m| m.len()).unwrap_or(0);
    eprintln!("  v2 written ({} bytes, save {} ms)", v2_size, v2_write_ms);

    eprintln!("verifying: reload v2 and compare counts");
    let t2 = std::time::Instant::now();
    let reload = match Brain::load_or_new(dst) {
        Ok(b) => b,
        Err(e) => { eprintln!("error: v2 reload failed: {}", e); return ExitCode::from(1); }
    };
    let v2_load_ms = t2.elapsed().as_millis();
    let r_node = reload.network.node_count();
    let r_bond = reload.network.bond_count();
    let r_kf: usize = reload.network.bonds().values().map(|b| b.keyframes().len()).sum();
    let r_scar = reload.scar_log.scars.len();
    eprintln!(
        "  v2 reload: {} nodes, {} bonds, {} keyframes, {} scars (load {} ms)",
        r_node, r_bond, r_kf, r_scar, v2_load_ms,
    );

    let mut ok = true;
    if r_node != v1_node { eprintln!("  MISMATCH node_count {} vs {}", v1_node, r_node); ok = false; }
    if r_bond != v1_bond { eprintln!("  MISMATCH bond_count {} vs {}", v1_bond, r_bond); ok = false; }
    if r_kf != v1_kf { eprintln!("  MISMATCH keyframe_count {} vs {}", v1_kf, r_kf); ok = false; }
    if r_scar != v1_scar { eprintln!("  MISMATCH scar_count {} vs {}", v1_scar, r_scar); ok = false; }

    if !ok {
        eprintln!("migration FAILED count verification");
        return ExitCode::from(1);
    }

    let ratio = if v1_size > 0 { v2_size as f64 / v1_size as f64 } else { 0.0 };
    eprintln!(
        "migration OK: {} -> {}  (size {:.1}% of v1, v2 load {}ms vs v1 load {}ms)",
        src.display(), dst.display(), ratio * 100.0, v2_load_ms, v1_load_ms,
    );
    ExitCode::SUCCESS
}

/// Migrate a v2 binary brain file to a v3 binary brain file.
///
/// Mirrors the v1→v2 migrator: reads `src` (refusing if it is not v2),
/// writes `dst` as v3 binary, then reloads `dst` and verifies counts
/// match. Source is never modified.
fn run_migration_v2_to_v3(src: &PathBuf, dst: &PathBuf) -> ExitCode {
    use lu4d::brain_io;

    if !src.exists() {
        eprintln!("error: source file does not exist: {}", src.display());
        return ExitCode::from(1);
    }
    if src == dst {
        eprintln!("error: SRC and DST must differ -- refusing to overwrite source");
        return ExitCode::from(1);
    }

    let src_format = match brain_io::detect_format_path(src) {
        Ok(f) => f,
        Err(e) => { eprintln!("error: cannot read {}: {}", src.display(), e); return ExitCode::from(1); }
    };
    if src_format != brain_io::BrainFormat::V2Binary {
        eprintln!(
            "error: source {} is not v2 binary (detected {:?}); refusing migration",
            src.display(), src_format,
        );
        return ExitCode::from(1);
    }

    eprintln!("loading v2: {}", src.display());
    let t0 = std::time::Instant::now();
    let mut brain = match Brain::load_or_new(src) {
        Ok(b) => b,
        Err(e) => { eprintln!("error: load failed: {}", e); return ExitCode::from(1); }
    };
    let v2_load_ms = t0.elapsed().as_millis();
    let v2_node = brain.network.node_count();
    let v2_bond = brain.network.bond_count();
    let v2_kf: usize = brain.network.bonds().values().map(|b| b.keyframes().len()).sum();
    let v2_scar = brain.scar_log.scars.len();
    let v2_size = std::fs::metadata(src).map(|m| m.len()).unwrap_or(0);
    eprintln!(
        "  v2: {} nodes, {} bonds, {} keyframes, {} scars ({} bytes, load {} ms)",
        v2_node, v2_bond, v2_kf, v2_scar, v2_size, v2_load_ms,
    );

    eprintln!("writing v3: {}", dst.display());
    let t1 = std::time::Instant::now();
    brain.save_format = BrainSaveFormat::V3Binary;
    if let Err(e) = brain.save_to_format(dst, BrainSaveFormat::V3Binary) {
        eprintln!("error: write failed: {}", e); return ExitCode::from(1);
    }
    let v3_write_ms = t1.elapsed().as_millis();
    let v3_size = std::fs::metadata(dst).map(|m| m.len()).unwrap_or(0);
    eprintln!("  v3 written ({} bytes, save {} ms)", v3_size, v3_write_ms);

    eprintln!("verifying: reload v3 and compare counts");
    let t2 = std::time::Instant::now();
    let reload = match Brain::load_or_new(dst) {
        Ok(b) => b,
        Err(e) => { eprintln!("error: v3 reload failed: {}", e); return ExitCode::from(1); }
    };
    let v3_load_ms = t2.elapsed().as_millis();
    let r_node = reload.network.node_count();
    let r_bond = reload.network.bond_count();
    let r_kf: usize = reload.network.bonds().values().map(|b| b.keyframes().len()).sum();
    let r_scar = reload.scar_log.scars.len();
    eprintln!(
        "  v3 reload: {} nodes, {} bonds, {} keyframes, {} scars (load {} ms)",
        r_node, r_bond, r_kf, r_scar, v3_load_ms,
    );

    let mut ok = true;
    if r_node != v2_node { eprintln!("  MISMATCH node_count {} vs {}", v2_node, r_node); ok = false; }
    if r_bond != v2_bond { eprintln!("  MISMATCH bond_count {} vs {}", v2_bond, r_bond); ok = false; }
    if r_kf != v2_kf { eprintln!("  MISMATCH keyframe_count {} vs {}", v2_kf, r_kf); ok = false; }
    if r_scar != v2_scar { eprintln!("  MISMATCH scar_count {} vs {}", v2_scar, r_scar); ok = false; }

    if !ok {
        eprintln!("migration FAILED count verification");
        return ExitCode::from(1);
    }

    let ratio = if v2_size > 0 { v3_size as f64 / v2_size as f64 } else { 0.0 };
    eprintln!(
        "migration OK: {} -> {}  (size {:.1}% of v2, v3 load {}ms vs v2 load {}ms)",
        src.display(), dst.display(), ratio * 100.0, v3_load_ms, v2_load_ms,
    );
    ExitCode::SUCCESS
}


/// Migrate an uncompressed v3 brain file to a Phase 3 per-block
/// zstd-compressed v3 brain file.
///
/// Verifies that `src` is uncompressed v3, writes `dst` as compressed
/// v3, then reloads `dst` and verifies node/bond/keyframe/scar counts
/// match. Source is never modified.
fn run_migration_v3_to_v3_compressed(src: &PathBuf, dst: &PathBuf) -> ExitCode {
    use lu4d::brain_io;

    if !src.exists() {
        eprintln!("error: source file does not exist: {}", src.display());
        return ExitCode::from(1);
    }
    if src == dst {
        eprintln!("error: SRC and DST must differ -- refusing to overwrite source");
        return ExitCode::from(1);
    }

    let src_format = match brain_io::detect_format_path(src) {
        Ok(f) => f,
        Err(e) => { eprintln!("error: cannot read {}: {}", src.display(), e); return ExitCode::from(1); }
    };
    if src_format != brain_io::BrainFormat::V3Binary {
        eprintln!(
            "error: source {} is not v3 binary (detected {:?}); refusing migration",
            src.display(), src_format,
        );
        return ExitCode::from(1);
    }

    eprintln!("loading v3 (uncompressed): {}", src.display());
    let t0 = std::time::Instant::now();
    let mut brain = match Brain::load_or_new(src) {
        Ok(b) => b,
        Err(e) => { eprintln!("error: load failed: {}", e); return ExitCode::from(1); }
    };
    let load_ms = t0.elapsed().as_millis();
    let v3_node = brain.network.node_count();
    let v3_bond = brain.network.bond_count();
    let v3_kf: usize = brain.network.bonds().values().map(|b| b.keyframe_count()).sum();
    let v3_scar = brain.scar_log.scars.len();
    let v3_size = std::fs::metadata(src).map(|m| m.len()).unwrap_or(0);
    eprintln!(
        "  v3: {} nodes, {} bonds, {} keyframes, {} scars ({} bytes, load {} ms)",
        v3_node, v3_bond, v3_kf, v3_scar, v3_size, load_ms,
    );

    eprintln!("writing v3-compressed: {}", dst.display());
    let t1 = std::time::Instant::now();
    brain.save_format = BrainSaveFormat::V3BinaryCompressed;
    if let Err(e) = brain.save_to_format(dst, BrainSaveFormat::V3BinaryCompressed) {
        eprintln!("error: write failed: {}", e); return ExitCode::from(1);
    }
    let write_ms = t1.elapsed().as_millis();
    let v3c_size = std::fs::metadata(dst).map(|m| m.len()).unwrap_or(0);
    let ratio = if v3_size > 0 { v3c_size as f64 / v3_size as f64 } else { 0.0 };
    eprintln!(
        "  v3-compressed written ({} bytes, {:.1}% of source, save {} ms)",
        v3c_size, ratio * 100.0, write_ms,
    );

    eprintln!("verifying: reload v3-compressed and compare counts");
    let t2 = std::time::Instant::now();
    let reload = match Brain::load_or_new(dst) {
        Ok(b) => b,
        Err(e) => { eprintln!("error: v3-compressed reload failed: {}", e); return ExitCode::from(1); }
    };
    let reload_ms = t2.elapsed().as_millis();
    let r_node = reload.network.node_count();
    let r_bond = reload.network.bond_count();
    let r_kf: usize = reload.network.bonds().values().map(|b| b.keyframe_count()).sum();
    let r_scar = reload.scar_log.scars.len();
    eprintln!(
        "  reload: {} nodes, {} bonds, {} keyframes, {} scars (load {} ms)",
        r_node, r_bond, r_kf, r_scar, reload_ms,
    );

    let mut ok = true;
    if r_node != v3_node { eprintln!("  MISMATCH node_count {} vs {}", v3_node, r_node); ok = false; }
    if r_bond != v3_bond { eprintln!("  MISMATCH bond_count {} vs {}", v3_bond, r_bond); ok = false; }
    if r_kf != v3_kf { eprintln!("  MISMATCH keyframe_count {} vs {}", v3_kf, r_kf); ok = false; }
    if r_scar != v3_scar { eprintln!("  MISMATCH scar_count {} vs {}", v3_scar, r_scar); ok = false; }

    if !ok {
        eprintln!("migration FAILED count verification");
        return ExitCode::from(1);
    }

    eprintln!(
        "migration OK: {} -> {}  (size {:.1}% of v3, reload {}ms vs load {}ms)",
        src.display(), dst.display(), ratio * 100.0, reload_ms, load_ms,
    );
    ExitCode::SUCCESS
}


/// `--encrypt-init`: generate a fresh 32-byte master key, write it to
/// the keyfile (mode 0640, refuses to overwrite), and print the
/// 24-word BIP39 recovery phrase to stdout. The phrase is shown
/// ONCE and is the operator's responsibility to record off-server.
///
/// Returns nonzero if the keyfile already exists or cannot be written.
fn run_encrypt_init(keyfile: &PathBuf) -> ExitCode {
    if keyfile.exists() {
        eprintln!("error: keyfile already exists: {}", keyfile.display());
        eprintln!("  refusing to overwrite -- delete it first if this is intentional");
        return ExitCode::from(1);
    }
    let key = MasterKey::generate();
    let phrase = match key.to_recovery_phrase() {
        Ok(p) => p,
        Err(e) => { eprintln!("error: encode recovery phrase: {}", e); return ExitCode::from(1); }
    };
    if let Err(e) = key.save_to_keyfile(keyfile) {
        eprintln!("error: write keyfile {}: {}", keyfile.display(), e);
        return ExitCode::from(1);
    }
    eprintln!("keyfile written: {} (mode 0640)", keyfile.display());
    eprintln!();
    eprintln!("=== RECOVERY PHRASE (24 words) ===");
    println!("{}", phrase);
    eprintln!("=== END ===");
    eprintln!();
    eprintln!("Save the phrase above somewhere safe and OFF this server.");
    eprintln!("It is the only way to regenerate the keyfile if it is lost.");
    ExitCode::SUCCESS
}

/// `--from-recovery-phrase`: read a 24-word phrase from stdin,
/// derive the master key, and write it to the keyfile. Refuses to
/// overwrite an existing keyfile.
fn run_from_recovery_phrase(keyfile: &PathBuf) -> ExitCode {
    if keyfile.exists() {
        eprintln!("error: keyfile already exists: {}", keyfile.display());
        eprintln!("  refusing to overwrite -- delete it first if this is intentional");
        return ExitCode::from(1);
    }
    eprintln!("Paste the 24-word recovery phrase, then press Enter:");
    let mut buf = String::new();
    if let Err(e) = std::io::stdin().read_line(&mut buf) {
        eprintln!("error: read stdin: {}", e);
        return ExitCode::from(1);
    }
    let key = match MasterKey::from_recovery_phrase(&buf) {
        Ok(k) => k,
        Err(e) => { eprintln!("error: parse recovery phrase: {}", e); return ExitCode::from(1); }
    };
    if let Err(e) = key.save_to_keyfile(keyfile) {
        eprintln!("error: write keyfile {}: {}", keyfile.display(), e);
        return ExitCode::from(1);
    }
    eprintln!("keyfile written: {} (mode 0640)", keyfile.display());
    ExitCode::SUCCESS
}

/// `--migrate-v3-compressed-to-encrypted SRC DST [--keyfile P]`:
/// reads compressed v3 at SRC, writes Phase 4 encrypted-compressed
/// v3 to DST, reloads DST with the same key, and verifies counts
/// match. Source is never mutated; if SRC is encrypted already (or
/// is uncompressed v3), refuses with an explicit error.
fn run_migration_v3_compressed_to_encrypted(
    src: &PathBuf,
    dst: &PathBuf,
    keyfile: &PathBuf,
) -> ExitCode {
    use lu4d::brain_io;

    if !src.exists() {
        eprintln!("error: source file does not exist: {}", src.display());
        return ExitCode::from(1);
    }
    if src == dst {
        eprintln!("error: SRC and DST must differ -- refusing to overwrite source");
        return ExitCode::from(1);
    }
    let src_format = match brain_io::detect_format_path(src) {
        Ok(f) => f,
        Err(e) => { eprintln!("error: cannot read {}: {}", src.display(), e); return ExitCode::from(1); }
    };
    if src_format != brain_io::BrainFormat::V3Binary {
        eprintln!(
            "error: source {} is not v3 binary (detected {:?}); refusing migration",
            src.display(), src_format,
        );
        return ExitCode::from(1);
    }

    let key = match MasterKey::load_from_keyfile(keyfile) {
        Ok(k) => k,
        Err(e) => { eprintln!("error: keyfile {}: {}", keyfile.display(), e); return ExitCode::from(1); }
    };

    eprintln!("loading v3 compressed: {}", src.display());
    let t0 = std::time::Instant::now();
    // Source must be unencrypted compressed (or unencrypted plain v3).
    // Pass no key on the load side so an already-encrypted SRC fails
    // explicitly rather than corrupting silently.
    let mut brain = match Brain::load_or_new_with_mode(src, BrainLoadMode::Auto) {
        Ok(b) => b,
        Err(e) => { eprintln!("error: load failed: {}", e); return ExitCode::from(1); }
    };
    let load_ms = t0.elapsed().as_millis();
    let v3_node = brain.network.node_count();
    let v3_bond = brain.network.bond_count();
    let v3_kf: usize = brain.network.bonds().values().map(|b| b.keyframe_count()).sum();
    let v3_scar = brain.scar_log.scars.len();
    let v3_size = std::fs::metadata(src).map(|m| m.len()).unwrap_or(0);
    eprintln!(
        "  v3: {} nodes, {} bonds, {} keyframes, {} scars ({} bytes, load {} ms)",
        v3_node, v3_bond, v3_kf, v3_scar, v3_size, load_ms,
    );

    eprintln!("writing v3-encrypted: {}", dst.display());
    let t1 = std::time::Instant::now();
    brain.encryption_key = Some(key.clone());
    brain.save_format = BrainSaveFormat::V3BinaryEncrypted;
    if let Err(e) = brain.save_to_format(dst, BrainSaveFormat::V3BinaryEncrypted) {
        eprintln!("error: write failed: {}", e); return ExitCode::from(1);
    }
    let write_ms = t1.elapsed().as_millis();
    let v3e_size = std::fs::metadata(dst).map(|m| m.len()).unwrap_or(0);
    let ratio = if v3_size > 0 { v3e_size as f64 / v3_size as f64 } else { 0.0 };
    eprintln!(
        "  v3-encrypted written ({} bytes, {:.1}% of source, save {} ms)",
        v3e_size, ratio * 100.0, write_ms,
    );

    eprintln!("verifying: reload v3-encrypted with keyfile and compare counts");
    let t2 = std::time::Instant::now();
    let reload = match Brain::load_or_new_with_mode_and_key(
        dst, BrainLoadMode::Auto, Some(key.clone()),
    ) {
        Ok(b) => b,
        Err(e) => { eprintln!("error: v3-encrypted reload failed: {}", e); return ExitCode::from(1); }
    };
    let reload_ms = t2.elapsed().as_millis();
    let r_node = reload.network.node_count();
    let r_bond = reload.network.bond_count();
    let r_kf: usize = reload.network.bonds().values().map(|b| b.keyframe_count()).sum();
    let r_scar = reload.scar_log.scars.len();
    eprintln!(
        "  reload: {} nodes, {} bonds, {} keyframes, {} scars (load {} ms)",
        r_node, r_bond, r_kf, r_scar, reload_ms,
    );

    let mut ok = true;
    if r_node != v3_node { eprintln!("  MISMATCH node_count {} vs {}", v3_node, r_node); ok = false; }
    if r_bond != v3_bond { eprintln!("  MISMATCH bond_count {} vs {}", v3_bond, r_bond); ok = false; }
    if r_kf != v3_kf { eprintln!("  MISMATCH keyframe_count {} vs {}", v3_kf, r_kf); ok = false; }
    if r_scar != v3_scar { eprintln!("  MISMATCH scar_count {} vs {}", v3_scar, r_scar); ok = false; }
    if !ok {
        eprintln!("migration FAILED count verification");
        return ExitCode::from(1);
    }
    eprintln!(
        "migration OK: {} -> {}  (size {:.1}% of source, reload {}ms vs load {}ms)",
        src.display(), dst.display(), ratio * 100.0, reload_ms, load_ms,
    );
    ExitCode::SUCCESS
}


/// One-step v1 JSON -> v3enc (or v3) migration.
///
/// Reads a v1 JSON brain file (the plaintext format that v1 JSON removal
/// took out of the production save path), writes it back in v3enc by
/// default (v3 unencrypted if the user explicitly opts out via
/// --out-format v3, or no key is available). Verifies counts round-trip
/// before exiting.
///
/// internal notes: the legacy v1 read path is preserved so
/// existing v1 brains can be migrated; new code never produces v1.
fn run_migrate_from_v1(
    src: &std::path::Path,
    dst: &std::path::Path,
    out_format: Option<BrainSaveFormat>,
    keyfile: Option<&std::path::Path>,
) -> ExitCode {
    use lu4d::brain_io;

    if !src.exists() {
        eprintln!("error: source file does not exist: {}", src.display());
        return ExitCode::from(1);
    }
    if src == dst {
        eprintln!("error: SRC and DST must differ -- refusing to overwrite source");
        return ExitCode::from(1);
    }

    let src_format = match brain_io::detect_format_path(src) {
        Ok(f) => f,
        Err(e) => { eprintln!("error: cannot read {}: {}", src.display(), e); return ExitCode::from(1); }
    };
    if src_format != brain_io::BrainFormat::V1Json {
        eprintln!(
            "error: source {} is not v1 JSON (detected {:?}); refusing migration",
            src.display(), src_format,
        );
        return ExitCode::from(1);
    }

    eprintln!("loading v1: {}", src.display());
    let t0 = std::time::Instant::now();
    // Audit fix internal notes / Problem #111 follow-up): the migration
    // utility must use the isolated v1 reader, not the production
    // load_or_new (which now rejects v1 outright).
    let mut brain = match Brain::load_v1_for_migration_only(src) {
        Ok(b) => b,
        Err(e) => { eprintln!("error: load failed: {}", e); return ExitCode::from(1); }
    };
    let v1_load_ms = t0.elapsed().as_millis();
    let v1_node = brain.network.node_count();
    let v1_bond = brain.network.bond_count();
    let v1_size = std::fs::metadata(src).map(|m| m.len()).unwrap_or(0);
    eprintln!("  v1: {} nodes, {} bonds ({} bytes, load {} ms)",
        v1_node, v1_bond, v1_size, v1_load_ms);

    // Decide output format.
    let key = match keyfile {
        Some(p) => match MasterKey::load_from_keyfile(p) {
            Ok(k) => Some(k),
            Err(e) => { eprintln!("error: keyfile load failed: {}", e); return ExitCode::from(1); }
        },
        None => {
            let default = std::path::Path::new(DEFAULT_KEYFILE_PATH);
            if default.exists() { MasterKey::load_from_keyfile(default).ok() } else { None }
        }
    };
    let chosen = match out_format {
        Some(f) if f == BrainSaveFormat::V1Json => {
            eprintln!("error: --out-format v1 is not a valid migration target");
            return ExitCode::from(2);
        }
        Some(f) => f,
        None => {
            if key.is_some() { BrainSaveFormat::V3BinaryEncrypted }
            else { BrainSaveFormat::V3Binary }
        }
    };
    if chosen == BrainSaveFormat::V3BinaryEncrypted && key.is_none() {
        eprintln!("error: --out-format v3enc requires a keyfile (pass --keyfile or place one at {})",
            DEFAULT_KEYFILE_PATH);
        return ExitCode::from(1);
    }
    brain.encryption_key = key.clone();
    brain.save_format = chosen;

    eprintln!("writing {}: {}", chosen.as_str(), dst.display());
    let t1 = std::time::Instant::now();
    if let Err(e) = brain.save_to_format(dst, chosen) {
        eprintln!("error: write failed: {}", e);
        return ExitCode::from(1);
    }
    let write_ms = t1.elapsed().as_millis();
    let dst_size = std::fs::metadata(dst).map(|m| m.len()).unwrap_or(0);
    eprintln!("  {} written ({} bytes, save {} ms)", chosen.as_str(), dst_size, write_ms);

    eprintln!("verifying: reload destination and compare counts");
    let t2 = std::time::Instant::now();
    let reload = match Brain::load_or_new_with_mode_and_key(
        dst, BrainLoadMode::Auto, key,
    ) {
        Ok(b) => b,
        Err(e) => { eprintln!("error: reload failed: {}", e); return ExitCode::from(1); }
    };
    let reload_ms = t2.elapsed().as_millis();
    let r_node = reload.network.node_count();
    let r_bond = reload.network.bond_count();
    if r_node != v1_node || r_bond != v1_bond {
        eprintln!(
            "error: count mismatch -- v1 had {} nodes / {} bonds, reload has {} / {}",
            v1_node, v1_bond, r_node, r_bond,
        );
        return ExitCode::from(1);
    }
    eprintln!("  reload: {} nodes, {} bonds ({} ms) -- OK", r_node, r_bond, reload_ms);
    eprintln!("migration complete: v1 -> {} ({}), source untouched", chosen.as_str(), dst.display());
    ExitCode::SUCCESS
}
