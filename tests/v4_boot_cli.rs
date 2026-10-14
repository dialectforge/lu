//! OP-BRAIN-V4 Phase 8 -- CLI-level integration tests for the
//! `--brain` + `--brain-format` detection logic. These cases live
//! here (not in the lib unit tests) because the detection/override
//! logic that resolves a mismatch is in `src/main.rs`, and the
//! error path is only reachable by actually invoking the daemon
//! binary.
//!
//! Coverage:
//!
//! 1. `--brain-format v4` pointed at a v3 file -> exit 2 with a
//!    clear error.
//! 2. `--brain-format v3` pointed at a v4 directory -> exit 2 with
//!    a clear error.
//! 3. `--brain-format v4` pointed at a non-existent path -> exit 2
//!    with a clear "run migration first" hint.
//! 4. `--brain-format auto` against an empty directory (no
//!    MANIFEST) -> exit 1 with a directory-without-MANIFEST error.

use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};

/// Locate the freshly-built `lu4d` binary. The test relies on
/// `cargo test` having already produced `target/release/lu4d` (or
/// `target/debug/lu4d`). Prefer release because the rest of the
/// Phase 8 test surface runs under `cargo test --release`.
fn lu4d_binary() -> std::path::PathBuf {
    let mut p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("target");
    let release = p.join("release/lu4d");
    if release.exists() {
        return release;
    }
    let debug = p.join("debug/lu4d");
    if debug.exists() {
        return debug;
    }
    panic!(
        "could not locate lu4d binary at {} or {} -- run `cargo build` first",
        release.display(),
        debug.display(),
    );
}

/// Spawn `lu4d <args>` with stdin closed (so the REPL exits
/// immediately if we get past the load step) and capture stdout +
/// stderr.
fn run_lu4d(args: &[&str]) -> (i32, String) {
    let bin = lu4d_binary();
    let mut child = Command::new(&bin)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap_or_else(|e| panic!("spawn {}: {}", bin.display(), e));
    // Close stdin so any REPL-loaded interactive read returns EOF.
    drop(child.stdin.take());
    let out = child.wait_with_output().expect("wait_with_output");
    let mut combined = String::new();
    combined.push_str(&String::from_utf8_lossy(&out.stdout));
    combined.push_str(&String::from_utf8_lossy(&out.stderr));
    (out.status.code().unwrap_or(-1), combined)
}

fn write_dummy_v3(path: &Path) {
    // The CLI errors fail fast on the mismatch BEFORE the loader
    // is touched, so the file contents don't need to be a real
    // v3 brain -- only the file-vs-directory metadata matters.
    let mut f = std::fs::File::create(path).unwrap();
    f.write_all(b"not a real brain").unwrap();
    f.flush().unwrap();
}

fn write_dummy_v4_dir(dir: &Path) {
    // Same shortcut: the CLI `--brain-format v3` guard rejects a
    // directory before the loader is touched.
    std::fs::create_dir_all(dir).unwrap();
    std::fs::File::create(dir.join("MANIFEST")).unwrap();
}

#[test]
fn cli_brain_format_v4_against_v3_file_errors() {
    let dir = tempfile::tempdir().unwrap();
    let v3 = dir.path().join("brain.lbr");
    write_dummy_v3(&v3);
    let (code, out) = run_lu4d(&[
        "--brain", v3.to_str().unwrap(),
        "--brain-format", "v4",
        "--no-save",
    ]);
    assert_eq!(code, 2,
        "expected exit 2 for --brain-format v4 on a v3 file; got {} -- output:\n{}",
        code, out);
    assert!(out.contains("is not a directory"),
        "error must say v4 requires a directory; got:\n{}", out);
}

#[test]
fn cli_brain_format_v3_against_v4_dir_errors() {
    let dir = tempfile::tempdir().unwrap();
    let v4 = dir.path().join("v4dir");
    write_dummy_v4_dir(&v4);
    let (code, out) = run_lu4d(&[
        "--brain", v4.to_str().unwrap(),
        "--brain-format", "v3",
        "--no-save",
    ]);
    assert_eq!(code, 2,
        "expected exit 2 for --brain-format v3 on a v4 directory; got {} -- output:\n{}",
        code, out);
    assert!(out.contains("is a directory"),
        "error must say v3 requires a file; got:\n{}", out);
}

#[test]
fn cli_brain_format_v4_on_missing_path_errors() {
    let dir = tempfile::tempdir().unwrap();
    let missing = dir.path().join("does-not-exist-yet");
    let (code, out) = run_lu4d(&[
        "--brain", missing.to_str().unwrap(),
        "--brain-format", "v4",
        "--no-save",
    ]);
    assert_eq!(code, 2,
        "expected exit 2 for --brain-format v4 on a missing path; got {} -- output:\n{}",
        code, out);
    assert!(out.contains("does not exist"),
        "error must say the path does not exist; got:\n{}", out);
    assert!(out.contains("lu4d_migrate_v3_to_v4"),
        "error must mention the migration tool; got:\n{}", out);
}

#[test]
fn cli_brain_format_auto_against_empty_dir_errors() {
    let dir = tempfile::tempdir().unwrap();
    let empty = dir.path().join("empty");
    std::fs::create_dir_all(&empty).unwrap();
    let (code, out) = run_lu4d(&[
        "--brain", empty.to_str().unwrap(),
        "--no-save",
    ]);
    assert_eq!(code, 1,
        "expected exit 1 for --brain auto on empty dir; got {} -- output:\n{}",
        code, out);
    assert!(out.contains("no MANIFEST"),
        "error must mention missing MANIFEST; got:\n{}", out);
    assert!(out.contains("lu4d_migrate_v3_to_v4"),
        "error must hint at migration tool; got:\n{}", out);
}

#[test]
fn cli_brain_format_garbage_errors() {
    let dir = tempfile::tempdir().unwrap();
    let v3 = dir.path().join("brain.lbr");
    write_dummy_v3(&v3);
    let (code, out) = run_lu4d(&[
        "--brain", v3.to_str().unwrap(),
        "--brain-format", "junk",
        "--no-save",
    ]);
    assert_eq!(code, 2,
        "expected exit 2 for unknown --brain-format value; got {} -- output:\n{}",
        code, out);
    assert!(out.contains("--brain-format"),
        "error must mention the option name; got:\n{}", out);
}
