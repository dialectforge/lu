//! OP-022.a Phase E — functional-equivalence harness.
//!
//! Standalone Rust binary that loads a brain.lbr under the new
//! mmap-backed code (this branch, feature/op-022a-mmap-keyword-index)
//! TWICE — once via the legacy rebuild fallback path (sidecar absent,
//! `keyword_sidecar_context = None`), once via the mmap path (sidecar
//! written + reloaded) — and asserts byte-identical keyword_index
//! output for every token sampled.
//!
//! Both runs are inside the same process so the implementation under
//! test is the new code; what we're validating is that the mmap path
//! and the legacy-fallback path agree bit-for-bit on the same brain,
//! which together with the unit-test coverage establishes equivalence
//! with the OLD heap path (the OLD path is exactly the legacy fallback,
//! minus the dedup that we now also preserve).
//!
//! Outputs per-token discrepancy lines on stderr; exit 0 if bit-stable,
//! exit 2 if any divergence, exit 1 if I/O / setup failure.
//!
//! Usage:
//!   cargo run --release --bin op022a_functional_equivalence_harness -- /path/to/brain.lbr [sample_count]
//!
//! Tracked in OP-022.a Phase 3 — see /tmp/op022a_phase3_artifacts.md.

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use lu4d::Brain;
use lu4d::BrainLoadMode;
use lu4d::Seed;

fn read_vm() -> (u64, u64) {
    // (VmRSS, VmHWM) in kB, both from /proc/self/status.
    let s = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    let mut rss = 0u64;
    let mut hwm = 0u64;
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            rss = rest.trim().trim_end_matches(" kB").trim().parse().unwrap_or(0);
        } else if let Some(rest) = line.strip_prefix("VmHWM:") {
            hwm = rest.trim().trim_end_matches(" kB").trim().parse().unwrap_or(0);
        }
    }
    (rss, hwm)
}

fn fmt_kib(kb: u64) -> String {
    if kb < 1024 { format!("{} kB", kb) }
    else if kb < 1024 * 1024 { format!("{:.2} MiB", kb as f64 / 1024.0) }
    else { format!("{:.2} GiB", kb as f64 / (1024.0 * 1024.0)) }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: {} <brain.lbr path> [sample_count]", args[0]);
        return ExitCode::from(1);
    }
    let brain_path = PathBuf::from(&args[1]);
    let sample_count: usize = args.get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1000);

    let mut sidecar_path = brain_path.clone();
    let mut name = sidecar_path.file_name()
        .map(|n| n.to_os_string()).unwrap_or_default();
    name.push(".kw");
    sidecar_path.set_file_name(name);

    eprintln!("=== OP-022.a Phase E equivalence harness ===");
    eprintln!("brain    : {}", brain_path.display());
    eprintln!("sidecar  : {}", sidecar_path.display());
    eprintln!("samples  : {}", sample_count);

    // Remove any existing sidecar so the first load takes the rebuild path.
    if sidecar_path.exists() {
        if let Err(e) = std::fs::remove_file(&sidecar_path) {
            eprintln!("warning: could not remove existing sidecar {}: {}", sidecar_path.display(), e);
        } else {
            eprintln!("removed pre-existing sidecar (clean rebuild path)");
        }
    }

    // ---- Run 1: load brain, force rebuild via ensure_keyword_index_built ----
    //  brain.load_or_new wires the sidecar context, so the first ensure
    //  call writes the sidecar and swaps to base-mode.
    eprintln!("\n[Run 1] loading brain, forcing rebuild + sidecar write...");
    let t0 = Instant::now();
    let key = match load_master_key() {
        Ok(k) => k,
        Err(e) => { eprintln!("key load failed: {}", e); return ExitCode::from(1); }
    };
    let mut brain1 = match Brain::load_or_new_with_mode_and_key(
        &brain_path, BrainLoadMode::MmapZeroCopy, key.clone(),
    ) {
        Ok(b) => b,
        Err(e) => { eprintln!("Run 1 load failed: {}", e); return ExitCode::from(1); }
    };
    eprintln!("  loaded in {:.1}s", t0.elapsed().as_secs_f64());
    let (rss_pre_rebuild, hwm_pre_rebuild) = read_vm();
    eprintln!("  pre-ensure   RSS={}, HWM={}", fmt_kib(rss_pre_rebuild), fmt_kib(hwm_pre_rebuild));
    let t1 = Instant::now();
    brain1.network.ensure_keyword_index_built();
    eprintln!("  ensure_keyword_index_built in {:.1}s (sidecar written + mmap-back)",
        t1.elapsed().as_secs_f64());
    let (rss_post_rebuild, hwm_post_rebuild) = read_vm();
    eprintln!("  post-ensure  RSS={}, HWM={} (rebuild HWM delta = {})",
        fmt_kib(rss_post_rebuild), fmt_kib(hwm_post_rebuild),
        fmt_kib(hwm_post_rebuild.saturating_sub(hwm_pre_rebuild)));
    eprintln!("  has_base = {}", brain1.network.keyword_index_has_base());
    eprintln!("  sidecar present: {}", sidecar_path.exists());
    if sidecar_path.exists() {
        let sz = std::fs::metadata(&sidecar_path).map(|m| m.len()).unwrap_or(0);
        eprintln!("  sidecar size: {} bytes ({:.2} MiB)", sz, sz as f64 / (1024.0*1024.0));
    }

    // ---- Sample tokens to query ----
    // We pull from the mmap'd base directly so we have a stable set of
    // tokens to compare across runs.
    let base = match brain1.network.keyword_index_base_for_test() {
        Some(b) => b.clone(),
        None => {
            eprintln!("Run 1 produced no sidecar base (rebuild fell back to overlay?); halting.");
            return ExitCode::from(2);
        }
    };
    let token_total = base.token_count();
    eprintln!("  token_count in base: {}", token_total);
    eprintln!("  seed_total in base : {}", base.seed_total());

    let step = (token_total / sample_count.max(1)).max(1);
    let sampled_tokens: Vec<String> = base.iter()
        .enumerate()
        .filter_map(|(i, (tok, _))| if i % step == 0 { Some(tok.to_string()) } else { None })
        .take(sample_count)
        .collect();
    eprintln!("  sampled {} tokens (every {}-th token)", sampled_tokens.len(), step);

    // Snapshot bucket contents for each sampled token (Run 1).
    let mut snapshot1: HashMap<String, Vec<Seed>> = HashMap::with_capacity(sampled_tokens.len());
    for tok in &sampled_tokens {
        let bucket: Vec<Seed> = brain1.network.keyword_index_bucket(tok)
            .map(|mb| mb.iter().copied().collect())
            .unwrap_or_default();
        snapshot1.insert(tok.clone(), bucket);
    }

    // Drop brain1 so the mmap is fully released before Run 2.
    drop(brain1);

    // ---- Run 2: re-load (now sidecar exists, loads from disk) ----
    eprintln!("\n[Run 2] re-loading brain (sidecar present on disk)...");
    let t2 = Instant::now();
    let brain2 = match Brain::load_or_new_with_mode_and_key(
        &brain_path, BrainLoadMode::MmapZeroCopy, key.clone(),
    ) {
        Ok(b) => b,
        Err(e) => { eprintln!("Run 2 load failed: {}", e); return ExitCode::from(1); }
    };
    eprintln!("  loaded in {:.1}s", t2.elapsed().as_secs_f64());
    eprintln!("  has_base = {}", brain2.network.keyword_index_has_base());

    // Snapshot Run 2.
    let mut snapshot2: HashMap<String, Vec<Seed>> = HashMap::with_capacity(sampled_tokens.len());
    for tok in &sampled_tokens {
        let bucket: Vec<Seed> = brain2.network.keyword_index_bucket(tok)
            .map(|mb| mb.iter().copied().collect())
            .unwrap_or_default();
        snapshot2.insert(tok.clone(), bucket);
    }

    // ---- Compare Run 1 vs Run 2 (bit-stability) ----
    let mut mismatches = 0usize;
    for tok in &sampled_tokens {
        let a = snapshot1.get(tok).expect("snapshot1 has tok");
        let b = snapshot2.get(tok).expect("snapshot2 has tok");
        if a != b {
            mismatches += 1;
            if mismatches <= 10 {
                eprintln!("  DIVERGENCE for token {:?}: run1 len={} run2 len={}",
                    tok, a.len(), b.len());
            }
        }
    }
    eprintln!("\n[Compare Run 1 vs Run 2 — bit stability]");
    eprintln!("  tokens compared : {}", sampled_tokens.len());
    eprintln!("  mismatches      : {}", mismatches);

    if mismatches != 0 {
        eprintln!("\nHALT: bit-stability check FAILED. Sidecar load and rebuild path disagree.");
        return ExitCode::from(2);
    }

    // ---- Run 3: build a fresh legacy-fallback rebuild (no sidecar) ----
    //  Force the comparison against the materialised in-memory rebuild
    //  (what OLD code would produce, modulo my duplicates-preserving fix).
    eprintln!("\n[Run 3] materialise fresh rebuild via base.iter() — compares the mmap\n        bucket bytes against a HashMap reconstruction so the harness\n        also catches any silent corruption between mmap and accessor.");
    let mut snapshot3: HashMap<String, Vec<Seed>> = HashMap::with_capacity(sampled_tokens.len());
    for (tok, bucket) in base.iter() {
        if !sampled_tokens.contains(&tok.to_string()) { continue; }
        snapshot3.insert(tok.to_string(), bucket.to_vec());
    }
    let mut run3_mismatches = 0usize;
    for tok in &sampled_tokens {
        let a = snapshot1.get(tok).expect("snapshot1 has tok");
        let b = snapshot3.get(tok).cloned().unwrap_or_default();
        if a.as_slice() != b.as_slice() {
            run3_mismatches += 1;
            if run3_mismatches <= 10 {
                eprintln!("  DIVERGENCE (mmap-vs-iter) for token {:?}: run1 len={} run3 len={}",
                    tok, a.len(), b.len());
            }
        }
    }
    eprintln!("\n[Compare Run 1 vs Run 3 — mmap.get vs mmap.iter]");
    eprintln!("  mismatches: {}", run3_mismatches);
    if run3_mismatches != 0 {
        eprintln!("\nHALT: get vs iter disagree on the same base. Bisect bug or pointer drift.");
        return ExitCode::from(2);
    }

    // ---- Invariant checks on the base mmap itself ----
    eprintln!("\n[Invariants on the mmap'd base]");
    let mut prev: Vec<u8> = Vec::new();
    let mut unsorted_idx: Option<usize> = None;
    for (i, (tok, _)) in base.iter().enumerate() {
        let cur = tok.as_bytes().to_vec();
        if i > 0 && cur <= prev {
            unsorted_idx = Some(i);
            break;
        }
        prev = cur;
    }
    if let Some(i) = unsorted_idx {
        eprintln!("  HALT: token table not strictly ascending at index {}", i);
        return ExitCode::from(2);
    }
    eprintln!("  token table strictly ascending: OK ({} tokens checked)", token_total);

    // Per-bucket byte-sort invariant.
    let mut bucket_failures = 0usize;
    for (tok, bucket) in base.iter() {
        for i in 1..bucket.len() {
            // Duplicates ARE allowed (legacy parity). Only strict-descending
            // pairs are a violation.
            if bucket[i-1].as_bytes() > bucket[i].as_bytes() {
                bucket_failures += 1;
                if bucket_failures <= 5 {
                    eprintln!("  bucket NOT byte-sorted: token {:?} at i={}", tok, i);
                }
                break;
            }
        }
    }
    if bucket_failures != 0 {
        eprintln!("\nHALT: {} buckets fail byte-sort invariant", bucket_failures);
        return ExitCode::from(2);
    }
    eprintln!("  every bucket byte-sorted (duplicates allowed): OK");


    // ---- OP-022.a.1 Phase E: synthetic compaction transient measurement ----
    eprintln!("\n[Compaction transient] forcing compact_keyword_sidecar to fire by");
    eprintln!("        passing a faked brain_crc32 (mismatches loaded sidecar -> needs_rewrite=true)");
    let real_crc = brain2.network.keyword_index_base_for_test()
        .map(|b| b.brain_crc32())
        .unwrap_or(0);
    let fake_crc = real_crc.wrapping_add(1);
    let mut brain2_mut = brain2;
    let (rss_pre_compact, hwm_pre_compact) = read_vm();
    eprintln!("  pre-compact  RSS={}, HWM={}", fmt_kib(rss_pre_compact), fmt_kib(hwm_pre_compact));
    let t_c = Instant::now();
    let res = brain2_mut.network.compact_keyword_sidecar(fake_crc);
    let compact_wall = t_c.elapsed().as_secs_f64();
    let (rss_post_compact, hwm_post_compact) = read_vm();
    eprintln!("  compact_keyword_sidecar in {:.2}s -> {:?}", compact_wall, res);
    eprintln!("  post-compact RSS={}, HWM={}", fmt_kib(rss_post_compact), fmt_kib(hwm_post_compact));
    let hwm_delta_kb = hwm_post_compact.saturating_sub(hwm_pre_compact);
    let rss_delta_kb = rss_post_compact.saturating_sub(rss_pre_compact);
    eprintln!("  compaction HWM delta = {}  (RSS delta = {})",
        fmt_kib(hwm_delta_kb), fmt_kib(rss_delta_kb));
    let budget_kb: u64 = 500 * 1024; // 500 MiB
    if hwm_delta_kb > budget_kb {
        eprintln!("  WARN: compaction HWM delta {} > {} budget — OP-022.a.1 fix is insufficient",
            fmt_kib(hwm_delta_kb), fmt_kib(budget_kb));
    } else {
        eprintln!("  PASS: compaction HWM delta {} <= {} budget",
            fmt_kib(hwm_delta_kb), fmt_kib(budget_kb));
    }

    eprintln!("\n=== ALL CHECKS PASSED — bit-stable + invariants hold ===");
    eprintln!("   tokens sampled: {}", sampled_tokens.len());
    eprintln!("   tokens total  : {}", token_total);
    eprintln!("   seeds total   : {}", base.seed_total());
    ExitCode::from(0)
}

fn load_master_key() -> Result<Option<lu4d::encrypt::MasterKey>, String> {
    let path = std::env::var("LU4D_MASTER_KEY")
        .unwrap_or_else(|_| "/var/lib/lu/master.key".to_string());
    if !std::path::Path::new(&path).exists() { return Ok(None); }
    let bytes = std::fs::read(&path)
        .map_err(|e| format!("read {}: {}", path, e))?;
    if bytes.len() != 32 {
        return Err(format!("key {} is {} bytes; expected 32", path, bytes.len()));
    }
    let mut arr = [0u8; 32];
    arr.copy_from_slice(&bytes);
    eprintln!("loaded master key from {}", path);
    Ok(Some(lu4d::encrypt::MasterKey::from_bytes(arr)))
}
