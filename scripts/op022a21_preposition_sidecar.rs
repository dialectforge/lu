//! OP-022.a.2.1 — pre-position sidecar against live brain.
//!
//! Loads brain.lbr (no sidecar present) via the new code path, runs
//! ensure_keyword_index_built to write the sidecar with the CORRECT
//! stamped brain_crc32, verifies a few token lookups, exits.
//!
//! No synthetic compaction, no bit-stability check (already validated
//! in Phase E for this binary). Safe to leave sidecar in place for
//! daemon startup.
//!
//! Usage:  op022a21_preposition_sidecar /var/lib/lu/brain.lbr
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;
use lu4d::Brain;
use lu4d::BrainLoadMode;

fn read_vm() -> (u64, u64) {
    let s = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    let mut rss = 0u64; let mut hwm = 0u64;
    for line in s.lines() {
        if let Some(r) = line.strip_prefix("VmRSS:") { rss = r.trim().trim_end_matches(" kB").trim().parse().unwrap_or(0); }
        else if let Some(r) = line.strip_prefix("VmHWM:") { hwm = r.trim().trim_end_matches(" kB").trim().parse().unwrap_or(0); }
    }
    (rss, hwm)
}
fn fmt(kb: u64) -> String {
    if kb < 1024 { format!("{} kB", kb) }
    else if kb < 1024*1024 { format!("{:.2} MiB", kb as f64 / 1024.0) }
    else { format!("{:.2} GiB", kb as f64 / (1024.0*1024.0)) }
}
fn load_key() -> Result<Option<lu4d::encrypt::MasterKey>, String> {
    let path = std::env::var("LU4D_MASTER_KEY").unwrap_or_else(|_| "/var/lib/lu/master.key".to_string());
    if !std::path::Path::new(&path).exists() { return Ok(None); }
    let b = std::fs::read(&path).map_err(|e| format!("read {}: {}", path, e))?;
    if b.len() != 32 { return Err(format!("key {} is {} bytes", path, b.len())); }
    let mut a = [0u8; 32]; a.copy_from_slice(&b);
    eprintln!("[prepos] loaded master key from {}", path);
    Ok(Some(lu4d::encrypt::MasterKey::from_bytes(a)))
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 { eprintln!("usage: {} <brain.lbr path>", args[0]); return ExitCode::from(1); }
    let brain_path = PathBuf::from(&args[1]);
    if !brain_path.exists() { eprintln!("[prepos] FATAL: {} does not exist", brain_path.display()); return ExitCode::from(1); }
    let sidecar_path = brain_path.with_extension(format!("{}.kw", brain_path.extension().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default()));
    if sidecar_path.exists() { eprintln!("[prepos] FATAL: sidecar already exists at {}; delete first", sidecar_path.display()); return ExitCode::from(1); }
    eprintln!("[prepos] brain   : {}", brain_path.display());
    eprintln!("[prepos] sidecar : {} (will be created)", sidecar_path.display());
    let key = match load_key() { Ok(k) => k, Err(e) => { eprintln!("[prepos] key load: {}", e); return ExitCode::from(1); } };
    let (rss0, hwm0) = read_vm();
    eprintln!("[prepos] pre-load   RSS={} HWM={}", fmt(rss0), fmt(hwm0));
    let t_load = Instant::now();
    let mut brain = match Brain::load_or_new_with_mode_and_key(&brain_path, BrainLoadMode::MmapZeroCopy, key) {
        Ok(b) => b,
        Err(e) => { eprintln!("[prepos] FATAL load: {}", e); return ExitCode::from(1); }
    };
    let load_wall = t_load.elapsed().as_secs_f64();
    let (rss1, hwm1) = read_vm();
    eprintln!("[prepos] post-load  RSS={} HWM={} ({}s)", fmt(rss1), fmt(hwm1), load_wall);
    let t_ensure = Instant::now();
    brain.network.ensure_keyword_index_built();
    let ensure_wall = t_ensure.elapsed().as_secs_f64();
    let (rss2, hwm2) = read_vm();
    eprintln!("[prepos] post-ensure RSS={} HWM={} ({}s)", fmt(rss2), fmt(hwm2), ensure_wall);
    if !sidecar_path.exists() { eprintln!("[prepos] FATAL: sidecar was not created at {}", sidecar_path.display()); return ExitCode::from(1); }
    let meta = std::fs::metadata(&sidecar_path).unwrap();
    eprintln!("[prepos] sidecar size: {} bytes ({:.2} MiB)", meta.len(), meta.len() as f64 / (1024.0*1024.0));
    let base = brain.network.keyword_index_base_for_test();
    if let Some(b) = base {
        eprintln!("[prepos] has_base = true, token_count = {}, seed_total = {}", b.token_count(), b.seed_total());
        let mut tokens_seen = 0usize; let mut seeds_seen = 0usize;
        for (tok, bucket) in b.iter().take(10) { tokens_seen += 1; seeds_seen += bucket.len(); let _ = tok; }
        eprintln!("[prepos] iter sanity check: first {} tokens hold {} seeds (sample)", tokens_seen, seeds_seen);
    } else { eprintln!("[prepos] FATAL: has_base = false after ensure_keyword_index_built"); return ExitCode::from(2); }
    let (_rss3, hwm3) = read_vm();
    eprintln!("[prepos] DONE. final HWM={} (rebuild HWM delta from pre-ensure = {})", fmt(hwm3), fmt(hwm3.saturating_sub(hwm1)));
    ExitCode::from(0)
}
