//! OP-022.diag Phase A — save-spike composition diagnostic harness.
//!
//! Loads a brain (snapshot path expected; daemon must be stopped),
//! spawns a background sampler thread that captures /proc/self/status
//! (VmRSS, RssAnon, RssFile, VmHWM) every 250ms with a current-stage
//! marker, then drives:
//!   1. brain load
//!   2. to_brain_file_no_network (clone of side-table state)
//!   3. drop the brain_file
//!   4. save_to_format(V3enc) — includes the internal compact
//!   5. exit
//! Sampler output: /tmp/save_spike_samples.csv
//! Stage marker shared via /tmp/save_spike_stage (rewrite on each transition).
//!
//! Usage: op022_save_spike_diag <brain.lbr path> <output_path>
//!
//! Daemon-stopped: locked harness-isolation rule. The brain is loaded a second
//! time alongside the running daemon -> 30 GiB host OOMs (see
//! docs/op022a_rollback_playbook.md). Always:
//!   sudo -n /bin/systemctl stop lu4d-curriculum-ingest
//!   cp -p /var/lib/lu/brain.lbr /var/lib/lu/brain-diag-input.lbr
//!   ./target/release/op022_save_spike_diag /var/lib/lu/brain-diag-input.lbr /var/lib/lu/brain-diag-input.lbr
//!   sudo -n /bin/systemctl start lu4d-curriculum-ingest
//!
//! Baseline measurements (production brain at SHA 6742377a, OP-022.eh binary,
//! captured 2026-05-21). The +8 GiB save spike decomposes as:
//!   - 3.45 GiB: to_brain_file_no_network side-table clone
//!     (eliminated by OP-022.ef via SideTableSource trait + serialize_into)
//!   - 2.50 GiB: peak side-table bincode Vec at emit
//!     (eliminated by OP-022.ef via bincode::serialize_into to writer)
//!   - 1.04 GiB: network section persistent Vecs
//!     (node_seeds + node_entries + bond_keys + bond_entries in
//!     write_compressed_network_payload_to)
//!   - 1.85 GiB: per-block zstd compress scratch + glibc fragmentation
//!     (reduced by OP-022.h via Compressor::compress_to_buffer reuse)
//!   - 0.86 GiB: final side-table bincode burst at write end
//!
//! Total measured OP-022.ef + .h savings: -5.95 GiB save spike vs legacy path
//! (production a.2.1 22.69 GiB save+compact peak -> a.eh 15.34 GiB first save).
//!
//! Companion harness: scripts/buffer_diagnostic/ (read-only sampler for the
//! OLD-binary daemon; see scripts/buffer_diagnostic/README.md).
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};
use std::io::Write;
use lu4d::Brain;
use lu4d::BrainLoadMode;
use lu4d::BrainSaveFormat;

fn read_status() -> (u64, u64, u64, u64) {
    let s = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    let mut rss = 0u64; let mut anon = 0u64; let mut file = 0u64; let mut hwm = 0u64;
    for line in s.lines() {
        if let Some(r) = line.strip_prefix("VmRSS:")    { rss = r.trim().trim_end_matches(" kB").trim().parse().unwrap_or(0); }
        else if let Some(r) = line.strip_prefix("RssAnon:") { anon = r.trim().trim_end_matches(" kB").trim().parse().unwrap_or(0); }
        else if let Some(r) = line.strip_prefix("RssFile:") { file = r.trim().trim_end_matches(" kB").trim().parse().unwrap_or(0); }
        else if let Some(r) = line.strip_prefix("VmHWM:")   { hwm = r.trim().trim_end_matches(" kB").trim().parse().unwrap_or(0); }
    }
    (rss, anon, file, hwm)
}

fn read_stage() -> String {
    std::fs::read_to_string("/tmp/save_spike_stage").unwrap_or_else(|_| "INIT".into()).trim().to_string()
}
fn set_stage(s: &str) {
    let _ = std::fs::write("/tmp/save_spike_stage", s);
    eprintln!("[diag] >>>>> STAGE: {} <<<<<", s);
}

fn load_key() -> Result<Option<lu4d::encrypt::MasterKey>, String> {
    let p = std::env::var("LU4D_MASTER_KEY").unwrap_or_else(|_| "/var/lib/lu/master.key".to_string());
    if !std::path::Path::new(&p).exists() { return Ok(None); }
    let b = std::fs::read(&p).map_err(|e| format!("read {}: {}", p, e))?;
    if b.len() != 32 { return Err(format!("key {} is {} bytes", p, b.len())); }
    let mut a = [0u8; 32]; a.copy_from_slice(&b);
    Ok(Some(lu4d::encrypt::MasterKey::from_bytes(a)))
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 { eprintln!("usage: {} <brain.lbr in> <brain.lbr out>", args[0]); return ExitCode::from(1); }
    let brain_in = PathBuf::from(&args[1]);
    let brain_out = PathBuf::from(&args[2]);
    if !brain_in.exists() { eprintln!("[diag] FATAL: {} does not exist", brain_in.display()); return ExitCode::from(1); }
    set_stage("INIT");
    let csv_path = "/tmp/save_spike_samples.csv";
    let _ = std::fs::write(csv_path, "epoch_ms,t_rel_ms,stage,vm_rss_kb,rss_anon_kb,rss_file_kb,vm_hwm_kb\n");

    let stop = Arc::new(AtomicBool::new(false));
    let stop2 = stop.clone();
    let t0 = Instant::now();
    let sampler = std::thread::spawn(move || {
        let mut f = std::fs::OpenOptions::new().append(true).open(csv_path).unwrap();
        while !stop2.load(Ordering::Relaxed) {
            let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_millis();
            let t_rel = t0.elapsed().as_millis();
            let (rss, anon, file, hwm) = read_status();
            let stage = read_stage();
            let _ = writeln!(f, "{},{},{},{},{},{},{}", now, t_rel, stage, rss, anon, file, hwm);
            let _ = f.flush();
            std::thread::sleep(Duration::from_millis(250));
        }
    });

    set_stage("LOADING_KEY");
    let key = match load_key() { Ok(k) => k, Err(e) => { eprintln!("[diag] key load: {}", e); stop.store(true, Ordering::Relaxed); let _ = sampler.join(); return ExitCode::from(1); } };

    set_stage("LOADING_BRAIN");
    let mut brain = match Brain::load_or_new_with_mode_and_key(&brain_in, BrainLoadMode::MmapZeroCopy, key) {
        Ok(b) => b,
        Err(e) => { eprintln!("[diag] FATAL load: {}", e); stop.store(true, Ordering::Relaxed); let _ = sampler.join(); return ExitCode::from(1); }
    };
    set_stage("POST_LOAD_QUIESCE");
    std::thread::sleep(Duration::from_secs(3));

    // Ensure sidecar exists (the harness produces an empty keyword index
    // if not; we want apples-to-apples with prod save behavior).
    set_stage("PRE_ENSURE_KEYWORD");
    brain.network.ensure_keyword_index_built();
    set_stage("POST_ENSURE_KEYWORD_QUIESCE");
    std::thread::sleep(Duration::from_secs(3));

    // Probe: build BrainFile-no-network OUTSIDE save_to_format to measure
    // the side-table clone cost in isolation.
    set_stage("PRE_BUILD_BRAIN_FILE");
    let brain_file = brain.to_brain_file_no_network(3 /* FORMAT_VERSION_V3 */);
    set_stage("POST_BUILD_BRAIN_FILE");
    std::thread::sleep(Duration::from_secs(3));
    set_stage("PRE_DROP_BRAIN_FILE");
    drop(brain_file);
    set_stage("POST_DROP_BRAIN_FILE");
    std::thread::sleep(Duration::from_secs(3));

    // The real save (this is what production does — full save_to_format
    // including internal to_brain_file_no_network + write_v3 + compact).
    set_stage("PRE_SAVE_TO_FORMAT");
    let save_res = brain.save_to_format(&brain_out, BrainSaveFormat::V3BinaryEncrypted);
    set_stage("POST_SAVE_TO_FORMAT");
    std::thread::sleep(Duration::from_secs(3));
    match save_res {
        Ok(_) => eprintln!("[diag] save_to_format OK"),
        Err(e) => { eprintln!("[diag] save_to_format ERR: {:?}", e); stop.store(true, Ordering::Relaxed); let _ = sampler.join(); return ExitCode::from(2); }
    }

    set_stage("DONE");
    std::thread::sleep(Duration::from_secs(2));
    stop.store(true, Ordering::Relaxed);
    let _ = sampler.join();
    let (rss, anon, file, hwm) = read_status();
    eprintln!("[diag] FINAL: vm_rss={} KiB rss_anon={} KiB rss_file={} KiB vm_hwm={} KiB", rss, anon, file, hwm);
    ExitCode::from(0)
}
