//! OP-BRAIN-V5 Phase 0.c — bounded-RSS assertion CLI.
//!
//! Ad-hoc threshold checker that every subsequent v5 phase (1.0 through
//! 8) calls to gate progress. Sits on top of
//! [`lu_brain_bench::harness`] (the Phase 0.c library deliverable).
//!
//! ## Usage
//!
//! ```text
//! bench_v5_assert \
//!     --fixture <path>          \
//!     --phase <tag>             \
//!     --workload <name>         \
//!     --threshold-rss-gib <f64> \
//!     [--out <report-path>]     \
//!     [--interval-ms <u64>]     \
//!     [--repo-root <path>]
//! ```
//!
//! The `--workload` selector picks from a registered set built by
//! `register_workloads` (see source). Phase 1+ briefs register new
//! workloads here as they ship.
//!
//! ### Canonical fixture
//!
//! ```text
//! /var/lib/lu/bench-fixtures/v5-prod-shape-5m.bench
//! ```
//!
//! 5M articles, 27.5M nodes, 82.3M bonds, ~6.5 GB on disk. Generated
//! by `gen_prod_shape_fixture` (Phase 0.b). Sidecar at the same path
//! with `.stats.json` extension.
//!
//! ## RssReport JSON
//!
//! Pretty-printed sidecar emitted to `--out`. Top-level keys:
//!
//! - `verdict` — pass/fail, threshold, peak, phase, workload
//! - `rss_breakdown` — min, p50, p90, p99, peak (GiB)
//! - `sampler` — interval_ms, observed_sample_count,
//!   retained_sample_count, reservoir_engaged
//! - `workload_meta` — duration_secs, workload_result
//! - `fixture` — path, format_version, source_corpus,
//!   source_corpus_revision, generator_binary_md5
//! - `env` — binary_md5, git_head, hostname, run_timestamp_utc
//!
//! ## Budget API contract
//!
//! The harness LETS THE WORKLOAD RUN TO COMPLETION even if peak RSS
//! exceeds the threshold. The verdict is `pass = peak_rss_gib <=
//! threshold_rss_gib`. The binary exits with code 0 iff `pass = true`;
//! callers that want hard-abort behavior in scripts can rely on the
//! exit code without parsing the JSON.
//!
//! ## Adding a new workload
//!
//! 1. Implement an `FnOnce(&mut WorkloadCtx) -> Result<u64, HarnessError>`.
//! 2. Register it in [`register_workloads`].
//! 3. Decide the RSS budget in your phase brief, not in 0.c.

use std::path::PathBuf;
use std::process::ExitCode;

use lu_brain_bench::harness::{
    self_md5, resolve_git_head, resolve_hostname, with_rss_budget, workload_linear_scan,
    FixtureReader, RssBudget, WorkloadCtx, WorkloadRegistry,
    DEFAULT_SAMPLE_INTERVAL_MS,
};

fn main() -> ExitCode {
    match run() {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => {
            // Workload completed but exceeded budget. Verdict is
            // already on stderr + in the JSON sidecar; exit code 1 so
            // CI / phase-gate scripts pick it up.
            ExitCode::from(1)
        }
        Err(e) => {
            eprintln!("FATAL: {}", e);
            ExitCode::from(2)
        }
    }
}

fn run() -> Result<bool, String> {
    let args = parse_args()?;
    let mut registry = WorkloadRegistry::new();
    register_workloads(&mut registry);

    eprintln!("bench_v5_assert (OP-BRAIN-V5 Phase 0.c)");
    eprintln!("  fixture            = {}", args.fixture.display());
    eprintln!("  phase              = {}", args.phase);
    eprintln!("  workload           = {}", args.workload);
    eprintln!("  threshold_rss_gib  = {:.4}", args.threshold_rss_gib);
    eprintln!("  sampler interval   = {} ms", args.interval_ms);
    eprintln!("  out                = {}",
        args.out.as_ref().map(|p| p.display().to_string()).unwrap_or_else(|| "<stdout-only>".to_string()));
    eprintln!("  registered loads   = {:?}", registry.names());

    // Construct workload function via registry FIRST so unknown workload
    // names fail before we open the fixture.
    let workload_fn = registry.dispatch(&args.workload).map_err(|e| e.to_string())?;

    // Open the fixture (validates magic + format_version + reads header).
    let fixture = FixtureReader::open(&args.fixture).map_err(|e| e.to_string())?;
    let header = fixture.header().clone();
    eprintln!("  fixture header     = format_v{} corpus={} rev={} md5={}",
        header.format_version, header.source_corpus, header.source_corpus_revision,
        header.generator_binary_md5);

    // Env metadata.
    let binary_md5 = self_md5().map_err(|e| format!("self_md5: {}", e))?;
    let git_head = resolve_git_head(&args.repo_root);
    let hostname = resolve_hostname();
    eprintln!("  env                = host={} git={} bin_md5={}", hostname, git_head, binary_md5);

    let budget = RssBudget {
        threshold_rss_gib: args.threshold_rss_gib,
        interval: std::time::Duration::from_millis(args.interval_ms),
        phase: args.phase.clone(),
        workload: args.workload.clone(),
        fixture_path: args.fixture.clone(),
        fixture_header: header,
        binary_md5,
        git_head,
        hostname,
    };

    let mut ctx = WorkloadCtx { fixture, fixture_path: args.fixture.clone() };

    let (report, _result) = with_rss_budget::<_, u64>(budget, move || workload_fn(&mut ctx))
        .map_err(|e| e.to_string())?;

    eprintln!();
    eprintln!("verdict");
    eprintln!("  pass               = {}", report.pass);
    eprintln!("  threshold_rss_gib  = {:.4}", report.threshold_rss_gib);
    eprintln!("  peak_rss_gib       = {:.4}", report.peak_rss_gib);
    eprintln!("  duration_secs      = {:.2}", report.duration_secs);
    eprintln!("  workload_result    = {}", report.workload_result);
    eprintln!("  samples observed   = {} (retained {}, reservoir_engaged={})",
        report.observed_sample_count, report.retained_sample_count, report.reservoir_engaged);
    eprintln!("  p50/p90/p99 (GiB)  = {:.4} / {:.4} / {:.4}",
        report.p50_rss_gib, report.p90_rss_gib, report.p99_rss_gib);

    if let Some(out_path) = args.out.as_ref() {
        report.write_json(out_path).map_err(|e| format!("write_json: {}", e))?;
        eprintln!("  report             = {}", out_path.display());
    }

    Ok(report.pass)
}

// ----------------------------------------------------------------------
// Workload registry. Phase 1+ briefs add new workloads here.
// ----------------------------------------------------------------------

fn register_workloads(registry: &mut WorkloadRegistry) {
    // Phase 0.c self-test workload — the smoke that proves the harness
    // works end-to-end before any Phase 1+ workload runs against it.
    registry.register("linear_scan", workload_linear_scan);
}

// ----------------------------------------------------------------------
// CLI parser.
// ----------------------------------------------------------------------

struct Args {
    fixture: PathBuf,
    phase: String,
    workload: String,
    threshold_rss_gib: f64,
    out: Option<PathBuf>,
    interval_ms: u64,
    repo_root: PathBuf,
}

fn parse_args() -> Result<Args, String> {
    let mut fixture: Option<PathBuf> = None;
    let mut phase: Option<String> = None;
    let mut workload: Option<String> = None;
    let mut threshold_rss_gib: Option<f64> = None;
    let mut out: Option<PathBuf> = None;
    let mut interval_ms: u64 = DEFAULT_SAMPLE_INTERVAL_MS;
    let mut repo_root = PathBuf::from("/opt/lu/lu/v4");

    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--fixture" => fixture = Some(PathBuf::from(it.next().ok_or("--fixture needs a value")?)),
            "--phase" => phase = Some(it.next().ok_or("--phase needs a value")?),
            "--workload" => workload = Some(it.next().ok_or("--workload needs a value")?),
            "--threshold-rss-gib" => {
                let v = it.next().ok_or("--threshold-rss-gib needs a value")?;
                threshold_rss_gib = Some(v.parse().map_err(|e| format!("--threshold-rss-gib: {}", e))?);
            }
            "--out" => out = Some(PathBuf::from(it.next().ok_or("--out needs a value")?)),
            "--interval-ms" => {
                let v = it.next().ok_or("--interval-ms needs a value")?;
                interval_ms = v.parse().map_err(|e| format!("--interval-ms: {}", e))?;
            }
            "--repo-root" => repo_root = PathBuf::from(it.next().ok_or("--repo-root needs a value")?),
            "--help" | "-h" => { print_help(); std::process::exit(0); }
            other => return Err(format!("unknown arg: {}", other)),
        }
    }

    Ok(Args {
        fixture: fixture.ok_or("--fixture is required")?,
        phase: phase.ok_or("--phase is required")?,
        workload: workload.ok_or("--workload is required")?,
        threshold_rss_gib: threshold_rss_gib.ok_or("--threshold-rss-gib is required")?,
        out,
        interval_ms,
        repo_root,
    })
}

fn print_help() {
    println!("bench_v5_assert — OP-BRAIN-V5 Phase 0.c bounded-RSS assertion CLI");
    println!();
    println!("Usage: bench_v5_assert --fixture <path> --phase <tag> --workload <name>");
    println!("                       --threshold-rss-gib <float> [options]");
    println!();
    println!("  --fixture PATH           Path to a Phase 0.b .bench fixture");
    println!("  --phase TAG              Phase identifier (e.g. 0c, 1.0, 7.0)");
    println!("  --workload NAME          Workload registry key");
    println!("  --threshold-rss-gib N    Pass/fail RSS budget (GiB)");
    println!("  --out PATH               Optional JSON report sidecar path");
    println!("  --interval-ms N          Sampler interval (default 250)");
    println!("  --repo-root PATH         Repo root for git HEAD resolution");
    println!("                           (default /opt/lu/lu/v4)");
    println!();
    println!("Exit codes:");
    println!("  0  pass (peak_rss_gib <= threshold)");
    println!("  1  workload completed but exceeded budget");
    println!("  2  fatal error (fixture invalid, unknown workload, ...)");
    println!();
    println!("Canonical fixture:");
    println!("  /var/lib/lu/bench-fixtures/v5-prod-shape-5m.bench");
}

