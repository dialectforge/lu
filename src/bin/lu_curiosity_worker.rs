//! `lu_curiosity_worker` — the FILL half of the curiosity engine.
//!
//! # What this is
//!
//! Lu's curiosity engine has two halves. DETECT runs inside spread
//! activation and fires on essentially every user query; it lives in the
//! public `lu_guest_http` process. FILL — hub-fill, triangle closure,
//! single-hop research and chained research — has to WRITE bonds into
//! the canonical brain.
//!
//! Since the OP-007 privilege split those two halves have been in
//! different processes with no shared surface, so the loop has never
//! closed: serving had the producer and no consumer, training had the
//! consumer (the 2s auto-tick thread in the admin `mcp` module) and no
//! producer, because no ingest script ever calls `ask`.
//!
//! This binary is the consumer, as its own process. It:
//!
//!   1. loads the canonical brain READ-WRITE,
//!   2. hydrates its wonder queue from the durable journal the guest
//!      appends to (`lu4d::wonder_sidecar`),
//!   3. runs N `idle_tick`s — the real filling code, unchanged,
//!   4. flushes, and retires the wonders it consumed from the journal.
//!
//! # Why a separate process and not a thread in the guest
//!
//! Because the guest is internet-facing. Spawning the fill driver inside
//! it would make the public process a writer to the canonical brain and
//! discard the OP-007 privsep property. This binary has no listening
//! socket, no auth surface, and is never reachable from the network; the
//! guest has no brain-write capability and is confined by systemd to a
//! read-only view of /var/lib/lu. The journal directory is the only
//! thing they share, and it flows one way.
//!
//! # The operational constraint you must know about
//!
//! THIS PROCESS AND `lu-guest-http` CANNOT RUN AT THE SAME TIME on the
//! current box. Loading the 233 GiB v5 brain peaks around 54 GiB and
//! settles near 26 GiB, on 62 GiB of RAM. Two loaders do not fit;
//! earlier attempts to co-load a second reader were OOM-killed. There is
//! also a correctness reason: the guest watches the committed MANIFEST
//! frontier and re-opens a fresh reader when it advances, so a
//! concurrent writer would trigger a second full load inside the guest
//! itself.
//!
//! So the loop is a SLEEP CYCLE, not a background thread: Lu wonders
//! while awake (guest running, journal growing), and consolidates while
//! asleep (guest stopped, this worker draining). The systemd unit
//! declares `Conflicts=lu-guest-http.service` so that is enforced rather
//! than merely documented. Training is a third mutually-exclusive tenant
//! for the same reason.

use std::path::PathBuf;
use std::time::Instant;

use lu4d::encrypt::{MasterKey, DEFAULT_KEYFILE_PATH};
use lu4d::wonder_sidecar::WonderSidecar;
use lu4d::Brain;

/// `--no-research`: a fetcher that refuses every request.
///
/// Not a stub standing in for missing functionality -- it is the
/// mechanism by which `--no-research` is enforced. `research_one` sees a
/// transport failure and returns `RouterFailed`, which `idle_tick`
/// already handles. Internal fill (hub-fill / triangle closure) runs
/// first and unaffected, so `--no-research` means "consolidate what you
/// already know, do not go online".
struct OfflineFetcher;

impl lu4d::research::http::HttpFetcher for OfflineFetcher {
    fn fetch(
        &self,
        _url: &str,
    ) -> Result<lu4d::research::http::HttpResponse, lu4d::research::http::HttpError> {
        Err(lu4d::research::http::HttpError::Network(
            "--no-research: network access disabled for this run".to_string(),
        ))
    }
}

fn usage() -> ! {
    eprintln!(
        "lu_curiosity_worker --brain <dir> --wonder-queue <dir> [options]\n\
         \n\
         Drains the durable wonder journal and fills the gaps it names.\n\
         Must NOT run while lu-guest-http or an ingest unit holds the brain.\n\
         \n\
         Options:\n\
         --keyfile <path>   master key (default {})\n\
         --ticks <n>        max idle_tick iterations (default 10)\n\
         --auto-apply       apply fills automatically instead of leaving them\n\
                            Pending for lu4d_approve_changes. OFF by default.\n\
         --min-confidence <f>  confidence floor for --auto-apply (default 0.75)\n\
         --no-research      internal fill only; never touch the network\n\
         --dry-run          load, hydrate and report; change nothing\n\
         --rebuild-keyword-index  rebuild the keyword sidecar even if this run\n\
                            applied nothing (recovery: the guest is read-only\n\
                            and cannot repair a stale sidecar itself)\n",
        DEFAULT_KEYFILE_PATH
    );
    std::process::exit(2);
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut brain_path: Option<PathBuf> = None;
    let mut wonder_dir: Option<PathBuf> = None;
    let mut keyfile = PathBuf::from(DEFAULT_KEYFILE_PATH);
    let mut ticks: u32 = 10;
    let mut auto_apply = false;
    let mut min_confidence: f32 = 0.75;
    let mut allow_research = true;
    let mut dry_run = false;
    let mut force_kw_rebuild = false;

    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--brain" => brain_path = args.next().map(PathBuf::from),
            "--wonder-queue" => wonder_dir = args.next().map(PathBuf::from),
            "--keyfile" => {
                if let Some(p) = args.next() {
                    keyfile = PathBuf::from(p);
                }
            }
            "--ticks" => {
                ticks = args
                    .next()
                    .and_then(|v| v.parse().ok())
                    .ok_or("--ticks needs a number")?
            }
            "--auto-apply" => auto_apply = true,
            "--min-confidence" => {
                min_confidence = args
                    .next()
                    .and_then(|v| v.parse().ok())
                    .ok_or("--min-confidence needs a float")?
            }
            "--no-research" => allow_research = false,
            "--dry-run" => dry_run = true,
            "--rebuild-keyword-index" => force_kw_rebuild = true,
            "-h" | "--help" => usage(),
            other => {
                eprintln!("unknown arg {:?}", other);
                usage()
            }
        }
    }

    let brain_path = brain_path.ok_or("--brain <dir> is required")?;
    let wonder_dir = wonder_dir.ok_or("--wonder-queue <dir> is required")?;

    // ---- Safety interlock -------------------------------------------
    // The systemd unit enforces this with Conflicts=, but the binary is
    // also runnable by hand, and by hand is exactly when someone forgets.
    // A second full loader on this box does not fail gracefully; it gets
    // OOM-killed somewhere in the middle of a 6-minute load, possibly
    // taking the live service with it.
    if let Ok(out) = std::process::Command::new("pgrep")
        .args(["-x", "lu_guest_http"])
        .output()
    {
        if out.status.success() && !out.stdout.is_empty() {
            return Err(
                "REFUSING TO START: lu_guest_http is running. This box cannot hold two \
                 loaders of the canonical brain (peak ~54 GiB each on 62 GiB). Stop the \
                 guest first: sudo systemctl stop lu-guest-http"
                    .into(),
            );
        }
    }

    let sidecar = WonderSidecar::in_dir(&wonder_dir)?;
    eprintln!(
        "[curiosity-worker] journal {} | brain {} | ticks={} auto_apply={} research={} dry_run={}",
        sidecar.path().display(),
        brain_path.display(),
        ticks,
        auto_apply,
        allow_research,
        dry_run,
    );

    // Report what is waiting BEFORE the expensive load, so a run against
    // an empty journal costs nothing and says so.
    let journal = sidecar.load()?;
    let distinct = journal.dedup_signals();
    eprintln!(
        "[curiosity-worker] journal holds {} record(s) -> {} distinct gap(s) ({} unparseable line(s))",
        journal.records.len(),
        distinct.len(),
        journal.skipped,
    );
    if distinct.is_empty() && !force_kw_rebuild {
        eprintln!("[curiosity-worker] nothing to do. Exiting without loading the brain.");
        return Ok(());
    }
    for (i, s) in distinct.iter().take(10).enumerate() {
        let label = journal
            .labels()
            .get(&s.seed)
            .cloned()
            .unwrap_or_else(|| "<unlabelled>".to_string());
        eprintln!(
            "  [{}] {:?} intensity={:.3} seed={} label={}",
            i, s.signal_type, s.intensity, &s.seed.to_string()[..12], label,
        );
    }
    if dry_run {
        eprintln!("[curiosity-worker] --dry-run: stopping before load.");
        return Ok(());
    }

    // ---- Load READ-WRITE --------------------------------------------
    let t0 = Instant::now();
    let key = MasterKey::load_from_keyfile(&keyfile).ok();
    eprintln!("[curiosity-worker] loading brain read-write (this takes minutes)...");
    let mut brain = Brain::load_v5(&brain_path, key)?;
    let n_nodes = brain.network.node_count();
    eprintln!(
        "[curiosity-worker] loaded in {:.1}s -- {} concepts, {} associations",
        t0.elapsed().as_secs_f32(),
        n_nodes,
        brain.network.bond_count(),
    );

    // ---- OP-RECALL-1 (R5) VERIFICATION -------------------------------
    // R5's blast radius was arithmetic. `node_count()` returned 0 on
    // every cache-backed v5 brain, so `term_idf` short-circuited on
    // `n < 2.0` and returned a flat 1.0, `max_idf` was 1.0, and
    //
    //     base = 0.3 + 0.7 * (idf / max_idf).min(1.0) = 1.0
    //
    // for EVERY token, always. The entire TF-IDF weighting layer was a
    // constant, which is the arithmetic root of the bad seed ranking.
    //
    // Printed rather than assumed: if the spread below is flat, the fix
    // did not take. The worker is the cheapest place to measure it --
    // this box cannot afford a second full loader just for a table.
    if n_nodes == 0 {
        eprintln!("[curiosity-worker] WARNING: node_count()==0 -- R5 regression, IDF is inert");
    } else {
        let max_idf = (n_nodes as f32).ln();
        eprintln!(
            "[curiosity-worker] R5 check: max_idf={:.4} over {} nodes",
            max_idf, n_nodes
        );
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        for t in [
            "dog", "python", "language", "hash", "photosynthesis",
            "mitochondria", "volcano", "gravity", "flibbertigibbet",
        ] {
            let idf = brain.network.term_idf(t);
            let base = 0.3 + 0.7 * (idf / max_idf).min(1.0);
            if base < lo { lo = base; }
            if base > hi { hi = base; }
            eprintln!(
                "[curiosity-worker]   token {:<16} idf={:>8.4}  base_weight={:.4}",
                t, idf, base
            );
        }
        eprintln!(
            "[curiosity-worker] R5 verdict: base weights span {:.4}..{:.4} ({})",
            lo,
            hi,
            if hi - lo > 0.01 {
                "DIFFERENTIATING -- IDF is live"
            } else {
                "FLAT -- IDF still inert, investigate"
            },
        );
    }

    // ---- Hydrate the queue from the journal --------------------------
    brain.attach_wonder_sidecar(sidecar.clone());
    let restored = brain.hydrate_wonder_queue_from_sidecar()?;
    eprintln!(
        "[curiosity-worker] hydrated {} distinct wonder(s) into the queue",
        restored
    );

    brain.curiosity_config.auto_apply = auto_apply;
    brain.curiosity_config.auto_apply_min_confidence = min_confidence;

    // ---- Drain -------------------------------------------------------
    let pending_before = brain.pending.total_count();
    let bonds_before = brain.network.bond_count();
    let mut consumed: Vec<String> = Vec::new();
    let mut did_work = 0u32;

    for i in 0..ticks {
        if brain.wonder_queue.is_empty() {
            eprintln!("[curiosity-worker] queue drained after {} tick(s)", i);
            break;
        }
        // `min_idle_ms = 0`: this process exists only to do idle work, so
        // there is no "is the brain busy" question to answer. The gate is
        // meaningful in the interactive daemon, not here.
        let res = if allow_research {
            let fetcher = brain.default_research_fetcher();
            brain.idle_tick(&fetcher, 0)
        } else {
            // A fetcher that always fails, so `research_one` cannot reach
            // the network. Internal fill still runs and still short-
            // circuits before research when it produces anything.
            brain.idle_tick(&OfflineFetcher, 0)
        };
        eprintln!(
            "[curiosity-worker] tick {}: outcome={:?} did_work={} pending_added={} seed={:?}",
            i, res.outcome, res.did_work, res.pending_added_count, res.wonder_seed_processed,
        );
        if let Some(s) = res.wonder_seed_processed.clone() {
            consumed.push(s);
        }
        if res.did_work {
            did_work += 1;
        } else {
            eprintln!("[curiosity-worker] tick reported no work; stopping.");
            break;
        }
        let _ = brain.flush_gaps();
    }

    let pending_after = brain.pending.total_count();
    let bonds_after = brain.network.bond_count();
    // Did anything actually reach the GRAPH? Drives the sidecar rebuild
    // below: a run that only PROPOSED changes has invalidated nothing
    // and must not pay for a full index rebuild.
    let applied_count = brain.pending.applied_count();
    let applied_any = applied_count > 0;
    eprintln!(
        "[curiosity-worker] ticks_with_work={} pending {} -> {} | bonds {} -> {}",
        did_work, pending_before, pending_after, bonds_before, bonds_after,
    );

    // ---- Durability ---------------------------------------------------
    // The fills live in the MemTable + WAL until a flush commits them to
    // an SST. Without this the guest, which serves the COMMITTED
    // frontier, would never see them.
    match brain.flush_now() {
        Ok(()) => eprintln!("[curiosity-worker] flushed MemTable -> L0 (committed frontier advanced)"),
        Err(e) => eprintln!("[curiosity-worker] WARNING: flush failed: {}", e),
    }

    // ---- Rebuild the keyword sidecar THIS PROCESS invalidated --------
    // See the module header on the sleep cycle. `build_keyword_index_v5`
    // is called directly rather than `ensure_keyword_index_built`,
    // because the latter early-returns when a base is already installed
    // -- and the base installed right now is the one loaded before the
    // writes, i.e. exactly the stale one.
    if applied_any || force_kw_rebuild {
        eprintln!(
            "[curiosity-worker] {} bond(s) landed (force={}) -- rebuilding the keyword sidecar",
            applied_count, force_kw_rebuild
        );
        let t_kw = Instant::now();
        match brain.build_keyword_index_v5() {
            Ok(()) => eprintln!(
                "[curiosity-worker] keyword sidecar rebuilt in {:.1}s",
                t_kw.elapsed().as_secs_f32()
            ),
            Err(e) => eprintln!(
                "[curiosity-worker] CRITICAL: keyword sidecar rebuild FAILED ({}). The guest \
                 cannot rebuild it (read-only) and will answer every query with \
                 \"Nothing in my brain matched\". Rebuild it before serving.",
                e
            ),
        }
        if let Err(e) = brain.flush_now() {
            eprintln!("[curiosity-worker] WARNING: post-sidecar flush failed: {}", e);
        }
    } else {
        eprintln!("[curiosity-worker] nothing applied; keyword sidecar still valid, left alone");
    }

    // ---- Retire consumed wonders from the journal ---------------------
    // Only the ones actually processed. A wonder we never got to stays,
    // so the next run picks it up rather than losing it.
    if !consumed.is_empty() {
        let keep: Vec<_> = journal
            .records
            .iter()
            .filter(|r| {
                let prefix = r.signal.seed.to_string();
                !consumed.iter().any(|c| prefix.starts_with(c.as_str()))
            })
            .cloned()
            .collect();
        let dropped = journal.records.len() - keep.len();
        match sidecar.rewrite(&keep) {
            Ok(()) => eprintln!(
                "[curiosity-worker] journal compacted: retired {} record(s), {} remain",
                dropped,
                keep.len()
            ),
            Err(e) => eprintln!("[curiosity-worker] WARNING: journal compaction failed: {}", e),
        }
    }

    eprintln!("[curiosity-worker] done in {:.1}s", t0.elapsed().as_secs_f32());
    Ok(())
}
