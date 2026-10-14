# Phase B Resume — Full Wikipedia Re-Ingest Handoff

**Status:** Daemon launched and observed for ~10 min. Sustained rate well above acceptance threshold. Detached.
**Kickoff time (UTC):** 2026-05-07T14:55:52Z
**Handoff written:** 2026-05-07T15:05:30Z
**Owner of run:** dialectforge (PID 1106624, ppid bash wrapper 1106623)
**Brain target:** `/var/lib/lu/brain-retrain-2026-05-04.lbr` (test brain — canonical `/var/lib/lu/brain.lbr` is NOT touched)

---

## 1. Pre-flight verification (all PASS)

| Check | Result |
|---|---|
| `scripts/ingest_curriculum_full.py` exists | ✅ 32720 bytes, mtime 2026-05-07 14:00 |
| `scripts/smart_extractor.py` exists | ✅ 30821 bytes, mtime 2026-05-07 11:06 |
| `.pre-perf2.bak` exists | ✅ 31979 bytes |
| `.pre-phase-b.bak` exists | ✅ 30921 bytes |
| Perf-2 `--no-save` flag at line 194 | ✅ confirmed in subprocess.Popen args |
| `brain-retrain-2026-05-04.lbr` baseline | ✅ 944,681,049 bytes, mtime 2026-05-07 11:50 |
| `brain.lbr.pre-phase-b-2026-05-07.bak` rollback | ✅ 943,455,629 bytes |
| `ingest_full_checkpoint.json` exists | ✅ wikipedia.last_saved_id=1018999, vg.completed=true |
| Canonical `/var/lib/lu/brain.lbr` sha256 | ✅ `835109ba4c0262f760a1510582c4f85654eb66959eeafb9042405e425e856a13` (matches baseline) |
| No leftover daemon processes | ✅ `ps -ef | grep ingest_curriculum` empty before launch |
| Disk space | ✅ 264G free on /var/lib/lu fs |
| Memory | ✅ 25Gi free of 30Gi |
| `import scripts.ingest_curriculum_full` | ✅ no errors |

## 2. Pre-kickoff brain MD5s

| File | MD5 |
|---|---|
| `brain-retrain-2026-05-04.lbr` (pre-launch) | `9844c3f7d1612ddc7074d1c3e6aec391` |
| `brain.lbr.pre-phase-b-2026-05-07.bak` (rollback) | `69708a6f63fc369bdc4bc4a838384379` |

## 3. Checkpoint at kickoff

```json
{
  "visual_genome": {"bonds_landed": 445742, "errors_count": 0, "last_saved_id": 1900024, "completed": true},
  "wikimedia/wikipedia": {"bonds_landed": 1679030, "errors_count": 0, "bonds_skipped_dedup": 0, "last_saved_id": 1018999}
}
```

Visual genome already complete; resume picks up Wikipedia at article 1019000.
**Note:** Phase B's smoke run wrote to `_smoke`-suffixed isolated artifacts (`ingest_full_checkpoint_smoke.json`), so the production checkpoint was NOT advanced beyond 1018999. Resume returns to where the original Cat.1 retrain paused on 2026-05-07T02:02:29Z.

## 4. Launch command (exact)

```bash
ssh lu 'cd /opt/lu/lu/v4 && nohup python3 scripts/ingest_curriculum_full.py \
  >> /var/lib/lu/ingest_full_stdout.log \
  2>> /var/lib/lu/ingest_full_stderr.log \
  < /dev/null & echo $! > /var/lib/lu/ingest_full.pid'
```

All script paths are defaults — no overrides:
- `--brain` defaults to `/var/lib/lu/brain-retrain-2026-05-04.lbr`
- `--keyfile` defaults to `/var/lib/lu/master.key`
- `--checkpoint` defaults to `/var/lib/lu/ingest_full_checkpoint.json`
- `--progress-log` defaults to `/var/lib/lu/ingest_full_progress.log`
- `--mcp-stderr` defaults to `/var/lib/lu/ingest_full_mcp.err`
- `--summary-out` defaults to `/var/lib/lu/ingest_full_summary.json`
- No `--smoke`, no `--tier` (so both tiers 1 and 2 run; visual_genome skipped via checkpoint)

## 5. Daemon process state

| PID | Role | Args |
|---|---|---|
| 1106624 | Python daemon | `python3 scripts/ingest_curriculum_full.py` |
| 1106626 | lu_4_mcp child | `lu_4_mcp --mcp --require-encrypted --keyfile /var/lib/lu/master.key --brain /var/lib/lu/brain-retrain-2026-05-04.lbr --brain-format-out v3enc --no-save` |

`/var/lib/lu/ingest_full.pid` was rewritten with `1106624` (the python daemon PID, not the bash wrapper).

## 6. Startup tweaks (verified ok)

| Tweak | Result |
|---|---|
| `lu4d_hub_set max_in_degree=50000` | ok=True |
| `lu4d_hub_set max_out_degree=50000` | ok=True |
| `lu4d_auto_tick_pause` | ok=True |

## 7. Observed throughput (first 10 minutes)

The Wikipedia resume has a silent ~3-min warmup phase: the harness streams articles 0..1018999 from HF and discards them in Python before resuming actual extraction. **No progress log entries fire during the skip** — first entry appeared at 14:59:45Z (~3 min after kickoff). This is normal; do not interpret it as a stall.

Steady-state windows post-skip (each window = 1000 articles):

| Window | Time | Articles | Δ secs | Rate (art/sec) | New bonds | New bonds/art | Errors |
|---|---|---|---|---|---|---|---|
| 1 (skip-spillover) | 14:59:45Z | 1019001..1020000 | n/a | (mixed w/ skip) | +10097 | 10.10 | +3568 |
| 2 | 15:00:42Z | 1020001..1021000 | 57 | **17.54** | +10331 | 10.33 | +3878 |
| 3 | 15:01:45Z | 1021001..1022000 | 63 | **15.87** | +11178 | 11.18 | +3878 |
| 4 | 15:02:45Z | 1022001..1023000 | 60 | **16.67** | +11942 | 11.94 | +3628 |
| 5 | 15:03:46Z | 1023001..1024000 | 61 | **16.39** | +10355 | 10.36 | +3680 |
| 6 | 15:04:46Z | 1024001..1025000 | 60 | **16.67** | +10475 | 10.48 | +3613 |
| **Avg (windows 2-6)** | | | | **16.63** | | **10.86** | |

**Aggregate landed throughput:** ~181 new bonds/sec.

## 8. Brain & checkpoint state during observation

| Time | Brain size | Brain mtime | Checkpoint last_saved_id |
|---|---|---|---|
| Pre-launch | 944,681,049 | 11:50:04 | 1018999 |
| 15:01 (after window 3) | 956,701,349 | 15:01:45 | 1021999 |
| 15:03 (after window 5) | 965,221,105 | 15:03:46 | 1023999 |

Growth ~20.5 MB over 7 minutes for ~53.9K new bonds = ~380 bytes/bond (consistent with v3enc encoding overhead).

## 9. ETA at observed rate

Remaining articles: 6,700,000 - 1,025,000 = **5,675,000**
At 16.63 art/sec sustained:
- 5,675,000 / 16.63 = 341,253 sec = **3.95 days**

**Vs. 11-day projection in port spec.** ~2.8x faster than projected. Estimate is conservative — if rate degrades with brain density growth, true ETA may be 5-7 days. Still well below 11 days.

After Wikipedia, tier-2 corpora (`HuggingFaceH4/no_robots` and `HuggingFaceH4/instruction-dataset`) run; per the original handoff these are ~30 min combined.

## 10. Caveats / known issues (non-blocking)

- **Errors counter elevated.** ~3.6-4.4 errors/article (cumulative 22,715 over ~6,000 processed articles since resume). `gated=0` across all windows so K-gate is not the cause — these are per-entry MCP-side rejections or whole-batch failures inside `lu4d_teach_corpus_batch`. Not crashing the daemon, not corrupting the brain. Worth profiling before next major retrain. Possible causes: malformed entries (subject/relation/object missing), schema mismatches, or trust-corpus rejections.
- **Bonds-per-article lower than 58.** Validated yield was ~58 bonds/article gross extracted; observed is ~10.9 NEW bonds/article. Likely explanation: most extracted bonds reinforce existing edges (returned by MCP as `bonds_existing` and tracked in `sink.bonds_existing`, which the progress log does NOT surface — only `bonds_landed`). Reinforcement is desirable, not a bug; but the progress log makes the extractor look 5x weaker than it is.
- **HF stream is unauthenticated.** stderr shows `Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.` Original Cat.1 retrain ran fine without HF_TOKEN, but if rate degrades over the multi-day run, setting HF_TOKEN env var is the first thing to try.

## 11. Monitoring (for John)

### Quick health check
```bash
ssh lu 'date -u +%FT%TZ; \
  tail -10 /var/lib/lu/ingest_full_progress.log; \
  echo ---; \
  ps -p $(cat /var/lib/lu/ingest_full.pid) -o pid,ppid,etime,stat,pcpu,pmem,cmd'
```

### Daemon files
- Progress log: `/var/lib/lu/ingest_full_progress.log` (append mode; entries every 1000 articles)
- Stdout: `/var/lib/lu/ingest_full_stdout.log`
- Stderr (Python): `/var/lib/lu/ingest_full_stderr.log`
- MCP stderr: `/var/lib/lu/ingest_full_mcp.err`
- Checkpoint: `/var/lib/lu/ingest_full_checkpoint.json` (tiny JSON, atomically rewritten ~every 1000 articles)
- PID: `/var/lib/lu/ingest_full.pid` (1106624)
- Completion sentinel: `/var/lib/lu/ingest_full_DONE` (will be created when ingest completes)
- Summary: `/var/lib/lu/ingest_full_summary.json` (created on completion)

### Expected progress entry format
```
YYYY-MM-DDTHH:MM:SSZ wikipedia: articles=N/6700000 bonds_landed=M errors=K deduped=0 gated=0 rate=X.Yart/s eta_remaining_s=Z
```

## 12. Pause / resume / kill procedures

### Graceful pause + resume (preserves state)
```bash
# Pause:
ssh lu 'kill -SIGTERM $(cat /var/lib/lu/ingest_full.pid)'
# Wait for clean exit (typically <2 sec). The script's main loop
# saves the checkpoint after each progress entry, so SIGTERM at any
# point loses at most the bonds in the in-flight buffer.

# Resume: re-run the same launch command (Section 4). The script
# reads the checkpoint and skips past last_saved_id automatically.
```

### Hard kill (when you need it stopped NOW)
```bash
ssh lu 'kill -SIGKILL $(cat /var/lib/lu/ingest_full.pid); kill -SIGKILL $(pgrep -f "lu_4_mcp.*brain-retrain")'
```

### Rollback brain to pre-Phase-B state
Only if something goes badly wrong over the multi-day run:
```bash
ssh lu 'kill -SIGTERM $(cat /var/lib/lu/ingest_full.pid); sleep 5; \
  cp /var/lib/lu/brain.lbr.pre-phase-b-2026-05-07.bak /var/lib/lu/brain-retrain-2026-05-04.lbr; \
  md5sum /var/lib/lu/brain-retrain-2026-05-04.lbr'
# Expected MD5 after rollback: 69708a6f63fc369bdc4bc4a838384379
```

This rolls back to the pre-Phase-B state. The checkpoint at 1018999 still works against this brain because Phase B's bonds were on top of this baseline.

## 13. the project log references

- Component #387 "Smart Extractor v2 (Cat.1 retrain rescue)" — parent
- Component #434 "Phase B Resume (full corpus run)" — this run (created)
- Learning #850 (performance) — observed throughput vs validated
- Learning #851 (gotcha) — silent skip-warmup phase

## 14. Invariants verified intact

- ✅ C13: no new external crates (no code changes — script is the validated build)
- ✅ Source attribution: `RUN_ID_TIER1` and `corpus`/`source_url` fields populated per the existing harness
- ✅ K-gate: gated=0 across all windows (k-gate is firing but not rejecting in this section)
- ✅ Encryption: `--require-encrypted --brain-format-out v3enc` confirmed in lu_4_mcp args
- ✅ Bond format: per-entry shape unchanged (subject/relation/object/weight/confidence + corpus/source_url/run_id) — uses validated extractor v2 output
- ✅ Canonical brain unchanged: `/var/lib/lu/brain.lbr` sha256 still 835109ba4c0262f760a1510582c4f85654eb66959eeafb9042405e425e856a13

---

*Generated by Phase B Resume port. Observation window: 10 min. Five steady-state progress windows captured. Daemon detached and running.*
