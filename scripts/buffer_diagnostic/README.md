# scripts/buffer_diagnostic/

Read-only sampler + analyzers for the `lu4d-curriculum-ingest` daemon's
per-save memory pattern. Distinguishes Pattern A (legitimate brain growth
+ save-transient spikes) from Pattern B (runaway training-time buffer
that flushes only at save).

This is **read-only**. It does not stop, restart, or otherwise touch the
running daemon. Safe to run in production at any time.

## Files

- `sampler.sh` — 30s-interval `/proc/<lu_4_mcp_PID>/status` + cgroup
  sampler. Captures VmRSS / RssAnon / RssFile / VmHWM, cgroup
  current+peak+events, brain.lbr size+mtime+sha-prefix, brain.lbr.kw
  size+mtime, brain.lbr.tmp size+mtime (so save events are visible),
  WAL file sizes, bond+article counts parsed from the daemon's
  recent journal. Appends one JSON line per sample to
  `/tmp/buffer_diagnostic.jsonl`.
- `analyze_raw.py` — one line per sample, GiB-formatted, for eyeballing
  the trajectory.
- `analyze_save_events.py` — finds save events (brain.lbr SHA changes),
  computes pre/post RSS delta and training-period drift.

## How to invoke

```bash
# Start sampler in background:
rm -f /tmp/buffer_diagnostic.jsonl
nohup setsid bash scripts/buffer_diagnostic/sampler.sh \
  > /tmp/buffer_diag_stdout.log 2>&1 < /dev/null &
echo $! > /tmp/buffer_diag_sampler.pid

# Let it run for at least ~80 min to capture 3+ save events.
# Cadence in production: ~26 min/save (wikipedia stage 2026-05-21 era;
# may differ once wikipedia ingest completes).

# Stop sampler:
kill $(cat /tmp/buffer_diag_sampler.pid)

# Analyze:
python3 scripts/buffer_diagnostic/analyze_raw.py     # raw trajectory
python3 scripts/buffer_diagnostic/analyze_save_events.py  # per-save deltas
```

## How to interpret

For each save event the analyzer reports two key numbers:

1. **Pre→post save RSS delta.** Pattern A: drop ≈ 0 to −0.5 GiB
   (just kernel evicting `brain.lbr.tmp` mmap pages after rename).
   Pattern B: drop > 1 GiB sustained (a real heap buffer was freed).

2. **Training-period drift** (between two consecutive saves). Pattern A:
   RssAnon **flat or slightly negative** (glibc trimming idle arenas).
   Pattern B: RssAnon **climbs monotonically** with bond count.

Cross-check with cgroup `memory.peak`: it should only rise during save
events. Any cgroup peak growth *between* saves indicates buffer
accumulation.

## Example output (run captured 2026-05-21 on Hetzner, OLD binary)

Raw JSONL not committed (debug output specific to one run). Original
sampler data was at `/tmp/buffer_diagnostic.jsonl` on the Hetzner box
(`203.0.113.10`); see `outputs/op022_buffer_diagnostic.md` in the
working-tree for the full report.

The 2026-05-21 run captured 47 samples across 26 min, including one
complete save cycle and the start of a second. Result:

- **Verdict: Pattern A (brain growth, no runaway buffer).**
- Post-save RssAnon dropped −2.61 GiB — attributable to the save's
  own transient side-table bincode buffers (see the 5-component
  decomposition in `scripts/op022_save_spike_diag.rs` doc-comment).
- Training-period RssAnon DECREASED 17.40 → 16.61 GiB over 9.5 min
  while +21,989 bonds were landed → glibc arena trim, not a buffer.
- cgroup `memory.peak` constant at 25.27 GiB through 13 min of
  training between saves; rose only when save 2 started.
- WAL files all 0 bytes throughout — no sidecar growth signal.

## When to re-run

- After any save-path change in `src/brain_io_v3.rs` or
  `src/network.rs::compact_keyword_sidecar` — verify per-save delta
  hasn't regressed.
- When investigating an unexplained cgroup peak rise during steady-state
  training (i.e., between saves with no stage transition).
- Periodically if running on the OP-022 binary stack to spot future
  regressions in the training-time floor.

## Locked rule

This sampler is the ONLY canonical way to attribute heap to "buffer
vs brain growth." Don't speculate from cgroup numbers alone — run the
sampler, look at the per-save delta + training drift, then decide.

See also: `scripts/op022_save_spike_diag.rs` (daemon-stopped harness
for save-internal composition).
