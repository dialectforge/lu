# OP-BRAIN-V4 Phase 8 -- v3 vs v4 boot bench (2026-06-01)

Synthetic-bond cold-boot comparison. Numbers below are the
reproducible local baseline; canonical-brain cold-boot numbers
will be captured during operator-triggered production cutover
per `docs/op_brain_v4_cutover_runbook.md`.

## Setup

- target bond count: 200000
- bonds actually emitted (post-dedup): 199996
- v3 fixture: `/tmp/bench_v4_boot_200000.lbr` (36603498 bytes, 34.91 MiB)
- v4 directory: `/tmp/bench_v4_boot_200000_v4` (10823425 bytes across 1 SST(s), 10.32 MiB)
- fixture build wall-clock: 0.85s
- migration wall-clock: 0.24s

## v3 cold boot (`Brain::load_or_new_with_mode_and_key`)

- wall-clock: 0.120s (119 ms)
- nodes resident: 50000
- bonds resident: 199996
- RSS before boot: 291.2 MiB
- RSS after boot: 320.0 MiB
- RSS delta (load cost): 28.8 MiB
- VmHWM peak through v3 boot: 432.2 MiB

## v4 cold boot (`Brain::load_v4`)

- wall-clock: 0.003s (2 ms)
- SSTs registered in manifest: 1
- RSS before boot: 291.3 MiB
- RSS after boot: 291.3 MiB
- RSS delta (load cost): 0.0 MiB
- VmHWM peak through v4 boot (cumulative incl. v3 boot above): 432.2 MiB

## Ratios

- v3 / v4 boot-time ratio: **41.68x** (v3 is 41.68x v4)
- v3 / v4 load-RSS ratio: **v4 load delta below RSS measurement granularity (effectively 0 MiB)**; v3 paid 28.8 MiB to load this fixture.

## Honest assessment

v3 boot pays for full deserialization + adjacency rebuild on every
start: O(bonds) wall-clock and O(bonds) RAM both at peak.

v4 boot is structurally cheap: MANIFEST validation, WAL replay (empty
on a freshly-migrated brain), MemTable allocation, flush + compaction
worker spin-up. No SST is read at boot -- they are demand-paged the
first time `read_outgoing_bonds` lands on their subject range.

The win compounds with scale: doubling the bond count roughly doubles
v3 boot time and v3 RAM-at-boot, but leaves v4 boot time and v4
RAM-at-boot essentially flat. The canonical 15.5M-bond cold boot
(operator-captured during cutover) will be the real proof point.
