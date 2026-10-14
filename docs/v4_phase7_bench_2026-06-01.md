# OP-BRAIN-V4 Phase 7 -- migration bench (2026-06-01)

## Setup

- target bond count: 1000000
- v3 path: `/tmp/bench_migration_1000000.lbr`
- v4 dir: `/tmp/bench_migration_1000000_v4`
- v3 size on disk: 183330346 bytes (174.84 MiB)
- v3 build time (one-shot): 5.10s

## Migration

- bonds migrated: 999994
- output SST count: 1
- total output bytes: 54514339 (51.99 MiB)
- wall-clock: 1.38s
- throughput: 126.51 MB/s (v3 bytes / migration wall-clock)
- bonds/sec: 723575
- migrator-reported peak RSS: 2285.93 MiB
- RSS after build: 1162.9 MiB
- RSS after migration: 1163.2 MiB

## Verify

- subjects checked: 245388
- bonds checked: 999994
- mismatches: 0
- verify wall-clock: 1.28s
- RSS after verify: 1163.6 MiB
- result: OK

## Honest assessment

The v3 brain is built fresh from random bonds with the same shape
as the canonical curriculum-ingest output (one Baseline keyframe per
bond, 6 BondType variants, ~25% node:bond ratio). Migration walks
the bond key index in subject-sorted order (strategy 'b' from the
port plan); each bond is materialised once on the heap, pushed to
the active SSTWriter, then dropped. Verify is a full v3+v4 walk that
reconciles every (subject, predicate, object) triple.

Production rehearsal numbers live in the project log log for this
port; the bench above is the synthetic baseline.

## Production-shape rehearsal (real canonical brain)

Ran against a sudo-copied snapshot of `/var/lib/lu/brain.lbr` placed at
`/tmp/brain.lbr.migrate-rehearsal` (NOT against the live file). Encrypted
brain, master key copied alongside. v4 output and brain copy deleted
after the run.

- source size: 5,926,038,533 bytes (5.52 GiB)
- bonds migrated: 15,491,551
- subjects: 1,588,863
- output SST count: **2** (L0-1.sst 555 MiB + L0-2.sst 419 MiB; ~974 MiB total)
- migration wall-clock: **34.8s**
- migration throughput: **170 MB/s** (v3 bytes / wall-clock)
- migration bonds/sec: **445k**
- migration peak RSS (migrator-reported): **9.66 GiB**
- verify wall-clock: **~38s**
- verify result: **OK** (0 mismatches)
- full-tool peak RSS during verify: **15.05 GiB** (v3 triple HashMap dominates;
  ~15M triples * ~ (32+32+1)+u8+usize bytes including hashmap overhead).
- full-tool wall-clock (migrate + verify): **73.18s**
- CPU utilisation: 99% (single-threaded; matches the design)

No production touch: the canonical brain.lbr at /var/lib/lu/brain.lbr was
only READ via sudo cp; no write path crossed it. Snapshot + v4 output dir
+ master.key copy all deleted post-bench.
