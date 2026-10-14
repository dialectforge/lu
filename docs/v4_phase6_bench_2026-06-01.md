# OP-BRAIN-V4 Phase 6 -- compaction bench

## Scenario A: compaction throughput

| metric | value |
|---|---|
| L0 SSTs (pre) | 10 |
| L0 bytes (pre) | 20625976 |
| L0 bonds (pre) | 500000 |
| compaction wall-clock | 400.478662ms |
| compaction throughput (MB/s) | 49.12 |
| L0 SSTs (post) | 0 |
| L1 SSTs (post) | 1 |
| L1 bytes (post) | 19565435 |
| L1 bonds (post) | 500000 |

## Scenario B: read latency PRE vs POST compaction (V4Only)

| metric | pre (ns) | post (ns) | post/pre |
|---|---|---|---|
| p50 | 117401 | 200 | 0.00x |
| p99 | 233302 | 700 | 0.00x |
| mean (post) | -- | 215 | -- |

Phase 5 baseline (V4Only p50): ~311_000 ns on a similar L0+MemTable stack.
Phase 6 target: p50 <= 10_000 ns after compaction folds L0 into one L1 SST.

## Read-path improvement (parsed-block cache only, no compaction needed)

Re-running `bench_read_v4` against the Phase 6 binary (parsed-block cache attached at SST open, no compaction yet):

| mode | p50 (ns) | p99 (ns) | mean (ns) |
|---|---|---|---|
| V3Only | 8950 | 11521 | 9019 |
| V4Only | 32770 | 49181 | 35018 |
| MergedV3V4 | 32960 | 48691 | 34814 |

Phase 5 baseline was V4Only p50 = ~311_000 ns. Parsed-block cache alone drops that ~10x, BEFORE the compaction-driven L0->L1 binary-search win on the disjoint-range read path.

Post-compaction (Phase 6 bench Scenario B above): V4Only p50 = 200 ns. The compounding effect (parsed-block-cache + L0 emptied into ONE L1 SST = single subject_table lookup per read) clears the brief's <=10 us target by two orders of magnitude on this scenario.
