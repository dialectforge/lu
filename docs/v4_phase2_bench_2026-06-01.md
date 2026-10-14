# OP-BRAIN-V4 Phase 2 — `bench_sst_read` 2026-06-01

SST read-path bench. Synthetic SST built in `/tmp/`, then queried.

## Test corpus

| Field | Value |
|---|---|
| Bonds                  | 10000000 |
| Subject pool           | 1250000 |
| SST size on disk       | 423.7 MiB |
| Block count            | 2021 |
| Write wall-clock       | 11.17 s |
| Open + index load      | 0.08 ms |

## Lookup latency

`SST::get_bonds_for_subject` measured over 5000 queries against a 1250000-entry subject pool.

| Scenario | p50 | p99 |
|---|---|---|
| Cold cache | 222912 ns (222.91 µs) | 333602 ns (333.60 µs) |
| Warm cache | 60411 ns (60.41 µs) | 223731 ns (223.73 µs) |

## BlockCache hit rate

| Counter | Value |
|---|---|
| Hits (measurement window)   | 2620 |
| Misses (measurement window) | 2380 |
| Hit rate                    | 52.4% |

## Cost breakdown

| Op | p50 | p99 |
|---|---|---|
| Single block decompress (cache miss) | 167431 ns | 180572 ns |

The mmap-vs-decompress ratio (warm vs cold p50) is 3.7× — warm reads avoid the zstd decompress cost.

## Iteration throughput

| Op | Value |
|---|---|
| `iter_bonds` walked         | 10000000 bonds in 0.90 s |
| Throughput                  | 472.5 MB/s |

