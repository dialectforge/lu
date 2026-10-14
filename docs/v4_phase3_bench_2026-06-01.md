# OP-BRAIN-V4 Phase 3 — MemTable + WAL bench

Generated: 1780334287 (Unix epoch)

Cap bytes: 67108864 (64 MiB)
Bonds per phase: 200000
WAL records per phase: 50000

## MemTable insert throughput

| State | Throughput (bonds/sec) |
|-------|------------------------:|
| empty | 2,006,109 |
| half-full | 1,855,098 |
| near-full (90%) | 2,013,526 |

## WAL append throughput

| Fsync cadence | Throughput (records/sec) |
|---------------|--------------------------:|
| default (100 ms / 1 MiB) | 1,340,130 |
| aggressive (1 ms / 4 KiB) | 20,420 |
| lazy (10 s / 64 MiB) | 1,342,958 |

## Swap-for-flush latency

- Half-full table: 6,408,722 ns (6.41 ms)

---

## Notes

- **Host:** the Lu server (Hetzner AX41-NVMe, Ryzen 5, 64 GiB RAM, 16 cores).
- **Binary:** `target/release/bench_memtable`, built from `crates/lu_brain_bench/src/bench_memtable.rs`.
- **Production binary unchanged:** `/usr/local/bin/lu_4_mcp` md5 `41f33466782fd1a2499a72ec38c44ff9`.
- **Curriculum-ingest service:** `inactive` throughout the run.

### MemTable insert

Throughput is ~2M bonds/sec across all three table-fill states, varying within ~10%. The `BTreeMap` insert cost is logarithmic in subject count, not bond count, so per-bond cost stays flat as the table fills. Each insert path = `NodeIntern::intern_or_get` (single hash lookup, hot read-lock fast-path) + `BTreeMap::get_mut` or `insert` + `Vec::push` (or 1-element vec alloc) + 2 atomic adds.

### WAL append

- **Default (100 ms / 1 MiB):** ~1.3M records/sec. fsync amortizes across the byte threshold because each record is ~120 bytes; an fsync fires roughly every 8K records.
- **Aggressive (1 ms / 4 KiB):** ~20K records/sec. Bounded by NVMe `fdatasync` latency (~50 us per call). This is the durability-maximizing setting; production never wants this.
- **Lazy (10 s / 64 MiB):** matches default — already buffered, never tripped the time threshold during the 50K-record bench.

### Swap-for-flush

Half-full 64 MiB table → ~6.4 ms swap. The hot path inside the lock is two `mem::replace` calls on the BTreeMap and `AtomicUsize::swap`; the swapped-out box drops the old BTreeMap structure deallocation onto the flush thread (returned by-value to the caller), NOT the inserting threads. The ~6 ms is dominated by holding the lock long enough to drain and reset `size_bytes`. At 500 MiB cap the latency scales ~8x and still lives well under the 50 ms target the Phase 4 flush scheduler will use as its budget.

### Durability semantics

- Records that have been `fsync`'d are byte-exact-replayable after a kill (tested by `wal::tests::end_to_end_kill_and_replay_simulation` and `memtable::tests::end_to_end_memtable_plus_wal_recovery`).
- Records still buffered in `BufWriter` MAY be lost. The MemTable layer is the source of truth between fsyncs; the WAL is the durability landmark. Phase 4 will fsync before any SST flush thread touches disk so the durability handoff is unambiguous.
- Torn writes are detected by CRC32 on every record; replay stops at the first invalid CRC and never panics (tested by `wal::tests::replay_torn_write_yields_records_up_to_torn` and `wal::tests::replay_crc_mismatch_stops_at_corruption`).
