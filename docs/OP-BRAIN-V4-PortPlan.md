# OP-BRAIN-V4: The Memory-Bounded Brain

**Author:** John Martin + Claude
**Date:** 2026-06-01
**Status:** Plan — pre-implementation
**Project:** Lu Ai internal notes
**Target hardware:** Hetzner AX41-NVMe (Ryzen 5, 64 GiB RAM, 16 cores, 2× 512 GB Toshiba KXG60ZNV512G NVMe)

---

## Revision history

- **2026-06-01 (Phase 1 spec revision):** First implementation attempt halted at design-vs-reality reconciliation internal notes. The original Phase 1 spec assumed an in-memory string-keyed bond model that does not exist in the codebase. Lu's actual model uses 32-byte `Seed` keys, `TemporalBond` with rich `Vec<TemporalKeyframe>` history, and scar-aware weight reads via `weight_at(t, &ScarLog) -> ScarAwareReading`. The 13-byte `CompactBond` representation described in the Architecture / Data Model sections is **on-disk SST block format** (Phase 2/4), NOT an in-memory replacement. Phase 1 below has been re-scoped to **Seed → NodeId(u32) interning + adjacency `Vec<Seed>` → `Vec<NodeId>` shrink + keyframe-pool dedup** — preserves temporal/scar/scope architecture intact, ships 5–8× in-memory shrink (vs original 10× estimate). The other phases are unchanged.
- **2026-06-01 (Phase 2.1 amendment):** Phase 3 (MemTable + WAL, internal notes surfaced that the originally-shipped single-keyframe-per-bond SST format would silently drop synthesis / session / decay-stamped keyframe history at flush time. Per `feedback_curiosity_load_bearing`, that history is load-bearing for Lu's curiosity engine. The CompactBond on-disk layout is amended from 13 to 14 bytes: `u16 keyframe_idx` → `u16 keyframe_offset + u8 keyframe_count`, addressing a contiguous slice in the per-block keyframe pool. SST format version bumped 1 → 2; v1 SSTs (Phase 2 test fixtures only) are refused with a clear error. Cross-bond keyframe-pool dedup is deliberately deferred (writer simplicity > marginal compression win). Phase 4 (flush wiring) inherits the multi-keyframe pipeline unchanged.

## Executive summary

Today Lu's brain is fully heap-resident. ~17 GiB at 9.5M bonds; extrapolated ~600 GiB at the 6.7M-Wikipedia-article target. We have 64 GiB. A 10× gap brute force can't close.

**OP-BRAIN-V4 closes that gap by re-architecting the brain as a memory-mapped immutable archive plus a small in-RAM write buffer**, borrowing the working pattern from llama.cpp (`mmap` + page-cache for fixed-layout weights) and RocksDB / LMDB (LSM-tree for writes-on-disk). The Lu-original element is **subject-clustered block layout** so spread activation has the locality LLMs get from layer-by-layer access, plus **curiosity-driven prefetch** that uses Lu's wonder queue as a hot-set predictor.

After v4, resident RAM scales with working set (estimated ~5–10 GiB) instead of total brain size (estimated 100s of GiB on disk). Tier 1 fits trivially. Tier 2 + retrains fit. The architecture stops being the bottleneck.

> **Phase 2.1 on-disk estimate.** With the 14-byte CompactBond and per-bond keyframe ranges, expected on-disk-after-zstd budget is **~16–22 bytes/bond** (was 15–20 in the original 13-byte spec) for typical multi-keyframe corpora. Empirically Phase 2.1's `bench_sst_read` on a 10M-bond synthetic SST measured 424 MiB (~44 bytes/bond) at zstd level 3 with mostly random keys — random-key entropy floors the compressor; real corpora with subject locality compress harder. Block count grew 1.9% (2021 → 2059) and `iter_bonds` throughput rose to 513 MB/s (vs 472 MB/s on Phase 2).

This document specifies the architecture, the migration plan, and an 8-phase implementation sequence. Each phase has files affected, success criteria, tests, and estimated LOC. Total scope: ~6,000–9,000 LOC of careful Rust over an estimated 3–5 weeks of focused work. Production risk is managed via incremental compatibility shims so v3 brains continue to work throughout.

---

## Problem statement

### Current state (verified 2026-06-01)

- Brain in-memory: heap-resident `HashMap<NodeId, Vec<Bond>>` structures (`src/network.rs` lines ~1300–1700).
- Brain on disk: `brain.lbr` is a v3 compressed encrypted archive. Reads go through `BlockCache` (`src/compress.rs:221` and `:1510`). Writes do not.
- Per-bond RAM cost: ~1.7 KB (heap-allocated structs, HashMap entry overhead, provenance strings, keyframe vectors).
- Save path: full-snapshot rewrite of `brain.lbr.tmp` on every save, then atomic rename. Throughput bottleneck previously identified as zstd compression (39% per OP-022.s.v2 flamegraph); CRC32 verified non-bottleneck (OP-PARCRC reversion, internal notes.
- Memory protections in place: `MALLOC_ARENA_MAX=2`, `MALLOC_TRIM_THRESHOLD_=131072`, `LU_BLOCK_CACHE_BYTES=2G` (wired 2026-06-01 after audit found the env var was added in OP-TRAIN-CACHE-CAP commit `16240fe` but never plumbed through systemd).
- Hardware: 2× 512 GB NVMe currently in RAID 1, 398 GB free on `/`, 64 GiB RAM.

### Scaling math

- Linear extrapolation at current 1.7 KB/bond:
  - Tier 1 complete (6.7M articles × 50 bonds/article = ~335M bonds) → **~600 GiB resident**
  - Tier 1 + 2 + Visual Genome → **~1 TiB+ resident**
- Even with extreme quantization (target 30 bytes/bond → 60× reduction):
  - Tier 1 → ~10 GiB resident
  - Tier 1 + Tier 2 → ~30 GiB resident
  - Still doesn't scale past tier 2 cleanly, AND requires touching every line of bond storage code in one shot.

### What LLMs taught us

llama.cpp runs a 70B Q4-quantized model — ~40 GB on disk — using **~2–4 GB resident**. The technique is `mmap` + kernel page cache. Weights are immutable, fixed-layout, addressed by offset. Forward pass loads a layer's weights, computes, moves to the next layer. Cold layers' pages get evicted by the kernel. No application code participates in eviction — it's "free" via the page cache.

The core insight is layout + immutability + access locality. If your data has those three, the kernel does paging for you.

### The bridge to Lu

Lu's compressed `brain.lbr` archive **already has fixed layout** (header + block offsets + sorted bonds), and the read path already uses block-level access with an LRU. The problem is **only the write path** — every new bond goes into a parallel heap structure that holds everything.

If we adopt the LSM-tree pattern (write to a small MemTable in RAM, periodically flush to disk, periodic background compaction), the heap structure shrinks to a bounded write buffer. The archive on disk grows arbitrarily; RAM holds only the recently-written buffer plus the BlockCache.

That's the architecture.

---

## Architecture

### Three storage tiers

```
┌─────────────────────────────────────────────────────────────┐
│                         RAM (~5–10 GiB)                     │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │  MemTable    │  │  BlockCache  │  │  Hot indexes     │  │
│  │  (500 MB)    │  │  (2 GiB)     │  │  (subject→block) │  │
│  │  recent      │  │  decompressed│  │  (kw, type, ...) │  │
│  │  writes      │  │  blocks      │  │                  │  │
│  └──────┬───────┘  └──────▲───────┘  └──────────────────┘  │
└─────────┼─────────────────┼─────────────────────────────────┘
          │ flush            │ page-in via mmap fault
          │ (500 MB hits)    │ (kernel page cache)
          ▼                  │
┌─────────────────────────────────────────────────────────────┐
│                   NVMe (mmap, ~100s of GiB)                 │
│                                                             │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐  ┌─────────────────┐  │
│  │ Level 0 │ │ Level 1 │ │ Level 2 │  │   WAL           │  │
│  │ recent  │ │ compacted│ │ major   │  │  (append-only,  │  │
│  │ flushes │ │ subj-srt │ │ subj-srt│  │   durability    │  │
│  │ small   │ │ medium   │ │ large   │  │   for MemTable) │  │
│  └─────────┘ └─────────┘ └─────────┘  └─────────────────┘  │
│       ▲                                                     │
│       │ background compaction merges Level N → Level N+1    │
└─────────────────────────────────────────────────────────────┘
```

**MemTable.** Bounded in-memory write buffer (default 500 MB). New bonds land here. Reads check MemTable first. When full, atomically swapped with a fresh empty MemTable and the old one flushed to a Level-0 SSTable in the background. Backed by a write-ahead log (WAL) for durability — if the daemon crashes before flush, replay the WAL on next start.

**SSTable (Sorted String Table, by Lu convention "SST").** A single immutable subject-sorted block file on disk. Each SST contains: a header, a per-block index `(subject_min, subject_max, offset, length)`, and N zstd-compressed blocks. Subject sorting means that for any subject node X, all of X's bonds live in at most one block (usually exactly one). A spread activation step from X is one block fetch.

**Level structure (RocksDB-style leveled compaction).**
- **Level 0:** small recent SSTs (the result of MemTable flushes). Up to ~10 SSTs allowed. Reads must check all L0 SSTs (no global subject order across them).
- **Level N (N ≥ 1):** larger SSTs covering disjoint subject ranges. Each level is ~10× the size of the previous. Within a level, SSTs have disjoint subject ranges, so a read for subject X picks exactly one SST per level via binary search.
- Total levels at full scale: ~3–4 levels for the 6.7M-article target.

**Compaction.**
- **Minor compaction:** L0 → L1 when L0 exceeds the size threshold. Merge L0 SSTs and overlapping L1 SSTs into new L1 SSTs.
- **Major compaction:** Periodically merge L1 + L2 → fresh L2. Removes tombstones (deleted bonds), re-sorts by subject, applies confidence-tiered eviction.
- Compaction runs in a background thread pool. Doesn't block reads or writes.

**BlockCache.** Decompressed-block LRU, capped at `LU_BLOCK_CACHE_BYTES` (we have this knob now). Caches across all SSTs.

**Page cache (kernel-managed).** The SST files are `mmap`'d. The kernel automatically caches recently-read pages and evicts cold ones. This is free RAM savings — no application code needed.

### Read path

`Brain::get_outgoing_bonds(subject) -> Vec<Bond>`:

1. **Check MemTable:** any bonds for this subject in the in-memory write buffer? Collect them.
2. **Check L0 (up to ~10 SSTs):** for each L0 SST, look up subject in its block index. If subject in range, decompress block (via BlockCache), pull matching bonds.
3. **Check L1+L2+...:** for each level, binary-search by subject range to find the single SST that could contain this subject (because levels have disjoint subject ranges within them). Look up block, decompress, pull bonds.
4. **Merge results:** dedup by `(subject, predicate, object)`, take latest version per bond. Return.

For typical reads (single subject), this is `1 + L0_count + L_total ≈ 12–15` block touches at most. With BlockCache hits, usually 1–3 actual decompressions.

### Write path

`Brain::add_bond(subject, predicate, object, weight) -> Result`:

1. **Append to WAL** (durability barrier). WAL is an append-only file; sync on a configurable cadence (e.g. every 100 ms or every 1 MB).
2. **Insert into MemTable**. MemTable is an in-memory `BTreeMap<NodeId, Vec<CompactBond>>` (subject-keyed, sorted). Insert is O(log N).
3. **Check size:** if MemTable exceeds threshold (500 MB), atomically swap with empty new MemTable. Background thread takes the old MemTable, sorts by subject, writes to L0 SST file, fsyncs, updates the manifest. WAL segment for the flushed MemTable is truncated.
4. **Return success.**

### Compact bond representation

Per-bond on-disk format (in SST blocks). **Phase 2.1 layout (current):**

```
struct CompactBond {
    subject_id:      u32,    // 4 bytes, 4 billion node ceiling
    object_id:       u32,    // 4 bytes
    predicate:       u8,     // 5 bits BondType + 1 bit direction + 2 bits flags
    weight:          u8,     // 256 levels, monotonic w.r.t. current f32 weight
    keyframe_offset: u16,    // 2 bytes, start index in per-block keyframe pool
    keyframe_count:  u8,     // 1 byte, number of consecutive keyframes (0..=255)
    confidence:      u8,     // 8 bits, K-gate status + tier
} // = 14 bytes per bond
```

A bond's keyframe history is `pool[keyframe_offset .. keyframe_offset + keyframe_count]`. `keyframe_count == 0` is legal (no keyframes) but unusual; every well-formed bond should carry at least one.

Per-block keyframe pool: Phase 2 originally specified cross-bond dedup at the block level. **Phase 2.1 deliberately drops that dedup**: each `push_bond` appends its slice as a fresh contiguous run. Rationale: the dedup HashMap conflicted with the new range semantics, and zstd compression amortises the redundancy. A future phase may reintroduce dedup if profiles warrant it.

Target: **~16–22 bytes per bond on disk after zstd compression** (was ~15–20 in the original 13-byte spec; 14-byte raw + lower dedup ratio on the pool pushes the estimate up slightly), **~32–55 bytes per bond decompressed in BlockCache**, vs 1.7 KB heap currently. Still ~30–100× reduction in the resident representation.

### Subject-clustered layout (Lu-original element)

Within an SST, bonds are sorted by `subject_id`. A block ends at a subject boundary (never splits a subject's bonds across two blocks). This means:

- A spread activation step from subject X hits exactly one block per level.
- For a typical query (spread from 1–3 seeds, 2–3 hops), total block touches ≈ `seeds × hops × levels ≈ 1–3 × 2–3 × 3–4 = 6–36`. With BlockCache, often 2–10 actual decompressions.
- Predicate-typed queries ("what is X an IsA of?") can use a within-block predicate-secondary-index to binary-search the IsA bonds within X's block.

### Curiosity-driven prefetch (Lu-original element)

Lu's curiosity engine produces a wonder queue: nodes Lu wants to research next (`HubBoundary`, `DeadEnd`, etc.). These are *predicted hot nodes*. We can prefetch their blocks into BlockCache in the background:

- `lu4d_idle_tick` already runs at idle. Add a step: for each wonder in the queue, prefetch the corresponding block. The block lands in BlockCache before the actual spread activation needs it.
- For recent activation trails (Q.3 session memory), the next few likely subjects can be prefetched after a spread step completes.

This is the structural analog of LLM speculative decoding's KV-cache prefetch — using the model's own state to predict what's needed next.

### 2-NVMe hardware story

Current state: RAID 1 (mirrored) across the two drives. ~436 GiB usable, redundancy on disk failure.

Three viable configurations:

1. **Keep RAID 1, leave it alone.** Simplest. v4 just uses the existing filesystem. We get reliability; we get half the raw bandwidth. Recommended for initial v4 ship — minimize variables.

2. **Convert to RAID 0.** ~872 GiB usable, 2× sequential bandwidth, no disk-failure redundancy. Useful if compaction becomes I/O-bound. **Reversible:** if we have a verified brain backup elsewhere, drive failure means restore from backup, not data loss. Recommended after v4 ships and we measure compaction throughput.

3. **Split tiering: each drive serves its own data.** Drive 1: hot tier (MemTable WAL, recent L0 SSTs, BlockCache backing if we ever go beyond RAM). Drive 2: cold tier (older L2+ SSTs, archive layers). Background compaction is drive-1-to-drive-2 mostly, freeing drive 1 for active writes. Most architecturally clean. **Highest complexity.** Recommended for v4.1 or v4.2 once we know we need the parallel bandwidth.

**Phase 0 recommendation:** keep RAID 1. Phase 7 (deploy v4) re-evaluates. Phase 8 (post-deploy optimization) decides on RAID 0 vs split tiering based on measured compaction bandwidth.

---

## Data model

### NodeId

```rust
pub type NodeId = u32;  // 4 billion node ceiling
```

Current code uses string-keyed node lookup in several hot paths. Convert to `u32` IDs via a node-string-interning table (which stays in RAM — node count plateaus per `reference_lu_healthy_growth_curve`).

### BondType

```rust
#[repr(u8)]
pub enum BondType {
    RelatedTo = 0,
    IsA = 1,
    HasAspect = 2,
    PartOf = 3,
    // ... 28 variants total, must fit in 5 bits with room for flags
}
```

### CompactBond (on-disk)

14 bytes per bond as of Phase 2.1 (was 13 in Phase 2; see Revision history and Architecture section). Layout: `u32 subject_id + u32 object_id + u8 predicate + u8 weight + u16 keyframe_offset + u8 keyframe_count + u8 confidence`. SST format version is 2 and the reader refuses v1 with a clear error.

### Block

```
zstd-compressed payload of:
  - block_header (subject_min, subject_max, bond_count, keyframe_pool_size)
  - keyframe_pool: small dedup table of provenance+timestamp records
  - bonds: array of CompactBond, sorted by (subject_id, predicate, object_id)
```

Block size target: 256 KiB compressed (matches current v3). Decompressed: 1–2 MiB depending on compression ratio.

### SSTable

```
SST file header (magic, version, level, subject_range, bond_count, encryption_iv)
Block index: array of (subject_min_in_block, offset, length, decompressed_length)
Blocks: contiguous zstd-compressed encrypted payloads
SST footer (CRC32 of index, signature)
```

Per-SST encryption uses the existing master.key + per-SST IV scheme from v3.

### Manifest

`/var/lib/lu/brain/MANIFEST` — atomic-updated metadata file:

```
schema_version: 4
master_key_id: <fingerprint>
levels: [
  { level: 0, ssts: [ { id, file, subject_range, size, mtime } ... ] },
  { level: 1, ssts: [ ... ] },
  ...
]
memtable_wal: <current WAL path>
last_compaction: <timestamp>
```

Updated transactionally on flush, compaction, or recovery. Old manifests retained briefly for rollback.

### MemTable + WAL

In-memory:

```rust
struct MemTable {
    entries: BTreeMap<NodeId, Vec<MemBond>>,
    size_bytes: usize,
}

struct MemBond {
    object: NodeId,
    predicate: BondType,
    weight: u8,
    keyframe: Option<KeyframeRef>,
    seq: u64,         // monotonic, for tombstone ordering
}
```

WAL is append-only on disk:

```
WAL record:
  - seq: u64
  - subject_id: u32
  - bond_payload: CompactBond (14 bytes — Phase 2.1)
  - keyframe_payload (variable; the bond carries its full keyframe history)
```

On daemon start, replay WAL into MemTable. On MemTable flush, truncate WAL up to the flushed-seq.

---

## Phases

Eight phases, each landing a coherent slice that's testable and deployable independently. v3 compatibility is preserved through Phase 5; v4 becomes the canonical format at Phase 6.

### Phase 0 — Spec + Scaffolding + Bench Harness (2–3 days)

**Goal:** lock the design, build the test bed that measures v3 vs v4 at every milestone.

**Files:**
- `docs/glossary.md` — append v4 named parts (MemTable, SSTable, Manifest, Compaction, etc.)
- `docs/OP-BRAIN-V4-PortPlan.md` — this document, committed to repo as the design source of truth
- `crates/lu_brain_bench/` — new test crate; benchmarks resident RAM, throughput, latency for v3 baseline
- `tests/v4_invariants.rs` — empty placeholder, populated phase by phase

**Success criteria:**
- v3 baseline benchmarks recorded for: bond add throughput, single-subject read latency, spread activation 2-hop latency, total resident RAM at 10K/100K/1M bond counts.
- These numbers become the regression bar for v4. No phase ships if v4 worse than v3 on any of them by more than 10%.

**Tests:** benchmarks themselves are the tests.

**Risk:** LOW (no behavior change).

**LOC:** ~600 (mostly benchmark infrastructure).

### Phase 1 (revised 2026-06-01) — Seed interning + adjacency shrink + keyframe pool (5–7 days)

**Goal:** the real Phase 1 in-memory shrink. Preserves the temporal/scar/scope bond architecture intact. The 14-byte CompactBond representation (Phase 2.1 amended; was 13 in Phase 2) is correctly scoped to Phase 2/4 SST on-disk format only.

**Files:**
- `src/node_intern.rs` — new module. `Seed → NodeId(u32)` interning with reverse lookup. Thread-safe (`Arc<RwLock<NodeInternState>>` first; only shard to `dashmap` if benchmarks show contention). u32 ceiling = 4B nodes (plenty).
- `src/network.rs` adjacency conversion. `HashMap<Seed, Vec<Seed>>` for `outgoing`/`incoming` becomes `HashMap<Seed, Vec<NodeId>>`. Bond key map `HashMap<(Seed, Seed), TemporalBond>` is UNCHANGED in Phase 1 — bond storage shape stays Seed-keyed; only adjacency lists shrink.
- `src/bond.rs` — keyframe-pool dedup. `KeyframeSource::Baseline` is identical across the corpus (no payload). Intern into a small `Arc<TemporalKeyframe>` pool. `BondView::keyframes_iter()` already abstracts the borrow so callers do not change.
- `src/brain_io_v3.rs` — load path interns Seeds into NodeIds during adjacency rebuild; save path materializes NodeId → Seed via reverse lookup. v3 on-disk format UNCHANGED.

**What is NOT changed in Phase 1:**
- `TemporalBond` struct shape (keyframes, half-life, bond_type).
- Keyframe semantics (time-travel, scope, scar overlay).
- `BondView` API surface (already the fat-view).
- `weight_at()` ScarAwareReading API.
- Bond key map `HashMap<(Seed, Seed), TemporalBond>`.
- v3 archive on-disk format.

**Success criteria:**
- Adjacency-list memory at 1M bonds drops to ≤ 35% of v3 baseline (5–8× shrink on the adjacency component, which is the dominant component touched on every spread step).
- Total RSS at 1M bonds: target ≤ 800 MiB (vs 2.85 GiB v3 baseline — ≥3.5× overall shrink; the bond key map + keyframes remain so we will not hit 50 MiB).
- All 1630+ unit/integration tests still pass.
- Phase 0 benchmarks: bond add throughput ≥ v3, single-subject read p50 ≤ v3 + 30% (some intern indirection is acceptable; Phase 6 compaction will recover the gap structurally).

**Tests:**
- Round-trip: Seed → intern → NodeId → reverse-lookup → Seed equal.
- Concurrent intern: many threads interning the same Seed simultaneously — all get the same NodeId, no duplicates.
- Keyframe-pool dedup: 1000 identical Baseline keyframes share a single `Arc` allocation; reference count = 1000.
- Stress: 10M bonds across 1M nodes — verify spread activation, bond key lookup, adjacency reads all match v3 byte-for-byte.
- v3 archive round-trip: load a v3 brain, run new Phase 1 code, save, reload, verify bond-set equality with the original.

**Risk:** MEDIUM. Touches `src/network.rs` (graph core) and `src/bond.rs` (keyframe lifecycle). The intern-table lifetime model needs careful design — NodeIds must be stable across saves, and reverse-lookup must survive a load cycle. Lower risk than original spec because `BondView` already exists and we do NOT touch keyframe semantics.

**LOC:** ~1,200 (less than original estimate because we are not rebuilding the bond struct, only the adjacency and keyframe-pool layers).

### Phase 2 — SSTable read path (4–6 days)

**Goal:** build the SST file format. Implement read-only path: given an SST file, look up bonds by subject. No writes yet.

**Files:**
- `src/sst.rs` — new module with `SST` struct, block index, block reader (decompress + decrypt + slice).
- `src/sst_writer.rs` — new module to write SST files from sorted bond stream. (Used by Phase 4 for compaction.)
- `src/brain.rs` — add `Brain::read_sst(path) -> SST` (testing entry point).

**Success criteria:**
- Write a synthetic 100 MiB SST (10M bonds), read it back, verify subject lookups return correct bonds.
- Block index binary search confirmed O(log N) (not O(N) walk).
- BlockCache hit rate ≥ 80% on locality-friendly access patterns.

**Tests:**
- Write SST, read SST, equality.
- Random subject lookups: correctness + latency budget.
- Encrypted vs plaintext mode parity.
- Concurrent reads from multiple threads (read path must be lock-free or read-mutex-only).

**Risk:** LOW. Self-contained new module. Doesn't touch existing read path.

**LOC:** ~1,200.

### Phase 3 — MemTable + WAL (4–6 days)

**Goal:** bounded in-memory write buffer with WAL durability. Atomic swap-and-flush primitive. No SST flush yet (Phase 4).

**Files:**
- `src/memtable.rs` — new module with `MemTable` struct, atomic insert, snapshot-and-swap primitive
- `src/wal.rs` — new module with append-only WAL writer/reader, segment rotation, replay
- `src/brain.rs` — `Brain` gains an optional `Option<MemTable>` field (opt-in via flag for Phase 3 testing; default off)

**Success criteria:**
- Write 1M bonds to MemTable, verify size bounded.
- Crash test: kill daemon mid-write, restart, WAL replay restores state byte-exact.
- Concurrent inserts from multiple ingest threads: lock-free or fine-grained-lock correctness.

**Tests:**
- Round-trip: insert → WAL → kill → replay → query → equal.
- Stress: 100K bonds/sec sustained insert, WAL doesn't fall behind.
- Size threshold trigger: at 500 MB, atomic swap fires, snapshot exposed for flush.

**Risk:** MEDIUM. Concurrency. WAL durability semantics need careful design (fsync cadence vs throughput tradeoff). The "swap" primitive must be lock-free w.r.t. ongoing inserts to avoid blocking ingest.

**LOC:** ~1,300.

### Phase 4 — MemTable flush + L0 SST writes (3–5 days)

**Goal:** wire MemTable swap to a background SST flush thread. Each swap produces a new L0 SST on disk. Manifest updated atomically.

**Files:**
- `src/flush.rs` — new module with background flush thread, MemTable snapshot → SST writer pipeline
- `src/manifest.rs` — new module with atomic manifest read/write/update
- `src/brain.rs` — Brain owns the flush thread, starts on daemon boot

**Success criteria:**
- 10M bonds ingested via MemTable, flush thread produces ~20 L0 SSTs on disk, manifest correctly references all of them.
- Resident RAM bounded at MemTable cap + BlockCache cap throughout.
- WAL truncated after flush succeeds (proves the durability handoff).

**Tests:**
- Stress: 100M bonds, verify all readable post-flush.
- Crash mid-flush: SST partial-write detection, manifest rollback to pre-flush state, WAL replay covers the lost MemTable.
- Multiple concurrent flushes blocked (only one flush at a time per level).

**Risk:** MEDIUM-HIGH. The fsync ordering matters: SST must be durable before manifest update, manifest must be durable before WAL truncation. Existing OP-SAVE-ASYNC self-heal pattern internal notes is the design reference.

**LOC:** ~1,200.

### Phase 5 — Read path through MemTable + L0 SSTs (3–4 days)

**Goal:** integrate MemTable + L0 SST reads into the canonical `Brain::get_outgoing_bonds` path. Behind a feature flag — v3 path stays default until Phase 6.

**Files:**
- `src/brain.rs` — read path: check MemTable, check all L0 SSTs, merge with v3 archive, dedup by `(subject, predicate, object)`.
- `src/network.rs` — spread activation uses the merged read path.

**Success criteria:**
- All recall tests pass with v4 read path enabled.
- Spread activation latency ≤ v3 + 10%.
- Phase 0 benchmarks pass: full conversation tests show v4 + v3 hybrid produces identical bond sets to pure v3.

**Tests:**
- A/B comparison: same query against v3-only vs v3+v4 brain. Identical results.
- Spread activation 2-hop latency under various seed counts.
- Concurrent read + write stress.

**Risk:** MEDIUM. The merge logic across MemTable, L0 SSTs, and v3 archive is intricate. Wrong dedup ordering = stale bonds win. Test discipline is critical.

**LOC:** ~600.

### Phase 6 — Leveled compaction (5–7 days)

**Goal:** background compaction thread that promotes L0 → L1 → L2. Subject ranges within a level become disjoint after compaction, enabling single-SST-per-level reads.

**Files:**
- `src/compaction.rs` — new module with compaction worker, level promotion logic, subject-range splitting
- `src/manifest.rs` — updates to track per-level SST sets with subject ranges
- `src/brain.rs` — compaction thread lifecycle, scheduler (trigger on level size threshold)

**Success criteria:**
- After ingesting 500M bonds: L0 has ≤ 10 SSTs, L1 covers disjoint subject ranges, L2 covers disjoint subject ranges. Total SST count grows sub-linearly.
- Compaction throughput ≥ 100 MiB/s (NVMe should allow this easily with parallel CRC + the OP-PARZSTD pattern reused for compaction zstd).
- Spread activation latency unchanged or improved post-compaction (because levels are now binary-search-disjoint).

**Tests:**
- Long-running compaction test: 24-hour ingest + compaction loop, verify no SST file leaks, manifest never corrupts.
- Compaction-during-read: reads completing during a compaction get consistent snapshot view.
- Tombstone handling: deleted bonds removed during compaction (Phase 7 prerequisite).

**Risk:** HIGH. Compaction is the most complex piece. Subject-range splitting must be correct. Compaction-vs-read concurrency must be lock-free. Failure modes (disk full mid-compaction, killed mid-compaction) need explicit recovery paths.

**LOC:** ~1,800.

### Phase 7 — v3 → v4 migration tool (3–4 days)

**Goal:** convert an existing v3 brain (`brain.lbr`) into a v4 layout (`brain/` directory with manifest + SSTs + WAL). Standalone CLI tool, not invoked automatically.

**Files:**
- `bin/lu4d_migrate_v3_to_v4.rs` — new binary
- `src/migration_v3_v4.rs` — new module with migration logic

**Success criteria:**
- Migrate the production-snapshot v3 brain (5.9 GiB) to v4 layout. Verify bond-by-bond equality. Estimated runtime ~10 minutes (sequential read of v3 + sequential write of v4 SSTs).
- Migrated v4 brain loads cleanly, all queries return identical results to the v3 source.
- Migration is idempotent: running it twice doesn't double-create SSTs.

**Tests:**
- Migrate small (1K bonds), medium (1M bonds), large (10M bonds) v3 brains. Round-trip equality.
- Killed mid-migration: cleanup of partial output, re-run from scratch.

**Risk:** MEDIUM. Migration must be byte-exact; bugs here = silent data loss. The discipline is "if any bond is lost or duplicated during migration, abort and don't claim success."

**LOC:** ~700.

### Phase 8 — Cutover + deprecation of v3 write path (2–3 days)

**Goal:** flip v4 to default. v3 read still supported for migrated-brains compatibility, but no new v3 archives are written. Deprecate old code paths.

**Files:**
- `src/brain.rs` — default to v4 write path, v3 write path becomes feature-gated for emergency rollback
- `src/main.rs` — startup: if `--brain` points at a v3 file (not directory), refuse with helpful error pointing at the migration tool
- `scripts/ingest_curriculum_full.py` — no changes (the MCP API is unchanged; v4 is invisible to the harness)

**Success criteria:**
- Production deploy: migrate the current production brain, swap in v4 binary, restart ingest.
- Ingest resumes from checkpoint, runs at ≥ v3 throughput.
- Resident RAM at 6.7M articles ≤ 10 GiB (vs projected 600 GiB pre-v4).
- Tier 1 completion runs end-to-end without memory pressure.

**Tests:** production canary itself.

**Risk:** MEDIUM. Standard deploy risks. Mitigated by keeping v3 read available and the rollback path well-documented.

**LOC:** ~300.

---

## Total scope summary

| Phase | LOC est. | Days est. | Risk |
|-------|----------|-----------|------|
| 0 | 600 | 2–3 | LOW |
| 1 | 1,500 | 5–7 | MEDIUM |
| 2 | 1,200 | 4–6 | LOW |
| 3 | 1,300 | 4–6 | MEDIUM |
| 4 | 1,200 | 3–5 | MEDIUM-HIGH |
| 5 | 600 | 3–4 | MEDIUM |
| 6 | 1,800 | 5–7 | HIGH |
| 7 | 700 | 3–4 | MEDIUM |
| 8 | 300 | 2–3 | MEDIUM |
| **Total** | **~9,200** | **31–45 days** | — |

At an aggressive but realistic pace (parent + 1 sub-agent on each phase, with careful test discipline), this is **3–5 weeks of focused work**. Per the `feedback_timeline_optimism` discipline, the lower bound is more realistic for the easy phases and the higher bound is more realistic for Phase 6 (compaction). Plan for the upper bound, deliver toward the lower.

---

## Migration plan

### Pre-migration state (where we are now)

- v3 brain at `/var/lib/lu/brain.lbr` (5.9 GiB, 15.6M bonds).
- Curriculum-ingest paused.
- Phases 0–7 implemented and tested in `/tmp/lu_4_mcp.v4-*` build artifacts.
- v3 binary remains in production until Phase 8.

### Phase 7 dry run (no production impact)

1. Snapshot v3 brain to `/tmp/brain.lbr.v3-snapshot` (~10 minutes for 5.9 GiB on NVMe).
2. Run migration tool against the snapshot, output to `/tmp/brain-v4/`.
3. Boot a v4-enabled daemon against `/tmp/brain-v4/` with `--no-save`.
4. Run a 100-question recall test: every answer must be byte-identical to the v3 daemon's answer.
5. Audit MemTable+SST round-trip for 100K new bonds via `--no-save` test mode.

### Phase 8 cutover

1. Stop curriculum-ingest (current binary, current brain).
2. Snapshot v3 brain (paranoia: keep `/tmp/brain.lbr.pre-v4-cutover`).
3. Run migration to canonical `/var/lib/lu/brain/` (new directory).
4. Update systemd unit: `--brain /var/lib/lu/brain/` (directory, not file).
5. Deploy v4 binary.
6. Start ingest.
7. Watch: first MemTable flush succeeds, first compaction completes, resident RAM stays bounded.

### Rollback plan

If anything goes sideways:

1. Stop ingest.
2. Restore v3 binary (`/tmp/lu_4_mcp.pre-v4-cutover`).
3. Restore v3 brain snapshot to `/var/lib/lu/brain.lbr`.
4. Reset checkpoint to the pre-cutover state.
5. Update systemd unit back to `--brain /var/lib/lu/brain.lbr`.
6. Restart.

Total rollback time: ~15 minutes.

---

## Glossary additions

| Name | Definition | Phase |
|------|------------|-------|
| `CompactBond` | 14-byte packed bond representation (Phase 2.1; was 13 in Phase 2). Subject+object as `u32` NodeIds, predicate+flags in 1 byte, weight in 1 byte, `keyframe_offset: u16 + keyframe_count: u8` addressing a contiguous slice of the per-block keyframe pool, confidence `u8`. | 2 (amended 2.1) |
| `NodeIntern` | The string ↔ NodeId interning table. Stays in RAM; node count plateaus per healthy growth curve. | 1 |
| `SST` (Sorted Subject Table) | Immutable on-disk file containing zstd-compressed encrypted blocks of bonds, subject-sorted. Lu convention; analogous to RocksDB SSTable. | 2 |
| `Block` | A zstd-compressed encrypted payload inside an SST. 256 KiB target, contains a subject range of bonds. | 2 |
| `MemTable` | Bounded in-memory write buffer. `BTreeMap<NodeId, Vec<MemBond>>`. Capped at 500 MB by default. | 3 |
| `MemBond` | In-memory bond representation with sequence number for tombstone ordering. Slightly larger than CompactBond. | 3 |
| `WAL` (Write-Ahead Log) | Append-only on-disk log for MemTable durability. Replayed on daemon start. | 3 |
| `Manifest` | Atomic-updated metadata file listing all live SSTs and their level + subject ranges. | 4 |
| `Level` | A tier in the compaction hierarchy. L0 = recent flushes, L1+ = compacted, disjoint subject ranges. | 6 |
| `Minor compaction` | L0 → L1 compaction. Merges overlapping L0 SSTs with L1 SSTs. | 6 |
| `Major compaction` | Periodic global compaction. Removes tombstones, re-sorts, can apply confidence-tiered eviction. | 6 |
| `Tombstone` | Marker for a deleted bond. Lives in MemTable + SSTs until removed by compaction. | 6 |
| `Curiosity prefetch` | Background BlockCache prefetch driven by Lu's wonder queue. Predicts which subjects are about to be queried and warms their blocks. | (later optimization) |

---

## Open questions

1. **Tombstone semantics for bond updates.** When a bond's weight changes (via decay or critic vetting), is that a tombstone + insert, or an in-place update via sequence numbers? Sequence numbers are cleaner; locks may be needed. Decide in Phase 3.

2. **Compaction priority during ingest.** Should compaction throttle when ingest is hot? Otherwise both compete for CPU + disk bandwidth. Probably yes; configurable rate limit. Resolve in Phase 6.

3. **Sidecar files (.kw, .style.wal, .lb_session.wal) under v4.** Do they live alongside the SSTs in the brain directory? Most likely yes — they're already file-based and don't need MemTable. The .kw keyword index might need its own compaction story (separate port).

4. **Encryption key rotation.** Today master.key is fixed. If we ever rotate, all SSTs need re-encryption. Compaction is a natural time to do this. Defer to v4.1+.

5. **Multi-process safety.** v4 should support a read-only daemon (for queries) alongside a write daemon (for ingest), both reading the same SSTs. Manifest needs to be readable atomically by both. Defer to v4.1 if not needed at v4 ship time.

6. **Backup/restore semantics.** v3 is a single file; easy to back up. v4 is a directory with many files. Need a defined backup procedure (probably: snapshot the manifest + all referenced SSTs + the WAL, in that order). Document in Phase 8.

---

## Why this is the right path

- It's the same architectural pattern that 50 years of database systems converged on (LSM-trees, mmap'd storage, page-cache reliance) — Lu doesn't have to invent the boring parts.
- It's the same pattern the LLM ecosystem converged on for weight loading (mmap + immutable + offset-addressable + page cache) — Lu doesn't have to invent the layout strategy either.
- The Lu-original elements (subject-clustered layout, curiosity-driven prefetch) are real intellectual contributions because they exploit structure unique to associative knowledge graphs that databases don't share and LLMs don't have. These are publishable contributions and Glasswing-pitch fodder.
- The migration path is incremental. Phases 0–6 can be tested without touching production. Phase 7's migration is dry-runnable. Phase 8's cutover has a 15-minute rollback.
- The math works. 30 bytes/bond × 335M bonds = 10 GiB on disk, fits in BlockCache easily. The 64 GiB box can hold tier 1 + tier 2 + retrains comfortably.
- The hardware story uses what we have. 2× NVMe in RAID 1 today, optional RAID 0 or split tiering later for bandwidth. 64 GiB RAM is plenty.

The non-obvious win is that this architecture **makes Lu strictly better at scale, not just memory-bounded**. Cold blocks page out, hot blocks stay resident. Spread activation gets faster because subject-clustering means fewer block touches. Compaction can apply structural-value scoring to evict bonds nobody references — Lu's brain self-prunes over time.

If Lu ships v4, **brain size stops being a ceiling**. The model can ingest the entire English Wikipedia, Visual Genome, ArXiv, Stack Exchange, and Grokopedia — and still fit on a laptop with a USB SSD.

That's the prize.

---

## Decision log

- **2026-06-01:** John approved scoping the port plan; pause training, build the plan.
- **2026-06-01:** Hardware: keep RAID 1 for v4 ship, evaluate RAID 0 / split tiering post-deploy.
- **2026-06-01:** CompactBond target 13 bytes (subject + object + predicate + weight + keyframe + confidence). Quantization is non-lossy w.r.t. current f32 weights (256-level monotonic mapping is sufficient for activation purposes).
- **2026-06-01 (Phase 2.1):** CompactBond grown to 14 bytes — `keyframe_idx: u16` replaced by `keyframe_offset: u16 + keyframe_count: u8` so each bond carries a contiguous keyframe range from the per-block pool. Motivation: preserve synthesis / session / decay history through Phase 4 flush, per `feedback_curiosity_load_bearing`. SST format version 1 → 2; v1 fixtures (Phase 2 test-only) are refused at read with a clear error. Cross-bond keyframe-pool dedup deliberately deferred (writer simplicity wins; zstd absorbs the redundancy).
- **2026-06-01:** Compaction strategy = leveled (RocksDB-style), not tiered (Cassandra-style). Leveled gives single-SST-per-level reads at higher write amplification. Lu is read-dominated at production; leveled is the right tradeoff.
- **2026-06-01:** Subject-clustering chosen over object-clustering or predicate-clustering. Spread activation is subject-driven (start at seeds, follow outgoing bonds). Object queries are rare. Predicate queries are secondary.

---

## Status (2026-06-01)

All 8 phases shipped on `main`. Production cutover is operator-triggered via `docs/op_brain_v4_cutover_runbook.md`.

| Phase | Topic | Commit | the project log |
|-------|-------|--------|-----------|
| 0 | Port plan + v3 baseline bench + workspace crate | `05e2133` | #1117 |
| 1 | Seed interning + adjacency shrink | `b5bb9f2` | #1120 |
| 2 | SSTable read path + writer + block LRU | `f82e040` | #1121 |
| 2.1 | CompactBond 14-byte: multi-keyframe per bond | `33ee161` | #1123 |
| 3 | MemTable + WAL primitives | `b1e2ff7` | #1122 |
| 4 | Flush worker + L0 SST writes + manifest | `3b29e70` | #1124 |
| 5 | Three-layer read path (MemTable + L0 + v3) | `a920f2e` | #1125 |
| 6 | Leveled compaction + tombstones + parsed-block cache | `0518d28` | #1126 |
| 7 | v3 -> v4 migration tool with byte-exact verify | `0e46572` | #1127 |
| 8 | v3/v4 boot detection + v4 default flush/compaction + runbook | Phase-8 HEAD on main (parent 0e46572) | #1128 |

The production binary on Hetzner at the time of this writing is still md5 `41f33466782fd1a2499a72ec38c44ff9` (the OP-PARZSTD binary at internal notes. The Phase 8 binary is built on `main` but has NOT been deployed; cutover is the operator's call.

Production rehearsal (Phase 7, internal notes verified byte-exact migration on a 15.5M-bond canonical brain into a 974 MiB v4 directory (~17x on-disk shrink before compaction). The synthetic Phase 8 bench (`docs/v4_phase8_bench_2026-06-01.md`) shows a 41x cold-boot speedup and below-noise-floor RSS delta on a 200K-bond fixture; the canonical proof point will be captured during operator cutover.


---

## Phase 9 — wire write path through MemTable+WAL for v4 brains (2026-06-02)

### Why a phase 9 exists

Phases 1–8 set up MemTable + WAL + flush + compaction + read-path integration, and Phase 8 added the boot-detection plumbing that routes `--brain <v4-dir>` through `Brain::load_v4`. The cutover daemon shipped from main HEAD (Phase 8 hotfix `6e31999`) on 2026-06-01 booted cleanly against the migrated canonical brain, accepted `lu4d_teach_corpus_batch` calls for 6.88M bonds, and reported `bonds_landed=N` to the harness on every batch. Then John noticed something was off: the WAL at `/var/lib/lu/brain-v4/wal/current.wal` was still 8 bytes (the magic header alone). RSS climbed to 13.7 GB. The daemon was rolled back to the v3 binary `md5 41f33466` while the gap was diagnosed.

The gap: **`lu4d_teach_corpus_batch` -> `tool_teach_corpus_batch` -> `tool_teach_corpus` -> `brain.teach_corpus_with_kind(...)` -> `brain.teach_with_source_kind(...)` -> `brain.teach_with_kind(...)` wrote bonds to `brain.network.add_bond_with_scar_check`, NOT to the MemTable + WAL.** Every accepted teach landed in the v3 in-memory HashMaps; the WAL never saw them; `lu4d_save` short-circuited as a no-op for v4 brains (correct per Phase 8); a restart would have lost every claimed-landed bond.

### The fix

`Brain::teach_with_kind` is the unique chokepoint for every teach path in the system (verified by tracing `teach`, `teach_corpus`, `teach_with_source`, pending `apply_change`, research `apply_proposal`, and 23 MCP teach-shape entry points). Phase 9 adds a single `brain_dir_v4.is_some()` branch at the bond-formation step: when the brain was loaded via `load_v4`, route through `Brain::teach_with_kind_v4` which inserts the bond into the MemTable and appends a matching WAL record. The pre-existing v3 path is preserved bit-for-bit for v3 brains.

#### Scope

**In scope:**
- `Brain::teach_with_kind` v3/v4 dispatch (closes the gap for every MCP teach path automatically).
- `Brain::teach_with_kind_v4` helper: MemTable::insert + Wal::append + scar admission + reinforce-vs-new probing.
- `tool_status` v4 dispatch: `bond_count` sourced from MemTable + manifest SST entries; `node_count` sourced from MemTable subject_count + in-memory `network.nodes`; `format_version = "v4"`.
- `tool_hub_status` v4 dispatch: returns the configured caps + empty `top_hubs` + `hub_degree_unwired: true` flag (the v3 in-degree counters are not meaningful for v4-loaded brains).
- 8 unit tests + 1 end-to-end MCP wire test.

**Deliberately out of scope (deferred to follow-on ports):**
- **Hub-degree accountant for v4.** `teach_with_kind_v4` skips hub-degree write-side enforcement. The v3 in-degree counters live on `network` and stay zero for v4 brains; transplanting them to MemTable + SST requires a per-subject degree accountant rebuilt at boot from the manifest. Tracked separately.
- **Scar concept (cascade) for v4.** `Brain::scar_concept` and `Brain::unscar_concept` enumerate `network.bonds()` to find every bond touching a seed. For v4 brains those bonds are in MemTable + SSTs; scar-concept on a v4 brain currently affects only the v3 archive bonds (empty for fresh v4 brains). Per-bond scar (`Brain::scar_bond` / `Brain::unscar_bond`) keeps working because the scar log overlay is orthogonal to bond storage.
- **Variant promotion for v4.** `Brain::promote_variant` iterates `network.bonds()` and mutates existing bonds (`network.get_bond_mut`) by appending keyframes. MemTable bonds don't support keyframe append in-place; the v4 way to express promotion is to insert a fresh MemBond with the `PromotedFromVariant` source. Tracked separately.
- **Node persistence for v4.** SSTs carry per-block subject tables but NOT `KnowledgeNode` records. On v4 restart, `network.nodes` starts empty, so the keyword index can't map a tokenized query back to a Seed. Phase 9 e2e test confirms: within the same session, `lu4d_ask` retrieves inserted bonds correctly; after restart, status reports 100 bonds but `lu4d_ask` returns `total_idk` because the keyword index has no node entries. This is a known v4 architectural limitation, not a Phase 9 bug. The closing port will likely be a side-car nodes file alongside the SSTs (KnowledgeNode is tiny; the cost is in the per-node insert path).

### Updated cutover protocol

The pre-Phase-9 runbook (`docs/op_brain_v4_cutover_runbook.md`) is still correct for the boot sequence and migration steps. The Phase 9 amendment is a single new post-deploy sanity check:

1. After deploying the Phase 9 binary (`md5 85371606e1a14b180b1e8294080002af`) to `/usr/local/bin/lu_4_mcp` and restarting `lu4d-curriculum-ingest.service`, watch the WAL size:
   ```
   watch -n 5 'stat -c%s /var/lib/lu/brain-v4/wal/current.wal'
   ```
   The pre-Phase-9 daemon held this at 8 bytes forever. The Phase 9 daemon grows it on every accepted batch (~110 bytes/bond at the default keyframe shape).
2. Within 1 minute of starting ingest, expect WAL size to exceed 100 KB. If it stays at 8 bytes, ROLLBACK — the routing fix is not in effect.
3. After the first L0 flush cycle (default 30s poll, only triggers at MemTable 500 MiB cap), expect `/var/lib/lu/brain-v4/L0-*.sst` files to appear and the WAL to truncate. Confirm via the manifest: `lu4d_status` reports `bond_count > 0` even after restart.
4. RSS sanity: with the Phase 9 binary, RSS should stabilize at `MemTable cap (500 MB) + BlockCache cap (2 GB) + base overhead (~3 GB)` ≈ 5-6 GB resident under sustained ingest. The pre-Phase-9 13.7 GB RSS was the v3 HashMap leak (every bond duplicated into the in-memory graph).

### Verification artifacts

- Build artifact: `/tmp/lu_4_mcp.phase9-write-path-2026-06-02` (md5 `85371606e1a14b180b1e8294080002af`).
- Unit tests: `cargo test --release --lib cutover_v4_write_path_phase9_tests` — 8/8 pass.
- E2E wire test: `bash /tmp/phase9/e2e_test.sh` — WAL 0 -> 11,108 bytes, batch 100 landed, status 100 bonds, ask 1 hit, post-restart status 100 bonds.
- Test suite: 1763 lib tests + 14 integration tests (1777 total) all pass; baseline pre-Phase-9 was 1755 — delta is the 8 new Phase 9 tests + Phase 1-8 tests added in the same window.


## Phase 10 — node persistence (2026-06-02)

### The gap

Phase 9 (commit `a69a103`) closed the bond write-path-routing gap surfaced by the 2026-06-01 cutover. The Phase 9 end-to-end test confirmed that on a fresh v4 brain, `lu4d_teach_corpus_batch` of 100 entries grew the WAL from 0 to 11,108 bytes, the daemon's in-memory MemTable held 100 bonds, and `lu4d_status` reported the same count post-restart. **But** `lu4d_ask` for any inserted subject still returned `total_idk` after the restart cycle.

Diagnosis: WAL replay rebuilt the MemTable bonds, but the nodes those bonds referenced had no metadata. `KnowledgeNode` records — content text, `KnowledgeContent` kind, `Confidence` band, timestamps, embedding — lived only in `Brain::network.nodes` (an in-memory HashMap), and that HashMap was empty on a fresh boot. SST blocks carried a per-block subject table that mapped the local NodeId u32 to a `Seed`, but they did not carry the `KnowledgeNode` payload alongside.

The symptom: every read path that goes through `network.get_node(seed)` (lu4d_ask, narrate, recall) found `None` for every subject post-restart, so the merge surfaced bonds with no content to render.

### The fix

Phase 10 extends the v4 substrate to make nodes first-class durable records via the same MemTable + WAL + SST + compaction path the bonds use.

**SST format v2 -> v3.** The block payload header's `reserved: u32` field becomes `node_table_size: u32` (must equal `subject_table_size`). After the subject table, the block carries a parallel node table where each slot is a 4-byte length-prefix + bincode-encoded `Option<KnowledgeNode>`. The reader refuses v1/v2 SSTs with a clear error pointing at the migration tool. **There is no v2 -> v3 conversion path; operators must re-run `lu4d_migrate_v3_to_v4` to produce v3 SSTs.**

**WAL record-type discriminator.** A new byte sits immediately after the 4-byte length prefix: `0 = Bond`, `1 = Node`. Bond bodies are unchanged from Phase 3. Node bodies carry `seq:u64 + seed[32] + node_bytes_len:u32 + node_bytes`. CRC32 trailer scheme is the same. The magic header stays `LU4WAL\x01\x00`.

**MemTable nodes table.** A `BTreeMap<NodeId, (KnowledgeNode, u64)>` parallel to the existing bond entries. New methods: `insert_node`, `lookup_node`, `iter_nodes_for_flush`, `node_count`. `swap_for_flush` drains both maps atomically.

**Brain write-path wiring.** `teach_with_kind` now calls a new `persist_node_v4` helper after `network.add_node` (when on v4) for both subject and object. The helper inserts into MemTable nodes + appends a WAL Node record. Idempotent via MemTable membership check (so reinforce-existing teaches do not double-write).

**Brain boot-path wiring.** `load_v4` calls a new `rehydrate_network_nodes_from_v4_ssts` immediately after `enable_flush_worker`. It walks every SST in the manifest and re-inserts node payloads into `network.nodes` so post-restart `lu4d_ask` finds subjects. The WAL replay in `enable_memtable` also rebuilds `network.nodes` from any pending WAL Node records (covers the period between the last L0 flush and the kill).

**Flush worker.** `flush_once` collects `iter_nodes_for_flush` alongside bonds and uses `push_bond_with_nodes` (a new SSTWriter method) so subject + object KnowledgeNode payloads land in the SST's per-block node table. Isolated nodes (not referenced by any flush-time bond) land via a standalone `push_node`.

**Compaction.** `compact_once` collects per-Seed node winners using highest-SST-id (== highest seq) wins. The output writer attaches winners to bond pushes via `push_bond_with_nodes`.

**Migration tool.** Materialises the v3 archive's nodes into a `Seed -> KnowledgeNode` map up front and attaches subject/object node payloads on every bond push. Isolated nodes are swept into a final SST via `push_node`. The verify pass walks every v4 SST's `iter_nodes` and asserts every node byte-matches the v3 source via bincode round-trip. `MigrationReport.nodes_migrated` and `VerifyReport.nodes_verified` track the counts.

### Migration re-run requirement

**The migrated brain at `/var/lib/lu/brain-v4/` was produced by the Phase 8 migration tool, which writes SST v2 format.** The Phase 10 reader refuses v2 SSTs. Operators must re-run the Phase 10 migration tool against the v3 canonical brain (`/var/lib/lu/brain.lbr`) before deploying the Phase 10 binary:

```
lu4d_migrate_v3_to_v4 --v3 /var/lib/lu/brain.lbr --v4 /var/lib/lu/brain-v4-phase10/
```

The Phase 10 binary then loads `/var/lib/lu/brain-v4-phase10/` (rename to `/var/lib/lu/brain-v4/` after a clean snapshot of the pre-Phase-10 state). The verify pass at the end of `--run` will report `nodes_migrated` and `nodes_verified` both equal to the v3 archive's node count.

### Updated cutover runbook

Replace the Phase 9 deploy step with:

1. Snapshot `/var/lib/lu/brain.lbr` to a safe path.
2. Run the Phase 10 migration tool against the snapshot, writing a fresh `/var/lib/lu/brain-v4-phase10/`. Wait for the verify pass to succeed (expect `nodes_migrated == nodes_verified`).
3. Stop `lu4d-curriculum-ingest.service`.
4. Rename the Phase 8 `/var/lib/lu/brain-v4/` aside and swap in the new dir.
5. Deploy the Phase 10 binary (`md5 2aedd5cde6edacf1e2633393b7a2e05f` at `/tmp/lu_4_mcp.phase10-node-persistence-2026-06-02`) to `/usr/local/bin/lu_4_mcp`.
6. Restart the service. Sanity-check: `lu4d_status` reports `bond_count == migration's bonds_migrated`. `lu4d_ask` against any known subject in the v3 brain returns ≥1 hit. After 1 minute of ingest, WAL grows past 100 KB. After the first flush, L0 SSTs appear and node payloads round-trip through `iter_nodes`.

Rollback procedure: restore `/usr/local/bin/lu_4_mcp` from the v3 baseline (`md5 41f33466782fd1a2499a72ec38c44ff9`) AND restart against the pre-cutover `/var/lib/lu/brain.lbr` (the Phase 10 v4 directory is not v3-compatible, but the v3 binary ignores the v4 dir).

### Verification artifacts

- Build artifact: `/tmp/lu_4_mcp.phase10-node-persistence-2026-06-02` (md5 `2aedd5cde6edacf1e2633393b7a2e05f`).
- Unit tests: 13 new Phase 10 tests pass cleanly; total suite 1790 passing (baseline 1777, delta +13).
- The critical regression test is `phase10_node_metadata_persists_across_restart` in `src/brain.rs`: it teaches 10 bonds, drops the brain, reloads via `Brain::load_v4`, and asserts every subject and object Seed returns `Some(KnowledgeNode)` via `network.get_node`. Pre-Phase-10 this test would fail (the symptom that motivated the port).
- `phase10_node_metadata_persists_via_sst_path` exercises the same flow but forces a flush before kill, exercising the `rehydrate_network_nodes_from_v4_ssts` path.
- `phase10_migration_preserves_node_metadata` proves round-trip byte-exactness via `nodes_verified == nodes_migrated`.
- `phase10_compaction_dedups_node_records_keeping_latest` proves compaction picks the highest-SST-id winner per Seed.

### Phase 10.1 streaming fix 2026-06-02

**Discovery during 2026-06-02 re-cutover attempt.** The Phase 10
migration tool (`8b3039d`) ran 11 minutes against the 6.79 GB v3
production brain (18.6M bonds / ~9M nodes), grew RSS to ~28 GB, and
produced ZERO SST output before the operator aborted. Root cause:
`V3ToV4Migrator::run` preloaded every v3 `KnowledgeNode` into a
`HashMap<[u8;32], KnowledgeNode>` before pushing any bond -- a
load-everything-then-write antipattern the Phase 10 agent flagged as
"~190 MiB for 105K nodes, well under the migration headroom budget"
but which materialised ~20 GiB of heap at production scale. The
Phase 10 correctness tests (small synthetics) all passed; they did
not exercise the streaming property.

**The fix.** Single-pass, subject-sorted migration that streams node
payloads through `archive.node_by_seed` (O(1) HashMap lookup against
the mmap'd v3 archive) on demand. Only state retained across the
bond walk is (a) the sorted `Vec<(Seed, Seed)>` of bond keys --
unavoidable for sort, ~1.2 GB on the production brain -- and (b) a
`HashSet<[u8; 32]>` of emitted seed bytes for the isolated-node
sweep -- ~432 MiB at 9M nodes. Per-bond materialisation is two
owned `KnowledgeNode` instances that drop at the end of the loop
iteration. The isolated-node sweep walks `archive.iter_node_seeds()`
and fetches each missed seed via `archive.node_by_seed` for its
standalone `push_node` call. Total expected peak RSS at production:
~2 GiB, comfortably under the 4 GiB ceiling called out in the spec.
Matches Phase 7's bond-streaming model.

**New production-scale test that gates against regression.**
`phase10_1_streaming_migration_bounded_rss_at_production_shape` in
`src/migration_v3_v4.rs` builds a 600K-bond / 150K-node synthetic
v3 brain, runs the migration with a 16 MiB SST target (driving
multi-SST output), and asserts the peak RSS delta over the pre-
migration baseline is bounded by 1 GiB. Background thread samples
`/proc/self/status:VmRSS` every 25 ms. The pre-Phase-10.1 preload
path would breach this (~1.3-1.5 GiB delta for 150K nodes at ~3 KiB
each). Streaming measured at ~176 MiB delta during the validating
run; that extrapolates to ~2 GiB at the production 18.6M-bond shape.

### Verification artifacts (Phase 10.1)

- Build artifacts: `/tmp/lu_4_mcp.phase10.1-streaming-migration-2026-06-02` (md5 `7b32a72b592cfadb843eb2277591530a`) and `/tmp/lu4d_migrate_v3_to_v4.phase10.1-2026-06-02` (md5 `a6fe54ffe0bae190168e2bba96f26b05`).
- Test suite: 1791 total (1777 lib + 14 integration), all green; delta +1 over the 1790 baseline (the new production-scale RSS regression test).
- Production binary `/usr/local/bin/lu_4_mcp` UNCHANGED at md5 `41f33466782fd1a2499a72ec38c44ff9` (rolled-back v3, ingest continued without interruption throughout Phase 10.1 work).

### Re-cutover path

Operator may now re-run the Phase 10.1 migration tool against a
snapshot of `/var/lib/lu/brain.lbr` and expect bounded RSS (under
4 GiB) and SST progress within the first ~30 seconds. Cutover
protocol otherwise unchanged from Phase 10. The compaction-side
`Seed -> KnowledgeNode` map in `compact_once` is a separate concern
at a different scale point (bounded by per-compaction working set,
not total brain size) and is explicitly out of scope for Phase 10.1
per spec.
