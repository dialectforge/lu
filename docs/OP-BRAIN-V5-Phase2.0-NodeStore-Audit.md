# OP-BRAIN-V5 Phase 2.0 -- Node Store Audit

**Status:** Phase 2.0 read-only investigation complete 2026-06-09. No source files modified. Companion to `OP-BRAIN-V5-Design.md` Section 2 Q3/Q6/Q7/Q8 and Section 4 Phase 2. Modelled on the Phase 1.0 NodeId audit; codebase HEAD `9c69684` (Phase 1.4 wiring shipped).

**Scope per design doc Section 2 Q3 + Section 4 Phase 2:** stand up a separate "node store" with its own MemTable / WAL / SST / flush / compaction primitives, persisting `NodeIntern` as a `nodeintern.bin` sidecar inside the `.lbr` tar (Q3/Q6). Phase 2.0 reads the v4 substrate from the node-store perspective and produces a sub-port sequencing recommendation BEFORE any Phase 2.x code lands.

**Headline finding:** the v4 substrate (`memtable.rs`, `wal.rs`, `manifest.rs`, `flush.rs`) is **shape-salvageable but not parameterically reusable** as a shared `lu_substrate` crate without invasive rework. The cleanest path is a **per-store copy** keyed to node-shaped payloads: `src/node_store/{memtable,flush,sst,sst_writer,compaction,sidecar}.rs`, sharing the existing `src/wal.rs` and extending `src/manifest.rs`. Three concrete reasons (detailed Section 3): MemTable's bond-table is hard-wired to `Vec<MemBond>` payloads; WAL's record discriminator + fsync policy is shared but record codecs are payload-specific; flush.rs's `node_map` HashMap-the-world (`flush.rs:312-316`) only makes sense once nodes are routed to a separate store, at which point it goes away.

**Concrete sidecar sizing correction.** The design doc's "100M nodes x 24 B ~= 2.4 GB" sidecar estimate is the **NodeIntern forward-table cost only at u32 NodeIds**. With Phase 1.1 widening to u64 and the actual `NodeIntern` shape (`HashMap<Seed, NodeId>` + `Vec<Seed>`), per-node runtime cost is `Seed[32] + NodeId[8] + HashMap overhead ~= 48-64 B in HashMap + 32 B in reverse Vec ~= 80-96 B per node resident`. **At 100M nodes: ~8-9.6 GB resident, not 2.4 GB.** Still fits AX41-NVMe 64 GB budget but >3x the design-doc projection; this is SURPRISE-A below and warrants a Section 7 design-doc entry.

---

## 1. Scope -- what was investigated, what wasn't

**Investigated (read-only):**
- `src/node.rs` -- `KnowledgeNode`, `KnowledgeUnit`, `KnowledgeContent` (16 variants), `Confidence`, `EMBEDDING_DIM=384`.
- `src/node_intern.rs` -- `NodeIntern` HashMap+Vec shape, `intern_from_v3_archive`, `reverse_lookup` Vec indexing.
- `src/memtable.rs` -- `MemTableInner.{entries, nodes}`, `insert_node`, `iter_nodes_for_flush`.
- `src/wal.rs` -- record discriminators, `REC_TYPE_NODE_INTERN=2` payload, `replay_streaming_validated` firewall.
- `src/manifest.rs` -- `ManifestPayload`, `SSTEntry` (lines 107-126), `last_compaction` (line 163).
- `src/flush.rs` -- `flush_once`, `node_map` HashMap (line 312), `iter_nodes_for_flush` consumption.
- `src/compaction.rs` -- `node_winners` HashMap (line 362), `rows: Vec<MergeRow>` (line 357).
- `src/brain.rs` -- `persist_node_v4` (line 3951), `enable_memtable` REC_TYPE_NODE_INTERN replay arm (line 9942), `teach_with_kind_v4`.
- `crates/lu_brain_bench/` -- bench harness layout for cross-store extension precedent.
- `OP-BRAIN-V5-Design.md` + `OP-BRAIN-V5-Phase1.0-NodeId-Audit.md` -- guiding decisions.
- `OP-BRAIN-V4-Architectural-Audit-2026-06-02.md` -- current production sizing (9M nodes / 18.6M bonds).

**Not in scope:**
- Migration tool internals (Phase 7 -- Q4 was answered separately).
- Bond store SST format (Phase 3 -- Q2 is locked).
- Snapshot/v5enc tar pipeline (Phase 4 -- Q6 is locked at archive-format level).
- Read path MergedBondView dispatch (Phase 5 -- Q5 is locked).
- v3-archive HashMap-unsorted iter (Phase 7.0 -- Q4 revision is locked).

## 2. KnowledgeNode + current persistence state (Q1)

### 2.1 `KnowledgeNode` struct (the node payload)

Definition: `src/node.rs:312-330`. Fields (with rkyv `Archive` derive at line 309):

| Field | Type | Approx bytes | Notes |
|---|---|---|---|
| `seed` | `Seed` (`[u8; 32]`) | 32 | Cryptographic identity. |
| `content` | `String` | ~80 amortised | Human-readable label / summary. |
| `knowledge` | `KnowledgeUnit` | ~120 | `KnowledgeContent` + `Confidence` + 2x `i64` timestamps. |
| `embedding` | `Vec<f32>` (cap `EMBEDDING_DIM=384`) | **1536** | The dominant cost. 384 x 4 = 1536 bytes payload + 24 B Vec header. |
| `parent_seeds` | `Vec<Seed>` | 24 + (32*n) | Thinking-layer step links. Almost always empty in training corpus. |
| `typed_ratio` | `f32` | 4 | Cached typed-bond fraction (load-time recompute). |

**Total amortised per-node payload: ~1800 bytes** (the existing `APPROX_NODE_BYTES` constant at `memtable.rs:289` already commits to this number). The embedding dominates by ~85%.

### 2.2 The REAL `KnowledgeContent` enum -- 16 variants, NOT 6

Per `feedback_audit_nodetype_vs_kind`, the conceptual `NodeType { Fact, Concept, Procedure, Emotion, Meta, Media }` taxonomy is NOT what the source carries. The source-of-truth definition at `src/node.rs:46-79` is:

```rust
pub enum KnowledgeContent {
    Fact(String),       Concept(String),    Procedure(String),  Episode(String),
    Preference(String), Raw(String),        Code(String),       Image(String),
    Definition(String), Mechanism(String),  Composition(String), Purpose(String),
    Contrast(String),   Causal(String),     Example(String),    Meta(String),
}
```

All 16 variants take a single owned `String` payload. The `kind_str()` accessor at `node.rs:107-126` returns lowercase tags `"fact" | "concept" | "procedure" | "episode" | "preference" | "raw" | "code" | "image" | "definition" | "mechanism" | "composition" | "purpose" | "contrast" | "causal" | "example" | "meta"`. The retrain curriculum's locked `infer_kind_from_relation` table at `node.rs:235-273` covers all of them via 23 BR.3 relation aliases.

**Implication for node store sort order / compaction (SURPRISE-C below):** `KnowledgeContent.kind` is logically a **tag dimension**, not a sort dimension. Compaction-time dedup should key on `NodeId` (or equivalently `Seed` until interned), not on `kind`. Two `KnowledgeNode` records for the same `Seed` with different `kind`s violate the "node creation is idempotent" Q.1 contract at `brain.rs:4019` and should be a tombstone-aware merge with the highest-seq winner -- exactly the pattern already in `compaction.rs:362` (`node_winners` keyed on `Seed bytes`). **No special-case sort by `kind` is needed at the substrate layer.**

### 2.3 Current persistence: where do nodes live in v4 today?

Per Phase 0.a salvage audit + my re-verification:

1. **MemTable (RAM, bounded 500 MiB).** `memtable.rs:130` field `nodes: BTreeMap<NodeId, (KnowledgeNode, u64)>`. Inserted via `MemTable::insert_node` (`memtable.rs:277-302`). Drained via `iter_nodes_for_flush` (`memtable.rs:459-475`) returning `Vec<(Seed, KnowledgeNode, u64)>` sorted by Seed bytes.

2. **WAL (disk, append-only).** Two record types touch nodes:
   - `REC_TYPE_NODE=1` (`wal.rs:224`) -- bincode-serialised full `KnowledgeNode` payload alongside its `seed`. Written by `append_node` (`wal.rs:343`).
   - `REC_TYPE_NODE_INTERN=2` (`wal.rs:234`) -- Phase 1.2: `(seed[32], node_id: varint u64)` binding. Written by `append_node_intern` (`wal.rs:371`).

3. **L0+ SSTs (disk, immutable, encrypted blocks).** Phase 0.a confirmed nodes are **bonded into bond SSTs** via per-block tables. The actual ingress is `flush.rs:312-316`:
   ```rust
   let mut node_map: HashMap<[u8; 32], crate::node::KnowledgeNode> =
       HashMap::with_capacity(nodes.len());
   for (seed, n, _seq) in &nodes {
       node_map.insert(*seed.as_bytes(), n.clone());
   }
   ```
   This `node_map` is the **antipattern Phase 2 retires.** It is bounded by the MemTable cap (so single-flush RAM is bounded) but it ties node payload writes to the same SST as the bonds -- the v4 architectural audit's `OP-BRAIN-V4-Architectural-Audit-2026-06-02.md` Appendix B caught this; v5 separates them.

   Inside the SST, nodes live in the per-block `node_table` (`flush.rs:329-340` pushes via `SSTWriter::push_bond_with_nodes` and the standalone `SSTWriter::push_node`). Compaction reads them back via `parsed.node_table` (`compaction.rs:366-380`) into the `node_winners` HashMap-the-world.

4. **Network in-memory map (`AssociativeNetwork.nodes`).** `enable_memtable` REC_TYPE_NODE replay arm at `brain.rs:9925-9938` calls `self.network.add_node(node.clone())`. This is the v3-vintage in-memory map that recall paths walk. Phase 2 does NOT retire it (it stays as the hot-path lookup; see Q5 + Section 6 below).

### 2.4 `Brain::persist_node_v4` (`brain.rs:3951-4015`)

Phase 1.4 wired three steps under the WAL lock:

1. Idempotency check via `MemTable::lookup_node` -- if the seed is already in the MemTable nodes table, return.
2. `intern.intern_with_new_flag(seed)` -- under the WAL lock. If `new_intern`, emit `append_node_intern(seq, &seed, id)` FIRST.
3. `MemTable::insert_node(seed, node, &intern)` -- adds to BTreeMap.
4. `wal.append_node(seq, &seed, node)` -- the canonical REC_TYPE_NODE record.

**The Q8 ordering invariant is upheld:** REC_TYPE_NODE_INTERN strictly precedes any REC_TYPE_NODE / REC_TYPE_BOND that mentions this seed. This is the correct structural foundation for Phase 2; the node store's own MemTable-+-WAL-+-flush pipeline can adopt the same ordering pattern verbatim.

## 3. Substrate reuse analysis (Q2)

### 3.1 Per-file generic-potential verdict

| File | Generic potential | Verdict | Reasoning |
|---|---|---|---|
| `memtable.rs` (937 LOC) | **LOW** | Per-store copy | `MemTableInner` carries two distinct fields (`entries: BTreeMap<NodeId, Vec<MemBond>>` + `nodes: BTreeMap<NodeId, (KnowledgeNode, u64)>`) with different semantics. Genericizing over `Payload<T>` would make the `predicate_discriminant` sort key + `MemBondFlush` reshape + tombstone-aware `delete` either parametric (compile-time complexity) or store-specific (defeats genericisation). Per-store copy is **~250 LOC** for the node-only version (no MemBond stuff). |
| `wal.rs` (1726 LOC) | **MEDIUM** | Shared discriminator + framing; per-store record codecs | Record framing (length prefix, CRC32, fsync cadence) IS generic. Codecs (`encode_bond_body`, `encode_node_body`, `encode_node_intern_body`) are payload-specific. Compromise rejected for separate-WAL split (see Section 3.3). Keep wal.rs single-file shared by both stores; add node-store-specific append/replay helpers if needed. |
| `manifest.rs` (860 LOC) | **MEDIUM-HIGH** | Per-store SSTEntry; shared file via schema bump | `SSTEntry { id, file, subject_min, subject_max, size, mtime, bond_count }` -- node store needs `node_count` not `bond_count`, and `seed_min/seed_max` semantics differ from bond `subject_min/subject_max`. The `next_sst_id` counter, `levels_ge_1_disjoint` invariant, `referenced_paths`, `save_atomic` rkyv pipeline are reused. Extend `ManifestPayload` with `node_levels: Vec<LevelEntry<NodeSSTEntry>>` + add `next_epoch: u64` (Q7 below). Schema bump 1 -> 2. |
| `flush.rs` (809 LOC) | **N/A** | Per-store, totally different shape | Node-store flush has NO bonds, NO `node_map`, NO `seen_in_sst` HashSet. It is conceptually `for (seed, node, seq) in iter_nodes_for_flush: writer.push_node(seed, node, seq)`. Single-pass streaming write. **Roughly ~150 LOC for the node-store version vs 809 for v4 bond flush.** Zero salvage value beyond "study the v4 worker-shared pattern". |
| `sst.rs` (1836 LOC) | **N/A** | Per-store SST format | Bond SST has `CompactBond` + block-local subject_table + node_table + keyframe_pool. Node SST only needs `(NodeId, KnowledgeNode)` sorted by NodeId, optional embedding chunking for cache-friendliness. **Roughly ~400 LOC for a node-store SST reader** (no per-block subject interning, no keyframe handling). |
| `sst_writer.rs` (1076 LOC) | **N/A** | Per-store SST writer | Same story -- no `intern_subject`, no `subject_lookup` per-block table, no keyframe pool. Streaming writer that takes sorted `(NodeId, &KnowledgeNode)` pairs, block-compresses, encrypts, writes block index. **~250 LOC.** |
| `compaction.rs` (1030 LOC) | **MEDIUM (worker scaffold), N/A (merge loop)** | Per-store merge, shared scheduler pattern | The `FlushWorkerShared` / `CompactionWorkerShared` scheduler scaffold IS conceptually reusable (just owns paths, key, manifest handle, parsed-cache handle). The merge loop for the node store is a simpler streaming k-way merge over `(NodeId, KnowledgeNode)` pairs with highest-seq-wins dedup. Per-store merge body ~200 LOC. |

### 3.2 Shared `lu_substrate` crate -- verdict: **NO, do not extract**

Per the file-by-file table, the salvageable cross-store surface adds up to:

- WAL framing scaffold (~300 LOC inside `wal.rs`, not extracted).
- Manifest scaffold (the `save_atomic` + rkyv pipeline, ~150 LOC inside `manifest.rs`, not extracted).
- Worker scaffold pattern (poll interval, swap-for-flush, shared state).
- Total shared: ~650 LOC -- but reused via direct extension of existing files, not via a new crate.

Vs the cost of pulling them into a separate crate:
- New `Cargo.toml` + workspace member declaration.
- Cyclic-dep risk: `wal_framing.rs` needs `crate::node_intern::NodeId` for REC_TYPE_NODE_INTERN. NodeIntern lives in lu4d. Pulling NodeIntern into the shared crate too means pulling Seed in, which means pulling the whole brain-core into the shared crate -- the antipattern the audit warned against.
- Two separate substrate consumers (bond store + node store) but the bond store will ALSO be rewritten in Phase 3; so the shared code only exists once both are written, not progressively.

**Recommendation: do NOT extract `lu_substrate` as a separate crate.** Instead, leave the v4 bond substrate in `src/` (as-is, will be rewritten in Phase 3 anyway) and put the new node store in `src/node_store/` as a module. After Phase 3 rewrites the bond store, look back: if there is genuine duplication, extract THEN.

**Why this is the right call:** Phase 0.b's bench crate (`crates/lu_brain_bench/`) was extracted because it had a clear test-tooling boundary (no production code, separate build profile, runs against any brain). The node store is production substrate -- extracting it now would commit to an API surface before we know what the bond store needs after its own Phase 3 rewrite. Per `feedback_no_hacks`: do the refactor when the duplication is real, not when it's speculative.

### 3.3 The cross-store coupling: REC_TYPE_NODE_INTERN

`REC_TYPE_NODE_INTERN=2` records bind `Seed -> NodeId(u64)`. Both stores need them:
- Bond store: every bond record references NodeIds. The Q8 firewall (`wal.rs:90-120` -- `replay_streaming_validated`'s `UnknownNodeIdInBond` error) requires a NodeIntern record precedes any bond referring to that NodeId.
- Node store: every node record's `seed` becomes a NodeId via the same intern table. Same precedence invariant.

**Two options for routing NodeIntern records:**

(a) **Single shared WAL** holding `REC_TYPE_BOND | REC_TYPE_NODE | REC_TYPE_NODE_INTERN`. Both stores read from it during replay; each filters out the record types it doesn't care about. Pro: NodeIntern ordering invariant is enforced by a single append point. Con: couples flush/truncate cadence (you can't truncate the WAL past the bond store's lowest-pending-flush until the node store has also flushed past that point).

(b) **Per-store WALs** with NodeIntern records duplicated into each. Bond WAL holds `REC_TYPE_BOND | REC_TYPE_NODE_INTERN`; node WAL holds `REC_TYPE_NODE | REC_TYPE_NODE_INTERN`. Pro: independent flush/truncate cadence. Con: duplicate writes and replay logic must reconcile two streams.

**Recommendation: (a) single shared WAL.** Sub-reasons:
- The Q8 ordering invariant is structural and load-bearing. A single append point is the simplest enforcement.
- WAL truncation cadence is gated on flush, not on independence between stores. The shared WAL holds (live bonds since last bond flush) + (live nodes since last node flush); truncating past `min(bond_flushed_seq, node_flushed_seq)` is correct and tractable.
- v4 currently has a single WAL holding both REC_TYPE_BOND and REC_TYPE_NODE. Maintaining the single-WAL model through v5 is the smallest delta.

This means the existing `src/wal.rs` stays as the single shared WAL, with new node-store-shaped append/replay helpers added. The node store does NOT get its own WAL file; it gets its own MemTable + flush + SST + manifest, all sharing the bond store's WAL.

## 4. NodeIntern sidecar (`nodeintern.bin`) design recommendation (Q3 + Q6)

### 4.1 Three candidate formats

| Format | Pro | Con |
|---|---|---|
| (A) Single rkyv blob (atomic write on flush) | Simple, matches manifest pattern, integrity via rkyv `check_bytes` | At 100M nodes x 40-48 B/entry = ~4-5 GB blob -- atomic write becomes a real I/O event (~5-10s on NVMe), bloats peak RSS during write |
| (B) Streamed length-prefixed records (WAL-shape) | Streaming write (bounded peak RSS), append-only after first build | Replay cost is O(N) on every boot; cold-restart cost grows linearly with brain size |
| (C) Sorted node-store SST (sort key = NodeId, value = Seed) | Block-encrypted reuse of SST infrastructure, mmap-friendly partial loads | Re-encrypts a structure that doesn't need bond-store SST features (no compaction, no keyframes, no levels) -- overengineered |

### 4.2 Recommendation: **hybrid (B) + checkpoint snapshot**

Format:
- File: `nodeintern.bin` inside the `.lbr` tar at `node_store/nodeintern.bin`.
- Layout: 16-byte header (magic + schema_version + record_count u32 + reserved u32) followed by `record_count` records of `[seed: 32 B] [node_id: varint u64 LEB128]`, followed by 4-byte CRC32 trailer over the whole file.
- **Sorted by NodeId ascending** (i.e. by insertion order -- `NodeIntern.reverse` is already in this order). Sorting by NodeId means bulk-load is a single pass: `reverse[i] = seed_at_record_i; forward.insert(seed, NodeId(i))`. No sort needed at boot.
- Write path: `NodeIntern::save_to(path)` walks `reverse: Vec<Seed>` in order, streaming records to a `BufWriter`. Peak write-time RSS: ~64 KB (BufWriter cap) + the existing `NodeIntern` resident set. No extra full-file buffer.
- Read path: stream records back, push `Seed` to `reverse`, insert into `forward`. Single pass.

Why hybrid (B) over pure (B): the file IS streamed (bounded peak RSS during write -- solves the 5-10s atomic-write spike of (A)), but the cold-restart cost of (B) is mitigated by the **sidecar = snapshot, WAL = delta** discipline.

### 4.3 Sidecar <-> WAL coordination on cold restart

The discipline (matches Q8's "snapshot-persisted as the `nodeintern.bin` sidecar"):

1. **Sidecar exists (post-Phase 2 brain):** load `nodeintern.bin` -> `NodeIntern.reverse[]` + `.forward{}` are populated. Then replay WAL from `last_flushed_seq+1`. NodeIntern records in the WAL delta are folded into the existing intern table using the WAL's stored NodeId verbatim (`reverse[stored_id.0] = stored_seed; forward.insert(stored_seed, stored_id)`).

2. **Sidecar absent (pre-Phase 2 v4 brain):** no `nodeintern.bin` in the archive. Fall back to the current `enable_memtable` behaviour: `intern.intern_or_get(seed)` for each REC_TYPE_NODE_INTERN, **discarding the WAL's stored node_id** (the existing comment at `brain.rs:9941-9960` is honest about this).
   - Warning logged: `[brain] sidecar absent: pre-Phase 2 brain; NodeIds reassigned by insertion order`.
   - Migration path: the first save after enable_memtable writes a sidecar via `NodeIntern::save_to`. Subsequent boots take the sidecar path.

3. **Sidecar checksum fails:** treat as absent + warning. NodeIntern is rebuildable from the WAL + node-store SSTs; failing to load the sidecar is recoverable.

### 4.4 When can the sidecar-absent path retire?

After Phase 7 (migration tool) seeds the sidecar for any v3->v5 migrated brain, AND any production v5 brain has flushed once post-cutover. **Recommendation: leave the sidecar-absent fallback in for the entire v5 era as defence-in-depth.** The cost is ~20 LOC; the safety is preserving boot-recovery from any partial-write or copy-without-sidecar mishap.

### 4.5 Resident-size correction

Per Q3 the design doc projects "100M nodes x 24 bytes ~= 2.4 GB resident." That number is correct for the **on-disk sidecar payload** (32 B seed + 8 B varint NodeId ~= ~40 B raw, ~24 B per-entry if seeds were varint-able which they aren't -- actually 40 B/entry -> 4 GB on disk at 100M nodes). The **runtime resident** cost of `NodeIntern` is larger:

- `HashMap<Seed, NodeId>`: ~48-64 B per entry (HashMap has 1.5x load factor + control bytes + 32 B key + 8 B value rounded up + per-bucket overhead). At 100M entries: ~5-6 GB.
- `Vec<Seed>`: 32 B per entry. At 100M entries: ~3.2 GB.
- Total runtime resident: **~8-9 GB at 100M nodes.**

**SURPRISE-A -- design doc Section 7 entry recommended.** 3.3-3.8x the projected number. Still fits AX41-NVMe 64 GB budget but the Risk Register Section 5 "NodeIntern memory growth" entry should reflect this; revisit before exceeding 25 GB (~ 250M-300M nodes).

Mitigation candidates if NodeIntern resident becomes a top-3 RSS line item: (i) use a `HashSet<Seed>` instead of forward `HashMap`, deriving NodeId from insertion order via a parallel Vec; (ii) shard the HashMap; (iii) use a perfect-hash structure post-compaction. Phase 2 ships the straightforward shape; mitigation is post-cutover work.

## 5. Tombstone + epoch barrier design (Q7)

### 5.1 Current state

`manifest.rs:163` `pub last_compaction: i64` is the only timestamp on the manifest. No per-SST epoch exists today. Per `last_compaction` semantics at `manifest.rs:353-356`:

```rust
pub fn record_compaction(&mut self, ts: i64) {
    self.last_compaction = Some(ts);
}
```

A single global timestamp cannot express "this node-store SST is safe to drop tombstones because all bond-store SSTs at or below epoch E have been compacted past it."

### 5.2 Epoch type recommendation

Three candidates:

| Type | Pro | Con |
|---|---|---|
| Wall-clock i64 timestamp | Human-readable in logs | Clock-skew across daemon restarts; epoch comparison is fragile |
| Per-store monotonic counter | Independent stores; simple | Cross-store epoch coordination requires a separate translation |
| **Shared monotonic counter (one across both stores)** | Cross-store coordination is trivial (compare directly); resilient to clock skew | Need a single allocator on the manifest |

**Recommendation: shared monotonic counter on the manifest.** Add `pub next_epoch: u64` to `ManifestPayload` and `Manifest`, allocated like `next_sst_id`. Add `pub epoch: u64` to `SSTEntry` (both bond-store and node-store flavours). Each flush call allocates `next_epoch` and stamps it onto the new SSTEntry. Compaction-output SSTs inherit the MAX of their input SSTs' epochs (so a compacted SST advertises "every record in here is <= epoch E").

### 5.3 Per-store epoch on `SSTEntry`

Two practical paths:
- (i) **Single shared `SSTEntry` struct** with `epoch: u64` added (rkyv archive forward-compat -- `MANIFEST_SCHEMA_VERSION` bump to 2). Both bond-store and node-store SSTs use it.
- (ii) **Per-store `SSTEntry` variant** -- node store SSTEntry has `node_count` instead of `bond_count`, etc.

Recommendation: **(ii) per-store SSTEntry**. The bond_count vs node_count semantics differ enough that overloading a single field name causes diagnostic confusion. The node store's `SSTEntry` shape is:

```rust
pub struct NodeSSTEntry {
    pub id: u64,
    pub file: String,
    pub seed_min: [u8; 32],
    pub seed_max: [u8; 32],
    pub size: u64,
    pub mtime: i64,
    pub node_count: u64,
    pub epoch: u64,        // Q7 -- monotonic, shared with bond store's counter
}
```

Bond SSTEntry also gets `epoch: u64` (schema bump to 2 -- adding a u64 at the end is rkyv-forward-compatible).

### 5.4 Cross-store epoch barrier coordination

The node store's tombstoned NodeIds can be dropped only when no bond-store SST at the time of the tombstone could reference them. Concretely:

- Tombstone created at epoch E (i.e. the node-store SST holding the tombstone has `epoch = E`).
- Tombstone is safe to GC when every bond-store SST with `epoch <= E` has been compacted past epoch E (i.e. all bond-store SSTs at L0+ with epoch <= E have been promoted into compacted L1+ SSTs that themselves have `epoch >= E+1`).

Coordination primitive: `Manifest::oldest_live_bond_epoch() -> u64` returning `bond_levels.iter().flat_map(.ssts).map(|e| e.epoch).min().unwrap_or(0)`. Node-store compaction checks `tombstone_epoch < manifest.oldest_live_bond_epoch()` before dropping. Lock-free read against the manifest snapshot -- no `Arc<EpochTracker>` needed.

**Failure-mode tests (must be in Phase 0.c bench fixture before Phase 2.x ships):**
- Simultaneous node-store + bond-store compaction. Both should advance their epochs without deadlock or skipped barriers.
- Crash mid-compaction with partially-written epochs. Manifest atomic-save guarantees pre-or-post state; tombstones in flight are NOT GC'd until the post-compaction manifest commits.
- Daemon restart with tombstones from epoch E held in MemTable. After restart, the WAL replay reconstructs the MemTable state including tombstones at the original epoch, not the current epoch.

### 5.5 Per-store epoch interaction with NodeIntern

NodeIntern itself does NOT need epochs. It is append-only; a NodeId never recycles (Q1 contract: "no reuse on delete -- strictly append-only NodeId allocation"). A tombstone in the node store removes the *payload* (`KnowledgeNode`) for a NodeId; the NodeId-to-Seed mapping in NodeIntern stays forever. This keeps NodeIntern simple: no epoch GC, no tombstone-aware merging, just append-only.

## 6. Boot-replay arm rewrite (Phase 1.2 follow-up B)

### 6.1 Current state

`brain.rs:9941-9967` (REC_TYPE_NODE_INTERN replay arm in `Brain::enable_memtable`):

```rust
crate::wal::WalRecord::NodeIntern { seq, seed, node_id } => {
    // ... (long comment block explaining v4 NodeIds are not stable)
    let _ = seq;
    let _ = node_id;             // <-- discards stored node_id
    let _ = intern.intern_or_get(*seed);
}
```

The Phase 1.2 agent flagged this exactly: the stored `node_id` is dropped on the floor. The current arm calls `intern_or_get(seed)` which reassigns by insertion order, so post-restart NodeIds match the WAL's stored values only by coincidence (when replay order matches original insertion order, which it does today because WAL is append-only and replay is strict in-order -- but the contract is not made).

### 6.2 Recommended rewrite (Phase 2.1 or 2.2 depending on sequencing)

Two-case behaviour, both inside `enable_memtable` before the WAL replay loop:

**Case A -- sidecar present (post-Phase 2 brain):**
```rust
// Load sidecar BEFORE WAL replay.
let sidecar_path = brain_dir.join("nodeintern.bin");
if sidecar_path.exists() {
    let loaded = NodeIntern::load_from(&sidecar_path)?;
    intern = loaded;     // verbatim NodeIds restored
}
// Then WAL replay arm:
crate::wal::WalRecord::NodeIntern { seq, seed, node_id } => {
    let _ = seq;
    // Phase 2: stored node_id is contractual. Apply verbatim.
    intern.insert_with_id(*seed, *node_id)?;   // new NodeIntern API
}
```

The new `NodeIntern::insert_with_id(seed, NodeId)` method must:
- Reject if the seed is already mapped to a different NodeId (replay corruption -- log and abort load).
- Accept idempotent re-insertion (same seed, same node_id).
- Extend `reverse: Vec<Seed>` to length `node_id.0 + 1`, with the gap filled by `Seed::ZERO` sentinels (defensive -- the WAL ordering invariant means gaps shouldn't happen, but defensive padding survives a corrupted WAL with skipped records).
- Be `#[doc(hidden)]` and panic in non-replay contexts (so production code can't accidentally call it).

**Case B -- sidecar absent (pre-Phase 2 v4 brain -- migration window):**
- Log warning: `[brain] enable_memtable: nodeintern.bin sidecar absent; falling back to insertion-order NodeId assignment (pre-Phase 2 brain)`.
- Current replay arm behaviour (discard `node_id`, call `intern_or_get(seed)`). The existing comment block stays accurate.

### 6.3 When can Case B retire?

Per Section 4.4: leave Case B in indefinitely as defence-in-depth. The cost is one branch in the replay arm and one diagnostic log line. The benefit is graceful degradation if the sidecar file is corrupted, deleted, or missing from a hand-crafted test brain.

### 6.4 Phase 7 migration tool sidecar seed

`lu_brain_migrate_v3_v5` (Phase 7) must emit a `nodeintern.bin` sidecar as part of its output. The Phase 1.3 helper `NodeIntern::intern_from_v3_archive` already covers the **forward direction** (Seed -> NodeId assignment). The Phase 2 work adds `NodeIntern::save_to(path)` which the migration tool calls AFTER `intern_from_v3_archive` returns. **Verification gate (Q4): post-migration, sidecar record count must equal NodeIntern.len() must equal v3 node count.**

## 7. Crate structure recommendation (Q6 -- module vs crate)

### 7.1 Two candidates

- **(M) Module:** `src/node_store/{mod,memtable,flush,sst,sst_writer,compaction,sidecar}.rs`. Lives alongside `src/{memtable,wal,manifest,...}.rs` (the bond substrate).
- **(C) Separate crate:** `crates/lu_node_store/` with its own `Cargo.toml`, depends on `lu4d` for `Seed`, `KnowledgeNode`, `NodeIntern`.

### 7.2 Tradeoffs

| Factor | Module | Crate |
|---|---|---|
| Build incrementality | Whole-lu4d rebuild on any node-store change | Independent build; faster iteration during Phase 2.x |
| Dep graph | Simple -- `mod node_store;` in `lib.rs` | `lu4d` depends on `lu_node_store` depends on `lu4d` for types -> **circular**, requires hoisting `Seed`/`NodeIntern`/`KnowledgeNode` into yet another crate (`lu_brain_core` or similar) |
| Test isolation | Tests run with full lu4d compiled | Independent test crate, faster iteration |
| API surface discipline | `pub(crate)` is fine | Must explicitly `pub` what crosses the boundary -- forces API design discipline |
| Phase 0.b precedent | Bench crate is tooling | Bench crate is **test tooling** -- categorically different from production substrate |
| Phase 3 (bond store rewrite) | Bond store also moves into `src/bond_store/`; symmetry | Bond store moves into `crates/lu_bond_store/`; symmetric but doubles the crate-extraction work |

### 7.3 Recommendation: **(M) module -- `src/node_store/`**

Reasoning:
- The circular dep (lu_node_store <-> lu4d for Seed/NodeIntern/KnowledgeNode) is the killer. Resolving it requires extracting a `lu_brain_core` crate first, which is a Phase 0.5 / Phase 0.d port that doesn't exist and isn't blocking anything. Per `feedback_no_hacks`: don't carve out a Cargo workspace member to enable a refactor we haven't decided to do.
- The Phase 0.b bench crate precedent is a different category (test tooling, no production code path). Citing it for production substrate is the wrong precedent.
- Build-incrementality gain from crate extraction is real but small at Lu's current scale (lu4d is ~50K LOC; a clean recompile is <60s on Hetzner). Compile-time pressure isn't a top-3 friction.
- API-surface discipline can be enforced with module-level `pub(crate)` / `pub(super)`; doesn't require a crate boundary.

If after Phase 3 the bond store and node store have demonstrable cross-store API stability and the `lu_brain_core` extraction becomes worthwhile, do it then. Pattern matches `feedback_no_hacks`: refactor when the need is real, not when it's speculative.

### 7.4 Module layout

```
src/node_store/
    mod.rs             // pub re-exports, NodeStore facade
    memtable.rs        // NodeMemTable -- BTreeMap<NodeId, (KnowledgeNode, u64)>
    flush.rs           // NodeFlushWorker
    sst.rs             // NodeSST reader (block-encrypted, sort-by-NodeId)
    sst_writer.rs      // NodeSSTWriter (streaming sort-by-NodeId writer)
    compaction.rs      // NodeCompactionWorker (streaming k-way merge)
    sidecar.rs         // NodeIntern::save_to / load_from at nodeintern.bin
```

WAL stays at `src/wal.rs` (shared). NodeIntern stays at `src/node_intern.rs` (just gains save_to/load_from + insert_with_id). Manifest stays at `src/manifest.rs`; node-store SST levels live alongside bond-store SST levels via the schema-v2 `node_levels` field (see Section 5.3 + Q2.0.B below).

**Open question Q2.0.B (Section 11):** does node-store manifest live in its own file (`MANIFEST.nodes`) or as a `node_levels` field on the existing manifest? Recommendation: **same file, add `node_levels: Vec<LevelEntry<NodeSSTEntry>>` field via schema-version bump.** Two reasons: (i) atomic manifest commit covers both stores' metadata; (ii) cross-store epoch barrier reads from one manifest, not two. Resolving with John before Phase 2.1 spawns.

## 8. Migration story coordination (Q4)

### 8.1 What Phase 1.3 already covers

`NodeIntern::intern_from_v3_archive(archive)` at `src/node_intern.rs:215-225` iterates `archive.iter_bond_keys()` and calls `intern_or_get(from)` + `intern_or_get(to)` for each pair. This produces the `Seed -> NodeId` mapping. **This is the only half -- the forward index -- that Phase 1.3 covers.**

### 8.2 What's still needed for the node-store payload side

The v3 archive carries node payloads separately from bonds. Need:

- A helper on the migration side that iterates the v3 archive's node table and emits `(Seed, KnowledgeNode)` pairs.
- For each pair: `let node_id = intern.get(seed).expect("intern_from_v3_archive seeded this");` -- the NodeIntern is already populated from the bond walk, so this is pure lookup. (If the seed appears in node table but not in any bond, `intern_or_get` would fall through and assign a fresh ID -- this is correct behaviour but means `intern_from_v3_archive` should also iterate the node table for completeness; FOLLOW-UP-1 below.)
- Stream the `(NodeId, KnowledgeNode)` pairs to the node-store SST writer in NodeId order.

**The order matters:** sorting by NodeId means later compaction is cheap. Since NodeIds were assigned by `intern_from_v3_archive` in HashMap-iter order of `bond_index`, the migration tool must do an **external sort prep pass** for nodes too -- same constraint as Q4 revision for bonds. Both prep passes can share the external-sort scratch directory.

### 8.3 Verification gate additions

Per Q4 verification gate (random bond read-back + total count + entropy spot-check), add:

- Total node count match: `archive.node_count() == nodeintern.len() == node_store.total_node_count()`.
- Random node read-back: 1000 random Seeds from v3 -> `NodeIntern.get(seed) -> node_store.read(NodeId)` returns the byte-exact `KnowledgeNode` from v3.
- Sidecar integrity: `NodeIntern::load_from(written_sidecar)` round-trips to the same forward+reverse state.

### 8.4 Sidecar at migration output

`lu_brain_migrate_v3_v5` writes:
- `brain-v5/manifest`
- `brain-v5/wal/` (empty -- fresh WAL, no replay needed)
- `brain-v5/node_store/nodeintern.bin` (the sidecar)
- `brain-v5/node_store/sst/L0/*.sst` (node payloads)
- `brain-v5/bond_store/sst/L0/*.sst` (bond data)

All packaged into `brain.lbr` via the Q6 tar+zstd+encrypt pipeline at cutover time.

### 8.5 FOLLOW-UP-1

`NodeIntern::intern_from_v3_archive` currently only walks `iter_bond_keys()`. Any v3 node whose seed never appears as a bond endpoint (isolated node, query-meta nodes, etc.) will be missed. The Phase 7 migration tool must also walk the node table to intern those seeds, OR `intern_from_v3_archive` should be extended to iterate both. Recommendation: extend the helper as part of Phase 2.9 (rather than retro-fitting Phase 1.3). Low severity; affects orphan-node count only.

## 9. Surprises requiring Section 7 design-doc entries

### SURPRISE-A -- NodeIntern resident cost is 3-4x the design doc projection

Section 4.5 detail. **At 100M nodes: ~8-9 GB resident NOT 2.4 GB.** The 2.4 GB figure was correct for an idealised sidecar payload at the OLD u32 NodeIds. Actual resident footprint of `HashMap<Seed, NodeId(u64)>` + `Vec<Seed>` at 100M entries is materially larger.

**Recommended Section 7 entry:** "Phase 2.0 audit recomputes NodeIntern resident footprint from 2.4 GB -> ~8-9 GB at 100M nodes (HashMap overhead + u64 widening). Still inside AX41-NVMe 64 GB budget; Risk Register Section 5 threshold revised -- revisit before exceeding 25 GB resident (~ 250-300M nodes). Mitigations available if needed (shard, perfect-hash post-compaction)."

### SURPRISE-B -- Single shared WAL stays through v5, not two per-store WALs

Section 3.3 detail. The audit considered two-WAL split (one per store) and rejected it for Q8 invariant simplicity. **The design doc Section 2 Q3 phrasing "node store with its own MemTable + WAL + flush + compaction" implies a separate WAL, which is the wrong call.**

**Recommended Section 7 entry:** "Phase 2.0 audit refines Q3: node store gets its own MemTable + flush + SST + compaction, but **shares the existing WAL with bond store**. Two-WAL split rejected because (i) it would duplicate REC_TYPE_NODE_INTERN records and (ii) flush-truncate coordination becomes per-store rather than per-flush-watermark. Single shared WAL preserves the Q8 ordering invariant trivially."

### SURPRISE-C -- KnowledgeContent.kind is a TAG dimension, not a sort dimension

Section 2.2 detail. The 16-variant enum surfaced concern that compaction or sort order might need kind-awareness. It doesn't -- `kind` is data, `NodeId`/`Seed` is the sort key. Node creation is idempotent (`brain.rs:4019`); two records for the same seed are dedup'd by highest seq, regardless of kind.

**Recommended Section 7 entry (small):** "Phase 2.0 confirms KnowledgeContent.kind (16 variants) is a tag dimension. Node store sort order is NodeId only; compaction dedup is highest-seq-wins keyed on NodeId. No kind-specific handling at the substrate."

### SURPRISE-D -- Embedding dominates node payload (~85%)

Section 2.1 detail. Per-node ~1800 B is 1536 B embedding + 264 B everything else. Implications: node SST block compression won't help much (`Vec<f32>` is high-entropy after embedding training); cache-friendly NodeSST blocks should be sized to whole embeddings (e.g. 64 nodes/block = ~117 KB block including compression overhead). NOT a contradiction of any current design decision, but worth documenting.

**Recommended Section 7 entry:** "Phase 2.0 sizing: KnowledgeNode payload is ~1800 B per node, ~85% embedding (`Vec<f32; 384>`). Node store SST block size should target ~64-128 nodes/block, ~100-200 KB compressed. Embedding entropy means zstd compression saves <5% vs raw -- node store SST size budgeting should assume near-raw embedding storage."

### SURPRISE-E -- `enable_memtable` REC_TYPE_NODE arm independently rebuilds `network.nodes`

Section 2.3 + Section 6 detail. The existing REC_TYPE_NODE replay arm at `brain.rs:9924-9938` calls `self.network.add_node(node.clone())` -- populating the v3-vintage in-memory map. This is INDEPENDENT of the MemTable + sidecar story. Phase 2 must NOT retire this -- recall paths walk `network.nodes` as the hot lookup. The v4-substrate's REC_TYPE_NODE record stays as the canonical node-payload record; the sidecar covers only Seed<->NodeId binding.

**Not a contradiction of any Section 2 decision** but worth a clarifying Section 7 entry to avoid Phase 2.x agents accidentally pulling the rug:

"Phase 2.0 confirms REC_TYPE_NODE WAL record + `network.nodes` in-memory hot map remain through v5. Sidecar (`nodeintern.bin`) covers Seed<->NodeId binding ONLY, not node payloads. Node-store SSTs are the canonical disk-persisted payload; `network.nodes` is the hot RAM-resident lookup; REC_TYPE_NODE WAL is the durability path between flushes."

### SURPRISE-F -- `compaction.rs` Vec<MergeRow> antipattern applies to node store too

Section 3.1 detail. The Phase 0.a finding (`compaction.rs:357 rows: Vec<MergeRow>` materializes all bonds before sort/dedup) is a BIGGER memory hog at production scale than `node_winners`. The same antipattern is trivially recreated in any node-store compaction that collects `(NodeId, KnowledgeNode)` pairs into a Vec before merge. Phase 2.x compaction must use streaming k-way merge from the start -- confirmed in Section 3.1.

**Already captured** in design doc Section 7 (2026-06-08 Phase 0.a entry, Q7 revision). No new entry needed; flagged here for Phase 2.x implementer awareness.

## 10. Phase 2 sub-port sequencing recommendation

Bounded-RSS assertion target for every port: Phase 0.b's 5-10M-doc fixture, with `assert_bounded_rss!` invoked from Phase 0.c bench harness.

### Phase 2.1 -- Module scaffold + sidecar I/O

**Scope:**
- Create `src/node_store/{mod.rs, sidecar.rs}`. Stub `mod.rs` exporting `NodeStore` facade.
- Add `NodeIntern::save_to(path: &Path) -> Result<(), BrainError>` and `NodeIntern::load_from(path: &Path) -> Result<Self, BrainError>` to `src/node_intern.rs`.
- Sidecar format per Section 4.2: 16 B header + records `[seed:32, node_id: varint]` + CRC32 trailer.
- Add `NodeIntern::insert_with_id(seed, node_id) -> Result<(), NodeInternError>` for sidecar-driven verbatim restoration.

**Touch list:**
- `src/node_intern.rs` -- three new methods (~120 LOC).
- `src/node_store/mod.rs` -- facade stub (~30 LOC).
- `src/node_store/sidecar.rs` -- format + I/O (~150 LOC).
- `src/lib.rs` -- `pub mod node_store;`.
- Tests: round-trip 100K records, 10M records, corrupted sidecar -> load fails cleanly.

**Bounded-RSS:** sidecar write peak RSS <2x steady-state NodeIntern (BufWriter overhead only). Load peak RSS <1.5x steady-state.

**Estimated complexity:** ~300 LOC, ~4 hours.

**Dependencies:** Phase 1.x complete. No source-file collisions with the current main branch.

### Phase 2.2 -- Boot-replay rewrite (sidecar-aware)

**Scope:**
- Rewrite `Brain::enable_memtable` REC_TYPE_NODE_INTERN arm per Section 6.2. Sidecar present -> load + apply verbatim; sidecar absent -> keep current fallback with warning.
- Add `sidecar_path` parameter to `enable_memtable` (or derive it from the WAL directory).
- Tests: post-flush brain has sidecar -> restart preserves NodeIds byte-exact; deleted sidecar -> warning + fallback works.

**Touch list:**
- `src/brain.rs` -- `enable_memtable` arm rewrite (~40 LOC delta).
- Tests in `brain.rs` `#[cfg(test)] mod tests`.

**Bounded-RSS:** Replay peak RSS = sidecar load peak RSS + WAL replay delta. No HashMap-the-world.

**Estimated complexity:** ~80 LOC, ~3 hours.

**Dependencies:** Phase 2.1.

### Phase 2.3 -- NodeMemTable + flush-to-sidecar

**Scope:**
- `src/node_store/memtable.rs` -- `NodeMemTable` with `BTreeMap<NodeId, (KnowledgeNode, u64)>`. Inserts via `insert(node_id, node)` (intern is already up-to-date -- NodeIntern is shared). Swap-for-flush primitive.
- Refactor `MemTable::insert_node` / `lookup_node` / `iter_nodes_for_flush` / `node_count` to delegate to `NodeMemTable`. The existing v4 bond `MemTable` keeps the `entries` field for bonds; `nodes` field becomes a `NodeMemTable` reference.
- Sidecar flush wiring: at flush time, `NodeIntern::save_to(nodeintern.bin.tmp)` -> atomic rename. This is the FIRST save path that produces a sidecar -- until Phase 2.4 the node payloads still go into bond SSTs.

**Touch list:**
- `src/node_store/memtable.rs` -- new (~250 LOC).
- `src/memtable.rs` -- delegate node methods to NodeMemTable (~50 LOC delta).
- `src/flush.rs` -- call `NodeIntern::save_to` at end of `flush_once` (~20 LOC delta).
- Tests.

**Bounded-RSS:** flush peak RSS unchanged from v4 + sidecar write peak (~64 KB BufWriter).

**Estimated complexity:** ~400 LOC, ~6 hours.

**Dependencies:** Phase 2.1.

### Phase 2.4 -- NodeSST format + writer + reader

**Scope:**
- `src/node_store/sst_writer.rs` -- streaming writer accepting sorted `(NodeId, &KnowledgeNode)` pairs. Block size 64 nodes (~117 KB/block compressed). Block index keyed on `NodeId` min/max. Encryption pipeline reuses lu4d's `Encrypt` primitive.
- `src/node_store/sst.rs` -- reader. Block index binary search + block decrypt + linear scan within block.
- `NodeSSTEntry` struct (see Section 5.3). NOT yet wired to manifest -- Phase 2.5 does that.

**Touch list:**
- `src/node_store/sst_writer.rs` -- new (~300 LOC).
- `src/node_store/sst.rs` -- new (~250 LOC).
- Standalone tests (no Brain integration yet).

**Bounded-RSS:** writer peak RSS = 1 block buffer = ~120 KB. Reader peak RSS = 1 cached block.

**Estimated complexity:** ~600 LOC, ~10 hours.

**Dependencies:** Phase 2.1 (uses NodeIntern types).

### Phase 2.5 -- Manifest extension: node_levels + epoch field

**Scope:**
- Bump `MANIFEST_SCHEMA_VERSION` from 1 to 2.
- Add `pub epoch: u64` to existing `SSTEntry` (bond side).
- Add `pub node_levels: Vec<LevelEntry<NodeSSTEntry>>` to `ManifestPayload` (or split into a separate type if rkyv generic-over-T proves awkward).
- Add `pub next_epoch: u64` to `ManifestPayload` + `Manifest`.
- `Manifest::allocate_epoch() -> u64` method.
- `Manifest::oldest_live_bond_epoch() -> u64` and symmetric `oldest_live_node_epoch() -> u64`.
- Schema v1 -> v2 migration: load v1 manifests by treating missing `epoch` as 0 and `node_levels` as empty. v2-onward writes the new fields.

**Touch list:**
- `src/manifest.rs` -- schema bump + new fields + helpers (~200 LOC delta).
- Tests covering v1->v2 load, save+load v2.

**Bounded-RSS:** manifest size grows by ~24 B per SSTEntry + variable for node_levels. Atomic save unchanged shape.

**Estimated complexity:** ~250 LOC, ~5 hours.

**Dependencies:** Phase 2.4 (uses NodeSSTEntry).

### Phase 2.6 -- Node-store flush worker + routing

**Scope:**
- `src/node_store/flush.rs` -- `NodeFlushWorker` polls `NodeMemTable.is_full()` or scheduled cadence, swaps for flush, writes a NodeSST, allocates epoch via manifest, appends NodeSSTEntry to L0.
- Update `src/flush.rs` to STOP emitting node payloads in the bond SST. (`flush.rs:312-345` node_map + push_bond_with_nodes path is retired.)
- `Brain::enable_flush_worker` spawns BOTH the bond flush worker AND the node flush worker.

**Touch list:**
- `src/node_store/flush.rs` -- new (~250 LOC).
- `src/flush.rs` -- strip node payload routing (~80 LOC delta -- net negative LOC).
- `src/brain.rs` -- wire NodeFlushWorker (~50 LOC delta).
- Tests covering: node insert -> flush -> NodeSST read back -> bond SST does NOT contain node payload.

**Bounded-RSS:** node flush peak RSS = NodeMemTable cap + 1 block buffer. Independent of bond flush.

**Estimated complexity:** ~500 LOC, ~10 hours.

**Dependencies:** Phase 2.3 (NodeMemTable), Phase 2.4 (NodeSST), Phase 2.5 (manifest epoch).

### Phase 2.7 -- Node-store compaction + epoch barrier

**Scope:**
- `src/node_store/compaction.rs` -- streaming k-way merge of node SSTs. NO `node_winners` HashMap; NO `rows: Vec` materialisation. Tombstone-aware dedup keyed on NodeId; tombstone GC honors `oldest_live_bond_epoch` (Section 5.4).
- Epoch barrier coordination via `Manifest::oldest_live_bond_epoch()` read-only check.

**Touch list:**
- `src/node_store/compaction.rs` -- new (~400 LOC).
- Tests covering: simultaneous bond + node compaction; tombstone retained when bond epoch barrier blocks; tombstone GC'd when barrier advances.

**Bounded-RSS:** k-way merge peak RSS = (k input block buffers + 1 output block buffer). At k=8 inputs: <1 MB.

**Estimated complexity:** ~500 LOC, ~10 hours.

**Dependencies:** Phase 2.6.

### Phase 2.8 -- Read path integration

**Scope:**
- Update `Brain::get_node(seed)` to consult: (1) network.nodes hot map -> (2) NodeMemTable -> (3) NodeSST levels L0..LN. Falling-through resolution.
- Wire `NodeIntern.get(seed) -> NodeId` at boundary between (1)/(2) and (3).
- Q9 sanity check: K-gate + hub-fill + curiosity loop pass against the v5-backed Brain.

**Touch list:**
- `src/brain.rs` -- get_node refactor (~80 LOC delta).
- `src/network.rs` -- possibly small refactor if any path materializes node payloads from `network.nodes` instead of via the new fall-through.
- Integration tests against a 5M-record fixture.

**Bounded-RSS:** read peak RSS = 1 NodeSST block decrypt buffer + the existing network.nodes resident hot map.

**Estimated complexity:** ~200 LOC, ~5 hours.

**Dependencies:** Phase 2.6.

### Phase 2.9 -- Migration tool node-store output (Phase 7 prep)

**Scope:**
- `scripts/lu4d_migrate_v3_v5.rs` (or rename existing `lu4d_migrate_v3_to_v4.rs`) -- accepts v3 archive, emits node_store SSTs + sidecar + (Phase 3's) bond_store SSTs.
- Reuses `NodeIntern::intern_from_v3_archive` (Phase 1.3) + new `NodeIntern::save_to` (Phase 2.1) + new NodeSSTWriter (Phase 2.4).
- External sort prep for v3 archive's HashMap-iter unsortedness (Q4 revision -- applies to nodes too -- see Section 8.2).
- Extend `intern_from_v3_archive` to also iterate the v3 node table (FOLLOW-UP-1 from Section 8.5).
- Verification gate per Section 8.3.

**Touch list:**
- `scripts/lu4d_migrate_v3_v5.rs` -- substantial (~700 LOC).
- `src/node_intern.rs` -- extend `intern_from_v3_archive` to walk node table (~20 LOC delta).
- Bench harness extension to drive migration as a fixture build step.

**Bounded-RSS:** <2 GB per Q4 mandate.

**Estimated complexity:** ~700 LOC, ~16 hours.

**Dependencies:** Phase 2.4 + Phase 2.1. Independent of Phase 3 (bond store rewrite) -- can land before Phase 3 ships, just won't have a usable bond store to migrate INTO until Phase 3 is done.

### Phase 2 total LOC + complexity estimate

| Port | LOC | Est. hours |
|---|---|---|
| 2.1 sidecar I/O | 300 | 4 |
| 2.2 boot-replay rewrite | 80 | 3 |
| 2.3 NodeMemTable | 400 | 6 |
| 2.4 NodeSST format | 600 | 10 |
| 2.5 manifest epoch | 250 | 5 |
| 2.6 node flush worker | 500 | 10 |
| 2.7 node compaction + epoch barrier | 500 | 10 |
| 2.8 read path integration | 200 | 5 |
| 2.9 migration node-store output | 700 | 16 |
| **Total** | **~3530** | **~69** |

(Compare: Phase 1 widening + WAL + helper was ~250 LOC + ~6 hours; Phase 2 is materially bigger because it's a full per-store substrate, not a type widening.)

## 11. Open questions for John (Phase 2.0 -> Phase 2.1 gate)

**Q2.0.A -- Single shared WAL or two per-store WALs?** Recommendation: **single shared WAL** (Section 3.3, SURPRISE-B). Routes node-store and bond-store records into one append point; preserves Q8 ordering invariant trivially. Approve or override?

**Q2.0.B -- Manifest layout: extend existing manifest with `node_levels` field (schema v1 -> v2), or separate `MANIFEST.nodes` file?** Recommendation: **extend existing manifest**, schema v2 (Section 7.4 + Section 5.3). Atomic commit covers both stores; epoch barrier reads one manifest. Approve or override?

**Q2.0.C -- Node store as a module (`src/node_store/`) or a separate crate (`crates/lu_node_store/`)?** Recommendation: **module** (Section 7.3). Circular-dep cost of crate extraction > incremental build benefit at current scale. Approve or override?

**Q2.0.D -- Sidecar-absent fallback retirement.** Recommendation: leave indefinitely as defence-in-depth (Section 4.4). Cost ~20 LOC, benefit graceful degradation on sidecar corruption / hand-crafted test brains. Approve or override?

**Q2.0.E -- KnowledgeNode payload size projection.** Audit measures ~1800 B per node, dominated by embedding. SURPRISE-D notes block compression saves <5% on embedding entropy. Should NodeSST block compression be:
- (i) Still apply zstd (the existing bond SST pattern) -- ~5% gain on metadata, neutral on embedding;
- (ii) Skip compression entirely for node SST blocks -- saves CPU at flush/compact time;
- (iii) Compress metadata + raw embedding (split block layout)?

Recommendation: **(i) keep zstd, accept ~5% gain**. Simplest; matches bond SST pattern; CPU is not the bottleneck. Confirm or pick (ii)/(iii)?

**Q2.0.F -- Phase 2.x tag.** Recommend tag `pre-node-store-2026-06-09` on current HEAD `9c69684` BEFORE Phase 2.1 commit, so git-bisect crosses the node-store-introduction boundary cleanly. Approve?

**Q2.0.G -- Phase 2.9 migration tool naming.** Recommendation: rename existing `scripts/lu4d_migrate_v3_to_v4.rs` to `scripts/lu4d_migrate_v3_v5.rs` since v4 was never a cutover target (the audit retired v4 as production candidate). Or keep both as separate binaries until Phase 8 cutover completes? Confirm.

**Q2.0.H -- `intern_from_v3_archive` node-table extension.** FOLLOW-UP-1 (Section 8.5). Land in Phase 2.9 or as a Phase 1.x.cleanup follow-up that ships independently? Recommendation: Phase 2.9 since the only consumer is the migration tool. Confirm.

## 12. Method notes -- what I checked

- Hetzner reachability: `ssh lu hostname` -> the Lu server; HEAD `9c69684`.
- Source files audited per Section 1 scope; line numbers cited throughout.
- The 16-variant `KnowledgeContent` enum was inspected at source (`node.rs:46-79`), NOT inferred from the 6-name NodeType conceptual taxonomy (per `feedback_audit_nodetype_vs_kind`).
- Sizing arithmetic for SURPRISE-A is conservative HashMap+Vec accounting; real-world resident may be +-20% depending on allocator behaviour.
- No source files modified; no Cargo invocations run (this is read-only audit per the brief).
- No nested sub-agents spawned (L3-prohibition honoured per `feedback_no_subagent_delegation`).
- Existing uncommitted change to `scripts/lu4d_migrate_v3_to_v4.rs` (a +21/-1 diff that pre-existed this audit) was NOT swept into the docs commit (verified -- only the new doc file added).
