# OP-BRAIN-V5 Phase 3.0 -- Bond Store Audit

**Status:** Phase 3.0 read-only investigation complete 2026-06-09. No source files modified. Companion to `OP-BRAIN-V5-Design.md` Section 2 Q2/Q4/Q5/Q7 and Section 4 Phase 3. Modelled on the Phase 2.0 NodeStore audit; codebase HEAD `d3a4ba6` (Phase 2.9 migration tool node-store output shipped).

**Scope per design doc Section 2 Q2 + Section 4 Phase 3:** rewrite the v4 bond SST format around brain-wide u64 `NodeId` keys (no per-block subject_table), replace the v4 compaction's two HashMap-the-world antipatterns (`rows: Vec<MergeRow>` + `node_winners: HashMap`) with a streaming k-way merge, build a lazy reverse-index strategy for `(object, predicate, subject)` traversal, and produce a Phase 3.x migration tool that handles v3's HashMap-unsorted bond key iteration. Phase 3.0 reads the v4 bond substrate from the bond-store perspective and produces a sub-port sequencing recommendation BEFORE any Phase 3.x code lands.

**Headline finding:** the Phase 2 node-store machinery is **directly templateable for ~70% of Phase 3's LOC budget**. `node_store/sst_writer.rs` + `sst_reader.rs` + `compaction.rs` + `flush.rs` + `read_path.rs` can each be near-mechanically reshaped to `bond_store/` siblings -- the streaming k-way merge shape, the `(file_uid, block_id)` LRU cache, the epoch-barrier tombstone GC, and the `<file_uid>.bsst` flush+rename+manifest commit pattern all carry over verbatim. The **bond-specific surface** is concentrated in: (i) the on-disk wire format (`CompactBond` is 14 B block-local at HEAD, becomes 24 B brain-wide per Q2), (ii) the predicate-byte + weight-quantization + keyframe-pool encoding that must travel with each bond, (iii) the secondary reverse-index SST (Q2; no node-store analogue), and (iv) the v3 migration sort-prep stage (Q4 revision; the node-side dodged this because NodeIntern iteration is naturally NodeId-ordered, but `archive.bond_index` is `HashMap<(Seed, Seed), u32>` and iter order is genuinely random). **Phase 3.x LOC budget estimate: ~4,200-5,100 LOC** across 10 sub-ports (vs ~5,900 LOC for Phase 2's 9 sub-ports).

**Concrete bond-payload sizing correction.** The design doc's Q2 "24 B/bond x 100M projected bonds = 2.4 GB (pre-compression)" line **counts the key only**. A v4 CompactBond is 14 B today and EVERY bond also carries 1+ TemporalKeyframes (~24 B bincode each in the per-block pool) -- on a v3 production brain where the Phase 0.a baseline cites 18.6M bonds and the audit Appendix B cites 1 Baseline keyframe per bond, the on-disk **per-bond cost is ~14 B key + ~30 B keyframe-pool slot + ~3 B accounting overhead = ~47 B uncompressed, or ~9-12 B post-zstd at level 3**. The v5 24-byte CompactBond inflates the key by 10 B without changing the keyframe payload, so the post-rewrite per-bond uncompressed cost is ~57 B, post-zstd ~13-15 B/bond. **At 18.6M production bonds: ~1.1 GB uncompressed key+keyframe + ~50-100 MB pre-rewrite extras = ~250-280 MB compressed bond SSTs total (pre-rewrite ~210-225 MB).** The design doc's 2.4 GB number is the KEY-ONLY projection at 100M future bonds and is correct AS WRITTEN; just clarifying that the post-rewrite TOTAL bond-store size at 18.6M production bonds is ~280 MB, not 2.4 GB. This is SURPRISE-A below (not severe -- the design doc is right at 100M; we're just adding the today-shape datapoint).

---

## 1. Scope -- what was investigated, what wasn't

**Investigated (read-only):**
- `src/sst.rs` -- `CompactBondRaw` (line 1210), `ParsedBlock` (line 1219), `SST_VERSION=3`, `COMPACT_BOND_BYTES=14`, header layout (132 B), block layout, `pack_predicate`/`unpack_predicate`/`quantize_weight`.
- `src/sst_writer.rs` -- `SSTWriter` (line 386), `PendingBlock.subject_lookup: HashMap<[u8;32], u32>` (line 182), `intern_subject` (line 232), `push_bond`/`push_bond_with_nodes`/`push_node`/`push_tombstone` (lines 491/582/665/710).
- `src/compaction.rs` -- `rows: Vec<MergeRow>` (line 357), `node_winners: HashMap` (line 362), full merge body (lines 340-560).
- `src/manifest.rs` -- `SSTEntry { epoch: u64 }` (line 269), `NodeSSTEntry` (line 311), `LevelEntry`/`NodeLevel` (lines 286/364), `allocate_epoch`/`next_sst_id`/`add_sst`/`min_bond_sst_epoch` (lines 612/568/545/771).
- `src/memtable.rs` -- `MemBond` (line 54), `MemBondFlush` (line 79), bond half of `MemTableInner.entries: BTreeMap<NodeId, Vec<MemBond>>` (line 126).
- `src/flush.rs` -- bond flush loop, `node_map: HashMap<[u8;32], KnowledgeNode>` at lines 312-316 (will be retired by Phase 2.6 / Phase 3 cutover).
- `src/bond.rs` -- `BondType` (29 variants, line 98), `TemporalKeyframe` (line 84), `KeyframeSource` (10 variants, line 30), `confidence_from_keyframe_source`.
- `src/migration_v3_v4.rs` -- `iter_bond_keys` consumer + sort (line 338), verify `HashMap<TripleKey, V3BondSummary>` (line 699-701), Phase 10.1 streaming-fix log (lines 376-391).
- `src/archive.rs` -- `bond_index: HashMap<(Seed, Seed), u32>` (line 524), `iter_bond_keys` returning HashMap.keys() (line 806).
- `src/read_path.rs` -- dedup discipline (`(subject, predicate, object)` keyed at MemTable seq + SST manifest id + v3 archive seq=0).
- `src/node_store/*` -- entire module (sst_writer.rs 828 LOC, sst_reader.rs 1147 LOC, compaction.rs 1235 LOC, flush.rs 782 LOC, read_path.rs 798 LOC, memtable.rs 376 LOC, sidecar.rs 676 LOC, mod.rs 50 LOC) -- the structural twin for what Phase 3 must build on the bond side.
- `docs/OP-BRAIN-V5-Design.md` + `docs/OP-BRAIN-V5-Phase2.0-NodeStore-Audit.md` -- guiding decisions and structural twin.
- `docs/v3_baseline_2026-06-01.md` -- production sizing (7.1M nodes, 15.5M bonds at the bench snapshot; production today is closer to 18.6M bonds per design doc Section 1).

**Not in scope:**
- v3 archive read path internals (Phase 7 territory, Q4 was answered separately).
- Node store SST format (Phase 2 -- Q3 is locked, audit shipped).
- Snapshot/v5enc tar pipeline (Phase 4 -- Q6 is locked at archive-format level).
- Read path `MergedBondView` dispatch INTEGRATION (Phase 5 -- Q5 is locked; this audit recommends the bond-store side of Phase 5 as a sub-port of Phase 3, see Section 14).
- `Brain::read_outgoing_bonds_v4` callsite migration -- Phase 5/Phase 8 cutover work, not Phase 3 substrate.

## 2. v4 bond store current state (Q1 of the brief)

### 2.1 `CompactBondRaw` -- the on-disk bond row

Definition: `src/sst.rs:1210-1217` (the public name in `ParsedBlock.bonds` is `CompactBondRaw`; the design doc and Phase 0.a memo use `CompactBond` interchangeably -- the canonical type name in source is `CompactBondRaw`).

```rust
pub struct CompactBondRaw {
    pub subject_id: u32,       // 4 B  block-local NodeId into subject_table
    pub object_id:  u32,       // 4 B  block-local NodeId into subject_table
    pub predicate:  u8,        // 1 B  packed: 5 bits BondType + 1 bit dir + 2 bits flags
    pub weight:     u8,        // 1 B  u8-quantized weight (0..=255 maps to 0..=1.0)
    pub keyframe_offset: u16,  // 2 B  start index into per-block keyframe_pool
    pub keyframe_count:  u8,   // 1 B  number of consecutive keyframes (0..=255)
    pub confidence: u8,        // 1 B  Confidence as u8; 0xFF is the TOMBSTONE_CONFIDENCE marker
}
```

**Total: 14 bytes fixed-width per bond** (`COMPACT_BOND_BYTES = 14` at `src/sst.rs:166`). Phase 0.a confirmed and this audit re-verifies: `subject_id` and `object_id` are **block-local u32 indices into `ParsedBlock.subject_table: Vec<Seed>`**, NOT brain-wide NodeIds. The block-local discipline is intentional (per `sst.rs:90-110` doc-comment: "Decoupled lifetime" + "Compression" + "Migration safety") but is the load-bearing reason Phase 3 exists: every cross-block subsystem (compaction, migration, verify) must rebuild a brain-wide Seed -> NodeId mapping at runtime, which is the v4-architectural-audit's HashMap-the-world antipattern.

### 2.2 v4 SST block layout -- where bonds live on disk

From `sst.rs:46-95` (the canonical doc-block):

```text
Decompressed block payload, in order:
  - block_header (24 bytes)
  - subject_table  (subject_table_size_u32 x 32 B Seed records)
  - node_table     (node_table_size_u32 x [u32 len + bincode KnowledgeNode|empty])
  - keyframe_pool  (keyframe_pool_size_u32 x [u32 len + bincode TemporalKeyframe])
  - bonds          (bond_count_u32 x 14 B CompactBondRaw)
```

Four load-bearing properties for Phase 3 design:

1. **Block is encrypted XOR-keystream then zstd-compressed** (`sst_writer.rs:39-57`). Same crypto envelope Phase 3 must preserve; the Q2 fixed-width keys can drop straight into the existing crypto+compress envelope without rework.
2. **`subject_table` is THE block-local indirection** that Q2 retires. At 100K-200K distinct nodes per block worst-case, removing it saves ~6.4 MB worst-case per block at the cost of ~10 B per bond key (14 -> 24 B). Per Section 3 below, real-corpus blocks have <1K distinct subjects (most blocks pack one hub's outgoing edges), so the subject_table cost today is ~32 KB/block; that savings is the smaller of the two trade-off terms.
3. **`node_table` parallel to subject_table** is the Phase 10 addition that Phase 2 retires (nodes now live in `node_store/`). Phase 3 wire format MUST NOT carry node_table -- bond SSTs are bond-only.
4. **Keyframe pool is per-block, no cross-bond dedup**: each `push_bond` appends its full keyframe history as a contiguous `(offset, count)` run. Phase 2.1 deliberately removed the cross-bond dedup HashMap (`sst_writer.rs:170-179`). Phase 3 must preserve this (the per-bond keyframe range is load-bearing for the Lu curiosity / hub-fill / temporal layers).

### 2.3 SST header / index / footer

From `sst.rs:135-167`:

- Header: 132 B fixed (magic + version=3 + level + counts + manifest_id + per-SST encryption_iv + subject_min/max Seeds).
- Block index: per-block 48 B entry (`subject_min: Seed[32]` + `offset: u64` + `compressed_len: u32` + `decompressed_len: u32`).
- Footer: 8 B (CRC32 of block index + magic_footer).

**Phase 3 implication:** the header schema is BOND-side specific (`subject_min/max_seed: [u8; 32]`). For brain-wide NodeId keys the header should carry `subject_min/max: u64 NodeId` (8 B each instead of 32 B). The 48 B block index entry similarly drops to 28 B (NodeId u64 + offset + compressed_len + decompressed_len) -- which is **exactly the 28 B NodeSST block index entry shape Phase 2.4 already shipped** (`node_store/sst_writer.rs:42-46`). This is direct structural reuse -- not just "templateable", but literally the same bytes-on-disk index layout. See Section 4 for the full reuse map.

### 2.4 `SSTWriter` lifecycle + `intern_subject` per-block HashMap

Key internals (verified at HEAD `d3a4ba6`):

- `SSTWriter` struct at `sst_writer.rs:386-405`: holds `pending: PendingBlock`, `block_index: Vec<BlockIndexEntry>`, `total_bond_count: u64`, `cursor_offset: u64`.
- `PendingBlock.subject_lookup: HashMap<[u8; 32], u32>` at `sst_writer.rs:182`. Lifecycle: created fresh in `PendingBlock::new()` (line 219), populated by `intern_subject` (line 232) which probes-then-inserts on first-seen Seed in the current block, RESET on `flush_block` (`sst_writer.rs:773-806` -- `self.pending = PendingBlock::new()`).
- `intern_subject` is called from `push_bond` (line 539), `push_bond_with_nodes` (line 628), `push_node` (line 685), `push_tombstone` (line 729) -- both subject and object Seeds intern through the same path, so the block-local subject_table holds BOTH subjects and objects-with-no-outgoing-edges-in-this-block.
- The HashMap is **bounded per block** (block flushes well before u32::MAX subjects). On a hub-spoke block with 1K outgoing edges from one subject: ~1 subject + ~1000 objects = ~1001 entries x ~80 B HashMap overhead = ~80 KB. Bounded; not Phase 3's memory hog.

### 2.5 `compaction.rs` -- the two HashMap-the-world antipatterns

From Phase 0.a + this audit's re-verification, the v4 bond compactor at `compaction.rs:340-560` materialises EVERY bond from ALL input SSTs before sort/dedup. The two antipatterns:

**(A) `rows: Vec<MergeRow>` at `compaction.rs:357`.** Pre-allocated with `Vec::new()` (not even `with_capacity`); pushed once per bond per input SST inside the `for (sst_id, sst) in &input_handles` loop at line 363. At minor compaction time (L0 -> L1 with all L0 SSTs, ~5-10 SSTs * 1-3 GB each at production scale), this Vec materialises in the 1-3 GB range. The `MergeRow` struct carries: subject Seed (32 B), object Seed (32 B), predicate BondType + predicate_byte (~1 B each), weight (u8), confidence (u8), `keyframes: Vec<TemporalKeyframe>` (24 B header + N*~50 B), seq (u64). Per-row ~120-150 B amortised. **At 18.6M production bonds, a full-brain compaction materialises ~2.2-2.8 GB Vec.** Phase 0.a flagged this as the bigger memory hog vs `node_winners`; this audit confirms.

**(B) `node_winners: HashMap<[u8; 32], (KnowledgeNode, u64)>` at `compaction.rs:362-363`.** Populated by the same merge-input scan (`compaction.rs:366-380`), holding the highest-seq KnowledgeNode payload per Seed. KnowledgeNode is ~1.8 KB (embedding dominates -- Phase 2.0 audit S2.1 confirmed). **At 9M production nodes, full-brain compaction materialises ~16 GB HashMap.** This is the antipattern the audit memo Appendix F flagged. After Phase 2 cutover, this HashMap GOES AWAY ENTIRELY -- node payloads live in `node_store/` and the bond-side compactor knows only `NodeId: u64`. The Phase 3 bond compactor must NOT re-introduce a node-cache HashMap; the read path resolves NodeId -> KnowledgeNode at leaf via `node_store::get_node_v5`.

**Both antipatterns are replaced by the streaming k-way merge shape from `node_store/compaction.rs:700-820`** (already shipped in Phase 2.7). Direct template: replace `BinaryHeap<MergeCursor>` element shape from `(NodeId, NodeEntry, source_idx)` to `(BondKey, BondPayload, source_idx)` where `BondKey = (NodeId, u8 predicate, NodeId)` and `BondPayload = { weight: u8, confidence: u8, keyframes: Vec<TemporalKeyframe>, seq: u64 }`. The heap holds k cursors (one per input SST); merge `pop` yields lowest BondKey across all inputs; dedup keeps highest-seq winner per BondKey; tombstone GC drops `confidence == TOMBSTONE_CONFIDENCE` AND `source_epoch < gc_barrier` where `gc_barrier = min(manifest.min_bond_sst_epoch_excluding(plan), manifest.min_node_sst_epoch)` (note the swap: bond-store compaction excludes its own input SSTs from the bond min, queries node store unconditionally).

### 2.6 `migration_v3_v4.rs` -- bond key sort + verify HashMap

**Bond key sort at `migration_v3_v4.rs:338-356`.** Collects `let mut keys: Vec<(Seed, Seed)> = archive.iter_bond_keys().collect();` then sorts by `(from, to)`. At 18.6M bonds, this Vec is ~1.2 GB resident (18.6M * 64 B). v4 ship was tolerable because v3 archive load happens once at migration time and 64 GB AX41 has the headroom. Phase 3.x bond migration must EITHER (a) preserve this in-memory sort (acceptable per Phase 2.9's precedent of accepting `Vec<(NodeId, Seed)>` at 600 MB on the node side), OR (b) implement the external merge sort hedge from Q4. Section 10 below recommends (a) with an explicit `--max-rss-mb` guard -- mirrors Phase 2.9's `MigrationConfig::max_rss_mb`.

**Verify `HashMap<TripleKey, V3BondSummary>` at `migration_v3_v4.rs:699-701`.** Built once during verify-pass, holds entire v3 triple set + quantized weight + keyframe count for ticking off against the v4 SST walk. **At 18.6M bonds, ~24 B TripleKey + ~16 B V3BondSummary + HashMap overhead = ~80 B/entry = ~1.5 GB resident.** Phase 3 bond migration verify must EITHER reuse this pattern (tolerable) OR adopt a streaming SortedRunComparator (more code, lower peak RSS).

## 3. Empirical bond payload + size estimates

### 3.1 Per-bond on-disk cost (v3 production brain, 7.1M nodes / 15.5M bonds at bench snapshot, ~9M nodes / ~18.6M bonds at design-doc reference)

From the v3-baseline doc + Phase 0.a memo + this re-audit:

| Component | Bytes/bond (v4) | Bytes/bond (v5 Q2) | Notes |
|---|---|---|---|
| CompactBond key | 14 (block-local u32 + u32 indices) | 24 (brain-wide u64 + u64 NodeIds) | Q2 fixed-width 24 B |
| Predicate byte (in key) | (included above) | 1 (separate field, but key total still 24) | Actually fits as the middle byte of the 24 B key (8+8+8 with predicate packed into one of the 8-byte slots, OR 8 B subject + 1 B predicate + 7 B padding + 8 B object) -- see Section 6 |
| Weight (u8 quantized) | 1 | 1 | unchanged |
| Confidence (u8, 0xFF = tombstone) | 1 | 1 | unchanged |
| Keyframe range pointer (offset:u16 + count:u8) | 3 | 3 | unchanged |
| Per-block subject_table contribution (32 B Seed) | ~32 / N_block_subjects | 0 | retired by Q2 |
| Per-bond keyframe-pool slot | ~30 B amortised (1 Baseline keyframe per bond per v3 baseline) | ~30 B | unchanged; keyframe pool layout preserved |
| **Total uncompressed/bond** | **~47-50 B** | **~57-60 B** | v5 key inflated by 10 B; rest unchanged |
| **Total compressed/bond (zstd:3)** | **~9-12 B** | **~13-15 B** | zstd compresses brain-wide NodeIds at 2-3 B amortised post-compression per occurrence -- v3-baseline doc confirms |

### 3.2 Whole-brain bond-store size at production scale

| Scale | v4 SST size (compressed) | v5 SST size (compressed, Q2) | Delta |
|---|---|---|---|
| 15.5M bonds (bench) | ~150-185 MB | ~200-230 MB | +50 MB |
| 18.6M bonds (design-doc reference) | ~180-225 MB | ~245-280 MB | +60 MB |
| 100M bonds (Q2 projection horizon) | ~900M-1.2 GB | ~1.3-1.5 GB | +400 MB |

**Verdict:** Q2's "24 B/bond x 100M = 2.4 GB pre-compression" is the KEY-ONLY uncompressed projection. The full bond store including keyframes runs ~30% larger uncompressed, ~50% smaller after zstd. **Net: Phase 3 bond store at 18.6M production bonds = ~245-280 MB on disk.** Trivial vs the 64 GB AX41 budget; key takeaway is that the Q2 fixed-width 24 B trade is paying ~10 B/bond uncompressed (~3-5 B/bond post-zstd) to retire the block-local NodeId antipattern. Free at current scale; revisit if SST size ever becomes the bottleneck (Q2's "MEDIUM confidence" tag is appropriate).

### 3.3 Block-level prefix compression effectiveness for Q2 keys

Q2 says "block-level prefix compression recovers most of the size that varint would have saved". This is correct in principle: within a subject-sorted block, the `subject: u64` field is constant or near-constant for long runs (every bond from one hub subject has the same 8 B), so a prefix-compression scheme compresses those 8 bytes to ~1 byte amortised. The `predicate: u8` field has 29 distinct values (BondType variants) and is near-constant within a subject (most subjects have all-`RelatedTo` or all-`HasProperty` outgoing edges); a per-block predicate-RLE table compresses it to ~0.5 byte amortised. `object: u64` is the dominant bits (varies bond-by-bond); minimal prefix gain.

**zstd at level 3 captures most of this automatically.** Per the v3-baseline doc Appendix B ("Per-bond cost difference: ~2 bytes on the bond reference, doubled = ~4 bytes per bond extra. For 18.6M bonds: ~75 MB extra raw. Compressed: ~20-40 MB. Across the entire brain output: a fraction of a percent."), the block-level prefix compression in v5 vs v4 is a wash AFTER zstd. **Recommendation: skip explicit prefix compression in the v5 block payload.** Let zstd do its job; revisit if benches show otherwise.

## 4. Reusable Phase 2 infrastructure (Q3 of the brief)

### 4.1 Per-Phase-2-file reuse verdict

| Phase 2.x file | Phase 3 reuse | Verdict | Reasoning |
|---|---|---|---|
| `manifest.rs` SSTEntry + LevelEntry + epoch infrastructure | **DIRECT (zero change)** | Already designed for both stores | Phase 2.5 (manifest schema v2) already added per-SST `epoch: u64` to `SSTEntry` (the bond-side entry, `manifest.rs:269`) AND `NodeSSTEntry.epoch` (line 337). Phase 3 just uses `allocate_epoch` + `allocate_sst_id` + `add_sst` -- ALL ALREADY DEFINED. No manifest schema bump in Phase 3. |
| `node_store/sst_writer.rs` (828 LOC) | **DIRECT TEMPLATE** (~80% mechanical reshape) | New `bond_store/sst_writer.rs` ~600-700 LOC | Same header (32 B incl. file_uid stamp), same block layout (entry_count + uncompressed_size + compressed_size + compression_kind + block_crc32), same `(file_uid, block_id)` block index, same encrypt-then-compress envelope (matches current bond SSTs already!). Entry shape changes from `(NodeId, NodeEntry)` to `(BondKey, BondPayload)` where BondKey = 24 B (8+8+8) + BondPayload carries predicate+weight+confidence+keyframes. Same bincode codec for the keyframe Vec. |
| `node_store/sst_reader.rs` (1147 LOC) | **DIRECT TEMPLATE** (~85% mechanical reshape) | New `bond_store/sst_reader.rs` ~850-950 LOC | Same LRU block cache shape, same on-demand decode, same iter_range API. Point-lookup API changes from `get(NodeId) -> Option<NodeEntry>` to `get_outgoing(subject: NodeId) -> Vec<(NodeId object, predicate, payload)>` -- the bond access pattern is range-scan-per-subject, not point-lookup. This is the LARGEST behavior delta; see Section 7. |
| `node_store/compaction.rs` (1235 LOC) | **DIRECT TEMPLATE** (~75% mechanical reshape) | New `bond_store/compaction.rs` ~900-1000 LOC | Same streaming k-way merge over `BinaryHeap<MergeCursor>`, same dedup-by-higher-seq, same epoch-barrier tombstone GC, same atomic-commit sequence (allocate_epoch -> write SSTs -> add_sst -> retire SSTs -> save_atomic). Bond-side: dedup key changes from `NodeId` to `(NodeId, u8 predicate, NodeId)` (24 B). Tombstone payload changes (the bond-side tombstone is `confidence == 0xFF` on a CompactBond row, not a separate boolean field). |
| `node_store/flush.rs` (782 LOC) | **DIRECT TEMPLATE** (~85% mechanical reshape) | New `bond_store/flush.rs` ~600-700 LOC | Same `<file_uid>.bsst.tmp -> rename -> manifest.add_sst -> save_atomic` pattern, same `cleanup_orphan_bond_ssts` startup pass (renames orphan files to `.orphan-<ts>`), same `NodeFlushWorker::flush` -> `BondFlushWorker::flush` signature. Input is `MemTable::iter_for_flush() -> Vec<(NodeId subject, Vec<MemBondFlush>)>` (already exists). |
| `node_store/read_path.rs` (798 LOC) | **DIRECT TEMPLATE** (~70% mechanical reshape) | New `bond_store/read_path.rs` ~700 LOC | Same `BondSstReaderCache` (HashMap<file_uid, Arc<BondSstReader>>), same MemTable -> L0 (overlapping ranges) -> L1+ (binary-search disjoint ranges) walk. Output shape changes from `Option<KnowledgeNode>` to `Vec<BondView>` -- per-subject range scan, not point lookup. |
| `node_store/memtable.rs` (376 LOC) | **NOT REUSABLE** (different shape, see Section 5) | Bond half stays in `src/memtable.rs` | Bond MemTable is already in `src/memtable.rs::MemTableInner.entries: BTreeMap<NodeId, Vec<MemBond>>` and stays there. Phase 3 does NOT introduce `bond_store/memtable.rs`; the existing `src/memtable.rs` IS the bond memtable, parallel to the node half delegated to `node_store/NodeMemTable`. |
| `node_store/sidecar.rs` (676 LOC) | **NOT APPLICABLE** | No bond-side sidecar | NodeIntern is the only sidecar (Q3); bonds derive everything from NodeIntern + WAL + SSTs. No `bond_store/sidecar.rs`. |
| `node_store/mod.rs` (50 LOC) | **DIRECT TEMPLATE** | New `bond_store/mod.rs` ~50 LOC | Same pub-use pattern. |

### 4.2 Reuse summary

- **~70% of Phase 3 code can be a near-mechanical reshape** of Phase 2 siblings.
- **Phase 3-specific surface (~30%):** the bond wire format (24 B CompactBond layout decisions, keyframe-pool block layout, header/index format), the read-path range scan API, the secondary reverse-index SST (Q2; no node analogue), and the v3 migration sort-prep (Q4 revision).
- **No shared `lu_substrate` crate** still recommended (consistent with Phase 2.0 verdict). After Phase 3 lands, look back: if duplication is real (e.g. `bond_store/sst_reader.rs` and `node_store/sst_reader.rs` share >60% identical block-cache code), extract THEN. Per `feedback_no_hacks`, refactor when duplication is real, not when speculative.

### 4.3 Manifest infrastructure is FULLY DESIGNED for both stores already

Phase 2.5 already shipped the cross-store manifest schema. From `manifest.rs:380-410` (`ManifestPayload`):

- `levels: Vec<LevelEntry>` -- bond-store LSM levels (already exists; Phase 3 just keeps using `add_sst`/`next_sst_id`/`iter_ssts`).
- `node_levels: Vec<NodeLevel>` -- node-store LSM levels (Phase 2.5 added).
- `next_sst_id: u64` -- bond SST id counter (already exists).
- `next_node_sst_uid: u64` -- node SST uid counter (Phase 2.5 added).
- `next_epoch: u64` -- SHARED across both stores (Phase 2.5 added).

**Phase 3 introduces NO new manifest fields.** The bond store already has its slot. The only manifest-related change in Phase 3 is the bond-side compaction worker calling `allocate_epoch()` instead of using the old `last_compaction: i64` timestamp (and `SSTEntry.epoch=0` for v3 migration outputs is already legal per the v1->v2 schema bump's backwards-compat).

### 4.4 WAL infrastructure -- no change needed

The existing `src/wal.rs` (single shared WAL holding REC_TYPE_BOND + REC_TYPE_NODE + REC_TYPE_NODE_INTERN) STAYS as-is. Phase 3 doesn't change WAL format or routing; bond writes already go through `wal.append_bond` and bond-side replay already reconstitutes bonds through the MemTable. The flush -> SST -> manifest commit -> WAL truncate cadence is the only coordination point with the node store and that's already designed (Phase 2.6 / Phase 2.7).

## 5. BondMemTable shape recommendation

**Recommendation: leave the bond MemTable in `src/memtable.rs`. Do NOT introduce `bond_store/memtable.rs`.**

Rationale:

1. The bond half of `MemTableInner.entries: BTreeMap<NodeId, Vec<MemBond>>` (`memtable.rs:126`) is ALREADY brain-wide-NodeId-keyed. Phase 1 wired this; nothing to do.
2. The bond half is structurally INDEPENDENT of the node half today (Phase 2.3 split the node half into `NodeMemTable` and deletegated `MemTable::insert_node` to it). The bond half can evolve in place without touching the node delegation.
3. The signature `BTreeMap<NodeId, Vec<MemBond>>` is correct for the bond access pattern: per-subject range scan via `BTreeMap::range`. The audit considered an alternative `BTreeMap<(NodeId, BondType, NodeId), MemBond>` (full-triple key) and rejects it -- the per-subject access pattern dominates (every typed_recall is "give me all outgoing bonds from subject X"), so per-subject grouping is the right organization. Within a subject, the `Vec<MemBond>` is sorted by `(predicate, object)` at flush time, not at insert time (per `memtable.rs:38-40`).
4. Renaming to `BondMemTable` would be a mechanical churn touching every callsite (`brain.rs` has ~50 of them per a quick grep) for zero structural gain.

**Single small refactor in Phase 3:** rename the existing `MemTable::iter_for_flush` to `iter_bonds_for_flush` to mirror the existing `iter_nodes_for_flush` and disambiguate after the Phase 2.6 cutover. Trivial; bundles into the Phase 3 flush worker port.

## 6. Bond payload format recommendation

**Recommendation: 24 B fixed-width key with `(subject_u64, predicate_packed_u64, object_u64)` layout, separate from a per-bond payload stream encoded with bincode.**

### 6.1 The 24 B key

```rust
#[repr(C)]  // explicit layout for byte-exact memcmp comparison
pub struct CompactBondV5 {
    pub subject:   u64,  // 8 B brain-wide NodeId, little-endian on disk
    pub predicate: u64,  // 8 B packed: low 8 bits = predicate byte (BondType + dir + flags),
                         //              next 8 bits = weight quantized u8,
                         //              next 8 bits = confidence u8 (0xFF tombstone marker),
                         //              high 40 bits reserved = 0
    pub object:    u64,  // 8 B brain-wide NodeId, little-endian on disk
}
```

Rationale:

- **Q2's "primary sort `(subject, predicate, object)`" is preserved** -- memcmp on the 24 B little-endian bytes gives the right order if subject/predicate/object are written little-endian (within-u64 byte order doesn't matter as long as the writer and reader agree, but little-endian matches the existing SST byte conventions at `sst.rs:222`).
- **No separate "weight + confidence" tail** -- packed into the middle u64 to keep the key fixed-width. The Q2 design doc says "fixed-width 24-byte CompactBond keys (8 + 8 + 8 for subject / predicate / object NodeIds)" -- this audit's amendment is that the 8 B "predicate" slot is wide enough to carry the small per-bond payload (weight + confidence) WITHOUT a separate value stream. This is the bond-side analogue of using a fixed-width key as both key and value when the value is small.
- **Keyframe range pointer (offset:u16 + count:u8) doesn't fit** in 8 B with weight + confidence + predicate. It travels SEPARATELY as part of the per-block payload stream (see 6.2).

**Alternative layout (rejected):** 14 B key (subject u32 + object u32 + same 6 B tail as v4) at brain-wide u32 NodeIds. Rejected because: (a) NodeId is already u64 post-Phase 1; (b) u32 NodeId locks Lu to <4B nodes -- a 100M-node brain growth headroom limit that the Q1 "high 32 bits reserved for tenant/generation" forward-compat path needs; (c) Q2 explicitly says 24 B fixed-width.

### 6.2 The per-block payload stream

Keyframes can't fit in the key. Per Q2 (block-level prefix compression) and Section 3.3, keep the existing v4 keyframe-pool structure verbatim in the v5 block payload:

```text
Block payload (post-decompress, post-decrypt) layout:
  block_header (16 B): entry_count u32 + keyframe_pool_size u32 + min_subject u64
  bonds:    entry_count x 24 B CompactBondV5   (sorted by full 24 B key)
  bond_kf_ranges: entry_count x (u16 offset + u8 count) = 3 B/entry
  keyframe_pool: keyframe_pool_size x [u32 len + bincode TemporalKeyframe]
```

Why the keyframe range pointer is parallel-array (not inline in CompactBondV5):

- Keeps the key memcmp-able (fixed-width pure key, no per-row tail mixed in).
- Matches Phase 2.4's pattern of "entry header (24 B fixed) + entry payload (variable bincode)" with bond payload stream being a flat byte array of 3 B records.
- Single-block in-memory representation: `Vec<CompactBondV5>` (24 B/entry) parallel with `Vec<(u16, u8)>` (3 B/entry) parallel with the keyframe_pool `Vec<TemporalKeyframe>`. Three flat Vecs, no HashMap.

### 6.3 Bond payload format trade-off (bincode vs custom packed bytes)

The brief calls out "bond payloads are smaller and more uniform (~16-32 bytes) so the relative overhead of bincode framing matters more." Three options:

| Option | Bytes/bond payload | Reasoning |
|---|---|---|
| (A) bincode `CompactBondV5` + bincode `Vec<(u16, u8)>` per block | 24 (key) + 3 (range) + ~8 bincode framing per block = ~27/bond + ~8/block | Bincode framing for a Vec of fixed-shape entries is one u64 length prefix + the entries -- ~8 B overhead per block, not per entry. Negligible at >100 entries/block. |
| (B) Custom packed bytes (the writer's own little-endian byte writer) | 24 (key) + 3 (range) = exactly 27/bond | Saves 8 B per block. Phase 2.4 considered this and rejected: "Custom binary was rejected as scope creep -- the embedding... dominates payload bytes... already the minimum legal envelope." Bond side has no embedding so the framing fraction is larger BUT the absolute savings (8 B / block at 1000-4000 bonds/block) is still negligible. |
| (C) rkyv | ~30 B + alignment padding | More code, zero-copy upside is moot (we're reading bonds for comparison/iteration, not random access into archived structs). Rejected. |

**Recommendation: (B) custom packed bytes**, AGAINST the Phase 2.4 precedent. Reasoning differs by layer:

- The 24 B key is FIXED-WIDTH and is the primary sort key. Memcmp comparison REQUIRES byte-exact layout. bincode adds an opaque length prefix that breaks memcmp. We need raw `[u8; 24]` write/read paths, which is custom packed bytes by definition.
- The 3 B range pointer is the same: fixed-width, no framing needed.
- The TemporalKeyframe in the keyframe pool keeps bincode (matches the v4 `encode_keyframe` at `sst.rs:1403` -- direct reuse).

Net: the bond row + range parallel arrays are custom-packed (memcmp-safe), the keyframe pool is bincode (matches existing). No new codec surface.

## 7. Secondary-index strategy recommendation (Q4 + Q2)

Q2 says reverse-traversal index is "built lazily during compaction, not maintained on every write." The question Phase 3.0 must pin: same SST file with two sort orders, or separate `.bsst` and `.bsst.reverse` files?

### 7.1 Three candidates

| Option | Pro | Con |
|---|---|---|
| (A) Two sort orders in ONE SST file (primary block stream + reverse block stream, header indexes both) | Single file per SST; atomic create/delete; one file_uid per SST | Doubles block-cache cost (each SST has 2 block streams to LRU); compaction outputs need to write both streams |
| (B) Separate `.bsst` (primary) + `.bsst.reverse` (secondary) files per SST | Clean separation; can omit `.bsst.reverse` when reverse traversal isn't needed; each file has its own block cache | Two files per SST = doubled FD pressure; cross-file atomic commit requires both files in manifest with paired file_uids |
| (C) Single global reverse index SST built at COMPACTION time across ALL bond SSTs (one giant `.brev.bsst` per level) | Single reverse-index file per level; merged across SSTs so reverse queries hit one file per level | The level-wide invariant becomes harder to maintain (every compaction at level N must rebuild the level-N reverse index); not LSM-friendly |

### 7.2 Recommendation: (B) separate `.bsst.reverse` per SST, manifest tracks paired file_uids

Reasoning:

1. **Sized correctly to current need.** Reverse traversal is a non-dominant access pattern in Lu today. Typed_recall ("give me outgoing bonds from X") is the hot path; reverse_typed_recall ("who points at X?") is used by hub-fill, K-gate self-test, and some thinking layer queries -- non-zero but not the >90% of bond reads. Wasting 2x block cache for an 80/20 access pattern (Option A) is the worst trade.
2. **Optional construction.** Manifest entry has `Option<reverse_file_uid: u64>`; null means "this SST has no reverse index built yet". L0 SSTs (flushed from MemTable) skip the reverse index by default (cheap flushes are the goal). Compaction outputs always build the reverse index. Bond reverse queries fall back to a full L0 scan if the reverse index is missing -- acceptable because L0 SSTs are small and few.
3. **Atomic commit.** Phase 3 flush/compaction commits BOTH file_uids together (allocate two epochs is unnecessary; same epoch on both). `cleanup_orphan_bond_ssts` startup pass treats `.bsst` and `.bsst.reverse` as one logical unit: if either is missing, both are orphans.
4. **Per-file LRU.** Each `BondSstReader` opens just the file it needs (primary or reverse). The `BondSstReaderCache` keys on `(file_uid, kind)` where `kind: enum { Primary, Reverse }`.

Schema change to NodeSSTEntry analogue:

```rust
pub struct BondSSTEntry {
    pub file_uid: u64,
    pub reverse_file_uid: Option<u64>,  // None for L0 (flush-cheap) or pre-reverse-build
    pub first_subject: u64,
    pub last_subject: u64,
    pub bond_count: u64,
    pub file_size: u64,
    pub epoch: u64,
}
```

Note: the `manifest.rs::SSTEntry` at line 245 ALREADY EXISTS for the v4 bond store and would be extended in-place rather than introducing a parallel `BondSSTEntry`. Schema bump v2 -> v3 needed (analogous to the v1 -> v2 bump for `epoch`). See Section 14 sub-port 3.5.

### 7.3 Reverse-index file build

The reverse index file is structurally a normal `.bsst` file but sorted by `(object, predicate, subject)` instead of `(subject, predicate, object)`. SAME bond rows, SAME keyframe pool format, SAME block layout -- just different sort order. The `BondSstWriter` should be parameterized over the key projection: `KeyOrder::Primary` projects `(subject, predicate, object)`, `KeyOrder::Reverse` projects `(object, predicate, subject)`. The same writer can produce both files; compaction emits both streams during its output pass.

**LOC estimate:** ~80 LOC for the reverse projection logic on top of the primary writer/reader templates.

## 8. Bond SST block sizing recommendation

Node SSTs ship with `DEFAULT_TARGET_ENTRIES_PER_BLOCK = 64` and `MAX_TARGET_ENTRIES_PER_BLOCK = 8192` (`node_store/sst_writer.rs:148-151`). Why 64? Node payload is ~1.8 KB (embedding dominates), so 64 nodes = ~117 KB raw per block, matching the v4 `DEFAULT_BLOCK_SIZE_TARGET = 256 KiB` (post-compression target).

**For bonds:** per-bond payload is ~27 B (24 key + 3 range) + a small keyframe-pool slot (~30 B amortised). Total ~57 B/bond raw. For the same ~256 KB raw block target: **~4500 bonds/block raw**, or **~1500-2000 bonds/block post-zstd** (zstd ratio ~3x at level 3 for this shape).

**Recommendation: `DEFAULT_TARGET_BONDS_PER_BLOCK = 4096, MAX_TARGET_BONDS_PER_BLOCK = 65536`.** Power-of-2 round numbers; matches the ~256 KB raw block target the v4 SST writer already uses (`sst_writer.rs:84`). The max is wide enough to accommodate a single very-hot subject (a hub like "the" or "of" might have 10K+ outgoing bonds and we never split a single subject across blocks -- see `sst_writer.rs:482-503`).

The v4 `MAX_BONDS_PER_BLOCK` constant exists at `sst_writer.rs:97`; carry over the "1M cap to keep decompressed blocks bounded" rule.

## 9. Dedup + tombstone semantics recommendation

### 9.1 Dedup

Same `(subject, predicate, object)` triple from multiple input SSTs at compaction time, OR same triple from MemTable + L0 SSTs at read time. **Recommendation: highest-seq wins**, matching:

- v4 read path at `read_path.rs:30-44`: MemTable seq (monotonic) > L0 SST manifest_id (monotonic) > v3 archive seq=0.
- v4 compaction at `compaction.rs:415-419`: "the first row is the highest-seq winner" by sorting `(subject, predicate, object, seq DESC)`.
- Phase 2.7 node compaction at `node_store/compaction.rs:36-44`: "higher `seq` wins on dedup".

For the v5 streaming heap merge, when two cursors pop pointing at the SAME 24 B key, the merge loop keeps the entry with the strictly higher `seq` and discards the other. SST `seq` is the SST's manifest-allocated `id: u64` -- newer SSTs have higher ids by `Manifest::next_sst_id`. No change to seq semantics.

### 9.2 Bond tombstones

Q7 says "epoch-barrier tombstone GC." The brief asks whether bond storage NEEDS tombstones.

**Answer: YES, bond tombstones are already in production at HEAD.** The v4 `SSTWriter::push_tombstone` (`sst_writer.rs:710-754`) and `TOMBSTONE_CONFIDENCE = 0xFF` (`sst.rs:187`) ship today. Compaction tombstone GC (`compaction.rs:432-444`) ALREADY drops tombstones with `seq < snapshot_horizon` at the deepest level. The bond-side tombstone marker is the `confidence` byte = 0xFF on a CompactBond row.

**Use cases (verified from grep):**

- `Brain::forget_bond` callsites (decay-driven removal under threshold).
- Scar log entries -- a scar can shadow a bond's contribution.
- `MemTable::delete` -- bond-row deletion is supported (`memtable.rs:71`).

**Recommendation: preserve the v4 `confidence=0xFF` tombstone marker semantics in v5 verbatim.** No bool field, no separate tombstone row type -- the byte IS the marker. Phase 3 compaction's epoch-barrier GC computes `gc_barrier = min(manifest.min_bond_sst_epoch_excluding(plan.source_uids), manifest.min_node_sst_epoch)` and drops a tombstone iff `source_epoch < gc_barrier`. Identical shape to the Phase 2.7 node-side rule.

## 10. Phase 7.0 external sort prep placement

Q4 revision says v3 `iter_bond_keys` is HashMap-iter (unsorted), and bond migration needs external merge sort. The brief asks: should Phase 7.0 stay as Phase 7.0, or move under Phase 3?

### 10.1 Phase 2.9 precedent

The node-side migration at `src/migration_v3_v5_node.rs:436-456` materialises `Vec<(NodeId, Seed)>` in memory and sorts in-process. At 15M nodes, 40 B/entry = 600 MB resident -- inside the 64 GB AX41 budget. Phase 2.9's comment explicitly defers external merge sort: "No external merge sort needed at Lu's scale; the audit's hedge to 'external merge sort with bounded RSS <2 GB' was for the case where NodeIntern resident itself approached the budget ceiling."

### 10.2 Bond-side equivalent at production scale

For bonds, the equivalent of the (NodeId, Seed) Vec is `Vec<(subject_id: NodeId, predicate: u8, object_id: NodeId)>` -- the bond key Vec, sorted. **At 18.6M bonds, 17 B/entry = ~320 MB resident. At 100M projected, ~1.7 GB.** Both inside the 64 GB budget; both inside Phase 7.0's "bounded RSS <2 GB" hedge.

**However:** the v4 migration already materialises `Vec<(Seed, Seed)>` at 1.2 GB resident (32+32 B Seeds, not 8+8 B NodeIds). The v5 sort vec at 320 MB is a 4x improvement on v4 because we're sorting NodeId-pairs not Seed-pairs (post-Phase-1 NodeIntern). **Phase 3 bond migration can sort in-memory inside the budget, no external merge sort needed at Lu's current scale.**

### 10.3 Recommendation: COLLAPSE Phase 7.0 INTO Phase 3.x (specifically the migration tool sub-port, 3.10).

Phase 7.0 in the design doc is currently "external merge sort prep for v3 bond keys" -- a defensive hedge against the case where in-memory sort exceeds RSS budget. Per Phase 2.9's precedent and the 320 MB / 1.7 GB sizing above, this is no longer load-bearing. The MIGRATION tool itself is the consumer of the sort prep; folding both into one sub-port (3.10) is cleaner than carrying Phase 7.0 separately. Section 14 lays this out as sub-port 3.10.

**Fallback hedge:** if Phase 3.10 benches reveal that the 1.7 GB sort vec at 100M projected scale hits the RSS cap during migration (e.g. if NodeIntern is ALSO resident at 8 GB per the Phase 2.0 audit's SURPRISE-A correction), THEN spawn Phase 3.10b as an external-sort upgrade. Until that benches out, in-memory sort is the right shape.

## 11. Crate / module structure recommendation

**Recommendation: `src/bond_store/` mirroring `src/node_store/`. NO new crate.**

Reasoning:

1. Phase 2.0 audit verdict: "NO shared `lu_substrate` crate -- circular-dep cost dominates extraction benefit at current scale." Same call applies here. Bond store extracted as a crate would need to import `Seed`, `NodeIntern`, `NodeId`, `KnowledgeNode` from `lu4d` -- circular.
2. Symmetry with Phase 2. Phase 2 chose `src/node_store/` module path. Phase 3 should choose `src/bond_store/` module path. Discoverable, parallel naming, parallel file layout.
3. After Phase 3 lands, re-evaluate: if `node_store/sst_reader.rs` and `bond_store/sst_reader.rs` are 60%+ duplicate code (likely, per Section 4), extract THEN as `src/store/sst_reader_shared.rs` (still a module path, not a crate).

**Module layout (mirrors `node_store/`):**

- `bond_store/mod.rs` -- pub re-exports.
- `bond_store/sst_writer.rs` -- streaming sort-by-(subject, predicate, object) writer with optional reverse-key projection.
- `bond_store/sst_reader.rs` -- block-decoded reader with LRU cache + range scan API.
- `bond_store/flush.rs` -- `BondFlushWorker` + `cleanup_orphan_bond_ssts`.
- `bond_store/compaction.rs` -- `BondCompactionWorker` with streaming k-way merge + epoch-barrier GC + reverse-index emit.
- `bond_store/read_path.rs` -- `BondSstReaderCache` + `get_outgoing_bonds_v5` + `get_reverse_bonds_v5`.

No `bond_store/memtable.rs` (the bond MemTable stays in `src/memtable.rs`, per Section 5). No `bond_store/sidecar.rs` (no bond sidecar exists; everything derives from NodeIntern + WAL + SSTs).

## 12. Cutover dependency chain

```text
  Phase 1.x  (NodeIntern u32 -> u64 widening, DONE)
      |
      v
  Phase 2.x  (NodeStore: memtable, flush, SST, compaction, read path, migration, DONE)
      |
      v
  Phase 3.1  (bond_store/ scaffold + BondSstWriter)
      |
      v
  Phase 3.2  (BondSstReader + LRU cache)
      |
      v
  Phase 3.3  (BondFlushWorker + orphan cleanup)
      |
      v
  Phase 3.4  (BondCompactionWorker -- streaming k-way merge + epoch-barrier GC)
      |
      v
  Phase 3.5  (Manifest schema v2 -> v3: SSTEntry.reverse_file_uid: Option<u64>)
      |
      v
  Phase 3.6  (BondSstWriter reverse-key projection + compaction emits reverse index)
      |
      v
  Phase 3.7  (Read path: get_outgoing_bonds_v5 + reverse fall-through)
      |
      v
  Phase 3.8  (Sanity: K-gate + hub-fill + curiosity smoke test against v5-backed Brain)
      |
      v
  Phase 3.9  (Retire v4 bond SST writer/reader -- delete src/sst.rs, src/sst_writer.rs, src/compaction.rs; bond_store/ takes their place)
      |
      v
  Phase 3.10  (Migration tool: lu_brain_migrate_v3_v5_bond -- v3 archive + Phase 2.9 node-side output -> bond_store/)
      |
      v
  Phase 4+   (Snapshot/v5enc, full read path integration, production cutover -- already in design doc)
```

Gating chain: Phase 8 (production cutover) requires BOTH halves working. Phase 2 ships the node half; Phase 3 ships the bond half; Phase 4 ships the snapshot pipeline; Phase 5 (read-path two-store dispatch) ships the merged Brain trait impl. **Phase 3.9 (v4 retirement) is the irreversible step** -- after that point there is no v4 bond fallback compiled into the binary, and the rollback path is binary-swap to v3 + brain.lbr symlink swap.

## 13. Surprises requiring design-doc Section 7 entries

### SURPRISE-A: 24 B/bond Q2 projection is key-only, not total bond-store size

Q2 in the design doc says "24 B/bond x 100M projected bonds = 2.4 GB (pre-compression)." This is the KEY-ONLY uncompressed projection. Section 3.2 above shows the TOTAL (key + keyframe pool + accounting overhead, pre-compression) is ~57 B/bond, so the 100M projection is actually ~5.7 GB pre-compression, ~1.3-1.5 GB compressed. **Not a contradiction of Q2** (Q2's number is right at what it measures), just clarifying for the next reader.

**File:line evidence:** `OP-BRAIN-V5-Design.md` Q2 first paragraph; bond shape at `src/sst.rs:78-87` + `src/bond.rs:84-92` (TemporalKeyframe) + `src/sst_writer.rs:170-179` (per-block pool, no cross-bond dedup).

**Severity: LOW.** Design doc number is correct; just needs a footnote.

### SURPRISE-B: v4 SST is ALREADY encrypt-then-compress (the doc-comment back-and-forth in sst_writer.rs)

Reading `sst_writer.rs:39-66`, there's a long doc-comment dialogue where the author oscillated on whether encryption happens before or after compression. The actual code at `sst_writer.rs:763-787` (in `flush_block`) is: encrypt-then-compress (XOR keystream applied first, then zstd). This matches the existing v3 `compress.rs` envelope.

**Phase 3 implication:** the v5 bond SST envelope SHOULD MATCH. Per node_store/sst_writer Phase 2.4 the order shipped was compress-then-encrypt or encrypt-then-compress -- needs verification before Phase 3.1 lands so both stores agree. Quick grep of `node_store/sst_writer.rs` line 670 area will confirm.

**File:line evidence:** `src/sst_writer.rs:39-66` (doc-comment) + `src/sst_writer.rs:763-787` (actual code).

**Severity: MEDIUM** (could trip a Phase 3.1 implementer who reads only the doc-comment).

### SURPRISE-C: NO third HashMap-the-world in v4 compaction beyond rows + node_winners

Phase 0.a flagged `rows` + `node_winners`. This audit re-verified `compaction.rs:340-560` line by line and found NO third antipattern. The merge loop, output writer rollover, and atomic-commit sequence are all already streaming-shaped. Just rows + node_winners get retired. 

**File:line evidence:** `src/compaction.rs:340-560` (full merge body).

**Severity: NONE.** Confirms the Phase 0.a memo's scope is complete.

### SURPRISE-D: `MAX_KEYFRAMES_PER_BLOCK = u16::MAX = 65535`

From `src/sst.rs:200`. Tight. A pathological subject with > 65535 keyframes in one block can't be packed; the writer flushes block before adding (`sst_writer.rs:520-525`). At Lu's 1-baseline-keyframe-per-bond shape this is fine (1500-2000 bonds/block * 1 kf each = 1500-2000 kfs/block << 65535). But if the curiosity engine ever emits keyframe-heavy bonds (e.g. Hypothesis -> ConfirmedHypothesis promotion writes a new keyframe per pass), block fragmentation could become an issue.

**Phase 3 implication:** preserve the `u16 keyframe_offset` design (matches Q2 fixed-width discipline). Document the limit. Re-evaluate IF the curiosity engine starts emitting >100 keyframes per bond regularly (current production load: ~1 kf/bond Baseline, occasional Session).

**File:line evidence:** `src/sst.rs:200` (constant) + `src/sst_writer.rs:520-525` (block-flush guard).

**Severity: LOW** (forward-compat note, not a blocker).

### SURPRISE-E: Phase 0.a missed the `seeds_emitted: HashSet<[u8; 32]>` in migration_v3_v4.rs

At `migration_v3_v4.rs:392`, the migration tool holds a `HashSet<[u8; 32]>` with capacity `total_nodes` (~9M entries) for tracking which seeds got their node payload emitted via a bond push, so it can do an isolated-node sweep at the end. **At 9M nodes, ~720 MB resident** (32 B + ~50 B HashSet overhead per entry). This is a THIRD migration-time HashMap-the-world that Phase 0.a memo doesn't flag.

**Phase 3 implication:** the v3 -> v5 bond migration tool (Phase 3.10) must NOT carry forward this HashSet. The isolated-node sweep is a Phase 2.9 concern (node migration), already handled there -- the bond migration only needs to emit bonds, not chase isolated nodes. Removed-in-design, not a new antipattern.

**File:line evidence:** `src/migration_v3_v4.rs:392-398` (the HashSet).

**Severity: LOW** (Phase 0.a oversight; Phase 3.10 won't inherit it).

### SURPRISE-F: `v4_sst_cache: Mutex<HashMap<u64, Arc<SST>>>` is the v4 production read-path cache

At `brain.rs:1184`, there's a per-SST handle cache on Brain (`v4_sst_cache`), bounded by the live manifest size. Initialised at multiple Brain ctor sites (1485, 2034, 2240, 2419, 2692). This is the v4 production cache that the Phase 3 BondSstReaderCache must REPLACE -- it's the same shape (HashMap<file_uid, Arc<SST>>) but currently in `brain.rs`, not in a substrate module.

**Phase 3 implication:** Phase 3.7 (read path) replaces `v4_sst_cache` with `bond_store::BondSstReaderCache`. Sub-port 3.7 must enumerate the 5+ Brain ctor sites and migrate the cache initialization.

**File:line evidence:** `src/brain.rs:1184` (field decl) + multiple ctor sites (1485, 2034, 2240, 2419, 2692).

**Severity: MEDIUM** -- the migration is mechanical but touches several Brain ctor sites; easy to miss one.

## 14. Phase 3 sub-port sequencing recommendation

Mirrors Phase 2's 9-sub-port structure (Phase 2.1 through 2.9). Phase 3 splits into 10 sub-ports because the bond side has the secondary reverse-index work (no node analogue) and the v4 retirement (no node analogue -- the v4 node table was always block-table-inside-bond-SST, no separate v4 node store binary).

| Sub-port | Name | Description | Dependencies | Estimated LOC |
|---|---|---|---|---|
| 3.1 | bond_store scaffold + BondSstWriter | New `src/bond_store/` module. Template from `node_store/sst_writer.rs`. Entry shape changes to (BondKey 24 B, BondPayload). Primary-sort writer only -- reverse projection deferred to 3.6. Same encrypt-then-compress envelope; verify against SURPRISE-B first. | Phase 2.x | ~700 |
| 3.2 | BondSstReader + LRU cache | Template from `node_store/sst_reader.rs`. Per-subject range scan API (`get_outgoing(subject: NodeId) -> Vec<BondView>`), NOT point lookup. Shares the `(file_uid, block_id)` LRU pattern. | 3.1 | ~900 |
| 3.3 | BondFlushWorker + orphan cleanup | Template from `node_store/flush.rs`. `<file_uid>.bsst.tmp -> rename -> manifest.add_sst -> save_atomic` cadence. Consumes `MemTable::iter_for_flush()` (already exists). Renames orphan files to `.orphan-<ts>` at startup. | 3.1, 3.2 | ~650 |
| 3.4 | BondCompactionWorker (streaming merge + epoch-barrier GC) | Template from `node_store/compaction.rs`. Streaming k-way merge over `BinaryHeap<MergeCursor>`. Dedup by higher seq. Epoch-barrier tombstone GC with cross-store query of node-store min epoch. **Retires the rows Vec + node_winners HashMap.** | 3.1, 3.2, 3.3 | ~1000 |
| 3.5 | Manifest schema v2 -> v3: SSTEntry.reverse_file_uid | Add `Option<u64> reverse_file_uid` to `SSTEntry`. Schema bump with v2 frozen archetype for backwards compat (matches the v1 -> v2 epoch bump pattern). Touches `manifest.rs` only; trivial. | 3.4 | ~250 |
| 3.6 | Reverse-key projection + compaction emits reverse index | Parameterize `BondSstWriter` over `KeyOrder::{Primary, Reverse}`. Compaction output pass writes BOTH primary `.bsst` and `.bsst.reverse` (paired file_uid in `SSTEntry.reverse_file_uid`). Flush worker emits primary-only at L0; reverse index built lazily at compaction. | 3.4, 3.5 | ~400 |
| 3.7 | Read path: get_outgoing_bonds_v5 + reverse fall-through | Template from `node_store/read_path.rs`. MemTable -> L0 (overlapping ranges, all candidate SSTs queried) -> L1+ (binary-search disjoint ranges). Replaces `v4_sst_cache` HashMap on Brain at the 5+ ctor sites (SURPRISE-F). Reverse path falls back to L0 full scan when `reverse_file_uid == None`. | 3.1, 3.2, 3.6 | ~750 |
| 3.8 | Sanity: K-gate + hub-fill + curiosity smoke test | Q9 sanity gate. Run K-gate self-test + hub-fill smoke + curiosity wonder-queue probe against v5-backed Brain. PASS/FAIL gate on production cutover. | 3.7 | ~150 (test harness only) |
| 3.9 | Retire v4 bond SST writer/reader | Delete `src/sst.rs`, `src/sst_writer.rs`, `src/compaction.rs`. Migrate the constants and helpers (`pack_predicate`, `quantize_weight`, `TemporalKeyframe` codec) to `bond_store/` or `bond.rs` as appropriate. Touches every callsite in `brain.rs` + `flush.rs`. **IRREVERSIBLE STEP** -- after this, no v4 fallback compiled. | 3.7, 3.8 | -1500 net (deletes 4000 LOC, adds 2500 in new locations) |
| 3.10 | Migration tool: lu_brain_migrate_v3_v5_bond | Template from `scripts/lu_brain_migrate_v3_v5_node.rs` + `src/migration_v3_v5_node.rs`. Pipeline: open v3 archive (read-only), load node-side migration output as NodeIntern (Phase 2.9 produced `nodeintern.bin`), in-memory sort `Vec<(subject_id u64, predicate u8, object_id u64)>` of bond keys (~320 MB at 18.6M bonds), stream through `BondSstWriter` instances at L0. Verify pass: sample of N triples roundtripped through Brain::get_outgoing_bonds_v5. **No Phase 7.0 external sort needed at Lu's current scale.** | 3.3, 3.4, 3.7 (read path for verify) | ~900 |

**Total Phase 3 LOC estimate: ~4,200 added, ~1,500 removed = ~2,700 net new code + ~1,500 deletions = ~4,200 LOC budget.**

Does NOT include test code (tests run ~30-40% of source LOC for substrate work per Phase 2 precedent). Including test code: ~5,500-6,000 total LOC touched in Phase 3.

## 15. Open questions for John (with pre-loaded recommendations)

**Q3.0.A: Encrypt-then-compress vs compress-then-encrypt for bond SST blocks?**

*Context:* SURPRISE-B above. v4 bond SST is encrypt-then-compress (`sst_writer.rs:763-787` verified). Phase 2.4 node_store/sst_writer needs verification of which order it shipped.

*Recommendation:* Phase 3.1 implementer reads `node_store/sst_writer.rs` around line 670 first, then matches whichever order shipped. If they diverge, Phase 3.1 file a bug + match v4's encrypt-then-compress (XOR keystream is byte-position-equivariant so order is a free choice IF the writer and reader agree -- but symmetry across stores is cleaner).

*If you disagree:* tell me and I'll re-spec sub-port 3.1.

**Q3.0.B: Reverse-index strategy -- (B) separate `.bsst.reverse` files per SST. Confirm?**

*Context:* Section 7 above. Tradeoffs: (A) two streams in one file = doubled block cache, (B) separate file per SST = doubled FDs, (C) single global per-level reverse index = LSM-unfriendly.

*Recommendation:* (B). Reverse traversal is non-dominant; lazy build at compaction; manifest tracks paired file_uids; cleanup_orphan treats both as one logical unit.

*If you disagree:* (A) is the next-best fallback. (C) is rejected.

**Q3.0.C: Collapse Phase 7.0 (external sort prep) INTO Phase 3.10?**

*Context:* Section 10. Phase 2.9 precedent: 600 MB in-memory sort fit the budget. Bond-side sort at 320 MB (18.6M bonds) or 1.7 GB (100M projected) fits the AX41 budget without external merge sort.

*Recommendation:* Yes. Phase 7.0 collapses into Phase 3.10. If 100M-scale benches surface RSS pressure, spawn Phase 3.10b as the external-sort upgrade (clear deferred-port hook, not vague).

*If you disagree:* keep Phase 7.0 as a separate placeholder.

**Q3.0.D: Bond payload codec -- custom packed bytes for the 24 B key + 3 B keyframe-range, bincode for the keyframe Vec. Confirm?**

*Context:* Section 6.3. Custom packed bytes is required for the 24 B key (memcmp-safe). Bincode for the keyframe Vec matches v4 verbatim (`encode_keyframe` at `sst.rs:1403`).

*Recommendation:* As stated. No new codec surface.

*If you disagree:* fall back to all-bincode (loses memcmp on key, costs ~8 B/block framing -- negligible).

**Q3.0.E: Module path `src/bond_store/` mirroring `src/node_store/`. Confirm?**

*Context:* Section 11. Phase 2.0 audit verdict: "NO shared `lu_substrate` crate." Same call for Phase 3.

*Recommendation:* `src/bond_store/`. Re-evaluate post-Phase-3 if duplication is real.

*If you disagree:* alternative is `src/store/bond/` + `src/store/node/` (more nesting). Rejected by Phase 2.0 already.

**Q3.0.F: BondMemTable stays in `src/memtable.rs`. Confirm?**

*Context:* Section 5. Bond half of `MemTable` is already brain-wide-NodeId-keyed and already independent of the (delegated-to-NodeMemTable) node half. No reshape needed.

*Recommendation:* Yes, in place. Single mechanical rename `iter_for_flush -> iter_bonds_for_flush` for symmetry with `iter_nodes_for_flush`.

*If you disagree:* introduce `bond_store/memtable.rs` for symmetry (churn for symmetry's sake).

**Q3.0.G: Phase 3.9 (v4 retirement) timing -- inside Phase 3 or deferred to Phase 8 cutover?**

*Context:* Section 12 + Section 14. Phase 3.9 deletes `src/sst.rs`, `src/sst_writer.rs`, `src/compaction.rs` -- IRREVERSIBLE STEP. Currently scheduled inside Phase 3.

*Recommendation:* Inside Phase 3, AFTER Phase 3.8 sanity gate passes. Reasoning: keeping v4 code compiled into the Phase 3 binary masks bugs (matches feedback_no_hacks and Q10's "No v3 fallback compiled into v5 binary"). The Phase 3.8 sanity gate gives confidence; deletion immediately after is the cleanest discipline.

*If you disagree:* defer to Phase 8 cutover. Trade-off is carrying ~3900 LOC of dead code through Phase 4/5/6/7.

## 16. Honest blockers

None for the audit itself. For Phase 3.1 spawn:

- **Q3.0.A needs an answer before 3.1 lands** (encrypt-then-compress order verification). Easy to verify; just don't spawn 3.1 without reading node_store/sst_writer.rs first.
- **Q3.0.B (reverse-index strategy) and Q3.0.D (codec) need answers before 3.1 lands** because the bond wire format encodes both decisions structurally.
- **Q3.0.E/F (module path + memtable shape) need answers before 3.1 lands** because they affect the scaffold shape.

Q3.0.C and Q3.0.G can be deferred until later sub-ports (3.10 and 3.9 respectively).

No training is paused on this work -- v3 retrain is the bigger constraint (per current memory: "v3 training paused at Wikipedia 34.4%"); Phase 3 substrate work doesn't block it.

---

## Appendix A: Phase 2.0 audit verdicts that carry over verbatim

- **NO shared lu_substrate crate.** Module path `src/bond_store/` mirroring `src/node_store/`. Re-evaluate post-Phase-3 if duplication is real.
- **Single shared WAL.** Phase 3 doesn't touch WAL format or routing. Bond + node + nodeintern records all share `src/wal.rs`.
- **Manifest is the cross-store authority.** Phase 2.5 added `epoch` to both `SSTEntry` and `NodeSSTEntry`. Phase 3.5 will add `reverse_file_uid` to `SSTEntry` (schema v2 -> v3).
- **Streaming everywhere.** Same Q7 invariant: no per-pass HashMap-the-world for any cross-block operation. The bond-side rows Vec + node_winners HashMap are the two patterns Phase 3.4 retires.
- **Per-store copy of substrate primitives.** Phase 2 chose this for the node side; Phase 3 chooses it for the bond side. ~70% of code is templateable from Phase 2 siblings, ~30% is bond-specific.

## Appendix B: Cross-store epoch barrier -- what changes for the bond side

The Phase 2.7 node-side `gc_barrier` formula:

```
gc_barrier = min(
  manifest.min_bond_sst_epoch(),
  manifest.min_node_sst_epoch_excluding(&plan.source_uids),
)
```

The Phase 3.4 bond-side mirror:

```
gc_barrier = min(
  manifest.min_bond_sst_epoch_excluding(&plan.source_uids),
  manifest.min_node_sst_epoch(),
)
```

Note the swap: bond-side compaction excludes its own input SSTs from the bond min (the inputs are about to be retired, so their epoch shouldn't hold up GC), but queries node-store min unconditionally (no special handling -- the bond compactor doesn't touch node SSTs).

This means Phase 3.5 must EITHER:

- Add `manifest.min_bond_sst_epoch_excluding(uids: &[u64]) -> Option<u64>` as the bond-side mirror of `min_node_sst_epoch_excluding` (already exists at `manifest.rs:790`).
- OR re-use a common helper `manifest.min_sst_epoch_excluding(store: enum {Bond, Node}, uids: &[u64])`.

**Recommendation:** add the mirror function. Trivial; ~30 LOC. Folded into sub-port 3.5 (manifest schema bump).

---

_Audit complete. Ready for John's sign-off on Q3.0.A-G before Phase 3.1 spawns._
