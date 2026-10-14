> **SUPERSEDED** — see /GLOSSARY.md (regenerated 2026-10-13). This file is retained for history only.

# Lu Glossary — named parts of the extractor stack and ingest path

Last updated: 2026-05-31, post text-extractor v2 deploy (binary md5 `f3b4da4e2c79fe2a5cfad9ec9db397bc`).

Per the glossary discipline (memory: `feedback_glossary_discipline.md`), this file tracks the named parts of Lu's text-extraction stack — both the Python legacy still running in curriculum-ingest and the new Rust in-process extractor — so future contributors avoid name collisions or behavior drift between them.

When extending the extractor, **read this file first**, then update it as part of the same port.

---

## Two extractors exist (parallel, not yet merged)

### Python (legacy, currently running)

Process: `lu4d-curriculum-ingest.service` spawns `python3 -u scripts/ingest_curriculum_full.py --seed-trust --tier 1` which in turn spawns `lu_4_mcp` as a subprocess and JSON-RPC-talks to it.

Key files:
- `scripts/smart_extractor.py` — L1/L2/L2b/L3/L4/L5 layer functions. Top-level entry: `extract_bonds_from_wikipedia_article(article, l4_max_emit=80)`.
- `scripts/ingest_curriculum_full.py` — orchestrator + the LM-sentence / LM-anchor extractors:
  - `extract_response_bond(prompt, response)` — single-sentence Pattern A/B → `(subj, rel, obj, weight, conf)` tuple. Returns None if no bond.
  - `extract_response_bonds_multi_sentence(title, text, source_url)` — multi-sentence Path 1 (Pattern A/B subject) + Path 2 (pronoun-led, title-anchored).
  - `_truncate_to_np(text, max_len=60)` — NP truncator added 2026-05-30 to plug sentence-fragment hallucinations. Cuts at first punctuation / subordinator / verb-led / oversize.
  - `extract_wikipedia_bonds(title, text, source_url)` — lead-paragraph extractor (currently emits only the IsA bond; the legacy `(title, RelatedTo, lead[:200])` emission was deleted 2026-05-30 — see Port "Phase-1 NP truncation").

Bond emit: JSON-RPC `lu4d_teach_corpus_batch` (one batch per chunk, up to ~1000 triples). Goes through `teach_with_source` (scar gate + source attribution overlay).

### Rust (new, deployed 2026-05-31, NOT YET wired into curriculum-ingest)

Location: `src/extractor/` module (~3300 LOC across 18 files).

Layer files (one per layer):
- `src/extractor/layers/l1_see_also.rs`
- `src/extractor/layers/l2_section_headings.rs` (with L2-noise filter via `shared_noise::reference_noise()`)
- `src/extractor/layers/l2b_category_tail.rs` (tightened — last-60-non-empty-lines + contiguous-heading-run-of-3+ + reference-noise drop)
- `src/extractor/layers/l3_lead_patterns.rs` (15 typed patterns)
- `src/extractor/layers/l4_body_comentions.rs`
- `src/extractor/layers/l5_numerical_attrs.rs` (8 patterns)
- `src/extractor/layers/lm_multi_sentence.rs` (Path 1 + Path 2)
- `src/extractor/layers/shared_noise.rs` — `reference_noise()` set, shared between L2 and L2b

Support files:
- `src/extractor/mod.rs` — public API: `extract_from_article(text, title?, source_url?) -> Vec<Bond>`
- `src/extractor/types.rs` — `Bond` struct, `KindTag` enum mapping `BondType` → kind
- `src/extractor/labels.rs` — `clean_label`, `ok_term`, `trim_object`, `truncate_to_np`
- `src/extractor/patterns.rs` — all regex constants (uses `regex` crate for simple patterns + `fancy-regex` for lookahead-requiring L3 patterns)
- `src/extractor/sentence.rs` — `split_sentences`, `split_paragraphs`, `first_n_paragraphs`
- `src/extractor/aliases.rs` — `build_subject_aliases`, `sentence_anchors_subject`
- `src/extractor/tail_strip.rs` — `strip_tail_sections` (removes See Also / References / External Links / Bibliography / Further Reading before L4/L5)
- `src/extractor/dispatch.rs` — top-level `extract_from_article` that runs all 8 layers and dedups

MCP wrapper:
- `src/mcp_extractor.rs::tool_learn_text` — exposes `lu4d_learn_text(text, title?, source_url?, corpus)`. Calls extractor, applies `keep_bond` knowledge-aware filter against `brain.network.keyword_index`, pushes survivors to `brain.pending` via `push_pending_with_filter` (T.3 PendingLog approval gate preserved).

---

## Layer tags (both Python and Rust emit the same)

| Layer | Tag(s) | Relation | Weight | Subject anchored? |
|---|---|---|---|---|
| L1 | `L1` | RelatedTo | 0.55 | yes (title) |
| L2 | `L2` | HasAspect | 0.5 | yes (title) |
| L2b | `L2b` | IsA | 0.5 | yes (title) |
| L3 | `L3-isa-prefix`, `L3-loc`, `L3-part`, `L3-foundby`, `L3-foundin`, `L3-discby`, `L3-named`, `L3-meas`, `L3-born`, `L3-died`, `L3-writby`, `L3-comp`, `L3-known`, `L3-used`, `L3-type` | 15 typed (IsA, LocatedIn, PartOf, BornIn, DiedIn, FoundedBy, FoundedIn, DiscoveredBy, NamedAfter, WrittenBy, ComposedOf→PartOf, KnownFor, UsedFor, MeasuredIn) | 0.5-0.65 | yes — via `sentence_anchors_subject` |
| L4 | `L4` | RelatedTo | 0.4-0.55 (freq-weighted) | no (object-driven; frequency≥2 threshold) |
| L5 | `L5-foundyear`, `L5-pop`, `L5-area`, `L5-alt`, `L5-lang`, `L5-cap`, `L5-borndt`, `L5-dieddt` | FoundedIn, HasPopulation, HasArea, HasAltitude, HasOfficialLanguage, HasCapital, BornOn, DiedOn | 0.55-0.6 | yes — subject-anchored sentence-level |
| LM-sent | `L-multi-sentence` | typed (via `_relation_for_pred`) | 0.4 | yes (subject from Pattern A/B) |
| LM-anchor | `L-multi-anchor` | typed | 0.4 | yes (title anchor on pronoun-led) |

---

## Bond emission rules (both langs share)

- `clean_label(s)` first: strip whitespace, citation markers `[1]`, normalize internal whitespace.
- `ok_term(s, min, max)`: length gate + stop-word + alpha-ratio (`alpha >= max(2, len/2)`). This is why L5 FoundedIn on bare years never fires — year strings have 0 alphas.
- `trim_object(s)`: drop trailing dangling words (and/or/but/which/who/where/that/in/on/of/at/by/for/from/to/with/the).
- `truncate_to_np(s, max_len=60)` (added 2026-05-30 — Python and Rust both have it): cuts at first comma/semicolon/colon/em-dash, then at first subordinating conjunction (because, although, while, that, which, who, where, when, if, as, since, though, so that, in order to), hard caps at 60 chars, rejects if remaining text contains finite predicate verb or starts with past-participle / gerund.

## Shared constants (must stay in sync between Python and Rust)

- `_RELATION_TO_KIND` / `relation_to_kind` — 24 relations → KnowledgeContent kind tags (definition / composition / purpose / fact / concept). Mirrors `infer_kind_from_relation` in `src/node.rs`.
- `_L2B_REFERENCE_NOISE` / `shared_noise::reference_noise()` — 19-entry frozen set: explanatory notes, primary sources, secondary sources, tertiary sources, general and cited sources, citations, bibliography, further reading, external links, references, notes, footnotes, sources, works cited, see also, gallery, general sources, specific sources, general bibliography.
  - Python applies it to L2b only (`scripts/smart_extractor.py::extract_category_tail`).
  - Rust applies it to **both** L2 and L2b — strict improvement.

---

## Known Python regex bugs the Rust port mirrors faithfully

Do NOT "fix" these in only Rust — would diverge from the Python source-of-truth used by the running curriculum-ingest. Coordinated fix only.

1. ~~**L3 IsA title-prefix fails on long-lead articles.**~~ **FIXED 2026-05-31.** Lookahead alternates widened (in both Python `smart_extractor.py` and Rust `src/extractor/layers/l3_lead_patterns.rs` + `src/extractor/patterns.rs`) to allow NP to terminate at `[.;,]`/EOS/`\s+(?:that|which|who|where|and|or|used|of|in|for|from)\b`. NP inner-token min loosened from `{2,30}` to `{0,30}` so 2-letter words like "of"/"in" can appear inside the NP (e.g. "fraction of sunlight"). NP repetition cap raised from `{0,3}` to `{0,5}` (max 6 tokens). Helper `_emit_isa_candidates` (Python) / `emit_isa_candidates` (Rust) splits coordinator-joined NPs ("political philosophy and political movement") into two bonds when both halves pass `ok_term` + stop-noun-head check. Canonical Anarchism/Albedo/Photosynthesis/Paris now produce expected L3-isa bonds. New binary at `/tmp/lu_4_mcp.narrate-v2-l3fix-2026-05-31` (md5 `699b875c93f75334d5084b8113ad01cc`). Backfill of pre-existing articles is pending — see Lu project memory.
2. **L3 IsA fallback lookahead alternates are word-anchored.** ~~Fails on some lead shapes where the NP boundary is mid-prepositional-phrase.~~ **ALSO FIXED 2026-05-31** as part of bug #1 — fallback lookahead now uses `\s+\b` form throughout, matching the title-prefix change.
3. **`ok_term` alpha-ratio gate kills L5 FoundedIn on bare years.** "1801" has 0 alphas, fails `alpha >= max(2, len/2)`.

---

## MCP tools related to ingest

| Tool | What it does | Lang |
|---|---|---|
| `lu4d_teach_corpus_batch` | Bulk JSON-RPC triple ingest from curriculum-ingest. Up to ~1000 triples per call. Goes through `teach_with_source` (direct bond formation, scar gate). | Rust internal |
| `lu4d_file_read { route: TeachCorpus }` | File-based pre-extracted triple ingest (CSV/JSON only — text routes to UserStatement in LB). | Rust |
| `lu4d_file_extract_to_pending` | Convenience wrapper around file_read+teach for CSV/JSON. Routes through PendingLog (T.3 gate). | Rust |
| `lu4d_learn_text` (NEW 2026-05-31) | Rust in-process text extraction. Takes raw text + optional title/source_url, runs all 8 layers, knowledge-aware filter against keyword_index, queues to PendingLog. **Strategic deploy target for in-process learning.** Currently in production binary but NOT YET wired into curriculum-ingest service. | Rust |
| `lu4d_approve_pending` / `lu4d_reject_pending` / `lu4d_pending_changes` | T.3 PendingLog approval flow. | Rust |

---

## When extending

1. **Read this file first.** Check whether the name you're adding collides with existing constants / functions.
2. **Update both Python and Rust in the same port.** Bond shape is the source-of-truth contract; drift between langs eventually causes audit baseline divergence.
3. **Add the layer/constant/MCP tool to this file** as part of the change.
4. **Add a regression test** using fixtures at `tests/fixtures/anarchism_full_2026_05_31.txt`, `tests/fixtures/albedo_full_2026_05_31.txt`, or a new fixture if the new shape isn't covered.
5. **Audit baselines** are in `feedback_synthesis_bounded_by_node_quality` (agent memory) — if the new layer's quality story differs, update there too.

---

## Save path (compress + write)

Three save-related ports stacked on `src/compress.rs` + `src/brain_io_v3.rs`:

| Port | Date | What it does | Where |
|---|---|---|---|
| **OP-022.s.v5** | 2026-05-28 | Full-save BufWriter at `write_v3_compressed_encrypted_to_borrowed`. Coalesces ~115M tiny `write_all()` syscalls into ~6K page-aligned writes. **Throughput**: 5x partial-save speedup, 7.4 GiB peak RSS reduction. Earlier "CPU-bound" diagnosis was wrong — root cause was `vfs_write`/`fault_in_readable`/`copy_page_from_iter_atomic` syscall overhead. | `src/brain_io_v3.rs` ~line 2939 |
| **OP-SAVE-ASYNC** | 2026-06-01 | Async-save self-heal: `Brain::start_async_save` calls `finalize_save_if_ready` first so a prior worker's `.tmp -> .lbr` rename always commits before the next save begins. Fixes silent save loss + memory climb when async polls were shorter than save duration. | `src/save_async.rs` `Brain::start_async_save` |
| **OP-PARZSTD** | 2026-06-01 | Parallel zstd-compress + ordered-write pipeline in the slow re-encode + partial-passthrough Pass 2. Producer (entity packer) sends raw blocks to a bounded crossbeam-channel; N=`available_parallelism().min(16)` worker threads compress + XOR-encrypt; a writer thread reorders by `block_id` via `BTreeMap` and writes frames in monotonic order. **Byte-identical output** preserved across worker counts; verified by external sha256 comparison vs pre-change serial binary at 500/5K/50K-node fixtures. **Throughput**: ~1.66x speedup on 12-vCPU host (165 MB/s -> 274 MB/s on 312MB compressed output), ~89% of Amdahl ceiling (51% parallelizable from internal notes flamegraph). | `src/compress.rs` — `parzstd_worker` / `parzstd_writer` helpers + scoped-thread harness inside `write_compressed_network_payload_to` (slow path) and `write_compressed_network_payload_partial_passthrough_to` Pass 2 |

Notes on **OP-PARZSTD correctness** (load-bearing):

1. **Block independence.** `zstd::bulk::Compressor::compress_to_buffer` produces a fresh frame per call — no streaming/window state shared across blocks. The reader path (`CompressedNetwork::block_frame_bytes` + `zstd::bulk::decompress_to_buffer`) decodes each frame standalone.
2. **Keystream is a pure function of `block_id`.** `encrypt::keystream_chunk(key, block_index, chunk_index)` = sha256 of those three inputs. Parallel computation is bit-identical to serial.
3. **OP-022.s.v3 passthrough fast-path preserved.** `write_compressed_network_payload_passthrough_to` (full passthrough) and the partial path's Pass 1 (verbatim source-frame copy) are unchanged — they still serial-memcpy already-encrypted source frames and never enter the parallel pool. Only the Pass 2 overlay re-encode and the slow-path full re-encode parallelize.
4. **`Send` propagated up.** `write_compressed_network_payload_to::<W>` and `write_compressed_network_payload_partial_passthrough_to::<W>` gained `W: ... + Send`; the two top-level wrappers in `brain_io_v3.rs` (`write_v3_compressed_encrypted_to` and `write_v3_compressed_encrypted_to_borrowed`) gained `RW: ... + Send`. All production call sites pass `BufWriter<&mut File>` (Send) so this is transparent.
5. **Memory bounded.** `bounded(num_workers * 2)` work channel + `bounded(num_workers * 4)` done channel cap in-flight blocks at ~6N. At 256 KiB block size + 16 workers that's ~24 MiB raw + ~24 MiB compressed in flight, dwarfed by the existing per-bond index Vecs.

---

## Brain v4 — pending OP-BRAIN-V4

Named parts introduced by the planned OP-BRAIN-V4 port (see `docs/OP-BRAIN-V4-PortPlan.md`). Phase 0 (this addition) only records the names so future contributors don't collide; the implementations land in the indicated phases.

| Name | Definition | Phase |
|------|------------|-------|
| `CompactBond` | 14-byte packed bond representation (Phase 2.1; was 13 in Phase 2). Subject+object as `u32` NodeIds, predicate+flags in 1 byte, weight in 1 byte, `keyframe_offset: u16 + keyframe_count: u8` addressing a contiguous slice of the per-block keyframe pool, confidence `u8`. SST format version is 2. | 2 (amended 2.1) |
| `NodeIntern` | The string ↔ NodeId interning table. Stays in RAM; node count plateaus per healthy growth curve. | 1 |
| `SST` (Sorted Subject Table) | Immutable on-disk file containing zstd-compressed encrypted blocks of bonds, subject-sorted. Lu convention; analogous to RocksDB SSTable. **Shipped Phase 2; format version 2 after Phase 2.1 amendment.** | 2 |
| `Block` | A zstd-compressed encrypted payload inside an SST. 256 KiB target, contains a subject range of bonds. **Shipped Phase 2.** | 2 |
| `MemTable` | Bounded in-memory write buffer. `BTreeMap<NodeId, Vec<MemBond>>`. Capped at 500 MB by default. **Shipped Phase 3.** | 3 |
| `MemBond` | In-memory bond representation with sequence number for tombstone ordering. Slightly larger than CompactBond. **Shipped Phase 3.** | 3 |
| `WAL` (Write-Ahead Log) | Append-only on-disk log for MemTable durability. Replayed on daemon start. **Shipped Phase 3.** | 3 |
| `Manifest` | Atomic-updated metadata file listing all live SSTs and their level + subject ranges. **Shipped Phase 4.** | 4 |
| `LevelEntry` | One LSM level's SSTs in the manifest. L0 = recent flushes (possibly overlapping subject ranges); L1+ = compacted (disjoint, after Phase 6). **Shipped Phase 4.** | 4 |
| `SSTEntry` | One SST's manifest row: monotonic id, relative file path, subject_min/max seeds, size, mtime, bond_count. **Shipped Phase 4.** | 4 |
| `FlushWorker` | Background thread that swaps the MemTable, writes an L0 SST, updates the manifest atomically, and truncates the WAL. **Shipped Phase 4.** | 4 |
| `ReadPathMode` | Tri-state read-dispatch flag (`V3Only=0`, `V4Only=1`, `MergedV3V4=2`) on `Brain::read_path_mode`. Production default is `V3Only` -- byte-equivalent to pre-Phase-5 reads. Phase 8 flips the cutover. **Shipped Phase 5.** | 5 |
| `Brain::read_outgoing_bonds` | Canonical public read API. Dispatches on `ReadPathMode`. Returns a `Vec<MergedBondView>`. **Shipped Phase 5.** | 5 |
| `MergedBondView` | Owned, layer-tagged read result from the Phase 5 merge. Distinct from `sst::BondView` and `view::BondView` because those bind to their storage (Arc'd SST block / heap+archive); `MergedBondView` is the unified post-merge shape. **Shipped Phase 5.** | 5 |
| `bond::confidence_from_keyframe_source` | Phase 5 helper. Maps `KeyframeSource` -> `Confidence` per the asymmetry-aware ingest contract. **Shipped Phase 5.** | 5 |
| `Level` | A tier in the compaction hierarchy. L0 = recent flushes, L1+ = compacted, disjoint subject ranges. **Shipped Phase 6.** | 6 |
| `Minor compaction` | L0 → L1 compaction. Merges overlapping L0 SSTs with L1 SSTs. **Shipped Phase 6.** | 6 |
| `Major compaction` | Periodic global compaction. Removes tombstones, re-sorts, can apply confidence-tiered eviction. **Shipped Phase 6.** | 6 |
| `Tombstone` | Marker for a deleted bond. Lives in MemTable + SSTs until removed by compaction. **Shipped Phase 6.** | 6 |
| `Curiosity prefetch` | Background BlockCache prefetch driven by Lu's wonder queue. Predicts which subjects are about to be queried and warms their blocks. | (later optimization) |

### Phase 0 named parts (regression bar)

Landed 2026-06-01 in this commit:

| Name | What it is | Where |
|------|-----------|-------|
| `lu_brain_bench` (crate) | Single-purpose workspace member that establishes the v3 baseline. No production code path depends on it; v4 phases compare against its output. | `crates/lu_brain_bench/` |
| `bench_v3` (binary) | Runs four scenarios: (1) bond add throughput, (2) single-subject read latency, (3) spread-activation 2-hop latency, (4) RAM-at-bond-count milestones. Writes a Markdown report. Single-threaded by design. | `crates/lu_brain_bench/src/bench_v3.rs` |
| `docs/v3_baseline_2026-06-01.md` | The v3 baseline numbers themselves. Treat as a fixed regression bar; every v4 phase posts a comparison report next to it. | `docs/v3_baseline_2026-06-01.md` |

### Phase 2 named parts (SSTable read path)

Landed 2026-06-01 in this commit. Read-only from production for now; the writer exists for tests and for Phase 4's MemTable flush.

| Name | What it is | Where |
|------|-----------|-------|
| `SST` | Opened SSTable reader. Holds mmap + parsed block index + shared `BlockCache`. Exposes `header()`, `get_bonds_for_subject(&Seed)`, `iter_bonds()`. `Send + Sync`. | `src/sst.rs` |
| `SSTHeader` | Fixed 132-byte header: magic (`"LSO4"`), version, level, bond_count, block_count, manifest_id, index offset+length, encryption IV, subject_min/max seeds. | `src/sst.rs` |
| `BlockIndexEntry` | One row of the block index: `subject_min_seed`, frame `offset`, `compressed_length`, `decompressed_length`. 48 bytes on disk. Binary-searchable. | `src/sst.rs` |
| `BlockCache` (sst) | Shared bounded LRU of decompressed block payloads, keyed by `(sst_uid, block_id)`. Mirrors `src/compress.rs::BlockCache` semantics. | `src/sst.rs` |
| `BondView` | Read-only view of one bond from an SST block. Holds the block `Arc` alive via internal field. **Phase 2.1**: carries `keyframes: Vec<TemporalKeyframe>` (the bond's full history, materialised from the pool slice) plus `latest_keyframe()` / `keyframes_iter()` convenience accessors — not a single `Option<TemporalKeyframe>`. | `src/sst.rs` |
| `CompactBond` (on-disk) | **Phase 2.1 (14 bytes):** u32 subject_id + u32 object_id + u8 predicate (5b BondType + 1b direction + 2b flags) + u8 weight + u16 keyframe_offset + u8 keyframe_count + u8 confidence. Block-local NodeIds, not brain-wide. A bond's keyframe history is `pool[offset .. offset + count]`. Phase 2's 13-byte single-keyframe layout is retired; v1 SSTs are refused at read. | `src/sst.rs` |
| `SSTWriter` | Builds an SST from a subject-sorted bond stream. Enforces sort order; rejects out-of-order pushes. Compresses then encrypts each block. Used by Phase 2 tests, Phase 4 MemTable flush, Phase 6 compaction. | `src/sst_writer.rs` |
| `Brain::read_sst(path)` | Testing-only entry point to open an SST with a fresh cache. Phase 5 (`read_outgoing_bonds_v4`) wires the manifest-driven open path via `Brain::v4_sst_cache`. | `src/brain.rs` |
| `bench_sst_read` (binary) | OP-BRAIN-V4 Phase 2 bench: writes a 10M-bond SST in `/tmp/`, measures cold/warm `get_bonds_for_subject` p50/p99, BlockCache hit rate, single-block decompress cost, `iter_bonds` MB/s. | `crates/lu_brain_bench/src/bench_sst_read.rs` |
| `docs/v4_phase2_bench_2026-06-01.md` | Phase 2 bench numbers. | `docs/v4_phase2_bench_2026-06-01.md` |

### Phase 4 named parts (Flush worker + Manifest + L0 SSTs)

Landed 2026-06-01 in this commit. Wires the Phase 2 SST writer to the Phase 3 MemTable + WAL via a background flush thread. Production write paths still do NOT route through these (the curriculum-ingest service runs the OLD binary); Phase 5 wires the read path through MemTable + L0 SSTs.

| Name | What it is | Where |
|------|-----------|-------|
| `Manifest` | In-memory snapshot of the on-disk MANIFEST file. Holds `Vec<LevelEntry>`, `KeyFingerprint`, the live WAL path, last-compaction timestamp, and the monotonic `next_sst_id` counter. `Manifest::save_atomic` writes `.tmp`, fsyncs, rotates the existing `MANIFEST` to `MANIFEST.prev`, renames `.tmp -> MANIFEST`, and fsyncs the parent directory in that order. `Manifest::load` falls back to `MANIFEST.prev` on CRC/rkyv failure. | `src/manifest.rs` |
| `ManifestPayload` | rkyv-archive struct serialized inside the MANIFEST file. Schema version 1. The file layout is `magic(4) + schema_version(4) + payload_len(4) + reserved(4, zero) + rkyv_payload(N) + crc32(4)`. The 16-byte prefix exists specifically so the rkyv payload starts at an 8-byte-aligned offset (required by `rkyv::check_archived_root`); the loader copies the body into a `rkyv::AlignedVec` before validation. | `src/manifest.rs` |
| `KeyFingerprint` | 16-byte SHA-256 prefix of the master key used to encrypt the brain. `KeyFingerprint::none()` is all-zero for unencrypted brains. Stored in the manifest so a load can refuse a wrong-key load. | `src/manifest.rs` |
| `LevelEntry` | One LSM level's SSTs (level + Vec<SSTEntry>). L0 (recent flushes) may have overlapping subject ranges; L1+ become disjoint after Phase 6 compaction. | `src/manifest.rs` |
| `SSTEntry` | One SST file's manifest row: id (monotonic), file (relative path string), subject_min/subject_max (32-byte Seed bytes), size, mtime, bond_count. | `src/manifest.rs` |
| `manifest::cleanup_orphan_ssts` | Walks an `sst_dir`, finds every `L<digits>-<digits>.sst` not referenced by the manifest, and renames it to `<name>.orphan-<ts>`. Run at Brain startup (via `enable_flush_worker`) so a crash mid-flush that fsync'd the SST but never updated the manifest doesn't leak. Renames rather than deletes so operators can inspect. | `src/manifest.rs` |
| `manifest::sst_filename` | Canonical SST filename: `L<level>-<id>.sst`. | `src/manifest.rs` |
| `FlushWorker` | Spawns a background thread that polls `MemTable::is_full()` every `DEFAULT_POLL_INTERVAL_MS` (100 ms). When triggered, it: (1) clones the manifest and bumps `next_sst_id`; (2) calls `MemTable::swap_for_flush` to get an owned snapshot; (3) writes an L0 SST via `SSTWriter` and `finalize_with_meta` (fsync); (4) saves the updated manifest atomically (`save_atomic` fsyncs the file, rotates `.prev`, renames, fsyncs the dir); (5) calls `Wal::truncate_up_to(highest_seq + 1)` AFTER the manifest is durable. The flush_lock Mutex serialises concurrent `flush_now` calls. | `src/flush.rs` |
| `FlushWorker::flush_now` | Synchronous flush trigger. Holds the flush_lock; returns after the SST + manifest + WAL truncate are durable. No-op when the MemTable is empty. | `src/flush.rs` |
| `SSTFinalizeMeta` | Companion struct returned by `SSTWriter::finalize_with_meta`. Captures `bond_count`, `block_count`, `subject_min`, `subject_max`, `manifest_id` so the flush worker can build an `SSTEntry` without re-opening the file. | `src/sst_writer.rs` |
| `Brain::enable_memtable` | **Amended Phase 4:** in addition to opening the WAL + creating the MemTable, now REPLAYS any pre-existing records from the WAL into the fresh MemTable BEFORE returning. This is the durability gate: every WAL record that was fsync'd to disk is recovered on the next daemon start. Also stashes a clone of the network's `NodeIntern` in `Brain::flush_intern` so the Phase 4 flush worker shares the same Seed -> NodeId mapping as the read path. | `src/brain.rs` |
| `Brain::enable_flush_worker` | Opt-in Phase 4 initializer. Loads (or creates fresh) the manifest at `manifest_path`; runs `cleanup_orphan_ssts` against `sst_dir`; spawns the `FlushWorker`. Requires `enable_memtable` to have been called first. | `src/brain.rs` |
| `Brain::flush_now` | Public proxy for `FlushWorker::flush_now`. Returns an error if no worker is attached. | `src/brain.rs` |
| `Brain::manifest_snapshot` | Read-only `Arc<Manifest>` snapshot. Cheap to clone; readers never block the flush worker. | `src/brain.rs` |
| `AssociativeNetwork::node_intern` | New public accessor for the in-memory `NodeIntern`. Phase 4 wiring clones this handle and shares it across the MemTable + FlushWorker so subject Seeds map to the same NodeIds everywhere. | `src/network.rs` |
| `bench_flush` (binary) | OP-BRAIN-V4 Phase 4 bench: scenario A measures one big `flush_now` (SST throughput + bonds/sec); scenario B runs the worker with a small cap so it flushes repeatedly, measures sustained bonds/sec; scenario C measures `Manifest::save_atomic` latency in isolation. Output: Markdown table at `--out`. | `crates/lu_brain_bench/src/bench_flush.rs` |
| `docs/v4_phase4_bench_2026-06-01.md` | Phase 4 bench numbers. | `docs/v4_phase4_bench_2026-06-01.md` |

### Phase 5 named parts (read path integration)

Landed 2026-06-01 in this commit. Wires `Brain::read_outgoing_bonds` to consult MemTable + L0 SSTs + v3 archive in newest-first order, dedups by `(subject, predicate, object)` taking latest-seq, and re-derives confidence band from keyframe-source provenance. Behind a feature flag (`ReadPathMode`); production default is `V3Only` so shipped binaries are unchanged until Phase 8 cutover.

| Name | What it is | Where |
|------|-----------|-------|
| `ReadPathMode` | `V3Only=0` / `V4Only=1` / `MergedV3V4=2`. Stored on `Brain::read_path_mode: AtomicU8` and mirrored on `AssociativeNetwork::read_path_mode` so the spread loop can dispatch without a back-pointer to Brain. **Shipped Phase 5.** | `src/read_path.rs` |
| `MergedBondView` | Owned bond view returned by the Phase 5 merge. Fields: `subject`, `object`, `predicate`, `weight`, `confidence`, `keyframes: Vec<TemporalKeyframe>`, `seq`, `source_layer: BondSourceLayer`. | `src/read_path.rs` |
| `BondSourceLayer` | Diagnostic tag: `MemTable` / `L0Sst` / `V3Archive`. Records which layer the merge picked the bond from. | `src/read_path.rs` |
| `DedupKey` | 65-byte stable dedup key (`subject + object + predicate_discriminant`). Hashed in the merge `HashMap`. | `src/read_path.rs` |
| `bond::confidence_from_keyframe_source` | Pure function mapping `KeyframeSource` to `Confidence`. `Baseline / Hebbian / Manual / PromotedFromVariant` preserve stored band (return `None`). `Synthesis / Hypothesis / Variant / Agent` -> `Inferred`. `ConfirmedHypothesis` -> `Verified`. `Session(_)` -> `Unverified`. | `src/bond.rs` |
| `bond::deciding_keyframe_source` | Picks the highest-timestamp keyframe's source as the deciding signal for confidence re-derivation. | `src/bond.rs` |
| `bond::derive_confidence` | Convenience wrapper: applies `confidence_from_keyframe_source` against `stored` as the preserve-stored fallback. | `src/bond.rs` |
| `Brain::read_path_mode` field | `AtomicU8`, default 0 (`V3Only`). | `src/brain.rs` |
| `Brain::read_path_mode()` getter / `set_read_path_mode(mode)` setter | Switch the read-path mode. `set_*` also propagates to `AssociativeNetwork::set_read_path_mode_u8`. | `src/brain.rs` |
| `Brain::read_outgoing_bonds(subject)` | Canonical public read API. `V3Only` -> identical to today's `view_bond`-per-neighbour path; `V4Only / MergedV3V4` -> calls `read_outgoing_bonds_v4`. Returns `Vec<MergedBondView>`. | `src/brain.rs` |
| `Brain::read_outgoing_bonds_v4(subject)` | Three-layer merged read. MemTable (highest seq) -> L0 SSTs (sorted newest-id first, range-filtered by [subject_min, subject_max]) -> v3 (seq=0). Dedup by `DedupKey` keeping highest-seq winner. Confidence re-derived via `bond::derive_confidence`. | `src/brain.rs` |
| `Brain::orphan_cleanup_done` field | `Arc<AtomicBool>`, default `false`. Shared with `FlushWorker` so cleanup runs exactly once across `FlushWorker::start` calls. | `src/brain.rs` |
| `Brain::v4_sst_cache` field | `Mutex<HashMap<sst_id, Arc<SST>>>`. Per-SST handle cache so hot reads do not re-open files. | `src/brain.rs` |
| `Brain::v4_block_cache` field | `Arc<sst::BlockCache>`. Shared across L0 SST opens within the v4 read path. | `src/brain.rs` |
| `flush::run_orphan_cleanup_once` | Phase 5: idempotent one-shot orphan-SST cleanup gated by `FlushWorkerShared::orphan_cleanup_done` (an `Arc<AtomicBool>` shared with Brain). First `FlushWorker::start` runs it; subsequent calls no-op. | `src/flush.rs` |
| `FlushWorker::start` (amended Phase 5) | Now takes an additional `orphan_cleanup_done: Arc<AtomicBool>` argument and runs `run_orphan_cleanup_once` before the polling loop. Brain::enable_flush_worker no longer calls `cleanup_orphan_ssts` itself. | `src/flush.rs` |
| `AssociativeNetwork::set_read_path_mode_u8` / `read_path_mode_u8` | Mirror the mode flag onto the network so the spread loop can dispatch. | `src/network.rs` |
| `AssociativeNetwork::install_v4_read_handles` | Brain calls this after `enable_memtable` and `enable_flush_worker` so the network sees the MemTable, manifest snapshot, and SST dir for the v4 spread augmentation. | `src/network.rs` |
| `AssociativeNetwork::neighbors_outgoing_for_read_mode` | Hot-path spread neighbour lookup. `V3Only` -> exact `neighbors_outgoing_capped`. `V4Only / MergedV3V4` -> union of v3 + MemTable + L0 with hub-cap reapplied. | `src/network.rs` |
| `AssociativeNetwork::v4_sst_handle_cache` | Per-SST handle cache for the spread hot path (mirrors `Brain::v4_sst_cache`). | `src/network.rs` |
| `bench_read_v4` (binary) | OP-BRAIN-V4 Phase 5 bench: builds a v3 heap (1M bonds) + MemTable (10K) + 3 L0 SSTs (100K each), measures `read_outgoing_bonds` p50/p99/mean in each mode. Output: Markdown report at `--out`. | `crates/lu_brain_bench/src/bench_read_v4.rs` |
| `docs/v4_phase5_bench_2026-06-01.md` | Phase 5 bench numbers + honest assessment vs the brief's 200 ns projection. | `docs/v4_phase5_bench_2026-06-01.md` |

### Phase 6 named parts (Leveled compaction + tombstones + parsed-block cache)

Landed 2026-06-01 in this commit. Background compaction promotes L0 SSTs into disjoint-range L1 SSTs; subsequent major compactions promote N -> N+1. Tombstones use a reserved on-disk confidence byte (`0xFF`); the MemTable carries a `tombstone: bool` flag on `MemBond`. Parsed-block cache shaves the bytes->ParsedBlock parse cost from the V4Only read hot path.

| Name | What it is | Where |
|------|-----------|-------|
| `CompactionWorker` | Background thread that polls the manifest, decides minor (L0->L1) or major (N->N+1) compactions, opens input SSTs, merges + dedups + GCs tombstones, writes output SSTs, atomic-swaps the manifest. Single-threaded; one compaction in flight at a time. **Shipped Phase 6.** | `src/compaction.rs` |
| `compaction::compact_once` | The canonical compaction implementation. Holds `compaction_lock` for the duration so concurrent `compact_now` calls serialize. fsync ordering: new SSTs (writer-fsync) -> `save_atomic` (manifest+dir fsync) -> live-Arc swap -> cache invalidation -> old-file deletion. **Shipped Phase 6.** | `src/compaction.rs` |
| `compaction::decide_compaction` | Trigger logic: L0 SST count >= `DEFAULT_L0_TRIGGER` (4) triggers a minor compaction; level-N total size > `level_cap_bytes(N)` triggers a major. **Shipped Phase 6.** | `src/compaction.rs` |
| `compaction::compute_snapshot_horizon` | Tombstone GC discipline. Conservative horizon: `max(sst_id) - DEFAULT_L0_TRIGGER`. A tombstone with seq strictly less than the horizon may be dropped iff the destination level is the deepest in the manifest. **Shipped Phase 6.** | `src/compaction.rs` |
| `Manifest::find_ssts_in_range` | Returns SSTs at the given level whose subject range overlaps the query range. Used by compaction to pick destination-level SSTs that must be folded into the merge. **Shipped Phase 6.** | `src/manifest.rs` |
| `Manifest::swap_sst_set` | Atomic immutable update: removes a set of old SST ids and adds new entries at their declared levels, preserving the `next_sst_id` counter and sorting each level by subject_min. Debug-asserts the L1+ disjointness invariant. **Shipped Phase 6.** | `src/manifest.rs` |
| `Manifest::levels_ge_1_disjoint` | Invariant check: for every level >= 1, no two SSTs have overlapping subject ranges. Used by `swap_sst_set` and tests. **Shipped Phase 6.** | `src/manifest.rs` |
| `Manifest::record_compaction` / `ssts_at_level` / `level_size_bytes` / `level_sst_count` / `max_level` | Manifest accessors / mutators added for compaction triggers + reporting. **Shipped Phase 6.** | `src/manifest.rs` |
| `Tombstone` (on-disk) | A bond row in the SST whose `confidence` byte equals `TOMBSTONE_CONFIDENCE` (`0xFF`). The marker lives in the byte rather than a separate field because the `Confidence` enum only uses values 0..=4; growing the CompactBond would force a format version bump. **Shipped Phase 6.** | `src/sst.rs` |
| `is_tombstone_byte` / `TOMBSTONE_CONFIDENCE` | Public helpers + constant. **Shipped Phase 6.** | `src/sst.rs` |
| `BondView::is_tombstone` / `raw_confidence` field | `BondView` now exposes the raw confidence byte so callers can detect tombstones without re-deriving the Confidence band. **Shipped Phase 6.** | `src/sst.rs` |
| `MemBond.tombstone` field | New field on `MemBond`. `false` for normal inserts; `true` for `MemTable::delete` outputs. **Shipped Phase 6.** | `src/memtable.rs` |
| `MemBondFlush` | Flush-time projection of a MemBond, carrying the tombstone flag. Returned by the new `iter_for_flush` shape. **Shipped Phase 6.** | `src/memtable.rs` |
| `MemTable::delete` | Inserts a tombstone for `(subject, predicate, object)` with a fresh monotonic seq. The merge layer's seq-wins dedup then naturally shadows older same-key inserts. **Shipped Phase 6.** | `src/memtable.rs` |
| `SSTWriter::push_tombstone` | Writes a tombstone row to the in-flight SST. Subject-ordering contract identical to `push_bond`. **Shipped Phase 6.** | `src/sst_writer.rs` |
| `ParsedBlock` (public) | Materialised block view: `subject_table: Vec<Seed>`, `keyframe_pool: Vec<TemporalKeyframe>`, `bonds: Vec<CompactBondRaw>`. Was `pub(crate)` pre-Phase-6; promoted to `pub` so the compaction module can iterate parsed blocks across SSTs. **Shipped Phase 6.** | `src/sst.rs` |
| `ParsedBlockCache` | Bounded LRU of `Arc<ParsedBlock>` keyed by `(sst_uid, block_id)`. Default cap `DEFAULT_PARSED_BLOCK_CACHE_BYTES = 512 MiB`. `invalidate_sst(sst_uid)` evicts every entry for the given SST when the manifest swap removes it. **Shipped Phase 6.** | `src/sst.rs` |
| `SST::attach_parsed_cache` / `get_or_parse_block` | Phase 6 hot path: consults the parsed-block cache first; on miss falls through to `decompress_block` + `parse_block`, installs the result. Drives the V4Only p50 win. **Shipped Phase 6.** | `src/sst.rs` |
| `Brain::enable_compaction` / `compact_now` / `compaction_worker` | Opt-in compaction lifecycle on Brain. Requires `enable_flush_worker` to have been called first. **Shipped Phase 6.** | `src/brain.rs` |
| `Brain::v4_parsed_block_cache` field | `Arc<ParsedBlockCache>` shared with the read path and the compaction worker. **Shipped Phase 6.** | `src/brain.rs` |
| `MergedBondView.tombstone` field | True iff the merge winner is a tombstone row. The public read API filters tombstone winners out of the returned Vec; the flag is on the struct so callers that want to observe shadowing (e.g. compaction tests) can. **Shipped Phase 6.** | `src/read_path.rs` |
| `bench_compaction` (binary) | Phase 6 compaction bench: fills 10 L0 SSTs, compacts, measures throughput (MB/s) and pre/post V4Only read latency. **Shipped Phase 6.** | `crates/lu_brain_bench/src/bench_compaction.rs` |
| `docs/v4_phase6_bench_2026-06-01.md` | Phase 6 bench numbers + read-latency improvement vs Phase 5 baseline. **Shipped Phase 6.** | `docs/v4_phase6_bench_2026-06-01.md` |

### Phase 7 named parts (v3 -> v4 migration tool)

Landed 2026-06-01 in this commit. Standalone CLI that converts an
existing v3 brain.lbr into the v4 directory layout (MANIFEST + L0 SSTs +
empty WAL/). Byte-exact verify mandatory. Production touch forbidden by
design — operator copies brain.lbr to /tmp first.

| Name | What it is | Where |
|------|-----------|-------|
| `V3ToV4Migrator` | Top-level struct that owns v3 path, v4 dir, optional master key, target SST size + batch size. `run()` does the conversion; `verify()` round-trips every (subject, predicate, object) triple. **Shipped Phase 7.** | `src/migration_v3_v4.rs` |
| `MigrationReport` | Per-run statistics: bonds_migrated, ssts_written, elapsed, peak_rss_bytes, idempotent_no_op flag. **Shipped Phase 7.** | `src/migration_v3_v4.rs` |
| `VerifyReport` / `VerifyMismatch` | Round-trip equality result + per-bond mismatch enum (`MissingInV4`, `ExtraInV4`, `WeightDiffers`, `KeyframesDiffer`). **Shipped Phase 7.** | `src/migration_v3_v4.rs` |
| `PARTIAL_MIGRATION_FILE` (constant) | Marker file written at migration start, removed at successful finalize. Presence on entry triggers restart-clean. **Shipped Phase 7.** | `src/migration_v3_v4.rs` |
| `DEFAULT_SSTS_TARGET_SIZE` (constant) | 256 MiB; matches the natural L0 size from MemTable flushes. **Shipped Phase 7.** | `src/migration_v3_v4.rs` |
| `lu4d_migrate_v3_to_v4` (binary) | Standalone CLI: `--v3 <path> --v4 <dir> [--keyfile <path>] [--ssts-target-size <bytes>] [--verify-only]`. Always runs verify after a successful migration. **Shipped Phase 7.** | `scripts/lu4d_migrate_v3_to_v4.rs` |
| `bench_migration` (binary) | Phase 7 bench: builds a 1M-bond v3 brain, migrates to v4, measures wall-clock, MB/s, peak RSS, SST count. Output: Markdown report at `--out`. **Shipped Phase 7.** | `crates/lu_brain_bench/src/bench_migration.rs` |
| `docs/v4_phase7_bench_2026-06-01.md` | Phase 7 bench numbers + honest assessment + production-shape rehearsal results. **Shipped Phase 7.** | `docs/v4_phase7_bench_2026-06-01.md` |

### Phase 3 named parts (MemTable + WAL)

Landed 2026-06-01 in this commit. Opt-in primitive; production write paths do NOT route through these yet. Phase 4 (flush + L0 wiring) and Phase 5 (read path integration) come next.

| Name | What it is | Where |
|------|-----------|-------|
| `MemTable` | Bounded in-memory write buffer keyed by `NodeId`. `BTreeMap<NodeId, Vec<MemBond>>` under a `Mutex`; `size_bytes` and `seq_counter` in atomics to keep the `is_full()` poll path lock-free. Default cap 500 MiB. Exposes `insert`, `is_full`, `swap_for_flush`, `iter_for_flush`. | `src/memtable.rs` |
| `MemBond` | In-memory bond: `object: Seed`, `predicate: BondType`, `weight: f32` (full precision, quantized to u8 at SST flush time), `keyframes: Vec<TemporalKeyframe>`, `seq: u64`. | `src/memtable.rs` |
| `MemTable::swap_for_flush` | Atomic swap primitive: returns the old MemTable boxed for the flush thread, replaces self with a fresh empty one in one locked step. Inserts racing the swap land in exactly one of the two tables, never lost. Verified by `swap_is_atomic_against_concurrent_inserts` (4 writers + 1 swapper, 20K inserts). | `src/memtable.rs` |
| `Wal` | Append-only WAL writer wrapping a `BufWriter<File>`. Stores raw `Seed`s (not interned NodeIds) so durability does not depend on the in-memory intern state at boot. CRC32 trailer per record. Fsync cadence: 100 ms or 1 MiB, env-overridable via `LU_WAL_FSYNC_MS` / `LU_WAL_FSYNC_BYTES`. | `src/wal.rs` |
| `WalRecord` | Replayed-record struct: `seq, subject, object, predicate, weight, keyframes`. | `src/wal.rs` |
| `wal::replay` | Reads a WAL segment, validates magic + per-record CRC32, returns records up to the first torn/corrupt record. Never panics on truncation or CRC failure. | `src/wal.rs` |
| `Wal::truncate_up_to(seq)` | Compaction primitive called after MemTable flush succeeds. Rewrites the segment with `seq >= cutoff` records via tmp-file + atomic rename. | `src/wal.rs` |
| `Brain::enable_memtable` | Opt-in initializer attaching a `MemTable` + `Wal` pair to a Brain. Accessors `Brain::memtable()` / `Brain::wal()` expose the live handles. Default Brain construction leaves both `None`. | `src/brain.rs` |
| `bench_memtable` (binary) | OP-BRAIN-V4 Phase 3 bench: measures MemTable insert throughput at empty/half/near-full, WAL append throughput across three fsync cadences, swap-for-flush latency. | `crates/lu_brain_bench/src/bench_memtable.rs` |
| `docs/v4_phase3_bench_2026-06-01.md` | Phase 3 bench numbers. | `docs/v4_phase3_bench_2026-06-01.md` |

### Phase 8 named parts (v3/v4 boot detection + v4 default flush/compaction + lu4d_save no-op)

Landed 2026-06-01 in this commit. Closes OP-BRAIN-V4 with operator-triggered cutover: the daemon detects v3 .lbr files vs v4 directories at `--brain <path>`, routes through the matching loader, and `lu4d_save` becomes a documented no-op for v4 brains because durability is continuous via the flush worker. Operator runbook in `docs/op_brain_v4_cutover_runbook.md`.

| Name | What it is | Where |
|------|-----------|-------|
| `BrainLayout` | Enum returned by `Brain::detect_brain_layout`: `V3File`, `V4Dir`, `Missing`. `is_v3()` covers V3File + Missing (the v3 empty-create path); `is_v4()` is V4Dir only. **Shipped Phase 8.** | `src/brain.rs` |
| `BrainFormatOverride` | CLI override for the auto layout detection: `Auto`, `V3`, `V4`. Surfaced via `--brain-format <auto\|v3\|v4>`. Parser accepts `v3file`/`file`, `v4dir`/`dir`/`directory` synonyms. **Shipped Phase 8.** | `src/brain.rs` |
| `Brain::detect_brain_layout(path)` | File-system detection helper. Directory + `MANIFEST` -> `V4Dir`; file -> `V3File`; missing -> `Missing`; directory without MANIFEST -> `Format` error pointing at the migration tool. **Shipped Phase 8.** | `src/brain.rs` |
| `Brain::load(path, key)` | Canonical dispatch entry point. Detects layout, routes v3 file to `load_or_new_with_mode_and_key(.., Auto, key)`, v4 directory to `load_v4`. Missing path falls through to v3 empty-create so first-run flows keep working. **Shipped Phase 8.** | `src/brain.rs` |
| `Brain::load_v4(brain_dir, key)` | V4 boot path. Validates MANIFEST loadability (Phase 4's `.prev` fallback applies), constructs an empty Brain, computes the WAL path (`<dir>/wal/current.wal` by default; `LU_BRAIN_WAL_PATH` env override for tests), runs `enable_memtable` (replays the WAL), `enable_flush_worker`, `enable_compaction`, sets `read_path_mode = V4Only`, stashes `brain_dir_v4 = Some(dir)`. **Shipped Phase 8.** | `src/brain.rs` |
| `Brain::brain_dir_v4()` / `brain_dir_v4` field | `Option<PathBuf>`. `Some(dir)` iff the Brain was loaded via `load_v4`; consulted by `save()` to short-circuit and by `lu4d_save` to return the v4 no-op response. **Shipped Phase 8.** | `src/brain.rs` |
| `Brain::save()` (Phase 8 amendment) | For v4-loaded brains (`brain_dir_v4.is_some()`) the method becomes a no-op returning `Ok(())` immediately. Durability is provided by the flush worker (MemTable -> WAL -> L0 SST). Callers that need an immediate flush use `flush_now()` instead. **Amended Phase 8.** | `src/brain.rs` |
| `--brain-format <FMT>` (CLI) | Daemon flag added in Phase 8. `auto` (default) consults `detect_brain_layout`; `v3` forces the legacy `.lbr` path and errors if the target is a directory; `v4` forces the directory path and errors if the target is a file or missing. **Shipped Phase 8.** | `src/main.rs` |
| `lu4d_save` (Phase 8 amendment) | When the brain was loaded via the v4 directory path, the tool returns `{"saved": true, "brain_format": "v4", "noop": true, "path": <dir>}` for both blocking and async modes. v3 brains keep the prior behaviour. **Amended Phase 8.** | `src/mcp.rs` |
| `phase8_v4_boot_tests` (module) | 14 unit tests covering layout detection (file, dir w/ MANIFEST, dir w/o MANIFEST, missing path), `BrainFormatOverride` parsing, `Brain::load_v4` end-to-end on a migrated brain, V4Only read-path round-trip, missing-MANIFEST error, `Brain::load` dispatch for v3/v4/missing, the v4 `save()` no-op, WAL replay on boot, and `flush_now` producing an L0 SST. **Shipped Phase 8.** | `src/brain.rs` (in-module `#[cfg(test)] mod phase8_v4_boot_tests`) |
| `tests/v4_boot_cli.rs` | Integration tests that spawn the `lu4d` binary to assert the CLI-only mismatch paths: `--brain-format v4` against a v3 file errors with `is not a directory`; `--brain-format v3` against a v4 directory errors with `is a directory`; `--brain-format v4` against a missing path errors with the `lu4d_migrate_v3_to_v4` hint; auto against an empty directory errors with `no MANIFEST`; garbage `--brain-format` value rejected. **Shipped Phase 8.** | `tests/v4_boot_cli.rs` |
| `bench_v4_boot` (binary) | Phase 8 bench: builds a synthetic v3 brain (default 200K bonds), migrates to v4, measures cold v3 boot wall-clock + RSS delta vs cold v4 boot wall-clock + RSS delta. Output: Markdown report at `--out`. **Shipped Phase 8.** | `crates/lu_brain_bench/src/bench_v4_boot.rs` |
| `docs/v4_phase8_bench_2026-06-01.md` | Phase 8 bench numbers (cold v3 vs v4 boot at the synthetic 200K-bond baseline). **Shipped Phase 8.** | `docs/v4_phase8_bench_2026-06-01.md` |
| `docs/op_brain_v4_cutover_runbook.md` | Operator runbook for the v3 -> v4 production cutover: pre-checklist, ordered cutover steps (build, snapshot, migrate, verify, deploy, systemd, start), rollback procedure, post-cutover validation, decommission gate. **Shipped Phase 8.** | `docs/op_brain_v4_cutover_runbook.md` |

### Phase 9 named parts (write-path routing for v4 brains)

Landed 2026-06-02 in this commit. Closes the cutover-found gap surfaced by internal notes: Phases 1-8 built MemTable + WAL + flush + compaction + read-path integration but never routed live writes through them. The 2026-06-01 cutover daemon happily accepted `lu4d_teach_corpus_batch` for 6.88M bonds and silently wrote them to the v3 in-memory HashMaps; the WAL stayed at 8 bytes (magic header only). Phase 9 wires the write path so MCP-level teach calls land in the MemTable + WAL when `brain_dir_v4.is_some()`.

| Name | What it is | Where |
|------|-----------|-------|
| `Brain::teach_with_kind_v4(...)` | Phase 9 v4 write helper. Inserts the bond into the MemTable and appends the matching WAL record. Called from `teach_with_kind` when `brain_dir_v4.is_some()`. Scar-seed admission preserved (HashSet membership check), hub-degree write-side enforcement deliberately skipped (the MemTable has no in-degree concept). Reinforce-vs-new decided by probing the MemTable for an existing `(subject, predicate, object)` triple; the Phase 5 read-merge dedupes by latest-seq so a fresh insert structurally supersedes any older one. WAL-append failure issues a compensating tombstone via `MemTable::delete` so the on-disk state remains consistent with the in-memory state. **Shipped Phase 9.** | `src/brain.rs` |
| `teach_with_kind` (Phase 9 amendment) | The unique chokepoint for every teach path (`teach`, `teach_corpus`, `teach_with_source`, pending `apply_change`, research `apply_proposal`). Phase 9 adds the `brain_dir_v4.is_some()` branch at the bond-formation step; node creation in `network.nodes` is preserved on both paths (the in-memory map is transient on v4 but enables the same-session ask path). **Amended Phase 9.** | `src/brain.rs` |
| `tool_status` (Phase 9 amendment) | When `brain.brain_dir_v4().is_some()`, the response's `bond_count` is sourced from `MemTable::bond_count()` + sum of `SSTEntry::bond_count` across `manifest_snapshot().iter_ssts()`, and `format_version` is hard-set to `"v4"`. Pre-Phase-9 the status always reported `network.bond_count()` (the empty v3 graph) for v4 brains. **Amended Phase 9.** | `src/mcp.rs` |
| `tool_hub_status` (Phase 9 amendment) | When `brain.brain_dir_v4().is_some()`, the response returns the configured caps + an empty `top_hubs` list and a clear `hub_degree_unwired: true` flag. Rationale: the v3 in-degree / out-degree counters live on `network` and stay zero for v4-loaded brains; reporting them would be silently misleading. A future port will add a v4 hub-degree accountant. **Amended Phase 9.** | `src/mcp_cat3b.rs` |
| `cutover_v4_write_path_phase9_tests` (module) | 8 unit tests covering the Phase 9 invariants: teach-grows-WAL-past-magic, teach-lands-in-MemTable, v3-teach-unchanged, v4-merged-read-path-surfaces-teach, restart-after-teach-preserves-bonds (WAL replay), teach_corpus-on-v4-routes-through-MemTable, status-reports-v4-bond-count-from-MemTable, scarred-seed-refuses-v4-teach-without-inserting. **Shipped Phase 9.** | `src/brain.rs` (in-module `#[cfg(test)] mod cutover_v4_write_path_phase9_tests`) |
| Phase 9 end-to-end MCP wire test | Bash test at `/tmp/phase9/e2e_test.sh` that bootstraps a fresh v4 brain, sends `lu4d_teach_corpus_batch` with 100 entries over JSON-RPC, asserts WAL grows past 8 bytes, status reports 100 bonds, ask retrieves the inserted subject in-session, restart preserves the bond count via WAL replay. Verified 2026-06-02: WAL 0 -> 11,108 bytes, batch 100 landed, status 100 bonds, ask 1 hit (top: `phase9_s_7`), post-restart status 100 bonds. **Shipped Phase 9.** | `/tmp/phase9/e2e_test.sh` |


### Phase 10 named parts (node persistence)

Landed 2026-06-02. Closes the Phase 9 follow-on gap: the write path landed bonds in MemTable + WAL, but `KnowledgeNode` metadata (`content`, `KnowledgeContent` kind, `Confidence` band, timestamps, embedding) stayed in the RAM-only `network.nodes` HashMap. After a daemon restart, WAL replay rebuilt the MemTable bonds but the nodes those bonds referenced had empty metadata, so `lu4d_ask` returned `total_idk` (no subject content to surface). Phase 10 makes the substrate complete: nodes are now first-class durable records carried in the WAL, the MemTable, and the SST per-block tables.

**SST format bump: v2 -> v3.** The block payload header's trailing `reserved: u32` becomes `node_table_size: u32` (must equal `subject_table_size`). After the subject table, the block carries a parallel node table where each slot is a length-prefixed bincode-encoded `Option<KnowledgeNode>` (zero length = `None`). The reader refuses v1 and v2 SSTs with a clear error pointing at the migration tool. There is no v2 -> v3 conversion path; the v3 -> v4 migration tool was extended to emit v3 SSTs.

**WAL record-type discriminator.** A new byte sits immediately after the length prefix: `0 = Bond`, `1 = Node`. Bond bodies are unchanged from Phase 3; Node bodies carry `seq:u64 + seed[32] + node_bytes_len:u32 + node_bytes`. CRC32 trailer scheme is the same. The magic header is unchanged (`LU4WAL\x01\x00`); the production WAL at `/var/lib/lu/brain-v4/wal/current.wal` was a magic-only file at the time of the Phase 9 rollback, so no in-the-wild WAL needs format-mixing handling.

| Name | What it is | Where |
|------|-----------|-------|
| `node::serialize_knowledge_node` / `node::deserialize_knowledge_node` | Public bincode helpers used by the WAL, SST writer, and MemTable spill. The struct shape itself is unchanged from Phase 1. **Shipped Phase 10.** | `src/node.rs` |
| `WalRecord` (enum) | Pre-Phase-10 this was a struct of Bond fields. Phase 10 turns it into `{ Bond { .. }, Node { seq, seed, node } }`. Helpers `seq()`, `is_bond()`, `is_node()` give type-agnostic access. **Shipped Phase 10.** | `src/wal.rs` |
| `Wal::append_node(seq, seed, node)` | New method. Emits a Node-type WAL record with the same fsync cadence as `append`. **Shipped Phase 10.** | `src/wal.rs` |
| `MemTable::insert_node` / `lookup_node` / `iter_nodes_for_flush` / `node_count` | New methods. The MemTable inner state grows a `BTreeMap<NodeId, (KnowledgeNode, u64)>` parallel to `entries`. `swap_for_flush` drains both maps atomically. `iter_nodes_for_flush` yields seed-sorted `(Seed, KnowledgeNode, seq)` tuples for the flush worker. **Shipped Phase 10.** | `src/memtable.rs` |
| `SST_VERSION = 3` + per-block node table | The reader refuses v1/v2 SSTs. The block payload header's `reserved` u32 becomes `node_table_size`. Node table follows the subject table; each slot is a 4-byte length prefix + optional bincode KnowledgeNode payload. **Shipped Phase 10.** | `src/sst.rs` |
| `ParsedBlock::node_table` | `Vec<Option<KnowledgeNode>>` parallel to `subject_table` (same indexing). Cached alongside the bonds via `ParsedBlockCache`. **Shipped Phase 10.** | `src/sst.rs` |
| `BondView::subject_node()` / `object_node()` | Accessors returning `Option<&KnowledgeNode>` from the same block. Zero extra block reads. **Shipped Phase 10.** | `src/sst.rs` |
| `SST::iter_nodes()` | Walks every block and yields `(Seed, KnowledgeNode)` pairs from the per-block node tables. Used by the boot-time `network.nodes` rehydration path + migration verify. **Shipped Phase 10.** | `src/sst.rs` |
| `SSTWriter::push_bond_with_nodes(...)` / `push_node(...)` | New writer entry points. `push_bond_with_nodes` is the flush-time canonical method (bond + optional subject + optional object KnowledgeNode). `push_node` is the standalone path used by migration for isolated nodes (no outgoing bonds). The pre-Phase-10 `push_bond` still works (treats nodes as `None`). **Shipped Phase 10.** | `src/sst_writer.rs` |
| `Brain::persist_node_v4(seed, node)` | New helper. Mirrors `teach_with_kind_v4` for nodes: inserts into MemTable nodes + appends WAL Node record. Idempotent via MemTable membership check. Called from `teach_with_kind` immediately after `network.add_node` for v4-loaded brains. **Shipped Phase 10.** | `src/brain.rs` |
| `Brain::rehydrate_network_nodes_from_v4_ssts(...)` | New helper. Called by `load_v4` after `enable_flush_worker` to walk every SST in the manifest and re-insert node payloads into `network.nodes` -- so post-restart `lu4d_ask` can still call `network.get_node(seed)`. Idempotent. Skips unreadable SSTs with a logged warning. **Shipped Phase 10.** | `src/brain.rs` |
| `enable_memtable` (Phase 10 amendment) | WAL replay handles both Bond and Node records. Bond records re-populate the MemTable; Node records re-populate `network.nodes` AND the MemTable nodes table (so the next flush carries them onward). Log line counts both. **Amended Phase 10.** | `src/brain.rs` |
| `flush_once` (Phase 10 amendment) | The flush worker collects `iter_nodes_for_flush` alongside bonds, builds a `Seed -> KnowledgeNode` map, and routes bond pushes through `push_bond_with_nodes`. Any node that did not land via a bond row (e.g. tombstone-only paths) gets a standalone `push_node` in the same SST. **Amended Phase 10.** | `src/flush.rs` |
| `compact_once` (Phase 10 amendment) | The compaction scan collects per-Seed node winners using highest-SST-id as the dedup seq. The output writer attaches winners to the corresponding bond pushes via `push_bond_with_nodes`. **Amended Phase 10.** | `src/compaction.rs` |
| `V3ToV4Migrator::run` (Phase 10 amendment) | Materialises the v3 archive's nodes into a `Seed -> KnowledgeNode` map up front, attaches subject + object node payloads on every bond push via `push_bond_with_nodes`, and sweeps isolated nodes (in v3 archive but with zero outgoing bonds) into a final SST via `push_node`. `MigrationReport.nodes_migrated` reports the count. **Amended Phase 10.** | `src/migration_v3_v4.rs` |
| `V3ToV4Migrator::verify` (Phase 10 amendment) | After the bond pass, the verify walks every v4 SST's `iter_nodes` and compares each `(Seed, KnowledgeNode)` against the v3 archive via byte-exact bincode. `VerifyReport.nodes_verified` counts the matches. **Amended Phase 10.** | `src/migration_v3_v4.rs` |
| Phase 10 unit tests | 13 new tests across the v4 substrate: WAL Bond/Node interleave + truncate, MemTable insert/lookup/iter/swap, SST writer round-trips (push_bond_with_nodes, iter_nodes, push_node standalone), brain restart with WAL replay (the regression-from-Phase-9 fix), brain restart with flushed SST (rehydrate path), migration round-trip, compaction node dedup keeping latest. **Shipped Phase 10.** | `src/{wal,memtable,sst_writer,brain,migration_v3_v4,compaction}.rs` |

### Phase 10.1 named parts (streaming node migration fix)

Landed 2026-06-02 in this commit. Closes the production-scale RSS regression the Phase 10 migration tool surfaced during the 2026-06-02 re-cutover attempt. Phase 10 (`8b3039d`) preloaded every v3 `KnowledgeNode` into a HashMap before pushing bonds; on the 6.79 GB / 18.6M-bond production brain that materialised ~28 GiB of RSS before any SST byte was written, killing the cutover. Phase 10.1 replaces the preload with a streaming lookup per bond.

| Name | What it is | Where |
|------|-----------|-------|
| `V3ToV4Migrator::run` (Phase 10.1 amendment) | Replaces the Phase 10 `node_map: HashMap<[u8;32], KnowledgeNode>` preload with on-demand `archive.node_by_seed` lookups per bond. The owned `KnowledgeNode` materialised for each subject/object lives only for the duration of one `push_bond_with_nodes` call. The isolated-node sweep walks `archive.iter_node_seeds()` (Seed-only, no payload deref) and resolves each missed seed via the same on-demand lookup. State retained across the bond walk: sorted Vec of bond keys + `HashSet<[u8;32]>` of emitted seed bytes -- both O(N) but with tiny per-element cost (32-64 bytes) vs Phase 10's O(N * sizeof(KnowledgeNode) ~= 3 KiB). **Amended Phase 10.1.** | `src/migration_v3_v4.rs` |
| `phase10_1_streaming_migration_bounded_rss_at_production_shape` (test) | Production-scale regression test. Builds a 600K-bond / 150K-node synthetic v3 brain, runs the migration with `set_ssts_target_size(16 MiB)` (driving multi-SST output), spawns a 25 ms-tick sampling thread that reads `/proc/self/status:VmRSS` and tracks the max. Asserts peak RSS delta over baseline is bounded by 1 GiB. Pre-fix preload would breach this; streaming measured at ~176 MiB delta. **Shipped Phase 10.1.** | `src/migration_v3_v4.rs` |
| `read_vm_rss_bytes` (test helper) | Reads current `VmRSS` from `/proc/self/status`. Returns `None` on non-Linux or in sandboxes that block `/proc` -- the test skips cleanly in that case rather than failing on unrelated platform behavior. **Shipped Phase 10.1.** | `src/migration_v3_v4.rs` (in-module) |
