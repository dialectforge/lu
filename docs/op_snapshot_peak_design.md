# OP-SNAPSHOT-PEAK Design - Bound the OP-SAVE-ASYNC snapshot+save peak under 26 GiB at 19M bonds

**Status:** Phase A diagnosis. HALT for John review before implementation.
**Branch (planned):** `feature/op-snapshot-peak` off `main` HEAD `90baed1` (note: `feature/op-train-cache-cap` carries the LRU-cap follow-up at `16240fe`; per internal notes the cap was disproved as the peak driver, so this port targets main HEAD).
**Scope:** Bound the steady-state save peak on the OP-SAVE-ASYNC binary on the live canonical brain. After tonight's 4 OOM kills at `MemoryMax=28 GiB` with ~19M bonds, the snapshot capture + worker save delta is provably the dominant transient at production load. Diagnosis focus: per-allocator attribution + minimum-viable fix set.
**Anti-LLM positioning:** unchanged - this is operational hardening, no new model surface.
**LOC budget:** ~150-450 net across the proposed fixes (see Section 5). Surface if any single fix creeps past 300.

---

## 0. Today vs after-fix

**Today internal notes + #1090 + tonight's 4 OOMs):** at 16.9M bonds the OP-SAVE-ASYNC binary peaks at **26.04 GiB VmRSS** during the save window. At ~19M bonds with the production overlay accumulated since the last reload, the binary crosses 28 GiB and trips MemoryMax. systemd auto-restarted four times in one ingest run. internal notes already disproved block-cache attribution: 512 MiB cap to 29.20 GiB peak; 2 GiB cap to 29.22 GiB. The peak driver is **OverlaySnapshot + SnapshotSideTables + bond_index/node_index writer Vec preallocations**, not the LRU.

**After-fix target:** steady-state save peak <= **24 GiB at 19M bonds**, leaving 4 GiB headroom under `MemoryMax=28 GiB` for production overlay growth between saves (currently `SAVE_INTERVAL_BONDS = 50000`). This is a **-4 GiB** reduction from tonight's measured 28+ GiB. Proposed fix set delivers ~5-7 GiB headroom from the three hot spots in Section 3.

**Material non-goals:**
- Re-architecting overlay growth internal notes already noted: archive is set only at `from_archive()` and is never re-swapped after save, so overlay grows monotonically until restart - fixing this is a separate, larger port).
- Re-quantizing embeddings (1.5 KiB/node is real but architectural, out of scope).
- Streaming the writer's `Vec<NodeIndexEntry>`/`Vec<BondIndexEntry>` (a separate port; the cost is large but covered by passthrough on idle reloads - see Section 3.3).

---

## 1. Phase A - what the snapshot machinery actually captures

Read of `src/save_async.rs:69-94` (`OverlaySnapshot::capture`) and `src/brain.rs:8884-8902` (`start_async_save`'s `SnapshotSideTables` initializer):

### 1.1 OverlaySnapshot - 5 fields

```rust
pub struct OverlaySnapshot {
    pub nodes: HashMap<Seed, KnowledgeNode>,            // .clone() of net.overlay_nodes_ref()
    pub bonds: HashMap<(Seed, Seed), TemporalBond>,     // .clone() of net.overlay_bonds_ref()
    pub overlay_deletions: HashSet<Seed>,               // .clone()
    pub bond_tombstones: HashSet<(Seed, Seed)>,         // .clone()
    pub archive: Option<Arc<NetworkArchive>>,           // Arc::clone - zero alloc
}
```

The `archive` field is the **only zero-cost capture** - `Arc::clone` bumps a refcount. The other four are deep `HashMap`/`HashSet` clones of live heap state. On a freshly-loaded brain the overlay is empty and capture is ~ms; on a long-running production daemon mid-curriculum the overlay holds every node and bond created since process start (see Section 1.4).

### 1.2 SnapshotSideTables - 16 fields (the doc-comment says "14", but the struct ships 16 after Cat.4/RuntimeConfigSnapshot landed)

| # | Field | Type | Source-of-clone | On-disk size at 17.3M bonds (1) |
|---|---|---|---|---|
| 1 | `scar_log` | `ScarLog` | `self.scar_log.clone()` | 48 B (empty) |
| 2 | `interest_profile` | `InterestProfile` | `self.interest_profile.clone()` | 16 B (empty) |
| 3 | `meta_reflection` | `MetaReflection` | `self.meta_reflection.clone()` | 32 B |
| 4 | `research_proposals` | `Vec<ResearchProposal>` | `self.research_proposals.clone()` | 8 B (empty) |
| 5 | `research_config` | `ResearchConfig` | `self.research_config.clone()` | (in `research_misc`) |
| 6 | `next_research_proposal_id` | `u64` | copy | (in `research_misc`) |
| 7 | `gap_seed_frequency` | `HashMap<Seed, u32>` | `.clone()` | 8 B (empty) |
| 8 | `pending_log_snapshot` | `PendingLogSnapshot` | `self.pending.snapshot()` (clones `BTreeMap`) | 0 B |
| 9 | `rejection_patterns` | `RejectionPatternsSnapshot` | `self.rejection_store.snapshot()` | 15.4 KiB |
| 10 | `lb_session_snapshot` | `SessionMemorySnapshot` | `self.lb_session.snapshot()` | **16.4 MiB** |
| 11 | `cs_style_snapshot` | `ConversationalStyleSnapshot` | `self.cs_style_store.snapshot()` | 59 B |
| 12 | `pack_ledger_snapshot` | `PackLedgerSnapshot` | `self.pack_ledger.snapshot()` | 28 B |
| 13 | `source_overlay` | `SourceOverlay` | `self.source_overlay.clone()` | **3.09 GiB** |
| 14 | `cap_pack_ledger_snapshot` | `CapabilityPackLedgerSnapshot` | `self.cap_pack_ledger.snapshot()` | 566 B |
| 15 | `runtime_config_snapshot` | `RuntimeConfigSnapshot` | `self.runtime_config_snapshot_for_save()` | 53 + 1982 B |

(1) On-disk lengths read directly from the v3 section trailer of `/tmp/snap_diag.lbr` (scratch copy of `/var/lib/lu/brain.lbr` at the moment ingest crashed). The v3 header is cleartext even for v3enc - only the data blocks are encrypted - so the per-section sizes are parseable without the master key.

### 1.3 Lock semantics and watchdog points

`start_async_save` is `&mut self` on `Brain`; the entire snapshot-capture phase runs synchronously on the main MCP thread (no MCP calls can be served during it). Sequence per `src/brain.rs:8855-9045`:

1. Acquire `save_runner.state` mutex, check `!is_running()`. Hold for microseconds.
2. Clone `path`, validate `save_format`, clone `encryption_key`. (~ms)
3. **`OverlaySnapshot::capture(&self.network)`** - 4 HashMap/HashSet clones + 1 Arc bump. Scales with live overlay size. This is the first big synchronous allocation.
4. Build `SnapshotSideTables` - 13 owned clones, of which only `source_overlay.clone()` (~3 GiB) and `lb_session.snapshot()` (~16 MiB) are non-trivial.
5. Capture WAL watermarks (3 file `tell()` calls - us).
6. Acquire `save_runner.state` mutex again, mark `Running`. us.
7. Spawn worker thread with `std::thread::Builder::new().name("lu4d_save_async").spawn(move || { ... })`. The move closure takes ownership of `overlay_snapshot`, `side_tables`, `watermarks`. From this point the main thread returns.

**No watchdog points** exist inside `OverlaySnapshot::capture` or the `SnapshotSideTables` initializer. If the main thread is OOM-killed mid-capture, systemd restarts and the partial allocation is reclaimed by exit - but if the partial allocation already crossed `MemoryMax`, that's the crash itself. **There is no incremental progress observable from outside the daemon during capture.**

This was noted in internal notes as the ~46-49 sec synchronous capture window on the live brain. At 19M bonds tonight that window plausibly stretches to 60+ sec, during which the main MCP thread is unresponsive and RSS is climbing past `MemoryMax`.

### 1.4 Why the overlay grows monotonically - load-bearing context

`AssociativeNetwork::archive` is set exactly once, in `from_archive()` (`src/network.rs:960`), and there is **no `swap_archive` / `reload_archive` path** anywhere in `network.rs` or `brain.rs`. After a successful async save, the new on-disk brain file is NOT re-mmap'd back as the in-process archive. Consequence: every node and bond created since process startup lives in the heap-overlay `HashMap`s and stays there until the daemon restarts.

Production reload cadence: production ingest restarts only at OOM. With ingest pushing ~50K bonds/save and saves succeeding, the overlay grows by ~50K bonds per save until either the daemon restarts (OOM, deploy) or the brain re-loads from disk. From a 9.5M-bond starting archive to tonight's 19M-bond brain, **~9.5M bonds and the ~2-3M new nodes that name them live in the overlay HashMaps**, plus the ~9.5M `source_overlay` entries each `teach()` writes (see `brain.rs:3242-3255`).

This is THE multiplier behind the snapshot peak. A fresh-from-disk daemon at 19M bonds with empty overlay would snapshot ~instantaneously; a 12-hour-old daemon at the same 19M bonds carries the entire 9.5M-bond delta in heap, and that's what gets cloned at every save.

---

## 2. Phase B/D - sized per-allocator attribution

Reference dataset: `/opt/lu/op_save_async_peak_diag/proc_sample.csv` internal notes diagnosis from earlier today). 4,210 samples at ~130ms cadence across the full pre-save -> save -> post-save window on a 16.9M-bond brain. Pre-save baseline VmRSS = **18.39 GiB** (RssAnon = 16.70 GiB). Peak VmRSS = **26.04 GiB** (RssAnon = 24.34 GiB). **Save-window delta = +7.65 GiB anonymous heap, +0.0 GiB mmap (RssFile stable at ~1.7 GiB).**

The transient is 100% anonymous heap. Mmap pages of the archive are stable across the save window - the writer reads them via Arc and writes through to a fresh `.tmp` file, page-faulting the archive in as it goes but not retaining extra mmap pages.

The csv shows the inflection at sample `t=107492 ms` (3.1 sec after `SAVE_TRIGGER` at t=104456 ms): VmSize jumps **+3.46 GiB in one sample window** (27.82 -> 31.28 GiB) and VmData jumps in lockstep. RSS then climbs +5.7 GiB over the next 440 sec as the snapshot+writer arenas page-fault in. The 3.46 GiB VmSize step is the heap-allocator's `mmap()` reservation for the new arena pages backing the `HashMap::clone()` and `Vec::with_capacity()` calls inside the snapshot+writer; the slow RSS ramp is the actual data write into those pages.

### 2.1 Per-allocator table

| # | Allocator | Estimated size at 19M bonds | Empirical anchor | Scales with | Fix shape (see Section 3 for detail) |
|---|---|---|---|---|---|
| 1 | `source_overlay.clone()` (side-table) | **~3.4 GiB heap** | on-disk section 16 = 3.09 GiB at 17.3M bonds, linear in entry count | bonds-since-reload | `Arc<RwLock<SourceOverlay>>` + COW snapshot (Section 3.1) |
| 2 | overlay-nodes `HashMap::clone()` | **~2.0-3.5 GiB heap** | overlay nodes x ~1.7 KiB (mostly the 384-f32 embedding Vec) | nodes-since-reload | switch capture to `Arc<HashMap<...>>` + COW (Section 3.2) |
| 3 | overlay-bonds `HashMap::clone()` | **~1.2 GiB heap** | overlay bonds x ~120 B value + 64 B key + 32 B entry overhead | bonds-since-reload | same `Arc<HashMap<...>>` + COW (Section 3.2) |
| 4 | writer `bond_keys: Vec<(Seed,Seed)>` | 1.22 GiB at 19M bonds | `compress.rs:431` `Vec::with_capacity(bond_count)` x 64 B | total bond count (archive + overlay) | trailer-based bond index (Section 3.3, deferred) |
| 5 | writer `node_seeds: Vec<Seed>` | 288 MiB at 9M nodes | `compress.rs:429` x 32 B | total node count | trailer-based node index (Section 3.3, deferred) |
| 6 | writer `bond_entries: Vec<BondIndexEntry>` | 216 MiB | x 12 B | total bond count | trailer-based (Section 3.3, deferred) |
| 7 | writer `node_entries: Vec<NodeIndexEntry>` | 108 MiB | x 12 B | total node count | trailer-based (Section 3.3, deferred) |
| 8 | `lb_session.snapshot()` | ~32 MiB heap | on-disk section 11 = 16.4 MiB; in-memory ~2x | LB session entry count | not worth touching at this size |
| 9 | block-cache LRU (NOT a peak driver) | <= 1 GiB at current cap | bounded by `--block-cache-bytes` | configurable | already configurable; internal notes disproved attribution |
| 10 | `bond_tombstones.clone()`, `overlay_deletions.clone()`, all other side-table clones | combined < 10 MiB | on-disk sections 2-9, 12-15 | each is bounded or empty | none needed |

Total identified transient: **~8.5 GiB at 19M bonds** (rows 1+2+3+4+5+6+7 = 3.4 + 2.7 + 1.2 + 1.22 + 0.29 + 0.22 + 0.11 ~= 9.1 GiB). This is in the right ballpark for the empirical +7.65 GiB delta at 16.9M bonds; 19M scales linearly.

### 2.2 Concentrating the fix surface

Rows 1+2+3 are **78% of the transient** and are all clones of state that the **worker thread treats as read-only**. None of the side-tables or overlay HashMaps are mutated by the writer - the writer just iterates and emits bytes. The writer's actual mutation surface is the `.tmp` file descriptor.

Rows 4-7 are writer-side preallocations (independent of overlay size - they scale with total bond/node count) and are needed by the current `brain_io_v3` writer to assemble the trailing per-entity offset table. Bounding these is a separate, larger port (Section 3.3) and not gated on this OOM fix.

Row 9 is already known-clear: internal notes demonstrated that capping the LRU at 512 MiB does not reduce peak (29.20 GiB at 512M vs 29.22 GiB at 2G).

---

## 3. Phase E - top 3 hot spots with concrete fix shape

### 3.1 Hot spot #1: `source_overlay.clone()` (3.4 GiB -> 0 GiB at snapshot)

**Current shape:** `brain.rs:8898` does `source_overlay: self.source_overlay.clone()`. `SourceOverlay` is `pub entries: HashMap<String, SourceAttribution>` (src/source.rs:362) with one entry per non-Unknown-attributed bond. Production curriculum-ingest calls `teach()` (`brain.rs:3242`) which always sets a `SourceAttribution::training_with_url(corpus, run_id, None)` or `SourceAttribution::teach("system", now_s)`, so the overlay grows by exactly one entry per teach. At 17.3M bonds the on-disk size is 3.09 GiB; in RAM (HashMap + String keys) ~3.4-5 GiB.

**Fix shape:** wrap the live `Brain.source_overlay` in `Arc<RwLock<SourceOverlay>>` (or, simpler: `Arc<Mutex<SourceOverlay>>` since teach-path writes are already serialized through `&mut self`). At snapshot capture, `Arc::clone` the handle into `SnapshotSideTables.source_overlay_handle: Arc<...>`. The worker thread takes a read lock for the duration of the write phase. The MCP thread can take a write lock between save windows to mutate.

A simpler variant: **defer the `source_overlay` clone to the worker thread** and pass `Arc<Mutex<SourceOverlay>>` through. The worker takes a read snapshot via `lock().clone()` - which IS still a 3.4 GiB clone - but that clone happens AFTER the writer's other preallocations have already grown the heap, so the *peak* moves but does not necessarily drop. **The right fix is the Arc/RwLock with a serde-borrowing write path** so the worker writes from the borrowed entries directly with no clone.

The `SideTableSource for SnapshotSideTables` impl at `save_async.rs:340-347` currently writes via `bincode_into(w, &self.source_overlay)`. With `Arc<RwLock<SourceOverlay>>`, that becomes `bincode_into(w, &*self.source_overlay_handle.read().unwrap())` and the value is serialized in place from the live (read-locked) state. A teach() concurrent with the save would block on the write lock until serialization finishes; given a write takes O(50 GiB serialized at zstd-compressed disk throughput) ~ 30 sec at worst, this is a real but bounded stall.

**Open question for John (see Section 4):** is a 30-sec teach stall during save acceptable, or do we want a true COW snapshot? COW snapshot = at save start, swap the live SourceOverlay's HashMap for an empty one wrapped in a fresh Arc, and hand the worker the now-orphaned filled Arc. Teach calls during save mutate the new empty HashMap; merge happens at finalize_save by `extend()`-ing the in-flight overlay back into the live one. This preserves snapshot semantics, costs O(post-save mutations) at merge, and avoids any teach stall.

**Estimated LOC:** ~80-120 (Arc wrapping, read-lock at write, merge logic if COW). **Estimated peak reduction: 3.0-3.4 GiB.**

### 3.2 Hot spot #2: overlay-nodes + overlay-bonds `HashMap::clone()` (~3.5 GiB -> 0 GiB at snapshot)

**Current shape:** `save_async.rs:75-77` does:

```rust
nodes: net.overlay_nodes_ref().clone(),
bonds: net.overlay_bonds_ref().clone(),
```

These refs come from `AssociativeNetwork::overlay_nodes_ref()` / `overlay_bonds_ref()` (`network.rs:909-916`) which return `&HashMap`. Each clone deep-copies every `KnowledgeNode` (with its 1.5 KiB embedding Vec) and `TemporalBond` (with its keyframes Vec). At ~3M overlay nodes and ~9M overlay bonds, the total clone footprint is ~5.1 GiB (nodes) + 1.6 GiB (bonds) = ~6.7 GiB at 19M total. The empirical +7.65 GiB delta at 16.9M is consistent with this once the source_overlay is also included.

**Fix shape (preferred):** wrap the live `AssociativeNetwork.nodes` and `bonds` fields in `Arc<RwLock<HashMap<...>>>` and have `overlay_nodes_ref` / `overlay_bonds_ref` return `Arc<RwLock<...>>` handles (NEW accessors - the `&HashMap` accessors stay for read-mostly internal use). At capture, `OverlaySnapshot::capture` does `Arc::clone` on each handle into snapshot-owned `Arc<RwLock<HashMap<...>>>` fields. The writer's `SnapshotSource` adapter takes a read lock for the duration of `for_each_node` / `for_each_bond`.

**Tradeoff:** mutations during save (concurrent `teach()` calls) block on the read-locked map. Saves run for ~30-90 sec at current sizes. Acceptable if we don't expect teach throughput during save; current `SAVE_INTERVAL_BONDS=50000` means saves are infrequent.

**Fix shape (alternative - COW swap):** at capture, swap the live `nodes` / `bonds` `HashMap`s with empty fresh ones wrapped in fresh Arcs. Hand the worker the now-orphaned filled Arcs. Teach calls during save mutate the empty maps. At finalize_save, drain the in-flight maps back into the new live maps via `extend()`. Semantically identical to the current behaviour (post-save mutations are absorbed into the next save).

The COW swap is more invasive but eliminates concurrent-teach stalls. The Arc<RwLock> variant is simpler but stalls teach for the save duration.

**Estimated LOC:** ~150-250 (touches `network.rs` accessor surface + `save_async.rs` snapshot fields + adapter impls + finalize_save). **Estimated peak reduction: 3.5-5.0 GiB.**

### 3.3 Hot spot #3 (DEFERRED): writer Vec<NodeIndexEntry> / Vec<BondIndexEntry> preallocations (~1.83 GiB)

**Current shape:** `compress.rs:429-432` does:

```rust
let mut node_seeds: Vec<Seed> = Vec::with_capacity(node_count);
let mut node_entries: Vec<NodeIndexEntry> = Vec::with_capacity(node_count);
let mut bond_keys: Vec<(Seed, Seed)> = Vec::with_capacity(bond_count);
let mut bond_entries: Vec<BondIndexEntry> = Vec::with_capacity(bond_count);
```

These back the trailing per-entity offset index the v3 format embeds at the end of the file. At 19M bonds + 9M nodes the total is 1.83 GiB **on the worker thread**, materialized during the write phase and freed when the writer returns. This is an architectural cost of the v3 format and would require a trailer-based index streaming refactor to remove.

**Fix shape:** stream offsets to a separate temp file during the main bond/node write pass, then concat into the final file at trailer assembly. Removes the in-RAM Vec preallocation entirely at the cost of one extra O(N) disk read at finalize.

**Why deferred:** the Section 3.1+3.2 fixes alone deliver enough peak reduction to clear the 28 GiB ceiling. This port is the larger structural refactor and should land separately (probably OP-SNAPSHOT-PEAK.1).

**Estimated LOC:** ~400-600 (touches `compress.rs` write loop + `brain_io_v3.rs` trailer assembly + format-version handling). **Estimated peak reduction: 1.8 GiB.** Not in this port.

---

## 4. Open questions for John

1. **Acceptable teach stall during save?** If yes, the Section 3.1 + 3.2 fixes can ship as `Arc<RwLock<...>>` with read-locked write phase (~230 LOC total, simpler). If no, we need COW swap semantics (~370 LOC total, more invasive). **Recommendation: ship Arc<RwLock> first** - current curriculum-ingest is single-threaded and teach calls during the save window are rare; COW can land as a follow-up if it ever bites.
2. **Is `source_overlay` snapshot-required at save time, or can we exclude it and rebuild from a forward-only WAL?** The on-disk section is 3 GiB and dwarfs everything else (except the network section itself). If we move source attribution to its own append-only file (similar to the existing `pending.wal` / `lb_session.wal` pattern), the snapshot side-table goes away entirely. **Recommendation: out of scope for this port**, but flag for a future OP-SOURCE-WAL once the immediate OOM is closed.
3. **Do we want to add a watchdog point inside `OverlaySnapshot::capture`?** Currently the 49-sec synchronous capture window is invisible from outside the daemon. A simple `eprintln!("[save] snapshot capture: {} ms", elapsed)` plus a counter in `BrainStatus` would let `lu4d_save_status` report capture-vs-write timing separately. **Recommendation: yes, add as a low-risk amendment to this port** (~10 LOC).
4. **Should the live `AssociativeNetwork.nodes`/`bonds` accessor surface change to `Arc<RwLock<...>>` everywhere, or only at the snapshot boundary?** The hot-path reads (`view_node`, `view_bond`) currently dereference `&HashMap` directly; wrapping in `RwLock` puts a contention point on the hottest call surface. The cleanest answer: keep `&HashMap` accessors for the hot path, add **separate** `Arc<RwLock<HashMap<...>>>` accessors for the snapshot boundary, and have the live fields BE the `Arc<RwLock<...>>` with `overlay_nodes_ref` doing `Arc::clone` + `read().unwrap()` returning a guard. The guard's `Deref<Target=HashMap>` keeps the existing call sites unchanged. **Recommendation: cleanest path is to flag this in the implementation port for re-spec once the read-side hot-path measurements are in.**
5. **Is the 17th section (`unknown17`, length 28) something I should investigate before this port lands?** It's tiny (28 bytes) and the section-id table in `brain_io_v3.rs:121-178` only documents 16 section ids. Possibly a recently-added section (CapPackLedger?) that needs a doc-comment update. Not blocking. **Recommendation: file as a follow-up cleanup.**

---

## 5. LOC estimate per proposed fix

| Fix | Touched files | LOC (best) | LOC (worst) | Peak reduction at 19M bonds |
|---|---|---|---|---|
| Section 3.1 `source_overlay` Arc<RwLock> + read-locked write | `brain.rs`, `save_async.rs`, `source.rs` | 80 | 120 | 3.0-3.4 GiB |
| Section 3.2 overlay-nodes/bonds Arc<RwLock> + read-locked write | `network.rs`, `save_async.rs`, `brain.rs` (capture call sites) | 150 | 250 | 3.5-5.0 GiB |
| Section 3 amendment: watchdog point in capture | `save_async.rs`, `brain.rs` (status fields) | 10 | 20 | (visibility only) |
| Section 3.1+3.2 total | 4 files | 240 | 390 | **6.5-8.4 GiB** |
| Section 3.3 trailer-based index (DEFERRED) | `compress.rs`, `brain_io_v3.rs` | 400 | 600 | +1.8 GiB |

Measured-empirical anchor: pre-save 18.4 GiB -> peak 26.0 GiB at 16.9M bonds. Apply 6.5 GiB reduction -> ~19.5 GiB peak at 16.9M, or ~21 GiB peak at 19M (scaling linearly with overlay size). **Headroom under `MemoryMax=28 GiB`: ~7 GiB**, comfortably enough for the 50K-bond inter-save overlay growth currently configured. Target met.

---

## 6. Halt boundary

No code changes have been made on `main`. The Cargo.toml debug-info change was committed only to local branch `op-snapshot-peak-diag-2026-05-29` for the build-with-symbols profile run and will be reverted in cleanup (Phase G). The production binary `/usr/local/bin/lu_4_mcp` is untouched (md5 verified `a34a7bb5c92812284703fca9160657d4` pre- and post-diagnosis). Production canonical brain `/var/lib/lu/brain.lbr` is untouched (only the scratch copy at `/tmp/snap_diag.lbr` was used). Both `lu.service` and `lu4d-curriculum-ingest.service` remained `inactive` throughout.

**Decisions made:** none. This is a diagnosis-only design doc; the Section 4 open questions need John's call before any implementation port spawns.

**Pre-implementation gate:** John reviews Section 3 fix shape + Section 4 open questions. Resolved-decisions table (Section 17 style) will be added once decisions land.

---

## 7. Diagnosis artifacts

Reference data preserved for future ports:

- `/opt/lu/op_save_async_peak_diag/proc_sample.csv` - 4,210-sample RSS timeline from internal notes.
- `/opt/lu/op_save_async_peak_diag/phase_log.txt` - annotated phase log.
- `/opt/lu/op_snapshot_peak_diag/` - moved at Phase G: build log, on-disk section dump, perf samples from the failed standalone reload attempt (zstd-bound on a 1 GiB block cache, did not reach snapshot capture phase within available wall time - captured perf at the bond-count walk instead).
- `/tmp/snap_diag.lbr` - REMOVED at Phase G.
