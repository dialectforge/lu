# OP-BRAIN-V4 Architectural Audit

**Date:** 2026-06-02
**Author:** Claude (after John's request to stop chasing fires and find the structural mistake)
**Context:** Three failed production cutover attempts today exposed a recurring pattern of non-streaming HashMap-the-world subsystems. This document identifies the architectural decision that creates the pattern and proposes the path forward.

---

## 1. What we observed

Six distinct production-scale failures, all with the same shape:

| # | Subsystem | Pattern | Phase | Status |
|---|---|---|---|---|
| A | Migration node preload | `HashMap<Seed, KnowledgeNode>` built before any output | 10 | Fixed (10.1) |
| B | Migration bond key sort | `Vec<(Seed, Seed)>` of all bond keys collected then sorted | 10.1 | Open (~1.2 GB in keys at production scale) |
| C | Per-block keyframe pool | No cross-bond dedup of `Vec<TemporalKeyframe>` | 2.1 | Open (flagged by Phase 2.1 agent) |
| D | Per-block subject table | All Seeds touched (subject AND object) get entries | 2/10 | Partially addressed (10.2 dropped payloads but kept Seed entries) |
| E | Verify-phase node cross-check | HashMap of all v3 nodes vs v4 nodes for byte-exact comparison | 10/10.1 | Open (caused 38+ GB RSS climb in this session) |
| F | Compaction node winners | `HashMap<Seed, (KnowledgeNode, seq)>` per compaction pass | 6/10 | Open (flagged by Phase 10 agent: ~18 GiB for 10M nodes) |

Each failure had three properties:
1. Functionally correct on small synthetic fixtures
2. Bounded streaming alternative exists in principle
3. The chosen implementation materializes a global-shape data structure before producing any output

We fixed A and D, surfaced B/C/E/F. Each fix surfaced the next problem. The pattern is not coincidence.

## 2. The structural cause

The Phase 2 agent made a decision in good faith that ended up creating this pattern across the entire substrate.

**The decision:** SST blocks use **block-local NodeIds** (per-block `Vec<Seed>` mapping block-local `u32` -> Seed) instead of **brain-wide NodeIds** (Phase 1's `NodeIntern`).

**Their reasoning** (Phase 2 commit `f82e040`, sub-agent report):
> "Block-local NodeIds chosen over brain-wide -- small index range compresses better in zstd, and keeps SSTs self-describing across compactions."

The compression argument was real: a `u32` ranging over a small set of values zstds well. The self-describing argument was real too: any SST file is loadable in isolation without consulting an external NodeIntern.

But every subsystem that touches the SST format now needs to maintain a per-block (subject Seed -> block-local NodeId) mapping. And every cross-block operation needs to materialize this mapping for the entire working set.

The pattern propagates:

- **Migration** must collect all bond keys, sort, and walk them -- because the SSTWriter expects subject-sorted input AND each block needs its own subject table populated as bonds arrive.
- **Keyframe pool** is per-block -- because keyframe references in `CompactBond` are block-local `u16`s, so identical keyframes across bonds in the same block can dedupe but cross-block can't.
- **Per-block subject table** stores every Seed touched -- because both subject_id and object_id in `CompactBond` are block-local, so any Seed that's a bond endpoint in this block needs a block-local NodeId, even if it's just an object.
- **Verify** must build a global mapping to cross-check -- because v3 nodes and v4 nodes live under different addressing schemes, so reconciling them requires materializing both at full Seed resolution.
- **Compaction** must build a global Seed -> node mapping per pass -- because merging block-local NodeIds across input SSTs requires rebuilding the per-block subject tables in the output SSTs, which means seeing each Seed at least once.

In every case the global mapping table is the thing that doesn't scale. And the reason it exists is the block-local addressing decision.

## 3. The alternative architecture

**Use brain-wide NodeIds throughout.** Phase 1 already shipped `NodeIntern` -- a brain-wide `Seed -> NodeId(u32)` table. It exists. It's loaded at brain boot. Reads use it.

The Phase 2 agent saw this and chose to use a SECOND addressing layer (block-local) for SST compression. That layered addressing scheme is the architectural mistake.

### Proposed structure

```
Brain-wide NodeIntern (Seed <-> NodeId(u32), in-memory, already exists in Phase 1)
   |                                  |
   |  resolve at read time            |  append at write time
   v                                  v
Bond SSTs                          Node Store
   bonds in per-level files          KnowledgeNode payloads
   subject-sorted by NodeId          addressed by brain-wide NodeId
                                     own MemTable + WAL + flush
   CompactBond:                      Lookup: brain_node_id -> KnowledgeNode
     subject_node_id: u32            Compaction trivial:
     object_node_id: u32             merge by NodeId, latest seq wins,
     predicate, weight,              streaming.
     kf_offset, ...

   Per-block:
     keyframe pool
     NO subject table
     NO node table
```

### What this fixes

| Antipattern | Why it goes away |
|---|---|
| A. Migration node preload | Nodes stream into node store independently; bond walk references brain-wide NodeIds |
| B. Migration bond key sort | Bonds can stream from v3 archive's existing subject-ordered iterator -> SSTWriter directly. Sort is "free" because v3 already maintains sorted order. |
| C. Per-block keyframe pool no-dedup | Brain-wide keyframe pool with a single global Baseline keyframe shared by all bonds (the `KeyframeSource::Baseline` constant) |
| D. Per-block subject table | Gone. Bonds reference brain-wide NodeIds directly; no per-block remapping. |
| E. Verify cross-check | Stream v3 archive and v4 bond SSTs in lockstep, both sorted by brain-wide NodeId. No HashMap. |
| F. Compaction node winners | Node compaction is on the node store, not in bond SSTs. Merge by NodeId in subject-sort order, streaming. |

### What this costs

The Phase 2 compression argument was real but quantitatively small. Block-local `u32` NodeIds in a single subject range (say 100K-200K distinct values per block) zstd-compress in 1-2 bytes per occurrence. Brain-wide `u32` NodeIds across the full 0-9M range compress in 2-3 bytes per occurrence. Per-bond cost difference: ~2 bytes on the bond reference, doubled (subject + object) = ~4 bytes per bond extra. For 18.6M bonds: ~75 MB extra raw. Compressed: ~20-40 MB. Across the entire brain output: a fraction of a percent.

The "self-describing across compactions" argument was real but the workaround is trivial: a brain header in the manifest (or a sidecar file) carries the NodeIntern table. Loading an SST requires loading that table first. The NodeIntern is the smallest possible structure (~32 bytes per node x node count). At 9M nodes that's ~288 MB -- same order as a single SST block, trivial.

### What this requires

This is a v5 substrate, not a v4 patch. Roughly the same scope as the original 8-phase v4 plan but with the corrected architectural foundation. Components that were correctly designed (MemTable, WAL, manifest, flush worker shape, leveled compaction shape, the Phase 9 write-path-routing pattern) carry over to v5 unchanged. Components that depend on block-local addressing (`CompactBond` layout, per-block subject/keyframe/node tables, the SSTWriter+SSTReader interface, the parsed-block cache shape, the migration and compaction algorithms) need refactoring.

Estimated scope: **3-5 weeks of focused work**, NOT a single-session sprint. With a production-shape test fixture (5-10M records) gating every phase.

## 4. Lessons that should ride forward

1. **Synthetic correctness tests are not architectural validation.** Every phase shipped passing tests on small fixtures. None of those tests caught the antipattern. A production-shape fixture (5-10M records, not 100K) needs to be in the regression suite from Phase 0 onward for any future substrate rewrite.

2. **Decisions about identifier scope are architectural, not local.** The block-local NodeId decision lived in `SSTWriter::intern_subject` and looked like an implementation detail. It propagated to every subsystem and shaped every subsequent design. Identifier scope decisions need explicit architectural review, not sub-agent autonomy.

3. **The pattern of "build a global structure, then stream output" is a red flag in any port brief.** I wrote this pattern into multiple Phase 0-10 briefs without recognizing it. Any future brief that says "collect all X, then process" against a multi-million-record input should be challenged at brief time.

4. **The substrate work is still valuable.** ~15K LOC of careful Rust ports Phases 0-10.2 are NOT wasted -- MemTable, WAL, manifest, flush worker, leveled compaction, read-path integration, write-path routing, node-persistence-in-WAL, streaming-migration scaffold, the production-shape bench harness -- all of these carry forward to v5 with the architectural correction. The Phase 1 NodeIntern actually IS the foundation we should have used throughout.

## 5. Recommended next actions

**Immediate (this session):**

- Done: Commit Phase 10.2 (`c289e1c`)
- Pending: Push 46 unpushed commits from your local clone (Hetzner couldn't resolve `github-lu4` SSH alias)
- Done: This audit document
- Done: the project log entry capturing the audit + decision (#1134)

**Near-term (next session or two):**

- Leave production on v3 indefinitely. The v3 architecture works at scale.
- Open a v5 design doc (companion to OP-BRAIN-V4-PortPlan, supersedes the post-cutover sections of it)
- Inventory which v4 components carry forward unchanged vs which need refactoring
- Design the brain-wide-NodeId + decoupled-node-store substrate with explicit attention to streaming discipline at every layer

**v5 specifically:**

- Build a production-shape test fixture FIRST (5M-bond synthetic) and write the bench harness that asserts bounded RSS at every layer. Make this the regression gate from phase 1 onward.
- Reuse Phase 0 bench harness as the v3 baseline that v5 must match or beat.
- Reuse MemTable, WAL, manifest patterns (correctly streaming, no changes needed).
- Rebuild SST format with brain-wide NodeIds, decoupled node store, dedup'd keyframe pool.
- Migration becomes much simpler (no global sort: v3 archive already streams subject-sorted).
- Compaction becomes much simpler (merge by NodeId, streaming).
- Verify becomes streaming (two cursors, no HashMap).

## 6. Honest accounting of this session

We shipped:
- Phases 0-10.2: substrate complete in design, 23 commits, 16K+ LOC
- Multiple debug discoveries: write-path-not-wired (Phase 9), migration-not-streaming (Phase 10.1), object-node-duplicated (Phase 10.2)
- Three production cutover attempts with clean rollback to v3 each time

We did NOT ship:
- A working v4 production cutover

We learned:
- The substrate architecture is wrong in a specific identifiable way (block-local NodeIds), and the right fix is a v5 substrate rewrite with brain-wide NodeIds and decoupled node storage

The work was not wasted. The substrate code is largely correct in shape and most of it carries forward. But the cutover-path conclusion is honest: v4 doesn't work at production scale and won't be made to work via more fixes. v5 is the answer, and it's a separate, properly-scoped project.

---

## Appendix A: The pattern, traced in actual code

For each antipattern, the offending materialization site in current code:

**A. Migration node preload** (FIXED by Phase 10.1):
- `src/migration_v3_v4.rs:383` pre-10.1: `let mut node_map: HashMap<[u8; 32], KnowledgeNode> = HashMap::new();`
- Materialized all 9M v3 nodes before bond walk.

**B. Migration bond key sort** (STILL PRESENT):
- `src/migration_v3_v4.rs:338`: `let mut keys: Vec<(Seed, Seed)> = archive.iter_bond_keys().collect();`
- 18.6M x 64 bytes = ~1.2 GB raw Vec of bond keys, then in-memory sort.
- Source: bond iteration order from v3 archive is not subject-sorted natively; we have to sort.
- Brain-wide-NodeId alternative: v3 archive iteration IS subject-sorted by Seed; just stream it.

**C. Per-block keyframe pool no-dedup** (Phase 2.1 known limitation):
- `src/sst_writer.rs`: keyframe pool is `Vec<TemporalKeyframe>` per-block, no cross-bond dedup.
- For 18.6M bonds x 1 Baseline keyframe each = 18.6M keyframe copies.
- Brain-wide-NodeId alternative: single brain-wide Baseline keyframe constant, all bonds reference it; per-block pool only carries non-Baseline keyframes (rare).

**D. Per-block subject table** (PARTIALLY ADDRESSED by Phase 10.2):
- `src/sst_writer.rs::intern_subject`: every Seed touched in a block gets a 32-byte Seed + 4-byte placeholder.
- Phase 10.2 stopped attaching node PAYLOAD for objects but the Seed entry remains.
- Brain-wide-NodeId alternative: gone entirely.

**E. Verify-phase node cross-check** (CAUSED OOM IN THIS SESSION):
- `src/migration_v3_v4.rs::verify()`: builds a HashMap of v3 nodes and v4 nodes to cross-check byte-exact equality.
- At 9M nodes: ~38+ GB RSS observed, still climbing at abort.
- Brain-wide-NodeId alternative: stream v3 and v4 cursors in NodeId-sorted order, compare in lockstep, no HashMap.

**F. Compaction node winners** (FLAGGED BY PHASE 10 AGENT):
- `src/compaction.rs`: builds `HashMap<Seed, (KnowledgeNode, seq)>` per compaction pass.
- At 10M nodes: ~18 GB estimated.
- Brain-wide-NodeId alternative: compaction merges by NodeId in NodeId-sorted order, streaming, no HashMap.

## Appendix B: Counter-arguments to brain-wide NodeIds

The Phase 2 agent's compression and self-description arguments were not wrong. They were just outweighed by the streaming cost.

For completeness:

1. **Compression cost of brain-wide NodeIds.** With 9M total nodes, NodeIds span 0-9M (~23 bits). Block-local NodeIds in a hub-spoke block span 0-200K (~17 bits). The 6-bit difference per occurrence x 2 occurrences (subject + object) per bond x 18.6M bonds = ~28 MB raw cost. Compressed: ~10-20 MB. **Verdict: negligible.**

2. **Self-description argument.** With brain-wide NodeIds, an SST cannot be read in isolation -- you need the NodeIntern table. **Mitigation:** the NodeIntern lives in the manifest's sidecar file (the brain directory's `MANIFEST.intern` or similar). It's ~288 MB at 9M nodes. Every consumer of an SST needs the manifest anyway. The "self-description" property is preserved at the brain level, not at the SST file level. **Verdict: trivial fix.**

3. **Compaction simplicity argument.** With block-local NodeIds, an SST file is fully self-contained -- compaction can read one and write a different one without any external state. With brain-wide NodeIds, compaction needs the NodeIntern to resolve any Seed lookups (e.g., during merge). **Mitigation:** the NodeIntern is read-only during compaction; load once, share via `Arc`, no locking. **Verdict: trivial.**

The compression and self-description arguments are real but small. The streaming cost is real and large. The architecture should prioritize streaming.

---

*Audit complete. Action item: stop chasing fires. Open v5 design when ready.*
