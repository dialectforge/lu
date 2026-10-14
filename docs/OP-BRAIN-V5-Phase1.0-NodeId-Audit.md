# OP-BRAIN-V5 Phase 1.0 — NodeId Call-Site Audit

**Status:** Phase 1.0 read-only investigation complete 2026-06-09. No source files modified. Companion to `OP-BRAIN-V5-Design.md` §2 Q1 revision and §4 Phase 1.0. Codebase HEAD `1f2addb`.

**Scope per design doc §2 Q1 revision:** widening `pub struct NodeId(pub u32)` → `pub struct NodeId(pub u64)` (high 32 bits reserved, varint on wire for new formats). Phase 1.0 enumerates every call site BEFORE code change so Phase 1.x can sequence without u32↔u64 conversion churn rippling through unrelated code.

**Headline finding:** the NodeId newtype has a remarkably tight blast radius. Only **2 files outside the declaration site** mention the type by name (`src/memtable.rs`, `src/network.rs`). All other apparent hits in the initial grep (74 lines, 5 files) are doc-comments in `sst.rs` / `wal.rs` describing block-local `u32` indices (a different concept) or describing WAL's deliberate non-use of NodeId. The widening is a small, surgical port; the larger architectural Phase 1 work is **WAL persistence (Q8) and migration import (from v3 SeedRegistry)**, not the type widening itself.

---

## 1. Scope — call site counts by category

Grep base: `grep -rn "NodeId" src/ crates/ scripts/` → 74 hits across 5 files. After categorization:

| Category | Count | Notes |
|---|---|---|
| DECLARATION | 1 | `src/node_intern.rs:71` — `pub struct NodeId(pub u32);` |
| API_SURFACE | 4 | `NodeIntern::{get, intern_or_get, reverse_lookup, reverse_unchecked, reverse_collect}` signatures — all return/accept `NodeId` and are exported via `pub mod node_intern` in `src/lib.rs:6`. Counted once each but `reverse_collect` is generic on `IntoIterator<Item=NodeId>` so it's the bound that widens. |
| FN_SIGNATURE (internal) | 1 | `src/memtable.rs:fn iter_for_flush` etc. take `&NodeIntern` only, no NodeId in signature; the only internal NodeId-typed fn signature is `predicate_discriminant` (no) — actually zero internal fn signatures carry `NodeId` directly. All internal uses are local let-bindings inside method bodies, type-inferred. **Count is effectively 0** for explicit FN_SIGNATURE NodeId mentions outside the API_SURFACE row above. |
| STRUCT_FIELD | 4 | `src/memtable.rs:126` `BTreeMap<NodeId, Vec<MemBond>>`; `src/memtable.rs:130` `BTreeMap<NodeId, (KnowledgeNode, u64)>`; `src/network.rs:332` `HashMap<Seed, Vec<NodeId>>` (outgoing); `src/network.rs:334` `HashMap<Seed, Vec<NodeId>>` (incoming). |
| TRAIT_IMPL | 1 | `#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]` at `node_intern.rs:70`. Trait bounds on `NodeIntern::reverse_collect<I: IntoIterator<Item=NodeId>>` at `node_intern.rs:171`. Counted as one cluster. **No external trait impls.** |
| CONVERSION | 7 | `node_intern.rs:134` `NodeId(raw as u32)`; `node_intern.rs:144,152,174` `id.0 as usize`; `node_intern.rs:76` `pub fn as_u32(self) -> u32`; `node_intern.rs:293` test `let _: u32 = id.0;`. Plus the `debug_assert!(raw <= u32::MAX as usize, …)` at line 133 (a width sentinel that goes away when raw IS u64). |
| WIRE_FORMAT | **0** | **No production wire format currently encodes the NodeId newtype.** `#[derive(Archive, …)]` does NOT appear on `NodeId` or `NodeIntern`. The on-disk v3 archive carries raw Seeds. The v4 SST format's `subject_id`/`object_id` are raw `u32` block-local indices, NOT the `NodeId` newtype (per design doc §3 "NOT salvageable" list at `sst.rs:81-87`). WAL `wal.rs:1-30` doc-comment explicitly says "stores **raw Seeds** (32 bytes each), NOT brain-wide NodeIds". |
| TEST | 9 | `node_intern.rs:185-294` test module: `intern_or_get`, `reverse_lookup_round_trip`, `reverse_lookup_out_of_range_returns_none` (uses literal `NodeId(99)`), concurrent inserts, `reverse_collect_filters_out_of_range` (uses `NodeId(9999)`), newtype compile-time check (uses `NodeId(42)`, `let _: u32 = id.0`). All inside `#[cfg(test)]`. |
| DOC_COMMENT | ~47 | Bulk of the grep hits. All `///` or `//!` lines in `node_intern.rs`, `sst.rs` (block-local NodeId concept), `wal.rs` (deliberate non-use), `memtable.rs` (caveat about NodeId ordering), `network.rs` (Phase 1 rationale comments). |

**Total: 74 grep lines categorize as 1 DECLARATION + 4 API_SURFACE + 4 STRUCT_FIELD + 1 TRAIT_IMPL cluster + 7 CONVERSION + 0 WIRE_FORMAT + 9 TEST + ~47 DOC_COMMENT = 73 (1 cross-counted between DECLARATION and TRAIT_IMPL since the derive line wraps the struct).**

Non-Rust touches: **none.** The lu4d crate's MCP JSON-RPC tool surface (`src/mcp.rs`, `src/mcp_*.rs`) does not expose NodeId — all tool arguments and return types are Seeds, strings, or content payloads. Verified via `grep -rn "NodeId" crates/ scripts/` returning zero hits in scripts/ Rust files and zero in `crates/lu_brain_bench` Rust files (only `NodeIntern` is imported in bench, not `NodeId`).

## 2. Per-file breakdown

### 2.1 `src/node_intern.rs` (the declaration site)

| Line | Category | Snippet | Widening notes |
|---|---|---|---|
| 70 | TRAIT_IMPL | `#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]` | No change — all derive macros work identically on u64. |
| 71 | DECLARATION | `pub struct NodeId(pub u32);` | **Change to `pub struct NodeId(pub u64);`.** |
| 74 | API_SURFACE / DOC | `/// Raw u32 value. Equivalent to id.0.` | Doc update: u32 → u64. |
| 75-76 | API_SURFACE | `pub fn as_u32(self) -> u32 { self.0 }` | **Decision needed (open Q1.0.A).** Options: (a) rename to `as_u64` and change return; (b) keep `as_u32` as truncating helper + add `as_u64`; (c) drop method entirely (no callers — only used inside this file). Grep shows zero external callers; recommend **drop and add `as_u64`** (cleanest). |
| 79-84 | STRUCT_FIELD (internal) | `forward: HashMap<Seed, NodeId>` + `reverse: Vec<Seed>` | No change to field type; `Vec` indexing uses `id.0 as usize` so u64→usize still works on 64-bit Linux (the only deployed target). |
| 122 | API_SURFACE | `pub fn intern_or_get(&self, seed: Seed) -> NodeId` | Signature stays — return type alias widens transparently. |
| 133 | CONVERSION | `debug_assert!(raw <= u32::MAX as usize, "NodeIntern u32 overflow");` | **Delete this assert** — the whole point of widening is removing the 4.29B node ceiling. Optionally replace with `debug_assert!(raw <= u64::MAX as u128 as usize, …)` but that's vacuous on 64-bit; just drop it. |
| 134 | CONVERSION | `let id = NodeId(raw as u32);` | **Change to `NodeId(raw as u64);`.** |
| 142,144 | API_SURFACE / CONVERSION | `reverse_lookup(&self, id: NodeId) -> Option<Seed>` + `g.reverse.get(id.0 as usize).copied()` | `id.0 as usize` still works on 64-bit. **Add overflow guard** for the very-large-id case (post-widening, a malicious or stale id could carry a u64 > usize::MAX on 32-bit — non-issue for deployed target but pedantically: `if id.0 > usize::MAX as u64 { return None; }`). |
| 150,152 | API_SURFACE / CONVERSION | `reverse_unchecked(&self, id: NodeId) -> Seed` + `g.reverse[id.0 as usize]` | Same as above; widening transparent. |
| 171,174 | API_SURFACE / CONVERSION | `reverse_collect<I: IntoIterator<Item=NodeId>>` + `.filter_map(|id| g.reverse.get(id.0 as usize).copied())` | Same. |
| 185-294 | TEST | Test module: `NodeId(99)`, `NodeId(9999)`, `NodeId(42)`, `let _: u32 = id.0` | **All literal `NodeId(N)` calls keep working** (u32 literal coerces to u64). `let _: u32 = id.0` (line 293) — **this fails** after widening; change to `let _: u64 = id.0`. |

### 2.2 `src/memtable.rs`

| Line | Category | Snippet | Widening notes |
|---|---|---|---|
| 5-6 | DOC_COMMENT | `//! into brain-wide NodeId(u32) via the shared NodeIntern` | Doc update: u32 → u64. |
| 47 | (use stmt) | `use crate::node_intern::{NodeId, NodeIntern};` | No change. |
| 55-57 | DOC_COMMENT | `Object Seed (NOT interned — object NodeIds are block-local in the SST format …)` | Stale once Phase 3 lands new bond SST — leave as-is for Phase 1, revisit at Phase 3. |
| 126 | STRUCT_FIELD | `entries: BTreeMap<NodeId, Vec<MemBond>>` | Type widens transparently. `BTreeMap<u64, _>` and `BTreeMap<u32, _>` are wire-compatible only via serde, and this is a RAM-only struct (no Archive derive). **No issue.** Per-entry memory cost: BTreeMap leaf-key slot grows 4 bytes/entry. At 100M subjects in MemTable that's 400 MB extra — but MemTable cap is 500 MiB by default and a 100M-subject MemTable would already be ~7 GB just from MemBond payload, so the +4-byte-per-key delta is rounding error. |
| 127-130 | DOC + STRUCT_FIELD | `nodes: BTreeMap<NodeId, (KnowledgeNode, u64)>` | Same as 126. |
| 397-405 | DOC_COMMENT | `in NodeId-sorted order. … NodeIds are assigned by first-observation.` | No code change; doc accurate. |

All method bodies that touch NodeId use type-inferred local bindings (`let subj_id = intern.intern_or_get(subject);`) — these compile unchanged.

### 2.3 `src/network.rs`

| Line | Category | Snippet | Widening notes |
|---|---|---|---|
| 320-327 | DOC_COMMENT | `OP-BRAIN-V4 Phase 1: per-process Seed → NodeId(u32) intern table. … shrinking the dominant per-edge RAM cost by 8× on Vec content.` | **Doc update — critical:** widening to u64 cuts the per-edge RAM saving from 8× (32 → 4 bytes) to 4× (32 → 8 bytes). Still a win, but no longer 8×. Update comment to reflect actual ratio post-widening. Also see §6 SURPRISE-1 below. |
| 328 | STRUCT_FIELD | `node_intern: crate::node_intern::NodeIntern` | No change. |
| 329-332 | DOC + STRUCT_FIELD | `outgoing: HashMap<Seed, Vec<crate::node_intern::NodeId>>` | Type widens transparently. **RAM impact:** Vec<u32> → Vec<u64> doubles content size. At 18.6M bonds and ~1.5 avg out-degree, outgoing adjacency holds ~28M NodeIds = +112 MB resident on Hetzner. Acceptable, but should be noted in Risk Register §5 "NodeIntern memory growth" entry. |
| 333-334 | STRUCT_FIELD | `incoming: HashMap<Seed, Vec<crate::node_intern::NodeId>>` | Same as 332. |
| 822-826 | API_SURFACE | `pub fn node_intern(&self) -> &NodeIntern` | No change. |
| 996-1015 | (method body) | `intern_or_get(from)` + `intern_or_get(to)` + pushes into adjacency Vecs | All type-inferred; no source change. |
| 1015-1030 | (method body) | sort by reverse-lookup Seed bytes for determinism | No change. |
| 1476-1485 | (method body) | Same pattern as 996-1015 in `record_bond` path. | No change. |
| 1729-1741 | (method body) | `node_intern.get(from)` returning `Option<NodeId>` for retain match. | No change. |
| 1793-1845 | (method body) | adjacency cleanup on subject purge. | No change. |
| 3019-3036 | (method body) | `reverse_collect(v.iter().copied())` | No change. |

### 2.4 `src/sst.rs` — zero NodeId newtype use

All 5 grep hits at lines 62, 81, 82, 93, 106 are doc-comments describing **block-local `u32` indices** that are NOT the `NodeId` newtype. These are raw `u32`s in `CompactBondRaw` at `sst.rs:81-82` and the per-block `subject_table`. Per design doc §3 "NOT salvageable" list, all of `sst.rs` is rewritten in Phase 3 with brain-wide u64 NodeIds in `CompactBond` (Q2). **Phase 1 does not touch this file.**

### 2.5 `src/wal.rs` — zero NodeId newtype use

The single grep hit at line 6 is a doc-comment stating WAL deliberately does NOT use NodeIds. WAL stores raw Seeds. Phase 1 work in `wal.rs` is adding `REC_TYPE_NODE_INTERN=2` (Q8) — that record type WILL carry NodeIds in its payload, and that addition is the WIRE_FORMAT introduction (counted in §3 below, not in §1 since the code doesn't exist yet).

### 2.6 `crates/lu_brain_bench/` and `scripts/*.rs` — zero NodeId newtype use

Bench code imports `NodeIntern` (the table) at `bench_flush.rs:26` and `bench_memtable.rs:22` and calls `NodeIntern::new()`. Returned NodeIds flow through type-inferred bindings; widening transparent.

## 3. WIRE_FORMAT deep-dive

**Current state: zero on-disk encodings of the NodeId newtype exist in the codebase at HEAD `1f2addb`.** Phase 1 introduces the first one (Q8 WAL `REC_TYPE_NODE_INTERN=2`); Phase 2 introduces NodeIntern's `nodeintern.bin` sidecar (Q3); Phase 3 introduces the new CompactBond fixed-width 24-byte format (Q2). Each is a fresh design, not a widening of an existing on-disk struct.

| Encoding | File | Status today | Widening verdict | Design-doc tag |
|---|---|---|---|---|
| v3 brain.lbr archive (bond_index `HashMap<(Seed,Seed), u32>`) | `src/archive.rs:524`, `src/compress.rs:1931` | Live. **u32 here is the archive-internal node index, NOT NodeIntern's NodeId** (different concept — see §4). | **No widening needed.** This u32 stays. Migration tool re-interns into v5's u64 NodeIds at boundary. | Q4 (migration) |
| v3 WAL records (`REC_TYPE_BOND=0`, `REC_TYPE_NODE=1`) | `src/wal.rs:168-169` | Live. Carries raw Seeds, no NodeId. | **No widening needed.** WAL keeps Seed-keyed; new `REC_TYPE_NODE_INTERN=2` is the only NodeId-carrying record and it ships born-u64 (varint per §2 Q1). | Q8 |
| v4 SST `CompactBond` (14-byte block-local) | `src/sst.rs:78-87` | Live in v4 branch (not production — v3 is production). u32 fields are block-local subject_table indices, NOT NodeIntern NodeIds. | **Full rewrite per design doc §3 / Q2** — replaced with fixed-width 24-byte (8+8+8) using brain-wide u64 NodeIds. No widening of existing field. | Q2 |
| v4 manifest (`ManifestPayload` rkyv blob) | `src/manifest.rs:155-170` | Live in v4 branch. Does NOT carry NodeIntern. | No change. Q3 directive forbids embedding NodeIntern here. | Q3 |
| NodeIntern sidecar (`nodeintern.bin` inside .lbr tar) | Not yet implemented | New in Phase 2 | Fresh format — design decision is fixed-width u64 per NodeId, native-endian, length-prefixed. **No widening — born u64.** | Q3, Q6 |
| MCP JSON-RPC tool surface | `src/mcp*.rs` | Live. Tool args/returns are Seeds/strings/payloads. | **No NodeId on wire.** Verified via grep — zero NodeId mentions in mcp*.rs files. | n/a |

**Conclusion: there are no wire-format breaks induced by the Phase 1 u32→u64 widening of the NodeId newtype.** The new on-disk formats that DO carry NodeIds (Phase 1 WAL extension, Phase 2 sidecar, Phase 3 bond SST) are designed-from-scratch in Phase 2/3 and ship born-u64.

## 4. v3-archive ↔ NodeIntern boundary (the migration bridge)

The Phase 0.a audit caught a subtle terminology overlap: **the v3 archive's `bond_index: HashMap<(Seed,Seed), u32>` (at `archive.rs:524` and `compress.rs:1931`) uses `u32` but this is NOT a `NodeIntern::NodeId`.** It is the **archive-internal node index** — a per-archive integer assigned during v3 rkyv archive construction, used to compact the bond list. It has no relationship to the runtime `NodeIntern` table.

The handoff between the two systems happens at **`src/network.rs:999-1002`** in `AssociativeNetwork::from_archive` (the load path):

```rust
for (from, to) in archive.iter_bond_keys() {
    let from_id = net.node_intern.intern_or_get(from);  // <-- archive u32 → NodeIntern u32 (will be u64 post-Phase 1)
    let to_id = net.node_intern.intern_or_get(to);
    net.outgoing.entry(from).or_insert_with(Vec::new).push(to_id);
    net.incoming.entry(to).or_insert_with(Vec::new).push(from_id);
}
```

Note that `iter_bond_keys` returns `(Seed, Seed)` — the archive-internal u32 is **already resolved to Seeds before the boundary**. The handoff is **Seed → NodeIntern**, never u32 → u32. **This boundary is u32-clean today and stays widening-clean.** Post-Phase 1 the only change is that `from_id`/`to_id` carry u64 instead of u32, which propagates transparently.

For the Phase 7 migration tool `lu_brain_migrate_v3_v5` (Q4) the same boundary applies: read v3 archive → emit (Seed, Seed) pairs → v5 ingest → NodeIntern interns → emit u64 NodeIds into bond SST. **No u32 → u64 conversion is ever needed at the boundary** because the archive-internal u32 never escapes the v3 read path.

## 5. Phase 1 sub-port sequencing recommendation

Given (a) the tight blast radius (2 files outside declaration), (b) zero existing wire formats touched, and (c) no external crate exposure of NodeId, the canonical "widen declaration first, then ripple" sequence flattens to a single atomic source change. But Phase 1 in the design doc bundles three things together: widening + WAL `REC_TYPE_NODE_INTERN=2` (Q8) + migration import from v3 SeedRegistry. Recommended split:

### Phase 1.1 — Widen `NodeId` declaration u32 → u64 (atomic source change)

**Scope:** single commit, code-only, no on-disk format change.

**Touch list:**
- `src/node_intern.rs:71` — change `pub struct NodeId(pub u32);` to `pub struct NodeId(pub u64);`
- `src/node_intern.rs:74-76` — update `as_u32` method (recommend: drop and replace with `as_u64`; verify zero external callers first — grep confirms zero today)
- `src/node_intern.rs:133` — delete `debug_assert!(raw <= u32::MAX as usize, …)`
- `src/node_intern.rs:134` — change `NodeId(raw as u32)` to `NodeId(raw as u64)`
- `src/node_intern.rs:293` (test) — change `let _: u32 = id.0;` to `let _: u64 = id.0;`
- All doc-comments in `node_intern.rs`, `memtable.rs:5-6`, `network.rs:320-321,329-334` updated to say u64 and note adjacency RAM ratio drops from 8× to 4× (still a win vs raw Seed; see SURPRISE-1).
- All `id.0 as usize` sites (`node_intern.rs:144,152,174`) keep working on 64-bit; optionally add an explicit `if id.0 > usize::MAX as u64 { return None; }` guard at line 142 for defensive read paths.

**Gate:** `cargo check` clean + `cargo test --lib node_intern` green + bench harness `cargo check` clean (verifies bench crate flows through).

**Bounded-RSS assertion:** Phase 0.c bench `bench_memtable` measured against the 5M-record Phase 0.b fixture — RSS delta vs pre-widening must be ≤ +4 bytes × adjacency_edge_count, with no inflation in subject HashMap memory. Use the Phase 0.c `assert_bounded_rss!` API.

### Phase 1.2 — WAL `REC_TYPE_NODE_INTERN=2` record (Q8)

**Scope:** new WAL record type carrying `(Seed, NodeId(u64))` pairs in varint encoding. Replay reconstructs NodeIntern before bond replay.

Depends on 1.1 (NodeId is u64 by this point — the record is born-u64).

**Bounded-RSS assertion:** WAL replay against a 100M-record synthetic WAL must keep RSS bounded — no "buffer entire WAL then sort" antipattern.

### Phase 1.3 — `intern_from_v3_archive(archive: &Archive)` helper on NodeIntern

**Scope:** single-pass streaming intern of every Seed in the v3 archive's bond_index, used by both the runtime `from_archive` load path (`network.rs:999`) and the Phase 7 migration tool. Refactors the existing inline loop into a NodeIntern method so behavior is shared and testable.

**Gate:** behavior-equivalent diff against current `from_archive` (post-Phase 1.1 build) on a sampled v3 brain — assert identical NodeId counts and identical reverse_lookup outputs.

### Alternative sequencing considered (rejected)

- **"Wire format first, declaration last"** — rejected because no wire format currently exists for NodeId; Phase 1.2's WAL record is gated on the declaration being u64.
- **"Widen STRUCT_FIELDs first, declaration last"** — Rust does not permit this. The fields' element type must match the declaration. Widening must lead.
- **"Single mega-commit for 1.1+1.2+1.3"** — rejected because the WAL record-format work (1.2) and the migration helper (1.3) carry independent failure modes from the widening (1.1). A mega-commit blends them and makes git-bisect harder if a subsequent integration test fails.

**Rationale for the chosen split:** the declaration widening (1.1) is so tightly scoped that it can ship and bake as its own commit; subsequent ports build on a known-good u64 baseline. This matches the design doc's `feedback_gotchas_now.md` discipline — fold tightly-scoped gotchas in, but don't bundle independent failure modes.

## 6. Surprises requiring design-doc §7 entries

### SURPRISE-1 — Adjacency RAM saving drops from 8× to 4×

The Phase 1 (v4) adjacency Vec layout at `network.rs:329-334` was designed for u32 NodeId: Vec<u32> beats Vec<Seed> by 32/4 = 8×. Widening to u64 cuts that to 32/8 = 4×. Still a positive saving vs raw Seeds, but the original Phase 1 design memo's "8× win" callout becomes inaccurate. Concrete impact: at 18.6M bonds × ~1.5 avg out-degree ≈ 28M adjacency entries, the widening adds **+112 MB resident** on Hetzner. Inside AX41-NVMe 64 GB budget but worth noting in Risk Register §5 "NodeIntern memory growth" entry alongside the 100M-node 2.4 GB sidecar projection.

**Recommended §7 entry:** "Q1 widening recomputes Phase 1 adjacency saving from 8× → 4× vs raw Seeds. Net delta +112 MB resident at current 18.6M-bond brain. Still bounded; revisit if adjacency RAM becomes a top-3 line item in RSS profile."

### SURPRISE-2 — `node_intern.rs:133` u32::MAX debug_assert vanishes; no replacement is meaningful

The debug_assert was the only enforcement preventing silent NodeId wraparound. Post-widening, u64 wraparound at 18 quintillion is not a real-world failure mode, so no replacement assert is needed. Worth a one-line §7 entry to document that the safety net is gone by design.

### SURPRISE-3 — `NodeId::as_u32` method has zero external callers

The method at `node_intern.rs:75-76` was exposed defensively in Phase 1 (v4) for callers that might want raw u32 access. Grep confirms zero callers in src/, crates/, or scripts/. Recommend dropping the method outright in Phase 1.1 rather than keeping a truncating `as_u32` that could lose precision post-widening. Documented as Open Question §7 below — minor decision.

### Non-surprises explicitly checked and CLEAR

- No `NodeId::MAX` or `u32::MAX` sentinel usage outside the one debug_assert (verified by `grep -rn "u32::MAX\|NodeId::MAX"` in the 5 files).
- No bit-packing of two NodeIds into a u64 record (verified — no `(id_a as u64) << 32 | id_b as u64` patterns).
- No hash function assuming 4-byte input (NodeId uses derived `Hash`, which delegates to u32's hasher today and will delegate to u64's hasher post-widening — both standard).
- No HashMap-the-world keyed on NodeId outside the bounded MemTable BTreeMap (which is RAM-capped at 500 MiB by design, not by NodeId width).
- No v4-substrate-only sites that should NOT widen — every NodeId site is either runtime (kept) or doc-comment in a file scheduled for full rewrite (sst.rs, wal.rs's nodeid-mentions are doc-only and rewritten with the rest of those files in Phase 2/3).

## 7. Open questions for John (Phase 1.0 → Phase 1.1 gate)

**Q1.0.A — `NodeId::as_u32` disposition.** Drop the method (cleanest, zero callers) or keep as truncating helper for debug logs / display? Recommendation: drop and add `as_u64`. (Recommendation alone is enough — confirming so I can spawn 1.1 without a back-and-forth.)

**Q1.0.B — `id.0 as usize` defensive guards.** Current code does naked `id.0 as usize` for Vec indexing. On 64-bit Linux this is a no-op widening today (u32 → usize) but post-widening becomes a no-op same-width cast (u64 → usize on 64-bit). On 32-bit targets (not deployed — but the Cargo.toml doesn't pin target) this becomes a potentially-lossy truncation. Recommendation: add an explicit guard at `reverse_lookup` (the only public entry point that takes an externally-supplied NodeId) and document the assumption "64-bit target only" in the module's `//!` header. Confirm or override?

**Q1.0.C — Phase 1.1 commit message / tag.** Recommend tag `pre-nodeid-widen-2026-06-09` on current HEAD `1f2addb` BEFORE the Phase 1.1 commit, so a regression discovered later can git-bisect cleanly across the widening boundary. Approve?

## 8. Method notes — what I checked

- `grep -rn "NodeId" src/ crates/ scripts/` → 74 hits, 5 files. Confirmed no aliases (`type Foo = NodeId`), no parallel newtypes (`InternId`, `GlobalNodeId`).
- `grep -rn "NodeIntern\b"` → 46 hits outside declaration (table touches; needed for blast-radius reasoning even though no source change required).
- `grep -rn "intern_or_get\|reverse_lookup\|reverse_unchecked\|reverse_collect\|node_intern.get("` → enumerated every NodeIntern method call site to confirm none capture into typed-let bindings that would force source change.
- `grep -rn "u32::MAX\|NodeId::MAX"` → confirmed single relevant hit at `node_intern.rs:133`.
- `grep -rn "NodeId" crates/ scripts/` → confirmed bench code uses NodeIntern but never NodeId. Scripts/*.rs have zero NodeId mentions.
- `grep -rn "NodeId" src/mcp*.rs` → zero hits. MCP tool surface does not expose NodeId.
- Walked each non-trivial file region by sed line ranges to verify the categorization (memtable.rs 40-135, 177-330, 390-470; network.rs 315-340, 990-1030, 1470-1485, 1725-1740, 1790-1845, 3015-3025; node_intern.rs 60-180, 215-294; sst.rs 55-115; wal.rs 1-30).

_— audit complete —_
