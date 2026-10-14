# Lu — Faculty Reconnection Status + Ready-for-Live-Test Checklist

**Date:** 2026-07-24 · **Branch:** `main` (worktree `merge-main`), one linear line of
commits atop `f6a57d7`. **Serving/answer-path wiring only — no retrain touched;
`lu4d-cat1-ingest` ran undisturbed throughout.** Companion to
`docs/LIVE_PATH_WIRING_MAP.md` (#1352), `docs/CAPABILITY_REGISTRY.md` (#1353),
internal notes. All guest work unit-tested in isolation; LIVE end-to-end
validation on the real 233 GiB brain is DEFERRED to training completion.

## Commit line (local main, not pushed)

```
9cd998c OP-OFFNESS-1 port: offness.rs + embedding.rs onto main (module port)
cbd7867 OP-ROUTE-GUEST-6: Q.3 conversational/session memory into guest path
63b1c49 docs(curiosity): Faculty 7 fill-loop enable steps + stale-config fixes
36c332b OP-ROUTE-GUEST-5: act on router references_file_path/image_path
97b53aa OP-ROUTE-GUEST-4: Left-Brain reachability on guest (use_multi_think)
cea3c6f OP-ROUTE-GUEST-3: inspect-only file_read + image_describe on guest
60e779e OP-ROUTE-GUEST-2: full math cascade into guest_ask
```

## Per-item status

| # | Item | State | Evidence |
|---|------|-------|----------|
| 1 | Guest re-exposure (read/compute) | **PARTIAL — core done** | full math cascade (60e779e), file_read + image_describe inspect tools (cea3c6f); privsep test extended. Remaining as guest TOOLS: research fetch, style/sense/time read (see Deferred). |
| 2 | Session/context memory (Q.3) | **DONE + tested** | cbd7867 — backref + pronoun/elaboration rewrite + turn-write in guest_ask; 3 unit tests |
| 3 | Left Brain reachability | **DONE + tested** | 97b53aa — guest_wants_multi_think heuristic; 2 unit tests |
| 4 | Confidence resolution | **SCOPED (not shipped)** | deep v5 read-path change; plan below |
| 5 | Router agentic file/image | **DONE + tested** | 36c332b — acts on references_file/image_path; 2 unit tests |
| 6 | Curiosity fill loop | **DONE (docs/ops)** | 63b1c49 — producer already wired via drop-in; worker enable steps + 2 stale-config fixes documented |
| 7a | Cattle Dog / off-ness | **PORTED + tested; consult deferred** | 9cd998c — offness.rs + embedding.rs compile on main, 16 tests pass; answer-path consult wiring deferred |
| 7b | Code generation | **GENUINELY ABSENT — scoped** | roadmap only (#5239); see below |
| 7c | Image→geometry loop | **GENUINELY ABSENT — scoped** | solver orphaned; needs perception; see below |

## What compiled cleanly into the GUEST binary (privsep preserved)

- New guest tools: `lu4d_file_read` (inspect-only), `lu4d_image_describe`; plus
  the full math cascade and session-memory pre/post-pass inside `lu4d_ask`.
- **Privsep guarantee intact:** the string-scan test (`tests/privsep.rs`, extended)
  confirms NO mutating tool NAME (`lu4d_teach`, `lu4d_image_ingest`,
  `lu4d_file_extract_to_pending`, `lu4d_scar`, `lu4d_trust_set`,
  `lu4d_approve_pending`, ...) is present in the guest binary, while the new read
  tools ARE. Mutating paths stay physically compiled out (their tool-name
  literals live only in the gated `mcp::tool_definitions()`), and the guest brain
  is `Brain::load_readonly` (rejects writable). Session turn-writes go to the
  in-memory `session_memories` registry, never nodes/bonds.

## READY-FOR-LIVE-TEST battery (run when training completes + guest loads real brain)

1. **Math / word problem:** `convert 5 km to miles`, `mean of 2,4,6`, `solve x+5=10`,
   `area of a 3-4-5 triangle` (word-problem), `2*(3+4)` → all answer from the
   primitive with provenance, NOT idk. (unit-tested: units/stats/solver route)
2. **Session continuity:** start session → `what is a dog` → `tell me more`
   (elaborates prior) → `tell me about it` (pronoun→anchor) → `what was my first
   question?` (returns verbatim). (unit-tested against ephemeral brain)
3. **File query:** `summarize ./notes.md` (or `lu4d_file_read {path}`) → returns
   file text, inspect-only, bounded by allowed_read_paths.
4. **Image query:** `describe photo.png` (or `lu4d_image_describe {path}`) →
   description + triplets, forms no bonds (needs a real extractor configured; Stub
   otherwise).
5. **Definition:** `what is a dog` → grounded WordNet/GCIDE gloss (Tier 0) +
   neighbours; multi_think engaged (reasoning depth visible in structured trace).
6. **Reasoning depth:** `why is the sky blue` / `how does photosynthesis work` →
   ReAct thinking fires (thinking summary present).
7. **Confidence surfaced:** inspect `confidence_label` — EXPECT it still reads from
   the Unverified stub until item 4 ships (see below). This is the one battery
   item known to still be degraded.
8. **Nonsense refuses:** `what is the president of the moon` → idk refusal, no
   fabrication.

## Item 4 — Confidence resolution: concrete deferred plan (NOT shipped, by discipline)

Root cause: on any loaded v5 brain every node is a rehydration STUB
(`brain.rs:11696` → `Confidence::Unverified`; `network.rs:83` `SeamNode::Stub`),
so `narration.rs:1785` computes answer confidence from `Unverified` (0.5) for
EVERY answer. Clean mirror of OP-LABEL-1 (which resolved *content*):

1. `network.rs`: add a `conf_cache` LRU + `resolve_confidence_from_store(seed)`
   (same `get_node_v5` call `resolve_label_from_store` already makes — the node it
   returns carries `confidence()` right beside `content()`), and a pub
   `node_confidence(seed) -> Option<Confidence>` (resident → cache → store),
   exactly paralleling `node_label`.
2. `narration.rs:1785`: for the TOP activated node only (render-time, cheap; do
   NOT touch the hot spread seam — that perf design deliberately avoids store
   reads), when cache-backed and the resident is a stub, use the resolved
   confidence instead of the stub value.
3. Test: build a v5 node_store SST fixture (pattern already in `brain.rs:8017`)
   with a Verified node; load read-only; assert resolved confidence == Verified,
   not Unverified.

Deferred rather than rushed because it is a core read-path change across multiple
Network constructors that CANNOT be live-validated this session, and a subtle
regression would mislabel every answer (high blast radius). Medium size; the
pattern is proven, so it is low conceptual risk once done carefully.

## Item 7b — Code generation: ABSENT (scope)

Only `query_router.code_intent` (an intent slot, `query_router.rs:72`, comment
"Phase 6 will wire code generation here") exists; nothing consumes it and there is
NO generator. This is a genuine from-scratch feature (roadmap #5239), too large
for this window. Not started, by design. When built, it plugs into the same
router flag the way file/image now do (36c332b).

## Item 7c — Image→geometry loop: ABSENT (scope)

The exact geometry solver (`math/geometry.rs`, `a+b·π`, partition-verify) is real
but ORPHANED — nothing calls it. The missing half is PERCEPTION: an
image→Scene→solve pipeline that feeds the solver. `image_describe` (now on the
guest) produces triplets but no Scene topology for the solver. Building the
perception adapter (triplets → geometry `Scene`) is the real work; the solver end
is ready. Scoped, not started.

## Deferred remainder of item 1 (straightforward follow-ups, same pattern as file/image)

- `lu4d_time_status`, style-read, sense-read as explicit guest tools: pure read,
  low risk; add guest dispatch arms delegating to the existing read handlers.
- `research` fetch-and-answer: guest-safe for the graph (bond-count-unchanged
  invariant) but carries a separate **network-egress / SSRF** decision (#1356
  non-graph risks) — gate egress before exposing on a public surface.
- Physical compile-out of mutating HANDLER bodies (not just their tool-name
  literals): `tool_image_ingest`, file_extract_to_pending, research apply_proposal
  etc. are linked-but-unreachable today (same as pre-change main). Fully
  `#[cfg(feature="admin_mcp")]`-gating them + their tests is defense-in-depth
  beyond the current guarantee; larger mechanical refactor.
