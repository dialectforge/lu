# Lu — Live-Path Faculty Wiring Map

**Branch:** `main` @ `2ab23eb` · **Tree:** `/opt/lu/lu/merge-main` (worktree of `/opt/lu/lu/v4/.git`) · **Date:** 2026-07-23
**Scope:** read-only trace of both ask entry points through every faculty. No code/brain/training touched.
**Retrain state during trace:** `lu4d-cat1-ingest` active (undisturbed). `lu-curiosity-worker` inactive.

---

## TL;DR — the fracture in one paragraph

Before **OP-007 (privsep, commit `85c593f`, 2026-07-18)** Lu was **one binary** (`lu_4_mcp`) exposing ~40 MCP tools, each of which wired a faculty directly onto the surface a user hit. That is the "good-responses version" John remembers. The privsep split created **two binaries**: an admin binary (`lu_admin_mcp`, built with the default `admin_mcp` cargo feature — all tools) and a **public/guest binary** (`lu_guest_http`, built `default-features = false`, which *physically compiles the entire `src/mcp.rs` module out*). The guest surface exposes **only** `lu4d_ask` + `session_start/end`, and `lu4d_ask` routes through a **single** call: `Brain::narrate(query, depth, top_k, false)`. Every faculty whose positive handler lived in `src/mcp.rs` (full math, the Q.3 conversational-memory pre-pass, the explicit think/reason tools) was **stranded off the live path** — not deleted, just not reachable from `narrate()`. At the time of the split `narrate()` consumed the query router **observation-only**, so even the router couldn't switch faculties on. Since then (**OP-ROUTE-3a/3b/3c** and today's **OP-ROUTE-GUEST**, 2026-07-23) reasoning, variants, critic-vetting, agentic-thinking and arithmetic have been re-wired into `narrate()`'s PASS path one at a time. What is **still** stranded on the guest live path: **context/session memory (Q.3)**, **full math (`detect_math`)**, and a running **curiosity fill worker**; **confidence** is globally degraded to `Unverified` on any loaded v5 brain; and **off-ness / Cattle Dog never existed on `main` at all**.

**Hypothesis "good version = full binary, guest = crippled": CONFIRMED, with a date and a commit.** The crippling event is the `default-features = false` guest build at `85c593f`. The recovery is partially done and traceable in git.

---

## The two entry points

### GUEST live path (what a real user hits)
`crates/lu_guest_http/src/main.rs:104` loads the brain **read-only** (`Brain::load_readonly`), rejects a writable brain, optionally attaches a curiosity `WonderSidecar` (only if `--wonder-queue` is passed). Requests enter at:

`crates/lu_guest_http/src/guest_mcp.rs:70` `dispatch` → three tools only:
- `lu4d_ask` → **`guest_ask`** (`guest_mcp.rs:107`)
  1. `guest_mcp.rs:129` — `lu4d::math::detect_arithmetic` short-circuit (wired **today**, commit `439e2e8`)
  2. `guest_mcp.rs:160` — `greet_refusal::is_greeting` guard
  3. `guest_mcp.rs:175` — **`brain.narrate(query, depth, top_k, false)`** ← the entire cognitive surface
- `lu4d_session_start` / `lu4d_session_end` → `brain.session_start()/session_end()` (touch the lb_session store, but see Faculty 6)

### ADMIN path (dev/operator surface)
`src/mcp.rs` `dispatch` (~line 1393) → ~40 tools incl. `lu4d_ask` (`tool_ask`, 2157), `lu4d_narrate` (3243), `lu4d_reason` (3708), `lu4d_think`/`lu4d_multi_think`, `lu4d_wonder`, `lu4d_research*`, `lu4d_variants*`, and the Left-Brain `lu4d_session_memory_*` tools (`src/mcp_lb.rs`). Built with `default = [admin_mcp]`; **compiled out of the guest binary entirely.**

### The narrate() pipeline (the guest's whole world) — `src/narration.rs`
`narrate` (2863) → `near_miss::repair_query` (2871) → `narrate_inner` (2892):
greet refusal (2911) → `IntentClassifier::classify` (2960) → `QueryRouter::route` (2963) → SelfIdentity/CurrentInterests/MetaSelf short-circuit (2988) → **`brain.ask_with_config`** = seed + spread + recall (3086) → **idk_gate** refusal floor (3092) → *PASS path*: `multi_think` (3149), `reasoning_trace` (3160), `variant_arena` (3183), `thinking_summary` (3211) → `ModulationFlags::compute` → `Renderer::render` (16 shapes) + style hints.

---

## Faculty wiring table

Legend: 🟢 **LIVE** · 🟡 **CONDITIONAL** (wired but gated behind a heuristic/floor) · 🔴 **DARK** (present but never invoked on this path) · ⚫ **ABSENT** (not in `main`).

| # | Faculty | Entry (module:line) | Admin | Guest | What gates it | Failure mode when dark | Reconnect (size/risk) |
|---|---------|---------------------|-------|-------|---------------|------------------------|------------------------|
| 1 | **Seed selection + spread activation** | `brain.ask_with_config` → `network.rs` spread; `narration.rs:3086` | 🟢 | 🟢 | core, ungated | (n/a) | — |
| 2 | **idk / refusal floor** | `idk_gate.rs` `fast_pass_check`/`evaluate_signals`/`classify`; `narration.rs:3092` | 🟢 | 🟢 | `cfg.idk_gate_enabled` (default true) | confident fabricated "X is a Y" on unknowns | already wired (OP-014) |
| 3 | **recall → narration graph-walker** | `narration.rs:1086` `collect_outgoing` + `network.rs:1149` `node_label_or_empty` | 🟢 | 🟢 | node_store label resolver + read-only-reader residency (OP-LABEL-1/2) | `" is ."` empty nouns; 5× stopword mis-weighting (was #1315) | wired; **residency-dependent** — verify LRU/node_store resident on RO reader |
| 4 | **reasoning.rs (procedural / list / comparison)** | `reasoning.rs:264` `reason_with_config` (Procedural 308, ThinList 327, Comparison 351) | 🟢 | 🟡 | refusal path: always; PASS path: `routing.needs_deep_reasoning && !math && !meta` (`narration.rs:3160`) | shallow answer, no why-chain / no enumeration | wired OP-ROUTE-3a; broaden trigger heuristics |
| 5 | **Left Brain / thinking (Coordinator–Planner ReAct)** | `thinking.rs` `Brain::think`/`multi_think`; consumed `narration.rs:3149` (critic) & **`3211`** (agentic) | 🟢 (explicit `lu4d_think`/`lu4d_multi_think`) | 🟡 | `use_multi_think` **hardcoded `false`** on guest; else `needs_critic_vetting` OR `needs_agentic_thinking` + `cfg.thinking_min_activation` floor | no multi-step reasoning — **John's felt "lacking Left Brain"** | wired OP-ROUTE-3c; raise reachability (see checklist #2) |
| 6 | **Context / session memory (Q.3)** | store `lb_session/` (mod.rs 1097 LOC); Q.3 pre-pass in `mcp.rs` `tool_ask`/`tool_narrate` (`session_id` arg) | 🟢 | 🔴 **DARK** | `narrate()` signature has **no `session_id`**; Q.3 rewrite+turn-write live in the compiled-out `mcp` module | "tell me more" / pronouns / "what did I ask?" all fail; zero continuity | **thread `session_id` through `narrate`/`narrate_inner` + port Q.3 pre-pass into `guest_ask`.** Medium–large |
| 7 | **Curiosity producer + fill worker** | producer: `brain` detection + `guest main.rs:110` `WonderSidecar`; fill: `src/bin/lu_curiosity_worker.rs` (separate process) | 🟢 producer / worker separate | 🟡 producer, 🔴 fill | producer needs guest launched with `--wonder-queue`; fill worker = `lu-curiosity-worker.service` **currently inactive** | gaps detected but never drained; Lu never grows from what it wonders | start the service + launch guest with `--wonder-queue`. Small (ops) |
| 8 | **Math: word-problem / geometry / units / stats** | `math.rs:177` `detect_arithmetic` (guest) vs `math.rs:516` `detect_math` (stats 548 / units 565 / geometry / `math/wordprob.rs`), called only at **`mcp.rs:1641`** | 🟢 (full `detect_math`) | 🔴 **arithmetic only** | guest replicated only the `detect_arithmetic` subset; `detect_math` lives in compiled-out `mcp` | "convert 5 km to miles", "area of circle r=3", word problems → recall → idk refusal | **call `lu4d::math::detect_math` in `guest_ask`** (`math` is ungated, already linked). **Small / low-risk** |
| 9 | **Off-ness / Cattle Dog** | `offness.rs` — **not present** (only on branch `feature/op-offness-1-cattledog-port`) | ⚫ | ⚫ **ABSENT** | never merged to `main` | no off-topic / nonsense herding | merge/port the branch. Large (genuine feature port) |
| 10 | **Confidence model** | `brain.rs:~11640` `rehydration_node_stub` → `Confidence::Unverified`; `network.rs:83` `SeamNode::Stub` → `Unverified` | 🟡 degraded | 🟡 degraded | any **loaded v5 brain** returns content-free/Unverified stubs unless resident | every answer discloses "Unverified"; confidence-modulated shapes flatten | resolve confidence from node_store payload (mirror OP-LABEL-1 for content). Medium |
| 11 | **Query router branches** | `query_router.rs:28` `RoutingDecision`: `needs_math`(33) `needs_deep_reasoning`(45) `needs_variants`(49) `needs_critic_vetting`(53) `needs_agentic_thinking`(68) `is_meta_query`(77) | 🟢 | 🟡 all now consumed | historically observation-only (OP-ROUTE-2); consumption added OP-ROUTE-3a/b/c + GUEST | queries route to recall-only, faculties never switch on | mostly done; tune detector precision |

---

## Where the Left Brain and context memory actually are

**Left Brain (the "thinking layer" / Coordinator-Planner ReAct loop) = `src/thinking.rs`.** Header: *"A ReAct-style Observe → Plan → Act → Reflect loop layered on the temporal graph. One user call to `think()` runs one synchronous pass."* It is a `Brain` method (`Brain::think` / `Brain::multi_think`), **ungated and linked into the guest binary**. It is invoked from `narrate()` in two places, both on the PASS path (post-refusal): `narration.rs:3149` (`run_multi_think = use_multi_think || routing.needs_critic_vetting`) and `narration.rs:3211` (`needs_agentic_thinking` + activation floor). **So it is not absent and not fully dark — it is CONDITIONAL.** The reason John feels it's missing: the guest passes `use_multi_think = false`, so the layer *only* fires when the router's `needs_critic_vetting` ("is it true that…", "prove it") or `needs_agentic_thinking` ("how do I … step by step") heuristics match **and** the goal seed clears `cfg.thinking_min_activation`. For a plain "tell me about X", none of that trips, so Lu answers from recall alone and feels shallow.

**Context / session memory = `src/lb_session/` (1,747 LOC; the project-log-style session layer).** The **store** is owned by `Brain` (`brain.lb_session`) and is reachable on guest via `session_start`/`session_end`. But the **conversational pre-pass** — the "Q.3" logic that resolves pronouns, handles "tell me more", and answers "what was my first question?", then writes the turn back — lives in the **admin** `tool_ask`/`tool_narrate` handlers (`src/mcp.rs`, keyed on a `session_id` argument). `Brain::narrate` **takes no `session_id`** (`brain.rs:5780`), and `narrate_inner` hardcodes `session_id: None` in the `AskResult` it builds. **So on the guest path, session memory is a store with no reader/writer wired to it — genuinely DARK.** This is the single clearest "faculty stranded by the split," and it is exactly the gap John named.

---

## Reconnection checklist — priority order

1. **Full math on guest (Faculty 8).** *Smallest win, highest ratio.* In `guest_ask`, after the `detect_arithmetic` block, add a `lu4d::math::detect_math(query)` branch (the module is ungated and already linked). Restores units, stats, geometry, and word problems. ~15 LOC, low risk, mirrors the admin `try_arithmetic_response_with` fuller path at `mcp.rs:1641`.
2. **Left Brain reachability (Faculty 5).** *Medium.* Either (a) let the guest opt into `use_multi_think` for non-trivial queries, or (b) lower/tune `thinking_min_activation` and broaden `needs_agentic_thinking`. The machinery is wired; the thresholds are what keep it dark on ordinary queries.
3. **Context / session memory (Faculty 6).** *Medium–large, highest felt value.* Thread `session_id` through `Brain::narrate` → `narrate_inner`, and port the Q.3 pre-pass (elaboration / pronoun / backreference rewrite + post-turn write) out of `mcp.rs` into `narrate()` (or into `guest_ask` ahead of the narrate call). This is the biggest lift but resolves John's #1 complaint about continuity.
4. **Curiosity fill loop (Faculty 7).** *Small / ops.* Launch the guest with `--wonder-queue <dir>` and re-enable `lu-curiosity-worker.service` so detected gaps are drained.
5. **Confidence resolution on loaded brains (Faculty 10).** *Medium.* Resolve `Confidence` from the node_store payload the way OP-LABEL-1 resolved *content*, so answers stop uniformly disclosing "Unverified." Affects admin and guest alike.
6. **Off-ness / Cattle Dog (Faculty 9).** *Large.* Genuine feature port from `feature/op-offness-1-cattledog-port`; not a rewiring, a merge.

---

## Honesty ledger — absent vs merely unwired

- **Genuinely ABSENT from `main`:** off-ness / Cattle Dog (`offness.rs` exists only on `feature/op-offness-1-cattledog-port`).
- **Present but DARK on the guest live path:** Q.3 session-memory pre-pass; full `detect_math`; curiosity fill (worker inactive).
- **Present and CONDITIONAL on guest (wired, gated):** Left Brain thinking, reasoning chains, variants, critic vetting — all reachable via `narrate()` PASS-path router branches, but behind heuristics + activation floors, and with `use_multi_think` hardcoded false.
- **Globally DEGRADED (both binaries):** confidence (`Unverified` stub on loaded v5 brains).
- **Already LIVE on guest:** seed/spread recall, idk refusal floor, recall→narration graph-walker (residency-dependent), greet refusal, arithmetic, self-identity/meta short-circuits, 16-shape renderer.

---

## Provenance (all read-only, this session)
- Guest entry: `crates/lu_guest_http/src/{main.rs,guest_mcp.rs}`
- Narrate pipeline: `src/narration.rs` (2863–3245, 1086–1130)
- Router: `src/query_router.rs` (28–170)
- Faculties: `src/{idk_gate.rs, reasoning.rs, thinking.rs, curiosity.rs, math.rs, math/*, confidence.rs, network.rs, brain.rs}`, `src/lb_session/`, `src/mcp_lb.rs`, `src/bin/lu_curiosity_worker.rs`
- Regression root cause: `git show 85c593f` (OP-007 privsep, 2026-07-18) + `git log` OP-ROUTE-GUEST (`439e2e8`,`1734a17`,`b786356`, 2026-07-23), all in `main` @ `2ab23eb`.
