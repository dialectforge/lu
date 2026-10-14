# Lu — Capability Registry (Master Census)

**Branch:** `main` @ `d51d62f` · **Tree:** `/opt/lu/lu/merge-main` (worktree of `/opt/lu/lu/v4/.git`) · **Date:** 2026-07-23
**Scope:** read-only census of *every* capability-bearing module in the tree — not just those reachable from the ask path. No code / brain / training touched. Retrain `lu4d-cat1-ingest` active and undisturbed throughout.
**Companion:** extends `docs/LIVE_PATH_WIRING_MAP.md` (committed `d51d62f`, internal notes, which traced 11 faculties reachable from `narrate()`. This census adds the capabilities that are **NOT** reachable from the ask path — admin-only tools, orphaned subsystems, and things that live only on branches.

---

## How to read this

Every capability is in exactly one **state**:

| Badge | State | Meaning |
|-------|-------|---------|
| LIVE-GUEST | **LIVE-GUEST** | Reachable by a real user through `lu4d_ask` today. |
| CONDITIONAL | **CONDITIONAL** | Wired into the guest `narrate()` path but gated behind a heuristic / activation floor, so it fires only on some queries. |
| DARK-ADMIN | **DARK-ADMIN** | Fully built and compiled into the binary, but its only caller is the admin `src/mcp.rs` dispatch — which is **`#[cfg(feature = "admin_mcp")]`** and **compiled out of the guest binary**. Present, tested, unreachable by users. |
| ORPHANED | **ORPHANED** | The code exists and compiles, but **nothing in the production tree calls it** — not even admin. A subsystem with no caller. |
| ABSENT | **ABSENT-FROM-MAIN** | Not on `main` at all. Lives only on a feature branch, or exists only as a router *intent slot* + roadmap with no implementation. |

**The one structural fact that explains most of this:** at OP-007 (privsep, `85c593f`, 2026-07-18) Lu split into two binaries. The guest binary is built `default-features = false`, which compiles the entire `src/mcp.rs` module — the ~40-tool dispatch — **out**. Guest exposes only `lu4d_ask` + `session_start/end`, and `lu4d_ask` calls `Brain::narrate(...)` and nothing else. So the rule of thumb is: **if a capability's only entry point is a `lu4d_*` MCP tool, it is DARK on the live path**, because the whole tool surface is admin-only. The underlying *modules* (math, file_io, vision, capability_runtime) are still linked into the guest binary — they just have no caller on the guest surface.

---

## Part 1 — Plain-language accounting of every capability John named

This is the "account for EACH by name" section. For each thing John remembered, here is exactly where it lives and its honest state.

### 1. Sandbox / code-execution capability — **PRESENT on main, DARK-ADMIN, master-switch OFF**
It exists and it is real. It is the **capability-pack execution runtime**, `src/capability_runtime/mod.rs` (Port Cat.5.1). Its own header: *"Cat.5.1 ships the runtime half: load an installed pack, RE-VERIFY it … instantiate its declared tools behind an object-safe `PackTool` trait … make them callable through MCP dynamic dispatch."* The concrete executor is `SubprocessPackTool` — third-party code run across an **OS process boundary** under resource caps, default-DENY permissions, no `unsafe`. WASM is deferred; full-privilege dlopen is rejected outright.
- **Where:** `src/capability_runtime/` + MCP surface `src/mcp_cap_runtime.rs` (`lu4d_capability_pack_runtime_configure/status/load/unload`, and a `pack__*` dynamic-dispatch fallback in `mcp.rs::handle_tools_call_inner`).
- **State:** DARK-ADMIN. Reachable only via `src/mcp.rs` (compiled out of guest). Additionally gated by a **master switch that defaults OFF** plus an unsandboxed opt-in. So even on the admin binary it does nothing until explicitly configured on.
- **Not** a general "run any code" sandbox for the model's own use — it is an execution host for signed capability packs. That is the closest thing Lu has to a sandbox, and it is genuinely built.

### 2. Coding functions (code generation / code tools) — **GENUINELY ABSENT as an implementation**
There is **no code-generation capability** on `main`. What exists is only the *recognition* of a coding request: `RoutingDecision.code_intent` (`src/query_router.rs:72`), with the comment *"Phase 6 will wire code generation here."* The router can tell that a query is asking for code (it requires a code-generation verb like "write"/"generate"), sets the flag — and nothing consumes it. The roadmap for this is internal notes ("Lu Code Capability").
- **Partial exception — reading/writing code files:** `src/file_io.rs` has a `FileKind::Code` that reads and writes source files as UTF-8 text (see #5/#6 below). So Lu can *carry* code as a file; it cannot *generate* it.
- **Honest verdict:** the *intent detector* is present; the *generator* is absent (roadmap only). This is one of the two things John named that is truly not built, not merely unwired.

### 3. Image tools — **PRESENT on main, DARK-ADMIN**
`src/mcp_vision.rs` + `src/vision/` (Port R input). Three tools: `lu4d_image_ingest` (run the configured `SceneGraphExtractor`, route extracted triplets through teach/user_statement), `lu4d_image_describe` (inspect-only, human-readable), `lu4d_image_extractor_status` (diagnostic). The extractor is a pluggable trait (`SceneGraphExtractor`) with a `StubExtractor` (dev default) and a `SubprocessExtractor` (external CLI).
- **State:** DARK-ADMIN — all three tools are admin-only; the guest surface has no image entry point. The default extractor is the Stub unless a subprocess extractor is configured, so even on admin it returns nothing useful until wired to an external vision CLI.

### 4. Image geometry math system — **PRESENT on main but ORPHANED; the autonomous image→geometry loop is ABSENT from main**
Two distinct things here:
- **The geometry solver** `src/math/geometry.rs` (Port G, "verifiable geometry pipeline") is real and sophisticated: an exact `a + b·π` evaluator (π stays symbolic — answers come out `32*pi - 64`, not a float), a canonical primitive-area library (quarter-circle, circular segment, triangle, shoelace polygon), a `Scene` topology record, and a sum-to-whole **verifier** as an honesty gate. **But it is ORPHANED:** the only tree-wide reference to `geometry` outside its own module and tests is a *content bond* in a narration test (`"geometry" IsA "math"`). It is **not** in the arithmetic cascade, **not** called by `detect_math`, **not** called by vision. A fully built subsystem with zero production callers.
- **The image→scene→geometry autonomous pipeline** (perceive a diagram, build the Scene, solve exactly) that the project log learnings describe (geometry autonomous loop, 2026-06-28) lives as **Python prototypes and on the `v4` worktree**, not on `main`. On `main` the solver has no perception feeding it. So the "image geometry math system" as an end-to-end capability is ABSENT from main; only its exact-solver core is present-but-orphaned.

### 5. File tools — **PRESENT on main, DARK-ADMIN**
`src/file_io.rs` + `src/mcp_file_io.rs` (Port File-IO). Tools: `lu4d_file_read`, `lu4d_file_write`, `lu4d_file_extract_to_pending`. Files are treated as a *transport*: content flows through the existing teach / user_statement ingest paths, stamped with a `file://` provenance URI so `lu4d_revoke_by_source` still works.
- **State:** DARK-ADMIN. Compiled into the guest binary (its deps `csv`, `lopdf`/`pdf-extract` are linked into `lu_guest_http`), but the tool wrappers are reachable only through the admin dispatch.

### 6. Document tools — **SAME MODULE as file tools (Port File-IO), DARK-ADMIN**
In Lu, "document tools" are not a separate subsystem — they are the format-aware half of Port File-IO. `FileKind` = { Md, Txt, Csv, Json, **Pdf**, **Docx**, Code }.
- **READ:** md/txt verbatim, csv (pure-Rust `csv`), json (serde), **pdf** (`pdf-extract`, text only), **docx** (hand-rolled `<w:t>` extractor over `word/document.xml`), code (UTF-8).
- **WRITE:** md/txt, csv, json, **pdf** (`printpdf` single-page text), **docx** (hand-rolled OOXML emitter).
- `lu4d_file_extract_to_pending` is the structured-doc convenience (CSV/JSON → T.3 PendingLog).
- **State:** DARK-ADMIN, same as #5. So PDF and DOCX reading/writing genuinely exist and are genuinely unreachable by a user.

### 7. The three math systems — all PRESENT; distinguished below
Lu actually has **four** arithmetic/math detectors plus the orphaned geometry solver. They are distinct modules with a defined cascade order.

| System | Module / entry | What it handles | Cascade slot | State |
|--------|----------------|-----------------|--------------|-------|
| **Regular math (arithmetic)** | `src/math.rs` `detect_arithmetic` (`:177`) | single binary op, %, sqrt, comparisons, "half of" | F.1 | **LIVE-GUEST** (replicated into `guest_ask`, commit `439e2e8`) |
| **Full calculator** | `src/math.rs` `detect_math` (`:516`) | recursive-descent expressions, scientific fns, constants, **unit conversion**, **descriptive stats**, **linear solver** | F.2 | DARK-ADMIN (called at `mcp.rs:1641`; guest never calls it) |
| **Construction calculator** | `src/math/construction.rs` `detect_construction` | feet-inches, lumber/framing/concrete | F.4 (fires before F.2) | DARK-ADMIN |
| **Word-problem calculator** | `src/math/wordprob.rs` `detect_word_problem` | prose word problems w/ honest hedging + gap-signal emit | F.3 | DARK-ADMIN |
| **Verifiable geometry (image/geometry math)** | `src/math/geometry.rs` | exact shaded-area / `a+b·π`, partition-verify | *not in cascade* | **ORPHANED** |

The cascade `try_arithmetic_response_with` (`src/mcp.rs:1543`) runs **F.1 → F.4 → F.3 → F.2 → recall**. Geometry is not in it. So the three that John means as "regular math / word-problem calculator / image-geometry math" map cleanly to: **`math.rs` (regular)**, **`math/wordprob.rs` (word problem)**, **`math/geometry.rs` (geometry)** — with the important caveat that on the *guest* only regular arithmetic survives, and geometry has no caller anywhere. (Bonus orphans inside the math tree: `math/autochord.rs` — 0 callers; `math/arrangement.rs` — 1.)

### 8. "The part that identifies a prompt for its needs" — **PRESENT, and this is the architecturally central finding**
This is the **prompt-need identifier / dispatcher**, and Lu has it in layers. The load-bearing one is `src/query_router.rs` — `QueryRouter::route` produces a `RoutingDecision` that reads a query and decides which capability ports it needs. Its fields are the complete "needs" vocabulary:

```
needs_math            references_file_path      code_intent
needs_deep_reasoning  references_image_path     is_meta_query
needs_variants        needs_critic_vetting      needs_agentic_thinking
```

Its own header states the intended design: *"Phase 3+ ports will wire `think()`, `variants()`, **file_io, vision, and code-gen** invocations off these flags."* **This is the map of what Lu was supposed to become — agentic — versus what actually got wired.** The router already *detects* file-path references, image-path references, and code intent. **Nothing consumes those three.** Only `needs_math`, `needs_deep_reasoning`, `needs_variants`, `needs_critic_vetting`, `needs_agentic_thinking`, and `is_meta_query` got wired into `narrate()` (via OP-ROUTE-3a/3b/3c). So the dispatcher's *intended routing table* is strictly larger than its *actual* one: it can already tell that a query wants a file, an image, or code — and there is no arm that acts on that knowledge.

The **fuller planners** the router was meant to drive:
- `src/thinking.rs` — the ReAct **Observe → Plan → Act → Reflect** loop (the "Left Brain"). CONDITIONAL on guest.
- `src/agents.rs` — the **multi-agent** orchestrator (RecallAgent → SynthesisAgent → CriticAgent, scope-isolated). CONDITIONAL on guest (fires via `needs_critic_vetting`).
- `src/reasoning.rs` — chain articulation ("Therefore… / Which means…"). CONDITIONAL on guest.
- The admin `tool_ask` orchestration in `src/mcp.rs` adds the Q.3 session pre-pass on top — admin-only.

**Verdict:** the need-identifier is present and is the single most important lever for making Lu agentic. Making Lu act on `references_file_path` / `references_image_path` / `code_intent` (plus threading these through `narrate()` on the guest) is the difference between "a recall engine" and "an agent that reaches for the right tool." The machinery to *decide* is built; the machinery to *act on the decision* is the gap.

### 9. Off-ness / Cattle Dog — **GENUINELY ABSENT from main (branch-only)**
`src/offness.rs` (402 LOC), `eval/offness/`, and `examples/offness_harness.rs` exist **only** on `feature/op-offness-1-cattledog-port`. That branch is a large, stale divergence from `main` (it deletes `read_only_reader.rs`, `wonder_sidecar.rs`, `near_miss.rs`, `canonical.rs`, and rewrites `network.rs`; it also carries a branch-only `src/embedding.rs`). Porting it is a genuine feature merge against an old base, not a rewiring. Confirmed absent, as the wiring map said.

---

## Part 2 — Master capability registry

Grouped by state. Columns: **Capability · What it is · Module:file · On main? · Wired to a live path? · Invoked by (or orphaned) · Reconnect/port cost.**

### LIVE-GUEST — reachable by a real user today

| Capability | What it is | Module | On main | Live path | Invoked by | Cost |
|---|---|---|---|---|---|---|
| Seed + spread recall | associative activation over the graph | `network.rs`, `brain.ask_with_config` | yes | yes guest | `narrate()` core | — |
| idk / refusal floor | fabrication guard on unknowns | `idk_gate.rs` | yes | yes guest | `narrate():3092` | wired |
| Recall → narration graph-walker | turns activated bonds into prose | `narration.rs`, `network.rs` | yes | yes guest | `narrate()` | wired (residency-dependent) |
| Greet refusal | guards greetings/non-questions | `greet_refusal.rs` | yes | yes guest | `guest_ask:160` | wired |
| Regular arithmetic | F.1 single-op / %/sqrt | `math.rs::detect_arithmetic` | yes | yes guest | `guest_ask:129` (`439e2e8`) | wired |
| Self-identity / meta short-circuits | "who are you" etc. | `narration.rs:2988` | yes | yes guest | `narrate()` | wired |
| 16-shape renderer + style hints | output shaping | `narration.rs` Renderer | yes | yes guest | `narrate()` | wired |

### CONDITIONAL — wired to guest `narrate()`, gated behind heuristics

| Capability | What it is | Module | On main | Live path | Gate | Cost to fully reach |
|---|---|---|---|---|---|---|
| Reasoning chains | procedural / list / comparison articulation | `reasoning.rs` | yes | cond. guest | `needs_deep_reasoning && !math && !meta` | tune triggers |
| Left Brain / thinking | ReAct Observe→Plan→Act→Reflect | `thinking.rs` | yes | cond. guest | `use_multi_think` hardcoded `false`; else `needs_critic_vetting`/`needs_agentic_thinking` + activation floor | raise reachability |
| Multi-agent (Recall/Synthesis/Critic) | scope-isolated belief vetting | `agents.rs` | yes | cond. guest | fires via `needs_critic_vetting` | tune |
| Variant arena | candidate-answer generation + promote | `variants.rs` | yes | cond. guest | `needs_variants` | tune |
| Query router / IntentClassifier | **prompt-need identifier** | `query_router.rs`, `narration.rs` IntentClassifier | yes | cond. guest | 6 of 9 flags consumed; file/image/code flags **not** consumed | wire the 3 unused flags |
| Curiosity producer | gap detection during answering | `curiosity.rs`, `wonder_sidecar.rs` | yes | cond. guest | only if guest launched `--wonder-queue` | ops |

### DARK-ADMIN — built + compiled in, but only callable from the admin `mcp.rs` dispatch (compiled out of guest)

| Capability | What it is | Module | On main | Live path | Invoked by | Reconnect cost |
|---|---|---|---|---|---|---|
| **Sandbox / code-execution runtime** | subprocess execution of signed capability packs, resource-capped | `capability_runtime/`, `mcp_cap_runtime.rs` | yes | dark (master-switch OFF) | `mcp.rs` dynamic `pack__*` | large — security-sensitive; keep gated |
| Capability-pack schema (Cat.5) | signed `.lupack5` format + publisher trust + install ledger | `capability_pack/`, `mcp_cat5.rs` | yes | dark | `mcp.rs` (7 `lu4d_capability_pack_*`) | admin only by design |
| Content/safety packs (Cat.4) | signed `.lupack` knowledge bundles (add facts w/o retrain) | `pack/`, `mcp_pack.rs` | yes | dark | `mcp.rs` (6 `lu4d_pack_*`) | admin only by design |
| **Image / vision tools** | scene-graph triplet extraction from images | `vision/`, `mcp_vision.rs` | yes | dark | `mcp.rs` (3 `lu4d_image_*`) | wire to `guest_ask` + configure a real extractor |
| **File tools** | read/write md/txt/csv/json/code as transport | `file_io.rs`, `mcp_file_io.rs` | yes | dark | `mcp.rs` (`lu4d_file_read/write`) | thread through guest + router `references_file_path` |
| **Document tools** | PDF/DOCX/markdown read + emit; CSV/JSON → pending | `file_io.rs` (FileKind Pdf/Docx/Md), `mcp_file_io.rs` | yes | dark | `mcp.rs` (`lu4d_file_extract_to_pending`) | same as file tools |
| Full calculator (F.2) | expressions, units, stats, linear solver | `math.rs::detect_math` + `math/{expr,units,stats,solve}.rs` | yes | dark | `mcp.rs:1641` cascade | ~15 LOC to add `detect_math` to `guest_ask` |
| Construction calculator (F.4) | feet-inches / framing math | `math/construction.rs` | yes | dark | cascade `mcp.rs:1543` | rides on the F.2 guest wire |
| Word-problem calculator (F.3) | prose word problems + hedging | `math/wordprob.rs` | yes | dark | cascade `mcp.rs:1543` | rides on the F.2 guest wire |
| Web research | HTTP fetch + DuckDuckGo/Grokipedia + research loop/chain | `research/`, `mcp.rs` (`lu4d_research*`) | yes | dark | `mcp.rs` | large — network egress + guest surface |
| Teach / learn / ingest | `teach`, `learn_text`, `teach_corpus`, `user_statement` | `mcp.rs`, `mcp_extractor.rs`, `extractor/` | yes | dark | `mcp.rs` | admin/offline by design |
| Session memory (Q.3) store | Left-Brain conversational session layer | `lb_session/`, `mcp_lb.rs` | yes | store reachable, reader/writer dark | `mcp.rs` (`lu4d_session_memory_*`) | thread `session_id` through `narrate()` + port Q.3 pre-pass |
| Style adaptation | output-style profile + WAL | `style/`, `mcp_cs.rs` | yes | dark | `mcp.rs` (`lu4d_style_*`) | medium |
| Trust / source attribution | per-source trust weights + revoke-by-source | `trust.rs`, `source.rs`, `mcp_cat3a.rs` | yes | dark | `mcp.rs` (`lu4d_trust_*`) | admin |
| Scar / unscar | suppress bad concepts/bonds | `scar.rs` | yes | dark (overlay applies on read) | `mcp.rs` (`lu4d_scar_*`) | admin |
| Time service | authenticated time, anomaly, timetravel query | `time/`, `mcp.rs` (`lu4d_time_status`,`lu4d_timetravel`) | yes | dark | `mcp.rs` | admin |
| Sense runtime | runtime sensing/config | `sense.rs`, `mcp_sense.rs` | yes | dark | `mcp.rs` (`lu4d_sense_*`) | admin |
| Hub-degree governance | high-degree-node handling | `hub_degree.rs`, `mcp_cat3b.rs` | yes | dark | `mcp.rs` (`lu4d_hub_*`) | admin |
| Idle loop / auto-tick | background maintenance ticks | `idle/`, `mcp_t6.rs` | yes | dark | `mcp.rs` (`lu4d_auto_tick_*`,`lu4d_idle_tick`) | ops |
| Pending / proposals | approve/reject staged changes | `pending/`, `mcp_t3.rs` | yes | dark | `mcp.rs` (`lu4d_*_pending`,`lu4d_proposal*`) | admin |

### ORPHANED — exists and compiles, but nothing calls it (not even admin)

| Capability | What it is | Module | On main | Called by | Note |
|---|---|---|---|---|---|
| **Verifiable geometry solver** | exact `a+b·π` shaded-area engine + partition verifier | `math/geometry.rs` | yes | **nothing** (only a test content-bond) | intended driver = image→scene pipeline that isn't on main |
| Chord/harmony math | `autochord` | `math/autochord.rs` | yes | **nothing** | 0 non-test refs |
| Arrangement math | `arrangement` | `math/arrangement.rs` | yes | ~1 ref | near-orphan |

### ABSENT-FROM-MAIN — not present; branch-only or roadmap-only

| Capability | What it is | Where it actually is | Note |
|---|---|---|---|
| **Off-ness / Cattle Dog** | off-topic/nonsense herding, answer-mediation safety layer | `feature/op-offness-1-cattledog-port` (`src/offness.rs`, `eval/offness/`, `examples/offness_harness.rs`) | large stale-base port, not a rewire |
| **Code generation / coding functions** | write/generate code | *nowhere* — only `query_router.code_intent` intent slot + roadmap internal notes | detector present, generator absent |
| **Image→geometry autonomous pipeline** | perceive diagram → Scene → exact solve | Python prototypes + `v4` worktree | solver core is on main but orphaned (see ORPHANED) |
| Embedding system | vector embeddings | `feature/op-offness-1-cattledog-port` (`src/embedding.rs`) | branch-only; not on main |
| Corpus loaders | arXiv / Grokopedia / StackExchange ingest | `feature/{arxiv,grokopedia,stackexchange}-loader`, `feature/wire-cat1-loaders` | branch-only feeders |

---

## Part 3 — Totals (the true scope at a glance)

- **LIVE-GUEST:** 7 capabilities a real user can actually reach.
- **CONDITIONAL:** 6 wired into `narrate()` but heuristic-gated (incl. the need-identifier itself).
- **DARK-ADMIN:** ~20 capability groups fully built and compiled in, reachable only through the admin tool surface — spanning the entire `lu4d_*` catalog (**99 tool string literals; ~40 top-level dispatch arms**), of which the guest exposes exactly **3** (`lu4d_ask`, `lu4d_session_start`, `lu4d_session_end`).
- **ORPHANED:** 3 (geometry solver, autochord, arrangement) — built, verified in isolation, zero callers.
- **ABSENT-FROM-MAIN:** 5 (Cattle Dog, code-gen, image→geometry loop, embeddings, corpus loaders).

**Headline ratio:** guest surface = **3 of ~40** tools; **1 of 4** math systems; **0** file/document/image/research/sandbox capability reachable by a user. Almost the entire capability estate is present-and-compiled but dark, because the guest binary compiles the tool dispatch out and `guest_ask` only wires arithmetic + `narrate()`.

---

## Part 4 — Honesty ledger: absent vs merely unwired

- **Genuinely ABSENT (not built on main):** code generation (only a router intent slot); Cattle Dog / off-ness (branch only); the image→geometry autonomous loop (prototype/v4); embeddings (branch only).
- **Built but ORPHANED (no caller anywhere):** the verifiable-geometry solver; autochord; arrangement.
- **Built and DARK-ADMIN (works, but users can't reach it):** sandbox/execution runtime, image/vision tools, file tools, document (PDF/DOCX) tools, full calculator + construction + word-problem math, web research, teach/ingest, session-memory reader/writer, style, trust, scar, time, sense, hub, idle, pending — essentially the whole `lu4d_*` catalog.
- **Built and CONDITIONAL (fires sometimes on guest):** reasoning chains, Left-Brain thinking, multi-agent, variants, curiosity producer, and the query router (6/9 flags consumed).
- **LIVE for users:** recall, idk floor, graph-walker, greet refusal, regular arithmetic, self/meta short-circuits, renderer.

**The single most leveraged reconnection** is not any one tool — it is teaching `narrate()`/`guest_ask` to *act on the router flags it already computes*. `references_file_path`, `references_image_path`, and `code_intent` are detected on every query and consumed by nothing. Wire those (plus thread `session_id`), and the file, document, image, and — once built — code capabilities light up through the need-identifier that already exists. That is the architecture that turns Lu from a recall engine into an agent.

---

## Provenance (all read-only this session, `main` @ `d51d62f`)
- Tool catalog: `src/mcp.rs` dispatch (99 `lu4d_*` literals); guest surface `crates/lu_guest_http/src/guest_mcp.rs` (3 tools).
- Module census: `src/lib.rs` (only `mcp` is `#[cfg(feature="admin_mcp")]`); `src/math.rs` + `src/math/*`; `src/{file_io,mcp_file_io,mcp_vision}.rs` + `src/vision/*`; `src/capability_runtime/*`, `src/capability_pack/*`, `src/pack/*`, `src/mcp_cat5.rs`, `src/mcp_cap_runtime.rs`, `src/mcp_pack.rs`.
- Need-identifier: `src/query_router.rs` (RoutingDecision fields + header intent), `src/{thinking,agents,reasoning}.rs`.
- Math cascade: `src/mcp.rs:1539-1660` (`try_arithmetic_response_with`); geometry caller search tree-wide (only `narration.rs:3778` test bond).
- Branch survey: `git branch -a` (143 branches); `git diff --stat main feature/op-offness-1-cattledog-port` (offness.rs, embedding.rs branch-only).
- Companion: `docs/LIVE_PATH_WIRING_MAP.md` (#1352). Retrain `lu4d-cat1-ingest` = active/undisturbed.
