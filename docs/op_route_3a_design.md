# OP-ROUTE-3a Design - Wire reasoning chains into narrate PASS path

**Status:** Phase A design. HALT for John review before implementation.
**Branch (planned):** `feature/op-route-3a-reasoning-chains` off `main` HEAD `ada8d2b` (tag `post-op-route-2-2026-05-27`).
**Scope:** Phase 3a of the routing+depth plan. Phase 2 (OP-ROUTE-2) populates `RoutingDecision` on every `NarrationResult`; this port consumes the `needs_deep_reasoning` dimension to invoke `reasoning::reason_with_config` on the PASS path and feed the resulting `ReasoningTrace` into shape selection and prose rendering.
**Anti-LLM positioning:** preserved. The router is still pure heuristics; `reason_with_config` is the existing best-first search over the live bond graph - no new model surface added.
**LOC budget:** ~200-400 net per the plan. Surface if creeping past 600.

---

## 0. Today vs after-3a

**Today (`src/narration.rs:2110-2284`):** `narrate()` already calls `reasoning::reason_with_config` once - inside the `idk_gate_enabled` block at `:2148`, only when `fast_pass_check` says "investigate further." The trace is consumed by `idk_gate::evaluate_signals` to inform the refusal verdict. On the PASS path the trace is never built. Shape selection runs off `(intent, ask, multi, flags)` only; the answer prose is a per-shape template fill against `ShapeSlots`.

**After 3a:** when `RoutingDecision.needs_deep_reasoning` is true and the idk_gate verdict is Pass, narrate() calls `reason_with_config` a second time and threads the resulting `ReasoningTrace` through `ShapeSelector::pick` (to bias toward WhyChain) and through `extract_slots` (to expose articulation text to render). The trace lands on `NarrationResult.reasoning` for callers.

Non-deep-reasoning queries: zero new work, byte-identical answer to pre-3a. Parity invariant load-bearing.

---

## 1. Where in narrate() does `reason_with_config` get called?

Three plausible insertion points were on the table:

- **(i)** After `IntentClassifier::classify`, before `brain.ask`.
- **(ii)** After `brain.ask`, before `idk_gate::fast_pass_check`.
- **(iii)** After idk_gate Pass verdict confirmed (or gate disabled), before `ShapeSelector::pick`.

**Recommendation: (iii).** Rationale:

- The refusal path at `:2148` already builds its own trace inside the gate block and consumes it locally. Building a *second* trace at insertion point (i) or (ii) wastes the same work twice for refusal queries (gate calls `reason_with_config` itself; PASS-path call would too) AND wastes work on PASS queries that turn out not to need it (gate disabled + parity queries).
- Insertion (iii) fires only when (a) the gate either passed or was disabled AND (b) `needs_deep_reasoning` is true. Both conditions are cheap to check, and the resulting trace is then used end-to-end through shape selection and render.
- Insertion (i) would also force us to thread the trace into the gate evaluation, which is a wider blast radius than this port should take (the gate is load-bearing and tested elsewhere).
- Insertion (ii) has the same trace-reuse problem as (i) and additionally puts the chain build *before* we know whether the gate will refuse - so on refusal queries we build it twice.

Concretely, the new call site sits between the existing line `:2189` (where `flags` is computed) and line `:2190` (where `ShapeSelector::pick` runs). Pseudo-diff:

```
    let flags = ModulationFlags::compute(&ask, multi.as_ref(), cfg);
+   // OP-ROUTE-3a: build a reasoning trace when the router asked
+   // for deep reasoning. Only fires on the PASS path - the
+   // refusal path inside idk_gate_enabled already builds its own
+   // trace at :2148 and returns early.
+   let reasoning_trace: Option<crate::reasoning::ReasoningTrace> =
+       if routing.needs_deep_reasoning
+           && !routing.is_meta_query
+           && !routing.needs_math {
+           let r_cfg = crate::reasoning::ReasoningConfig::default();
+           let scope = crate::scope::TemporalScope::Current;
+           Some(crate::reasoning::reason_with_config(
+               brain, query, scope, &r_cfg,
+           ))
+       } else {
+           None
+       };
    let shape = ShapeSelector::pick(intent, &ask, multi.as_ref(), &flags, cfg);
```

The variable `reasoning_trace` is then threaded into `ShapeSelector::pick`, `extract_slots`, and `NarrationResult`. See sections 4 and 5.

---

## 2. When does it fire?

**Recommendation: solely on `RoutingDecision.needs_deep_reasoning == true`, with two cheap guards.**

The router is the single source of truth for "this query warrants the deep port." Adding a second axis here (e.g. "also fire on `QueryIntent::WhyChain`") would mean every WhyChain-classified query builds a chain - and short why-chain queries like *why is the sky blue?* are exactly what `needs_deep_reasoning` was tuned to NOT fire on (it requires either "why" + "because", 25+ words, multi-axis markers, etc.). Building reasoning chains on every short WhyChain query would (a) blow the LOC budget on a second decision surface and (b) regress latency on the most common shape category.

If the router is wrong (deep-reasoning should fire on short why-chains too), the fix lives in `query_router::needs_deep_reasoning`, not here. That is the right blast radius.

Two practical guards:

- **`routing.is_meta_query == false`.** A meta query ("what do you know about X") explicitly should not run a chain through the brain about its own state. Today `needs_deep_reasoning` and `is_meta_query` are computed from disjoint signals so co-firing is rare but not impossible (e.g. a 25+ word self-referential question would set both). Cheap to guard.
- **`routing.needs_math == false`.** `routing.needs_math` shouldn't co-fire with `needs_deep_reasoning` on the MCP path (math short-circuits in `try_arithmetic_response_with` per OP-ROUTE-1), but on the in-process `narrate()` path the math handler is upstream. Skip deep reasoning when `needs_math` is true; defer to the math path.

Final predicate:
```rust
let should_reason = routing.needs_deep_reasoning
    && !routing.is_meta_query
    && !routing.needs_math;
```

Both guards surfaced for John's confirmation in Section 9 (open questions 1 and 2).

---

## 3. What scope does the trace run in?

**Recommendation: `TemporalScope::Current`.**

This matches the existing refusal-side call at `:2150`. The narrate() pipeline does not accept a scope hint from upstream today (`narrate(brain, query, depth, top_k, use_multi_think, cfg)` - no scope param), and the in-scope path is `Brain::narrate_with_scope` (not the regular narrate). Pulling a scope param into narrate() to thread to reasoning would expand the function signature unnecessarily for this port; defer that to a future port that needs scope-aware narrate end-to-end.

The scope choice is bounded to one literal in one place, and changing it later is a one-liner. Document it as a known limitation in the function comment.

---

## 4. What happens to the trace?

Four consumers, in pipeline order:

### 4a. `ShapeSelector::pick` - chain-aware shape bias

Widen the signature:
```rust
pub fn pick(
    intent: QueryIntent,
    ask: &AskResult,
    multi: Option<&MultiAgentTrace>,
    flags: &ModulationFlags,
    cfg: &NarrationConfig,
    reasoning: Option<&crate::reasoning::ReasoningTrace>,  // NEW
) -> ResponseShape
```

New rule inserted between rule 9 (two-hop synthesis) and rule 9b (definition curiosity calibration), so it can override Synthesis but is itself overridden by the strong wondering/critic paths:

```rust
// 9.5 - OP-ROUTE-3a: if a structured reasoning chain is
// available (>= 2 steps, EvidenceKind::Structured, not
// no_chain), prefer WhyChain regardless of nominal intent.
// The chain itself is the strongest possible signal that
// a why-shape is the right shape, even if IntentClassifier
// landed on Process / Generic.
if let Some(trace) = reasoning {
    if !trace.no_chain
        && trace.steps.len() >= 2
        && trace.evidence_kind == crate::reasoning::EvidenceKind::Structured
    {
        return ResponseShape::WhyChain;
    }
}
```

Notes:
- We do NOT introduce a new shape (no `ReasonedAnswer` shape). WhyChain already exists and its template fits chain articulation. **Open question 3:** is a new shape worth it for chain prose that doesn't follow the "A happens because B which leads to C" template? Default: NO - reuse WhyChain. If render-side experimentation shows the template is too narrow, that's a Phase-3a-followup port.
- `Associative` chains do NOT trigger the bias. Per the EvidenceKind doc at `reasoning.rs:160`, those are co-occurrence-only and the `final_answer` is already hedged. Letting them force WhyChain would propagate that hedging into shape selection in a confusing way.
- All other ShapeSelector callers (the unit tests at `:2445`, `:2683`, `:2685`, `:2800`, `:2878`) get `None` passed for the new arg, preserving parity. ~8 call sites to update.

### 4b. `extract_slots` - articulation text in slots

Widen the signature:
```rust
fn extract_slots(
    brain: &crate::brain::Brain,
    query: &str,
    ask: &AskResult,
    multi: Option<&MultiAgentTrace>,
    cfg: &NarrationConfig,
    reasoning: Option<&crate::reasoning::ReasoningTrace>,  // NEW
) -> ShapeSlots
```

When `reasoning.is_some() && !trace.no_chain && trace.steps.len() >= 2`, populate two new `ShapeSlots` fields:

```rust
// In ShapeSlots:
#[serde(default)]
pub chain_articulation: Option<String>,  // trace.final_answer
#[serde(default)]
pub chain_confidence: f32,                // trace.confidence
```

`trace.final_answer` is already the pre-rendered prose form of the chain (per `reasoning.rs:194` comment - it strings the per-step articulations together). We do not re-render in narrate; we copy.

All internal call sites of `extract_slots` (tests at `:2714`, `:3133`, `:3144` plus the production call at `:2191`) get the new arg added. Tests pass `None` to preserve parity.

### 4c. `Renderer::render` / `render_core` - WhyChain prose with chain content

The WhyChain render arm at `:1698-1712` today fills three node names into a fixed template:
```rust
ResponseShape::WhyChain => match tier {
    VerbosityTier::Standard => format!(
        "{} happens because {}, which leads to {}, producing {}.",
        unknown_or(&s.primary),
        unknown_or(&s.secondary),
        unknown_or(&s.tertiary),
        unknown_or(&s.primary)
    ),
    ...
```

New behavior: if `s.chain_articulation` is `Some`, use it verbatim (it is already a complete prose sentence from `reasoning::compose_final_answer`); else fall through to the existing template. Same pattern the Comparison arm at `:1714` already uses for `walker_prose`. Roughly:
```rust
ResponseShape::WhyChain => {
    if let Some(prose) = &s.chain_articulation {
        prose.clone()
    } else {
        // existing template
        match tier { ... }
    }
}
```

This is the only render arm modified. Definition / Process / Synthesis are NOT touched in this port; the brief lists them as "possibly," but adding chain prose to three more shapes would (a) expand the LOC budget and (b) require per-shape parity tests. Defer to a follow-up port if there is signal that those shapes benefit. **Open question 4.**

### 4d. `NarrationResult` - trace surfaced for callers

See section 5.

---

## 5. NarrationResult extension

Add one field, parallel to the OP-ROUTE-2 `routing` field at `:2104`:

```rust
pub struct NarrationResult {
    ...
    pub routing: crate::query_router::RoutingDecision,

    /// OP-ROUTE-3a - reasoning trace built when
    /// `routing.needs_deep_reasoning` was true and the idk_gate
    /// path passed. `None` when deep reasoning was not invoked
    /// (the common case). Same backward-compat pattern as `routing`.
    #[serde(default)]
    pub reasoning: Option<crate::reasoning::ReasoningTrace>,
}
```

`ReasoningTrace` already derives `Serialize + Deserialize` (`reasoning.rs:190`), so wrapping it in `Option<...>` is free.

Why `Option`, not the default-trace pattern that `routing` uses? Because `ReasoningTrace` does not have a meaningful default - `Default::default()` would imply an empty chain, which is semantically the same as "reasoning ran and produced no chain" (`no_chain: true`). We want to distinguish "reasoning did not run" from "reasoning ran and found nothing." `Option<ReasoningTrace>` does that cleanly.

Backward-compat: `#[serde(default)]` lets older NarrationResult JSON (without the field) round-trip cleanly, same as `routing`, `source_kinds`, etc.

---

## 6. ModulationFlags impact

**Recommendation: no new flag in this port.**

The 5 existing flags (`curiosity_touched`, `bridge_touched`, `hypothesis_only_support`, `critic_vetted`, `wondering_touched`) are all about *recall-state* signals - they answer "did the brain feel uncertain / curious / cross-domain during recall?". The reasoning chain is a different layer - it is a *post-recall* structural property of the bond graph. Conflating them would muddy ModulationFlags' purpose.

Does the chain influence `critic_vetted`? No: critic_vetted means "multi_think ran and produced vetted hypotheses" - that path is a separate machine (`brain.multi_think`). The chain is built by `reason_with_config`. Same modulation surface, different intent.

A future port could add a `chain_articulated: bool` flag if signal selection needs it for prose modulation (e.g. "surface the chain in the answer prefix"). Out of scope here - the current Renderer change picks up the chain through `chain_articulation` slot, not through a flag.

Neutral consequence: `signals_used` does not gain a new entry. `routing.triggers` already carries `"deep_reasoning"` when the router fires, which is the right place to log "chain was attempted." Callers wanting to know if a chain *actually came back* check `result.reasoning.is_some()` and `result.reasoning.as_ref().map(|t| !t.no_chain)`.

---

## 7. Bounded scope - which config?

**Recommendation: `ReasoningConfig::default()`, exactly as the refusal-side call does at `:2149`.**

`NarrationConfig` does not currently expose reasoning knobs (max_chain_length, min_edge_weight, etc.), and adding them would (a) widen `NarrationConfig` for a single consumer and (b) couple narration tests to reasoning internals. `ReasoningConfig::default()` is well-calibrated (Port H bumped max_chain_length to 8, the deep-chain ceiling) and matches the refusal-side call exactly - that parity is itself useful: same query under different gate verdicts produces traces from the same config.

If a future port needs tuned reasoning knobs from narrate (e.g. "deeper chains for stakeholder questions"), that is a NarrationConfig extension and is out of scope here.

---

## 8. Tests planned

Unit tests, all in `#[cfg(test)] mod tests` at the bottom of `src/narration.rs`. Plan:

### Per-dimension - chain-aware shape selection

1. `shape_picks_whychain_when_structured_chain_present` - hand-build a `ReasoningTrace` with `steps.len() == 3`, `no_chain: false`, `EvidenceKind::Structured`. Call `ShapeSelector::pick(QueryIntent::Definition, &ask, None, &flags, &cfg, Some(&trace))`. Assert `ResponseShape::WhyChain`.
2. `shape_ignores_associative_chain` - same, but `EvidenceKind::Associative`. Assert shape is NOT WhyChain (would have been Definition).
3. `shape_ignores_no_chain_trace` - hand-build a trace with `no_chain: true`. Assert WhyChain bias does not fire.
4. `shape_ignores_single_step_chain` - `steps.len() == 1`. Assert bias does not fire.
5. `shape_no_chain_arg_preserves_intent` - pass `None` as the new arg. Assert shape selection matches pre-3a behavior for a definition query.

### Slot-fill

6. `extract_slots_populates_chain_when_trace_present` - hand-build a trace with `final_answer: "X happens because Y."`, `confidence: 0.82`. Call `extract_slots` with `Some(&trace)`. Assert `slots.chain_articulation == Some("X happens because Y.")` and `slots.chain_confidence == 0.82`.
7. `extract_slots_skips_chain_when_no_chain_true` - trace with `no_chain: true`. Assert `slots.chain_articulation.is_none()`.

### Render

8. `render_whychain_uses_chain_prose_when_present` - build `ShapeSlots` with `chain_articulation: Some("chain prose")`. Call `Renderer::render(WhyChain, ...)`. Assert output starts with `"chain prose"`.
9. `render_whychain_falls_back_to_template_when_chain_absent` - slots with `chain_articulation: None`, all primary/secondary/tertiary filled. Assert output matches the existing template format byte-for-byte (parity).

### Parity invariant (LOAD-BEARING)

10. `narrate_parity_no_deep_reasoning_query` - call `narrate()` against a hand-built tiny brain with `query = "what is photosynthesis?"`. Capture answer. Confirm `result.routing.needs_deep_reasoning == false`. Confirm `result.reasoning.is_none()`. Then assert the answer is byte-identical to the pre-3a expected value captured in the test as a string literal.
11. `narrate_invokes_reasoning_when_deep_reasoning_routes` - call `narrate()` with `query = "walk me through step by step how photosynthesis works in great detail considering chemical and biological factors"` (>=25 words, triggers `needs_deep_reasoning`). Confirm `result.reasoning.is_some()`. Confirm `result.shape == ResponseShape::WhyChain` IF the tiny brain has a buildable chain; ELSE fall back assertion (no_chain branch, shape returns to intent-derived default).

### NarrationResult round-trip

12. `narration_result_serde_round_trips_with_reasoning_none` - serialize a result with `reasoning: None`, deserialize, assert equality.
13. `narration_result_serde_round_trips_with_reasoning_some` - serialize a result with a populated trace, deserialize, assert structural equality.

Count: 13 new tests. Existing test count baseline pre-3a is the 1470+ already-passing. Target: all existing pass + 13 new.

---

## 9. Open questions for John

1. **Meta-query guard.** Skip deep reasoning when `routing.is_meta_query` is true? Recommendation: YES (don't reason about Lu's own state through the bond graph). Section 2.
2. **Math guard.** Skip deep reasoning when `routing.needs_math` is true? Recommendation: YES, defer to math path. Section 2.
3. **New ResponseShape variant?** Add a dedicated `ReasonedAnswer` shape, or reuse `WhyChain`? Recommendation: REUSE. Section 4a.
4. **Other render arms.** Also wire chain prose into Definition / Process / Synthesis arms? Recommendation: NO, defer. Section 4c.
5. **Trace on refusal path.** The refusal path at `:2148` already builds a trace and consumes it inside `idk_gate::evaluate_signals`. Should we ALSO surface that trace on `NarrationResult.reasoning` for refusal queries? Recommendation: YES, low cost - one extra `Some(trace)` carried into the Refusal-shape return at `:2165`. Improves caller observability uniformly. Listed because it widens the parity surface slightly (refusal-path NarrationResult JSON gains a populated `reasoning` field for some queries).
6. **Articulation length.** `trace.final_answer` for an 8-step chain can be 300-500 chars. Should we cap it at `cfg.max_answer_chars` like other shape outputs? Recommendation: YES - render layer already truncates at the same cap downstream of `render_core`; chain prose flows through that same path naturally, no special handling needed.

---

## 10. What is NOT in this port

- **No variant arena.** That is Phase 3b. `routing.needs_variants` stays observation-only.
- **No thinking layer.** That is Phase 3c. `routing.needs_critic_vetting` stays observation-only.
- **No parallelization.** That is Phase 4. The reasoning call is sequential after `brain.ask`.
- **No changes to `reasoning.rs`.** This port consumes the existing `reason_with_config` API; if a need surfaces to widen the API (e.g. "build the chain anchored on a specific seed"), surface as a follow-up port, not folded in here.
- **No new ResponseShape.** Reuse `WhyChain`.
- **No new ModulationFlags entry.** See section 6.
- **No scope plumbing.** `TemporalScope::Current` is hardcoded; scope-aware narrate is a future port.
- **No render-side change to Definition / Process / Synthesis.** Section 4c.
- **No production deploy.** Daemon stays stopped. Build + tests + 4-prompt sanity check only.

---

## 11. LOC estimate breakdown

Net, post-3a:

- `src/narration.rs` insertion point (iii) call + predicate guards: ~15 lines.
- `ShapeSelector::pick` signature + new rule + 8 internal callsite updates: ~30 lines (mostly mechanical None propagation).
- `extract_slots` signature + new slot population + 7 internal callsite updates: ~25 lines.
- `ShapeSlots` two new fields: ~6 lines (struct + serde attrs).
- `Renderer::render` WhyChain arm change: ~10 lines (if/else around the existing template).
- `NarrationResult` new field + the return-site population (PASS path + refusal path Q5): ~10 lines.
- 13 new unit tests: ~250 lines.

**Net: ~345 lines.** Inside the 200-400 LOC envelope.

---

## 12. Halt boundary

This document IS Phase A. Phase B (implementation) does NOT start until John responds to the open questions in section 9 and confirms the design. After that, the work proceeds branch -> implement -> tests -> build -> sanity -> halt-for-review, per the brief.
