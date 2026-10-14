# OP-ROUTE-3b Design - Wire variant arenas into narrate PASS path

**Status:** Phase A design. HALT for John review before implementation.
**Branch (planned):** `feature/op-route-3b-variant-arenas` off `main` HEAD `4a855c0`.
**Scope:** Phase 3b of the routing+depth plan. Phase 3a (reasoning chains) merged; this port consumes `RoutingDecision.needs_variants` to invoke `Brain::generate_variants` on the PASS path and feed the resulting `VariantArena` into shape selection, slot fill, and `NarrationResult`.
**Anti-LLM positioning:** preserved. Router stays pure heuristics; `generate_variants` is the existing perturbation engine over the live bond graph - no new model surface.
**LOC budget:** ~300-500 net per the brief. Surface if creeping past 700.

---

## 0. Today vs after-3b

**Today (`src/narration.rs:2264-2280`):** `narrate()` already consumes `routing.needs_deep_reasoning` (OP-ROUTE-3a) and threads a `ReasoningTrace` into `ShapeSelector::pick`, `extract_slots`, and `NarrationResult`. `routing.needs_variants` is computed and logged in `routing.triggers` but observation-only - no pipeline shape change. `Brain::generate_variants(goal_seed: Seed) -> VariantArena` exists at `src/brain.rs:4342` and is reachable only via the standalone `lu4d_variants_generate` MCP tool at `src/mcp.rs:2621`.

**After 3b:** when `RoutingDecision.needs_variants` is true and the idk_gate verdict is Pass, narrate() derives a goal seed from `AskResult.seed_hits`, calls `brain.generate_variants_with_config(goal, &cfg)`, and threads the resulting `VariantArena` (or a compact summary of it) through shape selection (one new rule biases toward `MultiHypothesisAnswer`), `extract_slots` (winner alt labels populated from variant scores), and `NarrationResult.variants` for callers.

Non-variant queries: zero new work, byte-identical answer to pre-3b. Parity invariant load-bearing.

**Material new surface vs 3a:** `generate_variants` is `&mut self` - it writes Variant-tagged keyframes into the live graph. Unlike 3a's `reason_with_config` (pure over the bond state), 3b is structurally a state mutation. Losers age out via the existing temporal half-life; the winner is NOT auto-promoted in this port (see section 12).

---

## 1. Goal seed derivation

Three candidates considered:

- **(a) Top activation from `ask.recall.activated[0].seed`.** Highest signal in the spread; what most pipelines mean by "what did the query land on."
- **(b) `ask.seed_hits[0]`.** The literal subject anchor - the seed that matched the query tokens before spread. Pre-spread, pre-pollution. Aligns with how Port #18a polysemy disambiguation already anchors.
- **(c) Highest-bonded slot node** - re-compute degree across activated nodes.

**Recommendation: (b) `ask.seed_hits[0]`, fallback to (a) `ask.recall.activated[0].seed`.**

Rationale:

- `seed_hits` is what the user asked about. Variants are meant to explore *alternative perspectives on the goal*; the goal is the subject, not the spread peak. Using top activation would let a strongly-bonded neighbor steal the anchor whenever the subject itself has lower self-activation than its highest neighbor (common for hub seeds).
- `seed_hits[0]` already drives Port #18a polysemy disambiguation - reusing it keeps the variant arena consistent with the sense the rest of the pipeline elected.
- Fallback to top activation handles the edge case where `seed_hits` is empty (rare - e.g. a pure-stopword query that still spread via session priors). If both are empty, skip variants entirely (return `None`); the predicate at section 3 already guards this.
- (c) was rejected as a third decision surface for marginal benefit - the existing predicate already restricts to deep-reasoning-class queries, and the goal-vs-peak distinction is what matters, not degree among peaks.

Concretely:
```rust
let goal_seed: Option<crate::seed::Seed> =
    ask.seed_hits.first().copied()
        .or_else(|| ask.recall.activated.first().map(|a| a.seed));
```

If `goal_seed` is `None` after both fallbacks, the variant block is a no-op for this query (consistent with predicate-gated execution).

---

## 2. Insertion point in `narrate()`

The OP-ROUTE-3a block lives at `src/narration.rs:2264-2273`, immediately before `ShapeSelector::pick` and `extract_slots`. The 3b block sits **immediately after** the 3a reasoning block and **before** `ShapeSelector::pick`. Pseudo-diff:

```
    // OP-ROUTE-3a: build a reasoning trace ...
    let reasoning_trace: Option<crate::reasoning::ReasoningTrace> = ...;
+   // OP-ROUTE-3b: spawn variant arena when the router asked for variants.
+   // Goal seed: ask.seed_hits[0] with top-activation fallback. Predicate
+   // mirrors 3a's guards (skip meta + math). Independent of reasoning -
+   // 3a and 3b can co-fire (a deep, multi-perspective question gets both
+   // a chain and an arena; the chain anchors the WhyChain shape, the
+   // arena anchors MultiHypothesisAnswer - and the multi-hypothesis rule
+   // wins per shape-selector ordering).
+   let variant_arena: Option<crate::variants::VariantArena> =
+       if routing.needs_variants
+           && !routing.is_meta_query
+           && !routing.needs_math
+       {
+           let goal_seed = ask.seed_hits.first().copied()
+               .or_else(|| ask.recall.activated.first().map(|a| a.seed));
+           goal_seed.and_then(|seed| {
+               // Min-activation threshold: skip arena spawn if the goal
+               // seed's spread activation is below a floor (cheap arena
+               // veto when the brain barely recognised the subject).
+               let act = ask.recall.activated.iter()
+                   .find(|a| a.seed == seed)
+                   .map(|a| a.activation)
+                   .unwrap_or(0.0);
+               if act < cfg.variant_min_activation { return None; }
+               Some(brain.generate_variants_with_config(
+                   seed, &crate::variants::VariantConfig::default(),
+               ))
+           })
+       } else {
+           None
+       };
    let shape = ShapeSelector::pick(
        intent, &ask, multi.as_ref(), &flags, cfg,
        reasoning_trace.as_ref(),
+       variant_arena.as_ref(),
    );
```

Why this ordering:

- **After reasoning, before shape pick** so the rule in section 6 can override 3a's WhyChain bias when both fire (a question that triggers both `needs_deep_reasoning` AND `needs_variants` is by definition multi-hypothesis territory; the variant arena's explicit alternative enumeration outranks a single chain).
- **Before slot fill** so `extract_slots` can populate winner-derived `winner_alt`/`winner_alt_count` slots (section 5).
- **After idk_gate Pass** so refusals don't pay the variant cost. The gate's early return at `:2165` is structurally upstream of this block.

---

## 3. Fire predicate

**Recommendation:**
```rust
let should_variant = routing.needs_variants
    && !routing.is_meta_query
    && !routing.needs_math
    && goal_seed.is_some()
    && top_activation_of(goal_seed) >= cfg.variant_min_activation;
```

The baseline (`needs_variants && !is_meta_query && !needs_math`) mirrors 3a's predicate. The two additions:

- **`goal_seed.is_some()`** - degenerate case (empty seed_hits AND empty activated) where no anchor exists. Cheap.
- **Min-activation threshold** - the brief explicitly asks whether to add this. **Recommendation: YES, default `variant_min_activation = 0.15`.** Justification: `generate_variants` writes 5 keyframes per arena (one per strategy) into the bond graph, and the lowest-cost arena is still ~5x the activation work of a single spread (variants spread inside `Variant{id}` scope per `variants.rs:467 score_arena`). Spending that on a query whose subject barely registered (e.g. typo, off-domain) yields five low-coherence arenas and pollutes the temporal-keyframe layer with junk that has to decay. 0.15 is below the `idk_gate` activation floor (so anything the gate passed has cleared this) but above the noise floor where activation came from chained session-prior bleed-through. Tunable, surfaced on `NarrationConfig`.

Both tunables land on `NarrationConfig`, not `VariantConfig` - the threshold is a narration-pipeline decision ("is this query worth a variant spawn"), not a variant-engine decision ("how should this arena score").

---

## 4. VariantConfig

**Recommendation: `VariantConfig::default()`, exactly as `tool_variants_generate` does at `src/mcp.rs:2625`.**

`VariantConfig` defaults (`src/variants.rs:173-186`) are well-calibrated: 5 strategies, max 200 activations per variant, 0.6/0.3/0.1 composite weights, `min_winner_composite = 0.4`. Adding a sub-struct on `NarrationConfig` (e.g. `narration_variant: VariantConfig`) would (a) widen `NarrationConfig` for a single consumer (b) couple narration tests to variant internals (c) mirror the same anti-pattern 3a rejected for `ReasoningConfig`. Parity with the MCP tool is itself useful: the same goal seed yields the same arena whether reached via `lu4d_variants_generate` or via narrate.

One narration-layer knob lives on `NarrationConfig`: `variant_min_activation: f32` (section 3). That is the *pipeline-gate* knob, not a variant-engine knob.

---

## 5. Result handling

Three options on the table:

- **(a) Extract winner's nodes, feed as slots to the existing shape.** Lightest touch, observable but no narrative surface.
- **(b) Switch to `MultiHypothesisAnswer` shape.** That enum variant exists (`src/narration.rs:190`) AND its render arm is implemented (`src/narration.rs:1808-1821`), currently driven by `multi_think` hypotheses. The slots it consumes (`winner_alt`, `winner_alt_count`, `primary`) map cleanly onto variant arena data: winner_id resolves to a label via the variant's anchor seed, runner-up scores become `winner_alt_count` and `winner_alt`.
- **(c) Modulate confidence on the existing shape.**

**Recommendation: (b) reuse `MultiHypothesisAnswer`.**

Rationale:

- The shape, its enum variant, and the render arm all already exist. Zero new shape, zero new render arm - just a new path that populates the existing slots from variant data instead of multi-think data.
- The arena's structure (one winner + scored alternatives) maps almost exactly onto what `MultiHypothesisAnswer` already articulates ("Most likely X. I considered N alternatives ..."). Reusing it makes the variant outcome **observable in the rendered prose**, which is the cheapest yet most reviewer-legible signal.
- Option (a) is cheaper but invisible to a reader; option (c) is most invisible of all. The whole point of running variants on the PASS path is to let multi-hypothesis answers emerge from structural perturbation, not from multi-think alone.
- One nuance: when **both** `multi_think` produces hypotheses AND variant arena produces a winner, multi-think wins (rule 8 in ShapeSelector fires before the new 9.4 - see section 6). This preserves the existing critic-vetting path for queries that triggered both signals; variants only drive shape selection when multi-think did NOT fire. This is the right asymmetry: critic-vetted hypotheses are higher-confidence than structural variant winners.

**Slot mapping (in `extract_slots`):**
```rust
if let Some(arena) = variants {
    if arena.winner_id.is_some() {
        // Winner's anchor label -> primary (only if slot empty).
        if let Some(winner_spec) = arena.variants.iter()
            .find(|v| Some(v.id) == arena.winner_id)
        {
            if s.primary.is_none() {
                if let Some(node) = brain.network.get_node(&winner_spec.anchor_seed) {
                    s.primary = Some(node.content().to_string());
                }
            }
        }
        // Runner-up anchor label + count of non-winner specs.
        let non_winner: Vec<&VariantSpec> = arena.variants.iter()
            .filter(|v| Some(v.id) != arena.winner_id)
            .collect();
        if !non_winner.is_empty() && s.winner_alt.is_none() {
            // Pick the runner-up by score.
            // ... lookup highest-composite non-winner ...
            if let Some(node) = brain.network.get_node(&runner_up.anchor_seed) {
                s.winner_alt = Some(node.content().to_string());
            }
            s.winner_alt_count = non_winner.len();
        }
    }
}
```

Guard `s.primary.is_none()` and `s.winner_alt.is_none()` so multi-think slot fill (which runs upstream of this block in extract_slots) wins on overlap.

---

## 6. ShapeSelector rule for variant-winner

**Rule number: 9.4** (sits between rule 9 "two-hop synthesis" and the existing rule 9.5 "WhyChain on structured chain"). 3a's rule is 9.5 (Section 4a of the 3a doc). 9.4 means the variant-winner bias **outranks** the chain bias when both fire - matching the section-5 rationale.

**Signature extension:**
```rust
pub fn pick(
    intent: QueryIntent,
    ask: &AskResult,
    multi: Option<&MultiAgentTrace>,
    flags: &ModulationFlags,
    cfg: &NarrationConfig,
    reasoning: Option<&crate::reasoning::ReasoningTrace>,
    variants: Option<&crate::variants::VariantArena>,  // NEW
) -> ResponseShape
```

**New rule body, inserted at the line after rule 9's `return ResponseShape::Synthesis;` and before the existing rule 9.5 reasoning bias:**
```rust
// 9.4 - OP-ROUTE-3b: a variant arena that produced a winner is a
// strong signal for MultiHypothesisAnswer regardless of nominal
// intent. The arena's score gate (`min_winner_composite = 0.4`) is
// the upstream confidence floor; if it picked a winner, the arena
// is internally consistent. Distinct from rule 8's multi-think
// path (critic-vetted hypotheses) - rule 8 runs first, so when
// multi-think AND variants both fire, the critic-vetted path wins.
if let Some(arena) = variants {
    if arena.winner_id.is_some() && arena.variants.len() >= 2 {
        return ResponseShape::MultiHypothesisAnswer;
    }
}
```

Guards:
- `winner_id.is_some()` - the arena's own confidence gate (`min_winner_composite`) already passed.
- `variants.len() >= 2` - need at least one alternative to articulate ("considered N alternatives" reads false otherwise).

All other `ShapeSelector::pick` call sites get `None` for the new arg, preserving parity. ~10 callsites to update (per 3a's count, same test list).

---

## 7. NarrationResult.variants field

**Schema: `Option<VariantArenaSummary>`** (not the full `VariantArena`).

Rationale for a compact summary, not the full arena: `VariantArena` carries 5 `VariantSpec`s with anchor seeds, source goals, timestamps, etc. - useful internally but heavy on the JSON wire. For caller observability we need the headline numbers, not the per-keyframe substrate.

```rust
/// OP-ROUTE-3b - compact summary of a VariantArena that landed on a
/// narrate PASS-path query. None when the router did not request
/// variants, or when the request was guarded out (meta / math /
/// missing goal seed / below activation threshold). Backward-compat:
/// older NarrationResult JSON without this field round-trips clean
/// via `#[serde(default)]`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VariantArenaSummary {
    /// Which perturbation strategy produced the winner. None if no
    /// winner was elected (arena ran but composite stayed below
    /// `min_winner_composite`).
    pub winner_strategy: Option<crate::variants::PerturbationStrategy>,
    /// Winner's composite score. 0.0 when no winner.
    pub winner_composite: f32,
    /// Number of activated nodes in the winner's scope (informational).
    pub winner_node_count: usize,
    /// Total candidate specs spawned (1-5, capped by `variants_per_goal`).
    pub num_candidates: usize,
    /// True when the arena ran but elected no winner (composite below
    /// floor). Distinct from "variants did not run at all" (the outer
    /// `Option<VariantArenaSummary>` is None for that).
    pub no_winner: bool,
}
```

Add to `NarrationResult` parallel to `reasoning`:
```rust
pub struct NarrationResult {
    ...
    #[serde(default)]
    pub reasoning: Option<crate::reasoning::ReasoningTrace>,

    /// OP-ROUTE-3b - variant arena summary built when
    /// `routing.needs_variants` was true and the PASS path elected
    /// to spawn. None when variants did not run.
    #[serde(default)]
    pub variants: Option<VariantArenaSummary>,
}
```

Why `Option<...>` instead of the default-flag pattern `routing` uses: distinguishing "variants did not run" from "variants ran with no winner" matters for observability and for the parity tests in section 10. `no_winner: bool` is the inner flag for the second case.

---

## 8. ModulationFlags impact

**Recommendation: no new flag in this port.**

Same argument as 3a Section 6: ModulationFlags is the *recall-state* signal surface (curiosity / bridge / hypothesis_only / critic_vetted / wondering). A variant arena is a *post-recall structural perturbation*. Mixing them blurs the layer.

The one nominally adjacent flag is `critic_vetted`. Variants and critic-vetting are independent paths (critic is multi-think; variants are the perturbation engine). A future port could introduce `variant_winner_elected: bool` if a downstream signal needs it for prose modulation - out of scope here.

`signals_used` does not gain an entry. `routing.triggers` already carries `"variants"` when the router fires (per QueryRouter contract); callers wanting "did an arena actually elect a winner" check `result.variants.as_ref().map(|v| v.winner_strategy.is_some())`.

---

## 9. Performance caveat

**Variants is sequential and ~5x the cost of a single PASS spread.** `score_arena` (`src/variants.rs:467`) iterates `specs` and calls `variant_activated_set` once per strategy (5 by default), each of which runs a scoped spread similar in cost to `brain.ask`'s recall. The five spreads do NOT share work today; each one re-walks the bond graph from its own anchor.

Additionally, `generate_variants` is `&mut self` - each invocation writes 5 keyframes (one per strategy) into the temporal bond layer. These decay naturally per `variant_decay_halflife_ms = 6h`, but the write path is non-trivial under the keyword-index mmap regime.

**Expected latency impact on a triggering query: 3-6x the pre-3b PASS-path narrate latency.** This is intentional: `routing.needs_variants` triggers on queries explicitly asking for alternatives / hypotheses / what-ifs - the user is paying for the multi-perspective surface. Non-triggering queries are unaffected (predicate-gated; zero new work).

**Out of scope:** Phase 4 will parallelize the five strategy spreads with rayon (per `src/variants.rs:1` module doc and the brief's section 12). This port lands the sequential path and the predicate gate; Phase 4 swaps the inner iterator. Reviewers should NOT expect free latency on variant-triggering queries until Phase 4 ships.

---

## 10. Tests planned

Unit tests, all in `#[cfg(test)] mod tests` at the bottom of `src/narration.rs`. Plan:

### Predicate firing

1. `variants_fires_on_routing_flag` - hand-build a `RoutingDecision` with `needs_variants: true`, all other flags false. Tiny brain with a known goal seed and one neighbor (so DomainShift has an anchor). Call `narrate()`. Assert `result.variants.is_some()` and `result.variants.unwrap().num_candidates >= 1`.
2. `variants_suppressed_on_meta_query` - `needs_variants: true, is_meta_query: true`. Assert `result.variants.is_none()`.
3. `variants_suppressed_on_math_query` - `needs_variants: true, needs_math: true`. Assert `result.variants.is_none()`.
4. `variants_suppressed_below_activation_threshold` - configure `cfg.variant_min_activation = 0.99` so a normal spread can't clear it. Assert `result.variants.is_none()` even with the routing flag set.
5. `variants_suppressed_on_missing_goal_seed` - construct an `AskResult` with empty `seed_hits` AND empty `recall.activated`. Assert `result.variants.is_none()`.

### Shape selection

6. `shape_picks_multihypothesisanswer_on_variant_winner` - hand-build a `VariantArena` with `winner_id: Some(0)`, two `VariantSpec`s. Call `ShapeSelector::pick(QueryIntent::Definition, &ask, None, &flags, &cfg, None, Some(&arena))`. Assert `ResponseShape::MultiHypothesisAnswer`.
7. `shape_ignores_arena_without_winner` - same, but `winner_id: None`. Assert shape is NOT MultiHypothesisAnswer (falls through to intent).
8. `shape_ignores_arena_with_one_spec` - winner present but `variants.len() == 1`. Assert bias does not fire.
9. `shape_multithink_outranks_variants` - both `multi_think` hypothesis trace AND variant winner present. Assert MultiHypothesisAnswer still wins, BUT it comes from the multi-think path (rule 8), not from rule 9.4 - verified by leaving `arena.variants[0].anchor_seed` distinct from multi-think hypothesis and observing which label lands in `slots.primary` after extract_slots.

### Slot-fill

10. `extract_slots_populates_winner_alt_from_arena` - hand-build arena with two specs (winner at id 0, anchor labeled "alpha"; runner-up at id 1, anchor labeled "beta"). Call `extract_slots(..., None, Some(&arena))`. Assert `slots.primary == Some("alpha")` and `slots.winner_alt == Some("beta")` and `slots.winner_alt_count == 1`.
11. `extract_slots_skips_arena_when_multi_already_filled_primary` - arena present AND multi-think already populated `slots.primary`. Assert slot retains multi-think value (arena does not overwrite).

### Render

12. `render_multihypothesisanswer_articulates_variant_winner` - build slots with `primary: Some("alpha")`, `winner_alt: Some("beta")`, `winner_alt_count: 1`. Call `Renderer::render(MultiHypothesisAnswer, ...)`. Assert output contains both "alpha" and "beta" and the literal substring "considered 1" or similar from the template.

### NarrationResult round-trip

13. `narration_result_serde_round_trips_with_variants_none` - serialize a result with `variants: None`, deserialize, assert equality.
14. `narration_result_serde_round_trips_with_variants_some` - serialize a result with a populated `VariantArenaSummary`, deserialize, assert structural equality.

### Parity invariant (LOAD-BEARING)

15. `narrate_parity_no_variants_query` - call `narrate()` against a tiny brain with `query = "what is photosynthesis?"`. Confirm `result.routing.needs_variants == false`. Confirm `result.variants.is_none()`. Assert answer byte-identical to a captured pre-3b string literal.

Count: 15 new tests. Pre-3b baseline is the post-3a passing-test count. Target: all existing pass + 15 new.

---

## 11. Open questions for John

1. **Min-activation threshold.** Default `0.15`? Floor seems right for cleanliness-of-variant-pollution but is calibrated by reasoning, not empirics. Could relax to `0.05` if you want more arenas firing during demo. Section 3.
2. **Min-activation threshold as a `NarrationConfig` knob vs a `VariantConfig` knob.** Recommendation: NarrationConfig (it's a pipeline-gate decision, not an arena-engine decision). Confirm? Section 3 / 4.
3. **Goal seed: `seed_hits[0]` over top activation.** Verified `seed_hits[0]` is what Port #18a polysemy already uses; aligning. Confirm OK to anchor variants on the same hit? Section 1.
4. **Multi-think outranks variants.** Recommendation: rule 8 (critic-vetted multi-think) runs before rule 9.4 (variants), so a query that triggers both is rendered from multi-think data. Confirm this is the intended asymmetry? Section 5.
5. **No promotion in this port.** `Brain::promote_variant` exists; we are NOT calling it on the narrate path. The winner is observable in `NarrationResult.variants` but its variant-tagged keyframes remain Variant-scope only (decay naturally). Confirm? Section 12.
6. **No new shape variant.** `MultiHypothesisAnswer` reused. Same call as 3a Q3 - confirm explicitly for the variant case? Section 5.
7. **VariantArenaSummary vs full VariantArena on the wire.** Recommendation: summary (5 fields). Full arena lives only in transient call scope. Confirm? Section 7.
8. **What if `generate_variants` panics or returns an empty arena under unusual brain state?** The function is built to return an empty arena (no panic), but the predicate currently treats any returned arena as success. If we want to drop empty arenas from `NarrationResult.variants`, that's one extra guard. Recommendation: keep the summary surfaced even when empty (signal that the engine ran), with `no_winner: true` and `num_candidates: 0`. Confirm?

---

## 12. What is NOT in this port

- **No rayon parallelization.** Phase 4. The five strategy spreads run sequentially.
- **No critic vetting.** That is the `routing.needs_critic_vetting` flag, observation-only until a separate port.
- **No variant caching across queries.** Each variant-triggering query spawns a fresh arena.
- **No winner promotion.** `Brain::promote_variant` is NOT called from narrate. Winner keyframes stay Variant-scope only and decay naturally per `variant_decay_halflife_ms`. A future port could add an auto-promotion knob, but it would change the visibility contract of the bond graph - out of scope here.
- **No new ResponseShape variant.** Reuse `MultiHypothesisAnswer`.
- **No new ModulationFlags entry.** See section 8.
- **No changes to `variants.rs` or `brain.rs::generate_variants*`.** This port consumes the existing API; widening the variant engine itself is a separate concern.
- **No render-side change to other shapes.** Definition / Process / Synthesis are untouched.
- **No production deploy.** Daemon stays stopped. Build + tests + 4-prompt sanity check only.

---

## 13. LOC estimate breakdown

Net, post-3b:

- `src/narration.rs` insertion point: variant block + predicate + goal-seed derivation: ~25 lines.
- `ShapeSelector::pick` signature + new rule 9.4 + ~10 internal callsite updates: ~30 lines.
- `extract_slots` signature + new slot population block + ~8 internal callsite updates: ~35 lines.
- `VariantArenaSummary` struct + derive + `From<&VariantArena>` builder: ~30 lines.
- `NarrationResult` new field + return-site population (PASS path; refusal path stays `None`): ~10 lines.
- `NarrationConfig` new field `variant_min_activation: f32` + default: ~6 lines.
- 15 new unit tests: ~280 lines.

**Net: ~416 lines.** Inside the 300-500 LOC envelope.

---

## 14. Halt boundary

This document IS Phase A. Phase B (implementation) does NOT start until John responds to the open questions in section 11 and confirms the design. After that: branch -> implement -> tests -> build -> sanity -> halt-for-review, per the brief.

---

## 15. Resolved decisions (2026-05-27)

John accepted all eight open-question recommendations on 2026-05-27. Recorded here so the dials are visible if any of these need to be revisited mid-implementation or post-deploy.

| # | Decision | Knob to flip if changed |
|---|---|---|
| 1 | `variant_min_activation = 0.15` (clean default, low pollution) | `NarrationConfig::variant_min_activation` — lower to 0.05 for more arenas firing during demos |
| 2 | Threshold lives on `NarrationConfig`, not `VariantConfig` | Move field if arena-engine needs the gate too |
| 3 | Goal seed = `ask.seed_hits[0]`, fallback `ask.recall.activated[0].seed` | Swap to top-activation node if polysemy disambiguation diverges from variant intent |
| 4 | Rule 8 (multi-think critic-vetted) outranks rule 9.4 (variants) when both fire | Renumber rules in `ShapeSelector::pick` |
| 5 | No `Brain::promote_variant` call from narrate; winners stay in `TemporalScope::Variant{id}` and decay | Add a config flag `auto_promote_variant_winners` to opt in later |
| 6 | Reuse `MultiHypothesisAnswer` shape; no new variant added to `ResponseShape` enum | Add new shape variant if variant-rendered prose needs to diverge from multi-think prose |
| 7 | `NarrationResult.variants: Option<VariantArenaSummary>` (5 fields), not full `VariantArena` | Widen the summary struct or add a `full_arena` debug field |
| 8 | Empty arenas still surface a summary with `no_winner: true, num_candidates: 0` | Predicate could drop the field entirely on empty; current shape preserves "engine ran" signal |

Implementation may proceed with these as the working defaults. Any change at implementation time requires updating this section so the doc stays the source of truth.
