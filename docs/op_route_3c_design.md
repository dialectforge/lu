# OP-ROUTE-3c Design - Wire the Thinking Layer (ReAct loop) into narrate's PASS path

**Status:** Phase A design. HALT for John review before implementation.
**Branch (planned):** `feature/op-route-3c-thinking-layer` off `main` HEAD `80cf84d`.
**Scope:** Phase 3c of the routing+depth plan. Phases 3a (reasoning chains) and 3b (variants) merged; this port consumes a NEW routing flag `needs_agentic_thinking` (recommended) and invokes `Brain::think` on the PASS path, threading a compact `ThinkingTraceSummary` into shape selection, slot fill, and `NarrationResult`.
**Anti-LLM positioning:** preserved. The router stays pure heuristics; `Brain::think` is the existing ReAct loop over the live bond graph -- no new model surface. The new flag is decided from the query string alone.
**LOC budget:** ~400-600 net per the brief. Surface if creeping past 800.

---

## 0. Today vs after-3c

**Today (`src/narration.rs:2363-2417`):** `narrate()` already consumes `routing.needs_deep_reasoning` (OP-ROUTE-3a) and `routing.needs_variants` (OP-ROUTE-3b), threading `Option<&ReasoningTrace>` and `Option<&VariantArena>` into `ShapeSelector::pick`, `extract_slots`, and `NarrationResult`. The thinking layer (`crate::thinking::think` / `Brain::think`) is reachable ONLY via the standalone `lu4d_think` MCP tool at `src/mcp.rs:2584`. No routing flag currently maps to it.

**After 3c:** when the (new) `RoutingDecision.needs_agentic_thinking` is true, the idk_gate verdict is Pass, and a goal seed is recoverable, `narrate()` calls `brain.think(query, ThinkConfig { max_steps: cfg.thinking_max_steps, .. })`, collapses the returned `ThoughtTrace` into a compact `ThinkingTraceSummary`, and threads that summary through shape selection (new rule 9.3 -- biases to `MultiStep`), `extract_slots` (final-answer prose populates a chain-style articulation slot), and `NarrationResult.thinking` for callers.

Non-thinking queries: zero new work, byte-identical answer to pre-3c. Parity invariant load-bearing.

**Material new surface vs 3a/3b:** `Brain::think` is the most expensive port we have wired so far. It is `&mut self`, performs *up to* `max_steps` full `brain.ask` recalls plus per-step PLAN/ACT/REFLECT phases, and writes goal-Meta + Hypothesis + ConfirmedHypothesis keyframes into the graph. Unlike 3a (pure read) and 3b (single arena spawn, 5x baseline), 3c is a multi-spread mutation loop. Hard caps and a tight predicate are load-bearing for latency hygiene.

---

## 1. What does the thinking layer DO that reasoning + variants + multi-think don't?

Read of `src/thinking.rs:181-700`:

- **`reasoning.rs` (3a wiring)** is single-pass chain articulation: take a query, walk the bond graph, articulate a structured `WhyChain` of evidence steps from one recall. Pure-read, no graph mutation, no follow-on recalls.
- **`variants.rs` (3b wiring)** is single-shot perturbation: spawn an arena of 5 strategy-tagged keyframes around one goal seed, score each by composite (coherence / novelty / parsimony), elect a winner. One mutation, no iteration.
- **`agents.rs` (multi_think, rule 7 shape)** is parallel triangulation: spawn N hypothesis-bearing agents from one recall, run a Critic vetting pass, return surviving / rejected beliefs. One recall, one critic pass, N hypotheses in parallel.
- **`thinking.rs`** is a *bounded ReAct loop*: Observe (recall) -> Plan (propose hypothesis bonds between definitional and consequential nodes) -> Act (write Hypothesis-source keyframes) -> Reflect (pre-write spread + post-write spread; promote confirmed hypotheses to ConfirmedHypothesis weight; drop the rest) -- iterated up to `max_steps` (default 5), with early exit when `step.reflection.score >= confidence_to_answer` (default 0.7). The final spread (the loop's last OBSERVE) is returned as `ThinkResult.final_answer: AskResult` -- materially richer than the baseline recall because the loop has been *adding* (and confirming) graph structure for the goal across all prior iterations.

**Why narrate would call this on top of the other three:** it is the only port that *grows the graph in service of the current query*. Reasoning articulates what is already there. Variants perturbs around the goal seed but does not iterate. Multi-think enumerates hypotheses but does not write them durably. Thinking writes Hypothesis bonds, then *re-spreads* to see which of those structural guesses the rest of the graph actually supports, then keeps the winners and drops the losers -- a structural commit/rollback cycle. The right query for it is one where the user has explicitly asked Lu to *work through* the answer: "how do I X step-by-step," "figure out why Y," "walk me through Z," "think about this."

This is the differentiating port. It is also the only one of the four whose output (`ThinkResult.final_answer`) is itself an `AskResult` -- meaning thinking can in principle *replace* the baseline `ask` that narrate runs first, not merely augment it. Section 4 returns to this.

---

## 2. Routing flag -- reuse `needs_deep_reasoning` or add `needs_agentic_thinking`?

Two paths:

**(a) Reuse `needs_deep_reasoning` with a stronger predicate.** Today this flag triggers on `word_count >= 25`, `why+because`, multi-axis markers, explain-markers ("step by step," "walk me through," "in depth," "reason about"), or `sentence_count >= 3`. Several of those markers (step by step, walk me through, reason through) are *exactly* the agentic-thinking signal. We could split into a stricter sub-predicate (e.g. "step by step" AND `?` markers) and reuse the same boolean.

**(b) Add a NEW 9th boolean `needs_agentic_thinking`.** Triggers narrower than `needs_deep_reasoning`: explicit multi-step / action requests, "how do I," "step-by-step," "figure out," "work through," "think about," plus inferred markers like multiple action verbs in series.

**Recommendation: (b) NEW flag `needs_agentic_thinking`.**

Rationale:

- `needs_deep_reasoning` already maps cleanly onto `reasoning.rs` (chain articulation). Overloading it with thinking semantics would muddy the 3a wiring: should a 25-word query get *both* a chain AND a ReAct loop? Probably not -- the cost is too high and the user did not ask for it.
- The trigger surfaces are genuinely distinct: chain articulation fits causal / structural queries ("why," "because," multi-axis comparison); agentic thinking fits procedural / exploratory ones ("how do I," "step by step," "figure out"). A literal scan of the existing `needs_deep_reasoning` markers shows the procedural ones ("step by step," "walk me through," "reason through") are arguably mis-routed today -- they fire `deep_reasoning` which becomes a chain, but they would be better served by a think loop.
- A new flag preserves the parity invariant trivially: existing queries that fire `needs_deep_reasoning` still get a chain; the procedural subset that ALSO fires `needs_agentic_thinking` gets a chain AND a think loop, with shape-selection precedence deciding which wins on the wire (section 8).
- The brief explicitly leans this direction ("add `needs_agentic_thinking` (or similar)"). The router has a clean precedent for adding flags (3.5 needs_variants, 3.6 needs_critic_vetting); slotting a 9th in is mechanical.

**Trigger conditions for the new flag (literal markers in lowercased query):**
```
let markers = [
    "how do i ", "how do you ", "how can i ", "how would i ",
    "step by step", "step-by-step",
    "walk me through", "work through",
    "think about", "think through",
    "figure out", "work out",
    "plan how to", "plan for",
    "break this down", "break it down",
    "think it over", "reason through",
];
```

Note `"step by step"`, `"walk me through"`, and `"reason through"` ALSO fire `needs_deep_reasoning` today. That dual-fire is intentional: an explicit procedural query is both a chain candidate AND a think candidate; precedence is resolved at shape selection.

**`triggers` push:** `d.triggers.push("agentic_thinking".into())` when the new flag fires.

---

## 3. Fire predicate

**Recommendation:**
```rust
let should_think = routing.needs_agentic_thinking
    && !routing.is_meta_query
    && !routing.needs_math
    && goal_seed.is_some()
    && top_activation_of(goal_seed) >= cfg.thinking_min_activation;
```

The baseline (`needs_agentic_thinking && !is_meta_query && !needs_math`) mirrors 3a's and 3b's predicates -- the same 3-guard floor. The two additions:

- **`goal_seed.is_some()`** -- degenerate guard (empty seed_hits AND empty activated). Thinking creates a Meta goal node from the query string but the per-step `brain.ask` still recalls against the indexed graph; an empty initial recall is a no-signal case where the loop will spin without convergence.
- **Min-activation threshold `cfg.thinking_min_activation`** -- default `0.15` mirroring `variant_min_activation` (Section 6). The cost asymmetry is the justification: thinking is the most expensive port we wire; running 5 spreads on a typo / off-domain query produces 5 noisy plans worth of Hypothesis bonds that have to decay.

**4-condition AND (Refusal suppression):** confirmed. The block sits in the PASS branch (after `idk_gate_enabled` early-return at `src/narration.rs:2309-2360` upstream). On Refusal, narrate constructs the `NarrationResult` at `:2335-2360` and returns without reaching the 3a/3b/3c blocks. Thinking inherits this property structurally, identical to variants. Explicit Refusal-path test in Section 13 (test #2 below) locks it in.

Both tunables land on `NarrationConfig`, not `ThinkConfig`. Same argument as 3b: the threshold is a *pipeline-gate* knob ("is this query worth a think pass?"), not a *think-engine* knob ("how should the loop converge?").

---

## 4. Insertion point in `narrate()`

The 3a block sits at `src/narration.rs:2363-2380`, the 3b block at `:2381-2413`. The 3c block goes **after both**, immediately before `ShapeSelector::pick` at `:2415`. Pseudo-diff:

```
    // OP-ROUTE-3a: build a reasoning trace ...
    let reasoning_trace: Option<crate::reasoning::ReasoningTrace> = ...;
    // OP-ROUTE-3b: spawn variant arena ...
    let variant_arena: Option<crate::variants::VariantArena> = ...;
+   // OP-ROUTE-3c: run agentic ReAct loop when the router asked for
+   // thinking. Skips meta / math (same as 3a/3b). Capped at
+   // `cfg.thinking_max_steps` to bound latency. Refusal path never
+   // reaches this block.
+   let thinking_summary: Option<crate::thinking::ThinkingTraceSummary> =
+       if routing.needs_agentic_thinking
+           && !routing.is_meta_query
+           && !routing.needs_math
+       {
+           let goal_seed = ask.seed_hits.first().copied().or_else(|| {
+               ask.recall.activated.first().map(|a| a.seed)
+           });
+           goal_seed.and_then(|seed| {
+               let act = ask.recall.activated.iter()
+                   .find(|a| a.seed == seed)
+                   .map(|a| a.activation).unwrap_or(0.0);
+               if act < cfg.thinking_min_activation { return None; }
+               let t_cfg = crate::thinking::ThinkConfig {
+                   max_steps: cfg.thinking_max_steps,
+                   ..crate::thinking::ThinkConfig::default()
+               };
+               let result = brain.think(query, t_cfg);
+               Some(crate::thinking::ThinkingTraceSummary::from(&result))
+           })
+       } else {
+           None
+       };
    let shape = ShapeSelector::pick(
        intent, &ask, multi.as_ref(), &flags, cfg,
        reasoning_trace.as_ref(),
        variant_arena.as_ref(),
+       thinking_summary.as_ref(),
    );
```

**Order: thinking is ADDITIVE, after reasoning and variants. It does NOT subsume reasoning.**

The brief asks whether thinking should fire FIRST and skip reasoning. Considered and rejected:

- The two ports have different *outputs*. Reasoning produces a structured chain (`ReasoningTrace { steps, evidence_kind, final_answer }`). Thinking produces a richer recall (`ThinkResult.final_answer: AskResult`) plus a hypothesis-promotion trace. A query that triggers both is a query the user wants BOTH for: the chain articulates *why* and the loop articulates *how*. Suppressing reasoning when thinking fires would silently drop the chain articulation slot.
- The two ports have different *costs and confidence properties*. Reasoning is pure-read and deterministic over the current graph state. Thinking writes hypothesis bonds whose acceptance depends on REFLECT-phase spread. If thinking happens to score low (the loop did not converge), we still want the chain articulation to land in the prose. Running thinking first and gating reasoning on its outcome introduces fragility for marginal latency gain.
- The dual-trigger case is rare in practice. `needs_deep_reasoning` AND `needs_agentic_thinking` co-fires only on queries that hit both predicates (e.g. "walk me through why X causes Y because Z"). For those queries the user is explicitly asking for a thorough response and the latency cost is paid by intent.

Thinking is fired AFTER reasoning AND variants so that any future precedence rules in `ShapeSelector::pick` see the full set of optional trace artifacts in one signature call.

---

## 5. Goal seed derivation

**Recommendation: identical to 3b -- `ask.seed_hits[0]` with fallback to `ask.recall.activated[0].seed`.**

```rust
let goal_seed = ask.seed_hits.first().copied()
    .or_else(|| ask.recall.activated.first().map(|a| a.seed));
```

Note that `Brain::think` does NOT take a goal seed -- it takes the full query string and constructs its own Meta goal node internally (`src/thinking.rs:189-200`). The goal-seed derivation in narrate is therefore **only used for the min-activation gate** (section 3), not passed through to thinking. That keeps thinking's goal-anchoring identical between the narrate path and the standalone `lu4d_think` MCP tool: both anchor on the query string, neither on a pre-resolved seed.

**Why we still derive `goal_seed` for the gate, not just use top activation directly:** consistency with 3b. The min-activation threshold means "the subject of the query registered above noise." For variants the goal seed determines the arena's anchor; for thinking it does not -- but the *gate* (is this query worth a think pass) is the same question. Using the same derivation keeps the predicate code structurally aligned across 3b and 3c and the parity invariants symmetric.

Does thinking take a richer input than just the query? It already takes the full query string, derives its own scope from `brain.sessions`, and observes via `brain.ask` per-step. No richer input is needed; passing `intent` or `ask` through would add coupling without a use site inside the loop.

---

## 6. ThinkingConfig (NarrationConfig widening)

**Recommendation: widen `NarrationConfig` with TWO new fields, do not introduce a `ThinkConfig` sub-struct.**

New fields:
```rust
/// OP-ROUTE-3c -- hard cap on Observe-Plan-Act-Reflect iterations when
/// narrate() spawns a think pass. The thinking-layer default
/// (`ThinkConfig::max_steps = 5`) is intentionally retained for
/// standalone `lu4d_think` callers who opted in via the MCP boundary;
/// narrate-path think passes are bounded tighter to keep
/// user-facing latency under control. Default `3` (vs the engine's
/// default `5`). Tunable; do not exceed `5` without measuring the
/// 4-prompt sanity bench cold-cache latency first.
#[serde(default = "default_thinking_max_steps")]
pub thinking_max_steps: usize,

/// OP-ROUTE-3c -- minimum top-activation a query's goal seed must
/// clear before narrate() spawns a think pass. Mirrors
/// `variant_min_activation` (Section 3 of the 3b doc). Default `0.15`.
#[serde(default = "default_thinking_min_activation")]
pub thinking_min_activation: f32,
```

With `fn default_thinking_max_steps() -> usize { 3 }` and `fn default_thinking_min_activation() -> f32 { 0.15 }`.

Rationale:

- A `ThinkConfig` sub-struct on `NarrationConfig` would couple narrate tests to thinking-engine internals (same anti-pattern 3a and 3b rejected for their respective sub-configs).
- Two narration-pipeline knobs (steps cap, activation gate) are all narrate needs to make policy. The remaining `ThinkConfig` knobs (`hypotheses_per_step`, `confirmed_hypothesis_weight`, `confidence_to_answer`, etc.) are engine internals that the standalone `lu4d_think` tool already exposes and that narrate should NOT override -- a parity-with-standalone-tool argument identical to the one 3b made for `VariantConfig::default()`.
- The narrate path builds its `ThinkConfig` from the engine default + the narrate `max_steps` override:
```rust
let t_cfg = crate::thinking::ThinkConfig {
    max_steps: cfg.thinking_max_steps,
    ..crate::thinking::ThinkConfig::default()
};
```

**Hard cap default `max_steps: 3`** (versus engine default 5). Rationale: each step is a full `brain.ask` recall plus a PLAN/ACT/REFLECT pass. At 10.8M bonds, a single recall is ~50-200ms cold-cache; a 5-step loop is ~1-3s. 3 steps lands the loop in ~600ms-1.5s on triggering queries, which is the upper bound we should accept on user-facing narrate latency without parallelization. The brief explicitly asks "we cannot accept unbounded" and recommends "e.g. `max_thinking_steps: usize = 5`" -- I recommend 3 in 3c as a tighter narrate-specific cap, surfacing on `NarrationConfig` so it's a single-knob tune.

Thinking's own internal `MAX_HYPOTHESIS_BONDS_PER_GOAL = 32` (`src/thinking.rs:39`) is a separate hard cap on bond writes per goal; narrate does not need to override it (32 across 3 steps means ~10 writes per step at the per-step cap of 3, well-bounded).

---

## 7. Result handling

Three options on the table:

- **(a) Reuse `MultiHypothesisAnswer` shape (like 3b's arena).** Wrong domain. The thinking trace is a temporal sequence of Observe/Plan/Act/Reflect tuples, not an enumeration of alternatives.
- **(b) Reuse `CriticVettedAnswer` shape (like multi-think).** Wrong semantics. The Critic vets external belief claims; the thinking layer's REFLECT phase vets ITS OWN proposed hypothesis bonds against re-spread. Conflating produces caveats / vetted-counts in the rendered prose that are not what either path means.
- **(c) Add a new shape variant, e.g. `AgenticAnswer` or `ThinkingAnswer`.** Adds enum surface (one new `ResponseShape` variant + one new render arm) but separates the rendering path cleanly.
- **(d) Reuse the existing `MultiStep` shape.** Already exists (rule 8 in current ordering). Its render arm articulates a multi-step response (perfect domain match for a ReAct-loop trace), and it is currently underused -- rule 8 is the "multi present, no vetted" fallthrough, which is a narrow case.

**Recommendation: (d) reuse `MultiStep` -- with rule 9.3 in the shape selector biasing toward it when a converged thinking trace is available.**

Rationale:

- The shape, its enum variant, AND its render arm already exist and articulate exactly the structure of a multi-step thought (per the rule 8 docstring). Zero new shape, zero new render arm.
- Domain match is good: a ReAct-loop's user-facing artifact IS a multi-step process narration. The current rule-8 trigger (`multi.vetted.is_empty() && !multi.recall.is_empty()`) is a structurally weaker signal for the same shape than "think loop converged with promoted hypotheses."
- Avoids enum surface growth that the 3b doc explicitly resisted (Section 5 / Section 15 dial #6). Same anti-bloat discipline.
- One nuance, parallel to 3b's: when **both** `multi_think` produces hypotheses-with-Critic-work AND thinking produces a converged trace, multi-think wins (rule 7 fires before the new 9.3 -- section 8). This preserves the existing critic-vetting precedence; thinking only drives shape selection when multi-think did NOT fire.

**Slot mapping (in `extract_slots`):** the thinking trace surfaces its `final_answer.recall.activated[0].content` as the chain-articulation slot. Note this OVERLAPS with 3a's `chain_articulation` slot. Two options:

- Add a second slot, e.g. `thinking_articulation: Option<String>`. Cleaner but adds slot surface.
- Reuse `chain_articulation`, with 3a winning if both fire (3a sits earlier in extract_slots and the slot is `is_none()`-guarded).

**Recommendation: reuse `chain_articulation` with 3a precedence.** A converged think trace's `final_answer` is the same *kind* of artifact as a structured reasoning chain's `final_answer` -- both are best-effort articulations of the brain's response to the query. The 3a chain is the higher-confidence one (Structured evidence kind, ≥2 steps, deterministic over current graph); when 3a fired, its articulation should win. When only 3c fires, its articulation lands in the slot unopposed. The slot-fill code only needs a guard symmetric to 3a's:

```rust
// OP-ROUTE-3c: surface final-answer prose from the thinking trace
// when one converged. Guarded by chain_articulation.is_none() so 3a
// retains precedence when both fired.
if let Some(t) = thinking {
    if s.chain_articulation.is_none()
        && t.steps_taken >= 1
        && t.final_answer_present
    {
        s.chain_articulation = t.final_answer_text.clone();
    }
}
```

This is why `ThinkingTraceSummary` carries `final_answer_text: Option<String>` (the top-activation node's content from `ThinkResult.final_answer.recall.activated[0]`) instead of the full `AskResult` -- the summary is enough for the slot fill, and we don't pay the wire cost of the full recall.

---

## 8. ShapeSelector rule for converged thinking

**Rule number: 9.3** (sits between rule 9 "two-hop synthesis" and the existing rule 9.4 "variant-winner MultiHypothesisAnswer"). So precedence in `ShapeSelector::pick`:

```
1. Refusal (no activations)
2-3. MetaSelf / TimetravelComparison
4. Bridge
5. Wondering
6. UncertainAnswer
7. Critic-vetted multi-think (rule 4 in docstring numbering)
8. (intent-classifier-driven multi-step fallthrough)
9. Two-hop synthesis
9.3. OP-ROUTE-3c: thinking converged => MultiStep    <-- NEW
9.4. OP-ROUTE-3b: variant winner => MultiHypothesisAnswer
9.5. OP-ROUTE-3a: structured chain => WhyChain
10-15. Intent fallthrough
```

Why 9.3 (above 9.4 variants, above 9.5 reasoning):

- An agentic think loop that *converged* (early_exit true OR steps_taken == max_steps with promoted hypotheses > 0) is a stronger signal that the brain produced a multi-step answer than either a variant winner or a single chain. The loop wrote AND confirmed structural guesses across multiple iterations.
- Below 9.4 / 9.5 would mean a co-firing variant arena or chain would steal the shape from a converged loop -- wrong asymmetry. Think loops are deliberately the most expensive port; if we paid the cost, the result deserves the shape.
- Above rule 7 (critic-vetted multi-think) would be wrong: multi-think with real critic work is higher confidence (parallel triangulation with vetting) than ReAct convergence. Same logic the 3b doc applied for rule 8 vs 9.4.

**Signature extension:**
```rust
pub fn pick(
    intent: QueryIntent,
    ask: &AskResult,
    multi: Option<&MultiAgentTrace>,
    flags: &ModulationFlags,
    cfg: &NarrationConfig,
    reasoning: Option<&crate::reasoning::ReasoningTrace>,
    variants: Option<&crate::variants::VariantArena>,
    thinking: Option<&crate::thinking::ThinkingTraceSummary>,  // NEW
) -> ResponseShape
```

**New rule body, inserted at the line after rule 9's `return ResponseShape::Synthesis;` and before the existing rule 9.4 variants bias:**
```rust
// 9.3 -- OP-ROUTE-3c: a converged thinking pass is the strongest
// signal we have that the brain produced a multi-step answer.
// "Converged" = at least one step taken AND (early_exit OR at least
// one hypothesis was promoted). A loop that ran max_steps without
// promoting anything is structurally inconclusive -- fall through
// to other rules.
if let Some(t) = thinking {
    let converged = t.steps_taken >= 1
        && (t.early_exit || t.hypotheses_promoted >= 1);
    if converged {
        return ResponseShape::MultiStep;
    }
}
```

Guards:
- `steps_taken >= 1` -- the loop actually ran. Guard against the degenerate case where the loop exited at step 0.
- `early_exit || hypotheses_promoted >= 1` -- the loop either hit `confidence_to_answer` early (a strong signal) OR produced at least one ConfirmedHypothesis (a weaker but still real signal). A loop that ran max_steps without promoting anything saw no structural support for any of its proposals; that's a noisy outcome and should not steal the shape.

All OTHER `ShapeSelector::pick` call sites get `None` for the new arg, preserving parity. Per 3a/3b's count, ~10 internal callsites to update.

---

## 9. NarrationResult.thinking field

**Schema: `Option<ThinkingTraceSummary>`** (not the full `ThinkResult` / `ThoughtTrace`).

Rationale for a compact summary: `ThoughtTrace` carries the full per-step `Observation` (top seeds + activations + bridges), per-step `Plan` (proposed `HypothesisCandidate`s with from/to seeds and contents), and per-step `Reflection` (confirmed and unconfirmed candidates). For 5 steps that's potentially 100+ seed references plus content strings -- heavy on the JSON wire and useless to most callers, who want "did thinking converge, what was the answer."

```rust
/// OP-ROUTE-3c -- compact summary of a `ThoughtTrace` that landed on
/// a narrate PASS-path query. None when the router did not request
/// agentic thinking, or when the request was guarded out (meta / math /
/// missing goal seed / below activation threshold). Backward-compat:
/// older NarrationResult JSON without this field round-trips clean
/// via `#[serde(default)]`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ThinkingTraceSummary {
    /// Number of Observe-Plan-Act-Reflect iterations that ran. Bounded
    /// above by `NarrationConfig::thinking_max_steps`.
    pub steps_taken: usize,
    /// True when the loop hit `confidence_to_answer` and exited before
    /// `max_steps`. False when the loop ran to the cap.
    pub early_exit: bool,
    /// Reflection score on the final iteration (`confirmed / proposed`).
    /// 0.0 when no plan had any hypotheses to propose on that step.
    pub final_score: f32,
    /// Total Hypothesis-source keyframes written across all steps.
    /// Bounded above by `thinking::MAX_HYPOTHESIS_BONDS_PER_GOAL = 32`.
    pub hypothesis_bonds_written: usize,
    /// Subset of `hypothesis_bonds_written` that the REFLECT phase
    /// promoted to ConfirmedHypothesis weight.
    pub hypotheses_promoted: usize,
    /// `true` when `final_answer.recall.activated` is non-empty.
    /// Lets callers distinguish "loop ran, no recall" from "loop ran,
    /// recall present."
    pub final_answer_present: bool,
    /// Top-activation content from the final recall, if any. Used by
    /// extract_slots to populate `chain_articulation` when 3a did not.
    /// None when `final_answer.recall.activated` was empty.
    #[serde(default)]
    pub final_answer_text: Option<String>,
    /// Aggregate confidence -- copy of `final_score` for parity with
    /// `ReasoningTrace.confidence`. Carried separately so callers can
    /// surface it without coupling to think-engine semantics.
    pub confidence: f32,
}

impl From<&crate::thinking::ThinkResult> for ThinkingTraceSummary {
    fn from(r: &crate::thinking::ThinkResult) -> Self {
        let final_answer_text = r.final_answer.recall.activated.first()
            .map(|a| a.content.clone());
        ThinkingTraceSummary {
            steps_taken: r.trace.steps.len(),
            early_exit: r.trace.early_exit,
            final_score: r.trace.final_score,
            hypothesis_bonds_written: r.trace.hypothesis_bonds_written,
            hypotheses_promoted: r.trace.hypotheses_promoted,
            final_answer_present: !r.final_answer.recall.activated.is_empty(),
            final_answer_text,
            confidence: r.trace.final_score,
        }
    }
}
```

Note `ThinkingTraceSummary` lives in `src/thinking.rs` alongside the existing structs, not in `src/narration.rs`. The `From<&ThinkResult>` impl can stay there. Mirrors how `VariantArenaSummary` lives in `src/variants.rs` (line 582).

Add to `NarrationResult` parallel to `reasoning` and `variants`:
```rust
pub struct NarrationResult {
    ...
    #[serde(default)]
    pub reasoning: Option<crate::reasoning::ReasoningTrace>,
    #[serde(default)]
    pub variants: Option<crate::variants::VariantArenaSummary>,

    /// OP-ROUTE-3c -- compact summary of the `ThoughtTrace` built
    /// when `routing.needs_agentic_thinking` was true and the PASS
    /// path elected to run a think loop. None when thinking did not
    /// run.
    #[serde(default)]
    pub thinking: Option<crate::thinking::ThinkingTraceSummary>,
}
```

Note the asymmetry with `reasoning`: 3a serializes the full `ReasoningTrace` on the wire (full structure). 3c serializes a summary. The wire-cost asymmetry reflects the artifact-size asymmetry: a `ReasoningTrace` is bounded by `MAX_CHAIN_LEN` steps, each step a small struct; a `ThoughtTrace` is up to 5 steps each carrying a full Observation/Plan/Reflection tuple with up to ~50 seeds per Observation. We do not put the full thing on the wire.

---

## 10. ModulationFlags impact

**Recommendation: no new flag in this port.**

Same argument as 3a Section 6 and 3b Section 8: `ModulationFlags` is the *recall-state* signal surface (curiosity / bridge / hypothesis_only / critic_vetted / wondering). A converged think loop is a *post-recall agentic mutation*, not a recall-state signal. Mixing them blurs the layer.

The one nominally adjacent flag is `hypothesis_only` (already exists). It indicates the *current recall* leaned on Hypothesis-source bonds. After 3c fires, the post-think `brain.ask` (which narrate runs upstream of the think block) does NOT re-run; the recall the rest of narrate sees is pre-think. So `hypothesis_only` continues to reflect the pre-think recall state, which is what shape selection (rules 5-7) needs. No interaction.

If a future port wants to surface "think loop converged with N promoted hypotheses" as a modulation signal for prose tone, that's a separate change. Out of scope here.

`signals_used` does not gain an entry. `routing.triggers` already carries `"agentic_thinking"` when the router fires; callers wanting "did a think loop actually converge" check `result.thinking.as_ref().map(|t| t.early_exit || t.hypotheses_promoted >= 1)`.

---

## 11. Performance caveat

**Thinking is sequential and the most expensive port we have wired.** Each iteration runs:

1. A full `brain.ask(query, 3, top_n_observations)` recall.
2. A `propose_hypotheses` pass over the observation (cheap; pure CPU).
3. A pre-write `spread_activation_weighted` from the top-3 observation seeds (a second recall).
4. Write up to `hypotheses_per_step` Hypothesis-source `TemporalBond`s into the graph (`&mut self`).
5. A post-write spread to vet which writes the brain supports structurally.
6. Promotion of confirmed hypotheses to ConfirmedHypothesis weight (more `&mut self`).

So per step we run ~2-3 spreads of comparable cost to the baseline `brain.ask`. At `max_steps: 3` (the recommended narrate cap), that is **~6-9x the baseline PASS-path recall cost**, plus the keyframe writes.

**Expected latency impact on a triggering query: 6-15x the pre-3c PASS-path narrate latency.** This is significantly worse than 3b's 3-6x (variant arenas) and worlds worse than 3a's near-zero (reasoning is read-only).

This is paid by *intent*: `needs_agentic_thinking` fires only on queries that explicitly ask the brain to work through a problem. The user-visible expectation on those queries is already "this will take a moment." Non-triggering queries are unaffected (predicate-gated; zero new work).

**Out of scope:** Phase 4 will parallelize where it can (the per-step pre-write and post-write spreads might collapse into one, the hypothesis-write loop might shard with rayon). This port lands the sequential path and the predicate gate. Reviewers should NOT expect free latency on thinking-triggering queries until Phase 4 ships.

**Bench discipline:** as part of Phase D sanity, a 4-prompt cold-cache latency bench must include at least one `needs_agentic_thinking`-triggering query and confirm the wall-clock budget. If the bench shows worse than 12x baseline latency on a triggering query with `max_steps: 3`, halt and re-tune (likely drop `max_steps` to 2, or raise `thinking_min_activation` to 0.25).

---

## 12. Tests planned

Unit tests, all in `#[cfg(test)] mod tests` at the bottom of `src/narration.rs` and `src/query_router.rs`. Plan:

### Query-router predicate (new, in `query_router.rs`)

1. `router_fires_agentic_thinking_on_how_do_i` -- query `"How do I deploy a Rust binary step by step?"`. Assert `d.needs_agentic_thinking == true` AND `d.triggers.iter().any(|t| t == "agentic_thinking")`.
2. `router_suppresses_agentic_thinking_on_definition` -- query `"What is mitosis?"`. Assert `d.needs_agentic_thinking == false`.
3. `router_co_fires_agentic_and_deep_on_walk_me_through` -- query `"Walk me through how photosynthesis works step by step."`. Assert BOTH `needs_agentic_thinking == true` AND `needs_deep_reasoning == true` (intentional dual-fire per Section 2).

### Predicate firing in narrate

4. `thinking_fires_on_routing_flag` -- hand-build a `RoutingDecision` with `needs_agentic_thinking: true`, all other flags false. Tiny brain with a known seed and goal-anchor activation above threshold. Call `narrate()`. Assert `result.thinking.is_some()` and `result.thinking.unwrap().steps_taken >= 1`.
5. `thinking_suppressed_on_meta_query` -- `needs_agentic_thinking: true, is_meta_query: true`. Assert `result.thinking.is_none()`.
6. `thinking_suppressed_on_math_query` -- `needs_agentic_thinking: true, needs_math: true`. Assert `result.thinking.is_none()`.
7. `thinking_suppressed_on_refusal_path` -- `idk_gate_enabled: true` AND a brain so small the gate refuses. Assert `result.thinking.is_none()` even with `needs_agentic_thinking: true`.
8. `thinking_suppressed_below_activation_threshold` -- set `cfg.thinking_min_activation = 0.99`. Assert `result.thinking.is_none()`.
9. `thinking_suppressed_on_missing_goal_seed` -- empty `seed_hits` and empty `recall.activated`. Assert `result.thinking.is_none()`.

### Max-steps cap

10. `thinking_max_steps_cap_honored` -- set `cfg.thinking_max_steps = 2`. Construct a brain where the loop would otherwise run all 5 steps (no early-exit). Assert `result.thinking.unwrap().steps_taken <= 2`.

### Convergence (halting condition)

11. `thinking_early_exit_recorded` -- brain configured so the first step's REFLECT score crosses `confidence_to_answer: 0.7`. Assert `result.thinking.unwrap().early_exit == true` and `steps_taken == 1`.

### Shape selection

12. `shape_picks_multistep_on_converged_thinking` -- hand-build `ThinkingTraceSummary { steps_taken: 2, early_exit: true, hypotheses_promoted: 1, .. }`. Call `ShapeSelector::pick(QueryIntent::Generic, &ask, None, &flags, &cfg, None, None, Some(&t))`. Assert `ResponseShape::MultiStep`.
13. `shape_ignores_unconverged_thinking` -- summary with `early_exit: false, hypotheses_promoted: 0`. Assert shape falls through (not MultiStep from rule 9.3).
14. `shape_multi_think_outranks_thinking` -- both critic-vetted `multi` AND converged `thinking` present. Assert shape comes from multi-think (rule 7), not from rule 9.3.

### Slot fill

15. `extract_slots_populates_chain_articulation_from_thinking` -- thinking summary with `final_answer_text: Some("alpha")`, NO reasoning trace. Assert `slots.chain_articulation == Some("alpha")`.
16. `extract_slots_thinking_yields_to_reasoning_on_overlap` -- both reasoning trace (Structured, 2 steps) AND thinking summary present. Assert `slots.chain_articulation` came from reasoning, not thinking.

### NarrationResult round-trip

17. `narration_result_serde_round_trips_with_thinking_none` -- serialize a result with `thinking: None`, deserialize, assert equality.
18. `narration_result_serde_round_trips_with_thinking_some` -- serialize a result with a populated `ThinkingTraceSummary`, deserialize, assert structural equality.

### Parity invariant (LOAD-BEARING)

19. `narrate_parity_no_thinking_query` -- call `narrate()` against a tiny brain with `query = "what is photosynthesis?"`. Confirm `result.routing.needs_agentic_thinking == false`. Confirm `result.thinking.is_none()`. Assert answer byte-identical to a captured post-3b string literal.

Count: 19 new tests (4 in query_router.rs, 15 in narration.rs). Pre-3c baseline is the post-3b passing-test count. Target: all existing pass + 19 new.

---

## 13. Open questions for John

1. **New routing flag `needs_agentic_thinking` (Section 2 recommendation).** Confirm we are adding a 9th boolean to `RoutingDecision` rather than overloading `needs_deep_reasoning`?
2. **Trigger marker list (Section 2).** Reasonable starting set or want to tighten / widen? In particular, "how do i" is a wide net -- could narrow to require an action verb downstream.
3. **`thinking_max_steps` default = 3 (Section 6).** Engine default is 5; narrate cap of 3 is tighter for latency. Bench-validatable. Confirm 3?
4. **`thinking_min_activation` default = 0.15 (Section 3).** Identical to `variant_min_activation`. Confirm parity?
5. **Insertion order: 3c AFTER 3a/3b, additive, no subsumption (Section 4).** A query that triggers `needs_deep_reasoning` AND `needs_agentic_thinking` runs BOTH. Confirm this dual-cost is intended on co-firing queries?
6. **Reuse `MultiStep` shape, no new `ResponseShape` variant (Section 7).** Matches 3b's no-new-shape decision. Confirm?
7. **Slot reuse: `chain_articulation` with 3a precedence (Section 7).** A converged think trace's final-answer prose lands in the same slot as a reasoning chain's articulation. 3a wins on overlap. Confirm, or want a separate `thinking_articulation` slot?
8. **`ThinkingTraceSummary` (compact) on the wire, full `ThoughtTrace` only in transient call scope (Section 9).** Asymmetric with 3a (which serializes the full trace) but symmetric with 3b's `VariantArenaSummary`. Confirm?
9. **Convergence definition for rule 9.3 (Section 8).** `steps_taken >= 1 && (early_exit || hypotheses_promoted >= 1)`. Strict enough? Want a stronger floor (e.g. `hypotheses_promoted >= 2`)?
10. **Latency-bench gate: 12x baseline ceiling on triggering queries (Section 11).** Halt-and-retune trigger if exceeded. Confirm threshold?
11. **No promotion to canonical scope from the narrate think path.** `Brain::think` already writes ConfirmedHypothesis keyframes (a kind of promotion within hypothesis scope), but we do NOT promote those to baseline canonical bond strength here. Confirm same dial as 3b's no-promotion stance?

---

## 14. What is NOT in this port

- **No rayon parallelization of the ReAct loop.** Phase 4. The 3-step loop runs sequentially.
- **No code-intent special handling.** `routing.code_intent` stays observation-only (Phase 6).
- **No file or image dispatch.** Phase 5.
- **No promotion of thinking-derived ConfirmedHypothesis keyframes to canonical scope.** They remain Hypothesis-scope and decay per the existing temporal half-life.
- **No new `ResponseShape` variant.** Reuse `MultiStep`.
- **No new `ModulationFlags` entry.** See section 10.
- **No changes to `thinking.rs` semantics.** The narrate path consumes the existing `Brain::think` API as-is. `ThinkingTraceSummary` and its `From` impl are the only additions to that module.
- **No render-side change to other shapes.** Definition / Process / Synthesis / Wondering / WhyChain / MultiHypothesisAnswer / CriticVettedAnswer are untouched.
- **No bench harness change.** The 4-prompt sanity bench gains one think-triggering prompt; the harness itself is unchanged.
- **No production deploy.** Daemon stays stopped, training stays paused. Build + tests + 4-prompt sanity check only.
- **No autopromote of variant_mode in thinking.** `ThinkConfig::variant_mode` stays `false` on the narrate path (the brief notes 3c and 3b are sibling capabilities; variant mode would double-couple them through the engine).

---

## 15. LOC estimate breakdown

Net, post-3c:

- `src/query_router.rs`: new field on `RoutingDecision`, new helper `needs_agentic_thinking()`, dispatch in `route()`, ~4 tests: ~70 lines.
- `src/narration.rs` insertion point: think block + predicate + goal-seed derivation: ~32 lines.
- `ShapeSelector::pick` signature + new rule 9.3 + ~10 internal callsite updates: ~32 lines.
- `extract_slots` signature + new slot-population block + ~8 internal callsite updates: ~30 lines.
- `src/thinking.rs`: `ThinkingTraceSummary` struct + `From<&ThinkResult>` impl: ~50 lines.
- `NarrationResult` new field + return-site population (PASS path; refusal path stays `None`): ~12 lines.
- `NarrationConfig` new fields `thinking_max_steps`, `thinking_min_activation` + defaults: ~14 lines.
- 15 new unit tests in `narration.rs` (tests 4-19 above): ~310 lines.

**Net: ~550 lines.** Inside the 400-600 LOC envelope, with the test surface dominant (consistent with 3a / 3b).

---

## 16. Halt boundary

This document IS Phase A. Phase B (implementation) does NOT start until John responds to the open questions in section 13 and confirms the design. After that: branch -> implement -> tests -> build -> sanity bench (with at least one think-triggering prompt) -> halt-for-review, per the brief.

Daemon STAYS STOPPED. Training STAYS PAUSED. No new branch created in this pass.

---

## 17. Resolved decisions (2026-05-27)

John reviewed the 11 open questions on 2026-05-27. Accepted 10 of 11 agent recommendations; flipped #3 (max_steps default).

| # | Decision | Knob to flip if changed |
|---|---|---|
| 1 | Add NEW 9th boolean `needs_agentic_thinking` to `RoutingDecision`, do not overload `needs_deep_reasoning` | Remove flag, route on existing `needs_deep_reasoning` with widened predicate |
| 2 | Trigger marker set accepted as starting point ("how do i", "step by step", "walk me through", "figure out", "think through") | Tighten in `QueryRouter::route` — narrow "how do i" to require a trailing action verb |
| 3 | **`thinking_max_steps = 5`** (engine default, NOT agent-recommended 3). John flipped this — wants thinking to have room to actually think on user-facing narrate; accepting the higher latency multiplier (up to ~20-25x baseline on max-step queries) | `NarrationConfig::thinking_max_steps` — lower to 3 if latency proves a real ceiling in bench |
| 4 | `thinking_min_activation = 0.15` (parity with `variant_min_activation`) | `NarrationConfig::thinking_min_activation` |
| 5 | Insertion order: 3c AFTER 3a/3b, additive, no subsumption. Co-firing queries pay all three latencies | Reorder in `narrate()`; add subsumption guard if 3a/3b output is sufficient |
| 6 | Reuse `MultiStep` shape, no new `ResponseShape` variant | Add new shape variant if thinking-rendered prose needs distinct rendering arm |
| 7 | Slot reuse: `chain_articulation` with 3a precedence on overlap | Add separate `thinking_articulation` slot if rendered prose needs to differ from reasoning chain |
| 8 | `ThinkingTraceSummary` (compact, 8 fields) on the wire, full `ThoughtTrace` only in transient call scope | Widen summary or add full-trace debug field |
| 9 | Convergence: `steps_taken >= 1 && (early_exit \|\| hypotheses_promoted >= 1)` | Raise floor to `hypotheses_promoted >= 2` for stricter shape rule |
| 10 | Latency-bench gate: halt-and-retune if triggering queries exceed 12x baseline. With max_steps=5 the budget is tighter — may trip at first bench | Raise ceiling or accept-and-flag if 5-step thinking legitimately wants more headroom |
| 11 | No promotion to canonical scope from narrate think path. ConfirmedHypothesis keyframes stay in hypothesis-scope. Same stance as 3b decision #5 | Add `auto_promote_thinking_hypotheses` config flag to opt in later |

**Headline flip:** Decision #3 — John chose `max_steps = 5` (engine default) over the agent-recommended `3`. Rationale: if we are paying the latency to wire thinking into narrate at all, give it the headroom to actually converge. Tracks the broader feedback pattern of distinguishing OPTIMIZATION gates (latency targets) from HARDWARE gates (correctness); 5 steps is an optimization knob, not a structural constraint. If decision #10 latency bench trips repeatedly, that is the signal to revisit, not a default-tighten.

Implementation may proceed with these as working defaults.
