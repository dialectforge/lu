# OP-NARRATE-1 Phase B — Design Surface

**Target:** `src/narration.rs` @ HEAD `ea4889b` (3691 LOC)
**Status:** DESIGN ONLY — halt for John's review before implementation.
**Diagnosis (Phase A, locked):** `apply_style_modulations` violates its own anti-hallucination docstring. The verbosity branch at `narration.rs:2099-2107` truncates rendered prose at the first `". "`, deleting facts. The empathy/technicality/verbosity branches at `:2002`, `:2080`, `:2093`, `:2114` inject fabricated content. Architectural answer: delete the post-processor; make `render_core` style-aware.

---

## 1. Signature change

Today (`narration.rs:1654`):
```rust
fn render_core(shape: ResponseShape, s: &ShapeSlots) -> String
```

Proposed:
```rust
fn render_core(shape: ResponseShape, s: &ShapeSlots, hints: &StyleHints) -> String
```

- **No new StyleHints fields needed.** Current fields cover the surface (`max_response_length_factor`, `formality`, `technicality`, `empathy`, `samples_count`). All four were *consulted* by `apply_style_modulations` and are sufficient to drive per-shape rendering.
- **No StyleThresholds parameter.** Thresholds (`0.4`, `0.7`) are stable across shapes and small enough to live as module-scope `const` (`VERBOSITY_TERSE_CUTOFF: f32 = 0.4;`). Configurability is a YAGNI risk — re-introduce a config struct only if a real consumer asks.
- **`StyleHints::tier()` helper** added to `src/style/mod.rs` (~10 LOC):
  ```rust
  pub enum VerbosityTier { Terse, Standard }
  impl StyleHints {
      pub fn verbosity_tier(&self) -> VerbosityTier { ... }
  }
  ```
- **Builder signature** at `narration.rs:1640-1647` (`ResponseShape::render`) threads `hints` through to `render_core`. `apply_prefix_modulations` and `apply_suffix_modulations` stay unchanged (they don't fabricate — prefix is `"Tentatively:"` tied to a real flag, suffix is curiosity/bridge/critic/scar tied to real signal).

---

## 2. Threshold mapping

**Two tiers only.** Brief preference is correct: fewer tiers = less surface area, and Phase A diagnostic data shows the live brain sits at `effective.verbosity = 0.304` (terse) and the only failure mode that matters is "terse must not delete facts." A middle "standard" tier adds a third render arm per shape with no behavior change vs. verbose.

```
factor >= 0.4 → Standard  (today's verbose template, unchanged)
factor <  0.4 → Terse     (new degraded form per shape, facts preserved)
```

The `> 0.7` "append more-to-unpack" branch dies with `apply_style_modulations` — it's fabricated content, not a real expansion.

**Formality (contraction expansion)**, **technicality (jargon/plain markers)**, **empathy (acknowledgment prefix)** all DIE. Reasons:
- Empathy prefix `"That's a good question —"` is fabricated relational content. There is no signal in the brain that this user asked a *good* question; it's a tic.
- Technicality markers `"...canonical formulation."` / `"In plain terms: this means the relationship holds simply."` are content-free filler that asserts a relationship without grounding.
- Formality contraction expansion is the *only* style branch that is technically truth-preserving (mechanical text substitution). It is still cut for this refactor because (a) it operates on already-rendered prose, fighting the architecture, and (b) the templates barely use contractions to begin with — grep across the 16 render arms shows `I've`, `I'm`, `there's`, `haven't`, `couldn't` in template strings — formality belongs in the per-shape template choice, not post-hoc substitution. Adding a formality tier is a separate future port if any user signal demands it; for now NEUTRAL behavior on formality is correct.

Net: **only verbosity drives terse-form selection.** The other three dimensions become no-ops until a future port adds them per-shape with real grounding.

---

## 3. Per-shape terse-form table (THE LOAD-BEARING CALL)

Cited line is the start of the render arm in `narration.rs`. Verbose form = today's output (unchanged). Terse form preserves every concrete slot; drops connective tissue, hedges, meta-commentary, and source-of-confidence prose.

| # | Shape | Line | Verbose form (today) | Terse form (proposed) |
|---|-------|------|----------------------|------------------------|
| 1 | Definition | 1656 | `{primary} is {secondary}.` *or* `walker_prose` | **walker_prose passthrough preserved.** Fallback: `{primary} is {secondary}.` (already terse — no change.) |
| 2 | Process | 1669 | walker_prose *or* `{query} works in {n} steps: first {step1}, then {step2}, then {step3}.` | walker_prose passthrough. Fallback: `{step1}; {step2}; {step3}.` (drops "works in N steps: first/then/then" connective tissue) |
| 3 | WhyChain | 1689 | `{primary} happens because {secondary}, which leads to {tertiary}, producing {primary}.` | `{primary}: {secondary} -> {tertiary}.` |
| 4 | Comparison | 1698 | walker_prose *or* `{a} and {b} share {shared}. They differ in how each relates to {tertiary}.` | walker_prose passthrough. Fallback: `{a} and {b}: shared {shared}; differ on {tertiary}.` |
| 5 | FactRecall | 1720 | `{primary}.` | **Already terse — no change.** |
| 6 | UncertainAnswer | 1723 | `I think it might be {primary}, but my support for that is thin. More associations to {query} would help close the gap.` | `Possibly {primary} (low support).` |
| 7 | MultiHypothesisAnswer | 1730 | `Most likely {primary}. I considered {N} alternatives — for example, {alt} — but the top candidate had stronger support.` | `Most likely {primary} (vs. {N} alternatives e.g. {alt}).` |
| 8 | BridgedAnswer | 1738 | `{primary}. While answering, I noticed a connection between {bridge_low} and {bridge_high}: they sit in different clusters but a weak link bridges them.` | `{primary}. Bridge: {bridge_low} <-> {bridge_high}.` |
| 9 | CriticVettedAnswer | 1746 | `{primary}. My critic caught {N} caveats: {caveat}. Confidence: {conf} ({label}).` | `{primary}. Caveats ({N}): {caveat}. Conf: {conf}.` |
| 10 | Refusal | 1758 | `I don't have a meaningful answer to "{query}". No bonds in my graph activated for this query.` | `No answer for "{query}" — no bonds activated.` |
| 11 | MetaSelf | 1764 | `On "{query}", I have {N} concepts and {M} associations. Confidence: {label}. Recently I activated {primary}.` | `"{query}": {N} concepts, {M} bonds, conf={label}, recent={primary}.` |
| 12 | TimetravelComparison | 1774 | `About "{query}": at an earlier point I primarily associated this with {secondary}. Now: {primary}. The shift came from added associations during recent sessions.` | `"{query}": was {secondary}, now {primary}.` |
| 13 | Synthesis | 1782 | walker_prose `+ Confidence: {conf}.` *or* `{primary} relates to {tertiary}. I'm inferring this from {via} — both are connected through that intermediate. Confidence: {conf}.` | walker_prose passthrough `+ Conf: {conf}.` Fallback: `{primary} -> {via} -> {tertiary} (conf {conf}).` |
| 14 | Wondering | 1801 | `{body} (I've been wondering about {wondering_about} — {wondering_kind}; it touches this query as a gap in my knowledge.)` | `{body} (gap: {wondering_about}.)` |
| 15 | **MultiStep** | 1816 | `Several parts to this. First: {primary}. Second: {secondary}. Third: {tertiary}. Together: these strands cover the question without a single clean answer.` | `{primary}; {secondary}; {tertiary}.` **(This is the live failure mode — Phase A showed 4/5 MultiStep prompts returned only "Several parts to this." because `". "` truncation hit the first period.)** |
| 16 | MultiSense | 1824 | `{subj_cap} has multiple senses. In one sense, {subj_lc} relates to {labels0}. In another, {subj_lc} relates to {labels1}. And in another, {subj_lc} relates to {labels2}. Which sense did you mean?` | `{subj_cap}: sense 1 = {labels0}; sense 2 = {labels1}; sense 3 = {labels2}. Which?` |

**Flags for John's attention:**
- **#7 MultiHypothesisAnswer & #15 MultiStep & #16 MultiSense** — the verbose forms exist *because* the shape's PURPOSE is to lay out multiple strands. Terse forms compress aggressively (semicolons replacing prose) but never drop a slot. If terseness fundamentally undercuts a shape's reason to exist, the alternative is to **ignore verbosity for these three shapes** (always render the verbose form). My recommendation: keep terse variants as proposed — they still convey N strands, just without connective tissue. But this is a real design call for you.
- **#13 Synthesis fallback** — uses `→` (arrow). All other terse forms use semicolon or em-dash. Consistent ASCII alternative: `via`.
- **#1 Definition, #2 Process, #4 Comparison, #13 Synthesis** — when walker_prose is `Some`, terse form passes it through unchanged. Walker prose is multi-bond composed prose; truncating it would be a separate (harder) port. For verbosity=terse with walker_prose=Some, we accept the verbose passthrough. Flag if you want a different policy.
- **#14 Wondering** — when walker_prose is `Some`, body is walker prose; terse keeps the body and shortens only the wondering caveat.

**No shape flagged "needs detailed look during implementation."** All 16 are summarizable as verbose-vs-terse table entries.

---

## 4. Fabricated content removal list

The four phrases (all in `apply_style_modulations`, all DELETED outright):

| # | Phrase | Line | Trigger | Why it's fabricated |
|---|--------|------|---------|---------------------|
| F1 | `"That's a good question —"` (prefix) | 2009 | `empathy >= 0.7` | Asserts user asked a good question — no signal grounds this. |
| F2 | `"Technically speaking, this is the canonical formulation."` (suffix) | 2085 | `technicality >= 0.7` | Asserts canonicality — no provenance check, applied to arbitrary prose. |
| F3 | `"In plain terms: this means the relationship holds simply."` (suffix) | 2096 | `technicality < 0.4` AND `s.len() > 60` | Asserts "the relationship holds simply" — fabricated meta-claim about content. |
| F4 | `"There is more to unpack here, but that is the core of it."` (suffix) | 2114 | `verbosity > 0.7` AND `chars < 240` | Asserts more exists to unpack — no grounding that this is true. |

**Truncation bug** at `:2101-2107` (`max_response_length_factor < 0.4` → `s.truncate(first_period + 1)`) is the load-bearing fact-deletion bug. Goes away with the function.

**Formality contraction expansion** at `:2014-2069` deletes too (see Section 2 reasoning). Mechanically truth-preserving but architecturally wrong (post-hoc text substitution fighting the templates).

**No external callers** of any of these phrases. Searched `src/`, `tests/`, `benches/`, `examples/`, `*.py` — all 5 hits for `canonical formulation` / `more to unpack` / `That's a good question` are inside `narration.rs` itself (4 inside `apply_style_modulations`, 5 inside the test module asserting the phrases appear — those tests get deleted).

---

## 5. What survives in apply_style_modulations

**Recommendation: gut the function entirely.** Delete `pub(crate) fn apply_style_modulations` and the comment block above it (`narration.rs:1957-2122`, ~165 LOC). Remove both call sites (`:2224` refusal-prose path, `:2340` main narrate path). Remove the `use crate::style::StyleHints;` at `:1988` if it's the only reference (it isn't — `StyleHints` is now used by `render_core` directly via `compute_style_hints` upstream).

**Length cap is already handled** by `cap_length` (`narration.rs:1943`) which uses `cfg.max_answer_chars` / `cfg.multi_step_max_chars`. That's the right home for a universal safety net. No new universal cap needed.

**No cross-shape decorations** survive the audit. Curiosity / bridge / critic / scar suffixes already live in `apply_suffix_modulations` (`:1882`) which is signal-gated, not style-gated, and stays put.

The `compute_style_hints` upstream call (in `narrate()` body at `:2331`) is preserved — `hints` get threaded down to `ResponseShape::render` → `render_core` instead of being applied as a post-pass.

---

## 6. Test plan + LOC estimate

**Existing tests (9) — all DELETED or REWRITTEN:**

| Test (line in `narration.rs`) | Disposition |
|--------------------------------|-------------|
| `cs2_narrate_terse_for_low_verbosity` (3535) | **Rewrite** as integration test asserting MultiStep terse form contains all three slot values, not the bug behavior. |
| `cs2_narrate_expansive_for_high_verbosity` (3551) | **Delete.** Asserts fabricated "more to unpack" — F4 removed. |
| `cs2_narrate_formal_for_high_formality` (3563) | **Delete.** Formality contraction expansion removed. |
| `cs2_narrate_technical_for_high_technicality` (3578) | **Delete.** Asserts fabricated "canonical formulation" — F2 removed. |
| `cs2_output_adaptation_disabled_returns_neutral_output` (3591) | **Rewrite.** Kill-switch invariant still matters — assert that `output_adaptation_enabled=false` produces standard (not terse) prose for all shapes. |
| `cs2_anti_hallucination_invariant_activated_bonds_unchanged_across_style_settings` (3614) | **Keep, expand.** This is the load-bearing invariant test. Add: assert *content slots* (numbers, names) appear identically in both style outputs. |
| `cs2_neutral_hints_no_op` (3668) | **Rewrite** to test the new `render_core(..., hints)` path with neutral hints. |
| `cs2_high_empathy_prefixes_acknowledgment` (3676) | **Delete.** Asserts fabricated "That's a good question" — F1 removed. |

**New tests (32 — two per shape):**

For each shape S in {Definition, Process, WhyChain, Comparison, FactRecall, UncertainAnswer, MultiHypothesisAnswer, BridgedAnswer, CriticVettedAnswer, Refusal, MetaSelf, TimetravelComparison, Synthesis, Wondering, MultiStep, MultiSense}:

- `test_shape_<s>_standard_form_renders_full_template` — pass `StyleHints { max_response_length_factor: 0.5, .. }`, assert verbose template output.
- `test_shape_<s>_terse_form_preserves_all_slots` — pass `StyleHints { max_response_length_factor: 0.2, .. }`, assert (a) every slot value appears in output, (b) output is shorter than standard form.

**New integration test:**
- `test_narrate_low_verbosity_preserves_multistep_body` — the load-bearing regression for today's bug. Fixture brain with MultiStep-eligible recall, low-verbosity user, assert all three slot values appear in `result.answer`.

**LOC estimate (refined):**

| Change | LOC |
|--------|-----|
| `apply_style_modulations` + docstring deletion | -165 |
| Removal of 2 call sites + 2 `now_ms` blocks | -22 |
| `render_core` signature change + `hints` thread-through (1 site) | +2 |
| `ResponseShape::render` signature change | +2 |
| New per-shape `match hints.verbosity_tier()` arms (16 shapes, ~6-10 LOC each) | +130 |
| `StyleHints::verbosity_tier()` + `VerbosityTier` enum in `src/style/mod.rs` | +12 |
| Existing tests deleted | -100 |
| Existing tests rewritten | +40 |
| New per-shape verbose+terse tests (32 tests) | +180 |
| New integration test | +30 |

**Net: +109 LOC.** Range: +80 to +160 depending on how compact the per-shape match arms come out. Architecture is meaningfully cleaner: zero post-hoc fact mutation; style choice is a render-time concern colocated with each template.

---

## 7. Backward compat surface

Searched and confirmed:

- **`apply_style_modulations` callers**: `src/narration.rs:2224` (refusal path), `src/narration.rs:2340` (main narrate path). Both internal. Both removed in this refactor.
- **`compute_style_hints` upstream**: `src/brain.rs:4444` — unchanged.
- **Fabricated phrase strings** (`"That's a good question"`, `"canonical formulation"`, `"more to unpack"`, `"In plain terms: this means"`): zero occurrences outside `src/narration.rs`. No external scripts, tests, benches, or docs depend on the phrases.
- **Public `NarrationResult.answer` field**: unchanged shape (still `String`). Existing MCP/JSON-RPC consumers see no schema change. Content of `answer` changes at low verbosity (fact-preserving instead of fact-deleting) — that IS the point.
- **`StyleHints` public API**: gains a `verbosity_tier()` method. No breaking changes.

---

## 8. Risks + open questions

**Risks**

1. **Aesthetic regression on default-verbosity prompts.** Default `max_response_length_factor = 0.5` (neutral) → renders Standard form, which is today's verbose template. No regression. The only behavior change is at `< 0.4`, which today produces the BUG and tomorrow produces a sane terse form. Low risk.
2. **Walker prose at terse verbosity.** Shapes #1/#2/#4/#13 pass walker_prose through unchanged when terse. Walker prose can be long. If post-deploy data shows users with `verbosity < 0.4` getting long walker prose, a follow-up port can truncate walker prose intelligently. Out of scope here.
3. **Loss of formality contraction expansion** (the only mechanically-defensible branch in the old function). If a formality-sensitive consumer surfaces, re-introduce as per-shape template choice or as a final-pass mechanical substitution that's *clearly not generative*. Low likelihood — there is no live user signal indicating formality matters.
4. **MultiStep / MultiHypothesisAnswer / MultiSense terse forms may feel "too clipped."** Their reason-to-exist is multi-strand. See Section 3 flag. If you prefer, we ignore verbosity for these three (always render verbose). Trivial code change.

**Open questions for John**

- **Q1**: Two-tier (Standard / Terse) or three-tier (Verbose / Standard / Terse)? My recommendation is two-tier. If you want a future expansive variant for `> 0.7`, it's a per-shape template addition done as a separate port.
- **Q2**: For shapes #7/#15/#16 (multi-strand shapes), accept the compressed terse form (proposed) OR exempt them from terseness (always verbose)? Recommendation: accept the proposed terse forms — they preserve every slot.
- **Q3**: Synthesis terse form fallback connector — `"via"` or `"->"`? Recommendation: `"via"` for ASCII discipline.
- **Q4**: Should the `"Tentatively:"` prefix in `apply_prefix_modulations` also become render-time? It's signal-gated (`flags.hypothesis_only_support`), not style-gated, so it survives this refactor by default. Defer.
- **Q5**: Does Phase B include rewriting CLAUDE.md / GLOSSARY.md narration entries to reflect the new architecture, or is that a separate doc-update port? Recommendation: fold doc updates into Phase C implementation, not Phase B design.

---

**End of design surface. Awaiting John's review on Q1-Q5 + per-shape table sign-off before Phase C implementation.**
