# OP-ROUTE-2 Design — QueryRouter Infrastructure

**Status:** Phase A design. HALT for John's review before implementation.
**Branch:** to be `feature/op-route-2-query-router` (off `main` HEAD `957322e`).
**Scope:** Phase 2 of the Lu routing+depth plan. Build the decision layer that Phase 3 (deep-port wiring) and Phase 4 (parallelization) will consume.
**Anti-LLM positioning:** load-bearing. Hand-coded heuristics only. No learning, no model calls, no embeddings.

---

## 1. `QueryRouter` module structure

New file: `src/query_router.rs`. New `pub mod query_router;` entry in `src/lib.rs` (immediately after `pub mod narration;` so the dependency direction is clear: narration may call query_router, never the reverse).

The module is a pure-function namespace. No struct state, no `&mut self`, no global cache. Thread-safety is trivial because every input is borrowed and every output is owned.

```rust
// src/query_router.rs
//! OP-ROUTE-2 — QueryRouter.
//!
//! Hand-coded routing layer that decides which capability ports a
//! given query should activate. Sits between IntentClassifier and the
//! recall pipeline in narrate(). Anti-LLM: every dimension is a
//! deterministic string/regex/length heuristic. No learning.

use crate::narration::QueryIntent;
use serde::{Deserialize, Serialize};

pub struct QueryRouter;

impl QueryRouter {
    /// Single entry point. Pure function over query + intent.
    /// Does NOT touch the brain; routing must be decidable from the
    /// query string and the already-computed intent alone. Brain
    /// context (recall state, hub-degree, etc.) flows into Phase 3
    /// stages, not into routing.
    pub fn route(query: &str, intent: QueryIntent) -> RoutingDecision { ... }
}
```

Notes:
- No `&Brain` parameter. The brief mentions "brain context" but on review, every routing dimension below is decidable from the query + intent alone. Pulling brain in would couple the router to recall ordering and make Phase 4 parallelization harder. If a future dimension genuinely needs brain context, we add a second entry point `QueryRouter::route_with_brain(query, intent, brain)` rather than overload the pure one. **Open question 1.**
- No struct state means we can call `QueryRouter::route` from anywhere (tests, MCP handlers other than narrate, future REPL) without setup ceremony.

---

## 2. `RoutingDecision` struct

```rust
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RoutingDecision {
    /// Query is arithmetic / construction / math-expression.
    /// Reuses crate::math detectors — does NOT reinvent them.
    pub needs_math: bool,

    /// Query references a filesystem path (e.g. "summarize ./notes.md",
    /// "open ~/Documents/x.pdf"). Phase 5 will wire file_io here.
    pub references_file_path: bool,

    /// Query references an image path or asks for visual interpretation.
    /// Phase 5 will wire vision here.
    pub references_image_path: bool,

    /// Long / multi-clause / causal-marker query that warrants think().
    /// Phase 3a will wire crate::thinking here.
    pub needs_deep_reasoning: bool,

    /// Counterfactual / alternative-framing query.
    /// Phase 3b will wire crate::variants here.
    pub needs_variants: bool,

    /// Controversial claim / "is it true that" query.
    /// Phase 3c will wire multi-agent critic vetting here.
    pub needs_critic_vetting: bool,

    /// Code-generation intent (write a function, generate code).
    /// Phase 6 will wire code generation here.
    pub code_intent: bool,

    /// Stable list of heuristic names that fired (for debugging
    /// "why this routing?" the same way signals_used works for
    /// narration shape selection).
    pub triggers: Vec<String>,
}
```

Defaults: all bools `false`, `triggers` empty. Phase 2 invariant: a `RoutingDecision::default()` produces narrate() output byte-identical to today.

Serde: `Serialize + Deserialize` so it round-trips through MCP responses and test fixtures. `Default` because `NarrationResult` carries it via `#[serde(default)]`.

---

## 3. Heuristic rules per dimension

All rules operate on `let q = query.trim().to_lowercase();` plus the original `query` for case-sensitive checks (file paths, code blocks). Each rule pushes its name into `triggers` when it fires.

### 3.1 `needs_math`

**Reuse, do not reinvent.** Math detection is already three primitives in `crate::math`.

```rust
let needs_math = crate::math::detect_arithmetic(query).is_some()
    || crate::math::construction::detect_construction(query).is_some()
    || crate::math::detect_math(query).is_some();
if needs_math { triggers.push("math".into()); }
```

This is the same triple-check `try_arithmetic_response_with` performs internally (`src/mcp.rs:1480`). The router does NOT call the primitives' compute paths — it only asks "does this look like math?" Phase 3 will short-circuit on this flag (or leave the existing `try_arithmetic_response_with` short-circuit alone — see Open question 4).

### 3.2 `references_file_path`

File-path heuristics. A query references a file path when:

1. Explicit URI scheme: starts with `file://`, contains ` file://`.
2. Contains a path-like token: a whitespace-bounded token that
   - starts with `/`, `./`, `../`, `~/`, or a Windows drive letter `[A-Za-z]:\\` or `[A-Za-z]:/`, AND
   - ends with a known file extension (case-insensitive): `.md .txt .csv .json .pdf .docx .doc .rs .py .js .ts .html .yaml .yml .toml .log .xml`.
3. OR contains a token ending with one of those extensions even without a path prefix, as long as the query also contains a file-action verb: `read`, `open`, `summarize`, `parse`, `load`, `analyze`, `cat`, `the file`.

```rust
fn looks_like_file_path(query: &str) -> Option<&'static str> {
    let lower = query.to_lowercase();
    if lower.contains("file://") { return Some("uri_scheme"); }
    const EXTS: &[&str] = &[
        ".md", ".txt", ".csv", ".json", ".pdf", ".docx", ".doc",
        ".rs", ".py", ".js", ".ts", ".html", ".yaml", ".yml",
        ".toml", ".log", ".xml",
    ];
    let has_action = [
        "read ", "open ", "summarize ", "parse ", "load ", "analyze ",
        "cat ", "the file", "this file",
    ].iter().any(|v| lower.contains(v));
    for tok in query.split_whitespace() {
        let t = tok.trim_end_matches(|c: char| matches!(c, "."|","|";"|")"));
        let lt = t.to_lowercase();
        let has_path_prefix = t.starts_with("/")
            || t.starts_with("./")
            || t.starts_with("../")
            || t.starts_with("~/")
            || (t.len() >= 3 && t.as_bytes()[1] == b":"
                && (t.as_bytes()[2] == b"\\"[0] || t.as_bytes()[2] == b"/"[0]));
        let has_ext = EXTS.iter().any(|e| lt.ends_with(e));
        if has_ext && (has_path_prefix || has_action) {
            return Some(if has_path_prefix { "path_with_ext" } else { "ext_with_action" });
        }
    }
    None
}
```

### 3.3 `references_image_path`

Image-specific subset. Fires when either:

1. A token ends with `.png .jpg .jpeg .gif .webp .bmp .tiff .heic` (case-insensitive), OR
2. Query contains a vision-intent phrase: `"describe the image"`, `"describe this image"`, `"look at this image"`, `"what is in the picture"`, `"what do you see"`, `"in this photo"`.

Kept disjoint from `references_file_path`. If both image extension + general action verb fire, image wins (more specific).

### 3.4 `needs_deep_reasoning`

Fires when ANY of:

- Word count >= 25 (raw `split_whitespace().count()`). Long queries usually have multi-clause structure.
- Contains a causal-with-because pattern: lowercased query matches regex `r"why\b.*\bbecause\b"` or `r"\bbecause\b.*\?"`.
- Contains a multi-axis comparison marker: `"considering"`, `"taking into account"`, `"weigh"`, `"evaluate"`, `"trade-off"`, `"trade off"`, `"tradeoff"`.
- Contains an explicit-explain marker: `"explain in detail"`, `"walk me through"`, `"step by step"`, `"step-by-step"`, `"in depth"`, `"reason about"`, `"reason through"`.
- Sentence count >= 3 (count `?` + `.` + `!` separators; a multi-sentence query is usually composed of context + question).

Does NOT fire just because intent is `WhyChain` — a short "why is the sky blue?" should NOT need deep reasoning. The threshold is structural complexity, not intent class.

### 3.5 `needs_variants`

Counterfactual / alternative-framing markers. Fires when query contains ANY of:

- `"what if "`, `"what would happen if"`, `"suppose "`, `"imagine if"`, `"imagine that"`.
- `"alternatively"`, `"from a different angle"`, `"from another angle"`, `"a different perspective"`, `"another way to look at"`.
- `"counterfactual"`, `"hypothetically"`.
- `"would it be different if"`, `"could it have been"`.

### 3.6 `needs_critic_vetting`

Claim-to-verify or controversial-framing markers. Fires when query contains ANY of:

- `"is it true that"`, `"is it true"`, `"true or false"`.
- `"are you sure"`, `"can you verify"`, `"can you confirm"`, `"double-check"`, `"double check"`, `"fact-check"`, `"fact check"`.
- `"i heard that"`, `"someone told me"`, `"is the claim"`, `"the claim that"`.
- `"prove it"`, `"how do you know"`, `"evidence for"`, `"evidence that"`.

Does NOT fire on `"why"` alone — too broad.

### 3.7 `code_intent`

Code-generation intent. Fires when ANY of:

- Query contains a fenced code block marker: ```` ``` ````, or an inline-code marker followed by code-like content (rare; backtick alone is too noisy, skip).
- Query contains BOTH a code-generation verb AND a code-context noun:
  - Verbs: `"write"`, `"generate"`, `"produce"`, `"create"`, `"give me"`, `"show me"`, `"implement"`.
  - Nouns: `"a function"`, `"a class"`, `"a script"`, `"a snippet"`, `"code"`, `"the code"`, `"some code"`, `"a regex"`, `"an sql"`, `"a query"`, `"a query for"`.
- Query contains a language-name + verb pair: language in `{"python", "rust", "javascript", "typescript", "go", "java", "c++", "c#", "bash", "sql", "html", "css"}` AND verb in `{"write", "generate", "in", "using"}` separated by ≤ 4 tokens.
  - e.g. `"write a function in python"` → fires (language=python, verb=write).
  - e.g. `"python is a language"` → does not fire (no generation verb).

### 3.8 Additional dimensions surfaced during design

The brief invited "more if you find clear patterns." Three candidates considered, only one accepted:

- **`asks_for_list`** ("give me 5 examples of..."). Rejected for Phase 2 — no port consumes it yet; better to add when Phase 3+ surfaces the demand.
- **`time_sensitive`** ("current", "latest", "today"). Rejected for the same reason. Lu's idk_gate already refuses time-sensitive recall outside its scope.
- **ACCEPTED: `is_meta_query`** mirroring `QueryIntent::MetaSelf` but at the routing layer, because Phase 3 might want to suppress deep-port invocation on meta queries (you don't `think()` about your own brain — that's a different kind of introspection). Defaults false; populated by checking `intent == QueryIntent::MetaSelf || intent == QueryIntent::TimetravelComparison`.

Final field list: 7 specified + 1 derived = 8 boolean dimensions + `triggers` debug list.

---

## 4. Integration point in `narrate()`

Current shape (`src/narration.rs:2103-2270`):

```rust
pub fn narrate(brain, query, depth, top_k, use_multi_think, cfg) -> NarrationResult {
    let intent = IntentClassifier::classify(query);
    let ask = brain.ask(query, depth, top_k);  // <-- recall happens BEFORE routing today
    if cfg.idk_gate_enabled { ... refusal-early-return ... }
    let multi = if use_multi_think { brain.multi_think(query).ok() } else { None };
    let flags = ModulationFlags::compute(&ask, multi.as_ref(), cfg);
    let shape = ShapeSelector::pick(intent, &ask, multi.as_ref(), &flags, cfg);
    let slots = extract_slots(brain, query, &ask, multi.as_ref(), cfg);
    ...
}
```

Phase 2 insertion: route immediately after `IntentClassifier::classify`, before `brain.ask`:

```rust
pub fn narrate(brain, query, depth, top_k, use_multi_think, cfg) -> NarrationResult {
    let intent = IntentClassifier::classify(query);
    let routing = crate::query_router::QueryRouter::route(query, intent);
    let ask = brain.ask(query, depth, top_k);
    if cfg.idk_gate_enabled { ... refusal-early-return — pass routing through ... }
    let multi = if use_multi_think { brain.multi_think(query).ok() } else { None };
    let flags = ModulationFlags::compute(&ask, multi.as_ref(), cfg);  // Phase 2: signature unchanged
    let shape = ShapeSelector::pick(intent, &ask, multi.as_ref(), &flags, cfg);  // Phase 2: signature unchanged
    let slots = extract_slots(brain, query, &ask, multi.as_ref(), cfg);  // Phase 2: signature unchanged
    ...
    NarrationResult { ..., routing }
}
```

### 4.1 Threading through downstream stages

The brief asks to pass `RoutingDecision` to `ModulationFlags::compute`, `ShapeSelector::pick`, and `extract_slots` even though they don't use it yet. **This is the awkward signature concern from the "no hacks" constraint.**

Two options:

**Option A — extend signatures now.** Add `_routing: &RoutingDecision` to each. Pro: Phase 3 doesn't need to re-touch signatures. Con: every test that constructs these manually has to supply a `RoutingDecision::default()` for no behavioral reason.

**Option B — defer threading to Phase 3.** Phase 2 only stores `routing` in the local scope of `narrate()` and on `NarrationResult`. Phase 3 widens signatures when there's an actual consumer.

**Recommendation: Option B.** The "no hacks" rule says don't add ceremony that exists only to satisfy a future port. Phase 3 is a separate PR; it can widen the signatures and update tests in one coherent change. Carrying empty params for a port that may never need them is exactly the kind of mid-refactor cruft we shed when adding the routing layer. **Open question 2.**

---

## 5. `NarrationResult` extension

Add one field:

```rust
pub struct NarrationResult {
    // ... existing fields ...

    /// OP-ROUTE-2 — routing decision computed in narrate() before
    /// recall. Phase 2: populated but not yet acted on (math
    /// short-circuit still lives in try_arithmetic_response_with on
    /// the MCP boundary). Phase 3+ will wire think/variants/reason
    /// invocations based on these flags.
    #[serde(default)]
    pub routing: crate::query_router::RoutingDecision,
}
```

`#[serde(default)]` makes deserialization backward-compatible — old JSON missing the `routing` field deserializes to `RoutingDecision::default()`. The serde docs and existing fields (`source_kinds`, `source_labels`, `source_summary`) confirm this pattern is already in use in this struct.

Refusal-early-return also populates `routing` (not `default()`) — even refusal callers may want to see what the router would have decided. The router is cheap (pure string scan) so we run it unconditionally before the gate.

---

## 6. Test plan

File: `src/query_router.rs` `#[cfg(test)] mod tests`. Each dimension gets at least 2 positives + 2 negatives. Plus a narration-parity integration test.

### 6.1 Unit tests — per dimension

| Dimension | Positive cases | Negative cases |
|---|---|---|
| `needs_math` | `"what is 2+2"`, `"how many board feet in a 2x4x8"` | `"what is a tree"`, `"how are you"` |
| `references_file_path` | `"summarize ./notes.md"`, `"open ~/docs/foo.pdf"` | `"summarize the situation"`, `"what is markdown"` |
| `references_image_path` | `"describe the image at /tmp/cat.png"`, `"what is in the picture above"` | `"describe a cat"`, `"png is a format"` |
| `needs_deep_reasoning` | 30-word multi-clause query, `"why is X because Y reasonable, considering Z?"` | `"why is the sky blue?"`, `"what is rust"` |
| `needs_variants` | `"what if we used Rust instead"`, `"from a different angle, is this idea sound?"` | `"what is rust"`, `"is this idea sound?"` |
| `needs_critic_vetting` | `"is it true that the moon is hollow?"`, `"how do you know that?"` | `"is the moon visible tonight?"`, `"why is the sky blue?"` |
| `code_intent` | `"write a function in python to sort a list"`, `"generate some rust code for fibonacci"` | `"python is a language"`, `"what is fibonacci"` |

All tests also assert `triggers` contains the right names — guards against silent reordering.

### 6.2 Parity test — `narrate` behavior unchanged in Phase 2

```rust
#[test]
fn narrate_phase2_does_not_change_answer_when_routing_only_observed() {
    let mut brain = test_brain_with_some_bonds();
    let cfg = NarrationConfig::default();
    // Math query — routing fires, but the result equals what we'd get
    // before the router was added (because narrate doesn't read routing yet).
    let r1 = crate::narration::narrate(&mut brain, "what is 2+2", 1, 4, false, &cfg);
    assert!(r1.routing.needs_math);
    // Re-run on the SAME inputs — outputs are deterministic and
    // routing presence does not perturb shape/slots/answer.
    let r2 = crate::narration::narrate(&mut brain, "what is 2+2", 1, 4, false, &cfg);
    assert_eq!(r1.answer, r2.answer);
    assert_eq!(r1.shape, r2.shape);
}
```

The load-bearing invariant: a routing flag firing in Phase 2 must NOT change the answer. Phase 3 is what wires the flag into a port invocation.

### 6.3 Serde compatibility test

```rust
#[test]
fn narration_result_deserializes_without_routing_field() {
    let old_json = r#"{"query":"x","answer":"y","shape":"FactRecall","intent":"FactRecall",
        "flags":{...},"signals_used":[],"noise_filtered_count":0,"ask":{...},"multi":null}"#;
    let r: NarrationResult = serde_json::from_str(old_json).unwrap();
    assert_eq!(r.routing, RoutingDecision::default());
}
```

(`RoutingDecision` will need `PartialEq` derive — small addition.)

---

## 7. What's deliberately NOT in this port

- **No port invocations.** Phase 2 does not call `brain.think`, `brain.generate_variants`, `brain.reason`, file_io tools, or vision tools based on routing. Those wirings are Phase 3a/b/c (deep reasoning), Phase 5 (file_io / vision), Phase 6 (code).
- **No parallelization.** Phase 4. The router stays synchronous and runs on the calling thread.
- **No UX-visible behavior change.** Same answers byte-for-byte (modulo the new `routing` field in serialized output, which old clients ignore via `#[serde(default)]` on their end).
- **Math short-circuit is NOT moved.** OP-ROUTE-1 already wired `try_arithmetic_response_with` into `lu4d_narrate` at the MCP boundary. Moving the short-circuit *inside* `narrate()` would be a behavioral change (different prose, different structured-content shape) and is out of scope. The router merely *observes* that the query is math-shaped — the existing short-circuit handles execution. **Open question 4.**
- **No signature widening for `ModulationFlags::compute` / `ShapeSelector::pick` / `extract_slots`.** Phase 3 does that when there's a real consumer (see §4.1 Option B).
- **No code in `narrate()` reading `routing` flags to alter pipeline shape.** Phase 2 is observation-only.

---

## 8. Open questions for John

1. **Should `QueryRouter::route` take `&Brain`?** I argue no (§1). Every dimension is decidable from query + intent. Pulling brain in couples the router to recall ordering and blocks Phase 4 parallelization. If a future dimension genuinely needs brain context, add a second entry point rather than overload.

2. **Defer signature widening to Phase 3? (§4.1)** I recommend Option B: don't pass `routing` into `ModulationFlags::compute`/`ShapeSelector::pick`/`extract_slots` until Phase 3 has a real consumer. The brief explicitly invites the alternative; the "no hacks" rule and the test-update cost push me toward deferral.

3. **Image-extension list scope.** I included `png/jpg/jpeg/gif/webp/bmp/tiff/heic`. Should it also catch `.svg` (vector — vision may or may not handle) and `.raw / .arw / .nef` (camera RAW)? Default: not for Phase 2; add when vision wiring exposes the requirement.

4. **Math short-circuit relocation.** The brief says "possibly moving the math short-circuit to be driven by the router instead of an explicit early check, but that's optional and surface a design choice." My recommendation: leave it alone. OP-ROUTE-1 just wired `try_arithmetic_response_with` into `lu4d_narrate` at the MCP layer. Moving it into `narrate()` itself would change the surface: today, math queries produce a structuredContent payload with `source: "arithmetic_primitive"` instead of a `NarrationResult` shape. Moving the short-circuit inside `narrate()` would either (a) change the response surface (breaking MCP consumers), or (b) require `narrate()` to return arithmetic-primitive prose dressed up as a `NarrationResult` with a fake shape (a hack). Defer.

5. **`code_intent` heuristic completeness.** I included a language-name list of 12. Should it cover Lu's dialects ("DialectForge", "DFoS", "Stratosphere") or stay general-programming? Default: general-programming only. Lu-internal vocab fires on Definition intent, not code-gen intent.

6. **Verb-noun proximity for `code_intent`.** My rule fires on "verb AND noun present anywhere." Should it require proximity (within N tokens) to avoid false positives like "I wrote a function yesterday — what is a class?" → fires both verbs. Trade-off: stricter rule reduces false positives but misses split phrasings. Default: no proximity rule for Phase 2; tighten if Phase 3 shows false-fire rate matters.

---

## 9. LOC estimate

- `src/query_router.rs` — ~400-500 LOC: 8 dimension functions (~30 LOC each = 240), `QueryRouter::route` orchestration (~30 LOC), tests (~200 LOC).
- `src/lib.rs` — +1 LOC.
- `src/narration.rs` — ~15 LOC: route call, refusal early-return field, final `NarrationResult` field, struct field declaration.

**Total: ~450-550 LOC.** Inside the 500-1000 budget. If implementation creeps past 800, surface as scope concern.

---

## 10. Halt protocol

Phase A ends here. Awaiting John's confirmation before:
- Branch creation.
- Code implementation.
- Test writing.
- Build / test runs.
- Push.

No daemon restart. No deploy. Production stays on OP-ROUTE-1 binary `b05e5056…` (currently stopped per John's directive).
