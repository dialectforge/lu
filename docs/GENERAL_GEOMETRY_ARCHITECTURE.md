# General Geometry Architecture — from 3 hardcoded classes to a general, honesty-floored geometric-puzzle solver

**Status:** DESIGN / SCOPING ONLY. No code changed, nothing built, nothing promoted, no ingest touched, `master.key` untouched. This document is read-only planning.

**Author context:** Written against box `/opt/lu/lu/merge-main` @ `33cdc33` (Phase-5, live on guest `:8787` pid 96674). Sources: internal notes #1258/#1262/#1263/#1267/#1269/#1353/#1412/#1548–#1552, the Rust code under `src/vision/` and `src/math/`, and the project-9 Python prototypes under `prototypes/`.

**North star (John's goal):** "recognize and solve any geometric puzzle image the first time it's shown." This document treats that as a *direction*, not a deliverable. "Literally any puzzle, always correct, first try" is an asymptote no exact solver reaches. What we *can* build is broadening general coverage, one **verified** figure-family at a time, on top of an **unbreakable honesty floor**: when perception is ambiguous, constraints are under-determined, or the algebra is outside what the solver can exactly do, the system emits **no number** and says so. A truthful refusal on the unsupported remainder is the feature — it is exactly the behavior John is trying to beat in the LLMs that confidently guess.

---

## 1. Honest current-state assessment

### 1.1 What the system does today (and it genuinely does it)

The live Phase-5 pipeline (`lu4d_geometry_solve` image mode, wired in `src/math/geometry_tool.rs::solve_image_bytes`) takes a raw JPEG/PNG of a geometry figure and returns an **exact** `a + b·π` answer *with provenance* — or an honest refusal. It is not a lookup: the anti-cheat has been proven repeatedly (novel leaves the code never saw return different, correctly-scaled answers; the only `include_bytes!(redleaf)` is `#[cfg(test)]`; #1551).

The pipeline is: **pixels → JPEG/PNG decode → perspective rectify → RANSAC arc/circle fit → digit OCR (scale) → build a `Partition` → `solve_and_verify` → exact `a+b·π`.** All pure-Rust, no external crates for decode.

**It recognizes exactly three figure classes.** From `src/vision/geo_perceive_p4.rs::perceive_photo` (the live entry), the dispatch is hardcoded (lines ~1396–1468):

1. **`grid_leaf`** — the 3×3 red-leaf-in-a-square (corner r=1 arc + two edge r=2/3 arcs + a pixel-measured `eye` lens arc). 7-tile partition.
2. **`leaf`** — the two-opposite-corner leaf, arc radius = side. 2 wing tiles.
3. **`quarter_disk`** — a single quarter circle. 1 complement tile.

Everything else returns `PerceiveP4Outcome::CantParse` with a precisely-diagnosed stage. The refusal string is literally (line ~1468):

> `unrecognised arc configuration: {n} corner(r=1) arc(s), {m} edge(r=2/3) arc(s) accepted from {k} RANSAC circle(s); supported classes are the two-opposite-corner leaf, the single quarter, and the 3x3 grid-leaf`

This is the honesty floor working correctly. It was verified live on John's own tilted-square-in-square + inscribed-circle image (`FullSizeRender.jpeg`, 1320×1161): the perceiver *did* genuinely read pixels (correct dims, ran RANSAC, found 5 circles) and then honestly stopped at `arc_topology` with **no number** (#1552).

### 1.2 Exactly why it cannot generalize — three hard structural walls, each cited

**Wall A — Perception is per-class template matching, not general primitive detection.**
`perceive_photo` does not detect *primitives* (arbitrary line segments, full circles, polygons, angle marks) and then reason about them. It fits circles with RANSAC and then asks a fixed question — "how many corner-radius arcs, how many edge-radius arcs?" — and matches that count against three hand-written templates (`geo_perceive_p4.rs` ~1396–1465). There is no general "what primitives are in this figure" stage. A tilted square, an inscribed circle, a triangle with a cevian, two tangent circles — none have a template, so all fall to `CantParse`. Phase-1 (`geo_perceive.rs`, #1548) is the same shape for clean PNGs: it measures a per-corner arc fraction and snaps to leaf/quarter.

**Wall B — There is no general constraint reader.**
The figure's *meaning* (this circle is inscribed; these two circles are tangent; this tick means equal length; this square is rotated 30°; "area = 24") is never extracted. The three templates hardcode their own topology and scale comes only from OCR'ing one printed digit (K-gated `OCR_MIN`). Labels are not associated to the elements they annotate. So even if perception saw the primitives, nothing turns them into the relations that define the puzzle.

**Wall C — The live solver is partition-of-a-whole over `a + b·π` only.**
The solver the perceiver calls is `geometry.rs::Partition::solve_and_verify`. Its number type is:

```rust
pub struct Exact { pub rational: Rational, pub pi: Rational }   // src/math/geometry.rs:131
pub struct Rational { pub num: i64, pub den: i64 }              // i64 num/den
```

That is **`a + b·π` and nothing else** — no `√`, no algebraic numbers, no transcendentals other than π. And the only thing the solver *does* is subtract known tiles from a known whole:

```rust
pub fn solve_target(&self) -> Exact { self.whole.sub(self.sum_known()) }  // geometry.rs:~424
```

`solve_and_verify` (geometry.rs:433) adds the honesty gate — an IDK floor (minimum tile confidence < 0.40 → `Idk`) and the sum-to-whole invariant (residual ≠ 0 in either the rational or the π coefficient → `Flagged`). This is a genuinely good verifier, but it presupposes you already have a valid decomposition into closed-form tiles. It cannot: take a square root, set up or solve an equation, place points from constraints, or handle any figure whose answer contains `√3`, `√2`, `arctan`, a circular-segment with a non-right central angle, etc. #1263 names this the long pole: *"Exact is a+b·π and cannot hold √3."* Three-mutually-tangent-circles (#1546/#1551) and the tilted-square family are blocked here.

### 1.3 The important nuance the census already found — there is orphaned capability

This is critical to an honest roadmap, and it is easy to miss: **a real algebraic-number engine and a real exact face-walk solver already exist in the tree, fully unit-tested, but are wired to nothing.**

- `src/math/algnum.rs` — `Alg`, an exact multi-quadratic algebraic-number type over `num-rational::BigRational`, prime-set basis. It **can** hold `√3`, `√2`, `1+√2`, denest `√(3+2√2)`, and its `sign()` is exact and float-free. API: `sqrt_rational`, `add/sub/mul/div/inverse`, `sign`, `eq`, `isqrt_field` (returns `Option`; `None` = flag a non-representable radical).
- `src/math/arrangement.rs` — exact seg/arc/arc intersections **over `Alg`**, a half-edge arrangement, face walk, and Green's-theorem `Area` returning `(Alg, Alg)` = (rational-with-radicals) + π·(coeff) + angle-atoms. `build_checked` returns `None` on a nested radical it can't handle (a flag, not a guess).
- `src/math/autochord.rs` — `solve_shaded` returns `Solved::Exact` **only** when a closed `a+b·π` form provably exists, else `Solved::Idk(reason)` — it already refuses a non-closed lens (`arccos(3/4)`).

The census (#1353) and the module graph confirm the wiring gap. In `src/math.rs`, `algnum`, `arrangement`, `autochord` are declared (`pub mod …`, lines 41–43) but a full-tree grep finds **zero** references to them anywhere else in `src/`. They are compiled, tested (Phase-3 Rust port, #1267, all green), and **orphaned**. The live perceiver path uses `geometry.rs`'s `Exact`/`Partition` exclusively.

**So "the √ long pole" is not a blank sheet.** The exact-radical *arithmetic* (`Alg`) is built. What is missing is (a) *wiring* an `Alg`-based solver into the perceiver-facing path, and — the deeper gap — (b) a solver that **derives unknowns** (an unknown radius, an unknown tilt, a side length) **from constraints**. `arrangement.rs` computes areas from primitives whose exact coordinates you already know; it does not solve for an unknown radius given "these three circles are mutually tangent." That derivation is the real general-solver gap.

### 1.4 …and there is a general constraint solver too, in prototype

`prototypes/geosolve/solver.py` (project-9, #1269) is a **metric/constraint solver**: model a figure as points + exact constraints (`dist`, `angle`, `perp`, `collinear`, `midpoint`, `on-circle/diameter`, `eqdist`, `between`), place it numerically from several random seeds (Levenberg–Marquardt), read the queried length/angle, and **recover the exact closed form** (small-denominator rational, `√(rational)`, or simple-degree). Its K-gate is exactly the honesty floor generalized: the recovered answer must be **invariant across all random valid placements** (varies → under-determined → **refuse**), and recovery accepts **only** small-denominator closed forms verified to ~1e-7 (`2·sin20°` and other exotic forms are **refused**, not approximated). Companion prototypes: `dsl.py` (constraint DSL), `nl2dsl.py` (NL givens → DSL), `prototypes/diagram/extract_diagram.py` (labeled-figure fact extraction: OCR point names, vertices from *line intersections*, green right-angle-mark → `perp`), and `full_pipeline.py` (diagram + text → answer).

This prototype is Python, not ported to Rust, and wired to nothing. But it is the single most important asset for the general goal: it already has the right *shape* (constraint solve + exact recovery + invariance refusal) and it already handles `√(rational)`, which is precisely what the tilted-square and tangent-circle families need.

**Current-state summary:** the live system is three template perceivers feeding a partition-subtraction solver limited to `a+b·π`, with a solid verify/refuse gate. Off the live path, the tree already contains (unwired) an exact algebraic-number ring, an exact arrangement/face solver, and — in Python prototype — a general constraint-invariance solver with the correct honesty gate. The gap to "general" is real and multi-phase, but it is not starting from zero.

---

## 2. Target general architecture

The general pipeline replaces "match one of three templates" with four honest stages, each of which can independently REFUSE:

```
  image ─▶ [A] GENERAL PERCEPTION ─▶ primitives + labels
                     │ (refuse: can't decode / ambiguous primitives)
                     ▼
          [B] LABEL↔ELEMENT ASSOCIATION ─▶ annotated primitives
                     │ (refuse: a label can't be bound to an element)
                     ▼
          [C] CONSTRAINT EXTRACTION ─▶ a constraint system (the puzzle's meaning)
                     │ (refuse: relations under/over-determined or contradictory)
                     ▼
          [D] GENERAL SOLVER + VERIFY/REFUSE ─▶ exact answer  OR  honest no-number
```

### 2.A General perception
Detect **arbitrary primitives**, not per-class templates: line segments, full circles, arcs (centre/radius/span), polygons, points, right-angle marks, and shading regions. Then general OCR of any numeric/letter labels. This is the successor to the RANSAC-count templates in `geo_perceive_p4.rs`. Reuse what exists: the pure-Rust PNG/JPEG decoders, perspective rectification, and RANSAC circle fitting are already in `geo_perceive.rs`/`geo_perceive_p4.rs` and are class-agnostic — they just need to feed a *general primitive list* instead of a template counter. Add: a line-segment detector (Hough or LSD-style), polygon assembly from segments (the diagram prototype already derives vertices from *line intersections*, which is the right idea), and a right-angle-mark detector (the diagram prototype detects the green square mark → `perp`).

### 2.B Label ↔ element association
General OCR gives glyphs and positions; the puzzle needs each label bound to *the element it annotates*: a number next to a segment is a **side length**; inside a region is an **area = N**; along a spoke from a centre is a **radius**; in a wedge is an **angle**; a letter at a junction **names a point**. Today scale is a single OCR'd digit assumed to be the cell size — there is no association model. This stage produces annotated primitives. It refuses when a label cannot be unambiguously bound (e.g. a number equidistant from two segments).

### 2.C Constraint extraction
Infer the **relations that define the puzzle** from primitives + bound labels: tangency (circle touches line/circle at one point), inscribed/circumscribed, concentric, equal-length ticks, right angles, parallel marks, `area = N` region labels, collinearity/betweenness. Output is a constraint system in a DSL. The prototype `geosolve/dsl.py` + `diagram/extract_diagram.py` already demonstrate this for the angle/length class (incidence + perp + betweenness from a synthetic diagram). Generalizing it to circles, tangency, and shading is the work. This stage refuses on contradictory or under-determined relation sets.

### 2.D General solver + generalized verify/refuse
A **coordinate-geometry + algebraic constraint solver** that sets up and solves the system. It needs `√`/algebraic numbers, quadratics, systems of equations, and area integrals for circular segments — a real generalization beyond partition-of-a-whole over `a+b·π`. **The build strategy is to compose two assets that already exist:**

- **Constraint solving → the `geosolve` engine** (numeric LM placement from multiple seeds → exact recovery). This is how you *derive* an unknown radius/tilt/length from tangency/inscribed/right-angle constraints. It already recovers `√(rational)`.
- **Exact representation & area → `algnum.rs::Alg` + `arrangement.rs`.** Once the constraint solver has fixed the exact primitives (as `Alg`), the arrangement engine computes exact shaded areas including `π` terms, and `autochord::solve_shaded` already refuses non-closed forms.

**Generalized honesty layer.** The verify/refuse gate that today is "sum-to-whole residual == 0" (`Partition::solve_and_verify`) becomes three composed gates, and the system emits a number only if **all** pass:
1. **Perception gate** (A/B): every primitive/label accepted only if the raw pixels already agree (the existing snap-then-verify K-gate discipline).
2. **Determinacy gate** (C/D): the constraint invariance check — the answer must be identical across independent valid placements; if it varies, the figure is under-determined → refuse (`geosolve`'s existing K-gate).
3. **Exactness gate** (D): recover a number only if a provable small-denominator / algebraic closed form exists (`Alg::isqrt_field` returns `None`, or `autochord` returns `Idk`, → refuse). Never approximate an exotic value with a large fraction.

This is what keeps "general" from becoming "guessing."

---

## 3. Honest feasibility — what a general geometry solver can and cannot do

**Tractable with exact constraint-solving (realistic to cover, family by family):**
- Figures reducible to distance/angle/perp/collinear/tangent/inscribed constraints whose answers are rational, `√(rational)`, or `a + b·π`: tilted-square-in-square, three-mutually-tangent-circles (Descartes/√3), Pythagorean-area figures, inscribed/circumscribed circle radii, shaded lens/leaf/petal partitions, cevian length/angle chases, right-angle length problems. These are where the `geosolve` + `Alg`/`arrangement` composition genuinely earns exact answers with an invariance proof.

**Not tractable / research-hard (must fall to honest refusal, and that is correct):**
- Puzzles requiring genuine **insight or an auxiliary construction** the solver would have to *invent* (add this specific line, reflect this point) — general auxiliary-construction search is open-ended and not something a constraint solver discovers reliably.
- Answers that are **not closed-form** in the solver's number field: `arccos(3/4)`, `2·sin20°`, nested radicals `Alg` can't denest, most transcendental mixes. `autochord`/`algnum` already refuse these by design.
- **Ambiguous or degenerate perception**: hand-drawn, low-contrast, occluded, or figures where the intended relation isn't visually determined (is that tangency or just close?). Refuse.
- **Combinatorial / competition-style** problems needing number theory or clever inequalities dressed as geometry.

The honest framing: coverage grows as a **union of verified families**. Each added family is a real, tested capability with an invariance/exactness proof. The complement — everything not yet a verified family, plus the permanently research-hard set — always returns an honest refusal. There is no phase at which the refusal set becomes empty, and pretending otherwise would be the exact failure mode (confident wrong answers) the project exists to beat.

---

## 4. Phased roadmap (each phase gated, verified, honesty-preserving)

Ordering note: I largely agree with the suggested P-A→P-F ordering, with one refinement grounded in what's already in the tree. The suggested P-A ("√ upgrade to `geometry.rs`") is partly already done (`algnum.rs` exists); the true foundation is a **solver that derives unknowns from constraints using exact radicals**, i.e. wiring `algnum` + porting `geosolve`. So P-A is reframed as "the exact algebraic constraint-solver layer." It remains first because it unblocks whole families at once and is the dependency for everything downstream.

**P-A — Exact algebraic constraint-solver layer (the foundation).**
*Enables:* the tilted-square-in-square and three-mutually-tangent-circles families and every Pythagorean/`√(rational)` figure — by *computation*, offline, before any new perception. *Difficulty:* medium-high (new Rust module; leans on existing `Alg`). *Dependencies:* `algnum.rs` (built). *Gate:* offline unit tests only; invariance K-gate + exactness gate; anti-cheat (no filename/lookup). *Honesty floor:* refuse on under-determined (invariance fails) or non-closed (no `Alg` form). **This is the chosen first buildable increment — spec in §5.**

**P-B — General primitive detector.** Replace the RANSAC-count templates with a general primitive list (segments + circles + polygons + arcs + points + right-angle marks + shading). *Enables:* perceiving figures that have no template. *Difficulty:* high (line/polygon detection is the new hard perception work). *Dependencies:* reuses existing decode/rectify/RANSAC. *Gate:* per-primitive snap-then-verify K-gate; refuse if pixels don't confirm. *Honesty:* an unconfirmed primitive is dropped; a figure with unexplained ink refuses.

**P-C — Label↔element association + general OCR.** Generalize OCR beyond a single scale digit and bind each label to its element (side/area/radius/angle/point-name). *Difficulty:* medium-high. *Dependencies:* P-B (needs primitives to bind to). *Gate:* refuse on ambiguous binding. *Honesty:* no guessed bindings.

**P-D — Constraint extraction.** Turn annotated primitives into a constraint system (tangency, inscribed, concentric, equal ticks, right angles, `area=N`, parallel). *Difficulty:* high (visual→relational inference). *Dependencies:* P-B, P-C; reuses `geosolve/dsl.py` + diagram-extractor ideas. *Gate:* refuse on contradictory/under-determined relations. *Honesty:* the DSL is the auditable intermediate — a wrong relation shows up as an invariance failure downstream.

**P-E — General coordinate/equation solver + generalized verify/refuse.** Fold the P-A solver + `arrangement` area engine behind the constraint system so extracted constraints drive a full solve; compose the three honesty gates (§2.D). *Difficulty:* medium (integration; both solvers exist by now). *Dependencies:* P-A, P-D. *Gate:* triple gate; number only if all pass.

**P-F — Live wiring + broad gated eval.** Wire the general path into `lu4d_geometry_solve` alongside (not replacing) the three template classes; run a broad **held-out corpus** of puzzle images, measuring solved-correct / honest-refuse / **wrong-number (must be ~0)**. *Difficulty:* medium. *Dependencies:* all prior. *Gate:* human-gated promote (relay-ff, John's yes), same discipline as Phase-1/4/5; the wrong-number rate is the release blocker, not the solved rate.

Across all phases the promote discipline is unchanged: build+test clean, full lib suite green with zero regression, serving surface diff-proven, live binary untouched until an explicit human-gated cutover, `master.key` and the wikidata ingest never touched.

---

## 5. The first buildable increment (P-A), spec'd for a build task

**Name:** `algsolve` — an exact algebraic constraint-solver module (new `src/math/algsolve.rs`), offline, wired to nothing, unit-tested only.

**Goal:** given a figure expressed as points + exact constraints, derive the queried quantity as an exact value (rational, `√(rational)` via `Alg`, or `a+b·π`), with an invariance proof — or refuse. This is the Rust realization of `prototypes/geosolve/solver.py`'s method, using `algnum.rs::Alg` for exact recovery/representation, and it is the dependency the whole general path stands on.

**Scope (in):**
1. A constraint model: `dist`, `angle`, `perp`, `collinear`, `midpoint`, `on_circle`, `tangent_circle_circle`, `tangent_circle_line`, `eqdist`, `between`, plus a `find(query)`.
2. Numeric placement: multi-seed Levenberg–Marquardt (central-difference Jacobian), gauge DOFs left free on purpose.
3. **Invariance K-gate:** solve from N≥5 independent random seeds; the queried value must agree across all valid placements to a tight tolerance or → `Refuse(under_determined)`.
4. **Exact recovery via `Alg`:** recover only small-denominator rational, `√(rational)` (through `Alg::sqrt_rational`/`isqrt_field`), or simple-degree; anything else → `Refuse(no_closed_form)`. Reuse `Alg::sign`/`eq` for the float-free exact check.
5. Two acceptance figures solved end-to-end from constraints, **no hardcoded answers**:
   - **Tilted-square-in-square + corner triangle / inscribed circle** (John's `FullSizeRender.jpeg` family): set up from the tilt + tangency/area constraints, derive the queried area/length as an exact `√`-bearing value.
   - **Three mutually tangent circles** (#1546): derive the unknown radius / shaded area (Descartes → `√3`), the case `Exact`/`a+b·π` provably cannot represent — demonstrating the upgrade.
6. Output type unifies with the existing verifier: reuse/extend `VerifyOutcome`-style `{Solved, Refuse{reason}}` so downstream wiring is uniform.

**Scope (out):** no perception, no OCR, no live wiring, no changes to `geometry.rs`/`geo_perceive*.rs`/`geometry_tool.rs`/`guest_mcp.rs` (serving surface stays byte-identical), no promote. Constraints are supplied by the test harness, exactly as `geosolve` does today.

**Anti-cheat / honesty test plan (the load-bearing part):**
- **Solve genuinely from constraints:** each acceptance figure is defined only by its constraint list + query; grep-prove the non-test solve path contains no filename, no image hash, no `IMG_`/`FullSizeRender`/answer constant. The answer must be *computed* from the placement + `Alg` recovery.
- **Novel-instance test:** re-parameterize each family (different tilt angle, different circle radii) the code has never seen; the exact answer must scale correctly (not a constant).
- **Under-determined → refuse:** feed a figure missing one constraint (e.g. drop the between/tangency that pins it); the invariance gate must fire → `Refuse(under_determined)`, **no number**.
- **No-closed-form → refuse:** feed a figure whose answer is `2·sin20°` or `arccos(3/4)`; recovery must refuse rather than emit a large-denominator fraction — mirror the existing `autochord` `arccos(3/4)` refusal test.
- **Never a lookup / never a guess:** the two positive tests and the two refusal tests run in the same suite; a green run means *both* that it solves the hard families *and* that it refuses the under-determined/exotic ones.
- **Gate:** full lu4d lib suite green with zero regression; `algsolve` reachable only from its own tests; serving path diff-proven untouched; live binary not reinstalled; ingest + `master.key` untouched.

**Why this first:** it is the foundation every downstream phase depends on, it is mostly a *port + wire* of assets already in the tree (`algnum.rs` built; `geosolve` prototype validated), and it immediately expands what is *solvable* — including John's tilted-square+circle image and the three-tangent-circle figure — the moment perception can hand it constraints. It moves the `√` long pole (#1263) from "blocked" to "built and tested," without touching a single live byte.

---

## 6. North-star statement

The goal is **broad, general coverage of geometric puzzles behind an unbreakable honesty floor.** "Recognize and solve any geometric puzzle image the first time" is the direction we optimize toward, not a promise of omniscience. We get there by adding figure families as **verified capability** — each one perceived from real pixels, its meaning extracted into an auditable constraint system, solved exactly, and proven by an invariance/exactness gate — never by widening a threshold until the system accepts everything.

On every figure outside the verified set — whether it is a family we simply haven't built yet, or one that is genuinely research-hard — the system emits **no number** and says honestly that it cannot parse or cannot solve it. That truthful refusal is not a failure. It is the entire point: it is the exact opposite of the confident-guessing LLM behavior John is trying to beat. A general solver that is *sometimes right and sometimes confidently wrong* is worse than useless for verifiable math; a solver that is *right on a growing set of families and honest everywhere else* is the thing worth building. Coverage expands; the honesty floor never moves.

---

*Read-only design artifact. No code, serving surface, brain, `master.key`, or ingest (pid 933219) was modified in its creation.*
