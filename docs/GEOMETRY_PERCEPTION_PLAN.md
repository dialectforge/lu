# Image → Geometry → Verified Answer: honest phased plan

Author: Lu build agent · 2026-08-06 · project 9 (the project log)
Repo: /opt/lu/lu/merge-main · main @ 2809908

## The honest starting point (what John correctly called out)

Today the geometry pipeline is REAL on the *solver* side and FAKE on the
*perception* side:

* `src/math/geometry.rs::red_leaf_scene()` is HAND-CODED. Its own comments say
  so: arc centres/radii are "FORCED by the grid ... never measured" and the eye
  corners are literally `add_point("eye_corner_lo", 4.0, 4.0)` with the comment
  "not pixel reads: (4,4),(8,8)".
* `solve_red_leaf()` takes NO image and NO parameters — it rebuilds that fixed
  12×12 scene and returns a constant `32*pi - 64`. The guest tool test is named
  `red_leaf_scene_solves_to_32pi_minus_64`.
* `geometry_tool.rs::solve_scene_json()` REFUSES a raw image on purpose
  ("no perceptual vision model is configured ... I will not guess a numeric
  answer from one"). That refusal is honest, but it means the claim "she solves
  from the image" is currently FALSE.

So the number is correct but it does NOT flow from pixels. The pixels are never
read. This plan makes pixels→answer genuinely true, one figure-class at a time,
and is explicit about the one place where the *math* (not the perception) is the
blocker.

What is already real and reusable (do not rebuild):
* `Exact` = `a + b·π` with exact `Rational` arithmetic; area primitives
  (`square`, `quarter_circle`, `square_minus_quarter_circle`, `triangle`, …).
* `Partition` + `VerifyOutcome` (Solved / Flagged / Idk): the sum-to-whole
  verifier and the IDK/K-gate that returns NO number when a load-bearing tile is
  below the confidence floor. This is the honesty engine and every phase routes
  its answer through it.
* The Python perception prototype (`prototypes/perception/perceive.py`, 6/6
  synthetic + 4/5 on the real `redleaf.jpeg`): square-detect → perspective
  rectify to a unit square (the square is the ruler, no scale-bar OCR) →
  sequential-RANSAC circle fit on the coloured region boundary → snap each
  (cx,cy,r) to a small-denominator rational and ACCEPT only if the raw pixel
  measurement was already within tolerance (snap-then-VERIFY, never force a
  snap). This is the approach we are porting.

Constraint discovered during study: the Rust tree deliberately avoids the
`image` crate (it hand-rolls header probes in `extractor.rs`). It DOES have
`flate2`/`miniz_oxide` in the lockfile, so PNG is decodable in-house; there is
NO JPEG decoder. `num-bigint`/`num-rational` are already dependencies (relevant
to P3 below).

---

## Phase 1 — real pixel perception for the red-leaf-on-grid class  ← THIS TASK

Goal: PNG bytes of a red-leaf-in-a-square → detect the square + the two
opposite-corner arcs from actual pixels → build the Scene → call the REAL solver
→ exact `a + b·π`, with the honesty floor intact.

Scope / method:
* In-house PNG decoder (flate2 inflate + PNG unfilter) → RGB pixels. No new
  heavy crate; PNG-only is an explicit, honest limit (JPEG is P4).
* Detect the square as the extent of the dark ("ink") pixels; require it to be
  roughly square or refuse.
* Detect the shaded region as the saturated-colour pixels; take its boundary,
  drop the points lying on the straight square edges.
* Arc detection by measuring, per square-corner, the fraction of boundary points
  at distance ≈ side, and the median radius; snap radius→rational in side-units
  and K-gate (accept only if the raw measurement already agreed). Two arcs at
  OPPOSITE corners each radius = 1·side ⇒ the red-leaf (vesica) class.
* Side length `S` (the integer "4"/"8" label) is supplied as a parameter for the
  numeric answer; the *structure* (square + which corners carry arcs + radius
  ratio) is what is derived from pixels. OCR of the printed digit is deferred to
  P4. Without `S` the perceiver still reports the structure and the symbolic
  form `(π/2 − 1)·S²` — but NO number (honesty floor).
* Build `Partition{ whole = square(S), wings = 2×(square−quarter) }`, target =
  leaf; `solve_and_verify` ⇒ `leaf = (π/2)·S² − S²`. All exact, verified,
  residual zero.

Anti-cheat (the whole point):
* The scene is derived ONLY from bytes + an integer `S`. There is NO filename,
  path, image-hash, or answer lookup anywhere on the path (grep-proven).
* Proven with NOVEL inputs the code has never seen: a freshly-drawn red-leaf at a
  side the code has never seen ⇒ correctly-scaled answer (e.g. S=5 ⇒
  (25/2)π − 25, S=7 ⇒ (49/2)π − 49), NOT the old constant 32π−64. And a
  STRUCTURALLY different figure at the same S=8 (a single quarter-circle) ⇒ a
  different scene/answer — proving pixels, not `S`, drive the structure.
* Honesty floor: an unparseable figure ⇒ "I can't extract this figure", NO
  number; a non-geometry image ⇒ the honest capability statement.

Effort: ~1 focused build (this task). Risk: real photos may be rotated/skewed —
the axis-aligned detector may not rectify a phone photo; that is an honest P4
limit, reported not hidden.

## Phase 2 — the tangent / rectangle class (IMG_5639)

The solver side already exists (`solve_control(tangent_cm)` → area = t²). Phase 2
is the perception: detect a rectangle + a corner quarter-circle + a semicircle,
read the tangent-length label, and emit the control scene from pixels. Needs
line-segment/rectangle detection and at least one real numeric-label read (OCR),
which is why it follows P1's structure-only perception. Effort: moderate.

## Phase 3 — the three-circles class (IMG_5643)  ← BLOCKED ON THE MATH, NOT PERCEPTION

FLAG: this class is blocked *before perception matters*. The answer for the
three-mutually-tangent-circles figure involves √3 (the curvilinear triangle
between three unit circles = (√3 − π/2)·r²-type forms). The current `Exact` type
is `a + b·π` with rational a,b — it CANNOT represent √3. So even with perfect
perception, the solver would either refuse (honest) or be forced to a decimal
(a floor violation).

Dependency: the algebraic-number upgrade (#1263, the long pole) must land in
`geometry.rs` FIRST — extend the exact number type from `a + b·π` to carry
algebraic irrationals (√-extensions), likely on top of the already-present
`num-bigint`/`num-rational` and the existing `src/math/algnum.rs` module. Only
then does perceiving the three circles produce a representable exact answer.
Effort: the algebraic-number work is the largest single item; perception for the
class is comparatively small once the type exists.

## Phase 4 — general OCR + detector robustness

Numeric-label OCR (read "4"/"8"/"5 cm" from pixels so scale is fully pixel-
derived, closing the P1 `S`-as-parameter gap), JPEG decode, perspective
rectification for skewed photos, and RANSAC generalisation (arbitrary circle/arc
positions, chords, line segments) — porting the remaining robustness of the
Python prototype. Effort: the largest perception item; incremental and testable.

## Phase 5 — wire the whole image→answer path to the guest surface + gating

Expose image→answer on `lu4d_geometry_solve` (additive optional `image_base64` +
`side`), route perception success → exact answer and any parse failure → honest
refusal, and gate it through the full acceptance + fabrication-bait + live-100Q
harness before promoting. (Phase 1 keeps this additive and behind an optional
arg so the existing scene-string calls stay byte-identical.)

---

## Honest status line

"Arbitrary geometry photo in → correct verified answer out" is NOT done and this
plan does not pretend a single phase gets there. P1 makes it genuinely true for
one class (red-leaf), from real PNG pixels, proven against novel inputs. P3 is
gated on a math upgrade that has nothing to do with vision. P4 is where "photo"
(JPEG, skew, OCR) becomes real. Each phase closes on evidence, not assertion.
