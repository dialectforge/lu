# Geometry perception — hardened (2026-06-28)

Generalises the red-leaf-specific prototype into a measured, gated perceiver.

## Pipeline (perceive.py)
1. **Square detection + rectification.** Largest 4-gon contour -> perspective
   warp to a canonical unit square. All geometry is then in SQUARE-SIDE units,
   so calibration uses the square itself as the ruler (no OCR scale-bar).
2. **Circle/arc detection.** Sequential RANSAC circle-fitting on the shaded
   region's boundary (colour-agnostic: any saturated fill). Robust to partial
   arcs and corner circles that Hough misses.
3. **Snap-then-verify K-gate.** Snap each (cx,cy,r) to the nearest rational with
   denominator <= 4; ACCEPT only if the raw pixel measurement was already within
   0.03 side-units of that rational (derived bound: small-denominator rationals
   are >= 0.083 apart, so < 0.042 is unambiguous). Otherwise FLAG — never force
   a snap / fabricate a primitive we cannot verify.

## Results
- Synthetic set (gen.py, ground truth): **6/6** figures fully recovered
  (quarter-circle, inscribed circle x2, vesica/leaf x2, semicircle) across
  varied scale, offset, rotation, colour, line width, JPEG noise.
- Real image redleaf.jpeg: **4/5** circles, all accepted match exactly
  ((0,12,12),(8,0,8),(4,12,8),(8,4,4)); the 5th is an interior eye arc not on
  the shaded boundary.

## Output feeds the exact engine
Accepted primitives are exact rationals -> directly consumable by the Phase-3
arrangement engine (square -> 4 segments, circles -> arcs).

## Honest gaps / next
- Only circles+square so far; chords/line-segments and arc-span (start/end)
  extraction still to add. Needs more varied REAL images to push robustness.
- Threshold 0.03 is conservative on noisy photos (it flagged a real arc at
  0.025 before loosening); principled but tunable.
