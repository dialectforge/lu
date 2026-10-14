# Phase 3 "complete it" — DONE (2026-06-28)

All four robustness gaps closed in Python before the Rust port. Everything runs
on the Lu server at /opt/lu/lu/v4/prototypes/phase3_exact/.
Canonical brain/service untouched; commits local only (not pushed).

## Files
- `algnum.py`        exact multi-quadratic field Q(sqrt d1..dk); float-free
                     recursive exact sign. DEPENDENCY-FREE, Rust-portable.
- `test_algnum.py`   3516 checks PASS.
- `geom_exact.py`    exact arrangement -> faces -> Green's-theorem areas; Niven
                     angle resolver; exact face walk w/ curvature tie-break +
                     branch-cut winding; exact generic-ray point-in-face.
- `auto_chord.py`    collinear-vertex candidate chords + shading-purity refine.
- `redleaf_exact.py` regression: left lobe 32*pi-64, faces sum 144. PASS.
- `nonpythag.py`     irrational crossings: 4/3*pi-sqrt3 and 2/3*pi-sqrt3/2. PASS.
- `test_tangency.py` tangent circles -> pi/2 half-disks, no cusp bleed. PASS.
- `autochord_demo.py` full leaf 48*pi-96, chord (0,0)-(12,12) DERIVED. PASS.

## Gaps — all closed
1. EXACT ALGEBRAIC NUMBERS. Multi-quadratic field replaces Fraction+snap.
   Float-free exact sign (recurse on radicals, compare squares). Proven on a
   figure with irrational crossings (1,+-sqrt3) -> 4/3*pi - sqrt(3).
2. TANGENCY/CUSP FACE WALK. Exact angular order + signed-curvature tie-break;
   plus a branch-cut WINDING fix (the sector angle lost a turn when an arc swept
   across the +x axis — this was the two-circle-lens bug). Validated by the
   tangency test and the pure-circle lens (2/3*pi - sqrt3/2).
3. ROBUST POINT-IN-FACE. Exact ray cast along a generic rational direction
   chosen to dodge vertices and tangencies.
4. AUTO-CHORD. Candidate chords = lines through >=3 existing vertices (the
   figure's latent axes); selected by SHADING PURITY (add fewest chords so no
   face holds both a shaded and an unshaded seed). Derives (0,0)-(12,12) for the
   full leaf with no hand-supplied chord or target.

## Niven upgrade
Angle atoms resolve to a rational multiple of pi exactly when cos^2 = c^2/R is
in {0,1/4,1/2,3/4,1} (multiples of pi/6, pi/4). Lets non-cardinal irrational
endpoints yield clean a+b*pi; otherwise flagged, never approximated.

## The three former limits — now SOLVED
- NESTED/DISCONNECTED ISLANDS: `faces_nested()` builds connected components +
  nesting, subtracts each island's exact footprint (its outer CW-cycle Area)
  from the immediate enclosing face. Tested: floating circle -> 144-4pi & 4pi;
  3-level square>circle>square -> 144-9pi, 9pi-4, 4; connected figures unchanged.
  (Also fixed: a lone full circle with no crossings now gets an antipode node so
  it forms a real 2-edge loop.)  -> test_nested.py
- IN-FIELD SQRT DENESTING: `isqrt_field()` denests sqrt(A+sqrt(p)B)=C+sqrt(p)D
  recursively, so irrational-but-in-field discriminants are exact (sqrt(3+2sqrt2)
  =1+sqrt2, sqrt(5+2sqrt6)=sqrt2+sqrt3, and a real arc/seg intersection with
  disc 88+48sqrt2=(4+6sqrt2)^2). Genuine new nested radicals still flagged, never
  approximated. Wired into sqrt_exact.  -> test_denest.py
- AUTO-CHORD CANDIDATES EXPANDED: `candidate_chords()` now uses every line
  through >=2 vertices plus all centre-to-centre lines (tangency axes), clipped
  to the bbox. Solves cases the >=3-collinear rule missed (inscribed-circle
  half-disk = 2*pi).  -> test_autochord_expanded.py

## Files added since first pass
- test_nested.py, test_autochord_expanded.py, test_denest.py
- geom_exact.py gained faces_nested + _components + lone-circle antipode + sqrt
  denesting hook; algnum.py gained isqrt_field; auto_chord.py gained
  candidate_chords.

## Remaining honest boundary
- Denesting covers anything expressible in the multi-quadratic field; a TRUE new
  irrationality (e.g. sqrt(1+sqrt2)) is still flagged (correct behaviour).
- With nested faces, "which face is point q in" = the SMALLEST-area face whose
  loop contains q (loops of enclosing faces overlap inner islands).

## Next: Rust port behind the IDK gate (geometry.rs Exact/Rational + AngleAtom +
  arrangement/half-edge + components/denesting). The Python engine is the spec.
