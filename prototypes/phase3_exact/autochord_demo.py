#!/usr/bin/env python3
"""Gap 4 demo: auto-derive the refining chord by shading purity (red leaf).

No chord, no target area, and no answer are hand-supplied: the only inputs are
the primitives and seed points marking shaded vs unshaded. The chord and the
exact shaded area are derived."""
from fractions import Fraction as F
from algnum import Alg
from geom_exact import Seg, Arc, fmt, area_float, R
from auto_chord import candidate_chords, refine_until_pure, build

def square(s=12):
    c = [(0, 0), (s, 0), (s, s), (0, s)]
    return [Seg(c[i], c[(i + 1) % 4]) for i in range(4)]

def red_leaf():
    Arc._next = 0
    return [Arc(0, 12, 12, -90, 0, "A"), Arc(8, 0, 8, 90, 180, "B"),
            Arc(4, 12, 8, 270, 360, "C"), Arc(4, 8, 4, -90, 0, "e1"),
            Arc(8, 4, 4, 90, 180, "e2")]

def chord_str(segs):
    return ", ".join(f"({s.a[0].to_float():.0f},{s.a[1].to_float():.0f})-"
                     f"({s.b[0].to_float():.0f},{s.b[1].to_float():.0f})" for s in segs) or "none"

# show the auto-derived candidate chords from the figure's own vertices/centres
nodes0, he0, circ0 = build(square(12), red_leaf())
cands = candidate_chords(nodes0, red_leaf())
print(f"{len(cands)} candidate chords from latent lines (vertices + centres)")

print("\n--- LEFT LOBE (one shaded seed; should need NO chord) ---")
res = refine_until_pure(red_leaf(), square(12),
                        shaded=[(7, 3)],
                        unshaded=[(10, F(1, 2)), (4, F(15, 2)), (6, 6)])
chords, area, (nodes, circ) = res
print("  chords used:", chord_str(chords))
print("  shaded area:", fmt(area, circ), "~", round(area_float(area, circ), 4))
assert chord_str(chords) == "none" and fmt(area, circ) == "32*pi - 64"

print("\n--- FULL LEAF (shaded across the tangency; chord must be derived) ---")
res = refine_until_pure(red_leaf(), square(12),
                        shaded=[(F(5, 2), 5), (5, 4), (F(11, 2), F(13, 2)), (F(19, 2), 8)],
                        unshaded=[(10, F(1, 2)), (4, F(15, 2)), (F(13, 2), F(11, 2))])
chords, area, (nodes, circ) = res
print("  chords used:", chord_str(chords))
print("  shaded area:", fmt(area, circ), "~", round(area_float(area, circ), 4))
assert fmt(area, circ) == "48*pi - 96", f"got {fmt(area,circ)}"
assert len(chords) == 1, f"expected one chord, got {len(chords)}"

print("\nAUTO-CHORD PASSED: full leaf = 48*pi - 96 with one derived chord (not given).")
