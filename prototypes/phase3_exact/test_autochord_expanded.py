#!/usr/bin/env python3
"""
Expanded auto-chord: a chord that the old (>=3 collinear vertices) rule MISSES.

Square [0,4]^2 with an inscribed circle r=2 @ (2,2) (tangent to all four sides).
Shaded = the LEFT half-disk. The separating chord is the vertical diameter from
(2,0) to (2,4): it passes through only TWO vertices (the tangency points) and
the centre -- exactly the case the >=3-collinear rule could not derive. The
expanded candidate set (>=2-vertex lines + centre lines) finds it.
"""
from fractions import Fraction as F
from algnum import Alg
from geom_exact import Seg, Arc, fmt, area_float, area_exact, R
from auto_chord import candidate_chords, collinear_chords, refine_until_pure, build

def box(x0, y0, x1, y1):
    c = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    return [Seg(c[i], c[(i + 1) % 4]) for i in range(4)]

Arc._next = 0
arcs = [Arc(2, 2, 2, 0, 0, "O")]            # inscribed circle, tangent to all sides
segs = box(0, 0, 4, 4)

nodes0, _, _ = build(segs, arcs)
old = collinear_chords(nodes0)
new = candidate_chords(nodes0, arcs)
print(f"old (>=3-collinear) candidates: {len(old)}")
print(f"new (>=2-vertex + centres) candidates: {len(new)}")

res = refine_until_pure(arcs, segs,
                        shaded=[(1, 2)],
                        unshaded=[(3, 2), (F(2, 5), F(2, 5)), (F(18, 5), F(2, 5)),
                                  (F(18, 5), F(18, 5)), (F(2, 5), F(18, 5))])
assert res is not None, "no refinement found"
chords, area, (nodes, circ) = res
def cs(s): return f"({s.a[0].to_float():.0f},{s.a[1].to_float():.0f})-({s.b[0].to_float():.0f},{s.b[1].to_float():.0f})"
print("chord(s) used:", ", ".join(cs(s) for s in chords))
print("shaded area:", fmt(area, circ), "~", round(area_float(area, circ), 4))
assert area_exact(area, circ) == (R(0), R(2)), fmt(area, circ)     # left half disk = 2pi
assert len(chords) == 1
print("\nEXPANDED AUTO-CHORD PASSED: left half-disk = 2*pi via a 2-vertex diameter chord.")
