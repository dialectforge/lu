#!/usr/bin/env python3
"""
Tangency / cusp robustness (gap 2).  The hard case the original report flagged:
two arcs that meet with the SAME tangent direction, so only the exact
signed-curvature tie-break can order them in the face walk.

Figure: two externally tangent unit circles, centres (0,0) and (2,0), touching
at (1,0). A horizontal line y=0 through both centres makes the arrangement
connected and splits each disk into halves. Each half-disk must come out as
pi/2 EXACTLY, the disks must not bleed into each other at the tangency, and the
whole must verify.
"""
from fractions import Fraction as F
from algnum import Alg
from geom_exact import Seg, Arc, build, faces, fmt, area_float, area_exact, point_in_face, R

Arc._next = 0
arcs = [Arc(0, 0, 1, 0, 0, "C1"), Arc(2, 0, 1, 0, 0, "C2")]
segs = [Seg((-1, 0), (3, 0))]            # diameter line through both, via tangency (1,0)

nodes, he, circ = build(segs, arcs)
fs = faces(nodes, he)
print(f"{len(nodes)} vertices, {len(he)//2} edges, {len(fs)} faces")
for L, a in sorted(fs, key=lambda z: area_float(z[1], circ)):
    print(f"   {fmt(a, circ):>14}  ~ {area_float(a, circ):8.4f}  ({len(L)} edges)")

half = R(F(1, 2))
probes = [
    ("C1 upper", (F(0), F(1, 2)), F(1, 2)),
    ("C1 lower", (F(0), F(-1, 2)), F(1, 2)),
    ("C2 upper", (F(2), F(1, 2)), F(1, 2)),
    ("C2 lower", (F(2), F(-1, 2)), F(1, 2)),
]
print()
for label, seed, exp_pic in probes:
    hit = [a for L, a in fs if point_in_face(seed, L, nodes, circ)
           and area_float(a, circ) > 1e-9]
    hit.sort(key=lambda a: area_float(a, circ))
    a = hit[0] if hit else None
    ex = area_exact(a, circ) if a is not None else None
    ok = ex is not None and ex[0].is_zero() and ex[1] == R(exp_pic)
    print(f"  {label:9} seed{seed}: {fmt(a, circ) if a else '<none>':>10} "
          f"~ {area_float(a, circ) if a else float('nan'):7.4f}  {'OK' if ok else 'MISMATCH'}")
    assert ok, f"{label}: got {ex}"

# disks must not merge: the two tangent circles stay distinct half-disks
print("\nTANGENCY TEST PASSED: half-disks = pi/2 each, no bleed across the cusp.")
