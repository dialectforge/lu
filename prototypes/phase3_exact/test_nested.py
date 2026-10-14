#!/usr/bin/env python3
"""Nested / disconnected islands: the enclosing face must subtract the island."""
from fractions import Fraction as F
from algnum import Alg
from geom_exact import (Seg, Arc, build, faces, faces_nested, fmt, area_float,
                        area_exact, point_in_face, R)

def box(x0, y0, x1, y1):
    c = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    return [Seg(c[i], c[(i + 1) % 4]) for i in range(4)]

def find_area(q, fs, nodes, circ):
    # with nested faces a point lies in several loops; the true face is the
    # SMALLEST-area one containing it.
    hits = [a for L, a in fs if point_in_face(q, L, nodes, circ)]
    return min(hits, key=lambda a: area_float(a, circ)) if hits else None

# ---- Case A: square 12x12 with a floating circle r=2 (not touching) ----
print("CASE A: square[0,12]^2 + floating circle r=2 @ (6,6)")
Arc._next = 0
arcs = [Arc(6, 6, 2, 0, 0, "O")]
nodes, he, circ = build(box(0, 0, 12, 12), arcs)
fs = faces_nested(nodes, he, circ)
for L, a in sorted(fs, key=lambda z: area_float(z[1], circ)):
    print(f"   {fmt(a, circ):>14} ~ {area_float(a, circ):8.4f}")
annulus = find_area((1, 1), fs, nodes, circ)
disk = find_area((6, 6), fs, nodes, circ)
assert area_exact(annulus, circ) == (R(144), R(-4)), fmt(annulus, circ)   # 144 - 4pi
assert area_exact(disk, circ) == (R(0), R(4)), fmt(disk, circ)            # 4pi
print("  OK: annulus = 144 - 4*pi, disk = 4*pi")

# ---- Case B: 3-level nesting  square > circle r=3 > small square[5,7] ----
print("\nCASE B: square[0,12]^2 > circle r=3 @ (6,6) > small square[5,7]^2")
Arc._next = 0
arcs = [Arc(6, 6, 3, 0, 0, "O")]
segs = box(0, 0, 12, 12) + box(5, 5, 7, 7)
nodes, he, circ = build(segs, arcs)
fs = faces_nested(nodes, he, circ)
for L, a in sorted(fs, key=lambda z: area_float(z[1], circ)):
    print(f"   {fmt(a, circ):>14} ~ {area_float(a, circ):8.4f}")
outer = find_area((1, 1), fs, nodes, circ)            # square minus disk
ring = find_area((6, F(17, 2)), fs, nodes, circ)      # disk minus small square
inner = find_area((6, 6), fs, nodes, circ)            # small square
assert area_exact(outer, circ) == (R(144), R(-9)), fmt(outer, circ)       # 144 - 9pi
assert area_exact(ring, circ) == (R(-4), R(9)), fmt(ring, circ)           # 9pi - 4
assert area_exact(inner, circ) == (R(4), R(0)), fmt(inner, circ)          # 4
print("  OK: outer = 144 - 9*pi, ring = 9*pi - 4, inner = 4")

# ---- Case C: connected figure (red leaf) must be UNCHANGED by nesting ----
print("\nCASE C: red leaf (connected) unchanged under faces_nested")
def red_leaf():
    Arc._next = 0
    return [Arc(0, 12, 12, -90, 0, "A"), Arc(8, 0, 8, 90, 180, "B"),
            Arc(4, 12, 8, 270, 360, "C"), Arc(4, 8, 4, -90, 0, "e1"),
            Arc(8, 4, 4, 90, 180, "e2")]
nodes, he, circ = build(box(0, 0, 12, 12), red_leaf())
plain = sorted(fmt(a, circ) for L, a in faces(nodes, he) if area_float(a, circ) > 1e-6)
nest = sorted(fmt(a, circ) for L, a in faces_nested(nodes, he, circ) if area_float(a, circ) > 1e-6)
assert plain == nest, f"{plain} != {nest}"
print("  OK: identical ->", nest)

print("\nNESTED-ISLAND TEST PASSED.")
