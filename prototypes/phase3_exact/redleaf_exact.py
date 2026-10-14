#!/usr/bin/env python3
"""Red-leaf regression on the EXACT engine: must reproduce 32*pi-64 and sum 144,
cross-checked against an independent numerical Green's integral."""
import math
from algnum import Alg
from geom_exact import (Seg, Arc, build, faces, fmt, area_float, is_closed,
                        resolve, point_in_face, R)

def square(s=12):
    c = [(0, 0), (s, 0), (s, s), (0, s)]
    return [Seg(c[i], c[(i + 1) % 4]) for i in range(4)]

def red_leaf_arcs():
    Arc._next = 0
    return [Arc(0, 12, 12, -90, 0, "A"), Arc(8, 0, 8, 90, 180, "B"),
            Arc(4, 12, 8, 270, 360, "C"), Arc(4, 8, 4, -90, 0, "e1"),
            Arc(8, 4, 4, 90, 180, "e2")]

def numeric_green(loop, nodes):
    """Independent: dense-sample each edge, sum 1/2*(x dy - y dx). No symbolic atoms."""
    tot = 0.0
    for h in loop:
        if not h.is_arc:
            ux, uy = h.tail_pt[0].to_float(), h.tail_pt[1].to_float()
            vx, vy = h.head_pt[0].to_float(), h.head_pt[1].to_float()
            tot += 0.5 * (ux * vy - vx * uy)
        else:
            cx, cy = h.O[0].to_float(), h.O[1].to_float()
            r = math.sqrt(h.Rr.to_float())
            P, Q = h.tail_pt, h.head_pt
            aP = math.atan2(P[1].to_float() - cy, P[0].to_float() - cx)
            aQ = math.atan2(Q[1].to_float() - cy, Q[0].to_float() - cx)
            d = aQ - aP
            if h.ccw and d < 0: d += 2 * math.pi
            if (not h.ccw) and d > 0: d -= 2 * math.pi
            N = 4000; prev = None
            for k in range(N + 1):
                th = aP + d * k / N
                x = cx + r * math.cos(th); y = cy + r * math.sin(th)
                if prev is not None:
                    px, py = prev
                    tot += 0.5 * (px * y - x * py)
                prev = (x, y)
    return tot

print("=" * 64)
print("PHASE 3 EXACT  -  red-leaf regression")
print("=" * 64)
arcs = red_leaf_arcs()
nodes, he, circ = build(square(12), arcs)
allf = faces(nodes, he)
interior = [(L, a) for L, a in allf if area_float(a, circ) > 1e-6]
print(f"\n{len(nodes)} vertices, {len(he)//2} edges, {len(allf)} total faces, "
      f"{len(interior)} interior")

tot = None
print("\ninterior faces (exact vs independent numeric contour):")
for L, a in sorted(interior, key=lambda z: area_float(z[1], circ)):
    s = area_float(a, circ); n = numeric_green(L, nodes)
    print(f"   {fmt(a, circ):>14}   exact~{s:9.5f}   numeric={n:9.5f}   diff={abs(s-n):.2e}")

acc = None
for L, a in interior:
    acc = a if acc is None else (acc + a)
print(f"\nsum-to-whole: {fmt(acc, circ)}   closed-form={is_closed(acc, circ)}")
assert fmt(acc, circ) == "144", "FACES DO NOT SUM TO 144"

seed = (7, 3)
lobe = [a for L, a in interior if point_in_face(seed, L, nodes, circ)]
print(f"\nleft lobe (seed {seed}): {len(lobe)} face(s)")
for a in lobe:
    print(f"   = {fmt(a, circ)}   ~ {area_float(a, circ):.6f}")
assert len(lobe) == 1 and fmt(lobe[0], circ) == "32*pi - 64", "LEFT LOBE WRONG"
print("\nLEFT LOBE = 32*pi - 64 (exact, auto-derived).  REGRESSION PASSED.")
