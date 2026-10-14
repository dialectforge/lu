#!/usr/bin/env python3
"""
Non-Pythagorean proof: figures whose intersection coordinates are IRRATIONAL,
where the old float-then-snap pipeline would corrupt the rational part. The
exact engine derives the exact closed form, and the angle resolver (Niven's
theorem regime) recognises the pi-multiple subtended angles.

Case 1: circle r=2 at origin cut by the line x=1.  Crossings at (1, +-sqrt3).
        Right circular segment  =  4/3*pi - sqrt(3).
Case 2: two unit circles, centres distance 1 apart.  Crossings at
        (1/2, +-sqrt3/2).  Lens (intersection) = 2/3*pi - sqrt(3)/2.
"""
import math
from fractions import Fraction as F
from algnum import Alg
from geom_exact import Seg, Arc, build, faces, fmt, area_float, area_exact, point_in_face, R

S = Alg.sqrt_rational

def full_circle(cx, cy, r, name):
    return Arc(cx, cy, r, 0, 0, name)        # P0 == P1 -> full circle span

def run(title, segs, arcs, probes):
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)
    nodes, he, circ = build(segs, arcs)
    fs = faces(nodes, he)
    print(f"{len(nodes)} vertices, {len(he)//2} edges, {len(fs)} faces")
    for label, seed, exp_rat, exp_pic in probes:
        hit = [a for L, a in fs if point_in_face(seed, L, nodes, circ)]
        cand = [a for a in hit if area_float(a, circ) > 1e-9]
        cand.sort(key=lambda a: area_float(a, circ))
        a = cand[0] if cand else None
        ex = area_exact(a, circ) if a is not None else None
        got = fmt(a, circ) if a is not None else "<none>"
        val = area_float(a, circ) if a is not None else float("nan")
        ok = ex is not None and ex[0] == exp_rat and ex[1] == exp_pic
        print(f"  {label:15} seed{seed}:  {got:22} ~ {val:9.5f}   {'OK' if ok else 'MISMATCH'}")
        assert ok, f"{label}: got rat={ex[0] if ex else None}, pic={ex[1] if ex else None}"

# ---- Case 1 ----
sq1 = [(-3, -3), (3, -3), (3, 3), (-3, 3)]
segs1 = [Seg(sq1[i], sq1[(i + 1) % 4]) for i in range(4)]
segs1.append(Seg((1, -3), (1, 3)))               # the cutting line x = 1
Arc._next = 0
arcs1 = [full_circle(0, 0, 2, "O2")]
run("CASE 1  circle r=2 cut by x=1  (crossings at (1, +-sqrt3))",
    segs1, arcs1,
    [("right segment", (F(3, 2), 0), -S(3), R(F(4, 3)))])    # 4/3*pi - sqrt3

# ---- Case 2 ----  (two circles alone: a connected arrangement, no island)
Arc._next = 0
arcs2 = [full_circle(0, 0, 1, "C1"), full_circle(1, 0, 1, "C2")]
run("CASE 2  two unit circles, centres 1 apart  (crossings at (1/2,+-sqrt3/2))",
    [], arcs2,
    [("lens", (F(1, 2), 0), -S(3) / R(2), R(F(2, 3)))])      # 2/3*pi - sqrt3/2

print("\nNON-PYTHAGOREAN PROOF PASSED: exact closed forms from irrational crossings.")
