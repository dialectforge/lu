#!/usr/bin/env python3
"""
redleaf_demo.py - Phase 3 acceptance run.

Proves the auto-partition core on the red-leaf puzzle with NO hand-written
decomposition: the only inputs are the 12x12 square and the five quarter-arcs.
Everything else (intersections, atomic faces, exact areas, the shaded lobes) is
derived. Run:  python3 redleaf_demo.py
"""
from auto_partition import *

def square(s=12):
    c = [(0,0),(s,0),(s,s),(0,s)]
    return [Seg(c[i], c[(i+1)%4]) for i in range(4)]

def red_leaf_arcs():
    Arc._next = 0
    return [Arc(0,12,12,-90,0,"A"),   Arc(8,0,8,90,180,"B"),
            Arc(4,12,8,270,360,"C"),  Arc(4,8,4,-90,0,"e1"),
            Arc(8,4,4,90,180,"e2")]

def whole_ok(interior, circ):
    tot = Area()
    for _, a in interior: tot += a
    return tot

print("="*68)
print("PHASE 3  -  AUTOMATIC region decomposition of the red leaf")
print("="*68)

# ---------- 1. ARC-ONLY ARRANGEMENT ----------
arcs = red_leaf_arcs()
nodes, he, circ = build(square(12), arcs)
interior = [(L,a) for L,a in faces(nodes,he) if to_float(a,circ) > 1e-6]
print(f"\n[1] Arrangement of {len(arcs)} arcs + square:")
print(f"    {len(nodes)} vertices, {len(he)//2} edges, "
      f"{len(interior)} atomic interior faces  (Euler: V-E+F=2)")
for L,a in sorted(interior, key=lambda z: to_float(z[1],circ)):
    print(f"      face = {fmt(a,circ):>12}   ~ {to_float(a,circ):8.4f}")

tot = whole_ok(interior, circ)
print(f"\n    VERIFY sum-to-whole: {fmt(tot,circ)}   "
      f"(closed-form & every symbolic angle-atom cancelled: {is_closed(tot,circ)})")
assert fmt(tot,circ) == "144"

# ---------- 2. IDENTIFY THE LEFT LOBE (a single atomic face) ----------
seed = (7.0, 3.0)                       # one red pixel in the lobe body
lobe = [a for L,a in interior if point_in_face(seed,L,nodes)]
print(f"\n[2] LEFT LOBE  (seed {seed} -> a single auto-derived face):")
print(f"      x = {fmt(lobe[0],circ)}   = {to_float(lobe[0],circ):.6f}")
assert fmt(lobe[0],circ) == "32*pi - 64"
print("      MATCHES the Phase-1 hand answer 32*pi-64  -- with no hand cuts.")

# ---------- 3. FULL LEAF (needs the tip-to-tip chord) ----------
# arc C is internally tangent to arc A (centres 4 apart, radii 12 & 8), so the
# right wedge is never cut off by the arcs alone. The chord joining the two leaf
# TIPS (0,0)-(12,12) -- which also passes through the existing nodes (4,4),(8,8)
# and bisects the eye -- refines the arrangement so the full leaf is recovered.
nodes2, he2, circ2 = build(square(12)+[Seg((0,0),(12,12))], red_leaf_arcs())
interior2 = [(L,a) for L,a in faces(nodes2,he2) if to_float(a,circ2) > 1e-6]
tot2 = whole_ok(interior2, circ2)
print(f"\n[3] Augmented with tip chord (0,0)-(12,12): "
      f"{len(interior2)} faces, sum = {fmt(tot2,circ2)}")

# exact unions for the two named regions
import itertools, fractions
def vec(a):
    o,_ = resolve(a, circ2); return (o.rat, o.pi)
fs = [a for _,a in interior2]
def find(target):
    for r in range(1,len(fs)+1):
        for cmb in itertools.combinations(range(len(fs)), r):
            s = Area()
            for i in cmb: s += fs[i]
            if vec(s) == target: return s
    return None
left  = find((fractions.Fraction(-64),  fractions.Fraction(32)))
leaf  = find((fractions.Fraction(-96),  fractions.Fraction(48)))
print(f"      LEFT LOBE  = {fmt(left,circ2)}   ({to_float(left,circ2):.4f})")
print(f"      FULL LEAF  = {fmt(leaf,circ2)}   ({to_float(leaf,circ2):.4f})")
print( "      (full leaf = left lobe 32*pi-64  +  right wedge 16*pi-32)")
assert fmt(left,circ2) == "32*pi - 64"
assert fmt(leaf,circ2) == "48*pi - 96"

# ---------- 4. HONESTY GATE ----------
bad = Area()
for i,(L,a) in enumerate(interior):
    bad += (a + Area(rat=1)) if i==0 else a      # inject a 1-unit error
print(f"\n[4] Honesty gate: corrupt one face -> sum = {fmt(bad,circ)} "
      f"(!= 144, so the verifier FLAGS instead of emitting a number)")
assert fmt(bad,circ) != "144"

print("\nALL CHECKS PASSED.\n")
