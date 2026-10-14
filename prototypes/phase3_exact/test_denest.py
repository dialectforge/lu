#!/usr/bin/env python3
"""In-field sqrt denesting: sqrt of an irrational field element when the result
stays in the multi-quadratic field. Genuine new nested radicals are flagged."""
from fractions import Fraction as F
from algnum import Alg, isqrt_field
from geom_exact import sqrt_exact, NeedsNestedRadical, Arc, Seg, inter_arc_seg, on_circle, R

S = Alg.sqrt_rational
ok = 0
def chk(c, m):
    global ok
    assert c, "FAIL: " + m
    ok += 1

# --- denesting identities ---
chk(isqrt_field(R(3) + R(2) * S(2)) == R(1) + S(2), "sqrt(3+2sqrt2)=1+sqrt2")
chk(isqrt_field(R(7) + R(4) * S(3)) == R(2) + S(3), "sqrt(7+4sqrt3)=2+sqrt3")
chk(isqrt_field(R(11) + R(6) * S(2)) == R(3) + S(2), "sqrt(11+6sqrt2)=3+sqrt2")
chk(isqrt_field(R(5) + R(2) * S(6)) == S(2) + S(3), "sqrt(5+2sqrt6)=sqrt2+sqrt3")
chk(isqrt_field(R(16)) == R(4), "sqrt(16)=4")
chk(isqrt_field(R(8)) == R(2) * S(2), "sqrt(8)=2sqrt2")
# real but NOT in the field -> None (genuine 4th-degree nested radical)
chk(isqrt_field(R(1) + S(2)) is None, "sqrt(1+sqrt2) not in field -> None")
chk(isqrt_field(R(-4)) is None, "sqrt(-4) -> None (negative)")
# round-trip on a few squares
for x in [R(2) + S(3), R(1) + S(2) + S(3) + S(6), R(5) - S(2), S(7) + R(3)]:
    sq = x * x
    r = isqrt_field(sq)
    chk(r is not None and r * r == sq, f"roundtrip {x!r}")

# --- sqrt_exact integration (used by the geometry engine) ---
chk(sqrt_exact(R(3) + R(2) * S(2)) == R(1) + S(2), "sqrt_exact denests")
chk(sqrt_exact(R(25)) == R(5), "sqrt_exact rational")
try:
    sqrt_exact(R(1) + S(2))
    chk(False, "should have flagged")
except NeedsNestedRadical:
    chk(True, "flags genuine nested radical")

# --- geometry: an intersection whose discriminant is 88+48sqrt2 = (4+6sqrt2)^2 ---
# vertical chord x=sqrt2 meets circle r=2 at (sqrt2, +-sqrt2); disc denests.
Arc._next = 0
arc = Arc(0, 0, 2, 0, 0, "O")
seg = Seg((S(2), S(2)), (S(2), R(-3)))
pts = inter_arc_seg(arc, seg)
chk(len(pts) == 2, "two intersection points")
chk(all(on_circle(arc, P) for P in pts), "both exactly on the circle")
chk(any(P[0] == S(2) and P[1] == S(2) for P in pts), "contains (sqrt2, sqrt2)")
chk(any(P[0] == S(2) and P[1] == -S(2) for P in pts), "contains (sqrt2, -sqrt2)")

print(f"DENEST TEST PASSED: {ok} checks. Irrational-but-field sqrt now exact.")
