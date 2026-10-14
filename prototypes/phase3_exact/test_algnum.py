#!/usr/bin/env python3
"""Hard tests for the exact algebraic-number field. Float used ONLY to
cross-check; all decisions inside Alg are exact."""
import random
from fractions import Fraction as F
from algnum import Alg

R = Alg.rational
S = Alg.sqrt_rational
n_ok = 0

def check(cond, msg):
    global n_ok
    assert cond, "FAIL: " + msg
    n_ok += 1

# --- construction / simplification ---
check(S(2).to_float() ** 2 - 2 < 1e-12, "sqrt2 numeric")
check((S(2) * S(2)) == R(2), "sqrt2*sqrt2 = 2 (exact)")
check((S(3) * S(3)) == R(3), "sqrt3*sqrt3 = 3")
check((S(2) * S(3)) == S(6), "sqrt2*sqrt3 = sqrt6")
check((S(2) * S(8)) == R(4), "sqrt2*sqrt8 = 4")
check(S(F(1, 2)) * S(2) == R(1), "sqrt(1/2)*sqrt2 = 1")
check(S(12) == R(2) * S(3), "sqrt12 = 2 sqrt3")
check(S(18) == R(3) * S(2), "sqrt18 = 3 sqrt2")
check(S(50) == R(5) * S(2), "sqrt50 = 5 sqrt2")

# --- ring axioms on random multi-quadratic elements ---
def rnd():
    e = Alg()
    for rad in (1, 2, 3, 5, 6, 7, 10, 15):
        if random.random() < 0.5:
            c = F(random.randint(-5, 5), random.randint(1, 4))
            e = e + R(c) * S(rad)
    return e

random.seed(7)
for _ in range(400):
    a, b, c = rnd(), rnd(), rnd()
    check((a + b) == (b + a), "comm +")
    check((a * b) == (b * a), "comm *")
    check(((a + b) + c) == (a + (b + c)), "assoc +")
    check(((a * b) * c) == (a * (b * c)), "assoc *")
    check((a * (b + c)) == (a * b + a * c), "distrib")
    # numeric agreement
    check(abs((a * b).to_float() - a.to_float() * b.to_float()) < 1e-6, "mul numeric")

# --- inverse / division ---
for _ in range(300):
    a = rnd()
    if a.is_zero():
        continue
    check((a * a.inverse()) == R(1), "a * 1/a = 1 (exact)")
    b = rnd()
    if not b.is_zero():
        check((a / b) * b == a, "(a/b)*b = a")
check((R(1) / (R(1) + S(2))) == (S(2) - R(1)), "1/(1+sqrt2) = sqrt2-1")
check((R(1) / (S(2) + S(3))) == (S(3) - S(2)), "1/(sqrt2+sqrt3) = sqrt3-sqrt2")

# --- EXACT SIGN: the load-bearing primitive ---
# nested / close values where float would be unreliable
# sqrt2 = 1.41421356237309504880...; bracket it at 1e-16 (float cannot)
check((S(2) - R(F(14142135623730950, 10000000000000000))).sign() == 1,
      "sqrt2 > 1.4142135623730950 (just below true value)")
check((S(2) - R(F(14142135623730951, 10000000000000000))).sign() == -1,
      "sqrt2 < 1.4142135623730951 (just above true value)")
check((S(2) + S(3) - S(17)).sign() == -1, "sqrt2+sqrt3 < sqrt17")
check((S(2) + S(3) - R(3)).sign() == 1, "sqrt2+sqrt3 > 3")
# (sqrt6 - sqrt2 - sqrt3 + 1) factors as (sqrt2-1)(sqrt3-1) > 0
check((S(6) - S(2) - S(3) + R(1)).sign() == 1, "(sqrt2-1)(sqrt3-1) > 0")
# a genuinely tiny nonzero algebraic number stays detected as nonzero
tiny = S(F(1000001, 1000000)) - S(1)               # sqrt(1.000001) - 1 > 0, ~5e-7
check(tiny.sign() == 1, "small positive radical difference detected")
check((-tiny).sign() == -1, "small negative detected")
# zero must be exactly zero
check((S(2) * S(2) - R(2)).sign() == 0, "exact zero sign")
check((S(8) - R(2) * S(2)).sign() == 0, "sqrt8 - 2sqrt2 = 0")

# sign consistent with float on random (non-tiny) elements
for _ in range(500):
    a = rnd()
    f = a.to_float()
    if abs(f) > 1e-4:
        check(a.sign() == (1 if f > 0 else -1), "sign matches float")

# comparison/ordering sanity used by the geometry sort
vals = [S(2), S(3), R(F(3, 2)), S(5), R(2), S(7) - R(1), R(0)]
fl = sorted(vals, key=lambda x: x.to_float())
ex = sorted(vals)                                  # uses exact __lt__
check([x.to_float() for x in fl] == [x.to_float() for x in ex], "exact sort == float sort")

print(f"algnum: ALL {n_ok} checks passed.")
