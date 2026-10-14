#!/usr/bin/env python3
"""
algnum.py - exact multi-quadratic real algebraic numbers, dependency-free.

Phase 3 of the verifiable-geometry pipeline needs intersection coordinates that
are EXACT, not float-snapped-to-rational. For circles/lines with rational data
every intersection lands in a multi-quadratic field Q(sqrt d1, ..., sqrt dk):
the radical axis of two circles is rational, and meeting it with a circle is a
rational quadratic, so each crossing adds at most one sqrt(rational).

Representation
--------------
A number is a Q-linear combination over a basis of square-root products. We
factor each squarefree radicand into primes and key each basis element by the
FROZENSET of its primes (empty set = the rational part, since sqrt(1)=1):

    a + b*sqrt(2) + c*sqrt(3) + d*sqrt(6)   <->
        { frozenset(): a, frozenset({2}): b, frozenset({3}): c,
          frozenset({2,3}): d }

Multiplication is exact because sqrt(A)*sqrt(B) = (prod of shared primes) *
sqrt(A symmetric-difference B). Division rationalises one prime at a time.

Sign / comparison is EXACT and FLOAT-FREE: it recurses on the number of
radicals, comparing squares. This is what makes the geometry verifiable rather
than "numerically probably right". No external dependencies, so it ports to
Rust cleanly.
"""

from fractions import Fraction as F
from math import sqrt as _fsqrt

# ----------------------------------------------------------------------
# integer helpers
# ----------------------------------------------------------------------

def _factor(n):
    """Prime factorisation of a positive integer -> {prime: exponent}."""
    assert n >= 1
    f = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            f[d] = f.get(d, 0) + 1
            n //= d
        d += 1 if d == 2 else 2
    if n > 1:
        f[n] = f.get(n, 0) + 1
    return f


def _sqrt_int_parts(n):
    """sqrt(n) = k*sqrt(s) with s squarefree, n a positive integer.
    Returns (k, frozenset(primes of s))."""
    if n == 1:
        return 1, frozenset()
    fac = _factor(n)
    k = 1
    primes = set()
    for p, e in fac.items():
        k *= p ** (e // 2)
        if e % 2:
            primes.add(p)
    return k, frozenset(primes)


# ----------------------------------------------------------------------
# the field element
# ----------------------------------------------------------------------

class Alg:
    """An element of a multi-quadratic real field.  terms: {frozenset(primes): F}."""
    __slots__ = ("t",)

    def __init__(self, terms=None):
        if terms is None:
            self.t = {}
        else:
            # copy + drop zeros, coerce to Fraction
            self.t = {k: F(v) for k, v in terms.items() if v != 0}

    # ---- constructors ----
    @staticmethod
    def rational(q):
        q = F(q)
        return Alg({frozenset(): q}) if q != 0 else Alg()

    @staticmethod
    def sqrt_rational(q):
        """Exact sqrt of a non-negative rational q."""
        q = F(q)
        if q < 0:
            raise ValueError("sqrt of negative rational")
        if q == 0:
            return Alg()
        n = q.numerator * q.denominator          # sqrt(a/b) = sqrt(a*b)/b
        k, primes = _sqrt_int_parts(n)
        return Alg({primes: F(k, q.denominator)})

    # ---- basic queries ----
    def is_zero(self):
        return not self.t

    def primes(self):
        s = set()
        for k in self.t:
            s |= k
        return s

    def copy(self):
        return Alg(self.t)

    # ---- arithmetic ----
    def __add__(self, o):
        o = _coerce(o)
        r = dict(self.t)
        for k, v in o.t.items():
            nv = r.get(k, F(0)) + v
            if nv == 0:
                r.pop(k, None)
            else:
                r[k] = nv
        return Alg(r)

    __radd__ = __add__

    def __neg__(self):
        return Alg({k: -v for k, v in self.t.items()})

    def __sub__(self, o):
        return self + (-_coerce(o))

    def __rsub__(self, o):
        return _coerce(o) + (-self)

    def __mul__(self, o):
        o = _coerce(o)
        r = {}
        for ka, ca in self.t.items():
            for kb, cb in o.t.items():
                common = ka & kb
                key = ka ^ kb
                coeff = ca * cb
                for p in common:
                    coeff *= p
                nv = r.get(key, F(0)) + coeff
                if nv == 0:
                    r.pop(key, None)
                else:
                    r[key] = nv
        return Alg(r)

    __rmul__ = __mul__

    def _sigma(self, p):
        """Conjugation flipping sqrt(p) -> -sqrt(p): negate coeffs of keys with p."""
        return Alg({k: (-v if p in k else v) for k, v in self.t.items()})

    def inverse(self):
        if self.is_zero():
            raise ZeroDivisionError("inverse of 0")
        ps = self.primes()
        if not ps:                                  # pure rational
            return Alg.rational(F(1) / self.t[frozenset()])
        p = next(iter(ps))
        conj = self._sigma(p)
        denom = self * conj                         # eliminates prime p
        return conj * denom.inverse()

    def __truediv__(self, o):
        return self * _coerce(o).inverse()

    def __rtruediv__(self, o):
        return _coerce(o) * self.inverse()

    # ---- exact, float-free sign ----
    def sign(self):
        return _sign(self)

    def __eq__(self, o):
        return (self - _coerce(o)).is_zero()

    def __lt__(self, o):
        return (self - _coerce(o)).sign() < 0

    def __le__(self, o):
        return (self - _coerce(o)).sign() <= 0

    def __gt__(self, o):
        return (self - _coerce(o)).sign() > 0

    def __ge__(self, o):
        return (self - _coerce(o)).sign() >= 0

    def __hash__(self):
        return hash(tuple(sorted((tuple(sorted(k)), v) for k, v in self.t.items())))

    # ---- escape hatches (display / sampling only, never for decisions) ----
    def to_float(self):
        s = 0.0
        for k, v in self.t.items():
            base = 1
            for p in k:
                base *= p
            s += float(v) * _fsqrt(base)
        return s

    def __repr__(self):
        if self.is_zero():
            return "0"
        parts = []
        for k in sorted(self.t, key=lambda s: (len(s), sorted(s))):
            v = self.t[k]
            if not k:
                parts.append(str(v))
            else:
                rad = 1
                for p in sorted(k):
                    rad *= p
                parts.append(f"{v}*sqrt({rad})")
        return " + ".join(parts)


def _coerce(o):
    if isinstance(o, Alg):
        return o
    return Alg.rational(o)


def _sign(x):
    """Exact sign of x in {-1,0,1}, recursing on radical count, comparing squares."""
    x = Alg(x.t)                                  # drop zeros
    if x.is_zero():
        return 0
    ps = x.primes()
    if not ps:
        q = x.t[frozenset()]
        return 1 if q > 0 else -1
    p = next(iter(ps))
    # split x = A + sqrt(p)*C, with A, C in the subfield without prime p
    A_terms, C_terms = {}, {}
    for k, v in x.t.items():
        if p in k:
            C_terms[k - {p}] = C_terms.get(k - {p}, F(0)) + v
        else:
            A_terms[k] = A_terms.get(k, F(0)) + v
    A = Alg(A_terms)
    C = Alg(C_terms)
    sA = _sign(A)
    sC = _sign(C)
    if sC == 0:
        return sA
    if sA == 0:
        return sC                                  # value is sqrt(p)*C, sqrt(p)>0
    if sA == sC:
        return sA
    # opposite signs: compare |A| vs sqrt(p)|C| through D = A^2 - p*C^2
    D = A * A - (Alg.rational(p) * (C * C))
    sD = _sign(D)
    if sD == 0:
        return 0
    return sA if sD > 0 else sC


def isqrt_field(a, _depth=0):
    """Exact square root of a within the multi-quadratic field, by denesting
    sqrt(A + sqrt(p) B) = C + sqrt(p) D recursively. Returns the non-negative
    root as an Alg, or None if no square root exists in the field (a genuine new
    nested radical, or a is negative)."""
    a = Alg(a.t)
    if a.is_zero():
        return Alg()
    if _sign(a) < 0:
        return None
    if _depth > 12:
        return None                                   # guard against ping-pong
    ps = a.primes()
    if not ps:                                        # rational: always a field sqrt
        return Alg.sqrt_rational(a.t[frozenset()])
    p = next(iter(ps))
    A_terms, B_terms = {}, {}
    for k, v in a.t.items():
        if p in k:
            B_terms[k - {p}] = B_terms.get(k - {p}, F(0)) + v
        else:
            A_terms[k] = A_terms.get(k, F(0)) + v
    A = Alg(A_terms); B = Alg(B_terms)
    if B.is_zero():
        return isqrt_field(A, _depth + 1)
    disc = A * A - Alg.rational(p) * (B * B)
    sd = isqrt_field(disc, _depth + 1)
    if sd is None:
        return None
    sqrtp = Alg.sqrt_rational(p)
    half = Alg.rational(F(1, 2))
    for s in (sd, -sd):
        C2 = (A + s) * half
        if _sign(C2) < 0:
            continue
        c = isqrt_field(C2, _depth + 1)
        if c is None or c.is_zero():
            continue
        d = B * (half * c.inverse())                  # D = B / (2C)
        cand = c + sqrtp * d
        if (cand * cand - a).is_zero():
            return cand if _sign(cand) >= 0 else -cand
    return None
