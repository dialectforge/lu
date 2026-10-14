#!/usr/bin/env python3
"""
geom_exact.py - Phase 3 (completed): EXACT automatic region decomposition.

Everything that used to be float-then-snap is now exact over the multi-quadratic
field in algnum.Alg:

  * intersections emit exact algebraic coordinates (no snap-to-rational);
  * tangency is detected exactly (discriminant sign == 0);
  * the face walk orders edges by EXACT outgoing tangent, with an EXACT
    signed-curvature tie-break for tangencies/cusps (no epsilon arc-length);
  * point-in-face is an EXACT ray cast;
  * areas are a + b*pi with the sector angle kept symbolic (angle atoms),
    and the closed-form gate is an exact zero test.

Scope boundary (honest): exact sqrt is implemented for RATIONAL radicands,
which covers every intersection among primitives with rational data and chords
between rational vertices. A figure that forces a sqrt of an already-irrational
quantity (nested radicals) is detected and flagged, not silently approximated.
"""

from fractions import Fraction as F
from math import sqrt as _fsqrt
import functools
from algnum import Alg, isqrt_field

R = Alg.rational


class NeedsNestedRadical(Exception):
    pass


def sqrt_exact(a):
    """Exact sqrt of an Alg. Rational radicands are immediate; irrational ones
    are denested into the multi-quadratic field when possible. A genuine new
    nested radical is flagged (NeedsNestedRadical), never approximated."""
    a = a if isinstance(a, Alg) else R(a)
    if not a.primes():
        return Alg.sqrt_rational(a.t.get(frozenset(), F(0)))
    b = isqrt_field(a)
    if b is None:
        raise NeedsNestedRadical(f"sqrt of {a!r} needs a new nested radical")
    return b


# ----------------------------------------------------------------------
# exact 2D vector helpers on (Alg, Alg)
# ----------------------------------------------------------------------

def sub(a, b):  return (a[0] - b[0], a[1] - b[1])
def add(a, b):  return (a[0] + b[0], a[1] + b[1])
def dot(a, b):  return a[0] * b[0] + a[1] * b[1]
def cross(a, b):  return a[0] * b[1] - a[1] * b[0]
def smul(s, a): return (s * a[0], s * a[1])

def _half(w):
    """0 for angle in [0,pi)  (upper half incl +x axis), 1 for [pi,2pi)."""
    sy = w[1].sign()
    if sy > 0:
        return 0
    if sy < 0:
        return 1
    return 0 if w[0].sign() > 0 else 1   # on x-axis: +x -> 0, -x -> 1

def angle_cmp(u, v):
    """Compare two nonzero vectors by CCW angle in [0,2pi). Exact."""
    hu, hv = _half(u), _half(v)
    if hu != hv:
        return -1 if hu < hv else 1
    c = cross(u, v).sign()
    if c > 0:
        return -1          # u is clockwise of v => smaller angle
    if c < 0:
        return 1
    return 0               # same direction

def _rot_conj(w, ref):
    """Rotate w by -angle(ref): multiply complex w by conj(ref)."""
    return (w[0] * ref[0] + w[1] * ref[1], w[1] * ref[0] - w[0] * ref[1])


# ----------------------------------------------------------------------
# primitives
# ----------------------------------------------------------------------

def _A(x):
    return x if isinstance(x, Alg) else R(x)

class Seg:
    kind = "seg"
    def __init__(self, a, b):
        self.a = (_A(a[0]), _A(a[1]))
        self.b = (_A(b[0]), _A(b[1]))

class Arc:
    """Circle centre O, radius^2 = Rr; arc from P0 to P1 going CCW."""
    _next = 0
    def __init__(self, cx, cy, r, start_deg, end_deg, name=""):
        self.O = (R(cx), R(cy))
        self.Rr = R(F(r) ** 2)
        self.rf = _fsqrt(self.Rr.to_float())
        self.P0 = self._pt_deg(cx, cy, r, start_deg)
        self.P1 = self._pt_deg(cx, cy, r, end_deg)
        self.full = (self.P0 == self.P1)
        self.name = name
        self.cid = Arc._next; Arc._next += 1

    @staticmethod
    def _pt_deg(cx, cy, r, deg):
        deg = deg % 360
        table = {0: (r, 0), 90: (0, r), 180: (-r, 0), 270: (0, -r)}
        if deg not in table:
            raise ValueError(f"non-cardinal degree {deg}; use Arc.from_points for general arcs")
        dx, dy = table[deg]
        return (R(cx) + R(dx), R(cy) + R(dy))

    @classmethod
    def from_points(cls, O, Rr, P0, P1, name=""):
        self = cls.__new__(cls)
        self.O = (R(O[0]), R(O[1])); self.Rr = R(Rr); self.rf = _fsqrt(self.Rr.to_float())
        self.P0 = (R(P0[0]), R(P0[1])); self.P1 = (R(P1[0]), R(P1[1]))
        self.full = (self.P0 == self.P1)
        self.name = name; self.cid = Arc._next; Arc._next += 1
        return self

    def span_le(self, P):
        """Is point P (on the circle) within the CCW span [P0, P1]?  exact."""
        s = sub(self.P0, self.O); e = sub(self.P1, self.O); w = sub(P, self.O)
        ep = _rot_conj(e, s); wp = _rot_conj(w, s)
        # full circle if P0 == P1
        if e == s:
            return True
        return angle_cmp(wp, ep) <= 0          # wp not past ep, and >=0 automatically


# ----------------------------------------------------------------------
# exact intersections  ->  list of (Alg, Alg) points
# ----------------------------------------------------------------------

def on_circle(arc, P):
    d = sub(P, arc.O)
    return (dot(d, d) - arc.Rr).is_zero()

def inter_seg_seg(s1, s2):
    (x1, y1), (x2, y2) = s1.a, s1.b
    (x3, y3), (x4, y4) = s2.a, s2.b
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if den.is_zero():
        return []
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
    u = ((x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2)) / den
    if t.sign() < 0 or (t - R(1)).sign() > 0:
        return []
    if u.sign() < 0 or (u - R(1)).sign() > 0:
        return []
    return [(x1 + t * (x2 - x1), y1 + t * (y2 - y1))]

def inter_arc_seg(arc, seg):
    (x1, y1), (x2, y2) = seg.a, seg.b
    ox, oy = arc.O
    dx, dy = x2 - x1, y2 - y1
    fx, fy = x1 - ox, y1 - oy
    A = dx * dx + dy * dy
    B = R(2) * (fx * dx + fy * dy)
    C = fx * fx + fy * fy - arc.Rr
    if A.is_zero():
        return []
    disc = B * B - R(4) * A * C
    sd = disc.sign()
    if sd < 0:
        return []
    root = sqrt_exact(disc)                       # rational disc -> exact
    out = []
    ts = [(-B - root) / (R(2) * A)] if sd == 0 else \
         [(-B - root) / (R(2) * A), (-B + root) / (R(2) * A)]
    for t in ts:
        if t.sign() < 0 or (t - R(1)).sign() > 0:
            continue
        P = (x1 + t * dx, y1 + t * dy)
        if arc.span_le(P):
            out.append(P)
    return out

def inter_arc_arc(a1, a2):
    ox1, oy1 = a1.O; ox2, oy2 = a2.O
    # radical axis: 2(ox2-ox1)x + 2(oy2-oy1)y = (ox2^2+oy2^2-R2) - (ox1^2+oy1^2-R1)
    Ax = R(2) * (ox2 - ox1); Ay = R(2) * (oy2 - oy1)
    k = (ox2 * ox2 + oy2 * oy2 - a2.Rr) - (ox1 * ox1 + oy1 * oy1 - a1.Rr)
    if Ax.is_zero() and Ay.is_zero():
        return []                                  # concentric
    pts = []
    # parametrise the radical line and meet circle 1
    if not Ay.is_zero():
        # y = (k - Ax x)/Ay  -> substitute
        m = -Ax / Ay; b = k / Ay                  # y = m x + b
        fx, fy = ox1, oy1
        qa = R(1) + m * m
        qb = R(2) * (m * (b - fy) - fx)
        qc = fx * fx + (b - fy) * (b - fy) - a1.Rr
        disc = qb * qb - R(4) * qa * qc
        if disc.sign() < 0:
            return []
        root = sqrt_exact(disc)
        xs = [(-qb - root) / (R(2) * qa)] if disc.is_zero() else \
             [(-qb - root) / (R(2) * qa), (-qb + root) / (R(2) * qa)]
        for x in xs:
            pts.append((x, m * x + b))
    else:
        # Ax != 0, vertical radical line x = k/Ax
        x = k / Ax; fx, fy = ox1, oy1
        qa = R(1); qb = R(-2) * fy
        qc = (x - fx) * (x - fx) + fy * fy - a1.Rr
        disc = qb * qb - R(4) * qa * qc
        if disc.sign() < 0:
            return []
        root = sqrt_exact(disc)
        ys = [(-qb - root) / R(2)] if disc.is_zero() else \
             [(-qb - root) / R(2), (-qb + root) / R(2)]
        for y in ys:
            pts.append((x, y))
    out = []
    for P in pts:
        if a1.span_le(P) and a2.span_le(P):
            out.append(P)
    return out


# ----------------------------------------------------------------------
# exact symbolic area:  rat + pi*pic + sum atoms
# atom key = (circle id, point) -> coeff (Alg). Cardinal atoms resolve to pi.
# ----------------------------------------------------------------------

class Area:
    __slots__ = ("rat", "pic", "atoms")
    def __init__(self, rat=None, pic=None, atoms=None):
        self.rat = rat if rat is not None else Alg()
        self.pic = pic if pic is not None else Alg()
        self.atoms = dict(atoms) if atoms else {}
    def copy(self):
        return Area(self.rat, self.pic, self.atoms)
    def __iadd__(self, o):
        self.rat = self.rat + o.rat
        self.pic = self.pic + o.pic
        for k, v in o.atoms.items():
            nv = self.atoms.get(k, Alg()) + v
            if nv.is_zero(): self.atoms.pop(k, None)
            else: self.atoms[k] = nv
        return self
    def __add__(self, o):
        r = self.copy(); r += o; return r
    def add_atom(self, key, coeff):
        nv = self.atoms.get(key, Alg()) + coeff
        if nv.is_zero(): self.atoms.pop(key, None)
        else: self.atoms[key] = nv

def _classify_angle(c, s, Rr):
    """theta = atan2(s, c) about a circle of radius^2 Rr, returned as
    unit = theta/pi in [0, 2) if theta is a rational multiple of pi, else None.

    Niven's theorem regime: a chord direction is a rational multiple of pi iff
    cos^2(theta) = c^2/Rr lands in {0, 1/4, 1/2, 3/4, 1} (i.e. theta is a
    multiple of pi/6 or pi/4). The branch cut is the +x axis (theta in [0,2pi)),
    matching the winding comparator used in build(). EXACT; nothing else is
    resolved (it is flagged, not approximated)."""
    ratio = (c * c) / Rr
    if ratio.primes():
        return None                                   # cos^2 not even rational
    q = ratio.t.get(frozenset(), F(0))
    sc = c.sign(); ss = s.sign()
    if q == 1:
        return F(0) if sc > 0 else F(1)               # 0 (E) or pi (W)
    if q == 0:
        return F(1, 2) if ss > 0 else F(3, 2)         # pi/2 (N) or 3pi/2 (S)
    if q == F(1, 4):                                  # cos = +-1/2
        if sc > 0:  return F(1, 3) if ss > 0 else F(5, 3)
        else:       return F(2, 3) if ss > 0 else F(4, 3)
    if q == F(1, 2):                                  # cos = +-sqrt2/2
        if sc > 0:  return F(1, 4) if ss > 0 else F(7, 4)
        else:       return F(3, 4) if ss > 0 else F(5, 4)
    if q == F(3, 4):                                  # cos = +-sqrt3/2
        if sc > 0:  return F(1, 6) if ss > 0 else F(11, 6)
        else:       return F(5, 6) if ss > 0 else F(7, 6)
    return None

def resolve(area, circles):
    """Fold resolvable angle atoms into the pi term; return (rat, pic, residual)."""
    rat = area.rat; pic = area.pic; residual = {}
    for (cid, px, py), coeff in area.atoms.items():
        cx, cy, Rr = circles[cid]
        unit = _classify_angle(px - cx, py - cy, Rr)
        if unit is not None:
            pic = pic + coeff * R(unit)
        else:
            residual[(cid, px, py)] = coeff
    return rat, pic, residual

def is_closed(area, circles):
    _, _, residual = resolve(area, circles)
    return not residual

def area_float(area, circles):
    import math
    rat, pic, residual = resolve(area, circles)
    if residual:
        return float("nan")
    return rat.to_float() + pic.to_float() * math.pi

def area_exact(area, circles):
    """Return (rat_Alg, pic_Alg) for a closed-form face, or None if residual."""
    rat, pic, residual = resolve(area, circles)
    if residual:
        return None
    return rat, pic

def _is_rational(x):
    return not x.primes()

def _rs(x):
    """String for a rational Alg."""
    q = x.t.get(frozenset(), F(0)) if not x.is_zero() else F(0)
    return str(q.numerator) if q.denominator == 1 else f"{q.numerator}/{q.denominator}"

def fmt(area, circles):
    rat, pic, residual = resolve(area, circles)
    if residual:
        return f"<non-closed: {len(residual)} residual atom(s)>"
    a, b = rat, pic
    if _is_rational(a) and _is_rational(b):
        if b.is_zero(): return _rs(a)
        if a.is_zero(): return f"{_rs(b)}*pi"
        if a.sign() < 0 and b.sign() > 0: return f"{_rs(b)}*pi - {_rs(-a)}"
        sign = "-" if b.sign() < 0 else "+"
        babs = b if b.sign() > 0 else -b
        return f"{_rs(a)} {sign} {_rs(babs)}*pi"
    # general: algebraic rational part and/or pi coefficient
    parts = []
    if not a.is_zero():
        parts.append(repr(a))
    if not b.is_zero():
        parts.append(f"({repr(b)})*pi" if not _is_rational(b) else f"{_rs(b)}*pi")
    return " + ".join(parts) if parts else "0"


# ----------------------------------------------------------------------
# half-edge arrangement + EXACT face walk
# ----------------------------------------------------------------------

class HE:
    __slots__ = ("u", "v", "area", "twin", "idx", "is_arc", "cid",
                 "O", "Rr", "tail_pt", "head_pt", "ccw", "tangent", "kappa_sign", "Rr_for_tie")

def _sort_along_arc(arc, pts):
    """Sort points (on the arc) in CCW traversal order from P0."""
    s = sub(arc.P0, arc.O)
    keyed = [( _rot_conj(sub(P, arc.O), s), P) for P in pts]
    keyed.sort(key=functools.cmp_to_key(lambda a, b: angle_cmp(a[0], b[0])))
    return [P for _, P in keyed]

def build(segs, arcs):
    nodes = []; index = {}
    circles = {a.cid: (a.O[0], a.O[1], a.Rr) for a in arcs}

    def nid(P):
        key = (P[0], P[1])
        if key not in index:
            index[key] = len(nodes); nodes.append(P)
        return index[key]

    seg_nodes = {i: set() for i in range(len(segs))}
    arc_nodes = {i: set() for i in range(len(arcs))}
    for i, a in enumerate(arcs):
        arc_nodes[i].add(a.P0); arc_nodes[i].add(a.P1)
    for i, s in enumerate(segs):
        seg_nodes[i].add(s.a); seg_nodes[i].add(s.b)

    def addp(store, key, P):
        store[key].add(P)

    for i in range(len(arcs)):
        for j in range(i + 1, len(arcs)):
            for P in inter_arc_arc(arcs[i], arcs[j]):
                arc_nodes[i].add(P); arc_nodes[j].add(P)
    for i in range(len(arcs)):
        for j in range(len(segs)):
            for P in inter_arc_seg(arcs[i], segs[j]):
                arc_nodes[i].add(P); seg_nodes[j].add(P)
    for i in range(len(segs)):
        for j in range(i + 1, len(segs)):
            for P in inter_seg_seg(segs[i], segs[j]):
                seg_nodes[i].add(P); seg_nodes[j].add(P)

    halfedges = []
    def add_pair(h1, h2):
        h1.idx = len(halfedges); halfedges.append(h1)
        h2.idx = len(halfedges); halfedges.append(h2)
        h1.twin = h2.idx; h2.twin = h1.idx

    # segment edges
    for i, s in enumerate(segs):
        ax, ay = s.a; bx, by = s.b
        dxs, dys = bx - ax, by - ay
        def param(P):
            return (P[0] - ax) / dxs if not dxs.is_zero() else (P[1] - ay) / dys
        pts = sorted(seg_nodes[i], key=functools.cmp_to_key(
            lambda P, Q: -1 if param(P) < param(Q) else (1 if param(P) > param(Q) else 0)))
        for k in range(len(pts) - 1):
            P, Q = pts[k], pts[k + 1]
            uid, vid = nid(P), nid(Q)
            tang_uv = sub(Q, P); tang_vu = sub(P, Q)
            h1 = HE(); h1.u = uid; h1.v = vid; h1.is_arc = False
            h1.area = Area(rat=(P[0] * Q[1] - Q[0] * P[1]) * R(F(1, 2)))
            h1.tail_pt = P; h1.head_pt = Q; h1.tangent = tang_uv
            h1.kappa_sign = 0; h1.Rr_for_tie = R(0)
            h2 = HE(); h2.u = vid; h2.v = uid; h2.is_arc = False
            h2.area = Area(rat=(Q[0] * P[1] - P[0] * Q[1]) * R(F(1, 2)))
            h2.tail_pt = Q; h2.head_pt = P; h2.tangent = tang_vu
            h2.kappa_sign = 0; h2.Rr_for_tie = R(0)
            add_pair(h1, h2)

    # arc edges (between angularly-adjacent split points, CCW)
    for i, a in enumerate(arcs):
        if a.full and len(arc_nodes[i]) < 2:
            # lone full circle: add the antipode of P0 so it forms a 2-edge loop
            arc_nodes[i].add((R(2) * a.O[0] - a.P0[0], R(2) * a.O[1] - a.P0[1]))
        pa = _sort_along_arc(a, list(arc_nodes[i]))
        m = len(pa)
        npairs = m if a.full else m - 1           # full circle wraps last->first
        for k in range(npairs):
            P, Q = pa[k], pa[(k + 1) % m]         # CCW: theta(P) < theta(Q)
            if P == Q:
                continue
            uid, vid = nid(P), nid(Q)
            keyP = (a.cid, P[0], P[1]); keyQ = (a.cid, Q[0], Q[1])
            halfR = a.Rr * R(F(1, 2))
            ox, oy = a.O
            # winding: does the CCW arc P->Q cross the +x branch cut (theta_Q < theta_P)?
            wrap = angle_cmp((Q[0] - ox, Q[1] - oy), (P[0] - ox, P[1] - oy)) < 0
            # ccw half-edge P->Q : +R/2 at head Q, -R/2 at tail P
            h1 = HE(); h1.u = uid; h1.v = vid; h1.is_arc = True; h1.cid = a.cid
            h1.O = a.O; h1.Rr = a.Rr
            drift1 = (ox * (Q[1] - P[1]) + oy * (P[0] - Q[0])) * R(F(1, 2))
            ar1 = Area(rat=drift1); ar1.add_atom(keyQ, halfR); ar1.add_atom(keyP, -halfR)
            if wrap:
                ar1.pic = ar1.pic + a.Rr            # +2pi*(R/2) winding term
            h1.area = ar1
            h1.tail_pt = P; h1.head_pt = Q; h1.ccw = True
            h1.tangent = (-(P[1] - oy), P[0] - ox)      # CCW tangent at tail P
            h1.kappa_sign = 1; h1.Rr_for_tie = a.Rr
            # cw twin Q->P : +R/2 at head P, -R/2 at tail Q
            h2 = HE(); h2.u = vid; h2.v = uid; h2.is_arc = True; h2.cid = a.cid
            h2.O = a.O; h2.Rr = a.Rr
            drift2 = (ox * (P[1] - Q[1]) + oy * (Q[0] - P[0])) * R(F(1, 2))
            ar2 = Area(rat=drift2); ar2.add_atom(keyP, halfR); ar2.add_atom(keyQ, -halfR)
            if wrap:
                ar2.pic = ar2.pic - a.Rr            # cw twin: -2pi*(R/2) winding term
            h2.area = ar2
            h2.tail_pt = Q; h2.head_pt = P; h2.ccw = False
            h2.tangent = ((Q[1] - oy), -(Q[0] - ox))    # CW tangent at tail Q
            h2.kappa_sign = -1; h2.Rr_for_tie = a.Rr
            add_pair(h1, h2)

    return nodes, halfedges, circles


def _edge_cmp(h1, h2):
    """Order half-edges leaving a common vertex by outgoing tangent (CCW),
    breaking exact tangencies by signed curvature (CW < straight < CCW;
    among same orientation, sharper arc is more extreme). EXACT."""
    c = angle_cmp(h1.tangent, h2.tangent)
    if c != 0:
        return c
    # same outgoing direction: tie-break by signed curvature.
    # sampled point sits at tangent rotated by +dtheta/2 (CCW) / -dtheta/2 (CW),
    # so ascending sampled-angle == ascending signed curvature.
    if h1.kappa_sign != h2.kappa_sign:
        return -1 if h1.kappa_sign < h2.kappa_sign else 1
    if h1.kappa_sign == 0:
        return 0
    # same sign & both arcs: larger |kappa| = smaller R.  ascending signed kappa:
    #   CCW(+): smaller kappa first => larger R first
    #   CW(-) : more negative first => smaller R first
    dR = (h1.Rr_for_tie - h2.Rr_for_tie).sign()
    if dR == 0:
        return 0
    if h1.kappa_sign > 0:
        return -1 if dR > 0 else 1
    else:
        return -1 if dR < 0 else 1

def faces(nodes, halfedges):
    out = {}
    for h in halfedges:
        out.setdefault(h.u, []).append(h)
    for v in out:
        out[v].sort(key=functools.cmp_to_key(_edge_cmp))
    posn = {}
    for v, lst in out.items():
        for k, h in enumerate(lst):
            posn[h.idx] = k
    def nxt(h):
        g = halfedges[h.twin]; lst = out[h.v]; k = posn[g.idx]
        return lst[(k - 1) % len(lst)]
    seen = set(); res = []
    for h0 in halfedges:
        if h0.idx in seen:
            continue
        loop = []; h = h0
        while h.idx not in seen:
            seen.add(h.idx); loop.append(h); h = nxt(h)
        area = Area()
        for e in loop:
            area += e.area
        res.append((loop, area))
    return res


# ----------------------------------------------------------------------
# EXACT point-in-face (ray cast +x from a rational interior query)
# ----------------------------------------------------------------------

_RAY_SLOPES = [F(0), F(1), F(-1), F(2), F(1, 2), F(-2), F(-1, 2),
               F(3), F(1, 3), F(3, 2), F(-3, 2), F(5, 7), F(7, 5)]

def _ray_is_generic(qx, qy, m, loop, circles):
    """A forward ray q + t*(1,m), t>0 is generic if it passes through no edge
    endpoint and is not tangent to any arc's circle ahead of q."""
    for h in loop:
        for V in (h.tail_pt, h.head_pt):
            if (V[0] - qx).sign() > 0 and ((V[1] - qy) - m * (V[0] - qx)).is_zero():
                return False
    for h in loop:
        if h.is_arc:
            cx, cy, Rr = circles[h.cid]
            # line y = m x + b0 meets circle: quadratic discriminant
            b0 = qy - m * qx
            A = R(1) + m * m
            B = R(2) * (m * (b0 - cy) - cx)
            C = cx * cx + (b0 - cy) * (b0 - cy) - Rr
            disc = B * B - R(4) * A * C
            if disc.sign() == 0:
                return False
    return True

def point_in_face(q, loop, nodes, circles):
    qx, qy = _A(q[0]), _A(q[1])
    m = None
    for cand in _RAY_SLOPES:
        if _ray_is_generic(qx, qy, cand, loop, circles):
            m = cand
            break
    if m is None:
        raise RuntimeError("no generic ray slope found")
    d = (R(1), R(m))
    cnt = 0
    for h in loop:
        if not h.is_arc:
            A = h.tail_pt; B = h.head_pt
            ex, ey = B[0] - A[0], B[1] - A[1]
            den = d[0] * ey - d[1] * ex
            if den.is_zero():
                continue
            # q + t d = A + u e  ->  solve
            rx, ry = A[0] - qx, A[1] - qy
            t = (rx * ey - ry * ex) / den
            u = (rx * d[1] - ry * d[0]) / den
            if t.sign() > 0 and u.sign() > 0 and (u - R(1)).sign() < 0:
                cnt += 1
        else:
            cx, cy, Rr = circles[h.cid]
            b0 = qy - R(m) * qx
            A = R(1) + R(m) * R(m)
            B = R(2) * (R(m) * (b0 - cy) - cx)
            C = cx * cx + (b0 - cy) * (b0 - cy) - Rr
            disc = B * B - R(4) * A * C
            if disc.sign() < 0:
                continue
            root = sqrt_exact(disc)
            for x in ((-B - root) / (R(2) * A), (-B + root) / (R(2) * A)):
                if (x - qx).sign() <= 0:               # behind ray
                    continue
                P = (x, R(m) * x + b0)
                if _on_he_arc_span(h, P):
                    cnt += 1
    return cnt % 2 == 1

def _on_he_arc_span(h, P):
    """Is P within the directed arc half-edge's angular span [tail, head] (CCW or CW)?"""
    O = h.O
    s = sub(h.tail_pt, O); e = sub(h.head_pt, O); w = sub(P, O)
    if not h.ccw:
        # CW: reverse roles -> treat span from head to tail CCW
        s, e = e, s
    ep = _rot_conj(e, s); wp = _rot_conj(w, s)
    if e == s:
        return True
    # strictly inside the span: 0 < ang(wp) < ang(ep)
    if angle_cmp(wp, ep) >= 0:
        return False
    at_tail = wp[1].is_zero() and wp[0].sign() > 0     # ang(wp) == 0
    return not at_tail


# ----------------------------------------------------------------------
# Nested / disconnected components: subtract islands (exact)
# ----------------------------------------------------------------------

def _numeric_signed_area(loop):
    """Float signed area of a cycle (CCW > 0). Used ONLY to classify a cycle as
    a bounded face vs an outer/hole contour; never for a reported number."""
    import math
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
            N = 200; prev = None
            for k in range(N + 1):
                th = aP + d * k / N
                x = cx + r * math.cos(th); y = cy + r * math.sin(th)
                if prev is not None:
                    tot += 0.5 * (prev[0] * y - x * prev[1])
                prev = (x, y)
    return tot

def _components(nodes, halfedges):
    parent = list(range(len(nodes)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for h in halfedges:
        ra, rb = find(h.u), find(h.v)
        if ra != rb:
            parent[ra] = rb
    return find

def faces_nested(nodes, halfedges, circles):
    """Like faces(), but corrects for disconnected/nested components: each
    component's footprint (its outer CW contour, exact Area) is subtracted from
    the smallest bounded face of another component that contains it. Returns the
    bounded (CCW) faces with island-corrected exact areas."""
    find = _components(nodes, halfedges)
    allf = faces(nodes, halfedges)
    classified = []
    neg_by_comp = {}
    for loop, area in allf:
        comp = find(loop[0].u)
        sa = _numeric_signed_area(loop)
        classified.append([loop, area, comp, sa])
        if sa < 0:
            neg_by_comp[comp] = (loop, area)           # one outer contour / comp
    positives = [c for c in classified if c[3] > 0]
    comp_node = {}
    for i in range(len(nodes)):
        comp_node.setdefault(find(i), i)
    # subtract each island footprint from its immediate enclosing positive face
    for comp, (nloop, narea) in neg_by_comp.items():
        rep = nodes[comp_node[comp]]
        enclosers = [c for c in positives
                     if c[2] != comp and point_in_face(rep, c[0], nodes, circles)]
        if not enclosers:
            continue                                   # global outer contour
        # smallest-area enclosing face = immediate parent
        parent = min(enclosers, key=lambda c: abs(c[3]))
        parent[1] = parent[1] + narea                  # narea is negative -> subtracts
    return [(loop, area) for loop, area, comp, sa in positives]
