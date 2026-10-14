#!/usr/bin/env python3
"""
auto_partition.py - Phase 3 prototype: AUTOMATIC region decomposition.

Closes the Phase-2 gap ("where to cut" was hand-written). Inputs are ONLY the
figure primitives (a bounding square + circular arcs). The pipeline:

  1. ARRANGEMENT  - intersect every curve with every other, split each curve
                    into atomic edges at the intersection points.
  2. FACES        - walk the planar subdivision (half-edge / DCEL style) to
                    extract the atomic faces (smallest closed regions).
  3. EXACT AREA   - each face area by Green's theorem  A = 1/2 * contour
                    integral of (x dy - y dx):
                       * line segment -> shoelace term            (rational)
                       * circular arc -> sector term r^2*dtheta/2 (the pi part)
                                         + a rational drift term.
                    Areas are EXACT: rational + b*pi, with the sector angle kept
                    symbolic so ugly intermediate angles cancel.
  4. VERIFY       - all faces must sum to the whole figure (144) with the pi
                    coefficient and every symbolic angle-atom cancelling to 0.

No hand-written decomposition anywhere: faces are DERIVED from the primitives.
"""

from fractions import Fraction as F
import math

# =====================================================================
# Exact area:  rational + pi*(rational) + sum_k coeff_k * alpha_k
# alpha_k = unresolved sector angle (atan2 of a point about a circle centre).
# Cardinal (axis-aligned) directions resolve to a rational multiple of pi and
# fold into the pi term. A clean (a+b*pi) area requires every NON-cardinal atom
# to cancel to zero.
# =====================================================================

class Area:
    __slots__ = ("rat", "pi", "atoms")
    def __init__(self, rat=F(0), pi=F(0), atoms=None):
        self.rat = F(rat); self.pi = F(pi)
        self.atoms = dict(atoms) if atoms else {}
    def copy(self):
        return Area(self.rat, self.pi, self.atoms)
    def __iadd__(self, o):
        self.rat += o.rat; self.pi += o.pi
        for k, v in o.atoms.items():
            nv = self.atoms.get(k, F(0)) + v
            if nv == 0: self.atoms.pop(k, None)
            else: self.atoms[k] = nv
        return self
    def __add__(self, o):
        r = self.copy(); r += o; return r
    def neg(self):
        return Area(-self.rat, -self.pi, {k: -v for k, v in self.atoms.items()})
    def add_atom(self, key, coeff):
        nv = self.atoms.get(key, F(0)) + coeff
        if nv == 0: self.atoms.pop(key, None)
        else: self.atoms[key] = nv

def resolve(area, circles):
    out = Area(area.rat, area.pi); residual = {}
    for (cid, px, py), coeff in area.atoms.items():
        cx, cy, r = circles[cid]
        dx, dy = px - cx, py - cy
        if dx == 0 or dy == 0:
            if dy == 0: unit = F(0) if dx > 0 else F(1)
            else:       unit = F(1, 2) if dy > 0 else F(-1, 2)
            out.pi += coeff * unit
        else:
            residual[(cid, px, py)] = coeff
    return out, residual

def fmt(area, circles):
    out, residual = resolve(area, circles)
    if residual:
        return f"<non-closed: residual atoms {residual}>"
    a, b = out.rat, out.pi
    def rs(x):
        return str(x.numerator) if x.denominator == 1 else f"{x.numerator}/{x.denominator}"
    if b == 0: return rs(a)
    if a == 0: return f"{rs(b)}*pi"
    if a < 0 and b > 0: return f"{rs(b)}*pi - {rs(-a)}"
    sign = "-" if b < 0 else "+"
    return f"{rs(a)} {sign} {rs(abs(b))}*pi"

def is_closed(area, circles):
    _, residual = resolve(area, circles); return not residual

def to_float(area, circles):
    out, _ = resolve(area, circles)
    return float(out.rat) + float(out.pi) * math.pi

# =====================================================================
# Geometry primitives
# =====================================================================

class Seg:
    def __init__(self, a, b):
        self.a = a; self.b = b; self.kind = "seg"

class Arc:
    """Circle centre (cx,cy) radius r, drawn CCW from start_deg to end_deg."""
    _next = 0
    def __init__(self, cx, cy, r, start_deg, end_deg, name=""):
        self.cx = F(cx); self.cy = F(cy); self.r = F(r)
        self.t0 = math.radians(start_deg); self.t1 = math.radians(end_deg)
        self.name = name; self.kind = "arc"
        self.cid = Arc._next; Arc._next += 1
    def angle_of(self, p):
        return math.atan2(float(p[1]-self.cy), float(p[0]-self.cx))

def snap(x):
    return F(x).limit_denominator(100000)

def on_arc_span(arc, p, tol=1e-7):
    a = arc.angle_of(p); span = arc.t1 - arc.t0
    da = (a - arc.t0) % (2*math.pi)
    if span >= 0:
        return -tol <= da <= span + tol
    return (da <= tol) or (da >= (2*math.pi + span) - tol)

# =====================================================================
# Intersections (float compute -> snap to exact rational, verify on-curve)
# =====================================================================

def inter_arc_arc(a1, a2):
    dx = float(a2.cx - a1.cx); dy = float(a2.cy - a1.cy)
    d = math.hypot(dx, dy); r1 = float(a1.r); r2 = float(a2.r)
    if d == 0 or d > r1 + r2 + 1e-9 or d < abs(r1 - r2) - 1e-9:
        return []
    aa = (r1*r1 - r2*r2 + d*d) / (2*d); h2 = r1*r1 - aa*aa
    if h2 < -1e-9: return []
    h = math.sqrt(max(h2, 0.0))
    xm = float(a1.cx) + aa*dx/d; ym = float(a1.cy) + aa*dy/d
    out = []
    for px, py in [(xm + h*dy/d, ym - h*dx/d), (xm - h*dy/d, ym + h*dx/d)]:
        P = (snap(px), snap(py))
        if on_arc_span(a1, P) and on_arc_span(a2, P):
            out.append(P)
    return out

def inter_arc_seg(arc, seg):
    (x1, y1), (x2, y2) = seg.a, seg.b
    cx, cy, r = float(arc.cx), float(arc.cy), float(arc.r)
    dx, dy = float(x2 - x1), float(y2 - y1)
    fx, fy = float(x1) - cx, float(y1) - cy
    A = dx*dx + dy*dy; B = 2*(fx*dx + fy*dy); C = fx*fx + fy*fy - r*r
    disc = B*B - 4*A*C
    if disc < -1e-9 or A == 0: return []
    disc = math.sqrt(max(disc, 0.0)); out = []
    for t in ((-B - disc)/(2*A), (-B + disc)/(2*A)):
        if -1e-9 <= t <= 1 + 1e-9:
            tt = snap(t)
            P = (snap(float(x1) + float(tt)*dx), snap(float(y1) + float(tt)*dy))
            if on_arc_span(arc, P):
                out.append(P)
    return out

def inter_seg_seg(s1, s2):
    (x1, y1), (x2, y2) = (F(s1.a[0]),F(s1.a[1])), (F(s1.b[0]),F(s1.b[1]))
    (x3, y3), (x4, y4) = (F(s2.a[0]),F(s2.a[1])), (F(s2.b[0]),F(s2.b[1]))
    den = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if den == 0: return []
    t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / den
    u = ((x1-x3)*(y1-y2) - (y1-y3)*(x1-x2)) / den
    if 0 <= t <= 1 and 0 <= u <= 1:
        return [(x1 + t*(x2-x1), y1 + t*(y2-y1))]
    return []

# ====================================================================
# Arrangement -> half-edges -> faces
# =====================================================================

class HalfEdge:
    __slots__ = ("u","v","dep","area","twin","idx","is_arc","cid",
                 "cx","cy","r","alo","ahi","ccw")

def _node_id(coord, nodes, index):
    if coord not in index:
        index[coord] = len(nodes); nodes.append(coord)
    return index[coord]

def build(segs, arcs):
    nodes = []; index = {}
    circles = {a.cid: (a.cx, a.cy, a.r) for a in arcs}
    seg_nodes = {i: set() for i in range(len(segs))}
    arc_nodes = {i: set() for i in range(len(arcs))}
    for i, a in enumerate(arcs):
        for ang in (a.t0, a.t1):
            p = (snap(round(float(a.cx + a.r*snap(math.cos(ang))), 6)),
                 snap(round(float(a.cy + a.r*snap(math.sin(ang))), 6)))
            arc_nodes[i].add(p)
    for i, s in enumerate(segs):
        seg_nodes[i].add((F(s.a[0]), F(s.a[1])))
        seg_nodes[i].add((F(s.b[0]), F(s.b[1])))
    for i in range(len(arcs)):
        for j in range(i+1, len(arcs)):
            for p in inter_arc_arc(arcs[i], arcs[j]):
                arc_nodes[i].add(p); arc_nodes[j].add(p)
    for i in range(len(arcs)):
        for j in range(len(segs)):
            for p in inter_arc_seg(arcs[i], segs[j]):
                arc_nodes[i].add(p); seg_nodes[j].add(p)
    for i in range(len(segs)):
        for j in range(i+1, len(segs)):
            for p in inter_seg_seg(segs[i], segs[j]):
                seg_nodes[i].add(p); seg_nodes[j].add(p)

    halfedges = []
    def add_pair(a, b):
        a.idx = len(halfedges); halfedges.append(a)
        b.idx = len(halfedges); halfedges.append(b)
        a.twin = b.idx; b.twin = a.idx

    for i, s in enumerate(segs):
        ax, ay = F(s.a[0]), F(s.a[1]); bx, by = F(s.b[0]), F(s.b[1])
        dxs, dys = bx-ax, by-ay
        def param(p, dxs=dxs, dys=dys, ax=ax, ay=ay):
            return (p[0]-ax)/dxs if dxs != 0 else (p[1]-ay)/dys
        pts = sorted(seg_nodes[i], key=lambda p: float(param(p)))
        for k in range(len(pts)-1):
            P, Q = pts[k], pts[k+1]
            uid = _node_id(P, nodes, index); vid = _node_id(Q, nodes, index)
            h1 = HalfEdge(); h1.u=uid; h1.v=vid; h1.is_arc=False
            h1.area = Area(rat=F(1,2)*(P[0]*Q[1]-Q[0]*P[1]))
            h2 = HalfEdge(); h2.u=vid; h2.v=uid; h2.is_arc=False
            h2.area = Area(rat=F(1,2)*(Q[0]*P[1]-P[0]*Q[1]))
            add_pair(h1, h2)

    for i, a in enumerate(arcs):
        pa = sorted(arc_nodes[i], key=lambda p: a.angle_of(p))
        for k in range(len(pa)-1):
            P, Q = pa[k], pa[k+1]
            angP = a.angle_of(P); angQ = a.angle_of(Q)
            uid = _node_id(P, nodes, index); vid = _node_id(Q, nodes, index)
            keyP = (a.cid, P[0], P[1]); keyQ = (a.cid, Q[0], Q[1]); r2 = a.r*a.r
            h1 = HalfEdge(); h1.u=uid; h1.v=vid; h1.is_arc=True; h1.cid=a.cid
            h1.cx=a.cx; h1.cy=a.cy; h1.r=a.r; h1.alo=angP; h1.ahi=angQ; h1.ccw=True
            ar1 = Area(rat=(a.cx/2)*(Q[1]-P[1]) + (a.cy/2)*(P[0]-Q[0]))
            ar1.add_atom(keyQ, r2/2); ar1.add_atom(keyP, -r2/2); h1.area = ar1
            h2 = HalfEdge(); h2.u=vid; h2.v=uid; h2.is_arc=True; h2.cid=a.cid
            h2.cx=a.cx; h2.cy=a.cy; h2.r=a.r; h2.alo=angQ; h2.ahi=angP; h2.ccw=False
            ar2 = Area(rat=(a.cx/2)*(P[1]-Q[1]) + (a.cy/2)*(Q[0]-P[0]))
            ar2.add_atom(keyP, r2/2); ar2.add_atom(keyQ, -r2/2); h2.area = ar2
            add_pair(h1, h2)
    return nodes, halfedges, circles

def faces(nodes, halfedges):
    """Canonical DCEL face extraction. next(h) = clockwise neighbour of twin(h)
    in the target node's cyclic order. Tangencies/cusps resolved by sorting on a
    finite epsilon-sample along each edge (curvature tie-break)."""
    DS = 1e-3   # sample step in ARC LENGTH (not angle): a circular arc's
                #   chord-tangent deviation is dtheta/2 = ds/(2r), so equal arc
                #   length separates same-tangent arcs by curvature/radius.
    def dep_key(h):
        ux, uy = float(nodes[h.u][0]), float(nodes[h.u][1])
        if h.is_arc:
            dtheta = DS / float(h.r)
            ang = h.alo + (dtheta if h.ccw else -dtheta)
            sx = float(h.cx) + float(h.r)*math.cos(ang)
            sy = float(h.cy) + float(h.r)*math.sin(ang)
        else:
            vx, vy = float(nodes[h.v][0]), float(nodes[h.v][1])
            sx = ux + DS*(vx-ux); sy = uy + DS*(vy-uy)
        return math.atan2(sy-uy, sx-ux)
    out = {}
    for h in halfedges:
        h.dep = dep_key(h); out.setdefault(h.u, []).append(h)
    for v in out:
        out[v].sort(key=lambda h: h.dep)
    posn = {}
    for v, lst in out.items():
        for k, h in enumerate(lst): posn[h.idx] = k
    def nxt(h):
        g = halfedges[h.twin]; lst = out[h.v]; k = posn[g.idx]
        return lst[(k-1) % len(lst)]
    seen = set(); res = []
    for h0 in halfedges:
        if h0.idx in seen: continue
        loop = []; h = h0
        while h.idx not in seen:
            seen.add(h.idx); loop.append(h); h = nxt(h)
        area = Area()
        for e in loop: area += e.area
        res.append((loop, area))
    return res

# =====================================================================
# Identification helpers
# =====================================================================

def point_in_face(q, loop, nodes):
    """Ray-cast (+x) crossing test against a face boundary of segs and arcs."""
    qx, qy = q; cnt = 0
    for h in loop:
        ux, uy = float(nodes[h.u][0]), float(nodes[h.u][1])
        vx, vy = float(nodes[h.v][0]), float(nodes[h.v][1])
        if not h.is_arc:
            if (uy > qy) != (vy > qy):
                xc = ux + (qy-uy)*(vx-ux)/(vy-uy)
                if xc > qx: cnt += 1
        else:
            cx, cy, r = float(h.cx), float(h.cy), float(h.r)
            disc = r*r - (qy-cy)**2
            if disc <= 0: continue
            sq = math.sqrt(disc)
            for xc in (cx+sq, cx-sq):
                if xc <= qx: continue
                ang = math.atan2(qy-cy, xc-cx)
                lo, hi = (h.alo, h.ahi) if h.ccw else (h.ahi, h.alo)
                da = (ang - lo) % (2*math.pi); span = (hi - lo) % (2*math.pi)
                if 1e-12 < da < span - 1e-12:
                    cnt += 1
    return cnt % 2 == 1
