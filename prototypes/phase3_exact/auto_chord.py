#!/usr/bin/env python3
"""
auto_chord.py - Gap 4: PRINCIPLED automatic chord derivation.

The arcs alone sometimes cannot separate a shaded region (e.g. the red leaf's
right wedge, blocked by an internal tangency). A refining chord is needed. The
original prototype hand-supplied it; here it is DERIVED, with two pieces:

  1. CANDIDATE chords come from the figure's own latent straight lines:
       - any line through >= 2 existing arrangement vertices, and
       - every centre-to-centre line of a pair of arcs (this captures tangency
         axes: the centres and an internal/external tangency point are collinear).
     Each line is clipped to the figure's bounding box so the chord spans it.

  2. SELECTION is by SHADING PURITY, not by knowing the answer. The user marks
     shaded and unshaded seed points (the same shading input the pipeline
     already needs). We add the fewest candidate chords so that every face is
     shading-pure (no face holds both a shaded and an unshaded seed). The shaded
     area is then the exact union of the shaded faces, and is invariant to which
     valid chord set is chosen.

Scope: still flags (does not invent) figures that need a chord with no support
from existing vertices or arc centres.
"""
import functools
import itertools
from algnum import Alg
from geom_exact import (Seg, build, faces, area_float, is_closed,
                        point_in_face, cross, sub, R)


def _collinear(A, B, C):
    return cross(sub(B, A), sub(C, A)).is_zero()


def _bbox(nodes):
    xs = [p[0] for p in nodes]; ys = [p[1] for p in nodes]
    xmin = xmax = xs[0]; ymin = ymax = ys[0]
    for x in xs:
        if (x - xmin).sign() < 0: xmin = x
        if (x - xmax).sign() > 0: xmax = x
    for y in ys:
        if (y - ymin).sign() < 0: ymin = y
        if (y - ymax).sign() > 0: ymax = y
    return xmin, xmax, ymin, ymax


def _clip_line(P0, d, bbox):
    """Clip the infinite line P0 + t*d to the axis-aligned bbox. Return a Seg
    spanning the two boundary crossings, or None."""
    xmin, xmax, ymin, ymax = bbox
    dx, dy = d
    pts = []
    if not dx.is_zero():
        for xv in (xmin, xmax):
            t = (xv - P0[0]) / dx
            y = P0[1] + t * dy
            if (y - ymin).sign() >= 0 and (y - ymax).sign() <= 0:
                pts.append((xv, y))
    if not dy.is_zero():
        for yv in (ymin, ymax):
            t = (yv - P0[1]) / dy
            x = P0[0] + t * dx
            if (x - xmin).sign() >= 0 and (x - xmax).sign() <= 0:
                pts.append((x, yv))
    # dedup exactly
    uniq = []
    for p in pts:
        if not any(p[0] == q[0] and p[1] == q[1] for q in uniq):
            uniq.append(p)
    if len(uniq) < 2:
        return None
    # the two extreme crossings along d
    uniq.sort(key=functools.cmp_to_key(
        lambda a, b: ((a[0] - b[0]) * dx + (a[1] - b[1]) * dy).sign()))
    A, B = uniq[0], uniq[-1]
    if A[0] == B[0] and A[1] == B[1]:
        return None
    return Seg(A, B)


def _line_key(P0, d):
    """Canonical (a,b,c) for line a*x+b*y=c through P0 with direction d."""
    a, b = d[1], -d[0]                  # normal
    c = a * P0[0] + b * P0[1]
    if not a.is_zero():
        return (R(1), b / a, c / a)
    return (R(0), R(1), c / b)


def candidate_chords(nodes, arcs):
    """Latent lines of the figure, clipped to the bounding box, deduplicated."""
    bbox = _bbox(nodes)
    lines = {}
    # lines through >=2 vertices
    n = len(nodes)
    for i in range(n):
        for j in range(i + 1, n):
            d = sub(nodes[j], nodes[i])
            if d[0].is_zero() and d[1].is_zero():
                continue
            lines.setdefault(_line_key(nodes[i], d), (nodes[i], d))
    # centre-to-centre lines (tangency axes)
    for i in range(len(arcs)):
        for j in range(i + 1, len(arcs)):
            d = sub(arcs[j].O, arcs[i].O)
            if d[0].is_zero() and d[1].is_zero():
                continue
            lines.setdefault(_line_key(arcs[i].O, d), (arcs[i].O, d))
    chords = []
    seen = set()
    for P0, d in lines.values():
        seg = _clip_line(P0, d, bbox)
        if seg is None:
            continue
        key = frozenset({(seg.a[0], seg.a[1]), (seg.b[0], seg.b[1])})
        if key in seen:
            continue
        seen.add(key)
        chords.append(seg)
    return chords


# backward-compatible name used by older demos
def collinear_chords(nodes):
    out = {}
    n = len(nodes)
    for i in range(n):
        for j in range(i + 1, n):
            members = tuple(sorted(k for k in range(n)
                                   if _collinear(nodes[i], nodes[j], nodes[k])))
            if len(members) >= 3:
                out[members] = members
    chords = []
    seen = set()
    for members in out.values():
        pts = [nodes[k] for k in members]
        A = pts[0]; d = sub(pts[1], A)
        proj = [(sub(p, A)[0] * d[0] + sub(p, A)[1] * d[1], p) for p in pts]
        proj.sort(key=functools.cmp_to_key(lambda a, b: (a[0] - b[0]).sign()))
        P, Q = proj[0][1], proj[-1][1]
        key = frozenset({(P[0], P[1]), (Q[0], Q[1])})
        if key in seen:
            continue
        seen.add(key)
        chords.append(Seg(P, Q))
    return chords


def _face_seed_types(L, nodes, circ, shaded, unshaded):
    hasS = any(point_in_face(s, L, nodes, circ) for s in shaded)
    hasU = any(point_in_face(u, L, nodes, circ) for u in unshaded)
    return hasS, hasU


def refine_until_pure(arcs, base_segs, shaded, unshaded, max_chords=2):
    """Add the fewest candidate chords that make every face shading-pure.
    Returns (chords_used, shaded_area_Area, (nodes, circ)) or None.
    Shaded area is invariant to which valid chord set is found."""
    nodes0, he0, circ0 = build(base_segs, arcs)
    cands = candidate_chords(nodes0, arcs)
    for k in range(0, max_chords + 1):
        for combo in itertools.combinations(cands, k):
            nodes, he, circ = build(list(base_segs) + list(combo), arcs)
            fs = [(L, a) for L, a in faces(nodes, he) if area_float(a, circ) > 1e-6]
            pure = True
            shaded_acc = None
            covered_S, covered_U = set(), set()
            for L, a in fs:
                hasS, hasU = _face_seed_types(L, nodes, circ, shaded, unshaded)
                if hasS and hasU:
                    pure = False
                    break
                if hasS:
                    shaded_acc = a if shaded_acc is None else (shaded_acc + a)
                    for idx, s in enumerate(shaded):
                        if point_in_face(s, L, nodes, circ):
                            covered_S.add(idx)
                if hasU:
                    for idx, u in enumerate(unshaded):
                        if point_in_face(u, L, nodes, circ):
                            covered_U.add(idx)
            if not pure:
                continue
            if len(covered_S) != len(shaded) or len(covered_U) != len(unshaded):
                continue
            if shaded_acc is None or not is_closed(shaded_acc, circ):
                continue
            return list(combo), shaded_acc, (nodes, circ)
    return None
