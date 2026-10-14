#!/usr/bin/env python3
"""Declarative constraint DSL -> solver model.

A problem is a list of statements; the interpreter builds the numeric Problem
(placed points + constraints) plus a query that evaluates the 'find' target
(including constructed points). This is the layer perception / NL parsing will
target, so the metric solver no longer needs a hand-coded model per problem.

Statements (one per line, '#' comments):
  fix NAME X Y                 # gauge: pin a point
  point NAME                   # free point (random init)
  point NAME at X Y            # free point, init hint
  dist A B = V                 # |AB| = V
  angle A B C = D              # angle ABC = D degrees
  angle A B C = K angle D E F  # angle ABC = K * angle DEF
  perp A B C D                 # AB perpendicular to CD
  collinear A B C
  midpoint M A B               # M = (A+B)/2   (also: 'diameter A B center M')
  diameter A B center M        # M midpoint of AB (AB a diameter)
  oncircle P center O thru T   # |P-O| = |O-T|
  construct P far A from B thru T   # P on circle(center A, r=|A-T|), on line through A&B, away from B
  find angle A B C
  find length A B
"""
import math, numpy as np
from solver import (Problem, solve, angle, c_dist, c_angle, c_angle_eq,
                    c_perp, c_coll, c_mid, c_eqdist, c_between)

def parse(spec):
    p = Problem(); constructs = []; find = None
    for raw in spec.strip().splitlines():
        line = raw.split("#")[0].strip()
        if not line: continue
        w = line.split()
        op = w[0]
        if op == "fix":
            p.pt(w[1], float(w[2]), float(w[3]))
        elif op == "point":
            if len(w) >= 4 and w[2] == "at":
                p.pt(w[1], init=(float(w[3]), float(w[4])))
            else:
                p.pt(w[1])
        elif op == "dist":               # dist A B = V
            a,b,v = w[1], w[2], float(w[4]); p.add(lambda P,a=a,b=b,v=v: c_dist(P,a,b,v))
        elif op == "angle":              # angle A B C = ...
            a,b,c = w[1],w[2],w[3]
            if w[5] == "angle" or (len(w) > 6 and "angle" in w):
                # angle A B C = K angle D E F
                if w[5] == "angle": k=1.0; d,e,f = w[6],w[7],w[8]
                else: k=float(w[5]); d,e,f = w[7],w[8],w[9]
                p.add(lambda P,a=a,b=b,c=c,d=d,e=e,f=f,k=k: c_angle_eq(P,a,b,c,d,e,f,k))
            else:
                deg=float(w[5]); p.add(lambda P,a=a,b=b,c=c,deg=deg: c_angle(P,a,b,c,deg))
        elif op == "perp":
            a,b,c,d = w[1:5]; p.add(lambda P,a=a,b=b,c=c,d=d: c_perp(P,a,b,c,d))
        elif op == "collinear":
            a,b,c = w[1:4]; p.add(lambda P,a=a,b=b,c=c: c_coll(P,a,b,c))
        elif op == "between":            # between X Y Z : X on segment YZ
            x,y,z = w[1:4]
            if x not in p.names: p.pt(x)
            p.add(lambda P,x=x,y=y,z=z: c_between(P,x,y,z))
        elif op in ("midpoint","diameter"):
            if op=="midpoint": m,a,b = w[1],w[2],w[3]
            else: a,b,m = w[1],w[2],w[4]      # diameter A B center M
            if m not in p.names: p.pt(m)      # auto-declare the midpoint point
            p.add(lambda P,m=m,a=a,b=b: c_mid(P,m,a,b))
        elif op == "oncircle":           # oncircle P center O thru T
            pp,o,t = w[1],w[3],w[5]; p.add(lambda P,pp=pp,o=o,t=t: c_eqdist(P,pp,o,o,t))
        elif op == "construct":          # construct P far A from B thru T
            name,_,a,_,b,_,t = w[1],w[2],w[3],w[4],w[5],w[6],w[7]
            constructs.append((name,a,b,t))
        elif op == "find":
            find = w[1:]                 # ['angle',A,B,C] or ['length',A,B]
    def build_constructs(P):
        Q=dict(P)
        for name,a,b,t in constructs:
            A,B,T = Q[a],Q[b],Q[t]
            d = A-B; d = d/np.linalg.norm(d)          # away-from-B direction
            Q[name] = A + np.linalg.norm(A-T)*d
        return Q
    if find[0]=="angle":
        a,b,c = find[1:4]
        q = lambda P: math.degrees(angle(build_constructs(P), a,b,c)); kind="angle"
    else:
        a,b = find[1:3]
        q = lambda P: math.dist(build_constructs(P)[a], build_constructs(P)[b]); kind="length"
    return p, q, kind

PROBLEMS = {
"problem3 (AB)": """
  fix B 0 0
  fix D 4 0
  fix C 7 0
  point A at 4 3
  perp A D B C
  angle A B D = 2 angle D A C
  find length A B
""",
"problem2 (semicircle x)": """
  fix O 0 0
  fix X1 1 0
  point Rt at 16 0
  point P at -8 14
  point M at 8 14
  point foot at 8 0
  collinear O X1 Rt
  collinear O X1 foot
  perp M foot O X1
  dist foot Rt = 8
  oncircle P center O thru Rt
  oncircle M center O thru Rt
  angle Rt O P = 120
  angle Rt O M = 60
  find length P M
""",
"problem1 (angle DBE)": """
  fix B 0 0
  fix C 1 0
  point A
  angle A B C = 40
  midpoint O A B
  midpoint Q B C
  construct D far O from Q thru B
  construct E far Q from O thru B
  find angle D B E
""",
}

if __name__=="__main__":
    for name,spec in PROBLEMS.items():
        p,q,kind = parse(spec)
        r = solve(p,q,kind)
        print(f"{name:24} -> {r['exact'] if r['ok'] else 'REFUSE: '+r['reason']}")
