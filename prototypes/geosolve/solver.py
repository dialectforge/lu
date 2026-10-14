#!/usr/bin/env python3
"""Metric/constraint geometry solver (angle & length problems).

Model a figure as points + exact constraints (distance, angle, perpendicular,
collinear, midpoint, on-circle/diameter). Place it numerically from random
seeds (Levenberg-Marquardt), read the queried length/angle, and RECOVER the
exact closed form. K-GATE: the answer must be invariant across several random
valid placements; if it varies, the figure is under-determined -> REFUSE.
"""
import numpy as np, math
from fractions import Fraction as F

# ----- constraint residuals (each returns a scalar that should be 0) -----
def c_dist(P, a, b, d):       return math.dist(P[a], P[b]) - d
def c_perp(P, a, b, c, d):
    ux,uy=P[b]-P[a]; vx,vy=P[d]-P[c]; return ux*vx+uy*vy
def c_coll(P, a, b, c):
    ux,uy=P[b]-P[a]; vx,vy=P[c]-P[a]; return ux*vy-uy*vx
def c_mid(P, m, a, b):        return list((P[m]-(P[a]+P[b])/2))
def c_oncircle(P, p, ctr, r): return math.dist(P[p], P[ctr]) - r
def c_eqdist(P, a, b, c, d):  return math.dist(P[a],P[b]) - math.dist(P[c],P[d])
def c_between(P, x, y, z):     return math.dist(P[y],P[x])+math.dist(P[x],P[z]) - math.dist(P[y],P[z])

def angle(P, a, b, c):
    """angle ABC in radians (at vertex b)."""
    u=P[a]-P[b]; v=P[c]-P[b]
    cu=math.atan2(u[1],u[0]); cv=math.atan2(v[1],v[0])
    d=abs(cu-cv)%(2*math.pi)
    return d if d<=math.pi else 2*math.pi-d
def c_angle(P, a, b, c, deg): return angle(P,a,b,c) - math.radians(deg)
def c_angle_eq(P, a,b,c, d,e,f, k=1.0):  # angle(abc) = k*angle(def)
    return angle(P,a,b,c) - k*angle(P,d,e,f)

class Problem:
    def __init__(self):
        self.names=[]; self.fixed={}; self.cons=[]; self.inits={}
    def pt(self, name, fx=None, fy=None, init=None):
        self.names.append(name)
        if fx is not None: self.fixed[name]=(fx,fy)
        if init is not None: self.inits[name]=init
        return name
    def add(self, fn): self.cons.append(fn)

    def _pack(self):
        self.free=[n for n in self.names if n not in self.fixed]
        return self.free
    def _coords(self, x):
        P={}
        for n in self.names:
            if n in self.fixed: P[n]=np.array(self.fixed[n],float)
        for i,n in enumerate(self.free):
            P[n]=np.array([x[2*i],x[2*i+1]],float)
        return P
    def residuals(self, x):
        P=self._coords(x); r=[]
        for fn in self.cons:
            v=fn(P)
            if isinstance(v,list): r+=v
            else: r.append(v)
        return np.array(r,float)

    def solve_once(self, seed):
        rng=np.random.default_rng(seed); self._pack()
        x=np.zeros(2*len(self.free))
        for i,n in enumerate(self.free):
            if n in self.inits: x[2*i],x[2*i+1]=self.inits[n]
            else: x[2*i],x[2*i+1]=rng.uniform(-3,3,2)
        lam=1e-3
        for _ in range(200):
            r=self.residuals(x); cost=r@r
            if cost<1e-22: break
            J=np.zeros((len(r),len(x)))
            for k in range(len(x)):
                h=1e-6; xp=x.copy(); xp[k]+=h; xm=x.copy(); xm[k]-=h
                J[:,k]=(self.residuals(xp)-self.residuals(xm))/(2*h)
            A=J.T@J+lam*np.eye(len(x)); g=J.T@r
            try: step=np.linalg.solve(A,g)
            except np.linalg.LinAlgError: break
            xn=x-step; rn=self.residuals(xn)
            if rn@rn<cost: x=xn; lam=max(lam*0.5,1e-9)
            else: lam=min(lam*3,1e6)
        return self._coords(x), self.residuals(x)@self.residuals(x)

# ----- exact recovery -----
# Geometry-puzzle answers are SMALL-denominator: only accept those, verified
# tight. A large denominator means we are fitting noise -> refuse (no fabrication).
def recover_length(v, maxden=64, tol=1e-7):
    fr=F(v).limit_denominator(maxden)
    if abs(float(fr)-v)<tol: return f"{fr}"
    f2=F(v*v).limit_denominator(maxden)
    if abs(float(f2)-v*v)<tol and f2!=0:
        if f2.denominator==1: return f"sqrt({f2.numerator})"
        return f"sqrt({f2.numerator}/{f2.denominator})"
    return None
def recover_angle(deg, maxden=24, tol=1e-6):
    fr=F(deg).limit_denominator(maxden)   # whole or simple fractional degrees
    if abs(float(fr)-deg)<tol: return f"{fr} deg"
    return None

def solve(problem, query, kind, runs=5, seeds=range(100,160)):
    vals=[]; sols=[]
    for s in seeds:
        P,cost=problem.solve_once(s)
        if cost<1e-12:
            vals.append(query(P)); sols.append(P)
        if len(vals)>=runs: break
    if len(vals)<runs:
        return {"ok":False,"reason":f"only {len(vals)} valid placements"}
    spread=max(vals)-min(vals)
    if spread>1e-6:
        return {"ok":False,"reason":f"under-determined (answer varies by {spread:.2e})"}
    v=float(np.mean(vals))
    exact = recover_angle(v) if kind=="angle" else recover_length(v)
    if exact is None:
        return {"ok":False,"reason":f"no clean closed form for {v:.6f}"}
    return {"ok":True,"value":v,"exact":exact,"runs":len(vals)}

# ============ PROBLEM 3: find AB ============
def problem3():
    p=Problem()
    p.pt("B",0,0); p.pt("D",4,0); p.pt("C",7,0)   # BD=4, DC=3, AD perpendicular
    p.pt("A", init=(4,3))                          # A free, hint above D
    p.add(lambda P: c_perp(P,"A","D","B","C"))                    # AD perpendicular to BC
    p.add(lambda P: c_angle_eq(P,"A","B","D", "D","A","C", 2.0))  # angle B = 2*angle DAC
    return p, lambda P: math.dist(P["A"],P["B"])

# ============ PROBLEM 2: semicircle, find x ============
def problem2():
    p=Problem()
    p.pt("O",0,0); p.pt("X1",1,0)                 # O = centre, X1 fixes the x-axis
    p.pt("Rt", init=(16,0)); p.pt("P",init=(-8,14)); p.pt("M",init=(8,14)); p.pt("foot",init=(8,0))
    p.add(lambda P: c_coll(P,"O","X1","Rt"))      # right end on the diameter
    p.add(lambda P: P["Rt"][1])                    # Rt.y = 0
    p.add(lambda P: c_eqdist(P,"O","P","O","Rt"))  # P on the circle
    p.add(lambda P: c_eqdist(P,"O","M","O","Rt"))  # M on the circle
    p.add(lambda P: c_angle(P,"Rt","O","P",120))   # left ray 60deg from diameter
    p.add(lambda P: c_angle(P,"Rt","O","M",60))    # right ray 60deg from diameter
    p.add(lambda P: c_coll(P,"O","X1","foot"))     # foot on the diameter
    p.add(lambda P: c_perp(P,"M","foot","O","X1")) # M-foot vertical
    p.add(lambda P: c_dist(P,"foot","Rt",8))       # the given length 8
    return p, lambda P: math.dist(P["P"],P["M"])

# ============ PROBLEM 1: find angle DBE ============
def problem1():
    # Place only A,B,C with BA length left FREE (the gauge DOF the answer must be
    # invariant to). O,Q (midpoints / circle centres) and D,E (the far
    # intersections of line OQ with each circle) are derived in the query, which
    # removes the two-intersection ambiguity.
    p=Problem()
    p.pt("B",0,0); p.pt("C",1,0); p.pt("A")        # A: random init, angle constraint only
    p.add(lambda P: c_angle(P,"A","B","C",40))
    def q(P):
        A,B,C=P["A"],P["B"],P["C"]
        O=(A+B)/2; Q=(B+C)/2
        u=Q-O; u=u/np.linalg.norm(u)
        rO=np.linalg.norm(O-B); rQ=np.linalg.norm(Q-B)
        D=O-rO*u; E=Q+rQ*u                         # far intersections
        return math.degrees(angle({"D":D,"B":B,"E":E},"D","B","E"))
    return p, q

if __name__=="__main__":
    for name,(p,q),kind in [("3 find AB",problem3(),"length"),
                            ("2 semicircle x",problem2(),"length"),
                            ("1 angle DBE",problem1(),"angle")]:
        print(f"PROBLEM {name}:", solve(p,q,kind))
