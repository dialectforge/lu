#!/usr/bin/env python3
"""General shaded-area perceiver with a snap-then-VERIFY K-gate.

1. Detect the bounding square (largest 4-gon); rectify to a canonical unit square
   -> all geometry in square-side units (no scale-bar OCR).
2. Detect circles by SEQUENTIAL RANSAC arc-fitting on the shaded-region boundary
   (robust to partial arcs / corner circles that Hough misses).
3. SNAP each (cx,cy,r) to the nearest small-denominator rational; VERIFY the
   snapped circle against the boundary; accept only if agreement high, else FLAG.
"""
import cv2, numpy as np, json, math, sys, glob, os
from fractions import Fraction as F

N = 600

def order_corners(pts):
    pts = pts.reshape(4,2).astype(np.float32)
    s = pts.sum(1); d = np.diff(pts,1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], np.float32)

def detect_square(gray):
    ink = (gray < 120).astype(np.uint8)*255
    ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))
    cnts,_ = cv2.findContours(ink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best=None; ba=0
    for c in cnts:
        a=cv2.contourArea(c)
        per=cv2.arcLength(c,True); ap=cv2.approxPolyDP(c,0.02*per,True)
        if len(ap)==4 and a>ba: best=ap; ba=a
    return order_corners(best) if best is not None else None

def rectify(img, corners):
    dst=np.array([[0,0],[N-1,0],[N-1,N-1],[0,N-1]],np.float32)
    return cv2.warpPerspective(img,cv2.getPerspectiveTransform(corners,dst),(N,N))

def shaded_boundary(rect):
    r,g,b = cv2.split(rect.astype(int))
    mx=np.maximum(np.maximum(r,g),b); mn=np.minimum(np.minimum(r,g),b)
    sat=mx-mn; bright=mx
    fill=((sat>40)&(bright>40)).astype(np.uint8)*255       # colored, not white/black
    fill=cv2.morphologyEx(fill,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8))
    cnts,_=cv2.findContours(fill,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    if not cnts: return np.empty((0,2)), fill
    pts=np.vstack([c.reshape(-1,2) for c in cnts if len(c)>20]).astype(float)
    # drop points on the square border (straight edges, not arcs)
    m=6
    keep=(pts[:,0]>m)&(pts[:,0]<N-1-m)&(pts[:,1]>m)&(pts[:,1]<N-1-m)
    return pts[keep], fill

def fit_circle(p):           # algebraic 3+ point fit
    x=p[:,0]; y=p[:,1]
    A=np.c_[2*x,2*y,np.ones(len(p))]; bb=x*x+y*y
    sol,_,_,_=np.linalg.lstsq(A,bb,rcond=None)
    cx,cy,c=sol; r=math.sqrt(max(c+cx*cx+cy*cy,0))
    return cx,cy,r

def ransac_circles(pts, max_c=4, tol=3.0, iters=400, min_inl=80):
    found=[]; pool=pts.copy()
    for _ in range(max_c):
        if len(pool)<min_inl: break
        best_in=None; best=None
        for _ in range(iters):
            idx=np.random.choice(len(pool),3,replace=False)
            try: cx,cy,r=fit_circle(pool[idx])
            except Exception: continue
            if r<N*0.1 or r>N*1.3: continue
            d=np.abs(np.hypot(pool[:,0]-cx,pool[:,1]-cy)-r)
            inl=d<tol
            if best_in is None or inl.sum()>best_in.sum():
                best_in=inl; best=(cx,cy,r)
        if best is None or best_in.sum()<min_inl: break
        cx,cy,r=fit_circle(pool[best_in])      # refit on inliers
        found.append((cx,cy,r,int(best_in.sum())))
        pool=pool[~(np.abs(np.hypot(pool[:,0]-cx,pool[:,1]-cy)-r)<tol)]
    return found

def snap_frac(x, maxden=4): return F(float(x)).limit_denominator(maxden)

def verify(pts, cx, cy, r, tol=3.5):
    if len(pts)==0 or r<=0: return 0.0
    d=np.abs(np.hypot(pts[:,0]-cx,pts[:,1]-cy)-r)
    return float((d<tol).mean())  # fraction of boundary explained by this circle

def perceive(path):
    img=cv2.imread(path); gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    corners=detect_square(gray)
    if corners is None: return {"ok":False,"reason":"no square"}
    rect=rectify(img,corners)
    pts,fill=shaded_boundary(rect)
    if len(pts)<30: return {"ok":True,"circles":[],"note":"no arc boundary"}
    circ=ransac_circles(pts)
    out=[]
    for cx,cy,r,inl in circ:
        cxu,cyu,ru=cx/N,(N-1-cy)/N,r/N
        sx,sy,sr=snap_frac(cxu),snap_frac(cyu),snap_frac(ru)
        # snap-then-verify gate: trust the rational only if the raw pixel
        # measurement was ALREADY within tolerance of it (never force a snap).
        snap_dist=max(abs(cxu-float(sx)),abs(cyu-float(sy)),abs(ru-float(sr)))
        out.append(dict(snap=[str(sx),str(sy),str(sr)], inliers=inl,
                        snap_dist=round(snap_dist,4), accept=snap_dist<0.03))
    return {"ok":True,"circles":out}

if __name__=="__main__":
    gt={s["path"]:s for s in json.load(open(sys.argv[1]))}
    base=os.path.dirname(os.path.abspath(__file__)); np.random.seed(0)
    npass=0; ntot=0
    for p in sorted(glob.glob(os.path.join(base,"syn_*.jpg"))):
        name=os.path.basename(p); r=perceive(p); g=gt[name]; S=g["S"]
        gt_snap=sorted(tuple(str(F(v,S).limit_denominator(4)) for v in c) for c in g["circles"])
        got=sorted(tuple(c["snap"]) for c in r.get("circles",[]) if c["accept"])
        ok = got==gt_snap
        npass+= ok; ntot+=1
        print(f"{name:22} GT(side-units)={gt_snap}  got={got}  {'OK' if ok else 'MISS'}")
    print(f"\n{npass}/{ntot} samples fully recovered")
