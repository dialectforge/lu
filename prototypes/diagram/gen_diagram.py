#!/usr/bin/env python3
"""Synthetic LABELED geometry diagrams with ground truth (triangle + cevian +
right-angle mark + letter labels). For measuring the diagram-fact extractor."""
import cv2, numpy as np, json, os, math, random
OUT=os.path.dirname(os.path.abspath(__file__))

def label_pos(p, centroid, off=26):
    v=np.array(p,float)-centroid; n=np.linalg.norm(v); 
    return tuple((np.array(p,float)+ (v/n)*off).astype(int))

def right_angle_mark(img, V, A, B, s=18, col=(20,120,20), w=2):
    # square mark at V in the corner between rays V->A and V->B
    va=np.array(A,float)-V; vb=np.array(B,float)-V
    va/=np.linalg.norm(va); vb/=np.linalg.norm(vb)
    p1=V+va*s; p2=V+vb*s; p3=V+(va+vb)*s
    for a,b in [(V+va*s,p3),(V+vb*s,p3)]:
        cv2.line(img,tuple(a.astype(int)),tuple(b.astype(int)),col,w)

def gen(i, seed):
    random.seed(seed); np.random.seed(seed)
    W=H=520; img=np.full((H,W,3),(252,250,247),np.uint8)
    Bx=random.randint(70,160); By=random.randint(360,440)
    Cx=random.randint(360,460); Cy=By+random.randint(-25,25)
    Ax=random.randint(180,330); Ay=random.randint(70,170)
    A=np.array([Ax,Ay]); B=np.array([Bx,By]); C=np.array([Cx,Cy])
    f=random.uniform(0.45,0.62); D=B+f*(C-B)
    pts={"A":A,"B":B,"C":C,"D":D.astype(int)}
    cen=np.mean([A,B,C],0)
    col=(40,40,40); lw=2
    for P,Qn in [("A","B"),("B","C"),("A","C")]:
        cv2.line(img,tuple(pts[P]),tuple(pts[Qn]),col,lw)
    cv2.line(img,tuple(A),tuple(pts["D"]),(90,90,90),lw)          # cevian
    right_angle_mark(img, pts["D"].astype(float), A.astype(float), C.astype(float))
    for nm,P in pts.items():
        lp=label_pos(P,cen); cv2.putText(img,nm,lp,cv2.FONT_HERSHEY_SIMPLEX,0.7,(180,60,60),2,cv2.LINE_AA)
        cv2.circle(img,tuple(P),3,(0,0,180),-1)
    if seed%2: img=np.clip(img.astype(int)+np.random.randint(-5,6,img.shape),0,255).astype(np.uint8)
    path=os.path.join(OUT,f"diag_{i:02d}.png"); cv2.imwrite(path,img)
    gt=dict(path=os.path.basename(path),
            points={k:[int(v[0]),int(v[1])] for k,v in pts.items()},
            segments=[["A","B"],["B","C"],["A","C"],["A","D"]],
            on_segment=["D","B","C"], perp=["A","D","B","C"])
    return gt

samples=[gen(i, 10+i) for i in range(4)]
json.dump(samples, open(os.path.join(OUT,"diagram_gt.json"),"w"), indent=2)
print(f"generated {len(samples)} labeled diagrams")
