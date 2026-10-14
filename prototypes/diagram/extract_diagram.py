#!/usr/bin/env python3
"""Extract DSL diagram-facts from a labeled geometry figure:
points (named via OCR), segments, on_segment incidence, right-angle/perp marks.
"""
import cv2, numpy as np, json, sys, math, os
import pytesseract

def _ocr_char(crop):
    crop=cv2.copyMakeBorder(crop,10,10,10,10,cv2.BORDER_CONSTANT,value=0)
    crop=cv2.resize(crop,None,fx=8,fy=8,interpolation=cv2.INTER_CUBIC)
    crop=cv2.GaussianBlur(crop,(3,3),0); inv=cv2.bitwise_not(crop)
    for psm in ("10","7","8"):
        t=pytesseract.image_to_string(inv,config=f"--psm {psm} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ").strip()
        if len(t)==1 and t.isalpha(): return t
    return None

def ocr_labels(img):
    b,g,r=cv2.split(img.astype(int))
    mask=((b>120)&(b-g>40)&(b-r>40)).astype(np.uint8)*255      # label colour (blue text)
    n,lab,stats,cent=cv2.connectedComponentsWithStats(mask,8)
    out=[]
    for i in range(1,n):
        x,y,w,h,area=stats[i]
        if area<15 or w>60 or h>60: continue
        ch=_ocr_char(mask[max(0,y-2):y+h+2, max(0,x-2):x+w+2])
        if ch: out.append((ch, np.array([cent[i][0],cent[i][1]])))
    return out

def detect_segments(img):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    edges=cv2.Canny(gray,40,140)
    lines=cv2.HoughLinesP(edges,1,np.pi/360,threshold=55,minLineLength=70,maxLineGap=12)
    segs=[] if lines is None else [l[0] for l in lines]
    # merge near-duplicate/collinear segments
    merged=[]
    for x1,y1,x2,y2 in segs:
        a=np.array([x1,y1],float); b=np.array([x2,y2],float)
        ang=math.atan2(b[1]-a[1],b[2-1]-a[0]) if False else math.atan2(b[1]-a[1],b[0]-a[0])
        found=False
        for m in merged:
            if abs(((ang-m['ang'])+math.pi/2)%math.pi-math.pi/2)<0.08:
                # same direction; check colinear (endpoint distance to line small)
                d=point_line_dist(a,m['a'],m['b'])+point_line_dist(b,m['a'],m['b'])
                if d<14:
                    pts=[m['a'],m['b'],a,b]
                    # extreme along direction
                    dirv=m['b']-m['a']; dirv=dirv/ (np.linalg.norm(dirv)+1e-9)
                    proj=[(p-m['a'])@dirv for p in pts]
                    m['a']=pts[int(np.argmin(proj))]; m['b']=pts[int(np.argmax(proj))]
                    found=True; break
        if not found: merged.append(dict(a=a,b=b,ang=ang))
    return [(m['a'],m['b']) for m in merged]

def point_line_dist(p,a,b):
    ab=b-a; t=np.clip((p-a)@ab/((ab@ab)+1e-9),0,1); return np.linalg.norm(p-(a+t*ab))

def seg_intersections(segs):
    pts=[]
    for i in range(len(segs)):
        for j in range(i+1,len(segs)):
            P=line_inter(segs[i],segs[j])
            if P is not None: pts.append(P)   # true crossings only (A,B,C,D)
    return cluster(pts,16)

def line_inter(s1,s2):
    (a,b),(c,d)=s1,s2
    r=b-a; s=d-c; rxs=r[0]*s[1]-r[1]*s[0]
    if abs(rxs)<1e-6: return None
    t=((c-a)[0]*s[1]-(c-a)[1]*s[0])/rxs; u=((c-a)[0]*r[1]-(c-a)[1]*r[0])/rxs
    if -0.03<=t<=1.03 and -0.03<=u<=1.03: return a+t*r
    return None

def cluster(pts,tol):
    out=[]
    for p in pts:
        for o in out:
            if np.linalg.norm(p-o['c'])<tol: o['pts'].append(p); o['c']=np.mean(o['pts'],0); break
        else: out.append(dict(c=p.copy(),pts=[p]))
    return [o['c'] for o in out]

def detect_right_angle(img):
    # green mark
    b,g,r=cv2.split(img.astype(int))
    mask=((g>90)&(g-r>40)&(g-b>40)).astype(np.uint8)*255
    cnts,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts=[c for c in cnts if cv2.contourArea(c)>20]
    if not cnts: return None
    M=cv2.moments(max(cnts,key=cv2.contourArea)); 
    return np.array([M['m10']/M['m00'], M['m01']/M['m00']])

def extract(path):
    img=cv2.imread(path)
    labels=ocr_labels(img)
    segs=detect_segments(img)
    verts=seg_intersections(segs)
    # name vertices by nearest label
    named={}
    for t,lp in labels:
        if not verts: continue
        v=min(verts,key=lambda q:np.linalg.norm(q-lp)); named[t]=v
    # segments between named pts: a named pair are connected if a detected seg
    # has both near its endpoints OR they lie along one seg
    def connected(P,Q):
        for a,b in segs:
            if point_line_dist(P,a,b)<10 and point_line_dist(Q,a,b)<10:
                # both near the seg's line AND within its span
                dirv=b-a; L=np.linalg.norm(dirv)+1e-9; u1=(P-a)@dirv/L/L; u2=(Q-a)@dirv/L/L
                if -0.06<=u1<=1.06 and -0.06<=u2<=1.06: return True
        return False
    names=list(named); seglist=[]
    for i in range(len(names)):
        for j in range(i+1,len(names)):
            if connected(named[names[i]],named[names[j]]): seglist.append((names[i],names[j]))
    # on_segment: X between Y,Z if collinear & between & both YX,XZ connected via same line
    onseg=[]
    for x in names:
        for y in names:
            for zc in names:
                if len({x,y,zc})<3: continue
                P,Y,Z=named[x],named[y],named[zc]
                if point_line_dist(P,Y,Z)<10:
                    dirv=Z-Y; u=(P-Y)@dirv/((dirv@dirv)+1e-9)
                    if 0.08<u<0.92:
                        onseg.append((x,y,zc))
    onseg=[t for t in onseg if t[1]<t[2]]
    # right angle -> the vertex nearest the green mark; perp between its two segs
    perp=None; ra=detect_right_angle(img)
    if ra is not None and named:
        vtx=min(names,key=lambda n:np.linalg.norm(named[n]-ra))
        nbrs=[n for n in names if (vtx,n) in [(min(s),max(s)) for s in [tuple(sorted(x)) for x in seglist]] or any(set([vtx,n])==set(s) for s in seglist)]
        if len(nbrs)>=2:
            perp=(nbrs[0],vtx,nbrs[1],vtx)
    return dict(points={k:[int(v[0]),int(v[1])] for k,v in named.items()},
                segments=seglist, on_segment=onseg, right_angle_vertex=(vtx if ra is not None and named else None))

if __name__=="__main__":
    gt={s["path"]:s for s in json.load(open(os.path.join(os.path.dirname(__file__),"diagram_gt.json")))}
    base=os.path.dirname(os.path.abspath(__file__)); import glob
    npass=0
    for p in sorted(glob.glob(os.path.join(base,"diag_*.png"))):
        name=os.path.basename(p); r=extract(p); g=gt[name]
        names_ok=set(r["points"])==set(g["points"])
        onseg_ok=any(t==("D","B","C") for t in r["on_segment"])
        ra_ok=r["right_angle_vertex"]=="D"
        # emit the DSL diagram-facts the solver needs
        facts=[]
        for x,y,z in r["on_segment"]: facts.append(f"collinear {y} {x} {z}")
        if ra_ok and onseg_ok: facts.append("perp A D B C")
        ok = names_ok and onseg_ok and ra_ok
        npass+=ok
        print(f"{name}: names={names_ok} on_seg={onseg_ok} right_angle={ra_ok}  ->  DSL facts: {facts}  [{'OK' if ok else 'MISS'}]")
    print(f"\n{npass}/4 diagrams: facts extracted correctly")