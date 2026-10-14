#!/usr/bin/env python3
"""Synthetic shaded-area puzzle generator with GROUND TRUTH.

Each sample: a bounding square of side S units containing circles. The square is
the calibration ruler (side = S units). We render with random style (scale,
offset, rotation, colors, line width, JPEG noise) and record the exact
primitives in UNIT coordinates so perception accuracy is measurable.
"""
import cv2, numpy as np, json, math, os, random

OUT = os.path.dirname(os.path.abspath(__file__))
random.seed(7); np.random.seed(7)

# figure templates: (name, S, [ (cx,cy,r) ... ], shading_rule)
# shading_rule(gx,gy, disks) -> bool mask of the shaded region in unit space
def t_quarter(S, disks):
    d = disks[0]
    inside_sq = True
    inside_disk = (gx-d[0])**2+(gy-d[1])**2 <= d[2]**2
    return ~inside_disk          # square minus quarter disk
def t_inscribed(S, disks):
    d = disks[0]
    return (gx-d[0])**2+(gy-d[1])**2 > d[2]**2   # corners
def t_leaf(S, disks):
    a,b = disks
    ina = (gx-a[0])**2+(gy-a[1])**2 <= a[2]**2
    inb = (gx-b[0])**2+(gy-b[1])**2 <= b[2]**2
    return ina & inb             # lens
def t_semi(S, disks):
    d = disks[0]
    return (gx-d[0])**2+(gy-d[1])**2 <= d[2]**2

TEMPLATES = [
    ("quarter",   8, [(0,0,8)],            t_quarter),
    ("inscribed", 10,[(5,5,5)],            t_inscribed),
    ("leaf",      6, [(0,0,6),(6,6,6)],    t_leaf),
    ("semi",      8, [(4,0,4)],            t_semi),
    ("inscribed2",12,[(6,6,6)],            t_inscribed),
    ("leaf2",     10,[(0,0,10),(10,10,10)],t_leaf),
]

gx=gy=None
def render(name, S, disks, rule, style, path):
    global gx, gy
    res = style["res"]; ppu = res/ (S* style["margin"])
    side_px = int(S*ppu)
    ox = int((res-side_px)*style["ox"]); oy = int((res-side_px)*style["oy"])
    # unit grid for shading mask, sampled per pixel inside the square
    img = np.full((res,res,3), style["bg"], np.uint8)
    # build shaded mask in pixel space
    ys,xs = np.mgrid[0:side_px,0:side_px]
    gx = xs/ppu; gy = (side_px-1-ys)/ppu       # unit coords (y up)
    mask = rule(S, disks)
    sub = img[oy:oy+side_px, ox:ox+side_px]
    sub[mask] = style["shade"]
    img[oy:oy+side_px, ox:ox+side_px] = sub
    lw = style["lw"]
    # square border
    cv2.rectangle(img,(ox,oy),(ox+side_px,oy+side_px),style["line"],lw)
    # circle outlines (clip to square via mask of ring)
    for (cx,cy,r) in disks:
        c_px=(int(ox+cx*ppu), int(oy+(S-cy)*ppu)); r_px=int(r*ppu)
        cv2.circle(img,c_px,r_px,style["line"],lw)
    # crop back to square region only (cv2.circle drew outside too); keep full, then mask outside square white
    outside = np.ones((res,res),bool); outside[oy:oy+side_px,ox:ox+side_px]=False
    img[outside]=style["bg"]
    cv2.rectangle(img,(ox,oy),(ox+side_px,oy+side_px),style["line"],lw)
    # rotation
    if style["rot"]!=0:
        M=cv2.getRotationMatrix2D((res/2,res/2),style["rot"],1.0)
        img=cv2.warpAffine(img,M,(res,res),borderValue=style["bg"])
    # noise + jpeg
    if style["noise"]>0:
        img=np.clip(img.astype(int)+np.random.randint(-style["noise"],style["noise"]+1,img.shape),0,255).astype(np.uint8)
    cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, style["jpeg"]])
    gt={"S":S,"circles":[[cx,cy,r] for (cx,cy,r) in disks],"square_units":S,"template":name}
    return gt

STYLES=[
  dict(res=512,margin=1.25,ox=0.4,oy=0.5,bg=(255,255,255),shade=(60,60,220),line=(20,20,20),lw=2,rot=0,noise=0,jpeg=95),
  dict(res=600,margin=1.4, ox=0.3,oy=0.6,bg=(250,248,245),shade=(40,170,40),line=(30,30,30),lw=3,rot=3,noise=6,jpeg=85),
  dict(res=480,margin=1.2, ox=0.5,oy=0.4,bg=(255,252,250),shade=(200,80,40),line=(10,10,10),lw=2,rot=-2,noise=4,jpeg=80),
  dict(res=560,margin=1.5, ox=0.6,oy=0.5,bg=(245,245,255),shade=(120,60,200),line=(25,25,25),lw=4,rot=1,noise=8,jpeg=90),
]

samples=[]
for i,(name,S,disks,rule) in enumerate(TEMPLATES):
    st=STYLES[i%len(STYLES)]
    p=os.path.join(OUT,f"syn_{i:02d}_{name}.jpg")
    gt=render(name,S,disks,rule,st,p)
    gt["path"]=os.path.basename(p)
    samples.append(gt)
json.dump(samples, open(os.path.join(OUT,"ground_truth.json"),"w"), indent=2)
print(f"generated {len(samples)} samples")
for s in samples: print(" ", s["path"], "S=",s["S"],"circles=",s["circles"])
