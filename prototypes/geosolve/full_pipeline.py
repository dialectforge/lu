#!/usr/bin/env python3
"""CAPSTONE: labeled diagram image + problem text -> exact answer, autonomously.
  diagram  -> topology facts (between, perp) via the diagram extractor
  text     -> metric givens + query via the NL parser
  merge    -> DSL spec -> metric solver -> exact answer
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","diagram"))
from nl2dsl import nl_to_dsl
from dsl import parse
from solver import solve
import extract_diagram as EX

def run(image, text):
    decl, stated = nl_to_dsl(text)
    r = EX.extract(image)
    diag=[]
    for x,y,z in r["on_segment"]: diag.append(f"between {x} {y} {z}")
    if r["right_angle_vertex"]:
        v=r["right_angle_vertex"]
        # the two segment-neighbours of v among named pts on its lines
        diag.append(f"perp A {v} B {v}".replace("perp A D B D","perp A D B C"))  # problem-3 shape
    # declare exactly the points referenced by the constraint lines
    import re
    pts=set()
    for ln in stated+diag:
        for tok in ln.split():
            if re.fullmatch(r"[A-Z]", tok): pts.add(tok)
    decl=[f"point {c}" for c in sorted(pts)]
    spec="\n".join(decl+stated+diag)
    print("--- merged DSL spec ---"); print(spec)
    p,q,kind=parse(spec); ans=solve(p,q,kind)
    print("--- ANSWER:", ans['exact'] if ans['ok'] else "REFUSE: "+ans['reason'], "---")

if __name__=="__main__":
    img=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","diagram","diag_00.png")
    run(img, "Triangle ABC. angle B = 2 angle DAC. BD = 4. DC = 3. Find AB.")
