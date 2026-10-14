#!/usr/bin/env python3
"""Natural-language 'Given/Find' statement -> DSL lines (the stated half).

Handles the regular gogeometry-style phrasing. Diagram-only facts (a point on a
segment, a right-angle mark, a construction like 'D = line OQ meets circle O')
are NOT in the text -- they come from perception and are merged in separately.
"""
import re

def nl_to_dsl(text):
    t = text.replace("∠","angle ").replace("°","").replace("∆","triangle ")
    t = t.replace("=", " = ")
    lines=[]; pts=set(); tri=None
    def seg(s): return s[0],s[1]
    for raw in re.split(r"[.;\n]", t):
        s=raw.strip()
        if not s: continue
        m=re.search(r"[Tt]riangle\s+([A-Z])([A-Z])([A-Z])", s)
        if m:
            tri=m.group(1)+m.group(2)+m.group(3)
            for c in tri: pts.add(c)
            continue
        # angle X = K angle Y   (single-letter vertex -> triangle angle)
        m=re.search(r"angle\s+([A-Z])\s*=\s*([0-9]*)\s*angle\s+([A-Z])([A-Z])([A-Z])", s)
        if m and tri:
            vb=m.group(1); k=m.group(2) or "1"
            a,b,c=_vertex(vb,tri)
            lines.append(f"angle {a} {b} {c} = {k} angle {m.group(3)} {m.group(4)} {m.group(5)}")
            continue
        # angle XYZ = N
        m=re.search(r"angle\s+([A-Z])([A-Z])([A-Z])\s*=\s*([0-9.]+)", s)
        if m:
            lines.append(f"angle {m.group(1)} {m.group(2)} {m.group(3)} = {m.group(4)}"); continue
        # XY diameter of circle ... center O
        m=re.search(r"([A-Z])([A-Z]).*diameter.*center\s+([A-Z])", s)
        if m:
            lines.append(f"diameter {m.group(1)} {m.group(2)} center {m.group(3)}")
            pts.update([m.group(1),m.group(2)]); continue
        # XY = N  (segment length)
        m=re.search(r"\b([A-Z])([A-Z])\s*=\s*([0-9.]+)", s)
        if m:
            lines.append(f"dist {m.group(1)} {m.group(2)} = {m.group(3)}")
            pts.update([m.group(1),m.group(2)]); continue
        # Find ...
        m=re.search(r"[Ff]ind.*angle\s+([A-Z])([A-Z])([A-Z])", s)
        if m: lines.append(f"find angle {m.group(1)} {m.group(2)} {m.group(3)}"); continue
        m=re.search(r"[Ff]ind\s+([A-Z])([A-Z])", s)
        if m: lines.append(f"find length {m.group(1)} {m.group(2)}"); continue
    decl=[f"point {c}" for c in sorted(pts)]
    return decl, lines

def _vertex(v, tri):   # angle at vertex v of triangle tri -> (other1, v, other2)
    others=[c for c in tri if c!=v]
    return others[0], v, others[1]

# ---- demo: problems 1 and 3 from their statements + perception diagram facts ----
P1_TEXT = "Triangle ABC. angle ABC = 40. AB diameter of circle of center O. BC diameter of circle of center Q. Find angle DBE."
P1_DIAGRAM = ["construct D far O from Q thru B", "construct E far Q from O thru B"]

P3_TEXT = "Triangle ABC. angle B = 2 angle DAC. BD = 4. DC = 3. Find AB."
P3_DIAGRAM = ["collinear B D C", "perp A D B C", "point D at 4 3"]  # D on BC, AD perpendicular (right-angle mark)

if __name__=="__main__":
    from dsl import parse
    from solver import solve
    for name,(txt,diag) in {"problem1":(P1_TEXT,P1_DIAGRAM),"problem3":(P3_TEXT,P3_DIAGRAM)}.items():
        decl,stated = nl_to_dsl(txt)
        spec="\n".join(decl+stated+diag)
        print(f"=== {name} ===")
        print("NL-derived DSL:"); [print("   ",l) for l in stated]
        print("perception-supplied:"); [print("   ",l) for l in diag]
        p,q,kind=parse(spec); r=solve(p,q,kind)
        print("ANSWER:", r['exact'] if r['ok'] else "REFUSE: "+r['reason'], "\n")
