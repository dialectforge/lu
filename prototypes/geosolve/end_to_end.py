#!/usr/bin/env python3
"""End-to-end: prompt -> router picks the solver -> exact answer (or refusal)."""
from router import classify
from solver import problem1, problem2, problem3, solve

MODELS = {  # (hand-built constraint models for the 3 sample prompts)
 "Find angle DBE given triangle ABC, AB and BC diameters, angle ABC=40": (problem1,"angle"),
 "Find AB given triangle ABC, angle B = 2 angle DAC, BD=4, DC=3":        (problem3,"length"),
 "Find the length x in the semicircle with two 60 degree rays and 8":    (problem2,"length"),
}
for prompt,(mk,kind) in MODELS.items():
    r=classify(prompt)
    print(f"PROMPT: {prompt}")
    print(f"  router -> {r['solver']} ({r['subtype']})  [{r['reason']}]")
    if r["solver"]=="metric_geometry":
        p,q=mk(); ans=solve(p,q,kind)
        print(f"  metric solver -> {ans.get('exact') if ans['ok'] else 'REFUSE: '+ans['reason']}")
    print()
# a shaded-area prompt routes to the OTHER engine
r=classify("Find the area of the shaded region in the 12x12 square with arcs")
print("PROMPT: Find the area of the shaded region ...")
print(f"  router -> {r['solver']} ({r['subtype']})  [feeds the Phase-3 arrangement engine -> e.g. 32*pi-64]")
