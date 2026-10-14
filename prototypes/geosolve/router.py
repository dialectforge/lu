#!/usr/bin/env python3
"""Solver-type router: decide from the PROMPT which engine to use.

Non-LLM, feature/keyword based. Lu has several exact engines; the wrong one
gives no answer, so the router must (a) detect geometry, (b) within geometry
split shaded-area (decomposition / Green) from metric (angle/length constraints),
and (c) REFUSE/clarify when the target is ambiguous rather than guess.
Mirrors the discipline of the existing query_router; intended to fold into it.
"""
import re

GEO_CTX = ["triangle","circle","square","rectangle","semicircle","arc","sector",
           "diameter","radius","tangent","inscribed","circumscrib","chord","polygon",
           "quadrilateral","hexagon","pentagon","vertex","vertices","angle","perimeter",
           "shaded","circumference","parallelogram","trapezoid","rhombus","altitude"]
ANGLE_Q = [r"(find|measure of|what is|compute)( the)? angle", r"∠", r"how many degrees"]
LEN_Q   = [r"\blength\b", r"\bperimeter\b", r"\bside\b", r"\bdistance\b", r"\bradius\b",
           r"\bhow long\b", r"\bfind x\b", r"\bfind\s+[A-Z]{2,3}\b", r"\bvalue of x\b"]
AREA_Q  = [r"\barea\b"]
ARITH   = re.compile(r"^[\s0-9+\-*/^().,x=]+$")

def _any(pats, t): return any(re.search(p, t) for p in pats)

def classify(prompt):
    t = prompt.strip(); tl = t.lower()
    # 1. arithmetic: bare expression OR word-form "N plus/times/... N"
    expr = re.sub(r"^(what('s| is)|calculate|compute|evaluate)\s+", "", tl).strip(" ?")
    word_arith = re.search(r"\b\d+\s+(plus|minus|times|divided by|multiplied by)\s+\d+", tl)
    if (ARITH.match(expr) and re.search(r"[-+*/^]", expr)) or word_arith:
        return dict(solver="arithmetic", subtype=None, confidence=0.95,
                    reason="arithmetic expression")
    shaded = "shaded" in tl
    a_q = _any(ANGLE_Q, tl); area_q = _any(AREA_Q, tl)
    # length target: keyword OR find+segment-name (original case) OR find x
    l_q = _any(LEN_Q, tl) or bool(re.search(r"(?i:find)\s+[A-Z]{2,3}\b", t)) \
          or bool(re.search(r"\b(find|value of)\s+x\b", tl))
    strong_target = a_q or l_q or shaded
    geo = sum(w in tl for w in GEO_CTX)
    if geo == 0 and not strong_target:
        return dict(solver="recall", subtype=None, confidence=0.6,
                    reason="no geometry signal -> associative recall, not a solver")
    # 2. within geometry: pick by the QUESTION TARGET
    if shaded and area_q:
        return dict(solver="shaded_area", subtype="decomposition", confidence=0.95,
                    reason="'shaded' + 'area' -> Phase-3 arrangement/Green engine")
    if a_q and not shaded:
        return dict(solver="metric_geometry", subtype="angle", confidence=0.9,
                    reason="angle target -> constraint solver")
    if l_q and not shaded:
        return dict(solver="metric_geometry", subtype="length", confidence=0.88,
                    reason="length/side target -> constraint solver")
    if area_q and not shaded:
        return dict(solver="metric_geometry", subtype="area", confidence=0.75,
                    reason="area of a dimensioned shape -> formula/constraint (not shaded decomposition)")
    if shaded:
        return dict(solver="shaded_area", subtype="decomposition", confidence=0.8,
                    reason="'shaded' present, area implied")
    return dict(solver="clarify", subtype=None, confidence=0.3,
                reason="geometry detected but the target (area? angle? length?) is unclear")

TESTS = [
 ("Find the area of the shaded region","shaded_area"),
 ("what is the shaded area inside the square","shaded_area"),
 ("the leaf-shaped shaded region area","shaded_area"),
 ("Find angle DBE","metric_geometry"),
 ("find the measure of the angle ABC in degrees","metric_geometry"),
 ("Find AB","metric_geometry"),
 ("find the length x in the semicircle","metric_geometry"),
 ("calculate the perimeter of the triangle","metric_geometry"),
 ("find the radius of the inscribed circle","metric_geometry"),
 ("what is the area of a triangle with sides 3 4 5","metric_geometry"),
 ("what is 7 plus 5","arithmetic"),
 ("12 * (3 + 4)","arithmetic"),
 ("who is the president of france","recall"),
 ("a circle and a triangle","clarify"),
]
if __name__=="__main__":
    ok=0
    for p,exp in TESTS:
        r=classify(p); good=r["solver"]==exp
        ok+=good
        print(f"[{'OK ' if good else 'XX '}] {p[:42]:42} -> {r['solver']:16} ({r['subtype']}) {r['reason']}")
    print(f"\n{ok}/{len(TESTS)} routed as expected")
