#!/usr/bin/env python3
"""OP-COMPARE -- entity-vs-entity comparison synthesis acceptance suite.

Guards the honest side-by-side comparison shape ("compare X and Y", "X vs Y",
"difference between X and Y", "how are X and Y different") composed ONLY from
the two entities' real typed edges: shared-first, then salient differences on
the SAME relation, with an honest not-comparable answer for different KINDS
(person vs place) and a clean degrade for thin / concept pairs. Every stated
attribute is a real cited edge -- never a fabricated contrast.

Drives the LIVE service (POST /auth, then POST /mcp lu4d_ask).
    sudo -E python3 scripts/comparison_acceptance.py [--port 8787]
Exit 0 = all passed. Non-zero = number of failures.
"""
import argparse, json, os, sys, urllib.request

def post(url, payload, token=None, timeout=180):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST")
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())

def extract_text(resp):
    r = resp.get("result", resp)
    content = r.get("content")
    if isinstance(content, list):
        parts = [c.get("text", "") for c in content if isinstance(c, dict)]
        if parts:
            return "\n".join(parts)
    for k in ("text", "answer", "narration"):
        if isinstance(r.get(k), str):
            return r[k]
    return json.dumps(r)

REFUSAL_MARKERS = [
    "don't have anything solid", "i don't have", "i do not have",
    "won't fabricate", "will not fabricate", "doesn't match anything",
    "nothing in my brain", "don't find", "can't find",
]

# (id, query, kind, arg, why)
PROBES = [
    # ---- RICH person pairs: shared type folds; cross-field stays honest -----
    ("CMP-curie-einstein", "compare Marie Curie and Albert Einstein", "contains_all",
     ["were both physicists", "chemistry", "einstein field equations",
      "warsaw in 1867", "ulm in 1879"],
     "Shared occupation folded ('were both physicists', past tense: both deceased); "
     "then per-side known-for and full lifespan -- all real BornOn/DiedOn/BornIn edges."),
    ("CMP-newton-darwin", "compare Isaac Newton and Charles Darwin", "contains_all",
     ["isaac newton was a mathematician", "charles darwin was a naturalist",
      "celestial mechanics", "on the origin of species",
      "woolsthorpe manor in 1642", "down house in 1882"],
     "Cross-field honest: NOT collapsed to a false shared type (Newton=mathematician, "
     "Darwin=naturalist in this brain); known-for + lifespan side by side."),
    ("CMP-newton-darwin-nofab", "compare Isaac Newton and Charles Darwin", "contains_none",
     ["both mathematicians", "both naturalists", "depends on what you value"],
     "No fabricated shared type; not the concept `comparative` fallthrough."),
    # ---- RICH place pairs: shared-first lead, then per-side diffs -----------
    ("CMP-france-germany", "compare France and Germany", "contains_all",
     ["both located in europe and use the euro", "capital is paris",
      "official language is french", "capital is berlin", "official language is german"],
     "Leads with SHARED (Europe + euro) in one grammatical sentence, then the "
     "capital/language differences per side. Currency folded as shared, not a diff."),
    ("CMP-france-germany-nojunk", "compare France and Germany", "contains_none",
     ["depends on what you value", "black players", "chicken on uniform"],
     "Replaces the old concept-comparative junk with grounded place facts."),
    ("CMP-japan-germany", "compare Japan and Germany", "contains_all",
     ["tokyo", "japanese", "berlin", "german", "japan uses the yen", "germany the euro"],
     "No shared type/located/currency -> no forced lead; per-side capital/language "
     "diff + an explicit DIFFERING-currency sentence (yen vs euro)."),
    # ---- connector coverage: vs / difference between / how are ... different -
    ("CMP-vs-form", "Marie Curie vs Charles Darwin", "contains_all",
     ["physicist", "naturalist", "warsaw in 1867", "the mount in 1809"],
     "Bare 'X vs Y' connector parses + composes the person comparison."),
    ("CMP-difference-between", "difference between France and Germany", "contains_all",
     ["both located in europe", "paris", "berlin"],
     "'difference between X and Y' routes entity pairs to comparison (before the "
     "relational link-between arm)."),
    ("CMP-how-different", "how are France and Germany different", "contains_all",
     ["paris", "berlin", "euro"],
     "'how are X and Y different' comparison sense (not the relational 'related')."),
    # ---- honest not-comparable across KINDS (person vs place) ---------------
    ("CMP-cross-kind", "compare Einstein and France", "contains_all",
     ["aren't really comparable", "physicist", "country"],
     "Different kinds -> honest 'not really comparable', states what each IS from a "
     "real IsA edge; never forces a contrast."),
    ("CMP-cross-kind-nofab", "compare Einstein and France", "contains_none",
     ["born in", "known for", "capital is paris"],
     "Cross-kind answer does NOT dump per-side attributes or invent a shared relation."),
    # ---- concept pair still handled by the concept `comparative` (no hijack) -
    ("CMP-concept-defer", "compare a dog and a cat", "contains_all",
     ["depends on what you value"],
     "Thin/common-noun nodes fail the rich-signal gate and fall through to the "
     "concept comparative unchanged (no entity-comparison hijack)."),
    ("CMP-concept-defer-nofab", "compare a dog and a cat", "contains_none",
     ["aren't really comparable"],
     "A concept pair is never mis-labelled 'not comparable' by the entity shape."),
    # ---- GUARDS: single-subject overviews + concept + refusal intact --------
    ("GUARD-who-einstein", "who is Albert Einstein", "contains_all",
     ["physicist", "in 1879", "in 1955"], "Single-subject person overview untouched."),
    ("GUARD-tell-france", "tell me about France", "contains_all",
     ["capital is paris", "currency is euro"], "Single-subject place overview untouched."),
    ("GUARD-who-wrote-hamlet", "who wrote Hamlet", "contains_all",
     ["shakespeare"], "AuthoredBy arm untouched."),
    ("GUARD-nonsense", "glorptar the magnificent", "refuses", None,
     "Nonsense -> honest refusal, never a fabricated comparison."),
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8787)
    args = ap.parse_args()
    base = f"http://127.0.0.1:{args.port}"
    pw = os.environ.get("LU_GUEST_PASSPHRASE")
    if not pw:
        print("FATAL: LU_GUEST_PASSPHRASE not set", file=sys.stderr); return 99
    token = post(f"{base}/auth", {"passphrase": pw}).get("token")
    if not token:
        print("FATAL: auth failed", file=sys.stderr); return 99
    print(f"=== OP-COMPARE acceptance: {len(PROBES)} probes ===\n")
    failures = 0
    for pid, q, kind, arg, why in PROBES:
        try:
            resp = post(f"{base}/mcp", {"method": "tools/call",
                "params": {"name": "lu4d_ask", "arguments": {"query": q}}}, token=token)
            raw = extract_text(resp); err = None
        except Exception as e:
            raw = ""; err = repr(e)
        low = (raw or "").lower()
        is_ref = any(m in low for m in REFUSAL_MARKERS)
        if err:
            ok, detail = False, f"transport error: {err}"
        elif kind == "contains_all":
            miss = [a for a in arg if a.lower() not in low]
            ok, detail = (not miss and not is_ref), f"missing {miss}" + (" (refused)" if is_ref else "")
        elif kind == "contains_none":
            bad = [a for a in arg if a.lower() in low]
            ok, detail = (not bad), f"must not contain {bad}"
        elif kind == "refuses":
            ok, detail = is_ref, "must be an honest refusal"
        else:
            ok, detail = False, f"unknown kind {kind}"
        tag = "PASS" if ok else "FAIL"
        if not ok:
            failures += 1
        print(f"[{tag}] {pid:26s} {q!r}")
        if not ok:
            print(f"        {detail}")
            print(f"        -> {raw!r}")
    print(f"\n=== {len(PROBES)-failures}/{len(PROBES)} passed, {failures} FAILED ===")
    return failures

if __name__ == "__main__":
    sys.exit(main())
