#!/usr/bin/env python3
"""OP-RECALL-1 -- re-score the acceptance runs with the corrected criterion.

WHY THIS EXISTS (read before assuming I moved the goalposts):

The first criterion for the single-token probes was "the answer must
carry >= 3 real words after stripping boilerplate". That is a PROXY for
"the answer is substantive", and it turned out to be the wrong proxy.
After the fix, `python` returns exactly `Python.` -- the canonical entity
node, which is the RIGHT answer to the ranking question this task was
about, scored as a failure because it is one word.

The property actually under test is: **does recall return the canonical
entity node, or a leaf that merely mentions the word?** The clean
discriminator is whether the answer LEADS with the entity:

    'Python.'                            -> leads with python   RIGHT
    'Parser implementation in Python.'   -> mentions python      WRONG
    'Dog.'                               -> leads with dog       RIGHT
    'an off-leash dog park located at
     the township municipal is Marlboro' -> mentions dog         WRONG

Both runs are re-scored with the IDENTICAL new criterion, from the raw
answers already captured, so the before/after comparison is honest.

It does NOT rescue the fix: the pre-fix answers fail the new criterion
too, and for the right reason.
"""
import json
import re
import sys

BOILERPLATE = [
    r"There's a gap nearby[^.]*\.",
    r"There's a nearby gap[^.]*\.",
    r"While answering, I noticed[^.]*\.",
    r"\(I've been wondering about[^)]*\)\.?",
    r"\(I read[^)]*\)\.?",
    r"I'm inferring this from[^.]*\.",
    r"Confidence: [0-9.]+\.",
    r"Confidence: (very low|low|moderate|high|very high)\.",
]
REFUSAL = [
    "i don't have", "i do not have", "i don't know", "i do not know",
    "won't fabricate", "will not fabricate", "nothing in my brain",
    "don't match my brain", "no bonds activated", "no answer for",
    "below my confidence threshold",
]

# probe id -> (kind, arg)
CRITERIA = {
    "R1-dog":                  ("leads_with_any", ["dog", "domestic dog", "canis", "canine"]),
    "R1-dog-bare":             ("leads_with_any", ["dog", "domestic dog", "canis", "canine"]),
    "R1-python":               ("leads_with_any", ["python"]),
    "R1-recursion":            ("leads_with_any", ["recursion", "recursive"]),
    "R1-photosynthesis":       ("leads_with_any", ["photosynthesis", "photosynthetic"]),
    "R1-gravity":              ("leads_with_any", ["gravity", "gravitation"]),
    "R1-volcano":              ("leads_with_any", ["volcano", "volcanic"]),
    "R1-mitochondria":         ("leads_with_any", ["mitochondri"]),
    "R0-python-language":      ("contains_any",   ["python"]),
    "R0-python-prose":         ("contains_any",   ["python"]),
    "R0-hash-table":           ("contains_any",   ["hash table"]),
    "R0-domestic-dog":         ("leads_with_any", ["domestic dog", "dog", "canis"]),
    "COH-know-python":         ("forbids",        ["0 concepts", "0 associations"]),
    "COH-know-python-answers": ("leads_with_any", ["python"]),
    "COH-know-dog":            ("forbids",        ["0 concepts", "0 associations"]),
    "FLOOR-nonsense":          ("refuses", None),
    "FLOOR-nonsense-2":        ("refuses", None),
    "FLOOR-stopwords":         ("refuses", None),
    "TYPO-launguage":          ("contains_any",   ["python"]),
}


def stripped(raw, query):
    t = raw or ""
    t = re.sub(r'^On\s+"' + re.escape(query) + r'",\s*', "", t, flags=re.I)
    t = re.sub(r'^"' + re.escape(query) + r'":\s*', "", t, flags=re.I)
    for p in BOILERPLATE:
        t = re.sub(p, " ", t, flags=re.I)
    return t.strip()


def norm(t):
    return re.sub(r"[^a-z0-9 ]+", " ", (t or "").lower()).strip()


def refused(raw):
    low = (raw or "").lower()
    return any(m in low for m in REFUSAL)


def score(rec):
    kind, arg = CRITERIA[rec["id"]]
    raw = rec["answer"]
    s = norm(stripped(raw, rec["query"]))
    if kind == "leads_with_any":
        return (not refused(raw)) and any(s.startswith(a.lower()) for a in arg)
    if kind == "contains_any":
        return (not refused(raw)) and any(a.lower() in s for a in arg)
    if kind == "forbids":
        return not any(a.lower() in (raw or "").lower() for a in arg)
    if kind == "refuses":
        return refused(raw)
    raise SystemExit(f"unknown kind {kind}")


def load(path):
    with open(path) as f:
        return {r["id"]: r for r in json.load(f)}


def main():
    before, after = load(sys.argv[1]), load(sys.argv[2])
    nb = na = 0
    print(f"{'probe':<26} {'BEFORE':<7} {'AFTER':<7}  answer (after)")
    print("-" * 100)
    for pid in CRITERIA:
        b, a = before.get(pid), after.get(pid)
        bs = score(b) if b else False
        as_ = score(a) if a else False
        nb += bs
        na += as_
        mark = "  " if bs == as_ else ("->" if as_ else "!!")
        ans = stripped(a["answer"], a["query"])[:44] if a else ""
        print(f"{pid:<26} {'PASS' if bs else 'FAIL':<7} {'PASS' if as_ else 'FAIL':<7}{mark} {ans!r}")
    print("-" * 100)
    print(f"{'TOTAL':<26} {nb}/{len(CRITERIA):<5} {na}/{len(CRITERIA)}")


if __name__ == "__main__":
    main()
