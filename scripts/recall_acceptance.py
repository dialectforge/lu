#!/usr/bin/env python3
"""OP-RECALL-1 -- recall acceptance suite.

The regression net that would have caught OP-SEED-DIAG's findings before
John did. Every probe corresponds to a real observed failure, or to a
property that must not regress while fixing one.

It drives the LIVE service (POST /auth, then POST /mcp lu4d_ask) rather
than loading a second reader: the canonical brain does not fit in RAM
twice on this box.

    sudo -E python3 recall_acceptance.py [--port 8787] [--json out.json]

Exit 0 = all passed. Non-zero = number of failures. Fails LOUDLY: every
failure prints the probe, the expectation, the normalised core answer and
the verbatim raw answer.

---------------------------------------------------------------------
WHY THERE IS A NORMALISER
---------------------------------------------------------------------
The first cut of this suite reported 17/19 PASS against a brain that was
visibly broken -- because Lu echoes the query back (`On "what is a dog",
...`) and appends curiosity chatter (`There's a gap nearby -- I have thin
coverage around X`). A naive substring check on the whole response is
therefore satisfied by the QUESTION, not the ANSWER, and an empty answer
with a gap note attached looks like a substantive reply.

So `core_answer()` strips the echo and the boilerplate first, and every
assertion runs against what is left. A test that cannot fail is worse
than no test: it is a false all-clear.
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.request

# Sentences Lu appends that are ABOUT the answering process rather than
# part of the answer. Stripped before assertions.
BOILERPLATE_PATTERNS = [
    r"There's a gap nearby[^.]*\.",
    r"There's a nearby gap[^.]*\.",
    r"While answering, I noticed[^.]*\.",
    r"\(I've been wondering about[^)]*\)\.?",
    r"\(I read[^)]*\)\.?",
    r"I'm inferring this from[^.]*\.",
    r"Confidence: [0-9.]+\.",
    r"Confidence: (very low|low|moderate|high|very high)\.",
]

REFUSAL_MARKERS = [
    "i don't have", "i do not have", "i don't know", "i do not know",
    "won't fabricate", "will not fabricate", "nothing in my brain",
    "don't match my brain", "no bonds activated", "no answer for",
    "below my confidence threshold",
]


def core_answer(raw, query):
    """The part of the response that is actually an ANSWER."""
    t = raw or ""
    # 1. Drop the echoed query in all the shapes the renderer uses.
    t = re.sub(r'^On\s+"' + re.escape(query) + r'",\s*', "", t, flags=re.I)
    t = re.sub(r'^"' + re.escape(query) + r'":\s*', "", t, flags=re.I)
    t = t.replace(f'"{query}"', " ")
    # 2. Drop process commentary.
    for pat in BOILERPLATE_PATTERNS:
        t = re.sub(pat, " ", t, flags=re.I)
    # 3. Drop the query's own words: an answer that only parrots the
    #    question has told us nothing.
    qwords = {w.lower() for w in re.findall(r"[a-z0-9]+", query, flags=re.I)}
    kept = [w for w in re.findall(r"[A-Za-z0-9()'-]+", t) if w.lower() not in qwords]
    return " ".join(kept).strip(), t.strip()


def looks_like_refusal(raw):
    t = (raw or "").lower()
    return any(m in t for m in REFUSAL_MARKERS)


# =====================================================================
# kinds:
#   core_contains_any -- the ANSWER (echo + boilerplate stripped) must
#                        mention one of `arg`.
#   substantive       -- the ANSWER must carry >= `arg` real words after
#                        stripping. Catches the observed "." replies that
#                        are neither an answer nor an honest refusal.
#   refuses           -- must be a refusal.
#   forbids           -- the RAW response must not contain any of `arg`.
# =====================================================================
PROBES = [
    # ---- R1: a single content token must reach the entity hub --------
    ("R1-dog", "what is a dog", "core_contains_any",
     ["canis", "canine", "domestic dog", "mammal", "breed", "wolf", "puppy", "animal"],
     "THE headline case. Pre-fix this returned 'an off-leash dog park located "
     "at the township municipal is Marlboro' -- two byte-sorted members of a "
     "22,718-entry bucket. Note the expectations deliberately EXCLUDE the bare "
     "word 'dog', which the junk answer also contained."),
    ("R1-dog-bare", "dog", "core_contains_any",
     ["canis", "canine", "domestic", "mammal", "breed", "wolf", "puppy", "animal"],
     "Question words are irrelevant to the defect -- the bare token failed "
     "identically. If this passes but R1-dog does not, the bug moved to intent "
     "routing."),
    ("R1-python", "python", "core_contains_any",
     ["programming", "language", "snake", "pythonidae", "guido", "software", "script"],
     "OP-SEED-DIAG's decisive control. Pre-fix: 'Parser implementation in "
     "Python'. 'parser' and 'implementation' are NOT accepted here."),
    ("R1-recursion", "recursion", "substantive", 3,
     "One of the 8/8 failing single-token probes. Pre-fix the core answer was "
     "EMPTY -- just a period plus a gap note."),
    ("R1-photosynthesis", "photosynthesis", "substantive", 3,
     "Pre-fix the entire response was '.'. A brain trained on Wikipedia that "
     "cannot say anything about photosynthesis is not usefully trained."),
    ("R1-gravity", "gravity", "substantive", 3, "One of the 8/8."),
    ("R1-volcano", "volcano", "substantive", 3,
     "Pre-fix: '. There's a nearby gap I haven't filled in.' -- empty."),
    ("R1-mitochondria", "mitochondria", "substantive", 3,
     "Pre-fix: empty core answer with a gap note attached."),

    # ---- the multi-token path must not regress -----------------------
    ("R0-python-language", "python language", "core_contains_any",
     ["programming", "python (programming"],
     "The path that ALREADY worked (returned 'Python (programming language)'). "
     "R1 must not break the majority-coverage scan while fixing the "
     "single-token one."),
    ("R0-python-prose", "tell me about the python language", "core_contains_any",
     ["programming", "python"],
     "Multi-word prose form; known-good. Regression canary for R2's raised "
     "MAJORITY_SCAN_CAP."),
    ("R0-hash-table", "what is a hash table", "substantive", 2,
     "Multi-token, known-good ('Distributed hash table')."),
    ("R0-domestic-dog", "domestic dog", "substantive", 1,
     "The canonical article title, reachable via the 2-token path even "
     "pre-fix. Proves the node exists, so any R1 failure is ranking, not "
     "extraction."),

    # ---- MetaSelf coherence ------------------------------------------
    ("COH-know-python", "What do you know about python", "forbids",
     ["0 concepts", "0 associations"],
     "The coherence bug verbatim: 'I have 0 concepts and 0 associations ... "
     "Recently I activated Parser implementation in Python.' Asserting "
     "emptiness and then talking violates the anti-hallucination floor in "
     "spirit. Two faults compose: R5 zeroed the counters, and MetaSelf is the "
     "wrong shape for a topical question."),
    ("COH-know-python-answers", "What do you know about python", "core_contains_any",
     ["programming", "language", "snake", "software"],
     "'What do you know about X' is a question about X, not a request for a "
     "brain census. It must answer on the subject."),
    ("COH-know-dog", "What do you know about dogs", "forbids",
     ["0 concepts", "0 associations"],
     "Same shape, different subject -- guards against a single-query fix."),

    # ---- the refusal floor must survive ------------------------------
    ("FLOOR-nonsense", "what is a flibbertigibbet quantum banana", "refuses", None,
     "THE most important probe here. Every ranking change is a chance to lower "
     "the refusal floor by over-seeding. If this ever returns an answer, the "
     "fix is worse than the bug."),
    ("FLOOR-nonsense-2", "zxqwv plorbnak", "refuses", None,
     "Pure noise, no real tokens."),
    ("FLOOR-stopwords", "what is the of and to a in", "refuses", None,
     "All hub-stopwords; produced 0 curiosity signals in the diag (correct). "
     "R3 makes is_hub_stopword live again on the cache path, so this is the "
     "probe that catches that going wrong."),

    # ---- OP-NEARMISS-1 typo tolerance must survive -------------------
    ("TYPO-launguage", "Tell me about the python launguage", "core_contains_any",
     ["programming", "python"],
     "OP-NEARMISS-1's case, whose first cut broke the refusal floor. Both "
     "halves are asserted here: this answers AND FLOOR-nonsense still "
     "refuses."),
]


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--json", default=None)
    ap.add_argument("--only", default=None)
    args = ap.parse_args()

    base = f"http://127.0.0.1:{args.port}"
    passphrase = os.environ.get("LU_GUEST_PASSPHRASE")
    if not passphrase:
        print("FATAL: LU_GUEST_PASSPHRASE not set "
              "(source /etc/lu/guest_auth.env as root)", file=sys.stderr)
        return 99

    auth = post(f"{base}/auth", {"passphrase": passphrase})
    token = auth.get("token")
    if not token:
        print(f"FATAL: auth failed: {auth}", file=sys.stderr)
        return 99

    probes = [p for p in PROBES if not args.only or args.only in p[0]]
    results, failures = [], 0
    print(f"=== OP-RECALL-1 acceptance suite: {len(probes)} probes ===\n")

    for pid, query, kind, arg, why in probes:
        t0 = time.time()
        try:
            resp = post(
                f"{base}/mcp",
                {"method": "tools/call",
                 "params": {"name": "lu4d_ask", "arguments": {"query": query}}},
                token=token,
            )
            raw, err = extract_text(resp), None
        except Exception as e:  # noqa: BLE001 -- a transport failure IS a failure
            raw, err = "", repr(e)
        ms = int((time.time() - t0) * 1000)
        core, stripped = core_answer(raw, query)

        if err:
            ok, detail = False, f"transport error: {err}"
        elif kind == "core_contains_any":
            low = core.lower()
            ok = any(a.lower() in low for a in arg)
            detail = f"core answer must mention one of {arg}"
        elif kind == "substantive":
            # Counted on `stripped` (echo + boilerplate removed) rather
            # than `core` (which additionally removes the query's own
            # words). For an ENTITY question the correct answer often IS
            # the query as a proper noun -- "domestic dog" -> "Domestic
            # dog." -- and counting that as zero content would fail a
            # right answer. `core` stays the basis for
            # `core_contains_any`, where parroting the question really
            # does prove nothing.
            words = [w for w in re.findall(r"[A-Za-z0-9'-]+", stripped)
                     if any(c.isalpha() for c in w)]
            ok = len(words) >= arg and not looks_like_refusal(raw)
            detail = (f"answer must carry >= {arg} real word(s) after "
                      f"stripping boilerplate; got {len(words)}")
        elif kind == "forbids":
            low = (raw or "").lower()
            bad = [a for a in arg if a.lower() in low]
            ok, detail = not bad, f"must not contain {arg}" + (f" -- found {bad}" if bad else "")
        elif kind == "refuses":
            ok, detail = looks_like_refusal(raw), "expected a refusal"
        else:
            ok, detail = False, f"unknown probe kind {kind!r}"

        if not ok:
            failures += 1
        results.append({"id": pid, "query": query, "kind": kind, "expect": arg,
                        "why": why, "pass": ok, "ms": ms,
                        "answer": raw, "core": core})

        print(f"[{'PASS' if ok else 'FAIL'}] {pid:<24} ({ms:>5} ms)  {query!r}")
        if not ok:
            print(f"         {detail}")
            print(f"         why : {why}")
            print(f"         CORE: {core[:400]!r}")
            print(f"         RAW : {raw[:400]!r}\n")

    print(f"\n=== {len(probes) - failures}/{len(probes)} passed, {failures} FAILED ===")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"full results -> {args.json}")
    return min(failures, 98)


if __name__ == "__main__":
    sys.exit(main())
