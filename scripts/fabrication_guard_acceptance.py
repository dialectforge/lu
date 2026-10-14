#!/usr/bin/env python3
"""OP-FABGUARD internal notes -- multi-token proper-name honesty floor.

Guards the fabrication breach where a made-up multi-token name over-matched a
MINOR sub-token (an ordinal/suffix like "III" or a lone common given name) and
the person-overview arm narrated the wrong node as the asked subject.

Root cause: resolve_entity's weak-coverage fallback accepted a node that shared
only an ordinal token ("III" -> Charles III) as the subject for a name the
brain does not hold. Fix: below the honesty floor (the winning node fails to
cover EVERY distinctive/surname token of a multi-token query), REFUSE.

BEFORE (canonical eb3ade2, verbatim live):
  who is Marcus Vandenburg III -> "Charles III is a writer. III was born in Buckingham Palace."
  who is Zxqwer the Great III  -> "Charles III is a writer. III was born in Buckingham Palace."

Drives the LIVE service (POST /auth, then POST /mcp lu4d_ask).
    sudo -E python3 scripts/fabrication_guard_acceptance.py [--port 8787]
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
    "nothing in my brain", "don't have anything",
]

# (id, query, kind, arg, why)
#   refuses      -- answer must be an honest refusal.
#   contains_all -- lowercased answer must contain EVERY string in arg AND not be a refusal.
PROBES = [
    # ---- fabricated / absent multi-token names MUST refuse -------------
    ("FAB-marcus", "who is Marcus Vandenburg III", "refuses", None,
     "BEFORE: 'Charles III is a writer. III was born in Buckingham Palace.' "
     "'III' over-matched Charles III. Now refuses: surname 'vandenburg' absent."),
    ("FAB-fakename", "who is John Fakenamington", "refuses", None,
     "Given name matches many, surname absent -> refuse, never fabricate."),
    ("FAB-zxqwer", "who is Zxqwer the Great III", "refuses", None,
     "BEFORE: same Charles III fabrication. Now refuses: only ordinal 'iii' matched."),
    ("FAB-pure-nonsense", "who is Blorptak Vandolorian Sr", "refuses", None,
     "Fully invented multi-token name incl. a suffix -> refuse."),
    # ---- real names MUST still resolve (no over-refusal) ---------------
    ("REAL-einstein", "who was Einstein", "contains_all", ["physicist"],
     "Single famous name via fame path -> physicist bio."),
    ("REAL-newton", "who is Newton", "contains_all", ["newton"],
     "Single famous name -> Isaac Newton."),
    ("REAL-curie", "who was Marie Curie", "contains_all", ["physicist"],
     "Two-token real name -> physicist bio."),
    ("REAL-charles3", "who is Charles III", "contains_all", ["charles iii"],
     "A genuinely-held 'X III' regnal name must STILL resolve -- do not over-refuse."),
    ("REAL-isakhanli", "who is Hamlet Isakhanli", "contains_all", ["isakhanli"],
     "Two-token real person (given name collides with the play 'Hamlet') -> person."),
    ("REAL-einstein-full", "who is Albert Einstein", "contains_all", ["physicist"],
     "Full real name -> physicist bio (exact-label path)."),
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
    print(f"=== OP-FABGUARD acceptance: {len(PROBES)} probes ===\n")
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
        elif kind == "refuses":
            ok, detail = is_ref, "must be an honest refusal"
        elif kind == "contains_all":
            miss = [a for a in arg if a.lower() not in low]
            ok, detail = (not miss and not is_ref), f"missing {miss}" + (" (refused)" if is_ref else "")
        else:
            ok, detail = False, f"unknown kind {kind}"
        tag = "PASS" if ok else "FAIL"
        if not ok:
            failures += 1
        print(f"[{tag}] {pid:20s} {q!r}")
        print(f"        -> {raw!r}")
        if not ok:
            print(f"        {detail}")
    print(f"\n=== {len(PROBES)-failures}/{len(PROBES)} passed, {failures} FAILED ===")
    return failures

if __name__ == "__main__":
    sys.exit(main())
