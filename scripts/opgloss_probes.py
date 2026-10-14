#!/usr/bin/env python3
"""OP-GLOSS acceptance probes (concept-gloss restore + compose + prose-prefix).

Drives the LIVE guest (POST /auth, then /mcp lu4d_ask), same contract as
recall_acceptance.py. Covers three deliverables:
  1. PART 0 data re-set: the four Batch-B concept glosses dropped by the
     breadthD compaction are restored (real CC0 Wikidata text).
  2. Prose provenance: no answer front-loads "From a single source on record,".
  3. Honest floor + standard guards still hold.

  sudo -E python3 opgloss_probes.py [--port 8787]
Exit 0 = all pass; non-zero = number of failures.
"""
import argparse, json, os, sys, urllib.request

REFUSAL = ["don't have anything solid", "i don't have", "i do not have", "won't fabricate",
           "will not fabricate", "didn't activate", "nothing in my brain", "no bonds activated"]

# (id, query, kind, arg)
#   contains_all  -> lowercased answer must contain every string in arg
#   forbids       -> answer must contain NONE of arg
#   refuses       -> answer must look like a refusal
PROBES = [
  ("DEF-radioactivity", "what is radioactivity", "contains_all", ["process by which an unstable atom emits radiation"]),
  ("DEF-decay", "what is radioactive decay", "contains_all", ["emission of subatomic particles by the decay"]),
  ("DEF-fission", "what is nuclear fission", "contains_all", ["nuclear reaction splitting the nucleus"]),
  ("DEF-duality", "what is wave-particle duality", "contains_all", ["quantum objects behave"]),
  ("DEF-entangle", "what is quantum entanglement", "contains_all", ["correlation between measurements"]),
  # Prose provenance: the stiff single-source lead-in must be gone everywhere.
  ("PREFIX-einstein", "who is Albert Einstein", "forbids", ["single source on record"]),
  ("PREFIX-einstein-body", "who is Albert Einstein", "contains_all", ["physicist"]),
  ("PREFIX-france", "tell me about France", "forbids", ["single source on record"]),
  ("PREFIX-france-body", "tell me about France", "contains_all", ["paris"]),
  ("PREFIX-warpeace", "what is War and Peace", "forbids", ["single source on record"]),
  ("PREFIX-warpeace-body", "what is War and Peace", "contains_all", ["tolstoy"]),
  # Guards: clean short glosses are NOT re-woven; entities still resolve.
  ("GUARD-python", "what is python", "contains_all", ["programming"]),
  ("GUARD-python-snake", "what is python", "contains_all", ["snake"]),
  ("GUARD-dog", "what is a dog", "contains_all", ["canis"]),
  ("GUARD-hamlet", "who wrote Hamlet", "contains_all", ["shakespeare"]),
  # Honest floor: nonsense must refuse -- the compose arm must not over-seed.
  ("FLOOR-nonsense", "what is a flibbertigibbet quantum banana", "refuses", None),
  ("FLOOR-nonsense2", "zxqwv plorbnak", "refuses", None),
]

def post(url, payload, token=None):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST")
    req.add_header("Content-Type", "application/json")
    if token: req.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read().decode())

def extract(resp):
    r = resp.get("result", resp)
    c = r.get("content")
    if isinstance(c, list):
        t = [x.get("text", "") for x in c if isinstance(x, dict)]
        if t: return "\n".join(t)
    for k in ("text", "answer", "narration"):
        if isinstance(r.get(k), str): return r[k]
    return json.dumps(r)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8787)
    a = ap.parse_args()
    base = "http://127.0.0.1:%d" % a.port
    pw = os.environ.get("LU_GUEST_PASSPHRASE")
    if not pw: print("FATAL: LU_GUEST_PASSPHRASE not set", file=sys.stderr); return 99
    token = post(base + "/auth", {"passphrase": pw}).get("token")
    if not token: print("FATAL: auth failed", file=sys.stderr); return 99
    fails = 0
    print("=== OP-GLOSS probes: %d ===" % len(PROBES))
    for pid, q, kind, arg in PROBES:
        resp = post(base + "/mcp", {"method": "tools/call",
              "params": {"name": "lu4d_ask", "arguments": {"query": q}}}, token=token)
        raw = extract(resp); low = raw.lower()
        if kind == "contains_all": ok = all(s.lower() in low for s in arg)
        elif kind == "forbids": ok = not any(s.lower() in low for s in arg)
        elif kind == "refuses": ok = any(m in low for m in REFUSAL)
        else: ok = False
        if not ok: fails += 1
        print("[%s] %-22s %r" % ("PASS" if ok else "FAIL", pid, q))
        if not ok: print("       arg=%r\n       RAW=%r" % (arg, raw[:300]))
    print("\n=== %d/%d passed, %d FAILED ===" % (len(PROBES) - fails, len(PROBES), fails))
    return min(fails, 98)

if __name__ == "__main__":
    sys.exit(main())
