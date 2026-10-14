#!/usr/bin/env python3
"""OP-PHYSICS-SERVE -- physics serving-fix acceptance + guards.

Companion to recall_acceptance.py / attr_routing_acceptance.py. Drives the
LIVE guest service (POST /auth, POST /mcp lu4d_ask) on --port (default 8787).

TWO SECTIONS:
  REQUIRED  -- must PASS. The half-life dispatch fix + every guard the fix
               could plausibly disturb (real arithmetic, python->snake,
               official-language, located-in, Einstein born, nonsense floor).
  DEFERRED  -- report-only. The Becquerel / Schrodinger / superposition /
               Bohr targets that VERIFY-FIRST proved are SOURCE-GAPs (the
               correct QID node is absent from the live canonical brain:
               e_becquerel / e_schrodinger buckets do not exist; e_bohr
               lacks the physicist Q7085, only a deg-1 homograph; quantum
               superposition has no groundable gloss shelf). No serving
               change can honestly answer these -- they need a data pull.
               Printed so the gap is visible, never failing the suite.

    sudo -E python3 scripts/physics_probe_acceptance.py [--port 8787]
Exit 0 = all REQUIRED passed. Non-zero = number of REQUIRED failures.
"""
import argparse, json, os, sys, urllib.request as U

REFUSAL = ["i don't have", "i do not have", "i don't know", "nothing in my brain",
           "don't match my brain", "no answer for", "below my confidence",
           "anything solid", "won't fabricate", "will not fabricate"]

# (id, query, kind, arg)
#   contains_any -> answer must contain one of arg (lowercased)
#   refuses      -> answer must look like a refusal
REQUIRED = [
    ("HALF-what",  "what is half-life", "contains_any",
        ["radioactive", "decay", "half its initial", "atoms", "disintegrate", "time required"]),
    ("HALF-bare",  "half-life", "contains_any",
        ["radioactive", "decay", "half its initial", "atoms", "disintegrate", "time period", "time required"]),
    ("ARITH-53",   "what is 5-3", "contains_any", ["= 2", "2 (computed)", "is 2"]),
    ("ARITH-sp",   "what is 5 - 3", "contains_any", ["= 2", "2 (computed)", "is 2"]),
    ("ARITH-mul",  "what is 6 * 7", "contains_any", ["42"]),
    ("PY-snake",   "python", "contains_any", ["snake", "boa", "pythonidae", "pythoninae", "reptile"]),
    ("PY-isa",     "is a python a snake", "contains_any", ["kind of snake", "is a snake", "yes"]),
    ("OL-france",  "official language of France", "contains_any", ["french"]),
    ("LOC-paris",  "where is Paris located", "contains_any", ["france"]),
    ("BORN-einstein", "where was Albert Einstein born", "contains_any", ["ulm"]),
    ("FLOOR-nonsense", "what is a flibbertigibbet quantum banana", "refuses", None),
]

DEFERRED = [
    ("DEFER-becquerel",   "who is Henri Becquerel"),
    ("DEFER-schrodinger", "who is Erwin Schr\u00f6dinger"),
    ("DEFER-superposition","what is superposition"),
    ("DEFER-bohr",        "who is Niels Bohr"),
]

def post(url, payload, token=None, timeout=180):
    r = U.Request(url, data=json.dumps(payload).encode(), method="POST")
    r.add_header("Content-Type", "application/json")
    if token: r.add_header("Authorization", "Bearer " + token)
    with U.urlopen(r, timeout=timeout) as x: return json.loads(x.read().decode())

def extract(resp):
    r = resp.get("result", resp); c = r.get("content")
    if isinstance(c, list):
        return "\n".join(x.get("text", "") for x in c if isinstance(x, dict))
    for k in ("text", "answer", "narration"):
        if isinstance(r.get(k), str): return r[k]
    return json.dumps(r)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8787)
    args = ap.parse_args()
    base = f"http://127.0.0.1:{args.port}"
    pw = os.environ.get("LU_GUEST_PASSPHRASE")
    if not pw: print("FATAL: LU_GUEST_PASSPHRASE not set", file=sys.stderr); return 99
    token = post(f"{base}/auth", {"passphrase": pw}).get("token")
    if not token: print("FATAL: auth failed", file=sys.stderr); return 99
    def ask(q):
        return extract(post(f"{base}/mcp", {"method": "tools/call",
            "params": {"name": "lu4d_ask", "arguments": {"query": q}}}, token))
    fails = 0
    print(f"=== OP-PHYSICS-SERVE: {len(REQUIRED)} REQUIRED probes ===\n")
    for pid, q, kind, arg in REQUIRED:
        raw = ask(q); low = raw.lower()
        if kind == "contains_any":
            ok = any(a in low for a in arg)
        elif kind == "refuses":
            ok = any(m in low for m in REFUSAL)
        else:
            ok = False
        if not ok: fails += 1
        print(f"[{'PASS' if ok else 'FAIL'}] {pid:<15} {q!r}")
        print(f"        -> {raw[:180]!r}")
        if not ok: print(f"        expected {kind} {arg}")
    print(f"\n--- DEFERRED (report-only, known SOURCE-GAP, never fails suite) ---")
    for pid, q in DEFERRED:
        raw = ask(q)
        print(f"[gap ] {pid:<18} {q!r}\n        -> {raw[:180]!r}")
    print(f"\n=== {len(REQUIRED)-fails}/{len(REQUIRED)} REQUIRED passed, {fails} FAILED ===")
    return min(fails, 98)

if __name__ == "__main__":
    sys.exit(main())
