#!/usr/bin/env python3
"""OP-CMP-RESOLVE internal notes acceptance probes.

Bug: the entity-comparison arm resolved BOTH subjects through the robust
resolve_entity, but when exactly ONE side failed the has_rich gate (a thin /
ambiguous node such as the place-sense of "Darwin") it returned None and fell
through to the weak comparative() shape, which uses the non-robust
resolve_subject -- failing "Marie Curie" ("I don't hold grounded facts about
Marie Curie") and dumping raw associations for Darwin ("monkey and territory").

Fix: when exactly one side is a substantive entity, entity_comparison now emits
an honest one-sided answer (naming what it DOES hold, with a real cited edge)
instead of degrading into the junk fallback. Working comparisons (both sides
substantive) and concept pairs (neither substantive) are unchanged.

    LU_GUEST_PASSPHRASE=... python3 scripts/comparison_bug4_probes.py --port 8788
Exit 0 = all passed.
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
    c = r.get("content")
    if isinstance(c, list):
        return "\n".join(x.get("text", "") for x in c if isinstance(x, dict))
    return json.dumps(r)

# (id, query, contains_all, contains_none, why)
PROBES = [
    ("B4-curie-vs-darwin", "Marie Curie vs Darwin",
     ["marie curie is a physicist"],
     ["monkey", "territory", "depends on what you value",
      "aren't really comparable", "don't hold grounded facts about marie curie"],
     "Curie resolves (physicist); Darwin honestly-ambiguous; NO raw-edge junk, NO weak fallback."),
    ("B4-compare-curie-darwin", "compare Curie and Darwin",
     ["physicist", "grounded, single-sense subject"],
     ["monkey", "territory", "depends on what you value"],
     "Same honest one-sided answer via the 'compare X and Y' connector."),
    ("B4-einstein-vs-newton", "Einstein vs Newton",
     ["physicist", "mathematician", "celestial mechanics"],
     ["grounded, single-sense subject", "depends on what you value"],
     "Both substantive -> still composes the full person-vs-person grid (no regression)."),
    ("B4-france-vs-germany", "France vs Germany",
     ["both located in europe", "paris", "berlin"],
     ["grounded, single-sense subject", "depends on what you value"],
     "Both substantive places -> still composes (no regression)."),
    ("B4-one-side-absent", "compare Albert Einstein and Marcus Vandenburg III",
     ["albert einstein is a physicist", "grounded, single-sense subject"],
     ["charles", "monkey", "depends on what you value"],
     "One side genuinely absent (FABGUARD refuses Marcus) -> honest 'can resolve X not Y', no fabrication."),
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
    print(f"=== OP-CMP-RESOLVE bug-4 probes: {len(PROBES)} ===\n")
    failures = 0
    for pid, q, need, forbid, why in PROBES:
        resp = post(f"{base}/mcp", {"method": "tools/call",
            "params": {"name": "lu4d_ask", "arguments": {"query": q}}}, token=token)
        low = extract_text(resp).lower()
        miss = [a for a in need if a.lower() not in low]
        bad = [a for a in forbid if a.lower() in low]
        ok = not miss and not bad
        if not ok:
            failures += 1
        print(f"[{'PASS' if ok else 'FAIL'}] {pid:24s} {q!r}")
        if not ok:
            print(f"        missing={miss} forbidden_present={bad}")
            print(f"        -> {extract_text(resp)!r}")
    print(f"\n=== {len(PROBES)-failures}/{len(PROBES)} passed, {failures} FAILED ===")
    return failures

if __name__ == "__main__":
    sys.exit(main())
