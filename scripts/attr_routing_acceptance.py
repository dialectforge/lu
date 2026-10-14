#!/usr/bin/env python3
"""OP-ATTR-ROUTE -- attribute-relation routing acceptance suite.

Companion to recall_acceptance.py. Every probe targets an attribute
phrasing whose typed edge is PROVEN present on the resolved entity in the
live canonical brain, so a FAIL is a ROUTING regression, not missing data.

History: pre-fix (base main e3b5ef2) `entity_query` (src/synthesis.rs) had
EntIntent {Who, Where, Capital, KnownFor, Born, WhatIs} -- no
OfficialLanguage member -- so 'official language of X' never selected the
HasOfficialLanguage edge and fell through to generic recall ('I don't have
anything solid on that.'). All 9 OL probes below FAILED pre-fix; the 6
capital/born probes PASSED (regression guards). Post-fix: 15/15.

    sudo -E python3 scripts/attr_routing_acceptance.py [--port 8787]

Exit 0 = all passed. Non-zero = number of failures.

SOURCE-GAP EXCLUSIONS (bond genuinely absent / not representable -- NOT
routing; excluded per the no-force rule):
  * currency of X   -- no HasCurrency BondType in the schema, and no
    currency PID in the two-tier ingest map (census: 0 currency edges).
  * author of <book/play> -- WrittenBy EXISTS (9,015 subjects / 10,363
    edges) but is P50+P170 CREATOR-of-artwork semantics, not book-author:
    "Romeo and Juliet" resolves to a PAINTING by Ford Madox Brown,
    "Hamlet" to a physicist, "The Origin of Species" to a lipidomics
    paper; the actual novels (War and Peace, Pride and Prejudice) carry
    ZERO out-edges. A general author arm would emit CONFIDENT WRONG
    answers -- worse than an honest refusal -- so it is NOT wired.
  * <landmark> located-in -- monument nodes (Eiffel Tower, Statue of
    Liberty) mis-resolve to same-named paintings that lack LocatedIn;
    CITY/country LocatedIn is populated and IS routed (LOC-* probes).
  * official language of Italy/China/Poland/Mexico/Sweden/Finland/Belarus
    -- HasOfficialLanguage edge absent on those nodes (Lu refuses honestly).
"""
import argparse, json, os, sys, time, urllib.request

# (id, query, [expected answer substrings lowercased], pre_fix_expected_pass)
PROBES = [
    # ---- OfficialLanguage: FLIP set (pre-fix FAIL -> post-fix PASS) ----
    ("OL-minsk",   "official language of Minsk",   ["belarusian", "russian"], False),
    ("OL-france",  "official language of France",  ["french"],               False),
    ("OL-germany", "official language of Germany", ["german"],               False),
    ("OL-spain",   "official language of Spain",   ["spanish"],              False),
    ("OL-japan",   "official language of Japan",   ["japanese"],             False),
    ("OL-russia",  "official language of Russia",  ["russian"],              False),
    ("OL-canada",  "official language of Canada",  ["english"],              False),
    ("OL-ukraine", "official language of Ukraine", ["ukrainian"],            False),
    ("OL-brazil",  "official language of Brazil",  ["portuguese"],           False),
    # ---- Capital: regression guards (pre-fix PASS, must stay PASS) ----
    ("CAP-france",  "capital of France",   ["paris"],    True),
    ("CAP-japan",   "capital of Japan",    ["tokyo"],    True),
    ("CAP-australia","capital of Australia",["canberra"], True),
    ("CAP-brazil",  "capital of Brazil",   ["brasília", "brasilia"], True),
    # ---- BornIn: regression guards (pre-fix PASS, must stay PASS) ----
    ("BORN-einstein", "where was Albert Einstein born", ["ulm"],    True),
    ("BORN-curie",    "where was Marie Curie born",     ["warsaw"], True),
    # ---- LocatedIn alt-phrasing: FLIP set (OP-ATTR-ROUTE B2) ----
    # LocatedIn is a large populated family (195,305 subjects / 394,147
    # edges in the live canonical). "where is X" already routed; these
    # alt-phrasings did NOT: "where is X located" left a trailing token
    # that corrupted entity resolution (Berlin -> a Berlin university),
    # and "what country is X in" had no route at all. Real live pre-fix
    # on this 8-probe set = 3/8, and 2 of those 3 resolved the WRONG
    # subject that merely CONTAINED the country name (e.g. University of
    # Sydney). Post-fix = 8/8 with correct subject resolution.
    ("LOC-paris-loc",  "where is Paris located",   ["france"],    False),
    ("LOC-tokyo-loc",  "where is Tokyo located",   ["japan"],     False),
    ("LOC-berlin-loc", "where is Berlin located",  ["germany"],   False),
    ("LOC-sydney-loc", "where is Sydney located",  ["australia"], False),
    ("LOC-paris-cty",  "what country is Paris in",  ["france"],    False),
    ("LOC-tokyo-cty",  "what country is Tokyo in",  ["japan"],     False),
    ("LOC-berlin-cty", "what country is Berlin in", ["germany"],   False),
    ("LOC-sydney-cty", "what country is Sydney in", ["australia"], False),
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
    c = r.get("content")
    if isinstance(c, list):
        parts = [x.get("text", "") for x in c if isinstance(x, dict)]
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
    args = ap.parse_args()
    base = f"http://127.0.0.1:{args.port}"
    pw = os.environ.get("LU_GUEST_PASSPHRASE")
    if not pw:
        print("FATAL: LU_GUEST_PASSPHRASE not set", file=sys.stderr); return 99
    token = post(f"{base}/auth", {"passphrase": pw}).get("token")
    if not token:
        print("FATAL: auth failed", file=sys.stderr); return 99

    results, failures, prefix_pass = [], 0, 0
    print(f"=== OP-ATTR-ROUTE acceptance suite: {len(PROBES)} probes ===\n")
    for pid, query, expect, prefix in PROBES:
        if prefix: prefix_pass += 1
        try:
            resp = post(f"{base}/mcp",
                        {"method": "tools/call",
                         "params": {"name": "lu4d_ask", "arguments": {"query": query}}},
                        token=token)
            raw = extract_text(resp)
        except Exception as e:
            raw = f"<transport error: {e!r}>"
        low = raw.lower()
        ok = any(s.lower() in low for s in expect)
        if not ok: failures += 1
        results.append({"id": pid, "query": query, "expect": expect,
                        "pass": ok, "pre_fix_pass": prefix, "answer": raw})
        print(f"[{'PASS' if ok else 'FAIL'}] {pid:<14} {query!r}")
        print(f"        -> {raw[:170]!r}")
        if not ok:
            print(f"        expected one of {expect}")
    npass = len(PROBES) - failures
    print(f"\n=== {npass}/{len(PROBES)} passed, {failures} FAILED ===")
    print(f"(stable pre-fix guard baseline: {prefix_pass}/{len(PROBES)} capital+born; "
          f"9 OfficialLanguage + 8 LocatedIn-alt probes had no reliable route pre-fix)")
    if args.json:
        with open(args.json, "w") as f: json.dump(results, f, indent=2)
    return min(failures, 98)

if __name__ == "__main__":
    sys.exit(main())
