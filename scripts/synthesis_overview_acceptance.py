#!/usr/bin/env python3
"""OP-OVERVIEW -- subject-overview synthesis acceptance suite.

Guards the richer subject-overview composition (weave birth/death place+YEAR
and more KnownFor into one coherent answer) AND the honesty floor it must
not break. Drives the LIVE service (POST /auth, then POST /mcp lu4d_ask).

    sudo -E python3 scripts/synthesis_overview_acceptance.py [--port 8787]

Exit 0 = all passed. Non-zero = number of failures. Every added token in a
richer answer is a real cited typed edge (a BornOn/DiedOn year, an extra
KnownFor, or the '...among others' trailing that a truncated-but-cited list
licenses) -- never a fabricated connective fact.
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
    "nothing in my brain",
]

# (id, query, kind, arg, why)
#   contains_all -- lowercased answer must contain EVERY string in arg.
#   contains_none -- lowercased answer must contain NONE of arg.
#   refuses -- answer must be an honest refusal.
PROBES = [
    # ---- RICHER person overviews (dates + more known-for) --------------
    ("RICH-einstein", "who is Albert Einstein", "contains_all",
     ["was a physicist", "in 1879", "in 1955", "among others"],
     "Weaves birth+death YEAR (BornOn 1879 / DiedOn 1955) and a 3rd known-for "
     "with an honest trail-off. Pre-change: no years, only 2 known-for."),
    ("RICH-einstein-tell", "tell me about Albert Einstein", "contains_all",
     ["born in ulm in 1879", "died in princeton in 1955"],
     "tell-me-about routes to the same enriched overview."),
    ("RICH-curie", "who is Marie Curie", "contains_all",
     ["born in warsaw in 1867", "died in sancellemoz in 1934"],
     "Birth+death place AND year woven."),
    ("RICH-darwin", "tell me about Charles Darwin", "contains_all",
     ["naturalist", "in 1809", "in 1882", "among others"],
     "Naturalist with 3+ works and both dates."),
    # ---- living person stays present tense, no invented death ----------
    ("LIVE-obama", "who is Barack Obama", "contains_all",
     ["is a politician"],
     "Living: present tense preserved."),
    ("LIVE-obama-nodeath", "who is Barack Obama", "contains_none",
     ["died", " was a "],
     "No death clause / false past tense from absence of a DiedIn/DiedOn edge."),
    # ---- honesty: thin / unresolved nodes refuse, never pad -----------
    ("ENRICHED-beethoven", "who is Ludwig van Beethoven", "contains_all", ["composer", "Bonn"],
     "Unresolved node -> honest refusal, not a padded/fabricated overview."),
    ("NONSENSE", "glorptar the magnificent", "refuses", None,
     "Nonsense -> refuse, no fabricated connection."),
    # ---- attribute arms still answer (no regression) ------------------
    ("ATTR-cap-fr", "what is the capital of France", "contains_all", ["paris"], "HasCapital."),
    ("ATTR-cur-fr", "what is the currency of France", "contains_all", ["euro"], "HasCurrency."),
    ("ATTR-lang-fr", "what is the official language of France", "contains_all", ["french"], "HasOfficialLanguage."),
    ("ATTR-born-date", "when was Albert Einstein born", "contains_all", ["1879"], "BornOn date arm."),
    ("ATTR-died-date", "when did Charles Darwin die", "contains_all", ["1882"], "DiedOn date arm."),
    ("ATTR-author", "who wrote War and Peace", "contains_all", ["tolstoy"], "AuthoredBy."),
    ("ATTR-hamlet", "who wrote Hamlet", "contains_all", ["shakespeare"], "AuthoredBy control."),
    ("ATTR-bornplace", "where was Albert Einstein born", "contains_all", ["ulm"], "BornIn place arm."),
    # ---- concept path intact (overview change must not leak) ----------
    ("CONCEPT-python", "what is python", "contains_all", ["language"], "python -> programming language."),
    ("CONCEPT-python-snake", "python", "contains_none", ["i don't have"], "bare python still answers (snake/lang)."),
    ("CONCEPT-dog", "what is a dog", "contains_all", ["canis"], "concept gloss unchanged."),
    # ---- OP-OVERVIEW-ROUTE: place overviews woven from real attribute edges --
    ("PLACE-france-tell", "tell me about France", "contains_all",
     ["is a country", "capital is paris", "official language is french", "currency is euro"],
     "tell-me-about a country now weaves HasCapital+HasOfficialLanguage+HasCurrency "
     "instead of stopping at the bare concept gloss. Pre-change: 'a republic in western Europe'."),
    ("PLACE-france-what", "what is France", "contains_all", ["paris", "french", "euro"],
     "what-is a country reaches the same enriched overview (shared who-is weave)."),
    ("PLACE-japan", "tell me about Japan", "contains_all", ["tokyo", "japanese", "yen"],
     "Japan: capital+language+currency."),
    ("PLACE-germany", "tell me about Germany", "contains_all", ["berlin", "german", "euro"],
     "Germany: capital+language+currency."),
    ("PLACE-canada", "tell me about Canada", "contains_all", ["ottawa", "canadian dollar"],
     "Canada: capital+currency across an ocean-less continent route."),
    ("PLACE-italy-partial", "tell me about Italy", "contains_all", ["capital is rome", "euro"],
     "Honest PARTIAL: Italy has HasCapital+HasCurrency but no HasOfficialLanguage edge."),
    ("PLACE-italy-omit", "tell me about Italy", "contains_none", ["official language"],
     "Absent attribute is OMITTED, never fabricated -- honest trail-off."),
    # ---- OP-OVERVIEW-ROUTE: bare work-node resolution + authorship weave -----
    ("WORK-warpeace-tell", "tell me about War and Peace", "contains_all",
     ["war and peace", "written by leo tolstoy"],
     "Bare 'tell me about <title>' resolves the AuthoredBy-bearing work node (#1506 "
     "prefer pass) and composes a work overview. Pre-change: 'I don't have anything solid'."),
    ("WORK-warpeace-what", "what is War and Peace", "contains_all", ["tolstoy"],
     "what-is a work reaches the same work overview."),
    # ---- honesty: a concept never grows a place/work overview ----------------
    ("NOLEAK-radioactivity", "what is radioactivity", "contains_none",
     ["its capital", "its currency", "official language", "written by"],
     "A bare concept is never over-woven into a place/work overview."),
    ("NOLEAK-python", "what is python", "contains_none",
     ["its capital", "its currency", "written by"],
     "The concept gloss path never leaks the place/work weave."),
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
    print(f"=== OP-OVERVIEW acceptance: {len(PROBES)} probes ===\n")
    failures = 0
    for pid, q, kind, arg, why in PROBES:
        try:
            resp = post(f"{base}/mcp", {"method": "tools/call",
                "params": {"name": "lu4d_ask", "arguments": {"query": q}}}, token=token)
            raw = extract_text(resp)
        except Exception as e:
            raw = ""; err = repr(e)
        else:
            err = None
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
        print(f"[{tag}] {pid:20s} {q!r}")
        if not ok:
            print(f"        {detail}")
            print(f"        -> {raw!r}")
    print(f"\n=== {len(PROBES)-failures}/{len(PROBES)} passed, {failures} FAILED ===")
    return failures

if __name__ == "__main__":
    sys.exit(main())
