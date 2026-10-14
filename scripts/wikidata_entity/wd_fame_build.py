#!/usr/bin/env python3
"""OP-FAME: build the FAME sidecar (the RIGHT notability signal).

fame(qid) = number of Wikipedia/Wikimedia SITELINKS (language editions) for
that QID -- a real prominence proxy. Isaac Newton (Q935) has ~300 sitelinks,
a same-named town ~20, so this inverts the wrong appearance-count ranking.

Source: Wikidata Query Service `wikibase:sitelinks` (CC0). Built OFFLINE off
the live brain: input is the QID list Lu actually holds (entity shelf),
output is a `qid<TAB>sitelink_count` TSV for $LU_ENTITY_FAME. Only the
sidecar file is produced; the brain is never mutated.

usage: wd_fame_build.py <qids_tsv> <out_tsv>
"""
import sys, time, urllib.parse, urllib.request

WDQS = "https://query.wikidata.org/sparql"
UA = "LuFameSidecar/1.0 (dialectforge; CC0 offline build; contact info@dialectforge.com)"
CHUNK = 5000

def fetch(qids):
    vals = " ".join("wd:Q%d" % q for q in qids)
    query = "SELECT ?item ?c WHERE { VALUES ?item { %s } ?item wikibase:sitelinks ?c . }" % vals
    data = urllib.parse.urlencode({"query": query, "format": "json"}).encode()
    req = urllib.request.Request(WDQS, data=data, headers={
        "Accept": "application/sparql-results+json", "User-Agent": UA})
    import json
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                d = json.load(r)
            out = {}
            for b in d["results"]["bindings"]:
                q = int(b["item"]["value"].rsplit("/", 1)[-1][1:])
                out[q] = int(b["c"]["value"])
            return out
        except Exception as e:
            wait = 3 * (attempt + 1)
            sys.stderr.write("  retry %d (%s) waiting %ds\n" % (attempt+1, type(e).__name__, wait))
            time.sleep(wait)
    raise SystemExit("WDQS failed after retries on chunk of %d" % len(qids))

def main():
    qids_tsv, out_tsv = sys.argv[1], sys.argv[2]
    qids = []
    for line in open(qids_tsv):
        p = line.split("\t", 1)
        if p and p[0].strip().isdigit():
            qids.append(int(p[0]))
    qids = sorted(set(qids))
    sys.stderr.write("input qids=%d\n" % len(qids))
    fame = {}
    t0 = time.time()
    for i in range(0, len(qids), CHUNK):
        chunk = qids[i:i+CHUNK]
        got = fetch(chunk)
        fame.update(got)
        if (i // CHUNK) % 10 == 0:
            sys.stderr.write("  chunk %d/%d fame_rows=%d %.0fs\n" % (
                i//CHUNK+1, (len(qids)+CHUNK-1)//CHUNK, len(fame), time.time()-t0))
        time.sleep(0.5)
    with open(out_tsv, "w") as f:
        for q in sorted(fame):
            f.write("%d\t%d\n" % (q, fame[q]))
    sys.stderr.write("DONE input=%d fame_rows=%d (0-sitelink qids omitted) bytes~ %.0fs\n" % (
        len(qids), len(fame), time.time()-t0))

if __name__ == "__main__":
    main()
