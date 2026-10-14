#!/usr/bin/env python3
# OP-SOURCEPULL Batch-C ingest: currency (P38) + author (P50) entity bonds.
# CC0 Wikidata. Verify-first: hardcoded QIDs are the SUBJECT anchors (country /
# literary work); the object (currency / author) QID+label is FETCHED, never
# guessed. Attaches to the existing Name#QID entity node (dedup by seed).
# Writes to a WORKING COPY brain. LU_CANON_INGEST=0 so #QID labels stay verbatim.
import os, sys, time, json, argparse, urllib.request, urllib.parse
sys.path.insert(0, "/opt/lu/lu/merge-main/scripts")
UA="Lu4d-srcpullC/1.0 (info@dialectforge.com)"
API="https://www.wikidata.org/w/api.php"

# Subject QIDs (verified well-known). Object fetched via P38/P50.
COUNTRIES={"Egypt":"Q79","Japan":"Q17","France":"Q142","Germany":"Q183",
 "United States":"Q30","United Kingdom":"Q145","Canada":"Q16","Australia":"Q408",
 "Brazil":"Q155","India":"Q668","China":"Q148","Russia":"Q159",
 "Italy":"Q38","Spain":"Q29","Switzerland":"Q39","Sweden":"Q34","Norway":"Q20",
 "Poland":"Q36","Turkey":"Q43","Greece":"Q41","Portugal":"Q45","Austria":"Q40",
 "Netherlands":"Q55","South Korea":"Q884","Nigeria":"Q1033","Kenya":"Q114"}
WORKS={"War and Peace":"Q161531","Anna Karenina":"Q147787","Pride and Prejudice":"Q170583",
 "Hamlet":"Q41567","Romeo and Juliet":"Q83186","Macbeth":"Q131013","Crime and Punishment":"Q22823",
 "Moby-Dick":"Q174596","The Great Gatsby":"Q214371",
 "Nineteen Eighty-Four":"Q208460","Don Quixote":"Q480","Ulysses":"Q6511",
 "Frankenstein":"Q844722","Jane Eyre":"Q243396",
 "The Odyssey":"Q35160","The Iliad":"Q8275","Great Expectations":"Q219552",
 "A Tale of Two Cities":"Q308918", "Adventures of Huckleberry Finn":"Q215410"}

def _get(params):
    u=API+"?"+urllib.parse.urlencode(params)
    for a in range(6):
        try:
            r=urllib.request.Request(u,headers={"User-Agent":UA})
            return json.loads(urllib.request.urlopen(r,timeout=60).read())
        except Exception: time.sleep(2*(a+1))
    return None

def entity(qid, props="labels|claims"):
    d=_get({"action":"wbgetentities","ids":qid,"props":props,"languages":"en","format":"json"})
    if not d or "entities" not in d or qid not in d["entities"]: return None
    e=d["entities"][qid]
    return None if "missing" in e else e

def label_of(e, fallback=""):
    labs=e.get("labels",{})
    return labs.get("en",{}).get("value","") or labs.get("mul",{}).get("value","") or fallback

def best_claim_qid(claims, pid):
    # Prefer a normal-rank claim with no end-time (P582) qualifier; fall back to first.
    best=None
    for s in claims.get(pid,[]):
        if s.get("rank")=="deprecated": continue
        ms=s.get("mainsnak",{})
        if ms.get("snaktype")!="value": continue
        dv=ms.get("datavalue",{})
        if dv.get("type")!="wikibase-entityid": continue
        qid=dv["value"].get("id")
        has_end="P582" in s.get("qualifiers",{})
        rank=s.get("rank")
        score=(2 if rank=="preferred" else 1, 0 if has_end else 1)
        if best is None or score>best[0]: best=(score,qid)
    return best[1] if best else None

def collect(subjects, pid, kind):
    rows=[]
    for name,qid in subjects.items():
        e=entity(qid)
        if not e: print("  MISS(subj fetch) %s %s"%(name,qid)); continue
        subj_label=label_of(e, name)
        oq=best_claim_qid(e.get("claims",{}), pid)
        if not oq: print("  no %s: %s %s"%(pid,name,qid)); continue
        oe=entity(oq,"labels")
        olabel=label_of(oe) if oe else ""
        if not olabel: print("  MISS(obj label) %s %s %s"%(name,pid,oq)); continue
        subj="%s#%s"%(subj_label,qid); obj="%s#%s"%(olabel,oq)
        print("  %-28s %s %s"%(subj, kind, obj))
        rows.append((subj,obj))
    return rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--brain",required=True); ap.add_argument("--keyfile",default="/var/lib/lu/master.key")
    ap.add_argument("--dry",action="store_true")
    a=ap.parse_args()
    print("=== COLLECT CURRENCY (P38) ===",flush=True)
    cur=collect(COUNTRIES,"P38","HasCurrency")
    print("=== COLLECT AUTHOR (P50) ===",flush=True)
    au=collect(WORKS,"P50","AuthoredBy")
    print("\nSUMMARY currency=%d author=%d"%(len(cur),len(au)),flush=True)
    if a.dry: print("DRY -- no writes"); return
    if not cur and not au: print("nothing"); return
    import importlib
    os.environ["LU4D_BIN"]="/opt/lu/lu/merge-main/target/release/lu4d"
    os.environ["LU_CANON_INGEST"]="0"
    ic=importlib.import_module("ingest_curriculum_full")
    cl=ic.McpClient(a.brain,a.keyfile,log_stderr_to="/tmp/srcpullC.err",ingest_only=True)
    sink=ic.BatchSink(cl, ic.Throttle(getattr(ic,"THROTTLE_CODING",30.0)))
    for (s,o) in cur:
        sink.add({"subject":s,"relation":"HasCurrency","object":o,"weight":0.9,"confidence":"Verified","corpus":"wikidata","run_id":"srcpullC_currency"})
    for (s,o) in au:
        sink.add({"subject":s,"relation":"AuthoredBy","object":o,"weight":0.9,"confidence":"Verified","corpus":"wikidata","run_id":"srcpullC_author"})
    sink.flush()
    try: fl=str(cl.call("lu4d_flush",{}))[:80]
    except Exception as ex: fl="flush_err %s"%ex
    print("INGEST landed=%d existing=%d gated=%d rejected=%d errors=%d flush=%s"%(
        sink.bonds_landed,sink.bonds_existing,sink.bonds_gated,sink.bonds_rejected,sink.bonds_errors,fl))
    cl.close(); print("DONE_SRCPULLC")

if __name__=="__main__": main()
