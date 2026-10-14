#!/usr/bin/env python3
"""OP-SOURCEPULL Batch-B ingest (Fix 1 concept definitions + Fix 3 person dates).
CC0 Wikidata via wbsearchentities+wbgetentities. Verify-first: every QID is
resolved BY LABEL (no hardcoded QID guessing) and re-verified before use; no
fabrication. Writes to a WORKING COPY brain dir.

  python3 srcpullB_ingest.py --brain <copydir> --keyfile <kf> [--dry]
"""
import os, sys, time, json, argparse, urllib.request, urllib.parse
sys.path.insert(0, "/opt/lu/lu/merge-main/scripts")

UA="Lu4d-srcpullB/1.0 (info@dialectforge.com)"
API="https://www.wikidata.org/w/api.php"

CONCEPTS=["radioactivity","radioactive decay","nuclear fission",
          "quantum field theory","wave-particle duality","quantum entanglement"]
PERSONS=["Albert Einstein","Isaac Newton","Charles Darwin","Niels Bohr","Max Planck",
 "Marie Curie","Galileo Galilei","Nikola Tesla","Michael Faraday","James Clerk Maxwell",
 "Werner Heisenberg","Erwin Schrodinger","Richard Feynman","Paul Dirac","Ernest Rutherford",
 "Enrico Fermi","Wolfgang Pauli","Max Born","John von Neumann","Alan Turing",
 "Ada Lovelace","Stephen Hawking","Carl Sagan","Gregor Mendel","Louis Pasteur",
 "Johannes Kepler","Nicolaus Copernicus","Ludwig Boltzmann","Dmitri Mendeleev","Antoine Lavoisier"]

def _get(params):
    u=API+"?"+urllib.parse.urlencode(params)
    for a in range(6):
        try:
            r=urllib.request.Request(u,headers={"User-Agent":UA})
            return json.loads(urllib.request.urlopen(r,timeout=60).read())
        except Exception as e:
            time.sleep(2*(a+1))
    return None

def search_hits(label):
    d=_get({"action":"wbsearchentities","search":label,"language":"en",
            "type":"item","limit":"7","format":"json"})
    if not d or "search" not in d: return []
    return [(h["id"], h.get("label","")) for h in d["search"]]

def entity(qid, props):
    d=_get({"action":"wbgetentities","ids":qid,"props":props,"languages":"en","format":"json"})
    if not d or "entities" not in d or qid not in d["entities"]: return None
    e=d["entities"][qid]
    if "missing" in e: return None
    return e

def p31_set(claims):
    out=set()
    for s in claims.get("P31",[]):
        dv=s.get("mainsnak",{}).get("datavalue",{})
        if dv.get("type")=="wikibase-entityid": out.add(dv["value"].get("id"))
    return out

def year_of(claims, pid):
    for s in claims.get(pid,[]):
        if s.get("rank")=="deprecated": continue
        ms=s.get("mainsnak",{})
        if ms.get("snaktype")!="value": continue
        dv=ms.get("datavalue",{})
        if dv.get("type")!="time": continue
        t=dv["value"].get("time","")  # e.g. +1879-03-14T00:00:00Z
        prec=dv["value"].get("precision",0)
        if prec<9: continue
        sign=t[0]; body=t[1:]
        try: y=int(body.split("-")[0])
        except Exception: continue
        if sign=="-": return "%d BC"%y
        return str(y)
    return None

def collect_concepts():
    BAD={"Q13442814","Q5","Q4167410","Q4167836","Q17633526","Q7318358"}  # article/human/disambig/category/wikinews
    rows=[]
    for label in CONCEPTS:
        picked=None
        for (qid,hitlabel) in search_hits(label):
            e=entity(qid,"labels|descriptions|claims")
            if not e: continue
            desc=e.get("descriptions",{}).get("en",{}).get("value","")
            p31=p31_set(e.get("claims",{}))
            if p31 & BAD: continue
            if not desc: continue
            picked=(qid,hitlabel,desc); break
        if not picked:
            print("  CONCEPT MISS (no groundable concept):",label); continue
        qid,hitlabel,desc=picked
        print("  CONCEPT OK %s=%s (%s) desc=%r"%(label,qid,hitlabel,desc))
        rows.append((label,desc))
    return rows

def collect_persons():
    rows=[]
    for name in PERSONS:
        hits=search_hits(name)
        if not hits: print("  PERSON MISS (no qid):",name); continue
        qid,hitlabel=hits[0]
        e=entity(qid,"labels|claims")
        if not e: print("  PERSON MISS (fetch):",name,qid); continue
        labs=e.get("labels",{})
        en=labs.get("en",{}).get("value","") or labs.get("mul",{}).get("value","") or hitlabel
        cl=e.get("claims",{})
        if "Q5" not in p31_set(cl): print("  PERSON REJECT (not human):",name,qid,en); continue
        by=year_of(cl,"P569"); dy=year_of(cl,"P570")
        if not en or (not by and not dy): print("  PERSON no dates:",name,qid,en); continue
        subj="%s#%s"%(en,qid)
        print("  PERSON OK %-22s %s born=%s died=%s"%(en,qid,by,dy))
        if by: rows.append((subj,"BornOn",by))
        if dy: rows.append((subj,"DiedOn",dy))
    return rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--brain",required=True); ap.add_argument("--keyfile",required=True)
    ap.add_argument("--dry",action="store_true")
    a=ap.parse_args()
    print("=== COLLECT CONCEPTS (Fix 1) ===",flush=True)
    cdefs=collect_concepts()
    print("=== COLLECT PERSON DATES (Fix 3) ===",flush=True)
    pdates=collect_persons()
    print("\nSUMMARY: concept_defs=%d  date_bonds=%d"%(len(cdefs),len(pdates)),flush=True)
    if a.dry:
        print("DRY -- no writes"); return
    if not cdefs and not pdates:
        print("nothing to ingest"); return
    import importlib
    os.environ["LU4D_BIN"]="/opt/lu/lu/merge-main/target/release/lu4d"
    ic=importlib.import_module("ingest_curriculum_full")
    # --- Fix 1: definitions (canon ON so stored key == fold(subject)) ---
    if cdefs:
        os.environ["LU_CANON_INGEST"]="1"
        cl=ic.McpClient(a.brain,a.keyfile,log_stderr_to="/tmp/srcpullB_def.err",ingest_only=True)
        entries=[{"label":l,"definition":d} for (l,d) in cdefs]
        r1=cl.call("lu4d_set_definition_batch",{"entries":entries,"update":False})
        r2=cl.call("lu4d_set_definition_batch",{"entries":entries,"update":True})
        try: fl=str(cl.call("lu4d_flush",{}))[:80]
        except Exception as ex: fl="flush_err %s"%ex
        print("DEF create:",json.dumps(r1)[:200])
        print("DEF update:",json.dumps(r2)[:200])
        print("DEF flush:",fl)
        cl.close()
    # --- Fix 3: date bonds (canon OFF so entity #QID labels stay verbatim) ---
    if pdates:
        os.environ["LU_CANON_INGEST"]="0"
        cl=ic.McpClient(a.brain,a.keyfile,log_stderr_to="/tmp/srcpullB_date.err",ingest_only=True)
        sink=ic.BatchSink(cl, ic.Throttle(getattr(ic,"THROTTLE_CODING",30.0)))
        for (s,r,o) in pdates:
            sink.add({"subject":s,"relation":r,"object":o,"weight":0.9,
                      "confidence":"Verified","corpus":"wikidata","run_id":"srcpullB_dates"})
        sink.flush()
        try: fl=str(cl.call("lu4d_flush",{}))[:80]
        except Exception as ex: fl="flush_err %s"%ex
        print("DATES landed=%d existing=%d gated=%d rejected=%d errors=%d flush=%s"%(
            sink.bonds_landed,sink.bonds_existing,sink.bonds_gated,sink.bonds_rejected,sink.bonds_errors,fl))
        cl.close()
    print("DONE_SRCPULLB_INGEST",flush=True)

if __name__=="__main__": main()
