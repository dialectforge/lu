#!/usr/bin/env python3
"""Recursion (CS) coverage top-up. John green-lit Q264164 (recursion, computer-science
sense: "algorithmic technique ... reducing it to a smaller instance of the same problem").
Mirrors topup_cc0.py EXACTLY (same WHITELIST, CONCEPT_CLASSES, tier derivation, merge-by-
group) but targets ONLY the verified Q264164 -- it does NOT re-run the CHECK list that would
re-insert the wrong Q131412 (Mary, Queen of Scots). CC0 wbgetentities, rank-filtered
preferred>normal. Only added if absent from base; present entities untouched.
"""
import time, json, urllib.request, urllib.parse
import duckdb
UA="Lu4d-entity-ingest/1.0 (info@dialectforge.com)"
API="https://www.wikidata.org/w/api.php"
OUT="/tmp/wd_twotier_edges.parquet"
WHITELIST=["P31","P106","P279","P17","P131","P30","P36","P800","P101","P19",
 "P20","P37","P361","P463","P527","P186","P50","P170","P61","P112",
 "P138","P1552","P828","P366"]
CONCEPT_CLASSES={"Q9143","Q175263","Q8366","Q190087","Q336","Q2465832",
 "Q11862829","Q24034552","Q246672","Q47574","Q11344","Q12136"}
TARGET="Q264164"
PER=8; OUTCAP=25

def get(ids,props):
    u=API+"?"+urllib.parse.urlencode({"action":"wbgetentities","ids":"|".join(ids),
        "props":props,"languages":"en","format":"json"})
    for a in range(6):
        try:
            r=urllib.request.Request(u,headers={"User-Agent":UA})
            return json.loads(urllib.request.urlopen(r,timeout=60).read())
        except Exception: time.sleep(2*(a+1))
    return None

con=duckdb.connect()
base_ids=set(r[0] for r in con.execute("SELECT DISTINCT s_id FROM read_parquet(?)",[OUT]).fetchall())
if TARGET in base_ids:
    print("already present; nothing to do"); raise SystemExit

lm={}; edges=[]; objs=set(); tiers={}
d=get([TARGET],"claims|labels|descriptions")
assert d and "entities" in d and TARGET in d["entities"], "fetch failed"
ent=d["entities"][TARGET]
assert "missing" not in ent, "target missing on API"
l=ent.get("labels",{}).get("en"); desc=ent.get("descriptions",{}).get("en",{}).get("value","")
if l: lm[TARGET]=l["value"]
print("VERIFY label=%r desc=%r" % (lm.get(TARGET), desc), flush=True)
assert lm.get(TARGET,"").lower()=="recursion", "label is not 'recursion' -- STOP"
cl=ent.get("claims",{})
is_concept = TARGET in CONCEPT_CLASSES
for pid in ("P31","P279"):
    for s in cl.get(pid,[]):
        ms=s.get("mainsnak",{}); dv=ms.get("datavalue",{})
        if dv.get("type")=="wikibase-entityid":
            if dv["value"].get("id") in CONCEPT_CLASSES: is_concept=True
tiers[TARGET]="concept" if is_concept else "entity"
oc=0
for pid in WHITELIST:
    if oc>=OUTCAP: break
    sts=cl.get(pid,[])
    pref=[s for s in sts if s.get("rank")=="preferred"]
    use=pref if pref else [s for s in sts if s.get("rank")!="deprecated"]
    rc=0
    for s in use:
        if rc>=PER or oc>=OUTCAP: break
        ms=s.get("mainsnak",{})
        if ms.get("snaktype")!="value": continue
        dv=ms.get("datavalue",{})
        if dv.get("type")!="wikibase-entityid": continue
        oid=dv["value"].get("id")
        if not oid: continue
        edges.append((TARGET,pid,oid)); objs.add(oid); rc+=1; oc+=1
miss=[o for o in objs if o not in lm]
for i in range(0,len(miss),50):
    d2=get(miss[i:i+50],"labels")
    if d2 and "entities" in d2:
        for qid,e2 in d2["entities"].items():
            l2=e2.get("labels",{}).get("en")
            if l2: lm[qid]=l2["value"]
rows=[]
for sid,pid,oid in edges:
    sl=lm.get(sid); ol=lm.get(oid)
    if not sl or not ol or len(sl)>120 or len(ol)>120: continue
    rows.append((sid,pid,oid,sl,ol,tiers.get(sid,"entity")))
print("tier:", tiers.get(TARGET), flush=True)
print("supplement rows:", len(rows), flush=True)
for r in rows: print("   ", r, flush=True)
assert rows, "no rows produced -- STOP"
con.execute("CREATE TABLE sup(s_id VARCHAR,r_id VARCHAR,e_id VARCHAR,s_label VARCHAR,e_label VARCHAR,tier VARCHAR)")
con.executemany("INSERT INTO sup VALUES (?,?,?,?,?,?)", rows)
con.execute("CREATE TABLE base AS SELECT * FROM read_parquet(?)", [OUT])
con.execute("""COPY (
  SELECT s_id,r_id,e_id, any_value(s_label) s_label, any_value(e_label) e_label, any_value(tier) tier
  FROM (SELECT * FROM base UNION ALL SELECT * FROM sup) GROUP BY s_id,r_id,e_id
) TO ? (FORMAT parquet)""", [OUT])
n=con.execute("SELECT count(*) FROM read_parquet(?)",[OUT]).fetchone()[0]
print("merged edges:", n, flush=True)
print("Q264164 rows:", con.execute("SELECT tier,s_label,r_id,e_label FROM read_parquet(?) WHERE s_id=?",[OUT,TARGET]).fetchall(), flush=True)
print("DONE_TOPUP_RECURSION", flush=True)
