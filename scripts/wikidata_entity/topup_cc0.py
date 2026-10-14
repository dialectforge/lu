#!/usr/bin/env python3
"""CC0 hub top-up. For coverage/hub QIDs ABSENT from the two-tier base extract
(CleverThis truthy has holes: Python Q28865, France Q142, recursion Q131412, ...)
fetch descriptions + whitelisted-PID claims from the Wikidata wbgetentities API
(CC0, rank-filtered preferred>normal) and merge into the SAME 6-col edge parquet
(s_id,r_id,e_id,s_label,e_label,tier). ONLY subjects with 0 base rows are added,
so present entities are untouched. Tier is derived per-entity from its own
P31/P279 vs the concept allowlist (concept if it is a member/self of an
allowlisted class, else entity). CC0 only -- no CC-BY-SA.
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
# coverage targets that the design REQUIRES present, + a curated famous hub set.
COVERAGE=["Q28865","Q142","Q131412"]  # known holes (Python, France, recursion)
CHECK=COVERAGE+["Q937","Q8366","Q207440","Q175263","Q80585",  # rest of the gate
 "Q90","Q935","Q30","Q145","Q183","Q159","Q148","Q17","Q408","Q64","Q84",
 "Q60","Q65","Q1490","Q220","Q1741","Q1963","Q2044","Q7186","Q307","Q619",
 "Q525","Q111","Q308","Q313","Q405","Q560","Q568","Q629"]
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
holes=[q for q in dict.fromkeys(CHECK) if q not in base_ids]
print("base subjects:", len(base_ids), flush=True)
print("holes to fill:", holes, flush=True)
if not holes:
    print("no holes; nothing to do"); raise SystemExit

lm={}; edges=[]; objs=set(); tiers={}
for i in range(0,len(holes),50):
    d=get(holes[i:i+50],"claims|labels|descriptions")
    if not d or "entities" not in d: continue
    for qid,ent in d["entities"].items():
        if "missing" in ent: print("  MISSING", qid, flush=True); continue
        l=ent.get("labels",{}).get("en")
        if l: lm[qid]=l["value"]
        cl=ent.get("claims",{})
        # derive tier: concept if self is an allowlist class or any P31/P279 -> allowlist class
        is_concept = qid in CONCEPT_CLASSES
        for pid in ("P31","P279"):
            for s in cl.get(pid,[]):
                ms=s.get("mainsnak",{}); dv=ms.get("datavalue",{})
                if dv.get("type")=="wikibase-entityid":
                    if dv["value"].get("id") in CONCEPT_CLASSES: is_concept=True
        tiers[qid]="concept" if is_concept else "entity"
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
                edges.append((qid,pid,oid)); objs.add(oid); rc+=1; oc+=1
# labels for object QIDs not already known
miss=[o for o in objs if o not in lm]
for i in range(0,len(miss),50):
    d=get(miss[i:i+50],"labels")
    if d and "entities" in d:
        for qid,ent in d["entities"].items():
            l=ent.get("labels",{}).get("en")
            if l: lm[qid]=l["value"]
rows=[]
for sid,pid,oid in edges:
    sl=lm.get(sid); ol=lm.get(oid)
    if not sl or not ol or len(sl)>120 or len(ol)>120: continue
    rows.append((sid,pid,oid,sl,ol,tiers.get(sid,"entity")))
print("supplement rows:", len(rows), flush=True)
print("tiers:", {q:tiers[q] for q in holes if q in tiers}, flush=True)
con.execute("CREATE TABLE sup(s_id VARCHAR,r_id VARCHAR,e_id VARCHAR,s_label VARCHAR,e_label VARCHAR,tier VARCHAR)")
con.executemany("INSERT INTO sup VALUES (?,?,?,?,?,?)", rows)
con.execute("CREATE TABLE base AS SELECT * FROM read_parquet(?)", [OUT])
con.execute("""COPY (
  SELECT s_id,r_id,e_id, any_value(s_label) s_label, any_value(e_label) e_label, any_value(tier) tier
  FROM (SELECT * FROM base UNION ALL SELECT * FROM sup) GROUP BY s_id,r_id,e_id
) TO ? (FORMAT parquet)""", [OUT])
n=con.execute("SELECT count(*) FROM read_parquet(?)",[OUT]).fetchone()[0]
print("merged edges:", n, flush=True)
for q in ["Q28865","Q142","Q131412"]:
    print(q, con.execute("SELECT tier,s_label,r_id,e_label FROM read_parquet(?) WHERE s_id=? LIMIT 6",[OUT,q]).fetchall(), flush=True)
print("DONE_TOPUP", flush=True)
