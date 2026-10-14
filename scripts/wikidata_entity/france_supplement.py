#!/usr/bin/env python3
"""Fill CleverThis holes (e.g. France Q142 absent) with rank-filtered (preferred>
normal) wbgetentities statements. ONLY adds edges for subjects that have 0 rows
in the base extract, so present entities (Paris/Einstein/Newton) are untouched.
CC0 (Wikidata). Merges into /tmp/wd_famous_edges.parquet.
"""
import time, json, urllib.request, urllib.parse
import duckdb
UA="Lu4d-entity-ingest/1.0 (info@dialectforge.com)"
API="https://www.wikidata.org/w/api.php"
OUT="/tmp/wd_famous_edges.parquet"
# defensive curated famous set (countries/cities/scientists/planets/elements) -
# only those ABSENT from the base actually get added.
CHECK=["Q142","Q937","Q90","Q935","Q30","Q145","Q183","Q159","Q148","Q17","Q408",
 "Q64","Q84","Q60","Q65","Q1490","Q220","Q1741","Q1963","Q2044","Q7186",
 "Q307","Q619","Q525","Q111","Q308","Q313","Q405","Q560","Q568","Q629",
 "Q23","Q76","Q1124","Q34266","Q29","Q38","Q39","Q40","Q45","Q189"]
WHITELIST=["P31","P106","P279","P17","P131","P30","P36","P800","P101","P19",
 "P20","P37","P361","P463","P527","P186","P50","P170","P61","P112",
 "P138","P1552","P828","P366"]
PER=8; OUTCAP=25
def get(ids,props):
    u=API+"?"+urllib.parse.urlencode({"action":"wbgetentities","ids":"|".join(ids),"props":props,"languages":"en","format":"json"})
    for a in range(6):
        try:
            r=urllib.request.Request(u,headers={"User-Agent":UA})
            return json.loads(urllib.request.urlopen(r,timeout=60).read())
        except Exception: time.sleep(2*(a+1))
    return None
# which CHECK qids are holes (0 rows in base)?
con=duckdb.connect()
base_ids=set(r[0] for r in con.execute("SELECT DISTINCT s_id FROM read_parquet(?)",[OUT]).fetchall())
holes=[q for q in dict.fromkeys(CHECK) if q not in base_ids]
print("holes to fill:", holes)
if not holes:
    print("no holes; nothing to do"); raise SystemExit
lm={}; edges=[]; objs=set()
for i in range(0,len(holes),50):
    d=get(holes[i:i+50],"claims|labels")
    if not d or "entities" not in d: continue
    for qid,ent in d["entities"].items():
        if "missing" in ent: continue
        l=ent.get("labels",{}).get("en")
        if l: lm[qid]=l["value"]
        cl=ent.get("claims",{}); oc=0
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
    rows.append((sid,pid,oid,sl,ol))
print("supplement rows:", len(rows))
con.execute("CREATE TABLE sup(s_id VARCHAR,r_id VARCHAR,e_id VARCHAR,s_label VARCHAR,e_label VARCHAR)")
con.executemany("INSERT INTO sup VALUES (?,?,?,?,?)", rows)
con.execute("CREATE TABLE base AS SELECT * FROM read_parquet(?)", [OUT])
con.execute("""COPY (
  SELECT s_id,r_id,e_id,any_value(s_label) s_label,any_value(e_label) e_label
  FROM (SELECT * FROM base UNION ALL SELECT * FROM sup) GROUP BY s_id,r_id,e_id
) TO ? (FORMAT parquet)""", [OUT])
n=con.execute("SELECT count(*) FROM read_parquet(?)",[OUT]).fetchone()[0]
print("merged edges:", n)
for q in ["Q142","Q90"]:
    print(q, con.execute("SELECT s_label,r_id,e_label FROM read_parquet(?) WHERE s_id=? AND r_id IN ('P36','P17','P37') LIMIT 8",[OUT,q]).fetchall())
print("DONE_SUP")
