import os,sys,json,time,importlib,urllib.request,urllib.parse
sys.path.insert(0,"/opt/lu/lu/merge-main/scripts")
UA="Lu4d-srcpullB/1.0 (info@dialectforge.com)"
API="https://www.wikidata.org/w/api.php"
QIDS={"radioactivity":"Q11448","radioactive decay":"Q3332922","nuclear fission":"Q11429",
 "quantum field theory":"Q54505","wave-particle duality":"Q193068","quantum entanglement":"Q215675"}
def get(p):
    u=API+"?"+urllib.parse.urlencode(p)
    for a in range(6):
        try: return json.loads(urllib.request.urlopen(urllib.request.Request(u,headers={"User-Agent":UA}),timeout=60).read())
        except Exception: time.sleep(2*(a+1))
    return None
def parent_label(qid):
    e=get({"action":"wbgetentities","ids":qid,"props":"claims","format":"json"})["entities"][qid]
    cl=e.get("claims",{})
    pids=[]
    for pid in ("P279","P31"):
        for s in cl.get(pid,[]):
            dv=s.get("mainsnak",{}).get("datavalue",{})
            if dv.get("type")=="wikibase-entityid": pids.append(dv["value"]["id"])
    for pq in pids:
        d2=get({"action":"wbgetentities","ids":pq,"props":"labels","languages":"en","format":"json"})
        lab=d2["entities"][pq].get("labels",{}).get("en",{}).get("value","")
        if lab and len(lab)<=60: return lab
    return None
rows=[]
for label,qid in QIDS.items():
    pl=parent_label(qid)
    if not pl: print("  no parent:",label,qid); continue
    print("  %-24s IsA %s"%(label,pl)); rows.append((label,pl))
if sys.argv[-1]=="--dry": print("DRY"); sys.exit(0)
os.environ["LU4D_BIN"]="/opt/lu/lu/merge-main/target/release/lu4d"
os.environ["LU_CANON_INGEST"]="1"
ic=importlib.import_module("ingest_curriculum_full")
BR=sys.argv[1]
cl=ic.McpClient(BR,"/var/lib/lu/master.key",log_stderr_to="/tmp/concept_isa.err",ingest_only=True)
sink=ic.BatchSink(cl, ic.Throttle(getattr(ic,"THROTTLE_CODING",30.0)))
for (label,pl) in rows:
    sink.add({"subject":label,"relation":"IsA","object":pl,"weight":0.8,"confidence":"Verified","corpus":"wikidata","run_id":"srcpullB_concept_isa"})
sink.flush()
print("IsA landed=%d existing=%d rejected=%d errors=%d"%(sink.bonds_landed,sink.bonds_existing,sink.bonds_rejected,sink.bonds_errors))
cl.call("lu4d_flush",{})
cl.close()
print("DONE")
