import os, sys, argparse
sys.path.insert(0, "/opt/lu/lu/merge-main/scripts")
import pyarrow.parquet as pq
PMAP={"P106":"IsA","P31":"IsA","P800":"KnownFor","P19":"BornIn","P50":"AuthoredBy"}
ap=argparse.ArgumentParser()
ap.add_argument("--brain",required=True)
ap.add_argument("--keyfile",default="/var/lib/lu/master.key")
ap.add_argument("--parquet",default="/tmp/wd_thin_edges.parquet")
a=ap.parse_args()
df=pq.read_table(a.parquet).to_pandas()
import importlib
os.environ["LU4D_BIN"]="/opt/lu/lu/merge-main/target/release/lu4d"
os.environ["LU_CANON_INGEST"]="0"
ic=importlib.import_module("ingest_curriculum_full")
cl=ic.McpClient(a.brain,a.keyfile,log_stderr_to="/tmp/thin_ingest.err",ingest_only=True)
sink=ic.BatchSink(cl, ic.Throttle(getattr(ic,"THROTTLE_CODING",30.0)))
sent=0
for _,r in df.iterrows():
    rel=PMAP.get(r["r_id"])
    if not rel: continue
    subj=str(r["s_label"])+"#"+str(r["s_id"])
    obj=str(r["e_label"])+"#"+str(r["e_id"])
    sink.add({"subject":subj,"relation":rel,"object":obj,"weight":0.9,"confidence":"Verified","corpus":"wikidata","run_id":"thin_bug8"})
    sent+=1
sink.flush()
try: fl=str(cl.call("lu4d_flush",{}))[:80]
except Exception as ex: fl="flush_err "+str(ex)
print(f"THIN sent={sent} landed={sink.bonds_landed} existing={sink.bonds_existing} gated={sink.bonds_gated} rejected={sink.bonds_rejected} errors={sink.bonds_errors} flush={fl}",flush=True)
cl.close()
print("DONE_THIN",flush=True)
