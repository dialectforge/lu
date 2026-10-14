#!/usr/bin/env python3
"""OP-BREADTH two-tier ingest (OP-SENSE-CONCEPT-QID fix) of the clean edge parquet
into a COPY (offline). Three sequential lean MCP children into the SAME brain dir:

  1. CONCEPT-PLAIN pass  (LU_CANON_INGEST=1, fold + concept-admission gate ON):
     concept-tier rows whose SUBJECT is MULTI-WORD -> plain fold-canonical concept
     nodes (anti-gravel; pilot-proven, e.g. "hash table", "data structure").

  2. CONCEPT-SENSE pass  (LU_CANON_INGEST=0, no fold, no gate):
     concept-tier rows whose SUBJECT is SINGLE-WORD -> a QID-anchored shelf node
     "<Label>#<QID>" (mirrors the entity shelf; segregated into the e_/eq keyword
     namespace by entity_qid_of) so a polysemous sense ("Python" #Q28865) NEVER
     label-folds onto the pre-existing everyday node (the WordNet snake keeps the
     bare "python" bucket). Plus, per distinct single-word subject, ONE option-2
     handle bond  csense<token> --IsA--> <Label>#<QID>  so the runtime sense-router
     (OP-SENSE-CONCEPT-QID) can 1-hop a definitional/language frame into the sense.

  3. ENTITY pass  (LU_CANON_INGEST=0): label#QID shelf nodes, unchanged.

This pass ONLY ADDS nodes -- it never folds a concept subject onto a foundation
node -- so the foundation and the passing acceptance set are preserved by
construction. Uses the CURRENT entity-aware binary target/release/lu4d (guard-
checked by caller). --dry-run N prints sample bonds per pass (+ coverage QIDs).
"""
import os, sys, time, argparse, re
import duckdb
sys.path.insert(0, "/opt/lu/lu/merge-main/scripts")

BIN="/opt/lu/lu/merge-main/target/release/lu4d"
PID_BT = {
 'P31':'IsA','P279':'IsA','P106':'IsA','P17':'LocatedIn','P131':'LocatedIn','P30':'LocatedIn',
 'P36':'HasCapital','P37':'HasOfficialLanguage','P800':'KnownFor','P101':'KnownFor',
 'P361':'PartOf','P463':'PartOf','P527':'ComposedOf','P186':'ComposedOf','P19':'BornIn',
 'P20':'DiedIn','P50':'WrittenBy','P170':'WrittenBy','P61':'DiscoveredBy','P112':'FoundedBy',
 'P138':'NamedAfter','P1552':'HasProperty','P828':'Causes','P366':'UsedFor'}
COVERAGE={'Q937':'Einstein','Q8366':'algorithm','Q28865':'Python','Q142':'France',
 'Q264164':'recursion','Q207440':'hash table','Q175263':'data structure','Q80585':'associative array'}

def concept_node(label):
    if not label: return None
    l=label.strip()
    if not l or len(l)>120: return None
    if l[0]=='Q' and l[1:].isdigit(): return None
    return l

def ent_node(label,qid):
    l=concept_node(label)
    if not l: return None
    return "%s#%s" % (l,qid)

def is_single_word(label):
    l=(label or '').strip()
    return bool(l) and (' ' not in l) and ('\t' not in l)

def csense_token(label):
    """Match the runtime router's tokenisation of the bare word: lowercase,
    strip non-alphanumerics, require len>=2 (router filter)."""
    t=re.sub(r'[^a-z0-9]','', (label or '').lower())
    return t if len(t)>=2 else None

def mk(subj,rel,obj,run):
    if not subj or not obj or subj==obj: return None
    return {"subject":subj,"relation":rel,"object":obj,"weight":0.7,
            "confidence":"Unverified","corpus":"wikidata","run_id":"wikidata_twotier_%s"%run}

def concept_plain_bond(s_id,r_id,e_id,s_label,e_label):
    # MULTI-WORD concept subjects only (single-word go to the sense pass).
    if is_single_word(s_label): return None
    rel=PID_BT.get(r_id)
    if not rel: return None
    return mk(concept_node(s_label), rel, concept_node(e_label), "concept")

def sense_bond(s_id,r_id,e_id,s_label,e_label):
    # SINGLE-WORD concept subjects -> QID-anchored shelf subject.
    if not is_single_word(s_label): return None
    rel=PID_BT.get(r_id)
    if not rel: return None
    return mk(ent_node(s_label,s_id), rel, concept_node(e_label), "sense")

def entity_bond(s_id,r_id,e_id,s_label,e_label):
    rel=PID_BT.get(r_id)
    if not rel: return None
    return mk(ent_node(s_label,s_id), rel, ent_node(e_label,e_id), "entity")

def load_rows(parquet):
    con=duckdb.connect()
    return con.execute("SELECT s_id,r_id,e_id,s_label,e_label,tier FROM read_parquet(?) ORDER BY tier",[parquet]).fetchall()

def handle_bonds(rows):
    """One csense<token> --IsA--> <Label>#<QID> handle per distinct single-word
    concept subject that has a routable token. Emitted in the canon=0 sense pass."""
    seen={}
    for s_id,r_id,e_id,s_label,e_label,rt in rows:
        if rt!='concept' or not is_single_word(s_label): continue
        if not PID_BT.get(r_id): continue
        tok=csense_token(s_label)
        shelf=ent_node(s_label,s_id)
        if not tok or not shelf: continue
        key=(tok,shelf)
        if key in seen: continue
        seen[key]=True
    out=[]
    for (tok,shelf) in seen:
        b=mk("csense"+tok, "IsA", shelf, "handle")
        if b: out.append(b)
    return out

def run_pass(name, bonds_iter, brain, keyfile, canon):
    import importlib
    os.environ["LU4D_BIN"]=BIN
    os.environ["LU_CANON_INGEST"]=canon
    ic=importlib.import_module("ingest_curriculum_full")
    client=ic.McpClient(brain, keyfile, log_stderr_to="/tmp/wd_twotier_%s.err"%name, ingest_only=True)
    sink=ic.BatchSink(client, ic.Throttle(getattr(ic,"THROTTLE_CODING",30.0)))
    t0=time.time(); count=0; skipped=0
    for b in bonds_iter:
        if not b: skipped+=1; continue
        sink.add(b); count+=1
    sink.flush()
    try: fl=str(client.call("lu4d_flush",{}))[:80]
    except Exception as e: fl="flush_err %s"%e
    dt=time.time()-t0
    client.close()
    rate=(sink.bonds_landed/dt) if dt>0 else 0
    print("=== PASS %s canon=%s ==="%(name,canon), flush=True)
    print("  sent=%d landed=%d existing=%d gated=%d rejected=%d errors=%d dedup=%d prefilter_skipped=%d"%(
        count, sink.bonds_landed, sink.bonds_existing, sink.bonds_gated, sink.bonds_rejected,
        sink.bonds_errors, sink.bonds_skipped_dedup, skipped), flush=True)
    print("  wall_sec=%.1f landed_bonds_per_sec=%.1f flush=%s"%(dt, rate, fl), flush=True)
    return {"name":name,"sent":count,"landed":sink.bonds_landed,"existing":sink.bonds_existing,
            "gated":sink.bonds_gated,"rejected":sink.bonds_rejected,"errors":sink.bonds_errors,
            "dedup":sink.bonds_skipped_dedup,"skipped":skipped,"wall":dt,"rate":rate}

def dry_run(rows, n):
    print("=== DRY RUN (no send) ===", flush=True)
    def show(label, gen):
        print("--- %s ---"%label, flush=True); c=0
        for b in gen:
            if not b: continue
            print("   %-34s --%-14s--> %-34s"%(b['subject'],b['relation'],b['object']), flush=True)
            c+=1
            if c>=n: break
    show("concept-plain (multi-word, canon=1)", (concept_plain_bond(*r[:5]) for r in rows if r[5]=='concept'))
    show("concept-sense (single-word shelf, canon=0)", (sense_bond(*r[:5]) for r in rows if r[5]=='concept'))
    show("csense handles (canon=0)", iter(handle_bonds(rows)))
    show("entity (canon=0)", (entity_bond(*r[:5]) for r in rows if r[5]=='entity'))
    print("--- coverage-QID sense check ---", flush=True)
    for s_id,r_id,e_id,s_label,e_label,rt in rows:
        if s_id in COVERAGE:
            b=sense_bond(s_id,r_id,e_id,s_label,e_label) if is_single_word(s_label) else concept_plain_bond(s_id,r_id,e_id,s_label,e_label)
            if not b and rt=='entity': b=entity_bond(s_id,r_id,e_id,s_label,e_label)
            if b: print("   %-9s [%s] %s"%(s_id,rt,"%s --%s--> %s"%(b['subject'],b['relation'],b['object'])), flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("brain")
    ap.add_argument("--parquet", default="/tmp/wd_twotier_edges.parquet")
    ap.add_argument("--keyfile", default="/var/lib/lu/master.key")
    ap.add_argument("--dry-run", type=int, default=0)
    a=ap.parse_args()
    rows=load_rows(a.parquet)
    nc=sum(1 for r in rows if r[5]=='concept'); ne=sum(1 for r in rows if r[5]=='entity')
    nsw=sum(1 for r in rows if r[5]=='concept' and is_single_word(r[3]))
    print("loaded %d rows (concept=%d [single-word-subj rows=%d] entity=%d) parquet=%s"%(len(rows),nc,nsw,ne,a.parquet), flush=True)
    if a.dry_run:
        dry_run(rows, a.dry_run); sys.exit(0)
    T0=time.time()
    print("BINARY:", BIN, flush=True)
    handles=handle_bonds(rows)
    print("csense handles to mint: %d"%len(handles), flush=True)
    r1=run_pass('concept', (concept_plain_bond(*r[:5]) for r in rows if r[5]=='concept'), a.brain, a.keyfile, "1")
    # sense pass: shelf edges for single-word concept subjects THEN the handles.
    def sense_stream():
        for r in rows:
            if r[5]=='concept':
                yield sense_bond(*r[:5])
        for h in handles:
            yield h
    r2=run_pass('sense', sense_stream(), a.brain, a.keyfile, "0")
    r3=run_pass('entity', (entity_bond(*r[:5]) for r in rows if r[5]=='entity'), a.brain, a.keyfile, "0")
    tot=time.time()-T0
    tl=r1['landed']+r2['landed']+r3['landed']
    print("=== TOTAL ===", flush=True)
    print("  landed_total=%d wall_total_sec=%.1f agg_landed_bonds_per_sec=%.1f"%(tl,tot,(tl/tot) if tot>0 else 0), flush=True)
    print("DONE_TWOTIER_INGEST", flush=True)
