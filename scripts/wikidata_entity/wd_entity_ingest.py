#!/usr/bin/env python3
"""OP-BREADTH Phase-1 ENTITY SUBSYSTEM ingest. Every Wikidata endpoint
(subject AND object) becomes a disjoint entity node labelled 'label#QID'.
Nothing merges onto a concept node -> the concept graph gains ZERO nodes
and ZERO edges. The rebuilt keyword index (entity-aware walk) indexes
these nodes ONLY under the 'e_<word>' namespace + 'eq<qid>'. Confidence
Unverified; corpus wikidata. PID -> BondType per the whitelist.
"""
import os, sys, time
import duckdb
sys.path.insert(0, "/opt/lu/lu/merge-main/scripts")

PID_BT = {
 'P31':'IsA','P279':'IsA','P106':'IsA','P17':'LocatedIn','P131':'LocatedIn','P30':'LocatedIn',
 'P36':'HasCapital','P37':'HasOfficialLanguage','P800':'KnownFor','P101':'KnownFor',
 'P361':'PartOf','P463':'PartOf','P527':'ComposedOf','P186':'ComposedOf','P19':'BornIn',
 'P20':'DiedIn','P50':'WrittenBy','P170':'WrittenBy','P61':'DiscoveredBy','P112':'FoundedBy',
 'P138':'NamedAfter','P1552':'HasProperty','P828':'Causes','P366':'UsedFor'}


def ent_node(label, qid):
    if not label:
        return None
    l = label.strip()
    if not l or len(l) > 120:
        return None
    if l[0] == 'Q' and l[1:].isdigit():   # bare-QID with no human label -> skip
        return None
    return "%s#%s" % (l, qid)


def ingest(brain, tsv="/tmp/wd_famous_edges.parquet", keyfile="/var/lib/lu/master.key"):
    os.environ.setdefault("LU4D_BIN", "/opt/lu/lu/merge-main/target/release/lu4d")
    os.environ.setdefault("LU_CANON_INGEST", "0")
    import ingest_curriculum_full as ic
    con = duckdb.connect()
    rows = con.execute(
        "SELECT s_id,r_id,e_id,s_label,e_label FROM read_parquet('" + tsv + "')"
    ).fetchall()
    print("loaded %d edge rows" % len(rows), flush=True)
    client = ic.McpClient(brain, keyfile, log_stderr_to="/tmp/wd_ent_mcp.err", ingest_only=True)
    sink = ic.BatchSink(client, ic.Throttle(getattr(ic, "THROTTLE_CODING", 30.0)))
    t0 = time.time(); count = 0; skipped = 0
    try:
        for s_id, r_id, e_id, s_label, e_label in rows:
            rel = PID_BT.get(r_id)
            subj = ent_node(s_label, s_id)
            obj = ent_node(e_label, e_id)
            if not rel or not subj or not obj or subj == obj:
                skipped += 1
                continue
            sink.add({"subject": subj, "relation": rel, "object": obj, "weight": 0.7,
                      "confidence": "Unverified", "corpus": "wikidata", "run_id": "wikidata_entity_v1"})
            count += 1
            if count % 250000 == 0:
                print("sent=%d landed=%d rejected=%d dedup=%d %.0fs" % (
                    count, sink.bonds_landed, sink.bonds_rejected, sink.bonds_skipped_dedup,
                    time.time() - t0), flush=True)
        sink.flush()
        print("FLUSH_ALL sent=%d landed=%d rejected=%d errors=%d dedup=%d skipped=%d" % (
            count, sink.bonds_landed, sink.bonds_rejected, sink.bonds_errors,
            sink.bonds_skipped_dedup, skipped), flush=True)
        try:
            print("FLUSH:", str(client.call("lu4d_flush", {}))[:60], flush=True)
        except Exception as e:
            print("flush err", e, flush=True)
    finally:
        try: client.close()
        except Exception: pass
    print("DONE edges=%d landed=%d %.0fs" % (count, sink.bonds_landed, time.time() - t0), flush=True)
    return {"edges": count, "landed": sink.bonds_landed}


if __name__ == "__main__":
    brain = sys.argv[1]
    print("WIKIDATA ENTITY INGEST brain=%s" % brain, flush=True)
    print("RESULT", ingest(brain), flush=True)
