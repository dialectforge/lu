#!/usr/bin/env python3
"""OP-CONCLUDE Tier-0 Moby Thesaurus loader (PUBLIC DOMAIN, Gutenberg #3202).

Emits synonym associations as RelatedTo edges ONLY -- NEVER IsA. RelatedTo is
INERT to the Tier-1.2 category-membership / common-ancestor walk (which follows
only {IsA, DefinedAs, Inherits, ExampleOf}), so imperfectly-sensed synonym
links CANNOT cause IS-A drift. No definitions. Enriches recall / association.

Direction: root -> RelatedTo -> synonym (querying a headword surfaces its
synonym cluster). Multi-word roots/synonyms become alias nodes; the canonical
admission gate (LU_CANON_INGEST) drops any junk.
"""
import os, sys, time

MOBY_DEFAULT = "/var/lib/lu/data/thesaurus/mthesaur.txt"

# HARD INVARIANT: this loader emits exactly ONE relation type.
RELATION = "RelatedTo"
assert RELATION != "IsA", "thesaurus must never emit IsA (drift-safety valve)"

# Cap synonyms per root to bound node degree -- Moby dumps common words with
# ~1000+ loose synonyms, which created recall super-hubs. Top-N by Moby order.
MAX_SYNS_PER_ROOT = int(os.environ.get("MOBY_MAX_SYNS", "25"))


def iter_thesaurus_bonds(path=MOBY_DEFAULT, limit=None):
    n = 0
    with open(path, encoding="latin-1") as f:
        for line in f:
            parts = [p.strip() for p in line.rstrip("\r\n").split(",")]
            parts = [p for p in parts if p]
            if len(parts) < 2:
                continue
            root = parts[0]
            seen = set()
            emitted = 0
            for syn in parts[1:]:
                if emitted >= MAX_SYNS_PER_ROOT:
                    break
                if not syn or syn.lower() == root.lower() or syn.lower() in seen:
                    continue
                seen.add(syn.lower())
                yield {"subject": root, "relation": RELATION, "object": syn,
                       "weight": 0.8, "confidence": "Verified"}
                emitted += 1
                n += 1
                if limit and n >= limit:
                    return


def ingest_thesaurus(client, cp, smoke_limit=None, path=MOBY_DEFAULT):
    import ingest_curriculum_full as ic
    corpus_id = "moby_thesaurus"
    ic.log_progress(f"thesaurus: starting {path} smoke={bool(smoke_limit)}")
    t0 = time.time()
    sink = ic.BatchSink(client, ic.Throttle(getattr(ic, "THROTTLE_CODING", 30.0)))
    run_id = getattr(ic, "RUN_ID_TIER1", None) or "moby_thesaurus"
    count = 0
    non_relatedto = 0
    for entry in iter_thesaurus_bonds(path, limit=smoke_limit):
        if entry["relation"] != "RelatedTo":
            non_relatedto += 1  # must stay 0 -- structural guard
            continue
        entry["corpus"] = corpus_id
        entry["run_id"] = run_id
        sink.add(entry)
        count += 1
        if count % 100000 == 0:
            ic.log_progress(
                f"thesaurus: bonds={count} landed={sink.bonds_landed} "
                f"rejected={sink.bonds_rejected} errors={sink.bonds_errors} "
                f"rate={count/max(1e-6, time.time()-t0):.0f}/s")
    sink.flush()
    ic.log_progress(
        f"thesaurus: DONE bonds={count} landed={sink.bonds_landed} "
        f"rejected={sink.bonds_rejected} errors={sink.bonds_errors} "
        f"non_relatedto_blocked={non_relatedto}")
    if cp is not None:
        cp[corpus_id] = {"bonds": count, "landed": sink.bonds_landed, "completed": True}
    return {"bonds": count, "landed": sink.bonds_landed,
            "rejected": sink.bonds_rejected, "errors": sink.bonds_errors,
            "non_relatedto_blocked": non_relatedto}


if __name__ == "__main__":
    os.environ.setdefault("LU4D_BIN", "/opt/lu/lu/merge-main/target/release/lu4d")
    os.environ.setdefault("LU_CANON_INGEST", "1")
    sys.path.insert(0, "/opt/lu/lu/merge-main/scripts")
    import ingest_curriculum_full as ic
    ic.CHECKPOINT = "/tmp/moby_ckpt.json"
    limit = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else None
    brain = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser("~/lu/moby_v5-sst")
    print(f"MOBY limit={limit} brain={brain}", flush=True)
    client = ic.McpClient(brain, "/var/lib/lu/master.key", log_stderr_to="/tmp/moby_mcp.err", ingest_only=True)
    t = time.time()
    r = ingest_thesaurus(client, {}, smoke_limit=limit, path=MOBY_DEFAULT)
    try:
        print("FLUSH:", str(client.call("lu4d_flush", {}))[:60], flush=True)
    except Exception as e:
        print("flush err", e)
    try:
        client.close()
    except Exception:
        pass
    print("RESULT", r, "elapsed=%.1fs" % (time.time()-t), flush=True)
