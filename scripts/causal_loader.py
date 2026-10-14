#!/usr/bin/env python3
"""OP-CONCLUDE Tier 1.3 causal loader: Quasimodo (CC-BY-2.0) causal predicates
-> Causes / LeadsTo edges. Reuses the properties_loader pipeline (typicality
rank, WordNet-head cleaning, sense-pin, density caps). Concept->concept only.

Forward (subject causes object): cause/can cause/result in/induce/trigger/
be cause of -> Causes ;  lead/lead to/can lead to/has_effect -> LeadsTo.
Reverse (subject is the effect): be caused by/result from -> swap to Causes.

Sense-pins BOTH cause and effect onto per-synset nodes. Confidence Inferred;
corpus quasimodo-causal; caps out<=CAUSAL_MAX_OUT / in<=CAUSAL_MAX_IN.
Single-source typicality floor CAUSAL_MIN (0.60). NOT ingested in dry-run.
"""
import os, sys, time
from collections import defaultdict
sys.path.insert(0, "/opt/lu/lu/merge-main/scripts")
import properties_loader as PL

MAX_OUT = int(os.environ.get("CAUSAL_MAX_OUT", "20"))
MAX_IN = int(os.environ.get("CAUSAL_MAX_IN", "50"))
CAUSAL_MIN = float(os.environ.get("CAUSAL_MIN", "0.60"))

FWD_LEADS = {"lead", "leads", "lead to", "leads to", "can lead to", "can lead", "has_effect"}
FWD_CAUSES = {"cause", "causes", "can cause", "result in", "results in", "induce",
              "induces", "can induce", "trigger", "triggers", "be cause of"}
REV = {"be caused by", "be caused", "result from", "results from", "be result of"}


def rel_and_dir(pred, subj, obj):
    """Return (cause, effect, relation) with cause->effect forward."""
    if pred in REV:
        return obj, subj, "Causes"
    if pred in FWD_LEADS:
        return subj, obj, "LeadsTo"
    return subj, obj, "Causes"


def read_causal(tsv, vocab):
    with open(tsv, encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").rstrip("\r").split("\t")
            if len(p) < 4:
                continue
            subj, pred, obj, typ = p[0].lower(), p[1], p[2].lower(), p[3]
            cause, effect, rel = rel_and_dir(pred, subj, obj)
            c = PL.clean_property(cause, vocab)
            e = PL.clean_property(effect, vocab)
            if not c or not e or c == e:
                continue
            try:
                score = float(typ)
            except ValueError:
                score = 0.0
            yield c, rel, e, score


def merged(tsv):
    vocab, lemma_noun, _ = PL.build_vocab_and_senses()
    # (cause, rel, effect) -> max typicality
    agg = {}
    for c, rel, e, sc in read_causal(tsv, vocab):
        k = (c, rel, e)
        if sc > agg.get(k, 0.0):
            agg[k] = sc
    # threshold + caps
    out_deg = defaultdict(int)
    in_deg = defaultdict(int)
    edges = []
    for (c, rel, e), sc in sorted(agg.items(), key=lambda x: -x[1]):
        if sc < CAUSAL_MIN:
            continue
        clab = PL.sense_pin(c, lemma_noun)
        elab = PL.sense_pin(e, lemma_noun)
        if clab == elab:
            continue
        if out_deg[clab] >= MAX_OUT or in_deg[elab] >= MAX_IN:
            continue
        out_deg[clab] += 1
        in_deg[elab] += 1
        edges.append((clab, rel, elab, round(sc, 2)))
    return edges


if __name__ == "__main__":
    tsv = os.environ.get("CAUSAL_TSV", "/tmp/quasimodo_causal.tsv")
    edges = merged(tsv)
    print("TOTAL causal edges after threshold+caps:", len(edges))
    byrel = defaultdict(int)
    for _, r, _, _ in edges:
        byrel[r] += 1
    print("by relation:", dict(byrel))
    # index for the dry-run questions
    outmap = defaultdict(list)  # cause -> [(rel,effect,score)]
    inmap = defaultdict(list)   # effect -> [(cause,rel,score)]
    for c, rel, e, sc in edges:
        outmap[c].append((rel, e, sc))
        inmap[e].append((c, rel, sc))
    TESTS = [
        ("rust", "in"), ("fire", "in"), ("smoking", "out"), ("rain", "in"),
        ("cancer", "in"), ("gravity", "in"), ("fall", "in"), ("cold", "in"),
        ("exercise", "out"), ("stress", "out"),
    ]
    for concept, direction in TESTS:
        print("\n########## %s  (%s) ##########" % (concept.upper(),
              "what causes X" if direction == "in" else "what does X cause"))
        if direction == "in":
            rows = sorted(inmap.get(concept, []), key=lambda x: -x[2])[:12]
            if not rows:
                print("   (no causal edges)")
            for c, rel, sc in rows:
                print("   %-24s -%s-> %-18s  %.2f" % (c, rel, concept, sc))
        else:
            rows = sorted(outmap.get(concept, []), key=lambda x: -x[2])[:12]
            if not rows:
                print("   (no causal edges)")
            for rel, e, sc in rows:
                print("   %-18s -%s-> %-24s  %.2f" % (concept, rel, e, sc))
# =====================================================================
# Tier 1.3 ingest driver (post-review). Reads the CLEANED corroborated
# causal set + applies the PERMANENT health-misinfo denylist + relabels.
# STANDING RULE: contested socio-political causal claims are excluded.
# =====================================================================
import time as _time

CLEAN_TSV = "/var/lib/lu/data/causal_cleaned.tsv"
DENY_TSV = "/var/lib/lu/data/causal_misinfo_denylist.tsv"
CAUSAL_INGEST_WEIGHT = 0.85  # corroborated + human-reviewed; Confidence Inferred
ING_MAX_OUT = int(os.environ.get("CAUSAL_MAX_OUT", "30"))
ING_MAX_IN = int(os.environ.get("CAUSAL_MAX_IN", "50"))

RELABEL = {
    "vomit.n.3": "vomiting", "cavity.n.3": "cavity", "itch.n.3": "itching",
    "sterility.n.2": "infertility", "fleshiness": "weight gain", "medicine.n.2": "medicine",
    "smoke.n.7": "smoke", "canker.n.2": "canker sore", "pill.n.5": "pill", "snore.n.2": "snoring",
}


def load_denylist(path=DENY_TSV):
    deny = set()
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                p = line.rstrip("\n").split("\t")
                if len(p) >= 2:
                    deny.add((p[0].strip().lower(), p[1].strip().lower()))
    except FileNotFoundError:
        pass
    return deny


def ingest_causal(brain, tsv=CLEAN_TSV, keyfile="/var/lib/lu/master.key"):
    os.environ.setdefault("LU4D_BIN", "/opt/lu/lu/merge-main/target/release/lu4d")
    os.environ.setdefault("LU_CANON_INGEST", "1")
    sys.path.insert(0, "/opt/lu/lu/merge-main/scripts")
    import ingest_curriculum_full as ic
    deny = load_denylist()
    out_deg = defaultdict(int); in_deg = defaultdict(int)
    entries = []
    with open(tsv, encoding="utf-8") as f:
        first = True
        for line in f:
            if first:
                first = False
                if line.lower().startswith("subject"):
                    continue
            p = line.rstrip("\n").split("\t")
            if len(p) < 3:
                continue
            s, rel, o = p[0].strip(), p[1].strip(), p[2].strip()
            s = RELABEL.get(s, s); o = RELABEL.get(o, o)
            if not s or not o or s.lower() == o.lower():
                continue
            if (s.lower(), o.lower()) in deny:          # PERMANENT misinfo denylist
                continue
            if rel not in ("Causes", "LeadsTo"):
                rel = "Causes"
            if out_deg[s] >= ING_MAX_OUT or in_deg[o] >= ING_MAX_IN:
                continue
            out_deg[s] += 1; in_deg[o] += 1
            entries.append({"subject": s, "relation": rel, "object": o,
                            "weight": CAUSAL_INGEST_WEIGHT, "confidence": "Inferred",
                            "corpus": "quasimodo-causal", "run_id": "causal_qa_v1"})
    print("causal: prepared %d edges (deny=%d applied)" % (len(entries), len(deny)), flush=True)
    client = ic.McpClient(brain, keyfile, log_stderr_to="/tmp/causal_mcp.err", ingest_only=True)
    sink = ic.BatchSink(client, ic.Throttle(getattr(ic, "THROTTLE_CODING", 30.0)))
    t0 = _time.time(); count = 0
    try:
        for e in entries:
            sink.add(e); count += 1
        sink.flush()
        print("causal: FLUSH_ALL sent=%d landed=%d rejected=%d errors=%d dedup=%d" % (
            count, sink.bonds_landed, sink.bonds_rejected, sink.bonds_errors,
            sink.bonds_skipped_dedup), flush=True)
        try:
            print("FLUSH:", str(client.call("lu4d_flush", {}))[:80], flush=True)
        except Exception as ex:
            print("flush err", ex, flush=True)
    finally:
        try: client.close()
        except Exception: pass
    print("causal: DONE edges=%d landed=%d %.0fs" % (count, sink.bonds_landed, _time.time() - t0), flush=True)
    return {"edges": count, "landed": sink.bonds_landed}
