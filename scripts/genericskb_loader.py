#!/usr/bin/env python3
"""OP-CONCLUDE Tier-0 GenericsKB-Best loader (CC-BY-4.0, AI2).

Emits HasProperty edges ONLY -- never IsA. Category statements ('a dog is a
canid', 'Dog isa animal') are SKIPPED (WordNet owns the taxonomy; emitting them
as HasProperty would be wrong and as IsA would risk drift). Property statements
('dogs are loyal', 'a dog has fur', 'birds can fly' -> capability mapped to
HasProperty) become subject-node -> HasProperty -> property-node, Confidence
Inferred, corpus=genericskb.

Sense-pin (crux): each subject lemma -> its per-synset node identity
(wordnet_loader.synset_label), disambiguated by gloss-overlap between the
generic sentence and each candidate NOUN synset's gloss; fallback = primary
sense (bare lemma). HasProperty is NOT traversed by the IS-A category walk
(is_isa_bond), so a mis-pin can only put an off property on a sense -- never
flip a category conclusion.

Density guard (thesaurus lesson): <=20 properties per subject (top score) AND
<=50 subjects per property-object (reverse-hub) so 'loyal'/'furry' can't become
high-in-degree hubs that break recall spread.
"""
import os, sys, re, time

TSV_DEFAULT = "/var/lib/lu/data/genericskb/generics_kb_best.tsv"
WN_DIR = "/var/lib/lu/data/wordnet/prolog"
SCORE_MIN = float(os.environ.get("GKB_SCORE_MIN", "0.5"))
MAX_PROPS_PER_SUBJECT = int(os.environ.get("GKB_MAX_OUT", "20"))
MAX_SUBJECTS_PER_PROP = int(os.environ.get("GKB_MAX_IN", "50"))
MAX_PROP_TOKENS = 3

RELATION = "HasProperty"
assert RELATION != "IsA", "genericskb must never emit IsA (drift-safety valve)"

_STOP = set("a an the of to in on at for and or but is are was were be been being "
            "they them their it its this that these those with as from by not do does "
            "can could will would may might have has had also often usually generally "
            "typically sometimes most many some all any".split())

# ---- sense-pin map: lemma -> [(synset_label, gloss_token_set), ...] (nouns) ----

def build_sense_map():
    sys.path.insert(0, "/opt/lu/lu/merge-main/scripts")
    import wordnet_loader as wl
    syn = wl.load_synsets(WN_DIR)
    glosses = wl.load_glosses(WN_DIR)
    lemma_map = {}
    for sid, d in syn.items():
        if d["pos"] != "n":
            continue
        lbl = wl.synset_label(d)
        g = glosses.get(sid, "")
        gtok = set(w for w in re.split(r"[^a-z0-9]+", g.lower()) if len(w) >= 3 and w not in _STOP)
        for _wn, w, _se in d["lemmas"]:
            lemma_map.setdefault(w.lower(), []).append((lbl, gtok))
    return lemma_map


def sense_pin(term, sentence, lemma_map):
    cands = lemma_map.get(term.lower())
    if not cands:
        return None
    if len(cands) == 1:
        return cands[0][0]
    stok = set(w for w in re.split(r"[^a-z0-9]+", sentence.lower()) if len(w) >= 3 and w not in _STOP)
    best, best_score = None, -1
    for lbl, gtok in cands:
        ov = len(stok & gtok)
        if ov > best_score:
            best, best_score = lbl, ov
    if best_score <= 0:
        # no gloss signal -> primary sense: the bare-lemma label if present, else first
        for lbl, _ in cands:
            if lbl.lower() == term.lower():
                return lbl
        return cands[0][0]
    return best

# ---- property parse: sentence -> property phrase or None ----

_ARTICLES = {"a", "an", "the"}


def parse_property(term, sentence):
    KIND_ATTR, KIND_PART, KIND_CAP, KIND_BEH = 4, 3, 2, 1
    s = sentence.strip().rstrip(".").strip()
    toks = re.split(r"\s+", s.lower())
    if not toks:
        return []
    i = 0
    if toks[i] in _ARTICLES:
        i += 1
    t = term.lower()
    if i < len(toks) and toks[i] in (t, t + "s", t + "es"):
        i += 1
    pred = toks[i:]
    if not pred:
        return []
    head, rest = pred[0], pred[1:]
    if head == "isa":
        return []
    if head in ("is", "are", "was", "were", "become", "becomes"):
        if rest and rest[0] in _ARTICLES:
            return []  # category: is a canid
        if len(rest) >= 2 and rest[0] == "capable" and rest[1] == "of":
            body, kind = rest[2:], KIND_CAP
        else:
            body, kind = rest, KIND_ATTR
    elif head in ("has", "have", "possess", "possesses", "contains", "contain"):
        body, kind = rest, KIND_PART
    elif head in ("can", "could", "cannot"):
        body, kind = rest, KIND_CAP
    else:
        body, kind = pred, KIND_BEH
    props, cur = [], []
    for w in body:
        if w in ("and", "or") or w.endswith(","):
            w2 = w.rstrip(",")
            if w2 and w2 not in ("and", "or"):
                cur.append(w2)
            if cur:
                props.append(cur); cur = []
            continue
        cur.append(w)
    if cur:
        props.append(cur)
    out = []
    for p in props:
        content = [w for w in p if w not in _ARTICLES and w not in _STOP and re.fullmatch(r"[a-z][a-z-]+", w)]
        content = content[:MAX_PROP_TOKENS]
        if not content:
            continue
        phrase = " ".join(content)
        if phrase == t or phrase in _STOP:
            continue
        out.append((phrase, kind))
    return out


def iter_property_bonds(tsv=TSV_DEFAULT, limit=None):
    lemma_map = build_sense_map()
    # PASS 1: collect valid (label, prop, score)
    per_subject = {}  # label -> {prop: best_score}
    n_read = 0
    with open(tsv, encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            term, score_s, sentence = parts[0], parts[1], parts[2]
            try:
                score = float(score_s)
            except ValueError:
                continue
            if score < SCORE_MIN:
                continue
            props = parse_property(term, sentence)
            if not props:
                continue
            label = sense_pin(term, sentence, lemma_map)
            if not label:
                continue
            d = per_subject.setdefault(label, {})
            for p, kind in props:
                if p == label.lower():
                    continue
                cur = d.get(p)
                if cur is None or (kind, score) > cur:
                    d[p] = (kind, score)
            n_read += 1
            if limit and n_read >= limit:
                break
    # PASS 2: density caps + emit
    prop_indeg = {}
    emitted = 0
    for label, props in per_subject.items():
        top = sorted(props.items(), key=lambda kv: (kv[1][0], kv[1][1], -len(kv[0].split())), reverse=True)[:MAX_PROPS_PER_SUBJECT]
        for prop, (kind, score) in top:
            if prop_indeg.get(prop, 0) >= MAX_SUBJECTS_PER_PROP:
                continue  # reverse-hub guard
            prop_indeg[prop] = prop_indeg.get(prop, 0) + 1
            yield {"subject": label, "relation": RELATION, "object": prop,
                   "weight": 0.85, "confidence": "Inferred"}
            emitted += 1


def ingest_genericskb(client, cp, smoke_limit=None, tsv=TSV_DEFAULT):
    import ingest_curriculum_full as ic
    corpus_id = "genericskb"
    ic.log_progress(f"genericskb: starting {tsv} smoke={bool(smoke_limit)}")
    t0 = time.time()
    sink = ic.BatchSink(client, ic.Throttle(getattr(ic, "THROTTLE_CODING", 30.0)))
    run_id = getattr(ic, "RUN_ID_TIER1", None) or "genericskb"
    count = 0
    non_hasproperty = 0
    for entry in iter_property_bonds(tsv, limit=smoke_limit):
        if entry["relation"] != "HasProperty":
            non_hasproperty += 1
            continue
        entry["corpus"] = corpus_id
        entry["run_id"] = run_id
        sink.add(entry)
        count += 1
        if count % 50000 == 0:
            ic.log_progress(f"genericskb: bonds={count} landed={sink.bonds_landed} "
                            f"rejected={sink.bonds_rejected} errors={sink.bonds_errors} "
                            f"rate={count/max(1e-6, time.time()-t0):.0f}/s")
    sink.flush()
    ic.log_progress(f"genericskb: DONE bonds={count} landed={sink.bonds_landed} "
                    f"rejected={sink.bonds_rejected} errors={sink.bonds_errors} "
                    f"non_hasproperty_blocked={non_hasproperty}")
    if cp is not None:
        cp[corpus_id] = {"bonds": count, "landed": sink.bonds_landed, "completed": True}
    return {"bonds": count, "landed": sink.bonds_landed, "rejected": sink.bonds_rejected,
            "errors": sink.bonds_errors, "non_hasproperty_blocked": non_hasproperty}


if __name__ == "__main__":
    os.environ.setdefault("LU4D_BIN", "/opt/lu/lu/merge-main/target/release/lu4d")
    os.environ.setdefault("LU_CANON_INGEST", "1")
    sys.path.insert(0, "/opt/lu/lu/merge-main/scripts")
    import ingest_curriculum_full as ic
    ic.CHECKPOINT = "/tmp/genericskb_ckpt.json"
    limit = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else None
    brain = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser("~/lu/genericskb_v5-sst")
    print(f"GENERICSKB limit={limit} brain={brain}", flush=True)
    client = ic.McpClient(brain, "/var/lib/lu/master.key", log_stderr_to="/tmp/genericskb_mcp.err", ingest_only=True)
    t = time.time()
    r = ingest_genericskb(client, {}, smoke_limit=limit, tsv=TSV_DEFAULT)
    try:
        print("FLUSH:", str(client.call("lu4d_flush", {}))[:60], flush=True)
    except Exception as e:
        print("flush err", e)
    try:
        client.close()
    except Exception:
        pass
    print("RESULT", r, "elapsed=%.1fs" % (time.time()-t), flush=True)
