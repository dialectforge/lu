#!/usr/bin/env python3
"""OP-RETRAIN-1 Tier-0 WordNet foundation loader (with glosses).

Maps Princeton WordNet (Prolog) STRUCTURE + DEFINITIONS onto Lu's schema,
bypassing the regex extractor:

  PASS 1 (definitions): every lemma -> concept node carrying its synset GLOSS
    as knowledge.content Definition(...) via lu4d_set_definition_batch. This
    grounds vocabulary with real MEANING ('dog' = 'a domesticated carnivorous
    mammal...') that 'what is X' recall surfaces. Uses each lemma's PRIMARY
    sense (smallest sense_num) gloss. Define-first so every node is born with
    its definition (persist_node_v4 is create-only).
  PASS 2 (structure): synonyms->RelatedTo, hyp->IsA, mm/ms/mp->PartOf,
    ant->Opposes, via lu4d_teach_corpus_batch (fixed admission gate).
"""
import os, re, sys, time

WN_DIR_DEFAULT = "/var/lib/lu/data/wordnet/prolog"
_S_RE  = re.compile(r"^s\((\d+),(\d+),'((?:[^']|'')*)',([nvar]),(\d+),(\d+)\)\.")
_G_RE  = re.compile(r"^g\((\d+),'((?:[^']|'')*)'\)\.")
_HYP_RE = re.compile(r"^hyp\((\d+),(\d+)\)\.")
_MER_RE = re.compile(r"^\w+\((\d+),(\d+)\)\.")
_ANT_RE = re.compile(r"^ant\((\d+),(\d+),(\d+),(\d+)\)\.")


def _unq(w):
    return w.replace("''", "'").strip()


def _clean_gloss(g):
    g = _unq(g)
    # WordNet gloss = "definition; \"example\"..."; keep the definition part
    # before the first example quote.
    cut = g.split('"')[0]
    return cut.rstrip("; ").strip() or g


def load_synsets(wn_dir):
    syn = {}
    with open(os.path.join(wn_dir, "wn_s.pl"), encoding="latin-1") as f:
        for line in f:
            m = _S_RE.match(line)
            if not m:
                continue
            sid = int(m.group(1)); wnum = int(m.group(2))
            word = _unq(m.group(3)); pos = m.group(4); sense = int(m.group(5))
            d = syn.setdefault(sid, {"lemmas": [], "pos": pos})
            d["lemmas"].append((wnum, word, sense))
    for sid, d in syn.items():
        d["lemmas"].sort()
        d["primary"] = d["lemmas"][0][1]
        d["byword"] = {wn: w for wn, w, _ in d["lemmas"]}
    return syn


def load_glosses(wn_dir):
    g = {}
    path = os.path.join(wn_dir, "wn_g.pl")
    if not os.path.exists(path):
        return g
    with open(path, encoding="latin-1") as f:
        for line in f:
            m = _G_RE.match(line)
            if m:
                g[int(m.group(1))] = _clean_gloss(m.group(2))
    return g


def lemma_definitions(syn, glosses):
    """lemma -> definition text, using the lemma's PRIMARY (lowest sense_num)
    synset gloss."""
    best = {}  # lemma -> (sense_num, gloss)
    for sid, d in syn.items():
        gloss = glosses.get(sid)
        if not gloss:
            continue
        for _wn, word, sense in d["lemmas"]:
            cur = best.get(word)
            if cur is None or sense < cur[0]:
                best[word] = (sense, gloss)
    return {w: g for w, (s, g) in best.items()}


def synset_label(d):
    """Per-SYNSET node identity (OP-CONCLUDE sense-keying). Bare primary
    lemma for a lemma's PRIMARY sense (sense_num==1); '<lemma>.<pos>.<n>'
    for non-primary senses. So 'canine' (tooth, sense 1) and 'canine.n.2'
    (dog family) are DISTINCT nodes -> the IS-A walk cannot cross senses.
    Bare primary-sense labels keep 'what is a dog' / recall working unchanged."""
    prim = d["primary"]; pos = d["pos"]; sense = d["lemmas"][0][2]
    return prim if sense == 1 else "%s.%s.%d" % (prim, pos, sense)


def synset_definitions(syn, glosses):
    """synset_label -> gloss (one node per synset that has a gloss)."""
    out = {}
    for sid, d in syn.items():
        g = glosses.get(sid)
        if not g:
            continue
        out.setdefault(synset_label(d), g)
    return out


def _iter_pairs(path, rx):
    if not os.path.exists(path):
        return
    with open(path, encoding="latin-1") as f:
        for line in f:
            m = rx.match(line)
            if m:
                yield int(m.group(1)), int(m.group(2))


def iter_wordnet_bonds(syn, pos=("n", "v", "a", "r"), limit=None, wn_dir=WN_DIR_DEFAULT):
    posset = set(pos); n = 0
    def emit(s, r, o):
        s = s.strip(); o = o.strip()
        if not s or not o or s.lower() == o.lower():
            return None
        return {"subject": s, "relation": r, "object": o, "weight": 0.95, "confidence": "Verified"}
    # synonyms: each non-primary lemma -> RelatedTo -> its synset node
    for sid, d in syn.items():
        if d["pos"] not in posset:
            continue
        lbl = synset_label(d)
        for _wn, w, _se in d["lemmas"]:
            if w == d["primary"]:
                continue
            b = emit(w, "RelatedTo", lbl)
            if b:
                yield b; n += 1
                if limit and n >= limit: return
    # hypernyms -> IsA (NOUN-only, per-synset)
    for c, p in _iter_pairs(os.path.join(wn_dir, "wn_hyp.pl"), _HYP_RE):
        dc, dp = syn.get(c), syn.get(p)
        if not dc or not dp or dc["pos"] != "n" or dp["pos"] != "n": continue
        b = emit(synset_label(dc), "IsA", synset_label(dp))
        if b:
            yield b; n += 1
            if limit and n >= limit: return
    # meronyms -> PartOf (per-synset)
    for fn in ("wn_mm.pl", "wn_ms.pl", "wn_mp.pl"):
        for mero, holo in _iter_pairs(os.path.join(wn_dir, fn), _MER_RE):
            dm, dh = syn.get(mero), syn.get(holo)
            if not dm or not dh: continue
            b = emit(synset_label(dm), "PartOf", synset_label(dh))
            if b:
                yield b; n += 1
                if limit and n >= limit: return
    # antonyms -> Opposes (per-synset)
    path = os.path.join(wn_dir, "wn_ant.pl")
    if os.path.exists(path):
        with open(path, encoding="latin-1") as f:
            for line in f:
                m = _ANT_RE.match(line)
                if not m: continue
                s1, w1, s2, w2 = (int(x) for x in m.groups())
                d1, d2 = syn.get(s1), syn.get(s2)
                if not d1 or not d2: continue
                b = emit(synset_label(d1), "Opposes", synset_label(d2))
                if b:
                    yield b; n += 1
                    if limit and n >= limit: return


def ingest_wordnet(client, cp, smoke_limit=None, wn_dir=WN_DIR_DEFAULT):
    import ingest_curriculum_full as ic
    corpus_id = "wordnet"
    ic.log_progress(f"wordnet: starting wn_dir={wn_dir} smoke={bool(smoke_limit)}")
    t0 = time.time()
    syn = load_synsets(wn_dir)
    glosses = load_glosses(wn_dir)
    defs = synset_definitions(syn, glosses)
    # PASS 1 -- definitions (define-first so nodes carry their meaning)
    defined = 0; buf = []
    throttle = ic.Throttle(getattr(ic, "THROTTLE_CODING", 30.0))
    items = list(defs.items())
    if smoke_limit:
        items = items[: max(1, smoke_limit // 3)]
    for word, gloss in items:
        buf.append({"label": word, "definition": gloss})
        if len(buf) >= 500:
            throttle.wait()
            r = client.call("lu4d_set_definition_batch", {"entries": buf})
            sc = ((r.get("result") or {}).get("structuredContent") or {})
            defined += int(sc.get("defined", 0)); buf = []
    if buf:
        r = client.call("lu4d_set_definition_batch", {"entries": buf})
        sc = ((r.get("result") or {}).get("structuredContent") or {})
        defined += int(sc.get("defined", 0))
    ic.log_progress(f"wordnet: PASS1 definitions attached={defined} of {len(items)} lemmas ({time.time()-t0:.0f}s)")
    # PASS 2 -- structural bonds
    sink = ic.BatchSink(client, ic.Throttle(getattr(ic, "THROTTLE_CODING", 30.0)))
    run_id = getattr(ic, "RUN_ID_TIER1", None) or "wordnet"
    count = 0
    for entry in iter_wordnet_bonds(syn, limit=smoke_limit, wn_dir=wn_dir):
        entry["corpus"] = corpus_id; entry["run_id"] = run_id
        sink.add(entry); count += 1
        if count % 50000 == 0:
            ic.log_progress(
                f"wordnet: bonds={count} landed={sink.bonds_landed} rejected={sink.bonds_rejected} "
                f"errors={sink.bonds_errors} deduped={sink.bonds_skipped_dedup} gated={sink.bonds_gated} "
                f"rate={count/max(1e-6,time.time()-t0):.0f}/s")
    sink.flush()
    ic.log_progress(
        f"wordnet: DONE definitions={defined} bonds={count} landed={sink.bonds_landed} "
        f"rejected={sink.bonds_rejected} errors={sink.bonds_errors}")
    if cp is not None:
        cp[corpus_id] = {"definitions": defined, "bonds_landed": sink.bonds_landed,
                         "rejected": sink.bonds_rejected, "completed": True}
    return {"definitions": defined, "bonds_landed": sink.bonds_landed,
            "rejected": sink.bonds_rejected, "errors": sink.bonds_errors}


if __name__ == "__main__":
    os.environ.setdefault("LU4D_BIN", "/opt/lu/lu/merge-main/target/release/lu4d")
    os.environ.setdefault("LU_CANON_INGEST", "1")
    sys.path.insert(0, "/opt/lu/lu/merge-main/scripts")
    import ingest_curriculum_full as ic
    ic.CHECKPOINT = "/tmp/wn_sample_ckpt.json"
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    brain = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser("~/lu/wn_sample_v5-sst")
    print(f"WORDNET SAMPLE limit={limit} brain={brain}", flush=True)
    client = ic.McpClient(brain, "/var/lib/lu/master.key", log_stderr_to="/tmp/wn_sample_mcp.err", ingest_only=True)
    t = time.time()
    r = ingest_wordnet(client, {}, smoke_limit=limit)
    try: print("FLUSH:", str(client.call("lu4d_flush", {}))[:80], flush=True)
    except Exception as e: print("flush err", e, flush=True)
    try: client.close()
    except Exception: pass
    print("RESULT", r, "elapsed=%.1fs" % (time.time()-t), flush=True)
