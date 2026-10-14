#!/usr/bin/env python3
"""OP-CONCLUDE combined property loader: Quasimodo (CC-BY-2.0) + Ascent++ (CC-BY-4.0).

Emits HasProperty edges ONLY (never IsA). Capability folded into HasProperty
(no new BondType). Property relations:
  Quasimodo predicate in {has_property, has_trait, have, can}  (rank by TYPICALITY)
  Ascent    rel       in {/r/HasProperty, /r/CapableOf, /r/HasA} (rank by support)

RANKING FIX: Quasimodo *saliency* is degenerate (~all 1.0) and cannot rank
single-source facts; *typicality* carries the real plausibility gradient
(0.0-1.0) and is what the TSV now supplies in column 4.

HIGH-CONFIDENCE THRESHOLD (trims forum noise, keeps characteristic props):
  - keep ALL cross-source corroborated edges ([both]) regardless of score;
  - single-source Quasimodo kept only if typicality       >= PROP_Q_MIN (0.60);
  - single-source Ascent    kept only if norm. support     >= PROP_A_MIN (0.25).
Empirically: cuts car{rusting .008, repossessed .019}, cat{spayed .019,
exam .19} while keeping fish{streamlined body .985}, elephant{tusks 1.0},
tree{sap .956}, bird{feather .999}.

Subjects restricted to WordNet noun surface forms (per-synset sense-pin).
Noise filter (bake-off): drop location facts, 'ac X'/'pay X' spam, digit-bearing
and proper-noun fragments (kept only if head content token is a known WordNet word).
Sense-pin: subject -> per-synset node (primary/bare-lemma). Confidence Inferred;
corpus provenance (quasimodo/ascent/quasimodo+ascent); density caps
out<=PROP_MAX_OUT(20, typicality-ranked, +0.2 corroboration boost) / in<=PROP_MAX_IN(50).
"""
import os, re, sys, time
from collections import defaultdict

WN = "/var/lib/lu/data/wordnet/prolog"
QUASI_PRED = {"has_property", "has_trait", "have", "can"}
ASCENT_REL = {"/r/HasProperty", "/r/CapableOf", "/r/HasA"}
MAX_OUT = int(os.environ.get("PROP_MAX_OUT", "30"))
MAX_IN = int(os.environ.get("PROP_MAX_IN", "50"))
Q_MIN = float(os.environ.get("PROP_Q_MIN", "0.60"))   # single-source Quasimodo typicality floor
A_MIN = float(os.environ.get("PROP_A_MIN", "0.25"))   # single-source Ascent norm-support floor
MAX_TOKENS = 3
RELATION = "HasProperty"
assert RELATION != "IsA"

_ART = {"a", "an", "the"}
_AUX = {"is", "are", "be", "being", "been", "can", "could", "has", "have", "had",
        "to", "do", "does", "very", "quite", "often", "usually", "also"}
_LOCPREP = {"located", "inside", "in", "at", "on", "near", "around", "between", "of"}
_STOP = _ART | {"and", "or", "they", "them", "it", "its", "their"}


def build_vocab_and_senses():
    sys.path.insert(0, "/opt/lu/lu/merge-main/scripts")
    import wordnet_loader as wl
    syn = wl.load_synsets(WN)
    vocab = set()
    lemma_noun = defaultdict(list)  # lemma -> [(sense_num, synset_label)]
    for sid, d in syn.items():
        lbl = wl.synset_label(d)
        for _wn, w, se in d["lemmas"]:
            wl_ = w.lower()
            vocab.add(wl_)
            for tok in re.split(r"[ _-]", wl_):
                if tok:
                    vocab.add(tok)
            if d["pos"] == "n":
                lemma_noun[wl_].append((se, lbl))
    return vocab, lemma_noun, wl


def known(tok, vocab):
    return tok in vocab or tok.rstrip("s") in vocab


def clean_property(raw, vocab):
    s = (raw or "").strip().lower().rstrip(".").strip()
    if not s:
        return None
    toks = re.split(r"\s+", s)
    i = 0
    while i < len(toks) and toks[i] in (_ART | _AUX):
        i += 1
    toks = toks[i:]
    if not toks:
        return None
    if toks[0] in ("ac", "pay"):
        return None
    if toks[0] in _LOCPREP:
        return None
    if len(toks) > MAX_TOKENS:
        return None
    content = [t for t in toks if re.fullmatch(r"[a-z][a-z-]+", t) and t not in _STOP]
    if not content:
        return None
    if any(any(ch.isdigit() for ch in t) for t in toks):
        return None
    if not known(content[-1], vocab):
        return None
    return " ".join(content)


def sense_pin(term, lemma_noun):
    cands = lemma_noun.get(term.lower())
    if not cands:
        return term.lower()
    cands = sorted(cands)
    return cands[0][1]


def read_quasimodo(tsv, vocab, subjects=None):
    # TSV cols: subject, predicate, object, typicality  (is_negative pre-filtered out)
    with open(tsv, encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").rstrip("\r").split("\t")
            if len(p) < 4:
                continue
            subj, pred, obj, typ = p[0].lower(), p[1], p[2], p[3]
            if subjects and subj not in subjects:
                continue
            if pred not in QUASI_PRED:
                continue
            prop = clean_property(obj, vocab)
            if not prop or prop == subj:
                continue
            try:
                score = float(typ)
            except ValueError:
                score = 0.0
            yield subj, prop, score, "quasimodo"


def read_ascent(tsv, vocab, subjects=None):
    max_sup = defaultdict(int)
    rows = []
    with open(tsv, encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").rstrip("\r").split("\t")
            if len(p) < 4:
                continue
            subj, rel, a2, sup = p[0].lower(), p[1], p[2], p[3]
            if subjects and subj not in subjects:
                continue
            if rel not in ASCENT_REL:
                continue
            prop = clean_property(a2, vocab)
            if not prop or prop == subj:
                continue
            try:
                s = int(sup)
            except ValueError:
                s = 1
            max_sup[subj] = max(max_sup[subj], s)
            rows.append((subj, prop, s))
    for subj, prop, s in rows:
        yield subj, prop, s / max(1, max_sup[subj]), "ascent"


def merged(subjects, quasi_tsv, ascent_tsv, lemma_noun=None, vocab=None):
    if vocab is None or lemma_noun is None:
        vocab, lemma_noun, _ = build_vocab_and_senses()
    per = defaultdict(dict)  # subject -> prop -> [qscore, ascore]
    for subj, prop, sc, src in read_quasimodo(quasi_tsv, vocab, subjects):
        d = per[subj].setdefault(prop, [0.0, 0.0])
        d[0] = max(d[0], sc)
    for subj, prop, sc, src in read_ascent(ascent_tsv, vocab, subjects):
        d = per[subj].setdefault(prop, [0.0, 0.0])
        d[1] = max(d[1], sc)
    out = {}
    prop_indeg = defaultdict(int)
    for subj, props in per.items():
        label = sense_pin(subj, lemma_noun)
        scored = []
        for prop, (q, a) in props.items():
            both = (q > 0 and a > 0)
            if not both:
                # single-source high-confidence threshold
                if q > 0 and q < Q_MIN:
                    continue
                if a > 0 and q == 0 and a < A_MIN:
                    continue
            boost = 0.2 if both else 0.0
            src = "both" if both else ("quasimodo" if q >= a else "ascent")
            scored.append((prop, max(q, a) + boost, src))
        scored.sort(key=lambda x: x[1], reverse=True)
        kept = []
        for prop, score, src in scored:
            if len(kept) >= MAX_OUT:
                break
            if prop_indeg[prop] >= MAX_IN:
                continue
            prop_indeg[prop] += 1
            kept.append((prop, round(score, 2), src))
        if kept:
            out[subj] = (label, kept)
    return out


_CORPUS = {"both": "quasimodo+ascent", "quasimodo": "quasimodo", "ascent": "ascent"}


def ingest_properties(brain, quasi_tsv, ascent_tsv, keyfile="/var/lib/lu/master.key"):
    os.environ.setdefault("LU4D_BIN", "/opt/lu/lu/merge-main/target/release/lu4d")
    os.environ.setdefault("LU_CANON_INGEST", "1")
    sys.path.insert(0, "/opt/lu/lu/merge-main/scripts")
    import ingest_curriculum_full as ic
    vocab, lemma_noun, _ = build_vocab_and_senses()
    t0 = time.time()
    print("properties: merging full TSVs (subjects=ALL noun)...", flush=True)
    res = merged(None, quasi_tsv, ascent_tsv, lemma_noun=lemma_noun, vocab=vocab)
    nsub = len(res); nedge = sum(len(v[1]) for v in res.values())
    print("properties: subjects=%d edges=%d (%.0fs merge)" % (nsub, nedge, time.time() - t0), flush=True)
    client = ic.McpClient(brain, keyfile, log_stderr_to="/tmp/properties_mcp.err", ingest_only=True)
    sink = ic.BatchSink(client, ic.Throttle(getattr(ic, "THROTTLE_CODING", 30.0)))
    count = 0
    try:
        for subj, (label, kept) in res.items():
            for prop, score, src in kept:
                entry = {"subject": label, "relation": RELATION, "object": prop,
                         "weight": float(score) if score <= 1.0 else 1.0,
                         "confidence": "Inferred",
                         "corpus": _CORPUS.get(src, "quasimodo+ascent"),
                         "run_id": "properties_qa_v1"}
                sink.add(entry); count += 1
                if count % 50000 == 0:
                    print("properties: sent=%d landed=%d rejected=%d errors=%d dedup=%d %.0fs" % (
                        count, sink.bonds_landed, sink.bonds_rejected, sink.bonds_errors,
                        sink.bonds_skipped_dedup, time.time() - t0), flush=True)
        sink.flush()
        print("properties: FLUSH_ALL sent=%d landed=%d rejected=%d errors=%d dedup=%d" % (
            count, sink.bonds_landed, sink.bonds_rejected, sink.bonds_errors,
            sink.bonds_skipped_dedup), flush=True)
        try:
            print("FLUSH:", str(client.call("lu4d_flush", {}))[:80], flush=True)
        except Exception as e:
            print("flush err", e, flush=True)
    finally:
        try: client.close()
        except Exception: pass
    print("properties: DONE edges_sent=%d landed=%d elapsed=%.0fs" % (
        count, sink.bonds_landed, time.time() - t0), flush=True)
    return {"subjects": nsub, "edges_sent": count, "landed": sink.bonds_landed}


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "ingest":
        brain = sys.argv[2]
        q = os.environ.get("QUASI_TSV", "/tmp/quasimodo_full.tsv")
        a = os.environ.get("ASCENT_TSV", "/tmp/ascent_full.tsv")
        print("PROPERTIES INGEST brain=%s Q=%s A=%s Q_MIN=%.2f A_MIN=%.2f" % (brain, q, a, Q_MIN, A_MIN), flush=True)
        r = ingest_properties(brain, q, a)
        print("RESULT", r, flush=True)
    else:
        SUBJECTS = ["dog", "fish", "cat", "bird", "tree", "car", "elephant", "apple", "knife", "computer"]
        q = os.environ.get("QUASI_TSV", "/tmp/quasimodo_full.tsv")
        a = os.environ.get("ASCENT_TSV", "/tmp/ascent_full.tsv")
        res = merged(set(SUBJECTS), q, a)
        total = 0
        for subj in SUBJECTS:
            if subj not in res:
                print("\n########## %s ##########  (no data)" % subj.upper()); continue
            label, kept = res[subj]
            total += len(kept)
            print("\n########## %s  (node: %s) ##########" % (subj.upper(), label))
            for prop, score, src in kept:
                print("   %-28s  %.2f  [%s]" % (prop, score, src))
        print("\nTOTAL edges (10 test subjects):", total)
