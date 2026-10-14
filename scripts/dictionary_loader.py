#!/usr/bin/env python3
"""OP-RETRAIN-1 Tier-0 dictionary loader (GCIDE / Webster's 1913, public domain).

Now that set_definition works, this is CLEAN (not regex-junk):
  PASS 1: headword -> concept node carrying its <def> as Definition content
          (lu4d_set_definition_batch).
  PASS 2: where the entry has an explicit <gen>GENUS</gen> tag (structured,
          not regex), emit headword IsA genus via lu4d_teach_corpus_batch.
Headwords are clean concepts -> gate reject near-zero.
"""
import os, re, sys, time, html

GCIDE_DIR_DEFAULT = "/var/lib/lu/data/gcide/gcide-0.53"
_ENT_RE = re.compile(r"<ent>([^<]+)</ent>")
_DEF_RE = re.compile(r"<def>(.*?)</def>", re.S)
_GEN_RE = re.compile(r"<gen>([^<]+)</gen>")
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def _clean(text):
    text = _TAG_RE.sub("", text)
    text = html.unescape(text)
    text = text.replace("<", "").replace("/", "")
    text = _WS_RE.sub(" ", text).strip()
    return text


def iter_entries(gcide_dir=GCIDE_DIR_DEFAULT):
    """Yield (headword, definition, genus_or_None), first entry per headword."""
    seen = set()
    files = sorted(f for f in os.listdir(gcide_dir) if f.startswith("CIDE."))
    for fn in files:
        path = os.path.join(gcide_dir, fn)
        with open(path, encoding="latin-1") as f:
            blocks = re.split(r"<p>", f.read())
        for blk in blocks:
            em = _ENT_RE.search(blk)
            dm = _DEF_RE.search(blk)
            if not em or not dm:
                continue
            hw = _clean(em.group(1))
            if not hw or hw.lower() in seen:
                continue
            definition = _clean(dm.group(1))
            if not definition or len(definition) < 4:
                continue
            seen.add(hw.lower())
            gm = _GEN_RE.search(dm.group(1))
            genus = _clean(gm.group(1)) if gm else None
            yield hw, definition, genus


def ingest_dictionary(client, cp, smoke_limit=None, gcide_dir=GCIDE_DIR_DEFAULT):
    import ingest_curriculum_full as ic
    corpus_id = "gcide"
    ic.log_progress(f"gcide: starting dir={gcide_dir} smoke={bool(smoke_limit)}")
    t0 = time.time()
    entries = list(iter_entries(gcide_dir))
    if smoke_limit:
        entries = entries[:smoke_limit]
    # PASS 1 -- definitions
    defined = 0; buf = []
    thr = ic.Throttle(getattr(ic, "THROTTLE_CODING", 30.0))
    for hw, definition, _g in entries:
        buf.append({"label": hw, "definition": definition})
        if len(buf) >= 500:
            thr.wait()
            r = client.call("lu4d_set_definition_batch", {"entries": buf})
            sc = ((r.get("result") or {}).get("structuredContent") or {})
            defined += int(sc.get("defined", 0)); buf = []
    if buf:
        r = client.call("lu4d_set_definition_batch", {"entries": buf})
        sc = ((r.get("result") or {}).get("structuredContent") or {})
        defined += int(sc.get("defined", 0))
    ic.log_progress(f"gcide: PASS1 definitions attached={defined} of {len(entries)} entries ({time.time()-t0:.0f}s)")
    # PASS 2 -- structured genus IsA bonds (tag-based, no regex genus guessing)
    sink = ic.BatchSink(client, ic.Throttle(getattr(ic, "THROTTLE_CODING", 30.0)))
    run_id = getattr(ic, "RUN_ID_TIER1", None) or "gcide"
    isa = 0
    # OP-CONCLUDE per-synset: GCIDE genus is bare-lemma (not sense-annotated),
    # so its IsA would reintroduce cross-sense drift on the sense-keyed WordNet
    # backbone. Keep GCIDE DEFINITIONS; drop its genus IsA.
    for hw, _d, genus in entries:
        _ = (hw, genus)
    sink.flush()
    ic.log_progress(
        f"gcide: DONE definitions={defined} genus_isa={isa} landed={sink.bonds_landed} "
        f"rejected={sink.bonds_rejected} errors={sink.bonds_errors}")
    if cp is not None:
        cp[corpus_id] = {"definitions": defined, "genus_isa": sink.bonds_landed, "completed": True}
    return {"definitions": defined, "genus_isa": sink.bonds_landed, "rejected": sink.bonds_rejected,
            "errors": sink.bonds_errors}


if __name__ == "__main__":
    os.environ.setdefault("LU4D_BIN", "/opt/lu/lu/merge-main/target/release/lu4d")
    os.environ.setdefault("LU_CANON_INGEST", "1")
    sys.path.insert(0, "/opt/lu/lu/merge-main/scripts")
    import ingest_curriculum_full as ic
    ic.CHECKPOINT = "/tmp/gcide_ckpt.json"
    limit = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else None
    brain = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser("~/lu/gcide_v5-sst")
    print(f"GCIDE limit={limit} brain={brain}", flush=True)
    client = ic.McpClient(brain, "/var/lib/lu/master.key", log_stderr_to="/tmp/gcide_mcp.err", ingest_only=True)
    t = time.time(); r = ingest_dictionary(client, {}, smoke_limit=limit)
    try: print("FLUSH:", str(client.call("lu4d_flush", {}))[:60], flush=True)
    except Exception as e: print("flush err", e)
    try: client.close()
    except Exception: pass
    print("RESULT", r, "elapsed=%.1fs" % (time.time()-t), flush=True)
