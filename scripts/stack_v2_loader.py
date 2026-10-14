#!/usr/bin/env python3
"""The Stack v2 (+ generic multi-language) STRUCTURAL code loader.

Streams source files through scripts/code_extractor.py (tree-sitter, declarative
structural bonds only) and lands them into a Lu v5 brain via the SAME disciplined
path as ingest_curriculum_full.ingest_python_code: BatchSink -> lu4d_teach_corpus_batch
(T.3 gating + TrainingCorpus stamping), file-indexed checkpointing, periodic async
save. No raw code text is ever emitted -- code_extractor gates every bond end.

TWO SOURCES (pluggable, same downstream landing code):

  * local  -- walk a local directory tree of real source files. Used to VALIDATE
              the full extractor->landing->dedup->RAM path on a bounded sample
              without any gated/credentialed access. Fully runnable.

  * hf     -- stream The Stack v2 IDs from HuggingFace and fetch file CONTENT from
              the Software Heritage S3 bucket (the documented access model). This
              path is REAL code against the documented API but requires, and hard-
              checks for: (1) an HF token whose account has ACCEPTED the gated
              bigcode/the-stack-v2 terms, (2) AWS credentials for the
              `softwareheritage` S3 bucket, and (3) the SoftwareHeritage/INRIA
              bulk-access agreement. Absent any of these it raises
              StackV2AccessError rather than silently degrading.

Corpus tag: "the-stack-v2". NOTE: for correct trust attribution this corpus key
must be added to Rust TrustTable::seed_curriculum_corpora (else every bond resolves
to default_multiplier=0.4). That is a separate, gated binary change -- see report.
"""
import gzip
import io
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import code_extractor as ce  # noqa: E402  (declarative structural extractor)

CORPUS_ID = "the-stack-v2"
CORPUS_ID_SCD = "starcoderdata"

# Map our extractor languages -> The Stack v2 HF dataset config names.
_STACK_V2_LANG_CONFIG = {
    "python": "Python",
    "javascript": "JavaScript",
    "typescript": "TypeScript",
    "tsx": "TypeScript",
    "rust": "Rust",
    "go": "Go",
    "java": "Java",
    "c": "C",
    "cpp": "C++",
}


class StackV2AccessError(RuntimeError):
    """Raised when the gated Stack v2 / Software Heritage access is not present."""


# ---------------------------------------------------------------------------
# Source A: local directory tree (runnable now; used for bounded validation)
# ---------------------------------------------------------------------------
def iter_local_tree(root, exts=None, limit=None):
    """Yield (path, ext, source_text) for supported source files under `root`,
    in sorted (resume-stable) order. `exts` defaults to all extensions the
    extractor supports. `limit` caps the number of files yielded."""
    if exts is None:
        exts = set(ce.supported_exts()) | {".py"}
    else:
        exts = set(exts)
    skip_dir = {".git", "__pycache__", "node_modules", "target", "build",
                "dist", ".venv", "venv", "vendor"}
    count = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in skip_dir)
        for fn in sorted(filenames):
            ext = os.path.splitext(fn)[1].lower()
            if ext not in exts:
                continue
            path = os.path.join(dirpath, fn)
            try:
                text = open(path, encoding="utf-8", errors="replace").read()
            except Exception:
                continue
            yield path, ext, text
            count += 1
            if limit and count >= limit:
                return


def local_source_url(root, path):
    rel = os.path.relpath(path, root) if os.path.isdir(root) else os.path.basename(path)
    return "codetree://" + rel.replace(os.sep, "/")


# ---------------------------------------------------------------------------
# Source B: The Stack v2 via HF ids + Software Heritage S3 content
# (documented access model; hard-gated on credentials)
# ---------------------------------------------------------------------------
def _require_hf_token():
    try:
        import huggingface_hub as hh
    except Exception as e:
        raise StackV2AccessError("huggingface_hub not installed: %s" % e)
    tok = hh.get_token()
    if not tok:
        raise StackV2AccessError(
            "No HuggingFace token found. The Stack v2 is GATED: log in with an "
            "account that has accepted https://huggingface.co/datasets/bigcode/"
            "the-stack-v2 terms (huggingface-cli login / HF_TOKEN env).")
    return tok


def _make_s3_client():
    try:
        import boto3
        from botocore.exceptions import NoCredentialsError  # noqa: F401
    except Exception as e:
        raise StackV2AccessError("boto3 not installed: %s" % e)
    session = boto3.Session()
    creds = session.get_credentials()
    if creds is None:
        raise StackV2AccessError(
            "No AWS credentials found. The Stack v2 file CONTENT lives in the "
            "Software Heritage S3 bucket and requires AWS credentials plus the "
            "SoftwareHeritage/INRIA bulk-access agreement to download.")
    return session.client("s3")


def _swh_download(s3, blob_id, src_encoding):
    """Fetch one file's content from the Software Heritage S3 bucket.
    Content is gzip-compressed under key content/<blob_id> (documented layout)."""
    key = "content/%s" % blob_id
    obj = s3.get_object(Bucket="softwareheritage", Key=key)
    with gzip.GzipFile(fileobj=io.BytesIO(obj["Body"].read())) as gz:
        return gz.read().decode(src_encoding or "utf-8", "replace")


def iter_stack_v2(languages, limit=None, per_lang_limit=None):
    """Yield (path, ext, source_text) streamed from The Stack v2.

    REQUIRES an accepted HF gate token AND AWS credentials -- raises
    StackV2AccessError otherwise (checked up front, before any network use).
    `languages` are extractor language names (e.g. 'rust','go'); mapped to
    Stack v2 configs. Streaming keeps memory bounded (no full-shard download).
    """
    tok = _require_hf_token()
    s3 = _make_s3_client()
    try:
        from datasets import load_dataset
    except Exception as e:
        raise StackV2AccessError("datasets not installed: %s" % e)

    ext_for = {"python": ".py", "javascript": ".js", "typescript": ".ts",
               "tsx": ".tsx", "rust": ".rs", "go": ".go", "java": ".java",
               "c": ".c", "cpp": ".cpp"}
    total = 0
    for lang in languages:
        config = _STACK_V2_LANG_CONFIG.get(lang)
        if not config:
            continue
        ext = ext_for.get(lang, "")
        ds = load_dataset("bigcode/the-stack-v2-dedup", config,
                          split="train", streaming=True, token=tok)
        n_lang = 0
        for row in ds:
            blob_id = row.get("blob_id")
            enc = row.get("src_encoding", "utf-8")
            path = row.get("path") or ("%s/%s" % (config, blob_id))
            if not blob_id:
                continue
            try:
                text = _swh_download(s3, blob_id, enc)
            except Exception:
                continue
            yield "stackv2://%s/%s" % (config, path), ext, text
            total += 1
            n_lang += 1
            if per_lang_limit and n_lang >= per_lang_limit:
                break
            if limit and total >= limit:
                return


# ---------------------------------------------------------------------------
# Source C: bigcode/starcoderdata via inline parquet content (HF-token only)
# StarCoderData ships file CONTENT inline in per-language parquet folders --
# no Software Heritage S3, no AWS credentials. We stream parquet shards via
# HfFileSystem range reads (one row group at a time -> bounded memory) so a
# full shard is never materialised. Corpus tag: "starcoderdata".
# ---------------------------------------------------------------------------
_STARCODER_FOLDER = {
    "python": "python", "javascript": "javascript", "typescript": "typescript",
    "rust": "rust", "go": "go", "java": "java", "c": "c", "cpp": "cpp",
}
_EXT_FOR_LANG = {
    "python": ".py", "javascript": ".js", "typescript": ".ts",
    "rust": ".rs", "go": ".go", "java": ".java", "c": ".c", "cpp": ".cpp",
}
# Metadata sentinel tokens StarCoderData prepends to file content.
_SCD_SENTINELS = ("<reponame>", "<filename>", "<gh_stars>", "<issue_start>",
                  "<jupyter_start>", "<commit_before>")
_NL = chr(10)


def _strip_starcoder_sentinels(text):
    """Drop leading StarCoderData metadata sentinel lines so the AST parser
    sees real source beginning at line 1. Only strips a LEADING run of
    sentinel-prefixed lines; sentinels later in the file are left alone."""
    while True:
        s = text.lstrip(_NL)
        if s.startswith(_SCD_SENTINELS):
            nl = s.find(_NL)
            if nl == -1:
                return ""
            text = s[nl + 1:]
            continue
        return text


def _require_hf_token_scd():
    tok = os.environ.get("HF_TOKEN")
    if not tok:
        try:
            import huggingface_hub as hh
            tok = hh.get_token()
        except Exception:
            tok = None
    if not tok:
        raise StackV2AccessError(
            "No HuggingFace token. bigcode/starcoderdata is GATED: provide "
            "HF_TOKEN (e.g. from /etc/lu/hf_token.env) for an account that has "
            "accepted https://huggingface.co/datasets/bigcode/starcoderdata terms.")
    return tok


def iter_starcoderdata(languages, limit=None, per_lang_limit=None, token=None,
                       shard_limit=None):
    """Yield (source_url, ext, source_text) streamed from bigcode/starcoderdata.

    Inline-content model: only an HF token (accepted gate) is required. Parquet
    shards are streamed a row group at a time via HfFileSystem range reads, so
    memory stays bounded regardless of shard size. `languages` are extractor
    language names (python, javascript, typescript, rust, go, java, c, cpp).
    """
    tok = token or _require_hf_token_scd()
    try:
        from huggingface_hub import HfFileSystem, HfApi
        import pyarrow.parquet as pq
    except Exception as e:
        raise StackV2AccessError("huggingface_hub/pyarrow missing: %s" % e)
    fs = HfFileSystem(token=tok)
    api = HfApi()
    repo_files = api.list_repo_files(
        "bigcode/starcoderdata", repo_type="dataset", token=tok)
    total = 0
    for lang in languages:
        folder = _STARCODER_FOLDER.get(lang)
        if not folder:
            continue
        default_ext = _EXT_FOR_LANG.get(lang, "")
        shards = sorted(f for f in repo_files
                        if f.startswith(folder + "/") and f.endswith(".parquet"))
        if shard_limit:
            shards = shards[:shard_limit]
        n_lang = 0
        done_lang = False
        for shard in shards:
            if done_lang:
                break
            fp = "datasets/bigcode/starcoderdata/" + shard
            fh = fs.open(fp, "rb")
            try:
                pf = pq.ParquetFile(fh)
                for batch in pf.iter_batches(
                        batch_size=64,
                        columns=["content", "max_stars_repo_path",
                                 "max_stars_repo_name"]):
                    for row in batch.to_pylist():
                        content = row.get("content")
                        if not content:
                            continue
                        text = _strip_starcoder_sentinels(content)
                        if not text:
                            continue
                        rel = row.get("max_stars_repo_path") or ""
                        repo = row.get("max_stars_repo_name") or folder
                        ext = os.path.splitext(rel)[1].lower()
                        if not ext or (ext != ".py" and ce.lang_for_ext(ext) is None):
                            ext = default_ext
                        if rel:
                            url = "starcoderdata://%s/%s/%s" % (folder, repo, rel)
                        else:
                            url = "starcoderdata://%s/%s" % (folder, repo)
                        yield url, ext, text
                        total += 1
                        n_lang += 1
                        if per_lang_limit and n_lang >= per_lang_limit:
                            done_lang = True
                            break
                        if limit and total >= limit:
                            return
                    if done_lang:
                        break
            finally:
                try:
                    fh.close()
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# Downstream landing -- faithful mirror of ingest_python_code's loop.
# ---------------------------------------------------------------------------
def run_ingest(client, cp, files_iter, corpus_id=CORPUS_ID, smoke_limit=None,
               make_url=None, dry_run=False):
    """Consume (path, ext, source) triples, extract structural bonds, land them.

    Mirrors ingest_curriculum_full.ingest_python_code EXACTLY for the landing
    half: BatchSink -> lu4d_teach_corpus_batch, file-indexed checkpoint,
    periodic async save. `dry_run=True` skips all client/brain interaction and
    only exercises extraction (used to measure extractor yield/RAM in isolation).
    Returns a summary dict.
    """
    import ingest_curriculum_full as icf

    if not dry_run:
        sink = icf.BatchSink(client, icf.Throttle(icf.THROTTLE_CODING))
        sink.bonds_landed = cp.get(corpus_id, {}).get("bonds_landed", 0)
        last_saved_id = cp.get(corpus_id, {}).get("last_saved_id", -1)
        last_save_t = time.monotonic()
        last_landed = sink.bonds_landed
    else:
        sink = None
        last_saved_id = -1

    start_t = time.monotonic()
    files_processed = 0
    triples_seen = 0
    files_zero = 0
    parse_errors = 0
    per_lang = {}

    for idx, (path, ext, text) in enumerate(files_iter):
        if smoke_limit and files_processed >= smoke_limit:
            break
        if not dry_run and idx <= last_saved_id:
            continue
        url = make_url(path) if make_url else path
        try:
            bonds = ce.extract_bonds_from_source(text, url, ext)
        except Exception:
            parse_errors += 1
            bonds = []
        triples_seen += len(bonds)
        lang = ce.lang_for_ext(ext) or ("python" if ext == ".py" else ext)
        pl = per_lang.setdefault(lang, {"files": 0, "bonds": 0})
        pl["files"] += 1
        pl["bonds"] += len(bonds)
        if not bonds:
            files_zero += 1
        if not dry_run:
            for b in bonds:
                sink.add({
                    "corpus": corpus_id,
                    "source_url": b.get("source_url") or url,
                    "run_id": getattr(icf, "RUN_ID_TIER1", "stackv2-validate"),
                    "subject": b["subject"],
                    "relation": b["relation"],
                    "object": b["object"],
                    "weight": b.get("weight", 0.8),
                    "confidence": b.get("confidence", "Verified"),
                })
            sink.flush()
            bonds_since_save = sink.bonds_landed - last_landed
            cp[corpus_id] = {"bonds_landed": sink.bonds_landed,
                             "last_saved_id": last_saved_id}
            icf.save_checkpoint(cp)
            new_t, _b, saved_ok = icf._maybe_save(
                client, last_save_t, bonds_since_save, cp, corpus_id, idx)
            if saved_ok:
                last_saved_id = idx
                last_landed = sink.bonds_landed
            last_save_t = new_t
        files_processed += 1

    elapsed = time.monotonic() - start_t
    summary = {
        "corpus": corpus_id,
        "files_processed": files_processed,
        "triples_extracted": triples_seen,
        "files_zero": files_zero,
        "parse_errors": parse_errors,
        "elapsed_sec": round(elapsed, 2),
        "files_per_sec": round(files_processed / elapsed, 2) if elapsed > 0 else 0,
        "bonds_per_file": round(triples_seen / files_processed, 2) if files_processed else 0,
        "per_lang": per_lang,
        "dry_run": dry_run,
    }
    if not dry_run:
        sink.flush()
        try:
            client.call("lu4d_save", {})
        except Exception:
            pass
        summary["bonds_landed"] = sink.bonds_landed
        summary["bonds_deduped"] = sink.bonds_skipped_dedup
        summary["bonds_existing"] = sink.bonds_existing
        summary["bonds_gated"] = sink.bonds_gated
        summary["bonds_errors"] = sink.bonds_errors
    return summary


# ---------------------------------------------------------------------------
# Standalone validation entrypoint (does NOT touch the canonical brain unless
# --brain is pointed at it explicitly).
# ---------------------------------------------------------------------------
def _main():
    import argparse
    import json
    import resource
    ap = argparse.ArgumentParser(description="Stack v2 / code loader validation")
    ap.add_argument("--source", choices=["local", "hf", "starcoderdata"], default="local")
    ap.add_argument("--root", help="local tree root (source=local)")
    ap.add_argument("--langs", default="rust,c,cpp,python,go,javascript",
                    help="comma extractor-lang list (source=hf)")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--per-lang-limit", type=int, default=None)
    ap.add_argument("--shard-limit", type=int, default=None,
                    help="cap parquet shards per language (source=starcoderdata)")
    ap.add_argument("--seed-curriculum", action="store_true",
                    help="call lu4d_trust_reset_to_defaults{seed_curriculum_corpora:true} "
                         "at startup so curriculum corpora (incl. starcoderdata=0.9) "
                         "resolve above the 0.4 default; applies to NEW bonds only")
    ap.add_argument("--dry-run", action="store_true",
                    help="extract only; no brain/client interaction")
    ap.add_argument("--brain", default=None, help="scratch brain path (landing)")
    ap.add_argument("--checkpoint", default=None,
                    help="checkpoint path for resume (default: ingest_curriculum_full.CHECKPOINT); "
                         "loaded at startup so a restart resumes instead of reprocessing")
    ap.add_argument("--keyfile", default="/var/lib/lu/master.key")
    ap.add_argument("--mcp-stderr", default=None)
    args = ap.parse_args()

    corpus_id = CORPUS_ID
    if args.source == "local":
        if not args.root:
            ap.error("--root required for source=local")
        root = args.root
        files_iter = iter_local_tree(root, limit=args.limit)
        make_url = lambda p: local_source_url(root, p)
    elif args.source == "starcoderdata":
        langs = [x.strip() for x in args.langs.split(",") if x.strip()]
        files_iter = iter_starcoderdata(langs, limit=args.limit,
                                        per_lang_limit=args.per_lang_limit,
                                        shard_limit=args.shard_limit)
        make_url = lambda p: p
        corpus_id = CORPUS_ID_SCD
    else:
        langs = [x.strip() for x in args.langs.split(",") if x.strip()]
        files_iter = iter_stack_v2(langs, limit=args.limit,
                                   per_lang_limit=args.per_lang_limit)
        make_url = lambda p: p

    client = None
    cp = {}
    if not args.dry_run:
        import ingest_curriculum_full as icf
        if args.checkpoint:
            icf.CHECKPOINT = args.checkpoint
        cp = icf.load_checkpoint()  # resume from prior progress if present
        if not args.brain:
            ap.error("--brain required unless --dry-run")
        client = icf.McpClient(args.brain, args.keyfile,
                               log_stderr_to=args.mcp_stderr,
                               ingest_only=True)
        if args.seed_curriculum:
            # Mirror ingest_curriculum_full startup: seed the locked
            # curriculum trust hierarchy (incl. starcoderdata=0.9) BEFORE the
            # first bond lands, so code bonds do not resolve to default 0.4.
            r = client.call("lu4d_trust_reset_to_defaults",
                            {"seed_curriculum_corpora": True})
            ok = icf.call_ok(r) if hasattr(icf, "call_ok") else bool(r)
            sc = ((r.get("result") or {}).get("structuredContent") or {}) if isinstance(r, dict) else {}
            sys.stderr.write("SEED_CURRICULUM ok=%s corpus_entries=%s%s"
                             % (ok, sc.get("corpus_entries"), chr(10)))
            if not ok:
                raise SystemExit("trust seed failed; aborting so bonds do not "
                                 "land at the 0.4 default")
    try:
        summary = run_ingest(client, cp, files_iter, corpus_id=corpus_id,
                             smoke_limit=args.limit, make_url=make_url,
                             dry_run=args.dry_run)
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass
    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    summary["loader_peak_rss_mb"] = round(peak_rss_mb, 1)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    _main()
