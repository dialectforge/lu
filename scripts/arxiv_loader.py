#!/usr/bin/env python3
r"""
arxiv_loader.py -- Cat.1 retrain Tier 1 declarative loader (ArXiv).

First of three Cat.1 retrain loaders (ArXiv, StackExchange, Grokopedia).
ArXiv is locked as Tier 1 declarative content per project_retrain_curriculum.

WHAT THIS DOES
==============
Streams ArXiv paper metadata (title, authors, abstract, categories) via
the OAI-PMH 2.0 protocol at https://export.arxiv.org/oai2 (which 301s to
https://oaipmh.arxiv.org/oai). For each paper, sends the abstract through
the existing asymmetry-aware extractor (extract_response_bond from
ingest_curriculum_full.py) and lands the resulting bond into the brain via
lu4d_teach_corpus_batch.

Per internal notes (Cat.3a source-trust weighting, commit 9a6111d) every
bond carries source attribution at formation time:
  - corpus = "arxiv"
  - source_url = "https://arxiv.org/abs/<arxiv_id>" (the MCP tool
    schema is `source_url`, not `source_id`; server stamps
    SourceAttribution::TrainingCorpus { corpus, run_id, url } so
    revoke-by-source / audit can find each bond by arxiv id).

The loader is intended to run against the canonical brain at
/var/lib/lu/brain.lbr in --no-save mode (per feedback_canonical_brain_rule.md);
it never calls lu4d_save. Periodic-save and end-of-run save are deliberately
omitted because the smoke target IS the canonical brain.

WHAT THIS DOES NOT DO
=====================
* Does NOT ingest full paper text. Abstracts only. Full-text ingestion is a
  separate project (would dwarf the Cat.1 budget and require LaTeX-to-text
  conversion that OAI-PMH does not provide).
* Does NOT re-implement extraction. extract_response_bond from
  ingest_curriculum_full.py is the single source of truth for the
  asymmetry boundary (Pattern A/B + NP-head blacklist, BR.8b -> BR.10b).
  Per feedback_relation_pattern_table_share.md, _relation_for_pred is
  inlined in ingest_curriculum_full.py because smart_extractor helpers
  are title-anchored; we reuse it via import, not re-derive.
* Does NOT bypass the BatchSink dedup or T.3 gating. Same code path as the
  Tier 2 ingest functions in ingest_curriculum_full.py.

ASYMMETRY BOUNDARY
==================
ArXiv abstracts are declarative ("We present...", "This paper proves...",
"In this work..."). Right Brain accepts declarative content as bonds.
There are no questions to discard.

The asymmetry-fix concern is leakage of declarative-frame tokens
("we present", "this paper", "in this work", "our results") into bond
SUBJECTS. The existing _FORBIDDEN_SUBJ_HEAD blacklist already rejects
"we" and "this" as Pattern A/B subject heads, so:
  "We present a method for X..."   -> Pattern B head "We" -> rejected
  "This paper proves Y..."         -> Pattern B head "This" -> rejected
  "In this work, we show Z..."     -> sentence does not match A or B (no
                                     leading determiner/capital before
                                     a predicate verb) -> bond dropped
For abstracts that DO begin with content nouns ("Convolutional networks
are...", "The Riemann hypothesis is..."), Pattern A/B fires correctly
and the subject is the actual NP. This is the same boundary BR.8b/BR.9
shipped for Tier 2 instruction datasets; we hold it identically.

Acceptance gate (a) verifies zero leakage on a 20-bond sample.

LATEX STRIP
===========
ArXiv abstracts contain LaTeX inline math and macros. Before extraction
we strip:
  * inline math: $...$
  * display math: $$...$$
  * single-arg macros: \cmd{arg}        -> arg
  * zero-arg macros:   \cmd             -> ''
  * leading "Abstract:" prefix (some Astro-PH abstracts include it)
  * collapsed whitespace
We do NOT attempt full LaTeX rendering. The strip set is conservative
because the asymmetry filter is sentence-initial -- as long as the
first sentence reads like English prose after stripping, extraction
behaves correctly.

RATE LIMIT
==========
ArXiv OAI-PMH asks clients to throttle to 1 req / 3 sec. Each ListRecords
call returns up to 1000 records (their default page size), so 100 papers
is one request. We sleep 3 s between paginated requests AND honor any
Retry-After header on 503. If we hit a 503, we retry the SAME request up
to 3 times with the server-suggested delay; if it still fails, we abort
the run (acceptance gate (d) is zero tolerance for 503s in smoke).

RESUME / DEDUP
==============
Per-process: BatchSink.seen rejects exact (subject, relation, object)
re-submission within a run.
Cross-process: a sidecar at ~/.lu/arxiv_progress.jsonl records every
arxiv_id we've seen (not just successfully landed -- we want
"already attempted, no leak in current data" to be a no-op too).
Re-running the smoke: every arxiv_id is in the sidecar -> 0 new bonds.

Different arxiv_ids that produce identical (s, r, o) bonds: the BatchSink
in-process dedup catches the second submission. Source attribution loses
information here (we record only the first arxiv_id), which matches how
Visual Genome dedups across image_ids. Acceptable for Cat.1; if needed,
follow-up port can add multi-source attribution at the brain layer.

INVOCATION
==========
Smoke (canonical brain, --no-save):
  python3 scripts/arxiv_loader.py \\
      --brain /var/lib/lu/brain.lbr \\
      --keyfile /var/lib/lu/master.key \\
      --set cs:cs:AI \\
      --max-papers 100

The loader prints periodic progress and a final summary with bonds_landed,
bonds_skipped_dedup, papers_seen, papers_already_in_sidecar.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

# Reuse the canonical asymmetry-aware extractor and BatchSink/McpClient/Throttle.
# Adding scripts/ to sys.path so we import the existing infrastructure rather
# than re-deriving it (no-hacks principle 1: fix at the layer, do not duplicate).
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from ingest_curriculum_full import (  # type: ignore
    BatchSink,
    McpClient,
    Throttle,
    call_ok,
    extract_response_bond,
    extract_response_bonds_multi_sentence,
)

OAI_BASE = "https://export.arxiv.org/oai2"  # 301s to oaipmh.arxiv.org/oai; curl/urllib follow it.
OAI_NS = {
    "oai": "http://www.openarchives.org/OAI/2.0/",
    "arxiv": "http://arxiv.org/OAI/arXiv/",
}
USER_AGENT = "lu-cat1-retrain-arxiv-loader/1.0 (mailto:info@dialectforge.com)"

DEFAULT_SIDECAR = os.path.expanduser("~/.lu/arxiv_progress.jsonl")
DEFAULT_BRAIN = "/var/lib/lu/brain.lbr"
DEFAULT_KEYFILE = "/var/lib/lu/master.key"

# OAI-PMH asks for >= 3 s between requests. We use 3.5 s to give headroom.
OAI_MIN_INTERVAL_SEC = 3.5
# Per-bond submission throttle (the BatchSink uses this between batches).
SUBMIT_RATE_HZ = 30.0
# OAI-PMH 503 retry policy.
MAX_503_RETRIES = 3
DEFAULT_503_BACKOFF_SEC = 30.0

RUN_TIMESTAMP = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
RUN_ID = f"cat1-arxiv-{RUN_TIMESTAMP}"
CORPUS_ID = "arxiv"


# =====================================================================
# LaTeX strip. Conservative, sentence-shape-preserving.
# =====================================================================

# Match $...$ (inline) and $$...$$ (display). Non-greedy.
_RE_DOLLAR_MATH = re.compile(r"\$\$.+?\$\$|\$[^$\n]+?\$", re.DOTALL)
# Match \command{arg} -> arg (keep one level). Repeated until stable.
_RE_LATEX_BRACE_MACRO = re.compile(r"\\[A-Za-z]+\s*\{([^{}]*)\}")
# Match \command (no braces) -> '' (e.g. \alpha, \\, \&).
_RE_LATEX_BARE_MACRO = re.compile(r"\\[A-Za-z]+|\\[^A-Za-z\s]")
# OAI Astro-PH style "Abstract: " prefix.
_RE_ABSTRACT_PREFIX = re.compile(r"^\s*abstract\s*:\s*", re.IGNORECASE)
# Whitespace collapse.
_RE_WS = re.compile(r"\s+")


def latex_strip(s: str) -> str:
    """Strip LaTeX math, macros, and Abstract: prefix from an arxiv abstract.

    Not a full LaTeX renderer. Goal is to leave the FIRST SENTENCE in a shape
    the asymmetry filter recognizes as English prose. We accept that nested
    braces or unbalanced math will be imperfect; downstream extraction either
    matches or drops the bond, both of which are correct.
    """
    if not s:
        return ""
    out = _RE_DOLLAR_MATH.sub(" ", s)
    # Brace macros: iterate until stable (handles a single level of nesting).
    for _ in range(3):
        new = _RE_LATEX_BRACE_MACRO.sub(r"\1", out)
        if new == out:
            break
        out = new
    out = _RE_LATEX_BARE_MACRO.sub(" ", out)
    out = _RE_ABSTRACT_PREFIX.sub("", out)
    out = _RE_WS.sub(" ", out).strip()
    return out


# =====================================================================
# Sidecar progress file (resume).
# =====================================================================

def _ensure_sidecar(path: str) -> None:
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)
    if not os.path.exists(path):
        # Create empty file.
        open(path, "a").close()


def load_seen_ids(path: str) -> set:
    """Return the set of arxiv_ids already recorded in the sidecar."""
    seen = set()
    if not os.path.exists(path):
        return seen
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            aid = rec.get("arxiv_id")
            if aid:
                seen.add(aid)
    return seen


def append_seen(path: str, arxiv_id: str, status: str, bonds: int) -> None:
    """Append a one-line JSON record for the just-processed paper."""
    rec = {
        "arxiv_id": arxiv_id,
        "status": status,            # "landed" | "no_bond" | "skipped_dedup" | "error"
        "bonds": bonds,
        "ts": datetime.now(timezone.utc).isoformat(),
        "run_id": RUN_ID,
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush()


# =====================================================================
# OAI-PMH paginated fetcher.
# =====================================================================

def _http_get(url: str) -> bytes:
    """GET with our UA, return bytes. Raise urllib.error.HTTPError on non-2xx."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def fetch_oai_page(set_spec: str, frm: str, until: str,
                   resumption_token: str | None) -> tuple[bytes, float]:
    """One OAI-PMH ListRecords request with 503 retry honoring Retry-After.

    Returns (body_bytes, last_sleep_seconds). On 503, sleeps the suggested
    Retry-After (or DEFAULT_503_BACKOFF_SEC) and retries. After
    MAX_503_RETRIES, raises.
    """
    if resumption_token:
        params = {"verb": "ListRecords", "resumptionToken": resumption_token}
    else:
        params = {
            "verb": "ListRecords",
            "metadataPrefix": "arXiv",
            "set": set_spec,
            "from": frm,
            "until": until,
        }
    url = OAI_BASE + "?" + urllib.parse.urlencode(params)
    last_sleep = 0.0
    attempt = 0
    while True:
        try:
            return _http_get(url), last_sleep
        except urllib.error.HTTPError as e:
            if e.code == 503 and attempt < MAX_503_RETRIES:
                ra = e.headers.get("Retry-After") if e.headers else None
                try:
                    backoff = float(ra) if ra else DEFAULT_503_BACKOFF_SEC
                except ValueError:
                    backoff = DEFAULT_503_BACKOFF_SEC
                print(f"[arxiv-loader] 503 from OAI-PMH; sleeping {backoff:.1f}s and retrying "
                      f"(attempt {attempt+1}/{MAX_503_RETRIES})", flush=True)
                time.sleep(backoff)
                last_sleep = backoff
                attempt += 1
                continue
            raise


def parse_oai_xml(body: bytes) -> tuple[list[dict], str | None]:
    """Parse an OAI-PMH ListRecords XML response.

    Returns (records, resumption_token). Each record dict has keys:
      arxiv_id, title, abstract, authors, categories, datestamp.
    Records with no abstract are dropped at parse time.
    """
    root = ET.fromstring(body)
    # First check for OAI-PMH error element.
    err = root.find("oai:error", OAI_NS)
    if err is not None:
        code = err.get("code", "?")
        msg = err.text or ""
        # noRecordsMatch is a normal end-of-stream indicator.
        if code == "noRecordsMatch":
            return [], None
        raise RuntimeError(f"OAI-PMH error code={code} msg={msg!r}")

    list_records = root.find("oai:ListRecords", OAI_NS)
    if list_records is None:
        return [], None
    records: list[dict] = []
    for rec_elem in list_records.findall("oai:record", OAI_NS):
        header = rec_elem.find("oai:header", OAI_NS)
        if header is None:
            continue
        if header.get("status") == "deleted":
            continue
        oai_identifier = (header.findtext("oai:identifier", default="", namespaces=OAI_NS) or "").strip()
        datestamp = (header.findtext("oai:datestamp", default="", namespaces=OAI_NS) or "").strip()

        meta = rec_elem.find("oai:metadata", OAI_NS)
        if meta is None:
            continue
        arxiv_meta = meta.find("arxiv:arXiv", OAI_NS)
        if arxiv_meta is None:
            continue
        arxiv_id = (arxiv_meta.findtext("arxiv:id", default="", namespaces=OAI_NS) or "").strip()
        if not arxiv_id:
            # Fall back to oai:identifier "oai:arXiv.org:1234.5678" tail.
            if oai_identifier.startswith("oai:arXiv.org:"):
                arxiv_id = oai_identifier.split(":", 2)[-1]
        if not arxiv_id:
            continue
        title = (arxiv_meta.findtext("arxiv:title", default="", namespaces=OAI_NS) or "").strip()
        abstract = (arxiv_meta.findtext("arxiv:abstract", default="", namespaces=OAI_NS) or "").strip()
        if not abstract:
            continue
        categories = (arxiv_meta.findtext("arxiv:categories", default="", namespaces=OAI_NS) or "").strip()
        # Authors: arxiv:authors / arxiv:author / arxiv:keyname
        authors = []
        authors_elem = arxiv_meta.find("arxiv:authors", OAI_NS)
        if authors_elem is not None:
            for a in authors_elem.findall("arxiv:author", OAI_NS):
                kn = (a.findtext("arxiv:keyname", default="", namespaces=OAI_NS) or "").strip()
                fn = (a.findtext("arxiv:forenames", default="", namespaces=OAI_NS) or "").strip()
                full = (fn + " " + kn).strip()
                if full:
                    authors.append(full)
        records.append({
            "arxiv_id": arxiv_id,
            "title": title,
            "abstract": abstract,
            "authors": authors,
            "categories": categories,
            "datestamp": datestamp,
        })
    rt_elem = list_records.find("oai:resumptionToken", OAI_NS)
    rt = rt_elem.text.strip() if (rt_elem is not None and rt_elem.text) else None
    if rt == "":
        rt = None
    return records, rt


def stream_papers(set_spec: str, frm: str, until: str,
                  max_papers: int):
    """Yield up to `max_papers` paper dicts from OAI-PMH, honoring rate limit.

    Sleeps OAI_MIN_INTERVAL_SEC between paginated GETs. The first GET has
    no preceding sleep (it's the only request). Yields records in OAI order.
    """
    yielded = 0
    resumption = None
    first = True
    while True:
        if not first:
            time.sleep(OAI_MIN_INTERVAL_SEC)
        first = False
        body, _ = fetch_oai_page(set_spec, frm, until, resumption)
        records, resumption = parse_oai_xml(body)
        for r in records:
            yield r
            yielded += 1
            if yielded >= max_papers:
                return
        if not resumption:
            return


# =====================================================================
# Per-paper bond formation.
# =====================================================================

def make_bonds(paper: dict) -> list:
    """Return list of bond dicts (zero or more) for one arxiv paper.

    Port B (R1, 2026-05-10): replaces the first-sentence-only
    extract_response_bond("", abstract) with
    extract_response_bonds_multi_sentence(title, abstract, source_url)
    which iterates every sentence in the abstract. Path 1
    (clean Pattern A/B subject), Path 2 (pronoun-led -> title anchor,
    subject_origin=\"anchor\"), and Path 3 (junk filter) all apply.
    Returns dict shape matching smart_extractor output, with an
    extra subject_origin field for post-retrain audit.
    """
    abstract = latex_strip(paper.get("abstract", ""))
    if not abstract:
        return []
    title = paper.get("title", "") or ""
    arxiv_id = paper.get("arxiv_id", "") or ""
    source_url = f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else ""
    return extract_response_bonds_multi_sentence(title, abstract, source_url)


# =====================================================================
# Subject leak filter (acceptance gate (a) zero-tolerance backstop).
#
# extract_response_bond's _FORBIDDEN_SUBJ_HEAD blocks the leading head
# token (e.g. "We", "This"). It does NOT block multi-word subjects whose
# FULL phrase reads as a declarative frame (e.g. "Our results show", which
# would be matched if it ever escaped Pattern A/B's NP shape).
#
# Per the brief: zero tolerance. We add a defensive substring check on
# the full subject phrase before submission. Anything that contains a
# declarative-frame token is dropped at the loader layer.
# =====================================================================
_LEAK_SUBSTRINGS = (
    "we present", "this paper", "in this work", "in this paper",
    "our results", "we propose", "we show", "we demonstrate",
    "the paper", "we describe", "we introduce", "we report",
    "we study", "we investigate", "this article", "this study",
    "this work", "our method", "our approach", "this manuscript",
)

# Bare-frame-noun subject filter moved to scripts/ingest_curriculum_full.py
# (commit 8c6a222, reconciliation port 2026-05-10) so every caller of
# extract_response_bond inherits it. Removed local _FRAME_NOUN_SUBJECTS
# frozenset. The substring leak filter and head-token filter below remain
# as loader-level defense-in-depth.


def subject_leaks(subject: str) -> bool:
    if not subject:
        return False
    s = subject.strip().lower()
    for pat in _LEAK_SUBSTRINGS:
        if pat in s:
            return True
    # Bare-frame-noun rejection happens upstream in
    # ingest_curriculum_full._extract_response_subject; no local copy.
    # Also reject if subject head is a known frame-leading word that
    # somehow slipped past _FORBIDDEN_SUBJ_HEAD (defense-in-depth).
    head = s.split()[0] if s.split() else ""
    if head in {"we", "our", "this", "the paper", "in"}:
        return True
    return False


# =====================================================================
# Main.
# =====================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description="ArXiv OAI-PMH abstract loader for Lu Cat.1 retrain.")
    ap.add_argument("--brain", default=DEFAULT_BRAIN)
    ap.add_argument("--keyfile", default=DEFAULT_KEYFILE)
    ap.add_argument("--set", dest="set_spec", default="cs:cs:AI",
                    help="OAI-PMH setSpec, e.g. cs:cs:AI, cs:cs:LG, math:math:AG")
    ap.add_argument("--from", dest="frm", default="2024-01-01",
                    help="OAI-PMH 'from' date YYYY-MM-DD")
    ap.add_argument("--until", default="2024-06-30",
                    help="OAI-PMH 'until' date YYYY-MM-DD")
    ap.add_argument("--max-papers", type=int, default=100)
    ap.add_argument("--sidecar", default=DEFAULT_SIDECAR)
    ap.add_argument("--mcp-stderr-log", default=None,
                    help="Optional path to capture MCP server stderr.")
    args = ap.parse_args()

    _ensure_sidecar(args.sidecar)
    seen_ids = load_seen_ids(args.sidecar)
    print(f"[arxiv-loader] sidecar {args.sidecar}: {len(seen_ids)} arxiv_ids already recorded",
          flush=True)

    # Spawn MCP server with --no-save.
    print(f"[arxiv-loader] spawning lu_4_mcp --mcp --no-save brain={args.brain}", flush=True)
    client = McpClient(args.brain, args.keyfile, log_stderr_to=args.mcp_stderr_log)
    sink = BatchSink(client, Throttle(SUBMIT_RATE_HZ))

    # Counters.
    papers_seen = 0
    papers_already_in_sidecar = 0
    papers_no_bond = 0
    papers_leak_filtered = 0
    submission_errors = 0

    t0 = time.monotonic()
    try:
        for paper in stream_papers(args.set_spec, args.frm, args.until, args.max_papers):
            papers_seen += 1
            aid = paper["arxiv_id"]
            if aid in seen_ids:
                papers_already_in_sidecar += 1
                # Do NOT re-process. Skip silently for resume.
                continue
            bonds = make_bonds(paper)
            if not bonds:
                append_seen(args.sidecar, aid, "no_bond", 0)
                papers_no_bond += 1
                continue
            # Port B (R1): make_bonds returns a list of dicts. Each
            # bond carries subject_origin ("subject" or "anchor").
            paper_queued = 0
            paper_leak_filtered = 0
            for b in bonds:
                subj = b["subject"]
                if subject_leaks(subj):
                    paper_leak_filtered += 1
                    continue
                entry = {
                    "corpus": CORPUS_ID,
                    "source_url": b.get("source_url") or f"https://arxiv.org/abs/{aid}",
                    "run_id": RUN_ID,
                    "subject": subj,
                    "relation": b["relation"],
                    "object": b["object"],
                    "weight": b["weight"],
                    "confidence": b["confidence"],
                }
                try:
                    queued = sink.add(entry)
                    if queued:
                        paper_queued += 1
                except Exception as e:
                    submission_errors += 1
                    append_seen(args.sidecar, aid, f"error:{type(e).__name__}", 0)
            # Sidecar record is per-paper, not per-bond, so we log a
            # rollup. The detail breakdown lives in the BatchSink counters
            # and the per-bond brain provenance.
            if paper_queued > 0:
                append_seen(args.sidecar, aid, f"queued:{paper_queued}", paper_queued)
            elif paper_leak_filtered > 0 and paper_leak_filtered == len(bonds):
                append_seen(args.sidecar, aid, "leak_filtered_all", 0)
                papers_leak_filtered += 1
            else:
                append_seen(args.sidecar, aid, "skipped_dedup_or_malformed", 0)
            if papers_seen % 25 == 0:
                print(f"[arxiv-loader]   ... seen={papers_seen} "
                      f"landed={sink.bonds_landed} dedup={sink.bonds_skipped_dedup} "
                      f"no_bond={papers_no_bond} leak={papers_leak_filtered}",
                      flush=True)
        # Flush any partial batch.
        sink.flush()
    finally:
        # Per feedback_canonical_brain_rule.md / --no-save semantics:
        # NEVER call lu4d_save against the canonical brain. Just close
        # the MCP server cleanly so it exits without invoking auto-save
        # (which is disabled by --no-save anyway).
        client.close()

    elapsed = time.monotonic() - t0

    # Final report.
    print()
    print("=" * 70)
    print(f"arxiv-loader DONE in {elapsed:.1f}s  run_id={RUN_ID}")
    print(f"  papers_seen                 : {papers_seen}")
    print(f"  papers_already_in_sidecar   : {papers_already_in_sidecar}")
    print(f"  papers_no_bond              : {papers_no_bond}")
    print(f"  papers_leak_filtered        : {papers_leak_filtered}")
    print(f"  bonds_landed (server-side)  : {sink.bonds_landed}")
    print(f"  bonds_existing (reinforced) : {sink.bonds_existing}")
    print(f"  bonds_gated  (T.3)          : {sink.bonds_gated}")
    print(f"  bonds_skipped_dedup (local) : {sink.bonds_skipped_dedup}")
    print(f"  bonds_errors                : {sink.bonds_errors}")
    print(f"  submission_errors           : {submission_errors}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
