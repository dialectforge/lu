#!/usr/bin/env python3
"""StackExchange Cat.1 retrain loader.

Tier 1 — ANSWERS ONLY. This is the canonical asymmetry-case loader.

Why: the project_retrain_curriculum plan classifies StackExchange as Tier 1
training material, but ONLY the declarative half of it. Questions are
interrogative-shape; bonding question text into Right Brain would be a
direct violation of feedback_training_data_asymmetry_actual.md (and would
inject the very hallucination-shape pattern the Left/Right architecture
exists to prevent). So we discard PostTypeId == 1 outright, ingest
PostTypeId == 2 (answers) only, and even within answers we strip
sentences that re-state or quote the question.

Pipeline shape:

  Posts.xml (StackExchange data dump, per-site)
    -> iterparse stream (no full-file load; 23 GB dumps stay viable)
    -> filter: PostTypeId==2, Score>=min, no DeletionDate
    -> HTML body -> plain text (BeautifulSoup; <code>/<pre> dropped
       entirely so source-code does not contaminate bonds)
    -> markdown-residue cleanup (**bold**, *italic*, `code`)
    -> sentence split
    -> question-frame pre-filter: drop sentences that lead with
       "how do i", "what is", "you're asking", etc. -- the answer-side
       quoting that would otherwise leak interrogative-shape into bond
       subjects.
    -> derive a declarative title anchor from the cleaned answer body
       (most-frequent proper-noun phrase, fallback to first declarative
       sentence head). NEVER use the question title; it is interrogative.
    -> feed {title, text, url} into smart_extractor.extract_bonds_from_
       wikipedia_article (existing infra; do NOT duplicate extraction
       logic per feedback_relation_pattern_table_share.md).
    -> stamp every bond with source=<site> + source_id=<post_id>
       (LOCAL bookkeeping; used to construct the wire-level URL).
    -> send to lu4d_teach_corpus via stdio MCP. The MCP tool schema
       (src/mcp.rs:258) accepts `source_url` per-bond -- NOT source_id;
       we wire source_url = f"https://<site>/a/<post_id>". This matches
       the unified `source_url` convention used by the arxiv and
       grokopedia loaders (reconciliation port, 2026-05-10).

Resume: ingested post_ids tracked in ~/.lu/stackoverflow_progress.jsonl.
Dedup: same post_id twice -> no-op. Two answers with verbatim-identical
body text -> ingested once, with both source_ids retained.

Encryption / canonical brain: caller passes --keyfile and --brain; we
default to /var/lib/lu/master.key and /var/lib/lu/brain.lbr per
feedback_canonical_brain_rule.md. Pass --no-save through to lu_4_mcp so
smoke tests never mutate production bytes.

Sites supported: any per-site Posts.xml from the Stack Exchange data dump
(https://archive.org/details/stackexchange). Primary target is Stack
Overflow (stackoverflow.com-Posts.7z); stats.stackexchange.com and
math.stackexchange.com are listed as higher-signal secondaries in the
curriculum spec. The --site arg sets the corpus label and source tag.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

# ------------------------------------------------------------------
# Path wiring -- import smart_extractor from the same scripts/ dir.
# ------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import smart_extractor  # noqa: E402 -- after sys.path tweak
from ingest_curriculum_full import extract_response_bonds_multi_sentence  # noqa: E402


# ------------------------------------------------------------------
# Defaults
# ------------------------------------------------------------------
DEFAULT_BRAIN = "/var/lib/lu/brain.lbr"
DEFAULT_KEYFILE = "/var/lib/lu/master.key"
DEFAULT_LU4D_BIN = "/opt/lu/lu/merge-main/target/release/lu4d"
DEFAULT_PROGRESS = os.path.expanduser("~/.lu/stackoverflow_progress.jsonl")
THROTTLE_HZ = 100.0


# ------------------------------------------------------------------
# Asymmetry pre-filter: question-frame token list.
#
# The hard claim from feedback_training_data_asymmetry_actual.md is that
# bond subjects must never be interrogative-shape. Stack Exchange answers
# routinely quote or re-state the question ("You're asking how to...",
# "The question is whether...", "If you want to know...") within an
# otherwise declarative post. We pre-filter at the sentence level before
# handing the text to smart_extractor. Lowercase, leading-whitespace-
# stripped substring match; we want recall, not surgical precision -- if
# we drop a borderline-declarative sentence, we lose one bond; if we let
# an interrogative slip, the asymmetry gate fails.
#
# Curated from the brief's explicit list plus the canonical wh-frames
# that show up in answer-side question restatement.
# ------------------------------------------------------------------
QUESTION_FRAME_PREFIXES = (
    "how do i ", "how do you ", "how can i ", "how can you ", "how to ",
    "how does ", "how is ", "how would ", "how should ",
    "what is ", "what are ", "what does ", "what do ", "what would ",
    "what should ", "what's ", "whats ",
    "why does ", "why do ", "why is ", "why are ", "why would ",
    "why should ", "why can't ", "why cant ", "why not ",
    "when do ", "when does ", "when is ", "when should ",
    "where do ", "where does ", "where is ", "where can ",
    "which is ", "which one ", "which of ",
    "who is ", "who are ",
    "is it ", "is there ", "are there ", "do you ", "do i ", "does it ",
    "should i ", "should you ", "can i ", "can you ", "could i ",
    "could you ", "would you ", "would i ",
    "you're asking ", "youre asking ", "you are asking ",
    "the question is ", "the question was ", "your question ",
    "the op asked ", "op asked ", "op is asking ",
    "if you want to know ", "if you want ", "you want to ",
    "i'm trying to ", "im trying to ", "i am trying to ",
    "i'm wondering ", "im wondering ",
    "i want to know ", "i wanted to know ",
)

# Inline interrogative-frame substring -- a stricter check for tokens
# that show up *inside* otherwise declarative-looking sentences. We use
# this only as a defence-in-depth audit token list; the prefix list above
# is the actual filter.
QUESTION_FRAME_INLINE_TOKENS = (
    "how do i ", "how do you ", "what is ", "why does ", "why is ",
    "should i ", "can i ", "you're asking", "the question is",
    "if you want ", "you want to ",
)


# ------------------------------------------------------------------
# Code-shape token list for the b-gate audit.
# ------------------------------------------------------------------
CODE_SHAPE_TOKENS = (
    "def ", "function ", "function(", "import ", "from ", "<div", "<span",
    "<script", "console.log", "print(", "println!", "fn ", "return ",
    "var ", "let ", "const ", "public ", "private ", "static ",
    "#include", "@override", "=>", "{", "}", "[]", "();", ");",
    "&&", "||", "::", "->", "===", "!==",
)


# ------------------------------------------------------------------
# HTML -> text. Strip <code>/<pre> entirely. Keep <a> text. Drop tags.
# ------------------------------------------------------------------
def html_to_text(html: str) -> str:
    """Convert StackExchange answer HTML to declarative-only plain text.

    StackExchange post HTML uses standard tags: <p>, <code>, <pre>,
    <blockquote>, <a>, <strong>, <em>, <ul>/<ol>/<li>, <h1>-<h6>. The
    rules:

    - <code> and <pre> are dropped wholesale. Code is not declarative
      natural language; even inline `code` text would pollute bond
      subjects with identifiers like `def `, `print(`, `<div>`.
    - <blockquote> is dropped. Quoted text on Stack Exchange almost
      always quotes the question. This is the second line of asymmetry
      defence after the sentence-level question-frame filter.
    - <a> is replaced by its text (link target dropped; visible anchor
      kept because answers often phrase named entities through links).
    - All other tags are flattened to their text content.
    - Markdown residue (sometimes present in body field of older dumps):
      **bold**, *italic*, backticks, leading `> ` quote markers.
    """
    if not html:
        return ""

    try:
        from bs4 import BeautifulSoup
    except Exception as e:  # pragma: no cover -- env probe in tests
        raise RuntimeError(
            "bs4 (beautifulsoup4) is required for stackexchange_loader. "
            "Install with: pip3 install --user --break-system-packages "
            "beautifulsoup4"
        ) from e

    soup = BeautifulSoup(html, "html.parser")

    # Drop code, pre, and blockquote subtrees entirely.
    for tag_name in ("code", "pre", "blockquote", "script", "style"):
        for el in soup.find_all(tag_name):
            el.decompose()

    # Replace <a> with its text (drop href).
    for a in soup.find_all("a"):
        a.replace_with(a.get_text(" ", strip=True))

    text = soup.get_text(" ", strip=True)

    # Markdown residue cleanup -- safe because we already lost any code
    # blocks (which would have legitimate backticks/asterisks).
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)   # bold
    text = re.sub(r"(?<!\*)\*([^*\n]+)\*", r"\1", text)  # italic
    text = re.sub(r"`([^`\n]+)`", r"\1", text)        # inline code remnants
    text = re.sub(r"^\s*>\s*", "", text, flags=re.MULTILINE)  # quote markers

    # Collapse whitespace.
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ------------------------------------------------------------------
# Sentence split + question-frame filter.
# ------------------------------------------------------------------
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def split_sentences(text: str) -> List[str]:
    if not text:
        return []
    # Pre-normalise: collapse runs of whitespace, then split on terminal
    # punctuation followed by capitalised char. This is the same heuristic
    # smart_extractor._split_sentences uses; we duplicate locally so we
    # can run the filter BEFORE the smart_extractor sees the text.
    s = _SENT_SPLIT.split(text)
    return [x.strip() for x in s if x and x.strip()]


def is_question_frame_sentence(sentence: str) -> bool:
    """True iff the sentence starts with an interrogative frame.

    Lowercased prefix-match against QUESTION_FRAME_PREFIXES. Also catches
    any sentence that ends with '?' (a literal question, possibly quoted
    in the answer).
    """
    if not sentence:
        return False
    if sentence.rstrip().endswith("?"):
        return True
    head = sentence.lstrip().lower()
    return any(head.startswith(p) for p in QUESTION_FRAME_PREFIXES)


def filter_question_frame_sentences(sentences: List[str]) -> List[str]:
    return [s for s in sentences if not is_question_frame_sentence(s)]


# ------------------------------------------------------------------
# Title anchor derivation.
#
# smart_extractor.extract_bonds_from_wikipedia_article expects a
# {title, text, url} dict. The title is the subject anchor for L3/L4
# (lead patterns + body co-mention). For StackExchange we cannot use
# the question title -- it's interrogative-shape. Instead we derive
# a declarative anchor from the answer body itself:
#
#   1. Find the most-frequent capitalised noun-phrase (1-4 words).
#   2. If none, fall back to the first declarative noun head.
#   3. If still nothing, return None and skip the article.
#
# Anchor quality matters: a noisy anchor produces noisy L4 co-mention
# bonds. We bias toward proper-noun-style phrases (capitalised, not
# sentence-initial, not single-letter).
# ------------------------------------------------------------------
_NP_TOKEN = re.compile(r"\b[A-Z][A-Za-z0-9+\-]{1,30}(?:\s+[A-Z][A-Za-z0-9+\-]{1,30}){0,3}\b")

_STOP_ANCHORS = {
    "i", "the", "a", "an", "this", "that", "these", "those",
    "you", "we", "they", "he", "she", "it",
    "yes", "no", "ok", "edit", "update", "note", "tldr", "tl;dr",
    "however", "but", "and", "or", "so", "then",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
}


def derive_title_anchor(text: str) -> Optional[str]:
    if not text:
        return None
    candidates: Counter = Counter()
    for m in _NP_TOKEN.finditer(text):
        phrase = m.group(0).strip()
        low = phrase.lower()
        if low in _STOP_ANCHORS:
            continue
        if len(phrase) < 3:
            continue
        # Sentence-initial bias: penalise phrases that immediately follow
        # ". " (likely sentence-start capitalisation, not a proper noun).
        candidates[phrase] += 1
    if not candidates:
        return None
    # Prefer multi-word over single-word at equal frequency; longer phrases
    # are more specific and produce cleaner L4 anchors.
    best = sorted(
        candidates.items(),
        key=lambda kv: (-kv[1], -len(kv[0].split()), kv[0].lower()),
    )[0]
    phrase, count = best
    if count < 1:
        return None
    return phrase


# ------------------------------------------------------------------
# XML stream.
# ------------------------------------------------------------------
def stream_answers(
    xml_path: str,
    min_score: int = 5,
    max_answers: Optional[int] = None,
) -> Iterator[Dict]:
    """Yield answer-post dicts from a StackExchange Posts.xml stream.

    Posts.xml top-level: <posts><row PostTypeId="..." Score="..."
    Body="..." Id="..." ParentId="..." CreationDate="..." /></posts>.

    Yielded dict: {id, parent_id, score, body, creation_date}.
    """
    yielded = 0
    context = ET.iterparse(xml_path, events=("end",))
    for event, elem in context:
        if elem.tag != "row":
            continue
        try:
            post_type = elem.get("PostTypeId")
            if post_type != "2":  # answers only
                elem.clear()
                continue
            if elem.get("DeletionDate"):  # tombstoned post
                elem.clear()
                continue
            score = int(elem.get("Score") or "0")
            if score < min_score:
                elem.clear()
                continue
            body = elem.get("Body") or ""
            if not body.strip():
                elem.clear()
                continue
            yield {
                "id": elem.get("Id"),
                "parent_id": elem.get("ParentId"),
                "score": score,
                "body": body,
                "creation_date": elem.get("CreationDate") or "",
            }
            yielded += 1
            if max_answers is not None and yielded >= max_answers:
                elem.clear()
                return
        finally:
            elem.clear()


# ------------------------------------------------------------------
# Body hash for dedup of verbatim-identical bodies.
# ------------------------------------------------------------------
def body_text_hash(text: str) -> str:
    norm = re.sub(r"\s+", " ", text or "").strip().lower()
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


# ------------------------------------------------------------------
# Bond extraction for one answer.
# ------------------------------------------------------------------
def extract_bonds_for_answer(
    answer: Dict,
    site: str,
) -> Tuple[List[Dict], Dict]:
    """Return (bonds, diagnostics) for a single answer post.

    Diagnostics: counts of sentences in, sentences after q-filter, anchor
    derived, etc. Useful for the smoke-test audit table.
    """
    diag = {
        "post_id": answer["id"],
        "score": answer["score"],
        "had_code": False,
        "sentences_in": 0,
        "sentences_after_qfilter": 0,
        "anchor": None,
        "bonds_emitted": 0,
        "skipped_reason": None,
    }

    # Track whether the raw HTML contained code blocks (for diagnostics).
    if "<code>" in (answer["body"] or "") or "<pre>" in (answer["body"] or ""):
        diag["had_code"] = True

    text = html_to_text(answer["body"])
    if not text:
        diag["skipped_reason"] = "empty_after_html_strip"
        return [], diag

    sents = split_sentences(text)
    diag["sentences_in"] = len(sents)
    sents = filter_question_frame_sentences(sents)
    diag["sentences_after_qfilter"] = len(sents)
    if not sents:
        diag["skipped_reason"] = "all_sentences_question_framed"
        return [], diag

    filtered_text = " ".join(sents)
    anchor = derive_title_anchor(filtered_text)
    if not anchor:
        diag["skipped_reason"] = "no_anchor"
        return [], diag
    diag["anchor"] = anchor

    article = {
        "title": anchor,
        "text": filtered_text,
        "url": f"https://{site}/a/{answer['id']}",
    }
    try:
        raw_bonds = smart_extractor.extract_bonds_from_wikipedia_article(
            article, l4_max_emit=80
        )
    except Exception as e:
        diag["skipped_reason"] = f"extractor_error:{type(e).__name__}"
        return [], diag

    # Port B (R1) complementary pass: pronoun-anchored bonds and
    # multi-sentence Pattern A/B bonds that smart_extractor's
    # title-alias filter would have dropped. Union + dedup on
    # (subject.lower(), relation, object.lower()). smart_extractor
    # wins on shared triples.
    try:
        extra_bonds = extract_response_bonds_multi_sentence(
            article["title"], article["text"], article["url"]
        )
    except Exception:
        extra_bonds = []
    if extra_bonds:
        seen_triples = set(
            ((b.get("subject") or "").lower(),
             b.get("relation") or "",
             (b.get("object") or "").lower())
            for b in raw_bonds
        )
        for eb in extra_bonds:
            key = (eb["subject"].lower(), eb["relation"], eb["object"].lower())
            if key in seen_triples:
                continue
            seen_triples.add(key)
            raw_bonds.append(eb)

    bonds: List[Dict] = []
    for b in raw_bonds:
        # Defence-in-depth: drop any emitted bond whose subject still
        # contains a question-frame token. The pre-filter should make
        # this impossible, but the asymmetry gate is load-bearing so we
        # belt-and-braces it.
        subj = (b.get("subject") or "").lower()
        if any(tok in subj for tok in QUESTION_FRAME_INLINE_TOKENS):
            continue
        # Code-contamination guard.
        if any(tok in subj for tok in CODE_SHAPE_TOKENS):
            continue
        b2 = dict(b)
        b2["source"] = site
        b2["source_id"] = answer["id"]
        bonds.append(b2)
    diag["bonds_emitted"] = len(bonds)
    return bonds, diag


# ------------------------------------------------------------------
# Throttle.
# ------------------------------------------------------------------
class Throttle:
    def __init__(self, rate_hz: float):
        self.min_interval = 1.0 / rate_hz if rate_hz > 0 else 0.0
        self.last = 0.0

    def wait(self):
        if self.min_interval <= 0:
            return
        now = time.monotonic()
        sleep_for = self.last + self.min_interval - now
        if sleep_for > 0:
            time.sleep(sleep_for)
        self.last = time.monotonic()


# ------------------------------------------------------------------
# MCP client -- subprocess JSON-RPC over stdio. Mirrors the pattern
# already in ingest_curriculum_full.py / ingest_curriculum_1pct.py so
# lu_4_mcp sees byte-identical client behaviour.
# ------------------------------------------------------------------
class McpClient:
    def __init__(
        self,
        brain_path: str,
        keyfile: str,
        no_save: bool,
        lu4d_bin: str = DEFAULT_LU4D_BIN,
        stderr_to: Optional[str] = None,
    ):
        self.brain_path = brain_path
        self.keyfile = keyfile
        self._stderr = (
            open(stderr_to, "ab") if stderr_to else subprocess.DEVNULL
        )
        argv = [
            lu4d_bin, "--mcp", "--require-encrypted",
            "--keyfile", keyfile, "--brain", brain_path,
            "--brain-format-out", "v3enc",
        ]
        if no_save:
            argv.append("--no-save")
        self.proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr,
            bufsize=1,
            text=True,
        )
        self._next_id = 1
        self._init()

    def _send(self, obj):
        self.proc.stdin.write(json.dumps(obj) + "\n")
        self.proc.stdin.flush()

    def _recv(self):
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("MCP server closed stdout unexpectedly")
        return json.loads(line)

    def _init(self):
        self._send({
            "jsonrpc": "2.0", "id": self._next_id, "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "stackexchange-loader", "version": "0.1"},
            },
        })
        self._next_id += 1
        self._recv()
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def call(self, name: str, args: Dict):
        rid = self._next_id
        self._next_id += 1
        self._send({
            "jsonrpc": "2.0", "id": rid, "method": "tools/call",
            "params": {"name": name, "arguments": args},
        })
        return self._recv()

    def close(self):
        try:
            if self.proc.stdin and not self.proc.stdin.closed:
                self.proc.stdin.close()
            self.proc.wait(timeout=300)
        except Exception:
            self.proc.kill()
        finally:
            if hasattr(self._stderr, "close"):
                self._stderr.close()


def call_ok(resp) -> bool:
    return isinstance(resp, dict) and "error" not in resp


# ------------------------------------------------------------------
# Progress / dedup state.
# ------------------------------------------------------------------
class ProgressState:
    """Tracks already-ingested post_ids and verbatim-body hashes.

    Layout: JSONL file at progress_path; one record per ingested post.
    Record shape:
        {"post_id": "...", "site": "...", "body_hash": "...",
         "bonds_landed": N, "ts": "..."}

    Loaded on start; appended-to as posts are ingested. Hash maps a
    body_hash -> list of (site, post_id) so two answers with identical
    body still keep both source_ids.
    """

    def __init__(self, path: str):
        self.path = path
        self.seen_post_ids: set = set()
        self.body_hash_index: Dict[str, List[Tuple[str, str]]] = {}
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        if os.path.exists(path):
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    pid = rec.get("post_id")
                    site = rec.get("site") or ""
                    bh = rec.get("body_hash")
                    if pid:
                        self.seen_post_ids.add(f"{site}:{pid}")
                    if bh:
                        self.body_hash_index.setdefault(bh, []).append(
                            (site, pid or "")
                        )

    def already_done(self, site: str, post_id: str) -> bool:
        return f"{site}:{post_id}" in self.seen_post_ids

    def body_is_duplicate(self, body_hash: str) -> bool:
        return body_hash in self.body_hash_index

    def mark_done(self, site: str, post_id: str, body_hash: str,
                  bonds_landed: int):
        self.seen_post_ids.add(f"{site}:{post_id}")
        self.body_hash_index.setdefault(body_hash, []).append((site, post_id))
        with open(self.path, "a") as f:
            f.write(json.dumps({
                "post_id": post_id,
                "site": site,
                "body_hash": body_hash,
                "bonds_landed": bonds_landed,
                "ts": datetime.now(timezone.utc).isoformat(),
            }) + "\n")


# ------------------------------------------------------------------
# Main ingest loop.
# ------------------------------------------------------------------
def run(args: argparse.Namespace) -> int:
    # Diagnostic dump (always written, separate from progress).
    diag_path = args.diag_out
    diag_fh = open(diag_path, "w") if diag_path else None

    prog = ProgressState(args.progress)

    client = None
    if args.connect:
        client = McpClient(
            brain_path=args.brain,
            keyfile=args.keyfile,
            no_save=args.no_save,
            lu4d_bin=args.lu4d_bin,
            stderr_to=args.mcp_stderr,
        )
    throttle = Throttle(args.throttle_hz)

    site = args.site
    bonds_landed = 0
    bonds_errors = 0
    bonds_emitted = 0
    answers_processed = 0
    answers_skipped_dedup = 0
    answers_skipped_no_anchor = 0
    answers_skipped_q_only = 0
    start = time.monotonic()

    print(f"[{datetime.now(timezone.utc).isoformat()}] "
          f"stackexchange_loader: site={site} xml={args.xml_path} "
          f"min_score={args.min_score} max={args.max_answers} "
          f"no_save={args.no_save} connect={args.connect}",
          flush=True)

    for answer in stream_answers(
        args.xml_path,
        min_score=args.min_score,
        max_answers=args.max_answers,
    ):
        pid = answer["id"]

        if prog.already_done(site, pid):
            answers_skipped_dedup += 1
            continue

        bonds, diag = extract_bonds_for_answer(answer, site)
        answers_processed += 1
        diag["site"] = site

        if diag.get("skipped_reason") == "no_anchor":
            answers_skipped_no_anchor += 1
        if diag.get("skipped_reason") == "all_sentences_question_framed":
            answers_skipped_q_only += 1

        # Body-text dedup (verbatim-identical bodies, rare but real).
        cleaned_text = html_to_text(answer["body"])
        bh = body_text_hash(cleaned_text)
        if prog.body_is_duplicate(bh):
            # Mark this post_id as done with 0 bonds; record the
            # additional source_id under the same body_hash so future
            # audits can reconstruct that the body was seen twice.
            prog.mark_done(site, pid, bh, 0)
            answers_skipped_dedup += 1
            if diag_fh:
                diag["skipped_reason"] = "body_duplicate"
                diag_fh.write(json.dumps(diag) + "\n")
            continue

        post_bonds_landed = 0
        for b in bonds:
            bonds_emitted += 1
            if not client:
                continue
            throttle.wait()
            try:
                resp = client.call("lu4d_teach_corpus", {
                    "corpus": f"stackexchange:{site}",
                    "run_id": args.run_id,
                    "subject": b["subject"],
                    "relation": b["relation"],
                    "object": b["object"],
                    "weight": b.get("weight", 0.5),
                    "confidence": b.get("confidence", "Inferred"),
                    "kind": b.get("kind"),
                    "source_url": f"https://{site}/a/{pid}",
                })
                if call_ok(resp):
                    bonds_landed += 1
                    post_bonds_landed += 1
                else:
                    bonds_errors += 1
            except Exception:
                bonds_errors += 1

        prog.mark_done(site, pid, bh, post_bonds_landed)
        if diag_fh:
            diag_fh.write(json.dumps(diag) + "\n")

        if answers_processed % 25 == 0:
            elapsed = time.monotonic() - start
            print(f"  ...{answers_processed} answers processed "
                  f"({bonds_emitted} bonds emitted, "
                  f"{bonds_landed} landed, "
                  f"{bonds_errors} errors, "
                  f"{elapsed:.1f}s)", flush=True)

    elapsed = time.monotonic() - start
    if client:
        client.close()
    if diag_fh:
        diag_fh.close()

    summary = {
        "site": site,
        "xml_path": args.xml_path,
        "min_score": args.min_score,
        "max_answers": args.max_answers,
        "no_save": args.no_save,
        "connect": args.connect,
        "answers_processed": answers_processed,
        "answers_skipped_dedup": answers_skipped_dedup,
        "answers_skipped_no_anchor": answers_skipped_no_anchor,
        "answers_skipped_q_only": answers_skipped_q_only,
        "bonds_emitted": bonds_emitted,
        "bonds_landed": bonds_landed,
        "bonds_errors": bonds_errors,
        "elapsed_s": round(elapsed, 2),
        "run_id": args.run_id,
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    print(json.dumps(summary, indent=2), flush=True)
    if args.summary_out:
        with open(args.summary_out, "w") as f:
            json.dump(summary, f, indent=2)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="StackExchange Cat.1 retrain loader (answers only).",
    )
    ap.add_argument("--xml-path", required=True,
                    help="Path to Posts.xml from a StackExchange data dump.")
    ap.add_argument("--site", required=True,
                    help="Site label, e.g. stackoverflow.com or "
                         "stats.stackexchange.com -- used for source attribution.")
    ap.add_argument("--min-score", type=int, default=5,
                    help="Score floor for answers (default 5).")
    ap.add_argument("--max-answers", type=int, default=None,
                    help="Cap on answers processed this run.")
    ap.add_argument("--progress", default=DEFAULT_PROGRESS,
                    help="Resume/dedup state file (JSONL).")
    ap.add_argument("--diag-out", default=None,
                    help="Optional per-answer diagnostic JSONL dump.")
    ap.add_argument("--summary-out", default=None,
                    help="Optional summary JSON path.")
    ap.add_argument("--run-id", default=None,
                    help="Run identifier. Default: stackexchange-<site>-<ts>.")
    # MCP wiring
    ap.add_argument("--connect", action="store_true",
                    help="Connect to lu_4_mcp and actually send bonds. "
                         "If omitted, runs in dry-extract mode (bonds emitted "
                         "but not sent; useful for offline subject audits).")
    ap.add_argument("--no-save", action="store_true",
                    help="Pass --no-save to lu_4_mcp. MANDATORY for any smoke "
                         "run that should not mutate brain.lbr.")
    ap.add_argument("--brain", default=DEFAULT_BRAIN)
    ap.add_argument("--keyfile", default=DEFAULT_KEYFILE)
    ap.add_argument("--lu4d-bin", default=DEFAULT_LU4D_BIN)
    ap.add_argument("--mcp-stderr", default=None,
                    help="Path to dump lu_4_mcp stderr.")
    ap.add_argument("--throttle-hz", type=float, default=THROTTLE_HZ)
    args = ap.parse_args(argv)

    if args.run_id is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.run_id = f"stackexchange-{args.site}-{ts}"
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
