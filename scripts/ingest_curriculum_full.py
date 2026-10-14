#!/usr/bin/env python3
"""
Cat.1 full retrain ingest harness.

Forked from scripts/ingest_curriculum_1pct.py internal notes.
Deltas vs. 1pct:
  - Removes the auto-scale-down to 0.1%; takes FULL slice of each Tier 1 corpus.
  - Adds resumability via checkpoint file at /var/lib/lu/ingest_full_checkpoint.json.
    Checkpoint records {corpus, last_saved_id, bonds_landed, errors_count, ...}
    and is updated only after a successful lu4d_save (so the on-disk brain and
    the checkpoint are always in sync). Resume: stream from the start of each
    corpus and skip-and-discard until last_saved_id is reached.
  - Per-corpus throttle: 30/s Wikipedia, 50/s Visual Genome, 30/s Tier 2.
  - Periodic lu4d_save every ~30 min of in-corpus work (or every 5000 bonds,
    whichever comes first); brain + WALs flushed each save.
  - Three-tier order: Visual Genome FIRST (validates quickly), then full
    Wikipedia, then Tier 2 (no_robots + instruction-dataset). Wikipedia-
    structured / arXiv / StackExchange / Grokopedia are DEFERRED to follow-up.
  - Logs to /var/lib/lu/ingest_full_progress.log with timestamp, bonds_landed,
    corpus, errors_count, ingest_rate.
  - On startup, writes a sentinel /var/lib/lu/ingest_full_DONE on clean exit.

DEVIATIONS FROM PHASE-3 SPEC (documented per C11 demo honesty):
  1. Subprocess.Popen pattern (inherited from 1pct), NOT mkfifo + backgrounded
     server. The 1pct author documented this same deviation: bidirectional
     JSON-RPC over a FIFO needs a response-matching layer that subprocess.Popen
     gives for free. Same long-lived-server outcome; one PID to monitor.
  2. lu4d_configure { ingest_auto_approve_threshold: 0.7 } is a no-op against
     the current binary (md5 07a96d2…f8d, git 1d0d945) — that field is not in
     lu4d_configure's input schema. lu4d_teach_corpus lands bonds directly with
     no pending-queue gating in this binary. The 1pct run successfully landed
     bonds without setting any threshold. Kept the trust hierarchy + hub caps,
     dropped the no-op call.
  3. wikimedia/structured-wikipedia is in the trust table but has no loader in
     the 1pct harness; treating it as deferred along with arXiv/SE/Grokopedia.

INVOCATION (BR.10b+ production retrain):
  python3 scripts/ingest_curriculum_full.py --seed-trust
  The --seed-trust flag is REQUIRED for production retrain. Without it,
  corpus_multipliers stays empty and every bond resolves to
  default_multiplier=0.4 instead of the seeded curriculum trust hierarchy.
  Smoke runs (--smoke N) still ingest correctly without --seed-trust;
  bonds just land at the 0.4 default. Use --seed-trust on smoke runs too
  if you want to validate trust-resolved weights end-to-end.

Resumability semantics:
  - The checkpoint represents the last KNOWN-PERSISTED position. After a
    successful lu4d_save, the checkpoint is updated to record the last
    processed corpus id at the time of save. On crash, restart of this
    harness re-spawns the MCP server (which loads brain at last-saved state)
    and skips the stream forward to last_saved_id. Up to ~30 min of
    redundant streaming may occur on resume; bonds are not double-counted at
    the brain level because the MCP server reloaded from a clean save point.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

# Phase B (2026-05-07): smart extractor v2 (iter-4, subject-anchored).
# Adds six-layer Wikipedia bond extraction. The old
# extract_wikipedia_bonds function is preserved below for reference
# but is no longer the call site -- ingest_wikipedia uses
# extract_bonds_from_wikipedia_article instead. K-gate, trust weights,
# checkpoint logic, hub caps and BatchSink are all unchanged.
import sys as _sys, os as _os
_SCRIPTS_DIR = _os.path.dirname(_os.path.abspath(__file__))
if _SCRIPTS_DIR not in _sys.path:
    _sys.path.insert(0, _SCRIPTS_DIR)
from smart_extractor import extract_bonds_from_wikipedia_article, collapse_to_richest_relation

PROGRESS_LOG = "/var/lib/lu/ingest_full_progress.log"
CHECKPOINT   = "/var/lib/lu/ingest_full_checkpoint.json"
DONE_SENTINEL = "/var/lib/lu/ingest_full_DONE"

RUN_TIMESTAMP = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
RUN_ID_TIER1  = f"cat1-retrain-tier1-{RUN_TIMESTAMP}"
RUN_ID_TIER2  = f"cat1-retrain-tier2-{RUN_TIMESTAMP}"

# Per-corpus pacing caps (calls/sec). Effective rate is min(cap, capability_rate).
THROTTLE_WIKIPEDIA = 30.0
THROTTLE_VG        = 50.0
THROTTLE_TIER2     = 30.0

# Periodic-save policy: whichever fires first.
SAVE_INTERVAL_SEC  = 30 * 60
SAVE_INTERVAL_BONDS = 50000


class Throttle:
    def __init__(self, rate_hz):
        self.min_interval = 1.0 / rate_hz
        self.last = 0.0
    def wait(self):
        now = time.monotonic()
        sleep_for = self.last + self.min_interval - now
        if sleep_for > 0:
            time.sleep(sleep_for)
        self.last = time.monotonic()


# Port BatchIngest -- see lu4d_teach_corpus_batch in src/mcp.rs.
# Cap is enforced server-side at TEACH_CORPUS_BATCH_MAX_ENTRIES=1000.
# 100 is 10x headroom; gives ~1.5kB-3kB JSON-RPC payloads per batch.
BATCH_SIZE = 100


# =====================================================================
# CORPUS ATTRIBUTION (BR.5 audit P1 #2 -- internal notes:
#
# Every entry submitted via BatchSink.add MUST include a 'corpus' key.
# It is set BY THIS FILE -- the harness/ingest wrapper -- at each
# sink.add({...}) call-site, NOT by the BondSink/sink layer and NOT
# by the bond extractor. The smart_extractor and visual_genome_loader
# emit (subject, relation, object) triples with no corpus opinion;
# the 'corpus' string is stamped in by the per-corpus ingest function
# below (ingest_visual_genome / ingest_wikipedia / ingest_tier2_dataset)
# using the local corpus_id constant declared at the top of that
# function.
#
# Values stamped today (must stay in sync with revoke_by_source
# selectors and with the corpus enum in lu_4_mcp's TrainingCorpus
# allow-list):
#   - 'ranjaykrishna/visual_genome' (ingest_visual_genome,  Tier 1 vision; matches TrustTable::seed_curriculum_corpora key)
#   - 'wikimedia/wikipedia'   (ingest_wikipedia,      Tier 1 text)
#   - <Tier 2 dataset_repo>   (ingest_tier2_dataset,  e.g.
#     'HuggingFaceH4/no_robots', 'HuggingFaceH4/instruction-dataset')
#
# The server (lu_4_mcp's tool_teach_corpus) requires 'corpus' as a
# required JSON arg and stamps SourceAttribution::TrainingCorpus
# with it. Brain rejects entries that arrive with corpus='<unknown>'
# (the default sentinel) -- this file's invariant is what keeps
# attribution non-empty in canonical.
#
# WHY THIS MATTERS FOR FUTURE REFACTORS: lu4d_revoke_by_source uses
# 'corpus' as a primary selector (see src/mcp.rs tool_revoke_by_source
# 'training_corpus'/'document' branches). If a future refactor lifts
# the stamp into a wrapper, decorator, or BatchSink default that
# silently drops it on one path, EVERY bond from that path becomes
# unrevokable by corpus -- attribution silently breaks. Audit the
# call-sites below before changing this file.
# =====================================================================
class BatchSink:
    """Buffer-and-submit wrapper around lu4d_teach_corpus_batch.

    Maintains a per-process seen_triples set for Python-side dedup --
    skips submission entirely when (subject, relation, object) was
    already submitted in this process. The set does NOT persist
    across daemon restarts; checkpoint resume + lu_4_mcp's per-bond
    dedup (existing-edge reinforcement) catch any duplicates that
    re-submit on resume.

    Anti-bypass invariant: each entry in `add()` is submitted via the
    SAME lu4d_teach_corpus_batch tool; that tool routes per-entry
    through tool_teach_corpus, which goes through brain.teach_corpus
    (T.3 gating + TrainingCorpus stamping). The batch is purely an
    IPC speedup -- no per-bond logic is bypassed.
    """

    def __init__(self, client, throttle, batch_size=BATCH_SIZE):
        self.client = client
        self.throttle = throttle
        self.batch_size = batch_size
        self.buffer = []
        self.seen = set()
        # Cumulative counters --- mirror response structuredContent.
        self.bonds_landed = 0       # new bonds from response
        self.bonds_existing = 0     # reinforced existing bonds
        self.bonds_gated = 0        # T.3 PendingLog (below threshold)
        self.bonds_rejected = 0     # OP-RETRAIN-1 concept-admission-gate rejects
        self.bonds_errors = 0       # per-entry errors + whole-batch failures
        self.bonds_skipped_dedup = 0  # Python-side dedup hits

    def add(self, entry):
        """Buffer one entry. Returns False if Python-side dedup skipped
        it OR the entry was malformed; True if it was queued."""
        s = entry.get("subject"); r = entry.get("relation"); o = entry.get("object")
        if not s or not r or not o:
            self.bonds_errors += 1
            return False
        key = (s, r, o)
        if key in self.seen:
            self.bonds_skipped_dedup += 1
            return False
        self.seen.add(key)
        self.buffer.append(entry)
        if len(self.buffer) >= self.batch_size:
            self.flush()
        return True

    def flush(self):
        """Submit the current buffer in one IPC call. No-op if empty."""
        if not self.buffer:
            return
        n = len(self.buffer)
        self.throttle.wait()
        try:
            resp = self.client.call("lu4d_teach_corpus_batch", {"entries": self.buffer})
            if call_ok(resp):
                sc = ((resp.get("result") or {}).get("structuredContent") or {})
                self.bonds_landed   += int(sc.get("bonds_landed", 0))
                self.bonds_existing += int(sc.get("bonds_deduped", 0))
                self.bonds_gated    += int(sc.get("bonds_gated", 0))
                self.bonds_rejected += int(sc.get("bonds_rejected", 0))
                errs = sc.get("errors") or []
                self.bonds_errors   += len(errs)
            else:
                # Whole-batch failure -- count every queued entry.
                self.bonds_errors += n
        except Exception:
            self.bonds_errors += n
        finally:
            self.buffer = []


# OP-011 (2026-05-11): McpChildDied -- raise when the lu_4_mcp child dies.
# Inherits SystemExit so it propagates past `except Exception:` blocks in
# BatchSink.flush etc. and exits the parent process with a non-zero code,
# letting systemd Restart=on-failure pick up the runner cleanly.
class McpChildDied(SystemExit):
    pass


class McpClient:
    def __init__(self, brain_path, keyfile, log_stderr_to=None, ingest_only=False):
        self.brain_path = brain_path
        self.keyfile = keyfile
        self._stderr = open(log_stderr_to, "ab") if log_stderr_to else subprocess.DEVNULL
        # OP-022.eh: glibc arena tuning for the lu_4_mcp child.
        # MALLOC_ARENA_MAX=2 caps glibc to 2 arenas (default = nproc, often 8-32)
        # to reduce cross-save fragmentation. MALLOC_TRIM_THRESHOLD_=131072 (128 KiB)
        # makes glibc more aggressive about returning freed pages to the OS instead
        # of retaining them in the arena. Both env vars must be set BEFORE glibc
        # initializes in the child process; passing them via Popen.env (which
        # becomes the child execve environment) achieves that.
        op022eh_env = os.environ.copy()
        op022eh_env.setdefault("MALLOC_ARENA_MAX", "2")
        op022eh_env.setdefault("MALLOC_TRIM_THRESHOLD_", "131072")
        # Daemon Perf-2 (2026-05-07): pass --no-save so the daemon does
        # NOT autosave the full brain after every tools/call. Saves are
        # still controlled by the harness via _maybe_save (every
        # SAVE_INTERVAL_BONDS or SAVE_INTERVAL_SEC) and the explicit
        # lu4d_save at the end of each ingest_* function. Per-batch
        # autosave was writing 944 MB/save and dominating wall time
        # (~16 sec/batch, ~0.11 art/sec). With --no-save the same
        # round-trip is ~0.25 sec (~7 art/sec). Crash safety is preserved
        # by checkpoint resume: bonds added between saves are re-formed
        # by re-extracting from last_saved_id+1.
        cmd = [os.environ.get("LU4D_BIN", "/opt/lu/lu/merge-main/target/release/lu4d"), "--mcp", "--require-encrypted",
               "--keyfile", keyfile, "--brain", brain_path,
               "--brain-format", "v5",
               "--no-save"]
        # OP-INGEST-LEAN: write-only ingest boots the v5 brain WITHOUT the
        # serving/recall residency (resident degree-map, cache-backed
        # adjacency + seam, keyword index). Bonds still land durably; the
        # serving structures are rebuilt when Lu is served.
        if ingest_only:
            cmd += ["--load-mode", "ingest-only"]
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._stderr,
            bufsize=1, text=True,
            env=op022eh_env,
        )
        self._next_id = 1
        self._init()
    def _send(self, obj):
        # OP-011: detect child death before write so a dead pipe converts
        # into McpChildDied (SystemExit) rather than a swallowed BrokenPipeError.
        rc = self.proc.poll()
        if rc is not None:
            try: log_progress(f"OP-011 McpChildDied on _send: returncode={rc}")
            except Exception: pass
            raise McpChildDied(f"lu_4_mcp child died (returncode={rc}) before _send")
        try:
            self.proc.stdin.write(json.dumps(obj) + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            rc = self.proc.poll()
            try: log_progress(f"OP-011 McpChildDied on _send write: rc={rc} err={e}")
            except Exception: pass
            raise McpChildDied(f"lu_4_mcp child died during _send (returncode={rc}, err={e})")
    def _recv(self):
        # OP-011: empty readline means child closed stdout (typically died).
        # Convert to McpChildDied so the parent exits cleanly for systemd restart.
        line = self.proc.stdout.readline()
        if not line:
            rc = self.proc.poll()
            try: log_progress(f"OP-011 McpChildDied on _recv: empty line, returncode={rc}")
            except Exception: pass
            raise McpChildDied(f"lu_4_mcp child closed stdout (returncode={rc})")
        return json.loads(line)
    def _init(self):
        self._send({"jsonrpc":"2.0","id":self._next_id,"method":"initialize",
                    "params":{"protocolVersion":"2024-11-05","capabilities":{},
                             "clientInfo":{"name":"ingest-full","version":"0.1"}}})
        self._next_id += 1
        self._recv()
        self._send({"jsonrpc":"2.0","method":"notifications/initialized"})
    def call(self, name, args):
        rid = self._next_id
        self._next_id += 1
        self._send({"jsonrpc":"2.0","id":rid,"method":"tools/call",
                    "params":{"name":name,"arguments":args}})
        return self._recv()
    def close(self):
        try:
            if self.proc.stdin and not self.proc.stdin.closed:
                self.proc.stdin.close()
            self.proc.wait(timeout=600)
        except Exception:
            self.proc.kill()
        finally:
            if hasattr(self._stderr, "close"):
                try: self._stderr.close()
                except: pass


def log_progress(line):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        with open(PROGRESS_LOG, "a") as f:
            f.write(f"{ts} {line}\n"); f.flush()
    except Exception:
        pass
    print(f"[{ts}] {line}", flush=True)


def call_ok(resp):
    return isinstance(resp, dict) and "error" not in resp


# ---- checkpoint ----

def load_checkpoint():
    if not os.path.exists(CHECKPOINT):
        return {}
    try:
        with open(CHECKPOINT) as f:
            return json.load(f)
    except Exception as e:
        log_progress(f"WARNING: checkpoint load failed: {e}; starting fresh")
        return {}

def save_checkpoint(cp):
    cp["updated_at"] = datetime.now(timezone.utc).isoformat()
    tmp = CHECKPOINT + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cp, f, indent=2)
    os.replace(tmp, CHECKPOINT)


# ---- bond extraction (same as 1pct) ----

IS_A_PATTERN = re.compile(
    r"\b(?:is|was|are|were)\s+(?:an?|the)\s+([a-z][\w\s\-]{2,60}?)\s+(?:that|which|who|where|in|of|on|located|known|used|with|by|for|from|to|and|\.|,|;)",
    re.IGNORECASE,
)

def first_sentence(text):
    if not text: return ""
    m = re.match(r"([^.!?\n]{20,400}[.!?])", text.strip())
    return m.group(1).strip() if m else text.strip()[:200]

def extract_wikipedia_bonds(title, text, source_url):
    bonds = []
    if not text or not title: return bonds
    lead = first_sentence(text)
    if not lead: return bonds
    # Phase-1 (2026-05-30): the legacy (title, RelatedTo, lead[:200])
    # emission was structurally self-referential noise — the lead sentence
    # is ABOUT the title, not a separate entity it relates to. Removed.
    # IS_A_PATTERN below provides typed-relation coverage cleanly.
    m = IS_A_PATTERN.search(lead)
    if m:
        cat = m.group(1).strip().rstrip(",.;:").lower()
        if 2 <= len(cat) <= 60:
            bonds.append((title, "IsA", cat, 0.6, "Inferred"))
    return bonds

# Lu BR.8b -- asymmetry fix attempt 2
# Two patterns only (A: determiner-led NP, B: capitalized-leading NP), each
# requiring a predicate verb to follow. If neither matches the first sentence
# of the response, return None and drop the bond at the call site -- correct
# behavior for non-declarative responses (lists, code, imperatives, etc.).
#
# A blacklist on the NP head filters out WH-interrogatives (what/who/when/...),
# bare pronouns (it/he/she/they/...), demonstrative pronouns (this/that/...),
# existential/locative (there/here), and common sentence-initial discourse
# markers (however/therefore/moreover/...). Without this filter, Pattern B
# would match "What is X?" -> "What" (interrogative leakage, hard fail).

_DETS_RE = r"(?:[Tt]he|[Aa]n?|[Tt]his|[Tt]hat|[Tt]hese|[Tt]hose|[Mm]y|[Yy]our|[Hh]is|[Hh]er|[Ii]ts|[Oo]ur|[Tt]heir|[Ss]ome|[Aa]ny|[Ee]ach|[Ee]very|[Aa]ll|[Bb]oth|[Nn]o)"

_PRED_RE = r"(?:is|was|are|were|be|being|been|am|has|have|had|does|do|did|will|would|should|could|can|may|might|must|means|refers|denotes|represents|includes|contains|involves|consists|comprises|signifies|describes|equals|defines|marks|originates|derives|originated|derived|occurred|happens|happened|exists|existed|came|comes|became|becomes|remains|stayed|stays|stood|stands)"

_PAT_A_RESP = re.compile(
    # BR.9 + BR.10b: NP-greedy fix -- block extra NP tokens from spanning
    # modals (could/would/should/might/may/must) and relative-clause openers
    # (that/which/who/whose/whom). BR.10b widens the BR.9 set after the
    # 'might be true but you' residual surfaced in the BR.9 follow-up.
    # Predicate verb captured in named group 'pred' for typed-relation routing.
    r"^\s*" + _DETS_RE
    + r"\s+([A-Za-z][\w'-]*(?:\s+(?!(?:could|would|should|might|may|must|that|which|who|whose|whom)\b)[A-Za-z][\w'-]*){0,4})\s+"
    + r"(?P<pred>" + _PRED_RE + r")\b",
    re.UNICODE,
)

_PAT_B_RESP = re.compile(
    # BR.9: predicate verb captured in named group 'pred' for typed-relation routing.
    r"^\s*([A-Z][\w'-]*(?:\s+[A-Z][\w'-]*){0,2})\s+"
    + r"(?P<pred>" + _PRED_RE + r")\b",
    re.UNICODE,
)

_FORBIDDEN_SUBJ_HEAD = frozenset([
    # WH-interrogatives -- hard fail criterion
    "what", "who", "when", "where", "why", "how", "which", "whose", "whom",
    # Bare pronouns -- anaphoric, no semantic content
    "it", "he", "she", "they", "we", "you", "i", "me", "us", "him", "them",
    # Demonstrative pronouns when not used as determiners
    "this", "that", "these", "those",
    # Existential / locative
    "there", "here",
    # Sentence-initial discourse markers / fillers
    "however", "therefore", "moreover", "furthermore", "indeed", "though",
    "although", "because", "since", "while", "yes", "no", "well", "so",
    "yet", "still", "anyway", "additionally", "also", "finally", "lastly",
    "first", "second", "third", "next", "then", "now", "basically",
    "essentially", "actually", "perhaps", "maybe",
])

# Reconciliation port (2026-05-10): bare-frame-noun subject filter promoted
# from scripts/arxiv_loader.py internal notes. Pattern A admits
# "This paper / The study / Our work / Our method" and captures the bare
# meta-frame noun ("paper", "study", "work", "method") as np_text.
# These are semantically meta and have no anchoring; the existing
# _FORBIDDEN_SUBJ_HEAD blacklist does not cover them because they are not
# WH-words / pronouns / discourse markers. Reject bare-frame-noun subjects
# (whole-string equality on np_text, NOT head match -- so multi-word
# expressions like "paper currency" or "study results" remain admitted).
#
# Originally caught in 2026-05-10 arxiv smoke: arxiv 2312.17258 produced
# (subj="paper", IsA, ...) from "This paper describes a machine learning
# approach...". Promoting the filter here (instead of leaving it loader-local)
# means every caller of extract_response_bond benefits -- current arxiv
# Cat.1 path, and any future declarative-frame ingest including the
# queued Port B (first-sentence-per-paragraph extension to
# extract_response_bond; the filter applies per-sentence so multi-paragraph
# iteration inherits it without change).
_FRAME_NOUN_SUBJECTS = frozenset({
    "paper", "study", "work", "article", "manuscript", "report",
    "thesis", "dissertation", "letter", "preprint", "draft",
    "method", "approach", "technique", "framework", "model",
    "system", "scheme", "algorithm", "procedure",
    # "method" / "model" / "system" etc. are debatable: they ARE real
    # CS nouns, but as standalone bond subjects from declarative prose
    # they are almost always meta-frame (from "Our method", "The model",
    # "This system" leading sentences). Cost of dropping a rare true
    # single-word bond is low; cost of admitting frame leakage is the
    # asymmetry-boundary failure (Cat.1 hard fail).
})


# ============================================================
# Phase-1 NP truncator (2026-05-30) — replaces raw [:200] tail dumps
# in the response-bond extractors. The pre-fix output was sentence
# fragments, which the hub-fill synthesis port then amplified into
# hallucination bonds. See feedback_synthesis_bounded_by_node_quality
# in agent memory for the audit data that motivated this.
# ============================================================
_NP_STOP_PUNCT_RE = re.compile(r'[,;:—–]')
_NP_STOP_CONJUNCTION_RE = re.compile(
    r'\s+(?:because|although|while|that|which|who|where|when|if|as|since|though|so that|in order to)\s+',
    re.IGNORECASE,
)
_NP_PREDICATE_VERB_RE = re.compile(
    r'\b(?:is|was|are|were|has|have|had|will|would|could|should|did|does|do|seems|appears|became|becomes)\b',
    re.IGNORECASE,
)
_NP_LEADING_VERB_RE = re.compile(r'^(?:[a-z]+(?:ed|ing))\b', re.IGNORECASE)

def _truncate_to_np(text, max_len=60):
    """Truncate a raw text fragment to a clean noun phrase.

    Cuts at sentence-internal punctuation or subordinating conjunctions.
    Hard-caps to max_len chars with word-boundary alignment. Rejects
    (returns None) if the result still contains a predicate verb or
    starts with a verb form — both indicate a clause fragment, not an NP.

    Callers MUST drop the bond when this returns None.
    """
    if not text:
        return None
    text = text.strip().strip(',;:—–')
    if not text:
        return None
    # 1. Cut at first sentence-internal punctuation.
    m = _NP_STOP_PUNCT_RE.search(text)
    if m:
        text = text[:m.start()].strip()
    if not text:
        return None
    # 2. Cut at first subordinating conjunction.
    m = _NP_STOP_CONJUNCTION_RE.search(text)
    if m:
        text = text[:m.start()].strip()
    if not text:
        return None
    # 3. Hard cap with word-boundary alignment.
    if len(text) > max_len:
        cut = text[:max_len].rstrip()
        last_space = cut.rfind(' ')
        if last_space > max_len * 0.5:
            text = cut[:last_space].strip()
        else:
            text = cut
    if not text or len(text) < 2:
        return None
    # 4. Reject if still contains a finite predicate verb.
    if _NP_PREDICATE_VERB_RE.search(text):
        return None
    # 5. Reject if starts with a verb form (past participle / gerund).
    if _NP_LEADING_VERB_RE.match(text):
        return None
    return text


def _extract_response_subject(sentence):
    """Return (subject_NP, predicate_verb) from a response sentence, or None.

    BR.9: now returns a 2-tuple so the caller can pick a typed BondType
    based on the predicate verb that fired Pattern A or B. Pattern guarantees
    a non-empty pred capture.

    Pattern A: determiner-led NP + predicate verb.
        e.g. "The cat is sleeping" -> ("cat", "is")
    Pattern B: capitalized-leading NP + predicate verb.
        e.g. "Einstein was a physicist" -> ("Einstein", "was")
    NP head is rejected if it's a WH-word, bare pronoun, demonstrative,
    existential/locative, or discourse marker.
    No fallback. Bond is dropped at the call site if this returns None.
    """
    if not sentence:
        return None
    s = sentence.strip()
    if not s:
        return None
    for pat in (_PAT_A_RESP, _PAT_B_RESP):
        m = pat.match(s)
        if not m:
            continue
        np_text = m.group(1).strip()
        if not (2 <= len(np_text) <= 80):
            continue
        head = np_text.split()[0].lower().strip(",.;:!?")
        if head in _FORBIDDEN_SUBJ_HEAD:
            continue
        # Reconciliation port (2026-05-10): bare-frame-noun subject reject.
        # See _FRAME_NOUN_SUBJECTS comment block above. Whole-string equality,
        # NOT head match -- preserves multi-word NPs like "paper currency".
        if np_text.lower().strip() in _FRAME_NOUN_SUBJECTS:
            continue
        pred = (m.group("pred") or "").strip().lower()
        return (np_text, pred)
    return None


# BR.9 -- predicate-verb -> typed BondType mapping (Tier 2 typed-relation).
# Phrase-anchored variants take priority over verb-only defaults so that
# common biographical / locational / origination shapes (born in, located
# in, named after, founded by, discovered by, written by, known for, used
# for, part of, consists of) emit the typed BondType the smart_extractor
# already knows about, instead of the BR.8b 100% RelatedTo monoshape.
#
# Mapping uses ONLY relations already present in
# scripts/smart_extractor.py::_RELATION_TO_KIND so the Rust side parses
# them via infer_kind_from_relation in lock-step. Anything unmatched falls
# back to RelatedTo (correct for ambiguous prose).
def _relation_for_pred(pred_verb, response_tail):
    pv = (pred_verb or "").lower()
    tail = (response_tail or "").lower()

    # Phrase-anchored typed relations (refine over the verb-based default).
    if "born in" in tail or "born on" in tail:
        return "BornIn"
    if ("located in" in tail or "situated in" in tail
            or "based in" in tail or "headquartered in" in tail):
        return "LocatedIn"
    if ("known for" in tail or "famous for" in tail
            or "noted for" in tail or "celebrated for" in tail):
        return "KnownFor"
    if ("founded by" in tail or "established by" in tail
            or "created by" in tail or "developed by" in tail
            or "invented by" in tail):
        return "FoundedBy"
    if "discovered by" in tail:
        return "DiscoveredBy"
    if "named after" in tail or "named for" in tail:
        return "NamedAfter"
    if ("used for" in tail or "used as" in tail or "used in" in tail
            or "employed for" in tail or "employed in" in tail):
        return "UsedFor"
    if "part of" in tail or "member of" in tail:
        return "PartOf"
    if ("written by" in tail or "composed by" in tail
            or "directed by" in tail or "designed by" in tail
            or "produced by" in tail or "painted by" in tail):
        return "WrittenBy"

    # Verb-based defaults (no phrase-anchored cue fired).
    if pv in {"is", "was", "are", "were", "be", "being", "been", "am",
              "means", "refers", "denotes", "represents", "signifies",
              "describes", "equals", "defines", "became", "becomes"}:
        return "IsA"
    if pv in {"has", "have", "had",
              "remains", "stayed", "stays", "stood", "stands"}:
        return "HasProperty"
    if pv in {"includes", "contains", "comprises", "consists"}:
        return "ComposedOf"

    # Ambiguous prose -- preserve the BR.8b default.
    return "RelatedTo"


def extract_response_bond(prompt, response):
    if not response: return None
    s = first_sentence(response)
    if not s or len(s) < 10: return None
    parsed = _extract_response_subject(s)
    if not parsed:
        return None
    subj, pred = parsed
    # BR.9: typed BondType from the predicate verb that fired Pattern A/B,
    # refined by phrase-anchored cues in the rest of the response. Replaces
    # the BR.8b 100% RelatedTo hardcode.
    rel = _relation_for_pred(pred, response[:600])
    # Phase-1 (2026-05-30): extract the tail AFTER subj+pred as the
    # object candidate, then truncate to a clean NP. Pre-fix returned
    # `s[:200]` — the whole sentence prefix including the subject and
    # predicate themselves, producing self-referential prose-chunk bonds.
    tail_m = re.search(
        re.escape(subj) + r'\s+' + re.escape(pred) + r'\s+([^.!?\n]+)',
        s,
        re.IGNORECASE,
    )
    if not tail_m:
        return None
    obj = _truncate_to_np(tail_m.group(1).strip())
    if not obj:
        return None
    return (subj, rel, obj, 0.4, "Inferred")


# ============================================================
# Port B (R1) -- multi-sentence extractor for declarative corpora
# ============================================================
# Adds two paths the existing extract_response_bond cannot reach:
#   - Path 1: every sentence (not just first) gets Pattern A/B
#             via the shared _extract_response_subject. Frame-noun
#             and forbidden-head filters inherited automatically
#             (reconciliation port, internal notes.
#   - Path 2: pronoun-led sentences anchor-bond to the article
#             title. subject_origin="anchor" marks these for
#             post-retrain audit. The sentence's predicate+tail
#             still go through _relation_for_pred + final
#             editorial/citation/code/interrogative guard.
#   - Path 3: explicit junk-sentence filter (interrogative,
#             code, citation marker, editorial frame, frame-noun
#             fragment, quote-wrapped, too-short).
#
# Output: list of bond dicts matching smart_extractor's bond
# dict shape, plus a `subject_origin` field. Callers can union
# with smart_extractor output and dedup on
# (subject.lower(), relation, object.lower()).
#
# Sentence segmenter: pysbd 0.3.4 (pure-Python, handles
# abbreviations like "Dr.", "U.S.", "e.g."). Lazy-imported so
# stub environments without pysbd can still import this module.
# ============================================================

_PYSBD_SEG = None
def _get_segmenter():
    global _PYSBD_SEG
    if _PYSBD_SEG is None:
        import pysbd
        _PYSBD_SEG = pysbd.Segmenter(language="en", clean=False)
    return _PYSBD_SEG


_PORTB_PRONOUN_LEAD_RE = re.compile(
    r"^\s*(?:it|this|that|these|those|he|she|they|we|you|i)\b",
    re.IGNORECASE,
)

# Interrogative-shape leads. Mirrors stackexchange-loader's filter set.
_PORTB_INTERROGATIVE_LEAD_RE = re.compile(
    r"^\s*(?:how|what|why|when|where|who|whose|whom|which)\b",
    re.IGNORECASE,
)
_PORTB_QUESTION_END_RE = re.compile(r"\?\s*$")

# Code-shape tokens. Substring presence in the sentence is enough.
_PORTB_CODE_TOKEN_RE = re.compile(
    r"(?:\bdef\s+\w|\bfunction\s+\w|\bimport\s+\w|console\.log"
    r"|\bclass\s+\w|\basync\s+def|=>"
    r"|<\/?[a-z][\w-]*>|<[a-z][\w-]*\s+[^>]*>"
    r"|#include\b|#define\b|\bnpm\s+install\b|\bpip\s+install\b"
    r"|\bSELECT\s+\*\s+FROM\b|\bINSERT\s+INTO\b|\bUPDATE\s+\w+\s+SET\b)",
    re.IGNORECASE,
)

# Bracketed numeric citation markers ([1], [12], [3,4], [5-7]).
_PORTB_CITATION_MARKER_RE = re.compile(r"\[\d+(?:[,\s\-\d]*)\]")

# Editorial-frame tokens. Mirrors BatchSink editorial-frame guard
# from the loaders.
_PORTB_EDITORIAL_FRAME_RE = re.compile(
    r"\b(?:according to|as noted|see also|as mentioned|as discussed"
    r"|as described|this paper|this article|this study|this report"
    r"|this work|this thesis|this manuscript|the present paper"
    r"|this section|this chapter|i argue|we argue|we propose"
    r"|i propose|i claim|we claim|i suggest|we suggest)\b",
    re.IGNORECASE,
)

# Sentence fully wrapped in matched single/double quotes.
_PORTB_QUOTE_WRAPPED_RE = re.compile(r'^\s*["\'].+["\']\s*[.!?]?\s*$')

# Whole-sentence frame-noun fragments. Sentence equals one of these
# (case-insensitive, with or without trailing period). Drop.
_PORTB_FRAME_FRAGMENT_SENTENCES = frozenset({
    "references", "see also", "this paper", "this article",
    "this study", "this work", "this report", "this section",
    "external links", "further reading", "notes", "footnotes",
    "bibliography", "introduction", "conclusion", "abstract",
    "summary", "background", "methods", "method", "results",
    "discussion", "acknowledgements", "acknowledgments",
})

_PORTB_MIN_SENTENCE_WORDS = 4


def _portb_is_junk(sentence):
    """Path 3 -- True if sentence emits no bond."""
    if not sentence:
        return True
    s = sentence.strip()
    if not s:
        return True
    # Length floor
    if len(s.split()) < _PORTB_MIN_SENTENCE_WORDS:
        return True
    # Whole-sentence frame-noun fragment
    if s.lower().rstrip(".!? \t") in _PORTB_FRAME_FRAGMENT_SENTENCES:
        return True
    # Interrogative
    if _PORTB_QUESTION_END_RE.search(s):
        return True
    if _PORTB_INTERROGATIVE_LEAD_RE.match(s):
        return True
    # Code shape
    if _PORTB_CODE_TOKEN_RE.search(s):
        return True
    # Citation marker
    if _PORTB_CITATION_MARKER_RE.search(s):
        return True
    # Editorial frame
    if _PORTB_EDITORIAL_FRAME_RE.search(s):
        return True
    # Quote-wrapped
    if _PORTB_QUOTE_WRAPPED_RE.match(s):
        return True
    return False


def _portb_starts_with_pronoun(sentence):
    return bool(_PORTB_PRONOUN_LEAD_RE.match(sentence or ""))


# Path 2 object-phrase extractor. After a leading pronoun, capture
# (pred_verb, tail_object). Tail trimmed at the first subordinate-
# clause connector so the object stays compact.
_PORTB_PATH2_RE = re.compile(
    r"^\s*(?:it|this|that|these|those|he|she|they|we|you|i)\s+"
    + r"(?P<pred>" + _PRED_RE + r")\s+(?P<tail>[^.!?\n]{3,200})",
    re.IGNORECASE,
)
_PORTB_TAIL_CLAUSE_SPLIT_RE = re.compile(
    r"\s+(?:that|which|because|since|while|although|though)\s+",
    re.IGNORECASE,
)


def _portb_path2_object(sentence):
    """Extract (pred_verb, tail_object) from a pronoun-led sentence
    or None if unparseable."""
    if not sentence:
        return None
    m = _PORTB_PATH2_RE.match(sentence)
    if not m:
        return None
    pred = (m.group("pred") or "").strip().lower()
    tail = (m.group("tail") or "").strip()
    tail = _PORTB_TAIL_CLAUSE_SPLIT_RE.split(tail, maxsplit=1)[0]
    tail = tail.strip(",;:")
    if len(tail) < 3 or len(tail) > 200:
        return None
    return (pred, tail)


# Local kind lookup. Mirrors smart_extractor._RELATION_TO_KIND exactly.
# Kept local so this curriculum-tier module does not import a
# private-named symbol (`_kind_for`) from a peer module. If you add a
# new relation, update BOTH this table AND
# smart_extractor._RELATION_TO_KIND AND src/node.rs::
# infer_kind_from_relation in lockstep.
_PORTB_RELATION_TO_KIND = {
    "IsA": "definition",
    "PartOf": "composition",
    "ComposedOf": "composition",
    "UsedFor": "purpose",
    "HasAspect": "fact",
    "HasProperty": "fact",
    "KnownFor": "fact",
    "LocatedIn": "fact",
    "FoundedBy": "fact",
    "FoundedIn": "fact",
    "DiscoveredBy": "fact",
    "NamedAfter": "fact",
    "WrittenBy": "fact",
    "BornIn": "fact",
    "DiedIn": "fact",
    "BornOn": "fact",
    "DiedOn": "fact",
    "MeasuredIn": "fact",
    "HasPopulation": "fact",
    "HasAltitude": "fact",
    "HasArea": "fact",
    "HasOfficialLanguage": "fact",
    "HasCapital": "fact",
    "RelatedTo": "concept",
}

def _portb_kind_for(relation):
    return _PORTB_RELATION_TO_KIND.get(relation, "concept")


def _portb_tail_passes_asymmetry(tail):
    """Final asymmetry-boundary check on the OBJECT phrase. Mirrors the
    junk-filter set but applied to the tail only. Belt+suspenders so a
    citation/editorial/code-shape token hiding mid-sentence doesn't
    survive to the bond's object field."""
    if not tail:
        return False
    if _PORTB_EDITORIAL_FRAME_RE.search(tail):
        return False
    if _PORTB_CITATION_MARKER_RE.search(tail):
        return False
    if _PORTB_CODE_TOKEN_RE.search(tail):
        return False
    if _PORTB_QUESTION_END_RE.search(tail):
        return False
    if _PORTB_INTERROGATIVE_LEAD_RE.match(tail):
        return False
    return True




# Path 2 metalinguistic-object filter. Pronoun-led sentences whose
# object phrase is one of these self-referential / discourse-meta
# terms produce semantically nonsensical anchor-bonds (e.g.
# "This is a question." => ("Article", IsA, "a question")). Filter
# them at the path-2 emit site. Loose match: lowercase + strip
# leading articles ("a ", "the ", "an ") then equality.
_PORTB_PATH2_METALINGUISTIC = frozenset({
    "question", "answer", "case", "issue", "problem",
    "point", "topic", "subject", "fact", "statement",
    "example", "thing", "matter", "claim", "argument",
    "assumption", "premise", "conclusion", "remark",
    "comment", "observation", "note", "hypothesis",
    "true", "false", "correct", "incorrect", "wrong",
    "right", "obvious", "clear", "important", "necessary",
    "possible", "impossible", "likely", "unlikely",
    "common", "rare", "good", "bad", "easy", "hard",
    "simple", "complex", "complicated", "useful", "useless",
})

def _portb_path2_object_metalinguistic(tail):
    """True if the tail object is a metalinguistic / discourse-meta
    term that should not anchor-bond to the article title."""
    if not tail:
        return True
    t = tail.lower().strip(" .,;:!?\t")
    # Strip leading article so "a question" -> "question".
    for art in ("a ", "an ", "the "):
        if t.startswith(art):
            t = t[len(art):]
            break
    if t in _PORTB_PATH2_METALINGUISTIC:
        return True
    return False



# Path 2 anchor title-shape guard. Article titles that contain `?` or `!`,
# or that start with a WH-question word, are interrogative in shape and
# CANNOT be safely anchor-bonded. The pronoun-led sentence would inherit
# the question shape as its bond subject -- gate (a) asymmetry leak.
# Surfaced by arxiv 2307.06166 audit (2026-05-10): title "Can Vision-
# Language Models be a Good Guesser? Exploring VLMs for Times and
# Location Reasoning". Path 2 anchor with that title produces
# (subject="Can Vision-Language Models be a Good Guesser? ...", IsA, ...)
# which is an asymmetry violation.
#
# Fix: validate title shape BEFORE Path 2 dispatch. Disable Path 2 for
# the entire document if title fails the shape check. Path 1 is
# unaffected (no anchor dependency).
_PORTB_TITLE_INTERROGATIVE_LEAD_RE = re.compile(
    r"^\s*(?:how|what|why|when|where|who|whose|whom|which|"
    r"does|do|did|is|are|was|were|can|could|should|would|will|"
    r"may|might|must|shall|am)\b",
    re.IGNORECASE,
)

def _portb_title_is_interrogative(title):
    """True if the title cannot safely anchor a Path 2 bond."""
    if not title:
        return False  # empty title is handled separately (Path 2 already disabled)
    t = title.strip()
    if "?" in t:
        return True
    # Trailing "!" is rhetorical-frame; reject too.
    if t.endswith("!"):
        return True
    if _PORTB_TITLE_INTERROGATIVE_LEAD_RE.match(t):
        return True
    return False


def extract_response_bonds_multi_sentence(title, text, source_url):
    """Port B (R1) -- multi-sentence Pattern A/B + anchor-bond extractor.

    Returns list of bond dicts shaped like smart_extractor's output
    plus a `subject_origin` field ("subject" for Path 1, "anchor" for
    Path 2). Iterates every sentence in `text` via pysbd.

    Asymmetry boundary is enforced TWICE per sentence: junk-filter on
    the whole sentence (Path 3) before path dispatch, then a final
    tail-only recheck before emit. This is intentional defence in
    depth; an editorial frame hidden mid-sentence still gets caught.

    Per-bond weight 0.4 / confidence "Inferred" -- matches the
    original extract_response_bond. `layer` is "L-multi-sentence"
    (Path 1) or "L-multi-anchor" (Path 2). Internal dedup is by
    (subject.lower(), relation, object.lower()) tuple so the same
    sentence cannot emit the same bond twice from this function.
    Loader-level dedup against smart_extractor output is the
    caller's responsibility (see arxiv/wikipedia/grokopedia/
    stackexchange complementary-pass wirings).
    """
    if not text:
        return []
    title = (title or "").strip()
    text_s = text.strip()
    if len(text_s) < 20:
        return []

    try:
        seg = _get_segmenter()
        sentences = seg.segment(text_s)
    except Exception:
        # pysbd unavailable / threw -- fall back to homegrown regex so
        # the codepath remains functional even on a stubbed env.
        sentences = re.split(r"(?<=[.!?])\s+|\n+", text_s)

    bonds = []
    seen = set()

    for raw in sentences:
        s = (raw or "").strip()
        if not s:
            continue
        if _portb_is_junk(s):
            continue

        # ---- Path 2: pronoun-led -> anchor-bond on title ----
        if _portb_starts_with_pronoun(s):
            if not title or _portb_title_is_interrogative(title):
                # No safe anchor source (missing or interrogative-shape).
                # Better no bond than wrong bond.
                continue
            parsed = _portb_path2_object(s)
            if not parsed:
                continue
            pred, tail = parsed
            if not _portb_tail_passes_asymmetry(tail):
                continue
            if _portb_path2_object_metalinguistic(tail):
                continue
            relation = _relation_for_pred(pred, tail)
            # Phase-1 (2026-05-30): NP truncation replaces raw [:200].
            obj = _truncate_to_np(tail)
            if not obj:
                continue
            key = (title.lower(), relation, obj.lower())
            if key in seen:
                continue
            seen.add(key)
            bonds.append({
                "subject": title,
                "relation": relation,
                "object": obj,
                "kind": _portb_kind_for(relation),
                "weight": 0.4,
                "confidence": "Inferred",
                "layer": "L-multi-anchor",
                "source_url": source_url,
                "subject_origin": "anchor",
            })
            continue

        # ---- Path 1: clean Pattern A/B subject ----
        parsed = _extract_response_subject(s)
        if not parsed:
            continue
        subj, pred = parsed
        # Re-locate the tail past the matched subject + predicate so
        # _relation_for_pred can see phrase-anchored cues.
        tail_m = re.search(
            re.escape(subj) + r"\s+" + re.escape(pred) + r"\s+([^.!?\n]{3,200})",
            s,
            re.IGNORECASE,
        )
        if not tail_m:
            continue
        tail = tail_m.group(1).strip().strip(",;:")
        if not _portb_tail_passes_asymmetry(tail):
            continue
        relation = _relation_for_pred(pred, tail)
        # Phase-1 (2026-05-30): NP truncation replaces raw [:200].
        obj = _truncate_to_np(tail)
        if not obj:
            continue
        key = (subj.lower(), relation, obj.lower())
        if key in seen:
            continue
        seen.add(key)
        bonds.append({
            "subject": subj,
            "relation": relation,
            "object": obj,
            "kind": _portb_kind_for(relation),
            "weight": 0.4,
            "confidence": "Inferred",
            "layer": "L-multi-sentence",
            "source_url": source_url,
            "subject_origin": "subject",
        })

    return bonds




# ---- ingest funcs ----

def _maybe_save(client, last_save_t, bonds_since_save, cp, corpus, last_id):
    """Periodic save. Returns (last_save_t, bonds_since_save, saved_ok).

    OP-SAVE-ASYNC — calls lu4d_save with mode="async" and then polls
    lu4d_save_status every ~1.5s until state is "completed" or
    "failed". saved_ok is True only when state == "completed".

    Callers MUST advance their local last_saved_id only when saved_ok
    is True, preserving the OP-012 cursor-discipline invariant.
    """
    now = time.monotonic()
    if (now - last_save_t) >= SAVE_INTERVAL_SEC or bonds_since_save >= SAVE_INTERVAL_BONDS:
        log_progress(f"{corpus}: lu4d_save async (interval) at last_id={last_id}")
        ok = False
        try:
            r = client.call("lu4d_save", {"mode": "async"})
            started = call_ok(r)
        except Exception as e:
            started = False
            log_progress(f"{corpus}: lu4d_save (start) FAILED: {e}")
        if started:
            # Poll until terminal state.
            # OP-SAVE-FINALIZE -- even if this poll cycle gives up, the next
            # start_async_save will finalize the worker-completed save, so no
            # data is lost. The deadline mainly protects the harness's own
            # forward progress signal (checkpoint advancement). Bumped to 30 min
            # to comfortably cover ~10 GB saves at observed ~22 MB/s CRC throughput.
            poll_deadline = time.monotonic() + 1800.0  # 30 min hard cap
            while True:
                if time.monotonic() > poll_deadline:
                    log_progress(f"{corpus}: lu4d_save_status poll deadline exceeded")
                    break
                try:
                    sr = client.call("lu4d_save_status", {})
                    sok = call_ok(sr)
                except Exception as e:
                    log_progress(f"{corpus}: lu4d_save_status FAILED: {e}")
                    break
                if not sok:
                    break
                state = None
                try:
                    sc = sr.get("result", {}).get("structuredContent", {})
                    state = sc.get("state")
                except Exception:
                    state = None
                if state == "completed":
                    ok = True
                    break
                if state == "failed":
                    err = sc.get("error") if isinstance(sc, dict) else None
                    log_progress(f"{corpus}: lu4d_save FAILED state: {err}")
                    break
                # state is "running" or "idle" (idle would be unexpected — treat as not-yet-running);
                # keep polling.
                time.sleep(1.5)
        if ok:
            cp[corpus] = cp.get(corpus, {})
            cp[corpus]["last_saved_id"] = last_id
            save_checkpoint(cp)
            log_progress(f"{corpus}: checkpoint updated last_saved_id={last_id}")
        return time.monotonic(), 0, ok
    return last_save_t, bonds_since_save, False


def ingest_visual_genome(client, cp, smoke_limit=None):
    import importlib.util, pathlib
    loader_path = pathlib.Path(__file__).resolve().parent / "visual_genome_loader.py"
    spec = importlib.util.spec_from_file_location("vg_loader", loader_path)
    vg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vg)

    # Aligns to TrustTable::seed_curriculum_corpora key (HF repo path).
    # Without this stamp, VG bonds resolve to default_multiplier=0.4
    # instead of the seeded 1.0 trust.
    corpus_id = "ranjaykrishna/visual_genome"
    target_total = smoke_limit if smoke_limit else 2_316_104  # canonical VG v1.4
    log_progress(f"visual_genome: starting; target={target_total} smoke={bool(smoke_limit)}")

    throttle = Throttle(THROTTLE_VG)
    sink = BatchSink(client, throttle)
    sink.bonds_landed   = cp.get(corpus_id, {}).get("bonds_landed", 0)
    sink.bonds_errors   = cp.get(corpus_id, {}).get("errors_count", 0)
    sink.bonds_skipped_dedup = cp.get(corpus_id, {}).get("bonds_skipped_dedup", 0)
    last_saved_id = cp.get(corpus_id, {}).get("last_saved_id", -1)
    images_seen = set()
    skipped_reason = None
    last_save_t = time.monotonic()
    last_landed = sink.bonds_landed
    pos = 0
    bonds_landed = sink.bonds_landed
    bonds_errors = sink.bonds_errors
    try:
        for s_name, raw_pred, o_name, image_id in vg.stream_triplets(max_triplets=target_total):
            if pos <= last_saved_id:
                pos += 1
                continue
            images_seen.add(image_id)
            relation = vg.map_predicate(raw_pred)
            url = f"vg-image-{image_id}"
            try:
                # CORPUS ATTRIBUTION call-site (Visual Genome, Tier 1).
                # corpus_id == 'ranjaykrishna/visual_genome' from the top of
                # ingest_visual_genome. See module-level CORPUS
                # ATTRIBUTION block above BatchSink.
                sink.add({
                    "corpus": corpus_id, "source_url": url, "run_id": RUN_ID_TIER1,
                    "subject": s_name, "relation": relation, "object": o_name,
                    "weight": 0.5, "confidence": "Unverified",
                })
            except Exception:
                sink.bonds_errors += 1
            pos += 1
            if pos % 500 == 0:
                # Flush whatever is buffered so the progress log
                # reflects bonds actually landed up to now.
                sink.flush()
                bonds_since_save = sink.bonds_landed - last_landed
                log_progress(f"visual_genome: pos={pos}/{target_total} landed={sink.bonds_landed} "
                             f"errors={sink.bonds_errors} images={len(images_seen)} "
                             f"deduped={sink.bonds_skipped_dedup} gated={sink.bonds_gated} rejected={sink.bonds_rejected}")
                cp[corpus_id] = {"bonds_landed":sink.bonds_landed,"errors_count":sink.bonds_errors,
                                 "bonds_skipped_dedup":sink.bonds_skipped_dedup,
                                 "last_saved_id":last_saved_id}
                save_checkpoint(cp)
                new_t, new_b, saved_ok = _maybe_save(client, last_save_t, bonds_since_save, cp, corpus_id, pos-1)
                if saved_ok:
                    last_saved_id = pos - 1
                    last_landed = sink.bonds_landed
                last_save_t = new_t
        # Flush any tail entries.
        sink.flush()
        bonds_landed = sink.bonds_landed
        bonds_errors = sink.bonds_errors
    except FileNotFoundError as e:
        skipped_reason = f"cache missing: {e}"
        log_progress(f"visual_genome: SKIPPED — {skipped_reason}")
    except Exception as e:
        skipped_reason = f"VG stream failed: {type(e).__name__}: {str(e)[:200]}"
        log_progress(f"visual_genome: SKIPPED — {skipped_reason}")

    # final save for corpus (only if we actually landed bonds)
    if bonds_landed > 0:
        try:
            r = client.call("lu4d_save", {"mode": "async"})
            if call_ok(r):
                cp[corpus_id] = {"bonds_landed":bonds_landed,"errors_count":bonds_errors,
                                 "last_saved_id":pos-1, "completed": True,
                                 "skipped_reason": skipped_reason}
                save_checkpoint(cp)
        except Exception:
            pass
    else:
        # No new bonds landed (SKIPPED or empty). Just mark completed in checkpoint.
        cp[corpus_id] = {"bonds_landed":bonds_landed,"errors_count":bonds_errors,
                         "last_saved_id":pos-1, "completed": True,
                         "skipped_reason": skipped_reason}
        save_checkpoint(cp)
    log_progress(f"visual_genome: DONE landed={bonds_landed} errors={bonds_errors} "
                 f"deduped={sink.bonds_skipped_dedup} gated={sink.bonds_gated} rejected={sink.bonds_rejected} "
                 f"images={len(images_seen)} skip={skipped_reason}")
    return {"bonds_landed":bonds_landed,"bonds_errors":bonds_errors,
            "bonds_skipped_dedup":sink.bonds_skipped_dedup,
            "bonds_gated":sink.bonds_gated,
            "images_processed":len(images_seen),"skipped_reason":skipped_reason}


def ingest_wikipedia(client, cp, smoke_limit=None):
    from datasets import load_dataset
    corpus_id = "wikimedia/wikipedia"
    target_total = smoke_limit if smoke_limit else 6_700_000
    log_progress(f"wikipedia: starting load_dataset(streaming) target={target_total} smoke={bool(smoke_limit)}")
    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)

    throttle = Throttle(THROTTLE_WIKIPEDIA)
    sink = BatchSink(client, throttle)
    sink.bonds_landed   = cp.get(corpus_id, {}).get("bonds_landed", 0)
    sink.bonds_errors   = cp.get(corpus_id, {}).get("errors_count", 0)
    sink.bonds_skipped_dedup = cp.get(corpus_id, {}).get("bonds_skipped_dedup", 0)
    last_saved_id = cp.get(corpus_id, {}).get("last_saved_id", -1)
    last_save_t = time.monotonic()
    last_landed = sink.bonds_landed
    articles = 0
    start_t = time.monotonic()
    for ex in ds:
        if articles >= target_total: break
        if articles <= last_saved_id:
            articles += 1
            continue
        title = ex.get("title",""); text = ex.get("text",""); url = ex.get("url","")
        # Phase B: smart extractor v2 (iter-4). Returns list of dicts shaped
        # for sink.add; extra fields like "layer"/"frequency" are ignored
        # by sink.add. K-gate (T.3) inside lu4d_teach_corpus_batch is the
        # quality filter.
        bonds = extract_bonds_from_wikipedia_article(ex)
        # Port B (R1) complementary pass: pronoun-anchored bonds that
        # smart_extractor's title-alias filter would drop, and any
        # multi-sentence Pattern A/B bonds outside the L3/L5
        # title-anchored sentence set. Union + dedup against
        # smart_extractor output on (subject.lower(), relation,
        # object.lower()) so smart_extractor wins on shared triples.
        extra_bonds = extract_response_bonds_multi_sentence(title, text, url)
        if extra_bonds:
            seen_triples = set(
                (b["subject"].lower(), b["relation"], b["object"].lower())
                for b in bonds
            )
            for eb in extra_bonds:
                key = (eb["subject"].lower(), eb["relation"], eb["object"].lower())
                if key in seen_triples:
                    continue
                seen_triples.add(key)
                bonds.append(eb)
        # OP-EXTRACT-WIDEN: collapse same-(subject,object) pairs to the richest
        # relation so a generic RelatedTo/IsA never shadows a typed relation at
        # the brain's pair-only dedup internal notes.
        bonds = collapse_to_richest_relation(bonds)
        for b in bonds:
            try:
                # CORPUS ATTRIBUTION call-site (Wikipedia, Tier 1).
                # corpus_id == 'wikimedia/wikipedia' from the top of
                # ingest_wikipedia. See module-level CORPUS
                # ATTRIBUTION block above BatchSink.
                sink.add({
                    "corpus": corpus_id,
                    "source_url": b.get("source_url") or url,
                    "run_id": RUN_ID_TIER1,
                    "subject": b["subject"],
                    "relation": b["relation"],
                    "object": b["object"],
                    "weight": b["weight"],
                    "confidence": b["confidence"],
                })
            except Exception:
                sink.bonds_errors += 1
        articles += 1
        if articles % 1000 == 0:
            sink.flush()
            bonds_since_save = sink.bonds_landed - last_landed
            elapsed = time.monotonic() - start_t
            rate = articles / elapsed if elapsed > 0 else 0
            eta = (target_total - articles) / rate if rate > 0 else float("inf")
            log_progress(f"wikipedia: articles={articles}/{target_total} "
                         f"bonds_landed={sink.bonds_landed} errors={sink.bonds_errors} "
                         f"deduped={sink.bonds_skipped_dedup} gated={sink.bonds_gated} rejected={sink.bonds_rejected} "
                         f"rate={rate:.1f}art/s eta_remaining_s={eta:.0f}")
            cp[corpus_id] = {"bonds_landed":sink.bonds_landed,"errors_count":sink.bonds_errors,
                             "bonds_skipped_dedup":sink.bonds_skipped_dedup,
                             "last_saved_id":last_saved_id}
            save_checkpoint(cp)
            new_t, new_b, saved_ok = _maybe_save(client, last_save_t, bonds_since_save, cp, corpus_id, articles-1)
            if saved_ok:
                last_saved_id = articles - 1
                last_landed = sink.bonds_landed
            last_save_t = new_t
    sink.flush()
    bonds_landed = sink.bonds_landed
    bonds_errors = sink.bonds_errors
    # final save
    try:
        r = client.call("lu4d_save", {})
        if call_ok(r):
            cp[corpus_id] = {"bonds_landed":bonds_landed,"errors_count":bonds_errors,
                             "last_saved_id":articles-1, "completed": True}
            save_checkpoint(cp)
    except Exception:
        pass
    log_progress(f"wikipedia: DONE articles={articles} landed={bonds_landed} errors={bonds_errors}")
    return {"articles_processed":articles,"bonds_landed":bonds_landed,"bonds_errors":bonds_errors}


def _safe_glob(pattern):
    import glob
    return sorted(glob.glob(pattern))


def ingest_arxiv(client, cp, smoke_limit=None,
                 set_spec="cs:cs:AI", frm="2024-01-01", until="2024-06-30",
                 max_papers=None):
    """Tier-1 arxiv abstracts via OAI-PMH internal notes.

    Shape Y wiring: importlib-loads scripts/arxiv_loader.py, drives its
    pure stream_papers + make_bonds helpers through the shared
    BatchSink/McpClient. Sidecar (~/.lu/arxiv_progress.jsonl) is reused
    for cross-run resume; cp[corpus_id] tracks within-run progress.
    """
    import importlib.util, pathlib
    loader_path = pathlib.Path(__file__).resolve().parent / "arxiv_loader.py"
    spec = importlib.util.spec_from_file_location("arxiv_loader_mod", loader_path)
    arx = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(arx)

    corpus_id = arx.CORPUS_ID  # "arxiv"
    sidecar = arx.DEFAULT_SIDECAR
    arx._ensure_sidecar(sidecar)
    seen_ids = arx.load_seen_ids(sidecar)
    target = smoke_limit if smoke_limit else (max_papers if max_papers else 5000)

    log_progress(
        f"arxiv: starting; set={set_spec} from={frm} until={until} "
        f"target={target} sidecar_seen={len(seen_ids)} "
        f"smoke={bool(smoke_limit)}"
    )

    sink = BatchSink(client, Throttle(THROTTLE_TIER2))
    sink.bonds_landed = cp.get(corpus_id, {}).get("bonds_landed", 0)
    sink.bonds_errors = cp.get(corpus_id, {}).get("errors_count", 0)
    sink.bonds_skipped_dedup = cp.get(corpus_id, {}).get("bonds_skipped_dedup", 0)
    last_landed = sink.bonds_landed
    last_save_t = time.monotonic()

    papers_seen = 0
    papers_already_seen = 0
    papers_no_bond = 0
    papers_leak_filtered_all = 0
    last_aid = None
    skipped_reason = None

    try:
        for paper in arx.stream_papers(set_spec, frm, until, target):
            papers_seen += 1
            aid = paper.get("arxiv_id") or ""
            last_aid = aid
            if aid in seen_ids:
                papers_already_seen += 1
                continue
            bonds = arx.make_bonds(paper)
            if not bonds:
                arx.append_seen(sidecar, aid, "no_bond", 0)
                papers_no_bond += 1
                continue
            paper_queued = 0
            paper_leak_filtered = 0
            for b in bonds:
                subj = b.get("subject", "") or ""
                if arx.subject_leaks(subj):
                    paper_leak_filtered += 1
                    continue
                # CORPUS ATTRIBUTION call-site (arxiv, Tier 1).
                # corpus_id == "arxiv" matches TrustTable seed.
                ok = sink.add({
                    "corpus": corpus_id,
                    "source_url": b.get("source_url") or f"https://arxiv.org/abs/{aid}",
                    "run_id": RUN_ID_TIER1,
                    "subject": subj,
                    "relation": b["relation"],
                    "object": b["object"],
                    "weight": b.get("weight", 0.5),
                    "confidence": b.get("confidence", "Unverified"),
                })
                if ok:
                    paper_queued += 1
            papers_leak_filtered_all += paper_leak_filtered
            if paper_queued > 0:
                arx.append_seen(sidecar, aid, f"queued:{paper_queued}", paper_queued)
            else:
                arx.append_seen(sidecar, aid,
                                f"all_filtered:{paper_leak_filtered}", 0)
            if papers_seen % 50 == 0:
                sink.flush()
                bonds_since_save = sink.bonds_landed - last_landed
                log_progress(
                    f"arxiv: papers={papers_seen}/{target} "
                    f"landed={sink.bonds_landed} no_bond={papers_no_bond} "
                    f"leak={papers_leak_filtered_all} already={papers_already_seen}"
                )
                cp[corpus_id] = {
                    "bonds_landed": sink.bonds_landed,
                    "errors_count": sink.bonds_errors,
                    "bonds_skipped_dedup": sink.bonds_skipped_dedup,
                    "last_arxiv_id": aid,
                }
                save_checkpoint(cp)
                new_t, _b_disc, saved_ok = _maybe_save(client, last_save_t, bonds_since_save,
                                       cp, corpus_id, aid)
                if saved_ok:
                    last_landed = sink.bonds_landed
                last_save_t = new_t
        sink.flush()
    except Exception as e:
        skipped_reason = f"arxiv stream failed: {type(e).__name__}: {str(e)[:200]}"
        log_progress(f"arxiv: SKIPPED \u2014 {skipped_reason}")

    try:
        r = client.call("lu4d_save", {})
        if call_ok(r):
            cp[corpus_id] = {
                "bonds_landed": sink.bonds_landed,
                "errors_count": sink.bonds_errors,
                "bonds_skipped_dedup": sink.bonds_skipped_dedup,
                "last_arxiv_id": last_aid,
                "completed": True,
                "skipped_reason": skipped_reason,
            }
            save_checkpoint(cp)
    except Exception:
        pass
    log_progress(
        f"arxiv: DONE landed={sink.bonds_landed} errors={sink.bonds_errors} "
        f"papers={papers_seen} no_bond={papers_no_bond} "
        f"leak={papers_leak_filtered_all} already={papers_already_seen} "
        f"skip={skipped_reason}"
    )
    return {
        "bonds_landed": sink.bonds_landed,
        "bonds_errors": sink.bonds_errors,
        "bonds_skipped_dedup": sink.bonds_skipped_dedup,
        "papers_seen": papers_seen,
        "papers_no_bond": papers_no_bond,
        "papers_leak_filtered": papers_leak_filtered_all,
        "papers_already_seen": papers_already_seen,
        "skipped_reason": skipped_reason,
    }


def ingest_stackexchange(client, cp, smoke_limit=None,
                         xml_paths=None, min_score=5):
    """Tier-1 StackExchange answers via offline XML dumps internal notes.

    Shape Y wiring. xml_paths defaults to glob('/tmp/se/*/Posts.xml').
    If none present, logs a SKIPPED status (not a hard error: the corpus
    is opportunistic, not required).
    """
    import importlib.util, pathlib
    loader_path = pathlib.Path(__file__).resolve().parent / "stackexchange_loader.py"
    spec = importlib.util.spec_from_file_location("se_loader_mod", loader_path)
    se = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(se)

    corpus_id = "stackexchange"  # matches TrustTable seed key
    if not xml_paths:
        xml_paths = _safe_glob("/tmp/se/*/Posts.xml")
    if not xml_paths:
        log_progress(
            "stackexchange: SKIPPED \u2014 no Posts.xml under /tmp/se/. "
            "Drop a StackExchange data dump and re-run."
        )
        return {
            "bonds_landed": 0,
            "bonds_errors": 0,
            "answers_seen": 0,
            "skipped_reason": "no_xml_dumps_present",
        }

    log_progress(
        f"stackexchange: starting; xml_paths={xml_paths} "
        f"min_score={min_score} smoke={bool(smoke_limit)}"
    )

    sink = BatchSink(client, Throttle(THROTTLE_TIER2))
    sink.bonds_landed = cp.get(corpus_id, {}).get("bonds_landed", 0)
    sink.bonds_errors = cp.get(corpus_id, {}).get("errors_count", 0)
    sink.bonds_skipped_dedup = cp.get(corpus_id, {}).get("bonds_skipped_dedup", 0)
    last_landed = sink.bonds_landed
    last_save_t = time.monotonic()

    answers_seen = 0
    answers_no_anchor = 0
    answers_q_only = 0
    skipped_reason = None
    last_post_id = None

    seen_post_ids_this_run = set()
    target = smoke_limit

    for xml_path in xml_paths:
        site = pathlib.Path(xml_path).parent.name or "stackexchange.com"
        try:
            for ans in se.stream_answers(
                xml_path, min_score=min_score, max_answers=target
            ):
                answers_seen += 1
                last_post_id = ans.get("id")
                if last_post_id in seen_post_ids_this_run:
                    continue
                seen_post_ids_this_run.add(last_post_id)
                try:
                    bonds, diag = se.extract_bonds_for_answer(ans, site)
                except Exception as e:
                    sink.bonds_errors += 1
                    diag = {"skipped_reason": f"extractor:{type(e).__name__}"}
                    bonds = []
                reason = diag.get("skipped_reason")
                if reason == "no_anchor":
                    answers_no_anchor += 1
                elif reason == "all_sentences_question_framed":
                    answers_q_only += 1
                for b in bonds:
                    # CORPUS ATTRIBUTION call-site (stackexchange, Tier 1).
                    sink.add({
                        "corpus": corpus_id,
                        "source_url": f"https://{site}/a/{ans['id']}",
                        "run_id": RUN_ID_TIER1,
                        "subject": b["subject"],
                        "relation": b["relation"],
                        "object": b["object"],
                        "weight": b.get("weight", 0.5),
                        "confidence": b.get("confidence", "Unverified"),
                    })
                if answers_seen % 100 == 0:
                    sink.flush()
                    bonds_since_save = sink.bonds_landed - last_landed
                    log_progress(
                        f"stackexchange: site={site} answers={answers_seen} "
                        f"landed={sink.bonds_landed} no_anchor={answers_no_anchor} "
                        f"q_only={answers_q_only}"
                    )
                    cp[corpus_id] = {
                        "bonds_landed": sink.bonds_landed,
                        "errors_count": sink.bonds_errors,
                        "bonds_skipped_dedup": sink.bonds_skipped_dedup,
                        "last_xml": xml_path,
                        "last_post_id": last_post_id,
                    }
                    save_checkpoint(cp)
                    new_t, _b_disc, saved_ok = _maybe_save(
                        client, last_save_t, bonds_since_save,
                        cp, corpus_id, last_post_id,
                    )
                    if saved_ok:
                        last_landed = sink.bonds_landed
                    last_save_t = new_t
                if smoke_limit and answers_seen >= smoke_limit:
                    break
        except Exception as e:
            skipped_reason = (
                f"se stream failed for {xml_path}: "
                f"{type(e).__name__}: {str(e)[:200]}"
            )
            log_progress(f"stackexchange: WARN \u2014 {skipped_reason}")
        if smoke_limit and answers_seen >= smoke_limit:
            break

    sink.flush()
    try:
        r = client.call("lu4d_save", {})
        if call_ok(r):
            cp[corpus_id] = {
                "bonds_landed": sink.bonds_landed,
                "errors_count": sink.bonds_errors,
                "bonds_skipped_dedup": sink.bonds_skipped_dedup,
                "last_post_id": last_post_id,
                "completed": True,
                "skipped_reason": skipped_reason,
            }
            save_checkpoint(cp)
    except Exception:
        pass
    log_progress(
        f"stackexchange: DONE landed={sink.bonds_landed} "
        f"errors={sink.bonds_errors} answers={answers_seen} "
        f"no_anchor={answers_no_anchor} q_only={answers_q_only} "
        f"skip={skipped_reason}"
    )
    return {
        "bonds_landed": sink.bonds_landed,
        "bonds_errors": sink.bonds_errors,
        "bonds_skipped_dedup": sink.bonds_skipped_dedup,
        "answers_seen": answers_seen,
        "answers_no_anchor": answers_no_anchor,
        "answers_q_only": answers_q_only,
        "skipped_reason": skipped_reason,
    }


# ===================================================================
# Phase-1 coding curriculum loaders (declarative coding knowledge).
#
# Two DECLARATIVE sources, ingested on the SAME Tier-1 declarative path
# wikipedia/stackexchange use (smart_extractor.extract_bonds_from_
# wikipedia_article + collapse_to_richest_relation + the R1 multi-
# sentence complementary pass). NEITHER uses the Tier-2 prompt/response
# path: both sources are pure declarative statements with no
# prompt/response pairing.
#
#   Loader 1: CodeSearchNet docstrings (code-search-net/code_search_net)
#             - func_documentation_string is the NL payload.
#             - func_name / repository_name / language are subject anchors.
#             - Raw code bodies (func_code_string) are NEVER ingested.
#             - corpus id 'code-search-net/code_search_net', trust 0.9.
#
#   Loader 2: StackOverflow high-score answers (mikex86/stackoverflow-posts)
#             - PostTypeId==2 (answers), Score>=5, skip empty bodies.
#             - Body is MARKDOWN (```fenced``` code + residual HTML), so we
#               strip fenced/indented code blocks FIRST, then reuse
#               stackexchange_loader.html_to_text + extract_bonds_for_answer
#               (drops <code>/<pre>/<blockquote>, q-frame filter, anchor,
#               code-shape guard) -- the identical declarative path.
#             - corpus id 'stackoverflow', trust 0.8.
# ===================================================================

# Code-shape tokens: a bond subject containing any of these is code, not
# declarative NL. Mirrors stackexchange_loader.CODE_SHAPE_TOKENS.
_CODE_SHAPE_TOKENS = (
    "def ", "function ", "function(", "import ", "<div", "<span",
    "<script", "console.log", "print(", "println!", "fn ", "return ",
    "var ", "let ", "const ", "public ", "private ", "static ",
    "#include", "@override", "=>", "();", ");", "&&", "||", "::",
    "->", "===", "!==",
)

THROTTLE_CODING = 30.0

# CodeSearchNet docstring cleanup. Docstrings carry reST/Google/numpy
# directive residue (':param x:', ':return:', '>>> ', '.. note::', etc.)
# and heavy indentation. We strip directive lines and code-example lines
# (doctest '>>>'/'...') because they are NOT declarative natural language
# -- leaving them in would pollute bond subjects with parameter names and
# code tokens, exactly the asymmetry leak the brief warns against.
_CSN_DIRECTIVE_RE = re.compile(
    r"^\s*(:param\b|:returns?\b|:rtype\b|:type\b|:raises?\b|:yields?\b"
    r"|:except\b|:keyword\b|:arg\b|:cvar\b|:ivar\b|:var\b"
    r"|@param\b|@return\b|@throws?\b|@see\b|@author\b|@deprecated\b"
    r"|args:|arguments:|returns:|return:|raises:|yields:|yield:"
    r"|parameters:|params:|kwargs:|attributes:|example[s]?:|note[s]?:"
    r"|see also:|todo:|warning:|usage:)",
    re.IGNORECASE,
)
_CSN_DOCTEST_RE = re.compile(r"^\s*(>>>|\.\.\.)")
_CSN_RST_DIRECTIVE_RE = re.compile(r"^\s*\.\.\s+\w+::")


def clean_csn_docstring(doc):
    """Reduce a CodeSearchNet docstring to declarative prose.

    Drops doctest lines (>>> / ...), reST/Google/numpy directive lines
    (:param:, Returns:, .. note::, etc.), collapses indentation/whitespace.
    Returns '' if nothing declarative remains.
    """
    if not doc:
        return ""
    kept = []
    for ln in doc.splitlines():
        if _CSN_DOCTEST_RE.match(ln):
            continue
        if _CSN_RST_DIRECTIVE_RE.match(ln):
            continue
        if _CSN_DIRECTIVE_RE.match(ln):
            continue
        kept.append(ln)
    text = " ".join(kept)
    # Inline backtick / markdown residue + whitespace collapse.
    text = re.sub(r"`([^`\n]+)`", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _csn_anchor(func_name, repo):
    """Build a clean subject anchor from func_name (strip Class. prefix and
    leading underscores), falling back to the repository short name."""
    name = (func_name or "").strip()
    if name:
        # 'ImageGraphCut.__msgc_step3' -> 'msgc_step3'
        name = name.split(".")[-1].lstrip("_")
        # snake_case / camelCase -> spaced words for a readable NL anchor.
        name = name.replace("_", " ").strip()
        name = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name).strip()
        if len(name) >= 3:
            return name
    repo = (repo or "").strip()
    if repo:
        return repo.split("/")[-1].replace("-", " ").replace("_", " ").strip()
    return None


# ---- REWORK 2026-06-16: summary-only clean-bond extraction ----
import re as _re_csn

# Strip a leading determiner so "a sorted copy" stays but "the" duplication
# in "is a the list" never happens. Used only when we synthesize objects.
_CSN_LEAD_DET_RE = _re_csn.compile(r"^(?:the|a|an)\s+", _re_csn.IGNORECASE)

# Imperative verb -> gerund nominalization for the action case.
# Map a leading imperative verb to its gerund so an imperative summary
# ("Compute the checksum") becomes a clean purpose phrase
# ("computing the checksum") that verbalizes after UsedFor (" is used for ").
_CSN_IRREGULAR_GERUND = {
    "set": "setting", "get": "getting", "put": "putting", "run": "running",
    "stop": "stopping", "begin": "beginning", "split": "splitting",
    "map": "mapping", "strip": "stripping", "drop": "dropping",
    "trim": "trimming", "wrap": "wrapping", "submit": "submitting",
    "commit": "committing", "emit": "emitting", "plot": "plotting",
    "fit": "fitting", "cancel": "cancelling", "label": "labelling",
}


def _csn_gerund(verb):
    """Nominalize an imperative verb to its -ing gerund. Lowercased.

    Handles common irregulars + the standard English -e drop / consonant
    doubling for short CVC stems. Robust + simple: anything unexpected
    just gets a trailing 'ing'.
    """
    v = verb.lower()
    if v in _CSN_IRREGULAR_GERUND:
        return _CSN_IRREGULAR_GERUND[v]
    if v.endswith("ie"):                       # die -> dying, tie -> tying
        return v[:-2] + "ying"
    if v.endswith("e") and not v.endswith("ee") and len(v) > 2:
        return v[:-1] + "ing"                   # compute -> computing
    return v + "ing"


# A summary object phrase is REJECTED if it still carries code / residue.
_CSN_OBJ_BAD_RE = _re_csn.compile(
    r"[<>{}\[\]|`=]|https?://|::|->|=>|\(\)|\bself\b|\bcls\b"
)
# Allowed object characters (after we accept it): letters, digits, spaces,
# and a small set of in-phrase punctuation. If a phrase is mostly symbols
# it won't survive this.
_CSN_OBJ_WORD_RE = _re_csn.compile(r"[A-Za-z]")

# Definitional summary openers -> (relation, strip_prefix_regex).
# Each maps a docstring summary that DEFINES the thing into a typed bond.
_CSN_DEF_PATTERNS = [
    # "Return(s) a sorted copy ..."  -> Produces "a sorted copy ..."
    (_re_csn.compile(r"^returns?\s+(.*)", _re_csn.IGNORECASE), "Produces"),
    # "Compute(s)/Create(s)/Generate(s)/Build(s) a X" (3rd-person, not
    # imperative) -> Produces.  Caught by the imperative branch when bare.
    # "A X that ..." / "An X ..."  -> IsA  (keep the determiner)
    (_re_csn.compile(r"^(an?\s+.*)", _re_csn.IGNORECASE), "IsA"),
    # "Represents a X" -> IsA "a X"
    (_re_csn.compile(r"^represents?\s+(.*)", _re_csn.IGNORECASE), "IsA"),
    # "Class/Helper/Wrapper/Decorator for X" -> UsedFor X
    (_re_csn.compile(r"^(?:class|helper|wrapper|decorator|utility|mixin)\s+"
                     r"(?:for|that|to)\s+(.*)", _re_csn.IGNORECASE), "UsedFor"),
]

# Imperative-action verbs we treat as "this function DOES X" -> UsedFor.
# (We nominalize the verb to a gerund: "Compute the checksum" ->
# UsedFor "computing the checksum".)  Restricted to a vetted set so we
# don't misfire on narrative docstrings.
_CSN_ACTION_VERBS = {
    "compute", "calculate", "create", "build", "generate", "make",
    "convert", "parse", "format", "encode", "decode", "serialize",
    "deserialize", "load", "save", "read", "write", "fetch", "send",
    "download", "upload", "render", "validate", "check", "verify",
    "extract", "filter", "sort", "merge", "split", "join", "map",
    "reduce", "find", "search", "remove", "delete", "add", "insert",
    "update", "set", "get", "register", "initialize", "configure",
    "connect", "close", "open", "start", "stop", "run", "execute",
    "apply", "normalize", "transform", "wrap", "strip", "trim",
    "compress", "decompress", "compile", "resolve", "mark", "count",
    "copy", "move", "rename", "clear", "reset", "flush", "commit",
    "submit", "emit", "plot", "draw", "print", "log", "cache",
    "retrieve", "store", "queue", "schedule", "dispatch", "handle",
    "process", "return",  # 'return' handled by def-pattern first
}

# Max object phrase length (words). A clean summary object is a short
# noun/gerund phrase, not a paragraph; cap keeps fragments out.
_CSN_OBJ_MAX_WORDS = 12


def _csn_first_sentence(doc):
    """First sentence/line of a cleaned docstring summary.

    Cut at the first sentence terminator ('. ', '.\\n', end) or the first
    blank-line break (already collapsed to a single space by
    clean_csn_docstring, so we cut at the first period). Returns the
    summary with surrounding whitespace stripped, no trailing period.
    """
    if not doc:
        return ""
    # clean_csn_docstring already joined lines with spaces; cut at first
    # sentence-final period followed by space/end, or first newline.
    m = _re_csn.search(r"\.(?:\s|$)", doc)
    summary = doc[:m.start()] if m else doc
    summary = summary.split("\n", 1)[0]
    return summary.strip().rstrip(".").strip()


def _csn_clean_object(phrase):
    """Return a clean object phrase or None if it can't verbalize.

    Rejects phrases with code/markup/URL residue, empties, single bare
    determiners, over-long fragments, or symbol-dominated text.
    """
    if not phrase:
        return None
    # Cut at a clause boundary that signals a SECOND statement / aside:
    # ';' (e.g. "...; see ldap module"), ' - ' / ' -- ' digressions, or a
    # mid-phrase colon that introduces a code/example aside
    # ("...classes admin site: @pages.register...").
    phrase = _re_csn.split(r"\s*;\s*|\s+--?\s+|:\s", phrase, 1)[0]
    # Numpy-doc section residue: cut ONLY when a section header is
    # immediately followed by a dash underline ("... Parameters ----------
    # zi") -- the real numpy signature -- so we do NOT clip ordinary prose
    # like "according to preset parameters". Also cut any standalone run of
    # >=3 dashes (a section underline that lost its header).
    phrase = _re_csn.split(
        r"\s+(?:Parameters|Returns?|Yields?|Raises?|Examples?|Notes?|"
        r"Attributes|Args|Arguments|See Also)\s+-{3,}", phrase, 1,
        flags=_re_csn.IGNORECASE)[0]
    phrase = _re_csn.split(r"-{3,}", phrase, 1)[0]
    phrase = phrase.strip().rstrip(".,;:").strip()
    # Collapse internal whitespace.
    phrase = _re_csn.sub(r"\s+", " ", phrase)
    if not phrase:
        return None
    # Reject dunder / leaked-identifier residue (e.g. '__cached_json').
    if "__" in phrase:
        return None
    # Drop a 'see also' / 'see X' tail that survived as its own clause.
    pl = phrase.lower()
    if pl.startswith(("see ", "see:", "cf ", "e.g", "i.e", "ref ")):
        return None
    if _CSN_OBJ_BAD_RE.search(phrase):
        return None
    # Reject a leaked source-file reference in the object
    # (e.g. "...in cosmology.py list", "...versioneer.py file...").
    if _re_csn.search(r"\b\w+\.(?:py|js|ts|json|cpp|cc|hpp|java|go|rb|rs)\b",
                      phrase):
        return None
    # Must contain real words, and not be just a determiner.
    words = phrase.split()
    if len(words) < 1 or len(words) > _CSN_OBJ_MAX_WORDS:
        return None
    if not _CSN_OBJ_WORD_RE.search(phrase):
        return None
    low = phrase.lower()
    if low in ("a", "an", "the", "it", "this", "that", "none", "true", "false"):
        return None
    # A trailing dangling conjunction/preposition signals a cut fragment.
    if words[-1].lower() in ("the", "a", "an", "of", "to", "for", "with",
                             "and", "or", "in", "on", "by", "from", "as",
                             "that", "which", "if", "when"):
        # Try trimming the dangling tail once.
        words = words[:-1]
        if not words:
            return None
        phrase = " ".join(words)
        if words[-1].lower() in ("the", "a", "an", "of", "to", "for"):
            return None
    return phrase


def _csn_summary_bond(anchor, doc, url):
    """Produce AT MOST ONE clean typed bond from a docstring summary.

    Returns a bond dict {subject, relation, object, weight, confidence,
    source_url} or None if no clean, verbalizable bond can be made.
    """
    summary = _csn_first_sentence(doc)
    if not summary or len(summary.split()) < 2:
        return None
    # Drop auto-generated / non-declarative stubs.
    low = summary.lower()
    if low.startswith(("todo", "fixme", "see ", "see:", "deprecated",
                       "note:", "warning:", "internal", "helper for",
                       "alias for", "same as", "wrapper around")):
        # 'wrapper around X' is actually useful -> handle below, others skip.
        if not low.startswith("wrapper around"):
            return None

    relation = None
    obj = None

    # 1) Definitional patterns first (richest, most reliable).
    for rx, rel in _CSN_DEF_PATTERNS:
        m = rx.match(summary)
        if m:
            cand = _csn_clean_object(m.group(1))
            if cand:
                relation, obj = rel, cand
                break

    # 2) 'wrapper around X' / 'wrapper for X' -> UsedFor X.
    if relation is None and low.startswith("wrapper around"):
        cand = _csn_clean_object(summary[len("wrapper around"):])
        if cand:
            relation, obj = "UsedFor", cand

    # 3) Imperative / 3rd-person action verb -> UsedFor <gerund phrase>.
    if relation is None:
        words = summary.split()
        first = _re_csn.sub(r"[^A-Za-z]", "", words[0]).lower()
        # 3rd-person 's' form: 'computes' -> 'compute'.
        base = first
        if base.endswith("es") and base[:-2] in _CSN_ACTION_VERBS:
            base = base[:-2]
        elif base.endswith("s") and base[:-1] in _CSN_ACTION_VERBS:
            base = base[:-1]
        if base in _CSN_ACTION_VERBS:
            rest = " ".join(words[1:])
            rest = _csn_clean_object(rest)
            if rest:
                # 'return ...' already covered by def-pattern; if it slipped
                # through (e.g. 'returns the result'), treat as Produces.
                if base == "return":
                    relation, obj = "Produces", rest
                else:
                    gerund = _csn_gerund(base)
                    relation, obj = "UsedFor", gerund + " " + rest

    if relation is None or not obj:
        return None
    # Final guard: object must not echo the subject (no-op / near-tautology
    # bond). Reject exact echo AND the case where the object's content words
    # (after stripping a leading gerund + determiner) begin with the full
    # subject phrase -- e.g. subject 'register' / object 'registering a
    # switch', or subject 'ordered values by indexes' / object 'values by
    # indexes'. Compares on lowercased alnum word tokens.
    def _wtok(s):
        return _re_csn.findall(r"[a-z0-9]+", s.lower())
    a_tok = _wtok(anchor)
    o_tok = _wtok(obj)
    if a_tok == o_tok:
        return None
    # Drop a leading gerund ("-ing") + determiner from the object before
    # comparing, so 'registering a switch' -> ['switch'] vs subject
    # ['register'] still trips on the stem.
    o_cmp = list(o_tok)
    if o_cmp and o_cmp[0].endswith("ing"):
        o_cmp = o_cmp[1:]
    if o_cmp and o_cmp[0] in ("a", "an", "the"):
        o_cmp = o_cmp[1:]
    # Subject stem (drop trailing 'e' for compute/computing-style match).
    a_stem = a_tok[0][:-1] if (a_tok and a_tok[0].endswith("e")) else (a_tok[0] if a_tok else "")
    o_first = o_tok[0] if o_tok else ""
    o_first_stem = o_first[:-3] if o_first.endswith("ing") else o_first
    if len(a_tok) == 1 and a_stem and o_first_stem.startswith(a_stem):
        # Single-word subject fully echoed as the object's leading verb.
        if o_cmp == a_tok[1:] or o_cmp == [] or set(o_cmp) <= set(a_tok):
            return None
    # Multi-word subject: object content words are a subset/echo of subject.
    if len(a_tok) > 1 and o_cmp and set(o_cmp) <= set(a_tok):
        return None
    return {
        "subject": anchor,
        "relation": relation,
        "object": obj,
        "weight": 0.5,
        "confidence": "Unverified",
        "source_url": url,
    }


def ingest_code_search_net(client, cp, smoke_limit=None):
    """Tier-1 CodeSearchNet function-documentation strings.

    DECLARATIVE path identical to ingest_wikipedia: anchor + filtered text
    -> smart_extractor.extract_bonds_from_wikipedia_article -> R1
    complementary pass -> collapse_to_richest_relation. Only the NL
    docstring is the payload; func_code_string is never read.
    """
    from datasets import load_dataset
    corpus_id = "code-search-net/code_search_net"
    target_total = smoke_limit if smoke_limit else 2_000_000
    log_progress(f"code_search_net: starting load_dataset(streaming) "
                 f"target={target_total} smoke={bool(smoke_limit)}")
    ds = load_dataset("code-search-net/code_search_net",
                      split="train", streaming=True)

    sink = BatchSink(client, Throttle(THROTTLE_CODING))
    sink.bonds_landed   = cp.get(corpus_id, {}).get("bonds_landed", 0)
    sink.bonds_errors   = cp.get(corpus_id, {}).get("errors_count", 0)
    sink.bonds_skipped_dedup = cp.get(corpus_id, {}).get("bonds_skipped_dedup", 0)
    last_saved_id = cp.get(corpus_id, {}).get("last_saved_id", -1)
    last_save_t = time.monotonic()
    last_landed = sink.bonds_landed
    rows = 0
    skipped_trivial = 0
    start_t = time.monotonic()
    for ex in ds:
        if rows >= target_total:
            break
        if rows <= last_saved_id:
            rows += 1
            continue
        doc = clean_csn_docstring(ex.get("func_documentation_string") or "")
        # Skip empty/trivial docstrings: need real declarative prose, not a
        # one-word label. >=40 chars AND >=6 words is the floor.
        if len(doc) < 40 or len(doc.split()) < 6:
            skipped_trivial += 1
            rows += 1
            continue
        anchor = _csn_anchor(ex.get("func_name"), ex.get("repository_name"))
        if not anchor:
            skipped_trivial += 1
            rows += 1
            continue
        # Code-shape guard on the SUBJECT anchor itself (mirror SO loader).
        if any(tok in anchor.lower() for tok in _CODE_SHAPE_TOKENS):
            skipped_trivial += 1
            rows += 1
            continue
        url = ex.get("func_code_url") or ""
        # REWORK: summary-only single clean typed bond. The full Wikipedia
        # extractor + multi-sentence body pass are DROPPED -- they produced
        # ~49% junk (leaked doctest var names, mid-clause fragments,
        # generic RelatedTo). One clean bond that verbalizes correctly beats
        # many that don't.
        try:
            b = _csn_summary_bond(anchor, doc, url)
        except Exception:
            b = None
            sink.bonds_errors += 1
        if b is None:
            skipped_trivial += 1
            rows += 1
            continue
        # Object code-shape guard: never let code tokens through the object.
        if any(tok in (b["object"] or "").lower() for tok in _CODE_SHAPE_TOKENS):
            skipped_trivial += 1
            rows += 1
            continue
        try:
            sink.add({
                "corpus": corpus_id,
                "source_url": b.get("source_url") or url,
                "run_id": RUN_ID_TIER1,
                "subject": b["subject"],
                "relation": b["relation"],
                "object": b["object"],
                "weight": b.get("weight", 0.5),
                "confidence": b.get("confidence", "Unverified"),
            })
        except Exception:
            sink.bonds_errors += 1
        rows += 1
        if rows % 1000 == 0:
            sink.flush()
            bonds_since_save = sink.bonds_landed - last_landed
            elapsed = time.monotonic() - start_t
            rate = rows / elapsed if elapsed > 0 else 0
            log_progress(f"code_search_net: rows={rows} "
                         f"bonds_landed={sink.bonds_landed} errors={sink.bonds_errors} "
                         f"deduped={sink.bonds_skipped_dedup} gated={sink.bonds_gated} rejected={sink.bonds_rejected} "
                         f"trivial_skipped={skipped_trivial} rate={rate:.1f}row/s")
            cp[corpus_id] = {"bonds_landed":sink.bonds_landed,"errors_count":sink.bonds_errors,
                             "bonds_skipped_dedup":sink.bonds_skipped_dedup,
                             "last_saved_id":last_saved_id}
            save_checkpoint(cp)
            new_t, _b, saved_ok = _maybe_save(client, last_save_t, bonds_since_save,
                                              cp, corpus_id, rows-1)
            if saved_ok:
                last_saved_id = rows - 1
                last_landed = sink.bonds_landed
            last_save_t = new_t
    sink.flush()
    bonds_landed = sink.bonds_landed
    bonds_errors = sink.bonds_errors
    try:
        r = client.call("lu4d_save", {})
        if call_ok(r):
            cp[corpus_id] = {"bonds_landed":bonds_landed,"errors_count":bonds_errors,
                             "bonds_skipped_dedup":sink.bonds_skipped_dedup,
                             "last_saved_id":rows-1, "completed": True}
            save_checkpoint(cp)
    except Exception:
        pass
    log_progress(f"code_search_net: DONE rows={rows} landed={bonds_landed} "
                 f"errors={bonds_errors} trivial_skipped={skipped_trivial}")
    return {"rows_processed":rows,"bonds_landed":bonds_landed,
            "bonds_errors":bonds_errors,"trivial_skipped":skipped_trivial}


def _import_pydocs_extractor():
    """Load the sibling pydocs_extractor module by path (zero-dep)."""
    import importlib.util, pathlib
    loader_path = pathlib.Path(__file__).resolve().parent / "pydocs_extractor.py"
    spec = importlib.util.spec_from_file_location("pydocs_extractor", loader_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Default location of the extracted python.org plain-text docs bundle
# (python-3.x-docs-text.tar.bz2 extracted). Override with --pydocs-root.
PYDOCS_ROOT_DEFAULT = "/var/lib/lu/data/pydocs/python-3.14-docs-text"


def _import_mdn_extractor():
    """Load the sibling mdn_extractor module by path (zero-dep)."""
    import importlib.util, pathlib
    loader_path = pathlib.Path(__file__).resolve().parent / "mdn_extractor.py"
    spec = importlib.util.spec_from_file_location("mdn_extractor", loader_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Default location of the cloned mdn/content repo (git clone --depth 1
# https://github.com/mdn/content). Override with --mdn-root.
MDN_ROOT_DEFAULT = "/var/lib/lu/data/mdn/content"


def _import_pycode_extractor():
    """Load the sibling pycode_extractor module by path (zero-dep)."""
    import importlib.util, pathlib
    loader_path = pathlib.Path(__file__).resolve().parent / "pycode_extractor.py"
    spec = importlib.util.spec_from_file_location("pycode_extractor", loader_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Default root for the Python structural code extractor (Phase-1).
# Points at the installed stdlib source. Override with --pycode-root.
PYCODE_ROOT_DEFAULT = "/usr/lib/python3.12"


def ingest_mdn(client, cp, smoke_limit=None, mdn_root=None):
    """Phase-2b Tier-1: MDN web-platform documentation (declarative
    web-platform concepts -- what a JS/CSS/HTML/Web-API feature IS,
    PRODUCES, or is USED FOR).

    DECLARATIVE path, same teach route as wikipedia/CSN/python-docs:
    extractor emits AT MOST ONE clean typed bond per concept page
    (mdn_extractor.extract_bond_from_page) -> BatchSink ->
    lu4d_teach_corpus_batch (T.3 gating + TrainingCorpus stamp). NOT
    _ingest_tier2. Corpus id 'mdn', trust 0.95.

    Checkpointing is FILE-INDEXED, identical to python_docs:
    cp[corpus]['last_saved_id'] is the index (into the sorted doc-file
    list) of the last file whose bonds were persisted. Resume re-streams
    from last_saved_id+1; per-bond server dedup makes re-extraction
    idempotent. smoke_limit caps the number of FILES processed.
    """
    mx = _import_mdn_extractor()
    root = mdn_root or MDN_ROOT_DEFAULT
    corpus_id = "mdn"
    if not os.path.isdir(root):
        log_progress(f"mdn: SKIPPED -- root not found: {root}")
        return {"skipped_reason": f"mdn root not found: {root}"}

    files = list(mx.iter_doc_files(root))
    if smoke_limit:
        files = files[:smoke_limit]
    log_progress(f"mdn: starting root={root} files={len(files)} "
                 f"smoke={bool(smoke_limit)}")

    sink = BatchSink(client, Throttle(THROTTLE_CODING))
    sink.bonds_landed   = cp.get(corpus_id, {}).get("bonds_landed", 0)
    sink.bonds_errors   = cp.get(corpus_id, {}).get("errors_count", 0)
    sink.bonds_skipped_dedup = cp.get(corpus_id, {}).get("bonds_skipped_dedup", 0)
    last_saved_id = cp.get(corpus_id, {}).get("last_saved_id", -1)
    last_save_t = time.monotonic()
    last_landed = sink.bonds_landed
    start_t = time.monotonic()
    files_processed = 0
    files_zero = 0

    for idx, path in enumerate(files):
        if idx <= last_saved_id:
            continue
        try:
            text = open(path, encoding="utf-8", errors="replace").read()
        except Exception as e:
            sink.bonds_errors += 1
            log_progress(f"mdn: read failed {path}: {e}")
            continue
        url = mx.doc_source_url(root, path)
        try:
            b = mx.extract_bond_from_page(text, url)
        except Exception as e:
            sink.bonds_errors += 1
            log_progress(f"mdn: extract failed {path}: {e}")
            b = None
        if not b:
            files_zero += 1
        else:
            # Object code-shape guard (mirror CSN/pydocs): never let code
            # tokens through the object phrase.
            if any(tok in (b["object"] or "").lower()
                   for tok in _CODE_SHAPE_TOKENS):
                files_zero += 1
            else:
                try:
                    sink.add({
                        "corpus": corpus_id,
                        "source_url": b.get("source_url") or url,
                        "run_id": RUN_ID_TIER1,
                        "subject": b["subject"],
                        "relation": b["relation"],
                        "object": b["object"],
                        "weight": b.get("weight", 0.5),
                        "confidence": b.get("confidence", "Unverified"),
                    })
                except Exception:
                    sink.bonds_errors += 1
        files_processed += 1
        # Flush + periodic save every file (files are small; keeps the
        # batch buffer bounded and checkpoints fine-grained).
        sink.flush()
        bonds_since_save = sink.bonds_landed - last_landed
        cp[corpus_id] = {"bonds_landed": sink.bonds_landed,
                         "errors_count": sink.bonds_errors,
                         "bonds_skipped_dedup": sink.bonds_skipped_dedup,
                         "last_saved_id": last_saved_id}
        save_checkpoint(cp)
        new_t, _b, saved_ok = _maybe_save(client, last_save_t,
                                          bonds_since_save, cp, corpus_id, idx)
        if saved_ok:
            last_saved_id = idx
            last_landed = sink.bonds_landed
        last_save_t = new_t
        if files_processed % 100 == 0:
            elapsed = time.monotonic() - start_t
            rate = files_processed / elapsed if elapsed > 0 else 0
            log_progress(f"mdn: files={files_processed}/{len(files)} "
                         f"bonds_landed={sink.bonds_landed} "
                         f"errors={sink.bonds_errors} "
                         f"deduped={sink.bonds_skipped_dedup} "
                         f"gated={sink.bonds_gated} rejected={sink.bonds_rejected} zero_files={files_zero} "
                         f"rate={rate:.2f}file/s")

    sink.flush()
    bonds_landed = sink.bonds_landed
    bonds_errors = sink.bonds_errors
    try:
        r = client.call("lu4d_save", {})
        if call_ok(r):
            cp[corpus_id] = {"bonds_landed": bonds_landed,
                             "errors_count": bonds_errors,
                             "bonds_skipped_dedup": sink.bonds_skipped_dedup,
                             "last_saved_id": len(files) - 1,
                             "completed": True}
            save_checkpoint(cp)
    except Exception:
        pass
    log_progress(f"mdn: DONE files={files_processed} "
                 f"landed={bonds_landed} errors={bonds_errors} "
                 f"deduped={sink.bonds_skipped_dedup} zero_files={files_zero}")
    return {"files_processed": files_processed,
            "bonds_landed": bonds_landed,
            "bonds_errors": bonds_errors,
            "bonds_skipped_dedup": sink.bonds_skipped_dedup,
            "zero_bond_files": files_zero}


def ingest_python_docs(client, cp, smoke_limit=None, pydocs_root=None):
    """Phase-2a Tier-1: Python official documentation (declarative coding
    concepts -- what a module/function/class IS or PRODUCES).

    DECLARATIVE path, same teach route as wikipedia/CSN: extractor emits
    one clean typed bond per concept (pydocs_extractor.summary_bond) ->
    BatchSink -> lu4d_teach_corpus_batch (T.3 gating + TrainingCorpus
    stamp). NOT _ingest_tier2. Corpus id 'python-docs', trust 0.95.

    Checkpointing is FILE-INDEXED: cp[corpus]['last_saved_id'] is the
    index (into the sorted doc-file list) of the last file whose bonds
    were persisted. Resume re-streams from last_saved_id+1; bonds added
    between saves are re-formed by re-extraction (idempotent, per-bond
    server dedup). smoke_limit caps the number of FILES processed.
    """
    px = _import_pydocs_extractor()
    root = pydocs_root or PYDOCS_ROOT_DEFAULT
    corpus_id = "python-docs"
    if not os.path.isdir(root):
        log_progress(f"python_docs: SKIPPED -- root not found: {root}")
        return {"skipped_reason": f"pydocs root not found: {root}"}

    files = list(px.iter_doc_files(root))
    if smoke_limit:
        files = files[:smoke_limit]
    log_progress(f"python_docs: starting root={root} files={len(files)} "
                 f"smoke={bool(smoke_limit)}")

    sink = BatchSink(client, Throttle(THROTTLE_CODING))
    sink.bonds_landed   = cp.get(corpus_id, {}).get("bonds_landed", 0)
    sink.bonds_errors   = cp.get(corpus_id, {}).get("errors_count", 0)
    sink.bonds_skipped_dedup = cp.get(corpus_id, {}).get("bonds_skipped_dedup", 0)
    last_saved_id = cp.get(corpus_id, {}).get("last_saved_id", -1)
    last_save_t = time.monotonic()
    last_landed = sink.bonds_landed
    start_t = time.monotonic()
    files_processed = 0
    sections_seen = 0
    files_zero = 0

    for idx, path in enumerate(files):
        if idx <= last_saved_id:
            continue
        try:
            text = open(path, encoding="utf-8", errors="replace").read()
        except Exception as e:
            sink.bonds_errors += 1
            log_progress(f"python_docs: read failed {path}: {e}")
            continue
        url = px.doc_source_url(root, path)
        try:
            bonds = px.extract_bonds_from_pydoc(text, url)
        except Exception as e:
            sink.bonds_errors += 1
            log_progress(f"python_docs: extract failed {path}: {e}")
            bonds = []
        sections_seen += max(1, len(bonds))
        if not bonds:
            files_zero += 1
        for b in bonds:
            # Object code-shape guard (mirror CSN): never let code tokens
            # through the object phrase.
            if any(tok in (b["object"] or "").lower()
                   for tok in _CODE_SHAPE_TOKENS):
                continue
            try:
                sink.add({
                    "corpus": corpus_id,
                    "source_url": b.get("source_url") or url,
                    "run_id": RUN_ID_TIER1,
                    "subject": b["subject"],
                    "relation": b["relation"],
                    "object": b["object"],
                    "weight": b.get("weight", 0.5),
                    "confidence": b.get("confidence", "Unverified"),
                })
            except Exception:
                sink.bonds_errors += 1
        files_processed += 1
        # Flush + periodic save every file (files are small; one flush per
        # file keeps the batch buffer bounded and checkpoints fine-grained).
        sink.flush()
        bonds_since_save = sink.bonds_landed - last_landed
        cp[corpus_id] = {"bonds_landed": sink.bonds_landed,
                         "errors_count": sink.bonds_errors,
                         "bonds_skipped_dedup": sink.bonds_skipped_dedup,
                         "last_saved_id": last_saved_id}
        save_checkpoint(cp)
        new_t, _b, saved_ok = _maybe_save(client, last_save_t,
                                          bonds_since_save, cp, corpus_id, idx)
        if saved_ok:
            last_saved_id = idx
            last_landed = sink.bonds_landed
        last_save_t = new_t
        if files_processed % 25 == 0:
            elapsed = time.monotonic() - start_t
            rate = files_processed / elapsed if elapsed > 0 else 0
            log_progress(f"python_docs: files={files_processed}/{len(files)} "
                         f"bonds_landed={sink.bonds_landed} "
                         f"errors={sink.bonds_errors} "
                         f"deduped={sink.bonds_skipped_dedup} "
                         f"gated={sink.bonds_gated} rejected={sink.bonds_rejected} zero_files={files_zero} "
                         f"rate={rate:.2f}file/s")

    sink.flush()
    bonds_landed = sink.bonds_landed
    bonds_errors = sink.bonds_errors
    try:
        r = client.call("lu4d_save", {})
        if call_ok(r):
            cp[corpus_id] = {"bonds_landed": bonds_landed,
                             "errors_count": bonds_errors,
                             "bonds_skipped_dedup": sink.bonds_skipped_dedup,
                             "last_saved_id": len(files) - 1,
                             "completed": True}
            save_checkpoint(cp)
    except Exception:
        pass
    log_progress(f"python_docs: DONE files={files_processed} "
                 f"landed={bonds_landed} errors={bonds_errors} "
                 f"deduped={sink.bonds_skipped_dedup} zero_files={files_zero}")
    return {"files_processed": files_processed,
            "bonds_landed": bonds_landed,
            "bonds_errors": bonds_errors,
            "bonds_skipped_dedup": sink.bonds_skipped_dedup,
            "zero_bond_files": files_zero}


# Markdown code-block stripping for StackOverflow bodies. The mikex86
# dump stores answer Body as MARKDOWN (```fenced``` blocks + 4-space
# indented blocks), NOT HTML <code>/<pre>. html_to_text only strips HTML
# code tags, so we must remove markdown code blocks FIRST or code tokens
# leak into bond subjects.
_MD_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_MD_FENCE_TILDE_RE = re.compile(r"~~~.*?~~~", re.DOTALL)


def strip_markdown_code(body):
    """Remove markdown fenced code blocks and 4-space-indented code lines
    from a StackOverflow markdown body, leaving declarative prose."""
    if not body:
        return ""
    body = _MD_FENCE_RE.sub(" ", body)
    body = _MD_FENCE_TILDE_RE.sub(" ", body)
    # Drop 4-space / tab indented code lines (markdown code blocks).
    kept = []
    for ln in body.splitlines():
        if re.match(r"^( {4,}|\t)\S", ln):
            continue
        kept.append(ln)
    return "\n".join(kept)


def ingest_stackoverflow(client, cp, smoke_limit=None, min_score=5):
    """Tier-1 StackOverflow high-score answers (mikex86/stackoverflow-posts).

    Reuses stackexchange_loader.extract_bonds_for_answer for the declarative
    extraction (html_to_text -> q-frame filter -> anchor -> smart_extractor
    -> R1 pass -> code-shape guard). Strips markdown code fences first since
    this dump's Body is markdown, not HTML.
    """
    from datasets import load_dataset
    import importlib.util, pathlib
    loader_path = pathlib.Path(__file__).resolve().parent / "stackexchange_loader.py"
    spec = importlib.util.spec_from_file_location("se_loader_mod_so", loader_path)
    se = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(se)

    corpus_id = "stackoverflow"
    target_total = smoke_limit if smoke_limit else 5_000_000
    log_progress(f"stackoverflow: starting load_dataset(streaming) "
                 f"target={target_total} min_score={min_score} smoke={bool(smoke_limit)}")
    ds = load_dataset("mikex86/stackoverflow-posts", split="train", streaming=True)

    sink = BatchSink(client, Throttle(THROTTLE_CODING))
    sink.bonds_landed = cp.get(corpus_id, {}).get("bonds_landed", 0)
    sink.bonds_errors = cp.get(corpus_id, {}).get("errors_count", 0)
    sink.bonds_skipped_dedup = cp.get(corpus_id, {}).get("bonds_skipped_dedup", 0)
    last_saved_id = cp.get(corpus_id, {}).get("last_saved_id", -1)
    last_save_t = time.monotonic()
    last_landed = sink.bonds_landed
    rows = 0            # answers (PostTypeId==2, Score>=min_score) seen
    answers_no_anchor = 0
    answers_q_only = 0
    last_post_id = None
    for ex in ds:
        if str(ex.get("PostTypeId")) != "2":
            continue
        if (ex.get("Score") or 0) < min_score:
            continue
        if ex.get("DeletionDate"):
            continue
        body = ex.get("Body") or ""
        if not body.strip():
            continue
        if rows >= target_total:
            break
        if rows <= last_saved_id:
            rows += 1
            continue
        last_post_id = ex.get("Id")
        cleaned = strip_markdown_code(body)
        answer = {"id": str(ex.get("Id")), "parent_id": str(ex.get("ParentId")),
                  "score": int(ex.get("Score") or 0), "body": cleaned,
                  "creation_date": ex.get("CreationDate") or ""}
        try:
            bonds, diag = se.extract_bonds_for_answer(answer, "stackoverflow.com")
        except Exception:
            bonds, diag = [], {"skipped_reason": "extractor_error"}
            sink.bonds_errors += 1
        reason = diag.get("skipped_reason")
        if reason == "no_anchor":
            answers_no_anchor += 1
        elif reason == "all_sentences_question_framed":
            answers_q_only += 1
        for b in bonds:
            try:
                sink.add({
                    "corpus": corpus_id,
                    "source_url": f"https://stackoverflow.com/a/{ex.get('Id')}",
                    "run_id": RUN_ID_TIER1,
                    "subject": b["subject"],
                    "relation": b["relation"],
                    "object": b["object"],
                    "weight": b.get("weight", 0.5),
                    "confidence": b.get("confidence", "Unverified"),
                })
            except Exception:
                sink.bonds_errors += 1
        rows += 1
        if rows % 500 == 0:
            sink.flush()
            bonds_since_save = sink.bonds_landed - last_landed
            log_progress(f"stackoverflow: answers={rows} "
                         f"landed={sink.bonds_landed} errors={sink.bonds_errors} "
                         f"no_anchor={answers_no_anchor} q_only={answers_q_only} "
                         f"deduped={sink.bonds_skipped_dedup} gated={sink.bonds_gated} rejected={sink.bonds_rejected}")
            cp[corpus_id] = {"bonds_landed":sink.bonds_landed,"errors_count":sink.bonds_errors,
                             "bonds_skipped_dedup":sink.bonds_skipped_dedup,
                             "last_saved_id":last_saved_id,"last_post_id":last_post_id}
            save_checkpoint(cp)
            new_t, _b, saved_ok = _maybe_save(client, last_save_t, bonds_since_save,
                                              cp, corpus_id, rows-1)
            if saved_ok:
                last_saved_id = rows - 1
                last_landed = sink.bonds_landed
            last_save_t = new_t
    sink.flush()
    bonds_landed = sink.bonds_landed
    bonds_errors = sink.bonds_errors
    try:
        r = client.call("lu4d_save", {})
        if call_ok(r):
            cp[corpus_id] = {"bonds_landed":bonds_landed,"errors_count":bonds_errors,
                             "bonds_skipped_dedup":sink.bonds_skipped_dedup,
                             "last_saved_id":rows-1,"last_post_id":last_post_id,
                             "completed": True}
            save_checkpoint(cp)
    except Exception:
        pass
    log_progress(f"stackoverflow: DONE answers={rows} landed={bonds_landed} "
                 f"errors={bonds_errors} no_anchor={answers_no_anchor} "
                 f"q_only={answers_q_only}")
    return {"answers_processed":rows,"bonds_landed":bonds_landed,
            "bonds_errors":bonds_errors,"answers_no_anchor":answers_no_anchor,
            "answers_q_only":answers_q_only}


def ingest_grokopedia(client, cp, smoke_limit=None,
                      seeds=None, sitemap_index_url=None):
    """Tier-1 Grokipedia articles via sitemap walk or explicit seeds.

    Shape Y wiring. In smoke mode, uses DEFAULT_SEEDS from the loader
    (diverse-topic seed list) capped at smoke_limit so we don't fetch
    the entire sitemap during dry-runs. In production, walks the sitemap.
    """
    import importlib.util, pathlib
    loader_path = (
        pathlib.Path(__file__).resolve().parent / "grokopedia_loader.py"
    )
    spec = importlib.util.spec_from_file_location("grok_loader_mod", loader_path)
    grk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(grk)

    corpus_id = grk.CORPUS_ID  # "grokipedia"
    progress_path = grk.DEFAULT_PROGRESS_PATH

    # Smoke mode: prefer the loader's diverse-topic seed list (cached/local)
    # over a live sitemap walk so smoke runs don't depend on the network.
    if smoke_limit and not seeds:
        seeds = list(grk.DEFAULT_SEEDS)[: max(1, int(smoke_limit) * 2)]

    log_progress(
        f"grokopedia: starting; "
        f"mode={'seeds' if seeds else 'sitemap'} "
        f"seeds_count={len(seeds) if seeds else 0} "
        f"target={smoke_limit or 'unbounded'} smoke={bool(smoke_limit)}"
    )

    sink = BatchSink(client, Throttle(THROTTLE_TIER2))
    sink.bonds_landed = cp.get(corpus_id, {}).get("bonds_landed", 0)
    sink.bonds_errors = cp.get(corpus_id, {}).get("errors_count", 0)
    sink.bonds_skipped_dedup = cp.get(corpus_id, {}).get("bonds_skipped_dedup", 0)
    last_landed = sink.bonds_landed
    last_save_t = time.monotonic()

    articles_seen = 0
    articles_no_bond = 0
    skipped_reason = None
    last_url = None

    sitemap_arg = sitemap_index_url or grk.SITEMAP_INDEX_URL
    try:
        for article in grk.stream_articles(
            seeds=seeds,
            sitemap_index_url=sitemap_arg,
            limit=smoke_limit,
            progress_path=progress_path,
        ):
            articles_seen += 1
            last_url = article.get("url")
            try:
                bonds = grk.extract_bonds(article)
            except Exception:
                bonds = []
                sink.bonds_errors += 1
            if not bonds:
                articles_no_bond += 1
            for b in bonds:
                # CORPUS ATTRIBUTION call-site (grokipedia, Tier 1).
                sink.add({
                    "corpus": corpus_id,
                    "source_url": last_url or "",
                    "run_id": RUN_ID_TIER1,
                    "subject": b["subject"],
                    "relation": b["relation"],
                    "object": b["object"],
                    "weight": b.get("weight", 0.5),
                    "confidence": b.get("confidence", "Unverified"),
                })
            # Persist resume record so the loader's own resume gate catches
            # this URL on a future invocation.
            try:
                grk._append_resume(progress_path, {
                    "url": last_url, "queued": len(bonds),
                })
            except Exception:
                pass
            if articles_seen % 25 == 0:
                sink.flush()
                bonds_since_save = sink.bonds_landed - last_landed
                log_progress(
                    f"grokopedia: articles={articles_seen}/{smoke_limit or '?'} "
                    f"landed={sink.bonds_landed} no_bond={articles_no_bond}"
                )
                cp[corpus_id] = {
                    "bonds_landed": sink.bonds_landed,
                    "errors_count": sink.bonds_errors,
                    "bonds_skipped_dedup": sink.bonds_skipped_dedup,
                    "last_url": last_url,
                }
                save_checkpoint(cp)
                new_t, _b_disc, saved_ok = _maybe_save(
                    client, last_save_t, bonds_since_save,
                    cp, corpus_id, articles_seen,
                )
                if saved_ok:
                    last_landed = sink.bonds_landed
                last_save_t = new_t
        sink.flush()
    except Exception as e:
        skipped_reason = (
            f"grokopedia stream failed: {type(e).__name__}: {str(e)[:200]}"
        )
        log_progress(f"grokopedia: WARN \u2014 {skipped_reason}")

    try:
        r = client.call("lu4d_save", {})
        if call_ok(r):
            cp[corpus_id] = {
                "bonds_landed": sink.bonds_landed,
                "errors_count": sink.bonds_errors,
                "bonds_skipped_dedup": sink.bonds_skipped_dedup,
                "last_url": last_url,
                "completed": True,
                "skipped_reason": skipped_reason,
            }
            save_checkpoint(cp)
    except Exception:
        pass
    log_progress(
        f"grokopedia: DONE landed={sink.bonds_landed} "
        f"errors={sink.bonds_errors} articles={articles_seen} "
        f"no_bond={articles_no_bond} skip={skipped_reason}"
    )
    return {
        "bonds_landed": sink.bonds_landed,
        "bonds_errors": sink.bonds_errors,
        "bonds_skipped_dedup": sink.bonds_skipped_dedup,
        "articles_seen": articles_seen,
        "articles_no_bond": articles_no_bond,
        "skipped_reason": skipped_reason,
    }




def _ingest_tier2(client, cp, name, dataset_repo, dataset_split,
                  prompt_field, response_field, total_estimate, smoke_limit=None):
    from datasets import load_dataset
    corpus_id = dataset_repo
    target_total = smoke_limit if smoke_limit else total_estimate
    log_progress(f"{name}: starting load_dataset(streaming) target={target_total} smoke={bool(smoke_limit)}")

    throttle = Throttle(THROTTLE_TIER2)
    sink = BatchSink(client, throttle)
    sink.bonds_landed   = cp.get(corpus_id, {}).get("bonds_landed", 0)
    sink.bonds_errors   = cp.get(corpus_id, {}).get("errors_count", 0)
    sink.bonds_skipped_dedup = cp.get(corpus_id, {}).get("bonds_skipped_dedup", 0)
    statements_landed = cp.get(corpus_id, {}).get("statements_landed", 0)
    last_saved_id = cp.get(corpus_id, {}).get("last_saved_id", -1)
    last_save_t = time.monotonic(); last_landed = sink.bonds_landed
    examples = 0
    skipped_reason = None
    # BR.9 line-698 fix: pre-init names referenced in the post-try cleanup
    # path so an immediate load_dataset() failure does not raise
    # UnboundLocalError on bonds_landed / bonds_errors.
    bonds_landed = sink.bonds_landed
    bonds_errors = sink.bonds_errors
    try:
        ds = load_dataset(dataset_repo, split=dataset_split, streaming=True)
        for ex in ds:
            if examples >= target_total: break
            if examples <= last_saved_id:
                examples += 1; continue
            # Fetch prompt + response — heuristics handle no_robots vs instruction-dataset
            if prompt_field == "messages":
                msgs = ex.get("messages", []) or []
                prompt, response = "", ""
                for m in msgs:
                    if m.get("role") == "user" and not prompt: prompt = m.get("content","")
                    elif m.get("role") == "assistant" and not response: response = m.get("content","")
            else:
                prompt = ex.get(prompt_field,"") or ex.get("instruction","")
                response = ex.get(response_field,"") or ex.get("response","") or ex.get("output","")
            if prompt:
                throttle.wait()
                try:
                    r = client.call("lu4d_user_statement",
                        {"text": prompt[:1000], "source_label": corpus_id})
                    if call_ok(r): statements_landed += 1
                except Exception:
                    pass
            bond = extract_response_bond(prompt, response)
            if bond:
                s,rel,o,w,c = bond
                try:
                    # CORPUS ATTRIBUTION call-site (Tier 2 instruction
                    # datasets). corpus_id is the dataset_repo passed
                    # in by the caller (e.g. 'HuggingFaceH4/no_robots').
                    # See module-level CORPUS ATTRIBUTION block above
                    # BatchSink.
                    sink.add({
                        "corpus":corpus_id, "run_id":RUN_ID_TIER2,
                        "subject":s, "relation":rel, "object":o,
                        "weight":w, "confidence":c,
                    })
                except Exception:
                    sink.bonds_errors += 1
            examples += 1
            if examples % 200 == 0:
                sink.flush()
                bonds_since_save = sink.bonds_landed - last_landed
                log_progress(f"{name}: examples={examples}/{target_total} "
                             f"statements={statements_landed} bonds_landed={sink.bonds_landed} "
                             f"errors={sink.bonds_errors} deduped={sink.bonds_skipped_dedup} "
                             f"gated={sink.bonds_gated} rejected={sink.bonds_rejected}")
                cp[corpus_id] = {"bonds_landed":sink.bonds_landed,"errors_count":sink.bonds_errors,
                                 "bonds_skipped_dedup":sink.bonds_skipped_dedup,
                                 "statements_landed":statements_landed,
                                 "last_saved_id":last_saved_id}
                save_checkpoint(cp)
                new_t, new_b, saved_ok = _maybe_save(client, last_save_t, bonds_since_save, cp, corpus_id, examples-1)
                if saved_ok:
                    last_saved_id = examples - 1
                    last_landed = sink.bonds_landed
                last_save_t = new_t
        sink.flush()
        bonds_landed = sink.bonds_landed
        bonds_errors = sink.bonds_errors
    except Exception as e:
        skipped_reason = f"{name} load failed: {type(e).__name__}: {str(e)[:200]}"
        log_progress(f"{name}: SKIPPED — {skipped_reason}")
    try:
        r = client.call("lu4d_save", {})
        if call_ok(r):
            cp[corpus_id] = {"bonds_landed":bonds_landed,"errors_count":bonds_errors,
                             "statements_landed":statements_landed,
                             "last_saved_id":examples-1, "completed": True,
                             "skipped_reason": skipped_reason}
            save_checkpoint(cp)
    except Exception:
        pass
    log_progress(f"{name}: DONE examples={examples} statements={statements_landed} "
                 f"landed={bonds_landed} errors={bonds_errors} skip={skipped_reason}")
    return {"examples_processed":examples,"statements_landed":statements_landed,
            "bonds_landed":bonds_landed,"bonds_errors":bonds_errors,
            "skipped_reason":skipped_reason}

def ingest_no_robots(client, cp, smoke_limit=None):
    return _ingest_tier2(client, cp, "no_robots", "HuggingFaceH4/no_robots",
                          "train", "messages", "messages", 10000, smoke_limit)

def ingest_instruction_dataset(client, cp, smoke_limit=None):
    return _ingest_tier2(client, cp, "instruction-dataset",
                          "HuggingFaceH4/instruction-dataset", "test",
                          "prompt", "completion", 330, smoke_limit)


# ---- main ----

def ingest_python_code(client, cp, smoke_limit=None, pycode_root=None):
    """Phase-1: Python STRUCTURAL code knowledge (classes, methods,
    params, types, calls, inheritance) parsed from Python SOURCE via
    pycode_extractor (stdlib ast, zero-dep).

    Unlike python_docs (declarative prose ABOUT code), this loader teaches
    Lu the STRUCTURE of code: `[the Dog class] -HasMethod-> [Dog.bark]`,
    `[Dog.bark] -HasParameter-> [loud]`, `[loud] -HasType-> [bool]`. Same
    declarative teach route (lu4d_teach_corpus_batch + T.3 gating), same
    brain as prose -- just corpus-tagged `python-code`, trust ~0.9.

    Checkpointing is FILE-INDEXED (mirrors ingest_python_docs):
    cp[corpus]['last_saved_id'] is the index into the sorted .py file list
    of the last file whose bonds were persisted. Resume re-streams from
    last_saved_id+1; per-bond server dedup makes re-extraction idempotent.
    smoke_limit caps the number of FILES processed.
    """
    px = _import_pycode_extractor()
    root = pycode_root or PYCODE_ROOT_DEFAULT
    corpus_id = "python-code"
    if not os.path.exists(root):
        log_progress(f"python_code: SKIPPED -- root not found: {root}")
        return {"skipped_reason": f"pycode root not found: {root}"}

    files = list(px.iter_py_files(root))
    if smoke_limit:
        files = files[:smoke_limit]
    log_progress(f"python_code: starting root={root} files={len(files)} "
                 f"smoke={bool(smoke_limit)}")

    sink = BatchSink(client, Throttle(THROTTLE_CODING))
    sink.bonds_landed   = cp.get(corpus_id, {}).get("bonds_landed", 0)
    sink.bonds_errors   = cp.get(corpus_id, {}).get("errors_count", 0)
    sink.bonds_skipped_dedup = cp.get(corpus_id, {}).get("bonds_skipped_dedup", 0)
    last_saved_id = cp.get(corpus_id, {}).get("last_saved_id", -1)
    last_save_t = time.monotonic()
    last_landed = sink.bonds_landed
    start_t = time.monotonic()
    files_processed = 0
    triples_seen = 0
    files_zero = 0
    syntax_errors = 0

    for idx, path in enumerate(files):
        if idx <= last_saved_id:
            continue
        try:
            text = open(path, encoding="utf-8", errors="replace").read()
        except Exception as e:
            sink.bonds_errors += 1
            log_progress(f"python_code: read failed {path}: {e}")
            continue
        url = px.file_source_url(root, path)
        try:
            bonds = px.extract_bonds_from_pyfile(text, url, filename=path)
        except SyntaxError as e:
            syntax_errors += 1
            bonds = []
        except Exception as e:
            sink.bonds_errors += 1
            log_progress(f"python_code: extract failed {path}: {e}")
            bonds = []
        triples_seen += len(bonds)
        if not bonds:
            files_zero += 1
        for b in bonds:
            # NOTE: deliberately NO _CODE_SHAPE_TOKENS guard here -- our
            # objects ARE clean concept names (type names, qualified
            # method names), never raw code lines. The prose-corpus guard
            # would wrongly drop legitimate structural objects.
            try:
                sink.add({
                    "corpus": corpus_id,
                    "source_url": b.get("source_url") or url,
                    "run_id": RUN_ID_TIER1,
                    "subject": b["subject"],
                    "relation": b["relation"],
                    "object": b["object"],
                    "weight": b.get("weight", 0.8),
                    "confidence": b.get("confidence", "Verified"),
                })
            except Exception:
                sink.bonds_errors += 1
        files_processed += 1
        sink.flush()
        bonds_since_save = sink.bonds_landed - last_landed
        cp[corpus_id] = {"bonds_landed": sink.bonds_landed,
                         "errors_count": sink.bonds_errors,
                         "bonds_skipped_dedup": sink.bonds_skipped_dedup,
                         "last_saved_id": last_saved_id}
        save_checkpoint(cp)
        new_t, _b, saved_ok = _maybe_save(client, last_save_t,
                                          bonds_since_save, cp, corpus_id, idx)
        if saved_ok:
            last_saved_id = idx
            last_landed = sink.bonds_landed
        last_save_t = new_t
        if files_processed % 25 == 0:
            elapsed = time.monotonic() - start_t
            rate = files_processed / elapsed if elapsed > 0 else 0
            log_progress(f"python_code: files={files_processed}/{len(files)} "
                         f"bonds_landed={sink.bonds_landed} "
                         f"errors={sink.bonds_errors} "
                         f"syntax_err={syntax_errors} "
                         f"deduped={sink.bonds_skipped_dedup} "
                         f"gated={sink.bonds_gated} rejected={sink.bonds_rejected} zero_files={files_zero} "
                         f"rate={rate:.2f}file/s")

    sink.flush()
    bonds_landed = sink.bonds_landed
    bonds_errors = sink.bonds_errors
    try:
        r = client.call("lu4d_save", {})
        if call_ok(r):
            cp[corpus_id] = {"bonds_landed": bonds_landed,
                             "errors_count": bonds_errors,
                             "bonds_skipped_dedup": sink.bonds_skipped_dedup,
                             "last_saved_id": len(files) - 1,
                             "completed": True}
            save_checkpoint(cp)
    except Exception:
        pass
    log_progress(f"python_code: DONE files={files_processed} "
                 f"landed={bonds_landed} errors={bonds_errors} "
                 f"syntax_err={syntax_errors} "
                 f"deduped={sink.bonds_skipped_dedup} zero_files={files_zero}")
    return {"files_processed": files_processed,
            "bonds_landed": bonds_landed,
            "bonds_errors": bonds_errors,
            "syntax_errors": syntax_errors,
            "bonds_skipped_dedup": sink.bonds_skipped_dedup,
            "triples_seen": triples_seen,
            "zero_bond_files": files_zero}


def ingest_stack_v2(client, cp, smoke_limit=None, source="local",
                    root=None, langs=None, limit=None):
    """Phase-2: multi-language STRUCTURAL code knowledge from The Stack v2 (or a
    local mixed-language tree), parsed via scripts/code_extractor.py (tree-sitter,
    declarative structural bonds only -- never raw code text). Lands via the SAME
    route as ingest_python_code (BatchSink -> lu4d_teach_corpus_batch, T.3 gating,
    file-indexed checkpoint). Corpus-tagged 'the-stack-v2'.

    source='local' walks `root` (bounded validation, fully runnable).
    source='hf'    streams The Stack v2 IDs + Software Heritage S3 content; this
                   REQUIRES an accepted HF gate token + AWS credentials and raises
                   StackV2AccessError otherwise (see scripts/stack_v2_loader.py).
    """
    import importlib.util
    import pathlib
    loader_path = pathlib.Path(__file__).resolve().parent / "stack_v2_loader.py"
    spec = importlib.util.spec_from_file_location("stack_v2_loader", loader_path)
    sv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sv)

    lim = smoke_limit or limit
    if source == "local":
        if not root or not os.path.exists(root):
            log_progress(f"stack_v2: SKIPPED -- local root not found: {root}")
            return {"skipped_reason": f"stack_v2 local root not found: {root}"}
        files_iter = sv.iter_local_tree(root, limit=lim)
        make_url = lambda pth: sv.local_source_url(root, pth)
    else:
        langlist = [x.strip() for x in (langs or "").split(",") if x.strip()] \
            or list(sv._STACK_V2_LANG_CONFIG.keys())
        try:
            files_iter = sv.iter_stack_v2(langlist, limit=lim)
        except sv.StackV2AccessError as e:
            log_progress(f"stack_v2: SKIPPED -- access gate: {e}")
            return {"skipped_reason": f"stack_v2 access gate: {e}"}
        make_url = lambda pth: pth
    log_progress(f"stack_v2: starting source={source} root={root} limit={lim}")
    summary = sv.run_ingest(client, cp, files_iter, corpus_id=sv.CORPUS_ID,
                            smoke_limit=lim, make_url=make_url)
    log_progress(f"stack_v2: DONE {summary}")
    return summary


def main():
    global PROGRESS_LOG, CHECKPOINT, DONE_SENTINEL
    parser = argparse.ArgumentParser()
    parser.add_argument("--brain", default="/var/lib/lu/brain-v5-20260611-enc")
    parser.add_argument("--keyfile", default="/var/lib/lu/master.key")
    parser.add_argument("--ingest-only", action="store_true", default=False,
                        help="OP-RETRAIN-1: boot the MCP child lean (--load-mode ingest-only) so a full retrain stays RAM-bounded; post-train serving-residency rebuild restores recall")
    parser.add_argument("--mcp-stderr", default="/var/lib/lu/ingest_full_mcp.err")
    parser.add_argument("--smoke", type=int, default=None,
                        help="Smoke mode: cap each corpus to this many examples")
    parser.add_argument("--tier", action="append", type=int, choices=[1,2], default=None,
                        help="Restrict to listed tier(s); default both")
    parser.add_argument("--summary-out", default="/var/lib/lu/ingest_full_summary.json")
    parser.add_argument("--pipe", default=None,
                        help="Accepted-and-ignored for spec compatibility (subprocess.Popen used)")
    parser.add_argument("--checkpoint", default=CHECKPOINT,
                        help="Override default checkpoint path")
    parser.add_argument("--progress-log", default=PROGRESS_LOG,
                        help="Override default progress-log path")
    parser.add_argument("--arxiv-set", default="cs:cs:AI",
                        help="arxiv OAI-PMH setSpec (default cs:cs:AI)")
    parser.add_argument("--arxiv-from", default="2024-01-01",
                        help="arxiv OAI-PMH from date (default 2024-01-01)")
    parser.add_argument("--arxiv-until", default="2024-06-30",
                        help="arxiv OAI-PMH until date (default 2024-06-30)")
    parser.add_argument("--arxiv-max-papers", type=int, default=5000,
                        help="arxiv max papers (default 5000; smoke overrides)")
    parser.add_argument("--se-xml-paths", action="append", default=None,
                        help="StackExchange Posts.xml path(s). Repeatable. Defaults to glob /tmp/se/*/Posts.xml.")
    parser.add_argument("--se-min-score", type=int, default=5,
                        help="StackExchange answer min score (default 5)")
    parser.add_argument("--coding-sources", default="code_search_net,stackoverflow",
                        help="Comma-list of Phase-1/2 coding sources to run "
                             "(code_search_net, stackoverflow, python_docs, "
                             "mdn). Default code_search_net,stackoverflow. "
                             "Use to run a single source for smoke tests.")
    parser.add_argument("--pydocs-root",
                        default="/var/lib/lu/data/pydocs/python-3.14-docs-text",
                        help="Root of the extracted python.org plain-text "
                             "docs bundle (library/, reference/ subtrees).")
    parser.add_argument("--mdn-root",
                        default="/var/lib/lu/data/mdn/content",
                        help="Root of the cloned mdn/content repo "
                             "(files/en-us/web/** reference trees).")
    parser.add_argument("--pycode-root", default=None,
                        help="Root dir (or single .py) of Python SOURCE to "
                             "structurally extract (Phase-1). Default: stdlib "
                             "/usr/lib/python3.12.")
    parser.add_argument("--stackv2-source", default="local",
                        choices=["local", "hf"],
                        help="The Stack v2 source: 'local' mixed-language tree "
                             "(bounded validation) or 'hf' (GATED: HF token + "
                             "AWS creds for Software Heritage S3 content).")
    parser.add_argument("--stackv2-root", default=None,
                        help="Root dir of local mixed-language source "
                             "(--stackv2-source local).")
    parser.add_argument("--stackv2-langs", default=None,
                        help="Comma extractor-lang list for --stackv2-source hf "
                             "(default: all supported).")
    parser.add_argument("--stackv2-limit", type=int, default=None,
                        help="Cap files processed for stack_v2 (bounded run).")
    parser.add_argument("--so-min-score", type=int, default=5,
                        help="StackOverflow answer min score (default 5)")
    parser.add_argument("--coding-only", action="store_true", default=False,
                        help="Run ONLY the Phase-1 coding sources within tier 1 "
                             "(skip visual_genome/wikipedia/arxiv/stackexchange). "
                             "For independent coding-curriculum smoke tests.")
    parser.add_argument("--grok-seeds", action="append", default=None,
                        help="Grokipedia explicit seed slugs. Repeatable. Smoke mode auto-fills from loader DEFAULT_SEEDS if empty.")
    parser.add_argument("--grok-sitemap", default=None,
                        help="Override Grokipedia sitemap index URL.")
    parser.add_argument("--seed-trust", action="store_true", default=False,
                        help="BR.10b: before ingest, invoke "
                             "lu4d_trust_reset_to_defaults with "
                             "seed_curriculum_corpora=true so corpus_multipliers "
                             "is non-empty (matches TrustTable seed keys: "
                             "wikimedia/wikipedia=1.0, ranjaykrishna/visual_genome=1.0, "
                             "GQA=0.9, arxiv=0.9, etc.). Without this flag, every "
                             "corpus resolves to default_multiplier=0.4 — only the "
                             "3 C2-aligned domain entries (grokipedia/wikipedia/duckduckgo) "
                             "are seeded by TrustTable::default(). Required for the "
                             "BR.10b retrain bar.")
    args = parser.parse_args()
    args.coding_sources = {
        x.strip() for x in (args.coding_sources or "").split(",") if x.strip()
    }

    # Smoke-vs-production checkpoint isolation (Cat.1-retrain post-mortem):
    # When --smoke is set, automatically route checkpoint and progress-log
    # to *_smoke siblings so smoke verification cannot pollute the production
    # state. The 2026-05-04 retrain shipped with smoke + full sharing the
    # default checkpoint, leaving stale {last_saved_id: 49, completed: true}
    # markers from Phase 2 visible to Phase 3.
    smoke_redirect_msg = None
    if args.smoke is not None:
        def _smoke_path(p):
            base, dot, ext = p.rpartition(".")
            return f"{base}_smoke.{ext}" if dot else f"{p}_smoke"
        orig_cp = args.checkpoint
        orig_pl = args.progress_log
        orig_su = args.summary_out
        orig_ds = DONE_SENTINEL
        args.checkpoint = _smoke_path(args.checkpoint)
        args.progress_log = _smoke_path(args.progress_log)
        args.summary_out = _smoke_path(args.summary_out)
        DONE_SENTINEL = _smoke_path(DONE_SENTINEL)
        smoke_redirect_msg = (
            f"smoke mode (--smoke={args.smoke}): rerouted "
            f"checkpoint {orig_cp} -> {args.checkpoint}, "
            f"progress {orig_pl} -> {args.progress_log}, "
            f"summary {orig_su} -> {args.summary_out}, "
            f"sentinel {orig_ds} -> {DONE_SENTINEL}"
        )

    PROGRESS_LOG = args.progress_log
    CHECKPOINT = args.checkpoint

    tiers = set(args.tier or [1, 2])
    log_progress(f"=== INGEST FULL START brain={args.brain} smoke={args.smoke} tiers={sorted(tiers)} ===")
    if smoke_redirect_msg:
        log_progress(smoke_redirect_msg)

    cp = load_checkpoint()
    log_progress(f"checkpoint: {json.dumps({k:v for k,v in cp.items() if k!='updated_at'})[:600]}")

    summary = {"started_at": datetime.now(timezone.utc).isoformat(),
               "brain": args.brain, "smoke": args.smoke,
               "run_timestamp": RUN_TIMESTAMP, "tiers": sorted(tiers)}

    client = McpClient(args.brain, args.keyfile, log_stderr_to=args.mcp_stderr, ingest_only=args.ingest_only)
    try:
        # Status before
        resp = client.call("lu4d_status", {})
        sc = resp.get("result", {}).get("structuredContent", {}) if call_ok(resp) else {}
        summary["bond_count_before"] = sc.get("bond_count", -1)
        summary["node_count_before"] = sc.get("node_count", -1)
        log_progress(f"status_before: bonds={summary['bond_count_before']} nodes={summary['node_count_before']}")

        # Cat.1-retrain perf tweaks internal notes
        # Hub-cap perf mitigation for Wikipedia-scale ingest:
        # default 5000 caps cause O(degree) eviction lookups on hub nodes
        # (year, person, place, ...) once at cap. Raised to 50000 just for
        # the retrain run; re-tighten via runtime config post-retrain.
        # NOTE: lu4d_configure does not accept hub_* fields in this binary;
        # the dedicated tool is lu4d_hub_set (one key/value per call).
        for k, v in (("max_in_degree", 50000), ("max_out_degree", 50000)):
            r = client.call("lu4d_hub_set", {"key": k, "value": v})
            log_progress(f"startup tweak: lu4d_hub_set {k}={v} ok={call_ok(r)}")
        # Pause T.6 auto-tick for the retrain duration so curiosity research
        # does not compete with ingest cycles for the MCP server.
        # Resume manually post-retrain with lu4d_auto_tick_resume.
        r = client.call("lu4d_auto_tick_pause", {})
        log_progress(f"startup tweak: lu4d_auto_tick_pause ok={call_ok(r)}")

        # BR.10b: trust seed pre-retrain wiring. When --seed-trust
        # is set (operator opt-in for production retrain), call
        # lu4d_trust_reset_to_defaults with seed_curriculum_corpora=true
        # so the curriculum trust hierarchy is in place BEFORE any
        # bonds land. Without this, every bond resolves to the 0.4
        # default_multiplier instead of the seeded 1.0/0.9/0.7 values.
        if args.seed_trust:
            r = client.call("lu4d_trust_reset_to_defaults",
                            {"seed_curriculum_corpora": True})
            log_progress(f"startup tweak: lu4d_trust_reset_to_defaults "
                         f"seed_curriculum_corpora=true ok={call_ok(r)}")
            sc = r.get("result", {}).get("structuredContent", {}) if call_ok(r) else {}
            cm = sc.get('corpus_multipliers', {}) or {}
            log_progress(f"trust seed: corpus_multipliers "
                         f"populated_count={len(cm)} "
                         f"keys={sorted(cm.keys())[:3]}...")
        else:
            log_progress("startup tweak: --seed-trust NOT set, "
                         "corpus_multipliers stays empty (every bond "
                         "resolves to default_multiplier=0.4)")

        if 1 in tiers:
            if not args.coding_only:
                # OP-RETRAIN-1 TIER 0 (foundation-first): lexical grounding via
                # WordNet BEFORE Wikipedia -- base vocabulary so everything after
                # bonds onto grounded concepts. Skips gracefully if data absent.
                try:
                    import wordnet_loader as _wn
                    if os.path.isdir(_wn.WN_DIR_DEFAULT):
                        summary["wordnet"] = _wn.ingest_wordnet(client, cp, args.smoke)
                    else:
                        log_progress(f"wordnet: SKIPPED -- prolog dir not found: {_wn.WN_DIR_DEFAULT}")
                        summary["wordnet"] = {"skipped_reason": "wordnet prolog dir absent"}
                except Exception as _e:
                    log_progress(f"wordnet: ERROR {type(_e).__name__}: {str(_e)[:200]}")
                    summary["wordnet"] = {"error": str(_e)[:200]}
                # OP-RETRAIN-1 TIER 0b: dictionary (GCIDE/Webster 1913) definitions
                # + structured genus IsA, after WordNet, before Wikipedia.
                try:
                    import dictionary_loader as _dict
                    if os.path.isdir(_dict.GCIDE_DIR_DEFAULT):
                        summary["gcide"] = _dict.ingest_dictionary(client, cp, args.smoke)
                    else:
                        log_progress(f"gcide: SKIPPED -- dir not found: {_dict.GCIDE_DIR_DEFAULT}")
                        summary["gcide"] = {"skipped_reason": "gcide dir absent"}
                except Exception as _e:
                    log_progress(f"gcide: ERROR {type(_e).__name__}: {str(_e)[:200]}")
                    summary["gcide"] = {"error": str(_e)[:200]}
                summary["visual_genome"] = ingest_visual_genome(client, cp, args.smoke)
                summary["wikipedia"] = ingest_wikipedia(client, cp, args.smoke)
                summary["arxiv"] = ingest_arxiv(
                    client, cp, args.smoke,
                    set_spec=args.arxiv_set,
                    frm=args.arxiv_from,
                    until=args.arxiv_until,
                    max_papers=args.arxiv_max_papers,
                )
                summary["stackexchange"] = ingest_stackexchange(
                    client, cp, args.smoke,
                    xml_paths=args.se_xml_paths,
                    min_score=args.se_min_score,
                )
            # Phase-1 coding curriculum (declarative coding knowledge).
            # Gated by --coding-sources so they can run independently for
            # the smoke test without dragging in wiki/arxiv/se.
            if "code_search_net" in args.coding_sources:
                summary["code_search_net"] = ingest_code_search_net(
                    client, cp, args.smoke)
            if "stackoverflow" in args.coding_sources:
                summary["stackoverflow"] = ingest_stackoverflow(
                    client, cp, args.smoke, min_score=args.so_min_score)
            if "python_docs" in args.coding_sources:
                summary["python_docs"] = ingest_python_docs(
                    client, cp, args.smoke, pydocs_root=args.pydocs_root)
            if "mdn" in args.coding_sources:
                summary["mdn"] = ingest_mdn(
                    client, cp, args.smoke, mdn_root=args.mdn_root)
            if "python_code" in args.coding_sources:
                summary["python_code"] = ingest_python_code(
                    client, cp, args.smoke, pycode_root=args.pycode_root)
            if "stack_v2" in args.coding_sources:
                summary["stack_v2"] = ingest_stack_v2(
                    client, cp, args.smoke, source=args.stackv2_source,
                    root=args.stackv2_root, langs=args.stackv2_langs,
                    limit=args.stackv2_limit)
            # --- grokopedia SKIPPED by operator (John) 2026-06-14 ---
            # grokopedia_loader.py imports bs4 (BeautifulSoup), which is not
            # installed on this box. Operator chose to SKIP grokopedia rather
            # than install bs4. This mirrors how visual_genome/stackexchange
            # record a skip. TO RE-ENABLE: `pip install beautifulsoup4` then
            # delete this skip block and uncomment the ingest_grokopedia call.
            log_progress("grokopedia: SKIPPED — deferred by operator "
                         "(bs4 not installed; skip-not-run, 2026-06-14)")
            summary["grokopedia"] = {"skipped_reason":
                "deferred by operator 2026-06-14: bs4 not installed, skip-not-run"}
            # summary["grokopedia"] = ingest_grokopedia(
            #     client, cp, args.smoke,
            #     seeds=args.grok_seeds,
            #     sitemap_index_url=args.grok_sitemap,
            # )
        if 2 in tiers and not args.coding_only:
            summary["no_robots"] = ingest_no_robots(client, cp, args.smoke)
            summary["instruction_dataset"] = ingest_instruction_dataset(client, cp, args.smoke)

        # final save
        log_progress("final lu4d_save")
        save_resp = client.call("lu4d_save", {})
        summary["final_save"] = save_resp.get("result", save_resp)

        resp = client.call("lu4d_status", {})
        sc = resp.get("result", {}).get("structuredContent", {}) if call_ok(resp) else {}
        summary["bond_count_after"] = sc.get("bond_count", -1)
        summary["node_count_after"] = sc.get("node_count", -1)
        log_progress(f"status_after: bonds={summary['bond_count_after']} nodes={summary['node_count_after']}")

        # success sentinel for the validator task
        with open(DONE_SENTINEL, "w") as f:
            json.dump({"completed_at": datetime.now(timezone.utc).isoformat(),
                       "bond_count_after": summary["bond_count_after"]}, f)
        log_progress(f"=== DONE — sentinel written to {DONE_SENTINEL} ===")
    finally:
        summary["ended_at"] = datetime.now(timezone.utc).isoformat()
        try:
            with open(args.summary_out, "w") as f:
                json.dump(summary, f, indent=2, default=str)
        except Exception as e:
            log_progress(f"summary write failed: {e}")
        client.close()

if __name__ == "__main__":
    main()
