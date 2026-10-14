#!/usr/bin/env python3
"""
scripts/grokopedia_loader.py -- Cat.1 retrain Tier 1 declarative loader (3/3).

Pulls articles from grokipedia.com (verified TLD; brief said "grokopedia",
the actual domain is grokipedia.com -- single 'i'), strips infoboxes /
refs / citation markers / navigation, and runs them through the existing
encyclopedia-shaped extractor so that bonds land on Right Brain via the
same asymmetry-aware path Wikipedia uses.

DESIGN AT A GLANCE

  sitemap-index.xml  --(rate-limited HTTP)-->  per-sitemap urlset XML
        |
        v
   /page/<slug>  HTML  --(BeautifulSoup; span.block paragraphs)-->
        |
        v
   {title, text, url} dict  -- smart_extractor.extract_bonds_from_wikipedia_article -->
        |
        v
   list[bond]  -- BatchSink (lu4d_teach_corpus_batch over MCP) -->  brain

The loader has TWO surfaces:

  1. Module API for the larger curriculum harness:
       stream_articles(seeds=..., limit=..., progress_path=...) -> Iterator[dict]
     Yields {'title','text','url','slug'} dicts compatible with
     smart_extractor.extract_bonds_from_wikipedia_article.

  2. CLI entry point for stand-alone smoke / acceptance runs:
       python3 scripts/grokopedia_loader.py --limit 100 --no-save
     Spawns lu_4_mcp --mcp --no-save (canonical brain by default,
     production bytes UNCHANGED), batches bonds via
     lu4d_teach_corpus_batch, prints a smoke summary.

ACCESS PATH (verified 2026-05-10):
  - https://grokipedia.com/robots.txt -> User-agent: *  Disallow: /api/
    Sitemap: https://assets.grokipedia.com/sitemap/sitemap-index.xml
  - sitemap-index lists ~tens of sub-sitemaps, each lists
    https://grokipedia.com/page/<slug> entries.
  - article HTML is server-rendered (no __NEXT_DATA__ hydration); article
    body is carried in <span class="... block ..."> paragraph carriers
    inside <article>. Plain HTTP + HTML parsing is sufficient. NO
    headless browser hack is required (per brief, that would have been
    surfaced as a complication, not papered over).
  - JSON-LD <script type="application/ld+json"> in <head> exposes
    {"@type":"Article","headline":...} which we use as the canonical
    title (preserves disambiguators like "X (philosopher)" cleanly).

ASYMMETRY DISCIPLINE (brief Step 3):
  Encyclopedia content is declarative. We REUSE
  scripts/smart_extractor.py:extract_bonds_from_wikipedia_article verbatim
  -- six-layer subject-anchored extraction (see-also, section, category,
  lead-pattern, body-comention, numerical-attr). The asymmetry boundary
  is enforced by smart_extractor's L3 subject-anchoring + the lead
  determiner-led / capitalised-leading NP patterns. We do NOT define a
  parallel extractor here; that would diverge.

  Editorial-frame leakage ("according to", "as noted", "see also",
  "this article", "as discussed") is filtered both by smart_extractor
  (subject-anchoring + stop-noun blacklists) and, defense in depth,
  by `_subject_has_editorial_frame` here at the bond-emit boundary.

ROBUSTNESS:
  - Throttle: 1 req / 2.5 s (HTTP_THROTTLE_HZ = 0.4); brief allowed
    1 req / 2-3 s.
  - 429: respects Retry-After header; falls back to capped exponential
    backoff if header absent. Hard halt after MAX_429_RETRIES.
  - Resume: ingested URLs persisted to ~/.lu/grokopedia_progress.jsonl
    (one JSON line per ingested URL with {url,slug,bonds_landed,ts}).
    Re-running the loader skips URLs already present in the resume log.
  - Dedup: URL-level (resume log) AND triple-level (BatchSink seen-set
    inherited from harness pattern).
  - User-Agent: "LuTrainer/1.0 (Cat.1 retrain; dialectforge)".

SOURCE ATTRIBUTION:
  corpus = "grokipedia" -- matches the C2 seed-domain key in
  TrustTable::default() (grokipedia / wikipedia / duckduckgo). The
  brief used the spelling "grokopedia"; aligning with the existing
  TrustTable seed key is the right call so trust multipliers resolve
  on day one (otherwise every grokopedia bond falls back to
  default_multiplier=0.4). Note recorded in the artifacts file.

  Per-bond stamping at the BatchSink call-site (mirrors the
  ingest_curriculum_full.py CORPUS ATTRIBUTION discipline):
     corpus     = "grokipedia"
     source_url = full https://grokipedia.com/page/<slug> URL
     run_id     = grokopedia-loader-<UTC timestamp>
     subject/relation/object/weight/confidence per smart_extractor output
  source_id (== slug) is encoded inside source_url; the brain stores
  the full URL on TrainingCorpus, so the slug is recoverable.

NON-GOALS:
  - We do NOT rewrite McpClient/BatchSink. We import them from
    ingest_curriculum_full so the wire-format invariants stay one
    file deep.
  - We do NOT emit bonds with a corpus other than "grokipedia".
  - We do NOT save the brain unless the operator explicitly passes
    --persist (default is --no-save semantics: production bytes
    untouched). Even with --persist, the BR.10b operator gate at
    /var/lib/lu/master.key is required by lu_4_mcp.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Iterator, List, Optional, Tuple
from xml.etree import ElementTree as ET

# Make sibling scripts/ importable when run from any cwd.
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from smart_extractor import extract_bonds_from_wikipedia_article  # noqa: E402
from ingest_curriculum_full import extract_response_bonds_multi_sentence  # noqa: E402

# bs4 is the only third-party HTML dep. It is already installed at
# /opt/lu/.local/lib/python3.12/site-packages/bs4 on the
# canonical Hetzner box (installed for this loader; pinned in the
# artifact file so a fresh checkout knows to install it).
from bs4 import BeautifulSoup  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GROKIPEDIA_BASE = "https://grokipedia.com"
SITEMAP_INDEX_URL = "https://assets.grokipedia.com/sitemap/sitemap-index.xml"
ROBOTS_URL = f"{GROKIPEDIA_BASE}/robots.txt"
USER_AGENT = "LuTrainer/1.0 (Cat.1 retrain; dialectforge)"

# Throttle: brief asks for 1 req / 2-3 sec at minimum.
HTTP_THROTTLE_HZ = 0.4   # 1 request every 2.5 seconds.
MAX_429_RETRIES = 5      # after this many consecutive 429s on one URL, halt.
DEFAULT_BACKOFF = 30.0   # seconds, when no Retry-After header.

DEFAULT_PROGRESS_PATH = os.path.expanduser("~/.lu/grokopedia_progress.jsonl")
DEFAULT_BRAIN_PATH = "/var/lib/lu/brain.lbr"
DEFAULT_KEYFILE = "/var/lib/lu/master.key"
DEFAULT_LU_BIN = "/opt/lu/lu/merge-main/target/release/lu4d"

CORPUS_ID = "grokipedia"  # matches TrustTable::default() seed key.

# Tags that are noise (scripts, nav, footer, infobox containers, etc.).
# `figure`/`figcaption` are stripped because Grokipedia infoboxes render as
# floated figures; their content (Born / Died / Spouse / etc.) is metadata
# the brief explicitly marks as non-bond.
_DROP_TAGS = (
    "script style noscript svg nav header footer form aside button "
    "iframe figure figcaption template link meta"
).split()

# Section headings that mark the editorial-tail of an encyclopedia article.
# Anything from these onward is metadata, not bond-yielding prose.
_TAIL_HEADINGS = {
    "see also", "references", "external links", "notes",
    "bibliography", "further reading", "sources", "footnotes",
    "citations", "works cited", "selected works",
}

# Belt-and-suspenders editorial-frame token list.
# smart_extractor's lead-pattern subject anchoring already drops most of
# these structurally (a sentence starting with "According to" doesn't
# match Pattern A or Pattern B). This second filter runs at the bond-
# emit boundary so the gate-a invariant is provable from THIS file alone.
_EDITORIAL_FRAME_TOKENS = (
    "according to", "as noted", "see also", "this article",
    "as discussed", "as mentioned", "as described", "as shown",
    "in this article", "the article", "this section",
)

# Citation marker pattern: [1], [12], [ 2 ], etc.
_CITATION_RX = re.compile(r"\[\s*\d+\s*\]")

# Sitemap XML namespace.
_SM_NS = "{http://www.sitemaps.org/schemas/sitemap/0.9}"

# ---------------------------------------------------------------------------
# HTTP layer
# ---------------------------------------------------------------------------


class HttpThrottle:
    """Simple wall-clock throttle. Mirrors ingest_curriculum_full.Throttle."""

    def __init__(self, rate_hz: float):
        self.min_interval = 1.0 / rate_hz
        self.last = 0.0

    def wait(self) -> None:
        now = time.monotonic()
        sleep_for = self.last + self.min_interval - now
        if sleep_for > 0:
            time.sleep(sleep_for)
        self.last = time.monotonic()


def _http_get(url: str, throttle: HttpThrottle, accept: str = "*/*",
              timeout: float = 30.0) -> Tuple[int, bytes, dict]:
    """Throttle, then fetch URL. Returns (status, body, headers_dict).

    On 429, honors Retry-After (seconds-int form OR HTTP-date is decoded
    as integer-seconds). Raises after MAX_429_RETRIES consecutive 429s.
    Other transport errors propagate.
    """
    retries_429 = 0
    while True:
        throttle.wait()
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": USER_AGENT,
                "Accept": accept,
                "Accept-Encoding": "identity",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read()
                return resp.status, body, dict(resp.headers)
        except urllib.error.HTTPError as e:
            status = e.code
            headers = dict(e.headers or {})
            if status == 429:
                retries_429 += 1
                if retries_429 > MAX_429_RETRIES:
                    raise RuntimeError(
                        f"grokipedia: {MAX_429_RETRIES} consecutive 429s on "
                        f"{url} -- throttle is wrong; halting per brief gate g"
                    )
                ra = headers.get("Retry-After") or headers.get("retry-after")
                wait = DEFAULT_BACKOFF
                if ra and ra.isdigit():
                    wait = float(ra)
                wait = min(wait * (2 ** (retries_429 - 1)), 600.0)
                print(f"[grokipedia] 429 on {url}; sleeping {wait:.1f}s "
                      f"(retry {retries_429}/{MAX_429_RETRIES})",
                      file=sys.stderr, flush=True)
                time.sleep(wait)
                continue
            # 4xx/5xx other than 429: propagate the error so the loader
            # logs and skips this single URL rather than masking failures.
            raise


# ---------------------------------------------------------------------------
# Sitemap discovery
# ---------------------------------------------------------------------------


def _parse_sitemap_index(xml_bytes: bytes) -> List[str]:
    """Extract <sitemap><loc> URLs from a sitemap index XML."""
    root = ET.fromstring(xml_bytes)
    return [loc.text.strip() for loc in root.findall(f"{_SM_NS}sitemap/{_SM_NS}loc") if loc.text]


def _parse_url_sitemap(xml_bytes: bytes) -> List[str]:
    """Extract <url><loc> URLs from a urlset sitemap XML."""
    root = ET.fromstring(xml_bytes)
    return [loc.text.strip() for loc in root.findall(f"{_SM_NS}url/{_SM_NS}loc") if loc.text]


def enumerate_article_urls(throttle: HttpThrottle,
                            sitemap_index_url: str = SITEMAP_INDEX_URL,
                            max_urls: Optional[int] = None,
                            sitemap_skip: int = 0) -> Iterator[str]:
    """Iterate /page/<slug> URLs by walking the sitemap index.

    Streams: yields URLs as we walk so callers can stop after `max_urls`
    without us pre-loading every sub-sitemap.
    """
    status, body, _ = _http_get(sitemap_index_url, throttle,
                                 accept="application/xml")
    if status != 200:
        raise RuntimeError(f"sitemap-index fetch returned HTTP {status}")
    sub_sitemaps = _parse_sitemap_index(body)
    if sitemap_skip:
        sub_sitemaps = sub_sitemaps[sitemap_skip:]
    yielded = 0
    for sm_url in sub_sitemaps:
        try:
            s, b, _ = _http_get(sm_url, throttle, accept="application/xml")
            if s != 200:
                print(f"[grokipedia] sub-sitemap HTTP {s}: {sm_url}",
                      file=sys.stderr, flush=True)
                continue
            urls = _parse_url_sitemap(b)
        except Exception as e:
            print(f"[grokipedia] sub-sitemap parse failed {sm_url}: {e}",
                  file=sys.stderr, flush=True)
            continue
        for u in urls:
            yield u
            yielded += 1
            if max_urls is not None and yielded >= max_urls:
                return


# ---------------------------------------------------------------------------
# HTML -> {title, text, url} extraction
# ---------------------------------------------------------------------------


def _extract_jsonld_article_title(soup: BeautifulSoup) -> Optional[str]:
    """Pull canonical title from JSON-LD <script type=application/ld+json>.

    Preferred over <h1>/<title> because it preserves disambiguators
    ("X (philosopher)") and is what the site itself treats as canonical.
    """
    for s in soup.find_all("script", attrs={"type": "application/ld+json"}):
        try:
            data = json.loads(s.string or "{}")
        except Exception:
            continue
        # JSON-LD blocks may be a single dict or a list.
        candidates = data if isinstance(data, list) else [data]
        for c in candidates:
            if isinstance(c, dict) and c.get("@type") == "Article":
                hl = (c.get("headline") or "").strip()
                if hl:
                    return hl
    return None


def _strip_tail_sections(text: str) -> str:
    """Drop everything from the first See-Also / References / etc. heading on."""
    out = []
    for para in text.split("\n\n"):
        head = para.strip().lower().rstrip(":").strip()
        if head in _TAIL_HEADINGS:
            break
        out.append(para)
    return "\n\n".join(out)


def html_to_article(html: str, url: str) -> Optional[dict]:
    """Convert a Grokipedia article HTML to a {title,text,url,slug} dict.

    Returns None if the page has no parseable article body (404 noise,
    placeholder pages, etc.). The returned dict is shaped exactly as
    smart_extractor.extract_bonds_from_wikipedia_article expects:
    {'title': str, 'text': str, 'url': str}.
    """
    # Two parses: one for JSON-LD (must run BEFORE we strip <script>),
    # one for body extraction (after .decompose() on noise nodes).
    soup_meta = BeautifulSoup(html, "html.parser")
    title = _extract_jsonld_article_title(soup_meta)

    soup = BeautifulSoup(html, "html.parser")
    for sel in _DROP_TAGS:
        for n in soup.select(sel):
            n.decompose()

    root = soup.find("article") or soup.find("main") or soup.body
    if root is None:
        return None

    if not title:
        # Fallback: <h1> inside <article>, then <title> tag.
        h1 = root.find("h1")
        if h1:
            title = h1.get_text(" ", strip=True)
        else:
            t = soup_meta.find("title")
            if t:
                title = re.sub(r"\s+\u2014\s+Grokipedia\s*$", "",
                                t.get_text(" ", strip=True))

    if not title:
        return None

    # Paragraph carriers on Grokipedia: <p>, <h1>-<h6>, <li>, <blockquote>,
    # <pre>, AND <span class="... block ..."> (Grokipedia's primary body
    # paragraph element -- discovered via DOM probe). We walk in document
    # order and dedupe by trimmed text to avoid double-counting nested
    # <span> when an outer <div> already covered it.
    lines: List[str] = []
    seen_text = set()
    for el in root.find_all(["h1", "h2", "h3", "h4", "h5", "h6",
                              "p", "li", "blockquote", "pre", "span"]):
        if el.name == "span":
            cls = el.get("class") or []
            if "block" not in cls:
                continue
        txt = el.get_text(separator=" ", strip=True)
        txt = re.sub(r"\s+", " ", txt).strip()
        if not txt or len(txt) < 3:
            continue
        if txt in seen_text:
            continue
        seen_text.add(txt)
        lines.append(txt)

    text = "\n\n".join(lines)
    # Strip citation markers everywhere (gate b: zero [\d+] in subjects).
    text = _CITATION_RX.sub("", text)
    # Collapse triple+ blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    # Drop See-Also / References / etc. tail sections.
    text = _strip_tail_sections(text)

    if len(text) < 80:
        # Pages with tiny body are usually placeholders / disambiguation
        # stubs; smart_extractor will return [] anyway, but skip up-front.
        return None

    slug = url.rsplit("/page/", 1)[-1] if "/page/" in url else url
    return {"title": title, "text": text, "url": url, "slug": slug}


# ---------------------------------------------------------------------------
# Resume log
# ---------------------------------------------------------------------------


def _load_resume(progress_path: str) -> set:
    """Return the set of URLs already ingested."""
    seen = set()
    if not os.path.exists(progress_path):
        return seen
    with open(progress_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                url = d.get("url")
                if url:
                    seen.add(url)
            except Exception:
                continue
    return seen


def _append_resume(progress_path: str, entry: dict) -> None:
    os.makedirs(os.path.dirname(progress_path) or ".", exist_ok=True)
    # Atomic append: open in append mode then flush.
    with open(progress_path, "a") as f:
        f.write(json.dumps(entry) + "\n")
        f.flush()
        os.fsync(f.fileno())


# ---------------------------------------------------------------------------
# Editorial-frame defense in depth
# ---------------------------------------------------------------------------


def _subject_has_editorial_frame(subject: str) -> bool:
    """Belt + suspenders: returns True if the bond subject contains an
    editorial-frame token. smart_extractor's structural filters already
    block these in the LEAD path; this function is a final guard at the
    BatchSink.add boundary so brief gate (a) is provable from THIS file
    alone ("30-bond subject sample contains zero editorial frames")."""
    s = (subject or "").lower()
    for tok in _EDITORIAL_FRAME_TOKENS:
        if tok in s:
            return True
    return False


# ---------------------------------------------------------------------------
# Module API: stream articles
# ---------------------------------------------------------------------------


def stream_articles(
    seeds: Optional[List[str]] = None,
    sitemap_index_url: str = SITEMAP_INDEX_URL,
    limit: Optional[int] = None,
    sitemap_skip: int = 0,
    progress_path: Optional[str] = DEFAULT_PROGRESS_PATH,
    throttle: Optional[HttpThrottle] = None,
) -> Iterator[dict]:
    """Yield article dicts. Skips URLs already in the resume log.

    `seeds` (optional): explicit /page/<slug> URLs OR bare slugs to fetch
        instead of (or before) walking the sitemap. Useful for the brief's
        "sample 100 articles from across topics" diverse-seed smoke test.
    `sitemap_index_url`: override for testing.
    `limit`: stop after yielding this many parsed-OK articles.
    `sitemap_skip`: skip first N sub-sitemaps in the index (useful when
        re-running and you want fresh URLs from later in the alphabet).
    `progress_path`: jsonl resume log; pass None to disable resume.
    """
    if throttle is None:
        throttle = HttpThrottle(HTTP_THROTTLE_HZ)

    seen_urls: set = _load_resume(progress_path) if progress_path else set()

    yielded = 0

    def _normalize_seed(s: str) -> str:
        s = s.strip()
        if s.startswith("http://") or s.startswith("https://"):
            return s
        if s.startswith("/page/"):
            return f"{GROKIPEDIA_BASE}{s}"
        return f"{GROKIPEDIA_BASE}/page/{s.lstrip('/')}"

    url_iter: Iterator[str]
    if seeds:
        url_iter = (_normalize_seed(s) for s in seeds)
    else:
        url_iter = enumerate_article_urls(
            throttle, sitemap_index_url=sitemap_index_url,
            max_urls=None, sitemap_skip=sitemap_skip,
        )

    for url in url_iter:
        if limit is not None and yielded >= limit:
            return
        if url in seen_urls:
            continue
        try:
            status, body, _ = _http_get(url, throttle, accept="text/html")
        except Exception as e:
            print(f"[grokipedia] HTTP error {url}: {type(e).__name__}: {e}",
                  file=sys.stderr, flush=True)
            continue
        if status != 200:
            print(f"[grokipedia] HTTP {status}: {url}",
                  file=sys.stderr, flush=True)
            continue
        try:
            article = html_to_article(body.decode("utf-8", errors="replace"),
                                       url)
        except Exception as e:
            print(f"[grokipedia] parse error {url}: {type(e).__name__}: {e}",
                  file=sys.stderr, flush=True)
            continue
        if not article:
            continue
        seen_urls.add(url)
        yielded += 1
        yield article


# ---------------------------------------------------------------------------
# Bond emission helpers
# ---------------------------------------------------------------------------


def extract_bonds(article: dict) -> List[dict]:
    """Run smart_extractor on a Grokipedia article dict, then apply the
    belt+suspenders editorial-frame guard. Returns list of bond dicts
    shaped for sink.add (subject, relation, object, weight, confidence).

    Port B (R1, 2026-05-10): unions smart_extractor output with
    extract_response_bonds_multi_sentence (pronoun-anchored / multi-
    sentence Pattern A/B bonds smart_extractor's title-alias filter
    drops). Dedup on (subject.lower(), relation, object.lower()).
    Editorial-frame guard then fires on the unioned set."""
    bonds = extract_bonds_from_wikipedia_article(article)
    try:
        extra_bonds = extract_response_bonds_multi_sentence(
            article.get("title", "") or "",
            article.get("text", "") or "",
            article.get("url", "") or "",
        )
    except Exception:
        extra_bonds = []
    if extra_bonds:
        seen_triples = set(
            ((b.get("subject") or "").lower(),
             b.get("relation") or "",
             (b.get("object") or "").lower())
            for b in bonds
        )
        for eb in extra_bonds:
            key = (eb["subject"].lower(), eb["relation"], eb["object"].lower())
            if key in seen_triples:
                continue
            seen_triples.add(key)
            bonds.append(eb)
    out = []
    dropped = 0
    for b in bonds:
        subj = b.get("subject") or ""
        if _subject_has_editorial_frame(subj):
            dropped += 1
            continue
        out.append(b)
    if dropped:
        print(f"[grokipedia] editorial-frame guard dropped {dropped} bonds "
              f"from {article.get('slug')}", file=sys.stderr, flush=True)
    return out


# ---------------------------------------------------------------------------
# CLI / standalone ingest mode
# ---------------------------------------------------------------------------


DEFAULT_SEEDS = [
    # Diverse seeds spanning philosophy, science, history, technology --
    # used when --seeds isn't provided so the smoke test exercises the
    # full extractor surface.
    "Albert_Einstein", "Isaac_Newton", "Plato", "Aristotle",
    "Socrates", "Immanuel_Kant", "Karl_Marx", "Confucius",
    "Charles_Darwin", "Marie_Curie", "Niels_Bohr", "Richard_Feynman",
    "Alan_Turing", "John_von_Neumann", "Ada_Lovelace", "Grace_Hopper",
    "Roman_Empire", "Byzantine_Empire", "French_Revolution",
    "Industrial_Revolution", "World_War_II", "Cold_War",
    "Mongol_Empire", "Ottoman_Empire", "Renaissance", "Enlightenment",
    "Linux_(kernel)", "Operating_system", "Compiler", "Algorithm",
    "Cryptography", "Public-key_cryptography", "Database",
    "Distributed_computing", "Machine_learning", "Artificial_intelligence",
    "Programming_language", "Functional_programming",
    "Quantum_mechanics", "General_relativity", "Special_relativity",
    "Thermodynamics", "Electromagnetism", "Photosynthesis",
    "DNA", "Cell_(biology)", "Evolution", "Mitosis", "Meiosis",
    "Black_hole", "Big_Bang", "Solar_System", "Milky_Way",
    "Periodic_table", "Carbon", "Hydrogen", "Oxygen",
    "Geology", "Plate_tectonics", "Volcano", "Earthquake",
    "Mathematics", "Calculus", "Geometry", "Algebra",
    "Number_theory", "Topology", "Set_theory", "Group_theory",
    "Graph_theory", "Probability",
    "Economics", "Microeconomics", "Macroeconomics", "Game_theory",
    "Constitutional_law", "Common_law", "Civil_law_(legal_system)",
    "Democracy", "Republic", "Federalism",
    "Christianity", "Buddhism", "Hinduism", "Islam", "Judaism",
    "Existentialism", "Stoicism", "Empiricism", "Rationalism",
    "Symphony", "Opera", "Jazz", "Blues", "Hip_hop_music",
    "Painting", "Sculpture", "Architecture", "Photography",
    "Cinema", "Television", "Internet", "World_Wide_Web",
    "Bitcoin", "Ethereum", "Blockchain",
    "Antarctica", "Mount_Everest", "Amazon_rainforest",
]


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def cli_ingest(args) -> int:
    """Stand-alone ingest mode for the brief's smoke / acceptance run."""
    # We import the harness components lazily so importing this module
    # for its API (without actually running the CLI) doesn't pull in the
    # whole curriculum harness.
    from ingest_curriculum_full import (
        McpClient, BatchSink, Throttle, call_ok,
    )  # noqa: E402

    print(f"[grokipedia] === smoke run start {_now_utc_iso()} ===",
          flush=True)
    print(f"[grokipedia]   brain={args.brain} no_save={args.no_save} "
          f"persist={args.persist} dry_run={args.dry_run}",
          flush=True)

    seeds: Optional[List[str]] = None
    if args.seeds:
        seeds = [s for s in args.seeds.split(",") if s.strip()]
    elif args.use_default_seeds:
        seeds = list(DEFAULT_SEEDS)
    # else: stream straight from the sitemap.

    progress_path = args.progress_log
    if args.no_resume:
        progress_path = None

    http_throttle = HttpThrottle(args.http_rate_hz)

    summary = {
        "started_at": _now_utc_iso(),
        "brain": args.brain,
        "limit": args.limit,
        "http_rate_hz": args.http_rate_hz,
        "corpus": CORPUS_ID,
        "sitemap_skip": args.sitemap_skip,
        "seeds_mode": ("explicit" if args.seeds else
                       "defaults" if args.use_default_seeds else "sitemap"),
        "persist": args.persist,
        "no_save_passthrough": args.no_save,
        "resume_log": progress_path,
    }

    # In dry-run we don't even spawn the MCP daemon -- pure HTML->bonds
    # path so the operator can sanity-check extraction quality without
    # touching brain bytes at all.
    client = None
    sink = None
    if not args.dry_run:
        client = McpClient(args.brain, args.keyfile,
                            log_stderr_to=args.mcp_stderr)
        # Status before -- captured for the acceptance artifact (gate f).
        try:
            r = client.call("lu4d_status", {})
            sc = (r.get("result", {}) or {}).get("structuredContent", {}) \
                if call_ok(r) else {}
            summary["bond_count_before"] = sc.get("bond_count", -1)
            summary["node_count_before"] = sc.get("node_count", -1)
        except Exception as e:
            summary["bond_count_before"] = f"status error: {e}"
        # Mirror the harness pattern: rate-limited BatchSink for batched
        # JSON-RPC submission. Throttle is in-process pacing, separate
        # from HTTP throttle.
        sink = BatchSink(client, Throttle(50.0))  # 50 calls/s in-process

    run_id = f"grokopedia-loader-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    summary["run_id"] = run_id

    articles_processed = 0
    bonds_emitted_total = 0  # pre-sink (post editorial-frame guard).
    parse_skipped = 0
    iter_t0 = time.monotonic()

    try:
        for article in stream_articles(
            seeds=seeds,
            limit=args.limit,
            sitemap_skip=args.sitemap_skip,
            progress_path=progress_path,
            throttle=http_throttle,
        ):
            bonds = extract_bonds(article)
            if not bonds:
                parse_skipped += 1
            articles_processed += 1
            for b in bonds:
                bonds_emitted_total += 1
                if args.dry_run:
                    print(f"  bond: {b['subject']!r} -- {b['relation']} "
                          f"-- {b['object']!r}", flush=True)
                    continue
                try:
                    sink.add({
                        "corpus": CORPUS_ID,
                        "source_url": article["url"],
                        "run_id": run_id,
                        "subject": b["subject"],
                        "relation": b["relation"],
                        "object": b["object"],
                        "weight": b["weight"],
                        "confidence": b["confidence"],
                    })
                except Exception as e:
                    print(f"[grokipedia] sink.add raised: {e}",
                          file=sys.stderr, flush=True)
            # Resume log entry (ONLY after we attempt to add the
            # article's bonds; on crash mid-article we re-extract on
            # restart, which is safe due to brain-side dedup).
            if progress_path:
                _append_resume(progress_path, {
                    "url": article["url"],
                    "slug": article["slug"],
                    "title": article["title"],
                    "bonds_emitted": len(bonds),
                    "ts": _now_utc_iso(),
                })
            if articles_processed % 25 == 0:
                elapsed = time.monotonic() - iter_t0
                rate = articles_processed / elapsed if elapsed > 0 else 0.0
                print(f"[grokipedia] articles={articles_processed} "
                      f"bonds_emitted={bonds_emitted_total} "
                      f"parse_skipped={parse_skipped} "
                      f"rate={rate:.2f}art/s", flush=True)
        if sink is not None:
            sink.flush()
    finally:
        if sink is not None:
            summary["sink_bonds_landed"] = sink.bonds_landed
            summary["sink_bonds_existing"] = sink.bonds_existing
            summary["sink_bonds_gated"] = sink.bonds_gated
            summary["sink_bonds_errors"] = sink.bonds_errors
            summary["sink_bonds_skipped_dedup"] = sink.bonds_skipped_dedup
        summary["articles_processed"] = articles_processed
        summary["bonds_emitted_total"] = bonds_emitted_total
        summary["parse_skipped"] = parse_skipped
        summary["ended_at"] = _now_utc_iso()

        if client is not None:
            try:
                r = client.call("lu4d_status", {})
                sc = (r.get("result", {}) or {}).get("structuredContent", {}) \
                    if call_ok(r) else {}
                summary["bond_count_after"] = sc.get("bond_count", -1)
                summary["node_count_after"] = sc.get("node_count", -1)
                summary["bond_delta"] = (summary["bond_count_after"] -
                                          summary["bond_count_before"]) \
                    if isinstance(summary.get("bond_count_before"), int) \
                    else None
            except Exception as e:
                summary["bond_count_after"] = f"status error: {e}"
            # Persist gate (f): default behavior is --no-save (the daemon
            # was launched with --no-save; we never call lu4d_save). Even
            # an explicit --persist is operator-gated.
            if args.persist:
                print("[grokipedia] --persist set; calling lu4d_save",
                      flush=True)
                try:
                    r = client.call("lu4d_save", {})
                    summary["final_save"] = (r.get("result", r)
                                              if isinstance(r, dict) else r)
                except Exception as e:
                    summary["final_save_error"] = str(e)
            client.close()

        if args.summary_out:
            try:
                with open(args.summary_out, "w") as f:
                    json.dump(summary, f, indent=2, default=str)
                print(f"[grokipedia] summary -> {args.summary_out}",
                      flush=True)
            except Exception as e:
                print(f"[grokipedia] summary write failed: {e}",
                      file=sys.stderr, flush=True)
        print(f"[grokipedia] === DONE {_now_utc_iso()} ===", flush=True)
        print(json.dumps(summary, indent=2, default=str), flush=True)
    return 0


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Cat.1 retrain Tier-1 declarative loader: Grokipedia.",
    )
    ap.add_argument("--brain", default=DEFAULT_BRAIN_PATH,
                    help="Brain file path (default: canonical /var/lib/lu/brain.lbr).")
    ap.add_argument("--keyfile", default=DEFAULT_KEYFILE,
                    help="Encrypted-brain keyfile path.")
    ap.add_argument("--mcp-stderr",
                    default="/var/lib/lu/grokopedia_loader_mcp.err",
                    help="Where to send the lu_4_mcp daemon's stderr.")
    ap.add_argument("--summary-out",
                    default="/var/lib/lu/grokopedia_loader_summary.json",
                    help="JSON summary file (acceptance artifact).")
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap articles to ingest (smoke runs use 100).")
    ap.add_argument("--seeds", type=str, default=None,
                    help="Comma-separated /page/<slug> URLs or bare slugs to "
                         "fetch instead of streaming the sitemap.")
    ap.add_argument("--use-default-seeds", action="store_true",
                    help="Use the embedded diverse-topic seed list "
                         "(philosophy, science, history, tech, etc.) -- the "
                         "brief's recommended smoke seed set.")
    ap.add_argument("--sitemap-skip", type=int, default=0,
                    help="Skip first N sub-sitemaps in the index.")
    ap.add_argument("--http-rate-hz", type=float, default=HTTP_THROTTLE_HZ,
                    help="Outbound HTTP rate cap, requests per second "
                         f"(default {HTTP_THROTTLE_HZ}; brief asks 0.33-0.5).")
    ap.add_argument("--progress-log", default=DEFAULT_PROGRESS_PATH,
                    help="Resume jsonl path.")
    ap.add_argument("--no-resume", action="store_true",
                    help="Disable resume-skip (useful for smoke re-runs).")
    ap.add_argument("--no-save", action="store_true", default=True,
                    help="Compatibility flag (DEFAULT TRUE): the lu_4_mcp "
                         "daemon is always spawned with --no-save by the "
                         "underlying McpClient. Use --persist to override.")
    ap.add_argument("--persist", action="store_true",
                    help="Operator opt-in: call lu4d_save at the end of the "
                         "run. Default off so canonical brain bytes are "
                         "unchanged (gate f).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print extracted bonds; do NOT spawn the MCP "
                         "daemon at all. Useful for quick HTML-extraction "
                         "validation.")
    return ap


def main(argv=None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    return cli_ingest(args)


if __name__ == "__main__":
    raise SystemExit(main())
