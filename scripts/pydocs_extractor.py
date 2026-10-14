"""Phase-2a: Python official documentation -> declarative typed bonds.

Self-contained extractor for the python.org plain-text docs bundle
(library/*.txt, reference/*.txt). Mirrors the DECLARATIVE summary-bond
discipline proven for CodeSearchNet (scripts/ingest_curriculum_full.py
_csn_summary_bond): one CLEAN typed bond per concept, verbalizing as
[subject] <connector> [object] via narration.rs. No _ingest_tier2 path.

Design notes -- why this is structural, not a hack:
  * Python docs text has a regular shape: a definition line at column 0
    (`class X(...)`, `name(args)`, or a module-header `"mod" --- blurb`),
    followed by indented descriptive prose. We parse into
    (anchor, summary-prose) blocks and run the CSN-style summary patterns.
  * The prose is clean, definitional English (Wikipedia-class), so the
    same def-pattern / action-verb machinery clears the quality bar with
    LESS junk than docstrings did -- but we keep every guard.
  * STRIPPED: signatures, parameter/returns/raises directive lines,
    `>>>` doctests, deprecation/version directives, reST artifacts,
    ASCII tables, section underlines, See-also/footnote blocks.

The module is import-safe with zero deps so it can be unit-smoke-tested
offline (no brain) and then imported by ingest_curriculum_full.py.
"""
import os
import re

# ----------------------------------------------------------------------
# Anchor construction
# ----------------------------------------------------------------------

# A column-0 `class fully.qualified.Name(sig)` line.
_DEF_CLASS_RE = re.compile(r"^class\s+([A-Za-z_][\w.]*)\s*[\(:]")
# A column-0 `module.func(...)` or `func(...)` line (function/method def).
# Requires an opening paren so we don't grab prose. Allows leading
# `@decorator` lines to be skipped by the caller.
_DEF_FUNC_RE = re.compile(r"^([A-Za-z_][\w.]*)\s*\(")
# A column-0 `name = value` data/attribute def is intentionally NOT mined
# (rarely declarative). A bare `module.attr` (no paren) def line:
_DEF_ATTR_RE = re.compile(r"^([a-z_][\w.]*\.[A-Za-z_]\w*)\s*$")

# Module header: `"datetime" --- Basic date and time types`
_MODULE_HDR_RE = re.compile(r'^"([a-z_][\w.]*)"\s+---\s+(.+?)\s*$')

# A reST text-render section underline (===, ---, ***, ^^^, ~~~, +++).
_UNDERLINE_RE = re.compile(r"^[=\-*^~+#\"']{3,}\s*$")
# An ASCII table border / row.
_TABLE_RE = re.compile(r"^\s*[+|]")
# A directive-ish body line we drop (parameter / returns / raises lists,
# version / deprecation directives, see-also, footnotes).
_BODY_DROP_RE = re.compile(
    r"^\s*(>>>|\.\.\.|Parameters?:|Returns?:|Return value:|Raises?:|Yields?:"
    r"|Arguments?:|Keyword Arguments?:|Variables?:|Attributes?:|Example[s]?:"
    r"|Note:|Warning:|Caution:|Important:|Tip:|See also:|Changed in version"
    r"|Deprecated since|New in version|Added in version|Availability:"
    r"|CPython implementation detail:|Footnotes:|Source code:)",
    re.IGNORECASE,
)
# Inline literal markup: text-render wraps code/identifiers in "double
# quotes". We KEEP the inner token (so `"dict"` -> dict) for readability.
_INLINE_LIT_RE = re.compile(r'"([^"\n]{1,60})"')
# reST emphasis markers *word* / **word** -> word.
_EMPH_RE = re.compile(r"\*{1,2}([^*\n]+?)\*{1,2}")
# Footnote ref [1] / [2] etc.
_FOOTNOTE_REF_RE = re.compile(r"\[\d+\]")

# Words that are clearly code-shape and disqualify an anchor.
_CODE_SHAPE_TOKENS = (
    "def ", "lambda ", "import ", "->", "=>", "::", "();", ");",
    "&&", "||", "===", "!==", "self.", "cls.",
)


def _humanize_dotted(name):
    """`collections.OrderedDict` -> `OrderedDict`; `os.path.join` ->
    `path join`. Keep the trailing component(s) past the top module so a
    submodule path stays informative but the leading package is dropped
    for readability. snake_case -> spaced, camelCase preserved (class
    names read fine as-is)."""
    name = name.strip()
    parts = name.split(".")
    # For dotted callables keep the last component (the actual name);
    # for module.submodule headers keep the last too.
    leaf = parts[-1]
    leaf = leaf.lstrip("_")
    if not leaf:
        return None
    spaced = leaf.replace("_", " ").strip()
    spaced = re.sub(r"\s+", " ", spaced)
    if len(spaced) < 2:
        return None
    return spaced


def _strip_inline_markup(text):
    """Remove text-render markup: "lit" -> lit, *emph* -> emph, footnote
    refs, collapse whitespace."""
    text = _FOOTNOTE_REF_RE.sub(" ", text)
    text = _INLINE_LIT_RE.sub(r"\1", text)
    text = _EMPH_RE.sub(r"\1", text)
    text = text.replace("``", "").replace("`", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ----------------------------------------------------------------------
# Summary-bond machinery (ported + adapted from _csn_summary_bond)
# ----------------------------------------------------------------------

_LEAD_DET_RE = re.compile(r"^(?:the|a|an)\s+", re.IGNORECASE)

_IRREGULAR_GERUND = {
    "set": "setting", "get": "getting", "put": "putting", "run": "running",
    "stop": "stopping", "begin": "beginning", "split": "splitting",
    "map": "mapping", "strip": "stripping", "drop": "dropping",
    "trim": "trimming", "wrap": "wrapping", "submit": "submitting",
    "commit": "committing", "emit": "emitting", "plot": "plotting",
    "fit": "fitting", "format": "formatting", "control": "controlling",
}


def _gerund(verb):
    v = verb.lower()
    if v in _IRREGULAR_GERUND:
        return _IRREGULAR_GERUND[v]
    if v.endswith("ie"):
        return v[:-2] + "ying"
    if v.endswith("e") and not v.endswith("ee") and len(v) > 2:
        return v[:-1] + "ing"
    return v + "ing"


_OBJ_BAD_RE = re.compile(r"[<>{}\[\]|`=#]|https?://|::|->|=>|\(\)|\bself\b|\bcls\b")
_OBJ_WORD_RE = re.compile(r"[A-Za-z]")
_OBJ_MAX_WORDS = 14

# Definitional openers on the first sentence -> (relation, captured-obj).
_DEF_PATTERNS = [
    # "Return(s)/Produce(s) a sorted copy ..." -> Produces
    (re.compile(r"^returns?\s+(.*)", re.IGNORECASE), "Produces"),
    (re.compile(r"^produces?\s+(.*)", re.IGNORECASE), "Produces"),
    # "Represent(s)/Is/Are a X" -> IsA
    (re.compile(r"^represents?\s+(.*)", re.IGNORECASE), "IsA"),
    (re.compile(r"^(?:is|are)\s+(an?\s+.*)", re.IGNORECASE), "IsA"),
    # bare "A X that ..." / "An X ..." -> IsA (keep determiner)
    (re.compile(r"^(an?\s+.*)", re.IGNORECASE), "IsA"),
    # "A subclass of X" handled by the determiner pattern above (-> IsA).
    # "Class/Helper/Wrapper/Decorator/Context manager for/that X" -> UsedFor
    (re.compile(r"^(?:a\s+)?(?:class|subclass|helper|wrapper|decorator|"
                r"utility|mixin|context manager|factory|function|method)\s+"
                r"(?:for|that|to|which)\s+(.*)", re.IGNORECASE), "UsedFor"),
]

_ACTION_VERBS = {
    "compute", "calculate", "create", "build", "generate", "make",
    "convert", "parse", "format", "encode", "decode", "serialize",
    "deserialize", "load", "save", "read", "write", "fetch", "send",
    "render", "validate", "check", "verify", "extract", "filter",
    "sort", "merge", "split", "join", "map", "find", "search",
    "remove", "delete", "add", "insert", "update", "set", "get",
    "register", "initialize", "configure", "connect", "close", "open",
    "start", "stop", "run", "execute", "apply", "normalize",
    "transform", "wrap", "strip", "compress", "decompress", "compile",
    "resolve", "count", "copy", "move", "rename", "clear", "reset",
    "flush", "store", "retrieve", "manage", "control", "provide",
    "support", "implement", "perform", "handle", "process", "raise",
    "yield", "iterate", "return",
}

# A summary that opens with these is non-declarative or unsafe; skip.
_SUMMARY_SKIP_PREFIX = (
    "todo", "fixme", "see ", "see:", "note:", "warning:", "deprecated",
    "this module", "this class", "this function", "this section",
    "the following", "this is", "these are", "this method", "this is the",
    "for example", "in addition", "also,", "however", "if ", "when ",
    "available on", "changed in", "new in", "added in",
)


def _first_sentence(prose):
    if not prose:
        return ""
    m = re.search(r"\.(?:\s|$)", prose)
    summary = prose[:m.start()] if m else prose
    summary = summary.split("\n", 1)[0]
    return summary.strip().rstrip(".").strip()


def _clean_object(phrase):
    if not phrase:
        return None
    # Cut at a clause boundary signalling a second statement / aside.
    phrase = re.split(r"\s*;\s*|\s+--?\s+|:\s", phrase, 1)[0]
    phrase = re.split(r"-{3,}", phrase, 1)[0]
    # Drop a trailing truncated example clause ("..., such as l",
    # "... e.g. x") -- the example body got cut by sentence splitting and
    # leaves a dangling fragment.
    phrase = re.split(r",?\s+(?:such as|e\.g\.?|i\.e\.?|for example|like)\s",
                      phrase, 1, flags=re.IGNORECASE)[0]
    phrase = phrase.strip().rstrip(".,;:").strip()
    phrase = re.sub(r"\s+", " ", phrase)
    if not phrase:
        return None
    if "__" in phrase:
        return None
    pl = phrase.lower()
    if pl.startswith(("see ", "see:", "cf ", "e.g", "i.e", "ref ")):
        return None
    if _OBJ_BAD_RE.search(phrase):
        return None
    if re.search(r"\b\w+\.(?:py|js|ts|json|cpp|cc|hpp|java|go|rb|rs)\b", phrase):
        return None
    words = phrase.split()
    if len(words) < 1 or len(words) > _OBJ_MAX_WORDS:
        return None
    if not _OBJ_WORD_RE.search(phrase):
        return None
    low = phrase.lower()
    if low in ("a", "an", "the", "it", "this", "that", "none", "true", "false"):
        return None
    # Trailing dangling conjunction/preposition signals a cut fragment.
    if words[-1].lower() in ("the", "a", "an", "of", "to", "for", "with",
                             "and", "or", "in", "on", "by", "from", "as",
                             "that", "which", "if", "when", "either", "both",
                             "is", "are", "be", "into"):
        words = words[:-1]
        if not words:
            return None
        phrase = " ".join(words)
        if words[-1].lower() in ("the", "a", "an", "of", "to", "for"):
            return None
    return phrase


def _wtok(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def summary_bond(anchor, prose, source_url):
    """At most ONE clean typed bond from (anchor, descriptive prose)."""
    summary = _strip_inline_markup(_first_sentence(prose))
    if not summary or len(summary.split()) < 3:
        return None
    low = summary.lower()
    if low.startswith(_SUMMARY_SKIP_PREFIX):
        return None

    relation = obj = None
    for rx, rel in _DEF_PATTERNS:
        m = rx.match(summary)
        if m:
            cand = _clean_object(m.group(1))
            if cand:
                relation, obj = rel, cand
                break

    if relation is None:
        words = summary.split()
        first = re.sub(r"[^A-Za-z]", "", words[0]).lower()
        base = first
        if base.endswith("es") and base[:-2] in _ACTION_VERBS:
            base = base[:-2]
        elif base.endswith("s") and base[:-1] in _ACTION_VERBS:
            base = base[:-1]
        if base in _ACTION_VERBS:
            rest = _clean_object(" ".join(words[1:]))
            if rest:
                if base in ("return", "produce", "yield"):
                    relation, obj = "Produces", rest
                else:
                    relation, obj = "UsedFor", _gerund(base) + " " + rest

    if relation is None or not obj:
        return None

    # IsA verbalizes via " is a " -- strip a leading determiner from the
    # object so we don't double it ("is a a dictionary"). Produces/UsedFor
    # connectors ("produces", "is used for") keep the determiner fine.
    if relation == "IsA":
        obj = _LEAD_DET_RE.sub("", obj).strip()
        if not obj or len(obj.split()) < 1:
            return None

    # No-op / echo guard (subject must not be the object).
    a_tok, o_tok = _wtok(anchor), _wtok(obj)
    if a_tok == o_tok:
        return None
    o_cmp = list(o_tok)
    if o_cmp and o_cmp[0].endswith("ing"):
        o_cmp = o_cmp[1:]
    if o_cmp and o_cmp[0] in ("a", "an", "the"):
        o_cmp = o_cmp[1:]
    if len(a_tok) >= 1 and o_cmp and set(o_cmp) <= set(a_tok):
        return None
    # Class-echo guard: "a memoryview object" --Produces--> "a memory view
    # object ..." restates the subject's head noun as the object's head.
    # When the object's first content noun matches the subject's distinctive
    # name token, the bond is a near-tautology -- drop it. (Subject framing
    # for class kind is "a <Name> object"; the distinctive token is <Name>.)
    a_content = [t for t in a_tok if t not in ("a", "an", "the", "object",
                                               "function")]
    o_content = [t for t in o_tok if t not in ("a", "an", "the")]
    if a_content and o_content:
        head = a_content[0]
        # object opens by restating subject head noun (allow it past a
        # leading article we already stripped above).
        if o_content[0] == head and relation == "Produces":
            return None
    return {
        "subject": anchor,
        "relation": relation,
        "object": obj,
        "weight": 0.5,
        "confidence": "Unverified",
        "source_url": source_url,
    }


# ----------------------------------------------------------------------
# Block parsing: split a doc .txt into (anchor, prose, kind) definition
# blocks plus the module-header block.
# ----------------------------------------------------------------------

def _module_anchor_and_bond(lines, source_url):
    """If the file opens with a `"mod" --- blurb` header, emit a module
    bond and return (module_human_name, bond_or_None). The module intro
    paragraph that follows the header rule is also a strong source: a
    `The "mod" module ...` opener -> bond."""
    bonds = []
    mod_name = None
    for ln in lines[:6]:
        m = _MODULE_HDR_RE.match(ln)
        if m:
            mod_name = m.group(1)
            blurb = _strip_inline_markup(m.group(2))
            human = _humanize_dotted(mod_name)
            if human and blurb and len(blurb.split()) >= 2:
                anchor = f"the {human} module"
                cand = _clean_object(blurb)
                if cand and _wtok(cand) != _wtok(human):
                    bonds.append({
                        "subject": anchor, "relation": "UsedFor",
                        "object": cand, "weight": 0.5,
                        "confidence": "Unverified", "source_url": source_url,
                    })
            break
    return mod_name, bonds


def _iter_def_blocks(lines):
    """Yield (anchor_raw, kind, prose) for each column-0 definition line
    followed by an indented descriptive body. prose is the body text up
    to the next column-0 line / blank-separated section, with directive /
    code / table lines dropped."""
    n = len(lines)
    i = 0
    while i < n:
        line = lines[i]
        if not line or line[0] in " \t":
            i += 1
            continue
        # Skip section headers (underlined on the NEXT line).
        if i + 1 < n and _UNDERLINE_RE.match(lines[i + 1]):
            i += 2
            continue
        if _UNDERLINE_RE.match(line) or _TABLE_RE.match(line):
            i += 1
            continue
        kind = anchor_raw = None
        mc = _DEF_CLASS_RE.match(line)
        mf = _DEF_FUNC_RE.match(line)
        ma = _DEF_ATTR_RE.match(line)
        if mc:
            anchor_raw, kind = mc.group(1), "class"
        elif mf:
            anchor_raw, kind = mf.group(1), "func"
        elif ma:
            anchor_raw, kind = ma.group(1), "attr"
        if anchor_raw is None:
            i += 1
            continue
        # Collect indented body until the next column-0 non-blank line.
        j = i + 1
        body = []
        # Skip continuation signature lines: more column-0 overloads of the
        # same name appear as consecutive def lines (optionally separated by
        # blank lines). Advance over those def-overload + blank runs ONLY,
        # and STOP at the first indented body line (do not consume prose).
        while j < n:
            ln = lines[j]
            if not ln.strip():
                # Blank: peek ahead -- only skip it if the NEXT non-blank
                # line is another column-0 overload def; otherwise it's the
                # gap before the body, so stop here.
                k = j
                while k < n and not lines[k].strip():
                    k += 1
                if k < n and lines[k] and lines[k][0] not in " \t" and \
                        (_DEF_CLASS_RE.match(lines[k]) or _DEF_FUNC_RE.match(lines[k])):
                    j = k  # jump to the next overload def line
                    continue
                break
            if ln[0] not in " \t" and (_DEF_CLASS_RE.match(ln)
                                       or _DEF_FUNC_RE.match(ln)):
                j += 1  # consecutive overload def line
                continue
            break  # indented body line (or unrelated column-0 line)
        # Now j is at first body line (indented) or next def. Gather body.
        while j < n:
            ln = lines[j]
            if ln and ln[0] not in " \t":
                break  # next column-0 def/section ends this block
            stripped = ln.strip()
            if not stripped:
                # blank line: stop at the FIRST paragraph break -- we only
                # want the leading descriptive paragraph (the definition).
                if body:
                    break
                j += 1
                continue
            if _BODY_DROP_RE.match(ln) or _TABLE_RE.match(stripped) \
                    or _UNDERLINE_RE.match(stripped):
                j += 1
                continue
            body.append(stripped)
            j += 1
        prose = " ".join(body).strip()
        if anchor_raw and prose:
            yield anchor_raw, kind, prose
        i = max(j, i + 1)


def extract_bonds_from_pydoc(text, source_url=""):
    """Top-level: text of one .txt doc -> list of clean typed bond dicts.

    De-dupes (subject,relation,object) within the file."""
    lines = text.split("\n")
    bonds = []
    seen = set()

    def _push(b):
        if not b:
            return
        key = (b["subject"], b["relation"], b["object"])
        if key in seen:
            return
        seen.add(key)
        bonds.append(b)

    _mod, mod_bonds = _module_anchor_and_bond(lines, source_url)
    for b in mod_bonds:
        _push(b)

    for anchor_raw, kind, prose in _iter_def_blocks(lines):
        if any(tok in anchor_raw.lower() for tok in _CODE_SHAPE_TOKENS):
            continue
        anchor = _humanize_dotted(anchor_raw)
        if not anchor:
            continue
        # Frame the anchor as a readable subject noun phrase.
        if kind == "func":
            subj = f"the {anchor} function"
        elif kind == "class":
            subj = f"a {anchor} object"
        else:
            subj = anchor
        b = summary_bond(subj, prose, source_url)
        # Reject if object code-shape leaked.
        if b and any(tok in b["object"].lower() for tok in _CODE_SHAPE_TOKENS):
            b = None
        _push(b)
    return bonds


# ----------------------------------------------------------------------
# File walking
# ----------------------------------------------------------------------

# Subtrees we mine (highest-value declarative content).
INCLUDE_SUBDIRS = ("library", "reference")
# Files within those we still skip (index/api-index/changelog shells).
_SKIP_FILE_RE = re.compile(
    r"(?:^|/)(?:index|asyncio-api-index|asyncio-llapi-index|"
    r"asyncio-dev|asyncio-platforms)\.txt$"
)


def iter_doc_files(root):
    """Yield abspaths of .txt files under root/library and root/reference."""
    for sub in INCLUDE_SUBDIRS:
        d = os.path.join(root, sub)
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            if not name.endswith(".txt"):
                continue
            p = os.path.join(d, name)
            if _SKIP_FILE_RE.search(p.replace("\\", "/")):
                continue
            yield p


def doc_source_url(root, path):
    """Map a local .txt path to its docs.python.org html URL (stamp)."""
    rel = os.path.relpath(path, root).replace("\\", "/")
    rel = rel[:-4] if rel.endswith(".txt") else rel
    return f"https://docs.python.org/3/{rel}.html"
