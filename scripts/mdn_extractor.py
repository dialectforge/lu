"""Phase-2b: MDN web-platform documentation -> declarative typed bonds.

Self-contained extractor for the mdn/content repo (files/en-us/web/**).
Mirrors the DECLARATIVE summary-bond discipline proven for Python docs
(scripts/pydocs_extractor.py summary_bond) and CodeSearchNet: ONE clean
typed bond per concept, verbalizing as [subject] <connector> [object]
via narration.rs. No _ingest_tier2 path.

Design notes -- why this is structural, not a hack:
  * Every MDN reference page is an `index.md` with YAML frontmatter
    (`title`, optional `short-title`, `page-type`) + a Markdown body
    whose LEAD paragraph is a clean definitional sentence of a highly
    regular shape:
        "The **`<name>`** <type-word> <verb> <object>."
    e.g. "The **`Event`** interface represents an event ...",
         "The **`flex-direction`** CSS property sets how ...",
         "The **`map()`** method of Array instances creates a new array ...".
    The bolded subject + following verb is the relation signal; the
    remainder is the object. This is Wikipedia-class definitional prose.
  * The subject anchor is built from the frontmatter title + page-type
    so the NODE is a clean self-contained concept ("a Promise object",
    "the flex-direction CSS property", "the <video> element"), not a
    raw identifier.
  * STRIPPED: YAML frontmatter, KumaScript macros `{{...}}`, fenced and
    inline code, `> [!NOTE]` callouts, blockquotes, HTML tags, images
    `![..](..)`, Markdown links (keep TEXT drop URL), tables, headings,
    and everything from the first `##` section onward (so See also /
    Specifications / Browser compatibility / Syntax never reached).

The module is import-safe with zero deps so it can be unit-smoke-tested
offline (no brain) and then imported by ingest_curriculum_full.py.
"""
import os
import re

# ----------------------------------------------------------------------
# Frontmatter + body split
# ----------------------------------------------------------------------

_FM_KEY_RE = re.compile(r'^([a-zA-Z0-9_-]+):\s*(.*)$')


def split_frontmatter(text):
    """Return (frontmatter_dict, body_text). MDN frontmatter is a leading
    `---` ... `---` YAML block; we parse only the flat scalar keys we
    need (title, short-title, page-type). Quotes are stripped."""
    fm = {}
    if not text.startswith("---"):
        return fm, text
    end = text.find("\n---", 3)
    if end == -1:
        return fm, text
    block = text[3:end]
    body = text[end + 4:]
    body = body.lstrip("\n")
    for ln in block.split("\n"):
        m = _FM_KEY_RE.match(ln)
        if not m:
            continue
        key, val = m.group(1).strip().lower(), m.group(2).strip()
        # Strip surrounding quotes.
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        fm[key] = val
    return fm, body


# ----------------------------------------------------------------------
# Lead-paragraph extraction (the declarative content)
# ----------------------------------------------------------------------

_MACRO_RE = re.compile(r"\{\{[^{}]*\}\}")          # KumaScript {{...}}
_IMG_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")        # ![alt](src)
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")   # [text](url) -> text
_REF_LINK_RE = re.compile(r"\[([^\]]+)\]\[[^\]]*\]")  # [text][ref] -> text
_HTML_TAG_RE = re.compile(r"<[^>\n]{1,80}>")
_BOLD_RE = re.compile(r"\*\*([^*\n]+?)\*\*")
_ITALIC_RE = re.compile(r"(?<!\*)\*([^*\n]+?)\*(?!\*)")
_UNDERSCORE_EMPH_RE = re.compile(r"(?<!\w)_([^_\n]+?)_(?!\w)")
_BACKTICK_RE = re.compile(r"`([^`\n]+?)`")
_FOOTNOTE_REF_RE = re.compile(r"\[\^?\d+\]")

# Lines that are NOT prose and must never start / be part of the lead.
_MACRO_ONLY_RE = re.compile(r"^\s*(?:\{\{[^{}]*\}\}\s*)+$")
_CALLOUT_RE = re.compile(r"^\s*>")                 # blockquote / > [!NOTE]
_HEADING_RE = re.compile(r"^\s*#")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")
_TABLE_RE = re.compile(r"^\s*\|")
_LIST_RE = re.compile(r"^\s*(?:[-*+]\s|\d+\.\s)")
_IMG_LINE_RE = re.compile(r"^\s*!\[")


def lead_paragraph(body):
    """Return the first descriptive prose paragraph of an MDN body.

    Walks lines until the first `##` section (declarative content always
    precedes the first H2 on reference pages). Skips macro-only lines,
    callouts, code fences, tables, lists, images. The first prose line
    starts the paragraph; subsequent prose lines accrue until a blank
    line / non-prose line ends it. Returns '' if no clean lead found.
    """
    para = []
    in_fence = False
    for ln in body.split("\n"):
        if _FENCE_RE.match(ln):
            in_fence = not in_fence
            if para:
                break
            continue
        if in_fence:
            continue
        if _HEADING_RE.match(ln):
            break  # reached first section; lead is done
        stripped = ln.strip()
        if not stripped:
            if para:
                break  # paragraph break ends the lead
            continue
        if _MACRO_ONLY_RE.match(ln) or _CALLOUT_RE.match(ln) \
                or _TABLE_RE.match(ln) or _IMG_LINE_RE.match(ln):
            if para:
                break
            continue
        if _LIST_RE.match(ln):
            # A list before any prose = not a definitional lead; stop.
            break
        para.append(stripped)
    return " ".join(para).strip()


# Arg-bearing xref macros whose quoted argument is a real noun we keep.
_XREF_MACRO_RE = re.compile(
    r"\{\{\s*(?:dom|js|css|svg|web)?xref|HTMLElement|SVGElement|"
    r"Glossary|CSSxRef|jsxref|domxref\s*\([^)]*\)\s*\}\}",
    re.IGNORECASE)
# Generic form: {{ Macro("arg1", "arg2", ...) }} -> keep last quoted arg.
_MACRO_ARG_RE = re.compile(
    r"\{\{\s*(?P<name>[A-Za-z_][\w]*)\s*\(([^}]*)\)\s*\}\}")
_QUOTED_RE = re.compile(r'["\']([^"\']+)["\']')
# Macro names whose quoted arg is a meaningful term to inline as text.
_KEEP_ARG_MACROS = frozenset((
    "domxref", "jsxref", "cssxref", "svgxref", "webextapiref",
    "htmlelement", "svgelement", "glossary", "cssxref", "httpheader",
    "httpmethod", "httpstatus", "rfc",
))


def _macro_to_text(m):
    name = m.group("name").lower()
    if name not in _KEEP_ARG_MACROS:
        return " "
    quoted = _QUOTED_RE.findall(m.group(2))
    if not quoted:
        return " "
    # Display text is the last quoted arg for 2-arg xref forms; the term
    # itself for 1-arg forms. Strip a leading namespace path if present.
    term = quoted[-1].strip()
    term = term.split("/")[-1].split(".")[-1]
    return " " + term + " "


def strip_markup(text):
    """Strip KumaScript, links, images, inline code, emphasis, HTML tags,
    footnote refs from a prose string; collapse whitespace. Keeps link
    TEXT, inline-code token content, and the meaningful argument of common
    xref macros for readability."""
    # Inline xref/glossary macro args as plain text first.
    text = _MACRO_ARG_RE.sub(_macro_to_text, text)
    # Delete any remaining (arg-less or unknown) macros.
    text = _MACRO_RE.sub(" ", text)
    text = _MACRO_RE.sub(" ", text)
    text = _IMG_RE.sub(" ", text)
    text = _MD_LINK_RE.sub(r"\1", text)
    text = _REF_LINK_RE.sub(r"\1", text)
    text = _BOLD_RE.sub(r"\1", text)
    text = _ITALIC_RE.sub(r"\1", text)
    text = _UNDERSCORE_EMPH_RE.sub(r"\1", text)
    text = _BACKTICK_RE.sub(r"\1", text)
    text = _FOOTNOTE_REF_RE.sub(" ", text)
    text = _HTML_TAG_RE.sub(" ", text)
    text = text.replace("\\", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ----------------------------------------------------------------------
# Subject anchor construction from title + page-type
# ----------------------------------------------------------------------

# Strip MDN title decoration: titles can be `"`flex-direction` CSS
# property"`, `"`<video>` HTML video embed element"`, `Array.prototype.map()`,
# `Document: querySelector() method`, `Spread syntax (...)`.
_TITLE_BACKTICK_RE = re.compile(r"`([^`]+)`")


def _clean_title_name(title):
    """Extract the bare concept NAME from a frontmatter title."""
    t = title.strip()
    # If the title contains a backticked token, that's the canonical name.
    m = _TITLE_BACKTICK_RE.search(t)
    if m:
        return m.group(1).strip()
    # `Owner: member() method` -> `member()`  (qualified member pages)
    if ":" in t:
        t = t.split(":", 1)[1].strip()
    # Drop a trailing parenthetical PUNCTUATION gloss like "Spread syntax
    # (...)" or "Optional chaining (?.)" -- keep the worded name only.
    t = re.sub(r"\s*\(([^A-Za-z)]*)\)\s*$", "", t).strip()
    return t.strip()


def _humanize_member(name):
    """`Array.prototype.map()` -> `map()`; `Object.keys()` -> `keys()`;
    `RegExp` -> `RegExp`. Keep the leaf member; drop the dotted owner
    chain and `prototype`."""
    name = name.strip()
    # Drop a trailing `()` arglessness marker but remember it was callable.
    parts = name.split(".")
    parts = [p for p in parts if p.lower() != "prototype"]
    leaf = parts[-1] if parts else name
    return leaf.strip()


# page-type -> (subject_template, default_relation_hint)
# subject_template uses {n} for the cleaned name.
# The relation hint is only a fallback; the verb-based mapping below wins.
def subject_for(title, short_title, page_type):
    """Build a clean self-contained subject noun phrase from frontmatter.
    Returns None for page-types we do not mine (landing/listing/guide)."""
    pt = (page_type or "").lower()
    raw = _clean_title_name(short_title or title)
    if not raw:
        return None

    # --- JavaScript ---
    if pt == "javascript-class":
        return f"a {raw} object"
    if pt in ("javascript-instance-method", "javascript-static-method",
              "javascript-function"):
        name = _humanize_member(raw)
        if not name.endswith(")"):
            name += "()"
        return f"the {name} method"
    if pt in ("javascript-instance-accessor-property",
              "javascript-instance-data-property",
              "javascript-static-data-property",
              "javascript-static-accessor-property",
              "javascript-global-property"):
        name = _humanize_member(raw)
        return f"the {name} property"
    if pt == "javascript-operator":
        return f"the {raw} operator"
    if pt == "javascript-statement":
        return f"the {raw} statement"
    if pt == "javascript-error":
        return f"a {raw} error"
    if pt in ("javascript-language-feature", "javascript-namespace",
              "javascript-constructor"):
        return raw

    # --- Web API ---
    if pt == "web-api-interface":
        return f"the {raw} interface"
    if pt in ("web-api-instance-method", "web-api-static-method"):
        name = _humanize_member(raw)
        if not name.endswith(")"):
            name += "()"
        return f"the {name} method"
    if pt in ("web-api-instance-property", "web-api-static-property"):
        name = _humanize_member(raw)
        return f"the {name} property"
    if pt == "web-api-constructor":
        return f"the {raw} constructor"
    if pt == "web-api-event":
        return f"the {raw} event"

    # --- CSS ---
    if pt in ("css-property", "css-shorthand-property"):
        return f"the {raw} CSS property"
    if pt == "css-function":
        return f"the {raw} CSS function"
    if pt == "css-type":
        return f"the {raw} CSS data type"
    if pt == "css-pseudo-class":
        return f"the {raw} CSS pseudo-class"
    if pt == "css-pseudo-element":
        return f"the {raw} CSS pseudo-element"
    if pt == "css-at-rule":
        return f"the {raw} CSS at-rule"
    if pt == "css-media-feature":
        return f"the {raw} media feature"
    if pt == "css-selector":
        return f"the {raw} CSS selector"

    # --- HTML ---
    if pt == "html-element":
        nm = raw if raw.startswith("<") else f"<{raw}>"
        return f"the {nm} HTML element"
    if pt == "html-attribute":
        return f"the {raw} HTML attribute"

    # Unmined page-types (guide / landing-page / listing-page / overview /
    # webgl-extension / attribute-value / at-rule-descriptor / keyword /
    # combinator).
    return None


# ----------------------------------------------------------------------
# Verb -> relation mapping + object extraction
# (ported + adapted from pydocs_extractor.summary_bond)
# ----------------------------------------------------------------------

_LEAD_DET_RE = re.compile(r"^(?:the|a|an)\s+", re.IGNORECASE)

_IRREGULAR_GERUND = {
    "set": "setting", "get": "getting", "put": "putting",
    "run": "running", "stop": "stopping", "split": "splitting",
    "map": "mapping", "strip": "stripping", "wrap": "wrapping",
    "submit": "submitting", "control": "controlling",
    "format": "formatting", "begin": "beginning",
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

_OBJ_BAD_RE = re.compile(r"[<>{}\[\]|`=#]|https?://|::|->|=>|\(\)")
_OBJ_WORD_RE = re.compile(r"[A-Za-z]")
_OBJ_MAX_WORDS = 20
_OBJ_MIN_WORDS = 2

# After the bolded subject, the lead reads "<type-word> <VERB> <object>".
# We anchor on the FIRST finite verb after the subject. These regexes run
# against the markup-stripped lead with the subject phrase removed.

# IsA: "... represents/is/describes an X"
_ISA_VERBS = ("represents", "is", "describes", "defines", "denotes",
              "embeds", "contains")
# Produces: "... returns/produces/creates/generates a new X"
_PRODUCES_VERBS = ("returns", "produces", "creates", "generates",
                   "yields", "constructs", "builds", "makes")
# UsedFor: "... sets/specifies/controls/allows/enables/lets/provides X"
_USEDFOR_VERBS = ("sets", "specifies", "controls", "allows", "enables",
                  "lets", "provides", "determines", "indicates",
                  "configures", "applies", "adds", "removes", "selects",
                  "matches", "executes", "performs", "handles",
                  "manages", "associates", "schedules", "converts",
                  "encodes", "decodes", "parses", "serializes",
                  "retrieves", "queries", "registers", "attaches",
                  "toggles", "replaces", "inserts", "appends",
                  "reads", "writes", "stores", "loads", "checks",
                  "validates", "calculates", "computes", "measures")

# Build one combined alternation, longest-first so multi-word never
# truncated. Each verb maps to its relation.
_VERB_RELATION = {}
for _v in _ISA_VERBS:
    _VERB_RELATION[_v] = "IsA"
for _v in _PRODUCES_VERBS:
    _VERB_RELATION[_v] = "Produces"
for _v in _USEDFOR_VERBS:
    _VERB_RELATION[_v] = "UsedFor"
_ALL_VERBS = sorted(_VERB_RELATION.keys(), key=len, reverse=True)
_VERB_ALT = "|".join(re.escape(v) for v in _ALL_VERBS)
# Match "<anything up to ~6 words> <VERB> <rest>" where the verb is the
# first matched finite verb; the leading span is the type-word(s) like
# "CSS property", "interface", "method of Array instances".
_VERB_RE = re.compile(
    r"^(?P<pre>(?:\S+\s+){0,8}?)(?P<verb>" + _VERB_ALT + r")\s+(?P<obj>.+)$",
    re.IGNORECASE,
)

_SUMMARY_SKIP_PREFIX = (
    "todo", "fixme", "see ", "see:", "note:", "warning:", "deprecated",
    "this page", "this section", "the following", "for example",
    "in addition", "however", "experimental",
)


def _first_sentence(prose):
    if not prose:
        return ""
    m = re.search(r"\.(?:\s|$)", prose)
    s = prose[:m.start()] if m else prose
    return s.strip().rstrip(".").strip()


_PAREN_RE = re.compile(r"\s*\([^)]*\)")
# Clause boundaries we truncate a long object at (keep the head clause).
_CLAUSE_CUT_RE = re.compile(
    r"\s+(?:defining |that |which |whose |where |when |based on |"
    r"used to |so that |in order to )",
    re.IGNORECASE)


def _clean_object(phrase):
    if not phrase:
        return None
    # Drop parenthetical asides (incl. "(or failure)", "(normal or reversed)").
    phrase = _PAREN_RE.sub("", phrase)
    phrase = re.split(r"\s*;\s*|\s+--?\s+|:\s", phrase, 1)[0]
    phrase = re.split(
        r",?\s+(?:such as|e\.g\.?|i\.e\.?|for example|like)\s",
        phrase, 1, flags=re.IGNORECASE)[0]
    phrase = phrase.strip().rstrip(".,;:").strip()
    phrase = re.sub(r"\s+", " ", phrase)
    if not phrase:
        return None
    pl = phrase.lower()
    if pl.startswith(("see ", "see:", "cf ", "e.g", "i.e", "ref ")):
        return None
    if _OBJ_BAD_RE.search(phrase):
        return None
    words = phrase.split()
    # Soft-truncate an over-long object at the first subordinate-clause
    # boundary (keeps a tight definitional head instead of rejecting).
    if len(words) > _OBJ_MAX_WORDS:
        mcut = _CLAUSE_CUT_RE.search(phrase)
        if mcut and _OBJ_MIN_WORDS <= len(phrase[:mcut.start()].split()) <= _OBJ_MAX_WORDS:
            phrase = phrase[:mcut.start()].strip().rstrip(",")
            words = phrase.split()
        else:
            # Hard word cap: keep the leading _OBJ_MAX_WORDS words, then
            # back off any dangling preposition/conjunction tail.
            words = words[:_OBJ_MAX_WORDS]
            while words and words[-1].lower() in (
                    "the","a","an","of","to","for","with","and","or",
                    "in","on","by","from","as","that","which","if","when",
                    "is","are","be","into","every","each","all","its"):
                words = words[:-1]
            phrase = " ".join(words)
    if len(words) < _OBJ_MIN_WORDS or len(words) > _OBJ_MAX_WORDS:
        return None
    if not _OBJ_WORD_RE.search(phrase):
        return None
    if pl in ("a", "an", "the", "it", "this", "that", "none", "true",
              "false"):
        return None
    # Trailing dangling conjunction/preposition signals a cut fragment.
    tail = words[-1].lower()
    if tail in ("the", "a", "an", "of", "to", "for", "with", "and", "or",
                "in", "on", "by", "from", "as", "that", "which", "if",
                "when", "either", "both", "is", "are", "be", "into",
                "whether", "where", "how"):
        words = words[:-1]
        if len(words) < _OBJ_MIN_WORDS:
            return None
        phrase = " ".join(words)
        if words[-1].lower() in ("the", "a", "an", "of", "to", "for"):
            return None
    return phrase


def _wtok(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def _strip_subject_prefix(lead, raw_name):
    """Remove the leading bolded subject restatement from the lead so the
    verb matcher sees only "<type-word> <verb> <object>". MDN leads open
    with "The <name> <type-word> ..."; after markup stripping the bold is
    gone but the literal name + a determiner remains. We drop a leading
    'the/a/an' + the name token(s) if present."""
    s = lead
    s = _LEAD_DET_RE.sub("", s, count=1)
    # Drop the concrete name token if it leads (case-insensitive, also the
    # <tag> / member() / dotted forms).
    nm = raw_name.strip().lstrip("<").rstrip(">").split(".")[-1]
    nm = nm.rstrip("()")
    if nm:
        # remove a leading run of the name and common glue ("map() method
        # of Array instances") handled by the {0,8} pre-span in _VERB_RE,
        # so here we only strip an exact leading name token.
        pat = re.compile(r"^" + re.escape(nm) + r"(?:\(\))?\s+",
                         re.IGNORECASE)
        s = pat.sub("", s, count=1)
    return s.strip()


def bond_from_lead(subject, lead, raw_name, source_url):
    """At most ONE clean typed bond from (subject, lead paragraph)."""
    summary = _first_sentence(strip_markup(lead))
    if not summary or len(summary.split()) < 4:
        return None
    low = summary.lower()
    if low.startswith(_SUMMARY_SKIP_PREFIX):
        return None

    stripped = _strip_subject_prefix(summary, raw_name)
    m = _VERB_RE.match(stripped)
    if not m:
        return None
    verb = m.group("verb").lower()
    relation = _VERB_RELATION.get(verb)
    if not relation:
        return None
    raw_obj = m.group("obj").strip()
    rol = raw_obj.lower()

    # Passive-lead reclassification. The generic matcher fires on the
    # copula "is/are" and would type "is used to read ..." or
    # "is fired when ..." as IsA, yielding ungrammatical "X is a used
    # to ..." / "X is a fired when ..." bonds. Handle the passive family
    # explicitly: "is used to/for" -> UsedFor; pure event/callback firing
    # leads ("is fired/dispatched/called/invoked/triggered/sent/raised/
    # emitted ...") carry no clean definitional object -> skip (no junk).
    if relation == "IsA":
        if rol.startswith(("used to ", "used for ", "designed to ",
                           "meant to ", "intended to ")):
            relation = "UsedFor"
            parts3 = raw_obj.split(" ", 2)
            tail = parts3[2] if len(parts3) == 3 else ""
            # 'used to read X' -> tail 'read X'; gerund the leading verb so
            # UsedFor reads 'is used for reading the contents ...'.
            tparts = tail.split(" ", 1)
            if tparts and tparts[0].isalpha():
                ger = _gerund(tparts[0])
                tail = ger + (" " + tparts[1] if len(tparts) > 1 else "")
            raw_obj = tail
        elif rol.split(" ", 1)[0] in (
                "fired", "dispatched", "called", "invoked", "triggered",
                "sent", "raised", "emitted", "thrown", "executed",
                "available", "supported", "deprecated", "defined",
                "implemented", "exposed", "returned", "set", "based",
                "part"):
            return None

    obj = _clean_object(raw_obj)
    if not obj:
        return None

    if relation == "IsA":
        obj = _LEAD_DET_RE.sub("", obj).strip()
        if not obj or len(obj.split()) < _OBJ_MIN_WORDS:
            return None
        # IsA object must read as a noun phrase, not a stray verb/adverb
        # tail. Reject if it still opens with a passive participle/adverb.
        if obj.lower().split(" ", 1)[0] in (
                "used", "fired", "called", "invoked", "based", "set",
                "returned", "defined", "like", "part", "available"):
            return None

    # No-op / echo guard (subject must not be the object).
    a_tok, o_tok = _wtok(subject), _wtok(obj)
    if a_tok == o_tok:
        return None
    o_cmp = list(o_tok)
    if o_cmp and o_cmp[0] in ("a", "an", "the"):
        o_cmp = o_cmp[1:]
    if len(a_tok) >= 1 and o_cmp and set(o_cmp) <= set(a_tok):
        return None

    return {
        "subject": subject,
        "relation": relation,
        "object": obj,
        "weight": 0.5,
        "confidence": "Unverified",
        "source_url": source_url,
    }


# ----------------------------------------------------------------------
# Top-level page extraction
# ----------------------------------------------------------------------

# page-types we never mine (no clean definitional concept).
_SKIP_PAGE_TYPES = frozenset((
    "guide", "landing-page", "listing-page", "web-api-overview",
    "webgl-extension", "webgl-extension-method", "html-attribute-value",
    "css-at-rule-descriptor", "css-keyword", "css-combinator",
    "deprecated", "",
))


def extract_bond_from_page(text, source_url=""):
    """Text of one MDN index.md -> at most one clean typed bond dict, or
    None. AT MOST ONE bond per page (spec)."""
    fm, body = split_frontmatter(text)
    title = fm.get("title", "")
    short = fm.get("short-title", "")
    page_type = fm.get("page-type", "")
    if not title:
        return None
    if page_type.lower() in _SKIP_PAGE_TYPES:
        return None
    subject = subject_for(title, short, page_type)
    if not subject:
        return None
    lead = lead_paragraph(body)
    if not lead:
        return None
    raw_name = _clean_title_name(short or title)
    return bond_from_lead(subject, lead, raw_name, source_url)


# ----------------------------------------------------------------------
# File walking
# ----------------------------------------------------------------------

# The reference subtrees we mine (highest-value declarative concepts).
# Relative to <root>/files/en-us.
INCLUDE_TREES = (
    "web/javascript/reference",
    "web/api",
    "web/css/reference",
    "web/html/reference",
)

# Path fragments that mark noise we always skip (tutorials, guides,
# how-tos, learn).
_SKIP_PATH_RE = re.compile(
    r"(?:^|/)(?:guides?|tutorials?|how_to|learn|community)(?:/|$)")


def iter_doc_files(root):
    """Yield abspaths of index.md files under <root>/files/en-us/<tree>
    for each INCLUDE_TREES entry, skipping guide/tutorial/how_to noise."""
    base = os.path.join(root, "files", "en-us")
    if not os.path.isdir(base):
        # Allow callers to pass the en-us dir directly.
        base = root
    for tree in INCLUDE_TREES:
        d = os.path.join(base, tree)
        if not os.path.isdir(d):
            continue
        for dirpath, dirnames, filenames in os.walk(d):
            rel = os.path.relpath(dirpath, base).replace("\\", "/")
            if _SKIP_PATH_RE.search(rel):
                # prune the walk into noise subtrees
                dirnames[:] = []
                continue
            if "index.md" in filenames:
                yield os.path.join(dirpath, "index.md")


def doc_source_url(root, path):
    """Map a local index.md path to its developer.mozilla.org URL."""
    base = os.path.join(root, "files", "en-us")
    if not os.path.isdir(base):
        base = root
    rel = os.path.relpath(os.path.dirname(path), base).replace("\\", "/")
    return f"https://developer.mozilla.org/en-US/docs/{rel}"
