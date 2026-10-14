#!/usr/bin/env python3
"""Phase-2 multi-language STRUCTURAL code extractor (tree-sitter backbone).

Extends scripts/pycode_extractor.py (stdlib-ast, Python-only) to JS/TS, Rust,
Go, Java, C and C++ for The Stack v2 ingest. Emits the SAME (subject, relation,
object) structural-bond vocabulary and the SAME concept-naming discipline as
the Python extractor, so bonds from every language merge into one graph.

Relations (must match src/extractor/types.rs bondtype_from_relation_str):
    HasMethod, HasAttribute, HasParameter, Returns,
    Inherits, Calls, HasType, Implements

Concept-naming discipline (identical to pycode_extractor):
  * nominal types (class/struct/enum/trait/interface) -> "the X class"
  * top-level functions                               -> "the foo function"
  * methods (qualified by owning type)                -> "Owner.method"
  * params / attributes                               -> bare name
  * types                                             -> bare last-segment name
  * call targets                                      -> bare callee or "recv.method"

DECLARATIVE / STRUCTURAL-ONLY HARD CONSTRAINT:
  Every subject and object is a clean concept name derived from a NAMED AST
  identifier / type node. We NEVER emit raw source text, string / numeric
  literals, comments, or expression / statement bodies. `_clean_ident` /
  `_clean_qual` gate every name; anything that is not a bare (or single-dotted)
  identifier is dropped rather than emitted. This mirrors the Python
  extractor's "name both ends cleanly or skip" rule.

Returned bonds are dicts shaped for ingest_curriculum_full.BatchSink:
    {"subject":..,"relation":..,"object":..,"weight":..,"confidence":..,
     "source_url":..}
The caller stamps corpus / run_id.
"""
import os
import re

# ---------------------------------------------------------------------------
# Relation tags -- MUST match the CamelCase names in bondtype_from_relation_str
# ---------------------------------------------------------------------------
REL_HAS_METHOD = "HasMethod"
REL_HAS_ATTRIBUTE = "HasAttribute"
REL_HAS_PARAMETER = "HasParameter"
REL_RETURNS = "Returns"
REL_INHERITS = "Inherits"
REL_CALLS = "Calls"
REL_HAS_TYPE = "HasType"
REL_IMPLEMENTS = "Implements"

DEFAULT_WEIGHT = 0.8
DEFAULT_CONFIDENCE = "Verified"  # source code is ground-truth structure

# ---------------------------------------------------------------------------
# Clean-name gates (the declarative/structural-only enforcement point)
# ---------------------------------------------------------------------------
_IDENT_RE = re.compile(r"^[A-Za-z_$][A-Za-z0-9_$]*$")
_QUAL_RE = re.compile(r"^[A-Za-z_$][A-Za-z0-9_$]*(?:\.[A-Za-z_$][A-Za-z0-9_$]*)+$")


def _clean_ident(s):
    """Return s iff it is a single bare identifier, else None."""
    if not s:
        return None
    s = s.strip()
    return s if _IDENT_RE.match(s) else None


def _clean_qual(s):
    """Return s iff it is a bare or single/multi-dotted identifier, else None."""
    if not s:
        return None
    s = s.strip()
    if _IDENT_RE.match(s) or _QUAL_RE.match(s):
        return s
    return None


def _last_seg(s):
    """Collapse a scoped/qualified type ref to its final identifier segment."""
    if not s:
        return None
    s = s.strip()
    for sep in ("::", "."):
        if sep in s:
            s = s.split(sep)[-1]
    return _clean_ident(s)


def class_concept(name):
    return "the %s class" % name


def func_concept(name):
    return "the %s function" % name


def method_concept(owner, meth):
    return "%s.%s" % (owner, meth)


# ---------------------------------------------------------------------------
# tree-sitter plumbing
# ---------------------------------------------------------------------------
_PARSERS = {}


def _get_parser(lang):
    p = _PARSERS.get(lang)
    if p is None:
        from tree_sitter_language_pack import get_parser
        p = get_parser(lang)
        _PARSERS[lang] = p
    return p


def _txt(node):
    """Decode a NAMED identifier/type node's text. Only ever called on nodes
    that are themselves identifiers or type names -- never on bodies."""
    if node is None:
        return None
    return node.text.decode("utf-8", "replace")


def _field(node, name):
    return node.child_by_field_name(name)


def _walk(node):
    """Yield every descendant of node (inclusive), depth-first."""
    stack = [node]
    while stack:
        n = stack.pop()
        yield n
        stack.extend(reversed(n.children))


_TYPE_NODE_TYPES = {
    "type_identifier", "primitive_type", "predefined_type", "identifier",
    "field_identifier", "scoped_type_identifier", "qualified_type",
    "generic_type", "pointer_type", "reference_type", "type",
}


def _first_type_name(node):
    """Best-effort primary type name from a type/annotation subtree."""
    if node is None:
        return None
    core = {"type_identifier", "primitive_type", "predefined_type",
            "identifier", "field_identifier", "scoped_type_identifier",
            "integral_type", "floating_point_type", "boolean_type",
            "void_type"}
    if node.type in core:
        return _last_seg(_txt(node))
    for c in _walk(node):
        if c.type in core:
            return _last_seg(_txt(c))
    # gated fallback: whole-node text iff it is itself a clean single ident
    return _last_seg(_txt(node))


# ---------------------------------------------------------------------------
# Per-language call-noise (builtins / macros that add clutter, not structure)
# ---------------------------------------------------------------------------
_NOISE_COMMON = {"print", "println", "printf", "format", "len", "str", "int",
                 "type", "super", "isinstance"}
_NOISE = {
    "javascript": _NOISE_COMMON | {"console", "require", "parseInt",
                                   "parseFloat", "String", "Number", "Boolean"},
    "typescript": _NOISE_COMMON | {"console", "require", "parseInt",
                                   "parseFloat", "String", "Number", "Boolean"},
    "tsx": _NOISE_COMMON | {"console", "require"},
    "rust": _NOISE_COMMON | {"panic", "assert", "assert_eq", "vec", "Some",
                             "Ok", "Err", "None", "Box", "clone", "unwrap",
                             "to_string", "into", "from"},
    "go": _NOISE_COMMON | {"make", "new", "append", "panic", "recover",
                           "cap", "copy", "delete", "close"},
    "java": _NOISE_COMMON | {"System", "valueOf", "toString", "equals",
                            "hashCode", "getClass"},
    "c": _NOISE_COMMON | {"malloc", "free", "sizeof", "memcpy", "memset",
                          "strlen", "strcmp", "fprintf", "sprintf"},
    "cpp": _NOISE_COMMON | {"malloc", "free", "sizeof", "cout", "cerr",
                           "endl", "make_shared", "make_unique", "move",
                           "static_cast", "dynamic_cast"},
}


def _noise_for(lang):
    return _NOISE.get(lang, _NOISE_COMMON)


# ===========================================================================
# JavaScript / TypeScript
# ===========================================================================
def _js_callee(call_node):
    f = _field(call_node, "function")
    if f is None:
        return None
    if f.type == "identifier":
        return _clean_ident(_txt(f))
    if f.type == "member_expression":
        obj = _field(f, "object")
        prop = _field(f, "property")
        pname = _clean_ident(_txt(prop)) if prop is not None else None
        if pname is None:
            return None
        if obj is not None and obj.type == "identifier":
            oname = _clean_ident(_txt(obj))
            if oname:
                return "%s.%s" % (oname, pname)
        return pname
    return None


def _js_collect_calls(body, lang, out):
    noise = _noise_for(lang)
    seen = set()
    if body is None:
        return []
    for n in _walk(body):
        if n.type == "call_expression":
            name = _js_callee(n)
            if not name:
                continue
            if name.split(".")[-1] in noise or name.split(".")[0] in noise:
                continue
            if name not in seen:
                seen.add(name)
                out.append(name)
    return out


def _js_params_and_return(subject, fn_node, bonds):
    params = _field(fn_node, "parameters")
    if params is not None:
        for p in params.named_children:
            name = None
            tnode = None
            if p.type == "identifier":
                name = _clean_ident(_txt(p))
            elif p.type in ("required_parameter", "optional_parameter"):
                pat = _field(p, "pattern")
                if pat is not None and pat.type == "identifier":
                    name = _clean_ident(_txt(pat))
                tnode = _field(p, "type")
            elif p.type == "assignment_pattern":
                left = _field(p, "left")
                if left is not None and left.type == "identifier":
                    name = _clean_ident(_txt(left))
            if name:
                bonds.append((subject, REL_HAS_PARAMETER, name))
                t = _first_type_name(tnode) if tnode is not None else None
                if t:
                    bonds.append((name, REL_HAS_TYPE, t))
    rt = _field(fn_node, "return_type")
    if rt is not None:
        t = _first_type_name(rt)
        if t and t not in ("void", "None"):
            bonds.append((subject, REL_RETURNS, t))


def _js_heritage(class_node, cls, bonds):
    for ch in class_node.children:
        if ch.type == "class_heritage":
            mode = None
            for h in ch.children:
                if h.type == "extends" or (h.type == "identifier" and h.text == b"extends"):
                    mode = "extends"
                    continue
                if h.type == "implements_clause":
                    for t in h.named_children:
                        bn = _first_type_name(t)
                        if bn:
                            bonds.append((cls, REL_IMPLEMENTS, class_concept(bn)))
                    continue
                if h.is_named and mode == "extends":
                    bn = _first_type_name(h)
                    if bn:
                        bonds.append((cls, REL_INHERITS, class_concept(bn)))
                    mode = None


def _js_class_body(class_node, cls, owner, lang, bonds):
    body = _field(class_node, "body")
    if body is None:
        return
    for item in body.named_children:
        if item.type == "method_definition":
            nm = _field(item, "name")
            mname = _clean_ident(_txt(nm)) if nm is not None else None
            if not mname:
                continue
            m = method_concept(owner, mname)
            bonds.append((cls, REL_HAS_METHOD, m))
            _js_params_and_return(m, item, bonds)
            for callee in _js_collect_calls(_field(item, "body"), lang, []):
                bonds.append((m, REL_CALLS, callee))
        elif item.type in ("field_definition", "public_field_definition",
                            "property_signature"):
            nm = _field(item, "name") or _field(item, "property")
            aname = _clean_ident(_txt(nm)) if nm is not None else None
            if not aname:
                continue
            bonds.append((cls, REL_HAS_ATTRIBUTE, aname))
            t = _first_type_name(_field(item, "type"))
            if t:
                bonds.append((aname, REL_HAS_TYPE, t))
        elif item.type == "method_signature":
            nm = _field(item, "name")
            mname = _clean_ident(_txt(nm)) if nm is not None else None
            if mname:
                m = method_concept(owner, mname)
                bonds.append((cls, REL_HAS_METHOD, m))
                _js_params_and_return(m, item, bonds)


def _extract_js(root, lang):
    bonds = []
    for n in _walk(root):
        if n.type in ("class_declaration", "class"):
            nm = _field(n, "name")
            cname = _clean_ident(_txt(nm)) if nm is not None else None
            if not cname:
                continue
            cls = class_concept(cname)
            _js_heritage(n, cls, bonds)
            _js_class_body(n, cls, cname, lang, bonds)
        elif n.type == "interface_declaration":
            nm = _field(n, "name")
            cname = _clean_ident(_txt(nm)) if nm is not None else None
            if not cname:
                continue
            cls = class_concept(cname)
            # interface extends
            for ch in n.children:
                if ch.type == "extends_type_clause" or ch.type == "extends_clause":
                    for t in ch.named_children:
                        bn = _first_type_name(t)
                        if bn:
                            bonds.append((cls, REL_INHERITS, class_concept(bn)))
            _js_class_body(n, cls, cname, lang, bonds)
        elif n.type == "function_declaration":
            # skip functions nested in classes (handled as methods)
            nm = _field(n, "name")
            fname = _clean_ident(_txt(nm)) if nm is not None else None
            if not fname:
                continue
            fn = func_concept(fname)
            _js_params_and_return(fn, n, bonds)
            for callee in _js_collect_calls(_field(n, "body"), lang, []):
                bonds.append((fn, REL_CALLS, callee))
    return bonds


# ===========================================================================
# Rust
# ===========================================================================
def _rust_callee(call_node):
    f = _field(call_node, "function")
    if f is None:
        return None
    if f.type == "identifier":
        return _clean_ident(_txt(f))
    if f.type == "scoped_identifier":
        nm = _field(f, "name")
        return _clean_ident(_txt(nm)) if nm is not None else _last_seg(_txt(f))
    if f.type == "field_expression":
        fld = _field(f, "field")
        val = _field(f, "value")
        mname = _clean_ident(_txt(fld)) if fld is not None else None
        if mname is None:
            return None
        if val is not None and val.type == "identifier":
            oname = _clean_ident(_txt(val))
            if oname:
                return "%s.%s" % (oname, mname)
        return mname
    return None


def _rust_collect_calls(body, out):
    noise = _noise_for("rust")
    seen = set()
    if body is None:
        return out
    for n in _walk(body):
        if n.type == "call_expression":
            name = _rust_callee(n)
            if not name:
                continue
            if name.split(".")[-1] in noise or name.split(".")[0] in noise:
                continue
            if name not in seen:
                seen.add(name)
                out.append(name)
    return out


def _rust_params_and_return(subject, fn_node, bonds):
    params = _field(fn_node, "parameters")
    if params is not None:
        for p in params.named_children:
            if p.type != "parameter":
                continue
            pat = _field(p, "pattern")
            name = None
            if pat is not None and pat.type == "identifier":
                name = _clean_ident(_txt(pat))
            if not name:
                continue
            bonds.append((subject, REL_HAS_PARAMETER, name))
            t = _first_type_name(_field(p, "type"))
            if t:
                bonds.append((name, REL_HAS_TYPE, t))
    rt = _field(fn_node, "return_type")
    if rt is not None:
        t = _first_type_name(rt)
        if t:
            bonds.append((subject, REL_RETURNS, t))


def _rust_struct_fields(struct_node, cls, bonds):
    for c in _walk(struct_node):
        if c.type == "field_declaration":
            nm = _field(c, "name")
            aname = _clean_ident(_txt(nm)) if nm is not None else None
            if not aname:
                continue
            bonds.append((cls, REL_HAS_ATTRIBUTE, aname))
            t = _first_type_name(_field(c, "type"))
            if t:
                bonds.append((aname, REL_HAS_TYPE, t))


def _extract_rust(root, lang):
    bonds = []
    for n in root.children:
        if n.type == "struct_item":
            nm = _field(n, "name")
            sname = _clean_ident(_txt(nm)) if nm is not None else None
            if sname:
                _rust_struct_fields(n, class_concept(sname), bonds)
        elif n.type in ("enum_item", "union_item"):
            pass  # nominal type present but no attributes emitted in v1
        elif n.type == "trait_item":
            nm = _field(n, "name")
            tname = _clean_ident(_txt(nm)) if nm is not None else None
            if not tname:
                continue
            cls = class_concept(tname)
            body = _field(n, "body")
            if body is not None:
                for it in body.named_children:
                    if it.type in ("function_signature_item", "function_item"):
                        mn = _field(it, "name")
                        mname = _clean_ident(_txt(mn)) if mn is not None else None
                        if mname:
                            m = method_concept(tname, mname)
                            bonds.append((cls, REL_HAS_METHOD, m))
                            _rust_params_and_return(m, it, bonds)
        elif n.type == "impl_item":
            type_node = _field(n, "type")
            tyname = _first_type_name(type_node)
            if not tyname:
                continue
            trait_node = _field(n, "trait")
            if trait_node is not None:
                trname = _first_type_name(trait_node)
                if trname:
                    bonds.append((class_concept(tyname), REL_IMPLEMENTS,
                                  class_concept(trname)))
            body = _field(n, "body")
            if body is not None:
                for it in body.named_children:
                    if it.type == "function_item":
                        mn = _field(it, "name")
                        mname = _clean_ident(_txt(mn)) if mn is not None else None
                        if not mname:
                            continue
                        m = method_concept(tyname, mname)
                        bonds.append((class_concept(tyname), REL_HAS_METHOD, m))
                        _rust_params_and_return(m, it, bonds)
                        for callee in _rust_collect_calls(_field(it, "body"), []):
                            bonds.append((m, REL_CALLS, callee))
        elif n.type == "function_item":
            nm = _field(n, "name")
            fname = _clean_ident(_txt(nm)) if nm is not None else None
            if not fname:
                continue
            fn = func_concept(fname)
            _rust_params_and_return(fn, n, bonds)
            for callee in _rust_collect_calls(_field(n, "body"), []):
                bonds.append((fn, REL_CALLS, callee))
    return bonds


# ===========================================================================
# Go
# ===========================================================================
def _go_callee(call_node):
    f = _field(call_node, "function")
    if f is None:
        return None
    if f.type == "identifier":
        return _clean_ident(_txt(f))
    if f.type == "selector_expression":
        operand = _field(f, "operand")
        fld = _field(f, "field")
        mname = _clean_ident(_txt(fld)) if fld is not None else None
        if mname is None:
            return None
        if operand is not None and operand.type == "identifier":
            oname = _clean_ident(_txt(operand))
            if oname:
                return "%s.%s" % (oname, mname)
        return mname
    return None


def _go_collect_calls(body, out):
    noise = _noise_for("go")
    seen = set()
    if body is None:
        return out
    for n in _walk(body):
        if n.type == "call_expression":
            name = _go_callee(n)
            if not name:
                continue
            if name.split(".")[-1] in noise or name.split(".")[0] in noise:
                continue
            if name not in seen:
                seen.add(name)
                out.append(name)
    return out


def _go_params(subject, params_node, bonds):
    if params_node is None:
        return
    for p in params_node.named_children:
        if p.type != "parameter_declaration":
            continue
        t = _first_type_name(_field(p, "type"))
        names = [c for c in p.named_children if c.type == "identifier"]
        for nm in names:
            name = _clean_ident(_txt(nm))
            if not name:
                continue
            bonds.append((subject, REL_HAS_PARAMETER, name))
            if t:
                bonds.append((name, REL_HAS_TYPE, t))


def _go_return(subject, fn_node, bonds):
    res = _field(fn_node, "result")
    if res is None:
        return
    t = _first_type_name(res)
    if t:
        bonds.append((subject, REL_RETURNS, t))


def _extract_go(root, lang):
    bonds = []
    for n in root.children:
        if n.type == "function_declaration":
            nm = _field(n, "name")
            fname = _clean_ident(_txt(nm)) if nm is not None else None
            if not fname:
                continue
            fn = func_concept(fname)
            _go_params(fn, _field(n, "parameters"), bonds)
            _go_return(fn, n, bonds)
            for callee in _go_collect_calls(_field(n, "body"), []):
                bonds.append((fn, REL_CALLS, callee))
        elif n.type == "method_declaration":
            recv = _field(n, "receiver")
            owner = None
            if recv is not None:
                for pd in _walk(recv):
                    if pd.type in ("type_identifier",):
                        owner = _clean_ident(_txt(pd))
                        break
            nm = _field(n, "name")
            mname = _clean_ident(_txt(nm)) if nm is not None else None
            if not owner or not mname:
                continue
            cls = class_concept(owner)
            m = method_concept(owner, mname)
            bonds.append((cls, REL_HAS_METHOD, m))
            _go_params(m, _field(n, "parameters"), bonds)
            _go_return(m, n, bonds)
            for callee in _go_collect_calls(_field(n, "body"), []):
                bonds.append((m, REL_CALLS, callee))
        elif n.type == "type_declaration":
            for spec in n.named_children:
                if spec.type != "type_spec":
                    continue
                nm = _field(spec, "name")
                tname = _clean_ident(_txt(nm)) if nm is not None else None
                if not tname:
                    continue
                cls = class_concept(tname)
                tnode = _field(spec, "type")
                if tnode is None:
                    continue
                if tnode.type == "struct_type":
                    for c in _walk(tnode):
                        if c.type == "field_declaration":
                            ft = _first_type_name(_field(c, "type"))
                            fnames = [x for x in c.named_children
                                      if x.type == "field_identifier"]
                            for fn_id in fnames:
                                aname = _clean_ident(_txt(fn_id))
                                if not aname:
                                    continue
                                bonds.append((cls, REL_HAS_ATTRIBUTE, aname))
                                if ft:
                                    bonds.append((aname, REL_HAS_TYPE, ft))
                elif tnode.type == "interface_type":
                    for c in _walk(tnode):
                        if c.type == "method_elem" or c.type == "method_spec":
                            mn = _field(c, "name")
                            mname = _clean_ident(_txt(mn)) if mn is not None else None
                            if mname:
                                m = method_concept(tname, mname)
                                bonds.append((cls, REL_HAS_METHOD, m))
                                _go_params(m, _field(c, "parameters"), bonds)
    return bonds


# ===========================================================================
# Java
# ===========================================================================
def _java_collect_calls(body, out):
    noise = _noise_for("java")
    seen = set()
    if body is None:
        return out
    for n in _walk(body):
        if n.type == "method_invocation":
            nm = _field(n, "name")
            mname = _clean_ident(_txt(nm)) if nm is not None else None
            if not mname:
                continue
            obj = _field(n, "object")
            name = mname
            if obj is not None and obj.type == "identifier":
                oname = _clean_ident(_txt(obj))
                if oname:
                    name = "%s.%s" % (oname, mname)
            if name.split(".")[-1] in noise or name.split(".")[0] in noise:
                continue
            if name not in seen:
                seen.add(name)
                out.append(name)
    return out


def _java_params(subject, params_node, bonds):
    if params_node is None:
        return
    for p in params_node.named_children:
        if p.type not in ("formal_parameter", "spread_parameter"):
            continue
        nm = _field(p, "name")
        name = _clean_ident(_txt(nm)) if nm is not None else None
        if not name:
            # spread_parameter puts the name in a variable_declarator
            for c in _walk(p):
                if c.type == "identifier":
                    name = _clean_ident(_txt(c))
                    break
        if not name:
            continue
        bonds.append((subject, REL_HAS_PARAMETER, name))
        t = _first_type_name(_field(p, "type"))
        if t:
            bonds.append((name, REL_HAS_TYPE, t))


def _java_class_body(node, cls, owner, bonds):
    body = _field(node, "body")
    if body is None:
        return
    for item in body.named_children:
        if item.type == "method_declaration":
            nm = _field(item, "name")
            mname = _clean_ident(_txt(nm)) if nm is not None else None
            if not mname:
                continue
            m = method_concept(owner, mname)
            bonds.append((cls, REL_HAS_METHOD, m))
            _java_params(m, _field(item, "parameters"), bonds)
            t = _first_type_name(_field(item, "type"))
            if t and t != "void":
                bonds.append((m, REL_RETURNS, t))
            for callee in _java_collect_calls(_field(item, "body"), []):
                bonds.append((m, REL_CALLS, callee))
        elif item.type == "field_declaration":
            t = _first_type_name(_field(item, "type"))
            for vd in item.named_children:
                if vd.type == "variable_declarator":
                    nm = _field(vd, "name")
                    aname = _clean_ident(_txt(nm)) if nm is not None else None
                    if not aname:
                        continue
                    bonds.append((cls, REL_HAS_ATTRIBUTE, aname))
                    if t:
                        bonds.append((aname, REL_HAS_TYPE, t))


def _extract_java(root, lang):
    bonds = []
    for n in _walk(root):
        if n.type == "class_declaration":
            nm = _field(n, "name")
            cname = _clean_ident(_txt(nm)) if nm is not None else None
            if not cname:
                continue
            cls = class_concept(cname)
            sc = _field(n, "superclass")
            if sc is not None:
                bn = _first_type_name(sc)
                if bn:
                    bonds.append((cls, REL_INHERITS, class_concept(bn)))
            iff = _field(n, "interfaces")
            if iff is not None:
                for c in _walk(iff):
                    if c.type == "type_identifier":
                        bn = _clean_ident(_txt(c))
                        if bn:
                            bonds.append((cls, REL_IMPLEMENTS, class_concept(bn)))
            _java_class_body(n, cls, cname, bonds)
        elif n.type == "interface_declaration":
            nm = _field(n, "name")
            cname = _clean_ident(_txt(nm)) if nm is not None else None
            if not cname:
                continue
            _java_class_body(n, class_concept(cname), cname, bonds)
    return bonds


# ===========================================================================
# C / C++
# ===========================================================================
def _c_declarator_name(decl):
    """Walk a (possibly nested) declarator to its function/identifier name."""
    if decl is None:
        return None
    node = decl
    for _ in range(8):
        if node is None:
            return None
        if node.type in ("identifier", "field_identifier", "type_identifier"):
            return _clean_ident(_txt(node))
        if node.type == "function_declarator":
            node = _field(node, "declarator")
            continue
        if node.type in ("pointer_declarator", "reference_declarator",
                         "parenthesized_declarator", "init_declarator"):
            node = _field(node, "declarator")
            continue
        if node.type == "qualified_identifier":
            nm = _field(node, "name")
            return _clean_ident(_txt(nm)) if nm is not None else _last_seg(_txt(node))
        inner = _field(node, "declarator")
        if inner is None:
            return None
        node = inner
    return None


def _c_func_declarator(decl):
    """Find the function_declarator inside a declarator subtree."""
    if decl is None:
        return None
    if decl.type == "function_declarator":
        return decl
    for c in _walk(decl):
        if c.type == "function_declarator":
            return c
    return None


def _c_params(subject, func_decl, bonds):
    if func_decl is None:
        return
    params = _field(func_decl, "parameters")
    if params is None:
        return
    for p in params.named_children:
        if p.type != "parameter_declaration":
            continue
        t = _first_type_name(_field(p, "type"))
        name = _c_declarator_name(_field(p, "declarator"))
        if not name:
            continue
        bonds.append((subject, REL_HAS_PARAMETER, name))
        if t:
            bonds.append((name, REL_HAS_TYPE, t))


def _c_callee(call_node):
    f = _field(call_node, "function")
    if f is None:
        return None
    if f.type == "identifier":
        return _clean_ident(_txt(f))
    if f.type == "field_expression":
        fld = _field(f, "field")
        return _clean_ident(_txt(fld)) if fld is not None else None
    if f.type == "qualified_identifier":
        nm = _field(f, "name")
        return _clean_ident(_txt(nm)) if nm is not None else None
    return None


def _c_collect_calls(body, lang, out):
    noise = _noise_for(lang)
    seen = set()
    if body is None:
        return out
    for n in _walk(body):
        if n.type == "call_expression":
            name = _c_callee(n)
            if not name or name in noise:
                continue
            if name not in seen:
                seen.add(name)
                out.append(name)
    return out


def _c_struct_fields(struct_node, cls, bonds):
    body = _field(struct_node, "body")
    if body is None:
        return
    for item in body.named_children:
        if item.type != "field_declaration":
            continue
        t = _first_type_name(_field(item, "type"))
        # a field_declaration may declare several field_identifiers
        got = False
        for c in item.named_children:
            if c.type in ("field_identifier",):
                aname = _clean_ident(_txt(c))
                if aname:
                    bonds.append((cls, REL_HAS_ATTRIBUTE, aname))
                    if t:
                        bonds.append((aname, REL_HAS_TYPE, t))
                    got = True
            elif c.type in ("pointer_declarator", "array_declarator"):
                aname = _c_declarator_name(c)
                if aname:
                    bonds.append((cls, REL_HAS_ATTRIBUTE, aname))
                    if t:
                        bonds.append((aname, REL_HAS_TYPE, t))
                    got = True
        _ = got


def _extract_c(root, lang):
    bonds = []
    is_cpp = lang == "cpp"
    for n in _walk(root):
        if n.type == "function_definition":
            # skip methods defined inside a class body (handled below for C++)
            fd = _c_func_declarator(_field(n, "declarator"))
            fname = _c_declarator_name(fd) if fd is not None else None
            if not fname:
                continue
            # is this nested in a class/struct body? attribute to owner if so
            owner = None
            if is_cpp:
                anc = n.parent
                while anc is not None:
                    if anc.type in ("class_specifier", "struct_specifier"):
                        onm = _field(anc, "name")
                        owner = _clean_ident(_txt(onm)) if onm is not None else None
                        break
                    anc = anc.parent
            if owner:
                cls = class_concept(owner)
                m = method_concept(owner, fname)
                bonds.append((cls, REL_HAS_METHOD, m))
                _c_params(m, fd, bonds)
                rt = _first_type_name(_field(n, "type"))
                if rt and rt not in ("void",):
                    bonds.append((m, REL_RETURNS, rt))
                for callee in _c_collect_calls(_field(n, "body"), lang, []):
                    bonds.append((m, REL_CALLS, callee))
            else:
                fn = func_concept(fname)
                _c_params(fn, fd, bonds)
                rt = _first_type_name(_field(n, "type"))
                if rt and rt not in ("void",):
                    bonds.append((fn, REL_RETURNS, rt))
                for callee in _c_collect_calls(_field(n, "body"), lang, []):
                    bonds.append((fn, REL_CALLS, callee))
        elif n.type in ("struct_specifier", "class_specifier", "union_specifier"):
            nm = _field(n, "name")
            cname = _clean_ident(_txt(nm)) if nm is not None else None
            if not cname:
                continue
            cls = class_concept(cname)
            # C++ base classes
            if is_cpp:
                for c in n.children:
                    if c.type == "base_class_clause":
                        for t in c.named_children:
                            bn = _first_type_name(t)
                            if bn:
                                bonds.append((cls, REL_INHERITS, class_concept(bn)))
            _c_struct_fields(n, cls, bonds)
            # C++ method DECLARATIONS inside class body (no definition)
            if is_cpp:
                body = _field(n, "body")
                if body is not None:
                    for item in body.named_children:
                        if item.type == "field_declaration":
                            fd = _c_func_declarator(_field(item, "declarator"))
                            if fd is not None:
                                mname = _c_declarator_name(fd)
                                if mname:
                                    m = method_concept(cname, mname)
                                    bonds.append((cls, REL_HAS_METHOD, m))
                                    _c_params(m, fd, bonds)
                                    rt = _first_type_name(_field(item, "type"))
                                    if rt and rt not in ("void",):
                                        bonds.append((m, REL_RETURNS, rt))
    return bonds


# ===========================================================================
# Dispatch
# ===========================================================================
_EXT_LANG = {
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript", ".mts": "typescript", ".cts": "typescript",
    ".tsx": "tsx",
    ".rs": "rust",
    ".go": "go",
    ".java": "java",
    ".c": "c", ".h": "c",
    ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp", ".c++": "cpp",
    ".hpp": "cpp", ".hh": "cpp", ".hxx": "cpp", ".h++": "cpp",
}

_EXTRACTORS = {
    "javascript": _extract_js,
    "typescript": _extract_js,
    "tsx": _extract_js,
    "rust": _extract_rust,
    "go": _extract_go,
    "java": _extract_java,
    "c": _extract_c,
    "cpp": _extract_c,
}


def lang_for_ext(ext):
    return _EXT_LANG.get(ext.lower())


def supported_exts():
    return sorted(_EXT_LANG.keys())


def supported_langs():
    return sorted(set(_EXT_LANG.values()))


def extract_triples(source, ext):
    """Parse `source` for the language implied by `ext` and return a list of
    (subject, relation, object) string triples. Python is delegated to the
    stdlib-ast pycode_extractor for parity. Returns [] for unsupported exts.
    Raises nothing for parse failures -- tree-sitter is error-tolerant; the
    clean-name gates drop any garbage that results."""
    ext = ext.lower()
    if ext == ".py":
        import importlib.util
        import pathlib
        p = pathlib.Path(__file__).resolve().parent / "pycode_extractor.py"
        spec = importlib.util.spec_from_file_location("pycode_extractor", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.extract_triples(source)
    lang = _EXT_LANG.get(ext)
    if lang is None:
        return []
    parser = _get_parser(lang)
    tree = parser.parse(bytes(source, "utf-8"))
    extractor = _EXTRACTORS[lang]
    raw = extractor(tree.root_node, lang)
    # Final structural-only gate: subject/object must be clean concepts.
    out = []
    for subj, rel, obj in raw:
        if _is_clean_concept(subj) and _is_clean_concept(obj):
            out.append((subj, rel, obj))
    return out


_CONCEPT_WRAP_RE = re.compile(
    r"^the [A-Za-z_$][A-Za-z0-9_$]* (?:class|function)$")


def _is_clean_concept(s):
    """True iff s is a bare ident, a dotted ident (Owner.method), or a wrapped
    'the X class' / 'the X function' concept. This is the last line of defence
    against raw code text leaking into a bond."""
    if not s or "\n" in s or "\t" in s:
        return False
    if _CONCEPT_WRAP_RE.match(s):
        return True
    return bool(_clean_qual(s))


def extract_bonds_from_source(source, source_url, ext):
    """Wrapper returning BatchSink-shaped dicts (corpus stamped by caller)."""
    out = []
    for subj, rel, obj in extract_triples(source, ext):
        out.append({
            "subject": subj,
            "relation": rel,
            "object": obj,
            "weight": DEFAULT_WEIGHT,
            "confidence": DEFAULT_CONFIDENCE,
            "source_url": source_url,
        })
    return out


if __name__ == "__main__":
    import sys
    tgt = sys.argv[1]
    ext = os.path.splitext(tgt)[1]
    src = open(tgt, encoding="utf-8", errors="replace").read()
    for s, r, o in extract_triples(src, ext):
        print("[%s] --%s--> [%s]" % (s, r, o))
