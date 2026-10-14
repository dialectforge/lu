#!/usr/bin/env python3
"""Phase-1 Python structural code extractor (zero external deps).

Walks a .py file's AST with the stdlib `ast` module and emits clean
(subject, relation, object) structural triples that fit Lu's BondType
schema. The relations used here are the Phase-1 code-structure variants
appended to `BondType` in src/bond.rs:

    HasMethod, HasAttribute, HasParameter, Returns,
    Inherits, Calls, HasType, Implements

Naming discipline (John-approved LOCKED design):
  * classes        -> "the Dog class"
  * functions      -> "the foo function"   (module-level)
  * methods        -> "Dog.bark"           (qualified by owning class)
  * params         -> bare name "loud"     (object of HasParameter)
  * types          -> bare type name "bool", "str"
  * call targets    -> bare callee name "open", "list.append"

The extractor is deliberately conservative: it only emits a triple when
it can name both ends cleanly. Anything it can't resolve to a clean
concept (e.g. a call on an arbitrary expression) is skipped rather than
emitting noise.

Returned bonds are dicts shaped for ingest_curriculum_full.BatchSink:
    {"subject":..,"relation":..,"object":..,"weight":..,"confidence":..}
The caller stamps corpus/run_id/source_url.
"""
import ast
import os

# Relation tag strings -- MUST match the CamelCase names accepted by
# Lu's parse_bond_type / bondtype_from_relation_str (src/brain.rs,
# src/extractor/types.rs). Phase-1 appended these to BondType.
REL_HAS_METHOD = "HasMethod"
REL_HAS_ATTRIBUTE = "HasAttribute"
REL_HAS_PARAMETER = "HasParameter"
REL_RETURNS = "Returns"
REL_INHERITS = "Inherits"
REL_CALLS = "Calls"
REL_HAS_TYPE = "HasType"
REL_IMPLEMENTS = "Implements"  # reserved for ABC/Protocol bases (future)

DEFAULT_WEIGHT = 0.8
DEFAULT_CONFIDENCE = "Verified"  # source code is ground-truth structure

# Params we never emit as HasParameter (receiver slots add no signal).
_SKIP_PARAMS = {"self", "cls"}


def class_concept(name):
    return "the %s class" % name


def func_concept(name):
    return "the %s function" % name


def method_concept(cls_name, meth_name):
    return "%s.%s" % (cls_name, meth_name)


def _type_name(annotation):
    """Render a type annotation AST node to a clean concept string.

    Handles the common shapes: bare names (`int`), dotted attributes
    (`typing.List`), subscripts (`List[int]` -> "List"), string
    forward-refs, and `None`. Returns None when the annotation can't be
    rendered cleanly (so the caller skips it rather than emit garbage).
    """
    if annotation is None:
        return None
    if isinstance(annotation, ast.Name):
        return annotation.id
    if isinstance(annotation, ast.Attribute):
        # typing.Optional -> "Optional" (last segment is the useful concept)
        return annotation.attr
    if isinstance(annotation, ast.Subscript):
        # List[int] / Optional[str] -> name the container type only
        return _type_name(annotation.value)
    if isinstance(annotation, ast.Constant):
        # `-> None` is ast.Constant(value=None); string forward refs too
        if annotation.value is None:
            return "None"
        if isinstance(annotation.value, str):
            return annotation.value.strip() or None
        return None
    if isinstance(annotation, ast.BinOp) and isinstance(annotation.op, ast.BitOr):
        # PEP 604  `int | None` -> name the left operand (primary type)
        return _type_name(annotation.left)
    return None


def _base_name(node):
    """Render a base-class expression to a bare class name, or None."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Subscript):  # Generic[T], Protocol[...] bases
        return _base_name(node.value)
    return None


def _callee_name(node):
    """Render a Call target to a bare callee concept, or None.

    `foo()`        -> "foo"
    `obj.method()` -> "obj.method"  (single-level receiver kept for signal)
    `a.b.c()`      -> "c"           (deep chains collapse to the method)
    """
    f = node.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        recv = f.value
        if isinstance(recv, ast.Name):
            return "%s.%s" % (recv.id, f.attr)
        return f.attr
    return None


# Builtins / dunder-internal callees that add noise rather than structure.
_CALL_NOISE = {
    "len", "str", "repr", "int", "isinstance", "type", "super",
    "getattr", "setattr", "hasattr", "print", "format",
}


def _collect_calls(fn_node):
    """Distinct clean callee names invoked in a function body."""
    seen = []
    seen_set = set()
    for n in ast.walk(fn_node):
        if isinstance(n, ast.Call):
            name = _callee_name(n)
            if not name:
                continue
            base = name.split(".")[-1]
            if base in _CALL_NOISE:
                continue
            if name not in seen_set:
                seen_set.add(name)
                seen.append(name)
    return seen


def _emit_params_and_return(subject, fn_node, bonds):
    """Emit HasParameter / HasType / Returns for a function/method node."""
    a = fn_node.args
    # positional + positional-only + keyword-only args carry annotations
    all_args = list(getattr(a, "posonlyargs", [])) + list(a.args) + list(a.kwonlyargs)
    for arg in all_args:
        if arg.arg in _SKIP_PARAMS:
            continue
        bonds.append((subject, REL_HAS_PARAMETER, arg.arg))
        t = _type_name(arg.annotation)
        if t:
            bonds.append((arg.arg, REL_HAS_TYPE, t))
    rt = _type_name(fn_node.returns)
    if rt and rt != "None":
        bonds.append((subject, REL_RETURNS, rt))


def extract_triples(source, filename="<unknown>"):
    """Parse Python `source` and return a list of (subject, relation,
    object) string triples. Raises SyntaxError on unparseable input --
    the caller catches and counts it.
    """
    tree = ast.parse(source, filename=filename)
    bonds = []

    for node in tree.body:
        # ---- module-level class ----
        if isinstance(node, ast.ClassDef):
            cls = class_concept(node.name)
            # inheritance (skip object / Generic-only bases handled by name)
            for base in node.bases:
                bn = _base_name(base)
                if bn and bn != "object":
                    bonds.append((cls, REL_INHERITS, class_concept(bn)))
            # methods + attributes
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    # keep __init__; drop other dunders to cut clutter
                    if item.name.startswith("__") and item.name != "__init__":
                        continue
                    m = method_concept(node.name, item.name)
                    bonds.append((cls, REL_HAS_METHOD, m))
                    _emit_params_and_return(m, item, bonds)
                    for callee in _collect_calls(item):
                        bonds.append((m, REL_CALLS, callee))
                elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    # class-level annotated attribute:  x: int
                    attr = item.target.id
                    if attr.startswith("__"):
                        continue
                    bonds.append((cls, REL_HAS_ATTRIBUTE, attr))
                    t = _type_name(item.annotation)
                    if t:
                        bonds.append((attr, REL_HAS_TYPE, t))
                elif isinstance(item, ast.Assign):
                    for tgt in item.targets:
                        if isinstance(tgt, ast.Name) and not tgt.id.startswith("__"):
                            bonds.append((cls, REL_HAS_ATTRIBUTE, tgt.id))

        # ---- module-level function ----
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("__"):
                continue
            fn = func_concept(node.name)
            _emit_params_and_return(fn, node, bonds)
            for callee in _collect_calls(node):
                bonds.append((fn, REL_CALLS, callee))

    return bonds


def extract_bonds_from_pyfile(source, source_url, filename="<unknown>"):
    """Wrapper returning BatchSink-shaped dicts (corpus stamped by caller)."""
    out = []
    for subj, rel, obj in extract_triples(source, filename=filename):
        if not subj or not obj:
            continue
        out.append({
            "subject": subj,
            "relation": rel,
            "object": obj,
            "weight": DEFAULT_WEIGHT,
            "confidence": DEFAULT_CONFIDENCE,
            "source_url": source_url,
        })
    return out


def iter_py_files(root):
    """Yield .py file paths under `root` in sorted order (resume-stable).

    Skips test dirs and dunder-internal packages that add little
    structural signal and a lot of noise.
    """
    if os.path.isfile(root) and root.endswith(".py"):
        yield root
        return
    skip_dir = {"test", "tests", "__pycache__", "site-packages", "dist-packages"}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in skip_dir)
        for fn in sorted(filenames):
            if fn.endswith(".py"):
                yield os.path.join(dirpath, fn)


def file_source_url(root, path):
    rel = os.path.relpath(path, root) if os.path.isdir(root) else os.path.basename(path)
    return "pycode://" + rel.replace(os.sep, "/")


if __name__ == "__main__":
    import sys
    tgt = sys.argv[1] if len(sys.argv) > 1 else __file__
    src = open(tgt, encoding="utf-8", errors="replace").read()
    for s, r, o in extract_triples(src, filename=tgt):
        print("[%s] --%s--> [%s]" % (s, r, o))
