#!/usr/bin/env python3
"""
Lu BR.0 Preflight Harness.

Runs BEFORE any 1+ hour ingest. Refuses to launch if any contract
between the Python extractor and the Rust binary, or between the
built binary and the deployed binary, or between the daemon launch
script and the safety flags, is misaligned.

Designed to be cheap (seconds, not minutes) and HERMETIC: the only
persistent state it touches is read-only inspection of source files
and binaries.  All MCP probes use --no-save against an ephemeral
tmp brain that is deleted on exit.

Usage:
    python3 scripts/preflight.py                # all phases
    python3 scripts/preflight.py --phase A      # just vocab
    python3 scripts/preflight.py --phase B      # just deploy
    python3 scripts/preflight.py --phase C      # just smoke
    python3 scripts/preflight.py --keep-going   # don't stop on first FAIL

Exits 0 if every executed check PASSes, non-zero on first FAIL
(unless --keep-going, in which case exit code = number of FAILs).
STUB checks pass with a noisy notice on stderr; they're tracked
separately in the summary so John knows what's deferred.

Audit punch-list coverage internal notes:
    A.1  BondType vocabulary mismatch              -> punch item #1
    A.2  NodeType vocabulary mismatch              -> punch item #2
    A.3  KnowledgeContent kind always emitted      -> punch item #3 (BR.3 LIVE)
    A.4  SourceKind producer-consumer subset       -> punch item #4
    A.5  Cross-language extension fields policy    -> punch item #5
    B.1  Binary md5 deployed == built              -> punch item #6
    B.2  Daemon launch flag audit (no autosave)    -> punch item #7
    B.3  RuntimeConfig setter coverage             -> punch item #8
    B.4  Working tree cleanliness                  -> punch item #9
    C.1  50-article ingest health                  -> punch item #10
    C.2  Dual-mode typed-KU smoke                  -> punch item #11
"""

import argparse
import hashlib
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# -----------------------------------------------------------------
# Repo paths -- preflight runs from anywhere but resolves these as
# relative to its own location so John can `python3 scripts/preflight.py`
# from the repo root.
# -----------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent
SRC_DIR    = REPO_ROOT / "src"
EXTRACTOR  = SCRIPT_DIR / "smart_extractor.py"

LOCAL_BINARY    = REPO_ROOT / "target" / "release" / "lu4d"
DEPLOYED_BINARY = Path("/usr/local/bin/lu_4_mcp")
SYSTEMD_UNIT    = Path("/etc/systemd/system/lu.service")

# Shell scripts under the dialectforge home that may launch the
# binary in a long-running ingest configuration.  We audit ALL of
# them rather than guess which one is canonical.
LAUNCH_SCRIPT_GLOBS = [
    Path("/opt/lu").glob("start_*.sh"),
    Path("/opt/lu").glob("lu_train_v*.py"),
]


# -----------------------------------------------------------------
# Tiny check-runner harness.  A check is a function (name -> Result).
# Each Result records pass/fail/stub plus a short human-readable
# detail message.  The harness prints a deterministic per-check
# line and an aggregate summary at the end.  Per-phase grouping is
# preserved.  STUB results are not failures -- they're tracked
# separately so John can audit what's deferred.
# -----------------------------------------------------------------
PASS = "PASS"
FAIL = "FAIL"
STUB = "STUB"


@dataclass
class Result:
    name: str
    status: str
    detail: str
    extra: Dict[str, Any] = field(default_factory=dict)


def _emoji(status: str) -> str:
    # Emoji-free status markers -- John's repo style is plain ASCII.
    return {"PASS": "[ok]", "FAIL": "[FAIL]", "STUB": "[stub]"}.get(status, "[?]")


def print_result(r: Result) -> None:
    line = f"{_emoji(r.status):>6} {r.name:<48} {r.detail}"
    print(line)


# =================================================================
# PHASE A -- Vocabulary contract checks
# =================================================================

# These are the relation strings the v2 (iter-4) smart_extractor emits.
# Source: scripts/smart_extractor.py lines 215-310 (_L3_PATTERNS) and
# 392-405 (_L5_PATTERNS), plus the constant strings in extract_see_also
# (RelatedTo), extract_section_headings (HasAspect), extract_category_tail
# (IsA), and extract_body_comentions (RelatedTo).
#
# We re-derive them dynamically below by scanning the extractor source
# rather than hard-coding so the check stays accurate as the extractor
# evolves.  Hard-coding here is documentation only.
_DOCUMENTED_EXTRACTOR_RELATIONS_v2_iter4 = sorted({
    "RelatedTo", "HasAspect", "IsA",
    "LocatedIn", "PartOf", "FoundedBy", "FoundedIn",
    "DiscoveredBy", "NamedAfter", "MeasuredIn",
    "BornIn", "DiedIn", "WrittenBy", "ComposedOf",
    "KnownFor", "UsedFor",
    "HasPopulation", "HasArea", "HasAltitude",
    "HasOfficialLanguage", "HasCapital",
    "BornOn", "DiedOn",
})


def extract_relations_from_extractor(extractor_path: Path) -> List[str]:
    """Parse smart_extractor.py and return every literal relation string
    it can emit.  Looks for two patterns:
        "relation": "X"
        (RX, "X", weight, ...)            # _L3_PATTERNS / _L5_PATTERNS rows
    Both styles appear in the iter-4 extractor."""
    src = extractor_path.read_text(encoding="utf-8")
    rels: set = set()
    # Style 1: literal dict entries.  These are the L1, L2, L2b, L4 layers.
    for m in re.finditer(r'"relation"\s*:\s*"([A-Za-z][A-Za-z0-9_]+)"', src):
        rels.add(m.group(1))
    # Style 2: tuple rows in the _L3_PATTERNS / _L5_PATTERNS lists.
    for m in re.finditer(
        r'\(_RX_[A-Z_]+,\s*"([A-Za-z][A-Za-z0-9_]+)"\s*,', src,
    ):
        rels.add(m.group(1))
    return sorted(rels)


class MCPClient:
    """Minimal long-lived JSON-RPC client over stdio.  Forks lu_4_mcp
    as a subprocess in --no-save mode against a fresh tmp brain so
    the canonical brain on disk is never touched.

    Modeled after lu_train_v4.py's McpClient but stripped to only
    the calls the preflight harness needs: initialize, tools/call,
    shutdown.  No background threads beyond the stdout reader."""

    def __init__(self, binary: Path, brain_path: Path, *, timeout_init: float = 10.0) -> None:
        self.proc = subprocess.Popen(
            [
                str(binary),
                "--mcp",
                "--brain", str(brain_path),
                "--no-save",
                # Keep the tmp brain in plaintext-compressed form; we never
                # save it anyway because of --no-save, but the format flag
                # silences any "detected encrypted" warnings on first read.
                "--brain-format-out", "v3c",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        self._q: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self._lock = threading.Lock()
        self._req_id = 0
        threading.Thread(target=self._read_loop, daemon=True).start()
        threading.Thread(target=self._err_drain, daemon=True).start()
        # Wait for the server to settle.  If it dies immediately we want a
        # crisp error, not a JSON-RPC timeout.
        time.sleep(0.4)
        if self.proc.poll() is not None:
            err = b""
            try: err = self.proc.stderr.read(4096)
            except Exception: pass
            raise RuntimeError(f"lu_4_mcp exited immediately: {err!r}")
        # Drive the MCP handshake.  Some tools error pre-initialize.
        self._initialize(timeout_init)

    def _read_loop(self) -> None:
        try:
            for raw in self.proc.stdout:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line: continue
                try:
                    self._q.put(json.loads(line))
                except json.JSONDecodeError:
                    pass
        except Exception:
            pass

    def _err_drain(self) -> None:
        # Drain stderr so the subprocess pipe never fills up.  We don't
        # surface lines unless a check explicitly asks for them.
        try:
            for _ in self.proc.stderr:
                pass
        except Exception:
            pass

    def _next_id(self) -> int:
        with self._lock:
            self._req_id += 1
            return self._req_id

    def send(self, method: str, params: Any, *, timeout: float = 30.0) -> Dict[str, Any]:
        rid = self._next_id()
        req = {"jsonrpc": "2.0", "id": rid, "method": method, "params": params}
        try:
            self.proc.stdin.write((json.dumps(req, ensure_ascii=False) + "\n").encode("utf-8"))
            self.proc.stdin.flush()
        except (OSError, BrokenPipeError) as e:
            return {"error": {"message": str(e)}}
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                resp = self._q.get(timeout=min(5.0, deadline - time.time()))
                if resp.get("id") == rid:
                    return resp
            except queue.Empty:
                if self.proc.poll() is not None:
                    return {"error": {"message": "lu_4_mcp died"}}
        return {"error": {"message": f"timeout {timeout:.0f}s on {method}"}}

    def notify(self, method: str, params: Any) -> None:
        msg = {"jsonrpc": "2.0", "method": method, "params": params}
        try:
            self.proc.stdin.write((json.dumps(msg, ensure_ascii=False) + "\n").encode("utf-8"))
            self.proc.stdin.flush()
        except Exception:
            pass

    def tool_call(self, name: str, arguments: Dict[str, Any], *, timeout: float = 30.0) -> Dict[str, Any]:
        return self.send("tools/call", {"name": name, "arguments": arguments}, timeout=timeout)

    def list_tools(self) -> List[Dict[str, Any]]:
        resp = self.send("tools/list", {}, timeout=10.0)
        if "result" in resp:
            return resp["result"].get("tools", [])
        return []

    def _initialize(self, timeout: float) -> None:
        resp = self.send("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "preflight", "version": "BR.0"},
        }, timeout=timeout)
        if "error" in resp:
            raise RuntimeError(f"initialize failed: {resp['error']}")
        self.notify("notifications/initialized", {})

    def close(self) -> None:
        try:
            self.send("shutdown", {}, timeout=2.0)
        except Exception:
            pass
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=3.0)
        except Exception:
            try: self.proc.kill()
            except Exception: pass


def _ephemeral_brain_dir() -> Path:
    """Make a fresh tmp dir for an ephemeral brain.  Caller is responsible
    for cleanup -- typically with shutil.rmtree in a finally clause.
    Hermetic: NEVER returns /var/lib/lu/* or anything under the canonical
    brain dir."""
    return Path(tempfile.mkdtemp(prefix="lu_preflight_", suffix=".brain"))


def check_a1_bondtype(binary: Path) -> Result:
    """A.1 -- BondType: every relation the extractor emits must parse via
    the production binary's parse_bond_type.  We don't call parse_bond_type
    directly because it's a private fn; we exercise it transitively
    through lu4d_teach with a discardable subject/object pair.  An
    ephemeral --no-save brain is used so the canonical brain is never
    touched.

    Today (binary 5b71af7 / 0dff89b) BondType has 11 variants plus the
    Q.1 typed aliases.  The extractor emits ~23 distinct relations.  We
    EXPECT ~17 of these to fail to parse -- that's the BR.0 motivation.
    The rebuild's job is to make this check pass."""
    if not binary.exists():
        return Result("A.1 BondType", FAIL, f"binary missing: {binary}")
    if not EXTRACTOR.exists():
        return Result("A.1 BondType", FAIL, f"extractor missing: {EXTRACTOR}")
    rels = extract_relations_from_extractor(EXTRACTOR)
    brain_dir = _ephemeral_brain_dir()
    brain_path = brain_dir / "preflight.lbr"
    accepted: List[str] = []
    rejected: List[Tuple[str, str]] = []
    try:
        client = MCPClient(binary, brain_path)
        try:
            for rel in rels:
                # Probe by issuing a teach call with throwaway subject/object.
                # parse_bond_type is the FIRST validation step inside
                # tool_teach, so a -32602 response with the exact error
                # text from BrainError::ParseBondType is a clean signal.
                resp = client.tool_call(
                    "lu4d_teach",
                    {
                        "subject": f"_preflight_subject_{rel}",
                        "relation": rel,
                        "object":  f"_preflight_object_{rel}",
                        "weight":  0.5,
                        "confidence": "Unverified",
                    },
                    timeout=10.0,
                )
                err = resp.get("error") or {}
                msg = err.get("message", "")
                # parse_bond_type returns BrainError::ParseBondType which
                # surfaces in the JSON-RPC error message as something like
                # "unknown bond type: \"LocatedIn\"" depending on Display
                # impl.  We treat ANY -32602 whose text mentions the
                # relation as a parse rejection.
                if err.get("code") == -32602 and rel.lower() in msg.lower():
                    rejected.append((rel, msg))
                elif "error" in resp:
                    # Non-parse error (auth, brain state).  Treat as a
                    # check failure rather than a vocabulary rejection.
                    rejected.append((rel, f"[non-parse-error] {msg}"))
                else:
                    accepted.append(rel)
        finally:
            client.close()
    finally:
        shutil.rmtree(brain_dir, ignore_errors=True)

    if not rejected:
        return Result(
            "A.1 BondType", PASS,
            f"all {len(rels)} extractor relations parse on Rust side",
            {"accepted": accepted, "rejected": []},
        )
    rejected_names = sorted(r[0] for r in rejected)
    return Result(
        "A.1 BondType", FAIL,
        f"{len(rejected)}/{len(rels)} extractor relations not in Rust BondType: {rejected_names}",
        {"accepted": accepted, "rejected": [{"rel": r, "err": e} for r, e in rejected]},
    )


def check_a2_nodetype() -> Result:
    """A.2 -- NodeType: every NodeType the extractor emits must be in the
    Rust NodeType enum.  Today: the extractor does NOT emit a node_type
    field at all (subject/object are bare strings; the Rust side calls
    KnowledgeContent::Concept by default).  And the Rust side has no
    standalone NodeType enum -- the analogue is KnowledgeContent.

    So this check is structurally a no-op today: there are no extractor
    NodeType values to assert against the Rust enum.  We document that
    fact rather than silently pass.  The rebuild may introduce a
    NodeType field on emitted bonds; when it does, this check upgrades
    to a real subset assertion."""
    if not EXTRACTOR.exists():
        return Result("A.2 NodeType", FAIL, f"extractor missing: {EXTRACTOR}")
    src = EXTRACTOR.read_text(encoding="utf-8")
    # Look for any '"node_type": "X"' or '"kind": "X"' style fields.
    nodetype_emits = re.findall(r'"(?:node_type|nodeType)"\s*:\s*"([A-Za-z0-9_]+)"', src)
    if not nodetype_emits:
        return Result(
            "A.2 NodeType", PASS,
            "extractor emits no node_type field (Rust default applies); upgrades to subset check post-rebuild",
            {"emitted": []},
        )
    # If the extractor DOES emit node_type, check each value lives in Rust.
    node_rs = (SRC_DIR / "node.rs").read_text(encoding="utf-8")
    # Try both NodeType enum (post-rebuild) and KnowledgeContent (today).
    rust_variants: set = set()
    for m in re.finditer(r'^\s*([A-Z][A-Za-z0-9_]+)(?:\s*\(|,)', node_rs, re.MULTILINE):
        rust_variants.add(m.group(1))
    missing = [n for n in set(nodetype_emits) if n not in rust_variants]
    if missing:
        return Result(
            "A.2 NodeType", FAIL,
            f"extractor emits node types not in Rust: {sorted(missing)}",
            {"emitted": sorted(set(nodetype_emits)), "missing": sorted(missing)},
        )
    return Result(
        "A.2 NodeType", PASS,
        f"all {len(set(nodetype_emits))} extractor node types accepted by Rust",
        {"emitted": sorted(set(nodetype_emits))},
    )


def check_a3_kind_presence() -> Result:
    """A.3 -- KnowledgeContent kind ALWAYS emitted by the extractor.

    BR.3: the extractor now emits a `kind` field on every bond.  This
    check enforces three properties on a 50-article sample:
      * 100% coverage   -- every bond has a non-empty `kind`.
      * concentration   -- no single kind exceeds 90% of the sample
                           (otherwise the typed-KU storage gains zero
                           routing leverage from the new field).
      * diversity       -- at least 4 distinct kinds appear (a healthy
                           extractor exercises definition + composition
                           + purpose + fact + concept; <4 means the
                           mapping has collapsed).

    The 50-article seed is the same _SMOKE_ARTICLES used by C.1, so the
    Phase A and Phase C checks share a fixture and any drift surfaces
    in both places.  This check is hermetic -- no daemon, no binary,
    just the Python extractor."""
    sys.path.insert(0, str(SCRIPT_DIR))
    try:
        from smart_extractor import extract_bonds_from_wikipedia_article  # type: ignore
    except Exception as e:
        return Result("A.3 KnowledgeContent kind", FAIL, f"cannot import smart_extractor: {e}")

    _seed_smoke_articles()
    if not _SMOKE_ARTICLES:
        return Result("A.3 KnowledgeContent kind", FAIL, "smoke article fixture empty")

    histogram: Counter = Counter()
    total = 0
    missing = 0
    for art in _SMOKE_ARTICLES:
        bonds = extract_bonds_from_wikipedia_article(art)
        for b in bonds:
            total += 1
            kind = b.get("kind")
            if not kind or not isinstance(kind, str):
                missing += 1
                continue
            histogram[kind] += 1

    if total == 0:
        return Result(
            "A.3 KnowledgeContent kind", FAIL,
            "no bonds produced from 50-article fixture (extractor regression?)",
        )

    distinct = sum(1 for v in histogram.values() if v > 0)
    top_kind, top_count = histogram.most_common(1)[0] if histogram else ("-", 0)
    top_share = top_count / total
    coverage = (total - missing) / total

    failures = []
    if missing > 0:
        failures.append(f"{missing}/{total} bonds missing `kind` field ({coverage:.1%} coverage)")
    if distinct < 4:
        failures.append(f"only {distinct} distinct kinds populated (need >= 4)")
    if top_share > 0.9:
        failures.append(f"top kind {top_kind!r} = {top_share:.1%} > 90%")

    detail = (
        f"articles={len(_SMOKE_ARTICLES)} bonds={total} distinct_kinds={distinct} "
        f"top={top_kind}@{top_share:.1%} coverage={coverage:.1%}"
    )
    payload = {
        "histogram": dict(histogram),
        "total_bonds": total,
        "missing_kind": missing,
        "distinct_kinds": distinct,
        "top_kind": top_kind,
        "top_share": top_share,
    }
    if failures:
        return Result(
            "A.3 KnowledgeContent kind", FAIL,
            detail + " :: " + "; ".join(failures), payload,
        )
    return Result("A.3 KnowledgeContent kind", PASS, detail, payload)


def check_a4_sourcekind() -> Result:
    """A.4 -- SourceKind: every variant the producer (extractor +
    harness) ever emits must be accepted by the consumer (Rust
    `SourceKind` enum).  In practice the harness only stamps
    `TrainingCorpus { corpus, run_id, url }` and falls through to the
    Rust side which permits Research / Teach / Document / etc.  We
    check the variant *tags* used in the harness against the Rust
    enum's `tag()` method values."""
    rs_path = SRC_DIR / "source.rs"
    if not rs_path.exists():
        return Result("A.4 SourceKind", FAIL, f"source.rs missing: {rs_path}")
    rs_src = rs_path.read_text(encoding="utf-8")
    # Parse `pub enum SourceKind { ... }` for variants.
    m = re.search(r'pub enum SourceKind\s*\{(.*?)^\}', rs_src, re.DOTALL | re.MULTILINE)
    if not m:
        return Result("A.4 SourceKind", FAIL, "could not parse SourceKind enum")
    body = m.group(1)
    rust_variants = set(re.findall(r'^\s*([A-Z][A-Za-z0-9_]+)\s*[\{,]', body, re.MULTILINE))
    # Producer side: scan the harness for what variant tags it uses.
    # Today the only producer is lu4d_teach_corpus which always stamps
    # SourceKind::TrainingCorpus.  Hard-coded here because there's no
    # programmatic way to derive it without running the harness.
    producer_variants = {"TrainingCorpus"}
    missing = producer_variants - rust_variants
    if missing:
        return Result(
            "A.4 SourceKind", FAIL,
            f"producer emits variants Rust does not accept: {sorted(missing)}",
            {"producer": sorted(producer_variants), "rust": sorted(rust_variants)},
        )
    return Result(
        "A.4 SourceKind", PASS,
        f"producer variants {sorted(producer_variants)} subset of Rust {sorted(rust_variants)}",
        {"producer": sorted(producer_variants), "rust": sorted(rust_variants)},
    )


def check_a5_extension_fields() -> Result:
    """A.5 -- Cross-language extension fields policy.

    The Python extractor emits `layer` and `frequency` per bond.  The
    Rust MCP side (lu4d_teach_corpus) does NOT have these fields in
    its input schema.  Today they are silently dropped because the Rust
    JSON deserializer does not use serde's `deny_unknown_fields`.

    Policy: (a) accepted no-op on Rust side -- Python may include
    extension fields; Rust ignores them.  This avoids a brittle
    "scrub before send" step in every harness that constructs a teach
    call.  We assert the current behavior matches:
        1. The extractor DOES emit `layer` and `frequency`.
        2. The Rust source contains NO `deny_unknown_fields` directive.
    If either invariant flips, this check fails so we know the
    contract drifted."""
    if not EXTRACTOR.exists():
        return Result("A.5 Extension fields", FAIL, f"extractor missing: {EXTRACTOR}")
    py_src = EXTRACTOR.read_text(encoding="utf-8")
    py_emits_layer = '"layer":' in py_src
    py_emits_freq  = '"frequency":' in py_src
    rust_strict = False
    for rs_file in SRC_DIR.rglob("*.rs"):
        try:
            if "deny_unknown_fields" in rs_file.read_text(encoding="utf-8", errors="ignore"):
                rust_strict = True
                break
        except Exception:
            continue
    if not (py_emits_layer and py_emits_freq):
        return Result(
            "A.5 Extension fields", FAIL,
            f"extractor no longer emits layer/frequency (layer={py_emits_layer}, frequency={py_emits_freq})",
        )
    if rust_strict:
        return Result(
            "A.5 Extension fields", FAIL,
            "Rust now uses #[serde(deny_unknown_fields)]; extension-field policy invalidated",
        )
    return Result(
        "A.5 Extension fields", PASS,
        "policy (a): extension fields accepted no-op; Rust silently ignores layer/frequency",
        {"py_emits_layer": py_emits_layer, "py_emits_frequency": py_emits_freq, "rust_strict": rust_strict},
    )


# =================================================================
# PHASE B -- Configuration & deployment checks
# =================================================================

def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def check_b1_binary_md5() -> Result:
    """B.1 -- the locally-built binary must byte-equal the deployed binary.
    Otherwise we'd be ingesting against a different binary than the one
    we've been testing.  Cargo.toml names the binary `lu4d`; production
    deploys it as `/usr/local/bin/lu_4_mcp`."""
    if not LOCAL_BINARY.exists():
        return Result(
            "B.1 Binary md5", FAIL,
            f"local binary not built at {LOCAL_BINARY} (run `cargo build --release`)",
        )
    if not DEPLOYED_BINARY.exists():
        return Result(
            "B.1 Binary md5", FAIL,
            f"deployed binary missing at {DEPLOYED_BINARY}",
        )
    local_md5 = _md5(LOCAL_BINARY)
    deploy_md5 = _md5(DEPLOYED_BINARY)
    if local_md5 != deploy_md5:
        return Result(
            "B.1 Binary md5", FAIL,
            f"md5 mismatch: local={local_md5[:12]} deployed={deploy_md5[:12]} (sudo cp + restart needed)",
            {"local": local_md5, "deployed": deploy_md5},
        )
    return Result(
        "B.1 Binary md5", PASS,
        f"local == deployed ({local_md5[:12]}...)",
        {"local": local_md5, "deployed": deploy_md5},
    )


def check_b2_autosave_audit() -> Result:
    """B.2 -- audit every potential daemon launch path (systemd unit +
    home-dir launch scripts) and assert that long-running ingests pass
    `--no-save`.  save_on_every_write is computed in main.rs from
    `!no_save && brain.path.is_some()`, so the absence of `--no-save`
    means autosave is ON.  Autosave during a 1+ hour ingest fans out
    to multi-second pauses on every write; we want the harness driving
    explicit `lu4d_save` calls instead."""
    findings: List[Tuple[str, str, bool]] = []
    # Walk all candidate launch scripts.  We treat "references lu_4_mcp or
    # the local binary" as the population to audit.
    candidates: List[Path] = []
    for g in LAUNCH_SCRIPT_GLOBS:
        candidates.extend(g)
    if SYSTEMD_UNIT.exists():
        candidates.append(SYSTEMD_UNIT)

    audited = 0
    for p in candidates:
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if "lu_4_mcp" not in text and "target/release/lu4d" not in text:
            continue
        audited += 1
        # Crude but honest: if the script references the binary AND does
        # not pass --no-save anywhere, autosave is ON for that script.
        # We don't try to follow conditional branches; any mention of
        # --no-save is enough to call it audited-clean.
        no_save = "--no-save" in text
        # Special-case lu.service: it's intentionally disabled and the
        # MCP server saves on shutdown anyway -- but a 1+ hour ingest
        # should not run via this unit, full stop.
        is_systemd = (p == SYSTEMD_UNIT)
        if is_systemd:
            # The unit is documented as disabled and not used for ingest;
            # we still flag it because it's the most visible launch path.
            findings.append((str(p), "systemd unit (disabled, no --no-save)", no_save))
        else:
            findings.append((str(p), "launch script", no_save))

    if audited == 0:
        return Result(
            "B.2 Autosave audit", FAIL,
            "could not find any launch path referencing lu_4_mcp -- audit incomplete",
        )
    bad = [f for f in findings if not f[2]]
    if bad:
        bad_summary = "; ".join(f"{Path(p).name} ({label})" for p, label, _ in bad)
        return Result(
            "B.2 Autosave audit", FAIL,
            f"{len(bad)}/{audited} launch path(s) lack --no-save: {bad_summary}",
            {"audited": audited, "findings": [{"path": p, "label": l, "no_save": ns} for p, l, ns in findings]},
        )
    return Result(
        "B.2 Autosave audit", PASS,
        f"all {audited} audited launch path(s) pass --no-save",
        {"audited": audited, "findings": [{"path": p, "label": l, "no_save": ns} for p, l, ns in findings]},
    )


def check_b3_runtime_config_setters() -> Result:
    """B.3 -- every field in RuntimeConfigSnapshot must be reachable
    from the MCP via some setter (`lu4d_configure` is the omnibus
    setter; per-port setters may exist for nested objects like the
    trust table).  Today the production binary uses a single
    `lu4d_configure` tool that takes a flat keyed bag of args.  We
    parse the struct's fields and the tool's `args.get(...)` keys
    and assert subset coverage."""
    rc_path = SRC_DIR / "runtime_config.rs"
    t5_path = SRC_DIR / "mcp_t5.rs"
    if not rc_path.exists() or not t5_path.exists():
        return Result(
            "B.3 RuntimeConfig setters", FAIL,
            f"missing source: {rc_path} / {t5_path}",
        )
    rc_src = rc_path.read_text(encoding="utf-8")
    t5_src = t5_path.read_text(encoding="utf-8")
    # Parse RuntimeConfigSnapshot field names.  Match `pub field_name:`
    # inside the struct body.
    m = re.search(r'pub struct RuntimeConfigSnapshot\s*\{(.*?)^\}', rc_src, re.DOTALL | re.MULTILINE)
    if not m:
        return Result("B.3 RuntimeConfig setters", FAIL, "could not parse RuntimeConfigSnapshot")
    fields = re.findall(r'^\s*pub\s+([a-z_][a-z0-9_]*)\s*:', m.group(1), re.MULTILINE)
    fields = list(dict.fromkeys(fields))  # dedup, preserve order
    # Setter side: every key referenced via `args.get("X")`.  Plus any
    # additional setter tools (lu4d_trust_set / lu4d_trust_remove etc.)
    # for nested objects -- we treat the nested-object field as covered
    # if its mutation is exposed via at least one MCP tool.
    setter_keys = set(re.findall(r'args\.get\("([a-z_][a-z0-9_]*)"\)', t5_src))
    # Port BR.2: per-port RuntimeConfig setters live in their own mcp*.rs
    # files (e.g. mcp_runtime_setters.rs), so we scan ALL mcp*.rs files
    # for `args.get("X")` keys -- not just mcp_t5.rs.
    extra_setter_tools: List[str] = []
    for rs_file in SRC_DIR.rglob("mcp*.rs"):
        try:
            tx = rs_file.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for m2 in re.finditer(r'"name":\s*"(lu4d_[a-z_]+_set[a-z_]*)"', tx):
            extra_setter_tools.append(m2.group(1))
        for m3 in re.finditer(r'args\.get\("([a-z_][a-z0-9_]*)"\)', tx):
            setter_keys.add(m3.group(1))
    # Map RuntimeConfigSnapshot fields to per-key setter coverage.
    # Direct match first; then nested-object fallback (trust_table is
    # covered by lu4d_trust_set, hub_degree is covered by hub_max_in_degree,
    # etc.).
    nested_alias = {
        "trust_table":            ("trust_default_multiplier",        "trust_learning_enabled"),
        "hub_degree":             ("hub_max_in_degree",               "hub_max_out_degree",
                                   "hub_write_enforcement",            "hub_read_enforcement"),
        "ingest_run_id_strategy": (None,),
        "image_extractor_kind":   (None,),
    }
    missing_fields: List[str] = []
    covered_via: Dict[str, str] = {}
    for f in fields:
        if f in setter_keys:
            covered_via[f] = f"args.get(\"{f}\")"
            continue
        # Nested-object aliasing.
        if f in nested_alias:
            aliases = nested_alias[f]
            hits = [a for a in aliases if a and a in setter_keys]
            if hits:
                covered_via[f] = f"nested via {hits}"
                continue
        missing_fields.append(f)

    if missing_fields:
        return Result(
            "B.3 RuntimeConfig setters", FAIL,
            f"{len(missing_fields)}/{len(fields)} fields lack a setter: {missing_fields}",
            {
                "fields": fields,
                "setter_keys": sorted(setter_keys),
                "missing": missing_fields,
                "covered_via": covered_via,
            },
        )
    return Result(
        "B.3 RuntimeConfig setters", PASS,
        f"all {len(fields)} fields have setters via lu4d_configure / per-port tools",
        {"fields": fields, "setter_keys": sorted(setter_keys)},
    )


def check_b4_clean_tree() -> Result:
    """B.4 -- working tree must be clean before launching a 1+ hour ingest:
    `git status --short` empty, no `*.bak` files anywhere in the repo
    (excluding .git and target).  This guards against accidentally
    ingesting against a working copy that has uncommitted changes you
    can't reproduce later, and against orphaned backup files from
    previous edit sessions."""
    proc = subprocess.run(
        ["git", "status", "--short"], cwd=str(REPO_ROOT),
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        return Result("B.4 Clean tree", FAIL, f"git status failed: {proc.stderr.strip()}")
    dirty = proc.stdout.strip()
    if dirty:
        return Result(
            "B.4 Clean tree", FAIL,
            f"working tree dirty: {len(dirty.splitlines())} entry(ies)",
            {"git_status": dirty},
        )
    # Find .bak files (excluding .git and target).
    bak_files: List[str] = []
    for root, dirs, files in os.walk(REPO_ROOT):
        # In-place dir prune.
        dirs[:] = [d for d in dirs if d not in (".git", "target")]
        for fn in files:
            if fn.endswith(".bak"):
                bak_files.append(os.path.relpath(os.path.join(root, fn), REPO_ROOT))
    if bak_files:
        return Result(
            "B.4 Clean tree", FAIL,
            f"{len(bak_files)} *.bak file(s) found: {bak_files[:5]}{'...' if len(bak_files) > 5 else ''}",
            {"bak_files": bak_files},
        )
    return Result("B.4 Clean tree", PASS, "git clean and no .bak files")


# =================================================================
# PHASE C -- Integration smoke checks
# =================================================================

# 50 small synthetic Wikipedia-shaped articles for the smoke ingest.
# Embedded inline rather than streamed from HuggingFace so the check
# is hermetic and cheap (no network, no ~/.cache writes).  Each
# article is shaped like the wikimedia/wikipedia HF dump: title +
# text + url.  Content is hand-crafted to exercise every extractor
# layer (L1 See also, L2 headings, L2b categories, L3 lead patterns,
# L4 body co-mentions, L5 numerical attrs).
_SMOKE_ARTICLES: List[Dict[str, str]] = []

def _seed_smoke_articles() -> None:
    if _SMOKE_ARTICLES:
        return
    _topics = [
        # (title, lead, body_extra)
        ("Albedo",          "Albedo is the fraction of sunlight reflected by a body. Albedo is measured in dimensionless units.",
                            "Albedo was introduced by Johann Heinrich Lambert in 1760. Lambert was a Swiss polymath. Lambert's work Photometria was published in 1760."),
        ("Anchovy",         "An anchovy is a small fish of the family Engraulidae. Anchovies are found in marine waters.",
                            "Anchovy populations vary. Anchovies were known to the ancient Greeks. Anchovy fisheries are a part of Mediterranean cuisine."),
        ("Beaver",          "The beaver is a large rodent. The beaver is part of the genus Castor.",
                            "Beavers build dams. Beaver fur was historically traded in North America. The beaver was named for its industrious habits."),
        ("Berlin",          "Berlin is the capital of Germany. Berlin is located in northeastern Germany.",
                            "Berlin has a population of 3,769,000. Berlin was founded in 1237. Berlin is part of the European Union. Berlin's official language is German."),
        ("Boson",           "A boson is a type of subatomic particle. Bosons are described by Bose-Einstein statistics.",
                            "The boson was named after Satyendra Nath Bose. Bosons are produced in particle accelerators. The Higgs boson was discovered by CERN."),
        ("Capybara",        "The capybara is a giant cavy rodent. The capybara is a type of mammal.",
                            "Capybaras are found in South America. Capybaras live in social groups. The capybara is the largest rodent."),
        ("Cobalt",          "Cobalt is a chemical element. Cobalt is a hard metal.",
                            "Cobalt was discovered by Georg Brandt in 1735. Brandt was a Swedish chemist. Cobalt is used for magnets."),
        ("Diatom",          "A diatom is a type of microalga. Diatoms are found in oceans and lakes.",
                            "Diatoms produce oxygen. Diatom fossils are part of diatomaceous earth. Diatoms were studied by Christian Gottfried Ehrenberg."),
        ("Dunedin",         "Dunedin is a city in New Zealand. Dunedin is located in the South Island.",
                            "Dunedin was founded in 1848 by Scottish settlers. Dunedin has a population of 134,000. Dunedin's University of Otago is famous."),
        ("Eel",             "An eel is an elongated fish. Eels are part of the order Anguilliformes.",
                            "Eels are found worldwide. Eels are used for sushi. The European eel is critically endangered."),
        ("Eisenstein",      "Eisenstein is a German surname. Eisenstein is famous for several historical figures.",
                            "Sergei Eisenstein was a Soviet filmmaker. Eisenstein directed Battleship Potemkin. Eisenstein was born in 1898."),
        ("Fjord",           "A fjord is a long narrow inlet. A fjord is formed by glacial erosion.",
                            "Fjords are found in Norway. Norway has thousands of fjords. The fjord is part of coastal geography. Fjord tourism is famous."),
        ("Galaxy",          "A galaxy is a system of stars. A galaxy is held together by gravity.",
                            "Galaxies were classified by Edwin Hubble. Galaxies are found throughout the universe. The Milky Way galaxy contains the Sun."),
        ("Geyser",          "A geyser is a hot spring. A geyser erupts periodically.",
                            "Geysers are found in Yellowstone. Old Faithful geyser is famous. Geyser eruptions are caused by underground heat."),
        ("Helsinki",        "Helsinki is the capital of Finland. Helsinki is located on the Gulf of Finland.",
                            "Helsinki was founded in 1550. Helsinki has a population of 658,000. Helsinki's official language is Finnish."),
        ("Hummingbird",     "A hummingbird is a small bird. A hummingbird is part of the family Trochilidae.",
                            "Hummingbirds are found in the Americas. Hummingbirds drink nectar. Hummingbird wings beat rapidly."),
        ("Iberia",          "Iberia is a peninsula in southwestern Europe. Iberia is part of Europe.",
                            "Iberia includes Spain and Portugal. Iberia was conquered by Rome. The Iberian Peninsula has area of 583,254 square kilometres."),
        ("Inkwell",         "An inkwell is a small container. An inkwell holds ink.",
                            "Inkwells were used for fountain pens. Inkwells are found in antique shops. The inkwell is part of writing history."),
        ("Jaguar",          "The jaguar is a large cat. The jaguar is part of the genus Panthera.",
                            "Jaguars are found in the Americas. Jaguars hunt at night. Jaguar fur is spotted."),
        ("Jordan",          "Jordan is a country in the Middle East. Jordan is located east of Israel.",
                            "Jordan was founded in 1946. Jordan has a population of 10,300,000. Jordan's capital is Amman. Jordan's official language is Arabic."),
        ("Kelp",            "Kelp is a large brown seaweed. Kelp is part of the order Laminariales.",
                            "Kelp forests are found in cold waters. Kelp is used for food. Kelp is composed of cellulose and alginate."),
        ("Kiwi",            "The kiwi is a flightless bird. The kiwi is endemic to New Zealand.",
                            "Kiwis are nocturnal. The kiwi is a symbol of New Zealand. Kiwis lay large eggs."),
        ("Limestone",       "Limestone is a sedimentary rock. Limestone is composed of calcium carbonate.",
                            "Limestone is used for cement. Limestone is found worldwide. Limestone caves form karst landscapes."),
        ("Lyon",            "Lyon is a city in France. Lyon is located in central France.",
                            "Lyon was founded in 43 BC. Lyon has a population of 522,000. Lyon's official language is French."),
        ("Magma",           "Magma is molten rock. Magma is found beneath the Earth's surface.",
                            "Magma erupts as lava. Magma is composed of silicates. Magma chambers feed volcanoes."),
        ("Mongolia",        "Mongolia is a country in East Asia. Mongolia is located between Russia and China.",
                            "Mongolia has a population of 3,400,000. Mongolia's capital is Ulaanbaatar. Mongolia's official language is Mongolian. Mongolia has area of 1,564,116 square kilometres."),
        ("Nebula",          "A nebula is an interstellar cloud. A nebula is composed of gas and dust.",
                            "Nebulae are found in galaxies. The Crab Nebula was observed by Chinese astronomers. Nebula photography is famous."),
        ("Nutmeg",          "Nutmeg is a spice. Nutmeg is derived from the seed of Myristica fragrans.",
                            "Nutmeg is used for cooking. Nutmeg was traded by the Dutch. The nutmeg trade was part of colonial history."),
        ("Obsidian",        "Obsidian is a natural volcanic glass. Obsidian is formed from cooled lava.",
                            "Obsidian was used for tools. Obsidian is found in volcanic regions. Obsidian flakes are sharp."),
        ("Oslo",            "Oslo is the capital of Norway. Oslo is located in southeastern Norway.",
                            "Oslo was founded in 1040. Oslo has a population of 697,000. Oslo's official language is Norwegian."),
        ("Pelican",         "The pelican is a large water bird. The pelican is part of the family Pelecanidae.",
                            "Pelicans have a large bill. Pelicans are found worldwide. The pelican was a symbol in ancient Egypt."),
        ("Plankton",        "Plankton consists of small organisms. Plankton drifts in oceans and lakes.",
                            "Plankton is part of the food chain. Plankton produces oxygen. Plankton is composed of phytoplankton and zooplankton."),
        ("Quartz",          "Quartz is a hard mineral. Quartz is composed of silicon dioxide.",
                            "Quartz is used for watches. Quartz is found in many rocks. Quartz crystals are famous."),
        ("Quito",           "Quito is the capital of Ecuador. Quito is located in the Andes mountains.",
                            "Quito was founded in 1534 by Spanish settlers. Quito has a population of 1,839,000. Quito's altitude of 2,850 metres makes it one of the highest capitals."),
        ("Reef",            "A reef is an underwater ridge. A reef is composed of coral or rock.",
                            "Reefs are found in tropical seas. The Great Barrier Reef is famous. Reef ecosystems support marine life."),
        ("Riga",            "Riga is the capital of Latvia. Riga is located on the Baltic Sea.",
                            "Riga was founded in 1201. Riga has a population of 605,000. Riga's official language is Latvian."),
        ("Sandstone",       "Sandstone is a sedimentary rock. Sandstone is composed of sand-sized grains.",
                            "Sandstone is used for building. Sandstone is found worldwide. Sandstone caves form interesting landscapes."),
        ("Sapporo",         "Sapporo is a city in Japan. Sapporo is located on Hokkaido island.",
                            "Sapporo was founded in 1868. Sapporo has a population of 1,973,000. Sapporo's snow festival is famous."),
        ("Sequoia",         "A sequoia is a giant tree. A sequoia is part of the family Cupressaceae.",
                            "Sequoias are found in California. Sequoia trees can live thousands of years. The General Sherman tree is a famous sequoia."),
        ("Sundial",         "A sundial is a device that tells time. A sundial uses the sun.",
                            "Sundials were used in ancient times. Sundials are part of garden design. The sundial is composed of a gnomon and a plate."),
        ("Tundra",          "Tundra is a treeless biome. Tundra is found in cold regions.",
                            "Tundra is part of arctic geography. Tundra plants are short. Tundra animals include caribou."),
        ("Turin",           "Turin is a city in Italy. Turin is located in northwestern Italy.",
                            "Turin was founded in 28 BC. Turin has a population of 870,000. Turin's official language is Italian."),
        ("Ukulele",         "A ukulele is a small guitar. A ukulele is a type of stringed instrument.",
                            "Ukuleles were invented in Hawaii. Ukuleles are used for music. The ukulele is part of Hawaiian culture."),
        ("Uzbekistan",      "Uzbekistan is a country in Central Asia. Uzbekistan is located between Kazakhstan and Turkmenistan.",
                            "Uzbekistan was founded in 1991. Uzbekistan has a population of 35,300,000. Uzbekistan's capital is Tashkent. Uzbekistan's official language is Uzbek."),
        ("Volcano",         "A volcano is a rupture in the Earth's crust. A volcano allows magma to escape.",
                            "Volcanoes are found along plate boundaries. Mount Vesuvius is a famous volcano. Volcanoes can produce ash clouds."),
        ("Vienna",          "Vienna is the capital of Austria. Vienna is located in eastern Austria.",
                            "Vienna was founded in 1 AD. Vienna has a population of 1,931,000. Vienna's official language is German. Vienna's coffee houses are famous."),
        ("Walrus",          "The walrus is a large marine mammal. The walrus is part of the family Odobenidae.",
                            "Walruses have tusks. Walruses are found in the Arctic. The walrus was hunted for ivory."),
        ("Wasabi",          "Wasabi is a Japanese plant. Wasabi is used as a condiment.",
                            "Wasabi is derived from the rhizome of Wasabia japonica. Wasabi is used with sushi. The wasabi paste is famous."),
        ("Xylophone",       "A xylophone is a musical instrument. A xylophone is composed of wooden bars.",
                            "Xylophones are used in orchestras. Xylophone music is part of folk traditions. The xylophone is a percussion instrument."),
        ("Yak",             "A yak is a large bovine. A yak is found in the Himalayas.",
                            "Yaks are domesticated. Yak milk is used for butter. Yaks are part of Tibetan culture."),
    ]
    for title, lead, body in _topics:
        text = (
            f"{lead}\n\n{body}\n\n"
            f"History\n\n{title} has a long history. {title} was important in ancient times.\n\n"
            f"See also\n\n {title} fish\n {title} science\n\n"
            f"References\n\nExternal links\n\nGeography\nNatural sciences\n"
        )
        _SMOKE_ARTICLES.append({
            "title": title,
            "text":  text,
            "url":   f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
        })


def check_c1_smoke_50() -> Result:
    """C.1 -- 50-article ingest into a hermetic tmp brain.  Validates:
      * errors/article < 0.1   (extractor x binary contract is sound)
      * >=8 distinct BondType variants land
      * no single bond type > 70% of landed bonds
      * degree_p90 >= 5         (graph connectivity is healthy)
    Today these will mostly fail because the BondType vocabulary
    rejection (A.1) means most extractor relations error out instead
    of landing.  That's the gate's job: refuse to launch a real ingest
    until A.1 + C.1 both pass."""
    if not LOCAL_BINARY.exists() and not DEPLOYED_BINARY.exists():
        return Result("C.1 50-article smoke", FAIL, "no binary available to probe")
    binary = LOCAL_BINARY if LOCAL_BINARY.exists() else DEPLOYED_BINARY
    _seed_smoke_articles()
    # Lazy-import the extractor so that B-only or A-only runs do not need
    # the script's parent dir on sys.path.
    sys.path.insert(0, str(SCRIPT_DIR))
    from smart_extractor import extract_bonds_from_wikipedia_article  # type: ignore

    brain_dir = _ephemeral_brain_dir()
    brain_path = brain_dir / "preflight_smoke.lbr"
    histogram: Counter = Counter()
    landed = 0
    errors = 0
    bonds_attempted = 0
    try:
        client = MCPClient(binary, brain_path, timeout_init=15.0)
        try:
            for art in _SMOKE_ARTICLES:
                bonds = extract_bonds_from_wikipedia_article(art)
                # Build a teach_corpus_batch payload.  Cap at 200 entries
                # per article -- in practice the extractor caps L4 at 80
                # so this is a safety net only.
                entries = []
                for b in bonds[:200]:
                    entries.append({
                        "corpus":     "preflight-smoke",
                        "source_url": b.get("source_url") or art["url"],
                        "subject":    b["subject"],
                        "relation":   b["relation"],
                        "object":     b["object"],
                        "weight":     b["weight"],
                        "confidence": b["confidence"],
                    })
                bonds_attempted += len(entries)
                resp = client.tool_call("lu4d_teach_corpus_batch", {"entries": entries}, timeout=60.0)
                if "error" in resp:
                    errors += len(entries)
                    continue
                struct = (resp.get("result") or {}).get("structuredContent") or {}
                landed += int(struct.get("bonds_landed") or 0)
                errors += len(struct.get("errors") or [])
                # Tally landed-by-relation: we don't get a per-bond echo
                # back for batch, so use the input shape as the histogram
                # source.  Errors for individual entries are subtracted
                # later if available.
                for e in entries:
                    histogram[e["relation"]] += 1

            # BR.6: trigger an in-memory consolidation pass so the
            # networks cached degree_p90 is recomputed against the
            # bonds we just ingested.  No save-roundtrip is required:
            # degree_p90 is derived from in-memory adjacency and is
            # refreshed at the end of every consolidation run (see
            # ConsolidationStats.degree_p90_after_consolidation in
            # src/consolidation.rs).  Consolidation is cheap on a
            # freshly-ingested 50-article graph -- one session, no
            # decayed keyframes, low triangle count -- so this stays
            # well under the per-check budget.
            consolidate_resp = client.tool_call(
                "lu4d_consolidate", {}, timeout=60.0,
            )
            cs = (consolidate_resp.get("result") or {}).get("structuredContent") or {}
            raw_p90 = cs.get("degree_p90_after_consolidation")
            try:
                degree_p90 = int(raw_p90) if raw_p90 is not None else None
            except (TypeError, ValueError):
                degree_p90 = None
        finally:
            client.close()
    finally:
        shutil.rmtree(brain_dir, ignore_errors=True)

    n_articles = len(_SMOKE_ARTICLES)
    err_per_art = errors / max(1, n_articles)
    distinct = sum(1 for v in histogram.values() if v > 0)
    total = sum(histogram.values()) or 1
    top = histogram.most_common(1)[0][1] if histogram else 0
    top_share = top / total

    # BR.6: degree_p90 was captured above by piggy-backing on
    # lu4d_consolidate, which calls Network::recompute_degree_p90 at
    # the end of every run.  None means the consolidate call failed
    # or returned a non-numeric value -- treated as a failure below.

    failures = []
    if err_per_art >= 0.1:
        failures.append(f"errors/article={err_per_art:.3f} >= 0.1")
    if distinct < 8:
        failures.append(f"only {distinct} distinct relations attempted")
    if top_share > 0.7:
        failures.append(f"top relation {top_share:.0%} > 70%")
    # BR.6: enforce minimum graph connectivity on the smoke run.  A
    # healthy mature Lu hits degree_p90 >= 8-10 (see memory note
    # reference_lu_healthy_growth_curve.md); for the 50-article smoke,
    # 5 is the minimum we should ever ship.  Phase B Health Check
    # found degree_p90 = 2 ("dictionary not language") which is what
    # triggered the entire rebuild -- this assertion is the smoke-time
    # guard against regressing to that failure mode.
    if degree_p90 is None:
        failures.append("degree_p90 unavailable (consolidate call failed)")
    elif degree_p90 < 5:
        failures.append(f"degree_p90={degree_p90} < 5 (graph too sparse)")

    deg_str = f"{degree_p90}" if degree_p90 is not None else "?"
    detail = (
        f"articles={n_articles} attempted={bonds_attempted} landed={landed} "
        f"errors={errors} ({err_per_art:.3f}/art) distinct_relations={distinct} "
        f"top_relation_share={top_share:.0%} degree_p90={deg_str} (>=5)"
    )
    if failures:
        return Result(
            "C.1 50-article smoke", FAIL,
            detail + " :: " + "; ".join(failures),
            {"histogram": dict(histogram), "landed": landed, "errors": errors,
             "bonds_attempted": bonds_attempted, "articles": n_articles,
             "degree_p90": degree_p90},
        )
    return Result(
        "C.1 50-article smoke", PASS, detail,
        {"histogram": dict(histogram), "landed": landed, "errors": errors,
         "bonds_attempted": bonds_attempted, "articles": n_articles,
         "degree_p90": degree_p90},
    )


def _run_smoke_with_typed_ku_flag(
    binary: Path, flag_value: bool
) -> Tuple[Counter, int, int, int, int, int]:
    """Helper: run the 50-article smoke against a hermetic --no-save
    brain with `recall_typed_ku_enabled = flag_value` set BEFORE any
    ingest call lands. Then sweep a fixed list of recall-probe queries
    via `lu4d_reason` and tally how many response articulations carry
    a `(kind)` tag -- that count is the load-bearing signal of the
    flag's recall effect, since storage / extractor histograms are
    not influenced by the flag.

    Returns (histogram, landed, errors, attempted, kind_tagged_responses,
    total_recall_probes).
    """
    _seed_smoke_articles()
    sys.path.insert(0, str(SCRIPT_DIR))
    from smart_extractor import extract_bonds_from_wikipedia_article  # type: ignore

    # Recall probes -- each is a subject in the smoke fixture that
    # gets at least one definitional / mechanism / fact bond ingested.
    # The articulation under flag=true should carry a `(definition)`,
    # `(fact)`, `(mechanism)`, etc. tag for at least some of these.
    recall_probes = [
        "Albedo", "Berlin", "Boson", "Capybara", "Galaxy",
        "Helsinki", "Magma", "Quartz", "Sequoia", "Volcano",
    ]
    typed_kinds = (
        "definition", "fact", "mechanism", "composition",
        "purpose", "causal", "contrast", "example",
        "code", "procedure", "episode", "preference", "image",
    )

    def has_typed_kind_tag(answer: str) -> bool:
        # `articulate_bond_typed` emits "(kind)", "(fk -> tk)",
        # "(-> tk)". Detect any kind appearing inside a parenthesised
        # span anywhere in the prose.
        for k in typed_kinds:
            if f"({k})" in answer or f"-> {k})" in answer or f"({k} -> " in answer:
                return True
        return False

    brain_dir = _ephemeral_brain_dir()
    brain_path = brain_dir / f"preflight_smoke_typed_ku_{flag_value}.lbr"
    histogram: Counter = Counter()
    landed = 0
    errors = 0
    bonds_attempted = 0
    kind_tagged_responses = 0
    total_recall_probes = 0
    try:
        client = MCPClient(binary, brain_path, timeout_init=15.0)
        try:
            client.tool_call(
                "lu4d_recall_set",
                {"recall_typed_ku_enabled": flag_value},
                timeout=15.0,
            )
            for art in _SMOKE_ARTICLES:
                bonds = extract_bonds_from_wikipedia_article(art)
                entries = []
                for b in bonds[:200]:
                    entries.append({
                        "corpus":     "preflight-smoke",
                        "source_url": b.get("source_url") or art["url"],
                        "subject":    b["subject"],
                        "relation":   b["relation"],
                        "object":     b["object"],
                        "weight":     b["weight"],
                        "confidence": b["confidence"],
                    })
                bonds_attempted += len(entries)
                resp = client.tool_call(
                    "lu4d_teach_corpus_batch", {"entries": entries}, timeout=60.0
                )
                if "error" in resp:
                    errors += len(entries)
                    continue
                struct = (resp.get("result") or {}).get("structuredContent") or {}
                landed += int(struct.get("bonds_landed") or 0)
                errors += len(struct.get("errors") or [])
                for e in entries:
                    histogram[e["relation"]] += 1
            # Now sweep the recall probes. Each probe runs lu4d_reason
            # against the brain that just ingested under the chosen
            # flag value and records whether the prose carries a
            # typed-kind tag.
            for probe in recall_probes:
                resp = client.tool_call(
                    "lu4d_reason", {"query": probe}, timeout=15.0
                )
                total_recall_probes += 1
                if "error" in resp:
                    continue
                struct = (resp.get("result") or {}).get("structuredContent") or {}
                final_answer = struct.get("final_answer") or ""
                if has_typed_kind_tag(final_answer):
                    kind_tagged_responses += 1
        finally:
            client.close()
    finally:
        shutil.rmtree(brain_dir, ignore_errors=True)
    return (
        histogram, landed, errors, bonds_attempted,
        kind_tagged_responses, total_recall_probes,
    )


def check_c2_dual_mode() -> Result:
    """C.2 -- run the 50-article smoke TWICE, once with
    `recall_typed_ku_enabled = true` (default) and once with `= false`,
    and emit the deltas as informational output.

    Assertions:
      * Both runs succeed (no transport errors, populated histogram).
      * Bond totals are within reasonable bounds in both modes.
        Equality is NOT enforced -- the typed path may activate
        slightly differently. The flag should ideally produce some
        observable behavioral delta (logged in detail).
    """
    if not LOCAL_BINARY.exists() and not DEPLOYED_BINARY.exists():
        return Result("C.2 dual-mode typed-KU", FAIL, "no binary available to probe")
    binary = LOCAL_BINARY if LOCAL_BINARY.exists() else DEPLOYED_BINARY

    try:
        (hist_on, landed_on, err_on, attempted_on,
         tagged_on, probes_on) = _run_smoke_with_typed_ku_flag(binary, True)
        (hist_off, landed_off, err_off, attempted_off,
         tagged_off, probes_off) = _run_smoke_with_typed_ku_flag(binary, False)
    except Exception as e:
        return Result("C.2 dual-mode typed-KU", FAIL, f"smoke run blew up: {e}")

    distinct_on = sum(1 for v in hist_on.values() if v > 0)
    distinct_off = sum(1 for v in hist_off.values() if v > 0)
    total_on = sum(hist_on.values())
    total_off = sum(hist_off.values())

    failures = []
    # Both runs must produce a populated histogram.
    if total_on == 0:
        failures.append("flag=true produced an empty histogram")
    if total_off == 0:
        failures.append("flag=false produced an empty histogram")
    # Volume parity: the flag gates recall, not storage. Histogram
    # totals SHOULD match within a tight bound -- treat >5% drift
    # as suspicious (storage path leaked a kind dependency).
    if total_on > 0 and total_off > 0:
        ratio = abs(total_on - total_off) / max(total_on, total_off)
        if ratio > 0.05:
            failures.append(
                f"histogram totals diverge by {ratio:.0%} "
                f"(flag=true: {total_on}, flag=false: {total_off})"
            )
    if abs(distinct_on - distinct_off) > 0:
        failures.append(
            f"distinct-relation count diverges by {abs(distinct_on - distinct_off)}"
            f" (flag=true: {distinct_on}, flag=false: {distinct_off})"
        )
    # The flag MUST have an effect on recall articulation. The probe
    # sweep runs the same query under both flag positions; the flag
    # is doing real work iff at least one probe response under
    # flag=true carries a typed-kind tag while flag=false carries
    # FEWER such tags. Identical counts in both modes signals Q.1
    # is not actually routing differently (BR.4 spec halt condition).
    if probes_on > 0 and tagged_on == tagged_off:
        failures.append(
            f"flag has NO effect on recall: tagged_on={tagged_on} == "
            f"tagged_off={tagged_off} across {probes_on} probes "
            "(BR.4 halt condition -- Q.1 not routing differently)"
        )
    if tagged_on < tagged_off:
        failures.append(
            f"flag inverted: tagged_on={tagged_on} < tagged_off={tagged_off}"
        )

    detail = (
        f"flag=true: landed={landed_on} bonds total={total_on} kinds={distinct_on} "
        f"recall_kind_tagged={tagged_on}/{probes_on} | "
        f"flag=false: landed={landed_off} bonds total={total_off} kinds={distinct_off} "
        f"recall_kind_tagged={tagged_off}/{probes_off} | "
        f"delta_landed={landed_on - landed_off:+d} "
        f"delta_recall_tagged={tagged_on - tagged_off:+d}"
    )
    structured = {
        "flag_true": {
            "histogram": dict(hist_on),  "landed": landed_on,
            "errors": err_on,  "attempted": attempted_on,
            "recall_kind_tagged": tagged_on, "recall_probes": probes_on,
        },
        "flag_false": {
            "histogram": dict(hist_off), "landed": landed_off,
            "errors": err_off, "attempted": attempted_off,
            "recall_kind_tagged": tagged_off, "recall_probes": probes_off,
        },
    }
    if failures:
        return Result(
            "C.2 dual-mode typed-KU", FAIL,
            detail + " :: " + "; ".join(failures),
            structured,
        )
    return Result("C.2 dual-mode typed-KU", PASS, detail, structured)


# =================================================================
# Driver
# =================================================================

PHASES: Dict[str, List[Tuple[str, Callable[[], Result]]]] = {
    "A": [
        ("A.1", lambda: check_a1_bondtype(_pick_binary())),
        ("A.2", check_a2_nodetype),
        ("A.3", check_a3_kind_presence),
        ("A.4", check_a4_sourcekind),
        ("A.5", check_a5_extension_fields),
    ],
    "B": [
        ("B.1", check_b1_binary_md5),
        ("B.2", check_b2_autosave_audit),
        ("B.3", check_b3_runtime_config_setters),
        ("B.4", check_b4_clean_tree),
    ],
    "C": [
        ("C.1", check_c1_smoke_50),
        ("C.2", check_c2_dual_mode),
    ],
}


def _pick_binary() -> Path:
    if LOCAL_BINARY.exists(): return LOCAL_BINARY
    if DEPLOYED_BINARY.exists(): return DEPLOYED_BINARY
    # Fall back to local path so the failure message is crisp.
    return LOCAL_BINARY


def main(argv: List[str]) -> int:
    p = argparse.ArgumentParser(description="Lu BR.0 preflight harness")
    p.add_argument("--phase", choices=["A", "B", "C"], default=None,
                   help="Run only one phase (default: all phases)")
    p.add_argument("--keep-going", action="store_true",
                   help="Run every check even after a FAIL (exit code = #FAILs)")
    p.add_argument("--json", action="store_true",
                   help="Emit a final JSON summary to stdout (after the human report)")
    args = p.parse_args(argv)

    selected = [args.phase] if args.phase else ["A", "B", "C"]
    overall: List[Result] = []
    fail_count = 0
    stub_count = 0
    print(f"=== Lu BR.0 Preflight ({', '.join(selected)}) ===")
    for phase in selected:
        print(f"\n-- Phase {phase} --")
        for short, fn in PHASES[phase]:
            try:
                r = fn()
            except Exception as e:
                r = Result(short, FAIL, f"check raised: {type(e).__name__}: {e}")
            overall.append(r)
            print_result(r)
            if r.status == FAIL:
                fail_count += 1
                if not args.keep_going:
                    print(f"\n=== STOPPED on first FAIL ({r.name}) ===")
                    if args.json:
                        print(json.dumps(_to_json(overall), indent=2))
                    return 1
            elif r.status == STUB:
                stub_count += 1

    print(f"\n=== Summary: {len(overall)} checks, {fail_count} FAIL, {stub_count} STUB ===")
    if args.json:
        print(json.dumps(_to_json(overall), indent=2))
    return 0 if fail_count == 0 else max(1, fail_count)


def _to_json(results: List[Result]) -> Dict[str, Any]:
    return {
        "checks": [
            {
                "name":   r.name,
                "status": r.status,
                "detail": r.detail,
                "extra":  r.extra,
            }
            for r in results
        ],
        "summary": {
            "total": len(results),
            "pass":  sum(1 for r in results if r.status == PASS),
            "fail":  sum(1 for r in results if r.status == FAIL),
            "stub":  sum(1 for r in results if r.status == STUB),
        },
    }


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
