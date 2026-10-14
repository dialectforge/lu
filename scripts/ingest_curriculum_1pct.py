#!/usr/bin/env python3
"""
1% retrain test ingest harness.

Tier 1: wikimedia/wikipedia (20231101.en) at 1%, ranjaykrishna/visual_genome at 1%.
Tier 2: HuggingFaceH4/no_robots at 1%, HuggingFaceH4/instruction-dataset at 1%.

Streams datasets via `datasets` library and sends bonds to a subprocess
lu_4_mcp server over stdio (subprocess.Popen). Throttles to 100 calls/sec.
Auto-scales-down to 0.1% if Wikipedia projected wall-clock > 4 hours.

Deviation from spec: uses subprocess.Popen + bidirectional pipes instead of
mkfifo + backgrounded MCP server, because bidirectional JSON-RPC over a FIFO
requires its own response-matching layer. Same long-lived-server outcome.
"""

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

# Reconciliation port (2026-05-10): canonical extract_response_bond lives in
# ingest_curriculum_full so the _FRAME_NOUN_SUBJECTS filter, _extract_response_subject
# Pattern A/B subject discipline, and BR.9 typed-relation mapping are all
# applied consistently. Local duplicate (older prompt-as-subject heuristic)
# removed; this shim imports the canonical version.
import sys as _sys2, os as _os2
_SCRIPTS_DIR = _os2.path.dirname(_os2.path.abspath(__file__))
if _SCRIPTS_DIR not in _sys2.path:
    _sys2.path.insert(0, _SCRIPTS_DIR)
from ingest_curriculum_full import extract_response_bond  # noqa: E402

PROGRESS_LOG = "/var/lib/lu/ingest_1pct_progress.log"
THROTTLE_HZ = 100.0  # max calls/sec
RUN_TIMESTAMP = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
RUN_ID_TIER1 = f"1pct-test-tier1-{RUN_TIMESTAMP}"
RUN_ID_TIER2 = f"1pct-test-tier2-{RUN_TIMESTAMP}"


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


class McpClient:
    def __init__(self, brain_path, keyfile, log_stderr_to=None):
        self.brain_path = brain_path
        self.keyfile = keyfile
        self._stderr = open(log_stderr_to, "ab") if log_stderr_to else subprocess.DEVNULL
        self.proc = subprocess.Popen(
            [
                "/usr/local/bin/lu_4_mcp",
                "--mcp",
                "--require-encrypted",
                "--keyfile",
                keyfile,
                "--brain",
                brain_path,
                "--brain-format-out",
                "v3enc",
            ],
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
            "params": {"protocolVersion": "2024-11-05",
                       "capabilities": {}, "clientInfo": {"name": "ingest-1pct", "version": "0.1"}}
        })
        self._next_id += 1
        self._recv()
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def call(self, name, args):
        rid = self._next_id
        self._next_id += 1
        self._send({
            "jsonrpc": "2.0", "id": rid, "method": "tools/call",
            "params": {"name": name, "arguments": args}
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


def log_progress(line):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(PROGRESS_LOG, "a") as f:
        f.write(f"{ts} {line}\n")
        f.flush()
    print(f"[{ts}] {line}", flush=True)


def call_ok(resp):
    """Did the MCP call succeed (no JSON-RPC error)?"""
    return isinstance(resp, dict) and "error" not in resp


# --- Bond extraction heuristics ---

STOP_AFTER_SENTENCE = 1  # use first sentence for the IsA pattern

IS_A_PATTERN = re.compile(
    r"\b(?:is|was|are|were)\s+(?:an?|the)\s+([a-z][\w\s\-]{2,60}?)\s+(?:that|which|who|where|in|of|on|located|known|used|with|by|for|from|to|and|\.|,|;)",
    re.IGNORECASE,
)


def first_sentence(text):
    if not text:
        return ""
    m = re.match(r"([^.!?\n]{20,400}[.!?])", text.strip())
    return m.group(1).strip() if m else text.strip()[:200]


def extract_wikipedia_bonds(title, text, source_url):
    """Return a list of (subject, relation, object, weight, confidence) tuples.

    1-3 bonds per article: lead summary + IsA categorical + optional NotableTerm.
    """
    bonds = []
    if not text or not title:
        return bonds
    lead = first_sentence(text)
    if not lead:
        return bonds
    # Bond 1: subject is_about lead summary (capped to 200 chars).
    bonds.append((title, "RelatedTo", lead[:200], 0.5, "Inferred"))
    # Bond 2: subject IsA category, if first-sentence pattern matches.
    m = IS_A_PATTERN.search(lead)
    if m:
        cat = m.group(1).strip().rstrip(",.;:").lower()
        if 2 <= len(cat) <= 60:
            bonds.append((title, "IsA", cat, 0.6, "Inferred"))
    return bonds


# --- Tier 1 ingest ---

def ingest_wikipedia(client, throttle, fraction, time_budget_secs):
    """Stream wikimedia/wikipedia and call lu4d_teach_corpus per article.

    Returns dict with stats. Implements rate-measurement checkpoint:
    after first 100 articles, projects total wall clock; if > 4h, auto-scales.
    """
    from datasets import load_dataset
    log_progress(f"wikipedia: starting load_dataset(streaming) fraction={fraction}")
    ds = load_dataset(
        "wikimedia/wikipedia", "20231101.en",
        split="train", streaming=True,
    )
    # 1% of ~6.7M articles ≈ 67000.
    target_total = int(6_700_000 * fraction)
    log_progress(f"wikipedia: target_total={target_total} articles ({fraction*100:.2f}% of ~6.7M)")

    bonds_landed = 0
    bonds_errors = 0
    articles_processed = 0
    start = time.monotonic()
    checkpoint_articles = 100
    auto_scaled = False

    for ex in ds:
        if articles_processed >= target_total:
            break
        title = ex.get("title", "")
        text = ex.get("text", "")
        url = ex.get("url", "")
        bonds = extract_wikipedia_bonds(title, text, url)
        for (s, r, o, w, c) in bonds:
            throttle.wait()
            try:
                resp = client.call("lu4d_teach_corpus", {
                    "corpus": "wikimedia/wikipedia",
                    "source_url": url,
                    "run_id": RUN_ID_TIER1,
                    "subject": s,
                    "relation": r,
                    "object": o,
                    "weight": w,
                    "confidence": c,
                })
                if call_ok(resp):
                    bonds_landed += 1
                else:
                    bonds_errors += 1
            except Exception:
                bonds_errors += 1
        articles_processed += 1

        if articles_processed % 100 == 0:
            elapsed = time.monotonic() - start
            rate = articles_processed / elapsed if elapsed > 0 else 0
            eta = (target_total - articles_processed) / rate if rate > 0 else float("inf")
            log_progress(
                f"wikipedia: articles={articles_processed}/{target_total} "
                f"bonds_landed={bonds_landed} errors={bonds_errors} "
                f"rate={rate:.1f}/s eta_remaining_s={eta:.0f}"
            )
            # Rate-measurement checkpoint at exactly 100 articles.
            if articles_processed == checkpoint_articles and not auto_scaled:
                projected_total = elapsed + eta
                if projected_total > time_budget_secs:
                    new_target = max(1000, target_total // 10)  # scale to 0.1%
                    log_progress(
                        f"wikipedia: AUTO-SCALE-DOWN — projected {projected_total:.0f}s > budget {time_budget_secs}s. "
                        f"Scaling target_total {target_total} -> {new_target}."
                    )
                    target_total = new_target
                    auto_scaled = True
            if articles_processed - bonds_errors == 0 and articles_processed >= 200:
                log_progress("wikipedia: ABORT — too many errors; bailing out.")
                break

    elapsed = time.monotonic() - start
    log_progress(
        f"wikipedia: DONE articles={articles_processed} bonds_landed={bonds_landed} "
        f"errors={bonds_errors} elapsed_s={elapsed:.0f} auto_scaled={auto_scaled}"
    )
    return {
        "articles_processed": articles_processed,
        "bonds_landed": bonds_landed,
        "bonds_errors": bonds_errors,
        "elapsed_s": elapsed,
        "auto_scaled": auto_scaled,
        "target_total": target_total,
    }



VG_PRED_MAP = {
    "on": "PartOf", "in": "PartOf", "of": "PartOf", "part of": "PartOf",
    "has": "HasProperty", "with": "HasProperty",
    "wearing": "HasProperty", "holding": "HasProperty",
    "carrying": "HasProperty", "made of": "PartOf",
}

def _vg_pred(p):
    return VG_PRED_MAP.get(p.lower().strip(), "RelatedTo")

def ingest_visual_genome(client, throttle, fraction):
    """Stream Visual Genome scene-graph triplets and ingest as bonds.

    Uses scripts/visual_genome_loader.py against the canonical relationships
    dump cached at /var/lib/lu/data/visual_genome/relationships.json.zip.
    The HuggingFace `ranjaykrishna/visual_genome` repo is a script-backed
    loader broken under datasets>=4.0; `Voxel51/GQA-Scene-Graph` ships only
    JPEGs. Direct dump + custom loader is the working path.

    Targets ~1% of the ~2.3M VG relationships at fraction=0.01.
    """
    import importlib.util, sys, pathlib
    loader_path = pathlib.Path(__file__).resolve().parent / "visual_genome_loader.py"
    spec = importlib.util.spec_from_file_location("vg_loader", loader_path)
    vg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vg)

    VG_TOTAL_RELATIONSHIPS = 2_316_104  # canonical VG v1.4 count
    target_total = max(50, int(VG_TOTAL_RELATIONSHIPS * fraction))
    corpus_id = "visual_genome"
    log_progress(
        f"visual_genome: starting cached-dump stream fraction={fraction} target={target_total}"
    )
    bonds_landed = 0
    bonds_errors = 0
    images_seen = set()
    skipped_reason = None
    start = time.monotonic()
    try:
        for s_name, raw_pred, o_name, image_id in vg.stream_triplets(max_triplets=target_total):
            images_seen.add(image_id)
            relation = vg.map_predicate(raw_pred)
            url = f"vg-image-{image_id}"
            throttle.wait()
            try:
                resp = client.call("lu4d_teach_corpus", {
                    "corpus": corpus_id,
                    "source_url": url,
                    "run_id": RUN_ID_TIER1,
                    "subject": s_name,
                    "relation": relation,
                    "object": o_name,
                    "weight": 0.5,
                    "confidence": "Unverified",
                })
                if call_ok(resp):
                    bonds_landed += 1
                else:
                    bonds_errors += 1
            except Exception:
                bonds_errors += 1
            if (bonds_landed + bonds_errors) % 500 == 0:
                log_progress(
                    f"visual_genome: triplets={bonds_landed + bonds_errors}/{target_total} "
                    f"landed={bonds_landed} errors={bonds_errors} images={len(images_seen)}"
                )
    except FileNotFoundError as e:
        skipped_reason = f"cache missing: {e}"
        log_progress(f"visual_genome: SKIPPED — {skipped_reason}")
    except Exception as e:
        skipped_reason = f"VG stream failed: {type(e).__name__}: {str(e)[:200]}"
        log_progress(f"visual_genome: SKIPPED — {skipped_reason}")
    elapsed = time.monotonic() - start
    log_progress(
        f"visual_genome: DONE images={len(images_seen)} bonds_landed={bonds_landed} "
        f"errors={bonds_errors} elapsed_s={elapsed:.0f} skip={skipped_reason}"
    )
    return {
        "images_processed": len(images_seen),
        "bonds_landed": bonds_landed,
        "bonds_errors": bonds_errors,
        "elapsed_s": elapsed,
        "skipped_reason": skipped_reason,
        "corpus_id": corpus_id,
        "target_total": target_total,
    }


def ingest_no_robots(client, throttle, fraction):
    from datasets import load_dataset
    log_progress(f"no_robots: starting load_dataset(streaming) fraction={fraction}")
    bonds_landed = 0
    bonds_errors = 0
    statements_landed = 0
    examples_processed = 0
    target_total = max(50, int(10000 * fraction))
    start = time.monotonic()
    skipped_reason = None
    try:
        ds = load_dataset("HuggingFaceH4/no_robots", split="train", streaming=True)
        for ex in ds:
            if examples_processed >= target_total:
                break
            messages = ex.get("messages", [])
            prompt = ""
            response = ""
            for m in messages:
                if m.get("role") == "user" and not prompt:
                    prompt = m.get("content", "")
                elif m.get("role") == "assistant" and not response:
                    response = m.get("content", "")
            if prompt:
                throttle.wait()
                try:
                    resp = client.call("lu4d_user_statement", {
                        "text": prompt[:1000],
                        "source_label": "HuggingFaceH4/no_robots",
                    })
                    if call_ok(resp):
                        statements_landed += 1
                except Exception:
                    pass
            bond = extract_response_bond(prompt, response)
            if bond:
                s, r, o, w, c = bond
                throttle.wait()
                try:
                    resp = client.call("lu4d_teach_corpus", {
                        "corpus": "HuggingFaceH4/no_robots",
                        "run_id": RUN_ID_TIER2,
                        "subject": s,
                        "relation": r,
                        "object": o,
                        "weight": w,
                        "confidence": c,
                    })
                    if call_ok(resp):
                        bonds_landed += 1
                    else:
                        bonds_errors += 1
                except Exception:
                    bonds_errors += 1
            examples_processed += 1
            if examples_processed % 50 == 0:
                log_progress(
                    f"no_robots: examples={examples_processed}/{target_total} "
                    f"statements={statements_landed} bonds_landed={bonds_landed} errors={bonds_errors}"
                )
    except Exception as e:
        skipped_reason = f"no_robots load failed: {type(e).__name__}: {e}"
        log_progress(f"no_robots: SKIPPED — {skipped_reason}")

    elapsed = time.monotonic() - start
    log_progress(
        f"no_robots: DONE examples={examples_processed} statements={statements_landed} "
        f"bonds_landed={bonds_landed} errors={bonds_errors} elapsed_s={elapsed:.0f} skip={skipped_reason}"
    )
    return {
        "examples_processed": examples_processed,
        "statements_landed": statements_landed,
        "bonds_landed": bonds_landed,
        "bonds_errors": bonds_errors,
        "elapsed_s": elapsed,
        "skipped_reason": skipped_reason,
    }


def ingest_instruction_dataset(client, throttle, fraction):
    from datasets import load_dataset
    log_progress(f"instruction-dataset: starting load_dataset(streaming) fraction={fraction}")
    bonds_landed = 0
    bonds_errors = 0
    statements_landed = 0
    examples_processed = 0
    target_total = max(20, int(330 * fraction))  # ~330 rows.
    start = time.monotonic()
    skipped_reason = None
    try:
        ds = load_dataset("HuggingFaceH4/instruction-dataset", split="test", streaming=True)
        for ex in ds:
            if examples_processed >= target_total:
                break
            prompt = ex.get("prompt", "") or ex.get("instruction", "")
            response = ex.get("completion", "") or ex.get("response", "") or ex.get("output", "")
            if prompt:
                throttle.wait()
                try:
                    resp = client.call("lu4d_user_statement", {
                        "text": prompt[:1000],
                        "source_label": "HuggingFaceH4/instruction-dataset",
                    })
                    if call_ok(resp):
                        statements_landed += 1
                except Exception:
                    pass
            bond = extract_response_bond(prompt, response)
            if bond:
                s, r, o, w, c = bond
                throttle.wait()
                try:
                    resp = client.call("lu4d_teach_corpus", {
                        "corpus": "HuggingFaceH4/instruction-dataset",
                        "run_id": RUN_ID_TIER2,
                        "subject": s,
                        "relation": r,
                        "object": o,
                        "weight": w,
                        "confidence": c,
                    })
                    if call_ok(resp):
                        bonds_landed += 1
                    else:
                        bonds_errors += 1
                except Exception:
                    bonds_errors += 1
            examples_processed += 1
            if examples_processed % 50 == 0:
                log_progress(
                    f"instruction-dataset: examples={examples_processed}/{target_total} "
                    f"statements={statements_landed} bonds_landed={bonds_landed} errors={bonds_errors}"
                )
    except Exception as e:
        skipped_reason = f"instruction-dataset load failed: {type(e).__name__}: {e}"
        log_progress(f"instruction-dataset: SKIPPED — {skipped_reason}")

    elapsed = time.monotonic() - start
    log_progress(
        f"instruction-dataset: DONE examples={examples_processed} statements={statements_landed} "
        f"bonds_landed={bonds_landed} errors={bonds_errors} elapsed_s={elapsed:.0f} skip={skipped_reason}"
    )
    return {
        "examples_processed": examples_processed,
        "statements_landed": statements_landed,
        "bonds_landed": bonds_landed,
        "bonds_errors": bonds_errors,
        "elapsed_s": elapsed,
        "skipped_reason": skipped_reason,
    }


# --- main ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", type=int, choices=[0, 1, 2], default=0,
                        help="0=both, 1=Tier 1 only, 2=Tier 2 only")
    parser.add_argument("--fraction", type=float, default=0.01,
                        help="Sampling fraction (default 0.01 = 1%%)")
    parser.add_argument("--brain", default="/var/lib/lu/brain-1pct-test.lbr")
    parser.add_argument("--keyfile", default="/var/lib/lu/master-1pct.key")
    parser.add_argument("--time-budget-secs", type=int, default=4 * 3600)
    parser.add_argument("--seed-curriculum", action="store_true",
                        help="Run lu4d_trust_reset_to_defaults with seed_curriculum_corpora=true")
    parser.add_argument("--summary-out", default="/var/lib/lu/ingest_1pct_summary.json")
    parser.add_argument("--mcp-stderr", default="/var/lib/lu/ingest_1pct_mcp.err")
    args = parser.parse_args()

    log_progress(f"=== INGEST START tier={args.tier} fraction={args.fraction} brain={args.brain} ===")
    summary = {
        "tier": args.tier,
        "fraction": args.fraction,
        "brain": args.brain,
        "run_timestamp": RUN_TIMESTAMP,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }

    client = McpClient(args.brain, args.keyfile, log_stderr_to=args.mcp_stderr)
    throttle = Throttle(THROTTLE_HZ)

    try:
        # Status before.
        resp = client.call("lu4d_status", {})
        sc = resp.get("result", {}).get("structuredContent", {}) if call_ok(resp) else {}
        summary["bond_count_before"] = sc.get("bond_count", -1)
        summary["node_count_before"] = sc.get("node_count", -1)
        log_progress(f"status_before: bonds={summary['bond_count_before']} nodes={summary['node_count_before']}")

        if args.seed_curriculum:
            log_progress("seeding curriculum corpora via lu4d_trust_reset_to_defaults")
            r = client.call("lu4d_trust_reset_to_defaults", {"seed_curriculum_corpora": True})
            log_progress(f"trust_reset response: {json.dumps(r.get('result', r))[:300]}")
            r2 = client.call("lu4d_trust_status", {})
            log_progress(f"trust_status: {json.dumps(r2.get('result', r2))[:500]}")
            r3 = client.call("lu4d_hub_status", {"top_n": 5})
            log_progress(f"hub_status: {json.dumps(r3.get('result', r3))[:500]}")

        if args.tier in (0, 1):
            summary["wikipedia"] = ingest_wikipedia(client, throttle, args.fraction, args.time_budget_secs)
            summary["visual_genome"] = ingest_visual_genome(client, throttle, args.fraction)

        if args.tier in (0, 2):
            summary["no_robots"] = ingest_no_robots(client, throttle, args.fraction)
            summary["instruction_dataset"] = ingest_instruction_dataset(client, throttle, args.fraction)

        log_progress("saving brain via lu4d_save")
        save_resp = client.call("lu4d_save", {})
        summary["save_response"] = save_resp.get("result", save_resp)

        # Status after.
        resp = client.call("lu4d_status", {})
        sc = resp.get("result", {}).get("structuredContent", {}) if call_ok(resp) else {}
        summary["bond_count_after"] = sc.get("bond_count", -1)
        summary["node_count_after"] = sc.get("node_count", -1)
        log_progress(f"status_after: bonds={summary['bond_count_after']} nodes={summary['node_count_after']}")

    finally:
        summary["ended_at"] = datetime.now(timezone.utc).isoformat()
        with open(args.summary_out, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        log_progress(f"=== INGEST END — summary written to {args.summary_out} ===")
        client.close()


if __name__ == "__main__":
    main()
