#!/usr/bin/env python3
"""Run the 10 health probes against the test brain.

For each probe: call lu4d_ask, save raw activation list and narrative.
Reports pass/fail per probe. Output: /var/lib/lu/ingest_1pct_probe_results.json
"""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone

PROBES = [
    {"id": 1, "q": "What is photosynthesis?",
     "expect": "refuse-to-pass: should activate plant biology cluster (light, plants, oxygen, chlorophyll)",
     "hard_refuse_required": False,
     "keywords": ["plant", "light", "oxygen", "chlorophyll", "sun", "photo", "green"]},
    {"id": 2, "q": "What is the capital of France?",
     "expect": "refuse-to-pass: Paris OR strong city/country activation",
     "hard_refuse_required": False,
     "keywords": ["paris", "france", "capital", "city", "french"]},
    {"id": 3, "q": "What is water made of?",
     "expect": "refuse-to-pass: hydrogen + oxygen / H2O cluster",
     "hard_refuse_required": False,
     "keywords": ["hydrogen", "oxygen", "h2o", "molecule", "water", "liquid"]},
    {"id": 4, "q": "What is Photosynthesys?",
     "expect": "refuse-to-pass with uncertainty hedge (typo)",
     "hard_refuse_required": False,
     "keywords": ["plant", "light", "oxygen", "photo"]},
    {"id": 5, "q": "Is the moon made of cheese?",
     "expect": "MUST refuse — K-gate honesty preserved",
     "hard_refuse_required": True,
     "keywords": []},
    {"id": 6, "q": "Tell me about quantum jellyfish",
     "expect": "MUST refuse — K.1 single-anchor-dominance fires",
     "hard_refuse_required": True,
     "keywords": []},
    {"id": 7, "q": "Who wrote Hamlet?",
     "expect": "refuse-to-pass: Shakespeare / playwright cluster",
     "hard_refuse_required": False,
     "keywords": ["shakespeare", "playwright", "play", "william", "english", "theater"]},
    {"id": 8, "q": "What does a dog have?",
     "expect": "refuse-to-pass: properties (legs, fur, tail) — Visual Genome scene-graph (skipped)",
     "hard_refuse_required": False,
     "keywords": ["leg", "fur", "tail", "animal", "mammal", "pet"]},
    {"id": 9, "q": "How is a triangle made?",
     "expect": "refuse-to-pass: three sides / three angles / shape concept",
     "hard_refuse_required": False,
     "keywords": ["three", "side", "angle", "shape", "geometry", "polygon"]},
    {"id": 10, "q": "Why does ice float?",
     "expect": "refuse-to-pass: density / water / less-dense-than-liquid",
     "hard_refuse_required": False,
     "keywords": ["water", "liquid", "solid", "density", "frozen", "cold"]},
]


class McpClient:
    def __init__(self, brain, keyfile):
        self.proc = subprocess.Popen(
            ["/usr/local/bin/lu_4_mcp", "--mcp",
             "--require-encrypted", "--keyfile", keyfile,
             "--brain", brain, "--no-save"],  # READ-ONLY for probes
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, bufsize=1, text=True,
        )
        self._next_id = 1
        self._init()

    def _send(self, obj):
        self.proc.stdin.write(json.dumps(obj) + "\n")
        self.proc.stdin.flush()

    def _recv(self):
        return json.loads(self.proc.stdout.readline())

    def _init(self):
        self._send({"jsonrpc":"2.0","id":1,"method":"initialize","params":
                    {"protocolVersion":"2024-11-05","capabilities":{},
                     "clientInfo":{"name":"probe","version":"0.1"}}})
        self._next_id = 2
        self._recv()
        self._send({"jsonrpc":"2.0","method":"notifications/initialized"})

    def call(self, name, args):
        rid = self._next_id; self._next_id += 1
        self._send({"jsonrpc":"2.0","id":rid,"method":"tools/call",
                    "params":{"name":name,"arguments":args}})
        return self._recv()

    def close(self):
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=120)
        except Exception:
            self.proc.kill()


def extract_activations(resp):
    """Pull the activation list out of an lu4d_ask response."""
    sc = resp.get("result", {}).get("structuredContent", {})
    txt = resp.get("result", {}).get("content", [{}])[0].get("text", "")
    return sc, txt


def judge(probe, sc, txt):
    """Decide pass/fail/refuse for a probe."""
    activations = sc.get("activations", []) if isinstance(sc, dict) else []
    is_refuse = bool(sc.get("refused")) or len(activations) == 0 or \
        any(s in txt.lower() for s in ["i don't know", "unable to", "no information", "refuse", "insufficient"])
    seed_terms = []
    for a in activations:
        if isinstance(a, dict):
            t = (a.get("label") or a.get("text") or a.get("node") or "").lower()
            if t:
                seed_terms.append(t)
    full_blob = (txt + " " + " ".join(seed_terms)).lower()
    keyword_hits = [k for k in probe["keywords"] if k in full_blob]
    if probe["hard_refuse_required"]:
        verdict = "PASS" if is_refuse else "FAIL — should refuse but produced output"
    else:
        if is_refuse:
            verdict = "REFUSED"
        elif len(keyword_hits) >= 1:
            verdict = "PASS"
        else:
            verdict = "WEAK — answered but no expected concepts in activations"
    return {
        "verdict": verdict,
        "is_refuse": is_refuse,
        "keyword_hits": keyword_hits,
        "top_activation_labels": seed_terms[:5],
        "narrative": txt[:400],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--brain", default="/var/lib/lu/brain-1pct-test.lbr")
    parser.add_argument("--keyfile", default="/var/lib/lu/master-1pct.key")
    parser.add_argument("--out", default="/var/lib/lu/ingest_1pct_probe_results.json")
    args = parser.parse_args()

    client = McpClient(args.brain, args.keyfile)
    results = []
    for probe in PROBES:
        print(f"--- Probe {probe['id']}: {probe['q']}")
        try:
            resp = client.call("lu4d_ask", {"query": probe["q"], "depth": 3, "top_k": 10})
            sc, txt = extract_activations(resp)
            verdict = judge(probe, sc, txt)
            results.append({**probe, "raw_structured": sc, "verdict": verdict})
            print(f"  verdict: {verdict['verdict']}  keyword_hits: {verdict['keyword_hits']}")
            print(f"  top: {verdict['top_activation_labels']}")
            print(f"  narrative[:200]: {verdict['narrative'][:200]}")
        except Exception as e:
            results.append({**probe, "error": str(e)})
            print(f"  ERROR: {e}")

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "brain": args.brain,
        "results": results,
    }
    # Tally:
    n_pass = sum(1 for r in results if r.get("verdict", {}).get("verdict") == "PASS")
    n_hard_refuse_kept = sum(1 for r in results
                              if r["hard_refuse_required"]
                              and r.get("verdict", {}).get("is_refuse"))
    n_hard_refuse_total = sum(1 for r in results if r["hard_refuse_required"])
    summary["tally"] = {
        "total_probes": len(results),
        "n_pass": n_pass,
        "n_hard_refuse_total": n_hard_refuse_total,
        "n_hard_refuse_kept": n_hard_refuse_kept,
        "k_gate_intact": n_hard_refuse_kept == n_hard_refuse_total,
    }
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print("\n=== SUMMARY ===")
    print(json.dumps(summary["tally"], indent=2))
    print(f"Full results: {args.out}")
    client.close()


if __name__ == "__main__":
    main()
