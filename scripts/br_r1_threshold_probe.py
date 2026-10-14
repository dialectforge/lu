#!/usr/bin/env python3
"""BR.R1 IDK-gate threshold probe extractor.

Reads a coordination_audit_*.json produced by
examples/audit_coordination_quality and emits a per-question
threshold-decision table tuned for the BR.R1 rebalance question:
  - Which threshold mode (A/B/C/D/Pass) the ask side fired
  - The query's max_activation and seed_count
  - The query CLASS for BR.R1 analysis:
      GENUINE_IDK     - expected_pass=false (Q11/Q12 Mars).
                        Lu's refusal discipline. Floor.
      SHOULD_PASS_NOW - expected_pass=true AND ask_classification
                        == IDK_INCORRECT. The recall-quality
                        ceiling we want to lift.
      PASSING         - expected_pass=true AND already PASS.
                        Must not regress.
      FALSE_PASS      - expected_pass=false AND verdict=Pass.
                        Worst-possible outcome; if any appear
                        post-fix the rebalance reverts.

Usage:
  br_r1_threshold_probe.py <coordination_audit.json>

Exits non-zero if the JSON is malformed.
"""
from __future__ import annotations

import json
import sys
from typing import Any


def classify_for_br_r1(r: dict) -> str:
    expected_pass = r["expected_pass"]
    ask_class = r["ask"]["classification"]
    verdict = r["ask"]["verdict"]  # "Pass" or refusal enum string
    if not expected_pass:
        if verdict == "Pass":
            return "FALSE_PASS"
        return "GENUINE_IDK"
    if ask_class == "REAL_CORRECT":
        return "PASSING"
    if ask_class == "IDK_INCORRECT":
        return "SHOULD_PASS_NOW"
    return f"OTHER:{ask_class}"


def emit(path: str) -> None:
    with open(path) as fh:
        doc = json.load(fh)
    print("id\tclass\tmode\tmax_act\tseed_count\task_classification")
    for r in doc["records"]:
        print("\t".join([
            r["id"],
            classify_for_br_r1(r),
            r["ask"]["mode_fired"],
            f"{r['ask']['max_activation']:.4f}",
            str(r["ask"]["seed_count"]),
            r["ask"]["classification"],
        ]))
    classes: dict = {}
    for r in doc["records"]:
        classes.setdefault(classify_for_br_r1(r), []).append(r["id"])
    print("---")
    for k in ("PASSING", "SHOULD_PASS_NOW", "GENUINE_IDK", "FALSE_PASS"):
        ids = classes.get(k, [])
        print(f"{k}\t{len(ids)}\t{','.join(ids)}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: br_r1_threshold_probe.py <coordination_audit.json>", file=sys.stderr)
        sys.exit(2)
    emit(sys.argv[1])
