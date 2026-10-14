#!/bin/bash
# BR.R1 IDK-gate threshold probe.
#
# Builds on top of the existing audit_coordination_quality harness
# (which already exposes per-question mode_fired + ask max_activation
# + ask_classification against the canonical brain) and adds a
# BR.R1-specific extraction:
#
#   - Class each question into PASSING / SHOULD_PASS_NOW /
#     GENUINE_IDK / FALSE_PASS
#   - Emit a per-question threshold-decision TSV
#   - Verify bit-stability across two runs (Principle 3 of CLAUDE.md)
#
# This probe is the BR.R1 acceptance instrument. Run it before any
# threshold change to capture the baseline; rerun after the change
# and diff the TSVs.
#
# Read-only against disk: audit_coordination_quality's
# isolate_audit_brain copy keeps brain.lbr untouched (lu4d's
# --no-save discipline, per feedback_canonical_brain_rule).
#
# Usage:
#   scripts/br_r1_threshold_probe.sh [out_dir]
#
# Defaults:
#   BRAIN=/var/lib/lu/brain.lbr
#   KEY=/var/lib/lu/master.key
#   out_dir=/tmp/br_r1_threshold_probe
#
# Exit codes:
#   0  bit-stable. BR.R1 TSV printed on stdout.
#   1  bit-instability across the two runs.
#   2  build / harness failure.
set -euo pipefail

BRAIN="${BRAIN:-/var/lib/lu/brain.lbr}"
KEY="${KEY:-/var/lib/lu/master.key}"
OUT="${1:-/tmp/br_r1_threshold_probe}"

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if ! command -v cargo >/dev/null 2>&1; then
    if [[ -x "$HOME/.cargo/bin/cargo" ]]; then
        export PATH="$HOME/.cargo/bin:$PATH"
    fi
fi

if [[ ! -f "$BRAIN" ]]; then
    echo "ERROR: canonical brain not found at $BRAIN" >&2
    exit 2
fi
if [[ ! -f "$KEY" ]]; then
    echo "ERROR: keyfile not found at $KEY" >&2
    exit 2
fi

rm -rf "$OUT"
mkdir -p "$OUT/run1" "$OUT/run2"

if ! cargo build --release --example audit_coordination_quality >"$OUT/build.log" 2>&1; then
    echo "ERROR: cargo build --release --example audit_coordination_quality failed" >&2
    tail -40 "$OUT/build.log" >&2
    exit 2
fi

BIN=./target/release/examples/audit_coordination_quality

for R in run1 run2; do
    if ! "$BIN" "$BRAIN" "$KEY" "$OUT/$R" >"$OUT/$R.stdout" 2>"$OUT/$R.stderr"; then
        echo "ERROR: $R harness invocation failed" >&2
        tail -40 "$OUT/$R.stderr" >&2
        exit 2
    fi
done

J1=$(ls -t "$OUT/run1"/coordination_audit_*.json 2>/dev/null | head -1)
J2=$(ls -t "$OUT/run2"/coordination_audit_*.json 2>/dev/null | head -1)
if [[ -z "$J1" || -z "$J2" ]]; then
    echo "ERROR: harness did not write coordination_audit_*.json" >&2
    exit 2
fi

T1="$OUT/run1.tsv"
T2="$OUT/run2.tsv"
python3 scripts/br_r1_threshold_probe.py "$J1" >"$T1"
python3 scripts/br_r1_threshold_probe.py "$J2" >"$T2"

if ! diff -q "$T1" "$T2" >/dev/null; then
    echo "ERROR: BR.R1 probe non-deterministic across two runs" >&2
    diff -u "$T1" "$T2" >&2 || true
    exit 1
fi

echo "# BR.R1 threshold probe (bit-stable across two runs)"
echo "# brain=$BRAIN"
echo "# repo HEAD=$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "# json=$J1"
cat "$T1"
