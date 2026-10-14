#!/bin/bash
# BR.C-Hyp.2.2 think-side probe.
#
# Runs `audit_coordination_quality` twice against the canonical brain.lbr
# (with --no-save discipline by way of the harness's per-process
# isolate_audit_brain copy) and captures the gate-12 signal
# (per-question think classification + coord_eligible) in a
# normalised TSV.
#
# Purpose: gate the BR.C-Hyp.2.2 ASK-only-gated coverage-promotion
# fix. The think path is supposed to be untouched; this probe makes
# any perturbation a halt criterion ahead of the change rather than
# a post-hoc finding.
#
# Self-verifies bit-stability across two runs (Principle 3 of
# CLAUDE.md: don't trust a measurement until it's bit-stable).
#
# Usage:
#   scripts/br_c_hyp_2_2_thinkside_probe.sh [out_dir]
#
# Defaults:
#   BRAIN=/var/lib/lu/brain.lbr
#   KEY=/var/lib/lu/master.key
#   out_dir=/tmp/br_c_hyp_2_2_probe
#
# Exit codes:
#   0  bit-stable. Gate-12 signal printed on stdout.
#   1  bit-instability detected (diff printed on stderr).
#   2  build / harness failure.
#
# Runtime: ~80s (two ~40s harness runs against canonical).
set -euo pipefail

BRAIN="${BRAIN:-/var/lib/lu/brain.lbr}"
KEY="${KEY:-/var/lib/lu/master.key}"
OUT="${1:-/tmp/br_c_hyp_2_2_probe}"

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Make cargo reachable from non-interactive shells.
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

# Newest JSON in each run dir.
J1=$(ls -t "$OUT/run1"/coordination_audit_*.json 2>/dev/null | head -1)
J2=$(ls -t "$OUT/run2"/coordination_audit_*.json 2>/dev/null | head -1)
if [[ -z "$J1" || -z "$J2" ]]; then
    echo "ERROR: harness did not write coordination_audit_*.json" >&2
    exit 2
fi

# Gate-12 signal extractor: per-question id, think_class, coord_eligible,
# ask_class, ask_max_act (4dp). Inline python so the probe is
# self-contained.
extract() {
    python3 - "$1" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
for r in d['records']:
    print("\t".join([
        r['id'],
        r['think']['classification'],
        "true" if r['coord_eligible'] else "false",
        r['ask']['classification'],
        f"{r['ask']['max_activation']:.4f}",
    ]))
PY
}

extract "$J1" >"$OUT/run1.tsv"
extract "$J2" >"$OUT/run2.tsv"

echo "# BR.C-Hyp.2.2 think-side probe — gate-12 signal"
echo "# brain=$BRAIN"
echo "# key=$KEY"
echo "# columns: id\tthink_class\tcoord_eligible\task_class\task_max_act"
echo "# (ask_max_act and ask_class are captured for context but the"
echo "#  gate-12 acceptance criterion is on think_class and coord_eligible.)"
cat "$OUT/run1.tsv"
echo

if diff -q "$OUT/run1.tsv" "$OUT/run2.tsv" >/dev/null; then
    COORD=$(awk -F'\t' '$3 == "true" {n++} END {print n+0}' "$OUT/run1.tsv")
    echo "BIT-STABLE: gate-12 signal identical across two runs."
    echo "coord_eligible_count=$COORD"
    exit 0
else
    echo "FAIL: gate-12 signal DIFFERS between runs (probe is non-deterministic):" >&2
    diff "$OUT/run1.tsv" "$OUT/run2.tsv" >&2
    exit 1
fi
