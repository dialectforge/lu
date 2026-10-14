#!/usr/bin/env bash
# verify_port.sh — Canonical verification rig for the lu4d MCP server.
#
# internal notes: this is the ONLY valid way to do live
# verification of a new port shipping to production. Fixture brains
# (the small in-memory graphs unit tests build via Brain::new() +
# teach()) are unit-test only — they do NOT exercise the archive-backed
# load path, the keyword index materialization, the same scoring
# distributions, or the same cascade thresholds the canonical brain
# produces. A port that passes all 780+ unit tests can still return
# "no anchor" against the canonical, as Issue A demonstrated.
#
# Drives the production lu_4_mcp binary against the canonical brain
# via JSON-RPC over stdio. For each (tool, args) probe, asserts that
# the response is NOT "no anchor" / "no chain" cold. Cold = the very
# first call after server start, before any warmup ask.
#
# Usage:
#   verify_port.sh <port_label> [<probe_file>]
#
# probe_file is a TSV with one probe per line: <tool>\t<query>
# Default: ./port_probes/<port_label>.tsv
#
# Exit codes:
#   0 — all probes returned real responses
#   1 — at least one probe returned "no anchor" / "no chain" / error
#   2 — usage / setup error

set -euo pipefail

PORT_LABEL="${1:-}"
if [[ -z "$PORT_LABEL" ]]; then
  echo "usage: $0 <port_label> [<probe_file>]" >&2
  exit 2
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
PROBE_FILE="${2:-${SCRIPT_DIR}/port_probes/${PORT_LABEL}.tsv}"

if [[ ! -f "$PROBE_FILE" ]]; then
  echo "error: probe file not found: $PROBE_FILE" >&2
  exit 2
fi

LU_BIN="${LU_BIN:-/usr/local/bin/lu_4_mcp}"
LU_BRAIN="${LU_BRAIN:-/var/lib/lu/brain.lbr}"
LU_KEYFILE="${LU_KEYFILE:-/var/lib/lu/master.key}"

if [[ ! -x "$LU_BIN" ]]; then
  echo "error: lu binary not executable: $LU_BIN" >&2
  exit 2
fi
if [[ ! -r "$LU_BRAIN" ]]; then
  echo "error: canonical brain not readable: $LU_BRAIN" >&2
  exit 2
fi

echo "=== verify_port: $PORT_LABEL ==="
echo "binary:  $LU_BIN ($(md5sum "$LU_BIN" | cut -d' ' -f1))"
echo "brain:   $LU_BRAIN ($(stat -c%s "$LU_BRAIN") bytes)"
echo "probes:  $PROBE_FILE"
echo

PYTHON3="${PYTHON3:-python3}"

export LU_BIN LU_BRAIN LU_KEYFILE PROBE_FILE PORT_LABEL

# Drive the MCP server in one Python invocation. Spawn lu_4_mcp
# --mcp --no-save, send initialize, then send each probe as a
# tools/call. Capture and assess each response.
"$PYTHON3" - <<'PY'
import json, os, subprocess, sys, time

LU_BIN = os.environ['LU_BIN']
LU_BRAIN = os.environ['LU_BRAIN']
LU_KEYFILE = os.environ['LU_KEYFILE']
PROBE_FILE = os.environ['PROBE_FILE']
PORT_LABEL = os.environ['PORT_LABEL']

probes = []
with open(PROBE_FILE) as f:
    for ln in f:
        ln = ln.strip()
        if not ln or ln.startswith('#'):
            continue
        parts = ln.split('\t')
        if len(parts) < 2:
            sys.stderr.write(f'malformed probe line: {ln}\n')
            sys.exit(2)
        tool, query = parts[0].strip(), parts[1].strip()
        probes.append((tool, query))

if not probes:
    sys.stderr.write('no probes found\n')
    sys.exit(2)

args = [LU_BIN, '--mcp', '--no-save', '--brain', LU_BRAIN, '--keyfile', LU_KEYFILE]
proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                        bufsize=1, universal_newlines=True)

def send(req):
    proc.stdin.write(json.dumps(req) + '\n')
    proc.stdin.flush()

def recv():
    line = proc.stdout.readline()
    if not line:
        sys.stderr.write('mcp server closed stdout unexpectedly\n')
        sys.stderr.write(proc.stderr.read())
        sys.exit(1)
    return json.loads(line)

# initialize
send({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"verify_port","version":"1"}}})
resp = recv()
if 'error' in resp:
    sys.stderr.write(f'initialize error: {resp}\n')
    sys.exit(1)
send({"jsonrpc":"2.0","method":"notifications/initialized"})

NO_ANCHOR_MARKERS = (
    'no anchor', 'No anchor', 'no chain', 'No chain',
    'don\'t have an anchor', 'don\'t have a reasoning chain',
    'no reasoning chain',
)

failures = 0
results = []
for i, (tool, query) in enumerate(probes, start=2):
    t0 = time.time()
    send({
        "jsonrpc":"2.0","id":i,"method":"tools/call",
        "params":{"name":tool,"arguments":(json.loads(query) if query.lstrip().startswith("{") else {"query":query})}
    })
    resp = recv()
    elapsed = (time.time() - t0) * 1000.0
    text = json.dumps(resp)
    bad = any(m in text for m in NO_ANCHOR_MARKERS)
    err = 'error' in resp
    status = 'OK'
    if err:
        status = 'ERROR'
        failures += 1
    elif bad:
        status = 'NO_ANCHOR'
        failures += 1
    print(f'  [{tool:14}] {query[:60]:<60}  {elapsed:>7.1f}ms  {status}')
    results.append((tool, query, status, elapsed))

proc.stdin.close()
proc.wait(timeout=10)

print()
print(f'=== {PORT_LABEL}: {len(probes) - failures}/{len(probes)} probes passed ===')
if failures:
    sys.exit(1)
PY
