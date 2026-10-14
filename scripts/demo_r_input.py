#!/usr/bin/env python3
"""Port R input — empirical demo against the live encrypted brain.

Flow:
  1. Write a tiny 1x1 PNG to /tmp/lu_image_test.png.
  2. Spawn lu_4_mcp --mcp against canonical brain.
  3. lu4d_status -> bond_count_before  (must be 10_819_438).
  4. lu4d_image_ingest { TeachCorpus { corpus: 'test/r-input-demo' } }
     with stub extractor -> exactly 1 bond.
  5. lu4d_status -> bond_count_during == bond_count_before + 1.
  6. lu4d_revoke_by_source { kind: 'training', corpus: 'test/r-input-demo',
                              dry_run: false } -> revokes the bond.
  7. lu4d_status -> bond_count_after == bond_count_before.

Exits 0 on success, 1 on any assertion failure.
"""
import json, os, subprocess, sys

LU_BIN = os.environ.get('LU_BIN', '/usr/local/bin/lu_4_mcp')
LU_BRAIN = os.environ.get('LU_BRAIN', '/var/lib/lu/brain.lbr')
LU_KEYFILE = os.environ.get('LU_KEYFILE', '/var/lib/lu/master.key')
EXPECTED_BOND_COUNT = 10_819_438

IMG_PATH = '/tmp/lu_image_test.png'
PNG_BYTES = bytes([
    0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,
    0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,
    0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
    0x08, 0x06, 0x00, 0x00, 0x00, 0x1F, 0x15, 0xC4,
    0x89, 0x00, 0x00, 0x00, 0x0A, 0x49, 0x44, 0x41,
    0x54, 0x78, 0x9C, 0x63, 0x00, 0x01, 0x00, 0x00,
    0x05, 0x00, 0x01, 0x0D, 0x0A, 0x2D, 0xB4, 0x00,
    0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE,
    0x42, 0x60, 0x82,
])
with open(IMG_PATH, 'wb') as f:
    f.write(PNG_BYTES)

args = [LU_BIN, '--mcp', '--no-save', '--brain', LU_BRAIN, '--keyfile', LU_KEYFILE]
proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE, bufsize=1, universal_newlines=True)

def send(req):
    proc.stdin.write(json.dumps(req) + '\n'); proc.stdin.flush()

def recv():
    line = proc.stdout.readline()
    if not line:
        sys.stderr.write('mcp closed unexpectedly\n')
        sys.stderr.write(proc.stderr.read())
        sys.exit(1)
    return json.loads(line)

send({'jsonrpc':'2.0','id':1,'method':'initialize',
      'params':{'protocolVersion':'2024-11-05','capabilities':{},
                'clientInfo':{'name':'demo_r_input','version':'1'}}})
resp = recv()
if 'error' in resp:
    sys.stderr.write(f'initialize error: {resp}\n'); sys.exit(1)
send({'jsonrpc':'2.0','method':'notifications/initialized'})

failures = []

def call(call_id, name, arguments):
    send({'jsonrpc':'2.0','id':call_id,'method':'tools/call',
          'params':{'name':name,'arguments':arguments}})
    return recv()

def structured(resp):
    return resp['result']['structuredContent']

# 1. status -> baseline
resp = call(2, 'lu4d_status', {})
s = structured(resp)
bond_count_before = s.get('bond_count') or s.get('bonds') or s.get('total_bonds')
print(f'[1] bond_count_before = {bond_count_before}')
if bond_count_before != EXPECTED_BOND_COUNT:
    failures.append(f'bond_count_before {bond_count_before} != expected {EXPECTED_BOND_COUNT}')

# 2. extractor status
resp = call(3, 'lu4d_image_extractor_status', {})
s = structured(resp)
print(f'[2] extractor_kind = {s["extractor_kind"]}, max_size = {s["image_max_size_bytes"]}, mimes = {s["image_supported_mime_types"]}')
if s['extractor_kind'] != 'stub':
    failures.append(f'extractor_kind {s["extractor_kind"]!r} != stub')
if s['image_max_size_bytes'] != 25_000_000:
    failures.append(f'image_max_size_bytes {s["image_max_size_bytes"]} != 25_000_000')

# 3. ingest
resp = call(4, 'lu4d_image_ingest', {
    'path': IMG_PATH,
    'route': {'TeachCorpus': {'corpus': 'test/r-input-demo', 'run_id': 'demo'}},
})
if 'error' in resp:
    failures.append(f'image_ingest error: {resp["error"]}')
    print(f'[3] FAILED ingest: {resp["error"]}')
else:
    s = structured(resp)
    print(f'[3] image_ingest: route={s["route"]}, bonds_formed={s["bonds_formed"]}, mime={s["mime_type"]}, extractor={s["extractor"]}')
    if s['bonds_formed'] != 1:
        failures.append(f'bonds_formed {s["bonds_formed"]} != 1')

# 4. status during
resp = call(5, 'lu4d_status', {})
s = structured(resp)
bond_count_during = s.get('bond_count') or s.get('bonds') or s.get('total_bonds')
print(f'[4] bond_count_during = {bond_count_during}')
if bond_count_during != bond_count_before + 1:
    failures.append(f'bond_count_during {bond_count_during} != before+1 {bond_count_before+1}')

# 5. revoke
resp = call(6, 'lu4d_revoke_by_source', {
    'kind': 'training',
    'corpus': 'test/r-input-demo',
    'dry_run': False,
})
if 'error' in resp:
    failures.append(f'revoke error: {resp["error"]}')
    print(f'[5] FAILED revoke: {resp["error"]}')
else:
    s = structured(resp)
    print(f'[5] revoke: matched={s.get("matched")}, revoked={s.get("revoked") or s.get("affected") or s.get("removed")}')

# 6. status after
resp = call(7, 'lu4d_status', {})
s = structured(resp)
bond_count_after = s.get('bond_count') or s.get('bonds') or s.get('total_bonds')
print(f'[6] bond_count_after = {bond_count_after}')
if bond_count_after != bond_count_before:
    failures.append(f'bond_count_after {bond_count_after} != before {bond_count_before}')

proc.stdin.close()
proc.wait(timeout=15)

print()
if failures:
    print('FAIL:')
    for f in failures:
        print(f'  - {f}')
    sys.exit(1)
else:
    print('OK — bond_count cycle preserved at', EXPECTED_BOND_COUNT)
    sys.exit(0)
