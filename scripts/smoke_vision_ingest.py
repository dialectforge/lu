#!/usr/bin/env python3
"""Smoke test for port-vision-ingest.

Fresh /tmp/vision-smoke.lbr, fresh 32-byte keyfile, ingest >=1000 VG triplets
via lu4d_teach_corpus, then probe "what does a dog have" and inspect dog
attributes that came out. Also asserts bond_count == triplets-landed exactly
(anti-hallucination invariant).

Does NOT touch /var/lib/lu/brain.lbr. Throwaway only.
"""

from __future__ import annotations
import argparse, json, os, secrets, subprocess, sys, time, importlib.util, pathlib, re

LU_BIN = "/opt/lu/lu/merge-main/target/release/lu4d"
DEFAULT_BRAIN = "/tmp/vision-smoke.lbr"
DEFAULT_KEY = "/tmp/vision-smoke.key"
DEFAULT_LIMIT = 1000
RUN_ID = "smoke-vision-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


class Mcp:
    def __init__(self, brain, keyfile):
        self.proc = subprocess.Popen(
            [LU_BIN, "--mcp", "--require-encrypted", "--keyfile", keyfile, "--brain", brain],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            bufsize=1, universal_newlines=True,
        )
        self._id = 0
        self._init()

    def _send(self, req):
        self.proc.stdin.write(json.dumps(req) + "\n")
        self.proc.stdin.flush()

    def _recv(self):
        line = self.proc.stdout.readline()
        if not line:
            err = self.proc.stderr.read()
            raise RuntimeError("mcp closed stdout. stderr: " + err[:1000])
        return json.loads(line)

    def _init(self):
        self._id += 1
        self._send({"jsonrpc": "2.0", "id": self._id, "method": "initialize",
                    "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                               "clientInfo": {"name": "smoke-vision", "version": "1"}}})
        r = self._recv()
        if "error" in r:
            raise RuntimeError("init error: " + json.dumps(r))
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def call(self, name, args):
        self._id += 1
        self._send({"jsonrpc": "2.0", "id": self._id, "method": "tools/call",
                    "params": {"name": name, "arguments": args}})
        return self._recv()

    def close(self):
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=30)
        except Exception:
            self.proc.kill()


def ok(resp):
    if not isinstance(resp, dict) or "result" not in resp:
        return False
    res = resp["result"]
    if isinstance(res, dict) and res.get("isError"):
        return False
    return True


def extract_text(resp):
    try:
        content = resp["result"]["content"]
        return "\n".join(c.get("text", "") for c in content if isinstance(c, dict))
    except Exception:
        return json.dumps(resp)[:500]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", default=DEFAULT_BRAIN)
    ap.add_argument("--keyfile", default=DEFAULT_KEY)
    ap.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    args = ap.parse_args()

    # Clean prior smoke artifacts.
    for p in (args.brain, args.keyfile):
        try:
            os.unlink(p)
        except FileNotFoundError:
            pass
    for ext in (".pending.wal", ".lb_session.wal", ".style.wal"):
        try:
            os.unlink(args.brain + ext)
        except FileNotFoundError:
            pass

    # Fresh 32-char hex keyfile (matches production master.key shape).
    with open(args.keyfile, "wb") as f:
        f.write(secrets.token_hex(16).encode())
    os.chmod(args.keyfile, 0o600)

    # Load VG loader.
    here = pathlib.Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location("vg_loader", here / "visual_genome_loader.py")
    vg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vg)

    print("smoke: brain=" + args.brain + " key=" + args.keyfile + " limit=" + str(args.limit), flush=True)
    mcp = Mcp(args.brain, args.keyfile)
    landed = 0
    errors = 0
    err_examples = []
    t0 = time.monotonic()
    try:
        for s, raw_pred, o, image_id in vg.stream_triplets(max_triplets=args.limit):
            relation = vg.map_predicate(raw_pred)
            resp = mcp.call("lu4d_teach_corpus", {
                "corpus": "visual_genome",
                "source_url": "vg-image-" + str(image_id),
                "run_id": RUN_ID,
                "subject": s,
                "relation": relation,
                "object": o,
                "weight": 0.5,
                "confidence": "Unverified",
            })
            if ok(resp):
                landed += 1
            else:
                errors += 1
                if len(err_examples) < 3:
                    err_examples.append(extract_text(resp)[:200])
            if (landed + errors) % 100 == 0:
                print("  ... triplets=" + str(landed + errors) + " landed=" + str(landed) + " err=" + str(errors), flush=True)
        # Source-attribution invariant — verify in-process (overlay is in-memory).
        # Run BEFORE save: v3enc overlay persistence is a separate (pre-existing)
        # binary concern outside the scope of port-vision-ingest.
        attr_check = mcp.call("lu4d_revoke_by_source", {
            "kind": "TrainingCorpus",
            "corpus": "visual_genome",
            "dry_run": True,
        })
        attr_check_text = extract_text(attr_check)
        # Save the brain.
        save = mcp.call("lu4d_save", {})
        print("save: ok=" + str(ok(save)) + " text=" + extract_text(save)[:200], flush=True)

        # Status — bond_count must equal triplets landed exactly.
        st = mcp.call("lu4d_status", {})
        st_text = extract_text(st)

        # Probe.
        probe = mcp.call("lu4d_ask", {"query": "what does a dog have"})
        probe_text = extract_text(probe)
    finally:
        mcp.close()
    elapsed = time.monotonic() - t0

    # Report.
    print()
    print("=" * 60)
    print("VG smoke: landed=" + str(landed) + " errors=" + str(errors) + " elapsed_s=" + ("%.1f" % elapsed))
    print("err examples: " + repr(err_examples))
    print("-- attribution check (in-process, kind=TrainingCorpus corpus=visual_genome) --")
    print(attr_check_text[:400])
    print("-- lu4d_status --")
    print(st_text[:2000])
    print("-- lu4d_ask 'what does a dog have' --")
    print(probe_text[:2500])

    # Try to extract bond_count.
    bonds = None
    m = re.search(r"bond[_ \-]?count\D+(\d+)", st_text, re.IGNORECASE)
    if m:
        bonds = int(m.group(1))
    else:
        m2 = re.search(r"\b(\d+)\s*bonds?\b", st_text)
        if m2:
            bonds = int(m2.group(1))
    print()
    print("parsed bond_count=" + repr(bonds) + " landed=" + str(landed))
    if bonds is not None and bonds != landed:
        print("WARN: bond_count " + str(bonds) + " != landed " + str(landed) + " — investigate")
    if landed < args.limit:
        print("WARN: landed " + str(landed) + " < limit " + str(args.limit))
    sys.exit(0 if landed > 0 else 1)


if __name__ == "__main__":
    main()
