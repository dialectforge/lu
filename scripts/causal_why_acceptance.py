#!/usr/bin/env python3
"""OP-CAUSAL-WHY -- causal-why synthesis acceptance suite.

Guards the honest causal-why shape in src/synthesis.rs::causal_why, composed
ONLY from real Causes/LeadsTo edges:
  * what-causes-X        -> incoming causes, multifactorial calibration when many
  * what-does-X-cause    -> outgoing effects
  * does-X-cause-Y       -> yes/no verify, honest no-link refusal
  * why-does-X-cause-Y   -> NEW: confirm the real X->Y edge, or a transitive
                            Causes/LeadsTo chain X->Z->Y, else honest-refuse
                            (never the generic slot-empty multi-hypothesis template)
Every stated link is a real cited edge -- never a fabricated cause or mechanism.

Drives the LIVE service (POST /auth, then POST /mcp lu4d_ask).
    sudo -E python3 scripts/causal_why_acceptance.py [--port 8787]
Exit 0 = all passed. Non-zero = number of failures.
"""
import argparse, json, os, sys, urllib.request

def post(url, payload, token=None, timeout=180):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST")
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())

def extract_text(resp):
    r = resp.get("result", resp)
    content = r.get("content")
    if isinstance(content, list):
        parts = [c.get("text", "") for c in content if isinstance(c, dict)]
        if parts:
            return "\n".join(parts)
    for k in ("text", "answer", "narration"):
        if isinstance(r.get(k), str):
            return r[k]
    return json.dumps(r)

# (id, query, kind, substrings, why)
#   kind "all"  -> every substring must be present (case-insensitive)
#   kind "none" -> no substring may be present
PROBES = [
    # ---- existing causal branches: regression guards -----------------------
    ("CW-cancer-multifactor", "what causes cancer", "all",
     ["isn't a single cause", "multifactorial", "contributing factors", "cigarette"],
     "Multifactor fan-in calibration on a >=4-cause subject, unchanged."),
    ("CW-virus-effect", "what does a virus cause", "all",
     ["disease", "->"],
     "Outgoing-effects branch, cited edges."),
    ("CW-verify-present", "does a cigarette cause cancer", "all",
     ["yes -- cigarette causes cancer"],
     "Yes/no verify confirms a real edge."),
    ("CW-verify-refuse", "does a vaccine cause autism", "all",
     ["no causal link", "vaccine", "autism"],
     "Honest no-link refusal -- misinfo cut held, no fabrication."),
    ("CW-verify-refuse-none", "does a vaccine cause autism", "none",
     ["yes --"],
     "Never affirms the fabricated vaccine->autism link."),
    ("CW-effect-refuse", "what causes gravity", "all",
     ["no cause on record", "gravity"],
     "No causal edge on subject -> honest refusal, never invented."),
    # ---- NEW why-does-X-cause-Y explain-link branch ------------------------
    ("CW-whylink-direct", "why does cigarette cause cancer", "all",
     ["cigarette is on record as a cause of cancer", "mechanism"],
     "Confirms the real X->Y edge and is honest it holds the link, not the mechanism."),
    ("CW-whylink-nogarbage", "why does asbestos cause cancer", "none",
     ["relates to  via", "considered 7 alternatives", "i don't have anything solid"],
     "Terminates causally -- kills the slot-empty multi-hypothesis template."),
    ("CW-whylink-asbestos", "why does asbestos cause cancer", "all",
     ["asbestos is on record as a cause of cancer"],
     "Second direct-edge confirm."),
    ("CW-whylink-leadsto", "why does air pollution cause acid rain", "all",
     ["air pollution is on record as leading to acid rain"],
     "LeadsTo edge phrased 'leading to', not 'a cause of'."),
    ("CW-whylink-transitive", "why does a virus cause death", "all",
     ["virus causes disease", "in turn", "death"],
     "No direct virus->death; honest 2-hop chain virus->disease->death, both cited."),
    ("CW-whylink-transitive-cite", "why does a virus cause death", "all",
     ["virus -causes-> disease", "disease -leadsto-> death"],
     "Both real edges cited in the chain."),
    ("CW-whylink-refuse", "why does a cigarette cause happiness", "all",
     ["no causal link from cigarette to happiness"],
     "No X->Y link and no chain -> honest refusal, never a fabricated why."),
    ("CW-whylink-refuse-none", "why does a cigarette cause happiness", "none",
     ["on record as", "in turn"],
     "Refusal must not leak a confirm or a chain."),
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8787)
    args = ap.parse_args()
    base = f"http://127.0.0.1:{args.port}"
    pw = os.environ.get("LU_GUEST_PASSPHRASE")
    if not pw:
        print("FATAL: LU_GUEST_PASSPHRASE not set", file=sys.stderr); return 99
    token = post(f"{base}/auth", {"passphrase": pw}).get("token")
    if not token:
        print("FATAL: auth failed", file=sys.stderr); return 99
    print(f"=== OP-CAUSAL-WHY acceptance: {len(PROBES)} probes ===\n")
    failures = 0
    for pid, q, kind, arg, why in PROBES:
        try:
            resp = post(f"{base}/mcp", {"method": "tools/call",
                "params": {"name": "lu4d_ask", "arguments": {"query": q}}}, token=token)
            raw = extract_text(resp); err = None
        except Exception as e:
            raw = ""; err = repr(e)
        low = (raw or "").lower()
        if err:
            ok, detail = False, f"transport error: {err}"
        elif kind == "all":
            miss = [a for a in arg if a.lower() not in low]
            ok, detail = (not miss), f"missing {miss}"
        elif kind == "none":
            bad = [a for a in arg if a.lower() in low]
            ok, detail = (not bad), f"must not contain {bad}"
        else:
            ok, detail = False, f"unknown kind {kind}"
        flag = "PASS" if ok else "FAIL"
        if not ok:
            failures += 1
        print(f"[{flag}] {pid}: {q}")
        print(f"        -> {raw}")
        if not ok:
            print(f"        !! {detail}")
    print(f"\n=== {len(PROBES)-failures}/{len(PROBES)} passed ===")
    return failures

if __name__ == "__main__":
    sys.exit(main())
