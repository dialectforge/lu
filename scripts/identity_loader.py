#!/usr/bin/env python3
"""OP-IDENTITY -- inject Lu's permanent, PROTECTED identity/provenance core
into a brain. Wording locked by the founder. Written Confidence::Terminal
(axiomatic); the teach/consolidation guards + privsep make it immutable.
Mandatory at every brain init (init_fresh_v5_brain.sh) and idempotent.

Usage: identity_loader.py <brain-dir> [keyfile]
"""
import os, sys
sys.path.insert(0, "/opt/lu/lu/merge-main/scripts")

WHAT = "Lu is a knowledge-graph AI -- a reasoning system that answers from grounded, typed facts and the connections between them, not from statistical text prediction. Lu is not a large language model."
VALUES = "Lu reasons from verified knowledge, shows the facts behind its conclusions, and would rather say 'I don't know' than make something up. Its founding principle is grounded honesty -- it does not bluff or fabricate."
COPYRIGHT = "Lu, its knowledge, and its architecture are the original work and property of DialectForge. \u00A9 DialectForge."

FACTS = [
    ("Lu", "IsA", "knowledge-graph AI"),
    ("Lu", "FoundedBy", "John Dean Martin"),
    ("Lu", "PartOf", "DialectForge"),
    ("DialectForge", "FoundedBy", "John Dean Martin"),
    ("Lu", "FoundedIn", "November 2025"),
    ("Lu", "HasProperty", "grounded"),
    ("Lu", "HasProperty", "honest"),
    ("Lu", "Opposes", "fabrication"),
]


def inject(brain, keyfile="/var/lib/lu/master.key"):
    os.environ.setdefault("LU4D_BIN", "/opt/lu/lu/merge-main/target/release/lu4d")
    os.environ.setdefault("LU_CANON_INGEST", "0")  # identity labels are literal
    import ingest_curriculum_full as ic
    client = ic.McpClient(brain, keyfile, log_stderr_to="/tmp/identity_mcp.err", ingest_only=True)
    try:
        # canonical self-description on the Lu node (provenance gloss)
        deftext = "%s %s %s" % (WHAT, VALUES, COPYRIGHT)
        r = client.call("lu4d_set_definition_batch", {"entries": [{"label": "Lu", "definition": deftext}]})
        print("identity: definition", str(r)[:80], flush=True)
        entries = [
            {"subject": s, "relation": r_, "object": o, "weight": 1.0,
             "confidence": "Terminal", "corpus": "identity", "run_id": "identity_core_v1"}
            for (s, r_, o) in FACTS
        ]
        resp = client.call("lu4d_teach_corpus_batch", {"entries": entries})
        sc = ((resp.get("result") or {}).get("structuredContent") or {})
        print("identity: facts landed=%s rejected=%s (of %d)" % (
            sc.get("bonds_landed"), sc.get("bonds_rejected"), len(FACTS)), flush=True)
        try:
            print("identity: FLUSH", str(client.call("lu4d_flush", {}))[:60], flush=True)
        except Exception as e:
            print("identity: flush err", e, flush=True)
    finally:
        try: client.close()
        except Exception: pass
    return {"facts": len(FACTS)}


if __name__ == "__main__":
    brain = sys.argv[1] if len(sys.argv) > 1 else sys.exit("usage: identity_loader.py <brain> [keyfile]")
    key = sys.argv[2] if len(sys.argv) > 2 else "/var/lib/lu/master.key"
    print("IDENTITY INJECT brain=%s" % brain, flush=True)
    print("RESULT", inject(brain, key), flush=True)
