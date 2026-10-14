#!/usr/bin/env python3
# OP-GLOSS PART 0 data re-set: restore Batch-B concept glosses dropped by the
# breadthD compaction (seq-dedup). CC0 Wikidata descriptions (captured via the
# srcpullB --dry fetch; identical source). Concepts ONLY (no date bonds) --
# bounded scope. set_definition create+update on a WORKING COPY.
import os, sys, json, argparse
sys.path.insert(0, "/opt/lu/lu/merge-main/scripts")

CDEFS = [
  ("radioactivity", "process by which an unstable atom emits radiation"),
  ("radioactive decay", "emission of subatomic particles by the decay of an atomic nucleus"),
  ("nuclear fission", "nuclear reaction splitting the nucleus of an atom into multiple parts"),
  ("quantum field theory", "theoretical framework combining classical field theory, special relativity, and quantum mechanics"),
  ("wave-particle duality", "conclusion that quantum objects behave at times like particles and at times like waves"),
  ("quantum entanglement", "correlation between measurements of quantum subsystems, even when spatially separated"),
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", required=True)
    ap.add_argument("--keyfile", required=True)
    a = ap.parse_args()
    os.environ["LU4D_BIN"] = "/opt/lu/lu/merge-main/target/release/lu4d"
    os.environ["LU_CANON_INGEST"] = "1"
    import importlib
    ic = importlib.import_module("ingest_curriculum_full")
    cl = ic.McpClient(a.brain, a.keyfile, log_stderr_to="/tmp/reset_concepts.err", ingest_only=True)
    entries = [{"label": l, "definition": d} for (l, d) in CDEFS]
    r1 = cl.call("lu4d_set_definition_batch", {"entries": entries, "update": False})
    r2 = cl.call("lu4d_set_definition_batch", {"entries": entries, "update": True})
    try: fl = str(cl.call("lu4d_flush", {}))[:120]
    except Exception as ex: fl = "flush_err %s" % ex
    print("CREATE:", json.dumps(r1)[:400])
    print("UPDATE:", json.dumps(r2)[:400])
    print("FLUSH :", fl)
    cl.close()

if __name__ == "__main__":
    main()
