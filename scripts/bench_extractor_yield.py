#!/usr/bin/env python3
"""Offline extractor yield bench (B0 baseline / B3 verify).
Replicates the ingest_wikipedia per-article pipeline WITHOUT touching any
brain: smart_extractor.extract_bonds_from_wikipedia_article -> union with
extract_response_bonds_multi_sentence -> collapse_to_richest_relation.
Counts emitted bonds/article, per-relation, per-layer. Deterministic stream.
Usage: bench_extractor_yield.py N
"""
import sys, os, json, collections
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))
sys.path.insert(0, "/opt/lu/lu/v4/scripts")
from smart_extractor import extract_bonds_from_wikipedia_article, collapse_to_richest_relation
from ingest_curriculum_full import extract_response_bonds_multi_sentence
from datasets import load_dataset

N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)

tot_bonds = 0
rel = collections.Counter()
layer = collections.Counter()
art = 0
zero_art = 0
for ex in ds:
    if art >= N: break
    title = ex.get("title",""); text = ex.get("text",""); url = ex.get("url","")
    bonds = extract_bonds_from_wikipedia_article(ex)
    extra = extract_response_bonds_multi_sentence(title, text, url)
    if extra:
        seen = set((b["subject"].lower(), b["relation"], b["object"].lower()) for b in bonds)
        for eb in extra:
            k = (eb["subject"].lower(), eb["relation"], eb["object"].lower())
            if k in seen: continue
            seen.add(k); bonds.append(eb)
    bonds = collapse_to_richest_relation(bonds)
    if not bonds: zero_art += 1
    tot_bonds += len(bonds)
    for b in bonds:
        rel[b["relation"]] += 1
        layer[b.get("layer","?")] += 1
    art += 1

print(json.dumps({
  "articles": art,
  "total_bonds": tot_bonds,
  "bonds_per_article": round(tot_bonds/art, 3) if art else 0,
  "zero_bond_articles": zero_art,
  "by_relation": dict(rel.most_common()),
  "by_layer": dict(layer.most_common()),
}, indent=2)); sys.stdout.flush(); os._exit(0)
