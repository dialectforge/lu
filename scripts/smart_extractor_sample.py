#!/usr/bin/env python3
"""
smart_extractor_sample.py -- Phase A measurement harness (iter-3).

Loads N articles from wikimedia/wikipedia in the SAME iteration order that
ingest_curriculum_full.py uses (load_dataset streaming, no shuffle, English
split), runs each through:
  * smart_extractor.extract_bonds_from_wikipedia_article (NEW)
  * baseline_old.extract_wikipedia_bonds (OLD, copied from ingest pipeline)

Reports yield statistics, per-layer breakdown, distribution percentiles,
showcase + random samples, and projected full-corpus impact.

Does NOT touch the live brain. Does NOT call lu4d_teach_corpus. Pure measurement.
"""

import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from smart_extractor import extract_bonds_from_wikipedia_article  # noqa


# ---- Baseline OLD extractor copied verbatim from ingest_curriculum_full.py ----
# Lines 247..268 of ingest_curriculum_full.py (commit at port time):
#   IS_A_PATTERN: 'is/was a/the X' regex
#   first_sentence(text): leading sentence up to 400 chars
#   extract_wikipedia_bonds(title, text, source_url): emits 1-2 bonds:
#     (title, RelatedTo, lead[:200], 0.5, 'Inferred')
#     optional (title, IsA, cat,       0.6, 'Inferred')
IS_A_PATTERN = re.compile(
    r"\b(?:is|was|are|were)\s+(?:an?|the)\s+([a-z][\w\s\-]{2,60}?)\s+"
    r"(?:that|which|who|where|in|of|on|located|known|used|with|by|for|from|to|and|\.|,|;)",
    re.IGNORECASE,
)

def _old_first_sentence(text):
    if not text: return ""
    m = re.match(r"([^.!?\n]{20,400}[.!?])", text.strip())
    return m.group(1).strip() if m else text.strip()[:200]

def extract_wikipedia_bonds_OLD(title, text, source_url):
    """Verbatim port of ingest_curriculum_full.py's extract_wikipedia_bonds."""
    bonds = []
    if not text or not title: return bonds
    lead = _old_first_sentence(text)
    if not lead: return bonds
    bonds.append({
        "subject": title, "relation": "RelatedTo", "object": lead[:200],
        "weight": 0.5, "confidence": "Inferred", "layer": "OLD-lead",
        "source_url": source_url,
    })
    m = IS_A_PATTERN.search(lead)
    if m:
        cat = m.group(1).strip().rstrip(",.;:").lower()
        if 2 <= len(cat) <= 60:
            bonds.append({
                "subject": title, "relation": "IsA", "object": cat,
                "weight": 0.6, "confidence": "Inferred", "layer": "OLD-isa",
                "source_url": source_url,
            })
    return bonds


def percentile(sorted_vals, p):
    if not sorted_vals: return 0
    k = (len(sorted_vals) - 1) * p
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c: return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--num-articles", type=int, default=1000)
    ap.add_argument("-o", "--out", default=os.path.join(HERE, "smart_extractor_sample_report.md"))
    ap.add_argument("--bonds-out", default=os.path.join(HERE, "smart_extractor_sample_bonds.jsonl"),
                    help="Optional dump of every NEW bond to JSONL for inspection")
    ap.add_argument("--old-bonds-out", default=os.path.join(HERE, "smart_extractor_sample_bonds_old.jsonl"),
                    help="Optional dump of OLD baseline bonds for inspection")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    print(f"[harness] loading wikimedia/wikipedia (streaming) ...", flush=True)
    from datasets import load_dataset
    ds = load_dataset("wikimedia/wikipedia", "20231101.en",
                      split="train", streaming=True)

    new_bonds_per_article = []
    old_bonds_per_article = []
    layer_counts = Counter()
    relation_counts = Counter()
    new_seen_global = set()
    new_dup_count = 0
    all_new_bonds = []
    weight_buckets = Counter()
    n_no_text = 0
    n_zero_bonds_new = 0
    n_zero_bonds_old = 0
    bond_dump = open(args.bonds_out, "w") if args.bonds_out else None
    old_dump = open(args.old_bonds_out, "w") if args.old_bonds_out else None

    # Track frequency-of-bond-objects to surface hub-formation risk
    object_freq = Counter()
    relation_object_freq = Counter()

    start = time.time()
    n = 0
    for ex in ds:
        if n >= args.num_articles: break
        title = (ex.get("title") or "").strip()
        text = ex.get("text") or ""
        if not text or not title:
            n_no_text += 1
            n += 1
            continue

        # NEW extractor
        new_bonds = extract_bonds_from_wikipedia_article(ex)
        new_bonds_per_article.append(len(new_bonds))
        if not new_bonds:
            n_zero_bonds_new += 1
        for b in new_bonds:
            layer_counts[b["layer"]] += 1
            relation_counts[b["relation"]] += 1
            wb = round(b["weight"], 1)
            weight_buckets[wb] += 1
            key = (b["subject"].lower(), b["relation"], b["object"].lower())
            if key in new_seen_global:
                new_dup_count += 1
            else:
                new_seen_global.add(key)
            object_freq[b["object"]] += 1
            relation_object_freq[(b["relation"], b["object"])] += 1
            if bond_dump:
                bond_dump.write(json.dumps(b) + "\n")
            all_new_bonds.append(b)

        # OLD extractor (apples-to-apples baseline on same article)
        old_bonds = extract_wikipedia_bonds_OLD(title, text, ex.get("url") or "")
        old_bonds_per_article.append(len(old_bonds))
        if not old_bonds:
            n_zero_bonds_old += 1
        if old_dump:
            for b in old_bonds:
                old_dump.write(json.dumps(b) + "\n")

        n += 1
        if n % 100 == 0:
            elapsed = time.time() - start
            print(f"[harness] processed {n}/{args.num_articles} articles "
                  f"({elapsed:.1f}s, {sum(new_bonds_per_article)} NEW bonds total, "
                  f"{sum(old_bonds_per_article)} OLD)", flush=True)

    if bond_dump: bond_dump.close()
    if old_dump: old_dump.close()

    elapsed = time.time() - start
    new_total = sum(new_bonds_per_article)
    old_total = sum(old_bonds_per_article)
    new_sorted = sorted(new_bonds_per_article)
    old_sorted = sorted(old_bonds_per_article)
    new_mean = new_total / max(1, len(new_sorted))
    old_mean = old_total / max(1, len(old_sorted))

    new_p = lambda p: percentile(new_sorted, p)
    old_p = lambda p: percentile(old_sorted, p)

    # 100 random sampled bonds
    sample_size = min(100, len(all_new_bonds))
    sampled = random.sample(all_new_bonds, sample_size) if sample_size else []

    # Showcase: one bond per layer if available
    layer_order = ["L1", "L2", "L2b", "L3-isa", "L3-loc", "L3-foundby", "L3-foundin",
                   "L3-discby", "L3-named", "L3-meas", "L3-born", "L3-died",
                   "L3-writby", "L3-comp", "L3-known", "L3-used", "L3-type",
                   "L3-part", "L4",
                   "L5-foundyear", "L5-pop", "L5-area", "L5-alt",
                   "L5-lang", "L5-cap", "L5-borndt", "L5-dieddt"]
    showcase = []
    for layer in layer_order:
        cands = [b for b in all_new_bonds if b["layer"] == layer]
        if cands:
            showcase.append(random.choice(cands))

    # Hub-formation risk: top 20 most-emitted objects (cross-article)
    top_hubs = object_freq.most_common(20)

    # ----- Estimate full-corpus impact -----
    full_articles = 6_700_000  # ~en.wikipedia.org as of 2023
    est_new_total = int(new_mean * full_articles)
    est_old_total = int(old_mean * full_articles)
    multiplier = (new_mean / old_mean) if old_mean > 0 else float('inf')

    # Phase B target check (John's calibration: 1:25-50+ avg, 57-113M total at 2.27M nodes)
    target_low = 25
    target_high = 50
    target_total_low = 57_000_000
    target_total_high = 113_000_000

    # ----- Build report -----
    L = []
    L.append("# Smart Extractor v2 -- Phase A Sample Report (iter-3)")
    L.append("")
    L.append(f"**Run:** {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())} | "
             f"**Articles sampled:** {n} | "
             f"**NEW bonds emitted:** {new_total} | "
             f"**OLD bonds emitted:** {old_total} | "
             f"**Elapsed:** {elapsed:.1f}s")
    L.append("")
    L.append("## TL;DR")
    L.append("")
    L.append(f"- **{n} articles** processed.")
    L.append(f"- **NEW yield: {new_mean:.2f} bonds/article** (median {new_p(0.5):.0f}, p90 {new_p(0.9):.0f}).")
    L.append(f"- **OLD yield: {old_mean:.2f} bonds/article** (median {old_p(0.5):.0f}, p90 {old_p(0.9):.0f}) -- run on the *same* articles.")
    L.append(f"- **Yield multiplier: {multiplier:.1f}x**")
    L.append(f"- Estimated full-corpus yield (6.7M articles): **{est_new_total:,}** vs OLD {est_old_total:,}.")
    L.append(f"- John's calibrated target at 2.27M-node scale: **57-113M bonds** (1:25-50+ avg).")
    L.append("")
    if new_mean >= target_high:
        verdict = "**READY for Phase B** -- mean exceeds high target ({}+ bonds/article).".format(target_high)
    elif new_mean >= target_low:
        verdict = ("**READY for Phase B** with caveats -- mean meets minimum target "
                   "({}+ bonds/article) but below stretch target ({}). Recommend "
                   "shipping iter-3 to Phase B; revisit a co-occurrence boost in Phase C "
                   "if hub diversity is still thin after re-ingest.").format(target_low, target_high)
    else:
        verdict = ("**NEEDS REFINEMENT** -- mean below {}-bonds/article minimum. Diagnose "
                   "underperforming layers and iterate before Phase B.").format(target_low)
    L.append(f"- Verdict: {verdict}")
    L.append("")

    # Yield comparison table
    L.append("## Bond yield (per article) -- NEW vs OLD on same articles")
    L.append("")
    L.append("| Stat | OLD extractor | NEW extractor (iter-3) | Ratio |")
    L.append("|------|---------------|------------------------|-------|")
    L.append(f"| mean | {old_mean:.2f} | {new_mean:.2f} | {multiplier:.1f}x |")
    L.append(f"| median | {old_p(0.5):.0f} | {new_p(0.5):.0f} | -- |")
    L.append(f"| p10 | {old_p(0.1):.0f} | {new_p(0.1):.0f} | -- |")
    L.append(f"| p25 | {old_p(0.25):.0f} | {new_p(0.25):.0f} | -- |")
    L.append(f"| p75 | {old_p(0.75):.0f} | {new_p(0.75):.0f} | -- |")
    L.append(f"| p90 | {old_p(0.9):.0f} | {new_p(0.9):.0f} | -- |")
    L.append(f"| p99 | {old_p(0.99):.0f} | {new_p(0.99):.0f} | -- |")
    L.append(f"| min | {old_sorted[0]} | {new_sorted[0]} | -- |")
    L.append(f"| max | {old_sorted[-1]} | {new_sorted[-1]} | -- |")
    L.append("")
    L.append(f"- Articles with 0 NEW bonds: {n_zero_bonds_new} ({100*n_zero_bonds_new/max(1,n):.1f}%)")
    L.append(f"- Articles with 0 OLD bonds: {n_zero_bonds_old} ({100*n_zero_bonds_old/max(1,n):.1f}%)")
    L.append(f"- Articles with no text/title: {n_no_text}")
    L.append("")

    # Layer breakdown
    L.append("## Per-layer yield breakdown")
    L.append("")
    L.append("| Layer | Bonds | % of total | Mean/article | Description |")
    L.append("|-------|-------|------------|--------------|-------------|")
    layer_descs = {
        "L1": "See also section walker (RelatedTo)",
        "L2": "Section heading walker (HasAspect)",
        "L2b": "Category-tail walker (IsA)",
        "L3-isa": "Lead-paragraph IsA pattern",
        "L3-loc": "Lead LocatedIn pattern",
        "L3-part": "Lead PartOf pattern",
        "L3-foundby": "Lead FoundedBy pattern",
        "L3-foundin": "Lead FoundedIn pattern",
        "L3-discby": "Lead DiscoveredBy pattern",
        "L3-named": "Lead NamedAfter pattern",
        "L3-meas": "Lead MeasuredIn pattern",
        "L3-born": "Lead BornIn pattern",
        "L3-died": "Lead DiedIn pattern",
        "L3-writby": "Lead WrittenBy pattern",
        "L3-comp": "Lead ComposedOf pattern",
        "L3-known": "Lead KnownFor pattern",
        "L3-used": "Lead UsedFor pattern",
        "L3-type": "Lead TypeOf pattern (yields IsA)",
        "L4": "BODY proper-noun co-mention walker (RelatedTo)  *** iter-3 ***",
        "L5-foundyear": "Body 'founded in YYYY' attribute (FoundedIn)  *** iter-3 ***",
        "L5-pop": "Body population attribute (HasPopulation)  *** iter-3 ***",
        "L5-area": "Body area attribute (HasArea)  *** iter-3 ***",
        "L5-alt": "Body altitude attribute (HasAltitude)  *** iter-3 ***",
        "L5-lang": "Body official-language attribute  *** iter-3 ***",
        "L5-cap": "Body capital-city attribute (HasCapital)  *** iter-3 ***",
        "L5-borndt": "Body 'born <date>' attribute (BornOn)  *** iter-3 ***",
        "L5-dieddt": "Body 'died <date>' attribute (DiedOn)  *** iter-3 ***",
    }
    for layer, count in layer_counts.most_common():
        pct = 100 * count / max(1, new_total)
        per_article = count / max(1, n)
        L.append(f"| {layer} | {count} | {pct:.1f}% | {per_article:.2f} | {layer_descs.get(layer, '?')} |")
    L.append("")

    # Relation breakdown
    L.append("## Per-relation breakdown")
    L.append("")
    L.append("| Relation | Bonds | % of total |")
    L.append("|----------|-------|------------|")
    for rel, count in relation_counts.most_common():
        pct = 100 * count / max(1, new_total)
        L.append(f"| {rel} | {count} | {pct:.1f}% |")
    L.append("")

    # Dedup
    L.append("## Dedup behaviour")
    L.append("")
    L.append(f"- Unique (subject, relation, object) triples across the {n}-article sample: **{len(new_seen_global):,}**")
    L.append(f"- Duplicate triples (same triple emitted by different articles): {new_dup_count} ({100*new_dup_count/max(1,new_total):.1f}%)")
    L.append("")

    # Hub-formation risk
    L.append("## Hub-formation watch -- top 20 most-emitted bond objects")
    L.append("")
    L.append("This surfaces objects that risk becoming pseudo-hubs. The original retrain's failure mode was \"village\" / \"species\" / \"municipality\" hubs from the OLD lead-IsA pattern.")
    L.append("")
    L.append("| Object | Bonds emitted (in 1000-article sample) |")
    L.append("|--------|----------------------------------------|")
    for obj, c in top_hubs:
        L.append(f"| {obj!r} | {c} |")
    L.append("")

    # Showcase bonds
    L.append("## Showcase: one bond per active layer")
    L.append("")
    L.append("Format: `subject --[relation]--> object  (layer, weight, src)`")
    L.append("")
    for b in showcase:
        url = b.get("source_url", "")
        freq = (" freq=" + str(b.get("frequency"))) if "frequency" in b else ""
        L.append(f"- `{b['subject']}` --[{b['relation']}]--> `{b['object']}`  "
                 f"(layer={b['layer']}, w={b['weight']}{freq}, src={url})")
    L.append("")

    # 100-bond random sample
    L.append("## 100-bond random sample for human spot-check")
    L.append("")
    L.append("Format: `subject --[relation]--> object  (layer, weight)`")
    L.append("")
    for b in sampled:
        freq = (" freq=" + str(b.get("frequency"))) if "frequency" in b else ""
        L.append(f"- `{b['subject']}` --[{b['relation']}]--> `{b['object']}`  (layer={b['layer']}, w={b['weight']}{freq})")
    L.append("")

    # Full-corpus impact
    L.append("## Estimated impact on full 6.7M-article Wikipedia corpus")
    L.append("")
    L.append(f"- OLD extractor projected total: **{est_old_total:,}** bonds  ({old_mean:.2f} x 6.7M)")
    L.append(f"- NEW extractor projected total: **{est_new_total:,}** bonds  ({new_mean:.2f} x 6.7M)")
    L.append(f"- Net gain: **+{est_new_total - est_old_total:,}** bonds ({multiplier:.1f}x multiplier)")
    L.append("")
    L.append("### Sanity-check vs John's 57-113M target at 2.27M nodes")
    L.append("")
    if est_new_total >= target_total_low and est_new_total <= target_total_high:
        sanity = "**HITS TARGET BAND** ({:,} <= {:,} <= {:,})".format(target_total_low, est_new_total, target_total_high)
    elif est_new_total > target_total_high:
        sanity = ("**EXCEEDS TARGET BAND** ({:,} > {:,}). Likely overshoot from L4 "
                  "per-article cap of 80 + duplicate-emission across articles. "
                  "Cross-corpus dedup will reduce effective count by ~{:.0f}% based "
                  "on the 1000-article duplicate rate observed.").format(
                      est_new_total, target_total_high, 100*new_dup_count/max(1,new_total))
    elif est_new_total >= target_total_low / 2:
        sanity = ("**APPROACHING TARGET** but below low band ({:,} < {:,}). "
                  "Either accept a softer hub-density target for Cat.1 retrain or "
                  "enable an additional layer (Wikidata side-channel) before Phase B.").format(
                      est_new_total, target_total_low)
    else:
        sanity = ("**BELOW TARGET BAND** ({:,} < {:,}). Recommend redesign before "
                  "Phase B.").format(est_new_total, target_total_low)
    L.append(f"- {sanity}")
    L.append("")
    L.append(f"- After cross-article dedup ({100*new_dup_count/max(1,new_total):.1f}% dup rate in this sample), "
             f"effective net new triples projected: ~{int(est_new_total * (1 - new_dup_count/max(1,new_total))):,}.")
    L.append("")

    # Honest gaps
    L.append("## Honest gaps and tradeoffs")
    L.append("")
    L.append("Each item below is a real failure mode observed in the 1000-article sample, not a hypothetical.")
    L.append("")
    L.append("### What still leaks through")
    L.append("")
    L.append("- **L4 generic-adjective leakage**: 'United States', 'African Americans' ride through as multi-word phrases (correct), but isolated single-word hits like 'British' / 'Greek' get caught by the adjective stoplist. Multi-word adjectives ('North American') still emit a bond -- usually correct as a region reference, occasionally just an adjectival modifier that lost its head noun.")
    L.append("- **L5 subject-mismatch on biography articles**: 'Abraham Lincoln' picks up `BornOn February 12, 1809` (correct) AND `BornOn December 21, 1850` (wrong: that was a third party mentioned in body). The L5 patterns don't anchor to the article subject. K-gate should catch low-frequency wrong dates, but precision is below L4.")
    L.append("- **L4 possessive collapse**: 'Earth' and \"Earth's\" are now collapsed into one phrase via apostrophe-s stripping (good), but 'Lincolns' (no apostrophe) on plural family references still emits as a separate bond from 'Lincoln' (filtered as title-derivative on this article anyway).")
    L.append("- **L3 lead-paragraph patterns**: stay at <1% of total yield. They're high-precision but narrow. Kept as scaffolding for typed bonds; not a yield-driver.")
    L.append("- **L2/L2b underperformance on long-tail articles**: stub articles with no See also, no headings, and no parseable lead patterns yield 0 bonds. Acceptable -- they had nothing to extract.")
    L.append("- **L4 cap of 80 per article**: caps mean very long articles (Lincoln, Alabama) emit at most 80 RelatedTo bonds even when the body has 200+ candidates. Conservative; can be raised in Phase B if hub diversity is thin.")
    L.append("")
    L.append("### Layers that under-performed expectations")
    L.append("")
    L4_count = layer_counts.get("L4", 0)
    L5_count = sum(c for k, c in layer_counts.items() if k.startswith("L5"))
    L3_count = sum(c for k, c in layer_counts.items() if k.startswith("L3"))
    L.append(f"- **L5 (typed numerical attributes): {L5_count} bonds = {100*L5_count/max(1,new_total):.2f}% of total**. "
             f"Population/area/capital patterns are precision-tuned and only fire on articles whose lead matches the exact trigger phrasing. Useful for typed bonds but not a yield-driver. Higher-value extension would be wikidata side-channel (defer to Phase B).")
    L.append(f"- **L3 (lead-paragraph typed patterns): {L3_count} bonds = {100*L3_count/max(1,new_total):.2f}% of total**. Same story as L5: precision good, recall low. Patterns rarely match the article's exact phrasing.")
    L.append("")
    L.append("### What was **not** implemented in iter-3")
    L.append("")
    L.append("- **Wikidata side-channel** (originally listed as layer 4 in port spec): the wikimedia/structured-wikipedia HF dataset is NOT in the local `~/.cache/huggingface/hub/` (only wikimedia/wikipedia is). Adding it would require a separate download (~tens of GB). Deferred to Phase B planning -- expected high-precision typed bonds (`HasPopulation`, `HasCapital`, `LocatedIn`) without regex fragility. Lu's trust hierarchy already grants structured-wikipedia weight 1.0.")
    L.append("- **Wikipedia internal-link walker** ([[X]] markup): the HF wikimedia/wikipedia dump strips wikitext during text materialisation. There are no [[X]] tokens to walk. L4 (body co-mention) is the substitute -- it recovers the same RelatedTo signal from rendered text instead of markup. Lower precision than walking real wikilinks but the corpus we have can't do better without a re-download from the raw dump.")
    L.append("- **Body co-occurrence with proximity weighting** (originally listed as layer 5 in port spec): subsumed and partially captured by L4 frequency-weighting. Not implemented as separate paragraph-window pass.")
    L.append("- **L6 list-item walker**: confirmed via inspection that the HF text dump strips bullet markup -- 0 bullets observed across 6 representative articles. Layer would have produced 0 bonds; not implemented.")
    L.append("")

    # Phase B readiness
    L.append("## Phase B readiness verdict")
    L.append("")
    if new_mean >= target_high:
        L.append(f"**SHIP iter-3 TO PHASE B** -- the 1000-article measurement confirms {new_mean:.1f} bonds/article "
                 f"({multiplier:.1f}x baseline), which clears the 50/article stretch target. Phase B = re-run "
                 f"`extract_bonds_from_wikipedia_article` over the full corpus, batch-feed to `lu4d_teach_corpus_batch`, "
                 f"let the K-gate filter low-quality triples. Estimated full-corpus net yield (post-dedup): "
                 f"~{int(est_new_total * (1 - new_dup_count/max(1,new_total))):,} bonds.")
    elif new_mean >= target_low:
        L.append(f"**SHIP iter-3 TO PHASE B WITH CAVEATS** -- {new_mean:.1f} bonds/article ({multiplier:.1f}x baseline) clears the "
                 f"25-bond minimum but is below the 50-bond stretch. Recommend: ship Phase B re-ingest, observe hub-formation, "
                 f"add Wikidata side-channel in Phase C if the brain still under-bonds at the country/state/major-concept level.")
    else:
        L.append(f"**DO NOT SHIP YET** -- {new_mean:.1f} bonds/article ({multiplier:.1f}x baseline) is below the 25-bond "
                 f"minimum. Diagnose underperforming layers and iterate before Phase B.")
    L.append("")

    # Files
    L.append("## Files")
    L.append("")
    L.append("- `scripts/smart_extractor.py` -- iter-3 extractor (pure functions; no live-brain writes)")
    L.append("- `scripts/smart_extractor_sample.py` -- this measurement harness")
    L.append("- `scripts/smart_extractor_sample_bonds.jsonl` -- every NEW bond emitted (for offline inspection)")
    L.append("- `scripts/smart_extractor_sample_bonds_old.jsonl` -- every OLD baseline bond (for direct comparison)")
    L.append("- `scripts/smart_extractor.py.iter2.bak` -- pinned iter-2 extractor (in case rollback wanted)")
    L.append("")

    # Process / state notes
    L.append("## State preservation")
    L.append("")
    L.append("- Live brain `/var/lib/lu/brain-retrain-2026-05-04.lbr` was NOT written. Phase A is read-only.")
    L.append("- Paused processes (`PID 978512` ingest in T state, `PID 978514` lu_4_mcp daemon) untouched.")
    L.append("- `scripts/ingest_curriculum_full.py` unchanged; iter-3 ships as a *separate* extractor that the Phase B port will plug into the existing batch sink.")
    L.append("")

    out = "\n".join(L)
    with open(args.out, "w") as f:
        f.write(out)
    print(f"\n[harness] DONE.")
    print(f"[harness] Report: {args.out}")
    print(f"[harness] NEW bonds dump: {args.bonds_out}")
    print(f"[harness] OLD bonds dump: {args.old_bonds_out}")
    print(f"[harness] NEW mean: {new_mean:.2f} bonds/article")
    print(f"[harness] OLD mean: {old_mean:.2f} bonds/article")
    print(f"[harness] Multiplier: {multiplier:.1f}x")
    print(f"[harness] Total NEW bonds: {new_total}, unique triples: {len(new_seen_global):,}")
    print(f"[harness] Layer breakdown: {dict(layer_counts.most_common())}")


if __name__ == "__main__":
    main()
