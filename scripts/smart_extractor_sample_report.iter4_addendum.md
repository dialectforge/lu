# Smart Extractor v2 — iter-4 Delta (subject-anchoring fix)

**Run:** 2026-05-07 11:07 UTC | **Articles re-run:** same 1000-article seed (seed=42), same `wikimedia/wikipedia` 20231101.en stream

## What changed in iter-4

**Single targeted fix from Phase A's "Honest gaps" section:** L5 patterns (and L3 lead-paragraph patterns sharing the same failure shape) now require the matched sentence to contain the article subject (title or a derived alias).

Helpers added to `smart_extractor.py`:

- `_split_sentences(text)` — sentence-boundary split on `.!?` followed by whitespace, plus newlines
- `_build_subject_aliases(title)` — `{title} ∪ {capitalised words in title with len ≥ 4}` (e.g. `"Abraham Lincoln" → {"Abraham Lincoln", "Abraham", "Lincoln"}`)
- `_sentence_anchors_subject(sentence, aliases)` — True iff any alias is a substring of the sentence

`extract_numerical_attrs` and `extract_lead_patterns` were rewritten to iterate sentences and skip sentences that do not anchor the subject. The title-prefix IsA regex (`title + " is/was a..."`) is left unchanged because it is anchored by construction.

## Yield delta vs iter-3 (same 1000 articles, same seed)

| Metric | iter-3 | iter-4 | Δ |
|---|---|---|---|
| Total bonds | 58,843 | 58,627 | −216 (−0.37%) |
| L5 total | 25 | 15 | −10 (−40%) |
| L3 total | 389 | 183 | −53% |
| L1 (See also) | 3,081 | 3,081 | 0 |
| L2 (headings) | 15,004 | 15,004 | 0 |
| L2b (categories) | 9,419 | 9,419 | 0 |
| L4 (body co-mention) | 30,925 | 30,925 | 0 |

L1/L2/L2b/L4 are bit-identical to iter-3 — the patch is surgical to L3+L5 as intended. Mean yield 58.63 bonds/article (vs iter-3 58.84) — well above the 50-bond stretch target.

## Bond-level verification on the Phase A example

The original "Honest gaps" report flagged Abraham Lincoln picking up `BornOn February 12, 1809` (correct) AND `BornOn December 21, 1850` (wrong third party). After the fix:

**Removed (subject sentence didn't contain title alias):**
- `Aldine Press --[DiedOn]--> February 6, 1515` (Aldine Press is a printing press, not a person)
- `Ambrosians --[DiedOn]--> September 14, 1510` (sect, not a person)
- `Arthur Miller --[BornOn]--> September 7, 1944` (his daughter Rebecca, not Arthur)
- `Arthur Miller --[BornOn]--> September 15, 1962` (his son Daniel, not Arthur)

**Kept (sentence anchored on subject — date may still be semantically wrong):**
- `Abraham Lincoln --[BornOn]--> February 12, 1809` ✅ correct
- `Abraham Lincoln --[BornOn]--> December 21, 1850` ❌ William Wallace Lincoln, his son
- `Abraham Lincoln --[DiedOn]--> February 1, 1850` ❌ Edward Baker Lincoln, his son
- `Akira Kurosawa --[BornOn]--> December 20, 1945` ❌ his son Hisao

These four "subject-anchored but semantically wrong" cases are an **honest residual gap**: the subject's surname appears in the sentence (Lincoln's children carry "Lincoln"; Kurosawa's son is referred to via the surname). True name-disambiguation would require pronoun coreference or relative-noun negative patterns ("his son", "her daughter") — out of scope for the Phase B subject-anchoring fix per the brief.

## Acceptance criterion

The brief said: *"Re-run the sample driver against the same 1000 articles, verify L5 yield goes from 25 → maybe 5-10 bonds, but every one is now article-subject-anchored."*

- L5 dropped 25 → **15** (40% reduction; near but slightly above the "5-10" range)
- Every remaining L5 bond is article-subject-anchored (substring match on title or alias passes)
- Subject-mismatch where the surname is shared with a relative is the residual gap; flagged honestly

## Files

- `scripts/smart_extractor.py` — iter-4 (with subject-anchoring helpers; lines: 692)
- `scripts/smart_extractor.py.iter3.bak` — pinned iter-3 (pre-anchoring)
- `scripts/smart_extractor.py.iter2.bak` — pinned iter-2
- `scripts/smart_extractor_sample_bonds.iter3.jsonl` — iter-3 bonds (preserved for diffing)
- `scripts/smart_extractor_sample_bonds.jsonl` — iter-4 bonds (canonical, post-anchoring)
- `scripts/smart_extractor_sample_report.iter3.md` — iter-3 report (preserved)
- `scripts/smart_extractor_sample_report.md` — iter-4 report (canonical)

## State preservation

- Live brain `/var/lib/lu/brain-retrain-2026-05-04.lbr` was NOT written by this run.
- Paused processes (`PID 978512` ingest in T state, `PID 978514` lu_4_mcp daemon) untouched.
- `scripts/ingest_curriculum_full.py` unchanged at this point — surgical swap is the next step.
