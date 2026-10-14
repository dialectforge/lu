# Attributions

Lu's foundation knowledge is built from public-domain and permissively-licensed
sources. This file records required attributions. Only public-domain (CC0),
MIT, and attribution-only (CC-BY) sources are used; share-alike (CC-BY-SA) and
non-commercial (CC-BY-NC) sources are excluded by policy.

## WordNet (Princeton) — public domain / WordNet License

Princeton University "About WordNet." WordNet. Princeton University. 2010.
George A. Miller (1995). WordNet: A Lexical Database for English.
Communications of the ACM Vol. 38, No. 11: 39-41.
Used for the IS-A / PartOf / Opposes taxonomic backbone and sense-anchored
definitions (per-synset node identity).

## GCIDE — GNU Collaborative International Dictionary of English (public domain)

Derived from Webster's Revised Unabridged Dictionary (1913), public domain.
Used for supplementary definitions.

## Quasimodo — CC-BY-2.0

Julien Romero, Simon Razniewski, Koninika Pal, Jeff Z. Pan, Archit Sakhadeo,
Gerhard Weikum (2019). "Commonsense Properties from Query Logs and Question
Answering Forums." CIKM 2019. Max Planck Institute for Informatics.
Dataset: https://huggingface.co/datasets/Aunsiels/Quasimodo — CC-BY-2.0.
Used for HasProperty edges (attribute/part/capability), ranked by saliency.
Edges tagged `corpus="quasimodo"` in-brain for provenance.

## Ascent++ (AscentKB) — CC-BY-4.0

Tuan-Phong Nguyen, Simon Razniewski, Julien Romero, Gerhard Weikum (2021).
"Advanced Semantics for Commonsense Knowledge Extraction." WWW 2021; and
"Refined Commonsense Knowledge from Large-Scale Web Contents" (Ascent++).
Max Planck Institute for Informatics.
Dataset: https://huggingface.co/datasets/tuanphong/ascent_kb — CC-BY-4.0.
Used for HasProperty edges (/r/HasProperty, /r/CapableOf, /r/HasA), ranked by
support. Edges tagged `corpus="ascent"` in-brain for provenance.

## GenericsKB (Allen Institute for AI) — CC-BY-4.0 (evaluated, not ingested)

Sumithra Bhakthavatsalam, Chloe Anastasiades, Peter Clark (2020). "GenericsKB:
A Knowledge Base of Generic Statements." AI2. arXiv:2005.00660. CC-BY-4.0.
Evaluated as a property source; superseded by Quasimodo+Ascent for property
cleanliness. Retained here in case it is later ingested (loader present).

---

_Maintained automatically as CC-BY / MIT / CC0 sources are added, so the sole
obligation (attribution/citation) is satisfied completely. Add each new source
with its citation here before ingesting it._

## Evaluated, NOT ingested
- e-CARE (12ml/e-CARE, MIT): phrase/sentence-level MCQ, not concept-level causal edges -- rejected (gravel).
- Quasimodo causal predicates (CC-BY-2.0): concept-level Causes/LeadsTo -- dry-run only, pending misinformation-filter decision.
- Ascent++ /r/Causes (CC-BY-4.0): 48k causal edges, corroboration source.

## Ingested (Tier 1.3 causal, post-review)
- Quasimodo causal predicates (CC-BY-2.0) + Ascent++ /r/Causes (CC-BY-4.0): corroborated concept-level Causes/LeadsTo edges, human-reviewed for factual accuracy; health-misinformation denylist + non-political standing rule applied. See scripts/clean_causal.py, causal_misinfo_denylist.tsv.

## Wikidata entity subsystem (OP-BREADTH Phase-1)

Entity-graph breadth (famous + notable entities: countries, cities, scientists,
works, etc.) is derived from **Wikidata**, dedicated to the public domain under
**CC0 1.0**. Bulk statements + English labels were extracted from the
**CleverThis/wikidata-truthy** dump on Hugging Face (license: CC0-1.0), with a
small rank-filtered top-up of a few entities missing from that snapshot via the
Wikidata `wbgetentities` API (also CC0). Ingested as `label#QID` entity nodes in
the segregated `e_<word>`/`eq<qid>` namespace; Confidence::Unverified; corpus="wikidata".
