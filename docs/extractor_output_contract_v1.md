# Extractor Output Contract for Lu v4 (v1)

**Status:** Specification. Ground truth for an LLM-extraction pipeline (Ollama -> JSONL -> Lu curriculum-ingest harness). Sourced from `src/node.rs`, `src/bond.rs`, `src/brain.rs`, `src/mcp.rs`, `src/source.rs`, `scripts/smart_extractor.py`, `scripts/ingest_curriculum_full.py` at `main` HEAD `758174f`.
**Audience:** the extractor author and whoever wires the JSONL into the ingest harness.
**Scope:** what shape a single extracted bond MUST take on the wire, what Lu does with it, and what fails closed if the shape is wrong.
**Non-goals:** training-curriculum prioritisation (Learning #639), trust-multiplier policy (Cat.3a), MCP transport details (handled by `McpClient` in the harness).

---

## 0. Headline contract (the one-line answer)

Every emitted bond MUST be a JSON object with the following shape, suitable as one entry in the `entries` array of `lu4d_teach_corpus_batch`:

```json
{
  "subject":     "<non-empty string, case-preserved>",
  "relation":    "<one of 29 canonical BondType names, see Section 2>",
  "object":      "<non-empty string, case-preserved>",
  "corpus":      "<non-empty string, e.g. \"wikimedia/wikipedia\">",
  "kind":        "<one of 16 KnowledgeContent kind tags, see Section 1>",
  "weight":      <f32 in [0.0, 1.0]>,
  "confidence":  "<Terminal|Verified|Unverified|Inferred|Contested>",
  "run_id":      "<optional run id, e.g. \"retrain-2026-05-29\">",
  "source_url":  "<optional per-bond URL or article slug>"
}
```

Mandatory: `subject`, `relation`, `object`, `corpus`. Everything else is optional but strongly encouraged for retrain-grade ingest. One JSONL line == one bond. The harness buffers up to 1000 entries per `lu4d_teach_corpus_batch` call (cap: `TEACH_CORPUS_BATCH_MAX_ENTRIES`, `src/mcp.rs:1862`).

The wire JSONL is INPUT to the harness, NOT to the MCP server directly. The harness maps each line into a `lu4d_teach_corpus` entry, batches 100 of them, and submits via `lu4d_teach_corpus_batch`.

---

## 1. `KnowledgeContent` enum -- 16 variants (`src/node.rs:49-83`)

The REAL enum has 16 variants; the conceptual 6-name NodeType taxonomy (Fact/Concept/Procedure/Emotion/Meta/Media) is marketing shape, NOT the on-disk schema. Source of truth is `KnowledgeContent` and its lowercase `kind_str()` tags. Variants:

| #  | Variant       | kind tag      | What it means                                                       | Extractor signal                                  |
|----|---------------|---------------|---------------------------------------------------------------------|---------------------------------------------------|
|  1 | `Fact`        | `fact`        | Factual assertion about the subject.                                | Numeric/spatial/event/notability attributes.      |
|  2 | `Concept`     | `concept`     | Concept or generic association (the default fallback).              | Generic RelatedTo bonds (L1/L4 in smart_extractor). |
|  3 | `Procedure`   | `procedure`   | Procedural instruction or rule.                                     | "To do X, do Y"; how-to text.                     |
|  4 | `Episode`     | `episode`     | Episodic memory or experience.                                      | First-person diary-shape content; rare in Wikipedia. |
|  5 | `Preference`  | `preference`  | Preference / value judgement.                                       | "X prefers Y"; opinionated text.                  |
|  6 | `Raw`         | `raw`         | Untyped content (back-compat / dumping ground).                     | Avoid; emit `concept` instead.                    |
|  7 | `Code`        | `code`        | Programming knowledge or code snippet.                              | Fenced code blocks or recognised lang signatures. |
|  8 | `Image`       | `image`       | Visual / imagery description.                                       | Visual Genome triplets; caption-shape content.    |
|  9 | `Definition`  | `definition`  | "X IS A Y" / definitional content.                                  | IsA/DefinedAs bonds.                              |
| 10 | `Mechanism`   | `mechanism`   | "X works by Y" / how-it-works content.                              | WorksBy/OperatesVia bonds.                        |
| 11 | `Composition` | `composition` | "X is composed of Y" / part-whole content.                          | PartOf/ComposedOf/MadeOf bonds.                   |
| 12 | `Purpose`     | `purpose`     | "X is for Y" / purpose content.                                     | UsedFor/IsFor bonds.                              |
| 13 | `Contrast`    | `contrast`    | "X differs from Y" / contrast content.                              | ContrastedWith/Unlike bonds.                      |
| 14 | `Causal`      | `causal`      | Cause-and-effect content.                                           | Causes/LeadsTo/ResultsIn bonds.                   |
| 15 | `Example`     | `example`     | Concrete example.                                                   | SuchAs/ForExample/Eg bonds.                       |
| 16 | `Meta`        | `meta`        | Goal/query marker used by the Thinking Layer.                       | DO NOT emit from extractor; reserved for ReAct.   |

**Parsing tolerance** (`src/node.rs:154-194` `from_kind_str`): case-insensitive, non-alphanumerics stripped. Accepted spellings include `facts`, `concepts`, `proc`/`howto`, `pref`, `untyped`, `snippet`/`source`, `img`/`visual`, `def`/`definedas`, `mech`/`worksby`/`howworks`, `composed`/`madeof`/`consistsof`, `isfor`/`usedfor`/`forwhat`, `unlike`/`differs`/`contrastedwith`, `cause`/`leadsto`/`resultsin`, `eg`/`suchas`/`forexample`, `goal`/`query`. Unknown kind => `lu4d_teach_corpus` rejects with JSON-RPC error `-32602` (`src/mcp.rs:1721`).

**Hard rule:** once a node exists, its kind is FIXED. `teach_with_kind` will NOT retype an existing node (`src/brain.rs:3504-3514`, `:3349` "kind is fixed at first observation"). The first bond mentioning subject "Albedo" sets Albedo's KnowledgeContent variant for the brain's lifetime.

---

## 2. `BondType` enum -- 29 first-class variants (`src/bond.rs:98-135`)

Memory said 28; the source has 29 (BR.10b added `HasArea` after BR.1's 18). Verbatim list with `base_weight` multipliers (applied in `weight_at`):

| Variant              | base_weight | Meaning                                       |
|----------------------|-------------|-----------------------------------------------|
| `IsA`                | 1.5         | Definitional / category membership.           |
| `HasProperty`        | 1.2         | Subject has property object.                  |
| `Causes`             | 1.3         | Subject causes object.                        |
| `PartOf`             | 1.1         | Subject is part of object.                    |
| `UsedFor`            | 1.0         | Subject is used for object.                   |
| `RelatedTo`          | 0.8         | Generic association. CATCH-ALL.               |
| `ContrastedWith`     | 0.7         | Subject contrasts with object.                |
| `DerivedFrom`        | 1.0         | Subject derived from object.                  |
| `Requires`           | 1.1         | Subject requires object.                      |
| `Produces`           | 1.2         | Subject produces object.                      |
| `Opposes`            | 0.6         | Subject opposes object.                       |
| `BornIn`             | 1.5         | (person) BornIn (place).                      |
| `BornOn`             | 1.5         | (person) BornOn (date).                       |
| `DiedIn`             | 1.5         | (person) DiedIn (place).                      |
| `DiedOn`             | 1.5         | (person) DiedOn (date).                       |
| `DiscoveredBy`       | 1.5         | (thing) DiscoveredBy (person).                |
| `FoundedBy`          | 1.5         | (org/place) FoundedBy (person).               |
| `FoundedIn`          | 1.1         | (org/place) FoundedIn (year/place).           |
| `HasAltitude`        | 1.2         | (place) HasAltitude (value).                  |
| `HasAspect`          | 1.2         | Subject has structural aspect/section.        |
| `HasCapital`         | 1.2         | (country/region) HasCapital (city).           |
| `HasOfficialLanguage`| 1.2         | (country) HasOfficialLanguage (language).     |
| `HasPopulation`      | 1.2         | (place) HasPopulation (count).                |
| `KnownFor`           | 1.2         | (entity) KnownFor (achievement).              |
| `LocatedIn`          | 1.1         | (entity) LocatedIn (place).                   |
| `MeasuredIn`         | 1.2         | (quantity) MeasuredIn (unit).                 |
| `NamedAfter`         | 1.5         | (entity) NamedAfter (eponym).                 |
| `WrittenBy`          | 1.5         | (work) WrittenBy (author).                    |
| `HasArea`            | 1.2         | (place) HasArea (value).                      |

**Catch-all:** `RelatedTo`. Memory's ~35% share alarm holds: if the extractor lets `RelatedTo` dominate, the histogram regresses and downstream typed-recall loses leverage. Target: keep `RelatedTo` under one third of emitted bonds.

**Parsing tolerance** (`src/brain.rs:411-462` `parse_bond_type`): case-insensitive, non-alphanumerics stripped. Aliases accepted: `is`->IsA, `has`->HasProperty, `cause`->Causes, `part`->PartOf, `uses`/`use`->UsedFor, `related`/`rel`->RelatedTo, `contrast`/`vs`->ContrastedWith, `derived`/`from`->DerivedFrom, `require`->Requires, `produce`->Produces, `oppose`->Opposes. Q.1 typed-alias relations map to the closest canonical BondType but preserve the original string for `infer_kind`:

  * `DefinedAs`/`means`/`definition` -> `IsA` (kind hint `definition`)
  * `WorksBy`/`OperatesVia`/`FunctionsThrough`/`mechanism` -> `UsedFor` (kind `mechanism`)
  * `ComposedOf`/`MadeOf`/`ConsistsOf`/`composition` -> `PartOf` (kind `composition`)
  * `IsFor`/`PurposeOf`/`purpose` -> `UsedFor` (kind `purpose`)
  * `Unlike`/`DiffersFrom`/`InContrastTo` -> `ContrastedWith` (kind `contrast`)
  * `LeadsTo`/`ResultsIn` -> `Causes` (kind `causal`)
  * `ExampleOf`/`SuchAs`/`ForExample`/`Eg`/`example` -> `DerivedFrom` (kind `example`)

Unknown relation => JSON-RPC `-32602` from `lu4d_teach_corpus` (per-entry error in batch).

**Confidence enum** (`src/node.rs:18-28`): `Terminal` (1.0, axiomatic), `Verified` (0.9, multi-source), `Unverified` (0.6, single-source), `Inferred` (0.5, derived), `Contested` (0.2). Parser at `src/brain.rs:465` accepts case-insensitive `terminal`/`axiom`/`axiomatic`, `verified`/`v`, `unverified`/`u`, `inferred`/`i`, `contested`/`c`. Unknown => `-32602`.

---

## 3. `teach_with_kind()` signature (`src/brain.rs:3504-3641`)

```rust
pub fn teach_with_kind(
    &mut self,
    subject: &str,
    relation: BondType,
    object: &str,
    weight: f32,
    confidence: Confidence,
    kind_hint: Option<&str>,
    relation_alias: Option<&str>,
) -> TeachResult
```

* **Mandatory:** `subject`, `relation`, `object`. Empty strings are accepted by the function but produce useless nodes; the MCP layer rejects empty `corpus` (`src/mcp.rs:1718`) but does NOT reject empty subject/object -- the extractor must validate.
* **Range-clamped:** `weight` is clamped to `[0.0, 1.0]` at three call-sites (`src/brain.rs:3568, 3603, 3614`). Out-of-range values silently saturate; no error.
* **Confidence:** discrete enum, not a float. Map to numeric only at the T.3 gate.
* **kind_hint:** when `Some("fact")`/etc, new subject/object nodes are created with that variant. Existing nodes are NEVER retyped.
* **relation_alias:** the original relation STRING (e.g. `"DefinedAs"`) used by `infer_kind` (`src/node.rs:276`) when `kind_hint` is None. Drives kind inference from the typed-alias relation aliases.
* **Defaulted:** if `kind_hint` is None AND `relation_alias` yields no inference AND text shape yields none, the new node is created as `KnowledgeContent::Concept(payload)`.

Bond dedup is by `(from_seed, to_seed)` PAIR -- NOT including `bond_type` (`src/brain.rs:3565`). A second teach with the same subject and object adds a keyframe to the existing bond and returns `reinforced_existing_bond: true`, EVEN IF the second teach has a different `BondType`. The bond_type on the existing bond does NOT change. **Extractor implication:** emit each (subject, object) pair with its richest typed relation; a later weaker RelatedTo emit is a wasted call and will reinforce the existing bond at a different type than intended.

Seed identity (`src/seed.rs:45`): `Seed::from_label` is a hash-spread of the raw UTF-8 bytes. CASE-SENSITIVE. "Albedo" and "albedo" form DIFFERENT nodes. The extractor MUST canonicalise casing before emit (recommend: preserve the casing as it appears in the article body; do NOT lowercase). Whitespace is NOT trimmed by `from_label`; the extractor must trim.

---

## 4. `lu4d_teach` and `lu4d_teach_corpus_batch` MCP shapes (`src/mcp.rs:228-292`)

### `lu4d_teach`
```json
{
  "subject":    "string (required)",
  "relation":   "string (required, BondType name or alias)",
  "object":     "string (required)",
  "weight":     "number (optional, default 0.8)",
  "confidence": "string (optional, default \"Verified\")",
  "kind":       "string (optional, KnowledgeContent variant tag)",
  "negated":    "bool (optional, default false) -- OP-EXTRACT-PREP"
}
```

**`negated` semantics (OP-EXTRACT-PREP, landed 2026-05-29):** when `true`, the call routes through `Brain::teach_with_kind` for node-and-bond formation (so the kind hint still participates in node creation) and then stamps `Brain::set_bond_negated` on the NegationOverlay sidecar. Default `false` preserves pre-OP behaviour bit-for-bit. The structured response always carries `negated: <bool>` for client-side audit.
Stamps the bond with `SourceAttribution::Teach { actor: "system", ts_ms }` UNLESS `ingest_config.corpus_default` is set, in which case it stamps `TrainingCorpus { corpus, run_id, url: None }` (`src/brain.rs:3216-3232`). For retrain-grade ingest, DO NOT use `lu4d_teach` -- prefer the explicit `lu4d_teach_corpus` path so attribution is unambiguous per call.

### `lu4d_teach_corpus`
Strict superset of `lu4d_teach`:
```json
{
  "subject":     "string (required)",
  "relation":    "string (required)",
  "object":      "string (required)",
  "corpus":      "string (required, NON-EMPTY; e.g. \"wikimedia/wikipedia\")",
  "weight":      "number (optional, default 0.8)",
  "confidence":  "string (optional, default \"Verified\")",
  "kind":        "string (optional)",
  "run_id":      "string (optional; defaults per ingest_run_id_strategy)",
  "source_url":  "string (optional, per-bond URL or article slug)",
  "negated":     "bool (optional, default false) -- OP-EXTRACT-PREP"
}
```

**`negated` semantics (OP-EXTRACT-PREP, landed 2026-05-29):** on the direct-apply branch, the flag drives `Brain::set_bond_negated` after `Brain::teach_corpus_with_kind` so the kind hint and TrainingCorpus attribution stamp are preserved. On the T.3 gated branch, the flag threads into `PendingChangeKind::AddBond { negated }` (was hardcoded `false` pre-OP), so approve/reject parity is restored. Default `false`. The structured response carries `negated: <bool>` on both branches.
Empty `corpus` rejected with `-32602`. Stamps `SourceAttribution::TrainingCorpus { corpus, run_id, url }`. When `ingest_auto_approve_threshold` is Some(t) and numeric confidence < t, the bond is queued on T.3 PendingLog with `gated: true` and `pending_id` instead of applying directly. Production binary today (per `ingest_curriculum_full.py` deviation note): `ingest_auto_approve_threshold` is NOT in `lu4d_configure`'s schema, so the gate is effectively off and every call lands directly.

### `lu4d_teach_corpus_batch` (preferred for high-throughput ingest)
```json
{
  "entries": [ <lu4d_teach_corpus entry>, ... ]   // max 1000 per call
}
```
Response `structuredContent`:
```json
{
  "bonds_landed":     <int>,
  "bonds_deduped":    <int>,
  "bonds_gated":      <int>,
  "entries_attempted":<int>,
  "errors":           [{"index": <i>, "code": <int>, "error": <string>}, ...]
}
```
Per-entry errors do NOT fail the whole batch. Anti-bypass invariant (`src/mcp.rs:1863-1956`): the batch wrapper calls `tool_teach_corpus` per entry, so trust resolution, T.3 gating, source stamping, hub-cap admission, and dedup ALL fire identically to N sequential calls. The batch is purely an IPC speedup. The grep-guard test `tingest_batch_grep_guard_uses_only_tool_teach_corpus` pins this invariant.

**Use for the new pipeline:** `lu4d_teach_corpus_batch` with 100 entries per call (the harness's `BATCH_SIZE` default, 10x headroom over the 1000 cap).

---

## 5. Embeddings -- DO NOT include in the extractor JSONL

`EMBEDDING_DIM = 384` (`src/node.rs:10`). Memory's 256-dim mention was stale. Critically, `teach_with_kind` constructs new nodes with `embedding: vec![0.0; EMBEDDING_DIM]` (`src/node.rs:377, 400`) -- i.e. ZERO at teach time. There is NO call to a text-embedding model in the teach path. Embedding population happens out-of-band (post-ingest job that walks all nodes and fills the field). The extractor MUST NOT include an embedding field in the JSONL; if it did, the MCP layer would silently ignore it (no schema entry for `embedding` in any teach tool).

**Verdict:** Lu-generated (post-ingest, asynchronously). JSONL omits embeddings entirely.

---

## 6. `SourceAttribution` shape (`src/source.rs:49-130`)

For extractor purposes, only the `TrainingCorpus` variant matters:
```rust
SourceKind::TrainingCorpus {
    corpus: String,             // required, e.g. "wikimedia/wikipedia"
    run_id: Option<String>,     // resolved per strategy if caller omits
    url:    Option<String>,     // per-bond URL or article slug
}
```
Wrapped in `SourceAttribution { kind, trust: Option<f32> }`. `trust` defaults to None and gets resolved from `trust_table.corpus_multipliers` at bond formation; the extractor does NOT supply trust.

**Required from extractor:** `corpus` (string). **Strongly recommended:** `source_url` -- carries the article slug or row id so a future GDPR delete-by-source pass can revoke every bond that came from one article.

**Dedup mechanism** -- two layers:
  1. **Python-side** (`BatchSink.seen`, `ingest_curriculum_full.py:184`): exact `(subject, relation, object)` 3-tuple match within one process. Skipped entries increment `bonds_skipped_dedup` and are NOT sent over MCP.
  2. **Brain-side** (`src/brain.rs:3565`): `(from_seed, to_seed)` pair match -- bond_type NOT included. A reinforcement adds a keyframe and returns `reinforced_existing_bond: true`. Recorded by the harness as `bonds_existing` (mapped from response `bonds_deduped`).

Key takeaway: the harness uses (subj, rel, obj) lower-case-insensitive in Python, but the brain uses seeded byte-identical (subj, obj). The two normalisation layers do NOT match. **The extractor should canonicalise subject/object spelling and casing BEFORE emit** to keep both layers aligned.

---

## 7. Asymmetry rule -- where bonds vs LB entries go

Learning #635 + #642: prompts (interrogative, hypothetical) are routed to Left Brain as `SessionEntryKind::UserStatement` via `lu4d_user_statement`. Responses / facts (declarative) form Right Brain bonds via `lu4d_teach_corpus`. The `lu4d_user_statement` tool has a hard invariant that it NEVER forms a Right Brain bond (`src/mcp.rs:294, 1974-1985`).

**For Wikipedia article extraction:** the article BODY is declarative-only. Every emitted bond should be a Right Brain bond -- there is no prompt content in a Wikipedia article. Use `lu4d_teach_corpus_batch` for everything. Do NOT route any fraction of Wikipedia content to `lu4d_user_statement`.

**For instruction-tuning / Q&A corpora (e.g. `no_robots`, `instruction-dataset`):** the prompt half goes to `lu4d_user_statement`; only the response half goes through `lu4d_teach_corpus`. This is the asymmetry guardrail. Out of scope for v1 of this contract (Wikipedia-first), but the extractor framework should be aware that future Tier-2 corpora will need the split.

**Current ingest path verification:** `ingest_curriculum_full.py`'s Wikipedia path uses `BatchSink.add` exclusively, which goes through `lu4d_teach_corpus_batch`. No Wikipedia bonds are routed to LB. The asymmetry discipline holds on this corpus.

---

## 8. `smart_extractor.py` baseline -- what we need to beat

`scripts/smart_extractor.py` (781 LOC, iter-4 subject-anchored) is the Python regex extractor in production. Output shape per bond (`extract_*` functions):

```python
{
    "subject":    title,                          # always the article title
    "relation":   "<one of 23 layer-emitted relations>",
    "object":     extracted_term,
    "kind":       _kind_for(relation),            # mirrors infer_kind_from_relation
    "weight":     <layer-specific f32>,           # L1=0.55, L2=0.5, L3 varies, L5=0.6
    "confidence": "Unverified" | "Inferred",
    "layer":      "L1" | "L2" | "L2b" | "L3-*" | "L4" | "L5-*",
    "source_url": article_url,
}
```
The harness appends `corpus` and `run_id` from its config and strips `layer` before submit. The wire shape into `lu4d_teach_corpus_batch` matches Section 4.

**What smart_extractor does well:**
  * Subject anchoring (iter-4 fix) -- L5 numerical patterns require the sentence to mention the article title, killing the BornOn-third-party bug.
  * Layered emission across six structural layers (see-also, headings, categories, lead-paragraph patterns, body co-mention, numerical attrs).
  * Per-relation kind tagging via `_RELATION_TO_KIND` (mirrors Rust's `infer_kind_from_relation`).
  * Per-article dedup on `(subject.lower(), relation, object.lower())`.

**What smart_extractor MISSES (LLM extractor target):**
  * Subject is ALWAYS the article title. Inter-entity bonds within the article body (e.g. "Lambert wrote Photometria" -> `Photometria WrittenBy Lambert`) are mostly LOST.
  * L4 emits everything as `RelatedTo` -- the highest-volume layer is fully untyped. This is the biggest histogram-skew lever.
  * No multi-hop synthesis: if the lead says "X was a German physicist who discovered Y", smart_extractor gets `X IsA physicist` (L3) but rarely `X DiscoveredBy Y` AND `X HasProperty German` AND `X BornIn Germany`.
  * No coreference resolution: pronouns and definite NPs do not become subjects of bonds.
  * Numerical-attribute extraction is limited to 9 hand-coded regex patterns (`_L5_PATTERNS`).
  * Negation is NOT detected (smart_extractor emits no negated bonds).
  * No structural facts about how entities relate beyond the article's subject (the "sub-event" depth).

**Verb-pattern -> BondType mapping** (BR.9 inlined `_relation_for_pred` in `ingest_curriculum_full.py` because `smart_extractor`'s helpers are title-anchored; see Learning glossary). The Python table is the wire-format source of truth, the Rust `infer_kind_from_relation` is the storage-side source of truth, and they MUST stay in lock-step. The LLM extractor MUST emit one of the 29 canonical BondType names OR one of the documented aliases.

---

## 9. Recommended JSONL line shape (synthesised)

One line == one bond. UTF-8, LF-terminated. No trailing comma. No nested arrays.

```json
{"subject":"Albedo","relation":"IsA","object":"reflectivity measure","kind":"definition","weight":0.85,"confidence":"Verified","corpus":"wikimedia/wikipedia","run_id":"retrain-2026-05-29","source_url":"https://en.wikipedia.org/wiki/Albedo","negated":false}
```

Mandatory fields (must be present and non-null):
  * `subject` -- non-empty string, no leading/trailing whitespace.
  * `relation` -- one of the 29 canonical BondType names from Section 2.
  * `object` -- non-empty string, no leading/trailing whitespace.
  * `corpus` -- non-empty string. For Wikipedia retrain: `"wikimedia/wikipedia"`.

Optional but strongly recommended:
  * `kind` -- one of the 16 kind tags from Section 1. If absent, Lu infers from `relation` -> may fall back to `concept`.
  * `weight` -- f32 in `[0.0, 1.0]`. Default 0.8. Use 0.85-0.95 for high-confidence typed extractions, 0.55-0.7 for noisy or generic associations.
  * `confidence` -- one of `Terminal|Verified|Unverified|Inferred|Contested`. Default `Verified`. LLM-extracted bonds from a single article should typically be `Verified` (one source); `Inferred` is appropriate for chained reasoning.
  * `run_id` -- string identifying this extraction run, e.g. `"retrain-2026-05-29"`. Lets later audits group all bonds from the run.
  * `source_url` -- article URL or stable slug. Required for GDPR delete-by-source.
  * `negated` -- bool. **Supported on all three MCP teach surfaces as of OP-EXTRACT-PREP (2026-05-29)**: `lu4d_teach`, `lu4d_teach_corpus`, and `lu4d_teach_corpus_batch` accept an optional `negated: bool` field; default `false`. The extractor SHOULD emit this flag whenever a sentence explicitly negates a relation ("X is NOT a Y", "X never visited Y"). The flag flows through to `NegationOverlay` via `Brain::set_bond_negated` on the direct-apply branch and via `PendingChangeKind::AddBond { negated }` on the T.3 gated branch.

Fields the extractor MUST NOT include:
  * `embedding` -- ignored, no schema entry.
  * `layer` -- smart_extractor diagnostic; not in MCP schema, harmless but noise.
  * Free-form metadata -- the MCP schema is closed; unknown keys are silently dropped by serde-untagged.

---

## 10. Failure modes

| Failure                                       | What Lu does                                                                                  | Extractor mitigation                                                |
|-----------------------------------------------|-----------------------------------------------------------------------------------------------|---------------------------------------------------------------------|
| Unknown `relation` name                       | `-32602` per-entry error in batch; bond is NOT formed.                                        | Validate against the 29-name list in Section 2 BEFORE emit.        |
| Unknown `kind` value                          | `-32602` per-entry error in batch; bond is NOT formed (`src/mcp.rs:1720`).                    | Validate against the 16-tag list in Section 1.                      |
| `weight` out of `[0.0, 1.0]`                  | Silently clamped to `[0.0, 1.0]` (`src/brain.rs:3568,3603,3614`).                            | Clamp at the extractor; never emit > 1.0 or < 0.0.                  |
| Unknown `confidence` label                    | `-32602` per-entry error.                                                                     | Stick to the 5 enum names.                                          |
| Empty `subject` or `object`                   | NOT rejected by MCP; produces a node with empty label that hashes to a fixed all-zero seed; collides with every other empty-label emit. POISON. | Reject empty strings at the extractor. Trim before emit.            |
| Empty `corpus`                                | `-32602` per-entry error (`src/mcp.rs:1718`).                                                | Always populate from a constant.                                    |
| Missing `corpus` field                        | `-32602` per-entry error ("`corpus` must be a non-empty string").                            | Always populate.                                                    |
| `negated` flag set                            | OP-EXTRACT-PREP: now honoured. Direct-apply -> Brain::set_bond_negated; T.3 gated -> PendingChangeKind::AddBond { negated }. | Emit when the article explicitly negates the relation; default false.|
| Duplicate `(subject, relation, object)` in same batch | Python-side dedup skips it; `bonds_skipped_dedup += 1`.                              | Stream-dedupe within the extractor too if the LLM tends to repeat.  |
| Different `bond_type` for an existing pair    | Second emit reinforces existing bond at OLD bond_type; new type silently ignored. Quiet bug. | Emit each (subj, obj) pair at its RICHEST relation; never downgrade.|
| Case-variant subject/object                   | Forms two distinct nodes ("Albedo" vs "albedo"). Quiet duplication.                          | Canonicalise casing at the extractor; preserve article-body casing. |
| Whitespace-padded subject/object              | Forms a distinct node from the trimmed form. Quiet duplication.                              | Trim at the extractor.                                              |
| Batch > 1000 entries                          | Whole batch rejected with `-32602`.                                                          | Cap the harness batch at 100 (10x headroom).                        |

All per-entry errors land in the response `errors` array with `{index, code, error}` so the harness can surface them without losing successful entries in the batch.

---

## 11. Recommended LLM extraction prompt skeleton

```
You are a knowledge-graph extractor for the Lu v4 brain.

Your job: read one Wikipedia article and emit a JSON ARRAY of typed-relation
bonds. Each bond captures one factual relationship from the article body.

OUTPUT FORMAT: strict JSON array. Each element is an object with these fields:
  - subject     (required, string, non-empty, no leading/trailing whitespace)
  - relation    (required, string, one of the 29 canonical BondType names below)
  - object      (required, string, non-empty, no leading/trailing whitespace)
  - kind        (required, string, one of the 16 KnowledgeContent kind tags below)
  - weight      (required, number in [0.0, 1.0])
  - confidence  (required, string, one of: Terminal, Verified, Unverified, Inferred, Contested)
  - negated     (required, bool; true ONLY if the article explicitly negates the relationship)

No other fields. No comments. No trailing whitespace. No prose around the array.

VALID BondType names (29, case-sensitive):
  IsA, HasProperty, Causes, PartOf, UsedFor, RelatedTo, ContrastedWith,
  DerivedFrom, Requires, Produces, Opposes,
  BornIn, BornOn, DiedIn, DiedOn, DiscoveredBy, FoundedBy, FoundedIn,
  HasAltitude, HasAspect, HasCapital, HasOfficialLanguage, HasPopulation,
  KnownFor, LocatedIn, MeasuredIn, NamedAfter, WrittenBy, HasArea

VALID kind tags (16, lowercase):
  fact, concept, procedure, episode, preference, raw, code, image,
  definition, mechanism, composition, purpose, contrast, causal, example, meta

RELATION SELECTION RULES:
  - Prefer the most SPECIFIC relation. Use RelatedTo ONLY when no specific
    relation fits; aim to keep RelatedTo under 1/3 of emitted bonds.
  - X IsA Y     => kind=definition
  - X HasProperty Y / X KnownFor Y / spatial-event-numeric attrs => kind=fact
  - X PartOf Y / X ComposedOf Y => kind=composition
  - X UsedFor Y => kind=purpose
  - X ContrastedWith Y => kind=contrast
  - X Causes Y => kind=causal
  - X DerivedFrom Y (as in "such as") => kind=example
  - RelatedTo => kind=concept

DEPTH GUIDANCE:
  - Emit as many bonds as the article text directly supports. Aim for the
    MAXIMUM number of typed relationships you can ground in the body. A
    typical 5KB article should yield 30-100 bonds.
  - Extract implicit relationships when the article makes them explicit
    in a single sentence (e.g. "Lambert, a German physicist, published
    Photometria in 1760" yields: Lambert IsA physicist; Lambert HasProperty
    German; Photometria WrittenBy Lambert; Photometria FoundedIn 1760 [or
    use a more apt date relation if available]).
  - The article subject is NOT always the bond subject. Inter-entity bonds
    within the body are first-class.
  - Properties, attributes, sub-events, locations, dates, eponymous origins
    -- all become bonds.

GROUNDING RULES (NO HALLUCINATION):
  - Every emitted bond MUST be directly supported by a sentence in the
    article. Do not infer beyond what the article states.
  - Coreference: when the article says "He was born in..." and the
    paragraph subject is "Lambert", emit "Lambert BornIn ..." -- the
    coreference resolution is part of your job.
  - When the article is uncertain ("is believed to have been born around
    1728"), set confidence=Unverified. When the article explicitly states
    something as canon, set confidence=Verified.
  - Set negated=true ONLY when the article explicitly negates the relation
    ("X is NOT a Y", "X never visited Y"). Default is false.

CANONICAL FORM RULES:
  - Subject and object strings preserve the article's casing.
  - Strip leading/trailing whitespace, citation markers like [1], and
    parenthetical pronunciation guides.
  - Use the entity's most common name in the article, not pronouns.

ARTICLE TITLE: {title}
ARTICLE URL:   {url}
ARTICLE BODY:
{body}

Return ONLY the JSON array. Nothing else.
```

Notes for the harness wrapping this prompt:
  * After the LLM returns the array, the harness fills `corpus`, `run_id`,
    `source_url` per-bond and submits as `entries` to
    `lu4d_teach_corpus_batch` in chunks of 100.
  * The harness validates each bond against the 29-relation and 16-kind
    lists BEFORE submission. Failed validation increments a local counter
    and the bond is dropped (NOT submitted as a malformed entry).
  * The harness also runs the Python `BatchSink.seen` dedup over
    `(subject, relation, object)` to skip in-process repeats.

---

## 12. Open questions for John

1. **`negated` flag on `lu4d_teach_corpus`.** RESOLVED 2026-05-29 via OP-EXTRACT-PREP
   Phase A.1 (feature/op-extract-prep off 758174f, halt-for-review). All three
   teach surfaces (`lu4d_teach`, `lu4d_teach_corpus`, `lu4d_teach_corpus_batch`)
   now accept optional `negated: bool` (default false). Direct-apply routes
   through `Brain::teach_corpus_with_kind` + `Brain::set_bond_negated`; T.3 gated
   path threads the flag into `PendingChangeKind::AddBond { negated }`. The LLM
   extractor SHOULD emit this flag.

2. **Bond dedup at a different bond_type.** STATUS 2026-05-29 via OP-EXTRACT-PREP
   Phase A.2 (HALTED on audit): widening the dedup key from `(Seed, Seed)` to
   `(Seed, Seed, BondType)` is NOT a 50-80 LOC change. It cascades through the
   on-disk archive layout (`bond_index: HashMap<(Seed, Seed), u32>` at
   `src/archive.rs:524`), the heap bond store (`bonds: HashMap<(Seed, Seed),
   TemporalBond>` at `src/network.rs:319`), `bond_tombstones: HashSet<(Seed,
   Seed)>`, `BondView` / `BondMutHandle` API surface (97 call sites for
   `get_bond(from, to)`), and 27+ uses of `iter_bond_keys() -> (Seed, Seed)` in
   spread-activation / recall / hub-degree / source-overlay parsing. The
   architectural premise that one bond exists per (subj, obj) pair is
   documented in `brain.rs:3163-3168`. A widening port requires (a) archive
   format version bump, (b) re-encode of the canonical 8.5M-node / 26M-bond
   brain, (c) rewrite of `get_bond` to either return `Vec<BondView>` or take a
   bond_type arg. **Mitigation for now:** Section 10 caveat stands -- the
   extractor MUST emit each (subj, obj) pair at its RICHEST relation, never
   downgrade. The harness `BatchSink.seen` already dedups on (subject, relation,
   object) within a process; the LLM extraction prompt also instructs richest-
   relation-only emission. Follow-up port name: `OP-EXTRACT-WIDEN`.

3. **Case canonicalisation policy.** Seed identity is byte-identical on the
   subject/object label. The LLM may emit "Albedo" in one sentence and
   "albedo" in another, forming two nodes. Recommend the harness lowercase
   subject/object for all non-proper-noun nodes? Or trust the LLM to pick
   one canonical form per entity per article? (My read: trust the LLM but
   validate -- have the harness log a warning when it sees both "Albedo"
   and "albedo" within one article's bond batch.)

4. **`run_id` strategy.** `ingest_run_id_strategy` defaults to `ServerStartup`;
   `ingest_curriculum_full.py` uses a `RUN_ID_TIER1` constant. Should the
   LLM-extraction pipeline use a per-run-id strategy (one id per
   extraction-batch invocation) or a per-corpus strategy (one id per corpus,
   matching the existing harness)? Per-run-id gives finer-grained revoke;
   per-corpus matches the existing pattern. Recommend: per-pipeline-invocation
   (e.g. `"llm-extract-2026-05-29-pass-1"`), but confirm.

5. **`lu4d_teach` vs `lu4d_teach_corpus_batch` for ingest.** Per `src/mcp.rs:1862`
   the batch cap is 1000 and the harness's default `BATCH_SIZE` is 100
   (10x headroom). At ingest throughput, `lu4d_teach_corpus_batch` is
   strictly better than per-bond `lu4d_teach_corpus` (amortises IPC,
   identical per-bond semantics per the anti-bypass invariant). For the
   new pipeline we should default to `lu4d_teach_corpus_batch` at
   batch-size 100. Confirm there is no operational reason to drop to
   per-bond calls (e.g. T.3 gating behaviour at non-default thresholds).

---

## 12a. Changes from v1 spec (OP-EXTRACT-PREP, 2026-05-29)

  * **Section 4 schemas.** `lu4d_teach`, `lu4d_teach_corpus`, and
    `lu4d_teach_corpus_batch` entry schemas now include an optional
    `negated: bool` field (default `false`). The MCP layer parses the flag,
    threads it through to the NegationOverlay on the direct-apply branch,
    and into `PendingChangeKind::AddBond { negated }` on the T.3 gated branch.
    Structured responses always carry `negated: <bool>` for audit.

  * **Section 6 dedup semantics.** No change in v1. The brain-side dedup
    key remains `(from_seed, to_seed)` -- pair-only. OP-EXTRACT-PREP Phase
    A.2 attempted to widen this to `(from_seed, to_seed, bond_type)` but
    halted on architectural audit: the on-disk archive layout and 97+
    call sites depend on the pair-only invariant. Widening is now a
    separate follow-up port (`OP-EXTRACT-WIDEN`) requiring archive format
    bump and canonical brain re-encode. Until that ships, Section 10's
    caveat stands: extractor MUST emit each (subj, obj) pair at its
    richest relation.

  * **Section 9 JSONL recommendation.** `negated` is now a supported
    field (was forward-compat-only in v1).

  * **Section 10 failure modes.** Updated `negated` row to reflect the
    landed flag.

  * **Section 12 open questions.** Q1 RESOLVED. Q2 status updated with
    Phase A.2 halt rationale and the deferred-port name.

## 13. Stay-in-sync checklist

If any of the following change, update this document AND `_RELATION_TO_KIND`
in `scripts/smart_extractor.py` AND `infer_kind_from_relation` in
`src/node.rs` in lock-step:

  * Adding a `BondType` variant.
  * Adding a `KnowledgeContent` variant.
  * Changing a relation alias parsed by `parse_bond_type`.
  * Changing a kind alias parsed by `from_kind_str`.
  * Changing `TEACH_CORPUS_BATCH_MAX_ENTRIES`.
  * Changing the JSON schema of `lu4d_teach`, `lu4d_teach_corpus`, or
    `lu4d_teach_corpus_batch`.
  * Changing `EMBEDDING_DIM` (would require a full re-embed pass).
  * Adding a path that mutates an existing node's `KnowledgeContent`
    variant after first observation -- this would invalidate the
    "kind is fixed at first observation" invariant the extractor relies on.
