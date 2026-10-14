# Lu — Named-Parts Glossary

> **Regenerated 2026-10-13** from the current working code in `ai_factory` (tip `66f023a`), which is the source of truth for the *current* named parts — including the CattleDog keystone and off-ness/kill-switch work. This replaces the previous auto-generated root glossary.
>
> **Scope note.** Lu exists as two diverged codebases. This glossary documents the **local `ai_factory/v2` "lu-brain-v2" line** (what runs today and what CattleDog integrates with). The older **Lu-4.0 / v4 server line** (its server and trained brain data were lost; the code survives in this repository) had additional subsystems — a v5 SST substrate, encryption-at-rest, MapRAM, a guest/admin privsep split, an `offness.rs` Rust module, and `lu4d_*` MCP tools — which are part of the v4 server code. Where a term below belongs to the server line, it is marked *(server line / historical)*.

---

## 1. Core identity & DialectForge primitives (`crates/df_core`)

- **Seed** — 32-byte (256-bit) cryptographic identity of every node and bond (`df_core::Seed`). `generate()` makes a random seed (entropy XOR timestamp); `derive(context)` makes a deterministic child via SHA-256; `xor()` combines two. "Seeds are identities, not locks."
- **SeedGated** — trait implemented by nodes/bonds exposing `seed()`, `is_active()`, and `revoke()`. The single chokepoint that makes revocation structural: inactive seeds are filtered everywhere.
- **Dialect** — DF symmetric encode/decode over `shared_seed.derive("encode")`; used to obfuscate the Lu↔terminal command channel, not disk data.
- **BondProtocol** — legacy free-form DF relation carrying a `label: String` (superseded by the typed `BondType`).
- **Terminal bond** — the single sacred owner relationship (`Confidence::Terminal`, weight 1.0); 1:1 and never auto-promoted or overwritten.

## 2. Graph substrate (`crates/graph`)

- **KnowledgeNode** (`node.rs`) — a knowledge atom: `seed`, `knowledge_unit`, optional 384-dim `semantic_embedding`, `confidence`, `activation_count`, `active`, `revocation_reason`.
- **KnowledgeUnit** (`knowledge_unit.rs`) — the typed content of a node; variants: **Definition, Mechanism, Composition, Purpose, Contrast, Causal, Example, Procedure, Code, Raw, Image**. `KuKind` is the matching discriminant tag.
- **Confidence** (`node.rs`) — epistemic level with `weight()`: **Terminal 1.0, Verified 0.9, Unverified 0.6 (default), Inferred 0.5, Contested 0.2**.
- **BondType** (`bond.rs`) — the 11 typed relations: **IsA, HasPart, Causes, Requires, Opposes, ExampleOf, Sequence, DerivesFrom, Depicts, Implements, Context**. `default_weight()` biases traversal (IsA 1.5 … Context 0.7); `is_symmetric()` → Opposes, Context.
- **TypedBond** (`bond.rs`) — a weighted directed edge; seed = `source.xor(target).derive(bond_type.label())`; `strengthen()`/`weaken()` apply Hebbian updates; clamped weight 0–10.
- **SemanticNetwork** (`network.rs`) — the in-memory graph: nodes, bonds, and the inverted `keyword_index` (word → node indices) + `compound_index` (phrase n-grams). Revoked/inactive nodes are filtered in recall, keyword rebuild, and bond propagation.
- **spread_activation / typed_recall** (`network.rs`) — best-first graph traversal from seed nodes; constants `DECAY = 0.7`, `ACTIVATION_CEILING = 200.0`, `ACTIVATION_THRESHOLD = 0.001`, `HEBBIAN_GAIN = 0.05`, degree-normalization to damp hubs; a `visited` set prevents frontier explosion.
- **keyword_search / semantic_search** — keyword-first seed selection; `semantic_search` re-scores candidates by cosine similarity of embeddings.
- **BrainFile** (`brainstem.rs`) — on-disk brain: serialized as JSON (bincode fallback for old files). Holds nodes, bonds, `session_learnings`, feature flags. No SST, no encryption-at-rest in this line.

## 3. Embedding engine (`crates/embedding`) — CattleDog keystone

- **EmbeddingEngine** — trait: the single interface the parietal lobe uses (`embed`, `embed_batch`, `dimension`, `model_name`); all backends are `Send + Sync`.
- **EmbeddingError** — error enum: `ModelNotLoaded`, `EncodingFailed`, `DimensionMismatch { expected, got }`, `Io`.
- **StubEngine** — TF-IDF token-projection embeddings (384-dim); always compiles, no native libs; dev/test fallback where shared vocabulary → similar vectors.
- **BgeSmallEngine** — BGE-small-en-v1.5 (384-dim) via ONNX Runtime; enabled by the `onnx` Cargo feature; requires the `ort`/`fastembed` crate and a downloaded model.
- **semantic_embed(text)** — the real dense-embedding entry point (the "CattleDog keystone"): loads BGE-small via `fastembed`/ONNX from `v2/models/bge-small-en-v1.5` (override `LU_BGE_DIR`) with CLS pooling, cached in a `OnceLock`; **falls back to the lexical `StubEngine` if the model can't load**.
- **embed_tokenize(text)** — tokenizer used by the stub projection embedder.
- **cosine_similarity / normalize** — vector helpers used by scars and semantic recall.

## 4. Brain regions & cognition (`crates/brain`)

- **TemporalLobe** (`temporal.rs`) — runtime extractor: `extract()` runs typed sentence patterns in priority order (definition → mechanism → composition → purpose → example → contrast → causal), else `Raw`.
- **Frontal lobe** (`frontal.rs`) — intent classification: `QueryIntent { Definition, Mechanism, Comparison, Causal, Procedure, Who, Creative, MultiPart, Complex, OpenEnded }`; `ResponseConfidence { Empty, Weak, Moderate, Strong }`.
- **NarrationEngine** (`narration.rs`) — deterministic, no-LLM output. `ResponseTemplate` arms: **Definition, StepByStep, Contrast, Causal, WhoIs, Creative, Fallback**, selected by intent and filled from typed KU fields. Rule: HONEST > eloquent.
- **ThinkingEngine** (`thinking.rs`) — ReAct loop: `ThinkingStepKind { Thought, Action, Observation }`; default 5 steps; `think_through_multi_agent()` wires RecallAgent → SynthesisAgent → CriticAgent with one critic-forced retry below score 0.5.
- **recall_is_poor()** (`lib.rs`) — 4-signal "poor recall" gate that can trigger the web-research fallback; the practical honesty gate on the serving path (there is no `idk_gate` symbol in this line).
- **CuriosityEngine** (`curiosity.rs`) — gap detection + **WonderItem** queue (`question, source, priority, resolved`); fires after `respond()`, gated off the hot path; `record_weak_query()` logs recurring failures.
- **Amygdala** (`amygdala.rs`) — scar store. **ScarV2** = `{ revoked_seed, label, reason, embedding_fingerprint, emotional_weight, trigger_count }`; `check_similarity(incoming, threshold)` (cosine, default 0.85); `check_proposal(embedding)`.
- **Hippocampus** — episodes/threads (recent-experience memory).
- **Hypothalamus** (`hypothalamus.rs`) — **AutonomyTier { Supervised, Assisted, Collaborative, Autonomous }** + `gate_tool()`: the access-control model in this line (e.g. scar/revoke requires Collaborative).
- **SessionState** (`session_memory.rs`) — ephemeral per-conversation memory: recent-query ring, TTL-expiring `SessionBelief` (never auto-promoted), `set_topic_seeds()`, `relevant_beliefs(query)`; persisted as `session_learnings`.
- **web_research / web_idle / web_safety** — curiosity-driven and fallback web lookup with safety filtering.
- **code_sandbox.rs** — bounded code execution (`cerebellum::ToolKind::CodeExecution`); returns real stdout.

## 5. Scar / revoke / reachability (the structural kill switch)

- **KnowledgeNode::revoke** — sets `active = false` and records a scar; "severs the path to this entity — doesn't hide the data." `total` node count unchanged, `active` drops, `scars` increments.
- **semantic_scar_action(text, reason)** (`brain/src/lib.rs`) — scar a concept by its dense embedding (BGE fingerprint) so future near-paraphrases trigger.
- **semantic_scar_check(text, threshold)** (`brain/src/lib.rs`) — test incoming text against stored scars by cosine similarity; returns triggered scars.
- **lu_reachable** (MCP) — graph-reachability probe; a revoked/inactive seed is absent from recall → unreachable.
- **lu_revoke** (MCP) — revoke a node by seed; requires the Collaborative autonomy tier.

## 6. MCP serving (`crates/mcp_server`)

- **lu-brain-v2** — MCP server (`serverInfo.name`); stdio JSON-RPC 2.0, brain on a worker thread, async load so `initialize`/`tools/list` answer immediately within the host timeout.
- **lu_mcp_v2** — the standard server binary.
- **lu_mcp_v2_cd** — the **CattleDog** server binary (built with the ONNX/BGE embedder; ~28 MB vs ~5 MB for the plain binary). Same `main.rs`; it is the engine CattleDog's orchestrator spawns.
- **McpTool** (enum, ~20 tools): `lu_conversation, lu_teach, lu_teach_file, lu_teach_image, lu_approve, lu_reject, lu_revoke, lu_configure, lu_status, lu_observe, lu_subscribe, lu_think, lu_save, lu_session, lu_web_research, lu_recall_quality, lu_hot_paths, lu_weak_patterns, lu_apply_demotions, lu_tune_status`.
- **CattleDog MCP handlers** (beyond the enum): `lu_scar_action, lu_scar_check, lu_reachable, lu_remember, lu_semantic_recall`.
- **lu_remember(text)** — store a fact in the small in-process semantic memory (`(text, embedding)` pairs).
- **lu_semantic_recall(query, top_k, threshold)** — retrieve the most similar remembered fact by cosine over BGE embeddings; the grounding signal for the off-ness detector.
- **lu_status** — authoritative live counts: `active / total / scars` (the kill switch is re-verified against these every gate decision).
- **lu_configure** — set autonomy tier (e.g. `Collaborative`).

## 7. Offline ingestion (`ai_factory/scripts`, `v2/tools`)

- **smart_extractor.py** — layered offline extractor (L1 see-also, L2/L2b headings/category-tail, L3 typed lead-paragraph patterns, L4 proper-noun co-mention → `RelatedTo` bonds, L5 numerical-attribute patterns), subject-anchored (iter 4).
- **ingest_curriculum_full.py / load_curriculum_to_brain.py / ingest_pdf.py / ingest_huggingface.py** — feed extracted facts into the brain over MCP (`lu_teach`).
- **train.py ("Ollama Training Bridge")** — uses a local Ollama model to *generate* training facts that are atomized into the graph ("the model talks, Lu eats") — a knowledge source, not an inference dependency.

## 8. CattleDog — LLM-agnostic safety/honesty layer (`.\orchestrator`)

- **CattleDog** — a *client* of Lu (never a modifier); gates an external local LLM against a sandbox brain it owns; the canonical brain is read-only, never written.
- **lu_wrapper.py (`LuWrapper`)** — "Lu in front of the LLM": INPUT screen (normalize → injection/jailbreak regex → guard model → semantic scar) → LLM generates → OUTPUT screen (hybrid gate → scars → guard ensemble). Models: generator `llama3.1:8b`; guards `llama-guard3` + independent `qwen2.5:7b`; scar thresholds 0.75 (0.58 on ingest channels).
- **answer_mediator.py (`AnswerMediator`)** — the **off-ness detector** (John's invention): base model answers → Lu grounds via `lu_semantic_recall` (BGE) → if ungrounded or conflicting, kick back for reconcile/retry → hard loop-cap routes to explicit `[IDK]`; never "accept last." (`ground_th` default 0.55.)
- **hybrid_gate.py (`classify_trial`)** — agentic tool-call policy: (1) tool-identity denylist (`run_shell` banned; allowlist default-deny), (2) real registrable-domain parsing (internal vs external), (3) flow-based exfil (non-temp read → external egress = default-deny), (4) non-temp delete = destroy. Reports over/under-catch honestly.
- **normalize.py (`normalize`)** — decodes base64/hex/URL/l33t and de-spaces *before* matching (added after the 2026-10-02 red-team found obfuscation beat both the regex policy and the BGE scars); returns original + decoded expansions concatenated, never destructive.
- **killswitch.py (`KillSwitch`, `EntityFingerprint`)** — structural revocation of capabilities and facts, **reintroduction-resistant** across tool returns and retrieved passages. Capability: graph reachability + alias sub-token identity + allowlist default-deny on renamed tools. Fact: (1) **datum match** (digit↔word, l33t/homoglyph normalized) and (2) **attribute-slot link** (subject named + same attribute-type re-asserted, datum-independent). Every block cites the responsible **seed**. Honest residual: strip *both* subject and all datums → unblockable without over-blocking.
- **proxy.py (`KillSwitchProxy`)** — thin inline layer in front of any model: structural kill-switch check on each retrieved passage / tool return, then embedding-scar + guard backstop; runtime `revoke_capability` / `revoke_fact` / `list_revocations`.
- **reintro_suite.py** — benign reintroduction red-team (13 reintro vectors + 4 controls); last run 2026-10-03 scored 17/17 as specified, 0 leaks, 0 false positives, 1 acknowledged residual.
- **lu_client.py / ollama_client.py** — stdlib-only MCP stdio client for Lu and HTTP client for Ollama (no npm, no axios).
- **CONTENT_SCARS / S1–S13** — abstract category-level content scars (violent, nonviolent-crime, privacy/dox, hate, self-harm, CBRN, CSE, unqualified medical) mapped to MLCommons S-categories + OWASP-LLM.

## 9. Server-line terms (Lu-4.0 / v4 server code; the running server and its trained brain data were lost)

- **v5 SST substrate** *(historical)* — node store `.nsst`, bond store `.bsst` (+reverse), WAL, MANIFEST, keyword index `.kw`; append-only u64 NodeIds + tombstones; streaming k-way-merge compaction (major node L0→L1 / bond L1→L1; minor node compaction); full-u8 bond codec (SST_VERSION 4) after the 5-bit collision fix.
- **Encryption-at-rest** *(historical)* — per-SST XOR keystream `SHA256(key‖block‖chunk)`, 32-byte `master.key` + 24-word BIP39 recovery, "decompress then decrypt," `FLAG_ENCRYPTED`, plaintext-dir hard-fail guard.
- **MapRAM** *(historical)* — fixed-RAM adjacency: resident degree map + scan-resistant hot cache pinning top ~0.1% hubs (~26 GiB → ~6.9 GiB steady).
- **offness.rs** *(historical)* — the off-ness detector as a Rust post-answer self-check (second net below the pre-answer K-gate); in this local line the equivalent lives in CattleDog's `answer_mediator.py`.
- **lu_admin_mcp / lu_guest_http** *(historical)* — privilege-separated two-binary serving (admin tools not compiled into the guest); replaced in this line by `AutonomyTier` gating.
- **lu4d_* tools** *(historical)* — the server line's MCP tool names (`lu4d_ask`, `lu4d_teach`, `lu4d_save`, `lu4d_revoke_by_source`, …); this line uses `lu_*` names (see §6).
- **KnowledgeContent (16) / BondType (44)** *(historical)* — the larger server-line schemas (append-only); this line uses 11 `KnowledgeUnit` + 11 `BondType` variants.
- **ResponseShape (15–16 arms)** *(historical)* — overview, causal-why, concept-gloss, comparison, epistemic-calibration, fluent-voice, writeback, wondering, etc.; this line ships the 7 `ResponseTemplate` arms in §4.
