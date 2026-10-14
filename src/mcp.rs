//! MCP (Model Context Protocol) server for Lu 4D.
//!
//! Speaks JSON-RPC 2.0 over stdio. Implements the subset of MCP needed
//! for tool invocation: `initialize`, `tools/list`, `tools/call`, and
//! a handful of notifications that the runtime sends but doesn't expect
//! a reply to.
//!
//! Protocol reference: https://modelcontextprotocol.io/specification

use crate::bond::BondType;
use crate::brain::{parse_bond_type, parse_confidence, Brain};
use crate::node::Confidence;
use serde_json::{json, Value};
use crate::mcp_envelope::wrap_content;
use std::io::{self, BufRead, Write};

/// The MCP protocol version this server speaks.
pub const MCP_PROTOCOL_VERSION: &str = "2024-11-05";

/// Run the MCP server loop over stdin/stdout until EOF.
///
/// Each line on stdin is expected to be a complete JSON-RPC 2.0 message.
/// Responses are written to stdout one per line. Logs and errors go to
/// stderr so they don't corrupt the JSON-RPC stream.
pub fn run_mcp_server(brain: &mut Brain) -> io::Result<()> {
    run_mcp_server_opts(brain, brain.path.is_some())
}

pub fn run_mcp_server_opts(brain: &mut Brain, save_on_every_write: bool) -> io::Result<()> {
    let stdout = io::stdout();
    let mut out = stdout.lock();
    let save_on_every_write = save_on_every_write && brain.path.is_some();

    eprintln!(
        "[lu4d-mcp] starting (nodes={}, bonds={}, brain_file={:?})",
        // OP-BRAIN-MapRAM: the cache-backed default does not build the
        // resident adjacency, so the honest counts come from the node
        // intern (nodes) and the resident MAP (bonds); fall back to the
        // resident adjacency only in forced-legacy mode.
        brain.network.node_intern().len(),
        brain
            .resident_map
            .as_ref()
            .map(|m| m.total_out_bonds() as usize)
            .unwrap_or_else(|| brain.network.bond_count()),
        brain.path,
    );

    // Issue A fix internal notes: build the archive-backed keyword
    // index ONCE at startup, before any handler runs. Pre-fix this
    // was only built lazily inside Brain::ask, so the very first call
    // to lu4d_reason / lu4d_narrate (which take &Brain immutably and
    // never trip the lazy build) returned "no anchor" until the user
    // happened to send an lu4d_ask first. Doing it eagerly here costs
    // a one-time page-fault burst against the mmap'd v3 archive, then
    // every subsequent query -- typed_ask, reason, narrate, agents --
    // sees a populated keyword_index from line one.
    // OP-022.b: lifted to the Brain-level entry so v5 brains build
    // the keyword index from the node store (v3/v4 delegate to the
    // Network archive walk, unchanged).
    // OP-INGEST-LEAN: the keyword index is a SERVING/recall structure.
    // A write-only ingest never runs a keyword query, and (re)building it
    // here would rebuild the ~12.5 GB index on the main thread before the
    // first bond can land. Skip it for ingest-only brains; it is built
    // when Lu is served.
    if !brain.is_ingest_only() {
        brain.ensure_keyword_index_built();
    }

    // Port T.6: rebind auto_tick to share the brain's ClockSource so
    // the driver and the rest of the brain agree on time. The default
    // initializer used SystemClockSource; if the brain has been built
    // with a different source (e.g., DF-Time-Service after T.1.1),
    // sharing keeps the T.1 invariant uniform.
    let new_at = crate::idle::AutoTickDriver::with_config(
        brain.clock.clone(),
        brain.auto_tick.config(),
    );
    brain.auto_tick = new_at;

    // Port T.6: spawn the stdin reader and auto-tick driver threads.
    // The main thread receives both channels via mpsc and dispatches.
    use std::sync::mpsc;
    use std::thread;
    use std::time::Duration;

    enum Msg {
        Line(String),
        StdinClosed,
        AutoTickWake,
    }

    let (tx, rx) = mpsc::channel::<Msg>();

    // Stdin reader thread.
    let tx_stdin = tx.clone();
    thread::spawn(move || {
        let stdin = io::stdin();
        let lock = stdin.lock();
        for line in lock.lines() {
            match line {
                Ok(l) => {
                    if tx_stdin.send(Msg::Line(l)).is_err() {
                        return;
                    }
                }
                Err(e) => {
                    eprintln!("[lu4d-mcp] stdin read error: {}", e);
                    let _ = tx_stdin.send(Msg::StdinClosed);
                    return;
                }
            }
        }
        let _ = tx_stdin.send(Msg::StdinClosed);
    });

    // Auto-tick driver thread. Wakes the main thread periodically when
    // gates pass (we re-check `has_wonders` on the main side since the
    // wonder queue is not Arc'd).
    let tx_tick = tx.clone();
    let driver = brain.auto_tick.clone();
    let idle_state = brain.idle_state.clone();
    thread::spawn(move || {
        let check_interval = Duration::from_millis(2_000);
        loop {
            thread::sleep(check_interval);
            // Optimistic gate: re-check on main with `has_wonders`.
            // We pass `true` here so a queue that just got populated
            // is not missed by the optimistic side.
            let dec = driver.evaluate(&idle_state, true);
            if matches!(dec, crate::idle::AutoTickDecision::Fire) {
                if tx_tick.send(Msg::AutoTickWake).is_err() {
                    return;
                }
            }
        }
    });

    while let Ok(msg) = rx.recv() {
        match msg {
            Msg::StdinClosed => break,
            Msg::AutoTickWake => {
                // Final gate: re-evaluate with the actual wonder-queue
                // state (only available on the Brain-owning thread).
                let has_wonders = !brain.wonder_queue.is_empty();
                let dec = brain.auto_tick.evaluate(&brain.idle_state, has_wonders);
                if matches!(dec, crate::idle::AutoTickDecision::Fire) {
                    let fetcher = brain.default_research_fetcher();
                    let _res = brain.idle_tick(&fetcher, 0);
                    brain.auto_tick.record_fired();
                    // Drain reasoning gaps (mirror handle_tools_call).
                    let _ = brain.flush_gaps();
                }
            }
            Msg::Line(line) => {
                let trimmed = line.trim();
                if trimmed.is_empty() { continue; }

                let req: Value = match serde_json::from_str(trimmed) {
                    Ok(v) => v,
                    Err(e) => {
                        let resp = error_response(Value::Null, -32700, &format!("parse error: {}", e));
                        write_message(&mut out, &resp)?;
                        continue;
                    }
                };

                let id = req.get("id").cloned();
                let method = req.get("method").and_then(|v| v.as_str()).unwrap_or("");
                let params = req.get("params").cloned().unwrap_or(Value::Null);

                // Notifications (no id) — handle what we can, ignore the rest.
                if id.is_none() {
                    if method == "notifications/initialized" || method == "initialized" {
                        eprintln!("[lu4d-mcp] client initialized");
                    } else if method == "notifications/cancelled" || method == "notifications/cancel" {
                        // no-op
                    } else {
                        eprintln!("[lu4d-mcp] ignoring notification: {}", method);
                    }
                    continue;
                }

                let id = id.unwrap_or(Value::Null);

                let resp = match method {
                    "initialize" => handle_initialize(id.clone(), params),
                    "tools/list" => handle_tools_list(brain, id.clone()),
                    "tools/call" => {
                        let result = handle_tools_call(brain, params);
                        if save_on_every_write {
                            if let Err(e) = brain.save() {
                                eprintln!("[lu4d-mcp] autosave failed: {}", e);
                            }
                        }
                        match result {
                            Ok(v) => success_response(id.clone(), v),
                            Err((code, msg)) => error_response(id.clone(), code, &msg),
                        }
                    }
                    "ping" => success_response(id.clone(), json!({})),
                    "shutdown" => {
                        let r = success_response(id.clone(), json!(null));
                        write_message(&mut out, &r)?;
                        break;
                    }
                    _ => error_response(id.clone(), -32601, &format!("method not found: {}", method)),
                };
                write_message(&mut out, &resp)?;
            }
        }
    }

    eprintln!("[lu4d-mcp] stdin closed, shutting down");
    Ok(())
}

fn write_message<W: Write>(out: &mut W, value: &Value) -> io::Result<()> {
    let line = serde_json::to_string(value).unwrap_or_else(|_| "{}".into());
    out.write_all(line.as_bytes())?;
    out.write_all(b"\n")?;
    out.flush()
}

fn success_response(id: Value, result: Value) -> Value {
    json!({"jsonrpc": "2.0", "id": id, "result": result})
}

fn error_response(id: Value, code: i64, message: &str) -> Value {
    json!({"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}})
}

fn handle_initialize(id: Value, _params: Value) -> Value {
    success_response(id, json!({
        "protocolVersion": MCP_PROTOCOL_VERSION,
        "capabilities": {"tools": {}},
        "serverInfo": {"name": "lu4d", "version": env!("CARGO_PKG_VERSION")},
    }))
}

fn handle_tools_list(brain: &Brain, id: Value) -> Value {
    let mut tools = tool_definitions();
    if let Some(arr) = tools.as_array_mut() {
        for d in crate::mcp_cap_runtime::static_tool_defs() { arr.push(d); }
        for d in brain.pack_registry.tool_decls() { arr.push(d); }
    }
    success_response(id, json!({"tools": tools}))
}

fn tool_definitions() -> Value {
    json!([
        {
            "name": "lu4d_teach",
            "description": "Add knowledge to the Lu 4D brain. Creates nodes for subject and object (reusing existing ones by deterministic seed) and adds a temporal bond with a keyframe. Q.1: accepts an optional `kind` to type new nodes (Definition, Mechanism, Composition, Purpose, Contrast, Causal, Example, Code, Fact, Procedure, Episode, Preference, Concept, Raw); when omitted, kind is inferred from the relation alias (DefinedAs->Definition, WorksBy->Mechanism, ComposedOf->Composition, IsFor->Purpose, Unlike->Contrast, LeadsTo->Causal, SuchAs->Example) or from text shape (code fences -> Code). OP-EXTRACT-PREP: optional `negated` bool stamps the bond via the NegationOverlay sidecar (X is NOT a Y); default false.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string", "description": "The source node label"},
                    "relation": {"type": "string", "description": "Bond type. Canonical: IsA, HasProperty, Causes, PartOf, UsedFor, RelatedTo, ContrastedWith, DerivedFrom, Requires, Produces, Opposes. Typed aliases (Q.1) accepted: DefinedAs, WorksBy, ComposedOf, IsFor, Unlike, LeadsTo, SuchAs (mapped to the closest canonical type)."},
                    "object": {"type": "string", "description": "The target node label"},
                    "weight": {"type": "number", "description": "Bond weight 0-1 (default 0.8)"},
                    "confidence": {"type": "string", "description": "Confidence: Terminal, Verified, Unverified, Inferred, Contested (default Verified)"},
                    "kind": {"type": "string", "description": "Optional KnowledgeContent variant tag (Q.1). One of: Fact, Concept, Procedure, Episode, Preference, Raw, Code, Image, Definition, Mechanism, Composition, Purpose, Contrast, Causal, Example, Meta. Case-insensitive. When omitted, the relation alias and text shape drive inference; falls back to Concept."},
                    "negated": {"type": "boolean", "description": "OP-EXTRACT-PREP. Optional. When true, the bond is stamped as negated via NegationOverlay (i.e. \"X is NOT a Y\"). Default false. Routes through Brain::teach_with_kind for node/bond formation, then sets the negation overlay flag — equivalent to Brain::teach_with_negation but preserves the kind hint."}
                },
                "required": ["subject", "relation", "object"]
            }
        },
        {
            "name": "lu4d_teach_corpus",
            "description": "Port T-Ingest. Strict superset of `lu4d_teach`: same subject/relation/object/weight/confidence/kind args, plus `corpus` (required, e.g. 'wikimedia/wikipedia', 'grokipedia.com'), `run_id` (optional caller-supplied string; if omitted the server fills it per `ingest_run_id_strategy`), and `source_url` (optional per-bond URL or row id within the corpus). Stamps the resulting bond with `SourceAttribution::TrainingCorpus { corpus, run_id, url }` so the source attribution is correctly tagged for retrieval, audit, and revoke-by-source. When `ingest_auto_approve_threshold` is unset OR confidence >= threshold, the bond applies directly (mirroring `lu4d_teach` semantics). When the threshold gate fires, the change is pushed onto the T.3 PendingLog as an `AddBond` with `proposed_by = TrainingCorpus`, returning a `pending_id` and `gated=true` so the caller can approve via `lu4d_approve_changes`. OP-EXTRACT-PREP: optional `negated` bool sets the NegationOverlay flag (forwarded through PendingChangeKind::AddBond { negated } on the gated path); default false.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string"},
                    "relation": {"type": "string"},
                    "object": {"type": "string"},
                    "weight": {"type": "number"},
                    "confidence": {"type": "string"},
                    "kind": {"type": "string"},
                    "corpus": {"type": "string", "description": "Required. Identifier of the training corpus this bond was extracted from (e.g. 'wikimedia/wikipedia')."},
                    "run_id": {"type": "string", "description": "Optional. Caller-supplied identifier for this ingestion run. If omitted, the server fills it per `ingest_run_id_strategy` (default `ServerStartup`)."},
                    "source_url": {"type": "string", "description": "Optional. Per-bond URL or row id within the corpus (article slug, dataset row id, etc.). Surfaces in attribution for GDPR/revoke purposes."},
                    "negated": {"type": "boolean", "description": "OP-EXTRACT-PREP. Optional. When true, the bond is stamped as negated via NegationOverlay (i.e. \"X is NOT a Y\"). Default false. Routes through Brain::teach_corpus_with_kind for node/bond formation, then sets the negation overlay flag — preserves kind hint and TrainingCorpus source attribution. Also threads into PendingChangeKind::AddBond { negated } on the T.3 gated path so approve/reject parity holds."}
                },
                "required": ["subject", "relation", "object", "corpus"]
            }
        },
        {
            "name": "lu4d_teach_corpus_batch",
            "description": "Port BatchIngest. IPC-amortising batch wrapper around `lu4d_teach_corpus`. Accepts up to 1000 entries per call; each entry is routed through the SAME internal path as a single `lu4d_teach_corpus` call (anti-bypass invariant: identical trust multiplier, T.3 PendingLog gating, source attribution stamping, hub-cap write-side, and pending/direct-apply branching). Per-entry errors do NOT fail the whole batch — they are reported in the response `errors` array. Returns aggregate counters (`bonds_landed`, `bonds_deduped`, `bonds_gated`, `entries_attempted`) so the harness can drive throughput without inspecting every per-entry response. Exists purely as an IPC speedup; the per-bond semantics are byte-equivalent to N sequential `lu4d_teach_corpus` calls (modulo system-clock timestamps). The grep-guard test `tingest_batch_grep_guard_uses_only_tool_teach_corpus` asserts the function body uses ONLY `tool_teach_corpus` as its inner write path.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "entries": {
                        "type": "array",
                        "description": "Array of teach_corpus entries. Each entry mirrors `lu4d_teach_corpus` args: subject, relation, object, corpus (required), plus optional weight, confidence, kind, run_id, source_url. Cap: 1000 entries per call.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "subject": {"type": "string"},
                                "relation": {"type": "string"},
                                "object": {"type": "string"},
                                "weight": {"type": "number"},
                                "confidence": {"type": "string"},
                                "kind": {"type": "string"},
                                "corpus": {"type": "string"},
                                "run_id": {"type": "string"},
                                "source_url": {"type": "string"},
                                "negated": {"type": "boolean"}
                            },
                            "required": ["subject", "relation", "object", "corpus"]
                        }
                    }
                },
                "required": ["entries"]
            }
        },
        {
            "name": "lu4d_user_statement",
            "description": "Port T-Ingest.2. Route Tier 2 prompt text (e.g. user-asked questions, instruction-tuning prompts) into Left Brain operational memory as a `SessionEntryKind::UserStatement` entry. Hard invariant: NEVER forms a Right Brain bond -- prompts are hypothetical-shape and bonding them as facts would inject hallucination-shaped data into the architecture designed to prevent hallucination. Use this paired with `lu4d_teach_corpus` (which DOES bond): prompt -> lu4d_user_statement, model response -> lu4d_teach_corpus. Returns entry_id, session_id (the one used or created), kind echoed as 'user_statement', and a structural confirmation that no bond was formed. When `session_id` is omitted, behaviour is governed by `tier2_prompt_default_session_strategy` (default Ephemeral: open a fresh LB session for this prompt). Caps text byte length at `tier2_prompt_max_text_bytes` (default 65536).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Required. The Tier 2 prompt text (e.g. 'Write a haiku about autumn')."},
                    "source_label": {"type": "string", "description": "Optional. Corpus identifier (e.g. 'HuggingFaceH4/no_robots') so audits can group entries by ingest source."},
                    "session_id": {"type": "integer", "description": "Optional. Attach this entry to an existing LB session. If omitted, the configured Tier 2 strategy decides (Ephemeral: open a fresh session; Reuse: attach to currently-active session, fall back to scratchpad)."}
                },
                "required": ["text"]
            }
        },

        {
            "name": "lu4d_file_read",
            "description": "Port File-IO. Read a file from local disk and either return parsed content (Inspect), route the content into Left Brain operational memory as a UserStatement entry (no bond formation, asymmetry-correct for prompt-shaped content), or route structured triples into Right Brain via teach_corpus semantics (TeachCorpus, CSV/JSON only). Supported formats: md, txt, csv, json, pdf, docx, code (any source-code file). PDF text is extracted via pdf-extract; DOCX text via a minimal <w:t> scan over word/document.xml. Honours runtime knobs `file_io_max_size_bytes` (default 50 MB) and `file_io_allowed_read_paths` (empty default = no restriction).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute path to the file to read."},
                    "kind": {"type": "string", "description": "Optional explicit format hint. One of: md, txt, csv, json, pdf, docx, code. Defaults to extension-based detection."},
                    "route": {
                        "type": "object",
                        "description": "How to route the parsed content. One of: { Inspect: {} } (return parsed payload), { UserStatement: { source_label?, session_id? } } (push as LB UserStatement entry), { TeachCorpus: { corpus, run_id? } } (route structured triples through teach_corpus; CSV/JSON only)."
                    }
                },
                "required": ["path", "route"]
            }
        },
        {
            "name": "lu4d_file_write",
            "description": "Port File-IO. Emit a file in the chosen kind. md/txt/code take a UTF-8 `content` string. csv expects `content` as an array-of-arrays-of-strings (header row at index 0). json accepts either a value or a json-encoded string. pdf renders `content` as plain text via printpdf with naive line wrapping (markdown not parsed). docx emits a minimal valid OOXML archive (one paragraph per '\n' in content). Writes are constrained to `file_io_allowed_write_paths` (default ['/var/lib/lu/outputs/', '/tmp/lu/']).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute output path."},
                    "kind": {"type": "string", "description": "One of: md, txt, csv, json, pdf, docx, code."},
                    "content": {"description": "Format-specific content. md/txt/code/pdf/docx: string; csv: array of array of strings; json: value or json-encoded string."}
                },
                "required": ["path", "kind", "content"]
            }
        },
        {
            "name": "lu4d_image_ingest",
            "description": "Port R input. Read an image file from disk and route the configured SceneGraphExtractor's triplets per the `route` arg through existing teach_corpus / user_statement pipes - OR return triplets without ingest (Inspect). Honours runtime knobs image_max_size_bytes (default 25 MB), image_supported_mime_types (default jpeg/png/webp/gif), and image_extractor_kind (default Stub). Stub returns a single placeholder triplet - production deployments should switch to Subprocess and configure image_extractor_subprocess_path. CapabilityPack extractor is documented but deferred to Cat.5.1.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute path to the image file."},
                    "mime_type": {"type": "string", "description": "Optional explicit mime type. Default: sniff leading bytes."},
                    "max_triplets": {"type": "integer", "description": "Optional advisory cap on extractor output."},
                    "route": {"type": "object", "description": "How to route extracted triplets. One of: { Inspect: {} } | { UserStatement: { source_label?, session_id? } } | { TeachCorpus: { corpus, run_id? } }."}
                },
                "required": ["path", "route"]
            }
        },
        {
            "name": "lu4d_image_describe",
            "description": "Port R input. Convenience wrapper: read an image and return the configured extractor's triplets as a human-readable description string (semicolon-joined 'subject predicate object'). No bond formation, no LB session entry - pure inspection. Honours the same size / mime / extractor knobs as lu4d_image_ingest.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "mime_type": {"type": "string"},
                    "max_triplets": {"type": "integer"}
                },
                "required": ["path"]
            }
        },
        {
            "name": "lu4d_image_extractor_status",
            "description": "Port R input. Diagnostic: report the configured extractor kind (stub | subprocess | capability_pack:<id>), the configured subprocess timeout, and (if applicable) whether the configured subprocess path exists on disk. Also returns the configured image_max_size_bytes and image_supported_mime_types so operators can verify section #15 is hydrated.",
            "inputSchema": {"type": "object", "properties": {}, "required": []}
        },
        {
            "name": "lu4d_file_extract_to_pending",
            "description": "Port File-IO. Convenience: read a structured file (csv or json) and queue extracted triples as PendingChangeKind::AddBond entries on the T.3 PendingLog with SourceKind::TrainingCorpus attribution. Anti-mutation invariant: routes ALL state mutation through `brain.pending.push` (NOT teach* / network::add_bond), so the existing approve / reject / revoke gates apply unchanged. Caps emission at `max_bonds_per_file` (default 10000). Text-only formats (md, txt, pdf, docx, code) are refused with a clear error pointing at lu4d_file_read { route: UserStatement }.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "corpus": {"type": "string", "description": "Required corpus identifier stamped onto every queued bond."},
                    "max_bonds_per_file": {"type": "integer", "description": "Hard cap on emitted bonds. Default 10000."},
                    "kind": {"type": "string", "description": "Optional explicit format hint."}
                },
                "required": ["path", "corpus"]
            }
        },
        {
            "name": "lu4d_learn_text",
            "description": "Text-extractor port (2026-05-31). Run the 8-layer Wikipedia-style smart extractor + multi-sentence Pattern A/B + pronoun-anchor extractor over a free-form text body and queue every surviving bond as PendingChangeKind::AddBond on the T.3 PendingLog with SourceKind::TrainingCorpus attribution. Anti-mutation invariant: routes ALL state mutation through `brain.pending.push` (NOT teach* / network::add_bond), so the existing approve / reject / revoke gates apply unchanged. Applies a knowledge-aware filter that keeps a candidate bond when either endpoint already exists in keyword_index OR when its shape is small + clean (so the graph can grow); drops verb-led NP fragments with two-unknown endpoints. Returns {bonds_emitted, bonds_dropped_knowledge_filter, bonds_dropped_validation, layer_breakdown, sample[5]}.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The article body to extract bonds from."},
                    "title": {"type": "string", "description": "Article title used as the subject of L1/L2/L2b/L3/L4/L5 bonds and the anchor for the Path-2 pronoun layer."},
                    "source_url": {"type": "string", "description": "Optional URL stamped onto every queued bond's source attribution."},
                    "corpus": {"type": "string", "description": "Required corpus identifier stamped onto every queued bond."}
                },
                "required": ["text", "corpus"]
            }
        },
        {
            "name": "lu4d_ask",
            "description": "Query the graph with temporal spread activation. Finds seed nodes via keyword search, then activates neighbors. Hebbian-strengthens co-activated bonds within the active session. Q.3: when `session_id` is supplied, the call reads prior conversational state (pronouns resolve to the highest-weight recent anchor; `tell me more` re-asks the previous turn topic; `what was my first question?` returns verbatim from history) and writes the new turn back to the session memory.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "depth": {"type": "integer", "description": "Max hops from seeds (default 3)"},
                    "top_k": {"type": "integer", "description": "Max results to return (default 10)"},
                    "bond_filter": {"type": "array", "items": {"type": "string"}, "description": "Optional bond types to traverse (e.g. [\"IsA\",\"HasProperty\"]). Omit to follow all bonds."},
                    "session_id": {"type": "integer", "description": "Q.3 session id from lu4d_session_start. When present, conversational memory (pronoun resolution, elaboration, backreference, parallel-session isolation) is engaged."}
                },
                "required": ["query"]
            }
        },
        {
            "name": "lu4d_status",
            "description": "Get brain stats: node count, bond count, session info.",
            "inputSchema": {"type": "object", "properties": {}}
        },
        {
            "name": "lu4d_time_status",
            "description": "Port T.1.1: report current time-source mode (system | df_service), DF Time Service trust state, last fetch drift, consecutive failures, and whether a localhost-loopback URL warning is active.",
            "inputSchema": {"type": "object", "properties": {}}
        },
        json!({
            "name": "lu4d_sense_status",
            "description": "Port Sense — diagnostics for runtime sense disambiguation. Given a polysemous seed and an optional context window, returns the detected sense clusters with their per-cluster scores and the elected winner (if any). Read-only.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The seed term to inspect."},
                    "context": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional context window terms."
                    }
                },
                "required": ["query"]
            }
        }),
        {
            "name": "lu4d_session_start",
            "description": "Start a new temporal session. Isolates Hebbian learning from baseline. Auto-ends any active session.",
            "inputSchema": {"type": "object", "properties": {}}
        },
        {
            "name": "lu4d_session_end",
            "description": "End the currently active session.",
            "inputSchema": {"type": "object", "properties": {}}
        },
        {
            "name": "lu4d_timetravel",
            "description": "Query the graph at a historical timestamp and diff against the current state. Shows which associations emerged, faded, strengthened, or weakened.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "at_ms": {"type": "integer", "description": "Absolute epoch millis to query at"},
                    "ms_ago": {"type": "integer", "description": "Milliseconds before now (alternative to at_ms)"}
                },
                "required": ["query"]
            }
        },
        {
            "name": "lu4d_consolidate",
            "description": "Run baseline consolidation. Promotes bonds strengthened across N distinct sessions to baseline and prunes decayed keyframes.",
            "inputSchema": {"type": "object", "properties": {}}
        },
        {
            "name": "lu4d_purge_stopwords",
            "description": "Remove all nodes whose content is a common stopword (function words, auxiliaries, pronouns, prepositions). Strips all bonds attached to those nodes. Returns counts of nodes and bonds removed.",
            "inputSchema": {"type": "object", "properties": {}}
        },
        {
            "name": "lu4d_prune_legacy_goals",
            "description": "One-shot: remove keyword-index entries for any node whose content begins with 'goal:' or 'step:' (case-insensitive). These were created by v2 thinking before add_node_unindexed existed and pollute keyword recall. Nodes and bonds are preserved; only keyword-discoverability is dropped. Returns the count pruned.",
            "inputSchema": {"type": "object", "properties": {}}
        },
        {
            "name": "lu4d_think",
            "description": "Run one Thinking-Layer pass: ReAct-style Observe -> Plan -> Act -> Reflect loop over the temporal graph. Generates hypothesis bonds at low weight, runs a sub-spread to test them, and promotes confirmed hypotheses. Synchronous, bounded depth, no LLM calls. Returns the final answer plus a ThoughtTrace.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_steps": {"type": "integer", "description": "Max iterations (default 5)"},
                    "variant_mode": {"type": "boolean", "description": "When true, run a 5-strategy variant arena per hypothesis target; only beliefs whose arena produced a winner survive. Default false."},
                    "confidence_to_answer": {"type": "number", "description": "Score >= this triggers early exit; otherwise loop runs up to max_steps. Default 0.7."},
                    "hypotheses_per_step": {"type": "integer", "description": "Cap on hypothesis bonds proposed per step. Default 3."}
                },
                "required": ["query"]
            }
        },
        {
            "name": "lu4d_wonder",
            "description": "Inspect the Curiosity Engine: returns the top-N highest-intensity curiosity signals (DeadEnd, IsolatedNode, KnowledgeDesert, HubBoundary) and a snapshot of the per-domain interest profile. Read-only; does not mutate the brain.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "top_n": {"type": "integer", "description": "Number of wonder-queue signals to return (default 10)"}
                }
            }
        },
        {
            "name": "lu4d_save",
            "description": "Explicitly persist the brain to its .lbr file. OP-SAVE-ASYNC: `mode` (default \"blocking\") = \"blocking\" returns only after the save is committed to disk (wire-compat); `mode=\"async\"` returns immediately after snapshot capture and the caller polls `lu4d_save_status` until state is `completed` or `failed`. OP-BRAIN-V4 Phase 8: when the brain was loaded via the v4 directory path (Brain::load_v4), this tool is a no-op returning success immediately because v4 durability is provided continuously by the flush worker (MemTable -> WAL -> L0 SST). Both blocking and async modes short-circuit to a success response in that case.",
            "inputSchema": {"type": "object", "properties": {
                "mode": {"type": "string", "description": "blocking (default) | async"}
            }}
        },
        {
            "name": "lu4d_save_status",
            "description": "OP-SAVE-ASYNC: poll the state of the most recent async save. Returns {state, started_at_ms, error}. state is one of: idle, running, completed, failed. Calling this also drains a completed worker (rename + WAL truncate + .kw compact), so state=completed implies the new brain.lbr is durable on disk.",
            "inputSchema": {"type": "object", "properties": {}}
        },
        {
            "name": "lu4d_flush",
            "description": "OP-FLUSH-1: force a synchronous MemTable -> L0 SST flush. Seals the current MemTable, writes an L0 SST, commits the MANIFEST, and truncates the WAL up to the flushed watermark. Unlike lu4d_save (a documented no-op for v4-directory brains that rely on the background flush worker), lu4d_flush drains on demand so the WAL is bounded before the next boot. Safe to call during background compaction (shared commit_lock). Returns {flushed:true} on success.",
            "inputSchema": {"type": "object", "properties": {}}
        },
        {
            "name": "lu4d_variants_generate",
            "description": "Spawn a Variant Generation arena for a goal seed. Runs 5 perturbation strategies (Baseline, DomainShift, Deep, Contrarian, Interdisciplinary), scores each in its own scope by coherence x novelty x parsimony, and returns the arena (specs + scores + winner_id, if any). Variant keyframes are invisible outside variant scope until lu4d_variants_promote is called.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "goal_seed": {"type": "string", "description": "Hex-encoded 32-byte goal seed"}
                },
                "required": ["goal_seed"]
            }
        },
        {
            "name": "lu4d_variants_promote",
            "description": "Promote a winning variant: re-tag every Variant-source keyframe with the given variant id as PromotedFromVariant (visible to baseline scope). Other variants' keyframes decay naturally. Returns the count of bonds promoted.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "winner_id": {"type": "integer"}
                },
                "required": ["winner_id"]
            }
        },
        {
            "name": "lu4d_reflect",
            "description": "Manually trigger a self-reflection pass over the recent QueryMetric ring buffer. Emits AdjustmentProposals for any heuristic that fires. Returns the count of new proposals.",
            "inputSchema": {"type": "object", "properties": {}}
        },
        {
            "name": "lu4d_proposals_list",
            "description": "List adjustment proposals from the meta-reflection log. Optional status_filter is one of: pending, approved, rejected, applied. Default lists all.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "status_filter": {"type": "string"}
                }
            }
        },
        {
            "name": "lu4d_proposal_decide",
            "description": "Approve or reject a proposal by id. decision is approve or reject. Approving applies the adjustment to the matching brain config knob and marks the proposal Applied; for notice-only targets the status moves to Approved without mutating state.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "decision": {"type": "string"},
                    "reason": {"type": "string"}
                },
                "required": ["id", "decision"]
            }
        },
        {
            "name": "lu4d_multi_think",
            "description": "Run one multi-agent thinking pass: RecallAgent -> SynthesisAgent -> CriticAgent over the temporal graph in scope-isolated views. Returns all three agent traces plus the consolidated (Critic-vetted) beliefs. Vetted beliefs are NOT auto-promoted to baseline; multi-think is observation, not commitment.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "critic_confidence_threshold": {"type": "number", "description": "Beliefs below this confidence are dropped by the Critic. Default 0.5."},
                    "max_recall_candidates": {"type": "integer", "description": "Cap on candidate seeds RecallAgent surfaces. Default 32."},
                    "max_hypotheses_per_synthesis": {"type": "integer", "description": "Cap on hypotheses SynthesisAgent emits per pass. Default 8."},
                    "agent_recall_depth": {"type": "integer", "description": "Spread depth used inside the Recall agent scope. Default 3."}
                },
                "required": ["query"]
            }
        },
        {
            "name": "lu4d_narrate",
            "description": "Produce a coherent English prose answer for a query. Wraps lu4d_ask (and optionally lu4d_multi_think) and runs the result through the 16-shape narration pipeline. Cognitive signals (curiosity, bridges, critic vetting, hypothesis-only support) modulate the chosen shape so the answer honestly discloses confidence and gaps. Pure neuro-symbolic; no LLM calls. Optional session_id activates the Q.3 conversational memory pre-pass so 'tell me more', pronoun substitution, and backreference queries resolve against prior turns in the same session.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "depth": {"type": "integer", "description": "Spread depth (default 3)"},
                    "top_k": {"type": "integer", "description": "Max activations (default 10)"},
                    "use_multi_think": {"type": "boolean", "description": "Run multi_think too so CriticVettedAnswer / MultiHypothesisAnswer shapes become available (default false)"},
                    "session_id": {"type": "integer", "description": "OP-NARRATE-V2 (2026-05-31): when set, applies Q.3 conversational rewrite (elaboration / pronoun substitution / backreference) before narration and records the turn after."}
                },
                "required": ["query"]
            }
        },
        {
            "name": "lu4d_scar_bond",
            "description": "Scar a directed bond (from -> to). Pass either hex seeds (from_seed/to_seed) OR content strings (from_content/to_content) which resolve via keyword search. Records a Scar in the log; the bond's keyframe history is preserved untouched. Every subsequent weight read on this bond returns effective_weight = 0.0. Forward-search anchored to wall-clock NOW. Type-enforced at the bond signature.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "from_seed": {"type": "string", "description": "Source seed as 64-char hex (optional if from_content given)"},
                    "to_seed": {"type": "string", "description": "Target seed as 64-char hex (optional if to_content given)"},
                    "from_content": {"type": "string", "description": "Source as content string; resolved via keyword search to top match"},
                    "to_content": {"type": "string", "description": "Target as content string; resolved via keyword search to top match"},
                    "reason": {"type": "string", "description": "Human-readable reason; surfaces in narration when the scar drives a refusal"}
                },
                "required": ["reason"]
            }
        },
        {
            "name": "lu4d_scar_concept",
            "description": "Cascade scar: scar every directed bond touching the given concept (incoming or outgoing). Pass either hex seed OR content string (resolved via keyword search). Per Lu's DF SEED_REVOKE semantics, the concept itself stays in the graph but no action layer can traverse a bond into or out of it.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "seed": {"type": "string", "description": "Concept seed as 64-char hex (optional if seed_content given)"},
                    "seed_content": {"type": "string", "description": "Concept as content string; resolved via keyword search to top match"},
                    "reason": {"type": "string"}
                },
                "required": ["reason"]
            }
        },
        {
            "name": "lu4d_scar_log",
            "description": "Read the scar log. Returns every Scar plus every UnscarRecord, sorted by timestamp. The optional since_ms filter trims the response to entries created on or after that wall-clock millisecond.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "since_ms": {"type": "integer", "description": "Optional epoch-ms lower bound"}
                }
            }
        },
        {
            "name": "lu4d_unscar_bond",
            "description": "Admin-only: remove every active scar on the directed bond (from -> to). Pass either hex seeds or content strings. Logs an UnscarRecord with the supplied reason. Rare operation; the unscar is itself an audit trail entry so the override is visible in the scar log.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "from_seed": {"type": "string", "description": "Source seed as 64-char hex (optional if from_content given)"},
                    "to_seed": {"type": "string", "description": "Target seed as 64-char hex (optional if to_content given)"},
                    "from_content": {"type": "string", "description": "Source as content string"},
                    "to_content": {"type": "string", "description": "Target as content string"},
                    "reason": {"type": "string"}
                },
                "required": ["reason"]
            }
        },
        {
            "name": "lu4d_unscar_concept",
            "description": "Cascade-undo of lu4d_scar_concept. Removes the seed from the forward-protection set (NEW bonds touching it can form again) AND supersedes every active scar on every bond touching the seed. One UnscarRecord per bond. Pass either hex seed or seed_content (resolved via keyword search).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "seed": {"type": "string", "description": "Concept seed as 64-char hex (optional if seed_content given)"},
                    "seed_content": {"type": "string", "description": "Concept as content string; resolved via keyword search to top match"},
                    "reason": {"type": "string"}
                },
                "required": ["reason"]
            }
        },
        {
            "name": "lu4d_research",
            "description": "Manual one-shot research. Queries Grokipedia first, then DuckDuckGo, extracts triples, runs the scar fingerprint check, and queues a Pending ResearchProposal for John's approval. Returns the new proposal id. Does NOT modify the knowledge graph — approval routes through lu4d_research_proposal_decide which itself goes through the scar-aware add_bond path.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Free-form search query"}
                },
                "required": ["query"]
            }
        },
        {
            "name": "lu4d_research_loop_run",
            "description": "Drive the autonomous research loop. Pops up to max_iterations wonders above the research_intensity_threshold, queries the web for each, and queues Pending ResearchProposals. Stops early if the wonder queue empties, the next wonder is below threshold, or the pending-proposal cap is hit. Returns the list of created proposal ids. Port J.2: pass chain=true to chase multi-hop threads instead — same as calling lu4d_research_chain_run directly.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "max_iterations": {"type": "integer", "description": "Default 1"},
                    "chain": {"type": "boolean", "description": "Port J.2: when true, route through research_chain_run for a multi-hop walk. Default false (single-hop, back-compat)."}
                }
            }
        },
        {
            "name": "lu4d_research_chain_run",
            "description": "Port J.2 — chained curiosity research. Kicks off a multi-hop research walk from the wonder-queue top (or from the optional seed_query). When a hop's proposal lands fresh bond endpoints whose neighborhoods are sparse, those endpoints become new CuriositySignals on the wonder queue; the chain continues until depth/cycle caps hit or the queue exhausts. NEVER mutates the knowledge graph — every hop's candidates queue as Pending ResearchProposals for John's approval. Returns a full ChainTrace: per-hop query, source URLs, candidate counts, considered/enqueued/filtered endpoint counts, total cycles, and termination reason (queue_empty / depth_cap / cycles_cap / disabled / initial_error).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "seed_query": {"type": "string", "description": "Optional. When supplied, synthesizes a high-intensity wonder for this query and uses it as the chain seed (mirrors lu4d_research). When omitted, chains from the current wonder-queue top."}
                }
            }
        },
        {
            "name": "lu4d_research_proposals_list",
            "description": "List research proposals. Optional status_filter narrows to a single state (Pending, Approved, Rejected, Applied). Returns full proposals including raw_excerpts and bond_candidates so John can decide.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "status_filter": {"type": "string", "description": "Optional: Pending|Approved|Rejected|Applied"}
                }
            }
        },
        {
            "name": "lu4d_research_proposal_decide",
            "description": "Approve or reject a Pending research proposal by id. Approval triggers teach() for every bond candidate (which itself goes through add_bond_with_scar_check, so a candidate touching a scarred seed is STILL refused). Rejection just records the decision; the proposal is preserved for audit.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "decision": {"type": "string", "enum": ["approve", "reject"]},
                    "reason": {"type": "string", "description": "Optional, used for reject"}
                },
                "required": ["id", "decision"]
            }
        },
        {
            "name": "lu4d_revoke_by_source",
            "description": "Port S.1 — GDPR / research-rollback primitive. Revokes every bond in the brain whose source attribution matches the supplied (kind + filters) predicate. Wraps Brain::revoke_bonds_by_source. ALWAYS call with dry_run=true (default) first to preview the count; pass dry_run=false to actually revoke. Returns matched_count (overlay entries that match) and revoked_count (network bonds removed; 0 when dry_run=true). ADMIN GATE (BR.5): dry_run=false is GATED behind admin_revoke_enabled in runtime_config (default false). Operators must call lu4d_admin_set with admin_revoke_enabled=true before invoking dry_run=false; otherwise the call returns -32603 with an explanation. Dry-run preview is not gated. Default-deny posture so the destructive path stays closed on shared canonical brains until an admin explicitly opens it.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "description": "Required. SourceKind variant to filter by. CANONICAL FORM IS LOWERCASE (the matcher lowercases the input before matching). One of: training_corpus, research, teach, document, unknown, chained_research, session_promote, pack_install. Legacy aliases also accepted (case-insensitive): TrainingCorpus / training / trainingcorpus, ChainedResearch / chainedresearch, SessionPromote / sessionpromote, PackInstall / packinstall."},
                    "dry_run": {"type": "boolean", "description": "When true (default), counts what WOULD be revoked but does not mutate the brain. When false, actually revokes — destructive."},
                    "corpus": {"type": "string", "description": "Filter (TrainingCorpus|Document only): exact-match corpus name. For Document, matches only docs whose corpus is Some(this)."},
                    "run_id": {"type": "string", "description": "Filter (TrainingCorpus only): exact-match run id. Matches only entries whose run_id is Some(this)."},
                    "wonder_seed": {"type": "string", "description": "Filter (Research only): exact-match wonder_seed. Closes the loop: one wonder => every bond it produced."},
                    "url_contains": {"type": "string", "description": "Filter (Research only): substring match against the source URL. Mutually composable with wonder_seed (AND)."},
                    "actor": {"type": "string", "description": "Filter (Teach only): exact-match teach actor (e.g. 'john', 'claude')."},
                    "before_ms": {"type": "integer", "description": "Filter (Teach only): keep only Teach entries with ts_ms < this (strict)."},
                    "after_ms": {"type": "integer", "description": "Filter (Teach only): keep only Teach entries with ts_ms > this (strict)."},
                    "doc_hash": {"type": "string", "description": "Filter (Document only): exact-match doc_hash. Closes the per-document delete-by-hash story."},
                    "chain_id": {"type": "string", "description": "Filter (ChainedResearch only): exact-match chain_id. Revokes every bond produced by a single chain run."},
                    "hop": {"type": "integer", "description": "Filter (ChainedResearch only): exact-match hop number within the chain."},
                    "parent_query_contains": {"type": "string", "description": "Filter (ChainedResearch only): substring match against the hop's parent_query."},
                    "session_id": {"type": "string", "description": "Filter (SessionPromote only): exact-match originating LB session id (stringified u64)."},
                    "entry_id": {"type": "integer", "description": "Filter (SessionPromote only): exact-match LB session entry id."},
                    "pack_id": {"type": "string", "description": "Filter (PackInstall only): exact-match pack id (e.g. \"df.medical-icd10\")."},
                    "publisher_pubkey_hex": {"type": "string", "description": "Filter (PackInstall only): exact-match publisher hex Ed25519 pubkey."},
                    "signed_before_ms": {"type": "integer", "description": "Filter (PackInstall only): keep only entries with signed_at_ms < this (strict)."}
                },
                "required": ["kind"]
            }
        },
        {
            "name": "lu4d_source_attribution_migrate",
            "description": "Port SourceKind-Variants — operator-driven migration of legacy Teach{actor:\"<prefix>:<id>\"} overlay entries to first-class ChainedResearch / SessionPromote / PackInstall variants. OFF BY DEFAULT (dry_run=true; even with dry_run=false the tool refuses to mutate unless confirm_canonical_mutation=true). Scans the source overlay for entries matching legacy prefixes (research_chain:, session_promote:, pack_install:) and rewrites them in place. Returns per-prefix scan/match/migrated counts. ANTI-MUTATION INVARIANT: existing bond weights are NEVER mutated; only the side-table SourceAttribution.kind is upgraded. NEVER auto-run on canonical without explicit operator approval — the canonical brain's pre-port bonds intentionally retain their legacy attribution as a forensic audit trail until the next retrain.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "dry_run": {"type": "boolean", "description": "When true (default), counts entries that WOULD be migrated but does not mutate the overlay."},
                    "confirm_canonical_mutation": {"type": "boolean", "description": "Required to be true (alongside dry_run=false) for any actual mutation. Belt-and-braces guard so a stray dry_run=false call does not silently rewrite the canonical brain."},
                    "limit_per_prefix": {"type": "integer", "description": "Optional safety cap on entries migrated per prefix in one call. Useful for staged migrations."}
                }
            }
        },
        {
            "name": "lu4d_idle_tick",
            "description": "Port T.2 — single tick of the idle work loop. When Lu is idle (no MCP queries in flight) and has been idle for at least min_idle_ms, pops the highest-intensity wonder above the research_intensity_threshold and runs the existing research path to queue a Pending ResearchProposal. Returns {did_work, wonder_seed_processed, outcome, idle_duration_ms, total_idle_ticks}. The outcome is one of: no_work_no_signals (queue empty / below threshold / research disabled), no_work_not_idle (a query is in flight or min_idle_ms gate not met), research_completed (proposal queued), research_failed (router/extractor error). KNOWN GAP: idle research currently auto-commits to the Pending proposal queue using the existing manual-research path; T.3 replaces this with the pending-changes-with-approval gate. KNOWN GAP: caller-driven for now; T.6 daemon will auto-drive this on an interval.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "min_idle_ms": {"type": "integer", "description": "Optional: minimum idle duration (ms) required before this tick will do work. Default 0 (no gate). Use a non-zero value to ensure Lu has been idle for at least N ms before draining a wonder."}
                }
            }
        },
        {
            "name": "lu4d_reason",
            "description": "ReasoningAgent — articulate the highest-confidence bond chain anchored on the query as a step-by-step natural-language trace. Each step references a real bond in the temporal graph (no invented facts). Temporal-aware: bonds visible to the chain are those valid under the supplied scope (default Current; pass scope='historical' with at_ms, or scope='session' to ride an active session). Returns both the human-readable final_answer AND the structured (from, BondType, to) audit list. Read-only — no agent-tagged keyframes are written.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "scope": {"type": "string", "description": "Optional: 'current' (default), 'historical' (requires at_ms), or 'session' (uses active session)."},
                    "at_ms": {"type": "integer", "description": "Optional: epoch-ms for scope='historical'."},
                    "max_chain_length": {"type": "integer", "description": "Maximum chain length; default 8 (Port H — Learning #578). Hard ceiling MAX_CHAIN_DEPTH=8 caps composition cost."},
                    "min_chain_length": {"type": "integer", "description": "Minimum chain length to articulate; default 2. Falls back to length 1 if no longer chain is constructible."},
                    "min_edge_weight": {"type": "number", "description": "Drop bonds whose in-scope effective weight is at or below this. Default 0.05."}
                },
                "required": ["query"]
            }
        },
        {
            "name": "lu4d_pending_changes",
            "description": "Port T.3 - list pending changes (idle-research-derived bonds and any other autonomous proposals awaiting human review). Filters: status (Pending|Approved|Rejected|Applied|all), source_kind (Research|Teach|Document|TrainingCorpus|Unknown|all), min_confidence, max_age_ms. Default limit 50. Returns entries with pending_id, change_summary, source, confidence, reason, status, plus total_count.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "status": {"type": "string"},
                    "source_kind": {"type": "string"},
                    "min_confidence": {"type": "number"},
                    "max_age_ms": {"type": "integer"},
                    "limit": {"type": "integer"}
                }
            }
        },
        {
            "name": "lu4d_approve_pending",
            "description": "Port T.3 - approve + apply pending changes. Provide either pending_ids:[u64], all:true (every Pending entry), or filter args (same as lu4d_pending_changes). Routes each AddBond through teach_with_source so the scar gate fires; SetNegation goes through the negation overlay; ModifyBondWeight appends a fresh Manual-source keyframe. Returns {approved_count, applied_count, failures[]}. Atomic-ish: all approvals happen first, then applies; per-id apply failures are surfaced in `failures` rather than rolled back.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "pending_ids": {"type": "array", "items": {"type": "integer"}},
                    "all": {"type": "boolean"},
                    "status": {"type": "string"},
                    "source_kind": {"type": "string"},
                    "min_confidence": {"type": "number"},
                    "max_age_ms": {"type": "integer"},
                    "approved_by": {"type": "string", "description": "Default 'john'"}
                }
            }
        },
        {
            "name": "lu4d_reject_pending",
            "description": "Port T.3 - reject pending changes (Pending -> Rejected). Same id-resolution shape as lu4d_approve_pending. Optional `reason` is recorded for audit. Rejection does not mutate the brain.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "pending_ids": {"type": "array", "items": {"type": "integer"}},
                    "all": {"type": "boolean"},
                    "status": {"type": "string"},
                    "source_kind": {"type": "string"},
                    "min_confidence": {"type": "number"},
                    "max_age_ms": {"type": "integer"},
                    "reason": {"type": "string"},
                    "rejected_by": {"type": "string", "description": "Default 'john'"}
                }
            }
        },
        {
            "name": "lu4d_modify_pending",
            "description": "Port T.3 - edit a still-Pending entry's confidence, weight (AddBond / ModifyBondWeight kinds), or reason. Returns the updated entry. Errors if the entry is already Approved/Rejected/Applied.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "pending_id": {"type": "integer"},
                    "modifications": {
                        "type": "object",
                        "properties": {
                            "confidence": {"type": "number"},
                            "weight": {"type": "number"},
                            "reason": {"type": "string"}
                        }
                    }
                },
                "required": ["pending_id"]
            }
        },
        {
            "name": "lu4d_pending_status",
            "description": "Port T.3 - summary of the pending changes log. Returns {pending_count, approved_unapplied_count, applied_total, rejected_total, last_commit_at_ms, thinking_indicator, active_queries}. thinking_indicator is 'thinking' when pending entries exist OR queries are in flight, else 'settled'.",
            "inputSchema": {
                "type": "object",
                "properties": {}
            }
        },
        {
            "name": "lu4d_freshness_status",
            "description": "Port T.5 - bond freshness/decay status. Returns bucket counts (last 24h, 7d, 30d, 90d, 180d, 365d, older), mean+median age in days, top-N stalest bonds (default N=10). Each stalest entry includes subject_text, object_text, bond_type, age_days, stored_weight, effective_weight_after_decay. Pure presentation -- stored bond weights are never mutated.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "top_n": {"type": "number", "description": "Number of stalest bonds to return (default 10)."}
                }
            }
        },
        {
            "name": "lu4d_configure",
            "description": "DEPRECATED (Port BR.2): prefer the per-port setters (lu4d_sense_set, lu4d_chained_research_set, lu4d_image_set, lu4d_file_io_set, lu4d_ingest_set, lu4d_recall_set, lu4d_admin_set) or the lu4d_configure_batch dispatcher; this omnibus tool covers only 10/30 RuntimeConfig fields and will be removed in a future port. Port T.5 / T.6 - set runtime knobs. T.5: decay_half_life_days (1-3650, default 90), decay_enabled (bool, default true). T.6: auto_tick_enabled (bool, default true), auto_tick_idle_threshold_ms (1000-86400000, default 30000), auto_tick_min_interval_ms (1000-86400000, default 60000), auto_tick_max_per_hour (1-36000, default 60). Returns the effective config + bounds after applying. Pass with no args to read the current config.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "decay_half_life_days": {"type": "number", "description": "Half-life in days (1-3650)."},
                    "decay_enabled": {"type": "boolean", "description": "Enable/disable freshness decay."},
                    "auto_tick_enabled": {"type": "boolean", "description": "Port T.6 - master enable for the auto-tick daemon driver."},
                    "auto_tick_idle_threshold_ms": {"type": "number", "description": "Port T.6 - minimum brain-idle duration before the driver fires a tick (ms, 1000-86400000)."},
                    "auto_tick_min_interval_ms": {"type": "number", "description": "Port T.6 - minimum elapsed time between two consecutive driver fires (ms, 1000-86400000)."},
                    "auto_tick_max_per_hour": {"type": "number", "description": "Port T.6 - sliding-window cap on driver fires per hour (1-36000)."},
                    "trust_default_multiplier": {"type": "number", "description": "Port Cat.3a - default source-trust multiplier applied to bonds whose source URL host doesn't match any TrustTable entry. Range [0.0, 2.0]; default 0.4."},
                    "trust_learning_enabled": {"type": "boolean", "description": "Port Cat.3a - master switch for source-trust weighting. When false, every lookup returns 1.0 (pass-through) regardless of the TrustTable. Default true."}
                }
            }
        },
        {
            "name": "lu4d_auto_tick_status",
            "description": "Port T.6 - read-only snapshot of the auto-tick daemon driver. Returns last_tick_at_ms, ticks_in_last_hour, run state (running / paused / disabled), next_tick_eta_ms (when calculable), full config snapshot. The driver fires `idle_tick` automatically when the brain has been idle past the configured threshold; this tool surfaces what it's doing without driving any work itself.",
            "inputSchema": {"type": "object", "properties": {}}
        },
        {
            "name": "lu4d_auto_tick_pause",
            "description": "Port T.6 - pause the auto-tick driver for the lifetime of this lu process. Returns prior state. Manual `lu4d_idle_tick` calls keep working; only the auto-driver is suspended.",
            "inputSchema": {"type": "object", "properties": {}}
        },
        {
            "name": "lu4d_auto_tick_resume",
            "description": "Port T.6 - resume the auto-tick driver after a `lu4d_auto_tick_pause`. Returns prior state.",
            "inputSchema": {"type": "object", "properties": {}}
        },
        {
            "name": "lu4d_rejection_patterns",
            "description": "Port T.8 - list learned rejection patterns. The proposal pre-filter (push_pending_with_filter) consults the same data; this tool surfaces what the filter is enforcing. Returns patterns sorted by last_rejection_at_ms desc, plus the proposals_filtered_total counter so John can see how many bad proposals Lu filtered out.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "include_expired": {"type": "boolean", "description": "Include patterns that have decayed past `pattern_decay_days` of inactivity. Default false."},
                    "limit": {"type": "integer", "description": "Maximum number of patterns to return. Default 20."}
                }
            }
        },
        {
            "name": "lu4d_trust_status",
            "description": "Port Cat.3a - read the source-trust table used at bond formation time. Returns entries (domain_pattern, multiplier, note), default_multiplier (applied to unknown domains), and learning_enabled (master switch). Default table per C2 hierarchy: grokipedia.com=1.0 (primary canonical), en.wikipedia.org=0.7 (secondary fallback after T.8 deprioritization), duckduckgo.com=0.5 (meta-source, search engine), default=0.4 (third-party / blogs / unknown).",
            "inputSchema": {"type": "object", "properties": {}}
        },
        {
            "name": "lu4d_trust_set",
            "description": "Port Cat.3a - add or update a domain trust entry. Multiplier must be in [0.0, 2.0]. Mutations persist to the JSON sidecar at <brain>.trust.json (NOT a v3 schema section). Existing bonds are NEVER mutated; trust applies only to NEW bonds at formation in the research path.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "domain": {"type": "string", "description": "Exact host to match against SourceAttribution.url (e.g. en.wikipedia.org). Case-insensitive. Required."},
                    "multiplier": {"type": "number", "description": "Multiplier in [0.0, 2.0]. 1.0 is neutral; <1.0 demotes; >1.0 amplifies. Required."},
                    "note": {"type": "string", "description": "Human-readable reason for the entry. Optional."}
                },
                "required": ["domain", "multiplier"]
            }
        },
        {
            "name": "lu4d_trust_remove",
            "description": "Port Cat.3a - remove a domain trust entry. The default multiplier still applies for that domain after removal. Persists the sidecar.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "domain": {"type": "string", "description": "Exact host to remove. Case-insensitive. Required."}
                },
                "required": ["domain"]
            }
        },
        {
            "name": "lu4d_trust_set_corpus",
            "description": "Port T-Ingest.1 - add or update a per-corpus trust multiplier used by `lu4d_teach_corpus` (and any other teach-path caller passing a TrainingCorpus source kind with trust=None). Multiplier must be in [0.0, 2.0]. Mutations affect ONLY new bond formation; existing bonds are never retroactively reweighted. Lookup is exact-match by corpus string (case-sensitive); on miss the table's default_multiplier applies. Tier 1 corpora typically map to 1.0 (wikipedia, structured-wikipedia, visual_genome, grokipedia); Tier 1.5 to 0.9 (arxiv, GQA); Tier 2 to 0.7-0.8 (no_robots, instruction-dataset, stackexchange). Use `lu4d_trust_reset_to_defaults { seed_curriculum_corpora: true }` to pre-populate the locked curriculum hierarchy in one call.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "corpus": {"type": "string", "description": "Exact corpus key (e.g. 'wikimedia/wikipedia', 'arxiv', 'stackexchange'). Case-sensitive. Required."},
                    "multiplier": {"type": "number", "description": "Multiplier in [0.0, 2.0]. 1.0 is neutral; <1.0 demotes; >1.0 amplifies. Required."}
                },
                "required": ["corpus", "multiplier"]
            }
        },
        {
            "name": "lu4d_trust_remove_corpus",
            "description": "Port T-Ingest.1 - remove a per-corpus trust entry. After removal the table's default_multiplier applies for that corpus on subsequent teach calls. Existing bonds are never mutated.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "corpus": {"type": "string", "description": "Exact corpus key to remove. Case-sensitive. Required."}
                },
                "required": ["corpus"]
            }
        },
        {
            "name": "lu4d_trust_reset_to_defaults",
            "description": "Port Cat.3a - restore the C2-aligned default trust table (grokipedia=1.0, wikipedia=0.7, duckduckgo=0.5, default=0.4, learning_enabled=true) and clear corpus_multipliers. Idempotent. Port T-Ingest.1: optional `seed_curriculum_corpora` flag (default false) pre-populates corpus_multipliers with the locked curriculum hierarchy after the reset (wikimedia/wikipedia=1.0, structured-wikipedia=1.0, visual_genome=1.0, GQA=0.9, arxiv=0.9, stackexchange=0.8, code-search-net/code_search_net=0.9, starcoderdata=0.9, stackoverflow=0.8, python-docs=0.95, grokipedia=1.0, no_robots=0.7, instruction-dataset=0.7).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "seed_curriculum_corpora": {"type": "boolean", "description": "When true, pre-populate corpus_multipliers with the locked curriculum hierarchy. Default false."}
                }
            }
        },
        {
            "name": "lu4d_hub_status",
            "description": "Port Cat.3b - read the hub-degree config + top-N hubs by current in_degree. Surfaces config (max_in_degree, max_out_degree, write/read enforcement, priority, exempt_node_kinds), counters (rejections, evictions, filtered_at_query), and bounds. Default cap 5000 in/out degree per node. Existing bonds NEVER mutated; caps apply to NEW bonds at write time and to neighbor lists at query time.",
            "inputSchema": {"type": "object", "properties": {"top_n": {"type": "integer", "description": "Number of top-degree hubs to return. Default 20, max 200."}}}
        },
        {
            "name": "lu4d_hub_set",
            "description": "Port Cat.3b - update one hub-degree-config field. Keys: max_in_degree, max_out_degree (each [100, 1000000]); write_enforcement, read_enforcement (bool); priority (NewestFirst/OldestFirst/LowestWeightFirst); exempt_node_kinds (array of strings). Mutations persist to <brain>.hub_degree.json sidecar (NOT a v3 schema section).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "Field to update. Required."},
                    "value": {"description": "New value (type depends on key). Required."}
                },
                "required": ["key", "value"]
            }
        },
        {
            "name": "lu4d_hub_reset_to_defaults",
            "description": "Port Cat.3b - restore default hub-degree config (max_in=max_out=5000, write/read=true, priority=LowestWeightFirst, exempt_node_kinds=[]). Idempotent.",
            "inputSchema": {"type": "object", "properties": {}}
        },
        {
            "name": "lu4d_hub_inspect",
            "description": "Port Cat.3b - degree info for a specific node by text. Returns in/out counts, whether the node would be filtered at query time, and the top-N outgoing bonds by latest weight. Read-only.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "node_text": {"type": "string", "description": "Node text to look up. Required."},
                    "top_n": {"type": "integer", "description": "Number of top-weight outgoing bonds to return. Default 10, max 200."}
                },
                "required": ["node_text"]
            }
        },
        {
            "name": "lu4d_session_memory_status",
            "description": "Port LB - Left Brain session memory counts: total entries, active session, archived sessions, counts by session_id and kind. The Left Brain layer holds operational/episodic memory (thinking traces, observations, hypotheses) that informs reasoning but never auto-promotes to fact bonds.",
            "inputSchema": {"type": "object", "properties": {}}
        },
        {
            "name": "lu4d_session_memory_inspect",
            "description": "Port LB - list Left Brain session entries, optionally filtered by session_id and/or kind. Read-only; reads are unrestricted (any caller can read entries from any session).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "session_id": {"type": "integer", "description": "Restrict to entries written under this session id."},
                    "kind": {"type": "string", "description": "One of: thinking_trace, observation, hypothesis, goal, wonder_item, research_result, user_statement."},
                    "limit": {"type": "integer", "description": "Maximum number of entries to return. Default 50."}
                }
            }
        },
        {
            "name": "lu4d_session_memory_propose",
            "description": "Port LB - convert a Left Brain session entry into a T.3 PendingChange::AddBond. The canonical brain is NOT mutated until a human approves the pending change via lu4d_approve_pending. This is the explicit propose/approve gate that prevents non-facts from becoming facts.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "entry_id": {"type": "integer", "description": "id of the session entry to promote."},
                    "bond_type": {"type": "string", "description": "BondType for the proposed bond. Default RelatedTo."},
                    "weight": {"type": "number", "description": "Bond weight 0.0-1.0. Default 0.6."},
                    "subject_text": {"type": "string", "description": "Optional override for subject text. Defaults to the entry subject."},
                    "object_text": {"type": "string", "description": "Optional override for object text. Defaults to the entry content (or subject when content is empty)."}
                },
                "required": ["entry_id"]
            }
        },
        {
            "name": "lu4d_session_memory_archive",
            "description": "Port LB - mark a session as archived (immutable). Subsequent writes scoped to that session id are rejected, and re-activating it errors. Reads still work — archived sessions remain visible to lu4d_session_memory_inspect.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "session_id": {"type": "integer", "description": "id of the session to archive."}
                },
                "required": ["session_id"]
            }
        },
        {
            "name": "lu4d_style_status",
            "description": "Port CS — return persisted + decay-effective conversational-style vector for a user_key (default \"default\").",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "user_key": {"type": "string"},
                    "now_ms": {"type": "integer", "minimum": 0}
                }
            }
        },
        {
            "name": "lu4d_style_reset",
            "description": "Port CS — clear the persisted conversational-style profile for a user_key (back to defaults).",
            "inputSchema": {
                "type": "object",
                "properties": {"user_key": {"type": "string"}}
            }
        },
        {
            "name": "lu4d_style_set",
            "description": "Port CS — explicit override of one or more conversational-style dimensions in [0.0, 1.0].",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "user_key": {"type": "string"},
                    "verbosity": {"type": "number", "minimum": 0, "maximum": 1},
                    "formality": {"type": "number", "minimum": 0, "maximum": 1},
                    "technicality": {"type": "number", "minimum": 0, "maximum": 1},
                    "empathy": {"type": "number", "minimum": 0, "maximum": 1},
                    "now_ms": {"type": "integer", "minimum": 0}
                }
            }
        },
        {
            "name": "lu4d_pack_install",
            "description": "Port Cat.4 — verify a .lupack signature against trusted publishers, decrypt content, and queue every bond as a PendingChange in T.3 (lu4d_approve_pending to merge).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "pack_path": {"type": "string"},
                    "now_ms": {"type": "integer", "minimum": 0}
                },
                "required": ["pack_path"]
            }
        },
        {
            "name": "lu4d_pack_uninstall",
            "description": "Port Cat.4 — revoke every bond installed by a pack via revoke_by_source on the synthetic Teach actor 'pack_install:<pack_id>'.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "pack_id": {"type": "string"},
                    "reason": {"type": "string"},
                    "dry_run": {"type": "boolean"},
                    "now_ms": {"type": "integer", "minimum": 0}
                },
                "required": ["pack_id"]
            }
        },
        {
            "name": "lu4d_pack_list",
            "description": "Port Cat.4 — list trusted publishers, installed packs, and revoked packs.",
            "inputSchema": {"type": "object", "properties": {}}
        },
        {
            "name": "lu4d_pack_publisher_add",
            "description": "Port Cat.4 — add a trusted publisher (64-char hex Ed25519 pubkey + human-readable name).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "pubkey_hex": {"type": "string"},
                    "name": {"type": "string"},
                    "notes": {"type": "string"},
                    "now_ms": {"type": "integer", "minimum": 0}
                },
                "required": ["pubkey_hex", "name"]
            }
        },
        {
            "name": "lu4d_pack_publisher_remove",
            "description": "Port Cat.4 — remove a trusted publisher from the ledger by pubkey.",
            "inputSchema": {
                "type": "object",
                "properties": {"pubkey_hex": {"type": "string"}},
                "required": ["pubkey_hex"]
            }
        },
        {
            "name": "lu4d_pack_create",
            "description": "Port Cat.4 — admin/producer tool. Sign a new .lupack from selected brain bonds using a 32-byte Ed25519 secret keyfile.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "pack_id": {"type": "string"},
                    "output_path": {"type": "string"},
                    "keyfile": {"type": "string"},
                    "pack_version": {"type": "string"},
                    "content_summary": {"type": "string"},
                    "license": {"type": "string"},
                    "minimum_lu_version": {"type": "string"},
                    "bond_limit": {"type": "integer", "minimum": 0},
                    "bond_type": {"type": "string"},
                    "now_ms": {"type": "integer", "minimum": 0}
                },
                "required": ["pack_id", "output_path", "keyfile"]
            }
        },
        {
            "name": "lu4d_capability_pack_install",
            "description": "Port Cat.5 — verify a .lupack5 signature against trusted capability publishers, refuse on revoked pack_id, refuse on already-installed (unless force=true), and record in the capability-pack ledger. SCHEMA-ONLY: does NOT execute the pack content; runtime loading lands in Cat.5.1.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "force": {"type": "boolean"},
                    "now_ms": {"type": "integer", "minimum": 0}
                },
                "required": ["path"]
            }
        },
        {
            "name": "lu4d_capability_pack_uninstall",
            "description": "Port Cat.5 — remove an installed capability pack from the ledger. Optionally pass revoke=true to add the pack_id to revoked_packs so it can no longer be re-installed.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "pack_id": {"type": "string"},
                    "revoke": {"type": "boolean"},
                    "now_ms": {"type": "integer", "minimum": 0}
                },
                "required": ["pack_id"]
            }
        },
        {
            "name": "lu4d_capability_pack_list",
            "description": "Port Cat.5 — list trusted capability publishers, installed capability packs, and revoked capability pack_ids.",
            "inputSchema": {"type": "object", "properties": {}}
        },
        {
            "name": "lu4d_capability_pack_publisher_add",
            "description": "Port Cat.5 — add a trusted capability publisher (64-char hex Ed25519 pubkey + label). SEPARATE from Cat.4's content-pack publisher list: trusting executable code is a different decision than trusting curated knowledge bundles.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "pubkey_hex": {"type": "string"},
                    "label": {"type": "string"},
                    "notes": {"type": "string"},
                    "now_ms": {"type": "integer", "minimum": 0}
                },
                "required": ["pubkey_hex", "label"]
            }
        },
        {
            "name": "lu4d_capability_pack_publisher_remove",
            "description": "Port Cat.5 — remove a trusted capability publisher by pubkey. Existing installs from that publisher continue; new installs are refused.",
            "inputSchema": {
                "type": "object",
                "properties": {"pubkey_hex": {"type": "string"}},
                "required": ["pubkey_hex"]
            }
        },
        {
            "name": "lu4d_capability_pack_create",
            "description": "Port Cat.5 — admin/producer tool. Sign a new .lupack5 from a manifest JSON file + opaque content payload using a 32-byte Ed25519 secret keyfile. Use --gen-capability-pack-keypair (CLI) to make the keypair.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "manifest_path": {"type": "string"},
                    "content_path": {"type": "string"},
                    "output_path": {"type": "string"},
                    "signing_keyfile": {"type": "string"},
                    "now_ms": {"type": "integer", "minimum": 0}
                },
                "required": ["manifest_path", "content_path", "output_path", "signing_keyfile"]
            }
        },
        // ---- Port BR.2: per-port RuntimeConfig setters + batch dispatcher ----
        {
            "name": "lu4d_sense_set",
            "description": "Port BR.2 - set runtime sense disambiguation knobs. Keys: sense_disambiguation_enabled (bool), sense_context_window (1..=4096), sense_loser_gain (0.0..=1.0). Pass with no args to read the current effective config. Mutations land on Brain.sense_config and persist via runtime_config section #15 on next save.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "sense_disambiguation_enabled": {"type": "boolean", "description": "Master switch for sense disambiguation."},
                    "sense_context_window": {"type": "integer", "description": "Context window size for sense scoring (1..=4096)."},
                    "sense_loser_gain": {"type": "number", "description": "Multiplier applied to non-winning sense clusters (0.0..=1.0)."}
                }
            }
        },
        {
            "name": "lu4d_chained_research_set",
            "description": "Port BR.2 - set chained-curiosity research knobs (Port J.2). Keys: chained_research_enabled (bool), max_chain_depth (1..=64), max_chain_cycles (1..=1024), chain_gap_threshold (0.0..=1.0). Mutations land on Brain.chain_research_config and persist via runtime_config section #15.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "chained_research_enabled": {"type": "boolean"},
                    "max_chain_depth": {"type": "integer"},
                    "max_chain_cycles": {"type": "integer"},
                    "chain_gap_threshold": {"type": "number"}
                }
            }
        },
        {
            "name": "lu4d_image_set",
            "description": "Port BR.2 - set image / vision ingest knobs (Port R input). Keys: image_extractor_kind ('stub' | 'subprocess' | {capability_pack: <id>}), image_extractor_subprocess_path (string | null), image_extractor_subprocess_timeout_ms (100..=600000), image_max_size_bytes (1024..=1073741824), image_supported_mime_types (array of strings). Mutations land on Brain.vision_config.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "image_extractor_kind": {"description": "'stub' | 'subprocess' | {capability_pack: id}"},
                    "image_extractor_subprocess_path": {"type": ["string", "null"]},
                    "image_extractor_subprocess_timeout_ms": {"type": "integer"},
                    "image_max_size_bytes": {"type": "integer"},
                    "image_supported_mime_types": {"type": "array", "items": {"type": "string"}}
                }
            }
        },
        {
            "name": "lu4d_file_io_set",
            "description": "Port BR.2 - set file-IO ingest / emission knobs (Port File-IO). Keys: file_io_max_size_bytes (1024..=2147483648), file_io_allowed_read_paths (array of dir prefixes; empty = unrestricted), file_io_allowed_write_paths (array of dir prefixes). Mutations land on Brain.file_io_config.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "file_io_max_size_bytes": {"type": "integer"},
                    "file_io_allowed_read_paths": {"type": "array", "items": {"type": "string"}},
                    "file_io_allowed_write_paths": {"type": "array", "items": {"type": "string"}}
                }
            }
        },
        {
            "name": "lu4d_ingest_set",
            "description": "Port BR.2 - set training-corpus ingest + Tier 2 prompt routing knobs (Ports T-Ingest and T-Ingest.2). Keys: ingest_corpus_default (string | null), ingest_run_id_strategy ('caller' | 'server_startup' | 'per_session'), ingest_auto_approve_threshold (number 0..=1 | null), tier2_prompt_default_session_strategy ('ephemeral' | 'reuse'), tier2_prompt_max_text_bytes (256..=16777216). Tier-2 fields are grouped here because they live under Port T-Ingest.2 in runtime_config and share the ingest configuration surface.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "ingest_corpus_default": {"type": ["string", "null"]},
                    "ingest_run_id_strategy": {"type": "string", "enum": ["caller", "server_startup", "per_session"]},
                    "ingest_auto_approve_threshold": {"type": ["number", "null"]},
                    "tier2_prompt_default_session_strategy": {"type": "string", "enum": ["ephemeral", "reuse"]},
                    "tier2_prompt_max_text_bytes": {"type": "integer"}
                }
            }
        },
        {
            "name": "lu4d_recall_set",
            "description": "Port BR.4 - recall-time typed-KU runtime kill switch. Keys: recall_typed_ku_enabled (bool, default true). When true the chain composer + procedure assembler in reasoning.rs feed endpoint KnowledgeContent kinds into articulate_bond_typed (Q.1 behavior); when false the leg builders pass kinds through as None so articulation collapses to the pre-Q.1 baseline. Storage / hub-cap / decay paths are unaffected. Mutations land on Brain.recall_config and persist via runtime_config section #15 on next save.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "recall_typed_ku_enabled": {"type": "boolean", "description": "Master switch for Q.1 typed-KU activation in recall articulation."}
                }
            }
        },
        {
            "name": "lu4d_admin_set",
            "description": "Port BR.5 - admin-tier runtime gates. Keys: admin_revoke_enabled (bool, default false). When false (default) lu4d_revoke_by_source with dry_run=false returns -32603; flip to true to open the destructive revoke path for this process. Persisted via runtime_config section #15 on next save so the gate state survives restarts. Pass with no args to read the current effective config. Default-deny posture: every brain ships with admin_revoke_enabled=false and operators must explicitly open the gate.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "admin_revoke_enabled": {"type": "boolean", "description": "Master switch for the lu4d_revoke_by_source destructive path. Default false."}
                }
            }
        },
        {
            "name": "lu4d_configure_batch",
            "description": "Port BR.2 - dispatch a heterogeneous list of {tool, args} setter calls in best-effort sequential order. NOT transactional: a failed entry leaves earlier successful entries applied and continues with later ones. Returns {total, applied, failed, results: [{index, tool, ok, structuredContent | error}]}. Allowed tools: lu4d_sense_set, lu4d_chained_research_set, lu4d_image_set, lu4d_file_io_set, lu4d_ingest_set, lu4d_recall_set, lu4d_admin_set, lu4d_configure.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "calls": {
                        "type": "array",
                        "description": "Non-empty array of {tool: string, args: object} entries.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "tool": {"type": "string"},
                                "args": {"type": "object"}
                            },
                            "required": ["tool"]
                        }
                    }
                },
                "required": ["calls"]
            }
        }

    ])
}

fn handle_tools_call(brain: &mut Brain, params: Value) -> Result<Value, (i64, String)> {
    let name = params.get("name").and_then(|v| v.as_str()).ok_or((-32602, "missing tool name".into()))?;
    let args = params.get("arguments").cloned().unwrap_or(Value::Object(Default::default()));

    // Port T.2: every MCP tool runs under an idle-state QueryGuard
    // so the brain knows when work is in flight. lu4d_idle_tick is
    // the single exception — it manages its own lifecycle (gate
    // checks idleness BEFORE doing work, so an outer guard would
    // always trip the gate). The guard is dropped at end-of-scope
    // including on early-return / panic paths from the dispatch.
    let _guard = if name != "lu4d_idle_tick" {
        Some(crate::idle::QueryGuard::new(&brain.idle_state))
    } else {
        None
    };

    let res = handle_tools_call_inner(brain, name, args);
    // Port J / curiosity retrofit: drain the gap_emitter into
    // the wonder queue at every tool boundary, so reasoning
    // gaps surfaced during the call become visible to
    // lu4d_wonder and the autonomous research loop on the
    // very next call.
    let _flushed = brain.flush_gaps();
    res
}

fn handle_tools_call_inner(brain: &mut Brain, name: &str, args: Value) -> Result<Value, (i64, String)> {
    match name {
        "lu4d_teach" => tool_teach(brain, args),
        "lu4d_teach_corpus" => tool_teach_corpus(brain, args),
        "lu4d_teach_corpus_batch" => tool_teach_corpus_batch(brain, args),
        "lu4d_user_statement" => tool_user_statement(brain, args),
        "lu4d_file_read" => crate::mcp_file_io::tool_file_read(brain, args),
        "lu4d_file_write" => crate::mcp_file_io::tool_file_write(brain, args),
        "lu4d_file_extract_to_pending" => crate::mcp_file_io::tool_file_extract_to_pending(brain, args),
        "lu4d_learn_text" => crate::mcp_extractor::tool_learn_text(brain, args),
        "lu4d_image_ingest" => crate::mcp_vision::tool_image_ingest(brain, args),
        "lu4d_image_describe" => crate::mcp_vision::tool_image_describe(brain, args),
        "lu4d_image_extractor_status" => crate::mcp_vision::tool_image_extractor_status(brain, args),
        "lu4d_set_definition_batch" => tool_set_definition_batch(brain, args),
        "lu4d_ask" => tool_ask(brain, args),
        "lu4d_status" => tool_status(brain),
        "lu4d_time_status" => crate::mcp_t11::tool_time_status(brain),
        "lu4d_sense_status" => crate::mcp_sense::tool_sense_status(brain, args),
        "lu4d_session_start" => tool_session_start(brain),
        "lu4d_session_end" => tool_session_end(brain),
        "lu4d_timetravel" => tool_timetravel(brain, args),
        "lu4d_consolidate" => tool_consolidate(brain),
        "lu4d_purge_stopwords" => tool_purge_stopwords(brain),
        "lu4d_prune_legacy_goals" => tool_prune_legacy_goals(brain),
        "lu4d_think" => tool_think(brain, args),
        "lu4d_wonder" => tool_wonder(brain, args),
        "lu4d_save" => tool_save(brain, args),
        "lu4d_save_status" => tool_save_status(brain),
        "lu4d_flush" => tool_flush(brain, args),
        "lu4d_variants_generate" => tool_variants_generate(brain, args),
        "lu4d_variants_promote" => tool_variants_promote(brain, args),
        "lu4d_reflect" => tool_reflect(brain),
        "lu4d_proposals_list" => tool_proposals_list(brain, args),
        "lu4d_proposal_decide" => tool_proposal_decide(brain, args),
        "lu4d_multi_think" => tool_multi_think(brain, args),
        "lu4d_narrate" => tool_narrate(brain, args),
        "lu4d_scar_bond" => tool_scar_bond(brain, args),
        "lu4d_scar_concept" => tool_scar_concept(brain, args),
        "lu4d_scar_log" => tool_scar_log(brain, args),
        "lu4d_unscar_bond" => tool_unscar_bond(brain, args),
        "lu4d_unscar_concept" => tool_unscar_concept(brain, args),
        "lu4d_research" => tool_research(brain, args),
        "lu4d_research_loop_run" => tool_research_loop_run(brain, args),
        "lu4d_research_chain_run" => tool_research_chain_run(brain, args),
        "lu4d_research_proposals_list" => tool_research_proposals_list(brain, args),
        "lu4d_research_proposal_decide" => tool_research_proposal_decide(brain, args),
        "lu4d_reason" => tool_reason(brain, args),
        "lu4d_revoke_by_source" => tool_revoke_by_source(brain, args),
        "lu4d_source_attribution_migrate" => tool_source_attribution_migrate(brain, args),
        "lu4d_idle_tick" => tool_idle_tick(brain, args),
        "lu4d_pending_changes" => crate::mcp_t3::tool_pending_changes(brain, args),
        "lu4d_approve_pending" => crate::mcp_t3::tool_approve_pending(brain, args),
        "lu4d_reject_pending" => crate::mcp_t3::tool_reject_pending(brain, args),
        "lu4d_modify_pending" => crate::mcp_t3::tool_modify_pending(brain, args),
        "lu4d_pending_status" => crate::mcp_t3::tool_pending_status(brain),
        "lu4d_freshness_status" => crate::mcp_t5::tool_freshness_status(brain, args),
        "lu4d_configure" => crate::mcp_t5::tool_configure(brain, args),
        "lu4d_auto_tick_status" => crate::mcp_t6::tool_auto_tick_status(brain),
        "lu4d_auto_tick_pause" => crate::mcp_t6::tool_auto_tick_pause(brain),
        "lu4d_auto_tick_resume" => crate::mcp_t6::tool_auto_tick_resume(brain),
        "lu4d_rejection_patterns" => crate::mcp_t8::tool_rejection_patterns(brain, args),
        // Port LB — Left Brain session memory.
        "lu4d_session_memory_status" => crate::mcp_lb::tool_session_memory_status(brain, args),
        "lu4d_session_memory_inspect" => crate::mcp_lb::tool_session_memory_inspect(brain, args),
        "lu4d_session_memory_propose" => crate::mcp_lb::tool_session_memory_propose(brain, args),
        "lu4d_session_memory_archive" => crate::mcp_lb::tool_session_memory_archive(brain, args),
        // Port CS — conversational style profile.
        "lu4d_style_status" => crate::mcp_cs::tool_style_status(brain, args),
        "lu4d_style_reset"  => crate::mcp_cs::tool_style_reset(brain, args),
        "lu4d_style_set"    => crate::mcp_cs::tool_style_set(brain, args),
        // Port Cat.4 — signed safety packs.
        "lu4d_pack_install"           => crate::mcp_pack::tool_pack_install(brain, args),
        "lu4d_pack_uninstall"         => crate::mcp_pack::tool_pack_uninstall(brain, args),
        "lu4d_pack_list"              => crate::mcp_pack::tool_pack_list(brain, args),
        "lu4d_pack_publisher_add"     => crate::mcp_pack::tool_pack_publisher_add(brain, args),
        "lu4d_pack_publisher_remove"  => crate::mcp_pack::tool_pack_publisher_remove(brain, args),
        "lu4d_pack_create"            => crate::mcp_pack::tool_pack_create(brain, args),
        "lu4d_capability_pack_install"           => crate::mcp_cat5::tool_capability_pack_install(brain, args),
        "lu4d_capability_pack_uninstall"         => crate::mcp_cat5::tool_capability_pack_uninstall(brain, args),
        "lu4d_capability_pack_list"              => crate::mcp_cat5::tool_capability_pack_list(brain, args),
        "lu4d_capability_pack_publisher_add"     => crate::mcp_cat5::tool_capability_pack_publisher_add(brain, args),
        "lu4d_capability_pack_publisher_remove"  => crate::mcp_cat5::tool_capability_pack_publisher_remove(brain, args),
        "lu4d_capability_pack_create"            => crate::mcp_cat5::tool_capability_pack_create(brain, args),
        "lu4d_capability_pack_runtime_configure" => crate::mcp_cap_runtime::tool_runtime_configure(brain, args),
        "lu4d_capability_pack_runtime_status"    => crate::mcp_cap_runtime::tool_runtime_status(brain, args),
        "lu4d_capability_pack_load"              => crate::mcp_cap_runtime::tool_pack_load(brain, args),
        "lu4d_capability_pack_unload"            => crate::mcp_cap_runtime::tool_pack_unload(brain, args),
        // Port Cat.3a: source-trust weighting MCP tools.
        "lu4d_trust_status"            => crate::mcp_cat3a::tool_trust_status(brain, args),
        "lu4d_trust_set"               => crate::mcp_cat3a::tool_trust_set(brain, args),
        "lu4d_trust_remove"            => crate::mcp_cat3a::tool_trust_remove(brain, args),
        "lu4d_trust_reset_to_defaults" => crate::mcp_cat3a::tool_trust_reset_to_defaults(brain, args),
        "lu4d_trust_set_corpus"        => crate::mcp_cat3a::tool_trust_set_corpus(brain, args),
        "lu4d_trust_remove_corpus"     => crate::mcp_cat3a::tool_trust_remove_corpus(brain, args),
        "lu4d_hub_status"              => crate::mcp_cat3b::tool_hub_status(brain, args),
        "lu4d_hub_set"                 => {
            let r = crate::mcp_cat3b::tool_hub_set(brain, args);
            // After a successful set, sync the read-side knobs to
            // the network so spread paths immediately see the new
            // cap. Failure to set is surfaced to the caller.
            if r.is_ok() { brain.resync_hub_read_config(); }
            r
        }
        "lu4d_hub_reset_to_defaults"   => {
            let r = crate::mcp_cat3b::tool_hub_reset_to_defaults(brain, args);
            if r.is_ok() { brain.resync_hub_read_config(); }
            r
        }
        "lu4d_hub_inspect"             => crate::mcp_cat3b::tool_hub_inspect(brain, args),
        // ---- Port BR.2: per-port RuntimeConfig setters + batch dispatcher ----
        "lu4d_sense_set"               => crate::mcp_runtime_setters::tool_sense_set(brain, args),
        "lu4d_chained_research_set"    => crate::mcp_runtime_setters::tool_chained_research_set(brain, args),
        "lu4d_image_set"               => crate::mcp_runtime_setters::tool_image_set(brain, args),
        "lu4d_file_io_set"             => crate::mcp_runtime_setters::tool_file_io_set(brain, args),
        "lu4d_ingest_set"              => crate::mcp_runtime_setters::tool_ingest_set(brain, args),
        "lu4d_recall_set"              => crate::mcp_runtime_setters::tool_recall_set(brain, args),
        "lu4d_admin_set"               => crate::mcp_runtime_setters::tool_admin_set(brain, args),
        "lu4d_configure_batch"         => crate::mcp_runtime_setters::tool_configure_batch(brain, args),
        other => {
            if brain.pack_registry.has_tool(other) {
                crate::mcp_cap_runtime::dispatch_pack_tool(brain, other, args)
            } else {
                Err((-32602, format!("unknown tool: {}", other)))
            }
        }
    }
}

// wrap_content moved to crate::mcp_envelope (OP-007: admin mcp module
// is feature-gated; the envelope helper must stay always-compiled).

// // PORT F WIRING — Port F (#576): math intent detector + arithmetic primitive.
// // PORT F.2 WIRING — Port F.2 expands the surface to a full expression
// parser, scientific functions, named constants, single-step unit
// conversions, descriptive statistics, English word-form operands,
// and a one-unknown linear solver.
//
// `try_arithmetic_response` now tries the legacy F.1 single-op path
// FIRST (so the 28 existing math tests stay verbatim) and then falls
// through to `crate::math::detect_math` for any F.2 capability. The
// structured payload distinguishes the two routes via the `source`
// field — `"arithmetic_primitive"` (legacy) vs `"expression_primitive"`
// (F.2). Both still mark the answer as computed-not-recalled.
// Test-only convenience wrapper: production callers use
// try_arithmetic_response_with(Some(brain), query) directly. The
// no-brain wrapper exists so the F.x cascade tests can exercise the
// detection logic without constructing a Brain. Gated to test builds
// to keep release compilation warning-free.
#[cfg(test)]
fn try_arithmetic_response(query: &str) -> Option<Value> {
    try_arithmetic_response_with(None, query)
}

fn try_arithmetic_response_with(brain: Option<&Brain>, query: &str) -> Option<Value> {
    // -- Legacy F.1 single-op path -----------------------------------
    if let Some(intent) = crate::math::detect_arithmetic(query) {
        return Some(match intent.compute() {
            Ok(result) => {
                let text = format!(
                    "{}\n\nProvenance: computed by Lu's arithmetic primitive, not recalled.",
                    result.prose
                );
                let structured = json!({
                    "source": "arithmetic_primitive",
                    "intent": result.intent,
                    "value": result.value,
                    "prose": result.prose,
                    "query": query,
                });
                wrap_content(text, structured)
            }
            Err(e) => {
                let text = format!(
                    "Cannot compute: {}.\n\nProvenance: routed to Lu's arithmetic primitive but computation failed.",
                    e
                );
                let structured = json!({
                    "source": "arithmetic_primitive",
                    "intent": intent,
                    "error": e,
                    "query": query,
                });
                wrap_content(text, structured)
            }
        });
    }

    // -- F.4 construction-calculator path (the project log Port F.4) -------
    // Construction-shaped queries (feet-inches tick marks, lumber/framing/
    // concrete vocabulary) fire BEFORE F.2 so a contractor's phrasing
    // gets first crack at construction-mode output. If the F.4 detector
    // returns None, we fall through to the F.2 expression parser as if
    // F.4 weren't there.
    if let Some(outcome) = crate::math::construction::detect_construction(query) {
        let prose_full = format!(
            "{}\n\nProvenance: computed by Lu's construction calculator, not recalled.",
            outcome.prose()
        );
        let outcome_json = serde_json::to_value(&outcome).unwrap_or(serde_json::Value::Null);
        let structured = json!({
            "source": "construction_primitive",
            "outcome": outcome_json,
            "query": query,
        });
        return Some(wrap_content(prose_full, structured));
    }


    // -- F.3 word-problem path (the project log Port F.3) ------------------
    // Prose-shaped queries with numbers + operation cues + a question
    // get an extraction pass before the F.2 expression parser would
    // see them (the parser fails on "Sarah had 5 apples"). When the
    // extraction is shaky, F.3 returns an honest hedge instead of a
    // fabricated answer — Lu's structural advantage is provenance.
    if let Some(outcome) = crate::math::wordprob::detect_word_problem(query) {
        // Port F.3 retrofit (Port J / curiosity): when extraction
        // produced a Hedge (no confident answer), emit a gap signal.
        if outcome.is_hedged() {
            if let Some(b) = brain {
                b.gap_emitter.emit(crate::gap_signal::GapSignal {
                    query: query.to_string(),
                    port_origin: "F.3",
                    gap_kind: crate::gap_signal::GapKind::WordProblemUnparsable,
                    severity: 0.5,
                    context: serde_json::json!({
                        "wordprob_outcome": "hedge",
                    }),
                });
            }
        }
        let prose_full = format!(
            "{}\n\nProvenance: routed through Lu's word-problem extractor.",
            outcome.prose()
        );
        let outcome_json = serde_json::to_value(&outcome).unwrap_or(serde_json::Value::Null);
        let confidence_str = match &outcome {
            crate::math::wordprob::WordProblemOutcome::Computed {
                confidence: crate::math::wordprob::WordConfidence::Confident, ..
            } => "confident",
            _ => "hedged",
        };
        let structured = json!({
            "source": "wordprob_primitive",
            "confidence": confidence_str,
            "outcome": outcome_json,
            "query": query,
        });
        return Some(wrap_content(prose_full, structured));
    }

    // -- F.2 unified path: expressions, conversions, stats, solver --
    let outcome = crate::math::detect_math(query)?;
    // OP-MATH-HYPHEN-GUARD (VERIFY-FIRST 2026-08-03): a hyphenated concept
    // term ("half-life") parses as `a - b` and fails the expression evaluator
    // with "unknown identifier"; that is not arithmetic. When the phrase
    // resolves to a real node, defer to recall instead of emitting a spurious
    // math error. Genuine arithmetic ("5 - 3") is numeric and never yields an
    // unknown-identifier error, and unknown phrases keep the honest math error
    // (recall would refuse anyway), so both the arithmetic path and the refusal
    // floor are untouched. Test callers pass brain=None -> the guard is inert.
    if let crate::math::MathOutcome::Error { route, error, .. } = &outcome {
        if route == "expression"
            && error.starts_with("unknown identifier")
            && brain.map_or(false, |b| math_hyphen_defers_to_recall(b, query))
        {
            return None;
        }
    }
    let prose_full = format!(
        "{}\n\nProvenance: computed by Lu's expression primitive, not recalled.",
        outcome.prose()
    );
    let outcome_json = serde_json::to_value(&outcome).unwrap_or(serde_json::Value::Null);
    let structured = json!({
        "source": "expression_primitive",
        "outcome": outcome_json,
        "query": query,
    });
    Some(wrap_content(prose_full, structured))
}

/// OP-MATH-HYPHEN-GUARD helper: true when a hyphenated word phrase the
/// expression evaluator rejected ("unknown identifier") actually names a node
/// the recall path can answer. Mirrors the diagnostics' existence test: the
/// query (definitional prefix stripped) as-is and with hyphens turned to
/// spaces, raw and canonical-folded, against the interned node set. No fuzzy
/// matching -- unknown phrases fall through to the honest math error.
fn math_hyphen_defers_to_recall(brain: &Brain, query: &str) -> bool {
    let ql = query.trim().to_lowercase();
    let core = ql
        .strip_prefix("what is the value of ")
        .or_else(|| ql.strip_prefix("what is the "))
        .or_else(|| ql.strip_prefix("what is "))
        .or_else(|| ql.strip_prefix("what's "))
        .or_else(|| ql.strip_prefix("what are "))
        .or_else(|| ql.strip_prefix("define "))
        .or_else(|| ql.strip_prefix("compute "))
        .or_else(|| ql.strip_prefix("calculate "))
        .or_else(|| ql.strip_prefix("evaluate "))
        .unwrap_or(&ql)
        .trim_end_matches(|c: char| matches!(c, '.' | '?' | '!'))
        .trim();
    if !core.contains('-') {
        return false;
    }
    for cand in [core.to_string(), core.replace('-', " ")] {
        let raw = crate::seed::Seed::from_label(&cand);
        if brain.network.node_intern().get(&raw).is_some() {
            return true;
        }
        let folded =
            crate::seed::Seed::from_label(&crate::canonical::fold_seed_label(&cand));
        if brain.network.node_intern().get(&folded).is_some() {
            return true;
        }
    }
    false
}

fn arg_str<'a>(args: &'a Value, key: &str) -> Result<&'a str, (i64, String)> {
    args.get(key).and_then(|v| v.as_str()).ok_or((-32602, format!("missing string arg '{}'", key)))
}

fn tool_teach(brain: &mut Brain, args: Value) -> Result<Value, (i64, String)> {
    let subject = arg_str(&args, "subject")?.to_string();
    let relation_str = arg_str(&args, "relation")?.to_string();
    let object = arg_str(&args, "object")?.to_string();
    let weight = args.get("weight").and_then(|v| v.as_f64()).unwrap_or(0.8) as f32;
    let confidence_str = args.get("confidence").and_then(|v| v.as_str()).unwrap_or("Verified");
    // Q.1: optional explicit kind tag. Validated up-front so a typo
    // surfaces as a 4xx-style invalid-params error rather than
    // silently falling back to Concept.
    let kind_arg: Option<String> = args.get("kind").and_then(|v| v.as_str()).map(|s| s.to_string());
    if let Some(ref k) = kind_arg {
        if crate::node::KnowledgeContent::from_kind_str(k, String::new()).is_none() {
            return Err((
                -32602,
                format!(
                    "unknown kind: {:?}. Allowed: Fact, Concept, Procedure, Episode, Preference, Raw, Code, Image, Definition, Mechanism, Composition, Purpose, Contrast, Causal, Example, Meta",
                    k
                ),
            ));
        }
    }

    let relation: BondType = parse_bond_type(&relation_str).map_err(|e| (-32602, e.to_string()))?;
    let confidence: Confidence = parse_confidence(confidence_str).map_err(|e| (-32602, e.to_string()))?;
    // OP-EXTRACT-PREP: optional negation flag. Default false preserves
    // pre-OP behaviour bit-for-bit. When true we still route through
    // teach_with_kind (so the kind hint participates in node creation)
    // and then set the NegationOverlay entry; this mirrors the gated
    // path at pending::apply::apply_change.
    let negated = args.get("negated").and_then(|v| v.as_bool()).unwrap_or(false);

    let result = brain.teach_with_kind(
        &subject,
        relation,
        &object,
        weight,
        confidence,
        kind_arg.as_deref(),
        Some(relation_str.as_str()),
    );
    if negated {
        brain.set_bond_negated(result.subject_seed, result.object_seed, relation, true);
    }

    // OP-TEACH-INDEX (Claude-1 gate #1538 row L): index any newly-created
    // node so the interactive teach->recall loop closes in-session on the
    // v5/MapRAM serving brain (teach persists via node_store and skips the
    // add_node keyword indexer). Only newly-created nodes are indexed;
    // pre-existing endpoints are already in the index. Scoped to this
    // interactive tool so bulk corpus ingest memory profile is unchanged.
    if result.created_subject {
        brain.network.index_taught_label(result.subject_seed, &subject);
    }
    if result.created_object {
        brain.network.index_taught_label(result.object_seed, &object);
    }

    // Q.1: surface the resolved kind tag in both prose and structured
    // output so MCP clients can introspect what variant the new (or
    // pre-existing) nodes carry. We re-derive the kind by reading the
    // subject node back; for an existing node this reflects what was
    // already there, not the requested kind.
    let resolved_subject_kind: Option<&'static str> = brain
        .network
        .get_node(&result.subject_seed)
        .map(|n| n.kind_str());
    let resolved_object_kind: Option<&'static str> = brain
        .network
        .get_node(&result.object_seed)
        .map(|n| n.kind_str());

    let kind_tag = match (resolved_subject_kind, resolved_object_kind) {
        (Some(s), Some(o)) if s == o => format!(" kind={}", s),
        (Some(s), Some(o)) => format!(" kind=({}, {})", s, o),
        (Some(s), None) => format!(" kind={}", s),
        (None, Some(o)) => format!(" kind={}", o),
        (None, None) => String::new(),
    };

    let text = format!(
        "Taught: {} -[{:?}]-> {} (weight={:.2}, conf={:?}){}{}{}{}",
        subject, relation, object, result.weight, confidence,
        if result.created_subject { " [new subject]" } else { "" },
        if result.created_object { " [new object]" } else { "" },
        if result.reinforced_existing_bond { " [reinforced]" } else { "" },
        kind_tag,
    );
    let mut structured = serde_json::to_value(&result).map_err(|e| (-32603, e.to_string()))?;
    if let Value::Object(ref mut m) = structured {
        if let Some(k) = resolved_subject_kind {
            m.insert("subject_kind".into(), Value::String(k.to_string()));
        }
        if let Some(k) = resolved_object_kind {
            m.insert("object_kind".into(), Value::String(k.to_string()));
        }
        // OP-EXTRACT-PREP: echo the negation flag actually applied.
        m.insert("negated".into(), Value::Bool(negated));
    }
    Ok(wrap_content(text, structured))
}


/// Port T-Ingest — `lu4d_teach_corpus` MCP wrapper.
///
/// Strict superset of `tool_teach` semantics: same arg shape (subject,
/// relation, object, weight, confidence, kind) with three additions —
/// `corpus` (required), `run_id` (optional caller-supplied; otherwise
/// resolved per `ingest_run_id_strategy`), and `source_url` (optional
/// per-bond URL within the corpus).
///
/// Routes through `Brain::teach_corpus` (which itself calls
/// `teach_with_source` after resolving the run_id) so the resulting bond
/// carries `SourceAttribution::TrainingCorpus { corpus, run_id, url }`.
///
/// Anti-bypass invariant (T.3 PendingLog gate): when
/// `brain.ingest_config.auto_approve_threshold = Some(t)` AND the call's
/// confidence is below `t`, the change is enqueued onto T.3 PendingLog
/// as `PendingChangeKind::AddBond` with `proposed_by = TrainingCorpus`,
/// returning `gated=true` plus a `pending_id` so the caller can approve
/// via `lu4d_approve_changes`. The grep-guard test
/// `tingest_grep_guard_no_t3_bypass` asserts this routing is the only
/// way out of the function — i.e. the function body must reference
/// either `teach_corpus` (direct path) or `pending.push` (gated path),
/// never `network::add_bond` directly.
fn tool_teach_corpus(brain: &mut Brain, args: Value) -> Result<Value, (i64, String)> {
    let subject = arg_str(&args, "subject")?.to_string();
    let relation_str = arg_str(&args, "relation")?.to_string();
    let object = arg_str(&args, "object")?.to_string();
    let corpus = arg_str(&args, "corpus")?.to_string();
    let weight = args.get("weight").and_then(|v| v.as_f64()).unwrap_or(0.8) as f32;
    let confidence_str = args.get("confidence").and_then(|v| v.as_str()).unwrap_or("Verified");
    let caller_run_id = args.get("run_id").and_then(|v| v.as_str()).map(|s| s.to_string());
    let source_url = args.get("source_url").and_then(|v| v.as_str()).map(|s| s.to_string());
    // OP-EXTRACT-PREP: optional negation flag. Threads into both
    // gated PendingChangeKind::AddBond (which already carries `negated`)
    // and the direct-apply path (set_bond_negated after teach_corpus_with_kind).
    let negated = args.get("negated").and_then(|v| v.as_bool()).unwrap_or(false);
    // Reject empty `corpus` early — empty-string corpus would silently
    // collide with the bare-teach default attribution semantics.
    if corpus.trim().is_empty() {
        return Err((-32602, "`corpus` must be a non-empty string".to_string()));
    }
    let kind_arg: Option<String> = args.get("kind").and_then(|v| v.as_str()).map(|s| s.to_string());
    if let Some(ref k) = kind_arg {
        if crate::node::KnowledgeContent::from_kind_str(k, String::new()).is_none() {
            return Err((
                -32602,
                format!(
                    "unknown kind: {:?}. Allowed: Fact, Concept, Procedure, Episode, Preference, Raw, Code, Image, Definition, Mechanism, Composition, Purpose, Contrast, Causal, Example, Meta",
                    k
                ),
            ));
        }
    }
    let relation: BondType = parse_bond_type(&relation_str).map_err(|e| (-32602, e.to_string()))?;
    let confidence: Confidence = parse_confidence(confidence_str).map_err(|e| (-32602, e.to_string()))?;

    // Map Confidence → numeric for the threshold comparison. Mirrors the
    // banding used by T.3 apply_change so semantics stay consistent.
    let conf_score: f32 = match confidence {
        Confidence::Terminal => 1.00,
        Confidence::Verified => 0.90,
        Confidence::Inferred => 0.65,
        Confidence::Unverified => 0.40,
        Confidence::Contested => 0.20,
    };

    // ---------------------------------------------------------------
    // T.3 gate: when `auto_approve_threshold` is Some(t) and the call's
    // confidence band falls below `t`, push onto PendingLog instead of
    // applying directly. The grep-guard test below ensures we always go
    // through one of these two branches and never bypass via
    // `network::add_bond` / `add_bond_with_scar_check` directly.
    // ---------------------------------------------------------------
    if let Some(threshold) = brain.ingest_config.auto_approve_threshold {
        if conf_score < threshold {
            let resolved_run_id = brain.resolve_ingest_run_id(caller_run_id.clone());
            let proposed_by = crate::source::SourceAttribution::training_with_url(
                corpus.clone(),
                resolved_run_id.clone(),
                source_url.clone(),
            );
            let now = crate::brain::now_ms() as u64;
            let change = crate::pending::PendingChange {
                pending_id: 0,
                proposed_at_ms: now,
                proposed_by: proposed_by.clone(),
                change_kind: crate::pending::PendingChangeKind::AddBond {
                    subject_text: subject.clone(),
                    object_text: object.clone(),
                    bond_type: relation,
                    weight,
                    // OP-EXTRACT-PREP: thread caller-supplied flag.
                    negated,
                },
                confidence: conf_score,
                reason: format!(
                    "T-Ingest gated: corpus={} conf={:.2} below threshold={:.2}",
                    corpus, conf_score, threshold,
                ),
                status: crate::pending::PendingStatus::Pending,
            };
            let pending_id = brain.pending.push(change).map_err(|e| (-32603, e.to_string()))?;
            let text = format!(
                "T-Ingest gated: queued pending id={} (corpus={} conf={:.2} threshold={:.2})",
                pending_id, corpus, conf_score, threshold,
            );
            let structured = serde_json::json!({
                "gated": true,
                "pending_id": pending_id,
                "corpus": corpus,
                "run_id": resolved_run_id,
                "source_url": source_url,
                "confidence": conf_score,
                "threshold": threshold,
                // OP-EXTRACT-PREP
                "negated": negated,
                "attribution": {
                    "kind": "TrainingCorpus",
                    "corpus": corpus,
                    "run_id": resolved_run_id,
                    "url": source_url,
                }
            });
            return Ok(wrap_content(text, structured));
        }
    }

    // Direct apply path — strict superset of lu4d_teach with TrainingCorpus stamp.
    // BR.3: thread the optional kind hint and the raw relation alias
    // into `teach_corpus_with_kind`. Explicit `kind` (already validated
    // above) wins; if absent, the relation alias drives `infer_kind`
    // (which BR.3 widened to cover all 23 smart_extractor relations);
    // if both miss, the default Concept variant is used. Behavior on
    // existing nodes is unchanged — kind is fixed at first observation.
    let result = brain.teach_corpus_with_kind(
        &subject,
        relation,
        &object,
        weight,
        confidence,
        &corpus,
        caller_run_id,
        source_url.clone(),
        kind_arg.as_deref(),
        Some(relation_str.as_str()),
    );
    // OP-EXTRACT-PREP: same overlay-set sidecar pattern as tool_teach and
    // the gated path. Stamp the NegationOverlay AFTER the bond exists.
    if negated {
        brain.set_bond_negated(result.subject_seed, result.object_seed, relation, true);
    }

    // Read back the stamped attribution for confirmation in the response.
    let stamped = brain.bond_source(&result.subject_seed, &result.object_seed, relation);
    let (resolved_corpus, resolved_run_id, resolved_url) = match &stamped.kind {
        crate::source::SourceKind::TrainingCorpus { corpus, run_id, url } => {
            (corpus.clone(), run_id.clone(), url.clone())
        }
        other => return Err((
            -32603,
            format!("T-Ingest invariant: expected TrainingCorpus stamp, got {:?}", other),
        )),
    };

    let text = format!(
        "Taught (corpus={}): {} -[{:?}]-> {} (weight={:.2}, conf={:?}){}{}{} attribution=TrainingCorpus run_id={:?}",
        corpus, subject, relation, object, result.weight, confidence,
        if result.created_subject { " [new subject]" } else { "" },
        if result.created_object { " [new object]" } else { "" },
        if result.reinforced_existing_bond { " [reinforced]" } else { "" },
        resolved_run_id,
    );
    let mut structured = serde_json::to_value(&result).map_err(|e| (-32603, e.to_string()))?;
    if let Value::Object(ref mut m) = structured {
        m.insert("gated".into(), Value::Bool(false));
        m.insert("corpus".into(), Value::String(resolved_corpus.clone()));
        m.insert("run_id".into(), match &resolved_run_id { Some(r) => Value::String(r.clone()), None => Value::Null });
        m.insert("source_url".into(), match &resolved_url { Some(u) => Value::String(u.clone()), None => Value::Null });
        m.insert("attribution".into(), serde_json::json!({
            "kind": "TrainingCorpus",
            "corpus": resolved_corpus,
            "run_id": resolved_run_id,
            "url": resolved_url,
        }));
        // OP-EXTRACT-PREP
        m.insert("negated".into(), Value::Bool(negated));
    }
    Ok(wrap_content(text, structured))
}

/// Port BatchIngest — maximum entries accepted by `lu4d_teach_corpus_batch`
/// per call. Cap exists to bound IPC payload + per-call memory and to keep
/// failure blast radius contained. Aligns with the harness's BATCH_SIZE=100
/// default (10x headroom over the harness submission size).
pub(crate) const TEACH_CORPUS_BATCH_MAX_ENTRIES: usize = 1000;

/// Port BatchIngest — `lu4d_teach_corpus_batch` MCP wrapper.
///
/// Pure IPC-amortising batch over `lu4d_teach_corpus`. Each entry in
/// `args.entries` is routed through the SAME `tool_teach_corpus`
/// internal path — mirror of the trust multiplier, T.3 PendingLog
/// gating, source attribution stamping, hub-cap write-side, and
/// pending-vs-direct-apply branching. The speedup comes from
/// amortising the MCP IPC round-trip across N entries, NOT from
/// bypassing any per-bond logic.
///
/// Anti-bypass invariant (STRUCTURAL): the
/// `tingest_batch_grep_guard_uses_only_tool_teach_corpus` test
/// asserts the function body's only inner write path is
/// `tool_teach_corpus(`. The body must NOT reference
/// `brain.network.add_bond`, `add_bond_with_scar_check`,
/// `brain.teach_corpus(` directly, or `brain.pending.push` directly
/// — those would skip the per-entry validation/parsing the single-call
/// path performs.
///
/// Anti-hallucination invariant (BEHAVIOUR): stored bond weights are
/// NEVER mutated by the batch path. Per-entry processing is
/// byte-equivalent to N sequential `tool_teach_corpus` calls (modulo
/// system-clock per-keyframe timestamps). The unit test
/// `tingest_batch_state_equivalent_to_sequential` pins this on
/// (bond_count, node_count, sorted bond keys, bond_type, current weight).
///
/// Per-entry errors do NOT fail the whole batch — each error is
/// captured into the response `errors` array with its index so the
/// caller can surface them without losing the successful entries.
///
/// Args: `{ entries: [TeachCorpusEntry, ...] }` where each entry has
/// the same shape as `lu4d_teach_corpus` args (subject, relation,
/// object, corpus required; weight, confidence, kind, run_id,
/// source_url optional).
///
/// Returns: `{ bonds_landed, bonds_deduped, bonds_gated,
/// entries_attempted, errors: [{index, code,
/// error}, ...] }`.
fn tool_teach_corpus_batch(brain: &mut Brain, args: Value) -> Result<Value, (i64, String)> {
    let entries = match args.get("entries").and_then(|v| v.as_array()) {
        Some(arr) => arr.clone(),
        None => return Err((-32602, "`entries` must be an array of teach_corpus entry objects".to_string())),
    };
    if entries.is_empty() {
        return Err((-32602, "`entries` must be a non-empty array".to_string()));
    }
    if entries.len() > TEACH_CORPUS_BATCH_MAX_ENTRIES {
        return Err((-32602, format!(
            "batch size {} exceeds cap {} (split into smaller batches)",
            entries.len(),
            TEACH_CORPUS_BATCH_MAX_ENTRIES,
        )));
    }

    let entries_attempted: usize = entries.len();
    let mut bonds_landed: usize = 0;
    let mut bonds_deduped: usize = 0;
    let mut bonds_gated: usize = 0;
    let mut bonds_rejected: usize = 0;
    let mut errors: Vec<Value> = Vec::new();

    for (i, entry) in entries.into_iter().enumerate() {
        // Route through THE SAME internal path as a single-entry
        // `lu4d_teach_corpus` call. This is the anti-bypass invariant:
        // every per-entry trust/T.3/source-stamp check fires identically
        // to the sequential path; the batch only amortises IPC.
        match tool_teach_corpus(brain, entry) {
            Ok(resp) => {
                let s = resp.get("structuredContent").cloned().unwrap_or(Value::Null);
                if s.get("gated").and_then(|v| v.as_bool()).unwrap_or(false) {
                    bonds_gated += 1;
                } else if s.get("admission_rejected").and_then(|v| v.as_bool()).unwrap_or(false) {
                    bonds_rejected += 1;
                } else if s.get("reinforced_existing_bond").and_then(|v| v.as_bool()).unwrap_or(false) {
                    bonds_deduped += 1;
                } else {
                    bonds_landed += 1;
                }
            }
            Err((code, msg)) => {
                errors.push(serde_json::json!({
                    "index": i,
                    "code": code,
                    "error": msg,
                }));
            }
        }
    }

    let text = format!(
        "teach_corpus_batch: attempted={} landed={} rejected={} deduped={} gated={} errors={}",
        entries_attempted, bonds_landed, bonds_rejected, bonds_deduped, bonds_gated, errors.len(),
    );
    let structured = serde_json::json!({
        "bonds_landed": bonds_landed,
        "bonds_rejected": bonds_rejected,
        "bonds_deduped": bonds_deduped,
        "bonds_gated": bonds_gated,
        "entries_attempted": entries_attempted,
        "canon_gate_rejections_total": crate::canonical::canon_gate_rejections_total(),
        "errors": errors,
    });
    Ok(wrap_content(text, structured))
}


/// Port T-Ingest.2 — `lu4d_user_statement` MCP wrapper.
///
/// Routes a Tier 2 prompt text into Left Brain operational memory as
/// a `SessionEntryKind::UserStatement` entry. NEVER forms a Right Brain
/// bond — prompts are hypothetical-shape and bonding them as facts
/// would inject hallucination-shaped data into the very architecture
/// designed to prevent hallucination (see
/// `agent/memory/feedback_training_data_asymmetry.md`).
///
/// Asymmetry-correct ingest pattern: pair this call with
/// `lu4d_teach_corpus` for the model's response — only the response
/// (a fact-shaped statement) becomes a Right Brain bond; the prompt
/// stays in LB as operational state.
///
/// Anti-bypass invariant (STRUCTURAL): the
/// `tingest2_grep_guard_no_bond_formation` test asserts the function
/// body uses ONLY `brain.lb_session.push_internal` (the canonical LB
/// write path) and NEVER reaches for any bond-mutating API. The
/// FORBIDDEN identifier list is split via `concat!()` at compile
/// time so the test source itself doesn't self-trigger the grep.
fn tool_user_statement(brain: &mut Brain, args: Value) -> Result<Value, (i64, String)> {
    let text = arg_str(&args, "text")?.to_string();
    let source_label = args.get("source_label").and_then(|v| v.as_str()).map(|s| s.to_string());
    let caller_session_id = args.get("session_id").and_then(|v| v.as_u64());

    // Empty-text reject (would silently bloat LB with no payload).
    if text.is_empty() {
        return Err((-32602, "`text` must be a non-empty string".to_string()));
    }
    // Byte-length cap (anti-blowup invariant).
    let max_bytes = brain.tier2_prompt_config.max_text_bytes;
    if text.len() > max_bytes {
        return Err((
            -32602,
            format!(
                "text length {} exceeds tier2_prompt_max_text_bytes ({})",
                text.len(),
                max_bytes,
            ),
        ));
    }

    // Resolve session id. Caller-supplied wins; otherwise the
    // configured Tier 2 strategy decides.
    let (resolved_session_id, ephemeral_created) = match caller_session_id {
        Some(sid) => (sid, false),
        None => match brain.tier2_prompt_config.default_session_strategy {
            crate::runtime_config::Tier2PromptSessionStrategy::Ephemeral => {
                let new_id = brain.session_start();
                (new_id, true)
            }
            crate::runtime_config::Tier2PromptSessionStrategy::Reuse => {
                let resolved = brain
                    .lb_session
                    .active_session()
                    .unwrap_or(crate::lb_session::SCRATCHPAD_SESSION_ID);
                (resolved, false)
            }
        },
    };

    // Encode the prompt + source_label as the entry payload. The
    // subject carries the source_label (or `user_statement` when
    // unset) so an `lu4d_session_memory_inspect` audit can group
    // by ingest source visually; the content carries the JSON
    // structure for programmatic filtering.
    let subject = source_label
        .clone()
        .unwrap_or_else(|| "user_statement".to_string());
    let content_payload = serde_json::json!({
        "text": text,
        "source_label": source_label,
        "byte_len": text.len(),
    })
    .to_string();

    let entry = crate::lb_session::SessionEntry::new(
        resolved_session_id,
        crate::brain::now_ms() as u64,
        crate::lb_session::SessionEntryKind::UserStatement,
        subject,
        content_payload,
        0.0_f32,
    );

    // Single canonical write path. The grep-guard test below pins
    // this so future edits cannot silently reroute through a
    // bond-mutating API.
    let entry_id = brain
        .lb_session
        .push_internal(entry)
        .map_err(|e| (-32603, format!("lb_session push_internal: {}", e)))?;

    let text_msg = format!(
        "lu4d_user_statement: entry_id={} session_id={} kind=user_statement bytes={} source_label={:?} ephemeral_session={}",
        entry_id,
        resolved_session_id,
        text.len(),
        source_label,
        ephemeral_created,
    );
    let structured = serde_json::json!({
        "entry_id": entry_id,
        "session_id": resolved_session_id,
        "kind": "user_statement",
        "source_label": source_label,
        "byte_len": text.len(),
        "ephemeral_session": ephemeral_created,
        "asymmetry_invariant": "no_bond_formation",
    });
    Ok(wrap_content(text_msg, structured))
}

/// OP-RETRAIN-1 -- batch attach DEFINITIONS (meaning text) to concept nodes.
/// Each entry {label, definition} creates the node with content=label and
/// knowledge.content = Definition(definition) via Brain::set_definition
/// (create-only, admission-gated). Foundation loaders call this to ground
/// vocabulary with real meanings that "what is X" recall then surfaces.
fn tool_set_definition_batch(brain: &mut Brain, args: Value) -> Result<Value, (i64, String)> {
    let entries = args
        .get("entries")
        .and_then(|v| v.as_array())
        .ok_or((-32602, "`entries` array required".to_string()))?;
    // OP-REDEFINE-1: `update=true` additively attaches the gloss to an
    // existing definition-less node (recovers create-only-skipped lemmas).
    let update = args.get("update").and_then(|v| v.as_bool()).unwrap_or(false);
    let mut defined = 0usize;
    let mut skipped = 0usize;
    for e in entries {
        let label = e.get("label").and_then(|v| v.as_str()).unwrap_or("");
        let def = e.get("definition").and_then(|v| v.as_str()).unwrap_or("");
        if label.is_empty() || def.is_empty() {
            skipped += 1;
            continue;
        }
        let ok = if update {
            brain.set_definition_update(label, def)
        } else {
            brain.set_definition(label, def)
        };
        if ok {
            defined += 1;
        } else {
            skipped += 1;
        }
    }
    let text = format!("set_definition_batch: defined={} skipped={}", defined, skipped);
    Ok(wrap_content(
        text,
        serde_json::json!({
            "defined": defined,
            "skipped": skipped,
            "canon_gate_rejections_total": crate::canonical::canon_gate_rejections_total(),
        }),
    ))
}

fn tool_ask(brain: &mut Brain, args: Value) -> Result<Value, (i64, String)> {
    let raw_query = arg_str(&args, "query")?.to_string();
    // Port CS — observe style from the raw input. Best-effort,
    // does NOT mutate canonical brain state (anti-hallucination invariant).
    if brain.style_config.learning_enabled {
        let sample = crate::style::compute_style_sample(&raw_query);
        let user_key = brain.style_config.user_key_str().to_string();
        let lr = brain.style_config.learning_rate;
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64).unwrap_or(0);
        brain.cs_style_store.observe(&user_key, &sample, lr, now_ms);
    }
    // Port F (#576): arithmetic intent fires BEFORE recall AND before
    // any conversational-memory rewrite. Math doesn't compose with
    // pronoun resolution.
    if let Some(resp) = try_arithmetic_response_with(Some(brain), &raw_query) { return Ok(resp); }
    // OP-GREET-REFUSAL -- defense-in-depth pre-K-gate pleasantry refusal
    // for the ask path. lu4d_ask does not go through `narrate`, so the
    // hook landed in `narration::narrate` does not cover it; this is the
    // ask-side mirror. Math precedes (above), greet refusal follows.
    // Returns the canned prose in the same wrap_content envelope the
    // K-gate refusal uses, with structured.source = "greet_refusal" so
    // downstream consumers can distinguish it from idk_gate output.
    if let Some(gclass) = crate::greet_refusal::is_greeting(&raw_query) {
        let prose = crate::greet_refusal::refusal_prose(gclass).to_string();
        let reason = crate::greet_refusal::refusal_reason(gclass);
        let structured = serde_json::json!({
            "source": "greet_refusal",
            "refusal_reason": reason,
            "raw_query": raw_query,
        });
        return Ok(wrap_content(prose, structured));
    }
    let depth = args.get("depth").and_then(|v| v.as_u64()).unwrap_or(3) as usize;
    let top_k = args.get("top_k").and_then(|v| v.as_u64()).unwrap_or(10) as usize;

    let bond_filter: Vec<BondType> = args
        .get("bond_filter")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str())
                .filter_map(|s| parse_bond_type(s).ok())
                .collect()
        })
        .unwrap_or_default();

    // ---------------------------------------------------------------
    // Port Q.3 (#438 / #587): conversational-memory pre-pass.
    //
    // When `session_id` is provided we inspect the prior turn(s) to
    //   1) short-circuit pure backreference queries from history,
    //   2) substitute bare pronouns with the highest-weight recent
    //      anchor (refusing to substitute when all anchors are below
    //      PRONOUN_RESOLUTION_THRESHOLD — clarification beats a stale
    //      guess),
    //   3) re-anchor "tell me more" / "go deeper" elaborations on
    //      the previous turn's effective query.
    // ---------------------------------------------------------------
    let session_id = args.get("session_id").and_then(|v| v.as_u64());

    // Backreference short-circuit: answer directly from turn_history
    // without invoking recall.
    if let Some(sid) = session_id {
        if let Some(mem) = brain.session_memories.get(sid) {
            if let Some(kind) = crate::session_memory::classify_backreference(&raw_query) {
                let resp = match kind {
                    crate::session_memory::BackrefKind::FirstQuestion => {
                        mem.first_turn().map(|t| t.raw_query.clone())
                    }
                    crate::session_memory::BackrefKind::LastQuestion => {
                        // The current call hasn't been recorded yet, so
                        // last_turn() is the prior turn — exactly what
                        // "what did I just ask?" wants.
                        mem.last_turn().map(|t| t.raw_query.clone())
                    }
                    crate::session_memory::BackrefKind::LastAnswer => {
                        mem.last_turn().map(|t| t.response_text.clone())
                    }
                };
                let body = match resp {
                    Some(s) if !s.is_empty() => s,
                    _ => "(no prior turn in this session)".to_string(),
                };
                let text = format!(
                    "[backreference] {}\n",
                    body.lines().next().unwrap_or(&body)
                );
                let structured = serde_json::json!({
                    "session_id": sid,
                    "backreference_kind": format!("{:?}", kind),
                    "answer": body,
                    "raw_query": raw_query,
                });
                // We intentionally do NOT record this as a turn — it's
                // a meta-query about the conversation, not a content
                // turn that should pollute anchors.
                return Ok(wrap_content(text, structured));
            }
        }
    }

    // Determine the effective query (after pronoun substitution /
    // elaboration rewrite). Default: same as raw.
    let mut effective_query = raw_query.clone();
    let mut rewrite_note: Option<String> = None;
    let mut needs_clarification: Option<String> = None;

    if let Some(sid) = session_id {
        // Snapshot what we need from the memory in a short borrow,
        // then drop it before calling brain.ask().
        let (resolved_anchor, prior_effective): (Option<String>, Option<String>) =
            match brain.session_memories.get(sid) {
                Some(mem) => (
                    mem.resolve_pronoun().map(|a| a.label.clone()),
                    mem.last_turn().map(|t| t.effective_query.clone()),
                ),
                None => (None, None),
            };

        // 1. Elaboration: re-anchor on prior effective query.
        if crate::session_memory::is_why_followup(&raw_query) {
            // Bare "why?" -- route to a causal query on the prior subject.
            // Honest + self-gating: causal synthesis refuses cleanly when
            // it holds no causal edges for that subject.
            if let Some(subject) = resolved_anchor.clone() {
                effective_query = format!("what causes {}", subject);
                rewrite_note = Some(format!("why -> causal on \"{}\"", subject));
            } else {
                needs_clarification = Some(
                    "I don't have a subject from this conversation to explain yet -- what would you like to know the cause of?".to_string(),
                );
            }
        } else if crate::session_memory::is_elaboration(&raw_query) {
            if let Some(prev) = prior_effective.clone() {
                effective_query = format!("{} (elaborate)", prev);
                rewrite_note = Some(format!(
                    "elaboration -> re-asking previous turn (\"{}\") in greater depth",
                    prev
                ));
            } else {
                needs_clarification = Some(
                    "I have nothing to elaborate on yet — could you ask a question first?".to_string(),
                );
            }
        }
        // 2. Continuation ("and what about Y?"): leave query alone,
        // just note the carryover.
        else if crate::session_memory::is_continuation(&raw_query) {
            // "what about it" -> resolve the bare pronoun to the prior
            // subject anchor; a fresh noun phrase ("what about cats") has
            // no bare pronoun and is left untouched for normal recall.
            if crate::session_memory::contains_bare_pronoun(&raw_query) {
                if let Some(anchor) = resolved_anchor.clone() {
                    effective_query =
                        crate::session_memory::substitute_pronoun(&raw_query, &anchor);
                    rewrite_note = Some(format!("continuation pronoun -> \"{}\"", anchor));
                }
            }
        }
        // 3. Bare pronoun: substitute or refuse.
        else if crate::session_memory::contains_bare_pronoun(&raw_query) {
            if let Some(anchor) = resolved_anchor {
                effective_query = crate::session_memory::substitute_pronoun(&raw_query, &anchor);
                rewrite_note = Some(format!("pronoun -> \"{}\"", anchor));
            } else {
                needs_clarification = Some(format!(
                    "I want to make sure I answer the right thing — could you tell me what you mean by the pronoun in \"{}\"? (No recent anchor in this conversation has a strong enough weight to substitute confidently.)",
                    raw_query
                ));
            }
        }
    }

    // Clarification short-circuit. Recorded as a turn so future
    // backreference queries see it.
    // -----------------------------------------------------------
    // OP-NEARMISS-1 -- spelling repair, applied AFTER the session
    // rewrites (so a substituted pronoun anchor is never re-spelled)
    // and BEFORE recall. Candidates come only from the keyword index,
    // so this can rename a token onto a term Lu KNOWS and can do
    // nothing else: it cannot invent a term, cannot invent a bond, and
    // does not touch a single K-gate threshold. Genuinely unknown words
    // have no repair target and reach the gate exactly as typed.
    // -----------------------------------------------------------
    if needs_clarification.is_none() {
        if let Some((repaired, reps)) =
            crate::near_miss::repair_query(brain.network.keyword_index_ref(), &effective_query)
        {
            let note = format!("spelling -> {}", crate::near_miss::describe(&reps));
            rewrite_note = Some(match rewrite_note.take() {
                Some(prev) => format!("{}; {}", prev, note),
                None => note,
            });
            effective_query = repaired;
        }
    }

    if let Some(msg) = needs_clarification {
        if let Some(sid) = session_id {
            let now = crate::brain::now_ms();
            let mem = brain.session_memories.get_or_create(sid, now);
            mem.record_turn(
                raw_query.clone(),
                raw_query.clone(),
                msg.clone(),
                Vec::new(),
                crate::session_memory::TurnTraceSummary::empty(),
                now,
            );
        }
        let structured = serde_json::json!({
            "session_id": session_id,
            "clarification_requested": true,
            "raw_query": raw_query,
        });
        return Ok(wrap_content(msg, structured));
    }

    let result = if bond_filter.is_empty() {
        brain.ask(&effective_query, depth, top_k)
    } else {
        brain.typed_ask(&effective_query, depth, top_k, &bond_filter)
    };

    // -----------------------------------------------------------
    // Port K (#581) — Honest "I don't know" gate.
    //
    // After recall (and possibly after Q.3 conversational rewrite)
    // run the three-signal check. When the recall pre-check looks
    // healthy we skip reasoning entirely; otherwise we run a
    // lightweight reasoning pass and consult its trace. The gate
    // produces an early-return refusal *before* normal narration
    // construction — Port D's hedging and Bug B's bare-word fallback
    // remain in their respective tiers below this one.
    //
    // Q.3 deference: when Q.3 successfully rewrote the query (i.e.
    // `rewrite_note.is_some()`), the session anchor was strong
    // enough by construction — Port K defers and lets recall stand.
    // The spec is explicit: 'if a session has prior context and
    // the current query has weak signal, K should DEFER to the
    // session's prior anchor for resolution rather than refusing.'
    let idk_thresholds = crate::idk_gate::IdkThresholds::default();
    let q3_rewrote = rewrite_note.is_some();
    if !q3_rewrote
        && !crate::idk_gate::fast_pass_check(brain, &result, &effective_query, &idk_thresholds)
    {
        let trace = {
            let cfg = crate::reasoning::ReasoningConfig::default();
            let scope = crate::scope::TemporalScope::Current;
            crate::reasoning::reason_with_config(brain, &effective_query, scope, &cfg)
        };
        let signals = crate::idk_gate::evaluate_signals(
            brain, &result, &effective_query, Some(&trace), &idk_thresholds,
        );
        let verdict = crate::idk_gate::classify(&signals, &idk_thresholds);
        if verdict != crate::idk_gate::IdkVerdict::Pass {
            // Port J retrofit: emit a gap signal so the curiosity
            // engine can target this query for autonomous research.
            // Mode → GapKind mapping mirrors `build_response` above.
            let dominance = signals.unmatched_tokens.is_empty()
                && signals.is_single_anchor_dominance(&idk_thresholds)
                && signals.dominant_token.is_some()
                && !signals.coverage_missing_tokens.is_empty();
            let gap_kind = match verdict {
                crate::idk_gate::IdkVerdict::TotalIdk =>
                    crate::gap_signal::GapKind::TotalIdk,
                crate::idk_gate::IdkVerdict::PartialMatchLeak if dominance =>
                    crate::gap_signal::GapKind::SingleAnchorDominance,
                crate::idk_gate::IdkVerdict::PartialMatchLeak
                    if signals.unmatched_tokens.is_empty() =>
                    crate::gap_signal::GapKind::UnmatchedTokenLeak,
                crate::idk_gate::IdkVerdict::PartialMatchLeak =>
                    crate::gap_signal::GapKind::PartialMatchLeak,
                crate::idk_gate::IdkVerdict::Pass => unreachable!(),
            };
            // Severity: TotalIdk = 0.9 (most urgent — Lu literally
            // doesn'''t know); leaks = 0.7; dominance = 0.6.
            let severity = match gap_kind {
                crate::gap_signal::GapKind::TotalIdk => 0.9,
                crate::gap_signal::GapKind::SingleAnchorDominance => 0.6,
                crate::gap_signal::GapKind::UnmatchedTokenLeak
                | crate::gap_signal::GapKind::PartialMatchLeak => 0.7,
                _ => 0.5,
            };
            let port_origin = match gap_kind {
                crate::gap_signal::GapKind::SingleAnchorDominance => "K.1",
                _ => "K",
            };
            brain.gap_emitter.emit(crate::gap_signal::GapSignal {
                query: effective_query.clone(),
                port_origin,
                gap_kind,
                severity,
                context: serde_json::json!({
                    "max_activation": signals.max_activation,
                    "max_seed_activation": signals.max_seed_activation,
                    "matched": signals.matched_tokens.len(),
                    "unmatched": signals.unmatched_tokens.len(),
                    "seed_count": signals.seed_count,
                }),
            });
            let (prose, structured) = crate::idk_gate::build_response(
                verdict, &signals, &raw_query, &effective_query, session_id,
            );
            // Q.3 post-pass: still record the turn so backreference
            // queries on later turns can see this refusal happened.
            if let Some(sid) = session_id {
                let now = crate::brain::now_ms();
                let mem = brain.session_memories.get_or_create(sid, now);
                mem.record_turn(
                    raw_query.clone(),
                    effective_query.clone(),
                    prose.clone(),
                    Vec::new(),
                    crate::session_memory::TurnTraceSummary::empty(),
                    now,
                );
            }
            return Ok(wrap_content(prose, structured));
        }
    }
    // -----------------------------------------------------------

    let mut text = format!(
        "Query: {}\nSeed hits: {}\nActivated: {}\nNodes visited: {}\nBonds evaluated: {}\n",
        effective_query,
        result.seed_hits.len(),
        result.recall.activated.len(),
        result.recall.nodes_visited,
        result.recall.bonds_evaluated,
    );
    if let Some(note) = &rewrite_note {
        text.push_str(&format!("[rewrite] {}\n", note));
    }
    for (i, a) in result.recall.activated.iter().take(top_k).enumerate() {
        text.push_str(&format!(
            "  {}. {} (activation={:.4}, depth={}, conf={:?})\n",
            i + 1, a.content, a.activation, a.depth, a.confidence,
        ));
    }
    // OP-RETRAIN-1: surface the DEFINITION of the top answer node so "what is X"
    // returns the grounded meaning (WordNet/dictionary gloss), not just its
    // neighbours. Reads get_node_v5 for the top node only (answer head) --
    // scoped, no perf hit on the traversal.
    if let Some(top) = result.recall.activated.first() {
        if let Some(id) = brain.network.node_intern().get(&top.seed) {
            if let Ok(Some(node)) = brain.get_node_v5(id) {
                if let crate::node::KnowledgeContent::Definition(def) = &node.knowledge.content {
                    if !def.is_empty() {
                        text.push_str(&format!("Definition of {}: {}\n", top.content, def));
                    }
                }
            }
        }
    }
    if result.hebbian_strengthened > 0 {
        text.push_str(&format!("Hebbian strengthening: {} bonds\n", result.hebbian_strengthened));
    }

    // ---------------------------------------------------------------
    // Port Q.3 post-pass: write the new turn to session memory so the
    // NEXT call can resolve "it"/"that"/"tell me more".
    // ---------------------------------------------------------------
    if let Some(sid) = session_id {
        let top_activations: Vec<String> = result
            .recall
            .activated
            .iter()
            .take(top_k.min(8))
            .map(|a| a.content.clone())
            .collect();
        let now = crate::brain::now_ms();
        let mem = brain.session_memories.get_or_create(sid, now);
        mem.record_turn(
            raw_query.clone(),
            effective_query.clone(),
            text.clone(),
            top_activations,
            crate::session_memory::TurnTraceSummary::empty(),
            now,
        );
    }

    let mut structured = serde_json::to_value(&result).map_err(|e| (-32603, e.to_string()))?;
    if let Some(obj) = structured.as_object_mut() {
        if let Some(sid) = session_id {
            obj.insert("session_id_q3".into(), serde_json::json!(sid));
            obj.insert("raw_query".into(), serde_json::json!(raw_query));
            obj.insert("effective_query".into(), serde_json::json!(effective_query));
            if let Some(note) = rewrite_note {
                obj.insert("rewrite_note".into(), serde_json::json!(note));
            }
        }
    }
    Ok(wrap_content(text, structured))
}

fn tool_status(brain: &mut Brain) -> Result<Value, (i64, String)> {
    // OP-BRAIN-V4 Phase 9: v4-aware status. For v4-loaded brains the v3
    // network HashMaps stay empty by design (bonds live in MemTable +
    // SSTs). The pre-Phase-9 status code read `brain.network.bond_count()`
    // and reported zero forever on the v4 cutover, masking the write-
    // path-routing bug for a full ingest run. Patch the BrainStatus in
    // place with v4-sourced node/bond counts so the harness gets honest
    // numbers.
    let mut s = brain.status();
    if brain.brain_dir_v4().is_some() {
        let mt_bonds = brain.memtable().map(|m| m.bond_count()).unwrap_or(0);
        let mt_subjects = brain.memtable().map(|m| m.subject_count()).unwrap_or(0);
        let sst_bonds: usize = brain
            .manifest_snapshot()
            .map(|snap| snap.iter_ssts().map(|(_, e)| e.bond_count as usize).sum())
            .unwrap_or(0);
        s.bond_count = mt_bonds + sst_bonds;
        // No node-on-disk format yet for v4: SSTs carry per-block
        // subject tables, not KnowledgeNode records. The closest
        // honest count is subject_count from the MemTable plus the
        // in-memory `network.nodes` HashMap. We sum them (with the
        // understanding that the same subject may be in both and is
        // over-counted at the margin -- the harness already prints
        // this as a soft indicator, not a load-bearing accounting
        // figure).
        s.node_count = if brain.read_path_mode() == crate::read_path::ReadPathMode::V5Only {
            // OP-BRAIN-V5 6.6: v5 nodes live in node_store SSTs, not
            // network.nodes; the node intern is the brain-wide count.
            brain.network.node_intern().len()
        } else {
            mt_subjects + brain.network.node_count()
        };
        // OP-BRAIN-V5 6.6: report the format honestly (was hard-coded
        // "v4"; a v5 brain reuses the brain_dir_v4 slot but read_path =
        // V5Only, so brain_format_label distinguishes them).
        s.format_version = brain.brain_format_label().to_string();
        if s.brain_file.is_none() {
            s.brain_file = brain.brain_dir_v4()
                .map(|p| p.display().to_string());
        }
    }
    let mut text = format!(
        "Lu 4D Status:\n  nodes: {}\n  bonds: {}\n  total sessions: {}\n  active session: {:?}\n  active activations: {}\n  active keyframes: {}\n  brain file: {:?}\n  format version: {}\n  encrypted at rest: {}\n  bridges (recent): {}\n  bridges dropped (overflow total): {}\n  recent metrics: {}\n  total synthesis triangles fired: {}\n  total confirmed hypotheses: {}\n  wonder queue size: {}\n  interest profile domains: {}\n  multi_think calls: {}\n  degree_p90: {}\n  max_activations: {}\n  max_activations_percentile: {}\n  effective_max_activations: {}\n  save_in_progress: {}\n  save_started_at_ms: {:?}",
        s.node_count, s.bond_count, s.total_sessions, s.active_session_id,
        s.active_session_activations, s.active_session_keyframes, s.brain_file, s.format_version, s.encrypted_at_rest,
        s.bridge_count, s.bridges_dropped_overflow_total,
        s.recent_metrics_count, s.total_synthesis_triangles_fired, s.total_confirmed_hypotheses,
        s.wonder_queue_size, s.interest_profile_domain_count, s.multi_think_call_count,
        s.degree_p90, s.max_activations, s.max_activations_percentile, s.effective_max_activations,
        s.save_in_progress, s.save_started_at_ms,
    );
    text.push_str(&format!(
        "\n  canon gate rejections (total): {}",
        crate::canonical::canon_gate_rejections_total()
    ));
    // Show up to 5 most-recent bridges as an at-a-glance preview. Full list
    // is in `structuredContent` for MCP clients that want everything.
    if !s.recent_bridges.is_empty() {
        text.push_str("\n  most recent bridges:");
        let n = s.recent_bridges.len();
        let preview = s.recent_bridges.iter().rev().take(5);
        for b in preview {
            text.push_str(&format!(
                "\n    {} <-> {} (weight={:.3}, jaccard={:.3})",
                b.low_content, b.high_content, b.bond_weight, b.jaccard,
            ));
        }
        if n > 5 {
            text.push_str(&format!("\n    ... and {} more", n - 5));
        }
    }
    let structured = serde_json::to_value(&s).map_err(|e| (-32603, e.to_string()))?;
    Ok(wrap_content(text, structured))
}

fn tool_session_start(brain: &mut Brain) -> Result<Value, (i64, String)> {
    let id = brain.session_start();
    Ok(wrap_content(format!("Started session {}", id), json!({"session_id": id})))
}

fn tool_session_end(brain: &mut Brain) -> Result<Value, (i64, String)> {
    match brain.session_end() {
        Some(id) => Ok(wrap_content(format!("Ended session {}", id), json!({"session_id": id}))),
        None => Ok(wrap_content("No active session".into(), json!({"session_id": null}))),
    }
}

fn tool_timetravel(brain: &mut Brain, args: Value) -> Result<Value, (i64, String)> {
    let query = arg_str(&args, "query")?.to_string();
    let now = crate::brain::now_ms();
    let at = if let Some(t) = args.get("at_ms").and_then(|v| v.as_i64()) {
        t
    } else if let Some(d) = args.get("ms_ago").and_then(|v| v.as_i64()) {
        now - d
    } else {
        now - 60 * 60 * 1000 // default: 1 hour ago
    };

    let diff = brain.timetravel(&query, at);
    let text = format!(
        "Time travel @ {} (query=\"{}\"): emerged={}, faded={}, strengthened={}, weakened={}, stable={}, avg_abs_delta={:.4}",
        at, query,
        diff.summary.emerged, diff.summary.faded,
        diff.summary.strengthened, diff.summary.weakened,
        diff.summary.stable, diff.summary.avg_abs_delta,
    );
    let structured = serde_json::to_value(&diff).map_err(|e| (-32603, e.to_string()))?;
    Ok(wrap_content(text, structured))
}

fn tool_consolidate(brain: &mut Brain) -> Result<Value, (i64, String)> {
    let stats = brain.consolidate();
    let text = format!(
        "Consolidation: promoted={}, pruned={}, examined={} | \
         Synthesis: examined={}, proposed={}, skipped_hub={}, skipped_threshold={} | \
         degree_p90_after_consolidation={}",
        stats.bonds_promoted, stats.keyframes_pruned, stats.bonds_examined,
        stats.triangles_examined, stats.triangles_proposed,
        stats.triangles_skipped_hub, stats.triangles_skipped_threshold,
        stats.degree_p90_after_consolidation,
    );
    let structured = serde_json::to_value(&stats).map_err(|e| (-32603, e.to_string()))?;
    Ok(wrap_content(text, structured))
}

fn tool_purge_stopwords(brain: &mut Brain) -> Result<Value, (i64, String)> {
    // Comprehensive list: grammatical function words that pollute recall
    // when they slip into the graph as nodes.
    const STOPWORDS: &[&str] = &[
        // articles / determiners
        "a", "an", "the", "this", "that", "these", "those",
        // prepositions
        "in", "on", "at", "to", "for", "of", "with", "by", "from",
        "into", "through", "during", "before", "after", "above", "below",
        "between", "up", "down", "out", "off", "over", "under", "again",
        "further", "about", "without", "within", "along", "across",
        "behind", "beyond", "among", "against", "toward", "towards",
        "upon", "around", "except", "per",
        // conjunctions
        "and", "or", "but", "so", "nor", "yet", "both", "either",
        "neither", "although", "though", "while", "whereas", "because",
        "since", "if", "unless", "until", "when", "whenever", "where",
        "wherever", "as", "than",
        // pronouns
        "i", "you", "he", "she", "it", "we", "they",
        "me", "him", "her", "us", "them",
        "my", "your", "his", "its", "our", "their",
        "mine", "yours", "hers", "ours", "theirs",
        "who", "whom", "whose", "which", "what",
        "myself", "yourself", "himself", "herself", "itself",
        "ourselves", "themselves",
        // auxiliary / modal verbs
        "be", "been", "being",
        "am", "is", "are", "was", "were",
        "have", "has", "had",
        "do", "does", "did",
        "will", "would", "shall", "should",
        "may", "might", "must", "can", "could",
        "need", "dare", "ought",
        // question / relative words
        "how", "why", "where", "when", "whether",
        // common filler / connector words that show up as query verbs
        "there", "here", "once", "now", "then", "just", "also", "even",
        "still", "already", "almost", "always", "never", "often",
        "very", "too", "so", "not", "no", "only", "same", "other",
        "such", "some", "any", "all", "each", "every", "both", "few",
        "more", "most", "much", "many", "well", "back", "else",
        // query-verb noise (appeared as top results in eval)
        "make", "makes", "made",
        "get", "gets", "got",
        "go", "goes", "went",
        "come", "comes", "came",
        "see", "sees", "saw",
        "know", "knows", "knew",
        "think", "thinks", "thought",
        "use", "uses", "used",
        "say", "says", "said",
        "give", "gives", "gave",
        "take", "takes", "took",
        "want", "wants", "wanted",
        "like", "likes", "liked",
        "look", "looks", "looked",
        "seem", "seems", "seemed",
        "let", "lets",
        "put", "puts",
        "set", "sets",
        "lot", "lots",
        "thing", "things",
        "way", "ways",
        "time", "times",
        "part", "parts",
        "place", "places",
        "day", "days",
        "man", "men",
        "new", "old",
        "first", "last",
        "long", "large", "small",
        "own",
    ];

    let before_nodes = brain.network.node_count();
    let before_bonds = brain.network.bond_count();
    let (nodes_removed, bonds_removed) = brain.network.purge_nodes(STOPWORDS);
    let after_nodes = brain.network.node_count();
    let after_bonds = brain.network.bond_count();

    let text = format!(
        "Purged {} stopword nodes and {} bonds. Graph: {}→{} nodes, {}→{} bonds.",
        nodes_removed, bonds_removed,
        before_nodes, after_nodes,
        before_bonds, after_bonds,
    );
    Ok(wrap_content(text, json!({
        "nodes_removed": nodes_removed,
        "bonds_removed": bonds_removed,
        "nodes_before": before_nodes,
        "nodes_after":  after_nodes,
        "bonds_before": before_bonds,
        "bonds_after":  after_bonds,
    })))
}

fn tool_prune_legacy_goals(brain: &mut Brain) -> Result<Value, (i64, String)> {
    let report = brain.prune_legacy_goal_nodes();
    let text = format!(
        "{} keyword entries pruned, {} interest entries pruned, {} orphan interest entries pruned",
        report.keyword_entries_pruned, report.interest_entries_pruned, report.orphan_interest_pruned,
    );
    Ok(wrap_content(text, json!({
        "keyword_entries_pruned": report.keyword_entries_pruned,
        "interest_entries_pruned": report.interest_entries_pruned,
        "orphan_interest_pruned": report.orphan_interest_pruned,
    })))
}

fn tool_think(brain: &mut Brain, args: Value) -> Result<Value, (i64, String)> {
    let query = arg_str(&args, "query")?.to_string();
    let mut opts = crate::thinking::ThinkConfig::default();
    if let Some(n) = args.get("max_steps").and_then(|v| v.as_u64()) {
        opts.max_steps = n as usize;
    }
    if let Some(vm) = args.get("variant_mode").and_then(|v| v.as_bool()) {
        opts.variant_mode = vm;
    }
    if let Some(c) = args.get("confidence_to_answer").and_then(|v| v.as_f64()) {
        opts.confidence_to_answer = c as f32;
    }
    if let Some(h) = args.get("hypotheses_per_step").and_then(|v| v.as_u64()) {
        opts.hypotheses_per_step = h as usize;
    }
    let result = brain.think(&query, opts);
    let mut text = format!(
        "Think: {}\n  steps: {}\n  hypothesis bonds written: {}\n  promoted: {}\n  final score: {:.2}\n  early exit: {}\nFinal recall: {} activated, {} nodes visited\n",
        query,
        result.trace.steps.len(),
        result.trace.hypothesis_bonds_written,
        result.trace.hypotheses_promoted,
        result.trace.final_score,
        result.trace.early_exit,
        result.final_answer.recall.activated.len(),
        result.final_answer.recall.nodes_visited,
    );
    for (i, a) in result.final_answer.recall.activated.iter().take(10).enumerate() {
        text.push_str(&format!(
            "  {}. {} (activation={:.4}, depth={})\n",
            i + 1, a.content, a.activation, a.depth,
        ));
    }
    let structured = serde_json::to_value(&result).map_err(|e| (-32603, e.to_string()))?;
    Ok(wrap_content(text, structured))
}

fn tool_variants_generate(brain: &mut Brain, args: Value) -> Result<Value, (i64, String)> {
    let goal_hex = arg_str(&args, "goal_seed")?.to_string();
    let bytes = hex_to_bytes_32(&goal_hex)
        .ok_or((-32602, "goal_seed must be 64 hex chars (32 bytes)".to_string()))?;
    let goal_seed = crate::seed::Seed::from_bytes(bytes);
    let arena = brain.generate_variants(goal_seed);

    let mut text = format!(
        "Variants generated for goal {}: {} specs, winner={:?}
",
        &goal_hex[..16.min(goal_hex.len())],
        arena.variants.len(),
        arena.winner_id,
    );
    for (s, sc) in arena.variants.iter().zip(arena.scores.iter()) {
        text.push_str(&format!(
            "  id={} {:?} composite={:.3} (coh={:.3}, nov={:.3}, par={:.3})
",
            s.id, s.strategy, sc.composite, sc.coherence, sc.novelty, sc.parsimony,
        ));
    }
    let structured = serde_json::to_value(&arena).map_err(|e| (-32603, e.to_string()))?;
    Ok(wrap_content(text, structured))
}

fn tool_variants_promote(brain: &mut Brain, args: Value) -> Result<Value, (i64, String)> {
    let winner_id = args.get("winner_id").and_then(|v| v.as_u64())
        .ok_or((-32602, "winner_id (u64) required".to_string()))?;
    let arena = crate::variants::VariantArena {
        variants: Vec::new(),
        scores: Vec::new(),
        winner_id: Some(winner_id),
    };
    let count = brain.promote_variant(&arena);
    let text = format!("Promoted variant {}: {} bond(s) re-tagged.
", winner_id, count);
    Ok(wrap_content(text, json!({"winner_id": winner_id, "promoted": count})))
}

/// Decode a 64-char hex string into [u8;32]. Returns None on bad input.
fn hex_to_bytes_32(s: &str) -> Option<[u8; 32]> {
    if s.len() != 64 { return None; }
    let mut out = [0u8; 32];
    for i in 0..32 {
        let byte = u8::from_str_radix(&s[i*2..i*2+2], 16).ok()?;
        out[i] = byte;
    }
    Some(out)
}

fn tool_wonder(brain: &Brain, args: Value) -> Result<Value, (i64, String)> {
    let top_n = args.get("top_n").and_then(|v| v.as_u64()).unwrap_or(10) as usize;
    let now = crate::brain::now_ms() as u64;
    let halflife = brain.curiosity_config.interest_decay_halflife_ms;
    let signals = brain.wonder_queue.top_n(top_n);
    let interest_snapshot = brain.interest_profile.snapshot(now, halflife);

    let mut text = format!(
        "Wonder queue: {} signal(s); interest profile: {} domain(s); curiosity dropped (overflow total): {}\n",
        brain.wonder_queue.len(),
        brain.interest_profile.interests.len(),
        brain.curiosity_dropped_overflow_total,
    );
    if signals.is_empty() {
        text.push_str("  (no signals yet — query the brain to generate them)\n");
    } else {
        text.push_str("Top signals:\n");
        for (i, s) in signals.iter().enumerate() {
            // Port J.1: if the seed isn't a real node (synthetic
            // gap-seed), fall back to the persisted gap_label_map
            // before the raw hex hash.
            let label = brain.network.get_node(&s.seed)
                .map(|n| n.content().to_string())
                .or_else(|| brain.gap_label_map.get(&s.seed).cloned())
                .unwrap_or_else(|| s.seed.to_string());
            text.push_str(&format!(
                "  {}. [{:?}] {} (intensity={:.3})\n",
                i + 1, s.signal_type, label, s.intensity,
            ));
        }
    }
    if !interest_snapshot.is_empty() {
        text.push_str("Interest profile (top 5):\n");
        for (seed, w, _) in interest_snapshot.iter().take(5) {
            let label = brain.network.get_node(seed)
                .map(|n| n.content().to_string()).unwrap_or_else(|| seed.to_string());
            text.push_str(&format!("  - {} (weight={:.3})\n", label, w));
        }
    }

    let interest_json: Vec<Value> = interest_snapshot.iter().map(|(s, w, t)| {
        let label = brain.network.get_node(s)
            .map(|n| n.content().to_string()).unwrap_or_default();
        json!({"seed": s.to_string(), "label": label, "decayed_weight": w, "last_touched_ms": t})
    }).collect();

    let signals_json: Vec<Value> = signals.iter().map(|s| {
        // Port J.1: gap_label_map fallback for synthetic seeds.
        let label = brain.network.get_node(&s.seed)
            .map(|n| n.content().to_string())
            .or_else(|| brain.gap_label_map.get(&s.seed).cloned())
            .unwrap_or_default();
        json!({
            "seed": s.seed.to_string(),
            "label": label,
            "signal_type": format!("{:?}", s.signal_type),
            "intensity": s.intensity,
            "timestamp_ms": s.timestamp_ms,
        })
    }).collect();

    Ok(wrap_content(text, json!({
        "wonder_queue_len": brain.wonder_queue.len(),
        "wonder_queue_cap": brain.curiosity_config.wonder_queue_cap,
        "interest_profile_len": brain.interest_profile.interests.len(),
        "interest_profile_cap": brain.curiosity_config.interest_profile_cap,
        "curiosity_dropped_overflow_total": brain.curiosity_dropped_overflow_total,
        "signals": signals_json,
        "interest_profile": interest_json,
    })))
}

fn tool_flush(brain: &mut Brain, _args: Value) -> Result<Value, (i64, String)> {
    // OP-FLUSH-1: on-demand synchronous MemTable -> L0 SST flush. Wraps
    // Brain::flush_now (src/brain.rs), which seals the current MemTable,
    // writes an L0 SST, commits the MANIFEST, and truncates the WAL up to
    // the flushed watermark. Unlike lu4d_save (a documented no-op for
    // v4-directory brains), this forces the drain synchronously so the WAL
    // stays bounded across restarts. Safe during background compaction
    // (shared commit_lock).
    let dir = brain.brain_dir_v4().map(|p| p.display().to_string());
    match brain.flush_now() {
        Ok(()) => {
            let text = match &dir {
                Some(d) => format!(
                    "lu4d_flush: sealed MemTable -> L0 SST, committed MANIFEST, truncated WAL at {}",
                    d
                ),
                None => "lu4d_flush: sealed MemTable -> L0 SST, committed MANIFEST, truncated WAL"
                    .to_string(),
            };
            Ok(wrap_content(text, json!({
                "flushed": true,
                "path": dir,
            })))
        }
        Err(e) => Err((-32603, format!("lu4d_flush: flush_now failed: {:?}", e))),
    }
}

fn tool_save(brain: &mut Brain, args: Value) -> Result<Value, (i64, String)> {
    // OP-SAVE-ASYNC — wire contract path (a) + path (b) under the
    // same tool. `mode` defaults to "blocking" for back-compat with
    // pre-OP-SAVE-ASYNC callers; the curriculum harness opts into
    // "async" + lu4d_save_status polling.
    let mode = args
        .get("mode")
        .and_then(|v| v.as_str())
        .unwrap_or("blocking")
        .to_lowercase();
    // OP-BRAIN-V4 Phase 8: v4-loaded brains save continuously via the
    // flush worker; lu4d_save is a documented no-op success for both
    // modes. The MemTable + WAL durability contract guarantees that
    // every bond accepted before this call is already in the WAL and
    // will land in an L0 SST at the next flush cycle.
    if let Some(dir) = brain.brain_dir_v4().map(|p| p.display().to_string()) {
        // OP-BRAIN-MapRAM persist: durability is continuous (MemTable ->
        // WAL -> L0 SST), so the bonds are already safe. But an explicit
        // lu4d_save is the natural checkpoint at which to refresh the
        // resident degree-map sidecar so the NEXT boot takes the fast
        // path instead of rebuilding from every bond SST. Best-effort:
        // a refresh failure must not turn a successful save into an
        // error (the brain is durable regardless).
        // OP-INGEST-LEAN: only refresh the resident degree-map sidecar
        // when this brain maintains a resident map (full serving load).
        // In ingest-only mode the map is intentionally absent, and
        // rebuilding it here would run a full bond-SST scan on every
        // save -- exactly the serving work training must avoid. The
        // sidecar is refreshed when Lu is next served.
        if brain.maintains_resident_map() {
            if let Err(e) = brain.refresh_resident_degmap() {
                eprintln!("[mcp] lu4d_save: resident degmap refresh failed (non-fatal): {}", e);
            }
        }
        // OP-REVOKE-V5: persist the source + negation overlays so the
        // "revocable by source" guarantee survives a restart (load_v5
        // reloads them; lu4d_save is otherwise a v5 no-op). Best-effort:
        // a failure must not turn a successful save into an error.
        if let Err(e) = brain.persist_overlays() {
            eprintln!("[mcp] lu4d_save: overlay persist failed (non-fatal): {}", e);
        }
        let text = format!(
            "v4 brain at {} saves continuously via flush worker; lu4d_save is a no-op",
            dir,
        );
        return Ok(wrap_content(text, json!({
            "saved": true,
            "path": dir,
            "mode": mode,
            "brain_format": "v4",
            "noop": true,
        })));
    }
    match mode.as_str() {
        "blocking" => {
            // Path (a). For V3BinaryEncrypted brains we route through
            // save_async_blocking so the snapshot-thread architecture is
            // exercised on every save (and the wire contract `ok=True iff
            // committed` is preserved). For other formats fall through to
            // the legacy synchronous save.
            let res = if brain.save_format == crate::brain::BrainSaveFormat::V3BinaryEncrypted
                && brain.path.is_some()
            {
                brain.save_async_blocking()
            } else {
                brain.save()
            };
            match res {
                Ok(()) => {
                    let path = brain.path.as_ref()
                        .map(|p| p.display().to_string())
                        .unwrap_or_else(|| "<no path>".into());
                    let text = format!("Brain saved to {}", path);
                    Ok(wrap_content(text, json!({"saved": true, "path": path, "mode": "blocking"})))
                }
                Err(e) => Err((-32603, format!("save failed: {}", e))),
            }
        }
        "async" => {
            if brain.save_format != crate::brain::BrainSaveFormat::V3BinaryEncrypted
                || brain.path.is_none()
            {
                return Err((-32603,
                    "lu4d_save mode=async requires V3BinaryEncrypted brain with a path".into()));
            }
            match brain.start_async_save() {
                Ok(()) => {
                    let (label, started, _err) = brain.save_state_snapshot();
                    let path = brain.path.as_ref()
                        .map(|p| p.display().to_string())
                        .unwrap_or_else(|| "<no path>".into());
                    let text = format!("Brain async-save started; poll lu4d_save_status");
                    Ok(wrap_content(text, json!({
                        "saved": false,
                        "started": true,
                        "state": label,
                        "started_at_ms": started,
                        "path": path,
                        "mode": "async",
                    })))
                }
                Err(e) => Err((-32603, format!("start_async_save failed: {}", e))),
            }
        }
        other => Err((-32602, format!("lu4d_save: unknown mode `{}` (want blocking|async)", other))),
    }
}

/// OP-SAVE-ASYNC — poll the state of the most recent async save.
/// Idempotent: each call also drains a completed worker (rename +
/// WAL truncate + .kw compact) before reading state, so the harness
/// can rely on `state == "completed"` meaning the new brain.lbr is
/// on disk and durable.
fn tool_save_status(brain: &mut Brain) -> Result<Value, (i64, String)> {
    // OP-BRAIN-V4 Phase 8.2 hotfix 2026-06-01: v4-loaded brains save
    // continuously via the flush worker; the v3 save_state_snapshot
    // is meaningless and returns "idle" forever. The harness polls
    // this tool waiting for `completed` and would hang indefinitely
    // on a v4 brain. Short-circuit with `completed` so the harness
    // checkpoint loop advances. Bonds accepted before this call are
    // already in the WAL and will land in an L0 SST at the next
    // flush cycle; that durability is the equivalent of v3 completed.
    if let Some(dir) = brain.brain_dir_v4().map(|p| p.display().to_string()) {
        let text = format!("v4 brain at {}: continuous-save semantic, returning completed", dir);
        return Ok(wrap_content(text, json!({
            "state": "completed",
            "started_at_ms": 0u64,
            "error": serde_json::Value::Null,
            "path": dir,
            "snapshot_capture_ms": 0u64,
            "brain_format": "v4",
            "noop": true,
        })));
    }
    // Try to finalize. On error we still report state via snapshot
    // (the worker has already transitioned to Failed and set err).
    let finalize_err = match brain.finalize_save_if_ready() {
        Ok(_finalised) => None,
        Err(e) => Some(format!("finalize: {}", e)),
    };
    let (label, started, err, snapshot_capture_ms) = brain.save_state_snapshot_with_capture();
    let merged_err = match (err, finalize_err) {
        (Some(a), Some(b)) => Some(format!("{} ; {}", a, b)),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    };
    let path = brain.path.as_ref()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| "<no path>".into());
    let text = format!("save state: {}", label);
    Ok(wrap_content(text, json!({
        "state": label,
        "started_at_ms": started,
        "error": merged_err,
        "path": path,
        // OP-SNAPSHOT-PEAK: synchronous capture phase duration in ms.
        "snapshot_capture_ms": snapshot_capture_ms,
    })))
}

// =====================================================================
// Self-reflection / Meta-learning tools
// =====================================================================

fn tool_reflect(brain: &mut Brain) -> Result<Value, (i64, String)> {
    let n = brain.reflect();
    let total = brain.meta_reflection.proposals.len();
    let pending = brain.meta_reflection.pending().count();
    let text = format!(
        "Self-reflection ran. New proposals this pass: {}. Total proposals: {}. Pending: {}.",
        n, total, pending,
    );
    Ok(wrap_content(text, json!({
        "new_proposals": n,
        "total_proposals": total,
        "pending_proposals": pending,
        "reflection_count": brain.meta_reflection.reflection_count,
        "metric_buffer_len": brain.performance_monitor.len(),
    })))
}

fn tool_proposals_list(brain: &mut Brain, args: Value) -> Result<Value, (i64, String)> {
    let filter = args.get("status_filter").and_then(|v| v.as_str()).map(|s| s.to_lowercase());
    let mut props_json: Vec<Value> = Vec::new();
    let mut text = String::new();
    for p in &brain.meta_reflection.proposals {
        let status_label = match &p.status {
            crate::self_reflection::ProposalStatus::Pending => "pending".to_string(),
            crate::self_reflection::ProposalStatus::Approved { .. } => "approved".to_string(),
            crate::self_reflection::ProposalStatus::Rejected { .. } => "rejected".to_string(),
            crate::self_reflection::ProposalStatus::Applied { .. } => "applied".to_string(),
        };
        if let Some(f) = &filter {
            if &status_label != f { continue; }
        }
        let dir_label = match p.direction {
            crate::self_reflection::AdjustmentDirection::Raise(d) => format!("Raise(+{:.3})", d),
            crate::self_reflection::AdjustmentDirection::Lower(d) => format!("Lower(-{:.3})", d),
        };
        text.push_str(&format!(
            "  #{} [{}] {:?} {}: {:.3} -> {:.3} ({} metrics)\n    {}\n",
            p.id, status_label, p.target, dir_label, p.current_value, p.proposed_value,
            p.supporting_metric_count, p.evidence,
        ));
        props_json.push(json!({
            "id": p.id,
            "status": status_label,
            "target": format!("{:?}", p.target),
            "direction": dir_label,
            "current_value": p.current_value,
            "proposed_value": p.proposed_value,
            "evidence": p.evidence,
            "supporting_metric_count": p.supporting_metric_count,
            "created_ms": p.created_ms,
        }));
    }
    if props_json.is_empty() {
        text = "(no proposals match the filter)".into();
    } else {
        let header = format!("Proposals ({}):\n", props_json.len());
        text = header + &text;
    }
    Ok(wrap_content(text, json!({
        "proposals": props_json,
        "filter_applied": filter,
    })))
}

fn tool_proposal_decide(brain: &mut Brain, args: Value) -> Result<Value, (i64, String)> {
    let id = args.get("id").and_then(|v| v.as_u64())
        .ok_or((-32602, "missing integer arg 'id'".to_string()))?;
    let decision = arg_str(&args, "decision")?.to_lowercase();
    let reason = args.get("reason").and_then(|v| v.as_str()).map(|s| s.to_string());

    let (ok, action_label) = match decision.as_str() {
        "approve" => (brain.approve_proposal(id), "approved"),
        "reject" => (brain.reject_proposal(id, reason.clone()), "rejected"),
        other => return Err((-32602, format!("unknown decision: {}", other))),
    };
    if !ok {
        let text = format!("No pending proposal with id {} (already decided or unknown).", id);
        return Ok(wrap_content(text, json!({"ok": false, "id": id})));
    }
    let p = brain.meta_reflection.proposals.iter().find(|p| p.id == id);
    let status_label = match p.map(|p| &p.status) {
        Some(crate::self_reflection::ProposalStatus::Applied { .. }) => "applied",
        Some(crate::self_reflection::ProposalStatus::Approved { .. }) => "approved",
        Some(crate::self_reflection::ProposalStatus::Rejected { .. }) => "rejected",
        _ => "pending",
    };
    let text = format!(
        "Proposal #{} {} ({}).",
        id, action_label, status_label,
    );
    Ok(wrap_content(text, json!({
        "ok": true,
        "id": id,
        "decision": decision,
        "status": status_label,
    })))
}


fn tool_multi_think(brain: &mut Brain, args: Value) -> Result<Value, (i64, String)> {
    let query = arg_str(&args, "query")?.to_string();
    let mut cfg = brain.multi_agent_config.clone();
    if let Some(c) = args.get("critic_confidence_threshold").and_then(|v| v.as_f64()) {
        cfg.critic_confidence_threshold = c as f32;
    }
    if let Some(n) = args.get("max_recall_candidates").and_then(|v| v.as_u64()) {
        cfg.max_recall_candidates = n as usize;
    }
    if let Some(n) = args.get("max_hypotheses_per_synthesis").and_then(|v| v.as_u64()) {
        cfg.max_hypotheses_per_synthesis = n as usize;
    }
    if let Some(n) = args.get("agent_recall_depth").and_then(|v| v.as_u64()) {
        cfg.agent_recall_depth = n as usize;
    }
    let trace = brain.multi_think_with_config(&query, &cfg).map_err(|e| (-32603, e))?;
    let mut text = format!(
        "Multi-think id={} query={}: recall={}, hypotheses={}, vetted={}, timed_out={}\n",
        trace.multi_think_id, query,
        trace.recall.len(), trace.hypotheses.len(), trace.vetted.len(),
        trace.timed_out,
    );
    for s in &trace.steps {
        text.push_str(&format!(
            "  {:?}: {}us, {} -> {}\n",
            s.role, s.elapsed_us, s.items_in, s.items_out,
        ));
    }
    if !trace.vetted.is_empty() {
        text.push_str("Vetted beliefs:\n");
        for (i, b) in trace.vetted.iter().take(8).enumerate() {
            text.push_str(&format!(
                "  {}. {} -> {} -> {} (conf={:.3})\n",
                i + 1, b.from_content, b.via_content, b.to_content, b.confidence,
            ));
        }
    }
    let structured = serde_json::to_value(&trace).map_err(|e| (-32603, e.to_string()))?;
    Ok(wrap_content(text, structured))
}

fn tool_narrate(brain: &mut Brain, args: Value) -> Result<Value, (i64, String)> {
    let raw_query = arg_str(&args, "query")?.to_string();
    // OP-ROUTE-1: Port F (#576) parity with tool_ask and tool_reason.
    // Arithmetic intent fires BEFORE narration because math is computed,
    // not recalled — without this short-circuit a math query phrased
    // through lu4d_narrate falls through to spread activation and returns
    // keyword-overlap noise. Returns the arithmetic JSON envelope
    // unchanged (source: "arithmetic_primitive" / "expression_primitive"
    // / "construction_primitive" / "wordprob_primitive"), matching the
    // contract the other two call sites honor. No fabricated
    // NarrationResult shell: the math envelope IS the response.
    if let Some(resp) = try_arithmetic_response_with(Some(brain), &raw_query) { return Ok(resp); }
    let depth = args.get("depth").and_then(|v| v.as_u64()).unwrap_or(3) as usize;
    let top_k = args.get("top_k").and_then(|v| v.as_u64()).unwrap_or(10) as usize;
    let use_multi = args
        .get("use_multi_think")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    // -----------------------------------------------------------
    // OP-NARRATE-V2 (2026-05-31) — Q.3 session-memory pre-pass for
    // narrate. Mirrors tool_ask's passthrough logic so 'tell me
    // more' / pronoun substitution / backreference queries resolve
    // against prior turns in the same session. The MCP boundary is
    // the only layer with session memory access (brain.narrate sees
    // only the rewritten query), so all rewriting happens here.
    // -----------------------------------------------------------
    let session_id = args.get("session_id").and_then(|v| v.as_u64());

    // Backreference short-circuit ("what did I just ask?"): answer
    // directly from turn_history without invoking recall.
    if let Some(sid) = session_id {
        if let Some(mem) = brain.session_memories.get(sid) {
            if let Some(kind) = crate::session_memory::classify_backreference(&raw_query) {
                let resp = match kind {
                    crate::session_memory::BackrefKind::FirstQuestion => {
                        mem.first_turn().map(|t| t.raw_query.clone())
                    }
                    crate::session_memory::BackrefKind::LastQuestion => {
                        mem.last_turn().map(|t| t.raw_query.clone())
                    }
                    crate::session_memory::BackrefKind::LastAnswer => {
                        mem.last_turn().map(|t| t.response_text.clone())
                    }
                };
                let body = match resp {
                    Some(s) if !s.is_empty() => s,
                    _ => "(no prior turn in this session)".to_string(),
                };
                let text = format!(
                    "[backreference] {}\n",
                    body.lines().next().unwrap_or(&body)
                );
                let structured = serde_json::json!({
                    "session_id": sid,
                    "backreference_kind": format!("{:?}", kind),
                    "answer": body,
                    "raw_query": raw_query,
                });
                return Ok(wrap_content(text, structured));
            }
        }
    }

    // Determine the effective query (after pronoun substitution /
    // elaboration rewrite). Default: same as raw.
    let mut effective_query = raw_query.clone();
    let mut rewrite_note: Option<String> = None;
    let mut needs_clarification: Option<String> = None;

    if let Some(sid) = session_id {
        let (resolved_anchor, prior_effective): (Option<String>, Option<String>) =
            match brain.session_memories.get(sid) {
                Some(mem) => (
                    mem.resolve_pronoun().map(|a| a.label.clone()),
                    mem.last_turn().map(|t| t.effective_query.clone()),
                ),
                None => (None, None),
            };

        if crate::session_memory::is_why_followup(&raw_query) {
            // Bare "why?" -- route to a causal query on the prior subject.
            // Honest + self-gating: causal synthesis refuses cleanly when
            // it holds no causal edges for that subject.
            if let Some(subject) = resolved_anchor.clone() {
                effective_query = format!("what causes {}", subject);
                rewrite_note = Some(format!("why -> causal on \"{}\"", subject));
            } else {
                needs_clarification = Some(
                    "I don't have a subject from this conversation to explain yet -- what would you like to know the cause of?".to_string(),
                );
            }
        } else if crate::session_memory::is_elaboration(&raw_query) {
            if let Some(prev) = prior_effective.clone() {
                effective_query = format!("{} (elaborate)", prev);
                rewrite_note = Some(format!(
                    "elaboration -> re-asking previous turn (\"{}\") in greater depth",
                    prev
                ));
            } else {
                needs_clarification = Some(
                    "I have nothing to elaborate on yet — could you ask a question first?".to_string(),
                );
            }
        } else if crate::session_memory::is_continuation(&raw_query) {
            // "what about it" -> resolve the bare pronoun to the prior
            // subject anchor; a fresh noun phrase ("what about cats") has
            // no bare pronoun and is left untouched for normal recall.
            if crate::session_memory::contains_bare_pronoun(&raw_query) {
                if let Some(anchor) = resolved_anchor.clone() {
                    effective_query =
                        crate::session_memory::substitute_pronoun(&raw_query, &anchor);
                    rewrite_note = Some(format!("continuation pronoun -> \"{}\"", anchor));
                }
            }
        } else if crate::session_memory::contains_bare_pronoun(&raw_query) {
            if let Some(anchor) = resolved_anchor {
                effective_query = crate::session_memory::substitute_pronoun(&raw_query, &anchor);
                rewrite_note = Some(format!("pronoun -> \"{}\"", anchor));
            } else {
                needs_clarification = Some(format!(
                    "I want to make sure I answer the right thing — could you tell me what you mean by the pronoun in \"{}\"? (No recent anchor in this conversation has a strong enough weight to substitute confidently.)",
                    raw_query
                ));
            }
        }
    }

    // Clarification short-circuit. Record as a turn so future
    // backreference queries see it.
    // -----------------------------------------------------------
    // OP-NEARMISS-1 -- spelling repair, applied AFTER the session
    // rewrites (so a substituted pronoun anchor is never re-spelled)
    // and BEFORE recall. Candidates come only from the keyword index,
    // so this can rename a token onto a term Lu KNOWS and can do
    // nothing else: it cannot invent a term, cannot invent a bond, and
    // does not touch a single K-gate threshold. Genuinely unknown words
    // have no repair target and reach the gate exactly as typed.
    // -----------------------------------------------------------
    if needs_clarification.is_none() {
        if let Some((repaired, reps)) =
            crate::near_miss::repair_query(brain.network.keyword_index_ref(), &effective_query)
        {
            let note = format!("spelling -> {}", crate::near_miss::describe(&reps));
            rewrite_note = Some(match rewrite_note.take() {
                Some(prev) => format!("{}; {}", prev, note),
                None => note,
            });
            effective_query = repaired;
        }
    }

    if let Some(msg) = needs_clarification {
        if let Some(sid) = session_id {
            let now = crate::brain::now_ms();
            let mem = brain.session_memories.get_or_create(sid, now);
            mem.record_turn(
                raw_query.clone(),
                raw_query.clone(),
                msg.clone(),
                Vec::new(),
                crate::session_memory::TurnTraceSummary::empty(),
                now,
            );
        }
        let structured = serde_json::json!({
            "session_id": session_id,
            "clarification_requested": true,
            "raw_query": raw_query,
        });
        return Ok(wrap_content(msg, structured));
    }

    let result = brain.narrate(&effective_query, depth, top_k, use_multi);
    let mut text = format!(
        "Narration ({}): {}",
        result.shape.label(),
        result.answer,
    );
    if let Some(note) = &rewrite_note {
        text.push_str(&format!("\n[rewrite] {}", note));
    }
    if !result.signals_used.is_empty() {
        text.push_str(&format!("
Signals: {}", result.signals_used.join(", ")));
    }

    // Post-pass: record this turn in session memory so a future
    // "tell me more" / pronoun query can see what we discussed. We
    // surface the activated content list (top up to 8) as anchors,
    // mirroring tool_ask.
    if let Some(sid) = session_id {
        let top_activations: Vec<String> = result
            .ask
            .recall
            .activated
            .iter()
            .take(top_k.min(8))
            .map(|a| a.content.clone())
            .collect();
        let now = crate::brain::now_ms();
        let mem = brain.session_memories.get_or_create(sid, now);
        mem.record_turn(
            raw_query.clone(),
            effective_query.clone(),
            text.clone(),
            top_activations,
            crate::session_memory::TurnTraceSummary::empty(),
            now,
        );
    }

    let mut structured = serde_json::to_value(&result).map_err(|e| (-32603, e.to_string()))?;
    if let Some(obj) = structured.as_object_mut() {
        if let Some(sid) = session_id {
            obj.insert("session_id".into(), serde_json::json!(sid));
            obj.insert("raw_query".into(), serde_json::json!(raw_query));
            obj.insert("effective_query".into(), serde_json::json!(effective_query));
            if let Some(note) = rewrite_note {
                obj.insert("rewrite_note".into(), serde_json::json!(note));
            }
        }
    }
    Ok(wrap_content(text, structured))
}

// =====================================================================
// Scar surface tools
// =====================================================================

/// Resolve a seed argument from either a hex string (preferred) or a content string
/// (resolved via keyword search to top-1 match). Used by safety-related tools so operators
/// don't have to look up hex seeds for everyday use.
fn resolve_seed(brain: &Brain, args: &Value, hex_key: &str, content_key: &str) -> Result<crate::seed::Seed, (i64, String)> {
    if let Some(hex) = args.get(hex_key).and_then(|v| v.as_str()) {
        return parse_seed_hex(hex);
    }
    if let Some(content) = args.get(content_key).and_then(|v| v.as_str()) {
        let hits = brain.network.keyword_search(content);
        return hits.first()
            .map(|(seed, _)| *seed)
            .ok_or((-32602, format!("no node found matching content {:?}", content)));
    }
    Err((-32602, format!("must provide either {:?} (64-char hex) or {:?} (content string)", hex_key, content_key)))
}

fn parse_seed_hex(hex: &str) -> Result<crate::seed::Seed, (i64, String)> {
    let cleaned: String = hex.chars().filter(|c| !c.is_whitespace()).collect();
    if cleaned.len() != 64 {
        return Err((-32602, format!("seed must be 64 hex chars; got {}", cleaned.len())));
    }
    let mut bytes = [0u8; 32];
    for i in 0..32 {
        let byte_str = &cleaned[i * 2..i * 2 + 2];
        bytes[i] = u8::from_str_radix(byte_str, 16)
            .map_err(|e| (-32602, format!("bad hex at byte {}: {}", i, e)))?;
    }
    Ok(crate::seed::Seed::from_bytes(bytes))
}

fn tool_scar_bond(brain: &mut Brain, args: Value) -> Result<Value, (i64, String)> {
    let from = resolve_seed(brain, &args, "from_seed", "from_content")?;
    let to = resolve_seed(brain, &args, "to_seed", "to_content")?;
    let reason = arg_str(&args, "reason")?.to_string();
    let scar = brain.scar_bond(from, to, reason);
    let text = format!(
        "Scarred {} -> {} (scar_id={}, reason: {})",
        &from.to_string()[..12],
        &to.to_string()[..12],
        scar.scar_id,
        scar.reason,
    );
    let structured = serde_json::to_value(&scar).map_err(|e| (-32603, e.to_string()))?;
    Ok(wrap_content(text, structured))
}

fn tool_scar_concept(brain: &mut Brain, args: Value) -> Result<Value, (i64, String)> {
    let seed = resolve_seed(brain, &args, "seed", "seed_content")?;
    let reason = arg_str(&args, "reason")?.to_string();
    let scars = brain.scar_concept(seed, reason);
    let text = format!(
        "Scarred concept {}: {} bond(s)",
        &seed.to_string()[..12],
        scars.len(),
    );
    let structured = serde_json::to_value(&scars).map_err(|e| (-32603, e.to_string()))?;
    Ok(wrap_content(text, structured))
}

fn tool_scar_log(brain: &mut Brain, args: Value) -> Result<Value, (i64, String)> {
    let since_ms = args.get("since_ms").and_then(|v| v.as_i64()).unwrap_or(i64::MIN);
    let scars: Vec<&crate::scar::Scar> = brain
        .scar_log
        .scars
        .iter()
        .filter(|s| s.created_ms >= since_ms)
        .collect();
    let unscars: Vec<&crate::scar::UnscarRecord> = brain
        .scar_log
        .unscars
        .iter()
        .filter(|u| u.unscarred_ms >= since_ms)
        .collect();
    let text = format!(
        "Scar log: {} active scar(s), {} unscar record(s) since ms={}",
        scars.len(),
        unscars.len(),
        if since_ms == i64::MIN { 0 } else { since_ms },
    );
    let structured = serde_json::json!({
        "scars": scars,
        "unscars": unscars,
    });
    Ok(wrap_content(text, structured))
}

fn tool_unscar_concept(brain: &mut Brain, args: Value) -> Result<Value, (i64, String)> {
    let seed = resolve_seed(brain, &args, "seed", "seed_content")?;
    let reason = arg_str(&args, "reason")?.to_string();
    let count = brain.unscar_concept(seed, reason.clone());
    let text = format!(
        "Unscarred concept {}: {} bond(s) cleared (reason: {})",
        &seed.to_string()[..12],
        count,
        reason,
    );
    let structured = serde_json::json!({
        "unscarred_bonds": count,
        "reason": reason,
    });
    Ok(wrap_content(text, structured))
}

fn tool_unscar_bond(brain: &mut Brain, args: Value) -> Result<Value, (i64, String)> {
    let from = resolve_seed(brain, &args, "from_seed", "from_content")?;
    let to = resolve_seed(brain, &args, "to_seed", "to_content")?;
    let reason = arg_str(&args, "reason")?.to_string();
    let count = brain.unscar_bond(from, to, reason.clone());
    let text = format!(
        "Unscarred {} -> {}: {} scar(s) removed (reason: {})",
        &from.to_string()[..12],
        &to.to_string()[..12],
        count,
        reason,
    );
    let structured = serde_json::json!({ "removed_count": count });
    Ok(wrap_content(text, structured))
}


// ── Port #10 — Autonomous Research Loop tool implementations ──

fn tool_research(brain: &mut Brain, args: Value) -> Result<Value, (i64, String)> {
    let query = arg_str(&args, "query")?.to_string();
    let fetcher = brain.default_research_fetcher();
    let id = brain
        .research_query(&fetcher, &query)
        .map_err(|e| (-32603, e.to_string()))?;
    let prop = brain
        .research_proposals
        .iter()
        .find(|p| p.id == id)
        .cloned()
        .ok_or((-32603, "proposal vanished after creation".to_string()))?;
    let text = format!(
        "Research proposal {} created (Pending). Candidates: {}, scar-blocked: {}.",
        id,
        prop.bond_candidates.len(),
        prop.scar_blocked_count
    );
    let structured = serde_json::to_value(&prop).map_err(|e| (-32603, e.to_string()))?;
    Ok(wrap_content(text, structured))
}

fn tool_research_loop_run(brain: &mut Brain, args: Value) -> Result<Value, (i64, String)> {
    // Port J.2: when `chain=true`, route through the chain driver and
    // surface the trace alongside the back-compat created_ids list.
    let chain = args.get("chain").and_then(|v| v.as_bool()).unwrap_or(false);
    if chain {
        let fetcher = brain.default_research_fetcher();
        let trace = brain.research_chain_run(&fetcher, None);
        let ids: Vec<u64> = trace.hops.iter().filter_map(|h| h.proposal_id).collect();
        let text = format!(
            "Chained research walked {} hop(s), {} proposal(s) queued, terminated: {:?}.",
            trace.hops.len(),
            trace.proposals_queued,
            trace.termination_reason
        );
        let structured = serde_json::json!({
            "created_ids": ids,
            "chain_trace": trace,
        });
        return Ok(wrap_content(text, structured));
    }
    let max_iterations = args
        .get("max_iterations")
        .and_then(|v| v.as_i64())
        .unwrap_or(1)
        .max(0) as usize;
    let fetcher = brain.default_research_fetcher();
    let ids = brain.research_loop_run(&fetcher, max_iterations);
    let text = format!(
        "Research loop produced {} proposal(s). Inspect with lu4d_research_proposals_list.",
        ids.len()
    );
    let structured = serde_json::json!({"created_ids": ids});
    Ok(wrap_content(text, structured))
}

fn tool_research_chain_run(brain: &mut Brain, args: Value) -> Result<Value, (i64, String)> {
    let seed = args
        .get("seed_query")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let fetcher = brain.default_research_fetcher();
    let trace = brain.research_chain_run(&fetcher, seed.as_deref());
    let text = format!(
        "Chain {}: {} hop(s), {} proposal(s) queued, terminated: {:?}.",
        trace.chain_id,
        trace.hops.len(),
        trace.proposals_queued,
        trace.termination_reason
    );
    let structured = serde_json::to_value(&trace).map_err(|e| (-32603, e.to_string()))?;
    Ok(wrap_content(text, structured))
}

fn tool_research_proposals_list(brain: &mut Brain, args: Value) -> Result<Value, (i64, String)> {
    let filter = args.get("status_filter").and_then(|v| v.as_str());
    let proposals: Vec<&crate::research::proposal::ResearchProposal> = brain
        .research_proposals
        .iter()
        .filter(|p| match filter {
            None => true,
            Some(f) => format!("{:?}", p.status).starts_with(f),
        })
        .collect();
    let text = format!("Returning {} proposal(s).", proposals.len());
    let structured =
        serde_json::to_value(&proposals).map_err(|e| (-32603, e.to_string()))?;
    Ok(wrap_content(text, structured))
}

fn tool_research_proposal_decide(brain: &mut Brain, args: Value) -> Result<Value, (i64, String)> {
    let id = args
        .get("id")
        .and_then(|v| v.as_u64())
        .ok_or((-32602, "missing integer arg 'id'".to_string()))?;
    let decision = arg_str(&args, "decision")?.to_string();
    let reason = args.get("reason").and_then(|v| v.as_str()).map(String::from);
    match decision.as_str() {
        "approve" => {
            let added = brain
                .approve_research_proposal(id)
                .map_err(|e| (-32603, e.to_string()))?;
            let text = format!(
                "Approved proposal {}. Bonds added: {}.",
                id, added
            );
            let structured = serde_json::json!({
                "id": id,
                "decision": "approve",
                "bonds_added": added,
            });
            Ok(wrap_content(text, structured))
        }
        "reject" => {
            brain
                .reject_research_proposal(id, reason.clone())
                .map_err(|e| (-32603, e.to_string()))?;
            let text = format!("Rejected proposal {}.", id);
            let structured = serde_json::json!({
                "id": id,
                "decision": "reject",
                "reason": reason,
            });
            Ok(wrap_content(text, structured))
        }
        other => Err((-32602, format!("decision must be 'approve' or 'reject', got '{}'", other))),
    }
}

// =====================================================================
// Reasoning surface tool (Port A — ReasoningAgent)
// =====================================================================

fn tool_reason(brain: &mut Brain, args: Value) -> Result<Value, (i64, String)> {
    let query = arg_str(&args, "query")?.to_string();
    // Port F (#576): arithmetic intent fires BEFORE reasoning.
    if let Some(resp) = try_arithmetic_response_with(Some(brain), &query) { return Ok(resp); }
    // Build the temporal scope from optional args. Default Current.
    let scope = build_reason_scope(brain, &args)?;
    // Build the reasoning config from optional args.
    let mut cfg = crate::reasoning::ReasoningConfig::default();
    if let Some(n) = args.get("max_chain_length").and_then(|v| v.as_u64()) {
        cfg.max_chain_length = n as usize;
    }
    if let Some(n) = args.get("min_chain_length").and_then(|v| v.as_u64()) {
        cfg.min_chain_length = n as usize;
    }
    if let Some(w) = args.get("min_edge_weight").and_then(|v| v.as_f64()) {
        cfg.min_edge_weight = w as f32;
    }
    let trace = crate::reasoning::reason_with_config(brain, &query, scope, &cfg);
    // Port J: aggregate evidence and articulate confidence band.
    let evidence_count = trace.bonds_used.len();
    let conf_band = crate::confidence::aggregate(
        evidence_count,
        trace.confidence,
        trace.no_chain,
    );
    // Port J.1: widened from {Single|Low} to also include Medium chains
    // whose trace.confidence is itself weak (< 0.55). This is the case
    // where evidence_count drags the band up to Medium but the chain's
    // composition is still topically thin (e.g. 'how do you write a for
    // loop in python' assembling RelatedTo across unrelated semantic
    // domains). Lower severity (0.40) than Single/Low so the wonder
    // queue still ranks them below the obvious gaps.
    let lowconf_severity: Option<f32> = match conf_band {
        crate::confidence::ConfidenceBand::Single => Some(0.55),
        crate::confidence::ConfidenceBand::Low => Some(0.65),
        crate::confidence::ConfidenceBand::Medium
            if trace.confidence < 0.55 => Some(0.40),
        _ => None,
    };
    if let Some(sev) = lowconf_severity {
        if !trace.no_chain {
            brain.gap_emitter.emit(crate::gap_signal::GapSignal {
                query: query.clone(),
                port_origin: "J",
                gap_kind: crate::gap_signal::GapKind::LowConfidence,
                severity: sev,
                context: serde_json::json!({
                    "evidence_count": evidence_count,
                    "trace_confidence": trace.confidence,
                    "band": conf_band.label(),
                }),
            });
        }
    }
    let header = if trace.no_chain {
        format!("Reasoning (no chain): {}", trace.final_answer)
    } else {
        format!(
            "Reasoning ({} steps, conf={:.3}, evidence={}, band={}):\n{}",
            trace.steps.len(),
            trace.confidence,
            evidence_count,
            conf_band.label(),
            trace.final_answer
        )
    };
    let mut structured = serde_json::to_value(&trace).map_err(|e| (-32603, e.to_string()))?;
    if let Some(obj) = structured.as_object_mut() {
        obj.insert(
            "confidence_band".to_string(),
            serde_json::Value::String(conf_band.label().to_string()),
        );
        obj.insert(
            "evidence_count".to_string(),
            serde_json::Value::Number(serde_json::Number::from(evidence_count)),
        );
        obj.insert(
            "confidence_articulation".to_string(),
            serde_json::Value::String(conf_band.articulate(evidence_count, trace.confidence)),
        );
        obj.insert(
            "gaps_emitted_this_call".to_string(),
            serde_json::Value::Number(serde_json::Number::from(brain.gap_emitter.pending_len() as u64)),
        );
    }
    let _flushed = brain.flush_gaps();
    Ok(wrap_content(header, structured))
}

/// Translate the optional `scope` / `at_ms` args into a `TemporalScope`.
/// Unknown scope strings produce a -32602 error so the caller learns
/// the supported set ('current' | 'historical' | 'session').
fn build_reason_scope(brain: &Brain, args: &Value) -> Result<crate::scope::TemporalScope, (i64, String)> {
    let kind = args
        .get("scope")
        .and_then(|v| v.as_str())
        .unwrap_or("current")
        .to_lowercase();
    match kind.as_str() {
        "current" => Ok(crate::scope::TemporalScope::Current),
        "historical" => {
            let at = args
                .get("at_ms")
                .and_then(|v| v.as_i64())
                .ok_or((-32602, "scope='historical' requires at_ms".to_string()))?;
            Ok(crate::scope::TemporalScope::Historical(at))
        }
        "session" => match brain.sessions.active_session() {
            Some(s) => Ok(crate::scope::TemporalScope::Session {
                id: s.id,
                start: s.start_time,
            }),
            None => Err((-32602, "scope='session' requires an active session".to_string())),
        },
        other => Err((-32602, format!("unknown scope {:?}; expected 'current'|'historical'|'session'", other))),
    }
}
// =====================================================================
// Port F (#576) — integration tests for arithmetic routing in tool_ask
// /tool_reason. These exercise `try_arithmetic_response` directly: that
// function is the single gate both surfaces consult before falling
// through to recall/reasoning, so testing it covers both call sites.
// =====================================================================
#[cfg(test)]
mod portf_arith_routing_tests {
    use super::*;

    fn structured(v: &Value) -> &Value {
        v.get("structuredContent").expect("structuredContent")
    }

    fn first_text(v: &Value) -> &str {
        v.get("content")
            .and_then(|c| c.get(0))
            .and_then(|t| t.get("text"))
            .and_then(|t| t.as_str())
            .expect("content[0].text")
    }

    #[test]
    fn ask_routes_arithmetic_through_math_primitive() {
        // "what is 7 plus 5" must produce a response flagged as math
        // and the rendered text must contain the answer (12). This is
        // the exact baseline-#598 query that previously returned
        // "Serge Gainsbourg" via spreading activation.
        let resp = try_arithmetic_response("what is 7 plus 5")
            .expect("arithmetic should fire");
        assert_eq!(
            structured(&resp).get("source").and_then(|v| v.as_str()),
            Some("arithmetic_primitive")
        );
        let text = first_text(&resp);
        assert!(text.contains("12"), "text should state the result, got {:?}", text);
        assert!(
            text.to_lowercase().contains("computed"),
            "text should advertise provenance, got {:?}",
            text
        );
    }

    #[test]
    fn ask_does_not_route_non_arithmetic() {
        // The classic recall query must NOT be intercepted — this is
        // what guarantees Port A/B/C/D/E happy paths still fire.
        assert!(try_arithmetic_response("what is mitochondria").is_none());
    }

    #[test]
    fn ask_does_not_route_mixed_query() {
        // Mixed queries with one numeric and one symbolic operand must
        // fall through to recall, not be coerced into a math result.
        assert!(try_arithmetic_response("what is 7 plus mitochondria").is_none());
    }

    #[test]
    fn ask_routes_percent_of() {
        let resp = try_arithmetic_response("what is 25 percent of 80")
            .expect("percent should fire");
        let text = first_text(&resp);
        assert!(text.contains("20"), "got {:?}", text);
    }

    #[test]
    fn ask_routes_square_root() {
        let resp = try_arithmetic_response("what is the square root of 16")
            .expect("sqrt should fire");
        let text = first_text(&resp);
        assert!(text.contains("4"), "got {:?}", text);
    }

    #[test]
    fn ask_routes_comparison() {
        let resp = try_arithmetic_response("is 100 greater than 50")
            .expect("comparison should fire");
        let text = first_text(&resp);
        assert!(text.to_lowercase().contains("yes"), "got {:?}", text);
    }

    #[test]
    fn ask_routes_half_of() {
        let resp = try_arithmetic_response("what is half of 200")
            .expect("half-of should fire");
        let text = first_text(&resp);
        assert!(text.contains("100"), "got {:?}", text);
    }

    #[test]
    fn ask_handles_division_by_zero_via_primitive() {
        // Even on error the response must be flagged as arithmetic so
        // callers can tell math FIRED rather than recall having missed.
        let resp = try_arithmetic_response("what is 5 divided by 0")
            .expect("div-by-zero query should still fire math");
        assert_eq!(
            structured(&resp).get("source").and_then(|v| v.as_str()),
            Some("arithmetic_primitive")
        );
        let text = first_text(&resp);
        assert!(
            text.to_lowercase().contains("cannot compute"),
            "got {:?}",
            text
        );
    }
// =====================================================================
    // Port F.2 — integration tests for the expression primitive routing
    // through `try_arithmetic_response`. These exercise the F.2 path end
    // to end: detection -> compute -> structuredContent shape with
    // `source: "expression_primitive"`.
    // =====================================================================

    #[test]
    fn f2_expression_routes_through_expression_primitive() {
        let resp = try_arithmetic_response("what is 2 + 3 * 4")
            .expect("expression query should fire math");
        assert_eq!(
            resp.get("structuredContent")
                .and_then(|s| s.get("source"))
                .and_then(|s| s.as_str()),
            Some("expression_primitive")
        );
        let text = resp
            .pointer("/content/0/text")
            .and_then(|t| t.as_str())
            .unwrap();
        assert!(text.contains("= 14"), "got {:?}", text);
        assert!(text.to_lowercase().contains("computed"));
    }

    #[test]
    fn f2_paren_grouping_routes_through_expression_primitive() {
        let resp = try_arithmetic_response("(7 + 5) * (3 - 1)").unwrap();
        let text = resp
            .pointer("/content/0/text")
            .and_then(|t| t.as_str())
            .unwrap();
        assert!(text.contains("= 24"), "got {:?}", text);
    }

    #[test]
    fn f2_function_call_routes_through_expression_primitive() {
        let resp = try_arithmetic_response("sqrt(144)").unwrap();
        let text = resp
            .pointer("/content/0/text")
            .and_then(|t| t.as_str())
            .unwrap();
        assert!(text.contains("= 12"), "got {:?}", text);
    }

    #[test]
    fn f2_constant_pi_routes_through_expression_primitive() {
        let resp = try_arithmetic_response("2 * pi").unwrap();
        let text = resp
            .pointer("/content/0/text")
            .and_then(|t| t.as_str())
            .unwrap();
        // Loose check — π is irrational, just confirm the prefix and
        // that it's not the integer 6.
        assert!(text.starts_with("2 * pi = 6.28"), "got {:?}", text);
    }

    #[test]
    fn f2_unit_conversion_routes_through_expression_primitive() {
        let resp = try_arithmetic_response("convert 100 km to miles").unwrap();
        assert_eq!(
            resp.get("structuredContent")
                .and_then(|s| s.get("source"))
                .and_then(|s| s.as_str()),
            Some("expression_primitive")
        );
        let text = resp
            .pointer("/content/0/text")
            .and_then(|t| t.as_str())
            .unwrap();
        assert!(
            text.contains("miles") && text.contains("62.13"),
            "got {:?}",
            text
        );
    }

    #[test]
    fn f2_seconds_in_a_day_routes_through_expression_primitive() {
        let resp = try_arithmetic_response("how many seconds in a day").unwrap();
        let text = resp
            .pointer("/content/0/text")
            .and_then(|t| t.as_str())
            .unwrap();
        assert!(text.contains("86400"), "got {:?}", text);
    }

    #[test]
    fn f2_temperature_routes_through_expression_primitive() {
        let resp = try_arithmetic_response("convert 32 F to C").unwrap();
        let text = resp
            .pointer("/content/0/text")
            .and_then(|t| t.as_str())
            .unwrap();
        assert!(text.contains("= 0"), "got {:?}", text);
    }

    #[test]
    fn f2_stat_mean_routes_through_expression_primitive() {
        let resp = try_arithmetic_response("mean of 1, 2, 3, 4, 5").unwrap();
        let text = resp
            .pointer("/content/0/text")
            .and_then(|t| t.as_str())
            .unwrap();
        assert!(text.contains("= 3"), "got {:?}", text);
    }

    #[test]
    fn f2_solver_routes_through_expression_primitive() {
        let resp = try_arithmetic_response("solve 2x = 14").unwrap();
        let text = resp
            .pointer("/content/0/text")
            .and_then(|t| t.as_str())
            .unwrap();
        assert!(text.contains("x = 7"), "got {:?}", text);
    }

    #[test]
    fn f2_word_number_arithmetic_routes_through_legacy() {
        // Word-number arithmetic is handled by the F.1 detector via
        // its now-extended parse_number — the source field stays
        // `arithmetic_primitive` even though the operands are words.
        let resp = try_arithmetic_response("twelve plus thirteen").unwrap();
        assert_eq!(
            resp.get("structuredContent")
                .and_then(|s| s.get("source"))
                .and_then(|s| s.as_str()),
            Some("arithmetic_primitive")
        );
        let text = resp
            .pointer("/content/0/text")
            .and_then(|t| t.as_str())
            .unwrap();
        assert!(text.contains("= 25"), "got {:?}", text);
    }

    #[test]
    fn f2_concept_query_still_falls_through_to_recall() {
        // The whole point of math routing: concept queries must NOT
        // be hijacked by either path.
        assert!(try_arithmetic_response("what is mitochondria").is_none());
        assert!(try_arithmetic_response("who invented the transistor").is_none());
    }
// ---- F.4 construction integration tests ----------------------
    // These exercise the construction cascade end-to-end through
    // `try_arithmetic_response`, asserting the route, the response
    // shape, and the (construction calculator) flag in the prose.

    fn f4_text(query: &str) -> String {
        let resp = try_arithmetic_response(query).expect("expected math response");
        assert_eq!(
            resp.get("structuredContent")
                .and_then(|s| s.get("source"))
                .and_then(|s| s.as_str()),
            Some("construction_primitive"),
            "query {:?} did not route to construction_primitive",
            query,
        );
        resp.pointer("/content/0/text")
            .and_then(|t| t.as_str())
            .map(String::from)
            .unwrap()
    }

    #[test]
    fn f4_fi_addition_routes_through_construction_primitive() {
        let text = f4_text("12'-3 1/2\" + 4'-7 1/4\"");
        assert!(text.contains("16'-10 3/4\""), "got {:?}", text);
        assert!(text.contains("(construction calculator)"), "got {:?}", text);
    }

    #[test]
    fn f4_board_feet_routes_through_construction_primitive() {
        let text = f4_text("board feet of 2x4x8");
        assert!(text.contains("5.33"), "got {:?}", text);
        assert!(text.contains("(construction calculator)"));
    }

    #[test]
    fn f4_stud_count_routes_through_construction_primitive() {
        let text = f4_text("studs in a 32 foot wall at 16 OC");
        assert!(text.contains("25 studs"), "got {:?}", text);
    }

    #[test]
    fn f4_sheet_goods_routes_through_construction_primitive() {
        let text = f4_text("sheets for a 32 by 8 wall");
        assert!(text.contains("8 sheets"), "got {:?}", text);
    }

    #[test]
    fn f4_rafter_routes_through_construction_primitive() {
        let text = f4_text("rafter for 12 foot run 6 foot rise");
        assert!(text.contains("13'-5\"") || text.contains("13'-4"), "got {:?}", text);
    }

    #[test]
    fn f4_pitch_routes_through_construction_primitive() {
        let text = f4_text("slope angle of 6/12 pitch");
        assert!(text.contains("26.57"), "got {:?}", text);
    }

    #[test]
    fn f4_concrete_routes_through_construction_primitive() {
        let text = f4_text("concrete for a 20 by 10 slab 4 inches thick");
        assert!(text.contains("2.47"), "got {:?}", text);
    }

    #[test]
    fn f4_stairs_routes_through_construction_primitive() {
        let text = f4_text("stairs for 108 inch rise");
        assert!(text.contains("14 risers"), "got {:?}", text);
    }

    #[test]
    fn f4_does_not_hijack_plain_expression() {
        // "2 + 3 * 4" must still route to F.2 expression_primitive,
        // NOT to construction.
        let resp = try_arithmetic_response("2 + 3 * 4").unwrap();
        assert_eq!(
            resp.get("structuredContent")
                .and_then(|s| s.get("source"))
                .and_then(|s| s.as_str()),
            Some("expression_primitive"),
        );
    }

    #[test]
    fn f4_does_not_hijack_unit_conversion() {
        // "convert 100 km to miles" still routes to F.2.
        let resp = try_arithmetic_response("convert 100 km to miles").unwrap();
        assert_eq!(
            resp.get("structuredContent")
                .and_then(|s| s.get("source"))
                .and_then(|s| s.as_str()),
            Some("expression_primitive"),
        );
    }

    #[test]
    fn f4_concept_query_still_falls_through() {
        assert!(try_arithmetic_response("what is mitochondria").is_none());
    }

    // ============================================================
    // F.3 integration tests — word-problem extraction through the
    // try_arithmetic_response cascade. These exercise the F.3 slot
    // (after F.1/F.4, before F.2) end-to-end.
    // ============================================================

    fn f3_resp(query: &str) -> serde_json::Value {
        try_arithmetic_response(query).expect("expected math response")
    }

    fn f3_source(resp: &serde_json::Value) -> Option<&str> {
        resp.get("structuredContent")
            .and_then(|s| s.get("source"))
            .and_then(|s| s.as_str())
    }

    fn f3_confidence(resp: &serde_json::Value) -> Option<&str> {
        resp.get("structuredContent")
            .and_then(|s| s.get("confidence"))
            .and_then(|s| s.as_str())
    }

    fn f3_text(resp: &serde_json::Value) -> String {
        resp.pointer("/content/0/text")
            .and_then(|t| t.as_str())
            .map(String::from)
            .unwrap()
    }

    #[test]
    fn f3_sarah_apples_routes_through_wordprob() {
        let resp = f3_resp("Sarah had 5 apples. She gave 2 to Mary. How many does Sarah have?");
        assert_eq!(f3_source(&resp), Some("wordprob_primitive"));
        assert_eq!(f3_confidence(&resp), Some("confident"));
        let text = f3_text(&resp);
        assert!(text.contains("3"), "got {:?}", text);
        assert!(text.contains("computed"), "got {:?}", text);
    }

    #[test]
    fn f3_bob_cookies_routes_through_wordprob() {
        let resp = f3_resp("Bob has 7 cookies. He buys 5 more. How many cookies does Bob have?");
        assert_eq!(f3_source(&resp), Some("wordprob_primitive"));
        let text = f3_text(&resp);
        assert!(text.contains("12"), "got {:?}", text);
    }

    #[test]
    fn f3_boxes_of_eggs_routes_through_wordprob() {
        let resp = f3_resp("A box has 12 eggs. There are 5 boxes. How many eggs total?");
        assert_eq!(f3_source(&resp), Some("wordprob_primitive"));
        let text = f3_text(&resp);
        assert!(text.contains("60"), "got {:?}", text);
    }

    #[test]
    fn f3_lisa_earnings_routes_through_wordprob() {
        let resp = f3_resp(
            "Lisa earns $15 per hour. She works 8 hours. How much does she earn?",
        );
        assert_eq!(f3_source(&resp), Some("wordprob_primitive"));
        let text = f3_text(&resp);
        assert!(text.contains("120"), "got {:?}", text);
    }

    #[test]
    fn f3_percent_off_routes_through_wordprob() {
        let resp = f3_resp("A shirt costs $20. It is 25% off. What is the sale price?");
        assert_eq!(f3_source(&resp), Some("wordprob_primitive"));
        let text = f3_text(&resp);
        assert!(text.contains("15"), "got {:?}", text);
    }

    #[test]
    fn f3_bat_and_ball_hedges_through_wordprob() {
        let resp = f3_resp(
            "A bat and a ball cost $1.10. The bat costs $1 more than the ball. How much is the ball?",
        );
        assert_eq!(f3_source(&resp), Some("wordprob_primitive"));
        assert_eq!(f3_confidence(&resp), Some("hedged"));
        let text = f3_text(&resp);
        assert!(text.contains("hedged") || text.contains("verify"), "got {:?}", text);
    }

    #[test]
    fn f3_does_not_hijack_plain_arithmetic() {
        // "what is 7 plus 5" is F.1 single-op territory.
        let resp = f3_resp("what is 7 plus 5");
        assert_eq!(f3_source(&resp), Some("arithmetic_primitive"));
    }

    #[test]
    fn f3_does_not_hijack_expression() {
        // "2 + 3 * 4" must still route to F.2 expression_primitive.
        let resp = f3_resp("2 + 3 * 4");
        assert_eq!(f3_source(&resp), Some("expression_primitive"));
    }

    #[test]
    fn f3_does_not_hijack_unit_conversion() {
        let resp = f3_resp("convert 100 km to miles");
        assert_eq!(f3_source(&resp), Some("expression_primitive"));
    }

    #[test]
    fn f3_does_not_hijack_construction() {
        let resp = f3_resp("board feet of 2x4x8");
        assert_eq!(f3_source(&resp), Some("construction_primitive"));
    }

    #[test]
    fn f3_concept_query_still_falls_through() {
        assert!(try_arithmetic_response("what is mitochondria").is_none());
    }


    // ===== Q.1 typed-KU MCP integration tests ======================
    #[test]
    fn q1_mcp_teach_with_explicit_kind() {
        let mut b = Brain::new();
        let args = serde_json::json!({
            "subject": "photosynthesis",
            "relation": "DefinedAs",
            "object": "conversion of light to chemical energy",
            "kind": "Definition",
        });
        let v = tool_teach(&mut b, args).expect("tool_teach ok");
        let s = serde_json::to_string(&v).unwrap();
        assert!(s.contains("kind=definition"), "missing kind tag: {}", s);
        let seed = crate::seed::Seed::from_label("photosynthesis");
        let n = b.network.get_node(&seed).expect("node exists");
        assert_eq!(n.kind_str(), "definition");
    }

    #[test]
    fn q1_mcp_teach_pattern_driven_when_kind_omitted() {
        let mut b = Brain::new();
        let args = serde_json::json!({
            "subject": "democracy",
            "relation": "DefinedAs",
            "object": "rule by the people",
        });
        let v = tool_teach(&mut b, args).expect("tool_teach ok");
        let s = serde_json::to_string(&v).unwrap();
        assert!(s.contains("kind=definition"), "expected pattern-driven definition: {}", s);
    }

    #[test]
    fn q1_mcp_teach_unknown_kind_rejected() {
        let mut b = Brain::new();
        let args = serde_json::json!({
            "subject": "x",
            "relation": "IsA",
            "object": "y",
            "kind": "NotAVariant",
        });
        let err = tool_teach(&mut b, args).err().expect("should reject");
        assert_eq!(err.0, -32602);
        assert!(err.1.contains("unknown kind"), "got: {}", err.1);
    }

    #[test]
    fn q1_mcp_teach_default_path_unchanged_for_canonical_relation() {
        // BR.3: IsA now infers to `definition` (X is a Y is the
        // canonical definitional bond). Pre-BR.3 this routed to
        // Concept; the rebuild widened `infer_kind_from_relation` so
        // bonds carry semantic-typed nodes by default. The Q.1
        // explicit-kind override path is still pinned by
        // `q1_mcp_teach_explicit_kind_routes_to_definition` below.
        let mut b = Brain::new();
        let args = serde_json::json!({
            "subject": "cat",
            "relation": "IsA",
            "object": "mammal",
        });
        let v = tool_teach(&mut b, args).expect("tool_teach ok");
        let s = serde_json::to_string(&v).unwrap();
        assert!(
            s.contains("kind=definition"),
            "expected IsA to infer kind=definition (BR.3 mapping): {}",
            s,
        );
        // Sanity: the bare-Concept fallback is still reachable for
        // truly generic associations like RelatedTo.
        let mut b2 = Brain::new();
        let args2 = serde_json::json!({
            "subject": "alpha",
            "relation": "RelatedTo",
            "object": "beta",
        });
        let v2 = tool_teach(&mut b2, args2).expect("tool_teach ok");
        let s2 = serde_json::to_string(&v2).unwrap();
        assert!(
            s2.contains("kind=concept"),
            "RelatedTo must still default to concept: {}",
            s2,
        );
    }

    #[test]
    fn q1_mcp_teach_composedof_alias_routes_to_composedof_with_composition_kind() {
        let mut b = Brain::new();
        let args = serde_json::json!({
            "subject": "water",
            "relation": "ComposedOf",
            "object": "hydrogen",
        });
        let v = tool_teach(&mut b, args).expect("tool_teach ok");
        let s = serde_json::to_string(&v).unwrap();
        assert!(s.contains("ComposedOf") || s.contains("composed_of"), "bond should route to ComposedOf");
        assert!(s.contains("kind=composition"));
    }

    #[test]
    fn q1_mcp_teach_code_pattern_from_text() {
        let mut b = Brain::new();
        let args = serde_json::json!({
            "subject": "def hello():\n    pass",
            "relation": "RelatedTo",
            "object": "python",
        });
        let v = tool_teach(&mut b, args).expect("tool_teach ok");
        let s = serde_json::to_string(&v).unwrap();
        assert!(s.contains("kind=code") || s.contains("code,"), "expected code kind: {}", s);
    }

    #[test]
    fn q1_mcp_teach_returns_structured_kind_field() {
        let mut b = Brain::new();
        let args = serde_json::json!({
            "subject": "engine",
            "relation": "WorksBy",
            "object": "combustion",
        });
        let v = tool_teach(&mut b, args).expect("tool_teach ok");
        // structuredContent (under content[1].structured) should carry
        // subject_kind / object_kind. Easier: search the serialized blob.
        let s = serde_json::to_string(&v).unwrap();
        assert!(s.contains("\"subject_kind\":\"mechanism\""), "missing subject_kind: {}", s);
        assert!(s.contains("\"object_kind\":\"mechanism\""), "missing object_kind: {}", s);
    }

    #[test]
    fn q1_mcp_teach_existing_node_keeps_prior_kind() {
        let mut b = Brain::new();
        // Seed with definition.
        let _ = tool_teach(&mut b, serde_json::json!({
            "subject": "democracy",
            "relation": "DefinedAs",
            "object": "rule by people",
        })).unwrap();
        // Reteach with no kind, generic IsA — must NOT downgrade.
        let v = tool_teach(&mut b, serde_json::json!({
            "subject": "democracy",
            "relation": "IsA",
            "object": "governance",
        })).unwrap();
        let s = serde_json::to_string(&v).unwrap();
        // democracy stays definition; new "governance" node is concept.
        assert!(s.contains("\"subject_kind\":\"definition\""), "democracy should stay definition: {}", s);
    }

    // =====================================================================
    // OP-EXTRACT-PREP -- Phase A.1 tests. Verifies the `negated` flag
    // is exposed on the lu4d_teach, lu4d_teach_corpus, and
    // lu4d_teach_corpus_batch MCP surfaces and flows through to the
    // NegationOverlay. Phase A.2 (widening the bond dedup key to
    // include BondType) was halted -- see the OP-EXTRACT-PREP report.
    // =====================================================================

    #[test]
    fn op_extract_prep_teach_accepts_negated_field() {
        let mut b = Brain::new();
        let args = serde_json::json!({
            "subject": "Lincoln",
            "relation": "IsA",
            "object": "fish",
            "negated": true,
        });
        let v = tool_teach(&mut b, args).expect("tool_teach ok");
        let subj = crate::seed::Seed::from_label("Lincoln");
        let obj = crate::seed::Seed::from_label("fish");
        assert!(
            b.bond_negated(&subj, &obj, BondType::IsA),
            "negation overlay flag not set",
        );
        let serialized = serde_json::to_string(&v).unwrap();
        assert!(
            serialized.contains("\"negated\":true"),
            "negated flag missing from structured response: {}",
            serialized,
        );
    }

    #[test]
    fn op_extract_prep_teach_corpus_accepts_negated_field() {
        let mut b = Brain::new();
        let args = serde_json::json!({
            "subject": "Mars",
            "relation": "HasProperty",
            "object": "blue oceans",
            "corpus": "test/op-extract-prep",
            "negated": true,
        });
        let v = tool_teach_corpus(&mut b, args).expect("tool_teach_corpus ok");
        let subj = crate::seed::Seed::from_label("Mars");
        let obj = crate::seed::Seed::from_label("blue oceans");
        assert!(
            b.bond_negated(&subj, &obj, BondType::HasProperty),
            "negation overlay flag not set on corpus path",
        );
        let serialized = serde_json::to_string(&v).unwrap();
        assert!(
            serialized.contains("\"negated\":true"),
            "negated flag missing from corpus structured response: {}",
            serialized,
        );
        assert!(
            serialized.contains("\"kind\":\"TrainingCorpus\""),
            "TrainingCorpus stamp lost: {}",
            serialized,
        );
    }

    #[test]
    fn op_extract_prep_teach_corpus_negated_defaults_to_false() {
        let mut b = Brain::new();
        let args = serde_json::json!({
            "subject": "Earth",
            "relation": "HasProperty",
            "object": "water",
            "corpus": "test/op-extract-prep",
        });
        let v = tool_teach_corpus(&mut b, args).expect("tool_teach_corpus ok");
        let subj = crate::seed::Seed::from_label("Earth");
        let obj = crate::seed::Seed::from_label("water");
        assert!(
            !b.bond_negated(&subj, &obj, BondType::HasProperty),
            "default-false negation flag should stay clear",
        );
        let serialized = serde_json::to_string(&v).unwrap();
        assert!(
            serialized.contains("\"negated\":false"),
            "default-false should still surface in structured response: {}",
            serialized,
        );
    }

    #[test]
    fn op_extract_prep_teach_corpus_batch_propagates_negated_per_entry() {
        // Each entry's `negated` is honoured independently. The bonds in
        // the test use distinct (subj, obj) pairs so we side-step the
        // pair-only network-bond dedup (the limitation surfaced by
        // Phase A.2 audit; see OP-EXTRACT-PREP halt report).
        let mut b = Brain::new();
        let args = serde_json::json!({
            "entries": [
                {
                    "subject": "Pluto",
                    "relation": "IsA",
                    "object": "planet",
                    "corpus": "test/op-extract-prep",
                    "negated": true,
                },
                {
                    "subject": "Earth",
                    "relation": "IsA",
                    "object": "planet",
                    "corpus": "test/op-extract-prep",
                },
            ]
        });
        let _v = tool_teach_corpus_batch(&mut b, args).expect("batch ok");
        let pluto = crate::seed::Seed::from_label("Pluto");
        let planet = crate::seed::Seed::from_label("planet");
        assert!(
            b.bond_negated(&pluto, &planet, BondType::IsA),
            "Pluto-IsA-planet overlay should reflect negated=true",
        );
        let earth = crate::seed::Seed::from_label("Earth");
        assert!(
            !b.bond_negated(&earth, &planet, BondType::IsA),
            "Earth-IsA-planet overlay should stay false",
        );
    }

    #[test]
    fn op_extract_prep_teach_corpus_gated_negated_threaded() {
        // The gated path used to hardcode `negated: false` in the queued
        // PendingChangeKind::AddBond; the caller-supplied flag must now
        // thread through so approve/reject parity holds.
        let mut b = Brain::new();
        b.ingest_config.auto_approve_threshold = Some(0.5);
        let args = serde_json::json!({
            "subject": "Atlantis",
            "relation": "IsA",
            "object": "real city",
            "confidence": "Unverified",
            "corpus": "test/op-extract-prep",
            "negated": true,
        });
        let v = tool_teach_corpus(&mut b, args).expect("tool_teach_corpus ok");
        let serialized = serde_json::to_string(&v).unwrap();
        assert!(
            serialized.contains("\"gated\":true"),
            "expected gated path: {}",
            serialized,
        );
        assert!(
            serialized.contains("\"negated\":true"),
            "gated path should echo negated flag: {}",
            serialized,
        );
        let pending: Vec<_> = b.pending.iter().collect();
        assert_eq!(pending.len(), 1, "expected single pending entry");
        match &pending[0].1.change_kind {
            crate::pending::PendingChangeKind::AddBond { negated, .. } => {
                assert!(*negated, "PendingChange should carry negated=true");
            }
            other => panic!("expected AddBond change, got {:?}", other),
        }
    }

    // =====================================================================
    // OP-ROUTE-1 \u2014 tool_narrate parity tests. Verifies the math
    // short-circuit added at line 2911 fires on the narrate surface the
    // same way it already fires on tool_ask (line 2085) and tool_reason
    // (line 3195). Before OP-ROUTE-1, a math query phrased through
    // lu4d_narrate fell through to spread activation and surfaced
    // keyword-overlap noise (e.g. \"what is two plus two?\" returned
    // arch-bridge content per the 2026-05-26 canonical conversation
    // test).
    // =====================================================================

    #[test]
    fn tool_narrate_arithmetic_short_circuit() {
        let mut b = Brain::new();
        let resp = tool_narrate(&mut b, serde_json::json!({
            "query": "what is 7 plus 5?"
        })).expect("tool_narrate should succeed");
        // OP-ROUTE-1 option (a): math returns the unchanged arithmetic
        // envelope (matches tool_ask / tool_reason). source MUST be the
        // primitive tag, NOT a NarrationResult shape.
        assert_eq!(
            structured(&resp).get("source").and_then(|v| v.as_str()),
            Some("arithmetic_primitive"),
            "narrate should return arithmetic envelope unchanged, got {:?}",
            resp
        );
        let text = first_text(&resp);
        assert!(text.contains("12"), "text should state the result, got {:?}", text);
        assert!(
            text.to_lowercase().contains("computed"),
            "text should advertise provenance, got {:?}",
            text
        );
    }

    #[test]
    fn tool_narrate_arithmetic_skips_recall() {
        // Math short-circuit must return BEFORE narrate() runs. The
        // arithmetic envelope has no `shape`, no `ask`, no `multi`,
        // no `signals_used` \u2014 those are NarrationResult fields. Their
        // absence is the empirical proof the short-circuit fired.
        let mut b = Brain::new();
        let resp = tool_narrate(&mut b, serde_json::json!({
            "query": "what is 7 plus 5?"
        })).unwrap();
        let sc = structured(&resp);
        assert!(sc.get("shape").is_none(), "arithmetic response must not carry NarrationResult.shape, got {:?}", sc);
        assert!(sc.get("ask").is_none(), "arithmetic response must not carry NarrationResult.ask, got {:?}", sc);
        assert!(sc.get("multi").is_none(), "arithmetic response must not carry NarrationResult.multi, got {:?}", sc);
        assert!(sc.get("signals_used").is_none(), "arithmetic response must not carry NarrationResult.signals_used, got {:?}", sc);
    }

    #[test]
    fn tool_narrate_non_math_unchanged() {
        // Non-math queries must NOT be short-circuited \u2014 they flow
        // through the existing narrate pipeline and surface a real
        // NarrationResult shape. The empty brain returns a Refusal
        // shape (idk_gate) rather than fabricating; either way it
        // must NOT be the arithmetic envelope.
        let mut b = Brain::new();
        let resp = tool_narrate(&mut b, serde_json::json!({
            "query": "what is photosynthesis?"
        })).unwrap();
        let sc = structured(&resp);
        // Source field, if present, is NOT a math primitive tag.
        let src = sc.get("source").and_then(|v| v.as_str()).unwrap_or("");
        assert!(
            !src.ends_with("_primitive"),
            "non-math query must not be coerced into arithmetic envelope, got source={:?}",
            src
        );
        // NarrationResult shape MUST be present.
        assert!(
            sc.get("shape").is_some(),
            "non-math narrate must return a NarrationResult with shape, got {:?}",
            sc
        );
    }

    #[test]
    fn tool_narrate_arithmetic_parity_with_ask() {
        // Same math query through tool_ask and tool_narrate must produce
        // the same computed value and provenance \u2014 they consult the
        // same gate.
        let mut b1 = Brain::new();
        let mut b2 = Brain::new();
        let resp_ask = tool_ask(&mut b1, serde_json::json!({
            "query": "what is 7 plus 5?"
        })).unwrap();
        let resp_narr = tool_narrate(&mut b2, serde_json::json!({
            "query": "what is 7 plus 5?"
        })).unwrap();
        // Same source.
        assert_eq!(
            structured(&resp_ask).get("source"),
            structured(&resp_narr).get("source"),
            "ask and narrate must use the same primitive source"
        );
        // Same numeric value (field present on F.1 arithmetic_primitive).
        assert_eq!(
            structured(&resp_ask).get("value"),
            structured(&resp_narr).get("value"),
            "ask and narrate must compute the same value"
        );
        // Same prose (the F.1 path generates identical prose).
        assert_eq!(
            structured(&resp_ask).get("prose"),
            structured(&resp_narr).get("prose"),
            "ask and narrate must generate identical math prose"
        );
    }
}


#[cfg(test)]
mod q3_session_memory_integration_tests {
    //! Port Q.3 (#438 / #587) — integration coverage on tool_ask
    //! routed through a Brain populated with a small synthetic
    //! line-graph. We exercise the full pipeline (memory pre-pass,
    //! recall, post-pass) WITHOUT touching the canonical brain.
    use super::*;
    use crate::bond::{BondType, TemporalBond};
    use crate::brain::Brain;
    use crate::node::{Confidence, KnowledgeContent, KnowledgeNode};
    use crate::seed::Seed;

    fn fixture_brain() -> Brain {
        // photosynthesis -[Produces]-> glucose -[Produces]-> energy
        // photosynthesis -[Produces]-> oxygen
        // paris -[IsA]-> city -[HasProperty]-> capital
        let mut b = Brain::new();
        let labels = ["photosynthesis", "glucose", "energy", "oxygen", "paris", "city", "capital"];
        for label in labels {
            let s = Seed::from_label(label);
            b.network.add_node(KnowledgeNode::with_seed(
                s, label.into(),
                KnowledgeContent::Concept(label.into()),
                Confidence::Verified, 0,
            ));
        }
        let bonds = [
            ("photosynthesis", "glucose", BondType::Produces),
            ("glucose", "energy", BondType::Produces),
            ("photosynthesis", "oxygen", BondType::Produces),
            ("paris", "city", BondType::IsA),
            ("city", "capital", BondType::HasProperty),
        ];
        for (a, c, t) in bonds {
            let sa = Seed::from_label(a);
            let sc = Seed::from_label(c);
            b.network.add_bond(TemporalBond::new(sa, sc, t, 0.9, 0)).unwrap();
        }
        // Port T.5.1 — fixture wires bonds at timestamp=0; production
        // decay (90d half-life) would crush them at any wall-clock NOW.
        // Disable decay so this fixture stays regression-equivalent to
        // pre-T.5.1 (the spec’s decay_enabled=false guarantee).
        b.decay_config.enabled = false;
        b
    }

    fn ask(b: &mut Brain, query: &str, sid: Option<u64>) -> serde_json::Value {
        let mut args = serde_json::json!({"query": query, "depth": 2, "top_k": 5});
        if let Some(s) = sid {
            args["session_id"] = serde_json::json!(s);
        }
        tool_ask(b, args).expect("tool_ask")
    }

    // Phase-4b narrate helper (composed path + session pre-pass).
    fn narrate(b: &mut Brain, query: &str, sid: Option<u64>) -> serde_json::Value {
        let mut args = serde_json::json!({"query": query, "depth": 2, "top_k": 5});
        if let Some(s) = sid {
            args["session_id"] = serde_json::json!(s);
        }
        tool_narrate(b, args).expect("tool_narrate")
    }

    // Phase-4b -- bare "why?" re-routes to a causal query on the prior
    // subject anchor (honest, self-gating). We assert the rewrite, not
    // the answer body (the fixture holds no Causes edges, so causal
    // synthesis refuses cleanly -- which is exactly the honest outcome).
    #[test]
    fn phase4b_why_followup_routes_to_causal() {
        let mut b = fixture_brain();
        let sid = b.session_start();
        let _ = ask(&mut b, "photosynthesis", Some(sid));
        let resp = narrate(&mut b, "why?", Some(sid));
        let s = structured(&resp);
        let note = s.get("rewrite_note").and_then(|v| v.as_str()).unwrap_or("");
        assert!(note.contains("why -> causal"), "expected why->causal route, got {:?}", note);
        let eff = s.get("effective_query").and_then(|v| v.as_str()).unwrap_or("");
        assert!(eff.to_lowercase().starts_with("what causes"),
            "why should become a causal query, got {:?}", eff);
        assert!(eff.to_lowercase().contains("photosynthesis"),
            "causal query should target the prior subject, got {:?}", eff);
    }

    // Phase-4b -- "what about it?" continuation resolves the bare pronoun
    // to the prior subject (previously left dangling by is_continuation).
    #[test]
    fn phase4b_what_about_it_resolves_to_prior_subject() {
        let mut b = fixture_brain();
        let sid = b.session_start();
        let _ = ask(&mut b, "photosynthesis", Some(sid));
        let resp = ask(&mut b, "what about it?", Some(sid));
        let s = structured(&resp);
        let note = s.get("rewrite_note").and_then(|v| v.as_str()).unwrap_or("");
        assert!(note.contains("continuation pronoun"),
            "expected continuation-pronoun rewrite, got {:?}", note);
        let eff = s.get("effective_query").and_then(|v| v.as_str()).unwrap_or("");
        assert!(eff.to_lowercase().contains("photosynthesis"),
            "it should resolve to the prior subject, got {:?}", eff);
    }

    // Phase-4b -- a fresh noun phrase after "what about" is NOT rewritten
    // (no bare pronoun => the user supplied a new topic).
    #[test]
    fn phase4b_what_about_fresh_np_is_untouched() {
        let mut b = fixture_brain();
        let sid = b.session_start();
        let _ = ask(&mut b, "photosynthesis", Some(sid));
        let resp = ask(&mut b, "what about paris?", Some(sid));
        let s = structured(&resp);
        let note = s.get("rewrite_note").and_then(|v| v.as_str()).unwrap_or("");
        assert!(!note.contains("continuation pronoun"),
            "fresh NP must not be pronoun-rewritten, got {:?}", note);
        let eff = s.get("effective_query").and_then(|v| v.as_str()).unwrap_or("");
        assert!(eff.to_lowercase().contains("paris"), "fresh topic kept, got {:?}", eff);
        assert!(!eff.to_lowercase().contains("photosynthesis"),
            "fresh topic must not be overwritten by prior subject, got {:?}", eff);
    }

    // Phase-4b -- human-readable multi-turn transcript (printed under
    // --nocapture for the before/after record).
    #[test]
    fn phase4b_multiturn_transcript() {
        let mut b = fixture_brain();
        let sid = b.session_start();
        for q in ["photosynthesis", "what about it?", "why?", "tell me more"] {
            let resp = narrate(&mut b, q, Some(sid));
            let s = structured(&resp);
            let eff = s.get("effective_query").and_then(|v| v.as_str()).unwrap_or("<none>");
            let note = s.get("rewrite_note").and_then(|v| v.as_str()).unwrap_or("<none>");
            println!("TURN q={:?} effective={:?} rewrite={:?}", q, eff, note);
        }
    }

    fn structured<'a>(v: &'a serde_json::Value) -> &'a serde_json::Value {
        v.get("structuredContent").expect("structuredContent")
    }

    fn first_text(v: &serde_json::Value) -> String {
        v.get("content")
            .and_then(|c| c.get(0))
            .and_then(|t| t.get("text"))
            .and_then(|t| t.as_str())
            .expect("content[0].text")
            .to_string()
    }

    // 1 — Single-turn baseline: no session_id => no Q.3 fields, no rewrite.
    #[test]
    fn q3_int_single_turn_baseline_works_without_session_id() {
        let mut b = fixture_brain();
        let resp = ask(&mut b, "photosynthesis", None);
        let s = structured(&resp);
        assert!(s.get("session_id_q3").is_none(),
                "no session_id arg => no Q.3 echo, got {:?}", s);
        assert!(s.get("rewrite_note").is_none());
        let text = first_text(&resp);
        assert!(text.contains("photosynthesis"), "got {:?}", text);
    }

    // 2 — Pronoun resolution: "What does it produce?" after a primer.
    #[test]
    fn q3_int_two_turn_pronoun_resolves_it_to_prior_topic() {
        let mut b = fixture_brain();
        let sid = b.session_start();
        let _ = ask(&mut b, "photosynthesis", Some(sid));
        let resp2 = ask(&mut b, "What does it produce?", Some(sid));
        let s = structured(&resp2);
        let eff = s.get("effective_query").and_then(|v| v.as_str()).unwrap_or("");
        assert!(eff.to_lowercase().contains("photosynthesis"),
            "pronoun should be substituted with photosynthesis, got effective_query={:?}", eff);
        let note = s.get("rewrite_note").and_then(|v| v.as_str()).unwrap_or("");
        assert!(note.contains("pronoun"), "expected pronoun rewrite note, got {:?}", note);
    }

    // 3 — Elaboration: "tell me more" re-asks the prior effective query.
    #[test]
    fn q3_int_elaboration_reanchors_on_prior_turn() {
        let mut b = fixture_brain();
        let sid = b.session_start();
        let _ = ask(&mut b, "paris", Some(sid));
        let resp2 = ask(&mut b, "Tell me more.", Some(sid));
        let s = structured(&resp2);
        let eff = s.get("effective_query").and_then(|v| v.as_str()).unwrap_or("");
        assert!(eff.contains("paris"),
            "elaboration should re-ask 'paris', got effective_query={:?}", eff);
        let note = s.get("rewrite_note").and_then(|v| v.as_str()).unwrap_or("");
        assert!(note.contains("elaboration"), "got {:?}", note);
    }

    // 4 — Backreference: "What was my first question?" returns the literal
    // raw_query of turn 1, character-for-character.
    #[test]
    fn q3_int_backreference_first_question_returns_raw_verbatim() {
        let mut b = fixture_brain();
        let sid = b.session_start();
        let q1 = "What is photosynthesis?";
        let _ = ask(&mut b, q1, Some(sid));
        let _ = ask(&mut b, "what about glucose?", Some(sid));
        let back = ask(&mut b, "What was my first question?", Some(sid));
        let s = structured(&back);
        assert_eq!(
            s.get("backreference_kind").and_then(|v| v.as_str()),
            Some("FirstQuestion")
        );
        assert_eq!(
            s.get("answer").and_then(|v| v.as_str()),
            Some(q1),
            "backreference must return raw query verbatim"
        );
    }

    // 5 — "What did I just ask?" returns the prior raw query.
    #[test]
    fn q3_int_backreference_last_question_returns_prior_raw() {
        let mut b = fixture_brain();
        let sid = b.session_start();
        let _ = ask(&mut b, "photosynthesis", Some(sid));
        let _ = ask(&mut b, "What does it produce?", Some(sid));
        let back = ask(&mut b, "What did I just ask?", Some(sid));
        let s = structured(&back);
        assert_eq!(
            s.get("answer").and_then(|v| v.as_str()),
            Some("What does it produce?"),
            "expected the prior raw query verbatim"
        );
    }

    // 6 — Session isolation: two parallel sessions don't bleed anchors.
    #[test]
    fn q3_int_two_sessions_isolated() {
        let mut b = fixture_brain();
        let sid_a = b.session_start();
        let _ = ask(&mut b, "photosynthesis", Some(sid_a));
        let _ = b.session_end();
        let sid_b = b.session_start();
        let _ = ask(&mut b, "paris", Some(sid_b));
        // Session B's pronoun resolution should land on paris, NOT
        // photosynthesis from session A.
        let mem_b = b.session_memories.get(sid_b).expect("sid_b");
        assert_eq!(mem_b.current_topic.as_deref(), Some("paris"));
        assert!(mem_b.anchors.iter().all(|a| a.label != "photosynthesis"),
            "session B should not see session A's anchors, got {:?}", mem_b.anchors);
    }

    // 7 — session_end clears state.
    #[test]
    fn q3_int_session_end_evicts_memory() {
        let mut b = fixture_brain();
        let sid = b.session_start();
        let _ = ask(&mut b, "photosynthesis", Some(sid));
        assert!(b.session_memories.get(sid).is_some());
        let _ = b.session_end();
        assert!(b.session_memories.get(sid).is_none(),
            "session_end should evict the operational memory");
    }

    // 8 — Idle eviction with a short timeout.
    #[test]
    fn q3_int_idle_eviction_drops_stale_session() {
        let mut b = fixture_brain();
        let sid = b.session_start();
        let _ = ask(&mut b, "photosynthesis", Some(sid));
        // Force the registry's idle clock by tweaking last_active far
        // into the past, then trigger an eviction sweep.
        if let Some(mem) = b.session_memories.get_mut(sid) {
            mem.last_active_ms = 0;
        }
        // 30 minutes of idleness exceeds DEFAULT_IDLE_TIMEOUT_MS.
        let now = crate::brain::now_ms();
        let evicted = b.session_memories.evict_idle(now);
        assert!(evicted.contains(&sid),
            "expected session {} to be evicted, got {:?}", sid, evicted);
        assert!(b.session_memories.get(sid).is_none());
    }

    // 9 — Stale-anchor clarification: pronoun fires but no anchor is
    // strong enough, so we ask for clarification rather than guessing.
    #[test]
    fn q3_int_pronoun_with_no_strong_anchor_asks_for_clarification() {
        let mut b = fixture_brain();
        let sid = b.session_start();
        // Drive a turn so a session memory exists, then knock all
        // anchors below threshold.
        let _ = ask(&mut b, "photosynthesis", Some(sid));
        if let Some(mem) = b.session_memories.get_mut(sid) {
            for a in mem.anchors.iter_mut() { a.weight = 0.05; }
        }
        let resp = ask(&mut b, "What does it produce?", Some(sid));
        let s = structured(&resp);
        assert_eq!(
            s.get("clarification_requested").and_then(|v| v.as_bool()),
            Some(true),
            "expected clarification, got structured={:?}", s
        );
    }

    // 10 — When the same anchor is re-asked, it gets re-promoted to
    // weight 1.0 instead of being decayed away.
    #[test]
    fn q3_int_repeated_anchor_stays_strong() {
        let mut b = fixture_brain();
        let sid = b.session_start();
        let _ = ask(&mut b, "photosynthesis", Some(sid));
        let _ = ask(&mut b, "photosynthesis", Some(sid));
        let _ = ask(&mut b, "photosynthesis", Some(sid));
        let mem = b.session_memories.get(sid).unwrap();
        let a = mem.anchors.iter().find(|a| a.label == "photosynthesis").unwrap();
        assert!(a.weight >= 0.9,
            "repeated anchor should stay strong, got weight={}", a.weight);
    }
}
#[cfg(test)]
mod port_k_idk_gate_integration_tests {
    //! Port K (#581) — integration coverage on `tool_ask` routed
    //! through a Brain populated with a small synthetic graph. The
    //! full pipeline runs (Q.3 pre-pass, recall, K gate, post-pass)
    //! WITHOUT touching the canonical brain.
    //!
    //! Sub-modules:
    //!   - `nonsense_tests` — queries that should TRIGGER K
    //!   - `weak_real_tests` — queries that look weak but are real;
    //!      should NOT trigger K
    //!   - `regression_tests` — strong queries / arithmetic / Bug B
    //!      bare-words; should NOT trigger K
    //!   - `conversation_context_tests` — Q.3 + K interaction
    use super::*;
    use crate::bond::{BondType, TemporalBond};
    use crate::brain::Brain;
    use crate::node::{Confidence, KnowledgeContent, KnowledgeNode};
    use crate::seed::Seed;

    fn fixture_brain() -> Brain {
        // Real concepts the gate should handle confidently. K.1
        // Signal 4 measures anchor strength as the count of a
        // token's top keyword-hit seed's 1-hop neighbors that
        // appear in the activated set, requiring >=
        // per_token_anchor_count_floor (default 3, derived from
        // #609 canonical-brain probes). The fixture wires each
        // rich concept to >=3 neighbors so it actually anchors
        // under that floor; sparser wiring (the pre-K.1 fixture
        // had photosynthesis -> glucose, oxygen only, count=2)
        // caused dominance sub_mode misclassification and
        // fully-connected real queries to false-positive.
        //   photosynthesis -[Produces]-> glucose -[Produces]-> energy
        //   photosynthesis -[Produces]-> oxygen
        //   photosynthesis -[HasProperty]-> chlorophyll
        //   photosynthesis -[RelatedTo]-> water
        //   glucose -[IsA]-> sugar
        //   oxygen -[IsA]-> gas
        //   oxygen -[RelatedTo]-> respiration
        //   paris -[IsA]-> city -[HasProperty]-> capital
        //   mitochondria -[Produces]-> energy
        //   mitochondria -[PartOf]-> cell
        //   dna -[StoresInfo]-> gene  (uses HasProperty as proxy)
        //   function -[RelatedTo]-> role
        //   relationship -[RelatedTo]-> connection
        let mut b = Brain::new();
        let labels = [
            "photosynthesis", "glucose", "energy", "oxygen",
            "chlorophyll", "water", "sugar", "gas", "respiration",
            "paris", "city", "capital",
            "mitochondria", "cell",
            "dna", "gene",
            "function", "role",
            "relationship", "connection",
        ];
        for label in labels {
            let s = Seed::from_label(label);
            b.network.add_node(KnowledgeNode::with_seed(
                s, label.into(),
                KnowledgeContent::Concept(label.into()),
                Confidence::Verified, 0,
            ));
        }
        let bonds = [
            ("photosynthesis", "glucose", BondType::Produces),
            ("glucose", "energy", BondType::Produces),
            ("photosynthesis", "oxygen", BondType::Produces),
            ("photosynthesis", "chlorophyll", BondType::HasProperty),
            ("photosynthesis", "water", BondType::RelatedTo),
            ("glucose", "sugar", BondType::IsA),
            ("oxygen", "gas", BondType::IsA),
            ("oxygen", "respiration", BondType::RelatedTo),
            ("paris", "city", BondType::IsA),
            ("city", "capital", BondType::HasProperty),
            ("mitochondria", "energy", BondType::Produces),
            ("mitochondria", "cell", BondType::PartOf),
            ("dna", "gene", BondType::HasProperty),
            ("function", "role", BondType::RelatedTo),
            ("relationship", "connection", BondType::RelatedTo),
        ];
        for (a, c, t) in bonds {
            let sa = Seed::from_label(a);
            let sc = Seed::from_label(c);
            b.network.add_bond(TemporalBond::new(sa, sc, t, 0.9, 0)).unwrap();
        }
        // Port T.5.1 — fixture wires bonds at timestamp=0; production
        // decay (90d half-life) would crush them at any wall-clock NOW.
        // Disable decay so this fixture stays regression-equivalent to
        // pre-T.5.1 (the spec’s decay_enabled=false guarantee).
        b.decay_config.enabled = false;
        b
    }

    fn ask(b: &mut Brain, query: &str, sid: Option<u64>) -> serde_json::Value {
        // top_k=50 (vs production default 5) gives the synthetic
        // fixture’s tiny activated map enough headroom that K.1’s
        // per-token neighbor coverage check sees real-concept
        // neighborhoods at full breadth instead of being truncated
        // to the top few seeds. The canonical brain has enough
        // mass that 5 is fine; the fixture does not.
        let mut args = serde_json::json!({"query": query, "depth": 2, "top_k": 50});
        if let Some(s) = sid {
            args["session_id"] = serde_json::json!(s);
        }
        tool_ask(b, args).expect("tool_ask")
    }

    fn structured<'a>(v: &'a serde_json::Value) -> &'a serde_json::Value {
        v.get("structuredContent").expect("structuredContent")
    }

    fn first_text(v: &serde_json::Value) -> String {
        v.get("content")
            .and_then(|c| c.get(0))
            .and_then(|t| t.get("text"))
            .and_then(|t| t.as_str())
            .expect("content[0].text")
            .to_string()
    }

    fn fired_idk(v: &serde_json::Value) -> bool {
        structured(v).get("source").and_then(|s| s.as_str()) == Some("idk_gate")
    }

    fn idk_mode(v: &serde_json::Value) -> Option<String> {
        structured(v).get("mode").and_then(|s| s.as_str()).map(String::from)
    }

    // ================================================================
    // nonsense_tests — K SHOULD fire
    // ================================================================
    mod nonsense_tests {
        use super::*;

        #[test]
        fn k_fires_on_pure_nonsense_flibbergast() {
            let mut b = fixture_brain();
            let resp = ask(&mut b, "flibbergast", None);
            assert!(fired_idk(&resp), "expected K to fire, got {:?}", first_text(&resp));
            assert_eq!(idk_mode(&resp).as_deref(), Some("total_idk"));
        }

        #[test]
        fn k_fires_on_alphanumeric_garbage_xqzblr12() {
            let mut b = fixture_brain();
            let resp = ask(&mut b, "xqzblr12", None);
            assert!(fired_idk(&resp), "expected K to fire, got {:?}", first_text(&resp));
            assert_eq!(idk_mode(&resp).as_deref(), Some("total_idk"));
        }

        #[test]
        fn k_fires_partial_leak_on_zorblax_photosynthesis() {
            // photosynthesis is real; zorblax is junk. Mode B fires.
            let mut b = fixture_brain();
            let resp = ask(&mut b, "zorblax photosynthesis", None);
            assert!(fired_idk(&resp), "expected K to fire, got {:?}", first_text(&resp));
            assert_eq!(idk_mode(&resp).as_deref(), Some("partial_match_leak"));
            let text = first_text(&resp);
            assert!(text.contains("photosynthesis"));
            assert!(text.contains("zorblax"));
            assert!(text.contains("won't fabricate"));
        }

        #[test]
        fn k_fires_partial_leak_on_quantum_jellyfish_dynamics() {
            // None of "quantum", "jellyfish", "dynamics" exist in the
            // fixture, so this is total_idk on the fixture (not
            // partial). Either mode is acceptable; assert refusal.
            let mut b = fixture_brain();
            let resp = ask(&mut b, "quantum jellyfish dynamics", None);
            assert!(fired_idk(&resp), "expected K to fire, got {:?}", first_text(&resp));
        }

        #[test]
        fn k_fires_on_flarbnozzle_widget() {
            let mut b = fixture_brain();
            let resp = ask(&mut b, "flarbnozzle widget", None);
            assert!(fired_idk(&resp), "expected K to fire, got {:?}", first_text(&resp));
        }

        #[test]
        fn k_total_idk_prose_explains_why() {
            let mut b = fixture_brain();
            let resp = ask(&mut b, "flibbergast", None);
            let text = first_text(&resp);
            // OP-QA-4: humanized decline -- plain language, no machinery/number.
            assert!(text.contains("anything solid"), "got {:?}", text);
            assert!(!text.contains("below my confidence threshold"), "machinery leaked: {:?}", text);
            assert!(!text.chars().any(|c| c.is_ascii_digit()), "activation number leaked: {:?}", text);
        }
    }

    // ================================================================
    // weak_real_tests — K SHOULD NOT fire
    // ================================================================
    mod weak_real_tests {
        use super::*;

        #[test]
        fn k_does_not_fire_on_obscure_real_concept() {
            // "dna gene" — both real, both in the fixture. Even with
            // a weak chain, K must not refuse.
            let mut b = fixture_brain();
            let resp = ask(&mut b, "dna", None);
            assert!(!fired_idk(&resp), "K should not refuse real query, got {:?}",
                first_text(&resp));
        }

        #[test]
        fn k_does_not_fire_on_mitochondria_cell() {
            let mut b = fixture_brain();
            let resp = ask(&mut b, "mitochondria cell", None);
            assert!(!fired_idk(&resp), "got {:?}", first_text(&resp));
        }

        #[test]
        fn k_does_not_fire_on_glucose_alone() {
            let mut b = fixture_brain();
            let resp = ask(&mut b, "glucose", None);
            assert!(!fired_idk(&resp), "got {:?}", first_text(&resp));
        }

        #[test]
        fn k_does_not_fire_on_oxygen_alone() {
            let mut b = fixture_brain();
            let resp = ask(&mut b, "oxygen", None);
            assert!(!fired_idk(&resp), "got {:?}", first_text(&resp));
        }

        #[test]
        fn k_does_not_fire_on_capital() {
            let mut b = fixture_brain();
            let resp = ask(&mut b, "capital", None);
            assert!(!fired_idk(&resp), "got {:?}", first_text(&resp));
        }
    }

    // ================================================================
    // regression_tests — standard happy paths must keep working
    // ================================================================
    mod regression_tests {
        use super::*;

        #[test]
        fn paris_returns_normal_recall() {
            let mut b = fixture_brain();
            let resp = ask(&mut b, "paris", None);
            assert!(!fired_idk(&resp), "got {:?}", first_text(&resp));
            let text = first_text(&resp);
            assert!(text.contains("paris"));
        }

        #[test]
        fn mitochondria_returns_normal_recall() {
            let mut b = fixture_brain();
            let resp = ask(&mut b, "mitochondria", None);
            assert!(!fired_idk(&resp), "got {:?}", first_text(&resp));
        }

        #[test]
        fn what_is_dna_returns_normal_recall() {
            let mut b = fixture_brain();
            let resp = ask(&mut b, "what is dna", None);
            assert!(!fired_idk(&resp), "got {:?}", first_text(&resp));
        }

        #[test]
        fn function_bare_word_still_works_via_bug_b_fallback() {
            // "function" is on the query-noise list. Bug B's bare-word
            // fallback rescues it. K must NOT refuse.
            let mut b = fixture_brain();
            let resp = ask(&mut b, "function", None);
            assert!(!fired_idk(&resp),
                "Bug B fallback path must not be refused by K, got {:?}",
                first_text(&resp));
        }

        #[test]
        fn relationship_bare_word_still_works() {
            let mut b = fixture_brain();
            let resp = ask(&mut b, "relationship", None);
            assert!(!fired_idk(&resp), "got {:?}", first_text(&resp));
        }

        #[test]
        fn arithmetic_routes_through_math_primitive_before_k() {
            // F.1 must short-circuit BEFORE K. The arithmetic primitive
            // owns the response and source flag is `arithmetic_primitive`.
            let mut b = fixture_brain();
            let resp = ask(&mut b, "what is 7 plus 5", None);
            assert!(!fired_idk(&resp), "K must not steal arithmetic");
            let s = structured(&resp);
            let source = s.get("source").and_then(|v| v.as_str()).unwrap_or("");
            assert!(
                source == "arithmetic_primitive" || source == "expression_primitive",
                "expected math primitive, got source={:?}", source,
            );
        }

        #[test]
        fn word_problem_routes_through_f3_before_k() {
            // F.3 word-problem extraction must fire before K. The
            // problem itself uses real-but-weak vocabulary.
            let mut b = fixture_brain();
            let resp = ask(
                &mut b,
                "Sarah had 5 apples. She gave 2 to Tom. How many does she have left?",
                None,
            );
            assert!(!fired_idk(&resp), "K must not steal F.3 word problems");
        }
    }

    // ================================================================
    // conversation_context_tests — Q.3 + K interaction
    // ================================================================
    mod conversation_context_tests {
        use super::*;

        #[test]
        fn k_does_not_fire_when_q3_resolves_pronoun_to_strong_anchor() {
            // Turn 1 establishes "photosynthesis" anchor.
            // Turn 2 "What does it produce?" — Q.3 substitutes "it" with
            // "photosynthesis". The rewritten query has strong signal —
            // K must not refuse.
            //
            // ("What about it?" trips `is_continuation` first per the
            // Q.3 ordering and is left alone, so it doesn't exercise
            // the pronoun-substitution path.)
            let mut b = fixture_brain();
            let sid = b.session_start();
            let _ = ask(&mut b, "photosynthesis", Some(sid));
            let resp = ask(&mut b, "What does it produce?", Some(sid));
            assert!(!fired_idk(&resp),
                "Q.3 rewrite should resolve via session, not refuse via K. Got {:?}",
                first_text(&resp));
            let s = structured(&resp);
            let note = s.get("rewrite_note").and_then(|v| v.as_str()).unwrap_or("");
            assert!(note.contains("pronoun"), "expected Q.3 pronoun rewrite, got {:?}", note);
            let eff = s.get("effective_query").and_then(|v| v.as_str()).unwrap_or("");
            assert!(eff.to_lowercase().contains("photosynthesis"),
                "effective_query should contain photosynthesis, got {:?}", eff);
        }

        #[test]
        fn k_does_not_fire_on_elaboration_after_real_query() {
            // Turn 1: "paris". Turn 2: "Tell me more." Q.3 re-anchors on paris.
            let mut b = fixture_brain();
            let sid = b.session_start();
            let _ = ask(&mut b, "paris", Some(sid));
            let resp = ask(&mut b, "Tell me more.", Some(sid));
            assert!(!fired_idk(&resp), "got {:?}", first_text(&resp));
        }

        #[test]
        fn k_fires_when_session_has_no_anchor_and_query_is_nonsense() {
            // Q.3 has no prior anchor; query is junk. K refuses cleanly
            // and the refusal turn IS recorded so backreferences work.
            let mut b = fixture_brain();
            let sid = b.session_start();
            let resp = ask(&mut b, "flibbergast", Some(sid));
            assert!(fired_idk(&resp), "got {:?}", first_text(&resp));
            // The refusal must be recorded as a turn — backreference
            // queries on this session should still see it.
            let mem = b.session_memories.get(sid).expect("session memory");
            assert!(mem.last_turn().is_some(),
                "K refusal must be recorded as a session turn");
        }
    }

    // ================================================================
    // k1_dominance_tests — K.1 single-anchor-dominance coverage
    //
    // The synthetic fixture ships with one rich neighborhood
    // (photosynthesis → glucose → energy / oxygen) plus a few
    // isolated decoy nodes whose only role is to have keyword-index
    // hits with zero connectivity. Queries that combine the rich
    // anchor with multiple decoys must fire Mode D; queries that
    // span multiple connected nodes must pass.
    // ================================================================
    mod k1_dominance_tests {
        use super::*;

        fn fixture_brain_with_decoys() -> Brain {
            let mut b = fixture_brain();
            // Add isolated decoy nodes — keyword hits but no bonds.
            // Their seeds get selected if a query mentions them, but
            // their 1-hop neighborhoods are empty, so K.1 marks them
            // as un-anchored.
            let decoys = ["decoyalpha", "decoybeta", "decoygamma"];
            for label in decoys {
                let s = Seed::from_label(label);
                b.network.add_node(KnowledgeNode::with_seed(
                    s, label.into(),
                    KnowledgeContent::Concept(label.into()),
                    Confidence::Verified, 0,
                ));
            }
            b
        }

        fn idk_sub_mode(v: &serde_json::Value) -> Option<String> {
            structured(v).get("sub_mode").and_then(|s| s.as_str()).map(String::from)
        }

        #[test]
        fn k1_fires_dominance_on_rich_anchor_plus_two_decoys() {
            // photosynthesis is connected to glucose / oxygen / energy
            // (rich); decoyalpha and decoybeta are isolated. Mode D
            // must fire and tag sub_mode = single_anchor_dominance.
            let mut b = fixture_brain_with_decoys();
            let resp = ask(&mut b, "photosynthesis decoyalpha decoybeta", None);
            assert!(fired_idk(&resp),
                "K.1 must refuse rich-anchor + decoys, got {:?}",
                first_text(&resp));
            assert_eq!(idk_mode(&resp).as_deref(), Some("partial_match_leak"));
            assert_eq!(idk_sub_mode(&resp).as_deref(),
                Some("single_anchor_dominance"),
                "expected sub_mode=single_anchor_dominance; got {:?}",
                idk_sub_mode(&resp));
            let text = first_text(&resp);
            assert!(text.contains("matched strongly on"),
                "expected dominance prose, got {:?}", text);
            assert!(text.contains("photosynthesis"));
            assert!(text.contains("won't fabricate"));
        }

        #[test]
        fn k1_does_not_fire_when_all_three_tokens_connected() {
            // photosynthesis glucose oxygen — all three are real,
            // connected concepts. Coverage 3/3 — K.1 must pass.
            let mut b = fixture_brain_with_decoys();
            let resp = ask(&mut b, "photosynthesis glucose oxygen", None);
            assert!(!fired_idk(&resp),
                "K.1 must not refuse fully-connected real query, got {:?}",
                first_text(&resp));
        }

        #[test]
        fn k1_does_not_fire_on_two_token_query_with_decoy() {
            // N = 2 falls below coverage_min_tokens. Even though only
            // one anchors, K.1 stays out and lets Modes B/C handle it.
            let mut b = fixture_brain_with_decoys();
            let resp = ask(&mut b, "photosynthesis decoyalpha", None);
            // Mode D must NOT fire. (Whether Mode B/C fires is a
            // separate concern — we check sub_mode is not
            // single_anchor_dominance regardless.)
            if fired_idk(&resp) {
                assert_ne!(idk_sub_mode(&resp).as_deref(),
                    Some("single_anchor_dominance"),
                    "K.1 must not fire for N=2; got dominance");
            }
        }

        #[test]
        fn k1_does_not_fire_on_single_token_real_query() {
            // N = 1 — trivially anchored, no coverage signal.
            let mut b = fixture_brain_with_decoys();
            let resp = ask(&mut b, "photosynthesis", None);
            assert!(!fired_idk(&resp), "got {:?}", first_text(&resp));
        }

        #[test]
        fn k1_structured_payload_surfaces_coverage_ratio() {
            // Mode D must populate sub_mode, coverage_ratio,
            // dominant_token, and coverage_missing_tokens.
            let mut b = fixture_brain_with_decoys();
            let resp = ask(&mut b, "photosynthesis decoyalpha decoybeta", None);
            assert!(fired_idk(&resp));
            let s = structured(&resp);
            assert_eq!(s["sub_mode"], "single_anchor_dominance");
            let signals = &s["signals"];
            assert_eq!(signals["content_token_count"], 3);
            assert_eq!(signals["anchored_token_count"], 1);
            let ratio = signals["coverage_ratio"].as_f64().unwrap();
            assert!(ratio < 0.5, "coverage_ratio should be < 0.5, got {}", ratio);
            assert_eq!(signals["dominant_token"], "photosynthesis");
            let missing = signals["coverage_missing_tokens"].as_array().unwrap();
            let missing_strs: Vec<String> = missing.iter()
                .map(|v| v.as_str().unwrap().to_string())
                .collect();
            assert!(missing_strs.contains(&"decoyalpha".to_string()));
            assert!(missing_strs.contains(&"decoybeta".to_string()));
        }

        #[test]
        fn k1_fires_dominance_on_three_decoys_against_rich_anchor() {
            // Mix the dominant token among decoys to verify
            // coverage works regardless of token order.
            let mut b = fixture_brain_with_decoys();
            let resp = ask(&mut b, "decoyalpha photosynthesis decoygamma", None);
            assert!(fired_idk(&resp), "got {:?}", first_text(&resp));
            assert_eq!(idk_sub_mode(&resp).as_deref(),
                Some("single_anchor_dominance"));
            let text = first_text(&resp);
            assert!(text.contains("\"photosynthesis\""),
                "dominant token must appear in prose, got {:?}", text);
        }
    }
}
// =====================================================================
// Port T.2 — lu4d_idle_tick MCP tool wrapper.
//
// Drives one iteration of the idle work loop. The MCP dispatch
// (`handle_tools_call`) special-cases this tool to skip the standard
// QueryGuard so the gate's idleness check sees the brain's genuine
// pre-tick state.
//
// Args (all optional):
//   - min_idle_ms: u64 — minimum idle duration required before this
//     tick will do work (default 0).
//
// Returns (structured):
//   - did_work: bool
//   - wonder_seed_processed: Option<String> (8-byte hex prefix of the
//     popped seed, or null when no signal was popped)
//   - outcome: "no_work_no_signals" | "no_work_not_idle" |
//              "research_completed" | "research_failed"
//   - idle_duration_ms: u64
//   - total_idle_ticks: u64
//
// KNOWN GAPS (closed by future ports):
//   - T.3: pending-changes-with-approval gate (today the proposal
//     auto-commits to the Pending queue using the manual-research
//     code path).
//   - T.6: daemon-driven auto-tick (today this is caller-driven).
// =====================================================================
fn tool_idle_tick(brain: &mut Brain, args: Value) -> Result<Value, (i64, String)> {
    let min_idle_ms = args
        .get("min_idle_ms")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let fetcher = brain.default_research_fetcher();
    let res = brain.idle_tick(&fetcher, min_idle_ms);
    let outcome_str = res.outcome.as_str();
    let text = format!(
        "idle_tick: outcome={} did_work={} idle_duration_ms={} total_idle_ticks={}",
        outcome_str,
        res.did_work,
        res.idle_duration_ms,
        res.total_idle_ticks,
    );
    let structured = serde_json::json!({
        "did_work": res.did_work,
        "wonder_seed_processed": res.wonder_seed_processed,
        "outcome": outcome_str,
        "idle_duration_ms": res.idle_duration_ms,
        "total_idle_ticks": res.total_idle_ticks,
        "pending_added_count": res.pending_added_count,
    });
    Ok(wrap_content(text, structured))
}

// =====================================================================
// Port S.1 (#637) — lu4d_revoke_by_source MCP tool wrapper.
//
// Exposes Brain::revoke_bonds_by_source (Port S, internal notes as
// a callable JSON-RPC tool — closes the GDPR / research-rollback
// loop. Builds a typed predicate from JSON args, then either counts
// matches (dry_run=true, default) or invokes the underlying revoke
// primitive (dry_run=false, destructive).
//
// ADMIN GATE (BR.5 audit P1 #3, internal notes: dry_run=false is
// gated behind brain.admin_revoke_enabled (persisted via section #15
// runtime_config). Default `false` (closed). Operators flip it on
// per-process via lu4d_admin_set. Dry-run preview stays open so
// "what would this revoke?" remains a cheap query for ops/audit.
// The structured payload echoes `admin_gated: true` once a brain
// has flipped the flag, and the gate-blocked path returns -32603
// with a directive to call lu4d_admin_set first. This closes the
// internal notes hole that would have shipped open with the
// lu.dialectforge.com PWA.
// =====================================================================
fn tool_revoke_by_source(brain: &mut Brain, args: Value) -> Result<Value, (i64, String)> {
    let kind_str = args
        .get("kind")
        .and_then(|v| v.as_str())
        .ok_or((-32602, "missing required arg: kind".to_string()))?
        .to_string();
    let dry_run = args.get("dry_run").and_then(|v| v.as_bool()).unwrap_or(true);

    // BR.5 admin gate: dry_run=false is destructive and gated. Dry-run
    // (the default) bypasses the gate so previews stay cheap.
    if !dry_run && !brain.admin_revoke_enabled {
        return Err((
            -32603,
            "lu4d_revoke_by_source: dry_run=false is gated by admin_revoke_enabled.              Call lu4d_admin_set with {\"admin_revoke_enabled\": true} before              invoking the destructive path. Default is closed (BR.5 default-deny)."
                .to_string(),
        ));
    }

    type Pred = Box<dyn Fn(&crate::source::SourceAttribution) -> bool>;
    let predicate: Pred = match kind_str.to_lowercase().as_str() {
        "trainingcorpus" | "training_corpus" | "training" => {
            let corpus_filter = args.get("corpus").and_then(|v| v.as_str()).map(String::from);
            let run_id_filter = args.get("run_id").and_then(|v| v.as_str()).map(String::from);
            Box::new(move |attr| {
                if let crate::source::SourceKind::TrainingCorpus { corpus, run_id, .. } = &attr.kind {
                    if let Some(cf) = &corpus_filter {
                        if corpus != cf { return false; }
                    }
                    if let Some(rf) = &run_id_filter {
                        match run_id {
                            Some(r) if r == rf => {}
                            _ => return false,
                        }
                    }
                    true
                } else { false }
            })
        }
        "research" => {
            let wonder_seed_filter = args.get("wonder_seed").and_then(|v| v.as_str()).map(String::from);
            let url_contains_filter = args.get("url_contains").and_then(|v| v.as_str()).map(String::from);
            Box::new(move |attr| {
                if let crate::source::SourceKind::Research { wonder_seed, url, .. } = &attr.kind {
                    if let Some(ws) = &wonder_seed_filter {
                        if wonder_seed != ws { return false; }
                    }
                    if let Some(uc) = &url_contains_filter {
                        match url {
                            Some(u) if u.contains(uc.as_str()) => {}
                            _ => return false,
                        }
                    }
                    true
                } else { false }
            })
        }
        "teach" => {
            let actor_filter = args.get("actor").and_then(|v| v.as_str()).map(String::from);
            let before_ms = args.get("before_ms").and_then(|v| v.as_u64());
            let after_ms = args.get("after_ms").and_then(|v| v.as_u64());
            Box::new(move |attr| {
                if let crate::source::SourceKind::Teach { actor, ts_ms } = &attr.kind {
                    if let Some(af) = &actor_filter {
                        if actor != af { return false; }
                    }
                    if let Some(b) = before_ms {
                        if *ts_ms >= b { return false; }
                    }
                    if let Some(a) = after_ms {
                        if *ts_ms <= a { return false; }
                    }
                    true
                } else { false }
            })
        }
        "document" => {
            let doc_hash_filter = args.get("doc_hash").and_then(|v| v.as_str()).map(String::from);
            let corpus_filter = args.get("corpus").and_then(|v| v.as_str()).map(String::from);
            Box::new(move |attr| {
                if let crate::source::SourceKind::Document { doc_hash, corpus } = &attr.kind {
                    if let Some(dh) = &doc_hash_filter {
                        if doc_hash != dh { return false; }
                    }
                    if let Some(cf) = &corpus_filter {
                        match corpus {
                            Some(c) if c == cf => {}
                            _ => return false,
                        }
                    }
                    true
                } else { false }
            })
        }
        // Port SourceKind-Variants — first-class selector kinds.
        "chained_research" | "chainedresearch" => {
            let chain_id_filter = args.get("chain_id").and_then(|v| v.as_str()).map(String::from);
            let hop_filter = args.get("hop").and_then(|v| v.as_u64()).map(|n| n as u32);
            let parent_query_contains = args.get("parent_query_contains").and_then(|v| v.as_str()).map(String::from);
            Box::new(move |attr| {
                if let crate::source::SourceKind::ChainedResearch { chain_id, hop, parent_query } = &attr.kind {
                    if let Some(cf) = &chain_id_filter {
                        if chain_id != cf { return false; }
                    }
                    if let Some(hf) = hop_filter {
                        if *hop != hf { return false; }
                    }
                    if let Some(pc) = &parent_query_contains {
                        if !parent_query.contains(pc.as_str()) { return false; }
                    }
                    true
                } else { false }
            })
        }
        "session_promote" | "sessionpromote" => {
            let session_id_filter = args.get("session_id").and_then(|v| v.as_str()).map(String::from);
            let entry_id_filter = args.get("entry_id").and_then(|v| v.as_u64());
            Box::new(move |attr| {
                if let crate::source::SourceKind::SessionPromote { session_id, entry_id } = &attr.kind {
                    if let Some(sf) = &session_id_filter {
                        if session_id != sf { return false; }
                    }
                    if let Some(ef) = entry_id_filter {
                        if *entry_id != ef { return false; }
                    }
                    true
                } else { false }
            })
        }
        "pack_install" | "packinstall" => {
            let pack_id_filter = args.get("pack_id").and_then(|v| v.as_str()).map(String::from);
            let pubkey_filter = args.get("publisher_pubkey_hex").and_then(|v| v.as_str()).map(String::from);
            let signed_before_ms = args.get("signed_before_ms").and_then(|v| v.as_u64());
            Box::new(move |attr| {
                if let crate::source::SourceKind::PackInstall { pack_id, publisher_pubkey_hex, signed_at_ms } = &attr.kind {
                    if let Some(pf) = &pack_id_filter {
                        if pack_id != pf { return false; }
                    }
                    if let Some(kf) = &pubkey_filter {
                        if publisher_pubkey_hex != kf { return false; }
                    }
                    if let Some(b) = signed_before_ms {
                        if *signed_at_ms >= b { return false; }
                    }
                    true
                } else { false }
            })
        }
        "unknown" => Box::new(|attr| matches!(attr.kind, crate::source::SourceKind::Unknown)),
        other => {
            return Err((
                -32602,
                format!(
                    "unknown kind '{}' (expected one of training_corpus|research|teach|document|unknown|chained_research|session_promote|pack_install; case-insensitive)",
                    other
                ),
            ));
        }
    };

    // Walk overlay entries to compute matched_count + kinds_summary
    // BEFORE any mutation. The borrow ends at the loop boundary so
    // the &mut brain handoff to revoke_bonds_by_source is clean.
    let mut kind_tags: std::collections::HashMap<&'static str, usize> = std::collections::HashMap::new();
    let mut matched_count: usize = 0;
    {
        let source_overlay = brain.source_overlay.read().unwrap();
        for (_key, attr) in &source_overlay.entries {
            if predicate(attr) {
                matched_count += 1;
                *kind_tags.entry(attr.kind.tag()).or_insert(0) += 1;
            }
        }
    }

    let revoked_count = if dry_run {
        0usize
    } else {
        // Box<dyn Fn> implements Fn by deref, so it satisfies the
        // F: Fn bound on revoke_bonds_by_source.
        let (_overlay_removed, bonds_removed) = brain.revoke_bonds_by_source(predicate);
        bonds_removed
    };

    let mut kinds_summary: Vec<Value> = kind_tags
        .into_iter()
        .map(|(tag, count)| json!({"kind": tag, "count": count}))
        .collect();
    kinds_summary.sort_by(|a, b| a["kind"].as_str().unwrap_or("").cmp(b["kind"].as_str().unwrap_or("")));

    let text = format!(
        "{} {} bond(s) matching kind={} (dry_run={}); revoked={}",
        if dry_run { "Would revoke" } else { "Revoked" },
        matched_count,
        kind_str,
        dry_run,
        revoked_count,
    );
    let structured = json!({
        "matched_count": matched_count,
        "revoked_count": revoked_count,
        "dry_run": dry_run,
        "kinds_summary": kinds_summary,
        "kind_filter": kind_str,
        // BR.5: admin_gated reflects whether this brain's destructive
        // revoke path is currently open. False = closed (default) =
        // dry_run=false would have been refused by the gate above.
        "admin_gated": brain.admin_revoke_enabled,
    });
    Ok(wrap_content(text, structured))
}

// =====================================================================
// Port SourceKind-Variants — lu4d_source_attribution_migrate
//
// Operator-driven, OFF-BY-DEFAULT migration that upgrades legacy
// `SourceKind::Teach { actor: "<prefix>:<id>" }` overlay entries to the
// first-class ChainedResearch / SessionPromote / PackInstall variants.
// The canonical brain (10.8M bonds) intentionally retains its legacy
// attribution as a forensic audit trail until the next retrain — this
// tool is documented and registered, but mutating canonical requires
// dry_run=false AND confirm_canonical_mutation=true. Stored bond
// weights are never touched (the anti-hallucination invariant); only
// the side-table SourceAttribution.kind is upgraded.
//
// Caveat: legacy `research_chain:` actor entries are migrated to a
// best-effort ChainedResearch — the original chain context (chain_id /
// hop / parent_query) was never stored on the legacy entry, so the
// migrated entry uses (chain_id = stripped suffix, hop = 0, parent_query
// = "<unknown>"). This is a one-way upgrade; the legacy data is lost
// in the rewrite.
// =====================================================================
fn tool_source_attribution_migrate(
    brain: &mut Brain,
    args: Value,
) -> Result<Value, (i64, String)> {
    let dry_run = args.get("dry_run").and_then(|v| v.as_bool()).unwrap_or(true);
    let confirm = args
        .get("confirm_canonical_mutation")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let limit_per_prefix = args
        .get("limit_per_prefix")
        .and_then(|v| v.as_u64())
        .map(|n| n as usize);

    if !dry_run && !confirm {
        return Err((
            -32602,
            "refusing to mutate without confirm_canonical_mutation=true (anti-mutation guard)".into(),
        ));
    }

    // Walk the overlay, collect per-prefix work, OPTIONALLY rewrite.
    let mut chain_seen = 0usize;
    let mut chain_migrated = 0usize;
    let mut session_seen = 0usize;
    let mut session_migrated = 0usize;
    let mut pack_seen = 0usize;
    let mut pack_migrated = 0usize;

    // Snapshot keys so we can mutate the overlay below.
    let to_consider: Vec<(String, crate::source::SourceAttribution)> = {
        let guard = brain.source_overlay.read().unwrap();
        guard
            .entries
            .iter()
            .filter_map(|(k, attr)| {
                if let crate::source::SourceKind::Teach { actor, .. } = &attr.kind {
                    if actor.starts_with("research_chain:")
                        || actor.starts_with("session_promote:")
                        || actor.starts_with("pack_install:")
                    {
                        return Some((k.clone(), attr.clone()));
                    }
                }
                None
            })
            .collect()
    };

    for (k, attr) in to_consider {
        let actor_str = match &attr.kind {
            crate::source::SourceKind::Teach { actor, .. } => actor.clone(),
            _ => continue,
        };
        if let Some(rest) = actor_str.strip_prefix("research_chain:") {
            chain_seen += 1;
            if !dry_run {
                if let Some(lim) = limit_per_prefix {
                    if chain_migrated >= lim { continue; }
                }
                let new_kind = crate::source::SourceKind::ChainedResearch {
                    chain_id: rest.to_string(),
                    hop: 0,
                    parent_query: "<unknown>".into(),
                };
                let mut new_attr = attr.clone();
                new_attr.kind = new_kind;
                brain.source_overlay.write().unwrap().entries.insert(k, new_attr);
                chain_migrated += 1;
            }
        } else if let Some(rest) = actor_str.strip_prefix("session_promote:") {
            session_seen += 1;
            // Legacy format: "S<sid>/E<eid>". Parse on a best-effort basis.
            let (sid, eid) = parse_session_promote_legacy(rest)
                .unwrap_or_else(|| (rest.to_string(), 0u64));
            if !dry_run {
                if let Some(lim) = limit_per_prefix {
                    if session_migrated >= lim { continue; }
                }
                let new_kind = crate::source::SourceKind::SessionPromote {
                    session_id: sid,
                    entry_id: eid,
                };
                let mut new_attr = attr.clone();
                new_attr.kind = new_kind;
                brain.source_overlay.write().unwrap().entries.insert(k, new_attr);
                session_migrated += 1;
            }
        } else if let Some(rest) = actor_str.strip_prefix("pack_install:") {
            pack_seen += 1;
            if !dry_run {
                if let Some(lim) = limit_per_prefix {
                    if pack_migrated >= lim { continue; }
                }
                let new_kind = crate::source::SourceKind::PackInstall {
                    pack_id: rest.to_string(),
                    publisher_pubkey_hex: String::new(),
                    signed_at_ms: 0,
                };
                let mut new_attr = attr.clone();
                new_attr.kind = new_kind;
                brain.source_overlay.write().unwrap().entries.insert(k, new_attr);
                pack_migrated += 1;
            }
        }
    }

    let total_seen = chain_seen + session_seen + pack_seen;
    let total_migrated = chain_migrated + session_migrated + pack_migrated;
    let text = format!(
        "source_attribution_migrate dry_run={} confirm={} chain={}/{} session={}/{} pack={}/{} total_migrated={} of {}",
        dry_run, confirm,
        chain_migrated, chain_seen,
        session_migrated, session_seen,
        pack_migrated, pack_seen,
        total_migrated, total_seen,
    );
    let structured = json!({
        "dry_run": dry_run,
        "confirm_canonical_mutation": confirm,
        "chain": {"seen": chain_seen, "migrated": chain_migrated},
        "session_promote": {"seen": session_seen, "migrated": session_migrated},
        "pack_install": {"seen": pack_seen, "migrated": pack_migrated},
        "total_seen": total_seen,
        "total_migrated": total_migrated,
    });
    Ok(wrap_content(text, structured))
}

/// Parse a legacy session_promote actor suffix of the form "S<sid>/E<eid>".
/// Returns (session_id_string, entry_id_u64) on success, None otherwise.
fn parse_session_promote_legacy(suffix: &str) -> Option<(String, u64)> {
    let mut parts = suffix.splitn(2, '/');
    let s_part = parts.next()?;
    let e_part = parts.next()?;
    let sid = s_part.strip_prefix('S')?;
    let eid_str = e_part.strip_prefix('E')?;
    let eid: u64 = eid_str.parse().ok()?;
    Some((sid.to_string(), eid))
}

#[cfg(test)]
mod port_sourcekind_variants_migrate_tests {
    use super::*;
    use crate::bond::BondType;
    use crate::seed::Seed;

    fn install(b: &mut Brain, from: &str, to: &str, bt: BondType, attr: crate::source::SourceAttribution) {
        b.source_overlay.write().unwrap().set(Seed::from_label(from), Seed::from_label(to), bt, attr);
    }

    #[test]
    fn migrate_dry_run_does_not_mutate() {
        let mut b = Brain::new();
        install(&mut b, "a", "b", BondType::IsA,
            crate::source::SourceAttribution::teach("research_chain:c1", 0));
        install(&mut b, "c", "d", BondType::IsA,
            crate::source::SourceAttribution::teach("session_promote:S7/E5", 0));
        install(&mut b, "e", "f", BondType::IsA,
            crate::source::SourceAttribution::teach("pack_install:df.test", 0));
        let resp = tool_source_attribution_migrate(&mut b, json!({})).expect("ok");
        let s = resp.get("structuredContent").unwrap();
        assert_eq!(s["chain"]["seen"], 1);
        assert_eq!(s["chain"]["migrated"], 0);
        assert_eq!(s["session_promote"]["seen"], 1);
        assert_eq!(s["session_promote"]["migrated"], 0);
        assert_eq!(s["pack_install"]["seen"], 1);
        assert_eq!(s["pack_install"]["migrated"], 0);
        // Overlay still legacy.
        let attr = b.source_overlay.read().unwrap().get(&Seed::from_label("a"), &Seed::from_label("b"), BondType::IsA);
        assert!(matches!(attr.kind, crate::source::SourceKind::Teach { .. }));
    }

    #[test]
    fn migrate_refuses_without_confirm() {
        let mut b = Brain::new();
        install(&mut b, "a", "b", BondType::IsA,
            crate::source::SourceAttribution::teach("research_chain:c1", 0));
        let err = tool_source_attribution_migrate(
            &mut b,
            json!({"dry_run": false}),
        ).unwrap_err();
        assert_eq!(err.0, -32602);
        assert!(err.1.contains("anti-mutation guard"));
    }

    #[test]
    fn migrate_with_confirm_upgrades_all_three() {
        let mut b = Brain::new();
        install(&mut b, "a", "b", BondType::IsA,
            crate::source::SourceAttribution::teach("research_chain:c1", 0));
        install(&mut b, "c", "d", BondType::IsA,
            crate::source::SourceAttribution::teach("session_promote:S7/E5", 0));
        install(&mut b, "e", "f", BondType::IsA,
            crate::source::SourceAttribution::teach("pack_install:df.test", 0));
        let resp = tool_source_attribution_migrate(
            &mut b,
            json!({"dry_run": false, "confirm_canonical_mutation": true}),
        ).expect("ok");
        let s = resp.get("structuredContent").unwrap();
        assert_eq!(s["total_migrated"], 3);
        // Now first-class.
        let a = b.source_overlay.read().unwrap().get(&Seed::from_label("a"), &Seed::from_label("b"), BondType::IsA);
        assert!(matches!(a.kind, crate::source::SourceKind::ChainedResearch { .. }));
        let c = b.source_overlay.read().unwrap().get(&Seed::from_label("c"), &Seed::from_label("d"), BondType::IsA);
        if let crate::source::SourceKind::SessionPromote { session_id, entry_id } = c.kind {
            assert_eq!(session_id, "7");
            assert_eq!(entry_id, 5);
        } else { panic!("expected SessionPromote"); }
        let e = b.source_overlay.read().unwrap().get(&Seed::from_label("e"), &Seed::from_label("f"), BondType::IsA);
        if let crate::source::SourceKind::PackInstall { pack_id, .. } = e.kind {
            assert_eq!(pack_id, "df.test");
        } else { panic!("expected PackInstall"); }
    }
}

#[cfg(test)]
mod port_s1_revoke_tool_tests {
    use super::*;
    use crate::bond::BondType;
    use crate::node::Confidence;

    fn structured(v: &Value) -> &Value {
        v.get("structuredContent").expect("structuredContent")
    }

    /// Mixed-kind fixture: 2 Teach bonds, 2 Research bonds (one per
    /// wonder seed), 1 Document bond. Mirrors the Port S
    /// `port_s_revoke_by_source_removes_target_bonds` shape so the
    /// invariants align.
    fn fixture_mixed_brain() -> Brain {
        let mut b = Brain::new();
        let teach_attr = crate::source::SourceAttribution::teach("alpha", 1);
        let research_attr_a = crate::source::SourceAttribution::research("W7", Some("https://example.com/a".into()), 2);
        let research_attr_b = crate::source::SourceAttribution::research("W9", Some("https://other.example/b".into()), 3);
        let doc_attr = crate::source::SourceAttribution::document("hashX", None);
        b.teach_with_source("alpha_a", BondType::IsA, "alpha_x", 0.9, Confidence::Verified, teach_attr.clone());
        b.teach_with_source("alpha_b", BondType::Causes, "alpha_y", 0.9, Confidence::Verified, teach_attr);
        b.teach_with_source("r_a", BondType::IsA, "r_x", 0.9, Confidence::Verified, research_attr_a);
        b.teach_with_source("r_b", BondType::HasProperty, "r_y", 0.9, Confidence::Verified, research_attr_b);
        b.teach_with_source("d_a", BondType::PartOf, "d_x", 0.9, Confidence::Verified, doc_attr);
        b
    }

    #[test]
    fn s1_tool_is_registered_in_schema() {
        let defs = tool_definitions();
        let arr = defs.as_array().expect("definitions array");
        let entry = arr.iter().find(|t| t.get("name").and_then(|n| n.as_str()) == Some("lu4d_revoke_by_source"))
            .expect("lu4d_revoke_by_source must be registered");
        let schema = entry.get("inputSchema").expect("inputSchema");
        let req = schema.get("required").and_then(|r| r.as_array()).expect("required");
        assert_eq!(req.len(), 1);
        assert_eq!(req[0].as_str(), Some("kind"));
        // Confirm dry_run + filter knobs are surfaced in the schema.
        let props = schema.get("properties").and_then(|p| p.as_object()).expect("properties");
        for k in ["kind", "dry_run", "corpus", "run_id", "wonder_seed", "url_contains", "actor", "before_ms", "after_ms", "doc_hash"] {
            assert!(props.contains_key(k), "schema missing property '{}'", k);
        }
    }

    #[test]
    fn s1_dry_run_is_safe_no_mutation() {
        let mut b = fixture_mixed_brain();
        let bonds_before = b.network.bond_count();
        let overlay_before = b.source_overlay.read().unwrap().len();
        let resp = tool_revoke_by_source(&mut b, json!({"kind": "Teach", "dry_run": true}))
            .expect("dry_run should succeed");
        let s = structured(&resp);
        assert_eq!(s["matched_count"].as_u64().unwrap(), 2, "two Teach bonds in fixture");
        assert_eq!(s["revoked_count"].as_u64().unwrap(), 0, "dry_run MUST NOT revoke");
        assert_eq!(s["dry_run"].as_bool().unwrap(), true);
        assert_eq!(s["admin_gated"].as_bool().unwrap(), false, "default-closed admin gate (BR.5)");
        // Brain is byte-for-byte unchanged.
        assert_eq!(b.network.bond_count(), bonds_before, "network unchanged");
        assert_eq!(b.source_overlay.read().unwrap().len(), overlay_before, "overlay unchanged");
    }

    #[test]
    fn s1_dry_run_is_default_when_arg_omitted() {
        // CRITICAL safety property: forgetting dry_run MUST default to
        // dry_run=true. Otherwise an honest mistake nukes data.
        let mut b = fixture_mixed_brain();
        let bonds_before = b.network.bond_count();
        let resp = tool_revoke_by_source(&mut b, json!({"kind": "Teach"}))
            .expect("call should succeed");
        let s = structured(&resp);
        assert_eq!(s["dry_run"].as_bool().unwrap(), true, "default MUST be dry_run");
        assert_eq!(s["revoked_count"].as_u64().unwrap(), 0);
        assert_eq!(b.network.bond_count(), bonds_before);
    }

    #[test]
    fn s1_admin_gate_blocks_dry_run_false_by_default() {
        // BR.5 audit P1 #3 internal notes: the default-deny gate
        // closes lu4d_revoke_by_source's destructive path. Operators
        // must call lu4d_admin_set with admin_revoke_enabled=true to
        // open it.
        let mut b = fixture_mixed_brain();
        assert!(!b.admin_revoke_enabled, "BR.5: gate must default to closed");
        let bonds_before = b.network.bond_count();
        let err = tool_revoke_by_source(&mut b, json!({"kind": "Teach", "dry_run": false}))
            .expect_err("default-closed gate must reject dry_run=false");
        assert_eq!(err.0, -32603, "internal_error code for closed gate");
        assert!(err.1.contains("admin_revoke_enabled"), "error must name the flag");
        assert_eq!(b.network.bond_count(), bonds_before, "no mutation when gate refuses");

        // Dry-run preview is NOT gated -- still works.
        let resp = tool_revoke_by_source(&mut b, json!({"kind": "Teach", "dry_run": true}))
            .expect("dry_run preview must bypass the gate");
        let s = structured(&resp);
        assert_eq!(s["matched_count"].as_u64().unwrap(), 2);
        assert_eq!(s["admin_gated"].as_bool().unwrap(), false);
        assert_eq!(b.network.bond_count(), bonds_before);

        // Open the gate and the destructive path proceeds.
        b.admin_revoke_enabled = true;
        let resp = tool_revoke_by_source(&mut b, json!({"kind": "Teach", "dry_run": false}))
            .expect("open gate must permit dry_run=false");
        let s = structured(&resp);
        assert_eq!(s["matched_count"].as_u64().unwrap(), 2);
        assert_eq!(s["revoked_count"].as_u64().unwrap(), 2);
        assert_eq!(s["admin_gated"].as_bool().unwrap(), true);
    }

    #[test]
    fn s1_actual_revoke_removes_target_bonds() {
        let mut b = fixture_mixed_brain();
        // BR.5: open the admin gate so dry_run=false is permitted. The
        // default-deny posture would otherwise fail the test with -32603.
        b.admin_revoke_enabled = true;
        let bonds_before = b.network.bond_count();
        let resp = tool_revoke_by_source(&mut b, json!({"kind": "Teach", "dry_run": false}))
            .expect("call should succeed");
        let s = structured(&resp);
        assert_eq!(s["matched_count"].as_u64().unwrap(), 2);
        assert_eq!(s["revoked_count"].as_u64().unwrap(), 2);
        assert_eq!(s["dry_run"].as_bool().unwrap(), false);
        assert_eq!(b.network.bond_count(), bonds_before - 2, "network shrank by exactly 2");
        // Research and Document bonds are untouched.
        assert!(matches!(
            b.bond_source(&crate::seed::Seed::from_label("r_a"), &crate::seed::Seed::from_label("r_x"), BondType::IsA).kind,
            crate::source::SourceKind::Research { .. }
        ));
        assert!(matches!(
            b.bond_source(&crate::seed::Seed::from_label("d_a"), &crate::seed::Seed::from_label("d_x"), BondType::PartOf).kind,
            crate::source::SourceKind::Document { .. }
        ));
    }

    #[test]
    fn s1_research_wonder_seed_filter_narrows_match() {
        let mut b = fixture_mixed_brain();
        // 2 Research bonds total; only 1 has wonder_seed = W7.
        let resp = tool_revoke_by_source(&mut b, json!({"kind": "Research", "wonder_seed": "W7", "dry_run": true}))
            .expect("call should succeed");
        let s = structured(&resp);
        assert_eq!(s["matched_count"].as_u64().unwrap(), 1, "exactly one bond carries wonder_seed=W7");
        assert_eq!(s["revoked_count"].as_u64().unwrap(), 0);
    }

    #[test]
    fn s1_research_url_contains_filter() {
        let mut b = fixture_mixed_brain();
        let resp = tool_revoke_by_source(&mut b, json!({"kind": "Research", "url_contains": "other.example", "dry_run": true}))
            .expect("call should succeed");
        let s = structured(&resp);
        assert_eq!(s["matched_count"].as_u64().unwrap(), 1, "one Research bond's URL contains 'other.example'");
    }

    #[test]
    fn s1_research_nonexistent_wonder_seed_zero_match() {
        // The exact behavior the live MCP probe asserts on canonical:
        // a wonder seed nobody has produces matched_count = 0 and
        // revoked_count = 0, regardless of dry_run.
        let mut b = fixture_mixed_brain();
        let resp = tool_revoke_by_source(&mut b, json!({"kind": "Research", "wonder_seed": "nonexistent", "dry_run": true}))
            .expect("call should succeed");
        let s = structured(&resp);
        assert_eq!(s["matched_count"].as_u64().unwrap(), 0);
        assert_eq!(s["revoked_count"].as_u64().unwrap(), 0);
    }

    #[test]
    fn s1_unknown_kind_returns_error() {
        let mut b = fixture_mixed_brain();
        let err = tool_revoke_by_source(&mut b, json!({"kind": "Bogus", "dry_run": true}))
            .expect_err("bogus kind must be rejected");
        assert_eq!(err.0, -32602, "invalid_params error code");
        assert!(err.1.contains("unknown kind"), "error msg should explain, got {:?}", err.1);
    }

    #[test]
    fn s1_missing_kind_returns_error() {
        let mut b = fixture_mixed_brain();
        let err = tool_revoke_by_source(&mut b, json!({"dry_run": true}))
            .expect_err("missing kind must be rejected");
        assert_eq!(err.0, -32602);
        assert!(err.1.contains("kind"));
    }

    #[test]
    fn s1_kinds_summary_reports_matched_kind() {
        let mut b = fixture_mixed_brain();
        let resp = tool_revoke_by_source(&mut b, json!({"kind": "Teach", "dry_run": true}))
            .expect("call should succeed");
        let s = structured(&resp);
        let summary = s["kinds_summary"].as_array().expect("kinds_summary array");
        assert_eq!(summary.len(), 1);
        assert_eq!(summary[0]["kind"].as_str(), Some("teach"));
        assert_eq!(summary[0]["count"].as_u64(), Some(2));
    }

    #[test]
    fn s1_dispatch_is_wired_in_handle_tools_call() {
        // Round-trip via the public dispatch entry point — confirms
        // the match arm in handle_tools_call_inner is wired up.
        let mut b = fixture_mixed_brain();
        let params = json!({
            "name": "lu4d_revoke_by_source",
            "arguments": {"kind": "Document", "dry_run": true},
        });
        let resp = handle_tools_call(&mut b, params).expect("dispatch must route");
        let s = structured(&resp);
        assert_eq!(s["matched_count"].as_u64().unwrap(), 1, "one Document bond in fixture");
        assert_eq!(s["revoked_count"].as_u64().unwrap(), 0);
    }
}


// =============================================================================
// Port T-Ingest tests — `lu4d_teach_corpus` MCP wrapper
// =============================================================================
#[cfg(test)]
mod tingest_tests {
    use super::*;
    use crate::brain::Brain;
    use crate::node::Confidence;
    use crate::source::SourceKind;
    use crate::runtime_config::IngestRunIdStrategy;
    use serde_json::json;

    fn structured(v: &Value) -> Value {
        v.get("structuredContent").cloned().unwrap_or(json!({}))
    }

    /// Test 1 — the tool exists and registers a TrainingCorpus stamp directly
    /// (default config: no auto_approve_threshold, no corpus_default).
    #[test]
    fn tingest_tool_exists_and_stamps_training_corpus_directly() {
        let mut b = Brain::new();
        let initial = b.network.bond_count();
        let args = json!({
            "subject": "photosynthesis",
            "relation": "IsA",
            "object": "chemical_process",
            "weight": 0.9,
            "confidence": "Verified",
            "corpus": "wikimedia/wikipedia",
            "source_url": "https://en.wikipedia.org/wiki/Photosynthesis",
        });
        let resp = tool_teach_corpus(&mut b, args).expect("teach_corpus ok");
        let s = structured(&resp);
        assert_eq!(s["gated"].as_bool(), Some(false));
        assert_eq!(s["corpus"].as_str(), Some("wikimedia/wikipedia"));
        assert_eq!(s["source_url"].as_str(), Some("https://en.wikipedia.org/wiki/Photosynthesis"));
        // The standard teach response shape (subject_seed, object_seed,
        // created_subject, etc.) is preserved.
        assert!(!s["subject_seed"].is_null(), "subject_seed missing from response");
        assert!(!s["object_seed"].is_null(), "object_seed missing from response");
        // Bond was added directly.
        assert_eq!(b.network.bond_count(), initial + 1);
    }

    /// Test 2 — the bond formed via lu4d_teach_corpus carries a
    /// `SourceKind::TrainingCorpus { corpus, run_id, url }` attribution.
    /// This is THE bond-level guarantee the 1% retrain test depends on.
    #[test]
    fn tingest_bond_attribution_is_training_corpus_with_url() {
        let mut b = Brain::new();
        let args = json!({
            "subject": "alpha_concept",
            "relation": "HasProperty",
            "object": "beta_property",
            "weight": 0.85,
            "confidence": "Verified",
            "corpus": "stackexchange/cooking",
            "run_id": "my_run_42",
            "source_url": "https://cooking.stackexchange.com/q/12345",
        });
        let _ = tool_teach_corpus(&mut b, args).expect("teach_corpus ok");
        let r = b.teach("alpha_concept", BondType::HasProperty, "beta_property", 0.0, Confidence::Verified);
        // teach() above is a no-op-equivalent reach to recover the seeds without polluting attribution.
        let attr = b.bond_source(&r.subject_seed, &r.object_seed, BondType::HasProperty);
        // After the second teach() call the source overlay was overwritten
        // — but in that second call, ingest_config.corpus_default is None,
        // so the resulting attribution is now Teach. We assert the FIRST
        // call stamped TrainingCorpus by inspecting the overlay BEFORE
        // the second teach() — re-do the test cleanly.
        let _ = attr;

        // Clean re-test: a fresh brain, only one teach_corpus call.
        let mut b2 = Brain::new();
        let args2 = json!({
            "subject": "alpha2",
            "relation": "HasProperty",
            "object": "beta2",
            "weight": 0.85,
            "confidence": "Verified",
            "corpus": "stackexchange/cooking",
            "run_id": "my_run_42",
            "source_url": "https://cooking.stackexchange.com/q/12345",
        });
        let _ = tool_teach_corpus(&mut b2, args2).expect("teach_corpus ok");
        let s_alpha = crate::seed::Seed::from_label("alpha2");
        let s_beta = crate::seed::Seed::from_label("beta2");
        let attr2 = b2.bond_source(&s_alpha, &s_beta, BondType::HasProperty);
        match &attr2.kind {
            SourceKind::TrainingCorpus { corpus, run_id, url } => {
                assert_eq!(corpus, "stackexchange/cooking");
                assert_eq!(run_id.as_deref(), Some("my_run_42"));
                assert_eq!(url.as_deref(), Some("https://cooking.stackexchange.com/q/12345"));
            }
            other => panic!("expected TrainingCorpus, got {:?}", other),
        }
    }

    /// Test 3 — when `auto_approve_threshold = Some(0.9)` and confidence is
    /// Inferred (band 0.65), the call routes through T.3 PendingLog instead
    /// of applying directly. bond_count is unchanged.
    #[test]
    fn tingest_routes_through_t3_when_below_threshold() {
        let mut b = Brain::new();
        b.ingest_config.auto_approve_threshold = Some(0.9);
        let initial = b.network.bond_count();
        let initial_pending = b.pending.pending_count();

        let args = json!({
            "subject": "gated_subject",
            "relation": "IsA",
            "object": "gated_object",
            "weight": 0.7,
            "confidence": "Inferred",
            "corpus": "test/gated",
        });
        let resp = tool_teach_corpus(&mut b, args).expect("teach_corpus ok");
        let s = structured(&resp);
        assert_eq!(s["gated"].as_bool(), Some(true), "expected gated=true");
        assert!(s["pending_id"].as_u64().is_some(), "expected pending_id");
        assert_eq!(s["corpus"].as_str(), Some("test/gated"));
        // No bond on the brain yet.
        assert_eq!(b.network.bond_count(), initial);
        // PendingLog grew by one.
        assert_eq!(b.pending.pending_count(), initial_pending + 1);
        // The pending entry's proposed_by attribution is TrainingCorpus.
        let pid = s["pending_id"].as_u64().unwrap();
        let entry = b.pending.get(pid).expect("pending entry exists");
        match &entry.proposed_by.kind {
            SourceKind::TrainingCorpus { corpus, .. } => {
                assert_eq!(corpus, "test/gated");
            }
            other => panic!("expected TrainingCorpus in pending entry, got {:?}", other),
        }
    }

    /// Test 4 — after T.3 approve+apply, the bond exists with TrainingCorpus.
    #[test]
    fn tingest_t3_approve_then_apply_lands_training_corpus_attribution() {
        let mut b = Brain::new();
        b.ingest_config.auto_approve_threshold = Some(0.9);
        let args = json!({
            "subject": "approved_subject",
            "relation": "IsA",
            "object": "approved_object",
            "weight": 0.5,
            "confidence": "Inferred",
            "corpus": "test/approved",
            "run_id": "r_approve_1",
            "source_url": "https://example.test/x",
        });
        let resp = tool_teach_corpus(&mut b, args).expect("teach_corpus ok");
        let pid = structured(&resp)["pending_id"].as_u64().expect("pending_id");

        b.pending.approve(pid, 1_000_000, "john").unwrap();
        crate::pending::apply_change(&mut b, pid).unwrap();
        assert!(b.pending.get(pid).unwrap().status.is_applied());

        let s_a = crate::seed::Seed::from_label("approved_subject");
        let s_b = crate::seed::Seed::from_label("approved_object");
        let attr = b.bond_source(&s_a, &s_b, BondType::IsA);
        match &attr.kind {
            SourceKind::TrainingCorpus { corpus, run_id, url } => {
                assert_eq!(corpus, "test/approved");
                assert_eq!(run_id.as_deref(), Some("r_approve_1"));
                assert_eq!(url.as_deref(), Some("https://example.test/x"));
            }
            other => panic!("expected TrainingCorpus after T.3 apply, got {:?}", other),
        }
    }

    /// Test 5 — when `ingest_corpus_default = Some("x")`, the BARE
    /// `lu4d_teach` ALSO stamps TrainingCorpus with corpus "x".
    #[test]
    fn tingest_corpus_default_propagates_to_bare_teach() {
        let mut b = Brain::new();
        b.ingest_config.corpus_default = Some("convenience/corpus".to_string());
        let r = b.teach("x_subj", BondType::IsA, "x_obj", 0.9, Confidence::Verified);
        let attr = b.bond_source(&r.subject_seed, &r.object_seed, BondType::IsA);
        match &attr.kind {
            SourceKind::TrainingCorpus { corpus, .. } => {
                assert_eq!(corpus, "convenience/corpus");
            }
            other => panic!("expected TrainingCorpus from corpus_default, got {:?}", other),
        }
    }

    /// Test 6 — `ingest_corpus_default = None` preserves byte-for-byte
    /// the pre-T-Ingest bare lu4d_teach behavior (Teach { actor="system" }).
    /// Regression-equivalence test.
    #[test]
    fn tingest_no_corpus_default_preserves_pre_tingest_teach_attribution() {
        let mut b = Brain::new();
        assert!(b.ingest_config.corpus_default.is_none(), "default must be None");
        let r = b.teach("reg_subj", BondType::IsA, "reg_obj", 0.9, Confidence::Verified);
        let attr = b.bond_source(&r.subject_seed, &r.object_seed, BondType::IsA);
        match &attr.kind {
            SourceKind::Teach { actor, .. } => {
                assert_eq!(actor, "system", "pre-T-Ingest teach uses system actor");
            }
            other => panic!("expected Teach{{system}}, got {:?}", other),
        }
    }

    /// Test 7 — run_id strategy: Caller (no auto-fill).
    #[test]
    fn tingest_run_id_strategy_caller() {
        let mut b = Brain::new();
        b.ingest_config.run_id_strategy = IngestRunIdStrategy::Caller;
        // Caller supplies run_id → use it.
        assert_eq!(b.resolve_ingest_run_id(Some("explicit".into())), Some("explicit".into()));
        // Caller omits run_id → None.
        assert_eq!(b.resolve_ingest_run_id(None), None);
    }

    /// Test 8 — run_id strategy: ServerStartup (default).
    #[test]
    fn tingest_run_id_strategy_server_startup() {
        let mut b = Brain::new();
        b.ingest_config.run_id_strategy = IngestRunIdStrategy::ServerStartup;
        // Caller omits → server fills in startup id.
        let r = b.resolve_ingest_run_id(None);
        assert!(r.is_some());
        assert_eq!(r.as_ref().unwrap(), &b.ingest_run_id);
        // Caller's own value wins.
        assert_eq!(b.resolve_ingest_run_id(Some("my_run".into())), Some("my_run".into()));
    }

    /// Test 9 — run_id strategy: PerSession (falls back to ServerStartup
    /// when no LB session is active).
    #[test]
    fn tingest_run_id_strategy_per_session_no_active_falls_back() {
        let mut b = Brain::new();
        b.ingest_config.run_id_strategy = IngestRunIdStrategy::PerSession;
        // No active LB session → fall back to startup id.
        let r = b.resolve_ingest_run_id(None);
        assert!(r.is_some());
        assert_eq!(r.as_ref().unwrap(), &b.ingest_run_id);
    }

    /// Test 10 — Anti-bypass STRUCTURAL grep guard. Ensures the
    /// `tool_teach_corpus` function body in src/mcp.rs goes through one of
    /// the two sanctioned paths — `teach_corpus` (which itself routes to
    /// `teach_with_source`) for direct apply, or `pending.push` for the
    /// gated path — and NEVER bypasses to `network.add_bond` or
    /// `add_bond_with_scar_check` directly.
    #[test]
    fn tingest_grep_guard_no_t3_bypass() {
        let src = std::fs::read_to_string("src/mcp.rs").expect("read src/mcp.rs");
        // Locate the function body.
        let start = src.find("fn tool_teach_corpus(").expect("tool_teach_corpus exists");
        // Body ends at the next NL+"fn " at column 0 after `start`.
        let after = &src[start..];
        let end_rel = after[1..].find("\nfn ").map(|i| i + 1).unwrap_or(after.len());
        let body = &after[..end_rel];
        // Sanctioned paths.
        // BR.3: accept either the legacy `brain.teach_corpus(` entry
        // point or the kind-aware `brain.teach_corpus_with_kind(`
        // delegator. Both stamp TrainingCorpus attribution and route
        // through `teach_with_source_kind`; only the kind-hint plumbing
        // differs.
        let has_teach_corpus = body.contains("brain.teach_corpus(")
            || body.contains("brain.teach_corpus_with_kind(");
        let has_pending_push = body.contains("brain.pending.push");
        assert!(
            has_teach_corpus && has_pending_push,
            "tool_teach_corpus must include `brain.teach_corpus(` or `brain.teach_corpus_with_kind(` (direct apply) and `brain.pending.push` (T.3 gated path); got teach_corpus={} pending_push={}",
            has_teach_corpus, has_pending_push,
        );
        // Forbidden bypasses — these would let a corpus ingest dodge
        // the gate and the source-overlay stamp.
        for forbidden in [
            "brain.network.add_bond(",
            "add_bond_with_scar_check(",
            "brain.teach(",        // bare teach without source — would default to Teach{system}
            "brain.teach_with_kind(", // direct creation path that would skip the corpus stamp
        ] {
            assert!(
                !body.contains(forbidden),
                "tool_teach_corpus body must not bypass via `{}` (would skip TrainingCorpus stamping or the T.3 gate)",
                forbidden,
            );
        }
    }

    /// Test 11 — empty corpus is rejected (would otherwise collide with
    /// the bare-teach default semantics).
    #[test]
    fn tingest_empty_corpus_rejected() {
        let mut b = Brain::new();
        let args = json!({
            "subject": "x", "relation": "IsA", "object": "y",
            "corpus": "   ",
        });
        let err = tool_teach_corpus(&mut b, args).err().expect("should reject empty corpus");
        assert_eq!(err.0, -32602);
    }

    /// Test 12 — full empirical loop: teach_corpus → pending → approve → apply →
    /// revoke_by_source → bond_count parity. Mirrors the live-brain demo.
    #[test]
    fn tingest_end_to_end_revoke_parity() {
        let mut b = Brain::new();
        b.ingest_config.auto_approve_threshold = Some(0.95);
        let initial = b.network.bond_count();

        // Push gated.
        let resp = tool_teach_corpus(&mut b, json!({
            "subject": "e2e_subject_xyz",
            "relation": "IsA",
            "object": "e2e_object_xyz",
            "weight": 0.5,
            "confidence": "Verified", // 0.90 < 0.95 → gated
            "corpus": "test/t-ingest-demo",
        })).unwrap();
        let pid = structured(&resp)["pending_id"].as_u64().unwrap();

        // Approve + apply.
        b.pending.approve(pid, 1_000_000, "john").unwrap();
        crate::pending::apply_change(&mut b, pid).unwrap();
        assert_eq!(b.network.bond_count(), initial + 1);

        // Revoke by source.
        let (_overlay, bonds_removed) = b.revoke_bonds_by_source(|attr| matches!(
            &attr.kind,
            SourceKind::TrainingCorpus { corpus, .. } if corpus == "test/t-ingest-demo"
        ));
        assert_eq!(bonds_removed, 1);
        assert_eq!(b.network.bond_count(), initial, "bond_count parity after revoke");
    }

    /// Test 13 — runtime_config section #15 round-trips the new ingest knobs.
    /// Defends against future serde drift in the persisted snapshot.
    #[test]
    fn tingest_runtime_config_section_15_round_trip() {
        use crate::runtime_config::{RuntimeConfigSnapshot, IngestRunIdStrategy};
        let mut snap = RuntimeConfigSnapshot::default();
        snap.ingest_corpus_default = Some("my_corpus".into());
        snap.ingest_run_id_strategy = IngestRunIdStrategy::PerSession;
        snap.ingest_auto_approve_threshold = Some(0.7);
        let bytes = snap.encode_v15_section().unwrap();
        let back = RuntimeConfigSnapshot::decode_v15_section(&bytes).unwrap();
        assert_eq!(back.ingest_corpus_default.as_deref(), Some("my_corpus"));
        assert!(matches!(back.ingest_run_id_strategy, IngestRunIdStrategy::PerSession));
        assert!((back.ingest_auto_approve_threshold.unwrap() - 0.7).abs() < 1e-6);
        // A pre-T-Ingest section #15 (no ingest_* fields) decodes with
        // defaults — also defended below for forward compat.
        let pre_tingest = b"{}";
        let defaults = RuntimeConfigSnapshot::decode_v15_section(pre_tingest).unwrap();
        assert!(defaults.ingest_corpus_default.is_none());
        assert!(matches!(defaults.ingest_run_id_strategy, IngestRunIdStrategy::ServerStartup));
        assert!(defaults.ingest_auto_approve_threshold.is_none());
    }

    // ===========================================================
    // Port BatchIngest tests -- `lu4d_teach_corpus_batch` MCP wrapper.
    //
    // The batch tool is a pure IPC speedup over `tool_teach_corpus`.
    // These tests pin (a) structural anti-bypass via grep guard,
    // (b) byte-equivalent brain state vs N sequential calls,
    // (c) cap enforcement at TEACH_CORPUS_BATCH_MAX_ENTRIES,
    // (d) per-entry error isolation (one bad entry does not fail
    // the batch).
    // ===========================================================

    /// Test BI-1 -- Anti-bypass STRUCTURAL grep guard.
    ///
    /// Asserts the `tool_teach_corpus_batch` function body uses ONLY
    /// `tool_teach_corpus(` as its inner write path. Direct routes
    /// (`brain.network.add_bond`, `add_bond_with_scar_check`,
    /// `brain.teach_corpus(`, or `brain.pending.push(`) would skip
    /// per-entry argument parsing, gating, or source attribution.
    /// The FORBIDDEN list is split via `concat!()` so this test
    /// source does not self-trigger the grep (mirrors T-Ingest /
    /// T-Ingest.2 pattern).
    #[test]
    fn tingest_batch_grep_guard_uses_only_tool_teach_corpus() {
        let src = std::fs::read_to_string("src/mcp.rs").expect("read src/mcp.rs");
        let start = src.find("fn tool_teach_corpus_batch(")
            .expect("tool_teach_corpus_batch exists");
        let after = &src[start..];
        let end_rel = after[1..].find("\nfn ").map(|i| i + 1).unwrap_or(after.len());
        let body = &after[..end_rel];

        // Sanctioned inner path -- the only call the body is allowed
        // to make for per-entry write semantics.
        assert!(
            body.contains("tool_teach_corpus(brain, entry)"),
            "tool_teach_corpus_batch must dispatch each entry through tool_teach_corpus(brain, entry)",
        );

        // FORBIDDEN inner-write identifiers -- split via concat!() so
        // this test source does not match itself.
        const FORBIDDEN: &[&str] = &[
            concat!("brain.", "network.add_bond("),
            concat!("add_bond_", "with_scar_check("),
            concat!("brain.", "teach("),
            concat!("brain.", "teach_corpus("),
            concat!("brain.", "teach_with_source("),
            concat!("brain.", "teach_with_kind("),
            concat!("brain.", "teach_with_negation("),
            concat!("brain.", "pending.push("),
        ];
        for f in FORBIDDEN {
            assert!(
                !body.contains(f),
                "tool_teach_corpus_batch body must NOT bypass via `{}` (would skip per-entry validation / T.3 gating / source stamp)",
                f,
            );
        }
    }

    /// Test BI-2 -- Byte-equivalent brain state.
    ///
    /// 100 entries via `lu4d_teach_corpus_batch` (in two calls of 50)
    /// produces a brain with the same (bond_count, node_count,
    /// sorted-bond keys, current weights) as 100 sequential
    /// `lu4d_teach_corpus` calls with the same payloads.
    ///
    /// Per-bond keyframe timestamps are sourced from `now_ms()` and
    /// will differ across runs; we deliberately compare bond shape +
    /// weight, not timestamps. This is the anti-hallucination
    /// invariant: stored bond weights are NEVER mutated by the batch
    /// path; only IPC is amortised.
    #[test]
    fn tingest_batch_state_equivalent_to_sequential() {
        let payloads: Vec<Value> = (0..100)
            .map(|i| serde_json::json!({
                "subject": format!("batch_subject_{:03}", i),
                "relation": "IsA",
                "object": format!("batch_object_{:03}", i),
                "weight": 0.8,
                "confidence": "Verified",
                "corpus": "test/batch_equivalence",
                "run_id": "bi_state_eq",
                "source_url": format!("row://{}", i),
            }))
            .collect();

        // Path A: sequential.
        let mut a = Brain::new();
        for p in &payloads {
            let _ = tool_teach_corpus(&mut a, p.clone()).expect("sequential teach_corpus ok");
        }

        // Path B: batched (two batches of 50).
        let mut b = Brain::new();
        for chunk in payloads.chunks(50) {
            let resp = tool_teach_corpus_batch(
                &mut b,
                serde_json::json!({ "entries": chunk.to_vec() }),
            ).expect("batch teach_corpus ok");
            let s = resp.get("structuredContent").cloned().unwrap_or(Value::Null);
            assert_eq!(s["entries_attempted"].as_u64(), Some(chunk.len() as u64));
            assert_eq!(s["errors"].as_array().map(|v| v.len()), Some(0));
        }

        // (1) bond_count + node_count parity.
        assert_eq!(
            a.network.bond_count(),
            b.network.bond_count(),
            "bond_count must match between sequential and batched paths",
        );
        assert_eq!(
            a.network.node_count(),
            b.network.node_count(),
            "node_count must match between sequential and batched paths",
        );

        // (2) Bond-key set parity (Seed does not derive Ord, so we compare via HashSet).
        let a_set: std::collections::HashSet<(crate::seed::Seed, crate::seed::Seed)> =
            a.network.iter_bond_keys_owned().into_iter().collect();
        let b_set: std::collections::HashSet<(crate::seed::Seed, crate::seed::Seed)> =
            b.network.iter_bond_keys_owned().into_iter().collect();
        assert_eq!(a_set, b_set, "bond-key set must match");
        let a_keys: Vec<_> = a_set.iter().copied().collect();

        // (3) Per-bond stored-weight parity.
        //
        // We compare `latest_raw_weight()` -- the as-stored numeric
        // weight in the most recent keyframe -- NOT `weight_at(t_now)`,
        // which applies time decay relative to a read timestamp; the
        // read timestamp would differ by O(ms) between A and B, so
        // the decayed reading would diverge by ~1 ULP. Stored
        // weights are written once by the same teach arithmetic per
        // entry, so they are bit-identical.
        //
        // Anti-hallucination invariant: the batch path must NOT
        // mutate stored bond weights. This is the load-bearing
        // assertion.
        for k in &a_keys {
            let av = a.network.get_bond(&k.0, &k.1)
                .expect("bond present in A");
            let bv = b.network.get_bond(&k.0, &k.1)
                .expect("bond present in B");
            assert_eq!(av.bond_type(), bv.bond_type(), "bond_type mismatch for {:?}", k);
            assert_eq!(
                av.keyframe_count(),
                bv.keyframe_count(),
                "keyframe_count mismatch for {:?}", k,
            );
            // Bit-exact stored-weight equality.
            let aw = av.latest_raw_weight();
            let bw = bv.latest_raw_weight();
            assert_eq!(
                aw.to_bits(),
                bw.to_bits(),
                "latest_raw_weight mismatch for {:?}: a={} b={}", k, aw, bw,
            );
        }
    }

    /// Test BI-3 -- Cap enforcement: 1001 entries returns -32602.
    #[test]
    fn tingest_batch_caps_at_1000() {
        let mut b = Brain::new();
        let entries: Vec<Value> = (0..(TEACH_CORPUS_BATCH_MAX_ENTRIES + 1))
            .map(|i| serde_json::json!({
                "subject": format!("cap_s_{}", i),
                "relation": "IsA",
                "object": format!("cap_o_{}", i),
                "corpus": "test/cap",
            }))
            .collect();
        let err = tool_teach_corpus_batch(&mut b, serde_json::json!({"entries": entries})).err()
            .expect("oversized batch must be rejected");
        assert_eq!(err.0, -32602);
        assert!(err.1.contains("exceeds cap"), "error msg: {}", err.1);
    }

    /// Test BI-4 -- Empty batch returns -32602 (rather than silently
    /// succeeding with zero work).
    #[test]
    fn tingest_batch_empty_rejected() {
        let mut b = Brain::new();
        let err = tool_teach_corpus_batch(&mut b, serde_json::json!({"entries": []})).err()
            .expect("empty batch must be rejected");
        assert_eq!(err.0, -32602);
    }

    /// Test BI-5 -- Per-entry errors do NOT fail the whole batch.
    /// 5 valid entries + 1 with empty corpus = 5 landed bonds and
    /// 1 entry-level error in the response.
    #[test]
    fn tingest_batch_per_entry_errors_isolated() {
        let mut b = Brain::new();
        let initial = b.network.bond_count();
        let entries = vec![
            serde_json::json!({"subject":"pe_s_0","relation":"IsA","object":"pe_o_0","corpus":"test/per_entry"}),
            serde_json::json!({"subject":"pe_s_1","relation":"IsA","object":"pe_o_1","corpus":"test/per_entry"}),
            // Bad entry -- corpus is empty, tool_teach_corpus rejects.
            serde_json::json!({"subject":"pe_s_2","relation":"IsA","object":"pe_o_2","corpus":"   "}),
            serde_json::json!({"subject":"pe_s_3","relation":"IsA","object":"pe_o_3","corpus":"test/per_entry"}),
            serde_json::json!({"subject":"pe_s_4","relation":"IsA","object":"pe_o_4","corpus":"test/per_entry"}),
            serde_json::json!({"subject":"pe_s_5","relation":"IsA","object":"pe_o_5","corpus":"test/per_entry"}),
        ];
        let resp = tool_teach_corpus_batch(&mut b, serde_json::json!({"entries": entries}))
            .expect("batch should not fail just because one entry is invalid");
        let s = resp.get("structuredContent").cloned().unwrap_or(Value::Null);
        assert_eq!(s["entries_attempted"].as_u64(), Some(6));
        assert_eq!(s["bonds_landed"].as_u64(), Some(5));
        let errs = s["errors"].as_array().expect("errors array").clone();
        assert_eq!(errs.len(), 1, "exactly one entry should have errored");
        assert_eq!(errs[0]["index"].as_u64(), Some(2), "the bad entry was at index 2");
        assert_eq!(errs[0]["code"].as_i64(), Some(-32602));
        assert_eq!(
            b.network.bond_count(),
            initial + 5,
            "5 good entries should each land a bond",
        );
    }

    /// Test BI-6 -- T.3 gated entries are reported under bonds_gated
    /// (NOT bonds_landed) and a pending_id exists per such entry.
    #[test]
    fn tingest_batch_t3_gating_reported_correctly() {
        let mut b = Brain::new();
        b.ingest_config.auto_approve_threshold = Some(0.95);
        let initial_bonds = b.network.bond_count();
        let initial_pending = b.pending.iter().count();

        let entries: Vec<Value> = (0..3).map(|i| serde_json::json!({
            "subject": format!("gate_s_{}", i),
            "relation": "IsA",
            "object": format!("gate_o_{}", i),
            "weight": 0.5,
            "confidence": "Inferred",   // 0.65 < 0.95 threshold
            "corpus": "test/gating",
        })).collect();

        let resp = tool_teach_corpus_batch(&mut b, serde_json::json!({"entries": entries})).unwrap();
        let s = resp.get("structuredContent").cloned().unwrap_or(Value::Null);
        assert_eq!(s["bonds_gated"].as_u64(), Some(3));
        assert_eq!(s["bonds_landed"].as_u64(), Some(0));
        // Bond count unchanged -- gated entries go to PendingLog, not network.
        assert_eq!(b.network.bond_count(), initial_bonds);
        // PendingLog grew by 3.
        assert_eq!(b.pending.iter().count(), initial_pending + 3);
    }

    // ===========================================================
    // Port T-Ingest.2 tests -- `lu4d_user_statement` MCP wrapper.
    //
    // Asymmetry invariant: the prompt path MUST land in Left Brain
    // operational memory (SessionEntryKind::UserStatement) and
    // MUST NOT move bond_count by even 1. Each test below pins a
    // facet of that invariant.
    // ===========================================================

    /// Test 1 -- tool exists, accepts the new args, and returns the
    /// proper response shape with `kind: "user_statement"` echoed.
    #[test]
    fn tingest2_tool_basic_response_shape() {
        let mut b = Brain::new();
        let resp = tool_user_statement(&mut b, json!({
            "text": "Write a haiku about autumn leaves.",
            "source_label": "test/t-ingest2-demo",
        })).expect("user_statement ok");
        let s = structured(&resp);
        assert_eq!(s["kind"], json!("user_statement"));
        assert_eq!(s["source_label"], json!("test/t-ingest2-demo"));
        assert_eq!(s["asymmetry_invariant"], json!("no_bond_formation"));
        assert!(s["entry_id"].as_u64().is_some());
        assert!(s["session_id"].as_u64().is_some());
        assert_eq!(s["byte_len"], json!(34));
    }

    /// Test 2 -- bond_count UNCHANGED before and after the call.
    /// This is the load-bearing assertion of T-Ingest.2.
    #[test]
    fn tingest2_bond_count_unchanged_single_call() {
        let mut b = Brain::new();
        let before = b.network.bond_count();
        let _resp = tool_user_statement(&mut b, json!({
            "text": "Explain the doppler effect to a 5-year-old.",
            "source_label": "test/t-ingest2-demo",
        })).expect("ok");
        let after = b.network.bond_count();
        assert_eq!(before, after, "prompt ingestion must NOT move bond_count");
    }

    /// Test 3 -- the entry is queryable via session_memory_inspect
    /// and shows up with kind UserStatement.
    #[test]
    fn tingest2_entry_queryable_with_correct_kind() {
        let mut b = Brain::new();
        let resp = tool_user_statement(&mut b, json!({
            "text": "What if the moon were made of cheese?",
            "source_label": "test/t-ingest2-hypothetical",
        })).unwrap();
        let session_id = structured(&resp)["session_id"].as_u64().unwrap();
        let entry_id = structured(&resp)["entry_id"].as_u64().unwrap();

        let inspect = crate::mcp_lb::tool_session_memory_inspect(
            &b,
            json!({ "session_id": session_id, "kind": "user_statement" }),
        ).unwrap();
        let entries = structured(&inspect)["entries"].as_array().unwrap().clone();
        let found = entries.iter().any(|e| {
            e["entry_id"].as_u64() == Some(entry_id)
                && e["kind"].as_str() == Some("user_statement")
        });
        assert!(found, "entry must be queryable as user_statement; got {:?}", entries);
    }

    /// Test 4 -- Anti-bond-formation STRUCTURAL grep guard.
    ///
    /// Asserts the `tool_user_statement` function body uses ONLY
    /// `brain.lb_session.push_internal` for the canonical write
    /// path and NEVER reaches for any bond-mutating API. The
    /// FORBIDDEN identifier list is split via `concat!()` at
    /// compile time so the test source itself doesn't self-trigger
    /// the grep (mirrors Sense / T-Ingest pattern).
    #[test]
    fn tingest2_grep_guard_no_bond_formation() {
        let src = std::fs::read_to_string("src/mcp.rs").expect("read src/mcp.rs");
        let start = src.find("fn tool_user_statement(")
            .expect("tool_user_statement exists");
        let after = &src[start..];
        let end_rel = after[1..].find("\nfn ").map(|i| i + 1).unwrap_or(after.len());
        let body = &after[..end_rel];

        // Sanctioned write path -- the only LB write the body is
        // allowed to reach for.
        assert!(
            body.contains("lb_session") && body.contains("push_internal"),
            "tool_user_statement body must use brain.lb_session.push_internal",
        );

        // FORBIDDEN identifiers split at compile time so this
        // test source doesn't self-trigger the grep.
        const FORBIDDEN: &[&str] = &[
            concat!("brain.", "teach("),
            concat!("brain.", "teach_corpus("),
            concat!("brain.", "teach_with_source("),
            concat!("brain.", "teach_with_kind("),
            concat!("brain.", "teach_with_negation("),
            concat!("brain.", "teach_from_text("),
            concat!("network.", "add_bond("),
            concat!("add_bond_", "with_scar_check("),
            concat!("pending.", "push("),
            concat!(".propose_to_", "pending("),
        ];
        for f in FORBIDDEN {
            assert!(
                !body.contains(f),
                "tool_user_statement body must NOT reference bond-mutating API '{}': prompts must not become Right Brain bonds (asymmetry invariant)",
                f,
            );
        }
    }

    /// Test 5 -- session_id = None creates an ephemeral session
    /// and the returned session_id is reusable.
    #[test]
    fn tingest2_ephemeral_session_creation() {
        let mut b = Brain::new();
        let resp = tool_user_statement(&mut b, json!({
            "text": "first prompt",
        })).unwrap();
        let s = structured(&resp);
        let sid = s["session_id"].as_u64().unwrap();
        assert_eq!(s["ephemeral_session"], json!(true));
        assert_ne!(sid, 0);
        // The returned session_id is usable to attach more entries.
        let resp2 = tool_user_statement(&mut b, json!({
            "text": "second prompt attached to first session",
            "session_id": sid,
        })).unwrap();
        let s2 = structured(&resp2);
        assert_eq!(s2["session_id"].as_u64().unwrap(), sid);
        assert_eq!(s2["ephemeral_session"], json!(false));
    }

    /// Test 6 -- session_id = Some(existing) attaches to that session.
    #[test]
    fn tingest2_attach_to_caller_session() {
        let mut b = Brain::new();
        let sid = b.session_start();
        let resp = tool_user_statement(&mut b, json!({
            "text": "prompt attached to existing session",
            "session_id": sid,
        })).unwrap();
        let s = structured(&resp);
        assert_eq!(s["session_id"].as_u64().unwrap(), sid);
        assert_eq!(s["ephemeral_session"], json!(false));
    }

    /// Test 7 -- source_label shows up in the entry metadata so
    /// downstream audits can group entries by ingest source.
    #[test]
    fn tingest2_source_label_visible_in_entry() {
        let mut b = Brain::new();
        let label = "HuggingFaceH4/no_robots";
        let resp = tool_user_statement(&mut b, json!({
            "text": "Tell me a story about a cat.",
            "source_label": label,
        })).unwrap();
        let session_id = structured(&resp)["session_id"].as_u64().unwrap();
        let inspect = crate::mcp_lb::tool_session_memory_inspect(
            &b,
            json!({ "session_id": session_id }),
        ).unwrap();
        let entries = structured(&inspect)["entries"].as_array().unwrap().clone();
        assert!(
            entries.iter().any(|e| {
                e["subject"].as_str() == Some(label)
                    && e["content"].as_str().unwrap_or("").contains(label)
            }),
            "source_label must appear in entry subject + content; got {:?}",
            entries,
        );
    }

    /// Test 8 -- regression: 100 calls in a loop leave bond_count
    /// unchanged (no cumulative drift).
    #[test]
    fn tingest2_no_cumulative_bond_drift() {
        let mut b = Brain::new();
        let before = b.network.bond_count();
        for i in 0..100 {
            let _ = tool_user_statement(&mut b, json!({
                "text": format!("prompt number {}", i),
                "source_label": "test/t-ingest2-loop",
            })).unwrap();
        }
        let after = b.network.bond_count();
        assert_eq!(before, after, "100 prompts must leave bond_count exactly unchanged");
    }

    /// Test 9 -- empty text is rejected (-32602).
    #[test]
    fn tingest2_empty_text_rejected() {
        let mut b = Brain::new();
        let err = tool_user_statement(&mut b, json!({"text": ""})).err().unwrap();
        assert_eq!(err.0, -32602);
    }

    /// Test 10 -- text length over `tier2_prompt_max_text_bytes` is rejected.
    #[test]
    fn tingest2_max_text_bytes_cap() {
        let mut b = Brain::new();
        b.tier2_prompt_config.max_text_bytes = 16;
        let err = tool_user_statement(&mut b, json!({
            "text": "this string is definitely longer than sixteen bytes",
        })).err().unwrap();
        assert_eq!(err.0, -32602);
    }

    /// Test 11 -- runtime_config section #15 round-trips the new
    /// Tier 2 prompt knobs (forward/backward compat).
    #[test]
    fn tingest2_runtime_config_section_15_round_trip() {
        use crate::runtime_config::{RuntimeConfigSnapshot, Tier2PromptSessionStrategy};
        let mut snap = RuntimeConfigSnapshot::default();
        snap.tier2_prompt_default_session_strategy = Tier2PromptSessionStrategy::Reuse;
        snap.tier2_prompt_max_text_bytes = 1024;
        let bytes = snap.encode_v15_section().unwrap();
        let back = RuntimeConfigSnapshot::decode_v15_section(&bytes).unwrap();
        assert!(matches!(back.tier2_prompt_default_session_strategy, Tier2PromptSessionStrategy::Reuse));
        assert_eq!(back.tier2_prompt_max_text_bytes, 1024);
        // Pre-T-Ingest.2 section #15 (no tier2_* fields) decodes with defaults.
        let pre = b"{}";
        let defaults = RuntimeConfigSnapshot::decode_v15_section(pre).unwrap();
        assert!(matches!(defaults.tier2_prompt_default_session_strategy, Tier2PromptSessionStrategy::Ephemeral));
        assert_eq!(defaults.tier2_prompt_max_text_bytes, 65536);
    }
}
#[cfg(test)]
mod op_greet_refusal_tool_ask_tests {
    use super::*;

    fn ask(query: &str) -> serde_json::Value {
        let mut b = Brain::new();
        let args = serde_json::json!({"query": query, "depth": 2, "top_k": 5});
        tool_ask(&mut b, args).expect("tool_ask")
    }

    fn first_text(v: &serde_json::Value) -> String {
        v.get("content")
            .and_then(|c| c.get(0))
            .and_then(|t| t.get("text"))
            .and_then(|t| t.as_str())
            .expect("content[0].text")
            .to_string()
    }

    fn structured<'a>(v: &'a serde_json::Value) -> &'a serde_json::Value {
        v.get("structuredContent").expect("structuredContent")
    }

    #[test]
    fn tool_ask_hello_returns_greet_refusal() {
        let resp = ask("hello");
        let text = first_text(&resp);
        assert!(text.starts_with("Hello."),
            "expected canned greeting prose from tool_ask, got {:?}", text);
        let sc = structured(&resp);
        assert_eq!(sc.get("source").and_then(|v| v.as_str()), Some("greet_refusal"));
        assert_eq!(
            sc.get("refusal_reason").and_then(|v| v.as_str()),
            Some("greet_refusal:greeting"));
    }

    #[test]
    fn tool_ask_thank_you_returns_thanks_refusal() {
        let resp = ask("thank you");
        let text = first_text(&resp);
        assert!(text.starts_with("You're welcome"),
            "expected canned thanks prose from tool_ask, got {:?}", text);
        let sc = structured(&resp);
        assert_eq!(
            sc.get("refusal_reason").and_then(|v| v.as_str()),
            Some("greet_refusal:thanks"));
    }

    #[test]
    fn tool_ask_sorry_returns_apology_refusal() {
        let resp = ask("sorry");
        let sc = structured(&resp);
        assert_eq!(
            sc.get("refusal_reason").and_then(|v| v.as_str()),
            Some("greet_refusal:apology"));
    }

    #[test]
    fn tool_ask_real_query_not_affected() {
        // "mitochondria" must NOT route through greet_refusal.
        let resp = ask("mitochondria");
        let sc = structured(&resp);
        // greet_refusal sets source="greet_refusal"; any other path
        // either omits source or uses "idk_gate" / brain.ask result.
        let src_tag = sc.get("source").and_then(|v| v.as_str()).unwrap_or("");
        assert_ne!(src_tag, "greet_refusal",
            "mitochondria must not trigger greet refusal, got source={:?}", src_tag);
    }

    #[test]
    fn tool_ask_substring_not_affected() {
        let resp = ask("say hello to the world");
        let sc = structured(&resp);
        let src_tag = sc.get("source").and_then(|v| v.as_str()).unwrap_or("");
        assert_ne!(src_tag, "greet_refusal",
            "substring match must not trigger greet refusal, got source={:?}", src_tag);
    }
}
