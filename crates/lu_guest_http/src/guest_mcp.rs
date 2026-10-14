//! The guest MCP trust gate (OP-007 / Port U).
//!
//! This is the SINGLE gate the guest HTTP shim routes through (the project log
//! p9 #540: "MCP stays the single trust gate"). It exposes ONLY the
//! guest-safe tools — `lu4d_ask` + session tools — and dispatches them
//! into the *same* narrate/ask pipeline the admin surface uses
//! (`Brain::narrate`, whose idk-gate refusal path is the anti-
//! hallucination floor).
//!
//! Privilege separation is enforced at TWO levels:
//!   1. Compile-time (the real guarantee): this crate builds `lu4d` with
//!      `default-features = false`, so the admin `mcp` module — where the
//!      admin tool handlers and their tool_definitions strings live — is
//!      not compiled into this binary at all.
//!   2. Runtime (defense in depth): the `dispatch` match below only knows
//!      the three guest tools; any other name is refused.
//!
//! The `tests/privsep.rs` integration test verifies level (1) by string-
//! scanning the built guest binary for admin tool names.

use crate::auth::Principal;
use lu4d::Brain;
use serde_json::{json, Value};

/// The complete set of tool names reachable on the guest surface.
pub const GUEST_TOOLS: &[&str] = &["lu4d_ask", "lu4d_session_start", "lu4d_session_end", "lu4d_image_describe", "lu4d_file_read", "lu4d_code", "lu4d_geometry_solve"];

/// MCP `tools/call` result envelope (guest-local copy; the admin
/// `mcp::wrap_content` is compiled out of this binary).
fn wrap_content(text: String, structured: Value) -> Value {
    json!({
        "content": [{"type": "text", "text": text}],
        "structuredContent": structured,
    })
}

/// The guest `tools/list` payload — exactly the three guest tools.
pub fn tools_list() -> Value {
    json!([
        {
            "name": "lu4d_ask",
            "description": "Ask Lu a question. Routes through Lu's narrate/ask pipeline and returns Lu's narrated answer. Refusals (\"I don't have meaningful information about that\") are produced by the idk gate inside the pipeline.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The question to ask Lu."},
                    "depth": {"type": "integer", "description": "Spread-activation depth (default 3)."},
                    "top_k": {"type": "integer", "description": "Max activations to consider (default 10)."}
                    ,"session_id": {"type": "integer", "description": "Optional Q.3 session id from lu4d_session_start. Enables conversational memory: pronoun resolution, tell-me-more elaboration, and backreference (what was my first question)."}
                },
                "required": ["query"]
            }
        },
        {
            "name": "lu4d_session_start",
            "description": "Start a conversation session and return its id.",
            "inputSchema": {"type": "object", "properties": {}}
        },
        {
            "name": "lu4d_session_end",
            "description": "End the active conversation session.",
            "inputSchema": {"type": "object", "properties": {}}
        },
        {
            "name": "lu4d_code",
            "description": "OWNER-ONLY. Write and run a small program from a natural-language request. Composes from grounded ops or a taught pattern, runs it in an isolated sandbox, and returns the working code + output; aborts honestly if it cannot ground the request.",
            "inputSchema": {"type": "object", "properties": {"prompt": {"type": "string"}, "input": {"type": "string", "description": "optional sample file content for file-reading programs"}}, "required": ["prompt"]}
        },
        {
            "name": "lu4d_geometry_solve",
            "description": "Solve a geometry problem exactly and return an exact a+b*pi K-gated answer with verification. TWO input modes. (1) IMAGE mode: pass image_base64 = the base64 bytes of a geometry PHOTO (JPEG or PNG). Lu perceives the pixels directly (rectify -> arcs -> OCR the printed scale -> reconstruct the scene) and solves the shaded area with the verified partition solver, returning the exact answer plus the extracted-scene provenance (class, OCR-d scale, arcs, skew, confidence). If it cannot genuinely parse the image (decode/rectify/arc/OCR/low-confidence) or it is not a geometry figure, it returns an honest capability statement with NO number -- it never guesses. (2) STRUCTURED mode: scene=red_leaf (shaded-area partition, verified 32*pi-64); scene=control with tangent_cm (rectangle + corner quarter-circle + semicircle tangent construction, area = t squared); or scene=partition with whole (Exact a+b*pi), tiles and target_role, solved by whole minus sum of known tiles with the sum-to-whole verifier and IDK/K-gate. Under-constrained or inconsistent scenes return an honest refusal with NO number.",
            "inputSchema": {"type": "object", "properties": {"image_base64": {"type": "string", "description": "base64 bytes of a geometry photo (JPEG/PNG); image mode"}, "scene": {"type": "string"}, "tangent_cm": {"type": "number"}, "whole": {"type": "object"}, "tiles": {"type": "array"}, "target_role": {"type": "string"}, "idk_floor": {"type": "number"}}, "required": []}
        }
    ])
}

/// Dispatch a guest `tools/call`. `_principal` is the identity injected by
/// the HTTP shim (owner vs guest); the guest surface grants the same three
/// tools to both, so it is currently used only for logging/attribution,
/// but threading it here keeps identity present at the trust gate.
pub fn dispatch(
    brain: &mut Brain,
    principal: &Principal,
    name: &str,
    args: &Value,
) -> Result<Value, (i64, String)> {
    match name {
        "lu4d_ask" => {
            // PART B: "give me an example of <known language>" is a request to
            // DEMONSTRATE code, not a recall query. Route owner requests to the
            // code faculty so Lu synthesises + runs a real sample. Non-language
            // "example of X" (e.g. "example of a mammal") returns None and stays
            // a normal recall/Q&A query.
            let owner = matches!(principal, Principal::Owner);
            let q_owned = args
                .get("query")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let ql = q_owned.to_lowercase();
            // PART B (existing): "give me an example of <known language>".
            let lang_example =
                owner && lu4d::code_exec::planner::detect_language_example(&ql).is_some();
            // OP-ROUTE-GUEST-6 (#1529 class A): imperative code synthesis. The
            // QueryRouter already DETECTS code_intent but nothing consumed it --
            // the same Phase-5 gap the image/file flags had in guest_ask. Route
            // an imperative code request to the code faculty ONLY when the
            // planner can actually ground a real runnable program; otherwise
            // fall through to narrate() for honest recall/refuse. Never
            // fabricates: the divert is gated on plan_only yielding code, and a
            // recall query never produces a plan.
            let code_synth = owner
                && !lang_example
                && lu4d::code_exec::planner::looks_like_code_request(&ql)
                && {
                    // Divert ONLY to a self-contained runnable program. A
                    // data-plumbing compose (input_name Some) needs a file the
                    // ask path has not supplied, so it must fall through to
                    // recall rather than emit a program that cannot run.
                    let plan = lu4d::code_exec::planner::plan_only(brain, &q_owned);
                    plan.code.is_some() && plan.input_name.is_none()
                };
            if lang_example || code_synth {
                guest_code(brain, principal, args)
            } else {
                guest_ask(brain, args)
            }
        }
        "lu4d_code" => guest_code(brain, principal, args),
        "lu4d_image_describe" => guest_image_describe(brain, args),
        "lu4d_geometry_solve" => lu4d::math::geometry_tool::solve_scene_json(args),
        "lu4d_file_read" => guest_file_read(brain, args),
        "lu4d_session_start" => {
            let id = brain.session_start();
            Ok(wrap_content(
                format!("Started session {}", id),
                json!({"session_id": id}),
            ))
        }
        "lu4d_session_end" => match brain.session_end() {
            Some(id) => Ok(wrap_content(
                format!("Ended session {}", id),
                json!({"session_id": id}),
            )),
            None => Ok(wrap_content(
                "No active session".to_string(),
                json!({"session_id": null}),
            )),
        },
        // Anything else — including every admin tool — is refused. The
        // admin handlers are not even linked into this binary; this arm is
        // the runtime backstop.
        other => Err((
            -32601,
            format!("tool '{}' is not available on the guest surface", other),
        )),
    }
}

/// OP-ROUTE-COH: rewrite a TOPICAL meta frame -- "what do you know about X",
/// "what have you learned about X", "what do you remember about X" -- to the
/// bare subject X. "what do you know about python" is a question ABOUT python,
/// not a request for a brain census. The IntentClassifier already declines
/// MetaSelf once a topical remainder survives (narration::has_topical_remainder),
/// but nothing rewrote the query, so recall still seeded the full noisy
/// "what do you know about X" string and spread into a bridge word-salad
/// ("Most likely  relates to  via . I considered 7 alternatives ..."). Seeding
/// the BARE subject surfaces X's real grounded neighbours instead.
///
/// Returns None (leave the query untouched) when the frame does not apply or
/// no subject follows the "about" -- bare "what do you know" still routes to
/// the MetaSelf census. Honesty-floor safe: a nonsense subject resolves to
/// nothing and the idk gate still refuses. Mirrors the "tell me about X"
/// normalization above; surfaces only REAL facts, fabricates nothing.
fn normalize_topical_meta(raw: &str) -> Option<String> {
    let ql = raw.trim().to_lowercase();
    const METAS: [&str; 3] = [
        "what do you know about ",
        "what have you learned about ",
        "what do you remember about ",
    ];
    for pre in METAS {
        if ql.starts_with(pre) {
            let mut subject = raw.trim()[pre.len()..]
                .trim()
                .trim_end_matches(|c: char| c == '?' || c == '.' || c == '!')
                .trim();
            // Strip a leading article so recall seeds the clean concept token.
            for art in ["the ", "a ", "an "] {
                if let Some(r) = subject.strip_prefix(art) {
                    subject = r.trim();
                    break;
                }
            }
            if subject.is_empty() || matches!(subject, "the" | "a" | "an") {
                return None;
            }
            return Some(subject.to_string());
        }
    }
    None
}

/// Guest `lu4d_ask`: greet-refusal guard (mirrors the admin ask path),
/// then the real narrate/ask pipeline. Read-only: never persists.
fn guest_ask(brain: &mut Brain, args: &Value) -> Result<Value, (i64, String)> {
    let query = args
        .get("query")
        .and_then(|v| v.as_str())
        .ok_or((-32602, "missing required 'query' string".to_string()))?;

    // OP-ROUTE-GUEST -- arithmetic short-circuit, mirroring the admin
    // ask path's `try_arithmetic_response_with` (src/mcp.rs:1543).
    // Math takes precedence over the greet guard here exactly as it does
    // at the admin MCP boundary.
    //
    // WHY THIS LIVES HERE AND NOT IN narrate(): `QueryRouter::route`
    // already sets `RoutingDecision::needs_math` correctly for these
    // queries, but narrate() consumes that flag ONLY as a negative
    // suppressor (it switches reasoning / variants / thinking OFF at
    // narration.rs:3138/3156/3191) -- OP-ROUTE-2 shipped the router
    // "observation-only" and the positive math handler was never added
    // to narrate(). The only positive handler is in `lu4d::mcp`, which
    // `default-features = false` compiles OUT of this binary. So on the
    // guest surface "2+2" fell through to recall, matched nothing, and
    // hit the idk floor. `lu4d::math` is ungated and IS linked in, so
    // the intercept belongs at this MCP boundary.
    if let Some(intent) = lu4d::math::detect_arithmetic(query) {
        return Ok(match intent.compute() {
            Ok(result) => wrap_content(
                format!(
                    "{}\n\nProvenance: computed by Lu's arithmetic primitive, not recalled.",
                    result.prose
                ),
                json!({
                    "source": "arithmetic_primitive",
                    "intent": result.intent,
                    "value": result.value,
                    "prose": result.prose,
                    "query": query,
                }),
            ),
            Err(e) => wrap_content(
                format!(
                    "Cannot compute: {}.\n\nProvenance: routed to Lu's arithmetic primitive but computation failed.",
                    e
                ),
                json!({
                    "source": "arithmetic_primitive",
                    "intent": intent,
                    "error": e,
                    "query": query,
                }),
            ),
        });
    }

    // OP-ROUTE-GUEST-2 -- full math cascade after the F.1 arithmetic block:
    // F.4 construction -> F.3 word-problem -> F.2 expression/units/stats/solver,
    // mirroring the admin try_arithmetic_response_with (src/mcp.rs:1543). Every
    // detector is &str -> Option pure compute (no &mut Brain); lu4d::math is
    // ungated and linked into the guest. internal notes: guest = read/compute.
    if let Some(resp) = try_guest_math(&*brain, query) {
        return Ok(resp);
    }

    // Defense-in-depth pleasantry refusal, matching the admin ask path.
    if let Some(gclass) = lu4d::greet_refusal::is_greeting(query) {
        let prose = lu4d::greet_refusal::refusal_prose(gclass).to_string();
        let reason = lu4d::greet_refusal::refusal_reason(gclass);
        return Ok(wrap_content(
            prose,
            json!({"source": "greet_refusal", "refusal_reason": reason, "raw_query": query}),
        ));
    }

    // OP-ROUTE-GUEST-6 -- Port Q.3 conversational/session memory into the guest
    // path. Session state lives in brain.session_memories (an in-memory episodic
    // registry) + the lb_session store -- NOT nodes/bonds -- so recording a turn
    // is guest-safe on the read-only brain internal notes Faculty 6 / #1356
    // dual-mode: session read + turn-write = guest-safe; promotion = admin).
    // Mirrors the admin tool_ask Q.3 pre/post-pass (src/mcp.rs:2248 / 2540).
    let raw_query = query.to_string();
    let session_id = args.get("session_id").and_then(|v| v.as_u64());

    // OP-QA-3: knowledge-enumeration meta ("tell me all/everything you know").
    // Lu's knowledge is a web of concepts, not a list it can recite; answering
    // it from top activations produced a stitched "several strands" word-salad.
    // Decline honestly BEFORE recall.
    if lu4d::session_memory::is_knowledge_enumeration(&raw_query) {
        return Ok(wrap_content(
            "I can't list out everything I know -- what I hold is a web of concepts, not a catalog I can recite. Ask me about a specific topic and I'll tell you what I have on it.".to_string(),
            json!({"source": "meta_enumeration", "raw_query": raw_query}),
        ));
    }

    // OP-QA-3: conversation-history backreference ("what was my first/last
    // question?", "what did you just say?"). Answer from session memory when we
    // have it; otherwise decline HONESTLY rather than route the question into
    // recall (which fabricated a "several strands" answer from unrelated
    // word-senses). "I don't keep a record" beats a made-up strand-answer.
    if let Some(kind) = lu4d::session_memory::classify_backreference(&raw_query) {
        let from_memory = session_id
            .and_then(|sid| brain.session_memories.get(sid))
            .and_then(|mem| match kind {
                lu4d::session_memory::BackrefKind::FirstQuestion => {
                    mem.first_turn().map(|t| t.raw_query.clone())
                }
                lu4d::session_memory::BackrefKind::LastQuestion => {
                    mem.last_turn().map(|t| t.raw_query.clone())
                }
                lu4d::session_memory::BackrefKind::LastAnswer => {
                    mem.last_turn().map(|t| t.response_text.clone())
                }
            })
            .filter(|v| !v.is_empty());
        match from_memory {
            Some(body) => {
                let text = format!("[backreference] {}\n", body.lines().next().unwrap_or(&body));
                return Ok(wrap_content(
                    text,
                    json!({
                        "session_id": session_id,
                        "backreference_kind": format!("{:?}", kind),
                        "answer": body,
                        "raw_query": raw_query,
                    }),
                ));
            }
            None => {
                return Ok(wrap_content(
                    "I don't keep a record of our conversation, so I can't tell you that.".to_string(),
                    json!({
                        "source": "meta_no_session_memory",
                        "backreference_kind": format!("{:?}", kind),
                        "raw_query": raw_query,
                    }),
                ));
            }
        }
    }

    let mut effective_query = raw_query.clone();
    let mut rewrite_note: Option<String> = None;
    // OP-ROUTE-COH2 (BUG-1 "tell me [more] about X"): a topical prose frame --
    // "tell me about X" / "tell me more about X" -- is a question ABOUT X, not a
    // request to elaborate the PREVIOUS turn. Normalize it to the proven,
    // subject-pinned definitional path ("what is X") -- exactly as the shipped
    // plain "tell me about X" rule did, now ALSO for the "...more about X" form
    // that the bare-"tell me more" elaboration rule used to steal and refuse.
    // Bare "tell me more" (no subject) and "tell me [more] about it/that" (a
    // pronoun) keep their own paths below; "yourself" stays MetaSelf.
    // `topical_claimed` then suppresses the elaboration/pronoun rewrites so they
    // cannot clobber the extracted subject.
    let mut topical_claimed = false;
    // Subject to retry biographically if the definitional frame refuses (a
    // proper-named person -- "Albert Einstein" -- answers "who is X" but refuses
    // "what is X"; see BUG-1 root-cause). None => no bio fallback.
    let mut bio_fallback_subject: Option<String> = None;

    // OP-ROUTE-GUEST-8 internal notes class D): conservative query-
    // understanding tolerance for arbitrary user phrasing. Runs before the
    // frame-normalization blocks so they and narrate see a canonical form.
    // Deliberately narrow: when a rewrite is ambiguous the query is left alone
    // and the honest gate handles it. Nothing here fabricates.
    {
        let base = raw_query
            .trim()
            .trim_end_matches(|c: char| c == '?' || c == '.' || c == '!')
            .trim();
        let bl = base.to_lowercase();
        // (a) Informal question-word forms the near_miss repairer skips (below
        //     its 5-char floor): "whos X" -> "who is X". No-apostrophe forms
        //     ONLY; "who's"/"what's" already resolve and are left untouched.
        const WH: &[(&str, &str)] = &[
            ("whos ", "who is "), ("wheres ", "where is "),
            ("whats ", "what is "), ("hows ", "how is "),
            ("whens ", "when is "),
        ];
        if !topical_claimed && effective_query == raw_query {
            if let Some((c, full)) = WH.iter().find(|(c, _)| bl.starts_with(c)) {
                let rest = base[c.len()..].trim();
                if !rest.is_empty() {
                    effective_query = format!("{}{}", full, rest);
                    rewrite_note = Some(format!("read as '{}'", effective_query));
                }
            }
        }
        // (b) Bare compound-subject comparison with no 'compare'/'between' cue:
        //     "X Y difference|differences" or "X vs Y" -> "compare X and Y".
        if !topical_claimed && effective_query == raw_query {
            let toks: Vec<&str> = bl.split_whitespace().collect();
            let pair: Option<(&str, &str)> = if toks.len() == 3
                && matches!(toks[2], "difference" | "differences" | "compared")
            {
                Some((toks[0], toks[1]))
            } else if toks.len() == 3 && matches!(toks[1], "vs" | "vs." | "versus") {
                Some((toks[0], toks[2]))
            } else {
                None
            };
            if let Some((a, b)) = pair {
                effective_query = format!("compare {} and {}", a, b);
                rewrite_note = Some(format!("read as '{}'", effective_query));
            }
        }
        // (c) Compound question "<clause A> and <wh-clause B>": answer each
        //     RESOLVABLE half instead of refusing the whole. Guarded so a
        //     single "compare X and Y" / "how do X and Y differ" never splits
        //     (only splits where the text after " and " begins a fresh
        //     wh-question). Honest: a half that refuses is simply dropped, and
        //     if neither half resolves we fall through to normal handling.
        {
            let src = effective_query.clone();
            let sl = src.to_lowercase();
            const WHHEAD: &[&str] = &[
                "who ", "whos ", "who's ", "what ", "whats ", "what's ",
                "where ", "wheres ", "where's ", "when ", "why ", "how ",
                "which ",
            ];
            let mut split_at: Option<usize> = None;
            let mut search = 0usize;
            while let Some(rel) = sl[search..].find(" and ") {
                let idx = search + rel;
                let after = &sl[idx + 5..];
                if WHHEAD.iter().any(|w| after.starts_with(w)) {
                    split_at = Some(idx);
                    break;
                }
                search = idx + 5;
            }
            if let Some(idx) = split_at {
                let left = src[..idx].trim().to_string();
                let right = src[idx + 5..].trim().to_string();
                let mut parts: Vec<String> = Vec::new();
                for clause in [left, right] {
                    if clause.is_empty() {
                        continue;
                    }
                    let r = brain.narrate(&clause, 3, 10, false);
                    if r.shape != lu4d::narration::ResponseShape::Refusal
                        && !r.answer.contains("don't have anything solid")
                    {
                        parts.push(r.answer.trim().to_string());
                    }
                }
                if !parts.is_empty() {
                    return Ok(wrap_content(
                        parts.join(" "),
                        json!({"source": "compound_split", "raw_query": raw_query}),
                    ));
                }
            }
        }
    }

    {
        let trimmed = raw_query.trim();
        let ql = trimmed.to_lowercase();
        // Longer prefix first so "tell me more about X" is not shadowed.
        let subj_off = if ql.starts_with("tell me more about ") {
            Some("tell me more about ".len())
        } else if ql.starts_with("tell me about ") {
            Some("tell me about ".len())
        } else {
            None
        };
        if let Some(off) = subj_off {
            let subject = trimmed[off..]
                .trim()
                .trim_end_matches(|c: char| c == '?' || c == '.' || c == '!')
                .trim();
            let is_self = ql.contains("yourself");
            let is_pronoun = lu4d::session_memory::contains_bare_pronoun(subject);
            if !subject.is_empty() && !is_self && !is_pronoun {
                // BUG-2 (definite follow-up): "the snake" / "the language" after a
                // turn that surfaced that sense should bind to THAT salient
                // referent, not a generic node. Resolve the definite head noun
                // against the prior turn's salient nouns; on a hit, pin it.
                let mut chosen = subject.to_string();
                if let Some(head) = lu4d::session_memory::definite_reference_head(subject) {
                    if let Some(sid) = session_id {
                        if let Some(mem) = brain.session_memories.get(sid) {
                            if let Some(referent) = mem.resolve_noun_reference(&head) {
                                rewrite_note = Some(format!(
                                    "definite reference '{}' -> {} (prior turn)",
                                    subject, referent
                                ));
                                chosen = referent;
                            }
                        }
                    }
                }
                // A capitalized subject token marks a proper name: eligible for
                // the biographical fallback if the definitional frame refuses.
                if subject
                    .split_whitespace()
                    .any(|w| w.chars().next().map_or(false, |c| c.is_uppercase()))
                {
                    bio_fallback_subject = Some(chosen.clone());
                }
                effective_query = format!("what is {}", chosen);
                topical_claimed = true;
            }
        }
    }
    // OP-ROUTE-GUEST-7 internal notes class C): an overview / summarize /
    // describe / full-picture frame over a PROPER-NAMED subject is a request
    // for the SAME enriched entity overview "who is X" produces. These verbs
    // previously never reached resolve_entity (generic idk) or fell into the
    // concept property-list path and emitted association salad ("A Charles is
    // black fingernail, Jones, and king"). Rewrite the frame straight to the
    // proven biographical path. Proper-named ONLY, so vision "describe the
    // image" and concept "describe photosynthesis" are left untouched, and the
    // honesty floor holds (an unresolvable name still refuses via "who is X").
    if !topical_claimed && effective_query == raw_query {
        const ENTITY_FRAMES: &[&str] = &[
            "give me an overview of ", "give me a full picture of ",
            "give me a brief overview of ", "give me a summary of ",
            "give me an overview on ", "give me the full picture of ",
            "an overview of ", "a full picture of ", "overview of ",
            "full picture of ", "summarize ", "summarise ", "describe ",
        ];
        let trimmed = raw_query.trim();
        let ql = trimmed.to_lowercase();
        if let Some(off) = ENTITY_FRAMES.iter().find_map(|p| ql.starts_with(p).then(|| p.len())) {
            let mut subject = trimmed[off..]
                .trim()
                .trim_end_matches(|c: char| c == '?' || c == '.' || c == '!')
                .trim()
                .to_string();
            // Strip a trailing biographical qualifier so the subject is the
            // bare name ("Marie Curie's life and work" -> "Marie Curie").
            const TAILS: &[&str] = &[
                "'s life and work", "'s life and works", "'s life and career",
                "'s contributions", "'s contribution", "'s achievements",
                "'s works", "'s work", "'s life", "'s biography", "'s career",
                " and his life and work", " and her life and work",
                " and his work", " and her work", " and their work",
                " and his works", " and her works", " and his life",
                " and her life", " and his contributions",
                " and her contributions", " and what he did", " and what she did",
            ];
            let sl = subject.to_lowercase();
            for t in TAILS {
                if sl.ends_with(t) {
                    subject.truncate(subject.len() - t.len());
                    subject = subject.trim().to_string();
                    break;
                }
            }
            let proper = !subject.is_empty()
                && subject
                    .split_whitespace()
                    .any(|w| w.chars().next().map_or(false, |c| c.is_uppercase()))
                && !lu4d::session_memory::contains_bare_pronoun(&subject)
                && !ql.contains("yourself");
            if proper {
                bio_fallback_subject = Some(subject.clone());
                effective_query = format!("who is {}", subject);
                topical_claimed = true;
            }
        }
    }
    // OP-ROUTE-GUEST-9 (occupation phrasing): "what was/is X's occupation |
    // profession | job" and "what did/does X do for a living" ask for the SAME
    // occupation the enriched "who is X" overview already emits ("<Name> was a
    // physicist ..."). The occupation fact is on the graph (Q4 surfaces Tesla as
    // "physicist", Q30 as "electrical engineering") but the occupation-question
    // phrasing never reached that arm and refused -- the same "flag detected but
    // never consumed" shape as the A (code_intent) and C (overview) fixes.
    // Rewrite a PROPER-NAMED occupation question straight to the proven
    // biographical path "who is X". Proper-named ONLY so a generic or uncovered
    // subject ("what is the occupation", "a teacher's job") still refuses
    // honestly -- the honesty floor holds.
    if !topical_claimed && effective_query == raw_query {
        let base = raw_query
            .trim()
            .trim_end_matches(|c: char| c == '?' || c == '.' || c == '!')
            .trim();
        let bl = base.to_lowercase();
        const OCC_LEADS: &[&str] =
            &["what was ", "what is ", "what were ", "what's "];
        const OCC_TAILS: &[&str] = &[
            "'s occupation", "'s profession", "'s job", "'s occupations",
        ];
        const LIVING_LEADS: &[&str] = &["what did ", "what does "];
        const LIVING_TAIL: &str = " do for a living";
        let mut occ_subject: Option<String> = None;
        if let Some(lead) = OCC_LEADS.iter().find(|p| bl.starts_with(**p)) {
            for tail in OCC_TAILS {
                if bl.ends_with(tail) {
                    let start = lead.len();
                    let end = base.len() - tail.len();
                    if end > start {
                        occ_subject = Some(base[start..end].trim().to_string());
                    }
                    break;
                }
            }
        }
        if occ_subject.is_none() {
            if let Some(lead) = LIVING_LEADS.iter().find(|p| bl.starts_with(**p)) {
                if bl.ends_with(LIVING_TAIL) {
                    let start = lead.len();
                    let end = base.len() - LIVING_TAIL.len();
                    if end > start {
                        occ_subject = Some(base[start..end].trim().to_string());
                    }
                }
            }
        }
        if let Some(subject) = occ_subject {
            let proper = !subject.is_empty()
                && subject
                    .split_whitespace()
                    .any(|w| w.chars().next().map_or(false, |c| c.is_uppercase()))
                && !lu4d::session_memory::contains_bare_pronoun(&subject)
                && !bl.contains("yourself");
            if proper {
                bio_fallback_subject = Some(subject.clone());
                effective_query = format!("who is {}", subject);
                rewrite_note = Some(format!("occupation question -> who is {}", subject));
                topical_claimed = true;
            }
        }
    }

    // OP-ROUTE-COH: topical 'what do you know about X' -> bare subject X.
    if !topical_claimed && effective_query == raw_query {
        if let Some(subject) = normalize_topical_meta(&raw_query) {
            effective_query = subject;
            topical_claimed = true;
        }
    }
    let mut needs_clarification: Option<String> = None;
    if !topical_claimed {
        if let Some(sid) = session_id {
            let (resolved_anchor, prior_effective): (Option<String>, Option<String>) =
                match brain.session_memories.get(sid) {
                    Some(mem) => (
                        mem.resolve_pronoun().map(|a| a.label.clone()),
                        mem.last_turn().map(|t| t.effective_query.clone()),
                    ),
                    None => (None, None),
                };
            if lu4d::session_memory::is_elaboration(&raw_query) {
                if let Some(prev) = prior_effective.clone() {
                    effective_query = format!("{} (elaborate)", prev);
                    rewrite_note = Some(format!("elaboration -> re-asking previous turn ({}) in greater depth", prev));
                } else {
                    needs_clarification =
                        Some("I have nothing to elaborate on yet -- could you ask a question first?".to_string());
                }
            } else if lu4d::session_memory::is_continuation(&raw_query) {
                // fresh noun phrase -- no rewrite.
            } else if lu4d::session_memory::contains_bare_pronoun(&raw_query) {
                if let Some(anchor) = resolved_anchor {
                    effective_query = lu4d::session_memory::substitute_pronoun(&raw_query, &anchor);
                    rewrite_note = Some(format!("pronoun -> {}", anchor));
                } else {
                    needs_clarification = Some(format!(
                        "Could you say what the pronoun in ({}) refers to? No recent anchor in this conversation is strong enough to substitute confidently.",
                        raw_query
                    ));
                }
            }
        }
    }

    // Clarification short-circuit -- recorded as a turn so later backreference
    // queries still see it.
    if let Some(msg) = needs_clarification {
        if let Some(sid) = session_id {
            let now = lu4d::brain::now_ms();
            let mem = brain.session_memories.get_or_create(sid, now);
            mem.record_turn(
                raw_query.clone(),
                raw_query.clone(),
                msg.clone(),
                Vec::new(),
                lu4d::session_memory::TurnTraceSummary::empty(),
                now,
            );
        }
        return Ok(wrap_content(
            msg.clone(),
            json!({"source": "clarification", "raw_query": raw_query}),
        ));
    }

    // From here the (possibly rewritten) effective query drives routing + narrate.
    let query: &str = effective_query.as_str();

    // OP-ROUTE-GUEST-5 -- agentic file/image routing. The query router already
    // DETECTS references_image_path / references_file_path (query_router.rs) but
    // nothing consumed them (the Phase-5 gap; #1353 finding 8). Now that
    // image_describe / file_read are on the guest surface, act on the flags:
    // extract the referenced path and route to the inspect-only reader. Falls
    // through to narrate() when no clean path token is present (safe).
    let intent = lu4d::narration::IntentClassifier::classify(query);
    let routing = lu4d::query_router::QueryRouter::route(query, intent);
    if routing.references_image_path {
        if let Some(path) = extract_path_token(query, true) {
            return guest_image_describe(brain, &json!({"path": path}));
        }
    }
    if routing.references_file_path {
        if let Some(path) = extract_path_token(query, false) {
            return guest_file_read(brain, &json!({"path": path}));
        }
    }

    let depth = args.get("depth").and_then(|v| v.as_u64()).unwrap_or(3) as usize;
    let top_k = args.get("top_k").and_then(|v| v.as_u64()).unwrap_or(10) as usize;

    // THE narrate/ask path. The idk-gate refusal floor lives inside this
    // call, so a low-confidence query returns an honest refusal rather
    // than a fabricated answer.
    // OP-ROUTE-GUEST-4 -- Left-Brain reachability. The guest historically
    // hardcoded use_multi_think=false, so the ReAct thinking loop only fired
    // when the router's needs_critic_vetting / needs_agentic_thinking matched
    // AND the goal seed cleared cfg.thinking_min_activation -- which a plain
    // "tell me about X" never trips internal notes Faculty 5). Opt non-
    // trivial conceptual queries into multi_think. multi_think is read/compute
    // (no graph mutation), so this stays guest-safe.
    let use_multi_think = guest_wants_multi_think(query);
    // OP-REASON-LOOP Phase 3 -- optional per-request reason<->retrieve depth.
    // Read-only knob (clamped 1..=8); absent -> brain default (1 = off), so
    // default guest behavior is unchanged. Lets a caller opt a single request
    // into the two-pass loop for eval/demo without a global config change.
    // OP-REASON-LOOP Phase 5 -- guest opt-in policy. A per-request
    // `max_context_iters` (clamped) still wins for eval; otherwise the
    // conservative policy opts multi-hop queries into a low iter cap and
    // leaves definitional / lookup / math / nonsense on the single-shot path.
    let req_iters = args.get("max_context_iters").and_then(|v| v.as_u64());
    let iters_override: Option<usize> = req_iters
        .map(|it| (it as usize).clamp(1, 8))
        .or_else(|| guest_reason_loop_iters(query));
    let mut nr = match iters_override {
        Some(it) if it > 1 => {
            let mut ncfg = brain.narration_config.clone();
            ncfg.max_context_iters = it;
            brain.narrate_with_config(query, depth, top_k, use_multi_think, &ncfg)
        }
        _ => brain.narrate(query, depth, top_k, use_multi_think),
    };

    // BUG-1 biographical fallback: a proper-named subject that refuses the
    // definitional frame ("what is Albert Einstein" -> Refusal) but answers the
    // biographical one ("who is Albert Einstein" -> a grounded Conclusion).
    // Fires ONLY when the definitional attempt actually refused, so a concept
    // the definitional path already answered is never re-routed; honesty holds
    // -- if the biographical retry also refuses (nonsense), keep the refusal.
    let primary_nonanswer = nr.shape == lu4d::narration::ResponseShape::Refusal
        || nr.answer.contains("don't have anything solid");
    if primary_nonanswer {
        if let Some(subj) = bio_fallback_subject.clone() {
            let bio_q = format!("who is {}", subj);
            let bio = brain.narrate(&bio_q, depth, top_k, use_multi_think);
            if bio.shape != lu4d::narration::ResponseShape::Refusal {
                effective_query = bio_q;
                nr = bio;
            }
        }
    }
    let mut text = nr.answer.clone();
    if let Some(summary) = &nr.source_summary {
        if !summary.is_empty() {
            text.push('\n');
            text.push_str(summary);
        }
    }
    if let Some(note) = &rewrite_note {
        text.push_str(&format!("\n(session rewrite: {})", note));
    }
    // OP-ROUTE-GUEST-6 -- Q.3 post-pass: record this turn so the NEXT call can
    // resolve "it" / "that" / "tell me more" against it. Writes to
    // session_memories (episodic, in-memory) -- never the graph.
    if let Some(sid) = session_id {
        let top_activations: Vec<String> = nr
            .ask
            .recall
            .activated
            .iter()
            .take(top_k.min(8))
            .map(|a| a.content.clone())
            .collect();
        let now = lu4d::brain::now_ms();
        let mem = brain.session_memories.get_or_create(sid, now);
        mem.record_turn(
            raw_query.clone(),
            effective_query.clone(),
            text.clone(),
            top_activations,
            lu4d::session_memory::TurnTraceSummary::empty(),
            now,
        );
        // OP-REASON-LOOP Phase 1.5 -- merge this prompt ContextSet into the
        // session-scoped hold, then evict to the cap at commit time (never
        // mid-answer). Guest-safe: session_context is in-RAM derived view data
        // on brain.session_memories, never nodes/bonds. Skipped when no
        // context_set was built (refusal / short-circuit paths -> None).
        if let Some(cs) = nr.context_set.as_ref() {
            let turn = mem.turn_history.len() as u64;
            mem.session_context.merge_prompt(cs, turn);
            mem.session_context.evict(turn);
        }
    }
    let structured = serde_json::to_value(&nr).map_err(|e| (-32603, e.to_string()))?;
    Ok(wrap_content(text, structured))
}

/// OP-MATH-HYPHEN-GUARD helper (guest mirror of src/mcp.rs): true when a
/// hyphenated word phrase the expression evaluator rejected ("unknown
/// identifier") actually names a node the recall path can answer. Existence
/// test only (query with definitional prefix stripped, as-is and
/// hyphens->spaces, raw and canonical-folded); no fuzzy match, so unknown
/// phrases fall through to the honest math error.
fn guest_math_hyphen_defers_to_recall(brain: &Brain, query: &str) -> bool {
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
        let raw = lu4d::seed::Seed::from_label(&cand);
        if brain.network.node_intern().get(&raw).is_some() {
            return true;
        }
        let folded =
            lu4d::seed::Seed::from_label(&lu4d::canonical::fold_seed_label(&cand));
        if brain.network.node_intern().get(&folded).is_some() {
            return true;
        }
    }
    false
}

/// Guest-side full-math cascade: F.4 construction -> F.3 word-problem -> F.2
/// expression/units/stats/solver. Mirrors the admin
/// `try_arithmetic_response_with` (src/mcp.rs:1543) minus the F.3 gap-signal
/// emit (a curiosity-producer concern needing `&Brain`; deferred to the
/// wonder-queue wiring). F.1 arithmetic is handled inline in `guest_ask`
/// ahead of this. Pure read/compute -- every detector is `&str -> Option`,
/// no graph mutation. internal notes.
fn try_guest_math(brain: &Brain, query: &str) -> Option<Value> {
    // F.4 construction-calculator (feet-inches, lumber/framing/concrete).
    if let Some(outcome) = lu4d::math::construction::detect_construction(query) {
        let prose_full = format!(
            "{}\n\nProvenance: computed by Lu's construction calculator, not recalled.",
            outcome.prose()
        );
        let outcome_json = serde_json::to_value(&outcome).unwrap_or(Value::Null);
        return Some(wrap_content(
            prose_full,
            json!({"source": "construction_primitive", "outcome": outcome_json, "query": query}),
        ));
    }

    // F.3 word-problem extractor (prose numbers + operation cues + question).
    if let Some(outcome) = lu4d::math::wordprob::detect_word_problem(query) {
        let prose_full = format!(
            "{}\n\nProvenance: routed through Lu's word-problem extractor.",
            outcome.prose()
        );
        let outcome_json = serde_json::to_value(&outcome).unwrap_or(Value::Null);
        let confidence_str = match &outcome {
            lu4d::math::wordprob::WordProblemOutcome::Computed {
                confidence: lu4d::math::wordprob::WordConfidence::Confident,
                ..
            } => "confident",
            _ => "hedged",
        };
        return Some(wrap_content(
            prose_full,
            json!({"source": "wordprob_primitive", "confidence": confidence_str, "outcome": outcome_json, "query": query}),
        ));
    }

    // F.2 unified expression path: expressions, unit conversions, descriptive
    // stats, linear solver.
    let outcome = lu4d::math::detect_math(query)?;
    // OP-MATH-HYPHEN-GUARD (VERIFY-FIRST 2026-08-03): a hyphenated concept
    // term ("half-life") parses as `a - b` and fails the expression evaluator
    // with "unknown identifier"; that is not arithmetic. When the phrase
    // resolves to a real node, defer to recall instead of emitting a spurious
    // math error. Genuine arithmetic ("5 - 3") is numeric and never yields an
    // unknown-identifier error, and unknown phrases keep the honest math error
    // (recall refuses them anyway), so the arithmetic path and the refusal
    // floor are both untouched. Mirrors the admin guard in src/mcp.rs.
    if let lu4d::math::MathOutcome::Error { route, error, .. } = &outcome {
        if route == "expression"
            && error.starts_with("unknown identifier")
            && guest_math_hyphen_defers_to_recall(brain, query)
        {
            return None;
        }
    }
    let prose_full = format!(
        "{}\n\nProvenance: computed by Lu's expression primitive, not recalled.",
        outcome.prose()
    );
    let outcome_json = serde_json::to_value(&outcome).unwrap_or(Value::Null);
    Some(wrap_content(
        prose_full,
        json!({"source": "expression_primitive", "outcome": outcome_json, "query": query}),
    ))
}
/// Guest `lu4d_image_describe` -- inspect-only vision. Delegates to the
/// admin-shared `tool_image_describe`, which runs the configured
/// SceneGraphExtractor and returns a human description + triplets and forms
/// NO bonds (proven by mcp_vision.rs vision_grep_guard_..._no_direct_mutation).
/// internal notes classifies image_describe as guest-safe (read/compute).
fn guest_image_describe(brain: &mut Brain, args: &Value) -> Result<Value, (i64, String)> {
    lu4d::mcp_vision::tool_image_describe(brain, args.clone())
}

/// Guest `lu4d_file_read` -- INSPECT-ONLY. Reads md/txt/csv/json/pdf/docx/code
/// via the ungated `lu4d::file_io::read_file` and returns a read-only
/// inspection. It NEVER routes content into teach/user_statement (those are
/// brain-mutating, admin-only per internal notes. The read path enforces
/// `cfg.allowed_read_paths`, bounding host-filesystem exposure.
fn guest_file_read(brain: &mut Brain, args: &Value) -> Result<Value, (i64, String)> {
    let path_str = args
        .get("path")
        .and_then(|v| v.as_str())
        .ok_or((-32602, "missing required 'path' string".to_string()))?;
    let path = std::path::PathBuf::from(path_str);
    let kind_hint = args
        .get("kind")
        .and_then(|v| v.as_str())
        .and_then(lu4d::file_io::FileKind::from_str_hint);
    let cfg = brain.file_io_config.clone();
    let read = lu4d::file_io::read_file(&path, kind_hint, &cfg).map_err(|e| (-32603, e))?;
    let mut structured = json!({
        "route": "Inspect",
        "file": path.to_string_lossy(),
        "file_kind": read.kind.as_str(),
        "byte_len": read.byte_len,
        "source_url": read.source_url_hint,
    });
    if let Some(t) = &read.text {
        structured["text"] = Value::String(t.clone());
    }
    if let Some(rows) = &read.csv_rows {
        structured["csv_rows"] = serde_json::to_value(rows).unwrap_or(Value::Null);
        structured["csv_row_count"] = json!(rows.len());
    }
    if let Some(j) = &read.json_value {
        structured["json_value"] = j.clone();
    }
    let head = match (&read.text, &read.csv_rows, &read.json_value) {
        (Some(t), _, _) => t.chars().take(160).collect::<String>(),
        (_, Some(rows), _) => format!("{} rows", rows.len()),
        (_, _, Some(_)) => "json".to_string(),
        _ => String::new(),
    };
    let text_msg = format!(
        "Read {} ({}, {} bytes): {:?}",
        path.display(),
        read.kind.as_str(),
        read.byte_len,
        head
    );
    Ok(wrap_content(text_msg, structured))
}

/// Decide whether the guest should engage the Left-Brain multi-think ReAct
/// loop (thinking.rs Observe->Plan->Act->Reflect) for this query. multi_think
/// is pure read/compute (no graph mutation), so opting in is guest-safe.
/// Targets ordinary conceptual questions ("tell me about X", "why does Y") that
/// otherwise answer from recall alone and feel shallow -- internal notes
/// Faculty 5 / #1356. Left as a heuristic knob to tune against the live brain.
/// OP-REASON-LOOP Phase 5 -- master switch for the guest reason-loop policy.
/// The policy machinery (guest_reason_loop_iters) + the per-request override
/// are complete and tested. Default-ON is DEFERRED: enabling the loop by
/// default for multi-hop queries surfaced a latency blocker on HUB-DENSE
/// topics (e.g. "why is the sky blue" ~22s at iters=2, over the 12s ceiling)
/// whose cost is the loop think()+seeded-spread on dense neighborhoods, NOT
/// the (already removed) classify reasoning pass -- a real fix (think-free
/// seed derivation, or an effective think-cost bound) revisits the Phase 3
/// loop and is a follow-up. Until then the loop stays OFF by default
/// (single-shot); the per-request `max_context_iters` override still
/// delivers the synthesis for eval. Flip to `true` once latency is bounded.
const GUEST_REASON_LOOP_ENABLED: bool = false;
/// Conservative default iteration cap when the policy opts a query in.
const GUEST_REASON_LOOP_ITERS: usize = 2;

/// OP-REASON-LOOP Phase 5 -- decide whether to run the reason<->retrieve
/// loop for a guest query, and at what iter cap. Conservative, by query
/// SHAPE (the only signal available before recall): opt in multi-hop /
/// relational / causal questions (where a single-pass recall hedges and
/// re-retrieval helps), leave definitional lookups and point queries on the
/// single-shot path. Math is short-circuited upstream and refusals never
/// reach the loop, so this only separates "conceptual multi-hop" from
/// "definitional / lookup". `None` == single-shot (iters=1).
/// Query-SHAPE eligibility for the reason loop -- flag-independent, so it is
/// unit-testable regardless of the master switch. Multi-hop / relational /
/// causal questions are eligible (a single-pass recall hedges); definitional
/// lookups and point queries are not.
fn reason_loop_query_eligible(query: &str) -> bool {
    let q = query.trim().to_lowercase();
    // Definitional / lookup: answered from the canonical gloss or a single
    // recall; the loop would add latency + tangential nodes without helping.
    const DEFINITIONAL: &[&str] = &[
        "what is ", "what are ", "what's ", "whats ", "define ",
        "definition of ", "what was ", "who is ", "who was ", "who are ",
    ];
    if DEFINITIONAL.iter().any(|prefix| q.starts_with(prefix)) {
        return false;
    }
    // Multi-hop / relational / causal: a single-pass recall tends to hedge;
    // re-retrieval from reasoning-derived seeds is where the loop earns it.
    const MULTIHOP: &[&str] = &[
        "how ", "how does", "how do", "why ", "why?", "explain ", "cause",
        "relationship", "difference between", "compare ", "connection between",
        "walk me through", "step by step", "what happens",
    ];
    MULTIHOP.iter().any(|cue| q.contains(cue))
}

/// Iteration cap for a guest query: the master switch AND shape eligibility.
/// `None` == single-shot (iters=1).
// OP-A0/Phase-5 demo flag: enable the reason<->retrieve loop for the free-form
// demo at RUNTIME without a rebuild. Default OFF. The compile-time
// GUEST_REASON_LOOP_ENABLED still forces it on if ever set true.
fn guest_reason_loop_enabled() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| GUEST_REASON_LOOP_ENABLED
        || matches!(std::env::var("LU_GUEST_REASON_LOOP").ok().as_deref(), Some("1") | Some("true") | Some("on")))
}
fn guest_reason_loop_iters_cap() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("LU_GUEST_REASON_LOOP_ITERS").ok()
        .and_then(|s| s.trim().parse().ok()).map(|n: usize| n.clamp(2, 8))
        .unwrap_or(GUEST_REASON_LOOP_ITERS))
}
fn guest_reason_loop_iters(query: &str) -> Option<usize> {
    if guest_reason_loop_enabled() && reason_loop_query_eligible(query) {
        Some(guest_reason_loop_iters_cap())
    } else {
        None
    }
}

fn guest_wants_multi_think(query: &str) -> bool {
    let q = query.trim().to_lowercase();
    const CUES: &[&str] = &[
        "tell me about", "tell me more", "explain", "why ", "why?", "how ",
        "how?", "describe", "compare", "what is", "what are", "what causes",
        "what happens", "walk me through", "reason", "step by step",
    ];
    if CUES.iter().any(|c| q.contains(c)) {
        return true;
    }
    // Otherwise engage for multi-concept queries (>= 4 tokens), leaving
    // 1-3 word point lookups on the fast recall-only path.
    q.split_whitespace().count() >= 4
}

/// Extract the first filesystem/image path token from a query, mirroring the
/// router token logic (trim + FILE_EXTS / IMAGE_EXTS). Used to ACT on the
/// router references_file_path / references_image_path flags. Returns None when
/// no token carries a recognized extension.
fn extract_path_token(query: &str, image: bool) -> Option<String> {
    const IMAGE_EXTS: &[&str] = &[".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff"];
    const FILE_EXTS: &[&str] = &[
        ".md", ".txt", ".csv", ".json", ".pdf", ".docx", ".doc", ".rs", ".py",
        ".js", ".ts", ".html", ".yaml", ".yml", ".toml", ".log", ".xml",
    ];
    let exts: &[&str] = if image { IMAGE_EXTS } else { FILE_EXTS };
    for tok in query.split_whitespace() {
        let t = tok.trim_end_matches(|c: char| matches!(c, '.' | ',' | ';' | ')' | '"' | '\''));
        let lt = t.to_lowercase();
        if exts.iter().any(|e| lt.ends_with(e)) {
            let pth = t.strip_prefix("file://").unwrap_or(t);
            return Some(pth.to_string());
        }
    }
    None
}

#[cfg(test)]
mod guest_math_tests {
    use super::try_guest_math;

    fn src(v: &serde_json::Value) -> String {
        v.get("structuredContent")
            .and_then(|s| s.get("source"))
            .and_then(|s| s.as_str())
            .unwrap_or("")
            .to_string()
    }

    // F.2 expression path: units, descriptive stats, linear solver -- all of
    // which were DARK on the guest before OP-ROUTE-GUEST-2 (arithmetic only).
    #[test]
    fn f2_units_stats_solver_route() {
        let r = try_guest_math("convert 100 km to miles").expect("units convert");
        assert_eq!(src(&r), "expression_primitive");
        let r = try_guest_math("mean of 1, 2, 3, 4, 5").expect("descriptive stats");
        assert_eq!(src(&r), "expression_primitive");
        let r = try_guest_math("solve x + 5 = 10").expect("linear solver");
        assert_eq!(src(&r), "expression_primitive");
    }

    // Negative control: non-math queries must fall through (return None) so
    // they reach the narrate/idk path, not a fabricated math answer.
    #[test]
    fn non_math_falls_through() {
        assert!(try_guest_math("what is mitochondria").is_none());
        assert!(try_guest_math("who is the president of the moon").is_none());
    }
}
#[cfg(test)]
mod guest_thinking_tests {
    use super::guest_wants_multi_think;
    use super::{guest_reason_loop_iters, reason_loop_query_eligible, GUEST_REASON_LOOP_ENABLED};

    #[test]
    fn conceptual_queries_engage_multi_think() {
        assert!(guest_wants_multi_think("tell me about dogs"));
        assert!(guest_wants_multi_think("tell me more"));
        assert!(guest_wants_multi_think("why is the sky blue"));
        assert!(guest_wants_multi_think("what is a dog"));
        assert!(guest_wants_multi_think("explain photosynthesis"));
        // >= 4 tokens even without an explicit cue.
        assert!(guest_wants_multi_think("the causes of the french revolution"));
    }

    #[test]
    fn point_lookups_stay_on_fast_path() {
        assert!(!guest_wants_multi_think("dog"));
        assert!(!guest_wants_multi_think("mitochondria"));
        assert!(!guest_wants_multi_think("capital france"));
    }

    #[test]
    fn reason_loop_policy_opts_in_multihop_not_definitional() {
        // Shape eligibility is flag-independent.
        assert!(reason_loop_query_eligible("how does the heart pump blood"));
        assert!(reason_loop_query_eligible("why is the sky blue"));
        assert!(reason_loop_query_eligible("explain photosynthesis"));
        assert!(!reason_loop_query_eligible("what is a dog"));
        assert!(!reason_loop_query_eligible("who is Napoleon"));
        assert!(!reason_loop_query_eligible("dog"));
        assert!(!reason_loop_query_eligible("mitochondria"));
        // The master switch gates the iter cap: eligible shape yields the cap
        // only when enabled, else single-shot.
        let it = guest_reason_loop_iters("how does the heart pump blood");
        if GUEST_REASON_LOOP_ENABLED {
            assert_eq!(it, Some(2));
        } else {
            assert_eq!(it, None);
        }
        // Definitional never opts in regardless of the switch.
        assert_eq!(guest_reason_loop_iters("what is a dog"), None);
    }
}
#[cfg(test)]
mod guest_routing_tests {
    use super::extract_path_token;

    #[test]
    fn extracts_file_and_image_paths() {
        assert_eq!(
            extract_path_token("summarize ./notes.md please", false).as_deref(),
            Some("./notes.md")
        );
        assert_eq!(
            extract_path_token("open ~/docs/report.pdf.", false).as_deref(),
            Some("~/docs/report.pdf")
        );
        assert_eq!(
            extract_path_token("read file:///tmp/a.txt", false).as_deref(),
            Some("/tmp/a.txt")
        );
        assert_eq!(
            extract_path_token("describe photo.png", true).as_deref(),
            Some("photo.png")
        );
    }

    #[test]
    fn no_path_returns_none() {
        assert!(extract_path_token("what is a dog", false).is_none());
        assert!(extract_path_token("tell me about cats", true).is_none());
        // a .png is not a FILE ext match, and a .md is not an IMAGE ext match
        assert!(extract_path_token("look at photo.png", false).is_none());
    }
}
#[cfg(test)]
mod guest_session_tests {
    use super::guest_ask;
    use lu4d::Brain;
    use serde_json::json;

    // Backreference wiring: with a seeded prior turn, "what was my first
    // question?" is answered from turn history (no recall). Deterministic --
    // does not depend on narrate output.
    #[test]
    fn backreference_reads_first_question_from_history() {
        let mut b = Brain::new();
        let sid = b.session_start();
        let now = lu4d::brain::now_ms();
        let mem = b.session_memories.get_or_create(sid, now);
        mem.record_turn(
            "what is a dog".into(),
            "what is a dog".into(),
            "a dog is a domesticated animal".into(),
            vec![],
            lu4d::session_memory::TurnTraceSummary::empty(),
            now,
        );
        let r = guest_ask(
            &mut b,
            &json!({"query": "what was my first question?", "session_id": sid}),
        )
        .unwrap();
        let sc = r.get("structuredContent").unwrap();
        assert_eq!(
            sc.get("backreference_kind").and_then(|v| v.as_str()),
            Some("FirstQuestion")
        );
        assert!(sc
            .get("answer")
            .and_then(|v| v.as_str())
            .unwrap()
            .contains("what is a dog"));
    }

    // Elaboration with no prior turn must ask for a question, not fabricate.
    #[test]
    fn elaboration_without_prior_requests_clarification() {
        let mut b = Brain::new();
        let sid = b.session_start();
        let r = guest_ask(&mut b, &json!({"query": "tell me more", "session_id": sid})).unwrap();
        let sc = r.get("structuredContent").unwrap();
        assert_eq!(sc.get("source").and_then(|v| v.as_str()), Some("clarification"));
    }

    // Full round-trip through narrate + the Q.3 post-pass: a content question
    // records a turn, and a follow-up backreference then reads it back.
    #[test]
    fn round_trip_records_turn_then_backreferences_it() {
        let mut b = Brain::new();
        let sid = b.session_start();
        let _ = guest_ask(&mut b, &json!({"query": "dog", "session_id": sid})).unwrap();
        let r = guest_ask(
            &mut b,
            &json!({"query": "what did I just ask?", "session_id": sid}),
        )
        .unwrap();
        let sc = r.get("structuredContent").unwrap();
        assert_eq!(
            sc.get("backreference_kind").and_then(|v| v.as_str()),
            Some("LastQuestion")
        );
        assert!(sc
            .get("answer")
            .and_then(|v| v.as_str())
            .unwrap()
            .contains("dog"));
    }

    // OP-ROUTE-GUEST-6 personal-pronoun fix: with a subject anchor seeded
    // from a prior turn, a "he" follow-up is rewritten to that subject before
    // narrate. On an empty brain narrate still refuses (honesty floor), but
    // the rewrite note proves the referent resolved -- the wire that was cut.
    #[test]
    fn personal_pronoun_followup_rewrites_to_subject_anchor() {
        let mut b = Brain::new();
        let sid = b.session_start();
        let now = lu4d::brain::now_ms();
        let mem = b.session_memories.get_or_create(sid, now);
        mem.record_turn(
            "who was Einstein".into(),
            "who was Einstein".into(),
            "Albert Einstein is a physicist. Einstein was born in Ulm.".into(),
            vec!["Einstein".into(), "Albert Einstein".into()],
            lu4d::session_memory::TurnTraceSummary::empty(),
            now,
        );
        let r = guest_ask(
            &mut b,
            &json!({"query": "where was he born", "session_id": sid}),
        )
        .unwrap();
        let text = r
            .get("content")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("text"))
            .and_then(|v| v.as_str())
            .unwrap_or("");
        assert!(
            text.contains("pronoun -> Einstein"),
            "expected pronoun rewrite note, got: {text}"
        );
    }

    // Honesty floor: a "he" follow-up with NO anchor must ask for
    // clarification, never fabricate a subject.
    #[test]
    fn personal_pronoun_without_anchor_requests_clarification() {
        let mut b = Brain::new();
        let sid = b.session_start();
        let r = guest_ask(
            &mut b,
            &json!({"query": "where was he born", "session_id": sid}),
        )
        .unwrap();
        let sc = r.get("structuredContent").unwrap();
        assert_eq!(sc.get("source").and_then(|v| v.as_str()), Some("clarification"));
    }
}
/// OWNER-ONLY code-writing tool (OP-CODE-EXEC Phase 6). Plans natively over the
/// brain (op-grounding, pattern retrieval, data-plumbing composition), then
/// delegates EXECUTION to the isolated code-exec broker (X1 sandbox) over a
/// localhost socket. The hardened guest never holds the X1 privilege. Invite
/// (guest) principals are refused: code execution is owner-gated.
fn guest_code(brain: &Brain, principal: &Principal, args: &Value) -> Result<Value, (i64, String)> {
    if !matches!(principal, Principal::Owner) {
        return Err((-32003, "code execution is owner-only".to_string()));
    }
    let prompt = args
        .get("prompt")
        .or_else(|| args.get("query"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim();
    if prompt.is_empty() {
        return Err((-32602, "missing 'prompt'".to_string()));
    }
    let plan = lu4d::code_exec::planner::plan_only(brain, prompt);
    let code = match plan.code.clone() {
        None => {
            let msg = plan
                .message
                .clone()
                .unwrap_or_else(|| "I can't ground this request, so I won't fake a solution.".to_string());
            return Ok(wrap_content(
                msg,
                json!({"verdict": "HONEST_ABORT", "path": plan.path, "ok": false, "code": Value::Null, "stdout": Value::Null}),
            ));
        }
        Some(c) => c,
    };
    // Optional user-provided input for a compose plan that reads a file.
    let mut inputs: Vec<Value> = Vec::new();
    if let (Some(name), Some(text)) = (
        plan.input_name.clone(),
        args.get("input").and_then(|v| v.as_str()),
    ) {
        if !text.is_empty() {
            inputs.push(json!({"name": name, "text": text}));
        }
    }
    let req = json!({"code": code, "inputs": inputs});
    let resp = match broker_call(&req) {
        Ok(r) => r,
        Err(e) => return Err((-32004, format!("code-exec broker unavailable: {}", e))),
    };
    let ok = resp.get("ok").and_then(|v| v.as_bool()).unwrap_or(false);
    let final_code = resp.get("code").and_then(|v| v.as_str()).unwrap_or(&code).to_string();
    let stdout = resp.get("stdout").and_then(|v| v.as_str()).unwrap_or("").to_string();
    let stderr = resp.get("stderr").and_then(|v| v.as_str()).unwrap_or("").to_string();
    let verdict = match (plan.kind, ok) {
        ("pattern", true) => "PATTERN_PASS",
        ("pattern", false) => "PATTERN_FAIL",
        (_, true) => "COMPOSE_PASS",
        (_, false) => "COMPOSE_FAIL",
    };
    let mut text = String::new();
    if ok {
        text.push_str(&format!("Here's a working program ({}):\n\n```python\n{}\n```\n", plan.path, final_code));
        if !stdout.is_empty() {
            text.push_str(&format!("\nOutput:\n{}", stdout.trim_end()));
        }
        if plan.input_name.is_some() && inputs.is_empty() {
            text.push_str("\n\n(Provide sample file content in the input box to run it on your data.)");
        }
    } else {
        text.push_str(&format!(
            "I composed a program but it did not run cleanly, so I won't claim it works.\n\n```python\n{}\n```\n",
            final_code
        ));
        if !stderr.is_empty() {
            text.push_str(&format!("\nError:\n{}", stderr.trim_end()));
        }
    }
    let download_name = if plan.kind == "pattern" {
        format!("{}.py", plan.path.replace("pattern:", ""))
    } else {
        "program.py".to_string()
    };
    Ok(wrap_content(
        text,
        json!({
            "verdict": verdict,
            "path": plan.path,
            "ok": ok,
            "code": final_code,
            "stdout": stdout,
            "stderr": stderr,
            "grounded_ops": plan.grounded_ops,
            "download_name": download_name,
            "input_name": plan.input_name,
        }),
    ))
}

/// Connect to the local code-exec broker socket, send one JSON request, read the
/// JSON reply. Localhost Unix socket only; no network.
fn broker_call(req: &Value) -> std::io::Result<Value> {
    use std::io::{Read, Write};
    use std::os::unix::net::UnixStream;
    let sock = std::env::var("LU_CODEEXEC_SOCK").unwrap_or_else(|_| "/run/lu-codeexec/sock".to_string());
    let mut s = UnixStream::connect(&sock)?;
    s.write_all(req.to_string().as_bytes())?;
    s.shutdown(std::net::Shutdown::Write)?;
    let mut buf = String::new();
    s.read_to_string(&mut buf)?;
    serde_json::from_str(&buf).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}

#[cfg(test)]
mod op_route_coh_tests {
    use super::normalize_topical_meta;

    #[test]
    fn topical_meta_rewrites_to_bare_subject() {
        assert_eq!(normalize_topical_meta("What do you know about python").as_deref(), Some("python"));
        assert_eq!(normalize_topical_meta("what do you know about dogs").as_deref(), Some("dogs"));
        assert_eq!(normalize_topical_meta("what have you learned about gravity?").as_deref(), Some("gravity"));
        assert_eq!(normalize_topical_meta("what do you remember about the mitochondria").as_deref(), Some("mitochondria"));
    }

    #[test]
    fn bare_meta_is_left_for_census() {
        assert_eq!(normalize_topical_meta("what do you know"), None);
        assert_eq!(normalize_topical_meta("what do you know about "), None);
        assert_eq!(normalize_topical_meta("what do you know about the"), None);
    }

    #[test]
    fn non_meta_frames_untouched() {
        assert_eq!(normalize_topical_meta("what is a hash table"), None);
        assert_eq!(normalize_topical_meta("tell me about python"), None);
        assert_eq!(normalize_topical_meta("python"), None);
    }
}
