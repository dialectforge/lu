//! OP-ROUTE-2 — QueryRouter.
//!
//! Hand-coded routing layer that decides which capability ports a
//! given query should activate. Sits between `IntentClassifier` and the
//! recall pipeline in `narrate()`. Anti-LLM positioning is load-bearing:
//! every dimension below is a deterministic string / length heuristic.
//! No learning, no embeddings, no model calls.
//!
//! Phase 2 (this port) is observation-only — `narrate()` populates a
//! `RoutingDecision` on the result but does NOT alter pipeline shape
//! based on it. Phase 3+ ports will wire `think()`, `variants()`,
//! file_io, vision, and code-gen invocations off these flags.
//!
//! The router is a pure function: no `&Brain`, no struct state, no
//! global cache. Inputs are borrowed; outputs are owned. Thread-safe
//! by construction.

use crate::narration::QueryIntent;
use serde::{Deserialize, Serialize};

/// Routing decision: which capability ports a query should activate.
///
/// All boolean dimensions default to `false`. A `RoutingDecision::default()`
/// represents "no special routing" and produces narrate() output
/// byte-identical to pre-OP-ROUTE-2 behavior (the Phase 2 parity
/// invariant).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RoutingDecision {
    /// Query is arithmetic / construction / math-expression.
    /// Reuses `crate::math` detectors. Informational only in Phase 2 —
    /// the math short-circuit still lives in `try_arithmetic_response_with`
    /// at the MCP boundary (OP-ROUTE-1).
    pub needs_math: bool,

    /// Query references a filesystem path (e.g. "summarize ./notes.md",
    /// "open ~/Documents/x.pdf"). Phase 5 will wire file_io here.
    pub references_file_path: bool,

    /// Query references an image path or asks for visual interpretation.
    /// Phase 5 will wire vision here.
    pub references_image_path: bool,

    /// Long / multi-clause / causal-marker query that warrants `think()`.
    /// Phase 3a will wire `crate::thinking` here.
    pub needs_deep_reasoning: bool,

    /// Counterfactual / alternative-framing query.
    /// Phase 3b will wire `crate::variants` here.
    pub needs_variants: bool,

    /// Controversial claim / "is it true that" query.
    /// Phase 3c will wire multi-agent critic vetting here.
    pub needs_critic_vetting: bool,

    /// OP-ROUTE-3c -- procedural / exploratory query that explicitly
    /// asks Lu to work through a problem (e.g. "how do i", "step by
    /// step", "walk me through", "figure out", "think through").
    /// When set AND `is_meta_query` / `needs_math` are clear AND
    /// a goal seed clears `NarrationConfig::thinking_min_activation`,
    /// `narrate()` spawns a `Brain::think` ReAct loop on the PASS path
    /// and threads a `ThinkingTraceSummary` through shape selection
    /// and slot fill. May co-fire with `needs_deep_reasoning` -- the
    /// procedural marker list overlaps with the deep-reasoning marker
    /// list intentionally (Section 2 of op_route_3c_design.md). The
    /// two flags address different needs: reasoning articulates an
    /// existing chain, while thinking grows the graph in service of
    /// the query.
    pub needs_agentic_thinking: bool,

    /// Code-generation intent (write a function, generate code).
    /// Phase 6 will wire code generation here.
    pub code_intent: bool,

    /// Meta-query about Lu's own state. Phase 3 may suppress deep-port
    /// invocation here (you don't `think()` about your own brain).
    /// Derived from `QueryIntent::MetaSelf` / `TimetravelComparison`.
    pub is_meta_query: bool,

    /// Stable list of heuristic names that fired (mirrors the
    /// `signals_used` pattern from narration shape selection). Useful
    /// for debugging "why this routing?".
    pub triggers: Vec<String>,
}

/// Pure-function namespace. No state, no `&self`.
pub struct QueryRouter;

impl QueryRouter {
    /// Single entry point. Pure function over `query + intent`.
    /// Does NOT touch the brain — every routing dimension is decidable
    /// from the query string and the already-computed intent alone.
    pub fn route(query: &str, intent: QueryIntent) -> RoutingDecision {
        let mut d = RoutingDecision::default();

        // 3.1 needs_math — reuse crate::math detectors.
        let math_arith = crate::math::detect_arithmetic(query).is_some();
        let math_constr = crate::math::construction::detect_construction(query).is_some();
        let math_general = crate::math::detect_math(query).is_some();
        if math_arith || math_constr || math_general {
            d.needs_math = true;
            d.triggers.push("math".into());
        }

        // 3.3 references_image_path — checked BEFORE file_path so an
        // image extension wins over a generic file-action match.
        if let Some(reason) = looks_like_image(query) {
            d.references_image_path = true;
            d.triggers.push(format!("image:{}", reason));
        }

        // 3.2 references_file_path — disjoint from image; if image
        // already fired, skip file-path so they stay mutually exclusive.
        if !d.references_image_path {
            if let Some(reason) = looks_like_file_path(query) {
                d.references_file_path = true;
                d.triggers.push(format!("file:{}", reason));
            }
        }

        // 3.4 needs_deep_reasoning — structural complexity OR causal-intent.
        if needs_deep_reasoning(query) {
            d.needs_deep_reasoning = true;
            d.triggers.push("deep_reasoning".into());
        }
        // OP-ROUTE-2.1: WhyChain intent always fires deep reasoning.
        // A short "Why does X?" query is intrinsically asking for an
        // explanatory chain even when it lacks structural complexity
        // markers. Triggering deep_reasoning lets the PASS path build
        // a chain per OP-ROUTE-3a. Distinct trigger string preserves
        // telemetry on which rule fired.
        if intent == QueryIntent::WhyChain && !d.needs_deep_reasoning {
            d.needs_deep_reasoning = true;
            d.triggers.push("deep_reasoning:whychain_intent".into());
        }

        // 3.5 needs_variants — counterfactual / alternative framing.
        if needs_variants(query) {
            d.needs_variants = true;
            d.triggers.push("variants".into());
        }

        // 3.6 needs_critic_vetting — claim-to-verify markers.
        if needs_critic_vetting(query) {
            d.needs_critic_vetting = true;
            d.triggers.push("critic_vetting".into());
        }

        // 3.6c OP-ROUTE-3c needs_agentic_thinking — procedural /
        // exploratory markers. Intentionally overlaps with the
        // needs_deep_reasoning marker set on "step by step" /
        // "walk me through" / "reason through" by design
        // (Section 2 of op_route_3c_design.md). The shape selector
        // resolves the dual-fire at rule 9.3 / 9.5 precedence.
        if needs_agentic_thinking(query) {
            d.needs_agentic_thinking = true;
            d.triggers.push("agentic_thinking".into());
        }

        // 3.7 code_intent — code-generation request.
        if code_intent(query) {
            d.code_intent = true;
            d.triggers.push("code_intent".into());
        }

        // 3.8 is_meta_query — derived from intent.
        if intent == QueryIntent::MetaSelf || intent == QueryIntent::TimetravelComparison {
            d.is_meta_query = true;
            d.triggers.push("meta_query".into());
        }

        d
    }
}

// ---------------------------------------------------------------------
// Per-dimension helpers
// ---------------------------------------------------------------------

const IMAGE_EXTS: &[&str] = &[
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff",
];

const FILE_EXTS: &[&str] = &[
    ".md", ".txt", ".csv", ".json", ".pdf", ".docx", ".doc",
    ".rs", ".py", ".js", ".ts", ".html", ".yaml", ".yml",
    ".toml", ".log", ".xml",
];

const VISION_PHRASES: &[&str] = &[
    "describe the image",
    "describe this image",
    "look at this image",
    "look at the image",
    "what is in the picture",
    "what's in the picture",
    "what do you see",
    "in this photo",
    "in the photo",
];

const FILE_ACTIONS: &[&str] = &[
    "read ", "open ", "summarize ", "parse ", "load ", "analyze ",
    "cat ", "the file", "this file",
];

fn token_has_path_prefix(t: &str) -> bool {
    if t.starts_with('/') || t.starts_with("./") || t.starts_with("../") || t.starts_with("~/") {
        return true;
    }
    let bytes = t.as_bytes();
    if bytes.len() >= 3 && bytes[1] == b':' && (bytes[2] == b'\\' || bytes[2] == b'/') {
        let c = bytes[0];
        if (c >= b'A' && c <= b'Z') || (c >= b'a' && c <= b'z') {
            return true;
        }
    }
    false
}

fn trim_token(t: &str) -> &str {
    t.trim_end_matches(|c: char| matches!(c, '.' | ',' | ';' | ')' | '"' | '\''))
}

fn looks_like_image(query: &str) -> Option<&'static str> {
    let lower = query.to_lowercase();
    for ph in VISION_PHRASES {
        if lower.contains(ph) {
            return Some("vision_phrase");
        }
    }
    for tok in query.split_whitespace() {
        let t = trim_token(tok);
        let lt = t.to_lowercase();
        if IMAGE_EXTS.iter().any(|e| lt.ends_with(e)) {
            return Some("image_ext");
        }
    }
    None
}

fn looks_like_file_path(query: &str) -> Option<&'static str> {
    let lower = query.to_lowercase();
    if lower.contains("file://") {
        return Some("uri_scheme");
    }
    let has_action = FILE_ACTIONS.iter().any(|v| lower.contains(v));
    for tok in query.split_whitespace() {
        let t = trim_token(tok);
        let lt = t.to_lowercase();
        let has_path_prefix = token_has_path_prefix(t);
        let has_ext = FILE_EXTS.iter().any(|e| lt.ends_with(e));
        if has_ext && (has_path_prefix || has_action) {
            return Some(if has_path_prefix { "path_with_ext" } else { "ext_with_action" });
        }
    }
    None
}

fn needs_deep_reasoning(query: &str) -> bool {
    let lower = query.to_lowercase();
    let word_count = query.split_whitespace().count();
    if word_count >= 25 {
        return true;
    }

    // Causal-with-because pattern.
    if lower.contains("why") && lower.contains("because") {
        return true;
    }
    if lower.contains("because") && lower.contains('?') {
        return true;
    }

    // Multi-axis comparison markers.
    let multi_axis = [
        "considering", "taking into account", "weigh ", "weighing",
        "evaluate", "trade-off", "trade off", "tradeoff",
    ];
    if multi_axis.iter().any(|m| lower.contains(m)) {
        return true;
    }

    // Explicit-explain markers.
    let explain = [
        "explain in detail", "walk me through", "step by step",
        "step-by-step", "in depth", "reason about", "reason through",
    ];
    if explain.iter().any(|m| lower.contains(m)) {
        return true;
    }

    // Sentence count >= 3.
    let sentence_count = query.chars().filter(|c| matches!(*c, '?' | '.' | '!')).count();
    if sentence_count >= 3 {
        return true;
    }

    false
}

fn needs_variants(query: &str) -> bool {
    let lower = query.to_lowercase();
    let markers = [
        "what if ", "what would happen if", "suppose ",
        "imagine if", "imagine that",
        "alternatively", "from a different angle", "from another angle",
        "a different perspective", "another way to look at",
        "counterfactual", "hypothetically",
        "would it be different if", "could it have been",
    ];
    markers.iter().any(|m| lower.contains(m))
}

fn needs_critic_vetting(query: &str) -> bool {
    let lower = query.to_lowercase();
    let markers = [
        "is it true that", "is it true", "true or false",
        "are you sure", "can you verify", "can you confirm",
        "double-check", "double check", "fact-check", "fact check",
        "i heard that", "someone told me", "is the claim",
        "the claim that", "prove it", "how do you know",
        "evidence for", "evidence that",
    ];
    markers.iter().any(|m| lower.contains(m))
}

fn needs_agentic_thinking(query: &str) -> bool {
    let lower = query.to_lowercase();
    // Marker list per Section 17 decision #2 of op_route_3c_design.md.
    // Tighten in this helper if "how do i" proves too wide a net.
    let markers = [
        "how do i ", "how do you ", "how can i ", "how would i ",
        "step by step", "step-by-step",
        "walk me through", "work through",
        "think about", "think through",
        "figure out", "work out",
        "plan how to", "plan for",
        "break this down", "break it down",
        "think it over", "reason through",
    ];
    markers.iter().any(|m| lower.contains(m))
}

fn code_intent(query: &str) -> bool {
    let lower = query.to_lowercase();

    // Fenced code block marker.
    if lower.contains("```") {
        return true;
    }

    // Verb + noun anywhere in query.
    let verbs = [
        "write ", "generate ", "produce ", "create ", "give me ",
        "show me ", "implement ",
    ];
    let nouns = [
        "a function", "a class", "a script", "a snippet",
        " code", "the code", "some code", "a regex",
        "an sql", " a query", "a query for",
    ];
    let has_verb = verbs.iter().any(|v| lower.contains(v));
    let has_noun = nouns.iter().any(|n| lower.contains(n));
    if has_verb && has_noun {
        return true;
    }

    // Language + verb proximity (<=4 tokens apart).
    let langs = [
        "python", "rust", "javascript", "typescript", "go",
        "java", "c++", "c#", "bash", "sql", "html", "css",
    ];
    let pair_verbs = ["write", "generate", "in", "using"];
    let toks: Vec<String> = lower
        .split_whitespace()
        .map(|t| trim_token(t).to_string())
        .collect();
    let lang_positions: Vec<usize> = toks
        .iter()
        .enumerate()
        .filter(|(_, t)| langs.iter().any(|l| t == l))
        .map(|(i, _)| i)
        .collect();
    if !lang_positions.is_empty() {
        // Require at least one CODE-GENERATION verb (write/generate)
        // somewhere within 4 tokens — "in"/"using" alone are too
        // weak (e.g. "is python a language" would otherwise fire).
        let strong_verbs = ["write", "generate", "produce", "create", "implement"];
        let verb_positions: Vec<usize> = toks
            .iter()
            .enumerate()
            .filter(|(_, t)| strong_verbs.iter().any(|v| t == v))
            .map(|(i, _)| i)
            .collect();
        for &lp in &lang_positions {
            for &vp in &verb_positions {
                let dist = if lp > vp { lp - vp } else { vp - lp };
                if dist <= 6 {
                    // Also require a code-noun nearby OR "in"/"using"
                    // bridging, to filter "write a poem about python".
                    let bridge = pair_verbs.iter().any(|pv| toks.iter().any(|t| t == pv));
                    if has_noun || bridge {
                        return true;
                    }
                }
            }
        }
    }

    false
}

// ---------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn route(q: &str) -> RoutingDecision {
        QueryRouter::route(q, QueryIntent::Generic)
    }

    // ---- needs_math ----

    #[test]
    fn routes_math_arithmetic_positive() {
        let d = route("what is 2+2");
        assert!(d.needs_math);
        assert!(d.triggers.iter().any(|t| t == "math"));
    }

    #[test]
    fn routes_math_word_form_positive() {
        let d = route("what is 7 plus 5?");
        assert!(d.needs_math);
    }

    #[test]
    fn routes_math_definition_negative() {
        let d = route("what is a tree");
        assert!(!d.needs_math);
    }

    #[test]
    fn routes_math_greeting_negative() {
        let d = route("how are you");
        assert!(!d.needs_math);
    }

    #[test]
    fn opmd_routes_step_by_step_what_is_times_to_math() {
        // OP-MATH-DETECT: original failure prompt from OP-ROUTE-3c
        // canonical sanity. CoT-framed arithmetic must short-circuit
        // to math, not the 33-second thinking path.
        let d = route("step by step, what is 47 times 13");
        assert!(d.needs_math, "original failure prompt must route to math");
        assert!(d.triggers.iter().any(|t| t == "math"));
    }

    // ---- references_file_path ----

    #[test]
    fn routes_file_path_relative_positive() {
        let d = route("summarize ./notes.md");
        assert!(d.references_file_path);
        assert!(d.triggers.iter().any(|t| t.starts_with("file:")));
    }

    #[test]
    fn routes_file_path_home_positive() {
        let d = route("open ~/docs/foo.pdf");
        assert!(d.references_file_path);
    }

    #[test]
    fn routes_file_path_situation_negative() {
        let d = route("summarize the situation");
        assert!(!d.references_file_path);
    }

    #[test]
    fn routes_file_path_concept_negative() {
        let d = route("what is markdown");
        assert!(!d.references_file_path);
    }

    // ---- references_image_path ----

    #[test]
    fn routes_image_path_positive() {
        let d = route("describe the image at /tmp/cat.png");
        assert!(d.references_image_path);
        assert!(!d.references_file_path, "image must be disjoint from file_path");
    }

    #[test]
    fn routes_image_vision_phrase_positive() {
        let d = route("what is in the picture above");
        assert!(d.references_image_path);
    }

    #[test]
    fn routes_image_animal_negative() {
        let d = route("describe a cat");
        assert!(!d.references_image_path);
    }

    #[test]
    fn routes_image_format_term_negative() {
        let d = route("png is a format");
        assert!(!d.references_image_path);
    }

    // ---- needs_deep_reasoning ----

    #[test]
    fn routes_deep_reasoning_long_query_positive() {
        // Word count >= 25 trigger.
        let q = "explain to me in great detail every single one of the many factors that \
                 contribute to ocean acidification including the chemical reactions involved \
                 and the downstream consequences for marine ecosystems and food webs";
        assert!(q.split_whitespace().count() >= 25);
        let d = route(q);
        assert!(d.needs_deep_reasoning);
    }

    #[test]
    fn routes_deep_reasoning_why_because_positive() {
        let d = route("why is X happening because of Y reasonable, considering Z?");
        assert!(d.needs_deep_reasoning);
    }

    #[test]
    fn routes_deep_reasoning_short_why_negative() {
        let d = route("why is the sky blue?");
        assert!(!d.needs_deep_reasoning);
    }

    #[test]
    fn routes_deep_reasoning_short_what_negative() {
        let d = route("what is rust");
        assert!(!d.needs_deep_reasoning);
    }

    // OP-ROUTE-2.1: WhyChain intent always fires deep reasoning,
    // even on short queries that lack structural complexity markers.
    #[test]
    fn routes_deep_reasoning_whychain_intent_positive() {
        let d = QueryRouter::route("why is the sky blue?", QueryIntent::WhyChain);
        assert!(d.needs_deep_reasoning);
        assert!(d.triggers.iter().any(|t| t == "deep_reasoning:whychain_intent"));
    }

    #[test]
    fn routes_deep_reasoning_whychain_intent_long_query_uses_structural_trigger() {
        // A long why query hits the structural rule first; the
        // WhyChain intent rule is gated on "not already set" so the
        // trigger string carries the structural marker, not the
        // whychain marker. This keeps telemetry clean.
        let q = "why does X happen because of Y reasonable, considering Z and all the                  downstream factors that influence the system at scale";
        let d = QueryRouter::route(q, QueryIntent::WhyChain);
        assert!(d.needs_deep_reasoning);
        assert!(d.triggers.iter().any(|t| t == "deep_reasoning"));
        // The whychain-intent trigger should NOT also be present.
        assert!(!d.triggers.iter().any(|t| t == "deep_reasoning:whychain_intent"));
    }

    #[test]
    fn routes_deep_reasoning_generic_intent_short_why_negative() {
        // A short why query under Generic intent does NOT fire deep
        // reasoning. The WhyChain rule is intent-gated; Generic
        // queries continue to require structural complexity.
        let d = QueryRouter::route("why is the sky blue?", QueryIntent::Generic);
        assert!(!d.needs_deep_reasoning);
    }

    // ---- needs_variants ----

    #[test]
    fn routes_variants_what_if_positive() {
        let d = route("what if we used Rust instead");
        assert!(d.needs_variants);
    }

    #[test]
    fn routes_variants_different_angle_positive() {
        let d = route("from a different angle, is this idea sound?");
        assert!(d.needs_variants);
    }

    #[test]
    fn routes_variants_rust_negative() {
        let d = route("what is rust");
        assert!(!d.needs_variants);
    }

    #[test]
    fn routes_variants_plain_question_negative() {
        let d = route("is this idea sound?");
        assert!(!d.needs_variants);
    }

    // ---- needs_critic_vetting ----

    #[test]
    fn routes_critic_is_it_true_positive() {
        let d = route("is it true that the moon is hollow?");
        assert!(d.needs_critic_vetting);
    }

    #[test]
    fn routes_critic_how_do_you_know_positive() {
        let d = route("how do you know that?");
        assert!(d.needs_critic_vetting);
    }

    #[test]
    fn routes_critic_visible_moon_negative() {
        let d = route("is the moon visible tonight?");
        assert!(!d.needs_critic_vetting);
    }

    #[test]
    fn routes_critic_short_why_negative() {
        let d = route("why is the sky blue?");
        assert!(!d.needs_critic_vetting);
    }

    // ---- code_intent ----

    #[test]
    fn routes_code_intent_write_python_positive() {
        let d = route("write a function in python to sort a list");
        assert!(d.code_intent);
    }

    #[test]
    fn routes_code_intent_generate_rust_positive() {
        let d = route("generate some rust code for fibonacci");
        assert!(d.code_intent);
    }

    #[test]
    fn routes_code_intent_language_statement_negative() {
        let d = route("python is a language");
        assert!(!d.code_intent);
    }

    #[test]
    fn routes_code_intent_concept_negative() {
        let d = route("what is fibonacci");
        assert!(!d.code_intent);
    }

    // ---- needs_agentic_thinking (OP-ROUTE-3c) ----

    #[test]
    fn op_route_3c_router_fires_agentic_thinking_on_how_do_i() {
        let d = route("How do I deploy a Rust binary step by step?");
        assert!(d.needs_agentic_thinking,
            "`how do i` + `step by step` must fire needs_agentic_thinking");
        assert!(d.triggers.iter().any(|t| t == "agentic_thinking"),
            "trigger label must surface in d.triggers");
    }

    #[test]
    fn op_route_3c_router_suppresses_agentic_thinking_on_definition() {
        let d = route("What is mitosis?");
        assert!(!d.needs_agentic_thinking,
            "plain definition query must NOT fire needs_agentic_thinking");
    }

    #[test]
    fn op_route_3c_router_co_fires_agentic_and_deep_on_walk_me_through() {
        // Intentional dual-fire per Section 2 of op_route_3c_design.md:
        // "walk me through" and "step by step" both fire the
        // pre-existing needs_deep_reasoning marker AND the new
        // needs_agentic_thinking marker. The shape selector
        // resolves the dual-fire downstream (rule 9.3 vs 9.5).
        let d = route("Walk me through how photosynthesis works step by step.");
        assert!(d.needs_agentic_thinking,
            "walk me through must fire needs_agentic_thinking");
        assert!(d.needs_deep_reasoning,
            "walk me through must also fire needs_deep_reasoning (intentional dual-fire)");
    }

    // ---- is_meta_query ----

    #[test]
    fn routes_meta_query_from_intent_positive() {
        let d = QueryRouter::route("what do you know about photosynthesis", QueryIntent::MetaSelf);
        assert!(d.is_meta_query);
    }

    #[test]
    fn routes_meta_query_generic_intent_negative() {
        let d = QueryRouter::route("what is photosynthesis", QueryIntent::Definition);
        assert!(!d.is_meta_query);
    }

    // ---- default / parity ----

    #[test]
    fn default_decision_all_false() {
        let d = RoutingDecision::default();
        assert!(!d.needs_math);
        assert!(!d.references_file_path);
        assert!(!d.references_image_path);
        assert!(!d.needs_deep_reasoning);
        assert!(!d.needs_variants);
        assert!(!d.needs_critic_vetting);
        assert!(!d.needs_agentic_thinking);
        assert!(!d.code_intent);
        assert!(!d.is_meta_query);
        assert!(d.triggers.is_empty());
    }

    #[test]
    fn baseline_definition_has_no_routes() {
        let d = QueryRouter::route("what is photosynthesis?", QueryIntent::Definition);
        assert!(!d.needs_math);
        assert!(!d.references_file_path);
        assert!(!d.references_image_path);
        assert!(!d.needs_deep_reasoning);
        assert!(!d.needs_variants);
        assert!(!d.needs_critic_vetting);
        assert!(!d.needs_agentic_thinking);
        assert!(!d.code_intent);
        assert!(!d.is_meta_query);
        assert!(d.triggers.is_empty());
    }
}
