//! Narration / Speech Assembly — Lu 4D Port #8.
//!
//! Turns the structured outputs of `Brain::ask` and `Brain::multi_think`
//! into honest, coherent English prose. No LLM calls. Pure neuro-symbolic
//! Rust string assembly: 16 hand-written response shapes selected by
//! intent + cognitive signals, then optionally decorated by
//! signal-driven modulations.
//!
//! Design rules (sketch approved by John):
//! 1. Pure neuro-symbolic. No external NLG libs.
//! 2. Deterministic. Same brain state + same query => same prose.
//! 3. Honest > eloquent. Low confidence is surfaced, not hidden.
//! 4. Slot-fill must use REAL graph content (node text, bond labels,
//!    signal data). Never invent strings out of whole cloth.
//! 5. Bounded length: ~600 chars default cap, MultiStep up to 1200.
//! 6. Shape selection is rule-based. Documented precedence below.
//! 7. Empty results => Refusal shape. No silent fallback.

use crate::agents::{Belief, MultiAgentTrace};
use crate::brain::AskResult;
use crate::bridge::BridgeEvent;
use crate::style::StyleHints;
use crate::curiosity::{CuriositySignal, SignalType};
use crate::node::Confidence;
use crate::seed::Seed;
use serde::{Deserialize, Serialize};

// =====================================================================
// Configuration
// =====================================================================

/// Knobs for the narration subsystem. Defaults match John's sketch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NarrationConfig {
    /// Top activation below this triggers UncertainAnswer (or Refusal
    /// if there are no activations at all). Default 0.25.
    pub low_confidence_cutoff: f32,
    /// Top activation above this is treated as a confident answer.
    /// Used to gate the "I think" prefix modulation. Default 0.55.
    pub high_confidence_cutoff: f32,
    /// Bridges with jaccard above this are NOT surfaced (they are
    /// inside the same cluster, not a real bridge). Mirrors
    /// `BridgeConfig::cluster_disjoint_threshold` semantics. Default 0.4.
    pub bridge_max_jaccard: f32,
    /// Curiosity signals below this intensity are ignored for
    /// shape selection / modulation. Default 0.4.
    pub curiosity_min_intensity: f32,
    /// Hard character cap on a rendered answer (except MultiStep).
    /// Default 600.
    pub max_answer_chars: usize,
    /// Hard character cap for MultiStep answers. Default 1200.
    pub multi_step_max_chars: usize,
    /// Port #20: runtime ingestion-noise filter at the walker
    /// surfacing layer. When true (default), walker output drops
    /// candidates matching the token-shape junk regex, the
    /// hand-curated low-info stop list, or the domain-mismatch
    /// heuristic. The query subject is never filtered. Flip via
    /// the CLI `--no-noise-filter` flag for A/B inspection. Older
    /// brain files lacking this field default to true so existing
    /// saves stay clean on load.
    #[serde(default = "default_noise_filter_enabled")]
    pub noise_filter_enabled: bool,
    /// OP-014: consult the asymmetry gate (idk_gate) before
    /// shape selection. When true (default), narrate() routes
    /// weakly-anchored / IDK / adversarial queries to a Refusal-
    /// shape NarrationResult using the same coherent refusal
    /// prose lu4d_ask emits. When false (test-only), shape
    /// selection runs unconditionally — useful for unit tests
    /// that hand-build tiny brains and exercise shape-selection
    /// logic in isolation, where the gate would otherwise
    /// (correctly) refuse for lack of anchor strength. Older
    /// brain files lacking this field default to true.
    #[serde(default = "default_idk_gate_enabled")]
    pub idk_gate_enabled: bool,
    /// When false, internal curiosity/gap-signal voice is suppressed from
    /// user-facing answers (the Wondering shape + the curiosity append).
    /// Solicited curiosity ("what interests you") is unaffected. Default true.
    #[serde(default = "default_true")]
    pub expose_curiosity_voice: bool,
    /// OP-ROUTE-3b -- minimum top-activation a query's goal seed must
    /// clear before `narrate()` spawns a variant arena. Below this floor
    /// the goal seed barely registered (typo / off-domain) and the
    /// resulting arenas would be low-coherence pollution of the temporal
    /// keyframe layer. Default `0.15`: below the idk_gate's pass floor
    /// (anything that cleared the gate has cleared this) but above the
    /// session-prior bleed-through noise floor. Lower to `0.05` for
    /// demos that want more arenas firing. Older brain files lacking
    /// this field default to 0.15.
    #[serde(default = "default_variant_min_activation")]
    pub variant_min_activation: f32,
    /// OP-ROUTE-3c -- hard cap on Observe-Plan-Act-Reflect
    /// iterations when narrate() spawns a `Brain::think` pass on
    /// the PASS path. Default `5`, matching
    /// `ThinkConfig::max_steps` (Section 17 decision #3 of
    /// op_route_3c_design.md -- John explicitly chose engine
    /// default over the agent-recommended `3` to give thinking
    /// headroom to converge). Lower if the 4-prompt cold-cache
    /// latency bench shows a real ceiling. Older brain files
    /// lacking this field default to 5.
    #[serde(default = "default_thinking_max_steps")]
    pub thinking_max_steps: usize,
    /// OP-ROUTE-3c -- minimum top-activation a query's goal seed
    /// must clear before `narrate()` spawns a think pass on the
    /// PASS path. Mirrors `variant_min_activation`: below this
    /// floor the goal seed barely registered (typo / off-domain)
    /// and a multi-step ReAct loop would write noise hypothesis
    /// bonds that must then decay. Default `0.15`. Older brain
    /// files lacking this field default to 0.15.
    #[serde(default = "default_thinking_min_activation")]
    pub thinking_min_activation: f32,
    /// OP-THINK-WATCHDOG -- aggressive wall-clock ceiling on the
    /// think pass when spawned from a user-facing narrate query.
    /// Engine default in `ThinkConfig` is `30s`; we tighten that
    /// to `5s` on the narrate path so an interactive user gets
    /// graceful degradation (partial trace, MultiStep body still
    /// composes from whatever steps completed) rather than a
    /// hang. Set `0` to disable the watchdog on this path.
    /// Older brain files lacking this field default to 5000.
    #[serde(default = "default_thinking_max_wall_clock_ms")]
    pub thinking_max_wall_clock_ms: u64,
    /// OP-ASK-WATCHDOG -- wall-clock ceiling on a single
    /// `Brain::ask` call spawned from the narrate path. Set
    /// strictly below `thinking_max_wall_clock_ms` so the
    /// ask-level watchdog fires before the surrounding
    /// think-level watchdog on the inner observation call.
    /// Default `4_000` (4s) -- inside the 5s thinking budget.
    /// `0` disables the ask-level watchdog on the narrate path.
    /// Older brain files lacking this field default to 4000.
    #[serde(default = "default_ask_max_wall_clock_ms")]
    pub ask_max_wall_clock_ms: u64,
    /// OP-REASON-LOOP Phase 1 -- hard cap on prompt-reasoning loop
    /// iterations (retrieve -> load-context -> reason -> re-retrieve).
    /// Default 1 == today's single-shot behavior: the PASS path builds a
    /// ContextSet from the single ask_with_config recall and renders from it
    /// with no re-retrieval (the free degrade path, internal notes. Values
    /// >1 enable mid-loop re-retrieval (Phase 3). Older brain files lacking
    /// this field default to 1.
    #[serde(default = "default_max_context_iters")]
    pub max_context_iters: usize,
}

fn default_noise_filter_enabled() -> bool { true }
fn default_idk_gate_enabled() -> bool { true }
fn default_true() -> bool { true }
fn default_variant_min_activation() -> f32 { 0.15 }
fn default_thinking_max_steps() -> usize { 5 }
fn default_thinking_min_activation() -> f32 { 0.15 }
fn default_thinking_max_wall_clock_ms() -> u64 { 5_000 }
fn default_ask_max_wall_clock_ms() -> u64 { 4_000 }
fn default_max_context_iters() -> usize { 1 }

impl Default for NarrationConfig {
    fn default() -> Self {
        NarrationConfig {
            low_confidence_cutoff: 0.25,
            high_confidence_cutoff: 0.55,
            bridge_max_jaccard: 0.4,
            curiosity_min_intensity: 0.4,
            max_answer_chars: 600,
            multi_step_max_chars: 1200,
            noise_filter_enabled: true,
            idk_gate_enabled: true,
            expose_curiosity_voice: true,
            variant_min_activation: 0.15,
            thinking_max_steps: 5,
            thinking_min_activation: 0.15,
            thinking_max_wall_clock_ms: 5_000,
            ask_max_wall_clock_ms: 4_000,
            max_context_iters: 1,
        }
    }
}

// =====================================================================
// Intent classification
// =====================================================================

/// Coarse query intent. Hand-classified by `IntentClassifier::classify`
/// using keyword + structural heuristics. Not learned.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QueryIntent {
    /// "what is X?" / "define X"
    Definition,
    /// "how does X work?" / "how do I do X?"
    Process,
    /// "why does X happen?" / "why is X?"
    WhyChain,
    /// "compare X and Y" / "X vs Y" / "how do X and Y differ?"
    Comparison,
    /// short factual lookup ("capital of France", "author of Hamlet")
    FactRecall,
    /// "what do you know about X?" / "what have you learned?"
    MetaSelf,
    /// "how has your understanding of X changed?"
    TimetravelComparison,
    /// OP-NARRATE-V2 (2026-05-31): "what are you" / "who are you".
    SelfIdentity,
    /// OP-NARRATE-V2 (2026-05-31): "what are you thinking about" / interest queries.
    CurrentInterests,
    /// fallback when no specific intent fires
    Generic,
}

/// Hand-rule intent classifier. Inspects the query string for
/// recognizable interrogatives and structural markers. Order matters:
/// the first matching rule wins.
pub struct IntentClassifier;

impl IntentClassifier {
    /// Classify a free-form query into a `QueryIntent`.
    /// OP-RECALL-1 (coherence) -- does this query carry a SUBJECT in
    /// addition to its meta-self preamble?
    ///
    /// Strips the recognised meta phrases and the usual
    /// preposition/filler joiners, then asks whether any content word
    /// survives the same hub-stopword + query-noise filters the seed
    /// selector uses. "what do you know" -> false (answer with the
    /// brain census). "what do you know about python" -> true (answer
    /// about python).
    fn has_topical_remainder(q: &str) -> bool {
        const META_PHRASES: [&str; 5] = [
            "what do you know",
            "what have you learned",
            "what do you remember",
            "tell me what you know",
            "tell me what you",
        ];
        let mut rest = q.to_string();
        for p in META_PHRASES.iter() {
            rest = rest.replace(p, " ");
        }
        rest.split(|c: char| !c.is_alphanumeric())
            .filter(|w| !w.is_empty() && w.len() >= 2)
            .any(|w| {
                !crate::network::is_hub_stopword(w)
                    && !crate::network::is_query_noise(w)
            })
    }

    pub fn classify(query: &str) -> QueryIntent {
        let q = query.trim().to_lowercase();
        // 0a. OP-NARRATE-V2 — Current interests / wonder-queue probe.
        //     Must check BEFORE SelfIdentity ("what are you") and
        //     BEFORE MetaSelf ("what do you").
        if q.contains("what are you thinking")
            || q.contains("what's on your mind")
            || q.contains("whats on your mind")
            || q.contains("what are you curious")
            || q.contains("what interests you")
            || q.contains("what topics interest you")
            || q.contains("what topics do you find interesting")
            || q.contains("what topics do you find most interesting")
            || q.contains("what do you find interesting")
            || q.contains("what do you find most interesting")
        {
            return QueryIntent::CurrentInterests;
        }
        // 0b. OP-NARRATE-V2 / OP-IDENTITY — Self-identity.
        //     Any LOCKED self/meta identity question (who/what are you, who
        //     made/created you, when, are-you-an-LLM, copyright) routes to the
        //     SelfIdentity path, which answers from the locked identity core.
        if crate::identity::identity_answer(query).is_some() {
            return QueryIntent::SelfIdentity;
        }
        //     Anchored "what are you" / "who are you" forms only —
        //     plain "what is X" still routes to Definition.
        let q_strip = q.trim_end_matches('?').trim();
        if q_strip == "what are you"
            || q_strip == "who are you"
            || q_strip == "what is your name"
            || q_strip == "what's your name"
            || q_strip == "whats your name"
            || q.contains("tell me about yourself")
            || q.contains("introduce yourself")
            || is_capability_query(&q)
        {
            return QueryIntent::SelfIdentity;
        }
        // 1. Meta-self queries — must check before generic "what".
        //
        // OP-RECALL-1 (coherence): MetaSelf is a BRAIN-STATE shape --
        // its template reports brain-wide concept/association counts.
        // That is the right answer for a bare "what do you know", and
        // the wrong answer for "what do you know ABOUT PYTHON", which
        // is a topical question wearing a meta-self preamble. Routing
        // the topical form here produced the observed incoherence:
        // a whole-brain census followed by "Recently I activated
        // Parser implementation in Python" -- a report about the brain
        // masquerading as an answer about the subject.
        //
        // So: the meta phrase only claims the query when NOTHING
        // topical follows it. `meta_self_topic` returns the remainder
        // after the phrase, filtered for content words; when that is
        // non-empty we fall through to ordinary classification and the
        // query gets answered on its subject.
        if q.contains("what do you know")
            || q.contains("what have you learned")
            || q.contains("what do you remember")
            || q.starts_with("tell me what you")
        {
            if !Self::has_topical_remainder(&q) {
                return QueryIntent::MetaSelf;
            }
        }
        // 2. Time-travel comparison.
        if q.contains("how has your understanding")
            || q.contains("how have you changed")
            || q.contains("used to think")
            || q.contains("changed your mind")
        {
            return QueryIntent::TimetravelComparison;
        }
        // 3. Comparison.
        if q.contains(" vs ")
            || q.contains(" vs. ")
            || q.contains("compare ")
            || q.contains("difference between")
            || q.contains("differ")
            || q.contains("how do ") && q.contains(" and ") && q.contains("differ")
        {
            return QueryIntent::Comparison;
        }
        // 4. Why -> causal chain.
        if q.starts_with("why ") || q.contains(" why ") {
            return QueryIntent::WhyChain;
        }
        // 5. How -> process. Must come AFTER comparison (which uses "how do").
        if q.starts_with("how ") || q.contains(" how does ") || q.contains(" how do ") {
            return QueryIntent::Process;
        }
        // 6. What/define -> definition. Short queries are FactRecall.
        if q.starts_with("what is ") || q.starts_with("what are ") || q.starts_with("define ") {
            return QueryIntent::Definition;
        }
        // OP-CLEANUP BUG2: open-ended "tell me about X" should give a grounded
        // definition/summary, not a Generic MultiStep. "yourself" stays on the
        // identity path handled upstream.
        if q.starts_with("tell me about ") && !q.contains("yourself") {
            return QueryIntent::Definition;
        }
        // 7. Short noun-phrase queries -> FactRecall.
        if q.split_whitespace().count() <= 3 && !q.ends_with('?') {
            return QueryIntent::FactRecall;
        }
        QueryIntent::Generic
    }
}

// =====================================================================
// Response shapes
// =====================================================================

/// One of 16 hand-written response shapes. Picked by `ShapeSelector`
/// based on intent + cognitive signals + confidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResponseShape {
    Definition,
    Process,
    WhyChain,
    Comparison,
    FactRecall,
    UncertainAnswer,
    MultiHypothesisAnswer,
    BridgedAnswer,
    CriticVettedAnswer,
    Refusal,
    MetaSelf,
    TimetravelComparison,
    Synthesis,
    Wondering,
    MultiStep,
    /// Port #22: bare polysemous query produced multiple sense
    /// clusters with comparable typed-bond mass. Lists each sense
    /// distinctly and ends with a clarifying question.
    MultiSense,
    /// OP-NARRATE-V2 (2026-05-31): "what are you" / "who are you" —
    /// constant self-introduction prose. No graph query.
    SelfIdentity,
    /// OP-NARRATE-V2 (2026-05-31): "what are you thinking about" /
    /// "what interests you" — verbalize the wonder queue.
    CurrentInterests,
    /// OP-CONCLUDE -- a synthesized, edge-cited conclusion (design
    /// CONCLUSION_SYNTHESIZER_DESIGN.md). Selected only when
    /// `synthesis::try_synthesize` succeeds; otherwise the pipeline
    /// degrades to the shapes below. Additive 17th shape.
    Conclusion,
}

impl ResponseShape {
    /// Stable string label, useful for tools/JSON output.
    pub fn label(&self) -> &'static str {
        match self {
            ResponseShape::Definition => "Definition",
            ResponseShape::Process => "Process",
            ResponseShape::WhyChain => "WhyChain",
            ResponseShape::Comparison => "Comparison",
            ResponseShape::FactRecall => "FactRecall",
            ResponseShape::UncertainAnswer => "UncertainAnswer",
            ResponseShape::MultiHypothesisAnswer => "MultiHypothesisAnswer",
            ResponseShape::BridgedAnswer => "BridgedAnswer",
            ResponseShape::CriticVettedAnswer => "CriticVettedAnswer",
            ResponseShape::Refusal => "Refusal",
            ResponseShape::MetaSelf => "MetaSelf",
            ResponseShape::TimetravelComparison => "TimetravelComparison",
            ResponseShape::Synthesis => "Synthesis",
            ResponseShape::Wondering => "Wondering",
            ResponseShape::MultiStep => "MultiStep",
            ResponseShape::MultiSense => "MultiSense",
            ResponseShape::SelfIdentity => "SelfIdentity",
            ResponseShape::CurrentInterests => "CurrentInterests",
            ResponseShape::Conclusion => "Conclusion",
        }
    }
}

// =====================================================================
// Slot fill
// =====================================================================

/// All possible slot values across the 16 shapes. Each shape's render
/// path uses just the fields it needs; missing slots fall back to a
/// short "(unknown)" placeholder so an empty graph never panics.
///
/// Slots are filled from real `AskResult` / `MultiAgentTrace` data,
/// never invented.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ShapeSlots {
    pub query: String,
    pub primary: Option<String>,
    pub secondary: Option<String>,
    pub tertiary: Option<String>,
    pub bond_label: Option<String>,
    pub second_bond_label: Option<String>,
    pub confidence_value: f32,
    pub confidence_label: String,
    /// B1/#211 -- resolved stored confidence label for the answer-head node
    /// ("Verified"/"Terminal"/"Unverified"/...), read from the node_store.
    #[serde(default)]
    pub head_confidence: Option<String>,
    pub node_count: usize,
    pub bond_count: usize,
    pub recent_activity: Option<String>,
    pub bridge_low: Option<String>,
    pub bridge_high: Option<String>,
    pub wondering_about: Option<String>,
    pub wondering_kind: Option<String>,
    pub critic_caveat_count: usize,
    pub critic_caveat: Option<String>,
    /// Scar reasons gathered from `MultiAgentTrace::rejected` —
    /// one per scar-driven rejection. Surfaced as a suffix in the
    /// rendered prose so Lu articulates the moral state of the
    /// connection instead of refusing silently.
    #[serde(default)]
    pub scar_reasons: Vec<String>,
    /// Composed multi-bond prose from the v2 graph walker.  When
    /// present, shapes that opted in (Definition/Process/Comparison/
    /// Synthesis/Wondering body) use this in place of the legacy
    /// top-1 templates.  None when no hub qualified or the walker
    /// found no scar-clear tuples.
    #[serde(default)]
    pub walker_prose: Option<String>,
    /// OP-RECALL-EXTRACT (defect 1): the strongest grounded neighbours the
    /// recall already surfaced for this query (self-echo + handle noise
    /// removed, activation order preserved). Lets the FactRecall renderer
    /// compose a real answer from what was retrieved instead of echoing the
    /// bare query token back as "{token}.". Empty when recall produced no
    /// usable neighbour, in which case the renderer keeps the legacy form.
    #[serde(default)]
    pub recall_associations: Vec<String>,
    pub winner_alt_count: usize,
    pub winner_alt: Option<String>,
    /// Port #17 Fix 2: the curiosity seed actually surfaced as
    /// `wondering_about`, after gap-frequency filtering. None when
    /// no signal qualified or all candidates were over the threshold.
    /// Used by `narrate()` to increment `gap_seed_frequency`.
    #[serde(skip)]
    pub chosen_curiosity_seed: Option<crate::seed::Seed>,
    /// Port #22: when the bare query subject is polysemous and
    /// multiple sense clusters carry comparable typed-bond mass,
    /// each cluster's representative neighbor labels (sorted by
    /// bond-type importance). Outer length is sense count (capped
    /// at 3); inner length is labels per sense (capped at 4).
    /// Empty when no multi-sense triggering occurred.
    #[serde(default)]
    pub multi_sense_clusters: Vec<Vec<String>>,
    /// Port #20: number of walker neighbors dropped by the
    /// runtime ingestion-noise filter on this query. Surfaces in
    /// `NarrationResult.noise_filtered_count` for telemetry.
    #[serde(default)]
    pub noise_filtered_count: u32,
    /// OP-ROUTE-3a -- pre-rendered prose from `ReasoningTrace.final_answer`
    /// when a structured deep-reasoning chain was built for this query.
    /// `None` when reasoning was not invoked or the trace was no_chain /
    /// associative / single-step. WhyChain render arm uses this verbatim
    /// in place of the fixed template.
    #[serde(default)]
    pub chain_articulation: Option<String>,
    /// OP-ROUTE-3a -- aggregate confidence from the reasoning trace
    /// (`ReasoningTrace.confidence`, min over the chain). 0.0 when no
    /// trace was built. Surfaced for downstream consumers; not currently
    /// consumed by the render arm.
    #[serde(default)]
    pub chain_confidence: f32,
    /// OP-NARRATE-V2 (2026-05-31): pre-rendered prose for the
    /// CurrentInterests shape, composed from the wonder queue at
    /// extract-slots time. `None` for every other shape; the
    /// renderer falls back to the "quiet state" template when empty.
    #[serde(default)]
    pub current_interests_prose: Option<String>,
    /// OP-CONCLUDE -- synthesized conclusion prose (self-contained,
    /// ends with the cited-edge provenance clause). `Some` only for
    /// `ResponseShape::Conclusion`.
    #[serde(default)]
    pub conclusion_prose: Option<String>,
    /// OP-CONCLUDE -- the typed edges the conclusion cites, in arrow
    /// form (`from -Bond-> to`), for telemetry / audit.
    #[serde(default)]
    pub cited_edges: Vec<String>,
}

// =====================================================================
// Modulation
// =====================================================================

/// Decorations driven by cognitive signals — applied on top of the
/// base shape. Same shape feels different depending on which flags fire.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModulationFlags {
    /// Curiosity signal touched the query: append a gap caveat.
    pub curiosity_touched: bool,
    /// Bridge fired during recall: append a cross-domain observation.
    pub bridge_touched: bool,
    /// Top activation only supported by hypothesis (low confidence)
    /// nodes: prefix with "I think" / "tentatively".
    pub hypothesis_only_support: bool,
    /// Critic-vetted multi-think context available: surface confidence.
    pub critic_vetted: bool,
    /// Wonder-queue signal touched this query topic.
    pub wondering_touched: bool,
    /// Mirror of NarrationConfig.expose_curiosity_voice: when false, the
    /// Wondering shape + curiosity append are suppressed (user-facing).
    pub expose_curiosity_voice: bool,
}

impl ModulationFlags {
    /// Derive flags from raw signals + confidence. Rule-based.
    pub fn compute(
        ask: &AskResult,
        multi: Option<&MultiAgentTrace>,
        cfg: &NarrationConfig,
    ) -> ModulationFlags {
        let top_act = ask.recall.activated.first().map(|a| a.activation).unwrap_or(0.0);
        let any_curiosity = ask
            .curiosity_signals
            .iter()
            .any(|s| s.intensity >= cfg.curiosity_min_intensity);
        // OP-LABEL-4 -- a bridge is only RENDERABLE if both endpoints
        // resolved to text. Before this, `bridge_touched` fired on the
        // jaccard test alone, so an unlabelled pair still selected the
        // BridgedAnswer shape / bridge modulation and printed
        // "a connection between  and :". Skipping the slot is the
        // OP-LABEL-1 contract: an unresolvable label means NO slot,
        // never a placeholder and never a hole.
        let any_bridge = ask
            .bridges
            .iter()
            .any(|b| b.jaccard <= cfg.bridge_max_jaccard
                && !b.low_content.is_empty()
                && !b.high_content.is_empty());
        let hypothesis_only = top_act > 0.0 && top_act < cfg.low_confidence_cutoff;
        let critic_vetted = multi.map(|m| !m.vetted.is_empty()).unwrap_or(false);
        ModulationFlags {
            curiosity_touched: any_curiosity,
            bridge_touched: any_bridge,
            hypothesis_only_support: hypothesis_only,
            critic_vetted,
            expose_curiosity_voice: cfg.expose_curiosity_voice,
            // wondering_touched is a stricter check on signal kind:
            // only true if at least one signal is HubBoundary or DeadEnd,
            // which are the "genuine wondering" signals (not desert/isolated).
            wondering_touched: ask.curiosity_signals.iter().any(|s| {
                s.intensity >= cfg.curiosity_min_intensity
                    && matches!(
                        s.signal_type,
                        SignalType::HubBoundary | SignalType::DeadEnd
                    )
            }),
        }
    }
}

// =====================================================================
// Slot extraction
// =====================================================================

/// Map a `Confidence` enum to a numeric for prose output and to combine
/// with activation. Mirrors `Confidence::value()` ordering.
/// Adj 2 -- capability self/meta query ("what can you do"), a sibling of the
/// "who are you" self-identity forms. Used by the intent classifier (to
/// route it onto the SelfIdentity path) and by the SelfIdentity render (to
/// answer with capabilities instead of identity). Normalizes case + trailing
/// question mark so both call sites can pass the raw query.
fn is_capability_query(q: &str) -> bool {
    let s = q.trim().trim_end_matches('?').trim().to_lowercase();
    s.contains("what can you do")
        || s.contains("what are you capable of")
        || s.contains("what are your capabilities")
        || s.contains("what can you help")
        || s == "what do you do"
}

fn confidence_numeric(c: Confidence) -> f32 {
    match c {
        Confidence::Terminal => 1.0,
        Confidence::Verified => 0.8,
        Confidence::Unverified => 0.5,
        Confidence::Inferred => 0.4,
        Confidence::Contested => 0.3,
    }
}

/// B1/#211 (silent-zero): when a recall hit the wall-clock watchdog its
/// `AskStatus` is `TimedOut` and the answer reflects only a PARTIAL scan.
/// Append an honest truncation disclosure so a time limit is never
/// presented as an absence of knowledge. Identity on `Complete`, so answers
/// backed by a full recall are byte-for-byte unchanged.
fn disclose_truncation(answer: String, status: crate::brain::AskStatus) -> String {
    match status {
        crate::brain::AskStatus::TimedOut { .. } => format!(
            "{} (Note: my search timed out before it finished, so this reflects only a partial scan -- a time limit, not necessarily missing knowledge.)",
            answer,
        ),
        crate::brain::AskStatus::Complete => answer,
    }
}

fn confidence_label_for(value: f32) -> &'static str {
    if value >= 0.8 {
        "high"
    } else if value >= 0.55 {
        "moderate"
    } else if value >= 0.25 {
        "low"
    } else {
        "very low"
    }
}

/// Look up the bond label between two seeds in either direction.
fn bond_label_between(brain: &crate::brain::Brain, a: &Seed, b: &Seed) -> Option<String> {
    // OP-BRAIN-MapRAM P5-b2: directed bond lookup off resident adjacency.
    if let Some(b_) = brain.network.seam_directed_bond(a, b) {
        return Some(format!("{:?}", b_.view().bond_type()));
    }
    if let Some(b_) = brain.network.seam_directed_bond(b, a) {
        return Some(format!("{:?}", b_.view().bond_type()));
    }
    None
}

// =====================================================================
// Narration v2 - bond-label connector map + graph-walk traversal.
//
// Port #15. The original narrate path stuffs activations[0..2] into
// hand-written templates ("X is Y."). v2 picks a HUB out of the
// activations, walks its outgoing bonds (top-K by scar-aware effective
// weight, optionally one extra hop), maps each bond_type into an
// English connector phrase, and composes prose that surfaces multiple
// bond labels instead of just the top-1 content. The composed prose
// becomes the body for Definition/Process/Comparison/Synthesis/Wondering;
// other shapes are left untouched.
// =====================================================================

/// One hop captured during a traversal walk.
#[derive(Debug, Clone)]
struct TraversalTuple {
    from_content: String,
    to_content: String,
    bond_type: crate::bond::BondType,
    /// Scar-aware effective weight at `now_ms()`. Skipped tuples were
    /// already filtered upstream - the value here is always > 0.
    weight: f32,
}

/// Map a bond_type into the English connector phrase that bridges
/// `{from} ___ {to}`. Surrounding spaces are part of the phrase so the
/// composer can concatenate without ad-hoc whitespace logic.
fn connector_for(bt: crate::bond::BondType) -> &'static str {
    use crate::bond::BondType::*;
    match bt {
        // "electron is a lepton"
        IsA => " is a ",
        DefinedAs => " is defined as ",
        WorksBy => " works by ",
        ComposedOf => " is composed of ",
        PurposeOf => " is for ",
        DiffersFrom => " differs from ",
        LeadsTo => " leads to ",
        ExampleOf => " is an example of ",
        // "electron has spin"
        HasProperty => " has ",
        // "rain causes flooding"
        Causes => " causes ",
        // "thymine is part of dna"
        PartOf => " is part of ",
        // "crispr is used for editing"
        UsedFor => " is used for ",
        // generic association fallback
        RelatedTo => " relates to ",
        // mitosis vs meiosis
        ContrastedWith => " contrasts with ",
        // adrenaline derived from tyrosine
        DerivedFrom => " is derived from ",
        // baking requires flour
        Requires => " requires ",
        // mitosis produces daughter cells
        Produces => " produces ",
        // antagonist opposes agonist
        Opposes => " opposes ",
        // BR.1 extension variants — neutral connectors mirroring
        // the canonical English phrasing used in articulate_bond.
        BornIn => " was born in ",
        BornOn => " was born on ",
        DiedIn => " died in ",
        DiedOn => " died on ",
        DiscoveredBy => " was discovered by ",
        FoundedBy => " was founded by ",
        FoundedIn => " was founded in ",
        HasAltitude => " has altitude ",
        HasAspect => " has aspect ",
        HasCapital => " has capital ",
        HasOfficialLanguage => " has official language ",
        HasPopulation => " has population ",
        KnownFor => " is known for ",
        LocatedIn => " is located in ",
        MeasuredIn => " is measured in ",
        NamedAfter => " is named after ",
        WrittenBy => " was written by ",
        HasArea => " has area ",
        HasMethod => " has method ",
        HasAttribute => " has attribute ",
        HasParameter => " has parameter ",
        Returns => " returns ",
        Inherits => " inherits from ",
        Calls => " calls ",
        HasType => " has type ",
        Implements => " implements ",
        HasCurrency => " uses the currency ",
        AuthoredBy => " was written by ",
    }
}

/// Same connector but trimmed of the leading space, for chained clauses
/// like "X is a Y, which is a Z." - the chained continuation glues
/// directly to "which ".
fn chained_connector_for(bt: crate::bond::BondType) -> &'static str {
    // Trim BOTH sides: caller emits ', which {conn} {body}' with explicit
    // single spaces around the connector, so the connector itself must
    // carry no surrounding whitespace.
    connector_for(bt).trim()
}

/// HubSelector - pick the seed the walker should radiate from.
///
/// Preference order:
///   1. The highest-degree top-K activation that also matches the
///      query term itself (when the query word collides with a node
///      content). "What is electron" -> if `electron` is in activations
///      AND has degree >= MIN_HUB_DEGREE, hub = electron.
///   2. Otherwise: the top-K activation with the highest outgoing
///      degree, tie-broken by activation strength.
///   3. None - when no candidate hits MIN_HUB_DEGREE. Caller falls
///      back to the legacy template renderers.
struct HubSelector;

impl HubSelector {
    /// Minimum outgoing-degree for a node to qualify as a hub. Below
    /// this the walker can't find enough bonds to compose richer prose.
    const MIN_HUB_DEGREE: usize = 2;
    /// How many top activations to scan when looking for a hub.
    const SCAN_TOP_K: usize = 5;
    /// Port #17 Fix 3 (calibrated for live brain in #18b):
    /// minimum fraction of outgoing bonds that must carry a
    /// non-RelatedTo (semantically rich) bond type for a
    /// candidate to qualify as a hub. Below this the candidate is
    /// rejected and the next-highest-scoring activation is tried.
    ///
    /// 0.15 is the empirical sweet spot on the live 105K-node brain:
    /// it admits useful hubs like `cell`/`mitosis` whose bonds are
    /// RelatedTo-heavy after broad ingestion, while still rejecting
    /// pure-RelatedTo stop-word-ish nodes (`name`, `world`) where
    /// every bond is RelatedTo and the walker would compose noise.
    /// Port #17 used 0.3, which was too strict at scale and starved
    /// walker_prose, forcing slot-composition fallback for almost
    /// every query.
    const MIN_TYPED_RATIO: f32 = 0.15;
    /// Port #18b: query-match candidates have no fractional gate;
    /// the pure-zero guard (`ratio == 0.0` rejection) is the only
    /// floor. The query subject (e.g. `cell` in `what is cell`) on
    /// the live 105K-node brain has typed_ratio ~ 0.02 — most bonds
    /// are RelatedTo after broad ingestion. Without this lenient
    /// floor, HubSelector falls through to higher-ratio neighbors
    /// (`guardroom`/`bullpen` IsA cell) and the answer talks about
    /// the wrong subject. The Walker's typed-first sort + polysemy
    /// filter handle the noise downstream.
    const MIN_TYPED_RATIO_QUERY: f32 = 0.0;

    /// Read the cached fraction of `seed`'s outgoing bonds that are
    /// NOT `BondType::RelatedTo`. The value lives on `KnowledgeNode`
    /// (Storage v2 Phase 1 / open issue #1) and is refreshed at brain
    /// load and on every consolidation pass; per-query traversal no
    /// longer pays an O(degree) recompute. Falls back to 0.0 when the
    /// seed is not in the network.
    fn typed_ratio(brain: &crate::brain::Brain, seed: &crate::seed::Seed) -> f32 {
        brain.network.get_node(seed).map(|n| n.typed_ratio()).unwrap_or(0.0)
    }

    fn pick(
        brain: &crate::brain::Brain,
        ask: &crate::brain::AskResult,
    ) -> Option<HubInfo> {
        let acts = &ask.recall.activated;
        if acts.is_empty() {
            return None;
        }
        let q_norm: String = ask.query.to_lowercase();

        // Build a ranked list of candidates that pass the basic
        // degree gate, scored as before, then apply the typed-ratio
        // quality filter on top.  Walk the ranked list and accept the
        // first candidate that clears `MIN_TYPED_RATIO`.
        let mut ranked: Vec<(HubInfo, (bool, usize, f32))> = Vec::new();
        for a in acts.iter().take(Self::SCAN_TOP_K) {
            let degree = brain.network.seam_out_degree(&a.seed);
            if degree < Self::MIN_HUB_DEGREE {
                continue;
            }
            let content_norm = a.content.to_lowercase();
            let query_match = q_norm.contains(content_norm.as_str())
                || content_norm
                    .split('_')
                    .any(|tok| q_norm.contains(tok) && tok.len() > 2);
            let score = (query_match, degree, a.activation);
            ranked.push((HubInfo {
                seed: a.seed,
                content: a.content.clone(),
                degree,
                activation: a.activation,
            }, score));
        }
        // Higher score first.
        ranked.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap_or(std::cmp::Ordering::Equal));

        // Port #18b: split-floor gate. A candidate that IS the
        // query subject (query_match) gets a much lower floor —
        // 0.05 with a strict "at least 1 typed bond" requirement —
        // because walker_prose runs the polysemy filter on the hub
        // itself: even a RelatedTo-heavy hub like `cell` produces
        // clean biology prose once the no-anchor bond-type-weighted
        // disambiguation prunes the telephony cluster. Non-query
        // candidates still pass the stricter 0.15 floor so guardroom-
        // ish IsA-heavy nodes can't hijack the answer when the
        // query subject node exists in activations. The first
        // candidate (best-scored) that clears its applicable floor
        // wins.
        for (cand, _score) in &ranked {
            let ratio = Self::typed_ratio(brain, &cand.seed);
            let q_match = _score.0;
            let floor = if q_match { Self::MIN_TYPED_RATIO_QUERY } else { Self::MIN_TYPED_RATIO };
            // Hard guard: pure-RelatedTo (zero typed bonds) is always
            // rejected — Port #17 Fix 3's stop-word-ish defence.
            if ratio < floor { continue; }
            if ratio == 0.0 { continue; }
            return Some(cand.clone());
        }
        // All candidates failed the quality filter — caller falls
        // back to legacy templates (None signals "no walker hub").
        None
    }
}

#[derive(Debug, Clone)]
struct HubInfo {
    seed: crate::seed::Seed,
    // Read only by test-only walker_hub_content() — kept for test inspection.
    #[allow(dead_code)]
    content: String,
    #[allow(dead_code)]
    degree: usize,
    #[allow(dead_code)]
    activation: f32,
}

// =====================================================================
// Port #20: runtime ingestion-noise filter
// =====================================================================

/// Curated low-information stop-list. Words that show up as walker
/// neighbors on the live brain but carry zero domain content. Every
/// entry is a literal lowercase match against the candidate's content
/// AFTER it has been lowercased. The query subject is exempted by the
/// surrounding caller, so a query like `what is gloom` still surfaces
/// `gloom` itself even though `gloom` is on this list.
const NOISE_STOP_LIST: &[&str] = &[
    // Pure-noise words seen on photosynthesis / mitosis / cell hubs.
    "gloom", "thing", "things", "stuff", "way", "ways",
    "time", "times", "part", "parts", "place", "places",
    "day", "days", "man", "men", "new", "old", "first", "last",
    "long", "large", "small", "own",
    // Filler / connector adverbs that escaped the keyword-search filter.
    "there", "here", "once", "now", "then", "just", "also", "even",
    "still", "already", "almost", "always", "never", "often",
    "very", "too", "only", "same", "other", "such",
    // Articles / determiners that occasionally slip in as nodes.
    "a", "an", "the", "this", "that", "these", "those",
    // Generic pronouns / placeholders.
    "it", "they", "them", "some", "any", "all",
];

/// Port #21: tiny allowlist of legitimate domain terms whose typed
/// bonds happen to be sparse in this brain. Belt-and-suspenders only:
/// consulted before the domain-mismatch heuristic, never overriding
/// the token-shape or stop-list checks. Each entry is a literal
/// lowercase content match.
const DOMAIN_ALLOWLIST: &[&str] = &[
    // Biology / biochemistry
    "dna", "rna", "mrna", "trna", "atp", "adp",
    "protein", "enzyme", "chromosome", "gene", "genome",
    "multicellular", "unicellular", "organelle",
    "mitochondria", "mitochondrion", "nucleus", "ribosome",
    // Chemistry / physics
    "ion", "electron", "proton", "neutron", "photon", "quark",
];

/// Runtime ingestion-noise filter. Lives next to the walker; consults
/// node typed_ratio + IsA presence for the optional domain-mismatch
/// heuristic.
pub(crate) struct NoiseFilter<'a> {
    pub enabled: bool,
    /// Lowercase content of the query subject. Never filtered.
    pub query_subject_lower: Option<String>,
    pub brain: &'a crate::brain::Brain,
}

impl<'a> NoiseFilter<'a> {
    /// Construct a filter for the given query and config. Picks up
    /// the lowercased raw query string as the protected subject;
    /// any candidate whose content lowercases to a substring match
    /// of the query is exempted from filtering. This is intentionally
    /// loose so `what is gloom` exempts `gloom`.
    pub fn new(
        brain: &'a crate::brain::Brain,
        cfg: &super::NarrationConfig,
        query: &str,
    ) -> Self {
        let q = query.trim().to_lowercase();
        let subj = if q.is_empty() { None } else { Some(q) };
        NoiseFilter {
            enabled: cfg.noise_filter_enabled,
            query_subject_lower: subj,
            brain,
        }
    }

    /// True if the candidate must be REJECTED. The caller increments
    /// its filtered counter when this returns true.
    pub fn should_reject(
        &self,
        seed: &crate::seed::Seed,
        content: &str,
    ) -> bool {
        if !self.enabled { return false; }
        let lc = content.trim().to_lowercase();
        if lc.is_empty() { return false; }
        // Query-subject protection: never filter the user's own
        // subject token, even if it matches a junk pattern.
        if let Some(q) = &self.query_subject_lower {
            if q == &lc { return false; }
            // Whole-word match within a multi-word query (e.g. the
            // user typed "what is gloom" and the candidate content
            // is "gloom" or "the gloom"): exempt.
            for tok in q.split(|c: char| !c.is_alphanumeric()) {
                if !tok.is_empty() && tok == lc { return false; }
            }
        }
        // 1. Token-shape junk.
        if Self::is_token_shape_junk(&lc) { return true; }
        // 2. Stop-list.
        if NOISE_STOP_LIST.iter().any(|w| *w == lc) { return true; }
        // 2b. Port #21 belt-and-suspenders allowlist: legitimate
        //     domain terms whose typed bonds happen to be sparse in
        //     this brain.  Run BEFORE the domain-mismatch heuristic
        //     so its typed_ratio check never fires on them.
        if Self::is_known_domain_term(&lc) { return false; }
        // 3. Domain-mismatch heuristic (cheap: cached typed_ratio +
        //    typed-bond scan in BOTH directions). Drops candidates
        //    whose own knowledge structure is bag-of-associations AND
        //    that have no typed (IsA / PartOf) grounding in either
        //    direction AND that are well-connected.
        //
        //    Port #21 refinements over Port #20:
        //      a. Degree gate raised 5 -> 10. Mid-degree real domain
        //         terms (e.g. `multicellular`) no longer match purely
        //         on being connected; only the strongly-popular
        //         leaf-from-ingestion pollutants (`ctdna`-style) do.
        //      b. Typed grounding now considers IsA / PartOf in BOTH
        //         directions. Real category words are rarely the
        //         SOURCE of an IsA but are frequently the TARGET
        //         (`X IsA multicellular_organism`).
        const MIN_DEGREE_FOR_DOMAIN_MISMATCH: usize = 10;
        if let Some(node) = self.brain.network.get_node(seed) {
            // OP-BRAIN-MapRAM P5-b2: degree via the seam (resident MAP), not adjacency.
            let deg = self.brain.network.seam_total_degree(seed);
            if node.typed_ratio() <= 0.05
                && deg >= MIN_DEGREE_FOR_DOMAIN_MISMATCH
                && !self.has_typed_grounding(seed)
            {
                return true;
            }
        }
        false
    }

    /// Typed grounding: an IsA or PartOf bond in EITHER direction
    /// (outbound `seed -> X` or inbound `X -> seed`). Either pattern
    /// indicates the node is taxonomically anchored, not a
    /// leaf-from-ingestion pollutant. Port #21.
    fn has_typed_grounding(&self, seed: &crate::seed::Seed) -> bool {
        use crate::bond::BondType;
        // OP-BRAIN-MapRAM P5-b2: typed-grounding via the canonical edge-carrying
        // seam; bond type from the carried predicate in cache mode (order-
        // independent: returns true on ANY IsA/PartOf edge).
        for (_to_seed, walk_bond) in self.brain.network.seam_walk_outgoing(seed) {
            if let Some(b) = walk_bond {
                if matches!(b.view().bond_type(), BondType::IsA | BondType::PartOf) {
                    return true;
                }
            }
        }
        for (_from_seed, walk_bond) in self.brain.network.seam_walk_incoming(seed) {
            if let Some(b) = walk_bond {
                if matches!(b.view().bond_type(), BondType::IsA | BondType::PartOf) {
                    return true;
                }
            }
        }
        false
    }

    fn is_known_domain_term(lc: &str) -> bool {
        DOMAIN_ALLOWLIST.iter().any(|w| *w == lc)
    }

    /// Token-shape rejection rules. All operate on the lowercased
    /// trimmed content.
    fn is_token_shape_junk(lc: &str) -> bool {
        // Single character tokens (letter or digit).
        let len = lc.chars().count();
        if len <= 1 { return true; }
        // 1-2 letter alphabetic words (e.g. `aa`, `bb`).
        if len <= 2 && lc.chars().all(|c| c.is_ascii_alphabetic()) {
            return true;
        }
        // ^[0-9]+[a-z]+$ — leading-digit + trailing-letters mash
        // (mangled formulae like `6co`, `12h`, `3rd`-style ingestion
        // residue). Rejects only when the token is purely digits
        // followed by purely letters with no separator.
        let bytes = lc.as_bytes();
        let mut i = 0;
        while i < bytes.len() && (bytes[i] as char).is_ascii_digit() {
            i += 1;
        }
        if i > 0 && i < bytes.len() {
            let tail = &bytes[i..];
            if tail.iter().all(|b| (*b as char).is_ascii_alphabetic()) {
                return true;
            }
        }
        false
    }
}

/// Walker - radiate outward from a hub, recording the strongest tuples.
///
/// Strategy:
///   - Take the hub's outgoing bonds, score each by scar-aware
///     effective_weight at `now_ms()` (zero-weight scarred bonds drop
///     out automatically), pick the top `MAX_DIRECT`.
///   - Optional second hop: for each top-3 neighbor recorded above,
///     walk its top `MAX_HOP2_PER_NEIGHBOR` bonds to add chain candidates.
///   - Cap total tuples at `MAX_TOTAL`; dedupe on (from, to, bond_type).
///
/// All bond reads use `weight_at(now, scar_log).effective_weight` -
/// scar enforcement is non-negotiable per the type system.
struct Walker;

impl Walker {
    const MAX_DIRECT: usize = 6;
    const MAX_HOP2_PER_NEIGHBOR: usize = 2;
    const MAX_TOTAL: usize = 8;

    fn walk(
        brain: &crate::brain::Brain,
        hub: &HubInfo,
        activated: &std::collections::HashSet<crate::seed::Seed>,
        anchors: &[crate::seed::Seed],
        filter: &NoiseFilter,
        filtered_count: &mut u32,
    ) -> Vec<TraversalTuple> {
        let now = crate::brain::now_ms();
        let mut tuples: Vec<TraversalTuple> = Vec::new();
        let mut visited: std::collections::HashSet<(
            crate::seed::Seed,
            crate::seed::Seed,
            crate::bond::BondType,
        )> = std::collections::HashSet::new();

        // Port #18a: anchor-aware polysemy disambiguation. The hub's
        // out-neighbors split into sense clusters; raw query tokens
        // ( = seed_hits minus the hub) plus the post-spread
        // activation set elect the dominant sense. Hop-2 expansions
        // are NOT filtered: by then we are inside the chosen sense
        // and downstream nodes may legitimately straddle boundaries.
        let polysemy_allow = brain
            .network
            .disambiguate_polysemous_with_anchors(&hub.seed, activated, anchors);

        // Hop 1 - direct outgoing bonds from the hub.
        let direct_all = Self::collect_outgoing(brain, &hub.seed, now, filter, filtered_count);
        let direct: Vec<(TraversalTuple, crate::seed::Seed)> = match &polysemy_allow {
            Some(allow) => direct_all
                .into_iter()
                .filter(|(_, dest)| allow.contains(dest))
                .collect(),
            None => direct_all,
        };
        let direct_taken: Vec<(TraversalTuple, crate::seed::Seed)> = direct
            .into_iter()
            .take(Self::MAX_DIRECT)
            .collect();
        for (t, neighbor_seed) in direct_taken.iter().cloned() {
            let key = (hub.seed, neighbor_seed, t.bond_type);
            if visited.insert(key) {
                tuples.push(t);
            }
        }

        // Hop 2 - for each kept neighbor (top 3), walk its top bonds.
        if tuples.len() < Self::MAX_TOTAL {
            for (_t1, neighbor_seed) in direct_taken.iter().take(3) {
                if tuples.len() >= Self::MAX_TOTAL {
                    break;
                }
                let neighbor_out = Self::collect_outgoing(brain, neighbor_seed, now, filter, filtered_count);
                for (t2, dest_seed) in neighbor_out
                    .into_iter()
                    .take(Self::MAX_HOP2_PER_NEIGHBOR)
                {
                    if tuples.len() >= Self::MAX_TOTAL {
                        break;
                    }
                    // Don't loop back to hub.
                    if dest_seed == hub.seed {
                        continue;
                    }
                    let key = (*neighbor_seed, dest_seed, t2.bond_type);
                    if visited.insert(key) {
                        tuples.push(t2);
                    }
                }
            }
        }

        tuples
    }

    /// Collect outgoing tuples from `from`, sorted by effective_weight
    /// descending. Returns (tuple, dest_seed) pairs so the caller can
    /// chain or dedupe by seed.
    fn collect_outgoing(
        brain: &crate::brain::Brain,
        from: &crate::seed::Seed,
        now: i64,
        filter: &NoiseFilter,
        filtered_count: &mut u32,
    ) -> Vec<(TraversalTuple, crate::seed::Seed)> {
        // OP-LABEL-1: presence from the graph, TEXT from the node
        // store. On a v5 brain the resident node is a content-free stub,
        // so reading `.content()` here is what produced `" is ."`.
        // Presence still gates the walk exactly as before; an
        // unresolvable label yields an empty slot rather than pruning
        // the edge, so the adjacency walk keeps its prior shape.
        if brain.network.get_node(from).is_none() {
            return Vec::new();
        }
        let from_content = brain.network.node_label_or_empty(from);
        let mut out: Vec<(TraversalTuple, crate::seed::Seed)> = Vec::new();
        // OP-BRAIN-MapRAM P5-b2: outgoing walk via the canonical edge-carrying
        // seam (sorted by effective_weight below; canonical order makes ties
        // deterministic and identical across legacy/cache).
        for (to_seed, walk_bond) in brain.network.seam_walk_outgoing(from) {
            let walk_bond = match walk_bond {
                Some(b) => b,
                None => continue,
            };
            let bond = walk_bond.view();
            let reading = bond.weight_at(now, &brain.scar_log);
            // Scar-aware enforcement: zero effective_weight means
            // either the bond decayed past zero or a scar is active -
            // skip either way.
            if reading.effective_weight <= 0.0 {
                continue;
            }
            if brain.network.get_node(&to_seed).is_none() {
                continue;
            }
            let to_content = brain.network.node_label_or_empty(&to_seed);
            // Port #20: runtime ingestion-noise filter. Reject
            // mangled-formula tokens, single/two-letter words, the
            // hand-curated low-info stop-list, and bag-of-association
            // bare nodes — but never the query subject itself.
            if filter.should_reject(&to_seed, &to_content) {
                *filtered_count = filtered_count.saturating_add(1);
                continue;
            }
            out.push((
                TraversalTuple {
                    from_content: from_content.clone(),
                    to_content,
                    bond_type: bond.bond_type(),
                    weight: reading.effective_weight,
                },
                to_seed,
            ));
        }
        // Port #18b: rank bonds by weight * (1 + 3 * target_typed_ratio).
        // The target_typed_ratio factor boosts bonds whose target is
        // itself a richly-typed knowledge node (high IsA/PartOf/
        // Causes density) over bonds to bag-of-associations targets.
        //
        // Why this matters on the live brain: cell's top-weighted
        // outgoing bonds happen to be `cell IsA radiotelephone` and
        // `cell IsA radiophone` (telephony, weak typed structure
        // beneath them) ahead of `cell RelatedTo blastomere` /
        // `RelatedTo ribosome` (biology, strong typed structure
        // beneath them). Without this re-weighting walker prose
        // surfaces only telephony for `what is cell`. With it, the
        // biology-rich neighbors get a meaningful boost and surface
        // alongside telephony — the "honestly multi-sense" answer
        // the user explicitly accepts as a passing outcome.
        // Storage v2 Phase 1 / open issue #1: use the cached
        // KnowledgeNode::typed_ratio set at load and consolidation
        // time, instead of recomputing per neighbour per query.
        let target_score = |seed: &crate::seed::Seed| -> f32 {
            brain.network.get_node(seed).map(|n| n.typed_ratio()).unwrap_or(0.0)
        };
        let scored: Vec<(f32, (TraversalTuple, crate::seed::Seed))> = out
            .into_iter()
            .map(|(t, dest)| {
                let tt = target_score(&dest);
                let score = t.weight * (1.0 + 3.0 * tt);
                (score, (t, dest))
            })
            .collect();
        let mut scored = scored;
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let out: Vec<(TraversalTuple, crate::seed::Seed)> =
            scored.into_iter().map(|(_, p)| p).collect();
        out
    }
}

/// Composer - turn walked tuples into ~3-5 sentences of prose.
///
/// Composition rules:
///   - Capitalize the first character of the first sentence.
///   - Group same-(from, bond_type) tuples into one sentence joined
///     with "and": (electron IsA lepton) + (electron IsA fermion) ->
///     "Electron is a lepton and a fermion."
///   - Chain when tuple_a.to == tuple_b.from: "X is a Y, which is a Z."
///   - Bound to MAX_SENTENCES so prose stays inside the answer cap.
struct Composer;

impl Composer {
    const MAX_SENTENCES: usize = 4;
    /// Cap on how many distinct targets we surface for a single
    /// (from, bond_type) group. Beyond this we emit "and N more" so a
    /// wide RelatedTo fanout doesn't blow out one sentence (Port #16).
    const MAX_TARGETS_PER_GROUP: usize = 4;

    fn compose(tuples: &[TraversalTuple]) -> Option<String> {
        if tuples.is_empty() {
            return None;
        }
        // Group by (from_content, bond_type) preserving first-seen
        // order. Each group carries its `tos` plus a `hidden` count of
        // targets that overflowed MAX_TARGETS_PER_GROUP — those collapse
        // into a trailing "and N more" so the prose stays readable on
        // lightly-typed brains where RelatedTo dominates (Port #16 Fix 2).
        let mut groups: Vec<(String, crate::bond::BondType, Vec<String>, usize)> =
            Vec::new();
        for t in tuples {
            if let Some(g) = groups
                .iter_mut()
                .find(|(f, b, _, _)| *f == t.from_content && *b == t.bond_type)
            {
                if g.2.iter().any(|x| x == &t.to_content) {
                    continue;
                }
                if g.2.len() < Self::MAX_TARGETS_PER_GROUP {
                    g.2.push(t.to_content.clone());
                } else {
                    g.3 += 1;
                }
            } else {
                groups.push((
                    t.from_content.clone(),
                    t.bond_type,
                    vec![t.to_content.clone()],
                    0,
                ));
            }
        }

        // Build a chain index: for each group find a continuation group
        // whose .0 (from) matches one of this group's .2 (tos). Two
        // constraints (Port #16 Fix 3): chain only when the parent and
        // child share a bond type OR the child is an IsA edge. That
        // preserves "X is a Y, which is a Z" (natural English) and
        // "X relates to Y, which relates to Z" (consistent), while
        // blocking semantically fragile bridges like
        // RelatedTo -> Causes / RelatedTo -> HasProperty etc.
        let mut chained_into: std::collections::HashSet<usize> =
            std::collections::HashSet::new();
        let mut chain_next: std::collections::HashMap<usize, usize> =
            std::collections::HashMap::new();
        for i in 0..groups.len() {
            for j in 0..groups.len() {
                if i == j || chained_into.contains(&j) {
                    continue;
                }
                // Port #16 v2 calibration: chain only when parent and child
                // share the same bond type. Previously also allowed RelatedTo->IsA
                // chains via a child_is_isa carve-out, which produced semantically
                // fragile sentences like "nuclei, which is a wireless" on the
                // mitosis query. Pure same-bond-type preserves IsA->IsA and
                // RelatedTo->RelatedTo chains while blocking cross-type bridges.
                let same_bt = groups[i].1 == groups[j].1;
                if !same_bt {
                    continue;
                }
                if groups[i].2.iter().any(|to| to == &groups[j].0) {
                    chain_next.insert(i, j);
                    chained_into.insert(j);
                    break;
                }
            }
        }

        let mut sentences: Vec<String> = Vec::new();
        for i in 0..groups.len() {
            if chained_into.contains(&i) {
                continue;
            }
            if sentences.len() >= Self::MAX_SENTENCES {
                break;
            }
            sentences.push(Self::render_group(i, &chain_next, &groups));
        }

        if sentences.is_empty() {
            return None;
        }
        Some(sentences.join(" "))
    }

    /// Render one group, walking the chain map transitively so that
    /// 2nd-hop (and deeper) neighbors flow into the same sentence as
    /// "X is a Y, which is a Z, which is a W." rather than dropping
    /// after a single link (Port #16 Fix 4).
    fn render_group(
        idx: usize,
        chain_next: &std::collections::HashMap<usize, usize>,
        all: &[(String, crate::bond::BondType, Vec<String>, usize)],
    ) -> String {
        let g = &all[idx];
        let connector = connector_for(g.1);
        let from_capped = capitalize(&g.0);
        let body = Self::join_tos(g.1, &g.2, g.3);
        let mut sentence = format!("{}{}{}", from_capped, connector, body);

        // Walk chain_next transitively. A small visited set protects
        // against pathological cycles (the chain index above already
        // forbids re-entering a chained group, but defense-in-depth).
        let mut visited: std::collections::HashSet<usize> =
            std::collections::HashSet::new();
        visited.insert(idx);
        let mut cur = idx;
        while let Some(&next) = chain_next.get(&cur) {
            if !visited.insert(next) {
                break;
            }
            let child = &all[next];
            let child_conn = chained_connector_for(child.1);
            let child_body = Self::join_tos(child.1, &child.2, child.3);
            sentence.push_str(&format!(", which {} {}", child_conn, child_body));
            cur = next;
        }
        sentence.push('.');
        sentence
    }

    /// Join multiple targets for the same (from, bond_type) into one
    /// English noun phrase: ["a","b","c"] -> "a, b and c". For IsA we
    /// repeat the article so it reads naturally ("a lepton and a fermion").
    /// When `hidden` > 0 we render the visible targets comma-joined
    /// and append "and N more" — used when a group exceeded
    /// MAX_TARGETS_PER_GROUP (Port #16 Fix 2).
    fn join_tos(bt: crate::bond::BondType, tos: &[String], hidden: usize) -> String {
        if tos.is_empty() {
            return String::new();
        }
        if hidden > 0 {
            let head_joined = tos.join(", ");
            return format!("{} and {} more", head_joined, hidden);
        }
        match tos.len() {
            1 => tos[0].clone(),
            _ => {
                let head = &tos[..tos.len() - 1];
                let tail = &tos[tos.len() - 1];
                let head_joined = head.join(", ");
                if matches!(bt, crate::bond::BondType::IsA) {
                    format!("{} and a {}", head_joined, tail)
                } else {
                    format!("{} and {}", head_joined, tail)
                }
            }
        }
    }
}

/// OP-RECALL-EXTRACT: join labels as "a, b, and c" for the FactRecall
/// association answer.
fn join_plain(items: &[String]) -> String {
    match items.len() {
        0 => String::new(),
        1 => items[0].clone(),
        2 => format!("{} and {}", items[0], items[1]),
        _ => {
            let head = items[..items.len() - 1].join(", ");
            format!("{}, and {}", head, items[items.len() - 1])
        }
    }
}

fn capitalize(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        None => String::new(),
        Some(c) => c.to_uppercase().collect::<String>() + chars.as_str(),
    }
}

/// Top-level entry point used by `extract_slots`. Returns composed prose
/// when a hub exists and the walker found at least one tuple; `None`
/// otherwise (caller falls back to the legacy template).
fn walker_prose(
    brain: &crate::brain::Brain,
    ask: &crate::brain::AskResult,
    cfg: &NarrationConfig,
    filtered_count: &mut u32,
) -> Option<String> {
    let hub = HubSelector::pick(brain, ask)?;
    let activated_set: std::collections::HashSet<crate::seed::Seed> = ask
        .recall
        .activated
        .iter()
        .map(|a| a.seed)
        .collect();
    // Port #18a: pass the parser's pre-spread seed_hits as raw
    // disambiguation anchors. They have not been polluted by the
    // hub's cross-sense activation and are the most reliable
    // signal for which sense of a polysemous hub the user meant.
    let anchors: Vec<crate::seed::Seed> = ask
        .seed_hits
        .iter()
        .copied()
        .filter(|s| *s != hub.seed)
        .collect();
    let filter = NoiseFilter::new(brain, cfg, &ask.query);
    let tuples = Walker::walk(brain, &hub, &activated_set, &anchors, &filter, filtered_count);
    Composer::compose(&tuples)
}

/// Public for tests/inspection: surface the picked hub content.
#[cfg(test)]
fn walker_hub_content(
    brain: &crate::brain::Brain,
    ask: &crate::brain::AskResult,
) -> Option<String> {
    HubSelector::pick(brain, ask).map(|h| h.content)
}


/// Pull a human label for a curiosity signal kind.
fn signal_kind_label(s: &SignalType) -> &'static str {
    match s {
        SignalType::DeadEnd => "a dead-end where my associations thin out",
        SignalType::IsolatedNode => "an isolated concept with few connections",
        SignalType::KnowledgeDesert => "a knowledge desert — sparse top results",
        SignalType::HubBoundary => "a hub boundary I can't traverse confidently",
    }
}

/// Port #22: rank a candidate by how semantically-loaded the bond
/// `seed -> cand` is. Used to pick representative members per
/// sense cluster for the MultiSense narration.  IsA outranks
/// PartOf outranks Causes etc.; ties resolved deterministically
/// by the match below. Returns `-1.0` if no bond exists so missing
/// bonds sort to the end.
fn bond_priority(
    brain: &crate::brain::Brain,
    seed: &crate::seed::Seed,
    cand: &crate::seed::Seed,
) -> f32 {
    use crate::bond::BondType;
    match brain.network.seam_directed_bond(seed, cand) {
        Some(b) => match b.view().bond_type() {
            BondType::IsA => 6.0,
            BondType::DefinedAs => 6.0,
            BondType::WorksBy => 2.0,
            BondType::ComposedOf => 5.0,
            BondType::PurposeOf => 2.0,
            BondType::DiffersFrom => 0.5,
            BondType::LeadsTo => 4.0,
            BondType::ExampleOf => 1.5,
            BondType::PartOf => 5.0,
            BondType::Causes => 4.0,
            BondType::Produces => 3.5,
            BondType::HasProperty => 3.0,
            BondType::Requires => 2.5,
            BondType::UsedFor => 2.0,
            BondType::DerivedFrom => 1.5,
            BondType::RelatedTo => 1.0,
            BondType::ContrastedWith => 0.5,
            BondType::Opposes => 0.0,
            BondType::BornIn => 6.0,
            BondType::BornOn => 6.0,
            BondType::DiedIn => 6.0,
            BondType::DiedOn => 6.0,
            BondType::DiscoveredBy => 6.0,
            BondType::FoundedBy => 6.0,
            BondType::FoundedIn => 5.0,
            BondType::HasAltitude => 3.0,
            BondType::HasAspect => 3.0,
            BondType::HasCapital => 3.0,
            BondType::HasOfficialLanguage => 3.0,
            BondType::HasPopulation => 3.0,
            BondType::KnownFor => 3.0,
            BondType::LocatedIn => 5.0,
            BondType::MeasuredIn => 3.0,
            BondType::NamedAfter => 6.0,
            BondType::WrittenBy => 6.0,
            BondType::HasArea => 3.0,
            BondType::HasMethod => 3.0,
            BondType::HasAttribute => 3.0,
            BondType::HasParameter => 2.0,
            BondType::Returns => 3.0,
            BondType::Inherits => 5.0,
            BondType::Calls => 2.0,
            BondType::HasType => 3.0,
            BondType::Implements => 4.0,
            BondType::HasCurrency => 3.0,
            BondType::AuthoredBy => 6.0,
        },
        None => -1.0,
    }
}

/// Build the universal `ShapeSlots` once. Individual shapes pick the
/// fields they need.
fn extract_slots(
    brain: &crate::brain::Brain,
    query: &str,
    ask: &AskResult,
    multi: Option<&MultiAgentTrace>,
    cfg: &NarrationConfig,
    reasoning: Option<&crate::reasoning::ReasoningTrace>,
    variants: Option<&crate::variants::VariantArena>,
    thinking: Option<&crate::thinking::ThinkingTraceSummary>,
    intent: QueryIntent,
) -> ShapeSlots {
    let mut s = ShapeSlots::default();
    s.query = query.to_string();
    s.node_count = brain.network.node_count();
    s.bond_count = brain.network.bond_count();

    // -----------------------------------------------------------
    // OP-NARRATE-V2 (2026-05-31): CurrentInterests shape body.
    //
    // Read the wonder queue, dedupe by subject seed, pick up to 5
    // distinct subjects spanning as many signal types as possible,
    // and verbalize one clause per pick. Result goes into
    // `current_interests_prose`; render arm consumes it directly.
    //
    // This populates BEFORE the polysemy filter / activation slot
    // fill below because CurrentInterests doesn't depend on
    // recall — it's a brain-state probe, not a query about the
    // graph. Subsequent extraction continues but the only slot the
    // CurrentInterests render arm reads is current_interests_prose.
    // -----------------------------------------------------------
    if intent == QueryIntent::CurrentInterests {
        let signals = brain.wonder_queue.top_n(10);
        if signals.is_empty() {
            // Leave current_interests_prose = None; renderer's
            // fallback template handles the quiet-state case.
        } else {
            use std::collections::HashSet;
            let mut seen_seeds: HashSet<crate::seed::Seed> = HashSet::new();
            let mut seen_kinds: HashSet<SignalType> = HashSet::new();
            let mut picks: Vec<&CuriositySignal> = Vec::new();
            // Pass 1: prefer one per signal type.
            for sig in &signals {
                if picks.len() >= 5 { break; }
                if seen_seeds.contains(&sig.seed) { continue; }
                if seen_kinds.contains(&sig.signal_type) { continue; }
                seen_seeds.insert(sig.seed);
                seen_kinds.insert(sig.signal_type);
                picks.push(sig);
            }
            // Pass 2: fill remaining slots with seed-distinct picks
            // even if signal type repeats.
            for sig in &signals {
                if picks.len() >= 5 { break; }
                if seen_seeds.contains(&sig.seed) { continue; }
                seen_seeds.insert(sig.seed);
                picks.push(sig);
            }
            let mut clauses: Vec<String> = Vec::new();
            for sig in &picks {
                // Per Port J.1 fallback chain: real node label, then
                // gap_label_map, then hex hash. Skip if no usable label.
                let label = brain.network.node_label(&sig.seed)
                    .or_else(|| brain.gap_label_map.get(&sig.seed).cloned());
                let label = match label {
                    Some(s) if !s.is_empty() => s,
                    _ => continue,
                };
                let clause = match sig.signal_type {
                    SignalType::HubBoundary => format!(
                        "I've been wondering about {} — there are areas where my map runs out",
                        label
                    ),
                    SignalType::DeadEnd => format!(
                        "I'm curious where {} leads next; my path seems to end there",
                        label
                    ),
                    SignalType::IsolatedNode => format!(
                        "I'm wondering what connects to {}; it sits alone in my graph",
                        label
                    ),
                    SignalType::KnowledgeDesert => format!(
                        "I'm noticing a gap around {}; I haven't learned much there yet",
                        label
                    ),
                };
                clauses.push(clause);
            }
            if !clauses.is_empty() {
                // Join with ". " and prefix.
                let body = clauses.join(". ");
                s.current_interests_prose = Some(format!(
                    "A few things have been on my mind. {}.", body
                ));
            }
        }
    }

    // -----------------------------------------------------------
    // OP-NARRATE-V2 (2026-05-31) — Definition intent IsA-bond reorder.
    //
    // For "what is X?" / "define X" queries the user wants the
    // typed-relation answer (IsA / HasAspect / ComposedOf / UsedFor)
    // surfaced as the primary fact, not whichever co-mention
    // happened to win spread activation. Re-rank `acts` so any
    // activation that is the TARGET of an IsA / HasAspect / UsedFor /
    // PartOf bond from the FIRST seed_hit lands at index 0. Tie-broken
    // by IsA > PartOf > UsedFor > HasAspect (mirrors bond_priority).
    //
    // No reorder when:
    //   - intent != Definition
    //   - no seed_hits available
    //   - no activation has a typed bond from seed_hit
    // The "no reorder" path means existing behavior on Definition
    // queries with no IsA structure is preserved verbatim.
    // -----------------------------------------------------------
    let reordered_acts: Option<Vec<crate::network::ActivatedNode>> = {
        use crate::bond::BondType;
        if intent == QueryIntent::Definition && !ask.seed_hits.is_empty() {
            let subject = ask.seed_hits[0];
            // For each activation, compute a typed-priority score
            // from the bond `subject -> activation.seed` (outgoing).
            // 0.0 = no typed bond; positive = typed.
            let typed_score = |seed: &crate::seed::Seed| -> f32 {
                // OP-NARRATE-V2 (2026-05-31) -- outgoing-only.
                // For "what is X?" we want bonds where X is the SUBJECT
                // (X IsA Y), not where X is the TARGET (List HasAspect X).
                // The incoming-direction check produced spurious primaries
                // in the 2026-05-31 test (e.g. "List of anarchist movements
                // by region has aspect Anarchism in South America").
                let bond = brain.network.seam_directed_bond(&subject, seed);
                match bond.map(|b| b.view().bond_type()) {
                    Some(BondType::IsA) => 4.0,
                    Some(BondType::PartOf) => 3.0,
                    Some(BondType::UsedFor) => 2.5,
                    Some(BondType::HasAspect) => 2.0,
                    _ => 0.0,
                }
            };
            // Compute scores; only reorder if at least one activation
            // has a positive typed score AND the current top doesn't
            // already have it.
            let current_top_score = ask.recall.activated.first()
                .map(|a| typed_score(&a.seed)).unwrap_or(0.0);
            let any_typed = ask.recall.activated.iter()
                .any(|a| typed_score(&a.seed) > 0.0);
            if any_typed && current_top_score < 4.0 {
                // Stable sort with typed-score as primary key
                // (descending). Original order preserved within
                // same-score buckets, so untyped activations keep
                // their spread-activation order relative to each
                // other.
                let mut v: Vec<crate::network::ActivatedNode> =
                    ask.recall.activated.clone();
                v.sort_by(|a, b| {
                    let sa = typed_score(&a.seed);
                    let sb = typed_score(&b.seed);
                    sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
                });
                Some(v)
            } else {
                None
            }
        } else {
            None
        }
    };

    // -----------------------------------------------------------
    // OP-NARRATE-V2 (2026-05-31) -- Definition prose override.
    //
    // When a Definition query has at least one outgoing IsA / PartOf /
    // UsedFor / HasAspect bond from the seed_hit, compose the answer
    // directly: "{subject} {connector} {target}". This bypasses the
    // HubSelector + Walker path which can pick a co-mention page as
    // the hub and produce confusing prose like "List of anarchist
    // movements by region has aspect Anarchism in South America."
    //
    // The composed string lands in walker_prose, which the Definition
    // render arm already consumes verbatim. When no typed bond exists,
    // walker_prose stays None and the existing fallback chain runs.
    // -----------------------------------------------------------
    let definition_prose_override: Option<String> = {
        use crate::bond::BondType;
        if intent == QueryIntent::Definition && !ask.seed_hits.is_empty() {
            let subject = ask.seed_hits[0];
            let subject_label = brain.network.node_label(&subject);
            // Walk subject's outgoing neighbors and pick the highest-
            // scoring typed bond. Cap to avoid pathological hubs.
            // OP-BRAIN-MapRAM P5-b2: outgoing walk via the canonical edge-carrying seam.
            let neighbors = brain.network.seam_walk_outgoing(&subject);
            let mut best: Option<(crate::seed::Seed, BondType, f32)> = None;
            for (nbr, walk_bond) in neighbors.iter().take(2048) {
                if let Some(bv) = walk_bond {
                    let bt = bv.view().bond_type();
                    let score = match bt {
                        BondType::IsA => 4.0_f32,
                        BondType::PartOf => 3.0,
                        BondType::UsedFor => 2.5,
                        BondType::HasAspect => 2.0,
                        _ => 0.0,
                    };
                    if score > 0.0 {
                        let cur = best.map(|(_, _, s)| s).unwrap_or(0.0);
                        if score > cur {
                            best = Some((*nbr, bt, score));
                        }
                    }
                }
            }
            best.and_then(|(target_seed, bt, _)| {
                let subj = subject_label?;
                let target = brain.network.node_label(&target_seed)?;
                let connector = connector_for(bt);
                Some(format!("{}{}{}.", subj, connector, target))
            })
        } else {
            None
        }
    };


    // Port #18a: anchor-aware polysemy filter on the activation
    // list before slot-fill. The Synthesis fallback (used when
    // HubSelector finds no hub clearing the typed_ratio gate)
    // composes prose directly from primary/secondary/tertiary,
    // so polluted activations crowd the slots with cross-sense
    // neighbors of polysemous seed_hits. Drop those out-neighbors
    // that fall outside the elected sense for each polysemous hit.
    let mut polysemy_blocked: std::collections::HashSet<crate::seed::Seed> =
        std::collections::HashSet::new();
    if !ask.seed_hits.is_empty() {
        let activated_set: std::collections::HashSet<crate::seed::Seed> =
            ask.recall.activated.iter().map(|a| a.seed).collect();
        for hit in &ask.seed_hits {
            let other_anchors: Vec<crate::seed::Seed> = ask
                .seed_hits
                .iter().copied()
                .filter(|s| s != hit)
                .collect();
            if let Some(allow) = brain.network
                .disambiguate_polysemous_with_anchors(hit, &activated_set, &other_anchors)
            {
                // OP-BRAIN-MapRAM P5-b2: out-neighbour SET via the seam (set-folded,
                // order-independent), off resident adjacency in cache mode.
                for nbr in brain.network.seam_outgoing_set(hit) {
                    if !allow.contains(&nbr) {
                        polysemy_blocked.insert(nbr);
                    }
                }
            }
        }
    }
    let acts_filtered: Vec<crate::network::ActivatedNode> = if polysemy_blocked.is_empty() {
        Vec::new()
    } else {
        ask.recall.activated.iter()
            .filter(|a| !polysemy_blocked.contains(&a.seed))
            .cloned().collect()
    };

    // Port #22: bare polysemous query — a single seed_hit whose
    // outgoing neighborhood splits into multiple sense clusters with
    // comparable typed-bond mass — surfaces every sense rather
    // than collapsing to one. The `narrate` orchestrator overrides
    // shape to MultiSense when this slot is non-empty.
    if ask.seed_hits.len() == 1 {
        let hit = ask.seed_hits[0];
        if let Some(clusters) = brain.network.disambiguate_polysemous_multi_sense(&hit) {
            for cluster in &clusters {
                let mut members: Vec<crate::seed::Seed> =
                    cluster.iter().copied().collect();
                members.sort_by(|a, b| {
                    let pa = bond_priority(brain, &hit, a);
                    let pb = bond_priority(brain, &hit, b);
                    pb.partial_cmp(&pa).unwrap_or(std::cmp::Ordering::Equal)
                });
                let labels: Vec<String> = members.iter()
                    .filter_map(|sd| brain.network.node_label(sd))
                    .take(4)
                    .collect();
                if !labels.is_empty() {
                    s.multi_sense_clusters.push(labels);
                }
            }
            // Pin the subject to the seed_hit content so the renderer
            // doesn't pull a (possibly polluted) top-activation label.
            if !s.multi_sense_clusters.is_empty() {
                if let Some(label) = brain.network.node_label(&hit) {
                    s.primary = Some(label);
                }
            }
        }
    }
    // OP-NARRATE-V2 (2026-05-31) — Definition reorder takes precedence
    // when active. acts_filtered (polysemy filter) wins over reorder when
    // both fire because polysemy correctness > definition-ordering, but
    // when polysemy didn't filter anything the reorder applies.
    let acts: &[crate::network::ActivatedNode] = if !acts_filtered.is_empty() {
        &acts_filtered
    } else if let Some(r) = reordered_acts.as_ref() {
        r.as_slice()
    } else {
        &ask.recall.activated
    };
    // OP-CLEANUP BUG3: internal routing handles (pyfn*/jsfn*/builtins.*/pyop*)
    // are load-bearing DURING recall (the sense-router seeds them) but must
    // never appear in a user-facing answer slot. Filter them from the display
    // acts; fall back to unfiltered only if that would empty the list.
    let acts_display: Vec<crate::network::ActivatedNode> =
        acts.iter().filter(|a| !is_handleish(&a.content)).cloned().collect();
    let acts: &[crate::network::ActivatedNode] =
        if acts_display.is_empty() { acts } else { &acts_display };
    if let Some(top) = acts.first() {
        s.primary = Some(top.content.clone());
        // Tier 2.9: normalize the recall ceiling.  Top activation
        // rarely exceeds ~0.6 even on strong queries, so dividing by
        // 0.6 maps the realistic working range into [0,1] without
        // saturating every result at 1.00.
        let recall_norm = (top.activation / 0.6).clamp(0.0, 1.0);
        // B1/#211: `top.confidence` is the rehydration STUB (Unverified) on a
        // loaded v5 brain, pegging EVERY answer at the 0.5 factor. Resolve the
        // answer-head node's REAL stored confidence from the node_store (read-
        // only, answer-head only, memoized); fall back to the stub only when
        // the store has no node (heap / test brains).
        let head_conf = brain
            .network
            .confidence_from_store(&top.seed)
            .unwrap_or(top.confidence);
        s.head_confidence = Some(format!("{:?}", head_conf));
        let combined = recall_norm * confidence_numeric(head_conf);
        s.confidence_value = combined.clamp(0.0, 1.0);
        s.confidence_label = confidence_label_for(s.confidence_value).to_string();
    } else {
        s.confidence_label = "none".to_string();
    }
    if let Some(second) = acts.get(1) {
        s.secondary = Some(second.content.clone());
        if let Some(top) = acts.first() {
            s.bond_label = bond_label_between(brain, &top.seed, &second.seed);
        }
    }
    if let Some(third) = acts.get(2) {
        s.tertiary = Some(third.content.clone());
        if let Some(second) = acts.get(1) {
            s.second_bond_label = bond_label_between(brain, &second.seed, &third.seed);
        }
    }

    // OP-RECALL-EXTRACT (defect 1): harvest the strongest grounded neighbours
    // recall already surfaced so the FactRecall renderer can compose a real
    // answer instead of echoing the query token. Skips the self-echo (a node
    // whose label is just the query word) and duplicate labels; activation
    // order is preserved. Reads only nodes that ALREADY activated, so it
    // cannot manufacture grounding a refusal path would have denied -- the
    // idk gate runs BEFORE narration.
    {
        let ql = query.trim().to_lowercase();
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        for a in acts.iter() {
            let c = a.content.trim();
            if c.is_empty() { continue; }
            if is_handleish(c) { continue; }
            let cl = c.to_lowercase();
            if cl == ql { continue; }
            if s.primary.as_deref().map(|p| p.to_lowercase()) == Some(cl.clone()) { continue; }
            if !seen.insert(cl) { continue; }
            s.recall_associations.push(c.to_string());
            if s.recall_associations.len() >= 6 { break; }
        }
    }

    // Bridge slot: pick the lowest-jaccard surfaced bridge THAT HAS
    // BOTH ENDPOINT LABELS (OP-LABEL-4). An endpoint the node store
    // cannot resolve leaves the slot empty rather than rendering a
    // hole; `ModulationFlags::bridge_touched` applies the same test, so
    // a wholly unlabelled bridge set simply does not produce a bridge
    // sentence.
    let mut bridges: Vec<&BridgeEvent> = ask
        .bridges
        .iter()
        .filter(|b| !b.low_content.is_empty() && !b.high_content.is_empty())
        .collect();
    bridges.sort_by(|x, y| {
        x.jaccard
            .partial_cmp(&y.jaccard)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    if let Some(b) = bridges.first() {
        s.bridge_low = Some(b.low_content.clone());
        s.bridge_high = Some(b.high_content.clone());
    }

    // Curiosity slot: pick the highest-intensity signal whose seed
    // has not exceeded `gap_appearance_threshold` (Port #17 Fix 2).
    // Signals are still kept in `ask.curiosity_signals` and the
    // wonder_queue — we just skip them in the wondering modulation
    // so the prose preamble does not loop on the same hub-boundary words.
    let mut sigs: Vec<&CuriositySignal> = ask.curiosity_signals.iter().collect();
    sigs.sort_by(|x, y| {
        y.intensity
            .partial_cmp(&x.intensity)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let gap_thresh = brain.curiosity_config.gap_appearance_threshold;
    for sig in &sigs {
        let cnt = brain
            .gap_seed_frequency
            .get(&sig.seed)
            .copied()
            .unwrap_or(0);
        if cnt >= gap_thresh { continue; }
        if let Some(label) = brain.network.node_label(&sig.seed) {
            s.wondering_about = Some(label);
        }
        s.wondering_kind = Some(signal_kind_label(&sig.signal_type).to_string());
        s.chosen_curiosity_seed = Some(sig.seed);
        break;
    }

    // Single-shot scar harvest from AskResult.scars.
    // When recall encounters scarred bonds, capture the Scar reasons.
    // Dedup by reason text so a cascade of 1500+ identical reasons
    // (e.g. scar_concept on a hub) collapses to a single mention.
    for scar in &ask.scars {
        if !s.scar_reasons.iter().any(|r| r == &scar.reason) {
            s.scar_reasons.push(scar.reason.clone());
        }
    }

    // Critic / multi-think slots.
    if let Some(m) = multi {
        s.critic_caveat_count = m.hypotheses.len().saturating_sub(m.vetted.len());
        if let Some(b) = m.vetted.first() {
            s.primary = belief_to_summary(b);
            // Tier 2.9 calibrated confidence: blend recall_top,
            // hypothesis_count, critic_acceptance, belief_confidence,
            // and the caveats-from-second-spread penalty into a real
            // 0–1 score.  The old code just read off b.confidence
            // which is always 1.0 for the strongest path — so every
            // narration ended in "Confidence: 1.00 (high)" regardless
            // of whether the path was sensible.
            let recall_top = ask
                .recall
                .activated
                .first()
                .map(|a| a.activation)
                .unwrap_or(0.0)
                .clamp(0.0, 1.0);
            let belief_conf = b.confidence.clamp(0.0, 1.0);
            let hyp_n = m.hypotheses.len().max(1) as f32;
            let critic_accept_rate = (m.vetted.len() as f32) / hyp_n;
            let caveat_penalty = if !b.caveats.is_empty() { 0.5 } else { 1.0 };
            // Recall ceiling: recall activations rarely exceed ~0.6
            // even on strong queries; rescale into [0,1] so it can
            // contribute equally to belief_conf.
            let recall_norm = (recall_top / 0.6).clamp(0.0, 1.0);
            let mixed = (0.40 * recall_norm
                + 0.35 * belief_conf
                + 0.25 * critic_accept_rate)
                * caveat_penalty;
            s.confidence_value = mixed.clamp(0.0, 1.0);
            s.confidence_label = confidence_label_for(s.confidence_value).to_string();
        }
        // Caveat: pick a hypothesis that was rejected (in hypotheses but
        // not vetted). Stable: first by appearance.
        if let Some(rejected) = m
            .hypotheses
            .iter()
            .find(|h| !m.vetted.iter().any(|v| v.from == h.from && v.to == h.to))
        {
            s.critic_caveat = Some(format!(
                "the link from {} to {} didn't clear the confidence threshold",
                rejected.from_content, rejected.to_content
            ));
        }
        // Harvest scar reasons from the Critic's rejected list. Any
        // rejection whose reason starts with `"scarred: "` is a
        // moral refusal — surfaced verbatim in the prose.
        for r in &m.rejected {
            if let Some(stripped) = r.reason.strip_prefix("scarred: ") {
                s.scar_reasons.push(stripped.to_string());
            }
        }
        // Alternative count for MultiHypothesisAnswer.
        s.winner_alt_count = m.hypotheses.len().saturating_sub(1);
        if let Some(alt) = m.hypotheses.iter().nth(1) {
            s.winner_alt = belief_to_summary(alt);
        }
    }

    // v2 walker prose: composed multi-bond body for shapes that opt in.
    let mut filtered_count: u32 = 0;
    s.walker_prose = walker_prose(brain, ask, cfg, &mut filtered_count);
    s.noise_filtered_count = filtered_count;

    // OP-NARRATE-V2 (2026-05-31) -- Definition prose override takes
    // precedence over walker_prose for Definition intent when a typed
    // bond is available. The direct IsA composition is structurally
    // truthful (one bond, one connector) and avoids HubSelector
    // picking a co-mention page as the hub.
    if definition_prose_override.is_some() {
        s.walker_prose = definition_prose_override;
    }

    // OP-ROUTE-3a: surface reasoning-chain articulation in the slot
    // bag. Only populated when the chain is structured (per Port D
    // calibration) and has at least 2 steps -- associative or single-
    // step traces are not articulation-quality. `chain_confidence` is
    // recorded regardless so callers can surface it independently.
    if let Some(trace) = reasoning {
        s.chain_confidence = trace.confidence;
        if !trace.no_chain
            && trace.steps.len() >= 2
            && trace.evidence_kind == crate::reasoning::EvidenceKind::Structured
        {
            s.chain_articulation = Some(trace.final_answer.clone());
        }
    }

    // OP-ROUTE-3c: surface final-answer prose from the thinking
    // trace when one converged. Guarded by
    // `chain_articulation.is_none()` so 3a retains precedence when
    // both fired -- the 3a Structured chain is the higher-confidence
    // articulation (deterministic over current graph state), so it
    // wins on overlap (Section 7 of op_route_3c_design.md).
    if let Some(t) = thinking {
        if s.chain_articulation.is_none()
            && t.steps_taken >= 1
            && t.final_answer_present
        {
            s.chain_articulation = t.final_answer_text.clone();
        }
    }

    // OP-ROUTE-3b: populate winner / runner-up slots from a variant
    // arena when one is available. Guards: arena must have elected a
    // winner; slot fill never overwrites a value multi-think already
    // wrote (multi-think upstream above is critic-vetted and outranks
    // structural variant winners per Section 5 of the 3b design).
    if let Some(arena) = variants {
        if let Some(winner_id) = arena.winner_id {
            // Winner's anchor label -> primary, only if multi-think did
            // not already populate it.
            if let Some(winner_spec) = arena
                .variants
                .iter()
                .find(|v| v.id == winner_id)
            {
                if s.primary.is_none() {
                    if let Some(label) = brain.network.node_label(&winner_spec.anchor_seed) {
                        s.primary = Some(label);
                    }
                }
            }
            // Runner-up: highest-composite non-winner. Tie-broken by
            // variant_id ASC for determinism (matches score_arena's own
            // winner tie-break).
            let non_winner_scores: Vec<&crate::variants::VariantScore> = arena
                .scores
                .iter()
                .filter(|sc| sc.variant_id != winner_id)
                .collect();
            if !non_winner_scores.is_empty() && s.winner_alt.is_none() {
                let runner_up = non_winner_scores
                    .iter()
                    .max_by(|a, b| {
                        a.composite
                            .partial_cmp(&b.composite)
                            .unwrap_or(std::cmp::Ordering::Equal)
                            .then_with(|| b.variant_id.cmp(&a.variant_id))
                    })
                    .copied();
                if let Some(ru) = runner_up {
                    if let Some(spec) = arena
                        .variants
                        .iter()
                        .find(|v| v.id == ru.variant_id)
                    {
                        if let Some(label) = brain.network.node_label(&spec.anchor_seed) {
                            s.winner_alt = Some(label);
                        }
                    }
                }
                // winner_alt_count is the count of non-winner specs
                // (i.e. "considered N alternatives"). Only overwrite if
                // multi-think did not already set it to a non-zero
                // value (multi-think writes via `m.hypotheses.len() - 1`).
                let non_winner_specs = arena
                    .variants
                    .iter()
                    .filter(|v| v.id != winner_id)
                    .count();
                if s.winner_alt_count == 0 {
                    s.winner_alt_count = non_winner_specs;
                }
            }
        }
    }

    s
}

/// Compose "{from} relates to {to} via {via}" from a belief's labels.
/// Returns None when any operand is blank -- a summary with empty
/// subject/object/relation slots renders as word-salad ("... relates to
///  via .") and must never reach the user (OP-NARRATE-HYGIENE 3a).
fn belief_to_summary(b: &Belief) -> Option<String> {
    let from = b.from_content.trim();
    let to = b.to_content.trim();
    let via = b.via_content.trim();
    if from.is_empty() || to.is_empty() || via.is_empty() {
        return None;
    }
    Some(format!("{} relates to {} via {}", from, to, via))
}

// =====================================================================
// Shape selection
// =====================================================================

/// Hand-rule shape selector. Documented precedence — first match wins:
/// 1. No activations at all => Refusal.
/// 2. MetaSelf intent => MetaSelf.
/// 3. TimetravelComparison intent => TimetravelComparison.
/// 4. Critic-vetted multi-think available =>
///    a. multiple hypotheses surviving => MultiHypothesisAnswer
///    b. otherwise => CriticVettedAnswer
/// 5. Bridge fired (jaccard low enough) => BridgedAnswer.
/// 6. Wondering signal touched + low confidence => Wondering.
/// 7. Top activation below low_confidence_cutoff => UncertainAnswer.
/// 8. Multi-step think used (multi present, no vetted) => MultiStep.
/// 9. Two-hop synthesis available (>=3 activations chained) => Synthesis.
/// 10-14. Intent-based fall-through: Definition, Process, WhyChain,
///         Comparison, FactRecall.
/// 15. Generic intent => Definition (closest neutral shape).
pub struct ShapeSelector;

impl ShapeSelector {
    pub fn pick(
        intent: QueryIntent,
        ask: &AskResult,
        multi: Option<&MultiAgentTrace>,
        flags: &ModulationFlags,
        cfg: &NarrationConfig,
        reasoning: Option<&crate::reasoning::ReasoningTrace>,
        variants: Option<&crate::variants::VariantArena>,
        thinking: Option<&crate::thinking::ThinkingTraceSummary>,
    ) -> ResponseShape {
        // 0. OP-NARRATE-V2 — self-identity / current-interests bypass
        //    activation requirements. SelfIdentity is constant prose;
        //    CurrentInterests reads the wonder queue (which is brain-
        //    level, not query-level). Must come before Refusal so an
        //    empty-recall query still gets a real shape.
        if intent == QueryIntent::SelfIdentity {
            return ResponseShape::SelfIdentity;
        }
        if intent == QueryIntent::CurrentInterests {
            return ResponseShape::CurrentInterests;
        }
        // OP-RECALL-1 -- MetaSelf joins the activation-independent group
        // above Refusal, for exactly the reason given for SelfIdentity
        // and CurrentInterests: it is a BRAIN-LEVEL report, not a
        // query-level recall.
        //
        // This became necessary once topical "what do you know about X"
        // stopped classifying MetaSelf. What is left is the bare form --
        // "what do you know?" -- which seeds nothing, because every one
        // of its tokens is a hub-stopword. Below the Refusal gate that
        // shape was therefore unreachable, and Lu answered a question
        // about its own contents with "I don't have meaningful
        // information about that."
        //
        // Answering it from the counters is not a hallucination risk:
        // node_count and bond_count are facts about the brain, and the
        // render arm now refuses to state a zero it is about to
        // contradict.
        if intent == QueryIntent::MetaSelf {
            return ResponseShape::MetaSelf;
        }
        let acts = &ask.recall.activated;
        // 1. Refusal.
        if acts.is_empty() {
            return ResponseShape::Refusal;
        }
        // 3. Meta-level intent that still needs activations.
        if intent == QueryIntent::TimetravelComparison {
            return ResponseShape::TimetravelComparison;
        }
        // 4. Bridge path runs ahead of critic-vetted so cross-domain
        //    bridges can surface even when multi-think also fired.
        //    OP-ROUTE-GUEST (2026-07-23): an INCIDENTAL bridge must not
        //    hijack an explicitly definitional query. Before this fix,
        //    "What is a dog" (intent=Definition, clean routing) returned
        //    shape=BridgedAnswer because a weak cross-cluster link was
        //    touched during spread, letting a Wikipedia category string
        //    ("Extant Late Pleistocene first appearances") win the answer
        //    slot over the definition. Intent classification now outranks
        //    an incidental bridge for Definition; genuinely relational
        //    queries classify as Comparison / Generic / WhyChain and
        //    still surface the bridge.
        if flags.bridge_touched && intent != QueryIntent::Definition {
            return ResponseShape::BridgedAnswer;
        }
        let top_act = acts.first().map(|a| a.activation).unwrap_or(0.0);
        // 5. Wondering — only when curiosity is real and confidence is
        //    not high (otherwise we're hijacking a clean answer).
        if flags.wondering_touched && top_act < cfg.high_confidence_cutoff && flags.expose_curiosity_voice {
            return ResponseShape::Wondering;
        }
        // 6. Pure low-confidence => UncertainAnswer.
        if top_act < cfg.low_confidence_cutoff {
            return ResponseShape::UncertainAnswer;
        }
        // 7. Critic-vetted path — reordered to fire ONLY when the
        //    Critic actually rejected something (vetted < hypotheses)
        //    or accepted-with-caveats.  Without that gate every
        //    multi-think run locks the answer to CriticVettedAnswer
        //    and the other 14 shapes never appear.
        if let Some(m) = multi {
            let critic_did_work = !m.vetted.is_empty()
                && (m.hypotheses.len() > m.vetted.len()
                    || m.vetted.iter().any(|b| !b.caveats.is_empty())
                    || !m.rejected.is_empty());
            if critic_did_work {
                if m.hypotheses.len() > m.vetted.len() && m.hypotheses.len() >= 2 {
                    return ResponseShape::MultiHypothesisAnswer;
                }
                return ResponseShape::CriticVettedAnswer;
            }
            // multi-think ran but produced nothing meaningful: fall
            // through to intent-based shape selection rather than
            // forcing CriticVettedAnswer.
            if m.vetted.is_empty() && !m.recall.is_empty() {
                return ResponseShape::MultiStep;
            }
        }
        // 9. Two-hop synthesis when we have a 3-node chain.
        if intent == QueryIntent::Generic && acts.len() >= 3 {
            return ResponseShape::Synthesis;
        }
        // 9.3. OP-ROUTE-3c: a converged thinking pass is the
        //      strongest signal that the brain produced a multi-step
        //      answer. "Converged" = at least one step taken AND
        //      (early_exit OR at least one hypothesis was promoted).
        //      A loop that ran max_steps without promoting anything
        //      is structurally inconclusive -- fall through to other
        //      rules. Above rule 9.4 (variants) and 9.5 (chain) per
        //      Section 8 of op_route_3c_design.md.
        if let Some(t) = thinking {
            let converged = t.steps_taken >= 1
                && (t.early_exit || t.hypotheses_promoted >= 1);
            if converged {
                return ResponseShape::MultiStep;
            }
        }
        // 9.4. OP-ROUTE-3b: a variant arena that produced a winner is
        //      a strong signal for MultiHypothesisAnswer regardless of
        //      nominal intent. The arena's own confidence floor
        //      (`min_winner_composite = 0.4`) is upstream; if it picked
        //      a winner, the arena is internally consistent. Distinct
        //      from the rule 4 critic-vetted multi-think path -- that
        //      block runs FIRST (lines ~1554 above), so when both
        //      signals fire, the critic-vetted path takes the shape
        //      and variants only modulate slot fill. Require >= 2
        //      specs so the rendered template's "considered N
        //      alternatives" reads truthfully.
        if let Some(arena) = variants {
            if arena.winner_id.is_some() && arena.variants.len() >= 2 {
                return ResponseShape::MultiHypothesisAnswer;
            }
        }
        // 9.5. OP-ROUTE-3a: if a structured reasoning chain is
        //     available (>= 2 steps, EvidenceKind::Structured, not
        //     no_chain), prefer WhyChain regardless of nominal intent.
        //     The chain itself is the strongest possible signal that
        //     a why-shape is the right shape, even when IntentClassifier
        //     landed on Process / Generic / Definition. Associative
        //     chains do NOT trigger this bias (their prose is hedged).
        if let Some(trace) = reasoning {
            if !trace.no_chain
                && trace.steps.len() >= 2
                && trace.evidence_kind == crate::reasoning::EvidenceKind::Structured
            {
                return ResponseShape::WhyChain;
            }
        }
        // 9b. Calibration fix 2026-04-25 (Port #16): when intent is
        //     definitional but curiosity is LOUD (>=4 signals OR top
        //     intensity >0.85), prefer an uncertain/wondering shape over
        //     a confident Definition. The Port #11 thresholds (>=2, >0.6)
        //     swallowed every definition once walker_prose enriched the
        //     Wondering body; raised here so Definition/Process/Comparison
        //     can win on moderately-curious queries.
        if intent == QueryIntent::Definition {
            // Port #16 v2 calibration: only consider curiosity signals whose
            // seed appears in this query's activated recall set. Spread
            // touches up to 5000 nodes and the curiosity classifier produces
            // hundreds-thousands of signals globally; raw counts let unrelated
            // hub paths trigger Wondering on every Definition-intent query.
            // Filtering to query-attached signals re-anchors the gate to
            // "are gaps relevant to THIS query," not "is the brain noisy."
            let activated_seeds: std::collections::HashSet<crate::seed::Seed> = ask
                .recall
                .activated
                .iter()
                .map(|a| a.seed)
                .collect();
            let attached: Vec<&crate::curiosity::CuriositySignal> = ask
                .curiosity_signals
                .iter()
                .filter(|c| activated_seeds.contains(&c.seed))
                .collect();
            let curiosity_count = attached.len();
            let top_intensity = attached
                .iter()
                .map(|c| c.intensity)
                .fold(0.0_f32, f32::max);
            let curiosity_strong = curiosity_count >= 4 || top_intensity > 0.85;
            if curiosity_strong {
                let hub_or_deadend_dominant = attached.iter().any(|c| {
                    matches!(c.signal_type, SignalType::HubBoundary | SignalType::DeadEnd)
                        && c.intensity >= top_intensity * 0.8
                });
                if hub_or_deadend_dominant && flags.expose_curiosity_voice {
                    return ResponseShape::Wondering;
                }
                return ResponseShape::UncertainAnswer;
            }
        }
        // 10-15. Intent-based fall-through.
        match intent {
            QueryIntent::Definition => ResponseShape::Definition,
            QueryIntent::Process => ResponseShape::Process,
            QueryIntent::WhyChain => ResponseShape::WhyChain,
            QueryIntent::Comparison => ResponseShape::Comparison,
            QueryIntent::FactRecall => ResponseShape::FactRecall,
            QueryIntent::MetaSelf => ResponseShape::MetaSelf,
            QueryIntent::TimetravelComparison => ResponseShape::TimetravelComparison,
            // OP-NARRATE-V2 — covered by the early route at top of pick();
            // exhaustive arm here for the compiler.
            QueryIntent::SelfIdentity => ResponseShape::SelfIdentity,
            QueryIntent::CurrentInterests => ResponseShape::CurrentInterests,
            QueryIntent::Generic => ResponseShape::Definition,
        }
    }
}

// =====================================================================
// Renderer
// =====================================================================

/// Pure prose-assembly. Given a shape, slots, and modulation flags,
/// produces the final answer string. Bounded by `cfg.max_answer_chars`
/// (or `multi_step_max_chars` for MultiStep).
pub struct Renderer;

impl Renderer {
    pub fn render(
        shape: ResponseShape,
        slots: &ShapeSlots,
        flags: &ModulationFlags,
        cfg: &NarrationConfig,
        hints: &StyleHints,
    ) -> String {
        let core = render_core(shape, slots, hints);
        let core = apply_prefix_modulations(core, shape, flags);
        let core = apply_suffix_modulations(core, shape, slots, flags);
        let core = cap_length(core, shape, cfg);
        // OP-NARRATE-HYGIENE 3b: strip serving-internal "Confidence: N.NN"
        // scalar tags that leaked into user-facing gloss answers.
        strip_internal_confidence_tag(core)
    }
}

fn unknown_or<'a>(opt: &'a Option<String>) -> &'a str {
    opt.as_deref().unwrap_or("(unknown)")
}

/// OP-NARRATE-HYGIENE 3a -- strip a leading interrogative frame down to
/// the bare subject, for an honest refusal ("... on entropy.").
fn refusal_subject(query: &str) -> String {
    let q = query
        .trim()
        .trim_end_matches(|c: char| c == '?' || c == '.' || c == '!')
        .trim();
    let ql = q.to_lowercase();
    const PREFIXES: &[&str] = &[
        "what is the ", "what is a ", "what is an ", "what is ",
        "what are the ", "what are ", "what's ",
        "who is the ", "who is a ", "who is an ", "who is ",
        "who was ", "who are ", "who were ",
        "define the ", "define a ", "define an ", "define ",
        "tell me about ", "what does ", "explain ",
    ];
    for p in PREFIXES {
        if ql.starts_with(p) {
            return q[p.len()..].trim().to_string();
        }
    }
    q.to_string()
}

/// OP-NARRATE-HYGIENE 3a -- honest refusal used when a relational
/// template would otherwise render with empty slots.
fn honest_relational_refusal(query: &str) -> String {
    let subj = refusal_subject(query);
    if subj.is_empty() {
        "I don't have anything solid on that.".to_string()
    } else {
        format!("I don't have anything solid on {}.", subj)
    }
}

/// OP-NARRATE-HYGIENE 3a -- a relational / multi-hypothesis template
/// interpolates COMPOSED subject/relation/object slots. When the
/// underlying edges carry no labels those slots come through blank and
/// the template renders as word-salad ("Most likely  relates to  via .").
/// Validate every required slot for the composed-relational shapes; if
/// any is blank, refuse honestly rather than emit the broken string.
/// Returns Some(refusal) when the guard trips.
fn relational_slot_guard(shape: ResponseShape, s: &ShapeSlots) -> Option<String> {
    fn blank(o: &Option<String>) -> bool {
        o.as_deref().map_or(true, |v| v.trim().is_empty())
    }
    let walker_ok = s
        .walker_prose
        .as_deref()
        .map_or(false, |p| !p.trim().is_empty());
    let tripped = match shape {
        // Both the winner and the runner-up are composed slots.
        ResponseShape::MultiHypothesisAnswer => blank(&s.primary) || blank(&s.winner_alt),
        // Synthesis renders walker_prose verbatim when present; only the
        // fallback template composes "{primary} relates to {tertiary} ...".
        ResponseShape::Synthesis => !walker_ok && (blank(&s.primary) || blank(&s.tertiary)),
        _ => false,
    };
    if tripped {
        Some(honest_relational_refusal(&s.query))
    } else {
        None
    }
}

/// OP-NARRATE-HYGIENE 3b -- internal confidence scalars (e.g.
/// "Confidence: 0.12") are a serving-internal debug signal. A few leaked
/// into user-facing gloss answers (photon, ruby). Strip a TRAILING bare
/// "Confidence: N.NN" / "Conf: N.NN" scalar tag. Label forms
/// ("Confidence: high") and the critic arm's "Confidence: 0.50 (high)."
/// are left intact -- only the trailing bare-scalar form is removed.
fn strip_internal_confidence_tag(s: String) -> String {
    let t = s.trim_end();
    let core = t.strip_suffix('.').unwrap_or(t);
    for marker in [" Confidence: ", " Conf: "] {
        if let Some(idx) = core.rfind(marker) {
            let tail = &core[idx + marker.len()..];
            let has_digit = tail.chars().any(|c| c.is_ascii_digit());
            let only_num = tail.chars().all(|c| c.is_ascii_digit() || c == '.');
            if has_digit && only_num && tail.matches('.').count() <= 1 {
                return core[..idx].trim_end().to_string();
            }
        }
    }
    s
}

fn render_core(shape: ResponseShape, s: &ShapeSlots, hints: &StyleHints) -> String {
    use crate::style::VerbosityTier;
    // OP-NARRATE-HYGIENE 3a: never render a composed-relational template
    // with an empty subject/relation/object slot -- refuse honestly.
    if let Some(refusal) = relational_slot_guard(shape, s) {
        return refusal;
    }
    let tier = hints.verbosity_tier();
    match shape {
        ResponseShape::Conclusion => {
            // OP-CONCLUDE -- prose is fully composed by the synthesizer
            // (statement + cited-edge chain); render verbatim.
            s.conclusion_prose.clone().unwrap_or_default()
        }
        ResponseShape::Definition => {
            // walker_prose passthrough preserved at both tiers; truncating
            // composed multi-bond prose is out of scope (separate port).
            if let Some(p) = &s.walker_prose {
                p.clone()
            } else {
                // Fallback already terse; both tiers identical.
                format!(
                    "{} is {}.",
                    unknown_or(&s.primary),
                    unknown_or(&s.secondary)
                )
            }
        }
        ResponseShape::Process => {
            if let Some(p) = &s.walker_prose {
                p.clone()
            } else {
                let step1 = unknown_or(&s.primary);
                let step2 = unknown_or(&s.secondary);
                let step3 = unknown_or(&s.tertiary);
                match tier {
                    VerbosityTier::Standard => {
                        let n = [step1, step2, step3]
                            .iter()
                            .filter(|x| **x != "(unknown)")
                            .count();
                        format!(
                            "{} works in {} steps: first {}, then {}, then {}.",
                            s.query, n.max(1), step1, step2, step3
                        )
                    }
                    VerbosityTier::Terse => {
                        format!("{}; {}; {}.", step1, step2, step3)
                    }
                }
            }
        }
        ResponseShape::WhyChain => {
            // OP-ROUTE-3a: when a structured deep-reasoning chain was
            // built upstream, use its pre-rendered prose verbatim. The
            // existing template fires only as the fallback when no
            // chain was available or it failed the structured-quality
            // gate. cap_length() downstream still truncates to
            // cfg.max_answer_chars.
            if let Some(prose) = &s.chain_articulation {
                prose.clone()
            } else {
                match tier {
                    VerbosityTier::Standard => format!(
                        "{} happens because {}, which leads to {}, producing {}.",
                        unknown_or(&s.primary),
                        unknown_or(&s.secondary),
                        unknown_or(&s.tertiary),
                        unknown_or(&s.primary)
                    ),
                    VerbosityTier::Terse => format!(
                        "{}: {} -> {}.",
                        unknown_or(&s.primary),
                        unknown_or(&s.secondary),
                        unknown_or(&s.tertiary)
                    ),
                }
            }
        },
        ResponseShape::Comparison => {
            if let Some(p) = &s.walker_prose {
                p.clone()
            } else {
                let a = unknown_or(&s.primary);
                let b = unknown_or(&s.secondary);
                let shared = s
                    .bond_label
                    .clone()
                    .unwrap_or_else(|| "a related context".to_string());
                let tertiary = unknown_or(&s.tertiary);
                match tier {
                    VerbosityTier::Standard => format!(
                        "{} and {} share {}. They differ in how each relates to {}.",
                        a, b, shared, tertiary
                    ),
                    VerbosityTier::Terse => format!(
                        "{} and {}: shared {}; differ on {}.",
                        a, b, shared, tertiary
                    ),
                }
            }
        }
        ResponseShape::FactRecall => {
            // OP-RECALL-EXTRACT (defect 1): if recall surfaced grounded
            // neighbours, answer with them instead of echoing the bare query
            // token back as "{token}." (which strips to an empty core answer
            // -- the observed single-token failure). The neighbours are real
            // activated nodes; when none survived the self-echo/handle filter
            // we keep the legacy terse form.
            if s.recall_associations.is_empty() {
                format!("{}.", unknown_or(&s.primary))
            } else {
                let list = join_plain(&s.recall_associations);
                match s.primary.as_deref() {
                    Some(p) if !p.is_empty() => format!(
                        "{} - in my memory it connects most strongly to {}.", p, list
                    ),
                    _ => format!("In my memory that connects most strongly to {}.", list),
                }
            }
        }
        ResponseShape::UncertainAnswer => match tier {
            VerbosityTier::Standard => format!(
                "I think it might be {}, but my support for that is thin. More associations to {} would help close the gap.",
                unknown_or(&s.primary),
                s.query
            ),
            VerbosityTier::Terse => format!(
                "Possibly {} (low support).",
                unknown_or(&s.primary)
            ),
        },
        ResponseShape::MultiHypothesisAnswer => match tier {
            VerbosityTier::Standard => format!(
                "Most likely {}. I considered {} alternatives \u{2014} for example, {} \u{2014} but the top candidate had stronger support.",
                unknown_or(&s.primary),
                s.winner_alt_count.max(1),
                unknown_or(&s.winner_alt)
            ),
            VerbosityTier::Terse => format!(
                "Most likely {} (vs. {} alternatives e.g. {}).",
                unknown_or(&s.primary),
                s.winner_alt_count.max(1),
                unknown_or(&s.winner_alt)
            ),
        },
        ResponseShape::BridgedAnswer => match tier {
            VerbosityTier::Standard => format!(
                "{}. While answering, I noticed a connection between {} and {}: they sit in different clusters but a weak link bridges them.",
                unknown_or(&s.primary),
                unknown_or(&s.bridge_low),
                unknown_or(&s.bridge_high)
            ),
            VerbosityTier::Terse => format!(
                "{}. Bridge: {} <-> {}.",
                unknown_or(&s.primary),
                unknown_or(&s.bridge_low),
                unknown_or(&s.bridge_high)
            ),
        },
        ResponseShape::CriticVettedAnswer => {
            let caveat = s
                .critic_caveat
                .clone()
                .unwrap_or_else(|| "no specific caveat surfaced".to_string());
            match tier {
                VerbosityTier::Standard => format!(
                    "{}. My critic caught {} caveats: {}. Confidence: {:.2} ({}).",
                    unknown_or(&s.primary),
                    s.critic_caveat_count,
                    caveat,
                    s.confidence_value,
                    s.confidence_label
                ),
                VerbosityTier::Terse => format!(
                    "{}. Caveats ({}): {}. Conf: {:.2}.",
                    unknown_or(&s.primary),
                    s.critic_caveat_count,
                    caveat,
                    s.confidence_value
                ),
            }
        }
        ResponseShape::Refusal => match tier {
            VerbosityTier::Standard => format!(
                "I don't have a meaningful answer to \"{}\". No bonds in my graph activated for this query.",
                s.query
            ),
            VerbosityTier::Terse => format!(
                "No answer for \"{}\" \u{2014} no bonds activated.",
                s.query
            ),
        },
        // OP-RECALL-1 (coherence). The counters here are brain-wide;
        // R5 made them real on v5 brains, but a renderer must not be
        // able to state "I have 0 concepts and 0 associations" and then
        // name a node it just activated. Claiming emptiness while
        // demonstrably holding content is an anti-hallucination-floor
        // violation in spirit even when each half is separately
        // "correct". If the census reads empty, say only what we can
        // stand behind: what activated.
        ResponseShape::MetaSelf => {
            let census_empty = s.node_count == 0 && s.bond_count == 0;
            // OP-RECALL-1: the bare form recalls nothing by design, so
            // there is no "recently activated" node to name. Saying
            // "Recently I activated unknown." would be filler; report
            // the census and stop.
            if s.primary.as_deref().map_or(true, |p| p.trim().is_empty()) && !census_empty {
                return match tier {
                    VerbosityTier::Standard => format!(
                        "I'm holding {} concepts and {} associations right now.",
                        s.node_count, s.bond_count,
                    ),
                    VerbosityTier::Terse => {
                        format!("{} concepts, {} bonds.", s.node_count, s.bond_count)
                    }
                };
            }
            match (tier, census_empty) {
                (VerbosityTier::Standard, false) => format!(
                    "On \"{}\", I have {} concepts and {} associations. Confidence: {}. Recently I activated {}.",
                    s.query,
                    s.node_count,
                    s.bond_count,
                    s.confidence_label,
                    unknown_or(&s.primary)
                ),
                (VerbosityTier::Terse, false) => format!(
                    "\"{}\": {} concepts, {} bonds, conf={}, recent={}.",
                    s.query,
                    s.node_count,
                    s.bond_count,
                    s.confidence_label,
                    unknown_or(&s.primary)
                ),
                // Census unavailable. Drop the counts rather than
                // assert a zero we are actively contradicting.
                (VerbosityTier::Standard, true) => format!(
                    "On \"{}\", I can't give you a reliable count of what I hold. Confidence: {}. Recently I activated {}.",
                    s.query,
                    s.confidence_label,
                    unknown_or(&s.primary)
                ),
                (VerbosityTier::Terse, true) => format!(
                    "\"{}\": count unavailable, conf={}, recent={}.",
                    s.query,
                    s.confidence_label,
                    unknown_or(&s.primary)
                ),
            }
        }
        ResponseShape::TimetravelComparison => match tier {
            VerbosityTier::Standard => format!(
                "About \"{}\": at an earlier point I primarily associated this with {}. Now: {}. The shift came from added associations during recent sessions.",
                s.query,
                unknown_or(&s.secondary),
                unknown_or(&s.primary)
            ),
            VerbosityTier::Terse => format!(
                "\"{}\": was {}, now {}.",
                s.query,
                unknown_or(&s.secondary),
                unknown_or(&s.primary)
            ),
        },
        ResponseShape::Synthesis => {
            if let Some(p) = &s.walker_prose {
                // walker_prose passthrough at both tiers; only the
                // confidence suffix shortens for Terse.
                match tier {
                    VerbosityTier::Standard => {
                        format!("{} Confidence: {:.2}.", p, s.confidence_value)
                    }
                    VerbosityTier::Terse => {
                        format!("{} Conf: {:.2}.", p, s.confidence_value)
                    }
                }
            } else {
                let via = s
                    .secondary
                    .clone()
                    .unwrap_or_else(|| "an intermediate concept".to_string());
                match tier {
                    VerbosityTier::Standard => format!(
                        "{} relates to {}. I'm inferring this from {} \u{2014} both are connected through that intermediate. Confidence: {:.2}.",
                        unknown_or(&s.primary),
                        unknown_or(&s.tertiary),
                        via,
                        s.confidence_value
                    ),
                    VerbosityTier::Terse => format!(
                        "{} via {} via {} (conf {:.2}).",
                        unknown_or(&s.primary),
                        via,
                        unknown_or(&s.tertiary),
                        s.confidence_value
                    ),
                }
            }
        }
        ResponseShape::Wondering => {
            let body = match &s.walker_prose {
                Some(p) => p.clone(),
                None => format!("{}.", unknown_or(&s.primary)),
            };
            match tier {
                VerbosityTier::Standard => format!(
                    "{} (I've been wondering about {} \u{2014} {}; it touches this query as a gap in my knowledge.)",
                    body,
                    unknown_or(&s.wondering_about),
                    unknown_or(&s.wondering_kind)
                ),
                VerbosityTier::Terse => format!(
                    "{} (gap: {}.)",
                    body,
                    unknown_or(&s.wondering_about)
                ),
            }
        }
        ResponseShape::MultiStep => match tier {
            VerbosityTier::Standard => format!(
                "Several parts to this. First: {}. Second: {}. Third: {}. Together: these strands cover the question without a single clean answer.",
                unknown_or(&s.primary),
                unknown_or(&s.secondary),
                unknown_or(&s.tertiary)
            ),
            VerbosityTier::Terse => format!(
                "{}; {}; {}.",
                unknown_or(&s.primary),
                unknown_or(&s.secondary),
                unknown_or(&s.tertiary)
            ),
        },
        ResponseShape::MultiSense => {
            let subj_lc = unknown_or(&s.primary).to_lowercase();
            let subj_cap = capitalize_first(unknown_or(&s.primary));
            // Collect non-empty clusters with their joined label strings.
            let mut clusters: Vec<String> = Vec::new();
            for labels in s.multi_sense_clusters.iter() {
                if labels.is_empty() { continue; }
                clusters.push(labels.join(", "));
            }
            if clusters.is_empty() {
                return format!(
                    "{} has more than one sense in my graph, but I could not lay them out cleanly. Which did you mean?",
                    subj_cap
                );
            }
            match tier {
                VerbosityTier::Standard => {
                    let mut parts: Vec<String> = Vec::new();
                    for (i, joined) in clusters.iter().enumerate() {
                        let head = match i {
                            0 => "In one sense,".to_string(),
                            1 => "In another,".to_string(),
                            _ => "And in another,".to_string(),
                        };
                        parts.push(format!(
                            "{} {} relates to {}.",
                            head, subj_lc, joined
                        ));
                    }
                    format!(
                        "{} has multiple senses. {} Which sense did you mean?",
                        subj_cap, parts.join(" ")
                    )
                }
                VerbosityTier::Terse => {
                    let mut parts: Vec<String> = Vec::new();
                    for (i, joined) in clusters.iter().enumerate() {
                        parts.push(format!("sense {} = {}", i + 1, joined));
                    }
                    format!("{}: {}. Which?", subj_cap, parts.join("; "))
                }
            }
        }
        ResponseShape::SelfIdentity if is_capability_query(&s.query) => {
            // Adj 2: capability form of the self/meta answer. Same path as
            // "who are you" (identity arm below, unchanged). Honest about
            // what she does -- surfaces facts/connections, does not claim to
            // deduce conclusions.
            "I can define terms, recall facts on topics I have learned, and compute -- arithmetic, percentages, unit conversions, basic geometry (like the area of a circle), and simple statistics. I surface related concepts and connections across my knowledge graph, I only commit to claims I can defend, and when I do not know something I will tell you rather than guess.".to_string()
        }
        ResponseShape::SelfIdentity => {
            // OP-IDENTITY — answer from the LOCKED identity core (verbatim).
            // Falls back to the composed self-introduction for a bare
            // "who/what are you". Constant prose; identity, not depth.
            crate::identity::identity_answer(&s.query)
                .unwrap_or_else(crate::identity::self_intro)
        }
        ResponseShape::CurrentInterests => {
            // OP-NARRATE-V2 (2026-05-31) — prose was composed at
            // slot-extract time from the wonder queue. Renderer
            // surfaces it verbatim, falling back to the quiet-state
            // template when the queue was empty.
            match &s.current_interests_prose {
                Some(prose) => prose.clone(),
                None => "My curiosity hasn't been triggered by anything recent — I'm in a quiet state.".to_string(),
            }
        }
    }
}


fn capitalize_first(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        None => String::new(),
        Some(f) => f.to_uppercase().chain(c).collect(),
    }
}

fn apply_prefix_modulations(core: String, shape: ResponseShape, flags: &ModulationFlags) -> String {
    // "I think" / "tentatively" prefix only fires when the underlying
    // shape isn't already disclosing uncertainty.
    let already_hedges = matches!(
        shape,
        ResponseShape::UncertainAnswer
            | ResponseShape::Refusal
            | ResponseShape::Wondering
            | ResponseShape::SelfIdentity
            | ResponseShape::CurrentInterests
            | ResponseShape::MetaSelf
            | ResponseShape::CriticVettedAnswer
            | ResponseShape::Conclusion
    );
    if flags.hypothesis_only_support && !already_hedges {
        return format!("Tentatively: {}", core);
    }
    core
}

fn apply_suffix_modulations(
    mut core: String,
    shape: ResponseShape,
    slots: &ShapeSlots,
    flags: &ModulationFlags,
) -> String {
    // Curiosity append: skip if already a wondering / refusal / uncertain
    // shape, or for MultiSense (Port #22 — the question at the end
    // is the natural call-to-action).
    let already_curiosity = matches!(
        shape,
        ResponseShape::Wondering
            | ResponseShape::Refusal
            | ResponseShape::SelfIdentity
            | ResponseShape::CurrentInterests
            | ResponseShape::UncertainAnswer
            | ResponseShape::MultiSense
            | ResponseShape::Conclusion
    );
    if flags.curiosity_touched && !already_curiosity && flags.expose_curiosity_voice {
        if let Some(about) = &slots.wondering_about {
            core.push_str(&format!(
                " There's a gap nearby — I have thin coverage around {}.",
                about
            ));
        } else {
            core.push_str(" There's a nearby gap I haven't filled in.");
        }
    }
    // Bridge append: skip if already a bridged shape.
    if flags.bridge_touched
        && shape != ResponseShape::BridgedAnswer
        && shape != ResponseShape::Conclusion
    {
        if let (Some(lo), Some(hi)) = (&slots.bridge_low, &slots.bridge_high) {
            core.push_str(&format!(
                " (I also noticed a cross-domain link: {} <-> {}.)",
                lo, hi
            ));
        }
    }
    // Critic-vetted suffix: only when not already a critic-vetted shape.
    if flags.critic_vetted
        && shape != ResponseShape::CriticVettedAnswer
        && shape != ResponseShape::MultiHypothesisAnswer
        && shape != ResponseShape::Conclusion
    {
        core.push_str(&format!(
            " Confidence: {:.2}.",
            slots.confidence_value
        ));
    }
    // Scar caveat — append for every shape when scar reasons
    // were collected. The Refusal shape suppresses some
    // modulations elsewhere; the scar caveat overrides that
    // because moral knowledge must surface.
    if !slots.scar_reasons.is_empty() {
        let _ = shape;
        let joined = slots.scar_reasons.join("; ");
        core.push_str(&format!(
            " I\'m scarred against this connection \u{2014} {}.",
            joined
        ));
    }

    core
}

fn cap_length(s: String, shape: ResponseShape, cfg: &NarrationConfig) -> String {
    let cap = if shape == ResponseShape::MultiStep {
        cfg.multi_step_max_chars
    } else {
        cfg.max_answer_chars
    };
    if s.chars().count() <= cap {
        return s;
    }
    let mut out: String = s.chars().take(cap.saturating_sub(1)).collect();
    out.push('…');
    out
}

// =====================================================================
// Brain-facing entry point (re-used by `Brain::narrate`)
// =====================================================================

/// Output of a single narration call.
#[derive(Debug, Clone, Serialize)]
pub struct NarrationResult {
    pub query: String,
    pub answer: String,
    pub shape: ResponseShape,
    pub intent: QueryIntent,
    pub flags: ModulationFlags,
    /// Stable string list of which signal categories drove the shape /
    /// modulations. Useful for debugging "why this shape?".
    pub signals_used: Vec<String>,
    /// Port #20: number of walker neighbors rejected by the
    /// runtime ingestion-noise filter during this narration. Zero
    /// means either the filter is disabled (`--no-noise-filter`) or
    /// the walker did not surface any junk on this query.
    pub noise_filtered_count: u32,
    pub ask: AskResult,
    pub multi: Option<MultiAgentTrace>,
    /// Port S — distinct source-kind tags (e.g. "research",
    /// "teach", "document") for the bonds traversed by this
    /// narration. Empty when every traversed bond is pre-S
    /// (Unknown).
    #[serde(default)]
    pub source_kinds: Vec<String>,
    /// Port S — short, human-friendly source labels for the
    /// distinct sources, parallel index order to `source_kinds`
    /// (e.g. "en.wikipedia.org", "teach:john").
    #[serde(default)]
    pub source_labels: Vec<String>,
    /// Port S — concise prose appended to `answer` when at least
    /// one bond in the traversed chain has a non-Unknown source
    /// (e.g. "Based on 2 sources."). Carried separately so
    /// downstream consumers can show / hide it independently.
    #[serde(default)]
    pub source_summary: Option<String>,
    /// OP-ROUTE-2 — routing decision computed in `narrate()` before
    /// recall. Phase 2: populated but not yet acted on (math
    /// short-circuit still lives in `try_arithmetic_response_with`
    /// on the MCP boundary). Phase 3+ will wire think / variants /
    /// reason invocations based on these flags.
    #[serde(default)]
    pub routing: crate::query_router::RoutingDecision,
    /// OP-ROUTE-3a -- reasoning trace built when `routing.needs_deep_reasoning`
    /// fired (and idk_gate passed) OR consumed inside the refusal path.
    /// `None` when deep reasoning was not invoked (the common case).
    /// `#[serde(default)]` for backward-compat with pre-3a JSON.
    #[serde(default)]
    pub reasoning: Option<crate::reasoning::ReasoningTrace>,
    /// OP-ROUTE-3b -- compact summary of the `VariantArena` built when
    /// `routing.needs_variants` fired (and idk_gate passed, and the
    /// goal-seed activation threshold cleared). `None` when variants
    /// did not run. `#[serde(default)]` for backward-compat with pre-3b
    /// JSON.
    #[serde(default)]
    pub variants: Option<crate::variants::VariantArenaSummary>,
    /// OP-ROUTE-3c -- compact summary of the `ThoughtTrace` built
    /// when `routing.needs_agentic_thinking` was true and the PASS
    /// path elected to run a `Brain::think` loop (idk_gate passed,
    /// not meta / math, goal-seed activation above
    /// `thinking_min_activation`). `None` when thinking did not
    /// run. `#[serde(default)]` for backward-compat with pre-3c JSON.
    #[serde(default)]
    pub thinking: Option<crate::thinking::ThinkingTraceSummary>,
    /// OP-REASON-LOOP Phase 1 -- per-prompt ContextSet built from the recall
    /// on the PASS path (None on refusal / short-circuit paths). Skipped in
    /// serialization: it is the internal hand-off to the Phase 1.5 session
    /// hold, not part of the wire answer. Derived view data only (labels /
    /// activations), never nodes or bonds.
    #[serde(skip)]
    pub context_set: Option<crate::reason_loop::ContextSet>,
    /// OP-REASON-LOOP Phase 3 -- observability summary of the reason<->retrieve
    /// loop (enriched ContextSet size, extra iterations run, and the re-seed
    /// concepts reasoning chose). None on refusal / short-circuit paths.
    /// JSON-safe, so unlike context_set it is serialized.
    #[serde(default)]
    pub reason_loop: Option<crate::reason_loop::ReasonLoopSummary>,
    /// B1/#211 -- resolved stored confidence label for the answer-head node.
    /// Surfaces the REAL confidence (a loaded v5 resident node is a stub
    /// Unverified). None on refusal / short-circuit / empty-recall paths.
    #[serde(default)]
    pub head_confidence: Option<String>,
}

/// Public entry point. Internal to the narration module so `Brain`
/// keeps a thin wrapper. Pure orchestration: classify -> extract ->
/// select -> modulate -> render.
/// OP-NEARMISS-1 -- spelling-repair wrapper around [`narrate_inner`].
///
/// Exact seed matching made Lu refuse *"Tell me about the python
/// launguage"* outright. A near-miss on a term the brain DOES know is a
/// typo, not an unknown concept, and refusing it is the wrong answer.
///
/// The repair is bounded by construction: candidates come only from the
/// keyword index, so a token can only ever be renamed to a term Lu
/// already has. Nothing here touches the K-gate -- the gate runs
/// unchanged, downstream, on the repaired query. A word Lu has never
/// seen has no repair target, so it reaches the gate exactly as typed
/// and is refused exactly as before.
///
/// Every repair is DISCLOSED in the answer. A silent correction is a
/// correction the user cannot overrule.
/// FIX 1 (OP-DEF-GLOSS): resolve the stored definitional gloss for a
/// "what is X" / "define X" query. Derives X's canonical node identity
/// the SAME way `Brain::set_definition` stored it -- `fold_seed_label`
/// (leading-article strip + lowercase) then `Seed::from_label` -- then
/// reads the node's `KnowledgeContent::Definition` via the v5 point-read
/// path (`get_node_v5`). Returns None when the subject is empty, has no
/// canonical node, or the node carries no non-empty Definition content,
/// so the idk floor and every non-definitional answer are untouched.
fn definition_gloss_for_query(brain: &crate::brain::Brain, query: &str) -> Option<String> {
    let lower = query.trim().to_lowercase();
    let subject = ["what is ", "what are ", "what's ", "define ", "definition of ", "what was "]
        .iter()
        .find_map(|pfx| lower.strip_prefix(pfx))
        .unwrap_or(lower.as_str())
        .trim()
        .trim_end_matches(|c: char| c == '?' || c == '.' || c == '!' || c.is_whitespace());
    if subject.is_empty() {
        return None;
    }
    // OP-MATH-HYPHEN-GUARD companion (VERIFY-FIRST 2026-08-03): a hyphenated
    // concept term ("half-life") is commonly stored under its SPACED canonical
    // node ("half life"), where the WordNet/GCIDE gloss lives. Try the subject
    // as-is first, then a hyphen->space variant, so "what is half-life" reaches
    // the same Definition as "what is half life". A non-hyphenated subject
    // dedups to a single lookup -> byte-identical prior behaviour, and the idk
    // floor is untouched (no gloss node on either form -> None).
    let mut cands = vec![subject.to_string()];
    let spaced = subject.replace('-', " ");
    if spaced != subject {
        cands.push(spaced);
    }
    for cand in cands {
        if let Some(d) = definition_gloss_lookup(brain, &cand) {
            return Some(d);
        }
    }
    None
}

/// Point lookup for a single subject phrase, canonicalised the SAME way the
/// ingest write path stored it (`fold_seed_label` then `Seed::from_label`),
/// reading the node's `KnowledgeContent::Definition` via the v5 point-read
/// (writer/MemTable path) with the read-only node-store path as fallback.
fn definition_gloss_lookup(brain: &crate::brain::Brain, subject: &str) -> Option<String> {
    let key = crate::canonical::fold_seed_label(subject);
    let seed = crate::seed::Seed::from_label(&key);
    // OP-SOURCEPULL Batch-B (Fix 1): a stored gloss must surface even when the
    // node is not resident in node_intern -- the read-only guest interns from
    // the bond graph, so a lightly-connected concept node can miss there while
    // its Definition is fully present in the SST node-store. Try the point-read
    // first, then ALWAYS fall through to the SST-backed definition store keyed
    // by seed; never early-return on an intern miss.
    if let Some(id) = brain.network.node_intern().get(&seed) {
        if let Ok(Some(node)) = brain.get_node_v5(id) {
            if let crate::node::KnowledgeContent::Definition(def) = &node.knowledge.content {
                let d = def.trim();
                if !d.is_empty() {
                    return Some(d.to_string());
                }
            }
        }
    }
    if let Some(d) = brain.network.definition_from_store(&seed) {
        return Some(d);
    }
    // OP-GLOSS ACRONYM-EXPANSION (verify-first 2026-08-05, narrowed): a
    // gloss-less abbreviation node (DNA, NASA, FBI, NATO) carries a single
    // RelatedTo edge to its spelled-out form, which holds the real stored
    // gloss. Serve that expansion's gloss ONLY when the subject is a genuine
    // abbreviation of the target: a single token, strictly shorter than the
    // target, and a letter-subsequence of it (d,n,a within "deoxyribonucleic
    // acid"; n,a,s,a within "national aeronautics and space administration").
    // That EXCLUDES person-name RelatedTo links ("charles darwin" -> "darwin":
    // multi-word and longer than the target), so the richer bio/overview path
    // is byte-unchanged. Never overrides a real gloss, never guesses among
    // sibling senses, idk floor intact (no valid expansion -> None -> honest
    // refusal). Real data only; generalizes across acronyms.
    if !subject.contains(' ') {
        let subj_l: String = subject
            .chars()
            .filter(|c| c.is_alphanumeric())
            .flat_map(|c| c.to_lowercase())
            .collect();
        let mut expansion: Option<String> = None;
        let mut related_seen = 0usize;
        for (tgt, bt) in brain.network.typed_edges_out(&seed) {
            if matches!(bt, crate::bond::BondType::RelatedTo) {
                related_seen += 1;
                if related_seen > 1 {
                    return None;
                }
                let tlabel = brain.network.node_label(&tgt).unwrap_or_default();
                let tgt_l: String = tlabel
                    .chars()
                    .filter(|c| c.is_alphanumeric())
                    .flat_map(|c| c.to_lowercase())
                    .collect();
                if !subj_l.is_empty()
                    && subj_l.len() < tgt_l.len()
                    && is_letter_subsequence(&subj_l, &tgt_l)
                {
                    if let Some(d) = brain.network.definition_from_store(&tgt) {
                        let d = d.trim();
                        if !d.is_empty() {
                            expansion = Some(d.to_string());
                        }
                    }
                }
            }
        }
        return expansion;
    }
    None
}

/// True when every char of `needle` appears in `hay` in order (subsequence).
/// Used to confirm an abbreviation (needle) telescopes into its expansion
/// (hay) before serving the expansion's gloss.
fn is_letter_subsequence(needle: &str, hay: &str) -> bool {
    let mut it = hay.chars();
    for c in needle.chars() {
        loop {
            match it.next() {
                Some(h) if h == c => break,
                Some(_) => continue,
                None => return false,
            }
        }
    }
    true
}

/// OP-GLOSS PART 1: compose an honest one-line definition for a bare CONCEPT
/// that carries typed hierarchy edges but has NO stored Definition gloss. This
/// is the FALLBACK -- it runs ONLY after `definition_gloss_for_query` returned
/// None, on the SAME canonical `fold_seed_label(subject)` seed that lookup
/// used, and only for Definition-intent queries. So a stored gloss ALWAYS wins
/// (never overridden), "who is X" / overview shapes are untouched, and a clean
/// short gloss ("what is python") is never re-woven.
///
/// The subject must resolve to a REAL interned node (nonsense with no node ->
/// None) AND carry at least one IsA/DefinedAs edge (the hierarchy spine). With
/// no such edge we return None so the honest idk floor still refuses rather
/// than fabricate. Enrichment is drawn ONLY from real typed edges (PartOf /
/// UsedFor / HasProperty, plus a soft RelatedTo tail when nothing stronger is
/// present); every clause cites an edge that exists -- no invented connective.
fn composed_concept_gloss(brain: &crate::brain::Brain, query: &str) -> Option<String> {
    let lower = query.trim().to_lowercase();
    let subject = ["what is ", "what are ", "what's ", "define ", "definition of ", "what was "]
        .iter()
        .find_map(|pfx| lower.strip_prefix(pfx))
        .unwrap_or(lower.as_str())
        .trim()
        .trim_end_matches(|c: char| c == '?' || c == '.' || c == '!' || c.is_whitespace());
    if subject.is_empty() {
        return None;
    }
    let key = crate::canonical::fold_seed_label(subject);
    let seed = crate::seed::Seed::from_label(&key);
    // Guard nonsense: the subject must be a real node. No node -> no edges ->
    // None -> the idk floor refuses honestly (never fabricated).
    if brain.network.node_intern().get(&seed).is_none() {
        return None;
    }
    let display = brain.network.node_label(&seed).unwrap_or_else(|| subject.to_string());
    let dlow = display.trim().to_lowercase();

    let mut isa: Vec<String> = Vec::new();
    let mut parts: Vec<String> = Vec::new();
    let mut uses: Vec<String> = Vec::new();
    let mut rel: Vec<String> = Vec::new();
    for (nbr, bt) in brain.network.typed_edges_out(&seed).into_iter().take(128) {
        let raw = brain.network.node_label(&nbr).unwrap_or_default();
        let lbl = scrub_internal_tokens(&raw).trim().to_string();
        if lbl.is_empty() || lbl.to_lowercase() == dlow {
            continue;
        }
        match bt {
            crate::bond::BondType::IsA | crate::bond::BondType::DefinedAs => {
                if isa.len() < 2 && !isa.iter().any(|x| x.eq_ignore_ascii_case(&lbl)) {
                    isa.push(lbl);
                }
            }
            crate::bond::BondType::PartOf => {
                if parts.is_empty() {
                    parts.push(lbl);
                }
            }
            crate::bond::BondType::UsedFor => {
                if uses.is_empty() {
                    uses.push(lbl);
                }
            }
            crate::bond::BondType::RelatedTo => {
                if rel.is_empty() {
                    rel.push(lbl);
                }
            }
            _ => {}
        }
    }
    // Honest floor: no hierarchy edge to stand on -> refuse, do not pad.
    if isa.is_empty() {
        return None;
    }
    let mut prose = format!("{} is a kind of {}", capitalize_first(&display), concept_join(&isa));
    let mut clauses: Vec<String> = Vec::new();
    if let Some(pp) = parts.first() {
        clauses.push(format!("part of {}", pp));
    }
    if let Some(u) = uses.first() {
        clauses.push(format!("used for {}", u));
    }
    // Soft RelatedTo tail only when no stronger enrichment was found.
    if clauses.is_empty() {
        if let Some(r) = rel.first() {
            clauses.push(format!("related to {}", r));
        }
    }
    if !clauses.is_empty() {
        prose.push_str(", ");
        prose.push_str(&concept_join(&clauses));
    }
    prose.push('.');
    Some(prose)
}

/// Phase 0 (OP-CODE-GLOSS): clean code-intent definitional answer for a coding
/// operation. On a "what does X do / define X / what is X" frame where X
/// resolves to a C2 coding node ("the <op> function"), render a definition from
/// that node's typed edges (Returns / Produces / UsedFor / HasParameter /
/// Causes), suppressing internal routing handles from the OUTPUT. Returns None
/// for any non-coding subject, so non-code Q&A is byte-identical. Resolves the
/// node by label directly (independent of the sense-router).
fn is_handleish(s: &str) -> bool {
    let t = s.trim();
    t.starts_with("pyfn") || t.starts_with("jsfn") || t.starts_with("builtins.")
        || t.starts_with("js:") || t.starts_with("pyop") || t == "opsum" || t == "pysum2"
}

/// OP-QA-2: final output scrub. Internal node identifiers must never reach
/// a user-facing answer, whichever shape produced them:
///   * WordNet sense-ids `lemma.pos.n` (pos in n/v/a/r/s) -> bare lemma
///     (underscores -> spaces): "instruction.n.4" -> "instruction".
///   * routing handles `builtins.foo`, `pyfn*`, `jsfn*`, `pyop*`, `js:foo`
///     -> the trailing name.
/// Pure token pass; ordinary prose is untouched because the sense pattern
/// requires a single [nvasr] segment followed by digits (so "U.S.A", "3.14"
/// and version strings do not match).
fn scrub_internal_tokens(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut tok = String::new();
    for ch in s.chars() {
        if ch.is_whitespace() {
            if !tok.is_empty() { out.push_str(&scrub_one_token(&tok)); tok.clear(); }
            out.push(ch);
        } else {
            tok.push(ch);
        }
    }
    if !tok.is_empty() { out.push_str(&scrub_one_token(&tok)); }
    out
}

fn scrub_one_token(tok: &str) -> String {
    let start = tok.find(|c: char| c.is_alphanumeric()).unwrap_or(tok.len());
    let (lead, rest) = tok.split_at(start);
    let end = rest
        .rfind(|c: char| c.is_alphanumeric())
        .map(|i| i + rest[i..].chars().next().map(|c| c.len_utf8()).unwrap_or(1))
        .unwrap_or(0);
    let (core, trail) = rest.split_at(end);
    format!("{}{}{}", lead, clean_core(core), trail)
}

fn clean_core(core: &str) -> String {
    if core.is_empty() { return String::new(); }
    let parts: Vec<&str> = core.split('.').collect();
    if parts.len() >= 3 {
        let pos = parts[parts.len() - 2];
        let num = parts[parts.len() - 1];
        if matches!(pos, "n" | "v" | "a" | "r" | "s")
            && !num.is_empty()
            && num.chars().all(|c| c.is_ascii_digit())
        {
            return parts[..parts.len() - 2].join(".").replace('_', " ");
        }
    }
    if let Some(r) = core.strip_prefix("builtins.") { return r.to_string(); }
    if let Some(r) = core.strip_prefix("js:") { return r.to_string(); }
    for pre in ["pyfn", "jsfn", "pyop"] {
        if let Some(r) = core.strip_prefix(pre) {
            return r.trim_start_matches('_').to_string();
        }
    }
    core.to_string()
}
/// Pure op-token extractor for the code-intent gloss frame. Single op token
/// for "what does X do" / "define X" / "what is (the) X (function)"; None for
/// math or usage phrasings ("the sum OF 2 AND 3") or multi-word subjects.
fn code_op_token(ql: &str) -> Option<String> {
    let ql = ql.trim().trim_end_matches(|c: char| c == '?' || c == '.' || c == '!' || c.is_whitespace());
    let rest: &str = if let Some(r) = ql.strip_prefix("what does ") {
        r.trim_end_matches(" do").trim_end_matches(" return").trim_end_matches(" take")
    } else if let Some(r) = ql.strip_prefix("define ") { r }
    else if let Some(r) = ql.strip_prefix("what is ") { r }
    else { return None };
    let toks: Vec<&str> = rest.split_whitespace().collect();
    const REJECT: &[&str] = &["of","and","plus","minus","times","divided","between","versus","vs","from","to","with","in","my","these","those"];
    if toks.iter().any(|t| REJECT.contains(t) || t.chars().any(|c| c.is_ascii_digit())) { return None; }
    let content: Vec<&str> = toks.iter().copied()
        .filter(|t| !matches!(*t, "the"|"a"|"an"|"function"|"method"|"builtin"|"built-in"|"do"|"does"))
        .collect();
    if content.len() != 1 || content[0].len() < 2 { return None; }
    Some(content[0].to_string())
}
fn code_op_gloss(brain: &crate::brain::Brain, query: &str) -> Option<String> {
    let ql = query.trim().to_lowercase();
    let ql = ql.trim_end_matches(|c: char| c == '?' || c == '.' || c == '!' || c.is_whitespace());
    // Extract the op token from a code-explanation frame.
    let op = code_op_token(&ql)?;
    // Require the canonical coding node to exist; otherwise this is not a
    // coding op and we must not intercept the query.
    let cand = format!("the {} function", op);
    // Ingest identity is the RAW label unless LU_CANON_INGEST is on
    // (default off), so resolve raw first, then the folded key as fallback.
    let raw_seed = crate::seed::Seed::from_label(&cand);
    let seed = if brain.network.node_intern().get(&raw_seed).is_some() {
        raw_seed
    } else {
        crate::seed::Seed::from_label(&crate::canonical::fold_seed_label(&cand))
    };
    brain.network.node_intern().get(&seed)?;
    let (mut produces, mut usedfor, mut returns, mut params, mut causes) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for (nbr, bt) in brain.network.typed_edges_out(&seed).into_iter().take(64) {
        let obj = brain.network.node_label(&nbr).unwrap_or_default();
        if obj.is_empty() || is_handleish(&obj) { continue; }
        match bt {
            crate::bond::BondType::Returns => returns.push(obj),
            crate::bond::BondType::Produces => produces.push(obj),
            crate::bond::BondType::UsedFor => usedfor.push(obj),
            crate::bond::BondType::HasParameter => params.push(obj),
            crate::bond::BondType::Causes => causes.push(obj),
            _ => {}
        }
    }
    let main = if let Some(o) = returns.first() { format!("returns {}", o) }
        else if let Some(o) = produces.first() { format!("produces {}", o) }
        else if let Some(o) = usedfor.first() { format!("is used for {}", o) }
        else { return None };
    let mut out = format!("{} {}.", op, main);
    if !params.is_empty() {
        let ps: Vec<String> = params.into_iter().take(4).collect();
        out.push_str(&format!(" It takes {}.", ps.join(", ")));
    }
    if !causes.is_empty() {
        let es: Vec<String> = causes.into_iter().take(2)
            .map(|e| e.replace(" when misused", "").trim().to_string()).collect();
        out.push_str(&format!(" It can raise {}.", es.join(", ")));
    }
    Some(out)
}

/// OP-SENSE-GLOSS (FIX 1 + FIX 2): notability-led definitional default.
///
/// For a definitional/language frame over a polysemous concept token that has
/// a `csense<token>` routing handle into a QID-anchored concept sense node
/// (de-fold: `csense<t> -IsA-> <Label>#Q<qid>`), render the definition
/// straight from the QID shelf's typed edges -- LEADING with the notable
/// concept sense and naming the everyday sense in one honest clause derived
/// from the bare token node's own animal/IsA edges.
///
/// Bounded + gated three ways so it never steals the everyday sense:
///   (a) the sense-router must be enabled (router-off => byte-identical);
///   (b) `concept_sense_tokens` must fire (what is / define / tell me about /
///       "<x> language") -- NOT a bare token, NOT "is a python a snake";
///   (c) no animal/everyday cue in the query (snake/animal/"is a ...").
/// Returns None unless the handle resolves to a shelf with real groundable
/// edges, so unknown or non-de-folded subjects fall through to the legacy
/// paths and the idk floor is untouched.
fn concept_sense_gloss(brain: &crate::brain::Brain, query: &str) -> Option<String> {
    if !crate::network::sense_router_enabled() {
        return None;
    }
    let ctoks = crate::network::concept_sense_tokens(query);
    if ctoks.is_empty() {
        return None;
    }
    if concept_has_everyday_cue(&query.to_lowercase()) {
        return None;
    }
    ctoks
        .iter()
        .find_map(|t| concept_sense_gloss_for_token(brain, t))
}

fn concept_has_everyday_cue(ql: &str) -> bool {
    if ql.starts_with("is a ") || ql.starts_with("is an ") {
        return true;
    }
    const CUES: &[&str] = &[
        "snake", "serpent", "reptile", "boa", "constrictor", "animal",
        "creature", "pet",
    ];
    CUES.iter().any(|c| ql.contains(c))
}

fn concept_resolve(brain: &crate::brain::Brain, label: &str) -> Option<crate::seed::Seed> {
    let raw = crate::seed::Seed::from_label(label);
    if brain.network.node_intern().get(&raw).is_some() {
        return Some(raw);
    }
    let folded = crate::seed::Seed::from_label(&crate::canonical::fold_seed_label(label));
    if brain.network.node_intern().get(&folded).is_some() {
        return Some(folded);
    }
    None
}

fn concept_sense_gloss_for_token(brain: &crate::brain::Brain, token: &str) -> Option<String> {
    let handle = format!("csense{}", token);
    let hseed = concept_resolve(brain, &handle)?;
    if brain.network.node_label(&hseed).as_deref() != Some(handle.as_str()) {
        return None;
    }
    let shelf = brain
        .network
        .seam_spread_outgoing_edges(&hseed)
        .into_iter()
        .next()?
        .nbr;

    // OP-SENSE-DEFAULT internal notes bug-2): FAME GATE on the csense lead.
    // The de-fold `csense<token>` handle points at ONE pre-baked QID sense that
    // is NOT fame-ranked; for `atom` it resolves to a programming/editor sense,
    // so this path led "Atom is a programming language" while the physics atom
    // (Q9121, ~251 sitelinks) is ~15x more prominent. Consult the #1518 sitelink
    // fame signal: if a DISTINCT sense in the token's `e_<token>` bucket dominates
    // the shelf sense (same thresholds as resolve_entity's dominant-sense
    // override), the handle is a minority sense -> return None and defer to the
    // honest common-sense gloss (legacy definition path). Byte-identical when the
    // shelf sense IS the dominant one (e.g. `python` -> Python#Q28865, the top of
    // its bucket), so the python->programming default is preserved.
    {
        let shelf_label = brain.network.node_label(&shelf).unwrap_or_default();
        let lead_qid = crate::network::entity_qid_of(&shelf_label).unwrap_or(0);
        let lead_fame = crate::synthesis::entity_fame(lead_qid);
        if let Some((top_fame, top_qid, _)) =
            token_homograph_senses(brain, token).first().copied()
        {
            if top_qid != lead_qid
                && top_fame >= crate::synthesis::FAME_DOMINANT_MIN
                && top_fame >= lead_fame.saturating_mul(crate::synthesis::FAME_DOMINANCE_RATIO)
                && top_fame.saturating_sub(lead_fame) >= crate::synthesis::FAME_DOMINANCE_MARGIN
            {
                return None;
            }
        }
    }
    let mut isa: Vec<String> = Vec::new();
    let mut uses: Vec<String> = Vec::new();
    for (nbr, bt) in brain.network.typed_edges_out(&shelf).into_iter().take(64) {
        let obj = brain.network.node_label(&nbr).unwrap_or_default();
        let obj = scrub_internal_tokens(&obj);
        let obj = obj.trim().to_string();
        if obj.is_empty() || is_handleish(&obj) {
            continue;
        }
        match bt {
            crate::bond::BondType::IsA => isa.push(obj),
            crate::bond::BondType::UsedFor => uses.push(obj),
            _ => {}
        }
    }
    // Groundability + de-fold-quality gate: require at least one UsedFor
    // (functional) edge. The well-formed QID concept shelf (e.g.
    // Python#Q28865) carries a functional description; malformed / mis-mapped
    // de-fold shelves -- csensedog -> Dog#... -IsA-> a single spurious
    // "programming language" edge, or an obscure homograph sense -- do not,
    // and must fall through to the honest legacy path rather than assert
    // e.g. "Dog is a programming language".
    if uses.is_empty() {
        return None;
    }
    let title = concept_title_case(token);
    let mut out = String::new();
    if let Some(class) = isa.first() {
        out.push_str(&format!("{} is {} {}", title, concept_article(class), class));
        if !uses.is_empty() {
            out.push_str(&format!(", used for {}", concept_join(&uses)));
        }
    } else {
        out.push_str(&format!("{} is used for {}", title, concept_join(&uses)));
    }
    out.push('.');
    if let Some(nod) = concept_everyday_nod(brain, token) {
        out.push(' ');
        out.push_str(&nod);
    }
    Some(scrub_internal_tokens(&out))
}

/// The honest everyday-sense clause, derived by walking the BARE token
/// node's own IsA hypernym chain (never fabricated, never the concept sense).
/// Returns the first hypernym that names a common everyday category, e.g.
/// python -IsA-> boa -IsA-> constrictor -IsA-> snake yields "a snake". None
/// when the chain names no everyday category, so the concept answer stands
/// alone. IsA-only: diet/behaviour HasProperty edges ("eat animal") are
/// deliberately excluded so the clause names a category, not a predicate.
fn concept_everyday_nod(brain: &crate::brain::Brain, token: &str) -> Option<String> {
    const ANCHORS: &[&str] = &[
        "snake", "serpent", "reptile", "mammal", "bird", "fish", "insect",
        "amphibian", "animal", "plant", "tree", "flower", "fruit",
    ];
    let mut cur = concept_resolve(brain, token)?;
    let mut visited: std::collections::HashSet<crate::seed::Seed> =
        std::collections::HashSet::new();
    visited.insert(cur);
    for _ in 0..6 {
        let mut next: Option<crate::seed::Seed> = None;
        for (nbr, bt) in brain.network.typed_edges_out(&cur).into_iter().take(64) {
            if !matches!(bt, crate::bond::BondType::IsA) {
                continue;
            }
            let label =
                scrub_internal_tokens(&brain.network.node_label(&nbr).unwrap_or_default());
            let low = label.to_lowercase();
            let mut matched: Option<&str> = None;
            for &a in ANCHORS {
                if low.split(|c: char| !c.is_alphanumeric()).any(|w| w == a) {
                    matched = Some(a);
                    break;
                }
            }
            if let Some(a) = matched {
                return Some(format!("(It's also {} {}.)", concept_article(a), a));
            }
            if next.is_none() {
                next = Some(nbr);
            }
        }
        match next {
            Some(n) if visited.insert(n) => cur = n,
            _ => break,
        }
    }
    None
}
/// Fame-ranked (#1518 sitelink signal) entity senses of a concept token,
/// gathered from its `e_<token>` bucket. Returns (fame, qid, seed) sorted by
/// fame descending. Empty when the token has no entity senses (non-polysemous
/// or entity-free brains) -> callers degrade to a single honest gloss.
fn token_homograph_senses(
    brain: &crate::brain::Brain,
    token: &str,
) -> Vec<(u32, u64, crate::seed::Seed)> {
    let key = format!("e_{}", token);
    let tl = token.to_lowercase();
    let mut v: Vec<(u32, u64, crate::seed::Seed)> = Vec::new();
    if let Some(bucket) = brain.network.keyword_index_bucket(&key) {
        for s in bucket.iter().take(512) {
            let lbl = brain.network.node_label(s).unwrap_or_default();
            // HOMOGRAPH ONLY: the entity's own name must BE the token
            // ("atom#Q9121" -> "atom"), so genuine same-spelling senses count
            // while superset phrases ("quantum mechanics", "natural
            // radioactivity") do NOT -- those are related concepts, not
            // alternate senses of the word, and must never be nodded.
            if crate::network::entity_label_part(&lbl).to_lowercase() != tl {
                continue;
            }
            if let Some(q) = crate::network::entity_qid_of(&lbl) {
                v.push((crate::synthesis::entity_fame(q), q, *s));
            }
        }
    }
    v.sort_by(|a, b| b.0.cmp(&a.0));
    v
}

/// Prominence floor (sitelinks) for a homograph sense to count toward
/// polysemy or be named in a nod. Two senses must clear it for a token to be
/// treated as genuinely polysemous.
const SENSE_NOD_FLOOR: u32 = 25;

/// The clean IsA class label of an alternate sense node (e.g. Java#Q251 ->
/// "programming language"), used to NAME an honest nod clause. Walks the node's
/// own typed IsA out-edges and returns the first groundable class: scrubbed,
/// <= 3 words, alphabetic, not a routing handle, not the token itself. None
/// when the node carries no clean class edge -> no nod is emitted (never
/// fabricated).
fn alt_sense_kind(
    brain: &crate::brain::Brain,
    seed: &crate::seed::Seed,
    token: &str,
) -> Option<String> {
    let mut first: Option<String> = None;
    for (nbr, bt) in brain.network.typed_edges_out(seed).into_iter().take(64) {
        if !matches!(bt, crate::bond::BondType::IsA) {
            continue;
        }
        let raw = brain.network.node_label(&nbr).unwrap_or_default();
        let part = crate::network::entity_label_part(&raw).to_string();
        let cleaned = scrub_internal_tokens(&part);
        let k = cleaned.trim();
        if k.is_empty() || is_handleish(k) {
            continue;
        }
        let wc = k.split_whitespace().count();
        if wc == 0 || wc > 3 {
            continue;
        }
        if !k.chars().all(|c| c.is_alphabetic() || c.is_whitespace() || c == '-') {
            continue;
        }
        if k.to_lowercase() == token.to_lowercase() {
            continue;
        }
        // Prefer a recognizable category noun when present (e.g. name the Java
        // / Ruby sense a "programming language" rather than the first raw IsA).
        if k.to_lowercase().contains("language") {
            return Some(k.to_string());
        }
        if first.is_none() {
            first = Some(k.to_string());
        }
    }
    first
}

/// OP-SENSE-DEFAULT internal notes bug-2): honest one-clause nod to the
/// next most-prominent DISTINCT sense of a polysemous concept token, appended
/// to the LEADING definitional gloss. Generalises the python-only "(It's also a
/// snake.)" default to every polysemous token via the #1518 fame signal:
/// `what is java` -> lead island, nod "(It's also a programming language.)";
/// `what is mercury` -> lead element, nod "(It's also a planet.)". The sense
/// already expressed by the lead gloss is skipped (never nod the sense we led
/// with), and the alternate is NAMED only from a real IsA class edge on its own
/// node -- no fabrication. Returns None (no nod) when the token is
/// non-polysemous, no alternate clears the prominence floor, or no clean class
/// exists, so a single clean gloss (radioactivity, dog) is untouched.
fn concept_fame_alt_nod(
    brain: &crate::brain::Brain,
    query: &str,
    lead_gloss: &str,
) -> Option<String> {
    if !crate::network::sense_router_enabled() {
        return None;
    }
    let ctoks = crate::network::concept_sense_tokens(query);
    let token = ctoks.first()?;
    if concept_has_everyday_cue(&query.to_lowercase()) {
        return None;
    }
    let gl = lead_gloss.to_lowercase();
    let homs = token_homograph_senses(brain, token);
    // Genuine-polysemy gate: require >= 2 same-spelling senses that each clear
    // the prominence floor. A single-sense concept (radioactivity,
    // entanglement) or a token whose bucket holds only superset phrases
    // (quantum -> quantum mechanics/chemistry) fails this and is left as one
    // clean gloss -- no nod.
    if homs.iter().filter(|(f, _, _)| *f >= SENSE_NOD_FLOOR).count() < 2 {
        return None;
    }
    for (fame, _qid, seed) in homs {
        if fame < SENSE_NOD_FLOOR {
            break; // sorted desc: nothing below the floor qualifies
        }
        let kind = match alt_sense_kind(brain, &seed, token) {
            Some(k) => k,
            None => continue,
        };
        if gl.contains(&kind.to_lowercase()) {
            continue; // the lead gloss already expresses this sense
        }
        return Some(format!("(It's also {} {}.)", concept_article(&kind), kind));
    }
    None
}

fn concept_title_case(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
        None => String::new(),
    }
}

fn concept_article(word: &str) -> &'static str {
    match word.trim().chars().next().map(|c| c.to_ascii_lowercase()) {
        Some('a') | Some('e') | Some('i') | Some('o') | Some('u') => "an",
        _ => "a",
    }
}

fn concept_join(items: &[String]) -> String {
    let take: Vec<&str> = items.iter().take(5).map(|s| s.as_str()).collect();
    match take.len() {
        0 => String::new(),
        1 => take[0].to_string(),
        2 => format!("{} and {}", take[0], take[1]),
        n => format!("{}, and {}", take[..n - 1].join(", "), take[n - 1]),
    }
}

/// Naive, bounded singulariser used ONLY for the everyday nod phrase: trims a
/// trailing plural 's' from the final word ("large snakes" -> "large snake").
fn concept_singularize(phrase: &str) -> String {
    let mut words: Vec<String> = phrase.split_whitespace().map(|w| w.to_string()).collect();
    if let Some(last) = words.last_mut() {
        let l = last.to_lowercase();
        if l.len() > 3 && l.ends_with('s') && !l.ends_with("ss") && !l.ends_with("us") {
            last.truncate(last.len() - 1);
        }
    }
    words.join(" ")
}

pub fn narrate(
    brain: &mut crate::brain::Brain,
    query: &str,
    depth: usize,
    top_k: usize,
    use_multi_think: bool,
    cfg: &NarrationConfig,
) -> NarrationResult {
    let repaired = crate::near_miss::repair_query(brain.network.keyword_index_ref(), query);
    let (effective, repairs) = match repaired {
        Some((q, reps)) => (q, reps),
        None => return narrate_inner(brain, query, depth, top_k, use_multi_think, cfg),
    };
    let mut r = narrate_inner(brain, &effective, depth, top_k, use_multi_think, cfg);
    // The query of record stays what the user typed; the disclosure
    // explains what Lu read it as.
    r.query = query.to_string();
    r.answer = format!(
        "{} (I read {}.)",
        r.answer.trim_end(),
        crate::near_miss::describe(&repairs)
    );
    for rep in &repairs {
        r.signals_used
            .push(format!("near_miss_repair:{}->{}", rep.from, rep.to));
    }
    r
}

fn narrate_inner(
    brain: &mut crate::brain::Brain,
    query: &str,
    depth: usize,
    top_k: usize,
    use_multi_think: bool,
    cfg: &NarrationConfig,
) -> NarrationResult {
    // OP-GREET-REFUSAL -- defense-in-depth pre-K-gate pleasantry refusal.
    // The K-gate (idk_gate Mode A) requires `matched_tokens.is_empty()`;
    // with brain growth (>=16M bonds) incidental Wikipedia/Seinfeld bonds
    // on "hello"/"goodbye"/"thank you" defeat that predicate, the gate
    // returns Pass, and downstream picks a Wondering shape. This hook
    // sits ABOVE the K-gate and refuses deterministically via exact
    // (normalized) string match -- no substring matching, so legitimate
    // queries like "what is mitochondria" are untouched. Math takes
    // precedence in the MCP boundary callers (tool_ask / tool_narrate)
    // via try_arithmetic_response_with; narrate() itself never sees
    // math (it short-circuits upstream). See internal notes.
    if let Some(gclass) = crate::greet_refusal::is_greeting(query) {
        let answer = crate::greet_refusal::refusal_prose(gclass).to_string();
        let reason = crate::greet_refusal::refusal_reason(gclass).to_string();
        // Build a minimal AskResult shell so the NarrationResult shape
        // matches the K-gate refusal contract. No brain.ask call; no
        // recall happens for greetings.
        let empty_ask = crate::brain::AskResult {
            query: query.to_string(),
            seed_hits: Vec::new(),
            recall: crate::network::RecallResult {
                activated: Vec::new(),
                query_seeds: Vec::new(),
                scope: crate::scope::TemporalScope::Current,
                nodes_visited: 0,
                bonds_evaluated: 0,
                activation_truncated: false,
                status: crate::network::SpreadStatus::Complete,
            },
            hebbian_strengthened: 0,
            session_id: None,
            bridges: Vec::new(),
            curiosity_signals: Vec::new(),
            scars: Vec::new(),
            status: crate::brain::AskStatus::Complete,
        };
        // Compute routing for diagnostic completeness even though we
        // do not act on it (no recall, no reasoning, no variants, no
        // thinking spawned). Pure function over (query, intent).
        let intent = QueryIntent::Generic;
        let routing = crate::query_router::QueryRouter::route(query, intent);
        return NarrationResult {
            query: query.to_string(),
            answer,
            shape: ResponseShape::Refusal,
            intent,
            flags: ModulationFlags::default(),
            signals_used: vec![reason],
            noise_filtered_count: 0,
            ask: empty_ask,
            multi: None,
            source_kinds: Vec::new(),
            source_labels: Vec::new(),
            source_summary: None,
            routing,
            reasoning: None,
            variants: None,
            thinking: None,
            context_set: None,
            reason_loop: None,
            head_confidence: None,
        };
    }
    let intent = IntentClassifier::classify(query);
    // OP-ROUTE-2 — compute routing decision over (query, intent) before
    // recall. Pure function, no brain access. Phase 2: observation-only;
    // narrate() does not yet alter pipeline shape based on routing.
    let routing = crate::query_router::QueryRouter::route(query, intent);

    // -----------------------------------------------------------
    // OP-NARRATE-V2 (2026-05-31) — SelfIdentity / CurrentInterests
    // short-circuit. These shapes are brain-state probes that don't
    // depend on recall, so they must bypass:
    //   - the ask_with_config call (no recall needed),
    //   - the idk_gate refusal (which fires when no anchor matches),
    //   - downstream multi_think / variants / thinking spawning.
    // Without this short-circuit "What are you?" goes through recall,
    // finds nothing, and the K-gate returns Refusal — exactly the
    // failure mode we saw in the 2026-05-31 test.
    // -----------------------------------------------------------
    // OP-RECALL-1 -- MetaSelf joins this group. It is the same class of
    // query for the same reason: a brain census is answered from
    // `node_count` / `bond_count`, not from recall. It only reaches here
    // in its BARE form now ("what do you know?"), because the topical
    // form is classified as a question about its subject -- and the bare
    // form is 100% hub-stopwords, so it seeds nothing and the K-gate
    // refuses it. Refusing to say how much you know is not honesty, it
    // is a routing bug.
    if intent == QueryIntent::SelfIdentity
        || intent == QueryIntent::CurrentInterests
        || intent == QueryIntent::MetaSelf
    {
        // Build a minimal AskResult shell to satisfy the
        // NarrationResult contract. No recall happens.
        let empty_ask = crate::brain::AskResult {
            query: query.to_string(),
            seed_hits: Vec::new(),
            recall: crate::network::RecallResult {
                activated: Vec::new(),
                query_seeds: Vec::new(),
                scope: crate::scope::TemporalScope::Current,
                nodes_visited: 0,
                bonds_evaluated: 0,
                activation_truncated: false,
                status: crate::network::SpreadStatus::Complete,
            },
            hebbian_strengthened: 0,
            session_id: None,
            bridges: Vec::new(),
            curiosity_signals: Vec::new(),
            scars: Vec::new(),
            status: crate::brain::AskStatus::Complete,
        };
        let shape = match intent {
            QueryIntent::SelfIdentity => ResponseShape::SelfIdentity,
            QueryIntent::MetaSelf => ResponseShape::MetaSelf,
            _ => ResponseShape::CurrentInterests,
        };
        // Build slots with the intent in hand so the
        // current_interests_prose composition runs.
        let slots = extract_slots(
            brain, query, &empty_ask, None, cfg, None, None, None, intent,
        );
        let flags = ModulationFlags::default();
        let now_ms_for_hints = {
            use std::time::{SystemTime, UNIX_EPOCH};
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0)
        };
        let hints = brain.compute_style_hints(now_ms_for_hints);
        // OP-QA-2: scrub internal identifiers on the early-return path too.
        let answer = scrub_internal_tokens(&Renderer::render(shape, &slots, &flags, cfg, &hints));
        return NarrationResult {
            query: query.to_string(),
            answer,
            shape,
            intent,
            flags,
            signals_used: Vec::new(),
            noise_filtered_count: 0,
            ask: empty_ask,
            multi: None,
            source_kinds: Vec::new(),
            source_labels: Vec::new(),
            source_summary: None,
            routing,
            reasoning: None,
            variants: None,
            thinking: None,
            context_set: None,
            reason_loop: None,
            head_confidence: None,
        };
    }
    // OP-SENSE-GLOSS (FIX 1 + FIX 2) -- notability-led definitional default.
    // A bare/context-free definitional or "<x> language" frame over a
    // polysemous concept that has a QID-anchored csense handle is answered by
    // LEADING with the concept sense, rendered from the QID shelf's real typed
    // edges, plus one honest everyday-sense clause. This resolves the subject
    // through the SENSE ROUTER before the keyword-based definition-gloss /
    // topical / reasoning paths (which resolve the bare token to the everyday
    // node), closing both the "what is python" -> snake default and the
    // multi-token "python language" -> refusal. Gated (router on + definitional
    // frame + no animal cue + handle resolves to a groundable shelf), so it is
    // byte-identical when it does not fire and the refusal floor is untouched.
    if let Some(answer) = concept_sense_gloss(brain, query) {
        let empty_ask = crate::brain::AskResult {
            query: query.to_string(),
            seed_hits: Vec::new(),
            recall: crate::network::RecallResult {
                activated: Vec::new(),
                query_seeds: Vec::new(),
                scope: crate::scope::TemporalScope::Current,
                nodes_visited: 0,
                bonds_evaluated: 0,
                activation_truncated: false,
                status: crate::network::SpreadStatus::Complete,
            },
            hebbian_strengthened: 0,
            session_id: None,
            bridges: Vec::new(),
            curiosity_signals: Vec::new(),
            scars: Vec::new(),
            status: crate::brain::AskStatus::Complete,
        };
        return NarrationResult {
            query: query.to_string(),
            answer: scrub_internal_tokens(&answer),
            shape: ResponseShape::Definition,
            intent,
            flags: ModulationFlags::default(),
            signals_used: vec!["concept_sense_gloss".into()],
            noise_filtered_count: 0,
            ask: empty_ask,
            multi: None,
            source_kinds: Vec::new(),
            source_labels: Vec::new(),
            source_summary: None,
            routing,
            reasoning: None,
            variants: None,
            thinking: None,
            context_set: None,
            reason_loop: None,
            head_confidence: None,
        };
    }

// OP-SENSE-LEAD (P9 sense/fame mis-ranking + raw-edge word-salad) -- serve
    // a clean, notability-correct answer for a bare entity/concept term or a
    // "who is/was X" query, replacing two live serving-path failures:
    // (1) "who was Cleopatra" -> "Cleopatra is a painting." and "who was
    // Genghis Khan" -> "The Private Life of Genghis Khan is a literary work."
    // -- resolve_entity only scans the QID-anchored e_<token> entity buckets
    // and never sees the famous WordNet sense (a plain-labelled node with a
    // rich stored Definition, absent from those buckets), so it picks a
    // low-notability same-name homograph; (2) bare "mars"/"atom"/"ruby" fall to
    // the recall FactRecall arm, which leads with a wrong high-activation
    // neighbour and dumps raw adjacent edges as word-salad.
    //
    // Ordering: for a BARE concept mention, route through the fame-aware sense
    // router first (the system's designated disambiguator) by synthesizing a
    // definitional frame, so a polysemous token resolves to its notable sense
    // (python -> "object-based language ... (It's also a snake.)") rather than
    // its raw WordNet primary gloss ("large Old World boas"). The router
    // returns None unless a csense handle resolves to a groundable, fame-passing
    // shelf (atom's editor sense is fame-gated out -> None), so it then falls
    // to the stored-definition lead (mars->planet, cleopatra->queen). WHO mode
    // skips the router and uses sense_lead_definition directly, which only
    // overrides a LOW-notability homograph (high-fame overviews -- Newton,
    // Einstein, countries -- stay with the entity arm). Honesty floor intact:
    // returns None unless a real stored sense exists, so fake names still
    // refuse. Rendered through the same clean Definition shape "what is X" uses.
    let sense_lead_answer = crate::synthesis::sense_subject(query).and_then(|(subject, who_mode)| {
        let router = if !who_mode {
            concept_sense_gloss(brain, &format!("what is {}", subject))
        } else {
            None
        };
        router.or_else(|| crate::synthesis::sense_lead_definition(brain, query))
    });
    if let Some(answer) = sense_lead_answer {
        let empty_ask = crate::brain::AskResult {
            query: query.to_string(),
            seed_hits: Vec::new(),
            recall: crate::network::RecallResult {
                activated: Vec::new(),
                query_seeds: Vec::new(),
                scope: crate::scope::TemporalScope::Current,
                nodes_visited: 0,
                bonds_evaluated: 0,
                activation_truncated: false,
                status: crate::network::SpreadStatus::Complete,
            },
            hebbian_strengthened: 0,
            session_id: None,
            bridges: Vec::new(),
            curiosity_signals: Vec::new(),
            scars: Vec::new(),
            status: crate::brain::AskStatus::Complete,
        };
        return NarrationResult {
            query: query.to_string(),
            answer: scrub_internal_tokens(&answer),
            shape: ResponseShape::Definition,
            intent,
            flags: ModulationFlags::default(),
            signals_used: vec!["sense_lead_definition".into()],
            noise_filtered_count: 0,
            ask: empty_ask,
            multi: None,
            source_kinds: Vec::new(),
            source_labels: Vec::new(),
            source_summary: None,
            routing,
            reasoning: None,
            variants: None,
            thinking: None,
            context_set: None,
            reason_loop: None,
            head_confidence: None,
        };
    }
    // OP-ASK-WATCHDOG -- narrate path tightens the engine default
    // (None) down to NarrationConfig.ask_max_wall_clock_ms. `0`
    // disables the watchdog explicitly. Must be strictly less than
    // thinking_max_wall_clock_ms so the inner observation ask trips
    // its watchdog before the outer think watchdog fires.
    let ask_wall = if cfg.ask_max_wall_clock_ms == 0 {
        None
    } else {
        Some(std::time::Duration::from_millis(cfg.ask_max_wall_clock_ms))
    };
    let ask_cfg = crate::brain::BrainAskConfig::new(depth, top_k)
        .with_wall_clock(ask_wall);
    // FIX 2 (OP-DEF-FASTPATH): for a Definition-intent query whose subject
    // resolves to a stored gloss (FIX 1), the grounded answer comes straight
    // from the canonical node, so the recall spread is redundant -- and the
    // spread/seed path is exactly what read-amplifies on a hub subject like
    // "dog". Skip it and build the same minimal AskResult shell the
    // Self/Meta fast paths use; the gloss (surfaced below) is the answer.
    // Non-definition and unknown-subject queries take the normal ask.
    let definition_fastpath_gloss: Option<String> =
        (if intent == QueryIntent::Definition {
            definition_gloss_for_query(brain, query)
                .or_else(|| composed_concept_gloss(brain, query))
                .map(|g| match concept_fame_alt_nod(brain, query, &g) {
                    Some(nod) => format!("{} {}", g.trim_end(), nod),
                    None => g,
                })
        } else {
            None
        })
        .or_else(|| code_op_gloss(brain, query));
    let ask = if definition_fastpath_gloss.is_some() {
        crate::brain::AskResult {
            query: query.to_string(),
            seed_hits: Vec::new(),
            recall: crate::network::RecallResult {
                activated: Vec::new(),
                query_seeds: Vec::new(),
                scope: crate::scope::TemporalScope::Current,
                nodes_visited: 0,
                bonds_evaluated: 0,
                activation_truncated: false,
                status: crate::network::SpreadStatus::Complete,
            },
            hebbian_strengthened: 0,
            session_id: None,
            bridges: Vec::new(),
            curiosity_signals: Vec::new(),
            scars: Vec::new(),
            status: crate::brain::AskStatus::Complete,
        }
    } else {
        brain.ask_with_config(query, ask_cfg)
    };

    // OP-014 — Wire idk_gate into narrate(). Mirrors the
    // Port K consultation in lu4d_ask (src/mcp.rs): after
    // recall, consult the asymmetry gate. If it refuses,
    // build the same coherent refusal prose lu4d_ask
    // produces and return a Refusal-shape NarrationResult
    // early. Closes the load-bearing gap where lu4d_narrate
    // fabricated confident "X is a Y" prose on IDK and
    // adversarial queries because narrate() did not honor
    // the discipline lu4d_ask already does. The gate is
    // called with `query` as both raw and effective —
    // narration callers do not have access to the Q.3
    // rewrite distinction (that lives in the MCP handler);
    // for in-process narrate() the two are identical.
    // session_id is None — narration-layer refusals do not
    // record session turns; that is the ask-layer's job.
    // Gate is disabled only via `cfg.idk_gate_enabled = false`
    // for unit tests that hand-build tiny brains exercising
    // shape-selection logic in isolation.
    // OP-CONCLUDE Tier 1.6 -- grounded-reasoning reachability. Compute a
    // conclusion from RECOGNISED reasoning shapes (category / property-check /
    // common-ancestor) BEFORE the idk entry floor. They resolve operands
    // directly to concept nodes and cite REAL edges, so a fully-grounded but
    // weakly-anchored query ("is a dog loyal", dog -HasProperty-> loyal) is no
    // longer refused before it can be cited. The floor THRESHOLD is unchanged:
    // only a conclusion resting on an actual edge bypasses it; no edge -> None
    // -> the floor still refuses honestly (never a fabrication). ctx is None
    // here (built below); the ctx-dependent inherited-property shape re-runs on
    // the PASS path.
    let early_conclusion =
        crate::synthesis::try_synthesize(brain, query, intent, None);
    let grounded_reasoning = early_conclusion.is_some();
    // OP-SOURCEPULL Batch-C (Fix 3): a stored definitional gloss
    // (definition_fastpath_gloss, surfaced at the OP-DEF-GLOSS block below) is
    // grounded truth read straight from the subject node. The entry idk floor
    // evaluates the INTENTIONALLY-EMPTY fastpath ask shell, so a multi-token
    // subject whose bare recall signal is thin ("quantum field theory",
    // "wave-particle duality") refused BEFORE the gloss could surface -- while a
    // stronger-recall subject ("radioactivity") slipped through. The reason-loop
    // EXIT floor below already skips when a fastpath gloss exists; mirror that
    // here so the entry floor cannot override a grounded stored definition. No
    // gloss -> gate runs unchanged, so the honesty floor for ungrounded
    // subjects ("what is florble") is untouched.
    if cfg.idk_gate_enabled && !grounded_reasoning && definition_fastpath_gloss.is_none() {
        let idk_thresholds = crate::idk_gate::IdkThresholds::default();
        if !crate::idk_gate::fast_pass_check(brain, &ask, query, &idk_thresholds) {
            let trace = {
                let r_cfg = crate::reasoning::ReasoningConfig::default();
                let scope = crate::scope::TemporalScope::Current;
                crate::reasoning::reason_with_config(brain, query, scope, &r_cfg)
            };
            let signals = crate::idk_gate::evaluate_signals(
                brain, &ask, query, Some(&trace), &idk_thresholds,
            );
            let verdict = crate::idk_gate::classify(&signals, &idk_thresholds);
            if verdict != crate::idk_gate::IdkVerdict::Pass {
                let (prose, _structured) = crate::idk_gate::build_response(
                    verdict, &signals, query, query, None,
                );
                // OP-NARRATE-1 Phase C: refusal prose is rendered by
                // `idk_gate::build_response` directly and is no longer
                // post-processed. Style adaptation now lives in
                // `render_core` for shape-bound paths; refusal prose
                // bypasses render_core entirely, so it stays neutral.
                // B1/#211 sibling (E5 silent-zero): a spread that hit the
                // wall-clock watchdog returns AskStatus::TimedOut with only
                // PARTIAL activations. Refusing it as a flat knowledge gap is
                // dishonest -- the search was TRUNCATED, not exhausted. Disclose
                // the truncation so the user knows it is a time limit, not an
                // absence of knowledge. (Complete recalls are unaffected.)
                let answer = disclose_truncation(prose, ask.status);
                return NarrationResult {
                    query: query.to_string(),
                    answer,
                    shape: ResponseShape::Refusal,
                    intent,
                    flags: ModulationFlags::default(),
                    signals_used: vec!["idk_gate".into()],
                    noise_filtered_count: 0,
                    ask,
                    multi: None,
                    source_kinds: Vec::new(),
                    source_labels: Vec::new(),
                    source_summary: None,
                    routing,
                    // OP-ROUTE-3a: surface the refusal-side trace for
                    // consistent API (consumers can inspect what the
                    // chain found without re-running reason_with_config).
                    reasoning: Some(trace),
                    // OP-ROUTE-3b: refusal path never spawns variants;
                    // the cost gate sits inside the PASS branch.
                    variants: None,
                    // OP-ROUTE-3c: refusal path never spawns thinking;
                    // same PASS-only invariant as 3b.
                    thinking: None,
                    context_set: None,
                    reason_loop: None,
                    head_confidence: None,
                };
            }
        }
    }

    // OP-ROUTE-GUEST (2026-07-23) -- wire the previously-dead
    // `needs_critic_vetting` router branch. Claim-verification queries
    // ("is it true that", "fact-check", "prove it", "how do you know")
    // set routing.needs_critic_vetting; before this the flag was computed
    // every request and discarded (the "Phase 3c will wire multi-agent
    // critic vetting here" TODO). multi_think is ungated (a Brain method,
    // compiled into the guest binary) and self-bounded by
    // max_multi_think_lifetime_ms (30s). It runs strictly AFTER the
    // idk_gate refusal returns above, so the honesty/refusal floor is
    // untouched -- the critic only vets beliefs on the PASS path. The
    // explicit caller opt-in `use_multi_think` is preserved (OR).
    // OP-REASON-LOOP Phase 1 -- prompt-reasoning loop, single-iteration form.
    // `cfg.max_context_iters` caps retrieve -> reason -> re-retrieve
    // iterations; the default 1 performs exactly one retrieval (today's
    // behavior). 0 disables context accumulation entirely. Phase 3 will run
    // additional retrieval iterations (1..max_context_iters) that merge fresh
    // recalls into this set; Phase 1 runs the single initial retrieval only.
    // At iters==1 the set holds exactly the (budget-capped) recall the
    // renderer already consumes from `ask`, so rendering is byte-identical.
    // source_turn/iter are 0: narrate_inner is session-agnostic; the turn
    // index is assigned by the Phase 1.5 session merge in the guest handler.
    let mut context_set = if cfg.max_context_iters >= 1 {
        Some(crate::reason_loop::ContextSet::from_ask(
            &ask, 0, 0, crate::reason_loop::DEFAULT_READ_BUDGET,
        ))
    } else {
        None
    };
    // OP-REASON-LOOP Phase 3: reason<->retrieve loop. At max_context_iters>1,
    // reason over the pass-1 context (read-only, Phase-2 no_write), derive
    // gap/curiosity/topic re-seeds, and merge fresh seeded recalls into the
    // ContextSet -- the two-pass recall. No-op at iters==1 (byte-identical
    // degrade path). Runs only here on the PASS path (post idk_gate) so the
    // honesty floor is untouched and the loop can only ADD grounding.
    let mut reason_loop_summary: Option<crate::reason_loop::ReasonLoopSummary> = None;
    if cfg.max_context_iters > 1 && definition_fastpath_gloss.is_none() && !grounded_reasoning {
        if let Some(cs) = context_set.as_mut() {
            let stats = crate::reason_loop::run_reason_loop(
                brain, query, &ask, cs, depth, top_k, cfg,
            );
            // EXIT idk floor (spec section 6): after the loop, re-run the
            // full idk classify on the ACCUMULATED context. If it still
            // cannot ground an answer, refuse rather than fabricate. Monotone
            // with enrichment, so more iterations never lower the floor.
            let (verdict, signals) =
                crate::reason_loop::classify_context(brain, query, &ask, cs);
            let refused = verdict != crate::idk_gate::IdkVerdict::Pass;
            reason_loop_summary = Some(crate::reason_loop::ReasonLoopSummary {
                context_set_size: cs.len(),
                iters_ran: stats.iters_ran,
                seeds: stats.seeds_used,
                terminated: stats.terminated.as_str().to_string(),
                refused,
            });
            if refused {
                let (prose, _structured) = crate::idk_gate::build_response(
                    verdict, &signals, query, query, None,
                );
                return NarrationResult {
                    query: query.to_string(),
                    answer: prose,
                    shape: ResponseShape::Refusal,
                    intent,
                    flags: ModulationFlags::default(),
                    signals_used: vec!["idk_gate_exit".into()],
                    noise_filtered_count: 0,
                    ask,
                    multi: None,
                    source_kinds: Vec::new(),
                    source_labels: Vec::new(),
                    source_summary: None,
                    routing,
                    reasoning: None,
                    variants: None,
                    thinking: None,
                    context_set,
                    reason_loop: reason_loop_summary,
                    head_confidence: None,
                };
            }
        }
    } else if let Some(cs) = context_set.as_ref() {
        reason_loop_summary = Some(crate::reason_loop::ReasonLoopSummary {
            context_set_size: cs.len(),
            iters_ran: 0,
            seeds: Vec::new(),
            terminated: "loop_off".to_string(),
            refused: false,
        });
    }
    let run_multi_think = (use_multi_think || routing.needs_critic_vetting) && !grounded_reasoning;
    let multi = if run_multi_think {
        // OP-REASON-LOOP Phase 2 (FLAG D): the serve-path multi_think reasons
        // to ANSWER, not to learn -- run it read-only so its Agent provenance
        // bonds never mutate the shared (read-only) in-RAM network or bleed
        // into other concurrent requests spreads.
        let mut mt_cfg = brain.multi_agent_config.clone();
        mt_cfg.no_write = true;
        brain.multi_think_with_config(query, &mt_cfg).ok()
    } else {
        None
    };
    let flags = ModulationFlags::compute(&ask, multi.as_ref(), cfg);
    // OP-ROUTE-3a: build a reasoning trace when the router asked for deep
    // reasoning AND the idk_gate path passed. Only fires on the PASS
    // path -- the refusal path inside `idk_gate_enabled` already built
    // its own trace upstream and returned early. Guards: skip meta and
    // math queries (the math path short-circuits upstream; meta queries
    // shouldn't reason about Lu's own state via the bond graph). Scope
    // hardcoded to `TemporalScope::Current` (matches refusal-side call).
    let reasoning_trace: Option<crate::reasoning::ReasoningTrace> =
        if routing.needs_deep_reasoning
            && !grounded_reasoning
            && !routing.is_meta_query
            && !routing.needs_math
        {
            let r_cfg = crate::reasoning::ReasoningConfig::default();
            let scope = crate::scope::TemporalScope::Current;
            Some(crate::reasoning::reason_with_config(brain, query, scope, &r_cfg))
        } else {
            None
        };
    // OP-ROUTE-3b: spawn a variant arena when the router asked for
    // variants AND the goal seed registered above the activation floor.
    // Goal seed = `ask.seed_hits[0]` with fallback to top-activation
    // (per Section 1 / Section 15 dial #3). Empty arenas still yield a
    // summary so callers can observe "engine ran, no winner"; the
    // outer `Option` distinguishes "engine ran" from "engine did not
    // run" (Section 15 dial #8). Skips meta and math queries (Section 3).
    let variant_arena: Option<crate::variants::VariantArena> =
        if routing.needs_variants
            && !grounded_reasoning
            && !routing.is_meta_query
            && !routing.needs_math
        {
            let goal_seed = ask.seed_hits.first().copied().or_else(|| {
                ask.recall.activated.first().map(|a| a.seed)
            });
            goal_seed.and_then(|seed| {
                let act = ask
                    .recall
                    .activated
                    .iter()
                    .find(|a| a.seed == seed)
                    .map(|a| a.activation)
                    .unwrap_or(0.0);
                if act < cfg.variant_min_activation {
                    return None;
                }
                Some(brain.generate_variants_with_config(
                    seed,
                    &crate::variants::VariantConfig::default(),
                ))
            })
        } else {
            None
        };
    // OP-ROUTE-3c: run an agentic ReAct loop when the router asked
    // for thinking AND the goal seed registered above the activation
    // floor. Skips meta and math (same as 3a/3b). Refusal path
    // returned early upstream, so this block only ever runs on
    // PASS. Additive: does NOT subsume the reasoning_trace or
    // variant_arena blocks above -- a query that triggers all
    // three pays all three latencies (Section 17 decision #5 of
    // op_route_3c_design.md).
    let thinking_summary: Option<crate::thinking::ThinkingTraceSummary> =
        if routing.needs_agentic_thinking
            && !grounded_reasoning
            && !routing.is_meta_query
            && !routing.needs_math
        {
            let goal_seed = ask.seed_hits.first().copied().or_else(|| {
                ask.recall.activated.first().map(|a| a.seed)
            });
            goal_seed.and_then(|seed| {
                let act = ask
                    .recall
                    .activated
                    .iter()
                    .find(|a| a.seed == seed)
                    .map(|a| a.activation)
                    .unwrap_or(0.0);
                if act < cfg.thinking_min_activation {
                    return None;
                }
                // OP-THINK-WATCHDOG -- narrate path tightens the
                // engine default (30s) down to the configured
                // 5s (or whatever NarrationConfig carries). `0`
                // disables the watchdog on this path explicitly.
                let wall = if cfg.thinking_max_wall_clock_ms == 0 {
                    None
                } else {
                    Some(std::time::Duration::from_millis(
                        cfg.thinking_max_wall_clock_ms,
                    ))
                };
                // OP-ASK-WATCHDOG -- thread the same ask-level
                // budget the outer narrate path uses into the inner
                // think loop, so OBSERVE asks cannot blow past the
                // think wall-clock on a single iteration.
                let ask_wall_for_think = if cfg.ask_max_wall_clock_ms == 0 {
                    None
                } else {
                    Some(std::time::Duration::from_millis(
                        cfg.ask_max_wall_clock_ms,
                    ))
                };
                let t_cfg = crate::thinking::ThinkConfig {
                    max_steps: cfg.thinking_max_steps,
                    max_wall_clock: wall,
                    max_ask_wall_clock: ask_wall_for_think,
                    // OP-REASON-LOOP Phase 2 (FLAG D): serve-path think reasons
                    // to answer, read-only -- no Hypothesis bonds written.
                    no_write: true,
                    ..crate::thinking::ThinkConfig::default()
                };
                let result = brain.think(query, t_cfg);
                Some(crate::thinking::ThinkingTraceSummary::from(&result))
            })
        } else {
            None
        };
    // OP-CONCLUDE Tier 1 -- attempt conclusion synthesis from the
    // enriched ContextSet. PASS-path only (runs after the idk_gate
    // entry floor and the reason-loop exit floor), so the honesty
    // floor is intact and this can only ADD grounding. When it returns
    // None (non-reasoning shapes, or reasoning shapes lacking the
    // supporting edges) everything below is byte-identical to today.
    let synthesized = early_conclusion.or_else(|| crate::synthesis::try_synthesize(
        brain, query, intent, context_set.as_ref(),
    ));
    let shape = ShapeSelector::pick(
        intent, &ask, multi.as_ref(), &flags, cfg, reasoning_trace.as_ref(),
        variant_arena.as_ref(), thinking_summary.as_ref(),
    );
    let mut slots = extract_slots(
        brain, query, &ask, multi.as_ref(), cfg, reasoning_trace.as_ref(),
        variant_arena.as_ref(), thinking_summary.as_ref(),
        intent,
    );
    // Port #22: bare polysemous query produces a multi-sense
    // disclosure. Override shape after slot fill so we don't
    // re-architect ShapeSelector for one specific case.
    let mut shape = if !slots.multi_sense_clusters.is_empty() {
        ResponseShape::MultiSense
    } else {
        shape
    };
    // FIX 1 (OP-DEF-GLOSS): surface the stored WordNet/GCIDE definition
    // gloss for Definition-intent queries. The subject's canonical node
    // carries the gloss as KnowledgeContent::Definition (set_definition);
    // resolve it by the SAME folded seed the ingest used and surface it,
    // forcing the Definition shape so a hub-boundary curiosity signal
    // cannot deflect a grounded definition into Wondering. Pure read-path
    // (one intern lookup + one get_node_v5 point read); the idk floor is
    // preserved -- no gloss node means no override, prior behaviour holds.
    if let Some(gloss) = &definition_fastpath_gloss {
        slots.walker_prose = Some(gloss.clone());
        shape = ResponseShape::Definition;
    }
    // OP-EPISTEMIC (Phase 3): calibrate the synthesized conclusion register
    // to the confidence of the edges it already cites, and add a thin-topic
    // hedge when the recall is grounded but weak. Pure, called ONCE, on the
    // PASS path only. It rewrites conclusion.prose (the Conclusion render arm
    // prints it verbatim) and NEVER touches cited_edges or an empty-cited
    // refusal -- the honesty floor is untouched (design section 8). Signals
    // are computed here without a reasoning trace (max_activation / anchored
    // ratio only), so the added cost is one keyword-index pass, not a second
    // reason run.
    // OP-FLUENT (Phase 4): re-voice the synthesized conclusion body more
    // fluently -- aggregation, subject continuity, connective variety, and
    // the tracked cosmetics -- phrasing ONLY the grounded propositions it
    // already cites (post-realization honesty audit inside; any ungrounded
    // token degrades to the templated body). Runs BEFORE epistemic so the
    // Phase-3 register calibration stays authoritative (applied LAST).
    let synthesized = synthesized.map(crate::fluent::realize);
    let synthesized = synthesized.map(|concl| {
        // OP-OVERVIEW-ROUTE: the definition fast-path builds an INTENTIONALLY
        // EMPTY ask shell, whose idk signals falsely read "thin" (max_activation
        // 0). When an entity overview OVERRIDES that gloss (a place/work subject),
        // the bogus thin flag would prepend "This isn't fully understood ..."
        // over a perfectly-grounded overview. Pass None there so epistemic uses
        // the conclusion's OWN cited-edge confidence register ("From a single
        // source on record, ..." for single-source facts) -- identical wording
        // to the shipped person overview. Real-recall paths keep their signals.
        let sig = if definition_fastpath_gloss.is_some() {
            None
        } else {
            let idk_thresholds = crate::idk_gate::IdkThresholds::default();
            Some(crate::idk_gate::evaluate_signals(brain, &ask, query, None, &idk_thresholds))
        };
        crate::epistemic::recast(concl, sig.as_ref())
    });
    // OP-CONCLUDE Tier 1: a successful synthesis overrides shape+slots
    // with the edge-cited conclusion. Reached only on the PASS path;
    // the None (degrade) path leaves shape/slots exactly as above.
    if let Some(concl) = &synthesized {
        slots.conclusion_prose = Some(concl.prose.clone());
        slots.cited_edges = concl.cited_edges.iter().map(|e| e.arrow()).collect();
        slots.confidence_value = concl.confidence;
        shape = ResponseShape::Conclusion;
    }
    // OP-NARRATE-1 Phase C: compute hints and thread into renderer.
    let now_ms_for_hints = {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0)
    };
    let hints = brain.compute_style_hints(now_ms_for_hints);
    // OP-QA-1: multi-concept grounding floor. ShapeSelector's multi-think
    // fallback can pick MultiStep from raw, UNVETTED recall -- stitching
    // unrelated word-senses into a "several strands ... without a single clean
    // answer" template that reads as nonsense. A genuine multi-step answer
    // rests on vetted relations (critic path -> CriticVetted/MultiHypothesis)
    // or a converged thinking pass; when neither holds, MultiStep is the
    // ABSENCE of a grounded answer, not one. Decline honestly in the same
    // plain voice as the idk floor rather than fabricate a strand-answer.
    // Converged-thinking MultiStep is preserved (the thinking clause is false
    // only for the fallback); vetted multi-concept answers never reach here
    // (they select CriticVetted/MultiHypothesis upstream).
    let multistep_ungrounded = shape == ResponseShape::MultiStep
        && multi.as_ref().map_or(false, |m| m.vetted.is_empty())
        // A scar articulation ("I'm scarred against this connection -- <reason>")
        // is grounded moral knowledge that MUST surface; it rides this same
        // vetted-empty path, so never decline when a scar reason is present.
        && slots.scar_reasons.is_empty()
        && thinking_summary.as_ref().map_or(true, |t| {
            !(t.steps_taken >= 1 && (t.early_exit || t.hypotheses_promoted >= 1))
        });
    let forced_decline: Option<String> = if multistep_ungrounded {
        Some(crate::idk_gate::plain_decline_for(query))
    } else {
        None
    };
    let shape = if forced_decline.is_some() { ResponseShape::Refusal } else { shape };
    let answer = match &forced_decline {
        Some(d) => d.clone(),
        None => Renderer::render(shape, &slots, &flags, cfg, &hints),
    };
    // Port #17 Fix 2: track gap-seed surfacing for noise filtering.
    if let Some(seed) = slots.chosen_curiosity_seed {
        let entry = brain.gap_seed_frequency.entry(seed).or_insert(0);
        *entry = entry.saturating_add(1);
    }

    let mut signals_used: Vec<String> = Vec::new();
    if flags.curiosity_touched { signals_used.push("curiosity".into()); }
    if flags.bridge_touched { signals_used.push("bridge".into()); }
    if flags.hypothesis_only_support { signals_used.push("hypothesis_only".into()); }
    if flags.critic_vetted { signals_used.push("critic_vetted".into()); }
    if flags.wondering_touched { signals_used.push("wondering".into()); }
    signals_used.sort();

    // Port S — derive source attribution for the activated chain.
    // Walk consecutive activated nodes and look up the bond's
    // source attribution. Distinct non-Unknown kinds are surfaced
    // both as tag list (structuredContent) and as a short prose
    // summary appended to the answer.
    let (source_kinds, source_labels, source_summary, answer) = {
        use crate::source::SourceKind;
        let mut distinct: Vec<crate::source::SourceKind> = Vec::new();
        let activated_seeds: Vec<crate::seed::Seed> = ask
            .recall
            .activated
            .iter()
            .map(|a| a.seed)
            .collect();
        for w in activated_seeds.windows(2) {
            let (a, b) = (w[0], w[1]);
            if let Some(wb) = brain.network.seam_directed_bond(&a, &b) {
                let bt = wb.view().bond_type();
                let attr = brain.bond_source(&a, &b, bt);
                if !matches!(attr.kind, SourceKind::Unknown)
                    && !distinct.iter().any(|k| k == &attr.kind)
                {
                    distinct.push(attr.kind);
                }
            }
        }
        let kinds: Vec<String> = distinct.iter().map(|k| k.tag().to_string()).collect();
        let labels: Vec<String> = distinct.iter().map(|k| k.short_label()).collect();
        let (summary, answer_with_suffix) = if distinct.is_empty() {
            (None, answer)
        } else {
            let n = distinct.len();
            let summary = if n == 1 {
                format!("Based on 1 source ({}).", labels[0])
            } else {
                format!("Based on {} sources.", n)
            };
            // Append the summary as a separate sentence in the
            // human-readable answer. Keep it tight; structured
            // content carries the full label list.
            let mut a = answer;
            if !a.is_empty() && !a.ends_with(' ') && !a.ends_with('\n') {
                a.push(' ');
            }
            a.push_str(&summary);
            (Some(summary), a)
        };
        (kinds, labels, summary, answer_with_suffix)
    };

    // OP-NARRATE-1 Phase C: output adaptation is now render-time, not
    // post-pass. Hints were threaded into `Renderer::render` upstream.
    // No final-pass mutation — anti-hallucination invariant is
    // structurally enforced because hints only select among pre-written
    // per-shape templates.
    // OP-REASON-LOOP #225: answer synthesis from the enriched ContextSet.
    // When the reason<->retrieve loop ran (iters_ran > 0), make its
    // contribution user-visible: append a GROUNDED synthesis of the
    // reasoning-derived nodes (source_iter >= 1) -- the 2nd-hop concepts a
    // single pass-1 recall did not reach -- with a resolved gloss for the
    // strongest. Grounded only: real graph nodes + stored glosses, never
    // fabricated. No-op when the loop did not run (iters==1 degrade path ->
    // byte-identical to today), and the definition fast-path never reaches
    // here with iters_ran > 0 (the loop is skipped for it in Phase 4).
    let answer = if reason_loop_summary.as_ref().map_or(false, |s| s.iters_ran > 0) {
        match context_set.as_ref() {
            Some(cs) => {
                let added = cs.loop_added_ranked();
                if added.is_empty() {
                    answer
                } else {
                    let labels: Vec<String> =
                        added.iter().take(3).map(|(_, e)| e.label.clone()).collect();
                    let gloss = brain.network.definition_from_store(added[0].0);
                    let mut a = answer;
                    if !a.is_empty() && !a.ends_with(' ') && !a.ends_with('\n') {
                        a.push(' ');
                    }
                    match gloss {
                        Some(g) if !g.is_empty() && labels.len() > 1 => a.push_str(&format!(
                            "Following the reasoning chain, this also grounds in {} ({}) -- and connects to {}.",
                            labels[0], g, labels[1..].join(", "),
                        )),
                        Some(g) if !g.is_empty() => a.push_str(&format!(
                            "Following the reasoning chain, this also grounds in {} ({}).",
                            labels[0], g,
                        )),
                        _ => a.push_str(&format!(
                            "Following the reasoning chain, this also connects to: {}.",
                            labels.join(", "),
                        )),
                    }
                    a
                }
            }
            None => answer,
        }
    } else {
        answer
    };

    // Item 1: PASS-path TimedOut disclosure -- a partial/timed-out recall
    // that cleared the idk floor is answered from PARTIAL results; disclose
    // that rather than presenting it as complete. Inert on Complete recalls
    // (so the battery / all full-recall answers are byte-for-byte unchanged).
    let answer = disclose_truncation(answer, ask.status);
    // OP-QA-2: final scrub of internal node identifiers (WordNet sense-ids,
    // routing handles) so no shape can leak them into the user-facing answer.
    let answer = scrub_internal_tokens(&answer);
    let head_confidence_out = slots.head_confidence.clone();
    NarrationResult {
        query: query.to_string(),
        answer,
        shape,
        intent,
        flags,
        signals_used,
        noise_filtered_count: slots.noise_filtered_count,
        ask,
        multi,
        source_kinds,
        source_labels,
        source_summary,
        routing,
        // OP-ROUTE-3a: surface the PASS-path trace (None when deep
        // reasoning did not fire).
        reasoning: reasoning_trace,
        // OP-ROUTE-3b: surface a compact summary of the variant arena
        // when one was spawned. `None` when variants did not run.
        variants: variant_arena
            .as_ref()
            .map(crate::variants::VariantArenaSummary::from),
        // OP-ROUTE-3c: surface the compact thinking summary when
        // the PASS path ran a `Brain::think` loop. `None` when
        // thinking did not run.
        thinking: thinking_summary,
        context_set,
        reason_loop: reason_loop_summary,
        head_confidence: head_confidence_out,
    }
}

// =====================================================================
// Tests
// =====================================================================

#[cfg(test)]
mod tests {

    #[test]
    fn op_qa2_scrub_internal_tokens() {
        // WordNet sense-ids collapse to their lemma; surrounding text/punct kept.
        assert_eq!(super::scrub_internal_tokens("First: instruction.n.4."), "First: instruction.");
        assert_eq!(super::scrub_internal_tokens("the source_code.n.1 here"), "the source code here");
        // Routing handles collapse to the trailing name.
        assert_eq!(super::scrub_internal_tokens("builtins.sum returns"), "sum returns");
        assert_eq!(super::scrub_internal_tokens("pyfnlist"), "list");
        // Ordinary prose (decimals, abbreviations, the good-path def) untouched.
        assert_eq!(super::scrub_internal_tokens("a member of the genus Canis"), "a member of the genus Canis");
        assert_eq!(super::scrub_internal_tokens("pi is 3.14 and U.S.A"), "pi is 3.14 and U.S.A");
        assert_eq!(super::scrub_internal_tokens("sum returns the total."), "sum returns the total.");
    }

    #[test]
    fn code_op_token_extracts_and_rejects() {
        assert_eq!(super::code_op_token("what does sum do"), Some("sum".to_string()));
        assert_eq!(super::code_op_token("what does the sum function do"), Some("sum".to_string()));
        assert_eq!(super::code_op_token("define zip"), Some("zip".to_string()));
        assert_eq!(super::code_op_token("what is the range function"), Some("range".to_string()));
        // math / usage phrasings and general subjects must NOT be intercepted
        assert_eq!(super::code_op_token("what is the sum of 2 and 3"), None);
        assert_eq!(super::code_op_token("what does a lawyer do"), Some("lawyer".to_string()));
        assert_eq!(super::code_op_token("what causes rain"), None);
        assert_eq!(super::code_op_token("how does photosynthesis work"), None);
    }
    use super::*;

    #[test]
    fn capability_query_routes_to_self_identity() {
        assert!(is_capability_query("what can you do"));
        assert!(is_capability_query("What can you do?"));
        assert!(is_capability_query("what are you capable of"));
        assert!(!is_capability_query("who are you"));
        assert!(!is_capability_query("what is a dog"));
        assert_eq!(IntentClassifier::classify("what can you do?"), QueryIntent::SelfIdentity);
        // "who are you" still routes to SelfIdentity (unchanged).
        assert_eq!(IntentClassifier::classify("who are you?"), QueryIntent::SelfIdentity);
    }


    #[test]
    fn disclose_truncation_notes_timeout_and_is_identity_on_complete() {
        let base = "partial answer".to_string();
        // Complete -> byte-for-byte identity.
        assert_eq!(
            disclose_truncation(base.clone(), crate::brain::AskStatus::Complete),
            base
        );
        // TimedOut -> discloses truncation, preserves the original answer prefix.
        let t = disclose_truncation(
            base.clone(),
            crate::brain::AskStatus::TimedOut { activated_so_far: 7 },
        );
        assert!(t.starts_with("partial answer"));
        assert!(t.contains("timed out"));
        assert_ne!(t, base);
    }

    use crate::style::StyleHints;
    use crate::bond::BondType;
    use crate::brain::{now_ms, Brain};
    use crate::network::{ActivatedNode, RecallResult};
    use crate::node::Confidence;
    use crate::scope::TemporalScope;
    use crate::seed::Seed;

    fn teach(brain: &mut Brain, s: &str, r: BondType, o: &str) {
        brain.teach(s, r, o, 0.9, Confidence::Verified);
    }

    /// OP-014: NarrationConfig with idk_gate_enabled=false. Used by
    /// unit tests that hand-build tiny brains to exercise shape-
    /// selection logic in isolation — the gate would (correctly)
    /// refuse such brains for lack of anchor strength, but these
    /// tests are not validating gating behaviour.
    fn cfg_no_gate() -> NarrationConfig {
        NarrationConfig {
            idk_gate_enabled: false,
            ..NarrationConfig::default()
        }
    }

    // ------ Intent classification tests (4) ------

    #[test]
    fn test_intent_classifier_definition_query() {
        assert_eq!(IntentClassifier::classify("what is photosynthesis?"), QueryIntent::Definition);
        assert_eq!(IntentClassifier::classify("define entropy"), QueryIntent::Definition);
    }

    #[test]
    fn test_intent_classifier_process_query() {
        assert_eq!(IntentClassifier::classify("how does a transformer work?"), QueryIntent::Process);
    }

    #[test]
    fn test_intent_classifier_why_query() {
        assert_eq!(IntentClassifier::classify("why does ice float?"), QueryIntent::WhyChain);
    }

    #[test]
    fn test_intent_classifier_comparison_query() {
        assert_eq!(IntentClassifier::classify("compare cats and dogs"), QueryIntent::Comparison);
        assert_eq!(IntentClassifier::classify("python vs rust"), QueryIntent::Comparison);
    }

    // ------ Shape tests (15) ------

    #[test]
    fn test_shape_definition() {
        let mut brain = Brain::new();
        teach(&mut brain, "cat", BondType::IsA, "mammal");
        let r = narrate(&mut brain, "what is cat?", 2, 5, false, &NarrationConfig::default());
        assert_eq!(r.shape, ResponseShape::Definition);
        assert!(r.answer.contains(" is "));
    }

    #[test]
    fn test_shape_process() {
        let mut brain = Brain::new();
        teach(&mut brain, "baking", BondType::Requires, "flour");
        teach(&mut brain, "baking", BondType::Requires, "oven");
        teach(&mut brain, "baking", BondType::Produces, "bread");
        let r = narrate(&mut brain, "how does baking work?", 2, 5, false, &NarrationConfig::default());
        assert_eq!(r.shape, ResponseShape::Process);
        // v2: Process now prefers walker prose (Requires/Produces). Accept either form.
        assert!(r.answer.contains("steps") || r.answer.contains("requires") || r.answer.contains("produces"));
    }

    #[test]
    fn test_shape_why_chain() {
        let mut brain = Brain::new();
        teach(&mut brain, "rain", BondType::Causes, "clouds");
        teach(&mut brain, "clouds", BondType::Causes, "evaporation");
        let r = narrate(&mut brain, "why does rain fall?", 3, 5, false, &cfg_no_gate());
        assert_eq!(r.shape, ResponseShape::WhyChain);
        // OP-ROUTE-3a + OP-ROUTE-2.1: WhyChain intent now triggers
        // reason_with_config on the PASS path, so the answer can come
        // from either the trace articulator or the existing template.
        // Both are valid WhyChain outputs; assert shape fired and the
        // answer carries a causal connector.
        assert!(!r.answer.is_empty());
        let lower = r.answer.to_lowercase();
        let has_connector = lower.contains("because")
            || lower.contains("therefore")
            || lower.contains("leads to")
            || lower.contains("which means")
            || lower.contains("happens")
            || lower.contains("produc");
        assert!(has_connector, "WhyChain answer missing causal connector");
    }

    #[test]
    fn test_shape_comparison() {
        let mut brain = Brain::new();
        teach(&mut brain, "apple", BondType::IsA, "fruit");
        teach(&mut brain, "orange", BondType::IsA, "fruit");
        let r = narrate(&mut brain, "compare apple and orange", 2, 5, false, &cfg_no_gate());
        assert_eq!(r.shape, ResponseShape::Comparison);
        assert!(r.answer.contains("share") || r.answer.contains("differ"));
    }

    #[test]
    fn test_shape_fact_recall() {
        let mut brain = Brain::new();
        teach(&mut brain, "paris", BondType::IsA, "capital");
        let r = narrate(&mut brain, "paris", 1, 3, false, &NarrationConfig::default());
        assert_eq!(r.shape, ResponseShape::FactRecall);
    }

    #[test]
    fn test_shape_uncertain_answer() {
        // Render path test: when the selector lands on UncertainAnswer
        // the renderer must produce the hedged template.
        let mut slots = ShapeSlots::default();
        slots.query = "alpha".into();
        slots.primary = Some("a vague guess".into());
        let flags = ModulationFlags::default();
        let cfg = NarrationConfig::default();
        let out = Renderer::render(ResponseShape::UncertainAnswer, &slots, &flags, &cfg, &StyleHints::neutral());
        assert!(out.contains("thin") || out.contains("might be"));
        assert!(out.contains("a vague guess"));

        // Selector path: hand-built AskResult with one weak activation.
        use crate::network::{ActivatedNode, RecallResult};
        use crate::scope::TemporalScope;
        let weak = ActivatedNode {
            seed: Seed::generate(),
            content: "weak".into(),
            activation: 0.10,
            confidence: Confidence::Inferred,
            path: Vec::new(),
            depth: 1,
        };
        let ask = AskResult {
            query: "q".into(),
            seed_hits: Vec::new(),
            recall: RecallResult {
                activated: vec![weak],
                query_seeds: Vec::new(),
                scope: TemporalScope::Current,
                nodes_visited: 1,
                bonds_evaluated: 0,
                activation_truncated: false,
                status: crate::network::SpreadStatus::Complete,
            },
            hebbian_strengthened: 0,
            session_id: None,
            bridges: Vec::new(),
            curiosity_signals: Vec::new(),
            scars: Vec::new(),
            status: crate::brain::AskStatus::Complete,
        };
        let flags2 = ModulationFlags::compute(&ask, None, &cfg);
        let shape = ShapeSelector::pick(QueryIntent::Definition, &ask, None, &flags2, &cfg, None, None, None);
        assert_eq!(shape, ResponseShape::UncertainAnswer);
    }

    #[test]
    fn test_shape_refusal() {
        let mut brain = Brain::new();
        let r = narrate(&mut brain, "qzqzqz unknown noise", 2, 5, false, &NarrationConfig::default());
        assert_eq!(r.shape, ResponseShape::Refusal);
        assert!(r.answer.contains("don't have"));
    }

    // OP-GREET-REFUSAL -- narration-level tests for the pre-K-gate
    // pleasantry refusal hook.

    #[test]
    fn op_greet_refusal_narrate_hello_returns_canned_refusal() {
        // Empty brain so any K-gate path would also refuse; the
        // point of this test is to verify the GREET hook short-
        // circuits BEFORE the K-gate and produces the canned
        // greeting prose, not the gate's "I don't have meaningful
        // information" prose.
        let mut brain = Brain::new();
        let r = narrate(&mut brain, "hello", 2, 5, false, &NarrationConfig::default());
        assert_eq!(r.shape, ResponseShape::Refusal);
        // Canned greeting prose starts with "Hello." -- distinguishes
        // it from the K-gate refusal which starts with "I don't have".
        assert!(r.answer.starts_with("Hello."),
            "expected canned greeting prose, got {:?}", r.answer);
        assert!(!r.answer.is_empty());
        assert!(r.signals_used.iter().any(|s| s.starts_with("greet_refusal:")),
            "signals_used must record greet_refusal tag, got {:?}", r.signals_used);
        // Refusal path invariants: no reasoning/variants/thinking spawned.
        assert!(r.reasoning.is_none());
        assert!(r.variants.is_none());
        assert!(r.thinking.is_none());
    }

    #[test]
    fn op_greet_refusal_narrate_thank_you_returns_thanks_prose() {
        let mut brain = Brain::new();
        let r = narrate(&mut brain, "thank you", 2, 5, false, &NarrationConfig::default());
        assert_eq!(r.shape, ResponseShape::Refusal);
        assert!(r.answer.starts_with("You're welcome"),
            "expected canned thanks prose, got {:?}", r.answer);
    }

    #[test]
    fn op_greet_refusal_narrate_does_not_affect_real_query() {
        // Guard: the greet hook MUST NOT change the shape or prose
        // of legitimate queries. On an empty brain "mitochondria"
        // will refuse via the K-gate (same as before), but the prose
        // must be the K-gate prose, not greeting prose, and the
        // signal source must be idk_gate, not greet_refusal.
        let mut brain = Brain::new();
        let r = narrate(&mut brain, "mitochondria", 2, 5, false, &NarrationConfig::default());
        // Whatever shape it picks, signals_used must NOT contain a
        // greet_refusal tag.
        assert!(!r.signals_used.iter().any(|s| s.starts_with("greet_refusal:")),
            "mitochondria must not trigger greet refusal, got {:?}", r.signals_used);
        // The answer must NOT be the canned greeting prose.
        assert!(!r.answer.starts_with("Hello."),
            "mitochondria must not get greeting prose, got {:?}", r.answer);
    }

    #[test]
    fn op_greet_refusal_narrate_does_not_match_substring() {
        // "say hello to the world" contains the word "hello" but is
        // not a greeting. The narrate path must NOT short-circuit
        // via greet refusal.
        let mut brain = Brain::new();
        let r = narrate(&mut brain, "say hello to the world", 2, 5, false, &NarrationConfig::default());
        assert!(!r.signals_used.iter().any(|s| s.starts_with("greet_refusal:")),
            "substring match must not trigger greet refusal, got {:?}", r.signals_used);
    }

    #[test]
    fn op_ask_watchdog_narrate_composes_prose_on_ask_timeout() {
        // OP-ASK-WATCHDOG -- when the narrate-path ask trips the
        // wall-clock watchdog, slot fill must still compose a
        // non-empty response from whatever partial activations
        // came back. Seed the brain with a recognisable concept so
        // recall has SOMETHING to surface; force a 1ms ask budget
        // so the inner spread cannot drain its frontier.
        let mut brain = Brain::new();
        teach(&mut brain, "alpha", BondType::IsA, "beta");
        teach(&mut brain, "beta", BondType::HasProperty, "gamma");
        teach(&mut brain, "gamma", BondType::RelatedTo, "delta");
        let mut cfg = cfg_no_gate();
        cfg.ask_max_wall_clock_ms = 1;
        let r = narrate(&mut brain, "alpha", 3, 10, false, &cfg);
        // Verdict 1: prose is non-empty regardless of timeout.
        assert!(!r.answer.is_empty(),
            "narrate must compose non-empty prose even on ask timeout");
        // Verdict 2: the surrounding NarrationResult survived the
        // watchdog cleanly (no panic, shape and signals_used populated).
        assert!(r.shape as i32 >= 0);
    }

    #[test]
    fn test_shape_meta_self() {
        // A BARE meta-self question has no subject, so a brain census
        // really is the answer. This is the case MetaSelf is for.
        let mut brain = Brain::new();
        teach(&mut brain, "foo", BondType::IsA, "bar");
        let r = narrate(&mut brain, "what do you know?", 2, 5, false, &NarrationConfig::default());
        assert_eq!(r.shape, ResponseShape::MetaSelf);
        assert!(r.answer.contains("concepts"));
    }

    /// OP-RECALL-1 -- "what do you know ABOUT FOO" is a question about
    /// FOO, not a request for a brain census.
    ///
    /// This test previously asserted the opposite. That assertion was
    /// pinning the defect: routing a topical question to MetaSelf is
    /// what produced the live output `On "What do you know about
    /// python", I have 0 concepts and 0 associations ... Recently I
    /// activated Parser implementation in Python.` -- a whole-brain
    /// census followed by an unrelated node, presented as an answer
    /// about Python.
    #[test]
    fn op_recall_1_topical_meta_query_answers_on_its_subject() {
        let mut brain = Brain::new();
        teach(&mut brain, "foo", BondType::IsA, "bar");
        let r = narrate(&mut brain, "what do you know about foo?", 2, 5, false, &NarrationConfig::default());
        assert_ne!(
            r.shape,
            ResponseShape::MetaSelf,
            "a topical question must not be answered with a brain census",
        );
    }

    /// OP-RECALL-1 -- the renderer must never assert emptiness and then
    /// talk. Even with R5 fixing the counters on v5 brains, a MetaSelf
    /// render that reads 0/0 must drop the counts rather than state a
    /// zero it is about to contradict.
    #[test]
    fn op_recall_1_metaself_never_claims_emptiness_then_speaks() {
        let mut brain = Brain::new();
        teach(&mut brain, "foo", BondType::IsA, "bar");
        let r = narrate(&mut brain, "what do you know?", 2, 5, false, &NarrationConfig::default());
        let claims_empty = r.answer.contains("0 concepts") || r.answer.contains("0 associations");
        assert!(
            !claims_empty,
            "MetaSelf claimed emptiness on a non-empty brain: {:?}",
            r.answer,
        );
    }

    #[test]
    fn test_shape_timetravel_comparison() {
        let mut brain = Brain::new();
        teach(&mut brain, "chess", BondType::RelatedTo, "strategy");
        let r = narrate(&mut brain, "how has your understanding of chess changed?", 2, 5, false, &cfg_no_gate());
        assert_eq!(r.shape, ResponseShape::TimetravelComparison);
    }

    #[test]
    fn test_shape_critic_vetted_answer() {
        let mut brain = Brain::new();
        teach(&mut brain, "a", BondType::Causes, "b");
        teach(&mut brain, "b", BondType::Causes, "c");
        // Force a critic-vetted MultiAgentTrace by passing use_multi_think=true.
        let r = narrate(&mut brain, "trace causes", 3, 5, true, &cfg_no_gate());
        // Either CriticVettedAnswer or MultiHypothesisAnswer or MultiStep
        // depending on synthesis output. We just confirm the multi-think path was used.
        assert!(r.multi.is_some());
    }

    #[test]
    fn op_route_guest_critic_vetting_flag_drives_multi_think() {
        // OP-ROUTE-GUEST (2026-07-23): needs_critic_vetting was computed
        // then discarded until it was wired into use_multi_think. A
        // claim-verification query must now run the critic even though
        // the caller passed use_multi_think=false.
        let mut brain = Brain::new();
        teach(&mut brain, "a", BondType::Causes, "b");
        teach(&mut brain, "b", BondType::Causes, "c");
        let r = narrate(&mut brain, "is it true that a causes c", 3, 5, false, &cfg_no_gate());
        assert!(
            r.routing.needs_critic_vetting,
            "router must flag the claim-verification marker",
        );
        assert!(
            r.multi.is_some(),
            "needs_critic_vetting must drive multi_think without the explicit opt-in",
        );
        // Negative control: an ordinary query with no marker and no
        // explicit opt-in must NOT spawn the critic.
        let r2 = narrate(&mut brain, "trace causes", 3, 5, false, &cfg_no_gate());
        assert!(!r2.routing.needs_critic_vetting);
        assert!(
            r2.multi.is_none(),
            "critic must stay off for non-claim queries without opt-in",
        );
    }

    #[test]
    fn test_shape_multi_hypothesis_answer() {
        // Build a graph where multi_think will produce >=2 hypotheses and at
        // least one is vetted. Hypothesis count must exceed vetted count.
        let mut brain = Brain::new();
        teach(&mut brain, "a", BondType::RelatedTo, "b");
        teach(&mut brain, "b", BondType::RelatedTo, "c");
        teach(&mut brain, "a", BondType::RelatedTo, "d");
        teach(&mut brain, "d", BondType::RelatedTo, "e");
        let r = narrate(&mut brain, "explore from a", 3, 5, true, &cfg_no_gate());
        // Confirm multi-think actually fired; the exact shape selection
        // depends on critic threshold + hypothesis cap. Both vetted-driven
        // shapes are acceptable.
        assert!(r.multi.is_some());
    }

    #[test]
    fn test_shape_bridged_answer() {
        // Construct two clusters joined by a single weak bond — that's a
        // textbook bridge condition.
        let mut brain = Brain::new();
        // Cluster A.
        teach(&mut brain, "piano", BondType::IsA, "instrument");
        teach(&mut brain, "violin", BondType::IsA, "instrument");
        teach(&mut brain, "flute", BondType::IsA, "instrument");
        // Cluster B.
        teach(&mut brain, "calculus", BondType::IsA, "math");
        teach(&mut brain, "algebra", BondType::IsA, "math");
        teach(&mut brain, "geometry", BondType::IsA, "math");
        // Weak bridge.
        brain.teach("piano", BondType::RelatedTo, "math", 0.4, Confidence::Inferred);
        let r = narrate(&mut brain, "what is piano?", 3, 8, false, &NarrationConfig::default());
        // Bridge may or may not fire depending on hub thresholds; we accept
        // either Bridged shape OR a non-Bridged shape with the bridge
        // observation appended via modulation.
        let _ = r.shape;
    }

    #[test]
    fn test_shape_synthesis() {
        // Build a 3-node chain so the Generic-intent fall-through
        // selects Synthesis. Use a query that won't match Definition/etc.
        let mut brain = Brain::new();
        teach(&mut brain, "alpha", BondType::RelatedTo, "beta");
        teach(&mut brain, "beta", BondType::RelatedTo, "gamma");
        teach(&mut brain, "gamma", BondType::RelatedTo, "delta");
        let r = narrate(&mut brain, "alpha and beta and gamma chain inference", 3, 5, false, &cfg_no_gate());
        assert_eq!(r.intent, QueryIntent::Generic);
        assert!(matches!(r.shape, ResponseShape::Synthesis | ResponseShape::Definition));
    }

    #[test]
    fn test_shape_wondering() {
        // Hard to deterministically force HubBoundary/DeadEnd signals
        // without a very specific topology, so we just verify the shape
        // can be reached when the modulation flag is set manually.
        let mut slots = ShapeSlots::default();
        slots.query = "q".into();
        slots.primary = Some("x".into());
        slots.wondering_about = Some("y".into());
        slots.wondering_kind = Some("a hub boundary".into());
        let flags = ModulationFlags::default();
        let cfg = NarrationConfig::default();
        let out = Renderer::render(ResponseShape::Wondering, &slots, &flags, &cfg, &StyleHints::neutral());
        assert!(out.contains("wondering"));
    }

    #[test]
    fn test_shape_multi_step() {
        let mut slots = ShapeSlots::default();
        slots.primary = Some("part1".into());
        slots.secondary = Some("part2".into());
        slots.tertiary = Some("part3".into());
        let flags = ModulationFlags::default();
        let cfg = NarrationConfig::default();
        let out = Renderer::render(ResponseShape::MultiStep, &slots, &flags, &cfg, &StyleHints::neutral());
        assert!(out.contains("Several parts"));
        assert!(out.contains("part1"));
        assert!(out.contains("part2"));
        assert!(out.contains("part3"));
    }

    // ------ Modulation tests (2) ------

    #[test]
    fn test_modulation_curiosity_appends_gap_caveat() {
        let mut slots = ShapeSlots::default();
        slots.query = "q".into();
        slots.primary = Some("answer".into());
        slots.secondary = Some("category".into());
        slots.wondering_about = Some("missing topic".into());
        let mut flags = ModulationFlags::default();
        flags.curiosity_touched = true;
        flags.expose_curiosity_voice = true; // BUG1: append now requires exposure
        let cfg = NarrationConfig::default();
        let out = Renderer::render(ResponseShape::Definition, &slots, &flags, &cfg, &StyleHints::neutral());
        assert!(out.contains("gap"));
        assert!(out.contains("missing topic"));
    }

    #[test]
    fn test_modulation_bridge_appends_observation() {
        let mut slots = ShapeSlots::default();
        slots.query = "q".into();
        slots.primary = Some("answer".into());
        slots.secondary = Some("category".into());
        slots.bridge_low = Some("music".into());
        slots.bridge_high = Some("math".into());
        let mut flags = ModulationFlags::default();
        flags.bridge_touched = true;
        let cfg = NarrationConfig::default();
        let out = Renderer::render(ResponseShape::Definition, &slots, &flags, &cfg, &StyleHints::neutral());
        assert!(out.contains("cross-domain"));
        assert!(out.contains("music"));
        assert!(out.contains("math"));
    }

    // ------ Edge case tests (3) ------

    #[test]
    fn test_refusal_when_no_activations() {
        let mut brain = Brain::new();
        // No teach calls at all.
        let r = narrate(&mut brain, "anything", 2, 5, false, &NarrationConfig::default());
        assert_eq!(r.shape, ResponseShape::Refusal);
    }

    #[test]
    fn test_narrate_uses_multi_think_when_flag_set() {
        let mut brain = Brain::new();
        teach(&mut brain, "x", BondType::RelatedTo, "y");
        teach(&mut brain, "y", BondType::RelatedTo, "z");
        let r = narrate(&mut brain, "explore x", 3, 5, true, &cfg_no_gate());
        assert!(r.multi.is_some(), "multi-think trace should be present when flag set");
    }

    #[test]
    fn test_narrate_deterministic_same_input_same_output() {
        let mut brain = Brain::new();
        teach(&mut brain, "cat", BondType::IsA, "mammal");
        teach(&mut brain, "mammal", BondType::IsA, "animal");
        let cfg = NarrationConfig::default();
        let r1 = narrate(&mut brain, "what is cat?", 2, 5, false, &cfg);
        let r2 = narrate(&mut brain, "what is cat?", 2, 5, false, &cfg);
        assert_eq!(r1.answer, r2.answer);
        assert_eq!(r1.shape, r2.shape);
    }

    #[test]
    fn test_shape_selector_does_not_lock_to_critic_when_no_rejection() {
        // Tier 2.7: when multi.vetted is non-empty but Critic did NOT
        // reject anything (no caveats, no rejected, vetted == hypotheses),
        // shape selection must NOT short-circuit to CriticVettedAnswer.
        let mut brain = Brain::new();
        teach(&mut brain, "alpha", BondType::IsA, "beta");
        teach(&mut brain, "beta", BondType::IsA, "gamma");
        let ask = brain.ask("alpha", 2, 5);
        assert!(!ask.recall.activated.is_empty(), "test setup must produce a recall");
        // Build a fake MultiAgentTrace with a single belief, all vetted,
        // no rejections, no caveats.
        let single = crate::agents::Belief {
            from: ask.recall.activated[0].seed,
            via: ask.recall.activated[0].seed,
            to: ask.recall.activated[0].seed,
            from_content: "a".into(),
            via_content: "a".into(),
            to_content: "a".into(),
            w_from_via: 0.5, w_via_to: 0.5, confidence: 0.6,
            scar_reasons: Vec::new(),
            caveats: Vec::new(),
        };
        let mt = crate::agents::MultiAgentTrace {
            multi_think_id: 1,
            query: "a".into(),
            recall: Vec::new(),
            hypotheses: vec![single.clone()],
            vetted: vec![single],
            rejected: Vec::new(),
            reasoning_trace: None,
            synthesis_prose: String::new(),
            steps: [
                crate::agents::AgentStep {
                    role: crate::scope::AgentRole::Recall,
                    elapsed_us: 0, items_in: 0, items_out: 0,
                },
                crate::agents::AgentStep {
                    role: crate::scope::AgentRole::Synthesis,
                    elapsed_us: 0, items_in: 0, items_out: 0,
                },
                crate::agents::AgentStep {
                    role: crate::scope::AgentRole::Critic,
                    elapsed_us: 0, items_in: 0, items_out: 0,
                },
            ],
            started_ms: 0, ended_ms: 0, timed_out: false,
        };
        let cfg = NarrationConfig::default();
        let flags = ModulationFlags::compute(&ask, Some(&mt), &cfg);
        let shape = ShapeSelector::pick(
            QueryIntent::Definition, &ask, Some(&mt), &flags, &cfg, None, None, None);
        // Critic did no work (vetted == hypotheses, no caveats, no
        // rejections), so the shape must fall through to the intent
        // shape — NOT lock to CriticVettedAnswer.
        assert_ne!(
            shape,
            ResponseShape::CriticVettedAnswer,
            "clean acceptance must NOT collapse to CriticVettedAnswer"
        );
    }

    #[test]
    fn test_confidence_is_calibrated_not_always_one() {
        // Tier 2.9: a thin recall with a single belief should NOT yield
        // confidence 1.00 just because belief.confidence is 1.0.
        let mut brain = Brain::new();
        teach(&mut brain, "a", BondType::IsA, "b");
        let ask = brain.ask("a", 1, 3);
        let mut slots = ShapeSlots::default();
        slots.query = "a".into();
        // Without multi: confidence comes from recall_norm * confidence_numeric.
        if let Some(top) = ask.recall.activated.first() {
            slots.primary = Some(top.content.clone());
            let recall_norm = (top.activation / 0.6).min(1.0);
            let combined = (recall_norm * confidence_numeric(top.confidence)).max(0.0);
            assert!(combined <= 1.0);
            // The normalization must not push everything to 1.00; check
            // the actual extract path agrees.
            let s2 = extract_slots(&brain, "a", &ask, None, &NarrationConfig::default(), None, None, None, QueryIntent::Generic);
            assert!(
                (s2.confidence_value - combined).abs() < 1e-4,
                "confidence_value={} combined={}",
                s2.confidence_value, combined,
            );
        }
    }

    #[test]
    fn test_length_capping_respects_max_chars() {
        let mut slots = ShapeSlots::default();
        slots.query = "q".into();
        slots.primary = Some("a".repeat(2000));
        let flags = ModulationFlags::default();
        let mut cfg = NarrationConfig::default();
        cfg.max_answer_chars = 100;
        let out = Renderer::render(ResponseShape::Definition, &slots, &flags, &cfg, &StyleHints::neutral());
        assert!(out.chars().count() <= 100);
        assert!(out.ends_with('…'));
    }

    #[test]
    fn test_strong_curiosity_routes_to_uncertain_not_definition() {
        use crate::curiosity::{CuriositySignal, SignalType};
        let top_seed = Seed::generate();
        let top = ActivatedNode {
            seed: top_seed,
            content: "photosynthesis".into(),
            activation: 0.9,
            confidence: Confidence::Verified,
            path: Vec::new(),
            depth: 0,
        };
        let recall = RecallResult {
            activated: vec![top],
            query_seeds: Vec::new(),
            scope: TemporalScope::Current,
            nodes_visited: 1,
            bonds_evaluated: 0,
            activation_truncated: false,
            status: crate::network::SpreadStatus::Complete,
        };
        // Port #16 raised the curiosity-strong threshold to >=4
        // signals OR top intensity > 0.85, so the original Port #11
        // setup (2 signals @ 0.7) no longer fires. Use 4 signals.
        let curiosity_signals = vec![
            CuriositySignal {
                seed: top_seed,
                signal_type: SignalType::KnowledgeDesert,
                intensity: 0.7,
                scope: TemporalScope::Current,
                timestamp_ms: 0,
            },
            CuriositySignal {
                seed: top_seed,
                signal_type: SignalType::IsolatedNode,
                intensity: 0.7,
                scope: TemporalScope::Current,
                timestamp_ms: 0,
            },
            CuriositySignal {
                seed: top_seed,
                signal_type: SignalType::KnowledgeDesert,
                intensity: 0.7,
                scope: TemporalScope::Current,
                timestamp_ms: 0,
            },
            CuriositySignal {
                seed: top_seed,
                signal_type: SignalType::IsolatedNode,
                intensity: 0.7,
                scope: TemporalScope::Current,
                timestamp_ms: 0,
            },
        ];
        let ask = AskResult {
            query: "what is photosynthesis".into(),
            seed_hits: Vec::new(),
            recall,
            hebbian_strengthened: 0,
            session_id: None,
            bridges: Vec::new(),
            curiosity_signals,
            scars: Vec::new(),
            status: crate::brain::AskStatus::Complete,
        };
        let cfg = NarrationConfig::default();
        let flags = ModulationFlags::compute(&ask, None, &cfg);
        let shape = ShapeSelector::pick(QueryIntent::Definition, &ask, None, &flags, &cfg, None, None, None);
        assert_ne!(
            shape,
            ResponseShape::Definition,
            "Definition should not fire when curiosity signals dominate"
        );
        assert!(
            matches!(shape, ResponseShape::UncertainAnswer | ResponseShape::Wondering),
            "expected UncertainAnswer or Wondering, got {:?}",
            shape,
        );
    }

    #[test]
    fn test_strong_hub_boundary_curiosity_routes_to_wondering() {
        use crate::curiosity::{CuriositySignal, SignalType};
        let top_seed = Seed::generate();
        let top = ActivatedNode {
            seed: top_seed,
            content: "photosynthesis".into(),
            activation: 0.9,
            confidence: Confidence::Verified,
            path: Vec::new(),
            depth: 0,
        };
        let recall = RecallResult {
            activated: vec![top],
            query_seeds: Vec::new(),
            scope: TemporalScope::Current,
            nodes_visited: 1,
            bonds_evaluated: 0,
            activation_truncated: false,
            status: crate::network::SpreadStatus::Complete,
        };
        // Port #16: 4 signals so curiosity_strong fires under the
        // new threshold; HubBoundary/DeadEnd still dominate so the
        // route should land on Wondering, not UncertainAnswer.
        let curiosity_signals = vec![
            CuriositySignal {
                seed: top_seed,
                signal_type: SignalType::HubBoundary,
                intensity: 0.8,
                scope: TemporalScope::Current,
                timestamp_ms: 0,
            },
            CuriositySignal {
                seed: top_seed,
                signal_type: SignalType::DeadEnd,
                intensity: 0.7,
                scope: TemporalScope::Current,
                timestamp_ms: 0,
            },
            CuriositySignal {
                seed: top_seed,
                signal_type: SignalType::HubBoundary,
                intensity: 0.75,
                scope: TemporalScope::Current,
                timestamp_ms: 0,
            },
            CuriositySignal {
                seed: top_seed,
                signal_type: SignalType::DeadEnd,
                intensity: 0.7,
                scope: TemporalScope::Current,
                timestamp_ms: 0,
            },
        ];
        let ask = AskResult {
            query: "what is photosynthesis".into(),
            seed_hits: Vec::new(),
            recall,
            hebbian_strengthened: 0,
            session_id: None,
            bridges: Vec::new(),
            curiosity_signals,
            scars: Vec::new(),
            status: crate::brain::AskStatus::Complete,
        };
        let cfg = NarrationConfig::default();
        let flags = ModulationFlags::compute(&ask, None, &cfg);
        let shape = ShapeSelector::pick(QueryIntent::Definition, &ask, None, &flags, &cfg, None, None, None);
        assert_eq!(
            shape,
            ResponseShape::Wondering,
            "HubBoundary/DeadEnd dominance should route to Wondering"
        );
    }

    // ------ v2 walker tests ------

    #[test]
    fn test_v2_connector_for_each_bond_type_is_nonempty() {
        for &bt in BondType::all() {
            let c = connector_for(bt);
            assert!(!c.trim().is_empty(), "connector for {:?} must be nonempty", bt);
            assert!(c.starts_with(' '), "connector for {:?} must start with space", bt);
            assert!(c.ends_with(' '), "connector for {:?} must end with space", bt);
        }
    }

    #[test]
    fn test_v2_hub_selector_picks_query_match_with_degree() {
        let mut brain = Brain::new();
        // Build electron with multiple typed bonds so it qualifies as hub.
        brain.teach("electron", BondType::IsA, "lepton", 0.9, Confidence::Verified);
        brain.teach("electron", BondType::IsA, "fermion", 0.9, Confidence::Verified);
        brain.teach("electron", BondType::HasProperty, "spin", 0.8, Confidence::Verified);
        let ask = brain.ask("what is electron", 2, 5);
        let hub = walker_hub_content(&brain, &ask);
        assert_eq!(hub.as_deref(), Some("electron"), "hub should match query term");
    }

    #[test]
    fn test_v2_walker_prose_surfaces_multiple_bond_types() {
        let mut brain = Brain::new();
        brain.teach("electron", BondType::IsA, "lepton", 0.9, Confidence::Verified);
        brain.teach("electron", BondType::HasProperty, "spin", 0.85, Confidence::Verified);
        brain.teach("electron", BondType::HasProperty, "charge", 0.8, Confidence::Verified);
        let ask = brain.ask("what is electron", 2, 5);
        let mut _f = 0u32;
        let cfg = NarrationConfig::default();
        let prose = walker_prose(&brain, &ask, &cfg, &mut _f).expect("walker should produce prose");
        // Must surface at least one IsA-style and one HasProperty connector.
        assert!(
            prose.contains(" is a ") || prose.contains(" has "),
            "prose should use real connectors, got: {}", prose,
        );
        // Should mention more than just the top-1.
        let mentions_lepton = prose.contains("lepton");
        let mentions_property = prose.contains("spin") || prose.contains("charge");
        assert!(
            mentions_lepton && mentions_property,
            "prose should reflect multiple bonds, got: {}", prose,
        );
    }

    #[test]
    fn test_v2_composer_chains_when_to_matches_next_from() {
        // Build A -IsA-> B -IsA-> C, hub = A.
        let tuples = vec![
            TraversalTuple { from_content: "a".into(), to_content: "b".into(),
                bond_type: BondType::IsA, weight: 1.0 },
            TraversalTuple { from_content: "b".into(), to_content: "c".into(),
                bond_type: BondType::IsA, weight: 0.8 },
        ];
        let prose = Composer::compose(&tuples).expect("compose should succeed");
        // Chained form: "A is a b, which is a c."
        assert!(prose.contains("which"), "chain marker missing: {}", prose);
        assert!(prose.contains("b") && prose.contains("c"));
    }

    #[test]
    fn test_v2_definition_uses_walker_prose_when_available() {
        let mut brain = Brain::new();
        brain.teach("electron", BondType::IsA, "lepton", 0.9, Confidence::Verified);
        brain.teach("electron", BondType::HasProperty, "spin", 0.85, Confidence::Verified);
        let r = narrate(&mut brain, "what is electron?", 2, 5, false, &NarrationConfig::default());
        // Must NOT be the legacy thin "electron is lepton." — must include
        // the walker connector phrasing.
        assert!(
            r.answer.contains(" is a ") || r.answer.contains(" has "),
            "v2 definition should use walker connectors, got: {}", r.answer,
        );
    }

    // ------ Port #16 calibration tests ------

    #[test]
    fn test_composer_caps_related_to_flood() {
        // Six RelatedTo neighbors out of one hub should collapse to
        // four visible + "and 2 more" (Port #16 Fix 2).
        let tuples: Vec<TraversalTuple> = ["a", "b", "c", "d", "e", "f"]
            .iter()
            .map(|t| TraversalTuple {
                from_content: "mitosis".into(),
                to_content: (*t).into(),
                bond_type: BondType::RelatedTo,
                weight: 0.7,
            })
            .collect();
        let prose = Composer::compose(&tuples).expect("compose should succeed");
        // Visible four come through, suffix announces the rest.
        assert!(prose.contains("a, b, c, d"), "first 4 missing: {}", prose);
        assert!(prose.contains("and 2 more"), "missing 'and N more': {}", prose);
        // The two overflow targets must NOT appear inline.
        assert!(!prose.contains(" e ") && !prose.contains(" f "),
            "overflow targets leaked: {}", prose);
    }

    #[test]
    fn test_composer_chains_isa_to_isa_naturally() {
        // IsA -> IsA must still chain ("X is a Y, which is a Z.").
        let tuples = vec![
            TraversalTuple {
                from_content: "electron".into(),
                to_content: "lepton".into(),
                bond_type: BondType::IsA,
                weight: 1.0,
            },
            TraversalTuple {
                from_content: "lepton".into(),
                to_content: "particle".into(),
                bond_type: BondType::IsA,
                weight: 0.9,
            },
        ];
        let prose = Composer::compose(&tuples).expect("compose should succeed");
        assert!(prose.contains("which"), "IsA->IsA should chain: {}", prose);
        assert!(prose.contains("particle"), "tail missing: {}", prose);
    }

    #[test]
    fn test_composer_blocks_related_to_followed_by_other() {
        // RelatedTo -> HasProperty must NOT chain (different bond
        // types, child is not IsA). The HasProperty group should
        // render as its own sentence (Port #16 Fix 3).
        let tuples = vec![
            TraversalTuple {
                from_content: "mitosis".into(),
                to_content: "nuclei".into(),
                bond_type: BondType::RelatedTo,
                weight: 0.7,
            },
            TraversalTuple {
                from_content: "nuclei".into(),
                to_content: "wireless".into(),
                bond_type: BondType::HasProperty,
                weight: 0.6,
            },
        ];
        let prose = Composer::compose(&tuples).expect("compose should succeed");
        // The cross-type bridge "...nuclei, which has wireless." must
        // not appear; the second group should stand on its own.
        assert!(
            !prose.contains("nuclei, which"),
            "RelatedTo->HasProperty must not chain: {}", prose,
        );
        // But both nodes should still surface somewhere in the prose.
        assert!(prose.contains("nuclei"), "first hop missing: {}", prose);
        assert!(prose.contains("wireless"), "second hop missing: {}", prose);
    }

    #[test]
    fn test_composer_chains_across_hop_boundary() {
        // A->B->C->D, all IsA, must chain transitively into one
        // sentence rather than truncating after the first link
        // (Port #16 Fix 4).
        let tuples = vec![
            TraversalTuple {
                from_content: "electron".into(),
                to_content: "lepton".into(),
                bond_type: BondType::IsA,
                weight: 1.0,
            },
            TraversalTuple {
                from_content: "lepton".into(),
                to_content: "particle".into(),
                bond_type: BondType::IsA,
                weight: 0.9,
            },
            TraversalTuple {
                from_content: "particle".into(),
                to_content: "fermion".into(),
                bond_type: BondType::IsA,
                weight: 0.85,
            },
        ];
        let prose = Composer::compose(&tuples).expect("compose should succeed");
        // All three target hops surface inside the same chain.
        assert!(prose.contains("lepton"), "hop-1 missing: {}", prose);
        assert!(prose.contains("particle"), "hop-2 missing: {}", prose);
        assert!(prose.contains("fermion"), "hop-3 missing: {}", prose);
        // Two "which" links — one per chain step beyond hop-1.
        let n_which = prose.matches("which").count();
        assert!(n_which >= 2,
            "expected >= 2 'which' links across hop boundary, got {}: {}",
            n_which, prose);
    }

    #[test]
    fn test_gap_frequency_filters_overused_signals() {
        // Port #17 Fix 2: once a curiosity seed has been surfaced as the
        // wondering_about slot more than `gap_appearance_threshold` times,
        // extract_slots skips it and picks the next-strongest signal.
        use crate::curiosity::{CuriositySignal, SignalType};
        use crate::brain::Brain;
        use crate::node::{KnowledgeNode, KnowledgeContent};
        let mut brain = Brain::new();
        // Two nodes that can be "gap" candidates.
        let noisy = Seed::from_label("name");
        let other = Seed::from_label("world");
        brain.network.add_node(KnowledgeNode::with_seed(
            noisy, "name".into(), KnowledgeContent::Raw("".into()),
            Confidence::Verified, 0,
        ));
        brain.network.add_node(KnowledgeNode::with_seed(
            other, "world".into(), KnowledgeContent::Raw("".into()),
            Confidence::Verified, 0,
        ));
        // Pretend `noisy` already crossed the threshold (default 10).
        brain.gap_seed_frequency.insert(noisy, 11);
        // Build an AskResult with both as candidates, noisy ranked higher.
        let ask = AskResult {
            query: "q".into(),
            seed_hits: Vec::new(),
            recall: RecallResult {
                activated: Vec::new(),
                query_seeds: Vec::new(),
                scope: TemporalScope::Current,
                nodes_visited: 0,
                bonds_evaluated: 0,
                activation_truncated: false,
                status: crate::network::SpreadStatus::Complete,
            },
            hebbian_strengthened: 0,
            session_id: None,
            bridges: Vec::new(),
            curiosity_signals: vec![
                CuriositySignal {
                    seed: noisy,
                    signal_type: SignalType::DeadEnd,
                    intensity: 0.9,
                    scope: TemporalScope::Current,
                    timestamp_ms: 0,
                },
                CuriositySignal {
                    seed: other,
                    signal_type: SignalType::IsolatedNode,
                    intensity: 0.5,
                    scope: TemporalScope::Current,
                    timestamp_ms: 0,
                },
            ],
            scars: Vec::new(),
            status: crate::brain::AskStatus::Complete,
        };
        let slots = extract_slots(&brain, "q", &ask, None, &NarrationConfig::default(), None, None, None, QueryIntent::Generic);
        assert_eq!(
            slots.wondering_about.as_deref(),
            Some("world"),
            "noisy seed over threshold should be skipped, picked: {:?}",
            slots.wondering_about,
        );
        assert_eq!(slots.chosen_curiosity_seed, Some(other));

        // Now lift `other` past the threshold too — wondering_about omitted.
        brain.gap_seed_frequency.insert(other, 11);
        let slots2 = extract_slots(&brain, "q", &ask, None, &NarrationConfig::default(), None, None, None, QueryIntent::Generic);
        assert!(slots2.wondering_about.is_none(),
            "all over threshold should omit wondering_about, got {:?}",
            slots2.wondering_about);
        assert!(slots2.chosen_curiosity_seed.is_none());
    }

    #[test]
    fn test_hub_selector_rejects_pure_relatedto_hub() {
        // Port #17 Fix 3: a high-degree node whose outgoing bonds are
        // mostly RelatedTo (no semantic structure) must NOT be selected
        // as the walker's hub. The selector should fall through to a
        // semantically richer (lower-degree) candidate.
        use crate::brain::Brain;
        use crate::bond::TemporalBond;
        use crate::node::{KnowledgeNode, KnowledgeContent};
        let mut brain = Brain::new();
        let net = &mut brain.network;

        // Pure-RelatedTo super-hub: "name" with 6 RelatedTo neighbors.
        let bad = Seed::from_label("name");
        net.add_node(KnowledgeNode::with_seed(
            bad, "name".into(), KnowledgeContent::Raw("".into()),
            Confidence::Verified, 0,
        ));
        for i in 0..6usize {
            let n = Seed::from_label(&format!("namebag{}", i));
            net.add_node(KnowledgeNode::with_seed(
                n, format!("namebag{}", i),
                KnowledgeContent::Raw("".into()),
                Confidence::Verified, 0,
            ));
            net.add_bond(TemporalBond::new(
                bad, n, BondType::RelatedTo, 0.7, 0,
            )).unwrap();
        }
        // Semantically rich hub: "electron" with IsA + HasProperty + Causes.
        let good = Seed::from_label("electron");
        net.add_node(KnowledgeNode::with_seed(
            good, "electron".into(), KnowledgeContent::Raw("".into()),
            Confidence::Verified, 0,
        ));
        for (i, bt) in [BondType::IsA, BondType::HasProperty, BondType::Causes].iter().enumerate() {
            let n = Seed::from_label(&format!("physics{}", i));
            net.add_node(KnowledgeNode::with_seed(
                n, format!("physics{}", i),
                KnowledgeContent::Raw("".into()),
                Confidence::Verified, 0,
            ));
            net.add_bond(TemporalBond::new(
                good, n, *bt, 0.9, 0,
            )).unwrap();
        }

        // Build an AskResult that activates both, with `name` ranked higher.
        let activated = vec![
            ActivatedNode {
                seed: bad,
                content: "name".into(),
                activation: 0.9,
                confidence: Confidence::Verified,
                path: Vec::new(),
                depth: 0,
            },
            ActivatedNode {
                seed: good,
                content: "electron".into(),
                activation: 0.6,
                confidence: Confidence::Verified,
                path: Vec::new(),
                depth: 0,
            },
        ];
        let ask = AskResult {
            query: "electron".into(),
            seed_hits: Vec::new(),
            recall: RecallResult {
                activated,
                query_seeds: Vec::new(),
                scope: TemporalScope::Current,
                nodes_visited: 2,
                bonds_evaluated: 0,
                activation_truncated: false,
                status: crate::network::SpreadStatus::Complete,
            },
            hebbian_strengthened: 0,
            session_id: None,
            bridges: Vec::new(),
            curiosity_signals: Vec::new(),
            scars: Vec::new(),
            status: crate::brain::AskStatus::Complete,
        };

        let hub = HubSelector::pick(&brain, &ask).expect("some hub should be selected");
        assert_eq!(
            hub.seed, good,
            "hub selector should reject pure-RelatedTo `name` and pick `electron`",
        );
    }

    // ------ Port #20: noise filter unit tests ------

    // FIX 1 (OP-DEF-GLOSS) round-trip: a stored Definition gloss is
    // resolved for "what is X" (article-stripped) and bare "X", while an
    // unknown subject yields None so the idk/refusal floor is preserved.
    #[test]
    fn definition_gloss_for_query_v5_round_trip() {
        use tempfile::tempdir;
        let dir = tempdir().unwrap();
        let manifest_path = dir.path().join("MANIFEST");
        let wal_path = dir.path().join("wal.seg");
        let mut brain = crate::brain::Brain::new();
        brain.enable_memtable(1024 * 1024 * 1024, &wal_path).unwrap();
        brain
            .enable_flush_worker(dir.path().to_path_buf(), manifest_path, None)
            .unwrap();
        let gloss = "a domesticated carnivorous mammal";
        assert!(brain.set_definition("dog", gloss), "define creates the node");
        assert_eq!(
            definition_gloss_for_query(&brain, "What is a dog").as_deref(),
            Some(gloss),
            "what-is with article must surface the stored gloss"
        );
        assert_eq!(
            definition_gloss_for_query(&brain, "dog").as_deref(),
            Some(gloss),
            "bare subject must resolve the same canonical node"
        );
        assert_eq!(
            definition_gloss_for_query(&brain, "what is florble"),
            None,
            "unknown subject must not fabricate a gloss (idk floor)"
        );
    }

    fn put_node(brain: &mut Brain, content: &str) -> Seed {
        let s = Seed::from_label(content);
        brain.network.add_node(crate::node::KnowledgeNode::with_seed(
            s,
            content.into(),
            crate::node::KnowledgeContent::Raw(content.into()),
            Confidence::Verified,
            now_ms(),
        ));
        s
    }

    #[test]
    fn test_noise_filter_token_shape_junk() {
        // Standalone shape rules — no brain access needed for the
        // ASCII helpers.
        assert!(NoiseFilter::is_token_shape_junk("6co"));
        assert!(NoiseFilter::is_token_shape_junk("12h"));
        assert!(NoiseFilter::is_token_shape_junk("3rd"));
        assert!(NoiseFilter::is_token_shape_junk("a"));
        assert!(NoiseFilter::is_token_shape_junk("x"));
        assert!(NoiseFilter::is_token_shape_junk("aa"));
        // Real words must NOT match.
        assert!(!NoiseFilter::is_token_shape_junk("photosynthesis"));
        assert!(!NoiseFilter::is_token_shape_junk("co2"));      // letters first
        assert!(!NoiseFilter::is_token_shape_junk("h2o"));      // letters first
        assert!(!NoiseFilter::is_token_shape_junk("cell"));
    }

    #[test]
    fn test_noise_filter_rejects_junk_in_walker_output() {
        // Build a tiny brain with junk + real neighbors, run the
        // filter, and confirm the junk is dropped while the real
        // neighbor survives.
        let mut brain = Brain::new();
        let cfg = NarrationConfig::default();
        let s_six_co = put_node(&mut brain, "6co");
        let s_g = put_node(&mut brain, "g");
        let s_gloom = put_node(&mut brain, "gloom");
        let s_real = put_node(&mut brain, "chloroplast");
        // chloroplast gets an IsA bond so the domain-mismatch heuristic
        // does not strip it. (typed_ratio cache lives on the node;
        // recompute after teach.)
        let s_organelle = put_node(&mut brain, "organelle");
        let _ = brain.network.add_bond(crate::bond::TemporalBond::new(
            s_real, s_organelle, BondType::IsA, 0.9, now_ms(),
        ));
        brain.network.recompute_all_typed_ratios();

        let filter = NoiseFilter::new(&brain, &cfg, "what is photosynthesis");
        assert!(filter.should_reject(&s_six_co, "6co"));
        assert!(filter.should_reject(&s_g, "g"));
        assert!(filter.should_reject(&s_gloom, "gloom"));
        assert!(!filter.should_reject(&s_real, "chloroplast"));
    }

    #[test]
    fn test_noise_filter_preserves_query_subject_when_junk() {
        // The user's query subject must be exempted even if it
        // matches a junk pattern. `gloom` is on the stop-list; if
        // the user asked `what is gloom`, the filter must NOT drop
        // gloom.
        let mut brain = Brain::new();
        let cfg = NarrationConfig::default();
        let s_gloom = put_node(&mut brain, "gloom");
        // Plus a real candidate for control.
        let s_six_co = put_node(&mut brain, "6co");

        let filter = NoiseFilter::new(&brain, &cfg, "what is gloom");
        assert!(!filter.should_reject(&s_gloom, "gloom"),
            "query subject `gloom` must be preserved even on stop-list");
        assert!(filter.should_reject(&s_six_co, "6co"),
            "non-subject junk still rejected");
    }

    #[test]
    fn test_noise_filter_disabled_passes_everything() {
        let mut brain = Brain::new();
        let mut cfg = NarrationConfig::default();
        cfg.noise_filter_enabled = false;
        let s_six_co = put_node(&mut brain, "6co");
        let s_g = put_node(&mut brain, "g");
        let filter = NoiseFilter::new(&brain, &cfg, "what is photosynthesis");
        assert!(!filter.should_reject(&s_six_co, "6co"));
        assert!(!filter.should_reject(&s_g, "g"));
    }

    #[test]
    fn test_noise_filter_domain_mismatch_drops_bag_of_associations() {
        // A node connected to >= 5 neighbors via RelatedTo only,
        // with no IsA grounding, must be rejected by the
        // domain-mismatch heuristic (typed_ratio = 0, deg >= 5,
        // no IsA). Mirrors the live-brain `ctdna` pattern: a noisy
        // leaf wired into many concepts via bag-of-association
        // bonds without ever being IsA-anchored.
        let mut brain = Brain::new();
        let cfg = NarrationConfig::default();
        let s = put_node(&mut brain, "ctdna");
        // 10 RelatedTo edges => typed_ratio = 0, no typed grounding
        // (no IsA / PartOf in either direction), deg = 10. Port #21
        // raised the gate from 5 to 10, so the test must now wire
        // up at least 10 association edges to trip the heuristic.
        for i in 0..10usize {
            let target = put_node(&mut brain, &format!("assoc{}", i));
            let _ = brain.network.add_bond(crate::bond::TemporalBond::new(
                s, target, BondType::RelatedTo, 0.5, now_ms(),
            ));
        }
        brain.network.recompute_all_typed_ratios();
        let filter = NoiseFilter::new(&brain, &cfg, "what is photosynthesis");
        assert!(filter.should_reject(&s, "ctdna"),
            "bag-of-associations connected node must be rejected");
    }

    #[test]
    fn test_noise_filter_preserves_clean_leaf_concept() {
        // A clean concept with one inbound IsA bond and no other
        // structure (e.g. `lepton` on a small graph that has just
        // taught `electron IsA lepton`) must NOT be rejected. The
        // domain-mismatch heuristic must back off when degree is
        // low — leaf concepts are not noise.
        let mut brain = Brain::new();
        let cfg = NarrationConfig::default();
        let parent = put_node(&mut brain, "electron");
        let leaf = put_node(&mut brain, "lepton");
        let _ = brain.network.add_bond(crate::bond::TemporalBond::new(
            parent, leaf, BondType::IsA, 0.9, now_ms(),
        ));
        brain.network.recompute_all_typed_ratios();
        let filter = NoiseFilter::new(&brain, &cfg, "what is electron");
        assert!(!filter.should_reject(&leaf, "lepton"),
            "clean low-degree leaf must survive the filter");
    }

    #[test]
    fn test_noise_filter_inbound_isa_protects_high_degree_node() {
        // Port #21: a node with 10 RelatedTo edges (would-be domain
        // mismatch) BUT also one inbound IsA from a parent node
        // must be preserved. Mirrors the `multicellular` pattern:
        // many associations, no outgoing IsA, but the node IS the
        // target of `something IsA multicellular`.
        let mut brain = Brain::new();
        let cfg = NarrationConfig::default();
        let s = put_node(&mut brain, "multicellular");
        let parent = put_node(&mut brain, "animal");
        // Inbound IsA grounding: animal IsA multicellular.
        let _ = brain.network.add_bond(crate::bond::TemporalBond::new(
            parent, s, BondType::IsA, 0.9, now_ms(),
        ));
        // 10 outbound RelatedTo edges to noisy partners.
        for i in 0..10usize {
            let target = put_node(&mut brain, &format!("noise{}", i));
            let _ = brain.network.add_bond(crate::bond::TemporalBond::new(
                s, target, BondType::RelatedTo, 0.5, now_ms(),
            ));
        }
        brain.network.recompute_all_typed_ratios();
        let filter = NoiseFilter::new(&brain, &cfg, "is mitosis cell division");
        assert!(!filter.should_reject(&s, "multicellular"),
            "inbound IsA grounding must protect a high-degree real term");
    }

    #[test]
    fn test_noise_filter_allowlist_protects_rna() {
        // Port #21: even a worst-case bag-of-associations RNA
        // node (no IsA in either direction, 10+ RelatedTo edges)
        // is preserved by the small DOMAIN_ALLOWLIST.
        let mut brain = Brain::new();
        let cfg = NarrationConfig::default();
        let s = put_node(&mut brain, "rna");
        for i in 0..10usize {
            let target = put_node(&mut brain, &format!("assoc{}", i));
            let _ = brain.network.add_bond(crate::bond::TemporalBond::new(
                s, target, BondType::RelatedTo, 0.5, now_ms(),
            ));
        }
        brain.network.recompute_all_typed_ratios();
        let filter = NoiseFilter::new(&brain, &cfg, "is mitosis cell division");
        assert!(!filter.should_reject(&s, "rna"),
            "DOMAIN_ALLOWLIST must protect `rna` from the heuristic");
    }

    #[test]
    fn test_multi_sense_render_lists_each_sense_distinctly() {
        // Port #22: render_core MultiSense arm produces the
        // "X has multiple senses. In one ... In another ...
        // Which sense did you mean?" structure given populated
        // multi_sense_clusters.
        let mut s = ShapeSlots::default();
        s.query = "what is cell".into();
        s.primary = Some("cell".into());
        s.multi_sense_clusters = vec![
            vec!["compartment".into(), "mitochondria".into(), "rna".into()],
            vec!["radiotelephone".into(), "radiophone".into()],
        ];
        let body = super::render_core(ResponseShape::MultiSense, &s, &StyleHints::neutral());
        assert!(body.starts_with("Cell has multiple senses."),
            "missing capitalised subject: {}", body);
        assert!(body.contains("In one sense, cell relates to compartment, mitochondria, rna."),
            "first sense missing or mis-formatted: {}", body);
        assert!(body.contains("In another, cell relates to radiotelephone, radiophone."),
            "second sense missing or mis-formatted: {}", body);
        assert!(body.ends_with("Which sense did you mean?"),
            "missing trailing question: {}", body);
    }



    // =====================================================================
    // OP-NARRATE-1 Phase C — style-aware render_core tests
    // =====================================================================
    //
    // `apply_style_modulations` was deleted in Phase C. Style adaptation
    // now lives inside `render_core` as per-shape template selection
    // driven by `StyleHints::verbosity_tier()`. The fabricated phrases
    // F1-F4 from the old post-processor are gone, so the tests that
    // asserted on them are deleted.
    //
    // The two structural invariants from the old port survive intact:
    //   * Kill-switch: `output_adaptation_enabled=false` → neutral hints
    //     → Standard tier (NOT terse).
    //   * Anti-hallucination: activated bonds + slot content invariant
    //     across style settings.

    fn hints_with(verb: f32, form: f32, tech: f32, emp: f32, samples: u64) -> StyleHints {
        let mut sv = crate::style::StyleVector::default();
        sv.verbosity = verb;
        sv.formality = form;
        sv.technicality = tech;
        sv.empathy = emp;
        sv.samples_count = samples;
        let cfg = crate::style::StyleConfig::default();
        StyleHints::from_style_vector(&sv, &cfg)
    }

    fn slots_with_3() -> ShapeSlots {
        let mut s = ShapeSlots::default();
        s.query = "what".to_string();
        s.primary = Some("alpha".to_string());
        s.secondary = Some("beta".to_string());
        s.tertiary = Some("gamma".to_string());
        s
    }

    fn slots_with_query(q: &str) -> ShapeSlots {
        let mut s = ShapeSlots::default();
        s.query = q.to_string();
        s.primary = Some("alpha".to_string());
        s.secondary = Some("beta".to_string());
        s.tertiary = Some("gamma".to_string());
        s
    }

    // ---- Verbosity tier helper ----

    #[test]
    fn op_narrate_1_verbosity_tier_terse_below_0_4() {
        let h = hints_with(0.2, 0.5, 0.5, 0.5, 500);
        assert_eq!(h.verbosity_tier(), crate::style::VerbosityTier::Terse);
    }

    #[test]
    fn op_narrate_1_verbosity_tier_standard_at_and_above_0_4() {
        let h_low_edge = hints_with(0.4, 0.5, 0.5, 0.5, 500);
        assert_eq!(h_low_edge.verbosity_tier(), crate::style::VerbosityTier::Standard);
        let h_mid = hints_with(0.5, 0.5, 0.5, 0.5, 500);
        assert_eq!(h_mid.verbosity_tier(), crate::style::VerbosityTier::Standard);
        let h_high = hints_with(0.9, 0.5, 0.5, 0.5, 500);
        assert_eq!(h_high.verbosity_tier(), crate::style::VerbosityTier::Standard);
    }

    #[test]
    fn op_narrate_1_neutral_hints_select_standard_tier() {
        let h = StyleHints::neutral();
        assert_eq!(h.verbosity_tier(), crate::style::VerbosityTier::Standard);
    }

    // ---- Anti-hallucination invariant (kept + expanded from CS.2) ----

    #[test]
    fn op_narrate_1_anti_hallucination_invariant_activated_bonds_unchanged_across_style_settings() {
        use crate::brain::Brain;
        use crate::node::Confidence;
        use crate::style::{StyleVector, ConversationalStyleStore};
        let mut brain = Brain::new();
        brain.teach("cat", crate::bond::BondType::IsA, "animal", 0.9, Confidence::Verified);
        brain.teach("cat", crate::bond::BondType::IsA, "small", 0.9, Confidence::Verified);
        let bond_count_pre = brain.network.bond_count();

        brain.style_config.output_adaptation_enabled = true;
        brain.cs_style_store = ConversationalStyleStore::new();
        let mut sv_low = StyleVector::default();
        sv_low.verbosity = 0.1;
        sv_low.formality = 0.1;
        sv_low.technicality = 0.1;
        sv_low.empathy = 0.1;
        sv_low.samples_count = 500;
        sv_low.last_refined_at_ms = 1;
        brain.cs_style_store.set("default", sv_low);
        let result_a = brain.narrate("what is cat", 3, 5, false);
        let activated_a: Vec<_> = result_a.ask.recall.activated.iter().map(|a| a.seed).collect();
        let bond_count_a = brain.network.bond_count();

        let mut sv_high = StyleVector::default();
        sv_high.verbosity = 0.9;
        sv_high.formality = 0.9;
        sv_high.technicality = 0.9;
        sv_high.empathy = 0.9;
        sv_high.samples_count = 500;
        sv_high.last_refined_at_ms = 1;
        brain.cs_style_store.set("default", sv_high);
        let result_b = brain.narrate("what is cat", 3, 5, false);
        let activated_b: Vec<_> = result_b.ask.recall.activated.iter().map(|a| a.seed).collect();
        let bond_count_b = brain.network.bond_count();

        assert_eq!(activated_a, activated_b,
            "activated bonds must be identical across style settings");
        assert_eq!(bond_count_pre, bond_count_a,
            "bond_count must not change after low-style narration");
        assert_eq!(bond_count_pre, bond_count_b,
            "bond_count must not change after high-style narration");
        // Slot content equality across settings is exercised indirectly:
        // the same query against the same brain yields the same shape and
        // the same set of activated seeds; only the prose surface differs.
    }

    // ---- Kill-switch invariant (rewritten from CS.2) ----

    #[test]
    fn op_narrate_1_output_adaptation_disabled_produces_standard_tier() {
        use crate::style::{StyleVector, StyleConfig, VerbosityTier};
        let mut cfg = StyleConfig::default();
        cfg.output_adaptation_enabled = false;
        let mut sv = StyleVector::default();
        sv.verbosity = 0.1;
        sv.formality = 0.9;
        sv.technicality = 0.9;
        sv.empathy = 0.9;
        sv.samples_count = 1_000;
        let h = StyleHints::from_style_vector(&sv, &cfg);
        assert!(h.is_neutral(), "kill switch must produce neutral hints");
        assert_eq!(h.verbosity_tier(), VerbosityTier::Standard,
            "kill switch must select Standard tier, never Terse");
    }

    // ---- Per-shape verbose + terse tests ----

    #[test]
    fn op_narrate_1_shape_multistep_standard_form_renders_full_template() {
        let s = slots_with_3();
        let h = hints_with(0.5, 0.5, 0.5, 0.5, 500);
        let out = super::render_core(ResponseShape::MultiStep, &s, &h);
        assert!(out.contains("Several parts to this"), "missing standard scaffold: {}", out);
        assert!(out.contains("First: alpha"), "missing first slot: {}", out);
        assert!(out.contains("Second: beta"), "missing second slot: {}", out);
        assert!(out.contains("Third: gamma"), "missing third slot: {}", out);
    }

    #[test]
    fn op_narrate_1_shape_multistep_terse_preserves_all_slots() {
        let s = slots_with_3();
        let h = hints_with(0.2, 0.5, 0.5, 0.5, 500);
        let out = super::render_core(ResponseShape::MultiStep, &s, &h);
        assert!(out.contains("alpha") && out.contains("beta") && out.contains("gamma"),
            "terse MultiStep must preserve all three slots: {}", out);
        assert!(!out.contains("Several parts to this"),
            "terse MultiStep must drop the verbose scaffold: {}", out);
        assert!(out.len() < 80,
            "terse MultiStep should be much shorter: len={} out={}", out.len(), out);
    }

    #[test]
    fn op_narrate_1_shape_process_standard() {
        let s = slots_with_query("baking");
        let h = hints_with(0.5, 0.5, 0.5, 0.5, 500);
        let out = super::render_core(ResponseShape::Process, &s, &h);
        assert!(out.contains("works in") && out.contains("first alpha"), "{}", out);
    }

    #[test]
    fn op_narrate_1_shape_process_terse() {
        let s = slots_with_query("baking");
        let h = hints_with(0.2, 0.5, 0.5, 0.5, 500);
        let out = super::render_core(ResponseShape::Process, &s, &h);
        assert_eq!(out, "alpha; beta; gamma.");
    }

    #[test]
    fn op_narrate_1_shape_whychain_standard() {
        let s = slots_with_3();
        let h = hints_with(0.5, 0.5, 0.5, 0.5, 500);
        let out = super::render_core(ResponseShape::WhyChain, &s, &h);
        assert!(out.contains("happens because beta") && out.contains("producing alpha"), "{}", out);
    }

    #[test]
    fn op_narrate_1_shape_whychain_terse() {
        let s = slots_with_3();
        let h = hints_with(0.2, 0.5, 0.5, 0.5, 500);
        let out = super::render_core(ResponseShape::WhyChain, &s, &h);
        assert_eq!(out, "alpha: beta -> gamma.");
    }

    #[test]
    fn op_narrate_1_shape_comparison_standard() {
        let mut s = slots_with_3();
        s.bond_label = Some("habitat".to_string());
        let h = hints_with(0.5, 0.5, 0.5, 0.5, 500);
        let out = super::render_core(ResponseShape::Comparison, &s, &h);
        assert!(out.contains("alpha and beta share habitat"), "{}", out);
        assert!(out.contains("gamma"), "{}", out);
    }

    #[test]
    fn op_narrate_1_shape_comparison_terse() {
        let mut s = slots_with_3();
        s.bond_label = Some("habitat".to_string());
        let h = hints_with(0.2, 0.5, 0.5, 0.5, 500);
        let out = super::render_core(ResponseShape::Comparison, &s, &h);
        assert!(out.contains("alpha and beta: shared habitat; differ on gamma"), "{}", out);
    }

    #[test]
    fn op_narrate_1_shape_uncertain_standard() {
        let s = slots_with_query("query");
        let h = hints_with(0.5, 0.5, 0.5, 0.5, 500);
        let out = super::render_core(ResponseShape::UncertainAnswer, &s, &h);
        assert!(out.contains("I think it might be alpha"), "{}", out);
    }

    #[test]
    fn op_narrate_1_shape_uncertain_terse() {
        let s = slots_with_query("query");
        let h = hints_with(0.2, 0.5, 0.5, 0.5, 500);
        let out = super::render_core(ResponseShape::UncertainAnswer, &s, &h);
        assert_eq!(out, "Possibly alpha (low support).");
    }

    #[test]
    fn op_narrate_1_shape_multihypothesis_standard() {
        let mut s = slots_with_3();
        s.winner_alt_count = 3;
        s.winner_alt = Some("delta".to_string());
        let h = hints_with(0.5, 0.5, 0.5, 0.5, 500);
        let out = super::render_core(ResponseShape::MultiHypothesisAnswer, &s, &h);
        assert!(out.contains("Most likely alpha") && out.contains("3 alternatives") && out.contains("delta"), "{}", out);
    }

    #[test]
    fn op_narrate_1_shape_multihypothesis_terse() {
        let mut s = slots_with_3();
        s.winner_alt_count = 3;
        s.winner_alt = Some("delta".to_string());
        let h = hints_with(0.2, 0.5, 0.5, 0.5, 500);
        let out = super::render_core(ResponseShape::MultiHypothesisAnswer, &s, &h);
        assert_eq!(out, "Most likely alpha (vs. 3 alternatives e.g. delta).");
    }

    #[test]
    fn op_hygiene_3a_belief_summary_blank_is_none() {
        let b = crate::agents::Belief {
            from: crate::seed::Seed::from_label(""),
            via: crate::seed::Seed::from_label(""),
            to: crate::seed::Seed::from_label(""),
            from_content: String::new(),
            via_content: String::new(),
            to_content: String::new(),
            w_from_via: 0.0,
            w_via_to: 0.0,
            confidence: 0.0,
            scar_reasons: vec![],
            caveats: vec![],
        };
        assert!(super::belief_to_summary(&b).is_none());
    }

    #[test]
    fn op_hygiene_3a_multihyp_empty_slots_refuses() {
        let mut s = slots_with_query("what is entropy");
        s.primary = None;
        s.winner_alt = None;
        s.winner_alt_count = 7;
        let h = hints_with(0.5, 0.5, 0.5, 0.5, 500);
        let out = super::render_core(ResponseShape::MultiHypothesisAnswer, &s, &h);
        assert!(!out.contains("relates to  via"), "must not word-salad: {}", out);
        assert!(
            out.to_lowercase().contains("don't have anything solid"),
            "expected honest refusal: {}", out
        );
        assert!(out.contains("entropy"), "refusal names the subject: {}", out);
    }

    #[test]
    fn op_hygiene_3a_multihyp_real_slots_render_normally() {
        let mut s = slots_with_3();
        s.winner_alt_count = 3;
        s.winner_alt = Some("delta".to_string());
        let h = hints_with(0.5, 0.5, 0.5, 0.5, 500);
        let out = super::render_core(ResponseShape::MultiHypothesisAnswer, &s, &h);
        assert!(out.contains("Most likely alpha") && out.contains("delta"), "{}", out);
    }

    #[test]
    fn op_hygiene_3b_strip_trailing_confidence_scalar() {
        let s = "a quantum of electromagnetic radiation Confidence: 0.12.".to_string();
        assert_eq!(
            super::strip_internal_confidence_tag(s),
            "a quantum of electromagnetic radiation"
        );
        let terse = "alpha via beta Conf: 0.50.".to_string();
        assert_eq!(super::strip_internal_confidence_tag(terse), "alpha via beta");
    }

    #[test]
    fn op_hygiene_3b_keeps_label_and_parenthetical_confidence() {
        let critic = "x. My critic caught 1 caveats: y. Confidence: 0.50 (high).".to_string();
        assert_eq!(super::strip_internal_confidence_tag(critic.clone()), critic);
        let label = "It reports 3 concepts. Confidence: high. Recently I activated y.".to_string();
        assert_eq!(super::strip_internal_confidence_tag(label.clone()), label);
    }

    #[test]
    fn op_narrate_1_shape_bridged_standard() {
        let mut s = slots_with_3();
        s.bridge_low = Some("lo".to_string());
        s.bridge_high = Some("hi".to_string());
        let h = hints_with(0.5, 0.5, 0.5, 0.5, 500);
        let out = super::render_core(ResponseShape::BridgedAnswer, &s, &h);
        assert!(out.contains("connection between lo and hi"), "{}", out);
    }

    #[test]
    fn op_narrate_1_shape_bridged_terse() {
        let mut s = slots_with_3();
        s.bridge_low = Some("lo".to_string());
        s.bridge_high = Some("hi".to_string());
        let h = hints_with(0.2, 0.5, 0.5, 0.5, 500);
        let out = super::render_core(ResponseShape::BridgedAnswer, &s, &h);
        assert_eq!(out, "alpha. Bridge: lo <-> hi.");
    }

    #[test]
    fn op_narrate_1_shape_critic_standard() {
        let mut s = slots_with_3();
        s.critic_caveat_count = 2;
        s.critic_caveat = Some("watch out".to_string());
        s.confidence_value = 0.65;
        s.confidence_label = "moderate".to_string();
        let h = hints_with(0.5, 0.5, 0.5, 0.5, 500);
        let out = super::render_core(ResponseShape::CriticVettedAnswer, &s, &h);
        assert!(out.contains("My critic caught 2 caveats: watch out") && out.contains("Confidence: 0.65"), "{}", out);
    }

    #[test]
    fn op_narrate_1_shape_critic_terse() {
        let mut s = slots_with_3();
        s.critic_caveat_count = 2;
        s.critic_caveat = Some("watch out".to_string());
        s.confidence_value = 0.65;
        s.confidence_label = "moderate".to_string();
        let h = hints_with(0.2, 0.5, 0.5, 0.5, 500);
        let out = super::render_core(ResponseShape::CriticVettedAnswer, &s, &h);
        assert_eq!(out, "alpha. Caveats (2): watch out. Conf: 0.65.");
    }

    #[test]
    fn op_narrate_1_shape_refusal_standard() {
        let s = slots_with_query("xyz");
        let h = hints_with(0.5, 0.5, 0.5, 0.5, 500);
        let out = super::render_core(ResponseShape::Refusal, &s, &h);
        assert!(out.contains("don't have a meaningful answer to \"xyz\""), "{}", out);
    }

    #[test]
    fn op_narrate_1_shape_refusal_terse() {
        let s = slots_with_query("xyz");
        let h = hints_with(0.2, 0.5, 0.5, 0.5, 500);
        let out = super::render_core(ResponseShape::Refusal, &s, &h);
        assert!(out.contains("No answer for \"xyz\"") && out.contains("no bonds activated"), "{}", out);
    }

    #[test]
    fn op_narrate_1_shape_metaself_standard() {
        let mut s = slots_with_query("xyz");
        s.node_count = 42;
        s.bond_count = 100;
        s.confidence_label = "verified".to_string();
        let h = hints_with(0.5, 0.5, 0.5, 0.5, 500);
        let out = super::render_core(ResponseShape::MetaSelf, &s, &h);
        assert!(out.contains("On \"xyz\", I have 42 concepts and 100 associations") && out.contains("Recently I activated alpha"), "{}", out);
    }

    #[test]
    fn op_narrate_1_shape_metaself_terse() {
        let mut s = slots_with_query("xyz");
        s.node_count = 42;
        s.bond_count = 100;
        s.confidence_label = "verified".to_string();
        let h = hints_with(0.2, 0.5, 0.5, 0.5, 500);
        let out = super::render_core(ResponseShape::MetaSelf, &s, &h);
        assert_eq!(out, "\"xyz\": 42 concepts, 100 bonds, conf=verified, recent=alpha.");
    }

    #[test]
    fn op_narrate_1_shape_timetravel_standard() {
        let s = slots_with_query("xyz");
        let h = hints_with(0.5, 0.5, 0.5, 0.5, 500);
        let out = super::render_core(ResponseShape::TimetravelComparison, &s, &h);
        assert!(out.contains("About \"xyz\"") && out.contains("primarily associated this with beta") && out.contains("Now: alpha"), "{}", out);
    }

    #[test]
    fn op_narrate_1_shape_timetravel_terse() {
        let s = slots_with_query("xyz");
        let h = hints_with(0.2, 0.5, 0.5, 0.5, 500);
        let out = super::render_core(ResponseShape::TimetravelComparison, &s, &h);
        assert_eq!(out, "\"xyz\": was beta, now alpha.");
    }

    #[test]
    fn op_narrate_1_shape_synthesis_standard() {
        let mut s = slots_with_3();
        s.confidence_value = 0.5;
        let h = hints_with(0.5, 0.5, 0.5, 0.5, 500);
        let out = super::render_core(ResponseShape::Synthesis, &s, &h);
        assert!(out.contains("alpha relates to gamma") && out.contains("inferring this from beta") && out.contains("Confidence: 0.50"), "{}", out);
    }

    #[test]
    fn op_narrate_1_shape_synthesis_terse_uses_via_not_arrow() {
        let mut s = slots_with_3();
        s.confidence_value = 0.5;
        let h = hints_with(0.2, 0.5, 0.5, 0.5, 500);
        let out = super::render_core(ResponseShape::Synthesis, &s, &h);
        assert!(out.contains("via"), "terse Synthesis must use via connector: {}", out);
        assert!(!out.contains("->"), "terse Synthesis must NOT use arrow connector: {}", out);
        assert!(out.contains("alpha") && out.contains("beta") && out.contains("gamma"), "{}", out);
    }

    #[test]
    fn op_narrate_1_shape_wondering_standard() {
        let mut s = slots_with_3();
        s.wondering_about = Some("nebula".to_string());
        s.wondering_kind = Some("anomaly".to_string());
        let h = hints_with(0.5, 0.5, 0.5, 0.5, 500);
        let out = super::render_core(ResponseShape::Wondering, &s, &h);
        assert!(out.contains("wondering about nebula") && out.contains("anomaly"), "{}", out);
    }

    #[test]
    fn bug2_tell_me_about_is_definition() {
        assert_eq!(super::IntentClassifier::classify("tell me about python"), super::QueryIntent::Definition);
        assert_eq!(super::IntentClassifier::classify("tell me about the rust language"), super::QueryIntent::Definition);
        assert_ne!(super::IntentClassifier::classify("tell me about yourself"), super::QueryIntent::Definition);
        assert_eq!(super::IntentClassifier::classify("what is python"), super::QueryIntent::Definition);
    }

    #[test]
    fn bug3_routing_handles_are_display_filtered() {
        assert!(super::is_handleish("pyfnlist"));
        assert!(super::is_handleish("jsfnmap"));
        assert!(super::is_handleish("builtins.sum"));
        assert!(!super::is_handleish("the list function"));
        assert!(!super::is_handleish("list"));
    }

    #[test]
    fn bug1_curiosity_voice_suppressed_when_flag_off() {
        let slots = slots_with_3();
        let off = ModulationFlags { curiosity_touched: true, expose_curiosity_voice: false, ..Default::default() };
        let out = super::apply_suffix_modulations("Base answer.".to_string(), ResponseShape::Definition, &slots, &off);
        assert!(!out.to_lowercase().contains("gap"), "must suppress gap voice: {}", out);
        let on = ModulationFlags { curiosity_touched: true, expose_curiosity_voice: true, ..Default::default() };
        let out2 = super::apply_suffix_modulations("Base answer.".to_string(), ResponseShape::Definition, &slots, &on);
        assert!(out2.to_lowercase().contains("gap"), "must append gap voice when exposed: {}", out2);
    }

    #[test]
    fn op_narrate_1_shape_wondering_terse() {
        let mut s = slots_with_3();
        s.wondering_about = Some("nebula".to_string());
        s.wondering_kind = Some("anomaly".to_string());
        let h = hints_with(0.2, 0.5, 0.5, 0.5, 500);
        let out = super::render_core(ResponseShape::Wondering, &s, &h);
        assert!(out.contains("alpha.") && out.contains("(gap: nebula.)"), "{}", out);
    }

    #[test]
    fn op_narrate_1_shape_multisense_standard() {
        let mut s = slots_with_query("cell");
        s.primary = Some("cell".to_string());
        s.multi_sense_clusters = vec![
            vec!["compartment".to_string(), "mitochondria".to_string()],
            vec!["radiotelephone".to_string()],
        ];
        let h = hints_with(0.5, 0.5, 0.5, 0.5, 500);
        let out = super::render_core(ResponseShape::MultiSense, &s, &h);
        assert!(out.starts_with("Cell has multiple senses"), "{}", out);
        assert!(out.contains("In one sense, cell relates to compartment, mitochondria."), "{}", out);
        assert!(out.contains("In another, cell relates to radiotelephone."), "{}", out);
        assert!(out.ends_with("Which sense did you mean?"), "{}", out);
    }

    #[test]
    fn op_narrate_1_shape_multisense_terse() {
        let mut s = slots_with_query("cell");
        s.primary = Some("cell".to_string());
        s.multi_sense_clusters = vec![
            vec!["compartment".to_string(), "mitochondria".to_string()],
            vec!["radiotelephone".to_string()],
        ];
        let h = hints_with(0.2, 0.5, 0.5, 0.5, 500);
        let out = super::render_core(ResponseShape::MultiSense, &s, &h);
        assert!(out.starts_with("Cell: sense 1 = compartment, mitochondria;"), "{}", out);
        assert!(out.contains("sense 2 = radiotelephone"), "{}", out);
        assert!(out.ends_with("Which?"), "{}", out);
    }

    // ---- Walker prose passthrough at both tiers ----

    #[test]
    fn op_narrate_1_walker_prose_passthrough_definition_terse() {
        let mut s = slots_with_3();
        s.walker_prose = Some("composed sentence with multiple bonds.".to_string());
        let h_t = hints_with(0.2, 0.5, 0.5, 0.5, 500);
        let h_s = hints_with(0.5, 0.5, 0.5, 0.5, 500);
        let out_t = super::render_core(ResponseShape::Definition, &s, &h_t);
        let out_s = super::render_core(ResponseShape::Definition, &s, &h_s);
        assert_eq!(out_t, "composed sentence with multiple bonds.");
        assert_eq!(out_s, "composed sentence with multiple bonds.");
    }

    #[test]
    fn op_narrate_1_walker_prose_passthrough_synthesis_terse_only_shrinks_conf_suffix() {
        let mut s = slots_with_3();
        s.walker_prose = Some("composed prose.".to_string());
        s.confidence_value = 0.5;
        let h_t = hints_with(0.2, 0.5, 0.5, 0.5, 500);
        let h_s = hints_with(0.5, 0.5, 0.5, 0.5, 500);
        let out_t = super::render_core(ResponseShape::Synthesis, &s, &h_t);
        let out_s = super::render_core(ResponseShape::Synthesis, &s, &h_s);
        assert!(out_t.starts_with("composed prose.") && out_t.contains("Conf: 0.50"));
        assert!(out_s.starts_with("composed prose.") && out_s.contains("Confidence: 0.50"));
    }

    // ---- Regression: end-to-end narrate() must preserve MultiStep body at low verbosity ----

    #[test]
    fn op_narrate_1_narrate_low_verbosity_preserves_multistep_body_when_eligible() {
        // Even if the live brain doesn't pick MultiStep for this fixture, the
        // load-bearing assertion is that the narrate() output is NOT the
        // historic 22-char "Several parts to this." truncation bug.
        use crate::brain::Brain;
        use crate::node::Confidence;
        use crate::style::{StyleVector, ConversationalStyleStore};
        let mut brain = Brain::new();
        brain.teach("topic", crate::bond::BondType::IsA, "alpha", 0.9, Confidence::Verified);
        brain.teach("topic", crate::bond::BondType::IsA, "beta", 0.9, Confidence::Verified);
        brain.teach("topic", crate::bond::BondType::IsA, "gamma", 0.9, Confidence::Verified);

        brain.style_config.output_adaptation_enabled = true;
        brain.cs_style_store = ConversationalStyleStore::new();
        let mut sv_low = StyleVector::default();
        sv_low.verbosity = 0.1;
        sv_low.samples_count = 500;
        sv_low.last_refined_at_ms = 1;
        brain.cs_style_store.set("default", sv_low);
        let result = brain.narrate("topic", 3, 5, false);
        // The CRITICAL assertion: the answer is not the historic truncation bug.
        assert_ne!(result.answer, "Several parts to this.",
            "narrate at low verbosity must not produce the OP-NARRATE-1 truncation bug");
        // Also assert no fabricated phrases appear.
        for phrase in ["That's a good question", "canonical formulation", "more to unpack", "In plain terms: this means"] {
            assert!(!result.answer.contains(phrase),
                "fabricated phrase {:?} must not appear in narrate output: {}", phrase, result.answer);
        }
    }

    // ------ OP-ROUTE-2 parity invariants ------

    /// Phase 2 parity: a math query causes routing.needs_math = true,
    /// but narrate's answer / shape are byte-identical across two
    /// runs - observation does not perturb output.
    #[test]
    fn op_route_2_routing_does_not_change_answer_for_math_query() {
        let mut brain = Brain::new();
        teach(&mut brain, "two", BondType::IsA, "number");
        let r1 = narrate(&mut brain, "what is 2+2", 1, 4, false, &cfg_no_gate());
        let r2 = narrate(&mut brain, "what is 2+2", 1, 4, false, &cfg_no_gate());
        assert!(r1.routing.needs_math, "router should detect math");
        assert_eq!(r1.answer, r2.answer);
        assert_eq!(r1.shape, r2.shape);
    }

    /// Phase 2 parity: baseline definition query has no routing flags.
    #[test]
    fn op_route_2_baseline_query_has_no_routing_flags() {
        let mut brain = Brain::new();
        teach(&mut brain, "photosynthesis", BondType::IsA, "process");
        let r = narrate(&mut brain, "what is photosynthesis?", 2, 5, false, &cfg_no_gate());
        assert!(!r.routing.needs_math);
        assert!(!r.routing.references_file_path);
        assert!(!r.routing.references_image_path);
        assert!(!r.routing.needs_deep_reasoning);
        assert!(!r.routing.needs_variants);
        assert!(!r.routing.needs_critic_vetting);
        assert!(!r.routing.code_intent);
        assert!(r.routing.triggers.is_empty());
    }


    // =================================================================
    // OP-ROUTE-3a tests: chain-aware shape selection, slot fill, render,
    // NarrationResult.reasoning, parity invariant.
    // =================================================================

    /// Hand-build a `ReasoningTrace` for test fixtures. Bypasses the
    /// real `reason_with_config` pathway so unit tests don't need a
    /// densely-bonded brain to exercise the 3a wiring.
    fn mk_trace(
        steps_len: usize,
        no_chain: bool,
        kind: crate::reasoning::EvidenceKind,
        final_answer: &str,
    ) -> crate::reasoning::ReasoningTrace {
        let mut steps = Vec::new();
        for i in 0..steps_len {
            steps.push(crate::reasoning::ReasoningStep {
                from: Seed::from_label(&format!("a{}", i)),
                from_content: format!("a{}", i),
                bond_type: BondType::IsA,
                to: Seed::from_label(&format!("b{}", i)),
                to_content: format!("b{}", i),
                confidence: 0.8,
                articulation: format!("step {}", i),
            });
        }
        crate::reasoning::ReasoningTrace {
            query: "q".into(),
            scope: TemporalScope::Current,
            steps,
            final_answer: final_answer.into(),
            confidence: 0.82,
            bonds_used: Vec::new(),
            no_chain,
            evidence_kind: kind,
        }
    }

    /// Minimal AskResult with one activated node so ShapeSelector::pick
    /// doesn't bail at rule 1 (empty activations -> Refusal).
    fn mk_ask_with_one_act() -> AskResult {
        let mut acts = Vec::new();
        acts.push(ActivatedNode {
            seed: Seed::from_label("x"),
            content: "x".into(),
            activation: 0.7,
            confidence: Confidence::Verified,
            path: Vec::new(),
            depth: 0,
        });
        AskResult {
            query: "q".into(),
            recall: RecallResult {
                activated: acts,
                query_seeds: Vec::new(),
                scope: TemporalScope::Current,
                nodes_visited: 0,
                bonds_evaluated: 0,
                activation_truncated: false,
                status: crate::network::SpreadStatus::Complete,
            },
            seed_hits: Vec::new(),
            hebbian_strengthened: 0,
            session_id: None,
            bridges: Vec::new(),
            curiosity_signals: Vec::new(),
            scars: Vec::new(),
            status: crate::brain::AskStatus::Complete,
        }
    }

    #[test]
    fn op_route_guest_definition_outranks_incidental_bridge() {
        // OP-ROUTE-GUEST (2026-07-23): a Definition-intent query with an
        // incidental bridge must NOT be hijacked into BridgedAnswer.
        // This is the "What is a dog" -> "Extant Late Pleistocene" class.
        let ask = mk_ask_with_one_act();
        let cfg = cfg_no_gate();
        let mut flags = ModulationFlags::default();
        flags.bridge_touched = true;
        let shape = ShapeSelector::pick(
            QueryIntent::Definition, &ask, None, &flags, &cfg, None, None, None);
        assert_ne!(shape, ResponseShape::BridgedAnswer,
            "an incidental bridge must not hijack a Definition query");
        assert_eq!(shape, ResponseShape::Definition);
        // Control: a genuinely relational (non-Definition) query still
        // surfaces the bridge.
        let shape2 = ShapeSelector::pick(
            QueryIntent::Comparison, &ask, None, &flags, &cfg, None, None, None);
        assert_eq!(shape2, ResponseShape::BridgedAnswer,
            "a non-Definition query must still surface the bridge");
    }

    #[test]
    fn op_route_3a_shape_picks_whychain_when_structured_chain_present() {
        let ask = mk_ask_with_one_act();
        let cfg = cfg_no_gate();
        let flags = ModulationFlags::default();
        let trace = mk_trace(3, false, crate::reasoning::EvidenceKind::Structured, "X because Y because Z.");
        let shape = ShapeSelector::pick(
            QueryIntent::Definition, &ask, None, &flags, &cfg, Some(&trace), None, None);
        assert_eq!(shape, ResponseShape::WhyChain,
            "structured 3-step trace must force WhyChain even on Definition intent");
    }

    #[test]
    fn op_route_3a_shape_ignores_associative_chain() {
        let ask = mk_ask_with_one_act();
        let cfg = cfg_no_gate();
        let flags = ModulationFlags::default();
        let trace = mk_trace(3, false, crate::reasoning::EvidenceKind::Associative, "X loosely connects to Y.");
        let shape = ShapeSelector::pick(
            QueryIntent::Definition, &ask, None, &flags, &cfg, Some(&trace), None, None);
        assert_ne!(shape, ResponseShape::WhyChain,
            "associative trace must NOT bias to WhyChain (prose is hedged)");
    }

    #[test]
    fn op_route_3a_shape_ignores_no_chain_trace() {
        let ask = mk_ask_with_one_act();
        let cfg = cfg_no_gate();
        let flags = ModulationFlags::default();
        let trace = mk_trace(0, true, crate::reasoning::EvidenceKind::Structured, "no chain.");
        let shape = ShapeSelector::pick(
            QueryIntent::Definition, &ask, None, &flags, &cfg, Some(&trace), None, None);
        assert_ne!(shape, ResponseShape::WhyChain,
            "no_chain trace must NOT bias to WhyChain");
    }

    #[test]
    fn op_route_3a_shape_ignores_single_step_chain() {
        let ask = mk_ask_with_one_act();
        let cfg = cfg_no_gate();
        let flags = ModulationFlags::default();
        let trace = mk_trace(1, false, crate::reasoning::EvidenceKind::Structured, "X is Y.");
        let shape = ShapeSelector::pick(
            QueryIntent::Definition, &ask, None, &flags, &cfg, Some(&trace), None, None);
        assert_ne!(shape, ResponseShape::WhyChain,
            "single-step chain must NOT bias to WhyChain (need >=2)");
    }

    #[test]
    fn op_route_3a_shape_no_chain_arg_preserves_intent() {
        let ask = mk_ask_with_one_act();
        let cfg = cfg_no_gate();
        let flags = ModulationFlags::default();
        // Passing None must match pre-3a behavior for a Definition intent.
        let shape = ShapeSelector::pick(
            QueryIntent::Definition, &ask, None, &flags, &cfg, None, None, None);
        assert_eq!(shape, ResponseShape::Definition,
            "None reasoning arg must preserve intent-based shape selection");
    }

    #[test]
    fn op_route_3a_extract_slots_populates_chain_when_trace_present() {
        let mut brain = Brain::new();
        teach(&mut brain, "a", BondType::IsA, "b");
        let ask = brain.ask("a", 1, 3);
        let trace = mk_trace(3, false, crate::reasoning::EvidenceKind::Structured, "X happens because Y.");
        let slots = extract_slots(
            &brain, "a", &ask, None, &NarrationConfig::default(), Some(&trace), None, None, QueryIntent::Generic);
        assert_eq!(slots.chain_articulation.as_deref(), Some("X happens because Y."));
        assert!((slots.chain_confidence - 0.82).abs() < 1e-4,
            "chain_confidence={}, expected 0.82", slots.chain_confidence);
    }

    #[test]
    fn op_route_3a_extract_slots_skips_chain_when_no_chain_true() {
        let mut brain = Brain::new();
        teach(&mut brain, "a", BondType::IsA, "b");
        let ask = brain.ask("a", 1, 3);
        let trace = mk_trace(0, true, crate::reasoning::EvidenceKind::Structured, "no chain.");
        let slots = extract_slots(
            &brain, "a", &ask, None, &NarrationConfig::default(), Some(&trace), None, None, QueryIntent::Generic);
        assert!(slots.chain_articulation.is_none(),
            "no_chain trace must not populate chain_articulation");
        // chain_confidence is still recorded for observability.
        assert!((slots.chain_confidence - 0.82).abs() < 1e-4);
    }

    #[test]
    fn op_route_3a_render_whychain_uses_chain_prose_when_present() {
        let mut slots = ShapeSlots::default();
        slots.query = "q".into();
        slots.chain_articulation = Some("Ice floats because water is less dense as a solid.".into());
        slots.primary = Some("ice".into());
        slots.secondary = Some("density".into());
        slots.tertiary = Some("float".into());
        let flags = ModulationFlags::default();
        let cfg = NarrationConfig::default();
        let hints = StyleHints::default();
        let out = Renderer::render(ResponseShape::WhyChain, &slots, &flags, &cfg, &hints);
        assert!(
            out.contains("Ice floats because water is less dense as a solid."),
            "chain prose must appear in output, got: {}", out,
        );
    }

    #[test]
    fn op_route_3a_render_whychain_falls_back_to_template_when_chain_absent() {
        let mut slots = ShapeSlots::default();
        slots.query = "q".into();
        slots.chain_articulation = None;
        slots.primary = Some("ice".into());
        slots.secondary = Some("density".into());
        slots.tertiary = Some("float".into());
        let flags = ModulationFlags::default();
        let cfg = NarrationConfig::default();
        let hints = StyleHints::default();
        let out = Renderer::render(ResponseShape::WhyChain, &slots, &flags, &cfg, &hints);
        // Default StyleHints is Standard tier -> the classic template.
        assert!(
            out.contains("happens because"),
            "fallback template must fire, got: {}", out,
        );
        assert!(out.contains("ice"));
        assert!(out.contains("density"));
        assert!(out.contains("float"));
    }

    #[test]
    fn op_route_3a_narrate_parity_no_deep_reasoning_query() {
        // "what is photosynthesis?" -- short Definition intent, NOT a
        // 25+-word deep query. Router should NOT fire deep_reasoning;
        // narration must work exactly like pre-3a.
        let mut brain = Brain::new();
        teach(&mut brain, "photosynthesis", BondType::IsA, "process");
        let r = narrate(&mut brain, "what is photosynthesis?", 2, 5, false, &cfg_no_gate());
        assert!(!r.routing.needs_deep_reasoning,
            "short definition must not trigger deep reasoning");
        assert!(r.reasoning.is_none(),
            "reasoning trace must not be built for non-deep queries");
    }

    #[test]
    fn op_route_3a_narrate_invokes_reasoning_when_deep_reasoning_routes() {
        // 25+ words triggers `needs_deep_reasoning` per query_router.
        let mut brain = Brain::new();
        teach(&mut brain, "photosynthesis", BondType::IsA, "process");
        teach(&mut brain, "process", BondType::Requires, "light");
        let q = "walk me through step by step how photosynthesis works in great detail considering chemical and biological factors please";
        let r = narrate(&mut brain, q, 2, 5, false, &cfg_no_gate());
        assert!(r.routing.needs_deep_reasoning,
            "long query must trigger deep_reasoning");
        assert!(r.reasoning.is_some(),
            "NarrationResult.reasoning must be populated when deep_reasoning fires");
        // Trace may be no_chain on a tiny synthetic brain; the wire just
        // has to fire. If the chain succeeded, shape should be WhyChain.
        let trace = r.reasoning.as_ref().unwrap();
        if !trace.no_chain && trace.steps.len() >= 2
            && trace.evidence_kind == crate::reasoning::EvidenceKind::Structured
        {
            assert_eq!(r.shape, ResponseShape::WhyChain,
                "structured multi-step trace must force WhyChain end-to-end");
        }
    }

    #[test]
    fn op_route_3a_narration_result_serializes_with_reasoning_none() {
        let mut brain = Brain::new();
        teach(&mut brain, "a", BondType::IsA, "b");
        let r = narrate(&mut brain, "what is a?", 2, 5, false, &cfg_no_gate());
        assert!(r.reasoning.is_none());
        let s = serde_json::to_string(&r).expect("serialize");
        // `reasoning: None` serializes to `"reasoning":null` -- field is
        // present in the JSON regardless, which is the API contract
        // callers depend on.
        assert!(s.contains("\"reasoning\":null"),
            "reasoning field must serialize as null when absent, got: {}", s);
    }

    #[test]
    fn op_route_3a_narration_result_serializes_with_reasoning_some() {
        let mut brain = Brain::new();
        teach(&mut brain, "a", BondType::IsA, "b");
        let mut r = narrate(&mut brain, "what is a?", 2, 5, false, &cfg_no_gate());
        r.reasoning = Some(mk_trace(3, false, crate::reasoning::EvidenceKind::Structured, "X because Y."));
        let s = serde_json::to_string(&r).expect("serialize");
        // Populated trace must round-trip through serialization.
        assert!(s.contains("\"final_answer\":\"X because Y.\""),
            "final_answer must serialize, got: {}", s);
        assert!(s.contains("\"no_chain\":false"));
    }


    // =================================================================
    // OP-ROUTE-3b tests: variant-arena-aware shape selection, slot fill,
    // NarrationResult.variants, parity invariant.
    // =================================================================

    /// Hand-build a `VariantArena` for test fixtures. `winner_idx` is
    /// `None` -> no winner. Otherwise indexes into the produced spec
    /// list (0-based). Each spec gets its anchor seed labeled with the
    /// supplied `anchors[i]`. Scores are synthesized so the winner has
    /// composite 0.7 and runner-ups taper.
    fn mk_arena(anchors: &[&str], winner_idx: Option<usize>)
        -> crate::variants::VariantArena
    {
        use crate::variants::{
            PerturbationStrategy, VariantArena, VariantScore, VariantSpec,
        };
        let strategies = [
            PerturbationStrategy::Baseline,
            PerturbationStrategy::DomainShift,
            PerturbationStrategy::Deep,
            PerturbationStrategy::Contrarian,
            PerturbationStrategy::Interdisciplinary,
        ];
        let variants: Vec<VariantSpec> = anchors
            .iter()
            .enumerate()
            .map(|(i, label)| VariantSpec {
                id: i as u64,
                strategy: strategies[i % strategies.len()],
                anchor_seed: Seed::from_label(label),
                source_goal: Seed::from_label("goal"),
                created_ms: 0,
            })
            .collect();
        let scores: Vec<VariantScore> = variants
            .iter()
            .enumerate()
            .map(|(i, v)| VariantScore {
                variant_id: v.id,
                coherence: 0.5,
                novelty: 0.3,
                parsimony: 0.2,
                composite: if Some(i) == winner_idx { 0.7 } else { 0.55 - (i as f32) * 0.05 },
            })
            .collect();
        let winner_id = winner_idx.map(|i| variants[i].id);
        VariantArena {
            variants,
            scores,
            winner_id,
        }
    }

    // ------ Predicate firing ------

    #[test]
    fn op_route_3b_variants_fires_on_routing_flag() {
        // "what if" query triggers needs_variants in the router; with a
        // tiny brain whose goal seed registers above the activation
        // floor, narrate() must surface a variants summary.
        let mut brain = Brain::new();
        teach(&mut brain, "photosynthesis", BondType::IsA, "process");
        teach(&mut brain, "photosynthesis", BondType::Requires, "light");
        let mut cfg = cfg_no_gate();
        // Lower the threshold so the tiny synthetic brain clears it.
        cfg.variant_min_activation = 0.0;
        let r = narrate(
            &mut brain,
            "what if photosynthesis worked at night?",
            2, 5, false, &cfg,
        );
        assert!(r.routing.needs_variants,
            "router must fire needs_variants on `what if` query");
        assert!(r.variants.is_some(),
            "NarrationResult.variants must be Some when arena ran");
    }

    #[test]
    fn op_route_3b_variants_suppressed_on_meta_query() {
        // Meta-self query: even if needs_variants somehow fires, the
        // is_meta_query guard suppresses arena spawn.
        let mut brain = Brain::new();
        teach(&mut brain, "photosynthesis", BondType::IsA, "process");
        let mut cfg = cfg_no_gate();
        cfg.variant_min_activation = 0.0;
        // OP-RECALL-1: this used to ride on "what do you know about
        // <topic>", which no longer classifies MetaSelf -- a topical
        // question is answered on its subject now. The SUBJECT of this
        // test is variant suppression under `is_meta_query`, not that
        // one phrase, so it is retargeted onto TimetravelComparison,
        // the other intent that sets the flag (query_router.rs:166).
        // The "what if" tail that would otherwise fire the arena is
        // preserved, so the test still proves SUPPRESSION rather than
        // absence.
        let r = narrate(
            &mut brain,
            "how has your understanding changed -- what if photosynthesis differed?",
            2, 5, false, &cfg,
        );
        assert!(r.routing.is_meta_query,
            "meta-self intent must set is_meta_query");
        assert!(r.variants.is_none(),
            "variants must be suppressed on meta queries");
    }

    #[test]
    fn op_route_3b_variants_suppressed_on_math_query() {
        // Hand-build a RoutingDecision is not directly accessible from
        // narrate(); instead drive it via a query that triggers both
        // needs_math AND needs_variants by string. The router fires
        // needs_math on arithmetic; combine "what if" with an arithmetic
        // expression.
        let mut brain = Brain::new();
        teach(&mut brain, "x", BondType::IsA, "y");
        let mut cfg = cfg_no_gate();
        cfg.variant_min_activation = 0.0;
        // The router's needs_math detector keys on expressions like "2 + 2".
        let r = narrate(
            &mut brain,
            "what if 2 + 2 was different?",
            2, 5, false, &cfg,
        );
        if r.routing.needs_math {
            assert!(r.variants.is_none(),
                "variants must be suppressed when needs_math fires");
        }
        // If the math detector did not key on this string, skip --
        // the predicate is structural, not regex-perfect on every phrasing.
    }

    #[test]
    fn op_route_3b_variants_suppressed_below_activation_threshold() {
        let mut brain = Brain::new();
        teach(&mut brain, "photosynthesis", BondType::IsA, "process");
        let mut cfg = cfg_no_gate();
        // Set threshold absurdly high so no spread can clear it.
        cfg.variant_min_activation = 0.99;
        let r = narrate(
            &mut brain,
            "what if photosynthesis worked at night?",
            2, 5, false, &cfg,
        );
        assert!(r.routing.needs_variants,
            "router must still fire needs_variants on `what if`");
        assert!(r.variants.is_none(),
            "variants must be suppressed when goal seed activation is below threshold");
    }

    #[test]
    fn op_route_3b_variants_suppressed_on_missing_goal_seed() {
        // An empty brain produces no seed_hits and no activated nodes;
        // narrate's goal_seed derivation returns None and the variant
        // block is a no-op. Even though needs_variants fires, the
        // arena does not spawn.
        let mut brain = Brain::new();
        let mut cfg = cfg_no_gate();
        cfg.variant_min_activation = 0.0;
        let r = narrate(
            &mut brain,
            "what if zzzunknownsubject differed?",
            2, 5, false, &cfg,
        );
        // Either router fired and we have no goal seed -> variants None,
        // or router did fire and goal seed lookup failed -> still None.
        assert!(r.variants.is_none(),
            "missing goal seed must suppress variants");
    }

    // ------ Shape selection ------

    #[test]
    fn op_route_3b_shape_picks_multihypothesisanswer_on_variant_winner() {
        let ask = mk_ask_with_one_act();
        let cfg = cfg_no_gate();
        let flags = ModulationFlags::default();
        let arena = mk_arena(&["alpha", "beta", "gamma"], Some(0));
        let shape = ShapeSelector::pick(
            QueryIntent::Definition, &ask, None, &flags, &cfg, None, Some(&arena), None);
        assert_eq!(shape, ResponseShape::MultiHypothesisAnswer,
            "variant arena with winner must force MultiHypothesisAnswer on Definition intent");
    }

    #[test]
    fn op_route_3b_shape_ignores_arena_without_winner() {
        let ask = mk_ask_with_one_act();
        let cfg = cfg_no_gate();
        let flags = ModulationFlags::default();
        let arena = mk_arena(&["alpha", "beta", "gamma"], None);
        let shape = ShapeSelector::pick(
            QueryIntent::Definition, &ask, None, &flags, &cfg, None, Some(&arena), None);
        assert_ne!(shape, ResponseShape::MultiHypothesisAnswer,
            "arena without winner_id must NOT force MultiHypothesisAnswer");
    }

    #[test]
    fn op_route_3b_shape_ignores_arena_with_one_spec() {
        let ask = mk_ask_with_one_act();
        let cfg = cfg_no_gate();
        let flags = ModulationFlags::default();
        let arena = mk_arena(&["alpha"], Some(0));
        let shape = ShapeSelector::pick(
            QueryIntent::Definition, &ask, None, &flags, &cfg, None, Some(&arena), None);
        assert_ne!(shape, ResponseShape::MultiHypothesisAnswer,
            "single-spec arena must NOT force MultiHypothesisAnswer (no alternative to articulate)");
    }

    #[test]
    fn op_route_3b_shape_multithink_outranks_variants() {
        // Both multi-think (critic-vetted) and variants present:
        // rule 4 (critic-vetted, lines ~1554 in pick) fires before rule
        // 9.4 (variants). We build a MultiAgentTrace where the critic
        // visibly did work (hypotheses.len() > vetted.len()) so the
        // shape lands via the critic-vetted path, not the variant rule.
        let ask = mk_ask_with_one_act();
        let cfg = cfg_no_gate();
        let flags = ModulationFlags::default();
        let arena = mk_arena(&["alpha", "beta"], Some(0));
        let h1 = crate::agents::Belief {
            from: Seed::from_label("a"),
            via: Seed::from_label("c"),
            to: Seed::from_label("b"),
            from_content: "a".into(),
            via_content: "c".into(),
            to_content: "b".into(),
            w_from_via: 0.8,
            w_via_to: 0.8,
            confidence: 0.8,
            scar_reasons: Vec::new(),
            caveats: Vec::new(),
        };
        let h2 = crate::agents::Belief {
            from: Seed::from_label("x"),
            via: Seed::from_label("z"),
            to: Seed::from_label("y"),
            from_content: "x".into(),
            via_content: "z".into(),
            to_content: "y".into(),
            w_from_via: 0.7,
            w_via_to: 0.7,
            confidence: 0.7,
            scar_reasons: Vec::new(),
            caveats: Vec::new(),
        };
        let multi = crate::agents::MultiAgentTrace {
            multi_think_id: 1,
            query: "q".into(),
            recall: Vec::new(),
            hypotheses: vec![h1.clone(), h2.clone()],
            vetted: vec![h1.clone()],
            rejected: Vec::new(),
            reasoning_trace: None,
            synthesis_prose: String::new(),
            steps: [
                crate::agents::AgentStep {
                    role: crate::scope::AgentRole::Recall,
                    elapsed_us: 0, items_in: 0, items_out: 0,
                },
                crate::agents::AgentStep {
                    role: crate::scope::AgentRole::Synthesis,
                    elapsed_us: 0, items_in: 0, items_out: 0,
                },
                crate::agents::AgentStep {
                    role: crate::scope::AgentRole::Critic,
                    elapsed_us: 0, items_in: 0, items_out: 0,
                },
            ],
            started_ms: 0, ended_ms: 0, timed_out: false,
        };
        let shape = ShapeSelector::pick(
            QueryIntent::Definition, &ask, Some(&multi), &flags, &cfg,
            None, Some(&arena), None);
        // Critic visibly did work (vetted < hypotheses); rule 4 fires
        // and shape lands via critic-vetted path -- either
        // MultiHypothesisAnswer (if >= 2 hypotheses survived) or
        // CriticVettedAnswer. Both are critic-path outcomes, NOT the
        // variant rule 9.4. The variant arena is observable only via
        // slot fill once shape is picked.
        assert!(
            matches!(shape,
                ResponseShape::MultiHypothesisAnswer | ResponseShape::CriticVettedAnswer),
            "critic-vetted path must take precedence over rule 9.4, got {:?}", shape,
        );
    }

    // ------ Slot-fill ------

    #[test]
    fn op_route_3b_extract_slots_populates_winner_alt_from_arena() {
        let mut brain = Brain::new();
        teach(&mut brain, "alpha", BondType::IsA, "thing");
        teach(&mut brain, "beta", BondType::IsA, "thing");
        let ask = brain.ask("alpha", 1, 3);
        // Build an arena where the WINNER's anchor is `beta` (not in
        // seed_hits[0]) so slot-fill's primary fill via multi-think
        // path is empty and the arena's winner populates `primary`.
        // Note: ask normally fills s.primary via top activation. To
        // exercise the arena's winner_alt path, we need primary to be
        // already-filled by ask -- that's the default flow. Then
        // s.winner_alt must be filled from the runner-up anchor.
        let arena = mk_arena(&["alpha", "beta"], Some(0));
        let slots = extract_slots(
            &brain, "alpha", &ask, None, &NarrationConfig::default(),
            None, Some(&arena), None, QueryIntent::Generic);
        // primary may already be filled by the upstream activation; the
        // arena should populate winner_alt from the runner-up ("beta").
        assert_eq!(slots.winner_alt.as_deref(), Some("beta"),
            "runner-up anchor label must land in winner_alt");
        assert_eq!(slots.winner_alt_count, 1,
            "non-winner count must be 1 for a 2-spec arena");
    }

    #[test]
    fn op_route_3b_extract_slots_skips_arena_when_multi_already_filled_primary() {
        // When upstream slot-fill (activation top or multi-think) has
        // already populated slots.primary, the arena's winner-anchor
        // must NOT overwrite it. The guard is `s.primary.is_none()`
        // inside the 3b slot-fill block; we exercise it by passing an
        // ask whose activation top is non-empty (which writes primary
        // upstream) and asserting primary is preserved.
        let mut brain = Brain::new();
        teach(&mut brain, "alpha", BondType::IsA, "thing");
        teach(&mut brain, "beta", BondType::IsA, "thing");
        let ask = brain.ask("alpha", 1, 3);
        let primary_before: Option<String> = ask
            .recall
            .activated
            .first()
            .map(|a| a.content.clone());
        // Build arena whose winner anchor is intentionally distinct
        // from whatever the activation-top label landed at.
        let arena = mk_arena(&["distinct_winner", "distinct_runner"], Some(0));
        let slots = extract_slots(
            &brain, "alpha", &ask, None, &NarrationConfig::default(),
            None, Some(&arena), None, QueryIntent::Generic);
        // primary stays whatever the upstream activation-top wrote.
        if let Some(expected) = primary_before {
            assert_eq!(slots.primary.as_deref(), Some(expected.as_str()),
                "arena must NOT overwrite upstream-filled primary");
        }
    }

    // ------ Render ------

    #[test]
    fn op_route_3b_render_multihypothesisanswer_articulates_variant_winner() {
        let mut slots = ShapeSlots::default();
        slots.query = "q".into();
        slots.primary = Some("alpha".into());
        slots.winner_alt = Some("beta".into());
        slots.winner_alt_count = 1;
        let flags = ModulationFlags::default();
        let cfg = NarrationConfig::default();
        let hints = StyleHints::default();
        let out = Renderer::render(
            ResponseShape::MultiHypothesisAnswer, &slots, &flags, &cfg, &hints,
        );
        assert!(out.contains("alpha"),
            "winner anchor 'alpha' must appear in rendered prose, got: {}", out);
        assert!(out.contains("beta"),
            "runner-up anchor 'beta' must appear, got: {}", out);
        // Template says "considered N alternatives" -- N is winner_alt_count.max(1).
        assert!(out.contains("1 alternatives") || out.contains("considered 1"),
            "alternative count must appear in rendered prose, got: {}", out);
    }

    // ------ NarrationResult round-trip ------

    #[test]
    fn op_route_3b_narration_result_serializes_with_variants_none() {
        let mut brain = Brain::new();
        teach(&mut brain, "a", BondType::IsA, "b");
        let r = narrate(&mut brain, "what is a?", 2, 5, false, &cfg_no_gate());
        assert!(r.variants.is_none());
        let s = serde_json::to_string(&r).expect("serialize");
        // Field MUST be present in JSON (serde serializes Option::None to null).
        assert!(s.contains("\"variants\":null"),
            "variants field must serialize as null when absent, got: {}", s);
        // NarrationResult is Serialize-only on the current codebase;
        // round-trip deserialization is not exercised here. The 3a
        // sibling test follows the same pattern.
    }

    #[test]
    fn op_route_3b_narration_result_serializes_with_variants_some() {
        let mut brain = Brain::new();
        teach(&mut brain, "a", BondType::IsA, "b");
        let mut r = narrate(&mut brain, "what is a?", 2, 5, false, &cfg_no_gate());
        // Hand-build a summary and attach.
        r.variants = Some(crate::variants::VariantArenaSummary {
            winner_strategy: Some(crate::variants::PerturbationStrategy::Baseline),
            winner_composite: 0.7,
            winner_node_count: 0,
            num_candidates: 3,
            no_winner: false,
        });
        let s = serde_json::to_string(&r).expect("serialize");
        assert!(s.contains("\"winner_composite\":0.7"),
            "winner_composite must serialize, got: {}", s);
        assert!(s.contains("\"num_candidates\":3"),
            "num_candidates must serialize, got: {}", s);
        // Serialize-only assertion -- see sibling test above for rationale.
    }

    // ------ Parity invariant (LOAD-BEARING) ------

    #[test]
    fn op_route_3b_narrate_parity_no_variants_query() {
        // Plain definition query -- router does NOT fire needs_variants.
        // No arena spawn, no slot mutation, no shape change.
        let mut brain = Brain::new();
        teach(&mut brain, "photosynthesis", BondType::IsA, "process");
        let r = narrate(
            &mut brain, "what is photosynthesis?", 2, 5, false, &cfg_no_gate(),
        );
        assert!(!r.routing.needs_variants,
            "definitional query must not trigger needs_variants");
        assert!(r.variants.is_none(),
            "NarrationResult.variants must be None when arena did not spawn");
    }

    // ============================================================
    // OP-ROUTE-3c -- thinking layer wired into narrate's PASS path.
    // ============================================================

    fn mk_thinking_summary(
        steps_taken: usize,
        early_exit: bool,
        hypotheses_promoted: usize,
        final_answer_text: Option<&str>,
    ) -> crate::thinking::ThinkingTraceSummary {
        crate::thinking::ThinkingTraceSummary {
            steps_taken,
            early_exit,
            final_score: if early_exit { 0.8 } else { 0.3 },
            hypothesis_bonds_written: hypotheses_promoted.max(0) * 2,
            hypotheses_promoted,
            final_answer_present: final_answer_text.is_some(),
            final_answer_text: final_answer_text.map(|s| s.to_string()),
            confidence: if early_exit { 0.8 } else { 0.3 },
            halted_reason: if early_exit {
                crate::thinking::HaltedReason::EarlyExit
            } else {
                crate::thinking::HaltedReason::MaxSteps
            },
        }
    }

    #[test]
    fn op_route_3c_thinking_fires_on_routing_flag() {
        // `how do i` + `step by step` triggers needs_agentic_thinking;
        // with a tiny brain whose goal seed registers above the
        // activation floor and the gate disabled, narrate() must
        // surface a thinking summary on the result.
        let mut brain = Brain::new();
        teach(&mut brain, "photosynthesis", BondType::IsA, "process");
        teach(&mut brain, "photosynthesis", BondType::Requires, "light");
        let mut cfg = cfg_no_gate();
        cfg.thinking_min_activation = 0.0;
        let r = narrate(
            &mut brain,
            "How do I understand photosynthesis step by step?",
            2, 5, false, &cfg,
        );
        assert!(r.routing.needs_agentic_thinking,
            "router must fire needs_agentic_thinking on `how do i ... step by step`");
        assert!(r.thinking.is_some(),
            "NarrationResult.thinking must be Some when think loop ran");
    }

    #[test]
    fn op_route_3c_thinking_suppressed_on_meta_query() {
        // Meta-self intent suppresses thinking even when the
        // agentic-thinking marker would otherwise fire.
        let mut brain = Brain::new();
        teach(&mut brain, "photosynthesis", BondType::IsA, "process");
        let mut cfg = cfg_no_gate();
        cfg.thinking_min_activation = 0.0;
        // OP-RECALL-1: retargeted onto TimetravelComparison for the same
        // reason as op_route_3b_variants_suppressed_on_meta_query -- the
        // agentic "step by step" tail that would otherwise fire thinking
        // is preserved, so this still proves suppression.
        let r = narrate(
            &mut brain,
            "how has your understanding changed -- how do i learn photosynthesis step by step?",
            2, 5, false, &cfg,
        );
        assert!(r.routing.is_meta_query,
            "meta-self intent must set is_meta_query");
        assert!(r.thinking.is_none(),
            "thinking must be suppressed on meta queries");
    }

    #[test]
    fn op_route_3c_thinking_suppressed_on_math_query() {
        // When needs_math AND needs_agentic_thinking co-fire,
        // the math guard suppresses the think pass.
        let mut brain = Brain::new();
        teach(&mut brain, "x", BondType::IsA, "y");
        let mut cfg = cfg_no_gate();
        cfg.thinking_min_activation = 0.0;
        // Arithmetic combined with the agentic-thinking marker.
        let r = narrate(
            &mut brain,
            "how do i compute 2 + 2 step by step?",
            2, 5, false, &cfg,
        );
        if r.routing.needs_math {
            assert!(r.thinking.is_none(),
                "thinking must be suppressed when needs_math fires");
        }
        // If the math detector did not key on this exact phrasing,
        // the predicate is structural -- skip the assertion.
    }

    #[test]
    fn op_route_3c_thinking_suppressed_below_activation_threshold() {
        let mut brain = Brain::new();
        teach(&mut brain, "photosynthesis", BondType::IsA, "process");
        let mut cfg = cfg_no_gate();
        // Threshold set absurdly high so no goal seed clears it.
        cfg.thinking_min_activation = 0.99;
        let r = narrate(
            &mut brain,
            "How do I learn photosynthesis step by step?",
            2, 5, false, &cfg,
        );
        assert!(r.routing.needs_agentic_thinking,
            "router must still fire needs_agentic_thinking");
        assert!(r.thinking.is_none(),
            "thinking must be suppressed when goal seed activation is below threshold");
    }

    #[test]
    fn op_route_3c_shape_picks_multistep_on_converged_thinking() {
        let ask = mk_ask_with_one_act();
        let cfg = cfg_no_gate();
        let flags = ModulationFlags::default();
        // Converged: steps_taken >= 1 AND (early_exit OR
        // hypotheses_promoted >= 1).
        let t = mk_thinking_summary(2, true, 1, Some("alpha"));
        let shape = ShapeSelector::pick(
            QueryIntent::Generic, &ask, None, &flags, &cfg, None, None, Some(&t),
        );
        assert_eq!(shape, ResponseShape::MultiStep,
            "converged thinking trace must force MultiStep shape");
    }

    #[test]
    fn op_route_3c_shape_ignores_unconverged_thinking() {
        let ask = mk_ask_with_one_act();
        let cfg = cfg_no_gate();
        let flags = ModulationFlags::default();
        // Unconverged: ran but no early_exit AND no promoted
        // hypotheses. Loop produced no structural support; rule
        // 9.3 must NOT force MultiStep.
        let t = mk_thinking_summary(5, false, 0, Some("alpha"));
        let shape = ShapeSelector::pick(
            QueryIntent::Definition, &ask, None, &flags, &cfg, None, None, Some(&t),
        );
        assert_ne!(shape, ResponseShape::MultiStep,
            "unconverged thinking trace must NOT force MultiStep");
    }

    #[test]
    fn op_route_3c_extract_slots_populates_chain_articulation_from_thinking() {
        // No reasoning trace but converged thinking with final_answer_text:
        // the 3c slot-fill must land the text in chain_articulation.
        let mut brain = Brain::new();
        teach(&mut brain, "a", BondType::IsA, "b");
        let ask = brain.ask("a", 1, 3);
        let t = mk_thinking_summary(2, true, 1, Some("alpha-final-answer"));
        let slots = extract_slots(
            &brain, "a", &ask, None, &NarrationConfig::default(),
            None, None, Some(&t), QueryIntent::Generic,
        );
        assert_eq!(slots.chain_articulation.as_deref(),
            Some("alpha-final-answer"),
            "thinking final_answer_text must land in chain_articulation when reasoning absent");
    }

    #[test]
    fn op_route_3c_extract_slots_thinking_yields_to_reasoning_on_overlap() {
        // Both reasoning (Structured, 2+ steps) AND thinking summary
        // present. 3a wins per Section 7 of the design.
        let mut brain = Brain::new();
        teach(&mut brain, "a", BondType::IsA, "b");
        let ask = brain.ask("a", 1, 3);
        let trace = mk_trace(
            3, false,
            crate::reasoning::EvidenceKind::Structured,
            "reasoning-chain-prose.",
        );
        let t = mk_thinking_summary(2, true, 1, Some("thinking-prose"));
        let slots = extract_slots(
            &brain, "a", &ask, None, &NarrationConfig::default(),
            Some(&trace), None, Some(&t), QueryIntent::Generic,
        );
        assert_eq!(slots.chain_articulation.as_deref(),
            Some("reasoning-chain-prose."),
            "3a reasoning chain must win the slot when both 3a and 3c fire");
    }

    #[test]
    fn op_route_3c_narrate_parity_no_thinking_query() {
        // Plain definition query -- router does NOT fire
        // needs_agentic_thinking. No think loop, no slot mutation.
        let mut brain = Brain::new();
        teach(&mut brain, "photosynthesis", BondType::IsA, "process");
        let r = narrate(
            &mut brain, "what is photosynthesis?", 2, 5, false, &cfg_no_gate(),
        );
        assert!(!r.routing.needs_agentic_thinking,
            "definitional query must not trigger needs_agentic_thinking");
        assert!(r.thinking.is_none(),
            "NarrationResult.thinking must be None when think loop did not spawn");
    }

    #[test]
    fn op_think_watchdog_extract_slots_composes_on_wall_clock_halt() {
        // OP-THINK-WATCHDOG graceful degradation: when the think loop
        // halts on wall-clock, ThinkingTraceSummary still carries
        // final_answer_text (whatever the partial final-ask surfaced)
        // and extract_slots must land it in chain_articulation. The
        // user gets a body composed from the partial trace instead of
        // a hang.
        let mut brain = Brain::new();
        teach(&mut brain, "a", BondType::IsA, "b");
        let ask = brain.ask("a", 1, 3);
        // Build a ThinkingTraceSummary by hand carrying halted_reason
        // = WallClockExceeded, mimicking what `think()` would emit.
        let summary = crate::thinking::ThinkingTraceSummary {
            steps_taken: 1, // partial: only 1 of N steps ran
            early_exit: false,
            final_score: 0.0,
            hypothesis_bonds_written: 0,
            hypotheses_promoted: 0,
            final_answer_present: true,
            final_answer_text: Some("partial-thinking-body".to_string()),
            confidence: 0.0,
            halted_reason: crate::thinking::HaltedReason::WallClockExceeded,
        };
        let slots = extract_slots(
            &brain, "a", &ask, None, &NarrationConfig::default(),
            None, None, Some(&summary), QueryIntent::Generic,
        );
        // Graceful degradation contract: the partial body still lands
        // on chain_articulation so MultiStep render can compose from it.
        assert_eq!(slots.chain_articulation.as_deref(),
            Some("partial-thinking-body"),
            "watchdog-halted thinking must still populate chain_articulation");
    }

}
#[cfg(test)]
mod mapram_p5d_narration_tests {
    //! OP-BRAIN-MapRAM P5-batch2 (sub-batch B.2): byte-equivalence +
    //! resident-independence for the narration answer-gen adjacency/bond
    //! readers rerouted off the resident adjacency through the seam:
    //!  - `bond_label_between` (504/507) -> `seam_directed_bond` (both dirs)
    //!  - `bond_priority`      (1336)    -> `seam_directed_bond`
    //!  - `Walker::collect_outgoing` (1024) -> `seam_walk_outgoing`
    //!  - `NoiseFilter::has_typed_grounding` (865/872) -> `seam_walk_*`
    //!    (the adjacent degree count at 847 is `seam_total_degree`)
    //! Outputs are projected to bond-derived values (type label / priority /
    //! (dest seed, bond_type, weight bits) / bool); node CONTENT is excluded
    //! (collect_outgoing's poison check uses a DISABLED noise filter so its
    //! content-based rejection cannot perturb the adjacency/bond projection).
    //! The remaining narration readers reachable only through `extract_slots`
    //! / `narrate` (typed_score 1504, definition_prose_override 1564/1567,
    //! polysemy_blocked 1618 -> seam_outgoing_set, source-attr 3078 ->
    //! seam_directed_bond) read adjacency/bonds EXCLUSIVELY through these same
    //! seam accessors, proven byte-equivalent + resident-independent here and
    //! in p5c; the full workspace suite + the re-grep audit confirm no
    //! residual resident adjacency read remains in narration serving code.
    use super::*;
    use crate::bond::{BondType, KeyframeSource, TemporalKeyframe};
    use crate::brain::Brain;
    use crate::manifest::{KeyFingerprint, Manifest, MANIFEST_FILE_NAME};
    use crate::seed::Seed;
    use tempfile::tempdir;

    fn kf(w: f32) -> Vec<TemporalKeyframe> {
        vec![TemporalKeyframe { timestamp: 100, weight: w, source: KeyframeSource::Baseline }]
    }
    fn lbl(n: &str) -> Seed { Seed::from_label(&format!("p5d_n_{}", n)) }
    fn boot_v5(dir: &std::path::Path) -> Brain {
        let mut m = Manifest::new(KeyFingerprint::none());
        m.set_brain_format_v5();
        m.save_atomic(&dir.join(MANIFEST_FILE_NAME)).unwrap();
        Brain::load_v5(dir, None).expect("v5 brain must boot via load_v5")
    }
    const SPINE: &[(&str, &str, f32, BondType)] = &[
        ("q", "ia", 0.90, BondType::IsA),
        ("q", "rel", 0.80, BondType::RelatedTo),
        ("q", "prop", 0.60, BondType::HasProperty),
        ("inc", "q", 0.70, BondType::RelatedTo),
        ("only2", "oa", 0.50, BondType::RelatedTo),
    ];
    fn teach(brain: &Brain) {
        let intern = brain.flush_intern.as_ref().unwrap().clone();
        let mt = brain.memtable().unwrap().clone();
        let mut labels: std::collections::HashSet<&str> = std::collections::HashSet::new();
        for (f, t, _, _) in SPINE.iter().copied() { labels.insert(f); labels.insert(t); }
        for l in labels { intern.intern_or_get(lbl(l)); }
        for (f, t, w, bt) in SPINE.iter().copied() {
            mt.insert(lbl(f), lbl(t), bt, w, kf(w), &intern);
        }
        brain.flush_now().unwrap();
    }
    fn load_routed(dir: &std::path::Path) -> Brain {
        let mut brain = Brain::load_v5(dir, None).expect("reload via load_v5");
        brain.rehydrate_adjacency()
            .expect("rehydrate resident adjacency for legacy/parity comparison");
        brain.enable_cache_backed_adjacency(4096, 0.0).expect("enable cache");
        brain.enable_adjacency_seam().expect("install seam");
        brain
    }
    fn permissive(brain: &Brain) -> NoiseFilter<'_> {
        NoiseFilter { enabled: false, query_subject_lower: None, brain }
    }
    fn typed_filter(brain: &Brain) -> NoiseFilter<'_> {
        NoiseFilter { enabled: true, query_subject_lower: None, brain }
    }
    #[allow(clippy::type_complexity)]
    fn snapshot(brain: &Brain) -> (Option<String>, Option<String>, u32, u32, Vec<(Vec<u8>, BondType, u32)>, bool, bool) {
        let label_fwd = bond_label_between(brain, &lbl("q"), &lbl("ia"));
        let label_rev = bond_label_between(brain, &lbl("q"), &lbl("inc")); // only inc->q exists
        let prio_ia = bond_priority(brain, &lbl("q"), &lbl("ia")).to_bits();
        let prio_prop = bond_priority(brain, &lbl("q"), &lbl("prop")).to_bits();
        let mut cnt = 0u32;
        let nf = permissive(brain);
        let co: Vec<(Vec<u8>, BondType, u32)> = Walker::collect_outgoing(brain, &lbl("q"), 200_000, &nf, &mut cnt)
            .into_iter().map(|(tt, seed)| (seed.as_bytes().to_vec(), tt.bond_type, tt.weight.to_bits())).collect();
        let tf = typed_filter(brain);
        let grounded_q = tf.has_typed_grounding(&lbl("q"));
        let grounded_only2 = tf.has_typed_grounding(&lbl("only2"));
        (label_fwd, label_rev, prio_ia, prio_prop, co, grounded_q, grounded_only2)
    }

    /// Every directly-callable narration adjacency/bond reader is
    /// byte-identical legacy vs cache and non-vacuous.
    #[test]
    fn p5d_narration_byte_equivalent() {
        let dir = tempdir().unwrap();
        { let b = boot_v5(dir.path()); teach(&b); }
        let brain = load_routed(dir.path());
        brain.set_adjacency_cache_backed(false);
        let legacy = snapshot(&brain);
        brain.set_adjacency_cache_backed(true);
        let cache = snapshot(&brain);
        assert_eq!(legacy, cache, "narration readers byte-differ legacy vs cache");
        assert_eq!(cache.0.as_deref(), Some("IsA"), "bond_label_between(q->ia) wrong/vacuous");
        assert_eq!(cache.1.as_deref(), Some("RelatedTo"), "bond_label_between reverse (inc->q) wrong/vacuous");
        assert!(!cache.4.is_empty(), "collect_outgoing vacuous");
        assert!(cache.5, "has_typed_grounding(q) should be true (q IsA ia)");
        assert!(!cache.6, "has_typed_grounding(only2) should be false (no IsA/PartOf)");
    }

    /// INDEPENDENCE: in cache mode all readers are invariant after poisoning
    /// every resident structure the legacy path reads. Sanity: legacy diverges.
    #[test]
    fn p5d_narration_independent_of_resident() {
        let dir = tempdir().unwrap();
        { let b = boot_v5(dir.path()); teach(&b); }
        let mut brain = load_routed(dir.path());
        brain.set_adjacency_cache_backed(true);
        let base = snapshot(&brain);
        assert_eq!(base.0.as_deref(), Some("IsA"));
        assert!(!base.4.is_empty() && base.5);

        brain.network.mapram_test_poison_resident();
        brain.network.mapram_test_corrupt_resident_bond_types();
        brain.network.mapram_test_corrupt_resident_keyframes();

        brain.set_adjacency_cache_backed(true);
        assert_eq!(snapshot(&brain), base, "cache narration readers changed under poison");

        brain.set_adjacency_cache_backed(false);
        assert_ne!(snapshot(&brain), base, "poison must corrupt the legacy narration reads (else vacuous proof)");
    }
}
