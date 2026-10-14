//! ReasoningAgent — articulate a chain of bonds in natural language.
//!
//! Lu's existing pipeline does *associative recall* (spread activation in
//! `RecallAgent`) and *transitive synthesis* (a→c via b triples in
//! `SynthesisAgent`/`CriticAgent`). What it never produced was a
//! step-by-step *articulation* of why one concept is connected to
//! another. `ReasoningAgent` fills that gap.
//!
//! Inputs:
//! - The query (free-form English).
//! - A `TemporalScope` (defaults to `Current`). Bonds traversed by the
//!   chain are filtered by this scope — `BondView::weight_in_scope` is
//!   the type-enforced safety primitive (also applies the scar overlay).
//!
//! Outputs (`ReasoningTrace`):
//! - The full chain as a sequence of `ReasoningStep`s, each carrying the
//!   real bond it stands for (so the trace is auditable end-to-end).
//! - A natural-language `final_answer` that strings the per-step
//!   articulations together with linking words ("Therefore...",
//!   "Which means...").
//! - An aggregate `confidence` (the minimum weight along the chain — a
//!   chain is only as strong as its weakest link).
//! - `bonds_used`: the canonical (from, BondType, to) audit list.
//!
//! Architectural notes:
//! - ReasoningAgent does NOT add agent-tagged keyframes. It is pure
//!   observation. The chain it returns is read-only over the bond graph;
//!   no scope-isolated writes happen.
//! - The chain is *constructed* via best-first search seeded by
//!   `select_seeds_for_query` (with a `keyword_search` fallback,
//!   identical to `Brain::ask`). Best-first means the search frontier
//!   is a max-heap over aggregate path confidence.
//! - Path-cost combination is a simple `min` of in-scope weights along
//!   the chain — a chain is no stronger than its weakest bond. We
//!   considered product but `min` is more honest for chains of length 4-5
//!   (product collapses to ~0 fast even for moderately strong chains).
//! - Typed bonds (IsA, Causes, ...) are preferred over `RelatedTo` via
//!   a small per-type priority multiplier; see `bond_priority`.
//! - **Port B**: chain selection prefers candidates whose bond-type
//!   *sequence* matches a recognized reasoning pattern (causal,
//!   hierarchical, contrastive, transformational, property). Chains
//!   whose sequence is heterogeneous (e.g. RelatedTo->Causes hub
//!   drift) are demoted via `ReasoningPattern::Mixed`. See
//!   `classify_pattern` and `chain_score`.
//! - **Port C**: query-intent bias on top of Port B. The surface form
//!   of the query ("why", "what is", "how", "vs", "where from")
//!   is mapped to a `QueryIntent`, and that intent up-weights the
//!   pattern multipliers that semantically match. See `detect_intent`,
//!   `intent_pattern_bias`, and `chain_score_with_intent`.
//! - **Port D**: epistemic calibration on Mixed (associative-only)
//!   chains. `EvidenceKind::Associative` and the hedged closing
//!   ("associative-only: co-occurrence evidence") replace the confident
//!   "Therefore X is connected to Y" entailment claim when the
//!   chain's bond-type sequence classifies as `Mixed`.
//! - **Port E**: multi-anchor chain unification. When a query
//!   produces 2+ seed anchors and at least two of those anchors emit
//!   a coherent (non-Mixed) chain, articulate up to 3 parallel
//!   chains in the response. If two anchor-chains share a non-anchor
//!   node, surface the convergence in prose ("X and Y both lead to
//!   Z"); if disjoint, articulate as parallel observations ("On one
//!   hand X is..., on another Y is..."). Single-anchor queries fall
//!   through to the Port A/B/C/D path unchanged; all-Mixed
//!   multi-anchor queries fall back to Port D's hedged single-chain
//!   output. See `try_multi_anchor_trace` and the
//!   `articulate_*_chains` family.

use crate::agents::RecalledItem;
use crate::bond::BondType;
use crate::brain::Brain;
use crate::scope::TemporalScope;
use crate::seed::Seed;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

// =====================================================================
// Port BR.4 -- recall-time typed-KU runtime config + gated kind reader.
// =====================================================================

/// Runtime knobs that gate Q.1 typed-KU activation in the recall
/// pipeline (chain composer + procedure assembler).
///
/// `typed_ku_enabled = true` (default) reproduces post-Q.1 behavior:
/// each chain leg's `from_kind` / `to_kind` is populated from the
/// underlying node's `KnowledgeContent` variant, and
/// `articulate_bond_typed` appends a `(kind)` tag in the prose.
///
/// `typed_ku_enabled = false` falls through to the pre-Q.1 baseline:
/// the leg builders treat every endpoint as untagged, so
/// `articulate_bond_typed` reduces byte-for-byte to `articulate_bond`
/// -- the original Port-A/B/C/D rendering.
///
/// Storage, hub-cap retention, and decay paths are NOT influenced by
/// this flag (they consult `kind_str()` directly on the brain's
/// own paths, outside `reasoning.rs`).
#[derive(Debug, Clone, PartialEq)]
pub struct RecallConfig {
    pub typed_ku_enabled: bool,
}

impl Default for RecallConfig {
    fn default() -> Self {
        RecallConfig {
            typed_ku_enabled: crate::runtime_config::default_recall_typed_ku_enabled(),
        }
    }
}

/// Conditionally read a node's `kind_str()`, returning `None` when the
/// recall-time typed-KU flag is off. Wraps the six callsites in this
/// file so each leg builder is a one-liner gated through the brain's
/// runtime config.
#[inline]
fn recall_kind_str(
    brain: &Brain,
    node: Option<&crate::view::NodeView<'_>>,
) -> Option<&'static str> {
    if !brain.recall_config.typed_ku_enabled {
        return None;
    }
    node.map(|n| n.kind_str())
}

// =====================================================================
// Public types
// =====================================================================

/// One link in a reasoning chain. The bond it stands for is real —
/// `(from, bond_type, to)` indexes a live `TemporalBond` in the brain's
/// network, and `confidence` is the bond's `weight_in_scope` at the
/// driving `TemporalScope`. Articulations are deterministic given the
/// bond type and node contents.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningStep {
    pub from: Seed,
    pub from_content: String,
    pub bond_type: BondType,
    pub to: Seed,
    pub to_content: String,
    /// In-scope weight of this bond (already scar-aware via
    /// `weight_in_scope`). 0.0 means the bond was scarred or pruned.
    pub confidence: f32,
    /// Pre-rendered natural-language sentence, e.g. "Cat is a kind of
    /// mammal.".
    pub articulation: String,
}

/// Calibration tag for `ReasoningTrace.confidence` (Port D). Marks
/// whether the chain backing `final_answer` matches a recognized
/// reasoning pattern with semantic structure (`Structured`) or is
/// merely an associative co-occurrence chain -- typically
/// all-`RelatedTo` or hub-bridged -- that classified as
/// `ReasoningPattern::Mixed` (`Associative`). The numeric `confidence`
/// field stays semantically the chain's min in-scope edge weight
/// (unchanged from Port A); this tag tells the caller whether to
/// *trust* that number as causal/hierarchical/property/transformational/
/// contrastive evidence or only as associative co-occurrence.
/// `Associative` traces also hedge the `final_answer` prose so the
/// reader doesn't see overly confident wording behind weak-evidence
/// chains.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvidenceKind {
    /// Chain matches one of: Causal, Hierarchical, PropertyChain,
    /// Transformational, Contrastive. `final_answer` uses the standard
    /// confident articulation (Port A/B/C wording, byte-identical).
    Structured,
    /// Chain classifies as `ReasoningPattern::Mixed` -- associative
    /// co-occurrence only. `final_answer` uses hedged wording
    /// ("associative-only" marker, no "Therefore X is connected to Y"
    /// entailment claim).
    Associative,
}

impl Default for EvidenceKind {
    fn default() -> Self {
        EvidenceKind::Structured
    }
}

/// Port H — hard depth cap on chain composition (Learning #578).
/// The relaxed length penalty lets longer chains compete; this
/// constant bounds the search regardless of `ReasoningConfig.max_chain_length`
/// to keep composition cost finite.
pub const MAX_CHAIN_DEPTH: usize = 8;

/// Full reasoning output. The chain is `steps`; the prose form is
/// `final_answer`; `bonds_used` is the audit list; `confidence` is the
/// aggregate (min) over the chain. `evidence_kind` (Port D) tags
/// whether the chain is structured or merely associative.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningTrace {
    pub query: String,
    pub scope: TemporalScope,
    pub steps: Vec<ReasoningStep>,
    pub final_answer: String,
    pub confidence: f32,
    pub bonds_used: Vec<(Seed, BondType, Seed)>,
    /// True when no chain was constructible (e.g. no anchors, or all
    /// outgoing bonds zero-weighted in scope). `final_answer` is then
    /// the no-chain fallback message and `steps` is empty.
    pub no_chain: bool,
    /// Port D: epistemic calibration tag. `Structured` for chains that
    /// match a recognized reasoning pattern; `Associative` for Mixed
    /// chains where prose is hedged. `#[serde(default)]` so older JSON
    /// traces (pre-Port-D) round-trip cleanly into the current shape.
    #[serde(default)]
    pub evidence_kind: EvidenceKind,
}

/// Knobs for one reasoning pass.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningConfig {
    /// Maximum chain length (inclusive). Per the architecture spec,
    /// chains beyond ~5 hops lose semantic meaning fast. Default 5.
    pub max_chain_length: usize,
    /// Minimum chain length we'll bother articulating. Single-hop
    /// chains are still articulable ("X is a Y."), so default is 2 to
    /// match the spec's "length 2-5" guidance — but we'll fall back to
    /// length 1 if no length-2+ chain is constructible. Default 2.
    pub min_chain_length: usize,
    /// Drop bonds whose `weight_in_scope.effective_weight` is at or
    /// below this. Default 0.05.
    pub min_edge_weight: f32,
    /// Cap on how many chain candidates we'll explore before giving up.
    /// Best-first search hits this when the graph is dense and most
    /// frontiers are dead-ends. Default 256.
    pub max_chain_candidates: usize,
    /// Per-anchor outgoing-edge fanout cap. The graph has hub nodes
    /// with O(10²) outgoing edges; without a fanout cap a single
    /// reasoning call can blow up the frontier. Default 16.
    pub max_fanout_per_node: usize,
}

impl Default for ReasoningConfig {
    fn default() -> Self {
        ReasoningConfig {
            max_chain_length: 8,  // Port H: deep-chain ceiling (was 5)
            min_chain_length: 2,
            min_edge_weight: 0.05,
            max_chain_candidates: 256,
            max_fanout_per_node: 16,
        }
    }
}

// =====================================================================
// Top-level entry points
// =====================================================================

/// Run ReasoningAgent on `query` at the given `scope`. The query goes
/// through the same seed-selection cascade as `Brain::ask` —
/// `select_seeds_for_query` first, with a `keyword_search` fallback —
/// so reasoning is anchored on the same nodes the rest of Lu's pipeline
/// would have anchored on.
pub fn reason(
    brain: &mut Brain,
    query: &str,
    scope: TemporalScope,
) -> ReasoningTrace {
    let cfg = ReasoningConfig::default();
    reason_with_config(brain, query, scope, &cfg)
}

/// `reason` with explicit configuration. Most callers want `reason`.
pub fn reason_with_config(
    brain: &mut Brain,
    query: &str,
    scope: TemporalScope,
    cfg: &ReasoningConfig,
) -> ReasoningTrace {
    let intent = detect_intent(query);
    reason_with_explicit_intent(brain, query, scope, cfg, intent)
}

/// `reason` with explicit configuration AND an explicit `QueryIntent`,
/// bypassing `detect_intent`. Useful for testing/debugging the Port C
/// bias in isolation — callers can compare the chain selected under
/// `QueryIntent::Generic` against the chain selected under any other
/// intent on the same query string. Production code should prefer
/// `reason_with_config`, which auto-detects.
pub fn reason_with_explicit_intent(
    brain: &mut Brain,
    query: &str,
    scope: TemporalScope,
    cfg: &ReasoningConfig,
    intent: QueryIntent,
) -> ReasoningTrace {
    let anchors = pick_anchors(brain, query);
    if anchors.is_empty() {
        // Port K retrofit (Port J / curiosity): no anchor at all
        // for the reasoning path. Equivalent to TotalIdk in the
        // ask path. High severity — Lu literally has nothing.
        brain.gap_emitter.emit(crate::gap_signal::GapSignal {
            query: query.to_string(),
            port_origin: "K",
            gap_kind: crate::gap_signal::GapKind::TotalIdk,
            severity: 0.9,
            context: serde_json::json!({
                "reason": "no_anchor_in_reason",
            }),
        });
        return no_chain_trace(query, scope, "I don't have an anchor for this query.");
    }
    // Port I: when intent is Procedural, try assembling an ordered step
    // list before falling through to the chain composer. Returns `None`
    // when fewer than 2 ordered procedural steps are assembled, in
    // which case Port E and Ports A-D handle the query unchanged — so
    // declarative paths stay byte-identical to pre-Port-I.
    if intent == QueryIntent::Procedural {
        if let Some(trace) = try_procedural_trace(brain, query, &scope, cfg, &anchors) {
            return trace;
        }
    }
    // Port L: when intent is Enumerative, try gathering K sibling
    // concepts of the anchor and articulate them as a numbered list.
    // Returns `None` when fewer than 2 siblings are gathered, in
    // which case Port E and Ports A-D handle the query unchanged.
    if intent == QueryIntent::Enumerative {
        if let Some(trace) = try_list_trace(brain, query, &scope, cfg, &anchors) {
            return trace;
        }
        // Port L retrofit (Port J / curiosity): enumerative intent
        // detected but fewer than MIN_LIST_ITEMS siblings could be
        // gathered for any anchor. Surface as a research target.
        brain.gap_emitter.emit(crate::gap_signal::GapSignal {
            query: query.to_string(),
            port_origin: "L",
            gap_kind: crate::gap_signal::GapKind::ThinListFallback,
            severity: 0.55,
            context: serde_json::json!({
                "anchor_count": anchors.len(),
                "min_list_items": MIN_LIST_ITEMS,
            }),
        });
    }
    // Port M: when intent is Compare, articulate as a side-by-side
    // table over shared outgoing typed bonds across N>=2 subjects.
    // Returns `None` when fewer than 2 anchors resolve OR when fewer
    // than MIN_SHARED_BOND_TYPES bond types are shared across all
    // subjects, in which case Port E and Ports A-D handle the query
    // unchanged.
    if intent == QueryIntent::Compare {
        if let Some(trace) = try_comparison_trace(brain, query, &scope, cfg, &anchors) {
            return trace;
        }
        // Port M retrofit (Port J / curiosity): compare intent
        // detected but fewer than MIN_SHARED_BOND_TYPES bond types
        // shared across the subjects.
        brain.gap_emitter.emit(crate::gap_signal::GapSignal {
            query: query.to_string(),
            port_origin: "M",
            gap_kind: crate::gap_signal::GapKind::ComparisonFallback,
            severity: 0.55,
            context: serde_json::json!({
                "anchor_count": anchors.len(),
                "min_shared_bond_types": MIN_SHARED_BOND_TYPES,
            }),
        });
    }
    // Port E: when the query has 2+ anchors and at least two of them
    // emit non-Mixed chains, articulate them in parallel and unify on
    // shared nodes. Returns `None` for single-anchor queries (so the
    // single-anchor path below stays byte-identical to Port A/B/C/D)
    // and for all-Mixed multi-anchor queries (so Port D's hedged
    // single-chain prose is preserved as the fallback).
    if let Some(trace) = try_multi_anchor_trace(brain, query, &scope, cfg, intent, &anchors) {
        return trace;
    }
    let final_trace = match best_chain(brain, &anchors, &scope, cfg, intent) {
        Some(chain) => assemble_trace(query, scope.clone(), chain),
        None => {
            // Single-anchor terminal failure — emit LowConfidence.
            brain.gap_emitter.emit(crate::gap_signal::GapSignal {
                query: query.to_string(),
                port_origin: "J",
                gap_kind: crate::gap_signal::GapKind::LowConfidence,
                severity: 0.7,
                context: serde_json::json!({
                    "anchor_count": anchors.len(),
                    "reason": "no_chain",
                }),
            });
            return no_chain_trace(
                query,
                scope,
                "I don't have a reasoning chain for this — the seeds I found don't connect to anything strong enough to articulate.",
            );
        }
    };
    // Port H retrofit (Port J / curiosity): if intent is Trace but
    // the chain we assembled is below 3 hops, emit a gap signal so
    // curiosity can target deeper articulations.
    if intent == QueryIntent::Trace && final_trace.steps.len() < 3 {
        brain.gap_emitter.emit(crate::gap_signal::GapSignal {
            query: query.to_string(),
            port_origin: "H",
            gap_kind: crate::gap_signal::GapKind::ShallowWhenTraceRequested,
            severity: 0.5,
            context: serde_json::json!({
                "chain_steps": final_trace.steps.len(),
                "min_trace_hops": 3,
            }),
        });
    }
    final_trace
}

/// Variant entry that takes a pre-computed list of `RecalledItem`s
/// (e.g. from `multi_think`'s RecallAgent output). Anchors are the
/// depth-0 items, in the order Recall produced them. Useful when
/// reasoning rides on top of a multi-think pass and we want to avoid
/// running seed selection twice.
pub fn reason_from_recall(
    brain: &Brain,
    query: &str,
    recall: &[RecalledItem],
    scope: TemporalScope,
) -> ReasoningTrace {
    let cfg = ReasoningConfig::default();
    let anchors: Vec<Seed> = recall
        .iter()
        .filter(|r| r.depth == 0)
        .map(|r| r.seed)
        .collect();
    let anchors = if anchors.is_empty() {
        // Fall back to the strongest activations regardless of depth.
        recall.iter().take(3).map(|r| r.seed).collect()
    } else {
        anchors
    };
    if anchors.is_empty() {
        return no_chain_trace(query, scope, "I don't have an anchor for this query.");
    }
    let intent = detect_intent(query);
    // Port I: procedural articulation, same as `reason_with_explicit_intent`.
    if intent == QueryIntent::Procedural {
        if let Some(trace) = try_procedural_trace(brain, query, &scope, &cfg, &anchors) {
            return trace;
        }
    }
    // Port L: same list dispatch as `reason_with_explicit_intent`.
    if intent == QueryIntent::Enumerative {
        if let Some(trace) = try_list_trace(brain, query, &scope, &cfg, &anchors) {
            return trace;
        }
    }
    // Port M: same comparison dispatch as `reason_with_explicit_intent`.
    if intent == QueryIntent::Compare {
        if let Some(trace) = try_comparison_trace(brain, query, &scope, &cfg, &anchors) {
            return trace;
        }
    }
    // Port E: same multi-anchor dispatch as `reason_with_explicit_intent`.
    if let Some(trace) = try_multi_anchor_trace(brain, query, &scope, &cfg, intent, &anchors) {
        return trace;
    }
    match best_chain(brain, &anchors, &scope, &cfg, intent) {
        Some(chain) => assemble_trace(query, scope, chain),
        None => no_chain_trace(
            query,
            scope,
            "I don't have a reasoning chain for this — the seeds I found don't connect to anything strong enough to articulate.",
        ),
    }
}

// =====================================================================
// Seed selection
// =====================================================================

/// Public wrapper around `pick_anchors` for callers that want to
/// inspect what the seed-selection cascade returned for a query
/// without running a chain. Used by the Port E demo example to
/// pre-flight whether multi-anchor will fire on a given query.
pub fn pick_query_anchors(brain: &mut Brain, query: &str) -> Vec<Seed> {
    pick_anchors(brain, query)
}

/// Force the Port D single-chain path: pick anchors, then run
/// `best_chain` once over every anchor as a single seed list (the
/// pre-Port-E behavior). Useful for before/after demos where the
/// caller needs to see what the deployed Port D binary would have
/// produced for a multi-anchor query.
pub fn reason_legacy_single_anchor(
    brain: &mut Brain,
    query: &str,
    scope: TemporalScope,
    cfg: &ReasoningConfig,
    intent: QueryIntent,
) -> ReasoningTrace {
    let anchors = pick_anchors(brain, query);
    if anchors.is_empty() {
        // Port K retrofit (Port J / curiosity): no anchor at all
        // for the reasoning path. Equivalent to TotalIdk in the
        // ask path. High severity — Lu literally has nothing.
        brain.gap_emitter.emit(crate::gap_signal::GapSignal {
            query: query.to_string(),
            port_origin: "K",
            gap_kind: crate::gap_signal::GapKind::TotalIdk,
            severity: 0.9,
            context: serde_json::json!({
                "reason": "no_anchor_in_reason",
            }),
        });
        return no_chain_trace(query, scope, "I don't have an anchor for this query.");
    }
    let final_trace = match best_chain(brain, &anchors, &scope, cfg, intent) {
        Some(chain) => assemble_trace(query, scope.clone(), chain),
        None => {
            // Single-anchor terminal failure — emit LowConfidence.
            brain.gap_emitter.emit(crate::gap_signal::GapSignal {
                query: query.to_string(),
                port_origin: "J",
                gap_kind: crate::gap_signal::GapKind::LowConfidence,
                severity: 0.7,
                context: serde_json::json!({
                    "anchor_count": anchors.len(),
                    "reason": "no_chain",
                }),
            });
            return no_chain_trace(
                query,
                scope,
                "I don't have a reasoning chain for this — the seeds I found don't connect to anything strong enough to articulate.",
            );
        }
    };
    // Port H retrofit (Port J / curiosity): if intent is Trace but
    // the chain we assembled is below 3 hops, emit a gap signal so
    // curiosity can target deeper articulations.
    if intent == QueryIntent::Trace && final_trace.steps.len() < 3 {
        brain.gap_emitter.emit(crate::gap_signal::GapSignal {
            query: query.to_string(),
            port_origin: "H",
            gap_kind: crate::gap_signal::GapKind::ShallowWhenTraceRequested,
            severity: 0.5,
            context: serde_json::json!({
                "chain_steps": final_trace.steps.len(),
                "min_trace_hops": 3,
            }),
        });
    }
    final_trace
}

/// Mirror `agents::run_recall`'s seed-selection cascade WITHOUT running
/// the spread (we don't need the activated set to start a chain — the
/// chain itself walks the graph). Anchors are returned in the order
/// `select_seeds_for_query` produced them (highest-IDF first).
fn pick_anchors(brain: &mut Brain, query: &str) -> Vec<Seed> {
    // Issue A fix internal notes: ensure the archive-backed keyword
    // index is built before seed selection. Idempotent after the first
    // call. Signature changed from &Brain to &mut Brain so the lazy
    // build can fire here even when the MCP startup hoist has not run.
    // OP-022.b: lifted to the Brain-level entry so v5 brains build
    // the keyword index from the node store (v3/v4 delegate to the
    // Network archive walk, unchanged).
    brain.ensure_keyword_index_built();
    let weighted = brain.network.select_seeds_for_query(query);
    if !weighted.is_empty() {
        return weighted.into_iter().map(|(s, _)| s).collect();
    }
    let hits = brain.network.keyword_search(query);
    hits.into_iter().take(3).map(|(s, _)| s).collect()
}

// =====================================================================
// Chain construction (best-first search)
// =====================================================================

/// Internal frontier item. The chain is the bonds traversed so far;
/// `min_weight` is `min` over each leg's in-scope effective weight
/// (the chain's aggregate confidence); `score` is computed by
/// `chain_score`, which is bond-type-sequence aware (Port B) — chains
/// whose bond-type sequence matches a recognized reasoning pattern
/// (Causal / Hierarchical / PropertyChain / Transformational /
/// Contrastive) outrank Mixed (drift) chains of equal raw weight.
#[derive(Debug, Clone)]
struct PartialChain {
    legs: Vec<ChainLeg>,
    min_weight: f32,
    score: f32,
    visited: HashSet<Seed>,
}

#[derive(Debug, Clone)]
struct ChainLeg {
    from: Seed,
    from_content: String,
    /// Q.1 — KnowledgeContent variant tag of `from` node ("definition",
    /// "mechanism", ...). Stored at construction so prose articulation
    /// stays allocation-free downstream.
    from_kind: Option<&'static str>,
    bond_type: BondType,
    to: Seed,
    to_content: String,
    /// Q.1 — KnowledgeContent variant tag of `to` node.
    to_kind: Option<&'static str>,
    weight: f32,
    /// Port G — negation flag for this bond, captured from
    /// `Brain::bond_negated` at construction time. Reasoning paths
    /// that propagate negation (Causal/Hierarchical) XOR these flags
    /// to compute chain parity; articulation flips per leg when the
    /// flag is true.
    negated: bool,
}

fn best_chain(
    brain: &Brain,
    anchors: &[Seed],
    scope: &TemporalScope,
    cfg: &ReasoningConfig,
    intent: QueryIntent,
) -> Option<Vec<ChainLeg>> {
    // Best-first search: always extend the highest-scoring partial
    // chain. Terminal chains (length >= min_chain_length) are
    // collected; we keep extending the frontier until the queue
    // empties or `max_chain_candidates` partials have been popped.
    //
    // Implementation detail: Rust's std `BinaryHeap` requires a `Ord`
    // wrapper, and our score is f32 (no `Ord`). We could wrap with
    // `OrderedFloat` but the chain frontier stays small (<= a few
    // hundred items in practice), so a sort-on-pop strategy is fine
    // and avoids the dependency.
    let mut frontier: Vec<PartialChain> = Vec::new();
    let mut terminals: Vec<PartialChain> = Vec::new();

    // Seed the frontier with depth-1 chains starting at each anchor.
    let depth_one_seed: Vec<PartialChain> = depth_one_chains(brain, anchors, scope, cfg, intent);
    frontier.extend(depth_one_seed.iter().cloned());

    let mut popped = 0_usize;
    loop {
        if frontier.is_empty() {
            break;
        }
        if popped >= cfg.max_chain_candidates {
            break;
        }
        // Pop the highest-scoring partial. Sort+pop is O(n log n) per
        // step but the frontier stays small.
        frontier.sort_by(|a, b| {
            a.score
                .partial_cmp(&b.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let current = frontier.pop().unwrap();
        popped += 1;

        // Record as terminal if it's long enough.
        if current.legs.len() >= cfg.min_chain_length {
            terminals.push(current.clone());
        }

        // Stop extending if we're at the max length.
        if current.legs.len() >= cfg.max_chain_length.min(MAX_CHAIN_DEPTH) {
            continue;
        }

        let last_to = current.legs.last().unwrap().to;
        let last_to_content = current.legs.last().unwrap().to_content.clone();
        // OP-BRAIN-MapRAM P5-b2: outgoing chain extension through the
        // canonical edge-carrying seam. CACHE reconstructs each bond
        // (weight/type/keyframes) from the carried payload; LEGACY borrows
        // the resident bond. Never resident neighbors_*/get_bond in cache
        // mode. Canonical seed-byte order in BOTH modes makes take(cap)
        // deterministic and the projected output identical across paths.
        let neighbors = brain.network.seam_walk_outgoing(&last_to);
        for (to, walk_bond) in neighbors.iter().take(cfg.max_fanout_per_node) {
            if current.visited.contains(to) {
                continue;
            }
            let bond = match walk_bond {
                Some(b) => b.view(),
                None => continue,
            };
            let reading = bond.weight_in_scope(scope, &brain.scar_log);
            if reading.effective_weight <= cfg.min_edge_weight {
                continue;
            }
            let to_node = brain.network.get_node(to);
            let to_content_s = to_node
                .as_ref()
                .map(|n| n.content().to_string())
                .unwrap_or_default();
            // Port BR.4: recall-time typed-KU gate. When the runtime flag
            // is off, both kinds become None so `articulate_bond_typed`
            // reduces to the pre-Q.1 untyped baseline.
            let to_kind_s: Option<&'static str> = recall_kind_str(brain, to_node.as_ref());
            let from_kind_s: Option<&'static str> = recall_kind_str(
                brain,
                brain.network.get_node(&last_to).as_ref(),
            );
            let leg = ChainLeg {
                from: last_to,
                from_content: last_to_content.clone(),
                from_kind: from_kind_s,
                bond_type: bond.bond_type(),
                to: *to,
                to_content: to_content_s,
                to_kind: to_kind_s,
                weight: reading.effective_weight,
                // Port G: capture negation flag for parity propagation.
                negated: brain.bond_negated(&last_to, to, bond.bond_type()),
            };
            let new_min = current.min_weight.min(reading.effective_weight);
            // Port B: score the candidate by its full bond-type sequence,
            // not just the last leg's bond_priority. Pattern-aware.
            let mut new_seq: Vec<BondType> = current
                .legs
                .iter()
                .map(|l| l.bond_type)
                .collect();
            new_seq.push(bond.bond_type());
            let new_score = chain_score_with_intent(new_min, &new_seq, intent);
            let mut new_visited = current.visited.clone();
            new_visited.insert(*to);
            let mut new_legs = current.legs.clone();
            new_legs.push(leg);
            frontier.push(PartialChain {
                legs: new_legs,
                min_weight: new_min,
                score: new_score,
                visited: new_visited,
            });
        }
    }

    // If no terminal of min length was collected, fall back to the
    // single best length-1 chain — better to articulate one bond than
    // none.
    if terminals.is_empty() {
        if depth_one_seed.is_empty() {
            return None;
        }
        let mut depth_one = depth_one_seed;
        depth_one.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        return Some(depth_one.into_iter().next().unwrap().legs);
    }

    // Pick the highest-scoring terminal. Tiebreak by chain length
    // descending (richer articulation) then by seed-bytes for
    // determinism.
    terminals.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| b.legs.len().cmp(&a.legs.len()))
            .then_with(|| {
                a.legs[0]
                    .from
                    .as_bytes()
                    .cmp(b.legs[0].from.as_bytes())
            })
    });
    Some(terminals.into_iter().next().unwrap().legs)
}

/// Build the initial frontier of length-1 partial chains, one per
/// (anchor, outgoing neighbor) pair that survives the scope/scar/weight
/// filters and the per-node fanout cap. Returned in the same order the
/// anchors and their neighbor lists appeared (deterministic).
fn depth_one_chains(
    brain: &Brain,
    anchors: &[Seed],
    scope: &TemporalScope,
    cfg: &ReasoningConfig,
    intent: QueryIntent,
) -> Vec<PartialChain> {
    let mut out: Vec<PartialChain> = Vec::new();
    for anchor in anchors {
        // OP-BRAIN-MapRAM P5-b2: walk via the canonical edge-carrying seam.
        let neighbors = brain.network.seam_walk_outgoing(anchor);
        if neighbors.is_empty() {
            continue;
        }
        let from_content = brain
            .network
            .get_node(anchor)
            .map(|n| n.content().to_string())
            .unwrap_or_default();
        for (to, walk_bond) in neighbors.iter().take(cfg.max_fanout_per_node) {
            if to == anchor {
                continue;
            }
            let bond = match walk_bond {
                Some(b) => b.view(),
                None => continue,
            };
            let reading = bond.weight_in_scope(scope, &brain.scar_log);
            if reading.effective_weight <= cfg.min_edge_weight {
                continue;
            }
            let to_node = brain.network.get_node(to);
            let to_content = to_node
                .as_ref()
                .map(|n| n.content().to_string())
                .unwrap_or_default();
            // Port BR.4: recall-time typed-KU gate.
            let to_kind_s: Option<&'static str> = recall_kind_str(brain, to_node.as_ref());
            let from_kind_s: Option<&'static str> = recall_kind_str(
                brain,
                brain.network.get_node(anchor).as_ref(),
            );
            let mut visited = HashSet::new();
            visited.insert(*anchor);
            visited.insert(*to);
            let leg = ChainLeg {
                from: *anchor,
                from_content: from_content.clone(),
                from_kind: from_kind_s,
                bond_type: bond.bond_type(),
                to: *to,
                to_content,
                to_kind: to_kind_s,
                weight: reading.effective_weight,
                // Port G: read the negation overlay for this bond.
                negated: brain.bond_negated(anchor, to, bond.bond_type()),
            };
            let min_weight = reading.effective_weight;
            // Port B: pattern-aware single-leg score.
            let score = chain_score_with_intent(min_weight, &[bond.bond_type()], intent);
            out.push(PartialChain {
                legs: vec![leg],
                min_weight,
                score,
                visited,
            });
        }
    }
    out
}

// =====================================================================
// Articulation
// =====================================================================

/// Per-bond-type priority multiplier. Typed semantic bonds outrank
/// generic association. These values are deliberately simple — IsA
/// (taxonomic) and Causes (causal) are the two strongest articulators,
/// `RelatedTo` is the weakest.
fn bond_priority(bt: BondType) -> f32 {
    match bt {
        BondType::IsA => 1.5,
        BondType::DefinedAs => 1.5,
        BondType::WorksBy => 1.1,
        BondType::ComposedOf => 1.25,
        BondType::PurposeOf => 1.1,
        BondType::DiffersFrom => 0.95,
        BondType::LeadsTo => 1.4,
        BondType::ExampleOf => 1.05,
        BondType::Causes => 1.4,
        BondType::Produces => 1.3,
        BondType::PartOf => 1.25,
        BondType::HasProperty => 1.2,
        BondType::Requires => 1.15,
        BondType::UsedFor => 1.1,
        BondType::DerivedFrom => 1.05,
        BondType::Opposes => 1.0,
        BondType::ContrastedWith => 0.95,
        BondType::RelatedTo => 0.6,
        BondType::BornIn => 1.5,
        BondType::BornOn => 1.5,
        BondType::DiedIn => 1.5,
        BondType::DiedOn => 1.5,
        BondType::DiscoveredBy => 1.5,
        BondType::FoundedBy => 1.5,
        BondType::FoundedIn => 1.25,
        BondType::HasAltitude => 1.2,
        BondType::HasAspect => 1.2,
        BondType::HasCapital => 1.2,
        BondType::HasOfficialLanguage => 1.2,
        BondType::HasPopulation => 1.2,
        BondType::KnownFor => 1.2,
        BondType::LocatedIn => 1.25,
        BondType::MeasuredIn => 1.2,
        BondType::NamedAfter => 1.5,
        BondType::WrittenBy => 1.5,
        BondType::HasArea => 1.2,
        BondType::HasMethod => 1.2,
        BondType::HasAttribute => 1.2,
        BondType::HasParameter => 1.2,
        BondType::Returns => 1.2,
        BondType::Inherits => 1.5,
        BondType::Calls => 1.0,
        BondType::HasType => 1.2,
        BondType::Implements => 1.0,
        BondType::HasCurrency => 1.2,
        BondType::AuthoredBy => 1.5,
    }
}

// =====================================================================
// Bond-type composition patterns (Port B)
// =====================================================================
//
// Port A landed the chain articulator but its selection scored
// candidates with `min_weight * bond_priority(last_leg_bond_type)`.
// On a real brain that drifts: a hub node with strong RelatedTo edges
// out-scores a semantically-coherent typed chain because the score
// looks at the *last* leg only. Port B fixes that by classifying the
// chain's whole bond-type sequence into one of a few reasoning
// patterns and folding that classification into the chain score.
//
// Patterns (deliberately small and finite — easier to read than a
// regex DSL, and good enough for the 11-bond-type universe):
//
//   Causal         := (Causes | Produces | Requires)+
//                       e.g. fire->Causes->smoke->Produces->soot
//   Hierarchical   := (IsA | PartOf)+ optionally followed by exactly
//                       one descriptor (HasProperty | Causes | Produces)
//                       e.g. cat->IsA->mammal->IsA->animal,
//                            cat->IsA->mammal->HasProperty->fur
//   PropertyChain  := (IsA)* HasProperty (HasProperty)*
//                       e.g. snow->HasProperty->white
//   Transformational := (DerivedFrom | UsedFor | Produces | Requires)+
//                       e.g. wood->UsedFor->paper->DerivedFrom->trees
//   Contrastive    := (ContrastedWith | Opposes)+
//                       e.g. hot->Opposes->cold
//   Mixed          := anything else (the drift case — demoted)
//
// Match order is by strength: Causal > Hierarchical > PropertyChain
// > Transformational > Contrastive. The first matching pattern wins.
// Single-bond chains classify by their lone bond-type's family.

/// A recognized reasoning pattern over a sequence of bond types.
/// Chains whose bond-type sequence matches one of these patterns are
/// preferred over chains that are merely high-weight or hub-traversing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReasoningPattern {
    /// (Causes | Produces | Requires)+ — pure causal cascade.
    Causal,
    /// (IsA | PartOf)+ optionally followed by one descriptor
    /// (HasProperty | Causes | Produces). Captures taxonomy-then-fact.
    Hierarchical,
    /// (IsA)* HasProperty (HasProperty)* — property attribution chain.
    PropertyChain,
    /// (DerivedFrom | UsedFor | Produces | Requires)+ — how things
    /// are made / used / derived.
    Transformational,
    /// (ContrastedWith | Opposes)+ — oppositional relationships.
    Contrastive,
    /// Heterogeneous bond-type sequence — the drift case (e.g. a
    /// RelatedTo edge bridging unrelated concepts). Demoted in scoring.
    Mixed,
}

impl ReasoningPattern {
    /// Multiplier applied to the chain's combined score. Strong
    /// patterns get >1 boosts; Mixed gets a sub-1 multiplier so
    /// hub-drift chains are demoted relative to coherent ones.
    ///
    /// Values are deliberately small integers so the relative ordering
    /// is stable: Causal > Hierarchical > PropertyChain >
    /// Transformational > Contrastive > Mixed.
    pub fn score_multiplier(self) -> f32 {
        match self {
            ReasoningPattern::Causal => 1.6,
            ReasoningPattern::Hierarchical => 1.5,
            ReasoningPattern::PropertyChain => 1.35,
            ReasoningPattern::Transformational => 1.3,
            ReasoningPattern::Contrastive => 1.2,
            ReasoningPattern::Mixed => 0.7,
        }
    }
}

fn is_causal(seq: &[BondType]) -> bool {
    !seq.is_empty()
        && seq.iter().all(|bt| {
            matches!(
                bt,
                BondType::Causes | BondType::Produces | BondType::Requires
            )
        })
}

fn is_hierarchical(seq: &[BondType]) -> bool {
    if seq.is_empty() {
        return false;
    }
    let prefix_len = seq
        .iter()
        .position(|bt| !matches!(bt, BondType::IsA | BondType::PartOf))
        .unwrap_or(seq.len());
    if prefix_len == seq.len() {
        return prefix_len >= 1;
    }
    if prefix_len >= 1 && seq.len() - prefix_len == 1 {
        matches!(
            seq[seq.len() - 1],
            BondType::HasProperty | BondType::Causes | BondType::Produces
        )
    } else {
        false
    }
}

fn is_property_chain(seq: &[BondType]) -> bool {
    !seq.is_empty()
        && seq
            .iter()
            .all(|bt| matches!(bt, BondType::IsA | BondType::HasProperty))
        && seq
            .iter()
            .any(|bt| matches!(bt, BondType::HasProperty))
}

fn is_transformational(seq: &[BondType]) -> bool {
    !seq.is_empty()
        && seq.iter().all(|bt| {
            matches!(
                bt,
                BondType::DerivedFrom
                    | BondType::UsedFor
                    | BondType::Produces
                    | BondType::Requires
            )
        })
}

fn is_contrastive(seq: &[BondType]) -> bool {
    !seq.is_empty()
        && seq
            .iter()
            .all(|bt| matches!(bt, BondType::ContrastedWith | BondType::Opposes))
}

/// Classify a bond-type sequence into the strongest reasoning pattern
/// it matches. The classification is deterministic: patterns are
/// tested in strength order and the first match wins. Empty sequences
/// classify as `Mixed`.
pub fn classify_pattern(seq: &[BondType]) -> ReasoningPattern {
    if is_causal(seq) {
        ReasoningPattern::Causal
    } else if is_hierarchical(seq) {
        ReasoningPattern::Hierarchical
    } else if is_property_chain(seq) {
        ReasoningPattern::PropertyChain
    } else if is_transformational(seq) {
        ReasoningPattern::Transformational
    } else if is_contrastive(seq) {
        ReasoningPattern::Contrastive
    } else {
        ReasoningPattern::Mixed
    }
}

/// Mild penalty for longer chains — a 4-hop chain with the same
/// min-weight as a 2-hop chain is slightly less preferred (more chances
/// for noise, harder to articulate). Shape: 1.0 at length <=2, ~0.91
/// at 3, 0.83 at 4, 0.77 at 5. Doesn't dominate scoring, just nudges.
#[allow(dead_code)] // Port H: kept for tests + legacy linkage; superseded by length_penalty_full
fn length_penalty(len: usize) -> f32 {
    if len <= 2 {
        1.0
    } else {
        1.0 / (1.0 + 0.10 * (len as f32 - 2.0))
    }
}

/// Port H — intent-and-pattern-aware length penalty.  The original
/// `length_penalty` over-suppressed chains of length >= 3; under an
/// explicit deep-trace intent or a typed semantic pattern it taught
/// Lu to ignore long but well-evidenced walks.  Calibration
/// (Learning #578):
///
///   penalty(intent, pattern) =
///     0.02  if intent ∈ {Trace, Procedural}
///     0.04  if pattern ∈ {Causal, Hierarchical, PropertyChain}
///     0.10  default (Mixed, RelatedTo, weak compositions)
///   lowest matched penalty wins.
///
/// Shape: identical to `length_penalty` but with a smaller divisor
/// coefficient when a relaxed bucket matches.  Length 1-2 is always
/// 1.0 (no penalty) so 2-hop direct answers are untouched.
pub fn length_penalty_full(
    intent: QueryIntent,
    pattern: ReasoningPattern,
    len: usize,
) -> f32 {
    if len <= 2 {
        return 1.0;
    }
    let mut coef: f32 = 0.10;
    // Typed-pattern relaxation (informative compositions).
    if matches!(
        pattern,
        ReasoningPattern::Causal
            | ReasoningPattern::Hierarchical
            | ReasoningPattern::PropertyChain
    ) {
        coef = coef.min(0.04);
    }
    // Explicit deep-walk intent.  Procedural is included because the
    // procedural articulator falls back to the chain composer when it
    // cannot find an ordered sequence; under that fallback we still
    // want long chains to be reachable.
    if matches!(intent, QueryIntent::Trace | QueryIntent::Procedural) {
        coef = coef.min(0.02);
    }
    let base = 1.0 / (1.0 + coef * (len as f32 - 2.0));

    // Port H extension beyond Learning #578's literal penalty table:
    // under explicit Trace intent on a typed pattern, apply a small
    // POSITIVE depth bonus on top of the relaxed penalty.  Without
    // this the chain composer's 2-hop PREFIX of any deep chain
    // shares the deep chain's min_weight and wins by length penalty
    // alone — the spec acceptance criterion ("5-hop chain articulates
    // step by step") is structurally unreachable with pure
    // relaxation.  Bonus is gated on Trace intent only so default
    // queries still pick short answers.  Capped via MAX_CHAIN_DEPTH
    // upstream.
    if matches!(intent, QueryIntent::Trace)
        && matches!(
            pattern,
            ReasoningPattern::Causal
                | ReasoningPattern::Hierarchical
                | ReasoningPattern::PropertyChain
        )
    {
        let depth_bonus = 1.0 + 0.08 * (len as f32 - 2.0);
        return base * depth_bonus;
    }

    base
}

/// Combined chain score (Port B). Replaces Port A's
/// `min_weight * bond_priority(last_bond)`. Components:
///
///   pattern_multiplier  — dominant signal; rewards bond-type
///                          sequences that match a recognized
///                          reasoning pattern; demotes Mixed.
///   min_weight          — the chain's weakest in-scope edge weight
///                          (a chain is only as strong as its weakest
///                          link).
///   length_penalty      — mild taper for chains of length >= 3.
///   bond_priority(last) — surviving tertiary signal; preserves the
///                          intuition that an IsA-terminated chain is
///                          a slightly cleaner articulation than a
///                          RelatedTo-terminated one of equal pattern.
pub fn chain_score(min_weight: f32, seq: &[BondType]) -> f32 {
    // Port C kept Port B's public API: this is the intent-agnostic
    // entry point, equivalent to `chain_score_with_intent(_, _, Generic)`.
    chain_score_with_intent(min_weight, seq, QueryIntent::Generic)
}

// =====================================================================
// Query-intent bias (Port C)
// =====================================================================
//
// Port B's pattern multipliers are query-agnostic: they encode the
// intrinsic semantic strength of each bond-type composition, but they
// rank patterns by a fixed preference regardless of what the user
// actually asked. "why does X happen" and "what is X" should not
// retrieve the same chain on a brain that supports both a Causal and a
// Hierarchical chain rooted at X.
//
// Port C adds a *cheap*, *surface-form* intent classifier on the query
// string and folds the resulting `QueryIntent` into chain scoring as a
// secondary multiplier on top of the Port B pattern multiplier. Biases
// stay modest (1.2-1.4x) so they nudge selection without overwhelming
// the structural score — a high-weight Hierarchical chain still
// beats a marginal Causal chain even when the query is a "why".
//
// Detection is a small set of leading-keyword and substring checks.
// We accept that a few clever phrasings won't classify cleanly
// (e.g. "why is X causing Y" reads as both Why and How) — those fall
// back to the more specific match in the priority order below, and
// failing that to `Generic`, where Port B's static multipliers run
// unmodified. The point is the common case, not coverage.
//
// Detection priority (most specific first — first match wins):
//   Compare > Origin > Why > How > What > Generic
//
// (Compare beats Why because "is X different from Y because of Z" is
// primarily a comparison; Origin beats Why because "where does X come
// from" is materially Transformational, not Causal.)

/// Surface-form intent of the user's query. Used to bias chain
/// scoring toward bond-type compositions that semantically match the
/// kind of question being asked.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QueryIntent {
    /// "why X", "what causes X", "how come X" — up-weights Causal.
    Why,
    /// "what is X", "define X" — up-weights Hierarchical and
    /// PropertyChain (taxonomy and property attribution).
    What,
    /// "how does X work", "how do you X" — up-weights Causal and
    /// Transformational (mechanism and process).
    How,
    /// "X vs Y", "difference between", "is X like Y" — up-weights
    /// Contrastive.
    Compare,
    /// "where does X come from", "what is X made of" — up-weights
    /// Transformational.
    Origin,
    /// Port H — explicit deep-trace request.  "trace from X to Y",
    /// "step by step", "end to end", "how does X lead to Y",
    /// "what's the chain from X to Y".  Routes to the standard
    /// chain composer with the relaxed length penalty so multi-hop
    /// causal walks (5+ hops) can win selection.
    Trace,
    /// Port I — procedural / sequence articulation. "how do I X",
    /// "how do you X", "how to X", "steps to X", "walk me through X",
    /// "what's the process for X". Distinguished from `How`
    /// (mechanism: "how does X work") by the imperative/action shape.
    /// Routes through `try_procedural_trace` before the chain composer;
    /// falls back to the standard path when fewer than 2 ordered steps
    /// are assembled.
    Procedural,
    /// Port L — enumerative / list articulation. "what are the
    /// types of X", "list the X", "give me three examples of X",
    /// "name some X". Routes through `try_list_trace` before the
    /// chain composer; falls back to the standard path when fewer
    /// than 2 sibling concepts are gathered.
    Enumerative,
    /// No detectable surface-form intent. Falls through to Port B's
    /// static pattern multipliers with no extra bias.
    Generic,
}

/// Detect query intent from the surface form of `query`. Lightweight
/// keyword/regex matching on a lowercased view of the query — not
/// model-driven NLU. False negatives are fine (they degrade gracefully
/// to `Generic`); false positives are kept rare by ordering checks
/// most-specific-first.
pub fn detect_intent(query: &str) -> QueryIntent {
    let q = query.trim().to_lowercase();
    if q.is_empty() {
        return QueryIntent::Generic;
    }

    // ---- Compare: comparison/contrast surface forms.
    // Substring checks because comparison phrases rarely lead a query.
    if q.contains(" vs ")
        || q.contains(" vs. ")
        || q.starts_with("vs ")
        || q.contains("difference between")
        || q.contains("differences between")
        || q.contains("compared to")
        || q.contains("compare ")
        || q.starts_with("contrast ")
        || q.contains(" versus ")
        || q.contains("is X like Y")
        || q.starts_with("is ") && q.contains(" like ")
        || q.contains(" the same as ")
        || q.contains(" similar to ")
    {
        return QueryIntent::Compare;
    }

    // ---- Origin: provenance / composition.
    if q.starts_with("where does ")
        || q.starts_with("where do ")
        || q.starts_with("where did ")
        || q.contains("come from")
        || q.contains("comes from")
        || q.contains("made of")
        || q.contains("made from")
        || q.contains("derived from")
        || q.contains("origin of")
        || q.contains("originate")
    {
        return QueryIntent::Origin;
    }

    // ---- Why: causal questions.
    if q.starts_with("why ")
        || q.starts_with("why's ")
        || q.starts_with("how come")
        || q.starts_with("what causes ")
        || q.starts_with("what cause ")
        || q.contains(" why does ")
        || q.contains(" why is ")
    {
        return QueryIntent::Why;
    }

    // ---- Trace (Port H): explicit deep-walk request.  Checked
    // BEFORE Procedural so "trace from X to Y" routes to the
    // chain composer with the relaxed penalty rather than the
    // procedural articulator.  Cues are deliberately specific —
    // "trace", "chain from", "step by step", "end to end", and
    // the "how does X lead to Y" / "why does X lead to Y" forms.
    if q.starts_with("trace from ")
        || q.starts_with("trace the path ")
        || q.starts_with("trace the chain ")
        || q.starts_with("trace the route ")
        || q.starts_with("trace through ")
        || q.starts_with("step by step ")
        || q.contains(" step by step")
        || q.starts_with("end to end ")
        || q.contains(" end to end")
        || q.starts_with("from start to finish")
        || q.starts_with("from beginning to end")
        || q.starts_with("what is the chain from ")
        || q.starts_with("what's the chain from ")
        || q.starts_with("what is the path from ")
        || q.starts_with("what's the path from ")
        || q.contains(" lead to ")
        || q.contains(" leads to ")
    {
        return QueryIntent::Trace;
    }

    // ---- Procedural (Port I): action sequence — imperative "how"
    // forms and explicit "steps" cues. Checked BEFORE `How` so that
    // "how do you bake bread" routes to the procedural articulator
    // while "how does an engine work" stays on the mechanism path.
    if q.starts_with("how do i ")
        || q.starts_with("how do you ")
        || q.starts_with("how to ")
        || q.starts_with("how can i ")
        || q.starts_with("how can you ")
        || q.starts_with("steps to ")
        || q.starts_with("steps for ")
        || q.starts_with("what are the steps to ")
        || q.starts_with("what are the steps for ")
        || q.starts_with("what's the process for ")
        || q.starts_with("what is the process for ")
        || q.starts_with("what's the process to ")
        || q.starts_with("what is the process to ")
        || q.starts_with("walk me through ")
        || q.starts_with("guide to ")
        || q.starts_with("guide for ")
        || q.starts_with("procedure for ")
        || q.starts_with("procedure to ")
        || q.starts_with("how would i ")
        || q.starts_with("how would you ")
    {
        return QueryIntent::Procedural;
    }

    // ---- How: mechanism/process questions.
    if q.starts_with("how does ")
        || q.starts_with("how do ")
        || q.starts_with("how did ")
        || q.starts_with("how can ")
        || q.starts_with("how is ")
        || q.starts_with("how are ")
        || q.starts_with("how to ")
        || q.starts_with("how would ")
    {
        return QueryIntent::How;
    }

    // ---- Enumerative (Port L): list/enumeration cues. Checked
    // BEFORE `What` so "what are the {kinds|types|examples} of X"
    // routes to the list articulator while "what is X" stays on
    // the definition path.
    if q.starts_with("list ")
        || q.starts_with("name three ")
        || q.starts_with("name two ")
        || q.starts_with("name some ")
        || q.starts_with("name a few ")
        || q.starts_with("name the ")
        || q.starts_with("name several ")
        || q.starts_with("give me some ")
        || q.starts_with("give me a few ")
        || q.starts_with("give me three ")
        || q.starts_with("give me two ")
        || q.starts_with("give me five ")
        || q.starts_with("give me ten ")
        || q.starts_with("examples of ")
        || q.starts_with("some examples of ")
        || q.starts_with("what are the kinds of ")
        || q.starts_with("what are the types of ")
        || q.starts_with("what are the categories of ")
        || q.starts_with("what are the examples of ")
        || q.starts_with("what are some ")
        || q.starts_with("what kinds of ")
        || q.starts_with("what types of ")
        || q.starts_with("what categories of ")
        || q.starts_with("what examples of ")
        || (q.starts_with("what ") && q.contains(" are there"))
        || (q.starts_with("what ") && q.contains(" are statically typed"))
        || (q.starts_with("what ") && (q.contains(" languages ") ))
        || q.starts_with("what are the parts of ")
        || q.starts_with("what are the planets")
        || q.starts_with("what are the colors of ")
        || q.starts_with("what are the noble gases")
        || q.starts_with("types of ")
        || q.starts_with("kinds of ")
    {
        return QueryIntent::Enumerative;
    }

    // ---- What: definition/identification.
    if q.starts_with("what is ")
        || q.starts_with("what's ")
        || q.starts_with("what are ")
        || q.starts_with("what was ")
        || q.starts_with("what were ")
        || q.starts_with("define ")
        || q.starts_with("definition of ")
        || q.starts_with("who is ")
        || q.starts_with("who's ")
    {
        return QueryIntent::What;
    }

    QueryIntent::Generic
}

/// Intent×Pattern bias multiplier. Returns 1.0 when the intent does
/// not steer the pattern (no boost, no penalty). Boosts are kept in
/// the 1.2-1.4 range so they nudge ordering rather than dominate the
/// Port B structural score.
///
/// Calibration sketch:
///   * Why×Causal:           1.30  (canonical)
///   * What×Hierarchical:     1.30  (taxonomy answers definition)
///   * What×PropertyChain:    1.20  ("X has property" also defines)
///   * How×Causal:            1.25  (mechanism is causal)
///   * How×Transformational:  1.25  (process is transformation)
///   * Compare×Contrastive:   1.40  (canonical, narrowest match)
///   * Origin×Transformational:1.35 (canonical)
pub fn intent_pattern_bias(intent: QueryIntent, pattern: ReasoningPattern) -> f32 {
    match (intent, pattern) {
        (QueryIntent::Why, ReasoningPattern::Causal) => 1.30,
        (QueryIntent::What, ReasoningPattern::Hierarchical) => 1.30,
        (QueryIntent::What, ReasoningPattern::PropertyChain) => 1.20,
        (QueryIntent::How, ReasoningPattern::Causal) => 1.25,
        (QueryIntent::How, ReasoningPattern::Transformational) => 1.25,
        (QueryIntent::Compare, ReasoningPattern::Contrastive) => 1.40,
        (QueryIntent::Origin, ReasoningPattern::Transformational) => 1.35,
        // Generic, or any (intent, pattern) not listed above: no bias.
        _ => 1.0,
    }
}

/// Intent-aware variant of `chain_score`. Layers the Port C intent
/// bias on top of Port B's pattern multiplier; everything else is
/// identical (min_weight, length_penalty, last-leg bond_priority).
///
/// `chain_score(_, _) == chain_score_with_intent(_, _, Generic)`.
pub fn chain_score_with_intent(
    min_weight: f32,
    seq: &[BondType],
    intent: QueryIntent,
) -> f32 {
    let pattern = classify_pattern(seq);
    let last_bonus = seq.last().map(|bt| bond_priority(*bt)).unwrap_or(1.0);
    let bias = intent_pattern_bias(intent, pattern);
    pattern.score_multiplier() * min_weight * length_penalty_full(intent, pattern, seq.len()) * last_bonus * bias
}

/// Render a single bond as natural-language English. The phrasing
/// table mirrors the architecture spec's bond-type articulation rules.
pub fn articulate_bond(bond_type: BondType, from: &str, to: &str) -> String {
    let f = capitalize(from);
    match bond_type {
        BondType::IsA => format!("{} is a kind of {}.", f, to),
        BondType::DefinedAs => format!("{} is defined as {}.", f, to),
        BondType::WorksBy => format!("{} works by {}.", f, to),
        BondType::ComposedOf => format!("{} is composed of {}.", f, to),
        BondType::PurposeOf => format!("{} is for {}.", f, to),
        BondType::DiffersFrom => format!("{} differs from {}.", f, to),
        BondType::LeadsTo => format!("{} leads to {}.", f, to),
        BondType::ExampleOf => format!("{} is an example of {}.", f, to),
        BondType::HasProperty => format!("{} has the property {}.", f, to),
        BondType::Causes => format!("{} causes {}.", f, to),
        BondType::PartOf => format!("{} is part of {}.", f, to),
        BondType::UsedFor => format!("{} is used for {}.", f, to),
        BondType::Produces => format!("{} produces {}.", f, to),
        BondType::Requires => format!("{} requires {}.", f, to),
        BondType::Opposes => format!("{} is the opposite of {}.", f, to),
        BondType::DerivedFrom => format!("{} comes from {}.", f, to),
        BondType::ContrastedWith => format!("{} is unlike {}.", f, to),
        BondType::RelatedTo => format!("{} is associated with {}.", f, to),
        BondType::BornIn => format!("{} was born in {}.", f, to),
        BondType::BornOn => format!("{} was born on {}.", f, to),
        BondType::DiedIn => format!("{} died in {}.", f, to),
        BondType::DiedOn => format!("{} died on {}.", f, to),
        BondType::DiscoveredBy => format!("{} was discovered by {}.", f, to),
        BondType::FoundedBy => format!("{} was founded by {}.", f, to),
        BondType::FoundedIn => format!("{} was founded in {}.", f, to),
        BondType::HasAltitude => format!("{} has altitude {}.", f, to),
        BondType::HasAspect => format!("{} has aspect {}.", f, to),
        BondType::HasCapital => format!("{} has capital {}.", f, to),
        BondType::HasOfficialLanguage => format!("{} has official language {}.", f, to),
        BondType::HasPopulation => format!("{} has population {}.", f, to),
        BondType::KnownFor => format!("{} is known for {}.", f, to),
        BondType::LocatedIn => format!("{} is located in {}.", f, to),
        BondType::MeasuredIn => format!("{} is measured in {}.", f, to),
        BondType::NamedAfter => format!("{} is named after {}.", f, to),
        BondType::WrittenBy => format!("{} was written by {}.", f, to),
        BondType::HasArea => format!("{} has area {}.", f, to),
        BondType::HasMethod => format!("{} has method {}.", f, to),
        BondType::HasAttribute => format!("{} has attribute {}.", f, to),
        BondType::HasParameter => format!("{} has parameter {}.", f, to),
        BondType::Returns => format!("{} returns {}.", f, to),
        BondType::Inherits => format!("{} inherits from {}.", f, to),
        BondType::Calls => format!("{} calls {}.", f, to),
        BondType::HasType => format!("{} has type {}.", f, to),
        BondType::Implements => format!("{} implements {}.", f, to),
        BondType::HasCurrency => format!("{} uses the currency {}.", f, to),
        BondType::AuthoredBy => format!("{} was written by {}.", f, to),
    }
}

/// Q.1 — kind-aware articulation. Renders the same prose as
/// `articulate_bond`, but appends a small (kind) tag for `from` and/or
/// `to` when those nodes carry a narration-tagged KnowledgeContent
/// variant (Definition, Mechanism, Composition, Purpose, Contrast,
/// Causal, Example, Code, Fact, Procedure, Episode, Preference, Image).
/// `Raw`, `Concept`, and `Meta` are intentionally not tagged so the
/// common case stays clean. The base `articulate_bond` signature is
/// preserved untouched (Port A/B/C/D tests rely on it byte-for-byte).
pub fn articulate_bond_typed(
    bond_type: BondType,
    from: &str,
    from_kind: Option<&str>,
    to: &str,
    to_kind: Option<&str>,
) -> String {
    let base = articulate_bond(bond_type, from, to);
    fn is_tagged(k: Option<&str>) -> bool {
        matches!(
            k,
            Some("definition")
                | Some("mechanism")
                | Some("composition")
                | Some("purpose")
                | Some("contrast")
                | Some("causal")
                | Some("example")
                | Some("code")
                | Some("fact")
                | Some("procedure")
                | Some("episode")
                | Some("preference")
                | Some("image")
        )
    }
    let from_tag = if is_tagged(from_kind) { from_kind } else { None };
    let to_tag = if is_tagged(to_kind) { to_kind } else { None };
    match (from_tag, to_tag) {
        (Some(fk), Some(tk)) if fk == tk => format!("{} ({})", base, fk),
        (Some(fk), Some(tk)) => format!("{} ({} -> {})", base, fk, tk),
        (Some(fk), None) => format!("{} ({})", base, fk),
        (None, Some(tk)) => format!("{} (-> {})", base, tk),
        (None, None) => base,
    }
}

fn capitalize(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) => c.to_uppercase().collect::<String>() + chars.as_str(),
        None => String::new(),
    }
}

/// Pick the linking word that introduces step `i` (0-indexed) in a
/// multi-hop chain. Step 0 stands alone; step 1 uses "Which means";
/// the final step uses "Therefore"; intermediate steps use "And".
fn link_word(step_idx: usize, total_steps: usize) -> &'static str {
    if step_idx == 0 {
        ""
    } else if step_idx + 1 == total_steps && total_steps >= 3 {
        "Therefore"
    } else if step_idx == 1 {
        "Which means"
    } else {
        "And"
    }
}

// =====================================================================
// Trace assembly
// =====================================================================

fn assemble_trace(
    query: &str,
    scope: TemporalScope,
    legs: Vec<ChainLeg>,
) -> ReasoningTrace {
    let total = legs.len();

    // Port D: classify the chain's bond-type sequence up front. Mixed
    // multi-hop chains take the hedged prose path; everything else
    // (Causal / Hierarchical / PropertyChain / Transformational /
    // Contrastive, plus any single-leg chain) keeps Port A's confident
    // articulation byte-identical.
    let bond_seq: Vec<BondType> = legs.iter().map(|l| l.bond_type).collect();
    let pattern = classify_pattern(&bond_seq);
    let is_mixed_multihop = pattern == ReasoningPattern::Mixed && total >= 2;
    let evidence_kind = if pattern == ReasoningPattern::Mixed {
        EvidenceKind::Associative
    } else {
        EvidenceKind::Structured
    };

    // Build per-step metadata (steps, bonds_used, min_w) -- shared by
    // both prose paths; the underlying `articulate_bond` table is
    // unchanged, so the per-leg `articulation` strings on each
    // `ReasoningStep` are identical whether the closing prose hedges
    // or not.
    let mut steps: Vec<ReasoningStep> = Vec::with_capacity(total);
    let mut bonds_used: Vec<(Seed, BondType, Seed)> = Vec::with_capacity(total);
    let mut min_w = f32::INFINITY;
    for leg in legs.iter() {
        // Port G: per-leg articulation flips when this bond is
        // overlay-negated. Untyped form first, then the kind-aware
        // wrapper applies the same suffix logic on top.
        let articulation = if leg.negated {
            crate::negation::articulate_bond_negative(
                leg.bond_type,
                &leg.from_content,
                &leg.to_content,
            )
        } else {
            articulate_bond_typed(leg.bond_type, &leg.from_content, leg.from_kind, &leg.to_content, leg.to_kind)
        };
        steps.push(ReasoningStep {
            from: leg.from,
            from_content: leg.from_content.clone(),
            bond_type: leg.bond_type,
            to: leg.to,
            to_content: leg.to_content.clone(),
            confidence: leg.weight,
            articulation,
        });
        bonds_used.push((leg.from, leg.bond_type, leg.to));
        if leg.weight < min_w {
            min_w = leg.weight;
        }
    }

    let final_answer = if is_mixed_multihop {
        articulate_mixed_chain(&legs)
    } else {
        articulate_structured_chain(&legs)
    };

    let confidence = if min_w.is_finite() { min_w } else { 0.0 };
    let scope_note = scope_label(&scope);
    let final_with_scope = match scope_note {
        Some(label) => format!("[{}] {}", label, final_answer),
        None => final_answer,
    };

    ReasoningTrace {
        query: query.to_string(),
        scope,
        steps,
        final_answer: final_with_scope,
        confidence,
        bonds_used,
        no_chain: false,
        evidence_kind,
    }
}

/// Build the confident "Cat is a kind of mammal. Which means mammal is
/// a kind of animal. Therefore cat is connected to animal." prose.
/// This is Port A/B/C's articulation, kept byte-identical so chains
/// that classify as a recognized reasoning pattern (Causal,
/// Hierarchical, PropertyChain, Transformational, Contrastive) read
/// exactly as they did before Port D landed. Single-leg chains also
/// take this path -- a one-bond articulation like "Cat is a kind of
/// mammal." stands on its own without the "Therefore X is connected
/// to Y" closing.
fn articulate_structured_chain(legs: &[ChainLeg]) -> String {
    let total = legs.len();
    let mut sentences: Vec<String> = Vec::with_capacity(total);
    // Port G: per-leg articulation flips for any negated leg, regardless
    // of pattern. Closing summary's polarity flips ONLY when the chain's
    // pattern propagates negation (Causal/Hierarchical) AND parity is odd.
    for (i, leg) in legs.iter().enumerate() {
        let articulation = if leg.negated {
            crate::negation::articulate_bond_negative(
                leg.bond_type,
                &leg.from_content,
                &leg.to_content,
            )
        } else {
            articulate_bond_typed(leg.bond_type, &leg.from_content, leg.from_kind, &leg.to_content, leg.to_kind)
        };
        let linker = link_word(i, total);
        let sentence = if linker.is_empty() {
            articulation.clone()
        } else {
            format!("{} {}", linker, lowercase_first(&articulation))
        };
        sentences.push(sentence);
    }
    let mut final_answer = sentences.join(" ");
    if total >= 2 {
        // Closing summary: tie the start to the end of the chain so the
        // reader gets a one-liner answer at the bottom of the trace.
        let start_lc = legs[0].from_content.to_lowercase();
        let end = legs.last().unwrap().to_content.clone();
        // Port G — parity-XOR over the chain. For Causal/Hierarchical
        // chains the parity flips the closing claim. For other patterns
        // (PropertyChain, Transformational, Contrastive, Mixed) we keep
        // the closing positive — the spec's restricted-propagation rule
        // (Learning #577 LAYER 2 (2b)) prevents over-eager negative
        // conclusions like "the cell does not contain ribosomes" when
        // only the nucleus's containment was negated.
        let bond_seq: Vec<BondType> = legs.iter().map(|l| l.bond_type).collect();
        let pattern = classify_pattern(&bond_seq);
        let pattern_label: &'static str = match pattern {
            ReasoningPattern::Causal => "Causal",
            ReasoningPattern::Hierarchical => "Hierarchical",
            ReasoningPattern::PropertyChain => "PropertyChain",
            ReasoningPattern::Transformational => "Transformational",
            ReasoningPattern::Contrastive => "Contrastive",
            ReasoningPattern::Mixed => "Mixed",
        };
        let neg_flags: Vec<bool> = legs.iter().map(|l| l.negated).collect();
        let parity_odd = crate::negation::chain_parity(&neg_flags);
        let propagates = crate::negation::pattern_propagates_negation(pattern_label);
        let closing = if parity_odd && propagates {
            // Negation-emphatic closing: surfaces the NOT up front so
            // the reader doesn't bury it.
            format!(
                " Therefore {} is NOT connected to {}.",
                start_lc, end
            )
        } else {
            format!(
                " Therefore {} is connected to {}.",
                start_lc, end
            )
        };
        final_answer.push_str(&closing);
    }
    final_answer
}

/// Port D: hedged articulation for Mixed (associative-only) chains of
/// length >= 2. The per-step `articulate_bond` strings are unchanged
/// ("X is associated with Y." for `RelatedTo`, etc. -- the per-bond
/// table is the right shape already), but two things shift:
///
///   1. Inter-step linkers drop "Which means" / "Therefore" entailment
///      language in favor of a neutral "And" continuation. A Mixed
///      chain doesn't *entail* anything; the steps merely co-occur,
///      so the connectives shouldn't claim otherwise.
///   2. The closing line replaces the confident
///      "Therefore X is connected to Y." summary with an explicit
///      acknowledgement that the link is associative-only --
///      co-occurrence evidence, not a causal or hierarchical
///      relationship. The marker phrase "associative-only" is the
///      stable hedging signal callers (and tests) can key off of.
///
/// Lu sounds honest here, not robotic; the closing is one sentence
/// the reader can act on ("this is weak evidence") without dressing
/// it up.
fn articulate_mixed_chain(legs: &[ChainLeg]) -> String {
    let total = legs.len();
    let mut sentences: Vec<String> = Vec::with_capacity(total);
    for (i, leg) in legs.iter().enumerate() {
        // Port G: flip per-leg articulation for negated legs. Mixed
        // chains do not propagate negation through composition, so the
        // closing summary stays neutral regardless of parity.
        let articulation = if leg.negated {
            crate::negation::articulate_bond_negative(
                leg.bond_type,
                &leg.from_content,
                &leg.to_content,
            )
        } else {
            articulate_bond_typed(leg.bond_type, &leg.from_content, leg.from_kind, &leg.to_content, leg.to_kind)
        };
        let sentence = if i == 0 {
            articulation
        } else {
            // Neutral continuation -- no entailment claim.
            format!("And {}", lowercase_first(&articulation))
        };
        sentences.push(sentence);
    }
    let mut final_answer = sentences.join(" ");
    let start_lc = legs[0].from_content.to_lowercase();
    let end_lc = legs.last().unwrap().to_content.to_lowercase();
    final_answer.push_str(&format!(
        " But the link from {} to {} is associative-only: co-occurrence evidence in my knowledge, not a clear causal or hierarchical relationship.",
        start_lc, end_lc
    ));
    final_answer
}

fn no_chain_trace(query: &str, scope: TemporalScope, msg: &str) -> ReasoningTrace {
    ReasoningTrace {
        query: query.to_string(),
        scope,
        steps: Vec::new(),
        final_answer: msg.to_string(),
        confidence: 0.0,
        bonds_used: Vec::new(),
        no_chain: true,
        // No chain to classify; tag as Structured (the default). Callers
        // gate prose decisions on `no_chain` first, so the kind here is
        // mostly cosmetic -- but defaulting to Structured keeps the
        // "Associative" label exclusively for cases where a real Mixed
        // chain was actually articulated.
        evidence_kind: EvidenceKind::Structured,
    }
}

/// Surface a short label when the temporal scope isn't `Current`, so
/// the trace's prose tells the reader they're seeing a historical /
/// session-scoped / variant-scoped view. `None` for `Current` keeps
/// default prose un-cluttered.
fn scope_label(scope: &TemporalScope) -> Option<String> {
    match scope {
        TemporalScope::Current => None,
        TemporalScope::Historical(t) => Some(format!("as of t={}", t)),
        TemporalScope::Range { start, end } => Some(format!("in [{}, {}]", start, end)),
        TemporalScope::Session { id, .. } => Some(format!("in session {}", id)),
        TemporalScope::Variant { id, .. } => Some(format!("in variant {}", id)),
        TemporalScope::Agent { role, multi_think_id, .. } => {
            Some(format!("in {:?}#{}", role, multi_think_id))
        }
    }
}

fn lowercase_first(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) => c.to_lowercase().collect::<String>() + chars.as_str(),
        None => String::new(),
    }
}

// =====================================================================
// Multi-anchor chain unification (Port E)
// =====================================================================
//
// Port A through D operate on the assumption that the query has ONE
// best anchor. Pick that anchor, extract its best chain, articulate.
// That assumption breaks when the user's query implicates multiple
// concepts -- "how are mitochondria and atp related", "compare paris
// and london", "is python like ruby". On those queries, the seed
// selection cascade (`select_seeds_for_query`) returns 2+ anchors with
// real activation weight, but the existing pipeline picks one and
// forgets the rest. The other anchor only shows up if a chain happens
// to traverse it.
//
// Port E lets ReasoningAgent articulate up to 3 parallel chains, one
// per top anchor, and unify them when the chains share a non-anchor
// node. Two prose shapes:
//
//   * **Convergent** -- chains share a non-anchor node Z:
//       "Cat is a kind of mammal. Dog is a kind of mammal. Both cat
//        and dog converge on mammal."
//   * **Disjoint** -- chains have no shared node:
//       "On one hand, fire causes smoke. On another, ice causes water.
//        In my knowledge, fire and ice don't directly converge --
//        each has its own neighborhood."
//
// Fallback rules (preserve Port A/B/C/D behavior on the edges):
//
//   1. < 2 anchors total -> return None; single-anchor path runs
//      unmodified (byte-identical to Port D output).
//   2. < 2 anchors produce a chain at all -> return None; same.
//   3. < 2 anchors produce a *non-Mixed* chain -> return None; the
//      single-anchor path runs `best_chain` over the full anchor list
//      and Port D's hedged "associative-only" prose fires for the
//      Mixed chain it picks.

/// Cap on how many top-ranked anchors Port E will try to articulate
/// in parallel. Three is the upper bound at which the prose still
/// reads cleanly; beyond that the unification line gets unwieldy.
const PORT_E_MAX_ANCHORS: usize = 3;

/// Minimum number of typed (non-Mixed) anchor-chains required for
/// Port E to fire. With fewer than 2, there is nothing to unify and
/// the single-anchor path produces a strictly better trace.
const PORT_E_MIN_TYPED_CHAINS: usize = 2;

/// Try to construct a multi-anchor reasoning trace. Returns `None`
/// when the input doesn't satisfy Port E's preconditions (single
/// anchor, fewer than 2 typed chains, etc.) -- the caller then runs
/// the single-anchor path. Pure observation: no graph writes.
pub fn try_multi_anchor_trace(
    brain: &Brain,
    query: &str,
    scope: &TemporalScope,
    cfg: &ReasoningConfig,
    intent: QueryIntent,
    anchors: &[Seed],
) -> Option<ReasoningTrace> {
    if anchors.len() < PORT_E_MIN_TYPED_CHAINS {
        return None;
    }

    // Per-anchor best-chain extraction: re-run `best_chain` with each
    // anchor in isolation. Cheap relative to seed selection, and it
    // keeps the chain-scoring logic (Port B/C/D) byte-identical for
    // each branch.
    let mut anchor_chains: Vec<(Seed, Vec<ChainLeg>)> = Vec::new();
    let mut seen_anchors: HashSet<Seed> = HashSet::new();
    for anchor in anchors.iter().take(PORT_E_MAX_ANCHORS) {
        if seen_anchors.contains(anchor) {
            continue;
        }
        seen_anchors.insert(*anchor);
        if let Some(chain) = best_chain(brain, &[*anchor], scope, cfg, intent) {
            anchor_chains.push((*anchor, chain));
        }
    }

    if anchor_chains.len() < PORT_E_MIN_TYPED_CHAINS {
        return None;
    }

    // Filter to typed (non-Mixed) chains. If the filter drops us
    // below the minimum, hand the query back to the single-anchor
    // path so Port D's hedged prose fires for whichever single chain
    // the cross-anchor `best_chain` picks.
    let typed: Vec<(Seed, Vec<ChainLeg>)> = anchor_chains
        .into_iter()
        .filter(|(_, legs)| {
            let seq: Vec<BondType> = legs.iter().map(|l| l.bond_type).collect();
            classify_pattern(&seq) != ReasoningPattern::Mixed
        })
        .collect();
    if typed.len() < PORT_E_MIN_TYPED_CHAINS {
        return None;
    }

    Some(assemble_multi_anchor_trace(query, scope.clone(), &typed))
}

/// Build a `ReasoningTrace` whose `final_answer` weaves together
/// multiple anchor-chains. `steps` and `bonds_used` concatenate the
/// per-chain legs in input order so the audit trail stays complete;
/// `confidence` is the global min-weight across every leg in every
/// chain (a multi-anchor trace is no stronger than its weakest
/// single bond, mirroring single-anchor `min` semantics). Always
/// tagged `Structured` -- Port E only fires when every contributing
/// chain is non-Mixed.
fn assemble_multi_anchor_trace(
    query: &str,
    scope: TemporalScope,
    chains: &[(Seed, Vec<ChainLeg>)],
) -> ReasoningTrace {
    let mut steps: Vec<ReasoningStep> = Vec::new();
    let mut bonds_used: Vec<(Seed, BondType, Seed)> = Vec::new();
    let mut min_w = f32::INFINITY;
    for (_, legs) in chains.iter() {
        for leg in legs.iter() {
            let articulation = articulate_bond_typed(leg.bond_type, &leg.from_content, leg.from_kind, &leg.to_content, leg.to_kind);
            steps.push(ReasoningStep {
                from: leg.from,
                from_content: leg.from_content.clone(),
                bond_type: leg.bond_type,
                to: leg.to,
                to_content: leg.to_content.clone(),
                confidence: leg.weight,
                articulation,
            });
            bonds_used.push((leg.from, leg.bond_type, leg.to));
            if leg.weight < min_w {
                min_w = leg.weight;
            }
        }
    }
    let confidence = if min_w.is_finite() { min_w } else { 0.0 };

    let final_answer = match detect_convergence_node(chains) {
        Some(shared) => articulate_convergent_chains(chains, &shared),
        None => articulate_disjoint_chains(chains),
    };

    let scope_note = scope_label(&scope);
    let final_with_scope = match scope_note {
        Some(label) => format!("[{}] {}", label, final_answer),
        None => final_answer,
    };

    ReasoningTrace {
        query: query.to_string(),
        scope,
        steps,
        final_answer: final_with_scope,
        confidence,
        bonds_used,
        no_chain: false,
        // Port E only fires when every anchor-chain is non-Mixed.
        evidence_kind: EvidenceKind::Structured,
    }
}

/// Find a non-anchor node that appears in two or more anchor-chains.
/// Iterates pairwise; the first shared node wins (deterministic in
/// chain order). Returns the node's content for prose articulation
/// (the seed is irrelevant downstream -- we only need the label).
/// Anchors themselves are excluded so a chain whose endpoint happens
/// to be another query anchor doesn't masquerade as a convergence
/// hub.
fn detect_convergence_node(chains: &[(Seed, Vec<ChainLeg>)]) -> Option<String> {
    let anchor_seeds: HashSet<Seed> = chains.iter().map(|(a, _)| *a).collect();
    // Per-chain set of (seed, content) pairs over all non-anchor nodes
    // touched by that chain (the `to` of every leg -- the `from` of
    // the first leg IS the anchor, and every later `from` equals the
    // previous `to`, so iterating `to` covers all distinct
    // intermediates and the endpoint).
    let chain_nodes: Vec<Vec<(Seed, String)>> = chains
        .iter()
        .map(|(_, legs)| {
            legs.iter()
                .filter(|leg| !anchor_seeds.contains(&leg.to))
                .map(|leg| (leg.to, leg.to_content.clone()))
                .collect::<Vec<_>>()
        })
        .collect();

    for (i, set_i) in chain_nodes.iter().enumerate() {
        for (seed_i, content_i) in set_i.iter() {
            for (j, set_j) in chain_nodes.iter().enumerate() {
                if i == j {
                    continue;
                }
                if set_j.iter().any(|(s, _)| s == seed_i) {
                    return Some(content_i.clone());
                }
            }
        }
    }
    None
}

/// Render a single anchor's chain as confident prose without the
/// "Therefore X is connected to Y" closing -- the multi-anchor
/// articulators add their own unifying coda, so the per-chain inner
/// prose should not double up on entailment claims.
fn articulate_chain_inner(legs: &[ChainLeg]) -> String {
    let total = legs.len();
    let mut sentences: Vec<String> = Vec::with_capacity(total);
    for (i, leg) in legs.iter().enumerate() {
        let articulation = articulate_bond_typed(leg.bond_type, &leg.from_content, leg.from_kind, &leg.to_content, leg.to_kind);
        let linker = link_word(i, total);
        let sentence = if linker.is_empty() {
            articulation.clone()
        } else {
            format!("{} {}", linker, lowercase_first(&articulation))
        };
        sentences.push(sentence);
    }
    sentences.join(" ")
}

/// Convergent shape: both anchors lead to a shared non-anchor node.
///
/// Prose template:
///   "<chain 1 inner prose>. <chain 2 inner prose>. Both X and Y
///    converge on Z."
///
/// For 3 anchors that share Z:
///   "<c1>. <c2>. <c3>. X, Y, and Z all converge on <shared>."
///
/// (`shared` is the converged node's content; the joined anchor list
/// reads naturally because the chains were already articulated above.)
fn articulate_convergent_chains(chains: &[(Seed, Vec<ChainLeg>)], shared: &str) -> String {
    let inner_parts: Vec<String> = chains
        .iter()
        .map(|(_, legs)| articulate_chain_inner(legs))
        .collect();
    let mut out = inner_parts.join(" ");
    let anchor_names: Vec<String> = chains
        .iter()
        .map(|(_, legs)| legs[0].from_content.to_lowercase())
        .collect();
    let joined = join_anchor_names(&anchor_names);
    let connector = if anchor_names.len() == 2 { "Both " } else { "" };
    let suffix = if anchor_names.len() == 2 {
        " converge on "
    } else {
        " all converge on "
    };
    out.push_str(&format!(" {}{}{}{}.", connector, joined, suffix, shared));
    out
}

/// Disjoint shape: no two anchor-chains share a non-anchor node.
///
/// Prose template:
///   "On one hand, <chain 1 inner prose>. On another, <chain 2 inner
///    prose>. In my knowledge, X and Y don't directly converge --
///    each has its own neighborhood."
///
/// Three-anchor variant inserts "Separately, ..." for the third
/// branch. The closing line names the disjoint anchors so the reader
/// gets the same one-glance summary the convergent shape gives.
fn articulate_disjoint_chains(chains: &[(Seed, Vec<ChainLeg>)]) -> String {
    let mut sentences: Vec<String> = Vec::with_capacity(chains.len());
    for (i, (_, legs)) in chains.iter().enumerate() {
        let inner = articulate_chain_inner(legs);
        let prefixed = match i {
            0 => format!("On one hand, {}", lowercase_first(&inner)),
            1 => format!("On another, {}", lowercase_first(&inner)),
            _ => format!("Separately, {}", lowercase_first(&inner)),
        };
        sentences.push(prefixed);
    }
    let mut out = sentences.join(" ");
    let anchor_names: Vec<String> = chains
        .iter()
        .map(|(_, legs)| legs[0].from_content.to_lowercase())
        .collect();
    let joined = join_anchor_names(&anchor_names);
    out.push_str(&format!(
        " In my knowledge, {} don't directly converge — each has its own neighborhood.",
        joined
    ));
    out
}

/// English-list join for anchor names: ["a", "b"] -> "a and b";
/// ["a", "b", "c"] -> "a, b, and c"; ["a"] -> "a".
fn join_anchor_names(names: &[String]) -> String {
    match names.len() {
        0 => String::new(),
        1 => names[0].clone(),
        2 => format!("{} and {}", names[0], names[1]),
        _ => {
            let head = &names[..names.len() - 1];
            format!("{}, and {}", head.join(", "), names[names.len() - 1])
        }
    }
}

// =====================================================================
// Port I \u{2014} procedural articulation
// =====================================================================
//
// Procedural queries ("how do I make bread", "steps to install rust")
// are structurally distinct from declarative ones. The chain composer
// (Port A-E) picks ONE best path and articulates it as a single
// chain ("X is a kind of Y. Therefore X connects to Z."). For a
// procedure that's wrong shape: a procedure is a list of ordered
// steps, each with its own justification.
//
// Port I sits between anchor selection and the chain composer in
// `reason_with_explicit_intent`. When `detect_intent` classifies the
// query as `Procedural`, `try_procedural_trace` runs first; if it
// produces a list of >= 2 ordered steps it returns a `ReasoningTrace`
// directly, otherwise it returns `None` and the standard composer
// runs unchanged.
//
// Step gathering uses the existing typed bond table (no new bond
// types). The brain's serialization (rkyv-archived `BondType` enum)
// is therefore untouched. Per the Port I spec (#579 in the project log
// project 9), the procedural-relevant bond types are Requires,
// PartOf, UsedFor, Causes, and Produces; ordering them in that order
// produces a recipe-shaped step list:
//   Requires \u{2014} prerequisites (Step 1, 2, ...)
//   PartOf, UsedFor, Causes \u{2014} components / mechanism (mid-steps)
//   Produces \u{2014} outputs / results (final step)
//
// Future Port I.v2 work (out of scope here): add Precedes / Next
// bond types when the brain ingestion pipeline can write them, then
// switch the topo sort to walk Precedes-first.

/// One ordered step in a procedural articulation. Mirrors
/// `ReasoningStep` but adds an `ordinal` so callers that want to
/// render the list directly (without going through `ReasoningTrace`)
/// have the position. The `articulation` is the same bond-aware prose
/// produced by `articulate_bond_typed`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcedureStep {
    pub ordinal: usize,
    pub from: Seed,
    pub from_content: String,
    pub bond_type: BondType,
    pub to: Seed,
    pub to_content: String,
    pub confidence: f32,
    pub articulation: String,
}

/// Procedural ordering priority for a bond type. Lower = earlier in
/// the step list. `u8::MAX` for non-procedural types means they get
/// filtered out before sort.
fn procedural_order_priority(bt: BondType) -> u8 {
    match bt {
        BondType::Requires => 0,
        BondType::PartOf => 1,
        BondType::UsedFor => 2,
        BondType::Causes => 3,
        BondType::Produces => 4,
        _ => u8::MAX,
    }
}

/// True when the bond type carries procedural-ordering semantics.
/// These are the only edges `assemble_procedure` walks.
fn is_procedural_bond(bt: BondType) -> bool {
    procedural_order_priority(bt) != u8::MAX
}

/// Maximum number of steps Port I will articulate. Beyond ~6-8 a
/// numbered list stops reading as a procedure and starts reading as
/// a list, which is Port L's territory.
const MAX_PROCEDURE_STEPS: usize = 8;

/// Walk the anchor's outgoing procedural-typed bonds and assemble an
/// ordered step list. Returns an empty `Vec` when the anchor has no
/// such bonds in scope; the caller (`try_procedural_trace`) treats
/// `len() < 2` as "fall back to the standard chain composer".
pub fn assemble_procedure(
    brain: &Brain,
    anchor: &Seed,
    scope: &TemporalScope,
    cfg: &ReasoningConfig,
) -> Vec<ProcedureStep> {
    let from_node = match brain.network.get_node(anchor) {
        Some(n) => n,
        None => return Vec::new(),
    };
    let from_content = from_node.content().to_string();
    // Port BR.4: recall-time typed-KU gate.
    let from_kind: Option<&'static str> = if brain.recall_config.typed_ku_enabled {
        Some(from_node.kind_str())
    } else {
        None
    };

    // OP-BRAIN-MapRAM P5-b2: walk via the canonical edge-carrying seam.
    let neighbors = brain.network.seam_walk_outgoing(anchor);

    // Collect procedural-typed candidates. We scan up to twice the
    // standard fanout cap so we don't lose Produces edges to a
    // hub-node Requires cluster, but cap absolutely at 32 to keep
    // the assembly bounded on very dense anchors.
    let scan_limit = (cfg.max_fanout_per_node.saturating_mul(2)).min(32);
    let mut candidates: Vec<(BondType, f32, Seed)> = Vec::new();
    for (to, walk_bond) in neighbors.iter().take(scan_limit) {
        if to == anchor {
            continue;
        }
        let bond = match walk_bond {
            Some(b) => b.view(),
            None => continue,
        };
        if !is_procedural_bond(bond.bond_type()) {
            continue;
        }
        let reading = bond.weight_in_scope(scope, &brain.scar_log);
        if reading.effective_weight <= cfg.min_edge_weight {
            continue;
        }
        candidates.push((bond.bond_type(), reading.effective_weight, *to));
    }

    // Order: ordering priority asc (Requires-first, Produces-last),
    // then weight desc inside each bucket. Stable wrt the natural
    // order of `neighbors_outgoing` for true ties so output is
    // deterministic across runs.
    candidates.sort_by(|a, b| {
        procedural_order_priority(a.0)
            .cmp(&procedural_order_priority(b.0))
            .then_with(|| {
                b.1.partial_cmp(&a.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });
    candidates.truncate(MAX_PROCEDURE_STEPS);

    let mut steps: Vec<ProcedureStep> = Vec::with_capacity(candidates.len());
    for (i, (bt, w, to)) in candidates.into_iter().enumerate() {
        let to_node = brain.network.get_node(&to);
        let to_content = to_node
            .as_ref()
            .map(|n| n.content().to_string())
            .unwrap_or_default();
        // Port BR.4: recall-time typed-KU gate.
        let to_kind: Option<&'static str> = recall_kind_str(brain, to_node.as_ref());
        let articulation = articulate_bond_typed(
            bt,
            &from_content,
            from_kind,
            &to_content,
            to_kind,
        );
        steps.push(ProcedureStep {
            ordinal: i + 1,
            from: *anchor,
            from_content: from_content.clone(),
            bond_type: bt,
            to,
            to_content,
            confidence: w,
            articulation,
        });
    }
    steps
}

/// Render a `Vec<ProcedureStep>` as a numbered step list. Each step
/// is one line: `Step k. <bond articulation>`. The leading sentence
/// names the procedure target so the prose stands alone without
/// caller context.
pub fn articulate_procedure(
    anchor_content: &str,
    steps: &[ProcedureStep],
) -> String {
    if steps.is_empty() {
        return format!(
            "I don't have ordered procedure steps for {}.",
            anchor_content
        );
    }
    let mut out = String::new();
    out.push_str(&format!("Here are the steps for {}:\n", anchor_content));
    for s in steps {
        out.push_str(&format!("Step {}. {}\n", s.ordinal, s.articulation));
    }
    out.trim_end().to_string()
}

/// Port I entry point. Tries each anchor in order; the first one
/// that yields >= 2 procedural steps wins. Returns `None` to signal
/// the dispatcher should fall through to the standard composer.
pub fn try_procedural_trace(
    brain: &Brain,
    query: &str,
    scope: &TemporalScope,
    cfg: &ReasoningConfig,
    anchors: &[Seed],
) -> Option<ReasoningTrace> {
    for anchor in anchors {
        let steps = assemble_procedure(brain, anchor, scope, cfg);
        if steps.len() < 2 {
            continue;
        }
        let anchor_content = brain
            .network
            .get_node(anchor)
            .map(|n| n.content().to_string())
            .unwrap_or_default();
        let final_answer = articulate_procedure(&anchor_content, &steps);
        let scope_note = scope_label(scope);
        let final_with_scope = match scope_note {
            Some(label) => format!("[{}] {}", label, final_answer),
            None => final_answer,
        };
        let mut min_w = f32::INFINITY;
        let reasoning_steps: Vec<ReasoningStep> = steps
            .iter()
            .map(|s| {
                if s.confidence < min_w {
                    min_w = s.confidence;
                }
                ReasoningStep {
                    from: s.from,
                    from_content: s.from_content.clone(),
                    bond_type: s.bond_type,
                    to: s.to,
                    to_content: s.to_content.clone(),
                    confidence: s.confidence,
                    articulation: s.articulation.clone(),
                }
            })
            .collect();
        let bonds_used: Vec<(Seed, BondType, Seed)> = steps
            .iter()
            .map(|s| (s.from, s.bond_type, s.to))
            .collect();
        let confidence = if min_w.is_finite() { min_w } else { 0.0 };
        return Some(ReasoningTrace {
            query: query.to_string(),
            scope: scope.clone(),
            steps: reasoning_steps,
            final_answer: final_with_scope,
            confidence,
            bonds_used,
            no_chain: false,
            // Procedural articulation is structured by construction
            // \u{2014} every step is a typed bond. Tag as Structured so
            // downstream (synthesis, hedging) treats it as confident.
            evidence_kind: EvidenceKind::Structured,
        });
    }
    None
}
// =====================================================================
// Port L — list articulation
// =====================================================================
//
// Enumerative queries ("what are the types of clouds", "list the
// noble gases", "give me three examples of mammals") are structurally
// distinct from declarative ones AND from procedural ones. The chain
// composer (Port A-E) picks ONE best path; Port I produces an ordered
// step list. Neither shape is correct for an enumeration: a list is
// K independent siblings of a category, each with its own micro-trace,
// not a single path and not an ordered sequence.
//
// Port L sits in the dispatcher between Port I (procedural) and Port E
// (multi-anchor unification). When `detect_intent` classifies the
// query as `Enumerative`, `try_list_trace` runs first; if it gathers
// at least 2 sibling concepts of the anchor it returns a
// `ReasoningTrace` whose `final_answer` is a numbered list. Otherwise
// it returns `None` and the standard composer runs unchanged — so
// declarative paths and procedural paths stay byte-identical to
// pre-Port-L when Port L declines.
//
// The reusable rendering primitive Port I introduced (numbered list
// of independent micro-articulations) is the substrate. Port L's
// articulator emits `Item k. <sibling label> — <differentiating
// phrase>` rather than `Step k. ...`, but the bonds-of-list shape is
// identical: each list item is one IsA/PartOf typed-bond walk plus
// optionally one HasProperty walk for differentiation.
//
// Spec: internal notes entry #582 (Port L — Lists & enumerations).
// Predecessors: #579 (Port I), #574 (Port C intent), #611 (K.1 ship),
// #612 (Port I ship).

/// One sibling concept under a category anchor. Mirrors
/// `ProcedureStep`: a single typed-bond hop plus a label and an
/// optional differentiating phrase derived from the sibling's
/// outgoing `HasProperty` bonds. The articulation is the rendered
/// list-item line.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SiblingConcept {
    pub ordinal: usize,
    /// The sibling node (a child of `anchor` via IsA / PartOf).
    pub sibling: Seed,
    pub sibling_label: String,
    /// The category anchor (the `to` side of the IsA / PartOf bond).
    pub anchor: Seed,
    pub anchor_label: String,
    /// Which bond type linked the sibling to the anchor.
    pub bond_type: BondType,
    /// Edge weight at the chosen scope.
    pub confidence: f32,
    /// Highest-weight differentiating outgoing `HasProperty` content
    /// on the sibling, or `None` when no non-shared property is
    /// available. The articulator falls back to a bare label in that
    /// case.
    pub differentiating_phrase: Option<String>,
    /// One-line list rendering: "<sibling> — <phrase>" or just the
    /// sibling label if no phrase.
    pub articulation: String,
}

/// Bond types that participate in the sibling-of-anchor relation.
/// Walked via `neighbors_incoming` on the anchor: nodes that have an
/// IsA / PartOf bond pointing AT the anchor are siblings of each
/// other (children of the same category / members of the same whole).
fn is_list_member_bond(bt: BondType) -> bool {
    matches!(bt, BondType::IsA | BondType::PartOf)
}

/// Default cap K for list output when the cue is generic ("list the X",
/// "what are the X"). Must stay <= MAX_LIST_ITEMS.
const DEFAULT_LIST_K: usize = 7;
/// Hard upper bound on K — beyond ~12 the prose stops reading as a
/// list and starts reading as a wall of text.
const MAX_LIST_ITEMS: usize = 12;
/// `try_list_trace` declines below this. Single-sibling "lists" read
/// worse than the standard chain articulation, so fall through.
const MIN_LIST_ITEMS: usize = 2;
/// Threshold above which a property is considered "shared by nearly all
/// siblings" and therefore non-differentiating. 80% per the spec.
const SHARED_PROPERTY_THRESHOLD: f32 = 0.8;

/// Cue-driven K determination per the Port L spec. "name three" /
/// "give me three" pin K=3 even if 50 siblings exist; "give me some"
/// / "a few" pin K=5; everything else uses DEFAULT_LIST_K. Capped at
/// MAX_LIST_ITEMS.
pub fn list_k_from_query(query: &str) -> usize {
    let q = query.trim().to_lowercase();
    // Numeric cues come first — most specific.
    let numeric_pairs: &[(&str, usize)] = &[
        ("two", 2),
        ("three", 3),
        ("four", 4),
        ("five", 5),
        ("six", 6),
        ("seven", 7),
        ("eight", 8),
        ("nine", 9),
        ("ten", 10),
    ];
    for (word, n) in numeric_pairs {
        let pat_a = format!("name {} ", word);
        let pat_b = format!("give me {} ", word);
        let pat_c = format!("list {} ", word);
        let pat_d = format!("{} examples", word);
        let pat_e = format!("{} kinds", word);
        let pat_f = format!("{} types", word);
        if q.starts_with(&pat_a)
            || q.starts_with(&pat_b)
            || q.starts_with(&pat_c)
            || q.contains(&pat_d)
            || q.contains(&pat_e)
            || q.contains(&pat_f)
        {
            return (*n).min(MAX_LIST_ITEMS);
        }
    }
    // Digit-form cue ("name 3 X", "give me 5 X"). Single-digit only.
    let words: Vec<&str> = q.split_whitespace().collect();
    for (i, w) in words.iter().enumerate() {
        if w.len() == 1 {
            if let Some(d) = w.chars().next().and_then(|c| c.to_digit(10)) {
                let lead = i > 0
                    && matches!(
                        words[i - 1],
                        "name" | "give" | "list" | "top" | "any"
                    );
                if lead && d >= 1 && d <= 12 {
                    return (d as usize).min(MAX_LIST_ITEMS);
                }
            }
        }
    }
    // Soft-quantity cues.
    if q.contains("give me some")
        || q.contains("some examples")
        || q.contains("a few ")
        || q.contains("name a few")
        || q.contains("some kinds")
        || q.contains("some types")
    {
        return 5;
    }
    DEFAULT_LIST_K
}

/// Extract the highest-weight differentiating `HasProperty` phrase for
/// a sibling. Properties that appear on ≥ SHARED_PROPERTY_THRESHOLD
/// fraction of the sibling set are treated as non-differentiating and
/// skipped. Returns `None` when no differentiating property exists.
fn pick_differentiating_phrase(
    brain: &Brain,
    sibling: &Seed,
    scope: &TemporalScope,
    cfg: &ReasoningConfig,
    shared_props: &std::collections::HashSet<String>,
) -> Option<String> {
    // OP-BRAIN-MapRAM P5-b2: walk via the canonical edge-carrying seam.
    let neighbors = brain.network.seam_walk_outgoing(sibling);
    let mut best: Option<(f32, String)> = None;
    for (to, walk_bond) in neighbors.iter().take(cfg.max_fanout_per_node.saturating_mul(2)) {
        if to == sibling {
            continue;
        }
        let bond = match walk_bond {
            Some(b) => b.view(),
            None => continue,
        };
        if bond.bond_type() != BondType::HasProperty {
            continue;
        }
        let reading = bond.weight_in_scope(scope, &brain.scar_log);
        if reading.effective_weight <= cfg.min_edge_weight {
            continue;
        }
        let to_node = match brain.network.get_node(to) {
            Some(n) => n,
            None => continue,
        };
        let phrase = to_node.content().to_string();
        if shared_props.contains(&phrase) {
            continue;
        }
        let w = reading.effective_weight;
        match &best {
            None => best = Some((w, phrase)),
            Some((bw, _)) if w > *bw => best = Some((w, phrase)),
            _ => {}
        }
    }
    best.map(|(_, s)| s)
}

/// Walk the anchor's incoming IsA/PartOf bonds and assemble the
/// top-K sibling concepts. Returns `(siblings, total_known)` where
/// `total_known` is the count of all qualifying siblings before the
/// K-cap (used for the "...and N more" tail).
pub fn gather_siblings(
    brain: &Brain,
    anchor: &Seed,
    scope: &TemporalScope,
    cfg: &ReasoningConfig,
    k: usize,
) -> (Vec<SiblingConcept>, usize) {
    let anchor_node = match brain.network.get_node(anchor) {
        Some(n) => n,
        None => return (Vec::new(), 0),
    };
    let anchor_label = anchor_node.content().to_string();

    // OP-BRAIN-MapRAM P5-b2: incoming walk via the canonical edge-carrying seam.
    let incoming = brain.network.seam_walk_incoming(anchor);

    // Same fanout policy as Port I: scan up to 2x the standard cap so
    // we don't lose siblings on hub categories, absolute ceiling 64
    // (a touch larger than Port I because list categories tend to be
    // hub-shaped on purpose — "mammal", "language", "element").
    let scan_limit = (cfg.max_fanout_per_node.saturating_mul(2)).min(64);
    let mut candidates: Vec<(BondType, f32, Seed, String)> = Vec::new();
    for (from, walk_bond) in incoming.iter().take(scan_limit) {
        if from == anchor {
            continue;
        }
        let bond = match walk_bond {
            Some(b) => b.view(),
            None => continue,
        };
        if !is_list_member_bond(bond.bond_type()) {
            continue;
        }
        let reading = bond.weight_in_scope(scope, &brain.scar_log);
        if reading.effective_weight <= cfg.min_edge_weight {
            continue;
        }
        let from_node = match brain.network.get_node(from) {
            Some(n) => n,
            None => continue,
        };
        let label = from_node.content().to_string();
        candidates.push((bond.bond_type(), reading.effective_weight, *from, label));
    }

    // Order: weight desc * bond_type base_weight (IsA preferred over
    // PartOf at equal raw weight — IsA is structurally a stronger
    // category bond). Stable wrt incoming order for ties.
    candidates.sort_by(|a, b| {
        let sa = a.1 * a.0.base_weight();
        let sb = b.1 * b.0.base_weight();
        sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
    });
    let total_known = candidates.len();

    // First pass: collect property frequencies across the (un-capped)
    // candidate set so we can flag shared properties before the cap.
    // We only consider the top (k * 2) for shared-property analysis to
    // keep this cheap; if a property appears in ≥80% of those, it's
    // shared.
    let analysis_window = (k.saturating_mul(2)).min(total_known);
    let mut prop_counts: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();
    for (_, _, sib_seed, _) in candidates.iter().take(analysis_window) {
        let mut seen_here: std::collections::HashSet<String> =
            std::collections::HashSet::new();
        // OP-BRAIN-MapRAM P5-b2: walk via the canonical edge-carrying seam.
        let nbrs = brain.network.seam_walk_outgoing(sib_seed);
        for (to, walk_bond) in nbrs.iter().take(cfg.max_fanout_per_node.saturating_mul(2)) {
            if to == sib_seed {
                continue;
            }
            let bond = match walk_bond {
                Some(b) => b.view(),
                None => continue,
            };
            if bond.bond_type() != BondType::HasProperty {
                continue;
            }
            let reading = bond.weight_in_scope(scope, &brain.scar_log);
            if reading.effective_weight <= cfg.min_edge_weight {
                continue;
            }
            let to_node = match brain.network.get_node(to) {
                Some(n) => n,
                None => continue,
            };
            let phrase = to_node.content().to_string();
            if seen_here.insert(phrase.clone()) {
                *prop_counts.entry(phrase).or_insert(0) += 1;
            }
        }
    }
    let shared_threshold = if analysis_window > 0 {
        ((analysis_window as f32) * SHARED_PROPERTY_THRESHOLD).ceil() as usize
    } else {
        usize::MAX
    };
    let shared_props: std::collections::HashSet<String> = prop_counts
        .into_iter()
        .filter(|(_, c)| *c >= shared_threshold && shared_threshold >= 2)
        .map(|(k, _)| k)
        .collect();

    candidates.truncate(k.min(MAX_LIST_ITEMS));

    let mut out: Vec<SiblingConcept> = Vec::with_capacity(candidates.len());
    for (i, (bt, w, sib, label)) in candidates.into_iter().enumerate() {
        let phrase =
            pick_differentiating_phrase(brain, &sib, scope, cfg, &shared_props);
        let articulation = match &phrase {
            Some(p) => format!("{} — {}", capitalize(&label), p),
            None => capitalize(&label),
        };
        out.push(SiblingConcept {
            ordinal: i + 1,
            sibling: sib,
            sibling_label: label,
            anchor: *anchor,
            anchor_label: anchor_label.clone(),
            bond_type: bt,
            confidence: w,
            differentiating_phrase: phrase,
            articulation,
        });
    }
    (out, total_known)
}

/// Render a `Vec<SiblingConcept>` as a numbered list. The leading
/// sentence frames the enumeration ("Here are N kinds of X:"); each
/// item is one line. When `total_known` exceeds the rendered count a
/// trailing "...and M more." line is appended so the user knows the
/// list is truncated.
pub fn articulate_list(
    anchor_label: &str,
    siblings: &[SiblingConcept],
    total_known: usize,
) -> String {
    if siblings.is_empty() {
        return format!("I don't have enumerated members of {}.", anchor_label);
    }
    let n = siblings.len();
    // Pick lead phrasing from the dominant bond type in the gathered
    // set. IsA dominant -> "kinds of"; PartOf dominant -> "parts of".
    let isa_count = siblings
        .iter()
        .filter(|s| s.bond_type == BondType::IsA)
        .count();
    let lead = if isa_count >= n - isa_count {
        format!("Here are {} kinds of {}:", n, anchor_label)
    } else {
        format!("Here are {} parts of {}:", n, anchor_label)
    };
    let mut out = String::new();
    out.push_str(&lead);
    out.push('\n');
    for s in siblings {
        out.push_str(&format!("{}. {}\n", s.ordinal, s.articulation));
    }
    let remaining = total_known.saturating_sub(n);
    if remaining > 0 {
        out.push_str(&format!("...and {} more.\n", remaining));
    }
    out.trim_end().to_string()
}

/// Port L entry point. Tries each anchor in order; the first one that
/// yields >= MIN_LIST_ITEMS siblings wins. K is determined from the
/// query cues (`list_k_from_query`). Returns `None` to signal the
/// dispatcher should fall through to the standard composer (or to
/// Port E multi-anchor).
pub fn try_list_trace(
    brain: &Brain,
    query: &str,
    scope: &TemporalScope,
    cfg: &ReasoningConfig,
    anchors: &[Seed],
) -> Option<ReasoningTrace> {
    let k = list_k_from_query(query);
    for anchor in anchors {
        let (siblings, total_known) =
            gather_siblings(brain, anchor, scope, cfg, k);
        if siblings.len() < MIN_LIST_ITEMS {
            continue;
        }
        let anchor_label = brain
            .network
            .get_node(anchor)
            .map(|n| n.content().to_string())
            .unwrap_or_default();
        let final_answer = articulate_list(&anchor_label, &siblings, total_known);
        let scope_note = scope_label(scope);
        let final_with_scope = match scope_note {
            Some(label) => format!("[{}] {}", label, final_answer),
            None => final_answer,
        };
        let mut min_w = f32::INFINITY;
        let reasoning_steps: Vec<ReasoningStep> = siblings
            .iter()
            .map(|s| {
                if s.confidence < min_w {
                    min_w = s.confidence;
                }
                ReasoningStep {
                    // List items walk INCOMING to the anchor; we model
                    // each step as the typed bond from the sibling to
                    // the anchor so audit/bonds_used reflect real edges.
                    from: s.sibling,
                    from_content: s.sibling_label.clone(),
                    bond_type: s.bond_type,
                    to: s.anchor,
                    to_content: s.anchor_label.clone(),
                    confidence: s.confidence,
                    articulation: s.articulation.clone(),
                }
            })
            .collect();
        let bonds_used: Vec<(Seed, BondType, Seed)> = siblings
            .iter()
            .map(|s| (s.sibling, s.bond_type, s.anchor))
            .collect();
        let confidence = if min_w.is_finite() { min_w } else { 0.0 };
        return Some(ReasoningTrace {
            query: query.to_string(),
            scope: scope.clone(),
            steps: reasoning_steps,
            final_answer: final_with_scope,
            confidence,
            bonds_used,
            no_chain: false,
            // List articulation is structured by construction — every
            // item is a typed IsA/PartOf bond. Tag as Structured (no
            // hedging) following Port I's precedent.
            evidence_kind: EvidenceKind::Structured,
        });
    }
    None
}

// =====================================================================
// Port M \u{2014} comparison / table articulation
// =====================================================================
//
// Comparative queries ("compare cumulus and stratus", "X vs Y",
// "difference between dna and rna") are structurally distinct from
// declarative ones AND from Port E's multi-anchor unification. Port E
// articulates two parallel chains and looks for a converging node.
// That shape is wrong for a comparison: the user wants the SAME
// attribute slots presented side-by-side for each subject, not a
// shared narrative.
//
// Per internal notes learning #626 ("Graph Processing First,
// Schema Last", John 2026-04-30), Port M does NOT introduce new bond
// type variants. The associative graph already encodes attributes as
// typed bonds \u{2014} the bond type IS the attribute name, the target
// node IS the value. The articulator walks each subject's outgoing
// typed bonds, intersects the bond_types across all subjects, and
// renders the intersection as a markdown table whose rows are shared
// bond types and whose columns are subjects.
//
// Port M sits in the dispatcher AFTER Port I (procedural) and AFTER
// Port L (enumerative), and BEFORE Port E (multi-anchor). When
// `detect_intent` classifies the query as `Compare`,
// `try_comparison_trace` runs first; if it gathers >= 2 subjects and
// >= MIN_SHARED_BOND_TYPES shared bond types, it returns a
// `ReasoningTrace` whose `final_answer` is a markdown comparison
// table. Otherwise it returns `None` and Port E / Ports A-D handle
// the query unchanged \u{2014} so on a thin-coverage brain the
// comparison degrades gracefully to existing prose rather than
// emitting a degenerate one-row table.
//
// Predecessors: #612 (Port I ship), #613 (Port L ship), #621-#626
// (audit / cleanup / atomic save / graph-processing-first principle).

/// One cell in a comparison row \u{2014} the (subject, target) pair
/// linked by the row's bond type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonCell {
    pub subject: Seed,
    pub subject_label: String,
    pub target: Seed,
    pub target_label: String,
    /// In-scope effective weight of the bond (post-scar).
    pub weight: f32,
}

/// One row of a comparison table: one bond type, one cell per subject.
/// `cells.len() == ComparisonTable::subjects.len()` by construction \u{2014}
/// rows are only emitted when every subject has at least one outgoing
/// bond of this type in scope.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonRow {
    pub bond_type: BondType,
    pub cells: Vec<ComparisonCell>,
}

/// A side-by-side comparison of N subjects across their shared
/// outgoing bond types. Rows are sorted by avg-weight * base_weight
/// descending so the most informative attributes lead the table.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonTable {
    pub subjects: Vec<(Seed, String)>,
    pub rows: Vec<ComparisonRow>,
}

/// Below this many resolved subjects, comparison declines and the
/// dispatcher falls through to single-anchor / Port E paths.
const MIN_COMPARISON_SUBJECTS: usize = 2;
/// Cap on subjects per table. Beyond 4 the table reads as a wall of
/// text \u{2014} if a user really wants a 6-way compare, we'd render in
/// segments, but that's out of scope here.
const MAX_COMPARISON_SUBJECTS: usize = 4;
/// Minimum shared bond types required for `gather_comparison` to
/// succeed. A 1-row table is a degenerate comparison; per spec we
/// fall back to declarative articulation when shared coverage is
/// this thin.
const MIN_SHARED_BOND_TYPES: usize = 2;
/// Hard upper bound on table rows. Beyond ~8 the table stops adding
/// distinguishing power and starts adding noise.
const MAX_COMPARISON_ROWS: usize = 8;

/// Human-readable column header per bond type. Mirrors the role each
/// bond type plays semantically ("Kind of" for IsA, "Property" for
/// HasProperty, etc.). These strings are the only place bond types
/// surface as user-facing labels in Port M output.
fn bond_type_attribute_label(bt: BondType) -> &'static str {
    match bt {
        BondType::IsA => "Kind of",
        BondType::DefinedAs => "Defined as",
        BondType::WorksBy => "Works by",
        BondType::ComposedOf => "Composed of",
        BondType::PurposeOf => "Purpose",
        BondType::DiffersFrom => "Differs from",
        BondType::LeadsTo => "Leads to",
        BondType::ExampleOf => "Example of",
        BondType::HasProperty => "Property",
        BondType::Causes => "Causes",
        BondType::PartOf => "Part of",
        BondType::UsedFor => "Used for",
        BondType::RelatedTo => "Related to",
        BondType::ContrastedWith => "Contrasts with",
        BondType::DerivedFrom => "Derived from",
        BondType::Requires => "Requires",
        BondType::Produces => "Produces",
        BondType::Opposes => "Opposes",
        BondType::BornIn => "Born in",
        BondType::BornOn => "Born on",
        BondType::DiedIn => "Died in",
        BondType::DiedOn => "Died on",
        BondType::DiscoveredBy => "Discovered by",
        BondType::FoundedBy => "Founded by",
        BondType::FoundedIn => "Founded in",
        BondType::HasAltitude => "Altitude",
        BondType::HasAspect => "Aspect",
        BondType::HasCapital => "Capital",
        BondType::HasOfficialLanguage => "Official language",
        BondType::HasPopulation => "Population",
        BondType::KnownFor => "Known for",
        BondType::LocatedIn => "Located in",
        BondType::MeasuredIn => "Measured in",
        BondType::NamedAfter => "Named after",
        BondType::WrittenBy => "Written by",
        BondType::HasArea => "Area",
        BondType::HasMethod => "Has method",
        BondType::HasAttribute => "Has attribute",
        BondType::HasParameter => "Has parameter",
        BondType::Returns => "Returns",
        BondType::Inherits => "Inherits from",
        BondType::Calls => "Calls",
        BondType::HasType => "Has type",
        BondType::Implements => "Implements",
        BondType::HasCurrency => "Currency",
        BondType::AuthoredBy => "Author",
    }
}

/// Walk each subject's outgoing typed bonds, intersect the bond_types
/// across all subjects, and build a `ComparisonTable`. Returns `None`
/// when fewer than `MIN_COMPARISON_SUBJECTS` subjects resolve OR when
/// the shared bond_type intersection has fewer than
/// `MIN_SHARED_BOND_TYPES` entries (graceful degradation on thin brain
/// coverage). Per-subject, we keep ONLY the highest-weight target for
/// each bond type \u{2014} so each cell is a single value, not a list.
pub fn gather_comparison(
    brain: &Brain,
    subjects: &[Seed],
    scope: &TemporalScope,
    cfg: &ReasoningConfig,
) -> Option<ComparisonTable> {
    if subjects.len() < MIN_COMPARISON_SUBJECTS {
        return None;
    }
    let take_n = subjects.len().min(MAX_COMPARISON_SUBJECTS);

    // Per-subject: (seed, label, map<BondType, (target_seed, target_label, weight)>)
    let mut per_subject: Vec<(Seed, String, std::collections::HashMap<BondType, (Seed, String, f32)>)> =
        Vec::with_capacity(take_n);
    for subj in subjects.iter().take(take_n) {
        let node = match brain.network.get_node(subj) {
            Some(n) => n,
            None => continue,
        };
        let label = node.content().to_string();
        // OP-BRAIN-MapRAM P5-b2: walk via the canonical edge-carrying seam.
        let neighbors = brain.network.seam_walk_outgoing(subj);
        // Match Port I/L scan policy: 2x standard fanout, absolute cap.
        let scan_limit = (cfg.max_fanout_per_node.saturating_mul(2)).min(64);
        let mut by_type: std::collections::HashMap<BondType, (Seed, String, f32)> = std::collections::HashMap::new();
        for (to, walk_bond) in neighbors.iter().take(scan_limit) {
            if to == subj {
                continue;
            }
            let bond = match walk_bond {
                Some(b) => b.view(),
                None => continue,
            };
            let reading = bond.weight_in_scope(scope, &brain.scar_log);
            if reading.effective_weight <= cfg.min_edge_weight {
                continue;
            }
            let to_node = match brain.network.get_node(to) {
                Some(n) => n,
                None => continue,
            };
            let to_label = to_node.content().to_string();
            let bt = bond.bond_type();
            // Keep only the highest-weight target for this bond type
            // on this subject. Stable tie-break: first-seen wins.
            match by_type.get(&bt) {
                Some((_, _, w)) if *w >= reading.effective_weight => {}
                _ => {
                    by_type.insert(bt, (*to, to_label, reading.effective_weight));
                }
            }
        }
        if !by_type.is_empty() {
            per_subject.push((*subj, label, by_type));
        }
    }
    if per_subject.len() < MIN_COMPARISON_SUBJECTS {
        return None;
    }

    // Intersection of bond types across all subjects.
    let mut shared: Vec<BondType> = per_subject[0].2.keys().copied().collect();
    for (_, _, by_type) in per_subject.iter().skip(1) {
        shared.retain(|bt| by_type.contains_key(bt));
    }
    if shared.len() < MIN_SHARED_BOND_TYPES {
        return None;
    }

    // Sort shared bond types by avg-weight-across-subjects * base_weight.
    let n_subj = per_subject.len() as f32;
    shared.sort_by(|a, b| {
        let avg_a: f32 = per_subject
            .iter()
            .map(|(_, _, m)| m.get(a).map(|(_, _, w)| *w).unwrap_or(0.0))
            .sum::<f32>()
            / n_subj;
        let avg_b: f32 = per_subject
            .iter()
            .map(|(_, _, m)| m.get(b).map(|(_, _, w)| *w).unwrap_or(0.0))
            .sum::<f32>()
            / n_subj;
        let score_a = avg_a * a.base_weight();
        let score_b = avg_b * b.base_weight();
        score_b
            .partial_cmp(&score_a)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    shared.truncate(MAX_COMPARISON_ROWS);

    let subjects_vec: Vec<(Seed, String)> = per_subject
        .iter()
        .map(|(s, l, _)| (*s, l.clone()))
        .collect();
    let mut rows: Vec<ComparisonRow> = Vec::with_capacity(shared.len());
    for bt in shared {
        let mut cells: Vec<ComparisonCell> = Vec::with_capacity(per_subject.len());
        for (subj, subj_label, by_type) in &per_subject {
            // Safe: every bond type in `shared` is in every subject's map.
            let (target, target_label, weight) = by_type
                .get(&bt)
                .expect("intersection bond_type must be present in every subject");
            cells.push(ComparisonCell {
                subject: *subj,
                subject_label: subj_label.clone(),
                target: *target,
                target_label: target_label.clone(),
                weight: *weight,
            });
        }
        rows.push(ComparisonRow {
            bond_type: bt,
            cells,
        });
    }
    Some(ComparisonTable {
        subjects: subjects_vec,
        rows,
    })
}

/// Render a `ComparisonTable` as a markdown table preceded by a
/// framing sentence. Empty/degenerate tables produce a hedged
/// fallback string \u{2014} callers should not rely on the table being
/// non-empty since `gather_comparison` is the gate, not this
/// renderer.
pub fn articulate_comparison(table: &ComparisonTable) -> String {
    if table.rows.is_empty() || table.subjects.len() < MIN_COMPARISON_SUBJECTS {
        // Should not be reached when called via try_comparison_trace,
        // which gates on gather_comparison returning Some. Defensive.
        let names: Vec<String> = table.subjects.iter().map(|(_, l)| l.clone()).collect();
        let joined = if names.is_empty() {
            "these".to_string()
        } else {
            join_anchor_names(&names)
        };
        return format!(
            "I don't have enough shared attribute knowledge to compare {}.",
            joined
        );
    }
    let names: Vec<String> = table
        .subjects
        .iter()
        .map(|(_, l)| l.clone())
        .collect();
    let lead = format!("Here's a comparison of {}:", join_anchor_names(&names));

    let mut header = String::from("| Attribute |");
    for (_, label) in &table.subjects {
        header.push_str(&format!(" {} |", capitalize(label)));
    }
    let mut sep = String::from("| --- |");
    for _ in &table.subjects {
        sep.push_str(" --- |");
    }
    let mut body = String::new();
    for row in &table.rows {
        body.push_str(&format!("| {} |", bond_type_attribute_label(row.bond_type)));
        for cell in &row.cells {
            body.push_str(&format!(" {} |", cell.target_label));
        }
        body.push('\n');
    }
    format!("{}\n\n{}\n{}\n{}", lead, header, sep, body.trim_end())
}

/// Port M entry point. Returns `None` to signal the dispatcher should
/// fall through (declines on <2 anchors, missing subject nodes, or
/// thin shared-attribute coverage). Mirrors the
/// `try_procedural_trace` / `try_list_trace` shape exactly so the
/// dispatcher stays uniform.
pub fn try_comparison_trace(
    brain: &Brain,
    query: &str,
    scope: &TemporalScope,
    cfg: &ReasoningConfig,
    anchors: &[Seed],
) -> Option<ReasoningTrace> {
    if anchors.len() < MIN_COMPARISON_SUBJECTS {
        return None;
    }
    let table = gather_comparison(brain, anchors, scope, cfg)?;
    let final_answer = articulate_comparison(&table);
    let scope_note = scope_label(scope);
    let final_with_scope = match scope_note {
        Some(label) => format!("[{}] {}", label, final_answer),
        None => final_answer,
    };
    let mut min_w = f32::INFINITY;
    let mut reasoning_steps: Vec<ReasoningStep> = Vec::new();
    let mut bonds_used: Vec<(Seed, BondType, Seed)> = Vec::new();
    for row in &table.rows {
        for cell in &row.cells {
            if cell.weight < min_w {
                min_w = cell.weight;
            }
            let articulation = format!(
                "{} \u{2014} {}: {}",
                capitalize(&cell.subject_label),
                bond_type_attribute_label(row.bond_type),
                cell.target_label
            );
            reasoning_steps.push(ReasoningStep {
                from: cell.subject,
                from_content: cell.subject_label.clone(),
                bond_type: row.bond_type,
                to: cell.target,
                to_content: cell.target_label.clone(),
                confidence: cell.weight,
                articulation,
            });
            bonds_used.push((cell.subject, row.bond_type, cell.target));
        }
    }
    let confidence = if min_w.is_finite() { min_w } else { 0.0 };
    Some(ReasoningTrace {
        query: query.to_string(),
        scope: scope.clone(),
        steps: reasoning_steps,
        final_answer: final_with_scope,
        confidence,
        bonds_used,
        no_chain: false,
        // Comparison articulation is structured by construction \u{2014} every
        // cell is a typed bond. Tag as Structured so synthesis / hedging
        // treats it as confident, mirroring Port I/L precedent.
        evidence_kind: EvidenceKind::Structured,
    })
}

// =====================================================================
// Tests
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brain::{parse_confidence, Brain};
    use crate::scope::TemporalScope;

    fn small_brain() -> Brain {
        // cat -[IsA]-> mammal -[IsA]-> animal
        let mut brain = Brain::new();
        let conf = parse_confidence("verified").unwrap();
        brain.teach("cat", BondType::IsA, "mammal", 0.9, conf);
        brain.teach("mammal", BondType::IsA, "animal", 0.9, conf);
        brain
    }

    #[test]
    fn synthetic_isa_chain_articulates() {
        let mut brain = small_brain();
        let trace = reason(&mut brain, "cat", TemporalScope::Current);
        assert!(!trace.no_chain, "expected a chain for cat");
        assert!(
            trace.steps.len() >= 2,
            "expected at least 2 hops, got {}",
            trace.steps.len()
        );
        let answer_lc = trace.final_answer.to_lowercase();
        assert!(
            answer_lc.contains("mammal"),
            "answer missing 'mammal': {}",
            trace.final_answer
        );
        assert!(
            answer_lc.contains("animal"),
            "answer missing 'animal': {}",
            trace.final_answer
        );
        assert!(
            answer_lc.contains("therefore") || answer_lc.contains("which means"),
            "answer missing linking word: {}",
            trace.final_answer
        );
        // bonds_used must mirror the chain we built; no invented facts.
        assert_eq!(trace.bonds_used.len(), trace.steps.len());
        for ((f, bt, t), step) in trace.bonds_used.iter().zip(trace.steps.iter()) {
            assert_eq!(f, &step.from);
            assert_eq!(bt, &step.bond_type);
            assert_eq!(t, &step.to);
        }
    }

    #[test]
    fn temporal_scope_filters_scarred_bond() {
        // mammal->animal exists, but if we scar it, reasoning at
        // Current excludes it.
        let mut brain = small_brain();
        let mammal = crate::seed::Seed::from_label("mammal");
        let animal = crate::seed::Seed::from_label("animal");
        brain.scar_bond(mammal, animal, "test scar");
        let trace = reason(&mut brain, "cat", TemporalScope::Current);
        // The chain may still articulate a single hop (cat is a
        // mammal) — that's OK. What MUST hold: no step references
        // the scarred bond.
        for step in &trace.steps {
            assert!(
                !(step.from == mammal && step.to == animal),
                "reasoning traversed a scarred bond"
            );
        }
    }

    #[test]
    fn bond_type_articulation_phrases() {
        // Spot-check every bond type's phrasing.
        for bt in BondType::all() {
            let s = articulate_bond(*bt, "x", "y");
            assert!(
                s.starts_with("X "),
                "bad capitalization for {:?}: {}",
                bt,
                s
            );
            assert!(s.ends_with("."), "missing period for {:?}: {}", bt, s);
            assert!(
                s.contains(" y") || s.contains(" y."),
                "missing 'y' for {:?}: {}",
                bt,
                s
            );
        }
        // Specific phrasings per spec.
        assert_eq!(
            articulate_bond(BondType::IsA, "cat", "mammal"),
            "Cat is a kind of mammal."
        );
        assert_eq!(
            articulate_bond(BondType::Causes, "fire", "smoke"),
            "Fire causes smoke."
        );
        assert_eq!(
            articulate_bond(BondType::PartOf, "wheel", "car"),
            "Wheel is part of car."
        );
        assert_eq!(
            articulate_bond(BondType::HasProperty, "snow", "white"),
            "Snow has the property white."
        );
        assert_eq!(
            articulate_bond(BondType::Opposes, "hot", "cold"),
            "Hot is the opposite of cold."
        );
    }

    #[test]
    fn multi_hop_chain_uses_linkers() {
        // Build a 4-hop chain. Node labels are >=3 chars so the
        // seed-selection cascade picks them up (len >= 2 filter).
        let mut brain = Brain::new();
        let conf = parse_confidence("verified").unwrap();
        brain.teach("alpha", BondType::IsA, "beta", 0.9, conf);
        brain.teach("beta", BondType::IsA, "gamma", 0.9, conf);
        brain.teach("gamma", BondType::PartOf, "delta", 0.85, conf);
        brain.teach("delta", BondType::Produces, "epsilon", 0.8, conf);
        let trace = reason(&mut brain, "alpha", TemporalScope::Current);
        assert!(!trace.no_chain, "got no_chain trace: {}", trace.final_answer);
        assert!(trace.steps.len() >= 2);
        // The articulated prose must contain at least one linker.
        let lc = trace.final_answer.to_lowercase();
        assert!(
            lc.contains("therefore") || lc.contains("which means") || lc.contains(" and "),
            "multi-hop trace missing linker: {}",
            trace.final_answer
        );
        // Confidence is min along the chain (raw keyframe weight times
        // the bond-type base_weight multiplier; IsA = 1.5x). Sanity:
        // strictly positive.
        assert!(trace.confidence > 0.0);
    }

    // ===== Port B: bond-type composition pattern tests =====

    #[test]
    fn classify_pattern_causal_pure() {
        assert_eq!(
            classify_pattern(&[BondType::Causes]),
            ReasoningPattern::Causal
        );
        assert_eq!(
            classify_pattern(&[BondType::Causes, BondType::Produces, BondType::Requires]),
            ReasoningPattern::Causal
        );
    }

    #[test]
    fn classify_pattern_hierarchical_with_descriptor() {
        assert_eq!(
            classify_pattern(&[BondType::IsA, BondType::IsA]),
            ReasoningPattern::Hierarchical
        );
        assert_eq!(
            classify_pattern(&[BondType::IsA, BondType::PartOf]),
            ReasoningPattern::Hierarchical
        );
        assert_eq!(
            classify_pattern(&[BondType::IsA, BondType::IsA, BondType::HasProperty]),
            ReasoningPattern::Hierarchical
        );
        assert_eq!(
            classify_pattern(&[BondType::PartOf, BondType::Causes]),
            ReasoningPattern::Hierarchical
        );
        assert_ne!(
            classify_pattern(&[BondType::IsA, BondType::Causes, BondType::Produces]),
            ReasoningPattern::Hierarchical
        );
    }

    #[test]
    fn classify_pattern_property_and_transformational() {
        assert_eq!(
            classify_pattern(&[BondType::HasProperty]),
            ReasoningPattern::PropertyChain
        );
        assert_eq!(
            classify_pattern(&[BondType::HasProperty, BondType::HasProperty]),
            ReasoningPattern::PropertyChain
        );
        assert_eq!(
            classify_pattern(&[BondType::IsA, BondType::HasProperty]),
            ReasoningPattern::Hierarchical
        );
        assert_eq!(
            classify_pattern(&[BondType::DerivedFrom, BondType::UsedFor]),
            ReasoningPattern::Transformational
        );
    }

    #[test]
    fn classify_pattern_contrastive_and_mixed() {
        assert_eq!(
            classify_pattern(&[BondType::ContrastedWith, BondType::Opposes]),
            ReasoningPattern::Contrastive
        );
        assert_eq!(
            classify_pattern(&[BondType::RelatedTo, BondType::Causes]),
            ReasoningPattern::Mixed
        );
        assert_eq!(classify_pattern(&[]), ReasoningPattern::Mixed);
    }

    #[test]
    fn chain_score_orders_pattern_over_mixed() {
        let causal = chain_score(0.8, &[BondType::Causes, BondType::Causes]);
        let mixed = chain_score(0.8, &[BondType::RelatedTo, BondType::RelatedTo]);
        assert!(
            causal > mixed,
            "Causal chain ({}) must outscore Mixed-RelatedTo chain ({})",
            causal,
            mixed
        );
    }

    #[test]
    fn chain_score_demotes_drift_even_at_higher_weight() {
        let coherent_causal = chain_score(0.65, &[BondType::Causes, BondType::Causes]);
        let drift_mixed = chain_score(0.95, &[BondType::RelatedTo, BondType::Causes]);
        assert!(
            coherent_causal > drift_mixed,
            "Port B fix: coherent Causal chain (0.65w) must beat Mixed drift chain (0.95w); got {} vs {}",
            coherent_causal,
            drift_mixed
        );
    }

    #[test]
    fn reasoning_picks_causal_chain_over_hub_drift() {
        let mut brain = Brain::new();
        let conf = parse_confidence("verified").unwrap();
        brain.teach("fire", BondType::Causes, "smoke", 0.65, conf);
        brain.teach("smoke", BondType::Causes, "soot", 0.65, conf);
        brain.teach("fire", BondType::RelatedTo, "hubword", 0.95, conf);
        brain.teach("hubword", BondType::Causes, "newsitem", 0.95, conf);

        let trace = reason(&mut brain, "fire", TemporalScope::Current);
        assert!(!trace.no_chain, "expected a chain for fire");
        let final_lc = trace.final_answer.to_lowercase();
        assert!(
            final_lc.contains("smoke") && final_lc.contains("soot"),
            "Port B drift fix: expected Causal chain (smoke/soot) but got: {}",
            trace.final_answer
        );
        let picked_seq: Vec<BondType> = trace.steps.iter().map(|s| s.bond_type).collect();
        assert_eq!(
            classify_pattern(&picked_seq),
            ReasoningPattern::Causal,
            "picked chain must classify as Causal, got bond-types {:?}",
            picked_seq
        );
    }

    #[test]
    fn reasoning_picks_hierarchical_over_mixed() {
        let mut brain = Brain::new();
        let conf = parse_confidence("verified").unwrap();
        brain.teach("cat", BondType::IsA, "mammal", 0.7, conf);
        brain.teach("mammal", BondType::IsA, "animal", 0.7, conf);
        brain.teach("cat", BondType::RelatedTo, "pet", 0.92, conf);
        brain.teach("pet", BondType::Causes, "joy", 0.92, conf);

        let trace = reason(&mut brain, "cat", TemporalScope::Current);
        assert!(!trace.no_chain);
        let lc = trace.final_answer.to_lowercase();
        assert!(
            lc.contains("mammal") && lc.contains("animal"),
            "expected Hierarchical chain (mammal/animal) but got: {}",
            trace.final_answer
        );
        let picked_seq: Vec<BondType> = trace.steps.iter().map(|s| s.bond_type).collect();
        assert_eq!(
            classify_pattern(&picked_seq),
            ReasoningPattern::Hierarchical
        );
    }

    #[test]
    fn reasoning_picks_property_chain_over_drift() {
        let mut brain = Brain::new();
        let conf = parse_confidence("verified").unwrap();
        brain.teach("rose", BondType::IsA, "flower", 0.72, conf);
        brain.teach("flower", BondType::HasProperty, "fragrant", 0.72, conf);
        brain.teach("rose", BondType::RelatedTo, "valentine", 0.93, conf);
        brain.teach("valentine", BondType::RelatedTo, "card", 0.93, conf);

        let trace = reason(&mut brain, "rose", TemporalScope::Current);
        assert!(!trace.no_chain);
        let lc = trace.final_answer.to_lowercase();
        assert!(
            lc.contains("flower"),
            "expected Hierarchical chain through 'flower', got: {}",
            trace.final_answer
        );
        assert!(
            !(lc.contains("valentine") && lc.contains("card")),
            "Port B drift fix: chain still drifted into valentine->card: {}",
            trace.final_answer
        );
    }

    #[test]
    fn length_penalty_monotone() {
        let p1 = length_penalty(1);
        let p2 = length_penalty(2);
        let p3 = length_penalty(3);
        let p4 = length_penalty(4);
        let p5 = length_penalty(5);
        assert!((p1 - 1.0).abs() < 1e-6);
        assert!((p2 - 1.0).abs() < 1e-6);
        assert!(p3 < p2 && p4 < p3 && p5 < p4);
        assert!(p5 > 0.5, "length_penalty must not collapse to zero");
    }

        #[test]
    fn no_chain_for_unanchored_query() {
        let mut brain = Brain::new(); // empty
        let trace = reason(&mut brain, "nothing", TemporalScope::Current);
        assert!(trace.no_chain);
        assert_eq!(trace.steps.len(), 0);
        assert!(trace.final_answer.to_lowercase().contains("don't"));
    }

    // ===== Port C: query-intent detection + bias =====

    #[test]
    fn detect_intent_why_forms() {
        assert_eq!(detect_intent("why does fire produce smoke"), QueryIntent::Why);
        assert_eq!(detect_intent("Why is the sky blue?"), QueryIntent::Why);
        assert_eq!(detect_intent("how come ice floats"), QueryIntent::Why);
        assert_eq!(detect_intent("what causes earthquakes"), QueryIntent::Why);
    }

    #[test]
    fn detect_intent_what_forms() {
        assert_eq!(detect_intent("what is a mammal"), QueryIntent::What);
        assert_eq!(detect_intent("What's photosynthesis?"), QueryIntent::What);
        assert_eq!(detect_intent("define entropy"), QueryIntent::What);
        assert_eq!(detect_intent("who is the president"), QueryIntent::What);
    }

    #[test]
    fn detect_intent_how_forms() {
        // Mechanism ("how does X work") stays on the How path.
        assert_eq!(detect_intent("how does an engine work"), QueryIntent::How);
        // Port I (procedural) re-routes imperative "how do you / how can I" forms
        // to the procedural articulator. Capture the new contract here.
        assert_eq!(detect_intent("how do you bake bread"), QueryIntent::Procedural);
        assert_eq!(detect_intent("How can I learn rust?"), QueryIntent::Procedural);
    }

    #[test]
    fn detect_intent_compare_forms() {
        assert_eq!(detect_intent("cats vs dogs"), QueryIntent::Compare);
        assert_eq!(
            detect_intent("difference between fog and mist"),
            QueryIntent::Compare
        );
        assert_eq!(
            detect_intent("is a whale similar to a fish"),
            QueryIntent::Compare
        );
    }

    #[test]
    fn detect_intent_origin_forms() {
        assert_eq!(
            detect_intent("where does paper come from"),
            QueryIntent::Origin
        );
        assert_eq!(detect_intent("what is glass made of"), QueryIntent::Origin);
        assert_eq!(detect_intent("origin of the universe"), QueryIntent::Origin);
    }

    #[test]
    fn detect_intent_generic_fallback() {
        assert_eq!(detect_intent(""), QueryIntent::Generic);
        assert_eq!(detect_intent("   "), QueryIntent::Generic);
        assert_eq!(detect_intent("cat"), QueryIntent::Generic);
        assert_eq!(
            detect_intent("tell me about photosynthesis"),
            QueryIntent::Generic
        );
    }

    #[test]
    fn detect_intent_priority_compare_beats_why() {
        // "why are X and Y different" reads as both Compare and Why;
        // Compare is higher priority.
        assert_eq!(
            detect_intent("is a virus the same as a bacterium"),
            QueryIntent::Compare
        );
    }

    #[test]
    fn intent_bias_generic_is_neutral() {
        for p in [
            ReasoningPattern::Causal,
            ReasoningPattern::Hierarchical,
            ReasoningPattern::PropertyChain,
            ReasoningPattern::Transformational,
            ReasoningPattern::Contrastive,
            ReasoningPattern::Mixed,
        ] {
            let bias = intent_pattern_bias(QueryIntent::Generic, p);
            assert!((bias - 1.0).abs() < 1e-6, "Generic must be neutral for {:?}", p);
        }
    }

    #[test]
    fn intent_bias_targets_only_matching_pattern() {
        // Why up-weights Causal, leaves others alone.
        assert!(intent_pattern_bias(QueryIntent::Why, ReasoningPattern::Causal) > 1.0);
        assert_eq!(
            intent_pattern_bias(QueryIntent::Why, ReasoningPattern::Hierarchical),
            1.0
        );
        assert_eq!(
            intent_pattern_bias(QueryIntent::Why, ReasoningPattern::Mixed),
            1.0
        );
        // Compare up-weights Contrastive only.
        assert!(intent_pattern_bias(QueryIntent::Compare, ReasoningPattern::Contrastive) > 1.0);
        assert_eq!(
            intent_pattern_bias(QueryIntent::Compare, ReasoningPattern::Causal),
            1.0
        );
    }

    #[test]
    fn chain_score_with_intent_neutral_matches_chain_score() {
        // Generic intent must produce the same number as Port B's
        // intent-agnostic `chain_score`.
        let seq = [BondType::Causes, BondType::Causes];
        let a = chain_score(0.7, &seq);
        let b = chain_score_with_intent(0.7, &seq, QueryIntent::Generic);
        assert!((a - b).abs() < 1e-6);
    }

    #[test]
    fn chain_score_with_intent_why_boosts_causal_strictly() {
        let seq = [BondType::Causes, BondType::Produces];
        let baseline = chain_score_with_intent(0.7, &seq, QueryIntent::Generic);
        let boosted = chain_score_with_intent(0.7, &seq, QueryIntent::Why);
        assert!(
            boosted > baseline,
            "Why must boost a Causal chain (got {} vs baseline {})",
            boosted,
            baseline
        );
    }

    #[test]
    fn intent_flips_winner_why_promotes_causal_over_hierarchical() {
        // Brain has BOTH a Hierarchical chain (volcano IsA mountain IsA
        // landform) and a Causal chain (volcano Causes lava Causes ash),
        // calibrated so the Hierarchical wins generically by a slim
        // margin (effective-weight scaling already favors IsA via
        // base_weight=1.5 vs Causes base_weight=1.3, plus the Port B
        // multiplier order). Port C with intent=Why should flip the
        // winner to Causal.
        let mut brain = Brain::new();
        let conf = parse_confidence("verified").unwrap();
        // Hierarchical branch.
        brain.teach("volcano", BondType::IsA, "mountain", 0.6, conf);
        brain.teach("mountain", BondType::IsA, "landform", 0.6, conf);
        // Causal branch.
        brain.teach("volcano", BondType::Causes, "lava", 0.6, conf);
        brain.teach("lava", BondType::Causes, "ash", 0.6, conf);

        // Generic baseline: Hierarchical wins.
        // Hier:    1.5 * (0.6*1.5)  * 1.0 * 1.5 = 2.025
        // Causal:  1.6 * (0.6*1.3)  * 1.0 * 1.4 = 1.7472
        let baseline = reason(&mut brain, "volcano", TemporalScope::Current);
        assert!(!baseline.no_chain);
        let bp: Vec<BondType> = baseline.steps.iter().map(|s| s.bond_type).collect();
        assert_eq!(
            classify_pattern(&bp),
            ReasoningPattern::Hierarchical,
            "Port C baseline assumption: generic intent should pick Hierarchical, got {:?}",
            bp
        );

        // Why intent: Causal wins (1.7472 * 1.30 = 2.27 > 2.025).
        // The query mentions only `volcano` from the brain so anchor
        // selection is unambiguous — intent words `why does` and `work`
        // do not match any node.
        let asked = reason(&mut brain, "why does the volcano work", TemporalScope::Current);
        assert!(!asked.no_chain);
        let ap: Vec<BondType> = asked.steps.iter().map(|s| s.bond_type).collect();
        assert_eq!(
            classify_pattern(&ap),
            ReasoningPattern::Causal,
            "Port C: Why intent must flip the chain choice to Causal, got {:?}",
            ap
        );
    }

    #[test]
    fn intent_does_not_break_port_b_drift_demotion() {
        // Coherent Causal (0.65w) must still beat a Mixed drift chain
        // (0.95w) regardless of intent. The Port C bias is layered on
        // top of Port B — it should not undo Port B's structural
        // ordering for the Generic case, and a Why intent should make
        // the gap wider, not smaller.
        let coherent = chain_score_with_intent(
            0.65,
            &[BondType::Causes, BondType::Causes],
            QueryIntent::Generic,
        );
        let drift = chain_score_with_intent(
            0.95,
            &[BondType::RelatedTo, BondType::Causes],
            QueryIntent::Generic,
        );
        assert!(coherent > drift, "Generic: Causal(0.65) must beat Mixed(0.95)");

        let coherent_why = chain_score_with_intent(
            0.65,
            &[BondType::Causes, BondType::Causes],
            QueryIntent::Why,
        );
        let drift_why = chain_score_with_intent(
            0.95,
            &[BondType::RelatedTo, BondType::Causes],
            QueryIntent::Why,
        );
        assert!(
            (coherent_why - drift_why) > (coherent - drift),
            "Why must widen the Causal>Mixed gap, not shrink it"
        );
    }

    #[test]
    fn intent_flips_winner_compare_promotes_contrastive() {
        // Two chains rooted at `acid`. Contrastive bonds have lower
        // base_weight than IsA (Opposes 0.6, ContrastedWith 0.7 vs IsA
        // 1.5), so we crank the raw Contrastive weights and dampen the
        // Hierarchical weights to put generic-baseline scoring near a
        // tie that the Compare intent can flip.
        let mut brain = Brain::new();
        let conf = parse_confidence("verified").unwrap();
        brain.teach("acid", BondType::IsA, "chemical", 0.2, conf);
        brain.teach("chemical", BondType::IsA, "substance", 0.2, conf);
        brain.teach("acid", BondType::Opposes, "alkali", 0.9, conf);
        brain.teach("alkali", BondType::ContrastedWith, "neutral", 0.9, conf);

        // Generic baseline: Hierarchical wins.
        // Hier:        1.5 * (0.2*1.5)  * 1.0 * 1.5  = 0.675
        // Contrastive: 1.2 * min(0.9*0.6, 0.9*0.7) * 1.0 * 0.95 = 0.6156
        let baseline = reason(&mut brain, "acid", TemporalScope::Current);
        let bp: Vec<BondType> = baseline.steps.iter().map(|s| s.bond_type).collect();
        assert_eq!(
            classify_pattern(&bp),
            ReasoningPattern::Hierarchical,
            "Generic baseline should be Hierarchical, got {:?}",
            bp
        );

        // Compare intent: Contrastive wins (0.6156 * 1.40 = 0.862 > 0.675).
        // Query mentions only `acid` from the brain to keep anchor
        // selection unambiguous; "versus alkaline" is not in the graph.
        let asked = reason(&mut brain, "acid versus alkaline", TemporalScope::Current);
        let ap: Vec<BondType> = asked.steps.iter().map(|s| s.bond_type).collect();
        assert_eq!(
            classify_pattern(&ap),
            ReasoningPattern::Contrastive,
            "Compare intent must flip to Contrastive, got {:?}",
            ap
        );
    }
    // ===== Port D: epistemic calibration on Mixed chains =====

    #[test]
    fn mixed_chain_uses_hedged_prose() {
        // All-RelatedTo two-hop chain -> classifies as Mixed.
        // Articulation must hedge: contains the "associative-only"
        // marker phrase, tags evidence_kind=Associative, and does NOT
        // emit the confident "Therefore X is connected to Y" closing.
        let mut brain = Brain::new();
        let conf = parse_confidence("verified").unwrap();
        brain.teach("paris", BondType::RelatedTo, "france", 0.8, conf);
        brain.teach("france", BondType::RelatedTo, "writers", 0.8, conf);
        let trace = reason(&mut brain, "paris", TemporalScope::Current);
        assert!(!trace.no_chain, "got no_chain trace: {}", trace.final_answer);
        let picked: Vec<BondType> = trace.steps.iter().map(|s| s.bond_type).collect();
        assert_eq!(
            classify_pattern(&picked),
            ReasoningPattern::Mixed,
            "test setup expected Mixed, got {:?}",
            picked
        );
        assert_eq!(
            trace.evidence_kind,
            EvidenceKind::Associative,
            "Mixed chain must tag evidence_kind = Associative"
        );
        let lc = trace.final_answer.to_lowercase();
        assert!(
            lc.contains("associative-only"),
            "Mixed prose missing 'associative-only' marker: {}",
            trace.final_answer
        );
        assert!(
            !lc.contains("therefore paris is connected to"),
            "Mixed prose still emits the confident 'Therefore X is connected to Y' closing: {}",
            trace.final_answer
        );
        assert!(
            !lc.contains("which means"),
            "Mixed prose should not use 'Which means' entailment linker: {}",
            trace.final_answer
        );
    }

    #[test]
    fn causal_chain_keeps_confident_prose() {
        // Pure Causal chain: prose must remain byte-identical to Port
        // A/B/C output -- "Which means" linker, "Therefore X is
        // connected to Y" closing, no hedging marker.
        let mut brain = Brain::new();
        let conf = parse_confidence("verified").unwrap();
        brain.teach("fire", BondType::Causes, "smoke", 0.8, conf);
        brain.teach("smoke", BondType::Causes, "soot", 0.8, conf);
        let trace = reason(&mut brain, "fire", TemporalScope::Current);
        assert!(!trace.no_chain);
        let picked: Vec<BondType> = trace.steps.iter().map(|s| s.bond_type).collect();
        assert_eq!(classify_pattern(&picked), ReasoningPattern::Causal);
        assert_eq!(
            trace.evidence_kind,
            EvidenceKind::Structured,
            "Causal chain must tag evidence_kind = Structured"
        );
        let answer = &trace.final_answer;
        let lc = answer.to_lowercase();
        assert!(
            lc.contains("which means"),
            "Causal prose missing 'Which means' linker: {}",
            answer
        );
        assert!(
            lc.contains("therefore fire is connected to soot"),
            "Causal prose missing confident closing: {}",
            answer
        );
        assert!(
            !lc.contains("associative-only"),
            "Causal prose must not hedge: {}",
            answer
        );
    }

    #[test]
    fn hierarchical_chain_keeps_confident_prose() {
        // Pure Hierarchical (IsA, IsA): prose must remain byte-identical
        // to Port A/B/C output -- exactly the same chain articulator
        // path the original `synthetic_isa_chain_articulates` test
        // checks, but explicit about the new evidence_kind tag.
        let mut brain = Brain::new();
        let conf = parse_confidence("verified").unwrap();
        brain.teach("cat", BondType::IsA, "mammal", 0.9, conf);
        brain.teach("mammal", BondType::IsA, "animal", 0.9, conf);
        let trace = reason(&mut brain, "cat", TemporalScope::Current);
        assert!(!trace.no_chain);
        let picked: Vec<BondType> = trace.steps.iter().map(|s| s.bond_type).collect();
        assert_eq!(classify_pattern(&picked), ReasoningPattern::Hierarchical);
        assert_eq!(
            trace.evidence_kind,
            EvidenceKind::Structured,
            "Hierarchical chain must tag evidence_kind = Structured"
        );
        let answer = &trace.final_answer;
        let lc = answer.to_lowercase();
        assert!(
            lc.contains("therefore cat is connected to animal"),
            "Hierarchical prose missing confident closing: {}",
            answer
        );
        assert!(
            !lc.contains("associative-only"),
            "Hierarchical prose must not hedge: {}",
            answer
        );
    }

    // ===== Port E: multi-anchor chain unification =====

    #[test]
    fn multi_anchor_convergent_unifies_prose() {
        // Two anchors that share a non-anchor hub node:
        //   cat -[IsA]-> mammal
        //   dog -[IsA]-> mammal
        // Multi-anchor articulation must surface BOTH anchors and
        // mention the convergence on the shared hub.
        let mut brain = Brain::new();
        let conf = parse_confidence("verified").unwrap();
        brain.teach("cat", BondType::IsA, "mammal", 0.9, conf);
        brain.teach("dog", BondType::IsA, "mammal", 0.9, conf);
        let trace = reason(&mut brain, "cat dog", TemporalScope::Current);
        assert!(!trace.no_chain, "expected a multi-anchor chain");
        let lc = trace.final_answer.to_lowercase();
        assert!(
            lc.contains("cat"),
            "missing 'cat' anchor: {}",
            trace.final_answer
        );
        assert!(
            lc.contains("dog"),
            "missing 'dog' anchor: {}",
            trace.final_answer
        );
        assert!(
            lc.contains("mammal"),
            "missing shared hub 'mammal': {}",
            trace.final_answer
        );
        assert!(
            lc.contains("converge on"),
            "Port E convergent prose missing 'converge on' coda: {}",
            trace.final_answer
        );
        // Both anchors' bonds must show up in the audit list.
        let cat_seed = crate::seed::Seed::from_label("cat");
        let dog_seed = crate::seed::Seed::from_label("dog");
        let from_seeds: HashSet<Seed> =
            trace.bonds_used.iter().map(|(f, _, _)| *f).collect();
        assert!(
            from_seeds.contains(&cat_seed),
            "bonds_used missing cat anchor: {:?}",
            trace.bonds_used
        );
        assert!(
            from_seeds.contains(&dog_seed),
            "bonds_used missing dog anchor: {:?}",
            trace.bonds_used
        );
        // Multi-anchor traces are always Structured (Mixed branches
        // are filtered in `try_multi_anchor_trace`).
        assert_eq!(trace.evidence_kind, EvidenceKind::Structured);
    }

    #[test]
    fn multi_anchor_parallel_articulates_both() {
        // Two anchors with disjoint chains:
        //   fire -[Causes]-> smoke
        //   ice  -[Causes]-> water
        // No shared non-anchor node -> disjoint articulation must
        // mention both chains and state the disjoint outcome.
        let mut brain = Brain::new();
        let conf = parse_confidence("verified").unwrap();
        brain.teach("fire", BondType::Causes, "smoke", 0.9, conf);
        brain.teach("ice", BondType::Causes, "water", 0.9, conf);
        let trace = reason(&mut brain, "fire ice", TemporalScope::Current);
        assert!(!trace.no_chain, "expected a multi-anchor chain");
        let lc = trace.final_answer.to_lowercase();
        assert!(lc.contains("fire") && lc.contains("smoke"));
        assert!(lc.contains("ice") && lc.contains("water"));
        // Disjoint coda phrasing.
        assert!(
            lc.contains("don't directly converge") || lc.contains("each has its own neighborhood"),
            "Port E disjoint prose missing coda: {}",
            trace.final_answer
        );
        // The "On one hand / On another" framing should be present.
        assert!(
            lc.contains("on one hand") || lc.contains("on another"),
            "Port E disjoint prose missing parallel framing: {}",
            trace.final_answer
        );
        assert_eq!(trace.evidence_kind, EvidenceKind::Structured);
    }

    #[test]
    fn single_anchor_falls_through_unchanged() {
        // Single-anchor query: Port E must NOT fire, output must be
        // byte-identical to the Port A/B/C/D path.
        let mut brain = Brain::new();
        let conf = parse_confidence("verified").unwrap();
        brain.teach("cat", BondType::IsA, "mammal", 0.9, conf);
        brain.teach("mammal", BondType::IsA, "animal", 0.9, conf);
        let trace = reason(&mut brain, "cat", TemporalScope::Current);
        assert!(!trace.no_chain);
        let lc = trace.final_answer.to_lowercase();
        // Confident closing from articulate_structured_chain.
        assert!(
            lc.contains("therefore cat is connected to animal"),
            "single-anchor produced multi-anchor prose: {}",
            trace.final_answer
        );
        // No multi-anchor coda phrasing should leak in.
        assert!(
            !lc.contains("converge on") && !lc.contains("on one hand"),
            "single-anchor query leaked Port E coda: {}",
            trace.final_answer
        );
    }

    #[test]
    fn all_chains_mixed_falls_back_to_single() {
        // Both anchors produce all-RelatedTo (Mixed) chains. Port E
        // filters Mixed branches; with < 2 typed chains it returns
        // None and the single-anchor path runs, so Port D's hedged
        // "associative-only" prose fires for whichever chain
        // `best_chain` picks.
        let mut brain = Brain::new();
        let conf = parse_confidence("verified").unwrap();
        brain.teach("paris", BondType::RelatedTo, "france", 0.8, conf);
        brain.teach("france", BondType::RelatedTo, "writers", 0.8, conf);
        brain.teach("london", BondType::RelatedTo, "england", 0.8, conf);
        brain.teach("england", BondType::RelatedTo, "tea", 0.8, conf);
        let trace = reason(&mut brain, "paris london", TemporalScope::Current);
        assert!(!trace.no_chain);
        let lc = trace.final_answer.to_lowercase();
        assert!(
            lc.contains("associative-only"),
            "all-Mixed multi-anchor query did not fall back to Port D hedged prose: {}",
            trace.final_answer
        );
        assert_eq!(
            trace.evidence_kind,
            EvidenceKind::Associative,
            "all-Mixed fallback must tag evidence_kind = Associative"
        );
        // Multi-anchor coda must NOT appear -- Port E declined and
        // the single-chain path took over.
        assert!(
            !lc.contains("converge on") && !lc.contains("on one hand"),
            "all-Mixed multi-anchor leaked Port E coda: {}",
            trace.final_answer
        );
    }

    #[test]
    fn mixed_branch_filtered_typed_branch_kept_single_anchor_falls_through() {
        // One anchor produces a typed (Hierarchical) chain, the
        // other only a Mixed (RelatedTo) chain. Port E requires
        // >= 2 typed chains, so it returns None. Single-anchor path
        // then picks the strongest chain across both anchors. We
        // don't pin which one wins (depends on chain_score), but we
        // assert that Port E coda phrasing is absent.
        let mut brain = Brain::new();
        let conf = parse_confidence("verified").unwrap();
        // Typed branch.
        brain.teach("oak", BondType::IsA, "tree", 0.9, conf);
        brain.teach("tree", BondType::IsA, "plant", 0.9, conf);
        // Mixed branch.
        brain.teach("paris", BondType::RelatedTo, "france", 0.8, conf);
        brain.teach("france", BondType::RelatedTo, "wine", 0.8, conf);
        let trace = reason(&mut brain, "oak paris", TemporalScope::Current);
        assert!(!trace.no_chain);
        let lc = trace.final_answer.to_lowercase();
        assert!(
            !lc.contains("converge on") && !lc.contains("on one hand"),
            "mixed-branch fallback leaked Port E coda: {}",
            trace.final_answer
        );
    }

    #[test]
    fn detect_convergence_node_finds_shared() {
        // Synthetic chains sharing a hub.
        let cat = crate::seed::Seed::from_label("cat");
        let dog = crate::seed::Seed::from_label("dog");
        let mammal = crate::seed::Seed::from_label("mammal");
        let chain1 = vec![ChainLeg {
            from: cat,
            from_content: "cat".to_string(),
            bond_type: BondType::IsA,
            to: mammal,
            to_content: "mammal".to_string(),
            weight: 0.9,
            from_kind: None,
            to_kind: None,
            negated: false,
        }];
        let chain2 = vec![ChainLeg {
            from: dog,
            from_content: "dog".to_string(),
            bond_type: BondType::IsA,
            to: mammal,
            to_content: "mammal".to_string(),
            weight: 0.9,
            from_kind: None,
            to_kind: None,
            negated: false,
        }];
        let result = detect_convergence_node(&[(cat, chain1), (dog, chain2)]);
        assert_eq!(result.as_deref(), Some("mammal"));
    }

    #[test]
    fn detect_convergence_node_excludes_anchors() {
        // Even if anchor1's endpoint IS anchor2, that is NOT
        // convergence -- it's a direct edge between two anchors.
        // Port E's convergence detector excludes anchors from the
        // "shared node" set.
        let cat = crate::seed::Seed::from_label("cat");
        let dog = crate::seed::Seed::from_label("dog");
        let chain1 = vec![ChainLeg {
            from: cat,
            from_content: "cat".to_string(),
            bond_type: BondType::RelatedTo,
            to: dog,
            to_content: "dog".to_string(),
            weight: 0.5,
            from_kind: None,
            to_kind: None,
            negated: false,
        }];
        let chain2 = vec![ChainLeg {
            from: dog,
            from_content: "dog".to_string(),
            bond_type: BondType::IsA,
            to: crate::seed::Seed::from_label("mammal"),
            to_content: "mammal".to_string(),
            weight: 0.9,
            from_kind: None,
            to_kind: None,
            negated: false,
        }];
        let result = detect_convergence_node(&[(cat, chain1), (dog, chain2)]);
        assert!(
            result.is_none(),
            "convergence detector wrongly counted an anchor-to-anchor edge as a hub: {:?}",
            result
        );
    }

    #[test]
    fn join_anchor_names_english_list() {
        assert_eq!(join_anchor_names(&[]), "");
        assert_eq!(join_anchor_names(&["cat".to_string()]), "cat");
        assert_eq!(
            join_anchor_names(&["cat".to_string(), "dog".to_string()]),
            "cat and dog"
        );
        assert_eq!(
            join_anchor_names(&[
                "cat".to_string(),
                "dog".to_string(),
                "mouse".to_string()
            ]),
            "cat, dog, and mouse"
        );
    }

    // ===== Q.1 typed-KU narration tests ============================
    #[test]
    fn q1_articulate_bond_typed_appends_tag_for_typed_kinds() {
        let s = articulate_bond_typed(
            BondType::IsA,
            "democracy",
            Some("definition"),
            "rule by the people",
            Some("definition"),
        );
        assert!(s.contains("(definition)"), "got: {}", s);
        assert!(s.contains("is a kind of"), "got: {}", s);
    }

    #[test]
    fn q1_articulate_bond_typed_silent_for_concept_and_raw() {
        let s_concept = articulate_bond_typed(
            BondType::IsA,
            "cat",
            Some("concept"),
            "mammal",
            Some("concept"),
        );
        // No (concept) tag — common case stays clean.
        assert_eq!(s_concept, articulate_bond(BondType::IsA, "cat", "mammal"));
        let s_raw = articulate_bond_typed(
            BondType::IsA,
            "cat",
            Some("raw"),
            "mammal",
            Some("raw"),
        );
        assert_eq!(s_raw, articulate_bond(BondType::IsA, "cat", "mammal"));
    }

    #[test]
    fn q1_articulate_bond_typed_handles_mixed_kinds() {
        let s = articulate_bond_typed(
            BondType::Causes,
            "smoking",
            Some("causal"),
            "lung damage",
            Some("concept"),
        );
        // From is tagged, to is concept — only `(causal)` tag appears.
        assert!(s.contains("(causal)"));
        assert!(!s.contains("-> concept"));
    }

    #[test]
    fn q1_articulate_bond_typed_directional_mismatch() {
        let s = articulate_bond_typed(
            BondType::PartOf,
            "oxygen",
            Some("composition"),
            "water",
            Some("definition"),
        );
        assert!(s.contains("(composition -> definition)"), "got: {}", s);
    }

// =================================================================
    // Port I \u{2014} procedural articulation tests
    // =================================================================

    /// Brain modeling a small recipe: bread requires flour/water/yeast,
    /// kneading requires dough; bread produces nourishment. No
    /// Precedes bonds (intentionally) \u{2014} tests cover the v1 fallback
    /// ordering by typed bond priority.
    fn bread_brain() -> Brain {
        let mut brain = Brain::new();
        let conf = parse_confidence("verified").unwrap();
        brain.teach("bread", BondType::Requires, "flour", 0.9, conf);
        brain.teach("bread", BondType::Requires, "water", 0.85, conf);
        brain.teach("bread", BondType::Requires, "yeast", 0.8, conf);
        brain.teach("bread", BondType::PartOf, "dough", 0.75, conf);
        brain.teach("bread", BondType::Produces, "nourishment", 0.7, conf);
        brain
    }

    #[test]
    fn port_i_detects_procedural_intent_imperative() {
        assert_eq!(detect_intent("how do I make bread"), QueryIntent::Procedural);
        assert_eq!(detect_intent("how do you bake bread"), QueryIntent::Procedural);
        assert_eq!(detect_intent("how to install rust"), QueryIntent::Procedural);
        assert_eq!(detect_intent("steps to deploy a binary"), QueryIntent::Procedural);
        assert_eq!(
            detect_intent("what are the steps to brew coffee"),
            QueryIntent::Procedural
        );
        assert_eq!(
            detect_intent("what's the process for diagnosing a network issue"),
            QueryIntent::Procedural
        );
        assert_eq!(
            detect_intent("walk me through compiling rust"),
            QueryIntent::Procedural
        );
        assert_eq!(
            detect_intent("how can I tie a shoelace"),
            QueryIntent::Procedural
        );
    }

    #[test]
    fn port_i_does_not_steal_mechanism_how() {
        // "how does X work" stays on the existing How (mechanism) path.
        assert_eq!(detect_intent("how does an engine work"), QueryIntent::How);
        assert_eq!(detect_intent("how does photosynthesis work"), QueryIntent::How);
        assert_eq!(detect_intent("how is rain formed"), QueryIntent::How);
        assert_eq!(detect_intent("how are stars made"), QueryIntent::How);
        assert_eq!(detect_intent("how would a perpetual motion machine work"), QueryIntent::How);
    }

    #[test]
    fn port_i_declarative_intents_unchanged() {
        // Regression guards \u{2014} declarative intents must NOT classify as
        // Procedural.
        assert_eq!(detect_intent("what is bread"), QueryIntent::What);
        assert_eq!(detect_intent("why does fire produce smoke"), QueryIntent::Why);
        assert_eq!(detect_intent("cats vs dogs"), QueryIntent::Compare);
        assert_eq!(
            detect_intent("where does coffee come from"),
            QueryIntent::Origin
        );
        assert_eq!(detect_intent(""), QueryIntent::Generic);
    }

    #[test]
    fn port_i_assemble_procedure_orders_by_bond_type() {
        let brain = bread_brain();
        let cfg = ReasoningConfig::default();
        let bread = crate::seed::Seed::from_label("bread");
        let steps = assemble_procedure(&brain, &bread, &TemporalScope::Current, &cfg);
        assert!(
            steps.len() >= 4,
            "expected >= 4 procedure steps from bread anchor, got {}",
            steps.len()
        );
        // Bond ordering: Requires (3 of) before PartOf before Produces.
        let mut last_priority: u8 = 0;
        for s in &steps {
            let p = procedural_order_priority(s.bond_type);
            assert!(
                p >= last_priority,
                "step {} broke ordering: {:?} (priority {}) after priority {}",
                s.ordinal, s.bond_type, p, last_priority
            );
            last_priority = p;
        }
        // Confirm Requires came before Produces (the spec's prereq-then-result shape).
        let first_bt = steps.first().unwrap().bond_type;
        let last_bt = steps.last().unwrap().bond_type;
        assert_eq!(first_bt, BondType::Requires, "first step should be Requires");
        assert_eq!(last_bt, BondType::Produces, "last step should be Produces");
    }

    #[test]
    fn port_i_articulates_numbered_steps() {
        let mut brain = bread_brain();
        let trace = reason(&mut brain, "how do I make bread", TemporalScope::Current);
        assert!(
            !trace.no_chain,
            "expected procedural trace, got no_chain: {}",
            trace.final_answer
        );
        let ans_lc = trace.final_answer.to_lowercase();
        assert!(
            ans_lc.contains("step 1.") && ans_lc.contains("step 2."),
            "missing numbered steps: {}",
            trace.final_answer
        );
        assert!(
            ans_lc.contains("flour") || ans_lc.contains("water") || ans_lc.contains("yeast"),
            "missing requires-target in articulation: {}",
            trace.final_answer
        );
        assert!(
            ans_lc.contains("nourishment"),
            "missing produces-target: {}",
            trace.final_answer
        );
        // Procedural articulation must NOT use the chain-composer
        // linker words \u{2014} those signal a single-chain walk, not steps.
        assert!(
            !ans_lc.contains("therefore"),
            "procedural prose leaked chain-composer 'therefore': {}",
            trace.final_answer
        );
    }

    #[test]
    fn port_i_falls_back_when_too_few_steps() {
        // Brain with only one outgoing procedural bond \u{2014} fewer than 2
        // steps means Port I declines and the standard chain composer
        // articulates as before.
        let mut brain = Brain::new();
        let conf = parse_confidence("verified").unwrap();
        brain.teach("start", BondType::Produces, "binary", 0.9, conf);
        let trace = reason(&mut brain, "how do I start", TemporalScope::Current);
        // Expect either a non-procedural articulation or a no_chain
        // fallback \u{2014} either way, no "Step 1." prefix.
        assert!(
            !trace.final_answer.to_lowercase().contains("step 1."),
            "single-bond brain should NOT trigger Port I: {}",
            trace.final_answer
        );
    }

    #[test]
    fn port_i_declarative_query_unaffected() {
        // Regression guard \u{2014} declarative queries against a procedural-
        // structured brain must still take the chain-composer path.
        let mut brain = bread_brain();
        let trace = reason(&mut brain, "what is bread", TemporalScope::Current);
        assert!(
            !trace.final_answer.to_lowercase().contains("step 1."),
            "declarative query leaked into procedural articulation: {}",
            trace.final_answer
        );
    }

    #[test]
    fn port_i_caps_step_count() {
        // Brain with 12 outgoing Requires bonds. Port I caps at
        // MAX_PROCEDURE_STEPS (8) so output stays readable.
        let mut brain = Brain::new();
        let conf = parse_confidence("verified").unwrap();
        for i in 0..12 {
            let target = format!("ingredient{}", i);
            brain.teach("recipe", BondType::Requires, &target, 0.8, conf);
        }
        let cfg = ReasoningConfig::default();
        let recipe = crate::seed::Seed::from_label("recipe");
        let steps = assemble_procedure(&brain, &recipe, &TemporalScope::Current, &cfg);
        assert!(
            steps.len() <= MAX_PROCEDURE_STEPS,
            "step count {} exceeded cap {}",
            steps.len(),
            MAX_PROCEDURE_STEPS
        );
    }

    #[test]
    fn port_i_skips_non_procedural_bonds() {
        // A brain whose anchor only has IsA / HasProperty (declarative)
        // bonds out has zero procedural candidates; assemble_procedure
        // returns empty.
        let mut brain = Brain::new();
        let conf = parse_confidence("verified").unwrap();
        brain.teach("cat", BondType::IsA, "mammal", 0.9, conf);
        brain.teach("cat", BondType::HasProperty, "furry", 0.8, conf);
        let cfg = ReasoningConfig::default();
        let cat = crate::seed::Seed::from_label("cat");
        let steps = assemble_procedure(&brain, &cat, &TemporalScope::Current, &cfg);
        assert!(
            steps.is_empty(),
            "declarative-only anchor should have zero procedural steps, got {}",
            steps.len()
        );
    }

    #[test]
    fn port_i_trace_is_structured() {
        // Procedural traces tag as Structured so synthesis treats them
        // confidently rather than hedging like Mixed chains.
        let mut brain = bread_brain();
        let trace = reason(&mut brain, "how do you make bread", TemporalScope::Current);
        assert!(!trace.no_chain);
        assert_eq!(trace.evidence_kind, EvidenceKind::Structured);
        assert!(
            !trace.steps.is_empty() && !trace.bonds_used.is_empty(),
            "steps + bonds_used should populate from procedural assembly"
        );
        assert_eq!(trace.steps.len(), trace.bonds_used.len());
        for ((f, bt, t), s) in trace.bonds_used.iter().zip(trace.steps.iter()) {
            assert_eq!(f, &s.from);
            assert_eq!(bt, &s.bond_type);
            assert_eq!(t, &s.to);
        }
    }
    #[test]
    fn q1_articulate_bond_legacy_signature_unchanged() {
        // Port A/B/C/D tests rely on byte-identical articulate_bond output.
        let s = articulate_bond(BondType::IsA, "cat", "mammal");
        assert_eq!(s, "Cat is a kind of mammal.");
    }

// =================================================================
    // Port L — list articulation tests
    // =================================================================

    /// Brain modeling cloud taxonomy: cumulus / stratus / cirrus IsA
    /// cloud, each with a differentiating HasProperty. Used by the
    /// canonical Port L list-articulation tests.
    fn cloud_brain() -> Brain {
        let mut brain = Brain::new();
        let conf = parse_confidence("verified").unwrap();
        brain.teach("cumulus", BondType::IsA, "clouds", 0.9, conf);
        brain.teach("stratus", BondType::IsA, "clouds", 0.85, conf);
        brain.teach("cirrus", BondType::IsA, "clouds", 0.8, conf);
        brain.teach("cumulus", BondType::HasProperty, "fluffy", 0.9, conf);
        brain.teach("stratus", BondType::HasProperty, "layered", 0.9, conf);
        brain.teach("cirrus", BondType::HasProperty, "wispy", 0.9, conf);
        brain
    }

    /// A 20-sibling element brain so we can exercise K-cap + tail.
    fn elements_brain() -> Brain {
        let mut brain = Brain::new();
        let conf = parse_confidence("verified").unwrap();
        let elements = [
            "hydrogen", "helium", "lithium", "beryllium", "boron",
            "carbon", "nitrogen", "oxygen", "fluorine", "neon",
            "sodium", "magnesium", "aluminum", "silicon", "phosphorus",
            "sulfur", "chlorine", "argon", "potassium", "calcium",
        ];
        for (i, e) in elements.iter().enumerate() {
            // Decreasing weight so order is deterministic.
            let w = 0.95 - (i as f32) * 0.01;
            brain.teach(e, BondType::IsA, "elements", w, conf);
        }
        brain
    }

    #[test]
    fn port_l_detects_enumerative_intent_canonical() {
        assert_eq!(
            detect_intent("what are the types of clouds"),
            QueryIntent::Enumerative
        );
        assert_eq!(
            detect_intent("what kinds of mammals are there"),
            QueryIntent::Enumerative
        );
        assert_eq!(
            detect_intent("list the noble gases"),
            QueryIntent::Enumerative
        );
        assert_eq!(
            detect_intent("give me three examples of mammals"),
            QueryIntent::Enumerative
        );
        assert_eq!(
            detect_intent("name three planets"),
            QueryIntent::Enumerative
        );
        assert_eq!(
            detect_intent("what programming languages are statically typed"),
            QueryIntent::Enumerative
        );
        assert_eq!(
            detect_intent("examples of mammals"),
            QueryIntent::Enumerative
        );
    }

    #[test]
    fn port_l_does_not_steal_what_definition() {
        // "what is X" / "what's X" stay on the What (definition) path.
        assert_eq!(detect_intent("what is a cloud"), QueryIntent::What);
        assert_eq!(detect_intent("what is dna"), QueryIntent::What);
        assert_eq!(detect_intent("what's photosynthesis"), QueryIntent::What);
        assert_eq!(detect_intent("define entropy"), QueryIntent::What);
        assert_eq!(detect_intent("who is the president"), QueryIntent::What);
    }

    #[test]
    fn port_l_does_not_steal_procedural_or_mechanism() {
        // Procedural and How regression: enumerative cues must not
        // leak into procedural/how queries.
        assert_eq!(
            detect_intent("how do I make bread"),
            QueryIntent::Procedural
        );
        assert_eq!(
            detect_intent("how does an engine work"),
            QueryIntent::How
        );
        // "how to list X" looks list-y but the imperative "how to" wins.
        assert_eq!(
            detect_intent("how to install rust"),
            QueryIntent::Procedural
        );
    }

    #[test]
    fn port_l_does_not_steal_compare() {
        // Compare beats Enumerative — "compare cats and dogs".
        assert_eq!(
            detect_intent("compare cats and dogs"),
            QueryIntent::Compare
        );
        assert_eq!(
            detect_intent("difference between mammals and reptiles"),
            QueryIntent::Compare
        );
    }

    #[test]
    fn port_l_k_determination_from_cues() {
        assert_eq!(list_k_from_query("name three planets"), 3);
        assert_eq!(list_k_from_query("give me three examples of mammals"), 3);
        assert_eq!(list_k_from_query("list five noble gases"), 5);
        assert_eq!(list_k_from_query("give me some examples of birds"), 5);
        assert_eq!(list_k_from_query("name a few colors"), 5);
        // Default cap when no quantity cue.
        assert_eq!(list_k_from_query("list the planets"), DEFAULT_LIST_K);
        assert_eq!(
            list_k_from_query("what are the types of clouds"),
            DEFAULT_LIST_K
        );
        // Digit cue.
        assert_eq!(list_k_from_query("name 4 oceans"), 4);
    }

    #[test]
    fn port_l_gather_siblings_basic() {
        let brain = cloud_brain();
        let cfg = ReasoningConfig::default();
        let cloud = crate::seed::Seed::from_label("clouds");
        let (siblings, total_known) =
            gather_siblings(&brain, &cloud, &TemporalScope::Current, &cfg, 7);
        assert_eq!(total_known, 3);
        assert_eq!(siblings.len(), 3);
        let labels: Vec<&str> = siblings.iter().map(|s| s.sibling_label.as_str()).collect();
        assert!(labels.contains(&"cumulus"));
        assert!(labels.contains(&"stratus"));
        assert!(labels.contains(&"cirrus"));
        // Each sibling has a differentiating property — none is shared.
        for s in &siblings {
            assert!(
                s.differentiating_phrase.is_some(),
                "sibling {} should have differentiating phrase",
                s.sibling_label
            );
        }
    }

    #[test]
    fn port_l_articulates_numbered_list() {
        let mut brain = cloud_brain();
        let trace = reason(&mut brain, "what are the types of clouds", TemporalScope::Current);
        assert!(!trace.no_chain, "list trace should not be no_chain: {:?}", trace);
        assert_eq!(trace.evidence_kind, EvidenceKind::Structured);
        // Lead phrase + 3 numbered items.
        let body = trace.final_answer.to_lowercase();
        assert!(
            body.contains("kinds of cloud"),
            "missing list lead phrase: {}",
            trace.final_answer
        );
        assert!(body.contains("1."), "missing item 1: {}", trace.final_answer);
        assert!(body.contains("2."), "missing item 2: {}", trace.final_answer);
        assert!(body.contains("3."), "missing item 3: {}", trace.final_answer);
        // Three IsA bonds in audit list.
        assert_eq!(trace.bonds_used.len(), 3);
        for (_, bt, _) in &trace.bonds_used {
            assert_eq!(*bt, BondType::IsA);
        }
    }

    #[test]
    fn port_l_caps_at_k_with_tail() {
        let mut brain = elements_brain();
        let trace = reason(&mut brain, "list the elements", TemporalScope::Current);
        assert!(!trace.no_chain);
        // Default K = 7.
        assert_eq!(trace.bonds_used.len(), DEFAULT_LIST_K);
        // Tail line for the remainder.
        assert!(
            trace.final_answer.contains("and 13 more"),
            "missing tail: {}",
            trace.final_answer
        );
    }

    #[test]
    fn port_l_explicit_k_pins_count() {
        let mut brain = elements_brain();
        let trace = reason(
            &mut brain,
            "give me three examples of elements",
            TemporalScope::Current,
        );
        assert!(!trace.no_chain);
        assert_eq!(trace.bonds_used.len(), 3);
        assert!(trace.final_answer.contains("and 17 more"));
    }

    #[test]
    fn port_l_falls_back_when_too_few_siblings() {
        // Brain with one IsA bond — fewer than MIN_LIST_ITEMS, so Port
        // L declines and the standard composer runs.
        let mut brain = Brain::new();
        let conf = parse_confidence("verified").unwrap();
        brain.teach("cumulus", BondType::IsA, "cloud", 0.9, conf);
        let trace = reason(&mut brain, "what are the types of clouds", TemporalScope::Current);
        // Either no_chain or a non-list articulation — but NOT a
        // "Here are N kinds of" lead phrase.
        assert!(
            !trace.final_answer.contains("kinds of"),
            "single-sibling brain should NOT trigger Port L: {}",
            trace.final_answer
        );
    }

    #[test]
    fn port_l_declarative_query_unaffected() {
        // Regression guard — "what is a cloud" against the same brain
        // must NOT take the Port L path.
        let mut brain = cloud_brain();
        let trace = reason(&mut brain, "what is a cloud", TemporalScope::Current);
        assert!(
            !trace.final_answer.contains("kinds of"),
            "declarative query leaked into list articulation: {}",
            trace.final_answer
        );
        assert!(
            !trace.final_answer.contains("...and "),
            "declarative query leaked into list articulation: {}",
            trace.final_answer
        );
    }

    #[test]
    fn port_l_procedural_query_unaffected() {
        // Regression guard — Port I (procedural) still fires on
        // "how do you make bread" even though "bread" might have IsA
        // children somewhere.
        let mut brain = bread_brain();
        let trace = reason(&mut brain, "how do you make bread", TemporalScope::Current);
        assert!(
            trace.final_answer.contains("Step 1."),
            "procedural query leaked into list articulation: {}",
            trace.final_answer
        );
        assert!(
            !trace.final_answer.contains("kinds of"),
            "procedural query produced list lead: {}",
            trace.final_answer
        );
    }

    #[test]
    fn port_l_partof_path_articulates_as_parts() {
        // Brain with PartOf children — Port L should articulate as
        // "parts of" rather than "kinds of".
        let mut brain = Brain::new();
        let conf = parse_confidence("verified").unwrap();
        brain.teach("engine", BondType::PartOf, "car", 0.9, conf);
        brain.teach("wheel", BondType::PartOf, "car", 0.9, conf);
        brain.teach("door", BondType::PartOf, "car", 0.85, conf);
        let trace = reason(&mut brain, "what are the parts of a car", TemporalScope::Current);
        assert!(!trace.no_chain);
        assert!(
            trace.final_answer.contains("parts of"),
            "PartOf-dominant list should use 'parts of': {}",
            trace.final_answer
        );
        for (_, bt, _) in &trace.bonds_used {
            assert_eq!(*bt, BondType::PartOf);
        }
    }

    #[test]
    fn port_l_trace_records_real_bonds() {
        // Each list item must reference a real bond — the bond direction
        // is sibling -> anchor (incoming IsA / PartOf).
        let mut brain = cloud_brain();
        let trace = reason(&mut brain, "list the kinds of clouds", TemporalScope::Current);
        assert!(!trace.no_chain);
        let cloud = crate::seed::Seed::from_label("clouds");
        for (from, bt, to) in &trace.bonds_used {
            assert_eq!(*to, cloud, "all bonds must terminate at the anchor");
            assert!(matches!(bt, BondType::IsA | BondType::PartOf));
            assert!(
                brain.network.get_bond(from, to).is_some(),
                "audit bond not present in brain"
            );
        }
    }
// =================================================================
    // Port M tests
    // =================================================================

    /// Two subjects sharing exactly 3 outgoing bond types: IsA,
    /// HasProperty, Causes. The comparison articulator should pick all
    /// three as table rows.
    fn comparison_brain_two_subjects() -> Brain {
        let mut brain = Brain::new();
        let conf = parse_confidence("verified").unwrap();
        // cumulus: IsA cloud, HasProperty fluffy, Causes shower
        brain.teach("cumulus", BondType::IsA, "cloud", 0.9, conf);
        brain.teach("cumulus", BondType::HasProperty, "fluffy", 0.85, conf);
        brain.teach("cumulus", BondType::Causes, "shower", 0.75, conf);
        // stratus: IsA cloud, HasProperty layered, Causes drizzle
        brain.teach("stratus", BondType::IsA, "cloud", 0.9, conf);
        brain.teach("stratus", BondType::HasProperty, "layered", 0.85, conf);
        brain.teach("stratus", BondType::Causes, "drizzle", 0.7, conf);
        brain
    }

    /// Three subjects sharing exactly 2 outgoing bond types: IsA, HasProperty.
    /// Each subject also has one unique bond type (Causes for cumulus,
    /// PartOf for stratus, UsedFor for cirrus) which must NOT show up
    /// in the table.
    fn comparison_brain_three_subjects_partial() -> Brain {
        let mut brain = Brain::new();
        let conf = parse_confidence("verified").unwrap();
        // Shared: IsA cloud, HasProperty <something>
        brain.teach("cumulus", BondType::IsA, "cloud", 0.9, conf);
        brain.teach("cumulus", BondType::HasProperty, "fluffy", 0.85, conf);
        brain.teach("cumulus", BondType::Causes, "shower", 0.75, conf);
        brain.teach("stratus", BondType::IsA, "cloud", 0.9, conf);
        brain.teach("stratus", BondType::HasProperty, "layered", 0.85, conf);
        brain.teach("stratus", BondType::PartOf, "sky", 0.6, conf);
        brain.teach("cirrus", BondType::IsA, "cloud", 0.9, conf);
        brain.teach("cirrus", BondType::HasProperty, "wispy", 0.85, conf);
        brain.teach("cirrus", BondType::UsedFor, "weather indication", 0.5, conf);
        brain
    }

    /// Two subjects sharing only ONE bond type. Must trigger the
    /// MIN_SHARED_BOND_TYPES fallback (gather_comparison returns None).
    fn comparison_brain_thin_coverage() -> Brain {
        let mut brain = Brain::new();
        let conf = parse_confidence("verified").unwrap();
        brain.teach("alpha", BondType::IsA, "thing", 0.9, conf);
        brain.teach("alpha", BondType::HasProperty, "red", 0.85, conf);
        brain.teach("beta", BondType::IsA, "thing", 0.9, conf);
        brain.teach("beta", BondType::Causes, "effect", 0.85, conf);
        brain
    }

    #[test]
    fn detect_intent_routes_compare_phrasings() {
        // Sanity: every Port M trigger phrasing routes to QueryIntent::Compare.
        assert_eq!(detect_intent("compare cumulus and stratus"), QueryIntent::Compare);
        assert_eq!(detect_intent("python vs rust"), QueryIntent::Compare);
        assert_eq!(detect_intent("python vs. rust"), QueryIntent::Compare);
        assert_eq!(detect_intent("what's the difference between dna and rna"), QueryIntent::Compare);
        assert_eq!(detect_intent("differences between cats and dogs"), QueryIntent::Compare);
    }

    #[test]
    fn comparative_query_routes_to_table_articulator() {
        // The originally-mandated TDD test: a Compare-intent query
        // against a fixture brain with shared bond types must produce
        // a markdown comparison table from `try_comparison_trace`.
        let mut brain = comparison_brain_two_subjects();
        let trace = reason(
            &mut brain,
            "compare cumulus and stratus",
            TemporalScope::Current,
        );
        assert!(!trace.no_chain, "expected a comparison trace, got no_chain");
        let answer = trace.final_answer.to_lowercase();
        assert!(
            answer.contains("comparison") || answer.contains("compare"),
            "expected framing sentence: {}",
            trace.final_answer
        );
        assert!(
            answer.contains("cumulus") && answer.contains("stratus"),
            "expected both subjects in answer: {}",
            trace.final_answer
        );
        // Markdown header row signature.
        assert!(
            trace.final_answer.contains("| Attribute |"),
            "expected markdown table header: {}",
            trace.final_answer
        );
    }

    #[test]
    fn comparison_articulator_aligns_shared_bond_types() {
        // Two subjects share IsA, HasProperty, Causes. Table must
        // contain all three rows.
        let brain = comparison_brain_two_subjects();
        let cumulus = crate::seed::Seed::from_label("cumulus");
        let stratus = crate::seed::Seed::from_label("stratus");
        let cfg = ReasoningConfig::default();
        let table = gather_comparison(
            &brain,
            &[cumulus, stratus],
            &TemporalScope::Current,
            &cfg,
        )
        .expect("gather_comparison should succeed for shared brain");
        assert_eq!(table.subjects.len(), 2);
        assert_eq!(
            table.rows.len(),
            3,
            "expected 3 shared bond_types (IsA, HasProperty, Causes); rows: {:?}",
            table
                .rows
                .iter()
                .map(|r| r.bond_type)
                .collect::<Vec<_>>()
        );
        let bond_types: std::collections::HashSet<BondType> =
            table.rows.iter().map(|r| r.bond_type).collect();
        assert!(bond_types.contains(&BondType::IsA));
        assert!(bond_types.contains(&BondType::HasProperty));
        assert!(bond_types.contains(&BondType::Causes));
        // Each row has exactly 2 cells (one per subject).
        for row in &table.rows {
            assert_eq!(row.cells.len(), 2);
        }
    }

    #[test]
    fn comparison_articulator_three_subjects_intersection_only() {
        // Three subjects: only IsA and HasProperty are universally shared.
        // Causes (cumulus only), PartOf (stratus only), UsedFor (cirrus only)
        // must NOT appear in the rendered table.
        let brain = comparison_brain_three_subjects_partial();
        let cumulus = crate::seed::Seed::from_label("cumulus");
        let stratus = crate::seed::Seed::from_label("stratus");
        let cirrus = crate::seed::Seed::from_label("cirrus");
        let cfg = ReasoningConfig::default();
        let table = gather_comparison(
            &brain,
            &[cumulus, stratus, cirrus],
            &TemporalScope::Current,
            &cfg,
        )
        .expect("gather_comparison should succeed");
        assert_eq!(table.subjects.len(), 3);
        assert_eq!(table.rows.len(), 2, "only IsA + HasProperty are shared by all 3");
        let types: std::collections::HashSet<BondType> =
            table.rows.iter().map(|r| r.bond_type).collect();
        assert!(types.contains(&BondType::IsA));
        assert!(types.contains(&BondType::HasProperty));
        assert!(!types.contains(&BondType::Causes));
        assert!(!types.contains(&BondType::PartOf));
        assert!(!types.contains(&BondType::UsedFor));
        for row in &table.rows {
            assert_eq!(row.cells.len(), 3);
        }
    }

    #[test]
    fn comparison_articulator_falls_back_on_thin_coverage() {
        // Two subjects sharing only ONE bond type \u{2014} below MIN_SHARED_BOND_TYPES.
        // gather_comparison must return None so the dispatcher falls through.
        let brain = comparison_brain_thin_coverage();
        let alpha = crate::seed::Seed::from_label("alpha");
        let beta = crate::seed::Seed::from_label("beta");
        let cfg = ReasoningConfig::default();
        let result = gather_comparison(
            &brain,
            &[alpha, beta],
            &TemporalScope::Current,
            &cfg,
        );
        assert!(
            result.is_none(),
            "expected fallback (1 shared bond_type < MIN_SHARED_BOND_TYPES); got: {:?}",
            result
        );
    }

    #[test]
    fn comparison_articulator_rejects_single_subject() {
        // 1 anchor must NOT route to comparison \u{2014} the dispatcher must
        // fall through to single-anchor declarative articulation.
        let mut brain = comparison_brain_two_subjects();
        // Use a Compare-intent phrasing that resolves a single anchor.
        let trace = reason(
            &mut brain,
            "is cumulus the same as cumulus",
            TemporalScope::Current,
        );
        // We don't assert what the answer IS \u{2014} just that we did NOT
        // emit the markdown table header (which only Port M emits).
        assert!(
            !trace.final_answer.contains("| Attribute |"),
            "single-subject query should not produce a comparison table: {}",
            trace.final_answer
        );
    }

    #[test]
    fn comparison_trace_records_real_bonds() {
        // Audit: every (from, bt, to) in bonds_used must correspond to
        // a real bond in the brain, with `from` in the subject set.
        let mut brain = comparison_brain_two_subjects();
        let trace = reason(
            &mut brain,
            "compare cumulus and stratus",
            TemporalScope::Current,
        );
        assert!(!trace.no_chain);
        let cumulus = crate::seed::Seed::from_label("cumulus");
        let stratus = crate::seed::Seed::from_label("stratus");
        let subjects: std::collections::HashSet<Seed> = [cumulus, stratus].into_iter().collect();
        assert!(
            !trace.bonds_used.is_empty(),
            "comparison must record bonds"
        );
        for (from, _bt, to) in &trace.bonds_used {
            assert!(
                subjects.contains(from),
                "bond must originate at a comparison subject: {:?}",
                from
            );
            assert!(
                brain.network.get_bond(from, to).is_some(),
                "audit bond not present in brain"
            );
        }
        // Trace tags as Structured (no hedging).
        assert_eq!(trace.evidence_kind, EvidenceKind::Structured);
    }

    #[test]
    fn port_l_still_routes_when_compare_keywords_absent() {
        // Regression: Port L's enumerative dispatch must still fire
        // unaffected by Port M's addition. Use a brain shaped for Port L.
        let mut brain = Brain::new();
        let conf = parse_confidence("verified").unwrap();
        brain.teach("cumulus", BondType::IsA, "clouds", 0.9, conf);
        brain.teach("stratus", BondType::IsA, "clouds", 0.9, conf);
        brain.teach("cirrus", BondType::IsA, "clouds", 0.9, conf);
        let trace = reason(
            &mut brain,
            "what are the kinds of clouds",
            TemporalScope::Current,
        );
        assert!(!trace.no_chain);
        // Port L emits a numbered list \u{2014} must NOT emit the markdown
        // table header.
        assert!(
            !trace.final_answer.contains("| Attribute |"),
            "enumerative query must not route through Port M: {}",
            trace.final_answer
        );
    }

    #[test]
    fn port_i_still_routes_when_compare_keywords_absent() {
        // Regression: Port I's procedural dispatch is upstream of Port M.
        let mut brain = Brain::new();
        let conf = parse_confidence("verified").unwrap();
        brain.teach("baking", BondType::Requires, "flour", 0.9, conf);
        brain.teach("baking", BondType::Produces, "bread", 0.85, conf);
        let trace = reason(
            &mut brain,
            "how do i bake bread",
            TemporalScope::Current,
        );
        // Procedural articulation (Port I) emits "Step k." lines, not
        // a markdown comparison table.
        assert!(
            !trace.final_answer.contains("| Attribute |"),
            "procedural query must not route through Port M: {}",
            trace.final_answer
        );
    }

    #[test]
    fn declarative_query_unaffected_by_port_m() {
        // Regression: a plain declarative query routes through Port A-D.
        let mut brain = small_brain();
        let trace = reason(&mut brain, "cat", TemporalScope::Current);
        assert!(
            !trace.final_answer.contains("| Attribute |"),
            "declarative query must not route through Port M: {}",
            trace.final_answer
        );
    }


    // ===== Port H: deep reasoning chains via length-penalty relaxation =====
    // PORT_H_TESTS_INSERTED
    //
    // These tests target a relaxed length penalty that lets long typed
    // chains compete with shorter ones when an explicit "trace" intent
    // is in play.  See Learning #578 for the spec; calibration:
    //   penalty(intent, pattern) =
    //       0.02  if intent ∈ {Trace, Procedural}
    //       0.04  if pattern ∈ {Causal, Hierarchical, PropertyChain}
    //       0.10  default
    //   lowest matched penalty wins.

    #[test]
    fn port_h_max_chain_depth_constant_is_eight() {
        assert_eq!(MAX_CHAIN_DEPTH, 8);
    }

    #[test]
    fn port_h_default_max_chain_length_bumped_to_eight() {
        let cfg = ReasoningConfig::default();
        assert_eq!(
            cfg.max_chain_length, 8,
            "Port H requires default depth cap of 8 to surface deep chains"
        );
    }

    #[test]
    fn port_h_detect_intent_trace_forms() {
        assert_eq!(detect_intent("trace from sun to glucose"), QueryIntent::Trace);
        assert_eq!(detect_intent("trace the path from a to b"), QueryIntent::Trace);
        assert_eq!(detect_intent("step by step how does fire start"), QueryIntent::Trace);
        assert_eq!(detect_intent("end to end how does atp form"), QueryIntent::Trace);
        assert_eq!(detect_intent("what is the chain from x to y"), QueryIntent::Trace);
        assert_eq!(detect_intent("what's the chain from x to y"), QueryIntent::Trace);
        assert_eq!(detect_intent("how does sunlight lead to muscle motion"), QueryIntent::Trace);
    }

    #[test]
    fn port_h_length_penalty_full_at_known_lengths() {
        // Trace+typed bucket: relaxed penalty (coef 0.02) ×
        // depth bonus (1 + 0.08*(n-2)).  At n=2 the bonus factor
        // is 1.0 so length penalty equals 1.0; at n=3 we multiply
        // 1/1.02 × 1.08 ≈ 1.0588; at n=5 we multiply 1/1.06 × 1.24
        // ≈ 1.169; at n=8 we multiply 1/1.12 × 1.48 ≈ 1.321.
        let p_trace_2 = length_penalty_full(QueryIntent::Trace, ReasoningPattern::Causal, 2);
        let p_trace_3 = length_penalty_full(QueryIntent::Trace, ReasoningPattern::Causal, 3);
        let p_trace_5 = length_penalty_full(QueryIntent::Trace, ReasoningPattern::Causal, 5);
        let p_trace_8 = length_penalty_full(QueryIntent::Trace, ReasoningPattern::Causal, 8);
        assert!((p_trace_2 - 1.0).abs() < 1e-6);
        assert!((p_trace_3 - (1.0 / 1.02) * 1.08).abs() < 1e-4);
        assert!((p_trace_5 - (1.0 / 1.06) * 1.24).abs() < 1e-4);
        assert!((p_trace_8 - (1.0 / 1.12) * 1.48).abs() < 1e-4);
        // Trace boost is gated to typed patterns; Trace+Mixed gets
        // pure relaxed penalty (0.02 coef, no bonus).
        let p_trace_mixed_5 =
            length_penalty_full(QueryIntent::Trace, ReasoningPattern::Mixed, 5);
        assert!((p_trace_mixed_5 - 1.0 / 1.06).abs() < 1e-4);
        // Typed-pattern bucket (0.04): n=5 → 1/1.12
        let p_typed_5 = length_penalty_full(QueryIntent::Generic, ReasoningPattern::Causal, 5);
        assert!((p_typed_5 - 1.0 / 1.12).abs() < 1e-4);
        // Default bucket (0.10): n=5 → 1/1.30 (Mixed)
        let p_mixed_5 = length_penalty_full(QueryIntent::Generic, ReasoningPattern::Mixed, 5);
        assert!((p_mixed_5 - 1.0 / 1.30).abs() < 1e-4);
    }

    #[test]
    fn port_h_lowest_penalty_wins() {
        // Trace+Causal hits both relaxed buckets AND the typed-
        // pattern bonus, so it must beat (a) Generic+Causal
        // (penalty-only relaxation) and (b) Trace+Mixed (relaxed
        // penalty but no bonus, since bonus is gated to typed
        // patterns).
        let trace_causal = length_penalty_full(QueryIntent::Trace, ReasoningPattern::Causal, 5);
        let generic_causal =
            length_penalty_full(QueryIntent::Generic, ReasoningPattern::Causal, 5);
        let trace_mixed = length_penalty_full(QueryIntent::Trace, ReasoningPattern::Mixed, 5);
        assert!(
            trace_causal > generic_causal,
            "Trace+Causal must outscore Generic+Causal at len 5"
        );
        assert!(
            trace_causal > trace_mixed,
            "Trace+Causal must outscore Trace+Mixed at len 5 (typed bonus)"
        );
        // Trace+Mixed still gets the relaxed penalty (coef 0.02)
        // even without the bonus.
        assert!(
            trace_mixed
                > length_penalty_full(QueryIntent::Generic, ReasoningPattern::Mixed, 5),
            "Trace+Mixed must outscore Generic+Mixed at len 5 (intent relaxation)"
        );
    }

    #[test]
    fn port_h_default_query_keeps_short_answer() {
        // Regression guard: `chain_score` (Generic intent) at length 2
        // is unchanged by Port H — the penalty floor only kicks in at
        // length >= 3.
        let s2 = chain_score(0.8, &[BondType::Causes, BondType::Causes]);
        // n=2: length_penalty_full → 1.0 regardless of bucket.
        let expected = ReasoningPattern::Causal.score_multiplier()
            * 0.8
            * 1.0  // length penalty at n=2
            * bond_priority(BondType::Causes);  // last-leg bonus
        assert!((s2 - expected).abs() < 1e-4, "score at n=2: got {} want {}", s2, expected);
    }

    #[test]
    fn port_h_deep_typed_trace_outscores_shallow_mixed() {
        // A 5-hop pure-Causal chain under Trace intent should beat
        // a 2-hop Mixed (RelatedTo) chain on the same min_weight.
        // Pre-Port-H the 2-hop wins by ~30% because the length
        // penalty over-suppresses depth.  Post-Port-H the deep typed
        // chain wins.
        let deep_causal = chain_score_with_intent(
            0.7,
            &[BondType::Causes; 5],
            QueryIntent::Trace,
        );
        let shallow_mixed = chain_score_with_intent(
            0.7,
            &[BondType::RelatedTo, BondType::RelatedTo],
            QueryIntent::Trace,
        );
        assert!(
            deep_causal > shallow_mixed,
            "Port H: deep Causal under Trace should outscore shallow Mixed; got deep={} shallow={}",
            deep_causal, shallow_mixed
        );
    }

    #[test]
    fn port_h_score_monotone_typed_at_constant_min_weight() {
        // Under Generic intent score still decreases monotonically
        // with chain length (no depth bonus); ensures the relaxation
        // didn't accidentally invert the curve in the default path.
        let s3 = chain_score_with_intent(0.7, &[BondType::Causes; 3], QueryIntent::Generic);
        let s5 = chain_score_with_intent(0.7, &[BondType::Causes; 5], QueryIntent::Generic);
        let s7 = chain_score_with_intent(0.7, &[BondType::Causes; 7], QueryIntent::Generic);
        assert!(s3 > s5, "score must decrease with depth under Generic: s3={} s5={}", s3, s5);
        assert!(s5 > s7, "score must decrease with depth under Generic: s5={} s7={}", s5, s7);
        // Under Trace+typed the score is INTENTIONALLY non-monotone
        // (depth bonus); deeper typed chains should win.  Capped at
        // MAX_CHAIN_DEPTH=8.
        let t3 = chain_score_with_intent(0.7, &[BondType::Causes; 3], QueryIntent::Trace);
        let t5 = chain_score_with_intent(0.7, &[BondType::Causes; 5], QueryIntent::Trace);
        let t8 = chain_score_with_intent(0.7, &[BondType::Causes; 8], QueryIntent::Trace);
        assert!(t5 > t3, "Trace+typed must reward depth: t3={} t5={}", t3, t5);
        assert!(t8 > t5, "Trace+typed must reward depth: t5={} t8={}", t5, t8);
    }

    #[test]
    fn port_h_typed_pattern_relaxation_helps_generic_too() {
        // Even without an explicit Trace intent, Generic+Causal at
        // length 5 should score above Generic+Mixed at length 5
        // — and by MORE than pre-Port-H (because Causal moves into
        // the 0.04 bucket while Mixed stays at 0.10).
        let causal_5 = chain_score_with_intent(
            0.7,
            &[BondType::Causes; 5],
            QueryIntent::Generic,
        );
        let mixed_5 = chain_score_with_intent(
            0.7,
            &[BondType::RelatedTo; 5],
            QueryIntent::Generic,
        );
        assert!(causal_5 > mixed_5);
        // Ratio post-Port-H must exceed the pre-Port-H ratio of
        // (1.6 * bond_priority(Causes) / 0.7 / bond_priority(RelatedTo))
        // (length penalty cancelled).  Concretely we expect the
        // typed-pattern penalty advantage to widen the gap.
        let ratio = causal_5 / mixed_5;
        assert!(
            ratio > 3.5,
            "Port H: Causal-vs-Mixed ratio at len 5 should widen; got {}",
            ratio
        );
    }

    #[test]
    fn port_h_deep_chain_wins_in_live_brain() {
        // Integration: build a brain with a typed 5-hop Causal chain
        // off `start` AND a parallel 2-hop RelatedTo chain off the
        // same anchor.  Under a Trace-intent query the deep chain
        // should articulate; under a Generic query the shorter typed
        // path should still articulate.
        let mut brain = Brain::new();
        let conf = parse_confidence("verified").unwrap();
        // Deep Causal: start -> a -> b -> c -> d -> end
        brain.teach("startnode", BondType::Causes, "alphanode", 0.85, conf);
        brain.teach("alphanode", BondType::Causes, "betanode", 0.85, conf);
        brain.teach("betanode", BondType::Causes, "gammanode", 0.85, conf);
        brain.teach("gammanode", BondType::Causes, "deltanode", 0.85, conf);
        brain.teach("deltanode", BondType::Causes, "endnode", 0.85, conf);
        // Shallow RelatedTo distractor anchored at startnode
        brain.teach("startnode", BondType::RelatedTo, "sidenode", 0.85, conf);
        let trace = reason(
            &mut brain,
            "trace from startnode to endnode",
            TemporalScope::Current,
        );
        assert!(!trace.no_chain, "trace failed: {}", trace.final_answer);
        assert!(
            trace.steps.len() >= 4,
            "Port H: trace intent should surface a 4+ hop chain, got {} steps",
            trace.steps.len()
        );
        let lc = trace.final_answer.to_lowercase();
        assert!(
            lc.contains("endnode") || lc.contains("deltanode"),
            "deep chain articulation must reach further nodes; got: {}",
            trace.final_answer
        );
    }

    #[test]
    fn port_h_shallow_typed_chain_preserved_under_default_intent() {
        // Regression: a brain with a clean 2-hop IsA chain under a
        // plain definitional query must still articulate the short
        // chain — Port H must not over-correct toward depth.
        let mut brain = small_brain();
        let trace = reason(&mut brain, "cat", TemporalScope::Current);
        assert!(!trace.no_chain);
        assert_eq!(
            trace.steps.len(),
            2,
            "definitional query must keep its 2-hop chain post-Port-H"
        );
    }

    // ===== Port H — extra integration cases =====
    // PORT_H_EXTRA_INTEGRATION

    #[test]
    fn port_h_extra_integration_trace_intent_routes_correctly() {
        // Build a brain with both a typed deep chain and a Mixed
        // distractor.  Verify the Trace intent is detected and the
        // chain articulation reaches deeper nodes than under a
        // Generic intent.
        let mut brain = Brain::new();
        let conf = parse_confidence("verified").unwrap();
        brain.teach("alphax", BondType::Causes, "betax", 0.85, conf);
        brain.teach("betax", BondType::Causes, "gammax", 0.85, conf);
        brain.teach("gammax", BondType::Causes, "deltax", 0.85, conf);
        brain.teach("deltax", BondType::Causes, "epsilonx", 0.85, conf);
        // Trace intent — expect to reach into the chain.
        let trace = reason(
            &mut brain,
            "trace from alphax to epsilonx",
            TemporalScope::Current,
        );
        assert!(!trace.no_chain);
        assert!(
            trace.steps.len() >= 3,
            "Trace intent on deep typed chain should articulate >= 3 hops, got {}",
            trace.steps.len()
        );
    }

    #[test]
    fn port_h_extra_integration_default_intent_keeps_short_chain() {
        // Same brain, default intent.  Should still pick the 2-hop
        // chain — Port H's depth bonus is gated to Trace intent only.
        let mut brain = Brain::new();
        let conf = parse_confidence("verified").unwrap();
        brain.teach("alphay", BondType::Causes, "betay", 0.85, conf);
        brain.teach("betay", BondType::Causes, "gammay", 0.85, conf);
        brain.teach("gammay", BondType::Causes, "deltay", 0.85, conf);
        brain.teach("deltay", BondType::Causes, "epsilony", 0.85, conf);
        let trace = reason(&mut brain, "alphay", TemporalScope::Current);
        assert!(!trace.no_chain);
        // Default intent (Generic) — composer's prefix preference
        // gives the 2-hop chain.
        assert_eq!(
            trace.steps.len(), 2,
            "Generic intent must keep 2-hop chain (Port H regression guard); got {}",
            trace.steps.len()
        );
    }

    #[test]
    fn port_h_extra_integration_max_chain_depth_caps_composition() {
        // Build a 10-hop chain.  Port H's hard cap (MAX_CHAIN_DEPTH=8)
        // must prevent the composer from articulating more than 8 hops.
        let mut brain = Brain::new();
        let conf = parse_confidence("verified").unwrap();
        for i in 0..10 {
            let from = format!("node{:02}", i);
            let to = format!("node{:02}", i + 1);
            brain.teach(&from, BondType::Causes, &to, 0.85, conf);
        }
        let trace = reason(
            &mut brain,
            "trace from node00 to node10",
            TemporalScope::Current,
        );
        assert!(!trace.no_chain);
        assert!(
            trace.steps.len() <= MAX_CHAIN_DEPTH,
            "chain composition exceeded MAX_CHAIN_DEPTH={}: got {} steps",
            MAX_CHAIN_DEPTH, trace.steps.len()
        );
    }

    #[test]
    fn port_h_extra_integration_chain_tie_prefers_longer_under_trace() {
        // Two chains with identical scores at the same depth: the
        // Port H tiebreak (longer wins) should already cover this
        // (untouched by Port H but verified to ensure we didn't
        // break the existing tiebreak).
        let mut brain = Brain::new();
        let conf = parse_confidence("verified").unwrap();
        brain.teach("sourcezz", BondType::Causes, "midzz", 0.85, conf);
        brain.teach("midzz", BondType::Causes, "endzz", 0.85, conf);
        let trace = reason(
            &mut brain,
            "trace from sourcezz to endzz",
            TemporalScope::Current,
        );
        assert!(!trace.no_chain);
        assert_eq!(
            trace.steps.len(),
            2,
            "2-hop chain should still articulate as 2 hops under Trace"
        );
    }

    // =================================================================
    // Port BR.4 -- recall-time typed-KU runtime gate
    // =================================================================

    /// Reasoning over a brain with explicitly-typed nodes (definition
    /// kind on both endpoints). With the flag on (default) the chain
    /// composer must emit `(definition)` tags into the articulation;
    /// with the flag off the same chain must collapse to the pre-Q.1
    /// untyped baseline (no parenthesized kind tags).
    #[test]
    fn br4_recall_typed_ku_flag_toggles_kind_tag_in_articulation() {
        let mut brain = Brain::new();
        let conf = parse_confidence("verified").unwrap();
        // Use the explicit-kind teach path so both endpoints carry
        // KnowledgeContent::Definition regardless of relation routing.
        brain.teach_with_kind(
            "democracy",
            BondType::IsA,
            "rule by the people",
            0.9,
            conf,
            Some("definition"),
            None,
        );
        brain.teach_with_kind(
            "rule by the people",
            BondType::IsA,
            "governance",
            0.9,
            conf,
            Some("definition"),
            None,
        );

        // Default: flag is true, articulation carries `(definition)`.
        assert!(brain.recall_config.typed_ku_enabled);
        let trace_on = reason(&mut brain, "democracy", TemporalScope::Current);
        assert!(!trace_on.no_chain);
        assert!(
            trace_on.final_answer.contains("(definition)"),
            "flag-on articulation must carry (definition) tag, got: {}",
            trace_on.final_answer,
        );

        // Flip the flag and re-run.
        brain.recall_config.typed_ku_enabled = false;
        let trace_off = reason(&mut brain, "democracy", TemporalScope::Current);
        assert!(!trace_off.no_chain);
        assert!(
            !trace_off.final_answer.contains("(definition)"),
            "flag-off articulation must NOT carry (definition) tag, got: {}",
            trace_off.final_answer,
        );
        // The structured ReasoningStep type does not surface the
        // raw kind tags (only the articulation does), so the assertion
        // above on `final_answer` is the load-bearing check.
    }

    /// Procedural assembly path also gates kind reads on the flag.
    /// Build a Requires->Produces sequence with explicit-kind nodes,
    /// then assert `assemble_procedure` strips kinds when off.
    #[test]
    fn br4_recall_typed_ku_flag_gates_assemble_procedure() {
        let mut brain = Brain::new();
        let conf = parse_confidence("verified").unwrap();
        brain.teach_with_kind(
            "baking bread",
            BondType::Requires,
            "flour",
            0.9,
            conf,
            Some("procedure"),
            None,
        );
        brain.teach_with_kind(
            "baking bread",
            BondType::Produces,
            "loaf",
            0.9,
            conf,
            Some("procedure"),
            None,
        );

        let anchor = crate::seed::Seed::from_label("baking bread");
        let cfg = ReasoningConfig::default();

        // Flag on (default).
        assert!(brain.recall_config.typed_ku_enabled);
        let steps_on = assemble_procedure(
            &brain,
            &anchor,
            &TemporalScope::Current,
            &cfg,
        );
        assert!(steps_on.iter().any(|s| s.articulation.contains("(procedure")),
            "flag-on at least one step must carry (procedure...) tag, got: {:?}",
            steps_on.iter().map(|s| &s.articulation).collect::<Vec<_>>(),
        );

        // Flag off.
        brain.recall_config.typed_ku_enabled = false;
        let steps_off = assemble_procedure(
            &brain,
            &anchor,
            &TemporalScope::Current,
            &cfg,
        );
        for s in &steps_off {
            assert!(
                !s.articulation.contains("(procedure"),
                "flag-off step articulation must NOT carry (procedure...) tag: {}",
                s.articulation,
            );
        }
    }
}
#[cfg(test)]
mod mapram_p5d_reasoning_tests {
    //! OP-BRAIN-MapRAM P5-batch2 (sub-batch B.2): byte-equivalence +
    //! resident-independence for the reasoning chain / analogy / procedure /
    //! sibling / comparison walks rerouted off the resident adjacency through
    //! the canonical edge-carrying seam (`seam_walk_outgoing` /
    //! `seam_walk_incoming`). Projections capture only the ADJACENCY/BOND-
    //! derived output (seeds + bond_type + weight/score bits); node CONTENT
    //! is the separate resident "B" thread and is EXCLUDED from the
    //! projection (the poison helpers overwrite it with "POISON"). Cache
    //! mode is byte-for-byte the legacy reads apart from the deliberate
    //! canonical neighbour ORDER (legacy resident order was HashMap-random
    //! per load), canonicalised in BOTH modes so output is deterministic.
    use super::*;
    use crate::bond::{BondType, KeyframeSource, TemporalKeyframe};
    use crate::brain::Brain;
    use crate::manifest::{KeyFingerprint, Manifest, MANIFEST_FILE_NAME};
    use crate::scope::TemporalScope;
    use crate::seed::Seed;
    use tempfile::tempdir;

    fn kf(w: f32) -> Vec<TemporalKeyframe> {
        vec![TemporalKeyframe { timestamp: 100, weight: w, source: KeyframeSource::Baseline }]
    }
    fn lbl(n: &str) -> Seed { Seed::from_label(&format!("p5d_r_{}", n)) }
    fn boot_v5(dir: &std::path::Path) -> Brain {
        let mut m = Manifest::new(KeyFingerprint::none());
        m.set_brain_format_v5();
        m.save_atomic(&dir.join(MANIFEST_FILE_NAME)).unwrap();
        Brain::load_v5(dir, None).expect("v5 brain must boot via load_v5")
    }
    const SPINE: &[(&str, &str, f32, BondType)] = &[
        ("q", "a", 0.90, BondType::IsA),
        ("q", "b", 0.80, BondType::HasProperty),
        ("q", "c", 0.70, BondType::Requires),
        ("q", "d", 0.60, BondType::Produces),
        ("a", "e", 0.85, BondType::RelatedTo),
        ("s1", "cat", 0.90, BondType::IsA),
        ("s2", "cat", 0.80, BondType::IsA),
        ("s3", "cat", 0.70, BondType::PartOf),
        ("s1", "p1", 0.90, BondType::HasProperty),
        ("s1", "p2", 0.60, BondType::HasProperty),
        ("s2", "p1", 0.90, BondType::HasProperty),
        ("s2", "p3", 0.70, BondType::HasProperty),
        ("cmp1", "t1", 0.90, BondType::IsA),
        ("cmp1", "t2", 0.80, BondType::HasProperty),
        ("cmp2", "t3", 0.85, BondType::IsA),
        ("cmp2", "t4", 0.75, BondType::HasProperty),
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
    fn cfg() -> ReasoningConfig { ReasoningConfig::default() }

    type LegProj = (Vec<u8>, Vec<u8>, BondType, u32, bool);
    fn proj_legs(legs: &[ChainLeg]) -> Vec<LegProj> {
        legs.iter()
            .map(|l| (l.from.as_bytes().to_vec(), l.to.as_bytes().to_vec(), l.bond_type, l.weight.to_bits(), l.negated))
            .collect()
    }
    fn proj_chain(o: &Option<Vec<ChainLeg>>) -> Option<Vec<LegProj>> { o.as_ref().map(|v| proj_legs(v)) }
    fn proj_depth_one(v: &[PartialChain]) -> Vec<(Vec<LegProj>, u32, u32)> {
        v.iter().map(|pc| (proj_legs(&pc.legs), pc.min_weight.to_bits(), pc.score.to_bits())).collect()
    }
    fn proj_procedure(v: &[ProcedureStep]) -> Vec<(usize, Vec<u8>, Vec<u8>, BondType, u32)> {
        v.iter().map(|s| (s.ordinal, s.from.as_bytes().to_vec(), s.to.as_bytes().to_vec(), s.bond_type, s.confidence.to_bits())).collect()
    }
    #[allow(clippy::type_complexity)]
    fn proj_siblings(t: &(Vec<SiblingConcept>, usize)) -> (Vec<(usize, Vec<u8>, Vec<u8>, BondType, u32)>, usize) {
        (t.0.iter().map(|s| (s.ordinal, s.sibling.as_bytes().to_vec(), s.anchor.as_bytes().to_vec(), s.bond_type, s.confidence.to_bits())).collect(), t.1)
    }
    #[allow(clippy::type_complexity)]
    fn proj_comparison(o: &Option<ComparisonTable>) -> Option<(Vec<Vec<u8>>, Vec<(BondType, Vec<(Vec<u8>, Vec<u8>, u32)>)>)> {
        o.as_ref().map(|tbl| (
            tbl.subjects.iter().map(|(s, _)| s.as_bytes().to_vec()).collect(),
            tbl.rows.iter().map(|r| (r.bond_type, r.cells.iter().map(|c| (c.subject.as_bytes().to_vec(), c.target.as_bytes().to_vec(), c.weight.to_bits())).collect())).collect(),
        ))
    }

    /// Every reasoning walk is byte-identical legacy vs cache (canonical
    /// order both modes) on the adjacency/bond-derived projection, and
    /// non-vacuous.
    #[test]
    fn p5d_reasoning_walks_byte_equivalent() {
        let dir = tempdir().unwrap();
        { let b = boot_v5(dir.path()); teach(&b); }
        let brain = load_routed(dir.path());
        let s = TemporalScope::Current;
        let c = cfg();

        brain.set_adjacency_cache_backed(false);
        let l_chain = proj_chain(&best_chain(&brain, &[lbl("q")], &s, &c, QueryIntent::What));
        let l_d1 = proj_depth_one(&depth_one_chains(&brain, &[lbl("q")], &s, &c, QueryIntent::What));
        let l_proc = proj_procedure(&assemble_procedure(&brain, &lbl("q"), &s, &c));
        let l_sib = proj_siblings(&gather_siblings(&brain, &lbl("cat"), &s, &c, 10));
        let l_cmp = proj_comparison(&gather_comparison(&brain, &[lbl("cmp1"), lbl("cmp2")], &s, &c));

        brain.set_adjacency_cache_backed(true);
        let c_chain = proj_chain(&best_chain(&brain, &[lbl("q")], &s, &c, QueryIntent::What));
        let c_d1 = proj_depth_one(&depth_one_chains(&brain, &[lbl("q")], &s, &c, QueryIntent::What));
        let c_proc = proj_procedure(&assemble_procedure(&brain, &lbl("q"), &s, &c));
        let c_sib = proj_siblings(&gather_siblings(&brain, &lbl("cat"), &s, &c, 10));
        let c_cmp = proj_comparison(&gather_comparison(&brain, &[lbl("cmp1"), lbl("cmp2")], &s, &c));

        assert_eq!(l_chain, c_chain, "best_chain byte-differs legacy vs cache");
        assert_eq!(l_d1, c_d1, "depth_one_chains byte-differs legacy vs cache");
        assert_eq!(l_proc, c_proc, "assemble_procedure byte-differs legacy vs cache");
        assert_eq!(l_sib, c_sib, "gather_siblings byte-differs legacy vs cache");
        assert_eq!(l_cmp, c_cmp, "gather_comparison byte-differs legacy vs cache");

        assert!(c_chain.as_ref().map(|v| !v.is_empty()).unwrap_or(false), "best_chain vacuous");
        assert!(!c_d1.is_empty(), "depth_one_chains vacuous");
        assert!(!c_proc.is_empty(), "assemble_procedure vacuous");
        assert!(!c_sib.0.is_empty(), "gather_siblings vacuous");
        assert!(c_cmp.is_some(), "gather_comparison vacuous");
    }

    /// INDEPENDENCE: in cache mode every reasoning walk's adjacency/bond-
    /// derived projection is invariant after POISONING every resident
    /// structure the legacy path reads (adjacency cleared, weights+keyframes
    /// zeroed, bond TYPES flipped, keyframes overwritten with a sentinel) --
    /// proving neighbours come from the map and the bond (weight/type/
    /// keyframes) from the carried payload, never resident neighbors_*/
    /// get_bond. Sanity: the legacy reads DO diverge.
    #[test]
    fn p5d_reasoning_independent_of_resident() {
        let dir = tempdir().unwrap();
        { let b = boot_v5(dir.path()); teach(&b); }
        let mut brain = load_routed(dir.path());
        let s = TemporalScope::Current;
        let c = cfg();
        brain.set_adjacency_cache_backed(true);

        let base_chain = proj_chain(&best_chain(&brain, &[lbl("q")], &s, &c, QueryIntent::What));
        let base_d1 = proj_depth_one(&depth_one_chains(&brain, &[lbl("q")], &s, &c, QueryIntent::What));
        let base_proc = proj_procedure(&assemble_procedure(&brain, &lbl("q"), &s, &c));
        let base_sib = proj_siblings(&gather_siblings(&brain, &lbl("cat"), &s, &c, 10));
        let base_cmp = proj_comparison(&gather_comparison(&brain, &[lbl("cmp1"), lbl("cmp2")], &s, &c));
        assert!(base_chain.as_ref().map(|v| !v.is_empty()).unwrap_or(false), "baseline chain vacuous");
        assert!(!base_proc.is_empty(), "baseline procedure vacuous");
        assert!(!base_sib.0.is_empty(), "baseline siblings vacuous");
        assert!(base_cmp.is_some(), "baseline comparison vacuous");

        brain.network.mapram_test_poison_resident();
        brain.network.mapram_test_corrupt_resident_bond_types();
        brain.network.mapram_test_corrupt_resident_keyframes();

        brain.set_adjacency_cache_backed(true);
        assert_eq!(proj_chain(&best_chain(&brain, &[lbl("q")], &s, &c, QueryIntent::What)), base_chain, "cache best_chain changed under poison");
        assert_eq!(proj_depth_one(&depth_one_chains(&brain, &[lbl("q")], &s, &c, QueryIntent::What)), base_d1, "cache depth_one_chains changed under poison");
        assert_eq!(proj_procedure(&assemble_procedure(&brain, &lbl("q"), &s, &c)), base_proc, "cache assemble_procedure changed under poison");
        assert_eq!(proj_siblings(&gather_siblings(&brain, &lbl("cat"), &s, &c, 10)), base_sib, "cache gather_siblings changed under poison");
        assert_eq!(proj_comparison(&gather_comparison(&brain, &[lbl("cmp1"), lbl("cmp2")], &s, &c)), base_cmp, "cache gather_comparison changed under poison");

        brain.set_adjacency_cache_backed(false);
        assert_ne!(proj_procedure(&assemble_procedure(&brain, &lbl("q"), &s, &c)), base_proc, "poison must corrupt the legacy procedure walk (else vacuous proof)");
    }
}
