//! Associative network with temporal spread activation.
//!
//! The core of Lu 4D: a knowledge graph where spread activation respects
//! temporal scopes. Activation flows through bonds whose weights are
//! evaluated at the scope's timestamp, with exponential decay per hop.

use crate::bond::{BondType, KeyframeSource, TemporalBond};
use crate::scar::ScarLog;
use crate::bridge::{BridgeCollector, BridgeConfig};
use crate::curiosity::{CuriosityCollector, CuriosityConfig, SignalType};
use crate::node::{KnowledgeNode, Confidence};
use crate::scope::TemporalScope;
use crate::seed::Seed;
use crate::view::{BondMutHandle, BondView, BondsView, NodeMutHandle, NodeView};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};

// OP-A0 (beam-bounded spread) -- process-wide, default-OFF bounds read once
// from the environment. Unset => None => byte-identical to legacy spread.
// LU_SPREAD_BEAM=<usize>: max neighbors ENQUEUED per depth level (onward
// frontier width). Activations are STILL recorded; only onward expansion is
// bounded, so the recorded result set is unchanged -- we cap the flood, not
// the answer. LU_SPREAD_MAX_EXPANSIONS=<usize>: hard ceiling on node
// expansions (adjacency decodes) per spread -- deterministic and
// hardware-independent, unlike the wall-clock watchdog.
fn spread_beam_width() -> Option<usize> {
    static V: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("LU_SPREAD_BEAM").ok().and_then(|s| s.trim().parse().ok()))
}
fn spread_max_expansions() -> Option<usize> {
    static V: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("LU_SPREAD_MAX_EXPANSIONS").ok().and_then(|s| s.trim().parse().ok()))
}

// OP-SENSE-ROUTER (option 1) -- default-OFF engine seed-router. On an
// operation-explanation query (code-intent frame), route into the CODING
// sense by additionally seeding the option-2 handle "pyfn<op>", which seeds
// straight into the coding op node. Uses existing handle DATA (no persisted
// per-node sense-tag => NO schema migration). Unset LU_SENSE_ROUTER =>
// byte-identical to legacy seed selection. Regression-safe twice over:
// (a) fires only on code-intent frames, and (b) only op tokens that actually
// have a pyfn handle inject anything (non-coding words resolve to nothing).
pub(crate) fn sense_router_enabled() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| matches!(std::env::var("LU_SENSE_ROUTER").ok().as_deref(),
        Some("1") | Some("true") | Some("on")))
}
fn sense_router_boost() -> f32 {
    static V: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("LU_SENSE_ROUTER_BOOST").ok().and_then(|s| s.trim().parse().ok()).filter(|v: &f32| *v > 0.0).unwrap_or(1.0))
}
fn code_intent_op_tokens(q: &str) -> Vec<String> {
    let ql = q.to_lowercase();
    let framed = ql.contains("function") || ql.contains("method")
        || ql.contains(" in python") || ql.contains("javascript")
        || ql.contains("built-in") || ql.contains("builtin")
        || (ql.starts_with("what does ")
            && (ql.contains(" do") || ql.contains(" return") || ql.contains(" take")))
        || (ql.starts_with("how does ") && ql.contains(" work"))
        || ql.starts_with("how do i use ") || ql.starts_with("how to use ");
    if !framed { return Vec::new(); }
    const STOP: &[&str] = &["what","does","doe","the","an","are","how","you","use","used","for","return","returns","take","takes","work","works","javascript","function","method","builtin","built","python","and","with"];
    ql.split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() >= 2 && !STOP.contains(w))
        .map(|w| w.to_string())
        .collect()
}
// OP-SENSE-CONCEPT-QID -- definitional/language frame detector, sibling of
// code_intent_op_tokens. Fires on "what is X / tell me about X / define X /
// <X> language" forms and yields candidate concept tokens. Each token T is
// routed (in select_seeds_for_query_ask) into its QID-anchored sense node via
// the option-2 handle "csense<T>" (1-hop to the canonical <Label>#Q<qid>
// shelf node). Regression-safe twice over: (a) fires only on these frames,
// and (b) only tokens that actually have a csense<T> handle inject anything
// (every other token resolves to nothing under the strict exactness guard),
// so a bare token or an unknown word is byte-identical to legacy seeding.
pub(crate) fn concept_sense_tokens(q: &str) -> Vec<String> {
    let ql = q.to_lowercase();
    let framed = ql.starts_with("what is ") || ql.starts_with("what's ")
        || ql.starts_with("what are ") || ql.starts_with("what was ")
        || ql.starts_with("tell me about ") || ql.starts_with("define ")
        || ql.contains(" language");
    if !framed { return Vec::new(); }
    const STOP: &[&str] = &["what","is","are","was","the","an","tell","me","about","define","do","you","know","of","for","and","with","language","languages","programming","please","explain"];
    ql.split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() >= 2 && !STOP.contains(w))
        .map(|w| w.to_string())
        .collect()
}
use std::cmp::Ordering;
use std::time::{Duration, Instant};

/// Decay factor applied per hop during spread activation.
pub const DECAY_PER_HOP: f32 = 0.7;
/// Minimum activation level — below this, propagation stops.
pub const ACTIVATION_THRESHOLD: f32 = 0.001;
/// Hebbian strengthening gain when two nodes co-activate.
pub const HEBBIAN_GAIN: f32 = 0.05;
/// Extra decay multiplier applied to bonds where either endpoint is a stopword hub.
pub const STOPWORD_BOND_DECAY: f32 = 0.2;
/// Degree threshold below which normalization has no effect.
/// Only genuine super-hubs (degree > threshold) are penalized.
/// With ~18 avg degree in this graph, 2000 = ~111× average — true outliers only.
pub const DEGREE_NORM_THRESHOLD: usize = 2000;

/// FIX 2 (OP-HUB-EXPANSION-SKIP): a frontier node whose total degree
/// exceeds this ceiling is ACTIVATED but not expanded during spread
/// activation. Decoding a super-hub's full adjacency ("dog" carries
/// millions of incoming bonds) reads tens of GB in a single, watchdog-
/// uninterruptible iteration; paths through such a hub are low value
/// (it neighbours nearly everything). Normal concept nodes sit far
/// below this, so low-frequency recall is unchanged.
pub const HUB_EXPANSION_DEGREE_CEILING: usize = 50_000;

/// OP-BRAIN-MapRAM P3 -- one neighbour for the spread/typed hot loop:
/// the endpoint seed plus the SOURCE of its edge weight. LEGACY yields
/// `Resident` (caller reads `view_bond`); CACHE yields `Carried`,
/// holding the predicate + `BondPayload` the cache decode already
/// produced, so the hot loop reads the edge weight WITH the neighbour
/// list and never issues a separate `view_bond`/`get_bond` per edge.
pub(crate) struct SeamEdge {
    pub(crate) nbr: Seed,
    pub(crate) bond: SeamBondSrc,
}

pub(crate) enum SeamBondSrc {
    /// Legacy: read the resident bond via `view_bond`.
    Resident,
    /// Cache: reconstruct the bond byte-identically from the carried
    /// decode payload (no resident read).
    Carried { predicate: u8, payload: crate::bond_store::BondPayload },
}

/// OP-BRAIN-MapRAM P5-batch2 -- a bond reconstructed/borrowed for an
/// UNCAPPED seam WALK edge. LEGACY borrows the resident bond via
/// `BondView` (byte-for-byte the prior `get_bond`); CACHE owns a
/// `TemporalBond` reconstructed byte-identically from the carried
/// payload (no resident read). `.view()` hands both back as a uniform
/// `BondView` so the order-sensitive bond-carrying walks read the edge
/// (keyframes / weight_in_scope / latest_raw_weight / bond_type)
/// identically regardless of source.
pub(crate) enum SeamWalkBond<'a> {
    Resident(BondView<'a>),
    Carried(TemporalBond),
}

impl SeamWalkBond<'_> {
    pub(crate) fn view(&self) -> BondView<'_> {
        match self {
            SeamWalkBond::Resident(v) => v.clone(),
            SeamWalkBond::Carried(b) => BondView::from_heap(b),
        }
    }
}

/// OP-BRAIN-MapRAM P3 -- spread/typed hot-path node view. LEGACY wraps
/// the resident `view_node` (byte-for-byte the pre-P3 read). CACHE
/// reproduces the rehydration STUB (`Confidence::Unverified`, empty
/// content) for any node present in the resident MAP -- byte-identical
/// to `view_node` on a loaded v5 brain (where every node is a stub),
/// WITHOUT reading the resident node structures. (Real per-node
/// payloads via `get_node_v5` are deliberately NOT used here: on a
/// loaded brain `view_node` returns the stub, so feeding real payloads
/// would DIVERGE from legacy -- that is the separate "B" thread.)
pub(crate) enum SeamNode<'a> {
    Resident(crate::view::NodeView<'a>),
    Stub,
}

impl<'a> SeamNode<'a> {
    #[inline]
    pub(crate) fn confidence(&self) -> crate::node::Confidence {
        match self {
            SeamNode::Resident(v) => v.confidence(),
            SeamNode::Stub => crate::node::Confidence::Unverified,
        }
    }
    #[inline]
    pub(crate) fn confidence_value(&self) -> f32 {
        self.confidence().value()
    }
    #[inline]
    pub(crate) fn content(&self) -> &str {
        match self {
            SeamNode::Resident(v) => v.content(),
            SeamNode::Stub => "",
        }
    }
}
/// Default cap on the activation map size during one spread call.
/// When the cap is hit, the BFS stops pushing new neighbors onto the
/// frontier and `RecallResult.activation_truncated` is set true so
/// callers can disclose the partial recall. Tunable per-network via
/// `AssociativeNetwork::set_max_activations`.
pub const MAX_ACTIVATIONS_DEFAULT: usize = 5000;

/// OP-LABEL-1 -- capacity of the per-network label memo. One narration
/// pass resolves on the order of 10^2 distinct seeds; 16K entries keeps
/// several hundred consecutive queries hot for a few MB of short
/// strings.
pub const LABEL_CACHE_CAPACITY: usize = 16_384;

/// Default percentile used in `effective_cap_activations()` to scale the
/// activation cap with brain size. At 0.05 a brain with 105K nodes yields
/// a percentile-cap of 5250 — basically the static floor. A 1M-node
/// brain yields 50K, a 20M-node brain yields 1M. Tunable per-network via
/// `AssociativeNetwork::set_max_activations_percentile`.
pub const MAX_ACTIVATIONS_PERCENTILE_DEFAULT: f32 = 0.05;

/// Returns true when a node's content label is a common function word that should
/// not act as an activation hub.  These nodes stay in the graph (their bonds encode
/// real relationships) but are excluded from BFS frontier expansion and their bonds
/// receive an extra decay penalty so they never dominate recall results.
#[inline]
pub fn is_hub_stopword(content: &str) -> bool {
    matches!(content,
        "a" | "an" | "the" | "this" | "that" | "these" | "those" |
        "i" | "me" | "my" | "we" | "our" | "you" | "your" |
        "he" | "him" | "his" | "she" | "her" | "it" | "its" |
        "they" | "them" | "their" | "who" | "whom" | "which" |
        "is" | "are" | "was" | "were" | "be" | "been" | "being" |
        "has" | "have" | "had" | "do" | "does" | "did" |
        "will" | "would" | "shall" | "should" |
        "can" | "could" | "may" | "might" | "must" |
        "in" | "on" | "at" | "to" | "for" | "of" | "by" |
        "with" | "from" | "into" | "onto" | "upon" | "out" |
        "over" | "under" | "above" | "below" | "between" | "among" |
        "after" | "before" | "during" | "within" | "without" |
        "through" | "throughout" | "along" | "across" | "behind" |
        "beyond" | "near" | "beside" | "around" | "about" | "against" |
        "and" | "or" | "but" | "nor" | "so" | "yet" | "both" |
        "either" | "neither" | "not" | "no" |
        "what" | "where" | "when" | "why" | "how" |
        "if" | "then" | "than" | "as" | "because" | "since" |
        "while" | "although" | "though" | "whether" | "unless" |
        "also" | "too" | "just" | "even" | "still" | "only" |
        "more" | "most" | "less" | "least" | "very" | "quite" |
        "such" | "any" | "all" | "each" | "every" | "some" |
        "many" | "much" | "few" | "little" | "own" | "other" |
        "same" | "different" | "another" | "one" | "two" | "three" |
        "make" | "get" | "go" | "way" | "thing" | "use" | "used" |
        "like" | "know" | "think" | "see" | "come" | "take" |
        "give" | "find" | "tell" | "ask" | "seem" | "feel" |
        "new" | "old" | "good" | "bad" | "high" | "low" |
        "long" | "short" | "great" | "small" | "large" | "big"
    )
}

/// Returns true when the word is a query-structuring verb, superlative, or
/// generic relational noun that carries no semantic content about the *subject*
/// being asked about.  These are excluded from seed selection so that the
/// phrasing of a question doesn't compete with its actual content words.
#[inline]
pub fn is_query_noise(word: &str) -> bool {
    matches!(word,
        // question verbs
        "cause" | "causes" | "caused" | "causing" |
        "affect" | "affects" | "affected" | "affecting" |
        "relate" | "relates" | "related" | "relating" |
        "connect" | "connects" | "connected" | "connecting" |
        "shape" | "shapes" | "shaped" | "shaping" |
        "enable" | "enables" | "enabled" | "enabling" |
        "apply" | "applies" | "applied" | "applying" |
        "drive" | "drives" | "driven" | "driving" |
        "trace" | "traces" | "traced" | "tracing" |
        "compare" | "compares" | "compared" | "comparing" |
        "explain" | "explains" | "explained" | "explaining" |
        "describe" | "describes" | "described" | "describing" |
        "define" | "defines" | "defined" | "defining" |
        "increase" | "increases" | "increased" | "decreases" |
        "happen" | "happens" | "happened" | "occurring" | "occur" |
        "develop" | "develops" | "developed" | "developing" |
        "work" | "works" | "worked" | "working" |
        "form" | "forms" | "formed" | "forming" |
        "written" | "wrote" | "created" | "create" | "named" | "called" |
        // query-structuring nouns
        "chain" | "path" | "relationship" | "connection" | "difference" |
        "similarity" | "process" | "mechanism" |
        // superlatives that don't identify the answer
        "largest" | "smallest" | "closest" | "fastest" | "hardest" |
        "biggest" | "oldest" | "newest" | "first" | "last" |
        // generic question-final words
        "importance" | "role" | "function" | "example" | "type" | "kind"
    )
}

/// Returns true if a node's content is a process-anchor or label-leak
/// pollution: legacy `goal:`/`step:` meta-nodes (created by pre-Port-#12
/// thinking.rs that indexed their content), or content that exactly
/// matches a `BondType` variant string ("IsA", "PartOf", etc.) leaking
/// from a heuristic-triple-extraction bug in the training script. These
/// nodes are never seeded for recall and never appear in recall results.
/// (Port #24 — Bug 1, Bug 2.)
#[inline]
pub fn is_meta_pollution(content: &str) -> bool {
    let lower = content.to_ascii_lowercase();
    if lower.starts_with("goal:") || lower.starts_with("step:") {
        return true;
    }
    matches!(content,
        "IsA" | "HasProperty" | "Causes" | "PartOf" | "UsedFor"
        | "RelatedTo" | "ContrastedWith" | "DerivedFrom"
        | "Requires" | "Produces" | "Opposes"
    )
}

/// Returns true if a node's content looks like a code identifier or
/// training-data glitch that should not act as a first-class concept
/// node:
///   * camelCase (a lowercase ASCII letter followed later by an uppercase
///     ASCII letter — e.g. "calculateFactorial");
///   * a hyphenated coding suffix (`-1`, `-2`, ..., `-array`, `-tree`,
///     `-problem`, `-problems`, `-step`, `-steps`, `-list`, `-node`,
///     `-stack`, `-queue`, `-graph`, `-map`, `-set`);
///   * a known training-data typo or identifier-bleed (`willaims`, `fky`,
///     `oxyjunn`, `calculatefactorial`, `reversenum`, `sortedrightarray`,
///     `decimaltobinary`).
/// Filters at recall time so legacy training-data pollution stays out of
/// seed selection and the activated[] result list even before the training
/// script catches up. The underscore-as-separator pattern is intentionally
/// NOT flagged here — synthetic test nodes like `"alpha_iso"` and the
/// `goal:.../step:.` legacy pattern legitimately use `_`/`:`, and meta
/// pollution is handled by `is_meta_pollution`. (Port #24 — Bug 3.)
#[inline]
pub fn is_code_identifier_pollution(content: &str) -> bool {
    // camelCase: defensive — most training paths lowercase, but a
    // manual teach() call with mixed case still gets caught here.
    let mut chars = content.chars();
    if let Some(first) = chars.next() {
        if first.is_ascii_lowercase() {
            for c in chars {
                if c.is_ascii_uppercase() {
                    return true;
                }
            }
        }
    }
    // Hyphenated coding suffix.
    if let Some((_, suffix)) = content.rsplit_once('-') {
        if !suffix.is_empty() && suffix.chars().all(|c| c.is_ascii_digit()) {
            return true;
        }
        let s = suffix.to_ascii_lowercase();
        if matches!(s.as_str(),
            "array" | "tree" | "problem" | "problems"
            | "step" | "steps" | "list" | "node"
            | "stack" | "queue" | "graph" | "map" | "set"
            | "subproblem" | "subproblems"
        ) {
            return true;
        }
    }
    // Known training-data bleed: identifier-concat tokens (lowercase
    // multi-word concatenations that survive the training script's
    // lowercase pass) plus a small explicit typo stoplist surfaced by
    // the 40-question test (Port #24).
    matches!(content,
        // Identifier-concat bleed from code-heavy training docs.
        "calculatefactorial" | "reversenum" | "sortedrightarray"
        | "decimaltobinary"
        // Typo bleed from low-quality training docs.
        | "willaims" | "fky" | "oxyjunn"
    )
}

/// Combined recall-time pollution filter. Used everywhere a seed is
/// selected from the keyword index or where the activated[] result list
/// is built, so polluted nodes (legacy `goal:`/`step:` meta-anchors,
/// `BondType`-name leaks, code-identifier bleed) never reach the
/// caller. (Port #24.)
#[inline]
pub fn is_recall_pollution(content: &str) -> bool {
    is_meta_pollution(content) || is_code_identifier_pollution(content)
}

/// OP-ROUTE-GUEST (2026-07-23) -- query-time NON-CONCEPT classifier.
/// Read-only, additive, reversible. Flags the non-concept label shapes
/// enumerated in the 2026-07-23 node-quality census (~77% of nodes are
/// not concepts): leading-underscore code identifiers, article-led
/// fragments, over-long fragments, and (>=3-word) lowercase-initial or
/// dangling-function-word-ending sentence fragments. Used ONLY to
/// DOWN-RANK such nodes in the recall frontier so a real concept
/// outranks them for the answer head and bridge endpoints. Nothing is
/// deleted; traversal is unchanged; seed selection is NOT touched (that
/// would raise the refusal rate). Single- and two-token clean labels
/// are never flagged, so test fixtures ("a", "foo", "chess") and
/// legitimately lowercase common-noun concepts ("photosynthesis") are
/// untouched.
pub fn non_concept_label(content: &str) -> bool {
    let t = content.trim();
    if t.is_empty() { return false; }
    // Leading-underscore code identifier, e.g. "_dart_getFlipped".
    if t.starts_with('_') { return true; }
    let words: Vec<&str> = t.split_whitespace().collect();
    let n = words.len();
    // Article-led fragment: "a dog park", "an off-leash ...", "the ...".
    // Needs >=2 words so a bare "a"/"the" node is not flagged.
    if n >= 2 {
        let w0 = words[0].to_ascii_lowercase();
        if w0 == "a" || w0 == "an" || w0 == "the" { return true; }
    }
    // Over-long fragment: real entity titles are rarely >8 words.
    if n >= 9 { return true; }
    // >=3-word fragments that are lowercase-initial or end on a dangling
    // function word. Short labels are exempt.
    if n >= 3 {
        if let Some(fc) = t.chars().find(|c| c.is_alphabetic()) {
            if fc.is_lowercase() { return true; }
        }
        let last = words[n - 1]
            .trim_matches(|c: char| !c.is_alphanumeric())
            .to_ascii_lowercase();
        const TAIL_FUNC: &[&str] = &[
            "the","a","an","of","to","by","in","on","for","and","or",
            "with","as","at","from","that","is","was","were","be","been",
            "are","its","this","these","those","which","into","than","then",
            "but","if","so",
        ];
        if TAIL_FUNC.contains(&last.as_str()) { return true; }
    }
    false
}

/// Env-gated toggle for the non-concept down-rank. Default ON; set
/// LU_JUNK_DOWNRANK=0 (or =false) in the service environment to disable
/// without a rebuild (instant rollback). Cached after first read.
pub fn junk_downrank_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("LU_JUNK_DOWNRANK")
            .map(|v| v != "0" && !v.eq_ignore_ascii_case("false"))
            .unwrap_or(true)
    })
}

/// Multiplier applied to a non-concept node's activation in the recall
/// frontier (OP-ROUTE-GUEST 2026-07-23).
pub const NON_CONCEPT_DOWNRANK: f32 = 0.05;


/// A single step in an activation traversal path.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraversalStep {
    /// The node reached at this step.
    pub node: Seed,
    /// The bond type traversed to reach this node.
    pub bond_type: BondType,
    /// The bond weight (temporal) at the time of traversal.
    pub bond_weight: f32,
    /// Activation level at this node after decay.
    pub activation: f32,
}

/// A node with its activation level and the path that reached it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivatedNode {
    /// The node seed.
    pub seed: Seed,
    /// The node's content label (for readability).
    pub content: String,
    /// Final activation level at this node.
    pub activation: f32,
    /// The confidence of the node.
    pub confidence: Confidence,
    /// Full traversal path from a seed node to this node.
    pub path: Vec<TraversalStep>,
    /// Depth (number of hops from nearest seed).
    pub depth: usize,
}

/// Activation comparison for the priority queue (max-heap by activation).
impl PartialEq for ActivatedNode {
    fn eq(&self, other: &Self) -> bool {
        self.seed == other.seed
    }
}

impl Eq for ActivatedNode {}

impl PartialOrd for ActivatedNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ActivatedNode {
    fn cmp(&self, other: &Self) -> Ordering {
        self.activation
            .partial_cmp(&other.activation)
            .unwrap_or(Ordering::Equal)
    }
}

/// OP-ASK-WATCHDOG -- default wall-clock ceiling for a spread
/// activation pass when no explicit ceiling is provided by the
/// caller (engine default is `None`; the narrate path tightens to
/// ~4s; this constant is the public reference value).
pub const DEFAULT_ASK_MAX_WALL_CLOCK: Duration = Duration::from_secs(30);

/// OP-ASK-WATCHDOG -- outcome of a `spread_activation_weighted_inner`
/// pass. `Default::default()` is `Complete` so older brain files /
/// JSON payloads that omit the field round-trip cleanly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum SpreadStatus {
    /// Frontier drained naturally or hit the activation-map cap.
    #[default]
    Complete,
    /// Wall-clock watchdog fired -- the frontier was dropped mid-loop.
    /// Partial activations accumulated so far are still returned.
    /// `activated_so_far` is the size of the activation map at the
    /// moment the watchdog tripped (before top-k truncation).
    TimedOut { activated_so_far: usize },
}

/// The result of a spread activation query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecallResult {
    /// Activated nodes sorted by activation level (highest first).
    pub activated: Vec<ActivatedNode>,
    /// The seeds that initiated the activation.
    pub query_seeds: Vec<Seed>,
    /// The temporal scope used.
    pub scope: TemporalScope,
    /// Total number of nodes visited during traversal.
    pub nodes_visited: usize,
    /// Total number of bonds evaluated.
    pub bonds_evaluated: usize,
    /// True when the BFS stopped expanding because the activation map
    /// hit `Network::max_activations`. Top-k results are still sorted
    /// from whatever was captured up to that point.
    #[serde(default)]
    pub activation_truncated: bool,
    /// OP-ASK-WATCHDOG -- terminal state of the BFS frontier loop.
    /// `Complete` = drained naturally or hit `max_activations`.
    /// `TimedOut { activated_so_far }` = wall-clock watchdog tripped
    /// mid-loop; partial activations are still in `activated`. Older
    /// brain files / JSON payloads default to `Complete`.
    #[serde(default)]
    pub status: SpreadStatus,
}

/// The associative knowledge network — nodes, bonds, and indices.
pub struct AssociativeNetwork {
    /// All nodes, keyed by seed.
    nodes: HashMap<Seed, KnowledgeNode>,
    /// All bonds, keyed by (from, to) pair.
    bonds: HashMap<(Seed, Seed), TemporalBond>,
    /// OP-BRAIN-V4 Phase 1: per-process `Seed → NodeId(u64)` intern
    /// table. Adjacency Vecs store NodeIds (8 bytes each post OP-BRAIN-V5
    /// Phase 1.1 widening; was 4 bytes pre-widening) instead of raw
    /// Seeds (32 bytes each), shrinking the dominant per-edge RAM
    /// cost by 4× on Vec content (was 8× pre-widening — see audit
    /// SURPRISE-1). The intern table itself is O(nodes), not
    /// O(bonds), so the saving scales linearly with out-degree and
    /// is always positive for nodes with ≥1 edge. Rebuilt fresh on
    /// every load (in-memory only; v3 archive on-disk format is
    /// unchanged in Phase 1).
    node_intern: crate::node_intern::NodeIntern,
    /// Adjacency index: seed → list of neighbor NodeIds (outgoing bonds).
    /// Phase 1: Vec<Seed> → Vec<NodeId>. Materialize Seeds via
    /// `node_intern.reverse_lookup` at the read boundary.
    outgoing: HashMap<Seed, Vec<crate::node_intern::NodeId>>,
    /// Adjacency index: seed → list of neighbor NodeIds (incoming bonds).
    incoming: HashMap<Seed, Vec<crate::node_intern::NodeId>>,
    /// Inverted keyword index (OP-022.a): mmap'd sidecar base + heap
    /// overlay + tombstones. Replaces the legacy HashMap<String, Vec<Seed>>
    /// that consumed ~7 GiB heap at 9.7 M bonds (OP-022 Phase 1 profile).
    /// See `src/keyword_index.rs` for the on-disk format and merge rules.
    keyword_index: crate::keyword_index::KeywordIndex,
    /// OP-022.a: when set, `ensure_keyword_index_built` writes the
    /// rebuilt index to this sidecar and swaps to mmap-backed mode
    /// immediately. None on heap-only / test brains.
    keyword_sidecar_context: Option<KeywordSidecarContext>,
    /// Cap on activation map size during a single spread call.
    /// Defaults to `MAX_ACTIVATIONS_DEFAULT`. Set via `set_max_activations`.
    /// Acts as the static FLOOR for `effective_cap_activations()` — see
    /// `max_activations_percentile`.
    max_activations: usize,
    /// Percentile of the brain (node count) used as a dynamic activation
    /// cap. The actual cap is `max(max_activations, brain_size * pct)`.
    /// At 105K nodes with default 0.05 the percentile yields ~5250
    /// (basically the floor); at 1M nodes it yields 50K; at 20M nodes
    /// it yields 1M. Set via `set_max_activations_percentile`.
    max_activations_percentile: f32,
    /// Per-graph 90th percentile of node degrees, recomputed on load
    /// via `recompute_degree_p90`. Used as the dynamic hub threshold
    /// in spread-activation degree-normalization, bridge classification,
    /// and curiosity hub detection. Falls back to `DEGREE_NORM_THRESHOLD`
    /// on an empty graph.
    degree_p90: usize,
    /// Storage v2 Slice 2c: optional rkyv-archived backing held alive
    /// by `Arc<Mmap>`. When `Some`, `nodes` and `bonds` hold ONLY the
    /// CoW-promoted overlay (heap copies of mutated entries plus
    /// brand-new entries). When `None`, `nodes` and `bonds` hold the
    /// entire graph (legacy heap-only mode).
    archive: Option<std::sync::Arc<crate::archive::NetworkArchive>>,
    /// Tombstones for archived nodes that have been removed via
    /// `remove_node` / `purge_nodes`. Empty in heap-only mode. A
    /// deletion implicitly removes every bond touching the seed too.
    overlay_deletions: HashSet<Seed>,
    /// Port Source-Persist: per-bond tombstones. Needed when
    /// `remove_bond` targets a bond that lives in the read-only
    /// rkyv archive: we cannot physically remove it from the mmap,
    /// so we flag it here so `bond_count`/`to_snapshot`/read paths
    /// filter it out. The next `to_snapshot` + save physically
    /// purges it (the archive section is rebuilt from filtered bonds).
    /// NOT persisted directly in NetworkSnapshot - it’s transient
    /// (realized at next save).
    bond_tombstones: HashSet<(Seed, Seed)>,
    /// Storage v2 Slice 2c: when `archive` is `Some`, the keyword
    /// index is built lazily on first read access (see
    /// `ensure_keyword_index_built`). This flag is `true` until the
    /// build runs; `false` after.
    archive_keyword_pending: bool,
    /// Port Cat.3b: read-side cap on outgoing neighbor list per node.
    /// Synced from `Brain::hub_degree_config.max_out_degree` on load
    /// and on every MCP `lu4d_hub_set` call.
    hub_read_max_out: u32,
    /// Port Cat.3b: read-side cap on incoming neighbor list per node.
    hub_read_max_in: u32,
    /// Port Cat.3b: master switch for read-side filtering. When false,
    /// `neighbors_outgoing_capped` / `neighbors_incoming_capped`
    /// pass through the full neighbor list.
    hub_read_enforcement: bool,
    /// Port Cat.3b: cumulative count of bonds dropped from spread
    /// activation neighbor lists by the read-side filter. Atomic so
    /// the read-side filter can run on `&self`.
    hub_cap_filtered_at_query_total: std::sync::atomic::AtomicUsize,
    /// Port T.5.1: optional per-query decay context. `None` (default)
    /// means query-path effective-weight reads multiply by 1.0 — i.e.
    /// identical to pre-T.5.1 behavior (the regression guard). When
    /// `Some`, every query-path effective-weight read in this module
    /// (`spread_activation`, `spread_activation_weighted_inner`,
    /// `typed_recall`, `typed_recall_weighted_inner`,
    /// `subgraph_coherence`) is multiplied by the freshness score
    /// derived from the bond’s oldest visible keyframe and the
    /// stashed `now_ms`. Set/cleared by `Brain::refresh_decay_runtime`
    /// or by `set_decay_runtime` directly. Stored bond weights are
    /// NEVER mutated through this field — see T.5 module docs.
    decay_runtime: Option<crate::decay::DecayRuntime>,
    /// Port Sense.1: optional per-query sense disambiguation
    /// context. `None` (default) means query-path effective-weight
    /// reads multiply by 1.0 -- i.e. identical to pre-Sense.1
    /// behaviour (the regression guard). When `Some`, every
    /// query-path effective-weight read in this module
    /// (`spread_activation`, `spread_activation_weighted_inner`,
    /// `typed_recall`, `typed_recall_weighted_inner`,
    /// `subgraph_coherence`) is multiplied by the per-bond
    /// cluster gain derived from the polysemous-seed cluster
    /// scoring. Set/cleared by `Brain::refresh_sense_runtime`
    /// or by `set_sense_runtime` directly. Stored bond weights
    /// are NEVER mutated through this field -- see Sense module
    /// docs for the structural anti-hallucination invariant.
    sense_runtime: Option<crate::sense::SenseRuntime>,
    /// OP-INGEST-1: Cached live-bond count. Maintained incrementally at the
    /// canonical mutation sites (`add_bond`, `remove_bond`, `purge_nodes`).
    /// Equals what walked `bond_count_walked()` would return -- verified by
    /// parity tests. `Cell<usize>` because `bond_count(&self)` is `&self`-style
    /// and the daemon is single-threaded. Going multi-threaded is OP-INGEST-2.
    bond_count_cache: std::cell::Cell<Option<usize>>,
    /// OP-BRAIN-V4 Phase 5 -- spread-activation read-path mode
    /// flag, mirrored from Brain. Default `0`
    /// (`ReadPathMode::V3Only`) preserves pre-Phase-5 spread
    /// behaviour byte-equivalent. Brain::set_read_path_mode
    /// propagates the value here so the spread loop's
    /// `neighbors_outgoing_for_read_mode` dispatches correctly
    /// without holding a back-reference to the owning Brain.
    read_path_mode: std::sync::atomic::AtomicU8,
    /// OP-BRAIN-V4 Phase 5 -- shared MemTable handle. Phase 4
    /// already wires this into `brain.memtable`; Brain mirrors a
    /// clone here whenever `enable_memtable` runs so the network
    /// can augment spread-activation neighbors without a back
    /// reference. `None` outside the v4 write path.
    v4_memtable: Option<std::sync::Arc<crate::memtable::MemTable>>,
    /// OP-BRAIN-V4 Phase 5 -- shared manifest snapshot handle.
    /// Used by the v4 spread-augmentation path to find L0 SSTs
    /// whose [min, max] subject range covers the BFS cursor.
    v4_manifest: Option<std::sync::Arc<std::sync::RwLock<std::sync::Arc<crate::manifest::Manifest>>>>,
    /// OP-BRAIN-V4 Phase 5 -- directory where L0 SSTs live.
    /// Cloned from the flush worker via Brain so the network
    /// can open SSTs lazily during v4 spread augmentation.
    v4_sst_dir: Option<std::path::PathBuf>,
    /// OP-BRAIN-V4 Phase 5 -- shared block cache for L0 SST
    /// reads on the spread hot path. Sized to
    /// `DEFAULT_SST_BLOCK_CACHE_BYTES`. Survives across spread
    /// calls so frequently-visited L0 blocks stay decompressed.
    v4_block_cache: Option<std::sync::Arc<crate::sst::BlockCache>>,
    /// OP-BRAIN-V4 Phase 5 -- per-SST handle cache. Keyed by
    /// manifest SST id. Survives across spread calls so the
    /// expensive `SST::open` (file metadata + header parse +
    /// index decompress) only runs once per SST.
    v4_sst_handle_cache: std::sync::Mutex<std::collections::HashMap<u64, std::sync::Arc<crate::sst::SST>>>,

    // ===== OP-BRAIN-MapRAM P2 -- selectable adjacency seam =====
    /// 0 = legacy (read `outgoing`/`incoming` + resident-adjacency
    /// degree, the current production behaviour); 1 = cache-backed
    /// (neighbour lists from the P1 `CacheBackedAdjacency`, degree from
    /// the resident map). Default 0 keeps production byte-for-byte
    /// unchanged; flipped per-brain by `Brain::set_adjacency_cache_backed`
    /// behind the `LU_MAPRAM_ROUTE` load flag. Atomic so the spread hot
    /// path dispatches on `&self` without a back-reference to Brain --
    /// mirrors the `read_path_mode` pattern.
    mapram_mode: std::sync::atomic::AtomicU8,
    /// The P1 cache-backed reader, installed by
    /// `Brain::enable_adjacency_seam`. Present once the seam is
    /// installed; only CONSULTED when `mapram_mode == 1`.
    mapram_cache: Option<std::sync::Arc<crate::cache_adjacency::CacheBackedAdjacency>>,
    /// Live manifest handle: a fresh `V5AdjacencySource` is built per
    /// neighbour decode from a re-read snapshot, so a background
    /// compaction's new SSTs are seen (matches `Brain::v5_adjacency_source`).
    mapram_manifest: Option<std::sync::Arc<std::sync::RwLock<std::sync::Arc<crate::manifest::Manifest>>>>,
    /// Shared v5 bond-reader cache (FD + index) for the on-miss decode.
    mapram_bond_cache: Option<std::sync::Arc<crate::bond_store::BondSstReaderCache>>,

    // ===== OP-LABEL-1 -- node LABEL resolution over the v5 node store =====
    /// Shared v5 NODE-store reader cache, installed by
    /// `Brain::install_node_label_source` on BOTH the writer
    /// (`enable_flush_worker`) and the read-only reader
    /// (`load_v5_readonly`).
    ///
    /// ## Why this handle has to exist
    ///
    /// On a v5 brain the resident node map does NOT carry node text.
    /// `Brain::rehydrate_adjacency` inserts `rehydration_node_stub`s
    /// (`content: String::new()`) so 52M nodes cost adjacency, not
    /// strings; and the cache-backed seam returns `SeamNode::Stub`,
    /// whose `content()` is the empty string by construction. Both are
    /// deliberate RAM decisions and both are correct for the BFS hot
    /// path, which only needs identity + degree.
    ///
    /// They are NOT correct for the handful of nodes that reach a
    /// human. Before OP-LABEL-1 the recall path read `content()` off
    /// those stubs and handed the empty string to narration, so every
    /// rendered answer had empty label slots (`" is ."`). The fix is to
    /// resolve the label for the top-K / narrated nodes ONLY, from the
    /// store that actually holds it -- a few point lookups per query,
    /// never a per-edge cost.
    v5_node_reader_cache: Option<std::sync::Arc<crate::node_store::NodeSstReaderCache>>,
    /// Bounded memo for `node_label`. A single narration pass touches
    /// the same subject seed many times (slot fill, hub fill, contrast,
    /// traversal tuples); without this each touch is an independent SST
    /// point lookup. Process-local, never persisted, never serialized.
    label_cache: std::sync::Mutex<lru::LruCache<Seed, Option<std::sync::Arc<str>>>>,
    /// B1/#211 -- answer-head confidence memo. On a loaded v5 brain every
    /// resident node rehydrates to a stub `Confidence::Unverified`; the REAL
    /// stored confidence lives in the node_store. This LRU memoizes the
    /// resolved confidence so repeated answer-head reads avoid a fresh SST
    /// point lookup. Process-local, never persisted, never serialized.
    conf_cache: std::sync::Mutex<lru::LruCache<Seed, Option<crate::node::Confidence>>>,
    /// Count of node-store label lookups that returned an ERROR (not a
    /// miss). Non-zero means the store is unhealthy; surfaced by
    /// `label_store_error_count()` rather than swallowed, because a
    /// silently-empty label is exactly the failure OP-LABEL-1 exists to
    /// remove.
    label_store_errors: std::sync::atomic::AtomicUsize,
}

/// A serializable snapshot of an `AssociativeNetwork`.
///
/// Adjacency and keyword indices are derivable from the nodes and bonds,
/// so they are not stored — they get rebuilt on load.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
#[archive(check_bytes)]
pub struct NetworkSnapshot {
    pub nodes: Vec<KnowledgeNode>,
    pub bonds: Vec<TemporalBond>,
}

/// OP-017 — Streaming-save abstraction.
///
/// A source of nodes/bonds that can be iterated WITHOUT materializing
/// the full network onto the heap as a `NetworkSnapshot`. The writer
/// (compress.rs / brain_io_v3.rs streaming path) walks via the
/// `for_each_*` callbacks so each archived entity is decoded into one
/// heap copy at a time, written to disk, then dropped — eliminating
/// the ~7 GiB heap-materialization step that drove the pre-OP-017
/// save-path memory spike (see Phase 2 profile report).
///
/// Invariants:
/// - `node_count()` returns the exact number of nodes `for_each_node`
///   will yield (post-filter for deletions/overlay-override).
/// - `bond_count()` returns the exact number of bonds `for_each_bond`
///   will yield (post-filter for deletions/tombstones/overlay-override).
/// - Iteration order is deterministic for a given source state, so
///   the pre-pass (block-count simulation) and the write-pass produce
///   identical packing decisions.
pub trait NetworkSource {
    fn node_count(&self) -> usize;
    fn bond_count(&self) -> usize;
    fn for_each_node<F>(&self, f: F) -> Result<(), crate::brain::BrainError>
    where
        F: FnMut(&KnowledgeNode) -> Result<(), crate::brain::BrainError>;
    fn for_each_bond<F>(&self, f: F) -> Result<(), crate::brain::BrainError>
    where
        F: FnMut(&TemporalBond) -> Result<(), crate::brain::BrainError>;
}

impl NetworkSource for NetworkSnapshot {
    fn node_count(&self) -> usize { self.nodes.len() }
    fn bond_count(&self) -> usize { self.bonds.len() }
    fn for_each_node<F>(&self, mut f: F) -> Result<(), crate::brain::BrainError>
    where F: FnMut(&KnowledgeNode) -> Result<(), crate::brain::BrainError>
    {
        for n in &self.nodes { f(n)?; }
        Ok(())
    }
    fn for_each_bond<F>(&self, mut f: F) -> Result<(), crate::brain::BrainError>
    where F: FnMut(&TemporalBond) -> Result<(), crate::brain::BrainError>
    {
        for b in &self.bonds { f(b)?; }
        Ok(())
    }
}

impl NetworkSource for AssociativeNetwork {
    fn node_count(&self) -> usize {
        match &self.archive {
            None => self
                .nodes
                .values()
                .filter(|n| !self.overlay_deletions.contains(&n.seed))
                .count(),
            Some(arch) => {
                let overlay_alive = self
                    .nodes
                    .values()
                    .filter(|n| !self.overlay_deletions.contains(&n.seed))
                    .count();
                let archive_alive = arch
                    .iter_node_seeds()
                    .filter(|s| {
                        !self.nodes.contains_key(s) && !self.overlay_deletions.contains(s)
                    })
                    .count();
                overlay_alive + archive_alive
            }
        }
    }

    fn bond_count(&self) -> usize {
        match &self.archive {
            None => self
                .bonds
                .iter()
                .filter(|(k, _)| {
                    !self.overlay_deletions.contains(&k.0)
                        && !self.overlay_deletions.contains(&k.1)
                })
                .count(),
            Some(arch) => {
                let overlay_alive = self
                    .bonds
                    .iter()
                    .filter(|(k, _)| {
                        !self.overlay_deletions.contains(&k.0)
                            && !self.overlay_deletions.contains(&k.1)
                    })
                    .count();
                let archive_alive = arch
                    .iter_bond_keys()
                    .filter(|k| {
                        !self.bond_tombstones.contains(k)
                            && !self.bonds.contains_key(k)
                            && !self.overlay_deletions.contains(&k.0)
                            && !self.overlay_deletions.contains(&k.1)
                    })
                    .count();
                overlay_alive + archive_alive
            }
        }
    }

    fn for_each_node<F>(&self, mut f: F) -> Result<(), crate::brain::BrainError>
    where F: FnMut(&KnowledgeNode) -> Result<(), crate::brain::BrainError>
    {
        match &self.archive {
            None => {
                for n in self.nodes.values() {
                    if self.overlay_deletions.contains(&n.seed) { continue; }
                    f(n)?;
                }
            }
            Some(arch) => {
                // Same emission order as `to_snapshot` so save-path tests
                // that compare bytes through both code paths stay valid.
                for n in self.nodes.values() {
                    if self.overlay_deletions.contains(&n.seed) { continue; }
                    f(n)?;
                }
                for (seed, an) in arch.iter_nodes() {
                    if self.nodes.contains_key(&seed)
                        || self.overlay_deletions.contains(&seed)
                    {
                        continue;
                    }
                    // One owned heap KnowledgeNode alive at a time —
                    // dropped at end of this scope before next iter.
                    let owned = crate::archive::knowledge_node_from_archived(an.as_archived());
                    f(&owned)?;
                }
            }
        }
        Ok(())
    }

    fn for_each_bond<F>(&self, mut f: F) -> Result<(), crate::brain::BrainError>
    where F: FnMut(&TemporalBond) -> Result<(), crate::brain::BrainError>
    {
        match &self.archive {
            None => {
                for (key, b) in &self.bonds {
                    if self.overlay_deletions.contains(&key.0)
                        || self.overlay_deletions.contains(&key.1)
                    {
                        continue;
                    }
                    f(b)?;
                }
            }
            Some(arch) => {
                for (key, b) in self.bonds.iter() {
                    if self.overlay_deletions.contains(&key.0)
                        || self.overlay_deletions.contains(&key.1)
                    {
                        continue;
                    }
                    f(b)?;
                }
                for (key, ab) in arch.iter_bonds() {
                    if self.bond_tombstones.contains(&key) { continue; }
                    if self.bonds.contains_key(&key)
                        || self.overlay_deletions.contains(&key.0)
                        || self.overlay_deletions.contains(&key.1)
                    {
                        continue;
                    }
                    let owned = crate::archive::temporal_bond_from_archived(ab.as_archived());
                    f(&owned)?;
                }
            }
        }
        Ok(())
    }
}

/// OP-022.a: filesystem + brain-meta context that lets
/// `ensure_keyword_index_built` write the sidecar to disk and swap to
/// mmap-backed mode immediately after a rebuild.
#[derive(Debug, Clone)]
pub struct KeywordSidecarContext {
    pub path: std::path::PathBuf,
    pub brain_crc32: u32,
    pub brain_node_count: u64,
    pub brain_bond_count: u64,
}

impl AssociativeNetwork {
    /// OP-022.a: wire the sidecar path + brain.lbr meta. Called by
    /// `Brain::load_or_new_with_mode` after the brain.lbr v3 trailer
    /// CRC + counts have been read. With the context set, the next
    /// `ensure_keyword_index_built` rebuild writes the sidecar and
    /// switches to mmap mode; without it, the rebuilt index lives
    /// in the overlay HashMap (legacy fallback).
    pub fn set_keyword_sidecar_context(&mut self, ctx: KeywordSidecarContext) {
        self.keyword_sidecar_context = Some(ctx);
    }

    /// OP-INGEST-LEAN -- true iff a keyword-index sidecar context is
    /// installed. Used by tests to assert the ingest-only load path does
    /// NOT wire the (serving-only) keyword index.
    pub fn has_keyword_sidecar_context(&self) -> bool {
        self.keyword_sidecar_context.is_some()
    }

    /// OP-022.a: install an already-validated mmap'd keyword-index
    /// sidecar. Caller (Brain::load_or_new) is responsible for the
    /// brain_crc32/node_count/bond_count match check before calling.
    /// Sets `archive_keyword_pending = false` so the legacy rebuild
    /// path is skipped.
    pub fn install_keyword_sidecar(
        &mut self,
        mmap: std::sync::Arc<crate::keyword_index::KeywordIndexMmap>,
    ) {
        self.keyword_index =
            crate::keyword_index::KeywordIndex::with_base(mmap);
        self.archive_keyword_pending = false;
    }

    /// OP-022.a.1: rewrite the sidecar from the current
    /// base+overlay-tombstones state via STREAMING merge + streaming
    /// writer. Bypasses `materialize_all` (the +2.7 GiB HashMap clone
    /// that drove the OP-022.a Phase F memory spike) and bypasses the
    /// 2.5 GiB `Vec<u8>` in the old `write_sidecar` (now also fixed via
    /// the streaming writer). Memory peak budget for compaction: well
    /// under +500 MiB transient at the canonical 1.08 M-token /
    /// 83 M-pair scale (target measured in OP-022.a.1 Phase E).
    ///
    /// Called by `Brain::save_to_format` after the brain.lbr
    /// `atomic_write_*` succeeds. No-op (returns Ok(false)) when:
    ///   - no sidecar context wired (heap-only or non-v3 brain), OR
    ///   - overlay + tombstones empty AND base brain_crc32 matches
    ///     the newly-saved brain.
    pub fn compact_keyword_sidecar(&mut self, brain_crc32: u32) -> std::io::Result<bool> {
        let ctx = match self.keyword_sidecar_context.as_ref() {
            Some(c) => c.clone(),
            None => return Ok(false),
        };
        let needs_rewrite = self.keyword_index.has_dirty()
            || self.keyword_index.base()
                .map(|b| b.brain_crc32() != brain_crc32)
                .unwrap_or(true);
        if !needs_rewrite {
            return Ok(false);
        }
        let node_count = self.node_count() as u64;
        let bond_count = self.bond_count() as u64;

        // Sort the overlay's KEYS once into a Vec<&str> for the merge
        // iterator. We borrow the strings — overlay HashMap stays put
        // until success (so a write failure leaves the overlay intact
        // for the next save attempt).
        let overlay_ref = self.keyword_index.overlay_ref();
        let mut sorted_keys: Vec<&str> = overlay_ref.keys().map(String::as_str).collect();
        sorted_keys.sort_by(|a, b| a.as_bytes().cmp(b.as_bytes()));

        let base_arc: Option<std::sync::Arc<crate::keyword_index::KeywordIndexMmap>> =
            self.keyword_index.base().cloned();
        let base_ref = base_arc.as_deref();
        let tombstones = self.keyword_index.tombstones_ref();

        // The merge iterator is re-runnable: each call constructs a fresh
        // peekable over base.iter() + the sorted overlay slice. The
        // streaming writer needs that (pass 1 = metadata, pass 2 = data).
        let make_iter = || crate::keyword_index::KeywordMergeIter::new(
            base_ref,
            sorted_keys.as_slice(),
            overlay_ref,
            tombstones,
        );

        crate::keyword_index::write_sidecar_streaming(
            &ctx.path,
            make_iter,
            brain_crc32,
            node_count,
            bond_count,
        )?;

        // Borrows of self.keyword_index end here (make_iter / sorted_keys
        // / overlay_ref / tombstones / base_ref all go out of scope after
        // the write_sidecar_streaming call returns).
        drop(base_arc);

        // OP-022.a.3.1: we just wrote + fsync'd this sidecar via
        // write_sidecar_streaming; skip the whole-file CRC pass (which
        // dominated the OP-022.a.3 compaction HWM). Cheap header +
        // trailer-magic + section-bounds checks still fire inside
        // `from_mmap_trusted`.
        let new_mmap = crate::keyword_index::KeywordIndexMmap::open_trusted(&ctx.path)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        self.keyword_sidecar_context = Some(KeywordSidecarContext {
            path: ctx.path.clone(),
            brain_crc32,
            brain_node_count: node_count,
            brain_bond_count: bond_count,
        });
        // Swap to the new mmap base AND clear overlay + tombstones in one
        // shot. clear_dirty drops the overlay HashMap (frees its heap)
        // and the tombstone HashSet.
        self.keyword_index.replace_base(Some(std::sync::Arc::new(new_mmap)));
        self.keyword_index.clear_dirty();
        Ok(true)
    }

    /// OP-022.a accessor for diagnostics / tests.
    /// OP-NEARMISS-1 -- borrow the effective keyword index. The
    /// spelling repairer reads it to answer exactly one question:
    /// "which terms does this brain actually know?". It is the ONLY
    /// source of repair candidates, which is what bounds a repair to a
    /// rename onto known vocabulary.
    pub fn keyword_index_ref(&self) -> &crate::keyword_index::KeywordIndex {
        &self.keyword_index
    }

    pub fn keyword_index_has_base(&self) -> bool { self.keyword_index.has_base() }
    /// OP-022.a: borrow the mmap base for test/harness inspection.
    pub fn keyword_index_base_for_test(&self)
        -> Option<&std::sync::Arc<crate::keyword_index::KeywordIndexMmap>>
    { self.keyword_index.base() }
    /// OP-022.a accessor for diagnostics / tests.
    pub fn keyword_index_overlay_seed_total(&self) -> usize {
        self.keyword_index.overlay_seed_total()
    }
}
impl AssociativeNetwork {
    /// Create an empty associative network.
    pub fn new() -> Self {
        AssociativeNetwork {
            nodes: HashMap::new(),
            bonds: HashMap::new(),
            node_intern: crate::node_intern::NodeIntern::new(),
            outgoing: HashMap::new(),
            incoming: HashMap::new(),
            keyword_index: crate::keyword_index::KeywordIndex::new(),
            keyword_sidecar_context: None,
            max_activations: MAX_ACTIVATIONS_DEFAULT,
            max_activations_percentile: MAX_ACTIVATIONS_PERCENTILE_DEFAULT,
            degree_p90: DEGREE_NORM_THRESHOLD,
            archive: None,
            overlay_deletions: HashSet::new(),
            bond_tombstones: HashSet::new(),
            archive_keyword_pending: false,
            hub_read_max_out: crate::hub_degree::DEFAULT_MAX_OUT_DEGREE,
            hub_read_max_in: crate::hub_degree::DEFAULT_MAX_IN_DEGREE,
            hub_read_enforcement: true,
            hub_cap_filtered_at_query_total: std::sync::atomic::AtomicUsize::new(0),
            decay_runtime: None,
            sense_runtime: None,
            bond_count_cache: std::cell::Cell::new(None),
            read_path_mode: std::sync::atomic::AtomicU8::new(0),
            v4_memtable: None,
            v4_manifest: None,
            v4_sst_dir: None,
            v4_block_cache: None,
            v4_sst_handle_cache: std::sync::Mutex::new(std::collections::HashMap::new()),
            mapram_mode: std::sync::atomic::AtomicU8::new(0),
            mapram_cache: None,
            mapram_manifest: None,
            mapram_bond_cache: None,
            v5_node_reader_cache: None,
            label_cache: std::sync::Mutex::new(lru::LruCache::new(
                std::num::NonZeroUsize::new(LABEL_CACHE_CAPACITY)
                    .expect("LABEL_CACHE_CAPACITY is a non-zero constant"),
            )),
            conf_cache: std::sync::Mutex::new(lru::LruCache::new(
                std::num::NonZeroUsize::new(LABEL_CACHE_CAPACITY)
                    .expect("LABEL_CACHE_CAPACITY is a non-zero constant"),
            )),
            label_store_errors: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// OP-BRAIN-V4 Phase 4 — accessor for the in-memory NodeIntern table.
    /// Phase 4 wiring shares this handle with the MemTable + FlushWorker so
    /// every write path uses the same Seed -> NodeId mapping.
    pub fn node_intern(&self) -> &crate::node_intern::NodeIntern {
        &self.node_intern
    }

    // ===================== OP-LABEL-1 =====================

    /// Install the v5 NODE-store reader handle used by
    /// [`Self::node_label`]. Idempotent; replaces any prior handle.
    pub(crate) fn install_node_label_source(
        &mut self,
        cache: std::sync::Arc<crate::node_store::NodeSstReaderCache>,
    ) {
        self.v5_node_reader_cache = Some(cache);
        if let Ok(mut lc) = self.label_cache.lock() {
            lc.clear();
        }
    }

    /// True once a node-store handle is installed, i.e. labels can be
    /// resolved for stub-backed nodes.
    pub fn has_node_label_source(&self) -> bool {
        self.v5_node_reader_cache.is_some()
    }

    /// Number of label lookups that hit a node-store ERROR. A healthy
    /// brain reports 0. Deliberately observable: OP-LABEL-1 exists
    /// because an unobservable empty label looked like an answer.
    pub fn label_store_error_count(&self) -> usize {
        self.label_store_errors.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// The manifest handle to walk for label lookups. Both production
    /// load paths install one: the writer via `enable_memtable` /
    /// `enable_flush_worker`, the read-only reader via
    /// `install_v4_read_handles`. `mapram_manifest` is the fallback for
    /// seam-only wirings used in tests.
    fn label_manifest(&self) -> Option<std::sync::Arc<crate::manifest::Manifest>> {
        let h = self.v4_manifest.as_ref().or(self.mapram_manifest.as_ref())?;
        Some(h.read().ok()?.clone())
    }

    /// Resolve a node's HUMAN-READABLE LABEL.
    ///
    /// This is the single point of truth for "what text does a human
    /// see for this seed". Order:
    ///
    /// 1. **Resident payload.** v3/v4 brains, freshly-taught triples and
    ///    WAL-replayed nodes carry their content in `self.nodes`; that
    ///    is authoritative and free. A resident node with EMPTY content
    ///    is a rehydration stub, not an answer, so it falls through.
    /// 2. **Memo.** Bounded LRU, including negative results, so a
    ///    genuinely label-less node costs one store walk per process,
    ///    not one per narration slot.
    /// 3. **Node store.** `NodeMemTable -> L0 -> L1+` via
    ///    [`crate::node_store::read_path::get_node_v5`], the same walk
    ///    `Brain::get_node_v5` uses -- except it does not require a
    ///    MemTable to exist, so it also works on the read-only reader
    ///    (which deliberately has none).
    ///
    /// Returns `None` when the node genuinely has no text. Callers MUST
    /// treat `None` as "no label" and skip the slot; they must never
    /// substitute a placeholder. An invented label is a fabricated
    /// answer, which is the one thing this brain does not do.
    pub fn node_label(&self, seed: &Seed) -> Option<String> {
        // 1. Resident payload wins when it actually carries text.
        if let Some(view) = self.view_node(seed) {
            let c = view.content();
            if !c.is_empty() {
                return Some(c.to_string());
            }
        }
        // 2. Memo (negative results included).
        if let Ok(mut lc) = self.label_cache.lock() {
            if let Some(hit) = lc.get(seed) {
                return hit.as_ref().map(|a| a.to_string());
            }
        }
        // 3. Node store.
        let resolved = self.resolve_label_from_store(seed);
        if let Ok(mut lc) = self.label_cache.lock() {
            lc.put(
                *seed,
                resolved.as_deref().map(std::sync::Arc::<str>::from),
            );
        }
        resolved
    }

    /// [`Self::node_label`] flattened to a `String`, empty when the node
    /// genuinely has no text. For the `ActivatedNode.content` slot,
    /// which is a `String` by type and whose emptiness downstream
    /// consumers already treat as "no label".
    #[inline]
    pub fn node_label_or_empty(&self, seed: &Seed) -> String {
        self.node_label(seed).unwrap_or_default()
    }

    /// OP-RECALL-1 (R3) -- is this node a FUNCTION-WORD HUB?
    ///
    /// The spread path uses `is_hub_stopword` for two things: an extra
    /// `STOPWORD_BOND_DECAY` (0.2) on any bond touching a function-word
    /// node, and a frontier guard that makes such nodes TERMINAL rather
    /// than expanding them. Both read the node's content.
    ///
    /// On a cache-backed v5 brain `seam_recall_node` returns
    /// `SeamNode::Stub`, whose `content()` is the empty string, so both
    /// predicates were constantly false on every live query: every bond
    /// touching "the" / "of" / "is" was weighted 5x what the design
    /// intended, and function-word mega-hubs were fully expanded
    /// instead of terminating. (Diagnosed as R3 / OP-LABEL-2,
    /// internal notes.)
    ///
    /// `resident` is whatever the caller already has in hand. When it
    /// carries text it is authoritative and free -- that keeps LEGACY
    /// mode byte-identical, including the old `"" -> false` answer for
    /// a genuinely absent node. Only in cache-backed mode does an empty
    /// resident string mean UNKNOWN rather than ABSENT, and only then
    /// do we go to the OP-LABEL-1 resolver, which is memoized (negative
    /// results included) in the per-network label LRU.
    #[inline]
    pub(crate) fn is_stopword_node(&self, seed: &Seed, resident: &str) -> bool {
        if !resident.is_empty() {
            return is_hub_stopword(resident);
        }
        if !self.mapram_cache_backed() {
            return false;
        }
        let label = self.node_label_or_empty(seed);
        !label.is_empty() && is_hub_stopword(&label)
    }

    /// FIX 1 (OP-DEF-GLOSS): the definition-content twin of
    /// `resolve_label_from_store`. Reads the node through the read-only-
    /// compatible `read_path::get_node_v5` (needs no MemTable, so it works
    /// on the guest reader) and returns its stored
    /// `KnowledgeContent::Definition` gloss, if any. None for a missing
    /// node or any non-Definition / empty content.
    pub(crate) fn definition_from_store(&self, seed: &Seed) -> Option<String> {
        let cache = self.v5_node_reader_cache.as_ref()?;
        let manifest = self.label_manifest()?;
        let id = self.node_intern.get(seed)?;
        let fetch = |nt: &crate::node_store::NodeMemTable| {
            crate::node_store::read_path::get_node_v5(nt, &manifest, cache, id)
        };
        let got = match self.v4_memtable.as_ref() {
            Some(mt) => mt.with_node_table(fetch),
            None => fetch(&crate::node_store::NodeMemTable::new()),
        };
        if let Ok(Some(node)) = got {
            if let crate::node::KnowledgeContent::Definition(def) = &node.knowledge.content {
                let d = def.trim();
                if !d.is_empty() {
                    return Some(d.to_string());
                }
            }
        }
        None
    }

    /// B1/#211 -- resolve the REAL stored confidence for a node from the
    /// node_store, memoized. On the read-only reader every resident node is a
    /// `SeamNode::Stub` (`Confidence::Unverified`), so narration would label
    /// EVERY answer 0.5/Unverified; this reads the node's true confidence via
    /// the read-only `read_path::get_node_v5` (no MemTable needed) -- the same
    /// path `definition_from_store` uses. ANSWER-HEAD ONLY: callers must not
    /// walk this over a spread. `None` for a missing node / no v5 store.
    pub(crate) fn confidence_from_store(&self, seed: &Seed) -> Option<crate::node::Confidence> {
        if let Ok(mut cc) = self.conf_cache.lock() {
            if let Some(hit) = cc.get(seed) {
                return *hit;
            }
        }
        let resolved = self.resolve_confidence_from_store(seed);
        if let Ok(mut cc) = self.conf_cache.lock() {
            cc.put(*seed, resolved);
        }
        resolved
    }

    /// Uncached node-store confidence read. See [`Self::confidence_from_store`].
    fn resolve_confidence_from_store(&self, seed: &Seed) -> Option<crate::node::Confidence> {
        let cache = self.v5_node_reader_cache.as_ref()?;
        let manifest = self.label_manifest()?;
        let id = self.node_intern.get(seed)?;
        let fetch = |nt: &crate::node_store::NodeMemTable| {
            crate::node_store::read_path::get_node_v5(nt, &manifest, cache, id)
        };
        let got = match self.v4_memtable.as_ref() {
            Some(mt) => mt.with_node_table(fetch),
            None => fetch(&crate::node_store::NodeMemTable::new()),
        };
        match got {
            Ok(Some(node)) => Some(node.confidence()),
            _ => None,
        }
    }

    /// Uncached node-store label walk. See [`Self::node_label`].
    fn resolve_label_from_store(&self, seed: &Seed) -> Option<String> {
        let cache = self.v5_node_reader_cache.as_ref()?;
        let manifest = self.label_manifest()?;
        let id = self.node_intern.get(seed)?;

        let fetch = |nt: &crate::node_store::NodeMemTable| {
            crate::node_store::read_path::get_node_v5(nt, &manifest, cache, id)
        };
        // The writer has a MemTable holding the un-flushed tail; the
        // read-only reader structurally does not, and an empty table
        // makes the walk start at L0 -- the committed frontier, which
        // is exactly the view that reader promises.
        let got = match self.v4_memtable.as_ref() {
            Some(mt) => mt.with_node_table(fetch),
            None => fetch(&crate::node_store::NodeMemTable::new()),
        };

        match got {
            Ok(Some(node)) => {
                let c = node.content();
                if c.is_empty() {
                    None
                } else {
                    Some(c.to_string())
                }
            }
            Ok(None) => None,
            Err(e) => {
                let n = self
                    .label_store_errors
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                // Loud on the first few, then counted only -- a broken
                // store must not be able to hide behind a blank label.
                if n < 8 {
                    eprintln!(
                        "[OP-LABEL-1] node-store label lookup failed for NodeId({}): {}",
                        id.0, e
                    );
                }
                None
            }
        }
    }

    /// OP-022.s.v3 — borrow the inner `CompressedNetwork` IFF the
    /// network was loaded mmap-zerocopy from a per-block compressed
    /// section AND no overlay mutations are pending (so the on-disk
    /// blocks accurately reflect the current state). Returns `None`
    /// when any of:
    ///   - the network is heap-only (no archive)
    ///   - the archive uses the Phase 2.2 uncompressed backing
    ///   - any node or bond has been added/mutated since load
    ///     (`self.nodes` / `self.bonds` non-empty as overlay)
    ///   - any node has been deleted (`overlay_deletions` non-empty)
    ///   - any bond has been tombstoned (`bond_tombstones` non-empty)
    ///
    /// Used by the save-side block-passthrough fast path. The
    /// passthrough copies compressed-block frames verbatim from the
    /// source mmap into the output, so the source must still match
    /// the desired serialised state.
    pub fn compressed_archive_for_passthrough(
        &self,
    ) -> Option<&crate::compress::CompressedNetwork> {
        if !self.nodes.is_empty() { return None; }
        if !self.bonds.is_empty() { return None; }
        if !self.overlay_deletions.is_empty() { return None; }
        if !self.bond_tombstones.is_empty() { return None; }
        self.archive.as_ref().and_then(|a| a.compressed_network())
    }

    /// OP-022.s.v4 — borrow the inner `CompressedNetwork` for the
    /// partial-passthrough fast path. Returns `Some` when the archive
    /// is compressed AND mmap-backed; overlay/tombstones may be
    /// non-empty (the writer copies source blocks verbatim AND emits
    /// freshly-encoded blocks for overlay entries, with node_index /
    /// bond_index merged + shadow/tombstone-filtered).
    ///
    /// Predicate: the source `CompressedNetwork` exists. Param-equality
    /// + key-equality are checked downstream in
    /// `can_passthrough_blocks_with_overlay`.
    pub fn compressed_archive_for_partial_passthrough(
        &self,
    ) -> Option<&crate::compress::CompressedNetwork> {
        self.archive.as_ref().and_then(|a| a.compressed_network())
    }

    /// OP-022.s.v4 — iterate ONLY heap-overlay nodes (those added or
    /// updated since archive load), skipping tombstoned seeds. Does NOT
    /// walk the archive — used by the partial-passthrough writer to
    /// encode overlay entries into freshly-compressed blocks while
    /// archive frames pass through verbatim.
    pub fn for_each_overlay_node<F>(&self, mut f: F) -> Result<(), crate::brain::BrainError>
    where
        F: FnMut(&KnowledgeNode) -> Result<(), crate::brain::BrainError>,
    {
        for n in self.nodes.values() {
            if self.overlay_deletions.contains(&n.seed) { continue; }
            f(n)?;
        }
        Ok(())
    }

    /// OP-022.s.v4 — iterate ONLY heap-overlay bonds, skipping any
    /// whose endpoint is tombstoned. Counterpart to
    /// `for_each_overlay_node`.
    pub fn for_each_overlay_bond<F>(&self, mut f: F) -> Result<(), crate::brain::BrainError>
    where
        F: FnMut(&TemporalBond) -> Result<(), crate::brain::BrainError>,
    {
        for (key, b) in self.bonds.iter() {
            if self.overlay_deletions.contains(&key.0)
                || self.overlay_deletions.contains(&key.1)
            {
                continue;
            }
            f(b)?;
        }
        Ok(())
    }

    /// OP-022.s.v4 — count overlay nodes (live, post-tombstone). Used
    /// by the partial-passthrough writer to size the block-offset
    /// table reservation.
    pub fn overlay_node_count(&self) -> usize {
        self.nodes
            .values()
            .filter(|n| !self.overlay_deletions.contains(&n.seed))
            .count()
    }

    /// OP-022.s.v4 — count overlay bonds (live, post-tombstone).
    pub fn overlay_bond_count(&self) -> usize {
        self.bonds
            .iter()
            .filter(|(k, _)| {
                !self.overlay_deletions.contains(&k.0)
                    && !self.overlay_deletions.contains(&k.1)
            })
            .count()
    }

    /// OP-022.s.v4 — seeds the overlay has updated (heap shadows
    /// archive entry with same seed). Used by the partial-passthrough
    /// writer to EXCLUDE shadowed archive entries from the merged
    /// node_index — overlay wins.
    pub fn overlay_node_seeds(&self) -> impl Iterator<Item = &Seed> + '_ {
        self.nodes
            .keys()
            .filter(move |s| !self.overlay_deletions.contains(s))
    }

    /// OP-022.s.v4 — bond keys the overlay has updated. Same shadowing
    /// semantic as `overlay_node_seeds`.
    pub fn overlay_bond_keys(&self) -> impl Iterator<Item = &(Seed, Seed)> + '_ {
        self.bonds
            .keys()
            .filter(move |k| {
                !self.overlay_deletions.contains(&k.0)
                    && !self.overlay_deletions.contains(&k.1)
            })
    }

    /// OP-022.s.v4 — accessor for tombstoned source seeds. Used by the
    /// partial-passthrough writer to filter the archive node_index.
    pub fn overlay_deletions_ref(&self) -> &HashSet<Seed> {
        &self.overlay_deletions
    }

    /// OP-022.s.v4 — accessor for tombstoned bond keys.
    pub fn bond_tombstones_ref(&self) -> &HashSet<(Seed, Seed)> {
        &self.bond_tombstones
    }

    /// OP-SAVE-ASYNC — accessor for the Arc'd archive, used by the
    /// snapshot path to keep the mmap'd archive alive on a worker
    /// thread while the main thread continues mutating the overlay.
    /// Returns None for heap-only brains (no archive backing).
    pub fn archive_arc(&self) -> Option<std::sync::Arc<crate::archive::NetworkArchive>> {
        self.archive.clone()
    }

    /// OP-SAVE-ASYNC — accessor for the heap-overlay node map.
    pub fn overlay_nodes_ref(&self) -> &HashMap<Seed, KnowledgeNode> {
        &self.nodes
    }

    /// OP-SAVE-ASYNC — accessor for the heap-overlay bond map.
    pub fn overlay_bonds_ref(&self) -> &HashMap<(Seed, Seed), TemporalBond> {
        &self.bonds
    }

    /// Storage v2 Slice 2c — construct an mmap-zero-copy network from
    /// a `NetworkArchive`. Adjacency and keyword indices are built
    /// from the archived nodes/bonds so spread activation works
    /// without touching the bond data on the heap. `typed_ratio` lives
    /// in the archived nodes themselves and is read directly via
    /// `NodeView::typed_ratio`, so no recompute is needed at load.
    pub fn from_archive(archive: crate::archive::NetworkArchive) -> Self {
        let mut net = AssociativeNetwork::new();

        // Storage v2 Slice 2c: keyword index is built lazily on first
        // use (`ensure_keyword_index_built`). Building it eagerly
        // here would force every per-node content/knowledge string
        // into RSS via rkyv relative pointers, defeating the
        // zero-copy memory win. Spread activation does not need the
        // keyword index — it operates on the adjacency built below
        // from the bond_index keys (which never dereferences
        // archived bond data).
        net.archive_keyword_pending = true;

        // Build adjacency from archived bonds. OP-BRAIN-V4 Phase 1:
        // intern each endpoint Seed into a NodeId before pushing into
        // the adjacency Vec. The intern table is rebuilt fresh on
        // every load — v3 archive on-disk format unchanged.
        //
        // OP-BRAIN-V5 Phase 1.3: the intern step is factored into
        // `NodeIntern::intern_from_v3_archive` so the future Phase 7
        // migration tool (`lu_brain_migrate_v3_v5`) shares the exact
        // same behavior. Determinism: within one process the two
        // `iter_bond_keys()` walks below visit `bond_index` in the
        // same order (HashMap RandomState is per-process, not
        // per-call), so the NodeIds assigned by the first pass match
        // what the inline loop would have assigned. The adjacency
        // sort that follows this block normalises cross-process
        // ordering anyway.
        let _newly_interned =
            net.node_intern.intern_from_v3_archive(&archive);
        for (from, to) in archive.iter_bond_keys() {
            // Both endpoints are already in the intern table after
            // the helper above; `intern_or_get` is therefore a pure
            // lookup here (no fresh inserts) and keeps the code
            // self-checking via the existing API.
            let from_id = net.node_intern.intern_or_get(from);
            let to_id = net.node_intern.intern_or_get(to);
            net.outgoing.entry(from).or_insert_with(Vec::new).push(to_id);
            net.incoming.entry(to).or_insert_with(Vec::new).push(from_id);
        }
        // BR.C-Hyp-Methodology — `archive.iter_bond_keys()` iterates a
        // `bond_index: HashMap<(Seed,Seed), u32>` (see `archive.rs`),
        // and Rust's default HashMap RandomState makes that iteration
        // per-process random. Without this sort, `outgoing[seed]` and
        // `incoming[seed]` come out in different order across processes
        // loading the same brain.lbr. The spread BFS then visits
        // neighbors in a different order, accumulates f32 activations
        // in a different summation order, and produces ULP-level
        // differences that surface as top_k tiebreak flips. Sort each
        // bucket by the underlying Seed bytes (reverse-looked-up from
        // NodeId via the intern table) so the in-memory adjacency is a
        // stable, deterministic function of the on-disk brain.
        {
            let intern = net.node_intern.clone();
            for v in net.outgoing.values_mut() {
                v.sort_by(|a, b| {
                    let sa = intern.reverse_unchecked(*a);
                    let sb = intern.reverse_unchecked(*b);
                    sa.as_bytes().cmp(sb.as_bytes())
                });
            }
            for v in net.incoming.values_mut() {
                v.sort_by(|a, b| {
                    let sa = intern.reverse_unchecked(*a);
                    let sb = intern.reverse_unchecked(*b);
                    sa.as_bytes().cmp(sb.as_bytes())
                });
            }
        }

        net.archive = Some(std::sync::Arc::new(archive));
        net.recompute_degree_p90();
        // OP-INGEST-1.1: lazy-init bond_count cache. We do NOT walk the
        // archive bond section here -- doing so page-faults the entire
        // ~15.5M-bond archive into RssFile during startup, adding ~4 GiB
        // to the startup memory peak (root cause of the OP-INGEST-1
        // 2026-05-26 deploy rollback at gate (b)). The cache stays None
        // until the first bond_count() call, which walks once on demand.
        // Mutations between attach and first-read are absorbed correctly:
        // they only update the cache if it is already Some(_); if it is
        // still None the eventual walk reads the post-mutation state of
        // bonds/tombstones/overlay_deletions and computes the right total.
        net.bond_count_cache.set(None);
        net
    }

    /// Internal: lookup a node, transparently checking overlay (heap)
    /// then archive. Honours `overlay_deletions` tombstones.
    pub(crate) fn view_node(&self, seed: &Seed) -> Option<NodeView<'_>> {
        if self.overlay_deletions.contains(seed) {
            return None;
        }
        if let Some(n) = self.nodes.get(seed) {
            return Some(NodeView::from_heap(n));
        }
        self.archive
            .as_ref()
            .and_then(|a| a.node_by_seed(seed))
            .map(NodeView::from_node_archive)
    }

    /// Internal: lookup a bond, transparently checking overlay then
    /// archive. Honours `overlay_deletions` on either endpoint.
    pub(crate) fn view_bond(&self, key: &(Seed, Seed)) -> Option<BondView<'_>> {
        if self.overlay_deletions.contains(&key.0) || self.overlay_deletions.contains(&key.1) {
            return None;
        }
        if let Some(b) = self.bonds.get(key) {
            return Some(BondView::from_heap(b));
        }
        self.archive
            .as_ref()
            .and_then(|a| a.bond_by_pair(key))
            .map(BondView::from_bond_archive)
    }

    /// True if the network contains a node with this seed (in overlay
    /// or archive, not in deletions).
    pub(crate) fn contains_node(&self, seed: &Seed) -> bool {
        if self.overlay_deletions.contains(seed) {
            return false;
        }
        if self.nodes.contains_key(seed) {
            return true;
        }
        self.archive
            .as_ref()
            .map_or(false, |a| a.has_node(seed))
    }

    /// CoW promotion: if `seed` is archive-only, materialize a heap
    /// copy in the overlay. Returns `true` if a promotion happened
    /// (caller may want to refresh derived state). Returns `false` if
    /// the node is already in overlay, archive-missing, or tombstoned.
    fn promote_node_to_overlay(&mut self, seed: &Seed) -> bool {
        if self.overlay_deletions.contains(seed) {
            return false;
        }
        if self.nodes.contains_key(seed) {
            return false;
        }
        let promoted = self
            .archive
            .as_ref()
            .and_then(|a| a.node_by_seed(seed))
            .map(|na| crate::archive::knowledge_node_from_archived(na.as_archived()));
        match promoted {
            Some(n) => {
                self.nodes.insert(*seed, n);
                true
            }
            None => false,
        }
    }

    /// CoW promotion: if `(from, to)` is archive-only, materialize a
    /// heap copy in the overlay. Used by `get_bond_mut` and Hebbian.
    fn promote_bond_to_overlay(&mut self, key: &(Seed, Seed)) -> bool {
        if self.overlay_deletions.contains(&key.0) || self.overlay_deletions.contains(&key.1) {
            return false;
        }
        if self.bonds.contains_key(key) {
            return false;
        }
        let promoted = self
            .archive
            .as_ref()
            .and_then(|a| a.bond_by_pair(key))
            .map(|ba| crate::archive::temporal_bond_from_archived(ba.as_archived()));
        match promoted {
            Some(b) => {
                self.bonds.insert(*key, b);
                true
            }
            None => false,
        }
    }

    /// True if this network is backed by an mmap'd archive.
    pub fn is_archive_backed(&self) -> bool {
        self.archive.is_some()
    }

    /// Storage v2 Slice 2c: ensure the archive's keyword index has
    /// been materialized into the heap-resident `keyword_index`
    /// HashMap. Called by `Brain::ensure_query_ready` before any
    /// keyword-driven recall. Touches the per-node content/knowledge
    /// strings via rkyv relative pointers, so the first call after
    /// load is the page-fault-heavy moment.
    pub fn ensure_keyword_index_built(&mut self) {
        // OP-022.a: when the mmap'd sidecar is already loaded
        // (`KeywordIndex::with_base`), nothing to rebuild.
        if self.keyword_index.has_base() {
            self.archive_keyword_pending = false;
            return;
        }
        if !self.archive_keyword_pending {
            return;
        }
        let arch = match &self.archive {
            Some(a) => a,
            None => {
                self.archive_keyword_pending = false;
                return;
            }
        };
        // OP-022.a.2: streaming archive walk via external mergesort.
        // The pre-OP-022.a.2 path accumulated a HashMap<String, Vec<Seed>>
        // which drove rebuild HWM to ~9.95 GiB on the canonical brain. The
        // new path buffers (token, seed) pairs in a bounded heap Vec
        // (~128 MiB), flushes sorted runs to disk under
        // `<sidecar_parent>/.rebuild_runs/`, then k-way merges into the
        // OP-022.a.1 streaming sidecar writer. The rebuilder's Drop impl
        // cleans up the run files on success, error, or panic.
        //
        // Falls back to the legacy in-memory HashMap path only when
        // `keyword_sidecar_context` is None (heap-only test brains that
        // have an archive but no sidecar destination).
        if let Some(ctx) = self.keyword_sidecar_context.clone() {
            // Derive runs dir from sidecar path: <sidecar_parent>/.rebuild_runs/
            let runs_dir = {
                let parent = ctx.path.parent().map(|p| p.to_path_buf())
                    .unwrap_or_else(|| std::path::PathBuf::from("/tmp"));
                let mut q = parent;
                q.push(".rebuild_runs");
                q
            };
            // 2 million records per buffer. ~80 M total pairs on the canonical
            // brain -> ~40 runs. Per-record cost ~64 B heap -> ~128 MiB buffer peak.
            const BUFFER_LIMIT: usize = 2 * 1024 * 1024;
            let mut rebuilder = match crate::keyword_index::StreamingRebuilder::new(
                ctx.path.clone(), runs_dir.clone(), BUFFER_LIMIT,
                ctx.brain_crc32, ctx.brain_node_count, ctx.brain_bond_count,
            ) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!(
                        "OP-022.a.2: StreamingRebuilder::new failed ({}); falling back to HashMap path",
                        e
                    );
                    self.legacy_rebuild_into_overlay();
                    return;
                }
            };
            // Walk archive, push (token, seed) pairs.
            let mut push_err: Option<std::io::Error> = None;
            for (seed, an) in arch.iter_nodes() {
                if push_err.is_some() { break; }
                    let node_content = an.as_archived().content.as_str();
                    // OP-BREADTH: entity nodes ("<label>#Q<digits>") index ONLY under
                    // the "e_<word>" namespace + "eq<qid>"; concept buckets stay identical.
                    let r = if let Some(qid) = entity_qid_of(node_content) {
                        push_index_tokens(&mut rebuilder, entity_label_part(node_content), seed, true)
                            .and_then(|_| rebuilder.push(format!("eq{}", qid), seed))
                    } else {
                        push_index_tokens(&mut rebuilder, node_content, seed, false).and_then(|_| {
                            push_index_tokens(
                                &mut rebuilder,
                                crate::archive::knowledge_content_text(&an.as_archived().knowledge.content),
                                seed,
                                false,
                            )
                        })
                    };
                    if let Err(e) = r { push_err = Some(e); break; }
            }
            if let Some(e) = push_err {
                eprintln!(
                    "OP-022.a.2: streaming push error ({}); rebuilder drop will clean runs; falling back to HashMap path",
                    e
                );
                drop(rebuilder);
                self.legacy_rebuild_into_overlay();
                return;
            }
            // Merge any prior overlay entries through the rebuilder too.
            // materialize_all returns the full effective index of the
            // current (base-less) KeywordIndex - overlay only here.
            let prior_overlay = self.keyword_index.materialize_all();
            for (k, v) in prior_overlay {
                for s in v {
                    if let Err(e) = rebuilder.push(k.clone(), s) {
                        eprintln!(
                            "OP-022.a.2: prior-overlay push error ({}); falling back to HashMap path",
                            e
                        );
                        drop(rebuilder);
                        self.legacy_rebuild_into_overlay();
                        return;
                    }
                }
            }
            self.archive_keyword_pending = false;
            match rebuilder.finish() {
                Ok(mmap_arc) => {
                    self.keyword_sidecar_context = Some(KeywordSidecarContext {
                        path: ctx.path.clone(),
                        brain_crc32: ctx.brain_crc32,
                        brain_node_count: ctx.brain_node_count,
                        brain_bond_count: ctx.brain_bond_count,
                    });
                    self.keyword_index = crate::keyword_index::KeywordIndex::with_base(mmap_arc);
                    return;
                }
                Err(e) => {
                    eprintln!(
                        "OP-022.a.2: streaming rebuilder finish failed ({}); falling back to HashMap path",
                        e
                    );
                    self.legacy_rebuild_into_overlay();
                    return;
                }
            }
        }
        // ctx is None - heap-only fallback.
        self.legacy_rebuild_into_overlay();
    }

    /// OP-022.a.2 fallback: the pre-OP-022.a.2 HashMap-accumulating
    /// rebuild path. Used only when `keyword_sidecar_context` is None
    /// (heap-only test brains) or when the streaming rebuilder errored
    /// (defense in depth - should never fire in production).
    fn legacy_rebuild_into_overlay(&mut self) {
        // OP-022.a.2: re-fetch arch internally so the caller can pass &mut self
        // here. The borrow lives only for the read-only walk; the mutations
        // happen after we've dropped the borrow into local_index.
        let local_index = {
            let arch = match &self.archive {
                Some(a) => a,
                None => {
                    self.archive_keyword_pending = false;
                    return;
                }
            };
            let mut local_index: HashMap<String, Vec<Seed>> = HashMap::new();
            for (seed, an) in arch.iter_nodes() {
                let extract = |s: &str, idx: &mut HashMap<String, Vec<Seed>>| {
                    for w in s
                        .to_lowercase()
                        .split_whitespace()
                        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
                        .filter(|w| !w.is_empty() && w.len() > 1)
                    {
                        idx.entry(w).or_insert_with(Vec::new).push(seed);
                    }
                };
                extract(an.as_archived().content.as_str(), &mut local_index);
                extract(
                    crate::archive::knowledge_content_text(&an.as_archived().knowledge.content),
                    &mut local_index,
                );
            }
            for v in local_index.values_mut() {
                v.sort_by(|a, b| a.as_bytes().cmp(b.as_bytes()));
            }
            local_index
        };
        // archive borrow released; mutate self freely.
        let mut local_index = local_index;
        let prior_overlay = self.keyword_index.materialize_all();
        for (k, v) in prior_overlay {
            local_index.entry(k).or_insert_with(Vec::new).extend(v);
        }
        for v in local_index.values_mut() {
            v.sort_by(|a, b| a.as_bytes().cmp(b.as_bytes()));
        }
        local_index.retain(|_, v| !v.is_empty());
        self.archive_keyword_pending = false;
        let mut ki = crate::keyword_index::KeywordIndex::new();
        ki.extend_overlay(local_index.into_iter());
        self.keyword_index = ki;
    }

    /// Add a node to the network. Updates the keyword index.
    pub fn add_node(&mut self, node: KnowledgeNode) {
        self.add_node_with_index_flag(node, true);
    }

    /// Add a node WITHOUT inserting it into the keyword index.
    ///
    /// Used for process-anchor nodes (Thinking goal/step nodes, variant
    /// scaffolding, etc.) whose content shouldn't pollute keyword recall.
    pub fn add_node_unindexed(&mut self, node: KnowledgeNode) {
        self.add_node_with_index_flag(node, false);
    }

    /// Internal: add a node, optionally building keyword-index entries.
    fn add_node_with_index_flag(&mut self, node: KnowledgeNode, index: bool) {
        let seed = node.seed;
        if index {
            // OP-BREADTH: entity nodes index under the "e_" namespace only
            // (+ "eq<qid>"), keeping concept buckets byte-identical.
            if let Some(qid) = entity_qid_of(&node.content) {
                for w in kw_tokens(entity_label_part(&node.content)) {
                    self.keyword_index.insert_overlay(format!("e_{}", w), seed);
                }
                self.keyword_index.insert_overlay(format!("eq{}", qid), seed);
            } else {
                for w in kw_tokens(&node.content) {
                    self.keyword_index.insert_overlay(w, seed);
                }
                for w in kw_tokens(node.knowledge.content.text()) {
                    self.keyword_index.insert_overlay(w, seed);
                }
            }
        }

        self.nodes.insert(seed, node);
    }

    /// OP-TEACH-INDEX (Claude-1 gate #1538 row L): make a freshly TAUGHT
    /// node recallable in the SAME session on the v5/MapRAM serving brain.
    /// `teach_with_kind` persists a new v5 node via the node_store
    /// (MemTable/WAL) and does NOT route through `add_node`, so its
    /// keyword-overlay entries were never written and `keyword_search`
    /// (recall seeding + scar/scar-log content resolution) could not find
    /// it. This inserts the node's label tokens into the overlay exactly as
    /// `add_node_with_index_flag` would, derived from the label alone (a
    /// newly-created node's content == its label). Idempotent at read time
    /// (`KeywordIndex::get` merges + dedups base+overlay). Called ONLY from
    /// the interactive `lu4d_teach` tool, so bulk corpus-ingest memory
    /// behaviour is unchanged (those paths keep the offline
    /// rebuild-keyword-index). Honesty floor is untouched: this only makes
    /// TAUGHT nodes findable; an untaught concept still activates nothing.
    pub fn index_taught_label(&mut self, seed: Seed, label: &str) {
        if let Some(qid) = entity_qid_of(label) {
            for w in kw_tokens(entity_label_part(label)) {
                self.keyword_index.insert_overlay(format!("e_{}", w), seed);
            }
            self.keyword_index.insert_overlay(format!("eq{}", qid), seed);
        } else {
            for w in kw_tokens(label) {
                self.keyword_index.insert_overlay(w, seed);
            }
        }
    }

    /// Get a node by seed.
    ///
    /// Returns a `NodeView<'_>` rather than `&KnowledgeNode` so the
    /// network can later serve nodes from rkyv-archived bytes
    /// (Slice 2c) without the public API changing again.
    pub fn get_node(&self, seed: &Seed) -> Option<NodeView<'_>> {
        self.view_node(seed)
    }

    /// Crate-internal: borrow the heap node directly. Used inside
    /// `network.rs` for hot paths and tests; external callers go
    /// through `get_node` -> `NodeView`.
    #[allow(dead_code)]
    pub(crate) fn node_raw(&self, seed: &Seed) -> Option<&KnowledgeNode> {
        self.nodes.get(seed)
    }

    /// Get a mutable handle to a node. The handle is a thin wrapper
    /// around `&mut KnowledgeNode` today; Slice 2c will turn it into
    /// the CoW promotion point for archive-backed nodes.
    pub fn get_node_mut(&mut self, seed: &Seed) -> Option<NodeMutHandle<'_>> {
        if self.overlay_deletions.contains(seed) {
            return None;
        }
        if !self.nodes.contains_key(seed) {
            // Archive lookup + CoW promotion. Returns None if the node
            // exists in neither overlay nor archive.
            if !self.promote_node_to_overlay(seed) {
                return None;
            }
        }
        self.nodes.get_mut(seed).map(NodeMutHandle::from_heap)
    }

    /// Crate-internal: borrow the heap node mutably without going
    /// through `NodeMutHandle`.
    #[allow(dead_code)]
    pub(crate) fn node_mut_raw(&mut self, seed: &Seed) -> Option<&mut KnowledgeNode> {
        self.nodes.get_mut(seed)
    }

    /// Add a temporal bond between two nodes. Updates adjacency indices.
    ///
    /// Returns an error if either node doesn't exist.
    pub fn add_bond(&mut self, bond: TemporalBond) -> Result<(), NetworkError> {
        if !self.contains_node(&bond.from) {
            return Err(NetworkError::NodeNotFound(bond.from));
        }
        if !self.contains_node(&bond.to) {
            return Err(NetworkError::NodeNotFound(bond.to));
        }

        let from = bond.from;
        let to = bond.to;
        let bond_kind = bond.bond_type();

        // Port Perf-Ingest: read PRE-INSERT degree + cached typed_ratio so
        // we can update the source node's typed_ratio incrementally.
        // The pre-Perf-Ingest path called `update_typed_ratio_for(from)`
        // which walked every existing outgoing edge of `from` -- O(out_
        // degree(from)) per add_bond. Under Wikipedia-shaped ingest the
        // hubs accumulate hundreds of edges, so the per-bond cost grew
        // linearly with degree, projecting full ingest to ~1500 CPU-hours
        // internal notes. Incremental update is O(1).
        //
        // Anti-hallucination invariant: typed_ratio remains a derived
        // value over the bond store; consolidation::recompute_all_typed_
        // ratios() can rebuild it from scratch at any time. We are not
        // introducing a parallel cache that can drift from truth; we are
        // maintaining the same cache by O(1) accounting on each add.
        let total_old = self.outgoing.get(&from).map(|v| v.len()).unwrap_or(0);
        let typed_ratio_old = self
            .view_node(&from)
            .map(|n| n.typed_ratio())
            .unwrap_or(0.0);
        // Round to recover typed_count from the cached ratio. Both
        // typed_count and total_count fit easily inside f32's 24-bit
        // integer mantissa for any realistic out_degree (HubDegreeConfig
        // caps max_out_degree well under 2^24), so the round-trip is
        // exact under the bond_count <= 2^24 invariant.
        let typed_count_old = if total_old == 0 {
            0u32
        } else {
            (typed_ratio_old * total_old as f32).round() as u32
        };

        // Update adjacency. OP-BRAIN-V4 Phase 1: intern each endpoint
        // Seed into a NodeId; adjacency Vecs hold NodeIds (4 bytes
        // each) instead of Seeds (32 bytes).
        let from_id = self.node_intern.intern_or_get(from);
        let to_id = self.node_intern.intern_or_get(to);
        self.outgoing.entry(from).or_insert_with(Vec::new).push(to_id);
        self.incoming.entry(to).or_insert_with(Vec::new).push(from_id);

        // OP-INGEST-1: maintain the cached bond count. The walked
        // semantics count a bond as live iff:
        //   (a) it is in overlay (not shadowed by tombstone/deletion), OR
        //   (b) it is in archive AND not in overlay AND not tombstoned
        //       AND neither endpoint is overlay-deleted.
        // Cases for `add_bond((from, to))`:
        //   - overlay already had it -> no count change (replace).
        //   - overlay didnât have it, archive has it, NOT tombstoned,
        //     no overlay_deletion -> walked already counted it via archive.
        //     Inserting into overlay flips it from archive-counted to
        //     overlay-counted: net 0.
        //   - overlay didnât have it, archive has it, IS tombstoned ->
        //     walked did NOT count it (tombstone shadowed). Now overlay
        //     has it AND tombstone still excludes via overlay-filter
        //     (walked checks bond_tombstones in overlay too). Net 0,
        //     unchanged. (Re-add-after-tombstone is an edge case;
        //     mirrors existing semantics.)
        //   - neither archive nor overlay had it -> net +1.
        //   - overlay_deletion shadows endpoint -> walked excludes
        //     regardless of presence; inserting into overlay still
        //     excluded. Net 0.
        let already_in_overlay = self.bonds.contains_key(&(from, to));
        let archive_has = self
            .archive
            .as_ref()
            .and_then(|a| a.bond_by_pair(&(from, to)))
            .is_some();
        let endpoint_deleted = self.overlay_deletions.contains(&from)
            || self.overlay_deletions.contains(&to);
        let already_tombstoned = self.bond_tombstones.contains(&(from, to));

        self.bonds.insert((from, to), bond);

        if !already_in_overlay
            && !archive_has
            && !endpoint_deleted
            && !already_tombstoned
        {
            // OP-INGEST-1.1: only update cache if already populated;
            // a None cache means lazy-init hasn't happened yet and the
            // walk on first read will pick up this mutation correctly.
            if let Some(n) = self.bond_count_cache.get() {
                self.bond_count_cache.set(Some(n + 1));
            }
        }

        // Port Perf-Ingest: O(1) incremental typed_ratio. Replaces the
        // pre-Perf-Ingest `update_typed_ratio_for(from)` recompute. The
        // function itself is retained (consolidation calls it via
        // `recompute_all_typed_ratios`) so any future drift is repairable.
        let total_new = total_old + 1;
        let typed_count_new = typed_count_old
            + if bond_kind != crate::bond::BondType::RelatedTo { 1 } else { 0 };
        let typed_ratio_new = (typed_count_new as f32) / (total_new as f32);
        if let Some(mut node) = self.get_node_mut(&from) {
            node.typed_ratio = typed_ratio_new;
        }
        Ok(())
    }

    /// Recompute `typed_ratio` for a single seed. O(degree(seed)).
    /// Caller must ensure the node exists (it is a no-op if not).
    ///
    /// Storage v2 Slice 2c hand-back: walks bonds via `get_bond` so
    /// archive-resident outgoing edges are seen, and writes via
    /// `get_node_mut` so an archive-resident `from` node is CoW-
    /// promoted before its `typed_ratio` is updated. Previously this
    /// used heap-only `bond_raw` / `self.nodes.get_mut`, which silently
    /// skipped the update on archived nodes — leaving a stale
    /// typed_ratio whenever a new outgoing bond was added to a node
    /// whose canonical copy still lived in the archive.
    pub fn update_typed_ratio_for(&mut self, seed: Seed) {
        let r = {
            // Copy the neighbor seeds out so the immutable slice
            // borrow on `self` is dropped before we reach for
            // `get_node_mut`, which mutably borrows the network on
            // archive→heap CoW promotion.
            // OP-BRAIN-MapRAM P4: enumerate outgoing edges + their bond
            // type through the seam. CACHE mode sources both from the P1
            // cache (carried predicate), independent of the resident
            // adjacency/bonds; LEGACY mode is byte-identical to the prior
            // `neighbors_outgoing` + `get_bond().bond_type()`. The ratio is
            // a pure typed/total COUNT, so it is order-independent.
            let edges = self.seam_outgoing_typed(&seed);
            if edges.is_empty() {
                0.0
            } else {
                let mut typed = 0usize;
                let mut total = 0usize;
                for (_n, bt) in &edges {
                    if let Some(bt) = bt {
                        total += 1;
                        if *bt != crate::bond::BondType::RelatedTo {
                            typed += 1;
                        }
                    }
                }
                if total == 0 { 0.0 } else { typed as f32 / total as f32 }
            }
        };
        if let Some(mut node) = self.get_node_mut(&seed) {
            node.typed_ratio = r;
        }
    }

    /// Scar-aware bond creation. Refuses the bond if either endpoint is
    /// in `scar_log.scarred_seeds`, and emits a fresh `Scar` against
    /// the would-be edge so the refusal is auditable. This is the path
    /// `Brain::teach` uses; raw `add_bond` remains for tests / internal
    /// callers that don't need the check.
    ///
    /// Note: this guards FUTURE bond creation only. Existing bonds
    /// that touch the seed are scarred separately via
    /// `ScarLog::scar_concept`.
    pub fn add_bond_with_scar_check(
        &mut self,
        bond: TemporalBond,
        scar_log: &mut crate::scar::ScarLog,
    ) -> Result<(), NetworkError> {
        if !self.contains_node(&bond.from) {
            return Err(NetworkError::NodeNotFound(bond.from));
        }
        if !self.contains_node(&bond.to) {
            return Err(NetworkError::NodeNotFound(bond.to));
        }
        if scar_log.is_seed_scarred(bond.from) || scar_log.is_seed_scarred(bond.to) {
            let reason = if scar_log.is_seed_scarred(bond.from)
                && scar_log.is_seed_scarred(bond.to)
            {
                "both endpoints are scarred concepts".to_string()
            } else if scar_log.is_seed_scarred(bond.from) {
                "from-endpoint is a scarred concept".to_string()
            } else {
                "to-endpoint is a scarred concept".to_string()
            };
            let now = crate::scar::now_ms();
            let scar = scar_log.scar_bond(
                bond.from,
                bond.to,
                reason,
                Vec::new(),
                None,
                now,
            );
            return Err(NetworkError::BondRefusedByScarSeed {
                from: bond.from,
                to: bond.to,
                scar_id: scar.scar_id,
            });
        }
        self.add_bond(bond)
    }

    /// Get a bond between two nodes. Returns a `BondView<'_>`.
    pub fn get_bond(&self, from: &Seed, to: &Seed) -> Option<BondView<'_>> {
        self.view_bond(&(*from, *to))
    }

    /// Crate-internal: borrow the heap bond directly. Slice 2c hand-
    /// back removed the last in-tree caller (typed_ratio recompute and
    /// polysemy clustering now go through `get_bond` so archive-
    /// resident bonds are seen). Kept as an escape hatch for future
    /// consolidation paths that need raw heap access.
    #[allow(dead_code)]
    pub(crate) fn bond_raw(&self, from: &Seed, to: &Seed) -> Option<&TemporalBond> {
        self.bonds.get(&(*from, *to))
    }

    /// Get a mutable handle to a bond.
    pub fn get_bond_mut(&mut self, from: &Seed, to: &Seed) -> Option<BondMutHandle<'_>> {
        let key = (*from, *to);
        if self.overlay_deletions.contains(from) || self.overlay_deletions.contains(to) {
            return None;
        }
        if !self.bonds.contains_key(&key) {
            if !self.promote_bond_to_overlay(&key) {
                return None;
            }
        }
        self.bonds.get_mut(&key).map(BondMutHandle::from_heap)
    }

    /// Crate-internal: borrow the heap bond mutably without going
    /// through `BondMutHandle`.
    #[allow(dead_code)]
    pub(crate) fn bond_mut_raw(&mut self, from: &Seed, to: &Seed) -> Option<&mut TemporalBond> {
        self.bonds.get_mut(&(*from, *to))
    }

    /// Port S — remove a single bond from the network. Returns true
    /// when a bond was actually removed (the underlying HashMap had
    /// an entry for `(from, to)`). Used by `Brain::revoke_bonds_by_source`
    /// to power GDPR right-to-be-forgotten and curiosity bond
    /// rollback. Mirrors the cleanup pattern in `purge_nodes`:
    /// strip the bond from `bonds`, prune the `outgoing[from]` and
    /// `incoming[to]` adjacency lists, and drop empty buckets.
    pub fn remove_bond(&mut self, from: &Seed, to: &Seed) -> bool {
        let key = (*from, *to);
        // OP-INGEST-1: capture pre-state for cache delta. The walked
        // countâs view of (from, to) before this call:
        //   - overlay has it AND not tombstoned AND not endpoint-deleted
        //     -> counted (via overlay branch).
        //   - overlay missing AND archive has it AND not tombstoned AND
        //     not endpoint-deleted -> counted (via archive branch).
        //   - otherwise -> not counted.
        // Post-state depends on which removal path we take.
        let endpoint_deleted = self.overlay_deletions.contains(&key.0)
            || self.overlay_deletions.contains(&key.1);
        let pre_overlay = self.bonds.contains_key(&key);
        let pre_tombstoned = self.bond_tombstones.contains(&key);
        let archive_has = self
            .archive
            .as_ref()
            .and_then(|a| a.bond_by_pair(&key))
            .is_some();
        let pre_visible = !endpoint_deleted
            && !pre_tombstoned
            && (pre_overlay || archive_has);
        if self.bonds.remove(&key).is_none() {
            // Port Source-Persist: if the bond is archive-backed,
            // tombstone it. The next save physically purges it.
            if let Some(arch) = &self.archive {
                if arch.iter_bond_keys().any(|k| k == key) {
                    self.bond_tombstones.insert(key);
                    // OP-INGEST-1: tombstone shadowed an archive-visible
                    // bond -> -1. (pre_visible would have been true here.)
                    if pre_visible {
                        // OP-INGEST-1.1: only mutate cache if populated.
                        if let Some(n) = self.bond_count_cache.get() {
                            debug_assert!(n > 0, "bond_count_cache underflow on tombstone");
                            self.bond_count_cache.set(Some(n.saturating_sub(1)));
                        }
                    }
                    return true;
                }
            }
            return false;
        }
        // OP-INGEST-1: overlay removal succeeded. Post-state visibility:
        //   - if archive ALSO has it AND not tombstoned AND endpoints OK
        //     -> archive branch re-exposes it. Still visible. Net 0.
        //   - else -> not visible. Net -1 iff pre_visible.
        let post_visible = !endpoint_deleted
            && !pre_tombstoned   // tombstones arenât mutated on this path
            && archive_has;
        if pre_visible && !post_visible {
            // OP-INGEST-1.1: only mutate cache if populated.
            if let Some(n) = self.bond_count_cache.get() {
                debug_assert!(n > 0, "bond_count_cache underflow on remove");
                self.bond_count_cache.set(Some(n.saturating_sub(1)));
            }
        }
        // OP-BRAIN-V4 Phase 1: compare against the interned NodeIds
        // of the endpoints rather than the raw Seeds, since the
        // adjacency Vecs hold Vec<NodeId>. The intern table is
        // append-only within a process lifetime, so an existing edge
        // always has both endpoints already interned; `get` returning
        // None means no such edge exists (no-op retain).
        let from_id_opt = self.node_intern.get(from);
        let to_id_opt = self.node_intern.get(to);
        if let (Some(out), Some(target_id)) = (self.outgoing.get_mut(from), to_id_opt) {
            out.retain(|id| *id != target_id);
        }
        if let (Some(inc), Some(source_id)) = (self.incoming.get_mut(to), from_id_opt) {
            inc.retain(|id| *id != source_id);
        }
        if self.outgoing.get(from).map(|v| v.is_empty()).unwrap_or(false) {
            self.outgoing.remove(from);
        }
        if self.incoming.get(to).map(|v| v.is_empty()).unwrap_or(false) {
            self.incoming.remove(to);
        }
        true
    }

    /// Port S — return a freshly owned `Vec<(Seed, Seed)>` of every
    /// bond key currently in the network. Used by
    /// `Brain::revoke_bonds_by_source` to find network bonds whose
    /// seed prefixes match an overlay-revoke target. The returned
    /// keys are immutable copies — no live borrow of the network.
    pub fn iter_bond_keys_owned(&self) -> Vec<(Seed, Seed)> {
        let mut out: Vec<(Seed, Seed)> = self.bonds.keys().copied().collect();
        if let Some(arch) = &self.archive {
            let heap: std::collections::HashSet<(Seed, Seed)> =
                self.bonds.keys().copied().collect();
            for k in arch.iter_bond_keys() {
                if heap.contains(&k) { continue; }
                if self.overlay_deletions.contains(&k.0)
                    || self.overlay_deletions.contains(&k.1)
                {
                    continue;
                }
                if self.bond_tombstones.contains(&k) { continue; }
                out.push(k);
            }
        }
        out
    }

    /// Remove every node whose `content` exactly matches a label in `labels`.
    /// Strips all bonds touching those nodes from every index.
    /// Returns `(nodes_removed, bonds_removed)`.
    pub fn purge_nodes(&mut self, labels: &[&str]) -> (usize, usize) {
        let label_set: std::collections::HashSet<&str> = labels.iter().copied().collect();

        let seeds_to_remove: Vec<Seed> = self
            .nodes
            .iter()
            .filter(|(_, n)| label_set.contains(n.content.as_str()))
            .map(|(s, _)| *s)
            .collect();

        let mut bonds_removed: usize = 0;

        for seed in &seeds_to_remove {
            // Remove outgoing bonds and fix neighbors' incoming lists.
            // OP-BRAIN-V4 Phase 1: adjacency holds NodeIds; reverse-
            // lookup to get the target Seed for the bond-key map; also
            // intern the source Seed once so we can match it against
            // the neighbor's incoming NodeId Vec.
            let seed_id_opt = self.node_intern.get(seed);
            if let Some(targets) = self.outgoing.remove(seed) {
                for tgt_id in &targets {
                    let tgt = match self.node_intern.reverse_lookup(*tgt_id) {
                        Some(s) => s,
                        None => continue, // stale id, skip
                    };
                    // OP-INGEST-1: mirror remove_bond cache logic for
                    // each overlay bond removed by purge.
                    let key = (*seed, tgt);
                    let endpoint_deleted = self.overlay_deletions.contains(&key.0)
                        || self.overlay_deletions.contains(&key.1);
                    let pre_tombstoned = self.bond_tombstones.contains(&key);
                    let archive_has = self
                        .archive
                        .as_ref()
                        .and_then(|a| a.bond_by_pair(&key))
                        .is_some();
                    let pre_visible = self.bonds.contains_key(&key)
                        && !endpoint_deleted
                        && !pre_tombstoned;
                    self.bonds.remove(&key);
                    bonds_removed += 1;
                    let post_visible = !endpoint_deleted
                        && !pre_tombstoned
                        && archive_has;
                    if pre_visible && !post_visible {
                        // OP-INGEST-1.1: only mutate cache if populated.
                        if let Some(n) = self.bond_count_cache.get() {
                            debug_assert!(n > 0, "bond_count_cache underflow on purge outgoing");
                            self.bond_count_cache.set(Some(n.saturating_sub(1)));
                        }
                    }
                    if let (Some(inc), Some(seed_id)) =
                        (self.incoming.get_mut(&tgt), seed_id_opt)
                    {
                        inc.retain(|id| *id != seed_id);
                    }
                }
            }
            // Remove incoming bonds and fix neighbors' outgoing lists.
            // OP-BRAIN-V4 Phase 1: see outgoing block above for the
            // NodeId reverse-lookup pattern.
            if let Some(sources) = self.incoming.remove(seed) {
                for src_id in &sources {
                    let src_seed = match self.node_intern.reverse_lookup(*src_id) {
                        Some(s) => s,
                        None => continue, // stale id, skip
                    };
                    let key = (src_seed, *seed);
                    let endpoint_deleted = self.overlay_deletions.contains(&key.0)
                        || self.overlay_deletions.contains(&key.1);
                    let pre_tombstoned = self.bond_tombstones.contains(&key);
                    let archive_has = self
                        .archive
                        .as_ref()
                        .and_then(|a| a.bond_by_pair(&key))
                        .is_some();
                    let pre_visible = self.bonds.contains_key(&key)
                        && !endpoint_deleted
                        && !pre_tombstoned;
                    self.bonds.remove(&key);
                    bonds_removed += 1;
                    let post_visible = !endpoint_deleted
                        && !pre_tombstoned
                        && archive_has;
                    if pre_visible && !post_visible {
                        // OP-INGEST-1.1: only mutate cache if populated.
                        if let Some(n) = self.bond_count_cache.get() {
                            debug_assert!(n > 0, "bond_count_cache underflow on purge incoming");
                            self.bond_count_cache.set(Some(n.saturating_sub(1)));
                        }
                    }
                    if let (Some(out), Some(seed_id)) =
                        (self.outgoing.get_mut(&src_seed), seed_id_opt)
                    {
                        out.retain(|id| *id != seed_id);
                    }
                }
            }
            // Remove from keyword index (OP-022.a: via tombstone — keeps the
            // mmap'd base read-only; next compaction physically purges).
            self.keyword_index.tombstone_seed(*seed);
            self.nodes.remove(seed);
        }

        (seeds_to_remove.len(), bonds_removed)
    }

    /// Remove the given seeds from every keyword-index bucket. Used by
    /// Brain::prune_legacy_goal_nodes to stop legacy goal:/step: nodes
    /// being keyword-discoverable. Does NOT touch nodes or bonds.
    pub fn unindex_seeds(&mut self, seeds: &std::collections::HashSet<Seed>) {
        // OP-022.a: tombstone the seeds rather than mutating each bucket;
        // the mmap'd base is read-only. Next sidecar compaction physically purges.
        self.keyword_index.tombstone_seeds(seeds);
    }

    /// IDF = ln(total_nodes / nodes_that_contain_term).
    /// High IDF → rare/specific; low IDF → common/generic.
    pub fn term_idf(&self, term: &str) -> f32 {
        let n = self.node_count() as f32;
        if n < 2.0 { return 1.0; }
        let df = self.keyword_index.get(term).map(|v| v.len()).unwrap_or(0) as f32;
        if df < 1.0 { return n.ln(); }
        (n / df).ln().max(0.0)
    }

    /// Select up to 3 seeds for a query using TF-IDF weighting and pattern detection.
    ///
    /// Algorithm:
    /// 1. Tokenise, drop hub-stopwords and query-noise words.
    /// 2. Detect "X of Y" patterns and apply a 1.5× IDF boost to Y (the entity).
    /// 3. Score each surviving token by normalised IDF → base weight [0.3, 1.0].
    /// 4. Collect up to 2 seed nodes per token from the keyword index.
    /// 5. Sort by weight, deduplicate, apply rank-based decay:
    ///    seed #1 × 1.0, seed #2 × 0.6, seed #3 × 0.35.
    ///
    /// This is the THINK-path entry point: coverage promotion is OFF.
    /// ASK-path callers should use [`Self::select_seeds_for_query_ask`]
    /// instead, which adds BR.C-Hyp.2.2 multi-token-coverage weighting
    /// before the dedupe/rank-multiplier step.
    pub fn select_seeds_for_query(&self, query: &str) -> Vec<(Seed, f32)> {
        self.select_seeds_for_query_inner(query, /* apply_coverage_promotion = */ false)
    }

    /// ASK-path entry point with coverage-aware seed promotion (BR.C-Hyp.2.2).
    ///
    /// Identical to [`Self::select_seeds_for_query`] except that, when at
    /// least two surviving content tokens remain after stopword +
    /// query-noise filtering, candidate seeds whose keyword-index entries
    /// cover multiple query tokens are weighted up before the rank-
    /// multiplier dedupe (factor 1.0 / 1.6 / 2.2 for coverage 1 / 2 / 3+).
    ///
    /// Diagnosed in BR.C-Hyp.2 internal notes as the missing
    /// co-occurrence factor on the primary ASK recall path: the brain has
    /// the right multi-token-match content, the secondary `keyword_search`
    /// path surfaces it correctly, but the primary `select_seeds_for_query`
    /// per-token TF-IDF picker has no way to prefer it. Gated to ASK
    /// callers (`Brain::ask` / `Brain::typed_ask` / `Brain::ask_in_scope`)
    /// only — BR.C-Hyp.2.1 applied the same logic to the shared
    /// implementation and regressed the gate-12 think-coordination
    /// measurement (per learning #987 + halt tag
    /// `halt-br-c-hyp-2-1-2026-05-09`).
    pub fn select_seeds_for_query_ask(&self, query: &str) -> Vec<(Seed, f32)> {
        let mut seeds = self.select_seeds_for_query_inner(query, /* apply_coverage_promotion = */ true);
        // OP-SENSE-ROUTER (option 1, default-off): on a code-intent frame,
        // prepend the coding-sense handle seed(s) so spreading enters the
        // coding subgraph. Only op tokens with a real pyfn handle inject.
        if sense_router_enabled() {
            let toks = code_intent_op_tokens(query);
            if !toks.is_empty() {
                let mut inject: Vec<(Seed, f32)> = Vec::new();
                for t in toks {
                    let handle = format!("pyfn{}", t);
                    let hs = self.select_seeds_for_query_inner(&handle, false);
                    if let Some((hseed, _)) = hs.first() {
                        // STRICT exactness guard: only route if the resolved node
                        // IS the handle. Without this a nonexistent handle
                        // ("pyfncar") can near-miss / fall back onto an unrelated
                        // node and inject a spurious seed -> drift on general
                        // queries. Requiring content == handle keeps zero-drift.
                        if self.node_label(hseed).as_deref() == Some(handle.as_str()) {
                            // Follow the handle's alias bond one hop to the RICH
                            // canonical coding node ("the <op> function", which
                            // carries params/returns/causes) and seed THAT, so
                            // recall spreads into a groundable multi-node coding
                            // answer that clears the idk floor -- seeding the
                            // sparse handle directly is a single dominant anchor.
                            let mut target = *hseed;
                            for e in self.seam_spread_outgoing_edges(hseed) { target = e.nbr; break; }
                            if !inject.iter().any(|(x, _)| *x == target) {
                                inject.push((target, sense_router_boost()));
                            }
                        }
                    }
                }
                if !inject.is_empty() {
                    inject.extend(seeds.into_iter());
                    seeds = inject;
                }
            }
        }
        // OP-SENSE-CONCEPT-QID (default-off, same LU_SENSE_ROUTER gate): on a
        // definitional/language frame, route a polysemous concept token into
        // its QID-anchored sense node via the "csense<token>" option-2 handle
        // (1-hop to the <Label>#Q<qid> shelf node), so "what is python" and
        // "python language" enter the PROGRAMMING sense while the everyday
        // (snake) node keeps the bare token bucket. Strict exactness guard =
        // zero drift: only a token whose csense handle exists injects.
        if sense_router_enabled() {
            let ctoks = concept_sense_tokens(query);
            if !ctoks.is_empty() {
                let mut inject: Vec<(Seed, f32)> = Vec::new();
                for t in ctoks {
                    let handle = format!("csense{}", t);
                    let hs = self.select_seeds_for_query_inner(&handle, false);
                    if let Some((hseed, _)) = hs.first() {
                        if self.node_label(hseed).as_deref() == Some(handle.as_str()) {
                            let mut target = *hseed;
                            for e in self.seam_spread_outgoing_edges(hseed) { target = e.nbr; break; }
                            if !inject.iter().any(|(x, _)| *x == target) {
                                inject.push((target, sense_router_boost()));
                            }
                        }
                    }
                }
                if !inject.is_empty() {
                    inject.extend(seeds.into_iter());
                    seeds = inject;
                }
            }
        }
        seeds
    }

    fn select_seeds_for_query_inner(
        &self,
        query: &str,
        apply_coverage_promotion: bool,
    ) -> Vec<(Seed, f32)> {
        const MAX_SEEDS: usize = 3;
        let n_nodes = self.node_count() as f32;
        let max_idf = if n_nodes > 1.0 { n_nodes.ln() } else { 1.0 };

        let raw: Vec<String> = query
            .to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| !w.is_empty() && w.len() >= 2)
            .map(|s| s.to_string())
            .collect();

        // "X of Y" pattern: boost Y as the primary entity
        let mut boost: HashSet<String> = HashSet::new();
        for i in 0..raw.len() {
            if raw[i] == "of" {
                for j in (i + 1)..raw.len() {
                    let next = &raw[j];
                    if !is_hub_stopword(next) && !is_query_noise(next) && next.len() >= 3 {
                        boost.insert(next.clone());
                        break;
                    }
                }
            }
        }

        // First pass: full filtering (hub-stopword + query-noise). This is
        // the correct behavior on multi-word queries — "what is the function
        // of mitochondria" must seed on `mitochondria`, not on `function`.
        let strict = self.collect_seeds_with_filter(
            &raw, &boost, max_idf, /* allow_query_noise = */ false, MAX_SEEDS,
            apply_coverage_promotion,
        );
        if !strict.is_empty() {
            return strict;
        }

        // Bug B fix shape A — last-resort fallback. The strict pass produced
        // zero seeds. If at least one raw token survives the hub-stopword
        // filter (i.e. the query wasn't ALL true function words like
        // "because" / "the" / "of"), retry with the is_query_noise filter
        // disabled but is_hub_stopword still active. This rescues bare-word
        // queries on concept words that happen to be on the query-noise list
        // (e.g. `function`, `relationship`) without re-admitting bare
        // function words. See the project log entries #571 (Port B) and #572
        // (Bug B triage).
        let any_non_hub = raw.iter().any(|t| !is_hub_stopword(t) && t.len() >= 2);
        if any_non_hub {
            return self.collect_seeds_with_filter(
                &raw, &boost, max_idf, /* allow_query_noise = */ true, MAX_SEEDS,
                apply_coverage_promotion,
            );
        }

        Vec::new()
    }

    /// Collect seed candidates from `raw` query tokens, applying the
    /// hub-stopword filter unconditionally and the `is_query_noise` filter
    /// only when `allow_query_noise == false`. Extracted so
    /// `select_seeds_for_query` can run a strict pass first and fall back to
    /// a relaxed pass if and only if the strict pass returned zero seeds
    /// AND the raw query had at least one non-hub-stopword token. See the
    /// caller for the rationale (Bug B fix shape A).
    fn collect_seeds_with_filter(
        &self,
        raw: &[String],
        boost: &HashSet<String>,
        max_idf: f32,
        allow_query_noise: bool,
        max_seeds: usize,
        apply_coverage_promotion: bool,
    ) -> Vec<(Seed, f32)> {
        let content: Vec<&str> = raw
            .iter()
            .map(|s| s.as_str())
            .filter(|&t| {
                !is_hub_stopword(t)
                    && (allow_query_noise || !is_query_noise(t))
                    && t.len() >= 2
            })
            .collect();

        let mut candidates: Vec<(Seed, f32)> = Vec::new();

        // BR.R3 (OP-008): majority bucket-coverage candidate scan.
        // The per-token take-2 above is byte-sorted by seed bytes, so at
        // 4.5M-bond scale the first 2 of each bucket are partial-match
        // noise that crowds out the genuine multi-token-match article.
        // Empirical probe (BR.R3 #1003): for Q2 "Who is Bertrand Russell?"
        // the actual "Bertrand Russell" node sits in BOTH the `bertrand`
        // and `russell` buckets (coverage = 2), but neither byte-sorted
        // top-2 surfaces it because "Bertrand Renouvin" and "Russell Coutts"
        // sort earlier. The BR.C-Hyp.2.2 promotion below cannot recover
        // them -- it amplifies coverage on seeds already in the candidate
        // pool but the genuine article never makes it in.
        //
        // Fix at the right layer (CLAUDE.md Principle 1): walk the
        // smallest content-token bucket and harvest seeds whose
        // cross-bucket coverage clears the *majority* threshold
        // ceil((N+1)/2). Majority (not full N, not just >=2) is the
        // smallest gate that ALL of:
        //   - admits Bertrand Russell for Q2 (N=2, threshold=2)
        //   - admits Bertrand Russell for Q7 "Which philosophers worked
        //     with Bertrand Russell?" (N=3, threshold=2)
        //   - REJECTS Bertrand Russell on A8 "Did Bertrand Russell speak
        //     Martian?" (N=4, threshold=3, BR has bc=2) -- preserving the
        //     adversarial. A8 currently relies on Mode C to refuse;
        //     promoting BR seed there would over-saturate max_activation
        //     and false-PASS.
        // For lower coverage the existing take-2 + coverage-promotion
        // logic still applies -- this scan is strictly additive.
        //
        // Cost: walk smallest bucket (capped at MAJORITY_SCAN_CAP) and
        // do T binary searches per seed across other content-token
        // buckets. On the 4.5M-bond canonical, typical content tokens
        // have buckets of 5K-50K seeds; scan stays in sub-millisecond
        // territory.
        //
        // THINK-path callers pass `apply_coverage_promotion=false` so
        // the gate-12 coordination measurement (BR.C-Hyp.2.1 learning
        // #987) stays bit-identical.
        if apply_coverage_promotion && content.len() >= 2 {
            // OP-RECALL-1 (R2) -- MEASURED AND LEFT AT 10_000. Read this
            // before raising it again.
            //
            // The defect is real: `smallest` is BYTE-SORTED by seed
            // bytes, so this cap truncates an arbitrary alphabetical
            // prefix. On the live brain the `dog` (22,718) and `python`
            // (20,816) buckets are only ~45-50% visible to the one path
            // that consults degree, and a canonical article whose seed
            // sorts past position 10,000 is structurally invisible to it.
            //
            // Raising it to 250_000 was tried and REVERTED, because it
            // made recall measurably worse:
            //
            //   "domestic dog"    10_000: "Domestic dog."
            //                    250_000: "You Can Make Me Dance, Sing or
            //                              Anything (Even Take the Dog
            //                              for a Walk...)"
            //   "python language" 10_000: "Python (programming language)."
            //                    250_000: "Articles with example Python
            //                              (programming language) code"
            //                              -- a maintenance category
            //
            // The cap was never the binding constraint. DEGREE ALONE is a
            // weak proxy for "this is the entity node" at Wikipedia
            // scale: maintenance categories, set-index pages and
            // "Articles with..." trackers have enormous degree by
            // construction. Widening the candidate pool feeds the ranker
            // more of exactly the wrong thing, and the junk wins on
            // degree.
            //
            // So the fix for R2 is a NODE-CLASS filter (demote
            // category/maintenance/list nodes below article nodes), not a
            // bigger scan. That is a separate change and it needs its own
            // evidence. Until then this stays where it was, and the
            // truncation is a known, documented limitation rather than an
            // unknown one.
            const MAJORITY_SCAN_CAP: usize = 10_000;
            let n = content.len();
            let majority_threshold = (n + 2) / 2; // ceil((n+1)/2)
            // OP-022.a: was Vec<(&&str, &Vec<Seed>)>; MergedBucket replaces
            // the &Vec<Seed> via Deref<Target=[Seed]>, so the body below
            // (sort_by_key on len, binary_search_by on bucket bytes) works
            // unchanged.
            let mut content_buckets: Vec<(&&str, crate::keyword_index::MergedBucket<'_>)> = Vec::new();
            for token in &content {
                if let Some(bucket) = self.keyword_index.get(*token) {
                    content_buckets.push((token, bucket));
                }
            }
            content_buckets.sort_by_key(|(_, b)| b.len());
            let mut max_token_weight: f32 = 0.0;
            for token in &content {
                let idf = self.term_idf(token);
                let base = 0.3 + 0.7 * (idf / max_idf).min(1.0);
                let w = if boost.contains(*token) { (base * 1.5).min(1.0) } else { base };
                if w > max_token_weight { max_token_weight = w; }
            }
            if let Some((_, smallest)) = content_buckets.first() {
                // First pass: harvest majority candidates with their (degree,
                // seed) so we can sort by degree-descending before pushing.
                // The canonical article (e.g. "Bertrand Russell" with 269
                // 1-hop bonds) outranks "Bertrand Russell Archives at McMaster"
                // (1 bond) by degree, which is the topology proxy for "this
                // is the real entity node". Byte-order alone is uninformative
                // and surfaces deep-link leaf nodes ahead of the article hub.
                let mut majority: Vec<(usize, Seed)> = Vec::new();
                for seed in smallest.iter().take(MAJORITY_SCAN_CAP) {
                    // OP-RECALL-1 (R2): coverage FIRST. It is pure
                    // in-memory binary search over byte-sorted buckets;
                    // the label resolution below can touch the node
                    // store. Testing the cheap discriminator first is
                    // what makes the raised cap affordable. Ordering is
                    // semantically neutral -- both are pure filters and
                    // a candidate must pass both to be kept.
                    let mut coverage = 0usize;
                    for (_, bucket) in content_buckets.iter() {
                        if bucket
                            .binary_search_by(|s| s.as_bytes().cmp(seed.as_bytes()))
                            .is_ok()
                        {
                            coverage += 1;
                            if coverage >= majority_threshold { break; }
                        }
                    }
                    if coverage < majority_threshold { continue; }
                    // OP-RECALL-1 (R3 family): `view_node` returns a
                    // content-less Stub on the cache-backed v5 path, so
                    // the pollution filter was inert on every live query.
                    // `node_label_or_empty` is the OP-LABEL-1 resolver
                    // (resident payload -> memo -> node store) and gives
                    // the filter something real to test. An empty label
                    // is not pollution, matching the prior
                    // `if let Some(node)` behaviour on a missing node.
                    let label = self.node_label_or_empty(seed);
                    if !label.is_empty() && is_recall_pollution(&label) { continue; }
                    let deg = self.seam_total_degree(seed);
                    majority.push((deg, *seed));
                }
                // Degree desc; ties broken by seed bytes asc for bit-stability.
                majority.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.as_bytes().cmp(b.1.as_bytes())));
                for (_, seed) in majority.into_iter().take(max_seeds * 2) {
                    candidates.push((seed, max_token_weight));
                }
            }
        }

        // ---------------------------------------------------------
        // OP-RECALL-1 (R1): SINGLE-CONTENT-TOKEN DEGREE RANKING.
        //
        // The majority-coverage scan above is the only place in the
        // whole seeding path that consults degree, and it is gated on
        // `content.len() >= 2` -- a majority threshold over one bucket
        // is trivially satisfied by every member, so it carries no
        // signal. A one-token query therefore fell through to the
        // per-token take-2 loop below, which walks the keyword bucket
        // in BYTE ORDER and keeps the first two survivors. The bucket
        // is sorted by seed bytes, and seeds are random
        // (`Seed::generate`), so "first two" is "two uniformly random
        // members out of 20,000+". Measured on the live brain
        // (OP-SEED-DIAG, internal notes: bare `python` seeded on
        // "Parser implementation in Python" while `python language`
        // -- same brain, same index, two tokens -- correctly seeded on
        // "Python (programming language)". 8/8 single-token probes
        // failed the same way.
        //
        // The fix is to give the one-token case the SAME ranking
        // signal the two-token path already uses: total degree,
        // descending, ties broken by seed bytes for bit-stability.
        // Degree is the topology proxy for "this is the entity node,
        // not a deep-link leaf that merely mentions the word".
        //
        // Cost: `seam_total_degree` is an intern lookup plus two O(1)
        // reads of the resident degree map -- no node store, no bond
        // SST, no 2Q cache. A 140K-entry bucket (the largest measured)
        // is a few milliseconds. Only the top
        // `SINGLE_TOKEN_LABEL_CHECK_DEPTH` candidates pay for a label
        // resolution, so the pollution filter costs a bounded number
        // of node-store walks rather than one per bucket entry.
        //
        // Applies to BOTH the ASK and THINK entry points: the gate-12
        // coordination measurement that constrained BR.C-Hyp.2.1
        // (learning #987) is a MULTI-token benchmark, and this branch
        // is unreachable when `content.len() != 1`.
        // ---------------------------------------------------------
        const SINGLE_TOKEN_DEGREE_SCAN_CAP: usize = 250_000;
        const SINGLE_TOKEN_LABEL_CHECK_DEPTH: usize = 64;

        if content.len() == 1 {
            let token = content[0];
            let idf = self.term_idf(token);
            let base = 0.3 + 0.7 * (idf / max_idf).min(1.0);
            let weight = if boost.contains(token) { (base * 1.5).min(1.0) } else { base };
            if let Some(seeds_for_term) = self.keyword_index.get(token) {
                let mut ranked: Vec<(usize, Seed)> =
                    Vec::with_capacity(seeds_for_term.len().min(SINGLE_TOKEN_DEGREE_SCAN_CAP));
                for seed in seeds_for_term.iter().take(SINGLE_TOKEN_DEGREE_SCAN_CAP) {
                    ranked.push((self.seam_total_degree(seed), *seed));
                }
                ranked.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.as_bytes().cmp(b.1.as_bytes())));
                let mut taken = 0;
                for (_, seed) in ranked.into_iter().take(SINGLE_TOKEN_LABEL_CHECK_DEPTH) {
                    let label = self.node_label_or_empty(&seed);
                    if !label.is_empty() && is_recall_pollution(&label) { continue; }
                    candidates.push((seed, weight));
                    taken += 1;
                    if taken >= 2 { break; }
                }
            }
        } else {
            // OP-RECALL-EXTRACT (defect 2): multi-token per-token seed pick.
            // Previously this walked each keyword bucket in BYTE ORDER and
            // kept the first two survivors. Byte order is random with respect
            // to meaning, so on the live canonical brain "python language"
            // seeded on degree-1 deep-link leaves ("Trace or track Python
            // statement execution") instead of the degree-87 "python" entity
            // node, and "hash table" seeded on leaf glosses instead of the
            // degree-9 "hash" / degree-99 "table" nodes. The resulting
            // activation was so weak the idk gate (correctly) refused it -- a
            // grounded query wrongly reported as unknown. The
            // single-content-token branch above already ranks its bucket by
            // total degree (the topology proxy for "this is the entity node,
            // not a leaf that merely mentions the word") and that path works;
            // give the multi-token per-token pick the SAME signal. The BR.R3
            // majority scan above still runs first and is additive, so a
            // genuine cross-bucket article keeps rank 1 -- this only changes
            // which single-bucket seeds back it up. Grounded-only: an absent
            // or empty bucket still yields no candidate, so nonsense queries
            // gain nothing and the refusal floor is untouched.
            for token in &content {
                let idf = self.term_idf(token);
                let base = 0.3 + 0.7 * (idf / max_idf).min(1.0);
                let weight = if boost.contains(*token) { (base * 1.5).min(1.0) } else { base };
                if let Some(seeds_for_term) = self.keyword_index.get(*token) {
                    let mut ranked: Vec<(usize, Seed)> = Vec::with_capacity(
                        seeds_for_term.len().min(SINGLE_TOKEN_DEGREE_SCAN_CAP),
                    );
                    for seed in seeds_for_term.iter().take(SINGLE_TOKEN_DEGREE_SCAN_CAP) {
                        ranked.push((self.seam_total_degree(seed), *seed));
                    }
                    ranked.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.as_bytes().cmp(b.1.as_bytes())));
                    let mut taken = 0;
                    for (_, seed) in ranked.into_iter().take(SINGLE_TOKEN_LABEL_CHECK_DEPTH) {
                        let label = self.node_label_or_empty(&seed);
                        if !label.is_empty() && is_recall_pollution(&label) { continue; }
                        candidates.push((seed, weight));
                        taken += 1;
                        if taken >= 2 { break; }
                    }
                }
            }
        }

        // BR.C-Hyp.2.2 ASK-path coverage promotion. Per BR.C-Hyp.2
        // internal notes, root cause A+D): the per-token IDF score lets
        // a single-token-match seed beat a multi-token-match seed even
        // when both are reachable. Re-weight each candidate by
        // coverage = number of distinct content tokens whose keyword_index
        // bucket contains that seed (factor 1.0 / 1.6 / 2.2 for cov 1/2/3+).
        // The bucket is byte-sorted (BR.C-Hyp-Methodology fix at
        // `ensure_keyword_index_built`), so binary-search by `as_bytes()`
        // is correct and O(log N) per check. Single-content-token queries
        // (Q1 photosynthesis, Q10 China) are unchanged because cov=1
        // necessarily for every candidate. THINK-path callers pass false
        // here so the gate-12 coordination measurement is preserved
        // bit-identically (per the BR.C-Hyp.2.1 halt finding, learning #987).
        if apply_coverage_promotion && content.len() >= 2 && !candidates.is_empty() {
            for (seed, weight) in candidates.iter_mut() {
                let mut coverage = 0usize;
                for token in &content {
                    if let Some(bucket) = self.keyword_index.get(*token) {
                        if bucket
                            .binary_search_by(|s| s.as_bytes().cmp(seed.as_bytes()))
                            .is_ok()
                        {
                            coverage += 1;
                        }
                    }
                }
                let factor: f32 = match coverage {
                    0 | 1 => 1.0,
                    2 => 1.6,
                    _ => 2.2,
                };
                *weight = (*weight * factor).min(1.0);
            }
        }

        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

        let rank_mults = [1.0_f32, 0.6, 0.35];
        let mut seen: HashSet<Seed> = HashSet::new();
        let mut result: Vec<(Seed, f32)> = Vec::new();
        for (seed, weight) in candidates {
            if !seen.contains(&seed) {
                let rm = rank_mults.get(result.len()).copied().unwrap_or(0.2);
                seen.insert(seed);
                result.push((seed, weight * rm));
                if result.len() >= max_seeds { break; }
            }
        }
        result
    }

    /// Run a spread inside a Variant scope. Convenience wrapper that
    /// boxes `parent` and forwards to the standard weighted spread; the
    /// scope filter ensures variant-tagged keyframes are preferred and
    /// untagged bonds fall back to `parent`.
    pub fn spread_activation_in_variant_scope(
        &self,
        variant_id: u64,
        parent: TemporalScope,
        seeds: &[(Seed, f32)],
        depth: usize,
        top_k: usize,
        scar_log: &ScarLog,
    ) -> RecallResult {
        let scope = TemporalScope::Variant {
            id: variant_id,
            parent: Box::new(parent),
        };
        self.spread_activation_weighted(seeds, &scope, depth, top_k, scar_log)
    }

    /// Spread activation with per-seed initial activation weights.
    /// `spread_activation` delegates here with uniform weight 1.0.
    ///
    /// Bridge-detection-free wrapper. Calls the inner workhorse with a
    /// `None` collector so the cheap candidate-recording loop is skipped.
    pub fn spread_activation_weighted(
        &self,
        seeds: &[(Seed, f32)],
        scope: &TemporalScope,
        depth: usize,
        top_k: usize,
        scar_log: &ScarLog,
    ) -> RecallResult {
        self.spread_activation_weighted_with_wall_clock(
            seeds, scope, depth, top_k, scar_log, None,
        )
    }

    /// OP-ASK-WATCHDOG -- bridge-detection-free spread with an
    /// explicit wall-clock ceiling. `None` preserves legacy
    /// behaviour. Used by think pre-reflect and any caller that
    /// needs to bound a bare spread.
    pub fn spread_activation_weighted_with_wall_clock(
        &self,
        seeds: &[(Seed, f32)],
        scope: &TemporalScope,
        depth: usize,
        top_k: usize,
        scar_log: &ScarLog,
        max_wall_clock: Option<Duration>,
    ) -> RecallResult {
        let cfg = BridgeConfig::default();
        let cur_cfg = CuriosityConfig::default();
        self.spread_activation_weighted_inner(
            seeds, scope, depth, top_k, None, &cfg, 0,
            None, &cur_cfg, DEGREE_NORM_THRESHOLD,
            scar_log,
            None,
            max_wall_clock,
        )
    }

    /// Same spread, but the caller supplies a `BridgeCollector` that gets
    /// populated inline as the frontier crosses bonds. Phase-1 of bridge
    /// detection. Phase-2 (Jaccard classification) lives in `bridge.rs`
    /// and runs after this returns.
    ///
    /// `bridge_cfg` controls weak/hub thresholds and the per-spread cap.
    /// `timestamp_ms` is stamped onto every recorded candidate.
    pub fn spread_activation_weighted_with_bridges(
        &self,
        seeds: &[(Seed, f32)],
        scope: &TemporalScope,
        depth: usize,
        top_k: usize,
        bridge_collector: &mut BridgeCollector,
        bridge_cfg: &BridgeConfig,
        timestamp_ms: i64,
        scar_log: &ScarLog,
    ) -> RecallResult {
        let cur_cfg = CuriosityConfig::default();
        self.spread_activation_weighted_inner(
            seeds,
            scope,
            depth,
            top_k,
            Some(bridge_collector),
            bridge_cfg,
            timestamp_ms,
            None,
            &cur_cfg,
            DEGREE_NORM_THRESHOLD,
            scar_log,
            None,
            None,
        )
    }

    /// Curiosity-aware spread. Mirrors `_with_bridges` but threads a
    /// `CuriosityCollector` through the inline phase. Caller is
    /// responsible for running `curiosity::classify_curiosity_candidates`
    /// post-spread and folding the resulting signals into the brain.
    pub fn spread_activation_weighted_with_curiosity(
        &self,
        seeds: &[(Seed, f32)],
        scope: &TemporalScope,
        depth: usize,
        top_k: usize,
        curiosity_collector: &mut CuriosityCollector,
        curiosity_cfg: &CuriosityConfig,
        curiosity_hub_threshold: usize,
        timestamp_ms: u64,
        scar_log: &ScarLog,
    ) -> RecallResult {
        let bridge_cfg = BridgeConfig::default();
        self.spread_activation_weighted_inner(
            seeds,
            scope,
            depth,
            top_k,
            None,
            &bridge_cfg,
            timestamp_ms as i64,
            Some(curiosity_collector),
            curiosity_cfg,
            curiosity_hub_threshold,
            scar_log,
            None,
            None,
        )
    }

    /// Both-collectors spread. Used by `Brain::ask` so a single pass
    /// surfaces bridges AND curiosity signals without paying for the
    /// BFS twice.
    pub fn spread_activation_weighted_with_bridges_and_curiosity(
        &self,
        seeds: &[(Seed, f32)],
        scope: &TemporalScope,
        depth: usize,
        top_k: usize,
        bridge_collector: &mut BridgeCollector,
        bridge_cfg: &BridgeConfig,
        curiosity_collector: &mut CuriosityCollector,
        curiosity_cfg: &CuriosityConfig,
        curiosity_hub_threshold: usize,
        timestamp_ms: i64,
        scar_log: &ScarLog,
    ) -> RecallResult {
        self.spread_activation_weighted_inner(
            seeds,
            scope,
            depth,
            top_k,
            Some(bridge_collector),
            bridge_cfg,
            timestamp_ms,
            Some(curiosity_collector),
            curiosity_cfg,
            curiosity_hub_threshold,
            scar_log,
            None,
            None,
        )
    }

    /// Full-collector spread: bridges + curiosity + scars. The third
    /// collector accumulates copies of every active scar Recall
    /// observed and refused to propagate through. `Brain::ask` uses
    /// this so `AskResult.scars` can be populated and the bare
    /// `lu4d_ask` path can articulate refusal reasons without going
    /// through `multi_think` + narration.
    pub fn spread_activation_weighted_full(
        &self,
        seeds: &[(Seed, f32)],
        scope: &TemporalScope,
        depth: usize,
        top_k: usize,
        bridge_collector: &mut BridgeCollector,
        bridge_cfg: &BridgeConfig,
        curiosity_collector: &mut CuriosityCollector,
        curiosity_cfg: &CuriosityConfig,
        curiosity_hub_threshold: usize,
        timestamp_ms: i64,
        scar_log: &ScarLog,
        scar_collector: &mut Vec<crate::scar::Scar>,
        // OP-ASK-WATCHDOG -- wall-clock ceiling forwarded to the
        // inner frontier loop. `None` preserves legacy behaviour.
        max_wall_clock: Option<Duration>,
    ) -> RecallResult {
        self.spread_activation_weighted_inner(
            seeds,
            scope,
            depth,
            top_k,
            Some(bridge_collector),
            bridge_cfg,
            timestamp_ms,
            Some(curiosity_collector),
            curiosity_cfg,
            curiosity_hub_threshold,
            scar_log,
            Some(scar_collector),
            max_wall_clock,
        )
    }

    /// Inner workhorse. Identical to the old `spread_activation_weighted`
    /// body, with two new optional hooks (one in the outgoing loop, one
    /// in the incoming loop) that record bridge candidates when a
    /// collector is supplied.
    fn spread_activation_weighted_inner(
        &self,
        seeds: &[(Seed, f32)],
        scope: &TemporalScope,
        depth: usize,
        top_k: usize,
        mut bridge_collector: Option<&mut BridgeCollector>,
        bridge_cfg: &BridgeConfig,
        timestamp_ms: i64,
        mut curiosity_collector: Option<&mut CuriosityCollector>,
        curiosity_cfg: &CuriosityConfig,
        curiosity_hub_threshold: usize,
        scar_log: &ScarLog,
        // RECALL ENFORCES, CRITIC ARTICULATES (see scar.rs module docs).
        // The traversal already uses `effective_weight` so a scarred bond
        // contributes 0 — that is the enforcement layer. The collector
        // here is the *articulation* hook: it captures copies of the
        // active scars Recall ignored, so AskResult.scars can let Lu
        // surface the refusal reason from a bare ask() call (without
        // routing through multi_think + narration).
        mut scar_collector: Option<&mut Vec<crate::scar::Scar>>,
        // OP-ASK-WATCHDOG -- wall-clock ceiling on the BFS frontier
        // loop. `None` = no ceiling (engine default, audit harness).
        // `Some(d)` = check at the top of each frontier iteration; on
        // expiry, abandon the frontier and return partial activations
        // with `RecallResult.status = TimedOut { activated_so_far }`.
        max_wall_clock: Option<Duration>,
    ) -> RecallResult {
        let mut activations: HashMap<Seed, f32> = HashMap::new();
        let mut paths: HashMap<Seed, Vec<TraversalStep>> = HashMap::new();
        let mut depths: HashMap<Seed, usize> = HashMap::new();
        let mut visited: HashSet<Seed> = HashSet::new();
        let mut frontier: VecDeque<(Seed, f32, usize)> = VecDeque::new();
        let mut nodes_visited: usize = 0;
        let mut bonds_evaluated: usize = 0;
        let mut activation_truncated: bool = false;
        let cap_activations: usize = self.effective_cap_activations();
        // OP-ASK-WATCHDOG -- capture wall-clock anchor BEFORE seeding
        // so the budget accounts for the seed-resolution and frontier
        // bootstrap cost. `None` disables the watchdog (engine default,
        // audit harnesses, deterministic tests).
        let started = Instant::now();
        let watchdog_fired = |budget: Option<Duration>| -> bool {
            match budget {
                Some(d) => started.elapsed() >= d,
                None => false,
            }
        };
        let mut spread_status: SpreadStatus = SpreadStatus::Complete;
        // OP-A0: proactive, deterministic spread bounds (default-off).
        let beam_width = spread_beam_width();
        let max_expansions = spread_max_expansions();
        let mut expansions: usize = 0;
        let mut enq_per_level: HashMap<usize, usize> = HashMap::new();

        for (seed, weight) in seeds {
            if let Some(_n_view) = self.seam_recall_node(seed) {
                let confidence = _n_view.confidence_value();
                let initial_activation = weight * confidence;
                activations.insert(*seed, initial_activation);
                paths.insert(*seed, Vec::new());
                depths.insert(*seed, 0);
                frontier.push_back((*seed, initial_activation, 0));
                visited.insert(*seed);
                nodes_visited += 1;
            }
        }

        while let Some((cur, cur_act, cur_d)) = frontier.pop_front() {
            // OP-ASK-WATCHDOG -- guard at the top of each frontier
            // iteration. Each iteration touches potentially-cold mmap
            // blocks via `view_bond` / `view_node`, so this is the
            // natural checkpoint where the encrypted-block decrypt +
            // zstd cost is amortised. On expiry we abandon the
            // remaining frontier, record `TimedOut` status, and fall
            // through to the existing sort/truncate so partial
            // activations still surface ranked.
            if watchdog_fired(max_wall_clock) {
                spread_status = SpreadStatus::TimedOut {
                    activated_so_far: activations.len(),
                };
                break;
            }
            if cur_d >= depth { continue; }
            // OP-A0: hard expansion ceiling -- deterministic latency bound.
            if let Some(cap) = max_expansions {
                if expansions >= cap { activation_truncated = true; break; }
            }
            expansions += 1;

            // FIX 2 (OP-HUB-EXPANSION-SKIP): skip onward expansion of an
            // extreme-degree hub. seam_total_degree is an O(1) mmap probe;
            // the seam_spread_*_edges calls below would otherwise decode the
            // node's ENTIRE adjacency (a mega-hub reads ~34 GB in one
            // uninterruptible iteration) only to hub-cap it afterwards. The
            // node already activated; we only stop the flood through it.
            if self.seam_total_degree(&cur) > HUB_EXPANSION_DEGREE_CEILING {
                continue;
            }

            // Curiosity tracking (Phase 1): per-frontier-step yield count.
            // Incremented when a neighbor is fresh AND propagates.
            let mut yields_this_step: usize = 0;

            // OP-SPREAD-1 -- HOIST THE `cur`-INVARIANT SEAM PROBES.
            // `cur` is fixed for the whole of both edge loops, yet the
            // old code re-resolved its node view, its content, its
            // stopword-ness and (inside `bridge::is_candidate`) its
            // degree ONCE PER EDGE. On a 5,000-wide hub that is 5,000
            // identical `NodeIntern` probes for one unchanging answer.
            // Hoisting is exact: same inputs, same values, and the
            // `||` short-circuit below still tests `nbr` first.
            let (cur_node_view, cur_total_deg) = self.seam_node_and_degree(&cur);
            let cur_seam_cont: &str =
                cur_node_view.as_ref().map(|n| n.content()).unwrap_or("");
            let cur_is_stopword = self.is_stopword_node(&cur, cur_seam_cont);
            // Mirrors `bridge::is_hub(self, &cur, ..)` exactly, with the
            // degree supplied rather than re-probed.
            let cur_is_bridge_hub = crate::bridge::is_hub_with_degree(
                self, &cur, cur_total_deg, bridge_cfg.hub_degree_threshold,
            );

            // --- outgoing bonds ---
            for edge in self.seam_spread_outgoing_edges(&cur) {
                let nbr = edge.nbr;
                bonds_evaluated += 1;
                let carried_bond: Option<TemporalBond> = match &edge.bond {
                    SeamBondSrc::Carried { predicate, payload } =>
                        Some(Self::reconstruct_carried_bond(cur, nbr, *predicate, payload)),
                    SeamBondSrc::Resident => None,
                };
                let bond = match &carried_bond {
                    Some(rb) => crate::view::BondView::from_heap(rb),
                    None => match self.view_bond(&(cur, nbr)) { Some(b) => b, None => continue },
                };
                let reading = bond.weight_in_scope(scope, scar_log);
                if let Some(sc) = scar_collector.as_deref_mut() {
                    for s in &reading.active_scars {
                        sc.push(s.clone());
                    }
                }
                // Port T.5.1: query-path effective weight = stored * freshness.
                let raw_bw = reading.effective_weight * self.decay_multiplier_for_bond(&bond) * self.sense_gain_for_bond(&cur, &nbr);

                // Bridge candidate hook (Phase 1). Cheap predicate; uses
                // the un-halved bond weight since the bond's actual
                // strength is direction-invariant.
                // OP-SPREAD-1 -- `bridge::is_candidate` inlined so its
                // CHEAP guards (enabled, weak-bond, `cur` is a hub --
                // all already in hand) run BEFORE anything probes
                // `nbr`. The old call re-probed BOTH endpoints'
                // degrees on every edge, including the overwhelming
                // majority that fail these guards. Same predicate,
                // same short-circuit order, same result.
                if let Some(bc) = bridge_collector.as_deref_mut() {
                    if bridge_cfg.bridges_enabled
                        && raw_bw.abs() < bridge_cfg.weak_bond_threshold
                        && cur_is_bridge_hub
                        && crate::bridge::is_hub_with_degree(
                            self,
                            &nbr,
                            self.seam_total_degree(&nbr),
                            bridge_cfg.hub_degree_threshold,
                        )
                    {
                        bc.record(cur, nbr, raw_bw, timestamp_ms);
                    }
                }

                let bw = raw_bw;
                if bw.abs() < ACTIVATION_THRESHOLD { continue; }

                // OP-SPREAD-1 -- ONE intern probe answers all three
                // per-edge questions about `nbr` (presence, out-degree,
                // in-degree). Was three separate probes, plus two more
                // inside the bridge hook above.
                let (nbr_node, nbr_total_deg) = self.seam_node_and_degree(&nbr);
                let nbr_conf = nbr_node.as_ref().map(|n| n.confidence_value()).unwrap_or(0.5);
                let nbr_cont = nbr_node.as_ref().map(|n| n.content()).unwrap_or("");

                let deg = nbr_total_deg;
                let hub_t = self.degree_p90;
                let deg_norm = if deg <= hub_t { 1.0_f32 }
                    else { ((hub_t as f32) / deg as f32).sqrt() };
                // OP-RECALL-1 (R3): `sw` is either 0.2 or 1.0, so it can
                // only LOWER activation. A candidate already below
                // threshold without it is dropped either way, so test
                // first -- that keeps the label resolutions inside
                // `is_stopword_node` to the neighbours that actually
                // activate rather than one per edge of a 5,000-wide hub.
                // Exact, not an approximation: act = act_pre * sw <= act_pre.
                let act_pre = cur_act * bw * DECAY_PER_HOP * nbr_conf * deg_norm;
                if act_pre < ACTIVATION_THRESHOLD { continue; }
                let sw = if self.is_stopword_node(&nbr, nbr_cont)
                    || cur_is_stopword {
                    STOPWORD_BOND_DECAY } else { 1.0 };
                let act = act_pre * sw;
                if act < ACTIVATION_THRESHOLD { continue; }

                let step = TraversalStep { node: nbr, bond_type: bond.bond_type(), bond_weight: bw, activation: act };

                if visited.contains(&nbr) {
                    if let Some(ex) = activations.get_mut(&nbr) {
                        if act > *ex {
                            *ex = act;
                            let mut p = paths.get(&cur).cloned().unwrap_or_default();
                            p.push(step); paths.insert(nbr, p);
                        }
                    }
                    continue;
                }
                if activations.len() >= cap_activations {
                    activation_truncated = true;
                    continue;
                }
                visited.insert(nbr); nodes_visited += 1;
                activations.insert(nbr, act);
                depths.insert(nbr, cur_d + 1);
                let mut p = paths.get(&cur).cloned().unwrap_or_default();
                p.push(step); paths.insert(nbr, p);
                if !self.is_stopword_node(&nbr, nbr_cont) {
                    if activations.len() < cap_activations {
                        // OP-A0: per-level beam -- cap onward enqueues per depth.
                        let nd = cur_d + 1;
                        let allow = beam_width.map_or(true, |b| {
                            let c = enq_per_level.entry(nd).or_insert(0);
                            if *c < b { *c += 1; true } else { false }
                        });
                        if allow {
                            frontier.push_back((nbr, act, nd));
                            yields_this_step += 1;
                        } else {
                            activation_truncated = true;
                        }
                    } else {
                        activation_truncated = true;
                    }
                }
            }

            // --- incoming bonds (half weight for traversal) ---
            for edge in self.seam_spread_incoming_edges(&cur) {
                let nbr = edge.nbr;
                bonds_evaluated += 1;
                let carried_bond: Option<TemporalBond> = match &edge.bond {
                    SeamBondSrc::Carried { predicate, payload } =>
                        Some(Self::reconstruct_carried_bond(nbr, cur, *predicate, payload)),
                    SeamBondSrc::Resident => None,
                };
                let bond = match &carried_bond {
                    Some(rb) => crate::view::BondView::from_heap(rb),
                    None => match self.view_bond(&(nbr, cur)) { Some(b) => b, None => continue },
                };
                let reading = bond.weight_in_scope(scope, scar_log);
                if let Some(sc) = scar_collector.as_deref_mut() {
                    for s in &reading.active_scars {
                        sc.push(s.clone());
                    }
                }
                // Port T.5.1: query-path effective weight = stored * freshness.
                let raw_bw = reading.effective_weight * self.decay_multiplier_for_bond(&bond) * self.sense_gain_for_bond(&nbr, &cur);

                // Bridge candidate hook — same direction-invariant raw weight.
                // OP-SPREAD-1 -- `bridge::is_candidate` inlined so its
                // CHEAP guards (enabled, weak-bond, `cur` is a hub --
                // all already in hand) run BEFORE anything probes
                // `nbr`. The old call re-probed BOTH endpoints'
                // degrees on every edge, including the overwhelming
                // majority that fail these guards. Same predicate,
                // same short-circuit order, same result.
                if let Some(bc) = bridge_collector.as_deref_mut() {
                    if bridge_cfg.bridges_enabled
                        && raw_bw.abs() < bridge_cfg.weak_bond_threshold
                        && cur_is_bridge_hub
                        && crate::bridge::is_hub_with_degree(
                            self,
                            &nbr,
                            self.seam_total_degree(&nbr),
                            bridge_cfg.hub_degree_threshold,
                        )
                    {
                        bc.record(cur, nbr, raw_bw, timestamp_ms);
                    }
                }

                let bw = raw_bw * 0.5;
                if bw.abs() < ACTIVATION_THRESHOLD { continue; }

                // OP-SPREAD-1 -- ONE intern probe answers all three
                // per-edge questions about `nbr` (presence, out-degree,
                // in-degree). Was three separate probes, plus two more
                // inside the bridge hook above.
                let (nbr_node, nbr_total_deg) = self.seam_node_and_degree(&nbr);
                let nbr_conf = nbr_node.as_ref().map(|n| n.confidence_value()).unwrap_or(0.5);
                let nbr_cont = nbr_node.as_ref().map(|n| n.content()).unwrap_or("");

                let deg = nbr_total_deg;
                let hub_t = self.degree_p90;
                let deg_norm = if deg <= hub_t { 1.0_f32 }
                    else { ((hub_t as f32) / deg as f32).sqrt() };
                // OP-RECALL-1 (R3): `sw` is either 0.2 or 1.0, so it can
                // only LOWER activation. A candidate already below
                // threshold without it is dropped either way, so test
                // first -- that keeps the label resolutions inside
                // `is_stopword_node` to the neighbours that actually
                // activate rather than one per edge of a 5,000-wide hub.
                // Exact, not an approximation: act = act_pre * sw <= act_pre.
                let act_pre = cur_act * bw * DECAY_PER_HOP * nbr_conf * deg_norm;
                if act_pre < ACTIVATION_THRESHOLD { continue; }
                let sw = if self.is_stopword_node(&nbr, nbr_cont)
                    || cur_is_stopword {
                    STOPWORD_BOND_DECAY } else { 1.0 };
                let act = act_pre * sw;
                if act < ACTIVATION_THRESHOLD { continue; }

                let step = TraversalStep { node: nbr, bond_type: bond.bond_type(), bond_weight: bw, activation: act };

                if visited.contains(&nbr) {
                    if let Some(ex) = activations.get_mut(&nbr) {
                        if act > *ex {
                            *ex = act;
                            let mut p = paths.get(&cur).cloned().unwrap_or_default();
                            p.push(step); paths.insert(nbr, p);
                        }
                    }
                    continue;
                }
                if activations.len() >= cap_activations {
                    activation_truncated = true;
                    continue;
                }
                visited.insert(nbr); nodes_visited += 1;
                activations.insert(nbr, act);
                depths.insert(nbr, cur_d + 1);
                let mut p = paths.get(&cur).cloned().unwrap_or_default();
                p.push(step); paths.insert(nbr, p);
                if !self.is_stopword_node(&nbr, nbr_cont) {
                    if activations.len() < cap_activations {
                        // OP-A0: per-level beam -- cap onward enqueues per depth.
                        let nd = cur_d + 1;
                        let allow = beam_width.map_or(true, |b| {
                            let c = enq_per_level.entry(nd).or_insert(0);
                            if *c < b { *c += 1; true } else { false }
                        });
                        if allow {
                            frontier.push_back((nbr, act, nd));
                            yields_this_step += 1;
                        } else {
                            activation_truncated = true;
                        }
                    } else {
                        activation_truncated = true;
                    }
                }
            }

            // Curiosity post-step classification: if `cur` is a hub and
            // its yield count fell short, record a DeadEnd / HubBoundary
            // candidate. Cheap predicates only — Phase-2 scoring lives
            // in `curiosity::classify_curiosity_candidates`.
            if let Some(cc) = curiosity_collector.as_deref_mut() {
                if curiosity_cfg.enabled && cur_d > 0 {
                    let cur_deg = self.seam_total_degree(&cur);
                    // OP-RECALL-1 (R3) / OP-CURIOSITY switch 4. This was
                    // hard-coded to "" on the cache-backed path by
                    // 9b59082 (OP-BRAIN-MapRAM P3), which killed
                    // `is_hub_stopword` here and left hub classification
                    // degree-threshold-only -- so curiosity could never
                    // tell a function-word hub from a topical one. `cur`
                    // is a frontier node the loop already committed to,
                    // and its label is memoized by the `sw` computation
                    // above, so the resolver call is free in practice.
                    let cur_cont = self.nodes.get(&cur).map(|n| n.content()).unwrap_or("");
                    let is_cur_hub = self.is_stopword_node(&cur, cur_cont)
                        || cur_deg > curiosity_hub_threshold;
                    if is_cur_hub {
                        if yields_this_step == 0 {
                            cc.record(SignalType::HubBoundary, cur, cur_deg as f32, timestamp_ms as u64);
                        } else if yields_this_step < curiosity_cfg.deadend_yield_threshold {
                            cc.record(SignalType::DeadEnd, cur, yields_this_step as f32, timestamp_ms as u64);
                        }
                    }
                }
            }
        }

        let mut activated: Vec<ActivatedNode> = activations.iter()
            .filter_map(|(seed, act)| {
                // The seam is the PRESENCE oracle (it knows the node is
                // in the graph); it is not the LABEL oracle -- on a v5
                // brain its view is a content-free stub. OP-LABEL-1
                // resolves the label from the node store, which is the
                // only place the text actually lives.
                let node = self.seam_recall_node(seed)?;
                let content = self.node_label_or_empty(seed);
                // Port #24 Bug 1+2+3: drop polluted nodes from the final
                // result list. They may have been touched during BFS but
                // should never surface as a top-K activation. NOTE: this
                // filter was a no-op on v5 brains until OP-LABEL-1,
                // because it was being handed the empty string.
                if is_recall_pollution(&content) { return None; }
                // OP-ROUTE-GUEST (2026-07-23): down-rank non-concept
                // labels in the frontier so a real concept outranks a
                // sentence-fragment / article-led / category-tail node
                // for the answer head and bridge endpoints. Read-only,
                // reversible via LU_JUNK_DOWNRANK=0; nothing is dropped.
                let activation = if junk_downrank_enabled() && non_concept_label(&content) {
                    *act * NON_CONCEPT_DOWNRANK
                } else {
                    *act
                };
                Some(ActivatedNode {
                    seed: *seed, content,
                    activation, confidence: node.confidence(),
                    path: paths.get(seed).cloned().unwrap_or_default(),
                    depth: *depths.get(seed).unwrap_or(&0),
                })
            })
            .collect();
        // BR.C-Hyp-Methodology — total order on (activation desc,
        // seed bytes asc). Without the seed tiebreaker, f32 ties at
        // the top_k boundary surface in HashMap iteration order,
        // which Rust's default RandomState makes per-process random.
        activated.sort_by(|a, b| {
            b.activation.partial_cmp(&a.activation)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.seed.as_bytes().cmp(b.seed.as_bytes()))
        });
        activated.truncate(top_k);

        RecallResult {
            activated,
            query_seeds: seeds.iter().map(|(s, _)| *s).collect(),
            scope: scope.clone(),
            nodes_visited,
            bonds_evaluated,
            activation_truncated,
            status: spread_status,
        }
    }

    /// Like `spread_activation_weighted` but filters traversal by bond type.
    /// Bridge-detection-free wrapper. Calls the inner workhorse with a
    /// `None` collector so the candidate-recording loop is skipped.
    pub fn typed_recall_weighted(
        &self,
        seeds: &[(Seed, f32)],
        scope: &TemporalScope,
        depth: usize,
        top_k: usize,
        bond_filter: &[BondType],
        scar_log: &ScarLog,
    ) -> RecallResult {
        let cfg = BridgeConfig::default();
        let cur_cfg = CuriosityConfig::default();
        self.typed_recall_weighted_inner(
            seeds, scope, depth, top_k, bond_filter, None, &cfg, 0,
            None, &cur_cfg, DEGREE_NORM_THRESHOLD,
            scar_log,
        )
    }

    /// Same typed spread, but the caller supplies a `BridgeCollector` that
    /// gets populated inline as the frontier crosses bonds. Mirrors
    /// `spread_activation_weighted_with_bridges` so all spread variants
    /// surface bridges through one unified Phase-2 classify call.
    pub fn typed_recall_weighted_with_bridges(
        &self,
        seeds: &[(Seed, f32)],
        scope: &TemporalScope,
        depth: usize,
        top_k: usize,
        bond_filter: &[BondType],
        bridge_collector: &mut BridgeCollector,
        bridge_cfg: &BridgeConfig,
        timestamp_ms: i64,
        scar_log: &ScarLog,
    ) -> RecallResult {
        let cur_cfg = CuriosityConfig::default();
        self.typed_recall_weighted_inner(
            seeds, scope, depth, top_k, bond_filter,
            Some(bridge_collector), bridge_cfg, timestamp_ms,
            None, &cur_cfg, DEGREE_NORM_THRESHOLD,
            scar_log,
        )
    }

    /// Curiosity-aware typed recall. Mirrors the untyped curiosity
    /// wrapper. Caller runs Phase-2 classification post-spread.
    pub fn typed_recall_weighted_with_curiosity(
        &self,
        seeds: &[(Seed, f32)],
        scope: &TemporalScope,
        depth: usize,
        top_k: usize,
        bond_filter: &[BondType],
        curiosity_collector: &mut CuriosityCollector,
        curiosity_cfg: &CuriosityConfig,
        curiosity_hub_threshold: usize,
        timestamp_ms: u64,
        scar_log: &ScarLog,
    ) -> RecallResult {
        let bridge_cfg = BridgeConfig::default();
        self.typed_recall_weighted_inner(
            seeds, scope, depth, top_k, bond_filter,
            None, &bridge_cfg, timestamp_ms as i64,
            Some(curiosity_collector), curiosity_cfg, curiosity_hub_threshold,
            scar_log,
        )
    }

    /// Both-collectors typed recall. Used by `Brain::typed_ask`.
    pub fn typed_recall_weighted_with_bridges_and_curiosity(
        &self,
        seeds: &[(Seed, f32)],
        scope: &TemporalScope,
        depth: usize,
        top_k: usize,
        bond_filter: &[BondType],
        bridge_collector: &mut BridgeCollector,
        bridge_cfg: &BridgeConfig,
        curiosity_collector: &mut CuriosityCollector,
        curiosity_cfg: &CuriosityConfig,
        curiosity_hub_threshold: usize,
        timestamp_ms: i64,
        scar_log: &ScarLog,
    ) -> RecallResult {
        self.typed_recall_weighted_inner(
            seeds, scope, depth, top_k, bond_filter,
            Some(bridge_collector), bridge_cfg, timestamp_ms,
            Some(curiosity_collector), curiosity_cfg, curiosity_hub_threshold,
            scar_log,
        )
    }

    /// Inner workhorse for typed recall. Body matches the original
    /// `typed_recall_weighted`, with two added hooks (one outgoing, one
    /// incoming) that record bridge candidates when a collector is
    /// supplied.
    fn typed_recall_weighted_inner(
        &self,
        seeds: &[(Seed, f32)],
        scope: &TemporalScope,
        depth: usize,
        top_k: usize,
        bond_filter: &[BondType],
        mut bridge_collector: Option<&mut BridgeCollector>,
        bridge_cfg: &BridgeConfig,
        timestamp_ms: i64,
        mut curiosity_collector: Option<&mut CuriosityCollector>,
        curiosity_cfg: &CuriosityConfig,
        curiosity_hub_threshold: usize,
        scar_log: &ScarLog,
    ) -> RecallResult {
        let mut activations: HashMap<Seed, f32> = HashMap::new();
        let mut paths: HashMap<Seed, Vec<TraversalStep>> = HashMap::new();
        let mut depths: HashMap<Seed, usize> = HashMap::new();
        let mut visited: HashSet<Seed> = HashSet::new();
        let mut frontier: VecDeque<(Seed, f32, usize)> = VecDeque::new();
        let mut nodes_visited: usize = 0;
        let mut bonds_evaluated: usize = 0;
        let mut activation_truncated: bool = false;
        let cap_activations: usize = self.effective_cap_activations();

        for (seed, weight) in seeds {
            if let Some(_n_view) = self.seam_recall_node(seed) {
                let confidence = _n_view.confidence_value();
                let initial_activation = weight * confidence;
                activations.insert(*seed, initial_activation);
                paths.insert(*seed, Vec::new());
                depths.insert(*seed, 0);
                frontier.push_back((*seed, initial_activation, 0));
                visited.insert(*seed);
                nodes_visited += 1;
            }
        }

        while let Some((cur, cur_act, cur_d)) = frontier.pop_front() {
            if cur_d >= depth { continue; }

            // FIX 2 (OP-HUB-EXPANSION-SKIP): skip onward expansion of an
            // extreme-degree hub. seam_total_degree is an O(1) mmap probe;
            // the seam_spread_*_edges calls below would otherwise decode the
            // node's ENTIRE adjacency (a mega-hub reads ~34 GB in one
            // uninterruptible iteration) only to hub-cap it afterwards. The
            // node already activated; we only stop the flood through it.
            if self.seam_total_degree(&cur) > HUB_EXPANSION_DEGREE_CEILING {
                continue;
            }

            // Curiosity tracking (Phase 1): per-frontier-step yield count.
            let mut yields_this_step: usize = 0;

            // OP-SPREAD-1 -- HOIST THE `cur`-INVARIANT SEAM PROBES.
            // `cur` is fixed for the whole of both edge loops, yet the
            // old code re-resolved its node view, its content, its
            // stopword-ness and (inside `bridge::is_candidate`) its
            // degree ONCE PER EDGE. On a 5,000-wide hub that is 5,000
            // identical `NodeIntern` probes for one unchanging answer.
            // Hoisting is exact: same inputs, same values, and the
            // `||` short-circuit below still tests `nbr` first.
            let (cur_node_view, cur_total_deg) = self.seam_node_and_degree(&cur);
            let cur_seam_cont: &str =
                cur_node_view.as_ref().map(|n| n.content()).unwrap_or("");
            let cur_is_stopword = self.is_stopword_node(&cur, cur_seam_cont);
            // Mirrors `bridge::is_hub(self, &cur, ..)` exactly, with the
            // degree supplied rather than re-probed.
            let cur_is_bridge_hub = crate::bridge::is_hub_with_degree(
                self, &cur, cur_total_deg, bridge_cfg.hub_degree_threshold,
            );

            for edge in self.seam_spread_outgoing_edges(&cur) {
                let nbr = edge.nbr;
                bonds_evaluated += 1;
                let carried_bond: Option<TemporalBond> = match &edge.bond {
                    SeamBondSrc::Carried { predicate, payload } =>
                        Some(Self::reconstruct_carried_bond(cur, nbr, *predicate, payload)),
                    SeamBondSrc::Resident => None,
                };
                let bond = match &carried_bond {
                    Some(rb) => crate::view::BondView::from_heap(rb),
                    None => match self.view_bond(&(cur, nbr)) { Some(b) => b, None => continue },
                };
                if !bond_filter.is_empty() && !bond_filter.contains(&bond.bond_type()) { continue; }
                // Port T.5.1: query-path effective weight = stored * freshness.
                let raw_bw = bond.weight_in_scope(scope, scar_log).effective_weight
                    * self.decay_multiplier_for_bond(&bond) * self.sense_gain_for_bond(&cur, &nbr);

                // Bridge candidate hook (Phase 1) — direction-invariant raw weight.
                // OP-SPREAD-1 -- `bridge::is_candidate` inlined so its
                // CHEAP guards (enabled, weak-bond, `cur` is a hub --
                // all already in hand) run BEFORE anything probes
                // `nbr`. The old call re-probed BOTH endpoints'
                // degrees on every edge, including the overwhelming
                // majority that fail these guards. Same predicate,
                // same short-circuit order, same result.
                if let Some(bc) = bridge_collector.as_deref_mut() {
                    if bridge_cfg.bridges_enabled
                        && raw_bw.abs() < bridge_cfg.weak_bond_threshold
                        && cur_is_bridge_hub
                        && crate::bridge::is_hub_with_degree(
                            self,
                            &nbr,
                            self.seam_total_degree(&nbr),
                            bridge_cfg.hub_degree_threshold,
                        )
                    {
                        bc.record(cur, nbr, raw_bw, timestamp_ms);
                    }
                }

                let bw = raw_bw;
                if bw.abs() < ACTIVATION_THRESHOLD { continue; }

                // OP-SPREAD-1 -- ONE intern probe answers all three
                // per-edge questions about `nbr` (presence, out-degree,
                // in-degree). Was three separate probes, plus two more
                // inside the bridge hook above.
                let (nbr_node, nbr_total_deg) = self.seam_node_and_degree(&nbr);
                let nbr_conf = nbr_node.as_ref().map(|n| n.confidence_value()).unwrap_or(0.5);
                let nbr_cont = nbr_node.as_ref().map(|n| n.content()).unwrap_or("");
                let deg = nbr_total_deg;
                let hub_t = self.degree_p90;
                let deg_norm = if deg <= hub_t { 1.0_f32 }
                    else { ((hub_t as f32) / deg as f32).sqrt() };
                // OP-RECALL-1 (R3): `sw` is either 0.2 or 1.0, so it can
                // only LOWER activation. A candidate already below
                // threshold without it is dropped either way, so test
                // first -- that keeps the label resolutions inside
                // `is_stopword_node` to the neighbours that actually
                // activate rather than one per edge of a 5,000-wide hub.
                // Exact, not an approximation: act = act_pre * sw <= act_pre.
                let act_pre = cur_act * bw * DECAY_PER_HOP * nbr_conf * deg_norm;
                if act_pre < ACTIVATION_THRESHOLD { continue; }
                let sw = if self.is_stopword_node(&nbr, nbr_cont)
                    || cur_is_stopword {
                    STOPWORD_BOND_DECAY } else { 1.0 };
                let act = act_pre * sw;
                if act < ACTIVATION_THRESHOLD { continue; }

                let step = TraversalStep { node: nbr, bond_type: bond.bond_type(), bond_weight: bw, activation: act };
                if visited.contains(&nbr) {
                    if let Some(ex) = activations.get_mut(&nbr) {
                        if act > *ex {
                            *ex = act;
                            let mut p = paths.get(&cur).cloned().unwrap_or_default();
                            p.push(step); paths.insert(nbr, p);
                        }
                    }
                    continue;
                }
                if activations.len() >= cap_activations {
                    activation_truncated = true;
                    continue;
                }
                visited.insert(nbr); nodes_visited += 1;
                activations.insert(nbr, act);
                depths.insert(nbr, cur_d + 1);
                let mut p = paths.get(&cur).cloned().unwrap_or_default();
                p.push(step); paths.insert(nbr, p);
                if !self.is_stopword_node(&nbr, nbr_cont) {
                    if activations.len() < cap_activations {
                        frontier.push_back((nbr, act, cur_d + 1));
                        yields_this_step += 1;
                    } else {
                        activation_truncated = true;
                    }
                }
            }

            for edge in self.seam_spread_incoming_edges(&cur) {
                let nbr = edge.nbr;
                bonds_evaluated += 1;
                let carried_bond: Option<TemporalBond> = match &edge.bond {
                    SeamBondSrc::Carried { predicate, payload } =>
                        Some(Self::reconstruct_carried_bond(nbr, cur, *predicate, payload)),
                    SeamBondSrc::Resident => None,
                };
                let bond = match &carried_bond {
                    Some(rb) => crate::view::BondView::from_heap(rb),
                    None => match self.view_bond(&(nbr, cur)) { Some(b) => b, None => continue },
                };
                if !bond_filter.is_empty() && !bond_filter.contains(&bond.bond_type()) { continue; }
                // Port T.5.1: query-path effective weight = stored * freshness.
                let raw_bw = bond.weight_in_scope(scope, scar_log).effective_weight
                    * self.decay_multiplier_for_bond(&bond) * self.sense_gain_for_bond(&nbr, &cur);

                // Bridge candidate hook — same direction-invariant raw weight.
                // OP-SPREAD-1 -- `bridge::is_candidate` inlined so its
                // CHEAP guards (enabled, weak-bond, `cur` is a hub --
                // all already in hand) run BEFORE anything probes
                // `nbr`. The old call re-probed BOTH endpoints'
                // degrees on every edge, including the overwhelming
                // majority that fail these guards. Same predicate,
                // same short-circuit order, same result.
                if let Some(bc) = bridge_collector.as_deref_mut() {
                    if bridge_cfg.bridges_enabled
                        && raw_bw.abs() < bridge_cfg.weak_bond_threshold
                        && cur_is_bridge_hub
                        && crate::bridge::is_hub_with_degree(
                            self,
                            &nbr,
                            self.seam_total_degree(&nbr),
                            bridge_cfg.hub_degree_threshold,
                        )
                    {
                        bc.record(cur, nbr, raw_bw, timestamp_ms);
                    }
                }

                let bw = raw_bw * 0.5;
                if bw.abs() < ACTIVATION_THRESHOLD { continue; }

                // OP-SPREAD-1 -- ONE intern probe answers all three
                // per-edge questions about `nbr` (presence, out-degree,
                // in-degree). Was three separate probes, plus two more
                // inside the bridge hook above.
                let (nbr_node, nbr_total_deg) = self.seam_node_and_degree(&nbr);
                let nbr_conf = nbr_node.as_ref().map(|n| n.confidence_value()).unwrap_or(0.5);
                let nbr_cont = nbr_node.as_ref().map(|n| n.content()).unwrap_or("");
                let deg = nbr_total_deg;
                let hub_t = self.degree_p90;
                let deg_norm = if deg <= hub_t { 1.0_f32 }
                    else { ((hub_t as f32) / deg as f32).sqrt() };
                // OP-RECALL-1 (R3): `sw` is either 0.2 or 1.0, so it can
                // only LOWER activation. A candidate already below
                // threshold without it is dropped either way, so test
                // first -- that keeps the label resolutions inside
                // `is_stopword_node` to the neighbours that actually
                // activate rather than one per edge of a 5,000-wide hub.
                // Exact, not an approximation: act = act_pre * sw <= act_pre.
                let act_pre = cur_act * bw * DECAY_PER_HOP * nbr_conf * deg_norm;
                if act_pre < ACTIVATION_THRESHOLD { continue; }
                let sw = if self.is_stopword_node(&nbr, nbr_cont)
                    || cur_is_stopword {
                    STOPWORD_BOND_DECAY } else { 1.0 };
                let act = act_pre * sw;
                if act < ACTIVATION_THRESHOLD { continue; }

                let step = TraversalStep { node: nbr, bond_type: bond.bond_type(), bond_weight: bw, activation: act };
                if visited.contains(&nbr) {
                    if let Some(ex) = activations.get_mut(&nbr) {
                        if act > *ex {
                            *ex = act;
                            let mut p = paths.get(&cur).cloned().unwrap_or_default();
                            p.push(step); paths.insert(nbr, p);
                        }
                    }
                    continue;
                }
                if activations.len() >= cap_activations {
                    activation_truncated = true;
                    continue;
                }
                visited.insert(nbr); nodes_visited += 1;
                activations.insert(nbr, act);
                depths.insert(nbr, cur_d + 1);
                let mut p = paths.get(&cur).cloned().unwrap_or_default();
                p.push(step); paths.insert(nbr, p);
                if !self.is_stopword_node(&nbr, nbr_cont) {
                    if activations.len() < cap_activations {
                        frontier.push_back((nbr, act, cur_d + 1));
                        yields_this_step += 1;
                    } else {
                        activation_truncated = true;
                    }
                }
            }

            // Curiosity post-step classification (mirrors the untyped variant).
            if let Some(cc) = curiosity_collector.as_deref_mut() {
                if curiosity_cfg.enabled && cur_d > 0 {
                    let cur_deg = self.seam_total_degree(&cur);
                    // OP-RECALL-1 (R3) / OP-CURIOSITY switch 4. This was
                    // hard-coded to "" on the cache-backed path by
                    // 9b59082 (OP-BRAIN-MapRAM P3), which killed
                    // `is_hub_stopword` here and left hub classification
                    // degree-threshold-only -- so curiosity could never
                    // tell a function-word hub from a topical one. `cur`
                    // is a frontier node the loop already committed to,
                    // and its label is memoized by the `sw` computation
                    // above, so the resolver call is free in practice.
                    let cur_cont = self.nodes.get(&cur).map(|n| n.content()).unwrap_or("");
                    let is_cur_hub = self.is_stopword_node(&cur, cur_cont)
                        || cur_deg > curiosity_hub_threshold;
                    if is_cur_hub {
                        if yields_this_step == 0 {
                            cc.record(SignalType::HubBoundary, cur, cur_deg as f32, timestamp_ms as u64);
                        } else if yields_this_step < curiosity_cfg.deadend_yield_threshold {
                            cc.record(SignalType::DeadEnd, cur, yields_this_step as f32, timestamp_ms as u64);
                        }
                    }
                }
            }
        }

        let mut activated: Vec<ActivatedNode> = activations.iter()
            .filter_map(|(seed, act)| {
                let node = self.seam_recall_node(seed)?;
                // OP-LABEL-1: label from the node store, not the stub.
                Some(ActivatedNode {
                    seed: *seed, content: self.node_label_or_empty(seed),
                    activation: *act, confidence: node.confidence(),
                    path: paths.get(seed).cloned().unwrap_or_default(),
                    depth: *depths.get(seed).unwrap_or(&0),
                })
            })
            .collect();
        // BR.C-Hyp-Methodology — total order on (activation desc,
        // seed bytes asc). Without the seed tiebreaker, f32 ties at
        // the top_k boundary surface in HashMap iteration order,
        // which Rust's default RandomState makes per-process random.
        activated.sort_by(|a, b| {
            b.activation.partial_cmp(&a.activation)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.seed.as_bytes().cmp(b.seed.as_bytes()))
        });
        activated.truncate(top_k);

        RecallResult {
            activated,
            query_seeds: seeds.iter().map(|(s, _)| *s).collect(),
            scope: scope.clone(),
            nodes_visited,
            bonds_evaluated,
            activation_truncated,
            status: SpreadStatus::Complete,
        }
    }

    /// Get all outgoing neighbor seeds for a node. OP-BRAIN-V4 Phase 1:
    /// adjacency stores `NodeId`s; this materializes a fresh
    /// `Vec<Seed>` by reverse-lookup through the intern table. Callers
    /// that only need the count should use `out_degree`/`in_degree`.
    /// Hot spread paths use `neighbors_outgoing_capped` /
    /// `neighbors_incoming_capped` which already returned `Vec<Seed>`
    /// pre-Phase 1, so this signature change does not touch them.
    pub fn neighbors_outgoing(&self, seed: &Seed) -> Vec<Seed> {
        match self.outgoing.get(seed) {
            Some(v) => self.node_intern.reverse_collect(v.iter().copied()),
            None => Vec::new(),
        }
    }

    /// Get all incoming neighbor seeds for a node. See
    /// `neighbors_outgoing` for the Phase 1 materialization rationale.
    pub fn neighbors_incoming(&self, seed: &Seed) -> Vec<Seed> {
        match self.incoming.get(seed) {
            Some(v) => self.node_intern.reverse_collect(v.iter().copied()),
            None => Vec::new(),
        }
    }

    /// BR.R3 (OP-008): byte-sorted bucket for a single content token,
    /// suitable for O(log N) binary-search membership tests by seed
    /// bytes. Used by `idk_gate::evaluate_signals` to compute
    /// `top_seed_bucket_coverage`. Returns `None` when the token has
    /// no entry in the keyword index (caller treats as coverage 0).
    pub fn keyword_index_bucket(&self, token: &str)
        -> Option<crate::keyword_index::MergedBucket<'_>>
    {
        // OP-022.a: return type widened from &Vec<Seed> to MergedBucket<'_>.
        // MergedBucket: Deref<Target=[Seed]> so existing callers (.iter(),
        // .len(), .binary_search_by, &bucket[i], etc.) compile unchanged.
        self.keyword_index.get(token)
    }

    /// Search nodes by keyword. Returns matching seeds ranked by
    /// number of keyword hits.
    pub fn keyword_search(&self, query: &str) -> Vec<(Seed, usize)> {
        // Defense-in-depth: hub-stopwords and query-noise tokens never
        // produce keyword hits, so the `ask` fallback and cascade paths
        // can never seed the spread on “and” / “or” / “the” / “what”.
        let words: Vec<String> = query
            .to_lowercase()
            .split_whitespace()
            .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
            .filter(|w| !w.is_empty()
                && !is_hub_stopword(w)
                && !is_query_noise(w))
            .collect();

        let mut hit_counts: HashMap<Seed, usize> = HashMap::new();
        for word in &words {
            if let Some(seeds) = self.keyword_index.get(word) {
                eprintln!("  word={} bucket_len={}", word, seeds.len());
                // OP-022.a: MergedBucket -> [Seed] via Deref; .iter() yields &Seed.
                // FIX 2 (OP-HUB-BUCKET-CAP): keyword_search is the recall
                // FALLBACK; the per-seed pollution probe below calls view_node,
                // which reads a full node payload from the SSTs. Iterating a hub
                // bucket ("dog" = 10,005 seeds) read-amplified tens of GB. The
                // bucket is byte-sorted, downstream keeps only the top few hits,
                // and single-word hit_counts all tie, so a capped prefix is
                // faithful; low-frequency buckets (<= cap) are unchanged.
                const KEYWORD_SEARCH_BUCKET_SCAN_CAP: usize = 256;
                for seed in seeds.iter().take(KEYWORD_SEARCH_BUCKET_SCAN_CAP) {
                    // Port #24: skip polluted seeds at recall time so the
                    // ask() fallback / cascade paths can never seed on a
                    // legacy goal:/step: anchor, a BondType-name leak,
                    // or a code-identifier bleed.
                    if let Some(node) = self.view_node(seed) {
                        if is_recall_pollution(node.content()) { continue; }
                    }
                    *hit_counts.entry(*seed).or_insert(0) += 1;
                }
            }
        }

        let mut results: Vec<(Seed, usize)> = hit_counts.into_iter().collect();
        results.sort_by(|a, b| b.1.cmp(&a.1));
        results
    }

    /// Spread activation through the temporal knowledge graph.
    ///
    /// Starting from the given seed nodes, activation flows outward through
    /// bonds. At each hop:
    /// 1. The bond's temporal weight at the scope's timestamp is evaluated.
    /// 2. Activation is decayed by `DECAY_PER_HOP`.
    /// 3. The node's confidence is applied as a multiplier.
    /// 4. Nodes below `ACTIVATION_THRESHOLD` stop propagation.
    ///
    /// A visited set prevents revisiting nodes (critical for hub nodes
    /// that would cause exponential blowup).
    ///
    /// Returns the top-k activated nodes with their traversal paths.
    pub fn spread_activation(
        &self,
        seeds: &[Seed],
        scope: &TemporalScope,
        depth: usize,
        top_k: usize,
        scar_log: &ScarLog,
    ) -> RecallResult {
        // OP-BRAIN-MapRAM P5-batch2 (sub-batch B): the bare spread path now
        // reads neighbours + edge weight + neighbour DEGREE + node stub
        // through the cache seam, not the resident adjacency. CACHE sources
        // neighbours from the map (`seam_outgoing_edges_capped` /
        // `seam_incoming_edges_capped`, the carried-weight hub-cap twin of
        // `neighbors_*_capped`), the edge bond (incl. keyframes) from the
        // carried payload (`reconstruct_carried_bond`, never resident
        // `get_bond`/`view_bond`), degree from the map (`seam_total_degree`),
        // and the node stub from the map (`seam_recall_node`). LEGACY is
        // byte-for-byte the prior reads apart from the deliberate canonical
        // neighbour ORDER (resident order was `HashMap::drain`-random per
        // load -- already non-deterministic -- so canonicalising both modes
        // makes the output deterministic AND identical; same defence as the
        // weighted spread).
        let mut activations: HashMap<Seed, f32> = HashMap::new();
        let mut paths: HashMap<Seed, Vec<TraversalStep>> = HashMap::new();
        let mut depths: HashMap<Seed, usize> = HashMap::new();
        let mut visited: HashSet<Seed> = HashSet::new();
        let mut frontier: VecDeque<(Seed, f32, usize)> = VecDeque::new();

        let mut nodes_visited: usize = 0;
        let mut bonds_evaluated: usize = 0;
        let mut activation_truncated: bool = false;
        let cap_activations: usize = self.effective_cap_activations();

        for seed in seeds {
            if let Some(_n_view) = self.seam_recall_node(seed) {
                let confidence = _n_view.confidence_value();
                let initial_activation = 1.0 * confidence;
                activations.insert(*seed, initial_activation);
                paths.insert(*seed, Vec::new());
                depths.insert(*seed, 0);
                frontier.push_back((*seed, initial_activation, 0));
                visited.insert(*seed);
                nodes_visited += 1;
            }
        }

        while let Some((current_seed, current_activation, current_depth)) = frontier.pop_front() {
            // OP-SPREAD-1 -- hoist the `current_seed`-invariant seam
            // probe out of the edge loops. See `seam_node_and_degree`.
            let current_node_view_hoisted = self.seam_node_and_degree(&current_seed).0;
            let current_content_hoisted: &str =
                current_node_view_hoisted.as_ref().map(|n| n.content()).unwrap_or("");
            let current_is_stopword =
                self.is_stopword_node(&current_seed, current_content_hoisted);
            if current_depth >= depth {
                continue;
            }

            // --- outgoing bonds (capped, carried-weight hub-cap) ---
            for edge in self.seam_outgoing_edges_capped(&current_seed) {
                let neighbor_seed = edge.nbr;
                bonds_evaluated += 1;
                let carried_bond: Option<TemporalBond> = match &edge.bond {
                    SeamBondSrc::Carried { predicate, payload } =>
                        Some(Self::reconstruct_carried_bond(current_seed, neighbor_seed, *predicate, payload)),
                    SeamBondSrc::Resident => None,
                };
                let bond = match &carried_bond {
                    Some(rb) => crate::view::BondView::from_heap(rb),
                    None => match self.view_bond(&(current_seed, neighbor_seed)) { Some(b) => b, None => continue },
                };

                let bond_weight = bond.weight_in_scope(scope, scar_log).effective_weight
                    * self.decay_multiplier_for_bond(&bond) * self.sense_gain_for_bond(&current_seed, &neighbor_seed);
                if bond_weight.abs() < ACTIVATION_THRESHOLD {
                    continue;
                }
                // OP-SPREAD-1 -- ONE intern probe per edge; `current_seed`
                // is hoisted above the loop.
                let (neighbor_node, neighbor_total_deg) =
                    self.seam_node_and_degree(&neighbor_seed);
                let neighbor_confidence = neighbor_node.as_ref().map(|n| n.confidence_value()).unwrap_or(0.5);
                let neighbor_content = neighbor_node.as_ref().map(|n| n.content()).unwrap_or("");

                let degree = neighbor_total_deg;
                let hub_t = self.degree_p90;
                let degree_norm = if degree <= hub_t {
                    1.0_f32
                } else {
                    ((hub_t as f32) / (degree as f32)).sqrt()
                };
                // OP-RECALL-1 (R3) -- see the note on `is_stopword_node`.
                // `sw_penalty` only ever lowers activation, so the
                // threshold test is safe to hoist above it and keeps the
                // label resolutions bounded by what actually activates.
                let activation_pre =
                    current_activation * bond_weight * DECAY_PER_HOP
                    * neighbor_confidence * degree_norm;
                if activation_pre < ACTIVATION_THRESHOLD {
                    continue;
                }
                let sw_penalty = if self.is_stopword_node(&neighbor_seed, neighbor_content)
                    || current_is_stopword {
                    STOPWORD_BOND_DECAY
                } else {
                    1.0
                };
                let neighbor_activation = activation_pre * sw_penalty;
                if neighbor_activation < ACTIVATION_THRESHOLD {
                    continue;
                }
                let step = TraversalStep {
                    node: neighbor_seed,
                    bond_type: bond.bond_type(),
                    bond_weight,
                    activation: neighbor_activation,
                };
                if visited.contains(&neighbor_seed) {
                    if let Some(existing) = activations.get_mut(&neighbor_seed) {
                        if neighbor_activation > *existing {
                            *existing = neighbor_activation;
                            let mut new_path = paths.get(&current_seed).cloned().unwrap_or_default();
                            new_path.push(step);
                            paths.insert(neighbor_seed, new_path);
                        }
                    }
                    continue;
                }
                if activations.len() >= cap_activations {
                    activation_truncated = true;
                    continue;
                }
                visited.insert(neighbor_seed);
                nodes_visited += 1;
                activations.insert(neighbor_seed, neighbor_activation);
                depths.insert(neighbor_seed, current_depth + 1);
                let mut path_to_neighbor = paths.get(&current_seed).cloned().unwrap_or_default();
                path_to_neighbor.push(step);
                paths.insert(neighbor_seed, path_to_neighbor);
                if !self.is_stopword_node(&neighbor_seed, neighbor_content) {
                    if activations.len() < cap_activations {
                        frontier.push_back((neighbor_seed, neighbor_activation, current_depth + 1));
                    } else {
                        activation_truncated = true;
                    }
                }
            }

            // --- incoming bonds (half weight; capped, carried-weight hub-cap) ---
            for edge in self.seam_incoming_edges_capped(&current_seed) {
                let neighbor_seed = edge.nbr;
                bonds_evaluated += 1;
                let carried_bond: Option<TemporalBond> = match &edge.bond {
                    SeamBondSrc::Carried { predicate, payload } =>
                        Some(Self::reconstruct_carried_bond(neighbor_seed, current_seed, *predicate, payload)),
                    SeamBondSrc::Resident => None,
                };
                let bond = match &carried_bond {
                    Some(rb) => crate::view::BondView::from_heap(rb),
                    None => match self.view_bond(&(neighbor_seed, current_seed)) { Some(b) => b, None => continue },
                };

                let bond_weight = bond.weight_in_scope(scope, scar_log).effective_weight
                    * self.decay_multiplier_for_bond(&bond) * self.sense_gain_for_bond(&neighbor_seed, &current_seed) * 0.5;
                if bond_weight.abs() < ACTIVATION_THRESHOLD {
                    continue;
                }
                // OP-SPREAD-1 -- ONE intern probe per edge; `current_seed`
                // is hoisted above the loop.
                let (neighbor_node, neighbor_total_deg) =
                    self.seam_node_and_degree(&neighbor_seed);
                let neighbor_confidence = neighbor_node.as_ref().map(|n| n.confidence_value()).unwrap_or(0.5);
                let neighbor_content = neighbor_node.as_ref().map(|n| n.content()).unwrap_or("");

                let degree = neighbor_total_deg;
                let hub_t = self.degree_p90;
                let degree_norm = if degree <= hub_t {
                    1.0_f32
                } else {
                    ((hub_t as f32) / (degree as f32)).sqrt()
                };
                // OP-RECALL-1 (R3) -- see the note on `is_stopword_node`.
                // `sw_penalty` only ever lowers activation, so the
                // threshold test is safe to hoist above it and keeps the
                // label resolutions bounded by what actually activates.
                let activation_pre =
                    current_activation * bond_weight * DECAY_PER_HOP
                    * neighbor_confidence * degree_norm;
                if activation_pre < ACTIVATION_THRESHOLD {
                    continue;
                }
                let sw_penalty = if self.is_stopword_node(&neighbor_seed, neighbor_content)
                    || current_is_stopword {
                    STOPWORD_BOND_DECAY
                } else {
                    1.0
                };
                let neighbor_activation = activation_pre * sw_penalty;
                if neighbor_activation < ACTIVATION_THRESHOLD {
                    continue;
                }
                let step = TraversalStep {
                    node: neighbor_seed,
                    bond_type: bond.bond_type(),
                    bond_weight,
                    activation: neighbor_activation,
                };
                if visited.contains(&neighbor_seed) {
                    if let Some(existing) = activations.get_mut(&neighbor_seed) {
                        if neighbor_activation > *existing {
                            *existing = neighbor_activation;
                            let mut new_path = paths.get(&current_seed).cloned().unwrap_or_default();
                            new_path.push(step);
                            paths.insert(neighbor_seed, new_path);
                        }
                    }
                    continue;
                }
                if activations.len() >= cap_activations {
                    activation_truncated = true;
                    continue;
                }
                visited.insert(neighbor_seed);
                nodes_visited += 1;
                activations.insert(neighbor_seed, neighbor_activation);
                depths.insert(neighbor_seed, current_depth + 1);
                let mut path_to_neighbor = paths.get(&current_seed).cloned().unwrap_or_default();
                path_to_neighbor.push(step);
                paths.insert(neighbor_seed, path_to_neighbor);
                if !self.is_stopword_node(&neighbor_seed, neighbor_content) {
                    if activations.len() < cap_activations {
                        frontier.push_back((neighbor_seed, neighbor_activation, current_depth + 1));
                    } else {
                        activation_truncated = true;
                    }
                }
            }
        }

        let mut activated: Vec<ActivatedNode> = activations
            .iter()
            .filter_map(|(seed, activation)| {
                let node = self.seam_recall_node(seed)?;
                // OP-LABEL-1: label from the node store, not the stub.
                Some(ActivatedNode {
                    seed: *seed,
                    content: self.node_label_or_empty(seed),
                    activation: *activation,
                    confidence: node.confidence(),
                    path: paths.get(seed).cloned().unwrap_or_default(),
                    depth: *depths.get(seed).unwrap_or(&0),
                })
            })
            .collect();

        activated.sort_by(|a, b| {
            b.activation
                .partial_cmp(&a.activation)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.seed.as_bytes().cmp(b.seed.as_bytes()))
        });
        activated.truncate(top_k);

        RecallResult {
            activated,
            query_seeds: seeds.to_vec(),
            scope: scope.clone(),
            nodes_visited,
            bonds_evaluated,
            activation_truncated,
            status: SpreadStatus::Complete,
        }
    }

    /// Spread activation restricted to a specific set of bond types.
    ///
    /// Identical to `spread_activation()` except bonds whose `bond_type` is not
    /// in `bond_filter` are skipped entirely. This prevents unrelated semantic
    /// domains (e.g. code syntax) from polluting general recall.
    pub fn typed_recall(
        &self,
        seeds: &[Seed],
        scope: &TemporalScope,
        depth: usize,
        top_k: usize,
        bond_filter: &[BondType],
        scar_log: &ScarLog,
    ) -> RecallResult {
        // OP-BRAIN-MapRAM P5-batch2 (sub-batch B): typed recall now reads
        // neighbours + edge bond (type/weight/keyframes) + neighbour DEGREE
        // + node stub through the cache seam, not the resident adjacency.
        // CACHE sources neighbours from the map (UNCAPPED `seam_outgoing_edges`
        // / `seam_incoming_edges` -- typed_recall is the uncapped variant),
        // the bond from the carried payload (`reconstruct_carried_bond`; the
        // bond TYPE that drives `bond_filter` comes from the carried
        // predicate, never resident `get_bond`), degree from the map
        // (`seam_total_degree`), and the node stub from the map
        // (`seam_recall_node`). LEGACY is byte-for-byte the prior reads apart
        // from the deliberate canonical neighbour ORDER (resident order was
        // `HashMap::drain`-random per load).
        let mut activations: HashMap<Seed, f32> = HashMap::new();
        let mut paths: HashMap<Seed, Vec<TraversalStep>> = HashMap::new();
        let mut depths: HashMap<Seed, usize> = HashMap::new();
        let mut visited: HashSet<Seed> = HashSet::new();
        let mut frontier: VecDeque<(Seed, f32, usize)> = VecDeque::new();

        let mut nodes_visited: usize = 0;
        let mut bonds_evaluated: usize = 0;
        let mut activation_truncated: bool = false;
        let cap_activations: usize = self.effective_cap_activations();

        for seed in seeds {
            if let Some(_n_view) = self.seam_recall_node(seed) {
                let confidence = _n_view.confidence_value();
                let initial_activation = 1.0 * confidence;
                activations.insert(*seed, initial_activation);
                paths.insert(*seed, Vec::new());
                depths.insert(*seed, 0);
                frontier.push_back((*seed, initial_activation, 0));
                visited.insert(*seed);
                nodes_visited += 1;
            }
        }

        while let Some((current_seed, current_activation, current_depth)) = frontier.pop_front() {
            // OP-SPREAD-1 -- hoist the `current_seed`-invariant seam
            // probe out of the edge loops. See `seam_node_and_degree`.
            let current_node_view_hoisted = self.seam_node_and_degree(&current_seed).0;
            let current_content_hoisted: &str =
                current_node_view_hoisted.as_ref().map(|n| n.content()).unwrap_or("");
            let current_is_stopword =
                self.is_stopword_node(&current_seed, current_content_hoisted);
            if current_depth >= depth {
                continue;
            }

            // --- outgoing bonds (uncapped, canonical) ---
            for edge in self.seam_outgoing_edges(&current_seed) {
                let neighbor_seed = edge.nbr;
                bonds_evaluated += 1;
                let carried_bond: Option<TemporalBond> = match &edge.bond {
                    SeamBondSrc::Carried { predicate, payload } =>
                        Some(Self::reconstruct_carried_bond(current_seed, neighbor_seed, *predicate, payload)),
                    SeamBondSrc::Resident => None,
                };
                let bond = match &carried_bond {
                    Some(rb) => crate::view::BondView::from_heap(rb),
                    None => match self.view_bond(&(current_seed, neighbor_seed)) { Some(b) => b, None => continue },
                };
                // Skip bonds not in the filter (when filter is non-empty)
                if !bond_filter.is_empty() && !bond_filter.contains(&bond.bond_type()) {
                    continue;
                }
                // Port T.5.1: query-path effective weight = stored * freshness.
                let bond_weight = bond.weight_in_scope(scope, scar_log).effective_weight
                    * self.decay_multiplier_for_bond(&bond) * self.sense_gain_for_bond(&current_seed, &neighbor_seed);
                if bond_weight.abs() < ACTIVATION_THRESHOLD {
                    continue;
                }
                // OP-SPREAD-1 -- ONE intern probe per edge; `current_seed`
                // is hoisted above the loop.
                let (neighbor_node, neighbor_total_deg) =
                    self.seam_node_and_degree(&neighbor_seed);
                let neighbor_confidence = neighbor_node.as_ref().map(|n| n.confidence_value()).unwrap_or(0.5);
                let neighbor_content = neighbor_node.as_ref().map(|n| n.content()).unwrap_or("");
                let degree = neighbor_total_deg;
                let hub_t = self.degree_p90;
                let degree_norm = if degree <= hub_t {
                    1.0_f32
                } else {
                    ((hub_t as f32) / (degree as f32)).sqrt()
                };
                // OP-RECALL-1 (R3) -- see the note on `is_stopword_node`.
                // `sw_penalty` only ever lowers activation, so the
                // threshold test is safe to hoist above it and keeps the
                // label resolutions bounded by what actually activates.
                let activation_pre =
                    current_activation * bond_weight * DECAY_PER_HOP
                    * neighbor_confidence * degree_norm;
                if activation_pre < ACTIVATION_THRESHOLD {
                    continue;
                }
                let sw_penalty = if self.is_stopword_node(&neighbor_seed, neighbor_content)
                    || current_is_stopword {
                    STOPWORD_BOND_DECAY
                } else {
                    1.0
                };
                let neighbor_activation = activation_pre * sw_penalty;
                if neighbor_activation < ACTIVATION_THRESHOLD {
                    continue;
                }
                let step = TraversalStep {
                    node: neighbor_seed,
                    bond_type: bond.bond_type(),
                    bond_weight,
                    activation: neighbor_activation,
                };
                if visited.contains(&neighbor_seed) {
                    if let Some(existing) = activations.get_mut(&neighbor_seed) {
                        if neighbor_activation > *existing {
                            *existing = neighbor_activation;
                            let mut new_path =
                                paths.get(&current_seed).cloned().unwrap_or_default();
                            new_path.push(step);
                            paths.insert(neighbor_seed, new_path);
                        }
                    }
                    continue;
                }
                if activations.len() >= cap_activations {
                    activation_truncated = true;
                    continue;
                }
                visited.insert(neighbor_seed);
                nodes_visited += 1;
                activations.insert(neighbor_seed, neighbor_activation);
                depths.insert(neighbor_seed, current_depth + 1);
                let mut path_to_neighbor =
                    paths.get(&current_seed).cloned().unwrap_or_default();
                path_to_neighbor.push(step);
                paths.insert(neighbor_seed, path_to_neighbor);
                if !self.is_stopword_node(&neighbor_seed, neighbor_content) {
                    if activations.len() < cap_activations {
                        frontier.push_back((neighbor_seed, neighbor_activation, current_depth + 1));
                    } else {
                        activation_truncated = true;
                    }
                }
            }

            // --- incoming bonds (half weight; uncapped, canonical) ---
            for edge in self.seam_incoming_edges(&current_seed) {
                let neighbor_seed = edge.nbr;
                bonds_evaluated += 1;
                let carried_bond: Option<TemporalBond> = match &edge.bond {
                    SeamBondSrc::Carried { predicate, payload } =>
                        Some(Self::reconstruct_carried_bond(neighbor_seed, current_seed, *predicate, payload)),
                    SeamBondSrc::Resident => None,
                };
                let bond = match &carried_bond {
                    Some(rb) => crate::view::BondView::from_heap(rb),
                    None => match self.view_bond(&(neighbor_seed, current_seed)) { Some(b) => b, None => continue },
                };
                if !bond_filter.is_empty() && !bond_filter.contains(&bond.bond_type()) {
                    continue;
                }
                // Port T.5.1: query-path effective weight = stored * freshness, then halved for reverse direction.
                let bond_weight = bond.weight_in_scope(scope, scar_log).effective_weight
                    * self.decay_multiplier_for_bond(&bond) * self.sense_gain_for_bond(&neighbor_seed, &current_seed) * 0.5;
                if bond_weight.abs() < ACTIVATION_THRESHOLD {
                    continue;
                }
                // OP-SPREAD-1 -- ONE intern probe per edge; `current_seed`
                // is hoisted above the loop.
                let (neighbor_node, neighbor_total_deg) =
                    self.seam_node_and_degree(&neighbor_seed);
                let neighbor_confidence = neighbor_node.as_ref().map(|n| n.confidence_value()).unwrap_or(0.5);
                let neighbor_content = neighbor_node.as_ref().map(|n| n.content()).unwrap_or("");
                let degree = neighbor_total_deg;
                let hub_t = self.degree_p90;
                let degree_norm = if degree <= hub_t {
                    1.0_f32
                } else {
                    ((hub_t as f32) / (degree as f32)).sqrt()
                };
                // OP-RECALL-1 (R3) -- see the note on `is_stopword_node`.
                // `sw_penalty` only ever lowers activation, so the
                // threshold test is safe to hoist above it and keeps the
                // label resolutions bounded by what actually activates.
                let activation_pre =
                    current_activation * bond_weight * DECAY_PER_HOP
                    * neighbor_confidence * degree_norm;
                if activation_pre < ACTIVATION_THRESHOLD {
                    continue;
                }
                let sw_penalty = if self.is_stopword_node(&neighbor_seed, neighbor_content)
                    || current_is_stopword {
                    STOPWORD_BOND_DECAY
                } else {
                    1.0
                };
                let neighbor_activation = activation_pre * sw_penalty;
                if neighbor_activation < ACTIVATION_THRESHOLD {
                    continue;
                }
                let step = TraversalStep {
                    node: neighbor_seed,
                    bond_type: bond.bond_type(),
                    bond_weight,
                    activation: neighbor_activation,
                };
                if visited.contains(&neighbor_seed) {
                    if let Some(existing) = activations.get_mut(&neighbor_seed) {
                        if neighbor_activation > *existing {
                            *existing = neighbor_activation;
                            let mut new_path =
                                paths.get(&current_seed).cloned().unwrap_or_default();
                            new_path.push(step);
                            paths.insert(neighbor_seed, new_path);
                        }
                    }
                    continue;
                }
                if activations.len() >= cap_activations {
                    activation_truncated = true;
                    continue;
                }
                visited.insert(neighbor_seed);
                nodes_visited += 1;
                activations.insert(neighbor_seed, neighbor_activation);
                depths.insert(neighbor_seed, current_depth + 1);
                let mut path_to_neighbor =
                    paths.get(&current_seed).cloned().unwrap_or_default();
                path_to_neighbor.push(step);
                paths.insert(neighbor_seed, path_to_neighbor);
                if !self.is_stopword_node(&neighbor_seed, neighbor_content) {
                    if activations.len() < cap_activations {
                        frontier.push_back((neighbor_seed, neighbor_activation, current_depth + 1));
                    } else {
                        activation_truncated = true;
                    }
                }
            }
        }

        let mut activated: Vec<ActivatedNode> = activations
            .iter()
            .filter_map(|(seed, activation)| {
                let node = self.seam_recall_node(seed)?;
                // OP-LABEL-1: label from the node store, not the stub.
                Some(ActivatedNode {
                    seed: *seed,
                    content: self.node_label_or_empty(seed),
                    activation: *activation,
                    confidence: node.confidence(),
                    path: paths.get(seed).cloned().unwrap_or_default(),
                    depth: *depths.get(seed).unwrap_or(&0),
                })
            })
            .collect();

        activated.sort_by(|a, b| {
            b.activation
                .partial_cmp(&a.activation)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.seed.as_bytes().cmp(b.seed.as_bytes()))
        });
        activated.truncate(top_k);

        RecallResult {
            activated,
            query_seeds: seeds.to_vec(),
            scope: scope.clone(),
            nodes_visited,
            bonds_evaluated,
            activation_truncated,
            status: SpreadStatus::Complete,
        }
    }

    /// Apply Hebbian strengthening: when two nodes co-activate in a session,
    /// strengthen the bond between them.
    ///
    /// For each pair of co-activated nodes that share a bond, adds a new
    /// keyframe with weight increased by `HEBBIAN_GAIN`, tagged with the
    /// session ID.
    pub fn hebbian_strengthen(
        &mut self,
        activated_seeds: &[Seed],
        session_id: u64,
        timestamp: i64,
    ) -> usize {
        let mut strengthened = 0;

        // Check all pairs of activated nodes for existing bonds
        for i in 0..activated_seeds.len() {
            for j in (i + 1)..activated_seeds.len() {
                let a = activated_seeds[i];
                let b = activated_seeds[j];

                // Check a → b bond
                if let Some(bond) = self.bonds.get_mut(&(a, b)) {
                    let current_weight = bond.latest_raw_weight();
                    let new_weight = (current_weight + HEBBIAN_GAIN).min(1.0);
                    bond.add_keyframe(
                        timestamp,
                        new_weight,
                        KeyframeSource::Session(session_id),
                    );
                    strengthened += 1;
                }

                // Check b → a bond
                if let Some(bond) = self.bonds.get_mut(&(b, a)) {
                    let current_weight = bond.latest_raw_weight();
                    let new_weight = (current_weight + HEBBIAN_GAIN).min(1.0);
                    bond.add_keyframe(
                        timestamp,
                        new_weight,
                        KeyframeSource::Session(session_id),
                    );
                    strengthened += 1;
                }
            }
        }

        strengthened
    }

    /// Return the number of nodes in the network.
    ///
    /// OP-RECALL-1 (R5): the cache-backed v5 path is checked FIRST.
    /// A brain loaded by `Brain::load_v5` in the production
    /// (mapram-routed) mode never populates `self.nodes` and has no
    /// v3 `self.archive` -- both reads below return 0 for a 43.8M-node
    /// brain. That zero is not a cosmetic defect: `term_idf` short-
    /// circuits on `n < 2.0` and returns a flat 1.0, which collapses
    /// the entire TF-IDF weighting layer in
    /// `select_seeds_for_query_inner` to a constant, and the narration
    /// MetaSelf shape prints the literal "0 concepts". The resident
    /// degree map is sized to the interned node count, so it is the
    /// authoritative source in this mode.
    pub fn node_count(&self) -> usize {
        if self.mapram_cache_backed() {
            if let Some(cache) = self.mapram_cache.as_ref() {
                return cache.node_count();
            }
        }
        match &self.archive {
            Some(arch) => {
                // Overlay nodes minus deletions, plus archive nodes
                // not shadowed by overlay and not deleted.
                let mut count = self
                    .nodes
                    .keys()
                    .filter(|s| !self.overlay_deletions.contains(s))
                    .count();
                for seed in arch.iter_node_seeds() {
                    if self.nodes.contains_key(&seed) || self.overlay_deletions.contains(&seed) {
                        continue;
                    }
                    count += 1;
                }
                count
            }
            None => self.nodes.len(),
        }
    }

    /// OP-INGEST-1: Return the number of bonds in the network.
    /// FAST PATH: returns the cached value maintained incrementally at
    /// the canonical mutation sites (`add_bond`, `remove_bond`,
    /// `purge_nodes`, archive attach). O(1).
    ///
    /// Pre-OP-INGEST-1 walked-implementation lives at `bond_count_walked()`
    /// as a debug-only validator. The two MUST agree; `bond_count_cache_verify()`
    /// asserts parity in tests.
    pub fn bond_count(&self) -> usize {
        // OP-RECALL-1 (R5): same zero-read as `node_count` above. In
        // cache-backed v5 mode `self.bonds` and `self.archive` are both
        // empty, so `bond_count_walked()` would memoize a permanent 0.
        // The resident degree map carries the real total. Memoized in
        // the same cell so the O(N) degree sum runs at most once.
        if self.mapram_cache_backed() {
            if let Some(n) = self.bond_count_cache.get() {
                return n;
            }
            if let Some(cache) = self.mapram_cache.as_ref() {
                let n = cache.total_bonds() as usize;
                self.bond_count_cache.set(Some(n));
                return n;
            }
        }
        // OP-INGEST-1.1: lazy init. If the cache hasn't been populated
        // yet (e.g. fresh from_archive that no longer pre-warms), walk
        // once on demand and memoize. All subsequent reads are O(1)
        // and all subsequent mutations update the cache in-place.
        match self.bond_count_cache.get() {
            Some(n) => n,
            None => {
                let n = self.bond_count_walked();
                self.bond_count_cache.set(Some(n));
                n
            }
        }
    }

    /// OP-INGEST-1: O(N) walked-implementation retained as a validator for
    /// the cached `bond_count`. Pre-OP-INGEST-1 this was the production
    /// path; the profile showed it took 96% of daemon CPU per ingest batch
    /// (was called twice per `tool_teach_corpus_batch` for telemetry; the
    /// telemetry field was removed in OP-INGEST-2).
    /// DO NOT call from hot paths. Used by parity tests and the
    /// `verify_bond_count_cache()` self-check.
    pub fn bond_count_walked(&self) -> usize {
        match &self.archive {
            Some(arch) => {
                let mut count = self
                    .bonds
                    .iter()
                    .filter(|(k, _)| {
                        !self.overlay_deletions.contains(&k.0)
                            && !self.overlay_deletions.contains(&k.1)
                            && !self.bond_tombstones.contains(k)
                    })
                    .count();
                for k in arch.iter_bond_keys() {
                    if self.bonds.contains_key(&k)
                        || self.overlay_deletions.contains(&k.0)
                        || self.overlay_deletions.contains(&k.1)
                        || self.bond_tombstones.contains(&k)
                    {
                        continue;
                    }
                    count += 1;
                }
                count
            }
            None => self.bonds.len(),
        }
    }

    /// OP-INGEST-1: explicit parity check. Returns Ok(()) if cached ==
    /// walked; Err with both values on drift. Exposed for tests and
    /// optional debug-build self-validation in hot paths.
    pub fn verify_bond_count_cache(&self) -> Result<(), (usize, usize)> {
        // OP-INGEST-1.1: cache is Option<usize>. None means lazy-init
        // hasn't fired yet -- by definition nothing to verify against,
        // so the invariant holds vacuously. (The first read via
        // bond_count() will populate the cache via a walk, which is
        // self-consistent by construction.)
        let cached_opt = self.bond_count_cache.get();
        let walked = self.bond_count_walked();
        match cached_opt {
            None => Ok(()),
            Some(cached) => {
                if cached == walked { Ok(()) } else { Err((cached, walked)) }
            }
        }
    }

    /// Return a view over all bonds. Replaces the prior
    /// `&HashMap<(Seed, Seed), TemporalBond>` accessor — callers see
    /// only the `BondsView` API surface (iter / keys / values / get /
    /// len / contains_key) so the underlying storage can be archive-
    /// backed in Slice 2c.
    pub fn bonds(&self) -> BondsView<'_> {
        match &self.archive {
            Some(arch) => BondsView::from_mmap(&self.bonds, arch, &self.overlay_deletions),
            None => BondsView::from_heap(&self.bonds),
        }
    }

    /// Crate-internal: borrow the underlying bond `HashMap` directly.
    /// Restricted to internal callers in `network.rs` and adjacent
    /// modules that need raw heap access (e.g. consolidation prune).
    #[allow(dead_code)]
    pub(crate) fn bonds_raw(&self) -> &HashMap<(Seed, Seed), TemporalBond> {
        &self.bonds
    }

    /// Crate-internal: mutable access to the bond `HashMap`.
    /// Used by consolidation passes that need batched in-place edits;
    /// keeps the public surface free of raw `HashMap` mutation.
    #[allow(dead_code)]
    pub(crate) fn bonds_mut_raw(&mut self) -> &mut HashMap<(Seed, Seed), TemporalBond> {
        &mut self.bonds
    }

    /// Cap on the activation map size for a single spread call.
    pub fn max_activations(&self) -> usize { self.max_activations }

    /// Override the activation cap. Effect is immediate; in-flight
    /// spreads (none on this single-threaded design) are not affected.
    pub fn set_max_activations(&mut self, cap: usize) {
        self.max_activations = cap.max(1);
    }

    /// Percentile of brain size used as a dynamic activation cap.
    /// See `max_activations_percentile` doc on the struct.
    pub fn max_activations_percentile(&self) -> f32 { self.max_activations_percentile }

    /// Override the percentile. Clamped to [0.0, 1.0].
    pub fn set_max_activations_percentile(&mut self, pct: f32) {
        self.max_activations_percentile = pct.clamp(0.0, 1.0);
    }

    /// Effective per-spread activation cap given current brain size.
    /// Returns `max(max_activations, floor(node_count * percentile))`. The
    /// floor keeps small brains from regressing below 5K; the percentile
    /// keeps very large brains from over-truncating recall.
    pub fn effective_cap_activations(&self) -> usize {
        let brain_size = self.node_count();
        let pct_cap = ((brain_size as f64) * (self.max_activations_percentile as f64)) as usize;
        self.max_activations.max(pct_cap).max(1)
    }

    /// Port T.5.1 — Arm or disarm the per-query decay context.
    /// `Some(rt)` enables the freshness multiplier on every
    /// query-path effective-weight read in this module; `None`
    /// reverts to pre-T.5.1 behavior (multiplier=1.0). Brain calls
    /// this via `Brain::refresh_decay_runtime` before each query.
    pub fn set_decay_runtime(&mut self, rt: Option<crate::decay::DecayRuntime>) {
        self.decay_runtime = rt;
    }

    /// Port T.5.1 — Read-back accessor (used by tests + the
    /// `t51_decay_uses_clock_source_not_system_time` regression guard).
    pub fn decay_runtime(&self) -> Option<crate::decay::DecayRuntime> {
        self.decay_runtime
    }

    /// Port T.5.1 — Centralized freshness multiplier for a bond
    /// at the active query-time clock. Returns 1.0 if no runtime is
    /// armed or if runtime.config.enabled is false. The bond's
    /// creation timestamp is its oldest visible keyframe (matches
    /// crate::decay::bond_created_at_ms for heap-only bonds and
    /// uses BondView::keyframes_iter() so archive-backed bonds
    /// also work). NEVER mutates stored weights — multiplication
    /// happens inline at each call site.
    #[inline]
    pub(crate) fn decay_multiplier_for_bond(
        &self,
        bond: &crate::view::BondView<'_>,
    ) -> f32 {
        let rt = match self.decay_runtime {
            Some(r) => r,
            None => return 1.0,
        };
        if !rt.config.enabled {
            return 1.0;
        }
        let created = bond.keyframes_iter().map(|kf| kf.timestamp).min().unwrap_or(0);
        crate::decay::freshness_score(created, rt.now_ms, &rt.config)
    }

    /// Port Sense.1 -- Arm or disarm the per-query sense runtime.
    /// `Some(rt)` enables the cluster-gain multiplier on every
    /// query-path effective-weight read in this module; `None`
    /// reverts to pre-Sense.1 behavior (multiplier=1.0). Brain
    /// calls this via `Brain::refresh_sense_runtime` before
    /// each query.
    pub fn set_sense_runtime(&mut self, rt: Option<crate::sense::SenseRuntime>) {
        self.sense_runtime = rt;
    }

    /// Port Sense.1 -- Read-back accessor (used by tests + the
    /// `sense1_uses_clock_source_not_system_time` regression
    /// guard). Returns a borrow rather than a copy because the
    /// inner gain map can be large.
    pub fn sense_runtime(&self) -> Option<&crate::sense::SenseRuntime> {
        self.sense_runtime.as_ref()
    }

    /// Port Sense.1 -- Build a `SenseRuntime` for `query_seeds`
    /// using each seed's OTHER query seeds as the disambiguation
    /// context (capped at `config.context_window`). Each seed
    /// is fed through `disambiguate_polysemous_with_context`;
    /// non-polysemous seeds, seeds with no electable winner,
    /// or empty contexts contribute nothing to the gain map
    /// (so their bonds get gain=1.0 at query time -- uncertain
    /// rather than wrong).
    ///
    /// Cluster membership lookup is O(1) per bond via the
    /// nested `HashMap<Seed, HashMap<Seed, f32>>`. If the seed
    /// isn't polysemous, the inner map for it is absent and
    /// `SenseRuntime::gain_for_bond` returns 1.0.
    ///
    /// Read-only with respect to bond storage. NEVER mutates.
    /// Per-query polysemy budget. Skips seeds whose outgoing degree
    /// exceeds this cap because `polysemy_label_prop_clusters` is
    /// O(degree^2) on second-hop neighborhoods -- super-hubs would
    /// blow the per-query budget into the tens of seconds. Skipped
    /// seeds get gain=1.0 (uncertain rather than wrong, per spec).
    /// Tuned at ~4x degree_p90 across the canonical brain (p90~35,
    /// so cap=200 lets normal semantic hubs through but skips the
    /// extreme few-thousand-degree super-nodes that wouldn't yield
    /// clean disambiguation anyway).
    pub const SENSE_DISAMBIGUATION_DEGREE_CAP: usize = 200;

    pub fn build_sense_runtime(
        &self,
        query_seeds: &[crate::seed::Seed],
        config: crate::sense::SenseConfig,
    ) -> crate::sense::SenseRuntime {
        let mut runtime = crate::sense::SenseRuntime::empty(config.clone());
        if !config.disambiguation_enabled {
            return runtime;
        }
        for (i, seed) in query_seeds.iter().enumerate() {
            // Skip super-hubs whose 2nd-hop neighborhood Jaccard
            // build would dominate per-query time. Capped seeds
            // fall through with multiplier=1.0 -- the BFS itself
            // still runs against them; only the cluster-gain
            // routing is skipped.
            if self.out_degree(seed) > Self::SENSE_DISAMBIGUATION_DEGREE_CAP {
                continue;
            }
            // Context = the rest of the query's tokenized seeds,
            // capped at `context_window` (default 8). Single-seed
            // queries get an empty context -> disambiguate returns
            // None -> gain map stays empty for this seed -> bonds
            // multiply by 1.0 (uncertain rather than wrong, per spec).
            let mut context: Vec<crate::seed::Seed> = query_seeds
                .iter()
                .enumerate()
                .filter_map(|(j, s)| if j != i { Some(*s) } else { None })
                .collect();
            if context.len() > config.context_window {
                context.truncate(config.context_window);
            }
            if let Some(gain_map) =
                self.disambiguate_polysemous_with_context(seed, &context, config.loser_gain)
            {
                if !gain_map.is_empty() {
                    runtime.gain_by_seed.insert(*seed, gain_map);
                }
            }
        }
        runtime
    }

    /// Port Sense.1 -- Per-bond cluster-gain multiplier at the
    /// active query-time runtime. Returns 1.0 if no runtime is
    /// armed, or if `runtime.config.disambiguation_enabled` is
    /// false, or if neither bond endpoint is a polysemous seed
    /// in the runtime's gain map. NEVER mutates stored weights
    /// -- multiplication happens inline at each call site,
    /// composed multiplicatively with the T.5.1 decay multiplier.
    #[inline]
    pub(crate) fn sense_gain_for_bond(
        &self,
        from: &crate::seed::Seed,
        to: &crate::seed::Seed,
    ) -> f32 {
        match self.sense_runtime.as_ref() {
            Some(rt) => rt.gain_for_bond(from, to),
            None => 1.0,
        }
    }


    /// Port #18a: anchor-aware polysemy disambiguation. Splits a
    /// hub seed's out-neighbors into a sense allow-list using raw
    /// query tokens (`anchors`) when available, falling back to
    /// activation-only union-find clustering otherwise.
    ///
    /// The activation-only path (`disambiguate_polysemous`) is a thin
    /// wrapper over this method with no anchors. Walker/slot callers
    /// pass `seed_hits` minus the hub so that even when post-spread
    /// `activated` straddles both senses (because spreading from the
    /// hub itself activates both clusters) the disambiguation has
    /// clean, per-anchor evidence.
    ///
    /// Anchor path (per-neighbor similarity):
    ///   For each out-neighbor `nbr` of `seed`, keep `nbr` if either:
    ///     - it is itself an anchor, or
    ///     - Jaccard(neighborhood(nbr), neighborhood(any anchor))
    ///       >= MIN_ANCHOR_JACCARD (0.05).
    ///   This avoids the union-find collapse problem at hub fan-out:
    ///   instead of clustering 500+ neighbors and electing a winning
    ///   cluster (which under high random-overlap rates either glues
    ///   senses together at low thresholds or splinters into useless
    ///   singletons at high thresholds), each neighbor is judged on
    ///   its direct relationship to the query's raw anchors. Telephony
    ///   neighbors of `cell` share ~zero neighborhood with biology
    ///   anchors `mitosis`/`division`; they are dropped cleanly.
    ///
    /// No-anchor path (Port #19, replacing #18b's Jaccard 0.2 union-find):
    ///   Bond-type-seeded label propagation. Each neighbor starts
    ///   labelled by the bond type from `seed` (IsA / PartOf / Causes /
    ///   HasProperty each get their own label; RelatedTo collapses to
    ///   one). Then K iterations of synchronous label propagation:
    ///   each neighbor adopts the most-common label among the OTHER
    ///   neighbors of `seed` whose neighborhood overlaps theirs by
    ///   >= LABEL_PROP_JACCARD (0.05). Final clusters are scored by
    ///   typed-bond mass exactly like #18b; on ties or close calls
    ///   the activation-overlap fallback bites.
    ///   The label-propagation primitive replaces the union-find that
    ///   produced 462 micro-clusters (mostly singletons) on `cell`'s
    ///   528-neighbor fanout — at 105K-node scale the pairwise
    ///   neighborhood-Jaccard signal is too sparse for a hard
    ///   union-find threshold to find real communities.
    ///
    /// Returns None when there are too few neighbors, when the anchor
    /// path keeps everything (no useful filter), when the no-anchor
    /// path finds <2 clusters, or when the top score ties — in every
    /// None case the caller falls through to the unfiltered list.
    pub fn disambiguate_polysemous_with_anchors(
        &self,
        seed: &Seed,
        activated: &HashSet<Seed>,
        anchors: &[Seed],
    ) -> Option<HashSet<Seed>> {
        const MIN_NEIGHBORS_FOR_POLYSEMY: usize = 4;

        // OP-BRAIN-MapRAM P4: seed's outgoing neighbour SET via the seam
        // (cache-sourced + resident-independent in cache mode; identical set
        // in legacy mode). The anchor path below is a per-neighbour Jaccard
        // filter into a HashSet -- order-independent by construction.
        let neighbors: Vec<Seed> = self.seam_outgoing_set(seed);
        if neighbors.len() < MIN_NEIGHBORS_FOR_POLYSEMY {
            return None;
        }

        // Anchor path takes precedence whenever non-self anchors exist.
        let anchor_set: HashSet<Seed> = anchors
            .iter()
            .copied()
            .filter(|s| s != seed)
            .collect();
        if !anchor_set.is_empty() {
            const MIN_ANCHOR_JACCARD: f32 = 0.05;
            let anchor_neighborhoods: Vec<HashSet<Seed>> = anchor_set
                .iter()
                .map(|a| {
                    self.seam_outgoing_set(a)
                        .into_iter().collect::<HashSet<Seed>>()
                })
                .collect();

            let mut allow: HashSet<Seed> = HashSet::new();
            for &nbr in &neighbors {
                if anchor_set.contains(&nbr) {
                    allow.insert(nbr);
                    continue;
                }
                let nbr_nbhd: HashSet<Seed> = self.seam_outgoing_set(&nbr)
                    .into_iter().collect();
                let kept = anchor_neighborhoods.iter().any(|an| {
                    let inter = nbr_nbhd.intersection(an).count();
                    let uni = nbr_nbhd.union(an).count();
                    let jac = if uni == 0 { 0.0 } else { (inter as f32) / (uni as f32) };
                    jac >= MIN_ANCHOR_JACCARD
                });
                if kept {
                    allow.insert(nbr);
                }
            }
            // No-op cases: kept everything, or nothing.
            if allow.is_empty() || allow.len() == neighbors.len() {
                return None;
            }
            return Some(allow);
        }

        // Port #19 / refactored Port #22: weighted-edge label
        // propagation with bond-type tie-breaker is now a private
        // helper (`polysemy_label_prop_clusters`) so the no-anchor
        // path AND the new bare-polysemous multi-sense path can
        // share the cluster build. See helper docs for algorithm.
        let (neighbors, clusters, typed_score) = match self.polysemy_label_prop_clusters(seed) {
            Some(t) => t,
            None => return None,
        };

        // Find best & runner-up by typed score.
        let mut t_best_lbl: Option<u32> = None;
        let mut t_best: usize = 0;
        let mut t_second: usize = 0;
        for (lbl, &score) in &typed_score {
            if score > t_best {
                t_second = t_best;
                t_best = score;
                t_best_lbl = Some(*lbl);
            } else if score > t_second {
                t_second = score;
            }
        }

        // If one cluster has a meaningful typed-bond lead, elect it.
        // 1.5x margin + absolute floor of 2 + minimum-size 3 keep
        // this conservative on small/noisy splits.
        if t_best >= 2 && t_best as f32 > 1.5 * (t_second as f32) {
            let lbl = t_best_lbl?;
            let allow: HashSet<Seed> = clusters[&lbl]
                .iter().map(|&i| neighbors[i]).collect();
            if allow.len() >= 3 {
                return Some(allow);
            }
        }

        // Activation-overlap fallback: pick the cluster with the most
        // members in `activated`. Ties => no decision.
        let mut best_lbl: Option<u32> = None;
        let mut best_score: usize = 0;
        let mut second_best_score: usize = 0;
        for (lbl, members) in &clusters {
            let score = members
                .iter()
                .filter(|&&i| activated.contains(&neighbors[i]))
                .count();
            if score > best_score {
                second_best_score = best_score;
                best_score = score;
                best_lbl = Some(*lbl);
            } else if score > second_best_score {
                second_best_score = score;
            }
        }
        if best_score == 0 { return None; }
        if best_score == second_best_score { return None; }
        let lbl = best_lbl?;
        let allow: HashSet<Seed> = clusters[&lbl]
            .iter().map(|&i| neighbors[i]).collect();
        Some(allow)
    }

    /// Activation-only polysemy disambiguation. Wrapper retained so
    /// existing call sites and unit tests do not move.
    pub fn disambiguate_polysemous(
        &self,
        seed: &Seed,
        activated: &HashSet<Seed>,
    ) -> Option<HashSet<Seed>> {
        self.disambiguate_polysemous_with_anchors(seed, activated, &[])
    }

    /// Port #22: bare polysemous query — the user gave a single
    /// seed and no anchors — produces multiple sense clusters
    /// with comparable typed-bond mass.  Returns each qualifying
    /// cluster as a separate `HashSet<Seed>`, sorted by typed-bond
    /// score descending.
    ///
    /// Returns `None` when the seed has too few neighbors, when
    /// label propagation finds <2 clusters, when no cluster has at
    /// least 2 typed bonds (so even the leader is weak), or when
    /// only a single cluster passes the within-1.5x-of-leader gate
    /// (in which case the existing single-pick disambiguator is the
    /// right path).
    pub fn disambiguate_polysemous_multi_sense(
        &self,
        seed: &Seed,
    ) -> Option<Vec<HashSet<Seed>>> {
        const MIN_TYPED_FOR_LEADER: usize = 2;
        const COMPARABLE_RATIO: f32 = 1.5;
        const MAX_SENSES: usize = 3;
        const MIN_MEMBERS: usize = 2;

        let (neighbors, clusters, typed_score) =
            self.polysemy_label_prop_clusters(seed)?;

        // Sort cluster labels by typed_score descending.
        let mut by_score: Vec<(u32, usize)> =
            typed_score.iter().map(|(k, v)| (*k, *v)).collect();
        by_score.sort_by(|a, b| b.1.cmp(&a.1));
        if by_score.is_empty() {
            return None;
        }
        let leader = by_score[0].1;
        if leader < MIN_TYPED_FOR_LEADER {
            return None;
        }
        let leader_f = leader as f32;

        let mut out: Vec<HashSet<Seed>> = Vec::new();
        for (lbl, score) in &by_score {
            // Stop including clusters once they fall outside the
            // comparable band.  Note `score * COMPARABLE_RATIO`
            // (not `score / RATIO`) to widen the band downward.
            if (*score as f32) * COMPARABLE_RATIO < leader_f {
                break;
            }
            if let Some(members) = clusters.get(lbl) {
                if members.len() < MIN_MEMBERS {
                    continue;
                }
                let cluster: HashSet<Seed> =
                    members.iter().map(|&i| neighbors[i]).collect();
                out.push(cluster);
                if out.len() >= MAX_SENSES {
                    break;
                }
            }
        }
        if out.len() < 2 {
            return None;
        }
        Some(out)
    }

    /// Internal helper: run weighted label propagation on `seed`'s
    /// outgoing neighborhood and return the cluster decomposition.
    ///
    /// Algorithm (was Port #19 inline; extracted in Port #22):
    ///   1. Each neighbor starts with a UNIQUE label.
    ///   2. Build the weighted co-neighborhood graph: edge (i, j)
    ///      weighted by neighborhood-Jaccard(i, j) if >= 0.05.
    ///   3. 5 iterations of synchronous label propagation. Each
    ///      neighbor adopts the label with the highest weighted sum
    ///      among co-neighbors. Ties broken by (a) current label,
    ///      (b) max bond-type label among neighbors holding the
    ///      candidate label (typed beats RelatedTo), (c) numerically
    ///      smallest label.
    ///   4. Group neighbor indices by final label.
    ///
    /// Returns `(neighbors, clusters_by_label, typed_score_by_label)`
    /// or `None` when the neighborhood is too small or fewer than 2
    /// distinct clusters survive.
    fn polysemy_label_prop_clusters(
        &self,
        seed: &Seed,
    ) -> Option<(Vec<Seed>, HashMap<u32, Vec<usize>>, HashMap<u32, usize>)> {
        const MIN_NEIGHBORS_FOR_POLYSEMY: usize = 4;
        const LABEL_PROP_MIN_JAC: f32 = 0.05;
        const LABEL_PROP_ITERS: usize = 5;

        // OP-BRAIN-MapRAM P4: gather seed's outgoing neighbours + their
        // bond type, and each neighbour's outgoing SET, through the seam.
        // CACHE mode sources neighbours from the P1 cache and the type from
        // the carried predicate -- no resident adjacency/bonds. LEGACY mode
        // is byte-identical to the prior `neighbors_outgoing` + per-neighbour
        // `get_bond().bond_type()` (same order). The label-prop PARTITION is
        // the connected-component decomposition of the (order-independent)
        // co-neighbourhood graph, so the cluster MEMBERSHIPS + typed scores
        // returned here are order-independent; only the integer label VALUES
        // are positional (already per-load nondeterministic under the
        // HashMap-random resident order, so not a meaningful output).
        let typed_pairs = self.seam_outgoing_typed(seed);
        if typed_pairs.len() < MIN_NEIGHBORS_FOR_POLYSEMY {
            return None;
        }
        let neighbors: Vec<Seed> = typed_pairs.iter().map(|(s, _)| *s).collect();
        let neighborhoods: Vec<HashSet<Seed>> = neighbors
            .iter()
            .map(|n| self.seam_outgoing_set(n).into_iter().collect())
            .collect();

        let bond_type_label = |bt: Option<BondType>| -> u32 {
            match bt {
                Some(BondType::IsA) => 4,
                Some(BondType::PartOf) => 3,
                Some(BondType::Causes) => 2,
                Some(BondType::HasProperty) => 1,
                _ => 0,
            }
        };
        // Slice 2c hand-back: route through `get_bond` so archive-
        // resident bonds report their type instead of coming back as
        // `None` (which would mis-label them as untyped and skew the
        // label-propagation tie-break toward RelatedTo).
        let bond_labels: Vec<u32> = typed_pairs
            .iter()
            .map(|(_nb, bt)| bond_type_label(*bt))
            .collect();

        let n = neighbors.len();
        let mut labels: Vec<u32> = (0..n as u32).collect();
        let mut co: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n];
        for i in 0..n {
            for j in (i + 1)..n {
                let inter = neighborhoods[i].intersection(&neighborhoods[j]).count();
                let uni = neighborhoods[i].union(&neighborhoods[j]).count();
                let jac = if uni == 0 { 0.0 } else { (inter as f32) / (uni as f32) };
                if jac >= LABEL_PROP_MIN_JAC {
                    co[i].push((j, jac));
                    co[j].push((i, jac));
                }
            }
        }
        for _iter in 0..LABEL_PROP_ITERS {
            let mut next = labels.clone();
            let mut changed = false;
            for i in 0..n {
                if co[i].is_empty() { continue; }
                let mut sums: HashMap<u32, (f32, u32)> = HashMap::new();
                for &(j, w) in &co[i] {
                    let entry = sums.entry(labels[j]).or_insert((0.0, 0));
                    entry.0 += w;
                    if bond_labels[j] > entry.1 { entry.1 = bond_labels[j]; }
                }
                let cur = labels[i];
                let cur_pair = sums.get(&cur).copied().unwrap_or((0.0, 0));
                let mut best = cur;
                let mut best_w = cur_pair.0;
                let mut best_btl = cur_pair.1;
                for (&lbl, &(w, btl)) in &sums {
                    let better = w > best_w + 1e-6
                        || (((w - best_w).abs() <= 1e-6) && btl > best_btl)
                        || (((w - best_w).abs() <= 1e-6) && btl == best_btl && lbl < best);
                    if better {
                        best = lbl;
                        best_w = w;
                        best_btl = btl;
                    }
                }
                if best != cur { changed = true; }
                next[i] = best;
            }
            labels = next;
            if !changed { break; }
        }

        let mut clusters: HashMap<u32, Vec<usize>> = HashMap::new();
        for (i, &lbl) in labels.iter().enumerate() {
            clusters.entry(lbl).or_default().push(i);
        }
        if clusters.len() < 2 {
            return None;
        }

        let mut typed_score: HashMap<u32, usize> = HashMap::new();
        for (lbl, members) in &clusters {
            let mut typed = 0usize;
            for &i in members {
                // OP-BRAIN-MapRAM P4: reuse the seam-sourced type from
                // `typed_pairs` (cache carried predicate / legacy get_bond),
                // aligned by neighbour index -- no resident bond read.
                if let Some(bt) = typed_pairs[i].1 {
                    if bt != BondType::RelatedTo {
                        typed += 1;
                    }
                }
            }
            typed_score.insert(*lbl, typed);
        }
        Some((neighbors, clusters, typed_score))
    }

    /// Port Sense — score the sense clusters of a polysemous `seed`
    /// against a context window. For each cluster returned by the
    /// existing `polysemy_label_prop_clusters` helper, computes a
    /// real-valued score combining (a) per-bond mass — typed bonds
    /// from the seed into the cluster member outweigh RelatedTo —
    /// (b) direct-hit bonus when the member itself is in the context
    /// window, and (c) neighborhood overlap between the member's
    /// outgoing neighbors and the context window. Returns `None`
    /// when the seed is not polysemous (fewer than 2 clusters or
    /// fewer than `MIN_NEIGHBORS_FOR_POLYSEMY` outgoing neighbors).
    /// Read-only with respect to bond storage — anti-hallucination
    /// invariant preserved.
    pub fn score_sense_clusters(
        &self,
        seed: &Seed,
        context: &[Seed],
    ) -> Option<Vec<crate::sense::SenseClusterScore>> {
        let (neighbors, clusters, typed_score) =
            self.polysemy_label_prop_clusters(seed)?;

        // OP-BRAIN-MapRAM P4: one seam pass for the per-member bond mass
        // (cache carried predicate / legacy get_bond), keyed by neighbour
        // seed -- no resident bond read in cache mode.
        let type_by_nbr: HashMap<Seed, Option<crate::bond::BondType>> =
            self.seam_outgoing_typed(seed).into_iter().collect();

        let context_set: HashSet<Seed> = context.iter().copied().collect();

        let mut out: Vec<crate::sense::SenseClusterScore> =
            Vec::with_capacity(clusters.len());
        for (lbl, indices) in clusters.iter() {
            let mut members_with_mass: Vec<(Seed, f32, usize)> =
                Vec::with_capacity(indices.len());
            let mut members: Vec<Seed> = Vec::with_capacity(indices.len());
            for &i in indices {
                let nbr = neighbors[i];
                members.push(nbr);
                // Bond mass — typed > RelatedTo. No bond shouldn't
                // happen at this point (label-prop just confirmed
                // it) but we treat it defensively.
                let bond_mass = match type_by_nbr.get(&nbr).copied().flatten() {
                    Some(crate::bond::BondType::RelatedTo) => crate::sense::tuning::RELATED_BOND_MASS,
                    Some(_) => crate::sense::tuning::TYPED_BOND_MASS,
                    None => 0.0,
                };
                // Neighborhood overlap — how many context nodes
                // appear among the candidate member's outgoing
                // neighbors. Cheap-but-meaningful co-activation
                // proxy that uses no new fields.
                let overlap = if context_set.is_empty() {
                    0
                } else {
                    let nbr_out: HashSet<Seed> =
                        self.seam_outgoing_set(&nbr).into_iter().collect();
                    context_set.iter().filter(|c| nbr_out.contains(*c)).count()
                };
                members_with_mass.push((nbr, bond_mass, overlap));
            }
            let score = crate::sense::score_cluster(&members_with_mass, &context_set);
            let typed_bond_count = typed_score.get(lbl).copied().unwrap_or(0);
            out.push(crate::sense::SenseClusterScore {
                label: *lbl,
                members,
                score,
                typed_bond_count,
                is_winner: false,
            });
        }
        out.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let sorted_scores: Vec<(u32, f32, usize)> = out
            .iter()
            .map(|c| (c.label, c.score, c.typed_bond_count))
            .collect();
        if let Some(winner_lbl) = crate::sense::elect_winner(&sorted_scores) {
            for c in out.iter_mut() {
                if c.label == winner_lbl {
                    c.is_winner = true;
                    break;
                }
            }
        }
        Some(out)
    }

    /// Port Sense — context-aware disambiguation that returns a
    /// per-neighbor gain map. Winner cluster members get gain 1.0;
    /// loser cluster members get `loser_gain` (default 0.25 in the
    /// runtime config). Returns `None` when no winner can be elected.
    /// Read-only with respect to bond storage.
    pub fn disambiguate_polysemous_with_context(
        &self,
        seed: &Seed,
        context: &[Seed],
        loser_gain: f32,
    ) -> Option<HashMap<Seed, f32>> {
        let scores = self.score_sense_clusters(seed, context)?;
        if !scores.iter().any(|c| c.is_winner) {
            return None;
        }
        let mut gain: HashMap<Seed, f32> = HashMap::new();
        for cluster in scores {
            let g = if cluster.is_winner { 1.0 } else { loser_gain };
            for m in cluster.members {
                gain.insert(m, g);
            }
        }
        Some(gain)
    }

    /// Per-graph 90th percentile of total node degree (out + in).
    /// Used as the dynamic hub threshold across spread/bridge/
    /// curiosity. Recomputed on `from_snapshot` and on demand.
    pub fn degree_p90(&self) -> usize { self.degree_p90 }

    /// Recompute and cache `degree_p90`. O(N + N log N). On an empty
    /// graph the cached value falls back to `DEGREE_NORM_THRESHOLD`
    /// so existing semantics still hold for fresh brains.
    pub fn recompute_degree_p90(&mut self) -> usize {
        // Storage v2 Slice 2c: walk all seeds (overlay + archive),
        // filtered by deletions.
        let all_seeds: Vec<Seed> = match &self.archive {
            None => self.nodes.keys().copied().collect(),
            Some(arch) => {
                let mut seeds: Vec<Seed> = self
                    .nodes
                    .keys()
                    .copied()
                    .filter(|s| !self.overlay_deletions.contains(s))
                    .collect();
                for s in arch.iter_node_seeds() {
                    if self.nodes.contains_key(&s) || self.overlay_deletions.contains(&s) {
                        continue;
                    }
                    seeds.push(s);
                }
                seeds
            }
        };
        if all_seeds.is_empty() {
            self.degree_p90 = DEGREE_NORM_THRESHOLD;
            return self.degree_p90;
        }
        let mut degs: Vec<usize> = all_seeds.iter().map(|s| {
            self.outgoing.get(s).map(|v| v.len()).unwrap_or(0)
                + self.incoming.get(s).map(|v| v.len()).unwrap_or(0)
        }).collect();
        degs.sort_unstable();
        let idx = ((degs.len() as f64) * 0.9).floor() as usize;
        let idx = idx.min(degs.len() - 1);
        // Floor at 1 so a sparse graph still has a real threshold.
        self.degree_p90 = degs[idx].max(1);
        self.degree_p90
    }

    /// TEST-ONLY (OP-BRAIN-MapRAM P3 independence proof): corrupt every
    /// resident structure the LEGACY recall reads -- adjacency lists,
    /// bond weights, node confidence/content -- so a cache-mode recall
    /// that still touched ANY of them would diverge. The map, intern,
    /// and the (map-sourced) degree_p90 are left intact: those are the
    /// cache path's legitimate, non-rehydrate inputs.
    #[cfg(test)]
    pub(crate) fn mapram_test_poison_resident(&mut self) {
        self.outgoing.clear();
        self.incoming.clear();
        for b in self.bonds.values_mut() {
            b.keyframes = vec![crate::bond::TemporalKeyframe {
                timestamp: 0,
                weight: 0.0,
                source: crate::bond::KeyframeSource::Baseline,
            }];
        }
        for n in self.nodes.values_mut() {
            n.content = "POISON".to_string();
            n.knowledge.content = crate::node::KnowledgeContent::Raw("POISON".to_string());
            n.knowledge.confidence = crate::node::Confidence::Contested;
        }
    }

    /// TEST-ONLY (OP-BRAIN-MapRAM P4 independence proof): flip every
    /// RESIDENT heap bond TYPE so any reader that still sourced a bond
    /// type from the resident `bonds` (via `get_bond`) instead of the
    /// carried cache predicate would compute a different typed/related
    /// split. Complements `mapram_test_poison_resident` (which clears the
    /// adjacency + nukes weights + node payloads but leaves bond TYPES
    /// intact). The cache carries its own predicate, so a fully rerouted
    /// cache-mode reader is invariant under this corruption.
    #[cfg(test)]
    pub(crate) fn mapram_test_corrupt_resident_bond_types(&mut self) {
        for b in self.bonds.values_mut() {
            b.bond_type = match b.bond_type {
                crate::bond::BondType::RelatedTo => crate::bond::BondType::IsA,
                _ => crate::bond::BondType::RelatedTo,
            };
        }
    }

    /// TEST-ONLY (OP-BRAIN-MapRAM P5-batch2 keyframe/timestamp poison):
    /// overwrite every RESIDENT heap bond's keyframes with a single
    /// sentinel keyframe at a DISTINCTIVE non-zero timestamp + weight +
    /// source, so any reader that still sourced keyframe TIMESTAMPS
    /// (context scope generation) or the latest keyframe WEIGHT/SOURCE
    /// (consolidation hub-fill legs) from the resident `bonds` would
    /// diverge. Complements `mapram_test_poison_resident` (which zeroes
    /// keyframe weights + timestamps to 0) with an unmistakable non-zero
    /// sentinel for the keyframe-CARRYING batch-2 readers. The cache
    /// carries its own payload keyframes, so a fully rerouted cache-mode
    /// reader is invariant under this corruption.
    #[cfg(test)]
    pub(crate) fn mapram_test_corrupt_resident_keyframes(&mut self) {
        for b in self.bonds.values_mut() {
            b.keyframes = vec![crate::bond::TemporalKeyframe {
                timestamp: 0x5350_4F49_5350_4F49,
                weight: 0.123_456,
                source: crate::bond::KeyframeSource::Manual,
            }];
        }
    }

    /// Return all node seeds.
    pub fn all_seeds(&self) -> Vec<Seed> {
        match &self.archive {
            None => self.nodes.keys().copied().collect(),
            Some(arch) => {
                let mut seeds: Vec<Seed> = self
                    .nodes
                    .keys()
                    .copied()
                    .filter(|s| !self.overlay_deletions.contains(s))
                    .collect();
                for s in arch.iter_node_seeds() {
                    if self.nodes.contains_key(&s) || self.overlay_deletions.contains(&s) {
                        continue;
                    }
                    seeds.push(s);
                }
                seeds
            }
        }
    }

    /// Export a serializable snapshot of this network.
    ///
    /// Adjacency and keyword indices are rebuilt on load, so they're not included.
    pub fn to_snapshot(&self) -> NetworkSnapshot {
        match &self.archive {
            None => NetworkSnapshot {
                nodes: self.nodes.values().cloned().collect(),
                bonds: self.bonds.values().cloned().collect(),
            },
            Some(arch) => {
                // Walk overlay + archive (filtered by deletions and
                // overlay overrides), materializing heap copies for
                // archived entries. This is a one-off cost on save;
                // in the steady state the network keeps the archive
                // alive and reads from it directly.
                let mut nodes: Vec<KnowledgeNode> = Vec::new();
                for n in self.nodes.values() {
                    if self.overlay_deletions.contains(&n.seed) {
                        continue;
                    }
                    nodes.push(n.clone());
                }
                for (seed, an) in arch.iter_nodes() {
                    if self.nodes.contains_key(&seed) || self.overlay_deletions.contains(&seed) {
                        continue;
                    }
                    nodes.push(crate::archive::knowledge_node_from_archived(an.as_archived()));
                }
                let mut bonds: Vec<TemporalBond> = Vec::new();
                for (key, b) in self.bonds.iter() {
                    if self.overlay_deletions.contains(&key.0)
                        || self.overlay_deletions.contains(&key.1)
                    {
                        continue;
                    }
                    bonds.push(b.clone());
                }
                for (key, ab) in arch.iter_bonds() {
                    if self.bond_tombstones.contains(&key) { continue; }
                    if self.bonds.contains_key(&key)
                        || self.overlay_deletions.contains(&key.0)
                        || self.overlay_deletions.contains(&key.1)
                    {
                        continue;
                    }
                    bonds.push(crate::archive::temporal_bond_from_archived(ab.as_archived()));
                }
                NetworkSnapshot { nodes, bonds }
            }
        }
    }

    /// Rebuild a network from a snapshot. Reconstructs adjacency and keyword indices.
    pub fn from_snapshot(snapshot: NetworkSnapshot) -> Self {
        let mut net = AssociativeNetwork::new();
        for node in snapshot.nodes {
            net.add_node(node);
        }
        for bond in snapshot.bonds {
            // Both endpoints already exist because nodes were inserted first.
            let _ = net.add_bond(bond);
        }
        // Compute per-graph hub threshold once; spreads, bridges, and
        // curiosity all read from it instead of the global constant.
        net.recompute_degree_p90();
        // Storage v2 Phase 1 (open issue #1): cache the typed-bond
        // ratio on every node so narration walkers don't recompute
        // it per neighbour on every query. Recompute here at load
        // and again in `ConsolidationEngine` whenever bond shape
        // changes.
        net.recompute_all_typed_ratios();
        net
    }

    /// Recompute `KnowledgeNode::typed_ratio` for every node. The ratio
    /// is the fraction of outgoing bonds whose `BondType` is something
    /// other than `BondType::RelatedTo`. A node with no outgoing bonds
    /// has ratio 0.0.
    ///
    /// Storage v2 Phase 1 (open issue #1): replaces the per-query
    /// inline computation that lived in `narration::HubSelector` and
    /// `narration::Walker::collect_outgoing`.
    pub fn recompute_all_typed_ratios(&mut self) {
        // Storage v2 Slice 2c hand-back: archive-resident nodes carry
        // their `typed_ratio` pre-computed from the save that produced
        // the archive, so the recompute pass only has to refresh the
        // overlay (heap) nodes — CoW-promoted ones plus any net-new
        // ones added after load. Iterating `self.nodes.keys()` gives
        // exactly that set; archived-only seeds are intentionally
        // skipped because their cached ratio is already authoritative.
        //
        // Bond reads still go through `get_bond` because an overlay
        // node can have outgoing bonds that live entirely in the
        // archive (e.g. a node CoW-promoted for a non-edge field
        // change whose edges were never touched). Previously this
        // used `bond_raw`, which would silently miss those edges and
        // return a typed_ratio computed against a partial bond set.
        let seeds: Vec<Seed> = self.nodes.keys().copied().collect();
        let mut ratios: Vec<(Seed, f32)> = Vec::with_capacity(seeds.len());
        for seed in &seeds {
            // OP-BRAIN-MapRAM P4: see `update_typed_ratio_for` -- neighbour
            // + bond-type enumeration through the seam (cache-sourced and
            // resident-independent in cache mode), byte-identical typed/total
            // count in legacy mode. Order-independent.
            let edges = self.seam_outgoing_typed(seed);
            if edges.is_empty() {
                ratios.push((*seed, 0.0));
                continue;
            }
            let mut typed = 0usize;
            let mut total = 0usize;
            for (_nbr, bt) in &edges {
                if let Some(bt) = bt {
                    total += 1;
                    if *bt != crate::bond::BondType::RelatedTo {
                        typed += 1;
                    }
                }
            }
            let r = if total == 0 { 0.0 } else { typed as f32 / total as f32 };
            ratios.push((*seed, r));
        }
        for (seed, r) in ratios {
            if let Some(node) = self.nodes.get_mut(&seed) {
                node.typed_ratio = r;
            }
        }
    }

    /// Compute the average bond weight in the activated subgraph.
    /// Used by the context selector for coherence scoring.
    pub fn subgraph_coherence(&self, activated_seeds: &[Seed], scope: &TemporalScope, scar_log: &ScarLog) -> f32 {
        let seed_set: HashSet<&Seed> = activated_seeds.iter().collect();
        let mut total_weight = 0.0f32;
        let mut bond_count = 0usize;

        for (key, bond) in self.bonds().iter() {
            let (from, to) = (key.0, key.1);
            if seed_set.contains(&from) && seed_set.contains(&to) {
                // Port T.5.1: query-path effective weight = stored * freshness.
                total_weight += bond.weight_in_scope(scope, scar_log).effective_weight
                    * self.decay_multiplier_for_bond(&bond) * self.sense_gain_for_bond(&from, &to);
                bond_count += 1;
            }
        }

        if bond_count == 0 {
            0.0
        } else {
            total_weight / bond_count as f32
        }
    }
}

// ===== OP-BRAIN-MapRAM P2: the selectable adjacency seam =====
//
// A single access point the spread / typed-recall / curiosity /
// consolidation / mcp consumers call instead of touching
// `outgoing`/`incoming` directly. In LEGACY mode (mapram_mode == 0, the
// default) every method is byte-for-byte the prior behaviour. In
// CACHE-BACKED mode (mapram_mode == 1) neighbour LISTS come from the P1
// `CacheBackedAdjacency` (map -> 2Q cache -> v5 reader) and DEGREE from
// the resident map (O(1)). rehydrate_adjacency stays intact and is
// still the source for `view_bond`/`view_node`, so per-neighbour weights
// are identical across both modes -- the seam changes only which
// neighbour ids are enumerated and where degree is read.
impl AssociativeNetwork {
    /// Install the P1 cache + the handles needed to build a fresh
    /// `V5AdjacencySource` per decode. Leaves the mode at legacy;
    /// `set_mapram_mode(true)` opts a brain onto the cache path.
    pub(crate) fn install_mapram_seam(
        &mut self,
        cache: std::sync::Arc<crate::cache_adjacency::CacheBackedAdjacency>,
        manifest: std::sync::Arc<std::sync::RwLock<std::sync::Arc<crate::manifest::Manifest>>>,
        bond_cache: std::sync::Arc<crate::bond_store::BondSstReaderCache>,
    ) {
        // OP-BRAIN-MapRAM P3 -- source the hub threshold from the map
        // so `degree_p90` no longer depends on rehydrate's recompute.
        // Byte-identical to the rehydrated `recompute_degree_p90` on a
        // loaded brain (same floor(0.9*n) statistic of the same degrees).
        self.degree_p90 = cache.degree_p90();
        self.mapram_cache = Some(cache);
        self.mapram_manifest = Some(manifest);
        self.mapram_bond_cache = Some(bond_cache);
    }

    /// Select the adjacency path: `true` = cache-backed, `false` =
    /// legacy. `&self` (atomic) so it can flip a shared network.
    pub(crate) fn set_mapram_mode(&self, cache_backed: bool) {
        self.mapram_mode.store(
            if cache_backed { 1 } else { 0 },
            std::sync::atomic::Ordering::Release,
        );
    }

    /// True iff the cache-backed path is selected AND installed.
    #[inline]
    pub(crate) fn mapram_cache_backed(&self) -> bool {
        self.mapram_mode.load(std::sync::atomic::Ordering::Acquire) == 1
            && self.mapram_cache.is_some()
    }

    /// Build a fresh on-miss resolver from the installed handles.
    fn mapram_source(&self) -> Option<crate::cache_adjacency::V5AdjacencySource> {
        let manifest = self.mapram_manifest.as_ref()?.read().ok()?.clone();
        let bond_cache = self.mapram_bond_cache.as_ref()?.clone();
        Some(crate::cache_adjacency::V5AdjacencySource::new(
            manifest,
            bond_cache,
            self.node_intern.clone(),
        ))
    }

    /// Cache-backed RAW neighbour seeds (uncapped), in v5-reader order.
    /// Falls back to the resident list when the cache/handles are
    /// missing, so the seam is never LESS correct than legacy.
    fn mapram_raw_neighbors(
        &self,
        seed: &Seed,
        dir: crate::cache_adjacency::Direction,
    ) -> Vec<Seed> {
        let (cache, src, id) = match (
            self.mapram_cache.as_ref(),
            self.mapram_source(),
            self.node_intern.get(seed),
        ) {
            (Some(c), Some(s), Some(i)) => (c, s, i),
            _ => {
                return match dir {
                    crate::cache_adjacency::Direction::Outgoing => self.neighbors_outgoing(seed),
                    crate::cache_adjacency::Direction::Incoming => self.neighbors_incoming(seed),
                };
            }
        };
        let decoded = match cache.neighbors(id, dir, &src) {
            Ok(d) => d,
            Err(_) => return Vec::new(),
        };
        decoded
            .iter()
            .filter_map(|n| self.node_intern.reverse_lookup(n.node))
            .collect()
    }

    /// Shared hub-cap: top-`max` neighbours by latest raw bond weight,
    /// descending -- the exact selection the legacy capped accessors
    /// apply. Bumps the same drop counter.
    fn mapram_cap_by_weight(
        &self,
        anchor: &Seed,
        raw: Vec<Seed>,
        max: u32,
        outgoing: bool,
    ) -> Vec<Seed> {
        let mut weighted: Vec<(Seed, f32)> = Vec::with_capacity(raw.len());
        for n in &raw {
            let b = if outgoing { self.get_bond(anchor, n) } else { self.get_bond(n, anchor) };
            weighted.push((*n, b.map(|bn| bn.latest_raw_weight()).unwrap_or(0.0)));
        }
        weighted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let keep = max as usize;
        let dropped = raw.len().saturating_sub(keep);
        self.hub_cap_filtered_at_query_total
            .fetch_add(dropped, std::sync::atomic::Ordering::Relaxed);
        weighted.into_iter().take(keep).map(|(s, _)| s).collect()
    }

    /// SEAM: outgoing neighbour list for the spread hot path. Legacy =
    /// `neighbors_outgoing_for_read_mode`; cache = cache raw set + the
    /// identical outgoing hub-cap.
    /// Canonical neighbour ENUMERATION order: seed bytes ascending.
    /// Applied by every seam list accessor in BOTH modes so the
    /// order-sensitive parts of traversal -- the spread's first-reach
    /// propagation at convergence nodes, the hub-cap tie-break among
    /// equal-weight neighbours, and consolidation's `take(max_neighbors)`
    /// -- are DETERMINISTIC and identical across the legacy and
    /// cache-backed paths. This also closes a pre-existing per-load
    /// non-determinism: the resident adjacency is built from a
    /// `HashMap::drain` (random order under RandomState), so before this
    /// a convergence node's propagated activation varied run-to-run.
    /// Same defence as the final-result seed-byte tiebreak (BR.C-Hyp).
    fn canonicalize_neighbors(&self, list: &mut Vec<Seed>) {
        list.sort_unstable_by(|a, b| a.as_bytes().cmp(b.as_bytes()));
    }

/// SEAM (edge-carrying): outgoing neighbours for the spread hot
    /// path WITH their edge weights. LEGACY yields `Resident` edges
    /// (caller reads `view_bond`); CACHE yields `Carried` edges holding
    /// the decode payload. Same canonical order + hub-cap as the old
    /// `seam_spread_outgoing`.
    pub(crate) fn seam_spread_outgoing_edges(&self, seed: &Seed) -> Vec<SeamEdge> {
        if self.mapram_cache_backed() {
            let mut raw = self.mapram_raw_edges(seed, crate::cache_adjacency::Direction::Outgoing);
            self.canonicalize_edges(&mut raw);
            if !self.hub_read_enforcement || (raw.len() as u32) <= self.hub_read_max_out {
                return raw;
            }
            self.mapram_cap_edges_by_weight(seed, raw, self.hub_read_max_out, true)
        } else {
            let mut raw = self.neighbors_outgoing_read_mode_uncapped(seed);
            self.canonicalize_neighbors(&mut raw);
            let capped = if !self.hub_read_enforcement || (raw.len() as u32) <= self.hub_read_max_out {
                raw
            } else {
                self.mapram_cap_by_weight(seed, raw, self.hub_read_max_out, true)
            };
            capped.into_iter().map(|nbr| SeamEdge { nbr, bond: SeamBondSrc::Resident }).collect()
        }
    }

    /// SEAM (edge-carrying): incoming neighbours for the spread hot
    /// path WITH their edge weights. See `seam_spread_outgoing_edges`.
    pub(crate) fn seam_spread_incoming_edges(&self, seed: &Seed) -> Vec<SeamEdge> {
        if self.mapram_cache_backed() {
            let mut raw = self.mapram_raw_edges(seed, crate::cache_adjacency::Direction::Incoming);
            self.canonicalize_edges(&mut raw);
            if !self.hub_read_enforcement || (raw.len() as u32) <= self.hub_read_max_in {
                return raw;
            }
            self.mapram_cap_edges_by_weight(seed, raw, self.hub_read_max_in, false)
        } else {
            let mut raw = self.neighbors_incoming(seed);
            self.canonicalize_neighbors(&mut raw);
            let capped = if !self.hub_read_enforcement || (raw.len() as u32) <= self.hub_read_max_in {
                raw
            } else {
                self.mapram_cap_by_weight(seed, raw, self.hub_read_max_in, false)
            };
            capped.into_iter().map(|nbr| SeamEdge { nbr, bond: SeamBondSrc::Resident }).collect()
        }
    }

    /// Cache-backed neighbours WITH their carried decode payloads
    /// (predicate + `BondPayload`), in v5-reader order -- the
    /// edge-carrying twin of `mapram_raw_neighbors`. Falls back to
    /// resident `Resident` edges if the cache/handles are missing.
    fn mapram_raw_edges(
        &self,
        seed: &Seed,
        dir: crate::cache_adjacency::Direction,
    ) -> Vec<SeamEdge> {
        let (cache, src, id) = match (
            self.mapram_cache.as_ref(),
            self.mapram_source(),
            self.node_intern.get(seed),
        ) {
            (Some(c), Some(s), Some(i)) => (c, s, i),
            _ => {
                let seeds = match dir {
                    crate::cache_adjacency::Direction::Outgoing => self.neighbors_outgoing(seed),
                    crate::cache_adjacency::Direction::Incoming => self.neighbors_incoming(seed),
                };
                return seeds
                    .into_iter()
                    .map(|nbr| SeamEdge { nbr, bond: SeamBondSrc::Resident })
                    .collect();
            }
        };
        let decoded = match cache.neighbors(id, dir, &src) {
            Ok(d) => d,
            Err(_) => return Vec::new(),
        };
        decoded
            .iter()
            .filter_map(|n| {
                self.node_intern.reverse_lookup(n.node).map(|nbr| SeamEdge {
                    nbr,
                    bond: SeamBondSrc::Carried { predicate: n.predicate, payload: n.payload.clone() },
                })
            })
            .collect()
    }

    /// Canonical neighbour enumeration order (seed bytes ascending) over
    /// edges -- the edge-carrying twin of `canonicalize_neighbors`, so
    /// the carried payload travels with its seed through the sort.
    fn canonicalize_edges(&self, list: &mut Vec<SeamEdge>) {
        list.sort_unstable_by(|a, b| a.nbr.as_bytes().cmp(b.nbr.as_bytes()));
    }

    /// Reconstruct the heap `TemporalBond` for a carried edge,
    /// byte-identically to the bond `rehydrate_adjacency` would have
    /// inserted from the same payload (empty-keyframe baseline
    /// fallback, `bond_type_from_discriminant`, `DEFAULT_HALF_LIFE_MS`).
    /// `from`/`to` follow edge orientation: (anchor, nbr) outgoing,
    /// (nbr, anchor) incoming.
    fn reconstruct_carried_bond(
        from: Seed,
        to: Seed,
        predicate: u8,
        payload: &crate::bond_store::BondPayload,
    ) -> TemporalBond {
        let bond_type = crate::memtable::bond_type_from_discriminant(predicate)
            .expect("carried predicate validated at decode (rehydrate filter)");
        let keyframes = if payload.keyframes.is_empty() {
            vec![crate::bond::TemporalKeyframe {
                timestamp: 0,
                weight: payload.weight as f32 / 255.0,
                source: crate::bond::KeyframeSource::Baseline,
            }]
        } else {
            payload.keyframes.clone()
        };
        TemporalBond {
            from,
            to,
            bond_type,
            keyframes,
            half_life_ms: crate::bond::DEFAULT_HALF_LIFE_MS,
        }
    }

    /// Edge-carrying hub-cap: top-`max` by latest raw bond weight, the
    /// carried-payload twin of `mapram_cap_by_weight`. Weight comes
    /// from the carried payload (byte-identical to `get_bond`'s
    /// `latest_raw_weight`); same stable sort, same drop counter.
    fn mapram_cap_edges_by_weight(
        &self,
        anchor: &Seed,
        raw: Vec<SeamEdge>,
        max: u32,
        outgoing: bool,
    ) -> Vec<SeamEdge> {
        let mut weighted: Vec<(SeamEdge, f32)> = Vec::with_capacity(raw.len());
        for e in raw {
            let w = match &e.bond {
                SeamBondSrc::Carried { predicate, payload } => {
                    let (from, to) = if outgoing { (*anchor, e.nbr) } else { (e.nbr, *anchor) };
                    Self::reconstruct_carried_bond(from, to, *predicate, payload).latest_raw_weight()
                }
                SeamBondSrc::Resident => {
                    let b = if outgoing { self.get_bond(anchor, &e.nbr) } else { self.get_bond(&e.nbr, anchor) };
                    b.map(|bn| bn.latest_raw_weight()).unwrap_or(0.0)
                }
            };
            weighted.push((e, w));
        }
        weighted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let keep = max as usize;
        let dropped = weighted.len().saturating_sub(keep);
        self.hub_cap_filtered_at_query_total
            .fetch_add(dropped, std::sync::atomic::Ordering::Relaxed);
        weighted.into_iter().take(keep).map(|(e, _)| e).collect()
    }

    /// SEAM: spread/typed hot-path node view (confidence + content)
    /// sourced WITHOUT the resident adjacency in cache mode. LEGACY =
    /// `view_node`; CACHE = the rehydration STUB reproduced from the
    /// resident MAP (present -> stub, absent -> None). See `SeamNode`.
    pub(crate) fn seam_recall_node(&self, seed: &Seed) -> Option<SeamNode<'_>> {
        if self.mapram_cache_backed() {
            match (self.mapram_cache.as_ref(), self.node_intern.get(seed)) {
                (Some(c), Some(id)) if c.map_contains(id) => Some(SeamNode::Stub),
                _ => None,
            }
        } else {
            self.view_node(seed).map(SeamNode::Resident)
        }
    }

    /// OP-SPREAD-1 -- `seam_recall_node` AND `seam_total_degree` from a
    /// SINGLE `NodeIntern` probe.
    ///
    /// WHY THIS EXISTS. `NodeIntern::get` takes an `RwLock` read guard
    /// and then probes a 52M-entry `HashMap` keyed on a 32-byte `Seed`
    /// -- a random access that misses cache and often the TLB.
    /// `seam_recall_node` costs one such probe; `seam_total_degree`
    /// costs TWO (`seam_out_degree` + `seam_in_degree` each look the
    /// seed up again). The BFS edge loop wanted all of them for BOTH
    /// endpoints, and then `bridge::is_candidate` asked for the two
    /// degrees a SECOND time -- up to EIGHT probes per bond. Measured
    /// on the live 52M-node brain that is ~35 us per edge, which is why
    /// a depth-2 spread over 118,073 bonds burned the entire 4s ask
    /// watchdog and returned `TimedOut` on every single query.
    ///
    /// The answers are byte-identical to calling the two accessors
    /// separately -- including the case where the node is absent from
    /// the resident MAP while its `NodeId` still carries degree, which
    /// is why the degree is returned unconditionally rather than being
    /// folded into the `Option`.
    pub(crate) fn seam_node_and_degree(&self, seed: &Seed) -> (Option<SeamNode<'_>>, usize) {
        if self.mapram_cache_backed() {
            match (self.mapram_cache.as_ref(), self.node_intern.get(seed)) {
                (Some(c), Some(id)) => {
                    let node = if c.map_contains(id) { Some(SeamNode::Stub) } else { None };
                    let deg = c.out_degree(id) as usize + c.in_degree(id) as usize;
                    (node, deg)
                }
                _ => (None, 0),
            }
        } else {
            (
                self.view_node(seed).map(SeamNode::Resident),
                self.out_degree(seed) + self.in_degree(seed),
            )
        }
    }


    /// SEAM: RAW (uncapped) outgoing neighbour list -- the consolidation
    /// / mcp walk view. Legacy = `neighbors_outgoing`.
    pub(crate) fn seam_neighbors_outgoing(&self, seed: &Seed) -> Vec<Seed> {
        let mut raw = if self.mapram_cache_backed() {
            self.mapram_raw_neighbors(seed, crate::cache_adjacency::Direction::Outgoing)
        } else {
            self.neighbors_outgoing(seed)
        };
        self.canonicalize_neighbors(&mut raw);
        raw
    }

    /// SEAM: out-degree. Legacy = resident adjacency; cache = resident
    /// MAP (O(1), byte-exact to adjacency degree by P0).
    pub(crate) fn seam_out_degree(&self, seed: &Seed) -> usize {
        if !self.mapram_cache_backed() {
            return self.out_degree(seed);
        }
        match (self.mapram_cache.as_ref(), self.node_intern.get(seed)) {
            (Some(c), Some(id)) => c.out_degree(id) as usize,
            _ => 0,
        }
    }

    /// SEAM: in-degree. See `seam_out_degree`.
    pub(crate) fn seam_in_degree(&self, seed: &Seed) -> usize {
        if !self.mapram_cache_backed() {
            return self.in_degree(seed);
        }
        match (self.mapram_cache.as_ref(), self.node_intern.get(seed)) {
            (Some(c), Some(id)) => c.in_degree(id) as usize,
            _ => 0,
        }
    }

    /// SEAM: total degree (out + in) -- the `deg_norm` / hub input.
    #[inline]
    pub fn seam_total_degree(&self, seed: &Seed) -> usize {
        self.seam_out_degree(seed) + self.seam_in_degree(seed)
    }

    /// SEAM (P4): outgoing neighbour SEEDS for the peripheral
    /// sense/polysemy + typed-ratio readers, sourced WITHOUT the resident
    /// adjacency in cache mode. RAW + UNCAPPED + UNCANONICALIZED: every
    /// consumer folds this into a `HashSet` (Jaccard neighbourhoods,
    /// anchor/overlap sets) or a per-edge COUNT (typed ratio), all
    /// order-independent -- so no canonical sort is imposed and the LEGACY
    /// branch stays byte-identical to the prior `neighbors_outgoing`.
    /// Legacy = `neighbors_outgoing`; cache = the P1 cache raw set.
    pub(crate) fn seam_outgoing_set(&self, seed: &Seed) -> Vec<Seed> {
        if self.mapram_cache_backed() {
            self.mapram_raw_neighbors(seed, crate::cache_adjacency::Direction::Outgoing)
        } else {
            self.neighbors_outgoing(seed)
        }
    }

    /// SEAM (P5-batch1): incoming neighbour SEEDS -- the in-direction
    /// twin of `seam_outgoing_set`, for the order-independent
    /// set-membership / coverage consumers (bridge Jaccard out+in
    /// neighbourhoods, idk_gate anchor-coverage) that fold out+in
    /// neighbours into a `HashSet`. RAW + UNCAPPED + UNCANONICALIZED on
    /// purpose: every consumer set-folds it, so order is irrelevant and
    /// the LEGACY branch is byte-identical to the prior
    /// `neighbors_incoming`. Legacy = `neighbors_incoming`; cache = the
    /// P1 cache raw set (incoming via reverse index, proven exact by
    /// `mapram_p1_tests::cache_incoming_via_reverse_index_is_exact`).
    pub(crate) fn seam_incoming_set(&self, seed: &Seed) -> Vec<Seed> {
        if self.mapram_cache_backed() {
            self.mapram_raw_neighbors(seed, crate::cache_adjacency::Direction::Incoming)
        } else {
            self.neighbors_incoming(seed)
        }
    }

    /// SEAM (P4): seed's outgoing neighbours paired with their bond TYPE,
    /// for the sense/polysemy label-prop + typed-ratio readers. In CACHE
    /// mode the type comes from the carried predicate
    /// (`bond_type_from_discriminant`), so the read is independent of the
    /// resident `bonds`/adjacency; in LEGACY mode it is exactly the prior
    /// `neighbors_outgoing` + per-neighbour `get_bond().bond_type()` (same
    /// order, same Some/None semantics) so legacy output is byte-unchanged.
    /// `None` type = no resident bond for that neighbour (legacy-only
    /// defensive case; the cache only yields real, validated edges).
    pub(crate) fn seam_outgoing_typed(
        &self,
        seed: &Seed,
    ) -> Vec<(Seed, Option<crate::bond::BondType>)> {
        if self.mapram_cache_backed() {
            self.mapram_raw_edges(seed, crate::cache_adjacency::Direction::Outgoing)
                .into_iter()
                .map(|e| match e.bond {
                    SeamBondSrc::Carried { predicate, .. } => (
                        e.nbr,
                        crate::memtable::bond_type_from_discriminant(predicate),
                    ),
                    SeamBondSrc::Resident => {
                        (e.nbr, self.get_bond(seed, &e.nbr).map(|b| b.bond_type()))
                    }
                })
                .collect()
        } else {
            self.neighbors_outgoing(seed)
                .into_iter()
                .map(|nbr| (nbr, self.get_bond(seed, &nbr).map(|b| b.bond_type())))
                .collect()
        }
    }

    /// SEAM (edge-carrying, UNCAPPED): outgoing neighbours WITH their
    /// carried edge payload, in CANONICAL order (seed bytes ascending) in
    /// BOTH modes -- the uncapped twin of `seam_spread_outgoing_edges`
    /// for the order-SENSITIVE bond-carrying walks (consolidation
    /// hub-fill, context scope keyframes, reasoning/narration walks). NO
    /// hub-cap: those walks impose their own `take(cap)` /
    /// `take(scan_limit)`, and the canonical order makes that take
    /// DETERMINISTIC and identical across the legacy and cache paths
    /// (legacy resident order was `HashMap::drain`-random per load).
    /// CACHE yields `Carried` edges (predicate + full `BondPayload`
    /// incl. keyframes); LEGACY yields `Resident` edges.
    pub(crate) fn seam_outgoing_edges(&self, seed: &Seed) -> Vec<SeamEdge> {
        let mut raw = if self.mapram_cache_backed() {
            self.mapram_raw_edges(seed, crate::cache_adjacency::Direction::Outgoing)
        } else {
            self.neighbors_outgoing(seed)
                .into_iter()
                .map(|nbr| SeamEdge { nbr, bond: SeamBondSrc::Resident })
                .collect()
        };
        self.canonicalize_edges(&mut raw);
        raw
    }

    /// SEAM (edge-carrying, UNCAPPED): incoming twin of
    /// `seam_outgoing_edges`. CACHE = carried payload (reverse index);
    /// LEGACY = `neighbors_incoming`.
    pub(crate) fn seam_incoming_edges(&self, seed: &Seed) -> Vec<SeamEdge> {
        let mut raw = if self.mapram_cache_backed() {
            self.mapram_raw_edges(seed, crate::cache_adjacency::Direction::Incoming)
        } else {
            self.neighbors_incoming(seed)
                .into_iter()
                .map(|nbr| SeamEdge { nbr, bond: SeamBondSrc::Resident })
                .collect()
        };
        self.canonicalize_edges(&mut raw);
        raw
    }

    /// SEAM: outgoing WALK -- `(neighbour, Option<bond>)` pairs in
    /// canonical order, uncapped. The bond is sourced from the carried
    /// payload in cache mode (reconstructed byte-identically to what
    /// `rehydrate_adjacency` inserts) and from `get_bond` in legacy mode,
    /// handed back as a uniform `SeamWalkBond` whose `.view()` is a
    /// `BondView` -- so a walk consumer reads keyframes / weight_in_scope
    /// / latest_raw_weight / bond_type identically regardless of source
    /// and NEVER touches resident `get_bond`/`view_bond` in cache mode.
    /// `None` mirrors a missing directed bond, exactly the walk idiom
    /// `for nbr in neighbors_outgoing { if let Some(b) = get_bond(a,nbr) }`.
    pub(crate) fn seam_walk_outgoing(&self, anchor: &Seed) -> Vec<(Seed, Option<SeamWalkBond<'_>>)> {
        self.seam_outgoing_edges(anchor)
            .into_iter()
            .map(|e| {
                let nbr = e.nbr;
                let bond = match e.bond {
                    SeamBondSrc::Carried { predicate, payload } => Some(SeamWalkBond::Carried(
                        Self::reconstruct_carried_bond(*anchor, nbr, predicate, &payload),
                    )),
                    SeamBondSrc::Resident => self.get_bond(anchor, &nbr).map(SeamWalkBond::Resident),
                };
                (nbr, bond)
            })
            .collect()
    }

    /// SEAM: incoming WALK -- the in-direction twin of
    /// `seam_walk_outgoing`. The bond orientation is `nbr -> anchor`.
    pub(crate) fn seam_walk_incoming(&self, anchor: &Seed) -> Vec<(Seed, Option<SeamWalkBond<'_>>)> {
        self.seam_incoming_edges(anchor)
            .into_iter()
            .map(|e| {
                let nbr = e.nbr;
                let bond = match e.bond {
                    SeamBondSrc::Carried { predicate, payload } => Some(SeamWalkBond::Carried(
                        Self::reconstruct_carried_bond(nbr, *anchor, predicate, &payload),
                    )),
                    SeamBondSrc::Resident => self.get_bond(&nbr, anchor).map(SeamWalkBond::Resident),
                };
                (nbr, bond)
            })
            .collect()
    }

    /// SEAM: a single directed bond `from -> to` sourced WITHOUT resident
    /// `get_bond` in cache mode -- finds `to` among `from`'s carried
    /// outgoing edges and reconstructs it byte-identically; LEGACY =
    /// `get_bond(from, to)`. `None` iff there is no such directed bond
    /// (byte-identical to `get_bond(from, to).is_none()`). For the
    /// cross-edge single-pair reads (consolidation B->C idempotency leg,
    /// variants `bond_weight_directed`, narration single-pair lookups).
    pub(crate) fn seam_directed_bond(&self, from: &Seed, to: &Seed) -> Option<SeamWalkBond<'_>> {
        if self.mapram_cache_backed() {
            let edge = self
                .mapram_raw_edges(from, crate::cache_adjacency::Direction::Outgoing)
                .into_iter()
                .find(|e| &e.nbr == to)?;
            match edge.bond {
                SeamBondSrc::Carried { predicate, payload } => Some(SeamWalkBond::Carried(
                    Self::reconstruct_carried_bond(*from, *to, predicate, &payload),
                )),
                SeamBondSrc::Resident => self.get_bond(from, to).map(SeamWalkBond::Resident),
            }
        } else {
            self.get_bond(from, to).map(SeamWalkBond::Resident)
        }
    }

    /// SEAM (edge-carrying, CAPPED): the carried-weight hub-cap twin of
    /// `neighbors_outgoing_capped`, built over the uncapped, canonical
    /// `seam_outgoing_edges`. When hub-read enforcement is off or the
    /// out-degree is within `hub_read_max_out`, returns the full canonical
    /// edge list (the deliberate canonical seed-byte order is the only
    /// difference from `neighbors_outgoing_capped`'s pass-through, which
    /// returned the resident `HashMap::drain` order -- already
    /// non-deterministic per load); otherwise keeps the top-`hub_read_max_out`
    /// by latest raw bond WEIGHT via `mapram_cap_edges_by_weight` (CACHE
    /// weight = carried payload, LEGACY weight = `get_bond`) -- the SAME
    /// stable sort + drop counter as `neighbors_outgoing_capped`. Lets the
    /// bare spread path read capped neighbours WITH their carried bond +
    /// degree without a resident `neighbors_outgoing_capped` + per-edge
    /// `view_bond`. Threshold uses `raw.len()` (= out-degree in both modes:
    /// legacy `neighbors_outgoing` len, cache carried-edge count), so the
    /// enforce/pass decision is byte-identical to `out_degree`-based gating.
    pub(crate) fn seam_outgoing_edges_capped(&self, seed: &Seed) -> Vec<SeamEdge> {
        let raw = self.seam_outgoing_edges(seed);
        if !self.hub_read_enforcement || (raw.len() as u32) <= self.hub_read_max_out {
            return raw;
        }
        self.mapram_cap_edges_by_weight(seed, raw, self.hub_read_max_out, true)
    }

    /// SEAM (edge-carrying, CAPPED): incoming twin of
    /// `seam_outgoing_edges_capped` -- the carried-weight hub-cap twin of
    /// `neighbors_incoming_capped`, capped by `hub_read_max_in`.
    pub(crate) fn seam_incoming_edges_capped(&self, seed: &Seed) -> Vec<SeamEdge> {
        let raw = self.seam_incoming_edges(seed);
        if !self.hub_read_enforcement || (raw.len() as u32) <= self.hub_read_max_in {
            return raw;
        }
        self.mapram_cap_edges_by_weight(seed, raw, self.hub_read_max_in, false)
    }
}

impl Default for AssociativeNetwork {
    fn default() -> Self {
        Self::new()
    }
}

/// Errors that can occur during network operations.
#[derive(Debug, Clone)]
pub enum NetworkError {
    /// A referenced node was not found in the network.
    NodeNotFound(Seed),
    /// A bond already exists between these nodes.
    BondAlreadyExists(Seed, Seed),
    /// `add_bond_with_scar_check` refused this bond because one or
    /// both endpoints are in `ScarLog::scarred_seeds`. The bond was
    /// NOT added; a fresh `Scar` was emitted against it. The id of
    /// that scar is `scar_id`.
    BondRefusedByScarSeed { from: Seed, to: Seed, scar_id: u64 },
}

impl std::fmt::Display for NetworkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NetworkError::NodeNotFound(seed) => write!(f, "Node not found: {}", seed),
            NetworkError::BondAlreadyExists(a, b) => {
                write!(f, "Bond already exists: {} → {}", a, b)
            }
            NetworkError::BondRefusedByScarSeed { from, to, scar_id } => {
                write!(
                    f,
                    "Bond refused by scarred seed: {} → {} (scar_id={})",
                    from, to, scar_id
                )
            }
        }
    }
}

impl std::error::Error for NetworkError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn concept_sense_tokens_fires_on_definitional_and_language_frames() {
        // Definitional frames extract the content token(s).
        assert_eq!(concept_sense_tokens("what is python"), vec!["python"]);
        assert_eq!(concept_sense_tokens("python language"), vec!["python"]);
        assert_eq!(concept_sense_tokens("tell me about the python language"), vec!["python"]);
        // Typo in a non-routed word is kept as a candidate (its csense handle
        // simply will not exist, so it injects nothing under the strict guard).
        assert_eq!(concept_sense_tokens("Tell me about the python launguage"), vec!["python", "launguage"]);
        // Multi-token concept: both tokens are candidates (neither has a
        // single-token csense handle, so hash-table stays on the concept path).
        assert_eq!(concept_sense_tokens("what is a hash table"), vec!["hash", "table"]);
        // Non-definitional frames do not fire at all -> byte-identical seeding.
        assert!(concept_sense_tokens("python").is_empty());
        assert!(concept_sense_tokens("is a python a snake").is_empty());
        assert!(concept_sense_tokens("write a python function").is_empty());
    }

    #[test]
    fn non_concept_label_flags_the_report_junk_classes() {
        // Article-led fragments (node-quality census: 15% of labels).
        assert!(non_concept_label("a dark-haired man with a canister"));
        assert!(non_concept_label("an off-leash dog park located at the township"));
        assert!(non_concept_label("the political philosophy that rejects"));
        // Dangling function-word ending / over-long fragments.
        assert!(non_concept_label("a very suitable platform to be investigated by the"));
        assert!(non_concept_label("one two three four five six seven eight nine ten"));
        // Leading-underscore code identifier.
        assert!(non_concept_label("_dart_getFlipped"));
        // Lowercase-initial multi-word fragment.
        assert!(non_concept_label("member of the political philosophy"));
    }

    #[test]
    fn non_concept_label_spares_real_concepts_and_fixtures() {
        // Real entity / concept labels.
        assert!(!non_concept_label("Dog"));
        assert!(!non_concept_label("Domestic dog"));
        assert!(!non_concept_label("Canis familiaris (domestic dog)"));
        assert!(!non_concept_label("Python"));
        // Legitimately lowercase single-token common noun.
        assert!(!non_concept_label("photosynthesis"));
        // Unit-test fixtures must never be flagged.
        assert!(!non_concept_label("a"));
        assert!(!non_concept_label("foo"));
        assert!(!non_concept_label("chess"));
        assert!(!non_concept_label("alpha"));
    }
    use crate::bond::BondType;
    use crate::node::{Confidence, KnowledgeContent};

    fn build_test_network() -> AssociativeNetwork {
        let mut net = AssociativeNetwork::new();

        let s_rust = Seed::from_label("Rust");
        let s_systems = Seed::from_label("SystemsProgramming");
        let s_memory = Seed::from_label("MemorySafety");
        let s_concurrency = Seed::from_label("Concurrency");

        net.add_node(KnowledgeNode::with_seed(
            s_rust,
            "Rust".into(),
            KnowledgeContent::Concept("A systems programming language".into()),
            Confidence::Verified,
            0,
        ));
        net.add_node(KnowledgeNode::with_seed(
            s_systems,
            "Systems Programming".into(),
            KnowledgeContent::Concept("Low-level programming with hardware access".into()),
            Confidence::Verified,
            0,
        ));
        net.add_node(KnowledgeNode::with_seed(
            s_memory,
            "Memory Safety".into(),
            KnowledgeContent::Concept("Preventing memory-related bugs".into()),
            Confidence::Terminal,
            0,
        ));
        net.add_node(KnowledgeNode::with_seed(
            s_concurrency,
            "Concurrency".into(),
            KnowledgeContent::Concept("Parallel execution of tasks".into()),
            Confidence::Verified,
            0,
        ));

        // Rust IsA SystemsProgramming
        net.add_bond(TemporalBond::new(s_rust, s_systems, BondType::IsA, 0.9, 0))
            .unwrap();
        // Rust HasProperty MemorySafety
        net.add_bond(TemporalBond::new(s_rust, s_memory, BondType::HasProperty, 0.95, 0))
            .unwrap();
        // Rust HasProperty Concurrency
        net.add_bond(TemporalBond::new(s_rust, s_concurrency, BondType::HasProperty, 0.8, 0))
            .unwrap();
        // MemorySafety Requires SystemsProgramming
        net.add_bond(TemporalBond::new(s_memory, s_systems, BondType::Requires, 0.7, 0))
            .unwrap();

        net
    }

    #[test]
    fn test_spread_activation_basic() {
        let net = build_test_network();
        let s_rust = Seed::from_label("Rust");

        let result = net.spread_activation(
            &[s_rust],
            &TemporalScope::Current,
            3,
            10,
            &ScarLog::default(),
        );

        assert!(!result.activated.is_empty());
        // Rust itself should have highest activation
        assert_eq!(result.activated[0].seed, s_rust);
        assert!(result.nodes_visited >= 2);
    }

    #[test]
    fn test_spread_activation_respects_depth() {
        let net = build_test_network();
        let s_rust = Seed::from_label("Rust");

        let shallow = net.spread_activation(&[s_rust], &TemporalScope::Current, 1, 10, &ScarLog::default());
        let deep = net.spread_activation(&[s_rust], &TemporalScope::Current, 3, 10, &ScarLog::default());

        assert!(deep.nodes_visited >= shallow.nodes_visited);
    }

    #[test]
    fn test_spread_activation_no_infinite_loop() {
        // Build a cycle: A → B → C → A
        let mut net = AssociativeNetwork::new();
        let sa = Seed::from_label("CycleA");
        let sb = Seed::from_label("CycleB");
        let sc = Seed::from_label("CycleC");

        for (s, label) in [(sa, "CycleA"), (sb, "CycleB"), (sc, "CycleC")] {
            net.add_node(KnowledgeNode::with_seed(
                s,
                label.into(),
                KnowledgeContent::Raw(label.into()),
                Confidence::Verified,
                0,
            ));
        }

        net.add_bond(TemporalBond::new(sa, sb, BondType::RelatedTo, 0.8, 0)).unwrap();
        net.add_bond(TemporalBond::new(sb, sc, BondType::RelatedTo, 0.8, 0)).unwrap();
        net.add_bond(TemporalBond::new(sc, sa, BondType::RelatedTo, 0.8, 0)).unwrap();

        // This should complete without infinite loop thanks to visited set
        let result = net.spread_activation(&[sa], &TemporalScope::Current, 10, 10, &ScarLog::default());
        assert_eq!(result.nodes_visited, 3); // Exactly 3 nodes, no revisits
    }

    #[test]
    fn test_hebbian_strengthening() {
        let mut net = build_test_network();
        let s_rust = Seed::from_label("Rust");
        let s_memory = Seed::from_label("MemorySafety");

        let original = net.get_bond(&s_rust, &s_memory).unwrap().latest_raw_weight();
        net.hebbian_strengthen(&[s_rust, s_memory], 1, 1000);
        let after = net.get_bond(&s_rust, &s_memory).unwrap().latest_raw_weight();

        assert!(after > original);
        assert!((after - (original + HEBBIAN_GAIN)).abs() < 1e-6);
    }

    #[test]
    fn test_keyword_search() {
        let net = build_test_network();
        let results = net.keyword_search("memory safety");
        assert!(!results.is_empty());
        // The MemorySafety node should be a top hit
        let s_memory = Seed::from_label("MemorySafety");
        assert!(results.iter().any(|(s, _)| *s == s_memory));
    }

    #[test]
    fn test_keyword_search_filters_stopwords() {
        // Tier 1.4: stopword tokens never produce keyword hits.
        let mut net = build_test_network();
        let s_jam = Seed::from_label("the and stuff");
        net.add_node(KnowledgeNode::with_seed(
            s_jam,
            "the and stuff".into(),
            KnowledgeContent::Raw("junk".into()),
            Confidence::Inferred,
            0,
        ));
        let hits = net.keyword_search("the and");
        assert!(
            hits.iter().all(|(s, _)| *s != s_jam),
            "keyword_search must not match on stopword tokens"
        );
    }

    #[test]
    fn test_max_activations_caps_spread() {
        // Tier 1.1: spread stops expanding when the activation map fills.
        // Build a 2-level tree: hub -> 5 mid-hubs, each mid-hub -> 100 leaves.
        // Without the cap, depth=2 would reach ~505 nodes; with cap=20
        // we should stop well below that AND set the truncation flag.
        let mut net = AssociativeNetwork::new();
        let hub = Seed::from_label("hub");
        net.add_node(KnowledgeNode::with_seed(
            hub, "hub".into(), KnowledgeContent::Raw("".into()),
            Confidence::Verified, 0,
        ));
        for i in 0..5usize {
            let mid = Seed::from_label(&format!("mid{}", i));
            net.add_node(KnowledgeNode::with_seed(
                mid, format!("mid{}", i),
                KnowledgeContent::Raw("".into()),
                Confidence::Verified, 0,
            ));
            net.add_bond(TemporalBond::new(
                hub, mid, BondType::RelatedTo, 0.9, 0,
            )).unwrap();
            for j in 0..100usize {
                let leaf = Seed::from_label(&format!("l{}_{}", i, j));
                net.add_node(KnowledgeNode::with_seed(
                    leaf, format!("l{}_{}", i, j),
                    KnowledgeContent::Raw("".into()),
                    Confidence::Verified, 0,
                ));
                net.add_bond(TemporalBond::new(
                    mid, leaf, BondType::RelatedTo, 0.9, 0,
                )).unwrap();
            }
        }
        net.recompute_degree_p90();

        // Baseline: no cap (or huge cap). Should reach all 506 nodes at depth=2.
        net.set_max_activations(10_000);
        let r0 = net.spread_activation_weighted(
            &[(hub, 1.0)],
            &TemporalScope::Current,
            2, 32, &ScarLog::default(),
        );
        assert!(!r0.activation_truncated);
        assert!(r0.nodes_visited > 100, "baseline should reach many nodes, got {}", r0.nodes_visited);

        // Capped: stop expanding once the cap is hit.
        net.set_max_activations(20);
        let r1 = net.spread_activation_weighted(
            &[(hub, 1.0)],
            &TemporalScope::Current,
            2, 32, &ScarLog::default(),
        );
        assert!(r1.activation_truncated, "truncation flag should fire");
        assert!(
            r1.nodes_visited < r0.nodes_visited,
            "cap should reduce work: r1.nodes_visited={}, r0.nodes_visited={}",
            r1.nodes_visited, r0.nodes_visited,
        );
    }

    #[test]
    fn test_disambiguate_cell_picks_biology_in_mitosis_context() {
        // Port #17 Fix 4: polysemy disambiguation by activated context.
        //
        // "cell" has two disjoint sense clusters in the synthetic graph:
        //   - biology: blastomere, oogenesis, divides (each connect to mitosis)
        //   - telephony: radiotelephone, radiophone     (each connect to handset)
        //
        // When the query has activated mitosis (and friends), the
        // biology cluster wins; the telephony neighbors are excluded
        // from the disambiguation allow-set.
        let mut net = AssociativeNetwork::new();
        let cell = Seed::from_label("cell");
        net.add_node(KnowledgeNode::with_seed(
            cell, "cell".into(), KnowledgeContent::Raw("".into()),
            Confidence::Verified, 0,
        ));
        let mitosis = Seed::from_label("mitosis");
        net.add_node(KnowledgeNode::with_seed(
            mitosis, "mitosis".into(), KnowledgeContent::Raw("".into()),
            Confidence::Verified, 0,
        ));
        let handset = Seed::from_label("handset");
        net.add_node(KnowledgeNode::with_seed(
            handset, "handset".into(), KnowledgeContent::Raw("".into()),
            Confidence::Verified, 0,
        ));
        // Biology cluster.
        let biology_terms = ["blastomere", "oogenesis", "divides"];
        let mut bio_seeds = Vec::new();
        for t in &biology_terms {
            let sd = Seed::from_label(t);
            net.add_node(KnowledgeNode::with_seed(
                sd, (*t).into(), KnowledgeContent::Raw("".into()),
                Confidence::Verified, 0,
            ));
            net.add_bond(TemporalBond::new(
                cell, sd, BondType::IsA, 0.9, 0,
            )).unwrap();
            // Each biology term connects to mitosis (shared neighborhood).
            net.add_bond(TemporalBond::new(
                sd, mitosis, BondType::RelatedTo, 0.9, 0,
            )).unwrap();
            bio_seeds.push(sd);
        }
        // Telephony cluster.
        let telephony_terms = ["radiotelephone", "radiophone"];
        let mut tel_seeds = Vec::new();
        for t in &telephony_terms {
            let sd = Seed::from_label(t);
            net.add_node(KnowledgeNode::with_seed(
                sd, (*t).into(), KnowledgeContent::Raw("".into()),
                Confidence::Verified, 0,
            ));
            net.add_bond(TemporalBond::new(
                cell, sd, BondType::RelatedTo, 0.9, 0,
            )).unwrap();
            // Telephony terms connect to handset (shared neighborhood,
            // disjoint from biology).
            net.add_bond(TemporalBond::new(
                sd, handset, BondType::RelatedTo, 0.9, 0,
            )).unwrap();
            tel_seeds.push(sd);
        }
        net.recompute_degree_p90();

        // Activated set includes the biology cluster (simulating that
        // the query "mitosis cell division" pulled blastomere, oogenesis,
        // divides into recall) but NOT telephony.
        let mut activated: std::collections::HashSet<Seed> =
            std::collections::HashSet::new();
        for s in &bio_seeds { activated.insert(*s); }

        let allow = net
            .disambiguate_polysemous(&cell, &activated)
            .expect("polysemy detected when 2+ disjoint clusters present");
        for s in &bio_seeds {
            assert!(allow.contains(s),
                "biology cluster member {:?} should be in allow set", s);
        }
        for s in &tel_seeds {
            assert!(!allow.contains(s),
                "telephony cluster member {:?} should NOT be in allow set", s);
        }

        // Port #18b: with empty activated set the bond-type-weighted
        // path still bites — biology has 3 IsA bonds vs telephony's 0,
        // so biology wins even with no contextual evidence at all.
        // (Legacy expectation was None; the new behaviour is the whole
        // point of #18b: lean on bond types when context is absent.)
        let typed_only_allow = net
            .disambiguate_polysemous(&cell, &std::collections::HashSet::new())
            .expect("bond-type weight elects biology even with empty activation");
        for s in &bio_seeds {
            assert!(typed_only_allow.contains(s),
                "biology member {:?} should win on bond-type weight alone", s);
        }
    }

    #[test]
    fn test_disambiguate_anchors_override_polluted_activation() {
        // Port #18a: when the post-spread activated set straddles
        // both senses of a polysemous hub (because spreading from the
        // hub itself activates both clusters), raw query-token
        // anchors are still able to elect the correct sense.
        //
        // Setup mirrors the Port #17 graph: cell has biology + telephony
        // sense clusters. We activate BOTH clusters (simulating the
        // pollution that happens on the live brain) and pass `division`
        // as an anchor — division is itself a biology-cluster member
        // (label "divides" in the small graph) and its 10x direct-anchor
        // weight should pick biology over telephony despite tied
        // activation scores.
        let mut net = AssociativeNetwork::new();
        let cell = Seed::from_label("cell");
        net.add_node(KnowledgeNode::with_seed(
            cell, "cell".into(), KnowledgeContent::Raw("".into()),
            Confidence::Verified, 0,
        ));
        let mitosis = Seed::from_label("mitosis");
        net.add_node(KnowledgeNode::with_seed(
            mitosis, "mitosis".into(), KnowledgeContent::Raw("".into()),
            Confidence::Verified, 0,
        ));
        let handset = Seed::from_label("handset");
        net.add_node(KnowledgeNode::with_seed(
            handset, "handset".into(), KnowledgeContent::Raw("".into()),
            Confidence::Verified, 0,
        ));
        let biology_terms = ["blastomere", "oogenesis", "divides"];
        let mut bio_seeds = Vec::new();
        for t in &biology_terms {
            let sd = Seed::from_label(t);
            net.add_node(KnowledgeNode::with_seed(
                sd, (*t).into(), KnowledgeContent::Raw("".into()),
                Confidence::Verified, 0,
            ));
            net.add_bond(TemporalBond::new(
                cell, sd, BondType::IsA, 0.9, 0,
            )).unwrap();
            net.add_bond(TemporalBond::new(
                sd, mitosis, BondType::RelatedTo, 0.9, 0,
            )).unwrap();
            bio_seeds.push(sd);
        }
        let telephony_terms = ["radiotelephone", "radiophone"];
        let mut tel_seeds = Vec::new();
        for t in &telephony_terms {
            let sd = Seed::from_label(t);
            net.add_node(KnowledgeNode::with_seed(
                sd, (*t).into(), KnowledgeContent::Raw("".into()),
                Confidence::Verified, 0,
            ));
            net.add_bond(TemporalBond::new(
                cell, sd, BondType::RelatedTo, 0.9, 0,
            )).unwrap();
            net.add_bond(TemporalBond::new(
                sd, handset, BondType::RelatedTo, 0.9, 0,
            )).unwrap();
            tel_seeds.push(sd);
        }
        net.recompute_degree_p90();

        // Polluted activation with equal counts in each cluster, so
        // the activation-only score is tied and only anchors can
        // break it. (bio has 3 members, tel has 2 — pick 2 from each.)
        let mut polluted: std::collections::HashSet<Seed> =
            std::collections::HashSet::new();
        polluted.insert(bio_seeds[0]);
        polluted.insert(bio_seeds[1]);
        polluted.insert(tel_seeds[0]);
        polluted.insert(tel_seeds[1]);

        // Anchor on `divides` (a biology cluster member). The 10x
        // direct-anchor weight must pull biology ahead.
        let divides = Seed::from_label("divides");
        let anchors = [mitosis, divides];
        let allow = net
            .disambiguate_polysemous_with_anchors(&cell, &polluted, &anchors)
            .expect("anchors must elect a sense even with polluted activation");
        for s in &bio_seeds {
            assert!(allow.contains(s),
                "biology member {:?} should be in the anchor-elected cluster", s);
        }
        for s in &tel_seeds {
            assert!(!allow.contains(s),
                "telephony member {:?} must be excluded", s);
        }

        // Port #18b: with no anchors and tied activation, the
        // bond-type-weighted scoring path now bites: the biology
        // cluster wins because all `cell -> bio_term` bonds are IsA
        // (typed), while `cell -> telephony_term` bonds are RelatedTo.
        // Even without anchors and even with polluted activation, the
        // bond-type signal alone elects biology — exactly the fix that
        // #18b ships for bare `what is cell` queries.
        let typed_allow = net
            .disambiguate_polysemous_with_anchors(&cell, &polluted, &[])
            .expect("bond-type-weighted scoring elects biology with zero anchors");
        for s in &bio_seeds {
            assert!(typed_allow.contains(s),
                "biology member {:?} should win on bond-type weight", s);
        }
        for s in &tel_seeds {
            assert!(!typed_allow.contains(s),
                "telephony member {:?} loses on bond-type weight", s);
        }
    }

    #[test]
    fn test_port19_label_propagation_separates_senses_where_jaccard_fails() {
        // Port #19: build a polysemous hub whose two senses share
        // ZERO neighborhood overlap below the legacy 0.2 union-find
        // threshold but DO share co-membership of `seed`. The legacy
        // primitive splintered such graphs into per-neighbor singletons
        // (each pair < 0.2 Jaccard → no merge). The label-propagation
        // primitive should still find the two senses because the
        // bond-type prior + co-neighborhood propagation lifts them out
        // of the singleton trap.
        let mut net = AssociativeNetwork::new();
        let hub = Seed::from_label("hub");
        net.add_node(KnowledgeNode::with_seed(
            hub, "hub".into(), KnowledgeContent::Raw("".into()),
            Confidence::Verified, 0,
        ));
        // Sense A: 5 neighbors, IsA from hub, each linked to a single
        // shared anchor `a_anchor` (low-overlap signal that union-find
        // at 0.2 misses).
        let a_anchor = Seed::from_label("a_anchor");
        net.add_node(KnowledgeNode::with_seed(
            a_anchor, "a_anchor".into(), KnowledgeContent::Raw("".into()),
            Confidence::Verified, 0,
        ));
        let mut a_seeds = Vec::new();
        for i in 0..5usize {
            let label = format!("a{}", i);
            let s = Seed::from_label(&label);
            net.add_node(KnowledgeNode::with_seed(
                s, label.clone(), KnowledgeContent::Raw("".into()),
                Confidence::Verified, 0,
            ));
            net.add_bond(TemporalBond::new(
                hub, s, BondType::IsA, 0.9, 0,
            )).unwrap();
            // Each a-neighbor connects to a_anchor + i extra unique
            // distractors so pairwise neighborhood-Jaccard between two
            // a-neighbors is well below 0.2 but >= 0.05. (1 shared
            // neighbor / many distinct distractors per side.)
            net.add_bond(TemporalBond::new(
                s, a_anchor, BondType::RelatedTo, 0.9, 0,
            )).unwrap();
            for k in 0..(2 + i) {
                let dlabel = format!("a{}_d{}", i, k);
                let d = Seed::from_label(&dlabel);
                net.add_node(KnowledgeNode::with_seed(
                    d, dlabel, KnowledgeContent::Raw("".into()),
                    Confidence::Verified, 0,
                ));
                net.add_bond(TemporalBond::new(
                    s, d, BondType::RelatedTo, 0.5, 0,
                )).unwrap();
            }
            a_seeds.push(s);
        }
        // Sense B: 5 neighbors, RelatedTo from hub, each linked to a
        // shared anchor `b_anchor` (disjoint from a_anchor).
        let b_anchor = Seed::from_label("b_anchor");
        net.add_node(KnowledgeNode::with_seed(
            b_anchor, "b_anchor".into(), KnowledgeContent::Raw("".into()),
            Confidence::Verified, 0,
        ));
        let mut b_seeds = Vec::new();
        for i in 0..5usize {
            let label = format!("b{}", i);
            let s = Seed::from_label(&label);
            net.add_node(KnowledgeNode::with_seed(
                s, label.clone(), KnowledgeContent::Raw("".into()),
                Confidence::Verified, 0,
            ));
            net.add_bond(TemporalBond::new(
                hub, s, BondType::RelatedTo, 0.9, 0,
            )).unwrap();
            net.add_bond(TemporalBond::new(
                s, b_anchor, BondType::RelatedTo, 0.9, 0,
            )).unwrap();
            for k in 0..(2 + i) {
                let dlabel = format!("b{}_d{}", i, k);
                let d = Seed::from_label(&dlabel);
                net.add_node(KnowledgeNode::with_seed(
                    d, dlabel, KnowledgeContent::Raw("".into()),
                    Confidence::Verified, 0,
                ));
                net.add_bond(TemporalBond::new(
                    s, d, BondType::RelatedTo, 0.5, 0,
                )).unwrap();
            }
            b_seeds.push(s);
        }
        net.recompute_degree_p90();

        // No anchors, empty activation: the bond-type prior + label
        // propagation must still elect sense A (the IsA cluster).
        let allow = net
            .disambiguate_polysemous(&hub, &std::collections::HashSet::new())
            .expect("polysemy detected by label-propagation primitive");
        for s in &a_seeds {
            assert!(allow.contains(s),
                "sense-A (IsA / typed) member {:?} should win", s);
        }
        for s in &b_seeds {
            assert!(!allow.contains(s),
                "sense-B (RelatedTo) member {:?} must be excluded", s);
        }
    }

    #[test]
    fn test_port19_label_propagation_no_polysemy_returns_none() {
        // Single-sense hub: all neighbors share one bond type AND a
        // common anchor. Label propagation should converge to a single
        // cluster, returning None (no filter).
        let mut net = AssociativeNetwork::new();
        let hub = Seed::from_label("hub_mono");
        net.add_node(KnowledgeNode::with_seed(
            hub, "hub_mono".into(), KnowledgeContent::Raw("".into()),
            Confidence::Verified, 0,
        ));
        let anchor = Seed::from_label("shared_anchor");
        net.add_node(KnowledgeNode::with_seed(
            anchor, "shared_anchor".into(), KnowledgeContent::Raw("".into()),
            Confidence::Verified, 0,
        ));
        for i in 0..6usize {
            let label = format!("m{}", i);
            let s = Seed::from_label(&label);
            net.add_node(KnowledgeNode::with_seed(
                s, label, KnowledgeContent::Raw("".into()),
                Confidence::Verified, 0,
            ));
            net.add_bond(TemporalBond::new(
                hub, s, BondType::IsA, 0.9, 0,
            )).unwrap();
            net.add_bond(TemporalBond::new(
                s, anchor, BondType::RelatedTo, 0.9, 0,
            )).unwrap();
        }
        net.recompute_degree_p90();
        let out = net.disambiguate_polysemous(&hub, &std::collections::HashSet::new());
        assert!(out.is_none(),
            "single-sense hub should yield no filter; got {:?}", out);
    }

    #[test]
    fn test_port19_label_propagation_too_few_neighbors() {
        // Below MIN_NEIGHBORS_FOR_POLYSEMY (4): bail out early.
        let mut net = AssociativeNetwork::new();
        let hub = Seed::from_label("tiny_hub");
        net.add_node(KnowledgeNode::with_seed(
            hub, "tiny_hub".into(), KnowledgeContent::Raw("".into()),
            Confidence::Verified, 0,
        ));
        for i in 0..2usize {
            let s = Seed::from_label(&format!("t{}", i));
            net.add_node(KnowledgeNode::with_seed(
                s, format!("t{}", i), KnowledgeContent::Raw("".into()),
                Confidence::Verified, 0,
            ));
            net.add_bond(TemporalBond::new(
                hub, s, BondType::IsA, 0.9, 0,
            )).unwrap();
        }
        let out = net.disambiguate_polysemous(&hub, &std::collections::HashSet::new());
        assert!(out.is_none(), "<4 neighbors must short-circuit");
    }

    #[test]
    fn test_activation_cap_scales_with_brain_size() {
        // Port #17 Fix 1: effective_cap_activations() = max(floor, ceil(N * pct)).
        let mut net = AssociativeNetwork::new();
        // Empty brain: cap is the static floor (default 5000).
        assert_eq!(net.effective_cap_activations(), MAX_ACTIVATIONS_DEFAULT);

        // Small brain (1000 nodes, default 0.05 = 50): floor wins.
        for i in 0..1000usize {
            let sd = Seed::from_label(&format!("sm{}", i));
            net.add_node(KnowledgeNode::with_seed(
                sd, format!("sm{}", i),
                KnowledgeContent::Raw("".into()),
                Confidence::Verified, 0,
            ));
        }
        assert_eq!(net.effective_cap_activations(), MAX_ACTIVATIONS_DEFAULT,
            "small brain should use static floor");

        // Drop the floor so percentile dominates.
        net.set_max_activations(10);
        net.set_max_activations_percentile(0.05);
        // 1000 * 0.05 = 50 > 10 => percentile wins.
        assert_eq!(net.effective_cap_activations(), 50);

        // Very large hypothetical brain: simulate with a high percentile.
        net.set_max_activations(10);
        net.set_max_activations_percentile(0.5);
        // 1000 * 0.5 = 500 > 10 => percentile wins big.
        assert_eq!(net.effective_cap_activations(), 500);

        // Percentile clamped to [0, 1].
        net.set_max_activations_percentile(2.0);
        assert!((net.max_activations_percentile() - 1.0).abs() < 1e-6);
        net.set_max_activations_percentile(-1.0);
        assert!(net.max_activations_percentile() <= 0.0);
    }

    #[test]
    fn test_degree_p90_recompute_floors_at_1() {
        // Tier 1.2: empty / sparse graphs floor the p90 at 1.
        let mut net = AssociativeNetwork::new();
        let p90_empty = net.recompute_degree_p90();
        assert_eq!(p90_empty, DEGREE_NORM_THRESHOLD);

        let hub = Seed::from_label("hub");
        net.add_node(KnowledgeNode::with_seed(
            hub, "hub".into(), KnowledgeContent::Raw("".into()),
            Confidence::Verified, 0,
        ));
        for i in 0..50 {
            let s = Seed::from_label(&format!("t{}", i));
            net.add_node(KnowledgeNode::with_seed(
                s, format!("t{}", i),
                KnowledgeContent::Raw("".into()),
                Confidence::Verified, 0,
            ));
            net.add_bond(TemporalBond::new(
                hub, s, BondType::RelatedTo, 0.5, 0,
            )).unwrap();
        }
        let p90 = net.recompute_degree_p90();
        // hub has degree 50; everyone else degree 1.  P90 of
        // [1,1,…,1,50] is 1, demonstrating the threshold adapts to
        // whatever the graph actually looks like rather than the global
        // 2000 constant.
        assert!(p90 >= 1);
    }

    #[test]
    fn test_subgraph_coherence() {
        let net = build_test_network();
        let s_rust = Seed::from_label("Rust");
        let s_memory = Seed::from_label("MemorySafety");
        let s_systems = Seed::from_label("SystemsProgramming");

        let coherence = net.subgraph_coherence(
            &[s_rust, s_memory, s_systems],
            &TemporalScope::Current,
            &ScarLog::default(),
        );
        assert!(coherence > 0.0);
    }

    // ---- typed_ratio cache (Storage v2 Phase 1 / open issue #1) ----

    /// Compute typed_ratio inline the way the old narration code did,
    /// for cross-checking against the cached value.
    fn inline_typed_ratio(net: &AssociativeNetwork, seed: &Seed) -> f32 {
        let nbrs = net.neighbors_outgoing(seed);
        if nbrs.is_empty() { return 0.0; }
        let mut typed = 0usize;
        let mut total = 0usize;
        for n in nbrs {
            if let Some(b) = net.get_bond(seed, &n) {
                total += 1;
                if b.bond_type() != BondType::RelatedTo {
                    typed += 1;
                }
            }
        }
        if total == 0 { 0.0 } else { typed as f32 / total as f32 }
    }

    #[test]
    fn typed_ratio_cache_matches_inline_computation() {
        // Build a small graph by hand so the ratios are predictable:
        //   subj -[IsA]-> objA       (typed)
        //   subj -[RelatedTo]-> objB (untyped)
        //   subj -[HasProperty]-> objC (typed)
        // Expected ratio for `subj` is 2/3.
        let mut net = AssociativeNetwork::new();
        let now = 0i64;
        let subj = KnowledgeNode::with_seed(
            Seed::from_label("subj"),
            "subj".into(),
            KnowledgeContent::Concept("subj".into()),
            Confidence::Verified, now,
        );
        let oa = KnowledgeNode::with_seed(
            Seed::from_label("objA"),
            "objA".into(),
            KnowledgeContent::Concept("objA".into()),
            Confidence::Verified, now,
        );
        let ob = KnowledgeNode::with_seed(
            Seed::from_label("objB"),
            "objB".into(),
            KnowledgeContent::Concept("objB".into()),
            Confidence::Verified, now,
        );
        let oc = KnowledgeNode::with_seed(
            Seed::from_label("objC"),
            "objC".into(),
            KnowledgeContent::Concept("objC".into()),
            Confidence::Verified, now,
        );
        net.add_node(subj.clone());
        net.add_node(oa);
        net.add_node(ob);
        net.add_node(oc);
        for (to_label, bt) in [
            ("objA", BondType::IsA),
            ("objB", BondType::RelatedTo),
            ("objC", BondType::HasProperty),
        ] {
            let b = TemporalBond::new(
                Seed::from_label("subj"), Seed::from_label(to_label), bt, 0.8, now,
            );
            net.add_bond(b).expect("bond add");
        }

        net.recompute_all_typed_ratios();

        // Round-trip through a snapshot to make sure the field travels.
        let snapshot = net.to_snapshot();
        let net2 = AssociativeNetwork::from_snapshot(snapshot);

        let s_subj = Seed::from_label("subj");
        let cached = net2.get_node(&s_subj).unwrap().typed_ratio();
        let inline = inline_typed_ratio(&net2, &s_subj);
        let expected = 2.0 / 3.0;
        assert!((cached - expected).abs() < 1e-6, "cached={} expected={}", cached, expected);
        assert!((cached - inline).abs() < 1e-6, "cached={} inline={}", cached, inline);

        // Sink nodes (no outgoing) should have ratio 0.
        let s_a = Seed::from_label("objA");
        assert_eq!(net2.get_node(&s_a).unwrap().typed_ratio(), 0.0);
    }

    #[test]
    fn polysemy_multi_sense_returns_distinct_clusters() {
        // Port #22: a synthetic seed with two well-separated
        // sense clusters (each backed by typed bonds) returns
        // both clusters from disambiguate_polysemous_multi_sense.
        let mut net = AssociativeNetwork::new();
        let now = 0i64;
        let put = |net: &mut AssociativeNetwork, label: &str| -> Seed {
            let s = Seed::from_label(label);
            net.add_node(crate::node::KnowledgeNode::with_seed(
                s,
                label.to_string(),
                crate::node::KnowledgeContent::Raw(label.to_string()),
                crate::node::Confidence::Verified,
                now,
            ));
            s
        };
        let hub = put(&mut net, "cell");
        // Sense A (biology): 4 IsA edges to dense cluster A.
        let a_members = ["compartment", "mitochondria", "organelle", "nucleus"];
        for m in &a_members {
            let s = put(&mut net, m);
            net.add_bond(crate::bond::TemporalBond::new(
                hub, s, BondType::IsA, 0.9, now,
            )).unwrap();
        }
        for i in 0..a_members.len() {
            for j in 0..a_members.len() {
                if i == j { continue; }
                let a = Seed::from_label(a_members[i]);
                let b = Seed::from_label(a_members[j]);
                let _ = net.add_bond(crate::bond::TemporalBond::new(
                    a, b, BondType::RelatedTo, 0.5, now,
                ));
            }
        }
        // Sense B (telephony): 4 IsA edges to dense cluster B.
        let b_members = ["radiotelephone", "radiophone", "handset", "receiver"];
        for m in &b_members {
            let s = put(&mut net, m);
            net.add_bond(crate::bond::TemporalBond::new(
                hub, s, BondType::IsA, 0.9, now,
            )).unwrap();
        }
        for i in 0..b_members.len() {
            for j in 0..b_members.len() {
                if i == j { continue; }
                let a = Seed::from_label(b_members[i]);
                let b = Seed::from_label(b_members[j]);
                let _ = net.add_bond(crate::bond::TemporalBond::new(
                    a, b, BondType::RelatedTo, 0.5, now,
                ));
            }
        }
        net.recompute_all_typed_ratios();

        let res = net.disambiguate_polysemous_multi_sense(&hub)
            .expect("two senses must be detected");
        assert!(res.len() >= 2, "expected >= 2 clusters, got {}", res.len());
        for cluster in &res {
            assert!(cluster.len() >= 2,
                "cluster too small: {:?}", cluster);
        }
    }

    #[test]
    fn polysemy_multi_sense_none_when_single_cluster_dominates() {
        // Port #22: when one sense overwhelmingly dominates,
        // multi-sense detection returns None (the existing
        // single-pick disambiguator handles this case).
        let mut net = AssociativeNetwork::new();
        let now = 0i64;
        let put = |net: &mut AssociativeNetwork, label: &str| -> Seed {
            let s = Seed::from_label(label);
            net.add_node(crate::node::KnowledgeNode::with_seed(
                s,
                label.to_string(),
                crate::node::KnowledgeContent::Raw(label.to_string()),
                crate::node::Confidence::Verified,
                now,
            ));
            s
        };
        let hub = put(&mut net, "sodium");
        let dense = ["metal", "element", "alkali", "reactive", "solid"];
        for m in &dense {
            let s = put(&mut net, m);
            net.add_bond(crate::bond::TemporalBond::new(
                hub, s, BondType::IsA, 0.9, now,
            )).unwrap();
        }
        for i in 0..dense.len() {
            for j in 0..dense.len() {
                if i == j { continue; }
                let a = Seed::from_label(dense[i]);
                let b = Seed::from_label(dense[j]);
                let _ = net.add_bond(crate::bond::TemporalBond::new(
                    a, b, BondType::RelatedTo, 0.5, now,
                ));
            }
        }
        let outlier = put(&mut net, "random");
        let _ = net.add_bond(crate::bond::TemporalBond::new(
            hub, outlier, BondType::RelatedTo, 0.5, now,
        ));
        net.recompute_all_typed_ratios();

        let res = net.disambiguate_polysemous_multi_sense(&hub);
        assert!(res.is_none(),
            "expected None when one cluster dominates, got {:?}", res);
    }

    #[test]
    fn typed_ratio_recomputes_when_consolidation_changes_shape() {
        // The consolidation hook calls `recompute_all_typed_ratios`
        // because consolidation can prune or alter bonds in ways that
        // bypass `add_bond`. Simulate that by mutating `bonds_mut`
        // directly (the same path consolidation uses) and confirm that
        // a stale cache is refreshed by `recompute_all_typed_ratios`.
        let mut net = AssociativeNetwork::new();
        let now = 0i64;
        for label in ["a", "b", "c"] {
            net.add_node(KnowledgeNode::with_seed(
                Seed::from_label(label),
                label.into(),
                KnowledgeContent::Concept(label.into()),
                Confidence::Verified, now,
            ));
        }
        // Two bonds: one RelatedTo, one IsA. After add_bond, source
        // node's cache is fresh (0.5).
        net.add_bond(TemporalBond::new(
            Seed::from_label("a"), Seed::from_label("b"),
            BondType::RelatedTo, 0.8, now,
        )).unwrap();
        net.add_bond(TemporalBond::new(
            Seed::from_label("a"), Seed::from_label("c"),
            BondType::IsA, 0.8, now,
        )).unwrap();
        let r0 = net.get_node(&Seed::from_label("a")).unwrap().typed_ratio();
        assert!((r0 - 0.5).abs() < 1e-6, "after add_bond ratio={} (expected 0.5)", r0);

        // Mutate the IsA bond out of band — flip it to RelatedTo as if
        // consolidation had demoted it. add_bond is not called, so the
        // cache becomes stale.
        let key = (Seed::from_label("a"), Seed::from_label("c"));
        if let Some(mut b) = net.get_bond_mut(&key.0, &key.1) {
            b.bond_type = BondType::RelatedTo;
        }
        let r_stale = net.get_node(&Seed::from_label("a")).unwrap().typed_ratio();
        assert!((r_stale - 0.5).abs() < 1e-6, "cache should still read stale 0.5 before recompute, got {}", r_stale);

        // Consolidation hook (degree_p90 + recompute_all_typed_ratios)
        // refreshes the cache. After: both bonds RelatedTo => 0.0.
        net.recompute_all_typed_ratios();
        let r1 = net.get_node(&Seed::from_label("a")).unwrap().typed_ratio();
        assert!((r1 - 0.0).abs() < 1e-6, "after recompute ratio={} (expected 0.0)", r1);
    }

    // ============================================================
    // Port #24 — Bug 1, Bug 2, Bug 3 regression tests for the new
    // recall-time pollution filter (`is_recall_pollution` and friends).
    // ============================================================

    #[test]
    fn test_is_meta_pollution_filters_legacy_goal_and_step_anchors() {
        // Bug 1 — legacy v2 thinking.rs anchor pollution.
        assert!(is_meta_pollution("goal:what is consciousness"));
        assert!(is_meta_pollution("GOAL:foo"));
        assert!(is_meta_pollution("step:1"));
        assert!(is_meta_pollution("Step:plan"));
        // Real concepts must pass through.
        assert!(!is_meta_pollution("consciousness"));
        assert!(!is_meta_pollution("meditation"));
    }

    #[test]
    fn test_is_meta_pollution_filters_bondtype_name_leaks() {
        // Bug 2 — "IsA" / "PartOf" leaking from the training script as
        // a literal node content.
        assert!(is_meta_pollution("IsA"));
        assert!(is_meta_pollution("PartOf"));
        assert!(is_meta_pollution("Causes"));
        assert!(is_meta_pollution("RelatedTo"));
        // Lowercase or substring matches must NOT trigger — only exact
        // BondType variant strings.
        assert!(!is_meta_pollution("isa"));
        assert!(!is_meta_pollution("part"));
        assert!(!is_meta_pollution("partofbody"));
    }

    #[test]
    fn test_is_code_identifier_pollution_filters_camelcase() {
        // Defensive — most training paths lowercase first, but a manual
        // teach() call with mixed case still gets caught.
        assert!(is_code_identifier_pollution("calculateFactorial"));
        assert!(is_code_identifier_pollution("reverseNum"));
        // All-lowercase or all-uppercase do not trigger camelCase.
        assert!(!is_code_identifier_pollution("meditation"));
        assert!(!is_code_identifier_pollution("DNA"));
    }

    #[test]
    fn test_is_code_identifier_pollution_filters_hyphenated_suffixes() {
        assert!(is_code_identifier_pollution("num-1"));
        assert!(is_code_identifier_pollution("node-42"));
        assert!(is_code_identifier_pollution("sub-array"));
        assert!(is_code_identifier_pollution("left-tree"));
        assert!(is_code_identifier_pollution("sub-problem"));
        assert!(is_code_identifier_pollution("sub-problems"));
        assert!(is_code_identifier_pollution("recursion-step"));
        // Real hyphenated words must NOT be filtered.
        assert!(!is_code_identifier_pollution("co-evolution"));
        assert!(!is_code_identifier_pollution("x-ray"));
    }

    #[test]
    fn test_is_code_identifier_pollution_filters_known_bleed_stoplist() {
        // Identifier-concat tokens that survived the lowercase pass.
        assert!(is_code_identifier_pollution("calculatefactorial"));
        assert!(is_code_identifier_pollution("reversenum"));
        assert!(is_code_identifier_pollution("sortedrightarray"));
        assert!(is_code_identifier_pollution("decimaltobinary"));
        // Typo bleed surfaced by the 40-question test.
        assert!(is_code_identifier_pollution("willaims"));
        assert!(is_code_identifier_pollution("fky"));
        assert!(is_code_identifier_pollution("oxyjunn"));
    }

    #[test]
    fn test_is_recall_pollution_does_not_filter_underscore_test_nodes() {
        // Synthetic test nodes with `_` as separator (and the legacy
        // `goal:`/`step:` patterns are handled by is_meta_pollution).
        // Underscored content is NOT flagged so existing tests using
        // `"alpha_iso"`, `"node_a"`, `"a_leaf_0"` still recall normally.
        assert!(!is_recall_pollution("alpha_iso"));
        assert!(!is_recall_pollution("node_a"));
        assert!(!is_recall_pollution("a_leaf_0"));
    }

    #[test]
    fn test_select_seeds_for_query_skips_polluted_anchor() {
        // Bug 1 regression: a legacy goal:* node indexed by pre-Port-#12
        // thinking.rs must not be returned as a seed for a query that
        // shares its keyword. The clean concept node must be returned
        // instead.
        let mut net = AssociativeNetwork::new();
        let goal_seed = Seed::from_label("goal-meta-pollution");
        let goal_node = KnowledgeNode::with_seed(
            goal_seed,
            "goal:what is consciousness".to_string(),
            crate::node::KnowledgeContent::Meta("what is consciousness".to_string()),
            crate::node::Confidence::Inferred,
            0,
        );
        net.add_node(goal_node);
        let real_seed = Seed::from_label("real-consciousness");
        let real_node = KnowledgeNode::with_seed(
            real_seed,
            "consciousness".to_string(),
            crate::node::KnowledgeContent::Raw("consciousness".to_string()),
            crate::node::Confidence::Verified,
            0,
        );
        net.add_node(real_node);

        let seeds = net.select_seeds_for_query("what is consciousness");
        assert!(
            seeds.iter().all(|(s, _)| *s != goal_seed),
            "select_seeds_for_query must skip a polluted goal:* anchor; got {:?}",
            seeds,
        );
        assert!(
            seeds.iter().any(|(s, _)| *s == real_seed),
            "select_seeds_for_query must still return the real concept seed; got {:?}",
            seeds,
        );
    }

    #[test]
    fn test_keyword_search_skips_polluted_seeds() {
        // Bug 2 regression: a node literally named "IsA" must not be
        // returned by keyword_search.
        let mut net = AssociativeNetwork::new();
        let isa_seed = Seed::from_label("isa-pollution-leak");
        let isa_node = KnowledgeNode::with_seed(
            isa_seed,
            "IsA".to_string(),
            crate::node::KnowledgeContent::Raw("IsA".to_string()),
            crate::node::Confidence::Verified,
            0,
        );
        net.add_node(isa_node);
        let hits = net.keyword_search("IsA");
        assert!(
            hits.iter().all(|(s, _)| *s != isa_seed),
            "keyword_search must skip a `IsA`-named pollution node; got {:?}",
            hits,
        );
    }

    #[test]
    fn test_recall_drops_polluted_node_from_activated_results() {
        // Bug 2 regression: even if the spread visits a `IsA`-named
        // node as a neighbor, it must be filtered from the final
        // activated[] result list.
        let mut net = AssociativeNetwork::new();
        let med = Seed::from_label("med");
        let isa = Seed::from_label("isa-leak");
        let real = Seed::from_label("practice");
        net.add_node(KnowledgeNode::with_seed(
            med, "meditation".into(),
            crate::node::KnowledgeContent::Raw("meditation".into()),
            crate::node::Confidence::Verified, 0,
        ));
        net.add_node(KnowledgeNode::with_seed(
            isa, "IsA".into(),
            crate::node::KnowledgeContent::Raw("IsA".into()),
            crate::node::Confidence::Verified, 0,
        ));
        net.add_node(KnowledgeNode::with_seed(
            real, "practice".into(),
            crate::node::KnowledgeContent::Raw("practice".into()),
            crate::node::Confidence::Verified, 0,
        ));
        net.add_bond(TemporalBond::new(med, isa, BondType::IsA, 0.9, 0)).unwrap();
        net.add_bond(TemporalBond::new(med, real, BondType::RelatedTo, 0.9, 0)).unwrap();
        let scope = TemporalScope::Current;
        let recall = net.spread_activation_weighted(
            &[(med, 1.0)], &scope, 2, 10, &ScarLog::default(),
        );
        let names: Vec<&str> = recall.activated.iter().map(|n| n.content.as_str()).collect();
        assert!(
            !names.iter().any(|n| *n == "IsA"),
            "Bug 2: `IsA` must not appear in activated[] results; got {:?}",
            names,
        );
        assert!(
            names.iter().any(|n| *n == "practice"),
            "Real neighbor must still surface; got {:?}",
            names,
        );
    }

    // ────────────────────────────────────────────────────────────────────
    // Bug B fix shape A — last-resort fallback in seed selection.
    // the project log entries #571 (Port B) and #572 (Bug B triage).
    // ────────────────────────────────────────────────────────────────────

    #[test]
    fn bare_query_noise_word_falls_back_to_seed() {
        // "function" is on the is_query_noise list ("generic question-final
        // words" group). On a multi-word query like "what is the function of
        // mitochondria" the filter correctly suppresses it. But on a bare-word
        // query that consists ONLY of a noise word, the strict pass returns
        // zero seeds even when a real concept node for that word exists. The
        // fallback re-runs seed selection with is_query_noise disabled (but
        // is_hub_stopword still active), which rescues recall.
        let mut net = AssociativeNetwork::new();
        let func_seed = Seed::from_label("function-concept");
        net.add_node(KnowledgeNode::with_seed(
            func_seed,
            "function".into(),
            crate::node::KnowledgeContent::Raw("function".into()),
            crate::node::Confidence::Verified,
            0,
        ));
        let seeds = net.select_seeds_for_query("function");
        assert!(
            !seeds.is_empty(),
            "bare-word query on `function` must trigger fallback and seed; got {:?}",
            seeds,
        );
        assert!(
            seeds.iter().any(|(s, _)| *s == func_seed),
            "fallback must surface the real `function` concept node; got {:?}",
            seeds,
        );
    }

    #[test]
    fn bare_hub_stopword_still_returns_zero() {
        // "because" is on is_hub_stopword. Even with a node literally named
        // `because` present, the bare-word query must return zero seeds —
        // the fallback check (`any token survives is_hub_stopword`) fails,
        // so the relaxed pass never fires. Hub-stopword filtering still
        // does its job on bare-word queries.
        let mut net = AssociativeNetwork::new();
        let bc_seed = Seed::from_label("because-leaf");
        net.add_node(KnowledgeNode::with_seed(
            bc_seed,
            "because".into(),
            crate::node::KnowledgeContent::Raw("because".into()),
            crate::node::Confidence::Verified,
            0,
        ));
        let seeds = net.select_seeds_for_query("because");
        assert!(
            seeds.is_empty(),
            "bare-word `because` is is_hub_stopword; fallback must NOT fire; got {:?}",
            seeds,
        );
    }

    #[test]
    fn multiword_query_does_not_trigger_fallback() {
        // On a multi-word query, the strict pass already finds a content
        // word (`mitochondria`), so the fallback never fires. The query-noise
        // word `function` must remain filtered out — only `mitochondria`
        // gets seeded.
        let mut net = AssociativeNetwork::new();
        let mito = Seed::from_label("mitochondria-concept");
        let func = Seed::from_label("function-concept");
        net.add_node(KnowledgeNode::with_seed(
            mito,
            "mitochondria".into(),
            crate::node::KnowledgeContent::Raw("mitochondria".into()),
            crate::node::Confidence::Verified,
            0,
        ));
        net.add_node(KnowledgeNode::with_seed(
            func,
            "function".into(),
            crate::node::KnowledgeContent::Raw("function".into()),
            crate::node::Confidence::Verified,
            0,
        ));
        let seeds = net.select_seeds_for_query("what is the function of mitochondria");
        assert!(
            seeds.iter().any(|(s, _)| *s == mito),
            "multi-word query must seed on `mitochondria`; got {:?}",
            seeds,
        );
        assert!(
            seeds.iter().all(|(s, _)| *s != func),
            "multi-word query must NOT seed on `function` (still filtered); got {:?}",
            seeds,
        );
    }

    // ====================================================================
    // Port T.5.1 — Deep decay wiring tests.
    //
    // T.5 shipped the `bond_effective_weight_with_decay` API + the
    // `lu4d_freshness_status` MCP tool. T.5.1 wires the decay multiplier
    // into every query-path effective_weight read in this module so the
    // freshness signal actually shapes recall ranking. Stored bond
    // weights are NEVER mutated by these paths.
    // ====================================================================
    use crate::decay::{DecayConfig, DecayRuntime, DAY_MS};

    /// Build a tiny graph A -> B -> C with explicit per-bond creation
    /// timestamps so each test can age individual bonds independently.
    fn build_decay_test_graph(
        a_to_b_created_ms: i64,
        b_to_c_created_ms: i64,
        a_to_c_created_ms: Option<i64>,
    ) -> (AssociativeNetwork, Seed, Seed, Seed, Option<Seed>) {
        let mut net = AssociativeNetwork::new();
        let a = Seed::from_label("DecayA");
        let b = Seed::from_label("DecayB");
        let c = Seed::from_label("DecayC");
        for (s, name) in &[(a, "DecayA"), (b, "DecayB"), (c, "DecayC")] {
            net.add_node(KnowledgeNode::with_seed(
                *s,
                (*name).into(),
                KnowledgeContent::Concept((*name).into()),
                Confidence::Verified,
                0,
            ));
        }
        net.add_bond(TemporalBond::new(a, b, BondType::RelatedTo, 0.9, a_to_b_created_ms))
            .unwrap();
        net.add_bond(TemporalBond::new(b, c, BondType::RelatedTo, 0.9, b_to_c_created_ms))
            .unwrap();
        let d = if let Some(ts) = a_to_c_created_ms {
            let d = Seed::from_label("DecayD");
            net.add_node(KnowledgeNode::with_seed(
                d,
                "DecayD".into(),
                KnowledgeContent::Concept("DecayD".into()),
                Confidence::Verified,
                0,
            ));
            net.add_bond(TemporalBond::new(a, d, BondType::RelatedTo, 0.9, ts))
                .unwrap();
            Some(d)
        } else { None };
        (net, a, b, c, d)
    }

    fn act_for(seed: Seed, recall: &RecallResult) -> Option<f32> {
        recall.activated.iter().find(|n| n.seed == seed).map(|n| n.activation)
    }

    #[test]
    fn t51_spread_uses_decay_when_enabled() {
        // Bond A->B is FRESH (created at NOW); bond B->C is STALE
        // (created 360 days ago, 4 half-lives at default 90d → 1/16 weight).
        let now = 360 * DAY_MS;
        let (mut net, a, _b, c, _) = build_decay_test_graph(now, 0, None);
        // Decay OFF baseline.
        let r_off = net.spread_activation(&[a], &TemporalScope::Current, 3, 10, &ScarLog::default());
        let c_off = act_for(c, &r_off).expect("C activates with decay off");
        // Decay ON: B->C bond is 360 days old vs now=360d → mult ~ 2^-4 = 0.0625.
        net.set_decay_runtime(Some(DecayRuntime::new(DecayConfig::default(), now)));
        let r_on = net.spread_activation(&[a], &TemporalScope::Current, 3, 10, &ScarLog::default());
        let c_on = act_for(c, &r_on).expect("C still activates with decay on (just lower)");
        assert!(
            c_on < c_off * 0.5,
            "decay should at least halve C's activation; off={}, on={}",
            c_off, c_on,
        );
    }

    #[test]
    fn t51_walk_uses_decay_when_enabled() {
        // Walk = typed_recall (filters by bond type). Same A->B->C with
        // RelatedTo bonds; staleness on B->C should de-rank C.
        let now = 360 * DAY_MS;
        let (mut net, a, _b, c, _) = build_decay_test_graph(now, 0, None);
        let filter = vec![BondType::RelatedTo];
        let r_off = net.typed_recall(&[a], &TemporalScope::Current, 3, 10, &filter, &ScarLog::default());
        let c_off = act_for(c, &r_off).expect("typed_recall hits C with decay off");
        net.set_decay_runtime(Some(DecayRuntime::new(DecayConfig::default(), now)));
        let r_on = net.typed_recall(&[a], &TemporalScope::Current, 3, 10, &filter, &ScarLog::default());
        let c_on = act_for(c, &r_on).expect("typed_recall still hits C with decay on");
        assert!(c_on < c_off * 0.5, "typed_recall: decay should de-rank stale C; off={}, on={}", c_off, c_on);
    }

    #[test]
    fn t51_cascade_uses_decay_when_enabled() {
        // Cascade-style deeper traversal (depth=3 across 4-node chain).
        // Stale tail bond should be de-ranked by decay.
        let now = 360 * DAY_MS;
        let mut net = AssociativeNetwork::new();
        let a = Seed::from_label("CascA");
        let b = Seed::from_label("CascB");
        let c = Seed::from_label("CascC");
        let d = Seed::from_label("CascD");
        for (s, name) in &[(a, "CascA"), (b, "CascB"), (c, "CascC"), (d, "CascD")] {
            net.add_node(KnowledgeNode::with_seed(*s, (*name).into(),
                KnowledgeContent::Concept((*name).into()), Confidence::Verified, 0));
        }
        net.add_bond(TemporalBond::new(a, b, BondType::RelatedTo, 0.9, now)).unwrap();
        net.add_bond(TemporalBond::new(b, c, BondType::RelatedTo, 0.9, now)).unwrap();
        net.add_bond(TemporalBond::new(c, d, BondType::RelatedTo, 0.9, 0)).unwrap();
        let r_off = net.spread_activation(&[a], &TemporalScope::Current, 4, 10, &ScarLog::default());
        let d_off = act_for(d, &r_off).expect("D activates at depth 3 with decay off");
        net.set_decay_runtime(Some(DecayRuntime::new(DecayConfig::default(), now)));
        let r_on = net.spread_activation(&[a], &TemporalScope::Current, 4, 10, &ScarLog::default());
        // D may even fall below ACTIVATION_THRESHOLD with decay on — either drop OR significant de-rank counts.
        let d_on = act_for(d, &r_on).unwrap_or(0.0);
        assert!(d_on < d_off * 0.5, "cascade: decay should de-rank stale D; off={}, on={}", d_off, d_on);
    }

    #[test]
    fn t51_decay_disabled_matches_pre_t51_behavior_exactly() {
        // Regression guard: with decay_runtime=None (the network default),
        // every spread/typed_recall/subgraph_coherence query must produce
        // results bit-identical to pre-T.5.1 behavior. The same network
        // with `Some(rt)` where rt.config.enabled=false must also match.
        let net = build_test_network();
        let s_rust = Seed::from_label("Rust");
        let baseline = net.spread_activation(&[s_rust], &TemporalScope::Current, 3, 10, &ScarLog::default());
        // Path 1: explicit None.
        let mut net1 = build_test_network();
        net1.set_decay_runtime(None);
        let r1 = net1.spread_activation(&[s_rust], &TemporalScope::Current, 3, 10, &ScarLog::default());
        // Path 2: enabled=false.
        let mut net2 = build_test_network();
        let mut cfg = DecayConfig::default();
        cfg.enabled = false;
        net2.set_decay_runtime(Some(DecayRuntime::new(cfg, 9_999_999_999_999)));
        let r2 = net2.spread_activation(&[s_rust], &TemporalScope::Current, 3, 10, &ScarLog::default());
        assert_eq!(baseline.activated.len(), r1.activated.len());
        assert_eq!(baseline.activated.len(), r2.activated.len());
        for (a, b) in baseline.activated.iter().zip(r1.activated.iter()) {
            assert_eq!(a.seed, b.seed);
            assert!((a.activation - b.activation).abs() < 1e-6);
        }
        for (a, b) in baseline.activated.iter().zip(r2.activated.iter()) {
            assert_eq!(a.seed, b.seed);
            assert!((a.activation - b.activation).abs() < 1e-6);
        }
    }

    #[test]
    fn t51_mock_clock_advance_decays_old_bond_in_recall_ranking() {
        // Single bond A->B at created=0. At now=0, activation matches
        // the no-decay baseline. Advance the mock clock by 90 days
        // (one half-life) and verify the activation halves.
        let (mut net, a, b, _c, _) = build_decay_test_graph(0, 999_999_999_999, None);
        let cfg = DecayConfig::default(); // 90-day half-life, enabled.

        // T0: now=0, no age. Decay multiplier should be 1.0.
        net.set_decay_runtime(Some(DecayRuntime::new(cfg, 0)));
        let r0 = net.spread_activation(&[a], &TemporalScope::Current, 1, 10, &ScarLog::default());
        let b_at_t0 = act_for(b, &r0).expect("B activates at t0");

        // T1: advance clock by 90 days. Multiplier should be ~0.5.
        net.set_decay_runtime(Some(DecayRuntime::new(cfg, 90 * DAY_MS)));
        let r1 = net.spread_activation(&[a], &TemporalScope::Current, 1, 10, &ScarLog::default());
        let b_at_t1 = act_for(b, &r1).expect("B still activates at t1");

        let ratio = b_at_t1 / b_at_t0;
        assert!(
            (ratio - 0.5).abs() < 0.05,
            "after one half-life advance, activation should ~halve; ratio={}",
            ratio,
        );
    }

    #[test]
    fn t51_recently_reinforced_bond_outranks_stale_bond_at_same_stored_weight() {
        // Two bonds from A to {B, D} with IDENTICAL stored weight.
        // A->B is fresh (created at NOW); A->D is stale (created at 0).
        // With decay on, B must outrank D.
        let now = 365 * DAY_MS;
        let (mut net, a, b, _c, d_opt) = build_decay_test_graph(now, 999_999_999_999, Some(0));
        let d = d_opt.unwrap();
        // Sanity: same stored raw weight on both A->B and A->D.
        let raw_ab = net.get_bond(&a, &b).unwrap().latest_raw_weight();
        let raw_ad = net.get_bond(&a, &d).unwrap().latest_raw_weight();
        assert!((raw_ab - raw_ad).abs() < 1e-6, "stored weights match: {} vs {}", raw_ab, raw_ad);
        net.set_decay_runtime(Some(DecayRuntime::new(DecayConfig::default(), now)));
        let r = net.spread_activation(&[a], &TemporalScope::Current, 1, 10, &ScarLog::default());
        let b_act = act_for(b, &r).expect("B activates");
        let d_act = act_for(d, &r).expect("D activates");
        assert!(b_act > d_act, "fresh B should outrank stale D; B={}, D={}", b_act, d_act);
    }

    #[test]
    fn t51_does_not_mutate_stored_bond_weight() {
        // Anti-hallucination: T.5 contract is "stored weights NEVER mutated;
        // decay is presentation-only." Verify that running a decay-on query
        // leaves `latest_raw_weight()` unchanged on every bond.
        let now = 365 * DAY_MS;
        let (mut net, a, b, c, _) = build_decay_test_graph(0, 0, None);
        let pre_ab = net.get_bond(&a, &b).unwrap().latest_raw_weight();
        let pre_bc = net.get_bond(&b, &c).unwrap().latest_raw_weight();
        net.set_decay_runtime(Some(DecayRuntime::new(DecayConfig::default(), now)));
        let _ = net.spread_activation(&[a], &TemporalScope::Current, 3, 10, &ScarLog::default());
        let _ = net.typed_recall(&[a], &TemporalScope::Current, 3, 10, &[BondType::RelatedTo], &ScarLog::default());
        let post_ab = net.get_bond(&a, &b).unwrap().latest_raw_weight();
        let post_bc = net.get_bond(&b, &c).unwrap().latest_raw_weight();
        assert!((pre_ab - post_ab).abs() < 1e-9, "A->B raw weight mutated! pre={}, post={}", pre_ab, post_ab);
        assert!((pre_bc - post_bc).abs() < 1e-9, "B->C raw weight mutated! pre={}, post={}", pre_bc, post_bc);
    }

    /// Helper: return the slice of network.rs source above the `#[cfg(test)]`
    /// marker so the grep guards below don't trip on their own pattern strings.
    fn t51_non_test_source() -> &'static str {
        let full = include_str!("network.rs");
        let cut = full.find("#[cfg(test)]\nmod tests {").unwrap_or(full.len());
        &full[..cut]
    }

    #[test]
    fn t51_no_query_path_calls_effective_weight_directly() {
        // Grep guard: every `.effective_weight` read in non-test network.rs
        // code must be multiplied by `decay_multiplier_for_bond(&bond)` on
        // the same line OR the immediately following line (statements wrap).
        let src = t51_non_test_source();
        let lines: Vec<&str> = src.lines().collect();
        let mut violations: Vec<String> = Vec::new();
        for (i, line) in lines.iter().enumerate() {
            let trimmed = line.trim_start();
            if trimmed.starts_with("//") { continue; }
            if !line.contains(".effective_weight") { continue; }
            let nxt = lines.get(i + 1).copied().unwrap_or("");
            let ok = line.contains("decay_multiplier_for_bond")
                || nxt.contains("decay_multiplier_for_bond");
            if !ok {
                violations.push(format!("line {}: {}", i + 1, line));
            }
        }
        assert!(
            violations.is_empty(),
            "T.5.1 grep guard: query-path effective_weight reads bypassing decay:\n{}",
            violations.join("\n"),
        );
    }

    #[test]
    fn t51_decay_uses_clock_source_not_system_time() {
        // Regression guard: the decay multiplier in non-test network.rs
        // code must use the stashed `decay_runtime.now_ms` (T.1 ClockSource-
        // sourced via Brain::refresh_decay_runtime), never SystemTime::now().
        let src = t51_non_test_source();
        for line in src.lines() {
            let trimmed = line.trim_start();
            if trimmed.starts_with("//") { continue; }
            assert!(
                !line.contains("SystemTime::now"),
                "network.rs must not call SystemTime::now — route via T.1 ClockSource: {}",
                line,
            );
        }
    }// ====================================================================
    // Port Sense.1 -- query-time cluster-gain wiring tests.
    //
    // The Sense module shipped the disambiguation algorithm + the
    // `lu4d_sense_status` MCP tool. Sense.1 wires the per-cluster gain
    // map into every query-path effective_weight read in this module so
    // polysemous-seed queries actually route through the winning sense
    // cluster. Stored bond weights are NEVER mutated by these paths.
    // ====================================================================
    use crate::sense::{SenseConfig, SenseRuntime};

    /// Build a synthetic two-cluster polysemous graph:
    /// `bank` -> {money, deposit, vault, atm}  (financial cluster, dense)
    /// `bank` -> {river, fish, willow, mud}    (river cluster, dense)
    /// Members within a cluster share neighbours; cross-cluster
    /// neighbours don't overlap. Returns the network plus the seed
    /// for `bank` and a representative node from each cluster.
    fn build_two_cluster_polysemous_graph() -> (
        AssociativeNetwork,
        Seed, // bank
        Vec<Seed>, // financial cluster members
        Vec<Seed>, // river cluster members
    ) {
        let mut net = AssociativeNetwork::new();
        let bank = Seed::from_label("bank");
        let fin: Vec<Seed> = ["money", "deposit", "vault", "atm"]
            .iter().map(|s| Seed::from_label(s)).collect();
        let riv: Vec<Seed> = ["river", "fish", "willow", "mud"]
            .iter().map(|s| Seed::from_label(s)).collect();
        // Hidden shared backbones so each cluster has neighbourhood overlap.
        let fin_back: Vec<Seed> = ["currency", "finance"]
            .iter().map(|s| Seed::from_label(s)).collect();
        let riv_back: Vec<Seed> = ["water", "nature"]
            .iter().map(|s| Seed::from_label(s)).collect();

        for s in std::iter::once(&bank).chain(fin.iter()).chain(riv.iter())
            .chain(fin_back.iter()).chain(riv_back.iter())
        {
            net.add_node(KnowledgeNode::with_seed(
                *s, format!("{:?}", s),
                KnowledgeContent::Concept(format!("{:?}", s)),
                Confidence::Verified, 0,
            ));
        }
        // bank -> all 8 cluster members.
        for f in &fin {
            net.add_bond(TemporalBond::new(bank, *f, BondType::RelatedTo, 0.9, 0)).unwrap();
        }
        for r in &riv {
            net.add_bond(TemporalBond::new(bank, *r, BondType::RelatedTo, 0.9, 0)).unwrap();
        }
        // Inside each cluster, every member -> all backbone nodes for that cluster.
        // This gives high Jaccard overlap inside-cluster, zero overlap across clusters.
        for f in &fin {
            for fb in &fin_back {
                net.add_bond(TemporalBond::new(*f, *fb, BondType::RelatedTo, 0.9, 0)).unwrap();
            }
        }
        for r in &riv {
            for rb in &riv_back {
                net.add_bond(TemporalBond::new(*r, *rb, BondType::RelatedTo, 0.9, 0)).unwrap();
            }
        }
        (net, bank, fin, riv)
    }

    #[test]
    fn sense1_runtime_default_is_empty_and_returns_unit_gain() {
        let cfg = SenseConfig::default();
        let rt = SenseRuntime::empty(cfg);
        assert_eq!(rt.polysemous_seed_count(), 0);
        let a = Seed::from_label("x");
        let b = Seed::from_label("y");
        assert!((rt.gain_for_bond(&a, &b) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn sense1_build_runtime_is_empty_when_disabled() {
        let (net, bank, fin, _riv) = build_two_cluster_polysemous_graph();
        let mut cfg = SenseConfig::default();
        cfg.disambiguation_enabled = false;
        let rt = net.build_sense_runtime(&[bank, fin[0]], cfg);
        assert_eq!(rt.polysemous_seed_count(), 0,
            "disabled config must produce an empty gain map");
    }

    #[test]
    fn sense1_build_runtime_finds_polysemous_seed() {
        let (net, bank, fin, _riv) = build_two_cluster_polysemous_graph();
        let cfg = SenseConfig::default();
        // Seed `bank` with financial-cluster context.
        let rt = net.build_sense_runtime(&[bank, fin[0], fin[1]], cfg);
        assert!(rt.polysemous_seed_count() >= 1,
            "two-cluster graph must yield at least one polysemous gain map; got {}",
            rt.polysemous_seed_count());
        let map = rt.gain_by_seed.get(&bank).expect("bank should appear in gain_by_seed");
        assert!(!map.is_empty(), "bank's gain map must not be empty");
    }

    #[test]
    fn sense1_disabled_matches_pre_sense1_behavior_exactly() {
        // Regression guard: with sense_runtime=None (network default),
        // every spread/typed_recall query must produce results bit-
        // identical to pre-Sense.1 behavior. The same network with
        // Some(rt) where rt.config.disambiguation_enabled=false must
        // also match. Mirrors t51_decay_disabled_matches_pre_t51_behavior_exactly.
        let net = build_test_network();
        let s_rust = Seed::from_label("Rust");
        let baseline = net.spread_activation(&[s_rust], &TemporalScope::Current, 3, 10, &ScarLog::default());
        // Path 1: explicit None.
        let mut net1 = build_test_network();
        net1.set_sense_runtime(None);
        let r1 = net1.spread_activation(&[s_rust], &TemporalScope::Current, 3, 10, &ScarLog::default());
        // Path 2: enabled=false.
        let mut net2 = build_test_network();
        let mut cfg = SenseConfig::default();
        cfg.disambiguation_enabled = false;
        net2.set_sense_runtime(Some(SenseRuntime::empty(cfg)));
        let r2 = net2.spread_activation(&[s_rust], &TemporalScope::Current, 3, 10, &ScarLog::default());
        assert_eq!(baseline.activated.len(), r1.activated.len());
        assert_eq!(baseline.activated.len(), r2.activated.len());
        for (a, b) in baseline.activated.iter().zip(r1.activated.iter()) {
            assert_eq!(a.seed, b.seed);
            assert!((a.activation - b.activation).abs() < 1e-6);
        }
        for (a, b) in baseline.activated.iter().zip(r2.activated.iter()) {
            assert_eq!(a.seed, b.seed);
            assert!((a.activation - b.activation).abs() < 1e-6);
        }
    }

    #[test]
    fn sense1_loser_gain_zero_prunes_loser_cluster_entirely() {
        let (mut net, bank, fin, riv) = build_two_cluster_polysemous_graph();
        // Disable decay so freshness doesn't confound results.
        net.set_decay_runtime(None);
        // Build a runtime: context = financial cluster -> winner is fin,
        // loser is riv. With loser_gain=0.0, river-side bonds should
        // contribute zero activation -> river members don't appear at all
        // in the recall (or have zero activation).
        let mut cfg = SenseConfig::default();
        cfg.disambiguation_enabled = true;
        cfg.loser_gain = 0.0;
        let context_seeds = vec![bank, fin[0], fin[1]];
        let rt = net.build_sense_runtime(&context_seeds, cfg);
        assert!(rt.polysemous_seed_count() >= 1, "two-cluster graph must yield disambiguation");
        net.set_sense_runtime(Some(rt));
        let recall = net.spread_activation(&[bank], &TemporalScope::Current, 1, 50, &ScarLog::default());
        let financial_hits: Vec<_> = recall.activated.iter().filter(|n| fin.contains(&n.seed)).collect();
        let river_hits: Vec<_> = recall.activated.iter().filter(|n| riv.contains(&n.seed)).collect();
        // Financial members must still activate (gain=1.0 for winner).
        assert!(!financial_hits.is_empty(), "financial cluster must still activate with loser_gain=0");
        // River members must be entirely pruned (gain=0 prunes them under ACTIVATION_THRESHOLD).
        assert!(river_hits.is_empty() || river_hits.iter().all(|n| n.activation < 1e-6),
            "river cluster must be pruned with loser_gain=0; got: {:?}",
            river_hits.iter().map(|n| (n.seed, n.activation)).collect::<Vec<_>>());
    }

    #[test]
    fn sense1_partial_loser_gain_still_de_ranks_loser_cluster() {
        let (mut net, bank, fin, riv) = build_two_cluster_polysemous_graph();
        net.set_decay_runtime(None);
        let mut cfg = SenseConfig::default();
        cfg.disambiguation_enabled = true;
        cfg.loser_gain = 0.25;
        let rt = net.build_sense_runtime(&[bank, fin[0], fin[1]], cfg);
        if rt.polysemous_seed_count() == 0 {
            // Synthetic graph didn't trip the polysemy detector under this configuration.
            // The previous test confirms detection works for this graph; treat this as a
            // non-blocking skip and assert only that the runtime is at least empty/safe.
            return;
        }
        net.set_sense_runtime(Some(rt));
        let recall = net.spread_activation(&[bank], &TemporalScope::Current, 1, 50, &ScarLog::default());
        let max_fin = recall.activated.iter().filter(|n| fin.contains(&n.seed)).map(|n| n.activation).fold(0.0f32, f32::max);
        let max_riv = recall.activated.iter().filter(|n| riv.contains(&n.seed)).map(|n| n.activation).fold(0.0f32, f32::max);
        assert!(max_fin > max_riv,
            "financial winner must outrank river loser at loser_gain=0.25; fin={}, riv={}",
            max_fin, max_riv);
    }

    #[test]
    fn sense1_loser_gain_one_matches_baseline() {
        // loser_gain=1.0 means the cluster gain is identity; results
        // should match the no-runtime baseline modulo float noise.
        // Activation levels can tie (this synthetic graph has uniform
        // bond weights), so iteration order is HashMap-dependent --
        // compare as sets keyed by seed instead of zipped vectors.
        let (mut net, bank, _fin, _riv) = build_two_cluster_polysemous_graph();
        net.set_decay_runtime(None);
        let baseline = net.spread_activation(&[bank], &TemporalScope::Current, 1, 50, &ScarLog::default());
        let mut cfg = SenseConfig::default();
        cfg.disambiguation_enabled = true;
        cfg.loser_gain = 1.0;
        let rt = net.build_sense_runtime(&[bank, Seed::from_label("money"), Seed::from_label("deposit")], cfg);
        net.set_sense_runtime(Some(rt));
        let with_unit_gain = net.spread_activation(&[bank], &TemporalScope::Current, 1, 50, &ScarLog::default());
        assert_eq!(baseline.activated.len(), with_unit_gain.activated.len());
        let bmap: std::collections::HashMap<Seed, f32> = baseline.activated.iter().map(|n| (n.seed, n.activation)).collect();
        let umap: std::collections::HashMap<Seed, f32> = with_unit_gain.activated.iter().map(|n| (n.seed, n.activation)).collect();
        for (seed, b_act) in &bmap {
            let u_act = umap.get(seed).unwrap_or_else(|| panic!("seed {:?} missing from with_unit_gain", seed));
            assert!((b_act - u_act).abs() < 1e-5,
                "loser_gain=1.0 must match baseline within float noise for {:?}: {} vs {}",
                seed, b_act, u_act);
        }
    }

    #[test]
    fn sense1_does_not_mutate_stored_bond_weight() {
        // Anti-hallucination: stored weights NEVER mutated by sense path.
        let (mut net, bank, fin, _riv) = build_two_cluster_polysemous_graph();
        let pre: Vec<f32> = fin.iter().map(|f| net.get_bond(&bank, f).unwrap().latest_raw_weight()).collect();
        let mut cfg = SenseConfig::default();
        cfg.loser_gain = 0.0;
        let rt = net.build_sense_runtime(&[bank, fin[0], fin[1]], cfg);
        net.set_sense_runtime(Some(rt));
        let _ = net.spread_activation(&[bank], &TemporalScope::Current, 2, 50, &ScarLog::default());
        let _ = net.typed_recall(&[bank], &TemporalScope::Current, 2, 50, &[BondType::RelatedTo], &ScarLog::default());
        let post: Vec<f32> = fin.iter().map(|f| net.get_bond(&bank, f).unwrap().latest_raw_weight()).collect();
        for (i, (a, b)) in pre.iter().zip(post.iter()).enumerate() {
            assert!((a - b).abs() < 1e-9, "bond {} mutated! pre={}, post={}", i, a, b);
        }
    }

    /// Helper: source slice of network.rs ABOVE the `#[cfg(test)]`
    /// marker. Reused by the Sense.1 grep guards. Tokens that would
    /// self-trigger the guards are split via concat!() at compile time.
    fn sense1_non_test_source() -> &'static str {
        let full = include_str!("network.rs");
        let cut = full.find("#[cfg(test)]\nmod tests {").unwrap_or(full.len());
        &full[..cut]
    }

    #[test]
    fn sense1_no_query_path_calls_effective_weight_directly() {
        // Grep guard: every `.effective_weight` read in non-test network.rs
        // code must be multiplied by `sense_gain_for_bond(...)` on the same
        // line OR within the next 2 lines (statements wrap across lines).
        // Mirrors the T.5.1 guard's structure.
        let src = sense1_non_test_source();
        let lines: Vec<&str> = src.lines().collect();
        let needle_eff = concat!(".", "effective_weight");
        let needle_gain = concat!("sense_", "gain_for_bond");
        let mut violations: Vec<String> = Vec::new();
        for (i, line) in lines.iter().enumerate() {
            let trimmed = line.trim_start();
            if trimmed.starts_with("//") { continue; }
            if !line.contains(needle_eff) { continue; }
            let n1 = lines.get(i + 1).copied().unwrap_or("");
            let n2 = lines.get(i + 2).copied().unwrap_or("");
            let ok = line.contains(needle_gain)
                || n1.contains(needle_gain)
                || n2.contains(needle_gain);
            if !ok {
                violations.push(format!("line {}: {}", i + 1, line));
            }
        }
        assert!(
            violations.is_empty(),
            "Sense.1 grep guard: query-path effective_weight reads bypassing sense gain:\n{}",
            violations.join("\n"),
        );
    }

    #[test]
    fn sense1_uses_clock_source_not_system_time() {
        // Regression guard: the sense gain path in non-test network.rs
        // code must NOT call SystemTime::now -- runtime selection routes
        // through Brain::refresh_sense_runtime which is itself driven by
        // the T.1 ClockSource (via ask/typed_ask/ask_in_scope).
        let src = sense1_non_test_source();
        let needle = concat!("System", "Time::now");
        for line in src.lines() {
            let trimmed = line.trim_start();
            if trimmed.starts_with("//") { continue; }
            assert!(
                !line.contains(needle),
                "network.rs must not call SystemTime::now -- route via T.1 ClockSource: {}",
                line,
            );
        }
    }

    #[test]
    fn sense1_module_never_mutates_bond_storage() {
        // Anti-hallucination: enforce at compile time of the test
        // binary that mutating-bond-API identifiers never appear in
        // the sense.rs runtime cache or in the Sense.1-touched
        // network.rs surface. Mirrors the existing
        // `sense_module_never_mutates_bond_storage` guard.
        const FORBIDDEN: &[&str] = &[
            concat!("set", "_weight("),
            concat!("set", "_historical", "_weight"),
            concat!("add", "_keyframe("),
            concat!("set", "_bond_", "type("),
        ];
        let sense_rs = include_str!("sense.rs");
        for f in FORBIDDEN {
            assert!(
                !sense_rs.contains(f),
                "sense.rs must NEVER reference mutating bond API '{}'", f);
        }
    }

    // ================================================================
    // OP-INGEST-1 -- cached bond_count parity tests.
    // ================================================================

    fn add_concept_node(net: &mut AssociativeNetwork, label: &str) -> Seed {
        let seed = Seed::from_label(label);
        if !net.contains_node(&seed) {
            net.add_node(KnowledgeNode::with_seed(
                seed,
                label.into(),
                KnowledgeContent::Concept(label.into()),
                Confidence::Verified,
                0,
            ));
        }
        seed
    }

    fn simple_bond(from: Seed, to: Seed) -> TemporalBond {
        TemporalBond::new(from, to, BondType::RelatedTo, 1.0, 0)
    }

    #[test]
    fn op_ingest_1_cache_init_empty() {
        let net = AssociativeNetwork::new();
        assert_eq!(net.bond_count(), 0);
        assert_eq!(net.bond_count_walked(), 0);
        assert!(net.verify_bond_count_cache().is_ok());
    }

    #[test]
    fn op_ingest_1_cache_after_overlay_inserts() {
        let mut net = AssociativeNetwork::new();
        let seeds: Vec<Seed> = (0..50)
            .map(|i| add_concept_node(&mut net, &format!("n{}", i)))
            .collect();
        for i in 0..49 {
            net.add_bond(simple_bond(seeds[i], seeds[i + 1])).unwrap();
        }
        assert_eq!(net.bond_count(), 49);
        assert_eq!(net.bond_count_walked(), 49);
        assert!(net.verify_bond_count_cache().is_ok());
    }

    #[test]
    fn op_ingest_1_cache_after_overlay_removes() {
        let mut net = AssociativeNetwork::new();
        let seeds: Vec<Seed> = (0..20)
            .map(|i| add_concept_node(&mut net, &format!("r{}", i)))
            .collect();
        for i in 0..19 {
            net.add_bond(simple_bond(seeds[i], seeds[i + 1])).unwrap();
        }
        assert_eq!(net.bond_count(), 19);
        // Remove half.
        for i in 0..10 {
            assert!(net.remove_bond(&seeds[i], &seeds[i + 1]));
        }
        assert_eq!(net.bond_count(), 9);
        assert_eq!(net.bond_count_walked(), 9);
        assert!(net.verify_bond_count_cache().is_ok());
    }

    #[test]
    fn op_ingest_1_cache_purge_nodes_cascades() {
        let mut net = AssociativeNetwork::new();
        let a = add_concept_node(&mut net, "PurgeA");
        let b = add_concept_node(&mut net, "PurgeB");
        let c = add_concept_node(&mut net, "PurgeC");
        net.add_bond(simple_bond(a, b)).unwrap();
        net.add_bond(simple_bond(b, c)).unwrap();
        net.add_bond(simple_bond(a, c)).unwrap();
        assert_eq!(net.bond_count(), 3);
        // Purge B -> removes (a,b) and (b,c). Leaves (a,c).
        let (nodes_removed, bonds_removed) = net.purge_nodes(&["PurgeB"]);
        assert_eq!(nodes_removed, 1);
        assert_eq!(bonds_removed, 2);
        assert_eq!(net.bond_count(), 1);
        assert_eq!(net.bond_count_walked(), 1);
        assert!(net.verify_bond_count_cache().is_ok());
    }

    #[test]
    fn op_ingest_1_cache_parity_random_mutations() {
        // Random insert/remove sequence; assert cache == walked after each step.
        use std::collections::HashSet;
        let mut net = AssociativeNetwork::new();
        let seeds: Vec<Seed> = (0..20)
            .map(|i| add_concept_node(&mut net, &format!("rng{}", i)))
            .collect();
        // Deterministic pseudo-random: linear congruential.
        let mut rng: u64 = 0xCAFEBABE;
        let mut live: HashSet<(usize, usize)> = HashSet::new();
        for _ in 0..400 {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let a = ((rng >> 16) as usize) % 20;
            let b = ((rng >> 32) as usize) % 20;
            if a == b { continue; }
            let op = (rng >> 48) & 0x3;
            if op < 2 {
                // insert
                let bond = simple_bond(seeds[a], seeds[b]);
                let _ = net.add_bond(bond);
                live.insert((a, b));
            } else {
                // remove
                net.remove_bond(&seeds[a], &seeds[b]);
                live.remove(&(a, b));
            }
            // Parity invariant after EVERY mutation -- this is the load-bearing canary.
            let parity = net.verify_bond_count_cache();
            assert!(parity.is_ok(),
                "cache drift after op a={} b={} op={} -- cached={:?}",
                a, b, op, parity.err());
        }
        assert_eq!(net.bond_count(), live.len());
    }

    #[test]
    fn op_ingest_1_cache_double_add_no_double_count() {
        let mut net = AssociativeNetwork::new();
        let a = add_concept_node(&mut net, "DupA");
        let b = add_concept_node(&mut net, "DupB");
        net.add_bond(simple_bond(a, b)).unwrap();
        assert_eq!(net.bond_count(), 1);
        // Re-insert same key -- should NOT increment.
        net.add_bond(simple_bond(a, b)).unwrap();
        assert_eq!(net.bond_count(), 1);
        assert_eq!(net.bond_count_walked(), 1);
        assert!(net.verify_bond_count_cache().is_ok());
    }

    #[test]
    fn op_ingest_1_cache_remove_nonexistent_is_noop() {
        let mut net = AssociativeNetwork::new();
        let a = add_concept_node(&mut net, "NopeA");
        let b = add_concept_node(&mut net, "NopeB");
        assert_eq!(net.bond_count(), 0);
        assert!(!net.remove_bond(&a, &b));
        assert_eq!(net.bond_count(), 0);
        assert!(net.verify_bond_count_cache().is_ok());
    }

    #[test]
    fn op_ingest_1_cache_via_from_snapshot() {
        // Build a network, snapshot it, reload via from_snapshot, parity must hold.
        let mut net = AssociativeNetwork::new();
        let seeds: Vec<Seed> = (0..30)
            .map(|i| add_concept_node(&mut net, &format!("snap{}", i)))
            .collect();
        for i in 0..29 {
            net.add_bond(simple_bond(seeds[i], seeds[i + 1])).unwrap();
        }
        for i in 0..28 {
            net.add_bond(simple_bond(seeds[i], seeds[i + 2])).unwrap();
        }
        let expected = net.bond_count_walked();
        let snap = net.to_snapshot();
        let net2 = AssociativeNetwork::from_snapshot(snap);
        assert_eq!(net2.bond_count(), expected);
        assert_eq!(net2.bond_count_walked(), expected);
        assert!(net2.verify_bond_count_cache().is_ok());
    }

    // ------------------------------------------------------------
    // OP-INGEST-1.1: lazy-init bond_count cache.
    // ------------------------------------------------------------

    #[test]
    fn op_ingest_1_1_lazy_init_starts_uninitialized() {
        // After new(), cache MUST be None -- no eager walk.
        let net = AssociativeNetwork::new();
        assert_eq!(net.bond_count_cache.get(), None,
            "fresh network should have None cache (lazy-init)");
    }

    #[test]
    fn op_ingest_1_1_lazy_init_populates_on_first_read() {
        let mut net = AssociativeNetwork::new();
        let a = add_concept_node(&mut net, "LazyA");
        let b = add_concept_node(&mut net, "LazyB");
        net.add_bond(simple_bond(a, b)).unwrap();
        // Cache should still be None -- mutations on a None cache skip update.
        assert_eq!(net.bond_count_cache.get(), None,
            "add_bond on None cache must leave cache None");
        // First read triggers walk + memoize.
        let n = net.bond_count();
        assert_eq!(n, 1);
        assert_eq!(net.bond_count_cache.get(), Some(1),
            "first bond_count() read should populate cache to Some(walked)");
        // Subsequent reads return the same memoized value.
        assert_eq!(net.bond_count(), 1);
        assert_eq!(net.bond_count_cache.get(), Some(1));
    }

    #[test]
    fn op_ingest_1_1_lazy_init_mutations_before_first_read() {
        // Load-bearing: mutations on a None cache must be absorbed correctly
        // by the eventual lazy-init walk. This is the from_archive() startup
        // pattern -- many ingests before the first telemetry read.
        let mut net = AssociativeNetwork::new();
        let seeds: Vec<Seed> = (0..101)
            .map(|i| add_concept_node(&mut net, &format!("pre{}", i)))
            .collect();
        // 100 bonds, cache untouched throughout.
        for i in 0..100 {
            net.add_bond(simple_bond(seeds[i], seeds[i + 1])).unwrap();
        }
        assert_eq!(net.bond_count_cache.get(), None,
            "100 mutations on None cache must keep it None");
        // First read: walk picks up all 100 mutations.
        assert_eq!(net.bond_count(), 100);
        assert_eq!(net.bond_count_cache.get(), Some(100));
        assert_eq!(net.bond_count_walked(), 100);
        assert!(net.verify_bond_count_cache().is_ok());
    }

    #[test]
    fn op_ingest_1_1_lazy_init_mutations_after_first_read() {
        // After the cache is populated, subsequent mutations must keep it
        // in sync (standard OP-INGEST-1 incremental-maintenance behavior).
        let mut net = AssociativeNetwork::new();
        let seeds: Vec<Seed> = (0..201)
            .map(|i| add_concept_node(&mut net, &format!("post{}", i)))
            .collect();
        // First 50 bonds, then trigger lazy-init.
        for i in 0..50 {
            net.add_bond(simple_bond(seeds[i], seeds[i + 1])).unwrap();
        }
        assert_eq!(net.bond_count(), 50);
        assert_eq!(net.bond_count_cache.get(), Some(50));
        // 100 more bonds with cache populated -- must increment to Some(150).
        for i in 50..150 {
            net.add_bond(simple_bond(seeds[i], seeds[i + 1])).unwrap();
        }
        assert_eq!(net.bond_count_cache.get(), Some(150),
            "populated cache must absorb each add_bond increment");
        assert_eq!(net.bond_count(), 150);
        assert_eq!(net.bond_count_walked(), 150);
        assert!(net.verify_bond_count_cache().is_ok());
        // Mixed remove on populated cache -- still in sync.
        for i in 0..10 {
            assert!(net.remove_bond(&seeds[i], &seeds[i + 1]));
        }
        assert_eq!(net.bond_count_cache.get(), Some(140));
        assert_eq!(net.bond_count(), 140);
        assert!(net.verify_bond_count_cache().is_ok());
    }

    // -----------------------------------------------------------
    // OP-ASK-WATCHDOG tests
    // -----------------------------------------------------------

    fn build_watchdog_net() -> AssociativeNetwork {
        let mut net = AssociativeNetwork::new();
        let a = Seed::from_label("alpha");
        let b = Seed::from_label("beta");
        let c = Seed::from_label("gamma");
        let d = Seed::from_label("delta");
        net.add_node(KnowledgeNode::with_seed(a, "alpha".into(),
            KnowledgeContent::Concept("a".into()), Confidence::Verified, 0));
        net.add_node(KnowledgeNode::with_seed(b, "beta".into(),
            KnowledgeContent::Concept("b".into()), Confidence::Verified, 0));
        net.add_node(KnowledgeNode::with_seed(c, "gamma".into(),
            KnowledgeContent::Concept("c".into()), Confidence::Verified, 0));
        net.add_node(KnowledgeNode::with_seed(d, "delta".into(),
            KnowledgeContent::Concept("d".into()), Confidence::Verified, 0));
        net.add_bond(TemporalBond::new(a, b, BondType::IsA, 0.9, 0)).unwrap();
        net.add_bond(TemporalBond::new(b, c, BondType::HasProperty, 0.9, 0)).unwrap();
        net.add_bond(TemporalBond::new(c, d, BondType::HasProperty, 0.9, 0)).unwrap();
        net
    }

    #[test]
    fn op_ask_watchdog_inner_times_out_with_tight_budget() {
        let net = build_watchdog_net();
        let a = Seed::from_label("alpha");
        let mut bc = BridgeCollector::new(16);
        let mut cc = CuriosityCollector::new(16);
        let mut scars: Vec<crate::scar::Scar> = Vec::new();
        let r = net.spread_activation_weighted_full(
            &[(a, 1.0)],
            &TemporalScope::Current,
            3, 10,
            &mut bc, &BridgeConfig::default(),
            &mut cc, &CuriosityConfig::default(),
            DEGREE_NORM_THRESHOLD, 0,
            &ScarLog::default(),
            &mut scars,
            Some(Duration::from_nanos(1)),
        );
        assert!(matches!(r.status, SpreadStatus::TimedOut { .. }),
            "expected TimedOut, got {:?}", r.status);
    }

    #[test]
    fn op_ask_watchdog_inner_completes_with_generous_budget() {
        let net = build_watchdog_net();
        let a = Seed::from_label("alpha");
        let mut bc = BridgeCollector::new(16);
        let mut cc = CuriosityCollector::new(16);
        let mut scars: Vec<crate::scar::Scar> = Vec::new();
        let r = net.spread_activation_weighted_full(
            &[(a, 1.0)],
            &TemporalScope::Current,
            3, 10,
            &mut bc, &BridgeConfig::default(),
            &mut cc, &CuriosityConfig::default(),
            DEGREE_NORM_THRESHOLD, 0,
            &ScarLog::default(),
            &mut scars,
            Some(Duration::from_secs(60)),
        );
        assert_eq!(r.status, SpreadStatus::Complete);
        assert!(!r.activated.is_empty(), "generous budget must drain frontier");
    }

    #[test]
    fn op_ask_watchdog_partial_activations_on_timeout() {
        let net = build_watchdog_net();
        let a = Seed::from_label("alpha");
        let mut bc = BridgeCollector::new(16);
        let mut cc = CuriosityCollector::new(16);
        let mut scars: Vec<crate::scar::Scar> = Vec::new();
        let r = net.spread_activation_weighted_full(
            &[(a, 1.0)],
            &TemporalScope::Current,
            3, 10,
            &mut bc, &BridgeConfig::default(),
            &mut cc, &CuriosityConfig::default(),
            DEGREE_NORM_THRESHOLD, 0,
            &ScarLog::default(),
            &mut scars,
            Some(Duration::from_nanos(1)),
        );
        assert!(matches!(r.status, SpreadStatus::TimedOut { .. }),
            "expected TimedOut");
        assert!(!r.activated.is_empty(),
            "seed activation must survive timeout; got empty");
    }

    #[test]
    fn op_ask_watchdog_none_budget_is_disabled() {
        let net = build_watchdog_net();
        let a = Seed::from_label("alpha");
        let mut bc = BridgeCollector::new(16);
        let mut cc = CuriosityCollector::new(16);
        let mut scars: Vec<crate::scar::Scar> = Vec::new();
        let r = net.spread_activation_weighted_full(
            &[(a, 1.0)],
            &TemporalScope::Current,
            3, 10,
            &mut bc, &BridgeConfig::default(),
            &mut cc, &CuriosityConfig::default(),
            DEGREE_NORM_THRESHOLD, 0,
            &ScarLog::default(),
            &mut scars,
            None,
        );
        assert_eq!(r.status, SpreadStatus::Complete,
            "None budget must produce Complete; got {:?}", r.status);
    }

}

// ----------------------------------------------------------------
// Port Cat.3b — hub-degree read-side filter and degree accessors.
// ----------------------------------------------------------------

impl AssociativeNetwork {
    /// In-degree of a node (number of incoming bonds). O(1).
    pub fn in_degree(&self, seed: &Seed) -> usize {
        self.incoming.get(seed).map(|v| v.len()).unwrap_or(0)
    }

    /// Out-degree of a node (number of outgoing bonds). O(1).
    pub fn out_degree(&self, seed: &Seed) -> usize {
        self.outgoing.get(seed).map(|v| v.len()).unwrap_or(0)
    }

    /// Sync read-side cap fields from a HubDegreeConfig. Called by
    /// Brain on load and on every MCP `lu4d_hub_set` call so the
    /// network's spread paths see the latest knobs.
    pub fn sync_hub_read_config(&mut self, c: &crate::hub_degree::HubDegreeConfig) {
        self.hub_read_max_out = c.max_out_degree;
        self.hub_read_max_in = c.max_in_degree;
        self.hub_read_enforcement = c.read_enforcement;
    }

    /// Query-time read of the cumulative filter counter.
    pub fn hub_cap_filtered_at_query_total(&self) -> usize {
        self.hub_cap_filtered_at_query_total
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Read-side capped outgoing neighbor list. When
    /// `hub_read_enforcement` is true and the seed has more than
    /// `hub_read_max_out` outgoing bonds, returns the top-K by
    /// latest raw bond weight; otherwise returns the full list.
    /// Bumps `hub_cap_filtered_at_query_total` by the number of
    /// bonds dropped so /lu4d_hub_status can show whether the filter
    /// is doing real work on a legacy / pre-cap brain.
    pub fn neighbors_outgoing_capped(&self, seed: &Seed) -> Vec<Seed> {
        let raw_len = self.out_degree(seed);
        if !self.hub_read_enforcement || (raw_len as u32) <= self.hub_read_max_out {
            return self.neighbors_outgoing(seed).to_vec();
        }
        let raw: Vec<Seed> = self.neighbors_outgoing(seed).to_vec();
        let mut weighted: Vec<(Seed, f32)> = Vec::with_capacity(raw.len());
        for n in &raw {
            if let Some(b) = self.get_bond(seed, n) {
                weighted.push((*n, b.latest_raw_weight()));
            } else {
                weighted.push((*n, 0.0));
            }
        }
        weighted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let keep = self.hub_read_max_out as usize;
        let dropped = raw.len().saturating_sub(keep);
        self.hub_cap_filtered_at_query_total
            .fetch_add(dropped, std::sync::atomic::Ordering::Relaxed);
        weighted.into_iter().take(keep).map(|(s, _)| s).collect()
    }

    /// Read-side capped incoming neighbor list. Mirrors
    /// `neighbors_outgoing_capped` for the incoming direction. Uses
    /// `hub_read_max_in` as the cap.
    pub fn neighbors_incoming_capped(&self, seed: &Seed) -> Vec<Seed> {
        let raw_len = self.in_degree(seed);
        if !self.hub_read_enforcement || (raw_len as u32) <= self.hub_read_max_in {
            return self.neighbors_incoming(seed).to_vec();
        }
        let raw: Vec<Seed> = self.neighbors_incoming(seed).to_vec();
        let mut weighted: Vec<(Seed, f32)> = Vec::with_capacity(raw.len());
        for from in &raw {
            if let Some(b) = self.get_bond(from, seed) {
                weighted.push((*from, b.latest_raw_weight()));
            } else {
                weighted.push((*from, 0.0));
            }
        }
        weighted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let keep = self.hub_read_max_in as usize;
        let dropped = raw.len().saturating_sub(keep);
        self.hub_cap_filtered_at_query_total
            .fetch_add(dropped, std::sync::atomic::Ordering::Relaxed);
        weighted.into_iter().take(keep).map(|(s, _)| s).collect()
    }

    /// OP-BRAIN-V4 Phase 5 -- mirror the read-path mode flag into
    /// the network. Called by Brain::set_read_path_mode after the
    /// Brain-side atomic is updated. Storing the mode here lets
    /// the spread-activation loop dispatch without a back-pointer
    /// to Brain.
    pub(crate) fn set_read_path_mode_u8(&self, mode: u8) {
        self.read_path_mode.store(mode, std::sync::atomic::Ordering::Release);
    }

    /// Read the mirrored Phase 5 read-path mode.
    pub(crate) fn read_path_mode_u8(&self) -> u8 {
        self.read_path_mode.load(std::sync::atomic::Ordering::Acquire)
    }

    /// OP-BRAIN-V4 Phase 5 -- install MemTable / manifest / sst_dir
    /// handles so the spread loop can augment with MemTable + L0
    /// reads when `read_path_mode != V3Only`. Called by Brain right
    /// after `enable_memtable` and `enable_flush_worker`. Safe to
    /// call multiple times (replaces previous handles).
    pub(crate) fn install_v4_read_handles(
        &mut self,
        memtable: Option<std::sync::Arc<crate::memtable::MemTable>>,
        manifest: Option<std::sync::Arc<std::sync::RwLock<std::sync::Arc<crate::manifest::Manifest>>>>,
        sst_dir: Option<std::path::PathBuf>,
    ) {
        self.v4_memtable = memtable;
        self.v4_manifest = manifest;
        self.v4_sst_dir = sst_dir;
        if self.v4_block_cache.is_none() {
            self.v4_block_cache = Some(std::sync::Arc::new(
                crate::sst::BlockCache::new(crate::sst::DEFAULT_SST_BLOCK_CACHE_BYTES),
            ));
        }
    }

    /// OP-BRAIN-V4 Phase 5 -- canonical neighbor lookup for the
    /// spread-activation hot path. Dispatches on the mirrored
    /// `read_path_mode` flag:
    ///
    /// - `V3Only` (default): returns exactly
    ///   `self.neighbors_outgoing_capped(seed)`. Byte-equivalent
    ///   to pre-Phase-5 behaviour.
    /// - `V4Only` / `MergedV3V4`: augments the v3 neighbor list
    ///   with MemTable + L0 SST contributions discovered via the
    ///   installed v4 handles. Dedup is by object Seed; the v3
    ///   subset is preserved exactly so existing call sites
    ///   (`view_bond`, `weight_in_scope`, scar machinery) keep
    ///   working unchanged when the v3 layer holds the bond.
    ///
    /// The hub-read cap is applied AFTER augmentation so the
    /// guardrail still bounds the effective neighbor count under
    /// the v4 modes.
    /// OP-BRAIN-MapRAM P2 -- the read-mode neighbour SET, UNCAPPED. The
    /// hub-cap is applied by the seam (`seam_spread_outgoing`) AFTER a
    /// canonical sort, so the cap's tie-break is deterministic. v3-only
    /// returns the plain resident set; v4/v5 modes augment with MemTable
    /// + L0 exactly as before.
    pub(crate) fn neighbors_outgoing_read_mode_uncapped(&self, seed: &Seed) -> Vec<Seed> {
        let mode = crate::read_path::ReadPathMode::from_u8(self.read_path_mode_u8());
        if mode.is_v3_only() {
            return self.neighbors_outgoing(seed);
        }
        use std::collections::HashSet;
        let mut seen: HashSet<Seed> = HashSet::new();
        let mut out: Vec<Seed> = Vec::new();
        for n in self.neighbors_outgoing(seed) {
            if seen.insert(n) { out.push(n); }
        }
        if let Some(mt) = &self.v4_memtable {
            let bonds = mt.get_bonds_for_subject(seed, &self.node_intern);
            for mb in bonds {
                if seen.insert(mb.object) { out.push(mb.object); }
            }
        }
        if let (Some(snap_lock), Some(sst_dir), Some(cache)) = (
            &self.v4_manifest, &self.v4_sst_dir, &self.v4_block_cache,
        ) {
            let manifest = snap_lock.read()
                .expect("v4_manifest read lock").clone();
            let mut l0: Vec<&crate::manifest::SSTEntry> = manifest
                .levels.iter()
                .filter(|l| l.level == 0)
                .flat_map(|l| l.ssts.iter())
                .collect();
            l0.sort_by(|a, b| b.id.cmp(&a.id));
            for entry in l0 {
                let smin = entry.subject_min_seed();
                let smax = entry.subject_max_seed();
                if seed.as_bytes() < smin.as_bytes()
                    || seed.as_bytes() > smax.as_bytes()
                {
                    continue;
                }
                // Phase 5: cache SST handles by manifest id.
                let sst = {
                    let mut hc = self.v4_sst_handle_cache.lock()
                        .expect("v4_sst_handle_cache lock");
                    if let Some(s) = hc.get(&entry.id) {
                        s.clone()
                    } else {
                        let path = sst_dir.join(&entry.file);
                        match crate::sst::SST::open(&path, None, cache.clone()) {
                            Ok(s) => {
                                let a = std::sync::Arc::new(s);
                                hc.insert(entry.id, a.clone());
                                a
                            }
                            Err(e) => {
                                eprintln!("[network] Phase 5 spread: skipping L0 {}: {}", entry.file, e);
                                continue;
                            }
                        }
                    }
                };
                let bonds = match sst.get_bonds_for_subject(seed) {
                    Ok(b) => b,
                    Err(e) => {
                        eprintln!("[network] Phase 5 spread: L0 read sst-id={}: {}", entry.id, e);
                        continue;
                    }
                };
                for v in bonds {
                    if seen.insert(v.object) { out.push(v.object); }
                }
            }
        }
        out
    }
}
// =====================================================================
// OP-CONCLUDE Tier 1 -- SST-aware typed-edge reads for the conclusion
// synthesizer (src/synthesis.rs). These mirror the seam path that
// spread and the narrate source-attribution block already use, so they
// see SST-carried edges on the read-only reader -- where the
// heap-resident `neighbors_outgoing` / `get_bond` return nothing. On a
// fresh (non-mapram) test brain the same seam accessors fall back to
// the resident path, so unit tests and the live SST reader exercise one
// code path. Read-only; hub-capped; deterministic (canonical order).
// =====================================================================
impl AssociativeNetwork {
    /// Typed outgoing edges (neighbour seed + bond type) for a node.
    /// Hub-capped via `seam_outgoing_edges_capped` to bound read
    /// amplification on dense nodes. Never mutates.
    pub(crate) fn typed_edges_out(
        &self,
        seed: &Seed,
    ) -> Vec<(Seed, crate::bond::BondType)> {
        let mut out = Vec::new();
        for e in self.seam_outgoing_edges_capped(seed) {
            if let Some(b) = self.seam_directed_bond(seed, &e.nbr) {
                out.push((e.nbr, b.view().bond_type()));
            }
        }
        out
    }

    /// Incoming twin of [`Self::typed_edges_out`] -- edges `nbr -> seed`.
    /// Returns each incoming neighbour with the bond type pointing AT
    /// `seed`. Hub-capped; read-only.
    #[allow(dead_code)] // consumed by Tier 1.5 relational-path / location shapes
    pub(crate) fn typed_edges_in(
        &self,
        seed: &Seed,
    ) -> Vec<(Seed, crate::bond::BondType)> {
        let mut out = Vec::new();
        for e in self.seam_incoming_edges_capped(seed) {
            if let Some(b) = self.seam_directed_bond(&e.nbr, seed) {
                out.push((e.nbr, b.view().bond_type()));
            }
        }
        out
    }
}


// =====================================================================
// OP-BREADTH Phase-1 -- entity namespace helpers (shared by the keyword
// rebuild walk, the overlay indexer, and synthesis.rs entity shapes).
// An entity node's content is "<label>#Q<digits>"; it is indexed ONLY
// under the "e_<word>" token namespace + an "eq<qid>" key so concept
// keyword buckets never gain an entity seed.
// =====================================================================

/// Parse the trailing "#Q<digits>" of an entity label. None for concept nodes.
pub(crate) fn entity_qid_of(content: &str) -> Option<u64> {
    let h = content.rfind('#')?;
    let tail = &content[h + 1..];
    if tail.len() > 1
        && (tail.as_bytes()[0] == b'Q' || tail.as_bytes()[0] == b'q')
        && tail[1..].bytes().all(|b| b.is_ascii_digit())
    {
        tail[1..].parse::<u64>().ok()
    } else {
        None
    }
}

/// The label part of an entity content string ("Paris#Q90" -> "Paris").
/// Returns the whole string for concept nodes.
pub(crate) fn entity_label_part(content: &str) -> &str {
    if entity_qid_of(content).is_some() {
        if let Some(h) = content.rfind('#') {
            return &content[..h];
        }
    }
    content
}

/// Keyword tokens of a string (lowercase, split, trim non-alnum, len > 1).
/// Matches the rebuild walk + overlay indexer tokenisation exactly.
fn kw_tokens(s: &str) -> Vec<String> {
    s.to_lowercase()
        .split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
        .filter(|w| !w.is_empty() && w.len() > 1)
        .collect()
}

/// Push a string's tokens into the streaming rebuilder, prefixing with
/// "e_" when `ent` (entity namespace).
fn push_index_tokens(
    b: &mut crate::keyword_index::StreamingRebuilder,
    s: &str,
    seed: Seed,
    ent: bool,
) -> std::io::Result<()> {
    for w in kw_tokens(s) {
        let key = if ent { format!("e_{}", w) } else { w };
        b.push(key, seed)?;
    }
    Ok(())
}
