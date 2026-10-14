//! Lu Left-Brain operational/episodic memory layer (Port Q.3 — #438 / #587).
//!
//! Per-session conversational state: turn history, weighted pronoun
//! anchors with per-turn decay, topic-shift detection, and ephemeral
//! session beliefs. This is a separate concern from the keyframe-
//! isolation `SessionManager` in `session.rs` (which prevents Hebbian
//! updates from polluting baseline knowledge across sessions).
//!
//! ## Hard rule: session beliefs are ephemeral
//!
//! `SessionBelief` entries captured during a conversation are
//! **never** automatically promoted to the durable knowledge graph.
//! Promotion happens only on an explicit user signal (a direct
//! `lu4d_teach` call, or a future "remember that" handler). Accidental
//! promotion via repetition heuristics across sessions is **out of
//! scope** for Q.3.x and must remain so unless the design is
//! revisited — accidental promotion of a casual conversational
//! mention would corrupt the baseline graph in ways that are very
//! hard to undo.
//!
//! ## Anchor lifecycle
//!
//! Each turn's top activations are promoted to `Anchor` entries with
//! initial weight `1.0`. At the start of every subsequent turn, all
//! anchors are decayed by `ANCHOR_DECAY_FACTOR` (0.5) — so a strong
//! anchor 2 turns ago (weight 0.25) beats a weak secondary anchor 1
//! turn ago (weight 0.4 vs 0.5 from primary, but weak secondaries
//! seed at 0.5*x for some x<1). Pronoun resolution iterates anchors
//! by weight descending and picks the first whose weight clears
//! `PRONOUN_RESOLUTION_THRESHOLD` (0.3). When all anchors are below
//! threshold the resolver returns `None` so the handler can ask for
//! clarification rather than substitute a stale anchor.
//!
//! ## Raw query preservation
//!
//! `Turn.raw_query` stores the verbatim user string with no
//! normalization. Backreference queries ("what was my first
//! question?") return that text character-for-character. The
//! rewritten / pronoun-substituted form lives separately in
//! `Turn.effective_query`.

use crate::reasoning::{EvidenceKind, ReasoningPattern, ReasoningTrace};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

// =====================================================================
// Tunables
// =====================================================================

/// Anchor decay multiplier applied to all anchor weights at the start
/// of every new turn. 0.5 means each turn halves the weight, so
/// anchors fall under the resolution threshold (0.3) within ~2 turns
/// unless re-promoted by a fresh top activation.
pub const ANCHOR_DECAY_FACTOR: f32 = 0.5;

/// Pronoun resolution refuses to substitute a stale anchor whose
/// decayed weight has fallen at or below this floor. The handler
/// surfaces a clarifying response instead of guessing.
pub const PRONOUN_RESOLUTION_THRESHOLD: f32 = 0.3;

/// Maximum number of turns retained per session. Older turns are
/// evicted FIFO. Bounds memory on long-running conversations.
pub const MAX_TURN_HISTORY: usize = 32;

/// Topic-shift detection: when the new turn's top-activation set
/// shares less than this fraction with the prior `current_topic`
/// neighborhood, the previous topic is evicted (anchor weights
/// stay decayed; only the dominant `current_topic` slot resets).
pub const TOPIC_SHIFT_OVERLAP_THRESHOLD: f32 = 0.2;

/// Default idle timeout for `SessionMemoryRegistry`. Sessions whose
/// `last_active_ms` is older than this on the next access are
/// evicted lazily. Default 30 minutes.
pub const DEFAULT_IDLE_TIMEOUT_MS: i64 = 30 * 60 * 1000;

/// Maximum anchors retained per session. Beyond this, lowest-weight
/// anchors are dropped on each turn after promotion.
pub const MAX_ANCHORS: usize = 16;

/// Number of secondary anchors (after the primary top-activation) to
/// promote each turn. Each gets initial weight `0.5 * 0.9^k` so the
/// primary always dominates ranking.
pub const SECONDARY_ANCHOR_COUNT: usize = 4;

// =====================================================================
// Anchor + TurnTraceSummary + Turn + SessionBelief
// =====================================================================

/// A referenceable entity surfaced by a recent turn's top activations.
/// `weight` decays each turn (multiplied by `ANCHOR_DECAY_FACTOR`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Anchor {
    pub label: String,
    pub weight: f32,
    pub turns_since_introduced: u32,
}

impl Anchor {
    pub fn new(label: impl Into<String>, weight: f32) -> Self {
        Anchor {
            label: label.into(),
            weight,
            turns_since_introduced: 0,
        }
    }
}

/// Compact (~100-500 byte) summary of a turn's reasoning chain.
/// Replaces the full `ReasoningTrace` so per-session memory stays
/// bounded even on long conversations. Most fields are `Option`
/// because the lightweight `tool_ask` path doesn't compute a full
/// reasoning chain — only `top_anchors` will be populated for
/// recall-only turns.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct TurnTraceSummary {
    /// Node labels along the picked reasoning chain, in order. Only
    /// set when a `Structured` or articulated `Associative` chain
    /// was produced.
    pub primary_chain: Option<Vec<String>>,
    pub pattern: Option<ReasoningPattern>,
    pub confidence: Option<f32>,
    pub evidence_kind: Option<EvidenceKind>,
    /// Top 3-5 (label, weight) anchors at the time of this turn.
    /// Always populated even on recall-only turns.
    pub top_anchors: Vec<(String, f32)>,
}

impl TurnTraceSummary {
    /// Empty summary; `top_anchors` populated separately by the caller.
    pub fn empty() -> Self {
        Self::default()
    }

    /// Build a compact summary from a full `ReasoningTrace`. Only
    /// retains the chain's node labels and the metadata fields.
    pub fn from_trace(trace: &ReasoningTrace) -> Self {
        let primary_chain = if trace.no_chain || trace.steps.is_empty() {
            None
        } else {
            let mut labels: Vec<String> = Vec::with_capacity(trace.steps.len() + 1);
            if let Some(first) = trace.steps.first() {
                labels.push(first.from_content.clone());
            }
            for s in &trace.steps {
                labels.push(s.to_content.clone());
            }
            Some(labels)
        };
        let pattern = if !trace.steps.is_empty() {
            let bonds: Vec<_> = trace.steps.iter().map(|s| s.bond_type).collect();
            Some(crate::reasoning::classify_pattern(&bonds))
        } else {
            None
        };
        TurnTraceSummary {
            primary_chain,
            pattern,
            confidence: Some(trace.confidence),
            evidence_kind: Some(trace.evidence_kind),
            top_anchors: Vec::new(),
        }
    }
}

/// One conversational turn. `raw_query` is the verbatim user string
/// (no normalization, no tokenization) so backreference queries
/// return character-for-character what the user originally asked.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Turn {
    pub timestamp_ms: i64,
    /// Verbatim user query.
    pub raw_query: String,
    /// Query as actually executed against the brain. Equal to
    /// `raw_query` when no rewrite happened; differs when a pronoun
    /// was substituted or an elaboration was expanded.
    pub effective_query: String,
    pub response_text: String,
    pub top_activations: Vec<String>,
    pub trace_summary: TurnTraceSummary,
}

/// An ephemeral (subject, relation, object) the user asserted in
/// conversation. NEVER auto-promoted to the durable graph (see module
/// docs).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SessionBelief {
    pub subject: String,
    pub relation: String,
    pub object: String,
    pub source_turn_index: usize,
}

// =====================================================================
// Backreference / elaboration / continuation classifiers
// =====================================================================

/// Recognized backreference query shapes. Backreference queries are
/// answered directly from `turn_history` without invoking recall.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackrefKind {
    /// "what did I just ask?", "what was my last question?",
    /// "what was the previous question?".
    LastQuestion,
    /// "what was my first question?".
    FirstQuestion,
    /// "what was your last answer?", "what was the previous answer?".
    LastAnswer,
}

/// Classify a query as a backreference, if any.
pub fn classify_backreference(query: &str) -> Option<BackrefKind> {
    let q = normalize_for_classify(query);
    // Explicit first-question pattern (must precede last-question because
    // "what was my first question" also contains "question").
    if q.contains("first question") || q.contains("first thing i asked") || q.contains("first thing i said") {
        return Some(BackrefKind::FirstQuestion);
    }
    if q.contains("last answer")
        || q.contains("previous answer")
        || q.contains("your last response")
        || q.contains("the previous response")
    {
        return Some(BackrefKind::LastAnswer);
    }
    if q.contains("just ask")
        || q.contains("did i just ask")
        || q.contains("last question")
        || q.contains("previous question")
        || q.contains("the last thing i asked")
        || q.contains("what was the question")
    {
        return Some(BackrefKind::LastQuestion);
    }
    None
}

/// OP-QA-3: an EXHAUSTIVE knowledge-enumeration ask -- "tell me everything
/// you know", "tell me all of the code you know", "list all the X you know".
/// Lu's knowledge is a web of concepts, not a list it can read back, so these
/// cannot be satisfied from the graph and previously produced a stitched
/// "several strands" word-salad. The honest handler declines plainly. Kept
/// distinct from the bare "what do you know?" MetaSelf report (which is
/// answered from the brain-level counters) by REQUIRING an all/everything
/// quantifier bound to "you know".
pub fn is_knowledge_enumeration(query: &str) -> bool {
    let q = normalize_for_classify(query);
    let knows = q.contains("you know") || q.contains("your knowledge") || q.contains("in your brain");
    if !knows {
        return false;
    }
    q.contains("everything you know")
        || q.contains("all you know")
        || q.contains("all of your knowledge")
        || q.contains("everything in your brain")
        || (q.contains("all of the") && q.contains("you know"))
        || (q.contains("all the") && q.contains("you know"))
        || (q.contains("all ") && q.contains("you know") && q.starts_with("list"))
}

/// True for elaboration requests like "tell me more", "go deeper",
/// "explain that further", "elaborate".
pub fn is_elaboration(query: &str) -> bool {
    let q = normalize_for_classify(query);
    let needles = [
        "tell me more",
        "go deeper",
        "explain that further",
        "explain further",
        "elaborate",
        "expand on that",
        "more on that",
        "more about that",
        "tell me more about that",
        "say more",
        "keep going",
    ];
    needles.iter().any(|n| q.contains(n))
}

/// True for continuation patterns like "and what about X", "but X",
/// "what about Y". Distinct from bare-pronoun queries.
pub fn is_continuation(query: &str) -> bool {
    let q = normalize_for_classify(query);
    let prefixes = ["and ", "but ", "what about ", "how about ", "and what about ", "and how about "];
    prefixes.iter().any(|p| q.starts_with(p))
}

/// True for a bare causal follow-up like "why?", "why is that?",
/// "why though", "how come", "but why". These carry no subject of their
/// own; the handler re-routes them to a causal query on the prior turn's
/// subject anchor (an honest, self-gating route -- causal synthesis
/// refuses cleanly when it holds no causal edges for that subject).
/// Deliberately tight: only bare/near-bare forms match, so a fully
/// specified "why is the sky blue" is left to normal recall.
pub fn is_why_followup(query: &str) -> bool {
    let q = normalize_for_classify(query);
    matches!(
        q.as_str(),
        "why"
            | "why is that"
            | "why is that so"
            | "why so"
            | "why though"
            | "why that"
            | "but why"
            | "and why"
            | "how come"
    )
}

/// Bare pronouns this layer recognises for substitution. Two sets:
/// the impersonal set (it/that/this/they/...) and the personal
/// third-person set (he/she/him/her). The personal set was the
/// OP-ROUTE-GUEST-6 gap: a follow-up like "where was he born" carried a
/// referent the layer never recognised, so it fell through to raw recall
/// and lost the subject. Adding he/she/him/her lets the guest ask path
/// resolve the referent to the current subject anchor before narrate --
/// substitution only rewrites the query; narrate still gates on grounded
/// edges, so an unresolved or ungrounded referent still refuses honestly.
const BARE_PRONOUNS: &[&str] = &[
    "it", "that", "this", "they", "them", "those", "these", "he", "she", "him", "her",
];

/// True if the query contains any of the bare pronouns from
/// `BARE_PRONOUNS` as a standalone token.
pub fn contains_bare_pronoun(query: &str) -> bool {
    let q = normalize_for_classify(query);
    for p in BARE_PRONOUNS {
        let with_space = format!(" {} ", p);
        let with_punct_end = format!(" {}?", p);
        let with_punct_period = format!(" {}.", p);
        let prefix = format!("{} ", p);
        if q.contains(&with_space)
            || q.contains(&with_punct_end)
            || q.contains(&with_punct_period)
            || q.starts_with(&prefix)
            || q.ends_with(&format!(" {}", p))
            || q == *p
        {
            return true;
        }
    }
    false
}

/// Substitute the first bare pronoun in `query` with `replacement`.
/// Returns the rewritten query. Best-effort word-boundary matching.
pub fn substitute_pronoun(query: &str, replacement: &str) -> String {
    let lowered = query.to_lowercase();
    for p in BARE_PRONOUNS {
        // Find the first occurrence that is bounded by non-alpha on
        // both sides (or string boundary).
        let bytes = lowered.as_bytes();
        let plen = p.len();
        let mut i = 0;
        while i + plen <= bytes.len() {
            if &bytes[i..i + plen] == p.as_bytes() {
                let prev_ok = if i == 0 {
                    true
                } else {
                    let c = bytes[i - 1] as char;
                    !c.is_alphabetic() && c != '_'
                };
                let next_ok = if i + plen == bytes.len() {
                    true
                } else {
                    let c = bytes[i + plen] as char;
                    !c.is_alphabetic() && c != '_'
                };
                if prev_ok && next_ok {
                    let mut out = String::with_capacity(query.len() + replacement.len());
                    out.push_str(&query[..i]);
                    out.push_str(replacement);
                    out.push_str(&query[i + plen..]);
                    return out;
                }
            }
            i += 1;
        }
    }
    query.to_string()
}

/// BUG-2 (definite follow-up): extract the head noun of a *bare* definite
/// reference like "the snake" / "the language". Returns the lowercased noun
/// ONLY for a single-noun definite phrase, so a fully specified phrase ("the
/// python language", "the theory of relativity") does NOT trigger referent
/// binding and stays on the ordinary definitional path. The caller resolves
/// the head against the prior turn's salient nouns (`resolve_noun_reference`);
/// a non-matching head simply falls through. Pure; read-only.
pub fn definite_reference_head(subject: &str) -> Option<String> {
    let s = subject.trim().to_lowercase();
    let rest = s.strip_prefix("the ")?;
    let toks: Vec<&str> = rest.split_whitespace().collect();
    if toks.len() != 1 {
        return None;
    }
    let head: String = toks[0].chars().filter(|c| c.is_alphanumeric()).collect();
    if head.len() < 3 {
        return None;
    }
    Some(head)
}

/// Lowercase + collapse whitespace + strip surrounding quotes /
/// trailing punctuation for classify-only matching. Never applied to
/// the stored `raw_query`.
fn normalize_for_classify(query: &str) -> String {
    let mut s = query.trim().to_lowercase();
    while s.ends_with('?') || s.ends_with('.') || s.ends_with('!') || s.ends_with(',') {
        s.pop();
    }
    // Collapse runs of whitespace
    let mut out = String::with_capacity(s.len());
    let mut last_was_space = false;
    for c in s.chars() {
        if c.is_whitespace() {
            if !last_was_space {
                out.push(' ');
            }
            last_was_space = true;
        } else {
            out.push(c);
            last_was_space = false;
        }
    }
    out
}

// =====================================================================
// SessionMemory
// =====================================================================

/// Per-session operational/episodic memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMemory {
    pub session_id: u64,
    pub started_at_ms: i64,
    pub last_active_ms: i64,
    pub turn_history: VecDeque<Turn>,
    /// The dominant activation across recent turns. Reset on topic shift.
    pub current_topic: Option<String>,
    /// Anchors sorted descending by weight.
    pub anchors: Vec<Anchor>,
    pub session_beliefs: Vec<SessionBelief>,
    /// OP-REASON-LOOP Phase 1.5 -- session-scoped context hold (#1396 SS10).
    /// Accumulates each prompt ContextSet so follow-ups warm-start from the
    /// union of everything seen this session. In-RAM only (this SessionMemory
    /// lives in the non-WAL brain.session_memories registry), so it is
    /// #[serde(skip)]: never persisted, rebuilt fresh, derived view data only
    /// (labels/activations), never nodes or bonds.
    #[serde(skip)]
    pub session_context: crate::reason_loop::SessionContext,
}

impl SessionMemory {
    pub fn new(session_id: u64, now_ms: i64) -> Self {
        SessionMemory {
            session_id,
            started_at_ms: now_ms,
            last_active_ms: now_ms,
            turn_history: VecDeque::with_capacity(MAX_TURN_HISTORY),
            current_topic: None,
            anchors: Vec::new(),
            session_beliefs: Vec::new(),
            session_context: crate::reason_loop::SessionContext::new(),
        }
    }

    /// Multiply every anchor's weight by `ANCHOR_DECAY_FACTOR` and
    /// bump `turns_since_introduced`. Called at the start of each
    /// new turn before promoting fresh anchors.
    pub fn decay_anchors(&mut self) {
        for a in self.anchors.iter_mut() {
            a.weight *= ANCHOR_DECAY_FACTOR;
            a.turns_since_introduced = a.turns_since_introduced.saturating_add(1);
        }
    }

    /// Detect topic shift by comparing `new_top` activations against
    /// the prior `current_topic` and the prior turn's top_activations.
    /// Returns true when the new top set has less than
    /// `TOPIC_SHIFT_OVERLAP_THRESHOLD` overlap with the prior set
    /// AND the new set is non-empty AND there was a prior topic.
    pub fn detect_topic_shift(&self, new_top: &[String]) -> bool {
        if new_top.is_empty() || self.current_topic.is_none() {
            return false;
        }
        let prior_top = match self.turn_history.back() {
            Some(t) => &t.top_activations,
            None => return false,
        };
        if prior_top.is_empty() {
            return false;
        }
        let prior_set: std::collections::HashSet<String> =
            prior_top.iter().map(|s| s.to_lowercase()).collect();
        let mut hits = 0_usize;
        for n in new_top {
            if prior_set.contains(&n.to_lowercase()) {
                hits += 1;
            }
        }
        let overlap = hits as f32 / new_top.len().max(1) as f32;
        overlap < TOPIC_SHIFT_OVERLAP_THRESHOLD
    }

    /// Promote a label to anchor with `weight=1.0`. If an anchor with
    /// the same label (case-insensitive) already exists, it is
    /// reset to weight 1.0 and `turns_since_introduced=0`.
    fn promote_anchor_inner(&mut self, label: &str, weight: f32) {
        let lower = label.to_lowercase();
        if let Some(existing) = self
            .anchors
            .iter_mut()
            .find(|a| a.label.to_lowercase() == lower)
        {
            existing.weight = weight.max(existing.weight);
            existing.turns_since_introduced = 0;
        } else {
            self.anchors.push(Anchor::new(label.to_string(), weight));
        }
    }

    /// Sort anchors descending by weight, drop the tail beyond
    /// `MAX_ANCHORS`, and drop any with weight near zero.
    fn cull_anchors(&mut self) {
        // Keep things deterministic: sort by weight desc, then by
        // turns_since_introduced asc (newer breaks ties), then by
        // label asc (for full determinism in tests).
        self.anchors.sort_by(|a, b| {
            b.weight
                .partial_cmp(&a.weight)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.turns_since_introduced.cmp(&b.turns_since_introduced))
                .then(a.label.cmp(&b.label))
        });
        // Drop near-zero weights (numerically dead anchors).
        self.anchors.retain(|a| a.weight > 1e-4);
        if self.anchors.len() > MAX_ANCHORS {
            self.anchors.truncate(MAX_ANCHORS);
        }
    }

    /// Record a turn. Order:
    ///   1. Decay existing anchors.
    ///   2. Detect topic shift; clear `current_topic` if shifted.
    ///   3. Promote primary (top_activations[0]) at weight 1.0; set
    ///      as `current_topic`.
    ///   4. Promote secondaries at decreasing weights.
    ///   5. Cull anchors.
    ///   6. Append turn (evicting oldest if at MAX_TURN_HISTORY).
    ///   7. Update `last_active_ms`.
    pub fn record_turn(
        &mut self,
        raw_query: String,
        effective_query: String,
        response_text: String,
        top_activations: Vec<String>,
        mut trace_summary: TurnTraceSummary,
        now_ms: i64,
    ) {
        // 1. Decay existing anchors before promoting fresh ones.
        self.decay_anchors();

        // 2. Topic shift detection.
        if self.detect_topic_shift(&top_activations) {
            self.current_topic = None;
        }

        // 3 + 4. Promote anchors.
        if let Some(primary) = top_activations.first() {
            self.promote_anchor_inner(primary, 1.0);
            self.current_topic = Some(primary.clone());
        }
        for (k, label) in top_activations
            .iter()
            .skip(1)
            .take(SECONDARY_ANCHOR_COUNT)
            .enumerate()
        {
            // 0.5 * 0.9^k -- always below the primary's weight.
            let w = 0.5 * 0.9_f32.powi(k as i32);
            self.promote_anchor_inner(label, w);
        }

        // 5. Cull.
        self.cull_anchors();

        // Populate top_anchors on the trace summary from the live
        // (post-promotion) anchor list — top 5 by weight desc.
        trace_summary.top_anchors = self
            .anchors
            .iter()
            .take(5)
            .map(|a| (a.label.clone(), a.weight))
            .collect();

        // 6. Append turn.
        let turn = Turn {
            timestamp_ms: now_ms,
            raw_query,
            effective_query,
            response_text,
            top_activations,
            trace_summary,
        };
        if self.turn_history.len() >= MAX_TURN_HISTORY {
            self.turn_history.pop_front();
        }
        self.turn_history.push_back(turn);

        // 7. Update activity.
        self.last_active_ms = now_ms;
    }

    /// Resolve a bare pronoun to the highest-weight anchor whose
    /// (decayed) weight clears `PRONOUN_RESOLUTION_THRESHOLD`.
    /// Returns `None` when all anchors are below threshold so the
    /// handler can ask for clarification.
    pub fn resolve_pronoun(&self) -> Option<&Anchor> {
        // Anchors are kept sorted desc by weight via cull_anchors,
        // but anchors decayed since last record_turn may not be in
        // sorted order. Iterate and pick the max above threshold.
        self.anchors
            .iter()
            .filter(|a| a.weight > PRONOUN_RESOLUTION_THRESHOLD)
            .max_by(|a, b| {
                a.weight
                    .partial_cmp(&b.weight)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// BUG-2 (definite follow-up): resolve a definite noun reference ("the
    /// snake") to a salient noun carried from a PRIOR turn's answer. `head` is
    /// the lowercased head noun (see `definite_reference_head`). Resolution
    /// order, most specific first:
    ///   1. a prior top-activation / anchor LABEL containing `head` as a whole
    ///      token -> that specific label ("the language" -> "Python
    ///      (programming language)");
    ///   2. else, if `head` appears as a whole word in the LAST answer's prose
    ///      -> the bare head ("the snake" after "(it's also a snake)" ->
    ///      "snake").
    /// Returns `None` when the noun was NOT salient last turn, so an unrelated
    /// definite follow-up never mis-binds. Read-only over episodic session
    /// state; never the graph.
    pub fn resolve_noun_reference(&self, head: &str) -> Option<String> {
        let h = head.trim().to_lowercase();
        if h.len() < 3 {
            return None;
        }
        let mut labels: Vec<&str> = Vec::new();
        if let Some(t) = self.last_turn() {
            for a in &t.top_activations {
                labels.push(a.as_str());
            }
        }
        for a in &self.anchors {
            labels.push(a.label.as_str());
        }
        for label in &labels {
            if label.to_lowercase() == h {
                return Some((*label).to_string());
            }
        }
        for label in &labels {
            if label
                .to_lowercase()
                .split(|c: char| !c.is_alphanumeric())
                .any(|tok| tok == h)
            {
                return Some((*label).to_string());
            }
        }
        if let Some(t) = self.last_turn() {
            if t
                .response_text
                .to_lowercase()
                .split(|c: char| !c.is_alphanumeric())
                .any(|tok| tok == h)
            {
                return Some(h);
            }
        }
        None
    }

    /// Append a session-scoped (subject, relation, object) belief.
    /// NEVER auto-promoted (see module docs).
    pub fn record_belief(&mut self, subject: String, relation: String, object: String) {
        let idx = self.turn_history.len();
        self.session_beliefs.push(SessionBelief {
            subject,
            relation,
            object,
            source_turn_index: idx,
        });
    }

    pub fn is_idle(&self, now_ms: i64, idle_timeout_ms: i64) -> bool {
        now_ms.saturating_sub(self.last_active_ms) > idle_timeout_ms
    }

    pub fn turn_count(&self) -> usize {
        self.turn_history.len()
    }

    pub fn last_turn(&self) -> Option<&Turn> {
        self.turn_history.back()
    }

    pub fn first_turn(&self) -> Option<&Turn> {
        self.turn_history.front()
    }
}

// =====================================================================
// SessionMemoryRegistry
// =====================================================================

/// All live `SessionMemory` instances. Idle sessions are evicted
/// lazily on each access (no background task — keeps the design
/// dependency-free).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMemoryRegistry {
    sessions: HashMap<u64, SessionMemory>,
    pub idle_timeout_ms: i64,
}

impl Default for SessionMemoryRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl SessionMemoryRegistry {
    pub fn new() -> Self {
        SessionMemoryRegistry {
            sessions: HashMap::new(),
            idle_timeout_ms: DEFAULT_IDLE_TIMEOUT_MS,
        }
    }

    pub fn with_idle_timeout(idle_timeout_ms: i64) -> Self {
        SessionMemoryRegistry {
            sessions: HashMap::new(),
            idle_timeout_ms,
        }
    }

    /// Evict idle, then return existing-or-create-new.
    pub fn get_or_create(&mut self, id: u64, now_ms: i64) -> &mut SessionMemory {
        self.evict_idle(now_ms);
        self.sessions
            .entry(id)
            .or_insert_with(|| SessionMemory::new(id, now_ms))
    }

    pub fn get(&self, id: u64) -> Option<&SessionMemory> {
        self.sessions.get(&id)
    }

    pub fn get_mut(&mut self, id: u64) -> Option<&mut SessionMemory> {
        self.sessions.get_mut(&id)
    }

    /// Explicitly remove a session. Returns the removed memory if
    /// any. Called by `lu4d_session_end`.
    pub fn end(&mut self, id: u64) -> Option<SessionMemory> {
        self.sessions.remove(&id)
    }

    /// Remove all sessions whose `last_active_ms` is older than
    /// `idle_timeout_ms`. Returns the evicted ids.
    pub fn evict_idle(&mut self, now_ms: i64) -> Vec<u64> {
        let timeout = self.idle_timeout_ms;
        let evicted: Vec<u64> = self
            .sessions
            .iter()
            .filter_map(|(id, s)| {
                if s.is_idle(now_ms, timeout) {
                    Some(*id)
                } else {
                    None
                }
            })
            .collect();
        for id in &evicted {
            self.sessions.remove(id);
        }
        evicted
    }

    pub fn len(&self) -> usize {
        self.sessions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.sessions.is_empty()
    }

    pub fn ids(&self) -> Vec<u64> {
        self.sessions.keys().copied().collect()
    }
}

// =====================================================================
// Tests
// =====================================================================

#[cfg(test)]
mod tests {
    // Phase-4b -- bare causal follow-up classifier.
    #[test]
    fn why_followup_classifier() {
        use super::{is_continuation, is_why_followup};
        assert!(is_why_followup("why?"));
        assert!(is_why_followup("Why is that?"));
        assert!(is_why_followup("how come"));
        assert!(is_why_followup("but why"));
        assert!(!is_why_followup("why is the sky blue"));
        assert!(!is_why_followup("what is a dog"));
        // "but why" is also a continuation prefix; the handler checks
        // is_why_followup FIRST so the causal route wins.
        assert!(is_continuation("but why"));
    }

    use super::*;

    fn mk(now: i64) -> SessionMemory {
        SessionMemory::new(42, now)
    }

    fn rec(
        s: &mut SessionMemory,
        raw: &str,
        eff: &str,
        resp: &str,
        top: Vec<&str>,
        now: i64,
    ) {
        s.record_turn(
            raw.to_string(),
            eff.to_string(),
            resp.to_string(),
            top.into_iter().map(String::from).collect(),
            TurnTraceSummary::empty(),
            now,
        );
    }

    // 1
    #[test]
    fn q3_unit_new_session_is_empty() {
        let s = mk(1000);
        assert_eq!(s.session_id, 42);
        assert_eq!(s.started_at_ms, 1000);
        assert_eq!(s.last_active_ms, 1000);
        assert!(s.turn_history.is_empty());
        assert!(s.anchors.is_empty());
        assert!(s.session_beliefs.is_empty());
        assert!(s.current_topic.is_none());
    }

    // 2
    #[test]
    fn q3_unit_record_turn_appends_and_updates_last_active() {
        let mut s = mk(1000);
        rec(&mut s, "What is photosynthesis?", "What is photosynthesis?", "It is a process.", vec!["photosynthesis", "chlorophyll"], 2000);
        assert_eq!(s.turn_history.len(), 1);
        assert_eq!(s.last_active_ms, 2000);
        assert_eq!(s.current_topic.as_deref(), Some("photosynthesis"));
    }

    // 3 — RAW QUERY VERBATIM
    #[test]
    fn q3_unit_raw_query_is_stored_verbatim_for_backreference() {
        let mut s = mk(0);
        let raw = "What is photosynthesis?";
        rec(&mut s, raw, raw, "answer", vec!["photosynthesis"], 1);
        let back = s.first_turn().unwrap();
        // Character-for-character identity test.
        assert_eq!(back.raw_query, raw);
        assert_eq!(back.raw_query.as_bytes(), raw.as_bytes());
    }

    // 4 — anchor decay halves weight per turn
    #[test]
    fn q3_unit_anchor_weight_decays_by_half_each_turn() {
        let mut s = mk(0);
        rec(&mut s, "q1", "q1", "r1", vec!["alpha"], 1);
        let w0 = s.anchors.iter().find(|a| a.label == "alpha").unwrap().weight;
        assert!((w0 - 1.0).abs() < 1e-6);
        // New turn whose top is unrelated does NOT re-promote alpha
        rec(&mut s, "q2", "q2", "r2", vec!["zulu"], 2);
        let w1 = s.anchors.iter().find(|a| a.label == "alpha").unwrap().weight;
        assert!((w1 - 0.5).abs() < 1e-5, "got {}", w1);
        rec(&mut s, "q3", "q3", "r3", vec!["zulu"], 3);
        let w2 = s.anchors.iter().find(|a| a.label == "alpha").unwrap().weight;
        assert!((w2 - 0.25).abs() < 1e-5, "got {}", w2);
    }

    // 5
    #[test]
    fn q3_unit_pronoun_resolves_to_highest_weight_anchor_above_threshold() {
        let mut s = mk(0);
        rec(&mut s, "q1", "q1", "r1", vec!["photosynthesis"], 1);
        let r = s.resolve_pronoun().expect("should resolve");
        assert_eq!(r.label, "photosynthesis");
        assert!(r.weight > PRONOUN_RESOLUTION_THRESHOLD);
    }

    // 6 — STALE ANCHOR returns None below threshold
    #[test]
    fn q3_unit_pronoun_resolution_returns_none_when_all_anchors_below_threshold() {
        let mut s = mk(0);
        rec(&mut s, "q1", "q1", "r1", vec!["alpha"], 1);
        // Decay several turns with unrelated tops.
        rec(&mut s, "q2", "q2", "r2", vec!["zulu"], 2); // alpha -> 0.5
        rec(&mut s, "q3", "q3", "r3", vec!["yankee"], 3); // alpha -> 0.25, zulu -> 0.25
        rec(&mut s, "q4", "q4", "r4", vec!["xray"], 4); // all <= 0.25 (xray=1.0 fresh)
        // Xray IS above threshold; we'd resolve to it. But to test the
        // truly-stale case, knock weight directly.
        for a in s.anchors.iter_mut() { a.weight = 0.1; }
        assert!(s.resolve_pronoun().is_none(), "all anchors below threshold should return None, got {:?}", s.anchors);
    }

    // 7 — high-weight 2-turns-ago beats low-weight 1-turn-ago
    #[test]
    fn q3_unit_pronoun_prefers_high_weight_recent_over_low_weight_more_recent() {
        let mut s = mk(0);
        // Strong primary 2 turns ago.
        rec(&mut s, "q1", "q1", "r1", vec!["photosynthesis", "chlorophyll", "leaves"], 1);
        // 1-turn-ago has a SECONDARY anchor "weak" but no primary that overlaps.
        rec(&mut s, "q2", "q2", "r2", vec!["weak"], 2);
        // photosynthesis decayed to 0.5, weak is at 1.0. Resolution
        // should pick weak (highest current weight).
        let r = s.resolve_pronoun().unwrap();
        assert_eq!(r.label, "weak");
        // But if we then add another unrelated turn, weak decays to 0.5
        // and photosynthesis to 0.25 — pronoun resolution still picks
        // the highest.
        rec(&mut s, "q3", "q3", "r3", vec!["other"], 3);
        let r2 = s.resolve_pronoun().unwrap();
        assert_eq!(r2.label, "other"); // fresh primary at 1.0
    }

    // 8 — backreference classifier
    #[test]
    fn q3_unit_backreference_classifier() {
        assert_eq!(classify_backreference("What did I just ask?"), Some(BackrefKind::LastQuestion));
        // OP-QA-3 enumeration classifier.
        assert!(is_knowledge_enumeration("Tell me all of the program code you know?"));
        assert!(is_knowledge_enumeration("tell me everything you know"));
        assert!(is_knowledge_enumeration("list all the animals you know"));
        assert!(!is_knowledge_enumeration("what do you know"));
        assert!(!is_knowledge_enumeration("what is a dog"));
        assert!(!is_knowledge_enumeration("do you know python"));
        assert_eq!(classify_backreference("what was my last question"), Some(BackrefKind::LastQuestion));
        assert_eq!(classify_backreference("What was my first question?"), Some(BackrefKind::FirstQuestion));
        assert_eq!(classify_backreference("what was the previous answer?"), Some(BackrefKind::LastAnswer));
        assert_eq!(classify_backreference("Tell me about photosynthesis."), None);
    }

    // 9 — elaboration / continuation classifiers
    #[test]
    fn q3_unit_elaboration_and_continuation_classifiers() {
        assert!(is_elaboration("Tell me more."));
        assert!(is_elaboration("go deeper"));
        assert!(is_elaboration("explain that further"));
        assert!(!is_elaboration("What is X?"));

        assert!(is_continuation("And what about photosynthesis?"));
        assert!(is_continuation("but X?"));
        assert!(is_continuation("what about Y?"));
        assert!(!is_continuation("What is X?"));
    }

    // 10 — bare pronoun detection + substitution
    #[test]
    fn q3_unit_pronoun_detection_and_substitution() {
        assert!(contains_bare_pronoun("What does it produce?"));
        assert!(contains_bare_pronoun("Tell me about that."));
        assert!(contains_bare_pronoun("they"));
        assert!(!contains_bare_pronoun("What is photosynthesis?"));
        // "transit" contains "it" as substring but not as a token.
        assert!(!contains_bare_pronoun("What is transit?"));

        let out = substitute_pronoun("What does it produce?", "photosynthesis");
        assert_eq!(out, "What does photosynthesis produce?");
        let out2 = substitute_pronoun("Tell me about that.", "Paris");
        assert_eq!(out2, "Tell me about Paris.");

        // OP-ROUTE-GUEST-6 personal-pronoun gap: he/she/him/her must be
        // recognised and substituted so "where was he born" resolves the
        // referent to the subject anchor.
        assert!(contains_bare_pronoun("where was he born"));
        assert!(contains_bare_pronoun("what did she discover"));
        assert!(contains_bare_pronoun("what is he known for"));
        // Boundary guards: pronoun substrings inside other words are not
        // tokens ("where"/"there" contain he/her; "the" contains he).
        assert!(!contains_bare_pronoun("what is over there"));
        assert!(!contains_bare_pronoun("what is the theory"));
        let out3 = substitute_pronoun("where was he born", "Einstein");
        assert_eq!(out3, "where was Einstein born");
        let out4 = substitute_pronoun("what did she discover", "Curie");
        assert_eq!(out4, "what did Curie discover");
        // "where" must survive untouched (he/her are substrings, not tokens).
        let out5 = substitute_pronoun("where was he born", "Einstein");
        assert!(out5.starts_with("where was Einstein"));
    }

    // 10b — personal pronoun resolves to the current person anchor
    #[test]
    fn q3_unit_personal_pronoun_resolves_to_subject_anchor() {
        let mut s = mk(0);
        // Mirror the Einstein turn: recall put "Einstein" at the top.
        rec(&mut s, "who was Einstein", "who was Einstein", "...",
            vec!["Einstein", "influential", "Albert Einstein"], 1);
        let anchor = s.resolve_pronoun().expect("an anchor above threshold");
        assert_eq!(anchor.label, "Einstein");
        let rewritten = substitute_pronoun("where was he born", &anchor.label);
        assert_eq!(rewritten, "where was Einstein born");
    }

    // 11 — topic shift detection
    #[test]
    fn q3_unit_topic_shift_detection_clears_current_topic() {
        let mut s = mk(0);
        rec(&mut s, "q1", "q1", "r1", vec!["alpha", "beta", "gamma"], 1);
        assert_eq!(s.current_topic.as_deref(), Some("alpha"));
        // Same neighborhood — no shift
        rec(&mut s, "q2", "q2", "r2", vec!["alpha", "delta"], 2);
        assert_eq!(s.current_topic.as_deref(), Some("alpha"));
        // Total switch — should detect topic shift, then re-promote new primary
        rec(&mut s, "q3", "q3", "r3", vec!["xray", "yankee", "zulu"], 3);
        assert_eq!(s.current_topic.as_deref(), Some("xray"));
    }

    // 12 — registry idle eviction
    #[test]
    fn q3_unit_registry_evicts_idle_sessions() {
        let mut reg = SessionMemoryRegistry::with_idle_timeout(1000);
        {
            let _s = reg.get_or_create(1, 0);
        }
        {
            let _s = reg.get_or_create(2, 500);
        }
        // At t=2000, session 1 (idle for 2000ms) and session 2 (idle for 1500ms) are stale.
        let evicted = reg.evict_idle(2000);
        assert_eq!(evicted.len(), 2);
        assert!(reg.get(1).is_none());
        assert!(reg.get(2).is_none());
    }

    // 13 — registry lazy eviction on get_or_create
    #[test]
    fn q3_unit_registry_get_or_create_lazily_evicts() {
        let mut reg = SessionMemoryRegistry::with_idle_timeout(1000);
        {
            let _s = reg.get_or_create(1, 0);
        }
        // Touching a fresh ID at t=2000 should sweep session 1 first.
        let _s2 = reg.get_or_create(2, 2000);
        assert!(reg.get(1).is_none());
        assert!(reg.get(2).is_some());
    }

    // 14 — registry session_end removes
    #[test]
    fn q3_unit_registry_end_removes_session() {
        let mut reg = SessionMemoryRegistry::new();
        let _ = reg.get_or_create(7, 0);
        assert!(reg.get(7).is_some());
        let removed = reg.end(7);
        assert!(removed.is_some());
        assert!(reg.get(7).is_none());
    }

    // 15 — sessions are isolated
    #[test]
    fn q3_unit_two_sessions_do_not_share_anchors() {
        let mut reg = SessionMemoryRegistry::new();
        {
            let s1 = reg.get_or_create(1, 0);
            s1.record_turn("q".into(), "q".into(), "r".into(), vec!["alpha".into()], TurnTraceSummary::empty(), 1);
        }
        {
            let s2 = reg.get_or_create(2, 0);
            s2.record_turn("q".into(), "q".into(), "r".into(), vec!["zulu".into()], TurnTraceSummary::empty(), 1);
        }
        let s1 = reg.get(1).unwrap();
        let s2 = reg.get(2).unwrap();
        assert_eq!(s1.current_topic.as_deref(), Some("alpha"));
        assert_eq!(s2.current_topic.as_deref(), Some("zulu"));
        assert!(s1.anchors.iter().all(|a| a.label != "zulu"));
        assert!(s2.anchors.iter().all(|a| a.label != "alpha"));
    }

    // 16 — turn history capped
    #[test]
    fn q3_unit_turn_history_is_capped_at_max() {
        let mut s = mk(0);
        for i in 0..(MAX_TURN_HISTORY + 5) {
            rec(&mut s, &format!("q{}", i), &format!("q{}", i), "r", vec!["t"], i as i64 + 1);
        }
        assert_eq!(s.turn_history.len(), MAX_TURN_HISTORY);
        // First retained turn should be the (5)th overall (0-indexed 5).
        let first_kept = s.first_turn().unwrap();
        assert_eq!(first_kept.raw_query, format!("q{}", 5));
    }
}

#[cfg(test)]
mod bug2_referent_tests {
    use super::*;

    fn sess_with_turn(resp: &str, acts: Vec<&str>) -> SessionMemory {
        let mut s = SessionMemory::new(1, 0);
        s.record_turn(
            "what is python".into(),
            "what is python".into(),
            resp.into(),
            acts.into_iter().map(|x| x.to_string()).collect(),
            TurnTraceSummary::empty(),
            1,
        );
        s
    }

    #[test]
    fn definite_head_extracts_single_noun_only() {
        assert_eq!(definite_reference_head("the snake").as_deref(), Some("snake"));
        assert_eq!(definite_reference_head("the language").as_deref(), Some("language"));
        // multi-token definite phrase does NOT trigger referent binding
        assert_eq!(definite_reference_head("the python language"), None);
        assert_eq!(definite_reference_head("the theory of relativity"), None);
        // not a definite phrase
        assert_eq!(definite_reference_head("snake"), None);
        assert_eq!(definite_reference_head("Albert Einstein"), None);
    }

    #[test]
    fn resolve_binds_noun_named_in_prior_answer_prose() {
        // Definition-shape answers leave activations empty; the salient noun
        // must still be recoverable from the prior answer's prose.
        let s = sess_with_turn("Python is a language. (It's also a snake.)", vec![]);
        assert_eq!(s.resolve_noun_reference("snake").as_deref(), Some("snake"));
    }

    #[test]
    fn resolve_prefers_specific_activation_label() {
        let s = sess_with_turn(
            "Python is a language.",
            vec!["Python (programming language)"],
        );
        assert_eq!(
            s.resolve_noun_reference("language").as_deref(),
            Some("Python (programming language)")
        );
    }

    #[test]
    fn resolve_returns_none_for_unrelated_noun() {
        // "the snake" after an answer with no snake must NOT mis-bind.
        let s = sess_with_turn("Gravity is a fundamental interaction.", vec!["gravity"]);
        assert_eq!(s.resolve_noun_reference("snake"), None);
    }
}
