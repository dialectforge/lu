//! Port K (#581) — Honest "I don't know" gate.
//! Port K.1 — single-anchor-dominance coverage signal (Mode D).
//!
//! Closes the credibility gap identified by the #598 eval baseline:
//! Lu was fabricating weak associative chains on nonsense input
//! (`flibbergast`) and on partial-match queries that paired a real
//! token with a junk token (`zorblax photosynthesis`,
//! `quantum jellyfish dynamics`).
//!
//! K.1 (#609 successor) closes the single-anchor-dominance shape that
//! the original three-signal floor missed: queries like
//! `quantum jellyfish dynamics` where ONE token (`quantum`) has a
//! rich neighborhood and clears the seed-activation floor on its
//! own (max_activation 0.866) while the other tokens (`jellyfish`,
//! `dynamics`) contribute nothing relevant. The chain Lu would
//! articulate is built almost entirely from the dominant token's
//! neighborhood while the other query tokens are silently dropped.
//!
//! Three refusal modes:
//!
//! - **Mode A — Total IDK.** All three signals fire. The query has no
//!   meaningful traction at all. Prose explains *why* the brain has
//!   nothing to say.
//!
//!     1. `total_activation` across all activated nodes < `activation_floor` (1.5)
//!     2. ReasoningTrace is `no_chain` OR (`Associative` AND `confidence` < `mixed_confidence_floor` 0.4)
//!     3. max seed-node activation (`depth == 0`) < `seed_activation_floor` (0.6)
//!        AND no query token has a keyword-index hit
//!
//! - **Mode B — Partial-match leak.** Some query tokens match the
//!   brain, others don't. The cross-token chain isn't structured. We
//!   refuse to fabricate a connection between known and unknown.
//!
//! - **Mode D — Single-anchor dominance (K.1).** All tokens have
//!   keyword hits and the strongest anchor clears the seed floor (so
//!   Modes A/B/C all pass), but only ONE token actually anchors
//!   strongly: the others have hits but their max activation in the
//!   recall is below `per_token_activation_floor`. The brain has a
//!   rich neighborhood for one query token and effectively nothing
//!   for the rest, so any answer would be that one token's chain
//!   wearing the missing tokens as decoration. Refuse instead.
//!
//!     Gate condition: N >= `coverage_min_tokens` (default 3) AND
//!     `anchored_token_count / N` < `coverage_floor` (default 0.5)
//!     AND we would otherwise pass (max_activation >= seed floor).
//!     For N=2 the signal is too noisy and Mode B/C already cover
//!     the typical false-positive cases. For N=1 coverage is
//!     trivially 1.0 or 0.0, so the signal is irrelevant.
//!
//! Pairs with (does NOT replace) the existing layers:
//!
//! - **Bug B fix (#573)** — bare-word recall fallback runs first inside
//!   `select_seeds_for_query`. K only fires when nothing meaningful
//!   survives even the relaxed seed pass.
//! - **Port D (#407)** — Mixed-pattern hedged prose stays in place for
//!   legitimate weak-but-non-trivial chains. K is one tier above —
//!   refuses cleanly when even the hedged path would fabricate.
//! - **Q.3 (#39)** — pronoun / elaboration rewrite happens before
//!   recall. K runs against the rewritten query; if the session anchor
//!   was strong enough to rewrite cleanly, signal 1 will pass and K
//!   defers to normal narration. Only refuses when even the rewritten
//!   query produces no signal.

use crate::brain::{AskResult, Brain};
use crate::reasoning::{EvidenceKind, ReasoningTrace};

/// Tunable thresholds for the three-signal IDK gate. Defaults are
/// chosen to fire on the eval-flagged nonsense cases without refusing
/// legitimate-but-weak queries; see `nonsense_tests` and
/// `weak_real_tests` for the pinned-down behavior.
#[derive(Debug, Clone, Copy)]
pub struct IdkThresholds {
    /// Signal 1 floor: total activation across all activated nodes.
    pub activation_floor: f32,
    /// Signal 2 floor: confidence below which an Associative chain
    /// counts as "weak".
    pub mixed_confidence_floor: f32,
    /// Signal 3 floor: max activation at any depth-0 (seeded) node.
    pub seed_activation_floor: f32,
    /// Mode B trigger: how many unmatched query tokens are required
    /// to suspect a partial-match leak.
    pub partial_leak_min_unmatched: usize,
    /// Mode B requires at least this many tokens to ALSO match the
    /// brain (otherwise it would just be Mode A — full IDK).
    pub partial_leak_min_matched: usize,
    /// Mode C floor: max activation across all activated nodes.
    /// On richly-trained brains where junk words can have keyword
    /// hits via prior research ingestion, Mode B's matched/unmatched
    /// split misses queries like `zorblax photosynthesis` where
    /// every token has SOME hit but no token anchors strongly.
    /// Mode C fires when a multi-token query produces a shaky top
    /// activation (the strongest anchor in the result is below the
    /// threshold a real-concept seed would hit).
    pub multi_anchor_top_floor: f32,
    /// Mode C requires at least this many seed_hits.
    pub multi_anchor_min_seeds: usize,
    /// Mode D (K.1) per-token anchor strength floor. A query token
    /// counts as "anchored" when at least this many of its top
    /// keyword-hit seed's 1-hop neighbors appear in the recall's
    /// activated set (a `top_k`-truncated map; the canonical brain's
    /// 401k-node spread at top_k=5 keeps depth-1 neighbors of the
    /// strongest seed but trims most depth-2 leaves). Tuned
    /// empirically against the canonical brain at top_k=5 (Port
    /// K.1-final probe vs. earlier #609 probe re-ran):
    ///   quantum jellyfish dynamics:        0 / 3 / 1 → 1 anchors
    ///   python recipe pasta:               4 / 1 / 1 → 1 anchors
    ///   napoleon photosynthesis algorithm: 2 / 0 / 0 → 1 anchors
    ///   paris france capital:              4 / 2 / 2 → 3 anchor
    /// At floor 2 the must-fire / must-pass split is clean: every
    /// real multi-concept query keeps every token at ≥ 2, every
    /// dominance pattern leaves at least two tokens at < 2. Floor 3
    /// (the original spec value) regressed paris france capital on
    /// the present brain because france/capital each surface only
    /// 2 activated neighbors after top-k truncation.
    pub per_token_anchor_count_floor: usize,
    /// Diagnostic-only: per-token max-activation floor. Surfaced in
    /// the structured payload but NOT used for the anchored decision
    /// (the empirical study showed max-activation tracks IDF-weighted
    /// seed weight rather than neighborhood quality). Retained so
    /// downstream tools can still cross-check.
    pub per_token_activation_floor: f32,
    /// Mode D coverage ratio floor. When `anchored_token_count / N`
    /// drops below this and N >= `coverage_min_tokens`, Mode D fires.
    pub coverage_floor: f32,
    /// Mode D minimum content-token count. Below this we skip the
    /// coverage check entirely (N=1 is trivial; N=2 is too noisy and
    /// Modes B/C already catch the bad cases).
    pub coverage_min_tokens: usize,
    /// BR.R3 (OP-008): Mode C top-seed coverage escape floor. When
    /// `top_seed_bucket_coverage / content_token_count` clears this
    /// ratio AND Mode C's `multi_anchor_shaky` would otherwise refuse,
    /// the gate falls through to Mode D rather than returning
    /// PartialMatchLeak. Empirically anchored at 0.6 on the 4.5M-bond
    /// canonical: that floor admits Q3 American Revolution (1.0), Q4
    /// Albert Einstein (0.67), Q6 Umbundu (0.67), Q7 Bertrand Russell
    /// (0.67) while preserving every Q11/Q12/A1-A8 adversarial whose
    /// top seed (post-BR.R3-seed-selection) clears at most 0.5.
    pub top_seed_coverage_floor: f32,
}

impl Default for IdkThresholds {
    fn default() -> Self {
        Self {
            activation_floor: 1.5,
            mixed_confidence_floor: 0.4,
            seed_activation_floor: 0.6,
            partial_leak_min_unmatched: 1,
            partial_leak_min_matched: 1,
            multi_anchor_top_floor: 0.70,
            multi_anchor_min_seeds: 2,
            per_token_anchor_count_floor: 2,
            per_token_activation_floor: 0.30,
            coverage_floor: 0.5,
            coverage_min_tokens: 3,
            top_seed_coverage_floor: 0.6,
        }
    }
}

/// Captured signal values for one ask. Surfaced inside the gate's
/// structured payload so callers can audit *why* K fired.
#[derive(Debug, Clone)]
pub struct IdkSignals {
    pub total_activation: f32,
    pub max_seed_activation: f32,
    /// Top activation across ALL activated nodes (any depth).
    pub max_activation: f32,
    /// Number of seeds that initiated this spread.
    pub seed_count: usize,
    /// Query tokens (after the same lowercase / hub-stopword /
    /// query-noise filters `select_seeds_for_query` uses) that have
    /// at least one keyword-index hit.
    pub matched_tokens: Vec<String>,
    /// Query tokens with zero keyword-index hits.
    pub unmatched_tokens: Vec<String>,
    /// `ReasoningTrace.no_chain` from the gate's reasoning pass.
    pub chain_no_chain: bool,
    /// `evidence_kind == Associative AND confidence < mixed_confidence_floor`.
    pub chain_associative_weak: bool,
    /// `confidence` from the reasoning trace, if one was run. None
    /// when the gate skipped reasoning because pre-check signals
    /// already looked healthy.
    pub chain_confidence: Option<f32>,
    /// **Signal 4 (K.1).** For each content token (post-stopword /
    /// query-noise filter), the maximum activation any of its
    /// keyword-index hits reached during this spread. 0.0 when the
    /// token had no hits at all (token is in `unmatched_tokens`) or
    /// when none of its hits made it into the activated set.
    /// Diagnostic only — see
    /// `per_token_neighbor_activated_count` for the actual anchored
    /// decision input.
    pub per_token_max_activation: Vec<(String, f32)>,
    /// **Signal 4 (K.1).** For each content token, the number of
    /// 1-hop neighbors of the token's top keyword-hit seed that
    /// appear in the recall's activated set. Drives the anchored
    /// decision: a token counts as anchored when this is >=
    /// `per_token_anchor_count_floor`. 0 when the token has no
    /// keyword hits.
    pub per_token_neighbor_activated_count: Vec<(String, usize)>,
    /// **Signal 4 (K.1).** Number of content tokens whose
    /// `per_token_neighbor_activated_count` is >=
    /// `per_token_anchor_count_floor`.
    pub anchored_token_count: usize,
    /// **Signal 4 (K.1).** Total content-token count after filtering.
    /// Equal to `matched_tokens.len() + unmatched_tokens.len()`.
    pub content_token_count: usize,
    /// **Signal 4 (K.1).** The single token whose anchor activated
    /// strongest, when Mode D's pattern is detected (one token
    /// dominates while others fail to anchor). `None` when Mode D
    /// did not fire.
    pub dominant_token: Option<String>,
    /// **Signal 4 (K.1).** Tokens that did NOT clear
    /// `per_token_activation_floor` when Mode D fired. Surfaced for
    /// the refusal prose so the user sees which terms got dropped.
    /// Empty when Mode D did not fire.
    pub coverage_missing_tokens: Vec<String>,
    /// **Signal 5 (BR.R3).** Number of distinct content-token
    /// keyword-index buckets that contain the TOP ASK seed
    /// (`ask.seed_hits[0]`). Drives Mode C's
    /// `top_seeded_topically` escape: when the top seed itself
    /// covers a majority of query tokens via the bucket index,
    /// the recall is topically grounded even if not every token's
    /// keyword hit reaches the per-token seed-activation floor.
    /// 0 when `seed_hits` is empty.
    pub top_seed_bucket_coverage: usize,
}

impl IdkSignals {
    /// True when the K.1 single-anchor-dominance pattern is present:
    /// the query has enough content tokens to demand coverage, but
    /// fewer than `coverage_floor` of them anchored. Independent of
    /// the verdict — use this from `build_response` to pick the
    /// dominance prose, and from `classify` to fire Mode D after the
    /// other signals have passed.
    pub fn is_single_anchor_dominance(&self, thresholds: &IdkThresholds) -> bool {
        if self.content_token_count < thresholds.coverage_min_tokens {
            return false;
        }
        let ratio = self.anchored_token_count as f32 / self.content_token_count as f32;
        ratio < thresholds.coverage_floor
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IdkVerdict {
    /// Don't fire — let normal narration proceed.
    Pass,
    /// All-three signals — clean refusal.
    TotalIdk,
    /// Some tokens match, some don't — refuse to fabricate the bridge.
    PartialMatchLeak,
}

/// Tokenize a query the same way `select_seeds_for_query` does, but
/// keeping the surface tokens themselves (not seeds). Used by the
/// gate to ask "did each query token actually find anything in the
/// keyword index?"
pub fn extract_query_tokens(query: &str) -> Vec<String> {
    let raw: Vec<String> = query
        .to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty() && w.len() >= 3)
        .map(|s| s.to_string())
        .collect();
    raw.into_iter()
        .filter(|t| {
            !crate::network::is_hub_stopword(t)
                && !crate::network::is_query_noise(t)
        })
        .collect()
}

/// Cheap pre-check using only `AskResult`. Returns `true` when the
/// recall already looks healthy and the gate can short-circuit
/// (skipping the reasoning pass entirely).
///
/// Healthy means: total activation comfortably above the floor AND
/// every query token has at least one keyword-index match AND the
/// K.1 single-anchor-dominance pattern is NOT present. If any
/// condition fails we run the full check.
///
/// K.1 awareness: a query like `quantum jellyfish dynamics` has
/// every token clearing keyword-index hits and total activation
/// above the floor (quantum lights up the brain), but only one
/// token actually anchors a connected neighborhood — letting it
/// fast-pass would silently fabricate a chain across the others.
/// Run the cheap (no-trace) signal evaluation when the query has
/// enough content tokens to make the coverage signal meaningful.
pub fn fast_pass_check(
    brain: &Brain,
    ask: &AskResult,
    effective_query: &str,
    thresholds: &IdkThresholds,
) -> bool {
    let total_activation: f32 =
        ask.recall.activated.iter().map(|a| a.activation).sum();
    if total_activation < thresholds.activation_floor * 2.0 {
        return false;
    }
    let tokens = extract_query_tokens(effective_query);
    if tokens.is_empty() {
        // No tokens after filtering — Bug B path or pure-noise query.
        // Defer to full check.
        return false;
    }
    for tok in &tokens {
        if brain.network.keyword_search(tok).is_empty() {
            return false;
        }
    }
    // K.1 (Port #610) — dominance check before fast-pass: only
    // bother when we have enough content tokens for coverage to
    // mean something. Below `coverage_min_tokens` the dominance
    // signal is too noisy and Modes B/C handle the bad cases.
    if tokens.len() >= thresholds.coverage_min_tokens {
        let signals = evaluate_signals(brain, ask, effective_query, None, thresholds);
        if signals.is_single_anchor_dominance(thresholds) {
            return false;
        }
    }
    true
}

/// Compute every signal needed for classification. `trace` is
/// optional: pass `None` if the caller already decided the recall
/// looked healthy; signals 1 and 3 will still be computed and
/// classification will treat the chain signals as "strong" (i.e.
/// classify will not fire on chain-related conditions).
pub fn evaluate_signals(
    brain: &Brain,
    ask: &AskResult,
    effective_query: &str,
    trace: Option<&ReasoningTrace>,
    thresholds: &IdkThresholds,
) -> IdkSignals {
    let total_activation: f32 =
        ask.recall.activated.iter().map(|a| a.activation).sum();
    let max_seed_activation: f32 = ask
        .recall
        .activated
        .iter()
        .filter(|a| a.depth == 0)
        .map(|a| a.activation)
        .fold(0.0_f32, f32::max);
    let max_activation: f32 = ask
        .recall
        .activated
        .iter()
        .map(|a| a.activation)
        .fold(0.0_f32, f32::max);
    let seed_count = ask.seed_hits.len();

    // Index activated nodes by seed for O(1) lookup when computing
    // per-token max activation across many keyword hits.
    let mut act_by_seed: std::collections::HashMap<crate::seed::Seed, f32> =
        std::collections::HashMap::with_capacity(ask.recall.activated.len());
    for a in &ask.recall.activated {
        let entry = act_by_seed.entry(a.seed).or_insert(0.0_f32);
        if a.activation > *entry {
            *entry = a.activation;
        }
    }

    let tokens = extract_query_tokens(effective_query);
    let activated_set: std::collections::HashSet<crate::seed::Seed> =
        ask.recall.activated.iter().map(|a| a.seed).collect();
    let mut matched = Vec::new();
    let mut unmatched = Vec::new();
    let mut per_token_max_activation: Vec<(String, f32)> =
        Vec::with_capacity(tokens.len());
    let mut per_token_neighbor_activated_count: Vec<(String, usize)> =
        Vec::with_capacity(tokens.len());
    for tok in &tokens {
        let hits = brain.network.keyword_search(tok);
        if hits.is_empty() {
            unmatched.push(tok.clone());
            per_token_max_activation.push((tok.clone(), 0.0));
            per_token_neighbor_activated_count.push((tok.clone(), 0));
        } else {
            matched.push(tok.clone());
            // Diagnostic: max activation of any keyword-hit seed
            // for this token, even though we use neighborhood reach
            // for the anchored decision below.
            let token_max = hits
                .iter()
                .map(|(s, _)| *act_by_seed.get(s).unwrap_or(&0.0_f32))
                .fold(0.0_f32, f32::max);
            per_token_max_activation.push((tok.clone(), token_max));
            // Anchor strength = how many 1-hop neighbors of the
            // token's top keyword-hit seed appear in the activated
            // set. "Top" means the seed `keyword_search` ranked
            // first by hit_count; deterministic and matches what
            // the cascading recall path uses. Empirical study on
            // the canonical brain (#609 successor probe) showed
            // this distinguishes single-anchor dominance from
            // legitimate multi-anchor spread; per-token max
            // activation alone does not (it tracks IDF-weighted
            // seed weight, which is identical across all queries).
            let top_seed = hits[0].0;
            let mut nbr_set = std::collections::HashSet::new();
            for n in brain.network.seam_outgoing_set(&top_seed) {
                nbr_set.insert(n);
            }
            for n in brain.network.seam_incoming_set(&top_seed) {
                nbr_set.insert(n);
            }
            let activated_neighbor_count = nbr_set
                .iter()
                .filter(|n| activated_set.contains(n))
                .count();
            per_token_neighbor_activated_count
                .push((tok.clone(), activated_neighbor_count));
        }
    }

    let content_token_count = matched.len() + unmatched.len();
    let anchored_token_count = per_token_neighbor_activated_count
        .iter()
        .filter(|(_, c)| *c >= thresholds.per_token_anchor_count_floor)
        .count();

    // Pick the dominant token (highest neighborhood reach) and the
    // missing tokens (below floor) for the Mode D refusal prose.
    // These are populated unconditionally; classify() decides
    // whether to use them. dominant_token stays `None` when no
    // token reached the floor (Mode D requires at least one
    // genuine anchor).
    let dominant_token = per_token_neighbor_activated_count
        .iter()
        .filter(|(_, c)| *c >= thresholds.per_token_anchor_count_floor)
        .max_by_key(|(_, c)| *c)
        .map(|(t, _)| t.clone());
    let coverage_missing_tokens: Vec<String> = per_token_neighbor_activated_count
        .iter()
        .filter(|(_, c)| *c < thresholds.per_token_anchor_count_floor)
        .map(|(t, _)| t.clone())
        .collect();

    let (chain_no_chain, chain_associative_weak, chain_confidence) =
        match trace {
            Some(t) => (
                t.no_chain,
                matches!(t.evidence_kind, EvidenceKind::Associative)
                    && t.confidence < thresholds.mixed_confidence_floor,
                Some(t.confidence),
            ),
            None => (false, false, None),
        };

    // BR.R3 (OP-008): cross-token bucket coverage of the TOP ASK seed.
    // After the BR.R3 seed-selection fix in `select_seeds_for_query_ask`
    // surfaces canonical-article hubs ("Bertrand Russell", "Albert
    // Einstein", "American Revolution") as the top seed via degree-sorted
    // majority bucket coverage, this signal measures how many content-
    // token buckets the top seed sits in. Combined with the bucket scan,
    // it gives the gate a "this seed is topically dense across the query"
    // input that Mode C can use to escape PartialMatchLeak when individual
    // tokens fail to clear `seed_activation_floor` (e.g., "born" or
    // "philosophers" tokens that activate at 0.0 even with the right
    // article seeded). Computed only on the top ASK seed -- not the
    // sorted seed_hits[1..] -- because the rank-1 seed dominates spread
    // weight (rank_mult 1.0 vs 0.6 / 0.35).
    let mut top_seed_bucket_coverage: usize = 0;
    if let Some(top_seed) = ask.seed_hits.first() {
        for tok in &tokens {
            if let Some(bucket) = brain.network.keyword_index_bucket(tok) {
                if bucket
                    .binary_search_by(|s| s.as_bytes().cmp(top_seed.as_bytes()))
                    .is_ok()
                {
                    top_seed_bucket_coverage += 1;
                }
            }
        }
    }

    IdkSignals {
        total_activation,
        max_seed_activation,
        max_activation,
        seed_count,
        matched_tokens: matched,
        unmatched_tokens: unmatched,
        chain_no_chain,
        chain_associative_weak,
        chain_confidence,
        per_token_max_activation,
        per_token_neighbor_activated_count,
        anchored_token_count,
        content_token_count,
        dominant_token,
        coverage_missing_tokens,
        top_seed_bucket_coverage,
    }
}

/// Combine the signals into a verdict. Two refusal modes are
/// distinguished so the prose can be tailored.
pub fn classify(
    signals: &IdkSignals,
    thresholds: &IdkThresholds,
) -> IdkVerdict {
    let chain_weak = signals.chain_no_chain || signals.chain_associative_weak;

    // Mode A — total IDK: all three signals fire and no token even
    // matched the keyword index.
    let total = signals.total_activation < thresholds.activation_floor
        && chain_weak
        && signals.max_seed_activation < thresholds.seed_activation_floor
        && signals.matched_tokens.is_empty();
    if total {
        return IdkVerdict::TotalIdk;
    }

    // Mode B — partial-match leak: some tokens match the brain, some
    // don't. Even if the matched-token side has a structured chain,
    // that chain can't honor the unmatched concept (which by definition
    // isn't a node), so any answer would silently drop it. Refuse to
    // bridge known and unknown rather than fabricate a connection.
    //
    // Pre-spec: Mode B originally also required `chain_weak`, but the
    // eval-flagged `zorblax photosynthesis` pattern produces a fully
    // structured chain on the matched half (photosynthesis -> glucose
    // -> oxygen) while silently ignoring `zorblax`. That's the leak the
    // gate exists to close, so chain_weak is no longer required.
    let partial = signals.matched_tokens.len() >= thresholds.partial_leak_min_matched
        && signals.unmatched_tokens.len() >= thresholds.partial_leak_min_unmatched;
    if partial {
        return IdkVerdict::PartialMatchLeak;
    }

    // Mode C — multi-anchor shaky top: many seeds, none strong.
    // Catches `zorblax photosynthesis` on rich brains where junk
    // tokens have keyword-index hits but no anchor reaches a
    // real-concept activation level. NB: a query like `quantum
    // jellyfish dynamics` where ONE token (`quantum`) anchors
    // strongly will NOT fire Mode C — that case requires per-token
    // chain attribution and is beyond Port K's scope. Documented as
    // a known gap.
    //
    // BR.R2 (#989 successor) — well-seeded-across-all-tokens escape:
    // BR.R1 found that Q2 ("Who is Bertrand Russell?") and Q12 ("What
    // is the capital of Mars?") are IDENTICAL on the inputs Mode C
    // consumes — max_activation=0.6000, seed_count=2 — yet Q2 should
    // pass and Q12 should refuse. The discriminating signal that IS
    // available (and is bit-stable, unlike
    // `per_token_neighbor_activated_count` which carries HashMap
    // iteration order — internal notes is `per_token_max_activation`:
    // for Q2 both `bertrand` and `russell` reach the seed-activation
    // floor (0.6) because their top keyword hits are co-located in
    // the same Bertrand Russell neighborhood; for Q12 only `mars`
    // reaches it while `capital` (which pulled an Abu Dhabi junk
    // seed) sits at 0.248. The same property holds across the
    // 12-Q BR.R1 battery and 8 adversarial multi-token IDK queries
    // (including joint-anchor patterns like "Who is Bertrand Mars?"):
    // cross-token activation competition naturally prevents both
    // tokens from each clearing the seed floor when they target
    // unrelated concepts. So if every content token had a keyword-hit
    // that reached seed_activation_floor, the query is well-seeded
    // across all tokens and Mode C's shaky-top signal is
    // overruled — fall through to Mode D and Pass. This is the
    // "promote per_token_max_activation from diagnostic to classify
    // input" candidate named in #989, restricted to the narrowest
    // safe escape rather than a structural rewrite of Mode C.
    let multi_anchor_shaky = signals.seed_count >= thresholds.multi_anchor_min_seeds
        && signals.max_activation < thresholds.multi_anchor_top_floor;
    if multi_anchor_shaky {
        let well_seeded_across_all_tokens = !signals.per_token_max_activation.is_empty()
            && signals
                .per_token_max_activation
                .iter()
                .all(|(_, v)| *v >= thresholds.seed_activation_floor);
        // BR.R3 (OP-008) Mode C escape: when the top ASK seed itself
        // covers a majority of content tokens via the keyword-index
        // buckets (the topical-density property), the recall is
        // grounded enough to fall through to Mode D even if generic
        // tokens (e.g., "born", "philosophers") fail the per-token
        // seed-activation floor. Adversarial preservation comes from
        // the ratio test, NOT an absolute count: A8 "Did Bertrand
        // Russell speak Martian?" has top seed BR with bc=2 of N=4
        // (ratio 0.5, below floor 0.6) so this escape does not fire.
        let top_seeded_topically = signals.content_token_count > 0
            && (signals.top_seed_bucket_coverage as f32)
                / (signals.content_token_count as f32)
                >= thresholds.top_seed_coverage_floor;
        if !well_seeded_across_all_tokens && !top_seeded_topically {
            return IdkVerdict::PartialMatchLeak;
        }
        // well-seeded across every token OR topically-seeded top seed —
        // fall through to Mode D / Pass. (Mode D is gated to
        // coverage_min_tokens=3, so on a 2-token well-seeded query
        // like "Bertrand Russell" we land on Pass; on a 3+ token
        // well-seeded query the dominance check still gets the
        // last word.)
    }

    // Mode D (K.1) — single-anchor dominance. We get here only
    // when Modes A/B/C all decided the recall looked healthy enough
    // to pass: total activation OK, no unmatched tokens, the
    // strongest anchor cleared the seed floor. But on a query like
    // `quantum jellyfish dynamics` only ONE token actually anchors
    // strongly while the others have keyword hits that never
    // activate. The chain Lu would articulate is built almost
    // entirely from the dominant anchor's neighborhood while the
    // other tokens are silently dropped — fabrication. Refuse.
    //
    // Gated on `coverage_min_tokens` (default 3) because for N=2
    // the signal is too noisy: a query like `mitochondria function`
    // legitimately anchors on `mitochondria` alone. Above that, the
    // pattern dominates.
    if signals.is_single_anchor_dominance(thresholds) {
        // BR.R3 (OP-008) Mode D escape: same top-seed topical-density
        // signal Mode C uses. When the top ASK seed has cross-token
        // bucket coverage >= top_seed_coverage_floor, the recall is
        // grounded enough that Mode D's per-token 1-hop anchor count
        // (which is unstable at top_k=5 truncation -- a 274-neighbor
        // article hub only contributes a handful of activated
        // neighbors after the cap) does NOT carry the refusal. The
        // adversarial preservation argument is identical to Mode C's:
        // partial-token-match noise seeds (Charles Einstein, Bertrand
        // Renouvin, List of craters on Mars) keep bc=1 of N, ratio
        // <= 0.5, below the 0.6 floor.
        let top_seeded_topically = signals.content_token_count > 0
            && (signals.top_seed_bucket_coverage as f32)
                / (signals.content_token_count as f32)
                >= thresholds.top_seed_coverage_floor;
        if !top_seeded_topically {
            // Reuse the partial_match_leak verdict so existing callers
            // and structured-payload consumers continue to switch on
            // the same enum; build_response distinguishes the prose by
            // re-checking the dominance condition on the signals.
            return IdkVerdict::PartialMatchLeak;
        }
    }

    IdkVerdict::Pass
}

/// OP-QA-4: does this query solicit a personal preference or opinion?
/// Lu holds none, so declines to such asks say so in plain language
/// rather than exposing recall machinery. Kept deliberately tight so it
/// does not swallow a legitimate "difference between X and Y" query.
pub fn is_preference_query(query: &str) -> bool {
    let q = query.trim().to_lowercase();
    const NEEDLES: &[&str] = &[
        "would you prefer",
        "do you prefer",
        "would you rather",
        "which do you prefer",
        "what do you prefer",
        "your favorite",
        "your favourite",
        "do you have a preference",
        "do you like",
        "which would you pick",
        "which would you choose",
        "what is your opinion",
        "your opinion on",
        "your opinion about",
    ];
    NEEDLES.iter().any(|n| q.contains(n))
}

/// OP-QA-4: the plain-language honest decline. No activation numbers, no
/// thresholds, no "weakly-anchored tokens" jargon. A preference/opinion
/// ask is told plainly that Lu holds none; anything else genuinely
/// ungrounded gets a simple, human "nothing solid" decline.
pub fn plain_decline_for(query: &str) -> String {
    if is_preference_query(query) {
        "I don't hold personal preferences.".to_string()
    } else {
        "I don't have anything solid on that.".to_string()
    }
}

/// Build the IDK refusal text + structured payload. `verdict` must be
/// non-`Pass`; callers should not invoke this on `Pass`.
pub fn build_response(
    verdict: IdkVerdict,
    signals: &IdkSignals,
    raw_query: &str,
    effective_query: &str,
    session_id: Option<u64>,
) -> (String, serde_json::Value) {
    let prose = match verdict {
        IdkVerdict::Pass => {
            // Defensive — Pass should never reach build_response.
            "(no idk verdict)".to_string()
        }
        IdkVerdict::TotalIdk => plain_decline_for(raw_query),
        IdkVerdict::PartialMatchLeak => {
            // Mode D (K.1) — single-anchor dominance. Re-check the
            // condition here so we can produce dominance-specific
            // prose. Distinguished from Mode B by
            // `unmatched_tokens.is_empty()` (every token had a
            // keyword hit) yet coverage was still poor.
            let dominance = signals.unmatched_tokens.is_empty()
                && signals.is_single_anchor_dominance(&IdkThresholds::default())
                && signals.dominant_token.is_some()
                && !signals.coverage_missing_tokens.is_empty();
            if dominance {
                let dom = signals.dominant_token.as_deref().unwrap_or("");
                let missing = quote_join(&signals.coverage_missing_tokens);
                let verb = if signals.coverage_missing_tokens.len() == 1 {
                    "didn't"
                } else {
                    "didn't"
                };
                format!(
                    "I matched strongly on \"{}\" but {} {} activate anything in my brain. I won't fabricate an answer that pretends to cover all of those.",
                    dom, missing, verb,
                )
            } else if signals.unmatched_tokens.is_empty() {
                // Mode C — multi-anchor shaky top. Every token has a
                // keyword-index hit but no anchor activates strongly.
                plain_decline_for(raw_query)
            } else if signals.matched_tokens.len() == 1 && signals.unmatched_tokens.len() == 1 {
                format!(
                    "I recognize \"{}\" but \"{}\" doesn't match anything in my brain. I won't fabricate a connection between them.",
                    signals.matched_tokens[0], signals.unmatched_tokens[0]
                )
            } else {
                format!(
                    "I recognize {} but {} {} match anything in my brain. I won't fabricate a connection between them.",
                    quote_join(&signals.matched_tokens),
                    quote_join(&signals.unmatched_tokens),
                    if signals.unmatched_tokens.len() == 1 { "doesn't" } else { "don't" },
                )
            }
        }
    };

    let mode = match verdict {
        IdkVerdict::TotalIdk => "total_idk",
        IdkVerdict::PartialMatchLeak => "partial_match_leak",
        IdkVerdict::Pass => "pass",
    };
    // K.1 sub-mode tag inside partial_match_leak. Distinguishes B/C
    // from D for downstream consumers (eval harnesses, dashboards).
    let sub_mode = if matches!(verdict, IdkVerdict::PartialMatchLeak) {
        if !signals.unmatched_tokens.is_empty() {
            "unmatched_token_leak"
        } else if signals.is_single_anchor_dominance(&IdkThresholds::default())
            && signals.dominant_token.is_some()
            && !signals.coverage_missing_tokens.is_empty()
        {
            "single_anchor_dominance"
        } else {
            "multi_anchor_shaky_top"
        }
    } else {
        ""
    };
    let confidence_tag = match verdict {
        IdkVerdict::TotalIdk => "below_threshold",
        IdkVerdict::PartialMatchLeak => "partial_match_leak",
        IdkVerdict::Pass => "",
    };

    let coverage_ratio = if signals.content_token_count > 0 {
        signals.anchored_token_count as f32 / signals.content_token_count as f32
    } else {
        0.0_f32
    };

    let structured = serde_json::json!({
        "source": "idk_gate",
        "mode": mode,
        "sub_mode": sub_mode,
        "confidence": confidence_tag,
        "raw_query": raw_query,
        "effective_query": effective_query,
        "session_id": session_id,
        "signals": {
            "total_activation": signals.total_activation,
            "max_seed_activation": signals.max_seed_activation,
            "max_activation": signals.max_activation,
            "seed_count": signals.seed_count,
            "matched_tokens": signals.matched_tokens,
            "unmatched_tokens": signals.unmatched_tokens,
            "chain_no_chain": signals.chain_no_chain,
            "chain_associative_weak": signals.chain_associative_weak,
            "chain_confidence": signals.chain_confidence,
            "per_token_max_activation": signals.per_token_max_activation,
            "per_token_neighbor_activated_count": signals.per_token_neighbor_activated_count,
            "anchored_token_count": signals.anchored_token_count,
            "content_token_count": signals.content_token_count,
            "coverage_ratio": coverage_ratio,
            "dominant_token": signals.dominant_token,
            "coverage_missing_tokens": signals.coverage_missing_tokens,
        },
    });

    (prose, structured)
}

fn quote_join(items: &[String]) -> String {
    let quoted: Vec<String> = items.iter().map(|s| format!("\"{}\"", s)).collect();
    match quoted.len() {
        0 => String::new(),
        1 => quoted[0].clone(),
        2 => format!("{} and {}", quoted[0], quoted[1]),
        _ => {
            let (last, rest) = quoted.split_last().unwrap();
            format!("{}, and {}", rest.join(", "), last)
        }
    }
}

// ====================================================================
// Unit tests
// ====================================================================
#[cfg(test)]
mod tests {
    use super::*;

    fn make_signals(
        total_act: f32,
        max_seed: f32,
        matched: Vec<&str>,
        unmatched: Vec<&str>,
        no_chain: bool,
        weak_assoc: bool,
    ) -> IdkSignals {
        // Default behaviour for legacy tests: every matched token is
        // assumed fully anchored at activation max_seed, every
        // unmatched token has activation 0. This keeps the K (Modes
        // A/B/C) tests untouched while letting K.1 tests use the
        // dedicated `make_signals_dominance` helper below.
        let matched_strs: Vec<String> = matched.iter().map(|s| s.to_string()).collect();
        let unmatched_strs: Vec<String> = unmatched.iter().map(|s| s.to_string()).collect();
        let mut per_tok: Vec<(String, f32)> = Vec::new();
        let mut per_tok_n: Vec<(String, usize)> = Vec::new();
        for t in &matched_strs {
            per_tok.push((t.clone(), max_seed.max(0.6)));
            // Legacy tests assume matched tokens anchor strongly:
            // give them >= floor neighbor count.
            per_tok_n.push((t.clone(), 5));
        }
        for t in &unmatched_strs {
            per_tok.push((t.clone(), 0.0));
            per_tok_n.push((t.clone(), 0));
        }
        let th = IdkThresholds::default();
        let anchored_token_count = per_tok_n
            .iter()
            .filter(|(_, c)| *c >= th.per_token_anchor_count_floor)
            .count();
        let content_token_count = matched_strs.len() + unmatched_strs.len();
        IdkSignals {
            total_activation: total_act,
            max_seed_activation: max_seed,
            max_activation: max_seed.max(total_act / 4.0),
            seed_count: matched_strs.len() + unmatched_strs.len(),
            matched_tokens: matched_strs,
            unmatched_tokens: unmatched_strs,
            chain_no_chain: no_chain,
            chain_associative_weak: weak_assoc,
            chain_confidence: None,
            per_token_max_activation: per_tok,
            per_token_neighbor_activated_count: per_tok_n,
            anchored_token_count,
            content_token_count,
            dominant_token: None,
            coverage_missing_tokens: Vec::new(),
            top_seed_bucket_coverage: 0,
        }
    }

    /// K.1 helper: build IdkSignals with explicit per-token
    /// neighborhood-reach counts so tests can pin down dominance
    /// scenarios precisely. Each `(token, neighbor_count)` pair is
    /// treated as a matched token (keyword hit present); unmatched
    /// tokens go in `unmatched`. Activation is mirrored from the
    /// neighbor count for diagnostic-only purposes.
    fn make_signals_dominance(
        per_token: Vec<(&str, usize)>,
        unmatched: Vec<&str>,
        max_act: f32,
        max_seed: f32,
        no_chain: bool,
    ) -> IdkSignals {
        let th = IdkThresholds::default();
        let mut matched: Vec<String> = Vec::new();
        let mut per_tok_n: Vec<(String, usize)> = Vec::new();
        let mut per_tok_a: Vec<(String, f32)> = Vec::new();
        for (t, c) in &per_token {
            matched.push(t.to_string());
            per_tok_n.push((t.to_string(), *c));
            // Diagnostic activation: scale neighbor count loosely.
            per_tok_a.push((t.to_string(), (*c as f32) * 0.1));
        }
        let unmatched_strs: Vec<String> =
            unmatched.iter().map(|s| s.to_string()).collect();
        for t in &unmatched_strs {
            per_tok_n.push((t.clone(), 0));
            per_tok_a.push((t.clone(), 0.0));
        }
        let anchored_token_count = per_tok_n
            .iter()
            .filter(|(_, c)| *c >= th.per_token_anchor_count_floor)
            .count();
        let content_token_count = matched.len() + unmatched_strs.len();
        let dominant_token = per_tok_n
            .iter()
            .filter(|(_, c)| *c >= th.per_token_anchor_count_floor)
            .max_by_key(|(_, c)| *c)
            .map(|(t, _)| t.clone());
        let coverage_missing_tokens: Vec<String> = per_tok_n
            .iter()
            .filter(|(_, c)| *c < th.per_token_anchor_count_floor)
            .map(|(t, _)| t.clone())
            .collect();
        let total_activation: f32 = per_tok_a.iter().map(|(_, a)| *a).sum::<f32>() * 5.0;
        IdkSignals {
            total_activation,
            max_seed_activation: max_seed,
            max_activation: max_act,
            seed_count: matched.len(),
            matched_tokens: matched,
            unmatched_tokens: unmatched_strs,
            chain_no_chain: no_chain,
            chain_associative_weak: false,
            chain_confidence: None,
            per_token_max_activation: per_tok_a,
            per_token_neighbor_activated_count: per_tok_n,
            anchored_token_count,
            content_token_count,
            dominant_token,
            coverage_missing_tokens,
            top_seed_bucket_coverage: 0,
        }
    }

    #[test]
    fn extract_query_tokens_drops_hub_stopwords() {
        let toks = extract_query_tokens("what is the function of mitochondria");
        // "function" is in the query-noise list (per Bug B context),
        // so it should be dropped at this filter level. "mitochondria"
        // must survive.
        assert!(toks.iter().any(|t| t == "mitochondria"), "toks={:?}", toks);
    }

    #[test]
    fn extract_query_tokens_drops_short_tokens() {
        let toks = extract_query_tokens("a b c xy quantum");
        assert_eq!(toks, vec!["quantum".to_string()]);
    }

    #[test]
    fn extract_query_tokens_handles_punctuation() {
        let toks = extract_query_tokens("flibbergast?!");
        assert_eq!(toks, vec!["flibbergast".to_string()]);
    }

    #[test]
    fn classify_total_idk_fires_when_all_three_signals_low() {
        let s = make_signals(0.5, 0.0, vec![], vec!["flibbergast"], true, false);
        assert_eq!(classify(&s, &IdkThresholds::default()), IdkVerdict::TotalIdk);
    }

    #[test]
    fn classify_total_idk_fires_with_weak_associative_chain() {
        let s = make_signals(0.4, 0.2, vec![], vec!["xqz"], false, true);
        assert_eq!(classify(&s, &IdkThresholds::default()), IdkVerdict::TotalIdk);
    }

    #[test]
    fn classify_does_not_fire_when_total_activation_strong() {
        // Strong activation defeats Mode A even with everything else weak.
        let s = make_signals(50.0, 1.5, vec!["mitochondria"], vec![], false, false);
        assert_eq!(classify(&s, &IdkThresholds::default()), IdkVerdict::Pass);
    }

    #[test]
    fn classify_does_not_fire_when_chain_structured() {
        // Even with low activation, a Structured chain means we have
        // something honest to articulate (Port D-style hedging is fine).
        let s = make_signals(0.5, 0.2, vec![], vec!["obscurething"], false, false);
        assert_eq!(classify(&s, &IdkThresholds::default()), IdkVerdict::Pass);
    }

    #[test]
    fn classify_partial_match_leak_fires() {
        // "zorblax photosynthesis" — photosynthesis matches, zorblax doesn't,
        // and the cross-chain isn't structured.
        let s = make_signals(
            5.0, 1.0,
            vec!["photosynthesis"], vec!["zorblax"],
            true, false,
        );
        assert_eq!(classify(&s, &IdkThresholds::default()), IdkVerdict::PartialMatchLeak);
    }

    #[test]
    fn classify_partial_match_leak_fires_even_with_structured_chain_on_matched_half() {
        // Eval-flagged pattern: matched concept produces a fine
        // chain, but the unmatched token is silently ignored. The
        // gate must still refuse to bridge known and unknown.
        let s = make_signals(
            5.0, 1.0,
            vec!["photosynthesis"], vec!["zorblax"],
            false, false,
        );
        assert_eq!(classify(&s, &IdkThresholds::default()), IdkVerdict::PartialMatchLeak);
    }

    #[test]
    fn classify_total_does_not_fire_if_a_token_matched() {
        // matched_tokens non-empty disqualifies Mode A; falls through
        // to Mode B partial-leak detection.
        let s = make_signals(
            0.5, 0.2,
            vec!["photosynthesis"], vec!["zorblax"],
            true, false,
        );
        assert_eq!(classify(&s, &IdkThresholds::default()), IdkVerdict::PartialMatchLeak);
    }

    #[test]
    fn classify_total_does_not_fire_if_seed_activation_above_floor() {
        let s = make_signals(0.5, 0.8, vec![], vec!["xqz"], true, false);
        // max_seed_activation 0.8 >= 0.6 floor — fails Mode A.
        // matched empty, unmatched 1 — but matched=0 fails Mode B.
        assert_eq!(classify(&s, &IdkThresholds::default()), IdkVerdict::Pass);
    }

    #[test]
    fn classify_threshold_overrides_work() {
        // Tighten activation floor to 10.0 so a normal-but-weak query
        // would fire Mode A under the override.
        let mut th = IdkThresholds::default();
        th.activation_floor = 10.0;
        let s = make_signals(2.0, 0.2, vec![], vec!["obscure"], true, false);
        assert_eq!(classify(&s, &th), IdkVerdict::TotalIdk);
    }

    #[test]
    fn op_qa4_preference_and_plain_decline_wording() {
        assert!(is_preference_query("would you prefer a dog or a cat?"));
        assert!(is_preference_query("Do you prefer tea or coffee"));
        assert!(!is_preference_query("what is the difference between a dog and a cat"));
        let pref = plain_decline_for("would you prefer a dog or a cat?");
        assert_eq!(pref, "I don't hold personal preferences.");
        let ungrounded = plain_decline_for("quantum jellyfish dynamics");
        assert_eq!(ungrounded, "I don't have anything solid on that.");
        // No machinery must leak in either.
        for s in [&pref, &ungrounded] {
            assert!(!s.contains("activat"), "{}", s);
            assert!(!s.chars().any(|c| c.is_ascii_digit()), "{}", s);
            assert!(!s.contains("anchor"), "{}", s);
            assert!(!s.contains("threshold"), "{}", s);
        }
    }

    #[test]
    fn build_response_total_idk_explains_with_top_match_when_present() {
        let s = make_signals(0.4, 0.30, vec![], vec!["flibbergast"], true, false);
        let (prose, structured) = build_response(
            IdkVerdict::TotalIdk, &s, "flibbergast", "flibbergast", None,
        );
        // OP-QA-4: humanized decline -- plain language, no activation number.
        assert!(prose.contains("anything solid"), "prose={:?}", prose);
        assert!(!prose.contains("0.300"), "machinery number must not leak: {:?}", prose);
        assert_eq!(structured["source"], "idk_gate");
        assert_eq!(structured["mode"], "total_idk");
    }

    #[test]
    fn build_response_total_idk_when_no_seed_activation() {
        let s = make_signals(0.0, 0.0, vec![], vec!["xqzblr"], true, false);
        let (prose, _) = build_response(
            IdkVerdict::TotalIdk, &s, "xqzblr", "xqzblr", None,
        );
        // OP-QA-4: humanized decline for the zero-activation case too.
        assert!(prose.contains("anything solid"), "prose={:?}", prose);
    }

    #[test]
    fn build_response_partial_match_leak_quotes_both_sides() {
        let s = make_signals(
            5.0, 1.0,
            vec!["photosynthesis"], vec!["zorblax"],
            true, false,
        );
        let (prose, structured) = build_response(
            IdkVerdict::PartialMatchLeak, &s,
            "zorblax photosynthesis", "zorblax photosynthesis", None,
        );
        assert!(prose.contains("recognize \"photosynthesis\""));
        assert!(prose.contains("\"zorblax\""));
        assert!(prose.contains("won't fabricate"));
        assert_eq!(structured["mode"], "partial_match_leak");
    }

    #[test]
    fn build_response_partial_match_leak_handles_multiple_unmatched() {
        let s = make_signals(
            5.0, 1.0,
            vec!["dynamics"], vec!["quantum", "jellyfish"],
            true, false,
        );
        let (prose, _) = build_response(
            IdkVerdict::PartialMatchLeak, &s,
            "quantum jellyfish dynamics", "quantum jellyfish dynamics", None,
        );
        assert!(prose.contains("\"quantum\""));
        assert!(prose.contains("\"jellyfish\""));
    }

    #[test]
    fn build_response_session_id_propagates() {
        let s = make_signals(0.4, 0.0, vec![], vec!["xqz"], true, false);
        let (_, structured) = build_response(
            IdkVerdict::TotalIdk, &s, "xqz", "xqz", Some(42),
        );
        assert_eq!(structured["session_id"], 42);
    }

    #[test]
    fn quote_join_handles_zero_one_two_many() {
        assert_eq!(quote_join(&[]), "");
        assert_eq!(quote_join(&["a".into()]), "\"a\"");
        assert_eq!(quote_join(&["a".into(), "b".into()]), "\"a\" and \"b\"");
        assert_eq!(
            quote_join(&["a".into(), "b".into(), "c".into()]),
            "\"a\", \"b\", and \"c\"",
        );
    }

    #[test]
    fn idk_thresholds_defaults_match_spec() {
        let th = IdkThresholds::default();
        assert!((th.activation_floor - 1.5).abs() < 1e-6);
        assert!((th.mixed_confidence_floor - 0.4).abs() < 1e-6);
        assert!((th.seed_activation_floor - 0.6).abs() < 1e-6);
        assert_eq!(th.partial_leak_min_unmatched, 1);
        assert_eq!(th.partial_leak_min_matched, 1);
    }

    // ================================================================
    // K.1 — Signal 4: anchor coverage / single-anchor dominance.
    // ================================================================

    #[test]
    fn k1_thresholds_defaults_match_spec() {
        let th = IdkThresholds::default();
        assert_eq!(th.per_token_anchor_count_floor, 2);
        assert!((th.per_token_activation_floor - 0.30).abs() < 1e-6);
        assert!((th.coverage_floor - 0.5).abs() < 1e-6);
        assert_eq!(th.coverage_min_tokens, 3);
    }

    #[test]
    fn k1_dominance_fires_on_quantum_jellyfish_dynamics() {
        // Eval-flagged shape: one strong anchor, two empty hits.
        let s = make_signals_dominance(
            vec![("quantum", 7), ("jellyfish", 0), ("dynamics", 1)],
            vec![],
            0.866, 0.866, false,
        );
        assert!(s.is_single_anchor_dominance(&IdkThresholds::default()));
        assert_eq!(
            classify(&s, &IdkThresholds::default()),
            IdkVerdict::PartialMatchLeak,
        );
        assert_eq!(s.dominant_token.as_deref(), Some("quantum"));
        assert_eq!(s.coverage_missing_tokens, vec!["jellyfish".to_string(), "dynamics".to_string()]);
    }

    #[test]
    fn k1_dominance_fires_on_python_recipe_pasta() {
        // Two strong anchors out of three is still below 0.5
        // coverage — the third token having a keyword hit but no
        // activation means we'd silently drop it. Wait: 2/3 = 0.667
        // > 0.5, so this should NOT fire. Pin that down.
        let s = make_signals_dominance(
            vec![("python", 7), ("recipe", 5), ("pasta", 0)],
            vec![],
            0.85, 0.85, false,
        );
        assert_eq!(s.anchored_token_count, 2);
        assert_eq!(s.content_token_count, 3);
        // 2/3 >= 0.5, so dominance does NOT fire. The query should
        // pass K.1 even though one token is weak.
        assert!(!s.is_single_anchor_dominance(&IdkThresholds::default()));
        assert_eq!(
            classify(&s, &IdkThresholds::default()),
            IdkVerdict::Pass,
        );
    }

    #[test]
    fn k1_dominance_fires_on_napoleon_photosynthesis_algorithm() {
        // Three real concepts but no chain: only one anchored at the
        // floor (the dominant token). Coverage 1/3 < 0.5 — fire.
        let s = make_signals_dominance(
            vec![
                ("napoleon", 7),
                ("photosynthesis", 0),
                ("algorithm", 0),
            ],
            vec![],
            0.80, 0.80, false,
        );
        assert!(s.is_single_anchor_dominance(&IdkThresholds::default()));
        assert_eq!(
            classify(&s, &IdkThresholds::default()),
            IdkVerdict::PartialMatchLeak,
        );
    }

    #[test]
    fn k1_does_not_fire_on_n_equals_2() {
        // Two-token query with one weak anchor MUST pass K.1 even
        // when coverage is 1/2 = 0.5. Mode B/C handle the bad cases
        // for N=2 already; K.1 staying out avoids regressions on
        // legitimate weak-anchor pairs (e.g. "mitochondria function"
        // where `function` is a query-noise stop and gets dropped
        // before this even runs).
        let s = make_signals_dominance(
            vec![("quantum", 7), ("jellyfish", 0)],
            vec![],
            0.85, 0.85, false,
        );
        assert!(!s.is_single_anchor_dominance(&IdkThresholds::default()));
        // Note: classify will still return Pass here because
        // unmatched_tokens is empty, max_activation 0.85 > 0.70
        // multi_anchor_top_floor, etc.
        assert_eq!(
            classify(&s, &IdkThresholds::default()),
            IdkVerdict::Pass,
        );
    }

    #[test]
    fn k1_does_not_fire_on_paris_france_capital() {
        // All three tokens anchor strongly — must pass.
        let s = make_signals_dominance(
            vec![("paris", 9), ("france", 3), ("capital", 3)],
            vec![],
            0.85, 0.85, false,
        );
        assert!(!s.is_single_anchor_dominance(&IdkThresholds::default()));
        assert_eq!(
            classify(&s, &IdkThresholds::default()),
            IdkVerdict::Pass,
        );
    }

    #[test]
    fn k1_does_not_fire_on_mitochondria_function_cell() {
        // Three real biology tokens. All anchor.
        let s = make_signals_dominance(
            vec![
                ("mitochondria", 6),
                ("function", 5),
                ("cell", 6),
            ],
            vec![],
            0.85, 0.85, false,
        );
        assert_eq!(s.anchored_token_count, 3);
        assert!(!s.is_single_anchor_dominance(&IdkThresholds::default()));
        assert_eq!(
            classify(&s, &IdkThresholds::default()),
            IdkVerdict::Pass,
        );
    }

    #[test]
    fn k1_does_not_fire_on_single_token_query() {
        // N=1 — trivially anchored. Must pass.
        let s = make_signals_dominance(
            vec![("mitochondria", 9)],
            vec![],
            0.85, 0.85, false,
        );
        assert!(!s.is_single_anchor_dominance(&IdkThresholds::default()));
        assert_eq!(
            classify(&s, &IdkThresholds::default()),
            IdkVerdict::Pass,
        );
    }

    #[test]
    fn k1_existing_modes_take_priority_over_dominance() {
        // Mode B (unmatched-token leak) should fire BEFORE Mode D
        // even when the dominance pattern would also match. Verify
        // the verdict is PartialMatchLeak but that build_response
        // routes to Mode B prose, not Mode D.
        let s = make_signals_dominance(
            vec![("quantum", 7), ("physics", 0)],
            vec!["flibbergast"],
            0.85, 0.85, true,
        );
        // Expect classify to return PartialMatchLeak via Mode B,
        // since unmatched_tokens is non-empty and matched is
        // non-empty — Mode B fires first.
        assert_eq!(
            classify(&s, &IdkThresholds::default()),
            IdkVerdict::PartialMatchLeak,
        );
        let (prose, structured) = build_response(
            IdkVerdict::PartialMatchLeak, &s,
            "quantum physics flibbergast", "quantum physics flibbergast", None,
        );
        assert!(prose.contains("\"flibbergast\""));
        // The sub_mode should be unmatched_token_leak, NOT
        // single_anchor_dominance, because unmatched_tokens is
        // non-empty.
        assert_eq!(structured["sub_mode"], "unmatched_token_leak");
    }

    #[test]
    fn k1_threshold_overrides_per_token_anchor_count_floor() {
        // Tighten the per-token anchor count floor to 5 — a
        // previously "anchored" token at neighbor_count 3 now
        // falls below.
        let mut th = IdkThresholds::default();
        th.per_token_anchor_count_floor = 5;
        let mut s = make_signals_dominance(
            vec![("alpha", 9), ("beta", 3), ("gamma", 3)],
            vec![],
            0.95, 0.95, false,
        );
        // Recompute under the override.
        let anchored = s
            .per_token_neighbor_activated_count
            .iter()
            .filter(|(_, c)| *c >= th.per_token_anchor_count_floor)
            .count();
        s.anchored_token_count = anchored;
        assert_eq!(anchored, 1);
        assert!(s.is_single_anchor_dominance(&th));
    }

    #[test]
    fn k1_build_response_dominance_prose_names_dominant_and_missing() {
        let s = make_signals_dominance(
            vec![("quantum", 7), ("jellyfish", 0), ("dynamics", 1)],
            vec![],
            0.866, 0.866, false,
        );
        let (prose, structured) = build_response(
            IdkVerdict::PartialMatchLeak, &s,
            "quantum jellyfish dynamics", "quantum jellyfish dynamics", None,
        );
        assert!(prose.contains("matched strongly on \"quantum\""), "prose={:?}", prose);
        assert!(prose.contains("\"jellyfish\""));
        assert!(prose.contains("\"dynamics\""));
        assert!(prose.contains("won't fabricate"));
        assert_eq!(structured["sub_mode"], "single_anchor_dominance");
        assert_eq!(structured["signals"]["anchored_token_count"], 1);
        assert_eq!(structured["signals"]["content_token_count"], 3);
    }

    #[test]
    fn k1_structured_payload_exposes_per_token_activations() {
        let s = make_signals_dominance(
            vec![("quantum", 7), ("jellyfish", 0), ("dynamics", 1)],
            vec![],
            0.866, 0.866, false,
        );
        let (_, structured) = build_response(
            IdkVerdict::PartialMatchLeak, &s,
            "quantum jellyfish dynamics", "quantum jellyfish dynamics", None,
        );
        let per_token = &structured["signals"]["per_token_max_activation"];
        assert!(per_token.is_array());
        assert_eq!(per_token.as_array().unwrap().len(), 3);
        let per_nbr = &structured["signals"]["per_token_neighbor_activated_count"];
        assert!(per_nbr.is_array());
        assert_eq!(per_nbr.as_array().unwrap().len(), 3);
        let dominant = &structured["signals"]["dominant_token"];
        assert_eq!(dominant, "quantum");
        let missing = &structured["signals"]["coverage_missing_tokens"];
        assert_eq!(missing.as_array().unwrap().len(), 2);
    }
}
