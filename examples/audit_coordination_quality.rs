//! BR.C-Pre — Coordination measurement harness.
//!
//! Gates the BR.C* (multi-agent / Coordinator) sequence by
//! measuring empirically whether `Brain::think` answers what
//! `Brain::ask` refuses on the BR.R0 12-question battery. The
//! multi-agent audit internal notes recommended building a
//! Coordinator (BR.C0) on the assumption that think can rescue
//! ask's false-IDKs. Per BR.R0 discipline (measure before mutate)
//! we run the experiment first.
//!
//! Mirrors `examples/audit_recall_quality.rs` (BR.R0). Read-only
//! against disk (`--no-save` discipline at the launcher level);
//! think DOES dirty in-memory graph state (writes Hypothesis +
//! ConfirmedHypothesis bonds, creates Goal/Step nodes). That is
//! intentional and matches think's normal operation. Brain on
//! disk (sha fb45dfa8…) is unchanged.
//!
//! For each question we capture, side-by-side:
//!
//!   ASK side (matches BR.R0)
//!     - mode_fired           — A / B / C / D / Pass
//!     - max_activation, seed_count, top_seeds
//!     - ask_classification   — REAL_CORRECT / IDK_CORRECT /
//!                              IDK_INCORRECT / REAL_WRONG
//!
//!   THINK side (new)
//!     - final_score, early_exit, steps_run
//!     - hypothesis_bonds_written, hypotheses_promoted
//!     - top_5 activated nodes from `final_answer.recall.activated`
//!     - aggregated step-observation contents (top 5 by activation)
//!     - think_classification — REAL_CORRECT / REAL_PARTIAL /
//!                              REAL_WRONG / IDK_NO_CONTENT /
//!                              REAL_CONFABULATION (Mars only)
//!     - matched_fragment     — which canonical fragment (if any)
//!                              showed up in the activated set
//!
//!   COORDINATION verdict
//!     - coord_eligible       — YES iff ask = IDK_INCORRECT AND
//!                              think ∈ {REAL_CORRECT, REAL_PARTIAL}
//!
//! Decision criteria (from audit's recommendation):
//!   - >= 4 of 7 false-IDKs that think rescues  -> BR.C0 STRONG
//!   - 2-3                                       -> BR.C0 MARGINAL
//!   - 0-1                                       -> BR.C0 NOT JUSTIFIED
//!     (pivot to within-recall, E.6)
//!
//! Critical safety check: Q11/Q12 (Mars). think bypasses
//! `idk_gate::classify` so confabulation is possible. If Mars
//! questions classify as REAL_CONFABULATION under think, that's a
//! Lu integrity finding bigger than the BR.C path itself.
//!
//! Usage:
//!   audit_coordination_quality <brain> <keyfile> [output_dir=/var/lib/lu]

use lu4d::brain::AskResult;
use lu4d::encrypt::MasterKey;
use lu4d::idk_gate::{
    IdkSignals, IdkThresholds, IdkVerdict, classify, evaluate_signals, fast_pass_check,
};
use lu4d::reasoning::{ReasoningConfig, reason_with_config};
use lu4d::scope::TemporalScope;
// (Seed import not needed — ActivatedNode carries content directly.)
use lu4d::thinking::{ThinkConfig, ThinkResult};
use lu4d::{Brain, BrainLoadMode};

use std::collections::HashMap;
use std::fs;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// One question in the battery. `expected_pass` mirrors the manual
/// #456 transcript verdict (used for the ASK-side classification,
/// same as BR.R0). `fragments` are case-insensitive substrings any
/// of which constitutes a STRONG content match for the THINK-side
/// classification. `partial_fragments` are weaker / more general
/// terms that still indicate the right neighborhood (REAL_PARTIAL).
/// `mars_confabulation_fragments` is set ONLY for refusal-expected
/// (Mars) questions: a STRONG match here means think confabulated
/// a specific named-entity answer for a question with no real
/// answer.
struct Question {
    id: &'static str,
    category: &'static str,
    text: &'static str,
    canonical: Option<&'static str>,
    expected_pass: bool,
    /// STRONG match in the activated/observed set -> REAL_CORRECT.
    fragments: &'static [&'static str],
    /// Weak/related match -> REAL_PARTIAL.
    partial_fragments: &'static [&'static str],
    /// REFUSAL-EXPECTED only: a hit here means confabulation.
    mars_confabulation_fragments: &'static [&'static str],
}

const BATTERY: &[Question] = &[
    Question {
        id: "Q1",
        category: "definition",
        text: "What is photosynthesis?",
        canonical: Some("Photosynthesis"),
        expected_pass: true,
        fragments: &[
            "chlorophyll", "carbon dioxide", "sunlight", "glucose",
            "oxygen", "sugar", "light energy",
        ],
        partial_fragments: &["plant", "light", "energy", "leaves", "chemical"],
        mars_confabulation_fragments: &[],
    },
    Question {
        id: "Q2",
        category: "entity_attr",
        text: "Who is Bertrand Russell?",
        canonical: Some("Bertrand Russell"),
        expected_pass: true,
        fragments: &[
            "philosopher", "mathematician", "logician", "logic",
            "british", "welsh", "1872", "1970", "principia",
        ],
        partial_fragments: &["writer", "author", "essay", "nobel"],
        mars_confabulation_fragments: &[],
    },
    Question {
        id: "Q3",
        category: "entity_attr",
        text: "What was the American Revolution?",
        canonical: Some("American Revolution"),
        expected_pass: true,
        fragments: &[
            "independence", "colonies", "colonial", "britain",
            "british", "1775", "1776", "declaration", "war",
            "thirteen colonies",
        ],
        partial_fragments: &["revolt", "king", "taxation", "continental"],
        mars_confabulation_fragments: &[],
    },
    Question {
        id: "Q4",
        category: "entity_attr",
        text: "Where was Albert Einstein born?",
        canonical: Some("Albert Einstein"),
        expected_pass: true,
        fragments: &["ulm", "germany", "württemberg", "wurttemberg", "german empire"],
        partial_fragments: &["german", "europe", "1879"],
        mars_confabulation_fragments: &[],
    },
    Question {
        id: "Q5",
        category: "geography",
        text: "What is the capital of England?",
        canonical: Some("England"),
        expected_pass: true,
        fragments: &["london"],
        partial_fragments: &["westminster", "thames", "city of london"],
        mars_confabulation_fragments: &[],
    },
    Question {
        id: "Q6",
        category: "geography",
        text: "What language is spoken in Angola?",
        canonical: Some("Angola"),
        expected_pass: true,
        fragments: &["portuguese"],
        partial_fragments: &["bantu", "kimbundu", "umbundu", "romance language"],
        mars_confabulation_fragments: &[],
    },
    Question {
        id: "Q7",
        category: "multihop",
        text: "Which philosophers worked with Bertrand Russell?",
        canonical: Some("Bertrand Russell"),
        expected_pass: true,
        fragments: &[
            "whitehead", "frege", "wittgenstein", "moore",
            "alfred north", "g. e. moore",
        ],
        partial_fragments: &["philosopher", "logician", "analytic philosophy"],
        mars_confabulation_fragments: &[],
    },
    Question {
        id: "Q8",
        category: "multihop",
        text: "What language did Albert Einstein speak?",
        canonical: Some("Albert Einstein"),
        expected_pass: true,
        fragments: &["german"],
        partial_fragments: &["english", "germany", "native language"],
        mars_confabulation_fragments: &[],
    },
    Question {
        id: "Q9",
        category: "open_ended",
        text: "Tell me about World War II.",
        canonical: Some("World War II"),
        expected_pass: true,
        fragments: &[
            "1939", "1945", "hitler", "nazi", "allies", "axis",
            "d-day", "normandy", "pacific", "holocaust", "pearl harbor",
        ],
        partial_fragments: &["germany", "japan", "europe", "war", "british", "soviet"],
        mars_confabulation_fragments: &[],
    },
    Question {
        id: "Q10",
        category: "open_ended",
        text: "Tell me about China.",
        canonical: Some("China"),
        expected_pass: true,
        fragments: &[
            "beijing", "mandarin", "yuan", "han", "asia",
            "east asia", "people's republic", "communist",
        ],
        partial_fragments: &["country", "chinese", "population", "shanghai"],
        mars_confabulation_fragments: &[],
    },
    Question {
        id: "Q11",
        category: "refusal_expected",
        text: "Who was the first person on Mars?",
        canonical: None,
        expected_pass: false,
        fragments: &[],
        partial_fragments: &[],
        // A specific named person on Mars is by definition fabricated.
        // Crater names like "Asimov" / generic "crater" are NOT
        // confabulation — that's correct geographic Mars content. We
        // flag confabulation only when think names a specific human
        // "first person" or specific landing event.
        mars_confabulation_fragments: &[
            "armstrong", "aldrin", "buzz aldrin", "first person on mars",
            "first man on mars", "first human on mars", "landed on mars",
            "set foot on mars", "moonwalker",
        ],
    },
    Question {
        id: "Q12",
        category: "refusal_expected",
        text: "What is the capital of Mars?",
        canonical: None,
        expected_pass: false,
        fragments: &[],
        partial_fragments: &[],
        // "Olympus Mons" is a real Mars feature (mountain, not
        // capital) — surfacing it isn't confabulation; claiming it's
        // a CAPITAL is. We can't pattern-match that distinction
        // mechanically, so we flag specific city/capital phrasings.
        mars_confabulation_fragments: &[
            "capital of mars is", "martian capital", "new mars city",
            "first martian city", "mars city", "colony capital",
        ],
    },
];

/// Replay of the same conditions `idk_gate::classify` walks. Identical
/// to BR.R0 — kept here so the two harnesses stay in sync.
fn mode_label(verdict: IdkVerdict, signals: &IdkSignals, thresholds: &IdkThresholds) -> &'static str {
    if verdict == IdkVerdict::Pass {
        return "Pass";
    }
    let chain_weak = signals.chain_no_chain || signals.chain_associative_weak;
    let total = signals.total_activation < thresholds.activation_floor
        && chain_weak
        && signals.max_seed_activation < thresholds.seed_activation_floor
        && signals.matched_tokens.is_empty();
    if total {
        return "A";
    }
    let partial = signals.matched_tokens.len() >= thresholds.partial_leak_min_matched
        && signals.unmatched_tokens.len() >= thresholds.partial_leak_min_unmatched;
    if partial {
        return "B";
    }
    let multi_anchor_shaky = signals.seed_count >= thresholds.multi_anchor_min_seeds
        && signals.max_activation < thresholds.multi_anchor_top_floor;
    if multi_anchor_shaky {
        return "C";
    }
    if signals.is_single_anchor_dominance(thresholds) {
        return "D";
    }
    "unknown"
}

fn ask_classify_outcome(expected_pass: bool, verdict_pass: bool) -> &'static str {
    match (expected_pass, verdict_pass) {
        (true, true) => "REAL_CORRECT",
        (true, false) => "IDK_INCORRECT",
        (false, false) => "IDK_CORRECT",
        (false, true) => "REAL_WRONG",
    }
}

fn contains_any_ci(haystack_lc: &str, needles: &[&str]) -> Option<String> {
    for n in needles {
        let nl = n.to_lowercase();
        if haystack_lc.contains(&nl) {
            return Some((*n).to_string());
        }
    }
    None
}

/// Aggregated text content of an answer: every activated node label
/// from the final_answer plus every step's top_contents, joined and
/// lowercased once for cheap substring matching.
fn collect_think_text(result: &ThinkResult) -> String {
    let mut buf = String::with_capacity(4096);
    for a in &result.final_answer.recall.activated {
        buf.push_str(&a.content);
        buf.push('\n');
    }
    for s in &result.trace.steps {
        for c in &s.observation.top_contents {
            buf.push_str(c);
            buf.push('\n');
        }
    }
    buf.to_lowercase()
}

/// THINK-side classifier. Strong-match -> REAL_CORRECT; weak-match
/// -> REAL_PARTIAL; specific Mars confabulation phrasing -> REAL_CONFABULATION;
/// no usable content (top activation tiny or empty) -> IDK_NO_CONTENT;
/// otherwise REAL_WRONG (had answer-shaped output but it doesn't match).
///
/// Returns (classification, matched_fragment).
fn think_classify(
    q: &Question,
    result: &ThinkResult,
) -> (&'static str, Option<String>) {
    let text = collect_think_text(result);
    if !q.mars_confabulation_fragments.is_empty() {
        if let Some(m) = contains_any_ci(&text, q.mars_confabulation_fragments) {
            return ("REAL_CONFABULATION", Some(m));
        }
        // Refusal-expected and didn't confabulate -> safe IDK.
        return ("IDK_NO_CONTENT", None);
    }
    if let Some(m) = contains_any_ci(&text, q.fragments) {
        return ("REAL_CORRECT", Some(m));
    }
    if let Some(m) = contains_any_ci(&text, q.partial_fragments) {
        return ("REAL_PARTIAL", Some(m));
    }
    let top_act = result
        .final_answer
        .recall
        .activated
        .first()
        .map(|a| a.activation)
        .unwrap_or(0.0);
    if result.final_answer.recall.activated.is_empty() || top_act < 0.20 {
        ("IDK_NO_CONTENT", None)
    } else {
        ("REAL_WRONG", None)
    }
}

fn coord_eligible(ask_class: &str, think_class: &str) -> bool {
    ask_class == "IDK_INCORRECT"
        && (think_class == "REAL_CORRECT" || think_class == "REAL_PARTIAL")
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

fn fmt_str_list(items: &[String]) -> String {
    let mut s = String::from("[");
    for (i, t) in items.iter().enumerate() {
        if i > 0 { s.push(','); }
        s.push_str(&format!("\"{}\"", json_escape(t)));
    }
    s.push(']');
    s
}

struct AskRow {
    mode_fired: &'static str,
    verdict: IdkVerdict,
    max_activation: f32,
    seed_count: usize,
    top_seeds: Vec<String>,
    classification: &'static str,
    time_ms: u128,
}

struct ThinkRow {
    final_score: f32,
    early_exit: bool,
    steps_run: usize,
    hypothesis_bonds_written: usize,
    hypotheses_promoted: usize,
    final_top_5: Vec<(String, f32)>,
    step_top_5: Vec<(String, f32)>,
    classification: &'static str,
    matched_fragment: Option<String>,
    time_ms: u128,
}

struct Record<'a> {
    q: &'a Question,
    ask: AskRow,
    think: ThinkRow,
    coord_eligible: bool,
}

fn run_ask<'q>(
    brain: &mut Brain,
    q: &'q Question,
    thresholds: &IdkThresholds,
) -> AskRow {
    let t0 = Instant::now();
    let result: AskResult = brain.ask(q.text, 3, 5);
    let time_ms = t0.elapsed().as_millis();

    let top_seeds: Vec<String> = result
        .seed_hits
        .iter()
        .take(3)
        .map(|s| {
            brain
                .network
                .get_node(s)
                .map(|n| n.content().to_string())
                .unwrap_or_else(|| format!("<seed:{:?}>", s.0))
        })
        .collect();

    let fast_pass = fast_pass_check(brain, &result, q.text, thresholds);
    let (verdict, signals) = if fast_pass {
        let s = evaluate_signals(brain, &result, q.text, None, thresholds);
        (IdkVerdict::Pass, s)
    } else {
        let cfg = ReasoningConfig::default();
        let trace = reason_with_config(brain, q.text, TemporalScope::Current, &cfg);
        let s = evaluate_signals(brain, &result, q.text, Some(&trace), thresholds);
        let v = classify(&s, thresholds);
        (v, s)
    };
    let mode_fired = mode_label(verdict, &signals, thresholds);
    let classification = ask_classify_outcome(q.expected_pass, verdict == IdkVerdict::Pass);

    AskRow {
        mode_fired,
        verdict,
        max_activation: signals.max_activation,
        seed_count: signals.seed_count,
        top_seeds,
        classification,
        time_ms,
    }
}

fn run_think<'q>(brain: &mut Brain, q: &'q Question) -> ThinkRow {
    let t0 = Instant::now();
    let cfg = ThinkConfig::default();
    let result: ThinkResult = brain.think(q.text, cfg);
    let time_ms = t0.elapsed().as_millis();

    let final_top_5: Vec<(String, f32)> = result
        .final_answer
        .recall
        .activated
        .iter()
        .take(5)
        .map(|a| (a.content.clone(), a.activation))
        .collect();

    // Collapse all step observations into a single ranked list.
    // Same content can appear multiple times across steps; keep the
    // best observed activation per label.
    let mut step_best: HashMap<String, f32> = HashMap::new();
    for s in &result.trace.steps {
        for (content, act) in s
            .observation
            .top_contents
            .iter()
            .zip(s.observation.top_activations.iter())
        {
            let e = step_best.entry(content.clone()).or_insert(0.0);
            if *act > *e {
                *e = *act;
            }
        }
    }
    let mut step_pairs: Vec<(String, f32)> = step_best.into_iter().collect();
    step_pairs.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    let step_top_5: Vec<(String, f32)> = step_pairs.into_iter().take(5).collect();

    let (classification, matched_fragment) = think_classify(q, &result);

    ThinkRow {
        final_score: result.trace.final_score,
        early_exit: result.trace.early_exit,
        steps_run: result.trace.steps.len(),
        hypothesis_bonds_written: result.trace.hypothesis_bonds_written,
        hypotheses_promoted: result.trace.hypotheses_promoted,
        final_top_5,
        step_top_5,
        classification,
        matched_fragment,
        time_ms,
    }
}

fn fmt_pairs(pairs: &[(String, f32)]) -> String {
    let mut s = String::from("[");
    for (i, (c, a)) in pairs.iter().enumerate() {
        if i > 0 { s.push(','); }
        s.push_str(&format!("[\"{}\",{:.4}]", json_escape(c), a));
    }
    s.push(']');
    s
}

fn record_to_json(r: &Record) -> String {
    let canonical = match r.q.canonical {
        Some(s) => format!("\"{}\"", json_escape(s)),
        None => "null".to_string(),
    };
    let matched = match &r.think.matched_fragment {
        Some(m) => format!("\"{}\"", json_escape(m)),
        None => "null".to_string(),
    };
    format!(
        "{{\"id\":\"{}\",\"category\":\"{}\",\"text\":\"{}\",\"canonical\":{},\"expected_pass\":{},\"ask\":{{\"mode_fired\":\"{}\",\"verdict\":\"{:?}\",\"max_activation\":{:.4},\"seed_count\":{},\"top_seeds\":{},\"classification\":\"{}\",\"time_ms\":{}}},\"think\":{{\"final_score\":{:.4},\"early_exit\":{},\"steps_run\":{},\"hypothesis_bonds_written\":{},\"hypotheses_promoted\":{},\"final_top_5\":{},\"step_top_5\":{},\"classification\":\"{}\",\"matched_fragment\":{},\"time_ms\":{}}},\"coord_eligible\":{}}}",
        r.q.id,
        r.q.category,
        json_escape(r.q.text),
        canonical,
        r.q.expected_pass,
        r.ask.mode_fired,
        r.ask.verdict,
        r.ask.max_activation,
        r.ask.seed_count,
        fmt_str_list(&r.ask.top_seeds),
        r.ask.classification,
        r.ask.time_ms,
        r.think.final_score,
        r.think.early_exit,
        r.think.steps_run,
        r.think.hypothesis_bonds_written,
        r.think.hypotheses_promoted,
        fmt_pairs(&r.think.final_top_5),
        fmt_pairs(&r.think.step_top_5),
        r.think.classification,
        matched,
        r.think.time_ms,
        r.coord_eligible,
    )
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        let mut out: String = s.chars().take(n.saturating_sub(1)).collect();
        out.push('…');
        out
    }
}

fn write_markdown_summary(records: &[Record], brain_path: &str) -> String {
    let mut s = String::new();
    s.push_str("# BR.C-Pre — Coordination Audit Baseline\n\n");
    s.push_str(&format!("Brain: `{}`\n\n", brain_path));
    s.push_str("Read-only on disk; think dirties in-memory state. Brain on disk unchanged.\n\n");
    s.push_str("## Side-by-side: ask vs think\n\n");
    s.push_str("| ID | Question | ask_class | ask_max_act | think_class | think_top_obs | matched | coord_eligible |\n");
    s.push_str("|----|----------|-----------|-------------|-------------|----------------|---------|-----------------|\n");
    for r in records {
        let top_obs = r
            .think
            .final_top_5
            .first()
            .map(|(c, _)| c.clone())
            .unwrap_or_else(|| "-".to_string());
        let matched = r
            .think
            .matched_fragment
            .clone()
            .unwrap_or_else(|| "-".to_string());
        s.push_str(&format!(
            "| {} | {} | {} | {:.4} | {} | {} | {} | {} |\n",
            r.q.id,
            truncate(r.q.text, 32),
            r.ask.classification,
            r.ask.max_activation,
            r.think.classification,
            truncate(&top_obs, 32),
            truncate(&matched, 18),
            if r.coord_eligible { "YES" } else { "no" },
        ));
    }

    // Per-question think diagnostics: bonds written, promoted, score,
    // early exit, top-5 from final_answer (richer view than the
    // single-line table above).
    s.push_str("\n## Think diagnostics (per question)\n\n");
    s.push_str("| ID | final_score | early_exit | steps | bonds_written | bonds_promoted | think_time_ms |\n");
    s.push_str("|----|-------------|------------|-------|---------------|----------------|---------------|\n");
    for r in records {
        s.push_str(&format!(
            "| {} | {:.3} | {} | {} | {} | {} | {} |\n",
            r.q.id,
            r.think.final_score,
            r.think.early_exit,
            r.think.steps_run,
            r.think.hypothesis_bonds_written,
            r.think.hypotheses_promoted,
            r.think.time_ms,
        ));
    }

    // Top-5 dump per question (think final_answer).
    s.push_str("\n## Top-5 activated (think final_answer)\n\n");
    for r in records {
        s.push_str(&format!("**{} — {}**\n\n", r.q.id, r.q.text));
        if r.think.final_top_5.is_empty() {
            s.push_str("  (no activations)\n\n");
            continue;
        }
        for (c, a) in &r.think.final_top_5 {
            s.push_str(&format!("  - [{:.4}] {}\n", a, truncate(c, 80)));
        }
        s.push('\n');
    }

    // Counts + decision.
    let mut ask_class_counts: HashMap<&'static str, usize> = HashMap::new();
    let mut think_class_counts: HashMap<&'static str, usize> = HashMap::new();
    let mut coord_count: usize = 0;
    let mut false_idk_count: usize = 0;
    let mut mars_confab: Vec<&'static str> = Vec::new();
    for r in records {
        *ask_class_counts.entry(r.ask.classification).or_insert(0) += 1;
        *think_class_counts.entry(r.think.classification).or_insert(0) += 1;
        if r.ask.classification == "IDK_INCORRECT" {
            false_idk_count += 1;
        }
        if r.coord_eligible {
            coord_count += 1;
        }
        if r.think.classification == "REAL_CONFABULATION" {
            mars_confab.push(r.q.id);
        }
    }

    s.push_str("\n## Distributions\n\n");
    s.push_str("### Ask classification\n\n");
    let mut av: Vec<(&&str, &usize)> = ask_class_counts.iter().collect();
    av.sort_by(|a, b| b.1.cmp(a.1));
    for (k, c) in &av {
        s.push_str(&format!("- {}: {}\n", k, c));
    }
    s.push_str("\n### Think classification\n\n");
    let mut tv: Vec<(&&str, &usize)> = think_class_counts.iter().collect();
    tv.sort_by(|a, b| b.1.cmp(a.1));
    for (k, c) in &tv {
        s.push_str(&format!("- {}: {}\n", k, c));
    }

    s.push_str(&format!(
        "\n## Coordination headline\n\n- false-IDKs (ask refused with answer existing): {}\n- coord-eligible (ask refused + think correct/partial): {}\n",
        false_idk_count, coord_count,
    ));

    let recommendation = if coord_count >= 4 {
        "BR.C0 STRONG — multi-agent hypothesis empirically supported"
    } else if coord_count >= 2 {
        "BR.C0 MARGINAL — Coordinator worth shipping but expect modest leverage; consider compounding with BR.C-Sub or session-memory"
    } else {
        "BR.C0 NOT JUSTIFIED — pivot back to within-recall (E.6); think does not rescue ask's false-IDKs"
    };
    s.push_str(&format!("- recommendation: **{}**\n", recommendation));

    if !mars_confab.is_empty() {
        s.push_str(&format!(
            "\n## CRITICAL — think confabulation on refusal-expected\n\n- Mars questions classified REAL_CONFABULATION: {:?}\n- This is a Lu safety finding bigger than the BR.C path. think bypasses idk_gate; if it produces named-entity Mars answers, the entire think surface needs an IDK gate before it can be exposed to a Coordinator or any user.\n",
            mars_confab,
        ));
    } else {
        s.push_str("\n## Mars safety check\n\n- No REAL_CONFABULATION on Q11/Q12. think did not produce specific named-entity answers for Mars questions on this run. (NOT a guarantee in general — think still has no IDK gate; this is one observation on this brain at this revision.)\n");
    }

    s
}

/// BR.C-Hyp-Methodology — isolate audit runs from canonical WAL state.
///
/// The production load path (`Brain::load_or_new_with_mode_and_key`)
/// always attaches a `<brain>.lb_session.wal` writer (and the
/// `<brain>.pending.wal` and `<brain>.style.wal` writers) keyed off the
/// brain file's path. Every `Brain::ask` and `Brain::think` call then
/// fsyncs `WonderItem` / `ThinkingTrace` entries to that WAL via
/// `lb_session.push_internal`. So a vanilla audit run pointed at
/// `/var/lib/lu/brain.lbr` cumulates state in
/// `/var/lib/lu/brain.lbr.lb_session.wal`, and the WAL replays into
/// the in-memory store on every subsequent load — back-to-back audit
/// runs against a byte-identical brain file see different in-memory
/// state, and the measurement isn't reproducible.
///
/// This helper sidesteps the WAL contamination by copying the canonical
/// `brain.lbr` to a per-process tmp directory and returning the
/// isolated path. The audit then loads from the tmp path, so the load
/// path opens fresh empty WALs in the tmp dir; canonical state is
/// untouched, audit runs are hermetic, and the canonical brain.lbr
/// sha256 + WAL md5 are bit-stable across the audit.
///
/// Returns `(tmp_dir, isolated_brain_path)`. The caller is responsible
/// for removing `tmp_dir` at end of run (best-effort).
fn isolate_audit_brain(canonical: &str) -> (std::path::PathBuf, std::path::PathBuf) {
    let pid = std::process::id();
    let ts_us = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_micros())
        .unwrap_or(0);
    let tmp_dir = std::env::temp_dir().join(format!("lu-audit-{}-{}", pid, ts_us));
    fs::create_dir_all(&tmp_dir).expect("create audit tmp dir");
    let tmp_brain = tmp_dir.join("brain.lbr");
    fs::copy(canonical, &tmp_brain).expect("copy brain.lbr to audit tmp dir");
    eprintln!(
        "audit isolation: canonical={} -> isolated={}",
        canonical,
        tmp_brain.display()
    );
    (tmp_dir, tmp_brain)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let brain_path = args.next().expect("brain path required");
    let keyfile = args.next().expect("keyfile required");
    let output_dir = args.next().unwrap_or_else(|| "/var/lib/lu".to_string());

    // BR.C-Hyp-Methodology — isolate from canonical WAL. See
    // `isolate_audit_brain` doc comment above. `brain_path` continues
    // to be the audit's *reference* identity (recorded in the JSON
    // output) and the source for the copy; the load itself targets the
    // isolated tmp path so all WAL writes land there, not in
    // /var/lib/lu.
    let (audit_tmp_dir, audit_brain_path) = isolate_audit_brain(&brain_path);

    eprintln!("loading {} (key={}) ...", audit_brain_path.display(), keyfile);
    let key = MasterKey::load_from_keyfile(&keyfile).expect("load keyfile");
    let mut brain = Brain::load_or_new_with_mode_and_key(&audit_brain_path, BrainLoadMode::Auto, Some(key))
        .expect("failed to load brain");
    eprintln!(
        "loaded. nodes={} bonds={}",
        brain.network.all_seeds().len(),
        brain.network.bonds().len()
    );

    let thresholds = IdkThresholds::default();
    eprintln!(
        "thresholds: activation_floor={} seed_activation_floor={} multi_anchor_top_floor={} multi_anchor_min_seeds={}",
        thresholds.activation_floor,
        thresholds.seed_activation_floor,
        thresholds.multi_anchor_top_floor,
        thresholds.multi_anchor_min_seeds,
    );

    let mut records: Vec<Record> = Vec::with_capacity(BATTERY.len());
    for q in BATTERY {
        eprintln!("  running {} — {:?}", q.id, q.text);
        let ask = run_ask(&mut brain, q, &thresholds);
        eprintln!(
            "    ASK   mode={} max_act={:.4} class={} time_ms={}",
            ask.mode_fired, ask.max_activation, ask.classification, ask.time_ms
        );
        let think = run_think(&mut brain, q);
        eprintln!(
            "    THINK final_score={:.3} early_exit={} bonds={} prom={} class={} time_ms={}",
            think.final_score,
            think.early_exit,
            think.hypothesis_bonds_written,
            think.hypotheses_promoted,
            think.classification,
            think.time_ms,
        );
        let coord_eligible = coord_eligible(ask.classification, think.classification);
        records.push(Record { q, ask, think, coord_eligible });
    }

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let json_path = format!("{}/coordination_audit_{}.json", output_dir, timestamp);
    let md_path = format!("{}/coordination_audit_{}.md", output_dir, timestamp);

    let mut json = String::from("{\n  \"brain_path\": \"");
    json.push_str(&json_escape(&brain_path));
    json.push_str("\",\n  \"timestamp_unix\": ");
    json.push_str(&format!("{}", timestamp));
    json.push_str(",\n  \"thresholds\": {");
    json.push_str(&format!(
        "\"activation_floor\":{},\"seed_activation_floor\":{},\"multi_anchor_top_floor\":{},\"multi_anchor_min_seeds\":{}",
        thresholds.activation_floor,
        thresholds.seed_activation_floor,
        thresholds.multi_anchor_top_floor,
        thresholds.multi_anchor_min_seeds,
    ));
    json.push_str("},\n  \"records\": [\n");
    for (i, r) in records.iter().enumerate() {
        if i > 0 { json.push_str(",\n"); }
        json.push_str("    ");
        json.push_str(&record_to_json(r));
    }
    json.push_str("\n  ]\n}\n");

    fs::write(&json_path, &json).expect("write json");
    eprintln!("wrote {}", json_path);

    let md = write_markdown_summary(&records, &brain_path);
    fs::write(&md_path, &md).expect("write md");
    eprintln!("wrote {}", md_path);

    println!("{}", md);

    // BR.C-Hyp-Methodology — best-effort cleanup of the audit-isolation
    // tmp dir. If this fails (disk full, permissions), leave it for the
    // OS tmp reaper. Canonical state was never touched.
    let _ = fs::remove_dir_all(&audit_tmp_dir);
}
