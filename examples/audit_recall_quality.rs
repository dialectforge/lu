//! BR.R0 — Recall measurement harness.
//!
//! Codifies the 12-question battery internal notes transcript) as
//! structured measurement, mirroring `audit_depth.rs` /
//! `audit_neighborhood.rs` for the recall pipeline. For each
//! question we run `Brain::ask` end-to-end PLUS the IDK gate flow
//! (fast-pass → reason → evaluate_signals → classify) and capture:
//!
//!   - mode_fired           — A / B / C / D / Pass
//!   - max_activation       — top activation across all results
//!   - seed_count           — number of seeds returned
//!   - top_3_seed_contents  — first 3 seeds (label only)
//!   - path_depth           — BFS hops from any seed to the
//!                            canonical subject node (-1 if
//!                            unreachable, omitted when no
//!                            canonical applies, e.g. Mars)
//!   - classification       — REAL_CORRECT / IDK_CORRECT /
//!                            IDK_INCORRECT / REAL_WRONG, derived
//!                            from observed verdict vs the
//!                            per-question expectation.
//!   - idk_signals          — full IdkSignals struct
//!   - time_ms              — wall-clock for the ask
//!
//! Empirical baseline 2026-05-09 on the 25K+canonical brain
//! (sha fb45dfa8…), per the manual #456 transcript:
//!   - 2 PASS   — Q1 photosynthesis, Q10 China
//!   - 2 IDK_CORRECT — Q11/Q12 Mars (refusal expected)
//!   - 8 IDK_INCORRECT — Q2-Q9 (false-IDK at activations 0.375-0.60,
//!                       all below `multi_anchor_top_floor=0.70`)
//!
//! Expected ranges (real-query maxima from the post-revert baseline):
//!   - max_activation:  0.375 - 0.85
//!   - seed_count:      1 - 3
//!   - Mode C threshold (multi_anchor_top_floor): 0.70 — empirically
//!     too high; rebalance is BR.R1.
//!
//! Read-only — never mutates the brain on disk. `--no-save` discipline.
//!
//! Usage:
//!   audit_recall_quality <brain> <keyfile> [output_dir=/var/lib/lu]

use lu4d::brain::AskResult;
use lu4d::encrypt::MasterKey;
use lu4d::idk_gate::{
    IdkSignals, IdkThresholds, IdkVerdict, classify, evaluate_signals, fast_pass_check,
};
use lu4d::reasoning::{ReasoningConfig, reason_with_config};
use lu4d::scope::TemporalScope;
use lu4d::seed::Seed;
use lu4d::{Brain, BrainLoadMode};

use std::collections::{HashMap, HashSet, VecDeque};
use std::fs;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// One question in the battery. `expected_pass` mirrors the manual
/// #456 transcript verdict; the harness compares it to the observed
/// verdict to populate `classification`.
struct Question {
    id: &'static str,
    category: &'static str,
    text: &'static str,
    /// Canonical hub label expected to anchor a real answer. `None`
    /// when no canonical applies (Mars-style refusal queries).
    canonical: Option<&'static str>,
    /// True when the manual battery showed Lu producing a real,
    /// substantive answer (Q1 photosynthesis, Q10 China). False
    /// when the manual battery showed an IDK refusal (everything
    /// else, including the correct-refusal Mars cases).
    expected_pass: bool,
}

const BATTERY: &[Question] = &[
    Question {
        id: "Q1",
        category: "definition",
        text: "What is photosynthesis?",
        canonical: Some("Photosynthesis"),
        expected_pass: true,
    },
    Question {
        id: "Q2",
        category: "entity_attr",
        text: "Who is Bertrand Russell?",
        canonical: Some("Bertrand Russell"),
        expected_pass: true,
    },
    Question {
        id: "Q3",
        category: "entity_attr",
        text: "What was the American Revolution?",
        canonical: Some("American Revolution"),
        expected_pass: true,
    },
    Question {
        id: "Q4",
        category: "entity_attr",
        text: "Where was Albert Einstein born?",
        canonical: Some("Albert Einstein"),
        expected_pass: true,
    },
    Question {
        id: "Q5",
        category: "geography",
        text: "What is the capital of England?",
        canonical: Some("England"),
        expected_pass: true,
    },
    Question {
        id: "Q6",
        category: "geography",
        text: "What language is spoken in Angola?",
        canonical: Some("Angola"),
        expected_pass: true,
    },
    Question {
        id: "Q7",
        category: "multihop",
        text: "Which philosophers worked with Bertrand Russell?",
        canonical: Some("Bertrand Russell"),
        expected_pass: true,
    },
    Question {
        id: "Q8",
        category: "multihop",
        text: "What language did Albert Einstein speak?",
        canonical: Some("Albert Einstein"),
        expected_pass: true,
    },
    Question {
        id: "Q9",
        category: "open_ended",
        text: "Tell me about World War II.",
        canonical: Some("World War II"),
        expected_pass: true,
    },
    Question {
        id: "Q10",
        category: "open_ended",
        text: "Tell me about China.",
        canonical: Some("China"),
        expected_pass: true,
    },
    Question {
        id: "Q11",
        category: "refusal_expected",
        text: "Who was the first person on Mars?",
        canonical: None,
        expected_pass: false,
    },
    Question {
        id: "Q12",
        category: "refusal_expected",
        text: "What is the capital of Mars?",
        canonical: None,
        expected_pass: false,
    },
];

/// Re-derive which mode fired by replaying the same conditions
/// `idk_gate::classify` walks (and in the same order). Returns one
/// of: "A", "B", "C", "D", "Pass".
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

fn find_seed_by_label(brain: &Brain, label: &str) -> Option<Seed> {
    let target = label.to_lowercase();
    for s in brain.network.all_seeds() {
        if let Some(view) = brain.network.get_node(&s) {
            if view.content().to_lowercase() == target {
                return Some(s);
            }
        }
    }
    None
}

fn shortest_path_hops(adj: &HashMap<Seed, Vec<Seed>>, from: &[Seed], target: Seed) -> i64 {
    if from.iter().any(|s| *s == target) {
        return 0;
    }
    let mut best: i64 = -1;
    for start in from {
        let mut dist: HashMap<Seed, i64> = HashMap::new();
        let mut q: VecDeque<Seed> = VecDeque::new();
        dist.insert(*start, 0);
        q.push_back(*start);
        let mut found = -1i64;
        while let Some(cur) = q.pop_front() {
            if cur == target {
                found = dist[&cur];
                break;
            }
            let cd = dist[&cur];
            if cd >= 10 {
                continue;
            }
            if let Some(neis) = adj.get(&cur) {
                for &n in neis {
                    if !dist.contains_key(&n) {
                        dist.insert(n, cd + 1);
                        q.push_back(n);
                    }
                }
            }
        }
        if found >= 0 && (best < 0 || found < best) {
            best = found;
        }
        if best == 1 {
            break;
        }
    }
    best
}

fn build_undirected_adj(brain: &Brain) -> HashMap<Seed, Vec<Seed>> {
    let mut adj: HashMap<Seed, Vec<Seed>> = HashMap::new();
    let mut seen: HashSet<(Seed, Seed)> = HashSet::new();
    for ((from, to), _bv) in brain.network.bonds().iter() {
        let pair = if from.0 <= to.0 { (*from, *to) } else { (*to, *from) };
        if !seen.insert(pair) {
            continue;
        }
        adj.entry(*from).or_default().push(*to);
        if from != to {
            adj.entry(*to).or_default().push(*from);
        }
    }
    adj
}

fn classify_outcome(expected_pass: bool, verdict_pass: bool) -> &'static str {
    match (expected_pass, verdict_pass) {
        (true, true) => "REAL_CORRECT",
        (true, false) => "IDK_INCORRECT",
        (false, false) => "IDK_CORRECT",
        (false, true) => "REAL_WRONG",
    }
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

fn fmt_token_act(pairs: &[(String, f32)]) -> String {
    let mut s = String::from("[");
    for (i, (t, v)) in pairs.iter().enumerate() {
        if i > 0 { s.push(','); }
        s.push_str(&format!("[\"{}\",{:.4}]", json_escape(t), v));
    }
    s.push(']');
    s
}

fn fmt_token_count(pairs: &[(String, usize)]) -> String {
    let mut s = String::from("[");
    for (i, (t, v)) in pairs.iter().enumerate() {
        if i > 0 { s.push(','); }
        s.push_str(&format!("[\"{}\",{}]", json_escape(t), v));
    }
    s.push(']');
    s
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

fn signals_to_json(s: &IdkSignals) -> String {
    format!(
        "{{\"total_activation\":{:.4},\"max_seed_activation\":{:.4},\"max_activation\":{:.4},\"seed_count\":{},\"matched_tokens\":{},\"unmatched_tokens\":{},\"chain_no_chain\":{},\"chain_associative_weak\":{},\"chain_confidence\":{},\"per_token_max_activation\":{},\"per_token_neighbor_activated_count\":{},\"anchored_token_count\":{},\"content_token_count\":{},\"dominant_token\":{},\"coverage_missing_tokens\":{}}}",
        s.total_activation,
        s.max_seed_activation,
        s.max_activation,
        s.seed_count,
        fmt_str_list(&s.matched_tokens),
        fmt_str_list(&s.unmatched_tokens),
        s.chain_no_chain,
        s.chain_associative_weak,
        match s.chain_confidence {
            Some(c) => format!("{:.4}", c),
            None => "null".to_string(),
        },
        fmt_token_act(&s.per_token_max_activation),
        fmt_token_count(&s.per_token_neighbor_activated_count),
        s.anchored_token_count,
        s.content_token_count,
        match &s.dominant_token {
            Some(t) => format!("\"{}\"", json_escape(t)),
            None => "null".to_string(),
        },
        fmt_str_list(&s.coverage_missing_tokens),
    )
}

struct Record<'a> {
    q: &'a Question,
    mode_fired: &'static str,
    verdict: IdkVerdict,
    max_activation: f32,
    max_seed_activation: f32,
    seed_count: usize,
    top_seeds: Vec<String>,
    path_depth: Option<i64>,
    classification: &'static str,
    signals: IdkSignals,
    time_ms: u128,
    activated_count: usize,
    top_activated_label: String,
}

fn record_to_json(r: &Record) -> String {
    let canonical = match r.q.canonical {
        Some(s) => format!("\"{}\"", json_escape(s)),
        None => "null".to_string(),
    };
    let path = match r.path_depth {
        Some(d) => format!("{}", d),
        None => "null".to_string(),
    };
    format!(
        "{{\"id\":\"{}\",\"category\":\"{}\",\"text\":\"{}\",\"canonical\":{},\"expected_pass\":{},\"mode_fired\":\"{}\",\"verdict\":\"{:?}\",\"max_activation\":{:.4},\"max_seed_activation\":{:.4},\"seed_count\":{},\"top_seeds\":{},\"path_depth_to_canonical\":{},\"classification\":\"{}\",\"activated_count\":{},\"top_activated_label\":\"{}\",\"time_ms\":{},\"idk_signals\":{}}}",
        r.q.id,
        r.q.category,
        json_escape(r.q.text),
        canonical,
        r.q.expected_pass,
        r.mode_fired,
        r.verdict,
        r.max_activation,
        r.max_seed_activation,
        r.seed_count,
        fmt_str_list(&r.top_seeds),
        path,
        r.classification,
        r.activated_count,
        json_escape(&r.top_activated_label),
        r.time_ms,
        signals_to_json(&r.signals),
    )
}

fn run_one<'q>(
    brain: &mut Brain,
    q: &'q Question,
    adj: &HashMap<Seed, Vec<Seed>>,
    thresholds: &IdkThresholds,
) -> Record<'q> {
    let t0 = Instant::now();
    let result: AskResult = brain.ask(q.text, 3, 5);
    let time_ms = t0.elapsed().as_millis();

    // Use `result.seed_hits` (the seeds the spread actually started
    // from) — this includes the `keyword_search` fallback path that
    // fires when `select_seeds_for_query` returns empty (e.g.
    // ask-style queries on the canonical brain). Captures top-3 by
    // first-occurrence; AskResult preserves seed order.
    let seed_list: Vec<Seed> = result.seed_hits.clone();
    let top_seeds: Vec<String> = seed_list
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

    let top_activated_label = result
        .recall
        .activated
        .first()
        .and_then(|a| brain.network.get_node(&a.seed).map(|n| n.content().to_string()))
        .unwrap_or_else(|| "<none>".to_string());

    let path_depth = match q.canonical {
        None => None,
        Some(label) => {
            let target = find_seed_by_label(brain, label);
            match target {
                None => Some(-2),
                Some(_) if seed_list.is_empty() => Some(-1),
                Some(t) => Some(shortest_path_hops(adj, &seed_list, t)),
            }
        }
    };

    let classification = classify_outcome(q.expected_pass, verdict == IdkVerdict::Pass);

    Record {
        q,
        mode_fired,
        verdict,
        max_activation: signals.max_activation,
        max_seed_activation: signals.max_seed_activation,
        seed_count: signals.seed_count,
        top_seeds,
        path_depth,
        classification,
        signals,
        time_ms,
        activated_count: result.recall.activated.len(),
        top_activated_label,
    }
}

fn fmt_path(p: Option<i64>) -> String {
    match p {
        None => "-".to_string(),
        Some(-2) => "no-canonical".to_string(),
        Some(-1) => "unreachable".to_string(),
        Some(d) => format!("{}", d),
    }
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

fn write_markdown_summary(records: &[Record], brain_path: &str, brain_sha_hint: &str) -> String {
    let mut s = String::new();
    s.push_str("# BR.R0 — Recall Audit Baseline\n\n");
    s.push_str(&format!("Brain: `{}`\n\n", brain_path));
    s.push_str(&format!("Brain sha (hint): `{}`\n\n", brain_sha_hint));
    s.push_str("Read-only harness; no brain mutations persisted.\n\n");
    s.push_str("## Per-question summary\n\n");
    s.push_str("| ID | Category | Mode | max_act | max_seed | seeds | top_seed | path | classification |\n");
    s.push_str("|----|----------|------|---------|----------|-------|----------|------|----------------|\n");
    for r in records {
        let top_seed = r.top_seeds.first().cloned().unwrap_or_else(|| "-".to_string());
        s.push_str(&format!(
            "| {} | {} | {} | {:.4} | {:.4} | {} | {} | {} | {} |\n",
            r.q.id,
            r.q.category,
            r.mode_fired,
            r.max_activation,
            r.max_seed_activation,
            r.seed_count,
            truncate(&top_seed, 28),
            fmt_path(r.path_depth),
            r.classification,
        ));
    }

    let mut mode_counts: HashMap<&'static str, usize> = HashMap::new();
    let mut class_counts: HashMap<&'static str, usize> = HashMap::new();
    let mut max_acts: Vec<f32> = Vec::new();
    for r in records {
        *mode_counts.entry(r.mode_fired).or_insert(0) += 1;
        *class_counts.entry(r.classification).or_insert(0) += 1;
        max_acts.push(r.max_activation);
    }
    s.push_str("\n## Mode distribution\n\n");
    let mut mv: Vec<(&&str, &usize)> = mode_counts.iter().collect();
    mv.sort_by(|a, b| b.1.cmp(a.1));
    for (m, c) in &mv {
        s.push_str(&format!("- {}: {}\n", m, c));
    }
    s.push_str("\n## Classification distribution\n\n");
    let mut cv: Vec<(&&str, &usize)> = class_counts.iter().collect();
    cv.sort_by(|a, b| b.1.cmp(a.1));
    for (k, c) in &cv {
        s.push_str(&format!("- {}: {}\n", k, c));
    }
    if !max_acts.is_empty() {
        let mut sorted = max_acts.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let min = sorted.first().copied().unwrap_or(0.0);
        let max = sorted.last().copied().unwrap_or(0.0);
        let mean = max_acts.iter().sum::<f32>() / max_acts.len() as f32;
        let median = sorted[sorted.len() / 2];
        s.push_str("\n## max_activation distribution (real-query empirical)\n\n");
        s.push_str(&format!(
            "- min={:.4}  median={:.4}  mean={:.4}  max={:.4}  n={}\n",
            min, median, mean, max, max_acts.len()
        ));
        s.push_str("- raw (sorted): ");
        for (i, v) in sorted.iter().enumerate() {
            if i > 0 { s.push_str(", "); }
            s.push_str(&format!("{:.4}", v));
        }
        s.push('\n');
    }
    s.push_str("\nExpected — 2 PASS (Q1, Q10), 2 IDK_CORRECT (Q11, Q12), 8 IDK_INCORRECT (Q2-Q9). Material divergence is itself a finding.\n");
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

    eprintln!("building undirected adjacency for path probe ...");
    let adj = build_undirected_adj(&brain);
    eprintln!("adj_size={}", adj.len());

    let thresholds = IdkThresholds::default();
    eprintln!(
        "thresholds: activation_floor={} seed_activation_floor={} multi_anchor_top_floor={} multi_anchor_min_seeds={} coverage_floor={} coverage_min_tokens={}",
        thresholds.activation_floor,
        thresholds.seed_activation_floor,
        thresholds.multi_anchor_top_floor,
        thresholds.multi_anchor_min_seeds,
        thresholds.coverage_floor,
        thresholds.coverage_min_tokens,
    );

    let mut records: Vec<Record> = Vec::with_capacity(BATTERY.len());
    for q in BATTERY {
        eprintln!("  running {} — {:?}", q.id, q.text);
        let r = run_one(&mut brain, q, &adj, &thresholds);
        eprintln!(
            "    mode={} max_act={:.4} seeds={} class={} time_ms={}",
            r.mode_fired, r.max_activation, r.seed_count, r.classification, r.time_ms
        );
        records.push(r);
    }

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let json_path = format!("{}/recall_audit_{}.json", output_dir, timestamp);
    let md_path = format!("{}/recall_audit_{}.md", output_dir, timestamp);

    let mut json = String::from("{\n  \"brain_path\": \"");
    json.push_str(&json_escape(&brain_path));
    json.push_str("\",\n  \"timestamp_unix\": ");
    json.push_str(&format!("{}", timestamp));
    json.push_str(",\n  \"thresholds\": {");
    json.push_str(&format!(
        "\"activation_floor\":{},\"seed_activation_floor\":{},\"multi_anchor_top_floor\":{},\"multi_anchor_min_seeds\":{},\"coverage_floor\":{},\"coverage_min_tokens\":{}",
        thresholds.activation_floor,
        thresholds.seed_activation_floor,
        thresholds.multi_anchor_top_floor,
        thresholds.multi_anchor_min_seeds,
        thresholds.coverage_floor,
        thresholds.coverage_min_tokens,
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

    let md = write_markdown_summary(&records, &brain_path, "see brain.lbr sha256 alongside");
    fs::write(&md_path, &md).expect("write md");
    eprintln!("wrote {}", md_path);

    println!("{}", md);

    // BR.C-Hyp-Methodology — best-effort cleanup of the audit-isolation
    // tmp dir. If this fails (disk full, permissions), leave it for the
    // OS tmp reaper. Canonical state was never touched.
    let _ = fs::remove_dir_all(&audit_tmp_dir);
}
