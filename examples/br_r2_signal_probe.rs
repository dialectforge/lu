//! BR.R2 — Signal-quality discrimination probe.
//!
//! Closes the BR.R1 halt (#989) finding that Q2 (Bertrand Russell, should-PASS)
//! and Q12 (capital of Mars, genuine-IDK) are IDENTICAL on every signal
//! `idk_gate::classify` currently consumes — max_activation=0.6000,
//! seed_count=2, Mode C — so the threshold layer cannot separate them.
//!
//! This probe is investigation-only (no fix). It dumps the FULL
//! IdkSignals payload + per-seed bond-shape stats for the BR.R1
//! 12-question battery PLUS adversarial multi-token genuine-IDK
//! queries, so we can see EMPIRICALLY which dimensions discriminate
//! should-PASS from genuine-IDK on the canonical brain.
//!
//! Signals captured per question (beyond what audit_coordination_quality
//! emits):
//!   - per_token_max_activation       (Vec<(token, f32)>)
//!   - per_token_neighbor_activated_count (Vec<(token, usize)>)
//!   - anchored_token_count / content_token_count / coverage_ratio
//!   - matched_tokens / unmatched_tokens
//!   - per top-seed: content label, activation, depth, count of bonds
//!     to nodes that are themselves keyword-hits of ANY query token
//!     (anchor-aligned) vs not (unrelated). This surfaces the
//!     "shallow-surface-match" shape — a single token pulled a seed
//!     whose bond neighborhood is unrelated to the rest of the query.
//!
//! Adversarial battery: 4 multi-token genuine-IDK queries that stress
//! the seed_count>=3 case BR.R1 warned about. If a discriminator we
//! consider promoting to classify input false-PASSes any of these,
//! the candidate is rejected.
//!
//! Read-only on disk via the same isolate_audit_brain pattern
//! audit_coordination_quality uses (BR.C-Hyp-Methodology).
//!
//! Usage:
//!   br_r2_signal_probe <brain> <keyfile> [output_dir=/tmp/br_r2_probe]

use lu4d::brain::AskResult;
use lu4d::encrypt::MasterKey;
use lu4d::idk_gate::{
    IdkThresholds, IdkVerdict, classify, evaluate_signals, extract_query_tokens, fast_pass_check,
};
use lu4d::reasoning::{ReasoningConfig, reason_with_config};
use lu4d::scope::TemporalScope;
use lu4d::seed::Seed;
use lu4d::{Brain, BrainLoadMode};

use std::collections::HashSet;
use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

struct Q {
    id: &'static str,
    class: &'static str, // "SHOULD_PASS" | "GENUINE_IDK" | "ADV_IDK"
    text: &'static str,
}

/// 12 BR.R1 originals (BR.R0 battery) + 4 adversarial multi-token
/// genuine-IDK queries chosen to stress the seed_count>=3 case.
///
/// A1..A4 are content-gap queries Lu's brain has no knowledge of.
/// They keep 3+ content tokens (after the same stopword filter
/// idk_gate uses) so they can probe whether a candidate discriminator
/// would false-PASS at N>=3 — the threat shape #989 explicitly
/// warned about.
const BATTERY: &[Q] = &[
    Q { id: "Q1",  class: "SHOULD_PASS", text: "What is photosynthesis?" },
    Q { id: "Q2",  class: "SHOULD_PASS", text: "Who is Bertrand Russell?" },
    Q { id: "Q3",  class: "SHOULD_PASS", text: "What was the American Revolution?" },
    Q { id: "Q4",  class: "SHOULD_PASS", text: "Where was Albert Einstein born?" },
    Q { id: "Q5",  class: "SHOULD_PASS", text: "What is the capital of England?" },
    Q { id: "Q6",  class: "SHOULD_PASS", text: "What language is spoken in Angola?" },
    Q { id: "Q7",  class: "SHOULD_PASS", text: "Which philosophers worked with Bertrand Russell?" },
    Q { id: "Q8",  class: "SHOULD_PASS", text: "What language did Albert Einstein speak?" },
    Q { id: "Q9",  class: "SHOULD_PASS", text: "Tell me about World War II." },
    Q { id: "Q10", class: "SHOULD_PASS", text: "Tell me about China." },
    Q { id: "Q11", class: "GENUINE_IDK", text: "Who was the first person on Mars?" },
    Q { id: "Q12", class: "GENUINE_IDK", text: "What is the capital of Mars?" },
    // Adversarial multi-token genuine-IDK
    Q { id: "A1", class: "ADV_IDK", text: "Who was the first king of Mars?" },
    Q { id: "A2", class: "ADV_IDK", text: "What is the capital city of Mars?" },
    Q { id: "A3", class: "ADV_IDK", text: "What language did Martians speak in the year 2089?" },
    Q { id: "A4", class: "ADV_IDK", text: "Who composed the Martian national anthem?" },
    // Joint-anchor adversarials: both tokens hit strong unrelated keyword anchors.
    Q { id: "A5", class: "ADV_IDK", text: "Who is Bertrand Mars?" },
    Q { id: "A6", class: "ADV_IDK", text: "Was Russell Einstein a philosopher?" },
    Q { id: "A7", class: "ADV_IDK", text: "What is the photosynthesis of Mars?" },
    Q { id: "A8", class: "ADV_IDK", text: "Did Bertrand Russell speak Martian?" },
];

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

fn fmt_str_f32(xs: &[(String, f32)]) -> String {
    let mut s = String::from("[");
    for (i, (t, v)) in xs.iter().enumerate() {
        if i > 0 { s.push(','); }
        s.push_str(&format!("[\"{}\",{:.4}]", json_escape(t), v));
    }
    s.push(']');
    s
}

fn fmt_str_usize(xs: &[(String, usize)]) -> String {
    let mut s = String::from("[");
    for (i, (t, v)) in xs.iter().enumerate() {
        if i > 0 { s.push(','); }
        s.push_str(&format!("[\"{}\",{}]", json_escape(t), v));
    }
    s.push(']');
    s
}

fn fmt_str_list(xs: &[String]) -> String {
    let mut s = String::from("[");
    for (i, t) in xs.iter().enumerate() {
        if i > 0 { s.push(','); }
        s.push_str(&format!("\"{}\"", json_escape(t)));
    }
    s.push(']');
    s
}

fn mode_label(verdict: IdkVerdict, signals: &lu4d::idk_gate::IdkSignals, thresholds: &IdkThresholds) -> &'static str {
    if verdict == IdkVerdict::Pass {
        return "Pass";
    }
    let chain_weak = signals.chain_no_chain || signals.chain_associative_weak;
    let total = signals.total_activation < thresholds.activation_floor
        && chain_weak
        && signals.max_seed_activation < thresholds.seed_activation_floor
        && signals.matched_tokens.is_empty();
    if total { return "A"; }
    let partial = signals.matched_tokens.len() >= thresholds.partial_leak_min_matched
        && signals.unmatched_tokens.len() >= thresholds.partial_leak_min_unmatched;
    if partial { return "B"; }
    let multi_anchor_shaky = signals.seed_count >= thresholds.multi_anchor_min_seeds
        && signals.max_activation < thresholds.multi_anchor_top_floor;
    if multi_anchor_shaky { return "C"; }
    if signals.is_single_anchor_dominance(thresholds) { return "D"; }
    "unknown"
}

fn isolate_audit_brain(canonical: &str) -> (std::path::PathBuf, std::path::PathBuf) {
    let pid = std::process::id();
    let ts_us = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_micros())
        .unwrap_or(0);
    let tmp_dir = std::env::temp_dir().join(format!("lu-br-r2-{}-{}", pid, ts_us));
    fs::create_dir_all(&tmp_dir).expect("create probe tmp dir");
    let tmp_brain = tmp_dir.join("brain.lbr");
    fs::copy(canonical, &tmp_brain).expect("copy brain.lbr to probe tmp dir");
    eprintln!(
        "br_r2 isolation: canonical={} -> isolated={}",
        canonical,
        tmp_brain.display()
    );
    (tmp_dir, tmp_brain)
}

struct SeedShape {
    content: String,
    activation: f32,
    depth: usize,
    nbr_total: usize,
    nbr_anchor_aligned: usize, // neighbors that are keyword-hits of ANY query token
    nbr_activated: usize,      // neighbors present in the activated set
}

/// For each top-seed, compute bond-shape stats: total neighbor count,
/// how many neighbors are themselves keyword-hits of ANY query token
/// (anchor-aligned bond), and how many are activated. This surfaces
/// the "shallow-surface-match" shape: a single-token-match seed whose
/// neighborhood is unrelated to the rest of the query.
fn seed_shape(
    brain: &Brain,
    seed: &Seed,
    ask: &AskResult,
    tokens: &[String],
) -> SeedShape {
    let content = brain
        .network
        .get_node(seed)
        .map(|n| n.content().to_string())
        .unwrap_or_else(|| format!("<seed:{:?}>", seed.0));
    // activation + depth: search the activated list (depth-0 should be present)
    let (activation, depth) = ask
        .recall
        .activated
        .iter()
        .find(|a| a.seed == *seed)
        .map(|a| (a.activation, a.depth))
        .unwrap_or((0.0, 0));
    let activated_set: HashSet<Seed> = ask.recall.activated.iter().map(|a| a.seed).collect();
    let mut nbr_set: HashSet<Seed> = HashSet::new();
    for n in brain.network.neighbors_outgoing(seed) {
        nbr_set.insert(n);
    }
    for n in brain.network.neighbors_incoming(seed) {
        nbr_set.insert(n);
    }
    // Build union of keyword-hits across all query tokens for the
    // anchor-aligned test.
    let mut anchor_hit_set: HashSet<Seed> = HashSet::new();
    for tok in tokens {
        for (s, _) in brain.network.keyword_search(tok) {
            anchor_hit_set.insert(s);
        }
    }
    let nbr_total = nbr_set.len();
    let nbr_anchor_aligned = nbr_set.iter().filter(|n| anchor_hit_set.contains(n)).count();
    let nbr_activated = nbr_set.iter().filter(|n| activated_set.contains(n)).count();
    SeedShape { content, activation, depth, nbr_total, nbr_anchor_aligned, nbr_activated }
}

fn record_to_json(
    q: &Q,
    verdict: IdkVerdict,
    mode: &str,
    signals: &lu4d::idk_gate::IdkSignals,
    seed_shapes: &[SeedShape],
) -> String {
    let mut s = String::new();
    s.push_str("{");
    s.push_str(&format!("\"id\":\"{}\",\"class\":\"{}\",\"text\":\"{}\",",
        q.id, q.class, json_escape(q.text)));
    s.push_str(&format!("\"verdict\":\"{:?}\",\"mode\":\"{}\",", verdict, mode));
    s.push_str(&format!(
        "\"max_activation\":{:.4},\"max_seed_activation\":{:.4},\"total_activation\":{:.4},\"seed_count\":{},",
        signals.max_activation, signals.max_seed_activation,
        signals.total_activation, signals.seed_count
    ));
    s.push_str(&format!(
        "\"content_token_count\":{},\"anchored_token_count\":{},",
        signals.content_token_count, signals.anchored_token_count
    ));
    s.push_str(&format!(
        "\"matched_tokens\":{},\"unmatched_tokens\":{},",
        fmt_str_list(&signals.matched_tokens),
        fmt_str_list(&signals.unmatched_tokens),
    ));
    s.push_str(&format!(
        "\"per_token_max_activation\":{},\"per_token_neighbor_activated_count\":{},",
        fmt_str_f32(&signals.per_token_max_activation),
        fmt_str_usize(&signals.per_token_neighbor_activated_count),
    ));
    // top seeds
    s.push_str("\"top_seeds\":[");
    for (i, ss) in seed_shapes.iter().enumerate() {
        if i > 0 { s.push(','); }
        s.push_str(&format!(
            "{{\"content\":\"{}\",\"activation\":{:.4},\"depth\":{},\"nbr_total\":{},\"nbr_anchor_aligned\":{},\"nbr_activated\":{}}}",
            json_escape(&ss.content), ss.activation, ss.depth,
            ss.nbr_total, ss.nbr_anchor_aligned, ss.nbr_activated,
        ));
    }
    s.push(']');
    s.push('}');
    s
}

fn main() {
    let mut args = std::env::args().skip(1);
    let brain_path = args.next().expect("brain path required");
    let keyfile = args.next().expect("keyfile required");
    let output_dir = args.next().unwrap_or_else(|| "/tmp/br_r2_probe".to_string());
    fs::create_dir_all(&output_dir).ok();

    let (tmp_dir, audit_brain_path) = isolate_audit_brain(&brain_path);

    eprintln!("loading {} (key={}) ...", audit_brain_path.display(), keyfile);
    let key = MasterKey::load_from_keyfile(&keyfile).expect("load keyfile");
    let mut brain =
        Brain::load_or_new_with_mode_and_key(&audit_brain_path, BrainLoadMode::Auto, Some(key))
            .expect("failed to load brain");
    eprintln!(
        "loaded. nodes={} bonds={}",
        brain.network.all_seeds().len(),
        brain.network.bonds().len()
    );

    let thresholds = IdkThresholds::default();

    let mut json = String::from("{\n  \"brain_path\": \"");
    json.push_str(&json_escape(&brain_path));
    json.push_str("\",\n  \"records\": [\n");
    let mut header = String::new();
    header.push_str("id\tclass\tverdict\tmode\tmax_act\tseed_count\tcontent_toks\tanchored\tper_tok_max\tper_tok_nbr\ttop_seed_contents\n");
    println!("{}", header.trim_end());

    for (i, q) in BATTERY.iter().enumerate() {
        eprintln!("  running {} — {:?}", q.id, q.text);
        let ask = brain.ask(q.text, 3, 5);
        let tokens = extract_query_tokens(q.text);
        let fast_pass = fast_pass_check(&brain, &ask, q.text, &thresholds);
        let signals = if fast_pass {
            evaluate_signals(&brain, &ask, q.text, None, &thresholds)
        } else {
            let cfg = ReasoningConfig::default();
            let trace = reason_with_config(&mut brain, q.text, TemporalScope::Current, &cfg);
            evaluate_signals(&brain, &ask, q.text, Some(&trace), &thresholds)
        };
        let verdict = if fast_pass { IdkVerdict::Pass } else { classify(&signals, &thresholds) };
        let mode = mode_label(verdict, &signals, &thresholds);

        // top-3 seed shapes
        let seed_shapes: Vec<SeedShape> = ask
            .seed_hits
            .iter()
            .take(3)
            .map(|s| seed_shape(&brain, s, &ask, &tokens))
            .collect();

        // TSV row
        let per_tok_max: String = signals
            .per_token_max_activation
            .iter()
            .map(|(t, v)| format!("{}={:.3}", t, v))
            .collect::<Vec<_>>()
            .join(",");
        let per_tok_nbr: String = signals
            .per_token_neighbor_activated_count
            .iter()
            .map(|(t, v)| format!("{}={}", t, v))
            .collect::<Vec<_>>()
            .join(",");
        let top_contents: String = seed_shapes
            .iter()
            .map(|ss| {
                let c = if ss.content.len() > 38 { format!("{}…", &ss.content[..37]) } else { ss.content.clone() };
                format!("[{:.2}|{}/{}]{}", ss.activation, ss.nbr_anchor_aligned, ss.nbr_total, c)
            })
            .collect::<Vec<_>>()
            .join(" || ");
        println!(
            "{}\t{}\t{:?}\t{}\t{:.4}\t{}\t{}\t{}\t{}\t{}\t{}",
            q.id, q.class, verdict, mode,
            signals.max_activation, signals.seed_count,
            signals.content_token_count, signals.anchored_token_count,
            per_tok_max, per_tok_nbr, top_contents,
        );

        // JSON record
        if i > 0 { json.push_str(",\n"); }
        json.push_str("    ");
        json.push_str(&record_to_json(q, verdict, mode, &signals, &seed_shapes));
    }
    json.push_str("\n  ]\n}\n");

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let out_path = format!("{}/br_r2_probe_{}.json", output_dir, timestamp);
    fs::write(&out_path, &json).expect("write json");
    eprintln!("wrote {}", out_path);

    let _ = fs::remove_dir_all(&tmp_dir);
}
