//! BR.C-Hyp — Hypothesis-promotion investigation harness.
//!
//! Diagnoses why `Brain::think` promotes ZERO hypotheses on the
//! canonical 25K-corpus brain (BR.C-Pre baseline,
//! `coordination_audit_baseline_2026-05-09.{json,md}` —
//! `final_score=0.000` and `hypotheses_promoted=0` across all 12
//! BR.R0 questions). Read-only port: no production-code mutation
//! beyond the off-by-default `verbose_trace` observability flag
//! added to `ThinkConfig`.
//!
//! Mirrors the shape of `examples/audit_recall_quality.rs` and
//! `examples/audit_coordination_quality.rs`. We re-run the FIVE
//! BR.C-Pre coord-eligible questions (Q2 Bertrand Russell,
//! Q3 American Revolution, Q6 Angola language, Q7 Russell-philosophers,
//! Q9 World War II) — these are the ones where ASK refused but
//! THINK rescued the answer at the OBSERVATION layer despite
//! promoting zero hypotheses. The investigation question is:
//! WHY zero?
//!
//! Per-question we capture, from the structured `ThoughtTrace`:
//!
//!   - hypotheses_proposed: sum over steps of `step.plan.hypotheses.len()`
//!     (post variant-mode filter; this is what reaches the bond-write
//!     loop)
//!   - hypotheses_written: trace.hypothesis_bonds_written
//!     (after the `already-bonded` short-circuit)
//!   - hypotheses_promoted: trace.hypotheses_promoted
//!     (REFLECT-pass count: `to` activated in pre-write spread)
//!   - rejected_by_REFLECT: sum over steps of
//!     `step.reflection.unconfirmed.len()` — written but `to` did
//!     not activate in pre-reflect
//!   - per-step score distribution
//!   - 5 sample unconfirmed (rejected) hypotheses verbatim per
//!     question, with the from/to content strings and the
//!     observation-time activation that put them in the top-N
//!     in the first place
//!   - matched fragment status (REAL_CORRECT / REAL_PARTIAL /
//!     IDK / REAL_WRONG) on the final_answer (sanity check that
//!     think still rescues these despite zero promotions, per
//!     BR.C-Pre)
//!
//! Note on "rejected_by_score": the current src/thinking.rs
//! REFLECT logic (lines 368-388) does NOT use the per-step `score`
//! as a promotion gate. The score is only used for early-exit
//! against `confidence_to_answer`. Promotion is binary on
//! `pre_reflect_acts.get(&h.to) > 0.0`. This harness reports
//! "rejected_by_score" as N/A and explains why in the markdown
//! summary — that is itself a finding, not a measurement gap.
//!
//! Discipline:
//!   - Read-only on disk. `--no-save` at the launcher level (this
//!     binary never calls `brain.save_*`).
//!   - think DOES dirty in-memory graph (writes Hypothesis bonds,
//!     creates Goal/Step nodes). That matches think's normal
//!     operation. Disk brain is unchanged; caller verifies sha
//!     before/after.
//!
//! Usage:
//!   audit_hypothesis_promotion <brain> <keyfile> [output_dir=/var/lib/lu]
//!
//! Tags into BR.C-Hyp (internal notes). Recovery anchor:
//! `pre-br-c-hyp-2026-05-09` at commit `9d6b8ff`.

use lu4d::encrypt::MasterKey;
use lu4d::thinking::{HypothesisCandidate, ThinkConfig, ThinkResult};
use lu4d::{Brain, BrainLoadMode};

use std::fs;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// Subset of the BR.R0 battery: the 5 coord-eligible questions
/// from BR.C-Pre. Same `fragments` / `partial_fragments` content
/// signatures as `audit_coordination_quality.rs` so think_classify
/// stays comparable across audits.
struct Question {
    id: &'static str,
    text: &'static str,
    fragments: &'static [&'static str],
    partial_fragments: &'static [&'static str],
}

const BATTERY: &[Question] = &[
    Question {
        id: "Q2",
        text: "Who is Bertrand Russell?",
        fragments: &[
            "philosopher", "mathematician", "logician", "logic",
            "british", "welsh", "1872", "1970", "principia",
        ],
        partial_fragments: &["writer", "author", "essay", "nobel"],
    },
    Question {
        id: "Q3",
        text: "What was the American Revolution?",
        fragments: &[
            "independence", "colonies", "colonial", "britain",
            "british", "1775", "1776", "declaration", "war",
            "thirteen colonies",
        ],
        partial_fragments: &["revolt", "king", "taxation", "continental"],
    },
    Question {
        id: "Q6",
        text: "What language is spoken in Angola?",
        fragments: &["portuguese"],
        partial_fragments: &["bantu", "kimbundu", "umbundu", "romance language"],
    },
    Question {
        id: "Q7",
        text: "Which philosophers worked with Bertrand Russell?",
        fragments: &[
            "whitehead", "frege", "wittgenstein", "moore",
            "alfred north", "g. e. moore",
        ],
        partial_fragments: &["philosopher", "logician", "analytic philosophy"],
    },
    Question {
        id: "Q9",
        text: "Tell me about World War II.",
        fragments: &[
            "1939", "1945", "hitler", "nazi", "allies", "axis",
            "d-day", "normandy", "pacific", "holocaust", "pearl harbor",
        ],
        partial_fragments: &["germany", "japan", "europe", "war", "british", "soviet"],
    },
];

fn contains_any_ci(haystack_lc: &str, needles: &[&str]) -> Option<String> {
    for n in needles {
        let nl = n.to_lowercase();
        if haystack_lc.contains(&nl) {
            return Some((*n).to_string());
        }
    }
    None
}

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

fn think_classify(
    q: &Question,
    result: &ThinkResult,
) -> (&'static str, Option<String>) {
    let text = collect_think_text(result);
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

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        let mut out: String = s.chars().take(n.saturating_sub(1)).collect();
        out.push('…');
        out
    }
}

fn fmt_hypothesis(h: &HypothesisCandidate) -> String {
    format!(
        "{{\"from\":\"{}\",\"to\":\"{}\",\"init_act_from\":{:.4},\"init_act_to\":{:.4}}}",
        json_escape(&h.from_content),
        json_escape(&h.to_content),
        h.initial_activation_from,
        h.initial_activation_to,
    )
}

fn fmt_hyps(hs: &[HypothesisCandidate]) -> String {
    let mut s = String::from("[");
    for (i, h) in hs.iter().enumerate() {
        if i > 0 { s.push(','); }
        s.push_str(&fmt_hypothesis(h));
    }
    s.push(']');
    s
}

struct StepRow {
    index: usize,
    obs_top_n: usize,
    obs_top0_act: f32,
    proposed: usize,
    confirmed: usize,
    unconfirmed: usize,
    score: f32,
    sample_unconfirmed: Vec<HypothesisCandidate>,
    sample_confirmed: Vec<HypothesisCandidate>,
}

struct Record<'a> {
    q: &'a Question,
    final_score: f32,
    early_exit: bool,
    steps_run: usize,
    bonds_written: usize,
    bonds_promoted: usize,
    classification: &'static str,
    matched_fragment: Option<String>,
    time_ms: u128,
    steps: Vec<StepRow>,
}

fn run_think_audit<'q>(brain: &mut Brain, q: &'q Question) -> Record<'q> {
    let t0 = Instant::now();
    // verbose_trace=false here so the JSON/MD is the canonical artifact;
    // re-run with verbose_trace=true on the side if a human wants the
    // streaming trace lines.
    let cfg = ThinkConfig::default();
    let result: ThinkResult = brain.think(q.text, cfg);
    let time_ms = t0.elapsed().as_millis();

    let (classification, matched_fragment) = think_classify(q, &result);

    // Per-step audit: capture hypothesis fan-out, gate decisions,
    // and 5 verbatim samples of each rejection / pass class.
    let mut steps: Vec<StepRow> = Vec::with_capacity(result.trace.steps.len());
    for s in &result.trace.steps {
        let proposed = s.plan.hypotheses.len();
        let confirmed = s.reflection.confirmed.len();
        let unconfirmed = s.reflection.unconfirmed.len();

        let sample_unconfirmed: Vec<HypothesisCandidate> = s
            .reflection
            .unconfirmed
            .iter()
            .take(5)
            .cloned()
            .collect();
        let sample_confirmed: Vec<HypothesisCandidate> = s
            .reflection
            .confirmed
            .iter()
            .take(5)
            .cloned()
            .collect();

        let obs_top_n = s.observation.top_seeds.len();
        let obs_top0_act = s
            .observation
            .top_activations
            .first()
            .copied()
            .unwrap_or(0.0);

        steps.push(StepRow {
            index: s.index,
            obs_top_n,
            obs_top0_act,
            proposed,
            confirmed,
            unconfirmed,
            score: s.reflection.score,
            sample_unconfirmed,
            sample_confirmed,
        });
    }

    Record {
        q,
        final_score: result.trace.final_score,
        early_exit: result.trace.early_exit,
        steps_run: result.trace.steps.len(),
        bonds_written: result.trace.hypothesis_bonds_written,
        bonds_promoted: result.trace.hypotheses_promoted,
        classification,
        matched_fragment,
        time_ms,
        steps,
    }
}

fn record_to_json(r: &Record) -> String {
    let matched = match &r.matched_fragment {
        Some(m) => format!("\"{}\"", json_escape(m)),
        None => "null".to_string(),
    };
    let mut steps_json = String::from("[");
    for (i, s) in r.steps.iter().enumerate() {
        if i > 0 { steps_json.push(','); }
        steps_json.push_str(&format!(
            "{{\"index\":{},\"obs_top_n\":{},\"obs_top0_act\":{:.4},\"proposed\":{},\"confirmed\":{},\"unconfirmed\":{},\"score\":{:.4},\"sample_confirmed\":{},\"sample_unconfirmed\":{}}}",
            s.index,
            s.obs_top_n,
            s.obs_top0_act,
            s.proposed,
            s.confirmed,
            s.unconfirmed,
            s.score,
            fmt_hyps(&s.sample_confirmed),
            fmt_hyps(&s.sample_unconfirmed),
        ));
    }
    steps_json.push(']');

    format!(
        "{{\"id\":\"{}\",\"text\":\"{}\",\"think\":{{\"final_score\":{:.4},\"early_exit\":{},\"steps_run\":{},\"bonds_written\":{},\"bonds_promoted\":{},\"classification\":\"{}\",\"matched_fragment\":{},\"time_ms\":{}}},\"steps\":{}}}",
        r.q.id,
        json_escape(r.q.text),
        r.final_score,
        r.early_exit,
        r.steps_run,
        r.bonds_written,
        r.bonds_promoted,
        r.classification,
        matched,
        r.time_ms,
        steps_json,
    )
}

fn aggregate_proposed(r: &Record) -> usize {
    r.steps.iter().map(|s| s.proposed).sum()
}
fn aggregate_unconfirmed(r: &Record) -> usize {
    r.steps.iter().map(|s| s.unconfirmed).sum()
}

fn write_markdown_summary(records: &[Record], brain_path: &str) -> String {
    let mut s = String::new();
    s.push_str("# BR.C-Hyp — Hypothesis-Promotion Investigation\n\n");
    s.push_str(&format!("Brain: `{}`\n\n", brain_path));
    s.push_str("Read-only on disk; `--no-save` discipline at the launcher level.\n");
    s.push_str("think dirties in-memory graph (writes Hypothesis bonds, Goal/Step\n");
    s.push_str("nodes). Disk brain unchanged; caller verifies sha at start and end.\n\n");

    // Headline.
    let total_proposed: usize = records.iter().map(aggregate_proposed).sum();
    let total_written: usize = records.iter().map(|r| r.bonds_written).sum();
    let total_promoted: usize = records.iter().map(|r| r.bonds_promoted).sum();
    let total_unconfirmed: usize = records.iter().map(aggregate_unconfirmed).sum();
    s.push_str("## Headline\n\n");
    s.push_str(&format!(
        "- Questions audited: {}\n- Hypotheses proposed (post variant filter): {}\n- Hypotheses bond-written (after `already-bonded` skip): {}\n- Hypotheses **promoted** (REFLECT pass): {}\n- Hypotheses rejected_by_REFLECT (written but `to` did not pre-activate): {}\n- Hypotheses rejected_by_score: N/A — current REFLECT gate is binary on `pre_reflect_acts.get(&h.to) > 0.0`; `score` is per-step ratio used only for early-exit, not promotion. (Cause D structurally inapplicable in this codebase.)\n\n",
        records.len(), total_proposed, total_written, total_promoted, total_unconfirmed,
    ));

    // Per-question rollup.
    s.push_str("## Per-question rejection breakdown\n\n");
    s.push_str("| ID | proposed | written | promoted | rejected_REFLECT | final_score | think_class | matched |\n");
    s.push_str("|----|----------|---------|----------|------------------|-------------|-------------|---------|\n");
    for r in records {
        let proposed = aggregate_proposed(r);
        let unconfirmed = aggregate_unconfirmed(r);
        let matched = r
            .matched_fragment
            .clone()
            .unwrap_or_else(|| "-".to_string());
        s.push_str(&format!(
            "| {} | {} | {} | {} | {} | {:.3} | {} | {} |\n",
            r.q.id,
            proposed,
            r.bonds_written,
            r.bonds_promoted,
            unconfirmed,
            r.final_score,
            r.classification,
            truncate(&matched, 18),
        ));
    }

    // Per-question step-level scoreboard.
    s.push_str("\n## Per-question per-step score distribution\n\n");
    for r in records {
        s.push_str(&format!("### {} — {}\n\n", r.q.id, r.q.text));
        s.push_str("| step | obs_top_n | obs_top0_act | proposed | confirmed | unconfirmed | score |\n");
        s.push_str("|------|-----------|--------------|----------|-----------|-------------|-------|\n");
        for st in &r.steps {
            s.push_str(&format!(
                "| {} | {} | {:.4} | {} | {} | {} | {:.4} |\n",
                st.index, st.obs_top_n, st.obs_top0_act,
                st.proposed, st.confirmed, st.unconfirmed, st.score,
            ));
        }
        s.push('\n');
    }

    // Sample rejected hypotheses verbatim.
    s.push_str("\n## Sample rejected hypotheses (verbatim from trace)\n\n");
    s.push_str("For each question and step, up to 5 unconfirmed hypotheses with\n");
    s.push_str("the from/to content and observation-time activations. The\n");
    s.push_str("single rejection mechanism in the current code is\n");
    s.push_str("`pre_reflect_acts.get(&h.to) <= 0.0` — i.e. spreading from\n");
    s.push_str("`observation.top_seeds[..3]` did not reach the proposed `to` seed.\n\n");
    for r in records {
        s.push_str(&format!("### {} — {}\n\n", r.q.id, r.q.text));
        for st in &r.steps {
            if st.sample_unconfirmed.is_empty() {
                continue;
            }
            s.push_str(&format!(
                "**step {}** (proposed={} confirmed={} unconfirmed={} score={:.3})\n\n",
                st.index, st.proposed, st.confirmed, st.unconfirmed, st.score,
            ));
            for h in &st.sample_unconfirmed {
                s.push_str(&format!(
                    "  - REJECTED (to not pre-activated): from={:?} -> to={:?} (init_act_from={:.3} init_act_to={:.3})\n",
                    truncate(&h.from_content, 60),
                    truncate(&h.to_content, 60),
                    h.initial_activation_from,
                    h.initial_activation_to,
                ));
            }
            s.push('\n');
        }
    }

    // Sample confirmed (if any).
    let any_confirmed: bool = records
        .iter()
        .any(|r| r.steps.iter().any(|st| !st.sample_confirmed.is_empty()));
    if any_confirmed {
        s.push_str("\n## Sample CONFIRMED hypotheses (any)\n\n");
        for r in records {
            for st in &r.steps {
                if st.sample_confirmed.is_empty() {
                    continue;
                }
                s.push_str(&format!(
                    "- {} step {} CONFIRMED: ", r.q.id, st.index,
                ));
                for (i, h) in st.sample_confirmed.iter().enumerate() {
                    if i > 0 { s.push_str("; "); }
                    s.push_str(&format!(
                        "{:?}->{:?}",
                        truncate(&h.from_content, 40),
                        truncate(&h.to_content, 40),
                    ));
                }
                s.push('\n');
            }
        }
    } else {
        s.push_str("\n## Sample CONFIRMED hypotheses\n\n_(none — zero promotions across all audited questions, matching BR.C-Pre baseline)_\n");
    }

    // Cause-mapping reminder.
    s.push_str("\n## Candidate causes (reference, see report.md for diagnosis)\n\n");
    s.push_str("- A. PRE-WRITE REFLECT gate too strict (top-3 spread origin too narrow / depth too short)\n");
    s.push_str("- B. propose_hypotheses produces only tautologies (correctly gated)\n");
    s.push_str("- C. Brain genuinely too sparse at 25K-doc depth\n");
    s.push_str("- D. Scoring threshold too high — N/A in current code (binary REFLECT, no score-gate)\n");
    s.push_str("- E. Code bug in reflection cycle (gate fails open/closed unconditionally)\n");

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

    let mut records: Vec<Record> = Vec::with_capacity(BATTERY.len());
    for q in BATTERY {
        eprintln!("  running {} — {:?}", q.id, q.text);
        let r = run_think_audit(&mut brain, q);
        eprintln!(
            "    final_score={:.3} early_exit={} steps={} written={} promoted={} class={} time_ms={}",
            r.final_score, r.early_exit, r.steps_run,
            r.bonds_written, r.bonds_promoted, r.classification, r.time_ms,
        );
        records.push(r);
    }

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let json_path = format!("{}/hypothesis_audit_{}.json", output_dir, timestamp);
    let md_path = format!("{}/hypothesis_audit_{}.md", output_dir, timestamp);

    let mut json = String::from("{\n  \"brain_path\": \"");
    json.push_str(&json_escape(&brain_path));
    json.push_str("\",\n  \"timestamp_unix\": ");
    json.push_str(&format!("{}", timestamp));
    json.push_str(",\n  \"records\": [\n");
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
