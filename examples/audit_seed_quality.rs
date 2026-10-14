//! BR.C-Hyp.2 — top_seeds quality investigation harness.
//!
//! READ-ONLY. For each of the 12 questions in the BR.R0 battery, we
//! capture a full picture of the recall-input pipeline so the audit
//! can attribute pollution in `top_seeds` (the activated nodes
//! returned by `Brain::ask`) to one of five candidate causes:
//!
//!   A. Token-level OR matching without specificity weighting.
//!   B. Stopword/tokenization gap.
//!   C. Activation propagation from polysemic seeds.
//!   D. Insufficient co-occurrence weighting in keyword_index lookup.
//!   E. Hub-attention or curiosity-bias contamination.
//!
//! Per-question capture:
//!   - raw_tokens          — query split on non-alphanumeric, len>=2,
//!                           BEFORE the stopword/query-noise filter
//!                           (mirrors `select_seeds_for_query`).
//!   - content_tokens      — survivors after both filters; these are
//!                           the words that actually drive recall.
//!   - per_token[]         — for every raw token: stopword/noise flags,
//!                           IDF, and (if not filtered out) the
//!                           bucket size + first 5 candidates from
//!                           `keyword_search(single_token)`. The
//!                           bucket is the universe
//!                           `select_seeds_for_query` picks 2 from
//!                           per token.
//!   - selected_seeds      — what `select_seeds_for_query(query)`
//!                           actually returned (label + weight). Up
//!                           to 3.
//!   - keyword_search_hits — top 5 of `keyword_search(query)`, which
//!                           DOES count co-occurrence. Comparing this
//!                           list to selected_seeds shows whether the
//!                           per-token primary path differs from the
//!                           multi-token-aware fallback path.
//!   - top20[]             — first 20 entries of `recall.activated`
//!                           (the actual `top_seeds` list). For each:
//!                           label, KnowledgeContent.kind_str,
//!                           activation, depth, in/out-degree, and
//!                           the matched query tokens (count + list)
//!                           computed by tokenizing the seed content
//!                           and intersecting with content_tokens.
//!   - top3_neighborhoods  — for the top-3 `top_seeds`: 5 sample
//!                           outgoing-bond neighbors (label + bond
//!                           type) so the report can sanity-check
//!                           seed-neighborhood relevance.
//!
//! Reproducibility uses the BR.C-Hyp-Methodology pattern: the
//! canonical brain.lbr is copied to a per-process tmp directory
//! before load, so all WAL writes hit the tmp dir.
//!
//! Usage:
//!   audit_seed_quality <brain> <keyfile> [output_dir=/var/lib/lu]

use lu4d::brain::AskResult;
use lu4d::encrypt::MasterKey;
use lu4d::network::{is_hub_stopword, is_query_noise};
use lu4d::seed::Seed;
use lu4d::{Brain, BrainLoadMode};

use std::collections::HashSet;
use std::fs;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

struct Question {
    id: &'static str,
    category: &'static str,
    text: &'static str,
    canonical: Option<&'static str>,
}

const BATTERY: &[Question] = &[
    Question { id: "Q1",  category: "definition",       text: "What is photosynthesis?",                          canonical: Some("Photosynthesis") },
    Question { id: "Q2",  category: "entity_attr",      text: "Who is Bertrand Russell?",                         canonical: Some("Bertrand Russell") },
    Question { id: "Q3",  category: "entity_attr",      text: "What was the American Revolution?",                canonical: Some("American Revolution") },
    Question { id: "Q4",  category: "entity_attr",      text: "Where was Albert Einstein born?",                  canonical: Some("Albert Einstein") },
    Question { id: "Q5",  category: "geography",        text: "What is the capital of England?",                  canonical: Some("England") },
    Question { id: "Q6",  category: "geography",        text: "What language is spoken in Angola?",               canonical: Some("Angola") },
    Question { id: "Q7",  category: "multihop",         text: "Which philosophers worked with Bertrand Russell?", canonical: Some("Bertrand Russell") },
    Question { id: "Q8",  category: "multihop",         text: "What language did Albert Einstein speak?",         canonical: Some("Albert Einstein") },
    Question { id: "Q9",  category: "open_ended",       text: "Tell me about World War II.",                      canonical: Some("World War II") },
    Question { id: "Q10", category: "open_ended",       text: "Tell me about China.",                             canonical: Some("China") },
    Question { id: "Q11", category: "refusal_expected", text: "Who was the first person on Mars?",                canonical: None },
    Question { id: "Q12", category: "refusal_expected", text: "What is the capital of Mars?",                     canonical: None },
];

const TOP_N: usize = 20;
const NEIGHBOR_SAMPLE: usize = 5;
const PER_TOKEN_SAMPLE: usize = 5;
const KW_SEARCH_SAMPLE: usize = 5;

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 8);
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
    for (i, it) in items.iter().enumerate() {
        if i > 0 { s.push(','); }
        s.push('"');
        s.push_str(&json_escape(it));
        s.push('"');
    }
    s.push(']');
    s
}

fn tokenize_query(query: &str, drop_hub_stopwords: bool, drop_query_noise: bool) -> Vec<String> {
    query
        .to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty() && w.len() >= 2)
        .filter(|w| !drop_hub_stopwords || !is_hub_stopword(w))
        .filter(|w| !drop_query_noise || !is_query_noise(w))
        .map(|s| s.to_string())
        .collect()
}

fn tokenize_content(content: &str) -> HashSet<String> {
    content
        .to_lowercase()
        .split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
        .filter(|w| !w.is_empty() && w.len() > 1)
        .collect()
}

fn isolate_audit_brain(canonical: &str) -> (std::path::PathBuf, std::path::PathBuf) {
    let pid = std::process::id();
    let ts_us = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_micros())
        .unwrap_or(0);
    let tmp_dir = std::env::temp_dir().join(format!("lu-audit-seed-{}-{}", pid, ts_us));
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

struct PerToken {
    token: String,
    is_stopword: bool,
    is_noise: bool,
    idf: f32,
    bucket_size: Option<usize>,
    sample_candidates: Option<Vec<String>>,
}

struct Top20 {
    rank: usize,
    seed: Seed,
    content: String,
    kind: &'static str,
    activation: f32,
    depth: usize,
    in_degree: usize,
    out_degree: usize,
    matched_tokens: Vec<String>,
    matched_count: usize,
}

struct Neighborhood {
    rank: usize,
    seed_label: String,
    samples: Vec<(String, String)>,
}

struct Record<'a> {
    q: &'a Question,
    raw_tokens: Vec<String>,
    content_tokens: Vec<String>,
    per_token: Vec<PerToken>,
    selected_seeds: Vec<(String, f32)>,
    keyword_search_hits: Vec<(String, usize)>,
    top20: Vec<Top20>,
    top3_neighborhoods: Vec<Neighborhood>,
    seed_count: usize,
    activated_count: usize,
    time_ms_ask: u128,
}

fn run_one<'q>(brain: &mut Brain, q: &'q Question) -> Record<'q> {
    let raw_tokens = tokenize_query(q.text, false, false);
    let content_tokens = tokenize_query(q.text, true, true);

    let mut per_token: Vec<PerToken> = Vec::new();
    for tok in &raw_tokens {
        let stop = is_hub_stopword(tok);
        let noise = is_query_noise(tok);
        let idf = brain.network.term_idf(tok);
        let (bucket_size, sample_candidates) = if !stop && !noise {
            let hits = brain.network.keyword_search(tok);
            let samples: Vec<String> = hits
                .iter()
                .take(PER_TOKEN_SAMPLE)
                .filter_map(|(s, _)| brain.network.get_node(s).map(|n| n.content().to_string()))
                .collect();
            (Some(hits.len()), Some(samples))
        } else {
            (None, None)
        };
        per_token.push(PerToken {
            token: tok.clone(),
            is_stopword: stop,
            is_noise: noise,
            idf,
            bucket_size,
            sample_candidates,
        });
    }

    let kw_results = brain.network.keyword_search(q.text);
    let keyword_search_hits: Vec<(String, usize)> = kw_results
        .iter()
        .take(KW_SEARCH_SAMPLE)
        .filter_map(|(s, n)| {
            brain
                .network
                .get_node(s)
                .map(|node| (node.content().to_string(), *n))
        })
        .collect();

    let selected = brain.network.select_seeds_for_query(q.text);
    let selected_seeds: Vec<(String, f32)> = selected
        .iter()
        .filter_map(|(s, w)| {
            brain
                .network
                .get_node(s)
                .map(|n| (n.content().to_string(), *w))
        })
        .collect();

    let t0 = Instant::now();
    let result: AskResult = brain.ask(q.text, 3, TOP_N);
    let time_ms_ask = t0.elapsed().as_millis();

    let mut top20: Vec<Top20> = Vec::new();
    for (i, a) in result.recall.activated.iter().take(TOP_N).enumerate() {
        let (content, kind) = match brain.network.get_node(&a.seed) {
            Some(n) => (n.content().to_string(), n.kind_str()),
            None => (format!("<seed:{:?}>", a.seed.0), "<unknown>"),
        };
        let in_degree = brain.network.in_degree(&a.seed);
        let out_degree = brain.network.out_degree(&a.seed);

        let seed_tokens = tokenize_content(&content);
        let matched_tokens: Vec<String> = content_tokens
            .iter()
            .filter(|t| seed_tokens.contains(*t))
            .cloned()
            .collect();
        let matched_count = matched_tokens.len();

        top20.push(Top20 {
            rank: i + 1,
            seed: a.seed,
            content,
            kind,
            activation: a.activation,
            depth: a.depth,
            in_degree,
            out_degree,
            matched_tokens,
            matched_count,
        });
    }

    let mut top3_neighborhoods: Vec<Neighborhood> = Vec::new();
    for (i, t) in top20.iter().take(3).enumerate() {
        let neighbors: Vec<Seed> = brain.network.neighbors_outgoing(&t.seed).to_vec();
        let mut samples: Vec<(String, String)> = Vec::new();
        for n in neighbors.iter().take(NEIGHBOR_SAMPLE) {
            let label = brain
                .network
                .get_node(n)
                .map(|node| node.content().to_string())
                .unwrap_or_else(|| format!("<seed:{:?}>", n.0));
            let bt = brain
                .network
                .get_bond(&t.seed, n)
                .map(|b| format!("{:?}", b.bond_type()))
                .unwrap_or_else(|| "<no-bond>".to_string());
            samples.push((label, bt));
        }
        top3_neighborhoods.push(Neighborhood {
            rank: i + 1,
            seed_label: t.content.clone(),
            samples,
        });
    }

    Record {
        q,
        raw_tokens,
        content_tokens,
        per_token,
        selected_seeds,
        keyword_search_hits,
        top20,
        top3_neighborhoods,
        seed_count: result.seed_hits.len(),
        activated_count: result.recall.activated.len(),
        time_ms_ask,
    }
}

fn per_token_to_json(p: &PerToken) -> String {
    let bucket = match p.bucket_size {
        Some(n) => format!("{}", n),
        None => "null".to_string(),
    };
    let samples = match &p.sample_candidates {
        Some(v) => fmt_str_list(v),
        None => "null".to_string(),
    };
    format!(
        "{{\"token\":\"{}\",\"is_stopword\":{},\"is_noise\":{},\"idf\":{:.4},\"bucket_size\":{},\"sample_candidates\":{}}}",
        json_escape(&p.token),
        p.is_stopword,
        p.is_noise,
        p.idf,
        bucket,
        samples,
    )
}

fn top20_to_json(t: &Top20) -> String {
    format!(
        "{{\"rank\":{},\"label\":\"{}\",\"kind\":\"{}\",\"activation\":{:.4},\"depth\":{},\"in_degree\":{},\"out_degree\":{},\"matched_count\":{},\"matched_tokens\":{}}}",
        t.rank,
        json_escape(&t.content),
        t.kind,
        t.activation,
        t.depth,
        t.in_degree,
        t.out_degree,
        t.matched_count,
        fmt_str_list(&t.matched_tokens),
    )
}

fn neighborhood_to_json(n: &Neighborhood) -> String {
    let mut samples = String::from("[");
    for (i, (lbl, bt)) in n.samples.iter().enumerate() {
        if i > 0 { samples.push(','); }
        samples.push_str(&format!(
            "{{\"neighbor\":\"{}\",\"bond_type\":\"{}\"}}",
            json_escape(lbl),
            json_escape(bt),
        ));
    }
    samples.push(']');
    format!(
        "{{\"rank\":{},\"seed\":\"{}\",\"samples\":{}}}",
        n.rank,
        json_escape(&n.seed_label),
        samples,
    )
}

fn record_to_json(r: &Record) -> String {
    let canonical = match r.q.canonical {
        Some(s) => format!("\"{}\"", json_escape(s)),
        None => "null".to_string(),
    };
    let mut s = String::new();
    s.push_str(&format!(
        "{{\"id\":\"{}\",\"category\":\"{}\",\"text\":\"{}\",\"canonical\":{},",
        r.q.id,
        r.q.category,
        json_escape(r.q.text),
        canonical,
    ));
    s.push_str(&format!(
        "\"raw_tokens\":{},\"content_tokens\":{},",
        fmt_str_list(&r.raw_tokens),
        fmt_str_list(&r.content_tokens),
    ));
    s.push_str("\"per_token\":[");
    for (i, p) in r.per_token.iter().enumerate() {
        if i > 0 { s.push(','); }
        s.push_str(&per_token_to_json(p));
    }
    s.push_str("],");
    s.push_str("\"selected_seeds\":[");
    for (i, (lbl, w)) in r.selected_seeds.iter().enumerate() {
        if i > 0 { s.push(','); }
        s.push_str(&format!(
            "{{\"label\":\"{}\",\"weight\":{:.4}}}",
            json_escape(lbl),
            w,
        ));
    }
    s.push_str("],");
    s.push_str("\"keyword_search_hits\":[");
    for (i, (lbl, n)) in r.keyword_search_hits.iter().enumerate() {
        if i > 0 { s.push(','); }
        s.push_str(&format!(
            "{{\"label\":\"{}\",\"hits\":{}}}",
            json_escape(lbl),
            n,
        ));
    }
    s.push_str("],");
    s.push_str("\"top20\":[");
    for (i, t) in r.top20.iter().enumerate() {
        if i > 0 { s.push(','); }
        s.push_str(&top20_to_json(t));
    }
    s.push_str("],");
    s.push_str("\"top3_neighborhoods\":[");
    for (i, n) in r.top3_neighborhoods.iter().enumerate() {
        if i > 0 { s.push(','); }
        s.push_str(&neighborhood_to_json(n));
    }
    s.push_str("],");
    s.push_str(&format!(
        "\"seed_count\":{},\"activated_count\":{},\"time_ms_ask\":{}",
        r.seed_count, r.activated_count, r.time_ms_ask,
    ));
    s.push('}');
    s
}

fn record_to_md(r: &Record) -> String {
    let mut s = String::new();
    s.push_str(&format!(
        "## {} — {}\n**Query:** \"{}\"  •  category: `{}`  •  canonical: {}\n\n",
        r.q.id,
        r.q.category,
        r.q.text,
        r.q.category,
        match r.q.canonical {
            Some(c) => format!("`{}`", c),
            None => "_none (refusal_expected)_".to_string(),
        },
    ));
    s.push_str(&format!(
        "**Tokens (raw):** {}  \n**Tokens (content, post-filter):** {}  \n**Selected seeds (TF-IDF, primary path):** ",
        if r.raw_tokens.is_empty() { "_(none)_".to_string() } else { r.raw_tokens.join(", ") },
        if r.content_tokens.is_empty() { "_(none)_".to_string() } else { r.content_tokens.join(", ") },
    ));
    if r.selected_seeds.is_empty() {
        s.push_str("_(empty — fallback to keyword_search)_\n\n");
    } else {
        s.push('\n');
        for (lbl, w) in &r.selected_seeds {
            s.push_str(&format!("  - {:.3} `{}`\n", w, lbl));
        }
        s.push('\n');
    }
    s.push_str("**Per-token:**\n");
    s.push_str("| Token | stopword | noise | IDF | bucket | first 5 candidates |\n");
    s.push_str("|---|---|---|---|---|---|\n");
    for p in &r.per_token {
        let stop = if p.is_stopword { "Y" } else { " " };
        let noise = if p.is_noise { "Y" } else { " " };
        let bucket = match p.bucket_size {
            Some(n) => format!("{}", n),
            None => "-".to_string(),
        };
        let samples = match &p.sample_candidates {
            Some(v) if !v.is_empty() => v
                .iter()
                .map(|c| format!("`{}`", c.replace('|', "&#124;")))
                .collect::<Vec<_>>()
                .join("; "),
            _ => "-".to_string(),
        };
        s.push_str(&format!(
            "| `{}` | {} | {} | {:.3} | {} | {} |\n",
            p.token, stop, noise, p.idf, bucket, samples,
        ));
    }
    s.push_str("\n**`keyword_search` (co-occurrence-weighted, secondary path):**\n");
    if r.keyword_search_hits.is_empty() {
        s.push_str("_(empty)_\n");
    } else {
        for (lbl, n) in &r.keyword_search_hits {
            s.push_str(&format!("  - hits={} `{}`\n", n, lbl));
        }
    }
    s.push('\n');
    s.push_str(&format!(
        "**top_seeds top-{} (from `recall.activated`):**\n",
        r.top20.len()
    ));
    s.push_str("| # | act | depth | matched | kind | in/out deg | label | matched tokens |\n");
    s.push_str("|---|---|---|---|---|---|---|---|\n");
    for t in &r.top20 {
        let label = t.content.replace('|', "&#124;");
        let matched = if t.matched_tokens.is_empty() {
            "-".to_string()
        } else {
            t.matched_tokens.join(",")
        };
        s.push_str(&format!(
            "| {} | {:.4} | {} | {} | {} | {}/{} | `{}` | {} |\n",
            t.rank, t.activation, t.depth, t.matched_count, t.kind, t.in_degree, t.out_degree, label, matched,
        ));
    }
    s.push_str("\n**Top-3 neighborhoods (5 outgoing bonds each):**\n");
    if r.top3_neighborhoods.is_empty() {
        s.push_str("_(no top seeds)_\n");
    } else {
        for n in &r.top3_neighborhoods {
            s.push_str(&format!("\n_#{} `{}`:_\n", n.rank, n.seed_label));
            if n.samples.is_empty() {
                s.push_str("  _(no outgoing bonds)_\n");
            } else {
                for (lbl, bt) in &n.samples {
                    s.push_str(&format!("  - [{}] `{}`\n", bt, lbl));
                }
            }
        }
    }
    s.push_str(&format!(
        "\n_seed_count={}  activated_count={}  time_ms_ask={}_\n\n---\n\n",
        r.seed_count, r.activated_count, r.time_ms_ask,
    ));
    s
}

fn main() {
    let mut args = std::env::args().skip(1);
    let brain_path = args.next().expect("brain path required");
    let keyfile = args.next().expect("keyfile required");
    let output_dir = args.next().unwrap_or_else(|| "/var/lib/lu".to_string());

    let (audit_tmp_dir, audit_brain_path) = isolate_audit_brain(&brain_path);

    eprintln!("loading {} (key={}) ...", audit_brain_path.display(), keyfile);
    let key = MasterKey::load_from_keyfile(&keyfile).expect("load keyfile");
    let mut brain = Brain::load_or_new_with_mode_and_key(
        &audit_brain_path,
        BrainLoadMode::Auto,
        Some(key),
    )
    .expect("failed to load brain");
    eprintln!(
        "loaded. nodes={} bonds={}",
        brain.network.all_seeds().len(),
        brain.network.bonds().len()
    );

    brain.network.ensure_keyword_index_built();
    eprintln!("keyword index built (eager).");

    let mut records: Vec<Record> = Vec::with_capacity(BATTERY.len());
    for q in BATTERY {
        eprintln!("  running {} — {:?}", q.id, q.text);
        let r = run_one(&mut brain, q);
        eprintln!(
            "    selected={} kw_hits={} top20={} t={}ms",
            r.selected_seeds.len(),
            r.keyword_search_hits.len(),
            r.top20.len(),
            r.time_ms_ask,
        );
        records.push(r);
    }

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let json_path = format!("{}/seed_quality_audit_{}.json", output_dir, timestamp);
    let md_path = format!("{}/seed_quality_audit_{}.md", output_dir, timestamp);

    let mut json = String::from("{\n  \"brain_path\": \"");
    json.push_str(&json_escape(&brain_path));
    json.push_str("\",\n  \"timestamp_unix\": ");
    json.push_str(&format!("{}", timestamp));
    json.push_str(",\n  \"records\": [\n");
    for (i, r) in records.iter().enumerate() {
        json.push_str("    ");
        json.push_str(&record_to_json(r));
        if i + 1 < records.len() {
            json.push(',');
        }
        json.push('\n');
    }
    json.push_str("  ]\n}\n");
    fs::write(&json_path, &json).expect("write json");

    let mut md = String::from("# Seed-quality audit (BR.C-Hyp.2)\n\n");
    md.push_str(&format!("_brain_: `{}`  •  _timestamp_: {}\n\n", brain_path, timestamp));
    md.push_str(
        "This harness captures the recall-input pipeline for the BR.R0 12-question \
         battery so seed-quality pollution can be attributed to one of the five \
         BR.C-Hyp.2 candidate causes (A: token-level OR matching without \
         specificity weighting; B: stopword/tokenization gap; C: activation \
         propagation from polysemic seeds; D: insufficient co-occurrence weighting; \
         E: hub-attention or curiosity-bias contamination).\n\n",
    );
    for r in &records {
        md.push_str(&record_to_md(r));
    }
    fs::write(&md_path, &md).expect("write md");

    eprintln!("\nwrote {}", json_path);
    eprintln!("wrote {}", md_path);

    if let Err(e) = fs::remove_dir_all(&audit_tmp_dir) {
        eprintln!(
            "warning: could not remove audit tmp dir {}: {}",
            audit_tmp_dir.display(),
            e
        );
    }
}
