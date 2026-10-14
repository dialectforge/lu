//! BR.R3 — Topical-density discrimination probe.
//!
//! Closes OP-008 / internal notes: at 4.5M bonds, the 25K-PASSing Q2
//! "Who is Bertrand Russell?" REGRESSES to PartialMatchLeak because
//! `select_seeds_for_query_inner` takes the first 2 non-polluted seeds
//! per token from the keyword bucket. With 4.5M bonds the per-token
//! buckets are huge and the first 2 are partial-token-match noise
//! ("Bertrand Renouvin", "Russell Coutts") that outrank the actual
//! "Bertrand Russell" article — even though the article DOES appear
//! in both tokens' buckets (i.e. coverage>=2 across query tokens).
//!
//! This probe is investigation-only (no fix). It dumps, per query,
//! the top-N candidate seeds (deeper than the take-2-per-token limit
//! the production code uses) annotated with multiple discriminator
//! candidates so we can see EMPIRICALLY which dimension cleanly
//! separates true seeds from partial-token-match noise.
//!
//! Captured per candidate seed:
//!   - content (node text), seed bytes (deterministic)
//!   - per-token bucket presence (1/0)
//!   - bucket_coverage = sum(per-token bucket presence)
//!   - per-token content substring match (1/0)
//!   - content_coverage = sum(per-token content match)
//!   - nbr_total / nbr_anchor_aligned (1-hop, BR.R2 shape)
//!   - nbr_token_coverage = count of distinct query tokens whose
//!     keyword-hits appear in the seed's 1-hop neighborhood (NEW;
//!     the cross-token co-occurrence signal called out by the
//!     BR.R3 brief)
//!   - activation_in_spread / depth_in_spread (if currently activated)
//!
//! Per-query summary also captures the CURRENT seeds returned by
//! `select_seeds_for_query_ask`, the candidate's rank-by-bucket-coverage,
//! and the current ask() verdict.
//!
//! Same disk-isolation pattern as BR.R2 (read-only on canonical).
//!
//! Usage:
//!   br_r3_topical_density_probe <brain> <keyfile> [output_dir=/tmp/br_r3_probe]

use lu4d::brain::AskResult;
use lu4d::encrypt::MasterKey;
use lu4d::idk_gate::{
    IdkThresholds, IdkVerdict, classify, evaluate_signals, extract_query_tokens, fast_pass_check,
};
use lu4d::reasoning::{ReasoningConfig, reason_with_config};
use lu4d::scope::TemporalScope;
use lu4d::seed::Seed;
use lu4d::{Brain, BrainLoadMode};

use std::collections::{HashMap, HashSet};
use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

struct Q {
    id: &'static str,
    class: &'static str, // "SHOULD_PASS" | "GENUINE_IDK" | "ADV_IDK"
    text: &'static str,
}

/// Same 20-question battery BR.R2 uses, so the verdict deltas can
/// be apples-to-apples compared.
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
    Q { id: "A1", class: "ADV_IDK", text: "Who was the first king of Mars?" },
    Q { id: "A2", class: "ADV_IDK", text: "What is the capital city of Mars?" },
    Q { id: "A3", class: "ADV_IDK", text: "What language did Martians speak in the year 2089?" },
    Q { id: "A4", class: "ADV_IDK", text: "Who composed the Martian national anthem?" },
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

fn isolate_audit_brain(canonical: &str) -> (std::path::PathBuf, std::path::PathBuf) {
    let pid = std::process::id();
    let ts_us = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_micros())
        .unwrap_or(0);
    let tmp_dir = std::env::temp_dir().join(format!("lu-br-r3-{}-{}", pid, ts_us));
    fs::create_dir_all(&tmp_dir).expect("create probe tmp dir");
    let tmp_brain = tmp_dir.join("brain.lbr");
    fs::copy(canonical, &tmp_brain).expect("copy brain.lbr to probe tmp dir");
    eprintln!(
        "br_r3 isolation: canonical={} -> isolated={}",
        canonical,
        tmp_brain.display()
    );
    (tmp_dir, tmp_brain)
}

struct Candidate {
    seed: Seed,
    content: String,
    bucket_hits: Vec<u8>,        // per content-token: 1 if seed in token's bucket
    content_hits: Vec<u8>,       // per content-token: 1 if content contains token substring
    nbr_total: usize,
    nbr_anchor_aligned: usize,
    nbr_token_coverage: usize,   // distinct tokens whose keyword-hits appear in 1-hop
    activation: f32,
    depth: usize,
}

fn record_candidate_json(c: &Candidate, tokens: &[String]) -> String {
    let bucket_cov: usize = c.bucket_hits.iter().map(|&b| b as usize).sum();
    let content_cov: usize = c.content_hits.iter().map(|&b| b as usize).sum();
    let mut s = String::new();
    s.push('{');
    s.push_str(&format!("\"content\":\"{}\",", json_escape(&c.content)));
    s.push_str(&format!(
        "\"bucket_coverage\":{},\"content_coverage\":{},\"nbr_total\":{},\"nbr_anchor_aligned\":{},\"nbr_token_coverage\":{},\"activation\":{:.4},\"depth\":{},",
        bucket_cov, content_cov, c.nbr_total, c.nbr_anchor_aligned, c.nbr_token_coverage,
        c.activation, c.depth
    ));
    s.push_str("\"per_token\":[");
    for (i, t) in tokens.iter().enumerate() {
        if i > 0 { s.push(','); }
        s.push_str(&format!(
            "{{\"token\":\"{}\",\"in_bucket\":{},\"in_content\":{}}}",
            json_escape(t), c.bucket_hits[i], c.content_hits[i]
        ));
    }
    s.push(']');
    s.push('}');
    s
}

fn build_candidates(
    brain: &Brain,
    tokens: &[String],
    ask: &AskResult,
    per_token_cap: usize,
) -> Vec<Candidate> {
    // 1. For each content token, get its keyword_search bucket and
    //    stable-sort by seed bytes (so probe runs are bit-stable).
    let mut per_token_buckets: Vec<Vec<Seed>> = Vec::with_capacity(tokens.len());
    for tok in tokens {
        let mut hits: Vec<Seed> = brain
            .network
            .keyword_search(tok)
            .into_iter()
            .map(|(s, _)| s)
            .collect();
        hits.sort_by(|a, b| a.as_bytes().cmp(b.as_bytes()));
        per_token_buckets.push(hits);
    }
    // 2. Universe of candidates: union of first `per_token_cap` of each
    //    bucket. Tracked in a HashMap keyed by seed.
    let mut universe: HashMap<Seed, Vec<u8>> = HashMap::new();
    for (i, bucket) in per_token_buckets.iter().enumerate() {
        for seed in bucket.iter().take(per_token_cap) {
            let entry = universe
                .entry(*seed)
                .or_insert_with(|| vec![0u8; tokens.len()]);
            entry[i] = 1;
        }
    }
    // Also include ALL keyword_index hits across the full bucket, not
    // just the first `per_token_cap`, when computing bucket_coverage —
    // a seed that appeared LATE in the "bertrand" bucket might still
    // appear EARLY in the "russell" bucket and we want its coverage
    // recorded honestly. So re-scan full buckets to set bucket bits
    // for already-known universe members.
    for (i, bucket) in per_token_buckets.iter().enumerate() {
        let bucket_set: HashSet<Seed> = bucket.iter().copied().collect();
        for (seed, hits) in universe.iter_mut() {
            if bucket_set.contains(seed) {
                hits[i] = 1;
            }
        }
    }
    // 3. Anchor-hit set for nbr_anchor_aligned (union across tokens) +
    //    per-token sets for nbr_token_coverage.
    let mut anchor_union: HashSet<Seed> = HashSet::new();
    let mut per_token_hit_sets: Vec<HashSet<Seed>> = Vec::with_capacity(tokens.len());
    for bucket in &per_token_buckets {
        let set: HashSet<Seed> = bucket.iter().copied().collect();
        anchor_union.extend(set.iter().copied());
        per_token_hit_sets.push(set);
    }
    // 4. Activated seeds from current spread.
    let activated_act: HashMap<Seed, (f32, usize)> = ask
        .recall
        .activated
        .iter()
        .map(|a| (a.seed, (a.activation, a.depth)))
        .collect();
    // 5. Build candidates, computing per-seed shape.
    let mut candidates: Vec<Candidate> = Vec::with_capacity(universe.len());
    for (seed, bucket_hits) in universe.iter() {
        let content = brain
            .network
            .get_node(seed)
            .map(|n| n.content().to_string())
            .unwrap_or_else(|| format!("<seed:{:?}>", &seed.as_bytes()[..4]));
        let content_lc = content.to_lowercase();
        let content_hits: Vec<u8> = tokens
            .iter()
            .map(|t| if content_lc.contains(t.as_str()) { 1 } else { 0 })
            .collect();
        // 1-hop neighborhood (outgoing union incoming) -- bounded to
        // avoid blowup on hub seeds.
        let mut nbr_set: HashSet<Seed> = HashSet::new();
        for n in brain.network.neighbors_outgoing(seed) {
            nbr_set.insert(n);
            if nbr_set.len() >= 5000 { break; }
        }
        for n in brain.network.neighbors_incoming(seed) {
            nbr_set.insert(n);
            if nbr_set.len() >= 5000 { break; }
        }
        let nbr_total = nbr_set.len();
        let nbr_anchor_aligned = nbr_set.iter().filter(|n| anchor_union.contains(n)).count();
        let mut nbr_token_coverage = 0;
        for set in &per_token_hit_sets {
            if nbr_set.iter().any(|n| set.contains(n)) {
                nbr_token_coverage += 1;
            }
        }
        let (activation, depth) = activated_act.get(seed).copied().unwrap_or((0.0, 0));
        candidates.push(Candidate {
            seed: *seed,
            content,
            bucket_hits: bucket_hits.clone(),
            content_hits,
            nbr_total,
            nbr_anchor_aligned,
            nbr_token_coverage,
            activation,
            depth,
        });
    }
    // 6. Stable rank: by (bucket_coverage desc, content_coverage desc,
    //    nbr_token_coverage desc, nbr_anchor_aligned desc, seed bytes asc).
    candidates.sort_by(|a, b| {
        let a_bc: usize = a.bucket_hits.iter().map(|&b| b as usize).sum();
        let b_bc: usize = b.bucket_hits.iter().map(|&b| b as usize).sum();
        let a_cc: usize = a.content_hits.iter().map(|&b| b as usize).sum();
        let b_cc: usize = b.content_hits.iter().map(|&b| b as usize).sum();
        b_bc.cmp(&a_bc)
            .then(b_cc.cmp(&a_cc))
            .then(b.nbr_token_coverage.cmp(&a.nbr_token_coverage))
            .then(b.nbr_anchor_aligned.cmp(&a.nbr_anchor_aligned))
            .then(a.seed.as_bytes().cmp(b.seed.as_bytes()))
    });
    candidates
}

fn current_select_seeds_summary(brain: &Brain, query: &str) -> String {
    let weighted = brain.network.select_seeds_for_query_ask(query);
    let parts: Vec<String> = weighted
        .iter()
        .map(|(s, w)| {
            let content = brain
                .network
                .get_node(s)
                .map(|n| n.content().to_string())
                .unwrap_or_else(|| String::from("<unknown>"));
            let c = if content.len() > 60 { format!("{}…", &content[..58]) } else { content };
            format!("[{:.3}]{}", w, c)
        })
        .collect();
    parts.join(" || ")
}

fn main() {
    let mut args = std::env::args().skip(1);
    let brain_path = args.next().expect("brain path required");
    let keyfile = args.next().expect("keyfile required");
    let output_dir = args.next().unwrap_or_else(|| "/tmp/br_r3_probe".to_string());
    let per_token_cap: usize = std::env::var("BR_R3_PER_TOKEN_CAP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(50);
    let top_n_dump: usize = std::env::var("BR_R3_TOP_N_DUMP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
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
    json.push_str("\",\n  \"per_token_cap\": ");
    json.push_str(&per_token_cap.to_string());
    json.push_str(",\n  \"top_n_dump\": ");
    json.push_str(&top_n_dump.to_string());
    json.push_str(",\n  \"records\": [\n");

    println!("id\tclass\tverdict\tmax_act\tn_cands\trank1_bc\trank1_cc\trank1_ntc\trank1_content\tcurrent_seeds");

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

        let candidates = build_candidates(&brain, &tokens, &ask, per_token_cap);
        let current = current_select_seeds_summary(&brain, q.text);

        let r1 = candidates.first();
        let (r1_bc, r1_cc, r1_ntc, r1_content) = match r1 {
            Some(c) => (
                c.bucket_hits.iter().map(|&b| b as usize).sum::<usize>(),
                c.content_hits.iter().map(|&b| b as usize).sum::<usize>(),
                c.nbr_token_coverage,
                if c.content.len() > 60 { format!("{}…", &c.content[..58]) } else { c.content.clone() },
            ),
            None => (0, 0, 0, String::new()),
        };
        println!(
            "{}\t{}\t{:?}\t{:.4}\t{}\t{}\t{}\t{}\t{}\t{}",
            q.id, q.class, verdict, signals.max_activation,
            candidates.len(), r1_bc, r1_cc, r1_ntc,
            r1_content, current,
        );

        if i > 0 { json.push_str(",\n"); }
        json.push_str("    {");
        json.push_str(&format!("\"id\":\"{}\",\"class\":\"{}\",\"text\":\"{}\",",
            q.id, q.class, json_escape(q.text)));
        json.push_str(&format!("\"verdict\":\"{:?}\",\"max_activation\":{:.4},\"seed_count\":{},\"matched\":[",
            verdict, signals.max_activation, signals.seed_count));
        for (j, t) in signals.matched_tokens.iter().enumerate() {
            if j > 0 { json.push(','); }
            json.push_str(&format!("\"{}\"", json_escape(t)));
        }
        json.push_str("],\"unmatched\":[");
        for (j, t) in signals.unmatched_tokens.iter().enumerate() {
            if j > 0 { json.push(','); }
            json.push_str(&format!("\"{}\"", json_escape(t)));
        }
        json.push_str("],\"tokens\":[");
        for (j, t) in tokens.iter().enumerate() {
            if j > 0 { json.push(','); }
            json.push_str(&format!("\"{}\"", json_escape(t)));
        }
        json.push_str("],\"current_seeds\":\"");
        json.push_str(&json_escape(&current));
        json.push_str("\",\"n_candidates\":");
        json.push_str(&candidates.len().to_string());
        json.push_str(",\"top_candidates\":[");
        for (j, c) in candidates.iter().take(top_n_dump).enumerate() {
            if j > 0 { json.push(','); }
            json.push_str(&record_candidate_json(c, &tokens));
        }
        json.push_str("]}");
    }
    json.push_str("\n  ]\n}\n");

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let out_path = format!("{}/br_r3_probe_{}.json", output_dir, timestamp);
    fs::write(&out_path, &json).expect("write json");
    eprintln!("wrote {}", out_path);

    let _ = fs::remove_dir_all(&tmp_dir);
}
