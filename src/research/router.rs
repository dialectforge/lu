//! Research config + router that orchestrates Grokipedia → DDG fallback.
//!
//! Grokipedia is queried FIRST per stored preference
//! (memory/feedback_web_search_order.md). DDG fires only if Grokipedia
//! returns nothing useful (NoResults or HTTP error). The router does
//! not synthesize triples — that's `extract::extract_triples`. It only
//! collects evidence: a list of (source_url, excerpt) pairs.

use crate::research::duckduckgo;
use crate::research::grokipedia;
use crate::research::http::HttpFetcher;
use serde::{Deserialize, Serialize};

/// Knobs for the autonomous research loop.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResearchConfig {
    /// Master switch. When false, `Brain::research_loop_run` no-ops and
    /// `lu4d_research_loop_run` returns an empty proposal list.
    pub enabled: bool,
    /// Wonder-queue intensity below which the loop stops popping.
    pub research_intensity_threshold: f32,
    /// Hard cap on Pending proposals. The loop stops collecting once
    /// this many are queued so John isn't drowned in approvals.
    pub max_proposals_pending: usize,
    /// Per-fetch HTTP timeout.
    pub http_timeout_ms: u64,
    /// User-agent string. Identifies the bot to operators of polled
    /// services so they can rate-limit / block / contact us.
    pub user_agent: String,
    /// Cap on excerpts (clean-text snippets) attached to a single
    /// proposal — keeps the proposal payload bounded.
    pub max_excerpts_per_proposal: usize,
    /// Cap on bond candidates per proposal.
    pub max_bond_candidates_per_proposal: usize,
    /// Cosine threshold above which a candidate's fingerprint is
    /// considered a match for an existing scar fingerprint and the
    /// candidate is dropped. Defaults to 0.85.
    pub scar_fingerprint_match_threshold: f32,
}

impl Default for ResearchConfig {
    fn default() -> Self {
        ResearchConfig {
            enabled: true,
            research_intensity_threshold: 0.6,
            max_proposals_pending: 16,
            http_timeout_ms: 10_000,
            user_agent:
                "LuAi/0.1 (research-bot; info@dialectforge.com)".to_string(),
            max_excerpts_per_proposal: 5,
            max_bond_candidates_per_proposal: 8,
            scar_fingerprint_match_threshold: 0.85,
        }
    }
}

/// One piece of evidence: where it came from, and the relevant prose.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResearchEvidence {
    pub source_url: String,
    pub excerpt: String,
    /// Which provider produced this evidence. Helps debugging the
    /// fallback ordering and lets the audit log show provenance.
    pub provider: ResearchProvider,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResearchProvider {
    Grokipedia,
    DuckDuckGo,
}

#[derive(Debug, Clone)]
pub enum RouterError {
    AllProvidersFailed { tried: Vec<String> },
}

impl std::fmt::Display for RouterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RouterError::AllProvidersFailed { tried } => {
                write!(f, "all research providers failed: {:?}", tried)
            }
        }
    }
}

impl std::error::Error for RouterError {}

/// Try Grokipedia first, then DDG (Instant Answer, then HTML
/// fallback). Returns the union of evidence pieces, capped by
/// `cfg.max_excerpts_per_proposal`. The order of evidence preserves
/// provider preference.
pub fn run_router(
    fetcher: &dyn HttpFetcher,
    cfg: &ResearchConfig,
    query: &str,
) -> Result<Vec<ResearchEvidence>, RouterError> {
    let mut evidence: Vec<ResearchEvidence> = Vec::new();
    let mut tried: Vec<String> = Vec::new();

    // 1) Grokipedia.
    tried.push("grokipedia".to_string());
    match grokipedia::search_and_fetch_top(fetcher, query) {
        Ok((hit, text)) => {
            // Truncate the article body to a manageable excerpt size.
            // The proposal layer will further sentence-split.
            let snippet = head_chars(&text, 4000);
            evidence.push(ResearchEvidence {
                source_url: hit.url,
                excerpt: snippet,
                provider: ResearchProvider::Grokipedia,
            });
        }
        Err(_) => {
            // Fall through to DDG.
        }
    }

    if evidence.len() >= cfg.max_excerpts_per_proposal {
        return Ok(evidence);
    }

    // 2) DDG Instant Answer.
    tried.push("ddg-instant-answer".to_string());
    if let Ok(hit) = duckduckgo::instant_answer(fetcher, query) {
        evidence.push(ResearchEvidence {
            source_url: hit.url.clone(),
            excerpt: hit.snippet.clone(),
            provider: ResearchProvider::DuckDuckGo,
        });
    }

    if evidence.len() >= cfg.max_excerpts_per_proposal {
        return Ok(evidence);
    }

    // 3) DDG HTML fallback. Pulls a couple of result pages and uses
    // their snippets directly without further fetching — keeps the
    // total request count for one research call small.
    //
    // Port T.8 (constraint C2): partition DDG hits into non-Wikipedia
    // and Wikipedia buckets, and prefer the non-Wikipedia bucket. The
    // C2 invariant (Grokipedia first, DDG fallback) is satisfied by
    // the router order above; the leak previously seen in the T.6
    // demo (6/79 pending entries from en.wikipedia.org) came from DDG
    // returning Wikipedia URLs as the top hits. Wikipedia is allowed
    // as a final-fallback inside DDG only when no non-Wikipedia
    // snippet survived.
    tried.push("ddg-html".to_string());
    if let Ok(hits) = duckduckgo::html_search(fetcher, query, 5) {
        let (non_wiki, wiki): (Vec<_>, Vec<_>) =
            hits.into_iter().partition(|h| !is_wikipedia_url(&h.url));
        for h in non_wiki.into_iter().chain(wiki.into_iter()) {
            if evidence.len() >= cfg.max_excerpts_per_proposal {
                break;
            }
            if h.snippet.trim().is_empty() {
                continue;
            }
            evidence.push(ResearchEvidence {
                source_url: h.url,
                excerpt: h.snippet,
                provider: ResearchProvider::DuckDuckGo,
            });
        }
    }

    if evidence.is_empty() {
        return Err(RouterError::AllProvidersFailed { tried });
    }
    Ok(evidence)
}

/// Port T.8 (constraint C2): a URL belongs to Wikipedia iff its host
/// ends in `wikipedia.org`. Any subdomain (`en.`, `simple.`, ...)
/// counts. Used by `run_router` to deprioritize Wikipedia hits inside
/// the DuckDuckGo fallback so they only land if no non-Wikipedia hit
/// survived. NOT a hard ban — Wikipedia is a legitimate final
/// fallback per John's preference.
pub fn is_wikipedia_url(url: &str) -> bool {
    let after_scheme = url.split_once("://").map(|(_, r)| r).unwrap_or(url);
    let host = after_scheme
        .split(|c: char| c == '/' || c == '?' || c == '#')
        .next()
        .unwrap_or("");
    let host = host.to_ascii_lowercase();
    host == "wikipedia.org" || host.ends_with(".wikipedia.org")
}

fn head_chars(s: &str, max: usize) -> String {
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i >= max {
            break;
        }
        out.push(c);
    }
    out
}

/// Fingerprint a piece of text to a small float vector. Used by the
/// scar-fingerprint check on bond candidates.
///
/// Algorithm: lowercase, split on whitespace, take the first 32 token
/// hashes mod 1.0. Crude but stable; the exact function doesn't matter
/// as long as similar inputs map to similar vectors and the brain uses
/// the same function on both the candidate and the scar at storage
/// time. See `proposal::cosine_similarity`.
pub fn fingerprint_text(text: &str) -> Vec<f32> {
    let mut out = vec![0.0_f32; 32];
    let lower = text.to_lowercase();
    for tok in lower.split_whitespace().take(64) {
        let h = simple_hash(tok);
        let bucket = (h % 32) as usize;
        out[bucket] += 1.0;
    }
    // Normalize to unit length so cosine similarity is meaningful.
    let mag: f32 = out.iter().map(|v| v * v).sum::<f32>().sqrt();
    if mag > 0.0 {
        for v in out.iter_mut() {
            *v /= mag;
        }
    }
    out
}

fn simple_hash(s: &str) -> u64 {
    let mut h: u64 = 1469598103934665603;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(1099511628211);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::research::http::MockFetcher;

    fn cfg() -> ResearchConfig {
        ResearchConfig::default()
    }

    fn grok_search(query: &str) -> String {
        format!("https://grokipedia.com/search?q={}", query.replace(' ', "+"))
    }

    fn ddg_ia(query: &str) -> String {
        format!(
            "https://api.duckduckgo.com/?format=json&no_redirect=1&q={}",
            query.replace(' ', "+")
        )
    }

    fn ddg_html(query: &str) -> String {
        format!(
            "https://html.duckduckgo.com/html/?q={}",
            query.replace(' ', "+")
        )
    }

    #[test]
    fn test_router_grokipedia_first_when_both_succeed() {
        let m = MockFetcher::new();
        m.route_ok(
            &grok_search("rust"),
            200,
            r#"<html><body><a href="/page/Rust">Rust</a></body></html>"#,
        );
        m.route_ok(
            "https://grokipedia.com/page/Rust",
            200,
            "<html><body><main><p>Rust is a language.</p></main></body></html>",
        );
        m.route_ok(
            &ddg_ia("rust"),
            200,
            r#"{"AbstractText":"backup","Heading":"Rust"}"#,
        );
        let cfg = cfg();
        let ev = run_router(&m, &cfg, "rust").unwrap();
        // Grokipedia should be the first evidence item.
        assert_eq!(ev[0].provider, ResearchProvider::Grokipedia);
    }

    #[test]
    fn test_router_falls_back_to_ddg_when_grok_empty() {
        let m = MockFetcher::new();
        m.route_ok(
            &grok_search("xyzzy"),
            200,
            "<html><body>no matches</body></html>",
        );
        m.route_ok(
            &ddg_ia("xyzzy"),
            200,
            r#"{"AbstractText":"DDG knew it.","Heading":"XYZZY","AbstractURL":"https://ex"}"#,
        );
        let cfg = cfg();
        let ev = run_router(&m, &cfg, "xyzzy").unwrap();
        assert_eq!(ev.len(), 1);
        assert_eq!(ev[0].provider, ResearchProvider::DuckDuckGo);
        assert!(ev[0].excerpt.contains("DDG knew it"));
    }

    #[test]
    fn test_router_falls_back_to_ddg_html_when_ia_empty() {
        let m = MockFetcher::new();
        m.route_ok(
            &grok_search("xyzzy"),
            200,
            "<html><body>no matches</body></html>",
        );
        m.route_ok(&ddg_ia("xyzzy"), 200, r#"{"AbstractText":""}"#);
        m.route_ok(
            &ddg_html("xyzzy"),
            200,
            r##"<html><body>
                <a class="result__a" href="/l/?uddg=https%3A%2F%2Fa.example%2F">A</a>
                <a class="result__snippet" href="#">snippet A</a>
                <a class="result__a" href="/l/?uddg=https%3A%2F%2Fb.example%2F">B</a>
                <a class="result__snippet" href="#">snippet B</a>
            </body></html>"##,
        );
        let cfg = cfg();
        let ev = run_router(&m, &cfg, "xyzzy").unwrap();
        assert!(!ev.is_empty());
        assert_eq!(ev[0].provider, ResearchProvider::DuckDuckGo);
    }

    #[test]
    fn test_router_all_fail_errs() {
        let m = MockFetcher::new();
        m.route_ok(&grok_search("xyzzy"), 200, "<html></html>");
        m.route_ok(&ddg_ia("xyzzy"), 200, r#"{"AbstractText":""}"#);
        m.route_ok(&ddg_html("xyzzy"), 200, "<html></html>");
        let cfg = cfg();
        let err = run_router(&m, &cfg, "xyzzy").unwrap_err();
        match err {
            RouterError::AllProvidersFailed { tried } => {
                assert!(tried.contains(&"grokipedia".to_string()));
                assert!(tried.contains(&"ddg-instant-answer".to_string()));
                assert!(tried.contains(&"ddg-html".to_string()));
            }
        }
    }

    #[test]
    fn test_fingerprint_is_unit_normalized() {
        let f = fingerprint_text("rust is a memory safe systems programming language");
        let mag: f32 = f.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((mag - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_fingerprint_similarity_is_meaningful() {
        let a = fingerprint_text("rust is a memory safe language");
        let b = fingerprint_text("rust is a memory safe systems language");
        let c = fingerprint_text("kittens are fluffy soft mammals");
        let sim_ab: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        let sim_ac: f32 = a.iter().zip(&c).map(|(x, y)| x * y).sum();
        assert!(sim_ab > sim_ac);
    }

    /// Port T.8 (constraint C2): even when Grokipedia returns no
    /// useful content AND DDG returns a mix of Wikipedia + non-
    /// Wikipedia results, the non-Wikipedia hit must come first in
    /// the evidence list. This is the structural fix for the 6/79
    /// wikipedia leak John observed in the T.6 demo.
    #[test]
    fn t8_grokopedia_queried_before_wikipedia_in_research_pipeline() {
        let m = MockFetcher::new();
        // Grokipedia returns nothing -> DDG fallback fires.
        m.route_ok(&grok_search("physics"), 200, "<html><body>no matches</body></html>");
        m.route_ok(&ddg_ia("physics"), 200, r#"{"AbstractText":""}"#);
        // DDG html returns a wikipedia link FIRST, a non-wikipedia link SECOND.
        m.route_ok(
            &ddg_html("physics"),
            200,
            r##"<html><body>
                <a class="result__a" href="/l/?uddg=https%3A%2F%2Fen.wikipedia.org%2Fwiki%2FPhysics">Wiki</a>
                <a class="result__snippet" href="#">leaked wiki snippet</a>
                <a class="result__a" href="/l/?uddg=https%3A%2F%2Fphys.example%2Fpage">Phys</a>
                <a class="result__snippet" href="#">clean physics snippet</a>
            </body></html>"##,
        );
        let cfg = cfg();
        let ev = run_router(&m, &cfg, "physics").unwrap();
        assert!(!ev.is_empty());
        // First hit must NOT be wikipedia.
        assert!(
            !is_wikipedia_url(&ev[0].source_url),
            "first DDG-fallback hit should be non-wikipedia, got {}",
            ev[0].source_url
        );
        // The wikipedia hit may still be present (it is a legitimate
        // last-resort fallback) but only after the non-wikipedia hit.
        let wiki_idx = ev.iter().position(|e| is_wikipedia_url(&e.source_url));
        let non_idx = ev.iter().position(|e| !is_wikipedia_url(&e.source_url));
        if let (Some(w), Some(n)) = (wiki_idx, non_idx) {
            assert!(n < w, "non-wiki must precede wiki in evidence list");
        }
    }

    #[test]
    fn t8_is_wikipedia_url_classifier() {
        assert!(is_wikipedia_url("https://en.wikipedia.org/wiki/X"));
        assert!(is_wikipedia_url("http://simple.wikipedia.org/wiki/Y"));
        assert!(is_wikipedia_url("https://wikipedia.org/wiki/Z"));
        assert!(!is_wikipedia_url("https://grokipedia.com/page/X"));
        assert!(!is_wikipedia_url("https://phys.example/page"));
    }
}
