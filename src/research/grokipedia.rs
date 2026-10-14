//! Grokipedia client. xAI's encyclopedia at https://grokipedia.com.
//!
//! Endpoint situation (probed 2026-04-25, John's Lu v4 Port #10):
//! - There is NO documented public API. `robots.txt` only disallows
//!   `/api/`, so we do NOT touch `/api/`.
//! - The public surface is server-rendered HTML. `/search?q=<query>`
//!   returns a result page with `<a href="/page/<title>">…</a>` links.
//! - `/page/<title>` returns full article HTML including an `<article>`
//!   element with prose. Excerpts are also exposed in `og:description`.
//!
//! We use the public HTML pages with a clearly-identifying user-agent.
//! The router invokes us at most a handful of times per research call
//! and the loop driver inserts a 1-second pause between sequential
//! Grokipedia fetches, satisfying the spec's rate-limit requirement.

use crate::research::extract::extract_clean_text;
use crate::research::http::{HttpError, HttpFetcher};
use scraper::{Html, Selector};

/// One Grokipedia search hit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GrokipediaHit {
    pub title: String,
    pub url: String,
    /// Optional snippet from `og:description` of the destination page,
    /// or from the search-result card if exposed inline. Filled by
    /// `fetch_excerpts`, not by `search`.
    pub excerpt: Option<String>,
}

/// Failure modes for Grokipedia operations. Carries the underlying
/// `HttpError` for diagnostics.
#[derive(Debug, Clone)]
pub enum GrokipediaError {
    Http(HttpError),
    NoResults,
    ParseFailed(String),
}

impl std::fmt::Display for GrokipediaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GrokipediaError::Http(e) => write!(f, "grokipedia http: {}", e),
            GrokipediaError::NoResults => write!(f, "grokipedia: no results"),
            GrokipediaError::ParseFailed(m) => write!(f, "grokipedia parse failed: {}", m),
        }
    }
}

impl std::error::Error for GrokipediaError {}

const SEARCH_BASE: &str = "https://grokipedia.com/search?q=";
const PAGE_BASE: &str = "https://grokipedia.com";

/// URL-encode a query for the search endpoint. Spaces → `+`, plus the
/// usual percent-encoding for reserved chars. Implemented inline so we
/// don't pull in a `url` crate dep at the call surface.
fn encode_query(q: &str) -> String {
    let mut out = String::new();
    for c in q.chars() {
        match c {
            ' ' => out.push('+'),
            'a'..='z' | 'A'..='Z' | '0'..='9' | '-' | '_' | '.' | '~' => out.push(c),
            _ => {
                let mut buf = [0u8; 4];
                for &b in c.encode_utf8(&mut buf).as_bytes() {
                    out.push_str(&format!("%{:02X}", b));
                }
            }
        }
    }
    out
}

/// Search Grokipedia. Returns up to `max` hits. The search URL is
/// constructed from the query; results are scraped from the static
/// HTML response.
pub fn search(
    fetcher: &dyn HttpFetcher,
    query: &str,
    max: usize,
) -> Result<Vec<GrokipediaHit>, GrokipediaError> {
    if max == 0 {
        return Ok(Vec::new());
    }
    let url = format!("{}{}", SEARCH_BASE, encode_query(query));
    let resp = fetcher.fetch(&url).map_err(GrokipediaError::Http)?;
    if resp.status >= 400 {
        return Err(GrokipediaError::Http(HttpError::Status {
            status: resp.status,
            body: String::new(),
        }));
    }
    let doc = Html::parse_document(&resp.body);
    let sel = Selector::parse(r#"a[href^="/page/"]"#)
        .map_err(|e| GrokipediaError::ParseFailed(format!("{:?}", e)))?;

    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut hits: Vec<GrokipediaHit> = Vec::new();
    for el in doc.select(&sel) {
        let href = match el.value().attr("href") {
            Some(h) => h.to_string(),
            None => continue,
        };
        if !seen.insert(href.clone()) {
            continue;
        }
        let title_text = el.text().collect::<String>().trim().to_string();
        // Fall back to the slug if the link text was empty (icon-only links).
        let title = if title_text.is_empty() {
            href.trim_start_matches("/page/")
                .replace('_', " ")
                .replace("%20", " ")
        } else {
            title_text
        };
        let url = format!("{}{}", PAGE_BASE, href);
        hits.push(GrokipediaHit { title, url, excerpt: None });
        if hits.len() >= max {
            break;
        }
    }
    if hits.is_empty() {
        return Err(GrokipediaError::NoResults);
    }
    Ok(hits)
}

/// Fetch a single Grokipedia article and return clean prose. The
/// returned string is suitable for direct sentence-splitting via
/// `extract::split_sentences`.
pub fn fetch_article_text(
    fetcher: &dyn HttpFetcher,
    page_url: &str,
) -> Result<String, GrokipediaError> {
    let resp = fetcher.fetch(page_url).map_err(GrokipediaError::Http)?;
    if resp.status >= 400 {
        return Err(GrokipediaError::Http(HttpError::Status {
            status: resp.status,
            body: String::new(),
        }));
    }
    let text = extract_clean_text(&resp.body);
    if text.trim().is_empty() {
        return Err(GrokipediaError::ParseFailed(
            "no extractable text".to_string(),
        ));
    }
    Ok(text)
}

/// Convenience: search + fetch the top hit's article text in one call.
/// Returns `(hit, article_text)`. Used by the router when it commits
/// to Grokipedia as the source.
pub fn search_and_fetch_top(
    fetcher: &dyn HttpFetcher,
    query: &str,
) -> Result<(GrokipediaHit, String), GrokipediaError> {
    let hits = search(fetcher, query, 5)?;
    let top = hits.into_iter().next().ok_or(GrokipediaError::NoResults)?;
    let text = fetch_article_text(fetcher, &top.url)?;
    Ok((top, text))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::research::http::MockFetcher;

    fn search_html() -> &'static str {
        r#"<html><body>
            <main>
              <ul>
                <li><a href="/page/Rust_(programming_language)">Rust (programming language)</a></li>
                <li><a href="/page/Rust_for_Linux">Rust for Linux</a></li>
                <li><a href="/page/Programming_language">Programming language</a></li>
              </ul>
            </main>
          </body></html>"#
    }

    fn article_html() -> &'static str {
        r#"<html><head><script>nope</script></head>
            <body>
              <nav>SKIP</nav>
              <article>
                <h1>Rust (programming language)</h1>
                <p>Rust is a memory-safe systems programming language.</p>
                <p>Rust uses ownership and borrowing.</p>
              </article>
            </body></html>"#
    }

    #[test]
    fn test_encode_query_basic() {
        assert_eq!(encode_query("hello world"), "hello+world");
        assert_eq!(encode_query("a&b"), "a%26b");
    }

    #[test]
    fn test_search_parses_page_links() {
        let m = MockFetcher::new();
        m.route_ok(
            "https://grokipedia.com/search?q=rust",
            200,
            search_html(),
        );
        let hits = search(&m, "rust", 5).unwrap();
        assert_eq!(hits.len(), 3);
        assert_eq!(hits[0].title, "Rust (programming language)");
        assert!(hits[0].url.starts_with("https://grokipedia.com/page/"));
    }

    #[test]
    fn test_search_dedupes() {
        let m = MockFetcher::new();
        m.route_ok(
            "https://grokipedia.com/search?q=rust",
            200,
            r#"<html><body>
                <a href="/page/X">X1</a>
                <a href="/page/X">X2</a>
                <a href="/page/Y">Y</a>
            </body></html>"#,
        );
        let hits = search(&m, "rust", 5).unwrap();
        assert_eq!(hits.len(), 2);
    }

    #[test]
    fn test_search_empty_returns_no_results_err() {
        let m = MockFetcher::new();
        m.route_ok(
            "https://grokipedia.com/search?q=rust",
            200,
            "<html><body>nothing here</body></html>",
        );
        assert!(matches!(
            search(&m, "rust", 5).unwrap_err(),
            GrokipediaError::NoResults
        ));
    }

    #[test]
    fn test_fetch_article_text_returns_clean_prose() {
        let m = MockFetcher::new();
        m.route_ok(
            "https://grokipedia.com/page/Rust_(programming_language)",
            200,
            article_html(),
        );
        let txt = fetch_article_text(
            &m,
            "https://grokipedia.com/page/Rust_(programming_language)",
        )
        .unwrap();
        assert!(txt.contains("memory-safe systems programming language"));
        assert!(!txt.contains("nope"));
        assert!(!txt.contains("SKIP"));
    }

    #[test]
    fn test_search_and_fetch_top_combines() {
        let m = MockFetcher::new();
        m.route_ok(
            "https://grokipedia.com/search?q=rust",
            200,
            search_html(),
        );
        m.route_ok(
            "https://grokipedia.com/page/Rust_(programming_language)",
            200,
            article_html(),
        );
        let (hit, txt) = search_and_fetch_top(&m, "rust").unwrap();
        assert_eq!(hit.title, "Rust (programming language)");
        assert!(txt.contains("memory-safe"));
    }

    /// Live network. Tagged `#[ignore]` for CI hygiene; run with
    /// `cargo test --release -- --ignored`.
    #[test]
    #[ignore]
    fn test_grokipedia_query_real_live() {
        let f = crate::research::http::UreqFetcher::new(
            10_000,
            "LuAi/0.1 (research-bot; info@dialectforge.com)",
        );
        let hits = search(&f, "rust programming language", 5)
            .expect("grokipedia should return at least one result for 'rust'");
        assert!(!hits.is_empty());
    }
}
