//! DuckDuckGo client. Two endpoints, in fallback order:
//!
//! 1. Instant Answer API: https://api.duckduckgo.com/?q=<q>&format=json
//!    Returns a JSON object with `Abstract`, `AbstractText`,
//!    `AbstractURL`, and a `RelatedTopics` array. When the abstract is
//!    non-empty, this is the cheapest, cleanest result.
//! 2. HTML fallback: https://html.duckduckgo.com/html/?q=<q>
//!    No-JS results page. Result links live in `a.result__a` elements.
//!    The links are wrapped — DDG redirects through `/l/?uddg=<url>`
//!    so we extract the final URL from the `uddg` query parameter.
//!
//! Both endpoints are part of DDG's documented public surface; no API
//! key is required.

use crate::research::http::{HttpError, HttpFetcher};
use scraper::{Html, Selector};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DuckDuckGoHit {
    pub title: String,
    pub url: String,
    pub snippet: String,
}

#[derive(Debug, Clone)]
pub enum DuckDuckGoError {
    Http(HttpError),
    NoResults,
    ParseFailed(String),
}

impl std::fmt::Display for DuckDuckGoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DuckDuckGoError::Http(e) => write!(f, "ddg http: {}", e),
            DuckDuckGoError::NoResults => write!(f, "ddg: no results"),
            DuckDuckGoError::ParseFailed(m) => write!(f, "ddg parse failed: {}", m),
        }
    }
}

impl std::error::Error for DuckDuckGoError {}

const IA_BASE: &str = "https://api.duckduckgo.com/?format=json&no_redirect=1&q=";
const HTML_BASE: &str = "https://html.duckduckgo.com/html/?q=";

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

/// Query the Instant Answer API. Returns a single hit when DDG has an
/// abstract for the query; `NoResults` otherwise.
pub fn instant_answer(
    fetcher: &dyn HttpFetcher,
    query: &str,
) -> Result<DuckDuckGoHit, DuckDuckGoError> {
    let url = format!("{}{}", IA_BASE, encode_query(query));
    let resp = fetcher.fetch(&url).map_err(DuckDuckGoError::Http)?;
    if resp.status >= 400 {
        return Err(DuckDuckGoError::Http(HttpError::Status {
            status: resp.status,
            body: String::new(),
        }));
    }
    let v: serde_json::Value = serde_json::from_str(&resp.body)
        .map_err(|e| DuckDuckGoError::ParseFailed(e.to_string()))?;
    let abstract_text = v
        .get("AbstractText")
        .and_then(|x| x.as_str())
        .unwrap_or("")
        .to_string();
    if abstract_text.trim().is_empty() {
        return Err(DuckDuckGoError::NoResults);
    }
    let title = v
        .get("Heading")
        .and_then(|x| x.as_str())
        .unwrap_or(query)
        .to_string();
    let abstract_url = v
        .get("AbstractURL")
        .and_then(|x| x.as_str())
        .unwrap_or("")
        .to_string();
    Ok(DuckDuckGoHit {
        title,
        url: abstract_url,
        snippet: abstract_text,
    })
}

/// HTML fallback. Used when the Instant Answer API returns nothing
/// useful. Returns up to `max` results parsed from the no-JS page.
pub fn html_search(
    fetcher: &dyn HttpFetcher,
    query: &str,
    max: usize,
) -> Result<Vec<DuckDuckGoHit>, DuckDuckGoError> {
    if max == 0 {
        return Ok(Vec::new());
    }
    let url = format!("{}{}", HTML_BASE, encode_query(query));
    let resp = fetcher.fetch(&url).map_err(DuckDuckGoError::Http)?;
    if resp.status >= 400 {
        return Err(DuckDuckGoError::Http(HttpError::Status {
            status: resp.status,
            body: String::new(),
        }));
    }
    let doc = Html::parse_document(&resp.body);
    let result_sel = Selector::parse("a.result__a")
        .map_err(|e| DuckDuckGoError::ParseFailed(format!("{:?}", e)))?;
    let snippet_sel = Selector::parse("a.result__snippet, .result__snippet")
        .map_err(|e| DuckDuckGoError::ParseFailed(format!("{:?}", e)))?;

    let snippets: Vec<String> = doc
        .select(&snippet_sel)
        .map(|e| e.text().collect::<String>().trim().to_string())
        .collect();

    let mut hits: Vec<DuckDuckGoHit> = Vec::new();
    for (i, el) in doc.select(&result_sel).enumerate() {
        let raw_href = el.value().attr("href").unwrap_or("").to_string();
        let title = el.text().collect::<String>().trim().to_string();
        let url = unwrap_ddg_redirect(&raw_href);
        let snippet = snippets.get(i).cloned().unwrap_or_default();
        if !title.is_empty() && !url.is_empty() {
            hits.push(DuckDuckGoHit { title, url, snippet });
        }
        if hits.len() >= max {
            break;
        }
    }
    if hits.is_empty() {
        return Err(DuckDuckGoError::NoResults);
    }
    Ok(hits)
}

/// DDG result links look like `/l/?kh=-1&uddg=https%3A%2F%2Fexample.com`.
/// Pull the `uddg` value if present; otherwise return the input
/// unchanged. Decodes basic percent-encoding.
fn unwrap_ddg_redirect(href: &str) -> String {
    let needle = "uddg=";
    let idx = match href.find(needle) {
        Some(i) => i + needle.len(),
        None => return href.to_string(),
    };
    let tail = &href[idx..];
    let end = tail.find('&').unwrap_or(tail.len());
    let raw = &tail[..end];
    percent_decode(raw)
}

fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let Ok(byte) = u8::from_str_radix(
                std::str::from_utf8(&bytes[i + 1..i + 3]).unwrap_or("00"),
                16,
            ) {
                out.push(byte);
                i += 3;
                continue;
            }
        }
        if bytes[i] == b'+' {
            out.push(b' ');
        } else {
            out.push(bytes[i]);
        }
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::research::http::MockFetcher;

    #[test]
    fn test_unwrap_ddg_redirect() {
        let h = "/l/?kh=-1&uddg=https%3A%2F%2Fexample.com%2Fpath&rut=foo";
        assert_eq!(unwrap_ddg_redirect(h), "https://example.com/path");
        assert_eq!(unwrap_ddg_redirect("https://direct.example/"), "https://direct.example/");
    }

    #[test]
    fn test_instant_answer_parses_abstract() {
        let m = MockFetcher::new();
        m.route_ok(
            "https://api.duckduckgo.com/?format=json&no_redirect=1&q=rust",
            200,
            r#"{"AbstractText":"Rust is a language.","Heading":"Rust","AbstractURL":"https://en.wikipedia.org/wiki/Rust"}"#,
        );
        let hit = instant_answer(&m, "rust").unwrap();
        assert_eq!(hit.title, "Rust");
        assert_eq!(hit.snippet, "Rust is a language.");
        assert_eq!(hit.url, "https://en.wikipedia.org/wiki/Rust");
    }

    #[test]
    fn test_instant_answer_empty_abstract_no_results() {
        let m = MockFetcher::new();
        m.route_ok(
            "https://api.duckduckgo.com/?format=json&no_redirect=1&q=zzz",
            200,
            r#"{"AbstractText":"","Heading":""}"#,
        );
        let err = instant_answer(&m, "zzz").unwrap_err();
        assert!(matches!(err, DuckDuckGoError::NoResults));
    }

    #[test]
    fn test_html_search_parses_results() {
        let m = MockFetcher::new();
        m.route_ok(
            "https://html.duckduckgo.com/html/?q=rust",
            200,
            r##"<html><body>
                <div class="result">
                    <a class="result__a" href="/l/?kh=-1&uddg=https%3A%2F%2Frust-lang.org%2F">The Rust Language</a>
                    <a class="result__snippet" href="#">A safe systems language.</a>
                </div>
                <div class="result">
                    <a class="result__a" href="/l/?kh=-1&uddg=https%3A%2F%2Fdoc.rust-lang.org%2F">Rust Docs</a>
                    <a class="result__snippet" href="#">Documentation hub.</a>
                </div>
            </body></html>"##,
        );
        let hits = html_search(&m, "rust", 5).unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].url, "https://rust-lang.org/");
        assert_eq!(hits[0].title, "The Rust Language");
        assert_eq!(hits[0].snippet, "A safe systems language.");
    }

    #[test]
    fn test_html_search_no_results_errs() {
        let m = MockFetcher::new();
        m.route_ok("https://html.duckduckgo.com/html/?q=zzz", 200, "<html></html>");
        let err = html_search(&m, "zzz", 5).unwrap_err();
        assert!(matches!(err, DuckDuckGoError::NoResults));
    }

    /// Live network.
    #[test]
    #[ignore]
    fn test_duckduckgo_instant_answer_real_live() {
        let f = crate::research::http::UreqFetcher::new(
            10_000,
            "LuAi/0.1 (research-bot; info@dialectforge.com)",
        );
        let hit = instant_answer(&f, "what is rust programming language")
            .expect("DDG IA should answer this");
        assert!(!hit.snippet.is_empty());
    }
}
