//! HTTP fetch trait + the production `ureq` implementation +
//! a `MockFetcher` for tests.
//!
//! All research-loop network calls go through `HttpFetcher`. The
//! production path uses `ureq` (synchronous, no tokio runtime — matches
//! the rest of Lu). Tests inject `MockFetcher` to make routing /
//! extraction / proposal logic deterministic without hitting the wire.
//!
//! Per the port-spec "NO STUBS ANYWHERE" rule: extraction, proposal
//! synthesis, and the scar-fingerprint check all run on REAL bytes
//! against REAL HTML in their own tests; only the network boundary
//! is mockable, and even then several tests (`#[ignore]`-able for CI)
//! exercise the live endpoints to prove the stack end-to-end.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Duration;

/// One fetched response. `body` is decoded as UTF-8 (lossy on bad bytes,
/// matching browser behavior closely enough for prose extraction).
#[derive(Debug, Clone)]
pub struct HttpResponse {
    pub status: u16,
    pub body: String,
    pub final_url: String,
}

/// Network failure modes. Everything outside the 2xx range is folded
/// into `Status` so callers can distinguish "server said no" from
/// "never reached the server".
#[derive(Debug, Clone)]
pub enum HttpError {
    Timeout,
    Network(String),
    InvalidUrl(String),
    Status { status: u16, body: String },
}

impl std::fmt::Display for HttpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HttpError::Timeout => write!(f, "http timeout"),
            HttpError::Network(m) => write!(f, "http network error: {}", m),
            HttpError::InvalidUrl(u) => write!(f, "invalid url: {}", u),
            HttpError::Status { status, .. } => write!(f, "http status {}", status),
        }
    }
}

impl std::error::Error for HttpError {}

/// Trait boundary for the network. Production: `UreqFetcher`. Tests:
/// `MockFetcher`.
pub trait HttpFetcher: Send + Sync {
    fn fetch(&self, url: &str) -> Result<HttpResponse, HttpError>;
}

/// Real network fetcher. Synchronous `ureq` agent with a configured
/// timeout and user-agent string. Honors HTTP redirects (ureq default
/// is up to 5 hops).
pub struct UreqFetcher {
    agent: ureq::Agent,
    user_agent: String,
}

impl UreqFetcher {
    pub fn new(timeout_ms: u64, user_agent: impl Into<String>) -> Self {
        let timeout = Duration::from_millis(timeout_ms);
        let agent = ureq::AgentBuilder::new()
            .timeout_connect(timeout)
            .timeout_read(timeout)
            .timeout_write(timeout)
            .build();
        UreqFetcher {
            agent,
            user_agent: user_agent.into(),
        }
    }
}

impl HttpFetcher for UreqFetcher {
    fn fetch(&self, url: &str) -> Result<HttpResponse, HttpError> {
        let req = self.agent.get(url).set("User-Agent", &self.user_agent);
        match req.call() {
            Ok(resp) => {
                let status = resp.status();
                let final_url = resp.get_url().to_string();
                let body = resp
                    .into_string()
                    .map_err(|e| HttpError::Network(e.to_string()))?;
                Ok(HttpResponse { status, body, final_url })
            }
            Err(ureq::Error::Status(code, resp)) => {
                let body = resp.into_string().unwrap_or_default();
                Err(HttpError::Status { status: code, body })
            }
            Err(ureq::Error::Transport(t)) => {
                let kind = format!("{:?}", t.kind());
                if kind.contains("InvalidUrl") {
                    Err(HttpError::InvalidUrl(url.to_string()))
                } else {
                    Err(HttpError::Network(t.to_string()))
                }
            }
        }
    }
}

/// Test-only fetcher backed by a URL → response map. Calls to URLs
/// not in the map return `HttpError::Network("unrouted")` so a test
/// failure is loud rather than silent.
pub struct MockFetcher {
    responses: Mutex<HashMap<String, Result<HttpResponse, HttpError>>>,
    pub call_log: Mutex<Vec<String>>,
}

impl MockFetcher {
    pub fn new() -> Self {
        MockFetcher {
            responses: Mutex::new(HashMap::new()),
            call_log: Mutex::new(Vec::new()),
        }
    }

    pub fn route(&self, url: &str, response: Result<HttpResponse, HttpError>) {
        self.responses
            .lock()
            .unwrap()
            .insert(url.to_string(), response);
    }

    pub fn route_ok(&self, url: &str, status: u16, body: &str) {
        self.route(
            url,
            Ok(HttpResponse {
                status,
                body: body.to_string(),
                final_url: url.to_string(),
            }),
        );
    }

    pub fn route_empty_ok(&self, url: &str) {
        self.route_ok(url, 200, "");
    }

    pub fn calls(&self) -> Vec<String> {
        self.call_log.lock().unwrap().clone()
    }
}

impl HttpFetcher for MockFetcher {
    fn fetch(&self, url: &str) -> Result<HttpResponse, HttpError> {
        self.call_log.lock().unwrap().push(url.to_string());
        match self.responses.lock().unwrap().get(url).cloned() {
            Some(r) => r,
            None => Err(HttpError::Network(format!(
                "unrouted in MockFetcher: {}",
                url
            ))),
        }
    }
}

impl Default for MockFetcher {
    fn default() -> Self {
        MockFetcher::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_fetcher_routes_and_logs() {
        let m = MockFetcher::new();
        m.route_ok("https://example.com/a", 200, "hi");
        let r = m.fetch("https://example.com/a").unwrap();
        assert_eq!(r.status, 200);
        assert_eq!(r.body, "hi");
        assert_eq!(m.calls(), vec!["https://example.com/a".to_string()]);
    }

    #[test]
    fn test_mock_fetcher_unrouted_errs() {
        let m = MockFetcher::new();
        let err = m.fetch("https://example.com/missing").unwrap_err();
        assert!(matches!(err, HttpError::Network(_)));
    }

    /// Live network. Marked `#[ignore]` so CI without internet keeps
    /// passing; run with `cargo test --release -- --ignored`.
    #[test]
    #[ignore]
    fn test_http_fetch_real_url_live() {
        let f = UreqFetcher::new(10_000, "LuAi/0.1 (research-bot; info@dialectforge.com)");
        let r = f
            .fetch("https://example.com/")
            .expect("example.com should resolve");
        assert_eq!(r.status, 200);
        assert!(r.body.to_lowercase().contains("example domain"));
    }
}
