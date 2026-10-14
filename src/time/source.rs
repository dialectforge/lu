//! `ClockSource` trait — the swappable time backend.
//!
//! Two impls:
//!  - `SystemClockSource`: OS clock, unsigned. Default in dev; same as
//!    pre-T.1 behaviour.
//!  - `DFTimeServiceSource`: T.1 shipped a stub; **T.1.1 lights up the
//!    real client** — periodic HTTP GET against the DF Time Service,
//!    Ed25519 signature verification, drift check, fallback to
//!    monotonic projection on failure.
//!
//! ### T.1.1 wiring
//!
//! `DFTimeServiceSource::new_real(config)` spawns a background polling
//! thread (see `df_client::DFRealClient`). `fetch()` returns the
//! monotonic-projected anchor from the most recent successful poll, or
//! `ClockError::FetchFailed` if no fetch has succeeded yet (caller
//! falls back to `SystemClockSource` on that error during the initial
//! window).
//!
//! `DFTimeServiceSource::stub()` is preserved so T.1's existing tests
//! and the `--time-source df` CLI flag (with no endpoint) continue to
//! work — the stub returns OS time with a deterministic fake signature
//! and is the right primitive for tests that exercise the
//! authenticated-source code path WITHOUT requiring a live endpoint.

use super::anomaly::AnomalyDetector;
use super::authenticated::{AuthenticatedTime, Signature, SourceId};
use super::config::TimeConfig;
use super::df_client::{DFRealClient, DFTimeServiceClientConfig, DFRealState, TrustState};
use super::monotonic::MonotonicClock;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

/// Errors raised by clock-related operations across T.1.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClockError {
    /// The configured source could not produce an anchor (network down,
    /// stub disabled, etc.).
    FetchFailed(String),
    /// Scar-sensitive op refused because the brain's current anchor is
    /// not authenticated (default for `SystemClockSource`).
    TimeNotAuthenticated,
    /// The anomaly detector tripped on the most recent refresh; the
    /// brain is in degraded mode until a clean anchor lands.
    AnomalyDetected(String),
    /// The anchor passed its expiry without a successful refresh.
    AnchorExpired { age_ms: u64 },
}

impl std::fmt::Display for ClockError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ClockError::FetchFailed(m) => write!(f, "clock fetch failed: {}", m),
            ClockError::TimeNotAuthenticated => {
                write!(f, "time is not authenticated; scar-sensitive ops refused")
            }
            ClockError::AnomalyDetected(m) => write!(f, "clock anomaly: {}", m),
            ClockError::AnchorExpired { age_ms } => {
                write!(f, "anchor expired ({} ms past deadline)", age_ms)
            }
        }
    }
}

impl std::error::Error for ClockError {}

/// The swappable backend. T.1.1 lights up the real `DFTimeServiceSource`
/// behind the same trait; production code holds `Arc<dyn ClockSource>`
/// so the swap is a one-line config change.
pub trait ClockSource: Send + Sync {
    /// Fetch an authenticated time anchor.
    fn fetch(&self) -> Result<AuthenticatedTime, ClockError>;
    /// Human-readable id (`"system"`, `"df-time-service-stub"`,
    /// `"df-time-service"`, …).
    fn name(&self) -> &str;
    /// True iff this source returns signed anchors that downstream
    /// scar-sensitive ops accept as authenticated.
    fn is_authenticated(&self) -> bool;
}

/// Default source. Reads `SystemTime::now()`, never signs. Equivalent to
/// pre-T.1 behaviour.
pub struct SystemClockSource {
    cfg: TimeConfig,
}

impl SystemClockSource {
    pub fn new() -> Self {
        SystemClockSource {
            cfg: TimeConfig::default(),
        }
    }
    pub fn with_config(cfg: TimeConfig) -> Self {
        SystemClockSource { cfg }
    }
}

impl Default for SystemClockSource {
    fn default() -> Self {
        Self::new()
    }
}

impl ClockSource for SystemClockSource {
    fn fetch(&self) -> Result<AuthenticatedTime, ClockError> {
        let utc_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .map_err(|e| ClockError::FetchFailed(format!("system time before epoch: {}", e)))?;
        let mono = MonotonicClock::now();
        Ok(AuthenticatedTime::unsigned(
            utc_ms,
            mono,
            SourceId::system(),
            utc_ms.saturating_add(self.cfg.refresh_interval_ms),
        ))
    }
    fn name(&self) -> &str {
        "system"
    }
    fn is_authenticated(&self) -> bool {
        false
    }
}

/// DF Time Service backend. **T.1.1 ships the real client.** Three
/// constructors:
///  - `new(endpoint)` — back-compat shim used by `--time-source df`
///    when only an endpoint URL is supplied. Spawns the real client
///    using the placeholder pubkey baked at compile time.
///  - `new_real(config)` — the authoritative T.1.1 constructor.
///    Caller passes in the full `DFTimeServiceClientConfig` from
///    runtime config.
///  - `stub()` — T.1's in-process stub. Returns OS time with a
///    deterministic fake signature. Preserved for tests and for the
///    `--time-source df` flag without an endpoint.
pub struct DFTimeServiceSource {
    cfg: TimeConfig,
    inner: DFInner,
}

enum DFInner {
    Stub {
        endpoint: Option<String>,
    },
    Real {
        client: Arc<DFRealClient>,
    },
}

impl DFTimeServiceSource {
    /// Back-compat: T.1's signature, now lit up to spawn a real client
    /// against the placeholder pubkey. Kept so existing `--time-source
    /// df --df-time-endpoint URL` invocations work without further
    /// flags.
    pub fn new(endpoint: String) -> Self {
        let mut cfg = DFTimeServiceClientConfig::default();
        cfg.url = endpoint;
        Self::new_real(cfg)
    }

    /// T.1.1 authoritative constructor. Spawns the polling thread.
    pub fn new_real(client_cfg: DFTimeServiceClientConfig) -> Self {
        let cfg = client_cfg.time_cfg;
        let client = Arc::new(DFRealClient::spawn(client_cfg, None));
        DFTimeServiceSource {
            cfg,
            inner: DFInner::Real { client },
        }
    }

    /// T.1.1: spawn with an `AnomalyDetector` sink so failures surface
    /// in `Brain::warnings()`.
    pub fn new_real_with_anomaly_sink(
        client_cfg: DFTimeServiceClientConfig,
        sink: Arc<Mutex<AnomalyDetector>>,
    ) -> Self {
        let cfg = client_cfg.time_cfg;
        let client = Arc::new(DFRealClient::spawn(client_cfg, Some(sink)));
        DFTimeServiceSource {
            cfg,
            inner: DFInner::Real { client },
        }
    }

    /// In-process stub: real OS wall-clock + deterministic fake
    /// signature. T.1 tests use this; it is also the fall-back when
    /// `--time-source df` is requested without `--df-time-endpoint`.
    pub fn stub() -> Self {
        DFTimeServiceSource {
            cfg: TimeConfig::default(),
            inner: DFInner::Stub { endpoint: None },
        }
    }

    pub fn with_config(mut self, cfg: TimeConfig) -> Self {
        self.cfg = cfg;
        self
    }

    pub fn endpoint(&self) -> Option<&str> {
        match &self.inner {
            DFInner::Stub { endpoint } => endpoint.as_deref(),
            DFInner::Real { client } => Some(client.config.url.as_str()),
        }
    }

    /// T.1.1: snapshot of the polling-thread state. Returns `None` for
    /// stub mode.
    pub fn real_state_snapshot(&self) -> Option<DFRealState> {
        match &self.inner {
            DFInner::Stub { .. } => None,
            DFInner::Real { client } => Some(client.state_snapshot()),
        }
    }

    /// T.1.1: handle on the real client (for tests + status surface).
    pub fn real_client(&self) -> Option<Arc<DFRealClient>> {
        match &self.inner {
            DFInner::Stub { .. } => None,
            DFInner::Real { client } => Some(client.clone()),
        }
    }

    /// True iff the most recent poll trust state is `Trusted`.
    pub fn is_trusted(&self) -> bool {
        match self.real_state_snapshot() {
            Some(s) => matches!(s.trust_state, TrustState::Trusted),
            None => true, // stub is always considered trusted (it always returns).
        }
    }
}

impl ClockSource for DFTimeServiceSource {
    fn fetch(&self) -> Result<AuthenticatedTime, ClockError> {
        match &self.inner {
            DFInner::Stub { .. } => {
                let utc_ms = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map(|d| d.as_millis() as u64)
                    .map_err(|e| {
                        ClockError::FetchFailed(format!("system time before epoch: {}", e))
                    })?;
                let mono = MonotonicClock::now();
                Ok(AuthenticatedTime::signed(
                    utc_ms,
                    mono,
                    SourceId::df_time_service_stub(),
                    Signature::stub_for(utc_ms),
                    utc_ms.saturating_add(self.cfg.refresh_interval_ms),
                ))
            }
            DFInner::Real { client } => client.project_now(),
        }
    }
    fn name(&self) -> &str {
        match &self.inner {
            DFInner::Stub { .. } => "df-time-service-stub",
            DFInner::Real { .. } => "df-time-service",
        }
    }
    fn is_authenticated(&self) -> bool {
        // Stub and real both produce signed anchors. Real may return
        // `FetchFailed` until the first successful poll, but if a fetch
        // returns Ok the anchor IS authenticated.
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn system_source_unauthenticated() {
        let s = SystemClockSource::new();
        assert!(!s.is_authenticated());
        assert_eq!(s.name(), "system");
        let t = s.fetch().expect("fetch ok");
        assert!(t.signature.is_none());
        assert_eq!(t.fetched_from, SourceId::system());
    }

    #[test]
    fn df_stub_authenticated() {
        let d = DFTimeServiceSource::stub();
        assert!(d.is_authenticated());
        assert_eq!(d.name(), "df-time-service-stub");
        let t = d.fetch().expect("fetch ok");
        assert!(t.signature.is_some());
        assert_eq!(t.fetched_from, SourceId::df_time_service_stub());
    }

    #[test]
    fn t11_df_real_fetch_returns_fetch_failed_until_first_poll() {
        // Point at a deliberately-non-resolving endpoint so the poll
        // thread fails. Before any fetch has succeeded, fetch() must
        // return FetchFailed (not panic, not produce uninitialized
        // anchor). The poll thread will keep failing in the background
        // but the test exits before that matters.
        let mut cfg = DFTimeServiceClientConfig::default();
        cfg.url = "http://127.0.0.1:1/never".into();
        cfg.poll_interval_ms = 60_000;
        let d = DFTimeServiceSource::new_real(cfg);
        match d.fetch() {
            Err(ClockError::FetchFailed(_)) => (),
            other => panic!("expected FetchFailed before first successful poll: {:?}", other),
        }
        assert_eq!(d.name(), "df-time-service");
    }
}
