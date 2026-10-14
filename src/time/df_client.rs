//! Port T.1.1 — DF Time Service CLIENT.
//!
//! Implements the client side of the hub-and-spoke time-trust
//! architecture (see `agent/memory/project_time_service_architecture.md`).
//! Replaces T.1's `DFTimeServiceSource` stub with a real polling HTTP
//! client that:
//!
//! 1. Periodically `GET`s `<url>` (default `https://time.dialectforge.com/v1/now`).
//! 2. Parses the JSON response (`{ utc_ms, server_ms, key_id, signature_hex }`).
//! 3. Verifies the Ed25519 signature against a pinned DF pubkey.
//! 4. Checks drift between received `utc_ms` and the locally-projected
//!    monotonic-anchored UTC. Refuses if drift exceeds `max_drift_ms`.
//! 5. On any failure (network, parse, signature, drift) reports an
//!    anomaly via T.1's `AnomalyDetector` and falls back to monotonic
//!    projection from the most recent trusted anchor. Never panics.
//!
//! T.1.1 is the **CLIENT** half. The **server** half — the binary that
//! responds at `/v1/now` — is a separate deployment concern (DNS, TLS,
//! Cloudflare config) handled outside the Lu codebase.
//!
//! ### Wire contract
//!
//! `GET /v1/now` returns `200 OK` with body:
//! ```json
//! {
//!   "utc_ms":        1777800000000,
//!   "server_ms":     1234567890,
//!   "key_id":        "df-prod-2026",
//!   "signature_hex": "<128-char hex Ed25519 signature>"
//! }
//! ```
//!
//! The signature is over the byte sequence
//! `utc_ms.to_le_bytes() || server_ms.to_le_bytes() || key_id.as_bytes()`.
//! Pure little-endian + UTF-8 — language-portable, no canonical-JSON
//! gymnastics required.
//!
//! ### Default for canonical Lu (Hetzner production)
//!
//! `time_source: "system"`. The canonical Lu IS the time authority;
//! pointing it at `df_service` would create a circular dependency.
//! The localhost-loopback warning fires if a misconfigured df-service
//! URL points at the local machine.

use super::anomaly::{AnomalyDetector, AnomalyKind};
use super::authenticated::{AuthenticatedTime, Signature, SourceId};
use super::config::TimeConfig;
use super::monotonic::MonotonicClock;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Default DF Time Service endpoint (Cloudflare-fronted production).
/// Canonical Lu does NOT call this (it runs in `system` mode). User
/// Lus distributed via Port V default to this URL.
pub const DEFAULT_DF_TIME_URL: &str = "https://time.dialectforge.com/v1/now";

/// **Build-time placeholder pubkey.** TODO(John): regenerate when the
/// server-side time-service binary ships. For T.1.1 we ship with this
/// constant baked in so the client compiles and tests pass; production
/// canonical Lu runs `system` mode anyway, so the placeholder never
/// loads in production until the server-side ships.
///
/// The matching test private key is documented in T.1.1's the project log
/// learning entry. Real production rotation is a separate (later) port.
pub const PLACEHOLDER_DF_PUBKEY_HEX: &str =
    "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a";

pub const DEFAULT_POLL_INTERVAL_MS: u64 = 60_000;
pub const DEFAULT_MAX_DRIFT_MS: u64 = 5_000;

/// HTTP request timeout per fetch. Kept short so a slow endpoint can't
/// stall the polling thread arbitrarily long.
const HTTP_TIMEOUT_MS: u64 = 10_000;

/// Threshold for `Unreachable` trust state. After N consecutive
/// failures the anomaly is reported via the `AnomalyDetector` so it
/// surfaces in `lu_status`'s `warnings` field.
const UNREACHABLE_THRESHOLD: u32 = 3;

/// Configuration for the real DF Time Service client.
#[derive(Debug, Clone)]
pub struct DFTimeServiceClientConfig {
    pub url: String,
    pub pubkey_hex: String,
    pub poll_interval_ms: u64,
    pub max_drift_ms: u64,
    pub time_cfg: TimeConfig,
}

/// Brain-level time-source mode + DF client knobs. Lives on `Brain`,
/// hydrated from `RuntimeConfigSnapshot` on load, snapshotted back to
/// section #15 on save.
#[derive(Debug, Clone)]
pub struct TimeSourceConfig {
    /// `"system"` (canonical Lu, default) | `"df_service"` (user Lus).
    pub mode: String,
    pub url: String,
    pub pubkey_hex: String,
    pub poll_interval_ms: u64,
    pub max_drift_ms: u64,
}

impl Default for TimeSourceConfig {
    fn default() -> Self {
        TimeSourceConfig {
            mode: "system".into(),
            url: DEFAULT_DF_TIME_URL.into(),
            pubkey_hex: PLACEHOLDER_DF_PUBKEY_HEX.into(),
            poll_interval_ms: DEFAULT_POLL_INTERVAL_MS,
            max_drift_ms: DEFAULT_MAX_DRIFT_MS,
        }
    }
}

impl TimeSourceConfig {
    /// Build the lower-level `DFTimeServiceClientConfig` used by the
    /// polling thread.
    pub fn to_client_config(&self) -> DFTimeServiceClientConfig {
        DFTimeServiceClientConfig {
            url: self.url.clone(),
            pubkey_hex: self.pubkey_hex.clone(),
            poll_interval_ms: self.poll_interval_ms,
            max_drift_ms: self.max_drift_ms,
            time_cfg: TimeConfig::default(),
        }
    }

    /// True iff this brain should use the real DF Time Service client.
    pub fn is_df_service(&self) -> bool {
        self.mode == "df_service"
    }
}

impl Default for DFTimeServiceClientConfig {
    fn default() -> Self {
        DFTimeServiceClientConfig {
            url: DEFAULT_DF_TIME_URL.into(),
            pubkey_hex: PLACEHOLDER_DF_PUBKEY_HEX.into(),
            poll_interval_ms: DEFAULT_POLL_INTERVAL_MS,
            max_drift_ms: DEFAULT_MAX_DRIFT_MS,
            time_cfg: TimeConfig::default(),
        }
    }
}

/// Aggregate trust state surfaced via `lu4d_time_status`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrustState {
    /// No fetch attempted yet (or no fetch has succeeded since startup).
    Initializing,
    /// Most-recent fetch verified cleanly.
    Trusted,
    /// Fetch returned but failed verification (bad signature, drift).
    /// Lu falls back to monotonic projection from the last trusted
    /// anchor.
    Degraded,
    /// Endpoint unreachable for >= UNREACHABLE_THRESHOLD consecutive
    /// fetches.
    Unreachable,
}

/// Shared state between the polling thread and the `ClockSource::fetch`
/// caller. Held inside an `Arc<Mutex<_>>` so the poll thread can mutate
/// while the brain reads.
#[derive(Debug, Clone)]
pub struct DFRealState {
    /// Most-recent successfully-fetched-and-verified anchor. `None` until
    /// the first successful fetch.
    pub last_anchor: Option<AuthenticatedTime>,
    /// Wall-clock millis when the most recent successful fetch returned.
    pub last_fetch_ms: Option<u64>,
    /// Drift (received_utc_ms − locally_projected_ms) on most recent
    /// successful fetch. Signed: positive means the server thinks we're
    /// behind, negative means ahead.
    pub last_drift_ms: Option<i64>,
    /// Number of consecutive failed fetches since the last success.
    pub consecutive_failures: u32,
    /// Aggregate trust state (see `TrustState`).
    pub trust_state: TrustState,
    /// Last anomaly recorded by the polling thread.
    pub last_anomaly: Option<AnomalyKind>,
    /// Whether the configured URL resolves to a localhost/loopback host.
    /// Used to surface a startup warning — the canonical Lu must NEVER
    /// call itself for time (circular dependency).
    pub localhost_warning: bool,
    /// Total successful fetches since startup (cumulative).
    pub total_successful_fetches: u64,
    /// Total failed fetches since startup (cumulative).
    pub total_failed_fetches: u64,
}

impl DFRealState {
    pub fn initializing() -> Self {
        DFRealState {
            last_anchor: None,
            last_fetch_ms: None,
            last_drift_ms: None,
            consecutive_failures: 0,
            trust_state: TrustState::Initializing,
            last_anomaly: None,
            localhost_warning: false,
            total_successful_fetches: 0,
            total_failed_fetches: 0,
        }
    }
}

/// JSON shape returned by `GET /v1/now`. Matches the wire contract.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignedTimeResponse {
    pub utc_ms: u64,
    pub server_ms: u64,
    pub key_id: String,
    pub signature_hex: String,
}

/// Why a fetch failed, in enough detail to wire to `AnomalyKind`.
#[derive(Debug, Clone)]
pub enum FetchFailure {
    /// HTTP-level error (network down, timeout, non-200).
    Endpoint(String),
    /// Body parsed but JSON shape was wrong / hex-decode failed.
    Parse(String),
    /// Ed25519 verification failed.
    SignatureInvalid,
    /// Drift exceeded `max_drift_ms`.
    DriftExcess { drift_ms: u64 },
    /// Pubkey config malformed (hex decode failed at startup).
    PubkeyConfig(String),
}

impl FetchFailure {
    pub fn as_anomaly(&self, consecutive_failures: u32) -> AnomalyKind {
        match self {
            FetchFailure::Endpoint(_) | FetchFailure::Parse(_) => {
                AnomalyKind::TimeServiceUnreachable { consecutive_failures }
            }
            FetchFailure::SignatureInvalid | FetchFailure::PubkeyConfig(_) => {
                AnomalyKind::TimeServiceSignatureInvalid
            }
            FetchFailure::DriftExcess { drift_ms } => {
                AnomalyKind::TimeServiceDriftExcess {
                    drift_ms: *drift_ms,
                }
            }
        }
    }
}

/// Reconstruct the byte sequence that the server signs over.
/// Pure little-endian + UTF-8 — language-portable.
pub fn signed_message_bytes(utc_ms: u64, server_ms: u64, key_id: &str) -> Vec<u8> {
    let mut buf = Vec::with_capacity(8 + 8 + key_id.len());
    buf.extend_from_slice(&utc_ms.to_le_bytes());
    buf.extend_from_slice(&server_ms.to_le_bytes());
    buf.extend_from_slice(key_id.as_bytes());
    buf
}

/// Parse + verify a `/v1/now` JSON payload against the pinned pubkey.
/// Returns the trusted UTC ms on success, or `FetchFailure` on any
/// failure mode. Pure function — no side effects, easy to unit-test.
pub fn verify_signed_payload(
    json_bytes: &[u8],
    pubkey: &ed25519_dalek::VerifyingKey,
) -> Result<SignedTimeResponse, FetchFailure> {
    let resp: SignedTimeResponse = serde_json::from_slice(json_bytes)
        .map_err(|e| FetchFailure::Parse(format!("JSON decode: {}", e)))?;
    let sig_bytes = hex_decode(&resp.signature_hex)
        .map_err(|e| FetchFailure::Parse(format!("signature_hex decode: {}", e)))?;
    if sig_bytes.len() != 64 {
        return Err(FetchFailure::Parse(format!(
            "signature_hex must be 128 hex chars (64 bytes), got {} bytes",
            sig_bytes.len()
        )));
    }
    let sig_arr: [u8; 64] = sig_bytes
        .as_slice()
        .try_into()
        .map_err(|_| FetchFailure::Parse("signature length mismatch".into()))?;
    let msg = signed_message_bytes(resp.utc_ms, resp.server_ms, &resp.key_id);
    let sig = ed25519_dalek::Signature::from_bytes(&sig_arr);
    use ed25519_dalek::Verifier;
    pubkey
        .verify(&msg, &sig)
        .map_err(|_| FetchFailure::SignatureInvalid)?;
    Ok(resp)
}

/// Parse a hex-encoded Ed25519 pubkey string into a `VerifyingKey`.
pub fn parse_pubkey_hex(hex_str: &str) -> Result<ed25519_dalek::VerifyingKey, String> {
    let bytes = hex_decode(hex_str).map_err(|e| format!("pubkey hex decode: {}", e))?;
    if bytes.len() != 32 {
        return Err(format!(
            "DF pubkey must be 32 bytes (64 hex chars), got {}",
            bytes.len()
        ));
    }
    let arr: [u8; 32] = bytes
        .as_slice()
        .try_into()
        .map_err(|_| "pubkey length mismatch".to_string())?;
    ed25519_dalek::VerifyingKey::from_bytes(&arr)
        .map_err(|e| format!("pubkey from_bytes: {}", e))
}

/// Minimal hex decoder. We don't pull in the `hex` crate to avoid a new
/// dependency for ~20 lines of code.
fn hex_decode(s: &str) -> Result<Vec<u8>, String> {
    if s.len() % 2 != 0 {
        return Err(format!("hex string odd length: {}", s.len()));
    }
    let mut out = Vec::with_capacity(s.len() / 2);
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let hi = hex_nibble(bytes[i])?;
        let lo = hex_nibble(bytes[i + 1])?;
        out.push((hi << 4) | lo);
        i += 2;
    }
    Ok(out)
}

fn hex_nibble(b: u8) -> Result<u8, String> {
    match b {
        b'0'..=b'9' => Ok(b - b'0'),
        b'a'..=b'f' => Ok(b - b'a' + 10),
        b'A'..=b'F' => Ok(b - b'A' + 10),
        other => Err(format!("non-hex byte: 0x{:02x}", other)),
    }
}

/// Detect localhost / loopback in the configured URL. Used for the
/// startup warning — a canonical Lu pointing `df_service` mode at its
/// own host is the circular-dependency footgun called out in
/// `project_time_service_architecture.md`.
pub fn url_is_localhost(url: &str) -> bool {
    let lower = url.to_ascii_lowercase();
    // Strip scheme.
    let after_scheme = lower.split("://").nth(1).unwrap_or(&lower);
    // Strip path.
    let host_port = after_scheme.split('/').next().unwrap_or("");
    // Handle bracketed IPv6 literals: `[::1]:8080` → host = "::1".
    let host = if let Some(rest) = host_port.strip_prefix('[') {
        rest.split(']').next().unwrap_or("")
    } else {
        // IPv4 / hostname: split on the LAST colon to avoid eating an
        // unbracketed IPv6 (which is an invalid URL anyway, but keep
        // the behaviour permissive).
        host_port.rsplit_once(':').map(|(h, _)| h).unwrap_or(host_port)
    };
    matches!(host, "localhost" | "127.0.0.1" | "::1" | "0.0.0.0")
}

// ---------------------------------------------------------------------
// Polling thread + Source impl
// ---------------------------------------------------------------------

/// The real DF Time Service client. Spawns a background thread that
/// polls the configured endpoint every `poll_interval_ms`. The
/// `fetch()` method (see `source.rs`) returns a monotonic-projected
/// anchor from the most recent successful fetch.
pub struct DFRealClient {
    pub config: DFTimeServiceClientConfig,
    pub state: Arc<Mutex<DFRealState>>,
    pub anomaly_sink: Option<Arc<Mutex<AnomalyDetector>>>,
    stop_signal: Arc<AtomicBool>,
    poll_handle: Mutex<Option<std::thread::JoinHandle<()>>>,
}

impl DFRealClient {
    /// Spawn the polling thread and return the client. The thread
    /// detaches on `Drop`.
    pub fn spawn(
        config: DFTimeServiceClientConfig,
        anomaly_sink: Option<Arc<Mutex<AnomalyDetector>>>,
    ) -> Self {
        let state = Arc::new(Mutex::new(DFRealState::initializing()));
        if url_is_localhost(&config.url) {
            if let Ok(mut s) = state.lock() {
                s.localhost_warning = true;
            }
            eprintln!(
                "[lu4d] T.1.1: WARNING — df_service URL {} is localhost/loopback. \
                 If this is the canonical Lu, you have a circular dependency.",
                config.url
            );
        }
        let stop_signal = Arc::new(AtomicBool::new(false));
        let pubkey = match parse_pubkey_hex(&config.pubkey_hex) {
            Ok(k) => k,
            Err(e) => {
                if let Ok(mut s) = state.lock() {
                    s.trust_state = TrustState::Degraded;
                    s.last_anomaly = Some(AnomalyKind::TimeServiceSignatureInvalid);
                }
                eprintln!(
                    "[lu4d] T.1.1: pubkey config invalid: {}; df_service mode will fail",
                    e
                );
                // Still start the client; the poll thread will keep
                // reporting Unreachable / SignatureInvalid until the
                // user fixes config via lu4d_configure.
                return DFRealClient {
                    config,
                    state,
                    anomaly_sink,
                    stop_signal,
                    poll_handle: Mutex::new(None),
                };
            }
        };
        let poll_state = state.clone();
        let poll_stop = stop_signal.clone();
        let poll_cfg = config.clone();
        let poll_sink = anomaly_sink.clone();
        let handle = std::thread::Builder::new()
            .name("df-time-poll".into())
            .spawn(move || {
                poll_loop(poll_cfg, pubkey, poll_state, poll_stop, poll_sink);
            })
            .ok();
        DFRealClient {
            config,
            state,
            anomaly_sink,
            stop_signal,
            poll_handle: Mutex::new(handle),
        }
    }

    /// Test-only constructor — does NOT spawn the polling thread.
    /// Tests drive the state via `process_signed_payload_for_test`.
    pub fn for_test(config: DFTimeServiceClientConfig) -> Self {
        let state = Arc::new(Mutex::new(DFRealState::initializing()));
        if url_is_localhost(&config.url) {
            if let Ok(mut s) = state.lock() {
                s.localhost_warning = true;
            }
        }
        DFRealClient {
            config,
            state,
            anomaly_sink: None,
            stop_signal: Arc::new(AtomicBool::new(false)),
            poll_handle: Mutex::new(None),
        }
    }

    /// Test helper: drive the verification path with a JSON payload
    /// without spinning up an HTTP server. The pubkey from
    /// `self.config.pubkey_hex` is used.
    pub fn process_signed_payload_for_test(
        &self,
        json_bytes: &[u8],
        now_ms: u64,
    ) -> Result<(), AnomalyKind> {
        let pubkey = parse_pubkey_hex(&self.config.pubkey_hex)
            .map_err(|_| AnomalyKind::TimeServiceSignatureInvalid)?;
        process_one_payload(
            &self.config,
            &pubkey,
            json_bytes,
            now_ms,
            &self.state,
            self.anomaly_sink.as_ref(),
        )
    }

    /// Snapshot of the current state. Cheap; clones the struct.
    pub fn state_snapshot(&self) -> DFRealState {
        self.state
            .lock()
            .map(|s| s.clone())
            .unwrap_or_else(|_| DFRealState::initializing())
    }

    /// Build a `ClockSource::fetch` result from the current state.
    /// Used by `DFTimeServiceSource::fetch` in `source.rs`.
    pub fn project_now(&self) -> Result<AuthenticatedTime, super::source::ClockError> {
        let state = self
            .state
            .lock()
            .map_err(|e| super::source::ClockError::FetchFailed(format!("state lock: {}", e)))?;
        let anchor = match &state.last_anchor {
            Some(a) => a.clone(),
            None => {
                return Err(super::source::ClockError::FetchFailed(
                    "DF Time Service: no successful fetch yet".into(),
                ));
            }
        };
        let now_mono = MonotonicClock::now();
        let projected_utc = anchor.project_now_ms(now_mono);
        Ok(AuthenticatedTime::signed(
            projected_utc,
            now_mono,
            anchor.fetched_from.clone(),
            anchor
                .signature
                .clone()
                .unwrap_or_else(|| Signature::stub_for(projected_utc)),
            projected_utc.saturating_add(self.config.time_cfg.refresh_interval_ms),
        ))
    }
}

impl Drop for DFRealClient {
    fn drop(&mut self) {
        self.stop_signal.store(true, Ordering::SeqCst);
        // Don't block: the poll thread sleeps in chunks of 100ms so it
        // exits within that window. We `take()` the handle but DON'T
        // join — Lu's MCP shutdown path is bounded and we don't want to
        // wait on a possibly-blocked HTTP call.
        if let Ok(mut guard) = self.poll_handle.lock() {
            let _ = guard.take();
        }
    }
}

fn poll_loop(
    config: DFTimeServiceClientConfig,
    pubkey: ed25519_dalek::VerifyingKey,
    state: Arc<Mutex<DFRealState>>,
    stop: Arc<AtomicBool>,
    anomaly_sink: Option<Arc<Mutex<AnomalyDetector>>>,
) {
    // First fetch happens immediately, then every poll_interval_ms.
    loop {
        if stop.load(Ordering::SeqCst) {
            return;
        }
        do_one_poll(&config, &pubkey, &state, anomaly_sink.as_ref());
        // Sleep in 100ms chunks so we react to stop quickly.
        let mut remaining = config.poll_interval_ms;
        while remaining > 0 {
            if stop.load(Ordering::SeqCst) {
                return;
            }
            let chunk = remaining.min(100);
            std::thread::sleep(Duration::from_millis(chunk));
            remaining = remaining.saturating_sub(chunk);
        }
    }
}

fn do_one_poll(
    config: &DFTimeServiceClientConfig,
    pubkey: &ed25519_dalek::VerifyingKey,
    state: &Arc<Mutex<DFRealState>>,
    anomaly_sink: Option<&Arc<Mutex<AnomalyDetector>>>,
) {
    let now_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);
    let agent = ureq::AgentBuilder::new()
        .timeout(Duration::from_millis(HTTP_TIMEOUT_MS))
        .build();
    let resp = match agent.get(&config.url).call() {
        Ok(r) => r,
        Err(e) => {
            record_failure(
                state,
                anomaly_sink,
                FetchFailure::Endpoint(format!("{}", e)),
            );
            return;
        }
    };
    let status = resp.status();
    if status != 200 {
        record_failure(
            state,
            anomaly_sink,
            FetchFailure::Endpoint(format!("non-200: {}", status)),
        );
        return;
    }
    let body = match resp.into_string() {
        Ok(b) => b,
        Err(e) => {
            record_failure(
                state,
                anomaly_sink,
                FetchFailure::Endpoint(format!("body read: {}", e)),
            );
            return;
        }
    };
    let _ = process_one_payload(config, pubkey, body.as_bytes(), now_ms, state, anomaly_sink);
}

/// Common path used by both the poll thread and tests. Verifies the
/// payload, applies the drift check, updates state, and reports
/// anomalies. Returns `Ok(())` on success, or the recorded
/// `AnomalyKind` on any failure.
pub fn process_one_payload(
    config: &DFTimeServiceClientConfig,
    pubkey: &ed25519_dalek::VerifyingKey,
    json_bytes: &[u8],
    now_ms: u64,
    state: &Arc<Mutex<DFRealState>>,
    anomaly_sink: Option<&Arc<Mutex<AnomalyDetector>>>,
) -> Result<(), AnomalyKind> {
    match verify_signed_payload(json_bytes, pubkey) {
        Ok(resp) => {
            // Drift check: project the most recent anchor (if any) to now
            // using monotonic, compare to received utc_ms. For the very
            // first fetch (no prior anchor) we accept whatever the server
            // says.
            let drift_ms_signed: i64 = {
                let projected = if let Ok(s) = state.lock() {
                    s.last_anchor.as_ref().map(|a| {
                        a.project_now_ms(MonotonicClock::now())
                    })
                } else {
                    None
                };
                match projected {
                    Some(p) => resp.utc_ms as i64 - p as i64,
                    None => 0,
                }
            };
            let drift_abs = drift_ms_signed.unsigned_abs();
            if drift_abs > config.max_drift_ms {
                let kind = AnomalyKind::TimeServiceDriftExcess { drift_ms: drift_abs };
                if let Ok(mut s) = state.lock() {
                    s.consecutive_failures = s.consecutive_failures.saturating_add(1);
                    s.total_failed_fetches = s.total_failed_fetches.saturating_add(1);
                    s.last_anomaly = Some(kind.clone());
                    s.trust_state = TrustState::Degraded;
                }
                if let Some(sink) = anomaly_sink {
                    if let Ok(mut det) = sink.lock() {
                        det.record_anomaly(kind.clone());
                    }
                }
                return Err(kind);
            }
            // Verified + drift-OK: update the trusted anchor.
            let mono = MonotonicClock::now();
            let sig_bytes = hex_decode(&resp.signature_hex).unwrap_or_default();
            let new_anchor = AuthenticatedTime::signed(
                resp.utc_ms,
                mono,
                SourceId::new(format!("df-time-service:{}", resp.key_id)),
                Signature(sig_bytes),
                resp.utc_ms.saturating_add(config.time_cfg.refresh_interval_ms),
            );
            if let Ok(mut s) = state.lock() {
                s.last_anchor = Some(new_anchor);
                s.last_fetch_ms = Some(now_ms);
                s.last_drift_ms = Some(drift_ms_signed);
                s.consecutive_failures = 0;
                s.total_successful_fetches = s.total_successful_fetches.saturating_add(1);
                s.trust_state = TrustState::Trusted;
                s.last_anomaly = None;
            }
            Ok(())
        }
        Err(failure) => {
            let consecutive = state
                .lock()
                .ok()
                .map(|s| s.consecutive_failures.saturating_add(1))
                .unwrap_or(1);
            record_failure(state, anomaly_sink, failure.clone());
            Err(failure.as_anomaly(consecutive))
        }
    }
}

fn record_failure(
    state: &Arc<Mutex<DFRealState>>,
    anomaly_sink: Option<&Arc<Mutex<AnomalyDetector>>>,
    failure: FetchFailure,
) {
    let consecutive = if let Ok(mut s) = state.lock() {
        s.consecutive_failures = s.consecutive_failures.saturating_add(1);
        s.total_failed_fetches = s.total_failed_fetches.saturating_add(1);
        let kind = failure.as_anomaly(s.consecutive_failures);
        s.last_anomaly = Some(kind.clone());
        // Trust state transitions:
        //   - Endpoint/Parse: bump to Unreachable once threshold is crossed.
        //   - SignatureInvalid / DriftExcess: Degraded (we got a response
        //     but it's untrusted).
        s.trust_state = match &failure {
            FetchFailure::Endpoint(_) | FetchFailure::Parse(_) => {
                if s.consecutive_failures >= UNREACHABLE_THRESHOLD {
                    TrustState::Unreachable
                } else {
                    // Don't downgrade Trusted on a single transient error;
                    // the previous trust state is preserved until the
                    // threshold trips.
                    if matches!(s.trust_state, TrustState::Trusted) {
                        TrustState::Trusted
                    } else {
                        TrustState::Degraded
                    }
                }
            }
            FetchFailure::SignatureInvalid
            | FetchFailure::DriftExcess { .. }
            | FetchFailure::PubkeyConfig(_) => TrustState::Degraded,
        };
        s.consecutive_failures
    } else {
        1
    };
    if let Some(sink) = anomaly_sink {
        if let Ok(mut det) = sink.lock() {
            det.record_anomaly(failure.as_anomaly(consecutive));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::{Signer, SigningKey};

    fn test_keypair() -> (SigningKey, ed25519_dalek::VerifyingKey) {
        // Deterministic 32-byte seed → reproducible test key.
        let seed: [u8; 32] = [
            0xab, 0x12, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee,
            0xff, 0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc,
            0xdd, 0xee, 0xff, 0x00,
        ];
        let sk = SigningKey::from_bytes(&seed);
        let vk = sk.verifying_key();
        (sk, vk)
    }

    fn build_signed_payload(
        sk: &SigningKey,
        utc_ms: u64,
        server_ms: u64,
        key_id: &str,
    ) -> Vec<u8> {
        let msg = signed_message_bytes(utc_ms, server_ms, key_id);
        let sig = sk.sign(&msg);
        let sig_hex: String = sig.to_bytes().iter().map(|b| format!("{:02x}", b)).collect();
        let resp = SignedTimeResponse {
            utc_ms,
            server_ms,
            key_id: key_id.into(),
            signature_hex: sig_hex,
        };
        serde_json::to_vec(&resp).unwrap()
    }

    fn vk_to_hex(vk: &ed25519_dalek::VerifyingKey) -> String {
        vk.to_bytes().iter().map(|b| format!("{:02x}", b)).collect()
    }

    #[test]
    fn verify_signed_payload_round_trips_a_valid_signature() {
        let (sk, vk) = test_keypair();
        let payload = build_signed_payload(&sk, 1_777_800_000_000, 12345, "df-test-2026");
        let parsed = verify_signed_payload(&payload, &vk).expect("valid sig must parse");
        assert_eq!(parsed.utc_ms, 1_777_800_000_000);
        assert_eq!(parsed.key_id, "df-test-2026");
    }

    #[test]
    fn verify_signed_payload_rejects_a_tampered_utc() {
        let (sk, vk) = test_keypair();
        let mut payload = build_signed_payload(&sk, 1_000, 1, "df-test");
        // Mutate utc_ms post-sign.
        let mut v: serde_json::Value = serde_json::from_slice(&payload).unwrap();
        v["utc_ms"] = serde_json::json!(2_000);
        payload = serde_json::to_vec(&v).unwrap();
        match verify_signed_payload(&payload, &vk) {
            Err(FetchFailure::SignatureInvalid) => (),
            other => panic!("expected SignatureInvalid, got {:?}", other),
        }
    }

    #[test]
    fn verify_signed_payload_rejects_short_signature() {
        let (_sk, vk) = test_keypair();
        let resp = SignedTimeResponse {
            utc_ms: 1,
            server_ms: 1,
            key_id: "x".into(),
            signature_hex: "deadbeef".into(),
        };
        let payload = serde_json::to_vec(&resp).unwrap();
        match verify_signed_payload(&payload, &vk) {
            Err(FetchFailure::Parse(_)) => (),
            other => panic!("expected Parse, got {:?}", other),
        }
    }

    #[test]
    fn parse_pubkey_hex_round_trips() {
        let (_, vk) = test_keypair();
        let hex = vk_to_hex(&vk);
        let back = parse_pubkey_hex(&hex).unwrap();
        assert_eq!(back.to_bytes(), vk.to_bytes());
    }

    #[test]
    fn parse_pubkey_hex_rejects_wrong_length() {
        assert!(parse_pubkey_hex("ab").is_err());
        assert!(parse_pubkey_hex(&"a".repeat(63)).is_err());
    }

    #[test]
    fn url_is_localhost_detects_loopback_variants() {
        assert!(url_is_localhost("http://localhost:9999/now"));
        assert!(url_is_localhost("http://127.0.0.1/v1/now"));
        assert!(url_is_localhost("https://localhost/"));
        assert!(url_is_localhost("http://[::1]:8080/x"));
        assert!(!url_is_localhost("https://time.dialectforge.com/v1/now"));
        assert!(!url_is_localhost("https://example.com"));
    }

    #[test]
    fn signed_message_bytes_is_le_concat() {
        let bytes = signed_message_bytes(0x0102030405060708, 0x0807060504030201, "k");
        assert_eq!(
            bytes,
            vec![
                0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01, // utc_ms LE
                0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, // server_ms LE
                b'k',
            ]
        );
    }
}
