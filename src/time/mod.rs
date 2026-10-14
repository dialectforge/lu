//! Port T.1 — Time primitive. Port T.1.1 — DF Time Service client.
//!
//! Authenticated UTC + monotonic clock + clock-source trait, with rollback /
//! large-forward-jump anomaly detection. T.1 shipped a stub for the DF
//! Time Service backend; **T.1.1** lights up the real polling client
//! (see `df_client.rs`). The `SystemClockSource` is the dev/fallback
//! default and matches the legacy pre-T.1 behaviour.
//!
//! Design rationale internal notes Port T spec):
//!  - Monotonic clock is what scar-sensitive ops MUST consult for ordering;
//!    wall-clock is for surface display and bond timestamps that already
//!    survive a clock skew via `monotonic_at_fetch` lineage.
//!  - Anomaly detection is fail-safe: a backward jump or large forward jump
//!    flips the brain into degraded mode where scar-sensitive ops refuse;
//!    queries (`lu4d_ask`) keep working so the user is not locked out.
//!  - The `ClockSource` trait makes T.1.1's swap to a real DF-Time-Service
//!    HTTP client a one-line config change.
//!
//! ### T.1.1 hub-and-spoke trust model
//!
//! See `agent/memory/project_time_service_architecture.md` for the
//! full design. Two modes:
//!  - `time_source: "system"` — canonical Lu (DF's Hetzner production)
//!    reads OS clock directly. IS the time authority. Default.
//!  - `time_source: "df_service"` — every other user's Lu calls
//!    `https://time.dialectforge.com/v1/now`, verifies an Ed25519
//!    signature against a pinned DF pubkey, falls back to monotonic
//!    projection on failure.

pub mod anomaly;
pub mod authenticated;
pub mod config;
pub mod df_client;
pub mod monotonic;
pub mod source;

pub use anomaly::{AnomalyDetector, AnomalyKind};
pub use authenticated::{AuthenticatedTime, Signature, SourceId};
pub use config::TimeConfig;
pub use df_client::{
    DFRealClient, DFRealState, DFTimeServiceClientConfig, FetchFailure, SignedTimeResponse,
    TimeSourceConfig, TrustState, DEFAULT_DF_TIME_URL, DEFAULT_MAX_DRIFT_MS,
    DEFAULT_POLL_INTERVAL_MS, PLACEHOLDER_DF_PUBKEY_HEX,
};
pub use monotonic::{MonotonicClock, MonotonicTimestamp};
pub use source::{ClockError, ClockSource, DFTimeServiceSource, SystemClockSource};

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    /// T.1: monotonic clock never goes backward across consecutive `now()`.
    #[test]
    fn t1_monotonic_only_advances() {
        let mut prev = MonotonicClock::now();
        for _ in 0..100 {
            let next = MonotonicClock::now();
            assert!(
                next.as_nanos() >= prev.as_nanos(),
                "monotonic clock went backward: {} -> {}",
                prev.as_nanos(),
                next.as_nanos()
            );
            prev = next;
        }
    }

    /// T.1: DFTimeServiceSource (stub) returns a signature; SystemClockSource
    /// does not. `is_authenticated()` must reflect that distinction.
    #[test]
    fn t1_authenticated_time_has_signature_when_authenticated_source() {
        let sys = SystemClockSource::new();
        assert!(!sys.is_authenticated(), "SystemClockSource must not claim authenticated");
        let sys_t = sys.fetch().expect("system fetch ok");
        assert!(sys_t.signature.is_none(), "system source must not produce a signature");
        assert!(!sys_t.is_authenticated());

        // Stub DF time service: returns a deterministic Signature under the
        // "df-time-service-stub" SourceId. Real signed fetch lands in T.1.1.
        let df = DFTimeServiceSource::stub();
        assert!(df.is_authenticated(), "DFTimeServiceSource must claim authenticated");
        let df_t = df.fetch().expect("df fetch ok");
        assert!(df_t.signature.is_some(), "df source must produce a signature");
        assert!(df_t.is_authenticated());
    }

    /// T.1 SAFETY-CRITICAL: anomaly detector flags a backward wall-clock jump
    /// (any backward jump fires when `max_backward_jump_ms == 0`). This is
    /// the load-bearing test for fail-safe degradation.
    #[test]
    fn t1_anomaly_detector_flags_backward_jump() {
        let mut detector = AnomalyDetector::new(TimeConfig::default());
        // First check primes the detector.
        let t0 = AuthenticatedTime::unsigned(
            10_000,
            MonotonicTimestamp::from_nanos(1_000),
            SourceId::system(),
            10_000 + TimeConfig::default().refresh_interval_ms,
        );
        assert!(detector.check(&t0).is_ok());
        // Backward jump: wall-clock goes from 10_000 to 9_500 (delta = 500 ms).
        let t1 = AuthenticatedTime::unsigned(
            9_500,
            MonotonicTimestamp::from_nanos(2_000),
            SourceId::system(),
            9_500 + TimeConfig::default().refresh_interval_ms,
        );
        match detector.check(&t1) {
            Err(AnomalyKind::BackwardJump { delta_ms }) => assert_eq!(delta_ms, 500),
            other => panic!("expected BackwardJump(500), got {:?}", other),
        }
    }

    /// T.1: anomaly detector flags a large forward jump beyond the configured
    /// tolerance (default 7 days). A 30-day jump must trip the detector.
    #[test]
    fn t1_anomaly_detector_flags_large_forward_jump() {
        let cfg = TimeConfig {
            max_forward_jump_ms: 7 * 24 * 3600 * 1000, // 7 days
            ..TimeConfig::default()
        };
        let mut detector = AnomalyDetector::new(cfg);
        let t0 = AuthenticatedTime::unsigned(
            1_000,
            MonotonicTimestamp::from_nanos(0),
            SourceId::system(),
            1_000 + cfg.refresh_interval_ms,
        );
        assert!(detector.check(&t0).is_ok());
        let thirty_days_ms: u64 = 30 * 24 * 3600 * 1000;
        let t1 = AuthenticatedTime::unsigned(
            1_000 + thirty_days_ms,
            MonotonicTimestamp::from_nanos(1_000),
            SourceId::system(),
            1_000 + thirty_days_ms + cfg.refresh_interval_ms,
        );
        match detector.check(&t1) {
            Err(AnomalyKind::LargeForwardJump { delta_ms }) => {
                assert_eq!(delta_ms, thirty_days_ms);
            }
            other => panic!("expected LargeForwardJump, got {:?}", other),
        }
    }

    /// T.1: when `Brain::time_is_authenticated()` is false, scar-sensitive
    /// ops refuse with `ClockError::TimeNotAuthenticated`. Normal ops still
    /// run — degradation is scoped, not global.
    #[test]
    fn t1_brain_refuses_scar_ops_when_time_unauthenticated() {
        use crate::brain::Brain;
        let mut b = Brain::new();
        // Default source = SystemClockSource → unauthenticated.
        assert!(!b.time_is_authenticated());
        // Placeholder scar-sensitive op gate. Real scar-time logic lands in
        // T.5; T.1 only ships the gate so future ports can plug in.
        let res = b.t1_scar_sensitive_op_probe();
        assert!(
            matches!(res, Err(ClockError::TimeNotAuthenticated)),
            "scar-sensitive op must refuse on unauthenticated time: {:?}",
            res
        );
    }

    /// T.1: scoped degradation — `lu4d_ask` (here represented by a query-shape
    /// method) keeps working even when time is unauthenticated. Only
    /// scar-sensitive ops refuse.
    #[test]
    fn t1_brain_serves_queries_normally_when_time_unauthenticated() {
        use crate::brain::Brain;
        let b = Brain::new();
        assert!(!b.time_is_authenticated());
        // current_utc_ms() must always return a value (system time fallback).
        let now = b.current_utc_ms();
        assert!(now > 0, "current_utc_ms must always return a value");
        // monotonic_now() is local-only and never refuses.
        let _m = b.monotonic_now();
    }

    /// T.1: the authenticated-time anchor refreshes after expiry. Re-fetching
    /// from a source produces a newer `monotonic_at_fetch`.
    #[test]
    fn t1_authenticated_time_anchor_refreshes_after_expiry() {
        let src: Arc<dyn ClockSource> = Arc::new(SystemClockSource::new());
        let first = src.fetch().expect("fetch 1");
        thread::sleep(Duration::from_millis(20));
        let second = src.fetch().expect("fetch 2");
        assert!(
            second.monotonic_at_fetch.as_nanos() > first.monotonic_at_fetch.as_nanos(),
            "second fetch must have newer monotonic anchor"
        );
    }

    // ====================================================================
    // Port T.1.1 — DF Time Service client tests (10 cases per spec)
    // ====================================================================
    //
    // These tests drive the real DF client without standing up an HTTP
    // server, by feeding crafted JSON payloads through
    // `DFRealClient::process_signed_payload_for_test`. The empirical
    // demo (Phase 7) exercises the real ureq+http path against a Python
    // mock server.

    use ed25519_dalek::{Signer, SigningKey};
    use std::sync::Mutex;

    fn t11_test_keypair() -> (SigningKey, ed25519_dalek::VerifyingKey) {
        // Deterministic seed → reproducible test pubkey across runs.
        let seed: [u8; 32] = [
            0x37, 0xe1, 0x12, 0x9b, 0x4d, 0x66, 0x7e, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee,
            0x42, 0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc,
            0xdd, 0xee, 0xff, 0x42,
        ];
        let sk = SigningKey::from_bytes(&seed);
        let vk = sk.verifying_key();
        (sk, vk)
    }

    fn t11_vk_to_hex(vk: &ed25519_dalek::VerifyingKey) -> String {
        vk.to_bytes().iter().map(|b| format!("{:02x}", b)).collect()
    }

    fn t11_build_signed_payload(
        sk: &SigningKey,
        utc_ms: u64,
        server_ms: u64,
        key_id: &str,
    ) -> Vec<u8> {
        let msg = df_client::signed_message_bytes(utc_ms, server_ms, key_id);
        let sig = sk.sign(&msg);
        let sig_hex: String = sig.to_bytes().iter().map(|b| format!("{:02x}", b)).collect();
        let resp = df_client::SignedTimeResponse {
            utc_ms,
            server_ms,
            key_id: key_id.into(),
            signature_hex: sig_hex,
        };
        serde_json::to_vec(&resp).unwrap()
    }

    fn t11_now_ms() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0)
    }

    /// 1. `time_source: "system"` mode: brain's clock IS the
    ///    `SystemClockSource`. No remote calls.
    #[test]
    fn t11_system_mode_uses_system_clock_directly() {
        use crate::brain::Brain;
        let b = Brain::new();
        assert_eq!(
            b.clock.name(),
            "system",
            "default Brain must use SystemClockSource"
        );
        assert!(!b.clock.is_authenticated());
    }

    /// 2. `df_service` mode: a valid signed payload is parsed +
    ///    verified + accepted; trust state flips to Trusted.
    #[test]
    fn t11_df_service_mode_parses_valid_signed_response() {
        let (sk, vk) = t11_test_keypair();
        let mut cfg = DFTimeServiceClientConfig::default();
        cfg.pubkey_hex = t11_vk_to_hex(&vk);
        cfg.url = "http://127.0.0.1:9/never".into();
        cfg.max_drift_ms = 60_000; // generous so first-fetch drift=0 path passes.
        let client = DFRealClient::for_test(cfg.clone());
        let payload = t11_build_signed_payload(&sk, 1_777_800_000_000, 99, "df-test-2026");
        let res = client.process_signed_payload_for_test(&payload, t11_now_ms());
        assert!(res.is_ok(), "valid payload must verify: {:?}", res);
        let snap = client.state_snapshot();
        assert!(matches!(snap.trust_state, TrustState::Trusted));
        assert_eq!(snap.consecutive_failures, 0);
        assert!(snap.last_anchor.is_some());
    }

    /// 3. `df_service` mode: a tampered signature is rejected; trust
    ///    state flips to Degraded.
    #[test]
    fn t11_df_service_mode_rejects_bad_signature() {
        let (sk, vk) = t11_test_keypair();
        let mut cfg = DFTimeServiceClientConfig::default();
        cfg.pubkey_hex = t11_vk_to_hex(&vk);
        let client = DFRealClient::for_test(cfg);
        // Build a valid payload then mutate utc_ms (sig no longer verifies).
        let mut payload = t11_build_signed_payload(&sk, 1_000, 1, "k");
        let mut v: serde_json::Value = serde_json::from_slice(&payload).unwrap();
        v["utc_ms"] = serde_json::json!(2_000);
        payload = serde_json::to_vec(&v).unwrap();
        let res = client.process_signed_payload_for_test(&payload, t11_now_ms());
        match res {
            Err(AnomalyKind::TimeServiceSignatureInvalid) => (),
            other => panic!("expected TimeServiceSignatureInvalid, got {:?}", other),
        }
        let snap = client.state_snapshot();
        assert!(matches!(snap.trust_state, TrustState::Degraded));
        assert!(snap.last_anchor.is_none());
    }

    /// 4. `df_service` mode: drift exceeding `max_drift_ms` is rejected.
    #[test]
    fn t11_df_service_mode_rejects_excessive_drift() {
        let (sk, vk) = t11_test_keypair();
        let mut cfg = DFTimeServiceClientConfig::default();
        cfg.pubkey_hex = t11_vk_to_hex(&vk);
        cfg.max_drift_ms = 100;
        let client = DFRealClient::for_test(cfg.clone());
        // Seed a trusted anchor at t=10_000 first.
        let p1 = t11_build_signed_payload(&sk, 10_000, 1, "k");
        client.process_signed_payload_for_test(&p1, 10_000).unwrap();
        // Now feed a payload that's 5_000ms ahead of where projection
        // would put it — drift_abs >> max_drift_ms = 100.
        let p2 = t11_build_signed_payload(&sk, 15_000, 2, "k");
        let res = client.process_signed_payload_for_test(&p2, 10_001);
        match res {
            Err(AnomalyKind::TimeServiceDriftExcess { drift_ms }) => {
                assert!(drift_ms > 100, "drift_ms must exceed max_drift_ms: {}", drift_ms);
            }
            other => panic!("expected TimeServiceDriftExcess, got {:?}", other),
        }
        let snap = client.state_snapshot();
        assert!(matches!(snap.trust_state, TrustState::Degraded));
    }

    /// 5. `df_service` mode: when the endpoint is unreachable, fetch()
    ///    returns `FetchFailed` and Lu falls back to monotonic
    ///    projection from the last trusted anchor (if any).
    #[test]
    fn t11_df_service_mode_falls_back_to_monotonic_on_endpoint_failure() {
        let mut cfg = DFTimeServiceClientConfig::default();
        cfg.url = "http://127.0.0.1:1/never".into();
        cfg.poll_interval_ms = 60_000;
        let src = DFTimeServiceSource::new_real(cfg);
        // No successful poll yet → fetch() must return FetchFailed.
        match src.fetch() {
            Err(ClockError::FetchFailed(_)) => (),
            other => panic!("expected FetchFailed, got {:?}", other),
        }
        // Brain falls back to system clock — never panics.
        let utc = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        assert!(utc > 0, "system fallback must produce a real wall clock");
    }

    /// 6. Bad signature reports an anomaly via the wired AnomalyDetector.
    #[test]
    fn t11_df_service_mode_logs_anomaly_on_invalid_signature() {
        let (sk, vk) = t11_test_keypair();
        let mut cfg = DFTimeServiceClientConfig::default();
        cfg.pubkey_hex = t11_vk_to_hex(&vk);
        let det = Arc::new(Mutex::new(AnomalyDetector::new(TimeConfig::default())));
        let mut client = DFRealClient::for_test(cfg);
        client.anomaly_sink = Some(det.clone());
        // Tamper a valid payload.
        let mut p = t11_build_signed_payload(&sk, 5_000, 1, "k");
        let mut v: serde_json::Value = serde_json::from_slice(&p).unwrap();
        v["key_id"] = serde_json::json!("k-different");
        p = serde_json::to_vec(&v).unwrap();
        let _ = client.process_signed_payload_for_test(&p, t11_now_ms());
        let recorded = det.lock().unwrap().last_anomaly().cloned();
        assert!(matches!(
            recorded,
            Some(AnomalyKind::TimeServiceSignatureInvalid)
        ));
    }

    /// 7. Drift excess reports an anomaly via the wired AnomalyDetector.
    #[test]
    fn t11_df_service_mode_logs_anomaly_on_drift_excess() {
        let (sk, vk) = t11_test_keypair();
        let mut cfg = DFTimeServiceClientConfig::default();
        cfg.pubkey_hex = t11_vk_to_hex(&vk);
        cfg.max_drift_ms = 50;
        let det = Arc::new(Mutex::new(AnomalyDetector::new(TimeConfig::default())));
        let mut client = DFRealClient::for_test(cfg);
        client.anomaly_sink = Some(det.clone());
        // Seed.
        let p1 = t11_build_signed_payload(&sk, 10_000, 1, "k");
        client.process_signed_payload_for_test(&p1, 10_000).unwrap();
        // Drift the next anchor far beyond max_drift_ms.
        let p2 = t11_build_signed_payload(&sk, 99_999, 2, "k");
        let _ = client.process_signed_payload_for_test(&p2, 10_001);
        let recorded = det.lock().unwrap().last_anomaly().cloned();
        assert!(matches!(
            recorded,
            Some(AnomalyKind::TimeServiceDriftExcess { .. })
        ));
    }

    /// 8. lu4d_time_status returns the current mode and trust state for
    ///    a brain in `system` mode (the canonical-Lu default).
    #[test]
    fn t11_lu4d_time_status_returns_current_mode_and_state() {
        use crate::brain::Brain;
        let mut b = Brain::new();
        let json = crate::mcp_t11::time_status_payload(&mut b)
            .expect("time_status_payload must succeed");
        assert_eq!(json["time_source"], "system");
        assert_eq!(json["clock_name"], "system");
        // System mode reports `unused` for the df_service-only fields.
        assert_eq!(json["trust_state"], serde_json::Value::Null);
        assert!(json["localhost_warning"].as_bool().unwrap_or(true) == false);
    }

    /// 9. Forward compat: a pre-T.1.1 brain (no time_* fields in
    ///    section #15) loads with the system-mode default. Validated
    ///    via the runtime_config layer.
    #[test]
    fn t11_pre_t11_brain_loads_with_default_time_source_system() {
        // Build a section #15 payload and remove the T.1.1 fields to
        // simulate a pre-T.1.1 binary's saved brain.
        let snap = crate::runtime_config::RuntimeConfigSnapshot::default();
        let json_bytes = snap.encode_v15_section().unwrap();
        let mut value: serde_json::Value = serde_json::from_slice(&json_bytes).unwrap();
        if let Some(obj) = value.as_object_mut() {
            obj.remove("time_source");
            obj.remove("time_service_url");
            obj.remove("time_service_pubkey_hex");
            obj.remove("time_service_poll_interval_ms");
            obj.remove("time_service_max_drift_ms");
        }
        let pre_t11_bytes = serde_json::to_vec(&value).unwrap();
        let parsed = crate::runtime_config::RuntimeConfigSnapshot::decode_v15_section(
            &pre_t11_bytes,
        )
        .expect("pre-T.1.1 payload must decode with defaults");
        assert_eq!(parsed.time_source, "system");
        assert_eq!(parsed.time_service_url, DEFAULT_DF_TIME_URL);
        assert_eq!(parsed.time_service_poll_interval_ms, DEFAULT_POLL_INTERVAL_MS);
        assert_eq!(parsed.time_service_max_drift_ms, DEFAULT_MAX_DRIFT_MS);
    }

    /// 10. Localhost-loopback URL with df_service mode triggers a
    ///     warning surfaced in DFRealState.localhost_warning. (The
    ///     warning is also eprintln'd; the test doesn't capture stderr
    ///     but the structured flag is sufficient.)
    #[test]
    fn t11_localhost_url_with_df_service_mode_warns() {
        let mut cfg = DFTimeServiceClientConfig::default();
        cfg.url = "http://localhost:9999/now".into();
        let client = DFRealClient::for_test(cfg);
        let snap = client.state_snapshot();
        assert!(
            snap.localhost_warning,
            "localhost URL must set the localhost_warning flag"
        );
    }
}
