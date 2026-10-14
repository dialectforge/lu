//! `AuthenticatedTime` — wall-clock ms + monotonic anchor + signature +
//! expiry. The fundamental unit of trust for time-sensitive operations.
//!
//! When `signature.is_some()`, the wall-clock came from a source whose
//! key the brain trusts (eventually: DF Time Service). When `None`, the
//! source was the OS and time is treated as unauthenticated — the
//! anomaly detector still runs on it, but scar-sensitive ops will
//! refuse.

use super::monotonic::MonotonicTimestamp;
use serde::{Deserialize, Serialize};

/// Identifier for the clock source that produced this timestamp.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SourceId(String);

impl SourceId {
    pub fn new<S: Into<String>>(s: S) -> Self {
        SourceId(s.into())
    }
    /// Built-in id for the OS clock.
    pub fn system() -> Self {
        SourceId("system".into())
    }
    /// Built-in id for the DF Time Service stub (T.1.1 will replace with
    /// the real signed source).
    pub fn df_time_service_stub() -> Self {
        SourceId("df-time-service-stub".into())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Opaque cryptographic signature over `(utc_ms, monotonic_at_fetch,
/// fetched_from, expiry_ms)`. T.1 uses a stub byte vector; T.1.1 will
/// replace with real Ed25519 over a canonical encoding.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Signature(pub Vec<u8>);

impl Signature {
    pub fn stub_for(utc_ms: u64) -> Self {
        // Deterministic placeholder so tests can compare. Real signing
        // lands in T.1.1 alongside the DF Time Service server.
        let mut bytes = b"DF-TIME-STUB:".to_vec();
        bytes.extend_from_slice(&utc_ms.to_le_bytes());
        Signature(bytes)
    }
}

/// A trusted (or untrusted) anchor binding wall-clock to monotonic.
#[derive(Debug, Clone)]
pub struct AuthenticatedTime {
    /// Unix epoch milliseconds.
    pub utc_ms: u64,
    /// Monotonic timestamp captured at the moment of fetch. Used to
    /// translate "current monotonic now" into "current UTC now" without
    /// re-fetching from the source on every call.
    pub monotonic_at_fetch: MonotonicTimestamp,
    /// Where this anchor came from.
    pub fetched_from: SourceId,
    /// `Some` for authenticated sources (DF Time Service); `None` for
    /// `SystemClockSource`.
    pub signature: Option<Signature>,
    /// Wall-clock ms after which this anchor is considered stale and
    /// must be refreshed. Set by the `ClockSource`.
    pub expiry_ms: u64,
}

impl AuthenticatedTime {
    /// Construct an unsigned (system-source) anchor. Used by
    /// `SystemClockSource` and by tests.
    pub fn unsigned(
        utc_ms: u64,
        monotonic_at_fetch: MonotonicTimestamp,
        fetched_from: SourceId,
        expiry_ms: u64,
    ) -> Self {
        AuthenticatedTime {
            utc_ms,
            monotonic_at_fetch,
            fetched_from,
            signature: None,
            expiry_ms,
        }
    }

    /// Construct a signed anchor. Used by `DFTimeServiceSource::stub`
    /// today; T.1.1 will replace with a real Ed25519-signed factory.
    pub fn signed(
        utc_ms: u64,
        monotonic_at_fetch: MonotonicTimestamp,
        fetched_from: SourceId,
        signature: Signature,
        expiry_ms: u64,
    ) -> Self {
        AuthenticatedTime {
            utc_ms,
            monotonic_at_fetch,
            fetched_from,
            signature: Some(signature),
            expiry_ms,
        }
    }

    /// True iff this anchor carries a signature from a trusted source.
    pub fn is_authenticated(&self) -> bool {
        self.signature.is_some()
    }

    /// Has this anchor passed its refresh deadline (relative to the
    /// supplied wall-clock now)?
    pub fn is_expired(&self, now_ms: u64) -> bool {
        now_ms >= self.expiry_ms
    }

    /// Project this anchor forward using a fresh monotonic reading,
    /// yielding the current wall-clock ms. The projection lets us call
    /// `current_utc_ms()` without round-tripping to the source on every
    /// call.
    pub fn project_now_ms(&self, now_mono: MonotonicTimestamp) -> u64 {
        let elapsed = now_mono.duration_since_ms(self.monotonic_at_fetch);
        self.utc_ms.saturating_add(elapsed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signed_marks_authenticated() {
        let t = AuthenticatedTime::signed(
            42,
            MonotonicTimestamp::from_nanos(0),
            SourceId::df_time_service_stub(),
            Signature::stub_for(42),
            42 + 60_000,
        );
        assert!(t.is_authenticated());
    }

    #[test]
    fn unsigned_is_not_authenticated() {
        let t = AuthenticatedTime::unsigned(
            1,
            MonotonicTimestamp::from_nanos(0),
            SourceId::system(),
            60_001,
        );
        assert!(!t.is_authenticated());
    }

    #[test]
    fn project_now_advances() {
        let t = AuthenticatedTime::unsigned(
            1_000,
            MonotonicTimestamp::from_nanos(0),
            SourceId::system(),
            60_001,
        );
        let projected = t.project_now_ms(MonotonicTimestamp::from_nanos(5_000_000)); // +5 ms
        assert_eq!(projected, 1_005);
    }

    #[test]
    fn expiry_check() {
        let t = AuthenticatedTime::unsigned(
            100,
            MonotonicTimestamp::from_nanos(0),
            SourceId::system(),
            200,
        );
        assert!(!t.is_expired(150));
        assert!(t.is_expired(200));
        assert!(t.is_expired(300));
    }
}
