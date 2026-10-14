//! Anomaly detector — flags backward jumps and large forward jumps in
//! the wall-clock relative to the monotonic clock.
//!
//! **Safety property:** when `check()` returns `Err(AnomalyKind)`, the
//! brain MUST flip into degraded mode (scar-sensitive ops refuse).
//! `Brain::time_is_authenticated()` consults the most recent anomaly
//! state alongside the source's signature flag.
//!
//! ### T.1.1 additions
//!
//! Three new variants surface DF Time Service client failures so they
//! flow through the same sticky-anomaly mechanism as backward/forward
//! jumps. The polling thread in `df_client.rs` calls `record_anomaly`
//! out-of-band (the failures don't go through `check()` because they
//! aren't "comparing two anchors" — they're "the trusted-source fetch
//! failed"). The variants:
//!  - `TimeServiceSignatureInvalid` — Ed25519 verify failed.
//!  - `TimeServiceDriftExcess` — server-claimed time differs from
//!    locally-projected time by more than `max_drift_ms`.
//!  - `TimeServiceUnreachable` — endpoint failed for >= N consecutive
//!    polls.

use super::authenticated::AuthenticatedTime;
use super::config::TimeConfig;
use super::monotonic::MonotonicTimestamp;
use serde::{Deserialize, Serialize};

/// What went wrong on the most recent `check()`, or what the DF time
/// client reported on its most recent poll.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnomalyKind {
    /// Wall-clock went backward by `delta_ms`.
    BackwardJump { delta_ms: u64 },
    /// Wall-clock jumped forward beyond `max_forward_jump_ms`.
    LargeForwardJump { delta_ms: u64 },
    /// The supplied anchor was already past its expiry.
    ExpiredAnchor { age_ms: u64 },
    /// T.1.1: DF Time Service Ed25519 verification failed.
    TimeServiceSignatureInvalid,
    /// T.1.1: DF Time Service-supplied UTC differs from locally-projected
    /// UTC by more than `max_drift_ms`.
    TimeServiceDriftExcess { drift_ms: u64 },
    /// T.1.1: DF Time Service endpoint unreachable for `consecutive_failures`
    /// polls in a row.
    TimeServiceUnreachable { consecutive_failures: u32 },
}

/// Stateful detector. One instance per Brain; the most recent
/// observation is held internally.
#[derive(Debug)]
pub struct AnomalyDetector {
    cfg: TimeConfig,
    last_wall_clock_ms: Option<u64>,
    last_monotonic: Option<MonotonicTimestamp>,
    /// Sticky anomaly flag: once tripped, stays tripped until
    /// `clear_anomaly()` is called (typically after a successful clean
    /// refresh).
    last_anomaly: Option<AnomalyKind>,
}

impl AnomalyDetector {
    pub fn new(cfg: TimeConfig) -> Self {
        AnomalyDetector {
            cfg,
            last_wall_clock_ms: None,
            last_monotonic: None,
            last_anomaly: None,
        }
    }

    /// Most recent anomaly, if any. `None` means clean.
    pub fn last_anomaly(&self) -> Option<&AnomalyKind> {
        self.last_anomaly.as_ref()
    }

    /// Reset the sticky anomaly flag. Call after a known-clean refresh.
    pub fn clear_anomaly(&mut self) {
        self.last_anomaly = None;
    }

    /// T.1.1: record an anomaly out-of-band. Used by the DF Time
    /// Service polling thread to surface signature/drift/unreachable
    /// failures into the same sticky-flag mechanism that
    /// `Brain::warnings()` and `lu_status` already consult.
    pub fn record_anomaly(&mut self, kind: AnomalyKind) {
        self.last_anomaly = Some(kind);
    }

    /// Inspect a fresh anchor. On success, updates internal state and
    /// returns `Ok(())`. On failure, sets `last_anomaly` and returns
    /// the `AnomalyKind`.
    pub fn check(&mut self, t: &AuthenticatedTime) -> Result<(), AnomalyKind> {
        // Backward / forward jumps are only meaningful relative to a
        // prior reading. The first call always succeeds and primes the
        // detector.
        if let (Some(prev_wall), Some(_prev_mono)) =
            (self.last_wall_clock_ms, self.last_monotonic)
        {
            if t.utc_ms < prev_wall {
                let delta_ms = prev_wall - t.utc_ms;
                if delta_ms > self.cfg.max_backward_jump_ms {
                    let kind = AnomalyKind::BackwardJump { delta_ms };
                    self.last_anomaly = Some(kind.clone());
                    return Err(kind);
                }
            } else {
                let delta_ms = t.utc_ms - prev_wall;
                if delta_ms > self.cfg.max_forward_jump_ms {
                    let kind = AnomalyKind::LargeForwardJump { delta_ms };
                    self.last_anomaly = Some(kind.clone());
                    return Err(kind);
                }
            }
        }
        self.last_wall_clock_ms = Some(t.utc_ms);
        self.last_monotonic = Some(t.monotonic_at_fetch);
        // Successful check clears any prior sticky anomaly. (A future
        // refinement would only clear after N consecutive clean checks.)
        self.last_anomaly = None;
        Ok(())
    }

    /// Out-of-band check: does a *now-projected* anchor age exceed its
    /// expiry? Used by `Brain::time_is_authenticated()` so an
    /// authenticated-but-stale anchor still degrades.
    pub fn check_freshness(&mut self, t: &AuthenticatedTime, now_ms: u64) -> Result<(), AnomalyKind> {
        if t.is_expired(now_ms) {
            let age_ms = now_ms.saturating_sub(t.expiry_ms);
            let kind = AnomalyKind::ExpiredAnchor { age_ms };
            self.last_anomaly = Some(kind.clone());
            return Err(kind);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::time::authenticated::SourceId;

    fn anchor(utc_ms: u64, mono_ns: u128, expiry: u64) -> AuthenticatedTime {
        AuthenticatedTime::unsigned(
            utc_ms,
            MonotonicTimestamp::from_nanos(mono_ns),
            SourceId::system(),
            expiry,
        )
    }

    #[test]
    fn first_check_always_succeeds() {
        let mut d = AnomalyDetector::new(TimeConfig::default());
        assert!(d.check(&anchor(1_000, 0, 60_001)).is_ok());
        assert!(d.last_anomaly().is_none());
    }

    #[test]
    fn backward_jump_fires_with_zero_tolerance() {
        let cfg = TimeConfig {
            max_backward_jump_ms: 0,
            ..TimeConfig::default()
        };
        let mut d = AnomalyDetector::new(cfg);
        assert!(d.check(&anchor(1_000, 0, 60_001)).is_ok());
        let res = d.check(&anchor(900, 1_000_000, 60_001));
        match res {
            Err(AnomalyKind::BackwardJump { delta_ms }) => assert_eq!(delta_ms, 100),
            other => panic!("unexpected: {:?}", other),
        }
        // Sticky.
        assert!(matches!(d.last_anomaly(), Some(AnomalyKind::BackwardJump { .. })));
    }

    #[test]
    fn forward_jump_within_tolerance_passes() {
        let cfg = TimeConfig {
            max_forward_jump_ms: 10_000,
            ..TimeConfig::default()
        };
        let mut d = AnomalyDetector::new(cfg);
        d.check(&anchor(1_000, 0, 60_001)).unwrap();
        d.check(&anchor(1_500, 1_000_000, 60_500)).unwrap();
        assert!(d.last_anomaly().is_none());
    }

    #[test]
    fn large_forward_jump_fires() {
        let cfg = TimeConfig {
            max_forward_jump_ms: 1_000,
            ..TimeConfig::default()
        };
        let mut d = AnomalyDetector::new(cfg);
        d.check(&anchor(1_000, 0, 60_001)).unwrap();
        let res = d.check(&anchor(10_000, 1_000_000, 70_001));
        match res {
            Err(AnomalyKind::LargeForwardJump { delta_ms }) => assert_eq!(delta_ms, 9_000),
            other => panic!("unexpected: {:?}", other),
        }
    }

    #[test]
    fn clear_anomaly_resets() {
        let cfg = TimeConfig {
            max_backward_jump_ms: 0,
            ..TimeConfig::default()
        };
        let mut d = AnomalyDetector::new(cfg);
        d.check(&anchor(1_000, 0, 60_001)).unwrap();
        let _ = d.check(&anchor(900, 1_000_000, 60_001));
        assert!(d.last_anomaly().is_some());
        d.clear_anomaly();
        assert!(d.last_anomaly().is_none());
    }

    #[test]
    fn expired_anchor_fires_freshness() {
        let mut d = AnomalyDetector::new(TimeConfig::default());
        let a = anchor(100, 0, 200);
        let res = d.check_freshness(&a, 250);
        match res {
            Err(AnomalyKind::ExpiredAnchor { age_ms }) => assert_eq!(age_ms, 50),
            other => panic!("unexpected: {:?}", other),
        }
    }

    #[test]
    fn t11_record_anomaly_sets_sticky_flag() {
        let mut d = AnomalyDetector::new(TimeConfig::default());
        d.record_anomaly(AnomalyKind::TimeServiceSignatureInvalid);
        assert!(matches!(
            d.last_anomaly(),
            Some(AnomalyKind::TimeServiceSignatureInvalid)
        ));
        d.record_anomaly(AnomalyKind::TimeServiceDriftExcess { drift_ms: 9_999 });
        match d.last_anomaly() {
            Some(AnomalyKind::TimeServiceDriftExcess { drift_ms }) => {
                assert_eq!(*drift_ms, 9_999);
            }
            other => panic!("unexpected: {:?}", other),
        }
        d.clear_anomaly();
        assert!(d.last_anomaly().is_none());
    }
}
