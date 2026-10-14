//! Monotonic clock — wraps `std::time::Instant` for ordering. Never goes
//! backward; immune to wall-clock skew, NTP slew, or daylight savings.
//!
//! **Use this for ordering scar-sensitive ops.** Wall-clock (UTC ms) is
//! for surface display only.

use std::time::Instant;

/// Process-relative monotonic timestamp. Comparable across calls within
/// one process; meaningless across restarts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MonotonicTimestamp {
    /// Nanoseconds since process start (or some other anchor). The value
    /// is opaque; only ordering and `duration_since` are meaningful.
    nanos: u128,
}

impl MonotonicTimestamp {
    /// Construct from raw nanoseconds. Useful in tests; in production
    /// always go through `MonotonicClock::now()`.
    pub fn from_nanos(nanos: u128) -> Self {
        MonotonicTimestamp { nanos }
    }

    /// Raw nanoseconds. Meaningful only for ordering / subtraction.
    pub fn as_nanos(&self) -> u128 {
        self.nanos
    }

    /// Milliseconds elapsed since `earlier`. Saturates at 0 if `earlier`
    /// is somehow newer (which can only happen with hand-constructed
    /// timestamps in tests; `MonotonicClock::now()` never goes backward).
    pub fn duration_since_ms(&self, earlier: MonotonicTimestamp) -> u64 {
        self.nanos
            .saturating_sub(earlier.nanos)
            .saturating_div(1_000_000) as u64
    }
}

/// Monotonic clock entry point.
pub struct MonotonicClock;

impl MonotonicClock {
    /// Capture a monotonic timestamp anchored to a process-local origin.
    pub fn now() -> MonotonicTimestamp {
        // Anchor lazily — first call sets origin; later calls measure delta.
        // Using `Instant::now().elapsed_since(origin)` would require a
        // OnceLock; we instead rely on `Instant`'s internal monotonicity
        // and convert to nanos relative to a per-process origin.
        static ORIGIN: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
        let origin = *ORIGIN.get_or_init(Instant::now);
        let elapsed = Instant::now().saturating_duration_since(origin);
        MonotonicTimestamp {
            nanos: elapsed.as_nanos(),
        }
    }

    /// Milliseconds elapsed between `earlier` and `now()`.
    pub fn duration_since_ms(earlier: MonotonicTimestamp) -> u64 {
        Self::now().duration_since_ms(earlier)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_nanos_round_trip() {
        let t = MonotonicTimestamp::from_nanos(123_456_789);
        assert_eq!(t.as_nanos(), 123_456_789);
    }

    #[test]
    fn duration_since_basic() {
        let a = MonotonicTimestamp::from_nanos(1_000_000); // 1 ms
        let b = MonotonicTimestamp::from_nanos(5_000_000); // 5 ms
        assert_eq!(b.duration_since_ms(a), 4);
    }

    #[test]
    fn duration_since_saturates() {
        let a = MonotonicTimestamp::from_nanos(5_000_000);
        let b = MonotonicTimestamp::from_nanos(1_000_000);
        assert_eq!(b.duration_since_ms(a), 0);
    }
}
