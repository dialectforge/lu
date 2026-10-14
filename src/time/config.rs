//! Knobs for the T.1 time substrate.
//!
//! Defaults are the production-safe choices documented in the project log
//! #639:
//!  - 4-hour refresh interval for authenticated anchors
//!  - zero-tolerance for backward jumps (any backward step trips the
//!    detector — a refresh from a single source must never go back)
//!  - 7-day forward-jump tolerance (covers reasonable scheduled-task
//!    lag, NTP skew, suspended laptops; still flags catastrophic
//!    misconfiguration like a clock advanced to next year)

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimeConfig {
    /// How often the brain re-fetches an authenticated anchor (ms).
    pub refresh_interval_ms: u64,
    /// Maximum tolerated backward wall-clock jump between consecutive
    /// fetches. Default 0 (zero tolerance — any backward jump fires).
    pub max_backward_jump_ms: u64,
    /// Maximum tolerated forward wall-clock jump between consecutive
    /// fetches. Default 7 days.
    pub max_forward_jump_ms: u64,
}

impl TimeConfig {
    pub const DEFAULT_REFRESH_MS: u64 = 4 * 60 * 60 * 1000; // 4 hours
    pub const DEFAULT_MAX_BACKWARD_MS: u64 = 0; // zero tolerance
    pub const DEFAULT_MAX_FORWARD_MS: u64 = 7 * 24 * 60 * 60 * 1000; // 7 days
}

impl Default for TimeConfig {
    fn default() -> Self {
        TimeConfig {
            refresh_interval_ms: TimeConfig::DEFAULT_REFRESH_MS,
            max_backward_jump_ms: TimeConfig::DEFAULT_MAX_BACKWARD_MS,
            max_forward_jump_ms: TimeConfig::DEFAULT_MAX_FORWARD_MS,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_production_safe() {
        let c = TimeConfig::default();
        assert_eq!(c.refresh_interval_ms, 4 * 60 * 60 * 1000);
        assert_eq!(c.max_backward_jump_ms, 0);
        assert_eq!(c.max_forward_jump_ms, 7 * 24 * 60 * 60 * 1000);
    }
}
