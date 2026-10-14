//! Port Cat.3b — hub-degree caps at bond formation + read time.
//!
//! Per Learning #638 ("Hub-degree management") and Learning #628
//! (Pre-Retrain Roadmap). Words like "system", "process", "function"
//! naturally accumulate millions of incoming bonds during training.
//! Runaway hubs slow recall (every query touches the hub), dilute
//! signal (hub neighbors dominate spread activation regardless of
//! relevance), and make the brain harder to consolidate / decay /
//! scar cleanly.
//!
//! Defense in depth — caps are enforced at BOTH ends:
//!   * Write-side: refuse new bonds (or evict an existing low-weight
//!     bond) when a hub is at cap. Keeps the canonical brain smaller,
//!     faster, simpler.
//!   * Read-side: at query time, when reading bonds for spread
//!     activation, filter to top-K by weight. Handles legacy /
//!     pre-cap brains where some hubs already exceed the cap.
//!
//! THIS IS RUNTIME WORK — NOT SCHEMA. The HubDegreeConfig persists
//! in a JSON sidecar at `<brain_path>.hub_degree.json` (mirroring
//! the Cat.3a `<brain>.trust.json` pattern from Learning #655).
//! No new v3 brain section is added; the canonical 1.26 GB v3enc
//! brain at /var/lib/lu/brain.lbr continues to load with its
//! existing schema and Cat.3b layers in purely as a runtime overlay.
//!
//! Existing bonds in the canonical brain are NEVER mutated by Cat.3b.
//! Read-side filters them at query time only — stored bonds are
//! unchanged. Write-side enforcement applies only to NEW bonds at
//! formation time.
//!
//! T.7 invariant: HubDegreeConfig has no impact on the encryption
//! path. Sidecar JSON at `<brain>.hub_degree.json` is plaintext (it
//! is policy, not knowledge).

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Lower bound on `max_in_degree` / `max_out_degree`. Below 100 the
/// caps would clip legitimate semantic neighborhoods (e.g. a Person
/// node with one bond per relative + one per address + one per
/// phone). 100 keeps configuration sensible.
pub const MIN_MAX_DEGREE: u32 = 100;

/// Upper bound on `max_in_degree` / `max_out_degree`. 1_000_000 is
/// well above the largest legitimate single-node degree we ever
/// expect — the canonical brain at 440K nodes / 10.8M bonds has p99
/// in_degree well below 100K. The bound exists to surface
/// configuration mistakes at the MCP layer, not to constrain real use.
pub const MAX_MAX_DEGREE: u32 = 1_000_000;

/// Default `max_in_degree`. Conservative starting point: well above
/// legitimate domain hubs (which sit in the hundreds), well below
/// the runaway hubs the retrain would otherwise produce. Easy to
/// raise if it's clipping legitimate signal; harder to lower without
/// data loss. See spec §6 in Learning #638.
pub const DEFAULT_MAX_IN_DEGREE: u32 = 5000;

/// Default `max_out_degree`. Same rationale as `DEFAULT_MAX_IN_DEGREE`.
/// Most nodes are net consumers (in_degree > out_degree), but
/// symmetric defaults make configuration easier to reason about.
pub const DEFAULT_MAX_OUT_DEGREE: u32 = 5000;

/// Eviction priority — when a hub is at cap and a new bond is being
/// considered, this decides what to drop.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HubPriority {
    /// Drop the new bond. Existing bonds are sticky. Conservative for
    /// brains where you want training-time bonds to be canonical.
    NewestFirst,
    /// Drop the oldest existing bond if the new one is to be admitted.
    /// Useful when hubs accumulate stale bonds over time.
    OldestFirst,
    /// Drop the lowest-weight existing bond if the new bond's weight
    /// exceeds it; otherwise drop the new bond. This is the default —
    /// it preserves the highest-signal bonds at hubs.
    LowestWeightFirst,
}

impl Default for HubPriority {
    fn default() -> Self {
        HubPriority::LowestWeightFirst
    }
}

impl HubPriority {
    pub fn as_str(self) -> &'static str {
        match self {
            HubPriority::NewestFirst => "NewestFirst",
            HubPriority::OldestFirst => "OldestFirst",
            HubPriority::LowestWeightFirst => "LowestWeightFirst",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "NewestFirst" => Some(HubPriority::NewestFirst),
            "OldestFirst" => Some(HubPriority::OldestFirst),
            "LowestWeightFirst" => Some(HubPriority::LowestWeightFirst),
            _ => None,
        }
    }
}

/// HubDegreeConfig — the full Cat.3b knob set. Persisted as a JSON
/// sidecar at `<brain_path>.hub_degree.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HubDegreeConfig {
    /// Hard cap on incoming bonds per node. Default 5000.
    pub max_in_degree: u32,
    /// Hard cap on outgoing bonds per node. Default 5000.
    pub max_out_degree: u32,
    /// Refuse new bonds at training/research time when cap hit.
    /// Default true.
    pub write_enforcement: bool,
    /// Filter at query time when reading bonds for spread activation.
    /// Default true.
    pub read_enforcement: bool,
    /// When at cap and choosing what to drop. Default
    /// `LowestWeightFirst`.
    pub priority: HubPriority,
    /// Node kinds that bypass the cap entirely. e.g. ["Concept",
    /// "Person"] would let high-degree concepts grow unbounded.
    /// Default empty (no exemptions).
    pub exempt_node_kinds: Vec<String>,
}

impl Default for HubDegreeConfig {
    fn default() -> Self {
        Self {
            max_in_degree: DEFAULT_MAX_IN_DEGREE,
            max_out_degree: DEFAULT_MAX_OUT_DEGREE,
            write_enforcement: true,
            read_enforcement: true,
            priority: HubPriority::default(),
            exempt_node_kinds: Vec::new(),
        }
    }
}

impl HubDegreeConfig {
    /// True when the given node kind should bypass the cap entirely.
    /// Case-insensitive match.
    pub fn is_exempt_kind(&self, kind: &str) -> bool {
        self.exempt_node_kinds
            .iter()
            .any(|k| k.eq_ignore_ascii_case(kind))
    }

    /// Validate `max_in_degree` / `max_out_degree` against the bounds
    /// constants. Returns an error string with the concrete bounds
    /// on rejection.
    pub fn validate_max_degree(v: u32) -> Result<(), String> {
        if v < MIN_MAX_DEGREE {
            return Err(format!(
                "max_degree out of bounds [{}, {}]: {}",
                MIN_MAX_DEGREE, MAX_MAX_DEGREE, v
            ));
        }
        if v > MAX_MAX_DEGREE {
            return Err(format!(
                "max_degree out of bounds [{}, {}]: {}",
                MIN_MAX_DEGREE, MAX_MAX_DEGREE, v
            ));
        }
        Ok(())
    }

    /// Set `max_in_degree`. Returns the prior value on success.
    pub fn set_max_in_degree(&mut self, v: u32) -> Result<u32, String> {
        Self::validate_max_degree(v)?;
        let prev = self.max_in_degree;
        self.max_in_degree = v;
        Ok(prev)
    }

    /// Set `max_out_degree`. Returns the prior value on success.
    pub fn set_max_out_degree(&mut self, v: u32) -> Result<u32, String> {
        Self::validate_max_degree(v)?;
        let prev = self.max_out_degree;
        self.max_out_degree = v;
        Ok(prev)
    }

    /// Set `write_enforcement`. Returns the prior value.
    pub fn set_write_enforcement(&mut self, v: bool) -> bool {
        let prev = self.write_enforcement;
        self.write_enforcement = v;
        prev
    }

    /// Set `read_enforcement`. Returns the prior value.
    pub fn set_read_enforcement(&mut self, v: bool) -> bool {
        let prev = self.read_enforcement;
        self.read_enforcement = v;
        prev
    }

    /// Set `priority`. Returns the prior value.
    pub fn set_priority(&mut self, p: HubPriority) -> HubPriority {
        let prev = self.priority;
        self.priority = p;
        prev
    }

    /// Reset to defaults.
    pub fn reset_to_defaults(&mut self) {
        *self = Self::default();
    }

    /// Persist to a JSON sidecar at `path`. Atomic write so a
    /// crashed write leaves the previous version on disk rather
    /// than a truncated file.
    pub fn save_to_path(&self, path: &Path) -> Result<(), String> {
        let bytes = serde_json::to_vec_pretty(self)
            .map_err(|e| format!("hub_degree json serialize: {}", e))?;
        crate::brain::atomic_write_bytes(path, &bytes)
            .map_err(|e| format!("hub_degree atomic_write: {:?}", e))
    }

    /// Load from a JSON sidecar at `path`. Missing file → returns the
    /// default config. Corrupt file → returns the default config AND
    /// logs a warning to stderr (never fails the brain load — same
    /// philosophy as T.3 / LB.1 WAL replay and Cat.3a trust table).
    /// Out-of-bounds values are clamped on load so a hand-edited JSON
    /// doesn't slip through.
    pub fn load_from_path(path: &Path) -> Self {
        match std::fs::read(path) {
            Ok(bytes) => match serde_json::from_slice::<HubDegreeConfig>(&bytes) {
                Ok(mut c) => {
                    if c.max_in_degree < MIN_MAX_DEGREE {
                        c.max_in_degree = MIN_MAX_DEGREE;
                    }
                    if c.max_in_degree > MAX_MAX_DEGREE {
                        c.max_in_degree = MAX_MAX_DEGREE;
                    }
                    if c.max_out_degree < MIN_MAX_DEGREE {
                        c.max_out_degree = MIN_MAX_DEGREE;
                    }
                    if c.max_out_degree > MAX_MAX_DEGREE {
                        c.max_out_degree = MAX_MAX_DEGREE;
                    }
                    c
                }
                Err(e) => {
                    eprintln!(
                        "hub_degree sidecar parse failed at {}: {} (defaulting)",
                        path.display(),
                        e
                    );
                    Self::default()
                }
            },
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Self::default(),
            Err(e) => {
                eprintln!(
                    "hub_degree sidecar read failed at {}: {} (defaulting)",
                    path.display(),
                    e
                );
                Self::default()
            }
        }
    }
}

/// Compute the sidecar path for a given brain path:
/// `<brain_path>.hub_degree.json`. Mirrors the Cat.3a trust sidecar
/// pattern.
pub fn sidecar_path_for(brain_path: &Path) -> PathBuf {
    let mut p = brain_path.to_path_buf();
    let mut name = p
        .file_name()
        .map(|s| s.to_os_string())
        .unwrap_or_else(|| std::ffi::OsString::from("brain.lbr"));
    name.push(".hub_degree.json");
    p.set_file_name(name);
    p
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;

    fn tmp_path(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "lu4_cat3b_test_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        dir.join(name)
    }

    // --- Cat.3b config tests --------------------------------------

    #[test]
    fn cat3b_hub_config_defaults_match_spec() {
        let c = HubDegreeConfig::default();
        assert_eq!(c.max_in_degree, 5000);
        assert_eq!(c.max_out_degree, 5000);
        assert!(c.write_enforcement);
        assert!(c.read_enforcement);
        assert_eq!(c.priority, HubPriority::LowestWeightFirst);
        assert!(c.exempt_node_kinds.is_empty());
    }

    #[test]
    fn cat3b_hub_config_loads_defaults_when_sidecar_missing() {
        let path = tmp_path("missing.hub_degree.json");
        // Path explicitly does not exist (we only created the parent
        // dir, not the file).
        let c = HubDegreeConfig::load_from_path(&path);
        assert_eq!(c, HubDegreeConfig::default());
    }

    #[test]
    fn cat3b_hub_config_persists_via_sidecar() {
        let path = tmp_path("brain.lbr.hub_degree.json");
        let mut c = HubDegreeConfig::default();
        c.set_max_in_degree(1234).unwrap();
        c.set_max_out_degree(2345).unwrap();
        c.set_write_enforcement(false);
        c.set_read_enforcement(false);
        c.set_priority(HubPriority::NewestFirst);
        c.exempt_node_kinds = vec!["Concept".to_string(), "Person".to_string()];
        c.save_to_path(&path).unwrap();
        let loaded = HubDegreeConfig::load_from_path(&path);
        assert_eq!(loaded, c);
    }

    #[test]
    fn cat3b_hub_config_clamps_out_of_bounds_on_load() {
        let path = tmp_path("oob.hub_degree.json");
        // Write a JSON that bypasses set_*'s validation.
        let raw = r#"{"max_in_degree":50,"max_out_degree":2000000,"write_enforcement":true,"read_enforcement":true,"priority":"LowestWeightFirst","exempt_node_kinds":[]}"#;
        fs::write(&path, raw).unwrap();
        let c = HubDegreeConfig::load_from_path(&path);
        assert_eq!(c.max_in_degree, MIN_MAX_DEGREE);
        assert_eq!(c.max_out_degree, MAX_MAX_DEGREE);
    }

    #[test]
    fn cat3b_hub_config_corrupt_sidecar_logs_and_defaults() {
        let path = tmp_path("corrupt.hub_degree.json");
        fs::write(&path, "{not json at all").unwrap();
        let c = HubDegreeConfig::load_from_path(&path);
        assert_eq!(c, HubDegreeConfig::default());
    }

    #[test]
    fn cat3b_hub_config_validate_rejects_out_of_bounds() {
        let mut c = HubDegreeConfig::default();
        assert!(c.set_max_in_degree(50).is_err());
        assert!(c.set_max_in_degree(2_000_000).is_err());
        assert!(c.set_max_out_degree(50).is_err());
        assert!(c.set_max_out_degree(2_000_000).is_err());
        // Boundary values are accepted.
        assert!(c.set_max_in_degree(MIN_MAX_DEGREE).is_ok());
        assert!(c.set_max_in_degree(MAX_MAX_DEGREE).is_ok());
    }

    #[test]
    fn cat3b_hub_config_exempt_kind_match_is_case_insensitive() {
        let mut c = HubDegreeConfig::default();
        c.exempt_node_kinds = vec!["Concept".to_string()];
        assert!(c.is_exempt_kind("Concept"));
        assert!(c.is_exempt_kind("concept"));
        assert!(c.is_exempt_kind("CONCEPT"));
        assert!(!c.is_exempt_kind("Person"));
    }

    #[test]
    fn cat3b_hub_config_reset_to_defaults_round_trips() {
        let mut c = HubDegreeConfig::default();
        c.set_max_in_degree(123).unwrap();
        c.set_write_enforcement(false);
        c.exempt_node_kinds = vec!["X".to_string()];
        c.reset_to_defaults();
        assert_eq!(c, HubDegreeConfig::default());
    }

    #[test]
    fn cat3b_hub_priority_parse_round_trips() {
        for p in [
            HubPriority::NewestFirst,
            HubPriority::OldestFirst,
            HubPriority::LowestWeightFirst,
        ] {
            assert_eq!(HubPriority::parse(p.as_str()), Some(p));
        }
        assert!(HubPriority::parse("garbage").is_none());
    }

    #[test]
    fn cat3b_sidecar_path_for_uses_brain_filename() {
        let p = sidecar_path_for(Path::new("/var/lib/lu/brain.lbr"));
        assert_eq!(p, Path::new("/var/lib/lu/brain.lbr.hub_degree.json"));
    }
}
