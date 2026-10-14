//! Port Cat.3a — source-trust weighting at bond formation time.
//!
//! Per Learning #628 (Pre-Retrain Roadmap) and the C2 web-source
//! ordering preference, every new bond produced by the autonomous
//! research path gets its `weight` and its `SourceAttribution.trust`
//! scaled by the trust multiplier of the source domain. Trusted
//! canonical encyclopedias (Grokipedia) get a 1.0 multiplier,
//! Wikipedia gets 0.7 (post-T.8 deprioritization), DDG gets 0.5
//! (search engine, not authoritative), unknown domains get the
//! default 0.4.
//!
//! THIS IS RUNTIME WORK — NOT SCHEMA. The TrustTable persists in a
//! sidecar JSON file at `<brain_path>.trust.json` (mirroring the
//! `<brain_path>.pending.wal` and `<brain_path>.lb_session.wal`
//! sidecar pattern from T.3 / LB.1). No new v3 brain section is
//! added; the canonical 1.26 GB v3enc brain at /var/lib/lu/brain.lbr
//! continues to load with its existing schema and Cat.3a layers in
//! purely as a runtime overlay.
//!
//! Existing bonds in the canonical brain are NEVER mutated by
//! Cat.3a. Trust weighting applies only to NEW bonds at formation
//! time. The retrain output's quality depends on this — without
//! trust weighting, retrained bonds have no signal-vs-noise
//! hierarchy at the source level (Learning #628).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Lower bound on a trust multiplier. Below this, a bond carries
/// effectively no signal — clamped to 0.0 means "completely
/// untrusted; weight collapses to zero."
pub const MIN_TRUST_MULTIPLIER: f32 = 0.0;
/// Upper bound on a trust multiplier. Above 2.0, a bond from a
/// single source could outweigh multiple corroborating sources —
/// clamped to 2.0 to keep aggregation behaved.
pub const MAX_TRUST_MULTIPLIER: f32 = 2.0;
/// Lower bound on `default_multiplier`. Same range as per-entry
/// multipliers — set via `lu4d_configure { trust_default_multiplier }`.
pub const MIN_DEFAULT_MULTIPLIER: f32 = 0.0;
pub const MAX_DEFAULT_MULTIPLIER: f32 = 2.0;

/// Default canonical-source multiplier per the C2 hierarchy in
/// `feedback_web_search_order.md`. Grokipedia is the primary
/// canonical source.
pub const GROKIPEDIA_DOMAIN: &str = "grokipedia.com";
pub const GROKIPEDIA_DEFAULT_MULTIPLIER: f32 = 1.0;

/// Wikipedia is the secondary fallback — explicitly deprioritized
/// after T.8's C2 work (Learning #650).
pub const WIKIPEDIA_DOMAIN: &str = "en.wikipedia.org";
pub const WIKIPEDIA_DEFAULT_MULTIPLIER: f32 = 0.7;

/// DuckDuckGo is a search engine — meta-source, not authoritative.
pub const DUCKDUCKGO_DOMAIN: &str = "duckduckgo.com";
pub const DUCKDUCKGO_DEFAULT_MULTIPLIER: f32 = 0.5;

/// Default multiplier for any domain not in the table — applied to
/// random third-party blogs and unknown sources. Conservatively
/// below 0.5 so unknown-source bonds carry less weight than known
/// canonical sources by default.
pub const DEFAULT_UNKNOWN_MULTIPLIER: f32 = 0.4;

/// One trust-table entry: a domain-pattern → multiplier mapping with
/// an optional human-readable note.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrustEntry {
    /// Domain to match against the host portion of
    /// `SourceAttribution.url`. v1 supports exact match only — no
    /// glob/regex (defer to v2 if needed).
    pub domain_pattern: String,
    /// Multiplier applied to the bond weight at formation time.
    /// Clamped to `[MIN_TRUST_MULTIPLIER, MAX_TRUST_MULTIPLIER]` on
    /// every set/load call.
    pub multiplier: f32,
    /// Human-readable explanation of why this entry exists. Optional
    /// so the MCP `lu4d_trust_set` call doesn't require a note.
    pub note: Option<String>,
}

impl TrustEntry {
    /// Build an entry, clamping the multiplier into the valid range.
    pub fn new(
        domain_pattern: impl Into<String>,
        multiplier: f32,
        note: Option<String>,
    ) -> Self {
        Self {
            domain_pattern: domain_pattern.into(),
            multiplier: clamp_multiplier(multiplier),
            note,
        }
    }
}

/// Clamp a multiplier into the valid `[MIN, MAX]` range. Mirrors
/// the policy in the spec: above 2.0 → 2.0, below 0.0 → 0.0.
pub fn clamp_multiplier(m: f32) -> f32 {
    if m.is_nan() {
        return DEFAULT_UNKNOWN_MULTIPLIER;
    }
    if m < MIN_TRUST_MULTIPLIER {
        return MIN_TRUST_MULTIPLIER;
    }
    if m > MAX_TRUST_MULTIPLIER {
        return MAX_TRUST_MULTIPLIER;
    }
    m
}

/// The trust table itself. Persisted as a sidecar JSON file.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrustTable {
    /// Per-domain entries. Lookup is linear (first match wins) so
    /// later additions can override earlier ones; v1 expects exact
    /// match so order rarely matters in practice.
    pub entries: Vec<TrustEntry>,
    /// Default multiplier when no domain matches. Configurable via
    /// `lu4d_configure { trust_default_multiplier }`.
    pub default_multiplier: f32,
    /// Master switch. When false, `lookup()` returns 1.0 regardless
    /// of the table — useful for A/B testing the trust-weighting
    /// hypothesis. Toggleable via `lu4d_configure { trust_learning_enabled }`.
    /// Default true.
    pub learning_enabled: bool,
    /// Port T-Ingest.1: per-corpus trust multipliers, keyed by exact
    /// corpus string (e.g. `wikimedia/wikipedia`, `arxiv`). Used by
    /// `Brain::teach_with_source` on the teach path when the bond’s
    /// `SourceKind::TrainingCorpus.corpus` matches a key. Missing key →
    /// `default_multiplier`. Forward-compat: `#[serde(default)]` so
    /// pre-T-Ingest.1 sidecar JSON (which has no field) hydrates as
    /// an empty map.
    #[serde(default)]
    pub corpus_multipliers: HashMap<String, f32>,
}

impl Default for TrustTable {
    /// Default table matches the C2 hierarchy:
    ///
    /// ```text
    ///   grokipedia.com   → 1.0 (primary canonical)
    ///   en.wikipedia.org → 0.7 (secondary fallback)
    ///   duckduckgo.com   → 0.5 (meta-source)
    ///   *                → 0.4 (unknown / third-party / blogs)
    /// ```
    fn default() -> Self {
        Self {
            entries: vec![
                TrustEntry::new(
                    GROKIPEDIA_DOMAIN,
                    GROKIPEDIA_DEFAULT_MULTIPLIER,
                    Some("Primary canonical source per C2".to_string()),
                ),
                TrustEntry::new(
                    WIKIPEDIA_DOMAIN,
                    WIKIPEDIA_DEFAULT_MULTIPLIER,
                    Some("Secondary fallback after T.8 deprioritization".to_string()),
                ),
                TrustEntry::new(
                    DUCKDUCKGO_DOMAIN,
                    DUCKDUCKGO_DEFAULT_MULTIPLIER,
                    Some("Search engine — meta-source, not authoritative".to_string()),
                ),
            ],
            default_multiplier: DEFAULT_UNKNOWN_MULTIPLIER,
            learning_enabled: true,
            corpus_multipliers: HashMap::new(),
        }
    }
}

impl TrustTable {
    /// Build an empty (no entries, default `default_multiplier`)
    /// table — useful in tests.
    pub fn empty() -> Self {
        Self {
            entries: Vec::new(),
            default_multiplier: DEFAULT_UNKNOWN_MULTIPLIER,
            learning_enabled: true,
            corpus_multipliers: HashMap::new(),
        }
    }

    /// Reset to the default C2-aligned table.
    pub fn reset_to_defaults(&mut self) {
        *self = Self::default();
    }

    /// Look up the multiplier for a URL string. Extracts the host
    /// portion and matches it exact against `entries[*].domain_pattern`.
    /// When `learning_enabled` is false, returns 1.0 (pass-through).
    /// When the URL is empty, has no host, or doesn't match any
    /// entry, returns `default_multiplier`.
    pub fn lookup(&self, url: &str) -> f32 {
        if !self.learning_enabled {
            return 1.0;
        }
        let host = match extract_host(url) {
            Some(h) => h,
            None => return self.default_multiplier,
        };
        for e in &self.entries {
            if e.domain_pattern.eq_ignore_ascii_case(&host) {
                return clamp_multiplier(e.multiplier);
            }
        }
        self.default_multiplier
    }

    /// Set or update an entry. Returns the previous multiplier when
    /// the domain already existed, or `None` for a new entry.
    /// Validates `multiplier` lies in `[MIN, MAX]` — out-of-range
    /// rejections are surfaced as `Err` (clamping is reserved for
    /// the constructor / load path; the explicit set MUST surface
    /// invalid input to the caller).
    pub fn set(
        &mut self,
        domain: impl Into<String>,
        multiplier: f32,
        note: Option<String>,
    ) -> Result<Option<f32>, String> {
        if multiplier.is_nan() {
            return Err("multiplier must be a finite number, not NaN".to_string());
        }
        if multiplier < MIN_TRUST_MULTIPLIER || multiplier > MAX_TRUST_MULTIPLIER {
            return Err(format!(
                "multiplier out of bounds [{}, {}]: {}",
                MIN_TRUST_MULTIPLIER, MAX_TRUST_MULTIPLIER, multiplier
            ));
        }
        let domain = domain.into();
        for e in self.entries.iter_mut() {
            if e.domain_pattern.eq_ignore_ascii_case(&domain) {
                let prev = e.multiplier;
                e.multiplier = multiplier;
                if note.is_some() {
                    e.note = note;
                }
                return Ok(Some(prev));
            }
        }
        self.entries.push(TrustEntry::new(domain, multiplier, note));
        Ok(None)
    }

    /// Remove an entry. Returns true if removed, false if absent.
    /// After removal, the default multiplier still applies via
    /// `lookup()` for that domain.
    pub fn remove(&mut self, domain: &str) -> bool {
        let len_before = self.entries.len();
        self.entries
            .retain(|e| !e.domain_pattern.eq_ignore_ascii_case(domain));
        self.entries.len() != len_before
    }

    /// Set the default multiplier. Validates the bounds.
    pub fn set_default_multiplier(&mut self, m: f32) -> Result<f32, String> {
        if m.is_nan() {
            return Err("default_multiplier must be a finite number, not NaN".to_string());
        }
        if m < MIN_DEFAULT_MULTIPLIER || m > MAX_DEFAULT_MULTIPLIER {
            return Err(format!(
                "default_multiplier out of bounds [{}, {}]: {}",
                MIN_DEFAULT_MULTIPLIER, MAX_DEFAULT_MULTIPLIER, m
            ));
        }
        let prev = self.default_multiplier;
        self.default_multiplier = m;
        Ok(prev)
    }

    /// Toggle the master switch. Returns the prior value.
    pub fn set_learning_enabled(&mut self, v: bool) -> bool {
        let prev = self.learning_enabled;
        self.learning_enabled = v;
        prev
    }

    /// Port T-Ingest.1: insert/update a per-corpus trust multiplier.
    /// Validates `multiplier` lies in `[MIN_TRUST_MULTIPLIER, MAX_TRUST_MULTIPLIER]`.
    /// Returns the previous value when the corpus key existed, else `None`.
    pub fn set_corpus(
        &mut self,
        corpus: impl Into<String>,
        multiplier: f32,
    ) -> Result<Option<f32>, String> {
        if multiplier.is_nan() {
            return Err("multiplier must be a finite number, not NaN".to_string());
        }
        if multiplier < MIN_TRUST_MULTIPLIER || multiplier > MAX_TRUST_MULTIPLIER {
            return Err(format!(
                "multiplier out of bounds [{}, {}]: {}",
                MIN_TRUST_MULTIPLIER, MAX_TRUST_MULTIPLIER, multiplier
            ));
        }
        let key = corpus.into();
        let prev = self.corpus_multipliers.insert(key, multiplier);
        Ok(prev)
    }

    /// Port T-Ingest.1: remove a per-corpus entry. Returns `true` when
    /// removed, `false` when absent. After removal `default_multiplier`
    /// applies for that corpus on subsequent lookups.
    pub fn remove_corpus(&mut self, corpus: &str) -> bool {
        self.corpus_multipliers.remove(corpus).is_some()
    }

    /// Port T-Ingest.1: look up the multiplier for a corpus key. Returns
    /// `default_multiplier` on miss. When `learning_enabled` is false,
    /// returns 1.0 (master switch pass-through, mirrors `lookup()`).
    pub fn lookup_corpus(&self, corpus: &str) -> f32 {
        if !self.learning_enabled {
            return 1.0;
        }
        match self.corpus_multipliers.get(corpus) {
            Some(m) => clamp_multiplier(*m),
            None => {
                // OP-C1: namespaced-corpus fallback -- a "ns:detail" key
                // inherits its "ns" entry (e.g. stackexchange:stackoverflow
                // -> stackexchange). Preserves per-source attribution
                // granularity while resolving trust instead of collapsing to
                // the 0.4 unknown default.
                if let Some((prefix, _)) = corpus.split_once(':') {
                    if let Some(m) = self.corpus_multipliers.get(prefix) {
                        return clamp_multiplier(*m);
                    }
                }
                self.default_multiplier
            }
        }
    }

    /// Port T-Ingest.1: resolve the trust multiplier for a `SourceKind`.
    /// This is the canonical resolution used by `Brain::teach_with_source`
    /// when `attribution.trust` is `None`.
    ///
    ///   * `TrainingCorpus { corpus, .. }` -> `lookup_corpus(corpus)`
    ///   * `Research { url: Some(u), .. }` -> `lookup(u)` (URL host) —
    ///                                       preserves C2 Grokopedia ordering
    ///   * `Research { url: None, .. }`    -> `default_multiplier`
    ///   * `Teach { .. }`                  -> `default_multiplier`
    ///   * `Document { .. }`               -> `default_multiplier`
    ///   * `Unknown`                       -> `default_multiplier`
    pub fn resolve_for_kind(&self, kind: &crate::source::SourceKind) -> f32 {
        if !self.learning_enabled {
            return 1.0;
        }
        match kind {
            crate::source::SourceKind::TrainingCorpus { corpus, .. } => {
                self.lookup_corpus(corpus)
            }
            crate::source::SourceKind::Research { url: Some(u), .. } => {
                self.lookup(u)
            }
            crate::source::SourceKind::Research { url: None, .. } => self.default_multiplier,
            crate::source::SourceKind::Teach { .. } => self.default_multiplier,
            crate::source::SourceKind::Document { .. } => self.default_multiplier,
            // Port SourceKind-Variants — chain-driven research bonds carry
            // their per-URL trust multiplier already resolved by the
            // research apply_proposal path (Cat.3a sentinel pattern); the
            // SourceKind itself maps to the default multiplier here.
            crate::source::SourceKind::ChainedResearch { .. } => self.default_multiplier,
            // Port SourceKind-Variants — LB.2 promotions are John-trusted;
            // the multiplier defaults to neutral. A future Port can wire
            // a per-session multiplier table.
            crate::source::SourceKind::SessionPromote { .. } => self.default_multiplier,
            // Port SourceKind-Variants — Cat.4 packs are publisher-signed.
            // First ship maps to default; a future Port can resolve from
            // a publisher-trust table keyed on `publisher_pubkey_hex`.
            crate::source::SourceKind::PackInstall { .. } => self.default_multiplier,
            crate::source::SourceKind::Unknown => self.default_multiplier,
        }
    }

    /// Port T-Ingest.1: pre-populate `corpus_multipliers` with the locked
    /// curriculum trust hierarchy. Operator convenience — invoked only
    /// when `lu4d_trust_reset_to_defaults` is called with the
    /// `seed_curriculum_corpora` flag set. Existing entries are
    /// overwritten by the seed values.
    pub fn seed_curriculum_corpora(&mut self) {
        let seeds: &[(&str, f32)] = &[
            ("wikimedia/wikipedia", 1.0),
            // OP-RETRAIN-1 Tier-0 lexical foundation (authoritative).
            ("wordnet", 1.0),
            ("gcide", 0.9),
            ("wikimedia/structured-wikipedia", 1.0),
            ("ranjaykrishna/visual_genome", 1.0),
            ("Voxel51/GQA-Scene-Graph", 0.9),
            ("arxiv", 0.9),
            ("stackexchange", 0.8),
            // Phase-1 coding curriculum (declarative coding knowledge).
            ("code-search-net/code_search_net", 0.9),
            // OP-A0/C1: python-code structural corpus (pycode_extractor.py).
            // Was absent => fell to the 0.4 unknown default despite ~0.9 intent.
            ("python-code", 0.9),
            ("stackoverflow", 0.8),
            // Phase-3: StarCoderData multi-language AST-structural code corpus.
            ("starcoderdata", 0.9),
            // Phase-3b: The Stack v2 (code_extractor.py tree-sitter AST path).
            ("the-stack-v2", 0.9),
            // Phase-2a: Python official documentation (declarative coding concepts).
            ("python-docs", 0.95),
            // Phase-2b: MDN web-platform documentation (declarative JS/CSS/HTML/Web-API concepts).
            ("mdn", 0.95),
            ("grokipedia", 1.0),
            ("HuggingFaceH4/no_robots", 0.7),
            ("HuggingFaceH4/instruction-dataset", 0.7),
        ];
        for (k, m) in seeds {
            self.corpus_multipliers.insert((*k).to_string(), *m);
        }
    }

    /// Persist the table to a JSON sidecar at `path`. Uses
    /// `atomic_write_bytes` so a crashed write leaves the previous
    /// version on disk rather than a truncated file.
    pub fn save_to_path(&self, path: &Path) -> Result<(), String> {
        let bytes = serde_json::to_vec_pretty(self)
            .map_err(|e| format!("trust_table json serialize: {}", e))?;
        crate::brain::atomic_write_bytes(path, &bytes)
            .map_err(|e| format!("trust_table atomic_write: {:?}", e))
    }

    /// Load the table from a JSON sidecar at `path`. Missing file →
    /// returns the default table. Corrupt file → returns the
    /// default table AND logs a warning to stderr (never fails the
    /// brain load — same philosophy as T.3 / LB.1 WAL replay).
    pub fn load_from_path(path: &Path) -> Self {
        match std::fs::read(path) {
            Ok(bytes) => match serde_json::from_slice::<TrustTable>(&bytes) {
                Ok(mut t) => {
                    // Re-clamp on load so a hand-edited JSON with
                    // out-of-range values doesn't slip through.
                    for e in t.entries.iter_mut() {
                        e.multiplier = clamp_multiplier(e.multiplier);
                    }
                    // Port T-Ingest.1: re-clamp corpus_multipliers entries too.
                    for v in t.corpus_multipliers.values_mut() {
                        *v = clamp_multiplier(*v);
                    }
                    if t.default_multiplier.is_nan()
                        || t.default_multiplier < MIN_DEFAULT_MULTIPLIER
                        || t.default_multiplier > MAX_DEFAULT_MULTIPLIER
                    {
                        t.default_multiplier = DEFAULT_UNKNOWN_MULTIPLIER;
                    }
                    t
                }
                Err(e) => {
                    eprintln!(
                        "trust_table sidecar parse failed at {}: {} (defaulting)",
                        path.display(),
                        e
                    );
                    Self::default()
                }
            },
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Self::default(),
            Err(e) => {
                eprintln!(
                    "trust_table sidecar read failed at {}: {} (defaulting)",
                    path.display(),
                    e
                );
                Self::default()
            }
        }
    }
}

/// Compute the sidecar path for a given brain path:
/// `<brain_path>.trust.json`. Mirrors the T.3 / LB.1 WAL pattern.
pub fn sidecar_path_for(brain_path: &Path) -> PathBuf {
    let mut p = brain_path.to_path_buf();
    let mut name = p
        .file_name()
        .map(|s| s.to_os_string())
        .unwrap_or_else(|| std::ffi::OsString::from("brain.lbr"));
    name.push(".trust.json");
    p.set_file_name(name);
    p
}

/// Extract the host portion of a URL. Strips http(s):// prefix and
/// any path/query suffix. Returns `None` for empty/malformed URLs.
/// Lowercases the result so case-insensitive matching is symmetric.
pub fn extract_host(url: &str) -> Option<String> {
    let trimmed = url.trim();
    if trimmed.is_empty() {
        return None;
    }
    let after_scheme = trimmed
        .strip_prefix("https://")
        .or_else(|| trimmed.strip_prefix("http://"))
        .unwrap_or(trimmed);
    let host = after_scheme.split('/').next()?;
    // Drop any port suffix ("host:8080" → "host").
    let host = host.split(':').next()?;
    if host.is_empty() {
        return None;
    }
    Some(host.to_ascii_lowercase())
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Cat.3a tests (the 10-test contract) ---------------------

    #[test]
    fn cat3a_trust_lookup_returns_grokipedia_multiplier_1_0() {
        let t = TrustTable::default();
        let m = t.lookup("https://grokipedia.com/articles/photosynthesis");
        assert!((m - 1.0).abs() < 1e-6, "got {}", m);
    }

    #[test]
    fn cat3a_trust_lookup_returns_wikipedia_multiplier_0_7() {
        let t = TrustTable::default();
        let m = t.lookup("https://en.wikipedia.org/wiki/Photosynthesis");
        assert!((m - 0.7).abs() < 1e-6, "got {}", m);
    }

    #[test]
    fn cat3a_trust_lookup_returns_default_for_unknown_domain() {
        let t = TrustTable::default();
        let m = t.lookup("https://random-blog.example.com/some-post");
        assert!(
            (m - DEFAULT_UNKNOWN_MULTIPLIER).abs() < 1e-6,
            "got {}",
            m
        );
    }

    #[test]
    fn cat3a_trust_multiplier_applied_to_new_bond_weight() {
        // Direct verification: weight * multiplier matches what the
        // research-loop apply_proposal does. Indirect end-to-end
        // verification is in the brain integration test below.
        let t = TrustTable::default();
        let base_weight = 0.7_f32;
        let final_grok = base_weight * t.lookup("https://grokipedia.com/x");
        let final_wiki = base_weight * t.lookup("https://en.wikipedia.org/wiki/X");
        let final_blog = base_weight * t.lookup("https://blog.example/x");
        assert!((final_grok - 0.70).abs() < 1e-6, "got {}", final_grok);
        assert!((final_wiki - 0.49).abs() < 1e-6, "got {}", final_wiki);
        assert!((final_blog - 0.28).abs() < 1e-6, "got {}", final_blog);
    }

    #[test]
    fn cat3a_trust_does_not_mutate_existing_bonds() {
        // Existing-bond invariant: setting / removing trust entries
        // never touches the `multiplier` of any existing TrustEntry
        // for an unrelated domain.
        let mut t = TrustTable::default();
        let grok_before = t.lookup("https://grokipedia.com/x");
        // Add an unrelated entry.
        t.set("new-source.example", 0.9, None).unwrap();
        let grok_after = t.lookup("https://grokipedia.com/x");
        assert!((grok_before - grok_after).abs() < 1e-6);
        // Removing the unrelated entry doesn't touch grok either.
        t.remove("new-source.example");
        let grok_after2 = t.lookup("https://grokipedia.com/x");
        assert!((grok_before - grok_after2).abs() < 1e-6);
    }

    #[test]
    fn cat3a_trust_disabled_passes_through_base_weight() {
        let mut t = TrustTable::default();
        t.set_learning_enabled(false);
        // Every lookup returns 1.0 — pass-through.
        assert!((t.lookup("https://grokipedia.com/x") - 1.0).abs() < 1e-6);
        assert!((t.lookup("https://en.wikipedia.org/x") - 1.0).abs() < 1e-6);
        assert!((t.lookup("https://blog.example/x") - 1.0).abs() < 1e-6);
        assert!((t.lookup("") - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cat3a_trust_clamps_multiplier_above_2_0_and_below_0_0() {
        // The constructor clamps; the explicit `set` rejects.
        let e_high = TrustEntry::new("x", 5.0, None);
        assert!((e_high.multiplier - MAX_TRUST_MULTIPLIER).abs() < 1e-6);
        let e_low = TrustEntry::new("x", -3.0, None);
        assert!((e_low.multiplier - MIN_TRUST_MULTIPLIER).abs() < 1e-6);
        // Explicit set rejects out-of-range.
        let mut t = TrustTable::empty();
        assert!(t.set("x", 5.0, None).is_err());
        assert!(t.set("x", -3.0, None).is_err());
        // In-bounds value succeeds.
        assert!(t.set("x", 1.5, None).is_ok());
    }

    #[test]
    fn cat3a_trust_set_persists_across_restart() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!(
            "cat3a_persist_test_{}.trust.json",
            std::process::id()
        ));
        // Make sure we start clean.
        let _ = std::fs::remove_file(&path);
        let mut t = TrustTable::default();
        t.set("my-canonical.example", 1.5, Some("test entry".into()))
            .unwrap();
        t.save_to_path(&path).expect("save");
        // "Restart": throw the in-memory table away, reload from disk.
        drop(t);
        let t2 = TrustTable::load_from_path(&path);
        let m = t2.lookup("https://my-canonical.example/foo");
        assert!((m - 1.5).abs() < 1e-6, "got {}", m);
        // The default entries also survive the round-trip.
        let grok = t2.lookup("https://grokipedia.com/x");
        assert!((grok - 1.0).abs() < 1e-6, "got {}", grok);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn cat3a_trust_remove_falls_back_to_default() {
        let mut t = TrustTable::default();
        // Wikipedia is in the default table at 0.7; remove it, the
        // default (0.4) takes over.
        let removed = t.remove(WIKIPEDIA_DOMAIN);
        assert!(removed);
        let m = t.lookup("https://en.wikipedia.org/wiki/X");
        assert!(
            (m - DEFAULT_UNKNOWN_MULTIPLIER).abs() < 1e-6,
            "got {}",
            m
        );
        // Removing again is a no-op.
        assert!(!t.remove(WIKIPEDIA_DOMAIN));
    }

    #[test]
    fn cat3a_trust_reset_restores_default_table() {
        let mut t = TrustTable::default();
        t.set("strange.example", 0.1, None).unwrap();
        t.remove(GROKIPEDIA_DOMAIN);
        t.set_default_multiplier(0.05).unwrap();
        t.set_learning_enabled(false);
        // Reset.
        t.reset_to_defaults();
        assert!((t.lookup("https://grokipedia.com/x") - 1.0).abs() < 1e-6);
        assert!((t.lookup("https://en.wikipedia.org/x") - 0.7).abs() < 1e-6);
        assert!((t.lookup("https://duckduckgo.com/?q=x") - 0.5).abs() < 1e-6);
        assert!(
            (t.lookup("https://blog.example/x") - DEFAULT_UNKNOWN_MULTIPLIER).abs() < 1e-6
        );
        assert!(t.learning_enabled);
        // The unrelated entry from earlier is gone.
        assert!(t.entries.iter().all(|e| e.domain_pattern != "strange.example"));
    }

    // ---- Auxiliary tests (host extraction, edge cases) -----------

    #[test]
    fn cat3a_extract_host_handles_common_url_shapes() {
        assert_eq!(extract_host("https://grokipedia.com"), Some("grokipedia.com".into()));
        assert_eq!(
            extract_host("https://en.wikipedia.org/wiki/Photosynthesis"),
            Some("en.wikipedia.org".into())
        );
        assert_eq!(extract_host("http://EXAMPLE.com:8080/x"), Some("example.com".into()));
        assert_eq!(extract_host(""), None);
        assert_eq!(extract_host("   "), None);
        // No-scheme URL still parses.
        assert_eq!(extract_host("example.org/path"), Some("example.org".into()));
    }

    #[test]
    fn cat3a_trust_set_returns_previous_multiplier_for_existing_domain() {
        let mut t = TrustTable::default();
        let prev = t
            .set(WIKIPEDIA_DOMAIN, 0.3, Some("demoting wikipedia further".into()))
            .unwrap();
        assert!(prev.is_some());
        assert!((prev.unwrap() - 0.7).abs() < 1e-6);
        let m = t.lookup("https://en.wikipedia.org/x");
        assert!((m - 0.3).abs() < 1e-6);
    }

    #[test]
    fn cat3a_trust_load_missing_file_yields_default_table() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!(
            "cat3a_missing_file_{}.trust.json",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&path);
        let t = TrustTable::load_from_path(&path);
        assert_eq!(t, TrustTable::default());
    }

    #[test]
    fn a0c1_coding_corpora_trust_resolves() {
        let mut t = TrustTable::default();
        t.seed_curriculum_corpora();
        let approx = |a: f32, b: f32| (a - b).abs() < 1e-6;
        assert!(approx(t.lookup_corpus("code-search-net/code_search_net"), 0.9), "CSN");
        assert!(approx(t.lookup_corpus("python-docs"), 0.95), "pydocs");
        assert!(approx(t.lookup_corpus("mdn"), 0.95), "mdn");
        assert!(approx(t.lookup_corpus("stackexchange"), 0.8), "stackexchange");
        assert!(approx(t.lookup_corpus("python-code"), 0.9), "python-code was 0.4");
        assert!(approx(t.lookup_corpus("stackexchange:python"), 0.8), "SE prefix fallback");
        assert!(approx(t.lookup_corpus("stackexchange:stackoverflow"), 0.8), "SE prefix fallback 2");
        assert!(approx(t.lookup_corpus("totally-unknown-xyz"), t.default_multiplier), "unknown->default");
    }
}
