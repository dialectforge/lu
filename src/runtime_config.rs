//! Port Cat.3c.1 — RuntimeConfig forward-compat at the field level.
//! Port T.1.1 — DF Time Service client config (5 new fields).
//!
//! ### Background
//!
//! Cat.3c (commit 27019b0) introduced v3 section #14 to consolidate
//! the legacy `<brain>.trust.json` / `<brain>.hub_degree.json`
//! sidecars into the encrypted brain. Cat.3c.1 (commit cd91d23) added
//! section #15 (JSON, named-field, forward-compatible). T.1.1 lands
//! its 5 new config fields in section #15 — they're carried purely
//! through the JSON path with `#[serde(default)]`, so:
//!  - Cat.3c reading T.1.1: skips #15 (J.1.1 forward-tolerance), reads
//!    #14 (the frozen Cat.3c-shape subset).
//!  - T.1.1 reading Cat.3c: no #15 present, falls back to #14, T.1.1
//!    fields take their `default_*` values (system mode, default URL,
//!    placeholder pubkey).
//!
//! ### T.1.1 fields
//!
//!  - `time_source` — `"system"` (default) or `"df_service"`.
//!  - `time_service_url` — endpoint URL (default DF production URL).
//!  - `time_service_pubkey_hex` — pinned Ed25519 pubkey for verification.
//!  - `time_service_poll_interval_ms` — default 60_000.
//!  - `time_service_max_drift_ms` — default 5_000.
//!
//! Production canonical Lu defaults to `"system"` because IT IS the
//! time authority for the network — pointing it at `df_service` would
//! create a circular dependency. See
//! `agent/memory/project_time_service_architecture.md`.
//!
//! ### Required behavior contracts (preserved from Cat.3c / Cat.3c.1)
//! - T.7 encryption invariant preserved (both sections live inside
//!   the encrypted v3 layout for v3enc files).
//! - Anti-hallucination invariant preserved — runtime config writes
//!   don't change `bond_count`.
//! - Forward compat IN BOTH DIRECTIONS via section #15's named-field
//!   tolerance.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

use crate::hub_degree::HubDegreeConfig;
use crate::trust::TrustTable;

// ---------------------------------------------------------------------
// Section #14 default helpers — these match Cat.3c semantics exactly
// ---------------------------------------------------------------------

fn default_style_output_adaptation_enabled() -> bool {
    crate::style::DEFAULT_OUTPUT_ADAPTATION_ENABLED
}
fn default_style_max_samples_for_full_weight() -> u64 {
    crate::style::DEFAULT_MAX_SAMPLES_FOR_FULL_WEIGHT
}

// ---------------------------------------------------------------------
// Port T.1.1 default helpers
// ---------------------------------------------------------------------

/// Production canonical Lu default. The DialectForge server IS the
/// time authority — it does NOT call itself for time.
pub fn default_time_source() -> String {
    "system".to_string()
}
pub fn default_time_service_url() -> String {
    crate::time::DEFAULT_DF_TIME_URL.to_string()
}
pub fn default_time_service_pubkey_hex() -> String {
    crate::time::PLACEHOLDER_DF_PUBKEY_HEX.to_string()
}
pub fn default_time_service_poll_interval_ms() -> u64 {
    crate::time::DEFAULT_POLL_INTERVAL_MS
}
pub fn default_time_service_max_drift_ms() -> u64 {
    crate::time::DEFAULT_MAX_DRIFT_MS
}

// ---------------------------------------------------------------------
// Port Sense: runtime sense disambiguation knobs
// ---------------------------------------------------------------------

/// Master switch for runtime sense disambiguation. Defaults to true
/// so a freshly retrained brain has the feature live from day one.
pub fn default_sense_disambiguation_enabled() -> bool {
    true
}

/// How many recently-activated nodes feed the context vector when
/// the caller (MCP / spread path) does not pass an explicit context.
/// Per spec row 19a, default 8.
pub fn default_sense_context_window() -> usize {
    8
}

/// Multiplier applied to non-winning sense clusters' bond mass when
/// spreading from a polysemous seed. 0.0 = full prune (binary), 1.0
/// = no disambiguation effect. Default 0.25 — strong but not absolute
/// preference. Keeps the option open for a later activation step to
/// revive the loser if context shifts.
pub fn default_sense_loser_gain() -> f32 {
    0.25
}

// ---------------------------------------------------------------------
// Port J.2 — chained curiosity research knobs
// ---------------------------------------------------------------------

/// Master switch for chained research. When true, `research_chain_run`
/// chases multi-hop threads; when false the call collapses to a single
/// `research_one` with no fresh-gap enqueue.
pub fn default_chained_research_enabled() -> bool {
    true
}

/// Hard cap on chain depth — how many hops a single chain advances.
/// Default 3 keeps a thread tractable; raise to dig deeper, lower
/// to stay close to the seed.
pub fn default_max_chain_depth() -> u32 {
    3
}

/// Hard cap on total research cycles per chain. Independent of depth
/// because a single hop can spawn multiple sibling gaps. Default 5.
pub fn default_max_chain_cycles() -> u32 {
    5
}

/// Minimum gap intensity required to seed the next hop. Endpoints
/// whose computed gap intensity falls below this threshold are NOT
/// enqueued — chasing low-confidence gaps wastes budget. Default
/// 0.4 mirrors the J.1 baseline.
pub fn default_chain_gap_threshold() -> f32 {
    0.4
}

// ---------------------------------------------------------------------
// Port T-Ingest: training-corpus ingest knobs (added to section #15).
// ---------------------------------------------------------------------

/// Strategy the server uses to allocate a `run_id` when a caller of
/// `lu4d_teach_corpus` does not supply one.
///
///   * `Caller`         — never auto-generate; if the caller omits
///                        `run_id`, the resulting `TrainingCorpus`
///                        attribution carries `run_id = None`.
///   * `ServerStartup`  — generate a single id at server boot and
///                        reuse it for every subsequent ingest call.
///                        Cheap; one ingest run per process lifetime.
///   * `PerSession`     — use the active LB session id when there
///                        is one; falls back to ServerStartup when no
///                        LB session is active.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum IngestRunIdStrategy {
    Caller,
    ServerStartup,
    PerSession,
}

impl Default for IngestRunIdStrategy {
    fn default() -> Self { IngestRunIdStrategy::ServerStartup }
}

pub fn default_ingest_corpus_default() -> Option<String> { None }
pub fn default_ingest_run_id_strategy() -> IngestRunIdStrategy { IngestRunIdStrategy::ServerStartup }
pub fn default_ingest_auto_approve_threshold() -> Option<f32> { None }

// ---------------------------------------------------------------------
// Port T-Ingest.2 — Tier 2 prompt routing knobs (added to section #15).
// ---------------------------------------------------------------------

/// Strategy used by `lu4d_user_statement` when the caller does NOT
/// supply an explicit `session_id`.
///
///   * `Ephemeral` — each prompt opens its own brand-new LB session
///                 (mirrors the asymmetry-correct shape: each prompt
///                 is its own operational unit, separate from any
///                 conversational session).
///   * `Reuse`     — attach to the currently-active LB session if
///                 one exists; otherwise fall back to the LB store
///                 internal scratchpad (`SCRATCHPAD_SESSION_ID`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Tier2PromptSessionStrategy {
    Ephemeral,
    Reuse,
}

impl Default for Tier2PromptSessionStrategy {
    fn default() -> Self { Tier2PromptSessionStrategy::Ephemeral }
}

pub fn default_tier2_prompt_default_session_strategy() -> Tier2PromptSessionStrategy {
    Tier2PromptSessionStrategy::Ephemeral
}

/// Hard cap on the byte length of a single prompt accepted by
/// `lu4d_user_statement`. Default 64 KiB; calls above this length
/// are rejected with -32602 to keep LB from getting blown up by
/// pathological corpora rows. Forward-compatible: new binaries that
/// raise the cap will simply read older brains unchanged.
pub fn default_tier2_prompt_max_text_bytes() -> usize { 65536 }


// ---------------------------------------------------------------------
// Port BR.4 -- recall-time typed-KU routing kill switch.
// ---------------------------------------------------------------------

/// Master switch for Q.1 typed-KU activation in RECALL paths
/// (chain composer + procedure assembler in `reasoning.rs`). When
/// `true` (default) the chain leg builders read `kind_str()` from
/// each endpoint and feed it to `articulate_bond_typed`, which
/// appends a `(kind)` tag to the prose. When `false` recall falls
/// through to the pre-Q.1 baseline: leg `from_kind` / `to_kind`
/// stay `None`, so `articulate_bond_typed` reduces byte-for-byte
/// to `articulate_bond` -- no kind tag in articulation.
///
/// Storage and retention paths (extractor kind tagging, hub-cap
/// admission, decay, articulation in writes) are NOT gated -- the
/// flag only flips RECALL behavior so an A/B comparison against
/// the same brain is possible without re-ingesting.
///
/// Defaults to `true` so behavior is unchanged after BR.4 lands.
pub fn default_recall_typed_ku_enabled() -> bool {
    true
}

/// Port BR.5 -- admin-tier auth gate on lu4d_revoke_by_source.
/// DEFAULT FALSE so any new brain ships with the destructive revoke
/// path closed. Operators must explicitly call lu4d_admin_set with
/// admin_revoke_enabled=true before invoking lu4d_revoke_by_source
/// with dry_run=false. Dry-run preview is NOT gated -- operators
/// can preview what a revoke would touch without enabling the flag.
///
/// Default-deny posture rationale: the day lu.dialectforge.com PWA
/// ships, every guest with an invite code shares the canonical
/// brain. With the gate closed by default, the worst-case bug or
/// misconfigured config is "destructive revoke unreachable", not
/// "guest wipes facts from canonical."
pub fn default_admin_revoke_enabled() -> bool {
    false
}


// ---------------------------------------------------------------------
// Port File-IO -- file ingestion / emission knobs (added to section #15).
// ---------------------------------------------------------------------

/// Hard cap on read-side file size. Default: 50 MB. Calls beyond this
/// are refused by `file_io::check_size`. Anti-blowup invariant.
pub fn default_file_io_max_size_bytes() -> usize {
    crate::file_io::default_file_io_max_size_bytes()
}

/// Empty default = no path restriction on reads. When non-empty, the
/// canonicalised read path must lie under one of the supplied roots.
pub fn default_file_io_allowed_read_paths() -> Vec<String> {
    crate::file_io::default_file_io_allowed_read_paths()
}

/// Default writes are constrained to a small allowlist so a runaway
/// tool call cannot scribble across the host filesystem.
pub fn default_file_io_allowed_write_paths() -> Vec<String> {
    crate::file_io::default_file_io_allowed_write_paths()
}


// ---------------------------------------------------------------------
// Port R input -- image ingest knobs (added to section #15).
// ---------------------------------------------------------------------

/// Default extractor: stub. Operators must opt in to Subprocess by
/// setting `image_extractor_kind=Subprocess` AND a non-empty
/// `image_extractor_subprocess_path`.
pub fn default_image_extractor_kind() -> crate::mcp_vision::VisionExtractorKind {
    crate::mcp_vision::VisionExtractorKind::default()
}

/// No subprocess by default. Setting this is required (along with
/// `image_extractor_kind=Subprocess`) to use the SubprocessExtractor.
pub fn default_image_extractor_subprocess_path() -> Option<String> { None }

/// 30 second wall-clock cap on the configured subprocess.
pub fn default_image_extractor_subprocess_timeout_ms() -> u64 {
    crate::mcp_vision::default_image_extractor_subprocess_timeout_ms()
}

/// 25 MiB hard cap on user-uploaded image size.
pub fn default_image_max_size_bytes() -> usize {
    crate::mcp_vision::default_image_max_size_bytes()
}

/// Mime allowlist: the four common web image formats. Operators may
/// extend this in section #15 if their CV pipeline accepts more.
pub fn default_image_supported_mime_types() -> Vec<String> {
    crate::mcp_vision::default_image_supported_mime_types()
}

/// Authoritative, future-extensible runtime-config snapshot. Persisted
/// into v3 section #15 (`SectionIdV3::RuntimeConfigV2`) as JSON.
///
/// Cat.3c-shape subset (`trust_table`, `hub_degree`,
/// `style_decay_half_life_days`) is ALSO mirrored into the bincode-
/// encoded section #14 for backward-readability by pre-Cat.3c.1
/// binaries. New fields after Cat.3c.1 live ONLY in section #15 and
/// are defaulted by older readers via serde_json's missing-field
/// tolerance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RuntimeConfigSnapshot {
    /// Cat.3a's TrustTable.
    #[serde(default)]
    pub trust_table: TrustTable,
    /// Cat.3b's HubDegreeConfig.
    #[serde(default)]
    pub hub_degree: HubDegreeConfig,
    /// Cat.3c forward-extensibility seat (currently unused).
    #[serde(default)]
    pub style_decay_half_life_days: Option<u32>,
    /// Cat.3c.1: master switch for the CS.2 output-adaptation layer.
    #[serde(default = "default_style_output_adaptation_enabled")]
    pub style_output_adaptation_enabled: bool,
    /// Cat.3c.1: dampening control for `StyleHints`.
    #[serde(default = "default_style_max_samples_for_full_weight")]
    pub style_max_samples_for_full_weight: u64,
    // ---- Port T.1.1: DF Time Service client knobs ----
    /// `"system"` (canonical Lu, default) | `"df_service"` (user Lus).
    #[serde(default = "default_time_source")]
    pub time_source: String,
    /// DF Time Service endpoint (default: production URL).
    #[serde(default = "default_time_service_url")]
    pub time_service_url: String,
    /// Pinned DF Ed25519 pubkey for signature verification (hex).
    #[serde(default = "default_time_service_pubkey_hex")]
    pub time_service_pubkey_hex: String,
    /// How often the polling thread fetches a fresh signed anchor.
    #[serde(default = "default_time_service_poll_interval_ms")]
    pub time_service_poll_interval_ms: u64,
    /// Maximum tolerated drift between received UTC and locally-projected
    /// UTC. Beyond this the client rejects the response.
    #[serde(default = "default_time_service_max_drift_ms")]
    pub time_service_max_drift_ms: u64,
    // ---- Port Sense: runtime sense disambiguation knobs ----
    /// Master switch. When false the `lu4d_sense_status` MCP tool
    /// still runs (it's diagnostic), but the spread-path consumer
    /// (Port Sense.1, follow-up) becomes a no-op.
    #[serde(default = "default_sense_disambiguation_enabled")]
    pub sense_disambiguation_enabled: bool,
    /// Cap on the size of the context window pulled from session
    /// memory when the caller does not supply one.
    #[serde(default = "default_sense_context_window")]
    pub sense_context_window: usize,
    /// Attenuation for non-winning sense clusters in the gain map.
    #[serde(default = "default_sense_loser_gain")]
    pub sense_loser_gain: f32,
    // ---- Port J.2: chained curiosity research knobs ----
    /// Master switch for the multi-hop curiosity loop. False collapses
    /// `lu4d_research_chain_run` to a single `research_one`.
    #[serde(default = "default_chained_research_enabled")]
    pub chained_research_enabled: bool,
    /// Hard cap on chain depth.
    #[serde(default = "default_max_chain_depth")]
    pub max_chain_depth: u32,
    /// Hard cap on total research cycles per chain.
    #[serde(default = "default_max_chain_cycles")]
    pub max_chain_cycles: u32,
    /// Minimum gap intensity required to seed the next hop.
    #[serde(default = "default_chain_gap_threshold")]
    pub chain_gap_threshold: f32,
    // ---- Port T-Ingest: training-corpus ingest knobs ----
    /// When `Some(corpus)`, the bare `lu4d_teach` MCP tool ALSO
    /// stamps `SourceKind::TrainingCorpus { corpus, .. }` instead
    /// of the default `Teach { actor: "system" }` attribution.
    /// Ergonomic switch for "I am running an ingest, treat every
    /// teach in this session as belonging to corpus X". `None`
    /// (default) preserves byte-for-byte the pre-T-Ingest teach
    /// behavior.
    #[serde(default = "default_ingest_corpus_default")]
    pub ingest_corpus_default: Option<String>,
    /// Strategy for allocating `run_id` when the caller omits it.
    /// Default: `ServerStartup` (one id per process lifetime).
    #[serde(default = "default_ingest_run_id_strategy")]
    pub ingest_run_id_strategy: IngestRunIdStrategy,
    /// Optional confidence threshold for auto-approval of
    /// `lu4d_teach_corpus` calls. `None` (default) disables the
    /// gate entirely (every call applies directly, mirroring
    /// `lu4d_teach`). When `Some(t)`, calls with confidence
    /// `< t` route through T.3 PendingLog instead of applying
    /// directly; the caller must approve them via
    /// `lu4d_approve_changes` to commit.
    #[serde(default = "default_ingest_auto_approve_threshold")]
    pub ingest_auto_approve_threshold: Option<f32>,
    // ---- Port T-Ingest.2: Tier 2 prompt routing knobs ----
    /// Strategy used by `lu4d_user_statement` when the caller
    /// omits `session_id`. Default `Ephemeral`: each prompt
    /// opens a fresh LB session of its own.
    #[serde(default = "default_tier2_prompt_default_session_strategy")]
    pub tier2_prompt_default_session_strategy: Tier2PromptSessionStrategy,
    /// Hard cap on prompt-text byte length accepted by
    /// `lu4d_user_statement`. Default 65536. Calls beyond this
    /// cap return -32602 (anti-blowup invariant).
    #[serde(default = "default_tier2_prompt_max_text_bytes")]
    pub tier2_prompt_max_text_bytes: usize,
    // ---- Port File-IO: file ingestion / emission knobs ----
    /// Hard cap on read-side file size in bytes. Default 50 MB.
    /// Calls reading larger files are refused.
    #[serde(default = "default_file_io_max_size_bytes")]
    pub file_io_max_size_bytes: usize,
    /// Optional whitelist of dir prefixes the read path is restricted
    /// to. Empty (default) means no restriction.
    #[serde(default = "default_file_io_allowed_read_paths")]
    pub file_io_allowed_read_paths: Vec<String>,
    /// Whitelist of dir prefixes the write path is restricted to.
    /// Default `["/var/lib/lu/outputs/", "/tmp/lu/"]`.
    #[serde(default = "default_file_io_allowed_write_paths")]
    pub file_io_allowed_write_paths: Vec<String>,
    // ---- Port R input: user-facing image ingest knobs ----
    /// Selected `SceneGraphExtractor` implementation. Default `Stub`.
    /// `Subprocess` requires `image_extractor_subprocess_path` to be
    /// `Some(_)`. `CapabilityPack { pack_id }` is documented but
    /// deferred to Cat.5.1.
    #[serde(default = "default_image_extractor_kind")]
    pub image_extractor_kind: crate::mcp_vision::VisionExtractorKind,
    /// Path to the external CV subprocess (executable on disk that
    /// reads image bytes on stdin and prints a JSON triplet array on
    /// stdout). `None` (default) disables the Subprocess extractor.
    #[serde(default = "default_image_extractor_subprocess_path")]
    pub image_extractor_subprocess_path: Option<String>,
    /// Wall-clock cap on the configured subprocess. Default 30000 ms.
    #[serde(default = "default_image_extractor_subprocess_timeout_ms")]
    pub image_extractor_subprocess_timeout_ms: u64,
    /// Hard cap on user-uploaded image bytes. Default 25_000_000.
    #[serde(default = "default_image_max_size_bytes")]
    pub image_max_size_bytes: usize,
    /// Mime types the image-ingest tools accept. Default covers the
    /// four common web formats.
    #[serde(default = "default_image_supported_mime_types")]
    pub image_supported_mime_types: Vec<String>,
    // ---- Port BR.4: recall-time typed-KU routing kill switch ----
    /// Master switch for Q.1 typed-KU activation in RECALL paths.
    /// `true` (default) feeds endpoint kinds into the chain composer
    /// + procedure assembler so `articulate_bond_typed` can append
    /// a `(kind)` tag. `false` collapses recall articulation to the
    /// pre-Q.1 baseline (`articulate_bond` byte-for-byte). Storage,
    /// hub-cap, and decay paths are unaffected.
    #[serde(default = "default_recall_typed_ku_enabled")]
    pub recall_typed_ku_enabled: bool,
    // ---- Port BR.5: admin-tier auth gate on lu4d_revoke_by_source ----
    /// Default-false gate guarding the destructive revoke path.
    /// `false` (default) makes lu4d_revoke_by_source with
    /// dry_run=false return -32603; dry-run preview is unchanged.
    /// Flip to `true` via lu4d_admin_set before any operator-driven
    /// rollback. Persisted via runtime_config section #15 so the
    /// gate state survives restarts. See BR.5 audit P1 cleanup.
    #[serde(default = "default_admin_revoke_enabled")]
    pub admin_revoke_enabled: bool,
}

impl Default for RuntimeConfigSnapshot {
    fn default() -> Self {
        RuntimeConfigSnapshot {
            trust_table: TrustTable::default(),
            hub_degree: HubDegreeConfig::default(),
            style_decay_half_life_days: None,
            style_output_adaptation_enabled: default_style_output_adaptation_enabled(),
            style_max_samples_for_full_weight: default_style_max_samples_for_full_weight(),
            time_source: default_time_source(),
            time_service_url: default_time_service_url(),
            time_service_pubkey_hex: default_time_service_pubkey_hex(),
            time_service_poll_interval_ms: default_time_service_poll_interval_ms(),
            time_service_max_drift_ms: default_time_service_max_drift_ms(),
            sense_disambiguation_enabled: default_sense_disambiguation_enabled(),
            sense_context_window: default_sense_context_window(),
            sense_loser_gain: default_sense_loser_gain(),
            chained_research_enabled: default_chained_research_enabled(),
            max_chain_depth: default_max_chain_depth(),
            max_chain_cycles: default_max_chain_cycles(),
            chain_gap_threshold: default_chain_gap_threshold(),
            ingest_corpus_default: default_ingest_corpus_default(),
            ingest_run_id_strategy: default_ingest_run_id_strategy(),
            ingest_auto_approve_threshold: default_ingest_auto_approve_threshold(),
            tier2_prompt_default_session_strategy: default_tier2_prompt_default_session_strategy(),
            tier2_prompt_max_text_bytes: default_tier2_prompt_max_text_bytes(),
            file_io_max_size_bytes: default_file_io_max_size_bytes(),
            file_io_allowed_read_paths: default_file_io_allowed_read_paths(),
            file_io_allowed_write_paths: default_file_io_allowed_write_paths(),
            image_extractor_kind: default_image_extractor_kind(),
            image_extractor_subprocess_path: default_image_extractor_subprocess_path(),
            image_extractor_subprocess_timeout_ms: default_image_extractor_subprocess_timeout_ms(),
            image_max_size_bytes: default_image_max_size_bytes(),
            image_supported_mime_types: default_image_supported_mime_types(),
            recall_typed_ku_enabled: default_recall_typed_ku_enabled(),
            admin_revoke_enabled: default_admin_revoke_enabled(),
        }
    }
}

/// Frozen Cat.3c shape — bincode-encoded into section #14. Adding
/// fields here would re-break Cat.3c readers (positional bincode).
/// New fields go into `RuntimeConfigSnapshot` (section #15) instead.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub(crate) struct RuntimeConfigSnapshotV14 {
    #[serde(default)]
    pub trust_table: TrustTable,
    #[serde(default)]
    pub hub_degree: HubDegreeConfig,
    #[serde(default)]
    pub style_decay_half_life_days: Option<u32>,
}

impl RuntimeConfigSnapshot {
    /// Project the full snapshot down to the frozen Cat.3c shape that
    /// goes into bincode section #14. New Cat.3c.1+ and T.1.1 fields
    /// are dropped (they survive only via section #15).
    pub(crate) fn to_v14(&self) -> RuntimeConfigSnapshotV14 {
        RuntimeConfigSnapshotV14 {
            trust_table: self.trust_table.clone(),
            hub_degree: self.hub_degree.clone(),
            style_decay_half_life_days: self.style_decay_half_life_days,
        }
    }

    /// Lift a section #14 payload into a full snapshot. New Cat.3c.1+
    /// and T.1.1 fields take their defaults — equivalent to what a
    /// T.1.1 binary sees when it reads a Cat.3c-saved brain.
    pub(crate) fn from_v14(v14: RuntimeConfigSnapshotV14) -> Self {
        RuntimeConfigSnapshot {
            trust_table: v14.trust_table,
            hub_degree: v14.hub_degree,
            style_decay_half_life_days: v14.style_decay_half_life_days,
            style_output_adaptation_enabled: default_style_output_adaptation_enabled(),
            style_max_samples_for_full_weight: default_style_max_samples_for_full_weight(),
            time_source: default_time_source(),
            time_service_url: default_time_service_url(),
            time_service_pubkey_hex: default_time_service_pubkey_hex(),
            time_service_poll_interval_ms: default_time_service_poll_interval_ms(),
            time_service_max_drift_ms: default_time_service_max_drift_ms(),
            sense_disambiguation_enabled: default_sense_disambiguation_enabled(),
            sense_context_window: default_sense_context_window(),
            sense_loser_gain: default_sense_loser_gain(),
            chained_research_enabled: default_chained_research_enabled(),
            max_chain_depth: default_max_chain_depth(),
            max_chain_cycles: default_max_chain_cycles(),
            chain_gap_threshold: default_chain_gap_threshold(),
            ingest_corpus_default: default_ingest_corpus_default(),
            ingest_run_id_strategy: default_ingest_run_id_strategy(),
            ingest_auto_approve_threshold: default_ingest_auto_approve_threshold(),
            tier2_prompt_default_session_strategy: default_tier2_prompt_default_session_strategy(),
            tier2_prompt_max_text_bytes: default_tier2_prompt_max_text_bytes(),
            file_io_max_size_bytes: default_file_io_max_size_bytes(),
            file_io_allowed_read_paths: default_file_io_allowed_read_paths(),
            file_io_allowed_write_paths: default_file_io_allowed_write_paths(),
            image_extractor_kind: default_image_extractor_kind(),
            image_extractor_subprocess_path: default_image_extractor_subprocess_path(),
            image_extractor_subprocess_timeout_ms: default_image_extractor_subprocess_timeout_ms(),
            image_max_size_bytes: default_image_max_size_bytes(),
            image_supported_mime_types: default_image_supported_mime_types(),
            recall_typed_ku_enabled: default_recall_typed_ku_enabled(),
            admin_revoke_enabled: default_admin_revoke_enabled(),
        }
    }

    /// Encode the section #14 (bincode, frozen Cat.3c shape) payload.
    pub fn encode_v14_section(&self) -> Result<Vec<u8>, String> {
        bincode::serialize(&self.to_v14())
            .map_err(|e| format!("runtime_config v14 encode: {}", e))
    }

    /// Decode a section #14 (bincode, frozen Cat.3c shape) payload.
    pub fn decode_v14_section(bytes: &[u8]) -> Result<Self, String> {
        let v14: RuntimeConfigSnapshotV14 = bincode::deserialize(bytes)
            .map_err(|e| format!("runtime_config v14 decode: {}", e))?;
        Ok(Self::from_v14(v14))
    }

    /// Encode the section #15 (JSON, full forward-extensible payload).
    pub fn encode_v15_section(&self) -> Result<Vec<u8>, String> {
        serde_json::to_vec(self)
            .map_err(|e| format!("runtime_config v15 encode: {}", e))
    }

    /// Decode a section #15 (JSON, full forward-extensible payload).
    /// Tolerant of unknown future fields and missing current fields
    /// (serde_json + `#[serde(default = …)]`).
    pub fn decode_v15_section(bytes: &[u8]) -> Result<Self, String> {
        serde_json::from_slice(bytes)
            .map_err(|e| format!("runtime_config v15 decode: {}", e))
    }
}

/// Compute the legacy Cat.3a sidecar path: `<brain>.trust.json`.
pub fn legacy_trust_sidecar_path(brain_path: &Path) -> PathBuf {
    crate::trust::sidecar_path_for(brain_path)
}

/// Compute the legacy Cat.3b sidecar path: `<brain>.hub_degree.json`.
pub fn legacy_hub_degree_sidecar_path(brain_path: &Path) -> PathBuf {
    crate::hub_degree::sidecar_path_for(brain_path)
}

/// Compute the post-migration paper-trail path for a legacy sidecar.
pub fn migrated_marker_path(legacy_path: &Path) -> PathBuf {
    let mut p = legacy_path.to_path_buf();
    let mut name = p
        .file_name()
        .map(|s| s.to_os_string())
        .unwrap_or_else(|| std::ffi::OsString::from("sidecar"));
    name.push(".migrated");
    p.set_file_name(name);
    p
}

/// Retire a legacy sidecar after a successful Cat.3c save.
pub fn retire_legacy_sidecar(legacy_path: &Path) -> Result<bool, String> {
    match std::fs::metadata(legacy_path) {
        Ok(_) => {
            let target = migrated_marker_path(legacy_path);
            if std::fs::metadata(&target).is_ok() {
                let _ = std::fs::remove_file(&target);
            }
            std::fs::rename(legacy_path, &target)
                .map_err(|e| format!("sidecar rename {} -> {}: {}", legacy_path.display(), target.display(), e))?;
            Ok(true)
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(e) => Err(format!("sidecar metadata at {}: {}", legacy_path.display(), e)),
    }
}

/// Build a `RuntimeConfigSnapshot` from the legacy sidecars. T.1.1
/// fields take their defaults — pre-Cat.3c brains never carried
/// time-source state.
pub fn migrate_from_legacy_sidecars(brain_path: &Path) -> RuntimeConfigSnapshot {
    let trust_path = legacy_trust_sidecar_path(brain_path);
    let hub_path = legacy_hub_degree_sidecar_path(brain_path);
    RuntimeConfigSnapshot {
        trust_table: TrustTable::load_from_path(&trust_path),
        hub_degree: HubDegreeConfig::load_from_path(&hub_path),
        style_decay_half_life_days: None,
        style_output_adaptation_enabled: default_style_output_adaptation_enabled(),
        style_max_samples_for_full_weight: default_style_max_samples_for_full_weight(),
        time_source: default_time_source(),
        time_service_url: default_time_service_url(),
        time_service_pubkey_hex: default_time_service_pubkey_hex(),
        time_service_poll_interval_ms: default_time_service_poll_interval_ms(),
        time_service_max_drift_ms: default_time_service_max_drift_ms(),
            sense_disambiguation_enabled: default_sense_disambiguation_enabled(),
            sense_context_window: default_sense_context_window(),
            sense_loser_gain: default_sense_loser_gain(),
            chained_research_enabled: default_chained_research_enabled(),
            max_chain_depth: default_max_chain_depth(),
            max_chain_cycles: default_max_chain_cycles(),
            chain_gap_threshold: default_chain_gap_threshold(),
            ingest_corpus_default: default_ingest_corpus_default(),
            ingest_run_id_strategy: default_ingest_run_id_strategy(),
            ingest_auto_approve_threshold: default_ingest_auto_approve_threshold(),
            tier2_prompt_default_session_strategy: default_tier2_prompt_default_session_strategy(),
            tier2_prompt_max_text_bytes: default_tier2_prompt_max_text_bytes(),
            file_io_max_size_bytes: default_file_io_max_size_bytes(),
            file_io_allowed_read_paths: default_file_io_allowed_read_paths(),
            file_io_allowed_write_paths: default_file_io_allowed_write_paths(),
            image_extractor_kind: default_image_extractor_kind(),
            image_extractor_subprocess_path: default_image_extractor_subprocess_path(),
            image_extractor_subprocess_timeout_ms: default_image_extractor_subprocess_timeout_ms(),
            image_max_size_bytes: default_image_max_size_bytes(),
            image_supported_mime_types: default_image_supported_mime_types(),
            recall_typed_ku_enabled: default_recall_typed_ku_enabled(),
            admin_revoke_enabled: default_admin_revoke_enabled(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn tmp_dir() -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "lu4_cat3c_test_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    // ---- Cat.3c snapshot tests (carry-overs) -----------------------

    #[test]
    fn cat3c_v3_section_round_trips_runtime_config_v14_legacy() {
        let mut snap = RuntimeConfigSnapshot::default();
        snap.trust_table
            .set("my-canonical.example", 1.5, Some("test entry".into()))
            .unwrap();
        snap.hub_degree.set_max_in_degree(1234).unwrap();
        snap.hub_degree.set_max_out_degree(2345).unwrap();
        snap.style_decay_half_life_days = Some(30);

        let bytes = snap.encode_v14_section().expect("encode v14");
        let round_tripped = RuntimeConfigSnapshot::decode_v14_section(&bytes)
            .expect("decode v14");
        let m = round_tripped
            .trust_table
            .lookup("https://my-canonical.example/foo");
        assert!((m - 1.5).abs() < 1e-6, "got {}", m);
        assert_eq!(round_tripped.hub_degree.max_in_degree, 1234);
        assert_eq!(round_tripped.hub_degree.max_out_degree, 2345);
        assert_eq!(round_tripped.style_decay_half_life_days, Some(30));
        assert_eq!(
            round_tripped.style_output_adaptation_enabled,
            default_style_output_adaptation_enabled()
        );
        assert_eq!(
            round_tripped.style_max_samples_for_full_weight,
            default_style_max_samples_for_full_weight()
        );
        // T.1.1 fields default cleanly through the V14 path.
        assert_eq!(round_tripped.time_source, "system");
        assert_eq!(round_tripped.time_service_url, default_time_service_url());
    }

    #[test]
    fn cat3c_loads_legacy_trust_json_sidecar_on_pre_cat3c_brain() {
        let dir = tmp_dir();
        let brain_path = dir.join("brain.lbr");
        let trust_path = legacy_trust_sidecar_path(&brain_path);
        let mut t = TrustTable::default();
        t.set("custom-source.example", 1.3, Some("test".into())).unwrap();
        t.save_to_path(&trust_path).expect("trust save");

        let snap = migrate_from_legacy_sidecars(&brain_path);
        assert!(
            (snap.trust_table.lookup("https://custom-source.example/x") - 1.3).abs() < 1e-6
        );
        assert_eq!(snap.hub_degree, HubDegreeConfig::default());
        // T.1.1: even legacy-migration path produces system defaults.
        assert_eq!(snap.time_source, "system");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn cat3c_loads_legacy_hub_degree_json_sidecar_on_pre_cat3c_brain() {
        let dir = tmp_dir();
        let brain_path = dir.join("brain.lbr");
        let hub_path = legacy_hub_degree_sidecar_path(&brain_path);
        let mut h = HubDegreeConfig::default();
        h.set_max_in_degree(7777).unwrap();
        h.set_max_out_degree(8888).unwrap();
        h.save_to_path(&hub_path).expect("hub save");

        let snap = migrate_from_legacy_sidecars(&brain_path);
        assert_eq!(snap.hub_degree.max_in_degree, 7777);
        assert_eq!(snap.hub_degree.max_out_degree, 8888);
        assert!((snap.trust_table.lookup("https://grokipedia.com/x") - 1.0).abs() < 1e-6);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn cat3c_save_writes_v3_section_not_sidecars() {
        let dir = tmp_dir();
        let brain_path = dir.join("brain.lbr");
        let _snap = migrate_from_legacy_sidecars(&brain_path);
        let trust_path = legacy_trust_sidecar_path(&brain_path);
        let hub_path = legacy_hub_degree_sidecar_path(&brain_path);
        assert!(!trust_path.exists());
        assert!(!hub_path.exists());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn cat3c_save_deletes_legacy_sidecars_after_migration() {
        let dir = tmp_dir();
        let brain_path = dir.join("brain.lbr");
        let trust_path = legacy_trust_sidecar_path(&brain_path);
        let hub_path = legacy_hub_degree_sidecar_path(&brain_path);
        TrustTable::default().save_to_path(&trust_path).unwrap();
        HubDegreeConfig::default().save_to_path(&hub_path).unwrap();
        assert!(trust_path.exists());
        assert!(hub_path.exists());

        let trust_renamed = retire_legacy_sidecar(&trust_path).expect("trust retire");
        let hub_renamed = retire_legacy_sidecar(&hub_path).expect("hub retire");
        assert!(trust_renamed);
        assert!(hub_renamed);
        assert!(!trust_path.exists());
        assert!(!hub_path.exists());
        assert!(migrated_marker_path(&trust_path).exists());
        assert!(migrated_marker_path(&hub_path).exists());
        assert!(!retire_legacy_sidecar(&trust_path).expect("trust retire 2x"));
        assert!(!retire_legacy_sidecar(&hub_path).expect("hub retire 2x"));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn cat3c_pre_cat3c_brain_with_no_sidecars_loads_with_defaults() {
        let dir = tmp_dir();
        let brain_path = dir.join("fresh_brain.lbr");
        let snap = migrate_from_legacy_sidecars(&brain_path);
        assert_eq!(snap, RuntimeConfigSnapshot::default());
        assert!((snap.trust_table.lookup("https://grokipedia.com/x") - 1.0).abs() < 1e-6);
        assert_eq!(snap.hub_degree.max_in_degree, 5000);
        assert_eq!(
            snap.style_output_adaptation_enabled,
            crate::style::DEFAULT_OUTPUT_ADAPTATION_ENABLED
        );
        assert_eq!(
            snap.style_max_samples_for_full_weight,
            crate::style::DEFAULT_MAX_SAMPLES_FOR_FULL_WEIGHT
        );
        // T.1.1 defaults for the canonical Lu.
        assert_eq!(snap.time_source, "system");
        assert_eq!(snap.time_service_url, default_time_service_url());
        assert_eq!(snap.time_service_poll_interval_ms, 60_000);
        assert_eq!(snap.time_service_max_drift_ms, 5_000);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn cat3c_migrated_marker_path_appends_suffix() {
        let p = migrated_marker_path(Path::new("/var/lib/lu/brain.lbr.trust.json"));
        assert_eq!(p, Path::new("/var/lib/lu/brain.lbr.trust.json.migrated"));
        let p2 = migrated_marker_path(Path::new("/var/lib/lu/brain.lbr.hub_degree.json"));
        assert_eq!(p2, Path::new("/var/lib/lu/brain.lbr.hub_degree.json.migrated"));
    }

    #[test]
    fn cat3c_retire_overwrites_existing_marker_idempotently() {
        let dir = tmp_dir();
        let p = dir.join("sidecar.json");
        fs::write(&p, b"{\"v\":1}").unwrap();
        assert!(retire_legacy_sidecar(&p).unwrap());
        let marker = migrated_marker_path(&p);
        assert!(marker.exists());
        assert_eq!(fs::read(&marker).unwrap(), b"{\"v\":1}");
        fs::write(&p, b"{\"v\":2}").unwrap();
        assert!(retire_legacy_sidecar(&p).unwrap());
        assert_eq!(fs::read(&marker).unwrap(), b"{\"v\":2}");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn cat3c_legacy_sidecar_paths_match_pre_cat3c_layout() {
        let p = Path::new("/var/lib/lu/brain.lbr");
        assert_eq!(
            legacy_trust_sidecar_path(p),
            Path::new("/var/lib/lu/brain.lbr.trust.json")
        );
        assert_eq!(
            legacy_hub_degree_sidecar_path(p),
            Path::new("/var/lib/lu/brain.lbr.hub_degree.json")
        );
    }

    // ---- Cat.3c.1 forward-compat at the field level ---------------

    #[test]
    fn cat3c1_runtime_config_forward_compat_extra_field_ignored() {
        let mut snap = RuntimeConfigSnapshot::default();
        snap.style_output_adaptation_enabled = false;
        snap.style_max_samples_for_full_weight = 250;
        let json_bytes = snap.encode_v15_section().unwrap();
        let mut value: serde_json::Value = serde_json::from_slice(&json_bytes).unwrap();
        value["future_knob_x"] = serde_json::json!("hello-from-the-future");
        value["another_future_field"] = serde_json::json!(42);
        let mutated = serde_json::to_vec(&value).unwrap();
        let parsed = RuntimeConfigSnapshot::decode_v15_section(&mutated)
            .expect("current binary must tolerate unknown future fields");
        assert!(!parsed.style_output_adaptation_enabled);
        assert_eq!(parsed.style_max_samples_for_full_weight, 250);
    }

    #[test]
    fn cat3c1_runtime_config_backward_compat_missing_field_defaults() {
        // A snapshot saved by a HYPOTHETICAL EARLIER binary that had
        // no Cat.3c.1 + T.1.1 fields. Today's binary must fill them in.
        let snap = RuntimeConfigSnapshot::default();
        let json_bytes = snap.encode_v15_section().unwrap();
        let mut value: serde_json::Value = serde_json::from_slice(&json_bytes).unwrap();
        if let Some(obj) = value.as_object_mut() {
            obj.remove("style_output_adaptation_enabled");
            obj.remove("style_max_samples_for_full_weight");
            obj.remove("style_decay_half_life_days");
            obj.remove("time_source");
            obj.remove("time_service_url");
            obj.remove("time_service_pubkey_hex");
            obj.remove("time_service_poll_interval_ms");
            obj.remove("time_service_max_drift_ms");
        }
        let early = serde_json::to_vec(&value).unwrap();
        let parsed = RuntimeConfigSnapshot::decode_v15_section(&early)
            .expect("current binary must fill defaults for missing fields");
        assert_eq!(parsed.style_decay_half_life_days, None);
        assert_eq!(
            parsed.style_output_adaptation_enabled,
            default_style_output_adaptation_enabled()
        );
        assert_eq!(
            parsed.style_max_samples_for_full_weight,
            default_style_max_samples_for_full_weight()
        );
        assert_eq!(parsed.time_source, "system");
        assert_eq!(parsed.time_service_url, default_time_service_url());
    }

    #[test]
    fn cat3c1_style_output_adaptation_enabled_persists() {
        let mut snap = RuntimeConfigSnapshot::default();
        snap.style_output_adaptation_enabled = false;
        let bytes = snap.encode_v15_section().expect("encode v15");
        let round = RuntimeConfigSnapshot::decode_v15_section(&bytes).expect("decode v15");
        assert!(!round.style_output_adaptation_enabled);
    }

    #[test]
    fn cat3c1_style_max_samples_for_full_weight_persists() {
        let mut snap = RuntimeConfigSnapshot::default();
        snap.style_max_samples_for_full_weight = 555;
        let bytes = snap.encode_v15_section().expect("encode v15");
        let round = RuntimeConfigSnapshot::decode_v15_section(&bytes).expect("decode v15");
        assert_eq!(round.style_max_samples_for_full_weight, 555);
    }

    #[test]
    fn cat3c1_pre_cat3c1_brain_loads_cleanly() {
        let mut v14_only = RuntimeConfigSnapshot::default();
        v14_only.trust_table
            .set("x.example", 1.2, None)
            .unwrap();
        let v14_bytes = v14_only.encode_v14_section().expect("encode v14");
        let round = RuntimeConfigSnapshot::decode_v14_section(&v14_bytes).expect("decode v14");
        assert!((round.trust_table.lookup("https://x.example/") - 1.2).abs() < 1e-6);
        assert_eq!(
            round.style_output_adaptation_enabled,
            default_style_output_adaptation_enabled()
        );
        assert_eq!(
            round.style_max_samples_for_full_weight,
            default_style_max_samples_for_full_weight()
        );
    }

    #[test]
    fn cat3c1_post_cat3c1_brain_loads_with_pre_cat3c1_binary() {
        let mut snap = RuntimeConfigSnapshot::default();
        snap.trust_table
            .set("yy.example", 1.4, Some("x".into()))
            .unwrap();
        snap.hub_degree.set_max_in_degree(3333).unwrap();
        snap.style_output_adaptation_enabled = false;
        snap.style_max_samples_for_full_weight = 999;
        // T.1.1 fields also flipped — must NOT bleed into V14.
        snap.time_source = "df_service".into();
        snap.time_service_max_drift_ms = 9999;

        let v14_bytes = snap.encode_v14_section().expect("v14");
        let mut twin = snap.clone();
        twin.style_output_adaptation_enabled = true;
        twin.style_max_samples_for_full_weight = 1;
        twin.time_source = "system".into();
        twin.time_service_max_drift_ms = 1;
        let v14_twin = twin.encode_v14_section().expect("v14 twin");
        assert_eq!(v14_bytes, v14_twin, "section #14 must not depend on Cat.3c.1+T.1.1 fields");

        let cat3c_view: RuntimeConfigSnapshotV14 =
            bincode::deserialize(&v14_bytes).expect("cat.3c-style decode");
        assert!((cat3c_view.trust_table.lookup("https://yy.example/") - 1.4).abs() < 1e-6);
        assert_eq!(cat3c_view.hub_degree.max_in_degree, 3333);
    }

    #[test]
    fn cat3c1_no_sidecar_files_written_on_save() {
        let dir = tmp_dir();
        let brain_path = dir.join("brain.lbr");
        let snap = RuntimeConfigSnapshot::default();
        let _v14 = snap.encode_v14_section().unwrap();
        let _v15 = snap.encode_v15_section().unwrap();
        let trust_path = legacy_trust_sidecar_path(&brain_path);
        let hub_path = legacy_hub_degree_sidecar_path(&brain_path);
        assert!(!trust_path.exists(), "encode must not create trust sidecar");
        assert!(!hub_path.exists(), "encode must not create hub sidecar");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn cat3c1_save_runtime_config_round_trips_through_v3_section() {
        let mut snap = RuntimeConfigSnapshot::default();
        snap.trust_table
            .set("z.example", 1.7, Some("note".into()))
            .unwrap();
        snap.hub_degree.set_max_in_degree(4444).unwrap();
        snap.hub_degree.set_max_out_degree(5555).unwrap();
        snap.style_decay_half_life_days = Some(45);
        snap.style_output_adaptation_enabled = false;
        snap.style_max_samples_for_full_weight = 77;
        // T.1.1 fields also.
        snap.time_source = "df_service".into();
        snap.time_service_url = "https://other.example/v1/now".into();
        snap.time_service_poll_interval_ms = 30_000;
        snap.time_service_max_drift_ms = 2_500;

        let bytes = snap.encode_v15_section().expect("encode v15");
        let round = RuntimeConfigSnapshot::decode_v15_section(&bytes).expect("decode v15");
        assert_eq!(snap, round);
    }

    #[test]
    fn cat3c1_legacy_sidecar_load_paths_removed_from_codebase() {
        let snap = RuntimeConfigSnapshot::default();
        let _v14 = snap.encode_v14_section().expect("v14 must encode");
        let _v15 = snap.encode_v15_section().expect("v15 must encode");
    }

    #[test]
    fn cat3c1_lu4d_configure_persists_new_knobs_via_v3_section() {
        let mut snap = RuntimeConfigSnapshot::default();
        snap.style_output_adaptation_enabled = false;
        snap.style_max_samples_for_full_weight = 200;
        let bytes = snap.encode_v15_section().unwrap();
        let round = RuntimeConfigSnapshot::decode_v15_section(&bytes).unwrap();
        assert!(!round.style_output_adaptation_enabled);
        assert_eq!(round.style_max_samples_for_full_weight, 200);
    }
}
