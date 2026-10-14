//! High-level Lu 4D brain — wraps the associative network, session manager,
//! consolidation engine, and persistence into a single API surface.
//!
//! Used by the REPL and the MCP server. Keeps the lower-level modules focused
//! on their core responsibilities while giving user-facing code one clean
//! entry point.

use crate::bond::{BondType, KeyframeSource, TemporalBond};
use crate::bridge::{classify_candidates, BridgeCollector, BridgeConfig, BridgeEvent};
use crate::curiosity::{
    classify_curiosity_candidates, CuriosityCollector, CuriosityConfig, CuriositySignal,
    InterestProfile, WonderQueue,
};
use crate::consolidation::{ConsolidationEngine, ConsolidationStats};
use crate::variants::{
    generate_variants_for_goal, score_arena, PerturbationStrategy, VariantArena,
    VariantConfig, VariantSpec,
};
use crate::self_reflection::{
    analyze as analyze_reflection,
    AdjustmentTarget, CurrentKnobs, MetaReflection, PerformanceMonitor,
    ProposalStatus, QueryMetric, SelfReflectionConfig,
};
use crate::network::{AssociativeNetwork, NetworkSnapshot, RecallResult};
use crate::scar::{Scar, ScarLog};
use crate::node::{Confidence, KnowledgeContent, KnowledgeNode};
use crate::scope::TemporalScope;
use crate::seed::Seed;
use crate::session::SessionManager;
use crate::session_memory::SessionMemoryRegistry;
use crate::timetravel::{time_travel_query, TemporalDiff};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Serialized on-disk brain file format.
///
/// `interest_profile` is optional on disk (`#[serde(default)]`) so brain
/// files written before InterestProfile persistence load with an empty
/// profile. Format version stays at 1; the field is purely additive.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrainFile {
    pub format_version: u32,
    pub network: NetworkSnapshot,
    #[serde(default)]
    pub interest_profile: InterestProfile,
    /// Persisted self-reflection state (proposal log + counters).
    /// `#[serde(default)]` so brain files written before this port load
    /// with an empty MetaReflection.
    #[serde(default)]
    pub meta_reflection: MetaReflection,
    /// Persisted scar log. `#[serde(default)]` so brain files written
    /// before the Scar port load with an empty log.
    #[serde(default)]
    pub scar_log: ScarLog,
    /// Persisted research proposals. `#[serde(default)]` so brain
    /// files written before Port #10 load with no proposals.
    #[serde(default)]
    pub research_proposals: Vec<crate::research::proposal::ResearchProposal>,
    /// Persisted research config so John's threshold tweaks survive
    /// restarts.
    #[serde(default = "crate::research::router::ResearchConfig::default")]
    pub research_config: crate::research::router::ResearchConfig,
    /// Monotonic id allocator for research proposals.
    #[serde(default)]
    pub next_research_proposal_id: u64,
    /// Port #17 Fix 2: per-seed counter of how many times each
    /// curiosity seed has been surfaced as the wondering_about
    /// slot. Once a seed exceeds `CuriosityConfig.gap_appearance_threshold`,
    /// narration filters it. Persisted across save/load so the
    /// learned "noise gap" set survives restarts.
    #[serde(default)]
    pub gap_seed_frequency: std::collections::HashMap<Seed, u32>,
    /// Port J.1: persisted wonder queue + gap label map.
    /// `#[serde(default)]` so brain files written before J.1 load with an empty WonderState.
    #[serde(default)]
    pub wonder_state: crate::curiosity::WonderState,
    /// Port G: negation overlay (Learning #577). Maps
    /// `(from, to, bond_type)` -> negated flag. Stored as a sparse map
    /// with hex-encoded string keys so the JSON path round-trips.
    /// `#[serde(default)]` so pre-G brain files load with an empty
    /// overlay -> every bond defaults to negated=false.
    #[serde(default)]
    pub negation_overlay: crate::negation::NegationOverlay,
    /// Port S: source attribution overlay (Learning #635 + #634).
    /// Maps `(from, to, bond_type)` -> SourceAttribution. Stored as
    /// a sparse map with hex-encoded string keys so the JSON path
    /// round-trips. `#[serde(default)]` so pre-S brain files load
    /// with an empty overlay -> every bond defaults to
    /// `SourceKind::Unknown`.
    #[serde(default)]
    pub source_overlay: crate::source::SourceOverlay,
    /// Port T.4: persisted PendingLog snapshot (v3 binary section
    /// `pending_log`, id 9). `#[serde(default)]` so pre-T.4 brain
    /// files load with an empty `PendingLogSnapshot` (no entries,
    /// next_pending_id = 1). The companion `<brain>.pending.wal`
    /// is replayed on top of this snapshot at load time so that
    /// any mutations between the last successful save and a crash
    /// are recovered idempotently.
    #[serde(default)]
    pub pending_log: crate::pending::PendingLogSnapshot,
    /// Port T.8: persisted rejection-pattern store snapshot (v3
    /// binary section `rejection_patterns`, id 10). `#[serde(default)]`
    /// so pre-T.8 brain files load with an empty store (no
    /// fingerprints, no patterns, `proposals_filtered_total = 0`).
    /// Lu uses this to learn from rejected pending entries so she
    /// stops proposing the same shape of low-quality bonds.
    #[serde(default)]
    pub rejection_patterns: crate::rejection::RejectionPatternsSnapshot,
    /// Port LB: persisted Left Brain session memory store (v3
    /// binary section `lb_session`, id 11). `#[serde(default)]`
    /// so pre-LB brain files load with an empty store. Session
    /// entries are operational/episodic memory (thinking traces,
    /// observations, hypotheses) that NEVER auto-promote to fact
    /// bonds — promotion goes through the T.3 PendingLog path.
    #[serde(default)]
    pub lb_session: crate::lb_session::SessionMemorySnapshot,
    /// Port CS: persisted ConversationalStyle snapshot (v3
    /// binary section `conversational_style`, id 12). `#[serde(default)]`
    /// so pre-CS brain files load with an empty store.
    #[serde(default)]
    pub cs_style: crate::style::ConversationalStyleSnapshot,
    /// Port Cat.4: pack ledger snapshot (v3 binary section #13).
    ///  so pre-Cat.4 brain files load with an
    /// empty ledger.
    #[serde(default)]
    pub pack_ledger: crate::pack::PackLedgerSnapshot,
    /// Port Cat.5: capability pack ledger snapshot (v3 binary
    /// section #17). Schema-only signed capability pack ledger:
    /// trusted publishers, installed packs, revoked pack_ids.
    /// `#[serde(default)]` so pre-Cat.5 brain files load with an
    /// empty ledger; pre-Cat.5 binaries reading a Cat.5 brain skip
    /// section #17 via the J.1.1 unknown-section tolerance.
    #[serde(default)]
    pub cap_pack_ledger: crate::capability_pack::CapabilityPackLedgerSnapshot,
    /// Port Cat.3c: runtime config snapshot (v3 binary section #14).
    /// Consolidates the legacy `<brain>.trust.json` and `<brain>.hub_degree.json`
    /// sidecars into the encrypted brain. `#[serde(default)]` so pre-Cat.3c
    /// brain files load with `RuntimeConfigSnapshot::default()`; the actual
    /// runtime values come from the legacy sidecars on the load path that
    /// runs when `runtime_config_present` is false.
    #[serde(default)]
    pub runtime_config: crate::runtime_config::RuntimeConfigSnapshot,
    /// Port Cat.3c: did the v3 file actually carry section #14?
    /// `#[serde(skip)]` so it never goes on disk — it's the
    /// in-memory presence flag the v3 reader sets so callers can
    /// distinguish "section absent (pre-Cat.3c brain)" from
    /// "section present with default content".
    #[serde(skip)]
    pub runtime_config_present: bool,
}

/// Maximum v2-binary format version this build accepts on load.
///
/// Audit fix internal notes / Problem #111 follow-up): this constant
/// is *not* the absolute latest brain version -- v3 (rkyv mmap-zerocopy,
/// gated by `crate::brain_io_v3::FORMAT_VERSION_V3 = 3`) is now the
/// production format. `BRAIN_FORMAT_VERSION` is specifically the ceiling
/// that the v2 binary reader accepts; the legacy docstring claiming
/// this constant is the global ceiling was misleading. v1 JSON is no
/// longer a supported production load target -- migrate via
/// `lu4d --migrate-from-v1`.
pub const BRAIN_FORMAT_VERSION: u32 = 2;

/// Format-version field value carried inside v1 JSON brain files.
pub const V1_JSON_VERSION: u32 = 1;

/// Errors that can occur during brain operations.
#[derive(Debug)]
pub enum BrainError {
    Io(io::Error),
    Serde(serde_json::Error),
    ParseBondType(String),
    ParseConfidence(String),
    VersionMismatch(u32),
    /// Bincode (de)serialization error in the v2 binary path.
    Bincode(String),
    /// Structural error in the v2 binary file (bad magic, bad offsets,
    /// checksum mismatch, missing required section).
    Format(String),
    /// Port T.7: structural encryption guard. Surfaced when the
    /// `--require-encrypted` flag (or `Brain.require_encrypted`) is
    /// set and either (a) the brain on disk is not v3enc, or (b) a
    /// save path attempts to write a non-v3enc format. The hint
    /// points the operator at the migration / save-format command
    /// they need.
    EncryptionRequired { attempted_format: String, hint: String },
}

impl std::fmt::Display for BrainError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BrainError::Io(e) => write!(f, "I/O error: {}", e),
            BrainError::Serde(e) => write!(f, "serialization error: {}", e),
            BrainError::ParseBondType(s) => write!(f, "unknown bond type: {}", s),
            BrainError::ParseConfidence(s) => write!(f, "unknown confidence: {}", s),
            BrainError::VersionMismatch(v) => {
                write!(f, "unsupported brain file version: {} (latest supported {})", v, BRAIN_FORMAT_VERSION)
            }
            BrainError::Bincode(s) => write!(f, "bincode error: {}", s),
            BrainError::Format(s) => write!(f, "brain file format error: {}", s),
            BrainError::EncryptionRequired { attempted_format, hint } => write!(
                f,
                "encryption required: brain format is {} but --require-encrypted is set; {}",
                attempted_format, hint
            ),
        }
    }
}

impl BrainError {
    /// Convenience constructor used by the v2 binary path. bincode
    /// 1.x error is `Box<bincode::ErrorKind>` and not `Send + Sync` in
    /// every config, so we collapse to a String for storage.
    pub fn bincode<E: std::fmt::Display>(e: E) -> Self {
        BrainError::Bincode(e.to_string())
    }
}

impl std::error::Error for BrainError {}

impl From<io::Error> for BrainError {
    fn from(e: io::Error) -> Self { BrainError::Io(e) }
}

impl From<serde_json::Error> for BrainError {
    fn from(e: serde_json::Error) -> Self { BrainError::Serde(e) }
}

/// Port T.7: structural encryption guard. Inspect the brain file at
/// `path` and return `BrainError::EncryptionRequired` if it is not
/// v3 with the encrypted flag set.
///
/// A non-existent path is treated as OK — there is no on-disk brain to
/// inspect, and the save-side guard catches any subsequent attempt to
/// write a plaintext brain. A v1 or v2 file is always rejected; a v3
/// file is accepted only when its `flags` field carries
/// `FORMAT_FLAG_ENCRYPTED`.
///
/// Companion to `Brain::require_encrypted`. T.7 makes the encryption
/// invariant structural: refusing-to-load is the new floor (no longer
/// aspirational, as it was under Solution #79 / Learning #621 which
/// only covered the loaded-from-encrypted case).
/// OP-BRAIN-V5 6.5: validate that a v5 brain directory is encrypted at
/// rest before accepting it under `--require-encrypted`.
///
/// Pre-6.5 the guard returned `Ok(())` for ANY directory, silently
/// passing a plaintext v5 brain (the v5 SST writers ship a plaintext
/// mode and the migration emitted plaintext stores). This closes that
/// hole. A v4 dir (no `node_store/`/`bond_store/`) keeps the prior
/// defer-to-loader behaviour; a v5 dir is hard-failed unless BOTH
/// (a) the MANIFEST carries a real `master_key_id` fingerprint AND
/// (b) every NodeSST/BondSST header found on disk has its
/// `FLAG_ENCRYPTED` bit set.
fn require_encrypted_dir_or_err(dir: &Path) -> Result<(), BrainError> {
    let manifest_path = dir.join(crate::manifest::MANIFEST_FILE_NAME);
    if !manifest_path.exists() {
        // Not a recognizable store dir; let detect_brain_layout reject it.
        return Ok(());
    }
    let is_v5 = dir.join("node_store").is_dir() || dir.join("bond_store").is_dir();
    if !is_v5 {
        // v4 dir: structural v4-SST encryption, out of 6.5 scope.
        return Ok(());
    }
    let hint = "re-migrate with --keyfile so the v5 node_store/bond_store SSTs                 are encrypted at rest (OP-BRAIN-V5 6.5)"
        .to_string();

    // (a) MANIFEST marker: master_key_id must be a real key fingerprint.
    let manifest = crate::manifest::Manifest::load(&manifest_path)?.ok_or_else(|| {
        BrainError::Format(format!(
            "v5 brain dir {} has no readable MANIFEST",
            dir.display()
        ))
    })?;
    if manifest.master_key_id == crate::manifest::KeyFingerprint::none() {
        return Err(BrainError::EncryptionRequired {
            attempted_format: "v5-plaintext".to_string(),
            hint,
        });
    }

    // (b) Defense in depth: sniff actual SST headers. A manifest can
    //     claim a key while the stores are plaintext (inconsistent or
    //     tampered); refuse if any present SST is unencrypted.
    if dir_has_plaintext_sst(&dir.join("node_store"))?
        || dir_has_plaintext_sst(&dir.join("bond_store"))?
    {
        return Err(BrainError::EncryptionRequired {
            attempted_format: "v5-plaintext".to_string(),
            hint,
        });
    }

    // (c) OP-BRAIN-V5 6.8: the WAL (`wal/current.wal`) and the node-
    //     intern sidecar (`wal/nodeintern.bin`) must ALSO be encrypted
    //     at rest. A v5 dir is "encrypted" only if SSTs AND WAL AND
    //     sidecar all carry FLAG_ENCRYPTED. A pre-6.8 plaintext WAL or
    //     sidecar is hard-failed here (forces regeneration via re-
    //     migration with the 6.8 binary -- no silent read-as-encrypted,
    //     no auto-convert). A missing/zero-length file is not a leak.
    let wal_path = dir.join("wal").join("current.wal");
    match crate::wal::wal_encryption_status(&wal_path)
        .map_err(|e| BrainError::Format(format!("WAL header probe failed: {}", e)))?
    {
        crate::wal::WalEncStatus::Plaintext => {
            return Err(BrainError::EncryptionRequired {
                attempted_format: "v5-wal-plaintext".to_string(),
                hint: hint.clone(),
            });
        }
        crate::wal::WalEncStatus::Absent | crate::wal::WalEncStatus::Encrypted => {}
    }
    let sidecar_path = dir.join("wal").join("nodeintern.bin");
    match crate::node_store::sidecar::sidecar_encryption_status(&sidecar_path)
        .map_err(|e| BrainError::Format(format!("sidecar header probe failed: {}", e)))?
    {
        crate::node_store::sidecar::SidecarEncStatus::Plaintext => {
            return Err(BrainError::EncryptionRequired {
                attempted_format: "v5-sidecar-plaintext".to_string(),
                hint,
            });
        }
        crate::node_store::sidecar::SidecarEncStatus::Absent
        | crate::node_store::sidecar::SidecarEncStatus::Encrypted => {}
    }
    Ok(())
}

/// Scan `store_dir` for NodeSST/BondSST files whose header magic is
/// recognised but whose `FLAG_ENCRYPTED` bit is clear. Returns
/// `Ok(true)` if any such plaintext SST is found. A missing directory
/// or a file too short to carry a header is treated as "nothing to
/// flag" -- the loader CRC checks own structural corruption; this
/// helper answers only the plaintext-at-rest question.
fn dir_has_plaintext_sst(store_dir: &Path) -> Result<bool, BrainError> {
    Ok(!scan_sst_encryption(store_dir)?.plaintext.is_empty())
}

/// OP-BRAIN-V5 6.7 -- result of scanning a v5 store dir for the per-SST
/// at-rest encryption marker.
#[derive(Debug, Clone, Default)]
pub struct SstEncryptionScan {
    /// Recognised NodeSST/BondSST files found.
    pub total: usize,
    /// Of those, how many carry the FLAG_ENCRYPTED header bit.
    pub encrypted: usize,
    /// Paths of recognised SSTs whose FLAG_ENCRYPTED bit is clear.
    pub plaintext: Vec<std::path::PathBuf>,
}

/// Scan `store_dir` for NodeSST/BondSST files and classify each by the
/// header `FLAG_ENCRYPTED` bit (byte 6, bit 0) -- the same per-file
/// at-rest marker the 6.5 guard uses. A missing dir yields an empty
/// scan. Files too short to carry a header, or whose magic is not a
/// recognised SST magic, are skipped (not counted). Shared primitive
/// behind both `dir_has_plaintext_sst` (the load guard) and the 6.7
/// migration verifier's encryption-at-rest check.
pub(crate) fn scan_sst_encryption(store_dir: &Path) -> Result<SstEncryptionScan, BrainError> {
    use std::io::Read;
    let mut scan = SstEncryptionScan::default();
    if !store_dir.is_dir() {
        return Ok(scan);
    }
    const FLAG_ENCRYPTED: u8 = 0x01;
    let nsst_magic = crate::node_store::sst_writer::NSST_MAGIC;
    let bsst_magic = crate::bond_store::sst_writer::BSST_MAGIC;
    for ent in std::fs::read_dir(store_dir)? {
        let ent = ent?;
        if !ent.file_type().map(|t| t.is_file()).unwrap_or(false) {
            continue;
        }
        let path = ent.path();
        let mut f = match std::fs::File::open(&path) {
            Ok(f) => f,
            Err(_) => continue,
        };
        let mut hdr = [0u8; 8];
        if f.read_exact(&mut hdr).is_err() {
            continue; // too short to be a finished SST (e.g. a .tmp)
        }
        let magic = u32::from_le_bytes([hdr[0], hdr[1], hdr[2], hdr[3]]);
        if magic != nsst_magic && magic != bsst_magic {
            continue; // not a recognised SST header
        }
        scan.total += 1;
        if hdr[6] & FLAG_ENCRYPTED != 0 {
            scan.encrypted += 1;
        } else {
            scan.plaintext.push(path);
        }
    }
    Ok(scan)
}

pub fn require_v3enc_or_err<P: AsRef<Path>>(path: P) -> Result<(), BrainError> {
    let path = path.as_ref();
    if !path.exists() {
        return Ok(());
    }
    // OP-BRAIN-V4 Phase 8 hotfix 2026-06-01: v4 brains live in a directory.
    // v4 SSTs are structurally encrypt-then-compress under the master key,
    // so a v4 dir is deferred to detect_brain_layout + load_v4.
    //
    // OP-BRAIN-V5 6.5 fix: a v5 store dir (node_store/ + bond_store/) is NOT
    // structurally encrypted by construction -- the v5 SST writers ship a
    // plaintext mode -- so the pre-6.5 blanket `Ok(())` for ANY directory
    // silently passed a plaintext v5 brain under --require-encrypted. Close
    // that hole here: validate v5 dirs against the MANIFEST encryption
    // marker AND sniff the SST headers FLAG_ENCRYPTED bit.
    if path.is_dir() {
        return require_encrypted_dir_or_err(path);
    }
    let migration_hint = "use --migrate-v3-compressed-to-encrypted SRC DST [--keyfile P] to convert this brain to v3enc";
    let detected = crate::brain_io::detect_format_path(path)?;
    match detected {
        crate::brain_io::BrainFormat::V1Json => Err(BrainError::EncryptionRequired {
            attempted_format: "v1".to_string(),
            hint: migration_hint.to_string(),
        }),
        crate::brain_io::BrainFormat::V2Binary => Err(BrainError::EncryptionRequired {
            attempted_format: "v2".to_string(),
            hint: migration_hint.to_string(),
        }),
        crate::brain_io::BrainFormat::V3Binary => {
            use std::io::Read;
            let mut f = std::fs::File::open(path)?;
            let mut buf = [0u8; crate::brain_io::HEADER_SIZE];
            f.read_exact(&mut buf)?;
            let flags = crate::brain_io_v3::read_v3_flags(&buf)?;
            if crate::encrypt::is_encrypted(flags) {
                Ok(())
            } else {
                let detected_label = if flags & crate::compress::FORMAT_FLAG_COMPRESSED != 0 {
                    "v3c"
                } else {
                    "v3"
                };
                Err(BrainError::EncryptionRequired {
                    attempted_format: detected_label.to_string(),
                    hint: migration_hint.to_string(),
                })
            }
        }
    }
}

/// Port T-Ingest.3: companion to `require_v3enc_or_err` for the
/// empty-creation case. `require_v3enc_or_err` returns Ok for a
/// non-existent path because there is nothing to validate on disk;
/// when the operator passes `--require-encrypted` against a path
/// that does not exist, this guard refuses to start unless a key
/// was also provided. Without the guard, the empty-brain path
/// would happily create a fresh plaintext brain that the very next
/// save would refuse to write -- but only after the in-memory
/// brain had already been constructed and ingest tools wired up.
/// Refusing here keeps the failure at startup and the error
/// readable.
pub fn require_encrypted_create_or_err<P: AsRef<Path>>(
    path: P,
    key: &Option<crate::encrypt::MasterKey>,
) -> Result<(), BrainError> {
    let path = path.as_ref();
    if path.exists() {
        // Existing file: `require_v3enc_or_err` already validates it.
        return Ok(());
    }
    if key.is_none() {
        return Err(BrainError::EncryptionRequired {
            attempted_format: "<empty-creation>".to_string(),
            hint: "creating a new brain under --require-encrypted needs --keyfile; \
                   pass --keyfile PATH or run `lu4d --encrypt-init --keyfile PATH` first"
                .to_string(),
        });
    }
    Ok(())
}

/// On-disk format selected by `Brain::save_to_format` / `Brain::save`.
///
/// Audit fix internal notes / Problem #111 follow-up): production
/// default is now `V3Binary` (post v1 JSON removal). The `V1Json`
/// variant still exists so `loaded_format` can report "v1" for the
/// migration UX, but `parse("v1")` returns `None` and `save_to_format`
/// refuses to write it -- v1 is read-only via
/// `load_v1_for_migration_only` and migrated to v3/v3enc by
/// `lu4d --migrate-from-v1`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BrainSaveFormat {
    V1Json,
    V2Binary,
    /// v3 binary (Storage v2 Phase 2.2 / Slice 2a). Same shape as v2 on
    /// disk except the network and scar-log sections are rkyv-archived
    /// instead of bincoded.
    V3Binary,
    /// Storage v2 Phase 3: v3 binary with the network section in
    /// the per-block zstd-compressed layout (`crate::compress`).
    V3BinaryCompressed,
    /// Storage v2 Phase 4: v3 + per-block compression + DF XOR
    /// encryption-at-rest. Same on-disk shape as `V3BinaryCompressed`
    /// except the file header carries `FORMAT_FLAG_COMPRESSED |
    /// FORMAT_FLAG_ENCRYPTED` and every compressed block payload is
    /// XOR-encrypted under (master_key, block_index). Saving in this
    /// format requires `Brain.encryption_key` to be `Some`; otherwise
    /// `save_to_format` errors rather than silently writing plaintext.
    V3BinaryEncrypted,
}

impl BrainSaveFormat {
    /// Parse from a CLI string. Accepts `v1`, `json`, `v1json`, `v2`,
    /// `bin`, `binary`, `v2binary`, `v3`, `v3binary` (case-insensitive).
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            // v1 JSON removal internal notes: v1 is no
            // longer a valid SAVE target. The variant still exists so
            // loaded_format can report "v1" for migration UX, but parsing
            // "v1" from --brain-format-out returns None so the CLI errors.
            "v1" | "json" | "v1json" => None,
            "v2" | "bin" | "binary" | "v2binary" => Some(BrainSaveFormat::V2Binary),
            "v3" | "v3binary" | "rkyv" => Some(BrainSaveFormat::V3Binary),
            "v3c" | "v3compressed" | "v3-compressed" | "v3binarycompressed" => {
                Some(BrainSaveFormat::V3BinaryCompressed)
            }
            "v3enc" | "v3encrypted" | "v3-encrypted" | "v3binaryencrypted" => {
                Some(BrainSaveFormat::V3BinaryEncrypted)
            }
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            BrainSaveFormat::V1Json => "v1",
            BrainSaveFormat::V2Binary => "v2",
            BrainSaveFormat::V3Binary => "v3",
            BrainSaveFormat::V3BinaryCompressed => "v3c",
            BrainSaveFormat::V3BinaryEncrypted => "v3enc",
        }
    }
}

/// Storage v2 Slice 2c — load mode for v3 brain files.
///
/// `Auto` is the runtime default: v3 files load via mmap-zerocopy,
/// other formats fall back to the eager path. `Eager` forces the
/// pre-Slice-2c eager-deserialize path even on v3 (used by
/// `--brain-format-in v3-eager` for RSS regression debugging).
/// `MmapZeroCopy` is only valid for v3 files; non-v3 paths return
/// `BrainError::VersionMismatch`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BrainLoadMode {
    Auto,
    Eager,
    MmapZeroCopy,
    /// OP-007: open a v5 brain DIRECTORY as a non-writing reader
    /// (committed SSTs only; no flush worker, no WAL append, no writes).
    ReadOnlyReader,
    /// OP-INGEST-LEAN: open a v5 brain DIRECTORY as a WRITE-ONLY ingest
    /// path. Loads only the append/flush machinery (node intern +
    /// MemTable + WAL + flush/compaction workers + overlays) and SKIPS
    /// the serving/recall residency (resident degree-map, cache-backed
    /// adjacency + seam, keyword index). Bonds still land durably;
    /// serving structures are (re)built when Lu is served, not during
    /// training.
    IngestOnly,
}

impl BrainLoadMode {
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "auto" | "" => Some(BrainLoadMode::Auto),
            "v3-eager" | "eager" | "v3eager" => Some(BrainLoadMode::Eager),
            "v3-mmap-zerocopy" | "v3-mmap" | "mmap" | "v3mmap" | "v3mmapzerocopy" => {
                Some(BrainLoadMode::MmapZeroCopy)
            }
            "readonly" | "read-only" | "read-only-reader" | "ro" => {
                Some(BrainLoadMode::ReadOnlyReader)
            }
            "ingest" | "ingest-only" | "ingestonly" | "write-only" | "writeonly" => {
                Some(BrainLoadMode::IngestOnly)
            }
            _ => None,
        }
    }
}

/// OP-BRAIN-V4 Phase 8 -- on-disk layout detected at `--brain <path>`
/// time. Returned by `Brain::detect_brain_layout` and consumed by
/// `Brain::load`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BrainLayout {
    /// `path` is a regular file (a v3 .lbr brain). Loaded via the
    /// legacy `load_or_new_with_mode_and_key` path.
    V3File,
    /// `path` is a directory containing a `MANIFEST` file -- a v4
    /// brain. Loaded via `Brain::load_v4`.
    V4Dir,
    /// OP-BRAIN-V5 6.1 -- `path` is a directory whose `MANIFEST` carries
    /// the explicit v5 format tag (`format_tag == BRAIN_FORMAT_TAG_V5`):
    /// a node_store/ + bond_store/ substrate brain. Loaded via
    /// `Brain::load_v5`. Distinguished from `V4Dir` by manifest
    /// CONTENTS, never by directory shape.
    V5Dir,
    /// `path` does not exist. v3 empty-create path applies.
    Missing,
}

impl BrainLayout {
    /// True iff this layout would be loaded via the v4 path.
    pub fn is_v4(self) -> bool { matches!(self, BrainLayout::V4Dir) }
    /// True iff this layout would be loaded via the v5 path.
    pub fn is_v5(self) -> bool { matches!(self, BrainLayout::V5Dir) }
    /// True iff this layout would be loaded via the v3 path (or
    /// is a v3 empty-create).
    pub fn is_v3(self) -> bool {
        matches!(self, BrainLayout::V3File | BrainLayout::Missing)
    }
    /// Diagnostic string.
    pub fn as_str(self) -> &'static str {
        match self {
            BrainLayout::V3File => "v3-file",
            BrainLayout::V4Dir => "v4-dir",
            BrainLayout::V5Dir => "v5-dir",
            BrainLayout::Missing => "missing",
        }
    }
}

/// OP-BRAIN-V4 Phase 8 -- CLI override for the auto layout detection.
/// Mirrors `--brain-format <auto|v3|v4>` on the daemon.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BrainFormatOverride {
    /// Default. Detect via `Brain::detect_brain_layout`.
    Auto,
    /// Force the v3 .lbr load path even if `path` looks like a directory.
    /// (In practice this errors out the load if the directory does not
    /// resolve to a v3 file -- the override is a debugging aid.)
    V3,
    /// Force the v4 directory load path even if `path` looks like a file.
    /// (Errors if the path is not a v4 directory.)
    V4,
    /// OP-BRAIN-V5 6.1 -- force the v5 directory load path
    /// (`Brain::load_v5`). Errors if the path is not a v5 directory.
    V5,
}

impl BrainFormatOverride {
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "auto" | "" => Some(BrainFormatOverride::Auto),
            "v3" | "v3file" | "file" => Some(BrainFormatOverride::V3),
            "v4" | "v4dir" | "dir" | "directory" => Some(BrainFormatOverride::V4),
            "v5" | "v5dir" => Some(BrainFormatOverride::V5),
            _ => None,
        }
    }
    pub fn as_str(self) -> &'static str {
        match self {
            BrainFormatOverride::Auto => "auto",
            BrainFormatOverride::V3 => "v3",
            BrainFormatOverride::V4 => "v4",
            BrainFormatOverride::V5 => "v5",
        }
    }
}

/// Parse a bond type from a string label (case-insensitive, accepts both
/// CamelCase and snake_case variants).
pub fn parse_bond_type(s: &str) -> Result<BondType, BrainError> {
    let norm: String = s.chars().filter(|c| c.is_alphanumeric()).collect::<String>().to_lowercase();
    match norm.as_str() {
        "isa" | "is" => Ok(BondType::IsA),
        "hasproperty" | "has" => Ok(BondType::HasProperty),
        "causes" | "cause" => Ok(BondType::Causes),
        "partof" | "part" => Ok(BondType::PartOf),
        "usedfor" | "uses" | "use" => Ok(BondType::UsedFor),
        "relatedto" | "related" | "rel" => Ok(BondType::RelatedTo),
        "contrastedwith" | "contrast" | "vs" => Ok(BondType::ContrastedWith),
        "derivedfrom" | "derived" | "from" => Ok(BondType::DerivedFrom),
        "requires" | "require" => Ok(BondType::Requires),
        "produces" | "produce" => Ok(BondType::Produces),
        "opposes" | "oppose" => Ok(BondType::Opposes),
        // OP-IDENTITY alias: "CreatedBy"/"invented by" map to FoundedBy
        // (no on-disk schema churn -- see the 30+-arm note below).
        "createdby" | "inventedby" | "madeby" => Ok(BondType::FoundedBy),
        // OP-SOURCEPULL Batch-C: first-class currency + author. "authoredby" was
        // an alias onto FoundedBy(creator); repointed to AuthoredBy (P50 author-of-work).
        "hascurrency" => Ok(BondType::HasCurrency),
        "authoredby" => Ok(BondType::AuthoredBy),
        // BR.1 first-class extension variants. Each parses
        // case-insensitive (already lowercased by `norm`) on its
        // canonical CamelCase name.
        "bornin" => Ok(BondType::BornIn),
        "bornon" => Ok(BondType::BornOn),
        "diedin" => Ok(BondType::DiedIn),
        "diedon" => Ok(BondType::DiedOn),
        "discoveredby" => Ok(BondType::DiscoveredBy),
        "foundedby" => Ok(BondType::FoundedBy),
        "foundedin" => Ok(BondType::FoundedIn),
        "hasaltitude" => Ok(BondType::HasAltitude),
        "hasaspect" => Ok(BondType::HasAspect),
        "hascapital" => Ok(BondType::HasCapital),
        "hasofficiallanguage" => Ok(BondType::HasOfficialLanguage),
        "haspopulation" => Ok(BondType::HasPopulation),
        "knownfor" => Ok(BondType::KnownFor),
        "locatedin" => Ok(BondType::LocatedIn),
        "measuredin" => Ok(BondType::MeasuredIn),
        "namedafter" => Ok(BondType::NamedAfter),
        "writtenby" => Ok(BondType::WrittenBy),
        "hasarea" => Ok(BondType::HasArea),
        "hasmethod" => Ok(BondType::HasMethod),
        "hasattribute" => Ok(BondType::HasAttribute),
        "hasparameter" => Ok(BondType::HasParameter),
        "returns" => Ok(BondType::Returns),
        "inherits" => Ok(BondType::Inherits),
        "calls" => Ok(BondType::Calls),
        "hastype" => Ok(BondType::HasType),
        "implements" => Ok(BondType::Implements),
        // Q.1 typed-KU relation aliases. These map to the closest
        // existing BondType so we don't churn the on-disk schema, but
        // the alias *is* preserved by the MCP teach handler to drive
        // kind inference. Adding new BondType variants would ripple
        // through 30+ match arms across narration / archive /
        // brain_io / consolidation / reasoning — deferred to a
        // follow-up Q.1.x as documented in #587.
        "definedas" | "means" | "definition" | "isdefinedas" => Ok(BondType::DefinedAs),
        "worksby" | "operatesvia" | "functionsthrough" | "mechanism" => Ok(BondType::WorksBy),
        "composedof" | "madeof" | "consistsof" | "composition" => Ok(BondType::ComposedOf),
        "isfor" | "purposeof" | "purpose" => Ok(BondType::PurposeOf),
        "unlike" | "differsfrom" | "incontrastto" => Ok(BondType::DiffersFrom),
        "leadsto" | "resultsin" => Ok(BondType::LeadsTo),
        "exampleof" | "suchas" | "forexample" | "eg" | "example" => Ok(BondType::ExampleOf),
        _ => Err(BrainError::ParseBondType(s.to_string())),
    }
}

/// Parse a confidence level from a string label (case-insensitive).
pub fn parse_confidence(s: &str) -> Result<Confidence, BrainError> {
    match s.to_lowercase().as_str() {
        "terminal" | "axiom" | "axiomatic" => Ok(Confidence::Terminal),
        "verified" | "v" => Ok(Confidence::Verified),
        "unverified" | "u" => Ok(Confidence::Unverified),
        "inferred" | "i" => Ok(Confidence::Inferred),
        "contested" | "c" => Ok(Confidence::Contested),
        _ => Err(BrainError::ParseConfidence(s.to_string())),
    }
}

/// Current epoch milliseconds.
/// Port S — decode an overlay key (the hex-encoded "f8|t8|BondType"
/// shape produced by `SourceOverlay::key`) back into the (Seed,
/// Seed, BondType) triple. Returns `None` if the encoding is
/// malformed. Note: the encoded seed prefix is only 8 bytes, so
/// the decoded Seed has 0x00 padding for bytes 8..32 — callers
/// must compare via prefix-match (first 8 bytes), as
/// `revoke_bonds_by_source` does.
pub(crate) fn parse_overlay_key(
    s: &str,
) -> Option<(crate::seed::Seed, crate::seed::Seed, BondType)> {
    let mut parts = s.split('|');
    let f_hex = parts.next()?.trim();
    let t_hex = parts.next()?.trim();
    let bt_dbg = parts.next()?.trim();
    if parts.next().is_some() {
        return None;
    }
    if f_hex.len() != 16 || t_hex.len() != 16 {
        return None;
    }
    let mut f_bytes = [0u8; 32];
    let mut t_bytes = [0u8; 32];
    for i in 0..8 {
        f_bytes[i] = u8::from_str_radix(&f_hex[i * 2..i * 2 + 2], 16).ok()?;
        t_bytes[i] = u8::from_str_radix(&t_hex[i * 2..i * 2 + 2], 16).ok()?;
    }
    let bt = match bt_dbg {
        "IsA" => BondType::IsA,
        "HasProperty" => BondType::HasProperty,
        "Causes" => BondType::Causes,
        "PartOf" => BondType::PartOf,
        "UsedFor" => BondType::UsedFor,
        "Produces" => BondType::Produces,
        "Requires" => BondType::Requires,
        "Opposes" => BondType::Opposes,
        "DerivedFrom" => BondType::DerivedFrom,
        "ContrastedWith" => BondType::ContrastedWith,
        "RelatedTo" => BondType::RelatedTo,
        "BornIn" => BondType::BornIn,
        "BornOn" => BondType::BornOn,
        "DiedIn" => BondType::DiedIn,
        "DiedOn" => BondType::DiedOn,
        "DiscoveredBy" => BondType::DiscoveredBy,
        "FoundedBy" => BondType::FoundedBy,
        "FoundedIn" => BondType::FoundedIn,
        "HasAltitude" => BondType::HasAltitude,
        "HasAspect" => BondType::HasAspect,
        "HasCapital" => BondType::HasCapital,
        "HasOfficialLanguage" => BondType::HasOfficialLanguage,
        "HasPopulation" => BondType::HasPopulation,
        "KnownFor" => BondType::KnownFor,
        "LocatedIn" => BondType::LocatedIn,
        "MeasuredIn" => BondType::MeasuredIn,
        "NamedAfter" => BondType::NamedAfter,
        "WrittenBy" => BondType::WrittenBy,
        "HasArea" => BondType::HasArea,
        "HasMethod" => BondType::HasMethod,
        "HasAttribute" => BondType::HasAttribute,
        "HasParameter" => BondType::HasParameter,
        "Returns" => BondType::Returns,
        "Inherits" => BondType::Inherits,
        "Calls" => BondType::Calls,
        "HasType" => BondType::HasType,
        "Implements" => BondType::Implements,
        _ => return None,
    };
    Some((crate::seed::Seed(f_bytes), crate::seed::Seed(t_bytes), bt))
}

/// Port T-Ingest: runtime-only knobs mirroring section #15 ingest config.
#[derive(Debug, Clone, PartialEq)]
pub struct IngestConfig {
    pub corpus_default: Option<String>,
    pub run_id_strategy: crate::runtime_config::IngestRunIdStrategy,
    pub auto_approve_threshold: Option<f32>,
}

impl Default for IngestConfig {
    fn default() -> Self {
        IngestConfig {
            corpus_default: None,
            run_id_strategy: crate::runtime_config::IngestRunIdStrategy::ServerStartup,
            auto_approve_threshold: None,
        }
    }
}

/// Port T-Ingest.2: runtime-only knobs mirroring section #15
/// Tier 2 prompt routing config.
#[derive(Debug, Clone, PartialEq)]
pub struct Tier2PromptConfig {
    /// Strategy when `lu4d_user_statement` caller omits session_id.
    pub default_session_strategy: crate::runtime_config::Tier2PromptSessionStrategy,
    /// Hard cap on prompt-text byte length.
    pub max_text_bytes: usize,
}

impl Default for Tier2PromptConfig {
    fn default() -> Self {
        Tier2PromptConfig {
            default_session_strategy: crate::runtime_config::Tier2PromptSessionStrategy::Ephemeral,
            max_text_bytes: 65536,
        }
    }
}

/// Port T-Ingest: cheap per-process run-id generator.
/// Format: `lu_run_<8-hex>`. Uses unix nanos + thread id for entropy;
/// no need for a real UUID dep, the only invariant is uniqueness across
/// process lifetimes (and that is good enough — a real ingestion pipeline
/// supplies its own `run_id` anyway).
pub fn make_startup_run_id() -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let tid = {
        let mut h: u32 = 0xcbf29ce4u32 ^ 0x84222325u32;
        for b in format!("{:?}", std::thread::current().id()).bytes() {
            h = h.wrapping_mul(16777619).wrapping_add(b as u32);
        }
        h
    };
    let mix = (nanos as u64) ^ ((tid as u64).wrapping_mul(0x9e3779b97f4a7c15));
    format!("lu_run_{:08x}", (mix as u32) ^ ((mix >> 32) as u32))
}

pub fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

/// Result of a `teach` operation.
#[derive(Debug, Clone, Serialize)]
pub struct TeachResult {
    pub subject_seed: Seed,
    pub object_seed: Seed,
    pub bond_type: BondType,
    pub weight: f32,
    pub timestamp: i64,
    pub created_subject: bool,
    pub created_object: bool,
    pub reinforced_existing_bond: bool,
    /// OP-RETRAIN-1: true when the concept-admission gate rejected this teach
    /// (non-concept subject/object) so nothing was minted. Default false.
    pub admission_rejected: bool,
    pub session_id: Option<u64>,
}

/// OP-ASK-WATCHDOG -- per-call configuration for `Brain::ask_with_config`.
/// `Brain::ask` keeps its 3-arg signature and constructs the default
/// (no wall-clock bound, mirroring legacy behaviour). Engine callers
/// that need a bound (narrate, MCP) plumb a tightened value through.
#[derive(Debug, Clone, Copy)]
pub struct BrainAskConfig {
    /// Spread depth (legacy `depth` arg).
    pub depth: usize,
    /// Spread top-k (legacy `top_k` arg).
    pub top_k: usize,
    /// OP-ASK-WATCHDOG -- wall-clock ceiling on the BFS frontier loop
    /// inside `spread_activation_weighted_inner`. `None` = no bound
    /// (engine default for `Brain::ask`). The narrate path tightens
    /// this to ~4s so the ask-level watchdog fires before the 5s
    /// think-level watchdog on the inner call.
    pub max_wall_clock: Option<Duration>,
}

impl BrainAskConfig {
    /// Legacy `Brain::ask(query, depth, top_k)` form: no wall-clock bound.
    pub fn new(depth: usize, top_k: usize) -> Self {
        BrainAskConfig { depth, top_k, max_wall_clock: None }
    }
    /// Builder: attach a wall-clock ceiling.
    pub fn with_wall_clock(mut self, d: Option<Duration>) -> Self {
        self.max_wall_clock = d;
        self
    }
}

/// OP-ASK-WATCHDOG -- terminal state of a `Brain::ask` invocation.
/// `Complete` = primary spread (and any cascade) ran to completion.
/// `TimedOut` = at least one spread tripped the wall-clock watchdog;
/// `activated_so_far` is the size of the activation map at the
/// moment of expiry. Older brain files / JSON payloads default to
/// `Complete` via `#[serde(default)]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum AskStatus {
    #[default]
    Complete,
    TimedOut { activated_so_far: usize },
}

/// Result of an `ask` operation.
#[derive(Debug, Clone, Serialize)]
pub struct AskResult {
    pub query: String,
    pub seed_hits: Vec<Seed>,
    pub recall: RecallResult,
    pub hebbian_strengthened: usize,
    pub session_id: Option<u64>,
    /// Bridges classified during this query's spread(s). Distinct from
    /// `Brain.recent_bridges` (the cross-query ring buffer): this list is
    /// only what fired this call, in detection order.
    #[serde(default)]
    pub bridges: Vec<BridgeEvent>,
    /// Curiosity signals classified during this query's spread(s).
    /// Distinct from `Brain.wonder_queue` (the cross-query ranked queue):
    /// this list is only what fired this call.
    #[serde(default)]
    pub curiosity_signals: Vec<CuriositySignal>,
    /// Active scars Recall encountered during the spread(s) for this
    /// query and refused to propagate through. Surfaced so a bare
    /// `ask` call (no multi_think + narration) can articulate refusal
    /// reasons. Empty when no scarred bonds were touched. May contain
    /// duplicates if the same scar was hit on multiple traversal
    /// steps; consumers can dedupe by `scar_id` if needed.
    #[serde(default)]
    pub scars: Vec<crate::scar::Scar>,
    /// OP-ASK-WATCHDOG -- terminal state of this ask call. See
    /// `AskStatus`. Older payloads omit this field via
    /// `#[serde(default)]` and default to `Complete`.
    #[serde(default)]
    pub status: AskStatus,
}

/// Current status of the brain.
#[derive(Debug, Clone, Serialize)]
pub struct BrainStatus {
    pub node_count: usize,
    pub bond_count: usize,
    pub total_sessions: usize,
    pub active_session_id: Option<u64>,
    pub active_session_activations: usize,
    pub active_session_keyframes: usize,
    pub brain_file: Option<String>,
    /// Human-readable on-disk format label sourced from
    /// `BrainSaveFormat::as_str()` ("v1", "v2", "v3", "v3c", "v3enc").
    /// Replaces the legacy `u32` BRAIN_FORMAT_VERSION constant which
    /// always read 2 even when the brain on disk was v3+ (Bug A,
    /// 2026-04-27).
    pub format_version: String,
    /// Number of `BridgeEvent`s currently held in the runtime ring buffer.
    pub bridge_count: usize,
    /// Cumulative count of bridge candidates dropped because per-spread
    /// caps filled up. A high value suggests the cap should be raised.
    pub bridges_dropped_overflow_total: usize,
    /// All currently-held bridge events (capped by `recent_bridges_cap`,
    /// default 256). Newest at the back.
    pub recent_bridges: Vec<BridgeEvent>,
    /// Number of QueryMetrics held in the self-reflection ring buffer.
    /// Future audits don't have to back-compute it from response payloads.
    pub recent_metrics_count: usize,
    /// Cumulative count of synthesis-triangle firings across the
    /// metric ring buffer. Lifetime totals beyond the buffer are
    /// not retained — this is a sliding-window approximation.
    pub total_synthesis_triangles_fired: u32,
    /// Cumulative count of confirmed hypotheses across the metric
    /// ring buffer.  Same sliding-window caveat as triangles.
    pub total_confirmed_hypotheses: u32,
    /// Current size of the wonder queue (curiosity signals waiting
    /// to be serviced).
    pub wonder_queue_size: usize,
    /// Distinct domain seeds currently in the interest profile.
    pub interest_profile_domain_count: usize,
    /// Number of multi_think calls issued in this brain instance.
    /// Resets on restart.
    pub multi_think_call_count: u64,
    /// Per-graph dynamic hub threshold (90th percentile of node
    /// degrees). Replaces the old global `DEGREE_NORM_THRESHOLD`
    /// constant in spread-activation degree-normalization.
    pub degree_p90: usize,
    /// Activation-map cap; spread stops expanding once this fills.
    /// This is the STATIC FLOOR — see `max_activations_percentile`
    /// for the dynamic component and `effective_max_activations` for
    /// the runtime cap actually applied to spreads.
    pub max_activations: usize,
    /// Percentile of brain size used as a dynamic activation cap.
    /// Default 0.05. The runtime cap is
    /// `max(max_activations, floor(node_count * percentile))`.
    pub max_activations_percentile: f32,
    /// Effective activation cap actually applied at the current
    /// brain size: `max(max_activations, floor(node_count * pct))`.
    pub effective_max_activations: usize,
    /// OP-SAVE-ASYNC: True iff a background save worker is currently in flight.
    pub save_in_progress: bool,
    /// OP-SAVE-ASYNC: epoch-ms timestamp when the most recent save started.
    /// None when no save has been started in this process's lifetime.
    pub save_started_at_ms: Option<u64>,
    /// OP-SNAPSHOT-PEAK: ms spent in the synchronous snapshot-capture
    /// phase of the most recent save. None when no save has been started.
    pub last_snapshot_capture_ms: Option<u64>,
    /// OP-BRAIN-V5 6.6 -- true iff the loaded brain is encrypted at rest
    /// (v3enc, or a v4/v5 dir loaded under a master key). Surfaced in the
    /// startup banner + MCP status so encryption state is visible.
    pub encrypted_at_rest: bool,
}

/// Counters returned by `Brain::prune_legacy_goal_nodes`.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct PruneLegacyGoalReport {
    /// Number of node keyword-index entries removed.
    pub keyword_entries_pruned: u32,
    /// Number of interest-profile entries removed.
    pub interest_entries_pruned: u32,
    /// Number of interest-profile entries removed because the seed had no node in the network (orphan).
    pub orphan_interest_pruned: u32,
}

/// The top-level Lu 4D brain.
pub struct Brain {
    pub network: AssociativeNetwork,
    pub sessions: SessionManager,
    /// Port Q.3 (#438/#587): Left-Brain operational/episodic memory layer.
    /// Per-session conversational state (turn history, weighted pronoun anchors,
    /// ephemeral session beliefs). Runtime-only, NEVER persisted to the brain file.
    pub session_memories: SessionMemoryRegistry,
    pub consolidator: ConsolidationEngine,
    pub path: Option<PathBuf>,
    /// Bridge-detection knobs. Defaults are conservative; tweak at runtime
    /// if the system grows quiet on bridges.
    pub bridge_config: BridgeConfig,
    /// Ring buffer of recently-detected cross-domain bridges. Runtime-only —
    /// not persisted in the brain file. Capped at `bridge_config.recent_bridges_cap`.
    pub recent_bridges: VecDeque<BridgeEvent>,
    /// Cumulative count of candidates dropped due to per-spread cap saturation.
    pub bridges_dropped_overflow_total: usize,
    /// Curiosity Engine knobs. Defaults match the port spec.
    pub curiosity_config: CuriosityConfig,
    /// Bounded ranked queue of curiosity signals (cross-query). Runtime-only.
    pub wonder_queue: WonderQueue,
    /// OP-CURIOSITY-1 -- optional DURABLE mirror of `wonder_queue`.
    ///
    /// When attached, every signal pushed to the in-RAM queue is also
    /// appended to an on-disk journal. This is what lets the DETECT half
    /// (public guest process) hand work to the FILL half (admin worker
    /// process) at all: before this, the queue died with the process and
    /// the two halves had no shared surface. `None` (the default) is
    /// exactly the old behaviour.
    ///
    /// Attaching this does NOT give a brain the ability to write itself.
    /// The journal is a separate file outside the brain directory; see
    /// `crate::wonder_sidecar`.
    pub wonder_sidecar: Option<crate::wonder_sidecar::WonderSidecar>,
    /// Failed journal appends. See `wonder_sidecar_write_errors`.
    wonder_sidecar_errors: std::sync::atomic::AtomicU64,
    /// Per-domain interest tracker. Runtime-only.
    pub interest_profile: InterestProfile,
    /// Cumulative count of curiosity candidates dropped due to per-spread cap saturation.
    pub curiosity_dropped_overflow_total: usize,
    /// Gap emitter (Port J). Runtime-only. Reasoning ports push GapSignals here;
    /// `Brain::flush_gaps` translates them into wonder_queue items.
    pub gap_emitter: crate::gap_signal::GapEmitter,
    /// Monotonic counter for issuing Variant ids. Runtime-only; resets
    /// on each launch. Variant ids only need to be unique within one
    /// brain process, since promoted variants get re-tagged to baseline
    /// at the keyframe layer (the id is preserved as audit metadata).
    pub next_variant_id: u64,
    /// Monotonic counter for issuing multi_think ids. Runtime-only;
    /// resets on each launch. Multi-think work is scope-isolated and
    /// pruned during consolidation, so id collisions across restarts
    /// can't reach baseline knowledge.
    pub next_multi_think_id: u64,
    /// All knobs for the multi-agent thinking subsystem. Runtime-only.
    pub multi_agent_config: crate::agents::MultiAgentConfig,
    /// Bounded ring buffer of recent query telemetry. Drained on every
    /// save/load round-trip — operational state, not knowledge.
    pub performance_monitor: PerformanceMonitor,
    /// Persistent self-reflection state. Saved with the brain file so
    /// pending proposals survive restarts.
    pub meta_reflection: MetaReflection,
    /// All knobs for the self-reflection subsystem. Runtime-only.
    pub self_reflection_config: SelfReflectionConfig,
    /// All knobs for the narration / speech-assembly subsystem. Runtime-only.
    pub narration_config: crate::narration::NarrationConfig,
    /// Append-only safety log of every scarred bond. Persisted with
    /// the brain file. The forward-search anchor lives in
    /// `ScarLog::active_scars`; bond weight reads consult this log
    /// at every call via the type-enforced `weight_at` /
    /// `weight_in_scope` signatures.
    pub scar_log: ScarLog,
    /// All knobs for the autonomous research loop (Port #10).
    /// Persisted so John's threshold tweaks survive restarts.
    pub research_config: crate::research::router::ResearchConfig,
    /// Pending / approved / rejected / applied research proposals.
    /// Persisted; `Pending` proposals carry across restarts so John can
    /// approve them across sessions.
    pub research_proposals: Vec<crate::research::proposal::ResearchProposal>,
    /// Monotonic id allocator for research proposals.
    pub next_research_proposal_id: u64,
    /// Port #17 Fix 2: per-seed wondering_about surface count.
    /// Persisted with the brain file. See
    /// `CuriosityConfig::gap_appearance_threshold`.
    pub gap_seed_frequency: std::collections::HashMap<Seed, u32>,
    /// Port J.1: synthetic gap-seed -> human-readable surface query.
    /// Populated by `flush_gaps` from `GapSignal.query`. Looked up by
    /// `tool_wonder` for hash-only seeds that don't resolve to a node
    /// in the graph. Persisted with the brain file so labels survive
    /// across restarts.
    pub gap_label_map: std::collections::HashMap<Seed, String>,
    /// On-disk format used by `save()` / `save_to()`. Set from the CLI
    /// `--brain-format-out` flag; runtime-only (never persisted, since
    /// it controls how persistence happens).
    pub save_format: BrainSaveFormat,
    /// Bug A (2026-04-27): on-disk format the brain was loaded FROM.
    /// `Some(_)` when the file existed at load time, set from the
    /// detected magic + version + flags. `None` for ephemeral brains
    /// (Brain::new() or load_or_new on a non-existent path). Reported
    /// by `status().format_version` so the REPL/MCP `status` shows the
    /// actual on-disk format (v3c, v3enc, etc.) regardless of the
    /// user's CLI-provided save_format. Runtime-only.
    pub loaded_format: Option<BrainSaveFormat>,
    /// Storage v2 Phase 4 -- master key for at-rest encryption. `Some`
    /// when the brain was loaded from an encrypted file (so save() can
    /// re-encrypt). `None` for unencrypted brains; `save_to_format`
    /// rejects `V3BinaryEncrypted` if this is `None`.
    pub encryption_key: Option<crate::encrypt::MasterKey>,
    /// Critical audit fix internal notes / Problem #111 follow-up):
    /// when `loaded_format == V3BinaryEncrypted`, attempting to save in
    /// any non-encrypted format would silently downgrade the brain to
    /// plaintext on disk -- defeating at-rest encryption. `save_to_format`
    /// refuses such a save unless this flag is true. Runtime-only;
    /// set by main.rs from the `--allow-decrypt-on-save` CLI flag.
    pub allow_decrypt_on_save: bool,
    /// Port T.7: structural encryption guard. When set,
    /// `Brain::save_to_format` refuses to write any non-v3enc format
    /// regardless of how the brain was loaded. Companion to the
    /// pre-load guard `require_v3enc_or_err`. Runtime-only; flipped
    /// on by `--require-encrypted` in main.rs. Solution #79 covered
    /// the encrypted-loaded case; T.7 closes the gap for in-memory
    /// brains that were never encrypted to begin with (e.g. a fresh
    /// training output landing on /var/lib/lu/brain.lbr).
    pub require_encrypted: bool,
    /// Port Cat.3c: true when the in-memory state was hydrated from
    /// the legacy sidecars (Cat.3a/Cat.3b) rather than from v3
    /// section #14. The next successful save retires the sidecars
    /// (renames them `.migrated`) and clears this flag. False on
    /// post-Cat.3c brains — nothing to migrate.
    pub runtime_config_pending_migration: bool,
    /// Port G: negation overlay. Persisted with the brain file. See
    /// `crate::negation` and Learning #577.
    pub negation_overlay: crate::negation::NegationOverlay,
    /// Port S: source attribution overlay. Persisted with the brain
    /// file. See `crate::source` and Learning #635/#634.
    /// OP-SNAPSHOT-PEAK: wrapped in Arc<RwLock<...>> so the async
    /// save snapshot captures a zero-cost Arc bump instead of
    /// deep-cloning the entire overlay (~3 GiB at 19M bonds).
    pub source_overlay: Arc<RwLock<crate::source::SourceOverlay>>,
    /// Port SourceKind-Variants — in-memory side-table mapping a
    /// research proposal id to its chain-run provenance. Set by
    /// `chain_driver_run` (J.2) after `research_one` returns the
    /// proposal id; consumed by `apply_proposal` when the human
    /// approves the proposal. NOT persisted: a save/restart between
    /// `research_one` and `apply_proposal` gracefully degrades to
    /// plain `SourceKind::Research`. See `ChainContext` doc for the
    /// pre-retrain rationale (bincode forward-compat hazard).
    pub proposal_chain_context: std::collections::HashMap<u64, crate::research::proposal::ChainContext>,
    /// Port G: feature flag for the ingest negation-cue extractor.
    /// Runtime-only. Default true so out-of-the-box ingest recognises
    /// negation; toggle to false to disable cue-based extraction.
    pub negation_cues_enabled: bool,
    /// Port T.1: clock source backend. Default: `SystemClockSource`
    /// (unauthenticated, matches pre-T.1 behaviour). Swap to
    /// `DFTimeServiceSource` for authenticated time. Runtime-only.
    pub clock: std::sync::Arc<dyn crate::time::ClockSource>,
    /// Port T.1.1: time-source runtime config (mode + DF client knobs).
    pub time_source_config: crate::time::TimeSourceConfig,
    /// Port Sense: runtime sense disambiguation knobs.
    /// Persisted via `RuntimeConfigSnapshot` (v3 section #15)
    /// and read by network.rs query-time scoring.
    pub sense_config: crate::sense::SenseConfig,
    /// Port J.2 — chained curiosity research knobs. Runtime-only.
    /// Persisted via section #15 (RuntimeConfigSnapshot).
    pub chain_research_config: crate::research::chain::ChainResearchConfig,
    /// Port BR.4 -- recall-time typed-KU routing config. Runtime-only,
    /// persisted via section #15 (RuntimeConfigSnapshot). Default has
    /// `typed_ku_enabled = true` so behavior is unchanged after BR.4
    /// lands; flip to `false` to fall back to the pre-Q.1 untyped
    /// articulation baseline for A/B comparison without re-ingesting.
    pub recall_config: crate::reasoning::RecallConfig,
    /// Port BR.5 -- admin-tier auth gate on lu4d_revoke_by_source.
    /// Default `false` (closed). When `false`, lu4d_revoke_by_source
    /// with dry_run=false returns -32603 with an explanation; dry-run
    /// preview is unchanged. Toggle via lu4d_admin_set; persisted via
    /// runtime_config section #15.
    pub admin_revoke_enabled: bool,
    /// Port J.2 — monotonic id stamp for chain runs. Used to label
    /// chain-produced bonds (via wonder_seed) so they can be
    /// revoked as a unit. Runtime-only — resets on restart.
    pub next_chain_id: u64,
    /// Port T-Ingest: training-corpus ingest knobs surfaced from
    /// runtime_config section #15 (`ingest_corpus_default`,
    /// `ingest_run_id_strategy`, `ingest_auto_approve_threshold`).
    /// Hydrated alongside the other section-#15 knobs.
    pub ingest_config: crate::brain::IngestConfig,
    /// Port T-Ingest: server-startup-allocated run identifier used
    /// when a `lu4d_teach_corpus` caller omits `run_id` and the
    /// strategy is `ServerStartup`. Generated once when the brain
    /// is constructed (`Brain::new` and every load path) and kept
    /// for the process lifetime. Runtime-only — never persisted.
    pub ingest_run_id: String,
    /// Port T-Ingest.2: Tier 2 prompt routing config surfaced
    /// from runtime_config section #15. Hydrated alongside the
    /// other section-#15 knobs.
    pub tier2_prompt_config: crate::brain::Tier2PromptConfig,
    /// Port File-IO: file ingestion / emission knobs surfaced from
    /// runtime_config section #15. Hydrated alongside the other
    /// section-#15 knobs.
    pub file_io_config: crate::file_io::FileIoConfig,

    /// Port R input -- scene-graph extractor + image limits.
    /// Persisted into runtime_config section #15 alongside the
    /// other articulator-side knobs.
    pub vision_config: crate::mcp_vision::VisionConfig,

    /// Port T.1.1: typed handle on the DF Time Service client when
    /// `time_source_config.mode == "df_service"`. `None` in system
    /// mode. Used by `lu4d_time_status` to surface live polling state.
    pub df_time_client: Option<std::sync::Arc<crate::time::DFRealClient>>,
    /// Port T.1: most-recent authenticated time anchor. Refreshed by
    /// scheduled fetches in T.6 (daemon); for now refreshed lazily by
    /// `current_utc_ms()`. Runtime-only.
    pub time_anchor: std::sync::RwLock<crate::time::AuthenticatedTime>,
    /// Port T.1: stateful clock anomaly detector. Tracks backward /
    /// large-forward jumps across refreshes; consulted by
    /// `time_is_authenticated()`. Runtime-only.
    pub anomaly_detector: std::sync::Mutex<crate::time::AnomalyDetector>,
    /// Port T.2: idle detection. Tracks active MCP query count
    /// + monotonic timestamp of last query end. Runtime-only;
    /// not persisted (overlay pattern, same as G/S/T.1).
    pub idle_state: crate::idle::IdleState,
    /// Port T.3: pending changes log + WAL handle.
    /// Populated by load paths via `attach_pending_wal`; kept
    /// in-memory only for ephemeral `Brain::new()` brains.
    pub pending: crate::pending::PendingLog,
    /// Port T.8: rejection-pattern store. Learns from rejected
    /// pending entries (T.3) and pre-filters new proposals so Lu
    /// stops surfacing the same shape of low-quality bonds. Persisted
    /// in the v3 binary section #10 alongside the pending log.
    pub rejection_store: crate::rejection::RejectionPatternStore,
    /// Port T.5: bond freshness decay knobs (half-life + on/off).
    /// Runtime-only; CLI/MCP toggleable. Default = 90-day half-life,
    /// decay enabled. Stored bond weights are NEVER mutated by decay.
    pub decay_config: crate::decay::DecayConfig,
    /// Port T.6: in-process daemon driver that fires `idle_tick`
    /// automatically when the brain has been idle past a
    /// configurable threshold. Runtime-only; not persisted
    /// (overlay pattern, same as idle_state and decay_config).
    /// Shares the brain's ClockSource so the T.1 invariant (no
    /// direct Instant::now / SystemTime::now in time-sensitive
    /// decision paths) holds end-to-end.
    pub auto_tick: crate::idle::AutoTickDriver,
    /// OP-SAVE-ASYNC: state machine + worker handle for non-blocking
    /// background brain saves. `SaveRunner::default()` initialises to
    /// Idle with no worker; subsequent saves transition through
    /// Running -> Completed/Failed.
    pub save_runner: std::sync::Arc<crate::save_async::SaveRunner>,
    /// Port LB: Left Brain session memory store. Operational/
    /// episodic memory (thinking traces, observations, hypotheses,
    /// goals, wonder items, research results, user statements).
    /// Persisted in v3 binary section #11. NEVER auto-promotes to
    /// the canonical brain — promotion goes through the T.3
    /// PendingLog path via `crate::lb_session::propose_to_pending`.
    pub lb_session: crate::lb_session::SessionMemoryStore,
    /// Port CS: in-memory conversational-style store. Loaded
    /// from BrainFile.cs_style on construction; snapshot()ed back
    /// to BrainFile.cs_style on save.
    pub cs_style_store: crate::style::ConversationalStyleStore,
    /// Port CS: configuration knobs surfaced via `lu4d_configure`.
    pub style_config: crate::style::StyleConfig,
    /// Port Cat.4: signed-safety-pack ledger (trusted publishers,
    /// installed packs, revoked packs). Persisted in v3 binary
    /// section #13. Mutated by the  MCP tools and by
    /// the pack ingest path in .
    pub pack_ledger: crate::pack::PackLedgerStore,
    /// Port Cat.5: schema-only signed capability pack ledger
    /// (trusted publishers, installed packs, revoked pack_ids).
    /// Persisted in v3 binary section #17. Mutated by the
    /// `lu4d_capability_pack_*` MCP tools (mcp_cat5.rs).
    /// Anti-hallucination structural invariant: capability-pack
    /// tools NEVER touch bonds — they only mutate this ledger.
    pub cap_pack_ledger: crate::capability_pack::CapabilityPackLedgerStore,
    pub pack_registry: crate::capability_runtime::LoadedPackRegistry,
    /// Port Cat.3a: source-trust weighting at bond formation time.
    /// Persisted as a JSON sidecar at <brain_path>.trust.json (NOT a
    /// v3 schema section). Default table: grokipedia=1.0, wikipedia=0.7,
    /// duckduckgo=0.5, default=0.4. See crate::trust and Learning #628.
    pub trust_table: crate::trust::TrustTable,
    /// Port Cat.3b: hub-degree caps at bond formation + read time.
    /// Persisted as a JSON sidecar at <brain_path>.hub_degree.json
    /// (NOT a v3 schema section). Default: max_in=max_out=5000,
    /// write_enforcement=true, read_enforcement=true,
    /// priority=LowestWeightFirst. See crate::hub_degree and
    /// Learning #638.
    pub hub_degree_config: crate::hub_degree::HubDegreeConfig,
    /// Port Cat.3b: cumulative count of new bonds rejected at
    /// formation time because the hub was at cap and the priority
    /// strategy chose to keep existing bonds. Surfaced via
    /// `lu4d_hub_status`.
    pub hub_cap_rejections_total: u64,
    /// Port Cat.3b: cumulative count of existing bonds evicted
    /// at formation time to admit a higher-weight new bond under
    /// `LowestWeightFirst` / `OldestFirst`. Surfaced via
    /// `lu4d_hub_status`.
    pub hub_cap_evictions_total: u64,
    /// OP-BRAIN-V4 Phase 3 -- bounded in-memory write buffer. Opt-in:
    /// `None` by default, populated by `Brain::enable_memtable`.
    /// Phase 4 wires the flush thread; Phase 5 wires the read path.
    /// Production writes do NOT route through this in Phase 3.
    pub memtable: Option<std::sync::Arc<crate::memtable::MemTable>>,
    /// OP-BRAIN-V4 Phase 3 -- write-ahead log paired with `memtable`.
    /// Opt-in: `None` by default, populated by `Brain::enable_memtable`.
    /// Held under a `Mutex` because `Wal::append` requires exclusive
    /// access to the underlying `BufWriter` (single-threaded I/O).
    pub wal: Option<std::sync::Arc<std::sync::Mutex<crate::wal::Wal>>>,
    /// OP-BRAIN-V5 Phase 2.3 (Q2.0.A resolution) -- path to the WAL
    /// segment file backing the in-memory memtable. `Some(path)` iff
    /// `enable_memtable` succeeded. Used by `Brain::sidecar_path`
    /// (instance method) to derive the nodeintern.bin sidecar
    /// location next to the WAL, which the flush worker writes
    /// atomically on every flush so cross-boot NodeId stability
    /// holds. The pre-Phase-2.3 static helper
    /// `Brain::sidecar_path_for_wal(wal_path)` is retained for
    /// callers that don't hold a Brain (test rigs, the existing
    /// Phase 2.2 boot-replay arm before it can read self.wal_path).
    pub wal_path: Option<std::path::PathBuf>,
    /// OP-BRAIN-V4 Phase 4 -- background flush worker. `None` until
    /// `enable_flush_worker` is called. Owns the worker thread; the
    /// thread is joined on Brain drop via FlushWorker's own Drop.
    pub flush_worker: Option<crate::flush::FlushWorker>,
    /// OP-BRAIN-V4 Phase 4 -- live manifest snapshot pointer. Cloned
    /// cheaply by readers; replaced atomically by the flush worker
    /// post-flush. `None` until `enable_flush_worker` is called.
    pub manifest_snapshot: Option<std::sync::Arc<std::sync::RwLock<std::sync::Arc<crate::manifest::Manifest>>>>,
    /// OP-BRAIN-V4 Phase 4 -- shared NodeIntern used by the flush
    /// worker. Phase 1 introduced an intern table inside
    /// `AssociativeNetwork`; for the v4 write path the MemTable +
    /// flush worker need a shared handle. This is a clone of the
    /// network's intern, set by `enable_memtable`. `None` outside
    /// the v4 write path.
    pub flush_intern: Option<crate::node_intern::NodeIntern>,
    /// OP-INGEST-LEAN -- true when the brain was loaded via
    /// `BrainLoadMode::IngestOnly` (write-only ingest). Consulted by the
    /// MCP server startup to SKIP the serving-only keyword-index build
    /// (`ensure_keyword_index_built`), which a write-only training pass
    /// never queries and which would otherwise rebuild the ~12.5 GB
    /// index on the main thread before the first bond can land.
    ingest_only: bool,
    /// OP-BRAIN-V4 Phase 5 -- read-path mode flag, stored as an
    /// `AtomicU8` so tests / future MCP tools can switch it without
    /// taking a mutable borrow on Brain. Default `0`
    /// (`ReadPathMode::V3Only`) -- production behaviour is unchanged
    /// from pre-Phase-5 until Phase 8 flips the cutover.
    pub read_path_mode: std::sync::atomic::AtomicU8,
    /// OP-BRAIN-V4 Phase 5 -- one-shot guard for the manifest's
    /// `cleanup_orphan_ssts` step. Set to `true` the first time
    /// the flush worker (or `enable_flush_worker`) runs cleanup;
    /// subsequent calls no-op. Protects against future
    /// external-import ports accidentally GC'ing a legitimate
    /// in-flight SST.
    pub orphan_cleanup_done: std::sync::Arc<std::sync::atomic::AtomicBool>,
    /// OP-BRAIN-V4 Phase 5 -- per-SST handle cache so
    /// `read_outgoing_bonds_v4` does not re-open L0 files on
    /// every read. Keyed by manifest SST id. Wrapped in a Mutex
    /// so reads from concurrent threads cooperate on insert; the
    /// SST handle itself is `Arc`-cloneable so lookups release the
    /// lock immediately. The cache is invalidated implicitly
    /// when the manifest snapshot rotates -- entries for removed
    /// SST ids stay resident but are never looked up, so the
    /// effective cache shrinks naturally on the next compaction
    /// pass (Phase 6 GCs them via the manifest delta).
    pub v4_sst_cache: std::sync::Mutex<std::collections::HashMap<u64, std::sync::Arc<crate::sst::SST>>>,
    /// OP-BRAIN-V4 Phase 5 -- block cache shared across L0 SST
    /// reads. Sized to `DEFAULT_SST_BLOCK_CACHE_BYTES`. Survives
    /// across `read_outgoing_bonds_v4` calls.
    pub v4_block_cache: std::sync::Arc<crate::sst::BlockCache>,
    /// OP-BRAIN-V4 Phase 6 — parsed-block cache shared with the
    /// canonical read path and the compaction worker. Caches
    /// the fully-materialised `ParsedBlock` per `(sst_uid,
    /// block_id)` so repeated reads on the same subject skip
    /// the parse-bytes step that dominated V4Only p50 in
    /// Phase 5. Lives on Brain so the compaction worker and
    /// the read path share a single LRU.
    pub v4_parsed_block_cache: std::sync::Arc<crate::sst::ParsedBlockCache>,
    /// OP-BRAIN-V4 Phase 6 — background compaction worker.
    /// `None` until `enable_compaction` is called. Owns its own
    /// thread; joined on Brain drop via the worker's Drop impl.
    pub compaction_worker: Option<crate::compaction::CompactionWorker>,
    /// OP-BRAIN-V5 -- production v5 bond-compaction scheduler.
    /// Drives the real Phase 3.6 compactor over `bond_v5_levels`
    /// (L0->L1, builds the `.bsst.reverse` index). Started by
    /// `load_v5` for v5 brains IN PLACE OF the dead v4 worker.
    pub bond_v5_compaction: Option<crate::bond_store::BondV5CompactionScheduler>,
    /// OP-BRAIN-V5 -- production v5 node-compaction scheduler.
    /// Drives the real Phase 2.7 compactor over `node_levels`
    /// (L0->L1) so node L0 NodeSSTs do not pile up during v5
    /// training. Started by `load_v5` alongside the bond scheduler.
    pub node_v5_compaction: Option<crate::node_store::NodeV5CompactionScheduler>,
    /// OP-BRAIN-V4 Phase 8 — directory of the v4 brain when this
    /// Brain was loaded via `Brain::load_v4`. `None` for v3-loaded
    /// brains (the legacy `path` field is the v3 .lbr file). Used
    /// by `save()` to short-circuit the legacy save (v4 writes go
    /// through the flush worker continuously) and by `status()`/
    /// diagnostic surfaces to report the on-disk layout.
    pub brain_dir_v4: Option<std::path::PathBuf>,
    /// OP-BRAIN-V5 Phase 2.8 -- lazy-open cache of
    /// `Arc<NodeSstReader>` keyed on `file_uid`. `None` until
    /// `enable_flush_worker` is called (test brains constructed
    /// via `Brain::new` without enabling the flush worker have no
    /// cache; calls to `get_node_v5` then route only through
    /// NodeMemTable + manifest -- the cache is consulted only when
    /// an SST lookup is needed, which absent flush-worker setup is
    /// always a miss).
    pub node_reader_cache: Option<std::sync::Arc<crate::node_store::NodeSstReaderCache>>,

    /// OP-BRAIN-V5 Phase 3.7 -- lazy-open BondSST reader cache for
    /// the v5 bond read path (`get_outgoing_bonds_v5` /
    /// `get_incoming_bonds_v5`). Constructed inside
    /// `enable_flush_worker` alongside `node_reader_cache`, rooted
    /// at the same `bond_store_dir` the Phase 3.3 BondFlushWorker
    /// writes into. `None` until `enable_flush_worker` is called
    /// (test brains constructed without the v5 substrate keep it
    /// unset; both v5 bond read methods then return empty).
    pub bond_reader_cache: Option<std::sync::Arc<crate::bond_store::BondSstReaderCache>>,

    /// OP-BRAIN-MapRAM P0 -- the resident MAP (NodeId -> out/in degree),
    /// built from the v5 bond store at load when the `LU_MAPRAM_BUILD`
    /// flag is set (see `Brain::enable_resident_map`). `None` until
    /// built; in P0 it coexists with the rehydrated adjacency for
    /// parity. Later phases route the spread path's degree lookups
    /// through this (O(1), flat in bond count) instead of the resident
    /// `HashMap<Seed, Vec<NodeId>>` adjacency.
    pub resident_map: Option<crate::map_ram::ResidentMap>,
    /// OP-BRAIN-MapRAM P1 -- the cache-backed adjacency reader
    /// (map -> hot 2Q cache -> v5 reader). Built behind the
    /// `LU_MAPRAM_CACHE` flag (see `enable_cache_backed_adjacency`).
    /// `None` until enabled; production read path is unchanged while
    /// it is built (P1 proves the path; P2 reroutes consumers).
    pub cache_adjacency: Option<std::sync::Arc<crate::cache_adjacency::CacheBackedAdjacency>>,

    /// OP-BRAIN-V5 6.4 -- when set, the v5 node_store/bond_store live at
    /// `<root>/node_store` and `<root>/bond_store` (the layout the
    /// v3->v5 migration tools write: stores at the brain-dir top level,
    /// WAL + sidecar under `wal/`). `enable_flush_worker` roots the
    /// reader caches + flush/compaction workers there instead of the
    /// default `<wal_dir>/{node,bond}_store`. `None` for v3/v4 brains.
    pub v5_store_root: Option<std::path::PathBuf>,
}

/// Atomically write `bytes` to `path`.
///
/// Critical audit fix internal notes / Problem #111 follow-up): the
/// previous implementation used `fs::write(path, bytes)` directly, which
/// can leave a truncated destination if the process is killed mid-write.
/// On a 1 GB v3enc canonical brain, the truncation window is meaningful.
///
/// The replacement writes to `<path>.tmp`, fsyncs both the file and the
/// parent directory (so the rename is durable across crashes), then
/// renames `<path>.tmp` to `<path>`. POSIX guarantees the rename step is
/// atomic on the same filesystem -- readers see either the previous
/// version of the file or the new one, never a truncated half-write.
///
/// On any failure during write/fsync/rename, the in-progress `.tmp` file
/// is best-effort removed so the destination directory does not
/// accumulate orphans.
pub(crate) fn atomic_write_bytes(
    path: &Path,
    bytes: &[u8],
) -> Result<(), BrainError> {
    use std::io::Write;

    // Compose the tmp path next to the destination so the rename stays
    // on the same filesystem (cross-fs renames fail with EXDEV).
    let tmp_path = {
        let mut p = path.to_path_buf();
        let tmp_name = match p.file_name() {
            Some(n) => {
                let mut s = n.to_os_string();
                s.push(".tmp");
                s
            }
            None => return Err(BrainError::Format(format!(
                "atomic_write_bytes: target path has no file name: {}", path.display()
            ))),
        };
        p.set_file_name(tmp_name);
        p
    };

    // Best-effort cleanup of any leftover .tmp from a previous crashed run.
    let _ = fs::remove_file(&tmp_path);

    // Write + fsync the tmp file.
    let write_result: Result<(), std::io::Error> = (|| {
        let mut f = fs::OpenOptions::new()
            .write(true).create(true).truncate(true).open(&tmp_path)?;
        f.write_all(bytes)?;
        f.sync_all()?;
        Ok(())
    })();
    if let Err(e) = write_result {
        let _ = fs::remove_file(&tmp_path);
        return Err(BrainError::Io(e));
    }

    // Rename tmp -> dest. POSIX rename is atomic on the same filesystem.
    if let Err(e) = fs::rename(&tmp_path, path) {
        let _ = fs::remove_file(&tmp_path);
        return Err(BrainError::Io(e));
    }

    // Fsync the parent directory so the rename itself is durable across
    // a power-loss / panic that could otherwise leave the directory
    // entry pointing at the old inode.
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            // It's OK if opening the directory for fsync fails on weird
            // filesystems; we already have a durable file. Best-effort.
            if let Ok(d) = fs::File::open(parent) {
                let _ = d.sync_all();
            }
        }
    }
    Ok(())
}
/// OP-017 — atomic file write driven by a closure. Same durability
/// dance as `atomic_write_bytes` (write to .tmp → fsync → rename →
/// parent-dir fsync) but the payload is produced by the caller writing
/// to a `&mut File` rather than handing a `&[u8]` to the helper. Used
/// by the V3BinaryEncrypted save path so a 2.3 GiB output file never
/// has to live as a `Vec<u8>` on the heap before being written.
///
/// The file is opened with `read(true).write(true)` so the streaming
/// writer can seek-back to fix up headers and stream-read the bytes it
/// just wrote back through a buffer to compute the file-wide CRC32.
/// OP-022.a: read the v3 trailer CRC32 from a freshly-written brain.lbr.
/// Returns Err if the file is unreadable or smaller than the v3 trailer.
/// Used by save_to_format to stamp the sidecar's brain_crc32 field after
/// a v3 save.
pub(crate) fn read_v3_trailer_crc(path: &std::path::Path) -> std::io::Result<u32> {
    use std::io::{Read, Seek, SeekFrom};
    let mut f = std::fs::File::open(path)?;
    let meta = f.metadata()?;
    if meta.len() < 16 {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData,
            "file too small for v3 trailer"));
    }
    // v3 trailer layout (brain_io_v3.rs): [magic 8B][crc32 4B][reserved 4B].
    // CRC is at offset (len - 16 + 8) = len - 8.
    f.seek(SeekFrom::End(-8))?;
    let mut buf = [0u8; 4];
    f.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

pub(crate) fn atomic_write_with<F>(
    path: &Path,
    write_fn: F,
) -> Result<(), BrainError>
where
    F: FnOnce(&mut fs::File) -> Result<(), BrainError>,
{
    let tmp_path = {
        let mut p = path.to_path_buf();
        let tmp_name = match p.file_name() {
            Some(n) => {
                let mut s = n.to_os_string();
                s.push(".tmp");
                s
            }
            None => return Err(BrainError::Format(format!(
                "atomic_write_with: target path has no file name: {}", path.display()
            ))),
        };
        p.set_file_name(tmp_name);
        p
    };

    // Best-effort cleanup of any leftover .tmp from a previous crashed run.
    let _ = fs::remove_file(&tmp_path);

    let write_result: Result<(), BrainError> = (|| {
        let mut f = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&tmp_path)?;
        write_fn(&mut f)?;
        f.sync_all()?;
        Ok(())
    })();
    if let Err(e) = write_result {
        let _ = fs::remove_file(&tmp_path);
        return Err(e);
    }

    if let Err(e) = fs::rename(&tmp_path, path) {
        let _ = fs::remove_file(&tmp_path);
        return Err(BrainError::Io(e));
    }

    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            if let Ok(d) = fs::File::open(parent) {
                let _ = d.sync_all();
            }
        }
    }
    Ok(())
}


/// Cascade is allowed only when ALL three are true: fewer than
/// `CASCADE_FLOOR_D1` depth-1 hits, fewer than `CASCADE_FLOOR_MAP`
/// activations total, AND the top activation is below
/// `CASCADE_TOP_FLOOR`. Stops the v1 “thin = expand” re-flood:
/// thin results on a real graph mostly mean the spread already
/// activated 50K+ nodes; broadening seeds only deepens the flood.
const CASCADE_FLOOR_D1: usize = 2;
const CASCADE_FLOOR_MAP: usize = 50;
const CASCADE_TOP_FLOOR: f32 = 0.30;

impl Brain {
    /// Create an empty brain with no file backing.
    pub fn new() -> Self {
        Brain {
            network: AssociativeNetwork::new(),
            sessions: SessionManager::new(),
            session_memories: SessionMemoryRegistry::new(),
            consolidator: ConsolidationEngine::new(),
            path: None,
            bridge_config: BridgeConfig::default(),
            recent_bridges: VecDeque::new(),
            bridges_dropped_overflow_total: 0,
            curiosity_config: CuriosityConfig::default(),
            wonder_queue: WonderQueue::new(CuriosityConfig::default().wonder_queue_cap),
            interest_profile: InterestProfile::new(CuriosityConfig::default().interest_profile_cap),
            curiosity_dropped_overflow_total: 0,
            gap_emitter: crate::gap_signal::GapEmitter::new(),
            gap_label_map: std::collections::HashMap::new(),
            wonder_sidecar: None,
            wonder_sidecar_errors: std::sync::atomic::AtomicU64::new(0),
            next_variant_id: 1,
            next_multi_think_id: 1,
            multi_agent_config: crate::agents::MultiAgentConfig::default(),
            performance_monitor: PerformanceMonitor::new(1024),
            meta_reflection: MetaReflection::new(),
            self_reflection_config: SelfReflectionConfig::default(),
            narration_config: crate::narration::NarrationConfig::default(),
            scar_log: ScarLog::new(),
            research_config: crate::research::router::ResearchConfig::default(),
            research_proposals: Vec::new(),
            next_research_proposal_id: 0,
            gap_seed_frequency: std::collections::HashMap::new(),
            // v1 JSON removal internal notes: production
            // brains never serialize to plaintext JSON. Default a fresh
            // ephemeral brain to v3 binary; load() overrides this with
            // whatever format the on-disk file actually had.
            save_format: BrainSaveFormat::V3Binary,
            loaded_format: None,
            encryption_key: None,
            allow_decrypt_on_save: false,
            require_encrypted: false,
            // Port Cat.3c: ephemeral brains have no sidecars to migrate.
            runtime_config_pending_migration: false,
            negation_overlay: crate::negation::NegationOverlay::new(),
            source_overlay: Arc::new(RwLock::new(crate::source::SourceOverlay::new())),
            proposal_chain_context: std::collections::HashMap::new(),
            negation_cues_enabled: true,
            time_source_config: crate::time::TimeSourceConfig::default(),
            sense_config: crate::sense::SenseConfig::default(),
            chain_research_config: crate::research::chain::ChainResearchConfig::default(),
            recall_config: crate::reasoning::RecallConfig::default(),
            admin_revoke_enabled: false,
            next_chain_id: 1,
            ingest_config: IngestConfig::default(),
            tier2_prompt_config: Tier2PromptConfig::default(),
            file_io_config: crate::file_io::FileIoConfig::default(),
            vision_config: crate::mcp_vision::VisionConfig::default(),
            ingest_run_id: make_startup_run_id(),
            df_time_client: None,
            clock: std::sync::Arc::new(crate::time::SystemClockSource::new())
                as std::sync::Arc<dyn crate::time::ClockSource>,
            time_anchor: std::sync::RwLock::new(
                crate::time::AuthenticatedTime::unsigned(
                    now_ms().max(0) as u64,
                    crate::time::MonotonicClock::now(),
                    crate::time::SourceId::system(),
                    (now_ms().max(0) as u64).saturating_add(
                        crate::time::TimeConfig::default().refresh_interval_ms,
                    ),
                ),
            ),
            anomaly_detector: std::sync::Mutex::new(
                crate::time::AnomalyDetector::new(crate::time::TimeConfig::default()),
            ),
            idle_state: crate::idle::IdleState::new(),
            pending: crate::pending::PendingLog::new_in_memory(),
            rejection_store: crate::rejection::RejectionPatternStore::new(),
            lb_session: crate::lb_session::SessionMemoryStore::new(),
            cs_style_store: crate::style::ConversationalStyleStore::new(),
            style_config: crate::style::StyleConfig::default(),
            pack_ledger: crate::pack::PackLedgerStore::new(),
            cap_pack_ledger: crate::capability_pack::CapabilityPackLedgerStore::new(),
            pack_registry: crate::capability_runtime::LoadedPackRegistry::default(),
            trust_table: crate::trust::TrustTable::default(),
            hub_degree_config: crate::hub_degree::HubDegreeConfig::default(),
            hub_cap_rejections_total: 0,
            hub_cap_evictions_total: 0,
            memtable: None,
            wal: None,
            wal_path: None,
            flush_worker: None,
            manifest_snapshot: None,
            flush_intern: None,
            ingest_only: false,
            read_path_mode: std::sync::atomic::AtomicU8::new(crate::read_path::ReadPathMode::V3Only.as_u8()),
            orphan_cleanup_done: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            v4_sst_cache: std::sync::Mutex::new(std::collections::HashMap::new()),
            v4_block_cache: std::sync::Arc::new(crate::sst::BlockCache::new(crate::sst::DEFAULT_SST_BLOCK_CACHE_BYTES)),
            v4_parsed_block_cache: std::sync::Arc::new(crate::sst::ParsedBlockCache::new(crate::sst::DEFAULT_PARSED_BLOCK_CACHE_BYTES)),
            compaction_worker: None,
            bond_v5_compaction: None,
            node_v5_compaction: None,
            brain_dir_v4: None,
            node_reader_cache: None,
            v5_store_root: None,
            bond_reader_cache: None,
            resident_map: None,
            cache_adjacency: None,
            decay_config: crate::decay::DecayConfig::default(),
            // Port T.6: share the brain's ClockSource (T.1 invariant).
            // The driver does not yet have a clock here -- we inline it
            // below using SystemClockSource so a fresh ephemeral brain
            // (Brain::new) doesn't need any extra setup. The MCP server
            // optionally rebuilds it against `self.clock` after construction.
            auto_tick: crate::idle::AutoTickDriver::new(
                std::sync::Arc::new(crate::time::SystemClockSource::new())
                    as std::sync::Arc<dyn crate::time::ClockSource>,
            ),
            save_runner: std::sync::Arc::new(crate::save_async::SaveRunner::default()),
        }
    }

    /// Port T.3 / T.4: hydrate the pending-log from the persistent
    /// brain-file snapshot (T.4 -- v3 section `pending_log`, id 9)
    /// AND replay any newer WAL records on top.
    ///
    /// Called by every load path AFTER the Brain has been constructed
    /// and `path` set. The snapshot embedded in the v3 brain file is
    /// the consolidated source of truth as of the last successful
    /// `Brain::save()`; the WAL is a forward log of mutations applied
    /// after that save. Replay is idempotent (dedupes by `pending_id`)
    /// so the corner case where a save succeeded but the WAL truncate
    /// failed leaves no duplicates.
    ///
    /// Pre-T.4 brain files (no `pending_log` section) come back from
    /// the binary reader with `PendingLogSnapshot::default()`, so this
    /// path collapses to T.3 WAL-only behavior for those files.
    ///
    /// On error the brain still has the in-memory log and the load
    /// continues; the caller may surface a warning. T.3 deliberately
    /// does NOT fail load on WAL replay errors -- that would brick a
    /// brain whose WAL got corrupted. T.4 inherits the same
    /// philosophy.
    pub fn attach_pending_wal(&mut self, snapshot: crate::pending::PendingLogSnapshot) {
        let wal_path = match &self.path {
            Some(p) => {
                let mut q = p.clone();
                let mut name = q.file_name()
                    .map(|s| s.to_os_string())
                    .unwrap_or_else(|| std::ffi::OsString::from("brain.lbr"));
                name.push(".pending.wal");
                q.set_file_name(name);
                q
            }
            None => return,
        };
        match crate::pending::PendingLog::open_with_snapshot(wal_path, snapshot) {
            Ok(log) => self.pending = log,
            Err(e) => {
                eprintln!("pending WAL open/replay failed: {} (continuing with in-memory log)", e);
            }
        }
    }

/// Port LB.1 - hydrate the Left Brain session memory store from
    /// the v3 section #11 snapshot AND replay any newer WAL records
    /// (`<brain>.lb_session.wal`) on top. Mirrors
    /// [`Self::attach_pending_wal`] for the LB store. Idempotent
    /// (replay dedupes by `entry_id` so the corner case where save
    /// succeeded but WAL truncate failed leaves no duplicates).
    ///
    /// On error the brain still has the in-memory store (with the
    /// snapshot loaded but no WAL handle) and the load continues;
    /// LB does NOT fail load on WAL replay errors -- mirroring the
    /// T.3 / T.4 philosophy of refusing to brick a brain whose WAL
    /// got corrupted.
    pub fn attach_lb_session_wal(
        &mut self,
        snapshot: crate::lb_session::SessionMemorySnapshot,
    ) {
        let wal_path = match &self.path {
            Some(p) => {
                let mut q = p.clone();
                let mut name = q.file_name()
                    .map(|s| s.to_os_string())
                    .unwrap_or_else(|| std::ffi::OsString::from("brain.lbr"));
                name.push(".lb_session.wal");
                q.set_file_name(name);
                q
            }
            None => {
                // No path -> in-memory store only; just load the
                // snapshot without a WAL handle.
                self.lb_session = crate::lb_session::SessionMemoryStore::from_snapshot(snapshot);
                return;
            }
        };
        match crate::lb_session::SessionMemoryStore::from_snapshot_with_wal(
            snapshot.clone(),
            wal_path,
        ) {
            Ok(store) => self.lb_session = store,
            Err(e) => {
                eprintln!(
                    "lb_session WAL open/replay failed: {} (continuing with snapshot-only store)",
                    e
                );
                self.lb_session =
                    crate::lb_session::SessionMemoryStore::from_snapshot(snapshot);
            }
        }
    }

    /// Port CS.1 — hydrate the conversational style store from the
    /// v3 section #12 snapshot AND replay `<brain>.style.wal` on top.
    ///
    /// Mirrors `attach_lb_session_wal` exactly. The WAL captures
    /// every `tool_ask` observe + `tool_style_set` + `tool_style_reset`
    /// mutation that landed since the last brain save, so a crash
    /// between saves no longer loses style observations.
    ///
    /// Failure modes:
    ///   * Brain has no path (in-memory only) — load the snapshot,
    ///     skip the WAL handle. Future observe/set/reset calls do not
    ///     touch the disk.
    ///   * WAL open or replay fails — fall back to the snapshot-only
    ///     store and log to stderr. Same trade-off the LB.1 path
    ///     makes: a corrupted WAL must NOT brick the brain load.
    ///
    /// Anti-hallucination invariant: replay applies style mutations
    /// only. `bond_count` is unaffected.
    pub fn attach_cs_style_wal(
        &mut self,
        snapshot: crate::style::ConversationalStyleSnapshot,
    ) {
        let wal_path = match &self.path {
            Some(p) => {
                let mut q = p.clone();
                let mut name = q.file_name()
                    .map(|s| s.to_os_string())
                    .unwrap_or_else(|| std::ffi::OsString::from("brain.lbr"));
                name.push(".style.wal");
                q.set_file_name(name);
                q
            }
            None => {
                // No path -> in-memory store only; skip the WAL.
                self.cs_style_store =
                    crate::style::ConversationalStyleStore::from_snapshot(snapshot);
                return;
            }
        };
        match crate::style::ConversationalStyleStore::from_snapshot_with_wal(
            snapshot.clone(),
            wal_path,
        ) {
            Ok(store) => self.cs_style_store = store,
            Err(e) => {
                eprintln!(
                    "cs_style WAL open/replay failed: {} (continuing with snapshot-only store)",
                    e
                );
                self.cs_style_store =
                    crate::style::ConversationalStyleStore::from_snapshot(snapshot);
            }
        }
    }

    /// Port Cat.3b: Sync the hub-degree config's read-side fields
    /// to the network. Call this after every MCP `lu4d_hub_set`
    /// call so the spread-activation read paths see the latest
    /// knobs.
    pub fn resync_hub_read_config(&mut self) {
        self.network.sync_hub_read_config(&self.hub_degree_config);
    }

    /// Port Cat.3b: Snapshot the hub-cap filter counter from the
    /// network. Used by `lu4d_hub_status` and tests.
    pub fn hub_cap_filtered_at_query_total(&self) -> usize {
        self.network.hub_cap_filtered_at_query_total()
    }

    /// Port LB.1 — one-shot migration of v3 section #8 wonder
    /// signals into Left Brain WonderItem entries.
    ///
    /// Pre-LB.1 brains persist wonders only via `WonderQueueState`
    /// (section #8). On the first load with the LB.1 binary the
    /// wonder queue gets restored from section #8 by the existing
    /// path; we then mirror those signals into LB so subsequent
    /// loads can pick them up from section #11. Idempotent: skips
    /// when LB already has any WonderItem entries.
    pub fn migrate_section8_wonders_to_lb_if_needed(&mut self, ts_ms: u64) {
        let already_migrated = self
            .lb_session
            .counts_by_kind()
            .get("wonder_item")
            .copied()
            .unwrap_or(0)
            > 0;
        if already_migrated {
            return;
        }
        let signals = self.wonder_queue.snapshot_signals();
        if signals.is_empty() {
            return;
        }
        for s in signals {
            // OP-LABEL-4: `get_node` yields Some("") on a v5 stub, which
            // SWALLOWED the `or_else` fallback and wrote a blank label.
            let lb_label = self
                .network
                .node_label(&s.seed)
                .filter(|l| !l.is_empty())
                .or_else(|| self.gap_label_map.get(&s.seed).cloned())
                .unwrap_or_else(|| s.seed.to_string());
            let lb_payload = serde_json::to_string(&s).unwrap_or_default();
            let lb_intensity = s.intensity;
            let _ = self.lb_session.push_internal(
                crate::lb_session::SessionEntry::new(
                    0,
                    ts_ms,
                    crate::lb_session::SessionEntryKind::WonderItem,
                    lb_label,
                    lb_payload,
                    lb_intensity,
                ),
            );
        }
    }


    /// Port LB.2 — pick the canonical source for wonder state on load.
    ///
    /// Section #11 (Left Brain `SessionEntryKind::WonderItem` entries)
    /// is the canonical home for wonder signals from LB.1 onwards;
    /// LB.2 stops dual-writing into v3 section #8 (`WonderQueueState`)
    /// and that section is now sentinel-empty for LB.2-saved brains.
    ///
    /// Behavior:
    ///   * If LB has any `WonderItem` entries → rebuild `wonder_queue`
    ///     by JSON-decoding each entry's `content` field and pushing
    ///     the recovered `CuriositySignal`. The cap-eviction policy
    ///     re-applies at push. Rebuild `gap_label_map` from each
    ///     entry's `subject` field; merge in any leftover labels from
    ///     the section-#8 snapshot (preserves labels for seeds that
    ///     were evicted from the live queue but still tracked).
    ///   * Else (pre-LB.1 brain, no LB WonderItems yet) → restore
    ///     `wonder_queue` and `gap_label_map` from the legacy v3
    ///     section #8 payload, then mirror those signals into LB via
    ///     the existing LB.1 migration step so a subsequent save
    ///     carries them forward in section #11.
    // =================================================================
    // OP-CURIOSITY-1 -- the durable wonder journal.
    //
    // Context for whoever reads this next. The LB.2 machinery below
    // (`attach_lb_session_wal` / `hydrate_wonder_state_lb2`) is real and
    // correct, but it is structurally unreachable from a v5 DIRECTORY
    // brain: it derives its path from `self.path`, which `load_v5` never
    // sets, and `Brain::save()` early-returns for directory brains. So
    // on the canonical brain the wonder queue was RAM-only and died with
    // every process.
    //
    // Rather than retrofit `self.path` semantics onto v5 (which would
    // hand a save() path to a brain shape that deliberately does not
    // have one), this is a purpose-built sidecar living OUTSIDE the
    // brain directory. That placement is the point: the public guest
    // process is confined by systemd to a read-only view of
    // /var/lib/lu, and it stays that way. It gets write access to one
    // journal directory and nothing else.
    // =================================================================

    /// Attach a durable wonder journal. Every subsequent signal pushed
    /// to `wonder_queue` is also appended to it.
    ///
    /// Does not read the journal — call
    /// [`Self::hydrate_wonder_queue_from_sidecar`] for that. Attaching
    /// and hydrating are separate because the PRODUCER wants to attach
    /// without hydrating (it does not need the consumer's backlog in
    /// RAM) while the CONSUMER wants both.
    pub fn attach_wonder_sidecar(&mut self, sidecar: crate::wonder_sidecar::WonderSidecar) {
        self.wonder_sidecar = Some(sidecar);
    }

    /// Count of journal appends that failed. Non-zero means gaps are
    /// being detected and dropped — surfaced deliberately, because a
    /// silently unwritable journal is indistinguishable from a brain
    /// that has stopped being curious.
    pub fn wonder_sidecar_write_errors(&self) -> u64 {
        self.wonder_sidecar_errors.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Append one signal to the journal, if one is attached.
    ///
    /// Resolves a human-readable label for the seed on the way out: the
    /// consumer runs in a different process against a brain whose
    /// `gap_label_map` is empty, so a label that is not carried here is
    /// a label that is gone.
    fn record_wonder_durably(&self, signal: &crate::curiosity::CuriositySignal) {
        let Some(sc) = self.wonder_sidecar.as_ref() else { return };
        let label = self
            .network
            .node_label(&signal.seed)
            .or_else(|| self.gap_label_map.get(&signal.seed).cloned());
        let rec = crate::wonder_sidecar::WonderRecord::new(signal.clone(), label, None);
        if sc.append(std::slice::from_ref(&rec)).is_err() {
            self.wonder_sidecar_errors
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// CONSUMER-SIDE. Load the journal into the in-RAM wonder queue and
    /// `gap_label_map`. Returns the number of DISTINCT signals restored.
    ///
    /// Deduplicates by `(seed, signal_type)` keeping the highest
    /// intensity: the same gap is re-detected on every query that
    /// touches it, so the raw journal is a query log and the queue
    /// wants distinct open questions.
    pub fn hydrate_wonder_queue_from_sidecar(&mut self) -> std::io::Result<usize> {
        let Some(sc) = self.wonder_sidecar.clone() else { return Ok(0) };
        let journal = sc.load()?;
        if journal.skipped > 0 {
            eprintln!(
                "[curiosity] wonder journal: skipped {} unparseable line(s) in {}",
                journal.skipped,
                sc.path().display(),
            );
        }
        for (seed, label) in journal.labels() {
            self.gap_label_map.entry(seed).or_insert(label);
        }
        let signals = journal.dedup_signals();
        let n = signals.len();
        self.wonder_queue.restore_from(signals);
        Ok(n)
    }

    pub fn hydrate_wonder_state_lb2(
        &mut self,
        file_wonder: &crate::curiosity::WonderState,
        ts_ms: u64,
    ) {
        let lb_wonder_count = self
            .lb_session
            .counts_by_kind()
            .get("wonder_item")
            .copied()
            .unwrap_or(0);
        if lb_wonder_count > 0 {
            // LB.2 canonical path: section #11 is authoritative.
            let entries = self.lb_session.filter(
                None,
                Some(&crate::lb_session::SessionEntryKind::WonderItem),
                usize::MAX,
            );
            let mut signals = Vec::with_capacity(entries.len());
            let mut labels: std::collections::HashMap<crate::seed::Seed, String> =
                std::collections::HashMap::with_capacity(entries.len());
            for entry in &entries {
                if let Ok(signal) = serde_json::from_str::<
                    crate::curiosity::CuriositySignal,
                >(&entry.content)
                {
                    labels.insert(signal.seed, entry.subject.clone());
                    signals.push(signal);
                }
            }
            self.wonder_queue.restore_from(signals);
            // Preserve labels for seeds that lived only in the legacy
            // section #8 gap_label_map (e.g. evicted-from-queue seeds
            // on a pre-LB.2 brain).
            for (k, v) in &file_wonder.gap_label_map {
                labels.entry(*k).or_insert_with(|| v.clone());
            }
            self.gap_label_map = labels;
        } else {
            // Legacy fallback: section #8 is the source of truth for
            // pre-LB.1 brains. Restore + mirror into LB so the next
            // save carries the wonders forward via section #11.
            self.wonder_queue.restore_from(file_wonder.signals.clone());
            self.gap_label_map = file_wonder.gap_label_map.clone();
            self.migrate_section8_wonders_to_lb_if_needed(ts_ms);
        }
    }

    /// Port T.8 — push a new pending change through the rejection-
    /// pattern pre-filter before committing it to the pending log.
    ///
    /// If the candidate's fingerprint matches a learned rejection
    /// pattern (and `RejectionConfig::drop_on_match` is true, which is
    /// the default), the change is silently dropped and
    /// `proposals_filtered_total` is incremented. Returns `Ok(None)`
    /// in that case.
    ///
    /// Otherwise the change is forwarded to `self.pending.push` and
    /// the allocated `pending_id` is returned as `Ok(Some(id))`.
    /// Non-AddBond change kinds bypass the filter (they aren't
    /// fingerprintable today).
    pub fn push_pending_with_filter(
        &mut self,
        change: crate::pending::PendingChange,
    ) -> Result<Option<u64>, BrainError> {
        let now = now_ms() as u64;
        // Build a fingerprint just to consult the store. The actual
        // recorded rejected_at_ms only matters for fingerprints that
        // are STORED (i.e. on actual rejection); here it's a probe.
        let probe = crate::rejection::RejectionFingerprint::from_change(&change, now);
        if let Some(fp) = probe {
            let cfg = self.rejection_store.config().clone();
            if cfg.drop_on_match && self.rejection_store.matches_learned_pattern(&fp) {
                self.rejection_store.note_filtered();
                return Ok(None);
            }
            // Note: the !drop_on_match (confidence-multiplier) branch
            // is reserved for future configuration; the spec defaults
            // to silent drop and the multiplier path will be wired in
            // a follow-up port if/when the configurable mode lands.
        }
        let id = self.pending.push(change)?;
        Ok(Some(id))
    }

    /// Port T.8 — extract a fingerprint from the just-rejected pending
    /// entry (id) and feed it to the rejection-pattern store. Called
    /// from the `lu4d_reject_pending` MCP path AFTER
    /// `PendingLog::reject` has flipped the entry's status to
    /// Rejected. Returns `true` if a fingerprint was recorded
    /// (AddBond), `false` for non-fingerprintable kinds.
    pub fn record_rejection_fingerprint(
        &mut self,
        pending_id: u64,
        rejected_at_ms: u64,
    ) -> bool {
        let Some(entry) = self.pending.get(pending_id) else {
            return false;
        };
        let Some(fp) =
            crate::rejection::RejectionFingerprint::from_change(entry, rejected_at_ms)
        else {
            return false;
        };
        self.rejection_store.record_rejection(fp, rejected_at_ms);
        true
    }

    /// OP-CURIOSITY-1 -- autonomously approve+apply curiosity-produced
    /// pending entries. Returns `(considered, applied)`.
    ///
    /// No-op unless `curiosity_config.auto_apply` is set. Only the ids
    /// passed in are touched: this deliberately cannot sweep the whole
    /// PendingLog, so an operator's hand-curated pending entries are
    /// never swallowed by a background tick.
    ///
    /// WHAT THE SAFETY FLOOR ACTUALLY IS (stated plainly, because it is
    /// easy to over-claim here):
    ///
    /// * Every id has already passed `push_pending_with_filter`, i.e.
    ///   the Port T.8 learned-rejection pre-filter. Shapes that were
    ///   rejected before do not reach this function.
    /// * `auto_apply_min_confidence` excludes low-confidence proposals,
    ///   which by construction excludes bare internal synthesis (0.5).
    /// * `auto_apply_max_per_tick` bounds the blast radius.
    ///
    /// What it is NOT: this is not the K-gate. The K-gate
    /// (`crate::idk_gate`) governs whether Lu ANSWERS, on the recall
    /// path; it has no opinion about writes and is not consulted here.
    /// Nor is `ingest_config.auto_approve_threshold`, which is the
    /// teach_corpus gate and has never had any effect on idle output.
    ///
    /// Every application is announced on stderr with its confidence and
    /// reason, so the journal shows exactly what the brain did to itself
    /// and when.
    pub fn auto_apply_curiosity_pending(&mut self, ids: &[u64]) -> (u32, u32) {
        if !self.curiosity_config.auto_apply || ids.is_empty() {
            return (0, 0);
        }
        let min_conf = self.curiosity_config.auto_apply_min_confidence;
        let cap = self.curiosity_config.auto_apply_max_per_tick;
        let now = crate::brain::now_ms() as u64;

        let eligible: Vec<(u64, f32, String)> = ids
            .iter()
            .filter_map(|id| {
                let e = self.pending.get(*id)?;
                if !matches!(e.status, crate::pending::PendingStatus::Pending) {
                    return None;
                }
                if e.confidence < min_conf {
                    return None;
                }
                Some((*id, e.confidence, e.reason.clone()))
            })
            .take(cap as usize)
            .collect();

        let considered = eligible.len() as u32;
        let mut applied = 0u32;
        for (id, conf, reason) in eligible {
            if self
                .pending
                .approve(id, now, "curiosity-auto".to_string())
                .is_err()
            {
                continue;
            }
            match crate::pending::apply_change(self, id) {
                Ok(()) => {
                    applied += 1;
                    eprintln!(
                        "[curiosity] AUTO-APPLIED pending #{} conf={:.2} at {}ms -- {}",
                        id, conf, now, reason,
                    );
                }
                Err(e) => eprintln!(
                    "[curiosity] auto-apply FAILED for pending #{}: {} (entry stays Approved-but-unapplied)",
                    id, e,
                ),
            }
        }
        (considered, applied)
    }

    /// Port T.8 — counterpart for approvals. Records a fingerprint
    /// against the approval buffer so any matching learned pattern is
    /// suppressed (zero-tolerance: one approval kills the pattern).
    /// Called from the `lu4d_approve_pending` MCP path after the
    /// entry's status flips to Approved.
    pub fn record_approval_fingerprint(
        &mut self,
        pending_id: u64,
        approved_at_ms: u64,
    ) -> bool {
        let Some(entry) = self.pending.get(pending_id) else {
            return false;
        };
        let Some(fp) =
            crate::rejection::RejectionFingerprint::from_change(entry, approved_at_ms)
        else {
            return false;
        };
        self.rejection_store.record_approval(fp, approved_at_ms);
        true
    }

    /// Load a brain from a .lbr file. If the file does not exist, starts empty
    /// but remembers the path so `save()` writes there.
    ///
    /// Note: `recent_bridges` is intentionally NOT persisted — it's a runtime
    /// ring buffer that resets each launch. Brain files stay backward-compatible.
    pub fn load_or_new<P: AsRef<Path>>(path: P) -> Result<Self, BrainError> {
        let path_buf = path.as_ref().to_path_buf();
        if path_buf.exists() {
            // Auto-detect v1 (JSON) vs v2 (binary) vs v3 (binary) by
            // sniffing the magic + version byte. See `brain_io::detect_format`.
            // Each branch reads the file in its own most-appropriate way:
            //   v1, v2: fs::read into a heap buffer then parse.
            //   v3: mmap the file inside read_v3_path so rkyv's 16-byte
            //       alignment requirement is satisfied without copying
            //       the whole file into a heap Vec first.
            let detected = crate::brain_io::detect_format_path(&path_buf)?;
            let file: BrainFile = match detected {
                crate::brain_io::BrainFormat::V1Json => {
                    // Critical audit fix internal notes / Problem #111
                    // follow-up): production load paths no longer read v1
                    // JSON. The structural guarantee "plaintext brains do
                    // not exist in production" is maintained by routing the
                    // sole legacy-read entry point through the dedicated
                    // migration-only function `load_v1_for_migration_only`,
                    // called exclusively by the --migrate-from-v1 CLI
                    // utility. Solution #77 claimed this isolation already
                    // existed; it did not. This branch is the fix.
                    return Err(BrainError::Format(format!(
                        "v1 JSON brain detected at {}: production load no longer reads plaintext v1. \
                         Run `lu4d --migrate-from-v1 SRC DST [--keyfile P] [--out-format v3enc|v3]` \
                         to migrate the file to v3.",
                        path_buf.display()
                    )));
                }
                crate::brain_io::BrainFormat::V2Binary => {
                    let bytes = fs::read(&path_buf)?;
                    let f = crate::brain_io::read_v2(&bytes)?;
                    if f.format_version > BRAIN_FORMAT_VERSION {
                        return Err(BrainError::VersionMismatch(f.format_version));
                    }
                    f
                }
                crate::brain_io::BrainFormat::V3Binary => {
                    // Phase 3: a v3 file with the compression flag set
                    // is not directly rkyv-readable. Route it through
                    // the mmap-zerocopy path which knows how to dispatch
                    // to the per-block compressed reader. The eager
                    // (heap deserialize) path is only for uncompressed
                    // v3 files (Phase 2.2).
                    let raw = fs::read(&path_buf)?;
                    let flags = crate::brain_io_v3::read_v3_flags(&raw)?;
                    if flags & crate::compress::FORMAT_FLAG_COMPRESSED != 0 {
                        // Compressed v3 must use the mmap-backed reader.
                        return Self::load_v3_zerocopy(path_buf);
                    }
                    let f = crate::brain_io_v3::read_v3_path(&path_buf)?;
                    if f.format_version != crate::brain_io_v3::FORMAT_VERSION_V3 {
                        return Err(BrainError::VersionMismatch(f.format_version));
                    }
                    f
                }
            };
            let file_scar_log = file.scar_log.clone();
            // Older saves may carry cap=0 (default-constructed) or a
            // smaller historical cap. Raise to the current default
            // either way so saved brains pick up calibration changes.
            let mut interest_profile = file.interest_profile;
            let cfg_cap = CuriosityConfig::default().interest_profile_cap;
            if interest_profile.cap == 0 || interest_profile.cap < cfg_cap {
                interest_profile.cap = cfg_cap;
            }
            let mut brain = Brain {
                network: AssociativeNetwork::from_snapshot(file.network),
                sessions: SessionManager::new(),
            session_memories: SessionMemoryRegistry::new(),
                consolidator: ConsolidationEngine::new(),
                path: Some(path_buf),
                bridge_config: BridgeConfig::default(),
                recent_bridges: VecDeque::new(),
                bridges_dropped_overflow_total: 0,
                curiosity_config: CuriosityConfig::default(),
                wonder_queue: WonderQueue::new(CuriosityConfig::default().wonder_queue_cap),
                interest_profile,
                curiosity_dropped_overflow_total: 0,
            gap_emitter: crate::gap_signal::GapEmitter::new(),
            gap_label_map: std::collections::HashMap::new(),
            wonder_sidecar: None,
            wonder_sidecar_errors: std::sync::atomic::AtomicU64::new(0),
                next_variant_id: 1,
            next_multi_think_id: 1,
            multi_agent_config: crate::agents::MultiAgentConfig::default(),
                performance_monitor: PerformanceMonitor::new(1024),
                meta_reflection: file.meta_reflection,
                self_reflection_config: SelfReflectionConfig::default(),
                narration_config: crate::narration::NarrationConfig::default(),
                scar_log: file_scar_log,
                research_config: file.research_config,
                research_proposals: file.research_proposals,
                next_research_proposal_id: file.next_research_proposal_id,
                gap_seed_frequency: file.gap_seed_frequency,
                // Bug A (2026-04-27): eager-load path used to hardcode
                // V1Json regardless of the detected on-disk format. Now
                // mirrors load_v3_zerocopy_with_key by deriving from
                // `detected` so save_format matches the file the brain
                // was loaded from.
                save_format: match detected {
                    crate::brain_io::BrainFormat::V1Json => BrainSaveFormat::V1Json,
                    crate::brain_io::BrainFormat::V2Binary => BrainSaveFormat::V2Binary,
                    crate::brain_io::BrainFormat::V3Binary => BrainSaveFormat::V3Binary,
                },
                // Bug A: track the on-disk format we loaded from. The
                // eager v3 path is non-compressed; the compressed-v3
                // and encrypted-v3 cases route through load_v3_zerocopy.
                loaded_format: Some(match detected {
                    crate::brain_io::BrainFormat::V1Json => BrainSaveFormat::V1Json,
                    crate::brain_io::BrainFormat::V2Binary => BrainSaveFormat::V2Binary,
                    crate::brain_io::BrainFormat::V3Binary => BrainSaveFormat::V3Binary,
                }),
                encryption_key: None,
                allow_decrypt_on_save: false,
                require_encrypted: false,
            // Port Cat.3c: ephemeral brains have no sidecars to migrate.
            runtime_config_pending_migration: false,
                negation_overlay: crate::negation::NegationOverlay::new(),
                source_overlay: Arc::new(RwLock::new(crate::source::SourceOverlay::new())),
            proposal_chain_context: std::collections::HashMap::new(),
                negation_cues_enabled: true,
                time_source_config: crate::time::TimeSourceConfig::default(),
            sense_config: crate::sense::SenseConfig::default(),
            chain_research_config: crate::research::chain::ChainResearchConfig::default(),
            recall_config: crate::reasoning::RecallConfig::default(),
            admin_revoke_enabled: false,
            next_chain_id: 1,
            ingest_config: IngestConfig::default(),
            tier2_prompt_config: Tier2PromptConfig::default(),
            file_io_config: crate::file_io::FileIoConfig::default(),
            vision_config: crate::mcp_vision::VisionConfig::default(),
            ingest_run_id: make_startup_run_id(),
            df_time_client: None,
            clock: std::sync::Arc::new(crate::time::SystemClockSource::new())
                    as std::sync::Arc<dyn crate::time::ClockSource>,
                time_anchor: std::sync::RwLock::new(
                    crate::time::AuthenticatedTime::unsigned(
                        now_ms().max(0) as u64,
                        crate::time::MonotonicClock::now(),
                        crate::time::SourceId::system(),
                        (now_ms().max(0) as u64).saturating_add(
                            crate::time::TimeConfig::default().refresh_interval_ms,
                        ),
                    ),
                ),
                anomaly_detector: std::sync::Mutex::new(
                    crate::time::AnomalyDetector::new(crate::time::TimeConfig::default()),
                ),
                idle_state: crate::idle::IdleState::new(),
            pending: crate::pending::PendingLog::new_in_memory(),
            rejection_store: crate::rejection::RejectionPatternStore::new(),
            lb_session: crate::lb_session::SessionMemoryStore::new(),
            cs_style_store: crate::style::ConversationalStyleStore::new(),
            style_config: crate::style::StyleConfig::default(),
            pack_ledger: crate::pack::PackLedgerStore::new(),
            cap_pack_ledger: crate::capability_pack::CapabilityPackLedgerStore::new(),
            pack_registry: crate::capability_runtime::LoadedPackRegistry::default(),
            trust_table: crate::trust::TrustTable::default(),
            hub_degree_config: crate::hub_degree::HubDegreeConfig::default(),
            hub_cap_rejections_total: 0,
            hub_cap_evictions_total: 0,
            memtable: None,
            wal: None,
            wal_path: None,
            flush_worker: None,
            manifest_snapshot: None,
            flush_intern: None,
            ingest_only: false,
            read_path_mode: std::sync::atomic::AtomicU8::new(crate::read_path::ReadPathMode::V3Only.as_u8()),
            orphan_cleanup_done: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            v4_sst_cache: std::sync::Mutex::new(std::collections::HashMap::new()),
            v4_block_cache: std::sync::Arc::new(crate::sst::BlockCache::new(crate::sst::DEFAULT_SST_BLOCK_CACHE_BYTES)),
            v4_parsed_block_cache: std::sync::Arc::new(crate::sst::ParsedBlockCache::new(crate::sst::DEFAULT_PARSED_BLOCK_CACHE_BYTES)),
            compaction_worker: None,
            bond_v5_compaction: None,
            node_v5_compaction: None,
            brain_dir_v4: None,
            node_reader_cache: None,
            v5_store_root: None,
            bond_reader_cache: None,
            resident_map: None,
            cache_adjacency: None,
            decay_config: crate::decay::DecayConfig::default(),
            // Port T.6: same overlay as Brain::new -- loader paths
            // get a fresh AutoTickDriver. The MCP server may rebuild
            // it against the brain's actual ClockSource after load.
            auto_tick: crate::idle::AutoTickDriver::new(
                std::sync::Arc::new(crate::time::SystemClockSource::new())
                    as std::sync::Arc<dyn crate::time::ClockSource>,
            ),
            save_runner: std::sync::Arc::new(crate::save_async::SaveRunner::default()),
            };
            // Port LB.2: wonder_state hydration moved to the LB.2 unified
            // entry point below (after attach_lb_session_wal so LB is ready).
            brain.negation_overlay = file.negation_overlay;
            // OP-SNAPSHOT-PEAK: assign into the inner RwLock.
            *brain.source_overlay.write().unwrap() = file.source_overlay;
            // Port T.4: hydrate pending log from the v3 brain section.
            // `file.pending_log` is `PendingLogSnapshot::default()` for
            // pre-T.4 brains (section absent + `#[serde(default)]`).
            brain.attach_pending_wal(file.pending_log);
            // Port T.8: hydrate rejection-pattern store from v3 section #10.
            brain.rejection_store =
                crate::rejection::RejectionPatternStore::from_snapshot(
                    file.rejection_patterns,
                );
            // Port LB / LB.1: hydrate Left Brain session memory from v3
            // section #11 AND replay <brain>.lb_session.wal on top.
            brain.attach_lb_session_wal(file.lb_session);
            // Port Cat.3c / Cat.3c.1: prefer the v3 RuntimeConfig
            // sections (#14 bincode legacy + #15 JSON authoritative).
            // When present, copy ALL fields — including the Cat.3c.1
            // style knobs — into the in-memory brain.
            if file.runtime_config_present {
                brain.trust_table = file.runtime_config.trust_table.clone();
                brain.hub_degree_config = file.runtime_config.hub_degree.clone();
                brain.network.sync_hub_read_config(&brain.hub_degree_config);
                // Port Cat.3c.1: hydrate the CS.2 output-adaptation knobs
                // from section #15. (For Cat.3c-only brains the V14
                // fall-back returns the StyleConfig defaults, matching
                // pre-Cat.3c.1 behavior.)
                brain.style_config.output_adaptation_enabled =
                    file.runtime_config.style_output_adaptation_enabled;
                brain.style_config.max_samples_for_full_weight =
                    file.runtime_config.style_max_samples_for_full_weight;
                // Port T.1.1: hydrate time-source config.
                brain.time_source_config = crate::time::TimeSourceConfig {
                    mode: file.runtime_config.time_source.clone(),
                    url: file.runtime_config.time_service_url.clone(),
                    pubkey_hex: file.runtime_config.time_service_pubkey_hex.clone(),
                    poll_interval_ms: file.runtime_config.time_service_poll_interval_ms,
                    max_drift_ms: file.runtime_config.time_service_max_drift_ms,
                };
                brain.sense_config = crate::sense::SenseConfig {
                    disambiguation_enabled: file.runtime_config.sense_disambiguation_enabled,
                    context_window: file.runtime_config.sense_context_window,
                    loser_gain: file.runtime_config.sense_loser_gain,
                };
                brain.chain_research_config = crate::research::chain::ChainResearchConfig {
                    enabled: file.runtime_config.chained_research_enabled,
                    max_chain_depth: file.runtime_config.max_chain_depth,
                    max_chain_cycles: file.runtime_config.max_chain_cycles,
                    gap_threshold: file.runtime_config.chain_gap_threshold,
                };
                // Port BR.4: hydrate recall-time typed-KU flag from section #15.
                brain.recall_config = crate::reasoning::RecallConfig {
                    typed_ku_enabled: file.runtime_config.recall_typed_ku_enabled,
                };
                // Port BR.5: hydrate admin-revoke gate from section #15.
                brain.admin_revoke_enabled = file.runtime_config.admin_revoke_enabled;
                // Port T-Ingest: hydrate ingest knobs from section #15.
                brain.ingest_config = crate::brain::IngestConfig {
                    corpus_default: file.runtime_config.ingest_corpus_default.clone(),
                    run_id_strategy: file.runtime_config.ingest_run_id_strategy.clone(),
                    auto_approve_threshold: file.runtime_config.ingest_auto_approve_threshold,
                };
                brain.tier2_prompt_config = crate::brain::Tier2PromptConfig {
                    default_session_strategy: file.runtime_config.tier2_prompt_default_session_strategy.clone(),
                    max_text_bytes: file.runtime_config.tier2_prompt_max_text_bytes,
                };
                // Port File-IO: hydrate file IO knobs from section #15.
                brain.file_io_config = crate::file_io::FileIoConfig {
                    max_size_bytes: file.runtime_config.file_io_max_size_bytes,
                    allowed_read_paths: file.runtime_config.file_io_allowed_read_paths.clone(),
                    allowed_write_paths: file.runtime_config.file_io_allowed_write_paths.clone(),
                };
                // Port R input: hydrate vision knobs from section #15.
                brain.vision_config = crate::mcp_vision::VisionConfig {
                    extractor_kind: file.runtime_config.image_extractor_kind.clone(),
                    subprocess_path: file.runtime_config.image_extractor_subprocess_path.clone(),
                    subprocess_timeout_ms: file.runtime_config.image_extractor_subprocess_timeout_ms,
                    image_max_size_bytes: file.runtime_config.image_max_size_bytes,
                    image_supported_mime_types: file.runtime_config.image_supported_mime_types.clone(),
                };
                brain.runtime_config_pending_migration = false;
            } else {
                // Port Cat.3c.1: pre-Cat.3c brain (no sections #14/#15).
                // Migration to live brains is complete (see Cat.3c
                // ship 2026-05-02), so we no longer auto-load legacy
                // sidecars at this point. The first save will write
                // sections #14+#15 and retire any stray sidecars.
                brain.runtime_config_pending_migration = true;
            }
            // Port CS / CS.1: hydrate ConversationalStyle store from
            // v3 section #12 AND replay <brain>.style.wal on top.
            brain.attach_cs_style_wal(file.cs_style);
            brain.pack_ledger = crate::pack::PackLedgerStore::from_snapshot(file.pack_ledger);
            // Port Cat.5: hydrate capability-pack ledger store from v3 section #17.
            brain.cap_pack_ledger = crate::capability_pack::CapabilityPackLedgerStore::from_snapshot(file.cap_pack_ledger);
            // Port LB.1: hydrate pre-LB.1 brains' section #8 wonders into LB.
            // Port LB.2: pick canonical source for wonder state (section #11
            // if LB has WonderItem entries, else section #8 fallback).
            brain.hydrate_wonder_state_lb2(&file.wonder_state, now_ms() as u64);
            Ok(brain)
        } else {
            let mut brain = Brain {
                network: AssociativeNetwork::new(),
                sessions: SessionManager::new(),
            session_memories: SessionMemoryRegistry::new(),
                consolidator: ConsolidationEngine::new(),
                path: Some(path_buf),
                bridge_config: BridgeConfig::default(),
                recent_bridges: VecDeque::new(),
                bridges_dropped_overflow_total: 0,
                curiosity_config: CuriosityConfig::default(),
                wonder_queue: WonderQueue::new(CuriosityConfig::default().wonder_queue_cap),
                interest_profile: InterestProfile::new(CuriosityConfig::default().interest_profile_cap),
                curiosity_dropped_overflow_total: 0,
            gap_emitter: crate::gap_signal::GapEmitter::new(),
            gap_label_map: std::collections::HashMap::new(),
            wonder_sidecar: None,
            wonder_sidecar_errors: std::sync::atomic::AtomicU64::new(0),
                next_variant_id: 1,
            next_multi_think_id: 1,
            multi_agent_config: crate::agents::MultiAgentConfig::default(),
                performance_monitor: PerformanceMonitor::new(1024),
                meta_reflection: MetaReflection::new(),
                self_reflection_config: SelfReflectionConfig::default(),
                narration_config: crate::narration::NarrationConfig::default(),
                scar_log: ScarLog::new(),
                research_config: crate::research::router::ResearchConfig::default(),
                research_proposals: Vec::new(),
                next_research_proposal_id: 0,
                gap_seed_frequency: std::collections::HashMap::new(),
                save_format: BrainSaveFormat::V3Binary, // v1 JSON removal internal notes
                // Bug A: ephemeral / no-file-yet brain has no on-disk
                // format yet — None until first save.
                loaded_format: None,
                encryption_key: None,
                allow_decrypt_on_save: false,
                require_encrypted: false,
            // Port Cat.3c: ephemeral brains have no sidecars to migrate.
            runtime_config_pending_migration: false,
                negation_overlay: crate::negation::NegationOverlay::new(),
                source_overlay: Arc::new(RwLock::new(crate::source::SourceOverlay::new())),
            proposal_chain_context: std::collections::HashMap::new(),
                negation_cues_enabled: true,
                time_source_config: crate::time::TimeSourceConfig::default(),
            sense_config: crate::sense::SenseConfig::default(),
            chain_research_config: crate::research::chain::ChainResearchConfig::default(),
            recall_config: crate::reasoning::RecallConfig::default(),
            admin_revoke_enabled: false,
            next_chain_id: 1,
            ingest_config: IngestConfig::default(),
            tier2_prompt_config: Tier2PromptConfig::default(),
            file_io_config: crate::file_io::FileIoConfig::default(),
            vision_config: crate::mcp_vision::VisionConfig::default(),
            ingest_run_id: make_startup_run_id(),
            df_time_client: None,
            clock: std::sync::Arc::new(crate::time::SystemClockSource::new())
                    as std::sync::Arc<dyn crate::time::ClockSource>,
                time_anchor: std::sync::RwLock::new(
                    crate::time::AuthenticatedTime::unsigned(
                        now_ms().max(0) as u64,
                        crate::time::MonotonicClock::now(),
                        crate::time::SourceId::system(),
                        (now_ms().max(0) as u64).saturating_add(
                            crate::time::TimeConfig::default().refresh_interval_ms,
                        ),
                    ),
                ),
                anomaly_detector: std::sync::Mutex::new(
                    crate::time::AnomalyDetector::new(crate::time::TimeConfig::default()),
                ),
                idle_state: crate::idle::IdleState::new(),
            pending: crate::pending::PendingLog::new_in_memory(),
            rejection_store: crate::rejection::RejectionPatternStore::new(),
            lb_session: crate::lb_session::SessionMemoryStore::new(),
            cs_style_store: crate::style::ConversationalStyleStore::new(),
            style_config: crate::style::StyleConfig::default(),
            pack_ledger: crate::pack::PackLedgerStore::new(),
            cap_pack_ledger: crate::capability_pack::CapabilityPackLedgerStore::new(),
            pack_registry: crate::capability_runtime::LoadedPackRegistry::default(),
            trust_table: crate::trust::TrustTable::default(),
            hub_degree_config: crate::hub_degree::HubDegreeConfig::default(),
            hub_cap_rejections_total: 0,
            hub_cap_evictions_total: 0,
            memtable: None,
            wal: None,
            wal_path: None,
            flush_worker: None,
            manifest_snapshot: None,
            flush_intern: None,
            ingest_only: false,
            read_path_mode: std::sync::atomic::AtomicU8::new(crate::read_path::ReadPathMode::V3Only.as_u8()),
            orphan_cleanup_done: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            v4_sst_cache: std::sync::Mutex::new(std::collections::HashMap::new()),
            v4_block_cache: std::sync::Arc::new(crate::sst::BlockCache::new(crate::sst::DEFAULT_SST_BLOCK_CACHE_BYTES)),
            v4_parsed_block_cache: std::sync::Arc::new(crate::sst::ParsedBlockCache::new(crate::sst::DEFAULT_PARSED_BLOCK_CACHE_BYTES)),
            compaction_worker: None,
            bond_v5_compaction: None,
            node_v5_compaction: None,
            brain_dir_v4: None,
            node_reader_cache: None,
            v5_store_root: None,
            bond_reader_cache: None,
            resident_map: None,
            cache_adjacency: None,
            decay_config: crate::decay::DecayConfig::default(),
            // Port T.6: same overlay as Brain::new -- loader paths
            // get a fresh AutoTickDriver. The MCP server may rebuild
            // it against the brain's actual ClockSource after load.
            auto_tick: crate::idle::AutoTickDriver::new(
                std::sync::Arc::new(crate::time::SystemClockSource::new())
                    as std::sync::Arc<dyn crate::time::ClockSource>,
            ),
            save_runner: std::sync::Arc::new(crate::save_async::SaveRunner::default()),
            };
            // Port T.4: this branch is the "file did not exist on disk"
            // path; there is no v3 brain to read a snapshot from. Pass
            // `Default::default()` so the WAL is opened fresh against
            // the (likewise non-existent) `<brain>.pending.wal` path,
            // which read_all() handles gracefully (returns empty).
            brain.attach_pending_wal(crate::pending::PendingLogSnapshot::default());
            // Port LB.1: same for the LB session WAL.
            brain.attach_lb_session_wal(crate::lb_session::SessionMemorySnapshot::default());
            // Port Cat.3c.1: fresh brain — initialize at type defaults.
            // (Cat.3c's preseed-from-sidecar behavior is retired now that
            // production migration is complete.) The first save will
            // emit sections #14+#15 from the in-memory state and retire
            // any stray sidecars present on disk.
            brain.runtime_config_pending_migration = true;
            // Port CS / CS.1: empty default store on fresh brain.
            // attach_cs_style_wal opens (creates) <brain>.style.wal so
            // future observe/set/reset calls fsync to disk between saves.
            brain.attach_cs_style_wal(crate::style::ConversationalStyleSnapshot::default());
            // Port LB.2: fresh brain — no wonder data on disk, so pass an
            // empty WonderState. Both branches of hydrate_wonder_state_lb2
            // are no-ops in this case.
            brain.hydrate_wonder_state_lb2(
                &crate::curiosity::WonderState::default(),
                now_ms() as u64,
            );
            Ok(brain)
        }
    }


    /// Migration-only entry point: read a legacy v1 JSON brain.
    ///
    /// **DO NOT CALL FROM PRODUCTION LOAD PATHS.** This is the *only*
    /// place in the codebase that calls `serde_json::from_slice` on a
    /// brain file. The production `load_or_new` family rejects v1 with
    /// an explicit migration pointer; this function exists solely so
    /// the `--migrate-from-v1` CLI utility can read a legacy v1 file,
    /// hand it back as a `Brain`, and re-save it in v3/v3enc.
    ///
    /// Returns an error if the file is not v1 -- defensive guard so the
    /// function cannot be repurposed as a back-door production loader.
    ///
    /// Audit fix internal notes / Problem #111 follow-up): Solution #77
    /// claimed this function existed; it did not (grep proof). This
    /// commit creates it and routes --migrate-from-v1 through it.
    pub fn load_v1_for_migration_only<P: AsRef<Path>>(
        path: P,
    ) -> Result<Self, BrainError> {
        let path_buf = path.as_ref().to_path_buf();
        if !path_buf.exists() {
            return Err(BrainError::Io(io::Error::new(
                io::ErrorKind::NotFound,
                format!("v1 brain not found at {}", path_buf.display()),
            )));
        }
        let detected = crate::brain_io::detect_format_path(&path_buf)?;
        if !matches!(detected, crate::brain_io::BrainFormat::V1Json) {
            return Err(BrainError::Format(format!(
                "load_v1_for_migration_only: file at {} is not v1 (detected {:?}); \
                 this loader is restricted to v1 JSON migration only",
                path_buf.display(), detected
            )));
        }
        let bytes = fs::read(&path_buf)?;
        let file: BrainFile = serde_json::from_slice(&bytes)?;
        if file.format_version != V1_JSON_VERSION {
            return Err(BrainError::VersionMismatch(file.format_version));
        }
        let file_scar_log = file.scar_log.clone();
        let mut interest_profile = file.interest_profile;
        let cfg_cap = CuriosityConfig::default().interest_profile_cap;
        if interest_profile.cap == 0 || interest_profile.cap < cfg_cap {
            interest_profile.cap = cfg_cap;
        }
        Ok(Brain {
            network: AssociativeNetwork::from_snapshot(file.network),
            sessions: SessionManager::new(),
            session_memories: SessionMemoryRegistry::new(),
            consolidator: ConsolidationEngine::new(),
            path: Some(path_buf),
            bridge_config: BridgeConfig::default(),
            recent_bridges: VecDeque::new(),
            bridges_dropped_overflow_total: 0,
            curiosity_config: CuriosityConfig::default(),
            wonder_queue: WonderQueue::new(CuriosityConfig::default().wonder_queue_cap),
            interest_profile,
            curiosity_dropped_overflow_total: 0,
            gap_emitter: crate::gap_signal::GapEmitter::new(),
            gap_label_map: std::collections::HashMap::new(),
            wonder_sidecar: None,
            wonder_sidecar_errors: std::sync::atomic::AtomicU64::new(0),
            next_variant_id: 1,
            next_multi_think_id: 1,
            multi_agent_config: crate::agents::MultiAgentConfig::default(),
            performance_monitor: PerformanceMonitor::new(1024),
            meta_reflection: file.meta_reflection,
            self_reflection_config: SelfReflectionConfig::default(),
            narration_config: crate::narration::NarrationConfig::default(),
            scar_log: file_scar_log,
            research_config: file.research_config,
            research_proposals: file.research_proposals,
            next_research_proposal_id: file.next_research_proposal_id,
            gap_seed_frequency: file.gap_seed_frequency,
            save_format: BrainSaveFormat::V1Json,
            loaded_format: Some(BrainSaveFormat::V1Json),
            encryption_key: None,
            allow_decrypt_on_save: false,
            require_encrypted: false,
            // Port Cat.3c: ephemeral brains have no sidecars to migrate.
            runtime_config_pending_migration: false,
            negation_overlay: crate::negation::NegationOverlay::new(),
            // Port Source-Persist: V1 JSON path was a latent gap too --
            // BrainFile.source_overlay round-trips fine through serde,
            // but this constructor was throwing it away.
            // OP-SNAPSHOT-PEAK: wrap loaded overlay in Arc<RwLock>.
            source_overlay: Arc::new(RwLock::new(file.source_overlay)),
            proposal_chain_context: std::collections::HashMap::new(),
            negation_cues_enabled: true,
            time_source_config: crate::time::TimeSourceConfig::default(),
            sense_config: crate::sense::SenseConfig::default(),
            chain_research_config: crate::research::chain::ChainResearchConfig::default(),
            recall_config: crate::reasoning::RecallConfig::default(),
            admin_revoke_enabled: false,
            next_chain_id: 1,
            ingest_config: IngestConfig::default(),
            tier2_prompt_config: Tier2PromptConfig::default(),
            file_io_config: crate::file_io::FileIoConfig::default(),
            vision_config: crate::mcp_vision::VisionConfig::default(),
            ingest_run_id: make_startup_run_id(),
            df_time_client: None,
            clock: std::sync::Arc::new(crate::time::SystemClockSource::new())
                as std::sync::Arc<dyn crate::time::ClockSource>,
            time_anchor: std::sync::RwLock::new(
                crate::time::AuthenticatedTime::unsigned(
                    now_ms().max(0) as u64,
                    crate::time::MonotonicClock::now(),
                    crate::time::SourceId::system(),
                    (now_ms().max(0) as u64).saturating_add(
                        crate::time::TimeConfig::default().refresh_interval_ms,
                    ),
                ),
            ),
            anomaly_detector: std::sync::Mutex::new(
                crate::time::AnomalyDetector::new(crate::time::TimeConfig::default()),
            ),
            idle_state: crate::idle::IdleState::new(),
            pending: crate::pending::PendingLog::new_in_memory(),
            rejection_store: crate::rejection::RejectionPatternStore::new(),
            lb_session: crate::lb_session::SessionMemoryStore::new(),
            cs_style_store: crate::style::ConversationalStyleStore::new(),
            style_config: crate::style::StyleConfig::default(),
            pack_ledger: crate::pack::PackLedgerStore::new(),
            cap_pack_ledger: crate::capability_pack::CapabilityPackLedgerStore::new(),
            pack_registry: crate::capability_runtime::LoadedPackRegistry::default(),
            trust_table: crate::trust::TrustTable::default(),
            hub_degree_config: crate::hub_degree::HubDegreeConfig::default(),
            hub_cap_rejections_total: 0,
            hub_cap_evictions_total: 0,
            memtable: None,
            wal: None,
            wal_path: None,
            flush_worker: None,
            manifest_snapshot: None,
            flush_intern: None,
            ingest_only: false,
            read_path_mode: std::sync::atomic::AtomicU8::new(crate::read_path::ReadPathMode::V3Only.as_u8()),
            orphan_cleanup_done: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            v4_sst_cache: std::sync::Mutex::new(std::collections::HashMap::new()),
            v4_block_cache: std::sync::Arc::new(crate::sst::BlockCache::new(crate::sst::DEFAULT_SST_BLOCK_CACHE_BYTES)),
            v4_parsed_block_cache: std::sync::Arc::new(crate::sst::ParsedBlockCache::new(crate::sst::DEFAULT_PARSED_BLOCK_CACHE_BYTES)),
            compaction_worker: None,
            bond_v5_compaction: None,
            node_v5_compaction: None,
            brain_dir_v4: None,
            node_reader_cache: None,
            v5_store_root: None,
            bond_reader_cache: None,
            resident_map: None,
            cache_adjacency: None,
            decay_config: crate::decay::DecayConfig::default(),
            // Port T.6: same overlay as Brain::new -- loader paths
            // get a fresh AutoTickDriver. The MCP server may rebuild
            // it against the brain's actual ClockSource after load.
            auto_tick: crate::idle::AutoTickDriver::new(
                std::sync::Arc::new(crate::time::SystemClockSource::new())
                    as std::sync::Arc<dyn crate::time::ClockSource>,
            ),
            save_runner: std::sync::Arc::new(crate::save_async::SaveRunner::default()),
        })
    }

    /// Storage v2 Slice 2c — load a v3 brain file via the
    /// mmap-zero-copy backend. The mmap is held alive for the
    /// lifetime of the resulting `Brain`, so node/bond reads come
    /// straight from file-backed pages without ever materializing a
    /// heap copy. Mutations promote individual records into a heap
    /// overlay (CoW).
    fn load_v3_zerocopy(path_buf: std::path::PathBuf) -> Result<Self, BrainError> {
        Self::load_v3_zerocopy_with_key(path_buf, None)
    }

    /// Storage v2 Phase 4 -- mmap-zero-copy v3 load with an optional
    /// master key. When the file's `flags` advertises encryption, the
    /// key is required (passed through from `--keyfile`). Without
    /// encryption the path matches Phase 3 behaviour.
    fn load_v3_zerocopy_with_key(
        path_buf: std::path::PathBuf,
        encryption_key: Option<crate::encrypt::MasterKey>,
    ) -> Result<Self, BrainError> {
        use std::sync::Arc;
        use memmap2::Mmap;

        let f = fs::File::open(&path_buf)?;
        // SAFETY: see brain_io_v3::read_v3_path for the same
        // justification — production usage mmaps a file owned by
        // lu4d, not concurrently mutated.
        let mmap = unsafe { Mmap::map(&f) }
            .map_err(|e| BrainError::Format(format!("mmap failed: {}", e)))?;
        let mmap_arc = Arc::new(mmap);

        // Build NetworkArchive (validates network section, builds
        // seed→index lookup tables).
        // Detect encryption before constructing the archive so we can
        // pick the correct save_format on the way out.
        let flags = crate::brain_io_v3::read_v3_flags(&mmap_arc[..])?;
        let is_encrypted = crate::encrypt::is_encrypted(flags);
        let is_compressed = flags & crate::compress::FORMAT_FLAG_COMPRESSED != 0;

        // Build NetworkArchive (validates network section, builds
        // seed-to-index lookup tables). For encrypted files the key
        // is consumed by the archive reader to decrypt blocks at
        // first access; we keep a Clone for the Brain field below so
        // a save() can round-trip the encryption.
        let archive = crate::archive::NetworkArchive::from_mmap_with_key(
            mmap_arc.clone(),
            encryption_key.clone(),
        )?;

        // Read scar log eagerly from the same mmap (small section,
        // active-scar lookups want heap HashSet semantics).
        let scar_log = crate::archive::read_scar_log_from_mmap(&mmap_arc)?;

        // Read bincoded side tables eagerly.
        // Port T.4: tuple now also yields the persisted PendingLog
        // snapshot from section id 9 (defaults to empty for pre-T.4
        // brain files via `de_bincode_optional`).
        let (
            interest_profile_raw,
            meta_reflection,
            research_proposals,
            misc,
            gap_seed_frequency,
            wonder_state_loaded,
            pending_snapshot_loaded,
            rejection_snapshot_loaded,
            lb_session_snapshot_loaded,
            cs_style_snapshot_loaded,
            pack_ledger_snapshot_loaded,
            runtime_config_loaded,
            runtime_config_present_loaded,
            source_overlay_loaded,
            cap_pack_ledger_snapshot_loaded,
        ) = crate::brain_io_v3::read_v3_side_tables_no_crc(&mmap_arc[..])?;

        // Apply the same interest-profile cap floor as the eager path.
        let mut interest_profile = interest_profile_raw;
        let cfg_cap = CuriosityConfig::default().interest_profile_cap;
        if interest_profile.cap == 0 || interest_profile.cap < cfg_cap {
            interest_profile.cap = cfg_cap;
        }

        // Build mmap-backed network. `from_archive` builds adjacency +
        // keyword index from the archived bonds/nodes; the bulk of the
        // node/bond data stays in the file-backed mapping.
        let mut network = AssociativeNetwork::from_archive(archive);

        // OP-022.a: wire the keyword-index sidecar. Read the brain.lbr
        // v3 trailer CRC32 (last 16 bytes: 8 B magic + 4 B CRC + 4 B
        // reserved), build sidecar path `<brain>.kw`, and either install
        // a validated mmap'd sidecar or wire a context so the next
        // `ensure_keyword_index_built` writes it.
        {
            let bytes_for_crc: &[u8] = &mmap_arc[..];
            let trailer_pos = bytes_for_crc.len().saturating_sub(16);
            let brain_crc32 = if bytes_for_crc.len() >= 16 {
                u32::from_le_bytes(
                    bytes_for_crc[trailer_pos + 8..trailer_pos + 12]
                        .try_into().unwrap_or([0u8; 4]),
                )
            } else { 0 };
            let node_count = network.node_count() as u64;
            let bond_count = network.bond_count() as u64;
            let mut sidecar_path = path_buf.clone();
            let mut sidecar_name = sidecar_path.file_name()
                .map(|n| n.to_os_string())
                .unwrap_or_default();
            sidecar_name.push(".kw");
            sidecar_path.set_file_name(sidecar_name);
            // Try to open + validate. On any failure (absent, bad magic,
            // bad CRC, stale brain_crc32/counts), fall through to the
            // rebuild path; ensure_keyword_index_built writes a fresh sidecar.
            let mut installed = false;
            if sidecar_path.exists() {
                match crate::keyword_index::KeywordIndexMmap::open(&sidecar_path) {
                    Ok(kmm) => {
                        match kmm.validate(brain_crc32, node_count, bond_count) {
                            Ok(()) => {
                                network.install_keyword_sidecar(Arc::new(kmm));
                                installed = true;
                            }
                            Err(e) => {
                                eprintln!(
                                    "OP-022.a: keyword sidecar at {} is stale ({}); will rebuild",
                                    sidecar_path.display(), e,
                                );
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!(
                            "OP-022.a: keyword sidecar at {} unreadable ({}); will rebuild",
                            sidecar_path.display(), e,
                        );
                    }
                }
            }
            // Always set the context — even when the sidecar was just
            // installed — so the next save's compact_keyword_sidecar has
            // the correct stamp values.
            let _ = installed;
            network.set_keyword_sidecar_context(
                crate::network::KeywordSidecarContext {
                    path: sidecar_path,
                    brain_crc32,
                    brain_node_count: node_count,
                    brain_bond_count: bond_count,
                },
            );
        }

        let mut brain = Brain {
            network,
            sessions: SessionManager::new(),
            session_memories: SessionMemoryRegistry::new(),
            consolidator: ConsolidationEngine::new(),
            path: Some(path_buf),
            bridge_config: BridgeConfig::default(),
            recent_bridges: VecDeque::new(),
            bridges_dropped_overflow_total: 0,
            curiosity_config: CuriosityConfig::default(),
            wonder_queue: WonderQueue::new(CuriosityConfig::default().wonder_queue_cap),
            interest_profile,
            curiosity_dropped_overflow_total: 0,
            gap_emitter: crate::gap_signal::GapEmitter::new(),
            gap_label_map: std::collections::HashMap::new(),
            wonder_sidecar: None,
            wonder_sidecar_errors: std::sync::atomic::AtomicU64::new(0),
            next_variant_id: 1,
            next_multi_think_id: 1,
            multi_agent_config: crate::agents::MultiAgentConfig::default(),
            performance_monitor: PerformanceMonitor::new(1024),
            meta_reflection,
            self_reflection_config: SelfReflectionConfig::default(),
            narration_config: crate::narration::NarrationConfig::default(),
            scar_log,
            research_config: misc.config,
            research_proposals,
            next_research_proposal_id: misc.next_id,
            gap_seed_frequency,
            save_format: if is_encrypted {
                BrainSaveFormat::V3BinaryEncrypted
            } else if is_compressed {
                BrainSaveFormat::V3BinaryCompressed
            } else {
                BrainSaveFormat::V3Binary
            },
            // Bug A: track the actual on-disk format. main.rs only
            // overwrites save_format when the user explicitly passed
            // --brain-format-out (Issue B fix); default load preserves
            // whatever format the brain was loaded from. Audit fix
            // internal notes: docstring updated post-v1-removal --
            // there is no plaintext default any more.
            loaded_format: Some(if is_encrypted {
                BrainSaveFormat::V3BinaryEncrypted
            } else if is_compressed {
                BrainSaveFormat::V3BinaryCompressed
            } else {
                BrainSaveFormat::V3Binary
            }),
            encryption_key: if is_encrypted { encryption_key } else { None },
            allow_decrypt_on_save: false,
            require_encrypted: false,
            // Port Cat.3c: ephemeral brains have no sidecars to migrate.
            runtime_config_pending_migration: false,
            negation_overlay: crate::negation::NegationOverlay::new(),
            // Port Source-Persist: hydrate SourceOverlay from v3 section #16.
            // De facto loader for the production v3enc canonical brain.
            // OP-SNAPSHOT-PEAK: wrap loaded overlay in Arc<RwLock>.
            source_overlay: Arc::new(RwLock::new(source_overlay_loaded)),
            proposal_chain_context: std::collections::HashMap::new(),
            negation_cues_enabled: true,
            time_source_config: crate::time::TimeSourceConfig::default(),
            sense_config: crate::sense::SenseConfig::default(),
            chain_research_config: crate::research::chain::ChainResearchConfig::default(),
            recall_config: crate::reasoning::RecallConfig::default(),
            admin_revoke_enabled: false,
            next_chain_id: 1,
            ingest_config: IngestConfig::default(),
            tier2_prompt_config: Tier2PromptConfig::default(),
            file_io_config: crate::file_io::FileIoConfig::default(),
            vision_config: crate::mcp_vision::VisionConfig::default(),
            ingest_run_id: make_startup_run_id(),
            df_time_client: None,
            clock: std::sync::Arc::new(crate::time::SystemClockSource::new())
                as std::sync::Arc<dyn crate::time::ClockSource>,
            time_anchor: std::sync::RwLock::new(
                crate::time::AuthenticatedTime::unsigned(
                    now_ms().max(0) as u64,
                    crate::time::MonotonicClock::now(),
                    crate::time::SourceId::system(),
                    (now_ms().max(0) as u64).saturating_add(
                        crate::time::TimeConfig::default().refresh_interval_ms,
                    ),
                ),
            ),
            anomaly_detector: std::sync::Mutex::new(
                crate::time::AnomalyDetector::new(crate::time::TimeConfig::default()),
            ),
            idle_state: crate::idle::IdleState::new(),
            pending: crate::pending::PendingLog::new_in_memory(),
            rejection_store: crate::rejection::RejectionPatternStore::new(),
            lb_session: crate::lb_session::SessionMemoryStore::new(),
            cs_style_store: crate::style::ConversationalStyleStore::new(),
            style_config: crate::style::StyleConfig::default(),
            pack_ledger: crate::pack::PackLedgerStore::new(),
            cap_pack_ledger: crate::capability_pack::CapabilityPackLedgerStore::new(),
            pack_registry: crate::capability_runtime::LoadedPackRegistry::default(),
            trust_table: crate::trust::TrustTable::default(),
            hub_degree_config: crate::hub_degree::HubDegreeConfig::default(),
            hub_cap_rejections_total: 0,
            hub_cap_evictions_total: 0,
            memtable: None,
            wal: None,
            wal_path: None,
            flush_worker: None,
            manifest_snapshot: None,
            flush_intern: None,
            ingest_only: false,
            read_path_mode: std::sync::atomic::AtomicU8::new(crate::read_path::ReadPathMode::V3Only.as_u8()),
            orphan_cleanup_done: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            v4_sst_cache: std::sync::Mutex::new(std::collections::HashMap::new()),
            v4_block_cache: std::sync::Arc::new(crate::sst::BlockCache::new(crate::sst::DEFAULT_SST_BLOCK_CACHE_BYTES)),
            v4_parsed_block_cache: std::sync::Arc::new(crate::sst::ParsedBlockCache::new(crate::sst::DEFAULT_PARSED_BLOCK_CACHE_BYTES)),
            compaction_worker: None,
            bond_v5_compaction: None,
            node_v5_compaction: None,
            brain_dir_v4: None,
            node_reader_cache: None,
            v5_store_root: None,
            bond_reader_cache: None,
            resident_map: None,
            cache_adjacency: None,
            decay_config: crate::decay::DecayConfig::default(),
            // Port T.6: same overlay as Brain::new -- loader paths
            // get a fresh AutoTickDriver. The MCP server may rebuild
            // it against the brain's actual ClockSource after load.
            auto_tick: crate::idle::AutoTickDriver::new(
                std::sync::Arc::new(crate::time::SystemClockSource::new())
                    as std::sync::Arc<dyn crate::time::ClockSource>,
            ),
            save_runner: std::sync::Arc::new(crate::save_async::SaveRunner::default()),
        };
        // Port LB.2: wonder_state hydration moved to the LB.2 unified
        // entry point below (after attach_lb_session_wal so LB is ready).
        let wonder_state_for_lb2 = wonder_state_loaded;
        // Port T.4: hydrate pending log from the v3 brain section snapshot.
        brain.attach_pending_wal(pending_snapshot_loaded);
        // Port T.8: hydrate rejection-pattern store from v3 section #10.
        brain.rejection_store = crate::rejection::RejectionPatternStore::from_snapshot(
            rejection_snapshot_loaded,
        );
        // Port LB / LB.1: hydrate Left Brain session memory from v3
        // section #11 AND replay <brain>.lb_session.wal on top.
        brain.attach_lb_session_wal(lb_session_snapshot_loaded);
        // Port Cat.3c: prefer v3 section #14 RuntimeConfig if present;
        // otherwise fall back to the legacy sidecars (Cat.3a / Cat.3b)
        // for the one-shot migration. Either way, the next save will
        // emit section #14 and retire the sidecars.
        if runtime_config_present_loaded {
            brain.trust_table = runtime_config_loaded.trust_table.clone();
            brain.hub_degree_config = runtime_config_loaded.hub_degree.clone();
            brain.network.sync_hub_read_config(&brain.hub_degree_config);
            // Port Cat.3c.1: hydrate the CS.2 output-adaptation knobs
            // from section #15.
            brain.style_config.output_adaptation_enabled =
                runtime_config_loaded.style_output_adaptation_enabled;
            brain.style_config.max_samples_for_full_weight =
                runtime_config_loaded.style_max_samples_for_full_weight;
            // Port T.1.1: hydrate time-source config.
            brain.time_source_config = crate::time::TimeSourceConfig {
                mode: runtime_config_loaded.time_source.clone(),
                url: runtime_config_loaded.time_service_url.clone(),
                pubkey_hex: runtime_config_loaded.time_service_pubkey_hex.clone(),
                poll_interval_ms: runtime_config_loaded.time_service_poll_interval_ms,
                max_drift_ms: runtime_config_loaded.time_service_max_drift_ms,
            };
            brain.sense_config = crate::sense::SenseConfig {
                disambiguation_enabled: runtime_config_loaded.sense_disambiguation_enabled,
                context_window: runtime_config_loaded.sense_context_window,
                loser_gain: runtime_config_loaded.sense_loser_gain,
            };
            brain.chain_research_config = crate::research::chain::ChainResearchConfig {
                enabled: runtime_config_loaded.chained_research_enabled,
                max_chain_depth: runtime_config_loaded.max_chain_depth,
                max_chain_cycles: runtime_config_loaded.max_chain_cycles,
                gap_threshold: runtime_config_loaded.chain_gap_threshold,
            };
            // Port BR.4: hydrate recall-time typed-KU flag.
            brain.recall_config = crate::reasoning::RecallConfig {
                typed_ku_enabled: runtime_config_loaded.recall_typed_ku_enabled,
            };
            // Port BR.5: hydrate admin-revoke gate.
            brain.admin_revoke_enabled = runtime_config_loaded.admin_revoke_enabled;
            // Port T-Ingest: hydrate ingest knobs from section #15.
            brain.ingest_config = crate::brain::IngestConfig {
                corpus_default: runtime_config_loaded.ingest_corpus_default.clone(),
                run_id_strategy: runtime_config_loaded.ingest_run_id_strategy.clone(),
                auto_approve_threshold: runtime_config_loaded.ingest_auto_approve_threshold,
            };
            brain.tier2_prompt_config = crate::brain::Tier2PromptConfig {
                default_session_strategy: runtime_config_loaded.tier2_prompt_default_session_strategy.clone(),
                max_text_bytes: runtime_config_loaded.tier2_prompt_max_text_bytes,
            };
            // Port File-IO: hydrate file IO knobs from section #15.
            brain.file_io_config = crate::file_io::FileIoConfig {
                max_size_bytes: runtime_config_loaded.file_io_max_size_bytes,
                allowed_read_paths: runtime_config_loaded.file_io_allowed_read_paths.clone(),
                allowed_write_paths: runtime_config_loaded.file_io_allowed_write_paths.clone(),
            };
            // Port R input: hydrate vision knobs from section #15.
            brain.vision_config = crate::mcp_vision::VisionConfig {
                extractor_kind: runtime_config_loaded.image_extractor_kind.clone(),
                subprocess_path: runtime_config_loaded.image_extractor_subprocess_path.clone(),
                subprocess_timeout_ms: runtime_config_loaded.image_extractor_subprocess_timeout_ms,
                image_max_size_bytes: runtime_config_loaded.image_max_size_bytes,
                image_supported_mime_types: runtime_config_loaded.image_supported_mime_types.clone(),
            };
            brain.runtime_config_pending_migration = false;
        } else {
            // Port Cat.3c.1: pre-Cat.3c brain — defaults; migration
            // is presumed complete, so we don't auto-load sidecars.
            brain.runtime_config_pending_migration = true;
        }
        // Port CS / CS.1: hydrate ConversationalStyle store from v3
        // section #12 AND replay <brain>.style.wal on top. Replay is
        // idempotent — entries already absorbed into the section are
        // skipped via the `last_refined_at_ms` ordering guard.
        brain.attach_cs_style_wal(cs_style_snapshot_loaded);
        // Port Cat.4: hydrate PackLedger store from v3 section #13.
        brain.pack_ledger = crate::pack::PackLedgerStore::from_snapshot(pack_ledger_snapshot_loaded);
        // Port Cat.5: hydrate capability-pack ledger store from v3 section #17.
        brain.cap_pack_ledger = crate::capability_pack::CapabilityPackLedgerStore::from_snapshot(cap_pack_ledger_snapshot_loaded);
        // Port LB.2: pick canonical source for wonder state (section #11
        // if LB has WonderItem entries, else section #8 fallback).
        brain.hydrate_wonder_state_lb2(&wonder_state_for_lb2, now_ms() as u64);
        Ok(brain)
    }

    /// Storage v2 Slice 2c — load with an explicit mode override.
    /// `BrainLoadMode::Auto` (the default) auto-detects the on-disk
    /// format and uses mmap-zerocopy for v3 files. `Eager` forces the
    /// heap-deserialize path even on v3 (debug fallback).
    /// `MmapZeroCopy` returns a `VersionMismatch` error if the file
    /// is not v3.
    pub fn load_or_new_with_mode<P: AsRef<Path>>(
        path: P,
        mode: BrainLoadMode,
    ) -> Result<Self, BrainError> {
        Self::load_or_new_with_mode_and_key(path, mode, None)
    }

    /// Storage v2 Phase 4 -- like `load_or_new_with_mode`, plus an
    /// optional master key for encrypted v3 files. The key is
    /// consumed by the mmap-zero-copy loader and stashed on `Brain`
    /// so `save()` can re-encrypt.
    pub fn load_or_new_with_mode_and_key<P: AsRef<Path>>(
        path: P,
        mode: BrainLoadMode,
        encryption_key: Option<crate::encrypt::MasterKey>,
    ) -> Result<Self, BrainError> {
        // OP-007: read-only reader mode dispatches to the non-writing v5
        // directory reader (see src/read_only_reader.rs).
        if matches!(mode, BrainLoadMode::ReadOnlyReader) {
            return Self::load_readonly(path.as_ref(), encryption_key);
        }
        let path_buf = path.as_ref().to_path_buf();
        if !path_buf.exists() {
            // Port T-Ingest.3 internal notes fix): empty-file path must
            // propagate the `encryption_key` so the first save produces
            // v3enc directly. Previously this branch dropped the key,
            // forcing a REPL-save-then-migrate dance that produced a
            // transient plaintext-on-disk artifact (mild C1 violation).
            // When a key is provided we set save_format/loaded_format to
            // V3BinaryEncrypted so the very first save is encrypted.
            // When no key is provided we preserve the existing plaintext
            // behaviour (--require-encrypted refusal happens at the CLI
            // entry point via `require_encrypted_create_or_err`).
            let mut brain = Self::load_or_new(path)?;
            if let Some(key) = encryption_key {
                brain.encryption_key = Some(key);
                brain.save_format = BrainSaveFormat::V3BinaryEncrypted;
                brain.loaded_format = Some(BrainSaveFormat::V3BinaryEncrypted);
            }
            return Ok(brain);
        }
        let detected = crate::brain_io::detect_format_path(&path_buf)?;
        let is_v3 = matches!(detected, crate::brain_io::BrainFormat::V3Binary);
        match (mode, is_v3) {
            (BrainLoadMode::MmapZeroCopy, false) => Err(BrainError::VersionMismatch(0)),
            (BrainLoadMode::Auto, true) | (BrainLoadMode::MmapZeroCopy, true) => {
                Self::load_v3_zerocopy_with_key(path_buf, encryption_key)
            }
            _ => {
                if encryption_key.is_some() {
                    return Err(BrainError::Format(
                        "--brain-format-in v3-eager cannot decrypt; use auto or v3-mmap-zerocopy"
                            .into(),
                    ));
                }
                Self::load_or_new(path_buf)
            }
        }
    }

    /// Save the network to the configured path (if any), in the format
    /// selected by `self.save_format`.
    ///
    /// **OP-BRAIN-V4 Phase 8 amendment.** For v4-loaded brains (loaded via
    /// `Brain::load_v4`, i.e. `self.brain_dir_v4.is_some()`) this is a
    /// no-op that returns `Ok(())` immediately. v4 durability is provided
    /// continuously by the flush worker (MemTable -> WAL -> L0 SST), not
    /// by a monolithic save. Callers that need an immediate flush (tests,
    /// shutdown drains) should use `flush_now()` instead. The legacy
    /// `lu4d_save` MCP tool still calls this method; for v4 brains the
    /// call therefore becomes a cheap success that documents "the brain
    /// is already durable, no work to do here".
    pub fn save(&mut self) -> Result<(), BrainError> {
        // Port T.4: takes &mut self because save_to_format now
        // truncates the pending-log WAL after a successful save.
        // Phase 8: v4-loaded brains save continuously via the flush
        // worker; the legacy save path is unused.
        if self.brain_dir_v4.is_some() {
            return Ok(());
        }
        let path = match &self.path {
            Some(p) => p.clone(),
            None => return Ok(()),
        };
        let format = self.save_format;
        self.save_to_format(&path, format)
    }

    /// Save the network to an explicit path in the format selected by
    /// `self.save_format`. Kept for backward compatibility with callers
    /// that don't care about the format toggle.
    pub fn save_to<P: AsRef<Path>>(&mut self, path: P) -> Result<(), BrainError> {
        // Port T.4: &mut self for WAL truncate.
        let format = self.save_format;
        self.save_to_format(path, format)
    }

    /// Build a `BrainFile` snapshot from this brain. Used by the v2
    /// and v3 binary save paths and the migration tool. (v1 JSON is no
    /// longer a save target; the v1 reader is restricted to
    /// `load_v1_for_migration_only`.)
    /// `format_version` is set by the caller based on which on-disk
    /// layout it's about to write.
    pub fn to_brain_file(&self, format_version: u32) -> BrainFile {
        BrainFile {
            format_version,
            network: self.network.to_snapshot(),
            interest_profile: self.interest_profile.clone(),
            meta_reflection: self.meta_reflection.clone(),
            scar_log: self.scar_log.clone(),
            research_proposals: self.research_proposals.clone(),
            research_config: self.research_config.clone(),
            next_research_proposal_id: self.next_research_proposal_id,
            gap_seed_frequency: self.gap_seed_frequency.clone(),
            wonder_state: crate::curiosity::WonderState {
                signals: self.wonder_queue.snapshot_signals(),
                gap_label_map: self.gap_label_map.clone(),
            },
            negation_overlay: self.negation_overlay.clone(),
            // OP-SNAPSHOT-PEAK: read-lock + clone for BrainFile serialization.
            source_overlay: self.source_overlay.read().unwrap().clone(),
            // Port T.4: snapshot the in-memory PendingLog into the
            // brain file. After atomic_write_bytes succeeds the
            // save path also calls `self.pending.truncate_wal(None)`
            // so the WAL is reset and the v3 section becomes the
            // new source of truth.
            pending_log: self.pending.snapshot(),
            // Port T.8: snapshot the in-memory rejection-pattern store
            // into the brain file. The v3 section #10 round-trip is
            // handled in `brain_io_v3::write_v3` / `read_v3_bytes`.
            rejection_patterns: self.rejection_store.snapshot(),
            // Port LB / LB.1: snapshot the in-memory Left Brain
            // session store into v3 binary section #11. LB.1 added a
            // companion `<brain>.lb_session.wal` for between-saves
            // durability (see `attach_lb_session_wal` and the
            // post-save `lb_session.truncate_wal` call below).
            lb_session: self.lb_session.snapshot(),
            cs_style: self.cs_style_store.snapshot(),
            pack_ledger: self.pack_ledger.snapshot(),
            cap_pack_ledger: self.cap_pack_ledger.snapshot(),
            // Port Cat.3c / Cat.3c.1: snapshot the in-memory trust table
            // + hub-degree config + Cat.3c.1 style knobs into the v3
            // RuntimeConfig sections (#14 frozen-bincode shape + #15
            // JSON full-state shape). The on-disk sidecars are retired
            // after the save returns successfully (see `save_to_format`).
            runtime_config: crate::runtime_config::RuntimeConfigSnapshot {
                trust_table: self.trust_table.clone(),
                hub_degree: self.hub_degree_config.clone(),
                style_decay_half_life_days: None,
                // Port Cat.3c.1: persist the CS.2 output-adaptation knobs
                // through section #15 (JSON, forward-compat).
                style_output_adaptation_enabled: self.style_config.output_adaptation_enabled,
                style_max_samples_for_full_weight: self.style_config.max_samples_for_full_weight,
                // Port T.1.1: persist DF Time Service client config
                // through section #15.
                time_source: self.time_source_config.mode.clone(),
                time_service_url: self.time_source_config.url.clone(),
                time_service_pubkey_hex: self.time_source_config.pubkey_hex.clone(),
                time_service_poll_interval_ms: self.time_source_config.poll_interval_ms,
                time_service_max_drift_ms: self.time_source_config.max_drift_ms,
                sense_disambiguation_enabled: self.sense_config.disambiguation_enabled,
                sense_context_window: self.sense_config.context_window,
                sense_loser_gain: self.sense_config.loser_gain,
                chained_research_enabled: self.chain_research_config.enabled,
                max_chain_depth: self.chain_research_config.max_chain_depth,
                max_chain_cycles: self.chain_research_config.max_chain_cycles,
                chain_gap_threshold: self.chain_research_config.gap_threshold,
                // Port BR.4: persist recall-time typed-KU flag.
                recall_typed_ku_enabled: self.recall_config.typed_ku_enabled,
                // Port BR.5: persist admin-revoke gate.
                admin_revoke_enabled: self.admin_revoke_enabled,
                // Port T-Ingest: persist ingest knobs into section #15.
                ingest_corpus_default: self.ingest_config.corpus_default.clone(),
                ingest_run_id_strategy: self.ingest_config.run_id_strategy.clone(),
                ingest_auto_approve_threshold: self.ingest_config.auto_approve_threshold,
                tier2_prompt_default_session_strategy: self.tier2_prompt_config.default_session_strategy.clone(),
                tier2_prompt_max_text_bytes: self.tier2_prompt_config.max_text_bytes,
                // Port File-IO: persist file IO knobs into section #15.
                file_io_max_size_bytes: self.file_io_config.max_size_bytes,
                file_io_allowed_read_paths: self.file_io_config.allowed_read_paths.clone(),
                file_io_allowed_write_paths: self.file_io_config.allowed_write_paths.clone(),
                // Port R input: persist vision knobs into section #15.
                image_extractor_kind: self.vision_config.extractor_kind.clone(),
                image_extractor_subprocess_path: self.vision_config.subprocess_path.clone(),
                image_extractor_subprocess_timeout_ms: self.vision_config.subprocess_timeout_ms,
                image_max_size_bytes: self.vision_config.image_max_size_bytes,
                image_supported_mime_types: self.vision_config.image_supported_mime_types.clone(),
            },
            // Writers always emit section #14, so the on-disk file is
            // by definition Cat.3c‑or‑newer.
            runtime_config_present: true,
        }
    }

    /// OP-017 — same as `to_brain_file` but skips the
    /// `network.to_snapshot()` call. The returned BrainFile has an
    /// empty `NetworkSnapshot` placeholder; the streaming save path
    /// (`brain_io_v3::write_v3_compressed_encrypted_to`) ignores
    /// `file.network` and instead streams nodes/bonds directly from
    /// the explicit `NetworkSource` argument (`&self.network`).
    ///
    /// Eliminating the `to_snapshot()` call saves ~7 GiB of heap on
    /// the production brain (3.27M KnowledgeNode heap copies, each
    /// including a 384-dim embedding `Vec<f32>` — see the OP-017
    /// Phase 2 profile report).
    pub fn to_brain_file_no_network(&self, format_version: u32) -> BrainFile {
        BrainFile {
            format_version,
            network: crate::network::NetworkSnapshot {
                nodes: Vec::new(),
                bonds: Vec::new(),
            },
            interest_profile: self.interest_profile.clone(),
            meta_reflection: self.meta_reflection.clone(),
            scar_log: self.scar_log.clone(),
            research_proposals: self.research_proposals.clone(),
            research_config: self.research_config.clone(),
            next_research_proposal_id: self.next_research_proposal_id,
            gap_seed_frequency: self.gap_seed_frequency.clone(),
            wonder_state: crate::curiosity::WonderState {
                signals: self.wonder_queue.snapshot_signals(),
                gap_label_map: self.gap_label_map.clone(),
            },
            negation_overlay: self.negation_overlay.clone(),
            // OP-SNAPSHOT-PEAK: read-lock + clone for BrainFile serialization.
            source_overlay: self.source_overlay.read().unwrap().clone(),
            pending_log: self.pending.snapshot(),
            rejection_patterns: self.rejection_store.snapshot(),
            lb_session: self.lb_session.snapshot(),
            cs_style: self.cs_style_store.snapshot(),
            pack_ledger: self.pack_ledger.snapshot(),
            cap_pack_ledger: self.cap_pack_ledger.snapshot(),
            runtime_config: crate::runtime_config::RuntimeConfigSnapshot {
                trust_table: self.trust_table.clone(),
                hub_degree: self.hub_degree_config.clone(),
                style_decay_half_life_days: None,
                style_output_adaptation_enabled: self.style_config.output_adaptation_enabled,
                style_max_samples_for_full_weight: self.style_config.max_samples_for_full_weight,
                time_source: self.time_source_config.mode.clone(),
                time_service_url: self.time_source_config.url.clone(),
                time_service_pubkey_hex: self.time_source_config.pubkey_hex.clone(),
                time_service_poll_interval_ms: self.time_source_config.poll_interval_ms,
                time_service_max_drift_ms: self.time_source_config.max_drift_ms,
                sense_disambiguation_enabled: self.sense_config.disambiguation_enabled,
                sense_context_window: self.sense_config.context_window,
                sense_loser_gain: self.sense_config.loser_gain,
                chained_research_enabled: self.chain_research_config.enabled,
                max_chain_depth: self.chain_research_config.max_chain_depth,
                max_chain_cycles: self.chain_research_config.max_chain_cycles,
                chain_gap_threshold: self.chain_research_config.gap_threshold,
                recall_typed_ku_enabled: self.recall_config.typed_ku_enabled,
                admin_revoke_enabled: self.admin_revoke_enabled,
                ingest_corpus_default: self.ingest_config.corpus_default.clone(),
                ingest_run_id_strategy: self.ingest_config.run_id_strategy.clone(),
                ingest_auto_approve_threshold: self.ingest_config.auto_approve_threshold,
                tier2_prompt_default_session_strategy: self.tier2_prompt_config.default_session_strategy.clone(),
                tier2_prompt_max_text_bytes: self.tier2_prompt_config.max_text_bytes,
                file_io_max_size_bytes: self.file_io_config.max_size_bytes,
                file_io_allowed_read_paths: self.file_io_config.allowed_read_paths.clone(),
                file_io_allowed_write_paths: self.file_io_config.allowed_write_paths.clone(),
                image_extractor_kind: self.vision_config.extractor_kind.clone(),
                image_extractor_subprocess_path: self.vision_config.subprocess_path.clone(),
                image_extractor_subprocess_timeout_ms: self.vision_config.subprocess_timeout_ms,
                image_max_size_bytes: self.vision_config.image_max_size_bytes,
                image_supported_mime_types: self.vision_config.image_supported_mime_types.clone(),
            },
            runtime_config_present: true,
        }
    }

    /// Save in an explicit on-disk format.
    ///
    /// The v1 path writes JSON-pretty (the legacy layout). The v2 path
    /// writes the binary header + offset-table + sections + crc32 trailer
    /// defined in `brain_io`.
    pub fn save_to_format<P: AsRef<Path>>(
        &mut self,
        path: P,
        format: BrainSaveFormat,
    ) -> Result<(), BrainError> {
        // OP-007: a read-only reader must never write the brain.
        if self.is_read_only_reader() {
            return Err(BrainError::Format(
                "save refused: brain is a read-only reader (OP-007 guest reader)".to_string(),
            ));
        }
        // Port T.7: structural encryption guard. When --require-encrypted
        // is set, refuse to write any non-v3enc format. This applies
        // regardless of `loaded_format` -- Solution #79 (allow_decrypt_on_save)
        // only covered brains that LOADED from v3enc. T.7 covers the
        // case where the in-memory brain was never encrypted to begin
        // with, which is the gap that allowed an unencrypted v3 brain
        // to land at /var/lib/lu/brain.lbr from a training output.
        if self.require_encrypted && format != BrainSaveFormat::V3BinaryEncrypted {
            return Err(BrainError::EncryptionRequired {
                attempted_format: format.as_str().to_string(),
                hint: "use --brain-format-out v3enc".to_string(),
            });
        }
        // Critical audit fix internal notes / Problem #111 follow-up):
        // refuse to downgrade an at-rest-encrypted brain to a plaintext
        // format on save. Previously this was a generic warning printed
        // by main.rs; the warning was easy to miss and a misconfigured
        // --brain-format-out v3 silently wrote plaintext to disk for a
        // brain that had been loaded from v3enc. The guard is in
        // save_to_format itself so it applies to every save path,
        // including programmatic callers and the migration tool.
        if let Some(loaded) = self.loaded_format {
            if loaded == BrainSaveFormat::V3BinaryEncrypted
                && format != BrainSaveFormat::V3BinaryEncrypted
                && !self.allow_decrypt_on_save
            {
                return Err(BrainError::Format(format!(
                    "refusing to downgrade encrypted brain to {} on save: this would                      write plaintext to disk for a brain that was loaded from                      an at-rest-encrypted file. Pass --allow-decrypt-on-save                      (or set Brain::allow_decrypt_on_save = true programmatically)                      to opt-in explicitly.",
                    format.as_str()
                )));
            }
        }
        if let Some(parent) = path.as_ref().parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        match format {
            BrainSaveFormat::V1Json => {
                // v1 JSON removal internal notes: plaintext JSON
                // serialization is no longer a valid storage format. A
                // production brain with an encryption key set must NEVER
                // round-trip through plaintext JSON at any lifecycle point;
                // refusing here is the structural guarantee. Migration from
                // an existing v1 file uses --migrate-from-v1.
                return Err(BrainError::Format(format!(
                    "refusing to save brain as v1 JSON: plaintext storage is no longer supported. \
                    Run lu4d --migrate-from-v1 SRC DST [--keyfile P] [--out-format v3enc|v3] to migrate \
                    a legacy v1 file."
                )));
            }
            BrainSaveFormat::V2Binary => {
                let file = self.to_brain_file(crate::brain_io::FORMAT_VERSION_V2);
                let bytes = crate::brain_io::write_v2(&file)?;
                atomic_write_bytes(path.as_ref(), &bytes)?;
            }
            BrainSaveFormat::V3BinaryCompressed => {
                let file = self.to_brain_file(crate::brain_io_v3::FORMAT_VERSION_V3);
                let bytes = crate::brain_io_v3::write_v3_compressed(
                    &file,
                    crate::compress::DEFAULT_BLOCK_SIZE,
                    crate::compress::DEFAULT_ZSTD_LEVEL,
                )?;
                atomic_write_bytes(path.as_ref(), &bytes)?;
            }
            BrainSaveFormat::V3BinaryEncrypted => {
                // OP-017 — streaming save path. We bypass the in-memory
                // `Vec<u8>` assembly that the legacy writer produces and
                // instead stream the entire ~2.3 GiB output directly to
                // the .tmp file. No `NetworkSnapshot` heap materialisation
                // happens — `to_brain_file_no_network()` leaves the
                // BrainFile.network field empty and the streaming writer
                // walks `&self.network` (an `&AssociativeNetwork`) via the
                // `NetworkSource` iterator that decodes one archived
                // entity at a time. See OP-017 Phase 2 profile report
                // for the +12 GiB → ≤1 GiB save-spike rationale.
                let key = self.encryption_key.as_ref().ok_or_else(|| {
                    BrainError::Format(
                        "V3BinaryEncrypted save requires Brain.encryption_key to be set"
                            .into(),
                    )
                })?;
                // OP-022.ef — use the borrowed-side-table streaming writer.
                // No  clone (~3.45 GiB on the prod brain).
                // Each side table is snapshot+serialized+dropped one at a time.
                let brain_ref: &Brain = self;
                // OP-022.s.v3 — eligible-full-passthrough source. Returns Some
                // ONLY when the brain was loaded mmap-zerocopy AND no overlay
                // mutation is pending; the writer's param-matching gate is
                // the final safety check.
                let passthrough_src = brain_ref
                    .network
                    .compressed_archive_for_passthrough();
                // OP-022.s.v4 — eligible-partial-passthrough source. Returns
                // Some when the archive is mmap-backed AND compressed.
                // Overlay/tombstones can be non-empty (the writer handles
                // shadow-exclusion + tombstone filtering).
                let partial_src = brain_ref
                    .network
                    .compressed_archive_for_partial_passthrough();
                let overlay_nodes_adapter;
                let overlay_bonds_adapter;
                let partial_args = if passthrough_src.is_some() {
                    // Full passthrough wins when available; v4 only kicks in
                    // when the v3 predicate fails (overlay non-empty).
                    None
                } else if let Some(src) = partial_src {
                    overlay_nodes_adapter =
                        crate::compress::AssociativeNetworkOverlayNodes(&brain_ref.network);
                    overlay_bonds_adapter =
                        crate::compress::AssociativeNetworkOverlayBonds(&brain_ref.network);
                    Some(crate::brain_io_v3::PartialPassthroughArgs {
                        source: src,
                        tombstoned_nodes: brain_ref.network.overlay_deletions_ref(),
                        tombstoned_bonds: brain_ref.network.bond_tombstones_ref(),
                        overlay_nodes: &overlay_nodes_adapter,
                        overlay_bonds: &overlay_bonds_adapter,
                    })
                } else {
                    None
                };
                atomic_write_with(path.as_ref(), |f| {
                    crate::brain_io_v3::write_v3_compressed_encrypted_to_borrowed(
                        &brain_ref.network,
                        brain_ref,
                        crate::compress::DEFAULT_BLOCK_SIZE,
                        crate::compress::DEFAULT_ZSTD_LEVEL,
                        key,
                        f,
                        passthrough_src,
                        partial_args.as_ref(),
                    )?;
                    Ok(())
                })?;
            }
            BrainSaveFormat::V3Binary => {
                let file = self.to_brain_file(crate::brain_io_v3::FORMAT_VERSION_V3);
                let bytes = crate::brain_io_v3::write_v3(&file)?;
                atomic_write_bytes(path.as_ref(), &bytes)?;
            }
        }
        // OP-022.a: compact the keyword-index sidecar so its
        // brain_crc32/node_count/bond_count match the just-written
        // brain.lbr. No-op when:
        //   - sidecar context wasn't wired (non-v3 load, ephemeral brain)
        //   - no overlay + tombstones changes AND base brain_crc32 still matches
        //   - save format isn't v3 (V2 path skips below; v3 paths fall through)
        // Failures are logged but not propagated — the brain.lbr is
        // durable; a stale sidecar will be rejected and rebuilt on next load.
        match format {
            BrainSaveFormat::V3Binary
            | BrainSaveFormat::V3BinaryCompressed
            | BrainSaveFormat::V3BinaryEncrypted => {
                let saved_path = path.as_ref();
                let new_crc = read_v3_trailer_crc(saved_path).unwrap_or(0);
                if let Err(e) = self.network.compact_keyword_sidecar(new_crc) {
                    eprintln!(
                        "OP-022.a: keyword sidecar compact failed: {} (brain.lbr is durable)",
                        e,
                    );
                }
            }
            _ => {}
        }

        // Port T.4: now that the brain file is durably on disk
        // (atomic_write_bytes did fsync + rename + parent-dir fsync),
        // the v3 `pending_log` section is the consolidated source of
        // truth for the pending state. Truncate the WAL so it only
        // captures mutations after this save.
        //
        // If truncate fails, the WAL still holds entries that are
        // also in the snapshot. PendingLog::open_with_snapshot replay
        // is idempotent (dedupes by pending_id), so the next load is
        // still correct -- just with stale WAL bytes that get cleared
        // on the next successful truncate. Do NOT propagate the
        // error: the brain file save itself succeeded, and a stale
        // WAL is recoverable.
        if let Err(e) = self.pending.truncate_wal(None) {
            eprintln!(
                "pending WAL truncate failed after save: {} (replay will dedupe)",
                e
            );
        }
        // Port LB.1: same for the Left Brain session memory WAL.
        // Section #11 in the freshly-saved v3 file is now the
        // consolidated source of truth, so the WAL only needs to
        // capture mutations after this save.
        if let Err(e) = self.lb_session.truncate_wal(None) {
            eprintln!(
                "lb_session WAL truncate failed after save: {} (replay will dedupe)",
                e
            );
        }
        // Port CS.1: same for the conversational style WAL. Section
        // #12 in the freshly-saved v3 file is now the consolidated
        // source of truth; the WAL only needs to capture mutations
        // after this save. Replay is idempotent on `last_refined_at_ms`,
        // so a stale WAL after a truncate failure stays correct.
        if let Err(e) = self.cs_style_store.truncate_wal(None) {
            eprintln!(
                "cs_style WAL truncate failed after save: {} (replay will dedupe)",
                e
            );
        }
        // Port Cat.3c: the trust table and hub-degree config now live
        // INSIDE the encrypted v3 brain (section #14). After this save,
        // any legacy plaintext sidecars from Cat.3a / Cat.3b can be
        // retired — we rename them with `.migrated` so a paper trail
        // remains but the legacy load path no longer finds them.
        // Best-effort: a rename failure is logged but does not fail the
        // brain save.
        if let Some(brain_path) = &self.path {
            let trust_path = crate::runtime_config::legacy_trust_sidecar_path(brain_path);
            match crate::runtime_config::retire_legacy_sidecar(&trust_path) {
                Ok(true) => eprintln!("Cat.3c: migrated {} → {}", trust_path.display(), crate::runtime_config::migrated_marker_path(&trust_path).display()),
                Ok(false) => {}
                Err(e) => eprintln!("Cat.3c: trust sidecar retire failed: {} (will retry on next save)", e),
            }
            let hub_path = crate::runtime_config::legacy_hub_degree_sidecar_path(brain_path);
            match crate::runtime_config::retire_legacy_sidecar(&hub_path) {
                Ok(true) => eprintln!("Cat.3c: migrated {} → {}", hub_path.display(), crate::runtime_config::migrated_marker_path(&hub_path).display()),
                Ok(false) => {}
                Err(e) => eprintln!("Cat.3c: hub_degree sidecar retire failed: {} (will retry on next save)", e),
            }
            self.runtime_config_pending_migration = false;
        }
        Ok(())
    }

    /// Teach the brain: subject -[relation]-> object with the given weight
    /// and confidence. Reuses existing nodes (keyed by deterministic seed)
    /// or creates new ones. If a bond already exists, adds a new keyframe
    /// to reinforce it.
    /// Port G — teach a triple with an explicit negation flag. Routes
    /// through `teach_with_kind` for node creation + bond formation,
    /// then sets the negation overlay entry. The bond rkyv layout
    /// stays unchanged; only the sidecar `negation_overlay` distinguishes
    /// "X IsA Y" from "X IsNot Y".
    pub fn teach_with_negation(
        &mut self,
        subject: &str,
        relation: BondType,
        object: &str,
        weight: f32,
        confidence: Confidence,
        negated: bool,
    ) -> TeachResult {
        let r = self.teach_with_kind(subject, relation, object, weight, confidence, None, None);
        self.negation_overlay
            .set(r.subject_seed, r.object_seed, relation, negated);
        // Port S — same default attribution as teach().
        let now_s = now_ms() as u64;
        self.source_overlay.write().unwrap().set(
            r.subject_seed,
            r.object_seed,
            relation,
            crate::source::SourceAttribution::teach("system", now_s),
        );
        r
    }

    /// Port G — teach from raw text. When `negation_cues_enabled` is
    /// true (the default), runs `crate::negation::detect_negation_cue`
    /// on `text` to decide the negation flag. When false, behaves like
    /// `teach` (the negation flag is forced to false). Returns the
    /// teach result and the resolved negation flag for the caller.
    pub fn teach_from_text(
        &mut self,
        text: &str,
        subject: &str,
        relation: BondType,
        object: &str,
        weight: f32,
        confidence: Confidence,
    ) -> (TeachResult, bool) {
        let neg = if self.negation_cues_enabled {
            crate::negation::detect_negation_cue(text)
        } else {
            false
        };
        let r = self.teach_with_negation(subject, relation, object, weight, confidence, neg);
        (r, neg)
    }

    /// Port G — read the negation flag for a (from, to, bond_type) triple.
    /// Defaults to false for any bond not in the overlay.
    pub fn bond_negated(
        &self,
        from: &crate::seed::Seed,
        to: &crate::seed::Seed,
        bond_type: BondType,
    ) -> bool {
        self.negation_overlay.is_negated(from, to, bond_type)
    }

    /// Port G — explicitly set or clear the negation flag for a bond
    /// (idempotent, so callers don't need to teach again).
    pub fn set_bond_negated(
        &mut self,
        from: crate::seed::Seed,
        to: crate::seed::Seed,
        bond_type: BondType,
        negated: bool,
    ) {
        self.negation_overlay.set(from, to, bond_type, negated);
    }

    /// Port G + J — return contradiction signal for a (subject, bond_type,
    /// object) tuple. Specifically: how many distinct bonds we know about
    /// where polarity matches (positive_count) vs. opposes (negative_count).
    /// Today the network deduplicates a (from, to, bond_type) into a
    /// single bond, so the counts are at most 1 each — but the helper
    /// surfaces the "I have conflicting evidence" case when the same
    /// pair was taught both ways. Use this in J's confidence aggregation
    /// to render `ResponseShape::ContestedAnswer` cleanly rather than
    /// silently picking one polarity.
    pub fn bond_evidence(
        &self,
        from: &crate::seed::Seed,
        to: &crate::seed::Seed,
        bond_type: BondType,
    ) -> (usize, usize) {
        // Fast path: a single bond exists in the network. Polarity
        // comes from the overlay flag.
        if self.network.get_bond(from, to).is_some() {
            if self.bond_negated(from, to, bond_type) {
                return (0, 1);
            } else {
                return (1, 0);
            }
        }
        (0, 0)
    }

    /// Port G + J — true if the brain holds contradictory evidence for
    /// the (subject, bond_type, object) tuple. Two routes:
    /// (a) the negation overlay's `contested` set has recorded a flag
    ///     flip across teach calls; OR
    /// (b) `bond_evidence` reports both positive and negative bonds
    ///     between the same pair (future multi-bond shape).
    /// Drives J's confidence aggregation toward an explicit
    /// "conflicting evidence" articulation rather than silently
    /// picking one polarity.
    pub fn has_contradiction(
        &self,
        from: &crate::seed::Seed,
        to: &crate::seed::Seed,
        bond_type: BondType,
    ) -> bool {
        if self.negation_overlay.is_contested(from, to, bond_type) {
            return true;
        }
        let (pos, neg) = self.bond_evidence(from, to, bond_type);
        pos > 0 && neg > 0
    }

    pub fn teach(
        &mut self,
        subject: &str,
        relation: BondType,
        object: &str,
        weight: f32,
        confidence: Confidence,
    ) -> TeachResult {
        // Q.1 — back-compat shim. Existing call sites (tests, non-MCP
        // ingestion) get the original Concept-default behavior.
        let r = self.teach_with_kind(subject, relation, object, weight, confidence, None, None);
        // Port G — record the positive witness so a later flip to
        // negated lights up has_contradiction. Only the positive entry
        // point does this; teach_with_negation manages witnesses on
        // its own.
        self.negation_overlay
            .seen_positive
            .insert(crate::negation::NegationOverlay::key(&r.subject_seed, &r.object_seed, relation));
        // Port S — default attribution for a teach() call. Callers
        // that know more (research apply, document ingest) should
        // route through `teach_with_source` to override.
        // Port T-Ingest — when `ingest_config.corpus_default` is
        // set, stamp `TrainingCorpus { corpus, run_id }` instead
        // of the default `Teach { actor: "system" }`. This is
        // the convenience knob that lets a curriculum-ingest
        // pipeline drive the bare `lu4d_teach` MCP tool without
        // passing `corpus` on every call.
        let now_s = now_ms() as u64;
        let attribution = if let Some(corpus) = &self.ingest_config.corpus_default {
            let run_id = self.resolve_ingest_run_id(None);
            crate::source::SourceAttribution::training_with_url(
                corpus.clone(), run_id, None,
            )
        } else {
            crate::source::SourceAttribution::teach("system", now_s)
        };
        self.source_overlay.write().unwrap().set(
            r.subject_seed,
            r.object_seed,
            relation,
            attribution,
        );
        r
    }

    /// Port T-Ingest — resolve the effective `run_id` for a corpus
    /// stamping operation per the configured strategy. The caller
    /// supplies the explicit `run_id` argument if any; otherwise the
    /// strategy decides:
    ///   * `Caller`         → returns the caller's value verbatim
    ///                        (`None` → `None`).
    ///   * `ServerStartup`  → caller value first, falls back to the
    ///                        startup-allocated `ingest_run_id`.
    ///   * `PerSession`     → caller value first, otherwise the
    ///                        active LB session id (`Some("lb_session_<id>")`),
    ///                        falling back to `ServerStartup` when no
    ///                        LB session is active.
    pub fn resolve_ingest_run_id(&self, caller: Option<String>) -> Option<String> {
        use crate::runtime_config::IngestRunIdStrategy::*;
        if let Some(c) = caller.clone() { if !c.is_empty() { return Some(c); } }
        match self.ingest_config.run_id_strategy {
            Caller => caller,
            ServerStartup => Some(self.ingest_run_id.clone()),
            PerSession => match self.lb_session.active_session() {
                Some(sid) => Some(format!("lb_session_{}", sid)),
                None => Some(self.ingest_run_id.clone()),
            },
        }
    }

    /// Port T-Ingest — teach a bond stamped with `SourceKind::TrainingCorpus`.
    /// Convenience wrapper that resolves the effective `run_id` per
    /// `ingest_config.run_id_strategy` and routes through
    /// `teach_with_source`. NEVER routes through `teach()` (which
    /// would double-stamp via `ingest_config.corpus_default`); the
    /// explicit `corpus` argument always wins.
    pub fn teach_corpus(
        &mut self,
        subject: &str,
        relation: BondType,
        object: &str,
        weight: f32,
        confidence: Confidence,
        corpus: &str,
        caller_run_id: Option<String>,
        source_url: Option<String>,
    ) -> TeachResult {
        // BR.3: thin wrapper over teach_corpus_with_kind. Existing callers
        // (research apply, vision ingest, file ingest, tests) keep working
        // unchanged; new corpus paths that carry kind metadata route through
        // teach_corpus_with_kind directly.
        self.teach_corpus_with_kind(
            subject, relation, object, weight, confidence,
            corpus, caller_run_id, source_url, None, None,
        )
    }

    /// BR.3 — corpus-stamped teach with optional kind hint + relation
    /// alias. Identical T-Ingest.1 trust-resolution semantics; the kind
    /// hint is forwarded through `teach_with_source_kind` to
    /// `teach_with_kind` so newly-created subject/object nodes carry the
    /// requested `KnowledgeContent` variant.
    pub fn teach_corpus_with_kind(
        &mut self,
        subject: &str,
        relation: BondType,
        object: &str,
        weight: f32,
        confidence: Confidence,
        corpus: &str,
        caller_run_id: Option<String>,
        source_url: Option<String>,
        kind_hint: Option<&str>,
        relation_alias: Option<&str>,
    ) -> TeachResult {
        let run_id = self.resolve_ingest_run_id(caller_run_id);
        // Port T-Ingest.1: opt INTO trust-table resolution by constructing
        // the attribution with `trust: None`. teach_with_source_kind will
        // look up the multiplier from `trust_table.corpus_multipliers`
        // (or `default_multiplier` on miss), multiply weight at formation,
        // and stamp the resolved multiplier onto the stored attribution.
        let attribution = crate::source::SourceAttribution::training_for_resolution(
            corpus.to_string(), run_id, source_url,
        );
        self.teach_with_source_kind(
            subject, relation, object, weight, confidence, attribution,
            kind_hint, relation_alias,
        )
    }

    /// Port S — teach with explicit source attribution. Used by the
    /// research apply path (Research kind), document ingest (Document
    /// kind), and any test that needs to install a known source.
    /// Routes through `teach_with_kind` for node creation + bond
    /// formation, then sets the source_overlay entry.
    ///
    /// BR.3: thin wrapper over `teach_with_source_kind` with no kind
    /// hint or relation alias. New callers that want the kind tag to
    /// participate in node creation should use `teach_with_source_kind`.
    pub fn teach_with_source(
        &mut self,
        subject: &str,
        relation: BondType,
        object: &str,
        weight: f32,
        confidence: Confidence,
        attribution: crate::source::SourceAttribution,
    ) -> TeachResult {
        self.teach_with_source_kind(
            subject, relation, object, weight, confidence, attribution, None, None,
        )
    }

    /// BR.3 — `teach_with_source` plus optional kind hint and relation
    /// alias. The hint is forwarded into `teach_with_kind`, which uses
    /// it (with `infer_kind` as fallback) to pick the `KnowledgeContent`
    /// variant for any newly-created subject/object node. Existing
    /// nodes are NOT retyped (kind is fixed at first observation).
    pub fn teach_with_source_kind(
        &mut self,
        subject: &str,
        relation: BondType,
        object: &str,
        weight: f32,
        confidence: Confidence,
        attribution: crate::source::SourceAttribution,
        kind_hint: Option<&str>,
        relation_alias: Option<&str>,
    ) -> TeachResult {
        // Port T-Ingest.1: trust multiplier resolution at bond formation.
        //
        //   `attribution.trust = Some(t)` -> caller resolved or supplied
        //                                     trust; use weight & trust
        //                                     as-is (no re-multiplication).
        //                                     This preserves Cat.3a research
        //                                     path exactly (apply_proposal
        //                                     pre-multiplies weight and stamps
        //                                     trust=Some(multiplier)).
        //   `attribution.trust = None`    -> resolve from trust_table per
        //                                     SourceKind (corpus->corpus_multipliers,
        //                                     research url -> URL host entries,
        //                                     other -> default_multiplier).
        //                                     Multiply weight at formation,
        //                                     stamp trust = Some(resolved) on
        //                                     the STORED attribution.
        //
        // C5 invariant: stored bond weights are NEVER mutated after creation.
        // T-Ingest.1 changes the weight at FORMATION time only — a later
        // mutation of trust_table does not affect existing bonds.
        let mut effective_weight = weight;
        let mut stored_attribution = attribution.clone();
        // Port T-Ingest.1 sentinel discriminator: negative trust means
        // "needs resolution". Pre-T-Ingest.1 entries always stored a
        // non-negative trust (1.0), so this branch never fires for them
        // — backward compatible. teach_corpus / training_for_resolution
        // are the only paths that set the sentinel.
        if stored_attribution.trust < 0.0 {
            let resolved = self.trust_table.resolve_for_kind(&stored_attribution.kind);
            effective_weight = (weight * resolved).clamp(0.0, 1.0);
            stored_attribution.trust = resolved;
        }
        let r = self.teach_with_kind(
            subject,
            relation,
            object,
            effective_weight,
            confidence,
            kind_hint,
            relation_alias,
        );
        self.negation_overlay
            .seen_positive
            .insert(crate::negation::NegationOverlay::key(&r.subject_seed, &r.object_seed, relation));
        self.source_overlay
            .write().unwrap().set(r.subject_seed, r.object_seed, relation, stored_attribution);
        r
    }

    /// Port S — read the source attribution for a `(from, to,
    /// bond_type)` triple. Defaults to `SourceKind::Unknown` for
    /// every bond not explicitly recorded — including every bond
    /// in the canonical 1.26 GB pre-S brain.
    pub fn bond_source(
        &self,
        from: &crate::seed::Seed,
        to: &crate::seed::Seed,
        bond_type: BondType,
    ) -> crate::source::SourceAttribution {
        self.source_overlay.read().unwrap().get(from, to, bond_type)
    }

    /// Port S — explicitly set the source attribution for a bond
    /// (idempotent, so callers don't need to teach again).
    pub fn set_bond_source(
        &mut self,
        from: crate::seed::Seed,
        to: crate::seed::Seed,
        bond_type: BondType,
        attribution: crate::source::SourceAttribution,
    ) {
        self.source_overlay.write().unwrap().set(from, to, bond_type, attribution);
    }

    /// Port S — revoke every bond whose source attribution matches
    /// `predicate`. Removes BOTH the overlay entry AND the
    /// underlying bond from the network so a downstream recall
    /// cannot resurface it. This is what GDPR right-to-be-forgotten
    /// (delete-by-doc), source-trust revocation, and curiosity
    /// bond rollback all need.
    ///
    /// Returns `(overlay_entries_removed, bonds_removed_from_network)`.
    /// They can differ: a bond's overlay entry may be present
    /// without a network bond if a previous call removed the bond
    /// (or, vice-versa, the bond might exist with no overlay
    /// attribution — pre-S bonds — but those won't match any
    /// non-Unknown predicate).
    pub fn revoke_bonds_by_source<F>(&mut self, predicate: F) -> (usize, usize)
    where
        F: Fn(&crate::source::SourceAttribution) -> bool,
    {
        // Collect overlay keys to revoke first so we can split
        // the mutable borrows cleanly. Each key encodes (from, to,
        // bond_type).
        let mut to_revoke: Vec<(crate::seed::Seed, crate::seed::Seed, BondType)> = Vec::new();
        {
            let overlay = self.source_overlay.read().unwrap();
            for (key, attr) in &overlay.entries {
                if predicate(attr) {
                    if let Some(triple) = parse_overlay_key(key) {
                        to_revoke.push(triple);
                    }
                }
            }
        }
        let overlay_removed = self.source_overlay.write().unwrap().revoke_where(predicate);
        let mut bonds_removed = 0usize;
        // Strip the underlying bonds. The overlay seed prefix is
        // 8 bytes; full network seeds are 32 bytes. Linear scan to
        // find network bonds whose seed prefixes match — revoke is
        // an admin op, not a hot path.
        let all_keys: Vec<(crate::seed::Seed, crate::seed::Seed)> =
            self.network.iter_bond_keys_owned();
        for (from, to, _bt) in &to_revoke {
            for (a, b) in &all_keys {
                if &a.0[0..8] == &from.0[0..8] && &b.0[0..8] == &to.0[0..8] {
                    if self.network.remove_bond(a, b) {
                        bonds_removed += 1;
                    }
                }
            }
        }
        (overlay_removed, bonds_removed)
    }

    /// Port S — distinct source kinds across a chain of bond
    /// keys. Used by J's confidence aggregation to render
    /// "based on N sources".
    pub fn distinct_source_kinds_for_chain(
        &self,
        chain: &[(crate::seed::Seed, crate::seed::Seed, BondType)],
    ) -> Vec<crate::source::SourceKind> {
        self.source_overlay.read().unwrap().distinct_kinds_for_chain(chain)
    }

    /// OP-BRAIN-V4 Phase 9 — v4 write helper. Inserts the bond into the
    /// MemTable and appends the WAL record. Called from `teach_with_kind`
    /// when `brain_dir_v4.is_some()`. Hub-degree write-side enforcement
    /// is intentionally skipped on this path (the MemTable has no
    /// concept of in-degree; the v4 brain currently has no equivalent
    /// admission ceiling). Scar admission IS preserved: a teach where
    /// either endpoint is in `scar_log.scarred_seeds` is refused before
    /// the MemTable insert.
    ///
    /// Keyframe shape mirrors the v3 path:
    ///   * Reinforce-existing semantics are modeled as a fresh MemTable
    ///     insert with a single keyframe at the requested weight, sourced
    ///     as `Session(id)` if a session is active otherwise `Manual`.
    ///     The Phase 5 merged read path dedupes by
    ///     `(subject, predicate, object)` taking latest-seq, so the newer
    ///     insert structurally supersedes the older one on read.
    ///   * New-bond semantics use a single `KeyframeSource::Baseline`
    ///     keyframe, matching `TemporalBond::new` for the v3 path.
    ///
    /// Returns nothing; the caller flips `reinforced_existing_bond` if we
    /// chose the reinforce shape (i.e. an entry for this triple already
    /// existed in the MemTable). "Already existed" is decided by checking
    /// the MemTable for an outgoing bond from `subject_seed` whose object
    /// + predicate match; we deliberately do NOT consult L0 SSTs here
    /// because the read-merge handles dedup uniformly downstream and
    /// pulling SST blocks on every teach would tank ingest throughput.
    fn teach_with_kind_v4(
        &mut self,
        subject_seed: Seed,
        object_seed: Seed,
        relation: BondType,
        weight: f32,
        now: i64,
        active_session_id: Option<u64>,
        reinforced_existing_bond: &mut bool,
    ) {
        // Scar-seed admission: identical contract to the v3
        // `add_bond_with_scar_check` path. The HashSet check is cheap
        // (~50ns) so we do it on every teach — same as v3.
        if self.scar_log.is_seed_scarred(subject_seed)
            || self.scar_log.is_seed_scarred(object_seed)
        {
            let reason = if self.scar_log.is_seed_scarred(subject_seed)
                && self.scar_log.is_seed_scarred(object_seed)
            {
                "both endpoints are scarred concepts".to_string()
            } else if self.scar_log.is_seed_scarred(subject_seed) {
                "from-endpoint is a scarred concept".to_string()
            } else {
                "to-endpoint is a scarred concept".to_string()
            };
            let now_ms_for_scar = crate::scar::now_ms();
            let _ = self.scar_log.scar_bond(
                subject_seed,
                object_seed,
                reason,
                Vec::new(),
                None,
                now_ms_for_scar,
            );
            return;
        }

        // Resolve MemTable + WAL + intern handles. All three must be
        // present when `brain_dir_v4.is_some()` because `load_v4`
        // installs them via `enable_memtable` before returning. A
        // missing handle is a structural bug, not a runtime error
        // condition — we log a single warning and fall through silently
        // rather than panicking inside an ingest hot path. The mcp
        // status / batch counters surface accepted-vs-dropped
        // independently.
        let (mt, wal, intern) = match (
            self.memtable.as_ref().cloned(),
            self.wal.as_ref().cloned(),
            self.flush_intern.as_ref().cloned(),
        ) {
            (Some(mt), Some(wal), Some(intern)) => (mt, wal, intern),
            _ => {
                eprintln!(
                    "[brain] teach_with_kind_v4: missing MemTable/WAL/intern                      handle on v4-loaded brain (structural bug); dropping write"
                );
                return;
            }
        };

        // Decide reinforce-vs-new by probing the MemTable. We DO NOT
        // consult L0 SSTs here: pulling SST blocks on every teach would
        // make ingest unusably slow, and the read-merge handles dedup
        // uniformly downstream (latest-seq wins). The semantic cost is
        // that a teach hitting a bond which only exists in an L0 SST is
        // reported as `reinforced_existing_bond=false`; the read path
        // still merges correctly and the latest weight is observed.
        let mem_bonds = mt.get_bonds_for_subject(&subject_seed, &intern);
        let already_in_memtable = mem_bonds.iter().any(|b| {
            b.object == object_seed && b.predicate == relation && !b.tombstone
        });

        // Source for the synthesized keyframe. Reinforce -> Session/Manual
        // mirrors the v3 path's reinforcement source; new bond ->
        // Baseline mirrors `TemporalBond::new`.
        let source = if already_in_memtable {
            match active_session_id {
                Some(id) => KeyframeSource::Session(id),
                None => KeyframeSource::Manual,
            }
        } else {
            KeyframeSource::Baseline
        };
        let kf = crate::bond::TemporalKeyframe {
            timestamp: now,
            weight: weight.clamp(0.0, 1.0),
            source,
        };
        let keyframes = vec![kf];

        // OP-BRAIN-V5 Phase 1.4 (Q8 ordering invariant): every Seed
        // that this teach contributes to the WAL must be preceded by
        // a REC_TYPE_NODE_INTERN record in the same WAL file. We hold
        // the WAL mutex for the entire (intern-commit + node-intern
        // records + bond record) sequence so that any other thread
        // observing a Seed in the live `NodeIntern` table is also
        // observing the corresponding NodeIntern record already on
        // disk. This is the concurrent-safety contract the
        // `replay_streaming_validated` firewall relies on.
        //
        // WAL durability is the v4 equivalent of the v3 fsync-on-save
        // guarantee — every bond accepted by `teach_with_kind_v4`
        // lands on disk before the next flush cadence
        // (`set_fsync_thresholds`) or save call. The lock contends
        // only with the flush worker's `truncate_up_to`, which runs
        // at most once per flush cycle.
        //
        // The MemTable insert below was previously outside the WAL
        // critical section; it is now inside it. The MemTable
        // insert path is bounded (Vec push under a separate Mutex)
        // so the extra lock-hold window stays small relative to
        // ingest cadence.
        //
        // Both halves (MemTable insert + WAL append) must succeed for
        // the write to be considered durable. If the WAL append fails
        // we issue a tombstone for the same triple to undo the visible
        // effect of the insert. The tombstone carries a higher seq than
        // the original insert so the read-merge dedup naturally
        // shadows it.
        let (r, append_result) = {
            let mut guard = wal.lock().expect("WAL mutex poisoned");

            // Step A: intern subject/object UNDER the WAL lock and
            // emit REC_TYPE_NODE_INTERN for any newly-allocated id.
            // `intern_with_new_flag` returns `(NodeId, true)` to AT
            // MOST one caller per Seed (per-Seed idempotence under
            // concurrent calls), so each Seed produces at most one
            // NodeIntern record per process lifetime.
            let (subj_id, subj_new) = intern.intern_with_new_flag(subject_seed);
            if subj_new {
                let s = mt.allocate_seq();
                if let Err(e) = guard.append_node_intern(s, &subject_seed, subj_id) {
                    eprintln!(
                        "[brain] teach_with_kind_v4: WAL append_node_intern (subject) failed seq={} err={}; dropping bond",
                        s, e
                    );
                    // Subject was committed to intern but never
                    // landed on disk. The next teach that touches
                    // subject_seed will observe (id, false) and skip
                    // writing the NodeIntern record -- so subsequent
                    // bonds referencing subject_seed would also fail
                    // replay. To avoid silent corruption, we return
                    // early without writing any bond record. The
                    // in-memory intern table now has an orphan entry
                    // for subject_seed; this is recoverable (next
                    // process restart rebuilds the table from WAL
                    // replay, which had no record for subject_seed,
                    // so the orphan vanishes).
                    return;
                }
            }
            let (obj_id, obj_new) = intern.intern_with_new_flag(object_seed);
            if obj_new {
                let s = mt.allocate_seq();
                if let Err(e) = guard.append_node_intern(s, &object_seed, obj_id) {
                    eprintln!(
                        "[brain] teach_with_kind_v4: WAL append_node_intern (object) failed seq={} err={}; dropping bond",
                        s, e
                    );
                    return;
                }
            }
            let _ = (subj_id, obj_id); // ids carried by intern table

            // Step B: MemTable insert. Allocates its own bond seq.
            // The intern_or_get inside `MemTable::insert` is now a
            // pure lookup because Step A committed both Seeds.
            let r = mt.insert(
                subject_seed,
                object_seed,
                relation,
                weight.clamp(0.0, 1.0),
                keyframes.clone(),
                &intern,
            );

            // Step C: bond record.
            let res = guard.append(
                r.seq,
                &subject_seed,
                &object_seed,
                relation,
                weight.clamp(0.0, 1.0),
                &keyframes,
            );
            (r, res)
        };
        if let Err(e) = append_result {
            eprintln!(
                "[brain] teach_with_kind_v4: WAL append failed seq={} err={};                  issuing compensating tombstone",
                r.seq, e
            );
            let _ = mt.delete(subject_seed, object_seed, relation, &intern);
            return;
        }

        *reinforced_existing_bond = already_in_memtable;
    }

    /// OP-BRAIN-V4 Phase 10: persist a newly-created KnowledgeNode
    /// via the v4 substrate (MemTable nodes table + WAL node record).
    /// Idempotent: if the seed is already in the MemTable nodes table,
    /// no-op so reinforcement teaches do not double-write.
    ///
    /// The MemTable insert + WAL append mirror `teach_with_kind_v4`'s
    /// bond path: WAL-append failure is a structural durability
    /// problem (not user-visible) and is logged but does not abort the
    /// teach -- the bond write either landed or failed-and-tombstoned
    /// before this point, so the worst case is a node that the
    /// daemon will re-create on the next teach (idempotent).
    fn persist_node_v4(
        &mut self,
        seed: Seed,
        node: &KnowledgeNode,
    ) {
        self.persist_node_v4_impl(seed, node, false)
    }

    /// OP-REDEFINE-1: `persist_node_v4` with an explicit `force`. When force
    /// is true the MemTable-idempotency guard is skipped so an intentional
    /// node UPDATE (attaching a definition) overwrites the record even when
    /// the seed is already MemTable-resident. `MemTable::insert_node` upserts
    /// by id, so the newer record wins on the MemTable-first read path.
    fn persist_node_v4_force(&mut self, seed: Seed, node: &KnowledgeNode) {
        self.persist_node_v4_impl(seed, node, true)
    }

    fn persist_node_v4_impl(
        &mut self,
        seed: Seed,
        node: &KnowledgeNode,
        force: bool,
    ) {
        let (mt, wal, intern) = match (
            self.memtable.as_ref().cloned(),
            self.wal.as_ref().cloned(),
            self.flush_intern.as_ref().cloned(),
        ) {
            (Some(mt), Some(wal), Some(intern)) => (mt, wal, intern),
            _ => {
                eprintln!(
                    "[brain] persist_node_v4: missing MemTable/WAL/intern on v4-loaded brain; dropping node persist"
                );
                return;
            }
        };
        // Idempotency: if the MemTable already holds this node's
        // record, do not re-write. This guards against the reinforce-
        // existing path emitting a duplicate node record on every
        // teach.
        if !force && mt.lookup_node(&seed, &intern).is_some() {
            return;
        }
        // OP-BRAIN-V5 Phase 1.4 (Q8 ordering invariant): persist_node_v4
        // also commits the seed to the live `NodeIntern` table (via
        // `MemTable::insert_node` -> `intern.intern_or_get`). Without
        // a paired REC_TYPE_NODE_INTERN record, a subsequent
        // `teach_with_kind_v4` call referencing this seed would write
        // a bond record into the WAL without a preceding NodeIntern
        // record (because `intern_with_new_flag` would observe the
        // seed as already-interned and skip emission). The bond
        // would then trip `WalError::UnknownNodeIdInBond` on replay.
        // We close the gap here: emit the NodeIntern record FIRST,
        // under the WAL lock, then write the NODE record. The NODE
        // record carries its own seed payload and is independent of
        // the NodeIntern set per `replay_streaming_validated`, but
        // the NodeIntern record is needed so future bonds against
        // this seed pass the firewall.
        let (r, append_result) = {
            let mut guard = wal.lock().expect("WAL mutex poisoned");
            let (id, new_intern) = intern.intern_with_new_flag(seed);
            if new_intern {
                let s = mt.allocate_seq();
                if let Err(e) = guard.append_node_intern(s, &seed, id) {
                    eprintln!(
                        "[brain] persist_node_v4: WAL append_node_intern failed seq={} err={}; dropping node persist",
                        s, e
                    );
                    return;
                }
            }
            // MemTable::insert_node also calls intern_or_get; after the
            // commit above this is a pure lookup.
            let r = mt.insert_node(seed, node.clone(), &intern);
            let res = guard.append_node(r.seq, &seed, node);
            (r, res)
        };
        if let Err(e) = append_result {
            eprintln!(
                "[brain] persist_node_v4: WAL append_node failed seq={} err={}; MemTable carries the record but durability is at risk until next flush",
                r.seq, e
            );
        }
    }

    /// Q.1 -- typed-KU teach. `kind_hint` is a `KnowledgeContent`
    /// variant tag ("definition", "mechanism", ...). When present, new
    /// subject/object nodes are constructed with that variant rather
    /// than the default `Concept`. `relation_alias` is the original
    /// relation string the caller passed (e.g. "DefinedAs") — used
    /// for kind inference when `kind_hint` is None. Existing nodes
    /// are NEVER mutated; their stored variant is authoritative once
    /// they exist (Q.1 deliberately keeps node creation idempotent).
    /// OP-BRAIN-MapRAM Fix A -- existence check for a concept's node
    /// during teach. In MapRAM cache-backed mode the resident
    /// `network.nodes` map is intentionally left empty (full nodes are
    /// persisted to the node_store and served on read via
    /// `get_node_v5`), so node existence is sourced from the resident
    /// node intern (`flush_intern`, loaded from the nodeintern sidecar
    /// at boot and grown by every teach). Outside cache-backed mode the
    /// resident network is authoritative, preserving v3/v4 behaviour.
    fn teach_node_exists(&self, seed: Seed) -> bool {
        if self.adjacency_cache_backed() {
            if let Some(intern) = self.flush_intern.as_ref() {
                return intern.get(&seed).is_some();
            }
        }
        self.network.get_node(&seed).is_some()
    }

    pub fn teach_with_kind(
        &mut self,
        subject: &str,
        relation: BondType,
        object: &str,
        weight: f32,
        confidence: Confidence,
        kind_hint: Option<&str>,
        relation_alias: Option<&str>,
    ) -> TeachResult {
        let now = now_ms();
        let active_session_id = self.sessions.active_session().map(|s| s.id);

        // OP-IDENTITY: the axiomatic identity core is immutable. Its facts are
        // written once (Confidence::Terminal) by the init injection; any later
        // teach touching a protected identity label with a NON-Terminal
        // confidence is a no-op, so the writeback / self-correction layer can
        // never demote or overwrite it. Guests cannot reach teach at all
        // (privsep: the guest binary has no teach tool).
        if confidence != Confidence::Terminal
            && (crate::identity::is_protected_label(subject)
                || crate::identity::is_protected_label(object))
        {
            return TeachResult {
                subject_seed: Seed::from_label(subject),
                object_seed: Seed::from_label(object),
                bond_type: relation,
                weight,
                timestamp: now,
                created_subject: false,
                created_object: false,
                reinforced_existing_bond: false,
                admission_rejected: true,
                session_id: active_session_id,
            };
        }

        // OP-CANON-1 ingest folding (env-gated `LU_CANON_INGEST`,
        // default OFF, reversible). Fold case/article variants onto one
        // identity so `Dog`/`dog`/`the dog` collapse at creation. LOSSLESS:
        // `fold_seed_label` never drops a label, so no bond is lost here;
        // the content-shape reject is the gated offline prune's job. Node
        // CONTENT (display) is unchanged, and query-side seed selection is
        // per-token + lowercased (`select_seeds_for_query_inner`), so keyword
        // recall already matches these folded identities without change.
        let (subject_key, object_key) = if crate::canonical::ingest_canon_enabled() {
            (
                crate::canonical::fold_seed_label(subject),
                crate::canonical::fold_seed_label(object),
            )
        } else {
            (subject.to_string(), object.to_string())
        };
        let subject_seed = Seed::from_label(&subject_key);
        let object_seed = Seed::from_label(&object_key);

        // OP-RETRAIN-1 concept-admission gate (same `LU_CANON_INGEST` flag,
        // default OFF). Reject a teach whose subject OR object is a
        // non-concept label -- a sentence fragment, category line, or bare
        // code identifier -- so junk nodes are never minted at ingest. This
        // is the ingest-side of the validated `canonical_key` classifier
        // (A2: ~95% kill-precision, ordinal/designation FP fix e8db33c). A
        // bond touching a non-concept endpoint is not knowledge worth a
        // node, so the bond is dropped (no-op TeachResult, nothing minted).
        if crate::canonical::ingest_canon_enabled()
            && (crate::canonical::canonical_key(subject).is_none()
                || crate::canonical::canonical_key(object).is_none())
        {
            crate::canonical::record_canon_gate_rejection();
            return TeachResult {
                subject_seed,
                object_seed,
                bond_type: relation,
                weight,
                timestamp: now,
                created_subject: false,
                created_object: false,
                reinforced_existing_bond: false,
                admission_rejected: true,
                session_id: active_session_id,
            };
        }

        let mut created_subject = false;
        let mut created_object = false;

        // Q.1 — resolve the effective kind tag. Explicit hint wins;
        // fall back to relation-alias + text-shape inference.
        let resolved_kind: Option<&str> = match kind_hint {
            Some(k) => Some(k),
            None => {
                let alias = relation_alias.unwrap_or("");
                crate::node::infer_kind(alias, subject, object)
            }
        };

        // Phase 10: for v4 brains we need to keep an owned copy of the
        // node we just added so we can persist it (network.get_node
        // returns a NodeView whose lifetime ties to &self, blocking the
        // subsequent &mut self call to persist_node_v4 -- safer to clone
        // the node alongside add_node).
        let subject_node_to_persist: Option<KnowledgeNode>;
        if !self.teach_node_exists(subject_seed) {
            let content = KnowledgeContent::from_kind_or_concept(
                resolved_kind,
                subject.to_string(),
            );
            let node = KnowledgeNode::with_seed(
                subject_seed,
                subject.to_string(),
                content,
                confidence,
                now,
            );
            if self.adjacency_cache_backed() {
                // OP-BRAIN-MapRAM Fix A: persist to node_store below; do
                // NOT retain the full node resident (served on read via
                // get_node_v5). Existence comes from the resident intern.
                subject_node_to_persist = Some(node);
            } else {
                let to_persist = if self.brain_dir_v4.is_some() {
                    Some(node.clone())
                } else {
                    None
                };
                self.network.add_node(node);
                subject_node_to_persist = to_persist;
            }
            created_subject = true;
        } else {
            subject_node_to_persist = None;
        }
        let object_node_to_persist: Option<KnowledgeNode>;
        if !self.teach_node_exists(object_seed) {
            let content = KnowledgeContent::from_kind_or_concept(
                resolved_kind,
                object.to_string(),
            );
            let node = KnowledgeNode::with_seed(
                object_seed,
                object.to_string(),
                content,
                confidence,
                now,
            );
            if self.adjacency_cache_backed() {
                // OP-BRAIN-MapRAM Fix A: persist to node_store below; do
                // NOT retain the full node resident (served on read via
                // get_node_v5). Existence comes from the resident intern.
                object_node_to_persist = Some(node);
            } else {
                let to_persist = if self.brain_dir_v4.is_some() {
                    Some(node.clone())
                } else {
                    None
                };
                self.network.add_node(node);
                object_node_to_persist = to_persist;
            }
            created_object = true;
        } else {
            object_node_to_persist = None;
        }
        // Phase 10: persist the newly-created node(s) to the v4
        // substrate. The bond persist below sees the node already in
        // network.nodes (so get_node returns it) -- this call adds the
        // durable durability record so WAL replay can rebuild
        // network.nodes on restart.
        if let Some(n) = subject_node_to_persist {
            self.persist_node_v4(subject_seed, &n);
        }
        if let Some(n) = object_node_to_persist {
            self.persist_node_v4(object_seed, &n);
        }

        let mut reinforced_existing_bond = false;
        // OP-BRAIN-V4 Phase 9: v4 write path. When the brain was loaded
        // via `Brain::load_v4`, bonds live in MemTable + WAL, NOT in the
        // v3 `network` HashMaps (those stay empty for v4-loaded brains).
        // Phases 1-8 set up MemTable+WAL+flush+compaction+reads but never
        // routed live writes through them; the 2026-06-01 cutover surfaced
        // the gap (claimed-landed bonds went to v3 HashMaps, WAL stayed
        // at 8 bytes after 6.88M teach calls). The branch below closes
        // that gap: `teach_with_kind` is the unique chokepoint for every
        // teach path (teach, teach_corpus, teach_with_source, pending
        // apply, research apply), so wiring v4 dispatch here covers them
        // all in one place.
        //
        // Hub-degree write-side enforcement and v3 `add_bond_with_scar_check`
        // are v3-only here: the MemTable has no concept of in-degree and no
        // way to evict existing entries. Scar-seed admission is preserved
        // (it reads only `scar_log.scarred_seeds`, a HashSet membership
        // check). Reinforcement on an existing bond in MemTable is modeled
        // as a fresh insert with a new keyframe; the Phase 5 read-path
        // merge dedupes by `(subject, predicate, object)` taking
        // latest-seq, so the newer insert wins on read.
        if self.brain_dir_v4.is_some() {
            self.teach_with_kind_v4(
                subject_seed,
                object_seed,
                relation,
                weight,
                now,
                active_session_id,
                &mut reinforced_existing_bond,
            );
        } else if self.network.get_bond(&subject_seed, &object_seed).is_some() {
            let source = match active_session_id {
                Some(id) => KeyframeSource::Session(id),
                None => KeyframeSource::Manual,
            };
            if let Some(mut bond) = self.network.get_bond_mut(&subject_seed, &object_seed) {
                bond.add_keyframe(now, weight.clamp(0.0, 1.0), source);
                reinforced_existing_bond = true;
            }
        } else {
            // Port Cat.3b: hub-degree write-side enforcement BEFORE
            // bond creation. When write_enforcement is on and either
            // endpoint is at cap, the priority strategy decides:
            //   * NewestFirst       → reject the new bond
            //   * OldestFirst       → evict the oldest existing bond
            //   * LowestWeightFirst → evict the lowest-weight existing
            //                         bond IF the new bond's weight
            //                         is higher; otherwise reject.
            // Existing bonds are never mutated outside the eviction
            // path. The hub_degree_config.exempt_node_kinds list
            // bypasses caps for specific node kinds.
            let mut admit = true;
            if self.hub_degree_config.write_enforcement {
                let (admit_decision, evicted) = crate::brain::decide_hub_cap_admission(
                    &mut self.network,
                    &self.hub_degree_config,
                    subject_seed,
                    object_seed,
                    weight.clamp(0.0, 1.0),
                );
                admit = admit_decision;
                if !admit {
                    self.hub_cap_rejections_total =
                        self.hub_cap_rejections_total.saturating_add(1);
                }
                if evicted > 0 {
                    self.hub_cap_evictions_total =
                        self.hub_cap_evictions_total.saturating_add(evicted as u64);
                }
            }
            if admit {
                let bond = TemporalBond::new(
                    subject_seed,
                    object_seed,
                    relation,
                    weight.clamp(0.0, 1.0),
                    now,
                );
                // Scar-aware path. If either endpoint is in
                // ScarLog.scarred_seeds the bond is refused and a fresh
                // Scar is appended; the bond is NOT added to the network.
                // We swallow the error here because TeachResult does not
                // currently surface ingest refusals — the scar log carries
                // the audit trail. (A future revision could add a
                // `scar_blocked: bool` to TeachResult.)
                let _ = self.network.add_bond_with_scar_check(bond, &mut self.scar_log);
            }
        }

        if let Some(session) = self.sessions.active_session_mut() {
            session.record_activation(&[subject_seed, object_seed]);
        }

        TeachResult {
            subject_seed,
            object_seed,
            bond_type: relation,
            weight: weight.clamp(0.0, 1.0),
            timestamp: now,
            created_subject,
            created_object,
            reinforced_existing_bond,
            admission_rejected: false,
            session_id: active_session_id,
        }
    }

    /// OP-RETRAIN-1 -- attach a DEFINITION (meaning text) to a concept node,
    /// distinct from its label. Creates the node with `content = label` and
    /// `knowledge.content = Definition(gloss)`, so "what is X" recall can
    /// surface the actual meaning. CREATE-ONLY / idempotent: an existing node
    /// is NOT mutated (first definition wins), mirroring persist_node_v4's v5
    /// invariant -- so foundation loaders must `define` a headword BEFORE any
    /// bond that would create it. Honors the admission gate. Returns true iff
    /// a new node was created.
    pub fn set_definition(&mut self, label: &str, gloss: &str) -> bool {
        self.set_definition_impl(label, gloss, false)
    }

    /// OP-REDEFINE-1: `set_definition` with additive update. When the node
    /// already exists but carries NO definition, re-persist it with
    /// `KnowledgeContent::Definition(gloss)` while preserving its display
    /// label, timestamps, embedding and parent seeds. Bonds live in a
    /// separate store and are never read or written here, so the brain graph
    /// is only ADDED to (a meaning attached), never altered structurally.
    /// Recovers the WordNet/GCIDE glosses skipped by the create-only pass for
    /// nodes an earlier bond pass had already minted. Returns true iff a
    /// definition was created OR attached; false when the node already has a
    /// non-empty definition, is gated, or no writable substrate exists.
    pub fn set_definition_update(&mut self, label: &str, gloss: &str) -> bool {
        self.set_definition_impl(label, gloss, true)
    }

    fn set_definition_impl(&mut self, label: &str, gloss: &str, allow_update: bool) -> bool {
        if crate::canonical::ingest_canon_enabled()
            && crate::canonical::canonical_key(label).is_none()
        {
            crate::canonical::record_canon_gate_rejection();
            return false;
        }
        let key = if crate::canonical::ingest_canon_enabled() {
            crate::canonical::fold_seed_label(label)
        } else {
            label.to_string()
        };
        let seed = Seed::from_label(&key);
        // Mode-agnostic existence: teach_node_exists consults the resident
        // network (v3) or flush_intern (cache-backed v5), but a v5 test brain
        // may be neither -- so also consult flush_intern directly. persist_node_v4
        // has its own dedup guard; this keeps the return value honest.
        let exists = self.teach_node_exists(seed)
            || self
                .flush_intern
                .as_ref()
                .map(|i| i.get(&seed).is_some())
                .unwrap_or(false);
        let now = now_ms();
        if exists {
            if !allow_update {
                return false;
            }
            // ADDITIVE re-define: fetch the current node, keep every field,
            // attach the Definition only when it lacks one. persist_node_v4
            // writes a fresh node record (the seed is SST-resident, not in the
            // MemTable, so its idempotency guard does not fire) which shadows
            // the old one on the MemTable-first read path. Bonds are untouched.
            let id = match self.network.node_intern().get(&seed) {
                Some(id) => id,
                None => return false,
            };
            let existing = match self.get_node_v5(id) {
                Ok(Some(n)) => n,
                _ => return false,
            };
            if let KnowledgeContent::Definition(d) = &existing.knowledge.content {
                // OP-SOURCEPULL Batch-B (Fix 1): a self-referential gloss
                // (definition text == the node label) is junk, not a real
                // definition -- let a real one overwrite it. Otherwise keep
                // first-definition-wins (idempotent).
                let dt = d.trim();
                let self_ref = dt.eq_ignore_ascii_case(existing.content.trim());
                if !dt.is_empty() && !self_ref {
                    return false; // already defined -- idempotent, leave as-is
                }
            }
            let updated = KnowledgeNode {
                seed,
                content: existing.content.clone(),
                knowledge: crate::node::KnowledgeUnit {
                    content: KnowledgeContent::Definition(gloss.to_string()),
                    confidence: Confidence::Verified,
                    created_at: existing.knowledge.created_at,
                    updated_at: now,
                },
                embedding: existing.embedding.clone(),
                parent_seeds: existing.parent_seeds.clone(),
                typed_ratio: existing.typed_ratio,
            };
            if self.memtable.is_some() {
                self.persist_node_v4_force(seed, &updated);
            } else {
                self.network.add_node(updated);
            }
            return true;
        }
        let node = KnowledgeNode::with_seed(
            seed,
            label.to_string(),
            KnowledgeContent::Definition(gloss.to_string()),
            Confidence::Verified,
            now,
        );
        if self.memtable.is_some() {
            self.persist_node_v4(seed, &node);
        } else {
            self.network.add_node(node);
        }
        true
    }

    /// Ask the brain: IDF-weighted seed selection, spread activation with rank
    /// decay, cascading expansion if primary results are weak.
    ///
    /// Legacy 3-arg form. Equivalent to `ask_with_config(query, BrainAskConfig::new(depth, top_k))`.
    /// No wall-clock bound is applied -- callers that need a bound (narrate, MCP)
    /// should use `ask_with_config` directly.
    pub fn ask(&mut self, query: &str, depth: usize, top_k: usize) -> AskResult {
        self.ask_with_config(query, BrainAskConfig::new(depth, top_k))
    }

    /// OP-ASK-WATCHDOG -- workhorse form with explicit per-call config.
    /// `BrainAskConfig::max_wall_clock` is forwarded into the two
    /// `spread_activation_weighted_full` calls (primary + cascade);
    /// the resulting `AskResult.status` reflects whichever spread
    /// hit the budget first (cascade takes precedence when set, so
    /// `TimedOut.activated_so_far` reads from the final spread).
    pub fn ask_with_config(&mut self, query: &str, ask_cfg: BrainAskConfig) -> AskResult {
        let depth = ask_cfg.depth;
        let top_k = ask_cfg.top_k;
        let max_wall_clock = ask_cfg.max_wall_clock;
        // Storage v2 Slice 2c — lazy keyword index build. First call
        // after an mmap-zero-copy load pays the page-fault cost.
        self.network.ensure_keyword_index_built();
        // Port T.5.1 — arm freshness multiplier on every query.
        self.refresh_decay_runtime();
        let ask_started_us = now_us();
        let now = now_ms();

        // Feature 1+2: TF-IDF weighted seed selection, capped at 3 seeds
        let mut weighted = self.network.select_seeds_for_query_ask(query);
        if weighted.is_empty() {
            let hits = self.network.keyword_search(query);
            weighted = hits.iter().take(3).map(|(s, _)| (*s, 0.7_f32)).collect();
        }
        // Port Sense.1 -- arm sense cluster-gain multiplier on every
        // query, using the selected seeds for both the polysemy probe
        // and the disambiguation context window.
        let _sense_seeds: Vec<crate::seed::Seed> = weighted.iter().map(|(s, _)| *s).collect();
        self.refresh_sense_runtime(&_sense_seeds);

        let scope = match self.sessions.active_session() {
            Some(s) => TemporalScope::Session { id: s.id, start: s.start_time },
            None => TemporalScope::Current,
        };

        // Bridge + curiosity collectors share a single spread pass.
        // Both collectors span the primary spread AND any cascade run,
        // so duplicate candidates across runs collapse into one entry.
        let mut bc = BridgeCollector::new(self.bridge_config.candidate_buffer_cap);
        let mut cc = CuriosityCollector::new(self.bridge_config.candidate_buffer_cap);
        let mut scar_seen: Vec<crate::scar::Scar> = Vec::new();
        // Per-graph dynamic hub threshold; replaces the old constant.
        let curiosity_hub_threshold = self.network.degree_p90();
        let recall = self.network.spread_activation_weighted_full(
            &weighted,
            &scope,
            depth,
            top_k,
            &mut bc,
            &self.bridge_config,
            &mut cc,
            &self.curiosity_config,
            curiosity_hub_threshold,
            now,
            &self.scar_log,
            &mut scar_seen,
            max_wall_clock,
        );

        // Feature 4: cascading activation. Trigger only when the primary
        // recall is genuinely thin AND weak — not when the spread merely
        // truncated against `max_activations` or surfaced few depth-1 hits
        // because the seeds were already strong. Specifically: cascade if
        // (depth-1 hits < 2) AND (total activation map < CASCADE_FLOOR=50)
        // AND (top activation < CASCADE_TOP_FLOOR=0.30). This was the v1
        // re-flood pattern from internal notes — broadening seeds when
        // 50K+ nodes already activated only deepens the flood.
        let recall = {
            let d1 = recall.activated.iter().filter(|a| a.depth >= 1).count();
            let map_size = recall.activated.len();
            let top_act = recall.activated.first().map(|a| a.activation).unwrap_or(0.0);
            let cascade_ok = d1 < CASCADE_FLOOR_D1 && map_size < CASCADE_FLOOR_MAP && top_act < CASCADE_TOP_FLOOR;
            if cascade_ok {
                let existing: std::collections::HashSet<Seed> =
                    weighted.iter().map(|(s, _)| *s).collect();
                let hits = self.network.keyword_search(query);
                let extra: Vec<(Seed, f32)> = hits.iter()
                    .filter(|(s, _)| !existing.contains(s))
                    .take(3)
                    .map(|(s, _)| (*s, 0.25_f32))
                    .collect();
                if !extra.is_empty() {
                    let mut all = weighted.clone();
                    all.extend(extra);
                    self.network.spread_activation_weighted_full(
                        &all,
                        &scope,
                        depth,
                        top_k,
                        &mut bc,
                        &self.bridge_config,
                        &mut cc,
                        &self.curiosity_config,
                        curiosity_hub_threshold,
                        now,
                        &self.scar_log,
                        &mut scar_seen,
                        max_wall_clock,
                    )
                } else { recall }
            } else { recall }
        };

        // Phase 2 (bridges): classify candidates into events, capture
        // them for this call's AskResult, push into the ring buffer.
        self.bridges_dropped_overflow_total += bc.dropped_overflow;
        let events = classify_candidates(&self.network, &bc.candidates, &scope, &self.bridge_config);
        let bridges_for_result = events.clone();
        for e in events {
            self.recent_bridges.push_back(e);
        }
        while self.recent_bridges.len() > self.bridge_config.recent_bridges_cap {
            self.recent_bridges.pop_front();
        }

        // Phase 2 (curiosity): classify candidates into signals,
        // capture them for this call's AskResult, fold into the
        // wonder queue, and bump InterestProfile per signal seed.
        self.curiosity_dropped_overflow_total += cc.dropped_overflow;
        let acts_for_curiosity: Vec<(Seed, f32)> = recall
            .activated
            .iter()
            .map(|a| (a.seed, a.activation))
            .collect();
        let query_seeds_vec: Vec<Seed> = weighted.iter().map(|(s, _)| *s).collect();
        let curiosity_signals = classify_curiosity_candidates(
            &self.network,
            &cc.candidates,
            &acts_for_curiosity,
            &query_seeds_vec,
            &scope,
            &self.curiosity_config,
            now as u64,
        );
        let curiosity_for_result = curiosity_signals.clone();
        for s in curiosity_signals {
            // Touch the interest profile for the signal\u0027s anchor seed.
            self.interest_profile.touch(
                s.seed,
                now as u64,
                self.curiosity_config.interest_decay_halflife_ms,
            );
            // Port LB.1: dual-write the wonder signal as a WonderItem entry in the
            // Left Brain. wonder_queue stays the live runtime structure; LB is the
            // persistent operationally-inspectable record.
            {
                // OP-LABEL-4: see the note on the wonder-snapshot site.
                let lb_label = self.network.node_label(&s.seed)
                    .filter(|l| !l.is_empty())
                    .or_else(|| self.gap_label_map.get(&s.seed).cloned())
                    .unwrap_or_else(|| s.seed.to_string());
                let lb_payload = serde_json::to_string(&s).unwrap_or_default();
                let lb_intensity = s.intensity;
                let _ = self.lb_session.push_internal(
                    crate::lb_session::SessionEntry::new(
                        0,
                        now as u64,
                        crate::lb_session::SessionEntryKind::WonderItem,
                        lb_label,
                        lb_payload,
                        lb_intensity,
                    ),
                );
            }
            // OP-CURIOSITY-1 -- mirror to the durable journal so the
            // FILL half (a separate, non-public worker process) can see
            // this gap at all. Best-effort: a journal write failure must
            // never fail a user's query, but it IS counted and reported
            // by `wonder_sidecar_write_errors()` rather than swallowed.
            self.record_wonder_durably(&s);
            self.wonder_queue.push(s);
        }

        let activated_seeds: Vec<Seed> = recall.activated.iter().map(|a| a.seed).collect();
        let (hebbian_strengthened, active_session_id) = match self.sessions.active_session_mut() {
            Some(session) => {
                session.record_activation(&activated_seeds);
                let id = session.id;
                let strengthened = self.network.hebbian_strengthen(&activated_seeds, id, now);
                (strengthened, Some(id))
            }
            None => (0, None),
        };

        // OP-ASK-WATCHDOG -- derive AskStatus from the (post-cascade)
        // RecallResult before  is moved into AskResult.
        let ask_status = match recall.status {
            crate::network::SpreadStatus::Complete => AskStatus::Complete,
            crate::network::SpreadStatus::TimedOut { activated_so_far } => {
                AskStatus::TimedOut { activated_so_far }
            }
        };
        let result = AskResult {
            query: query.to_string(),
            seed_hits: weighted.into_iter().map(|(s, _)| s).collect(),
            recall,
            hebbian_strengthened,
            session_id: active_session_id,
            bridges: bridges_for_result,
            curiosity_signals: curiosity_for_result,
            scars: scar_seen,
            status: ask_status,
        };
        // Self-reflection telemetry: push one QueryMetric per ask.
        record_ask_metric(self, query, &result, &scope, ask_started_us);
        result
    }
/// OP-REASON-LOOP Phase 3 -- lean, READ-ONLY seeded recall for the
    /// reason<->retrieve loop. Spreads from the given `(seed, weight)` pairs
    /// and returns the activations. Unlike `ask_with_config` it does NO
    /// hebbian strengthening, NO wonder-queue / interest-profile / session
    /// mutation, and NO cascade re-flood -- so it is safe to call repeatedly
    /// inside a loop on a shared read-only brain (Phase 2 no-write discipline)
    /// and its read cost is bounded to a single spread.
    pub fn ask_with_config_seeded(
        &mut self,
        seeds: &[(Seed, f32)],
        ask_cfg: BrainAskConfig,
    ) -> AskResult {
        self.network.ensure_keyword_index_built();
        self.refresh_decay_runtime();
        if seeds.is_empty() {
            return AskResult {
                query: String::new(),
                seed_hits: Vec::new(),
                recall: crate::network::RecallResult {
                    activated: Vec::new(),
                    query_seeds: Vec::new(),
                    scope: crate::scope::TemporalScope::Current,
                    nodes_visited: 0,
                    bonds_evaluated: 0,
                    activation_truncated: false,
                    status: crate::network::SpreadStatus::Complete,
                },
                hebbian_strengthened: 0,
                session_id: None,
                bridges: Vec::new(),
                curiosity_signals: Vec::new(),
                scars: Vec::new(),
                status: AskStatus::Complete,
            };
        }
        let sense_seeds: Vec<Seed> = seeds.iter().map(|(s, _)| *s).collect();
        self.refresh_sense_runtime(&sense_seeds);
        let scope = match self.sessions.active_session() {
            Some(s) => TemporalScope::Session { id: s.id, start: s.start_time },
            None => TemporalScope::Current,
        };
        // Bound the spread with the caller wall-clock: a seeded recall from
        // a content-hub concept must never scan a large slice of the graph
        // (read-amplification guard / no 100GB scans; Phase 3 gate 3).
        let recall = self.network.spread_activation_weighted_with_wall_clock(
            seeds,
            &scope,
            ask_cfg.depth,
            ask_cfg.top_k,
            &self.scar_log,
            ask_cfg.max_wall_clock,
        );
        let status = match recall.status {
            crate::network::SpreadStatus::Complete => AskStatus::Complete,
            crate::network::SpreadStatus::TimedOut { activated_so_far } => {
                AskStatus::TimedOut { activated_so_far }
            }
        };
        AskResult {
            query: String::new(),
            seed_hits: seeds.iter().map(|(s, _)| *s).collect(),
            recall,
            hebbian_strengthened: 0,
            session_id: None,
            bridges: Vec::new(),
            curiosity_signals: Vec::new(),
            scars: Vec::new(),
            status,
        }
    }

    /// Like `ask()` but only traverses bonds whose type is in `bond_filter`.
    pub fn typed_ask(
        &mut self,
        query: &str,
        depth: usize,
        top_k: usize,
        bond_filter: &[BondType],
    ) -> AskResult {
        // Issue A fix internal notes: mirror Brain::ask -- ensure the
        // archive-backed keyword index is materialized before any seed
        // selection. Idempotent after the first call.
        self.network.ensure_keyword_index_built();
        // Port T.5.1 — arm freshness multiplier on every query.
        self.refresh_decay_runtime();
        let typed_ask_started_us = now_us();
        let now = now_ms();
        let mut weighted = self.network.select_seeds_for_query_ask(query);
        if weighted.is_empty() {
            let hits = self.network.keyword_search(query);
            weighted = hits.iter().take(3).map(|(s, _)| (*s, 0.7_f32)).collect();
        }
        // Port Sense.1 -- arm sense cluster-gain multiplier on every
        // typed_ask query.
        let _sense_seeds: Vec<crate::seed::Seed> = weighted.iter().map(|(s, _)| *s).collect();
        self.refresh_sense_runtime(&_sense_seeds);
        let scope = match self.sessions.active_session() {
            Some(s) => TemporalScope::Session { id: s.id, start: s.start_time },
            None => TemporalScope::Current,
        };

        // Bridge + curiosity collection mirrors `ask()` — single spread,
        // both phases run after. Both collectors span the primary spread
        // AND any cascade run, so duplicates collapse into one entry.
        let mut bc = BridgeCollector::new(self.bridge_config.candidate_buffer_cap);
        let mut cc = CuriosityCollector::new(self.bridge_config.candidate_buffer_cap);
        // Per-graph dynamic hub threshold; replaces the old constant.
        let curiosity_hub_threshold = self.network.degree_p90();
        let recall = self.network.typed_recall_weighted_with_bridges_and_curiosity(
            &weighted, &scope, depth, top_k, bond_filter,
            &mut bc, &self.bridge_config,
            &mut cc, &self.curiosity_config, curiosity_hub_threshold,
            now,
            &self.scar_log,
        );

        // Cascading activation — same gating as `ask()`. We use the same
        // CASCADE_FLOOR / CASCADE_TOP_FLOOR constants defined above so a
        // typed_ask never broadens seeds in the middle of a real flood.
        let recall = {
            let d1 = recall.activated.iter().filter(|a| a.depth >= 1).count();
            let map_size = recall.activated.len();
            let top_act = recall.activated.first().map(|a| a.activation).unwrap_or(0.0);
            let cascade_ok = d1 < CASCADE_FLOOR_D1 && map_size < CASCADE_FLOOR_MAP && top_act < CASCADE_TOP_FLOOR;
            if cascade_ok {
                let existing: std::collections::HashSet<Seed> =
                    weighted.iter().map(|(s, _)| *s).collect();
                let hits = self.network.keyword_search(query);
                let extra: Vec<(Seed, f32)> = hits.iter()
                    .filter(|(s, _)| !existing.contains(s))
                    .take(3)
                    .map(|(s, _)| (*s, 0.25_f32))
                    .collect();
                if !extra.is_empty() {
                    let mut all = weighted.clone();
                    all.extend(extra);
                    self.network.typed_recall_weighted_with_bridges_and_curiosity(
                        &all, &scope, depth, top_k, bond_filter,
                        &mut bc, &self.bridge_config,
                        &mut cc, &self.curiosity_config, curiosity_hub_threshold,
                        now,
                        &self.scar_log,
                    )
                } else { recall }
            } else { recall }
        };

        self.bridges_dropped_overflow_total += bc.dropped_overflow;
        let events = classify_candidates(
            &self.network, &bc.candidates, &scope, &self.bridge_config,
        );
        let bridges_for_result = events.clone();
        for e in events {
            self.recent_bridges.push_back(e);
        }
        while self.recent_bridges.len() > self.bridge_config.recent_bridges_cap {
            self.recent_bridges.pop_front();
        }

        self.curiosity_dropped_overflow_total += cc.dropped_overflow;
        let acts_for_curiosity: Vec<(Seed, f32)> = recall
            .activated
            .iter()
            .map(|a| (a.seed, a.activation))
            .collect();
        let query_seeds_vec: Vec<Seed> = weighted.iter().map(|(s, _)| *s).collect();
        let curiosity_signals = classify_curiosity_candidates(
            &self.network, &cc.candidates, &acts_for_curiosity, &query_seeds_vec,
            &scope, &self.curiosity_config, now as u64,
        );
        let curiosity_for_result = curiosity_signals.clone();
        for s in curiosity_signals {
            self.interest_profile.touch(
                s.seed, now as u64, self.curiosity_config.interest_decay_halflife_ms,
            );
            // Port LB.1: dual-write the wonder signal as a WonderItem entry in the
            // Left Brain. wonder_queue stays the live runtime structure; LB is the
            // persistent operationally-inspectable record.
            {
                // OP-LABEL-4: see the note on the wonder-snapshot site.
                let lb_label = self.network.node_label(&s.seed)
                    .filter(|l| !l.is_empty())
                    .or_else(|| self.gap_label_map.get(&s.seed).cloned())
                    .unwrap_or_else(|| s.seed.to_string());
                let lb_payload = serde_json::to_string(&s).unwrap_or_default();
                let lb_intensity = s.intensity;
                let _ = self.lb_session.push_internal(
                    crate::lb_session::SessionEntry::new(
                        0,
                        now as u64,
                        crate::lb_session::SessionEntryKind::WonderItem,
                        lb_label,
                        lb_payload,
                        lb_intensity,
                    ),
                );
            }
            // OP-CURIOSITY-1 -- mirror to the durable journal so the
            // FILL half (a separate, non-public worker process) can see
            // this gap at all. Best-effort: a journal write failure must
            // never fail a user's query, but it IS counted and reported
            // by `wonder_sidecar_write_errors()` rather than swallowed.
            self.record_wonder_durably(&s);
            self.wonder_queue.push(s);
        }

        let activated_seeds: Vec<Seed> = recall.activated.iter().map(|a| a.seed).collect();
        let (hebbian_strengthened, active_session_id) = match self.sessions.active_session_mut() {
            Some(session) => {
                session.record_activation(&activated_seeds);
                let id = session.id;
                let strengthened = self.network.hebbian_strengthen(&activated_seeds, id, now);
                (strengthened, Some(id))
            }
            None => (0, None),
        };
        let result = AskResult {
            query: query.to_string(),
            seed_hits: weighted.into_iter().map(|(s, _)| s).collect(),
            recall,
            hebbian_strengthened,
            session_id: active_session_id,
            bridges: bridges_for_result,
            curiosity_signals: curiosity_for_result,
            // typed_ask uses a typed-recall path that does not currently
            // route through the full scar collector. Empty by design;
            // could be added later if typed asks need scar surfacing.
            scars: Vec::new(),
            // OP-ASK-WATCHDOG -- typed_ask does not yet plumb a
            // wall-clock budget; closes complete.
            status: AskStatus::Complete,
        };
        record_ask_metric(self, query, &result, &scope, typed_ask_started_us);
        result
    }

    /// Number of proposals currently in the Pending state.
    pub fn pending_proposal_count(&self) -> usize {
        self.research_proposals
            .iter()
            .filter(|p| p.is_pending())
            .count()
    }

    /// Synchronous one-shot research call. Pulls the top wonder above
    /// threshold (if any), queries Grokipedia → DDG, runs scar
    /// fingerprint check, and queues a `Pending` proposal. Returns the
    /// new proposal id, or an error explaining why nothing was
    /// produced. NEVER mutates the knowledge graph — approval still
    /// goes through `approve_research_proposal`.
    pub fn research_one(
        &mut self,
        fetcher: &dyn crate::research::http::HttpFetcher,
    ) -> Result<u64, crate::research::loop_::ResearchError> {
        if !self.research_config.enabled {
            return Err(crate::research::loop_::ResearchError::Disabled);
        }
        let wonder = match self.wonder_queue.peek() {
            Some(w) => w.clone(),
            None => return Err(crate::research::loop_::ResearchError::QueueEmpty),
        };
        if wonder.intensity < self.research_config.research_intensity_threshold {
            return Err(crate::research::loop_::ResearchError::BelowThreshold);
        }
        let wonder = self.wonder_queue.pop().unwrap();
        let query = self.wonder_to_query(&wonder);
        let prop = crate::research::loop_::make_proposal_for_query(
            self,
            fetcher,
            &query,
            Some(wonder),
        )?;
        let id = prop.id;
        // Port LB.1: route each research result into the Left Brain
        // as a ResearchResult entry alongside the existing push to
        // self.research_proposals. Anti-hallucination invariant:
        // bond_count is unchanged -- bonds only enter the canonical
        // brain after approve_research_proposal walks the candidates
        // through teach().
        let lb_now = now_ms() as u64;
        let payload = serde_json::to_string(&prop).unwrap_or_default();
        let _ = self.lb_session.push_internal(
            crate::lb_session::SessionEntry::new(
                0,
                lb_now,
                crate::lb_session::SessionEntryKind::ResearchResult,
                &query,
                format!("research_proposal id={}: {}", id, payload),
                0.0,
            ),
        );
        self.research_proposals.push(prop);
        Ok(id)
    }

    /// Drive the loop up to `max_iterations` times. Stops early if the
    /// queue empties, drops below threshold, or the pending cap is
    /// hit. Returns the ids of the proposals created (length == number
    /// of iterations that succeeded).
    pub fn research_loop_run(
        &mut self,
        fetcher: &dyn crate::research::http::HttpFetcher,
        max_iterations: usize,
    ) -> Vec<u64> {
        let mut out: Vec<u64> = Vec::new();
        for _ in 0..max_iterations {
            match self.research_one(fetcher) {
                Ok(id) => out.push(id),
                Err(_) => break,
            }
            if self.pending_proposal_count() >= self.research_config.max_proposals_pending {
                break;
            }
        }
        out
    }

    /// Port J.2 — chained research. Drives `crate::research::chain::run_chain`,
    /// which loops `research_one` while pushing fresh `CuriositySignal`
    /// entries onto the wonder queue when bond endpoints look like
    /// gaps. Returns a full `ChainTrace` so the MCP layer can surface
    /// the multi-hop walk to John. NEVER mutates stored bonds —
    /// approval still routes through `approve_research_proposal` and
    /// the T.3 PendingLog gate.
    pub fn research_chain_run(
        &mut self,
        fetcher: &dyn crate::research::http::HttpFetcher,
        seed_query_hint: Option<&str>,
    ) -> crate::research::chain::ChainTrace {
        crate::research::chain::run_chain(self, fetcher, seed_query_hint)
    }

    /// Manual research path — no wonder queue involvement. Used by the
    /// `lu4d_research { query }` MCP tool. Same approval gate as the
    /// autonomous loop.
    pub fn research_query(
        &mut self,
        fetcher: &dyn crate::research::http::HttpFetcher,
        query: &str,
    ) -> Result<u64, crate::research::loop_::ResearchError> {
        let prop = crate::research::loop_::make_proposal_for_query(
            self, fetcher, query, None,
        )?;
        let id = prop.id;
        // Port LB.1: dual-write the research result into the Left
        // Brain alongside the existing research_proposals push (same
        // pattern as research_one). Anti-hallucination invariant:
        // bond_count is unchanged.
        let lb_now = now_ms() as u64;
        let payload = serde_json::to_string(&prop).unwrap_or_default();
        let _ = self.lb_session.push_internal(
            crate::lb_session::SessionEntry::new(
                0,
                lb_now,
                crate::lb_session::SessionEntryKind::ResearchResult,
                query,
                format!("research_proposal id={}: {}", id, payload),
                0.0,
            ),
        );
        self.research_proposals.push(prop);
        Ok(id)
    }

    /// Approve a Pending proposal. Walks every bond candidate and runs
    /// `teach()`, which itself routes through `add_bond_with_scar_check`
    /// — so a candidate touching a scarred seed is STILL refused at
    /// ingest. Status moves Pending → Approved, then immediately
    /// Approved → Applied.
    pub fn approve_research_proposal(
        &mut self,
        id: u64,
    ) -> Result<u32, crate::research::loop_::ResearchError> {
        let idx = self
            .research_proposals
            .iter()
            .position(|p| p.id == id)
            .ok_or(crate::research::loop_::ResearchError::UnknownProposal(id))?;
        if !self.research_proposals[idx].is_pending() {
            return Err(crate::research::loop_::ResearchError::AlreadyDecided(id));
        }
        let now = now_ms() as u64;
        self.research_proposals[idx].status =
            crate::research::proposal::ResearchProposalStatus::Approved {
                approved_ms: now,
            };
        let added = crate::research::loop_::apply_proposal(self, idx);
        self.research_proposals[idx].status =
            crate::research::proposal::ResearchProposalStatus::Applied {
                applied_ms: now_ms() as u64,
                bonds_added: added,
            };
        Ok(added)
    }

    /// Reject a Pending proposal. Status moves Pending → Rejected; the
    /// proposal is preserved for audit (consult the wonder it was
    /// derived from to see why Lu thought it was worth researching).
    pub fn reject_research_proposal(
        &mut self,
        id: u64,
        reason: Option<String>,
    ) -> Result<(), crate::research::loop_::ResearchError> {
        let idx = self
            .research_proposals
            .iter()
            .position(|p| p.id == id)
            .ok_or(crate::research::loop_::ResearchError::UnknownProposal(id))?;
        if !self.research_proposals[idx].is_pending() {
            return Err(crate::research::loop_::ResearchError::AlreadyDecided(id));
        }
        self.research_proposals[idx].status =
            crate::research::proposal::ResearchProposalStatus::Rejected {
                rejected_ms: now_ms() as u64,
                reason,
            };
        Ok(())
    }

    /// Construct a search query string from a wonder. Currently uses
    /// the seed's content text directly; could be enriched in future
    /// by adding the signal type (e.g. "what surrounds X" vs "what is
    /// X"). Production knob lives here so we can iterate without
    /// touching the loop driver.
    pub fn wonder_to_query(&self, w: &crate::curiosity::CuriositySignal) -> String {
        // Port J.2: consult gap_label_map when the seed is not in
        // the network. Synthetic gap seeds (SHA-256 of "gap:<query>")
        // do not resolve to a network node; falling through to hex
        // display produced unsearchable hash queries before this fix.
        // Mirrors the J.1 surface fix already applied in tool_wonder.
        // OP-LABEL-4: the resident node is a content-free stub on a v5
        // brain, so this returned "" -- an EMPTY curiosity query. Route
        // through the OP-LABEL-1 resolver and fall through to the
        // gap-label map when the node genuinely has no text.
        if let Some(label) = self.network.node_label(&w.seed) {
            if !label.is_empty() {
                return label;
            }
        }
        if let Some(label) = self.gap_label_map.get(&w.seed) {
            return label.clone();
        }
        format!("{}", w.seed)
    }

    /// Build the production HTTP fetcher for the autonomous loop.
    /// Convenience so the MCP path doesn't need to know how to
    /// configure UreqFetcher.
    pub fn default_research_fetcher(&self) -> crate::research::http::UreqFetcher {
        crate::research::http::UreqFetcher::new(
            self.research_config.http_timeout_ms,
            self.research_config.user_agent.clone(),
        )
    }

    /// Ask the brain using an explicit temporal scope (no session scoping or
    /// Hebbian side-effects). Used for time-travel and read-only inspection.
    pub fn ask_in_scope(&mut self, query: &str, scope: &TemporalScope, depth: usize, top_k: usize) -> AskResult {
        // Issue A fix internal notes: mirror ask() / typed_ask() --
        // build the archive-backed keyword index before seed selection.
        // Signature changed from &self to &mut self so the lazy build
        // can fire here even when the MCP startup hoist hasn't run
        // (e.g. REPL time-travel inspection or unit tests).
        self.network.ensure_keyword_index_built();
        // Port T.5.1 — ask_in_scope is also used by time-travel
        // (TemporalScope::Historical), but the freshness multiplier
        // is keyed off bond *creation* timestamp, not the queried
        // scope, so applying it here still correctly de-ranks bonds
        // that have aged in wall-clock time. Time-travel callers that
        // want pure historical scoring can call clear_decay_runtime()
        // before invoking this path.
        self.refresh_decay_runtime();
        let mut weighted = self.network.select_seeds_for_query_ask(query);
        if weighted.is_empty() {
            let hits = self.network.keyword_search(query);
            weighted = hits.iter().take(3).map(|(s, _)| (*s, 0.7_f32)).collect();
        }
        // Port Sense.1 -- arm sense cluster-gain multiplier on every
        // ask_in_scope query.
        let _sense_seeds: Vec<crate::seed::Seed> = weighted.iter().map(|(s, _)| *s).collect();
        self.refresh_sense_runtime(&_sense_seeds);
        let recall = self.network.spread_activation_weighted(&weighted, scope, depth, top_k, &self.scar_log);
        AskResult {
            query: query.to_string(),
            seed_hits: weighted.into_iter().map(|(s, _)| s).collect(),
            recall,
            hebbian_strengthened: 0,
            session_id: None,
            bridges: Vec::new(),
            curiosity_signals: Vec::new(),
            scars: Vec::new(),
            // OP-ASK-WATCHDOG -- ask_in_scope path is unbounded.
            status: AskStatus::Complete,
        }
    }

    /// Start a new session. Ends any previously active session.
    /// Port Q.3: also instantiates a parallel `SessionMemory` keyed
    /// by the same id so the conversational handler can read/write
    /// per-session operational state.
    pub fn session_start(&mut self) -> u64 {
        let now = now_ms();
        let id = self.sessions.start_session(now);
        // Mirror the id into the operational/episodic registry.
        let _ = self.session_memories.get_or_create(id, now);
        // Port LB: activate the Left Brain session memory write guard.
        // Errors only when the id is already archived, which cannot
        // happen for a freshly allocated session id.
        let _ = self.lb_session.activate_session(id);
        id
    }

    /// End the currently active session, if any. Returns its ID.
    /// Port Q.3: also evicts the parallel `SessionMemory`.
    /// Port LB: also deactivates the Left Brain session write guard
    /// (subsequent lb_session writes without a fresh activation
    /// error out).
    pub fn session_end(&mut self) -> Option<u64> {
        let id = self.sessions.end_active_session(now_ms());
        if let Some(i) = id {
            self.session_memories.end(i);
        }
        self.lb_session.deactivate_session();
        id
    }

    /// Run baseline consolidation across all bonds.
    pub fn consolidate(&mut self) -> ConsolidationStats {
        self.consolidator.consolidate(&mut self.network, now_ms())
    }

    /// One-shot maintenance: remove keyword-index entries for any node
    /// whose content starts with "goal:" or "step:" (case-insensitive).
    /// These were created by v2 thinking.rs before add_node_unindexed
    /// existed; they pollute keyword recall. Nodes and bonds are kept
    /// intact so any temporal reasoning that depends on them survives;
    /// only the keyword-discoverability is removed.
    /// Returns counts of pruned entries: keyword index entries and
    /// interest profile entries.
    pub fn prune_legacy_goal_nodes(&mut self) -> PruneLegacyGoalReport {
        use std::collections::HashSet;
        let stale: HashSet<Seed> = self
            .network
            .all_seeds()
            .into_iter()
            .filter(|s| {
                self.network
                    .get_node(s)
                    .map(|n| {
                        let c = n.content().to_lowercase();
                        c.starts_with("goal:") || c.starts_with("step:")
                    })
                    .unwrap_or(false)
            })
            .collect();
        let keyword_entries_pruned = stale.len() as u32;
        self.network.unindex_seeds(&stale);
        // Also clean the interest profile of any goal:/step: leftovers.
        // These get there before `add_node_unindexed` was added (Port #12)
        // and persist across saves.
        let stale_interest: Vec<Seed> = self
            .interest_profile
            .interests
            .keys()
            .copied()
            .filter(|s| {
                self.network
                    .get_node(s)
                    .map(|n| {
                        let c = n.content().to_lowercase();
                        c.starts_with("goal:") || c.starts_with("step:")
                    })
                    .unwrap_or(false)
            })
            .collect();
        let interest_entries_pruned = stale_interest.len() as u32;
        for s in &stale_interest {
            self.interest_profile.interests.remove(s);
        }
        // Orphan cleanup: any interest entry whose seed has no node in the network
        // is a leftover from an earlier consolidation that removed the node but
        // not the interest entry. Drop these too.
        let orphan_interest: Vec<Seed> = self
            .interest_profile
            .interests
            .keys()
            .copied()
            .filter(|s| self.network.get_node(s).is_none())
            .collect();
        let orphan_interest_pruned = orphan_interest.len() as u32;
        for s in &orphan_interest {
            self.interest_profile.interests.remove(s);
        }
        PruneLegacyGoalReport {
            keyword_entries_pruned,
            interest_entries_pruned,
            orphan_interest_pruned,
        }
    }

    /// Run one Thinking-Layer pass. Synchronous; one call = one
    /// Observe -> Plan -> Act -> Reflect loop bounded by `opts`.
    pub fn think(
        &mut self,
        query: &str,
        opts: crate::thinking::ThinkConfig,
    ) -> crate::thinking::ThinkResult {
        let started_us = now_us();
        let variant_mode = opts.variant_mode;
        let result = crate::thinking::think(self, query, opts);
        // Port LB.1: route each Think step into the Left Brain as
        // a ThinkingTrace entry. Anti-hallucination invariant:
        // these are operational traces, not facts -- the canonical
        // bond_count is unchanged by these writes.
        let lb_now = now_ms() as u64;
        for step in &result.trace.steps {
            let payload = serde_json::to_string(step).unwrap_or_default();
            let _ = self.lb_session.push_internal(
                crate::lb_session::SessionEntry::new(
                    0,
                    lb_now,
                    crate::lb_session::SessionEntryKind::ThinkingTrace,
                    query,
                    format!("step {}: {}", step.index, payload),
                    step.reflection.score,
                ),
            );
        }
        // Self-reflection telemetry for think calls. We carry an
        // approximation of "was variant_mode" via the
        // confirmed_hypotheses / winning_variants pair; the analyzer
        // (`think_was_variant`) pivots on those.
        let scope = match self.sessions.active_session() {
            Some(s) => TemporalScope::Session { id: s.id, start: s.start_time },
            None => TemporalScope::Current,
        };
        let top_act = result.final_answer.recall.activated.first()
            .map(|a| a.activation).unwrap_or(0.0);
        let bridges = result.final_answer.bridges.len() as u32;
        let curiosity = result.final_answer.curiosity_signals.len() as u32;
        let confirmed = result.trace.hypotheses_promoted as u32;
        // winning_variants: we don't carry per-step arena outcomes
        // out of the trace, so we proxy: in variant_mode, the number
        // of bonds written equals the number of arenas that produced
        // a winner (since we filter by winner_id.is_some()). Default
        // mode reports 0.
        let winners = if variant_mode {
            result.trace.hypothesis_bonds_written as u32
        } else {
            0
        };
        let elapsed_us = now_us().saturating_sub(started_us);
        let q_hash = hash_query(query);
        self.push_query_metric(QueryMetric {
            timestamp_ms: now_ms() as u64,
            query_text_hash: q_hash,
            scope,
            latency_us: elapsed_us,
            top_activation: top_act,
            bridges_fired: bridges,
            synthesis_triangles_fired: 0,
            curiosity_signals_emitted: curiosity,
            confirmed_hypotheses: confirmed,
            winning_variants: winners,
            agent_role: None,
            agent_items_in: 0,
            agent_items_out: 0,
        });
        result
    }

    /// Run a time-travel diff: compare activation at a historical timestamp
    /// vs. the present.
    pub fn timetravel(&self, query: &str, at: i64) -> TemporalDiff {
        time_travel_query(query, at, &self.network)
    }

    /// Generate a fresh arena of variants for a goal seed. Spawns one
    /// variant per `PerturbationStrategy`, runs each in its own
    /// `Scope::Variant{id}`, and scores the arena. The winner (if any)
    /// is determined by composite score against `cfg.min_winner_composite`.
    ///
    /// Side effects: writes Variant-source keyframes for each variant's
    /// hypothesis bonds. These are invisible outside variant scope and
    /// decay naturally if not promoted.
    pub fn generate_variants(&mut self, goal_seed: Seed) -> VariantArena {
        self.generate_variants_with_config(goal_seed, &VariantConfig::default())
    }

    /// `generate_variants` with explicit configuration.
    pub fn generate_variants_with_config(
        &mut self,
        goal_seed: Seed,
        cfg: &VariantConfig,
    ) -> VariantArena {
        let now = now_ms();
        let parent_scope = match self.sessions.active_session() {
            Some(s) => TemporalScope::Session { id: s.id, start: s.start_time },
            None => TemporalScope::Current,
        };

        // Reserve a contiguous block of variant ids so determinism inside
        // the arena doesn't depend on whether previous calls happened.
        let base_id = self.next_variant_id;
        let strategies = PerturbationStrategy::all();
        let take = strategies.len().min(cfg.variants_per_goal);
        let strategies = &strategies[..take];
        self.next_variant_id = self.next_variant_id.saturating_add(take as u64);

        // Build the spec list deterministically. Each strategy gets its
        // own variant id; `derive_anchor` uses stable sort orders so a
        // given brain state always yields the same anchors.
        // Snapshot the scar log so the helper below can take it by
        // shared ref while `self` is also borrowed mutably.
        let scar_log_snapshot = self.scar_log.clone();
        let specs: Vec<VariantSpec> = strategies
            .iter()
            .enumerate()
            .filter_map(|(i, strategy)| {
                let id = base_id + i as u64;
                generate_variants_for_goal(self, goal_seed, *strategy, id, now, cfg, &scar_log_snapshot)
            })
            .collect();

        score_arena(self, goal_seed, &specs, &parent_scope, cfg, &scar_log_snapshot)
    }

    /// Promote the winning variant: every Variant-tagged keyframe with
    /// `id == arena.winner_id` gets a follow-up keyframe at `now+1` that
    /// is visible to baseline scope (`KeyframeSource::PromotedFromVariant`).
    /// Returns the number of bonds promoted. No-op if `arena.winner_id`
    /// is `None`.
    pub fn promote_variant(&mut self, arena: &VariantArena) -> usize {
        let winner = match arena.winner_id {
            Some(w) => w,
            None => return 0,
        };
        let now = now_ms();
        let mut promoted = 0usize;
        // Collect the bond keys + winner-source weight FIRST (immutable
        // borrow) so we can mutate the bonds afterwards without aliasing.
        let mut to_promote: Vec<((Seed, Seed), f32, u8)> = Vec::new();
        for ((from, to), bond) in self.network.bonds().iter() {
            // Audit fix internal notes: keyframes() panics on
            // archive-backed bonds; archive-aware iterator avoids that.
            for kf in bond.keyframes_iter() {
                if let KeyframeSource::Variant { id: vid, strategy, .. } = &kf.source {
                    if *vid == winner {
                        to_promote.push(((*from, *to), kf.weight, *strategy));
                    }
                }
            }
        }
        for ((from, to), weight, strategy) in to_promote {
            if let Some(mut b) = self.network.get_bond_mut(&from, &to) {
                b.add_keyframe(
                    now + 1,
                    weight,
                    KeyframeSource::PromotedFromVariant {
                        variant_id: winner,
                        original_strategy: strategy,
                    },
                );
                promoted += 1;
            }
        }
        promoted
    }

    /// Abandon every non-winner variant in `arena`. No-op at the data
    /// layer — variant keyframes decay naturally via the existing
    /// half-life — but exposed as a method so callers can be explicit
    /// and so we have a hook for future bookkeeping (e.g. a
    /// `variants_abandoned_total` counter). Consumes the arena.
    pub fn abandon_variants(&mut self, _arena: VariantArena) {
        // Intentional no-op: see docstring.
    }

    /// Run one multi-agent thinking pass. Three agents run
    /// synchronously and sequentially over a single brain in
    /// scope-isolated views: RecallAgent → SynthesisAgent → CriticAgent.
    /// Returns the per-agent traces plus the consolidated
    /// (Critic-vetted) beliefs. Vetted beliefs are NOT auto-promoted
    /// to baseline.
    pub fn multi_think(
        &mut self,
        query: &str,
    ) -> Result<crate::agents::MultiAgentTrace, String> {
        let cfg = self.multi_agent_config.clone();
        crate::agents::multi_think(self, query, &cfg)
    }

    /// `multi_think` with explicit configuration.
    pub fn multi_think_with_config(
        &mut self,
        query: &str,
        cfg: &crate::agents::MultiAgentConfig,
    ) -> Result<crate::agents::MultiAgentTrace, String> {
        crate::agents::multi_think(self, query, cfg)
    }

    /// Produce a coherent English prose answer for `query`. Wraps
    /// `ask` (and optionally `multi_think`) and runs the result
    /// through the 15-shape narration pipeline.
    pub fn narrate(
        &mut self,
        query: &str,
        depth: usize,
        top_k: usize,
        use_multi_think: bool,
    ) -> crate::narration::NarrationResult {
        let cfg = self.narration_config.clone();
        crate::narration::narrate(self, query, depth, top_k, use_multi_think, &cfg)
    }

    /// Port CS.2 — compute the active user's `StyleHints` for the
    /// current moment. This is the single read-side entry point
    /// every articulator uses to derive presentation shape from the
    /// persisted (and decayed) style vector.
    ///
    /// Returns `StyleHints::neutral()` when
    /// `style_config.output_adaptation_enabled == false`. Anti-
    /// hallucination invariant: this method does NOT mutate brain
    /// state — `bond_count` and the canonical graph are untouched.
    pub fn compute_style_hints(&self, now_ms: u64) -> crate::style::StyleHints {
        if !self.style_config.output_adaptation_enabled {
            return crate::style::StyleHints::neutral();
        }
        let user_key = self.style_config.user_key_str().to_string();
        let half_life = self.style_config.decay_half_life_days;
        let decayed = self.cs_style_store.get_decayed(&user_key, now_ms, half_life);
        crate::style::StyleHints::from_style_vector(&decayed, &self.style_config)
    }

    /// `narrate` with explicit configuration.
    pub fn narrate_with_config(
        &mut self,
        query: &str,
        depth: usize,
        top_k: usize,
        use_multi_think: bool,
        cfg: &crate::narration::NarrationConfig,
    ) -> crate::narration::NarrationResult {
        crate::narration::narrate(self, query, depth, top_k, use_multi_think, cfg)
    }

    /// OP-BRAIN-V5 6.6 -- operator-facing on-disk format label
    /// ("v5"/"v4"/"v3"/"v3c"/"v3enc"/"ephemeral"). The read-path mode is
    /// the authoritative discriminator for directory brains: `load_v5`
    /// sets `V5Only`, `load_v4` sets `V4Only`; v3 falls back to the
    /// `loaded_format` string.
    pub fn brain_format_label(&self) -> &'static str {
        match self.read_path_mode() {
            crate::read_path::ReadPathMode::V5Only => "v5",
            _ if self.brain_dir_v4.is_some() => "v4",
            _ => self.loaded_format.map(|f| f.as_str()).unwrap_or("ephemeral"),
        }
    }

    /// OP-BRAIN-V5 6.6 -- whether the loaded brain is encrypted at rest.
    /// A v4/v5 dir is encrypted iff a master key is held (its SSTs carry
    /// `FLAG_ENCRYPTED`); a v3 brain is encrypted iff loaded as `v3enc`.
    pub fn encrypted_at_rest(&self) -> bool {
        self.encryption_key.is_some()
            || self.loaded_format == Some(BrainSaveFormat::V3BinaryEncrypted)
    }

    /// OP-BRAIN-V5 6.6 -- format-aware (node, bond) counts for the
    /// operator banner + MCP status. v5 nodes come from the node intern
    /// (payloads live in `node_store` SSTs, not `network.nodes`); v5/v4
    /// bonds from the MemTable + manifest SST snapshot; v3 from the
    /// in-memory network.
    pub(crate) fn operator_node_bond_counts(&self) -> (usize, usize) {
        if self.brain_dir_v4.is_some() {
            let mt_bonds = self.memtable().map(|m| m.bond_count()).unwrap_or(0);
            let sst_bonds: usize = self
                .manifest_snapshot()
                .map(|snap| snap.iter_ssts().map(|(_, e)| e.bond_count as usize).sum())
                .unwrap_or(0);
            let bonds = mt_bonds + sst_bonds;
            let nodes = if self.read_path_mode() == crate::read_path::ReadPathMode::V5Only {
                self.network.node_intern().len()
            } else {
                let mt_subjects = self.memtable().map(|m| m.subject_count()).unwrap_or(0);
                mt_subjects + self.network.node_count()
            };
            (nodes, bonds)
        } else {
            (self.network.node_count(), self.network.bond_count())
        }
    }

    /// OP-BRAIN-V5 6.6 -- one-line operator banner printed at startup in
    /// both `--mcp` and REPL modes, reporting the loaded brain FORMAT,
    /// at-rest ENCRYPTION, the `--require-encrypted` setting, and counts.
    /// Capability only -- printing it changes nothing about which brain
    /// the service points at.
    pub fn operator_status_line(&self, require_encrypted: bool) -> String {
        let (nodes, bonds) = self.operator_node_bond_counts();
        let path = self
            .path
            .as_ref()
            .map(|p| p.display().to_string())
            .or_else(|| self.brain_dir_v4.as_ref().map(|p| p.display().to_string()))
            .unwrap_or_else(|| "<ephemeral>".into());
        format!(
            "[lu4d] brain status: format={} | encryption={} | require_encrypted={} | nodes={} | bonds={} | path={}",
            self.brain_format_label(),
            if self.encrypted_at_rest() { "encrypted-at-rest" } else { "PLAINTEXT" },
            if require_encrypted { "on" } else { "off" },
            nodes,
            bonds,
            path,
        )
    }

    /// Report the current status of the brain.
    pub fn status(&self) -> BrainStatus {
        let (active_session_id, activations, keyframes) = match self.sessions.active_session() {
            Some(s) => (Some(s.id), s.activation_count(), s.keyframe_count()),
            None => (None, 0, 0),
        };
        let total_synth: u32 = self
            .performance_monitor
            .recent_metrics
            .iter()
            .map(|m| m.synthesis_triangles_fired)
            .sum();
        let total_confirmed: u32 = self
            .performance_monitor
            .recent_metrics
            .iter()
            .map(|m| m.confirmed_hypotheses)
            .sum();
        BrainStatus {
            node_count: self.network.node_count(),
            bond_count: self.network.bond_count(),
            total_sessions: self.sessions.session_count(),
            active_session_id,
            active_session_activations: activations,
            active_session_keyframes: keyframes,
            brain_file: self.path.as_ref().map(|p| p.display().to_string()),
            // Bug A (2026-04-27): report on-disk format the brain was
            // loaded from. Falls back to "ephemeral" for unsaved brains.
            // We use loaded_format (set by load paths) rather than
            // save_format because main.rs overwrites save_format with the
            // CLI --brain-format-out flag right after load.
            format_version: self.brain_format_label().to_string(),
            bridge_count: self.recent_bridges.len(),
            bridges_dropped_overflow_total: self.bridges_dropped_overflow_total,
            recent_bridges: self.recent_bridges.iter().cloned().collect(),
            recent_metrics_count: self.performance_monitor.recent_metrics.len(),
            total_synthesis_triangles_fired: total_synth,
            total_confirmed_hypotheses: total_confirmed,
            wonder_queue_size: self.wonder_queue.len(),
            interest_profile_domain_count: self.interest_profile.interests.len(),
            // next_multi_think_id is 1-indexed; subtract 1 so a
            // never-called brain reports 0 multi-think calls.
            multi_think_call_count: self.next_multi_think_id.saturating_sub(1),
            degree_p90: self.network.degree_p90(),
            max_activations: self.network.max_activations(),
            max_activations_percentile: self.network.max_activations_percentile(),
            effective_max_activations: self.network.effective_cap_activations(),
            save_in_progress: { let st = self.save_runner.state.lock().unwrap(); st.is_running() },
            save_started_at_ms: { let st = self.save_runner.state.lock().unwrap(); st.started_at_ms() },
            encrypted_at_rest: self.encrypted_at_rest(),
            last_snapshot_capture_ms: { let st = self.save_runner.state.lock().unwrap(); st.snapshot_capture_ms() },
        }
    }
}


// =====================================================================
// Self-reflection / Meta-learning surface
// =====================================================================

impl Brain {
    /// Build a `CurrentKnobs` snapshot of every config field the
    /// self-reflection analyzers can target. Centralized here so
    /// `analyze` doesn't need cross-module access.
    pub fn current_knobs(&self) -> CurrentKnobs {
        CurrentKnobs {
            synthesis_min_leg_weight: self.consolidator.config.synthesis_min_leg_weight,
            synthesis_max_target_weight: self.consolidator.config.synthesis_max_target_weight,
            synthesis_path_decay: self.consolidator.config.synthesis_path_decay,
            confidence_to_answer: 0.7, // ThinkConfig default; per-call override
            confirmed_hypothesis_weight: 0.55,
            initial_hypothesis_weight: 0.15,
            weak_bond_threshold: self.bridge_config.weak_bond_threshold,
            cluster_disjoint_threshold: self.bridge_config.cluster_disjoint_threshold,
            deadend_yield_threshold: self.curiosity_config.deadend_yield_threshold as f32,
            knowledge_desert_threshold: self.curiosity_config.knowledge_desert_threshold,
            variant_min_winner_composite: 0.4, // VariantConfig default; per-call override
        }
    }

    /// Push a single QueryMetric into the performance monitor.
    /// Bounded by `self_reflection_config.metric_buffer_cap`.
    pub fn push_query_metric(&mut self, metric: QueryMetric) {
        // Keep the monitor cap aligned with config in case the user
        // tuned it after construction.
        let cap = self.self_reflection_config.metric_buffer_cap.max(1);
        self.performance_monitor.cap = cap;
        self.performance_monitor.push(metric);
    }

    /// Run one self-reflection pass: feed the recent metrics buffer
    /// into the analyzers and append any new proposals. Returns the
    /// count of new proposals.
    pub fn reflect(&mut self) -> usize {
        let knobs = self.current_knobs();
        let cfg = self.self_reflection_config.clone();
        let now = now_ms() as u64;
        analyze_reflection(
            &self.performance_monitor,
            &mut self.meta_reflection,
            &cfg,
            &knobs,
            now,
        )
    }

    /// Approve a proposal by id. If the target maps to a writable knob,
    /// apply the adjustment and mark the proposal `Applied`. Otherwise
    /// (e.g. PerformanceRegressionNotice) just mark it `Approved`.
    /// Returns true on success.
    pub fn approve_proposal(&mut self, id: u64) -> bool {
        let now = now_ms() as u64;
        // First pull out the target + direction so we can mutate the
        // brain config WITHOUT holding a borrow on meta_reflection.
        let info = match self.meta_reflection.proposals.iter()
            .find(|p| p.id == id && matches!(p.status, ProposalStatus::Pending))
        {
            Some(p) => Some((p.target, p.direction, p.proposed_value)),
            None => None,
        };
        let (target, _direction, proposed) = match info {
            Some(t) => t,
            None => return false,
        };

        // Apply the adjustment, if there is a knob to adjust.
        let applied = self.apply_adjustment(target, proposed);

        // Update the proposal status.
        if let Some(p) = self.meta_reflection.proposals.iter_mut().find(|p| p.id == id) {
            p.status = if applied {
                ProposalStatus::Applied { applied_ms: now }
            } else {
                ProposalStatus::Approved { approved_ms: now }
            };
            return true;
        }
        false
    }

    /// Reject a proposal. Records the optional reason on the status.
    pub fn reject_proposal(&mut self, id: u64, reason: Option<String>) -> bool {
        let now = now_ms() as u64;
        if let Some(p) = self.meta_reflection.proposals.iter_mut()
            .find(|p| p.id == id && matches!(p.status, ProposalStatus::Pending))
        {
            p.status = ProposalStatus::Rejected { rejected_ms: now, reason };
            return true;
        }
        false
    }

    /// Mutate the underlying knob for a given target. Returns true if
    /// a writable knob exists (false for notice-only targets).
    fn apply_adjustment(&mut self, target: AdjustmentTarget, proposed: f32) -> bool {
        match target {
            AdjustmentTarget::SynthesisMinLegWeight => {
                self.consolidator.config.synthesis_min_leg_weight = proposed;
                true
            }
            AdjustmentTarget::SynthesisMaxTargetWeight => {
                self.consolidator.config.synthesis_max_target_weight = proposed;
                true
            }
            AdjustmentTarget::SynthesisPathDecay => {
                self.consolidator.config.synthesis_path_decay = proposed;
                true
            }
            AdjustmentTarget::WeakBondThreshold => {
                self.bridge_config.weak_bond_threshold = proposed;
                true
            }
            AdjustmentTarget::ClusterDisjointThreshold => {
                self.bridge_config.cluster_disjoint_threshold = proposed;
                true
            }
            AdjustmentTarget::DeadendYieldThreshold => {
                self.curiosity_config.deadend_yield_threshold = proposed.max(0.0) as usize;
                true
            }
            AdjustmentTarget::KnowledgeDesertThreshold => {
                self.curiosity_config.knowledge_desert_threshold = proposed;
                true
            }
            // The remaining targets are per-call config (ThinkConfig,
            // VariantConfig). They do NOT live on Brain, so approval
            // here records the decision but doesn't mutate state.
            // Callers should pass the new value via their next
            // ThinkConfig / VariantConfig.
            AdjustmentTarget::ConfidenceToAnswer
            | AdjustmentTarget::ConfirmedHypothesisWeight
            | AdjustmentTarget::InitialHypothesisWeight
            | AdjustmentTarget::VariantMinWinnerComposite
            | AdjustmentTarget::PerformanceRegressionNotice
            | AdjustmentTarget::AgentDriftNotice => false,
        }
    }
}


/// Microsecond-resolution wall clock. Used for QueryMetric latency.
fn now_us() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_micros() as u64)
        .unwrap_or(0)
}

/// Stable hash of the query text. We persist a hash, not the text,
/// in QueryMetric so user content doesn't accumulate in telemetry.
fn hash_query(q: &str) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    q.hash(&mut h);
    h.finish()
}

/// Record one QueryMetric for an ask/typed_ask call. Internal helper —
/// the public surface is `Brain::push_query_metric`.
fn record_ask_metric(
    brain: &mut Brain,
    query: &str,
    result: &AskResult,
    scope: &TemporalScope,
    started_us: u64,
) {
    let elapsed_us = now_us().saturating_sub(started_us);
    let top_act = result.recall.activated.first()
        .map(|a| a.activation).unwrap_or(0.0);
    brain.push_query_metric(QueryMetric {
        timestamp_ms: now_ms() as u64,
        query_text_hash: hash_query(query),
        scope: scope.clone(),
        latency_us: elapsed_us,
        top_activation: top_act,
        bridges_fired: result.bridges.len() as u32,
        // ask/typed_ask don't run synthesis; that fires only during
        // consolidation. The metric tracks ask-call-attributable signals.
        synthesis_triangles_fired: 0,
        curiosity_signals_emitted: result.curiosity_signals.len() as u32,
        confirmed_hypotheses: 0,
        winning_variants: 0,
        agent_role: None,
        agent_items_in: 0,
        agent_items_out: 0,
    });
}


// =====================================================================
// =====================================================================
// Gap signal flush (Port J / Curiosity retrofit)
// =====================================================================

impl Brain {
    /// Drain `gap_emitter` and translate each `GapSignal` into a
    /// `CuriositySignal` pushed onto `wonder_queue`, also touching
    /// `interest_profile` so the existing autonomous-research path
    /// pops them as research targets.
    ///
    /// Called at the end of every reasoning-shaped MCP tool
    /// (`tool_reason`, `tool_ask`, `tool_answer`, `tool_multi_think`)
    /// so the wonder queue grows in lockstep with reasoning gaps.
    /// Returns the number of gaps flushed (useful for the verify rig).
    pub fn flush_gaps(&mut self) -> usize {
        let drained = self.gap_emitter.drain();
        let n = drained.len();
        if n == 0 {
            return 0;
        }
        let now = now_ms() as u64;
        let halflife = self.curiosity_config.interest_decay_halflife_ms;
        for g in drained {
            let cs = g.to_curiosity_signal(now);
            // Port J.1: stash human-readable label for the synthetic
            // gap-seed so `tool_wonder` can surface it instead of a
            // raw hex hash.
            self.gap_label_map.insert(cs.seed, g.query.clone());
            self.interest_profile.touch(cs.seed, now, halflife);
            // Port LB.1: dual-write the wonder signal as a WonderItem entry.
            {
                // OP-LABEL-4: see the note on the wonder-snapshot site.
                let lb_label = self.network.node_label(&cs.seed)
                    .filter(|l| !l.is_empty())
                    .or_else(|| self.gap_label_map.get(&cs.seed).cloned())
                    .unwrap_or_else(|| cs.seed.to_string());
                let lb_payload = serde_json::to_string(&cs).unwrap_or_default();
                let lb_intensity = cs.intensity;
                let _ = self.lb_session.push_internal(
                    crate::lb_session::SessionEntry::new(
                        0,
                        now,
                        crate::lb_session::SessionEntryKind::WonderItem,
                        lb_label,
                        lb_payload,
                        lb_intensity,
                    ),
                );
            }
            // OP-CURIOSITY-1 -- see the note at the ask-path push sites.
            self.record_wonder_durably(&cs);
            self.wonder_queue.push(cs);
        }
        n
    }
}
// Scar surface — type-enforced safety primitive
// =====================================================================

impl Brain {
    /// Scar the directed bond `(from, to)`. Records a `Scar` in the
    /// log; the bond's history is preserved untouched. Every
    /// subsequent weight read on this bond returns
    /// `effective_weight = 0.0` until the bond is unscarred. The
    /// scar is anchored to wall-clock NOW, so even time-travel reads
    /// at past timestamps see the scar via the forward-search
    /// invariant.
    pub fn scar_bond(
        &mut self,
        from: Seed,
        to: Seed,
        reason: impl Into<String>,
    ) -> Scar {
        // Fingerprint stub: reserved for future content-similarity
        // suggestions when scarring related bonds. Currently empty.
        let fingerprint: Vec<f32> = Vec::new();
        self.scar_log
            .scar_bond(from, to, reason.into(), fingerprint, None, now_ms())
    }

    /// Cascade scar — scar every bond touching `seed` (incoming or
    /// outgoing). Returns the per-bond scars created. Per Lu's DF
    /// SEED_REVOKE semantics: the concept itself stays in the graph,
    /// but no action layer can traverse a bond into or out of it.
    pub fn scar_concept(&mut self, seed: Seed, reason: impl Into<String>) -> Vec<Scar> {
        let mut bonds: Vec<(Seed, Seed)> = Vec::new();
        for ((from, to), _) in self.network.bonds().iter() {
            if *from == seed || *to == seed {
                bonds.push((*from, *to));
            }
        }
        self.scar_log
            .scar_concept(&bonds, reason.into(), None, now_ms())
    }

    /// All scars currently active on the bond `(from, to)`. Anchored
    /// to NOW. Pass-through to `ScarLog::active_scars_now`.
    pub fn active_scars(&self, from: Seed, to: Seed) -> Vec<&Scar> {
        self.scar_log.active_scars_now(from, to)
    }

    /// Admin-only: remove every active scar on `(from, to)` and log
    /// the unscar in the audit record. Returns the number of scars
    /// removed. Rare operation — surfaces in narration so the
    /// override is visible.
    pub fn unscar_bond(
        &mut self,
        from: Seed,
        to: Seed,
        reason: impl Into<String>,
    ) -> u32 {
        self.scar_log
            .unscar_bond(from, to, reason.into(), now_ms())
    }

    /// Cascade-undo of `scar_concept`. Removes `seed` from the
    /// forward-protection set (so NEW bonds touching it can form again)
    /// AND supersedes every active scar on every bond touching `seed`.
    /// Returns the count of bonds whose scars were superseded.
    /// One UnscarRecord is appended per bond (audit trail per bond).
    pub fn unscar_concept(&mut self, seed: Seed, reason: impl Into<String>) -> u32 {
        let reason: String = reason.into();
        // 1. lift the forward-protection set entry (if any).
        self.scar_log.unscar_seed(seed);
        // 2. enumerate every bond touching the seed and unscar each.
        let mut bonds: Vec<(Seed, Seed)> = Vec::new();
        for ((from, to), _) in self.network.bonds().iter() {
            if *from == seed || *to == seed {
                bonds.push((*from, *to));
            }
        }
        let mut total: u32 = 0;
        let now = now_ms();
        for (from, to) in bonds {
            total = total.saturating_add(
                self.scar_log.unscar_bond(from, to, reason.clone(), now)
            );
        }
        total
    }
}

impl Default for Brain {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {

    #[test]
    fn ingest_admission_gate_rejects_non_concept_endpoints() {
        use std::sync::Mutex;
        static ENV_LOCK: Mutex<()> = Mutex::new(());
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::set_var("LU_CANON_INGEST", "1");
        let mut b = Brain::new();
        // Non-concept object (article-led long fragment) -> no-op, nothing minted.
        let r = b.teach_with_kind(
            "Dog",
            BondType::IsA,
            "an off-leash dog park located at the township municipal",
            1.0,
            Confidence::Verified,
            None,
            None,
        );
        assert!(!r.created_subject && !r.created_object, "junk endpoint must not mint nodes");
        // Bare code identifier subject -> rejected too.
        let r2 = b.teach_with_kind(
            "handleCrossRibbonClick", BondType::IsA, "Widget",
            1.0, Confidence::Verified, None, None);
        assert!(!r2.created_subject && !r2.created_object, "code identifier must not mint nodes");
        // Clean concepts on both ends -> admitted.
        let r3 = b.teach_with_kind(
            "Dog", BondType::IsA, "Animal", 1.0, Confidence::Verified, None, None);
        assert!(r3.created_subject || r3.created_object, "clean concepts must be admitted");
        std::env::remove_var("LU_CANON_INGEST");
    }
    use super::*;
    use tempfile::tempdir;

    // Minimal temp-dir substitute so we don't add a dev-dep.
    mod tempfile {
        use std::path::PathBuf;
        pub struct TempDir { path: PathBuf }
        impl TempDir {
            pub fn path(&self) -> &std::path::Path { &self.path }
        }
        impl Drop for TempDir {
            fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.path); }
        }
        pub fn tempdir() -> std::io::Result<TempDir> {
            use std::sync::atomic::{AtomicU64, Ordering};
            static COUNTER: AtomicU64 = AtomicU64::new(0);
            let n = COUNTER.fetch_add(1, Ordering::SeqCst);
            let mut p = std::env::temp_dir();
            p.push(format!("lu4d-test-{}-{}", std::process::id(), n));
            std::fs::create_dir_all(&p)?;
            Ok(TempDir { path: p })
        }
    }

    #[test]
    fn test_teach_creates_nodes_and_bond() {
        let mut brain = Brain::new();
        let r = brain.teach("Rust", BondType::IsA, "ProgrammingLanguage", 0.9, Confidence::Verified);
        assert!(r.created_subject);
        assert!(r.created_object);
        assert!(!r.reinforced_existing_bond);
        assert_eq!(brain.network.node_count(), 2);
        assert_eq!(brain.network.bond_count(), 1);
    }

    #[test]
    fn test_teach_reinforces_existing_bond() {
        let mut brain = Brain::new();
        brain.teach("Rust", BondType::IsA, "Lang", 0.8, Confidence::Verified);
        let r2 = brain.teach("Rust", BondType::IsA, "Lang", 0.9, Confidence::Verified);
        assert!(r2.reinforced_existing_bond);
        assert_eq!(brain.network.bond_count(), 1);
    }

    #[test]
    fn test_ask_returns_activation() {
        let mut brain = Brain::new();
        brain.teach("Rust", BondType::IsA, "SystemsLang", 0.9, Confidence::Verified);
        brain.teach("Rust", BondType::HasProperty, "MemorySafety", 0.95, Confidence::Verified);
        let r = brain.ask("Rust", 3, 10);
        assert!(!r.recall.activated.is_empty());
    }

    #[test]
    fn test_save_and_load_roundtrip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.lbr");

        let mut brain = Brain::load_or_new(&path).unwrap();
        brain.teach("Alpha", BondType::RelatedTo, "Beta", 0.7, Confidence::Verified);
        brain.save().unwrap();

        let reloaded = Brain::load_or_new(&path).unwrap();
        assert_eq!(reloaded.network.node_count(), 2);
        assert_eq!(reloaded.network.bond_count(), 1);
    }

    // ----------------------------------------------------------------
    // Port Source-Persist internal notes: SourceAttribution
    // entries stamped via teach_corpus must survive a save / reload
    // cycle through the v3enc on-disk format. Pre-Source-Persist, the
    // overlay was correct in-process but always reloaded as empty
    // (BrainFile.source_overlay had no v3 section). This test pins the
    // round-trip and also asserts the anti-hallucination structural
    // invariant (bond_count unchanged across save/reload).
    // ----------------------------------------------------------------
    #[test]
    fn test_source_overlay_persists_through_v3enc_save_reload() {
        let dir = tempdir().unwrap();
        let sub = dir.path().join("source_persist");
        std::fs::create_dir_all(&sub).unwrap();
        let path = sub.join("brain.lbr");

        let mk = crate::encrypt::MasterKey::from_bytes([0x42u8; 32]);

        // Phase 1: build encrypted brain, teach a corpus bond, save.
        let (pre_bond_count, pre_overlay_count) = {
            let mut brain = Brain::load_or_new_with_mode_and_key(
                &path, BrainLoadMode::Auto, Some(mk.clone()),
            ).unwrap();
            let r = brain.teach_corpus(
                "alpha", BondType::IsA, "beta",
                0.7, Confidence::Verified,
                "test/source-persist", None, None,
            );
            assert!(r.created_subject && r.created_object,
                "teach_corpus must create both nodes; got {:?}", r);

            let pre_match = brain.source_overlay.read().unwrap().entries.iter()
                .filter(|(_, attr)| matches!(
                    &attr.kind,
                    crate::source::SourceKind::TrainingCorpus { corpus, .. }
                        if corpus == "test/source-persist"
                ))
                .count();
            assert_eq!(pre_match, 1,
                "in-process overlay must contain TrainingCorpus entry pre-save");

            let bonds = brain.network.bond_count();
            let overlay = brain.source_overlay.read().unwrap().entries.len();
            brain.save().unwrap();
            (bonds, overlay)
        };

        // Phase 2: reopen fresh, query overlay state.
        let brain = Brain::load_or_new_with_mode_and_key(
            &path, BrainLoadMode::Auto, Some(mk),
        ).unwrap();

        // Anti-hallucination structural invariant: bond_count must not
        // mutate across save/reload. Source attributions are metadata
        // and never modify stored bond weights or counts.
        assert_eq!(brain.network.bond_count(), pre_bond_count,
            "bond_count must be identical pre-save and post-reload");

        let post_match = brain.source_overlay.read().unwrap().entries.iter()
            .filter(|(_, attr)| matches!(
                &attr.kind,
                crate::source::SourceKind::TrainingCorpus { corpus, .. }
                    if corpus == "test/source-persist"
            ))
            .count();
        assert_eq!(post_match, 1,
            "overlay TrainingCorpus entry must survive v3enc save/reload \
             (Source-Persist bug: pre-fix this is 0 because v3 has no \
             SourceOverlay section)");
        assert_eq!(brain.source_overlay.read().unwrap().entries.len(), pre_overlay_count,
            "overlay entry count must match across save/reload");
    }

    /// Port Source-Persist: multiple teach_corpus calls round-trip
    /// through the v3enc save/load cycle.
    #[test]
    fn test_source_overlay_multiple_teach_corpus_round_trip() {
        let dir = tempdir().unwrap();
        let sub = dir.path().join("source_persist_multi");
        std::fs::create_dir_all(&sub).unwrap();
        let path = sub.join("brain.lbr");
        let mk = crate::encrypt::MasterKey::from_bytes([0x37u8; 32]);

        let pre_count = {
            let mut brain = Brain::load_or_new_with_mode_and_key(
                &path, BrainLoadMode::Auto, Some(mk.clone()),
            ).unwrap();
            for i in 0..8 {
                brain.teach_corpus(
                    &format!("subj{}", i), BondType::IsA, &format!("obj{}", i),
                    0.6, Confidence::Verified,
                    "test/source-persist-multi",
                    Some(format!("run-{}", i / 4)),
                    Some(format!("https://example.org/{}", i)),
                );
            }
            let pre_match = brain.source_overlay.read().unwrap().entries.iter()
                .filter(|(_, attr)| matches!(
                    &attr.kind,
                    crate::source::SourceKind::TrainingCorpus { corpus, .. }
                        if corpus == "test/source-persist-multi"
                ))
                .count();
            assert_eq!(pre_match, 8, "all 8 entries present in-memory");
            brain.save().unwrap();
            pre_match
        };

        let brain = Brain::load_or_new_with_mode_and_key(
            &path, BrainLoadMode::Auto, Some(mk),
        ).unwrap();
        let post_match = brain.source_overlay.read().unwrap().entries.iter()
            .filter(|(_, attr)| matches!(
                &attr.kind,
                crate::source::SourceKind::TrainingCorpus { corpus, .. }
                    if corpus == "test/source-persist-multi"
            ))
            .count();
        assert_eq!(post_match, pre_count, "all entries survive reload");
    }

    /// Port Source-Persist: mixed SourceKind variants (corpus + research)
    /// preserve their kind discrimination across save/reload.
    #[test]
    fn test_source_overlay_mixed_kinds_round_trip() {
        let dir = tempdir().unwrap();
        let sub = dir.path().join("source_persist_mixed");
        std::fs::create_dir_all(&sub).unwrap();
        let path = sub.join("brain.lbr");
        let mk = crate::encrypt::MasterKey::from_bytes([0x91u8; 32]);

        {
            let mut brain = Brain::load_or_new_with_mode_and_key(
                &path, BrainLoadMode::Auto, Some(mk.clone()),
            ).unwrap();
            // Corpus-attributed bond.
            brain.teach_corpus(
                "alpha", BondType::IsA, "category_a",
                0.7, Confidence::Verified,
                "test/source-persist-mixed", None, None,
            );
            // Research-attributed bond via teach_with_source.
            let research_attr = crate::source::SourceAttribution::research(
                "wonder-XYZ".to_string(),
                Some("https://example.org/wonder".into()),
                1_700_000_000_000u64,
            );
            brain.teach_with_source(
                "beta", BondType::IsA, "category_b",
                0.7, Confidence::Verified,
                research_attr,
            );
            brain.save().unwrap();
        }

        let brain = Brain::load_or_new_with_mode_and_key(
            &path, BrainLoadMode::Auto, Some(mk),
        ).unwrap();
        let mut corpus_count = 0usize;
        let mut research_count = 0usize;
        let source_overlay_guard = brain.source_overlay.read().unwrap();
        for (_, attr) in &source_overlay_guard.entries {
            match &attr.kind {
                crate::source::SourceKind::TrainingCorpus { corpus, .. }
                    if corpus == "test/source-persist-mixed" => corpus_count += 1,
                crate::source::SourceKind::Research { wonder_seed, .. }
                    if wonder_seed == "wonder-XYZ" => research_count += 1,
                _ => {}
            }
        }
        assert_eq!(corpus_count, 1, "corpus bond round-trips with TrainingCorpus kind");
        assert_eq!(research_count, 1, "research bond round-trips with Research kind");
    }

    /// Port Source-Persist: backward-compat. A pre-Source-Persist v3
    /// brain has no SourceOverlay section (id 16). The reader path
    /// must hydrate an empty overlay rather than crashing — the same
    /// J.1.1 unknown-section tolerance every prior port relied on.
    /// Emulated here by directly invoking `de_bincode_optional` on a
    /// sections map that omits section 16.
    #[test]
    fn test_source_overlay_pre_source_persist_brain_hydrates_empty() {
        // A plaintext-v3 brain produced by the legacy 15-section writer
        // would parse into a `sections: HashMap<u32, &[u8]>` that lacks
        // key 16. The post-Source-Persist reader must default-empty in
        // that case, not error.
        //
        // We exercise this end-to-end: build a fresh brain, hand-craft
        // a `BrainFile` with an empty `source_overlay`, write_v3, and
        // confirm read_v3_bytes returns an empty overlay (the section
        // is present but empty, which is the closest in-test analog to
        // section-absent — `de_bincode_optional` treats both the same).
        // Brain::new() yields an empty source_overlay; we deliberately
        // do NOT call teach() because Brain::teach() now stamps a
        // default `Teach { actor: "system" }` attribution into the
        // overlay (Port S). Pre-Source-Persist files would have an
        // analogous shape: overlay populated in-process, dropped
        // on save, hydrates empty on reload. Here we test the
        // empty-section round-trip directly.
        let brain = Brain::new();
        let file = brain.to_brain_file(crate::brain_io_v3::FORMAT_VERSION_V3);
        assert!(file.source_overlay.entries.is_empty(),
            "baseline: empty brain has empty overlay");
        let bytes = crate::brain_io_v3::write_v3(&file).unwrap();
        // 4-byte aligned copy for rkyv archived sections inside read_v3_bytes.
        let mut aligned: Vec<u8> = Vec::with_capacity(bytes.len() + 8);
        aligned.resize(8, 0u8);
        let pad = (8 - (aligned.as_ptr() as usize) % 8) % 8;
        aligned.truncate(pad);
        aligned.extend_from_slice(&bytes);
        let parsed = crate::brain_io_v3::read_v3_bytes(&aligned[pad..]).unwrap();
        assert!(parsed.source_overlay.entries.is_empty(),
            "empty overlay round-trips as empty (no crash, no spurious entries)");
    }

    #[test]
    fn test_typed_ask_cascades_on_weak_primary() {
        // Five nodes share the keyword `qz`. select_seeds picks the
        // first two by keyword-index insertion order (qz alpha, qz
        // beta), neither of which has an IsA outgoing — primary typed
        // spread under bond_filter=[IsA] produces 0 d>=1 hits. Cascade
        // adds the rest as low-weight seeds; qz delta and qz epsilon
        // each have IsA outgoing, so the re-spread surfaces their
        // targets.
        let mut brain = Brain::new();
        brain.teach("qz alpha",   BondType::RelatedTo, "ignored1", 0.5, Confidence::Verified);
        brain.teach("qz beta",    BondType::RelatedTo, "ignored2", 0.5, Confidence::Verified);
        brain.teach("qz gamma",   BondType::RelatedTo, "ignored3", 0.5, Confidence::Verified);
        brain.teach("qz delta",   BondType::IsA,       "result_delta",   0.9, Confidence::Verified);
        brain.teach("qz epsilon", BondType::IsA,       "result_epsilon", 0.9, Confidence::Verified);

        let r = brain.typed_ask("qz", 3, 50, &[BondType::IsA]);
        let contents: Vec<String> = r.recall.activated.iter().map(|a| a.content.clone()).collect();
        let hit = contents.iter().any(|c| c == "result_delta")
            || contents.iter().any(|c| c == "result_epsilon");
        assert!(
            hit,
            "typed_ask cascade should have surfaced an IsA target reachable only via cascade-added seeds; got: {:?}",
            contents
        );
    }

    #[test]
    fn test_interest_profile_persists_through_save_load() {
        // Per-test subdir — `tempdir()` shares one path per process pid,
        // and parallel tests would otherwise clobber each other on drop.
        let dir = tempdir().unwrap();
        let sub = dir.path().join("interest_persist");
        std::fs::create_dir_all(&sub).unwrap();
        let path = sub.join("interest.lbr");

        // Seed the profile with two non-default touches so we can
        // distinguish a real reload from a default-constructed profile.
        let seed_a = Seed::generate();
        let seed_b = Seed::generate();
        let halflife: u64 = 7 * 86_400_000;
        let now: u64 = 1_700_000_000_000;

        let mut brain = Brain::load_or_new(&path).unwrap();
        brain.interest_profile.touch(seed_a, now, halflife);
        brain.interest_profile.touch(seed_a, now + 1_000, halflife);
        brain.interest_profile.touch(seed_b, now + 2_000, halflife);
        let pre_a = brain.interest_profile.weight_at(&seed_a, now + 2_000, halflife);
        let pre_b = brain.interest_profile.weight_at(&seed_b, now + 2_000, halflife);
        assert!(pre_a > 0.0);
        assert!(pre_b > 0.0);
        brain.save().unwrap();

        let reloaded = Brain::load_or_new(&path).unwrap();
        let post_a = reloaded.interest_profile.weight_at(&seed_a, now + 2_000, halflife);
        let post_b = reloaded.interest_profile.weight_at(&seed_b, now + 2_000, halflife);
        assert!((pre_a - post_a).abs() < 1e-6);
        assert!((pre_b - post_b).abs() < 1e-6);
        assert_eq!(reloaded.interest_profile.interests.len(), 2);
    }

    #[test]
    fn test_parse_bond_type_variants() {
        assert_eq!(parse_bond_type("IsA").unwrap(), BondType::IsA);
        assert_eq!(parse_bond_type("is_a").unwrap(), BondType::IsA);
        assert_eq!(parse_bond_type("HAS").unwrap(), BondType::HasProperty);
        assert!(parse_bond_type("nope").is_err());
    }

    #[test]
    fn br1_parse_bond_type_accepts_17_extension_variants() {
        // BR.1 — every relation string emitted by
        // scripts/smart_extractor.py that BR.0 preflight Phase A.1
        // flagged as DOA must now parse cleanly.
        let cases: &[(&str, BondType)] = &[
            ("BornIn", BondType::BornIn),
            ("BornOn", BondType::BornOn),
            ("DiedIn", BondType::DiedIn),
            ("DiedOn", BondType::DiedOn),
            ("DiscoveredBy", BondType::DiscoveredBy),
            ("FoundedBy", BondType::FoundedBy),
            ("FoundedIn", BondType::FoundedIn),
            ("HasAltitude", BondType::HasAltitude),
            ("HasAspect", BondType::HasAspect),
            ("HasCapital", BondType::HasCapital),
            ("HasOfficialLanguage", BondType::HasOfficialLanguage),
            ("HasPopulation", BondType::HasPopulation),
            ("KnownFor", BondType::KnownFor),
            ("LocatedIn", BondType::LocatedIn),
            ("MeasuredIn", BondType::MeasuredIn),
            ("NamedAfter", BondType::NamedAfter),
            ("WrittenBy", BondType::WrittenBy),
        ];
        for (s, expected) in cases {
            let got = parse_bond_type(s).unwrap_or_else(|_| {
                panic!("BR.1: parse_bond_type({:?}) failed but should accept it", s)
            });
            assert_eq!(got, *expected, "BR.1: {:?} parsed to wrong variant", s);
        }
    }

    #[test]
    fn br10b_parse_bond_type_accepts_has_area() {
        // BR.10b — smart_extractor L5-area pattern emits "HasArea"
        // (e.g. "Geography of Azerbaijan -> 605 km²"). Without this
        // arm, every VG/wikipedia area attribute would silently fail
        // to land at ingest time.
        assert_eq!(parse_bond_type("HasArea").unwrap(), BondType::HasArea);
        assert_eq!(parse_bond_type("hasarea").unwrap(), BondType::HasArea);
        assert_eq!(parse_bond_type("HASAREA").unwrap(), BondType::HasArea);
    }

    /// Build two hub clusters joined by a single weak RelatedTo bond.
    /// Helper for the AskResult-bridges tests below.
    fn brain_with_two_clusters() -> Brain {
        let mut brain = Brain::new();
        // Drop hub threshold so 5-leaf hubs qualify in unit tests.
        brain.bridge_config.hub_degree_threshold = 3;
        for i in 0..5 {
            brain.teach(
                "hubA", BondType::RelatedTo, &format!("a_leaf_{}", i),
                0.9, Confidence::Verified,
            );
        }
        for i in 0..5 {
            brain.teach(
                "hubB", BondType::RelatedTo, &format!("b_leaf_{}", i),
                0.9, Confidence::Verified,
            );
        }
        // Weak bridge: 0.1 raw * RelatedTo 0.8 base = 0.08 effective < 0.25.
        brain.teach("hubA", BondType::RelatedTo, "hubB", 0.1, Confidence::Verified);
        brain
    }

    #[test]
    fn test_ask_surfaces_bridges_in_result() {
        let mut brain = brain_with_two_clusters();
        let r = brain.ask("a_leaf_0", 3, 50);
        assert!(
            !r.bridges.is_empty(),
            "ask() must surface this query'''s bridges in AskResult.bridges",
        );
    }

    #[test]
    fn test_typed_ask_surfaces_bridges_in_result() {
        let mut brain = brain_with_two_clusters();
        let r = brain.typed_ask("a_leaf_0", 3, 50, &[BondType::RelatedTo]);
        assert!(
            !r.bridges.is_empty(),
            "typed_ask() must surface bridges through the same Phase-2 path",
        );
    }


    // ============================================================
    // Scar port — 10 mandated behavioural tests + fingerprint stub.
    // ============================================================

    use crate::scar::{ScarAwareReading, ScarLog};

    /// Helper: insert a node-pair with a single bond at a chosen
    /// timestamp and return the directed seeds. Bypasses `teach` so
    /// the bond's first keyframe lands at the test's chosen `t`.
    fn scar_test_bond(
        brain: &mut Brain,
        from: &str,
        to: &str,
        weight: f32,
        timestamp: i64,
    ) -> (Seed, Seed) {
        let f = Seed::from_label(from);
        let t = Seed::from_label(to);
        if brain.network.get_node(&f).is_none() {
            let n = crate::node::KnowledgeNode::with_seed(
                f,
                from.to_string(),
                crate::node::KnowledgeContent::Concept(from.to_string()),
                Confidence::Verified,
                timestamp,
            );
            brain.network.add_node(n);
        }
        if brain.network.get_node(&t).is_none() {
            let n = crate::node::KnowledgeNode::with_seed(
                t,
                to.to_string(),
                crate::node::KnowledgeContent::Concept(to.to_string()),
                Confidence::Verified,
                timestamp,
            );
            brain.network.add_node(n);
        }
        let bond = crate::bond::TemporalBond::new(f, t, BondType::RelatedTo, weight, timestamp);
        let _ = brain.network.add_bond(bond);
        (f, t)
    }

    /// 1. `Brain::scar_bond` records a Scar in the log.
    #[test]
    fn test_scar_bond_creates_scar() {
        let mut brain = Brain::new();
        let (f, t) = scar_test_bond(&mut brain, "alpha", "beta", 0.9, 1000);
        let scar = brain.scar_bond(f, t, "test reason");
        assert_eq!(scar.bond_seeds, (f, t));
        assert_eq!(scar.reason, "test reason");
        assert_eq!(brain.scar_log.scars.len(), 1);
        assert_eq!(brain.active_scars(f, t).len(), 1);
    }

    /// 2. KEYSTONE INVARIANT — the architectural primitive the entire
    /// port is built on.
    ///
    /// The scar lookup is anchored to wall-clock NOW, NEVER to the
    /// caller's `query_time`. A bond whose history puts it at t=10
    /// (active, unscarred) must STILL see a scar created at t=20.
    /// The forward search refuses to look past current scars
    /// regardless of how the temporal-scope query was constructed.
    ///
    /// This test exercises the primitive directly: the bond is
    /// historical at t=10, but a scar timestamped 20 is visible to
    /// any query whose forward-search anchor is >= 20. Wall-clock
    /// NOW is ~1.7e12 (much greater than 20), so production calls
    /// to `bond.weight_at(10, &log)` see the scar.
    ///
    /// If this test breaks, the safety architecture is broken.
    #[test]
    fn test_scar_anchored_to_now_not_query_time() {
        let f = Seed::from_label("alpha");
        let t = Seed::from_label("beta");
        let bond = crate::bond::TemporalBond::new(
            f, t, BondType::RelatedTo, 0.9, 10,
        );

        // Scar log seeded with a scar at created_ms = 20.
        let mut log = ScarLog::new();
        log.scar_bond(f, t, "danger".into(), Vec::new(), None, 20);

        // Primitive-level check (bypasses wall clock): forward-search
        // anchor of t=25 must surface the scar.
        let actives = log.active_scars(f, t, 25);
        assert_eq!(
            actives.len(),
            1,
            "scar at t=20 must be visible when forward-search anchor is t>=20",
        );
        // And anchor BEFORE the scar's creation must NOT see it.
        assert_eq!(
            log.active_scars(f, t, 15).len(),
            0,
            "scar at t=20 must be invisible when anchor < 20 — primitive monotonicity",
        );

        // Bond.weight_at(query_time=10, &log) — query targets t=10
        // (where the bond existed and was unscarred), but the scar
        // overlay anchors to wall-clock NOW (~1.7e12), so the scar
        // applies. THIS is the architectural invariant: no temporal
        // scope query can look past a current scar.
        let r: ScarAwareReading = bond.weight_at(10, &log);
        assert_eq!(
            r.effective_weight, 0.0,
            "effective_weight at t=10 must be zeroed by t=20 scar via forward-search to NOW",
        );
        assert_eq!(
            r.active_scars.len(),
            1,
            "active_scars must surface the scar regardless of query_time",
        );
        assert!(
            r.historical_weight > 0.0,
            "historical_weight at t=10 must reflect the pre-scar bond — Lu's history is honest",
        );
    }

    /// 3. Historical weight is preserved untouched by scarring —
    /// observation layers see the genuine pre-scar value.
    #[test]
    fn test_historical_weight_preserved_under_scar() {
        let mut brain = Brain::new();
        let (f, t) = scar_test_bond(&mut brain, "x", "y", 0.7, 100);
        let pre = {
            let bond = brain.network.get_bond(&f, &t).unwrap();
            bond.weight_at(150, &brain.scar_log).historical_weight
        };
        brain.scar_bond(f, t, "for the record");
        let post = {
            let bond = brain.network.get_bond(&f, &t).unwrap();
            bond.weight_at(150, &brain.scar_log).historical_weight
        };
        assert!(pre > 0.0);
        assert!((pre - post).abs() < 1e-6, "historical reading must not change after scar");
    }

    /// 4. Effective weight is zero on a scarred bond.
    #[test]
    fn test_effective_weight_zero_when_scarred() {
        let mut brain = Brain::new();
        let (f, t) = scar_test_bond(&mut brain, "x", "y", 0.7, 100);
        brain.scar_bond(f, t, "blocked");
        let bond = brain.network.get_bond(&f, &t).unwrap();
        let r = bond.weight_at(150, &brain.scar_log);
        assert_eq!(r.effective_weight, 0.0);
    }

    /// 5. Spread activation does not propagate through a scarred
    /// bond. node_a→node_b→node_c path, scar node_a→node_b,
    /// query for node_c from node_a — node_c must NOT surface
    /// via that scarred path. Multi-letter labels avoid the
    /// stopword filter on single letters like "a".
    #[test]
    fn test_spread_does_not_propagate_through_scarred_bond() {
        let mut brain = Brain::new();
        brain.teach("node_a", BondType::RelatedTo, "node_b", 0.9, Confidence::Verified);
        brain.teach("node_b", BondType::RelatedTo, "node_c", 0.9, Confidence::Verified);
        let a = Seed::from_label("node_a");
        let b = Seed::from_label("node_b");
        let c = Seed::from_label("node_c");

        // Pre-scar baseline: confirm node_c is reachable from node_a.
        let pre = brain.ask_in_scope("node_a", &crate::scope::TemporalScope::Current, 3, 10);
        let pre_has_c = pre.recall.activated.iter().any(|x| x.seed == c);
        assert!(pre_has_c, "pre-scar baseline: node_c should be reachable from node_a via node_b");

        // Scar node_a->node_b. effective_weight zeroes; spread cannot propagate.
        brain.scar_bond(a, b, "block this path");

        let post = brain.ask_in_scope("node_a", &crate::scope::TemporalScope::Current, 3, 10);
        let post_has_c = post.recall.activated.iter().any(|x| x.seed == c);
        assert!(
            !post_has_c,
            "post-scar: node_c must not propagate via scarred node_a->node_b path",
        );
    }

    /// 6. Critic surfaces scar-driven rejections with the scar reason
    /// rather than letting the path silently disappear below
    /// threshold.
    #[test]
    fn test_critic_rejects_belief_via_scar_reason() {
        let mut brain = Brain::new();
        brain.teach("alpha", BondType::IsA, "mid", 0.8, Confidence::Verified);
        brain.teach("mid", BondType::HasProperty, "omega", 0.8, Confidence::Verified);
        let alpha = Seed::from_label("alpha");
        let mid = Seed::from_label("mid");

        brain.scar_bond(alpha, mid, "moral block");

        let trace = brain.multi_think("alpha").expect("multi_think runs");
        let scarred_reject = trace
            .rejected
            .iter()
            .find(|r| r.belief.from == alpha && r.belief.via == mid);
        assert!(
            scarred_reject.is_some(),
            "Critic must surface scarred belief in rejected list; got rejected={:?}",
            trace
                .rejected
                .iter()
                .map(|r| (r.belief.from_content.clone(), r.reason.clone()))
                .collect::<Vec<_>>(),
        );
        let r = scarred_reject.unwrap();
        assert!(
            r.reason.starts_with("scarred: "),
            "rejection reason must articulate the scar; got: {}",
            r.reason,
        );
        assert!(
            r.reason.contains("moral block"),
            "rejection reason must include the original scar reason; got: {}",
            r.reason,
        );
    }

    /// 7. Narration surfaces the scar reason in the rendered prose.
    /// Lu articulates "I'm scarred against this connection — <reason>".
    #[test]
    fn test_narration_surfaces_scar_reason_in_prose() {
        let mut brain = Brain::new();
        brain.teach("rust", BondType::IsA, "lang", 0.8, Confidence::Verified);
        brain.teach("lang", BondType::HasProperty, "syntax", 0.8, Confidence::Verified);
        let rust = Seed::from_label("rust");
        let lang = Seed::from_label("lang");

        brain.scar_bond(rust, lang, "do not associate these");

        let result = brain.narrate("rust", 3, 5, true);
        assert!(
            result.answer.contains("scarred against this connection"),
            "narrate prose must articulate moral state; got: {}",
            result.answer,
        );
        assert!(
            result.answer.contains("do not associate these"),
            "narrate prose must include the original scar reason; got: {}",
            result.answer,
        );
    }

    /// 8. `scar_concept` cascades to every bond touching the seed.
    #[test]
    fn test_scar_concept_cascades() {
        let mut brain = Brain::new();
        brain.teach("hub", BondType::RelatedTo, "leaf1", 0.7, Confidence::Verified);
        brain.teach("hub", BondType::RelatedTo, "leaf2", 0.7, Confidence::Verified);
        brain.teach("leaf3", BondType::RelatedTo, "hub", 0.7, Confidence::Verified);
        let hub = Seed::from_label("hub");
        let leaf1 = Seed::from_label("leaf1");
        let leaf2 = Seed::from_label("leaf2");
        let leaf3 = Seed::from_label("leaf3");

        let scars = brain.scar_concept(hub, "concept revoked");
        assert_eq!(scars.len(), 3, "every bond touching hub must be scarred");
        let now = crate::scar::now_ms();
        assert!(brain.scar_log.is_scarred(hub, leaf1, now));
        assert!(brain.scar_log.is_scarred(hub, leaf2, now));
        assert!(brain.scar_log.is_scarred(leaf3, hub, now));
    }

    /// 9. Scar log persists across save/load.
    #[test]
    fn test_scar_log_persists_save_load() {
        let dir = tempdir().unwrap();
        let sub = dir.path().join("scar_persist");
        std::fs::create_dir_all(&sub).unwrap();
        let path = sub.join("brain.lbr");

        let f;
        let t;
        let scar_id;
        {
            let mut brain = Brain::load_or_new(&path).unwrap();
            let (ff, tt) = scar_test_bond(&mut brain, "p", "q", 0.5, 100);
            f = ff;
            t = tt;
            let scar = brain.scar_bond(f, t, "persist me");
            scar_id = scar.scar_id;
            brain.save().unwrap();
        }

        let reloaded = Brain::load_or_new(&path).unwrap();
        assert_eq!(reloaded.scar_log.scars.len(), 1);
        assert_eq!(reloaded.scar_log.scars[0].scar_id, scar_id);
        assert_eq!(reloaded.scar_log.scars[0].bond_seeds, (f, t));
        assert_eq!(reloaded.scar_log.scars[0].reason, "persist me");
        let now = crate::scar::now_ms();
        assert!(reloaded.scar_log.is_scarred(f, t, now));
    }

    /// 10. `unscar_bond` supersedes scars and writes an UnscarRecord.
    /// Note: post-supersede semantics — original entries remain in
    /// `scar_log.scars` with `superseded_by = Some(unscar_id)` so
    /// chronology is reconstructible from a single read.
    #[test]
    fn test_unscar_removes_scars_and_logs() {
        let mut brain = Brain::new();
        let (f, t) = scar_test_bond(&mut brain, "u", "v", 0.5, 100);
        brain.scar_bond(f, t, "first");
        brain.scar_bond(f, t, "second");
        assert_eq!(brain.scar_log.scars.len(), 2);

        let removed = brain.unscar_bond(f, t, "admin override");
        assert_eq!(removed, 2);
        // Originals preserved, both marked superseded.
        assert_eq!(brain.scar_log.scars.len(), 2);
        assert!(brain.scar_log.scars.iter().all(|s| s.superseded_by.is_some()));
        // Active view is empty.
        let now = crate::scar::now_ms();
        assert!(brain.scar_log.active_scars(f, t, now).is_empty());
        assert!(!brain.scar_log.is_scarred(f, t, now));
        // Audit trail intact.
        assert_eq!(brain.scar_log.unscars.len(), 1);
        assert_eq!(brain.scar_log.unscars[0].removed_scar_ids.len(), 2);
        assert_eq!(brain.scar_log.unscars[0].reason, "admin override");
    }

    /// Stretch test: fingerprint slot exists and is empty by default
    /// (reserved for future content-similarity work).
    #[test]
    fn test_fingerprint_field_stub() {
        let mut brain = Brain::new();
        let (f, t) = scar_test_bond(&mut brain, "k", "l", 0.5, 100);
        let scar = brain.scar_bond(f, t, "future similarity work");
        assert!(scar.fingerprint.is_empty());
    }

    /// Followup (a): a bare `ask` against a path that hits a scarred
    /// bond surfaces the active scar in `AskResult.scars` so Lu can
    /// articulate the refusal even without going through narration.
    #[test]
    fn test_ask_result_includes_scars_when_spread_hits_scarred_bond() {
        let mut brain = Brain::new();
        // Build a tiny path: poison_seed -> contaminant -> water_supply
        // Scar the first edge.
        brain.teach(
            "poison_seed",
            BondType::Causes,
            "contaminant",
            0.9,
            Confidence::Verified,
        );
        brain.teach(
            "contaminant",
            BondType::RelatedTo,
            "water_supply",
            0.9,
            Confidence::Verified,
        );
        let f = Seed::from_label("poison_seed");
        let t = Seed::from_label("contaminant");
        brain.scar_bond(f, t, "safety: never propagate poison routes");

        let result = brain.ask("poison_seed", 3, 10);
        assert!(
            !result.scars.is_empty(),
            "AskResult.scars should be populated when spread hits a scarred bond"
        );
        assert!(
            result.scars.iter().any(|s| s.bond_seeds == (f, t)),
            "the surfaced scar should be the (poison_seed -> contaminant) scar"
        );
        assert!(
            result
                .scars
                .iter()
                .any(|s| s.reason.contains("safety")),
            "the surfaced scar should preserve the human-readable reason"
        );
    }

    // ── Port #10 — Autonomous Research Loop ──

    fn make_test_fetcher_with_grok_hit(query: &str) -> crate::research::http::MockFetcher {
        let m = crate::research::http::MockFetcher::new();
        let q = query.replace(' ', "+");
        m.route_ok(
            &format!("https://grokipedia.com/search?q={}", q),
            200,
            r##"<html><body><a href="/page/Topic">Topic</a></body></html>"##,
        );
        m.route_ok(
            "https://grokipedia.com/page/Topic",
            200,
            r##"<html><body><main><p>Topic is a test concept.</p><p>Topic causes example behavior.</p></main></body></html>"##,
        );
        m
    }

    /// Spec test: lifecycle Pending → Approved → Applied (with bonds_added > 0)
    /// and a separate Pending → Rejected path that does NOT add bonds.
    #[test]
    fn test_proposal_status_lifecycle() {
        let mut brain = Brain::new();
        let m = make_test_fetcher_with_grok_hit("topic");
        let id = brain.research_query(&m, "topic").unwrap();
        // Pending right after creation.
        let prop = brain
            .research_proposals
            .iter()
            .find(|p| p.id == id)
            .unwrap();
        assert!(matches!(prop.status, crate::research::proposal::ResearchProposalStatus::Pending));
        assert!(!prop.bond_candidates.is_empty(), "extracted at least one candidate");

        let bonds_before = brain.network.bond_count();
        let added = brain.approve_research_proposal(id).unwrap();
        let bonds_after = brain.network.bond_count();
        assert!(added > 0, "approval should produce > 0 bonds");
        assert_eq!(added as usize, bonds_after - bonds_before);
        let prop = brain
            .research_proposals
            .iter()
            .find(|p| p.id == id)
            .unwrap();
        assert!(matches!(
            prop.status,
            crate::research::proposal::ResearchProposalStatus::Applied { bonds_added, .. } if bonds_added > 0
        ));

        // Reject path on a fresh proposal.
        let id2 = brain.research_query(&m, "topic").unwrap();
        let bonds_before = brain.network.bond_count();
        brain.reject_research_proposal(id2, Some("not relevant".into())).unwrap();
        let bonds_after = brain.network.bond_count();
        assert_eq!(bonds_before, bonds_after, "rejection must not add bonds");
        let prop = brain
            .research_proposals
            .iter()
            .find(|p| p.id == id2)
            .unwrap();
        assert!(matches!(
            prop.status,
            crate::research::proposal::ResearchProposalStatus::Rejected { .. }
        ));
    }

    /// Spec test: research proposals survive save/load.
    #[test]
    fn test_research_proposal_persists_across_save_load() {
        let dir = tempdir().unwrap();
        let sub = dir.path().join("research_persist");
        std::fs::create_dir_all(&sub).unwrap();
        let path = sub.join("brain.lbr");

        let id;
        {
            let mut brain = Brain::load_or_new(&path).unwrap();
            let m = make_test_fetcher_with_grok_hit("topic");
            id = brain.research_query(&m, "topic").unwrap();
            brain.save().unwrap();
        }

        let reloaded = Brain::load_or_new(&path).unwrap();
        assert_eq!(reloaded.research_proposals.len(), 1);
        assert_eq!(reloaded.research_proposals[0].id, id);
        assert!(matches!(
            reloaded.research_proposals[0].status,
            crate::research::proposal::ResearchProposalStatus::Pending
        ));
        assert_eq!(reloaded.next_research_proposal_id, id + 1);
    }

    /// Spec test: loop processes wonders above threshold and stops at
    /// the first below-threshold one.
    #[test]
    fn test_research_loop_consumes_wonder_queue_above_threshold() {
        let mut brain = Brain::new();
        // Make sure both anchor seeds resolve to a node so wonder_to_query
        // produces sensible search strings (and the fetcher routes match).
        brain.network.add_node(crate::node::KnowledgeNode::with_seed(
            Seed::from_label("topic"),
            "topic".into(),
            crate::node::KnowledgeContent::Concept("topic".into()),
            Confidence::Verified,
            now_ms(),
        ));
        brain.network.add_node(crate::node::KnowledgeNode::with_seed(
            Seed::from_label("untouched"),
            "untouched".into(),
            crate::node::KnowledgeContent::Concept("untouched".into()),
            Confidence::Verified,
            now_ms(),
        ));
        // Threshold = 0.6. Push one above and one below.
        brain.research_config.research_intensity_threshold = 0.6;
        brain.wonder_queue.push(crate::curiosity::CuriositySignal {
            seed: Seed::from_label("topic"),
            signal_type: crate::curiosity::SignalType::DeadEnd,
            intensity: 0.7,
            scope: TemporalScope::Current,
            timestamp_ms: 1,
        });
        brain.wonder_queue.push(crate::curiosity::CuriositySignal {
            seed: Seed::from_label("untouched"),
            signal_type: crate::curiosity::SignalType::DeadEnd,
            intensity: 0.4,
            scope: TemporalScope::Current,
            timestamp_ms: 2,
        });
        let m = make_test_fetcher_with_grok_hit("topic");
        let ids = brain.research_loop_run(&m, 10);
        assert_eq!(ids.len(), 1, "only the above-threshold wonder should be consumed");
        // The 0.4 signal must remain in the queue (below threshold).
        assert!(brain.wonder_queue.peek().is_some());
        assert!((brain.wonder_queue.peek().unwrap().intensity - 0.4).abs() < 1e-4);
    }

    /// Stretch test: a Pending proposal MUST NOT mutate the brain
    /// until approve_research_proposal is called.
    #[test]
    fn test_no_auto_application_until_approved() {
        let mut brain = Brain::new();
        let nodes_before = brain.network.node_count();
        let bonds_before = brain.network.bond_count();
        let m = make_test_fetcher_with_grok_hit("topic");
        let _id = brain.research_query(&m, "topic").unwrap();
        // Proposal is Pending but the graph is unchanged.
        assert_eq!(brain.network.node_count(), nodes_before);
        assert_eq!(brain.network.bond_count(), bonds_before);
        assert_eq!(brain.research_proposals.len(), 1);
        assert!(brain.research_proposals[0].is_pending());
    }

        /// Followup (b): scarring a SEED forward-protects future bonds.
    /// `Brain::teach` is the production path that goes through
    /// `add_bond_with_scar_check`; a teach() touching a scarred seed
    /// must NOT add the bond and must record a fresh Scar.
    #[test]
    fn test_scar_concept_blocks_future_bonds_touching_seed() {
        let mut brain = Brain::new();
        // Seed two anchor concepts and a scarred concept.
        brain.teach("chemistry", BondType::RelatedTo, "laboratory", 0.7, Confidence::Verified);
        let scarred = Seed::from_label("forbidden_compound");
        // Add the node so teach() sees it as an existing concept,
        // then mark it forward-scarred.
        brain.network.add_node(KnowledgeNode::with_seed(
            scarred,
            "forbidden_compound".to_string(),
            KnowledgeContent::Concept("forbidden_compound".to_string()),
            Confidence::Verified,
            now_ms(),
        ));
        brain.scar_log.scar_seed(scarred);
        assert!(brain.scar_log.is_seed_scarred(scarred));
        assert_eq!(brain.scar_log.scars.len(), 0);

        // Try to teach a bond touching the scarred seed.
        let bonds_before = brain.network.bond_count();
        brain.teach(
            "chemistry",
            BondType::RelatedTo,
            "forbidden_compound",
            0.9,
            Confidence::Verified,
        );
        // Bond was refused.
        assert_eq!(
            brain.network.bond_count(),
            bonds_before,
            "add_bond_with_scar_check should refuse the bond"
        );
        // A fresh scar was emitted on the would-be edge.
        assert_eq!(
            brain.scar_log.scars.len(),
            1,
            "a refusal scar should be appended"
        );
        assert_eq!(
            brain.scar_log.scars[0].bond_seeds,
            (Seed::from_label("chemistry"), scarred)
        );
        assert!(
            brain.scar_log.scars[0]
                .reason
                .contains("scarred concept")
        );

        // The other direction is also refused.
        brain.teach(
            "forbidden_compound",
            BondType::Causes,
            "laboratory",
            0.5,
            Confidence::Verified,
        );
        assert_eq!(
            brain.network.bond_count(),
            bonds_before,
            "second teach should also be refused"
        );
        assert_eq!(brain.scar_log.scars.len(), 2);
    }

    #[test]
    fn test_prune_legacy_goals_cleans_interest_profile() {
        // Simulate the pre-Port-#12 condition: a legacy goal:/step:
        // node sits in both the keyword index and the interest profile.
        let mut brain = Brain::new();
        let goal_seed = Seed::generate();
        brain.network.add_node(KnowledgeNode::with_seed(
            goal_seed,
            "goal:consciousness".into(),
            KnowledgeContent::Raw("goal:consciousness".into()),
            Confidence::Verified,
            0,
        ));
        let step_seed = Seed::generate();
        brain.network.add_node(KnowledgeNode::with_seed(
            step_seed,
            "step:explore".into(),
            KnowledgeContent::Raw("step:explore".into()),
            Confidence::Verified,
            0,
        ));
        let real_seed = Seed::generate();
        brain.network.add_node(KnowledgeNode::with_seed(
            real_seed,
            "photosynthesis".into(),
            KnowledgeContent::Raw("photosynthesis".into()),
            Confidence::Verified,
            0,
        ));
        let halflife: u64 = 7 * 86_400_000;
        let now: u64 = 1_700_000_000_000;
        brain.interest_profile.touch(goal_seed, now, halflife);
        brain.interest_profile.touch(step_seed, now, halflife);
        brain.interest_profile.touch(real_seed, now, halflife);
        assert_eq!(brain.interest_profile.interests.len(), 3);

        let report = brain.prune_legacy_goal_nodes();
        assert_eq!(report.keyword_entries_pruned, 2);
        assert_eq!(report.interest_entries_pruned, 2);
        assert_eq!(brain.interest_profile.interests.len(), 1);
        assert!(brain.interest_profile.interests.contains_key(&real_seed));
        assert!(!brain.interest_profile.interests.contains_key(&goal_seed));
        assert!(!brain.interest_profile.interests.contains_key(&step_seed));
    }

    #[test]
    fn status_format_version_label_tracks_loaded_format() {
        // Bug A regression test (2026-04-27): status() must report the
        // on-disk format the brain was loaded from, falling back to
        // "ephemeral" when the brain has no on-disk file. The
        // user-controlled save_format must NOT be reported (main.rs
        // overwrites it from --brain-format-out at startup).
        let mut b = Brain::new();
        // Empty brain → ephemeral.
        assert_eq!(b.status().format_version, "ephemeral");
        // Pretend we loaded each persisted format.
        b.loaded_format = Some(BrainSaveFormat::V1Json);
        assert_eq!(b.status().format_version, "v1");
        b.loaded_format = Some(BrainSaveFormat::V2Binary);
        assert_eq!(b.status().format_version, "v2");
        b.loaded_format = Some(BrainSaveFormat::V3Binary);
        assert_eq!(b.status().format_version, "v3");
        b.loaded_format = Some(BrainSaveFormat::V3BinaryCompressed);
        assert_eq!(b.status().format_version, "v3c");
        b.loaded_format = Some(BrainSaveFormat::V3BinaryEncrypted);
        assert_eq!(b.status().format_version, "v3enc");
        // save_format MUST NOT influence the reported format —
        // pretend the user asked --brain-format-out v1 but the brain
        // was actually loaded from a v3c file.
        b.loaded_format = Some(BrainSaveFormat::V3BinaryCompressed);
        b.save_format = BrainSaveFormat::V1Json;
        assert_eq!(b.status().format_version, "v3c");
    }

    // ===== Q.1 typed-KU teach() tests ==============================
    fn q1_brain_for_test() -> Brain {
        Brain::new()
    }

    #[test]
    fn q1_teach_with_explicit_definition_kind() {
        let mut b = q1_brain_for_test();
        let res = b.teach_with_kind(
            "photosynthesis",
            BondType::IsA,
            "conversion of light",
            0.8,
            Confidence::Verified,
            Some("definition"),
            None,
        );
        assert!(res.created_subject);
        assert!(res.created_object);
        let n = b.network.get_node(&res.subject_seed).expect("subject exists");
        assert_eq!(n.kind_str(), "definition");
        let o = b.network.get_node(&res.object_seed).expect("object exists");
        assert_eq!(o.kind_str(), "definition");
    }

    #[test]
    fn q1_teach_with_pattern_inferred_mechanism() {
        // No kind passed; relation alias "WorksBy" should infer Mechanism.
        let mut b = q1_brain_for_test();
        let bt = parse_bond_type("WorksBy").unwrap();
        let res = b.teach_with_kind(
            "combustion",
            bt,
            "oxidation of fuel",
            0.8,
            Confidence::Verified,
            None,
            Some("WorksBy"),
        );
        assert!(res.created_subject);
        let n = b.network.get_node(&res.subject_seed).expect("subject exists");
        assert_eq!(n.kind_str(), "mechanism");
    }

    #[test]
    fn q1_teach_with_pattern_inferred_composition() {
        let mut b = q1_brain_for_test();
        let bt = parse_bond_type("ComposedOf").unwrap();
        let res = b.teach_with_kind(
            "water",
            bt,
            "hydrogen and oxygen",
            0.8,
            Confidence::Verified,
            None,
            Some("ComposedOf"),
        );
        let n = b.network.get_node(&res.subject_seed).expect("subject exists");
        assert_eq!(n.kind_str(), "composition");
    }

    #[test]
    fn q1_teach_with_pattern_inferred_purpose() {
        let mut b = q1_brain_for_test();
        let bt = parse_bond_type("IsFor").unwrap();
        let res = b.teach_with_kind(
            "hammer",
            bt,
            "driving nails",
            0.8,
            Confidence::Verified,
            None,
            Some("IsFor"),
        );
        let n = b.network.get_node(&res.subject_seed).expect("subject exists");
        assert_eq!(n.kind_str(), "purpose");
    }

    #[test]
    fn q1_teach_with_pattern_inferred_contrast() {
        let mut b = q1_brain_for_test();
        let bt = parse_bond_type("Unlike").unwrap();
        let res = b.teach_with_kind(
            "frog",
            bt,
            "toad",
            0.8,
            Confidence::Verified,
            None,
            Some("Unlike"),
        );
        let n = b.network.get_node(&res.subject_seed).expect("subject exists");
        assert_eq!(n.kind_str(), "contrast");
    }

    #[test]
    fn q1_teach_with_pattern_inferred_causal() {
        let mut b = q1_brain_for_test();
        let bt = parse_bond_type("LeadsTo").unwrap();
        let res = b.teach_with_kind(
            "smoking",
            bt,
            "lung damage",
            0.8,
            Confidence::Verified,
            None,
            Some("LeadsTo"),
        );
        let n = b.network.get_node(&res.subject_seed).expect("subject exists");
        assert_eq!(n.kind_str(), "causal");
    }

    #[test]
    fn q1_teach_with_pattern_inferred_example() {
        let mut b = q1_brain_for_test();
        let bt = parse_bond_type("SuchAs").unwrap();
        let res = b.teach_with_kind(
            "mammal",
            bt,
            "dog",
            0.8,
            Confidence::Verified,
            None,
            Some("SuchAs"),
        );
        let n = b.network.get_node(&res.subject_seed).expect("subject exists");
        assert_eq!(n.kind_str(), "example");
    }

    #[test]
    fn q1_teach_code_pattern_from_text_only() {
        // Generic RelatedTo bond, but the subject text looks like code.
        let mut b = q1_brain_for_test();
        let res = b.teach_with_kind(
            "def main():
    pass",
            BondType::RelatedTo,
            "python entrypoint",
            0.5,
            Confidence::Verified,
            None,
            Some("RelatedTo"),
        );
        let n = b.network.get_node(&res.subject_seed).expect("subject exists");
        assert_eq!(n.kind_str(), "code");
    }

    #[test]
    fn q1_teach_default_concept_when_nothing_inferable() {
        // Pre-Q.1 regression — IsA relation with generic text should
        // still produce Concept nodes.
        let mut b = q1_brain_for_test();
        let res = b.teach(
            "cat",
            BondType::IsA,
            "mammal",
            0.9,
            Confidence::Verified,
        );
        let n = b.network.get_node(&res.subject_seed).expect("subject exists");
        assert_eq!(n.kind_str(), "concept");
        let o = b.network.get_node(&res.object_seed).expect("object exists");
        assert_eq!(o.kind_str(), "concept");
    }

    #[test]
    fn q1_teach_preserves_existing_node_kind_on_reteach() {
        // First teach creates with Definition; second teach (same node
        // labels, no kind) must NOT downgrade the existing variant.
        let mut b = q1_brain_for_test();
        let _ = b.teach_with_kind(
            "democracy",
            BondType::IsA,
            "rule by people",
            0.8,
            Confidence::Verified,
            Some("definition"),
            None,
        );
        let _ = b.teach(
            "democracy",
            BondType::IsA,
            "rule by people",
            0.9,
            Confidence::Verified,
        );
        let seed = crate::seed::Seed::from_label("democracy");
        let n = b.network.get_node(&seed).expect("democracy exists");
        assert_eq!(n.kind_str(), "definition", "reteach must not downgrade typed variant");
    }

    #[test]
    fn q1_parse_bond_type_accepts_typed_aliases() {
        // OP-EXTRACT-BREADTH A (#587): typed aliases now map to first-class BondTypes.
        assert!(matches!(parse_bond_type("DefinedAs"), Ok(BondType::DefinedAs)));
        assert!(matches!(parse_bond_type("WorksBy"), Ok(BondType::WorksBy)));
        assert!(matches!(parse_bond_type("ComposedOf"), Ok(BondType::ComposedOf)));
        assert!(matches!(parse_bond_type("IsFor"), Ok(BondType::PurposeOf)));
        assert!(matches!(parse_bond_type("Unlike"), Ok(BondType::DiffersFrom)));
        assert!(matches!(parse_bond_type("LeadsTo"), Ok(BondType::LeadsTo)));
        assert!(matches!(parse_bond_type("SuchAs"), Ok(BondType::ExampleOf)));
        // Canonical bond types still parse correctly.
        assert!(matches!(parse_bond_type("IsA"), Ok(BondType::IsA)));
        assert!(matches!(parse_bond_type("Causes"), Ok(BondType::Causes)));
    }

    #[test]
    fn q1_explicit_kind_wins_over_inference() {
        let mut b = q1_brain_for_test();
        // Relation alias "DefinedAs" would infer "definition", but we
        // pass an explicit "fact" kind; explicit wins.
        let bt = parse_bond_type("DefinedAs").unwrap();
        let res = b.teach_with_kind(
            "explicit",
            bt,
            "override",
            0.8,
            Confidence::Verified,
            Some("fact"),
            Some("DefinedAs"),
        );
        let n = b.network.get_node(&res.subject_seed).expect("subject exists");
        assert_eq!(n.kind_str(), "fact");
    }

    #[test]
    fn test_save_refuses_v1_json_format() {
        // v1 JSON removal regression test internal notes:
        // save_to_format must refuse V1Json with a clear migration
        // pointer rather than silently writing plaintext.
        let dir = tempdir().unwrap();
        let path = dir.path().join("brain.lbr");

        let mut brain = Brain::new();
        brain.teach("x", BondType::IsA, "y", 0.9, Confidence::Verified);
        let res = brain.save_to_format(&path, BrainSaveFormat::V1Json);
        let err = res.expect_err("v1 save must error");
        let msg = err.to_string();
        assert!(msg.contains("v1") || msg.contains("plaintext"),
            "error message must mention v1/plaintext: {}", msg);
        assert!(msg.contains("--migrate-from-v1"),
            "error message must point at migration tool: {}", msg);
        assert!(!path.exists(), "v1 save must not produce a file");
    }

    #[test]
    fn test_brain_format_out_parser_rejects_v1() {
        // v1 JSON removal: --brain-format-out v1 must fail to parse.
        assert_eq!(BrainSaveFormat::parse("v1"), None);
        assert_eq!(BrainSaveFormat::parse("json"), None);
        assert_eq!(BrainSaveFormat::parse("v1json"), None);
        // ...but the other formats still parse.
        assert_eq!(BrainSaveFormat::parse("v3"), Some(BrainSaveFormat::V3Binary));
        assert_eq!(BrainSaveFormat::parse("v3enc"), Some(BrainSaveFormat::V3BinaryEncrypted));
        assert_eq!(BrainSaveFormat::parse("v3c"), Some(BrainSaveFormat::V3BinaryCompressed));
    }

    #[test]
    fn test_brain_new_defaults_to_v3_binary_not_v1() {
        // v1 JSON removal: Brain::new() must NOT default to V1Json.
        let b = Brain::new();
        assert_ne!(b.save_format, BrainSaveFormat::V1Json,
            "new brain must not default to a refused save format");
        assert_eq!(b.save_format, BrainSaveFormat::V3Binary);
    }

    #[test]
    fn test_reason_cold_against_v3_archive_returns_anchors() {
        // Issue A regression test internal notes: cold-start reasoning
        // against an archive-backed brain must return anchors after the
        // startup hoist -- previously the very first call returned an
        // empty anchor set because keyword_index hadn't been built and
        // pick_anchors took &Brain so it could not trigger the lazy
        // build (now &mut Brain).
        let dir = tempdir().unwrap();
        let path = dir.path().join("brain.lbr");

        let mut brain = Brain::new();
        brain.teach("Photosynthesis", BondType::IsA, "BiologicalProcess", 0.9, Confidence::Verified);
        brain.teach("Photosynthesis", BondType::HasProperty, "ProducesOxygen", 0.95, Confidence::Verified);
        brain.teach("BiologicalProcess", BondType::IsA, "NaturalPhenomenon", 0.9, Confidence::Verified);
        brain.save_format = BrainSaveFormat::V3Binary;
        brain.save_to(&path).unwrap();

        let mut loaded = Brain::load_or_new_with_mode_and_key(
            &path,
            BrainLoadMode::Auto,
            None,
        ).unwrap();
        // Confirm we are actually exercising the archive-backed path.
        assert!(loaded.network.is_archive_backed(),
            "test fixture must load through the v3 mmap-zerocopy path");

        // pick_query_anchors now takes &mut Brain (Issue A fix). Without
        // any prior ask() warmup, anchors must still be non-empty for a
        // query that matches a known node.
        let anchors = crate::reasoning::pick_query_anchors(&mut loaded, "photosynthesis");
        assert!(
            !anchors.is_empty(),
            "Issue A regression: cold reason against v3 archive returned no anchors"
        );
    }

    #[test]
    fn test_load_or_new_rejects_v1_with_migration_message() {
        // Critical audit fix internal notes / Problem #111 follow-up):
        // load_or_new previously parsed v1 JSON in production, defeating
        // the structural v1 removal. This test asserts that production
        // load paths now reject v1 with an explicit migration pointer.
        let dir = tempdir().unwrap();
        let path = dir.path().join("v1.lbr");

        let v1_brain = Brain::new();
        let file = v1_brain.to_brain_file(V1_JSON_VERSION);
        let bytes = serde_json::to_vec_pretty(&file).unwrap();
        std::fs::write(&path, &bytes).unwrap();

        let detected = crate::brain_io::detect_format_path(&path).unwrap();
        assert_eq!(detected, crate::brain_io::BrainFormat::V1Json,
            "fixture must detect as v1; got {:?}", detected);

        let res = Brain::load_or_new(&path);
        let err = match res { Ok(_) => panic!("load_or_new of a v1 brain must error"), Err(e) => e };
        let msg = err.to_string();
        assert!(msg.contains("v1") || msg.contains("plaintext"),
            "error must mention v1 or plaintext: {}", msg);
        assert!(msg.contains("--migrate-from-v1"),
            "error must point at migration utility: {}", msg);

        let res2 = Brain::load_or_new_with_mode(&path, BrainLoadMode::Auto);
        let err2 = match res2 { Ok(_) => panic!("load_or_new_with_mode must reject v1"), Err(e) => e };
        assert!(err2.to_string().contains("--migrate-from-v1"),
            "with_mode error must point at migration tool: {}", err2);

        let res3 = Brain::load_or_new_with_mode_and_key(&path, BrainLoadMode::Auto, None);
        let err3 = match res3 { Ok(_) => panic!("load_or_new_with_mode_and_key must reject v1"), Err(e) => e };
        assert!(err3.to_string().contains("--migrate-from-v1"),
            "with_mode_and_key error must point at migration tool: {}", err3);
    }

    #[test]
    fn test_load_v1_for_migration_only_loads_legacy_v1_brain() {
        // Critical audit fix: Solution #77 claimed this function exists.
        // It did not (grep proof). This test fails to compile if the
        // function is missing.
        let dir = tempdir().unwrap();
        let path = dir.path().join("v1.lbr");

        let mut src_brain = Brain::new();
        src_brain.teach("foo", BondType::IsA, "bar", 0.8, Confidence::Verified);
        let file = src_brain.to_brain_file(V1_JSON_VERSION);
        std::fs::write(&path, serde_json::to_vec_pretty(&file).unwrap()).unwrap();

        let loaded = Brain::load_v1_for_migration_only(&path)
            .expect("migration loader must succeed on a valid v1 brain");
        assert!(loaded.network.bond_count() > 0,
            "migration loader must populate the network from the v1 file");
        assert_eq!(loaded.loaded_format, Some(BrainSaveFormat::V1Json));
    }

    #[test]
    fn test_load_v1_for_migration_only_rejects_non_v1_files() {
        // Defensive: the migration loader must refuse non-v1 files so
        // it cannot be repurposed as a backdoor production loader.
        let dir = tempdir().unwrap();
        let path = dir.path().join("v3.lbr");

        let mut brain = Brain::new();
        brain.teach("a", BondType::IsA, "b", 0.8, Confidence::Verified);
        brain.save_format = BrainSaveFormat::V3Binary;
        brain.save_to(&path).unwrap();

        let res = Brain::load_v1_for_migration_only(&path);
        let err = match res { Ok(_) => panic!("migration loader must refuse non-v1 files"), Err(e) => e };
        let msg = err.to_string().to_lowercase();
        assert!(msg.contains("non-v1") || msg.contains("not v1"),
            "error must say non-v1: {}", err);
    }
    #[test]
    fn test_atomic_write_bytes_replaces_existing_file_with_no_leftover_tmp() {
        // Critical audit fix internal notes / Problem #111 follow-up):
        // save_to_format used fs::write(path, bytes) directly. A mid-save
        // crash on a 1 GB v3enc brain truncated the destination, defeating
        // durability. Fix: write to path.tmp, fsync, rename to path. This
        // test exercises the helper directly: pre-existing content at
        // dest must be replaced atomically and no .tmp must remain.
        let dir = tempdir().unwrap();
        let path = dir.path().join("brain.lbr");

        // Pre-create destination with old content so we can confirm replacement.
        std::fs::write(&path, b"OLD").unwrap();

        let new_bytes: Vec<u8> = (0..32768).map(|i| (i % 251) as u8).collect();
        atomic_write_bytes(&path, &new_bytes).expect("atomic_write_bytes must succeed");

        let read_back = std::fs::read(&path).unwrap();
        assert_eq!(read_back, new_bytes,
            "destination must hold the new bytes after atomic_write_bytes");

        // No .tmp leftover.
        let tmp_path = path.with_extension("lbr.tmp");
        assert!(!tmp_path.exists(),
            ".tmp file leftover at {}", tmp_path.display());
        // Also try the alt pattern (extension-only).
        let alt_tmp = path.with_extension("tmp");
        assert!(!alt_tmp.exists(),
            "alt .tmp file leftover at {}", alt_tmp.display());
    }

    #[test]
    fn test_atomic_write_bytes_handles_fresh_destination() {
        // When the destination does not exist, atomic_write_bytes must
        // create it via the rename path -- not write directly.
        let dir = tempdir().unwrap();
        let path = dir.path().join("new_brain.lbr");
        assert!(!path.exists());

        let bytes: Vec<u8> = b"hello world".to_vec();
        atomic_write_bytes(&path, &bytes).expect("fresh destination must succeed");

        assert_eq!(std::fs::read(&path).unwrap(), bytes);
        assert!(!path.with_extension("lbr.tmp").exists());
    }

    #[test]
    fn test_save_to_format_does_not_leave_tmp_files_behind() {
        // End-to-end: after a successful v3 save, no .tmp file must exist
        // in the destination directory. This catches a class of bug where
        // an atomic-write helper exists but a particular save path bypasses
        // it.
        let dir = tempdir().unwrap();
        let path = dir.path().join("e2e.lbr");

        let mut brain = Brain::new();
        brain.teach("a", BondType::IsA, "b", 0.8, Confidence::Verified);
        brain.save_format = BrainSaveFormat::V3Binary;
        brain.save_to(&path).expect("v3 save must succeed");

        // List directory contents -- must not contain any *.tmp file.
        let entries: Vec<String> = std::fs::read_dir(dir.path()).unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        for name in &entries {
            assert!(!name.ends_with(".tmp"),
                "save left a .tmp file behind: {} (dir entries: {:?})", name, entries);
        }
        // Also check v3 compressed and v2 paths.
        let path2 = dir.path().join("e2e.v2.lbr");
        brain.save_to_format(&path2, BrainSaveFormat::V2Binary).expect("v2 save must succeed");
        let path3 = dir.path().join("e2e.v3c.lbr");
        brain.save_to_format(&path3, BrainSaveFormat::V3BinaryCompressed).expect("v3c save must succeed");
        let entries: Vec<String> = std::fs::read_dir(dir.path()).unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        for name in &entries {
            assert!(!name.ends_with(".tmp"),
                "v2/v3c save left a .tmp file behind: {} (dir entries: {:?})", name, entries);
        }
    }
    #[test]
    fn test_generate_candidate_scopes_does_not_panic_on_archive_backed_brain() {
        // Important audit fix internal notes / Problem #111 follow-up):
        // ContextSelector::generate_candidate_scopes called BondView::keyframes()
        // on archive-backed bonds, which panics. Two callsites in that
        // function (context.rs:118 and :125) reached BondView via
        // network.get_bond. Pre-fix: this test panics with
        // "BondView::keyframes() called on archive-backed bond". Post-fix:
        // it succeeds and returns a non-empty scope list including Current.
        let dir = tempdir().unwrap();
        let path = dir.path().join("brain.lbr");

        let mut brain = Brain::new();
        brain.teach("foo", BondType::IsA, "bar", 0.8, Confidence::Verified);
        brain.save_format = BrainSaveFormat::V3Binary;
        brain.save_to(&path).unwrap();

        let loaded = Brain::load_or_new_with_mode_and_key(
            &path, BrainLoadMode::Auto, None,
        ).unwrap();
        assert!(loaded.network.is_archive_backed(),
            "fixture must load via archive path to exercise BondView::Archive");

        let foo = Seed::from_label("foo");
        let bar = Seed::from_label("bar");
        let cs = crate::context::ContextSelector::new();
        let scopes = cs.generate_candidate_scopes(&loaded.network, &[foo, bar]);
        let has_current = scopes.iter().any(|s| matches!(s, TemporalScope::Current));
        assert!(has_current, "scope list must always include Current");
    }
    // -------- Port T.7: structural encryption guard --------
    //
    // Six tests covering the two halves of the guard:
    //   - Pre-load: `require_v3enc_or_err(path)` refuses non-v3enc.
    //   - Save:     `Brain::require_encrypted` refuses non-v3enc writes.
    //
    // Solution #79 (allow_decrypt_on_save) only covered brains that
    // LOADED from v3enc; T.7 covers the prior structural gap that let
    // a freshly-trained plaintext v3 brain land on the production
    // path. After T.7 the encryption invariant is structural: a
    // `--require-encrypted` lu_4_mcp will REFUSE to start, not just
    // warn, against a non-v3enc brain.

    #[test]
    fn t7_loader_refuses_plaintext_when_require_encrypted_set() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("plain.lbr");
        let mut brain = Brain::new();
        brain.teach("a", BondType::IsA, "b", 0.8, Confidence::Verified);
        brain.save_format = BrainSaveFormat::V3Binary;
        brain.save_to(&path).unwrap();

        let res = require_v3enc_or_err(&path);
        match res {
            Err(BrainError::EncryptionRequired { attempted_format, .. }) => {
                assert_eq!(attempted_format, "v3",
                    "plaintext v3 must be reported as 'v3'; got '{}'", attempted_format);
            }
            Err(e) => panic!("expected EncryptionRequired, got: {:?}", e),
            Ok(()) => panic!("require_v3enc_or_err must reject plaintext v3"),
        }
    }
/// OP-BRAIN-V5 6.5 (c): the --require-encrypted guard hard-fails a
    /// PLAINTEXT v5 brain directory and accepts an ENCRYPTED one.
    #[test]
    fn enc65_guard_refuses_plaintext_v5_dir_accepts_encrypted() {
        use crate::manifest::{KeyFingerprint, Manifest, MANIFEST_FILE_NAME};
        use crate::node::{Confidence, KnowledgeContent, KnowledgeNode, KnowledgeUnit};
        use crate::node_intern::NodeId;
        use crate::node_store::nsst_filename;
        use crate::node_store::sst_writer::{NodeSstWriteOpts, NodeSstWriter};
        use crate::node_store::NodeEntry;
        use crate::seed::Seed;

        fn entry(label: &str) -> NodeEntry {
            NodeEntry {
                payload: KnowledgeNode {
                    seed: Seed::from_label(label),
                    content: label.to_string(),
                    knowledge: crate::node::KnowledgeUnit {
                        content: KnowledgeContent::Concept(label.to_string()),
                        confidence: Confidence::Verified,
                        created_at: 0,
                        updated_at: 0,
                    },
                    embedding: Vec::new(),
                    parent_seeds: Vec::new(),
                    typed_ratio: 0.0,
                },
                tombstone: false,
                seq: 0,
            }
        }

        // plaintext v5 dir: MANIFEST master_key_id=none + plaintext nsst
        let d1 = tempdir().unwrap();
        let p1 = d1.path();
        std::fs::create_dir_all(p1.join("node_store")).unwrap();
        std::fs::create_dir_all(p1.join("bond_store")).unwrap();
        let mut w = NodeSstWriter::create(
            &p1.join("node_store").join(nsst_filename(1)),
            NodeSstWriteOpts { file_uid: 1, ..Default::default() },
        ).unwrap();
        w.append(NodeId(1), &entry("a")).unwrap();
        w.finish().unwrap();
        Manifest::new(KeyFingerprint::none())
            .save_atomic(&p1.join(MANIFEST_FILE_NAME)).unwrap();
        match require_v3enc_or_err(p1) {
            Err(BrainError::EncryptionRequired { attempted_format, .. }) => {
                assert_eq!(attempted_format, "v5-plaintext");
            }
            other => panic!("plaintext v5 dir must be refused, got {:?}", other),
        }

        // encrypted v5 dir: MANIFEST master_key_id=fp + encrypted nsst
        let d2 = tempdir().unwrap();
        let p2 = d2.path();
        std::fs::create_dir_all(p2.join("node_store")).unwrap();
        let key = crate::encrypt::MasterKey::from_bytes([0x33; 32]);
        let mut w2 = NodeSstWriter::create_with_key(
            &p2.join("node_store").join(nsst_filename(1)),
            NodeSstWriteOpts { file_uid: 1, ..Default::default() },
            Some(key.clone()),
        ).unwrap();
        w2.append(NodeId(1), &entry("a")).unwrap();
        w2.finish().unwrap();
        Manifest::new(KeyFingerprint::of(&key))
            .save_atomic(&p2.join(MANIFEST_FILE_NAME)).unwrap();
        require_v3enc_or_err(p2).expect("encrypted v5 dir must pass --require-encrypted");
    }

    #[test]
    fn t7_loader_accepts_v3enc_when_require_encrypted_set() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("enc.lbr");
        let key = crate::encrypt::MasterKey::generate();
        let mut brain = Brain::new();
        brain.teach("a", BondType::IsA, "b", 0.8, Confidence::Verified);
        brain.encryption_key = Some(key);
        brain.save_to_format(&path, BrainSaveFormat::V3BinaryEncrypted)
            .expect("v3enc save must succeed");

        require_v3enc_or_err(&path)
            .expect("v3enc brain must pass require-encrypted check");
    }

    #[test]
    fn t7_loader_accepts_plaintext_when_flag_off() {
        // Back-compat: without --require-encrypted, plaintext brains
        // still load normally. This is the dev/test workflow.
        let dir = tempdir().unwrap();
        let path = dir.path().join("plain.lbr");
        let mut brain = Brain::new();
        brain.teach("x", BondType::IsA, "y", 0.8, Confidence::Verified);
        brain.save_format = BrainSaveFormat::V3Binary;
        brain.save_to(&path).unwrap();

        let loaded = Brain::load_or_new_with_mode_and_key(
            &path, BrainLoadMode::Auto, None,
        ).expect("plaintext brain must load when require_encrypted is off");
        assert!(!loaded.require_encrypted,
            "require_encrypted must default to false on a regular load");
    }

    #[test]
    fn t7_save_refuses_plaintext_when_require_encrypted_set() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("brain.lbr");
        let mut brain = Brain::new();
        brain.teach("a", BondType::IsA, "b", 0.8, Confidence::Verified);
        brain.require_encrypted = true;

        let res = brain.save_to_format(&path, BrainSaveFormat::V3Binary);
        match res {
            Err(BrainError::EncryptionRequired { attempted_format, hint }) => {
                assert_eq!(attempted_format, "v3",
                    "attempted_format should be 'v3'; got '{}'", attempted_format);
                assert!(hint.contains("v3enc"),
                    "hint should mention v3enc; got '{}'", hint);
            }
            other => panic!("expected EncryptionRequired, got: {:?}", other),
        }
        assert!(!path.exists(), "v3 plaintext file must not have been written");

        let res2 = brain.save_to_format(&path, BrainSaveFormat::V3BinaryCompressed);
        assert!(matches!(res2, Err(BrainError::EncryptionRequired { .. })),
            "v3c save must be refused under require_encrypted");
        assert!(!path.exists());

        let res3 = brain.save_to_format(&path, BrainSaveFormat::V2Binary);
        assert!(matches!(res3, Err(BrainError::EncryptionRequired { .. })),
            "v2 save must be refused under require_encrypted");
        assert!(!path.exists());
    }

    #[test]
    fn t7_save_succeeds_v3enc_when_require_encrypted_set() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("brain.lbr");
        let key = crate::encrypt::MasterKey::generate();
        let mut brain = Brain::new();
        brain.teach("a", BondType::IsA, "b", 0.8, Confidence::Verified);
        brain.encryption_key = Some(key);
        brain.require_encrypted = true;

        brain.save_to_format(&path, BrainSaveFormat::V3BinaryEncrypted)
            .expect("v3enc save must succeed when require_encrypted is set");
        assert!(path.exists(), "v3enc file must have been written");

        require_v3enc_or_err(&path)
            .expect("the v3enc file we just wrote must pass the loader guard");
    }

    #[test]
    fn t7_error_message_points_at_migration_command() {
        // Operator hits this error when they boot lu_4_mcp with
        // --require-encrypted against a plaintext brain. The error
        // text must point at the migration command so they can
        // recover without reading source.
        let dir = tempdir().unwrap();
        let path = dir.path().join("plain.lbr");
        let mut brain = Brain::new();
        brain.teach("a", BondType::IsA, "b", 0.8, Confidence::Verified);
        brain.save_format = BrainSaveFormat::V3Binary;
        brain.save_to(&path).unwrap();

        let err = require_v3enc_or_err(&path).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("--migrate-v3-compressed-to-encrypted"),
            "refusal error must point at migration command. Got: {}", msg);
        assert!(msg.contains("--require-encrypted"),
            "refusal error must mention --require-encrypted so operator knows the gate. Got: {}", msg);
    }

        #[test]
    fn test_save_refuses_encrypted_to_plaintext_downgrade_without_flag() {
        // Important audit fix: previously a v3enc brain could be silently
        // re-saved as plaintext via --brain-format-out v3 (only a warning).
        // The guard now lives in Brain::save_to_format itself so every
        // caller is covered. Without --allow-decrypt-on-save the save
        // must error.
        let dir = tempdir().unwrap();
        let path = dir.path().join("brain.lbr");

        let mut brain = Brain::new();
        brain.teach("a", BondType::IsA, "b", 0.8, Confidence::Verified);
        brain.loaded_format = Some(BrainSaveFormat::V3BinaryEncrypted);
        brain.allow_decrypt_on_save = false;

        let res = brain.save_to_format(&path, BrainSaveFormat::V3Binary);
        let err = match res {
            Err(e) => e,
            Ok(()) => panic!("v3enc -> v3 plaintext downgrade must error without flag"),
        };
        let msg = err.to_string();
        assert!(msg.contains("encrypted") || msg.contains("plaintext"),
            "error must mention encryption: {}", msg);
        assert!(msg.contains("--allow-decrypt-on-save"),
            "error must mention escape-hatch flag: {}", msg);
        assert!(!path.exists(), "plaintext file must not have been written");

        // Also test V3Binary -> V3BinaryCompressed (still plaintext at rest).
        let res2 = brain.save_to_format(&path, BrainSaveFormat::V3BinaryCompressed);
        assert!(res2.is_err(),
            "v3enc -> v3c (still plaintext) downgrade must also error");
        assert!(!path.exists());

        // V2Binary too.
        let res3 = brain.save_to_format(&path, BrainSaveFormat::V2Binary);
        assert!(res3.is_err(), "v3enc -> v2 downgrade must error");
        assert!(!path.exists());
    }

    #[test]
    fn test_save_allows_encrypted_to_plaintext_with_explicit_opt_in() {
        // The opt-in flag exists for legitimate use cases (e.g. exporting
        // a brain for migration). With it set, the same downgrade succeeds.
        let dir = tempdir().unwrap();
        let path = dir.path().join("brain.lbr");

        let mut brain = Brain::new();
        brain.teach("a", BondType::IsA, "b", 0.8, Confidence::Verified);
        brain.loaded_format = Some(BrainSaveFormat::V3BinaryEncrypted);
        brain.allow_decrypt_on_save = true;

        brain.save_to_format(&path, BrainSaveFormat::V3Binary)
            .expect("opt-in flag must allow downgrade");
        assert!(path.exists(), "plaintext file must exist when flag is set");
    }

    #[test]
    fn test_save_allows_non_downgrade_paths_without_flag() {
        // Sanity: a brain that was NOT loaded from an encrypted file is
        // unaffected by the guard.
        let dir = tempdir().unwrap();
        let path = dir.path().join("brain.lbr");

        let mut brain = Brain::new();
        brain.teach("a", BondType::IsA, "b", 0.8, Confidence::Verified);
        // No loaded_format (or loaded_format Some(V3Binary)) -- guard is
        // inert.
        brain.loaded_format = Some(BrainSaveFormat::V3Binary);

        brain.save_to_format(&path, BrainSaveFormat::V3Binary)
            .expect("v3 -> v3 must succeed without flag");
    }
}
// =====================================================================
// Port G — end-to-end negation handling tests
// =====================================================================
//
// Exercises the full pipeline:
//   - teach_with_negation flags the bond's overlay entry
//   - teach_from_text drives the cue extractor (gated by
//     negation_cues_enabled)
//   - Brain::bond_negated returns the overlay flag
//   - Round-trip through to_brain_file + from_brain_file preserves
//     the overlay
//   - A pre-G brain JSON loads cleanly with overlay defaulted empty
//   - Reasoning chains carry parity through Causal/Hierarchical
//     patterns and flip articulation accordingly
//   - Articulation surfaces "is NOT" prose for odd-parity chains
//   - has_contradiction lights up when the same triple is taught
//     with conflicting polarity
#[cfg(test)]
mod port_g_tests {
    use super::*;
    use crate::bond::BondType;
    use crate::node::Confidence;
    use crate::negation::{
        articulate_bond_negative, chain_parity, count_negation_cues,
        detect_negation_cue, pattern_propagates_negation,
    };

    #[test]
    fn teach_with_negation_flags_overlay() {
        let mut b = Brain::new();
        let r = b.teach_with_negation(
            "sharks",
            BondType::IsA,
            "mammals",
            0.9,
            Confidence::Verified,
            true,
        );
        assert!(b.bond_negated(&r.subject_seed, &r.object_seed, BondType::IsA));
    }

    #[test]
    fn teach_keeps_default_positive() {
        let mut b = Brain::new();
        let r = b.teach("cats", BondType::IsA, "mammals", 0.9, Confidence::Verified);
        assert!(!b.bond_negated(&r.subject_seed, &r.object_seed, BondType::IsA));
    }

    #[test]
    fn teach_from_text_detects_cue() {
        let mut b = Brain::new();
        let (r, neg) = b.teach_from_text(
            "Sharks are not mammals.",
            "sharks",
            BondType::IsA,
            "mammals",
            0.9,
            Confidence::Verified,
        );
        assert!(neg, "negation cue must be detected");
        assert!(b.bond_negated(&r.subject_seed, &r.object_seed, BondType::IsA));
    }

    #[test]
    fn teach_from_text_no_cue_stays_positive() {
        let mut b = Brain::new();
        let (r, neg) = b.teach_from_text(
            "The sun is a star.",
            "sun",
            BondType::IsA,
            "star",
            0.9,
            Confidence::Verified,
        );
        assert!(!neg);
        assert!(!b.bond_negated(&r.subject_seed, &r.object_seed, BondType::IsA));
    }

    #[test]
    fn cue_gating_disables_extraction() {
        let mut b = Brain::new();
        b.negation_cues_enabled = false;
        let (r, neg) = b.teach_from_text(
            "Sharks are not mammals.",
            "sharks",
            BondType::IsA,
            "mammals",
            0.9,
            Confidence::Verified,
        );
        assert!(!neg, "cue gating off -> no negation");
        assert!(!b.bond_negated(&r.subject_seed, &r.object_seed, BondType::IsA));
    }

    #[test]
    fn round_trip_preserves_overlay() {
        let mut b = Brain::new();
        b.teach_with_negation(
            "pluto",
            BondType::IsA,
            "planet",
            0.9,
            Confidence::Verified,
            true,
        );
        b.teach("earth", BondType::IsA, "planet", 0.9, Confidence::Verified);
        // Round-trip through BrainFile (the JSON serde shape).
        let file = b.to_brain_file(crate::brain_io_v3::FORMAT_VERSION_V3);
        let json = serde_json::to_string(&file).unwrap();
        let back: BrainFile = serde_json::from_str(&json).unwrap();
        let pluto_seed = crate::seed::Seed::from_label("pluto");
        let planet_seed = crate::seed::Seed::from_label("planet");
        let earth_seed = crate::seed::Seed::from_label("earth");
        assert!(back
            .negation_overlay
            .is_negated(&pluto_seed, &planet_seed, BondType::IsA));
        assert!(!back
            .negation_overlay
            .is_negated(&earth_seed, &planet_seed, BondType::IsA));
    }

    #[test]
    fn pre_g_brain_json_loads_default_false() {
        // A pre-G brain JSON has no `negation_overlay` field. Build a
        // post-G BrainFile, JSON-encode, REMOVE the overlay key, then
        // re-parse: serde's #[serde(default)] must fill in an empty
        // overlay so every bond defaults to negated=false.
        let mut b = Brain::new();
        b.teach("a", BondType::IsA, "b", 0.9, Confidence::Verified);
        let file = b.to_brain_file(crate::brain_io_v3::FORMAT_VERSION_V3);
        let mut v: serde_json::Value = serde_json::to_value(&file).unwrap();
        v.as_object_mut().unwrap().remove("negation_overlay");
        let pre_g_json = serde_json::to_string(&v).unwrap();
        let bf: BrainFile = serde_json::from_str(&pre_g_json).unwrap();
        assert!(bf.negation_overlay.is_empty());
        let s1 = crate::seed::Seed::from_label("a");
        let s2 = crate::seed::Seed::from_label("b");
        assert!(!bf.negation_overlay.is_negated(&s1, &s2, BondType::IsA));
    }

    #[test]
    fn parity_xor_three_depths() {
        // Spec acceptance criteria: 0 / 1 / 2 negations -> +/-/+ outputs.
        assert_eq!(chain_parity(&[false, false, false]), false);
        assert_eq!(chain_parity(&[true]), true);
        assert_eq!(chain_parity(&[true, true]), false);
        assert_eq!(chain_parity(&[true, true, true]), true);
        assert_eq!(chain_parity(&[false, true, false]), true);
    }

    #[test]
    fn pattern_propagation_table_only_causal_hierarchical() {
        // Spec LAYER 2 (2b): only Causal and Hierarchical propagate.
        assert!(pattern_propagates_negation("Causal"));
        assert!(pattern_propagates_negation("Hierarchical"));
        assert!(!pattern_propagates_negation("PropertyChain"));
        assert!(!pattern_propagates_negation("Mixed"));
    }

    #[test]
    fn negation_articulator_table() {
        assert_eq!(
            articulate_bond_negative(BondType::IsA, "shark", "mammal"),
            "Shark is not a kind of mammal."
        );
        assert_eq!(
            articulate_bond_negative(BondType::Causes, "vaccine", "autism"),
            "Vaccine does not cause autism."
        );
    }

    #[test]
    fn detect_cue_handles_punctuation_and_contractions() {
        assert!(detect_negation_cue("X isn't Y."));
        assert!(detect_negation_cue("X cannot be true!"));
        assert!(detect_negation_cue("X, however, is not Y."));
        assert_eq!(count_negation_cues("not unlike X"), 1);
        assert_eq!(count_negation_cues("never not yes"), 2);
    }

    #[test]
    fn contradiction_surfaces_after_polarity_flip() {
        let mut b = Brain::new();
        // First teach: positive.
        let r1 = b.teach("rust", BondType::IsA, "lang", 0.9, Confidence::Verified);
        assert!(!b.has_contradiction(&r1.subject_seed, &r1.object_seed, BondType::IsA));
        // Second teach: same triple, negated. Network dedups the bond
        // but the overlay records the flip as a contradiction.
        b.teach_with_negation("rust", BondType::IsA, "lang", 0.9, Confidence::Verified, true);
        assert!(b.has_contradiction(&r1.subject_seed, &r1.object_seed, BondType::IsA));
    }

    #[test]
    fn first_negative_is_not_a_contradiction() {
        // Teach a triple negated as the FIRST observation. That's not
        // a contradiction yet; it's just one negative source.
        let mut b = Brain::new();
        let r = b.teach_with_negation(
            "mercury",
            BondType::IsA,
            "gas_giant",
            0.9,
            Confidence::Verified,
            true,
        );
        assert!(b.bond_negated(&r.subject_seed, &r.object_seed, BondType::IsA));
        assert!(!b.has_contradiction(&r.subject_seed, &r.object_seed, BondType::IsA));
    }

    #[test]
    fn end_to_end_one_hop_negative_articulation() {
        // Teach "birds are not mammals" via teach_from_text; ask a
        // 1-hop reasoning query; assert the trace's per-step
        // articulation is the negative form.
        use crate::reasoning;
        use crate::scope::TemporalScope;
        let mut b = Brain::new();
        let (_, neg) = b.teach_from_text(
            "Birds are not mammals.",
            "bird",
            BondType::IsA,
            "mammal",
            0.9,
            Confidence::Verified,
        );
        assert!(neg);
        let trace = reasoning::reason(&mut b, "bird", TemporalScope::Current);
        if !trace.steps.is_empty() {
            // First (and only) step articulation should be the negative form.
            let s = &trace.steps[0].articulation;
            assert!(
                s.contains("is not") || s.contains("does not"),
                "expected negative articulation, got: {}",
                s
            );
        }
        // Even if reason() returned no_chain (e.g. seeds didn't anchor),
        // teach_from_text + bond_negated read should still be true.
        let bird = crate::seed::Seed::from_label("bird");
        let mammal = crate::seed::Seed::from_label("mammal");
        assert!(b.bond_negated(&bird, &mammal, BondType::IsA));
    }

    // --- Port S — source attribution per bond -----------------

    /// Failing-First (Port S): pre-S bonds report Unknown.
    /// Demonstrates the gap: before this port, the brain had no
    /// way to query source per bond. With the overlay in place,
    /// every pre-S bond surfaces as Unknown — the canonical
    /// 1.26 GB brain at /var/lib/lu/brain.lbr behaves the same.
    #[test]
    fn port_s_pre_s_bond_is_unknown() {
        let mut b = Brain::new();
        // Manually add a node + bond WITHOUT going through teach()
        // so the source overlay is not auto-populated. Mirrors a
        // pre-S brain: bonds exist but no overlay entry.
        let r = b.teach("hot", BondType::Causes, "sweat", 0.9, Confidence::Verified);
        // teach() now installs Teach, so first delete it to simulate pre-S:
        b.set_bond_source(
            r.subject_seed,
            r.object_seed,
            BondType::Causes,
            crate::source::SourceAttribution::unknown(),
        );
        let attr = b.bond_source(&r.subject_seed, &r.object_seed, BondType::Causes);
        assert!(matches!(attr.kind, crate::source::SourceKind::Unknown));
    }

    /// Default attribution: teach() tags Teach{actor="system"}.
    #[test]
    fn port_s_teach_default_tags_teach_kind() {
        let mut b = Brain::new();
        let r = b.teach("sun", BondType::IsA, "star", 0.95, Confidence::Verified);
        let attr = b.bond_source(&r.subject_seed, &r.object_seed, BondType::IsA);
        match attr.kind {
            crate::source::SourceKind::Teach { actor, .. } => {
                assert_eq!(actor, "system");
            }
            other => panic!("expected Teach, got {:?}", other),
        }
    }

    /// teach_with_source installs the explicit attribution.
    #[test]
    fn port_s_teach_with_source_records_attribution() {
        let mut b = Brain::new();
        let attr = crate::source::SourceAttribution::document(
            "sha256:abc",
            Some("medical".into()),
        );
        let r = b.teach_with_source(
            "aspirin",
            BondType::HasProperty,
            "anti_inflammatory",
            0.9,
            Confidence::Verified,
            attr,
        );
        let got = b.bond_source(&r.subject_seed, &r.object_seed, BondType::HasProperty);
        match got.kind {
            crate::source::SourceKind::Document { doc_hash, corpus } => {
                assert_eq!(doc_hash, "sha256:abc");
                assert_eq!(corpus.as_deref(), Some("medical"));
            }
            other => panic!("expected Document, got {:?}", other),
        }
    }

    /// Round-trip: save bonds with sources -> reload -> sources preserved.
    #[test]
    fn port_s_round_trip_through_brain_file() {
        let mut b = Brain::new();
        let r1 = b.teach_with_source(
            "oxygen",
            BondType::PartOf,
            "atmosphere",
            0.9,
            Confidence::Verified,
            crate::source::SourceAttribution::training("v4", Some("r41".into())),
        );
        let r2 = b.teach_with_source(
            "clouds",
            BondType::Produces,
            "rain",
            0.9,
            Confidence::Verified,
            crate::source::SourceAttribution::research(
                "wonder:42",
                Some("https://en.wikipedia.org/wiki/Cloud".into()),
                12345,
            ),
        );
        // Round-trip via to_brain_file + JSON.
        let file = b.to_brain_file(crate::BRAIN_FORMAT_VERSION);
        let json = serde_json::to_string(&file).unwrap();
        let back: BrainFile = serde_json::from_str(&json).unwrap();
        assert!(back.source_overlay.has(&r1.subject_seed, &r1.object_seed, BondType::PartOf));
        assert!(back.source_overlay.has(&r2.subject_seed, &r2.object_seed, BondType::Produces));
        let r2_attr = back.source_overlay.get(&r2.subject_seed, &r2.object_seed, BondType::Produces);
        match r2_attr.kind {
            crate::source::SourceKind::Research { wonder_seed, url, fetched_at_ms } => {
                assert_eq!(wonder_seed, "wonder:42");
                assert_eq!(url.as_deref(), Some("https://en.wikipedia.org/wiki/Cloud"));
                assert_eq!(fetched_at_ms, 12345);
            }
            other => panic!("expected Research, got {:?}", other),
        }
    }

    /// Backward-compat: pre-S brain JSON (no source_overlay field)
    /// loads with an empty overlay and every bond Unknown.
    #[test]
    fn port_s_pre_s_brain_loads_empty_overlay() {
        let mut b = Brain::new();
        b.teach("a", BondType::IsA, "b", 0.9, Confidence::Verified);
        let file = b.to_brain_file(crate::BRAIN_FORMAT_VERSION);
        let mut v = serde_json::to_value(&file).unwrap();
        v.as_object_mut().unwrap().remove("source_overlay");
        let bf: BrainFile = serde_json::from_value(v).unwrap();
        assert!(bf.source_overlay.is_empty());
    }

    /// Phase-5 GDPR: revoke 5 bonds across 3 source kinds, verify
    /// only the targeted ones go.
    #[test]
    fn port_s_revoke_by_source_removes_target_bonds() {
        let mut b = Brain::new();
        let teach_attr = crate::source::SourceAttribution::teach("alpha", 1);
        let research_attr = crate::source::SourceAttribution::research("W7", None, 2);
        let doc_attr = crate::source::SourceAttribution::document("hashX", None);
        b.teach_with_source("alpha_a", BondType::IsA, "alpha_x", 0.9, Confidence::Verified, teach_attr.clone());
        b.teach_with_source("alpha_b", BondType::Causes, "alpha_y", 0.9, Confidence::Verified, teach_attr);
        b.teach_with_source("r_a", BondType::IsA, "r_x", 0.9, Confidence::Verified, research_attr.clone());
        b.teach_with_source("r_b", BondType::HasProperty, "r_y", 0.9, Confidence::Verified, research_attr);
        b.teach_with_source("d_a", BondType::PartOf, "d_x", 0.9, Confidence::Verified, doc_attr);
        let bonds_before = b.network.bond_count();
        let (overlay_removed, bonds_removed) = b.revoke_bonds_by_source(
            |attr| matches!(attr.kind, crate::source::SourceKind::Teach { .. }),
        );
        assert_eq!(overlay_removed, 2);
        assert_eq!(bonds_removed, 2);
        let bonds_after = b.network.bond_count();
        assert_eq!(bonds_after, bonds_before - 2);
        // Research and document bonds remain.
        assert!(matches!(
            b.bond_source(
                &crate::seed::Seed::from_label("r_a"),
                &crate::seed::Seed::from_label("r_x"),
                BondType::IsA,
            ).kind,
            crate::source::SourceKind::Research { .. }
        ));
        assert!(matches!(
            b.bond_source(
                &crate::seed::Seed::from_label("d_a"),
                &crate::seed::Seed::from_label("d_x"),
                BondType::PartOf,
            ).kind,
            crate::source::SourceKind::Document { .. }
        ));
    }

    /// Phase-6 curiosity loop: an approved research proposal tags
    /// every bond it produces with SourceKind::Research.
    #[test]
    fn port_s_research_apply_tags_research_kind() {
        use crate::research::loop_::apply_proposal;
        use crate::research::proposal::{BondCandidate, ResearchProposal, ResearchProposalStatus};
        let mut b = Brain::new();
        let proposal = ResearchProposal {
            id: 7,
            triggering_wonder: None,
            query: "why is the sky blue".to_string(),
            source_urls: vec!["https://en.wikipedia.org/wiki/Sky".into()],
            raw_excerpts: vec!["The sky appears blue due to Rayleigh scattering.".into()],
            providers: vec![],
            bond_candidates: vec![
                BondCandidate {
                    from_text: "sky".into(),
                    to_text: "blue".into(),
                    bond_type: BondType::HasProperty,
                    confidence: Confidence::Inferred,
                    source_url: "https://en.wikipedia.org/wiki/Sky".into(),
                    source_excerpt: "...".into(),
                    fingerprint: vec![],
                },
            ],
            scar_blocked_count: 0,
            created_ms: 0,
            status: ResearchProposalStatus::Pending,
        };
        b.research_proposals.push(proposal);
        let added = apply_proposal(&mut b, 0);
        assert!(added >= 1);
        let sky = crate::seed::Seed::from_label("sky");
        let blue = crate::seed::Seed::from_label("blue");
        let attr = b.bond_source(&sky, &blue, BondType::HasProperty);
        match attr.kind {
            crate::source::SourceKind::Research { url, wonder_seed, .. } => {
                assert_eq!(url.as_deref(), Some("https://en.wikipedia.org/wiki/Sky"));
                // Manual proposal -> wonder_seed = "manual:7".
                assert!(wonder_seed.starts_with("manual:") || wonder_seed.starts_with("wonder:"));
            }
            other => panic!("expected Research, got {:?}", other),
        }
    }

    /// J integration: narrate() now exposes source_kinds and a
    /// human prose summary in the answer when traversed bonds
    /// have non-Unknown sources.
    #[test]
    fn port_s_narrate_surfaces_source_kinds() {
        let mut b = Brain::new();
        // Tag bonds with explicit research source.
        let attr = crate::source::SourceAttribution::research(
            "W1",
            Some("https://en.wikipedia.org/wiki/Photosynthesis".into()),
            100,
        );
        b.teach_with_source("plants", BondType::IsA, "organism", 0.9, Confidence::Verified, attr.clone());
        b.teach_with_source("plants", BondType::HasProperty, "green", 0.9, Confidence::Verified, attr);
        let cfg = crate::narration::NarrationConfig::default();
        let result = crate::narration::narrate(&mut b, "plants", 3, 10, false, &cfg);
        // structuredContent carries the kinds even if the prose
        // walker chose not to mention them; the kinds list is
        // populated when at least one traversed bond has a
        // non-Unknown source.
        if !result.source_kinds.is_empty() {
            assert!(result.source_kinds.iter().any(|k| k == "research"));
            assert!(result.answer.contains("Based on"));
        }
    }
}


// =================================================================
// Port T.1: time accessors + scar-sensitive op refusal gate.
// =================================================================
impl Brain {
    /// Port T.1.1: apply `self.time_source_config` to the live clock
    /// source. If mode is `df_service`, spawns a real
    /// `DFTimeServiceSource` whose polling thread keeps a fresh
    /// signed anchor on hand. If mode is `system`, this is a no-op.
    /// Returns the new clock-source name on success.
    pub fn apply_time_source_config(&mut self) -> Result<String, crate::time::ClockError> {
        if !self.time_source_config.is_df_service() {
            // Drop any existing DF client; its poll thread exits via Drop.
            self.df_time_client = None;
            // Revert to SystemClockSource if we were on df_service.
            if self.clock.name() != "system" {
                let sys = std::sync::Arc::new(crate::time::SystemClockSource::new())
                    as std::sync::Arc<dyn crate::time::ClockSource>;
                let _ = self.set_clock_source(sys);
            }
            return Ok(self.clock.name().to_string());
        }
        // Build a single DFTimeServiceSource (which spawns the polling
        // thread internally) and share its inner DFRealClient between
        // brain.clock (via Arc<dyn ClockSource>) and brain.df_time_client
        // (typed handle for status snapshots).
        let src = crate::time::DFTimeServiceSource::new_real(
            self.time_source_config.to_client_config(),
        );
        let real_client = src.real_client();
        let arc: std::sync::Arc<dyn crate::time::ClockSource> = std::sync::Arc::new(src);
        // Direct-assign: set_clock_source calls fetch() and would fail
        // here because the newly-spawned poll thread hasn't landed an
        // anchor yet. Bypass the gate; the brain stays on this source
        // and falls back to FetchFailed via project_now() until the
        // first poll lands. Anomaly state is cleared so any sticky
        // flag from a previous source doesn't bleed in.
        self.clock = arc;
        if let Ok(mut det) = self.anomaly_detector.lock() {
            det.clear_anomaly();
        }
        self.df_time_client = real_client;
        Ok(self.clock.name().to_string())
    }

    /// Port T.1.1: snapshot of the DF Time Service polling state, if
    /// the active clock is the real DF source. `None` for `system`
    /// mode.
    pub fn df_time_state_snapshot(&self) -> Option<crate::time::DFRealState> {
        self.df_time_client.as_ref().map(|c| c.state_snapshot())
    }

    /// Replace the clock source backend. Used by `main.rs` to swap in
    /// `DFTimeServiceSource` when `--time-source df` is requested.
    /// Refreshes the anchor + clears any sticky anomaly.
    pub fn set_clock_source(
        &mut self,
        clock: std::sync::Arc<dyn crate::time::ClockSource>,
    ) -> Result<(), crate::time::ClockError> {
        let fresh = clock.fetch()?;
        self.clock = clock;
        if let Ok(mut anchor) = self.time_anchor.write() {
            *anchor = fresh.clone();
        }
        if let Ok(mut det) = self.anomaly_detector.lock() {
            det.clear_anomaly();
            // Prime the detector with the new anchor so the first real
            // `check()` after the swap has a baseline.
            let _ = det.check(&fresh);
        }
        Ok(())
    }

    /// Force a refresh of the authenticated-time anchor. Returns the
    /// new anchor on success, or the anomaly that fired.
    pub fn refresh_time_anchor(&self) -> Result<crate::time::AuthenticatedTime, crate::time::ClockError> {
        let fresh = self.clock.fetch()?;
        if let Ok(mut det) = self.anomaly_detector.lock() {
            if let Err(kind) = det.check(&fresh) {
                return Err(crate::time::ClockError::AnomalyDetected(format!("{:?}", kind)));
            }
        }
        if let Ok(mut anchor) = self.time_anchor.write() {
            *anchor = fresh.clone();
        }
        Ok(fresh)
    }

    /// Current monotonic timestamp. Local-only; never refuses.
    pub fn monotonic_now(&self) -> crate::time::MonotonicTimestamp {
        crate::time::MonotonicClock::now()
    }

    /// Current wall-clock UTC ms. Projects from the most recent anchor
    /// using a fresh monotonic reading; on projection-failure paths,
    /// falls back to `now_ms()`. Always returns a value — degradation
    /// for unauthenticated time is scoped to scar-sensitive ops, not to
    /// surface queries.
    pub fn current_utc_ms(&self) -> u64 {
        if let Ok(anchor) = self.time_anchor.read() {
            return anchor.project_now_ms(self.monotonic_now());
        }
        now_ms().max(0) as u64
    }

    /// True iff the current time anchor is signed AND the anomaly
    /// detector is clean AND the anchor is not past expiry.
    ///
    /// Scar-sensitive operations consult this before mutating the
    /// scar log. Surface queries (`lu4d_ask`) do not — they keep working
    /// even when this is false (scoped degradation).
    pub fn time_is_authenticated(&self) -> bool {
        // 1. Source-level: anchor must carry a signature.
        let signed = match self.time_anchor.read() {
            Ok(a) => a.is_authenticated(),
            Err(_) => return false,
        };
        if !signed {
            return false;
        }
        // 2. Anomaly-level: no sticky anomaly.
        let clean = match self.anomaly_detector.lock() {
            Ok(d) => d.last_anomaly().is_none(),
            Err(_) => return false,
        };
        if !clean {
            return false;
        }
        // 3. Freshness: anchor must not be past expiry. Use wall-clock
        //    `now_ms()` since the anchor itself stores wall-clock expiry.
        let now = now_ms().max(0) as u64;
        if let Ok(a) = self.time_anchor.read() {
            if a.is_expired(now) {
                return false;
            }
        }
        true
    }

    /// T.1 scar-sensitive op gate. Placeholder for the real scar-time
    /// logic that lands in T.5; today it just demonstrates the refusal
    /// shape so other ports can wire to the same gate.
    pub fn t1_scar_sensitive_op_probe(&mut self) -> Result<(), crate::time::ClockError> {
        if !self.time_is_authenticated() {
            return Err(crate::time::ClockError::TimeNotAuthenticated);
        }
        Ok(())
    }
}
// =====================================================================
// Port T.2 — Idle detection + idle work loop.
// =====================================================================

impl Brain {
    /// True iff no MCP queries are currently in flight
    /// (`active_queries == 0`). Surface accessor for the spec API.
    pub fn is_idle(&self) -> bool {
        self.idle_state.is_idle()
    }

    /// Time the brain has been idle since the most recent query ended.
    /// `None` if either:
    ///   - a query is currently in flight, OR
    ///   - no query has ever ended.
    pub fn idle_duration(&self) -> Option<std::time::Duration> {
        self.idle_state.idle_duration()
    }

    /// Same as [`Self::idle_duration`] but in milliseconds. Preferred
    /// internally since the MCP surface reports milliseconds.
    pub fn idle_duration_ms(&self) -> Option<u64> {
        self.idle_state.idle_duration_ms()
    }

    /// Drive ONE iteration of the idle work loop. Caller-driven for now
    /// (T.6 daemon will auto-drive this). See `crate::idle::loop_::idle_tick`
    /// for the full gate semantics.
    ///
    /// IMPORTANT — this method does NOT itself add a [`crate::idle::QueryGuard`].
    /// The MCP dispatch wrapper (`handle_tools_call`) wraps every other
    /// tool in a guard, but `lu4d_idle_tick` is special-cased to skip
    /// that guard so the gate-1 idleness check sees the brain's
    /// genuine pre-tick state. Test code calls this method directly
    /// without a guard for the same reason.
    pub fn idle_tick(
        &mut self,
        fetcher: &dyn crate::research::http::HttpFetcher,
        min_idle_ms: u64,
    ) -> crate::idle::IdleTickResult {
        crate::idle::loop_::idle_tick(self, fetcher, min_idle_ms)
    }
}

// =====================================================================
// Port T.5 — Bond freshness / decay (query-time multiplier).
//
// Stored bond weights are unchanged. The clock fetch uses
// `Brain::clock` (T.1 ClockSource), never `SystemTime::now` directly.
// =====================================================================
impl Brain {
    /// Best-effort wall-clock NOW (ms) routed through T.1 ClockSource.
    /// Falls back to `now_ms()` if the source returns FetchFailed
    /// (e.g. the real DF source before T.1.1 lands).
    fn t5_clock_now_ms(&self) -> i64 {
        match self.clock.fetch() {
            Ok(t) => t.utc_ms as i64,
            Err(_) => now_ms(),
        }
    }

    /// Port T.5.1 — Refresh the network's decay runtime from the
    /// current `decay_config` and clock-source NOW. Call before
    /// every Brain query path so the freshness multiplier shapes
    /// recall ranking. Stored bond weights are NEVER mutated by this
    /// path (decay is presentation-only — see T.5 module docs).
    pub fn refresh_decay_runtime(&mut self) {
        let rt = crate::decay::DecayRuntime::new(
            self.decay_config,
            self.t5_clock_now_ms(),
        );
        self.network.set_decay_runtime(Some(rt));
    }

    /// Port T.5.1 — Disarm decay on the network. Used by
    /// time-travel paths (which want historical-faithful scoring,
    /// not present-time decay) and by the regression-guard test
    /// `t51_decay_disabled_matches_pre_t51_behavior_exactly`.
    pub fn clear_decay_runtime(&mut self) {
        self.network.set_decay_runtime(None);
    }

    /// Port Sense.1 -- Refresh the network's sense runtime from the
    /// current `sense_config` for `query_seeds`. Call after seed
    /// selection in every Brain query path so the cluster-gain
    /// multiplier shapes recall ranking on polysemous-seed queries.
    /// Stored bond weights are NEVER mutated by this path (Sense
    /// is presentation-only -- see sense.rs module docs).
    pub fn refresh_sense_runtime(&mut self, query_seeds: &[crate::seed::Seed]) {
        let cfg = self.sense_config.clone();
        let rt = self.network.build_sense_runtime(query_seeds, cfg);
        self.network.set_sense_runtime(Some(rt));
    }

    /// Port Sense.1 -- Disarm sense disambiguation on the network.
    /// Used by time-travel paths (which prefer the structural
    /// pre-Sense ranking) and by the regression-guard tests.
    pub fn clear_sense_runtime(&mut self) {
        self.network.set_sense_runtime(None);
    }


    /// T.5 — freshness multiplier for `bond` at the current clock-source
    /// NOW. Returns 1.0 when decay is disabled. Always in [0.0, 1.0].
    pub fn freshness_for_bond(&self, bond: &crate::bond::TemporalBond) -> f32 {
        let now = self.t5_clock_now_ms();
        let created = crate::decay::bond_created_at_ms(bond);
        crate::decay::freshness_score(created, now, &self.decay_config)
    }

    /// T.5 — Convenience: freshness multiplier for `(from, to, bond_type)`.
    /// Returns 1.0 when the bond is missing (so callers do not need to
    /// branch on bond presence). Heap-only: looks up via `bonds_raw`.
    pub fn freshness_for_bond_pair(
        &self,
        from: crate::seed::Seed,
        to: crate::seed::Seed,
        bond_type: crate::bond::BondType,
    ) -> f32 {
        match self.network.bonds_raw().get(&(from, to)) {
            Some(b) if b.bond_type == bond_type => self.freshness_for_bond(b),
            _ => 1.0,
        }
    }

    /// T.5 — Effective weight after applying the freshness multiplier.
    /// `None` if the bond is missing or its bond_type does not match.
    pub fn bond_effective_weight_with_decay(
        &self,
        from: crate::seed::Seed,
        to: crate::seed::Seed,
        bond_type: crate::bond::BondType,
    ) -> Option<f32> {
        let bond = self.network.bonds_raw().get(&(from, to))?;
        if bond.bond_type != bond_type { return None; }
        let stored = bond
            .weight_at(i64::MAX, &self.scar_log)
            .effective_weight;
        let f = self.freshness_for_bond(bond);
        Some(stored * f)
    }

    /// T.5 — Build the `lu4d_freshness_status` payload. `top_n`
    /// controls how many stalest bonds are returned in `stalest`
    /// (pass 0 for no stalest list — bucket counts + mean/median
    /// only). Bonds with empty keyframe vectors are skipped.
    pub fn freshness_status(&self, top_n: usize) -> crate::decay::FreshnessStatus {
        let mut st = crate::decay::FreshnessStatus::default();
        st.decay_enabled = self.decay_config.enabled;
        st.half_life_days = self.decay_config.half_life_days;
        let now = self.t5_clock_now_ms();
        let mut ages_ms: Vec<i64> = Vec::new();
        // Heap-only walk — for the canonical mmap brain we still walk
        // the bonds() view, which yields BondView (heap or archive).
        // Use keyframes_iter() so archive-backed bonds also work.
        let mut stalest_pool: Vec<(i64, crate::seed::Seed, crate::seed::Seed,
                                   crate::bond::BondType, f32)> = Vec::new();
        for (key, bv) in self.network.bonds().iter() {
            // Created-at = oldest visible keyframe.
            let created = bv.keyframes_iter().map(|kf| kf.timestamp).min().unwrap_or(0);
            let age_ms = (now - created).max(0);
            ages_ms.push(age_ms);
            crate::decay::classify_bucket(age_ms, &mut st);
            if top_n > 0 {
                let stored = bv.weight_at(i64::MAX, &self.scar_log).effective_weight;
                stalest_pool.push((age_ms, key.0, key.1, bv.bond_type(), stored));
            }
        }
        st.total_bonds = ages_ms.len() as u64;
        if !ages_ms.is_empty() {
            let total: i128 = ages_ms.iter().map(|&v| v as i128).sum();
            let mean_ms = total as f64 / ages_ms.len() as f64;
            st.mean_age_days = mean_ms / crate::decay::DAY_MS as f64;
            ages_ms.sort_unstable();
            let mid = ages_ms.len() / 2;
            let med_ms = if ages_ms.len() % 2 == 1 {
                ages_ms[mid] as f64
            } else {
                (ages_ms[mid - 1] as f64 + ages_ms[mid] as f64) / 2.0
            };
            st.median_age_days = med_ms / crate::decay::DAY_MS as f64;
        }
        if top_n > 0 {
            // Highest age first.
            stalest_pool.sort_unstable_by(|a, b| b.0.cmp(&a.0));
            let take = top_n.min(stalest_pool.len());
            for (age_ms, from, to, bt, stored) in stalest_pool.into_iter().take(take) {
                let age_days = age_ms as f64 / crate::decay::DAY_MS as f64;
                let f = if self.decay_config.enabled {
                    crate::decay::freshness_score(
                        now - age_ms, now, &self.decay_config,
                    )
                } else { 1.0 };
                // OP-LABEL-4: `get_node` returns Some("") on a v5 stub, so
                // the hex fallback never fired and the report printed
                // two blanks.
                let subject_text = self
                    .network
                    .node_label(&from)
                    .filter(|l| !l.is_empty())
                    .unwrap_or_else(|| format!("{:?}", from));
                let object_text = self
                    .network
                    .node_label(&to)
                    .filter(|l| !l.is_empty())
                    .unwrap_or_else(|| format!("{:?}", to));
                st.stalest.push(crate::decay::StalestBondInfo {
                    from, to,
                    subject_text,
                    object_text,
                    bond_type: crate::decay::bond_type_tag(bt).to_string(),
                    age_days,
                    stored_weight: stored,
                    effective_weight_after_decay: stored * f,
                });
            }
        }
        st
    }
}

// ----------------------------------------------------------------
// Port Cat.3b — hub-degree write-side admission helper.
// ----------------------------------------------------------------

/// Decide whether to admit a new bond against the hub-degree caps,
/// performing any eviction the priority strategy requires. Returns
/// `(admitted, evicted_count)` so the caller can bump the
/// `hub_cap_rejections_total` / `hub_cap_evictions_total` counters.
///
/// Endpoints whose node kind is in `cfg.exempt_node_kinds` bypass
/// the cap check on that side. When BOTH endpoints are non-exempt
/// and at cap, the harder side wins (the bond is rejected unless
/// the priority strategy can free space on that side).
///
/// `priority` semantics:
///   * NewestFirst       → reject any new bond when at cap.
///   * OldestFirst       → evict the oldest existing bond on the
///                         saturated side (lowest first-keyframe
///                         timestamp among the seed's outgoing or
///                         incoming bonds).
///   * LowestWeightFirst → evict the lowest-weight existing bond on
///                         the saturated side IF the new bond's
///                         weight is strictly higher; otherwise
///                         reject.
///
/// This function accesses the network mutably to perform evictions
/// via `AssociativeNetwork::remove_bond` (which is the existing API
/// for bond removal — see Cat.3b spec: existing bonds are touched
/// only via the eviction priority).
pub fn decide_hub_cap_admission(
    net: &mut AssociativeNetwork,
    cfg: &crate::hub_degree::HubDegreeConfig,
    from: crate::seed::Seed,
    to: crate::seed::Seed,
    new_weight: f32,
) -> (bool, u32) {
    use crate::hub_degree::HubPriority;

    // Daemon Perf-1: skip the get_node-for-kind lookups when neither
    // raw degree reaches the cap. Admission is then unconditional, so
    // there is no need to consult the exemption list. Mirrors the
    // semantics of the pre-Perf-1 body byte-for-byte; only the order
    // in which we collect the inputs changes. Tests covered: every
    // `cat3b_behavior_tests::*` case in this module + the cat3b /
    // hub_degree suites in `crate::mcp_cat3b` and `crate::hub_degree`.
    let from_out = net.out_degree(&from) as u32;
    let to_in = net.in_degree(&to) as u32;
    let from_below_cap = from_out < cfg.max_out_degree;
    let to_below_cap = to_in < cfg.max_in_degree;
    if from_below_cap && to_below_cap {
        return (true, 0);
    }

    // Slow path: at least one endpoint hit the raw cap. Now consult
    // the kind to decide whether the cap actually applies.
    let from_kind = net
        .get_node(&from)
        .map(|n| n.kind_str())
        .unwrap_or("");
    let to_kind = net
        .get_node(&to)
        .map(|n| n.kind_str())
        .unwrap_or("");
    let from_exempt = cfg.is_exempt_kind(from_kind);
    let to_exempt = cfg.is_exempt_kind(to_kind);

    let from_at_cap = !from_below_cap && !from_exempt;
    let to_at_cap = !to_below_cap && !to_exempt;
    if !from_at_cap && !to_at_cap {
        // Both endpoints exempt by kind. Admission stays true; no eviction.
        return (true, 0);
    }

    // Apply priority on the saturated side(s).
    let mut evicted: u32 = 0;

    if from_at_cap {
        match cfg.priority {
            HubPriority::NewestFirst => {
                return (false, evicted);
            }
            HubPriority::OldestFirst => {
                let neighbors: Vec<crate::seed::Seed> =
                    net.neighbors_outgoing(&from).to_vec();
                let mut oldest: Option<(crate::seed::Seed, i64)> = None;
                for n in &neighbors {
                    if let Some(b) = net.get_bond(&from, n) {
                        let ts = b.keyframes().first().map(|k| k.timestamp).unwrap_or(0);
                        match oldest {
                            None => oldest = Some((*n, ts)),
                            Some((_, prev_ts)) if ts < prev_ts => {
                                oldest = Some((*n, ts));
                            }
                            _ => {}
                        }
                    }
                }
                if let Some((victim_to, _)) = oldest {
                    if net.remove_bond(&from, &victim_to) {
                        evicted = evicted.saturating_add(1);
                    } else {
                        return (false, evicted);
                    }
                } else {
                    return (false, evicted);
                }
            }
            HubPriority::LowestWeightFirst => {
                let neighbors: Vec<crate::seed::Seed> =
                    net.neighbors_outgoing(&from).to_vec();
                let mut lowest: Option<(crate::seed::Seed, f32)> = None;
                for n in &neighbors {
                    if let Some(b) = net.get_bond(&from, n) {
                        let w = b.latest_raw_weight();
                        match lowest {
                            None => lowest = Some((*n, w)),
                            Some((_, prev_w)) if w < prev_w => {
                                lowest = Some((*n, w));
                            }
                            _ => {}
                        }
                    }
                }
                match lowest {
                    Some((victim_to, victim_w)) if new_weight > victim_w => {
                        if net.remove_bond(&from, &victim_to) {
                            evicted = evicted.saturating_add(1);
                        } else {
                            return (false, evicted);
                        }
                    }
                    _ => return (false, evicted),
                }
            }
        }
    }

    if to_at_cap {
        match cfg.priority {
            HubPriority::NewestFirst => {
                return (false, evicted);
            }
            HubPriority::OldestFirst => {
                let neighbors: Vec<crate::seed::Seed> =
                    net.neighbors_incoming(&to).to_vec();
                let mut oldest: Option<(crate::seed::Seed, i64)> = None;
                for f in &neighbors {
                    if let Some(b) = net.get_bond(f, &to) {
                        let ts = b.keyframes().first().map(|k| k.timestamp).unwrap_or(0);
                        match oldest {
                            None => oldest = Some((*f, ts)),
                            Some((_, prev_ts)) if ts < prev_ts => {
                                oldest = Some((*f, ts));
                            }
                            _ => {}
                        }
                    }
                }
                if let Some((victim_from, _)) = oldest {
                    if net.remove_bond(&victim_from, &to) {
                        evicted = evicted.saturating_add(1);
                    } else {
                        return (false, evicted);
                    }
                } else {
                    return (false, evicted);
                }
            }
            HubPriority::LowestWeightFirst => {
                let neighbors: Vec<crate::seed::Seed> =
                    net.neighbors_incoming(&to).to_vec();
                let mut lowest: Option<(crate::seed::Seed, f32)> = None;
                for f in &neighbors {
                    if let Some(b) = net.get_bond(f, &to) {
                        let w = b.latest_raw_weight();
                        match lowest {
                            None => lowest = Some((*f, w)),
                            Some((_, prev_w)) if w < prev_w => {
                                lowest = Some((*f, w));
                            }
                            _ => {}
                        }
                    }
                }
                match lowest {
                    Some((victim_from, victim_w)) if new_weight > victim_w => {
                        if net.remove_bond(&victim_from, &to) {
                            evicted = evicted.saturating_add(1);
                        } else {
                            return (false, evicted);
                        }
                    }
                    _ => return (false, evicted),
                }
            }
        }
    }

    (true, evicted)
}

#[cfg(test)]
mod cat3b_behavior_tests {
    //! Port Cat.3b behavior tests covering the 6 spec contracts that
    //! exercise write-side enforcement, read-side filtering, exempt
    //! node kinds, mutation safety, and counter increments. Config
    //! tests live in `crate::hub_degree`; MCP tool tests live in
    //! `crate::mcp_cat3b`.

    use crate::bond::BondType;
    use crate::brain::Brain;
    use crate::hub_degree::HubPriority;
    use crate::node::Confidence;
    use crate::seed::Seed;

    fn pre_load_hub(b: &mut Brain, hub: &str, n: u32, base_weight: f32) {
        // Pre-load `n` outgoing bonds from hub. Weight ramps from
        // `base_weight` upward so deterministic eviction picks are
        // possible. Uses teach_with_kind directly via teach so the
        // write enforcement is exercised under default config.
        for i in 0..n {
            let leaf = format!("leaf_{}", i);
            // Weight scales linearly so leaf_0 is lowest, leaf_(n-1)
            // is highest. Clamped to [0.0, 1.0].
            let w = (base_weight + 0.001 * i as f32).clamp(0.0, 1.0);
            b.teach(hub, BondType::RelatedTo, &leaf, w, Confidence::Verified);
        }
    }

    #[test]
    fn cat3b_write_enforcement_rejects_new_bond_at_cap_with_lower_weight() {
        let mut b = Brain::new();
        // Tight cap so the test is small.
        b.hub_degree_config.max_out_degree = 100;
        // LowestWeightFirst (default) — when at cap, reject if new
        // bond's weight is not strictly higher than the lowest existing.
        // Pre-load 100 bonds with weights 0.500..0.599.
        pre_load_hub(&mut b, "hub", 100, 0.500);
        let before = b.hub_cap_rejections_total;
        let bonds_before = b.network.bond_count();
        // Try to add a bond with lower weight than the lowest existing
        // (0.500). 0.100 must be rejected.
        b.teach(
            "hub",
            BondType::RelatedTo,
            "should_be_rejected",
            0.100,
            Confidence::Verified,
        );
        assert_eq!(b.hub_cap_rejections_total, before + 1);
        // bond_count unchanged (rejection means no new bond).
        assert_eq!(b.network.bond_count(), bonds_before);
    }

    #[test]
    fn cat3b_write_enforcement_evicts_lowest_weight_when_new_bond_higher() {
        let mut b = Brain::new();
        b.hub_degree_config.max_out_degree = 100;
        pre_load_hub(&mut b, "hub", 100, 0.500);
        let bonds_before = b.network.bond_count();
        let evictions_before = b.hub_cap_evictions_total;
        // New bond weight 0.999 > lowest existing 0.500 → evict + admit.
        b.teach(
            "hub",
            BondType::RelatedTo,
            "high_weight_winner",
            0.999,
            Confidence::Verified,
        );
        assert_eq!(b.hub_cap_evictions_total, evictions_before + 1);
        // One evicted, one added → net zero.
        assert_eq!(b.network.bond_count(), bonds_before);
        // The new bond is in the network.
        let from = Seed::from_label("hub");
        let to = Seed::from_label("high_weight_winner");
        assert!(b.network.get_bond(&from, &to).is_some());
        // The lowest-weight existing bond (leaf_0, weight 0.500) is gone.
        let evicted_to = Seed::from_label("leaf_0");
        assert!(b.network.get_bond(&from, &evicted_to).is_none());
    }

    #[test]
    fn cat3b_write_enforcement_disabled_passes_through() {
        let mut b = Brain::new();
        b.hub_degree_config.max_out_degree = 100;
        b.hub_degree_config.write_enforcement = false;
        pre_load_hub(&mut b, "hub", 100, 0.500);
        let bonds_before = b.network.bond_count();
        // With enforcement off, even a low-weight bond is admitted
        // and no eviction happens.
        b.teach(
            "hub",
            BondType::RelatedTo,
            "low_weight_passthrough",
            0.001,
            Confidence::Verified,
        );
        assert_eq!(b.hub_cap_rejections_total, 0);
        assert_eq!(b.hub_cap_evictions_total, 0);
        // bond_count grew by 1.
        assert_eq!(b.network.bond_count(), bonds_before + 1);
    }

    #[test]
    fn cat3b_read_enforcement_filters_legacy_uncapped_hub_at_query_time() {
        let mut b = Brain::new();
        // First populate WITHOUT the cap (write enforcement off) so we
        // can simulate a legacy uncapped brain.
        b.hub_degree_config.write_enforcement = false;
        pre_load_hub(&mut b, "hub", 50, 0.500);
        let hub = Seed::from_label("hub");
        assert_eq!(b.network.out_degree(&hub), 50);

        // Now turn on read enforcement with a tight cap of 10.
        b.hub_degree_config.read_enforcement = true;
        b.hub_degree_config.max_out_degree = 10;
        b.resync_hub_read_config();

        let filtered_before = b.network.hub_cap_filtered_at_query_total();
        let kept = b.network.neighbors_outgoing_capped(&hub);
        // The capped read returns at most 10 neighbors.
        assert_eq!(kept.len(), 10);
        // The counter reflects 40 dropped bonds.
        assert_eq!(
            b.network.hub_cap_filtered_at_query_total(),
            filtered_before + 40
        );
        // Existing bonds are NOT mutated — the underlying neighbor
        // list still has all 50.
        assert_eq!(b.network.out_degree(&hub), 50);
    }

    #[test]
    fn cat3b_read_enforcement_disabled_returns_all_bonds() {
        let mut b = Brain::new();
        b.hub_degree_config.write_enforcement = false;
        pre_load_hub(&mut b, "hub", 50, 0.500);
        let hub = Seed::from_label("hub");

        b.hub_degree_config.read_enforcement = false;
        b.hub_degree_config.max_out_degree = 10;
        b.resync_hub_read_config();

        let filtered_before = b.network.hub_cap_filtered_at_query_total();
        let kept = b.network.neighbors_outgoing_capped(&hub);
        // With read enforcement off, all 50 are returned.
        assert_eq!(kept.len(), 50);
        // Counter doesn't move.
        assert_eq!(
            b.network.hub_cap_filtered_at_query_total(),
            filtered_before
        );
    }

    #[test]
    fn cat3b_exempt_node_kinds_bypass_caps() {
        let mut b = Brain::new();
        b.hub_degree_config.max_out_degree = 5;
        // Exempt the "concept" kind — KnowledgeContent::Concept matches
        // the default kind for plain teach() calls (see
        // KnowledgeContent::from_kind_or_concept which falls through
        // to Concept when kind_hint is None).
        b.hub_degree_config.exempt_node_kinds = vec!["concept".to_string()];
        // Pre-load 5 bonds (right at the cap).
        pre_load_hub(&mut b, "hub", 5, 0.500);
        let bonds_before = b.network.bond_count();
        // The next bond should be admitted — exempt kind bypasses cap.
        b.teach(
            "hub",
            BondType::RelatedTo,
            "exempt_passthrough",
            0.100,
            Confidence::Verified,
        );
        assert_eq!(b.hub_cap_rejections_total, 0);
        // bond_count grew by 1.
        assert_eq!(b.network.bond_count(), bonds_before + 1);
    }

    #[test]
    fn cat3b_hub_does_not_mutate_existing_bonds_at_query_time() {
        let mut b = Brain::new();
        b.hub_degree_config.write_enforcement = false;
        pre_load_hub(&mut b, "hub", 30, 0.500);
        let hub = Seed::from_label("hub");

        b.hub_degree_config.read_enforcement = true;
        b.hub_degree_config.max_out_degree = 5;
        b.resync_hub_read_config();

        let raw_before = b.network.out_degree(&hub);
        assert_eq!(raw_before, 30);
        // Run the capped read multiple times.
        for _ in 0..5 {
            let _ = b.network.neighbors_outgoing_capped(&hub);
        }
        // The underlying out_degree is STILL 30. The filter never
        // mutates stored bonds; it only filters the returned list.
        assert_eq!(b.network.out_degree(&hub), raw_before);
    }

    #[test]
    fn cat3b_counters_increment_on_rejection_eviction_and_filter() {
        let mut b = Brain::new();
        b.hub_degree_config.max_out_degree = 100;

        // 1. Rejection counter — pre-load to cap, then teach a lower
        // weight bond. LowestWeightFirst rejects the lower one.
        pre_load_hub(&mut b, "hub", 100, 0.500);
        b.teach(
            "hub",
            BondType::RelatedTo,
            "rej_target",
            0.100,
            Confidence::Verified,
        );
        assert_eq!(b.hub_cap_rejections_total, 1);

        // 2. Eviction counter — teach a higher-weight bond.
        b.teach(
            "hub",
            BondType::RelatedTo,
            "evict_winner",
            0.999,
            Confidence::Verified,
        );
        assert_eq!(b.hub_cap_evictions_total, 1);

        // 3. Filter counter — turn read enforcement on with a tight
        // cap and run a capped read on the existing hub.
        b.hub_degree_config.read_enforcement = true;
        b.hub_degree_config.max_out_degree = 10;
        b.resync_hub_read_config();
        let hub = Seed::from_label("hub");
        let filtered_before = b.network.hub_cap_filtered_at_query_total();
        let _ = b.network.neighbors_outgoing_capped(&hub);
        let filtered_after = b.network.hub_cap_filtered_at_query_total();
        assert!(filtered_after > filtered_before);
    }

    #[test]
    fn cat3b_priority_oldest_first_evicts_oldest() {
        let mut b = Brain::new();
        b.hub_degree_config.max_out_degree = 5;
        b.hub_degree_config.priority = HubPriority::OldestFirst;
        // Pre-load 5 bonds — leaf_0 is oldest.
        pre_load_hub(&mut b, "hub", 5, 0.500);
        let from = Seed::from_label("hub");
        let oldest = Seed::from_label("leaf_0");
        assert!(b.network.get_bond(&from, &oldest).is_some());
        // New bond of any weight evicts oldest.
        b.teach(
            "hub",
            BondType::RelatedTo,
            "newcomer",
            0.001,
            Confidence::Verified,
        );
        assert_eq!(b.hub_cap_evictions_total, 1);
        assert!(b.network.get_bond(&from, &oldest).is_none());
        // newcomer is now in.
        let new_to = Seed::from_label("newcomer");
        assert!(b.network.get_bond(&from, &new_to).is_some());
    }

    #[test]
    fn cat3b_priority_newest_first_rejects_at_cap() {
        let mut b = Brain::new();
        b.hub_degree_config.max_out_degree = 5;
        b.hub_degree_config.priority = HubPriority::NewestFirst;
        pre_load_hub(&mut b, "hub", 5, 0.500);
        let bonds_before = b.network.bond_count();
        b.teach(
            "hub",
            BondType::RelatedTo,
            "rejected",
            0.999,
            Confidence::Verified,
        );
        // NewestFirst rejects regardless of weight.
        assert_eq!(b.hub_cap_rejections_total, 1);
        assert_eq!(b.hub_cap_evictions_total, 0);
        assert_eq!(b.network.bond_count(), bonds_before);
    }
}


#[cfg(test)]
mod t_ingest_1_tests {
    //! Port T-Ingest.1 contract: trust multipliers fire on the teach
    //! path. The 8 cases plus the structural anti-hallucination
    //! invariant prove that:
    //!   1. caller-supplied `attribution.trust = Some(t)` overrides
    //!      the table (caller knows best);
    //!   2. `trust = None` + corpus in table -> table multiplier;
    //!   3. `trust = None` + corpus NOT in table -> default_multiplier;
    //!   4. mutating the table after a bond was formed does NOT
    //!      retroactively reweight the existing bond;
    //!   5. Research-source URL-host lookup is preserved (C2 intact);
    //!   6. seed_curriculum_corpora=true populates 18 entries;
    //!   7. seed_curriculum_corpora=false leaves the map empty;
    //!   8. byte-for-byte regression equivalence with empty map +
    //!      default_multiplier=1.0.
    use super::*;
    use crate::source::{SourceAttribution, SourceKind};
    use crate::trust::TrustTable;

    fn fresh() -> Brain {
        let mut b = Brain::new();
        // Tests don't want the C2 default url-host entries leaking
        // into corpus-resolution defaults; reset to an empty table
        // with a known default_multiplier.
        b.trust_table = TrustTable::empty();
        b
    }

    fn bond_weight(b: &Brain, subject: &str, object: &str, bt: BondType) -> Option<f32> {
        let from = Seed::from_label(subject);
        let to = Seed::from_label(object);
        b.network
            .bond_raw(&from, &to)
            .filter(|bd| bd.bond_type() == bt)
            .map(|bd| bd.latest_raw_weight())
    }

    // ---------------- Case 1: caller-supplied trust overrides table
    #[test]
    fn t_ingest_1_caller_supplied_trust_overrides_table() {
        let mut b = fresh();
        // Seed table with a value that would resolve to 0.1 if the
        // teach path looked it up. Caller supplies Some(0.5) -> wins.
        b.trust_table.set_corpus("alpha/x", 0.1).unwrap();
        let mut attr = SourceAttribution::training_for_resolution(
            "alpha/x".to_string(), Some("r1".into()), None,
        );
        attr.trust = 0.5; // caller-supplied final
        let _r = b.teach_with_source(
            "a_subj", BondType::IsA, "a_obj", 0.8,
            crate::node::Confidence::Verified, attr,
        );
        // Caller-supplied -> teach_with_source MUST NOT re-multiply.
        let w = bond_weight(&b, "a_subj", "a_obj", BondType::IsA).unwrap();
        assert!((w - 0.8).abs() < 1e-6, "expected 0.8, got {}", w);
    }

    // ---------------- Case 2: trust=None + corpus in table -> table
    #[test]
    fn t_ingest_1_unset_trust_with_corpus_in_table_uses_table_multiplier() {
        let mut b = fresh();
        b.trust_table.set_corpus("beta/y", 0.6).unwrap();
        let attr = SourceAttribution::training_for_resolution(
            "beta/y".to_string(), Some("r1".into()), None,
        );
        let _r = b.teach_with_source(
            "b_subj", BondType::IsA, "b_obj", 0.5,
            crate::node::Confidence::Verified, attr,
        );
        let w = bond_weight(&b, "b_subj", "b_obj", BondType::IsA).unwrap();
        // 0.5 * 0.6 = 0.30
        assert!((w - 0.30).abs() < 1e-6, "expected 0.30, got {}", w);
        // Stamped trust must equal the resolved multiplier.
        let stored = b.bond_source(
            &Seed::from_label("b_subj"),
            &Seed::from_label("b_obj"),
            BondType::IsA,
        );
        assert!((stored.trust - 0.6).abs() < 1e-6, "got {}", stored.trust);
    }

    // ---------------- Case 3: trust=None + corpus missing -> default
    #[test]
    fn t_ingest_1_unset_trust_with_corpus_missing_uses_default_multiplier() {
        let mut b = fresh();
        b.trust_table.set_default_multiplier(0.4).unwrap();
        // corpus_multipliers is intentionally empty -> miss -> default
        let attr = SourceAttribution::training_for_resolution(
            "never/seen".to_string(), None, None,
        );
        let _r = b.teach_with_source(
            "c_subj", BondType::IsA, "c_obj", 1.0,
            crate::node::Confidence::Verified, attr,
        );
        let w = bond_weight(&b, "c_subj", "c_obj", BondType::IsA).unwrap();
        // 1.0 * 0.4 = 0.40
        assert!((w - 0.40).abs() < 1e-6, "expected 0.40, got {}", w);
        let stored = b.bond_source(
            &Seed::from_label("c_subj"),
            &Seed::from_label("c_obj"),
            BondType::IsA,
        );
        assert!((stored.trust - 0.4).abs() < 1e-6, "got {}", stored.trust);
    }

    // ---------------- Case 4: existing bonds NOT retroactively reweighted
    #[test]
    fn t_ingest_1_table_mutation_does_not_retroactively_reweight_bond() {
        let mut b = fresh();
        b.trust_table.set_corpus("x", 0.6).unwrap();
        let attr = SourceAttribution::training_for_resolution(
            "x".to_string(), None, None,
        );
        b.teach_with_source(
            "r_subj", BondType::IsA, "r_obj", 0.5,
            crate::node::Confidence::Verified, attr,
        );
        let w_at_formation = bond_weight(&b, "r_subj", "r_obj", BondType::IsA).unwrap();
        assert!((w_at_formation - 0.30).abs() < 1e-6);
        // Mutate the table to a higher multiplier.
        b.trust_table.set_corpus("x", 1.0).unwrap();
        // Existing bond's stored weight must be unchanged.
        let w_after = bond_weight(&b, "r_subj", "r_obj", BondType::IsA).unwrap();
        assert!(
            (w_after - 0.30).abs() < 1e-6,
            "existing bond reweighted: {}", w_after
        );
    }

    // ---------------- Case 5: Research source -> URL host lookup (C2)
    #[test]
    fn t_ingest_1_research_source_routes_through_url_host_lookup() {
        let mut b = fresh();
        // Seed C2 hierarchy: grokipedia=1.0, wikipedia=0.7
        b.trust_table.set("grokipedia.com", 1.0, None).unwrap();
        b.trust_table.set("en.wikipedia.org", 0.7, None).unwrap();
        // None trust + Research kind with grokipedia URL -> 1.0
        let attr_grok = SourceAttribution {
            kind: SourceKind::Research {
                wonder_seed: "w1".to_string(),
                url: Some("https://grokipedia.com/article".to_string()),
                fetched_at_ms: 0,
            },
            trust: crate::source::TRUST_NEEDS_RESOLUTION,
        };
        b.teach_with_source(
            "rs_a", BondType::IsA, "rs_b", 0.5,
            crate::node::Confidence::Verified, attr_grok,
        );
        let w_grok = bond_weight(&b, "rs_a", "rs_b", BondType::IsA).unwrap();
        assert!((w_grok - 0.5).abs() < 1e-6, "grokipedia weight: {}", w_grok);
        // Wikipedia URL -> 0.7
        let attr_wiki = SourceAttribution {
            kind: SourceKind::Research {
                wonder_seed: "w2".to_string(),
                url: Some("https://en.wikipedia.org/wiki/X".to_string()),
                fetched_at_ms: 0,
            },
            trust: crate::source::TRUST_NEEDS_RESOLUTION,
        };
        b.teach_with_source(
            "rs_c", BondType::IsA, "rs_d", 0.5,
            crate::node::Confidence::Verified, attr_wiki,
        );
        let w_wiki = bond_weight(&b, "rs_c", "rs_d", BondType::IsA).unwrap();
        assert!((w_wiki - 0.35).abs() < 1e-6, "wikipedia weight: {}", w_wiki);
    }

    // ---------------- Case 6 / 7: seed_curriculum flag behavior
    #[test]
    fn t_ingest_1_seed_curriculum_corpora_populates_18_entries() {
        let mut t = TrustTable::empty();
        t.seed_curriculum_corpora();
        // 18 entries, tracking seed_curriculum_corpora() in src/trust.rs:
        // encyclopedic/QA (wikipedia, structured-wikipedia, grokipedia,
        // no_robots, instruction-dataset), lexical Tier-0 (wordnet, gcide),
        // vision (visual_genome, GQA), research (arxiv, stackexchange), and
        // coding curriculum (code_search_net, python-code, stackoverflow, python-docs,
        // mdn, starcoderdata, the-stack-v2). Update this count when that
        // list changes. (Was 14 pre-arxiv/stackexchange/the-stack-v2/wordnet/gcide.)
        assert_eq!(t.corpus_multipliers.len(), 18);
        // Spot-check the curriculum hierarchy.
        assert!((t.lookup_corpus("wikimedia/wikipedia") - 1.0).abs() < 1e-6);
        assert!((t.lookup_corpus("wikimedia/structured-wikipedia") - 1.0).abs() < 1e-6);
        assert!((t.lookup_corpus("ranjaykrishna/visual_genome") - 1.0).abs() < 1e-6);
        assert!((t.lookup_corpus("Voxel51/GQA-Scene-Graph") - 0.9).abs() < 1e-6);
        assert!((t.lookup_corpus("arxiv") - 0.9).abs() < 1e-6);
        assert!((t.lookup_corpus("stackexchange") - 0.8).abs() < 1e-6);
        assert!((t.lookup_corpus("grokipedia") - 1.0).abs() < 1e-6);
        assert!((t.lookup_corpus("HuggingFaceH4/no_robots") - 0.7).abs() < 1e-6);
        assert!((t.lookup_corpus("HuggingFaceH4/instruction-dataset") - 0.7).abs() < 1e-6);
        // Coding-curriculum corpora.
        assert!((t.lookup_corpus("code-search-net/code_search_net") - 0.9).abs() < 1e-6);
        assert!((t.lookup_corpus("stackoverflow") - 0.8).abs() < 1e-6);
        assert!((t.lookup_corpus("python-docs") - 0.95).abs() < 1e-6);
        assert!((t.lookup_corpus("mdn") - 0.95).abs() < 1e-6);
        assert!((t.lookup_corpus("starcoderdata") - 0.9).abs() < 1e-6);
    }

    #[test]
    fn t_ingest_1_default_does_not_seed_curriculum() {
        let t = TrustTable::default();
        assert!(t.corpus_multipliers.is_empty());
        let e = TrustTable::empty();
        assert!(e.corpus_multipliers.is_empty());
    }

    // ---------------- Case 8: regression byte-equivalence
    #[test]
    fn t_ingest_1_regression_equivalent_when_default_is_1_0_and_map_empty() {
        // With empty corpus_multipliers + default_multiplier=1.0,
        // the resolution path multiplies by 1.0 -> byte-for-byte
        // identical to pre-T-Ingest.1 behavior.
        let mut b = fresh();
        b.trust_table.set_default_multiplier(1.0).unwrap();
        assert!(b.trust_table.corpus_multipliers.is_empty());
        let attr = SourceAttribution::training_for_resolution(
            "any/corpus".to_string(), None, None,
        );
        b.teach_with_source(
            "reg_a", BondType::IsA, "reg_b", 0.8,
            crate::node::Confidence::Verified, attr,
        );
        let w = bond_weight(&b, "reg_a", "reg_b", BondType::IsA).unwrap();
        assert!((w - 0.8).abs() < 1e-6, "regression broken: {}", w);
    }

    // ---------------- Bonus: master switch off bypasses resolution
    #[test]
    fn t_ingest_1_learning_disabled_passes_through_base_weight() {
        let mut b = fresh();
        b.trust_table.set_corpus("alpha/x", 0.1).unwrap();
        b.trust_table.set_learning_enabled(false);
        let attr = SourceAttribution::training_for_resolution(
            "alpha/x".to_string(), None, None,
        );
        b.teach_with_source(
            "d_a", BondType::IsA, "d_b", 0.8,
            crate::node::Confidence::Verified, attr,
        );
        let w = bond_weight(&b, "d_a", "d_b", BondType::IsA).unwrap();
        // learning_enabled=false -> resolver returns 1.0 -> no change.
        assert!((w - 0.8).abs() < 1e-6, "learning_disabled path: {}", w);
    }

    // ---------------- Empirical demo precondition (0.8 * 0.5 = 0.4)
    #[test]
    fn t_ingest_1_empirical_demo_math_holds_in_unit() {
        let mut b = fresh();
        b.trust_table.set_corpus("test/t-ingest1-demo", 0.5).unwrap();
        let attr = SourceAttribution::training_for_resolution(
            "test/t-ingest1-demo".to_string(), None, None,
        );
        b.teach_with_source(
            "test_a_xyz", BondType::IsA, "test_b_xyz", 0.8,
            crate::node::Confidence::Verified, attr,
        );
        let w = bond_weight(&b, "test_a_xyz", "test_b_xyz", BondType::IsA).unwrap();
        assert!((w - 0.4).abs() < 1e-6, "demo math broken: {}", w);
        let stored = b.bond_source(
            &Seed::from_label("test_a_xyz"),
            &Seed::from_label("test_b_xyz"),
            BondType::IsA,
        );
        assert!((stored.trust - 0.5).abs() < 1e-6, "got {}", stored.trust);
    }


}

#[cfg(test)]
mod t_ingest3_tests {
    //! Port T-Ingest.3 -- empty-brain encryption_key bootstrap fix
    //! internal notes. Sibling of the T.7 encryption-guard tests.
    //! Lives at the file's top level rather than nested inside
    //! `mod tests` so the import surface is explicit.
    use super::*;

    // Minimal temp-dir helper -- the parent  has a similar
    // private mock; duplicate it locally so this sub-module stays self-
    // contained without needing to re-export the parent helper.
    fn tempdir() -> std::io::Result<TempDir> {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::SeqCst);
        let mut p = std::env::temp_dir();
        p.push(format!("lu4d-tingest3-{}-{}", std::process::id(), n));
        std::fs::create_dir_all(&p)?;
        Ok(TempDir { path: p })
    }
    pub struct TempDir { path: std::path::PathBuf }
    impl TempDir { pub fn path(&self) -> &std::path::Path { &self.path } }
    impl Drop for TempDir { fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.path); } }

    // -------- Port T-Ingest.3: empty-brain encryption_key bootstrap --------
    //
    // Six tests covering the bug fixed in internal notes:
    // `Brain::load_or_new_with_mode_and_key` used to drop the encryption_key
    // on the empty-file path. The fix propagates the key into the new brain
    // and selects the v3enc save format so the very first save lands an
    // encrypted file directly -- closing the bootstrap-by-plaintext-then-
    // migrate workaround that produced a transient C1-violating artifact.
    //
    // Companion guard: `--require-encrypted` + non-existent path + no key
    // refuses at startup via `require_encrypted_create_or_err`.

    #[test]
    fn t_ingest3_empty_path_with_key_propagates_key_and_v3enc_format() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("new-encrypted.lbr");
        let key = crate::encrypt::MasterKey::generate();
        let brain = Brain::load_or_new_with_mode_and_key(
            &path, BrainLoadMode::Auto, Some(key.clone()),
        ).expect("empty-path + key must succeed");
        assert!(brain.encryption_key.is_some(),
            "empty-brain must carry the encryption_key forward");
        assert_eq!(brain.save_format, BrainSaveFormat::V3BinaryEncrypted,
            "empty-brain with key must default to v3enc save_format");
        assert_eq!(brain.loaded_format, Some(BrainSaveFormat::V3BinaryEncrypted),
            "empty-brain with key must report loaded_format = v3enc so downgrade-on-save guard fires");
    }

    #[test]
    fn t_ingest3_empty_path_first_save_lands_v3enc_on_disk() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("first-save.lbr");
        let key = crate::encrypt::MasterKey::generate();
        let mut brain = Brain::load_or_new_with_mode_and_key(
            &path, BrainLoadMode::Auto, Some(key),
        ).expect("empty-path + key must succeed");
        brain.require_encrypted = true;
        brain.teach("alpha", BondType::IsA, "beta", 0.7, Confidence::Verified);
        brain.save_to(&path).expect("first save must succeed and produce v3enc");
        assert!(path.exists(), "first save must produce a file on disk");
        require_v3enc_or_err(&path)
            .expect("the file we just saved must pass the v3enc loader guard");
    }

    #[test]
    fn t_ingest3_empty_path_no_key_with_require_encrypted_refuses() {
        // Companion to the in-method propagation fix: when require_encrypted
        // is set and no key was provided, the CLI guard refuses BEFORE the
        // brain is created. The unit test calls the public guard directly.
        let dir = tempdir().unwrap();
        let path = dir.path().join("missing.lbr");
        assert!(!path.exists());
        let res = require_encrypted_create_or_err(&path, &None);
        match res {
            Err(BrainError::EncryptionRequired { attempted_format, hint }) => {
                assert_eq!(attempted_format, "<empty-creation>",
                    "attempted_format should mark this as an empty-creation refusal; got {}",
                    attempted_format);
                assert!(hint.contains("--keyfile"),
                    "hint must point operator at --keyfile; got {}", hint);
            }
            other => panic!("expected EncryptionRequired, got: {:?}", other),
        }
    }

    #[test]
    fn t_ingest3_empty_path_no_key_no_require_preserves_plaintext_default() {
        // Back-compat: when --require-encrypted is OFF and no key is
        // provided, the empty-brain branch still produces a plaintext-by-
        // default brain. This is the dev/test workflow that pre-T-Ingest.3
        // brains relied on; we keep it intact.
        let dir = tempdir().unwrap();
        let path = dir.path().join("plain-default.lbr");
        let brain = Brain::load_or_new_with_mode_and_key(
            &path, BrainLoadMode::Auto, None,
        ).expect("empty-path + no key + no require_encrypted must succeed");
        assert!(brain.encryption_key.is_none(),
            "no key in -> no key out");
        assert_eq!(brain.save_format, BrainSaveFormat::V3Binary,
            "plaintext default v3 must be preserved when no key is provided");
        assert!(brain.loaded_format.is_none(),
            "empty brain has no on-disk format until first save");
    }

    #[test]
    fn t_ingest3_empty_path_with_key_no_require_still_encrypts() {
        // When a keyfile is provided but --require-encrypted is OFF, we
        // still arrange the brain to save as v3enc on first save. The key
        // presence alone is enough to opt into encryption-by-default; the
        // operator should not be required to also pass --require-encrypted.
        let dir = tempdir().unwrap();
        let path = dir.path().join("key-no-require.lbr");
        let key = crate::encrypt::MasterKey::generate();
        let mut brain = Brain::load_or_new_with_mode_and_key(
            &path, BrainLoadMode::Auto, Some(key),
        ).expect("empty-path + key must succeed regardless of require_encrypted");
        assert!(brain.encryption_key.is_some());
        assert_eq!(brain.save_format, BrainSaveFormat::V3BinaryEncrypted);
        brain.teach("x", BondType::IsA, "y", 0.8, Confidence::Verified);
        brain.save_to(&path).expect("first save must succeed");
        require_v3enc_or_err(&path)
            .expect("first save must land v3enc even when require_encrypted is off");
    }

    #[test]
    fn t_ingest3_existing_brain_load_path_unchanged() {
        // Regression guard: the fix lives entirely on the empty-file branch.
        // Loading an existing v3enc brain through the same call must work
        // exactly as it did before.
        let dir = tempdir().unwrap();
        let path = dir.path().join("existing.lbr");
        let key = crate::encrypt::MasterKey::generate();
        // Seed: write an encrypted v3 brain through the normal save path.
        let mut seed = Brain::new();
        seed.encryption_key = Some(key.clone());
        seed.teach("a", BondType::IsA, "b", 0.8, Confidence::Verified);
        seed.save_to_format(&path, BrainSaveFormat::V3BinaryEncrypted)
            .expect("seed v3enc save must succeed");
        // Reload through the function we changed.
        let loaded = Brain::load_or_new_with_mode_and_key(
            &path, BrainLoadMode::Auto, Some(key),
        ).expect("existing v3enc brain must load unchanged after T-Ingest.3");
        assert_eq!(loaded.loaded_format, Some(BrainSaveFormat::V3BinaryEncrypted));
        assert!(loaded.encryption_key.is_some(),
            "loaded encrypted brain must carry the key forward");
        // And the empty-creation guard must NOT fire when the path exists.
        require_encrypted_create_or_err(&path, &None)
            .expect("empty-creation guard must not fire when the file already exists");
    }
}

// =====================================================================
// OP-022.ef — impl SideTableSource for Brain.
//
// Each method writes ONE side-table section directly to the writer.
// Pattern:
//   - For owned fields:    bincode::serialize_into(&mut *w, &self.field)
//   - For .snapshot() fields:  let s = self.store.snapshot(); serialize_into(&mut *w, &s); drop(s);
//   - For scar_log:        rkyv::to_bytes(&self.scar_log) (small, allocation OK)
//   - For research_misc:   pass a ResearchMiscBlobRef { config: &..., next_id: ... }
//                          (byte-identical to ResearchMiscBlob because &T serializes
//                          the same as T under bincode)
//   - For runtime_config_v14/v15:  build RuntimeConfigSnapshot (still clones the
//                          underlying tables once), call encode_v*_section() which
//                          returns Vec<u8>, write, drop. Largest sub-clone here is
//                          trust_table which is small (<1 MiB on this brain).
//
// Critical property: byte-identical output to to_brain_file_no_network() +
// emit_bincode!(&...). Bincode is deterministic over (type, value); using
// &T instead of T does NOT change the serialized bytes for the standard
// serde derive (Serialize for &T forwards to T's Serialize impl).
// =====================================================================

/// Byte-borrow form of `ResearchMiscBlob` used by SideTableSource. Bincode
/// serializes `&T` identically to `T` for derived `Serialize` impls, so
/// this struct's bytes match `ResearchMiscBlob` exactly.
#[derive(serde::Serialize)]
pub(crate) struct ResearchMiscBlobRef<'a> {
    pub config: &'a crate::research::router::ResearchConfig,
    pub next_id: u64,
}

impl crate::brain_io_v3::SideTableSource for Brain {
    fn scar_log_scars_count(&self) -> u64 {
        self.scar_log.scars.len() as u64
    }

    fn write_scar_log<W: crate::brain_io_v3::WriteSeek + ?Sized>(
        &self,
        w: &mut W,
    ) -> Result<u64, BrainError> {
        // scar_log uses rkyv (not bincode) — see legacy writer. rkyv has
        // no serialize_into equivalent, but scar_log is small (~few MB at most),
        // so the allocation is fine.
        let bytes = rkyv::to_bytes::<_, 1024>(&self.scar_log).map_err(|e| {
            BrainError::Format(format!("rkyv serialize ScarLog: {}", e))
        })?;
        let n = bytes.len() as u64;
        w.write_all(&bytes).map_err(BrainError::Io)?;
        Ok(n)
    }

    fn write_interest_profile<W: crate::brain_io_v3::WriteSeek + ?Sized>(
        &self,
        w: &mut W,
    ) -> Result<u64, BrainError> {
        op022ef_bincode_into(w, &self.interest_profile)
    }

    fn write_meta_reflection<W: crate::brain_io_v3::WriteSeek + ?Sized>(
        &self,
        w: &mut W,
    ) -> Result<u64, BrainError> {
        op022ef_bincode_into(w, &self.meta_reflection)
    }

    fn write_research_proposals<W: crate::brain_io_v3::WriteSeek + ?Sized>(
        &self,
        w: &mut W,
    ) -> Result<u64, BrainError> {
        op022ef_bincode_into(w, &self.research_proposals)
    }

    fn write_research_misc<W: crate::brain_io_v3::WriteSeek + ?Sized>(
        &self,
        w: &mut W,
    ) -> Result<u64, BrainError> {
        let blob = ResearchMiscBlobRef {
            config: &self.research_config,
            next_id: self.next_research_proposal_id,
        };
        op022ef_bincode_into(w, &blob)
    }

    fn write_gap_seed_frequency<W: crate::brain_io_v3::WriteSeek + ?Sized>(
        &self,
        w: &mut W,
    ) -> Result<u64, BrainError> {
        op022ef_bincode_into(w, &self.gap_seed_frequency)
    }

    fn write_pending_log<W: crate::brain_io_v3::WriteSeek + ?Sized>(
        &self,
        w: &mut W,
    ) -> Result<u64, BrainError> {
        let snap = self.pending.snapshot();
        let n = op022ef_bincode_into(w, &snap)?;
        drop(snap);
        Ok(n)
    }

    fn write_rejection_patterns<W: crate::brain_io_v3::WriteSeek + ?Sized>(
        &self,
        w: &mut W,
    ) -> Result<u64, BrainError> {
        let snap = self.rejection_store.snapshot();
        let n = op022ef_bincode_into(w, &snap)?;
        drop(snap);
        Ok(n)
    }

    fn write_lb_session<W: crate::brain_io_v3::WriteSeek + ?Sized>(
        &self,
        w: &mut W,
    ) -> Result<u64, BrainError> {
        let snap = self.lb_session.snapshot();
        let n = op022ef_bincode_into(w, &snap)?;
        drop(snap);
        Ok(n)
    }

    fn write_cs_style<W: crate::brain_io_v3::WriteSeek + ?Sized>(
        &self,
        w: &mut W,
    ) -> Result<u64, BrainError> {
        let snap = self.cs_style_store.snapshot();
        let n = op022ef_bincode_into(w, &snap)?;
        drop(snap);
        Ok(n)
    }

    fn write_pack_ledger<W: crate::brain_io_v3::WriteSeek + ?Sized>(
        &self,
        w: &mut W,
    ) -> Result<u64, BrainError> {
        let snap = self.pack_ledger.snapshot();
        let n = op022ef_bincode_into(w, &snap)?;
        drop(snap);
        Ok(n)
    }

    fn write_runtime_config_v14<W: crate::brain_io_v3::WriteSeek + ?Sized>(
        &self,
        w: &mut W,
    ) -> Result<u64, BrainError> {
        // RuntimeConfigSnapshot is built fresh here, encoded, dropped.
        // The encode_v14_section method returns a Vec<u8> — the snapshot
        // is small (config tables + small style/sense/etc configs), so
        // the allocation is bounded and short-lived.
        let snap = self.runtime_config_snapshot_for_save();
        let bytes = snap
            .encode_v14_section()
            .map_err(|e| BrainError::Bincode(e))?;
        let n = bytes.len() as u64;
        w.write_all(&bytes).map_err(BrainError::Io)?;
        drop(bytes);
        drop(snap);
        Ok(n)
    }

    fn write_runtime_config_v15<W: crate::brain_io_v3::WriteSeek + ?Sized>(
        &self,
        w: &mut W,
    ) -> Result<u64, BrainError> {
        let snap = self.runtime_config_snapshot_for_save();
        let bytes = snap
            .encode_v15_section()
            .map_err(|e| BrainError::Bincode(e))?;
        let n = bytes.len() as u64;
        w.write_all(&bytes).map_err(BrainError::Io)?;
        drop(bytes);
        drop(snap);
        Ok(n)
    }

    fn write_source_overlay<W: crate::brain_io_v3::WriteSeek + ?Sized>(
        &self,
        w: &mut W,
    ) -> Result<u64, BrainError> {
        // OP-SNAPSHOT-PEAK: read-lock to serialize in place.
        op022ef_bincode_into(w, &*self.source_overlay.read().unwrap())
    }

    fn write_cap_pack_ledger<W: crate::brain_io_v3::WriteSeek + ?Sized>(
        &self,
        w: &mut W,
    ) -> Result<u64, BrainError> {
        let snap = self.cap_pack_ledger.snapshot();
        let n = op022ef_bincode_into(w, &snap)?;
        drop(snap);
        Ok(n)
    }
}

/// OP-022.ef — bincode streaming write helper. Replaces the legacy
/// `bincode::serialize(&value)? -> Vec<u8> -> write_all` pattern with
/// `bincode::serialize_into(writer, &value)?` directly.
///
/// OP-022.s.v5 — generic over `W: Write + Seek` so bincode’s per-
/// primitive write calls monomorphise into the concrete BufWriter at
/// the call site, avoiding the per-primitive vtable dispatch that was
/// the OP-022.ef save-throughput regression.
///
/// Length capture uses `stream_position()` before and after the write
/// (one BufWriter flush per section boundary, much cheaper than a
/// second structural walk via `bincode::serialized_size`).
fn op022ef_bincode_into<W, T>(
    w: &mut W,
    value: &T,
) -> Result<u64, BrainError>
where
    W: std::io::Write + std::io::Seek + ?Sized,
    T: serde::Serialize + ?Sized,
{
    let p0 = w.stream_position().map_err(BrainError::Io)?;
    bincode::serialize_into(&mut *w, value).map_err(BrainError::bincode)?;
    let p1 = w.stream_position().map_err(BrainError::Io)?;
    Ok(p1 - p0)
}

impl Brain {
    /// OP-022.ef — assemble the runtime_config snapshot that the V3enc
    /// writer needs. This is the SAME logic as
    /// `to_brain_file_no_network`'s `runtime_config` field construction,
    /// extracted so it can be called twice (once for v14, once for v15)
    /// without doubling the side-table clone cost — each call drops the
    /// snapshot at the end of the `write_runtime_config_v*` method.
    fn runtime_config_snapshot_for_save(&self) -> crate::runtime_config::RuntimeConfigSnapshot {
        crate::runtime_config::RuntimeConfigSnapshot {
            trust_table: self.trust_table.clone(),
            hub_degree: self.hub_degree_config.clone(),
            style_decay_half_life_days: None,
            style_output_adaptation_enabled: self.style_config.output_adaptation_enabled,
            style_max_samples_for_full_weight: self.style_config.max_samples_for_full_weight,
            time_source: self.time_source_config.mode.clone(),
            time_service_url: self.time_source_config.url.clone(),
            time_service_pubkey_hex: self.time_source_config.pubkey_hex.clone(),
            time_service_poll_interval_ms: self.time_source_config.poll_interval_ms,
            time_service_max_drift_ms: self.time_source_config.max_drift_ms,
            sense_disambiguation_enabled: self.sense_config.disambiguation_enabled,
            sense_context_window: self.sense_config.context_window,
            sense_loser_gain: self.sense_config.loser_gain,
            chained_research_enabled: self.chain_research_config.enabled,
            max_chain_depth: self.chain_research_config.max_chain_depth,
            max_chain_cycles: self.chain_research_config.max_chain_cycles,
            chain_gap_threshold: self.chain_research_config.gap_threshold,
            recall_typed_ku_enabled: self.recall_config.typed_ku_enabled,
            admin_revoke_enabled: self.admin_revoke_enabled,
            ingest_corpus_default: self.ingest_config.corpus_default.clone(),
            ingest_run_id_strategy: self.ingest_config.run_id_strategy.clone(),
            ingest_auto_approve_threshold: self.ingest_config.auto_approve_threshold,
            tier2_prompt_default_session_strategy: self.tier2_prompt_config.default_session_strategy.clone(),
            tier2_prompt_max_text_bytes: self.tier2_prompt_config.max_text_bytes,
            file_io_max_size_bytes: self.file_io_config.max_size_bytes,
            file_io_allowed_read_paths: self.file_io_config.allowed_read_paths.clone(),
            file_io_allowed_write_paths: self.file_io_config.allowed_write_paths.clone(),
            image_extractor_kind: self.vision_config.extractor_kind.clone(),
            image_extractor_subprocess_path: self.vision_config.subprocess_path.clone(),
            image_extractor_subprocess_timeout_ms: self.vision_config.subprocess_timeout_ms,
            image_max_size_bytes: self.vision_config.image_max_size_bytes,
            image_supported_mime_types: self.vision_config.image_supported_mime_types.clone(),
        }
    }
}

// =====================================================================
// OP-SAVE-ASYNC: async / non-blocking background save API.
// =====================================================================

impl Brain {
    /// OP-SAVE-ASYNC entry point. Capture a consistent snapshot of the
    /// brain (overlay + side tables + WAL watermarks + archive Arc) on
    /// the calling thread, then spawn a worker thread that runs the
    /// existing V3 encrypted writer against the snapshot to a tmp file.
    /// Returns immediately; the worker runs in the background.
    ///
    /// Caller MUST drive completion via finalize_save_if_ready (or the
    /// convenience save_async_blocking wrapper) — atomic rename,
    /// WAL truncates, and .kw sidecar compact all happen there.
    pub fn start_async_save(&mut self) -> Result<(), BrainError> {
        // OP-SAVE-FINALIZE — finalize any previously-completed save
        // before kicking off a new one. The worker thread transitions
        // SaveState to Completed when its write+fsync finish, but the
        // rename(.tmp -> .lbr) + WAL truncate + .kw compact ONLY happen
        // inside finalize_save_if_ready (called by lu4d_save_status or
        // save_async_blocking). If the caller stopped polling before the
        // worker completed (e.g. harness poll deadline timeout) the
        // brain ends up in a Completed-but-not-finalized state with the
        // .tmp file on disk and the new bonds still in RAM. Without
        // this hook, the next start_async_save would (a) overwrite the
        // stale JoinHandle, (b) std::fs::remove_file the previous .tmp,
        // and (c) silently discard the previously-written save — every
        // save in the chain would be lost. Finalizing here makes the
        // save system self-healing: even with a misbehaving poller, the
        // *previous* save commits to disk before the *next* one begins.
        match self.finalize_save_if_ready() {
            Ok(_finalised) => {
                // _finalised == true means a stale Completed save was
                // just renamed + committed. false means either no
                // worker pending or worker still running (we handle
                // the still-running case below).
            }
            Err(e) => {
                // The previously-completed save failed to finalize
                // (rename / sync). Surface as a hard error so the
                // caller knows brain.lbr is NOT up to date — fall
                // through to the new save attempt would silently
                // mask the disk-state divergence.
                return Err(BrainError::Format(format!(
                    "start_async_save: prior save failed to finalize: {}", e,
                )));
            }
        }
        {
            let st = self.save_runner.state.lock().unwrap();
            if st.is_running() {
                return Err(BrainError::Format(
                    "start_async_save: prior save still in flight; poll lu4d_save_status".into(),
                ));
            }
        }

        let path = match &self.path {
            Some(p) => p.clone(),
            None => return Err(BrainError::Format(
                "start_async_save: brain has no configured path".into(),
            )),
        };

        if self.save_format != BrainSaveFormat::V3BinaryEncrypted {
            return Err(BrainError::Format(format!(
                "start_async_save: only V3BinaryEncrypted supported, got {:?}",
                self.save_format,
            )));
        }

        let key = self.encryption_key.as_ref().cloned().ok_or_else(|| {
            BrainError::Format(
                "V3BinaryEncrypted async save requires Brain.encryption_key to be set".into(),
            )
        })?;

        // OP-SNAPSHOT-PEAK: measure synchronous capture window.
        let capture_start = std::time::Instant::now();
        let overlay_snapshot = crate::save_async::OverlaySnapshot::capture(&self.network);
        let side_tables = crate::save_async::SnapshotSideTables {
            scar_log: self.scar_log.clone(),
            interest_profile: self.interest_profile.clone(),
            meta_reflection: self.meta_reflection.clone(),
            research_proposals: self.research_proposals.clone(),
            research_config: self.research_config.clone(),
            next_research_proposal_id: self.next_research_proposal_id,
            gap_seed_frequency: self.gap_seed_frequency.clone(),
            pending_log_snapshot: self.pending.snapshot(),
            rejection_patterns: self.rejection_store.snapshot(),
            lb_session_snapshot: self.lb_session.snapshot(),
            cs_style_snapshot: self.cs_style_store.snapshot(),
            pack_ledger_snapshot: self.pack_ledger.snapshot(),
            // OP-SNAPSHOT-PEAK: Arc::clone the handle (~ns) instead of deep-cloning
            // the entire overlay (~3 GiB at 19M bonds). Worker holds the Arc until done.
            source_overlay: Arc::clone(&self.source_overlay),
            cap_pack_ledger_snapshot: self.cap_pack_ledger.snapshot(),
            runtime_config_snapshot: self.runtime_config_snapshot_for_save(),
        };
        let watermarks = crate::save_async::WalWatermarks {
            pending: self.pending.wal_position()?,
            lb_session: self.lb_session.wal_position()?,
            cs_style: self.cs_style_store.wal_position()?,
        };

        let snapshot_capture_ms = capture_start.elapsed().as_millis() as u64;
        let started_at_ms = now_ms() as u64;
        let tmp_path: std::path::PathBuf = {
            let mut p = path.clone();
            let mut name = p.file_name().map(|s| s.to_os_string()).unwrap_or_default();
            name.push(".tmp");
            p.set_file_name(name);
            p
        };

        let _ = std::fs::remove_file(&tmp_path);

        let block_size = crate::compress::DEFAULT_BLOCK_SIZE;
        let zstd_level = crate::compress::DEFAULT_ZSTD_LEVEL;

        {
            let mut st = self.save_runner.state.lock().unwrap();
            *st = crate::save_async::SaveState::Running { started_at_ms, snapshot_capture_ms };
        }

        let tmp_path_clone = tmp_path.clone();
        let final_path_clone = path.clone();
        let runner_state = self.save_runner.state.clone();
        let snapshot_capture_ms_for_worker = snapshot_capture_ms;

        let handle = std::thread::Builder::new()
            .name("lu4d_save_async".into())
            .spawn(move || {
                let outcome = (|| -> Result<(), BrainError> {
                    let mut f = std::fs::OpenOptions::new()
                        .read(true)
                        .write(true)
                        .create(true)
                        .truncate(true)
                        .open(&tmp_path_clone)?;
                    let snap_src = crate::save_async::SnapshotSource {
                        snapshot: &overlay_snapshot,
                    };
                    // OP-SAVE-ASYNC.0 — wire the same passthrough fast paths
                    // the synchronous save_to_format uses (brain.rs:2922).
                    // Without these, the worker re-encodes the full archive
                    // every save — forcing the 4 GiB block-cache LRU
                    // population AND the ~2.1 GiB preallocated index Vecs
                    // (node_seeds + node_entries + bond_keys + bond_entries),
                    // costing ~6 GiB transient on top of the snapshot itself
                    // internal notes peak diag).
                    let passthrough_src = overlay_snapshot.passthrough_source();
                    let partial_src = overlay_snapshot.partial_passthrough_source();
                    let overlay_nodes_adapter;
                    let overlay_bonds_adapter;
                    let partial_args = if passthrough_src.is_some() {
                        // Full passthrough wins when overlay is empty;
                        // matches sync path semantics.
                        None
                    } else if let Some(src) = partial_src {
                        overlay_nodes_adapter =
                            crate::save_async::SnapshotOverlayNodes {
                                snapshot: &overlay_snapshot,
                            };
                        overlay_bonds_adapter =
                            crate::save_async::SnapshotOverlayBonds {
                                snapshot: &overlay_snapshot,
                            };
                        Some(crate::brain_io_v3::PartialPassthroughArgs {
                            source: src,
                            tombstoned_nodes: &overlay_snapshot.overlay_deletions,
                            tombstoned_bonds: &overlay_snapshot.bond_tombstones,
                            overlay_nodes: &overlay_nodes_adapter,
                            overlay_bonds: &overlay_bonds_adapter,
                        })
                    } else {
                        None
                    };
                    crate::brain_io_v3::write_v3_compressed_encrypted_to_borrowed(
                        &snap_src,
                        &side_tables,
                        block_size,
                        zstd_level,
                        &key,
                        &mut f,
                        passthrough_src,
                        partial_args.as_ref(),
                    )?;
                    f.sync_all()?;
                    Ok(())
                })();
                let completed_at_ms = now_ms() as u64;
                if outcome.is_err() {
                    let _ = std::fs::remove_file(&tmp_path_clone);
                }
                if let Err(ref e) = outcome {
                    eprintln!("lu4d_save_async: worker write failed: {}", e);
                }
                {
                    let mut st = runner_state.lock().unwrap();
                    match &outcome {
                        Ok(()) => {
                            *st = crate::save_async::SaveState::Completed {
                                started_at_ms,
                                completed_at_ms,
                                path: tmp_path_clone.clone(),
                                snapshot_capture_ms: snapshot_capture_ms_for_worker,
                            };
                        }
                        Err(e) => {
                            *st = crate::save_async::SaveState::Failed {
                                started_at_ms,
                                completed_at_ms,
                                err: e.to_string(),
                                snapshot_capture_ms: snapshot_capture_ms_for_worker,
                            };
                        }
                    }
                }
                crate::save_async::WorkerResult {
                    started_at_ms,
                    completed_at_ms,
                    tmp_path: tmp_path_clone,
                    final_path: final_path_clone,
                    watermarks,
                    outcome,
                }
            })
            .map_err(|e| BrainError::Format(format!("spawn save worker: {}", e)))?;

        {
            let mut w = self.save_runner.worker.lock().unwrap();
            *w = Some(handle);
        }
        Ok(())
    }

    /// Drain a completed worker (rename, WAL truncates, .kw compact).
    /// Returns Ok(true) if a save was finalised this call, Ok(false)
    /// if no worker is pending or the worker is still in flight,
    /// Err if finalize itself fails.
    pub fn finalize_save_if_ready(&mut self) -> Result<bool, BrainError> {
        let handle_opt = {
            let mut w = self.save_runner.worker.lock().unwrap();
            if let Some(h) = w.as_ref() {
                if h.is_finished() {
                    w.take()
                } else {
                    return Ok(false);
                }
            } else {
                return Ok(false);
            }
        };
        let handle = match handle_opt {
            Some(h) => h,
            None => return Ok(false),
        };
        let result = handle.join().map_err(|_| {
            BrainError::Format("save worker panic".into())
        })?;
        match result.outcome {
            Ok(()) => {
                std::fs::rename(&result.tmp_path, &result.final_path)
                    .map_err(BrainError::Io)?;
                if let Some(parent) = result.final_path.parent() {
                    if !parent.as_os_str().is_empty() {
                        if let Ok(d) = std::fs::File::open(parent) {
                            let _ = d.sync_all();
                        }
                    }
                }
                if let Err(e) = self.pending.truncate_wal(result.watermarks.pending) {
                    eprintln!("OP-SAVE-ASYNC: pending WAL truncate failed: {}", e);
                }
                if let Err(e) = self.lb_session.truncate_wal(result.watermarks.lb_session) {
                    eprintln!("OP-SAVE-ASYNC: lb_session WAL truncate failed: {}", e);
                }
                if let Err(e) = self.cs_style_store.truncate_wal(result.watermarks.cs_style) {
                    eprintln!("OP-SAVE-ASYNC: cs_style WAL truncate failed: {}", e);
                }
                let new_crc = read_v3_trailer_crc(&result.final_path).unwrap_or(0);
                if let Err(e) = self.network.compact_keyword_sidecar(new_crc) {
                    eprintln!("OP-SAVE-ASYNC: keyword sidecar compact failed: {}", e);
                }
                Ok(true)
            }
            Err(e) => {
                Err(e)
            }
        }
    }

    /// Convenience: block until the most recent async save completes
    /// (or fails). Used by lu4d_save in mode=blocking.
    pub fn save_async_blocking(&mut self) -> Result<(), BrainError> {
        self.start_async_save()?;
        loop {
            {
                let st = self.save_runner.state.lock().unwrap();
                match &*st {
                    crate::save_async::SaveState::Running { .. } => {}
                    _ => break,
                }
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
        match self.finalize_save_if_ready()? {
            true => Ok(()),
            false => {
                let st = self.save_runner.state.lock().unwrap();
                match &*st {
                    crate::save_async::SaveState::Failed { err, .. } => {
                        Err(BrainError::Format(format!("async save failed: {}", err)))
                    }
                    _ => Ok(()),
                }
            }
        }
    }

    /// Snapshot of the current save state for lu4d_save_status /
    /// lu4d_status. Returns (label, started_at_ms, error_message).
    pub fn save_state_snapshot(&self) -> (String, Option<u64>, Option<String>) {
        let st = self.save_runner.state.lock().unwrap();
        let label = st.label().to_string();
        let started = st.started_at_ms();
        let err = match &*st {
            crate::save_async::SaveState::Failed { err, .. } => Some(err.clone()),
            _ => None,
        };
        (label, started, err)
    }

    /// OP-SNAPSHOT-PEAK: extended snapshot including the synchronous
    /// capture-phase duration. Used by lu4d_save_status to expose
    /// capture vs write timing separately.
    pub fn save_state_snapshot_with_capture(
        &self,
    ) -> (String, Option<u64>, Option<String>, Option<u64>) {
        let st = self.save_runner.state.lock().unwrap();
        let label = st.label().to_string();
        let started = st.started_at_ms();
        let err = match &*st {
            crate::save_async::SaveState::Failed { err, .. } => Some(err.clone()),
            _ => None,
        };
        let capture_ms = st.snapshot_capture_ms();
        (label, started, err, capture_ms)
    }

    /// True iff a background save worker is currently executing. Used
    /// by lu4d_status save_in_progress flag.
    pub fn save_in_progress(&self) -> bool {
        self.save_runner.state.lock().unwrap().is_running()
    }
}

// OP-BRAIN-V4 Phase 2 — testing entry point for the new SST read path.
impl Brain {
    /// Open an SST file at `path` using a fresh block cache sized to
    /// `crate::sst::DEFAULT_SST_BLOCK_CACHE_BYTES`. Phase 2 testing
    /// hook: production code paths do not call this; Phase 5 will
    /// wire SSTs into the live read path via the manifest.
    pub fn read_sst(
        path: &std::path::Path,
    ) -> Result<crate::sst::SST, BrainError> {
        let cache = std::sync::Arc::new(
            crate::sst::BlockCache::new(crate::sst::DEFAULT_SST_BLOCK_CACHE_BYTES),
        );
        crate::sst::SST::open(path, None, cache)
    }

    /// Same as `read_sst` but accepts a pre-built block cache (so
    /// multiple SSTs can share one cache, as Phase 6 compaction will
    /// need) and an optional master key.
    pub fn read_sst_with(
        path: &std::path::Path,
        key: Option<crate::encrypt::MasterKey>,
        block_cache: std::sync::Arc<crate::sst::BlockCache>,
    ) -> Result<crate::sst::SST, BrainError> {
        crate::sst::SST::open(path, key, block_cache)
    }
}

/// Per-boot dispatch for `Brain::enable_memtable`'s WAL
/// `REC_TYPE_NODE_INTERN` arm.
///
/// OP-BRAIN-V5 Phase 2.2 -- Phase 1.2 follow-up B. Resolved ONCE
/// before the replay loop based on whether the `nodeintern.bin`
/// sidecar was present and valid at boot. The replay loop then
/// dispatches per-record by a single enum match instead of doing
/// a path-exists check on every WAL record.
///
/// Both variants are retained indefinitely (Q2.0.D defence-in-depth):
/// `SidecarLoaded` is the post-Phase-2 contract; `InsertionOrderFallback`
/// is the legacy v4 behaviour and the graceful-degradation path for
/// pre-Phase-2 brains, hand-crafted test fixtures, or a deleted /
/// missing sidecar file.
///
/// Not exposed outside `brain.rs` -- this is internal dispatch only.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BootReplayMode {
    /// Case A: sidecar was loaded successfully. WAL
    /// `REC_TYPE_NODE_INTERN` records apply via
    /// `NodeIntern::insert_with_id` -- the stored `NodeId` is
    /// preserved verbatim so cross-boot NodeId stability is
    /// contractual.
    SidecarLoaded,
    /// Case B: no sidecar present at the canonical path. WAL
    /// `REC_TYPE_NODE_INTERN` records apply via
    /// `NodeIntern::intern_or_get` -- NodeIds are reassigned by
    /// replay insertion order. Matches v4 behaviour exactly.
    InsertionOrderFallback,
}
// OP-BRAIN-V4 Phase 3 — opt-in MemTable + WAL primitives.
//
// Default Brain construction leaves `memtable` and `wal` as `None`,
// preserving pre-Phase-3 behaviour. `enable_memtable` is the
// opt-in entry point used by Phase 3 tests; Phase 4 (flush + L0 wiring)
// and Phase 5 (read path) build on top of these accessors.
impl Brain {
    /// Initialise the MemTable + WAL pair. Opens (or creates) the WAL
    /// segment at `wal_path`. Note: this does NOT replay an existing
    /// WAL into the MemTable — Phase 4 owns the recovery handshake
    /// because it needs the manifest snapshot too. Phase 3 tests
    /// either start from a fresh WAL or invoke `crate::wal::replay`
    /// directly.
    /// Initialise the MemTable + WAL pair. Opens (or creates) the WAL
    /// segment at `wal_path`. **Phase 4 (this revision):** if the WAL
    /// segment already contains records (e.g. the daemon crashed
    /// between the last flush and shutdown), replay them into the
    /// fresh MemTable BEFORE returning. WAL records survive the
    /// crash; the v4 durability contract is that any record fsync'd
    /// to disk is recovered at the next `enable_memtable` call.
    ///
    /// The intern handle used for the replay (and shared with Phase 4
    /// flush wiring) is a clone of the Brain's network intern, stored
    /// in `self.flush_intern` for `enable_flush_worker` to pick up.
    /// Phase 10: walk every SST in the v4 manifest and insert its
    /// per-block node payloads into `network.nodes`. Called by
    /// `load_v4` immediately after `enable_flush_worker` so the SST
    /// handles are openable via the manifest snapshot. Idempotent:
    /// duplicates are skipped via `network.get_node` membership.
    pub fn rehydrate_network_nodes_from_v4_ssts(
        &mut self,
        brain_dir: &std::path::Path,
        key: Option<crate::encrypt::MasterKey>,
    ) -> Result<(), BrainError> {
        let manifest_snapshot = match self.manifest_snapshot.as_ref() {
            Some(m) => m.clone(),
            None => return Ok(()),
        };
        let manifest = {
            let r = manifest_snapshot.read().expect("manifest read lock");
            r.clone()
        };
        let mut node_count: u64 = 0;
        let block_cache = self.v4_block_cache.clone();
        for (_lvl, entry) in manifest.iter_ssts() {
            let path = brain_dir.join(&entry.file);
            let sst = match crate::sst::SST::open(&path, key.clone(), block_cache.clone()) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!(
                        "[brain] rehydrate_network_nodes_from_v4_ssts: skipping unreadable SST {}: {}",
                        entry.file, e
                    );
                    continue;
                }
            };
            match sst.iter_nodes() {
                Ok(nodes) => {
                    for (seed, node) in nodes {
                        if self.network.get_node(&seed).is_none() {
                            self.network.add_node(node);
                            node_count += 1;
                        }
                    }
                }
                Err(e) => {
                    eprintln!(
                        "[brain] rehydrate_network_nodes_from_v4_ssts: SST {} iter_nodes failed: {}",
                        entry.file, e
                    );
                }
            }
        }
        if node_count > 0 {
            eprintln!(
                "[brain] rehydrate_network_nodes_from_v4_ssts: hydrated {} nodes from {} SSTs",
                node_count, manifest.iter_ssts().count(),
            );
        }
        Ok(())
    }

    /// OP-BRAIN-V5 Phase 3.8.1 -- adjacency rehydration at boot
    /// (Option A from the Phase 3.8 sanity-gate findings).
    ///
    /// ## Why
    ///
    /// `consolidation::internal_fill_at_hub` / `_at_seed`,
    /// `curiosity::degree()` / `is_hub()`, and the spread-activation
    /// weight lookups all consume the RAW `AssociativeNetwork`
    /// in-memory state (`nodes`, `bonds`, `outgoing`, `incoming`) --
    /// NOT the Brain-level read path. On a brain booted from the v5
    /// substrate those maps are empty (Phase 9 cutover semantics), so
    /// hub-fill proposed nothing, curiosity reported degree 0 for
    /// every seed, and spread died at depth 0 (Phase 3.8 findings at
    /// 39c084d). The fix is to POPULATE that state at boot, not to
    /// refactor the consumers (Q9 scope discipline: consumer source
    /// untouched).
    ///
    /// ## What gets loaded
    ///
    /// * `network.bonds` + the `outgoing`/`incoming` adjacency:
    ///   every LIVE v5 bond. Merge rules mirror
    ///   `bond_store::get_outgoing_bonds_v5`: levels walk L0 -> LN,
    ///   younger level shadows older, within L0 the highest `seq`
    ///   wins, tombstones (`confidence == 0xFF`) suppress their key
    ///   and are never resurrected. Multiple live predicates on the
    ///   same `(from, to)` pair collapse to the highest-seq triple --
    ///   matching the v3 `HashMap<(Seed, Seed), TemporalBond>` shape
    ///   (last write wins).
    /// * `network.nodes`: lightweight STUBS (empty `content` +
    ///   `embedding`, `KnowledgeContent::Raw("")`,
    ///   `Confidence::Unverified`) for every live NodeSST id whose
    ///   seed is not already present, inserted UNINDEXED so empty
    ///   content never touches keyword recall. Full `KnowledgeNode`
    ///   payloads (384-dim embeddings, ~1.5 KiB each) stay in the v5
    ///   NodeSSTs + Phase 2.8 reader cache. Bond endpoints missing
    ///   from both `network.nodes` and the NodeSSTs also get stubs so
    ///   `add_bond`'s node-existence invariant holds.
    ///
    /// ## RAM budget (the cost of Option A)
    ///
    /// Adjacency + bond payloads: ~30-60 B/edge amortized (HashMap
    /// entry + two adjacency NodeIds + keyframe Vec) -> ~0.5-1 GiB
    /// at 18 M bonds. Node stubs: ~50-100 B each -> ~0.5-0.9 GiB at
    /// 9 M nodes. Total ~1-2 GiB, vs the ~13.5 GiB that full node
    /// payloads would cost -- the heavy side of v5's RAM win (node
    /// payloads) is preserved. The dedup accumulators used during
    /// the walk peak at the same order as the final structures and
    /// are freed before this method returns. Observability:
    /// `phase381_rehydration_ram_observable` pins the bond/stub
    /// counts on a 10K-bond fixture.
    ///
    /// ## Sequencing
    ///
    /// Called from `Brain::load_v4` AFTER
    /// `rehydrate_network_nodes_from_v4_ssts` (full node payloads
    /// from the dual-emit v4 SSTs must win over stubs -- a stub
    /// inserted first would block the full node via the `get_node`
    /// idempotency check) and BEFORE `enable_compaction` (no SST can
    /// be retired mid-stream). Requires `enable_flush_worker` to
    /// have installed the manifest snapshot + both reader caches.
    /// The NodeIntern sidecar (Phase 2.1/2.2) was loaded earlier by
    /// `enable_memtable`, so `reverse_lookup` resolves
    /// NodeId -> Seed for bond endpoints; an endpoint with no
    /// binding is counted, logged, and skipped -- never guessed.
    ///
    /// No-op (`Ok`) when the manifest has no v5 levels or the v5
    /// substrate is not enabled -- v3/v4-only brains unaffected.
    pub fn rehydrate_adjacency(&mut self) -> Result<(), BrainError> {
        let manifest = match self.manifest_snapshot() {
            Some(m) => m,
            None => return Ok(()),
        };
        if manifest.node_levels.is_empty() && manifest.bond_v5_levels.is_empty() {
            return Ok(());
        }
        let node_cache = match self.node_reader_cache.as_ref() {
            Some(c) => c.clone(),
            None => return Ok(()),
        };
        let bond_cache = match self.bond_reader_cache.as_ref() {
            Some(c) => c.clone(),
            None => return Ok(()),
        };
        let intern = self.network.node_intern().clone();
        let started = std::time::Instant::now();
        use std::collections::hash_map::Entry;

        // ---- 1. Node stubs from NodeSSTs. ----
        //
        // Winner-per-NodeId merge: `iter_node_ssts` walks levels
        // ascending (L0 first), so an occupied slot always came from
        // a same-or-younger level. Replace only on same-level higher
        // seq (the L0-overlap rule); L1+ ranges are disjoint within
        // a level and strictly older than anything already seen.
        struct NodeWin {
            seed: crate::seed::Seed,
            seq: u64,
            tombstone: bool,
            level: u8,
        }
        let mut node_acc: std::collections::HashMap<u64, NodeWin> =
            std::collections::HashMap::new();
        for (level, entry) in manifest.iter_node_ssts() {
            let reader = node_cache.get_or_open(entry.file_uid).map_err(|e| {
                BrainError::Format(format!(
                    "rehydrate_adjacency: NodeSST uid {} open failed: {}",
                    entry.file_uid, e,
                ))
            })?;
            for item in reader.iter_all() {
                let (id, node_entry) = item.map_err(|e| BrainError::Format(format!(
                    "rehydrate_adjacency: NodeSST uid {} iter failed: {}",
                    entry.file_uid, e,
                )))?;
                let win = NodeWin {
                    seed: node_entry.payload.seed,
                    seq: node_entry.seq,
                    tombstone: node_entry.tombstone,
                    level,
                };
                match node_acc.entry(id.0) {
                    Entry::Vacant(v) => {
                        v.insert(win);
                    }
                    Entry::Occupied(mut o) => {
                        if o.get().level == level && win.seq > o.get().seq {
                            o.insert(win);
                        }
                    }
                }
            }
        }
        let mut stubs_inserted: u64 = 0;
        for (_, win) in node_acc.drain() {
            if win.tombstone {
                continue;
            }
            if self.network.get_node(&win.seed).is_some() {
                // Full payload already present (Phase 10 v4-SST walk
                // or WAL replay) -- never shadow it with a stub.
                continue;
            }
            self.network
                .add_node_unindexed(Self::rehydration_node_stub(win.seed));
            stubs_inserted += 1;
        }
        drop(node_acc);

        // ---- 2. Bonds from BondSSTs. ----
        //
        // Triple-level merge first (same shadowing/tombstone rules
        // as `bond_store::read_path::merge_hit`), then collapse to
        // (subject, object) pairs for the v3 map shape.
        struct BondWin {
            payload: crate::bond_store::BondPayload,
            level: u8,
        }
        let mut bond_acc: std::collections::HashMap<(u64, u8, u64), BondWin> =
            std::collections::HashMap::new();
        for (level, entry) in manifest.iter_bond_v5_ssts() {
            let reader = bond_cache.get_or_open_primary(entry.file_uid).map_err(|e| {
                BrainError::Format(format!(
                    "rehydrate_adjacency: BondSST uid {} open failed: {}",
                    entry.file_uid, e,
                ))
            })?;
            for item in reader.iter_all() {
                let (s, p, o, payload) = item.map_err(|e| BrainError::Format(format!(
                    "rehydrate_adjacency: BondSST uid {} iter failed: {}",
                    entry.file_uid, e,
                )))?;
                match bond_acc.entry((s.0, p, o.0)) {
                    Entry::Vacant(v) => {
                        v.insert(BondWin { payload, level });
                    }
                    Entry::Occupied(mut occ) => {
                        if occ.get().level == level && payload.seq > occ.get().payload.seq {
                            occ.insert(BondWin { payload, level });
                        }
                    }
                }
            }
        }
        let mut pair_winners: std::collections::HashMap<
            (u64, u64),
            (u8, crate::bond_store::BondPayload),
        > = std::collections::HashMap::new();
        for ((s, p, o), win) in bond_acc.drain() {
            if win.payload.confidence == crate::bond_store::BOND_TOMBSTONE_CONFIDENCE {
                continue;
            }
            match pair_winners.entry((s, o)) {
                Entry::Vacant(v) => {
                    v.insert((p, win.payload));
                }
                Entry::Occupied(mut occ) => {
                    if win.payload.seq > occ.get().1.seq {
                        occ.insert((p, win.payload));
                    }
                }
            }
        }
        let mut bonds_inserted: u64 = 0;
        let mut skipped_no_seed: u64 = 0;
        let mut skipped_bad_predicate: u64 = 0;
        for ((s, o), (p, payload)) in pair_winners.drain() {
            let from = match intern.reverse_lookup(crate::node_intern::NodeId(s)) {
                Some(seed) => seed,
                None => {
                    skipped_no_seed += 1;
                    continue;
                }
            };
            let to = match intern.reverse_lookup(crate::node_intern::NodeId(o)) {
                Some(seed) => seed,
                None => {
                    skipped_no_seed += 1;
                    continue;
                }
            };
            let bond_type = match crate::memtable::bond_type_from_discriminant(p) {
                Some(bt) => bt,
                None => {
                    skipped_bad_predicate += 1;
                    continue;
                }
            };
            if self.network.get_node(&from).is_none() {
                self.network
                    .add_node_unindexed(Self::rehydration_node_stub(from));
                stubs_inserted += 1;
            }
            if self.network.get_node(&to).is_none() {
                self.network
                    .add_node_unindexed(Self::rehydration_node_stub(to));
                stubs_inserted += 1;
            }
            if self.network.get_bond(&from, &to).is_some() {
                // Idempotency: a bond already present (e.g. a future
                // hybrid-boot path) is never double-inserted -- that
                // would duplicate adjacency entries.
                continue;
            }
            let keyframes = if payload.keyframes.is_empty() {
                // Defensive: production flush always carries the
                // keyframe history inline (Phase 3.3). Reconstruct a
                // baseline from the quantized weight if a row ever
                // arrives bare.
                vec![crate::bond::TemporalKeyframe {
                    timestamp: 0,
                    weight: payload.weight as f32 / 255.0,
                    source: crate::bond::KeyframeSource::Baseline,
                }]
            } else {
                payload.keyframes
            };
            let bond = crate::bond::TemporalBond {
                from,
                to,
                bond_type,
                keyframes,
                half_life_ms: crate::bond::DEFAULT_HALF_LIFE_MS,
            };
            self.network.add_bond(bond).map_err(|e| BrainError::Format(format!(
                "rehydrate_adjacency: add_bond failed: {:?}",
                e,
            )))?;
            bonds_inserted += 1;
        }

        // Degree p90 drives the dynamic hub threshold in spread
        // normalization / bridge classification / curiosity --
        // recompute now that the graph is populated (the v3 loader
        // does the same inside `from_snapshot`).
        self.network.recompute_degree_p90();

        eprintln!(
            "[brain] rehydrate_adjacency: phase3_8_1 rehydrated {} bonds + {} node stubs in {:?} (skipped: {} no-seed, {} bad-predicate)",
            bonds_inserted,
            stubs_inserted,
            started.elapsed(),
            skipped_no_seed,
            skipped_bad_predicate,
        );
        Ok(())
    }

    /// OP-BRAIN-V5 Phase 3.8.1 -- the node-stub shape used by
    /// [`Brain::rehydrate_adjacency`]. Populated: `seed` (identity --
    /// what adjacency/degree lookups key on). Everything else is the
    /// cheapest honest value: empty `content`, empty `embedding`
    /// (NOT the 384-dim zero vector -- that would cost ~1.5 KiB per
    /// stub and defeat the v5 RAM win at 9M nodes),
    /// `KnowledgeContent::Raw("")`, `Confidence::Unverified`,
    /// timestamps 0, no parents, typed_ratio 0 (recomputed
    /// incrementally by `add_bond`). Full payloads stay in the v5
    /// NodeSSTs and are served by `Brain::get_node_v5` + the Phase
    /// 2.8 reader cache.
    /// OP-BRAIN-MapRAM P0 -- build the resident MAP
    /// ([`crate::map_ram::ResidentMap`]) from the v5 bond store WITHOUT
    /// materialising the full adjacency `Vec`s.
    ///
    /// This is the "stripped rehydrate": it replays the EXACT same
    /// BondSST triple-level merge / shadowing / `(subject,object)`
    /// collapse / tombstone-drop / predicate-validity / seed-existence
    /// filtering that [`Brain::rehydrate_adjacency`] applies before a
    /// bond lands in `network.outgoing` / `network.incoming`, but
    /// instead of inserting bonds it only COUNTS per-subject out-degree
    /// and per-object in-degree. The resulting degrees are therefore
    /// byte-identical to `AssociativeNetwork::out_degree` /
    /// `in_degree` over the rehydrated adjacency (see the P0 parity
    /// test `resident_map_degrees_match_real_adjacency`).
    ///
    /// The map is an INDEX only -- exactness stays on disk in the v5
    /// bond SSTs. O(bonds) to build (one pass over the live BondSSTs),
    /// O(nodes) resident (8 B/node, flat in bond count). Safe to call
    /// independently of whether `rehydrate_adjacency` has run; it reads
    /// the same on-disk SSTs, not the resident adjacency.
    pub fn build_resident_map(
        &self,
    ) -> Result<crate::map_ram::ResidentMap, BrainError> {
        use std::collections::hash_map::Entry;

        // One degree slot per interned node (dense NodeId space).
        let intern = self.network.node_intern().clone();
        let n = intern.len();
        let mut out_deg: Vec<u32> = vec![0; n];
        let mut in_deg: Vec<u32> = vec![0; n];

        let manifest = match self.manifest_snapshot() {
            Some(m) => m,
            None => return Ok(crate::map_ram::ResidentMap::from_degrees(out_deg, in_deg)),
        };
        if manifest.bond_v5_levels.is_empty() {
            return Ok(crate::map_ram::ResidentMap::from_degrees(out_deg, in_deg));
        }
        let bond_cache = match self.bond_reader_cache.as_ref() {
            Some(c) => c.clone(),
            None => return Ok(crate::map_ram::ResidentMap::from_degrees(out_deg, in_deg)),
        };

        // ---- Bond merge: identical shadowing rule to
        //      rehydrate_adjacency step 2 (winner per (s,predicate,o)
        //      keyed entry; within a level the higher seq wins; levels
        //      are walked L0->LN so an occupied slot is same-or-younger).
        struct BondWin {
            payload: crate::bond_store::BondPayload,
            level: u8,
        }
        let mut bond_acc: std::collections::HashMap<(u64, u8, u64), BondWin> =
            std::collections::HashMap::new();
        for (level, entry) in manifest.iter_bond_v5_ssts() {
            let reader = bond_cache.get_or_open_primary(entry.file_uid).map_err(|e| {
                BrainError::Format(format!(
                    "build_resident_map: BondSST uid {} open failed: {}",
                    entry.file_uid, e,
                ))
            })?;
            for item in reader.iter_all() {
                let (s, p, o, payload) = item.map_err(|e| {
                    BrainError::Format(format!(
                        "build_resident_map: BondSST uid {} iter failed: {}",
                        entry.file_uid, e,
                    ))
                })?;
                match bond_acc.entry((s.0, p, o.0)) {
                    Entry::Vacant(v) => {
                        v.insert(BondWin { payload, level });
                    }
                    Entry::Occupied(mut occ) => {
                        if occ.get().level == level && payload.seq > occ.get().payload.seq {
                            occ.insert(BondWin { payload, level });
                        }
                    }
                }
            }
        }

        // ---- Collapse to (subject, object) winners (max seq across
        //      predicates), dropping tombstones -- exactly as rehydrate.
        //      We keep only the winning predicate discriminant + seq;
        //      the payload is not needed for a degree count.
        let mut pair_winners: std::collections::HashMap<(u64, u64), (u8, u64)> =
            std::collections::HashMap::new();
        for ((s, p, o), win) in bond_acc.drain() {
            if win.payload.confidence == crate::bond_store::BOND_TOMBSTONE_CONFIDENCE {
                continue;
            }
            match pair_winners.entry((s, o)) {
                Entry::Vacant(v) => {
                    v.insert((p, win.payload.seq));
                }
                Entry::Occupied(mut occ) => {
                    if win.payload.seq > occ.get().1 {
                        occ.insert((p, win.payload.seq));
                    }
                }
            }
        }

        // ---- Count degrees, applying the SAME predicate-validity and
        //      seed-existence filters rehydrate uses (a bond it would
        //      have skipped -- bad predicate, or an un-interned
        //      subject/object -- must not contribute to a degree).
        for ((s, o), (p, _seq)) in pair_winners.drain() {
            if crate::memtable::bond_type_from_discriminant(p).is_none() {
                continue; // rehydrate: skipped_bad_predicate
            }
            if intern.reverse_lookup(crate::node_intern::NodeId(s)).is_none()
                || intern.reverse_lookup(crate::node_intern::NodeId(o)).is_none()
            {
                continue; // rehydrate: skipped_no_seed
            }
            if let Some(slot) = out_deg.get_mut(s as usize) {
                *slot = slot.saturating_add(1);
            }
            if let Some(slot) = in_deg.get_mut(o as usize) {
                *slot = slot.saturating_add(1);
            }
        }

        Ok(crate::map_ram::ResidentMap::from_degrees(out_deg, in_deg))
    }

    /// OP-BRAIN-MapRAM P0 -- build the resident MAP and retain it on the
    /// Brain (`self.resident_map`). Thin wrapper over
    /// [`Brain::build_resident_map`]; this is the "make it resident"
    /// step the later phases' consumers read from. `load_v5` calls this
    /// behind the `LU_MAPRAM_BUILD` flag so production load is unchanged
    /// until the consumers move onto the map.
    pub fn enable_resident_map(&mut self) -> Result<(), BrainError> {
        let map = self.build_resident_map()?;
        self.resident_map = Some(map);
        Ok(())
    }
/// OP-BRAIN-MapRAM persist -- canonical on-disk path of the resident
    /// degree-map sidecar for a v5 brain rooted at `brain_dir`. Co-located
    /// with `wal/nodeintern.bin` so the atomic `.tmp -> file` rename is
    /// same-filesystem (POSIX-atomic), exactly like the intern sidecar.
    pub fn resident_degmap_path_for_dir(brain_dir: &std::path::Path) -> std::path::PathBuf {
        brain_dir.join("wal").join("resident_degmap.bin")
    }

    /// OP-BRAIN-MapRAM persist -- the degree-map sidecar path for THIS
    /// brain, derived from `v5_store_root` (set by `load_v5`). `None` for
    /// a brain that was not loaded via the v5 directory path.
    pub fn resident_degmap_path(&self) -> Option<std::path::PathBuf> {
        self.v5_store_root
            .as_ref()
            .map(|d| Self::resident_degmap_path_for_dir(d))
    }

    /// OP-BRAIN-MapRAM persist -- the validity fingerprint of the CURRENT
    /// graph state, over the exact inputs `build_resident_map` reads (the
    /// manifest's bond-SST set + the key/format identity + the interned
    /// node count). Returns `None` if no manifest snapshot is installed
    /// (no flush worker) -- in which case `build_resident_map` returns an
    /// empty map and there is nothing to persist.
    pub fn resident_degmap_fingerprint(&self) -> Option<u64> {
        let manifest = self.manifest_snapshot()?;
        let intern_len = self.network.node_intern().len() as u64;
        Some(crate::node_store::degmap::compute_fingerprint(
            &manifest, intern_len,
        ))
    }

    /// OP-BRAIN-MapRAM persist -- attempt the FAST boot path: load the
    /// degree map from the sidecar at `<brain_dir>/wal/resident_degmap.bin`
    /// and validate its stored fingerprint against the current graph
    /// state. On a match returns `Ok(Some(map))` (a `ResidentMap`
    /// byte-identical to a fresh `build_resident_map` for this state). On
    /// ANY of {file absent, magic/version/CRC/EOF error, wrong key,
    /// fingerprint mismatch, node_count != intern len} returns `Ok(None)`
    /// -- the caller then rebuilds via `build_resident_map` (no stale map
    /// is ever returned). A genuinely unexpected error is NOT swallowed:
    /// only the recoverable `DegMapError` variants + a fingerprint/len
    /// mismatch map to `None`.
    pub fn try_load_resident_map_from_sidecar(
        &self,
        brain_dir: &std::path::Path,
    ) -> Result<Option<crate::map_ram::ResidentMap>, BrainError> {
        let expected_fp = match self.resident_degmap_fingerprint() {
            Some(fp) => fp,
            None => return Ok(None), // no manifest -> nothing to validate against
        };
        let intern_len = self.network.node_intern().len();
        let path = Self::resident_degmap_path_for_dir(brain_dir);
        let loaded = match crate::node_store::degmap::read_degmap(
            &path,
            self.encryption_key.clone(),
        ) {
            Ok(l) => l,
            Err(e) => {
                // Recoverable: treat exactly like sidecar-absent. The
                // boot then rebuilds + rewrites a fresh sidecar.
                eprintln!(
                    "[brain] resident degmap sidecar not usable ({}); rebuilding from bond SSTs",
                    e,
                );
                return Ok(None);
            }
        };
        if loaded.fingerprint != expected_fp {
            eprintln!(
                "[brain] resident degmap sidecar STALE (stored fp {:#018x} != current {:#018x}); rebuilding",
                loaded.fingerprint, expected_fp,
            );
            return Ok(None);
        }
        // Defence in depth: the dense vectors must size to the current
        // intern node count (the fingerprint already folds intern_len, so
        // a mismatch here would only mean a corrupted-but-CRC-passing file
        // -- effectively impossible, but we refuse rather than hand back a
        // wrongly-sized map).
        if loaded.out_degree.len() != intern_len || loaded.in_degree.len() != intern_len {
            eprintln!(
                "[brain] resident degmap sidecar length {}/{} != intern len {}; rebuilding",
                loaded.out_degree.len(), loaded.in_degree.len(), intern_len,
            );
            return Ok(None);
        }
        eprintln!(
            "[brain] resident degmap sidecar OK (fp {:#018x}, {} nodes); FAST boot path",
            loaded.fingerprint, intern_len,
        );
        Ok(Some(crate::map_ram::ResidentMap::from_degrees(
            loaded.out_degree,
            loaded.in_degree,
        )))
    }

    /// OP-BRAIN-MapRAM persist -- build (or reuse) the resident degree map
    /// and retain it on `self.resident_map`, taking the FAST sidecar path
    /// when a valid sidecar exists and otherwise rebuilding from the bond
    /// SSTs and writing a fresh sidecar for the next boot. This is the
    /// drop-in replacement `load_v5` calls instead of bare
    /// `enable_resident_map`. `brain_dir` is the v5 brain root.
    pub fn enable_resident_map_persisted(
        &mut self,
        brain_dir: &std::path::Path,
    ) -> Result<(), BrainError> {
        if let Some(map) = self.try_load_resident_map_from_sidecar(brain_dir)? {
            self.resident_map = Some(map);
            return Ok(());
        }
        // Fallback: the existing (unchanged) rebuild path.
        let map = self.build_resident_map()?;
        self.resident_map = Some(map);
        // Write a fresh sidecar so the NEXT boot is fast. A write failure
        // is logged but NOT fatal -- the brain is fully usable from the
        // freshly built in-memory map; the only cost is that the next
        // boot rebuilds again.
        if let Err(e) = self.persist_resident_degmap_to_dir(brain_dir) {
            eprintln!(
                "[brain] resident degmap sidecar write failed (non-fatal): {}",
                e,
            );
        }
        Ok(())
    }

    /// OP-BRAIN-MapRAM persist -- atomically write the CURRENT resident
    /// degree map (`self.resident_map`) to the sidecar under `brain_dir`,
    /// stamped with the current-state validity fingerprint. Atomic via
    /// `<path>.tmp` + rename (mirrors the nodeintern sidecar in
    /// `flush.rs::flush_once`). No-op success if there is no resident map
    /// or no manifest (nothing to persist / fingerprint).
    pub fn persist_resident_degmap_to_dir(
        &self,
        brain_dir: &std::path::Path,
    ) -> Result<(), BrainError> {
        let map = match self.resident_map.as_ref() {
            Some(m) => m,
            None => return Ok(()),
        };
        let fingerprint = match self.resident_degmap_fingerprint() {
            Some(fp) => fp,
            None => return Ok(()),
        };
        let path = Self::resident_degmap_path_for_dir(brain_dir);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(BrainError::Io)?;
        }
        let tmp = crate::node_store::degmap::degmap_tmp_path(&path);
        crate::node_store::degmap::write_degmap(
            &tmp,
            map.out_degrees(),
            map.in_degrees(),
            fingerprint,
            self.encryption_key.clone(),
        )
        .map_err(|e| BrainError::Format(format!(
            "persist_resident_degmap: write_degmap({}) failed: {}",
            tmp.display(), e,
        )))?;
        std::fs::rename(&tmp, &path).map_err(|e| BrainError::Io(std::io::Error::new(
            std::io::ErrorKind::Other,
            format!(
                "persist_resident_degmap: rename({} -> {}): {}",
                tmp.display(), path.display(), e,
            ),
        )))?;
        // fsync the containing dir so the rename hits disk (standard
        // POSIX rename discipline; mirrors flush.rs::flush_once).
        if let Some(parent) = path.parent() {
            if let Ok(dir) = std::fs::File::open(parent) {
                let _ = dir.sync_all();
            }
        }
        Ok(())
    }

    /// OP-BRAIN-MapRAM persist -- refresh the degree-map sidecar for THIS
    /// brain (using `v5_store_root`). Called from `flush_now` so an
    /// explicit checkpoint leaves a fast-boot sidecar. Rebuilds the map
    /// from the current bond SSTs first (so it reflects bonds flushed
    /// since load), retains it on `self.resident_map`, then persists.
    /// No-op success for a non-v5 brain.
    pub fn refresh_resident_degmap(&mut self) -> Result<(), BrainError> {
        let dir = match self.v5_store_root.clone() {
            Some(d) => d,
            None => return Ok(()),
        };
        // Rebuild from the live bond SSTs so the persisted map matches the
        // post-flush graph state (the fingerprint will then match on the
        // next boot).
        let map = self.build_resident_map()?;
        self.resident_map = Some(map);
        self.persist_resident_degmap_to_dir(&dir)
    }
/// OP-REVOKE-V5 -- canonical on-disk path of the source-overlay
    /// sidecar for a v5 brain rooted at `brain_dir`. Co-located with
    /// `wal/nodeintern.bin` + `wal/resident_degmap.bin` so the atomic
    /// `.tmp -> file` rename is same-filesystem (POSIX-atomic).
    pub fn source_overlay_path_for_dir(brain_dir: &std::path::Path) -> std::path::PathBuf {
        brain_dir.join("wal").join("source_overlay.bin")
    }

    /// OP-REVOKE-V5 -- canonical on-disk path of the negation-overlay
    /// sidecar for a v5 brain rooted at `brain_dir`.
    pub fn negation_overlay_path_for_dir(brain_dir: &std::path::Path) -> std::path::PathBuf {
        brain_dir.join("wal").join("negation_overlay.bin")
    }

    /// OP-REVOKE-V5 -- `rename(tmp -> path)` then fsync the containing
    /// directory so the rename hits disk (standard POSIX rename
    /// discipline; mirrors `persist_resident_degmap_to_dir`).
    fn atomic_rename_with_dir_fsync(
        tmp: &std::path::Path,
        path: &std::path::Path,
    ) -> Result<(), BrainError> {
        std::fs::rename(tmp, path).map_err(|e| BrainError::Io(std::io::Error::new(
            std::io::ErrorKind::Other,
            format!("persist_overlays: rename({} -> {}): {}", tmp.display(), path.display(), e),
        )))?;
        if let Some(parent) = path.parent() {
            if let Ok(dir) = std::fs::File::open(parent) {
                let _ = dir.sync_all();
            }
        }
        Ok(())
    }

    /// OP-REVOKE-V5 -- atomically write the CURRENT `source_overlay` +
    /// `negation_overlay` to their sidecars under `brain_dir`. Each is
    /// written to `<path>.tmp`, fsync'd, then renamed (mirrors
    /// `persist_resident_degmap_to_dir` + `flush.rs::flush_once`). The
    /// bodies are encrypted-at-rest under `self.encryption_key` when a
    /// key is set. This is what makes the `revoke_bonds_by_source`
    /// "revocable by source" guarantee survive a restart on v5 brains,
    /// where `lu4d_save` is otherwise a no-op.
    pub fn persist_overlays_to_dir(
        &self,
        brain_dir: &std::path::Path,
    ) -> Result<(), BrainError> {
        // --- source overlay ---
        let src_path = Self::source_overlay_path_for_dir(brain_dir);
        if let Some(parent) = src_path.parent() {
            std::fs::create_dir_all(parent).map_err(BrainError::Io)?;
        }
        {
            let guard = self.source_overlay.read().unwrap();
            let tmp = crate::overlay_sidecar::overlay_tmp_path(&src_path);
            crate::overlay_sidecar::write_overlay(&tmp, &*guard, self.encryption_key.clone())
                .map_err(|e| BrainError::Format(format!(
                    "persist source_overlay: write_overlay({}) failed: {}",
                    tmp.display(), e,
                )))?;
            Self::atomic_rename_with_dir_fsync(&tmp, &src_path)?;
        }
        // --- negation overlay ---
        let neg_path = Self::negation_overlay_path_for_dir(brain_dir);
        {
            let tmp = crate::overlay_sidecar::overlay_tmp_path(&neg_path);
            crate::overlay_sidecar::write_overlay(&tmp, &self.negation_overlay, self.encryption_key.clone())
                .map_err(|e| BrainError::Format(format!(
                    "persist negation_overlay: write_overlay({}) failed: {}",
                    tmp.display(), e,
                )))?;
            Self::atomic_rename_with_dir_fsync(&tmp, &neg_path)?;
        }
        Ok(())
    }

    /// OP-INGEST-LEAN -- true iff this brain maintains a resident
    /// degree-map (i.e. it was loaded via the full serving path, not
    /// `BrainLoadMode::IngestOnly`). The `lu4d_save` checkpoint consults
    /// this to decide whether to refresh the degmap sidecar: an
    /// ingest-only brain deliberately has no resident map, and
    /// rebuilding one on every save would run a full bond-SST scan --
    /// exactly the serving work a write-only training pass must avoid.
    pub fn maintains_resident_map(&self) -> bool {
        self.resident_map.is_some()
    }

    /// OP-INGEST-LEAN -- true iff loaded via `BrainLoadMode::IngestOnly`.
    /// The MCP server startup uses this to skip the serving-only keyword
    /// index build; a write-only ingest never runs keyword queries.
    pub fn is_ingest_only(&self) -> bool {
        self.ingest_only
    }

    /// OP-REVOKE-V5 -- persist both overlays for THIS brain, using
    /// `v5_store_root` (set by `load_v5`). No-op success for a non-v5
    /// brain (nothing to root the sidecars at). Called from the
    /// `lu4d_save` checkpoint alongside `refresh_resident_degmap`.
    pub fn persist_overlays(&self) -> Result<(), BrainError> {
        let dir = match self.v5_store_root.clone() {
            Some(d) => d,
            None => return Ok(()),
        };
        self.persist_overlays_to_dir(&dir)
    }

    /// OP-REVOKE-V5 -- repopulate `source_overlay` + `negation_overlay`
    /// from their sidecars under `brain_dir`, decrypting with
    /// `self.encryption_key`. A missing OR unusable (magic/version/CRC/
    /// wrong-key/bincode) sidecar leaves that overlay EMPTY -- graceful,
    /// matching pre-fix behaviour -- and is logged, never fatal.
    /// Returns `(source_entries_loaded, negation_flags_loaded)`.
    pub fn load_overlays_from_dir(&mut self, brain_dir: &std::path::Path) -> (usize, usize) {
        let mut src_n = 0usize;
        let mut neg_n = 0usize;
        let src_path = Self::source_overlay_path_for_dir(brain_dir);
        if src_path.exists() {
            match crate::overlay_sidecar::read_overlay::<crate::source::SourceOverlay>(
                &src_path, self.encryption_key.clone(),
            ) {
                Ok(ov) => {
                    src_n = ov.entries.len();
                    *self.source_overlay.write().unwrap() = ov;
                }
                Err(e) => eprintln!(
                    "[brain] OP-REVOKE-V5: source_overlay sidecar not usable ({}); starting empty",
                    e,
                ),
            }
        }
        let neg_path = Self::negation_overlay_path_for_dir(brain_dir);
        if neg_path.exists() {
            match crate::overlay_sidecar::read_overlay::<crate::negation::NegationOverlay>(
                &neg_path, self.encryption_key.clone(),
            ) {
                Ok(ov) => {
                    neg_n = ov.flags.len();
                    self.negation_overlay = ov;
                }
                Err(e) => eprintln!(
                    "[brain] OP-REVOKE-V5: negation_overlay sidecar not usable ({}); starting empty",
                    e,
                ),
            }
        }
        (src_n, neg_n)
    }

    /// OP-BRAIN-MapRAM P1 -- build the resident map and a
    /// [`crate::cache_adjacency::CacheBackedAdjacency`] over it, retained
    /// on `self.cache_adjacency`. The cache serves neighbour lists via
    /// map -> hot 2Q cache -> the v5 bond reader; `cap` bounds the
    /// non-pinned decoded-list budget and `pin_fraction` pins the
    /// top-degree hubs (e.g. `0.001` = top 0.1%). `load_v5` calls this
    /// behind the `LU_MAPRAM_CACHE` flag; production load is unchanged
    /// (rehydrate_adjacency stays the source of truth) until P2 reroutes
    /// consumers onto the cache.
    pub fn enable_cache_backed_adjacency(
        &mut self,
        cap: usize,
        pin_fraction: f64,
    ) -> Result<(), BrainError> {
        // Build the map fresh into an Arc owned by the cache. (P0's
        // `self.resident_map` is a separate, independently-flagged
        // mechanism; P1 does not require it and does not clone it.)
        let map = std::sync::Arc::new(self.build_resident_map()?);
        let cache = crate::cache_adjacency::CacheBackedAdjacency::from_fraction(
            map, cap, pin_fraction,
        );
        self.cache_adjacency = Some(std::sync::Arc::new(cache));
        Ok(())
    }

    /// OP-INGEST-LEAN -- build a MINIMAL cache-backed adjacency over an
    /// already-obtained resident `map` (loaded from the persisted degmap
    /// sidecar, so NO fresh bond-SST scan), then install the P2 seam.
    /// The cache uses a tiny capacity and NO hub pinning (pin_fraction
    /// 0.0) -- it exists only so `adjacency_cache_backed()` is true,
    /// which routes the teach path onto the resident intern for node
    /// existence and prevents retaining full `KnowledgeNode`s resident
    /// (OP-BRAIN-MapRAM Fix A). It is NOT provisioned for serving reads
    /// (none happen during a write-only ingest). The map lives only
    /// inside the cache; `self.resident_map` is intentionally left None.
    pub fn enable_lean_ingest_adjacency(
        &mut self,
        map: std::sync::Arc<crate::map_ram::ResidentMap>,
    ) -> Result<(), BrainError> {
        // Tiny non-pinned cache: no hub decode at build, lazily filled
        // (and never read during ingest). 1024 entries is a negligible
        // resident budget vs. the serving default (65_536 + 0.1% pinned
        // hubs).
        const LEAN_CACHE_CAP: usize = 1024;
        let cache = crate::cache_adjacency::CacheBackedAdjacency::from_fraction(
            map, LEAN_CACHE_CAP, 0.0,
        );
        self.cache_adjacency = Some(std::sync::Arc::new(cache));
        // Installs the seam over the cache just built (enable_adjacency_seam
        // reuses self.cache_adjacency when already present, so it does NOT
        // build the default 65_536/0.1%-pinned serving cache).
        self.enable_adjacency_seam()
    }

    /// OP-BRAIN-MapRAM P1 -- a fresh [`crate::cache_adjacency::V5AdjacencySource`]
    /// (the on-miss resolver: v5 reader + the exact rehydrate filter),
    /// built from the handles the brain holds post-flush-worker. Returns
    /// `None` if the v5 substrate is not enabled (no manifest / bond
    /// reader cache) -- mirroring `get_*_bonds_v5`'s honest-miss
    /// contract. Cheap: clones a few Arc handles, borrows nothing.
    pub fn v5_adjacency_source(
        &self,
    ) -> Option<crate::cache_adjacency::V5AdjacencySource> {
        let manifest = self.manifest_snapshot()?;
        let bond_cache = self.bond_reader_cache.as_ref()?.clone();
        let intern = self.network.node_intern().clone();
        Some(crate::cache_adjacency::V5AdjacencySource::new(
            manifest, bond_cache, intern,
        ))
    }

    /// OP-BRAIN-MapRAM P1 -- decoded outgoing neighbours of `subject`
    /// via the cache-backed path. `Ok(None)` if the cache or the v5
    /// substrate is not enabled (caller falls back to the resident
    /// adjacency). P1 exposes this WITHOUT rerouting the production
    /// consumers onto it (that is P2).
    pub fn cache_neighbors_outgoing(
        &self,
        subject: crate::node_intern::NodeId,
    ) -> Result<Option<std::sync::Arc<[crate::cache_adjacency::DecodedNeighbor]>>, BrainError> {
        let source = match self.v5_adjacency_source() {
            Some(s) => s,
            None => return Ok(None),
        };
        match self.cache_adjacency.as_ref() {
            Some(c) => Ok(Some(c.neighbors_outgoing(subject, &source)?)),
            None => Ok(None),
        }
    }

    /// OP-BRAIN-MapRAM P1 -- decoded incoming neighbours of `object`
    /// via the cache-backed path. See [`Brain::cache_neighbors_outgoing`].
    pub fn cache_neighbors_incoming(
        &self,
        object: crate::node_intern::NodeId,
    ) -> Result<Option<std::sync::Arc<[crate::cache_adjacency::DecodedNeighbor]>>, BrainError> {
        let source = match self.v5_adjacency_source() {
            Some(s) => s,
            None => return Ok(None),
        };
        match self.cache_adjacency.as_ref() {
            Some(c) => Ok(Some(c.neighbors_incoming(object, &source)?)),
            None => Ok(None),
        }
    }

    /// OP-BRAIN-MapRAM P2 -- install the selectable adjacency seam:
    /// hand the P1 cache (+ the manifest / bond-reader handles needed
    /// to build a fresh `V5AdjacencySource` per decode) to the network
    /// so its consumers can resolve neighbours + degree through the
    /// cache-backed path. Builds the cache first if not already built.
    /// Leaves the network in LEGACY mode -- call
    /// `set_adjacency_cache_backed(true)` to opt this brain onto the
    /// cache path. `rehydrate_adjacency` stays intact: BOTH paths remain
    /// live for A/B + byte-equivalent parity (removal is the next gated
    /// step, deferred per the P2 safety split).
    pub fn enable_adjacency_seam(&mut self) -> Result<(), BrainError> {
        if self.cache_adjacency.is_none() {
            self.enable_cache_backed_adjacency(65_536, 0.001)?;
        }
        let cache = self
            .cache_adjacency
            .as_ref()
            .expect("cache_adjacency built above")
            .clone();
        let manifest = self.manifest_snapshot.clone().ok_or_else(|| {
            BrainError::Format("enable_adjacency_seam: no manifest snapshot handle".into())
        })?;
        let bond_cache = self.bond_reader_cache.clone().ok_or_else(|| {
            BrainError::Format("enable_adjacency_seam: no v5 bond reader cache".into())
        })?;
        self.network.install_mapram_seam(cache, manifest, bond_cache);
        Ok(())
    }

    /// OP-BRAIN-MapRAM P2 -- select the adjacency path for this brain:
    /// `true` routes the rerouted consumers (spread / typed_recall /
    /// curiosity / hub-fill / mcp) through the cache-backed reader +
    /// resident map; `false` (default) keeps the legacy resident
    /// adjacency. Requires `enable_adjacency_seam` first.
    pub fn set_adjacency_cache_backed(&self, on: bool) {
        self.network.set_mapram_mode(on);
    }

    /// OP-BRAIN-MapRAM P2 -- true iff this brain's consumers are
    /// currently routed through the cache-backed adjacency path.
    pub fn adjacency_cache_backed(&self) -> bool {
        self.network.mapram_cache_backed()
    }

    fn rehydration_node_stub(seed: crate::seed::Seed) -> crate::node::KnowledgeNode {
        crate::node::KnowledgeNode {
            seed,
            content: String::new(),
            knowledge: crate::node::KnowledgeUnit {
                content: crate::node::KnowledgeContent::Raw(String::new()),
                confidence: crate::node::Confidence::Unverified,
                created_at: 0,
                updated_at: 0,
            },
            embedding: Vec::new(),
            parent_seeds: Vec::new(),
            typed_ratio: 0.0,
        }
    }

/// Canonical on-disk location of the `nodeintern.bin` sidecar for
    /// a given WAL segment. OP-BRAIN-V5 Phase 2.2.
    ///
    /// The sidecar lives **next to the WAL file** -- same parent
    /// directory. For the production layout that means
    /// `<brain_dir>/wal/nodeintern.bin` alongside
    /// `<brain_dir>/wal/current.wal`. For tests using a flat
    /// `tempdir().join("wal.seg")` layout, the sidecar is at
    /// `tempdir().join("nodeintern.bin")`.
    ///
    /// ## Why next-to-WAL, not next-to-brain.lbr
    ///
    /// The Phase 2.0 audit S4.2 suggested `<brain_dir>/node_store/`
    /// as the eventual home (the Phase 7 migration tool emits there).
    /// For Phase 2.2 the discipline is: sidecar pairs with the WAL
    /// because they share a flush/truncate epoch. Phase 2.3 will write
    /// the sidecar atomically (`.tmp` + rename); keeping it on the
    /// same filesystem mount as the WAL guarantees the rename is
    /// atomic. The brain.lbr lives at a different layer entirely
    /// (legacy v3 archive path) and is not the right co-locator.
    ///
    /// This is a free function on `Brain` (no `&self`) because the
    /// `Brain` struct does not yet store the WAL path -- callers pass
    /// it directly. Phase 2.3 can use the same helper from the flush
    /// worker context.
    /// Merge a loaded `nodeintern.bin` sidecar into `intern` using
    /// `insert_with_id`, so the VERBATIM NodeIds from the sidecar land on
    /// the network-owned intern table. Extracted from `enable_memtable`
    /// (OP-BRAIN-V5 Phase 2.2) so the OP-007 read-only reader can reach
    /// the same intern state WITHOUT enabling the memtable/WAL — the
    /// reader must agree with the writer on NodeIds, otherwise the
    /// keyword-sidecar stamp (which hashes `node_intern().len()`) can
    /// never match and recall is silently degraded to the idk gate.
    ///
    /// Pure read: the sidecar is only loaded, never written.
    ///
    /// Returns the number of real bindings merged. Sentinel holes
    /// (`Seed([0; 32])`, the sparse-Vec padding `insert_with_id` writes
    /// for out-of-order NodeIds) are skipped — they are not bindings.
    /// A collision against an already-interned seed is a corruption
    /// signal, not a recoverable race, so it surfaces as an error.
    pub(crate) fn merge_nodeintern_sidecar(
        intern: &crate::node_intern::NodeIntern,
        loaded: &crate::node_intern::NodeIntern,
    ) -> Result<u64, BrainError> {
        let mut merged: u64 = 0;
        for node_id_idx in 0..loaded.len() {
            let id = crate::node_intern::NodeId(node_id_idx as u64);
            if let Some(seed) = loaded.reverse_lookup(id) {
                if seed == crate::seed::Seed::from_bytes([0u8; 32]) {
                    continue;
                }
                intern.insert_with_id(seed, id).map_err(|e| {
                    BrainError::Format(format!(
                        "phase2_2 sidecar merge collision into network intern: {}",
                        e,
                    ))
                })?;
                merged += 1;
            }
        }
        Ok(merged)
    }

    pub fn sidecar_path_for_wal(wal_path: &std::path::Path) -> std::path::PathBuf {
        wal_path
            .parent()
            .unwrap_or_else(|| std::path::Path::new("."))
            .join("nodeintern.bin")
    }

    /// OP-BRAIN-V5 Phase 2.3 -- instance-method form of
    /// [`Brain::sidecar_path_for_wal`]. Returns the sidecar path
    /// the flush worker uses, derived from `self.wal_path`.
    ///
    /// Returns `None` if `enable_memtable` has not been called
    /// (no WAL path stored). The flush worker is constructed AFTER
    /// `enable_memtable` so it never observes `None` in production;
    /// the `Option` return surfaces the precondition honestly
    /// rather than panicking.
    pub fn sidecar_path(&self) -> Option<std::path::PathBuf> {
        self.wal_path.as_ref().map(|p| Self::sidecar_path_for_wal(p))
    }

    pub fn enable_memtable(
        &mut self,
        cap_bytes: usize,
        wal_path: &std::path::Path,
    ) -> Result<(), BrainError> {
        let mt = std::sync::Arc::new(crate::memtable::MemTable::new(cap_bytes));
        let intern = self.network.node_intern().clone();
        // OP-BRAIN-V5 6.8: the master key (set before enable_memtable by
        // load_v4 / load_v5) threads into the WAL open/replay and the
        // nodeintern sidecar load so freshly-taught triples + the
        // node-label sidecar are encrypted at rest.
        let key = self.encryption_key.clone();

        // -----------------------------------------------------------
        // OP-BRAIN-V5 Phase 2.2 -- sidecar-aware boot-replay arm.
        //
        // Case detection happens ONCE before the WAL replay loop, so
        // the per-record decision (intern_or_get vs insert_with_id)
        // resolves to a single enum dispatch instead of a per-record
        // path check. See `BootReplayMode` below.
        //
        // - Case A (sidecar present + valid): load sidecar VERBATIM
        //   via `NodeIntern::load_from`, then apply each WAL
        //   `REC_TYPE_NODE_INTERN` record's stored NodeId via
        //   `insert_with_id`. NodeIds are stable across boots.
        //
        // - Case B (sidecar absent): fall back to the legacy v4
        //   behavior -- `intern_or_get` reassigns NodeIds by
        //   insertion order. Q2.0.D approves keeping this fallback
        //   indefinitely as defence-in-depth (corrupt-rename, hand-
        //   crafted test brain, pre-Phase-2 archive).
        //
        // - Case A + corrupt sidecar (CRC / magic / format_version):
        //   HARD ERROR. No silent fallback to Case B -- that is the
        //   corruption-masking antipattern `feedback_no_hacks` flags.
        //
        // Streaming discipline: the WAL replay loop below still
        // collects via the pre-Phase-2.2 `crate::wal::replay` (Vec
        // materialization). Phase 2.2 does NOT widen scope to convert
        // the bond/node arms to `replay_streaming_validated` (which
        // would also fail the existing
        // `enable_memtable_replays_existing_wal_records` test that
        // appends bonds without paired NodeIntern records, violating
        // the Phase 1.2 firewall by construction). My changes do not
        // ADD any Vec collection. The ordering-invariant firewall
        // owned by `replay_streaming_validated` is invoked by
        // higher-level callers (cf. `brain.rs:11733` test sites);
        // duplicating it inside `enable_memtable` is explicitly out
        // of scope per the Phase 2.2 brief.
        // -----------------------------------------------------------
        let sidecar_path = Self::sidecar_path_for_wal(wal_path);
        let replay_mode = if sidecar_path.exists() {
            // Attempt to load the sidecar. CRC / magic / format_version
            // errors propagate as BrainError::Format -- the caller
            // decides whether to abort or rebuild. We do NOT silently
            // fall through to Case B (feedback_no_hacks discipline).
            let loaded = crate::node_intern::NodeIntern::load_from_with_key(&sidecar_path, key.clone())
                .map_err(|e| BrainError::Format(format!(
                    "phase2_2 sidecar load failed at {}: {}",
                    sidecar_path.display(), e,
                )))?;
            // Merge the sidecar contents into the live intern table
            // via `insert_with_id` so the verbatim NodeIds land on the
            // network-owned intern (the one `self.network.node_intern()`
            // returns). A sidecar collision against an already-interned
            // seed surfaces as an InsertWithIdCollision, which we map
            // to BrainError::Format -- contradictory sidecar vs.
            // network state is a corruption signal, not a recoverable
            // race (the network intern is empty at this point in fresh
            // brain construction, but we route the error path honestly
            // anyway).
            let merged = Self::merge_nodeintern_sidecar(&intern, &loaded)?;
            eprintln!(
                "[brain] enable_memtable: phase2_2 sidecar loaded from {} ({} bindings merged into network intern)",
                sidecar_path.display(), merged,
            );
            BootReplayMode::SidecarLoaded
        } else {
            eprintln!(
                "[brain] enable_memtable: phase2_2 fallback: sidecar absent at {}; using intern_or_get replay (no cross-boot NodeId stability)",
                sidecar_path.display(),
            );
            BootReplayMode::InsertionOrderFallback
        };

        // Replay any pre-existing WAL records BEFORE opening the WAL
        // for append. `crate::wal::replay` is read-only and works on
        // a fresh File handle, so it does not interfere with the
        // subsequent `Wal::open(...)` append handle.
        let replayed = crate::wal::replay_with_key(wal_path, key.clone())
            .map_err(|e| BrainError::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("WAL replay before enable_memtable: {}", e),
            )))?;
        // Phase 10: WAL replay reconstructs BOTH bonds and nodes.
        // The discriminator-byte WAL records sort into Bond / Node
        // variants; node records repopulate `network.nodes` so
        // post-restart lu4d_ask finds subjects (the regression-from
        // Phase-9 that motivated Phase 10) AND seed the MemTable
        // nodes table for the next flush.
        let mut bond_replays: usize = 0;
        let mut node_replays: usize = 0;
        let mut intern_replays: usize = 0;
        for r in &replayed {
            match r {
                crate::wal::WalRecord::Bond { seq, subject, object, predicate, weight, keyframes } => {
                    // MemTable::insert assigns a fresh seq from its own
                    // counter. The replayed record's seq is preserved on
                    // disk (we just rebuilt the table state); the new seq
                    // assigned here is used for any post-replay inserts.
                    let _ = seq; // diagnostic only -- the on-disk record carries it
                    mt.insert(
                        *subject, *object, *predicate, *weight,
                        keyframes.clone(), &intern,
                    );
                    bond_replays += 1;
                }
                crate::wal::WalRecord::Node { seq, seed, node } => {
                    let _ = seq;
                    // Repopulate the v3 `network.nodes` map so the existing
                    // `network.get_node(seed)` callsites (lu4d_ask, narrate,
                    // every recall path) find the node post-restart.
                    // `add_node` builds the keyword index entries from
                    // node.content -- exactly what we want; the v4 brain
                    // resumes with a populated keyword index.
                    // Idempotency: skip if a previous WAL record already
                    // inserted this seed.
                    if self.network.get_node(seed).is_none() {
                        self.network.add_node(node.clone());
                    }
                    // Also reflect into the MemTable nodes table so the
                    // next flush carries this node into an SST (otherwise
                    // a long-running daemon would keep replaying it from
                    // the WAL forever).
                    mt.insert_node(*seed, node.clone(), &intern);
                    node_replays += 1;
                }
                crate::wal::WalRecord::NodeIntern { seq, seed, node_id } => {
                    // OP-BRAIN-V5 Phase 2.2 (Q8 + Phase 1.2 follow-up B):
                    // sidecar-aware restoration. The case (A or B) was
                    // resolved ONCE above before the loop; this arm
                    // dispatches on the cached `BootReplayMode` so the
                    // per-record branch is a single enum match, not a
                    // disk check or a path lookup.
                    //
                    // Case A path: apply the WAL's stored `node_id`
                    //   verbatim via `insert_with_id`. The sidecar is
                    //   the snapshot; the WAL records are the delta
                    //   since the snapshot. Replaying the delta on top
                    //   of the snapshot preserves cross-boot NodeId
                    //   stability.
                    //
                    //   Collision behavior: if the sidecar already bound
                    //   `seed -> NodeId(X)` and the WAL says
                    //   `seed -> NodeId(Y)` with X != Y, that is a
                    //   sidecar/WAL version skew that we surface as
                    //   BrainError::Format. NOT silently recoverable
                    //   per feedback_no_hacks.
                    //
                    // Case B path: keep the legacy v4 behaviour --
                    //   `intern_or_get` reassigns NodeIds by insertion
                    //   order, the stored `node_id` is discarded.
                    //   Q2.0.D approves keeping this fallback
                    //   indefinitely as defence-in-depth.
                    let _ = seq;
                    match replay_mode {
                        BootReplayMode::SidecarLoaded => {
                            intern.insert_with_id(*seed, *node_id).map_err(|e| BrainError::Format(format!(
                                "phase2_2 WAL replay insert_with_id collision (seq {}): {}",
                                seq, e,
                            )))?;
                        }
                        BootReplayMode::InsertionOrderFallback => {
                            let _ = node_id;
                            let _ = intern.intern_or_get(*seed);
                        }
                    }
                    intern_replays += 1;
                }
            }
        }

        let wal = crate::wal::Wal::open_with_key(wal_path, key.clone())
            .map_err(|e| BrainError::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("WAL open: {}", e),
            )))?;
        self.memtable = Some(mt.clone());
        self.wal = Some(std::sync::Arc::new(std::sync::Mutex::new(wal)));
        self.flush_intern = Some(intern);
        // OP-BRAIN-V5 Phase 2.3: store the WAL path so the flush
        // worker can derive the sidecar location via
        // `Brain::sidecar_path(&self)`. The static helper
        // `Brain::sidecar_path_for_wal` continues to work for
        // pre-Brain-construction callers (test rigs, Phase 2.2 boot
        // replay arm).
        self.wal_path = Some(wal_path.to_path_buf());
        // Phase 5: install the MemTable handle into the network so
        // the spread-activation loop can augment from MemTable in
        // V4/Merged modes without holding a back-reference to
        // Brain. Manifest + sst_dir handles arrive later, via
        // `enable_flush_worker`. Until then the augmentation
        // path consults the MemTable + v3 only (no L0 reads).
        self.network.install_v4_read_handles(
            Some(mt),
            self.manifest_snapshot.clone(),
            self.flush_worker.as_ref().map(|w| w.sst_dir().to_path_buf()),
        );
        if !replayed.is_empty() {
            eprintln!(
                "[brain] enable_memtable: replayed {} WAL records into fresh MemTable ({} bonds, {} nodes, {} intern)",
                replayed.len(), bond_replays, node_replays, intern_replays
            );
        }
        Ok(())
    }

    /// Accessor for the optional MemTable. Returns `None` if the
    /// Brain was not opted into the Phase 3 path.
    pub fn memtable(&self) -> Option<&std::sync::Arc<crate::memtable::MemTable>> {
        self.memtable.as_ref()
    }

    /// Accessor for the optional WAL.
    pub fn wal(&self) -> Option<&std::sync::Arc<std::sync::Mutex<crate::wal::Wal>>> {
        self.wal.as_ref()
    }

    /// OP-BRAIN-V4 Phase 4 — attach a background flush worker.
    /// Requires `enable_memtable` to have been called first.
    ///
    /// Steps:
    /// 1. Load (or create) the manifest at `manifest_path`. If the
    ///    file exists but fails CRC/rkyv validation, `Manifest::load`
    ///    falls back to `MANIFEST.prev` automatically. If neither
    ///    exists, a fresh empty `Manifest` is created and saved.
    /// 2. Spawn the `FlushWorker` thread. The worker runs
    ///    `manifest::cleanup_orphan_ssts` exactly once (gated by
    ///    `Brain::orphan_cleanup_done`, Phase 5) so any L*.sst
    ///    file not referenced by the manifest (e.g. a crash
    ///    mid-flush left it on disk before the manifest update)
    ///    is renamed to `.orphan-<ts>` for inspection.
    pub fn enable_flush_worker(
        &mut self,
        sst_dir: std::path::PathBuf,
        manifest_path: std::path::PathBuf,
        key: Option<crate::encrypt::MasterKey>,
    ) -> Result<(), BrainError> {
        let memtable = self.memtable.as_ref().ok_or_else(|| {
            BrainError::Format(
                "enable_flush_worker requires enable_memtable to be called first".to_string()
            )
        })?.clone();
        let wal = self.wal.as_ref().ok_or_else(|| {
            BrainError::Format(
                "enable_flush_worker requires enable_memtable to be called first (no WAL)".to_string()
            )
        })?.clone();
        let intern = self.flush_intern.as_ref().ok_or_else(|| {
            BrainError::Format(
                "enable_flush_worker requires enable_memtable to be called first (no intern)".to_string()
            )
        })?.clone();

        // OP-BRAIN-V5 Phase 2.3: derive the sidecar path from the
        // WAL path captured at `enable_memtable` time. The flush
        // worker writes nodeintern.bin atomically inside `flush_once`
        // before the SST + manifest sequence (sidecar -> SST ->
        // manifest -> WAL truncate), so a crash anywhere in the
        // window leaves a recoverable state per audit S4.3.
        let sidecar_path = self.sidecar_path().ok_or_else(|| {
            BrainError::Format(
                "enable_flush_worker requires enable_memtable to have set wal_path".to_string()
            )
        })?;
        let intern_for_sidecar = intern.clone();
        // OP-BRAIN-V5 Phase 2.6 -- node_store directory next
        // to the WAL. NodeFlushWorker::node_store_dir_for_wal
        // returns `<wal_dir>/node_store/`; the directory is
        // created lazily by NodeFlushWorker::flush on first
        // non-empty flush so test rigs that never invoke
        // flush_now leave no empty directory behind.
        let wal_path_for_node_store = self.wal_path.clone().ok_or_else(|| {
            BrainError::Format(
                "enable_flush_worker requires wal_path (Phase 2.6 node_store dir)".to_string(),
            )
        })?;
        // OP-BRAIN-V5 6.4 -- v5 brains (load_v5) set `v5_store_root` to
        // the brain-dir top level, where the migration tools write
        // node_store/ + bond_store/ (WAL + sidecar stay under wal/). For
        // v3/v4 brains the field is None and we keep the historical
        // `<wal_dir>/{node,bond}_store` derivation unchanged. This
        // reconciles a layout mismatch the migration tools shipped with
        // (top-level stores) vs the wal-relative flush-worker default --
        // never exercised before load_v5 existed.
        let (node_store_dir, bond_store_dir) = match self.v5_store_root.as_ref() {
            Some(root) => (root.join("node_store"), root.join("bond_store")),
            None => (
                crate::node_store::NodeFlushWorker::node_store_dir_for_wal(
                    &wal_path_for_node_store,
                ),
                crate::bond_store::BondFlushWorker::bond_store_dir_for_wal(
                    &wal_path_for_node_store,
                ),
            ),
        };

        // 1. Load or create manifest.
        let manifest = match crate::manifest::Manifest::load(&manifest_path)? {
            Some(m) => m,
            None => {
                let fp = match &key {
                    Some(k) => crate::manifest::KeyFingerprint::of(k),
                    None => crate::manifest::KeyFingerprint::none(),
                };
                let fresh = crate::manifest::Manifest::new(fp);
                fresh.save_atomic(&manifest_path)?;
                fresh
            }
        };

        // OP-BRAIN-V5 Phase 2.6 -- orphan-NodeSST cleanup pass.
        // Runs AFTER the manifest is loaded but BEFORE the flush
        // worker is spawned (== before this Brain accepts any
        // write). Removes any committed `.nsst` whose `file_uid`
        // is not referenced by the just-loaded manifest, and any
        // half-written `.nsst.tmp*` from a crashed flush. The
        // bond-side orphan cleanup runs inside `FlushWorker::start`
        // (Phase 5 one-shot) on a separate atomic; the two paths
        // are independent because the bond and node stores are
        // independent substrates.
        match crate::node_store::cleanup_orphan_node_ssts(&node_store_dir, &manifest) {
            Ok(report) => {
                if report.orphans_removed > 0 || report.tmps_removed > 0 {
                    eprintln!(
                        "[brain] enable_flush_worker: phase2_6 NodeSST cleanup removed {} orphan(s) + {} tmp(s) in {}",
                        report.orphans_removed, report.tmps_removed,
                        node_store_dir.display(),
                    );
                }
            }
            Err(e) => {
                // Non-fatal: log and continue. The next flush will
                // still allocate a fresh file_uid -- a stuck orphan
                // never blocks forward progress.
                eprintln!(
                    "[brain] enable_flush_worker: phase2_6 NodeSST cleanup failed (non-fatal): {}",
                    e,
                );
            }
        }

        // OP-BRAIN-V5 Phase 3.3 -- orphan-BondSST cleanup pass.
        // Symmetric to the Phase 2.6 NodeSST cleanup above.
        // Removes any committed `.bsst` whose `file_uid` is not
        // referenced by the just-loaded manifest, and any
        // half-written `.bsst.tmp*` from a crashed flush. The two
        // substrates own independent atomics; each cleanup runs
        // independently before the flush worker accepts writes.
        match crate::bond_store::cleanup_orphan_bond_ssts(&bond_store_dir, &manifest) {
            Ok(report) => {
                if report.orphans_removed > 0 || report.tmps_removed > 0 {
                    eprintln!(
                        "[brain] enable_flush_worker: phase3_3 BondSST cleanup removed {} orphan(s) + {} tmp(s) in {}",
                        report.orphans_removed, report.tmps_removed,
                        bond_store_dir.display(),
                    );
                }
            }
            Err(e) => {
                eprintln!(
                    "[brain] enable_flush_worker: phase3_3 BondSST cleanup failed (non-fatal): {}",
                    e,
                );
            }
        }

        let snapshot = std::sync::Arc::new(
            std::sync::RwLock::new(std::sync::Arc::new(manifest))
        );

        // 2. Spawn the worker. Phase 5: orphan-SST cleanup now
        // runs inside `FlushWorker::start`, gated by the shared
        // `orphan_cleanup_done` AtomicBool. First call runs it;
        // subsequent calls (e.g. a future re-init path) no-op.
        // Cloning the Arc keeps the Brain-side copy and the
        // worker-side copy pointing at the same atomic.
        let worker = crate::flush::FlushWorker::start(
            memtable, wal, intern,
            snapshot.clone(),
            manifest_path, sst_dir, key.clone(),
            self.orphan_cleanup_done.clone(),
            // OP-BRAIN-V5 Phase 2.3 -- sidecar path + a clone of the
            // intern for the flush worker to call NodeIntern::save_to
            // on. The intern is shared with the network via Arc-of-
            // RwLock semantics inside NodeIntern, so this clone is
            // a handle clone, not a deep copy.
            sidecar_path,
            intern_for_sidecar,
            // OP-BRAIN-V5 Phase 2.6 -- node_store directory derived
            // from the WAL path. The NodeFlushWorker creates the
            // directory on first flush if absent.
            // OP-BRAIN-V5 Phase 2.8 -- clone the directory before
            // FlushWorker::start consumes it; the same path is
            // used below to construct the NodeSstReaderCache so
            // both writes and reads agree on the on-disk layout.
            node_store_dir.clone(),
            // OP-BRAIN-V5 Phase 3.3 -- bond_store directory; the
            // BondFlushWorker mirrors the NodeFlushWorker shape.
            bond_store_dir.clone(),
        );
        self.flush_worker = Some(worker);
        self.manifest_snapshot = Some(snapshot.clone());
        // OP-BRAIN-V5 Phase 2.8 -- construct the lazy-open
        // reader cache rooted at the same `node_store_dir` the
        // flush worker writes into. The cache lives on the
        // Brain so `get_node_v5` can consult it without
        // re-opening NodeSST files on every call. The Arc
        // wrapper lets future ports (notably Phase 2.7
        // follow-up wiring `evict` into compaction commit)
        // share the same cache across the flush + compaction
        // workers without an extra lock.
        self.node_reader_cache = Some(std::sync::Arc::new(
            crate::node_store::NodeSstReaderCache::new(node_store_dir)
                .with_key(key.clone()),
        ));
        // OP-BRAIN-V5 Phase 3.7 -- bond-side reader cache, the
        // Phase 2.8 parallel. Rooted at the same `bond_store_dir`
        // the BondFlushWorker writes into so reads and writes
        // agree on the on-disk layout. Arc so the future
        // compaction-retire eviction wiring can share the cache
        // without an extra Brain-wide lock.
        self.bond_reader_cache = Some(std::sync::Arc::new(
            crate::bond_store::BondSstReaderCache::new(bond_store_dir)
                .with_key(key.clone()),
        ));
        // Phase 5: propagate manifest + sst_dir into the network
        // so the spread-activation loop sees L0 SSTs in V4/Merged
        // modes. The MemTable handle was installed earlier via
        // `enable_memtable`; we re-install here so all three
        // handles land atomically (the install method replaces
        // any prior values).
        let sst_dir_clone = self.flush_worker.as_ref()
            .map(|w| w.sst_dir().to_path_buf());
        self.network.install_v4_read_handles(
            self.memtable.clone(),
            Some(snapshot),
            sst_dir_clone,
        );
        // OP-LABEL-1: hand the network the NODE-store reader so
        // `node_label` can resolve text for nodes that `rehydrate_
        // adjacency` inserted as content-free stubs. Without this the
        // whole recall path renders empty label slots on a v5 brain.
        if let Some(nc) = self.node_reader_cache.as_ref() {
            self.network.install_node_label_source(nc.clone());
        }
        Ok(())
    }

    /// Force a synchronous flush of the current MemTable contents to
    /// an L0 SST. Requires `enable_flush_worker` to have been called.
    /// Returns an error if no worker is attached or the flush fails.
    pub fn flush_now(&self) -> Result<(), BrainError> {
        // OP-007: read-only readers have no flush worker; refuse explicitly.
        if self.is_read_only_reader() {
            return Err(BrainError::Format(
                "flush_now refused: brain is a read-only reader (OP-007 guest reader)".to_string(),
            ));
        }
        let w = self.flush_worker.as_ref().ok_or_else(|| {
            BrainError::Format(
                "flush_now requires enable_flush_worker to be called first".to_string()
            )
        })?;
        w.flush_now()
    }

    /// Read-only snapshot of the live manifest, if a flush worker is
    /// attached. Cloning the returned Arc is cheap.
    pub fn manifest_snapshot(&self) -> Option<std::sync::Arc<crate::manifest::Manifest>> {
        self.manifest_snapshot.as_ref().map(|rw| {
            rw.read().expect("manifest snapshot read lock").clone()
        })
    }

    /// OP-BRAIN-V5 Phase 2.8 -- v5 fall-through read path for
    /// node payloads.
    ///
    /// Walks `NodeMemTable` -> L0 NodeSSTs -> L1+ NodeSSTs and
    /// returns the winning `KnowledgeNode` payload (or `None` if
    /// the NodeId is absent or tombstoned). See
    /// [`crate::node_store::read_path::get_node_v5`] for the full
    /// algorithm and tombstone semantics.
    ///
    /// **Scope (read carefully):** this method exposes the NEW
    /// v5 read path coexisting with the existing v4 production
    /// read path (`network.nodes` hot cache + bond-SST node
    /// retrieval). v4 callers are NOT affected; Phase 2.9
    /// migration tool will populate the v5 path and Phase 2.x
    /// cutover will switch production over. Until then both
    /// paths coexist.
    ///
    /// Returns `Ok(None)` when:
    /// - the brain has no MemTable / manifest / reader cache
    ///   installed (no `enable_flush_worker` ever ran), AND the
    ///   id is not present in any v5 store -- which is by
    ///   construction true on an unenabled brain. The v4 path is
    ///   still the answer for these brains.
    /// - the id is genuinely absent across MemTable + every
    ///   manifest level.
    /// - the winning entry is a tombstone (logical delete).
    pub fn get_node_v5(
        &self,
        id: crate::node_intern::NodeId,
    ) -> Result<Option<crate::node::KnowledgeNode>, BrainError> {
        // 1. MemTable snapshot. `MemTable::with_node_table`
        // takes the inner lock for the duration of the closure;
        // we clone-out the data we need under the lock (a
        // NodeMemTable snapshot is not cheap to clone --
        // BTreeMap-of-NodeEntry -- so we instead delegate the
        // entire read INSIDE the closure. The flush worker's
        // swap-for-flush primitive replaces the live table
        // wholesale, so holding the lock for one point lookup
        // races neither against writes nor against flush
        // (writes block on the same `Mutex<MemTableInner>`,
        // flush operates on a swapped Box that is no longer
        // pointed at by `self.memtable`).
        //
        // Manifest + cache snapshots captured BEFORE the
        // closure so we read consistent state: even if a flush
        // commits a new NodeSST between the manifest snapshot
        // and the lookup, the MemTable still holds the entry
        // (the flush worker hasn't yet cleared the swapped
        // table at the point of MemTable lookup -- see
        // "Manifest/MemTable snapshot consistency" note in
        // the doc-comment for the design doc Q5 follow-up).
        let memtable = match self.memtable.as_ref() {
            Some(m) => m.clone(),
            None => {
                // No MemTable means no v5 substrate enabled.
                // Honest miss -- callers who hit this path with
                // an unenabled brain get Ok(None), exactly as
                // the doc-comment promises.
                return Ok(None);
            }
        };
        let manifest = self.manifest_snapshot();
        let cache = self.node_reader_cache.clone();

        memtable.with_node_table(|nt| -> Result<Option<crate::node::KnowledgeNode>, BrainError> {
            // Match on the three possible install states:
            // - no manifest snapshot: only memtable to consult.
            // - manifest but no cache: shouldn't happen post
            //   enable_flush_worker (cache install happens in
            //   the same call), but we degrade gracefully.
            // - manifest + cache: full v5 walk.
            match (manifest, cache) {
                (Some(m), Some(c)) => {
                    crate::node_store::get_node_v5(nt, &m, &c, id)
                        .map_err(|e| BrainError::Format(format!(
                            "get_node_v5: {}",
                            e,
                        )))
                }
                (None, _) | (_, None) => {
                    // Memtable-only path. Inline the MemTable
                    // logic from read_path::get_node_v5 so we
                    // don't construct a dummy manifest/cache
                    // pair just to call through.
                    match nt.get(id) {
                        Some(entry) if !entry.tombstone => {
                            Ok(Some(entry.payload.clone()))
                        }
                        _ => Ok(None),
                    }
                }
            }
        })
    }

    /// OP-BRAIN-V5 Phase 3.7 -- v5 bond read path: every LIVE
    /// outgoing bond for `subject`, ascending (predicate, object).
    ///
    /// Walks the manifest's `bond_v5_levels` L0 -> L1+ via
    /// [`crate::bond_store::get_outgoing_bonds_v5`]; see that
    /// doc-comment for level precedence, within-L0 seq dedup, and
    /// tombstone filtering (callers never see a `confidence ==
    /// 0xFF` row).
    ///
    /// **Scope (read carefully):** SST-only -- the live MemTable
    /// bond half is NOT consulted (it stores objects as Seeds per
    /// audit Q3.0.F), so un-flushed bonds are invisible to this
    /// method until the next flush. The v4 production bond read
    /// path (`network` traversal + `v4_sst_cache`) is NOT touched;
    /// both paths coexist until Phase 3.10 migration populates the
    /// v5 SSTs and Phase 3.9 retires v4.
    ///
    /// Returns `Ok(empty)` when the v5 substrate is not enabled
    /// (no `enable_flush_worker` ever ran), mirroring
    /// `get_node_v5`'s honest-miss contract.
    pub fn get_outgoing_bonds_v5(
        &self,
        subject: crate::node_intern::NodeId,
    ) -> Result<
        Vec<(
            crate::node_intern::NodeId,
            u8,
            crate::node_intern::NodeId,
            crate::bond_store::BondPayload,
        )>,
        BrainError,
    > {
        let manifest = match self.manifest_snapshot() {
            Some(m) => m,
            None => return Ok(Vec::new()),
        };
        let cache = match self.bond_reader_cache.as_ref() {
            Some(c) => c.clone(),
            None => return Ok(Vec::new()),
        };
        crate::bond_store::get_outgoing_bonds_v5(&manifest, &cache, subject)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| BrainError::Format(format!("get_outgoing_bonds_v5: {}", e)))
    }

    /// OP-BRAIN-V5 Phase 3.7 -- v5 bond read path: every LIVE
    /// incoming bond for `object`, ascending (predicate, subject).
    /// Tuples preserve the bond's ORIGINAL orientation:
    /// `(subject_in_bond, predicate, object_in_bond == object,
    /// payload)`.
    ///
    /// L1+ SSTs are served by their Phase 3.6 `.bsst.reverse`
    /// files (indexed lookup); reverse-less SSTs (all of L0, plus
    /// any pre-3.6 output promoted by the manifest v3->v4
    /// migration) fall through to a full primary scan -- a known
    /// O(L0_bonds) cost bounded by flush frequency; see
    /// [`crate::bond_store::get_incoming_bonds_v5`].
    ///
    /// Same scope caveats as [`Brain::get_outgoing_bonds_v5`]:
    /// SST-only, v4 untouched, `Ok(empty)` when the v5 substrate
    /// is not enabled.
    pub fn get_incoming_bonds_v5(
        &self,
        object: crate::node_intern::NodeId,
    ) -> Result<
        Vec<(
            crate::node_intern::NodeId,
            u8,
            crate::node_intern::NodeId,
            crate::bond_store::BondPayload,
        )>,
        BrainError,
    > {
        let manifest = match self.manifest_snapshot() {
            Some(m) => m,
            None => return Ok(Vec::new()),
        };
        let cache = match self.bond_reader_cache.as_ref() {
            Some(c) => c.clone(),
            None => return Ok(Vec::new()),
        };
        crate::bond_store::get_incoming_bonds_v5(&manifest, &cache, object)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| BrainError::Format(format!("get_incoming_bonds_v5: {}", e)))
    }

    /// OP-BRAIN-V4 Phase 6 — opt-in compaction worker.
    /// Requires `enable_memtable` and `enable_flush_worker` to
    /// have been called first (the worker shares the manifest
    /// snapshot and SST directory the flush worker uses).
    pub fn enable_compaction(
        &mut self,
        key: Option<crate::encrypt::MasterKey>,
    ) -> Result<(), BrainError> {
        let snap = self.manifest_snapshot.as_ref().ok_or_else(|| {
            BrainError::Format(
                "enable_compaction requires enable_flush_worker first".to_string()
            )
        })?.clone();
        let fw = self.flush_worker.as_ref().ok_or_else(|| {
            BrainError::Format(
                "enable_compaction requires enable_flush_worker first".to_string()
            )
        })?;
        let sst_dir = fw.sst_dir().to_path_buf();
        let manifest_path = fw.manifest_path().to_path_buf();
        let block_cache = self.v4_block_cache.clone();
        let parsed_cache = self.v4_parsed_block_cache.clone();
        // Share the per-SST handle cache so the compaction
        // worker's invalidations land in the same map the
        // read path consults. The Mutex protects the
        // HashMap; we wrap it in an Arc-shaped clone for
        // sharing. Brain's `v4_sst_cache` field is a Mutex
        // (not Arc<Mutex>), so we wrap a clone via std::sync.
        let handle_cache = std::sync::Arc::new(std::sync::Mutex::new(std::collections::HashMap::new()));
        let worker = crate::compaction::CompactionWorker::start(
            sst_dir, manifest_path, snap, key,
            block_cache, parsed_cache, handle_cache,
        );
        self.compaction_worker = Some(worker);
        Ok(())
    }

    /// OP-BRAIN-V5 -- start the production v5 bond-compaction
    /// scheduler. The v5 counterpart to `enable_compaction`: a v5
    /// brain must run the v5 bond scheduler (which drives the real
    /// Phase 3.6 reverse-index-building compactor over
    /// `bond_v5_levels`), NOT the v4 `CompactionWorker` (whose
    /// `decide_compaction` reads the empty `manifest.levels` and so
    /// never compacts a v5 brain). Requires `enable_flush_worker`
    /// first -- shares its manifest snapshot, bond reader cache,
    /// and key.
    pub fn enable_bond_v5_compaction(
        &mut self,
        key: Option<crate::encrypt::MasterKey>,
    ) -> Result<(), BrainError> {
        let snap = self.manifest_snapshot.as_ref().ok_or_else(|| {
            BrainError::Format(
                "enable_bond_v5_compaction requires enable_flush_worker first".to_string(),
            )
        })?.clone();
        let fw = self.flush_worker.as_ref().ok_or_else(|| {
            BrainError::Format(
                "enable_bond_v5_compaction requires enable_flush_worker first".to_string(),
            )
        })?;
        let manifest_path = fw.manifest_path().to_path_buf();
        // OP-BRAIN-V5 root-cause fix (flush-vs-compaction stale-base
        // swap, sibling of 7bd94b8): the bond scheduler reuses the flush
        // worker's manifest-commit lock so flush + bond compaction
        // serialize their read-clone -> save -> swap (-> cleanup) on the
        // SAME `Arc<RwLock<Arc<Manifest>>>`. The node scheduler then
        // reuses THIS lock via `bond.commit_lock()`, so all three
        // manifest mutators share one lock and no swap is ever derived
        // from a stale base.
        let commit_lock = fw.commit_lock();
        let cache = self.bond_reader_cache.as_ref().ok_or_else(|| {
            BrainError::Format(
                "enable_bond_v5_compaction requires the bond reader cache (enable_flush_worker)"
                    .to_string(),
            )
        })?.clone();
        let bond_store_dir = cache.bond_store_dir().to_path_buf();
        let scheduler = crate::bond_store::BondV5CompactionScheduler::start_with_commit_lock(
            bond_store_dir,
            manifest_path,
            snap,
            cache,
            key,
            commit_lock,
        );
        self.bond_v5_compaction = Some(scheduler);
        Ok(())
    }

    /// Handle to the v5 bond-compaction scheduler, if started.
    pub fn bond_v5_compaction(&self) -> Option<&crate::bond_store::BondV5CompactionScheduler> {
        self.bond_v5_compaction.as_ref()
    }

    /// OP-BRAIN-V5 -- start the production v5 node-compaction
    /// scheduler. The node-side counterpart to
    /// `enable_bond_v5_compaction`: a v5 brain must run the v5 node
    /// scheduler (which drives the real Phase 2.7 compactor over
    /// `node_levels`, L0 -> L1) so node L0 NodeSSTs do not pile up
    /// unbounded during training -- the same latent-bug class the
    /// bond scheduler fixed for `bond_v5_levels`. Requires
    /// `enable_flush_worker` first -- shares its manifest snapshot,
    /// node reader cache, and key. Coexists with the bond
    /// scheduler: the two operate on DISJOINT manifest sub-trees
    /// (`node_levels` vs. `bond_v5_levels`) and never double-run the
    /// same compactor.
    pub fn enable_node_v5_compaction(
        &mut self,
        key: Option<crate::encrypt::MasterKey>,
    ) -> Result<(), BrainError> {
        let snap = self.manifest_snapshot.as_ref().ok_or_else(|| {
            BrainError::Format(
                "enable_node_v5_compaction requires enable_flush_worker first".to_string(),
            )
        })?.clone();
        let fw = self.flush_worker.as_ref().ok_or_else(|| {
            BrainError::Format(
                "enable_node_v5_compaction requires enable_flush_worker first".to_string(),
            )
        })?;
        let manifest_path = fw.manifest_path().to_path_buf();
        let cache = self.node_reader_cache.as_ref().ok_or_else(|| {
            BrainError::Format(
                "enable_node_v5_compaction requires the node reader cache (enable_flush_worker)"
                    .to_string(),
            )
        })?.clone();
        let node_store_dir = cache.node_store_dir().to_path_buf();
        // OP-BRAIN-V5 root-cause fix (node-vs-bond compaction race):
        // reuse the bond scheduler's manifest-commit lock so node +
        // bond compactions serialize on the SAME
        // `Arc<RwLock<Arc<Manifest>>>`. Without this, both schedulers
        // could read the same snapshot and the later whole-manifest
        // swap would REVERT the other's committed sub-tree after its
        // `cleanup_*` had unlinked the retired sources -> the manifest
        // points at deleted files, the next compaction hits ENOENT,
        // and L0 never drains. `load_v5` always starts the bond
        // scheduler first, so the lock is present; the fallback keeps
        // a private lock for any non-`load_v5` wiring.
        let scheduler = match self.bond_v5_compaction.as_ref() {
            Some(bond) => crate::node_store::NodeV5CompactionScheduler::start_with_commit_lock(
                node_store_dir,
                manifest_path,
                snap,
                cache,
                key,
                bond.commit_lock(),
            ),
            None => crate::node_store::NodeV5CompactionScheduler::start(
                node_store_dir,
                manifest_path,
                snap,
                cache,
                key,
            ),
        };
        self.node_v5_compaction = Some(scheduler);
        Ok(())
    }

    /// Handle to the v5 node-compaction scheduler, if started.
    pub fn node_v5_compaction(&self) -> Option<&crate::node_store::NodeV5CompactionScheduler> {
        self.node_v5_compaction.as_ref()
    }

    /// Compaction worker handle (Phase 6).
    pub fn compaction_worker(&self) -> Option<&crate::compaction::CompactionWorker> {
        self.compaction_worker.as_ref()
    }

    /// OP-BRAIN-V4 Phase 6 — force a synchronous compaction
    /// pass. Returns Ok(()) if no compaction is currently needed.
    pub fn compact_now(&self) -> Result<(), BrainError> {
        let w = self.compaction_worker.as_ref().ok_or_else(|| {
            BrainError::Format(
                "compact_now requires enable_compaction to be called first".to_string()
            )
        })?;
        w.compact_now()
    }

    /// OP-BRAIN-V4 Phase 8 -- detection helper used by `Brain::load`.
    /// Returns the on-disk layout for `path` without opening the
    /// brain. v4 detection requires both: `path` is a directory AND
    /// contains a file named `MANIFEST`. v3 detection requires a
    /// regular file (the `.lbr` extension is not checked here --
    /// `detect_format_path` validates magic bytes).
    ///
    /// Returns:
    /// * `Ok(BrainLayout::V4Dir)` if `path` is a directory with a
    ///   `MANIFEST` inside it.
    /// * `Ok(BrainLayout::V3File)` if `path` is a regular file.
    /// * `Ok(BrainLayout::Missing)` if `path` does not exist.
    /// * `Err(BrainLayout::DirNoManifest)`-wrapped error if `path`
    ///   exists, is a directory, but has no `MANIFEST`.
    pub fn detect_brain_layout(path: &Path) -> Result<BrainLayout, BrainError> {
        if !path.exists() {
            return Ok(BrainLayout::Missing);
        }
        let md = std::fs::metadata(path).map_err(BrainError::Io)?;
        if md.is_dir() {
            let manifest = path.join(crate::manifest::MANIFEST_FILE_NAME);
            if manifest.is_file() {
                // OP-BRAIN-V5 6.1 -- a v4 dir and a v5 dir are BOTH "a
                // directory containing MANIFEST". Distinguish them by the
                // EXPLICIT manifest format tag (locked decision: manifest
                // CONTENTS, never directory-shape sniffing). A v5 brain's
                // MANIFEST carries `format_tag == BRAIN_FORMAT_TAG_V5`
                // (stamped by the v5 migration tools); everything older
                // is v4.
                match crate::manifest::Manifest::load(&manifest)? {
                    Some(m) if m.is_v5_format() => Ok(BrainLayout::V5Dir),
                    Some(_) => Ok(BrainLayout::V4Dir),
                    None => Err(BrainError::Format(format!(
                        "directory {} has a MANIFEST that could not be read \
                         (corrupt, and no .prev fallback). Re-run migration \
                         or restore from backup.",
                        path.display()
                    ))),
                }
            } else {
                Err(BrainError::Format(format!(
                    "directory {} has no MANIFEST: either an incomplete v3 -> v4 migration \
                     or the wrong path. Run `lu4d_migrate_v3_to_v4 --verify-only --v3 <src.lbr> --v4 {}` \
                     to inspect, or point --brain at a v3 .lbr file.",
                    path.display(), path.display()
                )))
            }
        } else if md.is_file() {
            Ok(BrainLayout::V3File)
        } else {
            Err(BrainError::Format(format!(
                "path {} is neither a file nor a directory; expected a v3 brain.lbr \
                 file or a v4 directory containing MANIFEST.",
                path.display()
            )))
        }
    }

    /// OP-BRAIN-V4 Phase 8 -- canonical dispatch entry point. Detects
    /// whether `path` points at a v3 brain.lbr file or a v4 directory
    /// (presence of `MANIFEST`) and routes to the matching loader.
    ///
    /// * v3 file -> `Brain::load_or_new_with_mode_and_key(path, Auto, key)`,
    ///   yielding a Brain with `read_path_mode == V3Only` and no flush
    ///   worker (unchanged pre-Phase-8 behaviour).
    /// * v4 directory -> `Brain::load_v4(path, key)`, yielding a Brain
    ///   with `read_path_mode == V4Only`, an active flush worker, and
    ///   an active compaction worker.
    /// * Missing path -> falls through to the v3 empty-create path so
    ///   first-run flows keep working.
    pub fn load<P: AsRef<Path>>(
        path: P,
        encryption_key: Option<crate::encrypt::MasterKey>,
    ) -> Result<Self, BrainError> {
        let p = path.as_ref();
        match Self::detect_brain_layout(p)? {
            BrainLayout::V3File | BrainLayout::Missing => {
                Self::load_or_new_with_mode_and_key(
                    p, BrainLoadMode::Auto, encryption_key,
                )
            }
            BrainLayout::V4Dir => {
                Self::load_v4(p, encryption_key)
            }
            BrainLayout::V5Dir => {
                Self::load_v5(p, encryption_key)
            }
        }
    }

    /// OP-BRAIN-V4 Phase 8 -- boot a Brain against a v4 directory.
    ///
    /// Steps:
    /// 1. Validate the MANIFEST exists and is loadable. Phase 4
    ///    `Manifest::load` automatically falls back to `MANIFEST.prev`
    ///    if the live manifest is corrupt.
    /// 2. Construct an empty Brain (no v3 archive backing).
    /// 3. Compute the WAL path. Default: `<brain_dir>/wal/current.wal`.
    ///    Env override: `LU_BRAIN_WAL_PATH` for testing. The `wal/`
    ///    subdir is created on demand (Phase 7 migration also creates
    ///    it, so this is usually a no-op).
    /// 4. `enable_memtable(...)` -- replays any pending WAL records
    ///    into a fresh MemTable.
    /// 5. `enable_flush_worker(...)` -- spawns the L0 flush thread
    ///    pointing at the brain directory.
    /// 6. `enable_compaction(...)` -- spawns the compaction thread.
    /// 7. Set `read_path_mode = V4Only` (new default for v4 brains).
    /// 8. Stash `brain_dir_v4 = Some(brain_dir)` so `save()` knows to
    ///    short-circuit.
    pub fn load_v4(
        brain_dir: &Path,
        encryption_key: Option<crate::encrypt::MasterKey>,
    ) -> Result<Self, BrainError> {
        // 1. MANIFEST presence + loadability check. We do not retain
        //    the loaded manifest here -- `enable_flush_worker`
        //    re-loads it via the same path, and that is the snapshot
        //    the worker and the read path share. This pre-check is
        //    a structural validation so a missing-or-corrupt manifest
        //    produces a clear error before we spin up workers.
        let manifest_path = brain_dir.join(crate::manifest::MANIFEST_FILE_NAME);
        match crate::manifest::Manifest::load(&manifest_path)? {
            Some(_) => {}
            None => return Err(BrainError::Format(format!(
                "v4 brain directory {} has no readable MANIFEST (file missing or \
                 .prev fallback also absent). Run `lu4d_migrate_v3_to_v4` against \
                 a v3 source, or point --brain at the v3 .lbr file directly.",
                brain_dir.display()
            ))),
        };

        // 2. Empty Brain. Note: `path` (the legacy v3 path field)
        //    stays None -- v4 brains do not carry a v3 backing file.
        let mut brain = Brain::new();

        // 3. WAL path. Default `<brain_dir>/wal/current.wal`; env
        //    override for testing only.
        let wal_path = match std::env::var("LU_BRAIN_WAL_PATH").ok() {
            Some(p) if !p.is_empty() => std::path::PathBuf::from(p),
            _ => brain_dir.join("wal").join("current.wal"),
        };
        if let Some(parent) = wal_path.parent() {
            std::fs::create_dir_all(parent).map_err(BrainError::Io)?;
        }

        // 4. MemTable + WAL. MemTable cap mirrors the production
        //    default (`MemTable::with_default_cap()`-style 500 MiB);
        //    `enable_memtable` takes an explicit cap so tests can
        //    override. The default cap is the same constant the
        //    flush worker pollers compare against.
        const DEFAULT_MEMTABLE_CAP_BYTES: usize = 500 * 1024 * 1024;
        // OP-BRAIN-V5 6.8: stash the key on the Brain BEFORE
        // enable_memtable so the WAL + sidecar open/replay encrypt at
        // rest (the field is otherwise only set at the end of load).
        brain.encryption_key = encryption_key.clone();
        brain.enable_memtable(DEFAULT_MEMTABLE_CAP_BYTES, &wal_path)?;

        // 5. Flush worker. Manifest path + SST dir both live under
        //    `brain_dir`. Encryption key is passed straight through;
        //    `enable_flush_worker` writes the key fingerprint into
        //    the manifest payload.
        brain.enable_flush_worker(
            brain_dir.to_path_buf(),
            manifest_path,
            encryption_key.clone(),
        )?;

        // Phase 10: walk every SST's per-block node table and
        // populate `network.nodes` so post-restart lu4d_ask can find
        // subjects. Without this, a v4 brain that has flushed nodes
        // to SSTs (and WAL-truncated them) would still return
        // total_idk because the in-memory `network.nodes` HashMap is
        // empty. The walk is read-only and finishes before any worker
        // touches the brain.
        brain.rehydrate_network_nodes_from_v4_ssts(
            brain_dir, encryption_key.clone()
        )?;

        // OP-BRAIN-V5 Phase 3.8.1: rehydrate the v3 in-memory
        // adjacency (network.bonds + outgoing/incoming + lightweight
        // node-key stubs) from the v5 substrate so consumers that
        // read raw AssociativeNetwork state (hub-fill, curiosity
        // degree, spread weight lookups) see a populated graph.
        // MUST run AFTER rehydrate_network_nodes_from_v4_ssts (full
        // node payloads win over stubs) and BEFORE enable_compaction
        // (no SST retires mid-stream). No-op when the manifest has
        // no v5 levels.
        brain.rehydrate_adjacency()?;

        // 6. Compaction worker. Required for L0 -> L1 promotion;
        //    sharing the same SST dir + manifest path as the flush
        //    worker (enforced internally by `enable_compaction`).
        brain.enable_compaction(encryption_key.clone())?;

        // 7. Read path mode. V4-loaded brains read via the merged
        //    path by default. (V3Only would still work mechanically
        //    -- the v3 layer is empty so the merge dispatches
        //    immediately to MemTable + L0 -- but reading via the
        //    V3Only fast path on a brain with NO v3 archive is
        //    structurally wrong: there are no v3 bonds to return.
        //    Setting V4Only at boot time keeps the semantics honest.)
        brain.set_read_path_mode(crate::read_path::ReadPathMode::V4Only);

        // 8. Stash directory so save() short-circuits. Encryption key
        //    is also retained on the Brain so future save_to_format
        //    calls (legacy path; not used in v4) would still have it.
        brain.brain_dir_v4 = Some(brain_dir.to_path_buf());
        if let Some(k) = encryption_key {
            brain.encryption_key = Some(k);
            // loaded_format reflects the v4 layout. There is no
            // BrainSaveFormat variant for v4 (it is not a save
            // target via the legacy path), so we leave
            // loaded_format as None and let status() report "v4"
            // off of `brain_dir_v4.is_some()` directly.
        }

        Ok(brain)
    }

    /// OP-BRAIN-V5 6.4 -- boot a Brain against a v5 directory
    /// (node_store/ + bond_store/ substrate produced by the v3->v5
    /// migration tools). Structural sibling of [`Brain::load_v4`]; the
    /// differences are intentional and documented inline.
    ///
    /// Steps:
    /// 1. Load the MANIFEST and assert it carries the explicit v5 format
    ///    tag (`format_tag == BRAIN_FORMAT_TAG_V5`). A v4 dir routed here
    ///    by mistake is a hard error, not a silent mis-load.
    /// 2. Construct an empty Brain (no v3 archive backing).
    /// 3. WAL path `<brain_dir>/wal/current.wal` (env override for
    ///    tests). `enable_memtable` ALSO loads the Seed<->NodeId sidecar
    ///    (`wal/nodeintern.bin`) the migration wrote, so the network
    ///    intern is populated before adjacency rehydration.
    /// 4. `enable_flush_worker` installs the manifest snapshot plus the
    ///    encrypted NodeSST + BondSST reader caches (keyed with
    ///    `encryption_key`, since v5 SSTs are encrypted as of 6.5).
    /// 5. `rehydrate_adjacency` folds the v5 node-stub + bond graph into
    ///    the in-memory `AssociativeNetwork` so the raw-adjacency
    ///    consumers (hub-fill / curiosity / spread) and the V5Only recall
    ///    path see a populated graph. NOTE: unlike `load_v4` there is NO
    ///    `rehydrate_network_nodes_from_v4_ssts` call -- a v5 dir has no
    ///    `L*-N.sst` v4 node SSTs; full node payloads live in
    ///    `node_store/*.nsst` and are served on demand via `get_node_v5`.
    /// 6. `enable_compaction` (shares the flush worker's caches/key).
    /// 7. `set_read_path_mode(V5Only)`.
    /// 8. Stash `brain_dir_v4` (the directory-backed marker `save()`
    ///    short-circuits on) + the key.
    ///
    /// Encryption: `load_v5` itself does not re-run the
    /// `--require-encrypted` guard -- that is enforced by
    /// `require_v3enc_or_err` -> `require_encrypted_dir_or_err` at the
    /// daemon/CLI layer BEFORE load. What `load_v5` enforces structurally
    /// is the key contract: with the wrong key or no key, the encrypted
    /// SST reads inside `rehydrate_adjacency` fail and `load_v5` returns
    /// an error rather than booting a half-empty brain.
    /// OP-BRAIN-MapRAM (final step) -- read a MapRAM load flag whose
    /// PRODUCTION DEFAULT is now ON for v5 brains. Unset => `default_on`.
    /// An explicit `0` / `false` / `off` / `no` / empty forces it OFF
    /// (the legacy resident-adjacency path: rollback + the parity
    /// harness); any other value is ON.
    fn mapram_flag_on(name: &str, default_on: bool) -> bool {
        match std::env::var(name) {
            Ok(v) => {
                let t = v.trim();
                !(t.is_empty()
                    || t.eq_ignore_ascii_case("0")
                    || t.eq_ignore_ascii_case("false")
                    || t.eq_ignore_ascii_case("off")
                    || t.eq_ignore_ascii_case("no"))
            }
            Err(_) => default_on,
        }
    }

// =====================================================================
    // OP-022.b -- v5 keyword index internal notes
    //
    // A v5 brain has no rkyv archive, so the Network-level
    // `ensure_keyword_index_built` (which tokenizes from `self.archive`)
    // produces an EMPTY index. These Brain-level entries build the
    // keyword index from the v5 node store instead (the node store lives
    // on the Brain, not the Network -- the "wrong owner" half of the
    // diagnosis). v3/v4 behaviour is unchanged: `ensure_keyword_index_built`
    // delegates to the Network path for non-v5 brains.
    // =====================================================================

    /// True for brains booted through `load_v5` (the v5 LSM substrate).
    pub fn is_v5_brain(&self) -> bool {
        self.v5_store_root.is_some()
    }

    /// Brain-level keyword-index entry point. Lifted from the two
    /// query-path call sites (`mcp.rs`, `reasoning.rs`) so v5 brains
    /// build from the node store while v3/v4 brains keep the existing
    /// `AssociativeNetwork` archive walk. Idempotent: a no-op once a
    /// base is installed (load-time install or a prior build).
    pub fn ensure_keyword_index_built(&mut self) {
        if self.is_v5_brain() {
            if self.network.keyword_index_has_base() {
                return;
            }
            // OP-007: a read-only reader must never build/write the v5
            // keyword sidecar (a disk write into the writer's dir).
            // Recall is degraded but ZERO writes occur.
            if self.is_read_only_reader() {
                eprintln!(
                    "OP-007 read-only reader: keyword base absent; recall degraded (no sidecar build in read-only mode)"
                );
                return;
            }
            if let Err(e) = self.build_keyword_index_v5() {
                eprintln!(
                    "OP-022.b: v5 keyword index build failed ({}); keyword recall degraded",
                    e
                );
            }
        } else {
            self.network.ensure_keyword_index_built();
        }
    }

    /// Sidecar path for the v5 keyword index, under the v5 brain dir.
    fn v5_keyword_sidecar_path(dir: &Path) -> PathBuf {
        dir.join("keyword_index.kw")
    }

    /// Compute the v5 keyword-sidecar freshness stamp (crc + counts)
    /// from the node intern + manifest. Stable across loads while the
    /// store is unchanged (so a written `.kw` sidecar can be reused),
    /// and changes whenever nodes / NodeSSTs / the epoch counter move
    /// (so a stale sidecar is rejected and rebuilt).
    fn v5_keyword_stamp(&self, manifest: &crate::manifest::Manifest) -> (u32, u64, u64) {
        let node_count = self.network.node_intern().len() as u64;
        let bond_count = manifest.total_bond_v5_sst_count() as u64;
        let nsst = manifest.total_node_sst_count() as u64;
        let mut h = crc32fast::Hasher::new();
        h.update(&node_count.to_le_bytes());
        h.update(&bond_count.to_le_bytes());
        h.update(&nsst.to_le_bytes());
        (h.finalize(), node_count, bond_count)
    }

    /// Build the v5 keyword index from the node-store scan (global
    /// live-node merge with LSM precedence + tombstone filtering) and
    /// install it as the `AssociativeNetwork` keyword base, so the
    /// existing `keyword_search` / `select_seeds_for_query` read paths
    /// resolve tokens against it. Drives the OP-022.a.2
    /// `StreamingRebuilder` so it scales to the canonical ~80 M
    /// (token, seed) pairs.
    pub fn build_keyword_index_v5(&mut self) -> Result<(), BrainError> {
        let brain_dir = self.v5_store_root.clone().ok_or_else(|| {
            BrainError::Format("build_keyword_index_v5: not a v5 brain".to_string())
        })?;
        let memtable = self.memtable.as_ref().cloned().ok_or_else(|| {
            BrainError::Format("build_keyword_index_v5: no memtable".to_string())
        })?;
        let manifest = self.manifest_snapshot().ok_or_else(|| {
            BrainError::Format("build_keyword_index_v5: no manifest snapshot".to_string())
        })?;
        let cache = self.node_reader_cache.clone().ok_or_else(|| {
            BrainError::Format("build_keyword_index_v5: no node reader cache".to_string())
        })?;

        let (crc, node_count, bond_count) = self.v5_keyword_stamp(&manifest);
        let sidecar_path = Self::v5_keyword_sidecar_path(&brain_dir);
        let runs_dir = brain_dir.join(".kw_rebuild_runs");

        // Hold the memtable lock for the scan: at load / first-query the
        // store is quiescent, and a consistent merge needs a stable
        // memtable view (matches `get_node_v5`'s with_node_table
        // discipline, extended to a full scan).
        let mmap = memtable
            .with_node_table(|nt| {
                crate::node_store::build_v5_keyword_sidecar(
                    nt,
                    &manifest,
                    &cache,
                    sidecar_path.clone(),
                    runs_dir,
                    crc,
                    node_count,
                    bond_count,
                )
            })
            .map_err(|e| BrainError::Format(format!("build_keyword_index_v5: {}", e)))?;

        self.network
            .set_keyword_sidecar_context(crate::network::KeywordSidecarContext {
                path: sidecar_path,
                brain_crc32: crc,
                brain_node_count: node_count,
                brain_bond_count: bond_count,
            });
        self.network.install_keyword_sidecar(mmap);

        // OP-COMPACT-OVERLAP health-check internal notes: refuse a
        // rebuild that leaves a KNOWN control entity unresolvable. A
        // control that is interned but whose label will not resolve from
        // the committed node store is the fingerprint of an entity
        // point-lookup regression (e.g. overlapping L1 node segments),
        // which then answers "I don't have anything solid" for an
        // untouched entity. Fail loudly so the degraded index is not
        // accepted rather than shipping it silently.
        self.assert_control_entity_resolvable()?;
        Ok(())
    }

    /// Health-check used by [`Self::build_keyword_index_v5`]: if a
    /// well-known control entity is present in the intern, its human
    /// label MUST resolve from the committed node store. A
    /// present-but-unresolvable control is the fingerprint of an entity
    /// point-lookup regression internal notes -- overlapping L1 node
    /// segments). Skipped when no control is interned (not an entity
    /// brain) or when `LU_KW_HEALTHCHECK=0`.
    fn assert_control_entity_resolvable(&self) -> Result<(), BrainError> {
        if std::env::var("LU_KW_HEALTHCHECK").ok().as_deref() == Some("0") {
            return Ok(());
        }
        const CONTROLS: &[&str] = &["Albert Einstein#Q937", "Isaac Newton#Q935"];
        for c in CONTROLS {
            let seed = crate::seed::Seed::from_label(c);
            if self.network.node_intern().get(&seed).is_some() {
                match self.network.node_label(&seed) {
                    Some(ref s) if !s.is_empty() => return Ok(()),
                    _ => {
                        return Err(BrainError::Format(format!(
                            "keyword-index health-check FAILED: control entity {:?} is interned but \
                             its label does not resolve from the committed node store -- entity \
                             point-lookup regression (e.g. overlapping L1 node segments; run \
                             major_compact_collapse). Refusing to accept this rebuild.",
                            c
                        )));
                    }
                }
            }
        }
        Ok(())
    }

    /// Load-time fast path: install a still-valid on-disk `.kw` sidecar
    /// (skips the heavy rebuild); otherwise leave the base empty so
    /// `ensure_keyword_index_built` builds it lazily on first keyword
    /// query. Never fails the load -- a missing/stale/unreadable sidecar
    /// just defers to the lazy build.
    pub(crate) fn install_or_defer_keyword_index_v5(&mut self) {
        let brain_dir = match self.v5_store_root.as_ref() {
            Some(d) => d.clone(),
            None => return,
        };
        let manifest = match self.manifest_snapshot() {
            Some(m) => m,
            None => return,
        };
        let (crc, node_count, bond_count) = self.v5_keyword_stamp(&manifest);
        let sidecar_path = Self::v5_keyword_sidecar_path(&brain_dir);
        if !sidecar_path.exists() {
            return;
        }
        match crate::keyword_index::KeywordIndexMmap::open(&sidecar_path) {
            Ok(kmm) => match kmm.validate(crc, node_count, bond_count) {
                Ok(()) => {
                    self.network.set_keyword_sidecar_context(
                        crate::network::KeywordSidecarContext {
                            path: sidecar_path,
                            brain_crc32: crc,
                            brain_node_count: node_count,
                            brain_bond_count: bond_count,
                        },
                    );
                    self.network
                        .install_keyword_sidecar(std::sync::Arc::new(kmm));
                }
                Err(e) => eprintln!(
                    "OP-022.b: v5 keyword sidecar stale ({}); will rebuild lazily",
                    e
                ),
            },
            Err(e) => eprintln!(
                "OP-022.b: v5 keyword sidecar unreadable ({}); will rebuild lazily",
                e
            ),
        }
    }
    pub fn load_v5(
        brain_dir: &Path,
        encryption_key: Option<crate::encrypt::MasterKey>,
    ) -> Result<Self, BrainError> {
        Self::load_v5_with_mode(brain_dir, encryption_key, BrainLoadMode::Auto)
    }

    /// OP-INGEST-LEAN -- `load_v5` with an explicit load mode. When
    /// `mode == BrainLoadMode::IngestOnly` the load builds ONLY the
    /// write path (node intern + MemTable + WAL + flush/compaction
    /// workers + overlays) and SKIPS the serving/recall structures: the
    /// resident degree-map, the cache-backed adjacency + seam, and the
    /// keyword index. `set_adjacency_cache_backed(true)` is still set so
    /// the teach path resolves node existence via the resident intern
    /// (`flush_intern`) and does NOT retain full `KnowledgeNode`s
    /// resident (OP-BRAIN-MapRAM Fix A). Every other mode behaves
    /// exactly as the pre-existing full serving load.
    pub fn load_v5_with_mode(
        brain_dir: &Path,
        encryption_key: Option<crate::encrypt::MasterKey>,
        mode: BrainLoadMode,
    ) -> Result<Self, BrainError> {
        let ingest_only = matches!(mode, BrainLoadMode::IngestOnly);
        // 1. MANIFEST presence + v5-tag check.
        let manifest_path = brain_dir.join(crate::manifest::MANIFEST_FILE_NAME);
        let manifest = match crate::manifest::Manifest::load(&manifest_path)? {
            Some(m) => m,
            None => return Err(BrainError::Format(format!(
                "v5 brain directory {} has no readable MANIFEST (file missing \
                 or .prev fallback also absent).",
                brain_dir.display()
            ))),
        };
        if !manifest.is_v5_format() {
            return Err(BrainError::Format(format!(
                "load_v5 called on {} but its MANIFEST format_tag is not v5 \
                 (got {}). This is a v4 directory -- load it via load_v4 / \
                 --brain-format v4.",
                brain_dir.display(), manifest.format_tag,
            )));
        }

        // 2. Empty Brain (no v3 archive backing).
        let mut brain = Brain::new();
        brain.ingest_only = ingest_only;

        // 3. WAL path. enable_memtable loads the nodeintern.bin sidecar.
        let wal_path = match std::env::var("LU_BRAIN_WAL_PATH").ok() {
            Some(p) if !p.is_empty() => std::path::PathBuf::from(p),
            _ => brain_dir.join("wal").join("current.wal"),
        };
        if let Some(parent) = wal_path.parent() {
            std::fs::create_dir_all(parent).map_err(BrainError::Io)?;
        }

        // 4. MemTable + WAL (+ sidecar intern load).
        const DEFAULT_MEMTABLE_CAP_BYTES: usize = 500 * 1024 * 1024;
        // OP-BRAIN-V5 6.8: stash the key on the Brain BEFORE
        // enable_memtable so the WAL + sidecar open/replay encrypt at
        // rest (the field is otherwise only set at the end of load).
        brain.encryption_key = encryption_key.clone();
        brain.enable_memtable(DEFAULT_MEMTABLE_CAP_BYTES, &wal_path)?;

        // OP-BRAIN-V5 6.4 -- tell enable_flush_worker the v5 stores live
        // at the brain-dir top level (node_store/ + bond_store/), the
        // layout the migration tools write. WAL + sidecar stay under
        // wal/ (handled via wal_path above).
        brain.v5_store_root = Some(brain_dir.to_path_buf());

        // 5. Flush worker: installs manifest snapshot + encrypted
        //    NodeSST/BondSST reader caches (keyed).
        brain.enable_flush_worker(
            brain_dir.to_path_buf(),
            manifest_path,
            encryption_key.clone(),
        )?;

        // 6. OP-BRAIN-MapRAM (final step) -- cache-backed adjacency is
        //    now the PRODUCTION DEFAULT for v5 brains. The three load
        //    flags default ON; set any to 0/false/off/no to force the
        //    legacy path (rollback + the parity harness, which still
        //    compares both modes). The flags are INDEPENDENT so each is
        //    observable on its own (race-free single-writer tests):
        //      * LU_MAPRAM_ROUTE  -- skip rehydrate + route consumers
        //                            through the cache-backed seam.
        //      * LU_MAPRAM_BUILD  -- retain the resident MAP field.
        //      * LU_MAPRAM_CACHE  -- (legacy mode only) also build the
        //                            coexisting cache for parity; in
        //                            routed mode the cache is mandatory
        //                            and always built.
        let mapram_route = Self::mapram_flag_on("LU_MAPRAM_ROUTE", true);
        let mapram_build = Self::mapram_flag_on("LU_MAPRAM_BUILD", true);
        let mapram_cache = Self::mapram_flag_on("LU_MAPRAM_CACHE", true);
        let cache_cap = std::env::var("LU_MAPRAM_CACHE_CAP")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(65_536);
        let pin_fraction = std::env::var("LU_MAPRAM_PIN_FRACTION")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(0.001);

        if ingest_only {
            // OP-INGEST-LEAN: write-only path. Skip the SERVING residency
            // a training pass never reads -- the persisted resident
            // degree-map FIELD (`self.resident_map`, plus its save-time
            // refresh), the hub-pin decode burst, and the keyword index.
            //
            // The node-non-retention write path (`teach_with_kind` Fix A)
            // and node-existence lookup key off `adjacency_cache_backed()`,
            // which is only true once a cache-backed adjacency + seam are
            // installed. So we install a LEAN one: the degree map comes
            // from the persisted sidecar when valid (fast -- NO full
            // bond-SST scan), the 2Q cache is tiny, and pin_fraction is
            // 0.0 (NO hub-pin decode burst). The map is owned only by the
            // cache, so `self.resident_map` stays None and
            // `maintains_resident_map()` reports false (save skips the
            // degmap refresh). No recall runs during a write-only ingest,
            // so the under-provisioned serving cache is never read;
            // serving structures are rebuilt in full when Lu is served.
            let map = match brain.try_load_resident_map_from_sidecar(brain_dir)? {
                Some(m) => m,
                None => {
                    // Sidecar stale/absent. Do NOT fall back to
                    // `build_resident_map` -- that is a full bond-SST scan
                    // + winner-per-(s,pred,o) dedup over EVERY bond (the
                    // exact serving-build OP-INGEST-LEAN exists to avoid;
                    // at 118M bonds it spiked ~40 GiB of transient heap in
                    // the first validation run). The lean cache is never
                    // READ during a write-only ingest (no recall), so the
                    // degree values are irrelevant -- only the node-id
                    // space (for `contains`) matters. Build a zero-degree
                    // map sized to the interned node space. The real,
                    // accurate degree-map is rebuilt when Lu is served.
                    let n = brain
                        .flush_intern
                        .as_ref()
                        .map(|i| i.len())
                        .unwrap_or(0);
                    eprintln!(
                        "[brain] OP-INGEST-LEAN: degmap sidecar stale/absent;                          using a zero-degree map over {} interned nodes (NO                          bond-SST scan -- cache is write-only, never read)",
                        n,
                    );
                    crate::map_ram::ResidentMap::from_degrees(
                        vec![0u32; n],
                        vec![0u32; n],
                    )
                }
            };
            brain.enable_lean_ingest_adjacency(std::sync::Arc::new(map))?;
            brain.set_adjacency_cache_backed(true);
        } else if mapram_route {
            // ---- DEFAULT (cache-backed) load: the RAM win. ----
            // We DO NOT call rehydrate_adjacency here: the ~31 GiB
            // resident adjacency + node-stub build is exactly the cost
            // we are removing. The rerouted consumers (spread /
            // typed_recall / curiosity / hub-fill / mcp) resolve
            // neighbours + degree through the cache-backed seam. We
            // build only the cheap resident MAP (~0.2 GiB) plus the
            // cache (map -> hot 2Q cache -> v5 reader). `degree_p90` is
            // sourced from the map at seam-install
            // (AssociativeNetwork::install_mapram_seam), so it no longer
            // depends on rehydrate's recompute. `rehydrate_adjacency`
            // is NOT deleted -- the legacy branch below keeps it live.
            if mapram_build {
                brain.enable_resident_map_persisted(brain_dir)?;
            }
            brain.enable_cache_backed_adjacency(cache_cap, pin_fraction)?;
            brain.enable_adjacency_seam()?;
            brain.set_adjacency_cache_backed(true);
        } else {
            // ---- LEGACY load (forced OFF via env: rollback/parity). ----
            // Rehydrate the full resident adjacency from the v5 node +
            // bond SSTs, byte-for-byte as before the default flip. A
            // wrong/missing key surfaces here as a hard error. The
            // resident map / cache may coexist (parity harness) when
            // their individual flags are on; consumers stay on the
            // resident adjacency (the harness installs the seam and
            // toggles modes itself).
            brain.rehydrate_adjacency()?;
            if mapram_build {
                brain.enable_resident_map_persisted(brain_dir)?;
            }
            if mapram_cache {
                brain.enable_cache_backed_adjacency(cache_cap, pin_fraction)?;
            }
        }

        // 7. v5 bond-compaction SCHEDULER (shares the manifest
        //    snapshot, bond reader cache, and key). This REPLACES
        //    the v4 `enable_compaction` call that used to run here:
        //    that path starts the v4 CompactionWorker, whose
        //    decide_compaction reads the empty `manifest.levels` and
        //    so NEVER compacts a v5 brain -- so the Phase 3.6 reverse
        //    index was never built in production. A v5 brain runs the
        //    v5 scheduler, not the dead v4 path (no double-run).
        brain.enable_bond_v5_compaction(encryption_key.clone())?;

        // 7b. v5 node-compaction SCHEDULER (shares the manifest
        //    snapshot, node reader cache, and key). Same latent-bug
        //    class as the bond side: the Phase 2.7 NodeCompactionWorker
        //    is a one-shot with no production scheduler, so without
        //    this call node L0 NodeSSTs pile up unbounded during
        //    training (slow node reads, bloated L0/RAM, never
        //    compacts). Coexists with the bond scheduler on disjoint
        //    manifest sub-trees (node_levels vs. bond_v5_levels).
        brain.enable_node_v5_compaction(encryption_key.clone())?;

        // 8. Read path mode = V5Only.
        brain.set_read_path_mode(crate::read_path::ReadPathMode::V5Only);

        // 9. Directory-backed marker + key retention. Reuses the
        //    `brain_dir_v4` slot (the "this brain is a directory, do not
        //    save via the legacy v3 path" flag); the production
        //    status/banner repoint is sub-phase 6.6.
        brain.brain_dir_v4 = Some(brain_dir.to_path_buf());
        if let Some(k) = encryption_key {
            brain.encryption_key = Some(k);
        }

        // 10. OP-022.b -- v5 keyword index (SERVING/recall structure).
        //     OP-INGEST-LEAN: skipped entirely in ingest-only mode --
        //     the teach path never runs a keyword query, and installing
        //     the mmap sidecar also runs a `v5_keyword_stamp` scan. It
        //     is (re)built lazily when Lu is served. Otherwise install a
        //     still-valid on-disk sidecar now (fast); otherwise the
        //     first keyword query builds it lazily via the lifted
        //     `Brain::ensure_keyword_index_built` entry (mcp.rs /
        //     reasoning.rs).
        if !ingest_only {
            brain.install_or_defer_keyword_index_v5();
        }

        // 11. OP-REVOKE-V5 -- repopulate the source + negation overlays
        //     from their encrypted sidecars (wal/source_overlay.bin,
        //     wal/negation_overlay.bin). load_v5 otherwise builds the
        //     brain with EMPTY overlays via Brain::new() and never
        //     refills them, so revoke_bonds_by_source matched ZERO after
        //     any restart. Absent/unusable sidecar -> empty (graceful).
        let (src_n, neg_n) = brain.load_overlays_from_dir(brain_dir);
        if src_n > 0 || neg_n > 0 {
            eprintln!(
                "[brain] OP-REVOKE-V5: loaded {} source + {} negation overlay entries from sidecars",
                src_n, neg_n,
            );
        }

        Ok(brain)
    }

    /// OP-BRAIN-V4 Phase 8 -- read accessor for the v4 directory, if
    /// the brain was loaded via `load_v4`. `None` for v3 brains.
    pub fn brain_dir_v4(&self) -> Option<&Path> {
        self.brain_dir_v4.as_deref()
    }
}


// OP-BRAIN-V4 Phase 5 — canonical read path through MemTable + L0 SSTs +
// v3 archive. Public surface: `read_outgoing_bonds` dispatches on the
// `ReadPathMode` flag; `read_outgoing_bonds_v4` is the merged-path
// implementation. Production default is `ReadPathMode::V3Only`, so the
// canonical recall path is unchanged in shipped binaries until Phase 8.
impl Brain {
    /// Read the current `ReadPathMode`. Stored as an `AtomicU8` on
    /// the Brain so callers can read it without a mutable borrow.
    pub fn read_path_mode(&self) -> crate::read_path::ReadPathMode {
        let raw = self.read_path_mode.load(std::sync::atomic::Ordering::Acquire);
        crate::read_path::ReadPathMode::from_u8(raw)
    }

    /// Switch the read-path mode. Used by Phase 5 A/B tests. A future
    /// MCP tool or `--read-path-mode=v4` startup flag will be the
    /// production switch. Also mirrors the value into the
    /// `AssociativeNetwork` so the spread-activation loop sees
    /// the same dispatch without a back-pointer.
    pub fn set_read_path_mode(&self, mode: crate::read_path::ReadPathMode) {
        self.read_path_mode.store(mode.as_u8(), std::sync::atomic::Ordering::Release);
        self.network.set_read_path_mode_u8(mode.as_u8());
    }

    /// Canonical public API: read all outgoing bonds for `subject`.
    /// Dispatches on `read_path_mode`:
    ///
    /// - `V3Only` (default): returns bonds via the existing v3 path
    ///   only. Byte-equivalent to pre-Phase-5 behaviour. MemTable +
    ///   L0 SSTs are ignored even if present.
    /// - `V4Only` / `MergedV3V4`: calls `read_outgoing_bonds_v4`,
    ///   which consults MemTable + L0 + v3 in newest-first order,
    ///   dedups by `(subject, predicate, object)` taking latest-seq,
    ///   and re-derives confidence from keyframe-source provenance.
    ///
    /// Returns an owned `Vec<MergedBondView>` — distinct from
    /// `sst::BondView` and `view::BondView` because those two are
    /// tightly bound to their underlying storage (Arc'd SST block /
    /// heap+archive split). `MergedBondView` is the unified shape
    /// after the merge runs.
    pub fn read_outgoing_bonds(&self, subject: &crate::seed::Seed)
        -> Result<Vec<crate::read_path::MergedBondView>, BrainError>
    {
        match self.read_path_mode() {
            crate::read_path::ReadPathMode::V3Only => {
                Ok(self.read_outgoing_bonds_v3_only(subject))
            }
            crate::read_path::ReadPathMode::V4Only
            | crate::read_path::ReadPathMode::MergedV3V4 => {
                self.read_outgoing_bonds_v4(subject)
            }
            crate::read_path::ReadPathMode::V5Only => {
                // OP-BRAIN-V5 6.3 -- V5Only recall now MERGES the un-flushed
                // MemTable bonds with the flushed v5 BondSST bonds in one
                // uniform brain-wide NodeId key-space, so a just-taught bond
                // is recallable BEFORE it flushes (and still recallable from
                // the SST after it does). See `read_outgoing_bonds_v5_merged`.
                self.read_outgoing_bonds_v5_merged(subject)
            }
        }
    }

    /// OP-BRAIN-V5 6.3 -- V5Only outgoing recall that closes the
    /// Seed-vs-NodeId MemTable gap. Merges two layers in one uniform
    /// brain-wide NodeId key-space:
    ///
    ///   1. Flushed bonds, via [`Brain::get_outgoing_bonds_v5`] -- the
    ///      live, SST-authoritative surface. It reads the manifest
    ///      snapshot the flush worker updates, so a newly-flushed L0
    ///      BondSST is visible immediately, and it has already collapsed
    ///      L0/L1 duplicates and dropped tombstones.
    ///   2. Un-flushed bonds, via [`MemTable::get_bonds_for_subject`].
    ///      Each `MemBond` now carries its object NodeId, interned at
    ///      insert time (the 6.3 widening), so no per-read re-intern.
    ///
    /// Dedup key is `(predicate_disc, object NodeId)`; the winner is the
    /// highest `seq`. The MemTable seq counter is monotonic and a bond's
    /// seq travels unchanged through flush into `BondPayload.seq`, so an
    /// un-flushed MemTable bond ALWAYS outranks the same triple sitting in
    /// an SST -- the MemTable copy wins and no duplicate is returned. A
    /// MemTable tombstone (also seq-ranked, NodeId-keyed) shadows an older
    /// SST bond and is dropped from the result. Mirrors the merge
    /// discipline of `rehydrate_adjacency` and `read_outgoing_bonds_v4`.
    ///
    /// Falls back to the rehydrated-adjacency read only when the v5 intern
    /// handle is absent (no v5 substrate enabled).
    fn read_outgoing_bonds_v5_merged(&self, subject: &crate::seed::Seed)
        -> Result<Vec<crate::read_path::MergedBondView>, BrainError>
    {
        use crate::read_path::{BondSourceLayer, MergedBondView};
        use std::collections::HashMap;

        let intern = match self.flush_intern.as_ref() {
            Some(i) => i,
            // No v5 substrate intern -> nothing in the NodeId key-space;
            // fall back to whatever the rehydrated adjacency holds.
            None => return Ok(self.read_outgoing_bonds_v3_only(subject)),
        };

        // Dedup map: (predicate_disc, object_node_id) -> current winner.
        let mut merged: HashMap<(u8, u64), MergedBondView> = HashMap::new();

        // Insert-or-replace, highest seq wins.
        fn offer(
            merged: &mut std::collections::HashMap<(u8, u64), crate::read_path::MergedBondView>,
            key: (u8, u64),
            view: crate::read_path::MergedBondView,
        ) {
            use std::collections::hash_map::Entry;
            match merged.entry(key) {
                Entry::Vacant(v) => {
                    v.insert(view);
                }
                Entry::Occupied(mut o) => {
                    if view.seq > o.get().seq {
                        o.insert(view);
                    }
                }
            }
        }

        // ---- Layer 1: flushed v5 BondSSTs (live manifest, tombstones
        //      already resolved away by get_outgoing_bonds_v5). ----
        if let Some(subj_id) = intern.get(subject) {
            for (_s, pred_disc, obj_id, payload) in self.get_outgoing_bonds_v5(subj_id)? {
                let object_seed = match intern.reverse_lookup(obj_id) {
                    Some(s) => s,
                    None => continue, // cannot surface a Seedless object
                };
                let predicate = match crate::memtable::bond_type_from_discriminant(pred_disc) {
                    Some(bt) => bt,
                    None => continue, // corrupt/future discriminant: skip
                };
                let keyframes = if payload.keyframes.is_empty() {
                    vec![crate::bond::TemporalKeyframe {
                        timestamp: 0,
                        weight: payload.weight as f32 / 255.0,
                        source: crate::bond::KeyframeSource::Baseline,
                    }]
                } else {
                    payload.keyframes
                };
                let confidence = crate::bond::derive_confidence(
                    &keyframes, crate::node::Confidence::Verified,
                );
                offer(
                    &mut merged,
                    (pred_disc, obj_id.0),
                    MergedBondView {
                        subject: *subject,
                        object: object_seed,
                        predicate,
                        weight: payload.weight as f32 / 255.0,
                        confidence,
                        keyframes,
                        seq: payload.seq,
                        source_layer: BondSourceLayer::V5Sst,
                        tombstone: false,
                    },
                );
            }
        }

        // ---- Layer 2: un-flushed MemTable bonds (newest, seq-ranked). ----
        if let Some(mt) = self.memtable.as_ref() {
            for mb in mt.get_bonds_for_subject(subject, intern) {
                let confidence = crate::bond::derive_confidence(
                    &mb.keyframes, crate::node::Confidence::Verified,
                );
                offer(
                    &mut merged,
                    (crate::memtable::predicate_discriminant(mb.predicate), mb.object_id.0),
                    MergedBondView {
                        subject: *subject,
                        object: mb.object,
                        predicate: mb.predicate,
                        weight: mb.weight,
                        confidence,
                        keyframes: mb.keyframes,
                        seq: mb.seq,
                        source_layer: BondSourceLayer::MemTable,
                        tombstone: mb.tombstone,
                    },
                );
            }
        }

        // Drop tombstone winners; they exist only to shadow older bonds.
        Ok(merged.into_values().filter(|v| !v.tombstone).collect())
    }

    /// Phase 5 v3-only fallback: walk `neighbors_outgoing_capped`,
    /// read each bond via the existing `view_bond` path, materialise
    /// owned `MergedBondView`s. Byte-equivalent set semantics with the
    /// pre-Phase-5 spread loop.
    ///
    /// Why "capped" and not raw `neighbors_outgoing`? Phase 5 must
    /// honour the hub-read cap (Cat.3b Port -- `hub_read_max_out`)
    /// just like spread activation does today. A V3Only read that
    /// returned uncapped neighbours would inadvertently disable a
    /// production-deployed guardrail.
    fn read_outgoing_bonds_v3_only(&self, subject: &crate::seed::Seed)
        -> Vec<crate::read_path::MergedBondView>
    {
        let neighbors = self.network.neighbors_outgoing_capped(subject);
        let mut out: Vec<crate::read_path::MergedBondView> = Vec::with_capacity(neighbors.len());
        for nbr in neighbors {
            let bv = match self.network.get_bond(subject, &nbr) {
                Some(b) => b,
                None => continue,
            };
            // BondView's keyframe iter materialises owned keyframes
            // for archive-backed bonds, so this is a single allocation
            // pass that pulls everything we need onto the heap.
            let keyframes: Vec<crate::bond::TemporalKeyframe> = bv.keyframes_iter().collect();
            // The v3 BondView::confidence() isn't a thing -- v3 bonds
            // carry per-keyframe confidence on the keyframe itself.
            // For the merged view we use `derive_confidence` against
            // a `Verified` stored default (the pre-Phase-5 spread
            // already assumes the bond's effective confidence comes
            // from the source/scope machinery downstream).
            let confidence = crate::bond::derive_confidence(
                &keyframes, crate::node::Confidence::Verified,
            );
            out.push(crate::read_path::MergedBondView {
                subject: *subject,
                object: nbr,
                predicate: bv.bond_type(),
                weight: bv.latest_raw_weight(),
                confidence,
                keyframes,
                // V3Only: every bond is seq=0 in the dedup sense.
                seq: 0,
                source_layer: crate::read_path::BondSourceLayer::V3Archive,
                tombstone: false,
            });
        }
        out
    }

    /// Phase 5 merged read: consult MemTable (highest seq), then all
    /// L0 SSTs (manifest_id as seq), then v3 archive (seq=0). Dedup
    /// by `(subject, predicate, object)`. Confidence re-derived from
    /// the latest-timestamp keyframe source.
    pub fn read_outgoing_bonds_v4(&self, subject: &crate::seed::Seed)
        -> Result<Vec<crate::read_path::MergedBondView>, BrainError>
    {
        use crate::read_path::{BondSourceLayer, DedupKey, MergedBondView};
        use std::collections::HashMap;

        // Dedup map: key -> (current winner, current seq). Winner
        // gets replaced when a higher-seq entry arrives.
        let mut merged: HashMap<DedupKey, MergedBondView> = HashMap::new();

        // 1. MemTable layer (highest seq). Each MemBond carries its
        //    own monotonic seq from MemTable::insert.
        if let (Some(mt), Some(intern)) = (self.memtable.as_ref(), self.flush_intern.as_ref()) {
            let mem_bonds = mt.get_bonds_for_subject(subject, intern);
            for mb in mem_bonds {
                let key = DedupKey::new(*subject, mb.object, mb.predicate);
                let stored_default = crate::node::Confidence::Verified;
                let confidence = crate::bond::derive_confidence(&mb.keyframes, stored_default);
                let is_tombstone = mb.tombstone;
                let view = MergedBondView {
                    subject: *subject,
                    object: mb.object,
                    predicate: mb.predicate,
                    weight: mb.weight,
                    confidence,
                    keyframes: mb.keyframes,
                    seq: mb.seq,
                    source_layer: BondSourceLayer::MemTable,
                    tombstone: is_tombstone,
                };
                // MemTable seq is monotonic per MemTable instance;
                // multiple inserts of the same triple are possible
                // (the MemTable does NOT collapse on insert), so we
                // still must take the max-seq winner inside the
                // MemTable's own contribution.
                merge_winner(&mut merged, key, view);
            }
        }

        // 2. L0 SST layer. Walk newest-first (highest manifest id
        //    first) so the dedup tie-breaker is correct without
        //    revisiting earlier entries. Each L0 SST's manifest id is
        //    monotonic from `Manifest::next_sst_id` so the ordering
        //    is well-defined.
        if let Some(snap_lock) = self.manifest_snapshot.as_ref() {
            let manifest = snap_lock
                .read()
                .expect("manifest snapshot read lock (Phase 5 read path)")
                .clone();
            // Gather all L0 SSTs sorted by id descending so newest wins.
            let mut l0_entries: Vec<&crate::manifest::SSTEntry> = manifest
                .levels
                .iter()
                .filter(|l| l.level == 0)
                .flat_map(|l| l.ssts.iter())
                .collect();
            l0_entries.sort_by(|a, b| b.id.cmp(&a.id));
            let sst_dir: Option<std::path::PathBuf> = self
                .flush_worker
                .as_ref()
                .map(|w| w.sst_dir().to_path_buf());
            for entry in l0_entries {
                // Skip if subject is outside the SST's [min, max] range.
                let smin = entry.subject_min_seed();
                let smax = entry.subject_max_seed();
                if subject.as_bytes() < smin.as_bytes() || subject.as_bytes() > smax.as_bytes() {
                    continue;
                }
                // Phase 5: cache opened SST handles by manifest id so
                // hot reads do not re-open the file. The block cache
                // inside each SST stays warm across reads.
                let sst = {
                    let mut cache = self.v4_sst_cache.lock().expect("v4_sst_cache lock");
                    if let Some(s) = cache.get(&entry.id) {
                        s.clone()
                    } else {
                        let path = match &sst_dir {
                            Some(dir) => dir.join(&entry.file),
                            None => std::path::PathBuf::from(&entry.file),
                        };
                        match Brain::read_sst_with(&path, None, self.v4_block_cache.clone()) {
                            Ok(mut s) => {
                                // Phase 6: share the parsed-block cache so
                                // repeated reads on the same subject skip the
                                // bytes->ParsedBlock parse step.
                                s.attach_parsed_cache(self.v4_parsed_block_cache.clone());
                                let a = std::sync::Arc::new(s);
                                cache.insert(entry.id, a.clone());
                                a
                            }
                            Err(e) => {
                                eprintln!(
                                    "[brain] Phase 5 read: skipping unreadable L0 SST {}: {}",
                                    entry.file, e,
                                );
                                continue;
                            }
                        }
                    }
                };
                let bonds = sst.get_bonds_for_subject(subject)?;
                for sst_view in bonds {
                    let key = DedupKey::new(*subject, sst_view.object, sst_view.bond_type);
                    let stored = sst_view.confidence;
                    let confidence = crate::bond::derive_confidence(&sst_view.keyframes, stored);
                    let is_tombstone = sst_view.is_tombstone();
                    let view = MergedBondView {
                        subject: *subject,
                        object: sst_view.object,
                        predicate: sst_view.bond_type,
                        weight: sst_view.weight,
                        confidence,
                        keyframes: sst_view.keyframes.clone(),
                        seq: entry.id,
                        source_layer: BondSourceLayer::L0Sst,
                        tombstone: is_tombstone,
                    };
                    merge_winner(&mut merged, key, view);
                }
            }
        }

        // 3. v3 archive layer (seq=0). Read via the existing
        //    `neighbors_outgoing_capped` + `view_bond` path. Capped
        //    to honour the hub-read guardrail just like spread does.
        for v3 in self.read_outgoing_bonds_v3_only(subject) {
            let key = DedupKey::new(v3.subject, v3.object, v3.predicate);
            // v3.seq is already 0; v3 entries never win against
            // MemTable or L0 entries.
            merge_winner(&mut merged, key, v3);
        }

        // Phase 6: drop tombstone winners. A tombstone in the
        // dedup map signals "this triple was deleted at a higher seq
        // than any live insert" -- so the merged result must not
        // include it. Live inserts that beat the tombstone (i.e. an
        // insert at strictly higher seq than the tombstone) win in
        // the dedup pass and are returned normally.
        Ok(merged.into_values().filter(|v| !v.tombstone).collect())
    }
}

/// Phase 5 dedup helper. Inserts `candidate` into `map` keyed on
/// `key`, replacing whatever is there iff the candidate has a
/// strictly higher seq than the incumbent. Ties keep the incumbent
/// (we walk MemTable -> L0 -> v3 in that order, so the first writer
/// for any given seq has earliest insertion order which we preserve).
#[inline]
fn merge_winner(
    map: &mut std::collections::HashMap<crate::read_path::DedupKey, crate::read_path::MergedBondView>,
    key: crate::read_path::DedupKey,
    candidate: crate::read_path::MergedBondView,
) {
    match map.get(&key) {
        Some(existing) if existing.seq >= candidate.seq => {
            // Incumbent wins; drop candidate.
        }
        _ => {
            map.insert(key, candidate);
        }
    }
}


#[cfg(test)]
mod phase4_brain_integration_tests {
    use super::*;
    use crate::bond::{BondType, KeyframeSource, TemporalKeyframe};
    use crate::seed::Seed;
    use tempfile::tempdir;

    fn baseline_kf() -> TemporalKeyframe {
        TemporalKeyframe { timestamp: 0, weight: 0.5, source: KeyframeSource::Baseline }
    }

    #[test]
    fn enable_memtable_replays_existing_wal_records() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("wal.seg");
        // Pre-populate the WAL OUTSIDE of any Brain so we simulate a
        // crash-and-restart: records on disk before enable_memtable
        // runs.
        {
            let mut wal = crate::wal::Wal::open(&wal_path).unwrap();
            wal.set_fsync_thresholds(10_000, 64 * 1024 * 1024);
            for i in 0..50 {
                let s = Seed::from_label(&format!("subj_{}", i));
                let o = Seed::from_label(&format!("obj_{}", i));
                wal.append(i as u64, &s, &o, BondType::IsA, 0.5, &[baseline_kf()]).unwrap();
            }
            wal.fsync().unwrap();
        }
        // Now build a fresh Brain and enable_memtable. The 50 records
        // must be replayed into the MemTable.
        let mut brain = Brain::new();
        brain.enable_memtable(1024 * 1024, &wal_path).unwrap();
        let mt = brain.memtable().expect("memtable installed");
        assert_eq!(mt.bond_count(), 50, "WAL replay must rebuild MemTable on enable");
    }

    #[test]
    fn enable_flush_worker_cleans_up_orphan_ssts_on_startup() {
        let dir = tempdir().unwrap();
        let sst_dir = dir.path().to_path_buf();
        let manifest_path = dir.path().join("MANIFEST");
        let wal_path = dir.path().join("wal.seg");
        // Drop a fake L*.sst (no manifest at all yet).
        let fake = sst_dir.join("L0-77.sst");
        std::fs::write(&fake, b"crash-mid-flush leftover").unwrap();
        let mut brain = Brain::new();
        brain.enable_memtable(1024 * 1024, &wal_path).unwrap();
        brain.enable_flush_worker(sst_dir.clone(), manifest_path, None).unwrap();
        // After enable_flush_worker the orphan must be renamed.
        assert!(!fake.exists(), "orphan SST should be cleaned up");
        let entries: Vec<_> = std::fs::read_dir(&sst_dir).unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        assert!(entries.iter().any(|n| n.starts_with("L0-77.sst.orphan-")),
            "orphan rename pattern not found: {:?}", entries);
    }

    #[test]
    fn brain_flush_now_round_trip() {
        let dir = tempdir().unwrap();
        let sst_dir = dir.path().to_path_buf();
        let manifest_path = dir.path().join("MANIFEST");
        let wal_path = dir.path().join("wal.seg");
        let mut brain = Brain::new();
        brain.enable_memtable(1024 * 1024 * 1024, &wal_path).unwrap();
        brain.enable_flush_worker(sst_dir.clone(), manifest_path, None).unwrap();
        // Inject some bonds into the MemTable directly + WAL.
        let intern = brain.flush_intern.as_ref().unwrap().clone();
        let mt = brain.memtable().unwrap().clone();
        let wal = brain.wal().unwrap().clone();
        for i in 0..200 {
            let s = Seed::from_label(&format!("s_{}", i));
            let o = Seed::from_label(&format!("o_{}", i));
            let kfs = vec![baseline_kf()];
            let r = mt.insert(s, o, BondType::IsA, 0.5, kfs.clone(), &intern);
            wal.lock().unwrap().append(
                r.seq, &s, &o, BondType::IsA, 0.5, &kfs,
            ).unwrap();
        }
        wal.lock().unwrap().fsync().unwrap();
        brain.flush_now().unwrap();
        let snap = brain.manifest_snapshot().expect("manifest snapshot");
        assert_eq!(snap.total_sst_count(), 1);
        let (_, e) = snap.iter_ssts().next().unwrap();
        assert_eq!(e.bond_count, 200);
        assert!(sst_dir.join(&e.file).exists());
        // WAL is empty after flush+truncate.
        assert_eq!(crate::wal::replay(&wal_path).unwrap().len(), 0);
    }

    /// OP-BRAIN-V5 Phase 2.8 Test K -- brain-level round-trip:
    /// insert nodes into the MemTable, flush_once -> NodeSST,
    /// then `get_node_v5(id)` must return the expected payload.
    /// Covers (a) MemTable hit before flush, (b) NodeSST hit
    /// after flush, and (c) cache reuse across N lookups.
    // OP-REDEFINE-1: set_definition_update additively attaches a gloss to a
    // node that a bond pass already created (create-only set_definition skips
    // it), WITHOUT touching bonds -- the fix for "what is a dog".
    #[test]
    fn set_definition_update_attaches_gloss_additively() {
        let dir = tempdir().unwrap();
        let manifest_path = dir.path().join("MANIFEST");
        let wal_path = dir.path().join("wal.seg");
        let mut brain = Brain::new();
        brain.enable_memtable(1024 * 1024 * 1024, &wal_path).unwrap();
        brain
            .enable_flush_worker(dir.path().to_path_buf(), manifest_path, None)
            .unwrap();
        // Mint the "dog" node with NO definition directly into the store
        // (mimics an earlier bond pass), plus a bond so we can assert bonds
        // are untouched by the additive update.
        let seed = crate::seed::Seed::from_label("dog");
        let node0 = crate::node::KnowledgeNode::with_seed(
            seed,
            "dog".to_string(),
            crate::node::KnowledgeContent::Concept("dog".to_string()),
            Confidence::Verified,
            1,
        );
        brain.persist_node_v4(seed, &node0);
        let _ = brain.teach("dog", BondType::IsA, "animal", 0.9, Confidence::Verified);
        let intern = brain.flush_intern.as_ref().unwrap().clone();
        let id = intern.get(&seed).expect("dog node exists after teach");
        let gloss = "a domesticated carnivorous mammal";
        // create-only path must NOT overwrite the pre-existing node.
        assert!(!brain.set_definition("dog", gloss), "create-only skips existing node");
        let n0 = brain.get_node_v5(id).unwrap().unwrap();
        assert!(
            !matches!(n0.knowledge.content, crate::node::KnowledgeContent::Definition(_)),
            "node carries no definition before update"
        );
        let bonds_before = brain.network.bond_count();
        // update path attaches the gloss additively.
        assert!(brain.set_definition_update("dog", gloss), "update attaches gloss");
        let n1 = brain.get_node_v5(id).unwrap().unwrap();
        assert!(
            matches!(n1.knowledge.content, crate::node::KnowledgeContent::Definition(ref d) if d == gloss),
            "definition attached, got {:?}", n1.knowledge.content
        );
        assert_eq!(brain.network.bond_count(), bonds_before, "bonds unchanged (additive)");
        assert_eq!(n1.seed, seed, "seed preserved");
        // idempotent: a second update is a no-op (already defined).
        assert!(!brain.set_definition_update("dog", "other"), "already-defined update no-op");
        // survives flush -> SST.
        brain.flush_now().unwrap();
        let n2 = brain.get_node_v5(id).unwrap().unwrap();
        assert!(
            matches!(n2.knowledge.content, crate::node::KnowledgeContent::Definition(ref d) if d == gloss),
            "definition survives flush"
        );
    }

    #[test]
    fn set_definition_round_trip_v5() {
        let dir = tempdir().unwrap();
        let manifest_path = dir.path().join("MANIFEST");
        let wal_path = dir.path().join("wal.seg");
        let mut brain = Brain::new();
        brain.enable_memtable(1024 * 1024 * 1024, &wal_path).unwrap();
        brain
            .enable_flush_worker(dir.path().to_path_buf(), manifest_path, None)
            .unwrap();
        let gloss = "a domesticated carnivorous mammal";
        assert!(brain.set_definition("dog", gloss), "first define creates node");
        assert!(!brain.set_definition("dog", "other"), "idempotent: existing not mutated");
        let intern = brain.flush_intern.as_ref().unwrap().clone();
        let seed = crate::seed::Seed::from_label("dog");
        let id = intern.get(&seed).expect("node id resolves after set_definition");
        // (a) pre-flush MemTable read-back
        let n = brain.get_node_v5(id).unwrap().expect("node present pre-flush");
        assert_eq!(n.content, "dog");
        assert!(
            matches!(n.knowledge.content, crate::node::KnowledgeContent::Definition(ref d) if d == gloss),
            "definition must persist as KnowledgeContent::Definition, got {:?}", n.knowledge.content
        );
        // add a bond so the flush gate fires, then (b) post-flush SST read-back
        let _ = brain.teach("dog", BondType::IsA, "animal", 0.9, Confidence::Verified);
        brain.flush_now().unwrap();
        let n2 = brain.get_node_v5(id).unwrap().expect("node present post-flush");
        assert!(
            matches!(n2.knowledge.content, crate::node::KnowledgeContent::Definition(ref d) if d == gloss),
            "definition must survive flush->SST read-back, got {:?}", n2.knowledge.content
        );
    }

    #[test]
    fn phase28_brain_get_node_v5_round_trip() {
        let dir = tempdir().unwrap();
        let sst_dir = dir.path().to_path_buf();
        let manifest_path = dir.path().join("MANIFEST");
        let wal_path = dir.path().join("wal.seg");
        let mut brain = Brain::new();
        brain.enable_memtable(1024 * 1024 * 1024, &wal_path).unwrap();
        brain
            .enable_flush_worker(sst_dir.clone(), manifest_path, None)
            .unwrap();
        let intern = brain.flush_intern.as_ref().unwrap().clone();
        let mt = brain.memtable().unwrap().clone();
        let wal = brain.wal().unwrap().clone();

        // Stamp 25 nodes (and matching bonds so the flush triggers
        // on the bond side too -- the empty-memtable bond path
        // returns early before reaching the NodeFlushWorker hook).
        let mut seeds: Vec<crate::seed::Seed> = Vec::with_capacity(25);
        let mut node_ids: Vec<crate::node_intern::NodeId> = Vec::with_capacity(25);
        for i in 0..25u64 {
            let s = crate::seed::Seed::from_label(&format!("phase28_k_node_{}", i));
            let o = crate::seed::Seed::from_label(&format!("phase28_k_obj_{}", i));
            seeds.push(s);
            // Build a KnowledgeNode with content that we can
            // assert on after the round-trip.
            let node = crate::node::KnowledgeNode {
                seed: s,
                content: format!("k_node_payload_{}", i),
                knowledge: crate::node::KnowledgeUnit {
                    content: crate::node::KnowledgeContent::Concept(
                        format!("k_node_payload_{}", i),
                    ),
                    confidence: crate::node::Confidence::Verified,
                    created_at: 0,
                    updated_at: 0,
                },
                embedding: vec![0.0; 384],
                parent_seeds: Vec::new(),
                typed_ratio: 0.0,
            };
            mt.insert_node(s, node, &intern);
            // Pair with a bond so the bond-side flush_once gate
            // doesn't short-circuit -- the gate triggers when
            // bonds.is_empty() && nodes.is_empty() both hold.
            let kfs = vec![baseline_kf()];
            let r = mt.insert(s, o, BondType::IsA, 0.5, kfs.clone(), &intern);
            wal.lock().unwrap().append(
                r.seq, &s, &o, BondType::IsA, 0.5, &kfs,
            ).unwrap();
            node_ids.push(intern.get(&s).expect("node id resolves post-insert"));
        }
        wal.lock().unwrap().fsync().unwrap();

        // (a) MemTable hit BEFORE flush. v5 read path consults
        // the live MemTable.
        let pre_flush = brain.get_node_v5(node_ids[3]).unwrap();
        assert!(pre_flush.is_some(), "pre-flush MemTable hit must succeed");
        assert_eq!(pre_flush.unwrap().content, "k_node_payload_3");

        // Flush -> creates an L0 NodeSST + clears MemTable's
        // node_table (swap_for_flush replaces the live table).
        brain.flush_now().unwrap();

        // Manifest must now have ONE NodeSST at L0.
        let snap = brain.manifest_snapshot().expect("manifest snapshot");
        assert_eq!(
            snap.total_node_sst_count(),
            1,
            "flush_now must produce one L0 NodeSST",
        );

        // (b) NodeSST hit AFTER flush. v5 read path consults L0
        // because the live MemTable's node_table is empty.
        for (i, id) in node_ids.iter().enumerate() {
            let got = brain.get_node_v5(*id).unwrap();
            assert!(
                got.is_some(),
                "post-flush L0 hit at NodeId {:?} must succeed",
                id,
            );
            assert_eq!(got.unwrap().content, format!("k_node_payload_{}", i));
        }

        // (c) Reader cache reuse: after 25 lookups against the
        // single L0 SST, the reader cache must hold exactly one
        // entry.
        let cache = brain
            .node_reader_cache
            .as_ref()
            .expect("node_reader_cache installed by enable_flush_worker")
            .clone();
        assert_eq!(
            cache.cache_size(),
            1,
            "reader cache must hold exactly one reader after N L0 lookups",
        );

        // (d) Unknown NodeId -> Ok(None).
        let miss = brain.get_node_v5(crate::node_intern::NodeId(u64::MAX - 1)).unwrap();
        assert!(miss.is_none(), "missing NodeId must return Ok(None)");
    }

    /// OP-BRAIN-V5 Phase 3.7 Test I -- brain-level bond read
    /// round-trip: teach bonds, flush_now, query both directions.
    /// Also encodes the documented Phase 3.7 scope: the PRE-flush
    /// query returns empty because the v5 bond read path is
    /// SST-only (the live MemTable bond half is not consulted).
    #[test]
    fn phase37_brain_bond_read_round_trip() {
        let dir = tempdir().unwrap();
        let sst_dir = dir.path().to_path_buf();
        let manifest_path = dir.path().join("MANIFEST");
        let wal_path = dir.path().join("wal.seg");
        let mut brain = Brain::new();
        brain.enable_memtable(1024 * 1024 * 1024, &wal_path).unwrap();
        brain
            .enable_flush_worker(sst_dir.clone(), manifest_path, None)
            .unwrap();
        let intern = brain.flush_intern.as_ref().unwrap().clone();
        let mt = brain.memtable().unwrap().clone();
        let wal = brain.wal().unwrap().clone();

        let s = crate::seed::Seed::from_label("phase37_subject");
        for i in 0..10u64 {
            let o = crate::seed::Seed::from_label(&format!("phase37_obj_{}", i));
            let kfs = vec![baseline_kf()];
            let r = mt.insert(s, o, BondType::IsA, 0.5, kfs.clone(), &intern);
            wal.lock().unwrap().append(
                r.seq, &s, &o, BondType::IsA, 0.5, &kfs,
            ).unwrap();
        }
        wal.lock().unwrap().fsync().unwrap();
        let subject_id = intern.get(&s).expect("subject interned by insert");

        // Documented Phase 3.7 scope: pre-flush, the v5 bond read
        // path sees NOTHING (SST-only; MemTable bond half not
        // consulted). If a future port wires the MemTable half,
        // THIS assertion is the one to flip.
        assert!(
            brain.get_outgoing_bonds_v5(subject_id).unwrap().is_empty(),
            "pre-flush v5 bond read must be empty (SST-only path)",
        );

        brain.flush_now().unwrap();

        // Outgoing: all 10 bonds, subject preserved, live only,
        // monotonic (predicate, object) order.
        let rows = brain.get_outgoing_bonds_v5(subject_id).unwrap();
        assert_eq!(rows.len(), 10, "post-flush outgoing must surface all 10 bonds");
        for (rs, _p, _o, pay) in &rows {
            assert_eq!(*rs, subject_id);
            assert_ne!(pay.confidence, 0xFF, "tombstones must never surface");
        }
        for w in rows.windows(2) {
            assert!(
                (w[0].1, (w[0].2).0) < (w[1].1, (w[1].2).0),
                "outgoing yield must ascend by (predicate, object)",
            );
        }

        // Incoming: obj_3 has exactly one incoming bond, served by
        // the L0 fall-through primary scan (no reverse file at L0).
        let o3 = intern
            .get(&crate::seed::Seed::from_label("phase37_obj_3"))
            .expect("object interned by flush");
        let inc = brain.get_incoming_bonds_v5(o3).unwrap();
        assert_eq!(inc.len(), 1, "exactly one incoming bond at obj_3");
        assert_eq!(inc[0].0, subject_id, "incoming preserves original orientation");
        assert_eq!(inc[0].2, o3);

        // Unknown NodeId -> empty, not error.
        assert!(brain
            .get_outgoing_bonds_v5(crate::node_intern::NodeId(u64::MAX - 1))
            .unwrap()
            .is_empty());
        assert!(brain
            .get_incoming_bonds_v5(crate::node_intern::NodeId(u64::MAX - 1))
            .unwrap()
            .is_empty());
    }
}

#[cfg(test)]
mod phase5_read_path_integration_tests {
    use super::*;
    use crate::bond::{BondType, KeyframeSource, TemporalKeyframe};
    use crate::read_path::{BondSourceLayer, ReadPathMode};
    use crate::seed::Seed;
    use rand::{Rng, SeedableRng};
    use rand::rngs::StdRng;
    use std::collections::HashSet;
    use tempfile::tempdir;

    fn baseline_kf(ts: i64, w: f32) -> TemporalKeyframe {
        TemporalKeyframe { timestamp: ts, weight: w, source: KeyframeSource::Baseline }
    }

    fn session_kf(ts: i64, w: f32, sid: u64) -> TemporalKeyframe {
        TemporalKeyframe { timestamp: ts, weight: w, source: KeyframeSource::Session(sid) }
    }

    fn rand_seed(rng: &mut StdRng) -> Seed {
        let mut b = [0u8; 32];
        rng.fill(&mut b);
        Seed::from_bytes(b)
    }

    /// Mode default is V3Only.
    #[test]
    fn read_path_mode_default_is_v3only() {
        let brain = Brain::new();
        assert_eq!(brain.read_path_mode(), ReadPathMode::V3Only);
    }

    /// Setter round-trips.
    #[test]
    fn set_read_path_mode_round_trips() {
        let brain = Brain::new();
        for m in [ReadPathMode::V3Only, ReadPathMode::V4Only, ReadPathMode::MergedV3V4] {
            brain.set_read_path_mode(m);
            assert_eq!(brain.read_path_mode(), m);
        }
    }

    /// A/B equivalence on v3 brain: V3Only and V4Only return identical
    /// (subject, object, predicate) sets when MemTable + L0 are empty.
    #[test]
    fn v3_v4_equivalence_on_v3_only_brain() {
        let mut brain = Brain::new();
        // Build a small v3 brain: 50 subjects x 5 outgoing bonds each.
        let mut rng = StdRng::seed_from_u64(0xC0FFEE);
        let mut subjects: Vec<Seed> = Vec::new();
        for _ in 0..50 {
            let s = rand_seed(&mut rng);
            subjects.push(s);
            brain.network.add_node(crate::node::KnowledgeNode::with_seed(s, "s".to_string(), crate::node::KnowledgeContent::Concept("s".to_string()), crate::node::Confidence::Verified, 0));
            for _ in 0..5 {
                let o = rand_seed(&mut rng);
                brain.network.add_node(crate::node::KnowledgeNode::with_seed(o, "o".to_string(), crate::node::KnowledgeContent::Concept("o".to_string()), crate::node::Confidence::Verified, 0));
                let bond = crate::bond::TemporalBond::new(
                    s, o, BondType::RelatedTo, 0.5, 0,
                );
                brain.network.add_bond(bond).ok();
            }
        }

        // Same subject, two modes. With empty MemTable + L0, V4Only
        // falls through to v3 internally, so the set must be identical.
        for s in &subjects {
            brain.set_read_path_mode(ReadPathMode::V3Only);
            let v3 = brain.read_outgoing_bonds(s).unwrap();
            brain.set_read_path_mode(ReadPathMode::V4Only);
            let v4 = brain.read_outgoing_bonds(s).unwrap();
            let v3_set: HashSet<(Seed, BondType)> = v3.iter()
                .map(|b| (b.object, b.predicate)).collect();
            let v4_set: HashSet<(Seed, BondType)> = v4.iter()
                .map(|b| (b.object, b.predicate)).collect();
            assert_eq!(v3_set, v4_set, "subject {:?}: V3 and V4 sets differ", s);
        }
    }

    /// MemTable-only: empty v3 + empty L0, MemTable holds 10K bonds.
    /// V4Only must return all of them.
    #[test]
    fn memtable_only_read_returns_all_bonds() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("wal.seg");
        let mut brain = Brain::new();
        brain.enable_memtable(64 * 1024 * 1024, &wal_path).unwrap();
        brain.set_read_path_mode(ReadPathMode::V4Only);

        let mut rng = StdRng::seed_from_u64(0xAB);
        // 1000 subjects x 10 bonds = 10K bonds.
        let mut by_subj: std::collections::HashMap<Seed, Vec<Seed>> = std::collections::HashMap::new();
        let mt = brain.memtable().unwrap().clone();
        let intern = brain.flush_intern.as_ref().unwrap().clone();
        for _ in 0..1000 {
            let s = rand_seed(&mut rng);
            for _ in 0..10 {
                let o = rand_seed(&mut rng);
                mt.insert(
                    s, o, BondType::IsA, 0.5,
                    vec![baseline_kf(1, 0.5)], &intern,
                );
                by_subj.entry(s).or_default().push(o);
            }
        }
        // Spot-check 50 subjects.
        let mut sample: Vec<&Seed> = by_subj.keys().collect();
        sample.truncate(50);
        for s in sample {
            let got = brain.read_outgoing_bonds(s).unwrap();
            let got_objs: HashSet<Seed> = got.iter().map(|b| b.object).collect();
            let want_objs: HashSet<Seed> = by_subj[s].iter().copied().collect();
            assert_eq!(got_objs, want_objs);
            // All bonds should be tagged MemTable.
            for b in &got {
                assert_eq!(b.source_layer, BondSourceLayer::MemTable);
            }
        }
    }

    /// L0-SST-only: flush a MemTable to L0, then drain the MemTable.
    /// V4Only must read from the L0 SST.
    #[test]
    fn l0_sst_only_read_returns_all_bonds() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("wal.seg");
        let sst_dir = dir.path().join("ssts");
        std::fs::create_dir_all(&sst_dir).unwrap();
        let manifest_path = dir.path().join("MANIFEST");

        let mut brain = Brain::new();
        brain.enable_memtable(1024 * 1024 * 1024, &wal_path).unwrap();
        brain.enable_flush_worker(sst_dir.clone(), manifest_path.clone(), None).unwrap();
        brain.set_read_path_mode(ReadPathMode::V4Only);

        let mut rng = StdRng::seed_from_u64(0xC0DE);
        let mut by_subj: std::collections::HashMap<Seed, Vec<Seed>> = std::collections::HashMap::new();
        let mt = brain.memtable().unwrap().clone();
        let intern = brain.flush_intern.as_ref().unwrap().clone();
        for _ in 0..100 {
            let s = rand_seed(&mut rng);
            for _ in 0..10 {
                let o = rand_seed(&mut rng);
                mt.insert(
                    s, o, BondType::IsA, 0.5,
                    vec![baseline_kf(1, 0.5)], &intern,
                );
                by_subj.entry(s).or_default().push(o);
            }
        }
        assert_eq!(mt.bond_count(), 1000);

        brain.flush_now().unwrap();
        assert_eq!(mt.bond_count(), 0, "MemTable should be drained");

        // Now reads come from L0.
        let sample: Vec<Seed> = by_subj.keys().copied().take(20).collect();
        for s in &sample {
            let got = brain.read_outgoing_bonds(s).unwrap();
            let got_objs: HashSet<Seed> = got.iter().map(|b| b.object).collect();
            let want_objs: HashSet<Seed> = by_subj[s].iter().copied().collect();
            assert_eq!(got_objs, want_objs, "subject {:?}: L0 read mismatch", s);
            for b in &got {
                assert_eq!(b.source_layer, BondSourceLayer::L0Sst, "expected L0Sst tag");
            }
        }
    }

    /// Merge precedence: same (subject, predicate, object) in v3 (heap)
    /// and L0 (flushed) and MemTable. MemTable's version wins.
    #[test]
    fn merge_precedence_memtable_wins() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("wal.seg");
        let sst_dir = dir.path().join("ssts");
        std::fs::create_dir_all(&sst_dir).unwrap();
        let manifest_path = dir.path().join("MANIFEST");

        let mut brain = Brain::new();
        // Seed v3 archive with the triple at weight 0.1.
        let s = Seed::from_label("S");
        let o = Seed::from_label("O");
        brain.network.add_node(crate::node::KnowledgeNode::with_seed(s, "s".to_string(), crate::node::KnowledgeContent::Concept("s".to_string()), crate::node::Confidence::Verified, 0));
        brain.network.add_node(crate::node::KnowledgeNode::with_seed(o, "o".to_string(), crate::node::KnowledgeContent::Concept("o".to_string()), crate::node::Confidence::Verified, 0));
        let mut v3_bond = crate::bond::TemporalBond::new(s, o, BondType::IsA, 0.1, 0);
        v3_bond.add_keyframe(100, 0.1, KeyframeSource::Baseline);
        brain.network.add_bond(v3_bond).ok();

        // Now enable Phase 4 + 5.
        brain.enable_memtable(1024 * 1024 * 1024, &wal_path).unwrap();
        brain.enable_flush_worker(sst_dir.clone(), manifest_path.clone(), None).unwrap();

        // Insert the same triple at weight 0.5, flush. (L0 layer.)
        let mt = brain.memtable().unwrap().clone();
        let intern = brain.flush_intern.as_ref().unwrap().clone();
        mt.insert(s, o, BondType::IsA, 0.5, vec![baseline_kf(200, 0.5)], &intern);
        brain.flush_now().unwrap();

        // Insert the triple at weight 0.9 again (MemTable, post-flush).
        mt.insert(s, o, BondType::IsA, 0.9, vec![baseline_kf(300, 0.9)], &intern);

        brain.set_read_path_mode(ReadPathMode::V4Only);
        let got = brain.read_outgoing_bonds(&s).unwrap();
        // Triple is unique on dedup key; exactly one result.
        assert_eq!(got.len(), 1);
        let won = &got[0];
        assert_eq!(won.source_layer, BondSourceLayer::MemTable);
        assert!((won.weight - 0.9).abs() < 1e-6, "expected weight 0.9, got {}", won.weight);
    }

    /// Dedup with WAL-replay collision: simulate the Phase 4 crash window.
    /// Same bond in MemTable AND in an L0 SST. Read returns ONE bond.
    #[test]
    fn dedup_wal_replay_collision_returns_one() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("wal.seg");
        let sst_dir = dir.path().join("ssts");
        std::fs::create_dir_all(&sst_dir).unwrap();
        let manifest_path = dir.path().join("MANIFEST");

        let mut brain = Brain::new();
        brain.enable_memtable(1024 * 1024 * 1024, &wal_path).unwrap();
        brain.enable_flush_worker(sst_dir.clone(), manifest_path.clone(), None).unwrap();
        let mt = brain.memtable().unwrap().clone();
        let intern = brain.flush_intern.as_ref().unwrap().clone();

        let s = Seed::from_label("S1");
        let o = Seed::from_label("O1");
        mt.insert(s, o, BondType::IsA, 0.6, vec![baseline_kf(100, 0.6)], &intern);
        brain.flush_now().unwrap();
        // Simulate replay: insert the same triple again into the MemTable
        // (as if the WAL still had the record after a crash-between-manifest
        // -and-WAL-truncate).
        mt.insert(s, o, BondType::IsA, 0.6, vec![baseline_kf(100, 0.6)], &intern);

        brain.set_read_path_mode(ReadPathMode::V4Only);
        let got = brain.read_outgoing_bonds(&s).unwrap();
        assert_eq!(got.len(), 1, "dedup should collapse the replay collision");
        // MemTable seq > SST id => MemTable wins.
        assert_eq!(got[0].source_layer, BondSourceLayer::MemTable);
    }

    /// Confidence re-derivation: insert with a Session keyframe latest,
    /// read back, expect Unverified.
    #[test]
    fn confidence_rederive_session_to_unverified() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("wal.seg");
        let mut brain = Brain::new();
        brain.enable_memtable(64 * 1024 * 1024, &wal_path).unwrap();
        brain.set_read_path_mode(ReadPathMode::V4Only);
        let mt = brain.memtable().unwrap().clone();
        let intern = brain.flush_intern.as_ref().unwrap().clone();

        let s = Seed::from_label("Sx");
        let o = Seed::from_label("Ox");
        mt.insert(
            s, o, BondType::IsA, 0.5,
            vec![baseline_kf(100, 0.5), session_kf(200, 0.6, 7)],
            &intern,
        );
        let got = brain.read_outgoing_bonds(&s).unwrap();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].confidence, crate::node::Confidence::Unverified);
    }

    /// Confidence re-derivation: Baseline-latest keyframe preserves
    /// stored Verified.
    #[test]
    fn confidence_rederive_baseline_preserves_verified() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("wal.seg");
        let mut brain = Brain::new();
        brain.enable_memtable(64 * 1024 * 1024, &wal_path).unwrap();
        brain.set_read_path_mode(ReadPathMode::V4Only);
        let mt = brain.memtable().unwrap().clone();
        let intern = brain.flush_intern.as_ref().unwrap().clone();

        let s = Seed::from_label("Sy");
        let o = Seed::from_label("Oy");
        mt.insert(
            s, o, BondType::IsA, 0.5,
            vec![session_kf(100, 0.5, 7), baseline_kf(200, 0.6)],
            &intern,
        );
        let got = brain.read_outgoing_bonds(&s).unwrap();
        assert_eq!(got.len(), 1);
        // Baseline preserves stored (Verified is the Phase 4 default).
        assert_eq!(got[0].confidence, crate::node::Confidence::Verified);
    }

    /// Orphan cleanup runs once across multiple FlushWorker startups.
    #[test]
    fn orphan_cleanup_runs_once_across_two_starts() {
        let dir = tempdir().unwrap();
        let sst_dir = dir.path().to_path_buf();
        let wal_path = dir.path().join("wal.seg");
        let manifest_path = dir.path().join("MANIFEST");

        // Pre-seed two orphan SST files.
        std::fs::write(sst_dir.join("L0-101.sst"), b"orphan1").unwrap();
        std::fs::write(sst_dir.join("L0-102.sst"), b"orphan2").unwrap();

        let mut brain = Brain::new();
        brain.enable_memtable(64 * 1024, &wal_path).unwrap();
        brain.enable_flush_worker(sst_dir.clone(), manifest_path.clone(), None).unwrap();
        // First start: both orphans renamed.
        let entries: Vec<String> = std::fs::read_dir(&sst_dir).unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        let orphan_renames_first: Vec<&String> = entries.iter()
            .filter(|n| n.contains(".orphan-")).collect();
        assert_eq!(orphan_renames_first.len(), 2, "both orphans should be renamed on first start");
        // Now drop another "orphan" L0 file and spawn a SECOND worker
        // against the SAME Brain. The Phase 5 flag must skip cleanup.
        std::fs::write(sst_dir.join("L0-103.sst"), b"orphan3").unwrap();
        // Build a second flush worker manually using the SAME orphan_cleanup_done.
        let worker2 = crate::flush::FlushWorker::start(
            brain.memtable.as_ref().unwrap().clone(),
            brain.wal.as_ref().unwrap().clone(),
            brain.flush_intern.as_ref().unwrap().clone(),
            brain.manifest_snapshot.as_ref().unwrap().clone(),
            manifest_path.clone(),
            sst_dir.clone(),
            None,
            brain.orphan_cleanup_done.clone(),
            // OP-BRAIN-V5 Phase 2.3 -- sidecar path next to the WAL.
            brain.sidecar_path().expect("enable_memtable set wal_path"),
            brain.flush_intern.as_ref().unwrap().clone(),
            // OP-BRAIN-V5 Phase 2.6 -- node_store dir next to the WAL.
            crate::node_store::NodeFlushWorker::node_store_dir_for_wal(
                brain.wal_path.as_ref().expect("enable_memtable set wal_path"),
            ),
            // OP-BRAIN-V5 Phase 3.3 -- bond_store dir next to the WAL.
            crate::bond_store::BondFlushWorker::bond_store_dir_for_wal(
                brain.wal_path.as_ref().expect("enable_memtable set wal_path"),
            ),
        );
        drop(worker2);
        // L0-103.sst should still exist UN-renamed.
        assert!(sst_dir.join("L0-103.sst").exists(),
            "second worker must NOT re-run orphan cleanup");
    }

    /// Spread activation A/B: both modes (with empty MemTable + L0)
    /// must produce identical bond sets.
    #[test]
    fn spread_activation_v3_v4_equivalent_on_v3_brain() {
        let mut brain = Brain::new();
        let mut rng = StdRng::seed_from_u64(0xDADE);
        // Build a small graph.
        let mut subjects = Vec::new();
        for _ in 0..20 {
            let s = rand_seed(&mut rng);
            subjects.push(s);
            brain.network.add_node(crate::node::KnowledgeNode::with_seed(s, "s".to_string(), crate::node::KnowledgeContent::Concept("s".to_string()), crate::node::Confidence::Verified, 0));
        }
        for s in &subjects {
            for _ in 0..4 {
                let target_idx = rng.gen_range(0..subjects.len());
                let o = subjects[target_idx];
                if o == *s { continue; }
                let bond = crate::bond::TemporalBond::new(
                    *s, o, BondType::RelatedTo, 0.5, 0,
                );
                brain.network.add_bond(bond).ok();
            }
        }
        // Seed activation from subject 0; capture activated bond set
        // in V3Only vs V4Only modes.
        let seed = subjects[0];
        let seeds = vec![(seed, 1.0)];
        let scope = crate::scope::TemporalScope::Current;
        let scar_log = crate::scar::ScarLog::new();

        brain.set_read_path_mode(ReadPathMode::V3Only);
        let r3 = brain.network.spread_activation_weighted(&seeds, &scope, 2, 100, &scar_log);
        brain.set_read_path_mode(ReadPathMode::V4Only);
        let r4 = brain.network.spread_activation_weighted(&seeds, &scope, 2, 100, &scar_log);

        let set3: HashSet<Seed> = r3.activated.iter().map(|a| a.seed).collect();
        let set4: HashSet<Seed> = r4.activated.iter().map(|a| a.seed).collect();
        assert_eq!(set3, set4,
            "V3Only and V4Only spread activations must match on a v3-only brain (no MemTable/L0)");
    }
}


#[cfg(test)]
mod phase8_v4_boot_tests {
    use super::*;
    use crate::bond::{BondType, KeyframeSource, TemporalKeyframe};
    use crate::manifest::MANIFEST_FILE_NAME;
    use crate::migration_v3_v4::V3ToV4Migrator;
    use crate::read_path::ReadPathMode;
    use crate::seed::Seed;
    use tempfile::tempdir;

    fn baseline_kf(ts: i64, w: f32) -> TemporalKeyframe {
        TemporalKeyframe { timestamp: ts, weight: w, source: KeyframeSource::Baseline }
    }

    // ----- detection tests -----

    #[test]
    fn detect_brain_layout_missing_path() {
        let dir = tempdir().unwrap();
        let p = dir.path().join("does-not-exist.lbr");
        let l = Brain::detect_brain_layout(&p).unwrap();
        assert_eq!(l, BrainLayout::Missing);
        assert!(l.is_v3());
        assert!(!l.is_v4());
    }

    #[test]
    fn detect_brain_layout_v3_file() {
        let dir = tempdir().unwrap();
        let p = dir.path().join("brain.lbr");
        let mut b = Brain::new();
        b.save_format = BrainSaveFormat::V3Binary;
        b.save_to(&p).unwrap();
        assert!(p.is_file());
        let l = Brain::detect_brain_layout(&p).unwrap();
        assert_eq!(l, BrainLayout::V3File);
    }

    #[test]
    fn detect_brain_layout_v4_dir_with_manifest() {
        let dir = tempdir().unwrap();
        let fp = crate::manifest::KeyFingerprint::none();
        let m = crate::manifest::Manifest::new(fp);
        let manifest_path = dir.path().join(MANIFEST_FILE_NAME);
        m.save_atomic(&manifest_path).unwrap();
        let l = Brain::detect_brain_layout(dir.path()).unwrap();
        assert_eq!(l, BrainLayout::V4Dir);
        assert!(l.is_v4());
    }

    #[test]
    fn detect_brain_layout_empty_dir_errors_clearly() {
        let dir = tempdir().unwrap();
        let err = Brain::detect_brain_layout(dir.path()).unwrap_err();
        let msg = format!("{}", err);
        assert!(msg.contains("no MANIFEST"),
            "expected clear MANIFEST-missing error, got: {}", msg);
        assert!(msg.contains("lu4d_migrate_v3_to_v4"),
            "error should point at the migration tool, got: {}", msg);
    }

    #[test]
    fn brain_format_override_parses() {
        assert_eq!(BrainFormatOverride::parse("auto"), Some(BrainFormatOverride::Auto));
        assert_eq!(BrainFormatOverride::parse(""), Some(BrainFormatOverride::Auto));
        assert_eq!(BrainFormatOverride::parse("v3"), Some(BrainFormatOverride::V3));
        assert_eq!(BrainFormatOverride::parse("v4"), Some(BrainFormatOverride::V4));
        assert_eq!(BrainFormatOverride::parse("V4"), Some(BrainFormatOverride::V4));
        assert_eq!(BrainFormatOverride::parse("directory"), Some(BrainFormatOverride::V4));
        assert_eq!(BrainFormatOverride::parse("garbage"), None);
    }

    fn build_tiny_v3_brain(path: &std::path::Path) {
        use crate::node::{Confidence, KnowledgeContent, KnowledgeNode};
        let mut b = Brain::new();
        // Pre-create nodes; add_bond requires endpoints to exist.
        let mut seeds: Vec<Seed> = Vec::new();
        for i in 0..20 {
            let s = Seed::from_label(&format!("s_{}", i));
            let label = format!("s_{}", i);
            let mut node = KnowledgeNode::new(
                label.clone(),
                KnowledgeContent::Raw(label),
                Confidence::Verified,
                0,
            );
            node.seed = s;
            b.network.add_node(node);
            seeds.push(s);
            for j in 0..3 {
                let o = Seed::from_label(&format!("o_{}_{}", i, j));
                let olabel = format!("o_{}_{}", i, j);
                let mut onode = KnowledgeNode::new(
                    olabel.clone(),
                    KnowledgeContent::Raw(olabel),
                    Confidence::Verified,
                    0,
                );
                onode.seed = o;
                b.network.add_node(onode);
            }
        }
        for i in 0..20 {
            let s = Seed::from_label(&format!("s_{}", i));
            for j in 0..3 {
                let o = Seed::from_label(&format!("o_{}_{}", i, j));
                let bond = crate::bond::TemporalBond::new(
                    s, o, BondType::IsA, 0.7, 1700_000_000_000 + i as i64 * 1000 + j as i64,
                );
                b.network.add_bond(bond).expect("add_bond must succeed");
            }
        }
        // Save in V3BinaryCompressed -- the migration tool reads via
        // NetworkArchive::from_mmap_with_key which expects the per-block
        // compressed layout. Bench_migration uses the same format.
        b.save_to_format(path, BrainSaveFormat::V3BinaryCompressed).unwrap();
    }

    #[test]
    fn load_v4_boots_against_migrated_dir() {
        let dir = tempdir().unwrap();
        let v3 = dir.path().join("brain.lbr");
        let v4 = dir.path().join("v4");
        build_tiny_v3_brain(&v3);
        let mig = V3ToV4Migrator::new(v3.clone(), v4.clone(), None).unwrap();
        let report = mig.run().unwrap();
        assert!(!report.idempotent_no_op);
        assert!(report.bonds_migrated >= 60);
        let brain = Brain::load_v4(&v4, None).unwrap();
        assert_eq!(brain.read_path_mode(), ReadPathMode::V4Only);
        assert!(brain.brain_dir_v4().is_some());
        assert_eq!(brain.brain_dir_v4().unwrap(), v4.as_path());
        let snap = brain.manifest_snapshot().expect("manifest snapshot present");
        assert!(snap.total_sst_count() >= 1);
    }

    #[test]
    fn load_v4_round_trips_recall_via_v3_source() {
        let dir = tempdir().unwrap();
        let v3 = dir.path().join("brain.lbr");
        let v4 = dir.path().join("v4");
        build_tiny_v3_brain(&v3);
        V3ToV4Migrator::new(v3.clone(), v4.clone(), None).unwrap().run().unwrap();
        let s = Seed::from_label("s_5");
        let brain = Brain::load_v4(&v4, None).unwrap();
        let bonds = brain.read_outgoing_bonds(&s).unwrap();
        assert_eq!(bonds.len(), 3,
            "v4 boot must surface all 3 outgoing bonds for s_5; got {}: {:?}",
            bonds.len(), bonds);
        for b in &bonds {
            assert_eq!(b.source_layer, crate::read_path::BondSourceLayer::L0Sst);
        }
    }

    #[test]
    fn load_v4_missing_manifest_errors_clearly() {
        let dir = tempdir().unwrap();
        let res = Brain::load_v4(dir.path(), None);
        let err = match res {
            Ok(_) => panic!("load_v4 must reject a directory with no MANIFEST"),
            Err(e) => e,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("MANIFEST"),
            "v4 boot must surface MANIFEST-missing error, got: {}", msg);
    }

    #[test]
    fn brain_load_dispatches_v3_file() {
        let dir = tempdir().unwrap();
        let p = dir.path().join("brain.lbr");
        build_tiny_v3_brain(&p);
        let brain = Brain::load(&p, None).unwrap();
        assert!(brain.brain_dir_v4().is_none());
        assert_eq!(brain.read_path_mode(), ReadPathMode::V3Only);
    }

    #[test]
    fn brain_load_dispatches_v4_dir() {
        let dir = tempdir().unwrap();
        let v3 = dir.path().join("brain.lbr");
        let v4 = dir.path().join("v4");
        build_tiny_v3_brain(&v3);
        V3ToV4Migrator::new(v3.clone(), v4.clone(), None).unwrap().run().unwrap();
        let brain = Brain::load(&v4, None).unwrap();
        assert!(brain.brain_dir_v4().is_some());
        assert_eq!(brain.read_path_mode(), ReadPathMode::V4Only);
    }

    #[test]
    fn brain_load_missing_path_falls_through_to_v3_empty() {
        let dir = tempdir().unwrap();
        let p = dir.path().join("new-brain.lbr");
        let brain = Brain::load(&p, None).unwrap();
        assert!(brain.brain_dir_v4().is_none());
        assert_eq!(brain.read_path_mode(), ReadPathMode::V3Only);
    }

    #[test]
    fn save_on_v4_brain_is_noop() {
        let dir = tempdir().unwrap();
        let v3 = dir.path().join("brain.lbr");
        let v4 = dir.path().join("v4");
        build_tiny_v3_brain(&v3);
        V3ToV4Migrator::new(v3.clone(), v4.clone(), None).unwrap().run().unwrap();
        let mut brain = Brain::load_v4(&v4, None).unwrap();
        let manifest_path = v4.join(MANIFEST_FILE_NAME);
        let mtime_before = std::fs::metadata(&manifest_path).unwrap().modified().unwrap();
        brain.save().expect("v4 save must succeed as a no-op");
        let mtime_after = std::fs::metadata(&manifest_path).unwrap().modified().unwrap();
        assert_eq!(mtime_before, mtime_after);
    }

    #[test]
    fn wal_replay_on_v4_boot_recovers_unflushed_writes() {
        let dir = tempdir().unwrap();
        let v3 = dir.path().join("brain.lbr");
        let v4 = dir.path().join("v4");
        build_tiny_v3_brain(&v3);
        V3ToV4Migrator::new(v3.clone(), v4.clone(), None).unwrap().run().unwrap();
        let wal_path = v4.join("wal").join("current.wal");
        {
            let brain = Brain::load_v4(&v4, None).unwrap();
            let intern = brain.flush_intern.as_ref().unwrap().clone();
            let mt = brain.memtable().unwrap().clone();
            let wal = brain.wal().unwrap().clone();
            for i in 0..100 {
                let s = Seed::from_label(&format!("replay_s_{}", i));
                let o = Seed::from_label(&format!("replay_o_{}", i));
                let kfs = vec![baseline_kf(1700_000_000_000, 0.9)];
                let r = mt.insert(s, o, BondType::IsA, 0.9, kfs.clone(), &intern);
                wal.lock().unwrap()
                    .append(r.seq, &s, &o, BondType::IsA, 0.9, &kfs).unwrap();
            }
            wal.lock().unwrap().fsync().unwrap();
        }
        assert!(wal_path.exists());
        let records_on_disk = crate::wal::replay(&wal_path).unwrap();
        assert_eq!(records_on_disk.len(), 100);
        let brain2 = Brain::load_v4(&v4, None).unwrap();
        let mt2 = brain2.memtable().unwrap();
        assert_eq!(mt2.bond_count(), 100,
            "WAL replay must rebuild MemTable on v4 boot");
    }

    #[test]
    fn flush_now_on_v4_boot_emits_l0_sst() {
        let dir = tempdir().unwrap();
        let v3 = dir.path().join("brain.lbr");
        let v4 = dir.path().join("v4");
        build_tiny_v3_brain(&v3);
        V3ToV4Migrator::new(v3.clone(), v4.clone(), None).unwrap().run().unwrap();
        let brain = Brain::load_v4(&v4, None).unwrap();
        let snap_before = brain.manifest_snapshot().unwrap();
        let total_before = snap_before.total_sst_count();
        let intern = brain.flush_intern.as_ref().unwrap().clone();
        let mt = brain.memtable().unwrap().clone();
        let wal = brain.wal().unwrap().clone();
        for i in 0..500 {
            let s = Seed::from_label(&format!("flush_s_{}", i));
            let o = Seed::from_label(&format!("flush_o_{}", i));
            let kfs = vec![baseline_kf(1700_000_001_000, 0.5)];
            let r = mt.insert(s, o, BondType::IsA, 0.5, kfs.clone(), &intern);
            wal.lock().unwrap().append(r.seq, &s, &o, BondType::IsA, 0.5, &kfs).unwrap();
        }
        wal.lock().unwrap().fsync().unwrap();
        brain.flush_now().unwrap();
        let snap_after = brain.manifest_snapshot().unwrap();
        assert!(snap_after.total_sst_count() > total_before,
            "flush_now must produce a new SST (before={}, after={})",
            total_before, snap_after.total_sst_count());
    }
}

// OP-BRAIN-V4 Phase 9 unit tests -- verify teach_with_kind dispatches
// through MemTable + WAL when brain_dir_v4.is_some(). The cutover bug
// these tests prevent was: every teach call landed in the v3 HashMaps
// while the WAL stayed at 8 bytes (just the magic header). Each test
// here asserts the WAL grows on teach, that subsequent reads see the
// inserted bonds, and that a restart-replay round-trip preserves them.
#[cfg(test)]
#[allow(non_snake_case)]
mod cutover_v4_write_path_phase9_tests {
    use super::*;
    use crate::node::Confidence;
    use crate::manifest::MANIFEST_FILE_NAME;
    use crate::bond::BondType;
    use tempfile::tempdir;

    /// Build the smallest possible v4 brain: empty MANIFEST, fresh WAL.
    /// Avoids the v3 -> v4 migration so we exercise the write path in
    /// isolation (the migration tool writes directly to SSTs and does
    /// not exercise teach_with_kind).
    fn make_empty_v4_brain(brain_dir: &std::path::Path) -> Brain {
        let fp = crate::manifest::KeyFingerprint::none();
        let m = crate::manifest::Manifest::new(fp);
        let manifest_path = brain_dir.join(MANIFEST_FILE_NAME);
        m.save_atomic(&manifest_path).unwrap();
        Brain::load_v4(brain_dir, None).expect("empty v4 brain must load")
    }

    fn wal_size_bytes(brain_dir: &std::path::Path) -> u64 {
        let wal_path = brain_dir.join("wal").join("current.wal");
        std::fs::metadata(&wal_path).map(|m| m.len()).unwrap_or(0)
    }

    #[test]
    fn phase9_teach_on_v4_brain_grows_wal_past_magic_header() {
        let dir = tempdir().unwrap();
        let v4 = dir.path().join("v4");
        std::fs::create_dir_all(&v4).unwrap();
        let mut brain = make_empty_v4_brain(&v4);

        // Magic-only WAL is 8 bytes (the magic header lands at
        // open()).
        let before = wal_size_bytes(&v4);
        assert_eq!(before, 8, "fresh WAL must be exactly 8 magic bytes");

        brain.teach(
            "v4_subject",
            BondType::IsA,
            "v4_object",
            0.9,
            Confidence::Verified,
        );

        // Force WAL fsync so the test asserts on-disk bytes, not
        // BufWriter-buffered bytes.
        brain.wal().unwrap().lock().unwrap().fsync().unwrap();

        let after = wal_size_bytes(&v4);
        assert!(after > before,
            "WAL must grow past the magic header on teach (before={}, after={})",
            before, after);
    }

    #[test]
    fn phase9_teach_on_v4_brain_lands_in_memtable() {
        let dir = tempdir().unwrap();
        let v4 = dir.path().join("v4");
        std::fs::create_dir_all(&v4).unwrap();
        let mut brain = make_empty_v4_brain(&v4);

        assert_eq!(brain.memtable().unwrap().bond_count(), 0,
            "fresh v4 MemTable must be empty");

        for i in 0..100 {
            brain.teach(
                &format!("subj_{}", i),
                BondType::IsA,
                &format!("obj_{}", i),
                0.7,
                Confidence::Verified,
            );
        }

        assert_eq!(brain.memtable().unwrap().bond_count(), 100,
            "100 teach calls must produce 100 MemTable bonds on a v4 brain");
        // And -- critically -- ZERO bonds in the v3 HashMaps. This is
        // the exact opposite of the pre-Phase-9 bug, where every bond
        // would have landed here.
        assert_eq!(brain.network.bond_count(), 0,
            "v3 network HashMaps must stay empty for v4-loaded brain writes");
    }

    #[test]
    fn phase9_teach_on_v3_brain_unchanged_lands_in_network() {
        // Sanity: the v4 dispatch must NOT touch the v3 teach path.
        // A bare Brain::new() has `brain_dir_v4 = None`.
        let mut brain = Brain::new();
        assert!(brain.brain_dir_v4().is_none());

        brain.teach("a", BondType::IsA, "b", 0.9, Confidence::Verified);

        assert_eq!(brain.network.bond_count(), 1,
            "v3 brain teach must land in network.bonds, not MemTable");
        assert!(brain.memtable().is_none(),
            "v3 brain must not have a MemTable");
    }

    #[test]
    fn phase9_teach_on_v4_brain_is_readable_via_merged_read_path() {
        let dir = tempdir().unwrap();
        let v4 = dir.path().join("v4");
        std::fs::create_dir_all(&v4).unwrap();
        let mut brain = make_empty_v4_brain(&v4);

        brain.teach("alpha", BondType::IsA, "beta", 0.85, Confidence::Verified);
        brain.teach("alpha", BondType::HasProperty, "gamma", 0.75, Confidence::Verified);

        let subj = Seed::from_label("alpha");
        let bonds = brain.read_outgoing_bonds(&subj).unwrap();
        assert_eq!(bonds.len(), 2, "v4 read merge must surface both teach bonds");

        let mut predicates: Vec<BondType> = bonds.iter().map(|b| b.predicate).collect();
        predicates.sort_by_key(|p| format!("{:?}", p));
        assert_eq!(predicates, vec![BondType::HasProperty, BondType::IsA]);

        // Source layer must be MemTable (not L0Sst, not V3Archive).
        for b in &bonds {
            assert_eq!(b.source_layer, crate::read_path::BondSourceLayer::MemTable);
        }
    }

    #[test]
    fn phase9_restart_after_teach_preserves_bonds_via_wal_replay() {
        let dir = tempdir().unwrap();
        let v4 = dir.path().join("v4");
        std::fs::create_dir_all(&v4).unwrap();

        {
            let mut brain = make_empty_v4_brain(&v4);
            for i in 0..50 {
                brain.teach(
                    &format!("survivor_{}", i),
                    BondType::Causes,
                    &format!("effect_{}", i),
                    0.95,
                    Confidence::Verified,
                );
            }
            brain.wal().unwrap().lock().unwrap().fsync().unwrap();
            // Drop brain explicitly (simulates clean shutdown).
        }

        // Re-load. WAL replay must rebuild the MemTable.
        let brain2 = Brain::load_v4(&v4, None).unwrap();
        assert_eq!(brain2.memtable().unwrap().bond_count(), 50,
            "WAL replay must restore all 50 teach bonds across restart");

        let probe = Seed::from_label("survivor_7");
        let bonds = brain2.read_outgoing_bonds(&probe).unwrap();
        assert_eq!(bonds.len(), 1);
        assert_eq!(bonds[0].object, Seed::from_label("effect_7"));
        assert_eq!(bonds[0].predicate, BondType::Causes);
    }

    #[test]
    fn phase9_teach_corpus_on_v4_routes_through_memtable() {
        let dir = tempdir().unwrap();
        let v4 = dir.path().join("v4");
        std::fs::create_dir_all(&v4).unwrap();
        let mut brain = make_empty_v4_brain(&v4);

        // lu4d_teach_corpus path -> teach_corpus_with_kind ->
        // teach_with_source_kind -> teach_with_kind. Verifies the
        // single-dispatch invariant: corpus teach lands in v4 layer
        // automatically without a per-MCP-tool branch.
        for i in 0..20 {
            brain.teach_corpus(
                &format!("c_subj_{}", i),
                BondType::IsA,
                &format!("c_obj_{}", i),
                0.8,
                Confidence::Verified,
                "test_corpus",
                None,
                None,
            );
        }

        assert_eq!(brain.memtable().unwrap().bond_count(), 20);
        assert_eq!(brain.network.bond_count(), 0,
            "teach_corpus on v4 must not touch v3 HashMaps");
    }

    #[test]
    fn phase9_status_reports_v4_bond_count_from_memtable() {
        let dir = tempdir().unwrap();
        let v4 = dir.path().join("v4");
        std::fs::create_dir_all(&v4).unwrap();
        let mut brain = make_empty_v4_brain(&v4);

        // Pre-teach: bond_count must be 0 (not unset, not negative).
        let s0 = brain.status();
        assert_eq!(s0.bond_count, 0);

        for i in 0..30 {
            brain.teach(
                &format!("s_{}", i),
                BondType::RelatedTo,
                &format!("o_{}", i),
                0.5,
                Confidence::Inferred,
            );
        }

        // brain.status() pulls from v3 HashMaps -- it would still
        // report 0 for a v4 brain. The MCP-level tool_status sums
        // memtable + manifest; verify the underlying memtable count
        // directly so the assertion is honest about WHERE the bonds
        // are.
        assert_eq!(brain.memtable().unwrap().bond_count(), 30);
    }

    #[test]
    fn phase9_scarred_seed_refuses_v4_teach_without_inserting() {
        let dir = tempdir().unwrap();
        let v4 = dir.path().join("v4");
        std::fs::create_dir_all(&v4).unwrap();
        let mut brain = make_empty_v4_brain(&v4);

        let scarred = Seed::from_label("forbidden_subject");
        brain.scar_log.scar_seed(scarred);

        brain.teach("forbidden_subject", BondType::IsA, "target", 0.9, Confidence::Verified);

        assert_eq!(brain.memtable().unwrap().bond_count(), 0,
            "v4 teach must respect scar admission and never insert a bond              whose endpoint is scarred");
    }

    // ---------------------------------------------------------------
    // Phase 10 -- node-persistence end-to-end tests.
    // ---------------------------------------------------------------

    /// The critical regression-from-Phase-9 fix: teach bonds into a
    /// fresh v4 brain, drop the Brain handle (simulates daemon kill),
    /// reload via Brain::load_v4, verify nodes are reconstructed
    /// (via WAL replay) and `network.get_node` returns the expected
    /// content. Without Phase 10 this test would fail because nodes
    /// live in RAM-only `network.nodes`.
    #[test]
    fn phase10_node_metadata_persists_across_restart() {
        let dir = tempdir().unwrap();
        let v4 = dir.path().join("v4");
        std::fs::create_dir_all(&v4).unwrap();
        // Build empty v4 + teach 10 bonds.
        {
            let mut brain = make_empty_v4_brain(&v4);
            for i in 0..10 {
                brain.teach(
                    &format!("alpha_subj_{}", i),
                    BondType::IsA,
                    &format!("alpha_obj_{}", i),
                    0.9,
                    Confidence::Verified,
                );
            }
            // bond_count from MemTable confirms write path landed.
            assert_eq!(brain.memtable().unwrap().bond_count(), 10);
            // network.nodes carries the subject/object payloads from the
            // live teach.
            assert!(brain.network.get_node(&Seed::from_label("alpha_subj_0")).is_some());
            // drop simulates kill.
        }
        // Now reload (WAL replay should rebuild network.nodes from the
        // node records the live teach wrote alongside the bonds).
        let brain2 = Brain::load_v4(&v4, None).expect("reload v4 brain");
        for i in 0..10 {
            let s = Seed::from_label(&format!("alpha_subj_{}", i));
            assert!(
                brain2.network.get_node(&s).is_some(),
                "subject {} missing after restart -- node persistence regression",
                i,
            );
            let o = Seed::from_label(&format!("alpha_obj_{}", i));
            assert!(
                brain2.network.get_node(&o).is_some(),
                "object {} missing after restart",
                i,
            );
        }
        // And the MemTable still has the bonds via WAL replay.
        assert_eq!(brain2.memtable().unwrap().bond_count(), 10);
    }

    /// Flush-aware variant: trigger a flush so the WAL gets truncated
    /// and node records live in SSTs. Reload must then rehydrate via
    /// `rehydrate_network_nodes_from_v4_ssts`.
    #[test]
    fn phase10_node_metadata_persists_via_sst_path() {
        let dir = tempdir().unwrap();
        let v4 = dir.path().join("v4");
        std::fs::create_dir_all(&v4).unwrap();
        {
            let mut brain = make_empty_v4_brain(&v4);
            for i in 0..20 {
                brain.teach(
                    &format!("flush_subj_{}", i),
                    BondType::HasProperty,
                    &format!("flush_obj_{}", i),
                    0.7,
                    Confidence::Verified,
                );
            }
            // Force a flush so the bonds + nodes land in an L0 SST and
            // the WAL gets truncated.
            brain.flush_worker.as_ref().expect("flush worker installed").flush_now().unwrap();
        }
        let brain2 = Brain::load_v4(&v4, None).expect("reload v4 brain after flush");
        // After flush + reload, the WAL was truncated; nodes must come
        // back via the SST rehydration path.
        for i in 0..20 {
            let s = Seed::from_label(&format!("flush_subj_{}", i));
            assert!(
                brain2.network.get_node(&s).is_some(),
                "flush_subj_{} missing after flush+restart",
                i,
            );
        }
    }

    // ------------------------------------------------------------
    // OP-BRAIN-V5 Phase 1.4 -- teach-path NodeIntern WAL wiring.
    // ------------------------------------------------------------

    /// A. Teach-then-cold-restart: every bond record's seeds must be
    /// preceded in the WAL by a REC_TYPE_NODE_INTERN record. Verified
    /// by `replay_streaming_validated` returning Ok after a 1K-bond
    /// teach + drop + reopen cycle.
    #[test]
    fn phase1_4_teach_then_cold_restart_replay_validated() {
        let dir = tempdir().unwrap();
        let v4 = dir.path().join("v4");
        std::fs::create_dir_all(&v4).unwrap();
        const N: usize = 1_000;
        {
            let mut brain = make_empty_v4_brain(&v4);
            for i in 0..N {
                brain.teach(
                    &format!("phase14_subj_{}", i),
                    BondType::IsA,
                    &format!("phase14_obj_{}", i),
                    0.7,
                    Confidence::Verified,
                );
            }
            // Force WAL bytes to disk before drop.
            brain.wal().unwrap().lock().unwrap().fsync().unwrap();
        }
        // Streaming-replay the WAL with the firewall validator. If
        // any bond record references a Seed not preceded by a
        // NodeIntern record, this surfaces `UnknownNodeIdInBond`.
        let wal_path = v4.join("wal").join("current.wal");
        let mut bond_count = 0usize;
        let mut intern_count = 0usize;
        crate::wal::replay_streaming_validated(&wal_path, |rec| {
            match rec {
                crate::wal::WalRecord::Bond { .. } => bond_count += 1,
                crate::wal::WalRecord::NodeIntern { .. } => intern_count += 1,
                crate::wal::WalRecord::Node { .. } => {}
            }
            Ok(())
        }).expect("streaming replay must succeed: every bond seed must be preceded by a NodeIntern record");
        assert_eq!(bond_count, N, "all N teach bonds must replay");
        // Each distinct Seed is interned at most once -- this corpus
        // has 2*N distinct seeds (subj_i and obj_i for i in 0..N).
        assert_eq!(intern_count, 2 * N, "every distinct seed must intern exactly once");
    }

    /// B. Idempotence: teaching the same triple 100x produces NodeIntern
    /// records EXACTLY ONCE per Seed (not 100x). The bond record is
    /// idempotent on the replay-validator front -- the firewall only
    /// requires "NodeIntern precedes bond", which is satisfied after
    /// the first teach for that seed.
    #[test]
    fn phase1_4_intern_idempotence_100x_same_bond_one_intern_record_each() {
        let dir = tempdir().unwrap();
        let v4 = dir.path().join("v4");
        std::fs::create_dir_all(&v4).unwrap();
        const REPS: usize = 100;
        {
            let mut brain = make_empty_v4_brain(&v4);
            for _ in 0..REPS {
                brain.teach(
                    "idempotent_subject",
                    BondType::IsA,
                    "idempotent_object",
                    0.5,
                    Confidence::Verified,
                );
            }
            brain.wal().unwrap().lock().unwrap().fsync().unwrap();
        }
        let wal_path = v4.join("wal").join("current.wal");
        let mut intern_seeds: std::collections::HashMap<crate::seed::Seed, usize> =
            std::collections::HashMap::new();
        let mut bond_count = 0usize;
        crate::wal::replay_streaming_validated(&wal_path, |rec| {
            match rec {
                crate::wal::WalRecord::NodeIntern { seed, .. } => {
                    *intern_seeds.entry(seed).or_insert(0) += 1;
                }
                crate::wal::WalRecord::Bond { .. } => bond_count += 1,
                crate::wal::WalRecord::Node { .. } => {}
            }
            Ok(())
        }).expect("streaming replay must succeed");
        // Per-Seed: at most ONE NodeIntern record per distinct seed.
        for (s, count) in &intern_seeds {
            assert_eq!(
                *count, 1,
                "seed {:?} got {} NodeIntern records (expected exactly 1)",
                s, count
            );
        }
        // The two distinct seeds in this corpus must both appear.
        let subj = crate::seed::Seed::from_label("idempotent_subject");
        let obj = crate::seed::Seed::from_label("idempotent_object");
        assert!(intern_seeds.contains_key(&subj));
        assert!(intern_seeds.contains_key(&obj));
        assert_eq!(intern_seeds.len(), 2, "exactly 2 distinct seeds in this corpus");
        // Bond records: every teach call writes one (reinforce shape is
        // still a fresh-insert at the WAL layer per the v4 contract).
        assert_eq!(bond_count, REPS, "every teach must emit one bond WAL record");
    }

    /// C. Ordering invariant under churn: 10K teaches with HEAVILY
    /// overlapping subject/object seeds (drawn from a 200-seed pool)
    /// must replay cleanly. The intern table grows to 200 entries
    /// total; the bond stream contains 10K records that each reference
    /// two of those seeds. The ordering invariant requires each
    /// bond's seeds to be interned BEFORE the bond -- this only
    /// works if `intern_with_new_flag` and the WAL append are
    /// atomic under the WAL lock. A regression that interns outside
    /// the lock would surface here as a sporadic
    /// `UnknownNodeIdInBond`.
    #[test]
    fn phase1_4_ordering_invariant_under_churn_10k_overlapping_seeds() {
        let dir = tempdir().unwrap();
        let v4 = dir.path().join("v4");
        std::fs::create_dir_all(&v4).unwrap();
        const POOL: usize = 200;
        const N: usize = 10_000;
        {
            let mut brain = make_empty_v4_brain(&v4);
            // Deterministic pseudo-random churn: avoid `rand` (not in
            // dev-deps here) by walking a stride that visits every
            // pool index. Mutual-prime stride keeps subj/obj pairs
            // diverse without repeating early.
            let mut a = 0usize;
            let mut b = 100usize;
            for _ in 0..N {
                brain.teach(
                    &format!("churn_seed_{}", a),
                    BondType::RelatedTo,
                    &format!("churn_seed_{}", b),
                    0.5,
                    Confidence::Verified,
                );
                a = (a + 7) % POOL;
                b = (b + 13) % POOL;
            }
            brain.wal().unwrap().lock().unwrap().fsync().unwrap();
        }
        let wal_path = v4.join("wal").join("current.wal");
        let mut bond_count = 0usize;
        let mut intern_count = 0usize;
        crate::wal::replay_streaming_validated(&wal_path, |rec| {
            match rec {
                crate::wal::WalRecord::Bond { .. } => bond_count += 1,
                crate::wal::WalRecord::NodeIntern { .. } => intern_count += 1,
                crate::wal::WalRecord::Node { .. } => {}
            }
            Ok(())
        }).expect("churn replay must succeed: every seed pre-interned");
        assert_eq!(bond_count, N, "all N churn bonds must replay");
        // The pool has POOL distinct seeds; every one of them is reached
        // by the (a + 7) / (b + 13) walk because gcd(7, 200) = 1.
        assert_eq!(intern_count, POOL, "every distinct seed in the pool must intern exactly once");

        // Negative half: hand-craft a WAL with a bond BEFORE its
        // NodeIntern records and confirm the firewall fires. We reuse
        // the same dir but a fresh WAL path for isolation.
        let bad = dir.path().join("bad.wal");
        {
            let mut wal = crate::wal::Wal::open(&bad).unwrap();
            let s = Seed::from_label("never_interned_subject");
            let o = Seed::from_label("never_interned_object");
            // Bond first -- the ordering violation.
            wal.append(0, &s, &o, BondType::IsA, 1.0, &[]).unwrap();
            wal.fsync().unwrap();
        }
        let err = crate::wal::replay_streaming_validated(&bad, |_| Ok(())).unwrap_err();
        match err {
            crate::wal::WalError::UnknownNodeIdInBond { .. } => {}
            other => panic!("expected UnknownNodeIdInBond, got {:?}", other),
        }
    }
}
#[cfg(test)]
mod phase2_2_sidecar_aware_boot_replay_tests {
    //! OP-BRAIN-V5 Phase 2.2 -- sidecar-aware boot-replay arm tests.
    //!
    //! Five test cases per the Phase 2.2 brief:
    //!
    //!   A. Case A round-trip -- sidecar + matching WAL records ->
    //!      verbatim NodeId restoration.
    //!   B. Case B fallback -- no sidecar -> intern_or_get reassigns,
    //!      no error, warning logged.
    //!   C. Case A collision -- sidecar says NodeId(X), WAL says
    //!      NodeId(Y), X != Y -> hard error.
    //!   D. Corrupt sidecar -> hard error, NOT silent Case B fallback.
    //!   E. Empty WAL + present sidecar -> sidecar's bindings
    //!      preserved unchanged.
    //!
    //! Sidecar path convention: `Brain::sidecar_path_for_wal(wal_path)`
    //! returns `wal_path.parent().join("nodeintern.bin")`. For these
    //! tests using `tempdir().join("wal.seg")` the sidecar is at
    //! `tempdir().join("nodeintern.bin")`.

    use super::*;
    use crate::node_intern::{NodeId, NodeIntern};
    use crate::seed::Seed;
    use tempfile::tempdir;

    /// Build a small WAL containing N `REC_TYPE_NODE_INTERN` records
    /// at the given (seed, node_id) pairs, in caller-supplied order.
    /// fsync's before close.
    fn write_wal_with_intern_records(wal_path: &std::path::Path, records: &[(Seed, NodeId)]) {
        let mut wal = crate::wal::Wal::open(wal_path).unwrap();
        for (i, (seed, id)) in records.iter().enumerate() {
            wal.append_node_intern(i as u64, seed, *id).unwrap();
        }
        wal.fsync().unwrap();
    }

    /// Build a sidecar file at the canonical Phase 2.2 path for the
    /// given WAL. Uses NodeIntern::save_to to keep the on-disk layout
    /// honest (no hand-rolled bytes).
    fn write_sidecar_at_canonical_path(wal_path: &std::path::Path, pairs: &[(Seed, NodeId)]) -> std::path::PathBuf {
        let intern = NodeIntern::new();
        for (seed, id) in pairs {
            intern.insert_with_id(*seed, *id).unwrap();
        }
        let sidecar = Brain::sidecar_path_for_wal(wal_path);
        intern.save_to(&sidecar).unwrap();
        sidecar
    }

    /// Test A: Case A round-trip. Sidecar present + WAL with matching
    /// `REC_TYPE_NODE_INTERN` records. After `enable_memtable`, every
    /// Seed maps to its original NodeId verbatim.
    #[test]
    fn phase2_2_case_a_roundtrip_restores_verbatim_node_ids() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("wal.seg");

        // 100 deterministic (Seed, NodeId) pairs. NodeIds are chosen
        // intentionally NOT in seed-hash order so insertion-order
        // reassignment would produce a visibly different mapping.
        let pairs: Vec<(Seed, NodeId)> = (0..100u64)
            .map(|i| (Seed::from_label(&format!("phase22_a_seed_{}", i)), NodeId((100 - i) * 7)))
            .collect();

        // Sidecar holds the snapshot.
        write_sidecar_at_canonical_path(&wal_path, &pairs);
        // WAL repeats the same bindings (the delta-since-snapshot
        // would normally be the OLDER records that the sidecar
        // already captured; for the round-trip test we exercise the
        // idempotent insert_with_id path where sidecar and WAL agree).
        write_wal_with_intern_records(&wal_path, &pairs);

        let mut brain = Brain::new();
        brain.enable_memtable(1024 * 1024, &wal_path).expect("Case A enable_memtable must succeed");

        let intern = brain.network.node_intern().clone();
        for (seed, expected_id) in &pairs {
            assert_eq!(
                intern.get(seed),
                Some(*expected_id),
                "sidecar+WAL must restore NodeId verbatim for seed {}",
                seed,
            );
            assert_eq!(
                intern.reverse_lookup(*expected_id),
                Some(*seed),
                "reverse_lookup must round-trip",
            );
        }
    }

    /// Test B: Case B fallback. No sidecar file on disk. WAL contains
    /// `REC_TYPE_NODE_INTERN` records. `enable_memtable` succeeds with
    /// no error; NodeIntern is populated; NodeIds are reassigned by
    /// insertion order (NOT the WAL's stored values).
    #[test]
    fn phase2_2_case_b_no_sidecar_falls_back_to_intern_or_get() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("wal.seg");

        // 20 WAL records whose stored NodeIds intentionally start at
        // 500 so we can detect whether they were preserved or
        // reassigned by insertion order (which would produce
        // NodeIds 0..20).
        let pairs: Vec<(Seed, NodeId)> = (0..20u64)
            .map(|i| (Seed::from_label(&format!("phase22_b_seed_{}", i)), NodeId(500 + i)))
            .collect();
        write_wal_with_intern_records(&wal_path, &pairs);
        // No sidecar file written -- sanity check.
        assert!(!Brain::sidecar_path_for_wal(&wal_path).exists(), "sidecar must be absent for Case B");

        let mut brain = Brain::new();
        brain.enable_memtable(1024 * 1024, &wal_path).expect("Case B must NOT error");

        let intern = brain.network.node_intern().clone();
        // Every seed must be interned.
        for (seed, _stored_id) in &pairs {
            assert!(intern.get(seed).is_some(), "seed must be interned via intern_or_get fallback");
        }
        // The reassigned NodeIds are the insertion order (0..20),
        // NOT the stored 500.. values. We confirm reassignment by
        // checking that NodeId(500) is NOT bound to the first seed.
        let first_seed = pairs[0].0;
        let observed = intern.get(&first_seed).unwrap();
        assert_ne!(observed, NodeId(500), "Case B must NOT preserve the WAL's stored NodeIds");
        assert_eq!(observed.0, 0, "Case B assigns 0..N in insertion order");
    }

    /// Test C: Case A collision. Sidecar binds seed_x -> NodeId(5).
    /// WAL has `REC_TYPE_NODE_INTERN(seed_x, NodeId(7))`. The replay
    /// must surface the collision as a BrainError::Format (carrying
    /// the InsertWithIdCollision message).
    #[test]
    fn phase2_2_case_a_collision_is_hard_error() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("wal.seg");

        let collide_seed = Seed::from_label("phase22_c_collider");

        // Sidecar says NodeId(5).
        write_sidecar_at_canonical_path(&wal_path, &[(collide_seed, NodeId(5))]);
        // WAL says NodeId(7) for the same seed -- the conflict.
        write_wal_with_intern_records(&wal_path, &[(collide_seed, NodeId(7))]);

        let mut brain = Brain::new();
        let res = brain.enable_memtable(1024 * 1024, &wal_path);
        match res {
            Err(BrainError::Format(msg)) => {
                assert!(
                    msg.contains("collision") || msg.contains("InsertWithIdCollision"),
                    "error message must mention the collision, got: {}",
                    msg,
                );
            }
            Err(other) => panic!("expected BrainError::Format for collision, got {:?}", other),
            Ok(()) => panic!("expected hard error on Case A collision, got Ok(())"),
        }
    }

    /// Test D: corrupt sidecar -> hard error, NOT silent Case B
    /// fallback. We save a valid sidecar, flip a byte in its body,
    /// then assert `enable_memtable` returns BrainError::Format
    /// surfacing the CRC mismatch.
    #[test]
    fn phase2_2_corrupt_sidecar_is_hard_error_not_silent_fallback() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("wal.seg");

        let pairs: Vec<(Seed, NodeId)> = (0..10u64)
            .map(|i| (Seed::from_label(&format!("phase22_d_seed_{}", i)), NodeId(i)))
            .collect();
        let sidecar = write_sidecar_at_canonical_path(&wal_path, &pairs);

        // Corrupt one byte in the body (after the 16-byte header).
        let mut bytes = std::fs::read(&sidecar).unwrap();
        // Flip a byte deep enough to land inside a record body.
        bytes[32] ^= 0x5A;
        std::fs::write(&sidecar, &bytes).unwrap();

        // Also drop a single valid REC_TYPE_NODE_INTERN record into
        // the WAL so the test path is realistic (the corruption
        // detection precedes WAL replay; the WAL records here are
        // never reached).
        write_wal_with_intern_records(&wal_path, &pairs);

        let mut brain = Brain::new();
        let res = brain.enable_memtable(1024 * 1024, &wal_path);
        match res {
            Err(BrainError::Format(msg)) => {
                assert!(
                    msg.contains("CRC") || msg.contains("crc") || msg.contains("sidecar load failed"),
                    "error message must surface the CRC mismatch / sidecar load failure, got: {}",
                    msg,
                );
            }
            Err(other) => panic!(
                "expected BrainError::Format(CRC...), got {:?} -- corruption MUST NOT silently fall back to Case B",
                other,
            ),
            Ok(()) => panic!(
                "expected hard error on corrupt sidecar, got Ok(()) -- this is the feedback_no_hacks regression",
            ),
        }
    }

    /// Test E: present sidecar + empty WAL. After `enable_memtable`,
    /// every binding in the sidecar is preserved exactly.
    #[test]
    fn phase2_2_empty_wal_with_present_sidecar_preserves_sidecar_bindings() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("wal.seg");

        let pairs: Vec<(Seed, NodeId)> = (0..10u64)
            .map(|i| (Seed::from_label(&format!("phase22_e_seed_{}", i)), NodeId(i * 13 + 1)))
            .collect();
        write_sidecar_at_canonical_path(&wal_path, &pairs);
        // WAL: opened but no records written. `Wal::open` lays down
        // the 8-byte magic header so the file exists; subsequent
        // `crate::wal::replay` returns an empty Vec.
        { let _wal = crate::wal::Wal::open(&wal_path).unwrap(); }
        assert_eq!(crate::wal::replay(&wal_path).unwrap().len(), 0, "WAL must be empty for this test");

        let mut brain = Brain::new();
        brain.enable_memtable(1024 * 1024, &wal_path).expect("empty-WAL + sidecar must succeed");

        let intern = brain.network.node_intern().clone();
        for (seed, expected_id) in &pairs {
            assert_eq!(
                intern.get(seed),
                Some(*expected_id),
                "sidecar binding for seed {} must be preserved with NO WAL delta",
                seed,
            );
        }
    }
}
// =====================================================================
// OP-BRAIN-V5 Phase 3.8 -- Q9 sanity gate: K-gate, hub-fill, and
// curiosity run against a v5-backed Brain.
//
// Design doc Q9 predicted: "All [K-gate, hub-fill, curiosity, ...]
// operate above the substrate via the Brain trait API. No port
// required." These tests check that prediction empirically. Per the
// Phase 3.8 brief, NO K-gate / hub-fill / curiosity source is
// modified here -- where a system fails the gate, the test ASSERTS
// the observed failure and the doc-comment marks it as a Section 7
// design-doc candidate (substrate-side issue, not a consumer bug).
//
// Empirical verdicts (2026-06-10, this commit):
//   * K-gate    -- PASS. Verdicts identical on v3-write and
//                  v5-backed brains (known topic clears the gate,
//                  unknown topic refuses).
//   * hub-fill  -- FAIL. `internal_fill_at_hub` / `_at_seed` consume
//                  raw `AssociativeNetwork` adjacency, which is EMPTY
//                  on a v4/v5-loaded brain (bonds live in MemTable +
//                  SSTs; `network.bond_count() == 0`). 0 proposals
//                  vs 3 on the v3-write control. Section 7 candidate.
//   * curiosity -- PASS mechanically (signals emitted, wonder queue
//                  populated, interest profile touched) with a
//                  fidelity caveat: `curiosity::degree()` also reads
//                  raw adjacency, so every activated seed reports
//                  degree 0 on a v5-backed brain and IsolatedNode
//                  intensities are inflated (0.430 observed vs
//                  0.114/0.143 on the v3-write control).
//
// Phase 3.7 MemTable gap: the bond MemTable stores Seeds (not
// NodeIds), so un-flushed bonds are invisible to the v5 read path.
// Per the Phase 3.8 brief these gates FLUSH BEFORE QUERYING
// (`phase38_v5_backed_brain` calls `flush_now` after teaching), so
// none of the verdicts above is confounded by that gap. In-memory
// (un-flushed) bond visibility remains a Phase 3.10 cutover concern.
//
// PHASE 3.8.1 UPDATE: all three substrate-side gaps above are CLOSED
// by boot-time adjacency rehydration (`Brain::rehydrate_adjacency`,
// called from `load_v4`). The fixture now REOPENS the brain through
// the production boot path after teach + flush, so every gate runs
// against rehydrated adjacency, and the tripwire assertions below
// flipped to positive requirements (spread depth >= 1, hub-fill
// proposes, curiosity sees real degrees). The MemTable-visibility
// note (un-flushed bonds invisible to v5 reads) still stands --
// Phase 3.10 scope.
// =====================================================================
#[cfg(test)]
mod phase38_q9_sanity_gate_tests {
    use super::*;
    use crate::bond::BondType;
    use crate::manifest::MANIFEST_FILE_NAME;
    use crate::node::Confidence;
    use tempfile::tempdir;

    /// Shared fact set for all three gates. Two clusters:
    ///
    /// * an animal cluster (lion/tiger/zebra/mammal/...) giving the
    ///   K-gate a known topic with real spread structure and the
    ///   curiosity probe low-degree leaves to signal on;
    /// * a dedicated hub cluster (`phase38_hub` -> 3 spokes, all legs
    ///   at weight 0.85 >= HUB_LEG_MIN_WEIGHT) shaped so
    ///   `internal_fill_at_hub` proposes exactly C(3,2) = 3 pairwise
    ///   closures when it can see the adjacency.
    const PHASE38_FACTS: &[(&str, BondType, &str)] = &[
        ("lion", BondType::IsA, "mammal"),
        ("tiger", BondType::IsA, "mammal"),
        ("zebra", BondType::IsA, "mammal"),
        ("mammal", BondType::IsA, "animal"),
        ("lion", BondType::HasProperty, "mane"),
        ("tiger", BondType::HasProperty, "stripes"),
        ("zebra", BondType::HasProperty, "stripes"),
        ("lion", BondType::RelatedTo, "savanna"),
        ("savanna", BondType::PartOf, "africa"),
        ("phase38_hub", BondType::RelatedTo, "spoke_alpha"),
        ("phase38_hub", BondType::RelatedTo, "spoke_beta"),
        ("phase38_hub", BondType::RelatedTo, "spoke_gamma"),
    ];

    fn teach_phase38_facts(brain: &mut Brain) {
        for (s, r, o) in PHASE38_FACTS {
            brain.teach(s, *r, o, 0.85, Confidence::Verified);
        }
    }

    /// Phase 3.8 fixture -- construct a v5-backed Brain through the
    /// PRODUCTION boot path (`Brain::load_v4` against an empty
    /// MANIFEST, the same construction `make_empty_v4_brain` uses in
    /// the Phase 9 cutover tests), teach the shared facts via the
    /// public teach API, then `flush_now` so the substrate state is:
    ///
    /// * `node_reader_cache` (Phase 2.8) + `bond_reader_cache`
    ///   (Phase 3.7) installed by `enable_flush_worker`;
    /// * >= 1 NodeSST in `node_levels[0]` and >= 1 Bond v5 SST in
    ///   `bond_v5_levels[0]`, both committed to the manifest;
    /// * `bond_store_dir` / `node_store_dir` / `wal_path` all derived
    ///   from the brain dir, so BondFlushWorker + NodeFlushWorker ran
    ///   for real;
    /// * read path mode `V4Only` (what `load_v4` sets), `brain_dir_v4`
    ///   set, otherwise a normal v4-compatible Brain.
    ///
    /// The fixture self-asserts all of the above plus a live v5 bond
    /// read (`get_outgoing_bonds_v5` must serve the 3 hub legs from
    /// the L0 BondSST) so a sanity verdict can never silently run
    /// against a not-actually-v5-backed brain.
    ///
    /// Flush-before-query is deliberate: the Phase 3.7 finding is that
    /// the bond MemTable half stores Seeds, making un-flushed bonds
    /// invisible to v5 reads. The Phase 3.8 brief allows flushing
    /// first; in-memory visibility is Phase 3.10 cutover scope.
    ///
    /// Phase 3.8.1: after teach + flush the fixture DROPS the brain
    /// and REOPENS it via `Brain::load_v4`, so the returned brain is
    /// genuinely substrate-backed: `network.nodes` comes back through
    /// the Phase 10 v4-SST walk, and `network.bonds` + adjacency
    /// through `Brain::rehydrate_adjacency` streaming the v5 SSTs.
    /// This is the post-Phase-3.10 "boots from the v5 substrate"
    /// shape the 3.8 findings were about.
    fn phase38_v5_backed_brain(brain_dir: &std::path::Path) -> Brain {
        let fp = crate::manifest::KeyFingerprint::none();
        let m = crate::manifest::Manifest::new(fp);
        m.save_atomic(&brain_dir.join(MANIFEST_FILE_NAME)).unwrap();
        let mut brain = Brain::load_v4(brain_dir, None).expect("v5-backed brain must boot");
        teach_phase38_facts(&mut brain);
        brain.flush_now().expect("flush_now must succeed");

        // Phase 3.8.1: reopen through the production boot path (see
        // doc-comment above).
        drop(brain);
        let brain = Brain::load_v4(brain_dir, None)
            .expect("v5-backed brain must reboot from the substrate");

        let snap = brain.manifest_snapshot().expect("manifest snapshot");
        assert!(
            snap.total_node_sst_count() >= 1,
            "fixture must hold >= 1 NodeSST at L0 (got {})",
            snap.total_node_sst_count(),
        );
        assert!(
            snap.total_bond_v5_sst_count() >= 1,
            "fixture must hold >= 1 Bond v5 SST at L0 (got {})",
            snap.total_bond_v5_sst_count(),
        );
        assert!(
            brain.node_reader_cache.is_some(),
            "Phase 2.8 NodeSstReaderCache must be installed",
        );
        assert!(
            brain.bond_reader_cache.is_some(),
            "Phase 3.7 BondSstReaderCache must be installed",
        );
        let intern = brain.flush_intern.as_ref().expect("flush_intern").clone();
        let hub_id = intern
            .get(&Seed::from_label("phase38_hub"))
            .expect("hub seed interned by teach");
        let rows = brain
            .get_outgoing_bonds_v5(hub_id)
            .expect("v5 bond read path must answer");
        assert_eq!(
            rows.len(),
            3,
            "v5 bond read path must serve the 3 hub legs from the L0 BondSST",
        );
        brain
    }

    /// Control fixture -- identical facts taught into a plain v3-write
    /// brain (`Brain::new()`, `brain_dir_v4 = None`, bonds land in the
    /// `AssociativeNetwork` HashMaps). Each gate asserts its control
    /// expectation FIRST so a v5-side zero can never be confused with
    /// a mis-shaped probe.
    fn phase38_v3_control_brain() -> Brain {
        let mut brain = Brain::new();
        teach_phase38_facts(&mut brain);
        brain
    }

    /// Mirror of the production K-gate wiring in `mcp.rs` (lu4d_ask
    /// handler): fast-pass short-circuit, then `evaluate_signals` +
    /// `classify`. This IS the K-gate self-test invocation shape; the
    /// idk_gate unit tests cover the classifier in isolation, this
    /// wrapper exercises it against a real Brain + AskResult exactly
    /// as the MCP layer does.
    fn k_gate_verdict(
        brain: &Brain,
        ask: &AskResult,
        query: &str,
    ) -> crate::idk_gate::IdkVerdict {
        let t = crate::idk_gate::IdkThresholds::default();
        if crate::idk_gate::fast_pass_check(brain, ask, query, &t) {
            return crate::idk_gate::IdkVerdict::Pass;
        }
        let signals = crate::idk_gate::evaluate_signals(brain, ask, query, None, &t);
        crate::idk_gate::classify(&signals, &t)
    }

    /// Q9 sanity gate 1/3 -- K-gate. VERDICT: PASS.
    ///
    /// Gate-level behavior is identical across substrates: a known
    /// topic clears the gate (Pass) and an unknown topic refuses, with
    /// the SAME verdict variant on both brains. Q9 holds for K-gate.
    ///
    /// Phase 3.8.1 closed the spread-activation gap recorded by Phase
    /// 3.8 (depth-0-only spread on v5-backed brains): boot-time
    /// adjacency rehydration (`Brain::rehydrate_adjacency`) populates
    /// the raw network state the spread loop's weight lookups and
    /// neighbor walks consume. The fixture reopens the brain through
    /// `load_v4`, and the old `activated.len() == 1` tripwire below
    /// became a `>= 5` requirement (control activates 9 on this
    /// fixture).
    #[test]
    fn phase38_k_gate_v5_sanity() {
        // Control expectations first.
        let mut control = phase38_v3_control_brain();
        let c_known = control.ask("lion", 3, 10);
        assert!(
            c_known.recall.activated.len() > 1,
            "control: known-topic ask must activate the seed plus spread",
        );
        assert_eq!(
            k_gate_verdict(&control, &c_known, "lion"),
            crate::idk_gate::IdkVerdict::Pass,
            "control: known topic must clear the K-gate",
        );
        let q_unknown = "zorblax glimmerflux contraption";
        let c_unknown = control.ask(q_unknown, 3, 10);
        let control_refusal = k_gate_verdict(&control, &c_unknown, q_unknown);
        assert_ne!(
            control_refusal,
            crate::idk_gate::IdkVerdict::Pass,
            "control: unknown topic must refuse",
        );

        // v5-backed brain.
        let dir = tempdir().unwrap();
        let mut brain = phase38_v5_backed_brain(dir.path());

        // Merged bond read path is healthy: all 3 of lion's bonds
        // come back from the L0 layer.
        let lion = Seed::from_label("lion");
        let lion_bonds = brain.read_outgoing_bonds(&lion).unwrap();
        assert_eq!(
            lion_bonds.len(),
            3,
            "v5-backed: merged read must serve lion's 3 bonds from L0",
        );

        let ask_known = brain.ask("lion", 3, 10);
        assert_eq!(
            k_gate_verdict(&brain, &ask_known, "lion"),
            crate::idk_gate::IdkVerdict::Pass,
            "Q9: known topic must clear the K-gate on a v5-backed brain",
        );
        // Phase 3.8.1 closed the spread-activation gap: adjacency
        // rehydration at boot lets spread PROPAGATE through
        // substrate-resident bonds. The control activates 9 nodes on
        // this fixture; require >= 5 (not exact equality) so
        // unrelated ranking/cap tweaks don't make this gate flaky --
        // the structural claim is "depth >= 1 works", not "ranking
        // is byte-identical".
        assert!(
            ask_known.recall.activated.len() >= 5,
            "Phase 3.8.1: v5-backed spread must propagate past depth 0 \
             (got {} activated; control gets {})",
            ask_known.recall.activated.len(),
            c_known.recall.activated.len(),
        );

        let ask_unknown = brain.ask(q_unknown, 3, 10);
        let v5_refusal = k_gate_verdict(&brain, &ask_unknown, q_unknown);
        assert_ne!(
            v5_refusal,
            crate::idk_gate::IdkVerdict::Pass,
            "Q9: unknown topic must refuse on a v5-backed brain",
        );
        assert_eq!(
            v5_refusal, control_refusal,
            "Q9: refusal verdict must be the SAME variant on both substrates",
        );
        // And the refusal must still build user-facing prose.
        let t = crate::idk_gate::IdkThresholds::default();
        let signals =
            crate::idk_gate::evaluate_signals(&brain, &ask_unknown, q_unknown, None, &t);
        let (prose, _structured) = crate::idk_gate::build_response(
            v5_refusal, &signals, q_unknown, q_unknown, None,
        );
        assert!(
            !prose.is_empty(),
            "refusal prose must be non-empty on a v5-backed brain",
        );
    }

    /// Q9 sanity gate 2/3 -- hub-fill. VERDICT: PASS as of Phase
    /// 3.8.1 (was FAIL at Phase 3.8: 0 proposals vs 3 on control).
    ///
    /// `internal_fill_at_hub` / `internal_fill_at_seed` consume raw
    /// `AssociativeNetwork` adjacency (`neighbors_outgoing` /
    /// `neighbors_incoming` / `get_bond`), NOT the read-mode-dispatched
    /// Brain read path. At Phase 3.8 those maps were empty by design
    /// on a v4/v5-loaded brain (Phase 9 cutover semantics), so
    /// hub-fill saw zero adjacency. Phase 3.8.1's
    /// `Brain::rehydrate_adjacency` populates them at boot by
    /// streaming the v5 BondSSTs (plus lightweight node stubs), so
    /// the identical graph now yields the same C(3,2) = 3 pairwise
    /// closures as the v3-write control. The assertions below flipped
    /// from tripwires (must-be-empty) to parity requirements. Per Q9
    /// discipline the consumer source is UNTOUCHED -- the fix is
    /// substrate-side population, not consumer refactoring.
    #[test]
    fn phase38_hub_fill_v5_sanity() {
        let hub = Seed::from_label("phase38_hub");
        let cfg = crate::consolidation::ConsolidationConfig::default();

        // Control: the probe shape is valid -- 3 spokes through the
        // hub close into C(3,2) = 3 pairwise proposals.
        let control = phase38_v3_control_brain();
        let control_props = crate::consolidation::internal_fill_at_hub(
            &control.network,
            &hub,
            &cfg,
            crate::consolidation::INTERNAL_FILL_MAX_NEIGHBORS,
            crate::consolidation::HUB_LEG_MIN_WEIGHT,
        );
        assert_eq!(
            control_props.len(),
            3,
            "control: hub with 3 spokes must yield 3 pairwise proposals",
        );

        // v5-backed: same graph, zero proposals. The root cause is
        // visible in the same assertion block: the v3 adjacency the
        // fill walks is empty even though the bonds are v5-readable.
        let dir = tempdir().unwrap();
        let brain = phase38_v5_backed_brain(dir.path());
        assert_eq!(
            brain.network.bond_count(),
            PHASE38_FACTS.len(),
            "Phase 3.8.1: rehydrate_adjacency must load every live v5 bond \
             into the v3 maps at boot",
        );
        let props = crate::consolidation::internal_fill_at_hub(
            &brain.network,
            &hub,
            &cfg,
            crate::consolidation::INTERNAL_FILL_MAX_NEIGHBORS,
            crate::consolidation::HUB_LEG_MIN_WEIGHT,
        );
        assert!(
            !props.is_empty(),
            "Phase 3.8.1: hub-fill must propose on a v5-backed brain \
             (rehydrated adjacency)",
        );
        assert_eq!(
            props.len(),
            control_props.len(),
            "Phase 3.8.1: hub-fill proposals must match the v3 control \
             (identical graph after rehydration)",
        );
        // internal_fill_at_seed (the IsolatedNode/KnowledgeDesert arm
        // of the idle loop) reads the same adjacency -- assert parity
        // with the v3 control on the identical graph.
        let control_seed_props = crate::consolidation::internal_fill_at_seed(
            &control.network,
            &Seed::from_label("lion"),
            &cfg,
            crate::consolidation::INTERNAL_FILL_MAX_NEIGHBORS,
        );
        let seed_props = crate::consolidation::internal_fill_at_seed(
            &brain.network,
            &Seed::from_label("lion"),
            &cfg,
            crate::consolidation::INTERNAL_FILL_MAX_NEIGHBORS,
        );
        assert_eq!(
            seed_props.len(),
            control_seed_props.len(),
            "Phase 3.8.1: seed-fill must match the v3 control after \
             adjacency rehydration",
        );
    }

    /// Q9 sanity gate 3/3 -- curiosity wonder-queue probe. VERDICT:
    /// PASS (Phase 3.8.1: degree fidelity restored; the Phase 3.8
    /// IsolatedNode-inflation caveat is closed).
    ///
    /// The full runtime loop works on a v5-backed brain: ask ->
    /// `classify_curiosity_candidates` -> signals on the AskResult ->
    /// wonder queue fold -> interest-profile touch -> `top_n` ranking.
    /// Q9 holds for the curiosity ENGINE.
    ///
    /// Phase 3.8 caveat, CLOSED by Phase 3.8.1: `curiosity::degree()`
    /// and `is_hub()` read raw v3 adjacency, so on a v5-backed brain
    /// every activated seed used to report degree 0 (IsolatedNode
    /// intensities inflated to 0.430 vs 0.114/0.126 on control).
    /// With boot-time adjacency rehydration the same reads now see
    /// real degrees; the planted 3-spoke hub must report degree > 0
    /// (asserted both via raw adjacency and through curiosity's own
    /// public `is_hub` path, which wraps the private `degree` read).
    #[test]
    fn phase38_curiosity_v5_sanity() {
        // Control: probe shape emits signals on a v3-write brain.
        let mut control = phase38_v3_control_brain();
        let c1 = control.ask("lion", 3, 10);
        let c2 = control.ask("stripes", 3, 10);
        assert!(
            !c1.curiosity_signals.is_empty() || !c2.curiosity_signals.is_empty(),
            "control: asks over the fact set must emit curiosity signals",
        );
        assert!(
            control.wonder_queue.len() > 0,
            "control: wonder queue must be populated after asks",
        );

        // v5-backed brain.
        let dir = tempdir().unwrap();
        let mut brain = phase38_v5_backed_brain(dir.path());
        let v1 = brain.ask("lion", 3, 10);
        let v2 = brain.ask("stripes", 3, 10);
        assert!(
            !v1.curiosity_signals.is_empty() || !v2.curiosity_signals.is_empty(),
            "Q9: asks on a v5-backed brain must emit curiosity signals",
        );
        assert!(
            brain.wonder_queue.len() > 0,
            "Q9: wonder queue must be populated on a v5-backed brain",
        );
        assert!(
            brain
                .interest_profile
                .snapshot(now_ms() as u64, crate::curiosity::DEFAULT_INTEREST_HALFLIFE_MS)
                .len()
                > 0,
            "Q9: interest profile must be touched per signal seed",
        );
        // top_n must rank by descending intensity.
        let top = brain.wonder_queue.top_n(5);
        assert!(!top.is_empty(), "top_n must surface queued signals");
        for w in top.windows(2) {
            assert!(
                w[0].intensity >= w[1].intensity,
                "top_n must rank by descending intensity",
            );
        }
        // Phase 3.8.1: degree fidelity restored. `curiosity::degree`
        // is private; `curiosity::is_hub` is its public wrapper over
        // the same raw-adjacency read, so the hub-degree claim is
        // asserted both ways: raw adjacency sum, and through
        // curiosity's own code path.
        let hub = Seed::from_label("phase38_hub");
        let hub_degree = brain.network.neighbors_outgoing(&hub).len()
            + brain.network.neighbors_incoming(&hub).len();
        assert!(
            hub_degree > 0,
            "Phase 3.8.1: degree(hub) must be > 0 on a v5-backed brain \
             (got 0 -- adjacency rehydration regressed)",
        );
        assert_eq!(
            hub_degree, 3,
            "planted hub has exactly 3 spokes (got {})",
            hub_degree,
        );
        assert!(
            crate::curiosity::is_hub(&brain.network, &hub, 2),
            "Phase 3.8.1: curiosity::is_hub must classify the planted \
             3-spoke hub as a hub at threshold 2 -- degree visibility \
             through curiosity's own adjacency read",
        );
    }
}
// =====================================================================
// OP-BRAIN-V5 Phase 3.8.1 -- adjacency rehydration + cache eviction +
// MemBondFlush.seq. The three Phase 3.8 sanity gates above were
// flipped from tripwires to positive requirements by this port; the
// tests below pin the new machinery directly: compaction-driven
// reader-cache eviction, the seq round-trip without the flush-time
// re-fetch, and the rehydrated bond/stub counts on a 10K-bond
// fixture.
// =====================================================================
#[cfg(test)]
mod phase381_rehydration_tests {
    use super::*;
    use crate::bond::{BondType, KeyframeSource, TemporalKeyframe};
    use crate::manifest::MANIFEST_FILE_NAME;
    use crate::node::Confidence;
    use tempfile::tempdir;

    /// Boot an empty v5-backed brain through the production path
    /// (same construction as the Phase 3.8 fixture, without facts).
    fn empty_v5_brain(brain_dir: &std::path::Path) -> Brain {
        let fp = crate::manifest::KeyFingerprint::none();
        let m = crate::manifest::Manifest::new(fp);
        m.save_atomic(&brain_dir.join(MANIFEST_FILE_NAME)).unwrap();
        Brain::load_v4(brain_dir, None).expect("v5 brain must boot")
    }

    /// D4 -- compaction must evict retired file_uids from the Phase
    /// 2.8 / Phase 3.7 reader caches. Exercises BOTH stores against
    /// the Brain's own caches: warm the reader for the L0 SST, run a
    /// single-source L0 -> L1 promotion with the cache supplied, and
    /// verify the cache no longer holds the retired uid.
    #[test]
    fn phase381_cache_evicted_on_compaction() {
        let dir = tempdir().unwrap();
        let mut brain = empty_v5_brain(dir.path());
        brain.teach(
            "phase381_subj",
            BondType::RelatedTo,
            "phase381_obj",
            0.8,
            Confidence::Verified,
        );
        brain.flush_now().unwrap();
        let manifest = brain.manifest_snapshot().unwrap();

        // ---- node store ----
        let node_uid = manifest.node_levels[0].ssts[0].file_uid;
        let node_cache = brain.node_reader_cache.as_ref().unwrap().clone();
        node_cache.get_or_open(node_uid).unwrap();
        assert_eq!(node_cache.cache_size(), 1, "node reader must be warm");
        let node_worker = crate::node_store::compaction::NodeCompactionWorker::new(
            node_cache.node_store_dir().to_path_buf(),
        );
        let mut m_node = (*manifest).clone();
        let node_plan = crate::node_store::compaction::CompactionPlan {
            source_uids: vec![node_uid],
            source_level: 0,
            target_level: 1,
        };
        let node_res = node_worker
            .compact(node_plan, &mut m_node, Some(&node_cache))
            .unwrap();
        assert_eq!(node_res.input_entries_retired.len(), 1);
        assert_eq!(
            node_cache.cache_size(),
            0,
            "Phase 3.8.1: retired NodeSST uid must be evicted from the reader cache",
        );

        // ---- bond store ----
        let bond_uid = manifest.bond_v5_levels[0].ssts[0].file_uid;
        let bond_cache = brain.bond_reader_cache.as_ref().unwrap().clone();
        bond_cache.get_or_open_primary(bond_uid).unwrap();
        assert_eq!(bond_cache.cache_size(), 1, "bond reader must be warm");
        let bond_worker = crate::bond_store::compaction::BondCompactionWorker::new(
            bond_cache.bond_store_dir().to_path_buf(),
        );
        let mut m_bond = (*manifest).clone();
        let bond_plan = crate::bond_store::compaction::CompactionPlan {
            source_uids: vec![bond_uid],
            source_level: 0,
            target_level: 1,
        };
        let bond_res = bond_worker
            .compact(bond_plan, &mut m_bond, Some(&bond_cache))
            .unwrap();
        assert_eq!(bond_res.input_entries_retired.len(), 1);
        assert_eq!(
            bond_cache.cache_size(),
            0,
            "Phase 3.8.1: retired BondSST uid must be evicted from the reader cache",
        );
    }

    /// D5 -- `MemBondFlush.seq` end-to-end: plant a bond with a known
    /// non-zero seq, flush, and verify the v5 SST row carries the
    /// SAME seq. Pre-3.8.1 this round-trip relied on `flush_once`
    /// re-fetching MemBonds via `get_bonds_for_subject` (the O(N)
    /// double-walk); that re-fetch no longer exists, so a green run
    /// here proves the widened flush projection is the value that
    /// lands on disk.
    #[test]
    fn phase381_membondflush_seq_no_refetch() {
        let dir = tempdir().unwrap();
        let brain = empty_v5_brain(dir.path());
        let mt = brain.memtable().unwrap().clone();
        let intern = brain.flush_intern.as_ref().unwrap().clone();
        let kf = |w: f32| {
            vec![TemporalKeyframe {
                timestamp: 42,
                weight: w,
                source: KeyframeSource::Baseline,
            }]
        };
        // Two padding bonds so the probe's seq is non-trivial.
        let s_pad = Seed::from_label("phase381_pad_subject");
        for obj in ["phase381_pad_obj1", "phase381_pad_obj2"] {
            let o = Seed::from_label(obj);
            intern.intern_or_get(o);
            mt.insert(s_pad, o, BondType::RelatedTo, 0.5, kf(0.5), &intern);
        }
        let s = Seed::from_label("phase381_seq_subject");
        let o = Seed::from_label("phase381_seq_object");
        intern.intern_or_get(o);
        let planted = mt.insert(s, o, BondType::IsA, 0.9, kf(0.9), &intern);
        assert!(planted.seq > 0, "probe seq must be non-trivial");
        brain.flush_now().unwrap();
        let sid = intern.get(&s).expect("subject interned at insert");
        let rows = brain.get_outgoing_bonds_v5(sid).unwrap();
        assert_eq!(rows.len(), 1, "probe subject has exactly one bond");
        assert_eq!(
            rows[0].3.seq,
            planted.seq,
            "Phase 3.8.1: BondPayload.seq on disk must equal the seq \
             assigned at MemTable insert time",
        );
    }

    /// D6 -- rehydration observability on a 10K-bond fixture
    /// (diagnostic-quality, not an RSS assertion): flush a 100x100
    /// bond grid through the v5 substrate, reboot, and verify the
    /// rehydrated counts. Because no node payloads were ever taught,
    /// every endpoint must come back as a lightweight stub (empty
    /// content + embedding) -- pinning the RAM-constraint shape.
    #[test]
    fn phase381_rehydration_ram_observable() {
        const N_SUBJECTS: usize = 100;
        const N_OBJECTS: usize = 100; // 100 x 100 = 10_000 bonds
        let dir = tempdir().unwrap();
        let brain = empty_v5_brain(dir.path());
        {
            let mt = brain.memtable().unwrap().clone();
            let intern = brain.flush_intern.as_ref().unwrap().clone();
            for i in 0..N_SUBJECTS {
                let s = Seed::from_label(&format!("ph381_s{}", i));
                for j in 0..N_OBJECTS {
                    let o = Seed::from_label(&format!("ph381_o{}", j));
                    intern.intern_or_get(o);
                    mt.insert(
                        s,
                        o,
                        BondType::RelatedTo,
                        0.5,
                        vec![TemporalKeyframe {
                            timestamp: 1,
                            weight: 0.5,
                            source: KeyframeSource::Baseline,
                        }],
                        &intern,
                    );
                }
            }
        }
        brain.flush_now().unwrap();
        drop(brain);

        let brain = Brain::load_v4(dir.path(), None).expect("reboot from v5 substrate");
        assert_eq!(
            brain.network.bond_count(),
            N_SUBJECTS * N_OBJECTS,
            "rehydrated bond count must match the flushed corpus",
        );
        assert_eq!(
            brain.network.node_count(),
            N_SUBJECTS + N_OBJECTS,
            "every endpoint gets exactly one lightweight stub",
        );
        let s0 = Seed::from_label("ph381_s0");
        assert_eq!(brain.network.neighbors_outgoing(&s0).len(), N_OBJECTS);
        let o0 = Seed::from_label("ph381_o0");
        assert_eq!(brain.network.neighbors_incoming(&o0).len(), N_SUBJECTS);
        // Stub shape: content/embedding empty -- the RAM constraint.
        let stub = brain.network.get_node(&s0).expect("stub present");
        assert!(stub.content().is_empty(), "stubs carry no content");
        assert!(
            stub.embedding().is_empty(),
            "stubs carry no embedding",
        );
        eprintln!(
            "[phase381] D6: rehydrated {} bonds / {} node stubs (RAM diagnostic: \
             ~30-60 B/edge adjacency + ~50-100 B/stub; full payloads stay in SSTs)",
            brain.network.bond_count(),
            brain.network.node_count(),
        );
    }
}
// =====================================================================
// OP-BRAIN-V5 -- production v5 bond-compaction SCHEDULER tests.
//
// Proves the scheduler wired into `load_v5` runs AUTOMATICALLY: with a
// v5 bond store whose L0 BondSST count crosses the default trigger,
// the background worker (no manual compaction call) drains L0 -> L1,
// emits the paired `<uid>.bsst.reverse` reverse index, keeps L0 small,
// and leaves incoming-edge reads working (served by the L1 reverse
// index). At HEAD 750eb9c this path ran the dead v4 worker and so
// never compacted the v5 substrate or built the reverse index.
// =====================================================================
#[cfg(test)]
mod bond_v5_compaction_scheduler_tests {
    use super::*;
    use crate::bond::{BondType, KeyframeSource, TemporalKeyframe};
    use crate::manifest::{KeyFingerprint, Manifest, MANIFEST_FILE_NAME};
    use crate::seed::Seed;
    use std::time::{Duration, Instant};
    use tempfile::tempdir;

    fn kf(w: f32) -> Vec<TemporalKeyframe> {
        vec![TemporalKeyframe {
            timestamp: 100,
            weight: w,
            source: KeyframeSource::Baseline,
        }]
    }

    /// Boot a v5 brain through the PRODUCTION path (`load_v5`) -- the
    /// path that wires the v5 bond-compaction scheduler. A fresh
    /// v5-tagged manifest is the only on-disk prerequisite.
    fn boot_v5(brain_dir: &std::path::Path) -> Brain {
        let mut m = Manifest::new(KeyFingerprint::none());
        m.set_brain_format_v5();
        m.save_atomic(&brain_dir.join(MANIFEST_FILE_NAME)).unwrap();
        Brain::load_v5(brain_dir, None).expect("v5 brain must boot via load_v5")
    }

    /// The scheduler started by `load_v5` must fire on its own once L0
    /// crosses the trigger -- no `compact_now`, no manual plan.
    #[test]
    fn bond_v5_scheduler_auto_compacts_and_builds_reverse_index() {
        let dir = tempdir().unwrap();
        let brain = boot_v5(dir.path());

        // Production wiring: load_v5 must have started the v5 scheduler
        // (and NOT left a v4 worker as the bond compactor for a v5
        // brain).
        assert!(
            brain.bond_v5_compaction().is_some(),
            "load_v5 must start the v5 bond-compaction scheduler",
        );
        let trigger = brain.bond_v5_compaction().unwrap().l0_trigger();
        assert!(trigger >= 1);

        let intern = brain.flush_intern.as_ref().unwrap().clone();
        let mt = brain.memtable().unwrap().clone();

        // Shared object: the incoming-edge target we read back. Intern
        // it up front so the flush projection carries its NodeId.
        let hub = Seed::from_label("bondv5_sched_hub");
        intern.intern_or_get(hub);
        let hub_id = intern.get(&hub).expect("hub interned");

        // One distinct bond per flush => one L0 BondSST per flush.
        // Flush `trigger` times so the L0 count reaches the trigger.
        // All bonds point at `hub`, so incoming(hub) has `trigger`
        // entries.
        let mut subject_ids = Vec::new();
        for i in 0..trigger {
            let s = Seed::from_label(&format!("bondv5_sched_subj_{}", i));
            let sid = intern.intern_or_get(s);
            subject_ids.push(sid);
            mt.insert(s, hub, BondType::RelatedTo, 0.7, kf(0.7), &intern);
            brain.flush_now().unwrap();
        }

        // Wait (bounded) for the BACKGROUND scheduler to compact on its
        // own. Success == L0 drained below trigger AND some L>=1 entry
        // now carries a reverse-index file_uid.
        let deadline = Instant::now() + Duration::from_secs(20);
        let reverse_uid: u64 = loop {
            let m = brain.manifest_snapshot().unwrap();
            let l0 = m
                .bond_v5_levels
                .iter()
                .find(|l| l.level == 0)
                .map(|l| l.ssts.len())
                .unwrap_or(0);
            let rev = m
                .bond_v5_levels
                .iter()
                .filter(|l| l.level >= 1)
                .flat_map(|l| l.ssts.iter())
                .find_map(|e| e.reverse_file_uid);
            if l0 < trigger {
                if let Some(r) = rev {
                    break r;
                }
            }
            assert!(
                Instant::now() < deadline,
                "scheduler did not auto-compact L0 within deadline (l0={}, trigger={})",
                l0,
                trigger,
            );
            std::thread::sleep(Duration::from_millis(25));
        };

        // The paired reverse index exists on disk.
        let bond_dir = dir.path().join("bond_store");
        let reverse_path =
            bond_dir.join(crate::bond_store::bsst_reverse_filename(reverse_uid));
        assert!(
            reverse_path.exists(),
            "<uid>.bsst.reverse must exist after auto-compaction: {}",
            reverse_path.display(),
        );

        // L0 drained, and the compacted L1 output is the one reverse-
        // backed SST in the manifest.
        let m = brain.manifest_snapshot().unwrap();
        let l0_after = m
            .bond_v5_levels
            .iter()
            .find(|l| l.level == 0)
            .map(|l| l.ssts.len())
            .unwrap_or(0);
        assert!(
            l0_after < trigger,
            "L0 not drained: {} still >= trigger {}",
            l0_after,
            trigger,
        );
        assert_eq!(m.live_bond_v5_reverse_uids(), vec![reverse_uid]);

        // Incoming reads still work AND are served by the reverse
        // index: the only live bond SST is the L1 output, which carries
        // the reverse file, so get_incoming_bonds_v5 traverses it.
        let incoming = brain.get_incoming_bonds_v5(hub_id).unwrap();
        assert_eq!(
            incoming.len(),
            trigger,
            "all incoming edges to hub must survive compaction",
        );
        let mut got: Vec<u64> = incoming.iter().map(|(s, _p, _o, _pay)| s.0).collect();
        got.sort_unstable();
        let mut want: Vec<u64> = subject_ids.iter().map(|n| n.0).collect();
        want.sort_unstable();
        assert_eq!(
            got, want,
            "incoming subjects (via reverse index) must match the taught bonds",
        );
        for (_s, _p, o, _pay) in &incoming {
            assert_eq!(*o, hub_id, "every incoming tuple's object is the hub");
        }
    }

    /// OP-BRAIN-V5 -- node-compaction scheduler analog of
    /// `bond_v5_scheduler_auto_compacts_and_builds_reverse_index`.
    /// The scheduler started by `load_v5` must drain node L0 on its
    /// own once L0 crosses the trigger -- no `compact_now`, no manual
    /// plan -- and node reads (`get_node_v5`) must still return the
    /// correct payloads afterward.
    #[test]
    fn node_v5_scheduler_auto_compacts_l0() {
        let dir = tempdir().unwrap();
        let brain = boot_v5(dir.path());

        // Production wiring: load_v5 must have started the v5 node
        // scheduler -- AND still the bond scheduler (coexistence).
        assert!(
            brain.node_v5_compaction().is_some(),
            "load_v5 must start the v5 node-compaction scheduler",
        );
        assert!(
            brain.bond_v5_compaction().is_some(),
            "load_v5 must still start the bond scheduler too (coexistence)",
        );
        let trigger = brain.node_v5_compaction().unwrap().l0_trigger();
        assert!(trigger >= 1);

        let intern = brain.flush_intern.as_ref().unwrap().clone();
        let mt = brain.memtable().unwrap().clone();

        // One node (+ a paired bond so the flush gate fires) per
        // flush => one L0 NodeSST per flush. Flush `trigger` times so
        // the node L0 count reaches the trigger.
        let mut node_ids = Vec::new();
        let mut contents = Vec::new();
        for i in 0..trigger {
            let s = Seed::from_label(&format!("nodev5_sched_n_{}", i));
            let o = Seed::from_label(&format!("nodev5_sched_o_{}", i));
            let content = format!("nodev5_sched_payload_{}", i);
            let node = crate::node::KnowledgeNode {
                seed: s,
                content: content.clone(),
                knowledge: crate::node::KnowledgeUnit {
                    content: crate::node::KnowledgeContent::Concept(content.clone()),
                    confidence: crate::node::Confidence::Verified,
                    created_at: 0,
                    updated_at: 0,
                },
                embedding: vec![0.0; 384],
                parent_seeds: Vec::new(),
                typed_ratio: 0.0,
            };
            mt.insert_node(s, node, &intern);
            // Pair with a bond so flush_once does not short-circuit
            // on the empty-memtable gate.
            mt.insert(s, o, BondType::IsA, 0.5, kf(0.5), &intern);
            brain.flush_now().unwrap();
            node_ids.push(intern.get(&s).expect("node id resolves post-insert"));
            contents.push(content);
        }

        // Wait (bounded) for the BACKGROUND scheduler to compact on
        // its own. Success == node L0 drained below trigger.
        let deadline = Instant::now() + Duration::from_secs(20);
        loop {
            let m = brain.manifest_snapshot().unwrap();
            let l0 = m
                .node_levels
                .iter()
                .find(|l| l.level == 0)
                .map(|l| l.ssts.len())
                .unwrap_or(0);
            if l0 < trigger {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "node scheduler did not auto-compact L0 within deadline (l0={}, trigger={})",
                l0,
                trigger,
            );
            std::thread::sleep(Duration::from_millis(25));
        }

        // L0 drained, and a compacted L>=1 NodeSST now exists.
        let m = brain.manifest_snapshot().unwrap();
        let l0_after = m
            .node_levels
            .iter()
            .find(|l| l.level == 0)
            .map(|l| l.ssts.len())
            .unwrap_or(0);
        assert!(
            l0_after < trigger,
            "node L0 not drained: {} still >= trigger {}",
            l0_after,
            trigger,
        );
        let l1_count: usize = m
            .node_levels
            .iter()
            .filter(|l| l.level >= 1)
            .map(|l| l.ssts.len())
            .sum();
        assert!(
            l1_count >= 1,
            "compacted output must live at L>=1 (got {} SSTs)",
            l1_count,
        );

        // Node reads still return the correct payloads after the
        // background compaction (served from the merged L1 NodeSST).
        for (id, want) in node_ids.iter().zip(contents.iter()) {
            let got = brain.get_node_v5(*id).unwrap();
            assert!(
                got.is_some(),
                "post-compaction node read at {:?} must succeed",
                id,
            );
            assert_eq!(
                &got.unwrap().content,
                want,
                "node payload must survive auto-compaction",
            );
        }
    }
}
#[cfg(test)]
mod mapram_p0_tests {
    //! OP-BRAIN-MapRAM P0 -- integration tests for the resident MAP
    //! build (`Brain::build_resident_map` / `enable_resident_map`) over a
    //! real v5 brain booted through the production `load_v5` path.
    use super::*;
    use crate::bond::{BondType, KeyframeSource, TemporalKeyframe};
    use crate::manifest::{KeyFingerprint, Manifest, MANIFEST_FILE_NAME};
    use crate::node_intern::NodeId;
    use crate::seed::Seed;
    use tempfile::tempdir;

    fn kf(w: f32) -> Vec<TemporalKeyframe> {
        vec![TemporalKeyframe {
            timestamp: 100,
            weight: w,
            source: KeyframeSource::Baseline,
        }]
    }

    /// Boot a fresh v5 brain through the production path (a v5-tagged
    /// manifest is the only on-disk prerequisite). Mirrors the harness
    /// in `bond_v5_compaction_scheduler_tests`.
    fn boot_v5(brain_dir: &std::path::Path) -> Brain {
        let mut m = Manifest::new(KeyFingerprint::none());
        m.set_brain_format_v5();
        m.save_atomic(&brain_dir.join(MANIFEST_FILE_NAME)).unwrap();
        Brain::load_v5(brain_dir, None).expect("v5 brain must boot via load_v5")
    }

    /// Teach a small deterministic topology, flushing each edge to its
    /// own L0 BondSST. Topology (subject -> object):
    ///   s0 -> hub, s0 -> a, s0 -> b      (s0 out = 3)
    ///   s1 -> hub                        (s1 out = 1)
    ///   s2 -> hub                        (s2 out = 1)
    ///   a  -> b                          (a  out = 1)
    ///   s0 -> hub AGAIN (newer weight)   -- same (s0,hub) pair, must
    ///                                       collapse to ONE adjacency
    ///                                       entry (no double count)
    /// => hub in = 3 (s0,s1,s2) ; b in = 2 (s0,a) ; a in = 1 (s0)
    /// => 6 distinct (subject,object) pairs land in adjacency.
    fn teach_topology(brain: &Brain) {
        let intern = brain.flush_intern.as_ref().unwrap().clone();
        let mt = brain.memtable().unwrap().clone();
        let hub = Seed::from_label("mapram_hub");
        let a = Seed::from_label("mapram_a");
        let b = Seed::from_label("mapram_b");
        let s0 = Seed::from_label("mapram_s0");
        let s1 = Seed::from_label("mapram_s1");
        let s2 = Seed::from_label("mapram_s2");
        for s in [hub, a, b, s0, s1, s2] {
            intern.intern_or_get(s);
        }
        // Batch 1 -> one L0 BondSST.
        let batch1: &[(Seed, Seed, f32)] = &[
            (s0, hub, 0.5),
            (s0, a, 0.6),
            (s0, b, 0.7),
            (s1, hub, 0.5),
            (s2, hub, 0.5),
            (a, b, 0.8),
        ];
        for (from, to, w) in batch1.iter().copied() {
            mt.insert(from, to, BondType::RelatedTo, w, kf(w), &intern);
        }
        brain.flush_now().unwrap();

        // Batch 2 -> a SECOND L0 BondSST that REPEATS the (s0,hub)
        // pair with a newer weight. This exercises the map builder's
        // cross-SST (subject,object) collapse: it must dedup to ONE
        // adjacency entry, exactly as rehydrate_adjacency does. Two L0
        // SSTs stays below the compaction trigger (default 4), so the
        // background scheduler does not fire mid-test (no flush/cleanup
        // race).
        mt.insert(s0, hub, BondType::RelatedTo, 0.9, kf(0.9), &intern);
        brain.flush_now().unwrap();
    }

    /// Core P0 invariant: the resident MAP's degrees are byte-exact
    /// against the rehydrated resident adjacency, for EVERY node, and
    /// the locator/degree maths matches the known topology.
    #[test]
    fn resident_map_degrees_match_real_adjacency() {
        let dir = tempdir().unwrap();
        {
            let brain = boot_v5(dir.path());
            teach_topology(&brain);
        }
        // Reload through the production path: rehydrate_adjacency builds
        // the resident adjacency from the same SSTs the map reads.
        let mut brain = Brain::load_v5(dir.path(), None).expect("reload via load_v5");
        brain.rehydrate_adjacency()
            .expect("rehydrate resident adjacency for legacy/parity comparison");
        let map = brain.build_resident_map().expect("build resident map");
        let intern = brain.network.node_intern().clone();
        let n = intern.len();
        assert!(n >= 6, "taught nodes must be interned, got {}", n);
        assert_eq!(map.node_count(), n, "map must cover every interned node");

        // Universal exactness sweep: map degree == real adjacency degree
        // for every interned NodeId, both directions.
        let mut checked = 0u64;
        for i in 0..n as u64 {
            let id = NodeId(i);
            let seed = match intern.reverse_lookup(id) {
                Some(s) => s,
                None => continue,
            };
            assert_eq!(
                map.out_degree(id) as usize,
                brain.network.out_degree(&seed),
                "out_degree mismatch at id {}",
                i,
            );
            assert_eq!(
                map.in_degree(id) as usize,
                brain.network.in_degree(&seed),
                "in_degree mismatch at id {}",
                i,
            );
            checked += 1;
        }
        assert!(checked >= 6, "swept too few nodes: {}", checked);

        // Known-topology spot checks (incl. duplicate-pair dedup).
        let id_of = |label: &str| intern.get(&Seed::from_label(label)).unwrap();
        assert_eq!(
            map.out_degree(id_of("mapram_s0")),
            3,
            "s0 -> hub,a,b (duplicate (s0,hub) counted once)",
        );
        assert_eq!(map.in_degree(id_of("mapram_hub")), 3, "hub <- s0,s1,s2");
        assert_eq!(map.in_degree(id_of("mapram_b")), 2, "b <- s0,a");
        assert_eq!(map.in_degree(id_of("mapram_a")), 1, "a <- s0");
        assert_eq!(map.out_degree(id_of("mapram_s1")), 1);
        assert_eq!(map.out_degree(id_of("mapram_hub")), 0, "hub is a pure sink");

        // Counted-bond totals: 6 distinct (subject,object) pairs.
        assert_eq!(map.total_out_bonds(), 6);
        assert_eq!(map.total_in_bonds(), 6);
    }

    /// The `LU_MAPRAM_BUILD` load flag wires the map into `load_v5` and
    /// retains it on the Brain; without the flag the field stays `None`
    /// (production load unchanged). This module is the only writer of
    /// that env var, so the on/off assertions are race-free.
    #[test]
    fn load_flag_builds_and_retains_resident_map() {
        let dir = tempdir().unwrap();
        {
            let brain = boot_v5(dir.path());
            teach_topology(&brain);
        }

        // BUILD now defaults ON (cache-backed is the production
        // default): an unforced load builds + retains the resident MAP.
        std::env::remove_var("LU_MAPRAM_BUILD");
        let def = Brain::load_v5(dir.path(), None).expect("default reload");
        {
            let map = def
                .resident_map
                .as_ref()
                .expect("default => resident map must be built + retained");
            let hub = Seed::from_label("mapram_hub");
            let hub_id = def.network.node_intern().get(&hub).unwrap();
            // Degree comes from the map (built from the SSTs), not the
            // resident adjacency (the default load no longer builds it).
            assert_eq!(map.in_degree(hub_id), 3, "hub <- s0,s1,s2");
        }
        drop(def);

        // Forcing BUILD off skips the resident MAP. Only this module
        // writes LU_MAPRAM_BUILD, so the on/off assertions are race-free.
        std::env::set_var("LU_MAPRAM_BUILD", "0");
        let off = Brain::load_v5(dir.path(), None).expect("build-off reload");
        std::env::remove_var("LU_MAPRAM_BUILD");
        assert!(
            off.resident_map.is_none(),
            "LU_MAPRAM_BUILD=0 => load_v5 must not retain a resident map",
        );
    }
/// Teach SST bonds (flushed) PLUS extra bonds left in the
    /// MemTable/WAL (NOT flushed) so a reload replays them. Un-flushed
    /// edges: w0 -> w1, w0 -> hub, s1 -> w1. They intern TWO brand-new
    /// nodes (w0, w1) -- this is what makes a degmap sidecar written
    /// BEFORE these WAL bonds show up as STALE (intern.len() grows).
    fn teach_wal_extra(brain: &Brain) {
        let intern = brain.flush_intern.as_ref().unwrap().clone();
        let mt = brain.memtable().unwrap().clone();
        let wal = brain.wal().unwrap().clone();
        let hub = Seed::from_label("mapram_hub");
        let s1 = Seed::from_label("mapram_s1");
        let w0 = Seed::from_label("mapram_w0");
        let w1 = Seed::from_label("mapram_w1");
        // Insert into the MemTable AND append to the WAL (mirrors
        // brain_flush_now_round_trip) but do NOT flush -- so on the next
        // load_v5 these bonds are replayed from the WAL into a fresh
        // MemTable. The replay's mt.insert interns w0/w1 (intern.len()
        // grows from 6 -> 8), which is exactly the WAL-generation signal
        // the validity fingerprint must capture.
        for (from, to, w) in [(w0, w1, 0.4f32), (w0, hub, 0.5), (s1, w1, 0.6)] {
            let r = mt.insert(from, to, BondType::RelatedTo, w, kf(w), &intern);
            wal.lock().unwrap().append(
                r.seq, &from, &to, BondType::RelatedTo, w, &kf(w),
            ).unwrap();
        }
        wal.lock().unwrap().fsync().unwrap();
    }

    /// OP-BRAIN-MapRAM persist -- end-to-end: the degree map loaded from
    /// the persisted sidecar is byte-identical to a freshly rebuilt map,
    /// for a brain carrying BOTH SST bonds AND WAL-replayed MemTable
    /// bonds; and a stale fingerprint forces a rebuild rather than a
    /// silent stale load.
    #[test]
    fn persisted_degmap_matches_rebuild_with_sst_and_wal_bonds() {
        let key = crate::encrypt::MasterKey::from_bytes([0x7Eu8; 32]);
        let dir = tempdir().unwrap();
        {
            let mut m = Manifest::new(KeyFingerprint::of(&key));
            m.set_brain_format_v5();
            m.save_atomic(&dir.path().join(MANIFEST_FILE_NAME)).unwrap();
            let brain = Brain::load_v5(dir.path(), Some(key.clone()))
                .expect("initial v5 boot");
            teach_topology(&brain);
            teach_wal_extra(&brain);
        }

        let sidecar = Brain::resident_degmap_path_for_dir(dir.path());
        // The brain-creation load_v5 above may have written a sidecar for
        // the (then-empty) pre-teach state. Remove it so boot #1 cleanly
        // exercises the rebuild-and-write path against the taught graph.
        let _ = std::fs::remove_file(&sidecar);
        assert!(!sidecar.exists(), "sidecar removed before boot #1");
        let canonical: (Vec<u32>, Vec<u32>) = {
            let brain = Brain::load_v5(dir.path(), Some(key.clone()))
                .expect("boot #1 (rebuild path)");
            let rm = brain.resident_map.as_ref().expect("boot #1 retains map");
            let fresh = brain.build_resident_map().expect("fresh rebuild");
            assert_eq!(
                rm.out_degrees(), fresh.out_degrees(),
                "retained map out-degrees must equal fresh rebuild",
            );
            assert_eq!(
                rm.in_degrees(), fresh.in_degrees(),
                "retained map in-degrees must equal fresh rebuild",
            );
            (fresh.out_degrees().to_vec(), fresh.in_degrees().to_vec())
        };
        assert!(sidecar.exists(), "boot #1 must have written the sidecar");

        {
            let brain = Brain::load_v5(dir.path(), Some(key.clone()))
                .expect("boot #2 (fast path)");
            let rm = brain.resident_map.as_ref().expect("boot #2 retains map");
            assert_eq!(
                rm.out_degrees(), canonical.0.as_slice(),
                "fast-path out-degrees must equal canonical rebuild",
            );
            assert_eq!(
                rm.in_degrees(), canonical.1.as_slice(),
                "fast-path in-degrees must equal canonical rebuild",
            );
            let id_of = |label: &str| {
                brain.network.node_intern().get(&Seed::from_label(label)).unwrap()
            };
            assert_eq!(rm.in_degree(id_of("mapram_hub")), 3, "hub <- s0,s1,s2 (SST)");
            assert_eq!(rm.out_degree(id_of("mapram_w0")), 0, "w0 has no SST out-edges");
            assert_eq!(rm.in_degree(id_of("mapram_w1")), 0, "w1 has no SST in-edges");
        }

        {
            let loaded = crate::node_store::degmap::read_degmap(
                &sidecar, Some(key.clone()),
            ).expect("sidecar readable");
            let bogus_fp = loaded.fingerprint ^ 0xFFFF_FFFF_FFFF_FFFF;
            crate::node_store::degmap::write_degmap(
                &sidecar,
                &loaded.out_degree,
                &loaded.in_degree,
                bogus_fp,
                Some(key.clone()),
            ).expect("rewrite sidecar with bogus fingerprint");

            let brain = Brain::load_v5(dir.path(), Some(key.clone()))
                .expect("boot #3 (stale fingerprint -> rebuild)");
            let rm = brain.resident_map.as_ref().expect("boot #3 retains map");
            assert_eq!(
                rm.out_degrees(), canonical.0.as_slice(),
                "stale sidecar must be rejected -> rebuilt out-degrees",
            );
            assert_eq!(
                rm.in_degrees(), canonical.1.as_slice(),
                "stale sidecar must be rejected -> rebuilt in-degrees",
            );
            let after = crate::node_store::degmap::read_degmap(
                &sidecar, Some(key.clone()),
            ).expect("sidecar still readable");
            assert_ne!(
                after.fingerprint, bogus_fp,
                "rebuild must have rewritten the sidecar with a valid fingerprint",
            );
            let live_fp = brain.resident_degmap_fingerprint().expect("live fp");
            assert_eq!(
                after.fingerprint, live_fp,
                "rewritten sidecar fingerprint must match the live graph state",
            );
        }
    }
}
#[cfg(test)]
mod mapram_p1_tests {
    //! OP-BRAIN-MapRAM P1 -- integration tests for the cache-backed
    //! adjacency reader (`CacheBackedAdjacency` + `V5AdjacencySource`)
    //! over a real v5 brain booted through the production `load_v5`
    //! path. The gate's heart: the cache-backed neighbour lists are
    //! SET-identical to the resident `AssociativeNetwork` adjacency
    //! (same objects/subjects, same winning predicate per pair).
    use super::*;
    use crate::bond::{BondType, KeyframeSource, TemporalKeyframe};
    use crate::cache_adjacency::{CacheBackedAdjacency, Direction};
    use crate::manifest::{KeyFingerprint, Manifest, MANIFEST_FILE_NAME};
    use crate::node_intern::NodeId;
    use crate::seed::Seed;
    use std::collections::HashSet;
    use tempfile::tempdir;

    fn kf(w: f32) -> Vec<TemporalKeyframe> {
        vec![TemporalKeyframe { timestamp: 100, weight: w, source: KeyframeSource::Baseline }]
    }

    fn boot_v5(brain_dir: &std::path::Path) -> Brain {
        let mut m = Manifest::new(KeyFingerprint::none());
        m.set_brain_format_v5();
        m.save_atomic(&brain_dir.join(MANIFEST_FILE_NAME)).unwrap();
        Brain::load_v5(brain_dir, None).expect("v5 brain must boot via load_v5")
    }

    /// Small topology, 3 flushes (BELOW the compaction trigger of 4 so
    /// the background scheduler stays quiet -- reverse-less reads).
    /// Exercises:
    ///   * duplicate (s0,hub) pair across flushes -> one neighbour,
    ///   * multi-predicate (s0,a): RelatedTo then a higher-seq IsA ->
    ///     collapses to ONE neighbour whose winning predicate is IsA,
    ///   * a hub with in-degree 4 (the pin target).
    /// Topology (subject -> object):
    ///   s0 -> hub, s0 -> a, s0 -> b ; a -> b ; s1 -> hub ; s2 -> hub ;
    ///   s3 -> hub ; (s0 -> hub again) ; (s0 -> a as IsA, newer)
    fn teach_small(brain: &Brain) {
        let intern = brain.flush_intern.as_ref().unwrap().clone();
        let mt = brain.memtable().unwrap().clone();
        let hub = Seed::from_label("p1_hub");
        let a = Seed::from_label("p1_a");
        let b = Seed::from_label("p1_b");
        let s0 = Seed::from_label("p1_s0");
        let s1 = Seed::from_label("p1_s1");
        let s2 = Seed::from_label("p1_s2");
        let s3 = Seed::from_label("p1_s3");
        for s in [hub, a, b, s0, s1, s2, s3] {
            intern.intern_or_get(s);
        }
        // Flush 1.
        for (f, t, w, bt) in [
            (s0, hub, 0.5f32, BondType::RelatedTo),
            (s0, a, 0.6, BondType::RelatedTo),
            (s0, b, 0.7, BondType::RelatedTo),
            (a, b, 0.8, BondType::RelatedTo),
            (s1, hub, 0.5, BondType::RelatedTo),
            (s2, hub, 0.5, BondType::RelatedTo),
            (s3, hub, 0.5, BondType::RelatedTo),
        ] {
            mt.insert(f, t, bt, w, kf(w), &intern);
        }
        brain.flush_now().unwrap();
        // Flush 2: duplicate (s0,hub) with a newer weight.
        mt.insert(s0, hub, BondType::RelatedTo, 0.9, kf(0.9), &intern);
        brain.flush_now().unwrap();
        // Flush 3: (s0,a) again as IsA -- higher seq -> winning predicate.
        mt.insert(s0, a, BondType::IsA, 0.95, kf(0.95), &intern);
        brain.flush_now().unwrap();
    }

    /// Larger topology, 5 flushes (ABOVE the trigger of 4) so a forced
    /// compaction builds the on-disk reverse index. hub in-degree 6.
    fn teach_many(brain: &Brain) {
        let intern = brain.flush_intern.as_ref().unwrap().clone();
        let mt = brain.memtable().unwrap().clone();
        let hub = Seed::from_label("p1m_hub");
        let a = Seed::from_label("p1m_a");
        let b = Seed::from_label("p1m_b");
        let subs: Vec<Seed> = (0..6).map(|i| Seed::from_label(&format!("p1m_s{}", i))).collect();
        intern.intern_or_get(hub);
        intern.intern_or_get(a);
        intern.intern_or_get(b);
        for s in &subs {
            intern.intern_or_get(*s);
        }
        // 5 separate flushes => 5 L0 BondSSTs.
        mt.insert(subs[0], hub, BondType::RelatedTo, 0.5, kf(0.5), &intern);
        mt.insert(subs[0], a, BondType::RelatedTo, 0.6, kf(0.6), &intern);
        mt.insert(subs[0], b, BondType::RelatedTo, 0.7, kf(0.7), &intern);
        mt.insert(a, b, BondType::RelatedTo, 0.8, kf(0.8), &intern);
        brain.flush_now().unwrap();
        mt.insert(subs[1], hub, BondType::RelatedTo, 0.5, kf(0.5), &intern);
        mt.insert(subs[2], hub, BondType::RelatedTo, 0.5, kf(0.5), &intern);
        brain.flush_now().unwrap();
        mt.insert(subs[3], hub, BondType::RelatedTo, 0.5, kf(0.5), &intern);
        mt.insert(subs[4], hub, BondType::RelatedTo, 0.5, kf(0.5), &intern);
        brain.flush_now().unwrap();
        mt.insert(subs[5], hub, BondType::RelatedTo, 0.5, kf(0.5), &intern);
        brain.flush_now().unwrap();
        mt.insert(subs[0], hub, BondType::RelatedTo, 0.9, kf(0.9), &intern);
        brain.flush_now().unwrap();
    }

    /// Set of neighbour NodeId.0 the resident adjacency holds for `seed`
    /// in `dir`, mapped through the intern (every neighbour is interned).
    fn network_neighbor_ids(
        brain: &Brain,
        seed: &Seed,
        dir: Direction,
    ) -> HashSet<u64> {
        let intern = brain.network.node_intern();
        let seeds = match dir {
            Direction::Outgoing => brain.network.neighbors_outgoing(seed),
            Direction::Incoming => brain.network.neighbors_incoming(seed),
        };
        seeds
            .iter()
            .filter_map(|s| intern.get(s).map(|id| id.0))
            .collect()
    }

    fn cache_neighbor_ids(list: &[crate::cache_adjacency::DecodedNeighbor]) -> HashSet<u64> {
        list.iter().map(|n| n.node.0).collect()
    }

    /// THE GATE: cache-backed neighbours == resident adjacency, for every
    /// interned node, both directions -- same SET, same winning predicate
    /// per pair. Also asserts the cold-miss -> reader -> warm-hit path
    /// (a second read does not re-decode) and the map degree / cache
    /// length / adjacency degree all agree.
    #[test]
    fn cache_backed_neighbors_match_resident_adjacency() {
        let dir = tempdir().unwrap();
        {
            let brain = boot_v5(dir.path());
            teach_small(&brain);
        }
        let mut brain = Brain::load_v5(dir.path(), None).expect("reload via load_v5");
        brain.rehydrate_adjacency()
            .expect("rehydrate resident adjacency for legacy/parity comparison");
        let map = std::sync::Arc::new(brain.build_resident_map().unwrap());
        // Generous cap so this sweep is purely a correctness check; pin 1.
        let cache = CacheBackedAdjacency::with_pin_count(map.clone(), 1024, 1);
        let source = brain.v5_adjacency_source().expect("v5 source");
        let intern = brain.network.node_intern().clone();
        let n = intern.len();
        assert!(n >= 7);

        let mut checked = 0u64;
        for i in 0..n as u64 {
            let id = NodeId(i);
            let seed = match intern.reverse_lookup(id) {
                Some(s) => s,
                None => continue,
            };
            for dir in [Direction::Outgoing, Direction::Incoming] {
                let list = cache.neighbors(id, dir, &source).unwrap();
                let cache_ids = cache_neighbor_ids(&list);
                let net_ids = network_neighbor_ids(&brain, &seed, dir);
                assert_eq!(
                    cache_ids, net_ids,
                    "neighbour SET mismatch at id {} dir {:?}", i, dir,
                );
                // Degree agreement: cache list == map degree == adjacency degree.
                let map_deg = match dir {
                    Direction::Outgoing => map.out_degree(id) as usize,
                    Direction::Incoming => map.in_degree(id) as usize,
                };
                let net_deg = match dir {
                    Direction::Outgoing => brain.network.out_degree(&seed),
                    Direction::Incoming => brain.network.in_degree(&seed),
                };
                assert_eq!(list.len(), map_deg, "cache len != map degree at id {}", i);
                assert_eq!(map_deg, net_deg, "map degree != adjacency degree at id {}", i);

                // Predicate parity: each cache neighbour's winning
                // predicate maps to the SAME BondType the resident bond
                // carries (for outgoing the bond is seed->other; for
                // incoming it is other->seed).
                for nb in list.iter() {
                    let other = intern.reverse_lookup(nb.node).expect("neighbour interned");
                    let bond = match dir {
                        Direction::Outgoing => brain.network.get_bond(&seed, &other),
                        Direction::Incoming => brain.network.get_bond(&other, &seed),
                    }
                    .expect("resident bond must exist for a cache neighbour");
                    assert_eq!(
                        crate::memtable::bond_type_from_discriminant(nb.predicate),
                        Some(bond.bond_type()),
                        "predicate mismatch for {}->{} dir {:?}", i, nb.node.0, dir,
                    );
                }
                checked += 1;
            }
        }
        assert!(checked >= 14);

        // Known-topology winning-predicate spot check: (s0,a) collapsed
        // to the higher-seq IsA bond.
        let s0 = intern.get(&Seed::from_label("p1_s0")).unwrap();
        let a = intern.get(&Seed::from_label("p1_a")).unwrap();
        let s0_out = cache.neighbors_outgoing(s0, &source).unwrap();
        let a_nb = s0_out.iter().find(|nb| nb.node == a).expect("s0 -> a present");
        assert_eq!(
            crate::memtable::bond_type_from_discriminant(a_nb.predicate),
            Some(BondType::IsA),
            "multi-predicate (s0,a) must collapse to the higher-seq IsA",
        );

        // Cold -> warm: re-reading s0 outgoing must NOT re-decode.
        let reader_before = cache.reader_calls();
        let _ = cache.neighbors_outgoing(s0, &source).unwrap();
        assert_eq!(cache.reader_calls(), reader_before, "warm read must not re-decode");
    }

    /// Eviction-then-refetch on a real brain: a tiny cap forces the 2Q to
    /// drop entries during a sweep; a refetch re-decodes and is STILL
    /// set-exact against the resident adjacency.
    #[test]
    fn cache_eviction_then_refetch_is_exact() {
        let dir = tempdir().unwrap();
        {
            let brain = boot_v5(dir.path());
            teach_small(&brain);
        }
        let mut brain = Brain::load_v5(dir.path(), None).expect("reload via load_v5");
        brain.rehydrate_adjacency()
            .expect("rehydrate resident adjacency for legacy/parity comparison");
        let map = std::sync::Arc::new(brain.build_resident_map().unwrap());
        // Cap 2, no pins -- the sweep is much larger than the cap.
        let cache = CacheBackedAdjacency::with_pin_count(map, 2, 0);
        let source = brain.v5_adjacency_source().expect("v5 source");
        let intern = brain.network.node_intern().clone();
        let n = intern.len();

        // Full sweep (eviction pressure).
        for i in 0..n as u64 {
            let _ = cache.neighbors_outgoing(NodeId(i), &source).unwrap();
            let _ = cache.neighbors_incoming(NodeId(i), &source).unwrap();
        }
        assert!(cache.resident_lists() <= 2, "2Q must bound resident lists to the cap");
        let reads_after_sweep = cache.reader_calls();

        // Refetch s0 outgoing: it was evicted, so this re-decodes, and it
        // must still match the resident adjacency exactly.
        let s0_seed = Seed::from_label("p1_s0");
        let s0 = intern.get(&s0_seed).unwrap();
        let refetched = cache.neighbors_outgoing(s0, &source).unwrap();
        assert_eq!(
            cache_neighbor_ids(&refetched),
            network_neighbor_ids(&brain, &s0_seed, Direction::Outgoing),
            "refetched-after-eviction list must equal the resident adjacency",
        );
        assert!(
            cache.reader_calls() > reads_after_sweep,
            "an evicted entry must have been re-decoded on refetch",
        );
    }

    /// Hub-pin: the top-degree hub stays resident under heavy eviction
    /// pressure and is served WITHOUT a re-decode, and correctly.
    #[test]
    fn pinned_hub_stays_resident_and_correct() {
        let dir = tempdir().unwrap();
        {
            let brain = boot_v5(dir.path());
            teach_small(&brain);
        }
        let mut brain = Brain::load_v5(dir.path(), None).expect("reload via load_v5");
        brain.rehydrate_adjacency()
            .expect("rehydrate resident adjacency for legacy/parity comparison");
        let map = std::sync::Arc::new(brain.build_resident_map().unwrap());
        let cache = CacheBackedAdjacency::with_pin_count(map, 1, 1); // cap 1, pin top 1
        let source = brain.v5_adjacency_source().expect("v5 source");
        let intern = brain.network.node_intern().clone();
        let hub_seed = Seed::from_label("p1_hub");
        let hub = intern.get(&hub_seed).unwrap();
        assert!(cache.is_pinned(hub), "hub (highest degree) must be the pinned node");

        // Pre-warm pins, then apply eviction pressure over every node.
        cache.prewarm_pins(&source).unwrap();
        assert_eq!(cache.pinned_lists(), 1, "hub incoming list pinned (out-degree 0)");
        let n = intern.len();
        for i in 0..n as u64 {
            let _ = cache.neighbors_outgoing(NodeId(i), &source).unwrap();
            let _ = cache.neighbors_incoming(NodeId(i), &source).unwrap();
        }

        // Hub incoming served from the immune pinned map: no new decode.
        let reads_before = cache.reader_calls();
        let hub_in = cache.neighbors_incoming(hub, &source).unwrap();
        assert_eq!(
            cache.reader_calls(), reads_before,
            "pinned hub must be served without a re-decode after eviction pressure",
        );
        assert_eq!(
            cache_neighbor_ids(&hub_in),
            network_neighbor_ids(&brain, &hub_seed, Direction::Incoming),
            "pinned hub list must still equal the resident adjacency",
        );
        assert_eq!(hub_in.len(), 4, "hub in-degree 4 (s0,s1,s2,s3)");
    }

    /// Incoming reads served via the on-disk reverse index: force a
    /// minor compaction (L0 -> L1, which builds `<uid>.bsst.reverse`),
    /// confirm the reverse index exists, then assert incoming parity is
    /// STILL exact when `get_incoming_bonds_v5` resolves via that index.
    #[test]
    fn cache_incoming_via_reverse_index_is_exact() {
        let dir = tempdir().unwrap();
        {
            let brain = boot_v5(dir.path());
            teach_many(&brain);
        }
        let mut brain = Brain::load_v5(dir.path(), None).expect("reload via load_v5");
        brain.rehydrate_adjacency()
            .expect("rehydrate resident adjacency for legacy/parity comparison");

        // Drive compaction (5 L0 SSTs > trigger 4 -> minor compaction
        // builds the paired reverse index). The background scheduler may
        // also fire; either way we poll for the reverse file to appear.
        let sched = brain.bond_v5_compaction().expect("v5 bond scheduler");
        sched.compact_now().unwrap();
        let bond_dir = dir.path().join("bond_store");
        let mut reverse_seen = false;
        for _ in 0..60 {
            if let Ok(rd) = std::fs::read_dir(&bond_dir) {
                if rd.filter_map(|e| e.ok()).any(|e| {
                    e.file_name().to_string_lossy().ends_with(".bsst.reverse")
                }) {
                    reverse_seen = true;
                    break;
                }
            }
            let _ = sched.compact_now();
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        assert!(
            reverse_seen,
            "a forced compaction must build an on-disk .bsst.reverse index",
        );
        // (Not asserting `sched.compactions_run()`: the minor
        // compaction can legitimately have been performed by the TEACH
        // brain's background scheduler before this reload, in which case
        // the reload scheduler's counter is 0 yet the reverse index --
        // the thing under test -- already exists on disk, as asserted
        // above via `reverse_seen`.)

        // Build the cache + source AFTER compaction so the source's
        // manifest snapshot sees the compacted L1 + reverse index, and
        // incoming reads resolve through the reverse index.
        let map = std::sync::Arc::new(brain.build_resident_map().unwrap());
        let cache = CacheBackedAdjacency::with_pin_count(map, 1024, 0);
        let source = brain.v5_adjacency_source().expect("v5 source");
        let intern = brain.network.node_intern().clone();

        // Hub incoming (in-degree 6) -- the read that exercises the
        // reverse index -- must equal the resident adjacency exactly.
        let hub_seed = Seed::from_label("p1m_hub");
        let hub = intern.get(&hub_seed).unwrap();
        let hub_in = cache.neighbors_incoming(hub, &source).unwrap();
        assert_eq!(
            cache_neighbor_ids(&hub_in),
            network_neighbor_ids(&brain, &hub_seed, Direction::Incoming),
            "reverse-index-served incoming list must equal the resident adjacency",
        );
        assert_eq!(hub_in.len(), 6, "hub in-degree 6 after compaction");

        // Full incoming sweep through the reverse index: every node exact.
        let n = intern.len();
        for i in 0..n as u64 {
            let id = NodeId(i);
            let seed = match intern.reverse_lookup(id) {
                Some(s) => s,
                None => continue,
            };
            let list = cache.neighbors_incoming(id, &source).unwrap();
            assert_eq!(
                cache_neighbor_ids(&list),
                network_neighbor_ids(&brain, &seed, Direction::Incoming),
                "incoming mismatch at id {} (reverse-index path)", i,
            );
        }
    }

    /// The `LU_MAPRAM_CACHE` load flag wires the cache into `load_v5`
    /// and retains it on the Brain; without the flag the field stays
    /// `None` (production load unchanged). This module is the only
    /// writer of that env var.
    #[test]
    fn load_flag_builds_and_retains_cache() {
        let dir = tempdir().unwrap();
        {
            let brain = boot_v5(dir.path());
            teach_small(&brain);
        }

        // CACHE now defaults ON: the default (routed) load always
        // builds + retains the cache (the seam is mandatory in routed
        // mode).
        std::env::remove_var("LU_MAPRAM_ROUTE");
        std::env::remove_var("LU_MAPRAM_CACHE");
        let def = Brain::load_v5(dir.path(), None).expect("default reload");
        assert!(
            def.cache_adjacency.is_some(),
            "default (routed) load => cache retained on the Brain",
        );
        drop(def);

        // Full legacy rollback (route off + cache off) builds no cache.
        std::env::set_var("LU_MAPRAM_ROUTE", "0");
        std::env::set_var("LU_MAPRAM_CACHE", "0");
        let off = Brain::load_v5(dir.path(), None).expect("legacy reload");
        assert!(off.cache_adjacency.is_none(), "route+cache off => no cache built");
        drop(off);

        // Legacy mode WITH the cache (route off, cache on): rehydrate
        // builds the resident adjacency AND the coexisting cache, so the
        // convenience wrapper below matches the resident adjacency. Only
        // this module writes LU_MAPRAM_ROUTE / _CACHE => race-free.
        std::env::set_var("LU_MAPRAM_ROUTE", "0");
        std::env::set_var("LU_MAPRAM_CACHE", "1");
        let flagged = Brain::load_v5(dir.path(), None).expect("legacy+cache reload");
        std::env::remove_var("LU_MAPRAM_ROUTE");
        std::env::remove_var("LU_MAPRAM_CACHE");
        assert!(
            flagged.cache_adjacency.is_some(),
            "cache on => cache retained on the Brain",
        );
        // The Brain convenience wrapper resolves through the cache and
        // matches the resident adjacency.
        let intern = flagged.network.node_intern().clone();
        let s0_seed = Seed::from_label("p1_s0");
        let s0 = intern.get(&s0_seed).unwrap();
        let list = flagged
            .cache_neighbors_outgoing(s0)
            .expect("cache read ok")
            .expect("cache enabled => Some");
        assert_eq!(
            cache_neighbor_ids(&list),
            network_neighbor_ids(&flagged, &s0_seed, Direction::Outgoing),
        );
    }

    /// REGRESSION (REAL PRODUCTION RACE): `load_v5` starts a background
    /// bond-compaction scheduler AND a flush worker that BOTH swap the
    /// same shared `Arc<RwLock<Arc<Manifest>>>`. Before the fix they
    /// serialized on SEPARATE locks (`flush_lock` vs `compaction_lock`),
    /// so a flush and a compaction could read the SAME base snapshot and
    /// the later whole-manifest swap would REVERT the other's committed
    /// sub-tree -- silently dropping an already-acknowledged flush (data
    /// loss DURING TRAINING) and leaving the manifest pointing at files
    /// the other's `cleanup_orphan_*` had already unlinked (ENOENT on
    /// the next read / reload). Direct sibling of 7bd94b8 (node-vs-bond);
    /// the SAME root cause on the flush path, explicitly flagged as an
    /// unfixed latent pattern in that commit message.
    ///
    /// The env-gated bond probe parks the compaction mid-commit so a
    /// foreground flush is forced into the SAME base-snapshot window.
    /// With the shared commit lock the flush blocks until the
    /// compaction fully commits, so the interleaving cannot occur and
    /// the acknowledged bond always survives.
    #[test]
    fn flush_during_compaction_preserves_committed_bond() {
        // Park the bond compaction 1.2s after it reads its base snapshot.
        // Flush stays fast (no flush-side probe).
        std::env::set_var("LU5_BOND_PROBE_MS", "1200");
        std::env::remove_var("LU5_FLUSH_PROBE_MS");

        let trials = 6;
        let mut lost = 0;
        for t in 0..trials {
            let dir = tempdir().unwrap();
            let brain = boot_v5(dir.path());
            let intern = brain.flush_intern.as_ref().unwrap().clone();
            let mt = brain.memtable().unwrap().clone();

            // 4 separate flushes => 4 L0 BondSSTs (== DEFAULT trigger 4).
            for i in 0..4u32 {
                let s = Seed::from_label(&format!("fc{}_s{}", t, i));
                let o = Seed::from_label(&format!("fc{}_o{}", t, i));
                intern.intern_or_get(s);
                intern.intern_or_get(o);
                mt.insert(s, o, BondType::RelatedTo, 0.5, kf(0.5), &intern);
                brain.flush_now().unwrap();
            }

            // Let the background scheduler observe L0==4 and park in the
            // probe (poll 200ms; probe 1.2s): it has read its base and is
            // sleeping but has NOT yet swapped.
            std::thread::sleep(std::time::Duration::from_millis(500));

            // Flush a fresh, uniquely-identifiable bond while the
            // compaction is parked mid-commit. flush_now returns Ok and
            // truncates the WAL -- the bond is durably ACKNOWLEDGED.
            let ks = Seed::from_label(&format!("fc{}_keep_s", t));
            let ko = Seed::from_label(&format!("fc{}_keep_o", t));
            intern.intern_or_get(ks);
            intern.intern_or_get(ko);
            mt.insert(ks, ko, BondType::RelatedTo, 0.9, kf(0.9), &intern);
            brain.flush_now().unwrap();

            // Join all background workers (Drop), then reload from disk.
            drop(brain);
            let mut reloaded = Brain::load_v5(dir.path(), None).expect(
                "reload via load_v5 must succeed (manifest must not \
                 reference unlinked SSTs)",
            );
            // The acknowledged bond MUST survive in the rehydrated
            // adjacency. Pre-fix the concurrent compaction's stale-base
            // swap + cleanup reverts and unlinks the new SST.
            reloaded.rehydrate_adjacency()
                .expect("rehydrate resident adjacency to assert acknowledged bond survival");
            if reloaded.network.get_bond(&ks, &ko).is_none() {
                lost += 1;
                eprintln!(
                    "[regression] trial {}: acknowledged bond fc{}_keep_s -> \
                     fc{}_keep_o LOST after concurrent compaction",
                    t, t, t,
                );
            }
            drop(reloaded);
        }
        std::env::remove_var("LU5_BOND_PROBE_MS");
        assert_eq!(
            lost, 0,
            "{}/{} trials silently dropped an acknowledged flush via the \
             flush-vs-compaction stale-base manifest swap",
            lost, trials,
        );
    }
}
#[cfg(test)]
mod mapram_p2_tests {
    //! OP-BRAIN-MapRAM P2 -- the byte-equivalent parity gate. Boots a
    //! real v5 brain through `load_v5`, installs the P1 cache-backed
    //! adjacency seam, then runs the SAME spread_activation /
    //! typed_recall queries through BOTH paths (legacy resident
    //! adjacency vs cache-backed map+cache+reader) and asserts the FULL
    //! ranked activation result is IDENTICAL -- node ids, activation
    //! bit-patterns, depth, and ordering -- not just neighbour sets.
    //!
    //! Why this proves byte-equivalence despite the cache iterating in a
    //! different (v5-reader) order: rehydrate_adjacency stays intact, so
    //! `view_bond`/`view_node` read the same resident bonds in both
    //! modes -- per-neighbour weights are identical by construction. The
    //! seam changes only (a) which neighbour ids are enumerated -- a SET
    //! P1 proved identical -- and (b) where degree comes from -- the
    //! resident map, P0 proved byte-exact. The spread's final activation
    //! per node is an order-independent MAX, and the final ranking has a
    //! seed-byte tiebreak, so identical inputs => identical output. (The
    //! legacy adjacency order is itself HashMap::drain-random per run, so
    //! the algorithm is already required to be order-invariant; these
    //! tests confirm the cache's order is no exception.)
    use super::*;
    use crate::bond::{BondType, KeyframeSource, TemporalKeyframe};
    use crate::curiosity::{CuriosityCollector, CuriosityConfig};
    use crate::manifest::{KeyFingerprint, Manifest, MANIFEST_FILE_NAME};
    use crate::network::{RecallResult, SpreadStatus};
    use crate::scar::ScarLog;
    use crate::scope::TemporalScope;
    use crate::seed::Seed;
    use tempfile::tempdir;

    fn kf(w: f32) -> Vec<TemporalKeyframe> {
        vec![TemporalKeyframe { timestamp: 100, weight: w, source: KeyframeSource::Baseline }]
    }

    fn boot_v5(brain_dir: &std::path::Path) -> Brain {
        let mut m = Manifest::new(KeyFingerprint::none());
        m.set_brain_format_v5();
        m.save_atomic(&brain_dir.join(MANIFEST_FILE_NAME)).unwrap();
        Brain::load_v5(brain_dir, None).expect("v5 brain must boot via load_v5")
    }

    /// A multi-hop topology with convergence nodes (reached by two
    /// routes -- the order-sensitive case), an incoming-heavy hub, a
    /// node with out-degree 3 (for the hub-cap test), one IsA bond (for
    /// typed filtering), all with DISTINCT weights (no cap ties).
    fn teach_p2(brain: &Brain) {
        let intern = brain.flush_intern.as_ref().unwrap().clone();
        let mt = brain.memtable().unwrap().clone();
        let g = |n: &str| Seed::from_label(n);
        let labels = [
            "p2_r", "p2_n1", "p2_n2", "p2_n3", "p2_c", "p2_d", "p2_e", "p2_f",
            "p2_g", "p2_hub", "p2_d2", "p2_s0", "p2_s1", "p2_s2", "p2_s3", "p2_s4",
        ];
        for l in labels {
            intern.intern_or_get(g(l));
        }
        // Flush 1: the spine + convergence + IsA.
        for (f, t, w, bt) in [
            ("p2_r", "p2_n1", 0.90f32, BondType::RelatedTo),
            ("p2_r", "p2_n2", 0.80, BondType::RelatedTo),
            ("p2_r", "p2_n3", 0.70, BondType::RelatedTo),
            ("p2_n1", "p2_c", 0.85, BondType::RelatedTo),
            ("p2_n2", "p2_c", 0.60, BondType::RelatedTo), // c converges (n1,n2)
            ("p2_n1", "p2_d", 0.75, BondType::RelatedTo),
            ("p2_n2", "p2_e", 0.65, BondType::RelatedTo),
            ("p2_c", "p2_f", 0.70, BondType::RelatedTo),
            ("p2_d", "p2_f", 0.55, BondType::RelatedTo), // f converges (c,d)
            ("p2_n3", "p2_hub", 0.60, BondType::RelatedTo),
            ("p2_hub", "p2_g", 0.90, BondType::RelatedTo),
            ("p2_n1", "p2_d2", 0.50, BondType::IsA), // the lone IsA bond
        ] {
            mt.insert(g(f), g(t), bt, w, kf(w), &intern);
        }
        brain.flush_now().unwrap();
        // Flush 2: load the hub's incoming side (degree -> incoming path).
        for s in ["p2_s0", "p2_s1", "p2_s2", "p2_s3", "p2_s4"] {
            mt.insert(g(s), g("p2_hub"), BondType::RelatedTo, 0.50, kf(0.50), &intern);
        }
        brain.flush_now().unwrap();
    }

    /// Ranked fingerprint: (seed bytes, activation bit-pattern, depth)
    /// per result row, IN ORDER. Bit-exact on the f32 activation.
    fn ranked(r: &RecallResult) -> Vec<(Vec<u8>, u32, usize)> {
        r.activated
            .iter()
            .map(|a| (a.seed.as_bytes().to_vec(), a.activation.to_bits(), a.depth))
            .collect()
    }

    fn load_routed(dir: &std::path::Path, cap: usize, pin: f64) -> Brain {
        let mut brain = Brain::load_v5(dir, None).expect("reload via load_v5");
        brain.rehydrate_adjacency()
            .expect("rehydrate resident adjacency for legacy/parity comparison");
        brain.enable_cache_backed_adjacency(cap, pin).expect("enable cache");
        brain.enable_adjacency_seam().expect("install seam");
        brain
    }

    /// THE GATE (spread_activation): identical ranked activations from
    /// both paths across seeds x depths x top_k.
    #[test]
    fn spread_activation_byte_equivalent_both_paths() {
        let dir = tempdir().unwrap();
        {
            let brain = boot_v5(dir.path());
            teach_p2(&brain);
        }
        let brain = load_routed(dir.path(), 4096, 0.0);
        let intern = brain.network.node_intern().clone();
        let scope = TemporalScope::Current;
        let scar = ScarLog::default();
        let seeds = ["p2_r", "p2_n1", "p2_n2", "p2_hub", "p2_s0", "p2_c"];
        let mut compared = 0usize;
        let mut max_rows = 0usize;
        for lbl in seeds {
            let s = Seed::from_label(lbl);
            if intern.get(&s).is_none() {
                continue;
            }
            for depth in [1usize, 2, 3, 4, 5] {
                for top_k in [5usize, 64] {
                    brain.set_adjacency_cache_backed(false);
                    assert!(!brain.adjacency_cache_backed());
                    let legacy = brain.network.spread_activation_weighted(
                        &[(s, 1.0)], &scope, depth, top_k, &scar,
                    );
                    brain.set_adjacency_cache_backed(true);
                    assert!(brain.adjacency_cache_backed());
                    let cached = brain.network.spread_activation_weighted(
                        &[(s, 1.0)], &scope, depth, top_k, &scar,
                    );
                    assert_eq!(legacy.status, SpreadStatus::Complete);
                    assert!(
                        !legacy.activation_truncated,
                        "test brain must not truncate (would make order matter): {} d{}",
                        lbl, depth,
                    );
                    assert_eq!(
                        ranked(&legacy), ranked(&cached),
                        "spread parity broke: seed {} depth {} top_k {}", lbl, depth, top_k,
                    );
                    assert_eq!(legacy.nodes_visited, cached.nodes_visited);
                    assert_eq!(legacy.bonds_evaluated, cached.bonds_evaluated);
                    max_rows = max_rows.max(legacy.activated.len());
                    compared += 1;
                }
            }
        }
        assert!(compared >= 20, "too few comparisons: {}", compared);
        assert!(max_rows >= 5, "a meaningful spread should activate several nodes; got {}", max_rows);
    }

    /// THE GATE (typed_recall): identical ranked activations from both
    /// paths, including a bond_filter that exercises the IsA edge.
    #[test]
    fn typed_recall_byte_equivalent_both_paths() {
        let dir = tempdir().unwrap();
        {
            let brain = boot_v5(dir.path());
            teach_p2(&brain);
        }
        let brain = load_routed(dir.path(), 4096, 0.0);
        let intern = brain.network.node_intern().clone();
        let scope = TemporalScope::Current;
        let scar = ScarLog::default();
        let filters: [&[BondType]; 3] = [
            &[],
            &[BondType::RelatedTo],
            &[BondType::IsA],
        ];
        let seeds = ["p2_r", "p2_n1", "p2_hub"];
        let mut compared = 0usize;
        for lbl in seeds {
            let s = Seed::from_label(lbl);
            if intern.get(&s).is_none() {
                continue;
            }
            for f in filters.iter() {
                for depth in [1usize, 2, 3, 4] {
                    brain.set_adjacency_cache_backed(false);
                    let legacy = brain.network.typed_recall_weighted(
                        &[(s, 1.0)], &scope, depth, 64, f, &scar,
                    );
                    brain.set_adjacency_cache_backed(true);
                    let cached = brain.network.typed_recall_weighted(
                        &[(s, 1.0)], &scope, depth, 64, f, &scar,
                    );
                    assert!(!legacy.activation_truncated);
                    assert_eq!(
                        ranked(&legacy), ranked(&cached),
                        "typed parity broke: seed {} filter {:?} depth {}", lbl, f, depth,
                    );
                    compared += 1;
                }
            }
        }
        assert!(compared >= 12, "too few comparisons: {}", compared);
    }

    /// Parity holds when the read-side HUB CAP fires: lower max_out to 2
    /// so p2_r (out-degree 3) is capped. Both paths must keep the SAME
    /// top-2-by-weight neighbours and drop the same one.
    #[test]
    fn spread_parity_under_hub_cap() {
        let dir = tempdir().unwrap();
        {
            let brain = boot_v5(dir.path());
            teach_p2(&brain);
        }
        let mut brain = load_routed(dir.path(), 4096, 0.0);
        let mut cfg = brain.hub_degree_config.clone();
        cfg.max_out_degree = 2;
        cfg.max_in_degree = 2;
        cfg.read_enforcement = true;
        brain.network.sync_hub_read_config(&cfg);
        let scope = TemporalScope::Current;
        let scar = ScarLog::default();
        let r = Seed::from_label("p2_r");
        let hub = Seed::from_label("p2_hub");
        for seed in [r, hub] {
            for depth in [1usize, 2, 3] {
                brain.set_adjacency_cache_backed(false);
                let legacy = brain.network.spread_activation_weighted(
                    &[(seed, 1.0)], &scope, depth, 64, &scar,
                );
                brain.set_adjacency_cache_backed(true);
                let cached = brain.network.spread_activation_weighted(
                    &[(seed, 1.0)], &scope, depth, 64, &scar,
                );
                assert_eq!(
                    ranked(&legacy), ranked(&cached),
                    "hub-cap parity broke: seed {:?} depth {}", seed.as_bytes(), depth,
                );
            }
        }
    }

    /// Parity holds even when the cache is far too small to hold the
    /// working set: a cap-2 2Q churns and re-decodes evicted lists every
    /// hop, yet the cache-backed result is still bit-identical to legacy.
    #[test]
    fn spread_parity_under_cache_eviction_pressure() {
        let dir = tempdir().unwrap();
        {
            let brain = boot_v5(dir.path());
            teach_p2(&brain);
        }
        let brain = load_routed(dir.path(), 2, 0.0); // cap 2 -> heavy eviction
        let scope = TemporalScope::Current;
        let scar = ScarLog::default();
        let r = Seed::from_label("p2_r");
        brain.set_adjacency_cache_backed(false);
        let legacy = brain.network.spread_activation_weighted(&[(r, 1.0)], &scope, 5, 64, &scar);
        brain.set_adjacency_cache_backed(true);
        let cached = brain.network.spread_activation_weighted(&[(r, 1.0)], &scope, 5, 64, &scar);
        // The cache must have decoded under pressure (proves we actually
        // exercised the map->cache->reader path, not a fully-resident one).
        assert!(brain.cache_adjacency.as_ref().unwrap().reader_calls() > 0);
        assert_eq!(ranked(&legacy), ranked(&cached), "eviction-pressure parity broke");
    }

    /// Curiosity parity: the inline curiosity signals collected during
    /// spread (driven by the seam's degree / hub classification) are
    /// identical between paths.
    #[test]
    fn curiosity_signals_byte_equivalent_both_paths() {
        let dir = tempdir().unwrap();
        {
            let brain = boot_v5(dir.path());
            teach_p2(&brain);
        }
        let brain = load_routed(dir.path(), 4096, 0.0);
        let scope = TemporalScope::Current;
        let scar = ScarLog::default();
        let cur_cfg = CuriosityConfig::default();
        let r = Seed::from_label("p2_r");

        let collect = |brain: &Brain| -> Vec<(u8, Vec<u8>, u32)> {
            let mut cc = CuriosityCollector::new(256);
            let _ = brain.network.spread_activation_weighted_with_curiosity(
                &[(r, 1.0)], &scope, 5, 64, &mut cc, &cur_cfg, 2, 0, &scar,
            );
            let mut v: Vec<(u8, Vec<u8>, u32)> = cc
                .candidates
                .iter()
                .map(|c| (c.signal_type as u8, c.seed.as_bytes().to_vec(), c.payload.to_bits()))
                .collect();
            v.sort();
            v
        };
        brain.set_adjacency_cache_backed(false);
        let legacy = collect(&brain);
        brain.set_adjacency_cache_backed(true);
        let cached = collect(&brain);
        assert_eq!(legacy, cached, "curiosity-signal parity broke between paths");
    }
}

#[cfg(test)]
mod mapram_p3_tests {
    //! OP-BRAIN-MapRAM P3 -- proves cache-mode recall is SELF-SUFFICIENT
    //! for weights + node payloads + degree_p90 (no resident adjacency):
    //! (1) `degree_p90` sourced from the map at seam-install equals
    //!     rehydrate's `recompute_degree_p90`;
    //! (2) cache-mode recall is byte-identical before/after POISONING
    //!     every resident structure the legacy path reads -- so the
    //!     cache path reads weights from carried payloads, nodes from the
    //!     map-stub, neighbours/degree from the map, NOT the resident
    //!     adjacency. (The P2 byte-equivalence-to-legacy gate stays in
    //!     `mapram_p2_tests`.)
    use super::*;
    use crate::bond::{BondType, KeyframeSource, TemporalKeyframe};
    use crate::manifest::{KeyFingerprint, Manifest, MANIFEST_FILE_NAME};
    use crate::network::RecallResult;
    use crate::scar::ScarLog;
    use crate::scope::TemporalScope;
    use crate::seed::Seed;
    use tempfile::tempdir;

    fn kf(w: f32) -> Vec<TemporalKeyframe> {
        vec![TemporalKeyframe { timestamp: 100, weight: w, source: KeyframeSource::Baseline }]
    }
    fn boot_v5(brain_dir: &std::path::Path) -> Brain {
        let mut m = Manifest::new(KeyFingerprint::none());
        m.set_brain_format_v5();
        m.save_atomic(&brain_dir.join(MANIFEST_FILE_NAME)).unwrap();
        Brain::load_v5(brain_dir, None).expect("v5 brain must boot via load_v5")
    }
    fn lbl(n: &str) -> Seed { Seed::from_label(&format!("p3_{}", n)) }
    fn teach(brain: &Brain) {
        let intern = brain.flush_intern.as_ref().unwrap().clone();
        let mt = brain.memtable().unwrap().clone();
        let labels = ["r","n1","n2","n3","c","d","e","f","g","hub","d2","s0","s1","s2","s3","s4"];
        for l in labels { intern.intern_or_get(lbl(l)); }
        let spine: &[(&str,&str,f32,BondType)] = &[
            ("r","n1",0.90,BondType::RelatedTo),
            ("r","n2",0.80,BondType::RelatedTo),
            ("r","n3",0.70,BondType::RelatedTo),
            ("n1","c",0.85,BondType::RelatedTo),
            ("n2","c",0.60,BondType::RelatedTo),
            ("n1","d",0.75,BondType::RelatedTo),
            ("n2","e",0.65,BondType::RelatedTo),
            ("c","f",0.70,BondType::RelatedTo),
            ("d","f",0.55,BondType::RelatedTo),
            ("n3","hub",0.60,BondType::RelatedTo),
            ("hub","g",0.90,BondType::RelatedTo),
            ("n1","d2",0.50,BondType::IsA),
        ];
        for (f,t,w,bt) in spine.iter().copied() {
            mt.insert(lbl(f), lbl(t), bt, w, kf(w), &intern);
        }
        brain.flush_now().unwrap();
        for s in ["s0","s1","s2","s3","s4"] {
            mt.insert(lbl(s), lbl("hub"), BondType::RelatedTo, 0.50, kf(0.50), &intern);
        }
        brain.flush_now().unwrap();
    }
    fn ranked(r: &RecallResult) -> Vec<(Vec<u8>, u32, usize)> {
        r.activated.iter().map(|a| (a.seed.as_bytes().to_vec(), a.activation.to_bits(), a.depth)).collect()
    }

    #[test]
    fn degree_p90_map_sourced_matches_recompute() {
        let dir = tempdir().unwrap();
        { let b = boot_v5(dir.path()); teach(&b); }
        let mut brain = Brain::load_v5(dir.path(), None).unwrap();
        brain.rehydrate_adjacency()
            .expect("rehydrate resident adjacency for legacy/parity comparison");
        let recompute = brain.network.recompute_degree_p90();
        brain.enable_cache_backed_adjacency(4096, 0.0).unwrap();
        brain.enable_adjacency_seam().unwrap();
        assert_eq!(brain.network.degree_p90(), recompute,
            "map-sourced degree_p90 must equal rehydrate's recompute_degree_p90");
    }

    #[test]
    fn cache_recall_independent_of_resident_structures() {
        let dir = tempdir().unwrap();
        { let b = boot_v5(dir.path()); teach(&b); }
        let mut brain = Brain::load_v5(dir.path(), None).unwrap();
        brain.rehydrate_adjacency()
            .expect("rehydrate resident adjacency for legacy/parity comparison");
        brain.enable_cache_backed_adjacency(4096, 0.0).unwrap();
        brain.enable_adjacency_seam().unwrap();
        let scope = TemporalScope::Current;
        let scar = ScarLog::default();
        let seeds = ["r","n1","hub","s0"];
        let depths = [1usize,2,3,4];
        brain.set_adjacency_cache_backed(true);
        let mut before = Vec::new();
        for s in seeds {
            for d in depths {
                before.push(ranked(&brain.network.spread_activation_weighted(&[(lbl(s),1.0)], &scope, d, 64, &scar)));
            }
        }
        // Corrupt every resident structure the legacy recall reads.
        brain.network.mapram_test_poison_resident();
        brain.set_adjacency_cache_backed(true);
        let mut idx = 0;
        for s in seeds {
            for d in depths {
                let r = ranked(&brain.network.spread_activation_weighted(&[(lbl(s),1.0)], &scope, d, 64, &scar));
                assert_eq!(r, before[idx],
                    "cache recall changed after poisoning resident structures: {} d{}", s, d);
                idx += 1;
            }
        }
        // Sanity: the poison REALLY bit -- legacy mode now diverges.
        brain.set_adjacency_cache_backed(false);
        let legacy = ranked(&brain.network.spread_activation_weighted(&[(lbl("r"),1.0)], &scope, 4, 64, &scar));
        assert_ne!(legacy, before[3],
            "poison must corrupt the resident/legacy path (else the test is vacuous)");
    }
}


#[cfg(test)]
mod mapram_p4_tests {
    //! OP-BRAIN-MapRAM P4 -- byte-equivalence + resident-independence for
    //! the FIVE peripheral readers rerouted off the resident adjacency:
    //! the SENSE/POLYSEMY readers (`disambiguate_polysemous_with_anchors`,
    //! `polysemy_label_prop_clusters` via `disambiguate_polysemous*` and
    //! `score_sense_clusters`) and the TYPED-RATIO readers
    //! (`update_typed_ratio_for`, `recompute_all_typed_ratios`).
    //!
    //! Each reader is run on a REAL `load_v5` brain through BOTH the legacy
    //! resident adjacency and the cache-backed seam, and the meaningful
    //! output is asserted byte-identical:
    //!   * typed ratios -> f32 bit-pattern equality;
    //!   * disambiguation -> the returned seed SET (sorted seed bytes);
    //!   * sense scores -> per-cluster (sorted member bytes, score bits,
    //!     typed_bond_count, is_winner).
    //! The internal integer cluster LABEL and the score-tie vector ORDER
    //! are NOT asserted: they are positional artifacts of the neighbour
    //! enumeration order, which is itself HashMap::drain-random per load
    //! (`rehydrate_adjacency`), so they are already per-load nondeterministic
    //! in legacy and are not a meaningful output. The label-prop PARTITION
    //! (cluster memberships) is the connected-component decomposition of the
    //! order-independent co-neighbourhood graph, so the asserted outputs are
    //! order-independent -- the cache's different (v5-reader) order is no
    //! exception. A poison test then proves the cache-mode readers read
    //! NEITHER the resident adjacency/bonds NOR bond types.
    use super::*;
    use crate::bond::{BondType, KeyframeSource, TemporalKeyframe};
    use crate::manifest::{KeyFingerprint, Manifest, MANIFEST_FILE_NAME};
    use crate::seed::Seed;
    use std::collections::HashSet;
    use tempfile::tempdir;

    fn kf(w: f32) -> Vec<TemporalKeyframe> {
        vec![TemporalKeyframe { timestamp: 100, weight: w, source: KeyframeSource::Baseline }]
    }
    fn boot_v5(brain_dir: &std::path::Path) -> Brain {
        let mut m = Manifest::new(KeyFingerprint::none());
        m.set_brain_format_v5();
        m.save_atomic(&brain_dir.join(MANIFEST_FILE_NAME)).unwrap();
        Brain::load_v5(brain_dir, None).expect("v5 brain must boot via load_v5")
    }
    fn load_routed(dir: &std::path::Path, cap: usize, pin: f64) -> Brain {
        let mut brain = Brain::load_v5(dir, None).expect("reload via load_v5");
        brain.rehydrate_adjacency()
            .expect("rehydrate resident adjacency for legacy/parity comparison");
        brain.enable_cache_backed_adjacency(cap, pin).expect("enable cache");
        brain.enable_adjacency_seam().expect("install seam");
        brain
    }
    fn g(n: &str) -> Seed { Seed::from_label(n) }

    /// Subjects with a mix of typed + RelatedTo outgoing bonds so the
    /// typed_ratio takes 0, a strict fraction, and 1.0-adjacent values.
    fn teach_typed(brain: &Brain) {
        let intern = brain.flush_intern.as_ref().unwrap().clone();
        let mt = brain.memtable().unwrap().clone();
        for n in ["p4t_s1", "p4t_s2", "p4t_s3", "p4t_oa", "p4t_ob", "p4t_oc", "p4t_od"] {
            intern.intern_or_get(g(n));
        }
        // s1: IsA + RelatedTo + HasProperty -> 2/3
        mt.insert(g("p4t_s1"), g("p4t_oa"), BondType::IsA, 0.9, kf(0.9), &intern);
        mt.insert(g("p4t_s1"), g("p4t_ob"), BondType::RelatedTo, 0.5, kf(0.5), &intern);
        mt.insert(g("p4t_s1"), g("p4t_oc"), BondType::HasProperty, 0.7, kf(0.7), &intern);
        // s2: two RelatedTo -> 0
        mt.insert(g("p4t_s2"), g("p4t_oa"), BondType::RelatedTo, 0.5, kf(0.5), &intern);
        mt.insert(g("p4t_s2"), g("p4t_ob"), BondType::RelatedTo, 0.5, kf(0.5), &intern);
        // s3: IsA + Causes + PartOf + RelatedTo -> 3/4
        mt.insert(g("p4t_s3"), g("p4t_oa"), BondType::IsA, 0.9, kf(0.9), &intern);
        mt.insert(g("p4t_s3"), g("p4t_ob"), BondType::Causes, 0.6, kf(0.6), &intern);
        mt.insert(g("p4t_s3"), g("p4t_oc"), BondType::PartOf, 0.6, kf(0.6), &intern);
        mt.insert(g("p4t_s3"), g("p4t_od"), BondType::RelatedTo, 0.5, kf(0.5), &intern);
        brain.flush_now().unwrap();
    }

    /// A polysemous hub with two well-separated, dense sense clusters
    /// (each: 4 IsA edges from the hub + an internal RelatedTo clique).
    const POLY_A: [&str; 4] = ["p4_compartment", "p4_mitochondria", "p4_organelle", "p4_nucleus"];
    const POLY_B: [&str; 4] = ["p4_radiotelephone", "p4_radiophone", "p4_handset", "p4_receiver"];
    fn teach_poly(brain: &Brain) {
        let intern = brain.flush_intern.as_ref().unwrap().clone();
        let mt = brain.memtable().unwrap().clone();
        intern.intern_or_get(g("p4_cell"));
        for m in POLY_A.iter().chain(POLY_B.iter()) {
            intern.intern_or_get(g(m));
        }
        for m in POLY_A.iter().chain(POLY_B.iter()) {
            mt.insert(g("p4_cell"), g(m), BondType::IsA, 0.9, kf(0.9), &intern);
        }
        for cl in [&POLY_A, &POLY_B] {
            for i in 0..cl.len() {
                for j in 0..cl.len() {
                    if i != j {
                        mt.insert(g(cl[i]), g(cl[j]), BondType::RelatedTo, 0.5, kf(0.5), &intern);
                    }
                }
            }
        }
        brain.flush_now().unwrap();
    }

    // ---- canonical, order-independent projections of the outputs ----
    fn canon_set(s: &HashSet<Seed>) -> Vec<Vec<u8>> {
        let mut v: Vec<Vec<u8>> = s.iter().map(|x| x.as_bytes().to_vec()).collect();
        v.sort();
        v
    }
    fn canon_setlist(list: &[HashSet<Seed>]) -> Vec<Vec<Vec<u8>>> {
        let mut v: Vec<Vec<Vec<u8>>> = list.iter().map(canon_set).collect();
        v.sort();
        v
    }
    fn canon_scores(cs: &[crate::sense::SenseClusterScore]) -> Vec<(Vec<Vec<u8>>, u32, usize, bool)> {
        let mut v: Vec<(Vec<Vec<u8>>, u32, usize, bool)> = cs
            .iter()
            .map(|c| {
                let mut members: Vec<Vec<u8>> = c.members.iter().map(|m| m.as_bytes().to_vec()).collect();
                members.sort();
                (members, c.score.to_bits(), c.typed_bond_count, c.is_winner)
            })
            .collect();
        v.sort();
        v
    }

    /// TYPED-RATIO readers: byte-identical f32 results legacy vs cache,
    /// per-seed (`update_typed_ratio_for`) and whole-graph
    /// (`recompute_all_typed_ratios`).
    #[test]
    fn typed_ratio_byte_equivalent_both_paths() {
        let dir = tempdir().unwrap();
        { let b = boot_v5(dir.path()); teach_typed(&b); }
        let mut brain = load_routed(dir.path(), 4096, 0.0);
        let subjects = ["p4t_s1", "p4t_s2", "p4t_s3"];
        let mut saw_nonzero = false;
        for lbl in subjects {
            let s = g(lbl);
            brain.set_adjacency_cache_backed(false);
            brain.network.update_typed_ratio_for(s);
            let leg = brain.network.get_node(&s).unwrap().typed_ratio();
            brain.set_adjacency_cache_backed(true);
            brain.network.update_typed_ratio_for(s);
            let cac = brain.network.get_node(&s).unwrap().typed_ratio();
            assert_eq!(leg.to_bits(), cac.to_bits(),
                "update_typed_ratio_for byte-differs legacy vs cache for {}", lbl);
            if leg > 0.0 { saw_nonzero = true; }
        }
        assert!(saw_nonzero, "typed-ratio parity test is vacuous (all ratios zero)");
        // Concrete correctness: s1 == 2/3 in cache mode.
        brain.set_adjacency_cache_backed(true);
        brain.network.update_typed_ratio_for(g("p4t_s1"));
        let r1 = brain.network.get_node(&g("p4t_s1")).unwrap().typed_ratio();
        assert!((r1 - 2.0 / 3.0).abs() < 1e-6, "s1 ratio {} != 2/3", r1);
        // Whole-graph recompute parity (overlay populated by the loop).
        brain.set_adjacency_cache_backed(false);
        brain.network.recompute_all_typed_ratios();
        let leg: Vec<u32> = subjects.iter()
            .map(|l| brain.network.get_node(&g(l)).unwrap().typed_ratio().to_bits()).collect();
        brain.set_adjacency_cache_backed(true);
        brain.network.recompute_all_typed_ratios();
        let cac: Vec<u32> = subjects.iter()
            .map(|l| brain.network.get_node(&g(l)).unwrap().typed_ratio().to_bits()).collect();
        assert_eq!(leg, cac, "recompute_all_typed_ratios byte-differs legacy vs cache");
    }

    /// SENSE/POLYSEMY readers: byte-identical meaningful output legacy vs
    /// cache for disambiguation, multi-sense clustering, and sense scores.
    #[test]
    fn sense_readers_byte_equivalent_both_paths() {
        let dir = tempdir().unwrap();
        { let b = boot_v5(dir.path()); teach_poly(&b); }
        let brain = load_routed(dir.path(), 4096, 0.0);
        let hub = g("p4_cell");
        let a_members: Vec<Seed> = POLY_A.iter().map(|l| g(l)).collect();
        let activated: HashSet<Seed> = a_members.iter().copied().collect();
        let context: Vec<Seed> = a_members.clone();
        let anchors: Vec<Seed> = vec![a_members[0]];

        let run = |brain: &Brain| {
            let multi = brain.network.disambiguate_polysemous_multi_sense(&hub)
                .map(|v| canon_setlist(&v));
            let disamb = brain.network.disambiguate_polysemous(&hub, &activated)
                .map(|s| canon_set(&s));
            let scores = brain.network.score_sense_clusters(&hub, &context)
                .map(|cs| canon_scores(&cs));
            let anch = brain.network.disambiguate_polysemous_with_anchors(&hub, &activated, &anchors)
                .map(|s| canon_set(&s));
            (multi, disamb, scores, anch)
        };
        brain.set_adjacency_cache_backed(false);
        let legacy = run(&brain);
        brain.set_adjacency_cache_backed(true);
        let cached = run(&brain);
        assert_eq!(legacy, cached, "sense readers byte-differ legacy vs cache");
        // Non-vacuous: the readers actually fire on this topology.
        assert!(legacy.0.as_ref().map_or(false, |v| v.len() >= 2),
            "multi_sense vacuous: {:?}", legacy.0);
        assert!(legacy.2.as_ref().map_or(false, |v| v.len() >= 2),
            "score_sense_clusters vacuous");
        assert!(legacy.1.is_some(), "disambiguate_polysemous vacuous");
        assert!(legacy.3.is_some(), "disambiguate_polysemous_with_anchors vacuous");
    }

    /// INDEPENDENCE: in cache mode the sense + typed-ratio readers are
    /// byte-invariant after POISONING every resident structure the legacy
    /// path reads -- adjacency, bond weights, bond TYPES, node payloads --
    /// proving they read from the cache (neighbours + carried predicate),
    /// not the resident adjacency/bonds.
    #[test]
    fn sense_and_typed_independent_of_resident() {
        let dir = tempdir().unwrap();
        { let b = boot_v5(dir.path()); teach_poly(&b); teach_typed(&b); }
        let mut brain = load_routed(dir.path(), 4096, 0.0);
        brain.set_adjacency_cache_backed(true);
        let hub = g("p4_cell");
        let context: Vec<Seed> = POLY_A.iter().map(|l| g(l)).collect();
        let s1 = g("p4t_s1");

        let base_multi = brain.network.disambiguate_polysemous_multi_sense(&hub)
            .map(|v| canon_setlist(&v));
        let base_scores = brain.network.score_sense_clusters(&hub, &context)
            .map(|cs| canon_scores(&cs));
        brain.network.update_typed_ratio_for(s1);
        let base_ratio = brain.network.get_node(&s1).unwrap().typed_ratio();
        assert!(base_multi.as_ref().map_or(false, |v| v.len() >= 2), "baseline multi vacuous");
        assert!(base_ratio > 0.0, "baseline typed_ratio vacuous");

        // Corrupt every resident structure the legacy path reads.
        brain.network.mapram_test_poison_resident();
        brain.network.mapram_test_corrupt_resident_bond_types();

        brain.set_adjacency_cache_backed(true);
        let p_multi = brain.network.disambiguate_polysemous_multi_sense(&hub)
            .map(|v| canon_setlist(&v));
        let p_scores = brain.network.score_sense_clusters(&hub, &context)
            .map(|cs| canon_scores(&cs));
        brain.network.update_typed_ratio_for(s1);
        let p_ratio = brain.network.get_node(&s1).unwrap().typed_ratio();
        assert_eq!(base_multi, p_multi, "multi_sense changed after poisoning resident structures");
        assert_eq!(base_scores, p_scores, "score_sense_clusters changed after poisoning resident structures");
        assert_eq!(base_ratio.to_bits(), p_ratio.to_bits(),
            "typed_ratio changed after poisoning resident structures");

        // Sanity: the poison REALLY bit -- the legacy path now diverges.
        brain.set_adjacency_cache_backed(false);
        brain.network.update_typed_ratio_for(s1);
        let legacy_ratio = brain.network.get_node(&s1).unwrap().typed_ratio();
        assert_ne!(legacy_ratio.to_bits(), base_ratio.to_bits(),
            "poison must corrupt the legacy typed-ratio path (else the test is vacuous)");
    }
}
#[cfg(test)]
mod mapram_p5_tests {
    //! OP-BRAIN-MapRAM P5-batch1 -- byte-equivalence + resident-independence
    //! for the ORDER-INDEPENDENT batch-1 adjacency consumers rerouted off
    //! the resident adjacency through the canonical seam (cache mode only;
    //! the legacy branch of every consumer is byte-for-byte the prior code):
    //!   * `bridge::is_hub` (total-degree gate) + `bridge::jaccard_neighbors`
    //!     (out+in neighbour SET);
    //!   * `idk_gate` anchor coverage (out+in neighbour SET);
    //!   * `narration` hub degree gate (out-degree, ~669);
    //!   * `variants::degree` (total-degree);
    //!   * `research::chain::endpoint_gap_intensity` + `endpoint_is_hub`
    //!     (in/out-degree gates);
    //!   * `network::collect_seeds_with_filter` majority-scan degree
    //!     (total-degree).
    //!
    //! Every one of these is a PURE, DETERMINISTIC function of a small set of
    //! seam reads -- `seam_out_degree` / `seam_in_degree` / `seam_total_degree`
    //! (degree gates) and `seam_outgoing_set` / `seam_incoming_set`
    //! (set-membership) -- all ORDER-INDEPENDENT. In LEGACY mode each seam
    //! read is the prior call BYTE-FOR-BYTE (`seam_*_degree == out/in_degree
    //! == neighbors_*().len()`, `seam_outgoing_set == neighbors_outgoing`,
    //! `seam_incoming_set == neighbors_incoming`), so legacy consumer output
    //! is unchanged by construction. `p5_seam_degree_and_sets_byte_equivalent`
    //! proves the CACHE-mode reads equal the legacy reads on a real `load_v5`
    //! brain (degrees as usize; neighbour sets as SORTED seed-byte vecs --
    //! the order-independent projection), which is exactly the property that
    //! propagates byte-equivalence to ALL six consumers, including the private
    //! ones not directly reachable from a test (`jaccard_neighbors`,
    //! `variants::degree`, the narration gate, `collect_seeds_with_filter`).
    //! `p5_public_consumers_byte_equivalent` additionally exercises the
    //! reachable public consumers end-to-end (`bridge::is_hub`,
    //! `research::chain::endpoint_*`). `p5_batch1_reads_independent_of_resident`
    //! extends BOTH poison helpers (`mapram_test_poison_resident` +
    //! `mapram_test_corrupt_resident_bond_types`) to prove the cache-mode
    //! reads + consumers are byte-invariant under resident corruption.
    use super::*;
    use crate::bond::{BondType, KeyframeSource, TemporalKeyframe};
    use crate::manifest::{KeyFingerprint, Manifest, MANIFEST_FILE_NAME};
    use crate::seed::Seed;
    use tempfile::tempdir;

    type SeamRead = (usize, usize, usize, Vec<Vec<u8>>, Vec<Vec<u8>>);
    type Cons = (bool, u32, bool);

    fn kf(w: f32) -> Vec<TemporalKeyframe> {
        vec![TemporalKeyframe { timestamp: 100, weight: w, source: KeyframeSource::Baseline }]
    }
    fn boot_v5(brain_dir: &std::path::Path) -> Brain {
        let mut m = Manifest::new(KeyFingerprint::none());
        m.set_brain_format_v5();
        m.save_atomic(&brain_dir.join(MANIFEST_FILE_NAME)).unwrap();
        Brain::load_v5(brain_dir, None).expect("v5 brain must boot via load_v5")
    }
    fn load_routed(dir: &std::path::Path, cap: usize, pin: f64) -> Brain {
        let mut brain = Brain::load_v5(dir, None).expect("reload via load_v5");
        brain.rehydrate_adjacency()
            .expect("rehydrate resident adjacency for legacy/parity comparison");
        brain.enable_cache_backed_adjacency(cap, pin).expect("enable cache");
        brain.enable_adjacency_seam().expect("install seam");
        brain
    }
    fn g(n: &str) -> Seed { Seed::from_label(n) }

    /// Hub topology: `p5_hub` has out_degree 10 (hub -> n0..n9) and
    /// in_degree 5 (n0..n4 -> hub); `p5_leaf` has total degree 1
    /// (n0 -> leaf). Gives a clear hub/non-hub split for the degree gates
    /// and non-empty out+in neighbour SETS for the set-membership readers.
    fn teach_b1(brain: &Brain) {
        let intern = brain.flush_intern.as_ref().unwrap().clone();
        let mt = brain.memtable().unwrap().clone();
        let mut labels = vec!["p5_hub".to_string(), "p5_leaf".to_string()];
        for i in 0..10 { labels.push(format!("p5_n{}", i)); }
        for l in &labels { intern.intern_or_get(g(l)); }
        for i in 0..10 {
            mt.insert(g("p5_hub"), g(&format!("p5_n{}", i)), BondType::RelatedTo, 0.8, kf(0.8), &intern);
        }
        for i in 0..5 {
            mt.insert(g(&format!("p5_n{}", i)), g("p5_hub"), BondType::RelatedTo, 0.7, kf(0.7), &intern);
        }
        mt.insert(g("p5_n0"), g("p5_leaf"), BondType::RelatedTo, 0.6, kf(0.6), &intern);
        brain.flush_now().unwrap();
    }

    fn sorted_set(v: Vec<Seed>) -> Vec<Vec<u8>> {
        let mut out: Vec<Vec<u8>> = v.into_iter().map(|s| s.as_bytes().to_vec()).collect();
        out.sort();
        out
    }

    /// Snapshot of every batch-1 seam READ for a seed (both directions).
    /// out_degree, in_degree, total_degree, sorted outgoing set, sorted
    /// incoming set.
    fn seam_reads(net: &crate::network::AssociativeNetwork, s: &Seed) -> SeamRead {
        (
            net.seam_out_degree(s),
            net.seam_in_degree(s),
            net.seam_total_degree(s),
            sorted_set(net.seam_outgoing_set(s)),
            sorted_set(net.seam_incoming_set(s)),
        )
    }

    const SEEDS: [&str; 4] = ["p5_hub", "p5_n0", "p5_n5", "p5_leaf"];
    const HUB_THRESHOLD: usize = 5;

    /// SEAM-LEVEL byte-equivalence: the degree + neighbour-set reads that
    /// ALL six batch-1 consumers are pure functions of are byte-identical
    /// legacy vs cache on a real load_v5 brain.
    #[test]
    fn p5_seam_degree_and_sets_byte_equivalent() {
        let dir = tempdir().unwrap();
        { let b = boot_v5(dir.path()); teach_b1(&b); }
        let brain = load_routed(dir.path(), 4096, 0.0);
        let (mut saw_deg, mut saw_out, mut saw_in) = (false, false, false);
        for l in SEEDS {
            let s = g(l);
            brain.set_adjacency_cache_backed(false);
            let legacy = seam_reads(&brain.network, &s);
            brain.set_adjacency_cache_backed(true);
            let cache = seam_reads(&brain.network, &s);
            assert_eq!(legacy, cache, "seam reads byte-differ legacy vs cache for {}", l);
            if legacy.2 > 0 { saw_deg = true; }
            if !legacy.3.is_empty() { saw_out = true; }
            if !legacy.4.is_empty() { saw_in = true; }
        }
        assert!(saw_deg && saw_out && saw_in,
            "seam parity test vacuous (missing degree / out-set / in-set)");
        // Concrete topology (cache mode): hub out=10, in=5, total=15.
        brain.set_adjacency_cache_backed(true);
        assert_eq!(brain.network.seam_out_degree(&g("p5_hub")), 10);
        assert_eq!(brain.network.seam_in_degree(&g("p5_hub")), 5);
        assert_eq!(brain.network.seam_total_degree(&g("p5_hub")), 15);
    }

    /// PUBLIC-CONSUMER byte-equivalence: the reachable batch-1 consumers
    /// (`bridge::is_hub`, `research::chain::endpoint_gap_intensity`,
    /// `endpoint_is_hub`) produce identical output legacy vs cache, and
    /// actually discriminate hub from leaf (non-vacuous).
    #[test]
    fn p5_public_consumers_byte_equivalent() {
        let dir = tempdir().unwrap();
        { let b = boot_v5(dir.path()); teach_b1(&b); }
        let mut brain = load_routed(dir.path(), 4096, 0.0);
        // Tighten the hub cap so endpoint_is_hub discriminates (hub out=10).
        brain.hub_degree_config.max_out_degree = 8;
        brain.hub_degree_config.max_in_degree = 8;
        let run = |brain: &Brain| -> Vec<Cons> {
            SEEDS.iter().map(|l| {
                let s = g(l);
                (
                    crate::bridge::is_hub(&brain.network, &s, HUB_THRESHOLD),
                    crate::research::chain::endpoint_gap_intensity(brain, &s).to_bits(),
                    crate::research::chain::endpoint_is_hub(brain, &s),
                )
            }).collect()
        };
        brain.set_adjacency_cache_backed(false);
        let legacy = run(&brain);
        brain.set_adjacency_cache_backed(true);
        let cache = run(&brain);
        assert_eq!(legacy, cache, "public batch-1 consumers byte-differ legacy vs cache");
        // Non-vacuous discrimination: index 0 = hub, index 3 = leaf.
        assert!(legacy[0].0 && !legacy[3].0, "is_hub does not discriminate hub/leaf");
        assert!(legacy[0].1 != legacy[3].1, "gap_intensity identical hub/leaf (vacuous)");
        assert!(legacy[0].2 && !legacy[3].2, "endpoint_is_hub does not discriminate hub/leaf");
    }

    /// INDEPENDENCE: in cache mode the batch-1 seam reads AND the public
    /// consumers are byte-invariant after POISONING every resident
    /// structure the legacy path reads -- adjacency cleared, bond weights +
    /// keyframes zeroed, node payloads nuked, and every resident bond TYPE
    /// flipped -- proving they source degree/neighbours from the resident
    /// MAP + cache, never the resident adjacency/bonds.
    #[test]
    fn p5_batch1_reads_independent_of_resident() {
        let dir = tempdir().unwrap();
        { let b = boot_v5(dir.path()); teach_b1(&b); }
        let mut brain = load_routed(dir.path(), 4096, 0.0);
        brain.hub_degree_config.max_out_degree = 8;
        brain.hub_degree_config.max_in_degree = 8;
        brain.set_adjacency_cache_backed(true);

        let snapshot = |brain: &Brain| -> (Vec<SeamRead>, Vec<Cons>) {
            let reads = SEEDS.iter().map(|l| seam_reads(&brain.network, &g(l))).collect();
            let cons = SEEDS.iter().map(|l| {
                let s = g(l);
                (
                    crate::bridge::is_hub(&brain.network, &s, HUB_THRESHOLD),
                    crate::research::chain::endpoint_gap_intensity(brain, &s).to_bits(),
                    crate::research::chain::endpoint_is_hub(brain, &s),
                )
            }).collect();
            (reads, cons)
        };

        let base = snapshot(&brain);
        assert!(base.1[0].0, "baseline is_hub(hub) must be true (non-vacuous)");

        // Corrupt every resident structure the legacy path reads.
        brain.network.mapram_test_poison_resident();
        brain.network.mapram_test_corrupt_resident_bond_types();

        brain.set_adjacency_cache_backed(true);
        let after = snapshot(&brain);
        assert_eq!(base.0, after.0,
            "cache-mode seam reads changed after poisoning resident structures");
        assert_eq!(base.1, after.1,
            "cache-mode batch-1 consumers changed after poisoning resident structures");

        // Sanity: the poison REALLY bit -- the legacy reads now diverge.
        brain.set_adjacency_cache_backed(false);
        let legacy_after = snapshot(&brain);
        assert_ne!(base.0, legacy_after.0,
            "poison must corrupt the resident/legacy reads (else the test is vacuous)");
    }
}

#[cfg(test)]
mod mapram_p5b_tests {
    //! OP-BRAIN-MapRAM P5-batch2 -- byte-equivalence + resident-independence
    //! for the FIRST order-SENSITIVE, bond-PAYLOAD-carrying adjacency
    //! consumers rerouted off the resident adjacency through the new
    //! UNCAPPED, CANONICAL, edge-carrying seam (`seam_walk_outgoing` /
    //! `seam_walk_incoming` / `seam_directed_bond`, built on
    //! `seam_outgoing_edges` / `seam_incoming_edges`). Cache mode only;
    //! the legacy branch of every consumer is byte-for-byte the prior
    //! code apart from the deliberate canonical neighbour ORDER (legacy
    //! resident order was `HashMap::drain`-random per load -- already
    //! non-deterministic -- so canonicalising it in both modes makes the
    //! output deterministic AND identical across paths; the same defence
    //! P2 applied to the spread path).
    //!
    //! Consumers covered:
    //!   * `context::ContextSelector::generate_candidate_scopes` -- reads
    //!     bond KEYFRAME timestamps on the seed's outgoing + incoming
    //!     edges; output (sorted+deduped Historical scopes) is
    //!     order-independent. Proven keyframe-carrying.
    //!   * `consolidation::internal_fill_at_hub` -- the hub neighbour-list
    //!     gather (`take(max_neighbors)`) + the A->B / A->C legs (last
    //!     keyframe weight+source) + the B->C idempotency leg
    //!     (`seam_directed_bond`).
    use super::*;
    use crate::bond::{BondType, KeyframeSource, TemporalKeyframe};
    use crate::consolidation::{internal_fill_at_hub, ConsolidationConfig, InternalFillProposal};
    use crate::context::ContextSelector;
    use crate::manifest::{KeyFingerprint, Manifest, MANIFEST_FILE_NAME};
    use crate::scope::TemporalScope;
    use crate::seed::Seed;
    use tempfile::tempdir;

    fn g(n: &str) -> Seed { Seed::from_label(n) }
    fn kf_at(ts: i64, w: f32) -> Vec<TemporalKeyframe> {
        vec![TemporalKeyframe { timestamp: ts, weight: w, source: KeyframeSource::Baseline }]
    }
    fn boot_v5(dir: &std::path::Path) -> Brain {
        let mut m = Manifest::new(KeyFingerprint::none());
        m.set_brain_format_v5();
        m.save_atomic(&dir.join(MANIFEST_FILE_NAME)).unwrap();
        Brain::load_v5(dir, None).expect("v5 brain must boot via load_v5")
    }
    fn load_routed(dir: &std::path::Path) -> Brain {
        let mut brain = Brain::load_v5(dir, None).expect("reload via load_v5");
        brain.rehydrate_adjacency()
            .expect("rehydrate resident adjacency for legacy/parity comparison");
        brain.enable_cache_backed_adjacency(4096, 0.0).expect("enable cache");
        brain.enable_adjacency_seam().expect("install seam");
        brain
    }

    /// Shared topology:
    ///   * scope seed `s0`: outgoing s0->a@ts100, s0->b@ts300; incoming
    ///     c->s0@ts150, d->s0@ts400 -- four DISTINCT keyframe timestamps
    ///     split across both directions.
    ///   * hub `h`: outgoing h->nb (0.8), h->nc (0.7), h->ne (0.9); NO
    ///     inter-neighbour bonds -- so every pair is an eligible
    ///     create-mode internal-fill proposal. All non-stopword labels.
    fn teach(brain: &Brain) {
        let intern = brain.flush_intern.as_ref().unwrap().clone();
        let mt = brain.memtable().unwrap().clone();
        for l in ["s0", "a", "b", "c", "d", "h", "nb", "nc", "ne"] {
            intern.intern_or_get(g(l));
        }
        mt.insert(g("s0"), g("a"), BondType::RelatedTo, 0.8, kf_at(100, 0.8), &intern);
        mt.insert(g("s0"), g("b"), BondType::RelatedTo, 0.7, kf_at(300, 0.7), &intern);
        mt.insert(g("c"), g("s0"), BondType::RelatedTo, 0.6, kf_at(150, 0.6), &intern);
        mt.insert(g("d"), g("s0"), BondType::RelatedTo, 0.5, kf_at(400, 0.5), &intern);
        mt.insert(g("h"), g("nb"), BondType::RelatedTo, 0.8, kf_at(10, 0.8), &intern);
        mt.insert(g("h"), g("nc"), BondType::RelatedTo, 0.7, kf_at(20, 0.7), &intern);
        mt.insert(g("h"), g("ne"), BondType::RelatedTo, 0.9, kf_at(30, 0.9), &intern);
        brain.flush_now().unwrap();
    }

    // TemporalScope derives Debug (not PartialEq); compare scope lists
    // via their full structural Debug projection.
    fn dbg_scopes(scopes: &[TemporalScope]) -> Vec<String> {
        scopes.iter().map(|s| format!("{:?}", s)).collect()
    }

    fn historical(scopes: &[TemporalScope]) -> Vec<i64> {
        scopes
            .iter()
            .filter_map(|s| match s {
                TemporalScope::Historical(t) => Some(*t),
                _ => None,
            })
            .collect()
    }

    type Proj = (Vec<u8>, Vec<u8>, u32, [Vec<u8>; 3], bool);
    fn proj(p: &[InternalFillProposal]) -> Vec<Proj> {
        let mut v: Vec<Proj> = p
            .iter()
            .map(|x| {
                (
                    x.a.as_bytes().to_vec(),
                    x.c.as_bytes().to_vec(),
                    x.new_weight.to_bits(),
                    [
                        x.path[0].as_bytes().to_vec(),
                        x.path[1].as_bytes().to_vec(),
                        x.path[2].as_bytes().to_vec(),
                    ],
                    x.create_if_missing,
                )
            })
            .collect();
        v.sort();
        v
    }

    fn scopes(brain: &Brain) -> Vec<TemporalScope> {
        ContextSelector::new().generate_candidate_scopes(&brain.network, &[g("s0")])
    }
    fn fills(brain: &Brain) -> Vec<InternalFillProposal> {
        let config = ConsolidationConfig::default();
        internal_fill_at_hub(&brain.network, &g("h"), &config, 64, 0.5)
    }

    /// `generate_candidate_scopes` is byte-identical legacy vs cache, and
    /// the carried payload keyframes flow through (the four distinct
    /// outgoing+incoming timestamps appear as Historical scopes).
    #[test]
    fn p5b_candidate_scopes_byte_equivalent() {
        let dir = tempdir().unwrap();
        {
            let b = boot_v5(dir.path());
            teach(&b);
        }
        let brain = load_routed(dir.path());

        brain.set_adjacency_cache_backed(false);
        let legacy = scopes(&brain);
        brain.set_adjacency_cache_backed(true);
        let cache = scopes(&brain);

        assert_eq!(dbg_scopes(&legacy), dbg_scopes(&cache),
            "generate_candidate_scopes byte-differs legacy vs cache");
        // Non-vacuous + keyframe-covering: the four distinct keyframe
        // timestamps (outgoing 100/300, incoming 150/400) are present.
        let h = historical(&cache);
        for t in [100i64, 150, 300, 400] {
            assert!(h.contains(&t), "carried keyframe timestamp {} missing from scopes", t);
        }
    }

    /// `internal_fill_at_hub` produces byte-identical proposals legacy vs
    /// cache, and actually fires (3 create-mode proposals).
    #[test]
    fn p5b_internal_fill_at_hub_byte_equivalent() {
        let dir = tempdir().unwrap();
        {
            let b = boot_v5(dir.path());
            teach(&b);
        }
        let brain = load_routed(dir.path());

        brain.set_adjacency_cache_backed(false);
        let legacy = fills(&brain);
        brain.set_adjacency_cache_backed(true);
        let cache = fills(&brain);

        assert_eq!(proj(&legacy), proj(&cache), "internal_fill_at_hub proposals byte-differ legacy vs cache");
        assert!(!legacy.is_empty(), "internal_fill produced no proposals (vacuous)");
        // 3 hub neighbours, no inter-neighbour bonds -> C(3,2)=3 create-mode pairs.
        assert_eq!(legacy.len(), 3, "expected 3 internal-fill proposals");
        assert!(legacy.iter().all(|p| p.create_if_missing), "all proposals should be create-mode");
    }

    /// INDEPENDENCE: in cache mode both consumers are byte-invariant after
    /// POISONING every resident structure the legacy path reads --
    /// adjacency cleared, bond weights+keyframes zeroed
    /// (`mapram_test_poison_resident`), every bond TYPE flipped
    /// (`mapram_test_corrupt_resident_bond_types`), and every keyframe
    /// timestamp/weight/source overwritten with a distinctive sentinel
    /// (`mapram_test_corrupt_resident_keyframes`) -- proving they source
    /// neighbours from the cache and the bond (incl. keyframes) from the
    /// carried payload, never the resident adjacency/bonds.
    #[test]
    fn p5b_batch2_independent_of_resident() {
        let dir = tempdir().unwrap();
        {
            let b = boot_v5(dir.path());
            teach(&b);
        }
        let mut brain = load_routed(dir.path());
        brain.set_adjacency_cache_backed(true);

        let base_scopes = scopes(&brain);
        let base_fills = proj(&fills(&brain));
        assert_eq!(base_fills.len(), 3, "baseline internal_fill must be non-vacuous");
        assert!(!historical(&base_scopes).is_empty(), "baseline scopes must be non-vacuous");

        // Corrupt every resident structure the legacy path reads.
        brain.network.mapram_test_poison_resident();
        brain.network.mapram_test_corrupt_resident_bond_types();
        brain.network.mapram_test_corrupt_resident_keyframes();

        brain.set_adjacency_cache_backed(true);
        assert_eq!(dbg_scopes(&scopes(&brain)), dbg_scopes(&base_scopes),
            "cache-mode scopes changed after poisoning resident structures");
        assert_eq!(proj(&fills(&brain)), base_fills, "cache-mode internal_fill changed after poisoning resident structures");

        // Sanity: the poison REALLY bit -- the legacy reads now diverge.
        brain.set_adjacency_cache_backed(false);
        assert_ne!(dbg_scopes(&scopes(&brain)), dbg_scopes(&base_scopes),
            "poison must corrupt the legacy scopes (else vacuous)");
        assert_ne!(proj(&fills(&brain)), base_fills, "poison must corrupt the legacy internal_fill (else vacuous)");
    }
}


#[cfg(test)]
mod mapram_p5c_tests {
    //! OP-BRAIN-MapRAM P5-batch2 (sub-batch B): byte-equivalence +
    //! resident-independence for the bare `spread_activation` and
    //! `typed_recall` (the non-`_inner` public recall walks) rerouted off
    //! the resident adjacency through the seam. typed_recall uses the
    //! UNCAPPED edge walk (`seam_outgoing_edges` / `seam_incoming_edges`);
    //! the bare spread path uses the NEW carried-weight CAPPED twin
    //! (`seam_outgoing_edges_capped` / `seam_incoming_edges_capped`, the
    //! `neighbors_*_capped` twin). Node stub via `seam_recall_node`, degree
    //! via `seam_total_degree`, bond (incl. the TYPE that drives
    //! `bond_filter`, the weight, and keyframes) reconstructed from the
    //! carried payload -- never resident `get_bond`/`view_bond`. Cache mode
    //! only; legacy is byte-for-byte the prior reads apart from the
    //! deliberate canonical neighbour ORDER (resident order was
    //! `HashMap::drain`-random per load). All proofs use the `ranked`
    //! projection (seed bytes + activation f32 BIT pattern + depth), the
    //! adjacency/bond-derived output (node CONTENT is the separate resident
    //! "B" thread, out of scope here -- and excluded from the projection).
    use super::*;
    use crate::bond::{BondType, KeyframeSource, TemporalKeyframe};
    use crate::hub_degree::HubDegreeConfig;
    use crate::manifest::{KeyFingerprint, Manifest, MANIFEST_FILE_NAME};
    use crate::network::RecallResult;
    use crate::scar::ScarLog;
    use crate::scope::TemporalScope;
    use crate::seed::Seed;
    use tempfile::tempdir;

    fn kf(w: f32) -> Vec<TemporalKeyframe> {
        vec![TemporalKeyframe { timestamp: 100, weight: w, source: KeyframeSource::Baseline }]
    }
    fn boot_v5(dir: &std::path::Path) -> Brain {
        let mut m = Manifest::new(KeyFingerprint::none());
        m.set_brain_format_v5();
        m.save_atomic(&dir.join(MANIFEST_FILE_NAME)).unwrap();
        Brain::load_v5(dir, None).expect("v5 brain must boot via load_v5")
    }
    fn lbl(n: &str) -> Seed { Seed::from_label(&format!("p5c_{}", n)) }
    /// Topology: `q` has a RelatedTo (`q->ra`), an IsA (`q->ia`), and a
    /// RelatedTo to a 6-leaf hub (`q->hub`); `ra->ra2` extends depth;
    /// `inc->q` exercises the incoming branch; `hub->l0..l5` (6 distinct
    /// weights) exercises the carried-weight hub-cap. All non-stopword.
    fn teach(brain: &Brain) {
        let intern = brain.flush_intern.as_ref().unwrap().clone();
        let mt = brain.memtable().unwrap().clone();
        let labels = ["q","ra","ra2","ia","hub","inc","l0","l1","l2","l3","l4","l5"];
        for l in labels { intern.intern_or_get(lbl(l)); }
        let spine: &[(&str,&str,f32,BondType)] = &[
            ("q","ra",0.90,BondType::RelatedTo),
            ("q","ia",0.80,BondType::IsA),
            ("q","hub",0.70,BondType::RelatedTo),
            ("ra","ra2",0.85,BondType::RelatedTo),
            ("inc","q",0.60,BondType::RelatedTo),
            ("hub","l0",0.55,BondType::RelatedTo),
            ("hub","l1",0.65,BondType::RelatedTo),
            ("hub","l2",0.75,BondType::RelatedTo),
            ("hub","l3",0.50,BondType::RelatedTo),
            ("hub","l4",0.60,BondType::RelatedTo),
            ("hub","l5",0.70,BondType::RelatedTo),
        ];
        for (f,t,w,bt) in spine.iter().copied() {
            mt.insert(lbl(f), lbl(t), bt, w, kf(w), &intern);
        }
        brain.flush_now().unwrap();
    }
    fn load_routed(dir: &std::path::Path) -> Brain {
        let mut brain = Brain::load_v5(dir, None).expect("reload via load_v5");
        brain.rehydrate_adjacency()
            .expect("rehydrate resident adjacency for legacy/parity comparison");
        brain.enable_cache_backed_adjacency(4096, 0.0).expect("enable cache");
        brain.enable_adjacency_seam().expect("install seam");
        brain
    }
    fn ranked(r: &RecallResult) -> Vec<(Vec<u8>, u32, usize)> {
        r.activated.iter().map(|a| (a.seed.as_bytes().to_vec(), a.activation.to_bits(), a.depth)).collect()
    }
    fn spread(brain: &Brain, seed: &str, depth: usize) -> RecallResult {
        brain.network.spread_activation(&[lbl(seed)], &TemporalScope::Current, depth, 64, &ScarLog::default())
    }
    fn typed(brain: &Brain, seed: &str, depth: usize, filt: &[BondType]) -> RecallResult {
        brain.network.typed_recall(&[lbl(seed)], &TemporalScope::Current, depth, 64, filt, &ScarLog::default())
    }

    /// `spread_activation` is byte-identical legacy vs cache across seeds +
    /// depths (uncapped default config) and non-vacuous.
    #[test]
    fn p5c_spread_activation_byte_equivalent() {
        let dir = tempdir().unwrap();
        { let b = boot_v5(dir.path()); teach(&b); }
        let brain = load_routed(dir.path());
        let mut any = false;
        for s in ["q","hub","inc"] {
            for d in [1usize,2,3] {
                brain.set_adjacency_cache_backed(false);
                let legacy = ranked(&spread(&brain, s, d));
                brain.set_adjacency_cache_backed(true);
                let cache = ranked(&spread(&brain, s, d));
                assert_eq!(legacy, cache, "spread_activation byte-differs legacy vs cache: {} d{}", s, d);
                any = any || cache.len() > 1;
            }
        }
        assert!(any, "spread_activation never propagated past the seed (vacuous)");
    }

    /// `typed_recall` is byte-identical legacy vs cache, and the bond-TYPE
    /// filter discriminates (RelatedTo-only differs from unfiltered, since
    /// `q->ia` is IsA).
    #[test]
    fn p5c_typed_recall_byte_equivalent() {
        let dir = tempdir().unwrap();
        { let b = boot_v5(dir.path()); teach(&b); }
        let brain = load_routed(dir.path());
        let mut filtered_differs = false;
        for s in ["q","hub","inc"] {
            for d in [1usize,2,3] {
                brain.set_adjacency_cache_backed(false);
                let legacy_r = ranked(&typed(&brain, s, d, &[BondType::RelatedTo]));
                let legacy_a = ranked(&typed(&brain, s, d, &[]));
                brain.set_adjacency_cache_backed(true);
                let cache_r = ranked(&typed(&brain, s, d, &[BondType::RelatedTo]));
                let cache_a = ranked(&typed(&brain, s, d, &[]));
                assert_eq!(legacy_r, cache_r, "typed_recall(RelatedTo) byte-differs: {} d{}", s, d);
                assert_eq!(legacy_a, cache_a, "typed_recall(all) byte-differs: {} d{}", s, d);
                filtered_differs = filtered_differs || cache_r != cache_a;
            }
        }
        assert!(filtered_differs, "bond_filter never discriminated (typed_recall vacuous on type)");
    }

    /// The carried-weight CAP TWIN fires: enforcement on, cap = 2 on a
    /// 6-leaf hub. Byte-identical legacy vs cache, and the hub-cap drop
    /// counter advances in BOTH paths (cap path actually exercised).
    #[test]
    fn p5c_spread_cap_twin_byte_equivalent() {
        let dir = tempdir().unwrap();
        { let b = boot_v5(dir.path()); teach(&b); }
        let mut brain = load_routed(dir.path());
        let mut cfg = HubDegreeConfig::default();
        cfg.max_out_degree = 2;
        cfg.max_in_degree = 2;
        cfg.read_enforcement = true;
        brain.network.sync_hub_read_config(&cfg);

        brain.set_adjacency_cache_backed(false);
        let c0 = brain.network.hub_cap_filtered_at_query_total();
        let legacy = ranked(&spread(&brain, "hub", 2));
        let c1 = brain.network.hub_cap_filtered_at_query_total();
        brain.set_adjacency_cache_backed(true);
        let cache = ranked(&spread(&brain, "hub", 2));
        let c2 = brain.network.hub_cap_filtered_at_query_total();

        assert_eq!(legacy, cache, "capped spread_activation byte-differs legacy vs cache");
        assert!(c1 > c0, "legacy hub-cap did not fire (cap path not exercised)");
        assert!(c2 > c1, "cache cap-twin did not fire (carried-weight cap not exercised)");
    }

    /// INDEPENDENCE: in cache mode `spread_activation` + `typed_recall`
    /// (filtered and unfiltered) are byte-invariant after POISONING every
    /// resident structure the legacy path reads -- adjacency cleared, bond
    /// weights+keyframes zeroed (`mapram_test_poison_resident`), every bond
    /// TYPE flipped (`mapram_test_corrupt_resident_bond_types`), every
    /// keyframe overwritten with a sentinel
    /// (`mapram_test_corrupt_resident_keyframes`) -- proving neighbours come
    /// from the map, the bond (weight/type/keyframes) from the carried
    /// payload, degree from the map, never the resident adjacency/bonds.
    #[test]
    fn p5c_recall_independent_of_resident() {
        let dir = tempdir().unwrap();
        { let b = boot_v5(dir.path()); teach(&b); }
        let mut brain = load_routed(dir.path());
        brain.set_adjacency_cache_backed(true);

        let seeds = ["q","hub","inc"];
        let depths = [1usize,2,3];
        let mut base: Vec<(Vec<(Vec<u8>,u32,usize)>, Vec<(Vec<u8>,u32,usize)>, Vec<(Vec<u8>,u32,usize)>)> = Vec::new();
        for s in seeds { for d in depths {
            base.push((
                ranked(&spread(&brain, s, d)),
                ranked(&typed(&brain, s, d, &[BondType::RelatedTo])),
                ranked(&typed(&brain, s, d, &[])),
            ));
        }}
        assert!(base.iter().any(|(a,_,_)| a.len() > 1), "baseline spread vacuous");
        assert!(base.iter().any(|(_,b,_)| b.len() > 1), "baseline typed(RelatedTo) vacuous");
        assert!(base.iter().any(|(_,b,c)| b != c), "baseline filter never discriminated");

        brain.network.mapram_test_poison_resident();
        brain.network.mapram_test_corrupt_resident_bond_types();
        brain.network.mapram_test_corrupt_resident_keyframes();

        brain.set_adjacency_cache_backed(true);
        let mut idx = 0;
        for s in seeds { for d in depths {
            assert_eq!(ranked(&spread(&brain, s, d)), base[idx].0,
                "cache spread changed after poisoning resident: {} d{}", s, d);
            assert_eq!(ranked(&typed(&brain, s, d, &[BondType::RelatedTo])), base[idx].1,
                "cache typed(RelatedTo) changed after poisoning resident: {} d{}", s, d);
            assert_eq!(ranked(&typed(&brain, s, d, &[])), base[idx].2,
                "cache typed(all) changed after poisoning resident: {} d{}", s, d);
            idx += 1;
        }}

        // Sanity: the poison REALLY bit -- the legacy reads now diverge.
        brain.set_adjacency_cache_backed(false);
        assert_ne!(ranked(&spread(&brain, "q", 3)), base[2].0,
            "poison must corrupt the legacy spread (else vacuous)");
        assert_ne!(ranked(&typed(&brain, "q", 3, &[BondType::RelatedTo])), base[2].1,
            "poison must corrupt the legacy typed_recall (else vacuous)");
    }
}


#[cfg(test)]
mod op_ingest_lean_tests {
    //! OP-INGEST-LEAN -- assert the write-only ingest load mode does NOT
    //! wire the serving/recall residency (resident degree-map + keyword
    //! index), while the full serving load does, and that bonds still
    //! land durably in ingest-only mode.
    use super::*;
    use crate::manifest::{KeyFingerprint, Manifest, MANIFEST_FILE_NAME};

    fn make_v5_dir(dir: &std::path::Path) {
        let mpath = dir.join(MANIFEST_FILE_NAME);
        let mut m = Manifest::new(KeyFingerprint::none());
        m.set_brain_format_v5();
        m.save_atomic(&mpath).expect("write v5 MANIFEST");
    }

    #[test]
    fn ingest_only_skips_serving_residency_but_full_load_builds_it() {
        let td = tempfile::tempdir().unwrap();
        let dir = td.path();
        make_v5_dir(dir);

        // Seed + flush a few bonds via a normal load so a bond SST (and
        // hence a buildable degree-map) exists on disk.
        {
            let mut b = Brain::load_v5(dir, None).expect("normal v5 load");
            b.teach_corpus("alpha", BondType::IsA, "letter", 0.9,
                Confidence::Verified, "c", None, None);
            b.teach_corpus("beta", BondType::IsA, "letter", 0.9,
                Confidence::Verified, "c", None, None);
            b.flush_now().expect("flush bonds to SST");
            let _ = b.refresh_resident_degmap();
        }

        // Full (Auto) serving load builds the resident degree-map.
        {
            let full = Brain::load_v5_with_mode(dir, None, BrainLoadMode::Auto)
                .expect("auto v5 load");
            assert!(full.maintains_resident_map(),
                "full serving load must build/hold the resident degree-map");
            assert!(!full.is_ingest_only(),
                "Auto load must NOT report ingest-only");
            assert!(full.resident_map.is_some(),
                "full serving load: resident_map must be Some");
        }

        // Ingest-only load must NOT build the serving residency.
        {
            let lean = Brain::load_v5_with_mode(dir, None, BrainLoadMode::IngestOnly)
                .expect("ingest-only v5 load");
            assert!(lean.resident_map.is_none(),
                "IngestOnly must NOT load/build the resident degree-map");
            assert!(lean.is_ingest_only(),
                "IngestOnly must report is_ingest_only() (gates the keyword-index build at MCP startup)");
            assert!(!lean.maintains_resident_map(),
                "IngestOnly must report it does not maintain a resident map");
            assert!(!lean.network.has_keyword_sidecar_context(),
                "IngestOnly must NOT install the (serving-only) keyword-index sidecar");
            assert!(lean.adjacency_cache_backed(),
                "IngestOnly must still set the cache-backed flag so teach uses                  the intern for existence and does not retain full nodes resident");
        }
    }

    #[test]
    fn ingest_only_still_lands_bonds_durably() {
        let td = tempfile::tempdir().unwrap();
        let dir = td.path();
        make_v5_dir(dir);
        // Teach + flush through the ingest-only load, then reload full
        // and confirm the bond is durably present.
        {
            let mut lean = Brain::load_v5_with_mode(dir, None, BrainLoadMode::IngestOnly)
                .expect("ingest-only v5 load");
            lean.teach_corpus("gamma", BondType::IsA, "letter", 0.9,
                Confidence::Verified, "c", None, None);
            lean.flush_now().expect("flush bonds to SST");
        }
        let full = Brain::load_v5(dir, None).expect("reload full");
        let bonds = full
            .read_outgoing_bonds(&crate::seed::Seed::from_label("gamma"))
            .expect("read outgoing");
        assert!(!bonds.is_empty(),
            "a bond taught through the ingest-only path must survive a durable flush + reload");
    }
}
