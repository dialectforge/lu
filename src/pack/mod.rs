//! Port Cat.4 — Signed Safety Pack Format (knowledge packs).
//!
//! A `.lupack` file is a self-describing binary blob that carries
//! a curated bundle of bonds, signed by a trusted publisher. Packs
//! are the on-disk format that lets Lu add knowledge at runtime
//! without retraining — the differentiator vs every LLM competitor
//! (Learning #563). Cat.4 ships the FORMAT and the LEDGER; the
//! subscription / DRM / URL distribution business model rides on
//! this format in post-retrain Port V.
//!
//! Layout (see `Pack::encode` / `Pack::decode` for canonical bytes):
//!
//! ```text
//! HEADER:
//!   magic           : 8 bytes  "LU4PACK\0"
//!   version         : u32      (1 for this format)
//!   manifest_len    : u32      (bincode-serialized PackManifest length)
//!
//! MANIFEST           : bincode(PackManifest)   (manifest_len bytes)
//! CONTENT            : bincode(PackContent)    (rest of file minus 64-byte sig)
//! FOOTER:
//!   signature       : 64 bytes (Ed25519 over header || manifest || content)
//! ```
//!
//! Cross-references:
//!   * Solution #79 / Learning #621 / Port T.7 — at-rest encryption.
//!     The pack content rides through the same DF-XOR + zstd path
//!     used for the v3enc brain envelope, just keyed off the user's
//!     master.key (so the producer + consumer share the same key —
//!     pre-Port-V single-user model).
//!   * Port T.4 (v3 section pattern) — `PackLedgerSnapshot` is the
//!     new section #13 (`SectionIdV3::PackLedger`). Forward-compat
//!     via `de_bincode_optional`; pre-Cat.4 brains load with empty.
//!   * Port T.3 (PendingLog) — pack ingest does NOT auto-merge bonds.
//!     Every candidate bond becomes a `PendingChange::AddBond` with
//!     `SourceKind::Teach { actor: "pack_install:<pack_id>" }`. The
//!     human still confirms via `lu4d_approve_pending`.
//!   * Port S / S.1 — source attribution + revoke-by-source.
//!     `lu4d_pack_uninstall` revokes by Teach-actor predicate that
//!     matches the synthetic `pack_install:<pack_id>` actor string.
//!   * Learning #637 — Cat.4 design (signed-publisher trust model).
//!
//! Anti-hallucination invariant: pack contents go through PendingLog,
//! NOT direct-merge into the canonical brain. Even valid signatures
//! from trusted publishers don't bypass human review. A future
//! `--auto-approve-trusted` config flag could change this for power
//! users; out of scope for first ship.
//!
//! Known limitation (deferred to Cat.4.1): `SourceKind::PackInstall`
//! is emulated via `SourceKind::Teach { actor: "pack_install:<pack_id>" }`
//! (same dodge Port LB used for `SessionPromote`). Adding a proper
//! enum variant would break serde forward-compat for every pre-Cat.4
//! binary reading a Cat.4-saved brain. The first-class variant lands
//! in the next-retrain schema.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;

use crate::bond::BondType;
use crate::encrypt::MasterKey;

// ---------------------------------------------------------------------
// Format constants
// ---------------------------------------------------------------------

/// Magic bytes at the head of every `.lupack` file. Eight bytes
/// (the trailing NUL is part of the magic, not a C-string terminator).
pub const PACK_MAGIC: [u8; 8] = *b"LU4PACK\0";

/// Format version this build emits AND accepts. Bump for any
/// backward-incompatible layout change.
pub const PACK_FORMAT_VERSION: u32 = 1;

/// Length of the Ed25519 signature footer in bytes.
pub const PACK_SIG_LEN: usize = 64;

/// Header byte length: magic (8) + version (4) + manifest_len (4).
pub const PACK_HEADER_LEN: usize = 16;

/// Block index used when XOR'ing pack content against the master key.
/// Distinct from the `0` block used by network bonds in `brain_io_v3`
/// so a pack file's ciphertext doesn't collide with a brain section's
/// keystream.
pub const PACK_ENCRYPT_BLOCK_INDEX: u64 = 0xCA40_5041_434B_0001;

/// Lu version this format binds against. Packs whose
/// `manifest.minimum_lu_version` is greater than this string are
/// rejected on install (semver string compare via `compare_lu_versions`).
pub const CURRENT_LU_VERSION: &str = env!("CARGO_PKG_VERSION");

// ---------------------------------------------------------------------
// Manifest
// ---------------------------------------------------------------------

/// A bond proposed by a pack. Round-trips through bincode inside
/// `PackContent`; on install each entry becomes a
/// `PendingChange::AddBond` routed through T.3.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PackBond {
    pub subject_text: String,
    pub object_text: String,
    pub bond_type: BondType,
    pub weight: f32,
    #[serde(default)]
    pub negated: bool,
    /// Confidence in the bond, 0.0..=1.0. Mapped to the
    /// `crate::node::Confidence` tier at apply time.
    pub confidence: f32,
    /// Optional human-readable rationale for the bond, surfaced in
    /// the PendingChange so the user knows why the pack proposed it.
    #[serde(default)]
    pub reason: String,
}

/// Pack manifest — the human-readable + machine-checkable header.
/// Bincode-serialized between `PACK_HEADER_LEN` and the start of
/// the content blob.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PackManifest {
    /// Globally-unique pack identifier, conventionally
    /// `<publisher>.<pack-name>`, e.g. `"df.medical-icd10"`.
    pub pack_id: String,
    /// Hex-encoded Ed25519 pubkey of the publisher (must match an
    /// entry in `PackLedgerSnapshot::trusted_publishers`).
    pub publisher_id: String,
    /// Semver of the pack itself, e.g. `"1.0.0"`.
    pub pack_version: String,
    /// Short human-readable summary, e.g. `"10K medical ICD-10 bonds"`.
    pub content_summary: String,
    pub bond_count: u64,
    pub node_count: u64,
    pub signed_at_ms: u64,
    /// Minimum Lu version required to install this pack.
    pub minimum_lu_version: String,
    /// Pack ids this pack depends on. Empty on first ship — full
    /// dependency resolution lands in Port V.
    #[serde(default)]
    pub depends_on_pack_ids: Vec<String>,
    /// Informational license tag ("MIT", "proprietary", ...).
    #[serde(default)]
    pub license: String,
}

/// Encrypted-then-bincode-encoded payload: the actual bond list a
/// pack ships. Wrapped in `PackContent` so future content shapes
/// (nodes-only packs, scar packs, etc.) can be added without
/// breaking format version 1.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PackContent {
    pub bonds: Vec<PackBond>,
    /// Free-form metadata the publisher wants to ship. Reserved for
    /// future use; ignored on install today.
    #[serde(default)]
    pub extra: BTreeMap<String, String>,
}

impl PackContent {
    pub fn new(bonds: Vec<PackBond>) -> Self {
        Self { bonds, extra: BTreeMap::new() }
    }
}

// ---------------------------------------------------------------------
// Pack file (encode + decode)
// ---------------------------------------------------------------------

/// In-memory representation of a `.lupack` file. Use `Pack::encode`
/// to render bytes (signed by a publisher's secret key) and
/// `Pack::decode_unverified` / `Pack::decode_and_verify` to read.
#[derive(Debug, Clone, PartialEq)]
pub struct Pack {
    pub manifest: PackManifest,
    pub content_ciphertext: Vec<u8>,
    pub signature: [u8; PACK_SIG_LEN],
}

impl Pack {
    /// Compute the bytes that get signed: header || manifest_bytes
    /// || content_ciphertext.
    pub(crate) fn signing_payload(
        manifest_bytes: &[u8],
        content_ciphertext: &[u8],
    ) -> Vec<u8> {
        let mut buf = Vec::with_capacity(
            PACK_HEADER_LEN + manifest_bytes.len() + content_ciphertext.len(),
        );
        buf.extend_from_slice(&PACK_MAGIC);
        buf.extend_from_slice(&PACK_FORMAT_VERSION.to_le_bytes());
        buf.extend_from_slice(&(manifest_bytes.len() as u32).to_le_bytes());
        buf.extend_from_slice(manifest_bytes);
        buf.extend_from_slice(content_ciphertext);
        buf
    }

    /// Encode self into the canonical `.lupack` byte layout.
    pub fn to_bytes(&self) -> Result<Vec<u8>, PackError> {
        let manifest_bytes = bincode::serialize(&self.manifest)
            .map_err(|e| PackError::Format(format!("manifest serialize: {}", e)))?;
        if manifest_bytes.len() > u32::MAX as usize {
            return Err(PackError::Format("manifest too large".into()));
        }
        let mut out =
            Self::signing_payload(&manifest_bytes, &self.content_ciphertext);
        out.extend_from_slice(&self.signature);
        Ok(out)
    }

    /// Parse the on-disk layout WITHOUT verifying the signature.
    /// Used internally; callers must use `decode_and_verify` to
    /// actually accept a pack into a brain.
    pub fn decode_unverified(bytes: &[u8]) -> Result<Self, PackError> {
        if bytes.len() < PACK_HEADER_LEN + PACK_SIG_LEN {
            return Err(PackError::Format(format!(
                "pack truncated: {} bytes < min {}",
                bytes.len(),
                PACK_HEADER_LEN + PACK_SIG_LEN
            )));
        }
        if bytes[0..8] != PACK_MAGIC {
            return Err(PackError::Format(format!(
                "bad magic: {:x?} (expected {:x?})",
                &bytes[0..8],
                &PACK_MAGIC
            )));
        }
        let mut v_buf = [0u8; 4];
        v_buf.copy_from_slice(&bytes[8..12]);
        let version = u32::from_le_bytes(v_buf);
        if version != PACK_FORMAT_VERSION {
            return Err(PackError::UnsupportedFormatVersion {
                got: version,
                expected: PACK_FORMAT_VERSION,
            });
        }
        let mut m_buf = [0u8; 4];
        m_buf.copy_from_slice(&bytes[12..16]);
        let manifest_len = u32::from_le_bytes(m_buf) as usize;
        let manifest_end = PACK_HEADER_LEN + manifest_len;
        let content_end = bytes.len().checked_sub(PACK_SIG_LEN).ok_or_else(|| {
            PackError::Format("pack file shorter than signature".into())
        })?;
        if manifest_end > content_end {
            return Err(PackError::Format(format!(
                "manifest_len {} overruns content end {}",
                manifest_len, content_end
            )));
        }
        let manifest: PackManifest =
            bincode::deserialize(&bytes[PACK_HEADER_LEN..manifest_end])
                .map_err(|e| PackError::Format(format!("manifest deserialize: {}", e)))?;
        let content_ciphertext = bytes[manifest_end..content_end].to_vec();
        let mut signature = [0u8; PACK_SIG_LEN];
        signature.copy_from_slice(&bytes[content_end..]);
        Ok(Pack {
            manifest,
            content_ciphertext,
            signature,
        })
    }
}

// ---------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------

#[derive(Debug)]
pub enum PackError {
    Io(std::io::Error),
    Format(String),
    UnsupportedFormatVersion { got: u32, expected: u32 },
    LuVersionTooOld { pack_minimum: String, current: String },
    UnknownPublisher { publisher_id: String },
    SignatureInvalid,
    PackRevoked { pack_id: String, reason: String },
    DecryptFailed(String),
    PublisherKey(String),
}

impl std::fmt::Display for PackError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PackError::Io(e) => write!(f, "pack io: {}", e),
            PackError::Format(s) => write!(f, "pack format: {}", s),
            PackError::UnsupportedFormatVersion { got, expected } => write!(
                f,
                "pack format version unsupported: got {}, expected {}",
                got, expected
            ),
            PackError::LuVersionTooOld { pack_minimum, current } => write!(
                f,
                "pack requires Lu >= {}, this build is {}",
                pack_minimum, current
            ),
            PackError::UnknownPublisher { publisher_id } => write!(
                f,
                "unknown publisher: {} (add to trusted_publishers first)",
                publisher_id
            ),
            PackError::SignatureInvalid => write!(f, "pack signature invalid"),
            PackError::PackRevoked { pack_id, reason } => write!(
                f,
                "pack revoked: {} ({})",
                pack_id, reason
            ),
            PackError::DecryptFailed(s) => write!(f, "pack content decrypt: {}", s),
            PackError::PublisherKey(s) => write!(f, "publisher key: {}", s),
        }
    }
}

impl std::error::Error for PackError {}

impl From<std::io::Error> for PackError {
    fn from(e: std::io::Error) -> Self { PackError::Io(e) }
}

// ---------------------------------------------------------------------
// Ledger
// ---------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrustedPublisher {
    pub pubkey_hex: String,
    pub name: String,
    pub added_at_ms: u64,
    #[serde(default)]
    pub notes: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InstalledPack {
    pub pack_id: String,
    pub publisher_id: String,
    pub pack_version: String,
    pub bond_count_added: u64,
    pub node_count_added: u64,
    pub installed_at_ms: u64,
    /// SHA-256 of the manifest bytes, hex-encoded. Lets the ledger
    /// detect (post-uninstall reinstall) that the same `pack_id` was
    /// republished with different contents.
    pub manifest_hash_hex: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RevokedPack {
    pub pack_id: String,
    pub revoked_at_ms: u64,
    pub reason: String,
}

/// On-disk shape persisted into v3 section #13 (`SectionIdV3::PackLedger`).
/// All maps are `BTreeMap` for deterministic bincode round-trips.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PackLedgerSnapshot {
    #[serde(default)]
    pub trusted_publishers: BTreeMap<String, TrustedPublisher>,
    #[serde(default)]
    pub installed_packs: BTreeMap<String, InstalledPack>,
    #[serde(default)]
    pub revoked_packs: BTreeMap<String, RevokedPack>,
    #[serde(default)]
    pub version: u32,
}

/// In-memory ledger store used by Brain. Mirrors the
/// `ConversationalStyleStore` shape from Port CS.
#[derive(Debug, Clone, Default)]
pub struct PackLedgerStore {
    pub trusted_publishers: BTreeMap<String, TrustedPublisher>,
    pub installed_packs: BTreeMap<String, InstalledPack>,
    pub revoked_packs: BTreeMap<String, RevokedPack>,
}

impl PackLedgerStore {
    pub fn new() -> Self { Self::default() }

    pub fn from_snapshot(snap: PackLedgerSnapshot) -> Self {
        Self {
            trusted_publishers: snap.trusted_publishers,
            installed_packs: snap.installed_packs,
            revoked_packs: snap.revoked_packs,
        }
    }

    pub fn snapshot(&self) -> PackLedgerSnapshot {
        PackLedgerSnapshot {
            trusted_publishers: self.trusted_publishers.clone(),
            installed_packs: self.installed_packs.clone(),
            revoked_packs: self.revoked_packs.clone(),
            version: 1,
        }
    }

    pub fn add_publisher(
        &mut self,
        pubkey_hex: &str,
        name: &str,
        notes: Option<String>,
        now_ms: u64,
    ) -> Result<TrustedPublisher, PackError> {
        let pk = pubkey_hex.trim().to_ascii_lowercase();
        if pk.len() != 64 || !pk.chars().all(|c| c.is_ascii_hexdigit()) {
            return Err(PackError::PublisherKey(format!(
                "pubkey must be 64 hex chars, got {} ({:?})",
                pk.len(), pk
            )));
        }
        let entry = TrustedPublisher {
            pubkey_hex: pk.clone(),
            name: name.to_string(),
            added_at_ms: now_ms,
            notes,
        };
        self.trusted_publishers.insert(pk, entry.clone());
        Ok(entry)
    }

    pub fn remove_publisher(&mut self, pubkey_hex: &str) -> Option<TrustedPublisher> {
        let pk = pubkey_hex.trim().to_ascii_lowercase();
        self.trusted_publishers.remove(&pk)
    }

    pub fn is_trusted(&self, pubkey_hex: &str) -> bool {
        let pk = pubkey_hex.trim().to_ascii_lowercase();
        self.trusted_publishers.contains_key(&pk)
    }

    pub fn record_install(&mut self, install: InstalledPack) {
        self.installed_packs.insert(install.pack_id.clone(), install);
    }

    pub fn record_revoke(&mut self, pack_id: &str, reason: &str, now_ms: u64) {
        self.revoked_packs.insert(pack_id.to_string(), RevokedPack {
            pack_id: pack_id.to_string(),
            revoked_at_ms: now_ms,
            reason: reason.to_string(),
        });
        self.installed_packs.remove(pack_id);
    }

    pub fn is_revoked(&self, pack_id: &str) -> bool {
        self.revoked_packs.contains_key(pack_id)
    }
}

// ---------------------------------------------------------------------
// Signing / verification
// ---------------------------------------------------------------------

/// Hex-encode bytes (lowercase, no separators).
pub fn hex_encode(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

/// Hex-decode a string into a fixed-length byte array.
pub fn hex_decode<const N: usize>(s: &str) -> Result<[u8; N], PackError> {
    let s = s.trim();
    if s.len() != N * 2 {
        return Err(PackError::Format(format!(
            "hex length mismatch: got {} expected {}",
            s.len(),
            N * 2
        )));
    }
    let mut out = [0u8; N];
    for i in 0..N {
        let hi = hex_nibble(s.as_bytes()[i * 2])?;
        let lo = hex_nibble(s.as_bytes()[i * 2 + 1])?;
        out[i] = (hi << 4) | lo;
    }
    Ok(out)
}

fn hex_nibble(b: u8) -> Result<u8, PackError> {
    match b {
        b'0'..=b'9' => Ok(b - b'0'),
        b'a'..=b'f' => Ok(b - b'a' + 10),
        b'A'..=b'F' => Ok(b - b'A' + 10),
        _ => Err(PackError::Format(format!("non-hex byte: {}", b))),
    }
}

/// Compare two semver-style strings of the form `"major.minor.patch"`.
/// Returns Ordering. Non-numeric components compare lexicographically.
pub fn compare_lu_versions(a: &str, b: &str) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    let pa: Vec<&str> = a.split('.').collect();
    let pb: Vec<&str> = b.split('.').collect();
    let n = pa.len().max(pb.len());
    for i in 0..n {
        let ap = pa.get(i).unwrap_or(&"0");
        let bp = pb.get(i).unwrap_or(&"0");
        match (ap.parse::<u64>(), bp.parse::<u64>()) {
            (Ok(x), Ok(y)) => match x.cmp(&y) {
                Ordering::Equal => continue,
                o => return o,
            },
            _ => match ap.cmp(bp) {
                Ordering::Equal => continue,
                o => return o,
            },
        }
    }
    Ordering::Equal
}

/// Generate a fresh Ed25519 keypair using the OS RNG. Returns
/// (secret_key_bytes, public_key_hex). Caller should write the
/// secret bytes to a 0600-mode keyfile and never commit them.
pub fn generate_keypair() -> ([u8; 32], String) {
    use ed25519_dalek::SigningKey;
    use rand_core::OsRng;
    let sk = SigningKey::generate(&mut OsRng);
    let pk_hex = hex_encode(sk.verifying_key().as_bytes());
    (sk.to_bytes(), pk_hex)
}

/// Build a `SigningKey` from raw 32-byte secret material. Used by
/// `pack_create` / tests.
pub fn signing_key_from_bytes(bytes: &[u8; 32]) -> ed25519_dalek::SigningKey {
    ed25519_dalek::SigningKey::from_bytes(bytes)
}

/// Build a `VerifyingKey` from a hex-encoded pubkey string. Used
/// by the install path to look up the publisher's verifier.
pub fn verifying_key_from_hex(
    pubkey_hex: &str,
) -> Result<ed25519_dalek::VerifyingKey, PackError> {
    let bytes: [u8; 32] = hex_decode(pubkey_hex)?;
    ed25519_dalek::VerifyingKey::from_bytes(&bytes).map_err(|e| {
        PackError::PublisherKey(format!("verifying key: {}", e))
    })
}

/// SHA-256 hex of a byte slice, used for `manifest_hash_hex`.
pub fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(bytes);
    hex_encode(&h.finalize())
}

// ---------------------------------------------------------------------
// Producer (pack_create)
// ---------------------------------------------------------------------

/// Inputs to `build_pack`: everything the producer side needs.
pub struct PackBuildSpec<'a> {
    pub manifest_seed: PackManifestSeed,
    pub bonds: Vec<PackBond>,
    pub signing_key: &'a ed25519_dalek::SigningKey,
    pub master_key: &'a MasterKey,
    pub now_ms: u64,
}

/// Manifest fields the caller supplies — `bond_count`, `node_count`,
/// `signed_at_ms` and `publisher_id` are derived from the bonds /
/// keypair / clock at build time.
#[derive(Debug, Clone)]
pub struct PackManifestSeed {
    pub pack_id: String,
    pub pack_version: String,
    pub content_summary: String,
    pub minimum_lu_version: String,
    pub depends_on_pack_ids: Vec<String>,
    pub license: String,
}

impl Default for PackManifestSeed {
    fn default() -> Self {
        Self {
            pack_id: "pack.unnamed".into(),
            pack_version: "0.0.1".into(),
            content_summary: "".into(),
            minimum_lu_version: CURRENT_LU_VERSION.into(),
            depends_on_pack_ids: Vec::new(),
            license: "proprietary".into(),
        }
    }
}

/// Encrypt + zstd-compress the content payload for shipment.
pub(crate) fn encrypt_content(
    content: &PackContent,
    master_key: &MasterKey,
) -> Result<Vec<u8>, PackError> {
    let raw = bincode::serialize(content)
        .map_err(|e| PackError::Format(format!("content serialize: {}", e)))?;
    let compressed = zstd::bulk::compress(&raw, 3)
        .map_err(|e| PackError::Format(format!("zstd compress: {}", e)))?;
    let mut blob = compressed;
    crate::encrypt::xor_in_place(master_key, PACK_ENCRYPT_BLOCK_INDEX, &mut blob);
    Ok(blob)
}

/// Decrypt + zstd-decompress + bincode-deserialize the content blob.
pub(crate) fn decrypt_content(
    ciphertext: &[u8],
    master_key: &MasterKey,
) -> Result<PackContent, PackError> {
    let mut blob = ciphertext.to_vec();
    crate::encrypt::xor_in_place(master_key, PACK_ENCRYPT_BLOCK_INDEX, &mut blob);
    let raw = zstd::bulk::decompress(&blob, 64 * 1024 * 1024)
        .map_err(|e| PackError::DecryptFailed(format!("zstd decompress: {}", e)))?;
    let content: PackContent = bincode::deserialize(&raw)
        .map_err(|e| PackError::DecryptFailed(format!("bincode: {}", e)))?;
    Ok(content)
}

/// Build a fully-signed `Pack` from a `PackBuildSpec`. The producer
/// tool (`lu4d_pack_create`) and the round-trip tests both use this.
pub fn build_pack(spec: PackBuildSpec<'_>) -> Result<Pack, PackError> {
    use ed25519_dalek::Signer;
    let publisher_id =
        hex_encode(spec.signing_key.verifying_key().as_bytes());
    let bond_count = spec.bonds.len() as u64;
    // Approximate node count: distinct subject + object texts.
    let mut texts: std::collections::HashSet<&str> =
        std::collections::HashSet::new();
    for b in &spec.bonds {
        texts.insert(b.subject_text.as_str());
        texts.insert(b.object_text.as_str());
    }
    let node_count = texts.len() as u64;
    let manifest = PackManifest {
        pack_id: spec.manifest_seed.pack_id,
        publisher_id,
        pack_version: spec.manifest_seed.pack_version,
        content_summary: spec.manifest_seed.content_summary,
        bond_count,
        node_count,
        signed_at_ms: spec.now_ms,
        minimum_lu_version: spec.manifest_seed.minimum_lu_version,
        depends_on_pack_ids: spec.manifest_seed.depends_on_pack_ids,
        license: spec.manifest_seed.license,
    };
    let manifest_bytes = bincode::serialize(&manifest)
        .map_err(|e| PackError::Format(format!("manifest serialize: {}", e)))?;
    let content = PackContent::new(spec.bonds);
    let content_ciphertext = encrypt_content(&content, spec.master_key)?;
    let payload = Pack::signing_payload(&manifest_bytes, &content_ciphertext);
    let sig = spec.signing_key.sign(&payload);
    Ok(Pack {
        manifest,
        content_ciphertext,
        signature: sig.to_bytes(),
    })
}

/// Write `pack` to `path` atomically (write to `<path>.tmp` then rename).
pub fn write_pack_file<P: AsRef<Path>>(pack: &Pack, path: P) -> Result<(), PackError> {
    let path = path.as_ref();
    let bytes = pack.to_bytes()?;
    let tmp = path.with_extension("lupack.tmp");
    std::fs::write(&tmp, &bytes)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// Read a `.lupack` file from disk; does NOT verify signature.
pub fn read_pack_file<P: AsRef<Path>>(path: P) -> Result<(Pack, Vec<u8>), PackError> {
    let bytes = std::fs::read(path.as_ref())?;
    let pack = Pack::decode_unverified(&bytes)?;
    Ok((pack, bytes))
}

// ---------------------------------------------------------------------
// Consumer (pack_install): verify + decrypt + route through PendingLog
// ---------------------------------------------------------------------

/// Result of a successful install — what got queued, plus the
/// `InstalledPack` ledger entry that callers should record AFTER
/// the user approves at least one of the queued PendingChanges.
#[derive(Debug)]
pub struct PackIngestResult {
    pub pack_id: String,
    pub publisher_id: String,
    pub pack_version: String,
    pub manifest_hash_hex: String,
    /// Ids of the PendingChanges enqueued. Same length as the
    /// pack's bond list (one PendingChange per bond).
    pub queued_pending_ids: Vec<u64>,
    pub queued_bond_count: u64,
    pub queued_node_count: u64,
}

/// Verify a `Pack` against a ledger, decrypt its content, and route
/// every bond as a `PendingChange::AddBond` through the T.3 path.
///
/// This is the single chokepoint that enforces the anti-hallucination
/// invariant: pack contents NEVER auto-merge. They wait for human
/// approval through `lu4d_approve_pending`.
pub fn ingest_pack(
    brain: &mut crate::brain::Brain,
    pack: &Pack,
    pack_bytes: &[u8],
    now_ms: u64,
) -> Result<PackIngestResult, PackError> {
    // 1) Format version check is already done by `decode_unverified`.
    // 2) minimum_lu_version check.
    if compare_lu_versions(&pack.manifest.minimum_lu_version, CURRENT_LU_VERSION)
        == std::cmp::Ordering::Greater
    {
        return Err(PackError::LuVersionTooOld {
            pack_minimum: pack.manifest.minimum_lu_version.clone(),
            current: CURRENT_LU_VERSION.into(),
        });
    }
    // 3) Trusted-publisher check.
    if !brain.pack_ledger.is_trusted(&pack.manifest.publisher_id) {
        return Err(PackError::UnknownPublisher {
            publisher_id: pack.manifest.publisher_id.clone(),
        });
    }
    // 4) Revocation check.
    if let Some(revoked) = brain.pack_ledger.revoked_packs.get(&pack.manifest.pack_id) {
        return Err(PackError::PackRevoked {
            pack_id: revoked.pack_id.clone(),
            reason: revoked.reason.clone(),
        });
    }
    // 5) Reconstruct signed payload from on-disk bytes (cheaper than
    //    re-serializing the manifest, and matches what was signed).
    let content_end = pack_bytes.len()
        .checked_sub(PACK_SIG_LEN)
        .ok_or_else(|| PackError::Format("pack truncated below signature".into()))?;
    let signed_slice = &pack_bytes[..content_end];
    let verifier = verifying_key_from_hex(&pack.manifest.publisher_id)?;
    let sig = ed25519_dalek::Signature::from_bytes(&pack.signature);
    use ed25519_dalek::Verifier;
    verifier
        .verify(signed_slice, &sig)
        .map_err(|_| PackError::SignatureInvalid)?;
    // 6) Decrypt content.
    let master_key = brain
        .encryption_key
        .as_ref()
        .ok_or_else(|| PackError::DecryptFailed(
            "brain has no master_key — Cat.4 pack install requires --keyfile".into(),
        ))?;
    let content = decrypt_content(&pack.content_ciphertext, master_key)?;
    // 7) Manifest hash for the ledger.
    let manifest_end = PACK_HEADER_LEN + {
        let mut buf = [0u8; 4];
        buf.copy_from_slice(&pack_bytes[12..16]);
        u32::from_le_bytes(buf) as usize
    };
    let manifest_hash_hex = sha256_hex(&pack_bytes[PACK_HEADER_LEN..manifest_end]);
    // 8) Route each bond as a PendingChange::AddBond.
    //    Port SourceKind-Variants — stamp first-class
    //    SourceKind::PackInstall { pack_id, publisher_pubkey_hex,
    //    signed_at_ms } (replaces the legacy
    //    Teach{actor:"pack_install:<pack_id>"} dodge).
    let pack_id_owned = pack.manifest.pack_id.clone();
    let publisher_pk_owned = pack.manifest.publisher_id.clone();
    let signed_at_ms = pack.manifest.signed_at_ms;
    let _ = now_ms; // PackInstall carries signed_at_ms; PendingChange.proposed_at_ms still uses now_ms below
    let mut queued: Vec<u64> = Vec::with_capacity(content.bonds.len());
    let mut texts: std::collections::HashSet<String> =
        std::collections::HashSet::new();
    for b in &content.bonds {
        texts.insert(b.subject_text.clone());
        texts.insert(b.object_text.clone());
        let change = crate::pending::PendingChange {
            pending_id: 0,
            proposed_at_ms: now_ms,
            proposed_by: crate::source::SourceAttribution::new(
                crate::source::SourceKind::PackInstall {
                    pack_id: pack_id_owned.clone(),
                    publisher_pubkey_hex: publisher_pk_owned.clone(),
                    signed_at_ms,
                },
            ),
            change_kind: crate::pending::PendingChangeKind::AddBond {
                subject_text: b.subject_text.clone(),
                object_text: b.object_text.clone(),
                bond_type: b.bond_type,
                weight: b.weight,
                negated: b.negated,
            },
            confidence: b.confidence.clamp(0.0, 1.0),
            reason: if b.reason.is_empty() {
                format!(
                    "pack:{}@{}: {} {:?} {}",
                    pack.manifest.pack_id,
                    pack.manifest.pack_version,
                    b.subject_text,
                    b.bond_type,
                    b.object_text
                )
            } else {
                b.reason.clone()
            },
            status: crate::pending::PendingStatus::Pending,
        };
        let id = brain
            .pending
            .push(change)
            .map_err(|e| PackError::Format(format!("pending push: {}", e)))?;
        queued.push(id);
    }
    Ok(PackIngestResult {
        pack_id: pack.manifest.pack_id.clone(),
        publisher_id: pack.manifest.publisher_id.clone(),
        pack_version: pack.manifest.pack_version.clone(),
        manifest_hash_hex,
        queued_pending_ids: queued,
        queued_bond_count: content.bonds.len() as u64,
        queued_node_count: texts.len() as u64,
    })
}

// ---------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------

#[cfg(test)]
mod tests;
