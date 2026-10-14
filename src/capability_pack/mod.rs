//! Port Cat.5 — Schema-only signed capability pack format.
//!
//! Mirrors Cat.4's signed *content* pack format (`src/pack/`), but
//! for CAPABILITIES — executable code that future runtimes (Cat.5.1)
//! load to extend Lu with new tools, file generators, image
//! generators, computer use, third-party integrations, etc. Cat.5
//! ships the schema-and-spec half: the on-disk layout, the publisher
//! trust model, and the install/uninstall/list ledger. Cat.5.1 ships
//! the runtime half (WASM / shared-lib / subprocess loaders).
//!
//! Layout (see `CapabilityPack::to_bytes` for the canonical bytes):
//!
//! ```text
//! HEADER:
//!   magic         : 8 bytes  "LU5PACK\0"
//!   version       : u32      (1 for this format)
//!   manifest_len  : u32      (bincode-serialized manifest length)
//!   content_len   : u32      (opaque content payload length)
//!
//! MANIFEST  : bincode(CapabilityPackManifest)   (manifest_len bytes)
//! CONTENT   : opaque bytes                      (content_len bytes)
//! FOOTER:
//!   signature   : 64 bytes  Ed25519 over `manifest_bytes || content_bytes`
//! ```
//!
//! Cross-references:
//!   * Port Cat.4 (`src/pack/`) — the parent pattern. Cat.5 is
//!     deliberately a near-copy at the format-and-ledger level so
//!     operators have one mental model for both pack types.
//!   * Port T.4 (v3 section pattern) — `CapabilityPackLedgerSnapshot`
//!     is v3 section #17 (`SectionIdV3::CapabilityPackLedger`).
//!     Forward-compat via `de_bincode_optional`; pre-Cat.5 brains load
//!     with an empty ledger, and pre-Cat.5 binaries skip the unknown
//!     section via the J.1.1 unknown-section tolerance.
//!   * Learning #654 — Cat.4 design note that motivated Cat.5.
//!
//! Anti-hallucination invariant — STRUCTURAL:
//! Cat.5 schema-only does NOT modify any bonds, does NOT call any
//! teach paths, does NOT execute any pack content. The schema is a
//! passive ledger. Bond count must be unchanged across install /
//! uninstall operations. The grep guard in tests asserts that
//! capability-pack tool bodies use ONLY `cap_pack_ledger.*`
//! mutations and never `brain.teach`, `add_bond`, or `pending.push`.
//!
//! Reuse vs separate publisher trust list:
//! Cat.5 keeps a SEPARATE `trusted_publishers` map from Cat.4.
//! Trusting a publisher to ship curated knowledge bundles (Cat.4)
//! is a different decision than trusting them to ship executable
//! code that runs in your process (Cat.5.1). Operators must opt in
//! explicitly per pack type.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

// ---------------------------------------------------------------------
// Format constants
// ---------------------------------------------------------------------

pub const PACK5_MAGIC: [u8; 8] = *b"LU5PACK\0";
pub const PACK5_FORMAT_VERSION: u32 = 1;
pub const PACK5_SIG_LEN: usize = 64;
pub const PACK5_HEADER_LEN: usize = 20;

// ---------------------------------------------------------------------
// Manifest
// ---------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PackRuntimeKind {
    Wasm,
    SharedLib,
    Subprocess,
    Reserved,
}

impl Default for PackRuntimeKind {
    fn default() -> Self { PackRuntimeKind::Wasm }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PackToolDecl {
    pub tool_name: String,
    pub description: String,
    #[serde(with = "args_schema_as_json_string")]
    pub args_schema: serde_json::Value,
}

// JSON args_schema rides bincode as a JSON-encoded string, since
// `serde_json::Value` uses `deserialize_any` which bincode does not
// support. The on-disk shape is stable across bincode and JSON;
// the public API still exposes `serde_json::Value` so callers can
// build schemas with the `json!()` macro.
mod args_schema_as_json_string {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    pub fn serialize<S: Serializer>(
        v: &serde_json::Value,
        ser: S,
    ) -> Result<S::Ok, S::Error> {
        let s = serde_json::to_string(v).map_err(serde::ser::Error::custom)?;
        s.serialize(ser)
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(
        de: D,
    ) -> Result<serde_json::Value, D::Error> {
        let s = String::deserialize(de)?;
        serde_json::from_str(&s).map_err(serde::de::Error::custom)
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PackPermissions {
    #[serde(default)]
    pub filesystem_read: Vec<String>,
    #[serde(default)]
    pub filesystem_write: Vec<String>,
    #[serde(default)]
    pub network_egress: Vec<String>,
    #[serde(default)]
    pub brain_read: bool,
    #[serde(default)]
    pub brain_write: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CapabilityPackManifest {
    pub pack_id: String,
    pub pack_version: String,
    pub publisher_pubkey_hex: String,
    pub signed_at_ms: u64,
    pub runtime_kind: PackRuntimeKind,
    #[serde(default)]
    pub tools: Vec<PackToolDecl>,
    #[serde(default)]
    pub permissions: PackPermissions,
    #[serde(default)]
    pub description: String,
}

// ---------------------------------------------------------------------
// Pack file (encode + decode)
// ---------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct CapabilityPack {
    pub manifest: CapabilityPackManifest,
    pub content: Vec<u8>,
    pub signature: [u8; PACK5_SIG_LEN],
}

impl CapabilityPack {
    pub(crate) fn signing_payload(
        manifest_bytes: &[u8],
        content_bytes: &[u8],
    ) -> Vec<u8> {
        let mut buf = Vec::with_capacity(manifest_bytes.len() + content_bytes.len());
        buf.extend_from_slice(manifest_bytes);
        buf.extend_from_slice(content_bytes);
        buf
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, CapabilityPackError> {
        let manifest_bytes = bincode::serialize(&self.manifest)
            .map_err(|e| CapabilityPackError::Format(format!("manifest serialize: {}", e)))?;
        if manifest_bytes.len() > u32::MAX as usize {
            return Err(CapabilityPackError::Format("manifest too large".into()));
        }
        if self.content.len() > u32::MAX as usize {
            return Err(CapabilityPackError::Format("content too large".into()));
        }
        let mut out = Vec::with_capacity(
            PACK5_HEADER_LEN + manifest_bytes.len() + self.content.len() + PACK5_SIG_LEN,
        );
        out.extend_from_slice(&PACK5_MAGIC);
        out.extend_from_slice(&PACK5_FORMAT_VERSION.to_le_bytes());
        out.extend_from_slice(&(manifest_bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(&(self.content.len() as u32).to_le_bytes());
        out.extend_from_slice(&manifest_bytes);
        out.extend_from_slice(&self.content);
        out.extend_from_slice(&self.signature);
        Ok(out)
    }

    pub fn decode_unverified(bytes: &[u8]) -> Result<Self, CapabilityPackError> {
        if bytes.len() < PACK5_HEADER_LEN + PACK5_SIG_LEN {
            return Err(CapabilityPackError::Format(format!(
                "pack truncated: {} bytes < min {}",
                bytes.len(),
                PACK5_HEADER_LEN + PACK5_SIG_LEN
            )));
        }
        if bytes[0..8] != PACK5_MAGIC {
            return Err(CapabilityPackError::Format(format!(
                "bad magic: {:x?} (expected {:x?})",
                &bytes[0..8],
                &PACK5_MAGIC
            )));
        }
        let mut v_buf = [0u8; 4];
        v_buf.copy_from_slice(&bytes[8..12]);
        let version = u32::from_le_bytes(v_buf);
        if version != PACK5_FORMAT_VERSION {
            return Err(CapabilityPackError::UnsupportedFormatVersion {
                got: version,
                expected: PACK5_FORMAT_VERSION,
            });
        }
        let mut m_buf = [0u8; 4];
        m_buf.copy_from_slice(&bytes[12..16]);
        let manifest_len = u32::from_le_bytes(m_buf) as usize;
        let mut c_buf = [0u8; 4];
        c_buf.copy_from_slice(&bytes[16..20]);
        let content_len = u32::from_le_bytes(c_buf) as usize;

        let expected_total =
            PACK5_HEADER_LEN + manifest_len + content_len + PACK5_SIG_LEN;
        if bytes.len() != expected_total {
            return Err(CapabilityPackError::Format(format!(
                "pack length {} != expected {} (header+manifest+content+sig)",
                bytes.len(),
                expected_total
            )));
        }
        let manifest_start = PACK5_HEADER_LEN;
        let manifest_end = manifest_start + manifest_len;
        let content_start = manifest_end;
        let content_end = content_start + content_len;
        let manifest: CapabilityPackManifest =
            bincode::deserialize(&bytes[manifest_start..manifest_end])
                .map_err(|e| CapabilityPackError::Format(format!("manifest deserialize: {}", e)))?;
        let content = bytes[content_start..content_end].to_vec();
        let mut signature = [0u8; PACK5_SIG_LEN];
        signature.copy_from_slice(&bytes[content_end..]);
        Ok(CapabilityPack {
            manifest,
            content,
            signature,
        })
    }
}

// ---------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------

#[derive(Debug)]
pub enum CapabilityPackError {
    Io(std::io::Error),
    Format(String),
    UnsupportedFormatVersion { got: u32, expected: u32 },
    UnknownPublisher { publisher_pubkey_hex: String },
    SignatureInvalid,
    PackRevoked { pack_id: String },
    PackAlreadyInstalled { pack_id: String },
    PublisherKey(String),
}

impl std::fmt::Display for CapabilityPackError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CapabilityPackError::Io(e) => write!(f, "capability pack io: {}", e),
            CapabilityPackError::Format(s) => write!(f, "capability pack format: {}", s),
            CapabilityPackError::UnsupportedFormatVersion { got, expected } => write!(
                f,
                "capability pack format version unsupported: got {}, expected {}",
                got, expected
            ),
            CapabilityPackError::UnknownPublisher { publisher_pubkey_hex } => write!(
                f,
                "unknown capability publisher: {} (add to capability trusted_publishers first)",
                publisher_pubkey_hex
            ),
            CapabilityPackError::SignatureInvalid => write!(f, "capability pack signature invalid"),
            CapabilityPackError::PackRevoked { pack_id } => write!(
                f,
                "capability pack revoked: {}",
                pack_id
            ),
            CapabilityPackError::PackAlreadyInstalled { pack_id } => write!(
                f,
                "capability pack already installed: {} (pass force=true to overwrite)",
                pack_id
            ),
            CapabilityPackError::PublisherKey(s) => write!(f, "capability publisher key: {}", s),
        }
    }
}

impl std::error::Error for CapabilityPackError {}

impl From<std::io::Error> for CapabilityPackError {
    fn from(e: std::io::Error) -> Self { CapabilityPackError::Io(e) }
}

// ---------------------------------------------------------------------
// Ledger
// ---------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrustedPublisher {
    pub pubkey_hex: String,
    pub label: String,
    pub added_at_ms: u64,
    #[serde(default)]
    pub notes: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InstalledPack {
    pub pack_id: String,
    pub pack_version: String,
    pub publisher_pubkey_hex: String,
    pub runtime_kind: PackRuntimeKind,
    pub installed_at_ms: u64,
    pub signed_at_ms: u64,
    pub manifest_hash_hex: String,
    #[serde(default)]
    pub tool_count: u32,
    #[serde(default)]
    pub source_path: String,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CapabilityPackLedgerSnapshot {
    #[serde(default)]
    pub trusted_publishers: BTreeMap<String, TrustedPublisher>,
    #[serde(default)]
    pub installed_packs: BTreeMap<String, InstalledPack>,
    #[serde(default)]
    pub revoked_packs: BTreeSet<String>,
    #[serde(default)]
    pub version: u32,
}

#[derive(Debug, Clone, Default)]
pub struct CapabilityPackLedgerStore {
    pub trusted_publishers: BTreeMap<String, TrustedPublisher>,
    pub installed_packs: BTreeMap<String, InstalledPack>,
    pub revoked_packs: BTreeSet<String>,
}

impl CapabilityPackLedgerStore {
    pub fn new() -> Self { Self::default() }

    pub fn from_snapshot(snap: CapabilityPackLedgerSnapshot) -> Self {
        Self {
            trusted_publishers: snap.trusted_publishers,
            installed_packs: snap.installed_packs,
            revoked_packs: snap.revoked_packs,
        }
    }

    pub fn snapshot(&self) -> CapabilityPackLedgerSnapshot {
        CapabilityPackLedgerSnapshot {
            trusted_publishers: self.trusted_publishers.clone(),
            installed_packs: self.installed_packs.clone(),
            revoked_packs: self.revoked_packs.clone(),
            version: 1,
        }
    }

    pub fn add_publisher(
        &mut self,
        pubkey_hex: &str,
        label: &str,
        notes: Option<String>,
        now_ms: u64,
    ) -> Result<TrustedPublisher, CapabilityPackError> {
        let pk = pubkey_hex.trim().to_ascii_lowercase();
        if pk.len() != 64 || !pk.chars().all(|c| c.is_ascii_hexdigit()) {
            return Err(CapabilityPackError::PublisherKey(format!(
                "pubkey must be 64 hex chars, got {} ({:?})",
                pk.len(), pk
            )));
        }
        let entry = TrustedPublisher {
            pubkey_hex: pk.clone(),
            label: label.to_string(),
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

    pub fn remove_install(&mut self, pack_id: &str) -> Option<InstalledPack> {
        self.installed_packs.remove(pack_id)
    }

    pub fn revoke(&mut self, pack_id: &str) {
        self.revoked_packs.insert(pack_id.to_string());
        self.installed_packs.remove(pack_id);
    }

    pub fn is_revoked(&self, pack_id: &str) -> bool {
        self.revoked_packs.contains(pack_id)
    }
}

// ---------------------------------------------------------------------
// Signing / verification
// ---------------------------------------------------------------------

pub fn hex_encode(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

pub fn hex_decode<const N: usize>(s: &str) -> Result<[u8; N], CapabilityPackError> {
    let s = s.trim();
    if s.len() != N * 2 {
        return Err(CapabilityPackError::Format(format!(
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

fn hex_nibble(b: u8) -> Result<u8, CapabilityPackError> {
    match b {
        b'0'..=b'9' => Ok(b - b'0'),
        b'a'..=b'f' => Ok(b - b'a' + 10),
        b'A'..=b'F' => Ok(b - b'A' + 10),
        _ => Err(CapabilityPackError::Format(format!("non-hex byte: {}", b))),
    }
}

pub fn generate_keypair() -> ([u8; 32], String) {
    use ed25519_dalek::SigningKey;
    use rand_core::OsRng;
    let sk = SigningKey::generate(&mut OsRng);
    let pk_hex = hex_encode(sk.verifying_key().as_bytes());
    (sk.to_bytes(), pk_hex)
}

pub fn signing_key_from_bytes(bytes: &[u8; 32]) -> ed25519_dalek::SigningKey {
    ed25519_dalek::SigningKey::from_bytes(bytes)
}

pub fn verifying_key_from_hex(
    pubkey_hex: &str,
) -> Result<ed25519_dalek::VerifyingKey, CapabilityPackError> {
    let bytes: [u8; 32] = hex_decode(pubkey_hex)?;
    ed25519_dalek::VerifyingKey::from_bytes(&bytes).map_err(|e| {
        CapabilityPackError::PublisherKey(format!("verifying key: {}", e))
    })
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(bytes);
    hex_encode(&h.finalize())
}

// ---------------------------------------------------------------------
// Producer
// ---------------------------------------------------------------------

pub struct CapabilityPackBuildSpec<'a> {
    pub manifest_seed: CapabilityPackManifestSeed,
    pub content: Vec<u8>,
    pub signing_key: &'a ed25519_dalek::SigningKey,
    pub now_ms: u64,
}

#[derive(Debug, Clone)]
pub struct CapabilityPackManifestSeed {
    pub pack_id: String,
    pub pack_version: String,
    pub runtime_kind: PackRuntimeKind,
    pub tools: Vec<PackToolDecl>,
    pub permissions: PackPermissions,
    pub description: String,
}

impl Default for CapabilityPackManifestSeed {
    fn default() -> Self {
        Self {
            pack_id: "capability.unnamed".into(),
            pack_version: "0.0.1".into(),
            runtime_kind: PackRuntimeKind::Wasm,
            tools: Vec::new(),
            permissions: PackPermissions::default(),
            description: String::new(),
        }
    }
}

pub fn build_capability_pack(
    spec: CapabilityPackBuildSpec<'_>,
) -> Result<CapabilityPack, CapabilityPackError> {
    use ed25519_dalek::Signer;
    let publisher_pubkey_hex =
        hex_encode(spec.signing_key.verifying_key().as_bytes());
    let manifest = CapabilityPackManifest {
        pack_id: spec.manifest_seed.pack_id,
        pack_version: spec.manifest_seed.pack_version,
        publisher_pubkey_hex,
        signed_at_ms: spec.now_ms,
        runtime_kind: spec.manifest_seed.runtime_kind,
        tools: spec.manifest_seed.tools,
        permissions: spec.manifest_seed.permissions,
        description: spec.manifest_seed.description,
    };
    let manifest_bytes = bincode::serialize(&manifest)
        .map_err(|e| CapabilityPackError::Format(format!("manifest serialize: {}", e)))?;
    let payload = CapabilityPack::signing_payload(&manifest_bytes, &spec.content);
    let sig = spec.signing_key.sign(&payload);
    Ok(CapabilityPack {
        manifest,
        content: spec.content,
        signature: sig.to_bytes(),
    })
}

pub fn write_capability_pack_file<P: AsRef<Path>>(
    pack: &CapabilityPack,
    path: P,
) -> Result<(), CapabilityPackError> {
    let path = path.as_ref();
    let bytes = pack.to_bytes()?;
    let tmp = path.with_extension("lupack5.tmp");
    std::fs::write(&tmp, &bytes)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

pub fn read_capability_pack_file<P: AsRef<Path>>(
    path: P,
) -> Result<(CapabilityPack, Vec<u8>), CapabilityPackError> {
    let bytes = std::fs::read(path.as_ref())?;
    let pack = CapabilityPack::decode_unverified(&bytes)?;
    Ok((pack, bytes))
}

pub fn verify_capability_pack(
    pack: &CapabilityPack,
) -> Result<(), CapabilityPackError> {
    let manifest_bytes = bincode::serialize(&pack.manifest)
        .map_err(|e| CapabilityPackError::Format(format!("manifest serialize: {}", e)))?;
    let payload = CapabilityPack::signing_payload(&manifest_bytes, &pack.content);
    let verifier = verifying_key_from_hex(&pack.manifest.publisher_pubkey_hex)?;
    let sig = ed25519_dalek::Signature::from_bytes(&pack.signature);
    use ed25519_dalek::Verifier;
    verifier
        .verify(&payload, &sig)
        .map_err(|_| CapabilityPackError::SignatureInvalid)
}

#[derive(Debug, Clone)]
pub struct CapabilityPackInstallResult {
    pub pack_id: String,
    pub pack_version: String,
    pub publisher_pubkey_hex: String,
    pub runtime_kind: PackRuntimeKind,
    pub manifest_hash_hex: String,
    pub tool_count: u32,
    pub already_installed_overwrite: bool,
}

/// Verify a `CapabilityPack` against a ledger and record the install.
/// **DOES NOT EXECUTE the pack content** — that's Cat.5.1's job.
pub fn install_capability_pack(
    ledger: &mut CapabilityPackLedgerStore,
    pack: &CapabilityPack,
    pack_bytes: &[u8],
    source_path: &str,
    force: bool,
    now_ms: u64,
) -> Result<CapabilityPackInstallResult, CapabilityPackError> {
    if !ledger.is_trusted(&pack.manifest.publisher_pubkey_hex) {
        return Err(CapabilityPackError::UnknownPublisher {
            publisher_pubkey_hex: pack.manifest.publisher_pubkey_hex.clone(),
        });
    }
    if ledger.is_revoked(&pack.manifest.pack_id) {
        return Err(CapabilityPackError::PackRevoked {
            pack_id: pack.manifest.pack_id.clone(),
        });
    }
    verify_capability_pack(pack)?;
    let already_installed = ledger
        .installed_packs
        .contains_key(&pack.manifest.pack_id);
    if already_installed && !force {
        return Err(CapabilityPackError::PackAlreadyInstalled {
            pack_id: pack.manifest.pack_id.clone(),
        });
    }
    let mut m_buf = [0u8; 4];
    m_buf.copy_from_slice(&pack_bytes[12..16]);
    let manifest_len = u32::from_le_bytes(m_buf) as usize;
    let manifest_start = PACK5_HEADER_LEN;
    let manifest_end = manifest_start + manifest_len;
    let manifest_hash_hex = sha256_hex(&pack_bytes[manifest_start..manifest_end]);

    let install = InstalledPack {
        pack_id: pack.manifest.pack_id.clone(),
        pack_version: pack.manifest.pack_version.clone(),
        publisher_pubkey_hex: pack.manifest.publisher_pubkey_hex.clone(),
        runtime_kind: pack.manifest.runtime_kind,
        installed_at_ms: now_ms,
        signed_at_ms: pack.manifest.signed_at_ms,
        manifest_hash_hex: manifest_hash_hex.clone(),
        tool_count: pack.manifest.tools.len() as u32,
        source_path: source_path.to_string(),
    };
    ledger.record_install(install);
    Ok(CapabilityPackInstallResult {
        pack_id: pack.manifest.pack_id.clone(),
        pack_version: pack.manifest.pack_version.clone(),
        publisher_pubkey_hex: pack.manifest.publisher_pubkey_hex.clone(),
        runtime_kind: pack.manifest.runtime_kind,
        manifest_hash_hex,
        tool_count: pack.manifest.tools.len() as u32,
        already_installed_overwrite: already_installed,
    })
}

// ---------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------

#[cfg(test)]
mod tests;
