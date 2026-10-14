//! Source/negation overlay sidecars -- OP-REVOKE-V5 persist.
//!
//! v5 brains build with an EMPTY `source_overlay` + `negation_overlay`
//! in `Brain::load_v5` and never repopulate them, so the
//! "revocable by source" guarantee (`Brain::revoke_bonds_by_source`,
//! which derives its match set from `source_overlay`) silently matched
//! ZERO after any restart: the attribution `lu4d_teach_corpus` stamps
//! into `source_overlay` lived only in RAM, and `lu4d_save` is a v5
//! no-op for the brain proper. This module persists both overlays to
//! encrypted side-tables under `wal/` so a fresh process can load the
//! attribution back.
//!
//! ## On-disk envelope
//!
//! Mirrors `node_store::degmap` / `node_store::sidecar` (the existing
//! v5 side-tables `wal/resident_degmap.bin` + `wal/nodeintern.bin`): a
//! 32-byte header (magic + version + flags + reserved + body length +
//! 16-byte IV), an XOR-keystream-encrypted body when a master key is
//! supplied (the box runs encrypted-at-rest -- a derived-from-content
//! sidecar must not be the one plaintext hole), and a trailing CRC32
//! over header || PLAINTEXT body (computed PRE-encryption, re-checked
//! POST-decryption so a wrong key fails cleanly with `CrcMismatch`
//! instead of feeding junk to bincode). The body is the bincode
//! serialisation of the overlay struct (both `SourceOverlay` and
//! `NegationOverlay` are plain serde types). Atomicity is the caller's
//! job: write `<path>.tmp`, fsync, rename (see
//! `Brain::persist_overlays_to_dir`).

use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use rand::RngCore;
use serde::de::DeserializeOwned;
use serde::Serialize;

use crate::encrypt::{xor_in_place, MasterKey};

/// Magic bytes at the start of every overlay sidecar ("OVLY").
/// Little-endian u32.
pub const OVERLAY_MAGIC: u32 = 0x594C_564F;

/// Wire-format version. v1 = encrypted-capable header (mirrors the
/// degmap sidecar's layout; we start at 1 for a new file type).
pub const OVERLAY_FORMAT_VERSION: u16 = 1;

/// Header flags bit 0: the body region is XOR-keystream encrypted.
pub const OVERLAY_FLAG_ENCRYPTED: u8 = 0x01;

/// Header size: 4 magic + 2 ver + 1 flags + 1 reserved + 8 body_len + 16 iv.
const HEADER_BYTES: usize = 32;

/// Trailer size: 4-byte CRC32 over header || PLAINTEXT body.
const TRAILER_BYTES: usize = 4;

/// BufReader / BufWriter capacity. Matches the std default.
const BUF_CAPACITY: usize = 64 * 1024;

/// Hard upper bound on the declared body length (8 GiB). Guards a
/// corrupted header from triggering a multi-GB allocation before the
/// CRC is ever checked. Comfortably exceeds any realistic overlay
/// (tens of millions of entries at ~64 bytes each).
const MAX_BODY_BYTES: u64 = 8 * 1024 * 1024 * 1024;

/// Keystream index for the overlay body, derived from the per-file IV.
/// Mirrors `node_store::degmap::derive_degmap_keystream_index` (its
/// helper is private, so we keep a parallel one with our own domain
/// constant -- the random IV is what makes the keystream unique per
/// file; the domain constant only separates this file type's keystream
/// space from the other sidecars').
fn derive_overlay_keystream_index(iv: &[u8; 16]) -> u64 {
    const OVERLAY_KS_DOMAIN: u64 = 0x4F56_4552_4C41_5931; // "OVERLAY1"
    let mut iv_mix: u64 = 0;
    for chunk in iv.chunks(8) {
        let mut tmp = [0u8; 8];
        tmp[..chunk.len()].copy_from_slice(chunk);
        iv_mix ^= u64::from_le_bytes(tmp);
    }
    OVERLAY_KS_DOMAIN.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ iv_mix
}

/// Errors surfaced by the overlay sidecar layer. A `read_overlay`
/// error is ALWAYS recoverable: the caller treats it identically to
/// "file absent" and starts from an empty overlay.
#[derive(Debug)]
pub enum OverlayError {
    Io(std::io::Error),
    MagicMismatch { observed: u32 },
    FormatVersionUnsupported { observed: u16 },
    CrcMismatch { expected: u32, observed: u32 },
    BodyLenImplausible { observed: u64, max: u64 },
    UnexpectedEof,
    /// Header marks the body encrypted but no key was supplied.
    EncryptedButNoKey,
    Bincode(bincode::Error),
}

impl std::fmt::Display for OverlayError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OverlayError::Io(e) => write!(f, "overlay IO: {}", e),
            OverlayError::MagicMismatch { observed } => write!(
                f,
                "overlay magic mismatch: expected {:#x}, observed {:#x}",
                OVERLAY_MAGIC, observed
            ),
            OverlayError::FormatVersionUnsupported { observed } => write!(
                f,
                "overlay format version unsupported: expected {}, observed {}",
                OVERLAY_FORMAT_VERSION, observed
            ),
            OverlayError::CrcMismatch { expected, observed } => write!(
                f,
                "overlay CRC32 mismatch: expected {:#010x}, observed {:#010x}",
                expected, observed
            ),
            OverlayError::BodyLenImplausible { observed, max } => write!(
                f,
                "overlay body_len {} exceeds upper bound {}",
                observed, max
            ),
            OverlayError::UnexpectedEof => write!(f, "overlay truncated / unexpected EOF"),
            OverlayError::EncryptedButNoKey => {
                write!(f, "overlay body is encrypted but no master key supplied")
            }
            OverlayError::Bincode(e) => write!(f, "overlay bincode: {}", e),
        }
    }
}

impl std::error::Error for OverlayError {}

impl From<std::io::Error> for OverlayError {
    fn from(e: std::io::Error) -> Self {
        OverlayError::Io(e)
    }
}

impl From<bincode::Error> for OverlayError {
    fn from(e: bincode::Error) -> Self {
        OverlayError::Bincode(e)
    }
}

/// `<path>.tmp` companion used for the atomic write.
pub fn overlay_tmp_path(path: &Path) -> PathBuf {
    let mut s = path.as_os_str().to_owned();
    s.push(".tmp");
    PathBuf::from(s)
}

/// Stats returned by [`write_overlay`].
#[derive(Debug, Clone, Copy)]
pub struct OverlayWriteStats {
    pub body_len: u64,
    pub file_size_bytes: u64,
    pub body_crc32: u32,
    pub elapsed: std::time::Duration,
}

/// Serialise `value` (bincode) into the encrypted-capable envelope at
/// `path`.
///
/// NON-atomic on its own: it `create+truncate`s `path` and `sync_all`s
/// it. Callers wanting crash-atomic replacement write to
/// [`overlay_tmp_path`] then `rename` (see
/// `Brain::persist_overlays_to_dir`).
pub fn write_overlay<T: Serialize>(
    path: &Path,
    value: &T,
    key: Option<MasterKey>,
) -> Result<OverlayWriteStats, OverlayError> {
    let started = Instant::now();
    let mut body: Vec<u8> = bincode::serialize(value)?;
    let body_len = body.len() as u64;

    let mut iv = [0u8; 16];
    rand::thread_rng().fill_bytes(&mut iv);

    let encrypted = key.is_some();
    let mut header = [0u8; HEADER_BYTES];
    header[0..4].copy_from_slice(&OVERLAY_MAGIC.to_le_bytes());
    header[4..6].copy_from_slice(&OVERLAY_FORMAT_VERSION.to_le_bytes());
    header[6] = if encrypted { OVERLAY_FLAG_ENCRYPTED } else { 0 };
    // byte 7 reserved = 0
    header[8..16].copy_from_slice(&body_len.to_le_bytes());
    header[16..32].copy_from_slice(&iv);

    // CRC over header || PLAINTEXT body, computed BEFORE encryption.
    // read_overlay decrypts then re-checks this CRC, so a wrong key
    // (garbage plaintext) fails cleanly with CrcMismatch instead of
    // being handed to bincode.
    let mut hasher = crc32fast::Hasher::new();
    hasher.update(&header);
    hasher.update(&body);
    let body_crc32 = hasher.finalize();

    if let Some(k) = key {
        let ksi = derive_overlay_keystream_index(&iv);
        xor_in_place(&k, ksi, &mut body);
    }

    let f = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(path)?;
    let mut w = BufWriter::with_capacity(BUF_CAPACITY, f);
    w.write_all(&header)?;
    w.write_all(&body)?;
    w.write_all(&body_crc32.to_le_bytes())?;
    let file_size = HEADER_BYTES as u64 + body.len() as u64 + TRAILER_BYTES as u64;
    let mut f = w.into_inner().map_err(|e| OverlayError::Io(e.into_error()))?;
    f.flush()?;
    f.sync_all()?;

    Ok(OverlayWriteStats {
        body_len,
        file_size_bytes: file_size,
        body_crc32,
        elapsed: started.elapsed(),
    })
}

/// Load + validate an overlay sidecar, returning the deserialised
/// value. ANY error (missing, magic, version, CRC, implausible length,
/// EOF, wrong key, bincode) is recoverable: the caller treats it as
/// "start from an empty overlay".
pub fn read_overlay<T: DeserializeOwned>(
    path: &Path,
    key: Option<MasterKey>,
) -> Result<T, OverlayError> {
    let f = File::open(path)?;
    let mut r = BufReader::with_capacity(BUF_CAPACITY, f);

    let mut header = [0u8; HEADER_BYTES];
    r.read_exact(&mut header)?;
    let magic = u32::from_le_bytes([header[0], header[1], header[2], header[3]]);
    if magic != OVERLAY_MAGIC {
        return Err(OverlayError::MagicMismatch { observed: magic });
    }
    let ver = u16::from_le_bytes([header[4], header[5]]);
    if ver != OVERLAY_FORMAT_VERSION {
        return Err(OverlayError::FormatVersionUnsupported { observed: ver });
    }
    let flags = header[6];
    let body_len = u64::from_le_bytes([
        header[8], header[9], header[10], header[11], header[12], header[13], header[14],
        header[15],
    ]);
    if body_len > MAX_BODY_BYTES {
        return Err(OverlayError::BodyLenImplausible {
            observed: body_len,
            max: MAX_BODY_BYTES,
        });
    }
    let mut iv = [0u8; 16];
    iv.copy_from_slice(&header[16..32]);

    let body_len = body_len as usize;
    let mut body = vec![0u8; body_len];
    r.read_exact(&mut body)?;
    let mut trailer = [0u8; TRAILER_BYTES];
    r.read_exact(&mut trailer)?;
    let observed_crc = u32::from_le_bytes(trailer);

    // Decrypt FIRST (if encrypted), then verify CRC over the PLAINTEXT.
    if flags & OVERLAY_FLAG_ENCRYPTED != 0 {
        let k = key.ok_or(OverlayError::EncryptedButNoKey)?;
        let ksi = derive_overlay_keystream_index(&iv);
        xor_in_place(&k, ksi, &mut body);
    }
    let mut hasher = crc32fast::Hasher::new();
    hasher.update(&header);
    hasher.update(&body);
    let expected_crc = hasher.finalize();
    if observed_crc != expected_crc {
        return Err(OverlayError::CrcMismatch {
            expected: expected_crc,
            observed: observed_crc,
        });
    }

    let value: T = bincode::deserialize(&body)?;
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::{SourceAttribution, SourceKind, SourceOverlay};
    use crate::seed::Seed;
    use crate::bond::BondType;

    fn tmp(name: &str) -> (tempfile::TempDir, std::path::PathBuf) {
        let d = tempfile::tempdir().unwrap();
        let p = d.path().join(name);
        (d, p)
    }

    fn sample_overlay() -> SourceOverlay {
        let mut ov = SourceOverlay::new();
        let s1 = Seed::from_bytes([1u8; 32]);
        let s2 = Seed::from_bytes([2u8; 32]);
        let s3 = Seed::from_bytes([3u8; 32]);
        ov.set(
            s1,
            s2,
            BondType::IsA,
            SourceAttribution::new(SourceKind::TrainingCorpus {
                corpus: "test-corpus-A".to_string(),
                run_id: Some("run-1".to_string()),
                url: None,
            }),
        );
        ov.set(
            s2,
            s3,
            BondType::PartOf,
            SourceAttribution::teach("alice", 12345),
        );
        ov
    }

    #[test]
    fn roundtrip_plaintext() {
        let (_d, p) = tmp("ov_plain.bin");
        let ov = sample_overlay();
        write_overlay(&p, &ov, None).unwrap();
        let back: SourceOverlay = read_overlay(&p, None).unwrap();
        assert_eq!(back.entries.len(), ov.entries.len());
        assert_eq!(back.entries, ov.entries);
    }

    #[test]
    fn roundtrip_encrypted() {
        let (_d, p) = tmp("ov_enc.bin");
        let key = MasterKey::from_bytes([0x5Au8; 32]);
        let ov = sample_overlay();
        write_overlay(&p, &ov, Some(key.clone())).unwrap();
        let back: SourceOverlay = read_overlay(&p, Some(key)).unwrap();
        assert_eq!(back.entries, ov.entries);
    }

    #[test]
    fn wrong_key_fails_crc_not_junk() {
        let (_d, p) = tmp("ov_wrongkey.bin");
        let key = MasterKey::from_bytes([0x11u8; 32]);
        let wrong = MasterKey::from_bytes([0x22u8; 32]);
        write_overlay(&p, &sample_overlay(), Some(key)).unwrap();
        let res: Result<SourceOverlay, _> = read_overlay(&p, Some(wrong));
        match res {
            Err(OverlayError::CrcMismatch { .. }) => {}
            other => panic!("expected CrcMismatch, got {:?}", other),
        }
    }

    #[test]
    fn encrypted_body_no_key_errors() {
        let (_d, p) = tmp("ov_nokey.bin");
        let key = MasterKey::from_bytes([0x33u8; 32]);
        write_overlay(&p, &sample_overlay(), Some(key)).unwrap();
        let res: Result<SourceOverlay, _> = read_overlay(&p, None);
        match res {
            Err(OverlayError::EncryptedButNoKey) => {}
            other => panic!("expected EncryptedButNoKey, got {:?}", other),
        }
    }

    #[test]
    fn empty_overlay_roundtrips() {
        let (_d, p) = tmp("ov_empty.bin");
        let ov = SourceOverlay::new();
        write_overlay(&p, &ov, None).unwrap();
        let back: SourceOverlay = read_overlay(&p, None).unwrap();
        assert!(back.entries.is_empty());
    }
}
