//! Resident degree-map sidecar -- OP-BRAIN-MapRAM persist.
//!
//! `Brain::load_v5` rebuilds the resident degree MAP
//! ([`crate::map_ram::ResidentMap`]) on EVERY boot by iterating every
//! bond across all v5 bond SSTs into a transient HashMap
//! (`Brain::build_resident_map`). The map is an O(nodes) index but the
//! rebuild is O(bonds): measured at 2m37s for a 20M-node brain and
//! 11m41s at 28M nodes. It is not persisted -- so this module persists
//! it.
//!
//! The sidecar serialises the two `Vec<u32>` degree arrays
//! (`out_degree`, `in_degree`) alongside a 64-bit **validity
//! fingerprint** of the exact graph state the degrees were counted
//! from. On boot `load_v5` recomputes the fingerprint from the live
//! manifest + intern state; on a match it loads the degrees verbatim
//! (the fast path, a couple of seconds); on a mismatch / missing /
//! corrupt file it falls back to `build_resident_map` (unchanged
//! behaviour) and writes a fresh sidecar so the NEXT boot is fast.
//!
//! ## Validity fingerprint
//!
//! `build_resident_map` reads exactly two inputs:
//!   1. The v5 bond SSTs named in `manifest.bond_v5_levels` (it iterates
//!      every one of them; it does NOT read the MemTable / WAL bonds).
//!   2. The node intern table -- via the seed-existence filter
//!      (`reverse_lookup`) that drops any bond whose subject or object
//!      is not interned, and via `intern.len()` which sizes the dense
//!      degree vectors.
//!
//! The fingerprint therefore folds:
//!   * `master_key_id` (the brain's key fingerprint -- a sidecar written
//!     under a different key must never be trusted),
//!   * `format_tag`,
//!   * `intern_len` (the post-WAL-replay interned node count -- this is
//!     what makes a sidecar written BEFORE extra WAL bonds that interned
//!     new nodes show up as stale; it also pins the dense-vector length),
//!   * for every live BondSSTEntry, the tuple
//!     `(level, file_uid, entry_count, epoch)` -- the SST set + their
//!     monotonic epochs. `file_uid` and `epoch` advance on every flush /
//!     compaction, so any change to the bond-SST set changes the
//!     fingerprint. The tuples are folded in SORTED order so manifest
//!     iteration order is irrelevant.
//!
//! Any change that would alter `build_resident_map`'s output changes at
//! least one of these inputs, so a stale sidecar is always rejected.
//! WAL/MemTable bonds that reference already-interned nodes change
//! neither `intern_len` nor the bond SSTs -- and they also do not change
//! `build_resident_map`'s output (it ignores the MemTable). So the
//! fingerprint is exact, not conservative, for that case.
//!
//! ## On-disk envelope
//!
//! Mirrors `node_store::sidecar` (the `nodeintern.bin` sidecar):
//! v2-style 32-byte header (magic + version + flags + reserved +
//! record-region length + 16-byte IV), an XOR-keystream-encrypted body
//! when a key is supplied (the box runs encrypted-at-rest -- an
//! unencrypted derived-from-content sidecar would violate the
//! guarantee), and a trailing CRC32 over header || PLAINTEXT body
//! (computed pre-encryption, re-checked post-decryption so a wrong key
//! fails cleanly with `CrcMismatch` instead of parsing junk). Atomicity
//! is the caller's job: write `<path>.tmp`, fsync, rename.

use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::time::Instant;

use rand::RngCore;

use crate::encrypt::{xor_in_place, MasterKey};
use crate::manifest::Manifest;

/// Magic bytes at the start of every degmap sidecar ("DGMP").
/// Little-endian u32.
pub const DEGMAP_MAGIC: u32 = 0x504D_4744;

/// Wire-format version. v1 = encrypted-capable header (mirrors the
/// nodeintern sidecar's v2 layout; we start at 1 for a new file type).
pub const DEGMAP_FORMAT_VERSION: u16 = 1;

/// Header flags bit 0: the body region is XOR-keystream encrypted.
pub const DEGMAP_FLAG_ENCRYPTED: u8 = 0x01;

/// Header size: 4 magic + 2 ver + 1 flags + 1 reserved + 8 body_len + 16 iv.
const HEADER_BYTES: usize = 32;

/// Trailer size: 4-byte CRC32 over header || PLAINTEXT body.
const TRAILER_BYTES: usize = 4;

/// Body preamble (inside the encrypted region): 8 fingerprint + 8 node_count.
const BODY_PREAMBLE_BYTES: usize = 16;

/// BufReader / BufWriter capacity.
const BUF_CAPACITY: usize = 64 * 1024;

/// Hard upper bound on the declared node count (1 billion). Guards a
/// corrupted header from triggering a multi-GB allocation before the
/// CRC is ever checked.
const MAX_NODE_COUNT: u64 = 1_000_000_000;

/// Keystream index for the degmap body, derived from the per-file IV.
/// Mirrors `node_store::sidecar::derive_sidecar_keystream_index` (its
/// helper is private, so we keep a parallel one with our own domain
/// constant -- the random IV is what makes the keystream unique per
/// file; the domain constant only separates this file type's keystream
/// space from the nodeintern sidecar's).
fn derive_degmap_keystream_index(iv: &[u8; 16]) -> u64 {
    const DEGMAP_KS_DOMAIN: u64 = 0x4445_474D_4150_3031; // "DEGMAP01"
    let mut iv_mix: u64 = 0;
    for chunk in iv.chunks(8) {
        let mut tmp = [0u8; 8];
        tmp[..chunk.len()].copy_from_slice(chunk);
        iv_mix ^= u64::from_le_bytes(tmp);
    }
    DEGMAP_KS_DOMAIN.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ iv_mix
}

/// Errors surfaced by the degmap sidecar layer. A `load_degmap` error
/// is ALWAYS recoverable: the caller treats it identically to "file
/// absent" and falls back to `Brain::build_resident_map`.
#[derive(Debug)]
pub enum DegMapError {
    Io(std::io::Error),
    MagicMismatch { observed: u32 },
    FormatVersionUnsupported { observed: u16 },
    CrcMismatch { expected: u32, observed: u32 },
    NodeCountImplausible { observed: u64, max: u64 },
    UnexpectedEof,
    /// Header marks the body encrypted but no key was supplied.
    EncryptedButNoKey,
}

impl std::fmt::Display for DegMapError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DegMapError::Io(e) => write!(f, "degmap IO: {}", e),
            DegMapError::MagicMismatch { observed } => write!(
                f,
                "degmap magic mismatch: expected {:#x}, observed {:#x}",
                DEGMAP_MAGIC, observed
            ),
            DegMapError::FormatVersionUnsupported { observed } => write!(
                f,
                "degmap format version unsupported: expected {}, observed {}",
                DEGMAP_FORMAT_VERSION, observed
            ),
            DegMapError::CrcMismatch { expected, observed } => write!(
                f,
                "degmap CRC32 mismatch: expected {:#010x}, observed {:#010x}",
                expected, observed
            ),
            DegMapError::NodeCountImplausible { observed, max } => write!(
                f,
                "degmap node_count {} exceeds upper bound {}",
                observed, max
            ),
            DegMapError::UnexpectedEof => write!(f, "degmap truncated / unexpected EOF"),
            DegMapError::EncryptedButNoKey => {
                write!(f, "degmap body is encrypted but no master key supplied")
            }
        }
    }
}

impl std::error::Error for DegMapError {}

impl From<std::io::Error> for DegMapError {
    fn from(e: std::io::Error) -> Self {
        DegMapError::Io(e)
    }
}

/// FNV-1a 64-bit accumulator -- a stable, dependency-free fold for the
/// validity fingerprint. (std `DefaultHasher` is NOT used: its output
/// is not guaranteed stable across Rust releases, and this fingerprint
/// is compared across separate process invocations / builds.)
#[derive(Clone, Copy)]
struct Fnv1a(u64);

impl Fnv1a {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01B3;

    fn new() -> Self {
        Fnv1a(Self::OFFSET)
    }

    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 ^= b as u64;
            self.0 = self.0.wrapping_mul(Self::PRIME);
        }
    }

    fn write_u64(&mut self, v: u64) {
        self.write(&v.to_le_bytes());
    }

    fn finish(self) -> u64 {
        self.0
    }
}

/// Compute the validity fingerprint over the exact inputs
/// `Brain::build_resident_map` reads: the manifest's bond-SST set + the
/// key/format identity + the interned node count.
///
/// `intern_len` is the post-WAL-replay `NodeIntern::len()` -- the dense
/// degree-vector length AND the generation tag for WAL-added intern
/// bindings (see the module docs).
pub fn compute_fingerprint(manifest: &Manifest, intern_len: u64) -> u64 {
    let mut h = Fnv1a::new();
    // A version byte so a future change to the fingerprint inputs forces
    // every existing sidecar to be treated as stale (rebuild + rewrite).
    h.write(b"DEGMAP-FP-v1");
    h.write(&manifest.master_key_id.0);
    h.write(&[manifest.format_tag]);
    h.write_u64(intern_len);

    // Per-SST tuples, folded in a canonical SORTED order so manifest
    // iteration order can never change the fingerprint.
    let mut tuples: Vec<(u8, u64, u64, u64)> = manifest
        .iter_bond_v5_ssts()
        .map(|(level, e)| (level, e.file_uid, e.entry_count, e.epoch))
        .collect();
    tuples.sort_unstable();
    h.write_u64(tuples.len() as u64);
    for (level, file_uid, entry_count, epoch) in tuples {
        h.write(&[level]);
        h.write_u64(file_uid);
        h.write_u64(entry_count);
        h.write_u64(epoch);
    }
    h.finish()
}

/// The validated contents of a degmap sidecar.
#[derive(Debug)]
pub struct DegMapLoad {
    /// Validity fingerprint stored at write time. The caller compares
    /// this against a freshly computed `compute_fingerprint`; only on a
    /// match are the degrees trusted.
    pub fingerprint: u64,
    pub out_degree: Vec<u32>,
    pub in_degree: Vec<u32>,
}

/// Stats returned by [`write_degmap`].
#[derive(Debug, Clone, Copy)]
pub struct DegMapWriteStats {
    pub node_count: u64,
    pub file_size_bytes: u64,
    pub fingerprint: u64,
    pub elapsed: std::time::Duration,
}

/// `<path>.tmp` companion used for the atomic write.
pub fn degmap_tmp_path(path: &Path) -> std::path::PathBuf {
    let mut s = path.as_os_str().to_owned();
    s.push(".tmp");
    std::path::PathBuf::from(s)
}

/// Serialise the two degree vectors + fingerprint to `path`.
///
/// NON-atomic on its own: it `create+truncate`s `path` and `fsync_all`s
/// it. Callers wanting crash-atomic replacement write to
/// [`degmap_tmp_path`] then `rename` (see
/// `Brain::persist_resident_degmap`).
pub fn write_degmap(
    path: &Path,
    out_degree: &[u32],
    in_degree: &[u32],
    fingerprint: u64,
    key: Option<MasterKey>,
) -> Result<DegMapWriteStats, DegMapError> {
    assert_eq!(
        out_degree.len(),
        in_degree.len(),
        "write_degmap: out_degree and in_degree length mismatch",
    );
    let started = Instant::now();
    let node_count = out_degree.len() as u64;

    // ---- body = preamble (fingerprint + node_count) || out[..] || in[..]
    // All little-endian. Both arrays are length `node_count`.
    let mut body: Vec<u8> =
        Vec::with_capacity(BODY_PREAMBLE_BYTES + out_degree.len() * 4 + in_degree.len() * 4);
    body.extend_from_slice(&fingerprint.to_le_bytes());
    body.extend_from_slice(&node_count.to_le_bytes());
    for &d in out_degree {
        body.extend_from_slice(&d.to_le_bytes());
    }
    for &d in in_degree {
        body.extend_from_slice(&d.to_le_bytes());
    }

    let mut iv = [0u8; 16];
    rand::thread_rng().fill_bytes(&mut iv);

    let encrypted = key.is_some();
    let mut header = [0u8; HEADER_BYTES];
    header[0..4].copy_from_slice(&DEGMAP_MAGIC.to_le_bytes());
    header[4..6].copy_from_slice(&DEGMAP_FORMAT_VERSION.to_le_bytes());
    header[6] = if encrypted { DEGMAP_FLAG_ENCRYPTED } else { 0 };
    // byte 7 reserved = 0
    header[8..16].copy_from_slice(&(body.len() as u64).to_le_bytes());
    header[16..32].copy_from_slice(&iv);

    // CRC over header || PLAINTEXT body, computed BEFORE encryption.
    let mut hasher = crc32fast::Hasher::new();
    hasher.update(&header);
    hasher.update(&body);
    let body_crc32 = hasher.finalize();

    if let Some(k) = key {
        let ksi = derive_degmap_keystream_index(&iv);
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
    let mut f = w.into_inner().map_err(|e| DegMapError::Io(e.into_error()))?;
    f.flush()?;
    f.sync_all()?;

    Ok(DegMapWriteStats {
        node_count,
        file_size_bytes: file_size,
        fingerprint,
        elapsed: started.elapsed(),
    })
}

/// Load + validate a degmap sidecar. Returns the stored fingerprint and
/// both degree vectors. ANY error (missing, magic, version, CRC,
/// implausible count, EOF, wrong key) is recoverable: the caller treats
/// it as "rebuild from scratch".
pub fn read_degmap(path: &Path, key: Option<MasterKey>) -> Result<DegMapLoad, DegMapError> {
    let f = File::open(path)?;
    let mut r = BufReader::with_capacity(BUF_CAPACITY, f);

    let mut header = [0u8; HEADER_BYTES];
    r.read_exact(&mut header)?;
    let magic = u32::from_le_bytes([header[0], header[1], header[2], header[3]]);
    if magic != DEGMAP_MAGIC {
        return Err(DegMapError::MagicMismatch { observed: magic });
    }
    let ver = u16::from_le_bytes([header[4], header[5]]);
    if ver != DEGMAP_FORMAT_VERSION {
        return Err(DegMapError::FormatVersionUnsupported { observed: ver });
    }
    let flags = header[6];
    let body_len = u64::from_le_bytes([
        header[8], header[9], header[10], header[11], header[12], header[13], header[14],
        header[15],
    ]);
    // body_len must at least hold the preamble; the node_count inside is
    // re-validated below. Reject an obviously-too-large declared length
    // before allocating.
    let max_body =
        BODY_PREAMBLE_BYTES as u64 + MAX_NODE_COUNT.saturating_mul(8); // 4B out + 4B in per node
    if body_len > max_body {
        return Err(DegMapError::NodeCountImplausible {
            observed: body_len,
            max: max_body,
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
    if flags & DEGMAP_FLAG_ENCRYPTED != 0 {
        let k = key.ok_or(DegMapError::EncryptedButNoKey)?;
        let ksi = derive_degmap_keystream_index(&iv);
        xor_in_place(&k, ksi, &mut body);
    }
    let mut hasher = crc32fast::Hasher::new();
    hasher.update(&header);
    hasher.update(&body);
    let expected_crc = hasher.finalize();
    if observed_crc != expected_crc {
        return Err(DegMapError::CrcMismatch {
            expected: expected_crc,
            observed: observed_crc,
        });
    }

    // ---- parse the (now plaintext) body ----
    if body.len() < BODY_PREAMBLE_BYTES {
        return Err(DegMapError::UnexpectedEof);
    }
    let fingerprint = u64::from_le_bytes([
        body[0], body[1], body[2], body[3], body[4], body[5], body[6], body[7],
    ]);
    let node_count = u64::from_le_bytes([
        body[8], body[9], body[10], body[11], body[12], body[13], body[14], body[15],
    ]);
    if node_count > MAX_NODE_COUNT {
        return Err(DegMapError::NodeCountImplausible {
            observed: node_count,
            max: MAX_NODE_COUNT,
        });
    }
    let n = node_count as usize;
    let expected_body = BODY_PREAMBLE_BYTES + n * 4 + n * 4;
    if body.len() != expected_body {
        return Err(DegMapError::UnexpectedEof);
    }
    let mut out_degree = vec![0u32; n];
    let mut in_degree = vec![0u32; n];
    let mut off = BODY_PREAMBLE_BYTES;
    for slot in out_degree.iter_mut() {
        *slot = u32::from_le_bytes([body[off], body[off + 1], body[off + 2], body[off + 3]]);
        off += 4;
    }
    for slot in in_degree.iter_mut() {
        *slot = u32::from_le_bytes([body[off], body[off + 1], body[off + 2], body[off + 3]]);
        off += 4;
    }

    Ok(DegMapLoad {
        fingerprint,
        out_degree,
        in_degree,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> (tempfile::TempDir, std::path::PathBuf) {
        let d = tempfile::tempdir().unwrap();
        let p = d.path().join(name);
        (d, p)
    }

    #[test]
    fn roundtrip_plaintext() {
        let (_d, p) = tmp("dm_plain.bin");
        let out = vec![3u32, 0, 7, 1];
        let inn = vec![0u32, 5, 2, 9];
        let fp = 0xDEAD_BEEF_1234_5678u64;
        write_degmap(&p, &out, &inn, fp, None).unwrap();
        let loaded = read_degmap(&p, None).unwrap();
        assert_eq!(loaded.fingerprint, fp);
        assert_eq!(loaded.out_degree, out);
        assert_eq!(loaded.in_degree, inn);
    }

    #[test]
    fn roundtrip_encrypted() {
        let (_d, p) = tmp("dm_enc.bin");
        let key = MasterKey::from_bytes([0x5Au8; 32]);
        let out = vec![1u32, 2, 3];
        let inn = vec![9u32, 8, 7];
        let fp = 42;
        write_degmap(&p, &out, &inn, fp, Some(key.clone())).unwrap();
        let loaded = read_degmap(&p, Some(key)).unwrap();
        assert_eq!(loaded.fingerprint, fp);
        assert_eq!(loaded.out_degree, out);
        assert_eq!(loaded.in_degree, inn);
    }

    #[test]
    fn empty_roundtrip() {
        let (_d, p) = tmp("dm_empty.bin");
        let key = MasterKey::from_bytes([0x11u8; 32]);
        write_degmap(&p, &[], &[], 7, Some(key.clone())).unwrap();
        let loaded = read_degmap(&p, Some(key)).unwrap();
        assert_eq!(loaded.fingerprint, 7);
        assert!(loaded.out_degree.is_empty());
        assert!(loaded.in_degree.is_empty());
    }

    #[test]
    fn wrong_key_is_crc_mismatch_not_garbage() {
        let (_d, p) = tmp("dm_wrongkey.bin");
        let key = MasterKey::from_bytes([0x22u8; 32]);
        let wrong = MasterKey::from_bytes([0x33u8; 32]);
        write_degmap(&p, &[5, 6], &[7, 8], 99, Some(key)).unwrap();
        match read_degmap(&p, Some(wrong)) {
            Err(DegMapError::CrcMismatch { .. }) => {}
            other => panic!("expected CrcMismatch, got {:?}", other),
        }
    }

    #[test]
    fn encrypted_without_key_errs() {
        let (_d, p) = tmp("dm_nokey.bin");
        let key = MasterKey::from_bytes([0x44u8; 32]);
        write_degmap(&p, &[1], &[2], 1, Some(key)).unwrap();
        match read_degmap(&p, None) {
            Err(DegMapError::EncryptedButNoKey) => {}
            other => panic!("expected EncryptedButNoKey, got {:?}", other),
        }
    }

    #[test]
    fn corrupt_body_is_crc_mismatch() {
        let (_d, p) = tmp("dm_corrupt.bin");
        write_degmap(&p, &[1, 2, 3], &[4, 5, 6], 1, None).unwrap();
        // Flip a byte inside the body region (after the 32-byte header).
        let mut bytes = std::fs::read(&p).unwrap();
        bytes[HEADER_BYTES + BODY_PREAMBLE_BYTES] ^= 0xFF;
        std::fs::write(&p, &bytes).unwrap();
        match read_degmap(&p, None) {
            Err(DegMapError::CrcMismatch { .. }) => {}
            other => panic!("expected CrcMismatch, got {:?}", other),
        }
    }

    #[test]
    fn bad_magic_rejected() {
        let (_d, p) = tmp("dm_magic.bin");
        write_degmap(&p, &[1], &[2], 1, None).unwrap();
        let mut bytes = std::fs::read(&p).unwrap();
        bytes[0] ^= 0xFF;
        std::fs::write(&p, &bytes).unwrap();
        match read_degmap(&p, None) {
            Err(DegMapError::MagicMismatch { .. }) => {}
            other => panic!("expected MagicMismatch, got {:?}", other),
        }
    }
}
