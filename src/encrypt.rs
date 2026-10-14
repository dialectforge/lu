//! Storage v2 Phase 4 — DF XOR encryption-at-rest for the v3 brain.
//!
//! Phase 1 reserved a `flags` bit for encryption (bit 1, see
//! `FORMAT_FLAG_ENCRYPTED`); this module is what flips the bit and
//! XOR-encrypts every per-block compressed payload after Phase 3's
//! zstd compress step. The on-disk brain is mathematically meaningless
//! without the master key — the file header, offset table, block
//! index, and block-frame headers stay readable so the loader can
//! locate sections without decrypting, but every byte of compressed
//! block payload is XORed against a per-block keystream derived from
//! the master key.
//!
//! ## Threat model
//!
//! At-rest encryption only. The keyfile lives next to the brain on
//! the same machine (default `/var/lib/lu/master.key`, mode 0640,
//! `lu:lu`); a process with read access to the keyfile already has
//! access to the brain itself, so this is not defense against a
//! compromised host. It IS defense against:
//!
//!   - cold backups / disk images leaking the brain's contents
//!   - lost laptops with the brain copied off but the keyfile not
//!   - someone scraping `/var/lib/lu/brain.lbr` without the keyfile
//!
//! The DF XOR construction is a stream cipher: each block's keystream
//! is `SHA-256(master_key || block_index_le8 || chunk_index_le4)`,
//! tiled to cover the block. Block index = file-relative compressed
//! block id (the same id Phase 3's writer assigns); chunk index runs
//! 0..ceil(block_len/32). Because the master key never leaves the
//! keyfile, an observer with only the encrypted file cannot recover
//! the keystream — there is no Diffie-Hellman handshake here, unlike
//! `df_tunnel.c`'s network protocol where a passive observer can
//! recover the seed.
//!
//! ## Recovery
//!
//! The 32-byte master key is encoded as a 24-word BIP39 mnemonic for
//! human transcription (the standard 256-bit-entropy + 8-bit-checksum
//! shape). The phrase is shown ONCE at `--encrypt-init` time and is
//! the user's responsibility to record off-server. Re-deriving the
//! keyfile from a recovery phrase is implemented as `--from-recovery-
//! phrase`.

use std::fs;
use std::path::Path;

use bip39::Mnemonic;
use rand::RngCore;
use sha2::{Digest, Sha256};

use crate::brain::BrainError;

/// `header.flags` bit 1 set → block payloads in the compressed network
/// section are XOR-encrypted with the master key. Bit 0 (compression)
/// MUST also be set when this bit is on; encryption-without-compression
/// is not a supported on-disk shape (the encryption layer rides on top
/// of the per-block frames the compressor lays down).
pub const FORMAT_FLAG_ENCRYPTED: u32 = 0x0000_0002;

/// Master key length in bytes. 256 bits / 32 bytes — the standard
/// AES-256-equivalent strength target. Also matches BIP39's 24-word
/// (256-bit-entropy) shape.
pub const KEY_SIZE: usize = 32;

/// Default on-disk path for the master key. Mode 0640 owned `lu:lu`;
/// `dialectforge` is in the `lu` group so John can read it for
/// migrations and `lu_4_mcp` (running as `lu`) can read it at load
/// time. Same threat boundary as `/var/lib/lu/brain.lbr` itself.
pub const DEFAULT_KEYFILE_PATH: &str = "/var/lib/lu/master.key";

/// Expected number of words in a Lu recovery phrase. 24 words encode
/// 256 bits of entropy + an 8-bit checksum — the largest BIP39 shape.
pub const RECOVERY_PHRASE_WORDS: usize = 24;

// ---------------------------------------------------------------------
// MasterKey newtype
// ---------------------------------------------------------------------

/// 32-byte symmetric encryption key. Wrapper around `[u8; KEY_SIZE]`
/// so the type system tells encryption-key bytes apart from arbitrary
/// `[u8; 32]` (e.g. seeds, hashes, etc.).
///
/// `Clone` is intentionally derived — the key is read into RAM once
/// at load time and copied into `Brain`/`NetworkArchive` so the same
/// key can be reused for save-back. Production-grade zeroize-on-drop
/// would require an extra crate; out of scope for Phase 4 since the
/// process address space is already a trusted boundary at our threat
/// model.
#[derive(Clone)]
pub struct MasterKey {
    bytes: [u8; KEY_SIZE],
}

impl MasterKey {
    /// Wrap a raw 32-byte array as a master key. Used by tests and
    /// by `from_recovery_phrase`.
    pub fn from_bytes(bytes: [u8; KEY_SIZE]) -> Self {
        Self { bytes }
    }

    /// Borrow the raw bytes (for keystream derivation).
    pub fn as_bytes(&self) -> &[u8; KEY_SIZE] {
        &self.bytes
    }

    /// Generate a fresh master key from the OS RNG. Used by
    /// `--encrypt-init`; ALL test paths must use this AND immediately
    /// throw the key away after the test.
    pub fn generate() -> Self {
        let mut bytes = [0u8; KEY_SIZE];
        rand::thread_rng().fill_bytes(&mut bytes);
        Self { bytes }
    }

    /// Read a master key from a 32-byte file. Errors if the file is
    /// missing or any size other than `KEY_SIZE` — we do not silently
    /// truncate or pad; a wrong size means the keyfile is from a
    /// different cipher and refusing to load is the safer answer.
    pub fn load_from_keyfile<P: AsRef<Path>>(path: P) -> Result<Self, BrainError> {
        let p = path.as_ref();
        let raw = fs::read(p).map_err(|e| {
            BrainError::Format(format!(
                "keyfile {} cannot be read: {}",
                p.display(),
                e
            ))
        })?;
        if raw.len() != KEY_SIZE {
            return Err(BrainError::Format(format!(
                "keyfile {} has wrong size: got {} bytes, expected {}",
                p.display(),
                raw.len(),
                KEY_SIZE
            )));
        }
        let mut bytes = [0u8; KEY_SIZE];
        bytes.copy_from_slice(&raw);
        Ok(Self { bytes })
    }

    /// Atomically write the key to `path` with mode 0640. Refuses to
    /// overwrite an existing keyfile — that would silently invalidate
    /// any encrypted brain encrypted under the old key.
    pub fn save_to_keyfile<P: AsRef<Path>>(&self, path: P) -> Result<(), BrainError> {
        let p = path.as_ref();
        if p.exists() {
            return Err(BrainError::Format(format!(
                "keyfile {} already exists — refusing to overwrite",
                p.display()
            )));
        }
        // Write to a sibling temp path then rename, so a crash
        // mid-write never leaves a partial keyfile.
        let parent = p.parent().ok_or_else(|| {
            BrainError::Format(format!("keyfile path has no parent: {}", p.display()))
        })?;
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).map_err(|e| {
                BrainError::Format(format!(
                    "create_dir_all({}) failed: {}",
                    parent.display(),
                    e
                ))
            })?;
        }
        let tmp = p.with_extension("key.tmp");
        fs::write(&tmp, &self.bytes).map_err(|e| {
            BrainError::Format(format!("write {} failed: {}", tmp.display(), e))
        })?;
        // Mode 0640 — owner rw, group r, world none. Best-effort: on
        // platforms without unix perms (Windows) this is a no-op.
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let perms = fs::Permissions::from_mode(0o640);
            fs::set_permissions(&tmp, perms).map_err(|e| {
                BrainError::Format(format!(
                    "chmod 0640 {} failed: {}",
                    tmp.display(),
                    e
                ))
            })?;
        }
        fs::rename(&tmp, p).map_err(|e| {
            let _ = fs::remove_file(&tmp);
            BrainError::Format(format!(
                "rename {} -> {} failed: {}",
                tmp.display(),
                p.display(),
                e
            ))
        })?;
        Ok(())
    }

    /// Encode the master key as a 24-word BIP39 phrase (English). The
    /// phrase is the human-transcribable backup; pasting it back into
    /// `--from-recovery-phrase` regenerates the exact same key bytes.
    pub fn to_recovery_phrase(&self) -> Result<String, BrainError> {
        let m = Mnemonic::from_entropy(&self.bytes).map_err(|e| {
            BrainError::Format(format!("BIP39 from_entropy failed: {}", e))
        })?;
        Ok(m.to_string())
    }

    /// Decode a 24-word BIP39 phrase (English) back into a master
    /// key. Whitespace-tolerant; parser normalises NFKD per BIP39.
    pub fn from_recovery_phrase(phrase: &str) -> Result<Self, BrainError> {
        // Collapse whitespace + lowercase so a phrase pasted from a
        // notebook with trailing spaces or stray newlines parses.
        let normalized = phrase
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
            .to_lowercase();
        let words = normalized.split_whitespace().count();
        if words != RECOVERY_PHRASE_WORDS {
            return Err(BrainError::Format(format!(
                "recovery phrase: expected {} words, got {}",
                RECOVERY_PHRASE_WORDS, words
            )));
        }
        let m = Mnemonic::parse(&normalized).map_err(|e| {
            BrainError::Format(format!("BIP39 parse failed: {}", e))
        })?;
        let entropy: Vec<u8> = m.to_entropy();
        if entropy.len() != KEY_SIZE {
            return Err(BrainError::Format(format!(
                "recovery phrase decoded to {} bytes, expected {}",
                entropy.len(),
                KEY_SIZE
            )));
        }
        let mut bytes = [0u8; KEY_SIZE];
        bytes.copy_from_slice(&entropy);
        Ok(Self { bytes })
    }
}

impl std::fmt::Debug for MasterKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // NEVER print the bytes — debug logs leak. The 4-byte fingerprint
        // (first 4 hex chars of SHA256(key)) is enough to tell two keys
        // apart in a log without leaking material.
        let mut h = Sha256::new();
        h.update(&self.bytes);
        let fp = h.finalize();
        write!(
            f,
            "MasterKey(<32 bytes, fingerprint={:02x}{:02x}{:02x}{:02x}>)",
            fp[0], fp[1], fp[2], fp[3]
        )
    }
}

// ---------------------------------------------------------------------
// Keystream + XOR helpers
// ---------------------------------------------------------------------

/// Compute the 32-byte keystream chunk at
/// `(block_index, chunk_index)`. Public so the writer and reader
/// share one source of truth; not generally useful to outside callers.
pub fn keystream_chunk(
    key: &MasterKey,
    block_index: u64,
    chunk_index: u32,
) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(key.as_bytes());
    h.update(block_index.to_le_bytes());
    h.update(chunk_index.to_le_bytes());
    let out = h.finalize();
    let mut buf = [0u8; 32];
    buf.copy_from_slice(&out);
    buf
}

/// XOR `data` in place against the keystream for `block_index`. Used
/// for both encryption AND decryption — XOR is its own inverse.
pub fn xor_in_place(key: &MasterKey, block_index: u64, data: &mut [u8]) {
    let mut chunk_index: u32 = 0;
    let mut offset: usize = 0;
    while offset < data.len() {
        let ks = keystream_chunk(key, block_index, chunk_index);
        let take = std::cmp::min(32, data.len() - offset);
        for i in 0..take {
            data[offset + i] ^= ks[i];
        }
        offset += take;
        chunk_index = chunk_index.wrapping_add(1);
    }
}

/// Convenience: encrypt `data` against `block_index`, returning a
/// fresh Vec. Uses `xor_in_place` internally.
pub fn encrypt_block(key: &MasterKey, block_index: u64, data: &[u8]) -> Vec<u8> {
    let mut out = data.to_vec();
    xor_in_place(key, block_index, &mut out);
    out
}

/// Convenience: decrypt `data` against `block_index`, returning a
/// fresh Vec. Identical operation to `encrypt_block`; named
/// distinctly so call sites read clearly.
pub fn decrypt_block(key: &MasterKey, block_index: u64, data: &[u8]) -> Vec<u8> {
    let mut out = data.to_vec();
    xor_in_place(key, block_index, &mut out);
    out
}

// ---------------------------------------------------------------------
// Path helpers
// ---------------------------------------------------------------------

/// True if `flags` indicates the file's network section is encrypted.
pub fn is_encrypted(flags: u32) -> bool {
    flags & FORMAT_FLAG_ENCRYPTED != 0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn k(byte: u8) -> MasterKey {
        MasterKey::from_bytes([byte; KEY_SIZE])
    }

    #[test]
    fn xor_round_trip_recovers_input() {
        let key = k(0xa5);
        let plain: Vec<u8> = (0u8..=255u8).cycle().take(8000).collect();
        let mut buf = plain.clone();
        xor_in_place(&key, 7, &mut buf);
        assert_ne!(buf, plain, "encryption must change bytes");
        xor_in_place(&key, 7, &mut buf);
        assert_eq!(buf, plain, "decrypt must recover plaintext");
    }

    #[test]
    fn xor_with_wrong_key_does_not_recover() {
        let key = k(0x11);
        let wrong = k(0x22);
        let plain = vec![42u8; 4096];
        let cipher = encrypt_block(&key, 0, &plain);
        let bad = decrypt_block(&wrong, 0, &cipher);
        assert_ne!(bad, plain);
    }

    #[test]
    fn xor_with_wrong_block_index_does_not_recover() {
        let key = k(0x11);
        let plain = vec![42u8; 4096];
        let cipher = encrypt_block(&key, 0, &plain);
        let bad = decrypt_block(&key, 1, &cipher);
        assert_ne!(bad, plain);
    }

    #[test]
    fn xor_handles_short_and_unaligned_lengths() {
        let key = k(0x33);
        for len in [0usize, 1, 31, 32, 33, 63, 64, 65, 127, 4097] {
            let plain: Vec<u8> = (0..len as u32)
                .map(|i| (i & 0xff) as u8)
                .collect();
            let mut buf = plain.clone();
            xor_in_place(&key, 13, &mut buf);
            xor_in_place(&key, 13, &mut buf);
            assert_eq!(buf, plain, "len={}", len);
        }
    }

    #[test]
    fn keystream_chunk_is_deterministic_and_block_indexed() {
        let key = k(0x77);
        let a0 = keystream_chunk(&key, 0, 0);
        let a1 = keystream_chunk(&key, 0, 0);
        assert_eq!(a0, a1);
        let b = keystream_chunk(&key, 1, 0);
        let c = keystream_chunk(&key, 0, 1);
        assert_ne!(a0, b);
        assert_ne!(a0, c);
        assert_ne!(b, c);
    }

    #[test]
    fn bip39_round_trip_recovers_key() {
        let key = MasterKey::generate();
        let phrase = key.to_recovery_phrase().unwrap();
        let words: Vec<&str> = phrase.split_whitespace().collect();
        assert_eq!(words.len(), 24, "24-word phrase expected");
        let recovered = MasterKey::from_recovery_phrase(&phrase).unwrap();
        assert_eq!(recovered.as_bytes(), key.as_bytes());
    }

    #[test]
    fn bip39_normalises_whitespace_and_case() {
        let key = MasterKey::from_bytes([0x42; KEY_SIZE]);
        let phrase = key.to_recovery_phrase().unwrap();
        // Mangle: extra spaces, trailing newline, mixed case.
        let mangled = format!("  {}\n", phrase.to_uppercase().replace(' ', "   "));
        let recovered = MasterKey::from_recovery_phrase(&mangled).unwrap();
        assert_eq!(recovered.as_bytes(), key.as_bytes());
    }

    #[test]
    fn bip39_rejects_wrong_length() {
        let err = MasterKey::from_recovery_phrase("abandon abandon abandon")
            .err()
            .expect("too few words must error");
        let msg = format!("{:?}", err);
        assert!(msg.to_lowercase().contains("24 words") || msg.to_lowercase().contains("got 3"),
                "expected length error, got {}", msg);
    }

    #[test]
    fn bip39_rejects_invalid_checksum() {
        // 24 valid wordlist words but the checksum won't match.
        let bogus = vec!["abandon"; 24].join(" ");
        let r = MasterKey::from_recovery_phrase(&bogus);
        assert!(r.is_err(), "all-abandon phrase must fail BIP39 checksum");
    }

    #[test]
    fn keyfile_round_trip() {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "lu4d-keyfile-test-{}-{}.bin",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        let _ = std::fs::remove_file(&p);
        let key = MasterKey::generate();
        key.save_to_keyfile(&p).expect("save keyfile");
        let loaded = MasterKey::load_from_keyfile(&p).expect("load keyfile");
        assert_eq!(loaded.as_bytes(), key.as_bytes());
        // refuse-overwrite
        let again = key.save_to_keyfile(&p);
        assert!(again.is_err());
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn keyfile_wrong_size_errors() {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "lu4d-keyfile-bad-{}-{}.bin",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::write(&p, b"too short").unwrap();
        let r = MasterKey::load_from_keyfile(&p);
        assert!(r.is_err());
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn debug_does_not_leak_key_bytes() {
        let key = MasterKey::from_bytes([0xff; KEY_SIZE]);
        let s = format!("{:?}", key);
        // The literal hex "ff" of the bytes must not appear consecutively
        // for the full 64 chars; a 4-byte fingerprint of the SHA256
        // could coincidentally have ff prefixes, so we only assert the
        // absence of the obvious 32-byte run.
        assert!(!s.contains("ffffffffffffffffffffffffffffffff"));
    }
}
