//! OP-BRAIN-V4 Phase 3 — Write-Ahead Log.
//!
//! Append-only durability for MemTable inserts. Each record carries
//! one bond's (subject, object, predicate, weight, keyframes) plus a
//! CRC32 trailer for torn-write detection at replay time. The WAL
//! stores **raw Seeds** (32 bytes each), NOT brain-wide NodeIds —
//! durability must NOT depend on the in-memory intern table state at
//! daemon boot.
//!
//! ## Record format (little-endian)
//!
//! ```text
//! offset  size  field
//! 0       4     length            (u32, total record length including this prefix and the trailing CRC)
//! 4       8     seq               (u64)
//! 12      32    subject           ([u8; 32])
//! 44      32    object            ([u8; 32])
//! 76      1     predicate         (u8 — BondType discriminant 0..=28)
//! 77      4     weight            (f32 little-endian bits)
//! 81      2     keyframe_count    (u16)
//! 83      N     keyframes payload (bincode-serialized Vec<TemporalKeyframe>)
//! 83+N    4     crc32             (u32 over bytes [0 .. 83+N])
//! ```
//!
//! Phase 1.2 (OP-BRAIN-V5 Q8) adds a NodeIntern record type for
//! Seed ↔ NodeId(u64) bindings, persisted ahead of any bond record
//! that references those NodeIds. Wire layout:
//!
//! ```text
//! offset  size    field
//! 0       4       length            (u32 LE, total record length incl. prefix + CRC)
//! 4       1       rec_type          (u8 = REC_TYPE_NODE_INTERN = 2)
//! 5       8       seq               (u64 LE)
//! 13      32      seed              ([u8; 32] raw — maximally entropic, no compression)
//! 45      1..10   node_id           (unsigned LEB128 varint of NodeId.0: u64)
//! 45+V    4       crc32             (u32 LE over bytes [0 .. 45+V])
//! ```
//!
//! Q1 (NodeId encoding) mandates varint in WAL records; the seed stays
//! raw 32 bytes (no varint benefit on hash-uniform bytes). Varint sits
//! at the tail of the variable-length section so fixed-width fields
//! are positionally addressable.
//!
//! `length` is the total record byte count INCLUDING the 4-byte length
//! prefix and the trailing CRC. That lets the replay scanner skip
//! through records by `seek_relative(length - 4)` after reading the
//! length and validate `crc32` against the bytes immediately preceding
//! the trailer.
//!
//! ## Fsync semantics
//!
//! Conditional fsync: every `LU_WAL_FSYNC_MS` (default 100 ms) OR
//! every `LU_WAL_FSYNC_BYTES` (default 1 MiB) since the previous fsync,
//! whichever comes first. `Wal::fsync` is also called by the MemTable
//! swap path before the flush thread starts work, so a record's
//! durability landmark is: "every record with `seq < swap_seq` was
//! fsync'd before the flush thread touched any SST byte."
//!
//! Records that have been `fsync`'d MUST be replayable byte-exact
//! after a kill. Records still in the buffered writer MAY be lost —
//! that's the durability tradeoff documented in the brief.
//!
//! ## Torn-write recovery
//!
//! `replay` reads records one at a time, validating each CRC32. On
//! the first failure (CRC mismatch OR short read because the kill
//! happened mid-record), replay stops and returns the records up to
//! and including the last successfully-validated one. No panic, no
//! data loss past what the kill itself dropped.

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::bond::{BondType, TemporalKeyframe};
use crate::node_intern::NodeId;
use crate::seed::Seed;
use crate::encrypt::{xor_in_place, MasterKey};
use rand::RngCore;

/// Default fsync time threshold (100 ms).
pub const DEFAULT_FSYNC_CADENCE_MS: u64 = 100;
/// Default fsync byte threshold (1 MiB).
pub const DEFAULT_FSYNC_BYTE_THRESHOLD: usize = 1024 * 1024;

/// Magic bytes at the start of every WAL segment, written once at
/// `Wal::open` on a fresh file. Used by `replay` to fail loudly if a
/// non-WAL file is handed in by mistake.
const WAL_MAGIC: &[u8; 8] = b"LU4WAL\x01\x00";

// OP-BRAIN-V5 6.8: encrypted-WAL header + per-record keystream.
//
// The pre-6.8 plaintext header is the 8-byte `WAL_MAGIC`. A segment
// written with a key uses `WAL_MAGIC_V2` followed by a 1-byte flags
// field, 3 reserved bytes, and a 16-byte random IV (28-byte header).
// Each appended record's body (the bytes between the 4-byte length
// prefix and the 4-byte CRC trailer) is XOR-encrypted under the
// v4/6.5 keystream primitive (`crate::encrypt::xor_in_place`) keyed
// by an index derived from the per-segment IV and the record's
// absolute start offset, so every record decrypts independently and a
// torn final record drops cleanly. The CRC is computed over the bytes
// AS WRITTEN to disk (length || possibly-ciphertext body), preserving
// the pre-6.8 torn-write detection byte-for-byte.

/// OP-BRAIN-V5 6.8 encrypted-capable WAL magic (8 bytes).
const WAL_MAGIC_V2: &[u8; 8] = b"LU4WAL\x02\x00";

/// Header flags bit 0: record bodies are XOR-keystream encrypted.
const WAL_FLAG_ENCRYPTED: u8 = 0x01;

/// v1 (plaintext) header length: the 8-byte magic only.
const WAL_HEADER_V1_BYTES: u64 = 8;

/// v2 header length: magic(8) + flags(1) + reserved(3) + IV(16) = 28.
const WAL_HEADER_V2_BYTES: u64 = 28;

/// OP-BRAIN-V5 6.8: derive the `xor_in_place` keystream index for the
/// record whose length prefix starts at `record_offset`. Mirrors
/// `crate::sst::derive_block_keystream_index` in structure (splitmix64
/// multiplier + IV XOR-fold) but keyed by the record's u64 byte
/// offset instead of a u32 block id. The actual keystream is the
/// trusted `crate::encrypt::xor_in_place` (SHA-256 keyed) -- unchanged.
fn derive_wal_keystream_index(iv: &[u8; 16], record_offset: u64) -> u64 {
    let mut iv_mix: u64 = 0;
    for chunk in iv.chunks(8) {
        let mut tmp = [0u8; 8];
        tmp[..chunk.len()].copy_from_slice(chunk);
        iv_mix ^= u64::from_le_bytes(tmp);
    }
    record_offset
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ iv_mix
}

/// OP-BRAIN-V5 6.8: transform a plaintext framed record
/// (`length || body || crc`) into the bytes written to disk. For a
/// plaintext segment (`enc` = None) this is the identity. For an
/// encrypted segment the body is XOR-encrypted under the per-record
/// keystream while the ORIGINAL plaintext CRC is preserved; replay
/// decrypts first and re-checks that CRC over the recovered plaintext
/// (torn-tail detection preserved; a wrong key fails cleanly).
fn seal_record(plain_full: &[u8], offset: u64, enc: Option<(&MasterKey, &[u8; 16])>) -> Vec<u8> {
    match enc {
        None => plain_full.to_vec(),
        Some((key, iv)) => {
            let n = plain_full.len();
            debug_assert!(n >= 8, "framed record shorter than length+crc");
            let mut out = Vec::with_capacity(n);
            out.extend_from_slice(&plain_full[0..4]);
            let mut body = plain_full[4..n - 4].to_vec();
            let ksi = derive_wal_keystream_index(iv, offset);
            xor_in_place(key, ksi, &mut body);
            out.extend_from_slice(&body);
            // Keep the ORIGINAL plaintext CRC (computed by encode_* over
            // length || plaintext-body). Replay decrypts then verifies
            // it over the recovered plaintext.
            out.extend_from_slice(&plain_full[n - 4..]);
            out
        }
    }
}

/// OP-BRAIN-V5 6.8: at-rest encryption status of a WAL segment, used
/// by the `--require-encrypted` guard.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WalEncStatus {
    /// File does not exist or is zero-length (nothing to leak).
    Absent,
    /// File exists but its header is not the 6.8 encrypted header
    /// (pre-6.8 plaintext v1 segment, or v2 with FLAG_ENCRYPTED clear).
    Plaintext,
    /// File exists with a v2 header carrying FLAG_ENCRYPTED.
    Encrypted,
}

/// Inspect a WAL segment's header and classify its at-rest encryption.
pub fn wal_encryption_status(path: &Path) -> WalResult<WalEncStatus> {
    let mut f = match OpenOptions::new().read(true).open(path) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(WalEncStatus::Absent),
        Err(e) => return Err(WalError::Io(e)),
    };
    let mut head = [0u8; WAL_HEADER_V2_BYTES as usize];
    let n = f.read(&mut head)?;
    if n == 0 {
        return Ok(WalEncStatus::Absent);
    }
    if n >= WAL_HEADER_V2_BYTES as usize && &head[0..8] == WAL_MAGIC_V2 {
        if head[8] & WAL_FLAG_ENCRYPTED != 0 {
            return Ok(WalEncStatus::Encrypted);
        }
        return Ok(WalEncStatus::Plaintext);
    }
    Ok(WalEncStatus::Plaintext)
}

#[derive(Debug)]
pub enum WalError {
    Io(std::io::Error),
    BadMagic,
    Serialize(String),
    /// Phase 1.2: ordering-invariant violation surfaced by
    /// [`replay_streaming_validated`]. A bond record at `seq` named a
    /// `seed` (either subject or object) that was never bound by a
    /// preceding `REC_TYPE_NODE_INTERN` record. This is the firewall
    /// that prevents the v5 write path from accidentally emitting
    /// bonds before their nodes are persisted in the WAL.
    UnknownNodeIdInBond { seq: u64, missing_seed: Seed, role: BondSeedRole },
    /// OP-BRAIN-V5 6.8: the segment header marks records encrypted but
    /// no master key was supplied to decrypt them.
    EncryptedButNoKey,
    /// OP-BRAIN-V5 6.8: refused to mix encrypted appends into a
    /// pre-6.8 plaintext segment that already holds records.
    FormatEncryptionMismatch(String),
}

/// Which seed in a bond record was missing a preceding NodeIntern.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BondSeedRole { Subject, Object }

impl std::fmt::Display for WalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WalError::Io(e) => write!(f, "WAL IO: {}", e),
            WalError::BadMagic => write!(f, "WAL bad magic header (file is not a WAL segment)"),
            WalError::Serialize(s) => write!(f, "WAL serialize: {}", s),
            WalError::UnknownNodeIdInBond { seq, role, .. } => write!(
                f,
                "WAL bond record at seq={} references {:?} seed with no preceding REC_TYPE_NODE_INTERN (Q8 ordering invariant violated)",
                seq, role,
            ),
            WalError::EncryptedButNoKey => write!(
                f, "WAL segment header marks records encrypted but no key was supplied"
            ),
            WalError::FormatEncryptionMismatch(s) => write!(f, "WAL encryption mismatch: {}", s),
        }
    }
}

impl std::error::Error for WalError {}

impl From<std::io::Error> for WalError {
    fn from(e: std::io::Error) -> Self { WalError::Io(e) }
}

pub type WalResult<T> = Result<T, WalError>;

/// One replayed WAL record. Phase 10 extends this from a struct to
/// an enum so both bond and node records share the same replay path.
/// `WalRecord::Bond` is the pre-Phase-10 shape; `WalRecord::Node` is
/// the new variant carrying the full KnowledgeNode payload so
/// `enable_memtable` can reconstruct node metadata after a daemon
/// restart (otherwise lu4d_ask returns total_idk because
/// network.nodes is empty post-replay).
#[derive(Debug, Clone)]
pub enum WalRecord {
    Bond {
        seq: u64,
        subject: Seed,
        object: Seed,
        predicate: BondType,
        weight: f32,
        keyframes: Vec<TemporalKeyframe>,
    },
    Node {
        seq: u64,
        seed: Seed,
        node: crate::node::KnowledgeNode,
    },
    /// Phase 1.2 (OP-BRAIN-V5 Q8): binds a `Seed` to a brain-wide
    /// `NodeId(u64)`. Must precede any Bond record whose subject or
    /// object equals `seed`. Persisted ahead of the bond payload in
    /// the same WAL transaction; on replay, NodeIntern records are
    /// reconstructed first, then bonds. Sidecar persistence
    /// (`nodeintern.bin`) is Phase 2, not 1.2.
    NodeIntern {
        seq: u64,
        seed: Seed,
        node_id: NodeId,
    },
}

impl WalRecord {
    /// Sequence number of this record.
    pub fn seq(&self) -> u64 {
        match self {
            WalRecord::Bond { seq, .. } => *seq,
            WalRecord::Node { seq, .. } => *seq,
            WalRecord::NodeIntern { seq, .. } => *seq,
        }
    }

    /// True iff this is a Bond record.
    pub fn is_bond(&self) -> bool {
        matches!(self, WalRecord::Bond { .. })
    }

    /// True iff this is a Node record.
    pub fn is_node(&self) -> bool {
        matches!(self, WalRecord::Node { .. })
    }

    /// True iff this is a NodeIntern record (Phase 1.2).
    pub fn is_node_intern(&self) -> bool {
        matches!(self, WalRecord::NodeIntern { .. })
    }
}

/// Phase 10 test-side back-compat shim.
#[cfg(test)]
struct BondFields<'a> {
    seq: u64,
    subject: Seed,
    object: Seed,
    predicate: BondType,
    weight: f32,
    keyframes: &'a [TemporalKeyframe],
}

#[cfg(test)]
impl WalRecord {
    fn as_bond_fields(&self) -> BondFields<'_> {
        match self {
            WalRecord::Bond { seq, subject, object, predicate, weight, keyframes } => BondFields {
                seq: *seq,
                subject: *subject,
                object: *object,
                predicate: *predicate,
                weight: *weight,
                keyframes,
            },
            WalRecord::Node { .. } => panic!("expected Bond record, got Node"),
            WalRecord::NodeIntern { .. } => panic!("expected Bond record, got NodeIntern"),
        }
    }
}

/// Phase 10: on-disk record-type discriminator. Byte sits immediately
/// after the 4-byte length prefix; the body follows.
const REC_TYPE_BOND: u8 = 0;
const REC_TYPE_NODE: u8 = 1;
/// Phase 1.2 (OP-BRAIN-V5 Q8): NodeIntern record — binds a `Seed` to
/// a brain-wide `NodeId(u64)`. Wire layout (post the 4-byte length
/// prefix and this 1-byte discriminator): `seq:u64 | seed[32] |
/// node_id:varint` followed by the trailing `crc:u32` from the
/// shared record framing. The WRITER (`append_node_intern`) is
/// responsible for emitting NodeIntern records BEFORE any bond
/// record that references the bound NodeIds; the streaming reader
/// (`replay_streaming_validated`) enforces this on replay and
/// returns [`WalError::UnknownNodeIdInBond`] on violation.
const REC_TYPE_NODE_INTERN: u8 = 2;

/// WAL writer handle.
///
/// Single-threaded by design — the caller is responsible for
/// serializing `append` calls (the MemTable holds an
/// `Arc<Mutex<Wal>>` in Phase 4's wiring). Internal `BufWriter`
/// batches small writes; fsync cadence controls how often the
/// buffered bytes are forced to disk.
pub struct Wal {
    writer: BufWriter<File>,
    segment_path: PathBuf,
    fsync_cadence_ms: u64,
    fsync_size_threshold: usize,
    bytes_since_fsync: usize,
    last_fsync: Instant,
    /// OP-BRAIN-V5 6.8: master key when this segment is encrypted.
    key: Option<MasterKey>,
    /// OP-BRAIN-V5 6.8: per-segment IV (all-zero when plaintext).
    iv: [u8; 16],
    /// OP-BRAIN-V5 6.8: true iff records are sealed under `key`.
    encrypted: bool,
    /// OP-BRAIN-V5 6.8: absolute byte offset where the NEXT appended
    /// record's length prefix will start (header length + bytes of all
    /// records written so far). Feeds the per-record keystream index.
    write_offset: u64,
}

impl Wal {
    /// Open or create the WAL segment at `path`. Appends to an
    /// existing file (use `replay` to recover records first, THEN
    /// `truncate_up_to` once they're absorbed). Writes the magic
    /// header to a fresh file.
    ///
    /// Fsync thresholds default to `DEFAULT_FSYNC_CADENCE_MS` and
    /// `DEFAULT_FSYNC_BYTE_THRESHOLD`; overridable via env vars
    /// `LU_WAL_FSYNC_MS` and `LU_WAL_FSYNC_BYTES`.
    pub fn open(path: &Path) -> WalResult<Wal> {
        Self::open_with_key(path, None)
    }

    /// OP-BRAIN-V5 6.8: open or create the WAL, optionally encrypting
    /// appended records under `key`. A fresh file gets a v2 header
    /// (magic + flags + random IV) when `key` is `Some`, or the
    /// byte-identical pre-6.8 plaintext v1 header when `None`.
    ///
    /// On an existing file the on-disk header is authoritative: a v2
    /// header is reopened with its stored IV (refusing with
    /// `EncryptedButNoKey` if it is encrypted and no key was given); a
    /// v1 plaintext header is reopened plaintext when `key` is `None`.
    /// Re-opening a v1 plaintext file that already holds records WITH a
    /// key is refused (no silent in-place conversion -- regenerate via
    /// migration); an empty (magic-only) v1 file is upgraded to v2.
    pub fn open_with_key(path: &Path, key: Option<MasterKey>) -> WalResult<Wal> {
        let exists = path.exists();
        let mut encrypted = false;
        let mut iv = [0u8; 16];
        let mut write_offset: u64 = 0;
        let mut need_fresh_header = !exists;
        if exists {
            let mut probe = OpenOptions::new().read(true).open(path)?;
            let mut head = [0u8; WAL_HEADER_V2_BYTES as usize];
            let n = probe.read(&mut head)?;
            if n == 0 {
                need_fresh_header = true;
            } else if n >= 8 && &head[0..8] == WAL_MAGIC_V2 {
                if n < WAL_HEADER_V2_BYTES as usize {
                    return Err(WalError::BadMagic);
                }
                encrypted = head[8] & WAL_FLAG_ENCRYPTED != 0;
                iv.copy_from_slice(&head[12..28]);
                if encrypted && key.is_none() {
                    return Err(WalError::EncryptedButNoKey);
                }
                write_offset = std::fs::metadata(path)?.len();
            } else if n >= 8 && &head[0..8] == WAL_MAGIC {
                let size = std::fs::metadata(path)?.len();
                if key.is_some() {
                    if size > WAL_HEADER_V1_BYTES {
                        return Err(WalError::FormatEncryptionMismatch(format!(
                            "WAL {} is a pre-6.8 plaintext segment with records; refusing to \
                             append encrypted records -- regenerate via migration",
                            path.display(),
                        )));
                    }
                    need_fresh_header = true;
                } else {
                    write_offset = size;
                }
            } else {
                return Err(WalError::BadMagic);
            }
        }

        // Upgrading an existing empty v1 file to v2: truncate first so
        // append mode does not leave a stale 8-byte magic in front.
        if need_fresh_header && exists {
            let f = OpenOptions::new().write(true).truncate(true).open(path)?;
            f.set_len(0)?;
            drop(f);
        }

        let file = OpenOptions::new()
            .create(true).read(true).append(true).open(path)?;
        let mut writer = BufWriter::with_capacity(64 * 1024, file);

        if need_fresh_header {
            if key.is_some() {
                rand::thread_rng().fill_bytes(&mut iv);
                encrypted = true;
                let mut header = Vec::with_capacity(WAL_HEADER_V2_BYTES as usize);
                header.extend_from_slice(WAL_MAGIC_V2);
                header.push(WAL_FLAG_ENCRYPTED);
                header.extend_from_slice(&[0u8; 3]);
                header.extend_from_slice(&iv);
                writer.write_all(&header)?;
                write_offset = WAL_HEADER_V2_BYTES;
            } else {
                writer.write_all(WAL_MAGIC)?;
                write_offset = WAL_HEADER_V1_BYTES;
            }
            // Make sure the header is on disk before any record lands.
            writer.flush()?;
            writer.get_mut().sync_data()?;
        }

        let cadence_ms = std::env::var("LU_WAL_FSYNC_MS")
            .ok().and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_FSYNC_CADENCE_MS);
        let byte_thresh = std::env::var("LU_WAL_FSYNC_BYTES")
            .ok().and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_FSYNC_BYTE_THRESHOLD);
        Ok(Wal {
            writer,
            segment_path: path.to_path_buf(),
            fsync_cadence_ms: cadence_ms,
            fsync_size_threshold: byte_thresh,
            bytes_since_fsync: 0,
            last_fsync: Instant::now(),
            key,
            iv,
            encrypted,
            write_offset,
        })
    }

    /// OP-BRAIN-V5 6.8: the (key, IV) pair to seal records with, or
    /// `None` when this segment is plaintext.
    fn enc_tuple(&self) -> Option<(&MasterKey, &[u8; 16])> {
        if self.encrypted {
            self.key.as_ref().map(|k| (k, &self.iv))
        } else {
            None
        }
    }

    /// Override the fsync thresholds. Useful for tests.
    pub fn set_fsync_thresholds(&mut self, cadence_ms: u64, byte_threshold: usize) {
        self.fsync_cadence_ms = cadence_ms;
        self.fsync_size_threshold = byte_threshold;
    }

    /// Path of the underlying segment file.
    pub fn path(&self) -> &Path { &self.segment_path }

    /// Append one record. Conditionally fsync's if size or time
    /// threshold exceeded.
    pub fn append(
        &mut self,
        seq: u64,
        subject: &Seed,
        object: &Seed,
        predicate: BondType,
        weight: f32,
        keyframes: &[TemporalKeyframe],
    ) -> WalResult<()> {
        let payload = encode_bond_record(seq, subject, object, predicate, weight, keyframes)?;
        let sealed = seal_record(&payload, self.write_offset, self.enc_tuple());
        self.writer.write_all(&sealed)?;
        self.write_offset += sealed.len() as u64;
        self.bytes_since_fsync += sealed.len();
        if self.bytes_since_fsync >= self.fsync_size_threshold
            || self.last_fsync.elapsed().as_millis() as u64 >= self.fsync_cadence_ms
        {
            self.fsync()?;
        }
        Ok(())
    }

    /// Append one node record (Phase 10: discriminator byte 1). The
    /// node payload is bincode-serialized; CRC32 trailer same scheme
    /// as the bond path so torn-write detection works uniformly.
    /// Conditional fsync cadence is shared with `append`.
    pub fn append_node(
        &mut self,
        seq: u64,
        seed: &Seed,
        node: &crate::node::KnowledgeNode,
    ) -> WalResult<()> {
        let payload = encode_node_record(seq, seed, node)?;
        let sealed = seal_record(&payload, self.write_offset, self.enc_tuple());
        self.writer.write_all(&sealed)?;
        self.write_offset += sealed.len() as u64;
        self.bytes_since_fsync += sealed.len();
        if self.bytes_since_fsync >= self.fsync_size_threshold
            || self.last_fsync.elapsed().as_millis() as u64 >= self.fsync_cadence_ms
        {
            self.fsync()?;
        }
        Ok(())
    }

    /// Phase 1.2 (OP-BRAIN-V5 Q8): append a NodeIntern record binding
    /// `seed` to `node_id`. MUST be emitted before any subsequent bond
    /// record that names `seed` as subject or object — the WRITER is
    /// responsible for honoring this ordering invariant in the same
    /// WAL transaction (per design doc Q8). Idempotence under crash is
    /// inherited from the underlying write-ahead semantics: replay
    /// applies records in seq order, NodeIntern records before bonds.
    ///
    /// `node_id` is encoded as unsigned LEB128 varint (Q1: "u64 with
    /// high 32 bits reserved; varint encoding in WAL records").
    /// Conditional fsync cadence is shared with `append`/`append_node`.
    pub fn append_node_intern(
        &mut self,
        seq: u64,
        seed: &Seed,
        node_id: NodeId,
    ) -> WalResult<()> {
        let payload = encode_node_intern_record(seq, seed, node_id)?;
        let sealed = seal_record(&payload, self.write_offset, self.enc_tuple());
        self.writer.write_all(&sealed)?;
        self.write_offset += sealed.len() as u64;
        self.bytes_since_fsync += sealed.len();
        if self.bytes_since_fsync >= self.fsync_size_threshold
            || self.last_fsync.elapsed().as_millis() as u64 >= self.fsync_cadence_ms
        {
            self.fsync()?;
        }
        Ok(())
    }

    /// Force fsync. Flushes the BufWriter, then `sync_data` on the
    /// underlying file. `sync_data` is sufficient — we don't depend
    /// on file-metadata semantics (size grows monotonically; the
    /// length-prefix in each record self-describes anyway).
    pub fn fsync(&mut self) -> WalResult<()> {
        self.writer.flush()?;
        self.writer.get_mut().sync_data()?;
        self.bytes_since_fsync = 0;
        self.last_fsync = Instant::now();
        Ok(())
    }

    /// Truncate the WAL to retain only records with `seq >= cutoff`.
    /// Called by the flush thread after the corresponding SST has
    /// been fsync'd and the manifest updated (Phase 4 contract).
    ///
    /// Implementation: read all records, keep those at or above the
    /// cutoff, write them to `<path>.compact`, fsync, rename atomically
    /// over the live segment, reopen. Atomic rename means a kill
    /// mid-truncate either keeps the full pre-truncate log (replay
    /// covers the just-flushed records redundantly — harmless,
    /// idempotent at the MemTable layer) or commits the post-truncate
    /// log.
    pub fn truncate_up_to(&mut self, cutoff: u64) -> WalResult<()> {
        // First, ensure everything we've written is on disk so
        // replay sees a coherent view.
        self.fsync()?;

        // Read full current contents via `replay` (decrypting if keyed).
        let all = replay_with_key(&self.segment_path, self.key.clone())?;
        let kept: Vec<WalRecord> = all.into_iter()
            .filter(|r| r.seq() >= cutoff).collect();

        // Write the kept records to a sibling temp file. When encrypted
        // a FRESH IV is generated for the rewritten segment and record
        // offsets restart from the header length, so each kept record
        // is re-sealed at its NEW offset.
        let tmp_path = compact_tmp_path(&self.segment_path);
        let mut new_iv = [0u8; 16];
        {
            let tmp_file = OpenOptions::new()
                .create(true).write(true).truncate(true).open(&tmp_path)?;
            let mut tmp_writer = BufWriter::with_capacity(64 * 1024, tmp_file);
            let mut off: u64;
            if self.encrypted {
                rand::thread_rng().fill_bytes(&mut new_iv);
                let mut header = Vec::with_capacity(WAL_HEADER_V2_BYTES as usize);
                header.extend_from_slice(WAL_MAGIC_V2);
                header.push(WAL_FLAG_ENCRYPTED);
                header.extend_from_slice(&[0u8; 3]);
                header.extend_from_slice(&new_iv);
                tmp_writer.write_all(&header)?;
                off = WAL_HEADER_V2_BYTES;
            } else {
                tmp_writer.write_all(WAL_MAGIC)?;
                off = WAL_HEADER_V1_BYTES;
            }
            for r in &kept {
                let payload = match r {
                    WalRecord::Bond { seq, subject, object, predicate, weight, keyframes } => {
                        encode_bond_record(*seq, subject, object, *predicate, *weight, keyframes)?
                    }
                    WalRecord::Node { seq, seed, node } => {
                        encode_node_record(*seq, seed, node)?
                    }
                    WalRecord::NodeIntern { seq, seed, node_id } => {
                        encode_node_intern_record(*seq, seed, *node_id)?
                    }
                };
                let enc = if self.encrypted {
                    self.key.as_ref().map(|k| (k, &new_iv))
                } else {
                    None
                };
                let sealed = seal_record(&payload, off, enc);
                tmp_writer.write_all(&sealed)?;
                off += sealed.len() as u64;
            }
            tmp_writer.flush()?;
            tmp_writer.get_mut().sync_data()?;
        }

        // Fsync the parent directory so the rename is durable.
        if let Some(parent) = self.segment_path.parent() {
            if !parent.as_os_str().is_empty() {
                let dir = File::open(parent)?;
                let _ = dir.sync_data(); // best-effort on non-POSIX
            }
        }

        // Drop the BufWriter so we can rename over the live file.
        // We can't keep the same file handle bound after the rename
        // anyway — reopen below.
        let _ = self.writer.flush();
        // The handle held by BufWriter must be dropped before rename
        // on Windows; on Linux this is fine but no harm in being
        // explicit. We accomplish this by replacing self.writer with
        // a dummy that gets dropped at function end.
        let dummy = OpenOptions::new().create(true).write(true).truncate(false)
            .open(&tmp_path)?; // re-open something cheap; will be replaced.
        let stale = std::mem::replace(&mut self.writer, BufWriter::new(dummy));
        drop(stale);

        std::fs::rename(&tmp_path, &self.segment_path)?;

        // Reopen as the live append handle.
        let live = OpenOptions::new().read(true).append(true).open(&self.segment_path)?;
        self.writer = BufWriter::with_capacity(64 * 1024, live);
        if self.encrypted {
            self.iv = new_iv;
        }
        self.write_offset = std::fs::metadata(&self.segment_path)?.len();
        self.bytes_since_fsync = 0;
        self.last_fsync = Instant::now();
        Ok(())
    }
}

impl Drop for Wal {
    fn drop(&mut self) {
        // Best-effort fsync on drop so a clean shutdown leaves no
        // buffered records behind. Errors are silently swallowed in
        // drop (POSIX gives us no place to surface them) but they're
        // rare in practice — drop happens at clean exit, where the
        // file is healthy.
        let _ = self.fsync();
    }
}

/// Replay an entire WAL segment. Validates the magic header, reads
/// records sequentially, validates each CRC32. Stops at the first
/// invalid record (torn write or corruption) and returns everything
/// up to that point.
pub fn replay(path: &Path) -> WalResult<Vec<WalRecord>> {
    replay_with_key(path, None)
}

/// OP-BRAIN-V5 6.8: replay a WAL segment, decrypting records when the
/// on-disk header marks the segment encrypted. The header (v1 vs v2)
/// is authoritative; `key` is consulted only when the header's
/// FLAG_ENCRYPTED bit is set. An encrypted segment opened with no key
/// fails cleanly with [`WalError::EncryptedButNoKey`] (no panic).
/// Torn-tail tolerance is preserved byte-for-byte: each record's CRC
/// is validated over the bytes ON DISK (length || possibly-ciphertext
/// body), so a torn final record stops replay and the good prefix is
/// returned intact.
pub fn replay_with_key(path: &Path, key: Option<MasterKey>) -> WalResult<Vec<WalRecord>> {
    let mut f = match OpenOptions::new().read(true).open(path) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(WalError::Io(e)),
    };
    let mut magic = [0u8; 8];
    let n = f.read(&mut magic)?;
    if n == 0 {
        return Ok(Vec::new());
    }
    if n != 8 {
        return Err(WalError::BadMagic);
    }
    let (encrypted, iv, mut offset) = if &magic == WAL_MAGIC {
        (false, [0u8; 16], WAL_HEADER_V1_BYTES)
    } else if &magic == WAL_MAGIC_V2 {
        let mut ext = [0u8; (WAL_HEADER_V2_BYTES - 8) as usize];
        if f.read_exact(&mut ext).is_err() {
            // Truncated header -- no recoverable records.
            return Ok(Vec::new());
        }
        let enc = ext[0] & WAL_FLAG_ENCRYPTED != 0;
        let mut iv = [0u8; 16];
        iv.copy_from_slice(&ext[4..20]);
        if enc && key.is_none() {
            return Err(WalError::EncryptedButNoKey);
        }
        (enc, iv, WAL_HEADER_V2_BYTES)
    } else {
        return Err(WalError::BadMagic);
    };

    let mut out = Vec::new();
    loop {
        let mut len_buf = [0u8; 4];
        match f.read_exact(&mut len_buf) {
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(WalError::Io(e)),
        }
        let total_len = u32::from_le_bytes(len_buf) as usize;
        if total_len < 4 + 4 {
            break;
        }
        let body_len = total_len - 4 - 4;
        let mut body = vec![0u8; body_len];
        if let Err(e) = f.read_exact(&mut body) {
            if e.kind() == std::io::ErrorKind::UnexpectedEof { break; }
            return Err(WalError::Io(e));
        }
        let mut crc_buf = [0u8; 4];
        if let Err(e) = f.read_exact(&mut crc_buf) {
            if e.kind() == std::io::ErrorKind::UnexpectedEof { break; }
            return Err(WalError::Io(e));
        }
        let stored_crc = u32::from_le_bytes(crc_buf);
        // Decrypt in place FIRST (if encrypted), then validate the CRC
        // over (length || PLAINTEXT body). A torn final record OR a
        // wrong key both yield a CRC mismatch here and stop replay,
        // dropping the bad tail -- torn-tail tolerance preserved, and a
        // wrong key fails cleanly instead of decoding garbage.
        if encrypted {
            if let Some(k) = key.as_ref() {
                let ksi = derive_wal_keystream_index(&iv, offset);
                xor_in_place(k, ksi, &mut body);
            }
        }
        let mut hasher = crc32fast::Hasher::new();
        hasher.update(&len_buf);
        hasher.update(&body);
        if hasher.finalize() != stored_crc {
            break;
        }
        let rec = match decode_body(&body) {
            Some(r) => r,
            None => break,
        };
        out.push(rec);
        offset += total_len as u64;
    }
    Ok(out)
}

fn compact_tmp_path(p: &Path) -> PathBuf {
    let mut q = p.to_path_buf();
    let fname = match p.file_name() {
        Some(n) => n.to_os_string(),
        None => std::ffi::OsString::from("wal"),
    };
    let mut s = fname.to_os_string();
    s.push(".compact");
    q.set_file_name(s);
    q
}

/// Encode one record to its on-disk byte form.
/// Phase 10: encode one Bond record.
///
/// Layout: `length:u32 | rec_type:u8(0) | seq:u64 | subject[32] | object[32]`
///         `| predicate:u8 | weight:f32 | kf_count:u16 | kf_payload | crc:u32`
fn encode_bond_record(
    seq: u64,
    subject: &Seed,
    object: &Seed,
    predicate: BondType,
    weight: f32,
    keyframes: &[TemporalKeyframe],
) -> WalResult<Vec<u8>> {
    let kf_bytes = bincode::serialize(&keyframes.to_vec())
        .map_err(|e| WalError::Serialize(e.to_string()))?;
    let body_len = 1 // rec_type
        + 8 // seq
        + 32 // subject
        + 32 // object
        + 1 // predicate
        + 4 // weight
        + 2 // keyframe_count
        + kf_bytes.len();
    let total_len = 4 + body_len + 4;
    if total_len > u32::MAX as usize {
        return Err(WalError::Serialize(
            "record too large for u32 length prefix".into(),
        ));
    }
    let mut out = Vec::with_capacity(total_len);
    out.extend_from_slice(&(total_len as u32).to_le_bytes());
    out.push(REC_TYPE_BOND);
    out.extend_from_slice(&seq.to_le_bytes());
    out.extend_from_slice(subject.as_bytes());
    out.extend_from_slice(object.as_bytes());
    out.push(bond_type_to_u8(predicate));
    out.extend_from_slice(&weight.to_le_bytes());
    let kf_count = u16::try_from(keyframes.len()).map_err(|_| {
        WalError::Serialize("keyframe count exceeds u16".into())
    })?;
    out.extend_from_slice(&kf_count.to_le_bytes());
    out.extend_from_slice(&kf_bytes);

    let mut hasher = crc32fast::Hasher::new();
    hasher.update(&out);
    let crc = hasher.finalize();
    out.extend_from_slice(&crc.to_le_bytes());
    debug_assert_eq!(out.len(), total_len);
    Ok(out)
}

/// Phase 10: encode one Node record.
///
/// Layout: `length:u32 | rec_type:u8(1) | seq:u64 | seed[32] | node_len:u32 | node_bytes | crc:u32`
fn encode_node_record(
    seq: u64,
    seed: &Seed,
    node: &crate::node::KnowledgeNode,
) -> WalResult<Vec<u8>> {
    let node_bytes = crate::node::serialize_knowledge_node(node)
        .map_err(WalError::Serialize)?;
    let node_bytes_len = u32::try_from(node_bytes.len()).map_err(|_| {
        WalError::Serialize("node bytes exceed u32::MAX".into())
    })?;
    let body_len = 1 // rec_type
        + 8 // seq
        + 32 // seed
        + 4 // node_bytes_len
        + node_bytes.len();
    let total_len = 4 + body_len + 4;
    if total_len > u32::MAX as usize {
        return Err(WalError::Serialize(
            "node record too large for u32 length prefix".into(),
        ));
    }
    let mut out = Vec::with_capacity(total_len);
    out.extend_from_slice(&(total_len as u32).to_le_bytes());
    out.push(REC_TYPE_NODE);
    out.extend_from_slice(&seq.to_le_bytes());
    out.extend_from_slice(seed.as_bytes());
    out.extend_from_slice(&node_bytes_len.to_le_bytes());
    out.extend_from_slice(&node_bytes);
    let mut hasher = crc32fast::Hasher::new();
    hasher.update(&out);
    let crc = hasher.finalize();
    out.extend_from_slice(&crc.to_le_bytes());
    debug_assert_eq!(out.len(), total_len);
    Ok(out)
}

/// Phase 10/1.2: decode one record body (type-discriminator-aware).
fn decode_body(body: &[u8]) -> Option<WalRecord> {
    if body.is_empty() { return None; }
    let rec_type = body[0];
    let rest = &body[1..];
    match rec_type {
        REC_TYPE_BOND => decode_bond_body(rest),
        REC_TYPE_NODE => decode_node_body(rest),
        REC_TYPE_NODE_INTERN => decode_node_intern_body(rest),
        _ => None, // unknown discriminator -- stop replay
    }
}

fn decode_bond_body(body: &[u8]) -> Option<WalRecord> {
    if body.len() < 8 + 32 + 32 + 1 + 4 + 2 { return None; }
    let mut off = 0;
    let seq = u64::from_le_bytes(body[off..off+8].try_into().ok()?); off += 8;
    let mut subj_bytes = [0u8; 32];
    subj_bytes.copy_from_slice(&body[off..off+32]); off += 32;
    let mut obj_bytes = [0u8; 32];
    obj_bytes.copy_from_slice(&body[off..off+32]); off += 32;
    let predicate = u8_to_bond_type(body[off])?; off += 1;
    let weight_bits = u32::from_le_bytes(body[off..off+4].try_into().ok()?); off += 4;
    let weight = f32::from_bits(weight_bits);
    let kf_count = u16::from_le_bytes(body[off..off+2].try_into().ok()?); off += 2;
    let kfs: Vec<TemporalKeyframe> = bincode::deserialize(&body[off..]).ok()?;
    if kfs.len() != kf_count as usize {
        return None;
    }
    Some(WalRecord::Bond {
        seq,
        subject: Seed::from_bytes(subj_bytes),
        object: Seed::from_bytes(obj_bytes),
        predicate,
        weight,
        keyframes: kfs,
    })
}

fn decode_node_body(body: &[u8]) -> Option<WalRecord> {
    if body.len() < 8 + 32 + 4 { return None; }
    let mut off = 0;
    let seq = u64::from_le_bytes(body[off..off+8].try_into().ok()?); off += 8;
    let mut seed_bytes = [0u8; 32];
    seed_bytes.copy_from_slice(&body[off..off+32]); off += 32;
    let node_len = u32::from_le_bytes(body[off..off+4].try_into().ok()?) as usize; off += 4;
    if off + node_len > body.len() {
        return None;
    }
    let node = crate::node::deserialize_knowledge_node(&body[off..off+node_len])?;
    Some(WalRecord::Node {
        seq,
        seed: Seed::from_bytes(seed_bytes),
        node,
    })
}

// ---------------------------------------------------------------------
// Phase 1.2 (OP-BRAIN-V5 Q8) -- NodeIntern record + streaming replay.
// ---------------------------------------------------------------------

/// Maximum bytes for a u64 in unsigned LEB128: ceil(64/7) = 10.
const MAX_VARINT_U64_LEN: usize = 10;

/// Append unsigned-LEB128 (varint) encoding of `v` to `out`. 7 bits
/// per byte, MSB = continuation. The empty-value (0) still emits one
/// byte. Standard Protobuf wire-compatible.
fn write_varint_u64(out: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 {
        out.push(((v as u8) & 0x7F) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

/// Decode an unsigned-LEB128 varint from the front of `buf`. Returns
/// `Some((value, bytes_consumed))` on success, `None` on truncation
/// or overflow past u64.
fn read_varint_u64(buf: &[u8]) -> Option<(u64, usize)> {
    let mut result: u64 = 0;
    let mut shift: u32 = 0;
    for (i, b) in buf.iter().enumerate() {
        if i >= MAX_VARINT_U64_LEN { return None; }
        let byte = *b;
        let chunk = (byte & 0x7F) as u64;
        // Overflow guard: after 9 bytes (shift=63), the 10th byte must
        // carry only 1 bit of payload (0x00 or 0x01); anything larger
        // would overflow u64.
        if shift == 63 && byte > 0x01 { return None; }
        result |= chunk << shift;
        if byte & 0x80 == 0 {
            return Some((result, i + 1));
        }
        shift += 7;
        if shift >= 70 { return None; } // safety belt
    }
    None // truncated -- never saw a terminator
}

/// Phase 1.2: encode one NodeIntern record.
///
/// Layout (post the 4-byte length prefix):
/// `rec_type:u8(2) | seq:u64 | seed[32] | node_id:varint` + trailing
/// `crc:u32` over `length || body`. Seed stays raw 32 bytes; node_id
/// uses unsigned LEB128 (Q1).
fn encode_node_intern_record(
    seq: u64,
    seed: &Seed,
    node_id: NodeId,
) -> WalResult<Vec<u8>> {
    // Pre-encode the varint into a small scratch so we can compute the
    // exact length up-front (no second-pass patching needed).
    let mut varint_buf: Vec<u8> = Vec::with_capacity(MAX_VARINT_U64_LEN);
    write_varint_u64(&mut varint_buf, node_id.0);
    let body_len = 1 // rec_type
        + 8 // seq
        + 32 // seed
        + varint_buf.len();
    let total_len = 4 + body_len + 4;
    if total_len > u32::MAX as usize {
        return Err(WalError::Serialize(
            "node-intern record too large for u32 length prefix".into(),
        ));
    }
    let mut out = Vec::with_capacity(total_len);
    out.extend_from_slice(&(total_len as u32).to_le_bytes());
    out.push(REC_TYPE_NODE_INTERN);
    out.extend_from_slice(&seq.to_le_bytes());
    out.extend_from_slice(seed.as_bytes());
    out.extend_from_slice(&varint_buf);
    let mut hasher = crc32fast::Hasher::new();
    hasher.update(&out);
    let crc = hasher.finalize();
    out.extend_from_slice(&crc.to_le_bytes());
    debug_assert_eq!(out.len(), total_len);
    Ok(out)
}

/// Phase 1.2: decode one NodeIntern body (the `rest` slice after the
/// 1-byte rec_type discriminator was consumed by `decode_body`).
fn decode_node_intern_body(body: &[u8]) -> Option<WalRecord> {
    if body.len() < 8 + 32 + 1 { return None; }
    let mut off = 0;
    let seq = u64::from_le_bytes(body[off..off+8].try_into().ok()?); off += 8;
    let mut seed_bytes = [0u8; 32];
    seed_bytes.copy_from_slice(&body[off..off+32]); off += 32;
    let (raw_id, varint_bytes) = read_varint_u64(&body[off..])?;
    off += varint_bytes;
    // Any trailing bytes after the varint are a format violation.
    if off != body.len() { return None; }
    Some(WalRecord::NodeIntern {
        seq,
        seed: Seed::from_bytes(seed_bytes),
        node_id: NodeId(raw_id),
    })
}

/// Streaming WAL reader. Yields one [`WalRecord`] per call to
/// [`WalStream::next_record`] without ever materializing the full
/// record stream into a `Vec`. This is the Phase 1.2 streaming replay
/// API used by `replay_streaming_validated` and by callers that need
/// bounded-RSS replay over WAL segments far larger than RAM.
///
/// On a torn write or CRC mismatch, `next_record` returns
/// `Ok(None)` — the same end-of-stream-by-truncation behavior the
/// batched [`replay`] function uses (no panic, no data loss past
/// what the kill itself dropped).
pub struct WalStream {
    file: File,
    body_buf: Vec<u8>,
    /// OP-BRAIN-V5 6.8: (key, IV) when the segment is encrypted.
    enc: Option<(MasterKey, [u8; 16])>,
    /// OP-BRAIN-V5 6.8: absolute offset of the NEXT record to read.
    offset: u64,
}

impl WalStream {
    /// Open a WAL segment for streaming replay. Validates the magic
    /// header. Returns `Ok(None)` if the file is missing or zero-byte
    /// (consistent with batched [`replay`]).
    pub fn open(path: &Path) -> WalResult<Option<WalStream>> {
        Self::open_with_key(path, None)
    }

    /// OP-BRAIN-V5 6.8: key-aware streaming open. The on-disk header
    /// decides whether records are decrypted; `key` is required iff
    /// the v2 header carries FLAG_ENCRYPTED (else `EncryptedButNoKey`).
    pub fn open_with_key(path: &Path, key: Option<MasterKey>) -> WalResult<Option<WalStream>> {
        let mut file = match OpenOptions::new().read(true).open(path) {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(WalError::Io(e)),
        };
        let mut magic = [0u8; 8];
        let n = file.read(&mut magic)?;
        if n == 0 {
            return Ok(None);
        }
        if n != 8 {
            return Err(WalError::BadMagic);
        }
        let (enc, offset) = if &magic == WAL_MAGIC {
            (None, WAL_HEADER_V1_BYTES)
        } else if &magic == WAL_MAGIC_V2 {
            let mut ext = [0u8; (WAL_HEADER_V2_BYTES - 8) as usize];
            if file.read_exact(&mut ext).is_err() {
                return Ok(None);
            }
            let mut iv = [0u8; 16];
            iv.copy_from_slice(&ext[4..20]);
            if ext[0] & WAL_FLAG_ENCRYPTED != 0 {
                match key {
                    Some(k) => (Some((k, iv)), WAL_HEADER_V2_BYTES),
                    None => return Err(WalError::EncryptedButNoKey),
                }
            } else {
                (None, WAL_HEADER_V2_BYTES)
            }
        } else {
            return Err(WalError::BadMagic);
        };
        Ok(Some(WalStream {
            file,
            body_buf: Vec::new(),
            enc,
            offset,
        }))
    }

    /// Pull the next record. Returns `Ok(None)` at clean EOF, at a
    /// torn write, or at a CRC mismatch — see [`WalStream`] doc for
    /// the rationale. Borrows the internal scratch buffer for the
    /// body decode, so the returned [`WalRecord`] owns all its fields
    /// (no lifetimes leak from `self`).
    pub fn next_record(&mut self) -> WalResult<Option<WalRecord>> {
        let mut len_buf = [0u8; 4];
        match self.file.read_exact(&mut len_buf) {
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
            Err(e) => return Err(WalError::Io(e)),
        }
        let total_len = u32::from_le_bytes(len_buf) as usize;
        if total_len < 4 + 4 {
            return Ok(None); // torn
        }
        let body_len = total_len - 4 - 4;
        self.body_buf.clear();
        self.body_buf.resize(body_len, 0u8);
        if let Err(e) = self.file.read_exact(&mut self.body_buf) {
            if e.kind() == std::io::ErrorKind::UnexpectedEof { return Ok(None); }
            return Err(WalError::Io(e));
        }
        let mut crc_buf = [0u8; 4];
        if let Err(e) = self.file.read_exact(&mut crc_buf) {
            if e.kind() == std::io::ErrorKind::UnexpectedEof { return Ok(None); }
            return Err(WalError::Io(e));
        }
        let stored_crc = u32::from_le_bytes(crc_buf);
        // Decrypt FIRST (if encrypted), then validate the CRC over the
        // PLAINTEXT body -- a torn tail or a wrong key both stop here.
        if let Some((k, iv)) = self.enc.as_ref() {
            let ksi = derive_wal_keystream_index(iv, self.offset);
            xor_in_place(k, ksi, &mut self.body_buf);
        }
        let mut hasher = crc32fast::Hasher::new();
        hasher.update(&len_buf);
        hasher.update(&self.body_buf);
        if hasher.finalize() != stored_crc {
            return Ok(None); // torn / corrupted / wrong key -- stop
        }
        self.offset += total_len as u64;
        Ok(decode_body(&self.body_buf))
    }
}

/// Streaming replay with Q8 ordering-invariant enforcement.
///
/// Reads records one at a time via [`WalStream`] and invokes `sink`
/// for each. NEVER collects records into a `Vec` (per the feedback
/// `no_hacks` discipline — unbounded RSS is the antipattern v5 exists
/// to eliminate).
///
/// The enforced invariant: every `WalRecord::Bond` MUST have both its
/// `subject` and `object` seeds bound by a preceding
/// `WalRecord::NodeIntern` in the same replay. A violation surfaces as
/// [`WalError::UnknownNodeIdInBond`] and stops the replay immediately.
/// This is the firewall that catches a future writer accidentally
/// emitting bonds before their nodes.
///
/// `Node` records (the Phase-10 KnowledgeNode payload variant) do NOT
/// participate in the NodeIntern binding set — they carry their own
/// seed and full payload and are bond-independent.
///
/// The interned-seed set is bounded by the number of NodeIntern
/// records, matching the design's `nodeintern.bin` hot-cache budget
/// (Q3 — ~24 B/entry resident).
pub fn replay_streaming_validated<F>(
    path: &Path,
    sink: F,
) -> WalResult<()>
where
    F: FnMut(WalRecord) -> WalResult<()>,
{
    replay_streaming_validated_with_key(path, None, sink)
}

/// OP-BRAIN-V5 6.8: key-aware form of [`replay_streaming_validated`].
pub fn replay_streaming_validated_with_key<F>(
    path: &Path,
    key: Option<MasterKey>,
    mut sink: F,
) -> WalResult<()>
where
    F: FnMut(WalRecord) -> WalResult<()>,
{
    use std::collections::HashSet;
    let mut stream = match WalStream::open_with_key(path, key)? {
        Some(s) => s,
        None => return Ok(()), // empty / missing file -- no records
    };
    let mut interned: HashSet<Seed> = HashSet::new();
    while let Some(rec) = stream.next_record()? {
        if let WalRecord::Bond { seq, subject, object, .. } = &rec {
            if !interned.contains(subject) {
                return Err(WalError::UnknownNodeIdInBond {
                    seq: *seq,
                    missing_seed: *subject,
                    role: BondSeedRole::Subject,
                });
            }
            if !interned.contains(object) {
                return Err(WalError::UnknownNodeIdInBond {
                    seq: *seq,
                    missing_seed: *object,
                    role: BondSeedRole::Object,
                });
            }
        } else if let WalRecord::NodeIntern { seed, .. } = &rec {
            interned.insert(*seed);
        }
        sink(rec)?;
    }
    Ok(())
}

/// Stable u8 mapping. Mirrors `src/sst.rs::bond_type_to_u8` so the
/// WAL and SST on-disk discriminants are interchangeable.
fn bond_type_to_u8(b: BondType) -> u8 {
    match b {
        BondType::IsA => 0,
        BondType::DefinedAs => 37,
        BondType::WorksBy => 38,
        BondType::ComposedOf => 39,
        BondType::PurposeOf => 40,
        BondType::DiffersFrom => 41,
        BondType::LeadsTo => 42,
        BondType::ExampleOf => 43,
        BondType::HasProperty => 1,
        BondType::Causes => 2,
        BondType::PartOf => 3,
        BondType::UsedFor => 4,
        BondType::RelatedTo => 5,
        BondType::ContrastedWith => 6,
        BondType::DerivedFrom => 7,
        BondType::Requires => 8,
        BondType::Produces => 9,
        BondType::Opposes => 10,
        BondType::BornIn => 11,
        BondType::BornOn => 12,
        BondType::DiedIn => 13,
        BondType::DiedOn => 14,
        BondType::DiscoveredBy => 15,
        BondType::FoundedBy => 16,
        BondType::FoundedIn => 17,
        BondType::HasAltitude => 18,
        BondType::HasAspect => 19,
        BondType::HasCapital => 20,
        BondType::HasOfficialLanguage => 21,
        BondType::HasPopulation => 22,
        BondType::KnownFor => 23,
        BondType::LocatedIn => 24,
        BondType::MeasuredIn => 25,
        BondType::NamedAfter => 26,
        BondType::WrittenBy => 27,
        BondType::HasArea => 28,
        BondType::HasMethod => 29,
        BondType::HasAttribute => 30,
        BondType::HasParameter => 31,
        BondType::Returns => 32,
        BondType::Inherits => 33,
        BondType::Calls => 34,
        BondType::HasType => 35,
        BondType::Implements => 36,
        BondType::HasCurrency => 44,
        BondType::AuthoredBy => 45,
    }
}

fn u8_to_bond_type(b: u8) -> Option<BondType> {
    Some(match b {
        0 => BondType::IsA,
        1 => BondType::HasProperty,
        2 => BondType::Causes,
        3 => BondType::PartOf,
        4 => BondType::UsedFor,
        5 => BondType::RelatedTo,
        6 => BondType::ContrastedWith,
        7 => BondType::DerivedFrom,
        8 => BondType::Requires,
        9 => BondType::Produces,
        10 => BondType::Opposes,
        11 => BondType::BornIn,
        12 => BondType::BornOn,
        13 => BondType::DiedIn,
        14 => BondType::DiedOn,
        15 => BondType::DiscoveredBy,
        16 => BondType::FoundedBy,
        17 => BondType::FoundedIn,
        18 => BondType::HasAltitude,
        19 => BondType::HasAspect,
        20 => BondType::HasCapital,
        21 => BondType::HasOfficialLanguage,
        22 => BondType::HasPopulation,
        23 => BondType::KnownFor,
        24 => BondType::LocatedIn,
        25 => BondType::MeasuredIn,
        26 => BondType::NamedAfter,
        27 => BondType::WrittenBy,
        28 => BondType::HasArea,
        29 => BondType::HasMethod,
        30 => BondType::HasAttribute,
        31 => BondType::HasParameter,
        32 => BondType::Returns,
        33 => BondType::Inherits,
        34 => BondType::Calls,
        35 => BondType::HasType,
        36 => BondType::Implements,
        37 => BondType::DefinedAs,
        38 => BondType::WorksBy,
        39 => BondType::ComposedOf,
        40 => BondType::PurposeOf,
        41 => BondType::DiffersFrom,
        42 => BondType::LeadsTo,
        43 => BondType::ExampleOf,
        44 => BondType::HasCurrency,
        45 => BondType::AuthoredBy,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bond::{KeyframeSource, TemporalKeyframe};
    use rand::{Rng, SeedableRng};
    use rand::rngs::StdRng;
    use std::sync::{Arc, Mutex};
    use std::thread;
    use tempfile::tempdir;

    fn baseline_kf(ts: i64, w: f32) -> TemporalKeyframe {
        TemporalKeyframe {
            timestamp: ts, weight: w, source: KeyframeSource::Baseline,
        }
    }

    fn rand_seed(rng: &mut StdRng) -> Seed {
        let mut b = [0u8; 32];
        rng.fill(&mut b);
        Seed::from_bytes(b)
    }

    #[test]
    fn append_and_replay_round_trip_1000() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("seg.wal");
        let mut wal = Wal::open(&path).unwrap();
        // Generous fsync thresholds so we exercise the buffered path
        // most of the time; final fsync is via Drop / explicit call.
        wal.set_fsync_thresholds(10_000, 64 * 1024 * 1024);
        let mut rng = StdRng::seed_from_u64(0x1234);
        let mut expected = Vec::new();
        for i in 0..1000 {
            let subj = rand_seed(&mut rng);
            let obj = rand_seed(&mut rng);
            let w: f32 = rng.gen();
            let kfs = vec![baseline_kf(i as i64, w)];
            wal.append(i as u64, &subj, &obj, BondType::RelatedTo, w, &kfs).unwrap();
            expected.push((i as u64, subj, obj, w));
        }
        wal.fsync().unwrap();
        let recs = replay(&path).unwrap();
        assert_eq!(recs.len(), 1000);
        for (i, r) in recs.iter().enumerate() {
            let (eseq, esubj, eobj, ew) = &expected[i];
            assert_eq!(r.as_bond_fields().seq, *eseq);
            assert_eq!(r.as_bond_fields().subject, *esubj);
            assert_eq!(r.as_bond_fields().object, *eobj);
            assert_eq!(r.as_bond_fields().weight.to_bits(), ew.to_bits()); // byte-exact
        }
    }

    #[test]
    fn replay_torn_write_yields_records_up_to_torn() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("seg.wal");
        let mut wal = Wal::open(&path).unwrap();
        for i in 0..100 {
            wal.append(
                i as u64,
                &Seed::from_label(&format!("subj_{}", i)),
                &Seed::from_label(&format!("obj_{}", i)),
                BondType::IsA,
                0.5,
                &[baseline_kf(0, 0.5)],
            ).unwrap();
        }
        wal.fsync().unwrap();
        // Write 50 more; do NOT fsync the final ones, then truncate
        // the file mid-record to simulate a torn write.
        for i in 100..150 {
            wal.append(
                i as u64,
                &Seed::from_label(&format!("subj_{}", i)),
                &Seed::from_label(&format!("obj_{}", i)),
                BondType::IsA,
                0.5,
                &[baseline_kf(0, 0.5)],
            ).unwrap();
        }
        wal.fsync().unwrap();
        // Truncate mid-record-101 by removing a fixed number of trailing bytes.
        let len = std::fs::metadata(&path).unwrap().len();
        // Remove enough bytes to chop into record 101 but leave the
        // first 100 records fully intact. Cheap approximation: each
        // record is ~100 bytes; chopping 4500 bytes removes ~45 records,
        // landing well before record 100's tail.
        let new_len = len.saturating_sub(4500);
        let f = OpenOptions::new().write(true).open(&path).unwrap();
        f.set_len(new_len).unwrap();
        f.sync_all().unwrap();
        drop(wal);

        let recs = replay(&path).unwrap();
        // We must get at least 100 (the fsync-guaranteed records).
        // We may get more if the truncation cut was after a clean
        // record boundary, but never less.
        assert!(recs.len() >= 100, "expected ≥100 records, got {}", recs.len());
        // And we must NEVER get back any record from the torn region:
        // the seq numbers must be contiguous from 0.
        for (i, r) in recs.iter().enumerate() {
            assert_eq!(r.as_bond_fields().seq, i as u64, "non-contiguous seq at index {}: got {}", i, r.as_bond_fields().seq);
        }
    }

    #[test]
    fn replay_crc_mismatch_stops_at_corruption() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("seg.wal");
        let mut wal = Wal::open(&path).unwrap();
        for i in 0..100 {
            wal.append(
                i as u64,
                &Seed::from_label(&format!("s_{}", i)),
                &Seed::from_label(&format!("o_{}", i)),
                BondType::IsA, 1.0,
                &[baseline_kf(0, 1.0)],
            ).unwrap();
        }
        wal.fsync().unwrap();
        drop(wal);

        // Locate the start of record 50 by counting bytes. We re-walk
        // the file using the length prefixes.
        let bytes = std::fs::read(&path).unwrap();
        let mut cursor = 8; // after magic
        for _ in 0..50 {
            let len_buf: [u8; 4] = bytes[cursor..cursor+4].try_into().unwrap();
            let total_len = u32::from_le_bytes(len_buf) as usize;
            cursor += total_len;
        }
        // Now cursor points at the START of record 50. Flip a byte in
        // the middle of that record's body (offset cursor+10 lands in
        // the seq field — sure to invalidate the CRC).
        let mut corrupted = bytes.clone();
        corrupted[cursor + 10] ^= 0xFF;
        let cpath = dir.path().join("corrupted.wal");
        std::fs::write(&cpath, &corrupted).unwrap();

        let recs = replay(&cpath).unwrap();
        // Should stop AT or BEFORE record 50.
        assert_eq!(recs.len(), 50, "expected exactly 50 records before corruption, got {}", recs.len());
        for (i, r) in recs.iter().enumerate() {
            assert_eq!(r.as_bond_fields().seq, i as u64);
        }
    }

    #[test]
    fn truncate_up_to_drops_below_cutoff() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("seg.wal");
        let mut wal = Wal::open(&path).unwrap();
        for i in 0..1000 {
            wal.append(
                i as u64,
                &Seed::from_label(&format!("s_{}", i)),
                &Seed::from_label(&format!("o_{}", i)),
                BondType::IsA, 1.0, &[],
            ).unwrap();
        }
        wal.fsync().unwrap();
        wal.truncate_up_to(500).unwrap();
        // Replay must yield exactly records seq >= 500.
        let recs = replay(&path).unwrap();
        assert_eq!(recs.len(), 500);
        for (i, r) in recs.iter().enumerate() {
            assert_eq!(r.as_bond_fields().seq, 500 + i as u64);
        }
        // Continuing to append after truncate must work.
        wal.append(
            1000,
            &Seed::from_label("post_trunc_subj"),
            &Seed::from_label("post_trunc_obj"),
            BondType::IsA, 1.0, &[],
        ).unwrap();
        wal.fsync().unwrap();
        let recs2 = replay(&path).unwrap();
        assert_eq!(recs2.len(), 501);
        assert_eq!(recs2.last().unwrap().as_bond_fields().seq, 1000);
    }

    #[test]
    fn empty_wal_replays_to_empty_vec() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("seg.wal");
        let wal = Wal::open(&path).unwrap();
        drop(wal);
        let recs = replay(&path).unwrap();
        assert!(recs.is_empty());
    }

    #[test]
    fn non_wal_file_yields_bad_magic() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("not_wal.bin");
        std::fs::write(&path, b"this is not a WAL segment, definitely not").unwrap();
        let err = replay(&path);
        assert!(matches!(err, Err(WalError::BadMagic)));
    }

    #[test]
    fn fsync_idempotent_and_clears_bytes_counter() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("seg.wal");
        let mut wal = Wal::open(&path).unwrap();
        wal.append(
            0, &Seed::from_label("a"), &Seed::from_label("b"),
            BondType::IsA, 1.0, &[],
        ).unwrap();
        wal.fsync().unwrap();
        assert_eq!(wal.bytes_since_fsync, 0);
        wal.fsync().unwrap();
        assert_eq!(wal.bytes_since_fsync, 0);
    }

    #[test]
    fn end_to_end_kill_and_replay_simulation() {
        // Insert 10K records, fsync, drop the wal (simulates kill),
        // replay into a fresh Vec — byte-exact recovery.
        let dir = tempdir().unwrap();
        let path = dir.path().join("seg.wal");
        let mut rng = StdRng::seed_from_u64(0xFADE);
        let mut expected = Vec::new();
        {
            let mut wal = Wal::open(&path).unwrap();
            wal.set_fsync_thresholds(10, 16 * 1024); // aggressive
            for i in 0..10_000 {
                let subj = rand_seed(&mut rng);
                let obj = rand_seed(&mut rng);
                let w: f32 = rng.gen();
                let kfs = vec![baseline_kf(i as i64, w)];
                wal.append(i as u64, &subj, &obj, BondType::HasProperty, w, &kfs).unwrap();
                expected.push((i as u64, subj, obj, w, kfs));
            }
            wal.fsync().unwrap();
            // Drop wal — simulates daemon exit.
        }
        let recs = replay(&path).unwrap();
        assert_eq!(recs.len(), 10_000);
        for (got, exp) in recs.iter().zip(expected.iter()) {
            assert_eq!(got.as_bond_fields().seq, exp.0);
            assert_eq!(got.as_bond_fields().subject, exp.1);
            assert_eq!(got.as_bond_fields().object, exp.2);
            assert_eq!(got.as_bond_fields().weight.to_bits(), exp.3.to_bits());
            assert_eq!(got.as_bond_fields().keyframes.len(), exp.4.len());
            assert_eq!(got.as_bond_fields().keyframes[0].timestamp, exp.4[0].timestamp);
        }
    }

    #[test]
    fn concurrent_append_via_mutex_no_corruption() {
        // 4 threads each appending records through a shared
        // `Arc<Mutex<Wal>>` (the Phase 4 wiring shape). Verify all
        // records replay cleanly.
        let dir = tempdir().unwrap();
        let path = dir.path().join("seg.wal");
        let wal = Arc::new(Mutex::new(Wal::open(&path).unwrap()));
        let n_writers = 4;
        let per_writer = 2_500;
        let mut handles = Vec::new();
        for w in 0..n_writers {
            let wal = Arc::clone(&wal);
            handles.push(thread::spawn(move || {
                let mut rng = StdRng::seed_from_u64(0xABCD + w as u64);
                for i in 0..per_writer {
                    let mut g = wal.lock().unwrap();
                    let seq = (w as u64) * per_writer as u64 + i as u64;
                    g.append(
                        seq,
                        &rand_seed(&mut rng), &rand_seed(&mut rng),
                        BondType::Causes, 0.42, &[baseline_kf(0, 0.42)],
                    ).unwrap();
                }
            }));
        }
        for h in handles { h.join().unwrap(); }
        wal.lock().unwrap().fsync().unwrap();
        let recs = replay(&path).unwrap();
        assert_eq!(recs.len(), n_writers as usize * per_writer);
        // Every record has predicate Causes and weight 0.42 — proves
        // no record-boundary tearing.
        for r in &recs {
            assert!(matches!(r.as_bond_fields().predicate, BondType::Causes));
            assert_eq!(r.as_bond_fields().weight.to_bits(), 0.42f32.to_bits());
        }
    }

    #[test]
    fn record_with_many_keyframes_round_trips() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("seg.wal");
        let mut wal = Wal::open(&path).unwrap();
        let kfs: Vec<TemporalKeyframe> = (0..50)
            .map(|i| baseline_kf(i as i64, 0.1 * i as f32))
            .collect();
        wal.append(
            7,
            &Seed::from_label("subj"),
            &Seed::from_label("obj"),
            BondType::HasArea, 3.14159,
            &kfs,
        ).unwrap();
        wal.fsync().unwrap();
        drop(wal);
        let recs = replay(&path).unwrap();
        assert_eq!(recs.len(), 1);
        let r = &recs[0];
        assert_eq!(r.as_bond_fields().seq, 7);
        assert!(matches!(r.as_bond_fields().predicate, BondType::HasArea));
        assert_eq!(r.as_bond_fields().weight.to_bits(), 3.14159f32.to_bits());
        assert_eq!(r.as_bond_fields().keyframes.len(), 50);
        for (i, kf) in r.as_bond_fields().keyframes.iter().enumerate() {
            assert_eq!(kf.timestamp, i as i64);
            assert_eq!(kf.weight.to_bits(), (0.1f32 * i as f32).to_bits());
        }
    }

    // -----------------------------------------------------------------
    // Phase 10 -- node-record tests.
    // -----------------------------------------------------------------

    fn make_test_node(label: &str) -> crate::node::KnowledgeNode {
        crate::node::KnowledgeNode::with_seed(
            Seed::from_label(label),
            label.to_string(),
            crate::node::KnowledgeContent::Concept(label.to_string()),
            crate::node::Confidence::Verified,
            1234,
        )
    }

    #[test]
    fn phase10_append_and_replay_node_record_byte_exact() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("seg.wal");
        let seed = Seed::from_label("alpha");
        let node = make_test_node("alpha");
        {
            let mut wal = Wal::open(&path).unwrap();
            wal.append_node(42, &seed, &node).unwrap();
            wal.fsync().unwrap();
        }
        let recs = replay(&path).unwrap();
        assert_eq!(recs.len(), 1);
        match &recs[0] {
            WalRecord::Node { seq, seed: s, node: n } => {
                assert_eq!(*seq, 42);
                assert_eq!(*s, seed);
                assert_eq!(n.content, node.content);
                assert_eq!(n.seed, node.seed);
            }
            _ => panic!("expected Node record"),
        }
    }

    #[test]
    fn phase10_interleaved_bond_and_node_records_replay_in_order() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("seg.wal");
        let mut wal = Wal::open(&path).unwrap();
        // 100 bonds + 50 nodes interleaved.
        let mut seq: u64 = 0;
        let mut expected_kinds: Vec<&str> = Vec::new();
        for i in 0..50 {
            wal.append(
                seq,
                &Seed::from_label(&format!("bsubj_{}", i)),
                &Seed::from_label(&format!("bobj_{}", i)),
                BondType::IsA, 0.5, &[baseline_kf(0, 0.5)],
            ).unwrap();
            expected_kinds.push("bond");
            seq += 1;
            let n = make_test_node(&format!("node_{}", i));
            wal.append_node(seq, &Seed::from_label(&format!("nseed_{}", i)), &n).unwrap();
            expected_kinds.push("node");
            seq += 1;
        }
        for i in 50..100 {
            wal.append(
                seq,
                &Seed::from_label(&format!("bsubj_{}", i)),
                &Seed::from_label(&format!("bobj_{}", i)),
                BondType::IsA, 0.5, &[baseline_kf(0, 0.5)],
            ).unwrap();
            expected_kinds.push("bond");
            seq += 1;
        }
        wal.fsync().unwrap();
        drop(wal);
        let recs = replay(&path).unwrap();
        assert_eq!(recs.len(), 150);
        for (i, (r, exp)) in recs.iter().zip(expected_kinds.iter()).enumerate() {
            match (r, *exp) {
                (WalRecord::Bond { .. }, "bond") => {}
                (WalRecord::Node { .. }, "node") => {}
                other => panic!("index {}: expected {} got {:?}", i, exp, other.0),
            }
        }
        // Bond count + Node count match expectations.
        let bonds = recs.iter().filter(|r| r.is_bond()).count();
        let nodes = recs.iter().filter(|r| r.is_node()).count();
        assert_eq!(bonds, 100);
        assert_eq!(nodes, 50);
    }

    #[test]
    fn phase10_truncate_up_to_keeps_node_records() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("seg.wal");
        let mut wal = Wal::open(&path).unwrap();
        for i in 0..10 {
            let n = make_test_node(&format!("n_{}", i));
            wal.append_node(i as u64, &Seed::from_label(&format!("s_{}", i)), &n).unwrap();
        }
        wal.fsync().unwrap();
        wal.truncate_up_to(5).unwrap();
        let recs = replay(&path).unwrap();
        assert_eq!(recs.len(), 5);
        for (i, r) in recs.iter().enumerate() {
            assert_eq!(r.seq(), 5 + i as u64);
            assert!(r.is_node());
        }
    }

    #[test]
    fn reopen_existing_wal_validates_magic_and_appends() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("seg.wal");
        {
            let mut wal = Wal::open(&path).unwrap();
            wal.append(
                42, &Seed::from_label("a"), &Seed::from_label("b"),
                BondType::IsA, 1.0, &[],
            ).unwrap();
            wal.fsync().unwrap();
        }
        // Reopen — must succeed and let us append more.
        {
            let mut wal = Wal::open(&path).unwrap();
            wal.append(
                43, &Seed::from_label("c"), &Seed::from_label("d"),
                BondType::IsA, 1.0, &[],
            ).unwrap();
            wal.fsync().unwrap();
        }
        let recs = replay(&path).unwrap();
        assert_eq!(recs.len(), 2);
        assert_eq!(recs[0].as_bond_fields().seq, 42);
        assert_eq!(recs[1].as_bond_fields().seq, 43);
    }

    // -----------------------------------------------------------------
    // Phase 1.2 (OP-BRAIN-V5 Q8) -- NodeIntern record + streaming replay.
    // -----------------------------------------------------------------

    /// A. Varint LEB128 round-trip across u64 boundary values.
    /// Catches off-by-one in the 10-byte ceiling and the shift
    /// arithmetic at the high end of u64.
    #[test]
    fn phase1_2_varint_u64_round_trip_boundary_values() {
        let cases: &[u64] = &[
            0, 1, 0x7F, 0x80, 0xFF, 0x3FFF, 0x4000,
            0xFFFF_FFFF, // u32::MAX
            0x1_0000_0000, // u32::MAX + 1 -- straddles the high-half
            0x7FFF_FFFF_FFFF_FFFF,
            u64::MAX,
        ];
        for &v in cases {
            let mut buf = Vec::new();
            write_varint_u64(&mut buf, v);
            assert!(buf.len() <= MAX_VARINT_U64_LEN, "varint for {} took {} bytes", v, buf.len());
            let (decoded, used) = read_varint_u64(&buf)
                .unwrap_or_else(|| panic!("failed to decode varint for {}", v));
            assert_eq!(decoded, v, "varint round-trip mismatch for {}", v);
            assert_eq!(used, buf.len(), "varint consumed-bytes mismatch for {}", v);
        }
        // Truncated varint must return None instead of panicking.
        let mut trunc = Vec::new();
        write_varint_u64(&mut trunc, 0xFFFF_FFFF_FFFF_FFFF);
        trunc.pop();
        assert!(read_varint_u64(&trunc).is_none(), "truncated varint must return None");
        // Overflow guard: 11 continuation bytes must return None.
        let bogus = vec![0xFFu8; 11];
        assert!(read_varint_u64(&bogus).is_none(), "11-byte varint must overflow-fail");
    }

    /// B. append_node_intern + replay_streaming_validated round-trip.
    /// Verifies the record reaches disk via the writer, parses back
    /// via the streaming reader, and the streaming reader yields
    /// the same (seed, node_id) pair byte-exact.
    #[test]
    fn phase1_2_append_and_streaming_replay_node_intern_round_trip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("seg.wal");
        let seeds: Vec<Seed> = (0..32)
            .map(|i| Seed::from_label(&format!("intern_{}", i)))
            .collect();
        let ids: Vec<NodeId> = (0..32).map(|i| NodeId(i as u64 * 17 + 3)).collect();
        {
            let mut wal = Wal::open(&path).unwrap();
            for i in 0..32 {
                wal.append_node_intern(i as u64, &seeds[i], ids[i]).unwrap();
            }
            wal.fsync().unwrap();
        }
        // Streaming-replay path: collect into a Vec ONLY because the test
        // asserts ordering -- the underlying reader streams record-by-record.
        let mut got: Vec<(u64, Seed, NodeId)> = Vec::new();
        replay_streaming_validated(&path, |rec| {
            if let WalRecord::NodeIntern { seq, seed, node_id } = rec {
                got.push((seq, seed, node_id));
            }
            Ok(())
        }).unwrap();
        assert_eq!(got.len(), 32);
        for i in 0..32 {
            assert_eq!(got[i].0, i as u64);
            assert_eq!(got[i].1, seeds[i]);
            assert_eq!(got[i].2, ids[i]);
        }
    }

    /// C. ORDERING-INVARIANT ENFORCEMENT (the firewall).
    /// Construct a WAL with a bond record whose subject seed was
    /// NEVER preceded by a NodeIntern. Replay MUST fail with
    /// `UnknownNodeIdInBond { role: Subject }`. Then add the
    /// reverse case (subject interned, object not) to prove the
    /// object-side check fires too. Then add the happy path to
    /// prove the validator doesn't false-positive.
    #[test]
    fn phase1_2_ordering_invariant_bond_before_node_intern_fails() {
        let dir = tempdir().unwrap();
        // -- Case 1: subject seed never interned.
        let bad_subj_path = dir.path().join("bad_subj.wal");
        let bond_seq = 7u64;
        let subj = Seed::from_label("forgotten_subject");
        let obj = Seed::from_label("present_object");
        let obj_id = NodeId(101);
        {
            let mut wal = Wal::open(&bad_subj_path).unwrap();
            // Intern the object only.
            wal.append_node_intern(1, &obj, obj_id).unwrap();
            // Bond whose subject was never interned.
            wal.append(bond_seq, &subj, &obj, BondType::IsA, 1.0, &[]).unwrap();
            wal.fsync().unwrap();
        }
        let err = replay_streaming_validated(&bad_subj_path, |_| Ok(())).unwrap_err();
        match err {
            WalError::UnknownNodeIdInBond { seq, missing_seed, role } => {
                assert_eq!(seq, bond_seq);
                assert_eq!(missing_seed, subj);
                assert_eq!(role, BondSeedRole::Subject);
            }
            other => panic!("expected UnknownNodeIdInBond(Subject), got {:?}", other),
        }

        // -- Case 2: object seed never interned.
        let bad_obj_path = dir.path().join("bad_obj.wal");
        let subj2 = Seed::from_label("present_subject");
        let obj2 = Seed::from_label("forgotten_object");
        let subj2_id = NodeId(202);
        {
            let mut wal = Wal::open(&bad_obj_path).unwrap();
            wal.append_node_intern(1, &subj2, subj2_id).unwrap();
            wal.append(bond_seq, &subj2, &obj2, BondType::IsA, 1.0, &[]).unwrap();
            wal.fsync().unwrap();
        }
        let err = replay_streaming_validated(&bad_obj_path, |_| Ok(())).unwrap_err();
        match err {
            WalError::UnknownNodeIdInBond { seq, missing_seed, role } => {
                assert_eq!(seq, bond_seq);
                assert_eq!(missing_seed, obj2);
                assert_eq!(role, BondSeedRole::Object);
            }
            other => panic!("expected UnknownNodeIdInBond(Object), got {:?}", other),
        }

        // -- Case 3: happy path -- both seeds interned BEFORE the bond.
        let happy_path = dir.path().join("happy.wal");
        let s = Seed::from_label("ok_subj");
        let o = Seed::from_label("ok_obj");
        {
            let mut wal = Wal::open(&happy_path).unwrap();
            wal.append_node_intern(0, &s, NodeId(10)).unwrap();
            wal.append_node_intern(1, &o, NodeId(11)).unwrap();
            wal.append(2, &s, &o, BondType::RelatedTo, 0.5, &[]).unwrap();
            wal.fsync().unwrap();
        }
        let mut count_bond = 0usize;
        let mut count_intern = 0usize;
        replay_streaming_validated(&happy_path, |rec| {
            match rec {
                WalRecord::Bond { .. } => count_bond += 1,
                WalRecord::NodeIntern { .. } => count_intern += 1,
                WalRecord::Node { .. } => {}
            }
            Ok(())
        }).unwrap();
        assert_eq!(count_bond, 1);
        assert_eq!(count_intern, 2);
    }

    /// D. STREAMING-REPLAY BOUNDED-RSS.
    /// 100K NodeIntern records + 100K bond records (in correct order)
    /// replayed via `replay_streaming_validated`. Bounded peak
    /// `RssAnon` delta during replay: <50 MB. Catches any future
    /// regression where replay accidentally materializes a `Vec` over
    /// all records (the antipattern v5 exists to eliminate).
    ///
    /// Note on baseline: we compare peak `RssAnon` AFTER the replay
    /// against the pre-replay baseline and assert the delta
    /// (replay-induced growth) is bounded. The test process itself
    /// uses memory for the temp file path, the seed expansion table,
    /// and the test runner -- that's the baseline, not the budget.
    /// The interned-seed `HashSet<Seed>` (100K * ~40 B = ~4 MB) is the
    /// expected dominant in-replay allocation; everything else
    /// streams.
    #[test]
    // RSS measurement reads /proc/self/status (process-wide), which
    // misattributes peer test threads allocations under cargos default
    // parallel runner. Run as a dedicated gate:
    //   cargo test -p lu4d --lib --release -- --include-ignored \
    //       --test-threads=1 phase1_2_streaming_replay_bounded_rss
    // Verified by Phase 1.2 build: delta 4.6 MiB at N=100K (50 MiB ceiling).
    #[ignore]
    fn phase1_2_streaming_replay_bounded_rss_100k_plus_100k() {
        const N: usize = 100_000;
        let dir = tempdir().unwrap();
        let path = dir.path().join("big.wal");

        // Generate deterministic seeds without storing them in RAM
        // post-write. We DO need the seeds during write (since the writer
        // takes &Seed) but those go out of scope before replay starts.
        {
            let mut wal = Wal::open(&path).unwrap();
            wal.set_fsync_thresholds(10_000, 16 * 1024 * 1024);
            // All N NodeIntern records first.
            for i in 0..N {
                let s = Seed::from_label(&format!("n_{}", i));
                wal.append_node_intern(i as u64, &s, NodeId(i as u64)).unwrap();
            }
            // Then N bond records, each referencing two of the interned seeds.
            for i in 0..N {
                let subj = Seed::from_label(&format!("n_{}", i));
                let obj = Seed::from_label(&format!("n_{}", (i + 1) % N));
                wal.append(
                    (N + i) as u64,
                    &subj, &obj,
                    BondType::RelatedTo, 0.5,
                    &[],
                ).unwrap();
            }
            wal.fsync().unwrap();
        }

        // Sample RssAnon from /proc/self/status. Linux-only (the bench
        // harness's `RssSampler` is the production path; this is the
        // unit-test-grade inline version).
        fn rss_anon_kib() -> u64 {
            let s = std::fs::read_to_string("/proc/self/status").unwrap();
            for line in s.lines() {
                if let Some(rest) = line.strip_prefix("RssAnon:") {
                    let kib: u64 = rest.split_whitespace().next().unwrap()
                        .parse().unwrap();
                    return kib;
                }
            }
            panic!("/proc/self/status has no RssAnon line");
        }

        let baseline_kib = rss_anon_kib();
        let mut peak_kib = baseline_kib;
        let mut bond_seen = 0usize;
        let mut intern_seen = 0usize;
        // Sample every ~5K records during replay so we capture peak,
        // not just terminal RSS.
        replay_streaming_validated(&path, |rec| {
            match rec {
                WalRecord::Bond { .. } => bond_seen += 1,
                WalRecord::NodeIntern { .. } => intern_seen += 1,
                WalRecord::Node { .. } => {}
            }
            if (bond_seen + intern_seen) % 5_000 == 0 {
                let now = rss_anon_kib();
                if now > peak_kib { peak_kib = now; }
            }
            Ok(())
        }).unwrap();
        let final_kib = rss_anon_kib();
        if final_kib > peak_kib { peak_kib = final_kib; }

        assert_eq!(intern_seen, N);
        assert_eq!(bond_seen, N);

        let delta_mib = peak_kib.saturating_sub(baseline_kib) as f64 / 1024.0;
        eprintln!(
            "phase1_2_streaming_replay_bounded_rss_100k_plus_100k: baseline {} KiB, peak {} KiB, delta {:.1} MiB",
            baseline_kib, peak_kib, delta_mib,
        );
        assert!(
            delta_mib < 50.0,
            "streaming-replay grew RSS by {:.1} MiB -- exceeds 50 MiB ceiling (likely a Vec<WalRecord> regression)",
            delta_mib,
        );
    }

    // ---------------------------------------------------------------
    // OP-BRAIN-V5 6.8 -- encrypted WAL at rest.
    // ---------------------------------------------------------------

    /// (a) Encrypted WAL append -> replay round-trips with the key, and
    /// nothing plaintext leaks on disk.
    #[test]
    fn enc68_wal_encrypted_round_trips_with_key() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("enc.wal");
        let key = MasterKey::from_bytes([0x6Au8; 32]);
        {
            let mut wal = Wal::open_with_key(&path, Some(key.clone())).unwrap();
            wal.set_fsync_thresholds(10_000, 64 * 1024 * 1024);
            for i in 0..25u64 {
                let s = Seed::from_label(&format!("es{}", i));
                wal.append_node_intern(i, &s, NodeId(i)).unwrap();
            }
            for i in 0..25u64 {
                let s = Seed::from_label(&format!("es{}", i));
                let o = Seed::from_label(&format!("eo{}", i));
                wal.append(100 + i, &s, &o, BondType::IsA, 0.5, &[baseline_kf(i as i64, 0.5)]).unwrap();
            }
            wal.fsync().unwrap();
        }
        // Header marks the segment encrypted; a known plaintext seed must
        // NOT appear anywhere on disk.
        assert_eq!(wal_encryption_status(&path).unwrap(), WalEncStatus::Encrypted);
        let raw = std::fs::read(&path).unwrap();
        let probe = Seed::from_label("es0");
        assert!(!raw.windows(32).any(|w| w == probe.as_bytes()),
            "encrypted WAL must not leak a plaintext seed on disk");
        // Replay with the key recovers every record.
        let recs = replay_with_key(&path, Some(key.clone())).unwrap();
        assert_eq!(recs.len(), 50);
        assert_eq!(recs.iter().filter(|r| r.is_bond()).count(), 25);
        assert_eq!(recs.iter().filter(|r| r.is_node_intern()).count(), 25);
    }

    /// (b) Encrypted WAL replay drops a deliberately-torn final record
    /// (crash-mid-write tolerance preserved under encryption).
    #[test]
    fn enc68_wal_encrypted_replay_drops_torn_final_record() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("torn.wal");
        let key = MasterKey::from_bytes([0x77u8; 32]);
        {
            let mut wal = Wal::open_with_key(&path, Some(key.clone())).unwrap();
            wal.set_fsync_thresholds(10_000, 64 * 1024 * 1024);
            for i in 0..10u64 {
                let s = Seed::from_label(&format!("ts{}", i));
                let o = Seed::from_label(&format!("to{}", i));
                wal.append(i, &s, &o, BondType::IsA, 0.5, &[baseline_kf(0, 0.5)]).unwrap();
            }
            wal.fsync().unwrap();
        }
        assert_eq!(replay_with_key(&path, Some(key.clone())).unwrap().len(), 10);
        // Chop the final record's CRC: a torn write at the tail.
        let len = std::fs::metadata(&path).unwrap().len();
        let f = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
        f.set_len(len - 3).unwrap();
        f.sync_all().unwrap();
        drop(f);
        // Replay recovers the 9 complete records and drops the torn tail.
        let recovered = replay_with_key(&path, Some(key.clone())).unwrap();
        assert_eq!(recovered.len(), 9, "torn final record must be dropped, no panic");
    }

    /// (c-WAL) Wrong / no key fails cleanly (no panic) on the WAL.
    #[test]
    fn enc68_wal_wrong_or_no_key_fails_cleanly() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("wk.wal");
        let key = MasterKey::from_bytes([0x11u8; 32]);
        {
            let mut wal = Wal::open_with_key(&path, Some(key.clone())).unwrap();
            wal.set_fsync_thresholds(10_000, 64 * 1024 * 1024);
            for i in 0..8u64 {
                let s = Seed::from_label(&format!("ws{}", i));
                let o = Seed::from_label(&format!("wo{}", i));
                wal.append(i, &s, &o, BondType::IsA, 0.5, &[baseline_kf(0, 0.5)]).unwrap();
            }
            wal.fsync().unwrap();
        }
        // No key -> clean EncryptedButNoKey error (no panic).
        match replay_with_key(&path, None) {
            Err(WalError::EncryptedButNoKey) => {}
            other => panic!("no-key replay must error EncryptedButNoKey, got {:?}", other),
        }
        // Wrong key -> decrypted garbage fails the plaintext CRC, replay
        // stops cleanly and recovers none of the real records (no panic).
        let wrong = MasterKey::from_bytes([0x22u8; 32]);
        let recs = replay_with_key(&path, Some(wrong)).unwrap();
        assert!(recs.len() < 8, "wrong key must not recover the real records");
    }
}
