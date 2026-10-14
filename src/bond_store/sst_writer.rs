//! `BondSstWriter` -- streaming on-disk writer for a BondStore SST.
//!
//! OP-BRAIN-V5 Phase 3.1 (Q2 of the design doc; Phase 3.0 audit
//! sub-port 3.1). The writer accepts an append-only stream of
//! `(subject: NodeId, predicate: u8, object: NodeId, &BondPayload)`
//! tuples sorted by the full triple key `(subject, predicate,
//! object)` ascending and emits a single immutable SST file that
//! the matching `BondSstReader` (Phase 3.2, future) will consume.
//!
//! ## Encryption layer -- Q3.0.A resolved against Q6
//!
//! Phase 3.0 audit Q3.0.A (SURPRISE-B) recommended "match v4"
//! which, at v4's `src/sst_writer.rs:763-787`, is encrypt-then-
//! compress at the per-file layer. However, design-doc Q6 locks
//! the v5 archive format as a single `.lbr` tar with OUTER
//! encryption -- all files inside are plaintext, the tar wrapper
//! carries the crypto envelope. Phase 2.4's `NodeSstWriter`
//! verified to be plaintext-on-disk (no `Encrypt`/`master_key`/
//! `nonce` in writer struct or `finish()`; no encryption header
//! in the 32 B wire format).
//!
//! **Resolution: Phase 3.1 ships plaintext-on-disk** matching
//! Phase 2.4 and complying with Q6. The Phase 3.0 audit Q3.0.A
//! recommendation is therefore REVISED -- the audit's "match v4"
//! advice was anchored in the v4 per-file model that Q6
//! intentionally retired. Surfaced as a `§7` design-doc entry
//! candidate (see Phase 3.0 audit § 15 Q3.0.A re-spec).
//!
//! This decision keeps both substrate stores symmetric:
//! `node_store/*.nsst` and `bond_store/*.bsst` are both plaintext
//! inside the encrypted `.lbr` tar.
//!
//! ## Wire format
//!
//! ```text
//! Header (32 B fixed):
//!   bytes 0..4    magic              u32 LE  = 0x42535354 ("BSST")
//!   bytes 4..6    format_version     u16 LE  = BSST_FORMAT_VERSION (=2)
//!   bytes 6..7    flags              u8      bit0 = FLAG_ENCRYPTED
//!   bytes 7..8    reserved           u8      = 0
//!   bytes 8..12   block_count        u32 LE  = number of blocks
//!   bytes 12..20  footer_offset      u64 LE  = absolute file offset of footer start
//!   bytes 20..28  file_uid           u64 LE  = caller-supplied stable identifier
//!   bytes 28..44  encryption_iv      [u8;16] = per-SST random IV (zeros if plaintext)
//!   bytes 44..48  header_crc32       u32 LE  = CRC-32/IEEE over bytes [0..44]
//!
//! Blocks (variable size, repeated `block_count` times, laid out
//! contiguously starting at file offset 32):
//!   block_header (17 B):
//!     bytes 0..4    entry_count        u32 LE
//!     bytes 4..8    uncompressed_size  u32 LE  (byte length of raw block body)
//!     bytes 8..12   compressed_size    u32 LE  (byte length of on-disk body)
//!     bytes 12      compression_kind   u8       (0=raw, 1=zstd)
//!     bytes 13..17  block_crc32        u32 LE  (CRC over the compressed body)
//!   block_body (compressed_size bytes):
//!     bytes per the compression_kind. After decompression: a
//!     sequence of `entry_count` entries each shaped:
//!       subject           u64 LE
//!       predicate         u8
//!       object            u64 LE
//!       payload_size      u32 LE       length of bincode payload
//!       payload_bytes     payload_size B bincode(BondPayload)
//!     Total per entry (uncompressed): 21 B framing + payload bytes.
//!     Entries inside a block are sorted by (subject, predicate,
//!     object) ascending; the stream as a whole is sorted by the
//!     same key (blocks own non-overlapping key ranges).
//!
//! Footer (variable size, written AFTER all blocks; absolute file
//! offset captured in the header's `footer_offset`):
//!   block_index (46 B per entry, `block_count` entries):
//!     bytes 0..8    first_subject     u64 LE
//!     bytes 8..9    first_predicate   u8
//!     bytes 9..17   first_object      u64 LE
//!     bytes 17..25  last_subject      u64 LE
//!     bytes 25..26  last_predicate    u8
//!     bytes 26..34  last_object       u64 LE
//!     bytes 34..42  block_offset      u64 LE
//!     bytes 42..46  block_size        u32 LE  (block_header + block_body)
//!   block_index_byte_len u32 LE  = block_count * 46
//!   footer_crc32         u32 LE  = CRC over block_index bytes ONLY
//!
//! Trailer (12 B, the final 12 bytes of the file -- enables reverse
//! seek validation when the footer offset is suspect):
//!   bytes 0..4    magic_tail         u32 LE  = 0x42535354 ("BSST")
//!   bytes 4..12   footer_offset_tail u64 LE  = mirrors header.footer_offset
//! ```
//!
//! Adjustments from the Phase 3.1 brief: the brief sketched a 40 B
//! footer entry "+ pad u8". The actual shipped layout is 46 B with
//! NO pad -- the brief's 40 B/entry count omitted the duplicated
//! triple bytes (each footer entry carries both first AND last
//! keys, not just one of them). Documented honestly here so the
//! Phase 3.2 reader stays in lockstep.
//!
//! ## BondPayload serialization
//!
//! `bincode` 1.x is the codec choice for the payload portion of
//! each entry, mirroring Phase 2.4's choice and the v4 per-block
//! keyframe codec at `src/sst.rs::encode_keyframe`. Rationale:
//!
//! - `TemporalKeyframe` already derives `Serialize` / `Deserialize`
//!   in `src/bond.rs:81` -- no new derive surface needed. The
//!   `Vec<TemporalKeyframe>` therefore bincode-encodes as a u64
//!   length prefix + N * (8 B timestamp + 4 B weight + KeyframeSource
//!   bincode); no rkyv archive required.
//! - The 24 B key (subject u64 + predicate u8 + object u64, with
//!   the predicate inlined between the two NodeIds for natural
//!   sort-order byte comparison) is NOT bincode'd. It is written
//!   verbatim into the entry header so the reader can decode the
//!   key without invoking bincode (cheap binary search inside a
//!   block).
//! - `weight: u8`, `confidence: u8`, `seq: u64` go through bincode
//!   along with the keyframe Vec -- single roundtrip per entry.
//!
//! Per Phase 3.0 audit Q3.0.D, this is the hybrid "custom packed
//! bytes for fixed-width key + bincode for variable-width
//! payload" the brief asked for. The audit's S6.2 proposed a
//! deeper packing scheme (predicate + weight + confidence packed
//! into a single u64) but for Phase 3.1 the simpler entry header
//! is the right shape -- the savings (~3 B/entry) are dominated by
//! the keyframe Vec and the zstd block compression downstream.
//! Re-evaluate if SST size becomes the bottleneck.
//!
//! ## TOMBSTONE convention
//!
//! Per Phase 3.0 audit S9.2 (Q3.0.G), `BondPayload.confidence ==
//! 0xFF` is the TOMBSTONE marker. This matches the v4 convention
//! at `src/sst.rs:187` (`TOMBSTONE_CONFIDENCE = 0xFF`) verbatim,
//! so the Phase 3.4 bond compactor's epoch-barrier GC can mirror
//! the v4 `compaction.rs:432-444` logic AND the Phase 2.7
//! `node_store/compaction.rs` logic without translation. The byte
//! IS the marker -- no separate bool field, no separate tombstone
//! row type. Test I verifies the byte round-trips intact.
//!
//! ## Compression observation
//!
//! Per Phase 3.0 audit S3.3 and S8: per-bond payload is ~57 B
//! uncompressed (24 B key + 3 B keyframe range + ~30 B keyframe
//! bincode), targeting ~13-15 B post-zstd at production scale.
//! Synthetic test payloads in the test suite below are smaller and
//! more uniform -- compression ratios observed in tests are NOT
//! representative of real-corpus rates. Phase 3.1 reports the
//! observed ratio diagnostically; production rates will be lower
//! (better) than test rates because real keyframe sources
//! (KeyframeSource enum, 10 variants) have higher entropy than the
//! deterministic test fixtures.
//!
//! ## Streaming discipline
//!
//! The writer holds AT MOST one in-progress block in memory --
//! `current_block_buf: Vec<u8>` plus a few small counters. A flush
//! happens when `current_block_entries == target_entries_per_block`
//! or when [`BondSstWriter::finish`] is called. The footer's
//! `block_index` accumulates one `BlockIndexEntry` per flushed
//! block (46 B each); at 1 M blocks this is 46 MB resident --
//! proportional only to block count, not entry count. The reusable
//! `zstd_buf: Vec<u8>` is cleared and refilled per block.
//!
//! Append requires strictly-increasing `(subject, predicate,
//! object)` triples. Out-of-order or duplicate calls error with
//! [`BondSstError::OutOfOrderAppend`] (Tests D-G).
//!
//! ## Atomicity
//!
//! `create()` opens `<path>.tmp.<file_uid>`; `finish()` fsyncs and
//! renames into `<path>`. Either `<path>` does not exist or it is
//! a complete valid SST (header CRC + per-block CRCs + footer CRC
//! all valid).

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crate::node_intern::NodeId;
use crate::encrypt::{xor_in_place, MasterKey};
use crate::sst::derive_block_keystream_index;
use rand::RngCore;

// ----------------------------------------------------------------------
// Wire-format constants.
// ----------------------------------------------------------------------

/// Magic bytes at the start of every BondSST file ("BSST").
pub const BSST_MAGIC: u32 = 0x42535354;

/// Current wire-format version. Bumped on any breaking change.
/// Reader rejects unknown versions with
/// [`BondSstError::FormatVersionUnsupported`].
/// v2 (OP-BRAIN-V5 6.5) widened the header 32 -> 48 B to carry a
/// per-SST encryption flag + IV (Option B, v4-mirrored keystream).
pub const BSST_FORMAT_VERSION: u16 = 2;

/// Header `flags` bit 0: block bodies are XOR-keystream encrypted
/// (encrypt-then-compress at write; decompress-then-decrypt at read),
/// mirroring the v4 SST scheme byte-for-byte.
pub const FLAG_ENCRYPTED: u8 = 0x01;

/// Fixed header size in bytes.
pub(crate) const HEADER_BYTES: usize = 48;

/// Per-block header size in bytes.
pub(crate) const BLOCK_HEADER_BYTES: usize = 4 + 4 + 4 + 1 + 4;

/// Per-entry framing size in bytes (subject + predicate + object + payload_size).
pub(crate) const ENTRY_HEADER_BYTES: usize = 8 + 1 + 8 + 4;

/// Per-index-entry size in bytes (first_triple 17 + last_triple 17 + offset 8 + size 4).
pub(crate) const INDEX_ENTRY_BYTES: usize = 17 + 17 + 8 + 4;

/// Trailer size in bytes (magic_tail + footer_offset_tail).
pub(crate) const TRAILER_BYTES: usize = 4 + 8;

/// Default entries per block. Per Phase 3.0 audit S8, the bond
/// store wants larger blocks than the node store because bond
/// payload is ~57 B vs ~1800 B/node. Audit recommends 4096 to hit
/// the ~256 KB raw block target. Phase 3.1 ships 1000 as the
/// brief's default -- conservative starting point that the Phase
/// 3.3 flush worker can override via opts. Revisit after benches.
pub const DEFAULT_TARGET_ENTRIES_PER_BLOCK: usize = 1000;

/// Maximum entries per block. Bounds the in-memory write buffer.
pub const MAX_TARGET_ENTRIES_PER_BLOCK: usize = 65536;

/// BufWriter capacity in bytes.
const BUFWRITER_CAPACITY: usize = 64 * 1024;

// ----------------------------------------------------------------------
// Public types.
// ----------------------------------------------------------------------

/// Compression scheme for block bodies. The on-disk discriminator
/// is the `u8` returned by [`Self::as_u8`].
///
/// Mirrors `node_store::CompressionKind` byte-for-byte so the two
/// stores' SSTs can share offline tooling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionKind {
    /// No compression -- block body is written verbatim.
    None,
    /// zstd compression at the configured level.
    Zstd { level: i32 },
}

impl CompressionKind {
    /// The on-disk u8 discriminator.
    pub fn as_u8(self) -> u8 {
        match self {
            CompressionKind::None => 0,
            CompressionKind::Zstd { .. } => 1,
        }
    }

    /// Construct from the on-disk u8 discriminator. The encode-side
    /// level is not round-tripped on disk; decompression is
    /// parameter free, so the reader-side `Zstd` variant carries
    /// `level: 0`.
    pub(crate) fn from_u8(b: u8) -> Option<CompressionKind> {
        match b {
            0 => Some(CompressionKind::None),
            1 => Some(CompressionKind::Zstd { level: 0 }),
            _ => None,
        }
    }
}

impl Default for CompressionKind {
    fn default() -> Self {
        CompressionKind::Zstd { level: 1 }
    }
}

/// Caller-supplied writer options.
#[derive(Debug, Clone)]
pub struct BondSstWriteOpts {
    /// Number of entries to accumulate before flushing a block.
    /// Clamped to `[1, MAX_TARGET_ENTRIES_PER_BLOCK]`.
    pub target_entries_per_block: usize,
    /// Compression scheme for block bodies.
    pub compression: CompressionKind,
    /// Stable identifier for this SST. Written into the header and
    /// returned by the matching reader's `file_uid()` accessor
    /// (Phase 3.2).
    pub file_uid: u64,
}

impl Default for BondSstWriteOpts {
    fn default() -> Self {
        BondSstWriteOpts {
            target_entries_per_block: DEFAULT_TARGET_ENTRIES_PER_BLOCK,
            compression: CompressionKind::default(),
            file_uid: 0,
        }
    }
}

/// Statistics returned by [`BondSstWriter::finish`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BondSstWriteStats {
    pub entries_written: u64,
    pub blocks_written: u32,
    pub uncompressed_bytes: u64,
    pub compressed_bytes: u64,
    pub footer_bytes: u64,
    pub file_size_bytes: u64,
    /// First triple written, if any.
    pub first_key: Option<(NodeId, u8, NodeId)>,
    /// Last triple written, if any.
    pub last_key: Option<(NodeId, u8, NodeId)>,
    /// Best-effort observation of peak resident-set-size in KB
    /// during the write, sampled from `/proc/self/status` (Linux).
    /// `None` on non-Linux platforms or when reading the file
    /// fails. Diagnostic only.
    pub peak_writer_rss_kb: Option<u64>,
}

/// The variable-width per-bond payload. Sits behind the 21 B entry
/// header and is bincode-serialised.
///
/// Per Phase 3.0 audit S6.3 (Q3.0.D): `weight` and `confidence`
/// are u8-quantised, `keyframes` reuses v4's `TemporalKeyframe`
/// (already `Serialize` + `Deserialize` at `src/bond.rs:81`),
/// `seq` is the monotonic SST id used by dedup / compaction.
///
/// `confidence == 0xFF` is the TOMBSTONE marker -- see module
/// doc-comment.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BondPayload {
    /// u8-quantised weight, mapping 0..=255 to 0.0..=1.0. Matches
    /// v4 `CompactBondRaw.weight`.
    pub weight: u8,
    /// u8 Confidence enum value; `0xFF` is the TOMBSTONE marker
    /// (matches v4 `TOMBSTONE_CONFIDENCE` at `src/sst.rs:187`).
    pub confidence: u8,
    /// Per-bond keyframe history. v4 packs these into a per-block
    /// pool; v5 carries them inline per bond (the pool was a
    /// per-block optimisation -- with brain-wide NodeIds the
    /// pooled offset+count indirection no longer pays for itself).
    pub keyframes: Vec<crate::bond::TemporalKeyframe>,
    /// Monotonic sequence number used by dedup / compaction
    /// (higher seq wins). Same semantics as Phase 2.4 NodeEntry.seq
    /// and v4 MemTable seq.
    pub seq: u64,
}

/// Errors surfaced by [`BondSstWriter`] and the matching reader.
#[derive(Debug)]
pub enum BondSstError {
    /// `std::io::Error` propagated from open / read / write / fsync
    /// / rename.
    Io(std::io::Error),
    /// Header magic was not [`BSST_MAGIC`].
    MagicMismatch { observed: u32 },
    /// Header `format_version` is not [`BSST_FORMAT_VERSION`].
    FormatVersionUnsupported { observed: u16 },
    /// CRC32 mismatch on the header, a block body, or the footer.
    CrcMismatch { scope: CrcScope, expected: u32, observed: u32 },
    /// Caller passed a triple not strictly greater than the
    /// previous append. Covers the three sub-dimensions: subject,
    /// predicate, object. Same-triple repeats are ALSO rejected --
    /// dedup is compaction's job, not the writer's.
    OutOfOrderAppend {
        previous: (NodeId, u8, NodeId),
        attempted: (NodeId, u8, NodeId),
    },
    /// Block body's on-disk `compression_kind` byte was outside the
    /// known range. Reader-side only.
    UnknownCompressionKind { observed: u8 },
    /// bincode failed to (de)serialise a `BondPayload`. Wraps the
    /// codec error as a `String` (bincode 1.x's error type is not
    /// always Send + Sync).
    Bincode(String),
    /// zstd encode / decode failed.
    Zstd(String),
    /// `target_entries_per_block` was 0 or > [`MAX_TARGET_ENTRIES_PER_BLOCK`].
    InvalidOpts(String),
    /// Reader-side: structural value out of bounds (footer offset
    /// past EOF, block size exceeds file size, etc.). Carries a
    /// short human description.
    Format(String),
    /// Reader-side: hit EOF before completing a structural read.
    UnexpectedEof,
    /// Reader-side: the SST header marks block bodies encrypted but
    /// the reader has no key to decrypt them.
    EncryptedButNoKey,
}

/// Scope of a CRC32 mismatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrcScope {
    Header,
    Block { block_id: u32 },
    Footer,
}

impl std::fmt::Display for BondSstError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BondSstError::Io(e) => write!(f, "BondSST IO: {}", e),
            BondSstError::MagicMismatch { observed } => write!(
                f,
                "BondSST magic mismatch: expected {:#x}, observed {:#x}",
                BSST_MAGIC, observed
            ),
            BondSstError::FormatVersionUnsupported { observed } => write!(
                f,
                "BondSST format version unsupported: expected {}, observed {}",
                BSST_FORMAT_VERSION, observed
            ),
            BondSstError::CrcMismatch { scope, expected, observed } => write!(
                f,
                "BondSST CRC32 mismatch ({:?}): expected {:#010x}, observed {:#010x}",
                scope, expected, observed
            ),
            BondSstError::OutOfOrderAppend { previous, attempted } => write!(
                f,
                "BondSST out-of-order append: previous=({}, {}, {}), attempted=({}, {}, {})",
                previous.0.0, previous.1, previous.2.0,
                attempted.0.0, attempted.1, attempted.2.0,
            ),
            BondSstError::UnknownCompressionKind { observed } => write!(
                f, "BondSST unknown compression_kind byte: {}", observed
            ),
            BondSstError::Bincode(s) => write!(f, "BondSST bincode error: {}", s),
            BondSstError::Zstd(s) => write!(f, "BondSST zstd error: {}", s),
            BondSstError::InvalidOpts(s) => write!(f, "BondSST invalid opts: {}", s),
            BondSstError::Format(s) => write!(f, "BondSST format: {}", s),
            BondSstError::EncryptedButNoKey => write!(
                f,
                "BondSST block bodies are encrypted but the reader has no key"
            ),
            BondSstError::UnexpectedEof => write!(f, "BondSST unexpected EOF"),
        }
    }
}

impl std::error::Error for BondSstError {}

impl From<std::io::Error> for BondSstError {
    fn from(e: std::io::Error) -> Self {
        BondSstError::Io(e)
    }
}

// ----------------------------------------------------------------------
// Writer.
// ----------------------------------------------------------------------

/// Streaming BondSST writer. See module doc-comment.
pub struct BondSstWriter {
    writer: Option<BufWriter<File>>,
    path: PathBuf,
    tmp_path: PathBuf,
    opts: BondSstWriteOpts,

    next_block_offset: u64,
    last_key: Option<(NodeId, u8, NodeId)>,
    first_key: Option<(NodeId, u8, NodeId)>,

    current_block_buf: Vec<u8>,
    current_block_entries: u32,
    current_block_first_key: Option<(NodeId, u8, NodeId)>,
    current_block_last_key: Option<(NodeId, u8, NodeId)>,

    block_index: Vec<BlockIndexEntry>,

    entries_written: u64,
    uncompressed_bytes: u64,
    compressed_bytes: u64,

    zstd_buf: Vec<u8>,

    /// OP-BRAIN-V5 6.5: per-SST encryption key. `None` = plaintext.
    key: Option<MasterKey>,
    /// Per-SST random IV mixed into the keystream index (v4-mirrored).
    /// All-zero when `key` is `None`.
    iv: [u8; 16],
}

/// One block_index entry. Mirrored on disk by the writer footer
/// and parsed by the Phase 3.2 reader.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BlockIndexEntry {
    pub first_subject: NodeId,
    pub first_predicate: u8,
    pub first_object: NodeId,
    pub last_subject: NodeId,
    pub last_predicate: u8,
    pub last_object: NodeId,
    pub block_offset: u64,
    pub block_size: u32,
}

impl BondSstWriter {
    /// Create a new BondSST writer targeting `path`. Writes go to a
    /// temporary file `<path>.tmp.<file_uid>` until `finish()`
    /// renames atomically.
    pub fn create(path: &Path, opts: BondSstWriteOpts) -> Result<Self, BondSstError> {
        Self::create_with_key(path, opts, None)
    }

    /// Create a writer that encrypts each block body under `key`
    /// (OP-BRAIN-V5 6.5, Option B). `key = None` writes byte-identical
    /// plaintext to the pre-6.5 writer. When `Some`, a fresh random
    /// 16-byte IV is generated and stamped into the header and each
    /// block body is XOR-encrypted before compression.
    pub fn create_with_key(
        path: &Path,
        opts: BondSstWriteOpts,
        key: Option<MasterKey>,
    ) -> Result<Self, BondSstError> {
        if opts.target_entries_per_block == 0
            || opts.target_entries_per_block > MAX_TARGET_ENTRIES_PER_BLOCK
        {
            return Err(BondSstError::InvalidOpts(format!(
                "target_entries_per_block must be in 1..={}; got {}",
                MAX_TARGET_ENTRIES_PER_BLOCK, opts.target_entries_per_block
            )));
        }

        let tmp_path = tmp_path_for(path, opts.file_uid);
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&tmp_path)?;
        let mut writer = BufWriter::with_capacity(BUFWRITER_CAPACITY, file);

        // Reserve the header slot; rewritten by `finish()` once
        // block_count and footer_offset are known.
        let zero_header = [0u8; HEADER_BYTES];
        writer.write_all(&zero_header)?;

        Ok(BondSstWriter {
            writer: Some(writer),
            path: path.to_path_buf(),
            tmp_path,
            opts,
            next_block_offset: HEADER_BYTES as u64,
            last_key: None,
            first_key: None,
            current_block_buf: Vec::with_capacity(4096),
            current_block_entries: 0,
            current_block_first_key: None,
            current_block_last_key: None,
            block_index: Vec::new(),
            entries_written: 0,
            uncompressed_bytes: 0,
            compressed_bytes: 0,
            zstd_buf: Vec::new(),
            iv: {
                let mut iv = [0u8; 16];
                if key.is_some() {
                    rand::thread_rng().fill_bytes(&mut iv);
                }
                iv
            },
            key,
        })
    }

    /// Append a single bond row. The triple `(subject, predicate,
    /// object)` must be STRICTLY greater than the previous append
    /// in lexicographic order. Same-triple repeats are rejected --
    /// dedup is compaction territory, not the writer's.
    pub fn append(
        &mut self,
        subject: NodeId,
        predicate: u8,
        object: NodeId,
        payload: &BondPayload,
    ) -> Result<(), BondSstError> {
        let key = (subject, predicate, object);
        if let Some(prev) = self.last_key {
            if !triple_lt(prev, key) {
                debug_assert!(
                    false,
                    "BondSstWriter::append called with non-monotonic key: {:?} -> {:?}",
                    (prev.0.0, prev.1, prev.2.0),
                    (key.0.0, key.1, key.2.0),
                );
                return Err(BondSstError::OutOfOrderAppend {
                    previous: prev,
                    attempted: key,
                });
            }
        }

        let payload_bytes = bincode::serialize(payload)
            .map_err(|e| BondSstError::Bincode(format!("serialize BondPayload: {}", e)))?;
        let payload_size = u32::try_from(payload_bytes.len()).map_err(|_| {
            BondSstError::Format(format!(
                "BondPayload serialised size {} exceeds u32::MAX",
                payload_bytes.len()
            ))
        })?;

        self.current_block_buf.extend_from_slice(&subject.0.to_le_bytes());
        self.current_block_buf.push(predicate);
        self.current_block_buf.extend_from_slice(&object.0.to_le_bytes());
        self.current_block_buf.extend_from_slice(&payload_size.to_le_bytes());
        self.current_block_buf.extend_from_slice(&payload_bytes);

        if self.current_block_first_key.is_none() {
            self.current_block_first_key = Some(key);
        }
        self.current_block_last_key = Some(key);
        self.current_block_entries += 1;
        if self.first_key.is_none() {
            self.first_key = Some(key);
        }
        self.last_key = Some(key);
        self.entries_written += 1;

        if self.current_block_entries as usize >= self.opts.target_entries_per_block {
            self.flush_block()?;
        }
        Ok(())
    }

    /// Finalise the SST. Consumes self; on success the temporary
    /// file has been renamed into the final path.
    pub fn finish(mut self) -> Result<BondSstWriteStats, BondSstError> {
        if self.current_block_entries > 0 {
            self.flush_block()?;
        }

        let footer_offset = self.next_block_offset;

        let block_count = u32::try_from(self.block_index.len()).map_err(|_| {
            BondSstError::Format(format!(
                "block_count {} exceeds u32::MAX",
                self.block_index.len()
            ))
        })?;
        let index_byte_len = (self.block_index.len() * INDEX_ENTRY_BYTES) as u32;

        let mut footer_payload: Vec<u8> = Vec::with_capacity(
            self.block_index.len() * INDEX_ENTRY_BYTES + 4 + 4,
        );
        for ent in &self.block_index {
            footer_payload.extend_from_slice(&ent.first_subject.0.to_le_bytes());
            footer_payload.push(ent.first_predicate);
            footer_payload.extend_from_slice(&ent.first_object.0.to_le_bytes());
            footer_payload.extend_from_slice(&ent.last_subject.0.to_le_bytes());
            footer_payload.push(ent.last_predicate);
            footer_payload.extend_from_slice(&ent.last_object.0.to_le_bytes());
            footer_payload.extend_from_slice(&ent.block_offset.to_le_bytes());
            footer_payload.extend_from_slice(&ent.block_size.to_le_bytes());
        }
        let footer_crc = crc32fast::hash(&footer_payload);
        footer_payload.extend_from_slice(&index_byte_len.to_le_bytes());
        footer_payload.extend_from_slice(&footer_crc.to_le_bytes());
        let footer_bytes = footer_payload.len() as u64;

        let mut trailer = [0u8; TRAILER_BYTES];
        trailer[0..4].copy_from_slice(&BSST_MAGIC.to_le_bytes());
        trailer[4..12].copy_from_slice(&footer_offset.to_le_bytes());

        {
            let writer = self
                .writer
                .as_mut()
                .expect("writer is Some until finish() consumes self");
            writer.write_all(&footer_payload)?;
            writer.write_all(&trailer)?;
        }

        let file_size = footer_offset + footer_bytes + TRAILER_BYTES as u64;

        let flags = if self.key.is_some() { FLAG_ENCRYPTED } else { 0u8 };
        let header_bytes =
            build_header(block_count, footer_offset, self.opts.file_uid, flags, &self.iv);
        {
            let writer = self.writer.as_mut().expect("writer is Some");
            writer.flush()?;
            writer.seek(SeekFrom::Start(0))?;
            writer.write_all(&header_bytes)?;
            writer.flush()?;
            writer.get_ref().sync_all()?;
        }

        self.writer.take();

        std::fs::rename(&self.tmp_path, &self.path)?;

        let peak_rss_kb = read_self_rss_peak_kb();

        Ok(BondSstWriteStats {
            entries_written: self.entries_written,
            blocks_written: block_count,
            uncompressed_bytes: self.uncompressed_bytes,
            compressed_bytes: self.compressed_bytes,
            footer_bytes,
            file_size_bytes: file_size,
            first_key: self.first_key,
            last_key: self.last_key,
            peak_writer_rss_kb: peak_rss_kb,
        })
    }

    fn flush_block(&mut self) -> Result<(), BondSstError> {
        debug_assert!(self.current_block_entries > 0);

        let uncompressed_size = u32::try_from(self.current_block_buf.len()).map_err(|_| {
            BondSstError::Format(format!(
                "block uncompressed size {} exceeds u32::MAX",
                self.current_block_buf.len()
            ))
        })?;

        // OP-BRAIN-V5 6.5: encrypt the raw block body in place BEFORE
        // compression, mirroring v4 `sst_writer.rs::flush_block`. XOR-
        // keystream is byte-position-equivariant, so encrypt-then-
        // compress (write) pairs with decompress-then-decrypt (read).
        if let Some(k) = &self.key {
            let block_id = self.block_index.len() as u32;
            let ksi = derive_block_keystream_index(self.opts.file_uid, block_id, &self.iv);
            xor_in_place(k, ksi, &mut self.current_block_buf);
        }

        let (compressed_bytes_slice, compression_kind_byte): (&[u8], u8) = match self.opts.compression
        {
            CompressionKind::None => (self.current_block_buf.as_slice(), 0u8),
            CompressionKind::Zstd { level } => {
                self.zstd_buf.clear();
                let mut encoder = zstd::Encoder::new(&mut self.zstd_buf, level)
                    .map_err(|e| BondSstError::Zstd(format!("encoder init: {}", e)))?;
                encoder
                    .write_all(&self.current_block_buf)
                    .map_err(|e| BondSstError::Zstd(format!("encode write: {}", e)))?;
                encoder
                    .finish()
                    .map_err(|e| BondSstError::Zstd(format!("encoder finish: {}", e)))?;
                (self.zstd_buf.as_slice(), 1u8)
            }
        };

        let compressed_size = u32::try_from(compressed_bytes_slice.len()).map_err(|_| {
            BondSstError::Format(format!(
                "block compressed size {} exceeds u32::MAX",
                compressed_bytes_slice.len()
            ))
        })?;
        let block_crc = crc32fast::hash(compressed_bytes_slice);

        let mut block_header = [0u8; BLOCK_HEADER_BYTES];
        block_header[0..4].copy_from_slice(&self.current_block_entries.to_le_bytes());
        block_header[4..8].copy_from_slice(&uncompressed_size.to_le_bytes());
        block_header[8..12].copy_from_slice(&compressed_size.to_le_bytes());
        block_header[12] = compression_kind_byte;
        block_header[13..17].copy_from_slice(&block_crc.to_le_bytes());

        {
            let writer = self.writer.as_mut().expect("writer is Some inside flush_block");
            writer.write_all(&block_header)?;
            writer.write_all(compressed_bytes_slice)?;
        }

        let block_size_on_disk = BLOCK_HEADER_BYTES as u64 + compressed_size as u64;
        let block_size_u32 = u32::try_from(block_size_on_disk).map_err(|_| {
            BondSstError::Format(format!(
                "on-disk block size {} exceeds u32::MAX",
                block_size_on_disk
            ))
        })?;

        let first = self
            .current_block_first_key
            .expect("first_key Some when entries > 0");
        let last = self
            .current_block_last_key
            .expect("last_key Some when entries > 0");

        self.block_index.push(BlockIndexEntry {
            first_subject: first.0,
            first_predicate: first.1,
            first_object: first.2,
            last_subject: last.0,
            last_predicate: last.1,
            last_object: last.2,
            block_offset: self.next_block_offset,
            block_size: block_size_u32,
        });

        self.next_block_offset += block_size_on_disk;
        self.uncompressed_bytes += uncompressed_size as u64;
        self.compressed_bytes += compressed_size as u64;

        self.current_block_buf.clear();
        self.current_block_entries = 0;
        self.current_block_first_key = None;
        self.current_block_last_key = None;

        Ok(())
    }
}

impl Drop for BondSstWriter {
    fn drop(&mut self) {
        // If dropped without finish(), the temporary file may still
        // be on disk. Best-effort cleanup; drop must not panic.
        if self.writer.is_some() {
            self.writer.take();
            let _ = std::fs::remove_file(&self.tmp_path);
        }
    }
}

// ----------------------------------------------------------------------
// Internal helpers (writer side).
// ----------------------------------------------------------------------

/// Lexicographic strict-less-than on the (subject, predicate,
/// object) triple. Wire-format ordering is little-endian byte
/// order on subject/object u64s, but the SORT order is numeric
/// ascending on each field -- u64 comparisons match memcmp for
/// big-endian only. We sort by numeric u64 value (`a.0 < b.0`)
/// which is what callers expect.
fn triple_lt(a: (NodeId, u8, NodeId), b: (NodeId, u8, NodeId)) -> bool {
    if a.0.0 != b.0.0 {
        return a.0.0 < b.0.0;
    }
    if a.1 != b.1 {
        return a.1 < b.1;
    }
    a.2.0 < b.2.0
}

/// Construct the 32-byte header. The `header_crc32` is computed
/// over bytes 0..28 (the 28 bytes preceding the crc field itself).
pub(crate) fn build_header(
    block_count: u32,
    footer_offset: u64,
    file_uid: u64,
    flags: u8,
    iv: &[u8; 16],
) -> [u8; HEADER_BYTES] {
    let mut buf = [0u8; HEADER_BYTES];
    buf[0..4].copy_from_slice(&BSST_MAGIC.to_le_bytes());
    buf[4..6].copy_from_slice(&BSST_FORMAT_VERSION.to_le_bytes());
    buf[6] = flags;
    // byte 7 reserved = 0
    buf[8..12].copy_from_slice(&block_count.to_le_bytes());
    buf[12..20].copy_from_slice(&footer_offset.to_le_bytes());
    buf[20..28].copy_from_slice(&file_uid.to_le_bytes());
    buf[28..44].copy_from_slice(iv);
    let crc = crc32fast::hash(&buf[..44]);
    buf[44..48].copy_from_slice(&crc.to_le_bytes());
    buf
}

/// Resolve the temporary path for a writer targeting `path` with
/// the given `file_uid`. Lives in the same parent directory so the
/// final `rename` is intra-filesystem (atomic per POSIX).
fn tmp_path_for(path: &Path, file_uid: u64) -> PathBuf {
    let parent = path.parent().unwrap_or(Path::new("."));
    let file_name = match path.file_name() {
        Some(n) => n.to_string_lossy().into_owned(),
        None => "bondsst".to_string(),
    };
    parent.join(format!("{}.tmp.{}", file_name, file_uid))
}

/// Best-effort read of `/proc/self/status` to extract `VmHWM`
/// (peak resident-set size since process start, in kilobytes).
/// Returns `None` on non-Linux or any I/O / parse failure --
/// metadata only.
pub(crate) fn read_self_rss_peak_kb() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            let mut it = rest.split_ascii_whitespace();
            if let Some(num) = it.next() {
                if let Ok(v) = num.parse::<u64>() {
                    return Some(v);
                }
            }
        }
    }
    None
}

// ----------------------------------------------------------------------
// Tests.
// ----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bond::{KeyframeSource, TemporalKeyframe};
    use tempfile::tempdir;

    /// Build a synthetic small BondPayload for the bulk tests.
    fn mk_payload(seq: u64) -> BondPayload {
        BondPayload {
            weight: 200,
            confidence: 64,
            keyframes: vec![TemporalKeyframe {
                timestamp: 1_700_000_000,
                weight: 0.75,
                source: KeyframeSource::Baseline,
            }],
            seq,
        }
    }

    /// Read the 32 B header off a finished file and validate magic,
    /// version, and CRC. Returns (block_count, footer_offset, file_uid).
    fn read_header(path: &Path) -> (u32, u64, u64) {
        let bytes = std::fs::read(path).expect("read SST file");
        assert!(bytes.len() >= HEADER_BYTES, "file too small for header");
        let magic = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        assert_eq!(magic, BSST_MAGIC, "magic mismatch");
        let version = u16::from_le_bytes([bytes[4], bytes[5]]);
        assert_eq!(version, BSST_FORMAT_VERSION);
        let block_count = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        let footer_offset = u64::from_le_bytes([
            bytes[12], bytes[13], bytes[14], bytes[15],
            bytes[16], bytes[17], bytes[18], bytes[19],
        ]);
        let file_uid = u64::from_le_bytes([
            bytes[20], bytes[21], bytes[22], bytes[23],
            bytes[24], bytes[25], bytes[26], bytes[27],
        ]);
        let header_crc_disk = u32::from_le_bytes([bytes[44], bytes[45], bytes[46], bytes[47]]);
        let expected_crc = crc32fast::hash(&bytes[..44]);
        assert_eq!(header_crc_disk, expected_crc, "header CRC must verify");
        (block_count, footer_offset, file_uid)
    }

    /// Decode a single block at `block_offset` from a finished file.
    /// Returns the uncompressed entry bytes. Test-only helper used
    /// by tests I and J for byte-level verification.
    fn decode_block(path: &Path, block_offset: u64) -> Vec<u8> {
        let bytes = std::fs::read(path).unwrap();
        let off = block_offset as usize;
        let entry_count = u32::from_le_bytes([
            bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3],
        ]);
        let uncompressed_size = u32::from_le_bytes([
            bytes[off + 4], bytes[off + 5], bytes[off + 6], bytes[off + 7],
        ]);
        let compressed_size = u32::from_le_bytes([
            bytes[off + 8], bytes[off + 9], bytes[off + 10], bytes[off + 11],
        ]);
        let compression_kind = bytes[off + 12];
        let _block_crc = u32::from_le_bytes([
            bytes[off + 13], bytes[off + 14], bytes[off + 15], bytes[off + 16],
        ]);
        let body_start = off + BLOCK_HEADER_BYTES;
        let body_end = body_start + compressed_size as usize;
        let body = &bytes[body_start..body_end];

        let uncompressed = match compression_kind {
            0 => body.to_vec(),
            1 => zstd::decode_all(body).expect("zstd decode"),
            other => panic!("unknown compression_kind {}", other),
        };
        assert_eq!(uncompressed.len(), uncompressed_size as usize);
        assert!(entry_count > 0 || uncompressed.is_empty());
        uncompressed
    }

    /// A. Empty round-trip. Writer created, no appends, then finish.
    #[test]
    fn writer_empty_roundtrip() {
        let tmp = tempdir().unwrap();
        let path = tmp.path().join("empty.bsst");
        let w = BondSstWriter::create(
            &path,
            BondSstWriteOpts {
                file_uid: 7,
                target_entries_per_block: 64,
                compression: CompressionKind::None,
            },
        )
        .unwrap();
        let stats = w.finish().unwrap();

        assert_eq!(stats.entries_written, 0);
        assert_eq!(stats.blocks_written, 0);
        assert_eq!(stats.first_key, None);
        assert_eq!(stats.last_key, None);

        let (block_count, footer_offset, file_uid) = read_header(&path);
        assert_eq!(block_count, 0);
        assert_eq!(file_uid, 7);
        assert_eq!(footer_offset, HEADER_BYTES as u64, "footer starts right after header when no blocks");

        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(bytes.len() as u64, stats.file_size_bytes);
    }

    /// B. Single-block round-trip. 500 monotonic appends; file size matches stats.
    #[test]
    fn writer_single_block_roundtrip() {
        let tmp = tempdir().unwrap();
        let path = tmp.path().join("single.bsst");
        let mut w = BondSstWriter::create(
            &path,
            BondSstWriteOpts {
                file_uid: 11,
                target_entries_per_block: 500,
                compression: CompressionKind::Zstd { level: 1 },
            },
        )
        .unwrap();

        for i in 0..500u64 {
            w.append(NodeId(i), 1, NodeId(i + 1_000_000), &mk_payload(i))
                .unwrap();
        }
        let stats = w.finish().unwrap();

        assert_eq!(stats.entries_written, 500);
        assert_eq!(stats.blocks_written, 1);
        assert_eq!(stats.first_key, Some((NodeId(0), 1, NodeId(1_000_000))));
        assert_eq!(stats.last_key, Some((NodeId(499), 1, NodeId(1_000_499))));

        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(bytes.len() as u64, stats.file_size_bytes);

        let (block_count, footer_offset, file_uid) = read_header(&path);
        assert_eq!(block_count, 1);
        assert_eq!(file_uid, 11);
        // Footer starts right after the single block.
        let block_size = BLOCK_HEADER_BYTES as u64 + stats.compressed_bytes;
        assert_eq!(footer_offset, HEADER_BYTES as u64 + block_size);
    }

    /// C. Multi-block with compression. 10K entries, target 1000/block.
    #[test]
    fn writer_multi_block_compression() {
        let tmp = tempdir().unwrap();
        let path = tmp.path().join("multi.bsst");
        let mut w = BondSstWriter::create(
            &path,
            BondSstWriteOpts {
                file_uid: 13,
                target_entries_per_block: 1000,
                compression: CompressionKind::Zstd { level: 1 },
            },
        )
        .unwrap();

        for i in 0..10_000u64 {
            w.append(NodeId(i), (i % 29) as u8, NodeId(i + 2_000_000), &mk_payload(i))
                .unwrap();
        }
        let stats = w.finish().unwrap();

        assert_eq!(stats.entries_written, 10_000);
        assert_eq!(stats.blocks_written, 10);
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(bytes.len() as u64, stats.file_size_bytes);

        let ratio =
            stats.compressed_bytes as f64 / stats.uncompressed_bytes.max(1) as f64;
        eprintln!(
            "writer_multi_block_compression: uncompressed={} B compressed={} B ratio={:.3}",
            stats.uncompressed_bytes, stats.compressed_bytes, ratio,
        );
        // No specific ratio asserted -- synthetic fixtures are
        // highly compressible. See Test K observation note.
        assert!(ratio < 1.0, "compressed body should be smaller than uncompressed for repeating fixtures");
    }

    /// Helper -- detect debug_assert panic OR typed-error rejection.
    /// debug_assert! is enabled in the test profile; the writer's
    /// release path returns the typed error instead of panicking.
    fn expect_ooo_rejection<F>(f: F)
    where
        F: FnOnce() -> Result<(), BondSstError> + std::panic::UnwindSafe,
    {
        let r = std::panic::catch_unwind(f);
        match r {
            Ok(Err(BondSstError::OutOfOrderAppend { .. })) => {}
            Ok(other) => panic!("expected OutOfOrderAppend, got {:?}", other),
            Err(_panic) => {
                // debug_assert! fired; equivalent evidence the
                // rejection arm was reached.
            }
        }
    }

    /// D. Subject-decreasing append rejected.
    #[test]
    fn writer_out_of_order_subject_rejected() {
        let tmp = tempdir().unwrap();
        let path = tmp.path().join("d.bsst");
        let mut w = BondSstWriter::create(
            &path,
            BondSstWriteOpts {
                file_uid: 21,
                target_entries_per_block: 64,
                compression: CompressionKind::None,
            },
        )
        .unwrap();
        w.append(NodeId(5), 1, NodeId(100), &mk_payload(1)).unwrap();
        expect_ooo_rejection(std::panic::AssertUnwindSafe(move || {
            w.append(NodeId(3), 1, NodeId(100), &mk_payload(2))
        }));
    }

    /// E. Predicate-decreasing append rejected (same subject).
    #[test]
    fn writer_out_of_order_predicate_rejected() {
        let tmp = tempdir().unwrap();
        let path = tmp.path().join("e.bsst");
        let mut w = BondSstWriter::create(
            &path,
            BondSstWriteOpts {
                file_uid: 22,
                target_entries_per_block: 64,
                compression: CompressionKind::None,
            },
        )
        .unwrap();
        w.append(NodeId(5), 10, NodeId(1), &mk_payload(1)).unwrap();
        expect_ooo_rejection(std::panic::AssertUnwindSafe(move || {
            w.append(NodeId(5), 5, NodeId(1), &mk_payload(2))
        }));
    }

    /// F. Object-decreasing append rejected (same subject + predicate).
    #[test]
    fn writer_out_of_order_object_rejected() {
        let tmp = tempdir().unwrap();
        let path = tmp.path().join("f.bsst");
        let mut w = BondSstWriter::create(
            &path,
            BondSstWriteOpts {
                file_uid: 23,
                target_entries_per_block: 64,
                compression: CompressionKind::None,
            },
        )
        .unwrap();
        w.append(NodeId(5), 10, NodeId(20), &mk_payload(1)).unwrap();
        expect_ooo_rejection(std::panic::AssertUnwindSafe(move || {
            w.append(NodeId(5), 10, NodeId(10), &mk_payload(2))
        }));
    }

    /// G. Same triple repeated is REJECTED (dedup is compaction's job).
    #[test]
    fn writer_duplicate_key_rejected() {
        let tmp = tempdir().unwrap();
        let path = tmp.path().join("g.bsst");
        let mut w = BondSstWriter::create(
            &path,
            BondSstWriteOpts {
                file_uid: 24,
                target_entries_per_block: 64,
                compression: CompressionKind::None,
            },
        )
        .unwrap();
        w.append(NodeId(7), 3, NodeId(42), &mk_payload(1)).unwrap();
        expect_ooo_rejection(std::panic::AssertUnwindSafe(move || {
            w.append(NodeId(7), 3, NodeId(42), &mk_payload(2))
        }));
    }

    /// H. Streaming writer bounded resident state -- structural
    /// invariant on `current_block_buf.capacity()` and
    /// `block_index.len()`. Same approach as Phase 2.4 Test H:
    /// assert on internal field state, not VmHWM, because parallel
    /// test runners make VmHWM deltas non-separable.
    #[test]
    fn writer_streaming_bounded_state_100k() {
        let baseline = read_self_rss_peak_kb();

        let tmp = tempdir().unwrap();
        let path = tmp.path().join("h.bsst");
        let target_entries_per_block: usize = 1000;
        let mut w = BondSstWriter::create(
            &path,
            BondSstWriteOpts {
                file_uid: 31,
                target_entries_per_block,
                compression: CompressionKind::Zstd { level: 1 },
            },
        )
        .unwrap();

        const N: u64 = 100_000;
        for i in 0..N {
            w.append(NodeId(i), (i % 29) as u8, NodeId(i + 3_000_000), &mk_payload(i))
                .unwrap();
            if i % 10_000 == 9_999 {
                let buf_cap_bytes = w.current_block_buf.capacity();
                assert!(
                    buf_cap_bytes <= 64 * 1024,
                    "current_block_buf.capacity() = {} after {} entries; expected <= 64 KB",
                    buf_cap_bytes,
                    i + 1,
                );
                let expected_blocks = (i as usize + 1) / target_entries_per_block;
                assert!(
                    w.block_index.len() == expected_blocks,
                    "block_index.len() = {}; expected {} after {} entries",
                    w.block_index.len(),
                    expected_blocks,
                    i + 1,
                );
            }
        }
        let stats = w.finish().unwrap();
        assert_eq!(stats.entries_written, N);
        let expected_blocks =
            (N as usize + target_entries_per_block - 1) / target_entries_per_block;
        assert_eq!(stats.blocks_written as usize, expected_blocks);

        if let (Some(b), Some(a)) = (baseline, read_self_rss_peak_kb()) {
            eprintln!(
                "writer_streaming_bounded_state_100k: VmHWM baseline={} KB, after={} KB, delta={} KB (diagnostic only)",
                b,
                a,
                a.saturating_sub(b),
            );
        }
    }

    /// I. TOMBSTONE confidence=0xFF byte round-trips through the
    /// wire format intact. Reads back via the lower-level
    /// `decode_block` helper and scans for the byte at the
    /// confidence position inside the bincode'd BondPayload.
    #[test]
    fn writer_tombstone_byte_round_trips() {
        let tmp = tempdir().unwrap();
        let path = tmp.path().join("i.bsst");
        let mut w = BondSstWriter::create(
            &path,
            BondSstWriteOpts {
                file_uid: 41,
                target_entries_per_block: 4,
                compression: CompressionKind::None,
            },
        )
        .unwrap();

        let payload = BondPayload {
            weight: 100,
            confidence: 0xFF, // TOMBSTONE marker
            keyframes: vec![],
            seq: 99,
        };
        w.append(NodeId(1), 0, NodeId(2), &payload).unwrap();
        let stats = w.finish().unwrap();
        assert_eq!(stats.blocks_written, 1);

        // Decode the (only) block.
        let block_offset = HEADER_BYTES as u64;
        let body = decode_block(&path, block_offset);

        // Entry framing: subject(8) + predicate(1) + object(8) +
        // payload_size(4) = 21 B, then the bincode'd BondPayload.
        assert!(body.len() > ENTRY_HEADER_BYTES);
        let payload_size = u32::from_le_bytes([
            body[17], body[18], body[19], body[20],
        ]) as usize;
        let payload_bytes = &body[ENTRY_HEADER_BYTES..ENTRY_HEADER_BYTES + payload_size];

        let decoded: BondPayload = bincode::deserialize(payload_bytes)
            .expect("BondPayload round-trip via bincode");
        assert_eq!(decoded.confidence, 0xFF, "TOMBSTONE byte preserved exactly");
        assert_eq!(decoded.weight, 100);
        assert_eq!(decoded.seq, 99);
        assert!(decoded.keyframes.is_empty());
    }

    /// J. file_uid round-trips into the wire header at bytes 20..28
    /// and the header CRC validates over [0..28].
    #[test]
    fn writer_file_uid_round_trips() {
        let tmp = tempdir().unwrap();
        let path = tmp.path().join("j.bsst");
        let mut w = BondSstWriter::create(
            &path,
            BondSstWriteOpts {
                file_uid: 42,
                target_entries_per_block: 4,
                compression: CompressionKind::None,
            },
        )
        .unwrap();
        w.append(NodeId(1), 0, NodeId(2), &mk_payload(1)).unwrap();
        let stats = w.finish().unwrap();
        assert_eq!(stats.entries_written, 1);

        let (_block_count, _footer_offset, file_uid) = read_header(&path);
        assert_eq!(file_uid, 42, "file_uid round-trips intact");

        // Also assert the larger sentinel value lands correctly.
        let path2 = tmp.path().join("j2.bsst");
        let mut w2 = BondSstWriter::create(
            &path2,
            BondSstWriteOpts {
                file_uid: 0xCAFEBABE_DEADBEEF,
                target_entries_per_block: 4,
                compression: CompressionKind::None,
            },
        )
        .unwrap();
        w2.append(NodeId(1), 0, NodeId(2), &mk_payload(1)).unwrap();
        w2.finish().unwrap();
        let (_, _, big_uid) = read_header(&path2);
        assert_eq!(big_uid, 0xCAFEBABE_DEADBEEF);
    }

    /// K. Compression-ratio observation only -- no specific number
    /// asserted because synthetic fixtures don't match production
    /// keyframe-source entropy. Diagnostic.
    #[test]
    fn writer_compression_observation() {
        let tmp = tempdir().unwrap();
        let path = tmp.path().join("k.bsst");
        let mut w = BondSstWriter::create(
            &path,
            BondSstWriteOpts {
                file_uid: 51,
                target_entries_per_block: 1000,
                compression: CompressionKind::Zstd { level: 1 },
            },
        )
        .unwrap();

        // Cycle through a small number of distinct payloads to
        // simulate the per-block compression amortisation.
        for i in 0..5_000u64 {
            let mut p = mk_payload(i);
            p.weight = (i % 256) as u8;
            p.keyframes.push(TemporalKeyframe {
                timestamp: 1_700_000_000 + (i as i64) * 60,
                weight: ((i as f32) % 100.0) / 100.0,
                source: KeyframeSource::Baseline,
            });
            w.append(NodeId(i), (i % 29) as u8, NodeId(i + 4_000_000), &p)
                .unwrap();
        }
        let stats = w.finish().unwrap();
        let ratio =
            stats.compressed_bytes as f64 / stats.uncompressed_bytes.max(1) as f64;
        eprintln!(
            "writer_compression_observation: uncompressed={} B compressed={} B ratio={:.3} -- synthetic fixture, production rates will differ",
            stats.uncompressed_bytes, stats.compressed_bytes, ratio,
        );
    }
}
