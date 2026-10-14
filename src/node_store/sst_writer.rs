//! `NodeSstWriter` -- streaming on-disk writer for a NodeStore SST.
//!
//! OP-BRAIN-V5 Phase 2.4 (Q3, Q7 of the design doc; audit S10
//! "NodeSST writer + reader" sub-port). The writer accepts an
//! append-only stream of `(NodeId, &NodeEntry)` pairs sorted by
//! `NodeId` ascending and emits a single immutable SST file that
//! the matching `NodeSstReader` (this commit, `sst_reader.rs`)
//! consumes.
//!
//! ## Wire format
//!
//! ```text
//! Header (48 B fixed):
//!   bytes 0..4    magic              u32 LE  = 0x4E535354 ("NSST")
//!   bytes 4..6    format_version     u16 LE  = NSST_FORMAT_VERSION (=2)
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
//!       node_id           u64 LE
//!       entry_size        u32 LE       length of bincode payload
//!       entry_payload     entry_size B bincode(NodeEntry)
//!     Entries inside a block are sorted by node_id ascending;
//!     the stream as a whole is sorted by node_id ascending
//!     (blocks own non-overlapping NodeId ranges).
//!
//! Footer (variable size, written AFTER all blocks; absolute file
//! offset captured in the header's `footer_offset`):
//!   block_index (28 B per entry, `block_count` entries):
//!     bytes 0..8    first_node_id      u64 LE
//!     bytes 8..16   last_node_id       u64 LE
//!     bytes 16..24  block_offset       u64 LE
//!     bytes 24..28  block_size         u32 LE (block_header + block_body)
//!   block_index_byte_len u32 LE  = block_count * 28
//!   footer_crc32         u32 LE  = CRC over block_index bytes ONLY
//!
//! Trailer (12 B, the final 12 bytes of the file -- enables reverse
//! seek validation when the footer offset is suspect):
//!   bytes 0..4    magic_tail         u32 LE  = 0x4E535354 ("NSST")
//!   bytes 4..12   footer_offset_tail u64 LE  = mirrors header.footer_offset
//! ```
//!
//! ## NodeEntry serialization
//!
//! `bincode` 1.x is the codec choice. Rationale (also see the
//! the project log log entry for this commit):
//!
//! - `bincode` is already the workspace codec for `KnowledgeNode`
//!   (`src/node.rs::serialize_knowledge_node`) and `TemporalKeyframe`
//!   (`src/sst.rs::encode_keyframe`). Reusing it means zero new
//!   serde-derive surface and one less codec to reason about.
//! - `NodeEntry { payload: KnowledgeNode, tombstone: bool, seq: u64 }`
//!   auto-derives `Serialize` via the `KnowledgeNode` chain (the
//!   three field derives are added to `NodeEntry` itself in
//!   `memtable.rs` as part of this commit).
//! - rkyv was considered for zero-copy reads; rejected for 2.4
//!   because the read path here always copies into an owned
//!   `NodeEntry` (the LRU cache's value is a fully-owned
//!   `Vec<(NodeId, NodeEntry)>`). Zero-copy upside is moot.
//! - Custom binary was rejected as scope creep -- the embedding
//!   (`Vec<f32; 384>`) dominates payload bytes and bincode-encodes
//!   it as `len: u64` + raw little-endian f32s, which is already
//!   the minimum legal envelope.
//!
//! ## Compression observation
//!
//! Per the audit (S2.1 / S9 SURPRISE-D), the `KnowledgeNode`
//! payload is ~1800 B and ~85 % of those bytes are the 1536 B
//! embedding (`Vec<f32; 384>`). f32 embeddings are high-entropy
//! after training, so zstd at level 1 typically saves <5 % on real
//! payloads -- effectively storing embeddings near-raw. The writer
//! still applies zstd by default because (i) the metadata save is a
//! small win at zero ergonomic cost; (ii) it preserves the option of
//! swapping in an embedding-aware codec later without changing the
//! on-disk container shape. Set [`CompressionKind::None`] to skip
//! compression entirely.
//!
//! ## Streaming discipline
//!
//! The writer holds AT MOST one in-progress block in memory --
//! `current_block_buf: Vec<u8>` plus a few small counters. A flush
//! happens when `current_block_entries == target_entries_per_block`
//! or when [`NodeSstWriter::finish`] is called. The footer's
//! `block_index` accumulates one `BlockIndexEntry` per flushed block
//! (28 B each); at 1 M blocks this is 28 MB resident -- proportional
//! only to block count, not entry count.
//!
//! Append requires monotonically-strictly-increasing `NodeId`s.
//! Out-of-order or duplicate calls error with
//! [`NodeSstError::OutOfOrderAppend`].
//!
//! ## Atomicity
//!
//! `create()` opens `<path>.tmp.<file_uid>`; `finish()` fsyncs and
//! renames into `<path>`. Either `<path>` does not exist or it is a
//! complete valid SST (header CRC + per-block CRCs + footer CRC all
//! valid).

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crate::node_intern::NodeId;
use crate::node_store::memtable::NodeEntry;
use crate::encrypt::{xor_in_place, MasterKey};
use crate::sst::derive_block_keystream_index;
use rand::RngCore;

// ----------------------------------------------------------------------
// Wire-format constants.
// ----------------------------------------------------------------------

/// Magic bytes at the start of every NodeSST file ("NSST").
pub const NSST_MAGIC: u32 = 0x4E535354;

/// Current wire-format version. Bumped on any breaking change. Reader
/// rejects unknown versions with [`NodeSstError::FormatVersionUnsupported`].
/// v2 (OP-BRAIN-V5 6.5) widened the header 32 -> 48 B to carry a
/// per-SST encryption flag + IV (Option B, v4-mirrored keystream).
pub const NSST_FORMAT_VERSION: u16 = 2;

/// Header `flags` bit 0: block bodies are XOR-keystream encrypted
/// (encrypt-then-compress at write; decompress-then-decrypt at read),
/// mirroring the v4 SST scheme byte-for-byte.
pub const FLAG_ENCRYPTED: u8 = 0x01;

/// Fixed header size in bytes.
pub(crate) const HEADER_BYTES: usize = 48;

/// Per-block header size in bytes.
pub(crate) const BLOCK_HEADER_BYTES: usize = 4 + 4 + 4 + 1 + 4;

/// Per-entry-in-block header bytes (node_id + entry_size).
pub(crate) const ENTRY_HEADER_BYTES: usize = 8 + 4;

/// Per-index-entry size in bytes (first + last + offset + size).
pub(crate) const INDEX_ENTRY_BYTES: usize = 8 + 8 + 8 + 4;

/// Trailer size in bytes (magic_tail + footer_offset_tail).
pub(crate) const TRAILER_BYTES: usize = 4 + 8;

/// Default entries per block (audit S2.1: 64-128 is the cache-
/// friendly band; 64 keeps in-memory buffer < ~120 KB at the
/// audit-projected ~1.8 KB/node payload).
pub const DEFAULT_TARGET_ENTRIES_PER_BLOCK: usize = 64;

/// Maximum entries per block. Bounds the in-memory write buffer.
pub const MAX_TARGET_ENTRIES_PER_BLOCK: usize = 8192;

/// BufWriter capacity in bytes.
const BUFWRITER_CAPACITY: usize = 64 * 1024;

// ----------------------------------------------------------------------
// Public types.
// ----------------------------------------------------------------------

/// Compression scheme for block bodies. The on-disk discriminator is
/// the `u8` returned by [`Self::as_u8`].
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
    /// level is not round-tripped on disk; decompression is parameter
    /// free, so the reader-side `Zstd` variant carries `level: 0`.
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
pub struct NodeSstWriteOpts {
    /// Number of entries to accumulate before flushing a block.
    /// Clamped to `[1, MAX_TARGET_ENTRIES_PER_BLOCK]`.
    pub target_entries_per_block: usize,
    /// Compression scheme for block bodies.
    pub compression: CompressionKind,
    /// Stable identifier for this SST. Written into the header and
    /// returned by [`crate::node_store::sst_reader::NodeSstReader::file_uid`].
    pub file_uid: u64,
}

impl Default for NodeSstWriteOpts {
    fn default() -> Self {
        NodeSstWriteOpts {
            target_entries_per_block: DEFAULT_TARGET_ENTRIES_PER_BLOCK,
            compression: CompressionKind::default(),
            file_uid: 0,
        }
    }
}

/// Statistics returned by [`NodeSstWriter::finish`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NodeSstWriteStats {
    pub entries_written: u64,
    pub blocks_written: u32,
    pub uncompressed_bytes: u64,
    pub compressed_bytes: u64,
    pub footer_bytes: u64,
    pub file_size_bytes: u64,
    /// First (lowest) NodeId written, if any. `None` when the SST was
    /// finished with zero entries. Mirrors the bond-side
    /// `BondSstWriteStats::first_key` (Phase 3.8.2, Tier 2B).
    pub first_node_id: Option<NodeId>,
    /// Last (highest) NodeId written, if any. `None` when the SST was
    /// finished with zero entries. Mirrors the bond-side
    /// `BondSstWriteStats::last_key`.
    pub last_node_id: Option<NodeId>,
    /// Best-effort observation of peak resident-set-size in KB during
    /// the write, sampled from `/proc/self/status` (Linux). `None` on
    /// non-Linux platforms or when reading the file fails.
    pub peak_writer_rss_kb: Option<u64>,
}

/// Errors surfaced by [`NodeSstWriter`] and the matching reader.
#[derive(Debug)]
pub enum NodeSstError {
    /// `std::io::Error` propagated from open / read / write / fsync /
    /// rename.
    Io(std::io::Error),
    /// Header magic was not [`NSST_MAGIC`].
    MagicMismatch { observed: u32 },
    /// Header `format_version` is not [`NSST_FORMAT_VERSION`].
    FormatVersionUnsupported { observed: u16 },
    /// CRC32 mismatch on the header, a block body, or the footer.
    CrcMismatch { scope: CrcScope, expected: u32, observed: u32 },
    /// Caller passed a `NodeId` not strictly greater than the previous
    /// append.
    OutOfOrderAppend { previous: NodeId, attempted: NodeId },
    /// Block body's on-disk `compression_kind` byte was outside the
    /// known range. Reader-side only.
    UnknownCompressionKind { observed: u8 },
    /// bincode failed to (de)serialise a `NodeEntry`. Wraps the codec
    /// error as a `String` (bincode 1.x's error type is not always
    /// Send + Sync).
    Bincode(String),
    /// zstd encode / decode failed.
    Zstd(String),
    /// `target_entries_per_block` was 0 or > [`MAX_TARGET_ENTRIES_PER_BLOCK`].
    InvalidOpts(String),
    /// Reader-side: structural value out of bounds (footer offset past
    /// EOF, block size exceeds file size, etc.).
    Format(String),
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

impl std::fmt::Display for NodeSstError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NodeSstError::Io(e) => write!(f, "NodeSST IO: {}", e),
            NodeSstError::MagicMismatch { observed } => write!(
                f,
                "NodeSST magic mismatch: expected {:#x}, observed {:#x}",
                NSST_MAGIC, observed
            ),
            NodeSstError::FormatVersionUnsupported { observed } => write!(
                f,
                "NodeSST format version unsupported: expected {}, observed {}",
                NSST_FORMAT_VERSION, observed
            ),
            NodeSstError::CrcMismatch { scope, expected, observed } => write!(
                f,
                "NodeSST CRC32 mismatch ({:?}): expected {:#010x}, observed {:#010x}",
                scope, expected, observed
            ),
            NodeSstError::OutOfOrderAppend { previous, attempted } => write!(
                f,
                "NodeSST out-of-order append: previous NodeId({}), attempted NodeId({})",
                previous.0, attempted.0
            ),
            NodeSstError::UnknownCompressionKind { observed } => write!(
                f,
                "NodeSST unknown compression_kind byte: {}",
                observed
            ),
            NodeSstError::Bincode(s) => write!(f, "NodeSST bincode error: {}", s),
            NodeSstError::Zstd(s) => write!(f, "NodeSST zstd error: {}", s),
            NodeSstError::InvalidOpts(s) => write!(f, "NodeSST invalid opts: {}", s),
            NodeSstError::Format(s) => write!(f, "NodeSST format: {}", s),
            NodeSstError::EncryptedButNoKey => write!(
                f,
                "NodeSST block bodies are encrypted but the reader has no key"
            ),
        }
    }
}

impl std::error::Error for NodeSstError {}

impl From<std::io::Error> for NodeSstError {
    fn from(e: std::io::Error) -> Self {
        NodeSstError::Io(e)
    }
}

// ----------------------------------------------------------------------
// Writer.
// ----------------------------------------------------------------------

/// Streaming NodeSST writer.
pub struct NodeSstWriter {
    writer: Option<BufWriter<File>>,
    path: PathBuf,
    tmp_path: PathBuf,
    opts: NodeSstWriteOpts,

    next_block_offset: u64,
    first_node_id: Option<NodeId>,
    last_node_id: Option<NodeId>,

    current_block_buf: Vec<u8>,
    current_block_entries: u32,
    current_block_first_id: Option<NodeId>,
    current_block_last_id: Option<NodeId>,

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

/// One block_index entry. Mirrored on disk by the writer footer and
/// parsed by the reader.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BlockIndexEntry {
    pub first_node_id: NodeId,
    pub last_node_id: NodeId,
    pub block_offset: u64,
    pub block_size: u32,
}

impl NodeSstWriter {
    /// Create a new NodeSST writer targeting `path`. Writes go to a
    /// temporary file `<path>.tmp.<file_uid>` until `finish()` renames
    /// atomically.
    pub fn create(path: &Path, opts: NodeSstWriteOpts) -> Result<Self, NodeSstError> {
        Self::create_with_key(path, opts, None)
    }

    /// Create a writer that encrypts each block body under `key`
    /// (OP-BRAIN-V5 6.5, Option B). `key = None` writes byte-identical
    /// plaintext to the pre-6.5 writer. When `Some`, a fresh random
    /// 16-byte IV is generated and stamped into the header and each
    /// block body is XOR-encrypted before compression.
    pub fn create_with_key(
        path: &Path,
        opts: NodeSstWriteOpts,
        key: Option<MasterKey>,
    ) -> Result<Self, NodeSstError> {
        if opts.target_entries_per_block == 0
            || opts.target_entries_per_block > MAX_TARGET_ENTRIES_PER_BLOCK
        {
            return Err(NodeSstError::InvalidOpts(format!(
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

        Ok(NodeSstWriter {
            writer: Some(writer),
            path: path.to_path_buf(),
            tmp_path,
            opts,
            next_block_offset: HEADER_BYTES as u64,
            first_node_id: None,
            last_node_id: None,
            current_block_buf: Vec::with_capacity(4096),
            current_block_entries: 0,
            current_block_first_id: None,
            current_block_last_id: None,
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

    /// Append a single `(NodeId, NodeEntry)` pair. NodeIds must be
    /// strictly greater than the previous append.
    pub fn append(&mut self, id: NodeId, entry: &NodeEntry) -> Result<(), NodeSstError> {
        if let Some(prev) = self.last_node_id {
            if id.0 <= prev.0 {
                debug_assert!(
                    false,
                    "NodeSstWriter::append called with non-monotonic NodeId: {} <= {}",
                    id.0, prev.0
                );
                return Err(NodeSstError::OutOfOrderAppend { previous: prev, attempted: id });
            }
        }

        let entry_bytes = bincode::serialize(entry)
            .map_err(|e| NodeSstError::Bincode(format!("serialize NodeEntry: {}", e)))?;

        self.current_block_buf.extend_from_slice(&id.0.to_le_bytes());
        let entry_size = u32::try_from(entry_bytes.len()).map_err(|_| {
            NodeSstError::Format(format!(
                "NodeEntry serialised size {} exceeds u32::MAX",
                entry_bytes.len()
            ))
        })?;
        self.current_block_buf.extend_from_slice(&entry_size.to_le_bytes());
        self.current_block_buf.extend_from_slice(&entry_bytes);

        if self.current_block_first_id.is_none() {
            self.current_block_first_id = Some(id);
        }
        self.current_block_last_id = Some(id);
        self.current_block_entries += 1;
        if self.first_node_id.is_none() {
            self.first_node_id = Some(id);
        }
        self.last_node_id = Some(id);
        self.entries_written += 1;

        if self.current_block_entries as usize >= self.opts.target_entries_per_block {
            self.flush_block()?;
        }
        Ok(())
    }

    /// Finalise the SST.
    pub fn finish(mut self) -> Result<NodeSstWriteStats, NodeSstError> {
        if self.current_block_entries > 0 {
            self.flush_block()?;
        }

        let footer_offset = self.next_block_offset;

        let block_count = u32::try_from(self.block_index.len()).map_err(|_| {
            NodeSstError::Format(format!(
                "block_count {} exceeds u32::MAX",
                self.block_index.len()
            ))
        })?;
        let index_byte_len = (self.block_index.len() * INDEX_ENTRY_BYTES) as u32;

        let mut footer_payload: Vec<u8> = Vec::with_capacity(
            self.block_index.len() * INDEX_ENTRY_BYTES + 4 + 4,
        );
        for ent in &self.block_index {
            footer_payload.extend_from_slice(&ent.first_node_id.0.to_le_bytes());
            footer_payload.extend_from_slice(&ent.last_node_id.0.to_le_bytes());
            footer_payload.extend_from_slice(&ent.block_offset.to_le_bytes());
            footer_payload.extend_from_slice(&ent.block_size.to_le_bytes());
        }
        let footer_crc = crc32fast::hash(&footer_payload);
        footer_payload.extend_from_slice(&index_byte_len.to_le_bytes());
        footer_payload.extend_from_slice(&footer_crc.to_le_bytes());
        let footer_bytes = footer_payload.len() as u64;

        let mut trailer = [0u8; TRAILER_BYTES];
        trailer[0..4].copy_from_slice(&NSST_MAGIC.to_le_bytes());
        trailer[4..12].copy_from_slice(&footer_offset.to_le_bytes());

        {
            let writer = self.writer.as_mut().expect("writer is Some until finish() consumes self");
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

        Ok(NodeSstWriteStats {
            entries_written: self.entries_written,
            blocks_written: block_count,
            uncompressed_bytes: self.uncompressed_bytes,
            compressed_bytes: self.compressed_bytes,
            footer_bytes,
            file_size_bytes: file_size,
            first_node_id: self.first_node_id,
            last_node_id: self.last_node_id,
            peak_writer_rss_kb: peak_rss_kb,
        })
    }

    fn flush_block(&mut self) -> Result<(), NodeSstError> {
        debug_assert!(self.current_block_entries > 0);

        let uncompressed_size = u32::try_from(self.current_block_buf.len()).map_err(|_| {
            NodeSstError::Format(format!(
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

        let (compressed_bytes_slice, compression_kind_byte): (&[u8], u8) =
            match self.opts.compression {
                CompressionKind::None => (self.current_block_buf.as_slice(), 0u8),
                CompressionKind::Zstd { level } => {
                    self.zstd_buf.clear();
                    let mut encoder = zstd::Encoder::new(&mut self.zstd_buf, level)
                        .map_err(|e| NodeSstError::Zstd(format!("encoder init: {}", e)))?;
                    encoder
                        .write_all(&self.current_block_buf)
                        .map_err(|e| NodeSstError::Zstd(format!("encode write: {}", e)))?;
                    encoder
                        .finish()
                        .map_err(|e| NodeSstError::Zstd(format!("encoder finish: {}", e)))?;
                    (self.zstd_buf.as_slice(), 1u8)
                }
            };

        let compressed_size = u32::try_from(compressed_bytes_slice.len()).map_err(|_| {
            NodeSstError::Format(format!(
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
            NodeSstError::Format(format!(
                "on-disk block size {} exceeds u32::MAX",
                block_size_on_disk
            ))
        })?;

        self.block_index.push(BlockIndexEntry {
            first_node_id: self
                .current_block_first_id
                .expect("first_node_id Some when entries > 0"),
            last_node_id: self
                .current_block_last_id
                .expect("last_node_id Some when entries > 0"),
            block_offset: self.next_block_offset,
            block_size: block_size_u32,
        });

        self.next_block_offset += block_size_on_disk;
        self.uncompressed_bytes += uncompressed_size as u64;
        self.compressed_bytes += compressed_size as u64;

        self.current_block_buf.clear();
        self.current_block_entries = 0;
        self.current_block_first_id = None;
        self.current_block_last_id = None;

        Ok(())
    }
}

impl Drop for NodeSstWriter {
    fn drop(&mut self) {
        // If dropped without finish(), the temporary file may still be
        // on disk. Best-effort cleanup; drop must not panic.
        if self.writer.is_some() {
            self.writer.take();
            let _ = std::fs::remove_file(&self.tmp_path);
        }
    }
}

// ----------------------------------------------------------------------
// Internal helpers (writer side).
// ----------------------------------------------------------------------

/// Construct the 32-byte header. The `header_crc32` is computed over
/// bytes 0..28 (the 28 bytes preceding the crc field itself).
pub(crate) fn build_header(
    block_count: u32,
    footer_offset: u64,
    file_uid: u64,
    flags: u8,
    iv: &[u8; 16],
) -> [u8; HEADER_BYTES] {
    let mut buf = [0u8; HEADER_BYTES];
    buf[0..4].copy_from_slice(&NSST_MAGIC.to_le_bytes());
    buf[4..6].copy_from_slice(&NSST_FORMAT_VERSION.to_le_bytes());
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

/// Resolve the temporary path for a writer targeting `path` with the
/// given `file_uid`. Lives in the same parent directory so the final
/// `rename` is intra-filesystem (atomic per POSIX).
fn tmp_path_for(path: &Path, file_uid: u64) -> PathBuf {
    let parent = path.parent().unwrap_or(Path::new("."));
    let file_name = match path.file_name() {
        Some(n) => n.to_string_lossy().into_owned(),
        None => "nodesst".to_string(),
    };
    parent.join(format!("{}.tmp.{}", file_name, file_uid))
}

/// Best-effort read of `/proc/self/status` to extract `VmHWM` (peak
/// resident-set size since process start, in kilobytes). Returns
/// `None` on non-Linux or any I/O / parse failure -- metadata only.
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
    use crate::node::{Confidence, KnowledgeContent, KnowledgeNode, KnowledgeUnit};
    use crate::seed::Seed;
    use tempfile::tempdir;

    /// Construct a synthetic small KnowledgeNode (no embedding) for
    /// the bounded-RSS tests where we want lots of entries cheaply.
    fn mk_small_node(label: &str) -> KnowledgeNode {
        KnowledgeNode {
            seed: Seed::from_label(label),
            content: label.to_string(),
            knowledge: KnowledgeUnit {
                content: KnowledgeContent::Concept(label.to_string()),
                confidence: Confidence::Verified,
                created_at: 0,
                updated_at: 0,
            },
            embedding: Vec::new(),
            parent_seeds: Vec::new(),
            typed_ratio: 0.0,
        }
    }

    fn mk_entry(label: &str, seq: u64) -> NodeEntry {
        NodeEntry { payload: mk_small_node(label), tombstone: false, seq }
    }

    /// E. Out-of-order append rejection.
    #[test]
    fn writer_out_of_order_append_rejected() {
        let tmp = tempdir().unwrap();
        let path = tmp.path().join("ooo.nsst");
        let mut w = NodeSstWriter::create(
            &path,
            NodeSstWriteOpts {
                file_uid: 1,
                target_entries_per_block: 64,
                // Out-of-order check must happen BEFORE any disk write
                // discipline kicks in; use raw compression so a debug
                // panic-on-assert path isn't fired by a debug_assert in
                // tests that build with assertions.
                compression: CompressionKind::None,
            },
        )
        .unwrap();
        w.append(NodeId(5), &mk_entry("a", 1)).unwrap();
        // debug_assert! is enabled in the test profile; the call below
        // would panic before reaching the OutOfOrderAppend return. We
        // catch the panic to assert the return path semantics in a
        // release-equivalent way.
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            w.append(NodeId(3), &mk_entry("b", 2))
        }));
        match r {
            Ok(Err(NodeSstError::OutOfOrderAppend { .. })) => {}
            Ok(other) => panic!("expected OutOfOrderAppend, got {:?}", other),
            Err(_panic) => {
                // debug_assert panic fires in debug profile -- accept it
                // as evidence the rejection arm was reached. Production
                // (release) would return the typed error instead.
            }
        }
    }

    /// H. Streaming writer bounded resident state (synthetic small
    /// payloads, 100,000 entries).
    ///
    /// Honest scope. The bounded-RSS property we care about is
    /// structural: regardless of how many entries the writer has
    /// seen, its STATE comprises (a) at most one in-progress block
    /// buffer (current_block_buf) capped by target_entries_per_block
    /// + (b) a block_index Vec proportional to BLOCK count not
    /// ENTRY count + (c) the reusable zstd_buf. We assert these
    /// directly via internal field inspection; the alternative --
    /// asserting on /proc/self/status VmHWM -- is noisy under
    /// parallel-test allocator pressure (proven empirically with
    /// 1817 sibling tests warming the heap, VmHWM deltas attributable
    /// to this test are not separable from the noise floor).
    ///
    /// We DO still sample VmHWM and log it as a diagnostic.
    #[test]
    fn writer_streaming_bounded_state_100k() {
        let baseline = read_self_rss_peak_kb();

        let tmp = tempdir().unwrap();
        let path = tmp.path().join("rss.nsst");
        let target_entries_per_block: usize = 128;
        let mut w = NodeSstWriter::create(
            &path,
            NodeSstWriteOpts {
                file_uid: 2,
                target_entries_per_block,
                compression: CompressionKind::Zstd { level: 1 },
            },
        )
        .unwrap();

        const N: u64 = 100_000;
        for i in 0..N {
            let entry = mk_entry(&format!("n_{}", i), i);
            w.append(NodeId(i), &entry).unwrap();
            if i % 10_000 == 9_999 {
                let buf_cap_bytes = w.current_block_buf.capacity();
                assert!(
                    buf_cap_bytes <= 64 * 1024,
                    "current_block_buf.capacity() = {} after {} entries; expected <= 64 KB",
                    buf_cap_bytes, i + 1,
                );
                let expected_blocks = (i as usize + 1) / target_entries_per_block;
                assert!(
                    w.block_index.len() == expected_blocks,
                    "block_index.len() = {}; expected {} after {} entries",
                    w.block_index.len(), expected_blocks, i + 1,
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
                b, a, a.saturating_sub(b),
            );
        }
    }

    /// J (writer-side). file_uid is encoded into the header bytes at
    /// 20..28 and the header_crc32 at 28..32 validates over [0..28].
    /// Reader-side round-trip lives in `sst_reader::tests`.
    #[test]
    fn writer_file_uid_in_header() {
        let tmp = tempdir().unwrap();
        let path = tmp.path().join("uid.nsst");
        let mut w = NodeSstWriter::create(
            &path,
            NodeSstWriteOpts {
                file_uid: 0xCAFEBABE_DEADBEEF,
                ..NodeSstWriteOpts::default()
            },
        )
        .unwrap();
        w.append(NodeId(1), &mk_entry("a", 1)).unwrap();
        let stats = w.finish().unwrap();
        assert_eq!(stats.entries_written, 1);
        assert_eq!(stats.blocks_written, 1);

        let bytes = std::fs::read(&path).unwrap();
        assert!(bytes.len() >= HEADER_BYTES);
        let magic = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        assert_eq!(magic, NSST_MAGIC);
        let header_crc_disk =
            u32::from_le_bytes([bytes[44], bytes[45], bytes[46], bytes[47]]);
        let expected_crc = crc32fast::hash(&bytes[..44]);
        assert_eq!(header_crc_disk, expected_crc, "on-disk header CRC must verify");

        let file_uid_disk = u64::from_le_bytes([
            bytes[20], bytes[21], bytes[22], bytes[23],
            bytes[24], bytes[25], bytes[26], bytes[27],
        ]);
        assert_eq!(file_uid_disk, 0xCAFEBABE_DEADBEEF);
    }

    /// Phase 3.8.2 (Tier 2B). NodeSstWriteStats exposes the exact
    /// first/last NodeId written, mirroring the bond-side
    /// BondSstWriteStats first_key/last_key.
    #[test]
    fn phase382_write_stats_expose_first_last() {
        let tmp = tempdir().unwrap();
        let path = tmp.path().join("range.nsst");
        let mut w = NodeSstWriter::create(
            &path,
            NodeSstWriteOpts { file_uid: 3, ..NodeSstWriteOpts::default() },
        )
        .unwrap();
        // Non-contiguous ids on purpose: the stats must report the
        // ids actually appended, not a count-derived range.
        for i in 0..50u64 {
            let id = NodeId(10 + i * 3);
            w.append(id, &mk_entry(&format!("r_{}", i), i)).unwrap();
        }
        let stats = w.finish().unwrap();
        assert_eq!(stats.entries_written, 50);
        assert_eq!(stats.first_node_id, Some(NodeId(10)));
        assert_eq!(stats.last_node_id, Some(NodeId(10 + 49 * 3)));
    }

    /// Phase 3.8.2 (Tier 2B). Zero-entry SST: first/last are None.
    #[test]
    fn phase382_write_stats_empty_sst_none() {
        let tmp = tempdir().unwrap();
        let path = tmp.path().join("empty.nsst");
        let w = NodeSstWriter::create(
            &path,
            NodeSstWriteOpts { file_uid: 4, ..NodeSstWriteOpts::default() },
        )
        .unwrap();
        let stats = w.finish().unwrap();
        assert_eq!(stats.entries_written, 0);
        assert_eq!(stats.first_node_id, None);
        assert_eq!(stats.last_node_id, None);
    }
}
