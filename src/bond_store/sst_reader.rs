//! `BondSstReader` -- on-demand reader for a BondStore SST.
//!
//! OP-BRAIN-V5 Phase 3.2 (Q2 of the design doc; Phase 3.0 audit
//! sub-port 3.2). The reader opens an SST produced by
//! [`crate::bond_store::sst_writer::BondSstWriter`] and serves
//! bounded range scans over the `(subject, predicate, object)`
//! triple key without materialising the whole file into RAM.
//!
//! See [`crate::bond_store::sst_writer`] for the wire format. The
//! reader holds the header + the footer's `block_index` in memory
//! throughout its lifetime; blocks are decoded on demand and cached
//! in a bounded LRU.
//!
//! ## Range-scan-primary API
//!
//! Phase 2.4 [`crate::node_store::sst_reader::NodeSstReader`]
//! exposes `get(NodeId)` as the dominant access pattern because
//! nodes are identified by a single u64 key and the most common
//! caller is "fetch this node by id". Bonds query differently:
//! the dominant graph-traversal pattern is "all bonds with
//! subject = s", a range scan over the contiguous key prefix
//! `(s, _, _)`. So this reader's primary API is
//! [`Self::iter_range`] over a `(NodeId, u8, NodeId)` half-open
//! range, plus [`Self::iter_subject`] as the convenience wrapper
//! the read path actually uses. There is no `get(triple)` --
//! callers that genuinely want a single (subject, predicate,
//! object) lookup write `iter_subject(s).find(|(p, o, _)| ...)`,
//! which is one block load and a linear scan within the block.
//!
//! ## LRU block cache
//!
//! Adapted from the shape of Phase 2.4
//! [`crate::node_store::sst_reader`]'s `BlockCacheInner`: keyed
//! on `block_id: u32`, capped by a configurable entry count
//! (default [`DEFAULT_BLOCK_CACHE_CAP`] = 64), MRU at the back of
//! a `VecDeque` and LRU at the front. The "always leave at least
//! one entry resident so an oversized block does not infinite-
//! evict" rule is preserved. The cache is held inside the reader
//! (not shared across readers in this phase -- Phase 3.7
//! `BondSstReaderCache` may revisit when the read path lands).
//!
//! What is salvageable from NodeSstReader's cache: the LRU
//! discipline, the saturating-arithmetic accounting, the
//! "always leave at least one entry resident" rule, and the
//! linear-scan within a block (block sizes are bounded so linear
//! search beats binary search at this scale).
//!
//! What is NOT salvageable: the v4 `ParsedBlock` shape (with its
//! per-block `subject_table`, `node_table`, `keyframe_pool`) is
//! specific to the v4 wire format and meaningless here. BondSST
//! blocks decode to a flat `Vec<(NodeId, u8, NodeId,
//! BondPayload)>` keyed on the full triple.

use std::collections::VecDeque;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use crate::node_intern::NodeId;
use crate::bond_store::sst_writer::{
    BlockIndexEntry, BondPayload, BondSstError, CompressionKind, CrcScope,
    BLOCK_HEADER_BYTES, BSST_FORMAT_VERSION, BSST_MAGIC, ENTRY_HEADER_BYTES,
    FLAG_ENCRYPTED, HEADER_BYTES, INDEX_ENTRY_BYTES, TRAILER_BYTES,
};
use crate::encrypt::{xor_in_place, MasterKey};
use crate::sst::derive_block_keystream_index;

/// Default cap on the in-memory LRU block cache (entries).
///
/// At the writer's default `target_entries_per_block = 1000` and
/// the Phase 3.0 audit's projected ~57 B/entry uncompressed +
/// ~13-15 B/entry compressed, a default cap of 64 blocks gives
/// ~3.5 MB worst-case resident block cache at production density.
/// Mirrors `NodeSstReader::DEFAULT_BLOCK_CACHE_CAP`.
pub const DEFAULT_BLOCK_CACHE_CAP: usize = 64;

/// Per-reader read-side statistics. Mutated under the reader's
/// internal lock and surfaced via [`BondSstReader::stats`].
#[derive(Debug, Clone, Copy, Default)]
pub struct BondSstReadStats {
    /// Number of cache lookups that hit the in-memory LRU.
    pub block_cache_hits: u64,
    /// Number of cache lookups that missed and required a disk
    /// decode.
    pub block_cache_misses: u64,
    /// Number of disk read syscalls (block reads only; the
    /// open-time header + footer reads are not counted).
    pub blocks_read: u64,
    /// Cumulative bytes read from disk (block bodies post-`pread`).
    pub disk_bytes_read: u64,
    /// Number of entries yielded by any iterator over this reader.
    pub entries_yielded: u64,
}

/// A decoded block -- the result of decompressing the on-disk body
/// and parsing its entry stream. The reader stores these in the
/// LRU cache after the first read.
#[derive(Debug, Clone)]
/// OP-SST-CACHE-1 -- the cache stores `Arc<DecodedBlock>`, NOT
/// `DecodedBlock`.
///
/// It used to store the block by value and `.clone()` it on EVERY
/// lookup -- and again on insert, so a miss cost two full copies and a
/// hit cost one. A block holds up to `MAX_TARGET_ENTRIES_PER_BLOCK`
/// (65,536) fixed-size tuples, so "hitting the cache" meant allocating
/// and memcpy-ing the entire block anyway. A cache that copies its
/// payload on every hit is not a cache, it is a copy with extra steps.
///
/// Measured on the live 235 GiB v5 brain (perf, `dog`, warm page
/// cache): `Vec::clone` + `malloc` + `free` accounted for ~11.7% of
/// all query CPU, on top of the ~30% spent re-deriving SHA-256
/// keystreams for blocks that were evicted between uses. Handing back
/// an `Arc` makes a hit O(1) and keeps the bytes alive for the
/// iterator's lifetime regardless of LRU eviction -- exactly the
/// discipline `compress.rs`'s older `ArcArchive` cache already used
/// and which this reader regressed from.
struct DecodedBlock {
    /// Sorted-by-(subject, predicate, object) entries from this
    /// block. Linear scan is correct because blocks are bounded
    /// (the writer's `target_entries_per_block` clamps to
    /// `MAX_TARGET_ENTRIES_PER_BLOCK = 65536`); binary search
    /// inside a block would add code surface for marginal gain
    /// at the typical 1000-entry block size.
    entries: Vec<(NodeId, u8, NodeId, BondPayload)>,
}

/// LRU block cache. Bounded by entry count -- the simplest
/// discipline that catches Vec-the-world bugs. Keyed on
/// `block_id: u32`; matches the NodeSstReader cache shape exactly.
struct BlockCacheInner {
    /// Most-recently-used at the BACK; least-recently-used at the
    /// FRONT. Each entry is `(block_id, DecodedBlock)`.
    entries: VecDeque<(u32, Arc<DecodedBlock>)>,
    cap: usize,
}

impl BlockCacheInner {
    fn new(cap: usize) -> Self {
        BlockCacheInner {
            entries: VecDeque::with_capacity(cap.max(1)),
            cap: cap.max(1),
        }
    }

    /// Move the entry for `block_id` to the MRU position; return an
    /// `Arc` handle if present, else None. See OP-SST-CACHE-1 on the
    /// cache struct: this used to deep-copy the block.
    fn get(&mut self, block_id: u32) -> Option<Arc<DecodedBlock>> {
        let pos = self.entries.iter().position(|(id, _)| *id == block_id)?;
        let entry = self.entries.remove(pos)?;
        let out = entry.1.clone();
        self.entries.push_back(entry);
        Some(out)
    }

    /// Insert `block_id -> block` at MRU. Evicts LRU entries until
    /// `entries.len() <= cap`, but always leaves at least one
    /// entry resident (matches NodeSstReader discipline).
    fn insert(&mut self, block_id: u32, block: Arc<DecodedBlock>) {
        // Remove any existing entry for this block_id (last-writer-wins).
        if let Some(pos) = self.entries.iter().position(|(id, _)| *id == block_id) {
            self.entries.remove(pos);
        }
        self.entries.push_back((block_id, block));
        while self.entries.len() > self.cap && self.entries.len() > 1 {
            self.entries.pop_front();
        }
    }
}

/// On-demand reader for a BondSST file.
pub struct BondSstReader {
    path: PathBuf,
    file: Mutex<File>,
    file_uid: u64,
    block_index: Vec<BlockIndexEntry>,
    file_size: u64,
    cache: Mutex<BlockCacheInner>,
    stats: Mutex<BondSstReadStats>,
    /// OP-BRAIN-V5 6.5: decryption key. `None` = expect plaintext.
    key: Option<MasterKey>,
    /// Header FLAG_ENCRYPTED bit.
    encrypted: bool,
    /// Per-SST IV from the header (zeros when plaintext).
    iv: [u8; 16],
}

impl BondSstReader {
    /// Open `path` and read the header + footer ONLY. No block
    /// reads. Uses [`DEFAULT_BLOCK_CACHE_CAP`].
    pub fn open(path: &Path) -> Result<Self, BondSstError> {
        Self::open_with_cache_cap_and_key(path, DEFAULT_BLOCK_CACHE_CAP, None)
    }

    /// Open with an explicit block-cache cap (plaintext / no key).
    pub fn open_with_cache_cap(
        path: &Path,
        cache_cap_blocks: usize,
    ) -> Result<Self, BondSstError> {
        Self::open_with_cache_cap_and_key(path, cache_cap_blocks, None)
    }

    /// Open with a decryption key (OP-BRAIN-V5 6.5). Required to read
    /// an encrypted SST; `None` reads plaintext only -- an encrypted
    /// SST decoded without a key fails cleanly with
    /// [`BondSstError::EncryptedButNoKey`].
    pub fn open_with_key(path: &Path, key: Option<MasterKey>) -> Result<Self, BondSstError> {
        Self::open_with_cache_cap_and_key(path, DEFAULT_BLOCK_CACHE_CAP, key)
    }

    /// Open with an explicit block-cache cap AND a decryption key.
    pub fn open_with_cache_cap_and_key(
        path: &Path,
        cache_cap_blocks: usize,
        key: Option<MasterKey>,
    ) -> Result<Self, BondSstError> {
        let file = File::open(path)?;
        let file_size = file.metadata()?.len();
        if file_size < (HEADER_BYTES + TRAILER_BYTES) as u64 {
            return Err(BondSstError::Format(format!(
                "file too small to be a BondSST: {} bytes",
                file_size
            )));
        }

        // --- Header --------------------------------------------------
        let header = read_header_validated(&file)?;

        // --- Footer --------------------------------------------------
        if header.footer_offset >= file_size {
            return Err(BondSstError::Format(format!(
                "header.footer_offset {} >= file_size {}",
                header.footer_offset, file_size
            )));
        }
        let block_index = read_footer_validated(&file, header.footer_offset, header.block_count)?;

        // --- Trailer (sanity) ----------------------------------------
        read_trailer_validated(&file, file_size, header.footer_offset)?;

        Ok(BondSstReader {
            path: path.to_path_buf(),
            file: Mutex::new(file),
            file_uid: header.file_uid,
            block_index,
            file_size,
            cache: Mutex::new(BlockCacheInner::new(cache_cap_blocks)),
            stats: Mutex::new(BondSstReadStats::default()),
            key,
            encrypted: header.flags & FLAG_ENCRYPTED != 0,
            iv: header.iv,
        })
    }

    /// Return the path the reader was opened against.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Stable file identifier passed by the writer.
    pub fn file_uid(&self) -> u64 {
        self.file_uid
    }

    /// Number of blocks in the SST.
    pub fn block_count(&self) -> u32 {
        self.block_index.len() as u32
    }

    /// Snapshot of read-side statistics.
    pub fn stats(&self) -> BondSstReadStats {
        *self.stats.lock().expect("stats lock poisoned")
    }

    /// Full-SST scan -- yields every `(subject, predicate, object,
    /// payload)` quadruple in ascending triple-key order.
    /// Convenience over [`Self::iter_range`] for callers (notably
    /// the Phase 3.4 `BondCompactionWorker` and the Phase 3.x
    /// migration tool) that want to stream the whole SST without
    /// picking sentinel bounds. Implemented as
    /// `iter_range((NodeId(0), 0, NodeId(0))..(NodeId(u64::MAX),
    /// 0, NodeId(0)))`; the upper bound treats the subject u64
    /// as the dominant sort key and `u64::MAX` is reserved (no
    /// real NodeId ever reaches it).
    pub fn iter_all(&self) -> BondSstRangeIter<'_> {
        self.iter_range(
            (NodeId(0), 0, NodeId(0)),
            (NodeId(u64::MAX), 0, NodeId(0)),
        )
    }

    /// Convenience: iterate all bonds with the given subject.
    /// Equivalent to `iter_range((subject, 0, NodeId(0)) ..
    /// (subject+1, 0, NodeId(0)))`. The `subject+1` upper bound
    /// saturates at `u64::MAX` so calling with `NodeId(u64::MAX)`
    /// degrades to iterating from `(u64::MAX, 0, 0)` to the same
    /// point and yields nothing (acceptable; production NodeIds
    /// never reach u64::MAX, see the reserved-sentinel discipline
    /// at [`Self::iter_all`]).
    pub fn iter_subject(&self, subject: NodeId) -> BondSstRangeIter<'_> {
        let next_subject = subject.0.saturating_add(1);
        self.iter_range(
            (subject, 0, NodeId(0)),
            (NodeId(next_subject), 0, NodeId(0)),
        )
    }

    /// Range scan `[start, end)` over the triple key
    /// `(subject, predicate, object)`. Yields entries in ascending
    /// triple order. Loads blocks on demand via the LRU cache.
    pub fn iter_range(
        &self,
        start: (NodeId, u8, NodeId),
        end: (NodeId, u8, NodeId),
    ) -> BondSstRangeIter<'_> {
        let start_block = if self.block_index.is_empty() {
            None
        } else {
            // Find the first block whose LAST key is >= start. The
            // block-index entries are sorted by first-triple ascending
            // (writer guarantees monotonic append + non-overlapping
            // block ranges).
            let mut lo = 0usize;
            let mut hi = self.block_index.len();
            while lo < hi {
                let mid = (lo + hi) / 2;
                let last_triple = (
                    self.block_index[mid].last_subject,
                    self.block_index[mid].last_predicate,
                    self.block_index[mid].last_object,
                );
                if triple_lt(last_triple, start) {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            if lo >= self.block_index.len() {
                None
            } else {
                Some(lo as u32)
            }
        };
        BondSstRangeIter {
            reader: self,
            start,
            end,
            next_block: start_block,
            current_block: None,
            current_idx: 0,
        }
    }

    // --- internals -----------------------------------------------------

    fn fetch_block(&self, block_id: u32) -> Result<Arc<DecodedBlock>, BondSstError> {
        // Fast path: cache.
        if let Some(b) = self.cache.lock().expect("cache lock poisoned").get(block_id) {
            let mut s = self.stats.lock().expect("stats lock poisoned");
            s.block_cache_hits += 1;
            return Ok(b);
        }
        // Slow path: decode from disk.
        let block = Arc::new(self.decode_block_from_disk(block_id)?);
        {
            let mut s = self.stats.lock().expect("stats lock poisoned");
            s.block_cache_misses += 1;
        }
        self.cache
            .lock()
            .expect("cache lock poisoned")
            .insert(block_id, block.clone());
        Ok(block)
    }

    fn decode_block_from_disk(&self, block_id: u32) -> Result<DecodedBlock, BondSstError> {
        let entry = *self
            .block_index
            .get(block_id as usize)
            .ok_or_else(|| BondSstError::Format(format!("block_id {} out of range", block_id)))?;

        // Sanity: the block must fit inside the file.
        let end = entry
            .block_offset
            .checked_add(entry.block_size as u64)
            .ok_or_else(|| {
                BondSstError::Format(format!("block {} offset+size overflow", block_id))
            })?;
        if end > self.file_size {
            return Err(BondSstError::Format(format!(
                "block {} extends past EOF: end={}, file_size={}",
                block_id, end, self.file_size
            )));
        }

        // Read the whole block (header + body) in one syscall under
        // the file lock. Decoding happens after the lock is released.
        let mut buf = vec![0u8; entry.block_size as usize];
        {
            let mut f = self.file.lock().expect("file lock poisoned");
            f.seek(SeekFrom::Start(entry.block_offset))?;
            f.read_exact(&mut buf)?;
        }
        {
            let mut s = self.stats.lock().expect("stats lock poisoned");
            s.blocks_read += 1;
            s.disk_bytes_read += buf.len() as u64;
        }

        // Parse block header.
        if buf.len() < BLOCK_HEADER_BYTES {
            return Err(BondSstError::Format(format!(
                "block {} on-disk size {} < header size {}",
                block_id,
                buf.len(),
                BLOCK_HEADER_BYTES
            )));
        }
        let entry_count = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
        let uncompressed_size = u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]);
        let compressed_size = u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]);
        let compression_kind_byte = buf[12];
        let block_crc = u32::from_le_bytes([buf[13], buf[14], buf[15], buf[16]]);

        let on_disk_body_len = (compressed_size as usize)
            .checked_add(BLOCK_HEADER_BYTES)
            .ok_or_else(|| {
                BondSstError::Format(format!(
                    "block {} compressed_size+header overflow",
                    block_id
                ))
            })?;
        if on_disk_body_len != buf.len() {
            return Err(BondSstError::Format(format!(
                "block {} on-disk size mismatch: header says {} body bytes (+ {} header), got {} total",
                block_id,
                compressed_size,
                BLOCK_HEADER_BYTES,
                buf.len()
            )));
        }
        let compressed_body = &buf[BLOCK_HEADER_BYTES..];

        // Validate block CRC.
        let computed = crc32fast::hash(compressed_body);
        if computed != block_crc {
            return Err(BondSstError::CrcMismatch {
                scope: CrcScope::Block { block_id },
                expected: block_crc,
                observed: computed,
            });
        }

        // Decompress.
        let compression = CompressionKind::from_u8(compression_kind_byte)
            .ok_or(BondSstError::UnknownCompressionKind {
                observed: compression_kind_byte,
            })?;
        let mut uncompressed: Vec<u8> = match compression {
            CompressionKind::None => compressed_body.to_vec(),
            CompressionKind::Zstd { .. } => zstd::decode_all(compressed_body)
                .map_err(|e| BondSstError::Zstd(format!("decode block {}: {}", block_id, e)))?,
        };
        if uncompressed.len() as u64 != uncompressed_size as u64 {
            return Err(BondSstError::Format(format!(
                "block {} decompressed size {} != header.uncompressed_size {}",
                block_id,
                uncompressed.len(),
                uncompressed_size
            )));
        }

        // OP-BRAIN-V5 6.5: decrypt (XOR-keystream) if encrypted. Mirror
        // of v4 `sst.rs::decompress_block`: on-disk order is
        // compress(encrypt(plaintext)), so read order is decompress
        // (above) THEN decrypt here.
        if self.encrypted {
            let k = self.key.as_ref().ok_or(BondSstError::EncryptedButNoKey)?;
            let ksi = derive_block_keystream_index(self.file_uid, block_id, &self.iv);
            xor_in_place(k, ksi, &mut uncompressed);
        }

        // Parse entries. Each entry: subject u64 + predicate u8 +
        // object u64 + payload_size u32 (21 B framing) + bincode payload.
        let mut entries: Vec<(NodeId, u8, NodeId, BondPayload)> =
            Vec::with_capacity(entry_count as usize);
        let mut cursor = 0usize;
        for _ in 0..entry_count {
            if cursor + ENTRY_HEADER_BYTES > uncompressed.len() {
                return Err(BondSstError::Format(format!(
                    "block {} truncated mid-entry header",
                    block_id
                )));
            }
            let subject = u64::from_le_bytes([
                uncompressed[cursor],
                uncompressed[cursor + 1],
                uncompressed[cursor + 2],
                uncompressed[cursor + 3],
                uncompressed[cursor + 4],
                uncompressed[cursor + 5],
                uncompressed[cursor + 6],
                uncompressed[cursor + 7],
            ]);
            let predicate = uncompressed[cursor + 8];
            let object = u64::from_le_bytes([
                uncompressed[cursor + 9],
                uncompressed[cursor + 10],
                uncompressed[cursor + 11],
                uncompressed[cursor + 12],
                uncompressed[cursor + 13],
                uncompressed[cursor + 14],
                uncompressed[cursor + 15],
                uncompressed[cursor + 16],
            ]);
            let payload_size = u32::from_le_bytes([
                uncompressed[cursor + 17],
                uncompressed[cursor + 18],
                uncompressed[cursor + 19],
                uncompressed[cursor + 20],
            ]) as usize;
            cursor += ENTRY_HEADER_BYTES;
            if cursor + payload_size > uncompressed.len() {
                return Err(BondSstError::Format(format!(
                    "block {} truncated mid-entry payload",
                    block_id
                )));
            }
            let payload: BondPayload =
                bincode::deserialize(&uncompressed[cursor..cursor + payload_size]).map_err(
                    |e| {
                        BondSstError::Bincode(format!(
                            "deserialize BondPayload block {}: {}",
                            block_id, e
                        ))
                    },
                )?;
            cursor += payload_size;
            entries.push((NodeId(subject), predicate, NodeId(object), payload));
        }
        if cursor != uncompressed.len() {
            return Err(BondSstError::Format(format!(
                "block {} has {} trailing bytes after {} entries",
                block_id,
                uncompressed.len() - cursor,
                entry_count
            )));
        }

        // Confirm against block_index claims (sanity).
        if let Some((s, p, o, _)) = entries.first() {
            if *s != entry.first_subject
                || *p != entry.first_predicate
                || *o != entry.first_object
            {
                return Err(BondSstError::Format(format!(
                    "block {} entries[0]=({},{},{}) != index.first=({},{},{})",
                    block_id,
                    s.0,
                    p,
                    o.0,
                    entry.first_subject.0,
                    entry.first_predicate,
                    entry.first_object.0
                )));
            }
        }
        if let Some((s, p, o, _)) = entries.last() {
            if *s != entry.last_subject
                || *p != entry.last_predicate
                || *o != entry.last_object
            {
                return Err(BondSstError::Format(format!(
                    "block {} entries[last]=({},{},{}) != index.last=({},{},{})",
                    block_id,
                    s.0,
                    p,
                    o.0,
                    entry.last_subject.0,
                    entry.last_predicate,
                    entry.last_object.0
                )));
            }
        }

        Ok(DecodedBlock { entries })
    }
}

// ----------------------------------------------------------------------
// Range iterator.
// ----------------------------------------------------------------------

/// Streaming iterator returned by [`BondSstReader::iter_range`],
/// [`BondSstReader::iter_subject`], and [`BondSstReader::iter_all`].
pub struct BondSstRangeIter<'a> {
    reader: &'a BondSstReader,
    start: (NodeId, u8, NodeId),
    end: (NodeId, u8, NodeId),
    /// Next block to load (None means iteration is exhausted).
    next_block: Option<u32>,
    /// In-flight decoded block.
    current_block: Option<Arc<DecodedBlock>>,
    /// Cursor inside `current_block.entries`.
    current_idx: usize,
}

impl<'a> Iterator for BondSstRangeIter<'a> {
    type Item = Result<(NodeId, u8, NodeId, BondPayload), BondSstError>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // Drain the current block.
            if let Some(blk) = &self.current_block {
                while self.current_idx < blk.entries.len() {
                    let (s, p, o, payload) = &blk.entries[self.current_idx];
                    self.current_idx += 1;
                    let key = (*s, *p, *o);
                    if triple_lt(key, self.start) {
                        continue;
                    }
                    if !triple_lt(key, self.end) {
                        // Past the end -- iteration done.
                        self.next_block = None;
                        self.current_block = None;
                        return None;
                    }
                    {
                        let mut st = self
                            .reader
                            .stats
                            .lock()
                            .expect("stats lock poisoned");
                        st.entries_yielded += 1;
                    }
                    return Some(Ok((*s, *p, *o, payload.clone())));
                }
                // Block exhausted; clear and continue to load next.
                self.current_block = None;
                self.current_idx = 0;
            }

            // Load next block.
            let bid = self.next_block?;
            if (bid as usize) >= self.reader.block_index.len() {
                self.next_block = None;
                return None;
            }
            // Bail early if the next block's first triple is already
            // past end.
            let first_triple = (
                self.reader.block_index[bid as usize].first_subject,
                self.reader.block_index[bid as usize].first_predicate,
                self.reader.block_index[bid as usize].first_object,
            );
            if !triple_lt(first_triple, self.end) {
                self.next_block = None;
                return None;
            }
            match self.reader.fetch_block(bid) {
                Ok(b) => {
                    self.current_block = Some(b);
                    self.current_idx = 0;
                    self.next_block = Some(bid + 1);
                }
                Err(e) => {
                    self.next_block = None;
                    return Some(Err(e));
                }
            }
        }
    }
}

// ----------------------------------------------------------------------
// Header / footer / trailer parsers.
// ----------------------------------------------------------------------

/// Parsed and validated header.
#[derive(Debug, Clone, Copy)]
struct ParsedHeader {
    block_count: u32,
    footer_offset: u64,
    file_uid: u64,
    flags: u8,
    iv: [u8; 16],
}

fn read_header_validated(mut file: &File) -> Result<ParsedHeader, BondSstError> {
    let mut buf = [0u8; HEADER_BYTES];
    file.seek(SeekFrom::Start(0))?;
    file.read_exact(&mut buf)?;

    let magic = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
    if magic != BSST_MAGIC {
        return Err(BondSstError::MagicMismatch { observed: magic });
    }
    let format_version = u16::from_le_bytes([buf[4], buf[5]]);
    if format_version != BSST_FORMAT_VERSION {
        return Err(BondSstError::FormatVersionUnsupported {
            observed: format_version,
        });
    }
    let block_count = u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]);
    let footer_offset = u64::from_le_bytes([
        buf[12], buf[13], buf[14], buf[15], buf[16], buf[17], buf[18], buf[19],
    ]);
    let file_uid = u64::from_le_bytes([
        buf[20], buf[21], buf[22], buf[23], buf[24], buf[25], buf[26], buf[27],
    ]);
    let flags = buf[6];
    let mut iv = [0u8; 16];
    iv.copy_from_slice(&buf[28..44]);
    let header_crc = u32::from_le_bytes([buf[44], buf[45], buf[46], buf[47]]);
    let computed = crc32fast::hash(&buf[..44]);
    if computed != header_crc {
        return Err(BondSstError::CrcMismatch {
            scope: CrcScope::Header,
            expected: header_crc,
            observed: computed,
        });
    }

    Ok(ParsedHeader {
        block_count,
        footer_offset,
        file_uid,
        flags,
        iv,
    })
}

fn read_footer_validated(
    mut file: &File,
    footer_offset: u64,
    expected_block_count: u32,
) -> Result<Vec<BlockIndexEntry>, BondSstError> {
    let index_bytes_total = (expected_block_count as usize)
        .checked_mul(INDEX_ENTRY_BYTES)
        .ok_or_else(|| {
            BondSstError::Format(format!(
                "block_count {} overflows index bytes",
                expected_block_count
            ))
        })?;
    // Footer = block_index + 4 B index_byte_len + 4 B footer_crc32.
    let footer_bytes_total = index_bytes_total + 4 + 4;
    let mut buf = vec![0u8; footer_bytes_total];
    file.seek(SeekFrom::Start(footer_offset))?;
    file.read_exact(&mut buf)?;

    let index_byte_len_disk = u32::from_le_bytes([
        buf[index_bytes_total],
        buf[index_bytes_total + 1],
        buf[index_bytes_total + 2],
        buf[index_bytes_total + 3],
    ]);
    if index_byte_len_disk as usize != index_bytes_total {
        return Err(BondSstError::Format(format!(
            "footer index_byte_len {} != expected {}",
            index_byte_len_disk, index_bytes_total
        )));
    }
    let footer_crc_disk = u32::from_le_bytes([
        buf[index_bytes_total + 4],
        buf[index_bytes_total + 5],
        buf[index_bytes_total + 6],
        buf[index_bytes_total + 7],
    ]);
    let computed = crc32fast::hash(&buf[..index_bytes_total]);
    if computed != footer_crc_disk {
        return Err(BondSstError::CrcMismatch {
            scope: CrcScope::Footer,
            expected: footer_crc_disk,
            observed: computed,
        });
    }

    let mut block_index = Vec::with_capacity(expected_block_count as usize);
    for i in 0..(expected_block_count as usize) {
        let off = i * INDEX_ENTRY_BYTES;
        // first_triple (17 B): subject u64 + predicate u8 + object u64
        let first_subject = u64::from_le_bytes([
            buf[off],
            buf[off + 1],
            buf[off + 2],
            buf[off + 3],
            buf[off + 4],
            buf[off + 5],
            buf[off + 6],
            buf[off + 7],
        ]);
        let first_predicate = buf[off + 8];
        let first_object = u64::from_le_bytes([
            buf[off + 9],
            buf[off + 10],
            buf[off + 11],
            buf[off + 12],
            buf[off + 13],
            buf[off + 14],
            buf[off + 15],
            buf[off + 16],
        ]);
        // last_triple (17 B): subject u64 + predicate u8 + object u64
        let last_subject = u64::from_le_bytes([
            buf[off + 17],
            buf[off + 18],
            buf[off + 19],
            buf[off + 20],
            buf[off + 21],
            buf[off + 22],
            buf[off + 23],
            buf[off + 24],
        ]);
        let last_predicate = buf[off + 25];
        let last_object = u64::from_le_bytes([
            buf[off + 26],
            buf[off + 27],
            buf[off + 28],
            buf[off + 29],
            buf[off + 30],
            buf[off + 31],
            buf[off + 32],
            buf[off + 33],
        ]);
        let block_offset = u64::from_le_bytes([
            buf[off + 34],
            buf[off + 35],
            buf[off + 36],
            buf[off + 37],
            buf[off + 38],
            buf[off + 39],
            buf[off + 40],
            buf[off + 41],
        ]);
        let block_size = u32::from_le_bytes([
            buf[off + 42],
            buf[off + 43],
            buf[off + 44],
            buf[off + 45],
        ]);
        let first_triple = (NodeId(first_subject), first_predicate, NodeId(first_object));
        let last_triple = (NodeId(last_subject), last_predicate, NodeId(last_object));
        if triple_lt(last_triple, first_triple) {
            return Err(BondSstError::Format(format!(
                "block_index[{}] first=({},{},{}) > last=({},{},{})",
                i,
                first_subject,
                first_predicate,
                first_object,
                last_subject,
                last_predicate,
                last_object
            )));
        }
        if i > 0 {
            let prev = &block_index[i - 1] as &BlockIndexEntry;
            let prev_last = (prev.last_subject, prev.last_predicate, prev.last_object);
            if !triple_lt(prev_last, first_triple) {
                return Err(BondSstError::Format(format!(
                    "block_index[{}] first=({},{},{}) overlaps prev.last=({},{},{})",
                    i,
                    first_subject,
                    first_predicate,
                    first_object,
                    prev.last_subject.0,
                    prev.last_predicate,
                    prev.last_object.0
                )));
            }
        }
        block_index.push(BlockIndexEntry {
            first_subject: NodeId(first_subject),
            first_predicate,
            first_object: NodeId(first_object),
            last_subject: NodeId(last_subject),
            last_predicate,
            last_object: NodeId(last_object),
            block_offset,
            block_size,
        });
    }
    Ok(block_index)
}

fn read_trailer_validated(
    mut file: &File,
    file_size: u64,
    expected_footer_offset: u64,
) -> Result<(), BondSstError> {
    let trailer_offset = file_size - TRAILER_BYTES as u64;
    let mut buf = [0u8; TRAILER_BYTES];
    file.seek(SeekFrom::Start(trailer_offset))?;
    file.read_exact(&mut buf)?;
    let magic = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
    if magic != BSST_MAGIC {
        return Err(BondSstError::MagicMismatch { observed: magic });
    }
    let offset_tail = u64::from_le_bytes([
        buf[4], buf[5], buf[6], buf[7], buf[8], buf[9], buf[10], buf[11],
    ]);
    if offset_tail != expected_footer_offset {
        return Err(BondSstError::Format(format!(
            "trailer footer_offset {} != header footer_offset {}",
            offset_tail, expected_footer_offset
        )));
    }
    Ok(())
}

/// Lexicographic strict-less-than on the (subject, predicate,
/// object) triple. Matches the writer's `triple_lt` (`src/bond_store/
/// sst_writer.rs`); duplicated here rather than re-exported so the
/// reader has no dependency on writer-private helpers.
fn triple_lt(a: (NodeId, u8, NodeId), b: (NodeId, u8, NodeId)) -> bool {
    if a.0.0 != b.0.0 {
        return a.0.0 < b.0.0;
    }
    if a.1 != b.1 {
        return a.1 < b.1;
    }
    a.2.0 < b.2.0
}

// ----------------------------------------------------------------------
// Tests.
// ----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bond::{KeyframeSource, TemporalKeyframe};
    use crate::bond_store::sst_writer::{BondSstWriteOpts, BondSstWriter};
    use std::fs::OpenOptions;
    use std::io::Write;
    use tempfile::tempdir;

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

    fn payloads_equal(a: &BondPayload, b: &BondPayload) -> bool {
        a.weight == b.weight
            && a.confidence == b.confidence
            && a.seq == b.seq
            && a.keyframes.len() == b.keyframes.len()
            && a.keyframes
                .iter()
                .zip(b.keyframes.iter())
                .all(|(x, y)| {
                    x.timestamp == y.timestamp
                        && (x.weight - y.weight).abs() < 1e-6
                        // KeyframeSource equality via debug repr (no Eq derived).
                        && format!("{:?}", x.source) == format!("{:?}", y.source)
                })
    }

    /// A. Empty file open.
    #[test]
    fn empty_file_open() {
        let tmp = tempdir().unwrap();
        let path = tmp.path().join("empty.bsst");
        let w = BondSstWriter::create(
            &path,
            BondSstWriteOpts {
                file_uid: 99,
                target_entries_per_block: 64,
                compression: CompressionKind::Zstd { level: 1 },
            },
        )
        .unwrap();
        let stats = w.finish().unwrap();
        assert_eq!(stats.entries_written, 0);
        assert_eq!(stats.blocks_written, 0);

        let r = BondSstReader::open(&path).unwrap();
        assert_eq!(r.file_uid(), 99);
        assert_eq!(r.block_count(), 0);
        let collected: Vec<_> = r.iter_all().collect();
        assert_eq!(collected.len(), 0);
        let s = r.stats();
        assert_eq!(s.blocks_read, 0, "no blocks should be read for empty iter");
        assert_eq!(s.entries_yielded, 0);
    }

    /// B. Single-block round-trip. 500 monotonic bonds.
    #[test]
    fn single_block_round_trip() {
        let tmp = tempdir().unwrap();
        let path = tmp.path().join("single.bsst");
        let mut w = BondSstWriter::create(
            &path,
            BondSstWriteOpts {
                file_uid: 7,
                target_entries_per_block: 500,
                compression: CompressionKind::Zstd { level: 1 },
            },
        )
        .unwrap();
        for i in 0u64..500 {
            w.append(NodeId(i), 1, NodeId(i + 1_000_000), &mk_payload(i))
                .unwrap();
        }
        let stats = w.finish().unwrap();
        assert_eq!(stats.entries_written, 500);
        assert_eq!(stats.blocks_written, 1);

        let r = BondSstReader::open(&path).unwrap();
        assert_eq!(r.file_uid(), 7);
        assert_eq!(r.block_count(), 1);

        let mut items: Vec<(NodeId, u8, NodeId, BondPayload)> = Vec::new();
        for it in r.iter_all() {
            items.push(it.unwrap());
        }
        assert_eq!(items.len(), 500);
        // Monotonic ordering.
        for w_idx in 1..items.len() {
            let prev = (items[w_idx - 1].0, items[w_idx - 1].1, items[w_idx - 1].2);
            let cur = (items[w_idx].0, items[w_idx].1, items[w_idx].2);
            assert!(triple_lt(prev, cur), "iter_all must be monotonic");
        }
        // Payload byte-equal sample.
        for (i, item) in items.iter().enumerate() {
            assert_eq!(item.0, NodeId(i as u64));
            assert_eq!(item.1, 1);
            assert_eq!(item.2, NodeId(i as u64 + 1_000_000));
            let expected = mk_payload(i as u64);
            assert!(
                payloads_equal(&item.3, &expected),
                "payload mismatch at i={}",
                i
            );
        }
    }

    /// C. Multi-block range-scan crossing block boundaries.
    /// 10K bonds, 1000 per block = 10 blocks. iter_range over a
    /// middle range. Verify only the necessary blocks were loaded.
    #[test]
    fn multi_block_range_scan_bounded_loads() {
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
        for i in 0u64..10_000 {
            // Single predicate so subject is the dominant sort key.
            w.append(NodeId(i), 0, NodeId(i + 2_000_000), &mk_payload(i))
                .unwrap();
        }
        let stats = w.finish().unwrap();
        assert_eq!(stats.blocks_written, 10);

        let r = BondSstReader::open(&path).unwrap();
        assert_eq!(r.block_count(), 10);

        // Scan subjects 3500..4500 -- crosses the block-3 / block-4
        // boundary (block N holds subjects [N*1000, N*1000+999]).
        let start = (NodeId(3500), 0, NodeId(0));
        let end = (NodeId(4500), 0, NodeId(0));
        let mut count = 0u64;
        let mut last_key: Option<(NodeId, u8, NodeId)> = None;
        for it in r.iter_range(start, end) {
            let (s, p, o, _) = it.unwrap();
            if let Some(prev) = last_key {
                assert!(triple_lt(prev, (s, p, o)), "monotonic");
            }
            assert!((3500..4500).contains(&s.0), "subject {} out of range", s.0);
            last_key = Some((s, p, o));
            count += 1;
        }
        assert_eq!(count, 1000, "scan must yield exactly 1000 entries");
        let s = r.stats();
        // Only blocks 3 and 4 should have been loaded.
        assert!(
            s.blocks_read <= 2,
            "iter_range loaded {} blocks; expected at most 2",
            s.blocks_read
        );
        assert!(s.blocks_read >= 1, "iter_range must read at least one block");
        assert_eq!(s.entries_yielded, 1000);
    }

    /// D. iter_subject convenience helper. 4 subjects each with
    /// 100 bonds. iter_subject(NodeId(2)) must yield exactly 100.
    #[test]
    fn iter_subject_yields_only_target_subject() {
        let tmp = tempdir().unwrap();
        let path = tmp.path().join("iter_sub.bsst");
        let mut w = BondSstWriter::create(
            &path,
            BondSstWriteOpts {
                file_uid: 21,
                target_entries_per_block: 64,
                compression: CompressionKind::Zstd { level: 1 },
            },
        )
        .unwrap();
        // Subjects 1, 2, 3, 4; for each: 100 bonds with predicate 0,
        // ascending object. Triples are monotonic globally.
        for subj in 1u64..=4 {
            for obj in 0u64..100 {
                w.append(NodeId(subj), 0, NodeId(obj), &mk_payload(subj * 1000 + obj))
                    .unwrap();
            }
        }
        let stats = w.finish().unwrap();
        assert_eq!(stats.entries_written, 400);

        let r = BondSstReader::open(&path).unwrap();
        let mut count = 0u64;
        for it in r.iter_subject(NodeId(2)) {
            let (s, _, _, _) = it.unwrap();
            assert_eq!(s, NodeId(2), "iter_subject must yield only subject 2");
            count += 1;
        }
        assert_eq!(count, 100, "iter_subject(2) must yield exactly 100");
    }

    /// E. Empty range yields nothing -- iter_range over a gap.
    #[test]
    fn empty_range_yields_nothing() {
        let tmp = tempdir().unwrap();
        let path = tmp.path().join("gap.bsst");
        let mut w = BondSstWriter::create(
            &path,
            BondSstWriteOpts {
                file_uid: 31,
                target_entries_per_block: 64,
                compression: CompressionKind::None,
            },
        )
        .unwrap();
        // Write subjects 1..10 only; query the gap at subject 50..60.
        for i in 1u64..=10 {
            w.append(NodeId(i), 0, NodeId(0), &mk_payload(i)).unwrap();
        }
        w.finish().unwrap();

        let r = BondSstReader::open(&path).unwrap();
        let mut count = 0u64;
        for _ in r.iter_range((NodeId(50), 0, NodeId(0)), (NodeId(60), 0, NodeId(0))) {
            count += 1;
        }
        assert_eq!(count, 0);
    }

    /// F. Out-of-range query (before all keys).
    #[test]
    fn out_of_range_before_all_keys() {
        let tmp = tempdir().unwrap();
        let path = tmp.path().join("before.bsst");
        let mut w = BondSstWriter::create(
            &path,
            BondSstWriteOpts {
                file_uid: 41,
                target_entries_per_block: 64,
                compression: CompressionKind::None,
            },
        )
        .unwrap();
        for i in 1000u64..1010 {
            w.append(NodeId(i), 0, NodeId(0), &mk_payload(i)).unwrap();
        }
        w.finish().unwrap();

        let r = BondSstReader::open(&path).unwrap();
        let count: u64 = r
            .iter_range((NodeId(0), 0, NodeId(0)), (NodeId(500), 0, NodeId(0)))
            .count() as u64;
        assert_eq!(count, 0);
    }

    /// G. Out-of-range query (after all keys).
    #[test]
    fn out_of_range_after_all_keys() {
        let tmp = tempdir().unwrap();
        let path = tmp.path().join("after.bsst");
        let mut w = BondSstWriter::create(
            &path,
            BondSstWriteOpts {
                file_uid: 51,
                target_entries_per_block: 64,
                compression: CompressionKind::None,
            },
        )
        .unwrap();
        for i in 0u64..10 {
            w.append(NodeId(i), 0, NodeId(0), &mk_payload(i)).unwrap();
        }
        w.finish().unwrap();

        let r = BondSstReader::open(&path).unwrap();
        let count: u64 = r
            .iter_range(
                (NodeId(1000), 0, NodeId(0)),
                (NodeId(2000), 0, NodeId(0)),
            )
            .count() as u64;
        assert_eq!(count, 0);
    }

    /// H. Corrupt header rejection -- flip a byte in the magic.
    #[test]
    fn corrupt_header_rejected() {
        let tmp = tempdir().unwrap();
        let path = tmp.path().join("hdr.bsst");
        let mut w = BondSstWriter::create(
            &path,
            BondSstWriteOpts {
                file_uid: 0,
                target_entries_per_block: 64,
                compression: CompressionKind::None,
            },
        )
        .unwrap();
        w.append(NodeId(1), 0, NodeId(2), &mk_payload(0)).unwrap();
        w.finish().unwrap();

        // Flip the first magic byte.
        {
            let mut f = OpenOptions::new().write(true).open(&path).unwrap();
            f.seek(SeekFrom::Start(0)).unwrap();
            f.write_all(&[0x00]).unwrap();
            f.sync_all().unwrap();
        }

        match BondSstReader::open(&path) {
            Err(BondSstError::MagicMismatch { .. }) => {}
            Err(e) => panic!("expected MagicMismatch, got Err({:?})", e),
            Ok(_) => panic!("expected MagicMismatch, got Ok"),
        }
    }

    /// I. Corrupt block CRC rejection. Open succeeds (header +
    /// footer intact); iter_range over the corrupted block returns
    /// Err(CrcMismatch).
    #[test]
    fn corrupt_block_crc_rejected_on_iter() {
        let tmp = tempdir().unwrap();
        let path = tmp.path().join("crc.bsst");
        let mut w = BondSstWriter::create(
            &path,
            BondSstWriteOpts {
                file_uid: 0,
                target_entries_per_block: 50,
                // Raw so the byte we flip is unambiguously in the
                // entry stream.
                compression: CompressionKind::None,
            },
        )
        .unwrap();
        for i in 0u64..200 {
            w.append(NodeId(i), 0, NodeId(i + 1), &mk_payload(i))
                .unwrap();
        }
        w.finish().unwrap();

        // Learn block 2's offset, then corrupt one byte in its body.
        let r = BondSstReader::open(&path).unwrap();
        assert!(r.block_count() >= 4);
        let target_block = 2u32;
        let target_offset =
            r.block_index[target_block as usize].block_offset + BLOCK_HEADER_BYTES as u64 + 10;
        drop(r);

        {
            let mut f = OpenOptions::new().write(true).open(&path).unwrap();
            f.seek(SeekFrom::Start(target_offset)).unwrap();
            let original = {
                let mut rf = File::open(&path).unwrap();
                rf.seek(SeekFrom::Start(target_offset)).unwrap();
                let mut b = [0u8; 1];
                rf.read_exact(&mut b).unwrap();
                b[0]
            };
            f.write_all(&[original ^ 0xFF]).unwrap();
            f.sync_all().unwrap();
        }

        // Iterate from block 2's first subject -- block 2 holds
        // subjects [100, 150) at target_entries_per_block=50.
        let r2 = BondSstReader::open(&path).unwrap();
        let mut iter = r2.iter_range((NodeId(100), 0, NodeId(0)), (NodeId(150), 0, NodeId(0)));
        let first = iter.next().expect("expected an error item");
        match first {
            Err(BondSstError::CrcMismatch {
                scope: CrcScope::Block { block_id },
                ..
            }) => assert_eq!(block_id, target_block),
            other => panic!("expected CrcMismatch on Block, got {:?}", other),
        }
    }

    /// J. LRU cache eviction. Open with cap=4. Walk enough blocks
    /// to force eviction; assert resident count never exceeds the
    /// cap and re-walking evicted blocks misses.
    #[test]
    fn lru_cache_eviction_observed() {
        let tmp = tempdir().unwrap();
        let path = tmp.path().join("lru.bsst");
        let mut w = BondSstWriter::create(
            &path,
            BondSstWriteOpts {
                file_uid: 0,
                target_entries_per_block: 50,
                compression: CompressionKind::Zstd { level: 1 },
            },
        )
        .unwrap();
        // 20 blocks of 50 entries -- enough to force eviction at cap=4.
        for i in 0u64..(20 * 50) {
            w.append(NodeId(i), 0, NodeId(i + 1), &mk_payload(i))
                .unwrap();
        }
        w.finish().unwrap();

        let cache_cap = 4usize;
        let r = BondSstReader::open_with_cache_cap(&path, cache_cap).unwrap();
        assert_eq!(r.block_count(), 20);

        // Touch one entry per block by issuing a per-subject scan
        // for the first NodeId of each block (subject = block * 50).
        for block in 0u32..20 {
            let subj = NodeId(block as u64 * 50);
            let mut iter = r.iter_subject(subj);
            assert!(iter.next().is_some(), "subject {} present", subj.0);
        }

        // Cache must be bounded.
        let resident = r.cache.lock().unwrap().entries.len();
        assert_eq!(
            resident, cache_cap,
            "LRU cache resident_count must equal cap after walking >cap distinct blocks"
        );

        let s_after_walk = r.stats();
        // 20 block reads -- one per scan, each loaded a fresh block.
        assert!(
            s_after_walk.blocks_read >= 20,
            "expected at least 20 disk block reads, got {}",
            s_after_walk.blocks_read
        );
        assert!(
            s_after_walk.blocks_read > cache_cap as u64,
            "stats.blocks_read ({}) must exceed cache_cap ({}); proves blocks WERE evicted",
            s_after_walk.blocks_read,
            cache_cap
        );

        // Re-walk the first 4 blocks. They were evicted -- so misses
        // increment, not hits.
        let misses_before = s_after_walk.block_cache_misses;
        for block in 0u32..4 {
            let subj = NodeId(block as u64 * 50);
            let mut iter = r.iter_subject(subj);
            assert!(iter.next().is_some());
        }
        let s_after_rewalk = r.stats();
        let new_misses = s_after_rewalk.block_cache_misses - misses_before;
        assert_eq!(
            new_misses, 4,
            "re-walking evicted blocks must miss, got {} new misses",
            new_misses
        );

        // Structural assertion: total misses must exceed cap, evidencing
        // the LRU evict-and-reload cycle.
        assert!(s_after_rewalk.block_cache_misses > cache_cap as u64);
    }

    /// K. file_uid round-trips.
    #[test]
    fn file_uid_round_trips_through_reader() {
        let tmp = tempdir().unwrap();
        let path = tmp.path().join("uid.bsst");
        let w = BondSstWriter::create(
            &path,
            BondSstWriteOpts {
                file_uid: 0xDEADBEEFCAFEBABE,
                target_entries_per_block: 64,
                compression: CompressionKind::None,
            },
        )
        .unwrap();
        w.finish().unwrap();

        let r = BondSstReader::open(&path).unwrap();
        assert_eq!(r.file_uid(), 0xDEADBEEFCAFEBABE);
    }

    /// L. Compression round-trip. Write with zstd, read back, all
    /// payloads byte-equal.
    #[test]
    fn compression_round_trip() {
        let tmp = tempdir().unwrap();
        let path = tmp.path().join("comp.bsst");
        let mut w = BondSstWriter::create(
            &path,
            BondSstWriteOpts {
                file_uid: 91,
                target_entries_per_block: 200,
                compression: CompressionKind::Zstd { level: 1 },
            },
        )
        .unwrap();
        let mut expected: Vec<(NodeId, u8, NodeId, BondPayload)> = Vec::new();
        for i in 0u64..1000 {
            let p = mk_payload(i);
            w.append(NodeId(i), 0, NodeId(i + 7), &p).unwrap();
            expected.push((NodeId(i), 0, NodeId(i + 7), p));
        }
        w.finish().unwrap();

        let r = BondSstReader::open(&path).unwrap();
        let mut got: Vec<(NodeId, u8, NodeId, BondPayload)> = Vec::new();
        for it in r.iter_all() {
            got.push(it.unwrap());
        }
        assert_eq!(got.len(), expected.len());
        for (a, b) in got.iter().zip(expected.iter()) {
            assert_eq!(a.0, b.0);
            assert_eq!(a.1, b.1);
            assert_eq!(a.2, b.2);
            assert!(payloads_equal(&a.3, &b.3));
        }
    }

    /// M. (NodeId+predicate ordering correctness). Build bonds
    /// where the same subject has multiple predicates AND multiple
    /// objects under one predicate. iter_subject must yield in
    /// exactly the writer's append order.
    #[test]
    fn key_ordering_correctness_per_subject() {
        let tmp = tempdir().unwrap();
        let path = tmp.path().join("ord.bsst");
        let mut w = BondSstWriter::create(
            &path,
            BondSstWriteOpts {
                file_uid: 0,
                target_entries_per_block: 64,
                compression: CompressionKind::None,
            },
        )
        .unwrap();
        // (5, 0, 10), (5, 1, 5), (5, 1, 10) -- monotonic on triple.
        let expected: Vec<(NodeId, u8, NodeId)> = vec![
            (NodeId(5), 0, NodeId(10)),
            (NodeId(5), 1, NodeId(5)),
            (NodeId(5), 1, NodeId(10)),
        ];
        for (idx, (s, p, o)) in expected.iter().enumerate() {
            w.append(*s, *p, *o, &mk_payload(idx as u64)).unwrap();
        }
        w.finish().unwrap();

        let r = BondSstReader::open(&path).unwrap();
        let got: Vec<(NodeId, u8, NodeId)> = r
            .iter_subject(NodeId(5))
            .map(|x| {
                let (s, p, o, _) = x.unwrap();
                (s, p, o)
            })
            .collect();
        assert_eq!(got, expected, "iter_subject must preserve triple order");
    }
/// OP-BRAIN-V5 6.5 (a): encrypted BondSST round-trips with the key.
    #[test]
    fn enc65_bond_sst_roundtrips_with_key() {
        let tmp = tempdir().unwrap();
        let path = tmp.path().join("enc.bsst");
        let key = MasterKey::from_bytes([0x7e; 32]);
        let mut w = BondSstWriter::create_with_key(
            &path,
            BondSstWriteOpts { file_uid: 9, target_entries_per_block: 16, compression: CompressionKind::Zstd { level: 1 } },
            Some(key.clone()),
        ).unwrap();
        for i in 0u64..64 {
            w.append(NodeId(i), 1, NodeId(i + 1_000_000), &mk_payload(i)).unwrap();
        }
        assert_eq!(w.finish().unwrap().entries_written, 64);
        let raw = std::fs::read(&path).unwrap();
        assert_eq!(raw[6] & 0x01, 0x01, "header FLAG_ENCRYPTED must be set");
        let r = BondSstReader::open_with_key(&path, Some(key.clone())).unwrap();
        let items: Vec<(NodeId, u8, NodeId, BondPayload)> = r.iter_all().map(|x| x.unwrap()).collect();
        assert_eq!(items.len(), 64);
        for (i, it) in items.iter().enumerate() {
            assert_eq!(it.0, NodeId(i as u64));
            assert_eq!(it.2, NodeId(i as u64 + 1_000_000));
            assert!(payloads_equal(&it.3, &mk_payload(i as u64)));
        }
    }

    /// OP-BRAIN-V5 6.5 (b): no-key / wrong-key BondSST reads fail cleanly.
    #[test]
    fn enc65_bond_sst_wrong_or_missing_key_fails_cleanly() {
        let tmp = tempdir().unwrap();
        let path = tmp.path().join("enc.bsst");
        let key = MasterKey::from_bytes([0x44; 32]);
        let mut w = BondSstWriter::create_with_key(
            &path,
            BondSstWriteOpts { file_uid: 2, target_entries_per_block: 32, compression: CompressionKind::Zstd { level: 1 } },
            Some(key.clone()),
        ).unwrap();
        for i in 0u64..40 { w.append(NodeId(i), 1, NodeId(i + 5), &mk_payload(i)).unwrap(); }
        w.finish().unwrap();
        let r_nokey = BondSstReader::open_with_key(&path, None).unwrap();
        match r_nokey.iter_all().next().expect("one attempt") {
            Err(BondSstError::EncryptedButNoKey) => {}
            other => panic!("expected EncryptedButNoKey, got {:?}", other),
        }
        let wrong = MasterKey::from_bytes([0x55; 32]);
        let r_wrong = BondSstReader::open_with_key(&path, Some(wrong)).unwrap();
        assert!(r_wrong.iter_all().next().expect("one attempt").is_err(), "wrong key must fail cleanly");
    }
}
