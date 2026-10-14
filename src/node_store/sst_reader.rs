//! `NodeSstReader` -- on-demand reader for a NodeStore SST.
//!
//! OP-BRAIN-V5 Phase 2.4 (Q3, Q7 of the design doc; audit S10
//! "NodeSST writer + reader" sub-port). The reader opens an SST
//! produced by [`crate::node_store::sst_writer::NodeSstWriter`]
//! and serves point lookups and bounded range scans without
//! materialising the whole file into RAM.
//!
//! See [`crate::node_store::sst_writer`] for the wire format. The
//! reader holds the header + the footer's `block_index` in memory
//! throughout its lifetime; blocks are decoded on demand and cached
//! in a bounded LRU.
//!
//! ## LRU block cache
//!
//! Adapted from the shape of v4 `ParsedBlockCache` (`src/sst.rs:649`)
//! but specialised for this store: keyed on `(file_uid, block_id)`,
//! capped by a configurable entry count (default 64 -- the audit
//! S2.1 sizing gives ~117 KB/block average, so ~7-8 MB resident at
//! the default cap). The cache is held inside the reader (not
//! shared across readers in this phase -- Phase 2.6 flush worker
//! may revisit this when the read path integration lands).
//!
//! What is salvageable from v4 `ParsedBlockCache`: the LRU keyed-on
//! `(file_uid, block_id)` discipline + the "always leave at least
//! one entry resident so an oversized block does not infinite-
//! evict" rule + the saturating-arithmetic on resident_bytes.
//!
//! What is NOT salvageable: the v4 `ParsedBlock` shape (per-block
//! `subject_table`, `node_table`, `keyframe_pool`) is bond-specific
//! and meaningless here. NodeSST blocks decode to a flat
//! `Vec<(NodeId, NodeEntry)>`.

use std::collections::VecDeque;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use crate::node_intern::NodeId;
use crate::node_store::memtable::NodeEntry;
use crate::node_store::sst_writer::{
    BlockIndexEntry, CompressionKind, CrcScope, NodeSstError, BLOCK_HEADER_BYTES, ENTRY_HEADER_BYTES,
    FLAG_ENCRYPTED, HEADER_BYTES, INDEX_ENTRY_BYTES, NSST_FORMAT_VERSION, NSST_MAGIC,
    TRAILER_BYTES,
};
use crate::encrypt::{xor_in_place, MasterKey};
use crate::sst::derive_block_keystream_index;

/// Default cap on the in-memory LRU block cache (entries).
pub const DEFAULT_BLOCK_CACHE_CAP: usize = 64;

/// Per-reader read-side statistics. Mutated under the reader's
/// internal lock and surfaced via [`NodeSstReader::stats`].
#[derive(Debug, Clone, Copy, Default)]
pub struct NodeSstReadStats {
    /// Number of `get()` calls served from the LRU cache.
    pub block_cache_hits: u64,
    /// Number of `get()` / iter blocks decoded from disk.
    pub block_cache_misses: u64,
    /// Number of `get()` calls that returned `Ok(None)` (no such
    /// NodeId in any block).
    pub get_misses: u64,
    /// Number of disk read syscalls (block reads only; the open-time
    /// header + footer reads are not counted).
    pub disk_block_reads: u64,
    /// Cumulative bytes read from disk (block bodies post-`pread`).
    pub disk_bytes_read: u64,
}

/// A decoded block -- the result of decompressing the on-disk body
/// and parsing its entry stream. The reader stores these in the LRU
/// cache after the first read.
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
    /// Sorted-by-NodeId entries from this block. Linear search within
    /// a block is correct because blocks are bounded to
    /// [`crate::node_store::sst_writer::MAX_TARGET_ENTRIES_PER_BLOCK`]
    /// entries (default 64) -- binary search would be marginal at
    /// best at this scale and add code surface for no measurable
    /// gain.
    entries: Vec<(NodeId, NodeEntry)>,
}

/// LRU block cache. Bounded by entry count -- the simplest discipline
/// that catches Vec-the-world bugs. Keyed on `(file_uid, block_id)`
/// so multiple readers sharing this cache (Phase 2.6+ extension)
/// don't collide.
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
    /// `entries.len() <= cap`, but always leaves at least one entry
    /// resident (matches the v4 discipline).
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

/// On-demand reader for a NodeSST file.
pub struct NodeSstReader {
    path: PathBuf,
    file: Mutex<File>,
    file_uid: u64,
    block_index: Vec<BlockIndexEntry>,
    file_size: u64,
    cache: Mutex<BlockCacheInner>,
    stats: Mutex<NodeSstReadStats>,
    /// OP-BRAIN-V5 6.5: decryption key. `None` = expect plaintext.
    key: Option<MasterKey>,
    /// Header FLAG_ENCRYPTED bit.
    encrypted: bool,
    /// Per-SST IV from the header (zeros when plaintext).
    iv: [u8; 16],
}

impl NodeSstReader {
    /// Open `path` and read the header + footer ONLY. No block reads.
    pub fn open(path: &Path) -> Result<Self, NodeSstError> {
        Self::open_with_cache_cap_and_key(path, DEFAULT_BLOCK_CACHE_CAP, None)
    }

    /// Open with an explicit block-cache cap (plaintext / no key).
    pub fn open_with_cache_cap(
        path: &Path,
        cache_cap: usize,
    ) -> Result<Self, NodeSstError> {
        Self::open_with_cache_cap_and_key(path, cache_cap, None)
    }

    /// Open with a decryption key (OP-BRAIN-V5 6.5). Required to read
    /// an encrypted SST; `None` reads plaintext only -- an encrypted
    /// SST decoded without a key fails cleanly with
    /// [`NodeSstError::EncryptedButNoKey`].
    pub fn open_with_key(path: &Path, key: Option<MasterKey>) -> Result<Self, NodeSstError> {
        Self::open_with_cache_cap_and_key(path, DEFAULT_BLOCK_CACHE_CAP, key)
    }

    /// Open with an explicit block-cache cap AND a decryption key.
    pub fn open_with_cache_cap_and_key(
        path: &Path,
        cache_cap: usize,
        key: Option<MasterKey>,
    ) -> Result<Self, NodeSstError> {
        let file = File::open(path)?;
        let file_size = file.metadata()?.len();
        if file_size < (HEADER_BYTES + TRAILER_BYTES) as u64 {
            return Err(NodeSstError::Format(format!(
                "file too small to be a NodeSST: {} bytes",
                file_size
            )));
        }

        // --- Header --------------------------------------------------
        let header = read_header_validated(&file)?;

        // --- Footer --------------------------------------------------
        if header.footer_offset >= file_size {
            return Err(NodeSstError::Format(format!(
                "header.footer_offset {} >= file_size {}",
                header.footer_offset, file_size
            )));
        }
        let footer_byte_len = file_size - header.footer_offset - TRAILER_BYTES as u64;
        if footer_byte_len < 8 {
            return Err(NodeSstError::Format(format!(
                "footer too small: {} bytes",
                footer_byte_len
            )));
        }
        let block_index = read_footer_validated(&file, header.footer_offset, header.block_count)?;

        // --- Trailer (sanity) ---------------------------------------
        read_trailer_validated(&file, file_size, header.footer_offset)?;

        Ok(NodeSstReader {
            path: path.to_path_buf(),
            file: Mutex::new(file),
            file_uid: header.file_uid,
            block_index,
            file_size,
            cache: Mutex::new(BlockCacheInner::new(cache_cap)),
            stats: Mutex::new(NodeSstReadStats::default()),
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
    pub fn stats(&self) -> NodeSstReadStats {
        *self.stats.lock().expect("stats lock poisoned")
    }

    /// Point lookup. Binary-search the block_index for the block that
    /// COULD contain `id`, fetch / decode it (consulting the LRU
    /// cache), then linear-search the block for the exact NodeId.
    pub fn get(&self, id: NodeId) -> Result<Option<NodeEntry>, NodeSstError> {
        // Empty SST.
        if self.block_index.is_empty() {
            let mut s = self.stats.lock().expect("stats lock poisoned");
            s.get_misses += 1;
            return Ok(None);
        }
        // The block that COULD contain `id` is the one whose
        // first_node_id <= id <= last_node_id. We binary-search by
        // first_node_id and then verify last_node_id covers `id`.
        let block_id = match self.find_block_for(id) {
            Some(b) => b,
            None => {
                let mut s = self.stats.lock().expect("stats lock poisoned");
                s.get_misses += 1;
                return Ok(None);
            }
        };

        let block = self.fetch_block(block_id)?;
        for (eid, entry) in &block.entries {
            if *eid == id {
                return Ok(Some(entry.clone()));
            }
            if eid.0 > id.0 {
                // Block is sorted -- past the target.
                break;
            }
        }
        let mut s = self.stats.lock().expect("stats lock poisoned");
        s.get_misses += 1;
        Ok(None)
    }

    /// Full-SST scan -- yields every `(NodeId, NodeEntry)` pair
    /// in ascending NodeId order. Convenience over
    /// [`iter_range`] for callers (notably Phase 2.7
    /// `NodeCompactionWorker`) that want to stream the whole SST
    /// without picking sentinel bounds. Implemented as
    /// `iter_range(NodeId(0)..NodeId(u64::MAX))`; the upper bound
    /// is exclusive but `u64::MAX` is reserved (no real NodeId
    /// ever reaches it).
    pub fn iter_all<'a>(&'a self) -> NodeSstRangeIter<'a> {
        self.iter_range(NodeId(0), NodeId(u64::MAX))
    }

    /// Range scan `[start, end)` -- yields `(NodeId, NodeEntry)`
    /// pairs in ascending NodeId order. Loads blocks on demand.
    pub fn iter_range<'a>(
        &'a self,
        start: NodeId,
        end: NodeId,
    ) -> NodeSstRangeIter<'a> {
        let start_block = if self.block_index.is_empty() {
            None
        } else {
            // Find the first block whose last_node_id >= start.
            let mut lo = 0usize;
            let mut hi = self.block_index.len();
            while lo < hi {
                let mid = (lo + hi) / 2;
                if self.block_index[mid].last_node_id.0 < start.0 {
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
        NodeSstRangeIter {
            reader: self,
            start,
            end,
            next_block: start_block,
            current_block: None,
            current_idx: 0,
        }
    }

    // --- internals ------------------------------------------------------

    /// Binary-search block_index to find the block whose NodeId range
    /// COULD contain `id`. Returns `None` if `id` is outside every
    /// block's range (below the first block's first_node_id, above
    /// the last block's last_node_id, or in a gap between blocks).
    fn find_block_for(&self, id: NodeId) -> Option<u32> {
        // The block_index is sorted by first_node_id ascending (writer
        // guarantees monotonic append). Binary-search for the
        // RIGHTMOST block whose first_node_id <= id.
        let mut lo = 0i64;
        let mut hi = self.block_index.len() as i64 - 1;
        let mut candidate: Option<usize> = None;
        while lo <= hi {
            let mid = ((lo + hi) / 2) as usize;
            if self.block_index[mid].first_node_id.0 <= id.0 {
                candidate = Some(mid);
                lo = mid as i64 + 1;
            } else {
                hi = mid as i64 - 1;
            }
        }
        let cand = candidate?;
        // Verify last_node_id covers `id`.
        if self.block_index[cand].last_node_id.0 >= id.0 {
            Some(cand as u32)
        } else {
            None
        }
    }

    fn fetch_block(&self, block_id: u32) -> Result<Arc<DecodedBlock>, NodeSstError> {
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

    fn decode_block_from_disk(&self, block_id: u32) -> Result<DecodedBlock, NodeSstError> {
        let entry = *self
            .block_index
            .get(block_id as usize)
            .ok_or_else(|| NodeSstError::Format(format!("block_id {} out of range", block_id)))?;

        // Sanity: the block must fit inside the file.
        let end = entry
            .block_offset
            .checked_add(entry.block_size as u64)
            .ok_or_else(|| {
                NodeSstError::Format(format!(
                    "block {} offset+size overflow",
                    block_id
                ))
            })?;
        if end > self.file_size {
            return Err(NodeSstError::Format(format!(
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
            s.disk_block_reads += 1;
            s.disk_bytes_read += buf.len() as u64;
        }

        // Parse block header.
        if buf.len() < BLOCK_HEADER_BYTES {
            return Err(NodeSstError::Format(format!(
                "block {} on-disk size {} < header size {}",
                block_id, buf.len(), BLOCK_HEADER_BYTES
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
                NodeSstError::Format(format!(
                    "block {} compressed_size+header overflow",
                    block_id
                ))
            })?;
        if on_disk_body_len != buf.len() {
            return Err(NodeSstError::Format(format!(
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
            return Err(NodeSstError::CrcMismatch {
                scope: CrcScope::Block { block_id },
                expected: block_crc,
                observed: computed,
            });
        }

        // Decompress.
        let compression = CompressionKind::from_u8(compression_kind_byte)
            .ok_or(NodeSstError::UnknownCompressionKind { observed: compression_kind_byte })?;
        let mut uncompressed: Vec<u8> = match compression {
            CompressionKind::None => compressed_body.to_vec(),
            CompressionKind::Zstd { .. } => zstd::decode_all(compressed_body)
                .map_err(|e| NodeSstError::Zstd(format!("decode block {}: {}", block_id, e)))?,
        };
        if uncompressed.len() as u64 != uncompressed_size as u64 {
            return Err(NodeSstError::Format(format!(
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
            let k = self.key.as_ref().ok_or(NodeSstError::EncryptedButNoKey)?;
            let ksi = derive_block_keystream_index(self.file_uid, block_id, &self.iv);
            xor_in_place(k, ksi, &mut uncompressed);
        }

        // Parse entries.
        let mut entries: Vec<(NodeId, NodeEntry)> = Vec::with_capacity(entry_count as usize);
        let mut cursor = 0usize;
        for _ in 0..entry_count {
            if cursor + ENTRY_HEADER_BYTES > uncompressed.len() {
                return Err(NodeSstError::Format(format!(
                    "block {} truncated mid-entry header",
                    block_id
                )));
            }
            let node_id = u64::from_le_bytes([
                uncompressed[cursor],
                uncompressed[cursor + 1],
                uncompressed[cursor + 2],
                uncompressed[cursor + 3],
                uncompressed[cursor + 4],
                uncompressed[cursor + 5],
                uncompressed[cursor + 6],
                uncompressed[cursor + 7],
            ]);
            let entry_size = u32::from_le_bytes([
                uncompressed[cursor + 8],
                uncompressed[cursor + 9],
                uncompressed[cursor + 10],
                uncompressed[cursor + 11],
            ]) as usize;
            cursor += ENTRY_HEADER_BYTES;
            if cursor + entry_size > uncompressed.len() {
                return Err(NodeSstError::Format(format!(
                    "block {} truncated mid-entry payload",
                    block_id
                )));
            }
            let entry: NodeEntry = bincode::deserialize(&uncompressed[cursor..cursor + entry_size])
                .map_err(|e| {
                    NodeSstError::Bincode(format!("deserialize NodeEntry block {}: {}", block_id, e))
                })?;
            cursor += entry_size;
            entries.push((NodeId(node_id), entry));
        }
        if cursor != uncompressed.len() {
            return Err(NodeSstError::Format(format!(
                "block {} has {} trailing bytes after {} entries",
                block_id,
                uncompressed.len() - cursor,
                entry_count
            )));
        }

        // Confirm against block_index claims (sanity).
        if let Some((first_id, _)) = entries.first() {
            if *first_id != entry.first_node_id {
                return Err(NodeSstError::Format(format!(
                    "block {} entries[0].id {} != index.first_node_id {}",
                    block_id, first_id.0, entry.first_node_id.0
                )));
            }
        }
        if let Some((last_id, _)) = entries.last() {
            if *last_id != entry.last_node_id {
                return Err(NodeSstError::Format(format!(
                    "block {} entries[last].id {} != index.last_node_id {}",
                    block_id, last_id.0, entry.last_node_id.0
                )));
            }
        }

        Ok(DecodedBlock { entries })
    }
}

// ----------------------------------------------------------------------
// Range iterator.
// ----------------------------------------------------------------------

/// Streaming iterator returned by [`NodeSstReader::iter_range`].
pub struct NodeSstRangeIter<'a> {
    reader: &'a NodeSstReader,
    start: NodeId,
    end: NodeId,
    /// Next block to load (None means iteration is exhausted).
    next_block: Option<u32>,
    /// In-flight decoded block.
    current_block: Option<Arc<DecodedBlock>>,
    /// Cursor inside `current_block.entries`.
    current_idx: usize,
}

impl<'a> Iterator for NodeSstRangeIter<'a> {
    type Item = Result<(NodeId, NodeEntry), NodeSstError>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // Drain the current block.
            if let Some(blk) = &self.current_block {
                while self.current_idx < blk.entries.len() {
                    let (eid, entry) = &blk.entries[self.current_idx];
                    self.current_idx += 1;
                    if eid.0 < self.start.0 {
                        continue;
                    }
                    if eid.0 >= self.end.0 {
                        // Past the end -- iteration done.
                        self.next_block = None;
                        self.current_block = None;
                        return None;
                    }
                    return Some(Ok((*eid, entry.clone())));
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
            // Bail early if the next block's first_node_id is already
            // past end.
            if self.reader.block_index[bid as usize].first_node_id.0 >= self.end.0 {
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

fn read_header_validated(mut file: &File) -> Result<ParsedHeader, NodeSstError> {
    let mut buf = [0u8; HEADER_BYTES];
    file.seek(SeekFrom::Start(0))?;
    file.read_exact(&mut buf)?;

    let magic = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
    if magic != NSST_MAGIC {
        return Err(NodeSstError::MagicMismatch { observed: magic });
    }
    let format_version = u16::from_le_bytes([buf[4], buf[5]]);
    if format_version != NSST_FORMAT_VERSION {
        return Err(NodeSstError::FormatVersionUnsupported { observed: format_version });
    }
    let block_count = u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]);
    let footer_offset = u64::from_le_bytes([
        buf[12], buf[13], buf[14], buf[15],
        buf[16], buf[17], buf[18], buf[19],
    ]);
    let file_uid = u64::from_le_bytes([
        buf[20], buf[21], buf[22], buf[23],
        buf[24], buf[25], buf[26], buf[27],
    ]);
    let flags = buf[6];
    let mut iv = [0u8; 16];
    iv.copy_from_slice(&buf[28..44]);
    let header_crc = u32::from_le_bytes([buf[44], buf[45], buf[46], buf[47]]);
    let computed = crc32fast::hash(&buf[..44]);
    if computed != header_crc {
        return Err(NodeSstError::CrcMismatch {
            scope: CrcScope::Header,
            expected: header_crc,
            observed: computed,
        });
    }

    Ok(ParsedHeader { block_count, footer_offset, file_uid, flags, iv })
}

fn read_footer_validated(
    mut file: &File,
    footer_offset: u64,
    expected_block_count: u32,
) -> Result<Vec<BlockIndexEntry>, NodeSstError> {
    let index_bytes_total = (expected_block_count as usize)
        .checked_mul(INDEX_ENTRY_BYTES)
        .ok_or_else(|| {
            NodeSstError::Format(format!(
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
        return Err(NodeSstError::Format(format!(
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
        return Err(NodeSstError::CrcMismatch {
            scope: CrcScope::Footer,
            expected: footer_crc_disk,
            observed: computed,
        });
    }

    let mut block_index = Vec::with_capacity(expected_block_count as usize);
    for i in 0..(expected_block_count as usize) {
        let off = i * INDEX_ENTRY_BYTES;
        let first = u64::from_le_bytes([
            buf[off], buf[off + 1], buf[off + 2], buf[off + 3],
            buf[off + 4], buf[off + 5], buf[off + 6], buf[off + 7],
        ]);
        let last = u64::from_le_bytes([
            buf[off + 8], buf[off + 9], buf[off + 10], buf[off + 11],
            buf[off + 12], buf[off + 13], buf[off + 14], buf[off + 15],
        ]);
        let block_offset = u64::from_le_bytes([
            buf[off + 16], buf[off + 17], buf[off + 18], buf[off + 19],
            buf[off + 20], buf[off + 21], buf[off + 22], buf[off + 23],
        ]);
        let block_size = u32::from_le_bytes([
            buf[off + 24], buf[off + 25], buf[off + 26], buf[off + 27],
        ]);
        if first > last {
            return Err(NodeSstError::Format(format!(
                "block_index[{}] first_node_id {} > last_node_id {}",
                i, first, last
            )));
        }
        if i > 0 {
            let prev = &block_index[i - 1] as &BlockIndexEntry;
            if first <= prev.last_node_id.0 {
                return Err(NodeSstError::Format(format!(
                    "block_index[{}] first_node_id {} overlaps prev.last_node_id {}",
                    i, first, prev.last_node_id.0
                )));
            }
        }
        block_index.push(BlockIndexEntry {
            first_node_id: NodeId(first),
            last_node_id: NodeId(last),
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
) -> Result<(), NodeSstError> {
    let trailer_offset = file_size - TRAILER_BYTES as u64;
    let mut buf = [0u8; TRAILER_BYTES];
    file.seek(SeekFrom::Start(trailer_offset))?;
    file.read_exact(&mut buf)?;
    let magic = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
    if magic != NSST_MAGIC {
        return Err(NodeSstError::MagicMismatch { observed: magic });
    }
    let offset_tail = u64::from_le_bytes([
        buf[4], buf[5], buf[6], buf[7],
        buf[8], buf[9], buf[10], buf[11],
    ]);
    if offset_tail != expected_footer_offset {
        return Err(NodeSstError::Format(format!(
            "trailer footer_offset {} != header footer_offset {}",
            offset_tail, expected_footer_offset
        )));
    }
    Ok(())
}

// Unused import shim so the BufReader path doesn't go unused on some
// builds (kept tight; reader currently does positional seeks).
fn _suppress_unused() {
    let _: Option<BufReader<File>> = None;
}

// ----------------------------------------------------------------------
// Tests.
// ----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::{Confidence, KnowledgeContent, KnowledgeNode, KnowledgeUnit};
    use crate::node_store::sst_writer::{
        CompressionKind, NodeSstWriteOpts, NodeSstWriter, MAX_TARGET_ENTRIES_PER_BLOCK,
    };
    use crate::seed::Seed;
    use std::fs::OpenOptions;
    use std::io::Write;
    use tempfile::tempdir;

    fn mk_node(label: &str) -> KnowledgeNode {
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

    fn mk_real_node(label: &str) -> KnowledgeNode {
        KnowledgeNode {
            seed: Seed::from_label(label),
            content: label.to_string(),
            knowledge: KnowledgeUnit {
                content: KnowledgeContent::Concept(label.to_string()),
                confidence: Confidence::Verified,
                created_at: 0,
                updated_at: 0,
            },
            // Real-shape 384-d embedding for the compression-observation
            // test. Deterministic content -- zstd will collapse this
            // heavily, which is NOT representative of trained
            // embeddings; the test documents the ceiling, not the floor.
            embedding: (0..384).map(|i| (i as f32) * 0.001).collect(),
            parent_seeds: Vec::new(),
            typed_ratio: 0.0,
        }
    }

    fn mk_entry(label: &str, seq: u64) -> NodeEntry {
        NodeEntry { payload: mk_node(label), tombstone: false, seq }
    }

    fn mk_real_entry(label: &str, seq: u64) -> NodeEntry {
        NodeEntry { payload: mk_real_node(label), tombstone: false, seq }
    }

    /// A. Empty roundtrip.
    #[test]
    fn empty_roundtrip() {
        let tmp = tempdir().unwrap();
        let path = tmp.path().join("empty.nsst");
        let w = NodeSstWriter::create(
            &path,
            NodeSstWriteOpts {
                file_uid: 42,
                target_entries_per_block: 64,
                compression: CompressionKind::Zstd { level: 1 },
            },
        )
        .unwrap();
        let stats = w.finish().unwrap();
        assert_eq!(stats.entries_written, 0);
        assert_eq!(stats.blocks_written, 0);

        let r = NodeSstReader::open(&path).unwrap();
        assert_eq!(r.file_uid(), 42);
        assert_eq!(r.block_count(), 0);
        assert!(matches!(r.get(NodeId(0)), Ok(None)));
        assert!(matches!(r.get(NodeId(1)), Ok(None)));
        assert!(matches!(r.get(NodeId(u64::MAX)), Ok(None)));
        let collected: Vec<_> = r.iter_range(NodeId(0), NodeId(u64::MAX)).collect();
        assert_eq!(collected.len(), 0);
    }

    /// B. Single-block roundtrip. 50 entries, 1..=50.
    #[test]
    fn single_block_roundtrip() {
        let tmp = tempdir().unwrap();
        let path = tmp.path().join("single.nsst");
        let mut w = NodeSstWriter::create(
            &path,
            NodeSstWriteOpts {
                file_uid: 7,
                target_entries_per_block: 64,
                compression: CompressionKind::Zstd { level: 1 },
            },
        )
        .unwrap();
        for i in 1u64..=50 {
            w.append(NodeId(i), &mk_entry(&format!("n_{}", i), i)).unwrap();
        }
        let stats = w.finish().unwrap();
        assert_eq!(stats.entries_written, 50);
        assert_eq!(stats.blocks_written, 1);

        let r = NodeSstReader::open(&path).unwrap();
        assert_eq!(r.file_uid(), 7);
        for i in 1u64..=50 {
            let got = r.get(NodeId(i)).unwrap();
            assert!(got.is_some(), "NodeId({}) missing", i);
            let entry = got.unwrap();
            assert_eq!(entry.seq, i);
            assert_eq!(entry.payload.content, format!("n_{}", i));
        }
        assert!(matches!(r.get(NodeId(51)), Ok(None)));
        assert!(matches!(r.get(NodeId(0)), Ok(None)));
    }

    /// C. Multi-block roundtrip with compression. 10,000 entries.
    #[test]
    fn multi_block_roundtrip_with_compression() {
        let tmp = tempdir().unwrap();
        let path = tmp.path().join("multi.nsst");
        let mut w = NodeSstWriter::create(
            &path,
            NodeSstWriteOpts {
                file_uid: 99,
                target_entries_per_block: 100,
                compression: CompressionKind::Zstd { level: 1 },
            },
        )
        .unwrap();
        const N: u64 = 10_000;
        for i in 0..N {
            w.append(NodeId(i), &mk_real_entry(&format!("n_{}", i), i)).unwrap();
        }
        let stats = w.finish().unwrap();
        assert_eq!(stats.entries_written, N);
        assert_eq!(stats.blocks_written, 100);
        eprintln!(
            "multi_block_roundtrip: uncompressed={} B, compressed={} B, ratio={:.3}",
            stats.uncompressed_bytes,
            stats.compressed_bytes,
            (stats.compressed_bytes as f64) / (stats.uncompressed_bytes.max(1) as f64),
        );

        let r = NodeSstReader::open(&path).unwrap();
        assert_eq!(r.block_count(), 100);
        // Sample 50 random NodeIds (deterministic for test stability).
        let samples = [0, 1, 7, 13, 42, 99, 100, 101, 499, 500, 501, 999, 1000, 1001, 1234,
            2000, 2500, 3000, 3333, 3999, 4000, 4321, 4567, 4999, 5000, 5001, 5500, 6000,
            6543, 6999, 7000, 7321, 7654, 7777, 7999, 8000, 8500, 8888, 8999, 9000, 9001,
            9100, 9234, 9500, 9876, 9990, 9998, 9999, 0, 1u64];
        for s in samples.iter() {
            let got = r.get(NodeId(*s)).unwrap().expect("present");
            assert_eq!(got.seq, *s);
            assert_eq!(got.payload.content, format!("n_{}", s));
        }
        assert!(matches!(r.get(NodeId(99_999)), Ok(None)));
        assert!(matches!(r.get(NodeId(N)), Ok(None))); // boundary
    }

    /// D. Range scan.
    #[test]
    fn range_scan_dense() {
        let tmp = tempdir().unwrap();
        let path = tmp.path().join("range.nsst");
        let mut w = NodeSstWriter::create(
            &path,
            NodeSstWriteOpts {
                file_uid: 0,
                target_entries_per_block: 100,
                compression: CompressionKind::Zstd { level: 1 },
            },
        )
        .unwrap();
        for i in 0u64..10_000 {
            w.append(NodeId(i), &mk_entry(&format!("n_{}", i), i)).unwrap();
        }
        let _ = w.finish().unwrap();

        let r = NodeSstReader::open(&path).unwrap();
        let mut count = 0u64;
        let mut last: Option<u64> = None;
        for item in r.iter_range(NodeId(1000), NodeId(2000)) {
            let (id, _entry) = item.unwrap();
            if let Some(p) = last {
                assert!(id.0 > p, "range_scan must be monotonic ascending: {} after {}", id.0, p);
            }
            assert!((1000..2000).contains(&id.0), "id {} outside [1000, 2000)", id.0);
            last = Some(id.0);
            count += 1;
        }
        assert_eq!(count, 1000);
    }

    /// F. Corrupt block CRC. Flip one byte in the middle of a block,
    /// then get() must surface NodeSstError::CrcMismatch.
    #[test]
    fn corrupt_block_crc_rejected() {
        let tmp = tempdir().unwrap();
        let path = tmp.path().join("crc.nsst");
        let mut w = NodeSstWriter::create(
            &path,
            NodeSstWriteOpts {
                file_uid: 0,
                target_entries_per_block: 50,
                compression: CompressionKind::None, // raw -- byte we flip is unambiguously in the body
            },
        )
        .unwrap();
        for i in 0u64..200 {
            w.append(NodeId(i), &mk_entry(&format!("n_{}", i), i)).unwrap();
        }
        let _ = w.finish().unwrap();

        // Open the reader to learn block offsets, then corrupt block 2's body.
        let r = NodeSstReader::open(&path).unwrap();
        assert!(r.block_count() >= 4);
        let target_block = 2u32;
        let target_offset = r.block_index[target_block as usize].block_offset
            + BLOCK_HEADER_BYTES as u64
            + 10;
        drop(r); // release file handle for the writeable open below.

        {
            let mut f = OpenOptions::new().write(true).open(&path).unwrap();
            f.seek(SeekFrom::Start(target_offset)).unwrap();
            let original = {
                let mut buf = [0u8; 1];
                let mut rf = File::open(&path).unwrap();
                rf.seek(SeekFrom::Start(target_offset)).unwrap();
                rf.read_exact(&mut buf).unwrap();
                buf[0]
            };
            f.write_all(&[original ^ 0xFF]).unwrap();
            f.sync_all().unwrap();
        }

        // Re-open and query a NodeId in block 2.
        let r2 = NodeSstReader::open(&path).unwrap();
        // block 2 holds NodeIds 100..150 (target_entries_per_block=50).
        let target_node = NodeId(120);
        let got = r2.get(target_node);
        match got {
            Err(NodeSstError::CrcMismatch { scope, .. }) => match scope {
                CrcScope::Block { block_id } => assert_eq!(block_id, target_block),
                other => panic!("expected Block scope, got {:?}", other),
            },
            other => panic!("expected CrcMismatch on Block, got {:?}", other),
        }
    }

    /// G. Footer corruption -- open() must reject with CrcMismatch.
    #[test]
    fn corrupt_footer_rejected_on_open() {
        let tmp = tempdir().unwrap();
        let path = tmp.path().join("footer.nsst");
        let mut w = NodeSstWriter::create(
            &path,
            NodeSstWriteOpts {
                file_uid: 0,
                target_entries_per_block: 50,
                compression: CompressionKind::Zstd { level: 1 },
            },
        )
        .unwrap();
        for i in 0u64..200 {
            w.append(NodeId(i), &mk_entry(&format!("n_{}", i), i)).unwrap();
        }
        let stats = w.finish().unwrap();

        // Locate the footer offset via the trailer.
        let bytes = std::fs::read(&path).unwrap();
        assert!(bytes.len() == stats.file_size_bytes as usize);
        let trailer_off = bytes.len() - TRAILER_BYTES;
        let footer_offset = u64::from_le_bytes([
            bytes[trailer_off + 4], bytes[trailer_off + 5],
            bytes[trailer_off + 6], bytes[trailer_off + 7],
            bytes[trailer_off + 8], bytes[trailer_off + 9],
            bytes[trailer_off + 10], bytes[trailer_off + 11],
        ]);
        // Flip one byte inside the block_index payload (well before the
        // footer's index_byte_len field).
        let corrupt_off = footer_offset + 5;
        {
            let mut f = OpenOptions::new().write(true).open(&path).unwrap();
            f.seek(SeekFrom::Start(corrupt_off)).unwrap();
            let mut rf = File::open(&path).unwrap();
            rf.seek(SeekFrom::Start(corrupt_off)).unwrap();
            let mut b = [0u8; 1];
            rf.read_exact(&mut b).unwrap();
            f.write_all(&[b[0] ^ 0xFF]).unwrap();
            f.sync_all().unwrap();
        }

        match NodeSstReader::open(&path) {
            Err(NodeSstError::CrcMismatch { scope: CrcScope::Footer, .. }) => {}
            Err(other) => panic!("expected Footer CrcMismatch on open, got Err({:?})", other),
            Ok(_) => panic!("expected Footer CrcMismatch on open, got Ok"),
        }
    }

    /// I. LRU cache eviction. 200 blocks, walk-then-walk; verify
    /// resident_count stays bounded under the cap.
    #[test]
    fn lru_cache_eviction() {
        let tmp = tempdir().unwrap();
        let path = tmp.path().join("lru.nsst");
        let mut w = NodeSstWriter::create(
            &path,
            NodeSstWriteOpts {
                file_uid: 0,
                target_entries_per_block: 50,
                compression: CompressionKind::Zstd { level: 1 },
            },
        )
        .unwrap();
        for i in 0u64..(200 * 50) {
            w.append(NodeId(i), &mk_entry(&format!("n_{}", i), i)).unwrap();
        }
        let _ = w.finish().unwrap();

        let r = NodeSstReader::open_with_cache_cap(&path, 8).unwrap();
        assert_eq!(r.block_count(), 200);

        // Touch one NodeId per block.
        for block in 0u32..200 {
            let id = NodeId(block as u64 * 50);
            let _ = r.get(id).unwrap().expect("present");
        }
        // The cache cap is 8 so resident_count must equal min(cap, blocks_touched).
        let resident = r.cache.lock().unwrap().entries.len();
        assert_eq!(resident, 8, "cache must bound resident block count");

        // Re-walk first 8 blocks -- those should still be cache misses
        // because eviction has pushed them out. Capture stats delta.
        let stats_before = r.stats();
        for block in 0u32..8 {
            let id = NodeId(block as u64 * 50);
            let _ = r.get(id).unwrap();
        }
        let stats_after = r.stats();
        let new_misses = stats_after.block_cache_misses - stats_before.block_cache_misses;
        assert_eq!(new_misses, 8, "re-walking evicted blocks must miss");
    }

    /// J. file_uid round-trip through the reader.
    #[test]
    fn file_uid_round_trip() {
        let tmp = tempdir().unwrap();
        let path = tmp.path().join("uid_r.nsst");
        let w = NodeSstWriter::create(
            &path,
            NodeSstWriteOpts {
                file_uid: 0xDEADBEEFCAFEBABE,
                target_entries_per_block: 64,
                compression: CompressionKind::None,
            },
        )
        .unwrap();
        let _ = w.finish().unwrap();

        let r = NodeSstReader::open(&path).unwrap();
        assert_eq!(r.file_uid(), 0xDEADBEEFCAFEBABE);
    }

    /// Reader rejects oversized target_entries_per_block via writer.
    #[test]
    fn writer_rejects_invalid_opts() {
        let tmp = tempdir().unwrap();
        let path = tmp.path().join("invalid.nsst");
        let res = NodeSstWriter::create(
            &path,
            NodeSstWriteOpts {
                file_uid: 0,
                target_entries_per_block: MAX_TARGET_ENTRIES_PER_BLOCK + 1,
                compression: CompressionKind::None,
            },
        );
        assert!(matches!(res, Err(NodeSstError::InvalidOpts(_))));
        let res2 = NodeSstWriter::create(
            &path,
            NodeSstWriteOpts {
                file_uid: 0,
                target_entries_per_block: 0,
                compression: CompressionKind::None,
            },
        );
        assert!(matches!(res2, Err(NodeSstError::InvalidOpts(_))));
    }
/// OP-BRAIN-V5 6.5 (a): encrypted NodeSST round-trips through the
    /// direct reader AND the keyed reader cache; no plaintext on disk.
    #[test]
    fn enc65_node_sst_roundtrips_reader_and_cache() {
        use crate::node_store::{nsst_filename, NodeSstReaderCache};
        let tmp = tempdir().unwrap();
        let dir = tmp.path().join("node_store");
        std::fs::create_dir_all(&dir).unwrap();
        let key = MasterKey::from_bytes([0x5a; 32]);
        let uid = 7u64;
        let path = dir.join(nsst_filename(uid));
        let mut w = NodeSstWriter::create_with_key(
            &path,
            NodeSstWriteOpts { file_uid: uid, target_entries_per_block: 4, compression: CompressionKind::Zstd { level: 1 } },
            Some(key.clone()),
        ).unwrap();
        for i in 0..20u64 {
            w.append(NodeId(i + 1), &mk_entry(&format!("nlabel_{}", i), i)).unwrap();
        }
        assert_eq!(w.finish().unwrap().entries_written, 20);
        let raw = std::fs::read(&path).unwrap();
        assert_eq!(raw[6] & 0x01, 0x01, "header FLAG_ENCRYPTED must be set");
        assert!(!raw.windows(7).any(|c| c == b"nlabel_"), "plaintext label leaked on disk");
        let r = NodeSstReader::open_with_key(&path, Some(key.clone())).unwrap();
        assert_eq!(r.get(NodeId(13)).unwrap().expect("present").payload.content, "nlabel_12");
        assert_eq!(r.iter_all().count(), 20);
        let cache = NodeSstReaderCache::new(dir.clone()).with_key(Some(key.clone()));
        let cr = cache.get_or_open(uid).unwrap();
        assert_eq!(cr.get(NodeId(1)).unwrap().expect("present").payload.content, "nlabel_0");
        assert_eq!(cr.iter_all().count(), 20);
    }

    /// OP-BRAIN-V5 6.5 (b): no-key / wrong-key reads fail cleanly.
    #[test]
    fn enc65_node_sst_wrong_or_missing_key_fails_cleanly() {
        let tmp = tempdir().unwrap();
        let path = tmp.path().join("enc.nsst");
        let key = MasterKey::from_bytes([0x11; 32]);
        let mut w = NodeSstWriter::create_with_key(
            &path,
            NodeSstWriteOpts { file_uid: 3, target_entries_per_block: 8, compression: CompressionKind::Zstd { level: 1 } },
            Some(key.clone()),
        ).unwrap();
        for i in 0..16u64 { w.append(NodeId(i + 1), &mk_entry(&format!("x_{}", i), i)).unwrap(); }
        w.finish().unwrap();
        let r_nokey = NodeSstReader::open_with_key(&path, None).unwrap();
        match r_nokey.get(NodeId(5)) {
            Err(NodeSstError::EncryptedButNoKey) => {}
            other => panic!("expected EncryptedButNoKey, got {:?}", other),
        }
        let wrong = MasterKey::from_bytes([0x22; 32]);
        let r_wrong = NodeSstReader::open_with_key(&path, Some(wrong)).unwrap();
        assert!(r_wrong.get(NodeId(5)).is_err(), "wrong key must fail cleanly");
    }
}
