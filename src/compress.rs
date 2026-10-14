//! Storage v2 Phase 3 — per-block zstd compression for the v3 brain.
//!
//! Phase 1 reserved a `compression` bit in the v3 file header; this
//! module is what flips the bit and re-encodes the network section so
//! the on-disk file shrinks by ~10x while the mmap-zero-copy load
//! path keeps RSS within ~10% of the uncompressed v3 file.
//!
//! ## Strategy
//!
//! The original v3 layout writes `NetworkSnapshot` as one giant
//! rkyv-archived blob. zstd compresses the rkyv blob itself extremely
//! well (424 MB → ~25 MB at level 3) because rkyv's natural alignment
//! padding leaves long runs of zeros, but you cannot decompress just
//! a slice of the rkyv blob at lookup time — `rkyv::archived_root`
//! needs a contiguous, validated, aligned buffer.
//!
//! Phase 3 sidesteps this by re-encoding each `KnowledgeNode` and
//! `TemporalBond` as ITS OWN INDEPENDENT rkyv archive (so each is
//! validatable / dereferenceable in isolation), packing those
//! per-entity archives into ~256 KB blocks of decompressed bytes,
//! and zstd-compressing each block independently. An on-disk seed →
//! `(block_id, offset, length)` index sits next to the blocks so a
//! seed lookup never has to walk the compressed payload.
//!
//! ## On-disk payload (one v3 section, with header `flags &
//! `FORMAT_FLAG_COMPRESSED` set):
//!
//! ```text
//! [header 32 bytes]
//!   u32 node_count
//!   u32 bond_count
//!   u32 block_count
//!   u32 block_size_target
//!   u32 zstd_level
//!   u32 reserved
//!   u64 reserved
//!
//! [node_index] node_count * 44 bytes
//!   [u8;32 seed][u32 block_id][u32 offset][u32 length]
//!
//! [bond_index] bond_count * 76 bytes
//!   [u8;32 from][u8;32 to][u32 block_id][u32 offset][u32 length]
//!
//! [block_offset_table] block_count * 8 bytes
//!   u64 absolute_offset_within_section_payload  (start of block frame)
//!
//! [blocks] block_count frames, each:
//!   [u32 compressed_len][u32 decompressed_len][compressed_bytes ...]
//!
//! Each decompressed block contains zero or more entries, each one a
//! complete rkyv archive of a `KnowledgeNode` OR a `TemporalBond`.
//! Entries are 16-byte aligned within the block; the index records
//! the entry's start offset and exact length.
//! ```
//!
//! ## Read-side lifetime story (OP-016: bounded LRU)
//!
//! The block cache is a bounded LRU of `Arc<rkyv::AlignedVec>` keyed
//! by block id, with byte-accurate eviction. OP-016 replaced the
//! previous monotonic `Vec<OnceLock<AlignedVec>>` arena — once a
//! block was decompressed it never freed, so workloads that touch
//! every block (retrain) grew RssAnon toward the full decompressed
//! network size and tripped systemd-watchdog OOM restarts. The cache
//! caps resident decompressed bytes at `DEFAULT_BLOCK_CACHE_BYTES`
//! (configurable per loader via `from_mmap_with_key_and_cap`).
//!
//! Lookups return `ArcArchive<T>` (an Arc-wrapped block handle plus
//! offset+length into the rkyv archive). The Arc keeps the bytes
//! alive for as long as any caller holds the handle, regardless of
//! LRU eviction state. AlignedVec guarantees a 16-byte-aligned start
//! address; per-entry offsets are 16-byte aligned within the block,
//! so slices into the AlignedVec are correctly aligned for
//! `archived_root`.
//!
//! Multiple readers may race on a cache miss: the cache mutex
//! serializes the install step (first-write-wins; losers drop their
//! decompressed buffer and read the winner's Arc clone). The lock
//! is held only across map mutation, never across decompression.

use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};

use lru::LruCache;

use memmap2::Mmap;

use crate::bond::TemporalBond;
use crate::brain::BrainError;
use crate::encrypt::{xor_in_place, MasterKey};
use crate::network::{NetworkSnapshot, NetworkSource};
use crate::node::KnowledgeNode;
use crate::seed::Seed;

/// `header.flags` bit 0 set → the file's network section is in the
/// per-block compressed layout described in this module's docstring.
/// Other flags bits remain reserved.
pub const FORMAT_FLAG_COMPRESSED: u32 = 0x0000_0001;

/// Default uncompressed-bytes target per block. 256 KB is the empirical
/// sweet spot in the offline measurement (`block_size=262144` →
/// 30.4 MB compressed for the production 424 MB network section, vs
/// 34 MB at 64 KB or 27 MB at 1 MB; smaller blocks page-fault less
/// data per access at runtime).
pub const DEFAULT_BLOCK_SIZE: usize = 256 * 1024;

/// Default zstd compression level. zstd levels 3 → 5 hit a well-known
/// elbow on rkyv-shaped data; the user's brief calls out that 22 is
/// too slow. Level 3 round-trips the production brain in ~0.5 s on the
/// Hetzner box and matches the ratio at level 5 within ~2%.
pub const DEFAULT_ZSTD_LEVEL: i32 = 3;

/// OP-016: default cap on bytes of decompressed blocks held resident
/// in the bounded LRU cache. 4 GiB. Configurable per loader via
/// `CompressedNetwork::from_mmap_with_key_and_cap`. The previous
/// monotonic `Vec<OnceLock<AlignedVec>>` arena (Phase 3) had no cap
/// and grew toward the full decompressed network size for retrain
/// workloads that touch every block, causing systemd-watchdog OOM
/// restarts. The new cap bounds resident bytes; the LRU evicts the
/// least-recently-touched block(s) until the total fits.
pub const DEFAULT_BLOCK_CACHE_BYTES: usize = 4 * 1024 * 1024 * 1024;

/// rkyv per-entity archives MUST be 16-byte aligned within a block so
/// `rkyv::archived_root` is sound. The block buffer's start is also
/// 16-byte aligned (`AlignedVec`), so an offset that is itself a
/// multiple of 16 keeps the resulting slice correctly aligned.
const ENTRY_ALIGN: usize = 16;

/// Size of the per-section header at the start of a compressed payload.
const SECTION_HEADER_SIZE: usize = 32;

/// On-disk node-index entry: 32-byte seed + (block_id, offset, length).
const NODE_INDEX_ENTRY_SIZE: usize = 32 + 4 + 4 + 4;

/// On-disk bond-index entry: 32-byte from-seed + 32-byte to-seed +
/// (block_id, offset, length).
const BOND_INDEX_ENTRY_SIZE: usize = 32 + 32 + 4 + 4 + 4;

/// On-disk block-offset-table entry.
const BLOCK_OFFSET_ENTRY_SIZE: usize = 8;

/// Per-block frame header: compressed_len + decompressed_len.
const BLOCK_FRAME_HEADER_SIZE: usize = 8;

// ---------------------------------------------------------------------
// Index records
// ---------------------------------------------------------------------

/// Where one archived `KnowledgeNode` lives in the compressed payload.
#[derive(Debug, Clone, Copy)]
pub struct NodeIndexEntry {
    pub block_id: u32,
    pub offset: u32,
    pub length: u32,
}

/// Where one archived `TemporalBond` lives in the compressed payload.
#[derive(Debug, Clone, Copy)]
pub struct BondIndexEntry {
    pub block_id: u32,
    pub offset: u32,
    pub length: u32,
}

// ---------------------------------------------------------------------
// OP-016: owning handle + bounded LRU cache for decompressed blocks
// ---------------------------------------------------------------------

/// Owning handle to a per-entity rkyv archive that lives inside a
/// decompressed block. Carries an `Arc<rkyv::AlignedVec>` clone of
/// the block buffer so the bytes stay alive while the handle is
/// held, even if the LRU has since evicted the entry from its map.
///
/// `T` is the unarchived shape (e.g. `KnowledgeNode`); `T::Archived`
/// (e.g. `ArchivedKnowledgeNode`) is what `archived()` returns.
pub struct ArcArchive<T: rkyv::Archive> {
    block: Arc<rkyv::AlignedVec>,
    offset: u32,
    length: u32,
    _phantom: PhantomData<fn() -> T>,
}

impl<T: rkyv::Archive> ArcArchive<T> {
    /// Borrow the archived entity. Lifetime ties to `&self`; the
    /// Arc keeps the bytes alive for as long as `self` is held.
    pub fn archived(&self) -> &T::Archived {
        let start = self.offset as usize;
        let end = start + self.length as usize;
        let slice = &self.block.as_slice()[start..end];
        // SAFETY: writer 16-byte aligns each entry's offset within
        // the block; AlignedVec start is 16-byte aligned; length is
        // the exact rkyv-serialized size recorded at write time. The
        // bytes were produced by this crate's writer and are covered
        // by the file-level CRC32 trailer (verified by
        // `parse_layout` upstream), so structural validation is
        // unnecessary on the hot read path.
        unsafe { rkyv::archived_root::<T>(slice) }
    }

    /// Underlying block Arc (advanced callers).
    pub fn block(&self) -> &Arc<rkyv::AlignedVec> {
        &self.block
    }
}

impl<T: rkyv::Archive> Clone for ArcArchive<T> {
    fn clone(&self) -> Self {
        Self {
            block: self.block.clone(),
            offset: self.offset,
            length: self.length,
            _phantom: PhantomData,
        }
    }
}

/// Internal: bounded LRU of decompressed blocks. The map carries
/// `Arc<AlignedVec>`; `resident_bytes` tracks the running total of
/// uncompressed bytes held in the map. Eviction pops the LRU back
/// until `resident_bytes <= cap_bytes`.
struct BlockCache {
    inner: LruCache<u32, Arc<rkyv::AlignedVec>>,
    cap_bytes: usize,
    resident_bytes: usize,
}

impl BlockCache {
    fn new(cap_bytes: usize) -> Self {
        Self {
            inner: LruCache::unbounded(),
            cap_bytes,
            resident_bytes: 0,
        }
    }

    /// Probe-and-clone-the-Arc, bumping the entry to MRU.
    fn get(&mut self, id: u32) -> Option<Arc<rkyv::AlignedVec>> {
        self.inner.get(&id).cloned()
    }

    /// Insert `buf` under `id`, then evict oldest entries until the
    /// resident byte total fits the cap. Always leaves at least one
    /// entry resident (so a single block larger than cap doesn't
    /// infinite-evict).
    fn insert(&mut self, id: u32, buf: Arc<rkyv::AlignedVec>) -> Arc<rkyv::AlignedVec> {
        let sz = buf.len();
        if let Some(prev) = self.inner.put(id, buf.clone()) {
            self.resident_bytes = self.resident_bytes.saturating_sub(prev.len());
        }
        self.resident_bytes = self.resident_bytes.saturating_add(sz);
        while self.resident_bytes > self.cap_bytes && self.inner.len() > 1 {
            if let Some((_, evicted)) = self.inner.pop_lru() {
                self.resident_bytes =
                    self.resident_bytes.saturating_sub(evicted.len());
            } else {
                break;
            }
        }
        buf
    }

    fn resident_count(&self) -> usize {
        self.inner.len()
    }

    fn resident_bytes_count(&self) -> usize {
        self.resident_bytes
    }
}

// ---------------------------------------------------------------------
// Writer
// ---------------------------------------------------------------------

/// Serialize a `NetworkSnapshot` into the per-block compressed v3
/// network-section payload. The returned `Vec<u8>` is exactly what
/// `brain_io_v3::write_v3_compressed` writes as the section data.
///
/// The function performs NO file I/O; it only produces the in-memory
/// payload. Padding/alignment of the section within the surrounding
/// v3 file is the caller's responsibility (handled by the same
/// `align_up` machinery that the uncompressed writer uses).
// ---------------------------------------------------------------------
// OP-PARZSTD (2026-06-01) — parallel zstd compression pipeline.
//
// The serial flush_current_block! macro spent ~39% of save wall-clock
// inside `zstd::bulk::Compressor::compress_to_buffer` and another ~12%
// inside the per-block sha2 keystream internal notes perf flamegraph).
// On a 16-vCPU host the entire block stream was funnelled through a
// single core; the other 15 sat idle. This module parallelises that
// work without changing on-disk bytes.
//
// Correctness rests on three load-bearing observations:
//
//  (a) Each block is an independent zstd frame.
//      `zstd::bulk::Compressor::compress_to_buffer` resets the CCtx
//      between calls and emits a self-contained frame — there is no
//      shared sliding window across blocks. Verified at the reader:
//      `CompressedNetwork::block_frame_bytes` slices a single frame
//      and `zstd::bulk::decompress_to_buffer` decodes it standalone
//      (see ~line 1534 of this file).
//
//  (b) Per-block XOR encryption is keyed on `(master_key, block_id)`.
//      `keystream_chunk` in src/encrypt.rs hashes
//      `sha256(key || block_index || chunk_index)`. Pure function of
//      block_id; identical output regardless of which thread computes
//      it.
//
//  (c) On-disk block layout is `block_id == frame_position`.
//      The reader matches each `block_offsets[i]` to logical block
//      `i`. As long as the writer thread emits frames in ascending
//      `block_id` order, byte layout is identical to the serial
//      writer.
//
// Pipeline shape:
//
//   producer thread (entity packer)
//          │
//          │  (block_id, raw_block) via bounded crossbeam channel
//          ▼
//   N worker threads (compress + XOR-encrypt)
//          │
//          │  CompressedBlock { block_id, bytes, decomp_len } via
//          │  bounded crossbeam channel
//          ▼
//   writer thread (BTreeMap reorder buffer, writes in block_id order,
//                  records block_offsets)
//
// Memory bound: both channels are bounded by `num_workers * 2`/`* 4`,
// so at most ~6N blocks of raw + compressed bytes are in flight
// (≤ ~3 MiB per block at production size → ≤ ~50 MiB at 16 workers).
//
// Passthrough preservation: this helper is only invoked from the
// slow re-encode writers. The OP-022.s.v3 full-passthrough writer
// (`write_compressed_network_payload_passthrough_to`) and the
// passthrough Pass 1 of the partial writer copy source frames
// verbatim and never touch this helper.
// ---------------------------------------------------------------------

use crossbeam_channel::{bounded, Receiver, Sender};

/// One result from a worker thread: the encrypted, compressed bytes
/// of `block_id`, along with the original decompressed length (which
/// the writer needs for the per-frame `decomp_len` header field).
struct CompressedBlock {
    block_id: u32,
    encrypted: Vec<u8>,
    decomp_len: u32,
}

/// Worker loop. Reads `(block_id, raw_block)` from `work_rx`, compresses
/// + (optionally) XOR-encrypts, and forwards a `CompressedBlock` to
/// `done_tx`. Reuses a single `zstd::bulk::Compressor` (CCtx) and a
/// scratch buffer across iterations — matches the OP-022.h hoist that
/// the serial writer relied on. Returns the first error encountered
/// (compression failure or oversized block); the scope then drains
/// remaining workers via join.
fn parzstd_worker(
    zstd_level: i32,
    block_size_target: usize,
    encryption_key: Option<MasterKey>,
    work_rx: Receiver<(u32, Vec<u8>)>,
    done_tx: Sender<CompressedBlock>,
    err_label: &'static str,
) -> Result<(), BrainError> {
    let mut compressor = zstd::bulk::Compressor::new(zstd_level)
        .map_err(|e| BrainError::Format(format!("{}: zstd init: {}", err_label, e)))?;
    let mut scratch: Vec<u8> = Vec::with_capacity(block_size_target + 16 * 1024);

    while let Ok((block_id, raw)) = work_rx.recv() {
        scratch.clear();
        compressor
            .compress_to_buffer(&raw, &mut scratch)
            .map_err(|e| BrainError::Format(format!("{}: zstd compress: {}", err_label, e)))?;
        if scratch.len() > u32::MAX as usize {
            return Err(BrainError::Format(format!(
                "{}: compressed block too large ({} bytes)",
                err_label,
                scratch.len()
            )));
        }
        if raw.len() > u32::MAX as usize {
            return Err(BrainError::Format(format!(
                "{}: decompressed block too large ({} bytes)",
                err_label,
                raw.len()
            )));
        }
        let decomp_len = raw.len() as u32;
        // Hand off `scratch` and re-prime it; this keeps a single
        // allocation alive per worker (one in `scratch`, one being
        // sent in flight), bounded by the channel capacity.
        let mut encrypted = std::mem::replace(
            &mut scratch,
            Vec::with_capacity(block_size_target + 16 * 1024),
        );
        if let Some(k) = encryption_key.as_ref() {
            xor_in_place(k, block_id as u64, &mut encrypted);
        }
        if done_tx
            .send(CompressedBlock { block_id, encrypted, decomp_len })
            .is_err()
        {
            // Writer thread has exited (likely due to its own error or
            // a prior worker error). Stop quietly; the scope will
            // surface the original error via join.
            return Ok(());
        }
    }
    Ok(())
}

/// Writer-thread loop. Receives `CompressedBlock`s, reorders them by
/// `block_id` via a `BTreeMap`, and writes ready frames in order to
/// `writer`. Tracks running absolute position so it can record one
/// `block_offsets[i]` per frame *without* calling `stream_position()`
/// (which would require seeking and break BufWriter coalescing). Caller
/// MUST pre-position `writer` at `frames_start_abs`.
///
/// Returns the final stream position (frames_end) and the recorded
/// block_offsets in block_id order. After return, caller can seek back
/// to `section_start` to write the header + indices + block_offsets.
fn parzstd_writer<W>(
    writer: &mut W,
    section_start: u64,
    frames_start_abs: u64,
    done_rx: Receiver<CompressedBlock>,
    err_label: &'static str,
    start_block_id: u32,
) -> Result<(u64, Vec<u64>), BrainError>
where
    W: std::io::Write,
{
    use std::collections::BTreeMap;
    let mut pending: BTreeMap<u32, CompressedBlock> = BTreeMap::new();
    let mut next: u32 = start_block_id;
    let mut cur_pos: u64 = frames_start_abs;
    let mut block_offsets: Vec<u64> = Vec::new();

    while let Ok(cb) = done_rx.recv() {
        pending.insert(cb.block_id, cb);
        while let Some(cb) = pending.remove(&next) {
            let cur_rel = cur_pos - section_start;
            block_offsets.push(cur_rel);
            let comp_len = (cb.encrypted.len() as u32).to_le_bytes();
            let decomp_len = cb.decomp_len.to_le_bytes();
            writer.write_all(&comp_len).map_err(BrainError::Io)?;
            writer.write_all(&decomp_len).map_err(BrainError::Io)?;
            writer.write_all(&cb.encrypted).map_err(BrainError::Io)?;
            cur_pos = cur_pos
                .checked_add(8u64 + cb.encrypted.len() as u64)
                .ok_or_else(|| {
                    BrainError::Format(format!(
                        "{}: frames stream position overflow",
                        err_label
                    ))
                })?;
            next = next.checked_add(1).ok_or_else(|| {
                BrainError::Format(format!(
                    "{}: block_id counter overflow",
                    err_label
                ))
            })?;
        }
    }

    if !pending.is_empty() {
        return Err(BrainError::Format(format!(
            "{}: writer exited with {} pending out-of-order blocks (first missing block_id={})",
            err_label,
            pending.len(),
            next
        )));
    }
    Ok((cur_pos, block_offsets))
}

/// Pick a worker count for the parallel compression pool. Caps at 16
/// (production AX41 vCPU count) so a beefier host doesn't starve other
/// concurrent save IO. Falls back to 4 if `available_parallelism`
/// fails to report.
fn parzstd_worker_count() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get().min(16))
        .unwrap_or(4)
        .max(1)
}


pub fn write_compressed_network_payload(
    snapshot: &NetworkSnapshot,
    block_size_target: usize,
    zstd_level: i32,
    encryption_key: Option<&MasterKey>,
) -> Result<Vec<u8>, BrainError> {
    // OP-017 — thin wrapper around the streaming writer for callers
    // (tests, migration tools) that still want a `Vec<u8>` payload. The
    // production save path goes through `write_compressed_network_payload_to`
    // directly against the tmp file to avoid materialising 2.3 GiB of
    // bytes on the heap during save.
    use std::io::Cursor;
    let mut cursor = Cursor::new(Vec::new());
    write_compressed_network_payload_to(
        snapshot,
        block_size_target,
        zstd_level,
        encryption_key,
        &mut cursor,
    )?;
    Ok(cursor.into_inner())
}

/// OP-017 streaming writer. Writes the compressed-network section
/// directly to `writer` starting at its current stream position.
///
/// Memory profile vs the legacy `Vec<u8>`-returning variant:
/// - Holds at most ONE rkyv-serialised entity buffer + ONE in-flight
///   block buffer (~272 KiB) + the per-entity index Vecs in memory
///   (node_count*44 B + bond_count*76 B ≈ 600 MiB at production scale).
/// - Compressed frame bodies are written directly to disk and dropped;
///   never accumulated in a `Vec<Vec<u8>>`.
/// - No `NetworkSnapshot` is materialised when called against an
///   `AssociativeNetwork` source — each archived node/bond is decoded
///   into a single owned heap copy that drops at the end of its loop
///   iteration.
///
/// Algorithm:
///   (a) pre-pass: walk the NetworkSource, rkyv-serialise each entity
///       to learn its byte length, simulate block-packing to compute
///       the exact `block_count`. Buffers are dropped after each
///       entity — one allocation alive at a time.
///   (b) reserve space at `writer` for [section_header | node_index |
///       bond_index | block_offset_table] by writing zero bytes.
///   (c) write-pass: walk the source again, rkyv-serialise each entity
///       (must produce byte-identical sizes to pass (a)), pack into
///       `current_block`, flush when full — compress, optionally
///       XOR-encrypt, write frame header + body to `writer`. Record
///       (block_id, offset, length) per entity and the frame's section-
///       relative start offset per block.
///   (d) seek back to `section_start`, write the real section_header,
///       node_index entries, bond_index entries, block_offset_table.
///       Restore writer to section end. Return total bytes written.
///
/// Pre-pass + write-pass packing decisions MUST agree. The shared
/// `if entry_end > block_size_target && current > 0 { flush }` rule is
/// applied byte-identically against the same iteration order (the
/// NetworkSource trait contract). A mismatch triggers a defensive
/// `BrainError::Format` rather than silently writing a corrupt section.
#[allow(unused_assignments)] // last flush bumps current_block_id unread
#[allow(unused_assignments)] // last flush bumps current_block_id unread
pub fn write_compressed_network_payload_to<N, W>(
    net: &N,
    block_size_target: usize,
    zstd_level: i32,
    encryption_key: Option<&MasterKey>,
    writer: &mut W,
) -> Result<u64, BrainError>
where
    N: NetworkSource + ?Sized,
    W: std::io::Write + std::io::Seek + Send,
{
    use std::io::SeekFrom;

    if !(1..=22).contains(&zstd_level) {
        return Err(BrainError::Format(format!(
            "compress: zstd level {} out of range 1..=22",
            zstd_level
        )));
    }
    if block_size_target < 4 * 1024 {
        return Err(BrainError::Format(format!(
            "compress: block_size_target {} too small (minimum 4 KiB)",
            block_size_target
        )));
    }

    let node_count = net.node_count();
    let bond_count = net.bond_count();

    // OP-017 v2 — see top-of-function comment in the pre-OP-PARZSTD
    // revision (preserved verbatim): skip the rkyv pre-pass and
    // reserve a worst-case block_offset_table sized at
    // `node_count + bond_count`.
    let block_count_upper = node_count
        .checked_add(bond_count)
        .ok_or_else(|| BrainError::Format("compress: entity count overflow".into()))?;
    if block_count_upper > u32::MAX as usize {
        return Err(BrainError::Format(format!(
            "compress: block_count_upper too large ({})",
            block_count_upper
        )));
    }

    let section_start = writer.stream_position().map_err(BrainError::Io)?;

    let header_size = SECTION_HEADER_SIZE as u64;
    let nodex_size = (node_count as u64) * (NODE_INDEX_ENTRY_SIZE as u64);
    let bondx_size = (bond_count as u64) * (BOND_INDEX_ENTRY_SIZE as u64);
    let table_size_upper = (block_count_upper as u64) * (BLOCK_OFFSET_ENTRY_SIZE as u64);
    let frames_start_rel = header_size + nodex_size + bondx_size + table_size_upper;

    // Zero-fill the placeholder (section_header + indices + reserved
    // table). The CRC pass downstream reads these bytes back; we need
    // them deterministic. 64 KiB scratch buffer.
    {
        let zeros = [0u8; 64 * 1024];
        let mut remaining = frames_start_rel as usize;
        while remaining > 0 {
            let n = remaining.min(zeros.len());
            writer.write_all(&zeros[..n]).map_err(BrainError::Io)?;
            remaining -= n;
        }
    }
    debug_assert_eq!(
        writer.stream_position().map_err(BrainError::Io)?,
        section_start + frames_start_rel
    );
    let frames_start_abs = section_start + frames_start_rel;

    // ---- Per-entity packing state (producer-side). ----
    let mut current_block: Vec<u8> =
        Vec::with_capacity(block_size_target + 16 * 1024);
    let mut current_block_id: u32 = 0;
    let mut node_seeds: Vec<Seed> = Vec::with_capacity(node_count);
    let mut node_entries: Vec<NodeIndexEntry> = Vec::with_capacity(node_count);
    let mut bond_keys: Vec<(Seed, Seed)> = Vec::with_capacity(bond_count);
    let mut bond_entries: Vec<BondIndexEntry> = Vec::with_capacity(bond_count);

    // OP-PARZSTD — spawn workers + writer thread in a scoped pool. The
    // producer (this thread, inside the scope) packs entities into
    // blocks and submits each completed block to the work channel.
    // Workers compress + XOR-encrypt in parallel; the writer thread
    // reorders by block_id and emits frames sequentially.
    let num_workers = parzstd_worker_count();
    let key_for_workers: Option<MasterKey> = encryption_key.cloned();

    let (work_tx, work_rx) = bounded::<(u32, Vec<u8>)>(num_workers * 2);
    let (done_tx, done_rx) = bounded::<CompressedBlock>(num_workers * 4);

    let (frames_end, block_offsets) = std::thread::scope(|s| -> Result<(u64, Vec<u64>), BrainError> {
        // Workers.
        let mut worker_handles = Vec::with_capacity(num_workers);
        for _ in 0..num_workers {
            let work_rx = work_rx.clone();
            let done_tx = done_tx.clone();
            let key = key_for_workers.clone();
            worker_handles.push(s.spawn(move || {
                parzstd_worker(
                    zstd_level,
                    block_size_target,
                    key,
                    work_rx,
                    done_tx,
                    "compress",
                )
            }));
        }
        // Drop the originals so the channels close once the producer
        // / workers drop their clones.
        drop(work_rx);
        drop(done_tx);

        // Writer thread.
        let writer_handle = s.spawn(|| {
            parzstd_writer(writer, section_start, frames_start_abs, done_rx, "compress", 0u32)
        });

        // Producer flush macro: submits the current block to the work
        // channel and primes a fresh `current_block` buffer. Bumps
        // `current_block_id`. If the channel is closed (workers all
        // died), surfaces a Format error; the worker/writer errors
        // come out at join time below and take precedence.
        macro_rules! flush_current_block {
            () => {{
                if !current_block.is_empty() {
                    let raw = std::mem::replace(
                        &mut current_block,
                        Vec::with_capacity(block_size_target + 16 * 1024),
                    );
                    if work_tx.send((current_block_id, raw)).is_err() {
                        // Workers/writer have all exited. Break out of
                        // the packing loop; the real error surfaces
                        // when we join below.
                        return Err(BrainError::Format(
                            "compress: parallel compression pool closed prematurely".into(),
                        ));
                    }
                    current_block_id = current_block_id.checked_add(1).ok_or_else(|| {
                        BrainError::Format(
                            "compress: current_block_id overflow".into(),
                        )
                    })?;
                }
            }};
        }

        // -------- nodes --------
        net.for_each_node(|node| {
            let bytes = rkyv::to_bytes::<_, 4096>(node).map_err(|e| {
                BrainError::Format(format!(
                    "compress: rkyv serialize KnowledgeNode (seed={:x?}): {}",
                    node.seed.0, e
                ))
            })?;
            let pad = pad_for(current_block.len(), ENTRY_ALIGN);
            let entry_start_in_block = current_block.len() + pad;
            let entry_end_in_block = entry_start_in_block + bytes.len();
            if entry_end_in_block > block_size_target && !current_block.is_empty() {
                flush_current_block!();
            }
            let pad = pad_for(current_block.len(), ENTRY_ALIGN);
            current_block.resize(current_block.len() + pad, 0);
            let offset = current_block.len();
            current_block.extend_from_slice(&bytes);
            let length = bytes.len();
            if offset > u32::MAX as usize || length > u32::MAX as usize {
                return Err(BrainError::Format(format!(
                    "compress: node entry exceeds u32 (offset={} len={})",
                    offset, length
                )));
            }
            node_seeds.push(node.seed);
            node_entries.push(NodeIndexEntry {
                block_id: current_block_id,
                offset: offset as u32,
                length: length as u32,
            });
            Ok(())
        })?;
        flush_current_block!();

        // -------- bonds --------
        net.for_each_bond(|bond| {
            let bytes = rkyv::to_bytes::<_, 1024>(bond).map_err(|e| {
                BrainError::Format(format!(
                    "compress: rkyv serialize TemporalBond (from={:x?}, to={:x?}): {}",
                    bond.from.0, bond.to.0, e
                ))
            })?;
            let pad = pad_for(current_block.len(), ENTRY_ALIGN);
            let entry_end_in_block = current_block.len() + pad + bytes.len();
            if entry_end_in_block > block_size_target && !current_block.is_empty() {
                flush_current_block!();
            }
            let pad = pad_for(current_block.len(), ENTRY_ALIGN);
            current_block.resize(current_block.len() + pad, 0);
            let offset = current_block.len();
            current_block.extend_from_slice(&bytes);
            let length = bytes.len();
            if offset > u32::MAX as usize || length > u32::MAX as usize {
                return Err(BrainError::Format(format!(
                    "compress: bond entry exceeds u32 (offset={} len={})",
                    offset, length
                )));
            }
            bond_keys.push((bond.from, bond.to));
            bond_entries.push(BondIndexEntry {
                block_id: current_block_id,
                offset: offset as u32,
                length: length as u32,
            });
            Ok(())
        })?;
        flush_current_block!();

        // Producer done — drop work_tx so workers see EOF and exit;
        // they drop their done_tx clones, then the writer thread sees
        // EOF and exits.
        drop(work_tx);

        // Join workers first so their errors take precedence.
        for h in worker_handles {
            h.join()
                .map_err(|_| BrainError::Format("compress: worker thread panicked".into()))??;
        }
        let (frames_end, block_offsets) = writer_handle
            .join()
            .map_err(|_| BrainError::Format("compress: writer thread panicked".into()))??;

        Ok((frames_end, block_offsets))
    })?;

    let block_count_real = block_offsets.len();
    if block_count_real > block_count_upper {
        // Should be impossible — we reserved one block per entity
        // worst-case, and the packer never emits more blocks than
        // entities.
        return Err(BrainError::Format(format!(
            "compress: block_count_real ({}) exceeds reserved upper bound ({})",
            block_count_real, block_count_upper
        )));
    }
    let block_count = block_count_real as u32;

    let section_total = frames_end - section_start;

    // ---- Seek back, fill section_header + indices + real block_offsets ----
    writer
        .seek(SeekFrom::Start(section_start))
        .map_err(BrainError::Io)?;

    let mut hdr = [0u8; SECTION_HEADER_SIZE];
    hdr[0..4].copy_from_slice(&(node_count as u32).to_le_bytes());
    hdr[4..8].copy_from_slice(&(bond_count as u32).to_le_bytes());
    hdr[8..12].copy_from_slice(&block_count.to_le_bytes());
    hdr[12..16].copy_from_slice(&(block_size_target as u32).to_le_bytes());
    hdr[16..20].copy_from_slice(&(zstd_level as u32).to_le_bytes());
    // bytes 20..32 reserved (already zero)
    writer.write_all(&hdr).map_err(BrainError::Io)?;

    for (seed, e) in node_seeds.iter().zip(node_entries.iter()) {
        writer.write_all(&seed.0).map_err(BrainError::Io)?;
        writer.write_all(&e.block_id.to_le_bytes()).map_err(BrainError::Io)?;
        writer.write_all(&e.offset.to_le_bytes()).map_err(BrainError::Io)?;
        writer.write_all(&e.length.to_le_bytes()).map_err(BrainError::Io)?;
    }
    for ((from, to), e) in bond_keys.iter().zip(bond_entries.iter()) {
        writer.write_all(&from.0).map_err(BrainError::Io)?;
        writer.write_all(&to.0).map_err(BrainError::Io)?;
        writer.write_all(&e.block_id.to_le_bytes()).map_err(BrainError::Io)?;
        writer.write_all(&e.offset.to_le_bytes()).map_err(BrainError::Io)?;
        writer.write_all(&e.length.to_le_bytes()).map_err(BrainError::Io)?;
    }
    // Write REAL block_offsets at the START of the reserved region.
    // Trailing (block_count_upper - block_count_real) * 8 bytes stay
    // zero from the placeholder pass — invisible to the reader
    // which sizes the table from hdr.block_count.
    for off in &block_offsets {
        writer.write_all(&off.to_le_bytes()).map_err(BrainError::Io)?;
    }

    // Restore writer to the section end so the caller can continue
    // streaming the next section.
    writer.seek(SeekFrom::Start(frames_end)).map_err(BrainError::Io)?;

    Ok(section_total)
}


// ---------------------------------------------------------------------
// OP-022.s.v3 — block-passthrough save path
// ---------------------------------------------------------------------

/// Parameters that govern the compressed-network section's on-disk
/// shape. Two sources are byte-compatible for passthrough iff every
/// field matches.
#[derive(Debug, Clone, Copy)]
pub struct CompressedSaveParams<'a> {
    pub block_size_target: usize,
    pub zstd_level: i32,
    pub encryption_key: Option<&'a MasterKey>,
}

/// Eligibility check: copy the source's compressed blocks straight to
/// the output writer iff every parameter matches AND the source's
/// encryption key (if any) is bytewise identical to the save key.
///
/// The caller is responsible for confirming there are NO pending
/// overlay mutations to the in-memory network — the source mmap'd
/// nodes/bonds must reflect the current state. This check sees only
/// the parameters, not the overlay.
pub fn can_passthrough_compressed_blocks(
    source: &CompressedNetwork,
    save: &CompressedSaveParams,
) -> bool {
    if source.block_size_target() as usize != save.block_size_target { return false; }
    if source.zstd_level() as i32 != save.zstd_level { return false; }
    match (source.encryption_key(), save.encryption_key) {
        (None, None) => {}
        (Some(a), Some(b)) => {
            if a.as_bytes() != b.as_bytes() { return false; }
        }
        _ => return false,
    }
    true
}

/// OP-022.s.v3 streaming passthrough writer. Emits a compressed-
/// network section to `writer` by copying every compressed block frame
/// verbatim from the source mmap — no zstd decompress, no XOR
/// decrypt/re-encrypt, no rkyv decode.
///
/// Block indices are preserved 1:1 (block N in source → block N in
/// output), so the per-block keystream (which is keyed on
/// `(master_key, block_index)`) stays valid and the output frames are
/// byte-identical to the source frames. Node and bond indices are
/// copied through unchanged because their `(block_id, offset_in_block,
/// length)` triples reference positions within the DECOMPRESSED block,
/// which is byte-identical when the block bytes are byte-identical.
///
/// Returns the total section length in bytes.
#[allow(clippy::too_many_lines)]
pub fn write_compressed_network_payload_passthrough_to<W>(
    source: &CompressedNetwork,
    save: &CompressedSaveParams,
    writer: &mut W,
) -> Result<u64, BrainError>
where
    W: std::io::Write + std::io::Seek,
{
    use std::io::SeekFrom;

    // Defensive: the caller should have already gated on this, but a
    // mismatch here would silently corrupt the output. Refuse loudly.
    if !can_passthrough_compressed_blocks(source, save) {
        return Err(BrainError::Format(
            "passthrough writer called with mismatched save params".into(),
        ));
    }

    let section_start = writer.stream_position().map_err(BrainError::Io)?;

    let node_count = source.node_count();
    let bond_count = source.bond_count();
    let block_count = source.block_count();
    if block_count > u32::MAX as usize {
        return Err(BrainError::Format(format!(
            "passthrough: block_count {} exceeds u32", block_count
        )));
    }

    let header_size = SECTION_HEADER_SIZE as u64;
    let nodex_size = (node_count as u64) * (NODE_INDEX_ENTRY_SIZE as u64);
    let bondx_size = (bond_count as u64) * (BOND_INDEX_ENTRY_SIZE as u64);
    let table_size = (block_count as u64) * (BLOCK_OFFSET_ENTRY_SIZE as u64);
    let frames_start_rel = header_size + nodex_size + bondx_size + table_size;

    // Reserve header + indices + offset table.
    {
        let zeros = [0u8; 64 * 1024];
        let mut remaining = frames_start_rel as usize;
        while remaining > 0 {
            let n = remaining.min(zeros.len());
            writer.write_all(&zeros[..n]).map_err(BrainError::Io)?;
            remaining -= n;
        }
    }

    // Copy each compressed block frame verbatim into the output and
    // record its new section-relative offset.
    let mut new_block_offsets: Vec<u64> = Vec::with_capacity(block_count);
    for block_id in 0..(block_count as u32) {
        let cur_abs = writer.stream_position().map_err(BrainError::Io)?;
        let cur_rel = cur_abs - section_start;
        new_block_offsets.push(cur_rel);
        let frame = source.block_frame_bytes(block_id)?;
        writer.write_all(frame).map_err(BrainError::Io)?;
    }

    let frames_end = writer.stream_position().map_err(BrainError::Io)?;
    let section_total = frames_end - section_start;

    // Seek back, write header + node_index + bond_index + block_offsets.
    writer
        .seek(SeekFrom::Start(section_start))
        .map_err(BrainError::Io)?;

    let mut hdr = [0u8; SECTION_HEADER_SIZE];
    hdr[0..4].copy_from_slice(&(node_count as u32).to_le_bytes());
    hdr[4..8].copy_from_slice(&(bond_count as u32).to_le_bytes());
    hdr[8..12].copy_from_slice(&(block_count as u32).to_le_bytes());
    hdr[12..16].copy_from_slice(&(save.block_size_target as u32).to_le_bytes());
    hdr[16..20].copy_from_slice(&(save.zstd_level as u32).to_le_bytes());
    // bytes 20..32 reserved (already zero)
    writer.write_all(&hdr).map_err(BrainError::Io)?;

    for (seed, e) in source.node_index_entries() {
        writer.write_all(&seed.0).map_err(BrainError::Io)?;
        writer.write_all(&e.block_id.to_le_bytes()).map_err(BrainError::Io)?;
        writer.write_all(&e.offset.to_le_bytes()).map_err(BrainError::Io)?;
        writer.write_all(&e.length.to_le_bytes()).map_err(BrainError::Io)?;
    }
    for ((from, to), e) in source.bond_index_entries() {
        writer.write_all(&from.0).map_err(BrainError::Io)?;
        writer.write_all(&to.0).map_err(BrainError::Io)?;
        writer.write_all(&e.block_id.to_le_bytes()).map_err(BrainError::Io)?;
        writer.write_all(&e.offset.to_le_bytes()).map_err(BrainError::Io)?;
        writer.write_all(&e.length.to_le_bytes()).map_err(BrainError::Io)?;
    }
    for off in &new_block_offsets {
        writer.write_all(&off.to_le_bytes()).map_err(BrainError::Io)?;
    }

    writer.seek(SeekFrom::Start(frames_end)).map_err(BrainError::Io)?;
    Ok(section_total)
}


// ---------------------------------------------------------------------
// OP-022.s.v4 — partial-passthrough save path
// (handles continuous-training overlay: archive blocks pass through
//  byte-identically while overlay nodes/bonds are freshly encoded into
//  appended blocks. node_index/bond_index are merged with shadow- and
//  tombstone-filtering so overlay versions always win on collision.)
// ---------------------------------------------------------------------

/// OP-022.s.v4 — eligibility for the partial-passthrough writer.
///
/// Identical to `can_passthrough_compressed_blocks` (block_size, zstd
/// level, encryption key must all match). Overlay being non-empty is
/// allowed — the v4 writer handles it.
///
/// Additionally guards `block_count + overlay_upper` against u32
/// overflow (combined block-id space).
pub fn can_passthrough_blocks_with_overlay(
    source: &CompressedNetwork,
    save: &CompressedSaveParams,
    overlay_block_upper: usize,
) -> bool {
    if !can_passthrough_compressed_blocks(source, save) {
        return false;
    }
    let combined = source.block_count().checked_add(overlay_block_upper);
    match combined {
        Some(c) if c <= u32::MAX as usize => true,
        _ => false,
    }
}

/// OP-022.s.v4 — partial-passthrough writer. Streams a compressed-
/// network section to `writer` by:
///   1. Copying every source compressed-block frame verbatim into
///      the output at preserved block_ids 0..N-1 (keystream alignment
///      stays valid because block_id is part of the XOR key).
///   2. Freshly encoding overlay nodes (then overlay bonds) into
///      appended blocks at block_ids N..N+M-1, using the same
///      `flush_current_block!` packing/encryption logic as
///      `write_compressed_network_payload_to`.
///   3. Building a merged node_index: every source node, EXCEPT
///      (a) seeds in `tombstones` and (b) seeds shadowed by the
///      overlay (overlay has same seed → overlay version wins); plus
///      every overlay node (always included).
///   4. Building a merged bond_index the same way.
///   5. Building a block_offset_table with combined N+M entries.
///   6. Seeking back, writing real header + indices + offsets.
///
/// Reader path needs zero changes — the output is shape-identical to a
/// full re-encode. From the reader's perspective, a partial-passthrough
/// file is indistinguishable from a clean re-encoded file.
///
/// Wasted disk: tombstoned/shadowed source entries' bytes remain inside
/// passthrough block frames (their seeds simply don't appear in the
/// merged index, so the reader never reaches them). Accepted small
/// waste for v4; compaction is v5 if it ever matters.
#[allow(clippy::too_many_lines, unused_assignments)]
pub fn write_compressed_network_payload_partial_passthrough_to<W>(
    source: &CompressedNetwork,
    save: &CompressedSaveParams,
    tombstoned_nodes: &std::collections::HashSet<Seed>,
    tombstoned_bonds: &std::collections::HashSet<(Seed, Seed)>,
    overlay_nodes: &dyn OverlayNodeSource,
    overlay_bonds: &dyn OverlayBondSource,
    writer: &mut W,
) -> Result<u64, BrainError>
where
    W: std::io::Write + std::io::Seek + Send,
{
    use std::io::SeekFrom;

    // Defensive: caller MUST gate on the predicate. A mismatch here
    // would silently corrupt the output (e.g. mixed encryption keys
    // across block_ids).
    let overlay_node_count = overlay_nodes.count();
    let overlay_bond_count = overlay_bonds.count();
    let overlay_block_upper = overlay_node_count
        .checked_add(overlay_bond_count)
        .ok_or_else(|| {
            BrainError::Format("partial-passthrough: overlay entity count overflow".into())
        })?;
    if !can_passthrough_blocks_with_overlay(source, save, overlay_block_upper) {
        return Err(BrainError::Format(
            "partial-passthrough writer called with mismatched save params or block overflow".into(),
        ));
    }

    // ---- Collect overlay seeds so we can SHADOW-EXCLUDE archive entries ----
    let mut shadow_node_seeds: std::collections::HashSet<Seed> =
        std::collections::HashSet::with_capacity(overlay_node_count);
    overlay_nodes.for_each_seed(&mut |s| {
        shadow_node_seeds.insert(*s);
    });
    let mut shadow_bond_keys: std::collections::HashSet<(Seed, Seed)> =
        std::collections::HashSet::with_capacity(overlay_bond_count);
    overlay_bonds.for_each_key(&mut |k| {
        shadow_bond_keys.insert(*k);
    });

    let section_start = writer.stream_position().map_err(BrainError::Io)?;

    let _src_node_count = source.node_count();
    let _src_bond_count = source.bond_count();
    let src_block_count = source.block_count();

    // ---- Compute filtered source-index size ---------------------------
    let mut src_live_node_count: usize = 0;
    for (seed, _) in source.node_index_entries() {
        if tombstoned_nodes.contains(&seed) { continue; }
        if shadow_node_seeds.contains(&seed) { continue; }
        src_live_node_count += 1;
    }
    let mut src_live_bond_count: usize = 0;
    for (key, _) in source.bond_index_entries() {
        if tombstoned_bonds.contains(&key) { continue; }
        if shadow_bond_keys.contains(&key) { continue; }
        src_live_bond_count += 1;
    }

    let final_node_count = src_live_node_count + overlay_node_count;
    let final_bond_count = src_live_bond_count + overlay_bond_count;
    if final_node_count > u32::MAX as usize
        || final_bond_count > u32::MAX as usize
    {
        return Err(BrainError::Format(format!(
            "partial-passthrough: merged counts exceed u32 (nodes={}, bonds={})",
            final_node_count, final_bond_count
        )));
    }

    // ---- Reserve header + indices + (src + overlay-upper) block-offset table ----
    let block_count_upper = src_block_count
        .checked_add(overlay_block_upper)
        .ok_or_else(|| {
            BrainError::Format("partial-passthrough: combined block count overflow".into())
        })?;
    let header_size = SECTION_HEADER_SIZE as u64;
    let nodex_size = (final_node_count as u64) * (NODE_INDEX_ENTRY_SIZE as u64);
    let bondx_size = (final_bond_count as u64) * (BOND_INDEX_ENTRY_SIZE as u64);
    let table_size_upper = (block_count_upper as u64) * (BLOCK_OFFSET_ENTRY_SIZE as u64);
    let frames_start_rel = header_size + nodex_size + bondx_size + table_size_upper;

    {
        let zeros = [0u8; 64 * 1024];
        let mut remaining = frames_start_rel as usize;
        while remaining > 0 {
            let n = remaining.min(zeros.len());
            writer.write_all(&zeros[..n]).map_err(BrainError::Io)?;
            remaining -= n;
        }
    }

    // ---- Pass 1: copy source block frames verbatim ------------------
    // OP-022.s.v3 passthrough — unchanged. block_id stays 0..N-1 so
    // encryption keystream alignment is preserved on disk.
    let mut new_block_offsets: Vec<u64> =
        Vec::with_capacity(block_count_upper);
    for block_id in 0..(src_block_count as u32) {
        let cur_abs = writer.stream_position().map_err(BrainError::Io)?;
        let cur_rel = cur_abs - section_start;
        new_block_offsets.push(cur_rel);
        let frame = source.block_frame_bytes(block_id)?;
        writer.write_all(frame).map_err(BrainError::Io)?;
    }

    // ---- Pass 2: encode overlay nodes + overlay bonds into new blocks ---
    // OP-PARZSTD parallelises this re-encode pass (Pass 1 is the
    // passthrough fast-path and stays serial — it's just a memcpy of
    // already-encrypted source frames).
    //
    // block_id starts at src_block_count and continues monotonically
    // so keystream alignment is preserved across the boundary.
    let encryption_key = save.encryption_key;
    let block_size_target = save.block_size_target;
    let zstd_level = save.zstd_level;

    let mut current_block: Vec<u8> =
        Vec::with_capacity(block_size_target + 16 * 1024);
    let mut current_block_id: u32 = src_block_count as u32;
    let mut overlay_node_seeds: Vec<Seed> = Vec::with_capacity(overlay_node_count);
    let mut overlay_node_entries: Vec<NodeIndexEntry> =
        Vec::with_capacity(overlay_node_count);
    let mut overlay_bond_keys_vec: Vec<(Seed, Seed)> =
        Vec::with_capacity(overlay_bond_count);
    let mut overlay_bond_entries: Vec<BondIndexEntry> =
        Vec::with_capacity(overlay_bond_count);

    // Capture absolute frames-start for the parallel writer (the
    // position right after the Pass 1 source-frame copy).
    let pass2_start_abs = writer.stream_position().map_err(BrainError::Io)?;

    let num_workers = parzstd_worker_count();
    let key_for_workers: Option<MasterKey> = encryption_key.cloned();

    let (work_tx, work_rx) = bounded::<(u32, Vec<u8>)>(num_workers * 2);
    let (done_tx, done_rx) = bounded::<CompressedBlock>(num_workers * 4);

    let start_block_id = current_block_id;
    let (pass2_end_abs, pass2_block_offsets) = std::thread::scope(
        |s| -> Result<(u64, Vec<u64>), BrainError> {
            let mut worker_handles = Vec::with_capacity(num_workers);
            for _ in 0..num_workers {
                let work_rx = work_rx.clone();
                let done_tx = done_tx.clone();
                let key = key_for_workers.clone();
                worker_handles.push(s.spawn(move || {
                    parzstd_worker(
                        zstd_level,
                        block_size_target,
                        key,
                        work_rx,
                        done_tx,
                        "partial-passthrough",
                    )
                }));
            }
            drop(work_rx);
            drop(done_tx);

            let writer_handle = s.spawn(|| {
                parzstd_writer(
                    writer,
                    section_start,
                    pass2_start_abs,
                    done_rx,
                    "partial-passthrough",
                    start_block_id,
                )
            });

            macro_rules! flush_current_block {
                () => {{
                    if !current_block.is_empty() {
                        let raw = std::mem::replace(
                            &mut current_block,
                            Vec::with_capacity(block_size_target + 16 * 1024),
                        );
                        if work_tx.send((current_block_id, raw)).is_err() {
                            return Err(BrainError::Format(
                                "partial-passthrough: parallel compression pool closed prematurely".into(),
                            ));
                        }
                        current_block_id = current_block_id.checked_add(1).ok_or_else(|| {
                            BrainError::Format(
                                "partial-passthrough: current_block_id overflow".into(),
                            )
                        })?;
                    }
                }};
            }

            // Overlay nodes
            overlay_nodes.for_each_node(&mut |node| {
                let bytes = rkyv::to_bytes::<_, 4096>(node).map_err(|e| {
                    BrainError::Format(format!(
                        "partial-passthrough: rkyv KnowledgeNode (seed={:x?}): {}",
                        node.seed.0, e
                    ))
                })?;
                let pad = pad_for(current_block.len(), ENTRY_ALIGN);
                let entry_end_in_block = current_block.len() + pad + bytes.len();
                if entry_end_in_block > block_size_target && !current_block.is_empty() {
                    flush_current_block!();
                }
                let pad = pad_for(current_block.len(), ENTRY_ALIGN);
                current_block.resize(current_block.len() + pad, 0);
                let offset = current_block.len();
                current_block.extend_from_slice(&bytes);
                let length = bytes.len();
                if offset > u32::MAX as usize || length > u32::MAX as usize {
                    return Err(BrainError::Format(format!(
                        "partial-passthrough: node entry exceeds u32 (offset={} len={})",
                        offset, length
                    )));
                }
                overlay_node_seeds.push(node.seed);
                overlay_node_entries.push(NodeIndexEntry {
                    block_id: current_block_id,
                    offset: offset as u32,
                    length: length as u32,
                });
                Ok(())
            })?;
            flush_current_block!();

            // Overlay bonds
            overlay_bonds.for_each_bond(&mut |bond| {
                let bytes = rkyv::to_bytes::<_, 1024>(bond).map_err(|e| {
                    BrainError::Format(format!(
                        "partial-passthrough: rkyv TemporalBond (from={:x?}, to={:x?}): {}",
                        bond.from.0, bond.to.0, e
                    ))
                })?;
                let pad = pad_for(current_block.len(), ENTRY_ALIGN);
                let entry_end_in_block = current_block.len() + pad + bytes.len();
                if entry_end_in_block > block_size_target && !current_block.is_empty() {
                    flush_current_block!();
                }
                let pad = pad_for(current_block.len(), ENTRY_ALIGN);
                current_block.resize(current_block.len() + pad, 0);
                let offset = current_block.len();
                current_block.extend_from_slice(&bytes);
                let length = bytes.len();
                if offset > u32::MAX as usize || length > u32::MAX as usize {
                    return Err(BrainError::Format(format!(
                        "partial-passthrough: bond entry exceeds u32 (offset={} len={})",
                        offset, length
                    )));
                }
                overlay_bond_keys_vec.push((bond.from, bond.to));
                overlay_bond_entries.push(BondIndexEntry {
                    block_id: current_block_id,
                    offset: offset as u32,
                    length: length as u32,
                });
                Ok(())
            })?;
            flush_current_block!();

            drop(work_tx);

            for h in worker_handles {
                h.join().map_err(|_| {
                    BrainError::Format("partial-passthrough: worker thread panicked".into())
                })??;
            }
            let (end, offs) = writer_handle.join().map_err(|_| {
                BrainError::Format("partial-passthrough: writer thread panicked".into())
            })??;
            Ok((end, offs))
        },
    )?;

    new_block_offsets.extend(pass2_block_offsets);
    // Sanity: writer thread starts cur_pos at pass2_start_abs and only
    // ever forwards. pass2_end_abs == writer.stream_position() now.
    let frames_end = pass2_end_abs;

    let final_block_count = new_block_offsets.len();
    if final_block_count > u32::MAX as usize {
        return Err(BrainError::Format(format!(
            "partial-passthrough: final block count {} exceeds u32",
            final_block_count
        )));
    }

    let section_total = frames_end - section_start;

    // ---- Seek back, write real header + merged indices + block offsets ----
    writer
        .seek(SeekFrom::Start(section_start))
        .map_err(BrainError::Io)?;

    let mut hdr = [0u8; SECTION_HEADER_SIZE];
    hdr[0..4].copy_from_slice(&(final_node_count as u32).to_le_bytes());
    hdr[4..8].copy_from_slice(&(final_bond_count as u32).to_le_bytes());
    hdr[8..12].copy_from_slice(&(final_block_count as u32).to_le_bytes());
    hdr[12..16].copy_from_slice(&(save.block_size_target as u32).to_le_bytes());
    hdr[16..20].copy_from_slice(&(save.zstd_level as u32).to_le_bytes());
    writer.write_all(&hdr).map_err(BrainError::Io)?;

    // Merged node index: live source entries first (in source iteration
    // order, byte-identical to source layout), then overlay entries.
    for (seed, e) in source.node_index_entries() {
        if tombstoned_nodes.contains(&seed) { continue; }
        if shadow_node_seeds.contains(&seed) { continue; }
        writer.write_all(&seed.0).map_err(BrainError::Io)?;
        writer.write_all(&e.block_id.to_le_bytes()).map_err(BrainError::Io)?;
        writer.write_all(&e.offset.to_le_bytes()).map_err(BrainError::Io)?;
        writer.write_all(&e.length.to_le_bytes()).map_err(BrainError::Io)?;
    }
    for (seed, e) in overlay_node_seeds.iter().zip(overlay_node_entries.iter()) {
        writer.write_all(&seed.0).map_err(BrainError::Io)?;
        writer.write_all(&e.block_id.to_le_bytes()).map_err(BrainError::Io)?;
        writer.write_all(&e.offset.to_le_bytes()).map_err(BrainError::Io)?;
        writer.write_all(&e.length.to_le_bytes()).map_err(BrainError::Io)?;
    }

    // Merged bond index, same shape.
    for (key, e) in source.bond_index_entries() {
        if tombstoned_bonds.contains(&key) { continue; }
        if shadow_bond_keys.contains(&key) { continue; }
        writer.write_all(&key.0.0).map_err(BrainError::Io)?;
        writer.write_all(&key.1.0).map_err(BrainError::Io)?;
        writer.write_all(&e.block_id.to_le_bytes()).map_err(BrainError::Io)?;
        writer.write_all(&e.offset.to_le_bytes()).map_err(BrainError::Io)?;
        writer.write_all(&e.length.to_le_bytes()).map_err(BrainError::Io)?;
    }
    for ((from, to), e) in overlay_bond_keys_vec.iter().zip(overlay_bond_entries.iter()) {
        writer.write_all(&from.0).map_err(BrainError::Io)?;
        writer.write_all(&to.0).map_err(BrainError::Io)?;
        writer.write_all(&e.block_id.to_le_bytes()).map_err(BrainError::Io)?;
        writer.write_all(&e.offset.to_le_bytes()).map_err(BrainError::Io)?;
        writer.write_all(&e.length.to_le_bytes()).map_err(BrainError::Io)?;
    }

    // Block offsets — REAL count first; trailing reserved zeros are
    // invisible to the reader (it sizes the table from hdr.block_count).
    for off in &new_block_offsets {
        writer.write_all(&off.to_le_bytes()).map_err(BrainError::Io)?;
    }

    writer.seek(SeekFrom::Start(frames_end)).map_err(BrainError::Io)?;
    Ok(section_total)
}

/// OP-022.s.v4 — overlay-node iteration trait. Decoupled from
/// `AssociativeNetwork` so unit tests can construct ad-hoc overlay
/// fixtures without holding a full network.
pub trait OverlayNodeSource {
    fn count(&self) -> usize;
    fn for_each_seed(&self, f: &mut dyn FnMut(&Seed));
    fn for_each_node(
        &self,
        f: &mut dyn FnMut(&crate::node::KnowledgeNode) -> Result<(), BrainError>,
    ) -> Result<(), BrainError>;
}

/// OP-022.s.v4 — overlay-bond iteration trait.
pub trait OverlayBondSource {
    fn count(&self) -> usize;
    fn for_each_key(&self, f: &mut dyn FnMut(&(Seed, Seed)));
    fn for_each_bond(
        &self,
        f: &mut dyn FnMut(&crate::bond::TemporalBond) -> Result<(), BrainError>,
    ) -> Result<(), BrainError>;
}

/// OP-022.s.v4 — `AssociativeNetwork` adapter for `OverlayNodeSource`.
pub struct AssociativeNetworkOverlayNodes<'a>(pub &'a crate::network::AssociativeNetwork);
impl<'a> OverlayNodeSource for AssociativeNetworkOverlayNodes<'a> {
    fn count(&self) -> usize {
        self.0.overlay_node_count()
    }
    fn for_each_seed(&self, f: &mut dyn FnMut(&Seed)) {
        for s in self.0.overlay_node_seeds() { f(s); }
    }
    fn for_each_node(
        &self,
        f: &mut dyn FnMut(&crate::node::KnowledgeNode) -> Result<(), BrainError>,
    ) -> Result<(), BrainError> {
        self.0.for_each_overlay_node(|n| f(n))
    }
}

/// OP-022.s.v4 — `AssociativeNetwork` adapter for `OverlayBondSource`.
pub struct AssociativeNetworkOverlayBonds<'a>(pub &'a crate::network::AssociativeNetwork);
impl<'a> OverlayBondSource for AssociativeNetworkOverlayBonds<'a> {
    fn count(&self) -> usize {
        self.0.overlay_bond_count()
    }
    fn for_each_key(&self, f: &mut dyn FnMut(&(Seed, Seed))) {
        for k in self.0.overlay_bond_keys() { f(k); }
    }
    fn for_each_bond(
        &self,
        f: &mut dyn FnMut(&crate::bond::TemporalBond) -> Result<(), BrainError>,
    ) -> Result<(), BrainError> {
        self.0.for_each_overlay_bond(|b| f(b))
    }
}

#[inline]
fn pad_for(pos: usize, align: usize) -> usize {
    let rem = pos % align;
    if rem == 0 {
        0
    } else {
        align - rem
    }
}

// ---------------------------------------------------------------------
// Reader — header + index parsing (cheap, always done at load time)
// ---------------------------------------------------------------------

/// Header parsed out of the start of a compressed network payload.
#[derive(Debug, Clone, Copy)]
struct CompressedSectionHeader {
    node_count: u32,
    bond_count: u32,
    block_count: u32,
    #[allow(dead_code)]
    block_size_target: u32,
    #[allow(dead_code)]
    zstd_level: u32,
}

fn read_section_header(bytes: &[u8]) -> Result<CompressedSectionHeader, BrainError> {
    if bytes.len() < SECTION_HEADER_SIZE {
        return Err(BrainError::Format(format!(
            "compressed section too short for header: {} bytes",
            bytes.len()
        )));
    }
    Ok(CompressedSectionHeader {
        node_count: u32::from_le_bytes(bytes[0..4].try_into().unwrap()),
        bond_count: u32::from_le_bytes(bytes[4..8].try_into().unwrap()),
        block_count: u32::from_le_bytes(bytes[8..12].try_into().unwrap()),
        block_size_target: u32::from_le_bytes(bytes[12..16].try_into().unwrap()),
        zstd_level: u32::from_le_bytes(bytes[16..20].try_into().unwrap()),
    })
}

/// A parsed-but-not-yet-decompressed view over a compressed network
/// payload. Holds shared `Arc<Mmap>` ownership of the underlying file
/// mapping, the seed → entry HashMaps, the block-offset table, and a
/// per-block `OnceLock` arena for lazy decompression.
pub struct CompressedNetwork {
    /// File-level mmap kept alive so block frames remain readable.
    mmap: Arc<Mmap>,
    /// Absolute byte offset of the section payload within `mmap`.
    section_offset: usize,
    /// Length of the section payload within `mmap`.
    section_length: usize,
    /// (block_offset_within_section, block_count) — used by
    /// `decompress_block` to locate each frame.
    block_offsets: Vec<u64>,
    /// OP-016: bounded LRU of decompressed blocks. Replaces the
    /// monotonic `Vec<OnceLock<AlignedVec>>` arena. Cap is configured
    /// at construction (default `DEFAULT_BLOCK_CACHE_BYTES`).
    block_cache: Mutex<BlockCache>,
    /// Immutable block count (so `block_count()` doesn't lock the
    /// cache mutex).
    block_count_total: usize,
    /// OP-022.s.v3: persisted from the source section header.
    /// Used by the save-side block-passthrough eligibility check
    /// to confirm the source was packed with the same target as the
    /// requested save.
    block_size_target: u32,
    /// OP-022.s.v3: persisted from the source section header.
    /// Used by the save-side block-passthrough eligibility check.
    /// zstd levels matching means the compressed bytes are byte-
    /// compatible to copy verbatim.
    zstd_level: u32,
    /// seed → (block_id, offset_in_block, length) for nodes.
    node_index: HashMap<Seed, NodeIndexEntry>,
    /// (from, to) → (block_id, offset_in_block, length) for bonds.
    bond_index: HashMap<(Seed, Seed), BondIndexEntry>,
    /// Phase 4: optional master key. When `Some`, every block's
    /// compressed payload is XOR-encrypted on disk and must be
    /// XOR-decrypted before zstd decompression. When `None`, the
    /// reader treats the file as Phase 3 unencrypted-compressed.
    encryption_key: Option<MasterKey>,
}

// SAFETY: The arena is OnceLock<AlignedVec> entries; AlignedVec is
// Send+Sync. Arc<Mmap> is Send+Sync. The HashMaps are owned. No raw
// pointers escape across threads.
unsafe impl Send for CompressedNetwork {}
unsafe impl Sync for CompressedNetwork {}

impl CompressedNetwork {
    /// Parse the section payload (header + indexes + block table)
    /// from `mmap` at `section_offset .. section_offset+section_length`.
    /// Decompresses NO blocks — those load lazily on first access.
    /// Equivalent to `from_mmap_with_key(.., None)`.
    pub fn from_mmap(
        mmap: Arc<Mmap>,
        section_offset: usize,
        section_length: usize,
    ) -> Result<Self, BrainError> {
        Self::from_mmap_with_key_and_cap(
            mmap, section_offset, section_length, None, DEFAULT_BLOCK_CACHE_BYTES,
        )
    }

    /// Phase 4 entrypoint. When `encryption_key` is `Some`, every
    /// block's compressed payload is XOR-decrypted on access. The
    /// header, index, and block-offset table are unencrypted (so
    /// the loader can build the seed→entry maps without the key)
    /// and are validated identically to the unencrypted path.
    /// Uses `DEFAULT_BLOCK_CACHE_BYTES` as the LRU cap; use
    /// `from_mmap_with_key_and_cap` to override.
    pub fn from_mmap_with_key(
        mmap: Arc<Mmap>,
        section_offset: usize,
        section_length: usize,
        encryption_key: Option<MasterKey>,
    ) -> Result<Self, BrainError> {
        Self::from_mmap_with_key_and_cap(
            mmap, section_offset, section_length, encryption_key,
            DEFAULT_BLOCK_CACHE_BYTES,
        )
    }

    /// OP-016: full constructor. `cap_bytes` is the bounded-LRU cap
    /// for resident decompressed bytes. Pass
    /// `DEFAULT_BLOCK_CACHE_BYTES` for the default 4 GiB cap.
    pub fn from_mmap_with_key_and_cap(
        mmap: Arc<Mmap>,
        section_offset: usize,
        section_length: usize,
        encryption_key: Option<MasterKey>,
        cap_bytes: usize,
    ) -> Result<Self, BrainError> {
        let end = section_offset
            .checked_add(section_length)
            .ok_or_else(|| BrainError::Format("compressed section overflow".into()))?;
        if end > mmap.len() {
            return Err(BrainError::Format(format!(
                "compressed section runs past mmap (offset={} length={} mmap_len={})",
                section_offset,
                section_length,
                mmap.len()
            )));
        }

        let payload = &mmap[section_offset..end];
        let hdr = read_section_header(payload)?;

        let node_count = hdr.node_count as usize;
        let bond_count = hdr.bond_count as usize;
        let block_count = hdr.block_count as usize;

        let needed = SECTION_HEADER_SIZE
            + node_count * NODE_INDEX_ENTRY_SIZE
            + bond_count * BOND_INDEX_ENTRY_SIZE
            + block_count * BLOCK_OFFSET_ENTRY_SIZE;
        if payload.len() < needed {
            return Err(BrainError::Format(format!(
                "compressed section truncated: have {}, need at least {}",
                payload.len(),
                needed
            )));
        }

        // node_index
        let mut node_index: HashMap<Seed, NodeIndexEntry> =
            HashMap::with_capacity(node_count);
        let mut cur = SECTION_HEADER_SIZE;
        for _ in 0..node_count {
            let mut seed = [0u8; 32];
            seed.copy_from_slice(&payload[cur..cur + 32]);
            let block_id = u32::from_le_bytes(payload[cur + 32..cur + 36].try_into().unwrap());
            let offset = u32::from_le_bytes(payload[cur + 36..cur + 40].try_into().unwrap());
            let length = u32::from_le_bytes(payload[cur + 40..cur + 44].try_into().unwrap());
            node_index.insert(
                Seed(seed),
                NodeIndexEntry {
                    block_id,
                    offset,
                    length,
                },
            );
            cur += NODE_INDEX_ENTRY_SIZE;
        }

        // bond_index
        let mut bond_index: HashMap<(Seed, Seed), BondIndexEntry> =
            HashMap::with_capacity(bond_count);
        for _ in 0..bond_count {
            let mut from = [0u8; 32];
            from.copy_from_slice(&payload[cur..cur + 32]);
            let mut to = [0u8; 32];
            to.copy_from_slice(&payload[cur + 32..cur + 64]);
            let block_id = u32::from_le_bytes(payload[cur + 64..cur + 68].try_into().unwrap());
            let offset = u32::from_le_bytes(payload[cur + 68..cur + 72].try_into().unwrap());
            let length = u32::from_le_bytes(payload[cur + 72..cur + 76].try_into().unwrap());
            bond_index.insert(
                (Seed(from), Seed(to)),
                BondIndexEntry {
                    block_id,
                    offset,
                    length,
                },
            );
            cur += BOND_INDEX_ENTRY_SIZE;
        }

        // block_offsets — stored as offsets WITHIN the section payload
        let mut block_offsets: Vec<u64> = Vec::with_capacity(block_count);
        for _ in 0..block_count {
            let off = u64::from_le_bytes(payload[cur..cur + 8].try_into().unwrap());
            block_offsets.push(off);
            cur += BLOCK_OFFSET_ENTRY_SIZE;
        }

        // Validate every block offset before we hand back `Self` so
        // bad files fail fast instead of panicking on first lookup.
        for (i, off) in block_offsets.iter().enumerate() {
            let off_us = *off as usize;
            if off_us + BLOCK_FRAME_HEADER_SIZE > payload.len() {
                return Err(BrainError::Format(format!(
                    "compressed block {} offset {} out of range (payload {})",
                    i,
                    off_us,
                    payload.len()
                )));
            }
            let comp_len =
                u32::from_le_bytes(payload[off_us..off_us + 4].try_into().unwrap()) as usize;
            if off_us + BLOCK_FRAME_HEADER_SIZE + comp_len > payload.len() {
                return Err(BrainError::Format(format!(
                    "compressed block {} body runs past payload (off={} comp_len={})",
                    i, off_us, comp_len
                )));
            }
        }

        Ok(CompressedNetwork {
            mmap,
            section_offset,
            section_length,
            block_offsets,
            block_cache: Mutex::new(BlockCache::new(cap_bytes)),
            block_count_total: block_count,
            block_size_target: hdr.block_size_target,
            zstd_level: hdr.zstd_level,
            node_index,
            bond_index,
            encryption_key,
        })
    }

    /// Number of nodes carried in this compressed payload.
    pub fn node_count(&self) -> usize {
        self.node_index.len()
    }

    /// Number of bonds carried in this compressed payload.
    pub fn bond_count(&self) -> usize {
        self.bond_index.len()
    }

    /// Number of compressed blocks (for diagnostics / tests).
    pub fn block_count(&self) -> usize {
        self.block_count_total
    }

    /// Number of currently-decompressed blocks held in the LRU cache.
    /// Useful in RSS measurement tests so we can assert how many
    /// blocks a given workload faulted in.
    pub fn blocks_resident(&self) -> usize {
        self.block_cache.lock().unwrap().resident_count()
    }

    /// OP-016: bytes of decompressed payload currently resident in
    /// the LRU cache. Approximate (excludes Arc strong-ref overhead).
    pub fn block_cache_bytes_resident(&self) -> usize {
        self.block_cache.lock().unwrap().resident_bytes_count()
    }

    /// OP-016: configured LRU cap in bytes.
    pub fn block_cache_cap_bytes(&self) -> usize {
        self.block_cache.lock().unwrap().cap_bytes
    }

    /// Decompress block `block_id` if not already cached, and return
    /// an owning `Arc<AlignedVec>` to its bytes. The Arc keeps the
    /// buffer alive for as long as any caller holds it, regardless
    /// of LRU eviction state. (OP-016: previously returned `&[u8]`
    /// tied to `&self`; the LRU cannot make that contract because
    /// the bytes may be evicted from the map while a live reference
    /// exists.)
    pub fn decompress_block(&self, block_id: u32) -> Result<Arc<rkyv::AlignedVec>, BrainError> {
        if (block_id as usize) >= self.block_count_total {
            return Err(BrainError::Format(format!(
                "compressed block {} out of range ({} blocks)",
                block_id, self.block_count_total
            )));
        }
        // Fast path: cache hit. Lock only across the map probe;
        // release before the (potentially slow) decompression so
        // concurrent misses on OTHER blocks aren't serialized.
        {
            let mut cache = self.block_cache.lock().unwrap();
            if let Some(arc) = cache.get(block_id) {
                return Ok(arc);
            }
        }
        // Slow path: read frame, decode into AlignedVec, install.
        let off_us = self.block_offsets[block_id as usize] as usize;
        let abs = self.section_offset + off_us;
        let header = &self.mmap[abs..abs + BLOCK_FRAME_HEADER_SIZE];
        let comp_len = u32::from_le_bytes(header[0..4].try_into().unwrap()) as usize;
        let dec_len = u32::from_le_bytes(header[4..8].try_into().unwrap()) as usize;
        let body_start = abs + BLOCK_FRAME_HEADER_SIZE;
        let body_end = body_start + comp_len;
        if body_end > self.section_offset + self.section_length {
            return Err(BrainError::Format(format!(
                "compressed block {} body runs past section (end={} section_end={})",
                block_id,
                body_end,
                self.section_offset + self.section_length
            )));
        }
        let compressed_slice = &self.mmap[body_start..body_end];
        // Phase 4: when an encryption key is loaded, the compressed
        // bytes are XOR-encrypted under (key, block_index). Copy
        // them into an owned buffer and decrypt before feeding zstd.
        // Without a key we feed zstd directly from the mmap slice,
        // identical to the Phase 3 path.
        let owned_decrypted: Vec<u8>;
        let compressed: &[u8] = match &self.encryption_key {
            Some(k) => {
                let mut buf = compressed_slice.to_vec();
                xor_in_place(k, block_id as u64, &mut buf);
                owned_decrypted = buf;
                &owned_decrypted[..]
            }
            None => compressed_slice,
        };
        let mut buf = rkyv::AlignedVec::with_capacity(dec_len);
        let mut dst = vec![0u8; dec_len];
        let actual = zstd::bulk::decompress_to_buffer(compressed, &mut dst[..])
            .map_err(|e| BrainError::Format(format!("zstd decompress: {}", e)))?;
        if actual != dec_len {
            return Err(BrainError::Format(format!(
                "zstd decompressed {} bytes, expected {}",
                actual, dec_len
            )));
        }
        buf.extend_from_slice(&dst);
        let arc = Arc::new(buf);
        // Install: re-lock, check for race (another thread may have
        // installed while we were decompressing), insert if first.
        // The mutex is held across the map mutation only, not across
        // the decompression above.
        let mut cache = self.block_cache.lock().unwrap();
        if let Some(existing) = cache.get(block_id) {
            // Race lost: drop our buffer, return theirs.
            return Ok(existing);
        }
        Ok(cache.insert(block_id, arc))
    }

    /// Look up an archived `KnowledgeNode` by seed. Returns an
    /// `ArcArchive` handle that owns the underlying decompressed
    /// block via Arc — the bytes stay valid for the lifetime of
    /// the handle, even if the LRU evicts the entry from its map.
    /// (OP-016: previously returned `Option<&'a ArchivedKnowledgeNode>`;
    /// the LRU cannot offer that lifetime contract.)
    pub fn node_by_seed(&self, seed: &Seed) -> Option<ArcArchive<KnowledgeNode>> {
        let entry = *self.node_index.get(seed)?;
        let block = self.decompress_block(entry.block_id).ok()?;
        // Bounds check: writer guarantees offset+length fits within
        // the block, but defend against malformed payloads.
        let block_len = block.len();
        let off = entry.offset as usize;
        let len = entry.length as usize;
        if off.checked_add(len).map_or(true, |end| end > block_len) {
            return None;
        }
        Some(ArcArchive {
            block,
            offset: entry.offset,
            length: entry.length,
            _phantom: PhantomData,
        })
    }

    /// Look up an archived `TemporalBond` by (from, to). Same
    /// semantics as `node_by_seed` (OP-016 ArcArchive handle).
    pub fn bond_by_pair(&self, key: &(Seed, Seed)) -> Option<ArcArchive<TemporalBond>> {
        let entry = *self.bond_index.get(key)?;
        let block = self.decompress_block(entry.block_id).ok()?;
        let block_len = block.len();
        let off = entry.offset as usize;
        let len = entry.length as usize;
        if off.checked_add(len).map_or(true, |end| end > block_len) {
            return None;
        }
        Some(ArcArchive {
            block,
            offset: entry.offset,
            length: entry.length,
            _phantom: PhantomData,
        })
    }

    /// True if `seed` is present.
    pub fn has_node(&self, seed: &Seed) -> bool {
        self.node_index.contains_key(seed)
    }

    /// True if `(from, to)` is present.
    pub fn has_bond(&self, key: &(Seed, Seed)) -> bool {
        self.bond_index.contains_key(key)
    }

    /// Iterate over `(seed, ArcArchive<KnowledgeNode>)`. Each
    /// yielded handle owns an Arc clone of its block, so iteration
    /// is robust against LRU eviction churn. The iterator triggers
    /// a block decompression for each distinct block it touches.
    /// (OP-016: previously yielded `&'a ArchivedKnowledgeNode`.)
    pub fn iter_nodes<'a>(
        &'a self,
    ) -> impl Iterator<Item = (Seed, ArcArchive<KnowledgeNode>)> + 'a {
        self.node_index
            .iter()
            .filter_map(move |(seed, entry)| {
                let block = self.decompress_block(entry.block_id).ok()?;
                let block_len = block.len();
                let off = entry.offset as usize;
                let len = entry.length as usize;
                if off.checked_add(len).map_or(true, |e| e > block_len) {
                    return None;
                }
                Some((
                    *seed,
                    ArcArchive {
                        block,
                        offset: entry.offset,
                        length: entry.length,
                        _phantom: PhantomData,
                    },
                ))
            })
    }

    /// Iterate over `((from, to), ArcArchive<TemporalBond>)`. Same
    /// page-fault / cache-eviction semantics as `iter_nodes`.
    pub fn iter_bonds<'a>(
        &'a self,
    ) -> impl Iterator<Item = ((Seed, Seed), ArcArchive<TemporalBond>)> + 'a {
        self.bond_index
            .iter()
            .filter_map(move |(key, entry)| {
                let block = self.decompress_block(entry.block_id).ok()?;
                let block_len = block.len();
                let off = entry.offset as usize;
                let len = entry.length as usize;
                if off.checked_add(len).map_or(true, |e| e > block_len) {
                    return None;
                }
                Some((
                    *key,
                    ArcArchive {
                        block,
                        offset: entry.offset,
                        length: entry.length,
                        _phantom: PhantomData,
                    },
                ))
            })
    }

    /// Iterator over just the (from, to) bond keys — does NOT
    /// decompress any blocks. Used by `AssociativeNetwork::from_archive`
    /// to build the heap-resident adjacency lists at load time
    /// without touching compressed payload.
    pub fn iter_bond_keys<'a>(&'a self) -> impl Iterator<Item = (Seed, Seed)> + 'a {
        self.bond_index.keys().copied()
    }

    /// Iterator over just the node seeds — does NOT decompress
    /// any blocks. Mirrors `iter_bond_keys`.
    pub fn iter_node_seeds<'a>(&'a self) -> impl Iterator<Item = Seed> + 'a {
        self.node_index.keys().copied()
    }

    /// Underlying mmap (for callers that want to share the same map
    /// with an eager scar-log reader, mirroring NetworkArchive's
    /// `mmap()`).
    pub fn mmap(&self) -> &Arc<Mmap> {
        &self.mmap
    }

    // ----------------------------------------------------------------
    // OP-022.s.v3 — save-side block passthrough accessors
    // ----------------------------------------------------------------

    /// Block size target recorded in the source section header.
    pub fn block_size_target(&self) -> u32 { self.block_size_target }

    /// zstd compression level recorded in the source section header.
    pub fn zstd_level(&self) -> u32 { self.zstd_level }

    /// `true` iff blocks were XOR-encrypted under a master key.
    pub fn is_encrypted(&self) -> bool { self.encryption_key.is_some() }

    /// Borrow the encryption key the source was opened with, if any.
    /// Used by `can_passthrough_blocks` to confirm the save key matches.
    pub fn encryption_key(&self) -> Option<&MasterKey> {
        self.encryption_key.as_ref()
    }

    /// Section-relative offset of compressed block `block_id`, i.e. the
    /// byte position of its 8-byte frame header inside the source
    /// section. Used by passthrough writer to slice the mmap.
    pub fn block_section_offset(&self, block_id: u32) -> Option<u64> {
        self.block_offsets.get(block_id as usize).copied()
    }

    /// Total bytes of one compressed block frame (header + body) in
    /// the source mmap. Reads only the 8-byte frame header so it never
    /// decompresses.
    pub fn block_frame_len(&self, block_id: u32) -> Result<usize, BrainError> {
        if (block_id as usize) >= self.block_count_total {
            return Err(BrainError::Format(format!(
                "compressed block {} out of range ({} blocks)",
                block_id, self.block_count_total
            )));
        }
        let off_us = self.block_offsets[block_id as usize] as usize;
        let abs = self.section_offset + off_us;
        let header = &self.mmap[abs..abs + BLOCK_FRAME_HEADER_SIZE];
        let comp_len = u32::from_le_bytes(header[0..4].try_into().unwrap()) as usize;
        Ok(BLOCK_FRAME_HEADER_SIZE + comp_len)
    }

    /// Borrow a compressed block frame (header + body) from the source
    /// mmap as a byte slice — zero copy, no decompress, no decrypt.
    /// Used by passthrough writer.
    pub fn block_frame_bytes(&self, block_id: u32) -> Result<&[u8], BrainError> {
        if (block_id as usize) >= self.block_count_total {
            return Err(BrainError::Format(format!(
                "compressed block {} out of range ({} blocks)",
                block_id, self.block_count_total
            )));
        }
        let off_us = self.block_offsets[block_id as usize] as usize;
        let abs = self.section_offset + off_us;
        let header = &self.mmap[abs..abs + BLOCK_FRAME_HEADER_SIZE];
        let comp_len = u32::from_le_bytes(header[0..4].try_into().unwrap()) as usize;
        let frame_end = abs + BLOCK_FRAME_HEADER_SIZE + comp_len;
        if frame_end > self.section_offset + self.section_length {
            return Err(BrainError::Format(format!(
                "block {} frame runs past section", block_id
            )));
        }
        Ok(&self.mmap[abs..frame_end])
    }

    /// Iterate `(seed, NodeIndexEntry)` for every node, in the
    /// HashMap-internal iteration order. Used by the passthrough
    /// writer to emit the node_index region. Order doesn't affect
    /// correctness (entries are keyed by seed on read) but matches
    /// `iter_nodes()` so a passthrough-saved file is byte-identical
    /// to a slow-path save against the same loaded source.
    pub fn node_index_entries(&self) -> impl Iterator<Item = (Seed, NodeIndexEntry)> + '_ {
        self.node_index.iter().map(|(s, e)| (*s, *e))
    }

    /// Iterate `((from, to), BondIndexEntry)` for every bond. See
    /// `node_index_entries` for the order-of-iteration rationale.
    pub fn bond_index_entries(&self) -> impl Iterator<Item = ((Seed, Seed), BondIndexEntry)> + '_ {
        self.bond_index.iter().map(|(k, e)| (*k, *e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bond::{BondType, KeyframeSource, TemporalBond, TemporalKeyframe};
    use crate::node::{Confidence, KnowledgeContent, KnowledgeNode, KnowledgeUnit};
    use crate::seed::Seed;

    fn dummy_node(label: &str) -> KnowledgeNode {
        let mut emb = vec![0.0_f32; crate::node::EMBEDDING_DIM];
        for (i, s) in emb.iter_mut().enumerate() {
            *s = ((i as f32) * 0.001 + label.len() as f32 * 0.01).sin();
        }
        KnowledgeNode {
            seed: Seed::from_label(label),
            content: label.to_string(),
            knowledge: KnowledgeUnit {
                content: KnowledgeContent::Concept(format!("concept of {}", label)),
                confidence: Confidence::Verified,
                created_at: 1_700_000_000_000,
                updated_at: 1_700_000_000_000,
            },
            embedding: emb,
            parent_seeds: Vec::new(),
            typed_ratio: 1.0,
        }
    }

    fn dummy_bond(from: &str, to: &str, bt: BondType, w: f32) -> TemporalBond {
        TemporalBond {
            from: Seed::from_label(from),
            to: Seed::from_label(to),
            bond_type: bt,
            keyframes: vec![TemporalKeyframe {
                timestamp: 1_700_000_000_000,
                weight: w,
                source: KeyframeSource::Baseline,
            }],
            half_life_ms: (30u64 * 24 * 3600 * 1000) as f64,
        }
    }

    fn dummy_snapshot() -> NetworkSnapshot {
        let nodes = vec![
            dummy_node("cell"),
            dummy_node("mitosis"),
            dummy_node("membrane"),
            dummy_node("unit_of_life"),
            dummy_node("cell_division"),
        ];
        let bonds = vec![
            dummy_bond("cell", "membrane", BondType::HasProperty, 0.8),
            dummy_bond("cell", "unit_of_life", BondType::IsA, 0.9),
            dummy_bond("mitosis", "cell_division", BondType::IsA, 0.95),
            dummy_bond("cell", "mitosis", BondType::Produces, 0.7),
        ];
        NetworkSnapshot { nodes, bonds }
    }

    fn payload_to_compressed_network(payload: Vec<u8>) -> CompressedNetwork {
        // Use a temp file mmap so the lifetime model matches production.
        let mut p = std::env::temp_dir();
        p.push(format!(
            "lu4d-compress-test-{}-{}.bin",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::write(&p, &payload).unwrap();
        let f = std::fs::File::open(&p).unwrap();
        let mmap = unsafe { Mmap::map(&f) }.unwrap();
        let cn =
            CompressedNetwork::from_mmap(Arc::new(mmap), 0, payload.len()).expect("from_mmap");
        let _ = std::fs::remove_file(&p);
        cn
    }

    #[test]
    fn round_trip_small_snapshot_recovers_every_node_and_bond() {
        let snap = dummy_snapshot();
        let payload =
            write_compressed_network_payload(&snap, 32 * 1024, DEFAULT_ZSTD_LEVEL, None).unwrap();
        let cn = payload_to_compressed_network(payload);

        assert_eq!(cn.node_count(), snap.nodes.len());
        assert_eq!(cn.bond_count(), snap.bonds.len());
        assert!(cn.block_count() >= 1);
        assert_eq!(cn.blocks_resident(), 0); // lazy

        for n in &snap.nodes {
            let h = cn
                .node_by_seed(&n.seed)
                .unwrap_or_else(|| panic!("missing node: {}", n.content));
            let archived = h.archived();
            assert_eq!(archived.content.as_str(), n.content);
            assert_eq!(archived.embedding.len(), n.embedding.len());
        }
        for b in &snap.bonds {
            let h = cn
                .bond_by_pair(&(b.from, b.to))
                .expect("bond present");
            assert_eq!(h.archived().keyframes.len(), b.keyframes.len());
        }
    }

    #[test]
    fn lazy_decompression_only_touches_blocks_actually_accessed() {
        // 200 nodes packed across multiple blocks. Touch only the
        // first node and verify only that one block (plus the bond
        // blocks if accessed via iter_bonds) is decompressed.
        let mut nodes: Vec<KnowledgeNode> = (0..200).map(|i| dummy_node(&format!("n{}", i))).collect();
        // tiny block size forces several blocks
        let snap = NetworkSnapshot {
            nodes: std::mem::take(&mut nodes),
            bonds: Vec::new(),
        };
        let payload = write_compressed_network_payload(&snap, 16 * 1024, 3, None).unwrap();
        let cn = payload_to_compressed_network(payload);
        assert!(cn.block_count() >= 2);
        assert_eq!(cn.blocks_resident(), 0);
        let _ = cn
            .node_by_seed(&Seed::from_label("n0"))
            .expect("first node");
        assert_eq!(cn.blocks_resident(), 1);
        // Touch one in a different block.
        let _ = cn
            .node_by_seed(&Seed::from_label(&format!("n{}", 199)))
            .expect("last node");
        assert!(cn.blocks_resident() >= 2);
    }

    #[test]
    fn write_then_parse_header_roundtrips_metadata() {
        let snap = dummy_snapshot();
        let payload = write_compressed_network_payload(&snap, 64 * 1024, 5, None).unwrap();
        let hdr = read_section_header(&payload).unwrap();
        assert_eq!(hdr.node_count as usize, snap.nodes.len());
        assert_eq!(hdr.bond_count as usize, snap.bonds.len());
        assert_eq!(hdr.zstd_level, 5);
        assert_eq!(hdr.block_size_target as usize, 64 * 1024);
    }

    #[test]
    fn missing_seed_returns_none() {
        let snap = dummy_snapshot();
        let payload = write_compressed_network_payload(&snap, 32 * 1024, 3, None).unwrap();
        let cn = payload_to_compressed_network(payload);
        assert!(cn.node_by_seed(&Seed::from_label("nope")).is_none());
        assert!(cn
            .bond_by_pair(&(Seed::from_label("a"), Seed::from_label("b")))
            .is_none());
    }
    #[test]
    fn encrypted_round_trip_recovers_every_node_and_bond() {
        use crate::encrypt::MasterKey;
        let snap = dummy_snapshot();
        let key = MasterKey::from_bytes([0xa5; crate::encrypt::KEY_SIZE]);
        let payload =
            write_compressed_network_payload(&snap, 32 * 1024, DEFAULT_ZSTD_LEVEL, Some(&key))
                .unwrap();
        // Use a temp file mmap so the lifetime model matches production.
        let mut p = std::env::temp_dir();
        p.push(format!(
            "lu4d-encrypt-test-{}-{}.bin",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::write(&p, &payload).unwrap();
        let f = std::fs::File::open(&p).unwrap();
        let mmap = unsafe { Mmap::map(&f) }.unwrap();
        let cn = CompressedNetwork::from_mmap_with_key(
            Arc::new(mmap), 0, payload.len(), Some(key)).expect("from_mmap_with_key");
        for n in &snap.nodes {
            let h = cn
                .node_by_seed(&n.seed)
                .unwrap_or_else(|| panic!("missing encrypted node: {}", n.content));
            assert_eq!(h.archived().content.as_str(), n.content);
        }
        for b in &snap.bonds {
            let h = cn.bond_by_pair(&(b.from, b.to)).expect("encrypted bond");
            assert_eq!(h.archived().keyframes.len(), b.keyframes.len());
        }
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn encrypted_payload_bytes_differ_from_unencrypted() {
        use crate::encrypt::MasterKey;
        let snap = dummy_snapshot();
        let key = MasterKey::from_bytes([0x42; crate::encrypt::KEY_SIZE]);
        let unencrypted =
            write_compressed_network_payload(&snap, 32 * 1024, 3, None).unwrap();
        let encrypted =
            write_compressed_network_payload(&snap, 32 * 1024, 3, Some(&key)).unwrap();
        // Layouts must match length-for-length (encryption is in-place
        // XOR over compressed bytes only — no extra header).
        assert_eq!(unencrypted.len(), encrypted.len());
        // The header/index region (everything before the first block
        // frame) must be byte-identical; the block-frame headers stay
        // cleartext, only the compressed bytes differ.
        // We don't know the exact frame-data offsets without parsing,
        // so just assert: at least *some* bytes differ.
        assert_ne!(unencrypted, encrypted, "encryption must change ciphertext");
    }

    #[test]
    fn encrypted_with_wrong_key_fails_to_decompress_or_decode() {
        use crate::encrypt::MasterKey;
        let snap = dummy_snapshot();
        let real_key = MasterKey::from_bytes([0x77; crate::encrypt::KEY_SIZE]);
        let wrong_key = MasterKey::from_bytes([0x99; crate::encrypt::KEY_SIZE]);
        let payload =
            write_compressed_network_payload(&snap, 32 * 1024, 3, Some(&real_key)).unwrap();
        let mut p = std::env::temp_dir();
        p.push(format!(
            "lu4d-encrypt-wrongkey-{}-{}.bin",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::write(&p, &payload).unwrap();
        let f = std::fs::File::open(&p).unwrap();
        let mmap = unsafe { Mmap::map(&f) }.unwrap();
        let cn = CompressedNetwork::from_mmap_with_key(
            Arc::new(mmap), 0, payload.len(), Some(wrong_key)).expect("index parses");
        // Looking up any node should either fail (zstd refuses to
        // decompress the garbled bytes) or return None (we treat zstd
        // failures as missing). Either way, no node bytes match.
        let attempts: Vec<_> = snap.nodes.iter()
            .map(|n| cn.node_by_seed(&n.seed))
            .collect();
        for r in attempts {
            assert!(r.is_none(), "wrong key must not yield a node");
        }
        let _ = std::fs::remove_file(&p);
    }


    // ======================================================
    // OP-016: bounded LRU eviction tests
    // ======================================================

    fn payload_to_compressed_network_with_cap(payload: Vec<u8>, cap: usize) -> CompressedNetwork {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "lu4d-op016-{}-{}.bin",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::write(&p, &payload).unwrap();
        let f = std::fs::File::open(&p).unwrap();
        let mmap = unsafe { Mmap::map(&f) }.unwrap();
        let cn = CompressedNetwork::from_mmap_with_key_and_cap(
            Arc::new(mmap), 0, payload.len(), None, cap,
        ).expect("from_mmap_with_key_and_cap");
        let _ = std::fs::remove_file(&p);
        cn
    }

    #[test]
    fn lru_eviction_keeps_resident_bytes_under_cap() {
        // 1000 nodes packed into many small blocks. Cap is 128 KiB,
        // well below the total decompressed size. After touching
        // every node, resident bytes should stay near or under cap
        // (modulo a single oversized block which the cache won't
        // evict below 1 entry).
        let nodes: Vec<KnowledgeNode> = (0..1000)
            .map(|i| dummy_node(&format!("evict_n{}", i)))
            .collect();
        let snap = NetworkSnapshot { nodes, bonds: Vec::new() };
        let payload = write_compressed_network_payload(&snap, 16 * 1024, 3, None).unwrap();
        let cap = 128 * 1024;
        let cn = payload_to_compressed_network_with_cap(payload, cap);
        assert!(cn.block_count() >= 4, "need multiple blocks; got {}", cn.block_count());

        // Touch every node — many block decompressions.
        for i in 0..1000 {
            let _h = cn.node_by_seed(&Seed::from_label(&format!("evict_n{}", i)));
        }

        let resident_bytes = cn.block_cache_bytes_resident();
        let resident_count = cn.blocks_resident();
        let total = cn.block_count();

        // Eviction MUST have occurred: resident < total.
        assert!(
            resident_count < total,
            "OP-016: eviction did not occur ({} resident of {} total, bytes={})",
            resident_count, total, resident_bytes,
        );
        // Resident bytes should be at or under cap, EXCEPT when a
        // single residual block is itself larger than cap (the cache
        // won't evict below 1 entry).
        assert!(
            resident_bytes <= cap || resident_count == 1,
            "OP-016: resident_bytes={} > cap={} with {} entries",
            resident_bytes, cap, resident_count,
        );
    }

    #[test]
    fn lru_re_decompress_after_eviction_returns_identical_bytes() {
        // After cache eviction, looking up the same node again must
        // produce byte-identical archived content (re-decompression
        // is deterministic).
        let nodes: Vec<KnowledgeNode> = (0..200)
            .map(|i| dummy_node(&format!("redec_n{}", i)))
            .collect();
        let snap = NetworkSnapshot { nodes, bonds: Vec::new() };
        let payload = write_compressed_network_payload(&snap, 16 * 1024, 3, None).unwrap();
        // Tight cap to force aggressive eviction.
        let cn = payload_to_compressed_network_with_cap(payload, 32 * 1024);

        // First touch of node 0.
        let h1 = cn.node_by_seed(&Seed::from_label("redec_n0")).expect("first lookup");
        let content_1 = h1.archived().content.as_str().to_string();
        let emb_1: Vec<f32> = h1.archived().embedding.iter().map(|x| f32::from(*x)).collect();
        drop(h1);

        // Touch enough other nodes to evict block 0.
        for i in 50..200 {
            let _ = cn.node_by_seed(&Seed::from_label(&format!("redec_n{}", i)));
        }

        // Re-touch node 0; this MUST re-decompress.
        let h2 = cn.node_by_seed(&Seed::from_label("redec_n0")).expect("re-lookup");
        let content_2 = h2.archived().content.as_str().to_string();
        let emb_2: Vec<f32> = h2.archived().embedding.iter().map(|x| f32::from(*x)).collect();

        assert_eq!(content_1, "redec_n0");
        assert_eq!(content_1, content_2);
        assert_eq!(emb_1, emb_2);
    }

    #[test]
    fn concurrent_miss_race_yields_identical_bytes() {
        // Multiple threads concurrently hit the same block_id on a
        // cold cache. The mutex-serialized install path must yield
        // identical Arc bytes to every thread.
        use std::thread;
        let nodes: Vec<KnowledgeNode> = (0..50)
            .map(|i| dummy_node(&format!("race_n{}", i)))
            .collect();
        let snap = NetworkSnapshot { nodes, bonds: Vec::new() };
        let payload = write_compressed_network_payload(&snap, 32 * 1024, 3, None).unwrap();
        let mut p = std::env::temp_dir();
        p.push(format!(
            "lu4d-op016-race-{}-{}.bin",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::write(&p, &payload).unwrap();
        let f = std::fs::File::open(&p).unwrap();
        let mmap = unsafe { Mmap::map(&f) }.unwrap();
        let cn = Arc::new(
            CompressedNetwork::from_mmap(Arc::new(mmap), 0, payload.len())
                .expect("from_mmap"),
        );
        let _ = std::fs::remove_file(&p);

        let mut handles = Vec::new();
        for _ in 0..8 {
            let cn_clone = cn.clone();
            handles.push(thread::spawn(move || {
                let h = cn_clone
                    .node_by_seed(&Seed::from_label("race_n0"))
                    .expect("race lookup");
                h.archived().content.as_str().to_string()
            }));
        }
        let contents: Vec<String> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        assert_eq!(contents.len(), 8);
        for c in &contents {
            assert_eq!(c, "race_n0");
        }
    }

    // ----------------------------------------------------------------
    // OP-022.s.v3 — block-passthrough save tests
    // ----------------------------------------------------------------

    use crate::encrypt::MasterKey;

    fn build_encrypted_payload(
        snap: &NetworkSnapshot,
        block_size: usize,
        zstd_level: i32,
        key: &MasterKey,
    ) -> Vec<u8> {
        write_compressed_network_payload(snap, block_size, zstd_level, Some(key)).unwrap()
    }

    fn cn_from_payload_with_key(payload: Vec<u8>, key: Option<MasterKey>) -> CompressedNetwork {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "lu4d-passthru-test-{}-{}.bin",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::write(&p, &payload).unwrap();
        let f = std::fs::File::open(&p).unwrap();
        let mmap = unsafe { Mmap::map(&f) }.unwrap();
        let cn = CompressedNetwork::from_mmap_with_key(
            Arc::new(mmap),
            0,
            payload.len(),
            key,
        )
        .expect("from_mmap_with_key");
        let _ = std::fs::remove_file(&p);
        cn
    }

    #[test]
    fn test_passthrough_eligibility_all_params_match() {
        let snap = dummy_snapshot();
        let key = MasterKey::from_bytes([7u8; 32]);
        let payload = build_encrypted_payload(&snap, 32 * 1024, 3, &key);
        let cn = cn_from_payload_with_key(payload, Some(key.clone()));
        let save = CompressedSaveParams {
            block_size_target: 32 * 1024,
            zstd_level: 3,
            encryption_key: Some(&key),
        };
        assert!(can_passthrough_compressed_blocks(&cn, &save));
    }

    #[test]
    fn test_passthrough_eligibility_different_block_size() {
        let snap = dummy_snapshot();
        let key = MasterKey::from_bytes([7u8; 32]);
        let payload = build_encrypted_payload(&snap, 32 * 1024, 3, &key);
        let cn = cn_from_payload_with_key(payload, Some(key.clone()));
        let save = CompressedSaveParams {
            block_size_target: 64 * 1024,
            zstd_level: 3,
            encryption_key: Some(&key),
        };
        assert!(!can_passthrough_compressed_blocks(&cn, &save));
    }

    #[test]
    fn test_passthrough_eligibility_different_zstd_level() {
        let snap = dummy_snapshot();
        let key = MasterKey::from_bytes([7u8; 32]);
        let payload = build_encrypted_payload(&snap, 32 * 1024, 3, &key);
        let cn = cn_from_payload_with_key(payload, Some(key.clone()));
        let save = CompressedSaveParams {
            block_size_target: 32 * 1024,
            zstd_level: 5,
            encryption_key: Some(&key),
        };
        assert!(!can_passthrough_compressed_blocks(&cn, &save));
    }

    #[test]
    fn test_passthrough_eligibility_different_encryption_key() {
        let snap = dummy_snapshot();
        let key_a = MasterKey::from_bytes([7u8; 32]);
        let key_b = MasterKey::from_bytes([8u8; 32]);
        let payload = build_encrypted_payload(&snap, 32 * 1024, 3, &key_a);
        let cn = cn_from_payload_with_key(payload, Some(key_a.clone()));
        let save = CompressedSaveParams {
            block_size_target: 32 * 1024,
            zstd_level: 3,
            encryption_key: Some(&key_b),
        };
        assert!(!can_passthrough_compressed_blocks(&cn, &save));
    }

    #[test]
    fn test_passthrough_eligibility_unencrypted_source_encrypted_save() {
        let snap = dummy_snapshot();
        let payload = write_compressed_network_payload(&snap, 32 * 1024, 3, None).unwrap();
        let cn = cn_from_payload_with_key(payload, None);
        let key = MasterKey::from_bytes([7u8; 32]);
        let save = CompressedSaveParams {
            block_size_target: 32 * 1024,
            zstd_level: 3,
            encryption_key: Some(&key),
        };
        assert!(!can_passthrough_compressed_blocks(&cn, &save));
    }

    #[test]
    fn test_passthrough_byte_identical_blocks_and_roundtrip() {
        // Load source, save via passthrough. Critical correctness
        // properties (the load-bearing test):
        //  (a) each output block frame is byte-identical to the source
        //      frame at the SAME logical block id — keystream is keyed
        //      on (master_key, block_id), so reordering corrupts.
        //  (b) reloading the passthrough output via `from_mmap_with_key`
        //      recovers every node and bond byte-identical to the source.
        let snap = dummy_snapshot();
        let key = MasterKey::from_bytes([42u8; 32]);
        let payload_a = build_encrypted_payload(&snap, 32 * 1024, 3, &key);
        let cn_a = cn_from_payload_with_key(payload_a, Some(key.clone()));

        let save = CompressedSaveParams {
            block_size_target: 32 * 1024,
            zstd_level: 3,
            encryption_key: Some(&key),
        };

        let mut cur = std::io::Cursor::new(Vec::<u8>::new());
        let n = write_compressed_network_payload_passthrough_to(&cn_a, &save, &mut cur)
            .expect("passthrough write");
        let out = cur.into_inner();
        assert_eq!(n as usize, out.len());

        let cn_b = cn_from_payload_with_key(out, Some(key.clone()));
        assert_eq!(cn_b.node_count(), snap.nodes.len());
        assert_eq!(cn_b.bond_count(), snap.bonds.len());
        assert_eq!(cn_b.block_count(), cn_a.block_count());

        // (a) block frames bytewise identical at same block_id.
        for bid in 0..(cn_a.block_count() as u32) {
            let frame_a = cn_a.block_frame_bytes(bid).unwrap();
            let frame_b = cn_b.block_frame_bytes(bid).unwrap();
            assert_eq!(frame_a, frame_b, "block frame mismatch at id {}", bid);
        }

        // (b) every node and bond recoverable, content byte-identical.
        for n in &snap.nodes {
            let h = cn_b.node_by_seed(&n.seed).expect("node missing");
            assert_eq!(h.archived().content.as_str(), n.content);
        }
        for b in &snap.bonds {
            let _h = cn_b.bond_by_pair(&(b.from, b.to)).expect("bond missing");
        }
    }

    #[test]
    fn test_passthrough_refused_at_writer_when_params_mismatch() {
        // Writer must refuse loudly on a param mismatch — defensive
        // guard against caller-side gating bugs.
        let snap = dummy_snapshot();
        let key = MasterKey::from_bytes([1u8; 32]);
        let payload = build_encrypted_payload(&snap, 32 * 1024, 3, &key);
        let cn = cn_from_payload_with_key(payload, Some(key.clone()));
        let wrong_save = CompressedSaveParams {
            block_size_target: 64 * 1024, // mismatched
            zstd_level: 3,
            encryption_key: Some(&key),
        };
        let mut cur = std::io::Cursor::new(Vec::<u8>::new());
        let r = write_compressed_network_payload_passthrough_to(&cn, &wrong_save, &mut cur);
        assert!(r.is_err());
    }

    // -----------------------------------------------------------------
    // OP-022.s.v4 — partial-passthrough writer tests
    // -----------------------------------------------------------------

    use std::collections::HashSet;

    /// Minimal `OverlayNodeSource` implementation backed by a Vec, for
    /// unit tests that don't want to construct a full AssociativeNetwork.
    struct TestOverlayNodes {
        nodes: Vec<KnowledgeNode>,
    }
    impl OverlayNodeSource for TestOverlayNodes {
        fn count(&self) -> usize { self.nodes.len() }
        fn for_each_seed(&self, f: &mut dyn FnMut(&Seed)) {
            for n in &self.nodes { f(&n.seed); }
        }
        fn for_each_node(
            &self,
            f: &mut dyn FnMut(&KnowledgeNode) -> Result<(), BrainError>,
        ) -> Result<(), BrainError> {
            for n in &self.nodes { f(n)?; }
            Ok(())
        }
    }

    struct TestOverlayBonds { bonds: Vec<TemporalBond> }
    impl OverlayBondSource for TestOverlayBonds {
        fn count(&self) -> usize { self.bonds.len() }
        fn for_each_key(&self, f: &mut dyn FnMut(&(Seed, Seed))) {
            for b in &self.bonds { f(&(b.from, b.to)); }
        }
        fn for_each_bond(
            &self,
            f: &mut dyn FnMut(&TemporalBond) -> Result<(), BrainError>,
        ) -> Result<(), BrainError> {
            for b in &self.bonds { f(b)?; }
            Ok(())
        }
    }

    fn empty_overlay_nodes() -> TestOverlayNodes { TestOverlayNodes { nodes: vec![] } }
    fn empty_overlay_bonds() -> TestOverlayBonds { TestOverlayBonds { bonds: vec![] } }
    fn empty_tombstones_nodes() -> HashSet<Seed> { HashSet::new() }
    fn empty_tombstones_bonds() -> HashSet<(Seed, Seed)> { HashSet::new() }

    #[test]
    fn test_partial_passthrough_empty_overlay() {
        // No overlay, no tombstones — every source block passes through,
        // every source seed remains addressable, block ids preserved.
        let snap = dummy_snapshot();
        let key = MasterKey::from_bytes([42u8; 32]);
        let payload_a = build_encrypted_payload(&snap, 32 * 1024, 3, &key);
        let cn_a = cn_from_payload_with_key(payload_a, Some(key.clone()));
        let save = CompressedSaveParams {
            block_size_target: 32 * 1024,
            zstd_level: 3,
            encryption_key: Some(&key),
        };
        let on = empty_overlay_nodes();
        let ob = empty_overlay_bonds();
        let tn = empty_tombstones_nodes();
        let tb = empty_tombstones_bonds();
        let mut cur = std::io::Cursor::new(Vec::<u8>::new());
        write_compressed_network_payload_partial_passthrough_to(
            &cn_a, &save, &tn, &tb, &on, &ob, &mut cur,
        ).expect("partial passthrough empty overlay");
        let out = cur.into_inner();
        let cn_b = cn_from_payload_with_key(out, Some(key.clone()));
        assert_eq!(cn_b.node_count(), snap.nodes.len());
        assert_eq!(cn_b.bond_count(), snap.bonds.len());
        assert_eq!(cn_b.block_count(), cn_a.block_count(),
            "empty overlay must not add any blocks");
        for n in &snap.nodes {
            assert!(cn_b.node_by_seed(&n.seed).is_some(), "missing {}", n.content);
        }
        for b in &snap.bonds {
            assert!(cn_b.bond_by_pair(&(b.from, b.to)).is_some());
        }
    }

    #[test]
    fn test_partial_passthrough_added_nodes_only() {
        let snap = dummy_snapshot();
        let key = MasterKey::from_bytes([7u8; 32]);
        let payload_a = build_encrypted_payload(&snap, 32 * 1024, 3, &key);
        let cn_a = cn_from_payload_with_key(payload_a, Some(key.clone()));
        let new_a = dummy_node("new_node_a");
        let new_b = dummy_node("new_node_b");
        let on = TestOverlayNodes { nodes: vec![new_a.clone(), new_b.clone()] };
        let ob = empty_overlay_bonds();
        let tn = empty_tombstones_nodes();
        let tb = empty_tombstones_bonds();
        let save = CompressedSaveParams {
            block_size_target: 32 * 1024,
            zstd_level: 3,
            encryption_key: Some(&key),
        };
        let mut cur = std::io::Cursor::new(Vec::<u8>::new());
        write_compressed_network_payload_partial_passthrough_to(
            &cn_a, &save, &tn, &tb, &on, &ob, &mut cur,
        ).expect("partial passthrough add nodes");
        let out = cur.into_inner();
        let cn_b = cn_from_payload_with_key(out, Some(key.clone()));
        assert_eq!(cn_b.node_count(), snap.nodes.len() + 2);
        for n in &snap.nodes {
            assert!(cn_b.node_by_seed(&n.seed).is_some(), "src node {} missing", n.content);
        }
        assert!(cn_b.node_by_seed(&new_a.seed).is_some(), "new_a missing");
        assert!(cn_b.node_by_seed(&new_b.seed).is_some(), "new_b missing");
        assert!(cn_b.block_count() > cn_a.block_count(), "overlay block not appended");
    }

    #[test]
    fn test_partial_passthrough_added_bonds_only() {
        let snap = dummy_snapshot();
        let key = MasterKey::from_bytes([11u8; 32]);
        let payload_a = build_encrypted_payload(&snap, 32 * 1024, 3, &key);
        let cn_a = cn_from_payload_with_key(payload_a, Some(key.clone()));
        let extra = dummy_bond("mitosis", "membrane", BondType::HasProperty, 0.5);
        let on = empty_overlay_nodes();
        let ob = TestOverlayBonds { bonds: vec![extra.clone()] };
        let tn = empty_tombstones_nodes();
        let tb = empty_tombstones_bonds();
        let save = CompressedSaveParams {
            block_size_target: 32 * 1024,
            zstd_level: 3,
            encryption_key: Some(&key),
        };
        let mut cur = std::io::Cursor::new(Vec::<u8>::new());
        write_compressed_network_payload_partial_passthrough_to(
            &cn_a, &save, &tn, &tb, &on, &ob, &mut cur,
        ).expect("partial passthrough add bond");
        let out = cur.into_inner();
        let cn_b = cn_from_payload_with_key(out, Some(key.clone()));
        assert_eq!(cn_b.bond_count(), snap.bonds.len() + 1);
        assert!(cn_b.bond_by_pair(&(extra.from, extra.to)).is_some());
        for b in &snap.bonds {
            assert!(cn_b.bond_by_pair(&(b.from, b.to)).is_some(), "src bond missing");
        }
    }

    #[test]
    fn test_partial_passthrough_with_tombstones() {
        // Deleted node bytes still live in passthrough'd block frames;
        // merged node_index excludes them; reader returns None.
        let snap = dummy_snapshot();
        let key = MasterKey::from_bytes([19u8; 32]);
        let payload_a = build_encrypted_payload(&snap, 32 * 1024, 3, &key);
        let cn_a = cn_from_payload_with_key(payload_a, Some(key.clone()));
        let on = empty_overlay_nodes();
        let ob = empty_overlay_bonds();
        let dead_seed = Seed::from_label("cell");
        let mut tn: HashSet<Seed> = HashSet::new();
        tn.insert(dead_seed);
        // Tombstone the bonds that involve `cell` so bond_count is consistent
        let mut tb: HashSet<(Seed, Seed)> = HashSet::new();
        for b in &snap.bonds {
            if b.from == dead_seed || b.to == dead_seed {
                tb.insert((b.from, b.to));
            }
        }
        let save = CompressedSaveParams {
            block_size_target: 32 * 1024,
            zstd_level: 3,
            encryption_key: Some(&key),
        };
        let mut cur = std::io::Cursor::new(Vec::<u8>::new());
        write_compressed_network_payload_partial_passthrough_to(
            &cn_a, &save, &tn, &tb, &on, &ob, &mut cur,
        ).expect("partial passthrough tombstones");
        let out = cur.into_inner();
        let cn_b = cn_from_payload_with_key(out, Some(key.clone()));
        // Tombstoned node is unreachable from reader.
        assert!(cn_b.node_by_seed(&dead_seed).is_none(),
            "tombstoned node still visible to reader");
        // Tombstoned bonds also unreachable.
        for (f, t) in tb.iter() {
            assert!(cn_b.bond_by_pair(&(*f, *t)).is_none(),
                "tombstoned bond still visible");
        }
        // Surviving nodes still present.
        for n in &snap.nodes {
            if n.seed != dead_seed {
                assert!(cn_b.node_by_seed(&n.seed).is_some(),
                    "non-tombstoned node {} missing", n.content);
            }
        }
    }

    #[test]
    fn test_partial_passthrough_with_shadowing() {
        // Overlay updates an existing seed (same seed, different content).
        // node_index must point at the overlay block (NOT the archive
        // block); reader returns the overlay version. This is the
        // v4-specific correctness invariant.
        let snap = dummy_snapshot();
        let key = MasterKey::from_bytes([23u8; 32]);
        let payload_a = build_encrypted_payload(&snap, 32 * 1024, 3, &key);
        let cn_a = cn_from_payload_with_key(payload_a, Some(key.clone()));
        // Build an overlay node with the SAME seed as "cell" but
        // different content.
        let mut shadow_node = dummy_node("cell");
        shadow_node.content = "UPDATED_CELL_CONTENT".to_string();
        let shadowed_seed = shadow_node.seed;
        let on = TestOverlayNodes { nodes: vec![shadow_node.clone()] };
        let ob = empty_overlay_bonds();
        let tn = empty_tombstones_nodes();
        let tb = empty_tombstones_bonds();
        let save = CompressedSaveParams {
            block_size_target: 32 * 1024,
            zstd_level: 3,
            encryption_key: Some(&key),
        };
        let mut cur = std::io::Cursor::new(Vec::<u8>::new());
        write_compressed_network_payload_partial_passthrough_to(
            &cn_a, &save, &tn, &tb, &on, &ob, &mut cur,
        ).expect("partial passthrough shadow");
        let out = cur.into_inner();
        let cn_b = cn_from_payload_with_key(out, Some(key.clone()));
        // node_count unchanged — shadow replaces, not adds.
        assert_eq!(cn_b.node_count(), snap.nodes.len(),
            "shadowed node must not double-count");
        let view = cn_b.node_by_seed(&shadowed_seed)
            .expect("shadowed seed still resolvable");
        // Critical: reader returns OVERLAY content, NOT archive.
        assert_eq!(view.archived().content.as_str(), "UPDATED_CELL_CONTENT",
            "reader returned stale archive version instead of overlay");
    }

    #[test]
    fn test_partial_passthrough_roundtrip() {
        // Save with overlay, reload, verify all live nodes/bonds
        // present, all tombstoned/shadowed-archive entries inaccessible.
        let snap = dummy_snapshot();
        let key = MasterKey::from_bytes([29u8; 32]);
        let payload_a = build_encrypted_payload(&snap, 32 * 1024, 3, &key);
        let cn_a = cn_from_payload_with_key(payload_a, Some(key.clone()));
        // Tombstone "membrane" and the bond cell->membrane.
        let dead = Seed::from_label("membrane");
        let mut tn = HashSet::new(); tn.insert(dead);
        let mut tb = HashSet::new();
        tb.insert((Seed::from_label("cell"), dead));
        // Add a new node and a new bond.
        let added = dummy_node("new_thing");
        let added_seed = added.seed;
        let on = TestOverlayNodes { nodes: vec![added.clone()] };
        // bond from cell -> new_thing
        let new_bond = dummy_bond("cell", "new_thing", BondType::HasProperty, 0.6);
        let ob = TestOverlayBonds { bonds: vec![new_bond.clone()] };
        let save = CompressedSaveParams {
            block_size_target: 32 * 1024,
            zstd_level: 3,
            encryption_key: Some(&key),
        };
        let mut cur = std::io::Cursor::new(Vec::<u8>::new());
        write_compressed_network_payload_partial_passthrough_to(
            &cn_a, &save, &tn, &tb, &on, &ob, &mut cur,
        ).expect("partial passthrough roundtrip");
        let out = cur.into_inner();
        let cn_b = cn_from_payload_with_key(out, Some(key.clone()));
        // Dead node + dead bond gone.
        assert!(cn_b.node_by_seed(&dead).is_none());
        assert!(cn_b.bond_by_pair(&(Seed::from_label("cell"), dead)).is_none());
        // Live source survives.
        for n in &snap.nodes {
            if n.seed != dead {
                assert!(cn_b.node_by_seed(&n.seed).is_some());
            }
        }
        // New stuff present.
        assert!(cn_b.node_by_seed(&added_seed).is_some());
        assert!(cn_b.bond_by_pair(&(new_bond.from, new_bond.to)).is_some());
    }

    #[test]
    fn test_partial_passthrough_block_ids_preserved() {
        // Source blocks keep ids 0..N exactly; overlay blocks follow at N..
        // Verify by counting blocks and checking first N frames byte-equal
        // to source frames at the SAME id.
        let snap = dummy_snapshot();
        let key = MasterKey::from_bytes([31u8; 32]);
        let payload_a = build_encrypted_payload(&snap, 32 * 1024, 3, &key);
        let cn_a = cn_from_payload_with_key(payload_a, Some(key.clone()));
        let on = TestOverlayNodes { nodes: vec![dummy_node("appended")] };
        let ob = empty_overlay_bonds();
        let tn = empty_tombstones_nodes();
        let tb = empty_tombstones_bonds();
        let save = CompressedSaveParams {
            block_size_target: 32 * 1024,
            zstd_level: 3,
            encryption_key: Some(&key),
        };
        let mut cur = std::io::Cursor::new(Vec::<u8>::new());
        write_compressed_network_payload_partial_passthrough_to(
            &cn_a, &save, &tn, &tb, &on, &ob, &mut cur,
        ).expect("partial passthrough block ids");
        let out = cur.into_inner();
        let cn_b = cn_from_payload_with_key(out, Some(key.clone()));
        assert!(cn_b.block_count() > cn_a.block_count(),
            "overlay should append at least one block");
        for bid in 0..(cn_a.block_count() as u32) {
            let a = cn_a.block_frame_bytes(bid).unwrap();
            let b = cn_b.block_frame_bytes(bid).unwrap();
            assert_eq!(a, b, "source block id {} not preserved", bid);
        }
    }

    #[test]
    fn test_partial_passthrough_byte_identical_archive_section() {
        // The first src_block_count frames of the output are byte-equal
        // to the source frames (this is the load-bearing invariant for
        // keystream alignment).
        let snap = dummy_snapshot();
        let key = MasterKey::from_bytes([37u8; 32]);
        let payload_a = build_encrypted_payload(&snap, 32 * 1024, 3, &key);
        let cn_a = cn_from_payload_with_key(payload_a, Some(key.clone()));
        let on = empty_overlay_nodes();
        let ob = empty_overlay_bonds();
        let tn = empty_tombstones_nodes();
        let tb = empty_tombstones_bonds();
        let save = CompressedSaveParams {
            block_size_target: 32 * 1024,
            zstd_level: 3,
            encryption_key: Some(&key),
        };
        let mut cur = std::io::Cursor::new(Vec::<u8>::new());
        write_compressed_network_payload_partial_passthrough_to(
            &cn_a, &save, &tn, &tb, &on, &ob, &mut cur,
        ).expect("partial passthrough byte identical");
        let out = cur.into_inner();
        let cn_b = cn_from_payload_with_key(out, Some(key.clone()));
        assert_eq!(cn_a.block_count(), cn_b.block_count());
        for bid in 0..(cn_a.block_count() as u32) {
            let a = cn_a.block_frame_bytes(bid).unwrap();
            let b = cn_b.block_frame_bytes(bid).unwrap();
            assert_eq!(a, b, "block {} differs (empty overlay must be byte-equal to v3)", bid);
        }
    }
}
