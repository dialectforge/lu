//! OP-BRAIN-V4 Phase 2 / Phase 2.1 — SST writer.
//!
//! Sibling to `src/sst.rs`. Builds a v4 SST file from a stream of
//! bonds pushed in subject-sorted order. Used by Phase 2 tests, and
//! by Phase 4's MemTable flush + Phase 6's compaction.
//!
//! # Phase 2.1 amendment (2026-06-01)
//!
//! `push_bond` now accepts a `&[TemporalKeyframe]` history slice and
//! appends *every* keyframe contiguously to the per-block keyframe
//! pool, recording an `(offset, count)` range on the CompactBond.
//! Phase 2's "pick the most-recent keyframe and pool it" lossy
//! reduction is gone — the SST now preserves synthesis / session /
//! decay history end-to-end, which Phase 4's flush wiring depends on.
//! Cross-bond keyframe-pool dedup is deliberately not implemented in
//! this phase (simplicity > marginal compression win); each
//! `push_bond` appends a fresh contiguous run.
//!
//! # Caller contract
//!
//! `push_bond` MUST be called with subjects in non-decreasing
//! `Seed::as_bytes()` order. The writer validates this and returns
//! `SSTWriterError::OutOfOrder` if violated — caller is responsible
//! for sorting beforehand (Phase 4 will pre-sort the MemTable
//! snapshot before flushing). This is the same load-bearing contract
//! that lets the reader use binary search on the block index.
//!
//! # File layout
//!
//! 1. Reserved 96 bytes for the header (filled in at finalize).
//! 2. Block stream, one frame per block, in subject-sorted order.
//!    Frame: `[u32 compressed_len][u32 decompressed_len][payload]`.
//! 3. Block index (sorted by subject_min_seed).
//! 4. 8-byte footer (CRC32 of index + magic).
//!
//! `finalize` seeks back to offset 0 and writes the now-known header.
//!
//! # Encryption
//!
//! `key=None` writes plaintext payloads. `key=Some(_)` encrypts the
//! block payload via `xor_in_place` BEFORE zstd-compressing — wait,
//! actually AFTER. The order is: assemble plaintext payload, xor
//! against `xor_in_place(key, derive_block_keystream_index)`, then
//! zstd-compress the encrypted bytes. The reader reverses: decompress
//! then decrypt. This matches the spec ("zstd-compressed then
//! encrypted" reading from the on-disk direction = encrypted then
//! compressed at write time, which is what we do).
//!
//! Wait — re-reading: "Block payload format: zstd-compressed-then-
//! encrypted (existing v3 encryption scheme...)". On disk we want
//! `encrypt(compress(plaintext))`. So at write time: compress
//! plaintext, then encrypt the compressed bytes. At read time:
//! decrypt the on-disk frame payload, then decompress.
//!
//! ✅ This module compresses-then-encrypts; the reader
//! (`sst.rs::decompress_block`) decompresses-then-decrypts.
//
//! Actually re-reading sst.rs::decompress_block — it does
//! `zstd::decompress(comp_payload)` first, then `xor_in_place` on the
//! decompressed bytes. So the on-disk order is
//! `compress(encrypt(plaintext))`: encrypt plaintext, then compress.
//! At read time: decompress, then decrypt.
//!
//! Both modules use the same order: **encrypt first, compress second**.
//! That matches v3's `compress.rs` semantics: encryption is XOR-
//! keystream, which is byte-position-equivariant, so the order is a
//! free choice as long as writer and reader agree. We agree on:
//! write = encrypt(plaintext) then compress; read = decompress then
//! decrypt.

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::Path;

use rand::Rng;

use crate::bond::{BondType, TemporalKeyframe};
use crate::brain::BrainError;
use crate::encrypt::{xor_in_place, MasterKey};
use crate::node::Confidence;
use crate::seed::Seed;
use crate::sst::{
    confidence_to_u8, derive_block_keystream_index, encode_keyframe,
    pack_predicate, quantize_weight, BlockIndexEntry, SSTHeader,
    BLOCK_PAYLOAD_HEADER_BYTES, COMPACT_BOND_BYTES, MAX_KEYFRAMES_PER_BLOCK,
    SST_BLOCK_FRAME_HEADER_BYTES, SST_FOOTER_BYTES, SST_FOOTER_MAGIC,
    SST_HEADER_BYTES, SST_INDEX_ENTRY_BYTES, SST_MAGIC, SST_VERSION,
    TOMBSTONE_CONFIDENCE,
};

/// Target uncompressed payload size for a block, in bytes. Once the
/// in-memory bond buffer exceeds this, the block is flushed.
pub const DEFAULT_BLOCK_SIZE_TARGET: usize = 256 * 1024;

/// Default zstd level — matches v3 (`src/compress.rs::DEFAULT_ZSTD_LEVEL`).
pub const DEFAULT_ZSTD_LEVEL: i32 = 3;

/// Hard cap on bonds per block. A pathological subject with millions
/// of bonds must still fit in `COMPACT_BOND_BYTES × cap`; we set 1M
/// to keep decompressed blocks bounded (~14 MiB worst case post-2.1).
/// In normal use the byte target trips first.
pub const MAX_BONDS_PER_BLOCK: usize = 1_000_000;

/// Re-export `MAX_KEYFRAMES_PER_BLOCK` for callers that import it via
/// the writer module. The authoritative cap lives in `sst::`
/// (`u16::MAX`) because `keyframe_offset` is a `u16`.
pub use crate::sst::MAX_KEYFRAMES_PER_BLOCK as _MAX_KEYFRAMES_PER_BLOCK_USE_SST;

#[derive(Debug)]
pub enum SSTWriterError {
    OutOfOrder { prev: Seed, this: Seed },
    Io(std::io::Error),
    /// Keyframe pool would exceed `u16::MAX` entries in this block.
    PoolOverflow,
    /// A single bond's keyframe history exceeds 255 entries (`u8`
    /// keyframe_count width). Phase 2.1: real bonds keep their full
    /// history but the on-disk count field is a single byte.
    KeyframeCountOverflow { bond_kf_count: usize },
    AlreadyFinalized,
    NodeTableOverflow,
}

impl std::fmt::Display for SSTWriterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SSTWriterError::OutOfOrder { prev, this } => write!(
                f, "SSTWriter: bond subjects must be sorted; got {:?} after {:?}",
                this, prev
            ),
            SSTWriterError::Io(e) => write!(f, "SSTWriter IO: {}", e),
            SSTWriterError::PoolOverflow => write!(
                f, "SSTWriter: keyframe pool exceeded {} entries in one block",
                MAX_KEYFRAMES_PER_BLOCK
            ),
            SSTWriterError::KeyframeCountOverflow { bond_kf_count } => write!(
                f, "SSTWriter: bond carries {} keyframes; CompactBond.keyframe_count is u8 (max 255)",
                bond_kf_count
            ),
            SSTWriterError::AlreadyFinalized => write!(
                f, "SSTWriter: cannot push_bond after finalize"
            ),
            SSTWriterError::NodeTableOverflow => write!(
                f, "SSTWriter: block-local subject table exceeded u32::MAX entries"
            ),
        }
    }
}

impl std::error::Error for SSTWriterError {}

impl From<std::io::Error> for SSTWriterError {
    fn from(e: std::io::Error) -> Self { SSTWriterError::Io(e) }
}

impl From<SSTWriterError> for BrainError {
    fn from(e: SSTWriterError) -> Self {
        match e {
            SSTWriterError::Io(io) => BrainError::Io(std::io::Error::new(std::io::ErrorKind::Other, format!("SSTWriter: {}", io))),
            other => BrainError::Format(format!("{}", other)),
        }
    }
}

// ---------------------------------------------------------------------
// Internal: pending block being assembled in memory.
// ---------------------------------------------------------------------

/// In-flight block being assembled in memory.
///
/// **Phase 2.1**: the cross-bond keyframe dedup HashMap was removed.
/// Each `push_bond` appends its keyframe slice as a fresh contiguous
/// run, even if those exact keyframes appear elsewhere in the pool.
/// Rationale: cross-bond dedup would amortise only the all-Baseline
/// case (where most bonds share the timestamp=0 keyframe) and adds
/// writer complexity that conflicts with the new per-bond range
/// semantics. Per-block compression (zstd) catches the redundancy
/// well enough that the deferred dedup is small-marginal.
struct PendingBlock {
    // Block-local subject table (also covers objects). A small dedup
    // map keyed by Seed bytes.
    subjects: Vec<Seed>,
    subject_lookup: std::collections::HashMap<[u8; 32], u32>,
    /// Phase 10: parallel to `subjects` -- one Option<KnowledgeNode> per
    /// subject slot. `None` is legal (object-only references). Promoted
    /// from None to Some when a caller passes a node payload for a Seed
    /// the block has already interned.
    nodes: Vec<Option<crate::node::KnowledgeNode>>,
    /// Phase 10: cached serialized bytes for `nodes`, so payload-size
    /// accounting is honest at the byte target.
    node_bytes_cache: Vec<Vec<u8>>,
    // Per-block keyframe pool. **No dedup** in Phase 2.1 -- the slice
    // for each bond is appended contiguously.
    keyframes: Vec<TemporalKeyframe>,
    // CompactBond rows.
    bonds: Vec<RawBond>,
    // Approximate decompressed size accumulator, used for block-size
    // target. Accurate enough to drive flushing without re-measuring
    // the bincode-encoded keyframes each push.
    approx_bytes: usize,
    // First subject in this block (for the block-index entry).
    subject_min: Option<Seed>,
    subject_max: Option<Seed>,
}

#[derive(Clone, Copy)]
struct RawBond {
    subject_id: u32,
    object_id: u32,
    predicate: u8,
    weight: u8,
    // Phase 2.1: start index + length in the per-block keyframe pool.
    keyframe_offset: u16,
    keyframe_count: u8,
    confidence: u8,
}

impl PendingBlock {
    fn new() -> Self {
        PendingBlock {
            subjects: Vec::new(),
            subject_lookup: std::collections::HashMap::new(),
            nodes: Vec::new(),
            node_bytes_cache: Vec::new(),
            keyframes: Vec::new(),
            bonds: Vec::new(),
            approx_bytes: BLOCK_PAYLOAD_HEADER_BYTES,
            subject_min: None,
            subject_max: None,
        }
    }

    fn intern_subject(&mut self, s: &Seed) -> Result<u32, SSTWriterError> {
        if let Some(&id) = self.subject_lookup.get(s.as_bytes()) {
            return Ok(id);
        }
        if self.subjects.len() >= u32::MAX as usize {
            return Err(SSTWriterError::NodeTableOverflow);
        }
        let id = self.subjects.len() as u32;
        self.subjects.push(*s);
        self.subject_lookup.insert(*s.as_bytes(), id);
        // Phase 10: maintain parallel node table slots. Push None;
        // attach_node promotes to Some on demand.
        self.nodes.push(None);
        self.node_bytes_cache.push(Vec::new());
        self.approx_bytes += 32 + 4; // 32 byte Seed + 4 byte node_len placeholder
        Ok(id)
    }

    /// Phase 10: attach (or replace) a KnowledgeNode payload to the
    /// already-interned slot at `id`. Idempotent: if the slot already
    /// holds an identical payload, no-op (we keep the first writer
    /// to preserve insertion order; differing payloads are overwritten
    /// because the caller flow is monotonic-seq).
    fn attach_node(
        &mut self,
        id: u32,
        node: &crate::node::KnowledgeNode,
    ) -> Result<(), SSTWriterError> {
        let slot = id as usize;
        if slot >= self.nodes.len() {
            return Err(SSTWriterError::NodeTableOverflow);
        }
        if self.nodes[slot].is_some() {
            return Ok(());
        }
        let bytes = crate::node::serialize_knowledge_node(node)
            .map_err(|e| SSTWriterError::Io(
                std::io::Error::new(std::io::ErrorKind::Other,
                    format!("serialize node: {}", e)),
            ))?;
        self.approx_bytes += bytes.len();
        self.nodes[slot] = Some(node.clone());
        self.node_bytes_cache[slot] = bytes;
        Ok(())
    }

    /// Append `kfs` to the keyframe pool as a contiguous run. Returns
    /// `(offset, count)` for the new range. Caller is responsible for
    /// ensuring `kfs.len() <= u8::MAX` and that the block has room for
    /// the run — those checks live in `push_bond` so they can trigger
    /// a `flush_block` before appending.
    fn append_keyframes(&mut self, kfs: &[TemporalKeyframe]) -> (u16, u8) {
        let offset = self.keyframes.len() as u16;
        for kf in kfs {
            let bytes = bincode::serialize(kf).expect("keyframe bincode serialize");
            self.approx_bytes += 4 + bytes.len();
            self.keyframes.push(kf.clone());
        }
        (offset, kfs.len() as u8)
    }

    fn push(
        &mut self,
        subject_id: u32,
        object_id: u32,
        predicate: u8,
        weight: u8,
        keyframe_offset: u16,
        keyframe_count: u8,
        confidence: u8,
    ) {
        self.bonds.push(RawBond {
            subject_id, object_id, predicate, weight,
            keyframe_offset, keyframe_count, confidence,
        });
        self.approx_bytes += COMPACT_BOND_BYTES;
    }

    fn is_empty(&self) -> bool {
        // Phase 10: a block is empty only when it carries neither
        // bonds nor node payloads. push_node alone (no bonds) still
        // needs to land on disk.
        self.bonds.is_empty() && self.nodes.iter().all(|n| n.is_none())
    }

    /// Serialize the block payload to a fresh `Vec<u8>` matching the
    /// `parse_block` reader contract.
    fn serialize_payload(&self) -> Vec<u8> {
        let st_bytes = self.subjects.len() * 32;
        // Phase 10: node table bytes -- 4-byte length-prefix per slot
        // plus the slot's serialized payload (may be 0 for None slots).
        let mut nt_bytes: usize = 0;
        for cached in &self.node_bytes_cache {
            nt_bytes += 4 + cached.len();
        }
        let mut kf_bytes: usize = 0;
        for kf in &self.keyframes {
            let b = bincode::serialize(kf).expect("keyframe bincode serialize");
            kf_bytes += 4 + b.len();
        }
        let bonds_bytes = self.bonds.len() * COMPACT_BOND_BYTES;
        let mut buf = Vec::with_capacity(
            BLOCK_PAYLOAD_HEADER_BYTES + st_bytes + nt_bytes + kf_bytes + bonds_bytes,
        );
        // Header.
        let smin_id: u32 = 0;
        let smax_id: u32 = if self.subjects.is_empty() { 0 } else {
            (self.subjects.len() - 1) as u32
        };
        buf.extend_from_slice(&smin_id.to_le_bytes());
        buf.extend_from_slice(&smax_id.to_le_bytes());
        buf.extend_from_slice(&(self.bonds.len() as u32).to_le_bytes());
        buf.extend_from_slice(&(self.keyframes.len() as u32).to_le_bytes());
        buf.extend_from_slice(&(self.subjects.len() as u32).to_le_bytes());
        // Phase 10: was `reserved` (zero); now node_table_size == subject_table_size.
        buf.extend_from_slice(&(self.subjects.len() as u32).to_le_bytes());
        debug_assert_eq!(buf.len(), BLOCK_PAYLOAD_HEADER_BYTES);
        // Subject table.
        for s in &self.subjects {
            buf.extend_from_slice(s.as_bytes());
        }
        // Phase 10: Node table. One length-prefixed payload per subject
        // slot (0 length => None).
        debug_assert_eq!(self.node_bytes_cache.len(), self.subjects.len());
        for cached in &self.node_bytes_cache {
            buf.extend_from_slice(&(cached.len() as u32).to_le_bytes());
            buf.extend_from_slice(cached);
        }
        // Keyframe pool.
        for kf in &self.keyframes {
            encode_keyframe(kf, &mut buf);
        }
        // Bonds (Phase 2.1: 14 bytes -- u32 + u32 + u8 + u8 + u16 + u8 + u8).
        for b in &self.bonds {
            buf.extend_from_slice(&b.subject_id.to_le_bytes());
            buf.extend_from_slice(&b.object_id.to_le_bytes());
            buf.push(b.predicate);
            buf.push(b.weight);
            buf.extend_from_slice(&b.keyframe_offset.to_le_bytes());
            buf.push(b.keyframe_count);
            buf.push(b.confidence);
        }
        debug_assert_eq!(
            buf.len(),
            BLOCK_PAYLOAD_HEADER_BYTES + st_bytes + nt_bytes + kf_bytes + bonds_bytes,
        );
        buf
    }
}

// ---------------------------------------------------------------------
// Writer
// ---------------------------------------------------------------------

pub struct SSTWriter {
    file: BufWriter<File>,
    level: u8,
    block_size_target: usize,
    zstd_level: i32,
    key: Option<MasterKey>,
    pending: PendingBlock,
    block_index: Vec<BlockIndexEntry>,
    total_bond_count: u64,
    cursor_offset: u64,
    finalized: bool,
    last_subject: Option<Seed>,
    overall_min: Option<Seed>,
    overall_max: Option<Seed>,
    manifest_id: u64,
    encryption_iv: [u8; 16],
}

/// Metadata captured at SST finalize time, used by Phase 4 flush
/// wiring to build a `SSTEntry` for the manifest without re-opening
/// the file.
#[derive(Debug, Clone, Copy)]
pub struct SSTFinalizeMeta {
    pub bond_count: u64,
    pub block_count: u64,
    pub subject_min: Seed,
    pub subject_max: Seed,
    pub manifest_id: u64,
}

impl SSTWriter {
    /// Create a new SST writer with default block_size_target and
    /// zstd_level. The `level` is the LSM-tree level number (0 = L0,
    /// recent flushes; 1+ = compacted).
    pub fn create(
        path: &Path,
        level: u8,
        key: Option<MasterKey>,
    ) -> Result<Self, SSTWriterError> {
        let file = OpenOptions::new()
            .create(true).write(true).truncate(true).read(true)
            .open(path)?;
        let mut file = BufWriter::with_capacity(1 * 1024 * 1024, file);
        // Reserve header space.
        file.write_all(&[0u8; SST_HEADER_BYTES])?;
        let mut rng = rand::thread_rng();
        let mut iv = [0u8; 16];
        rng.fill(&mut iv);
        let manifest_id: u64 = rng.gen();
        Ok(SSTWriter {
            file,
            level,
            block_size_target: DEFAULT_BLOCK_SIZE_TARGET,
            zstd_level: DEFAULT_ZSTD_LEVEL,
            key,
            pending: PendingBlock::new(),
            block_index: Vec::new(),
            total_bond_count: 0,
            cursor_offset: SST_HEADER_BYTES as u64,
            finalized: false,
            last_subject: None,
            overall_min: None,
            overall_max: None,
            manifest_id,
            encryption_iv: iv,
        })
    }

    /// Override the block-size target. Useful for tests that want
    /// many blocks at low bond counts.
    pub fn set_block_size_target(&mut self, bytes: usize) {
        self.block_size_target = bytes;
    }

    /// Override the zstd compression level (default 3).
    pub fn set_zstd_level(&mut self, level: i32) {
        self.zstd_level = level;
    }

    /// Override the manifest id. By default a fresh random id is
    /// assigned at `create`; Phase 4's MemTable flush will pass an
    /// id pulled from the live manifest.
    pub fn set_manifest_id(&mut self, id: u64) {
        self.manifest_id = id;
    }

    pub fn manifest_id(&self) -> u64 {
        self.manifest_id
    }

    /// Push one bond with its full keyframe history.
    ///
    /// **Phase 2.1**: `keyframes` is appended to the per-block pool as
    /// a contiguous run and recorded on the CompactBond as
    /// `(offset, count)`. Empty `keyframes` → the bond has
    /// `keyframe_count == 0` and no keyframes are appended (legal but
    /// unusual; every well-formed bond should carry ≥1).
    ///
    /// Caller still must push subjects in non-decreasing
    /// `Seed::as_bytes()` order. A keyframe-history slice longer than
    /// 255 returns `KeyframeCountOverflow` (single bond's history
    /// doesn't fit in the `u8` count field). A block that would
    /// overflow the per-block keyframe pool (cap `u16::MAX`) is
    /// flushed first.
    pub fn push_bond(
        &mut self,
        subject: &Seed,
        object: &Seed,
        bond_type: BondType,
        weight: f32,
        confidence: Confidence,
        keyframes: &[TemporalKeyframe],
    ) -> Result<(), SSTWriterError> {
        if self.finalized {
            return Err(SSTWriterError::AlreadyFinalized);
        }
        if keyframes.len() > u8::MAX as usize {
            return Err(SSTWriterError::KeyframeCountOverflow {
                bond_kf_count: keyframes.len(),
            });
        }
        if let Some(prev) = &self.last_subject {
            if subject.as_bytes() < prev.as_bytes() {
                return Err(SSTWriterError::OutOfOrder {
                    prev: *prev, this: *subject,
                });
            }
        }

        // Decide if we need to flush the current block before pushing
        // this bond. Two triggers, both gated on a subject change so
        // we never split a single subject's bonds across blocks:
        //   - approx_bytes exceeds the byte target;
        //   - bond count exceeds MAX_BONDS_PER_BLOCK;
        //   - the keyframe pool wouldn't fit `keyframes.len()` more
        //     entries without crossing `MAX_KEYFRAMES_PER_BLOCK`.
        let subject_changed = self.last_subject
            .map(|s| s != *subject)
            .unwrap_or(true);
        let pool_would_overflow = self.pending.keyframes.len()
            .saturating_add(keyframes.len())
            > MAX_KEYFRAMES_PER_BLOCK;
        if subject_changed
            && !self.pending.is_empty()
            && (self.pending.approx_bytes >= self.block_size_target
                || self.pending.bonds.len() >= MAX_BONDS_PER_BLOCK
                || pool_would_overflow)
        {
            self.flush_block()?;
        }
        // Even after a flush, an oversized single push (>u16::MAX kfs)
        // could blow the pool. Re-check.
        if self.pending.keyframes.len()
            .saturating_add(keyframes.len())
            > MAX_KEYFRAMES_PER_BLOCK
        {
            return Err(SSTWriterError::PoolOverflow);
        }

        // Intern subject & object into the current block.
        let subj_id = self.pending.intern_subject(subject)?;
        let obj_id  = self.pending.intern_subject(object)?;
        if self.pending.subject_min.is_none() {
            self.pending.subject_min = Some(*subject);
        }
        self.pending.subject_max = Some(*subject);

        // Append the full keyframe history as a contiguous range.
        let (kf_offset, kf_count) = self.pending.append_keyframes(keyframes);

        let predicate = pack_predicate(bond_type, false, 0);
        let weight_q = quantize_weight(weight);
        let conf_b = confidence_to_u8(confidence);
        self.pending.push(
            subj_id, obj_id, predicate, weight_q,
            kf_offset, kf_count, conf_b,
        );

        if self.overall_min.is_none() {
            self.overall_min = Some(*subject);
        }
        self.overall_max = Some(*subject);
        self.last_subject = Some(*subject);
        self.total_bond_count += 1;
        Ok(())
    }

    /// Phase 10: push one bond AND attach KnowledgeNode payloads to
    /// the subject and/or object Seed slots in the per-block node
    /// table. If a slot for the same Seed already carries a node
    /// payload (within this block), the existing payload is kept --
    /// the writer is monotonic-seq aware via the flush ordering, not
    /// per-slot timestamps. Either node argument may be `None`,
    /// which leaves the slot untouched (the existing `None` or
    /// previously-attached `Some` survives).
    pub fn push_bond_with_nodes(
        &mut self,
        subject: &Seed,
        subject_node: Option<&crate::node::KnowledgeNode>,
        object: &Seed,
        object_node: Option<&crate::node::KnowledgeNode>,
        bond_type: BondType,
        weight: f32,
        confidence: Confidence,
        keyframes: &[TemporalKeyframe],
    ) -> Result<(), SSTWriterError> {
        if self.finalized {
            return Err(SSTWriterError::AlreadyFinalized);
        }
        if keyframes.len() > u8::MAX as usize {
            return Err(SSTWriterError::KeyframeCountOverflow {
                bond_kf_count: keyframes.len(),
            });
        }
        if let Some(prev) = &self.last_subject {
            if subject.as_bytes() < prev.as_bytes() {
                return Err(SSTWriterError::OutOfOrder {
                    prev: *prev, this: *subject,
                });
            }
        }

        let subject_changed = self.last_subject
            .map(|s| s != *subject)
            .unwrap_or(true);
        let pool_would_overflow = self.pending.keyframes.len()
            .saturating_add(keyframes.len())
            > MAX_KEYFRAMES_PER_BLOCK;
        if subject_changed
            && !self.pending.is_empty()
            && (self.pending.approx_bytes >= self.block_size_target
                || self.pending.bonds.len() >= MAX_BONDS_PER_BLOCK
                || pool_would_overflow)
        {
            self.flush_block()?;
        }
        if self.pending.keyframes.len()
            .saturating_add(keyframes.len())
            > MAX_KEYFRAMES_PER_BLOCK
        {
            return Err(SSTWriterError::PoolOverflow);
        }

        let subj_id = self.pending.intern_subject(subject)?;
        if let Some(n) = subject_node {
            self.pending.attach_node(subj_id, n)?;
        }
        let obj_id = self.pending.intern_subject(object)?;
        if let Some(n) = object_node {
            self.pending.attach_node(obj_id, n)?;
        }
        if self.pending.subject_min.is_none() {
            self.pending.subject_min = Some(*subject);
        }
        self.pending.subject_max = Some(*subject);

        let (kf_offset, kf_count) = self.pending.append_keyframes(keyframes);
        let predicate = pack_predicate(bond_type, false, 0);
        let weight_q = quantize_weight(weight);
        let conf_b = confidence_to_u8(confidence);
        self.pending.push(
            subj_id, obj_id, predicate, weight_q,
            kf_offset, kf_count, conf_b,
        );

        if self.overall_min.is_none() {
            self.overall_min = Some(*subject);
        }
        self.overall_max = Some(*subject);
        self.last_subject = Some(*subject);
        self.total_bond_count += 1;
        Ok(())
    }

    /// Phase 10: standalone node push for cases where a node
    /// metadata record needs to land in the SST without a paired bond
    /// (e.g. migration of an isolated node or post-flush node-only
    /// records). Lazily-interns the seed into the current block and
    /// attaches the payload. The subject-sort contract still applies:
    /// the seed must not regress versus the writer's `last_subject`.
    pub fn push_node(
        &mut self,
        seed: &Seed,
        node: &crate::node::KnowledgeNode,
    ) -> Result<(), SSTWriterError> {
        if self.finalized {
            return Err(SSTWriterError::AlreadyFinalized);
        }
        if let Some(prev) = &self.last_subject {
            if seed.as_bytes() < prev.as_bytes() {
                return Err(SSTWriterError::OutOfOrder {
                    prev: *prev, this: *seed,
                });
            }
        }
        let subject_changed = self.last_subject
            .map(|s| s != *seed)
            .unwrap_or(true);
        if subject_changed
            && !self.pending.is_empty()
            && (self.pending.approx_bytes >= self.block_size_target
                || self.pending.bonds.len() >= MAX_BONDS_PER_BLOCK)
        {
            self.flush_block()?;
        }
        let id = self.pending.intern_subject(seed)?;
        self.pending.attach_node(id, node)?;
        if self.pending.subject_min.is_none() {
            self.pending.subject_min = Some(*seed);
        }
        self.pending.subject_max = Some(*seed);
        if self.overall_min.is_none() {
            self.overall_min = Some(*seed);
        }
        self.overall_max = Some(*seed);
        self.last_subject = Some(*seed);
        Ok(())
    }

    /// Phase 6: push a tombstone row for `(subject, predicate,
    /// object)`. Identical to `push_bond` in shape but writes the
    /// reserved `TOMBSTONE_CONFIDENCE` byte in the on-disk
    /// confidence field. Tombstones still respect the writer's
    /// non-decreasing-subject contract and still count toward
    /// `bond_count`.
    pub fn push_tombstone(
        &mut self,
        subject: &Seed,
        object: &Seed,
        bond_type: BondType,
    ) -> Result<(), SSTWriterError> {
        if self.finalized {
            return Err(SSTWriterError::AlreadyFinalized);
        }
        if let Some(prev) = &self.last_subject {
            if subject.as_bytes() < prev.as_bytes() {
                return Err(SSTWriterError::OutOfOrder {
                    prev: *prev, this: *subject,
                });
            }
        }
        let subject_changed = self.last_subject
            .map(|s| s != *subject)
            .unwrap_or(true);
        if subject_changed
            && !self.pending.is_empty()
            && (self.pending.approx_bytes >= self.block_size_target
                || self.pending.bonds.len() >= MAX_BONDS_PER_BLOCK)
        {
            self.flush_block()?;
        }
        let subj_id = self.pending.intern_subject(subject)?;
        let obj_id  = self.pending.intern_subject(object)?;
        if self.pending.subject_min.is_none() {
            self.pending.subject_min = Some(*subject);
        }
        self.pending.subject_max = Some(*subject);
        let (kf_offset, kf_count) = self.pending.append_keyframes(&[]);
        let predicate = pack_predicate(bond_type, false, 0);
        // Tombstone weight is zero; the dedup layer never reads
        // a tombstone's weight, but using 0 keeps any accidental
        // read deterministic.
        self.pending.push(
            subj_id, obj_id, predicate, 0u8,
            kf_offset, kf_count, TOMBSTONE_CONFIDENCE,
        );
        if self.overall_min.is_none() {
            self.overall_min = Some(*subject);
        }
        self.overall_max = Some(*subject);
        self.last_subject = Some(*subject);
        self.total_bond_count += 1;
        Ok(())
    }

    fn flush_block(&mut self) -> Result<(), SSTWriterError> {
        if self.pending.is_empty() {
            return Ok(());
        }
        let plaintext = self.pending.serialize_payload();
        let decompressed_length = plaintext.len() as u32;

        // Encrypt first (XOR-keystream is byte-position-equivariant).
        let mut encrypted = plaintext;
        if let Some(k) = &self.key {
            let block_idx = derive_block_keystream_index(
                self.manifest_id, self.block_index.len() as u32, &self.encryption_iv,
            );
            xor_in_place(k, block_idx, &mut encrypted);
        }

        // Compress.
        let compressed = zstd::bulk::compress(&encrypted, self.zstd_level)
            .map_err(|e| SSTWriterError::Io(
                std::io::Error::new(std::io::ErrorKind::Other,
                    format!("zstd compress: {}", e)
                )
            ))?;
        let compressed_length = compressed.len() as u32;

        // Write frame header + payload.
        let frame_off = self.cursor_offset;
        self.file.write_all(&compressed_length.to_le_bytes())?;
        self.file.write_all(&decompressed_length.to_le_bytes())?;
        self.file.write_all(&compressed)?;
        self.cursor_offset += SST_BLOCK_FRAME_HEADER_BYTES as u64
            + compressed_length as u64;

        let subject_min = self.pending.subject_min.unwrap();
        self.block_index.push(BlockIndexEntry {
            subject_min_seed: subject_min,
            offset: frame_off,
            compressed_length,
            decompressed_length,
        });

        self.pending = PendingBlock::new();
        Ok(())
    }

    /// Finalize the SST: flush the trailing block, write the block
    /// index, write the footer, then seek to offset 0 and write the
    /// now-known header. Consumes self. Returns metadata that lets
    /// Phase 4's flush worker build a `SSTEntry` for the manifest
    /// without re-opening the file.
    pub fn finalize_with_meta(mut self) -> Result<SSTFinalizeMeta, SSTWriterError> {
        if self.finalized {
            return Err(SSTWriterError::AlreadyFinalized);
        }
        self.finalized = true;
        self.flush_block()?;

        // Write block index.
        let index_offset = self.cursor_offset;
        let mut idx_bytes = vec![0u8; self.block_index.len() * SST_INDEX_ENTRY_BYTES];
        for (i, e) in self.block_index.iter().enumerate() {
            e.write_to(&mut idx_bytes[i * SST_INDEX_ENTRY_BYTES..(i + 1) * SST_INDEX_ENTRY_BYTES]);
        }
        let crc = crc32fast::hash(&idx_bytes);
        self.file.write_all(&idx_bytes)?;
        let index_length = idx_bytes.len() as u64;
        self.cursor_offset += index_length;

        // Footer: CRC32 + magic.
        self.file.write_all(&crc.to_le_bytes())?;
        self.file.write_all(&SST_FOOTER_MAGIC.to_le_bytes())?;
        self.cursor_offset += SST_FOOTER_BYTES as u64;

        // Flush buffered writes.
        self.file.flush()?;

        // Write the now-known header.
        let header = SSTHeader {
            magic: SST_MAGIC,
            version: SST_VERSION,
            level: self.level,
            bond_count: self.total_bond_count,
            block_count: self.block_index.len() as u64,
            manifest_id: self.manifest_id,
            index_offset,
            index_length,
            encryption_iv: self.encryption_iv,
            subject_min_seed: self.overall_min.unwrap_or(Seed::from_bytes([0u8; 32])),
            subject_max_seed: self.overall_max.unwrap_or(Seed::from_bytes([0u8; 32])),
        };
        let mut header_buf = vec![0u8; SST_HEADER_BYTES];
        header.write_to(&mut header_buf);

        // Seek back to start and write header.
        let inner: File = self.file.into_inner()
            .map_err(|e| SSTWriterError::Io(e.into_error()))?;
        let mut inner = inner;
        inner.seek(SeekFrom::Start(0))?;
        inner.write_all(&header_buf)?;
        inner.sync_all()?;
        Ok(SSTFinalizeMeta {
            bond_count: header.bond_count,
            block_count: header.block_count,
            subject_min: header.subject_min_seed,
            subject_max: header.subject_max_seed,
            manifest_id: header.manifest_id,
        })
    }

    /// Backwards-compat shim for callers that only need the success/
    /// failure result. Existing tests + Phase 2 callsites use this.
    pub fn finalize(self) -> Result<(), SSTWriterError> {
        self.finalize_with_meta().map(|_| ())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::seed::Seed;
    use crate::sst::{BlockCache, SST, DEFAULT_SST_BLOCK_CACHE_BYTES};
    use std::sync::Arc;

    #[test]
    fn out_of_order_push_errors() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("sst_writer_ooo_{}.sst", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let mut w = SSTWriter::create(&path, 0, None).unwrap();
        let a = Seed::from_bytes([0x10u8; 32]);
        let b = Seed::from_bytes([0x05u8; 32]);
        let o = Seed::from_bytes([0xAAu8; 32]);
        w.push_bond(&a, &o, BondType::IsA, 0.5, Confidence::Verified, &[]).unwrap();
        let r = w.push_bond(&b, &o, BondType::IsA, 0.5, Confidence::Verified, &[]);
        assert!(matches!(r, Err(SSTWriterError::OutOfOrder { .. })));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn small_block_size_creates_multiple_blocks() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("sst_writer_multiblock_{}.sst", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let mut w = SSTWriter::create(&path, 0, None).unwrap();
        w.set_block_size_target(4 * 1024);
        // 500 subjects, 4 bonds each.
        let mut subjects: Vec<Seed> = (0..500u32)
            .map(|i| {
                let mut b = [0u8; 32];
                b[0..4].copy_from_slice(&i.to_be_bytes());
                Seed::from_bytes(b)
            })
            .collect();
        subjects.sort_by(|a, b| a.as_bytes().cmp(b.as_bytes()));
        for s in &subjects {
            for j in 0..4u32 {
                let mut ob = [0u8; 32];
                ob[0..4].copy_from_slice(&(j + 100).to_be_bytes());
                let o = Seed::from_bytes(ob);
                w.push_bond(s, &o, BondType::IsA, 0.5, Confidence::Verified, &[]).unwrap();
            }
        }
        w.finalize().unwrap();
        let cache = Arc::new(BlockCache::new(DEFAULT_SST_BLOCK_CACHE_BYTES));
        let sst = SST::open(&path, None, cache).unwrap();
        assert!(sst.block_count() > 1, "expected multiple blocks, got {}", sst.block_count());
        assert_eq!(sst.header().bond_count, 2000);
        let total: usize = sst.iter_bonds().map(|r| r.map(|_| 1).unwrap_or(0)).sum();
        assert_eq!(total, 2000);
        let _ = std::fs::remove_file(&path);
    }

    // -----------------------------------------------------------------
    // Phase 10 -- node-table round-trip.
    // -----------------------------------------------------------------
    fn make_kn(label: &str) -> crate::node::KnowledgeNode {
        crate::node::KnowledgeNode::with_seed(
            Seed::from_label(label),
            label.to_string(),
            crate::node::KnowledgeContent::Concept(label.to_string()),
            Confidence::Verified,
            7,
        )
    }

    #[test]
    fn phase10_push_bond_with_nodes_round_trip() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("sst_writer_phase10_{}.sst", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let mut w = SSTWriter::create(&path, 0, None).unwrap();
        let subj = Seed::from_label("alpha");
        let obj = Seed::from_label("beta");
        let n_subj = make_kn("alpha");
        let n_obj = make_kn("beta");
        w.push_bond_with_nodes(
            &subj, Some(&n_subj),
            &obj, Some(&n_obj),
            BondType::IsA, 0.8, Confidence::Verified, &[],
        ).unwrap();
        w.finalize().unwrap();
        let cache = Arc::new(BlockCache::new(DEFAULT_SST_BLOCK_CACHE_BYTES));
        let sst = SST::open(&path, None, cache).unwrap();
        let bonds = sst.get_bonds_for_subject(&subj).unwrap();
        assert_eq!(bonds.len(), 1);
        let bv = &bonds[0];
        let got_subj_node = bv.subject_node().expect("subject node present");
        assert_eq!(got_subj_node.seed, n_subj.seed);
        assert_eq!(got_subj_node.content, n_subj.content);
        let got_obj_node = bv.object_node().expect("object node present");
        assert_eq!(got_obj_node.seed, n_obj.seed);
        assert_eq!(got_obj_node.content, n_obj.content);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn phase10_iter_nodes_returns_per_block_node_table() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("sst_writer_phase10_iter_{}.sst", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let mut w = SSTWriter::create(&path, 0, None).unwrap();
        // 10 sorted subjects, each with a node payload, each pushed
        // via push_bond_with_nodes against a shared object.
        let mut subjects: Vec<Seed> = (0..10u32)
            .map(|i| {
                let mut b = [0u8; 32];
                b[0..4].copy_from_slice(&i.to_be_bytes());
                Seed::from_bytes(b)
            })
            .collect();
        subjects.sort_by(|a, b| a.as_bytes().cmp(b.as_bytes()));
        let obj = Seed::from_bytes([0xFF; 32]);
        let obj_node = make_kn("shared_obj");
        let mut subj_nodes: Vec<crate::node::KnowledgeNode> = Vec::new();
        for (i, s) in subjects.iter().enumerate() {
            let n = crate::node::KnowledgeNode::with_seed(
                *s,
                format!("node_{}", i),
                crate::node::KnowledgeContent::Concept(format!("content_{}", i)),
                Confidence::Verified,
                42,
            );
            subj_nodes.push(n.clone());
            w.push_bond_with_nodes(
                s, Some(&n),
                &obj, Some(&obj_node),
                BondType::IsA, 0.5, Confidence::Verified, &[],
            ).unwrap();
        }
        w.finalize().unwrap();
        let cache = Arc::new(BlockCache::new(DEFAULT_SST_BLOCK_CACHE_BYTES));
        let sst = SST::open(&path, None, cache).unwrap();
        let pairs = sst.iter_nodes().unwrap();
        // 10 subject nodes + 1 object node = 11 distinct seeds.
        assert_eq!(pairs.len(), 11);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn phase10_push_node_standalone() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("sst_writer_phase10_standalone_{}.sst", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let mut w = SSTWriter::create(&path, 0, None).unwrap();
        let mut seeds: Vec<Seed> = (0..5u32)
            .map(|i| Seed::from_label(&format!("iso_{}", i)))
            .collect();
        seeds.sort_by(|a, b| a.as_bytes().cmp(b.as_bytes()));
        for s in &seeds {
            let n = make_kn(&format!("iso_payload_for_{:?}", s.as_bytes()[0]));
            w.push_node(s, &n).unwrap();
        }
        w.finalize().unwrap();
        let cache = Arc::new(BlockCache::new(DEFAULT_SST_BLOCK_CACHE_BYTES));
        let sst = SST::open(&path, None, cache).unwrap();
        let pairs = sst.iter_nodes().unwrap();
        assert_eq!(pairs.len(), 5);
        // No bonds were pushed; bond_count must be 0.
        assert_eq!(sst.header().bond_count, 0);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn block_index_correctness_across_boundaries() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("sst_writer_boundaries_{}.sst", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let mut w = SSTWriter::create(&path, 0, None).unwrap();
        w.set_block_size_target(1024); // very small
        // 30 subjects, predictable.
        let subjects: Vec<Seed> = (0..30u32)
            .map(|i| {
                let mut b = [0u8; 32];
                b[0] = (i * 8) as u8;
                b[1] = i as u8;
                Seed::from_bytes(b)
            })
            .collect();
        let mut sorted = subjects.clone();
        sorted.sort_by(|a, b| a.as_bytes().cmp(b.as_bytes()));
        let o = Seed::from_bytes([0xFF; 32]);
        for s in &sorted {
            w.push_bond(s, &o, BondType::IsA, 0.7, Confidence::Verified, &[]).unwrap();
        }
        w.finalize().unwrap();
        let cache = Arc::new(BlockCache::new(DEFAULT_SST_BLOCK_CACHE_BYTES));
        let sst = SST::open(&path, None, cache).unwrap();
        assert!(sst.block_count() >= 2);
        for s in &sorted {
            let v = sst.get_bonds_for_subject(s).unwrap();
            assert_eq!(v.len(), 1, "subject {:?}", s);
            assert_eq!(v[0].object, o);
        }
        let _ = std::fs::remove_file(&path);
    }
}
