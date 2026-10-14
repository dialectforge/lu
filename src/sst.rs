//! OP-BRAIN-V4 Phase 2 / Phase 2.1 — SSTable read path.
//!
//! An SST (Sorted Subject Table) is an immutable on-disk file of
//! subject-sorted bonds, packed into zstd-compressed encrypted blocks
//! with a binary-searchable block index. The format is designed for
//! the v4 LSM-tree storage layer: Phase 4's MemTable flush writes
//! SSTs; Phase 6's compaction merges them.
//!
//! Phase 2 shipped the read path + writer. **Phase 2.1 (2026-06-01)**
//! amends the block-level CompactBond layout from 13 to 14 bytes so
//! each bond can carry a *range* of keyframes from the per-block
//! keyframe pool, not a single index. Motivation: Phase 3 surfaced
//! that a single-keyframe SST would silently drop synthesis /
//! session / decay-stamped history at flush time. Lu's curiosity
//! engine is load-bearing on that history
//! (`feedback_curiosity_load_bearing`). On-disk SST format version
//! goes 1 → 2; v1 SSTs were Phase 2 test fixtures only and the v2
//! reader refuses them with a clear error message.
//!
//! # On-disk layout
//!
//! ```text
//! [header 132 bytes]
//!   u32 magic              = 0x4C534F34 ("LSO4")
//!   u32 version            = 2          // Phase 2.1 (was 1 in Phase 2)
//!   u8  level
//!   u8  reserved[3]
//!   u64 bond_count
//!   u64 block_count
//!   u64 manifest_id        // ties to Phase 4 manifest
//!   u64 index_offset       // byte offset to the block-index array
//!   u64 index_length       // bytes
//!   [u8;16] encryption_iv  // per-SST IV (stored even when no key)
//!   [u8;32] subject_min_seed
//!   [u8;32] subject_max_seed
//!
//! [blocks ...]             // count = block_count, each frame-prefixed
//!
//! [block_index]            // sorted by subject_min
//!   per entry (48 bytes):
//!     [u8;32] subject_min_seed
//!     u64 offset            // absolute byte offset of frame
//!     u32 compressed_length // payload bytes inside frame (post-encrypt)
//!     u32 decompressed_length // bytes after zstd-decompress
//!
//! [footer 8 bytes]
//!   u32 crc32_of_block_index
//!   u32 magic_footer       = 0x34534F4C ("4SOL")
//! ```
//!
//! Each block frame is: `[u32 compressed_len][u32 decompressed_len][payload]`.
//! The payload is `xor_in_place`-encrypted with the master key when a
//! key is present, then zstd-compressed. (Decompress → decrypt order
//! mirrors `src/compress.rs`'s v3 scheme so the same key derivation
//! applies.)
//!
//! Decompressed block payload, in order:
//!   - block_header (24 bytes): subject_min_seed_idx_u32, subject_max_seed_idx_u32,
//!                              bond_count_u32, keyframe_pool_size_u32,
//!                              subject_table_size_u32, node_table_size_u32
//!   - subject_table: `subject_table_size_u32` × 32-byte Seed records
//!     mapping block-local NodeId (u32) → Seed.
//!   - node_table: `node_table_size_u32` × `(u32 len + bincode-payload)`
//!     records. Phase 10 addition; `node_table_size_u32` must equal
//!     `subject_table_size_u32`. A zero-length payload means "no node
//!     metadata for this Seed in this block" (legal for object-only
//!     references). Decoded into `Option<KnowledgeNode>` per slot.
//!   - keyframe_pool: `keyframe_pool_size_u32` × variable-length keyframe
//!     records (bincode-encoded TemporalKeyframe, length-prefixed u32).
//!     A bond's keyframe history is `pool[offset .. offset + count]`
//!     (Phase 2.1). `keyframe_count == 0` means the bond carries no
//!     keyframes (unusual — every well-formed bond should have
//!     ≥1). Each push from the writer appends its keyframes as a
//!     fresh contiguous run; cross-bond dedup is deliberately not
//!     attempted (deferred per Phase 2.1 scope).
//!   - bonds: `bond_count_u32` × 14-byte CompactBond.
//!
//! # CompactBond layout (14 bytes, fixed — Phase 2.1)
//!
//! ```text
//!   u32 subject_id          // block-local NodeId into subject_table
//!   u32 object_id           // block-local NodeId into subject_table
//!   u8  predicate           // v4: full u8 BondType discriminant (v3 legacy: 5-bit BondType + 1-bit dir + 2-bit flags)
//!   u8  weight              // u8-quantized weight (0..=255 maps to 0..=1.0)
//!   u16 keyframe_offset     // start index into per-block keyframe_pool
//!   u8  keyframe_count      // number of consecutive keyframes (0..=255)
//!   u8  confidence          // Confidence as u8 (see src/node.rs)
//! ```
//!
//! Phase 2's 13-byte layout (single `u16 keyframe_idx` per bond) is
//! retired; the reader refuses SSTs whose header version is 1.
//!
//! # Block-local vs brain-wide NodeIds — why both
//!
//! Phase 1 introduced a brain-wide `NodeIntern` (Seed ↔ u32). SST
//! blocks deliberately do NOT use it directly: each block carries its
//! own subject table mapping block-local u32 → Seed. Rationale:
//!
//! - **Decoupled lifetime.** An SST can be read by a brain that has
//!   never seen its seeds. A merge-compaction in Phase 6 may stream
//!   bonds across SSTs whose intern tables differ. Block-local IDs
//!   keep the block self-describing.
//! - **Compression.** zstd compresses block-local indices much harder
//!   than brain-wide IDs because the local range is small (often
//!   <256 distinct nodes per block) — long runs of small varints.
//! - **Migration safety.** Brain-wide NodeIds rebuild every load; a
//!   stable on-disk format cannot depend on them.
//!
//! At read time the SST hands the caller a `BondView` carrying full
//! `Seed` values. Optional re-interning through the brain-wide table
//! is the caller's choice in Phase 5.
//!
//! # Concurrency
//!
//! `SST` is `Send + Sync`. The block cache uses `Mutex` for its map
//! (matches `src/compress.rs::BlockCache`). Reads never block writes
//! because Phase 2 has no concurrent writes: the file is finalised
//! before the reader opens it.

use std::fs::File;
use std::path::Path;
use std::sync::{Arc, Mutex};

use lru::LruCache;
use memmap2::Mmap;

use crate::bond::{BondType, TemporalKeyframe};
use crate::brain::BrainError;
use crate::encrypt::{xor_in_place, MasterKey};
use crate::node::Confidence;
use crate::seed::Seed;

// ---------------------------------------------------------------------
// On-disk constants
// ---------------------------------------------------------------------

pub const SST_MAGIC: u32 = 0x4C534F34; // "LSO4"
pub const SST_FOOTER_MAGIC: u32 = 0x34534F4C; // "4SOL"

/// Current on-disk SST format version. **Phase 10 bump** (was v2 in
/// Phase 2.1, v1 in Phase 2). v3 adds a per-block KnowledgeNode
/// table parallel to the existing subject table -- one node record
/// per subject_table slot (zero-length payload means "no node
/// payload at this slot", which is legal for objects that were
/// referenced but never had a node side-channel). The reader refuses
/// other versions with a clear error; the migration tool re-runs
/// the v3 -> v4 conversion to produce v3-format SSTs.
pub const SST_VERSION: u32 = 4; // OP-EXTRACT-BREADTH C: full-u8 predicate
pub const SST_VERSION_LEGACY_5BIT: u32 = 3; // legacy 5-bit predicate SSTs (still read)

/// Fixed header length. Must match `SSTHeader::write_to` / `read_from`.
pub const SST_HEADER_BYTES: usize = 132;

/// Block-index entry length on disk.
pub const SST_INDEX_ENTRY_BYTES: usize = 32 + 8 + 4 + 4; // 48 bytes

/// Footer length (CRC32 + magic).
pub const SST_FOOTER_BYTES: usize = 8;

/// Per-block frame header: compressed_len u32 + decompressed_len u32.
pub const SST_BLOCK_FRAME_HEADER_BYTES: usize = 8;

/// Block payload header (within the decompressed payload).
pub const BLOCK_PAYLOAD_HEADER_BYTES: usize = 24;

/// CompactBond fixed on-disk size (Phase 2.1: 14 bytes, was 13 in Phase 2).
pub const COMPACT_BOND_BYTES: usize = 14;

/// Default block-cache cap (bytes of decompressed payload held resident).
pub const DEFAULT_SST_BLOCK_CACHE_BYTES: usize = 256 * 1024 * 1024;

/// Default parsed-block cache cap (bytes, accounted heuristically).
///
/// Phase 6 addition. The pre-Phase-6 read path called `parse_block`
/// on every `get_bonds_for_subject`, which became the dominant cost
/// at V4Only p50 (311 us on a 3-L0-SST + 10K-MemTable scenario per
/// the Phase 5 bench). Caching the parsed form drops that to a
/// single subject-table linear scan inside the cached block.
pub const DEFAULT_PARSED_BLOCK_CACHE_BYTES: usize = 512 * 1024 * 1024;

/// Tombstone marker stored in the CompactBond `confidence` byte.
///
/// Phase 6 addition. A bond entry whose `confidence == TOMBSTONE_CONFIDENCE`
/// is a deletion -- it shadows any older same-(subject, predicate,
/// object) entry seen later in the merge order. The marker lives in
/// the on-disk byte rather than a separate field because the byte was
/// already only using values 0..=4.
pub const TOMBSTONE_CONFIDENCE: u8 = 0xFF;

/// True iff the on-disk confidence byte signals a tombstone.
#[inline]
pub fn is_tombstone_byte(b: u8) -> bool {
    b == TOMBSTONE_CONFIDENCE
}

/// Hard cap on the per-block keyframe pool. `keyframe_offset` is a
/// `u16` so the absolute pool size cannot exceed `u16::MAX + 1`; we
/// cap at `u16::MAX` so an offset of `u16::MAX - 1` plus a count of 1
/// remains valid. The writer flushes the current block before adding
/// keyframes that would breach this cap.
pub const MAX_KEYFRAMES_PER_BLOCK: usize = u16::MAX as usize;

// ---------------------------------------------------------------------
// Header
// ---------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub struct SSTHeader {
    pub magic: u32,
    pub version: u32,
    pub level: u8,
    pub bond_count: u64,
    pub block_count: u64,
    pub manifest_id: u64,
    pub index_offset: u64,
    pub index_length: u64,
    pub encryption_iv: [u8; 16],
    pub subject_min_seed: Seed,
    pub subject_max_seed: Seed,
}

impl SSTHeader {
    /// Serialize into the fixed-size header buffer.
    pub fn write_to(&self, buf: &mut [u8]) {
        assert!(buf.len() >= SST_HEADER_BYTES);
        let mut o = 0;
        buf[o..o + 4].copy_from_slice(&self.magic.to_le_bytes()); o += 4;
        buf[o..o + 4].copy_from_slice(&self.version.to_le_bytes()); o += 4;
        buf[o] = self.level; o += 1;
        buf[o..o + 3].copy_from_slice(&[0u8; 3]); o += 3; // reserved
        buf[o..o + 8].copy_from_slice(&self.bond_count.to_le_bytes()); o += 8;
        buf[o..o + 8].copy_from_slice(&self.block_count.to_le_bytes()); o += 8;
        buf[o..o + 8].copy_from_slice(&self.manifest_id.to_le_bytes()); o += 8;
        buf[o..o + 8].copy_from_slice(&self.index_offset.to_le_bytes()); o += 8;
        buf[o..o + 8].copy_from_slice(&self.index_length.to_le_bytes()); o += 8;
        buf[o..o + 16].copy_from_slice(&self.encryption_iv); o += 16;
        buf[o..o + 32].copy_from_slice(self.subject_min_seed.as_bytes()); o += 32;
        buf[o..o + 32].copy_from_slice(self.subject_max_seed.as_bytes()); o += 32;
        debug_assert_eq!(o, SST_HEADER_BYTES);
    }

    pub fn read_from(buf: &[u8]) -> Result<Self, BrainError> {
        if buf.len() < SST_HEADER_BYTES {
            return Err(BrainError::Format("SST header truncated".to_string()));
        }
        let mut o = 0;
        let magic = u32::from_le_bytes(buf[o..o + 4].try_into().unwrap()); o += 4;
        if magic != SST_MAGIC {
            return Err(BrainError::Format(format!(
                "SST magic mismatch: got 0x{:08x}, expected 0x{:08x}",
                magic, SST_MAGIC
            )));
        }
        let version = u32::from_le_bytes(buf[o..o + 4].try_into().unwrap()); o += 4;
        if version != SST_VERSION && version != SST_VERSION_LEGACY_5BIT {
            return Err(BrainError::Format(format!(
                "unsupported SST version: got {}, expected {} (Phase 10 added the per-block node table; \
                 v1 and v2 SSTs are not readable, re-run the v3->v4 migration tool to produce v3 SSTs)",
                version, SST_VERSION
            )));
        }
        let level = buf[o]; o += 1;
        o += 3; // reserved
        let bond_count = u64::from_le_bytes(buf[o..o + 8].try_into().unwrap()); o += 8;
        let block_count = u64::from_le_bytes(buf[o..o + 8].try_into().unwrap()); o += 8;
        let manifest_id = u64::from_le_bytes(buf[o..o + 8].try_into().unwrap()); o += 8;
        let index_offset = u64::from_le_bytes(buf[o..o + 8].try_into().unwrap()); o += 8;
        let index_length = u64::from_le_bytes(buf[o..o + 8].try_into().unwrap()); o += 8;
        let mut encryption_iv = [0u8; 16];
        encryption_iv.copy_from_slice(&buf[o..o + 16]); o += 16;
        let mut smin = [0u8; 32];
        smin.copy_from_slice(&buf[o..o + 32]); o += 32;
        let mut smax = [0u8; 32];
        smax.copy_from_slice(&buf[o..o + 32]); o += 32;
        debug_assert_eq!(o, SST_HEADER_BYTES);
        Ok(SSTHeader {
            magic,
            version,
            level,
            bond_count,
            block_count,
            manifest_id,
            index_offset,
            index_length,
            encryption_iv,
            subject_min_seed: Seed::from_bytes(smin),
            subject_max_seed: Seed::from_bytes(smax),
        })
    }
}

// ---------------------------------------------------------------------
// Block index
// ---------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub struct BlockIndexEntry {
    pub subject_min_seed: Seed,
    pub offset: u64,
    pub compressed_length: u32,
    pub decompressed_length: u32,
}

impl BlockIndexEntry {
    pub fn write_to(&self, buf: &mut [u8]) {
        assert!(buf.len() >= SST_INDEX_ENTRY_BYTES);
        buf[0..32].copy_from_slice(self.subject_min_seed.as_bytes());
        buf[32..40].copy_from_slice(&self.offset.to_le_bytes());
        buf[40..44].copy_from_slice(&self.compressed_length.to_le_bytes());
        buf[44..48].copy_from_slice(&self.decompressed_length.to_le_bytes());
    }

    pub fn read_from(buf: &[u8]) -> Result<Self, BrainError> {
        if buf.len() < SST_INDEX_ENTRY_BYTES {
            return Err(BrainError::Format("SST index entry truncated".to_string()));
        }
        let mut s = [0u8; 32];
        s.copy_from_slice(&buf[0..32]);
        Ok(BlockIndexEntry {
            subject_min_seed: Seed::from_bytes(s),
            offset: u64::from_le_bytes(buf[32..40].try_into().unwrap()),
            compressed_length: u32::from_le_bytes(buf[40..44].try_into().unwrap()),
            decompressed_length: u32::from_le_bytes(buf[44..48].try_into().unwrap()),
        })
    }
}

// ---------------------------------------------------------------------
// Predicate packing
// ---------------------------------------------------------------------

/// Pack a `BondType` + direction + 2 user flag bits into a single byte.
///
/// v4 layout: the full byte is the BondType discriminant (0..=255). The v3
/// legacy layout was [flags_hi:2 | direction:1 | bondtype:5], still decoded
/// for existing v3 SSTs via unpack_predicate_v3.
///
/// 29 BondType variants currently exist; the 5-bit field accommodates
/// up to 32. The single direction bit reserves room for future
/// inverted-edge encoding (Phase 6 compaction may want it). The two
/// `flag` bits are currently reserved for tombstone / synthesis-origin
/// markers — Phase 2 always writes them as zero, but the reader
/// surfaces them on the `BondView` for forward compatibility.
pub fn pack_predicate(bond_type: BondType, direction: bool, flags: u8) -> u8 {
    // OP-EXTRACT-BREADTH C: v4 stores the FULL u8 BondType discriminant (0..=255).
    // direction/flags were reserved-and-always-zero since Phase 2 and are no
    // longer stored inline; assert callers still pass zero so a future non-zero
    // use fails loud instead of being silently dropped.
    debug_assert!(!direction && flags == 0, "v4 predicate stores bond_type only; dir/flags must be zero");
    bond_type_to_u8(bond_type)
}

/// v4 full-u8 predicate decode: the whole byte is the BondType discriminant.
pub fn unpack_predicate(byte: u8) -> (BondType, bool, u8) {
    (u8_to_bond_type(byte), false, 0)
}

/// v3 legacy decode: bond_type in low 5 bits, reserved dir/flags in high bits.
/// Used only for SSTs stamped SST_VERSION_LEGACY_5BIT.
pub fn unpack_predicate_v3(byte: u8) -> (BondType, bool, u8) {
    let bt = byte & 0x1F;
    let dir = (byte >> 5) & 0b1 != 0;
    let flags = (byte >> 6) & 0b11;
    (u8_to_bond_type(bt), dir, flags)
}

/// Decode a predicate byte using the owning SST format version.
pub fn unpack_predicate_versioned(byte: u8, version: u32) -> (BondType, bool, u8) {
    if version >= SST_VERSION { unpack_predicate(byte) } else { unpack_predicate_v3(byte) }
}

fn bond_type_to_u8(b: BondType) -> u8 {
    // Discriminant order matches src/bond.rs (BR.1 + BR.10b stability
    // contract: append-only).
    match b {
        BondType::IsA => 0,
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
        BondType::DefinedAs => 37,
        BondType::WorksBy => 38,
        BondType::ComposedOf => 39,
        BondType::PurposeOf => 40,
        BondType::DiffersFrom => 41,
        BondType::LeadsTo => 42,
        BondType::ExampleOf => 43,
        BondType::HasCurrency => 44,
        BondType::AuthoredBy => 45,
    }
}

fn u8_to_bond_type(b: u8) -> BondType {
    match b {
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
        // Unknown discriminant — surface honestly as RelatedTo and let
        // upstream callers see the flags+raw byte path if needed. Phase
        // 2 fixtures never write unknown discriminants.
        _ => BondType::RelatedTo,
    }
}

// ---------------------------------------------------------------------
// Weight quantization (u8 ↔ f32)
// ---------------------------------------------------------------------

/// Quantize an f32 weight in `[0.0, 1.0]` to a `u8` (0..=255). Values
/// outside the range are clamped; NaN maps to 0.
pub fn quantize_weight(w: f32) -> u8 {
    if !w.is_finite() {
        return 0;
    }
    let clamped = w.max(0.0).min(1.0);
    (clamped * 255.0).round() as u8
}

/// Inverse of `quantize_weight`.
pub fn dequantize_weight(b: u8) -> f32 {
    (b as f32) / 255.0
}

// ---------------------------------------------------------------------
// BondView — what the reader hands back
// ---------------------------------------------------------------------

/// A read-only view of a single bond from an SST block.
///
/// Holds the decompressed block `Arc` alive (`_block` field) so the
/// underlying buffer cannot be freed while the view is in scope.
/// Designed to be cheap to clone (Arc + small POD).
///
/// **Phase 2.1**: `keyframes` is the full history slice for this bond
/// (`pool[keyframe_offset .. keyframe_offset + keyframe_count]`), no
/// longer a single `Option<TemporalKeyframe>`. Empty `Vec` means the
/// bond carried no keyframes — unusual but legal.
#[derive(Clone)]
pub struct BondView<'a> {
    pub subject: Seed,
    pub object: Seed,
    pub bond_type: BondType,
    pub direction: bool,
    pub flags: u8,
    pub weight: f32,
    pub confidence: Confidence,
    /// Phase 6: the raw on-disk confidence byte. `TOMBSTONE_CONFIDENCE`
    /// (`0xFF`) marks a tombstone row that does not map to a real
    /// Confidence band -- the dedup/merge layer above this needs to
    /// see the raw byte so it can drop the bond entry without
    /// re-deriving the band from the (meaningless) fallback.
    pub raw_confidence: u8,
    pub keyframes: Vec<TemporalKeyframe>,
    /// Phase 10: subject side's KnowledgeNode payload if the block
    /// carries one for this Seed. `None` is legal -- not every SST
    /// is the canonical home for every Seed's node metadata.
    pub subject_node_record: Option<crate::node::KnowledgeNode>,
    /// Phase 10: object side's KnowledgeNode payload (same caveat).
    pub object_node_record: Option<crate::node::KnowledgeNode>,
    // Pins the block buffer alive. Phantom on the lifetime so callers
    // see the borrow contract (`'a` ties to the SST handle).
    _block: Arc<Vec<u8>>,
    _phantom: std::marker::PhantomData<&'a ()>,
}

impl<'a> BondView<'a> {
    /// True iff this BondView is a tombstone row (Phase 6).
    pub fn is_tombstone(&self) -> bool {
        is_tombstone_byte(self.raw_confidence)
    }

    /// Phase 10: subject-side KnowledgeNode payload from the same
    /// block, if the block carried one.
    pub fn subject_node(&self) -> Option<&crate::node::KnowledgeNode> {
        self.subject_node_record.as_ref()
    }

    /// Phase 10: object-side KnowledgeNode payload from the same
    /// block, if the block carried one.
    pub fn object_node(&self) -> Option<&crate::node::KnowledgeNode> {
        self.object_node_record.as_ref()
    }
}

impl<'a> BondView<'a> {
    /// Iterator over this bond's keyframe history in pool order
    /// (oldest → newest if the writer pushed them that way).
    pub fn keyframes_iter(&self) -> impl Iterator<Item = &TemporalKeyframe> {
        self.keyframes.iter()
    }

    /// Convenience: the most-recent (highest-timestamp) keyframe, or
    /// `None` if this bond has no keyframes. Replaces the Phase 2
    /// `keyframe: Option<TemporalKeyframe>` field's role.
    pub fn latest_keyframe(&self) -> Option<&TemporalKeyframe> {
        self.keyframes.iter().max_by_key(|k| k.timestamp)
    }
}

impl<'a> std::fmt::Debug for BondView<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BondView")
            .field("subject", &self.subject)
            .field("object", &self.object)
            .field("bond_type", &self.bond_type)
            .field("weight", &self.weight)
            .field("confidence", &self.confidence)
            .field("keyframe_count", &self.keyframes.len())
            .finish()
    }
}

// ---------------------------------------------------------------------
// BlockCache — bounded LRU of decompressed block payloads
// ---------------------------------------------------------------------

/// Bounded LRU of decompressed block payloads. Public so callers can
/// share a single cache across multiple SSTs (Phase 6 compaction will).
///
/// Mirrors `src/compress.rs::BlockCache` semantics: insert evicts LRU
/// entries until `resident_bytes <= cap_bytes`. Always leaves at least
/// one entry resident (so a single oversized block doesn't infinite-evict).
pub struct BlockCache {
    inner: Mutex<BlockCacheInner>,
}

struct BlockCacheInner {
    map: LruCache<BlockCacheKey, Arc<Vec<u8>>>,
    cap_bytes: usize,
    resident_bytes: usize,
    hits: u64,
    misses: u64,
}

/// `(sst_uid, block_id)` — sst_uid is the manifest_id of the owning
/// SST, which is unique across all SSTs in a brain.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct BlockCacheKey {
    pub sst_uid: u64,
    pub block_id: u32,
}

impl BlockCache {
    pub fn new(cap_bytes: usize) -> Self {
        BlockCache {
            inner: Mutex::new(BlockCacheInner {
                map: LruCache::unbounded(),
                cap_bytes,
                resident_bytes: 0,
                hits: 0,
                misses: 0,
            }),
        }
    }

    pub fn cap_bytes(&self) -> usize {
        self.inner.lock().unwrap().cap_bytes
    }

    pub fn resident_bytes(&self) -> usize {
        self.inner.lock().unwrap().resident_bytes
    }

    pub fn resident_count(&self) -> usize {
        self.inner.lock().unwrap().map.len()
    }

    pub fn hit_rate(&self) -> f64 {
        let g = self.inner.lock().unwrap();
        let total = g.hits + g.misses;
        if total == 0 { 0.0 } else { (g.hits as f64) / (total as f64) }
    }

    pub fn hit_miss_counts(&self) -> (u64, u64) {
        let g = self.inner.lock().unwrap();
        (g.hits, g.misses)
    }

    fn get(&self, key: BlockCacheKey) -> Option<Arc<Vec<u8>>> {
        let mut g = self.inner.lock().unwrap();
        let hit = g.map.get(&key).cloned();
        if hit.is_some() { g.hits += 1; } else { g.misses += 1; }
        hit
    }

    fn insert(&self, key: BlockCacheKey, buf: Arc<Vec<u8>>) -> Arc<Vec<u8>> {
        let mut g = self.inner.lock().unwrap();
        let sz = buf.len();
        if let Some(prev) = g.map.put(key, buf.clone()) {
            g.resident_bytes = g.resident_bytes.saturating_sub(prev.len());
        }
        g.resident_bytes = g.resident_bytes.saturating_add(sz);
        while g.resident_bytes > g.cap_bytes && g.map.len() > 1 {
            if let Some((_, evicted)) = g.map.pop_lru() {
                g.resident_bytes = g.resident_bytes.saturating_sub(evicted.len());
            } else {
                break;
            }
        }
        buf
    }
}

// ---------------------------------------------------------------------
// ParsedBlockCache  (Phase 6) -- LRU of materialised ParsedBlock
// ---------------------------------------------------------------------

/// Bounded LRU of `Arc<ParsedBlock>` keyed by `(sst_uid, block_id)`.
///
/// Phase 6 addition. Distinct from `BlockCache` (which holds the
/// decompressed-but-unparsed payload bytes): the parse step itself
/// is the dominant cost at V4Only p50 because it walks the per-
/// block subject table, keyframe pool, and bond array allocating
/// `Vec`s. Caching the parsed form keeps both costs amortised.
///
/// Approximate byte accounting per entry:
///   subject_table.len() * 32  // Seed bytes
/// + keyframe_pool.len()  * 56  // TemporalKeyframe stack approx
/// + bonds.len()          * 14  // CompactBondRaw
/// + 128                       // Vec headers + Arc overhead
///
/// Eviction: insert evicts LRU entries until `resident_bytes <=
/// cap_bytes`, always leaving at least one entry resident so a
/// single oversized block does not infinite-evict.
pub struct ParsedBlockCache {
    inner: Mutex<ParsedBlockCacheInner>,
}

struct ParsedBlockCacheInner {
    map: LruCache<BlockCacheKey, Arc<ParsedBlock>>,
    cap_bytes: usize,
    resident_bytes: usize,
    hits: u64,
    misses: u64,
    parses: u64,
}

impl ParsedBlockCache {
    pub fn new(cap_bytes: usize) -> Self {
        ParsedBlockCache {
            inner: Mutex::new(ParsedBlockCacheInner {
                map: LruCache::unbounded(),
                cap_bytes,
                resident_bytes: 0,
                hits: 0,
                misses: 0,
                parses: 0,
            }),
        }
    }

    pub fn cap_bytes(&self) -> usize {
        self.inner.lock().unwrap().cap_bytes
    }

    pub fn resident_bytes(&self) -> usize {
        self.inner.lock().unwrap().resident_bytes
    }

    pub fn resident_count(&self) -> usize {
        self.inner.lock().unwrap().map.len()
    }

    pub fn hit_miss_counts(&self) -> (u64, u64) {
        let g = self.inner.lock().unwrap();
        (g.hits, g.misses)
    }

    /// Diagnostic: how many times `parse_block` was actually invoked
    /// vs served from the cache. Used by Phase 6 tests that assert
    /// parse-once semantics under repeated reads of the same subject.
    pub fn parse_count(&self) -> u64 {
        self.inner.lock().unwrap().parses
    }

    /// Evict every entry for the given `sst_uid`. Called by Phase 6
    /// compaction after the manifest atomic swap removes an SST, so
    /// the parsed-block memory drops back when the underlying file
    /// is no longer referenced.
    pub fn invalidate_sst(&self, sst_uid: u64) {
        let mut g = self.inner.lock().unwrap();
        let mut victims: Vec<BlockCacheKey> = Vec::new();
        for (k, _) in g.map.iter() {
            if k.sst_uid == sst_uid {
                victims.push(*k);
            }
        }
        for k in victims {
            if let Some(v) = g.map.pop(&k) {
                g.resident_bytes = g.resident_bytes
                    .saturating_sub(approx_parsed_block_bytes(&v));
            }
        }
    }

    fn get(&self, key: BlockCacheKey) -> Option<Arc<ParsedBlock>> {
        let mut g = self.inner.lock().unwrap();
        let hit = g.map.get(&key).cloned();
        if hit.is_some() { g.hits += 1; } else { g.misses += 1; }
        hit
    }

    fn insert(&self, key: BlockCacheKey, parsed: Arc<ParsedBlock>) -> Arc<ParsedBlock> {
        let mut g = self.inner.lock().unwrap();
        let sz = approx_parsed_block_bytes(&parsed);
        if let Some(prev) = g.map.put(key, parsed.clone()) {
            g.resident_bytes = g.resident_bytes
                .saturating_sub(approx_parsed_block_bytes(&prev));
        }
        g.resident_bytes = g.resident_bytes.saturating_add(sz);
        while g.resident_bytes > g.cap_bytes && g.map.len() > 1 {
            if let Some((_, evicted)) = g.map.pop_lru() {
                g.resident_bytes = g.resident_bytes
                    .saturating_sub(approx_parsed_block_bytes(&evicted));
            } else {
                break;
            }
        }
        parsed
    }

    fn bump_parses(&self) {
        let mut g = self.inner.lock().unwrap();
        g.parses += 1;
    }
}

/// Approximate byte cost of one parsed block in the cache. Heuristic;
/// the cap is a soft target.
fn approx_parsed_block_bytes(p: &ParsedBlock) -> usize {
    // Stack approx for the three Vec headers + Arc overhead.
    const BASE: usize = 128;
    // Seed is 32 bytes.
    const PER_SEED: usize = 32;
    // TemporalKeyframe ~= i64 + f32 + KeyframeSource (max 16 B variant + tag).
    const PER_KEYFRAME: usize = 56;
    const PER_BOND: usize = COMPACT_BOND_BYTES + 4; // alignment fudge
    BASE
        + p.subject_table.len() * PER_SEED
        + p.keyframe_pool.len() * PER_KEYFRAME
        + p.bonds.len() * PER_BOND
}

// ---------------------------------------------------------------------
// SST reader
// ---------------------------------------------------------------------

/// An opened SST. Holds the mmap alive so reads are zero-copy until
/// the per-block decompress step; decompressed blocks live in the
/// shared `BlockCache`.
pub struct SST {
    mmap: Arc<Mmap>,
    header: SSTHeader,
    block_index: Vec<BlockIndexEntry>,
    block_cache: Arc<BlockCache>,
    /// Phase 6 parsed-block cache. Optional so existing callers
    /// that constructed an `SST` pre-Phase-6 work unchanged; the
    /// canonical read path installs a shared cache via
    /// `attach_parsed_cache` after open.
    parsed_cache: Option<Arc<ParsedBlockCache>>,
    encryption_key: Option<MasterKey>,
}

impl SST {
    pub fn open(
        path: &Path,
        key: Option<MasterKey>,
        block_cache: Arc<BlockCache>,
    ) -> Result<SST, BrainError> {
        let file = File::open(path).map_err(|e| {
            BrainError::Io(std::io::Error::new(std::io::ErrorKind::Other, format!("open SST {}: {}", path.display(), e)))
        })?;
        let metadata = file.metadata().map_err(|e| {
            BrainError::Io(std::io::Error::new(std::io::ErrorKind::Other, format!("stat SST {}: {}", path.display(), e)))
        })?;
        let file_len = metadata.len();
        if file_len < (SST_HEADER_BYTES + SST_FOOTER_BYTES) as u64 {
            return Err(BrainError::Format(format!(
                "SST {} too small ({} bytes)",
                path.display(), file_len
            )));
        }
        // SAFETY: file is open read-only; the OS guarantees the
        // mapping stays valid for the file's lifetime. We hold the
        // Mmap inside the SST struct.
        let mmap = unsafe { Mmap::map(&file) }.map_err(|e| {
            BrainError::Io(std::io::Error::new(std::io::ErrorKind::Other, format!("mmap SST {}: {}", path.display(), e)))
        })?;
        let mmap = Arc::new(mmap);

        // Header.
        let header = SSTHeader::read_from(&mmap[..SST_HEADER_BYTES])?;

        // Footer (CRC32 + magic).
        let footer_off = file_len as usize - SST_FOOTER_BYTES;
        let footer = &mmap[footer_off..footer_off + SST_FOOTER_BYTES];
        let crc_recorded = u32::from_le_bytes(footer[0..4].try_into().unwrap());
        let magic_footer = u32::from_le_bytes(footer[4..8].try_into().unwrap());
        if magic_footer != SST_FOOTER_MAGIC {
            return Err(BrainError::Format(format!(
                "SST footer magic mismatch: got 0x{:08x}, expected 0x{:08x}",
                magic_footer, SST_FOOTER_MAGIC
            )));
        }

        // Block index.
        let idx_off = header.index_offset as usize;
        let idx_len = header.index_length as usize;
        if idx_off + idx_len > file_len as usize - SST_FOOTER_BYTES {
            return Err(BrainError::Format(format!(
                "SST index range out of bounds: {}+{} > {}",
                idx_off, idx_len, file_len as usize - SST_FOOTER_BYTES
            )));
        }
        let idx_bytes = &mmap[idx_off..idx_off + idx_len];

        // Validate CRC32.
        let crc_actual = crc32fast::hash(idx_bytes);
        if crc_actual != crc_recorded {
            return Err(BrainError::Format(format!(
                "SST block-index CRC mismatch: recorded 0x{:08x}, actual 0x{:08x}",
                crc_recorded, crc_actual
            )));
        }

        if idx_len != header.block_count as usize * SST_INDEX_ENTRY_BYTES {
            return Err(BrainError::Format(format!(
                "SST index length {} doesn't match block_count {} × {}",
                idx_len, header.block_count, SST_INDEX_ENTRY_BYTES
            )));
        }
        let mut block_index = Vec::with_capacity(header.block_count as usize);
        for i in 0..header.block_count as usize {
            let base = i * SST_INDEX_ENTRY_BYTES;
            block_index.push(BlockIndexEntry::read_from(
                &idx_bytes[base..base + SST_INDEX_ENTRY_BYTES],
            )?);
        }
        // Block index must be sorted by subject_min_seed for binary search.
        for w in block_index.windows(2) {
            if w[0].subject_min_seed.as_bytes() > w[1].subject_min_seed.as_bytes() {
                return Err(BrainError::Format(
                    "SST block index not sorted by subject_min_seed".to_string(),
                ));
            }
        }

        Ok(SST {
            mmap,
            header,
            block_index,
            block_cache,
            parsed_cache: None,
            encryption_key: key,
        })
    }

    pub fn header(&self) -> &SSTHeader {
        &self.header
    }

    pub fn block_count(&self) -> usize {
        self.block_index.len()
    }

    pub fn block_index(&self) -> &[BlockIndexEntry] {
        &self.block_index
    }

    pub fn is_encrypted(&self) -> bool {
        self.encryption_key.is_some()
    }

    pub fn block_cache(&self) -> &Arc<BlockCache> {
        &self.block_cache
    }

    /// Attach (or replace) the Phase 6 parsed-block cache. Cheap;
    /// expected to be called once per SST::open by the canonical
    /// read path so the cache is shared across SSTs.
    pub fn attach_parsed_cache(&mut self, cache: Arc<ParsedBlockCache>) {
        self.parsed_cache = Some(cache);
    }

    pub fn parsed_cache(&self) -> Option<&Arc<ParsedBlockCache>> {
        self.parsed_cache.as_ref()
    }

    /// Materialise the parsed form of `block_id`. Consults the
    /// parsed-block cache (if attached) first; on miss falls
    /// through to `decompress_block` + `parse_block` and installs
    /// the result. Phase 6 hot path.
    pub fn get_or_parse_block(&self, block_id: u32) -> Result<Arc<ParsedBlock>, BrainError> {
        let key = BlockCacheKey { sst_uid: self.header.manifest_id, block_id };
        if let Some(pc) = &self.parsed_cache {
            if let Some(hit) = pc.get(key) {
                return Ok(hit);
            }
        }
        let raw = self.decompress_block(block_id)?;
        let parsed = parse_block(&raw)?;
        let arc = Arc::new(parsed);
        if let Some(pc) = &self.parsed_cache {
            pc.bump_parses();
            return Ok(pc.insert(key, arc));
        }
        Ok(arc)
    }

    /// Decompress (and decrypt, if keyed) block `block_id`. Cached.
    pub fn decompress_block(&self, block_id: u32) -> Result<Arc<Vec<u8>>, BrainError> {
        let key = BlockCacheKey { sst_uid: self.header.manifest_id, block_id };
        if let Some(hit) = self.block_cache.get(key) {
            return Ok(hit);
        }
        let entry = self.block_index.get(block_id as usize).ok_or_else(|| {
            BrainError::Format(format!("SST block_id {} out of range", block_id))
        })?;

        let frame_off = entry.offset as usize;
        let frame_end = frame_off + SST_BLOCK_FRAME_HEADER_BYTES
            + entry.compressed_length as usize;
        if frame_end > self.mmap.len() {
            return Err(BrainError::Format(format!(
                "SST block {} frame extends past file end ({} > {})",
                block_id, frame_end, self.mmap.len()
            )));
        }
        let frame = &self.mmap[frame_off..frame_end];
        let comp_len = u32::from_le_bytes(frame[0..4].try_into().unwrap()) as usize;
        let decomp_len = u32::from_le_bytes(frame[4..8].try_into().unwrap()) as usize;
        if comp_len != entry.compressed_length as usize {
            return Err(BrainError::Format(format!(
                "SST block {} compressed_length mismatch: frame {} vs index {}",
                block_id, comp_len, entry.compressed_length
            )));
        }
        if decomp_len != entry.decompressed_length as usize {
            return Err(BrainError::Format(format!(
                "SST block {} decompressed_length mismatch: frame {} vs index {}",
                block_id, decomp_len, entry.decompressed_length
            )));
        }
        let comp_payload = &frame[SST_BLOCK_FRAME_HEADER_BYTES..];

        // Step 1: decompress (zstd).
        let mut decomp = vec![0u8; decomp_len];
        let written = zstd::bulk::decompress_to_buffer(comp_payload, &mut decomp)
            .map_err(|e| BrainError::Io(std::io::Error::new(std::io::ErrorKind::Other, format!(
                "SST block {} zstd decompress: {}", block_id, e
            ))))?;
        if written != decomp_len {
            return Err(BrainError::Format(format!(
                "SST block {} zstd decompressed {} bytes, expected {}",
                block_id, written, decomp_len
            )));
        }

        // Step 2: decrypt (XOR-keystream) if keyed.
        if let Some(k) = &self.encryption_key {
            let block_idx = derive_block_keystream_index(
                self.header.manifest_id, block_id, &self.header.encryption_iv,
            );
            xor_in_place(k, block_idx, &mut decomp);
        }

        let arc = Arc::new(decomp);
        Ok(self.block_cache.insert(key, arc))
    }

    /// Binary search the block index for the (at most one) block whose
    /// subject range covers `subject`. Returns None if `subject` is
    /// strictly less than the first block's min.
    fn find_block_for_subject(&self, subject: &Seed) -> Option<u32> {
        if self.block_index.is_empty() {
            return None;
        }
        // partition_point finds first index where pred is false.
        // We want the last block whose subject_min_seed <= subject.
        let key = subject.as_bytes();
        let p = self.block_index.partition_point(|e| {
            e.subject_min_seed.as_bytes() <= key
        });
        if p == 0 {
            // subject < first block's min: not present.
            None
        } else {
            Some((p - 1) as u32)
        }
    }

    /// Return all bonds whose subject equals `subject`. Phase 6:
    /// consults the parsed-block cache; tombstone rows are emitted
    /// to callers (so layered merges can shadow older same-key
    /// bonds) and the caller drops them after dedup.
    pub fn get_bonds_for_subject(
        &self,
        subject: &Seed,
    ) -> Result<Vec<BondView<'_>>, BrainError> {
        let bid = match self.find_block_for_subject(subject) {
            Some(b) => b,
            None => return Ok(Vec::new()),
        };
        // Phase 6: prefer the parsed-block cache; the raw block
        // arc is still needed to keep BondView<'_> lifetimes
        // valid across the iteration so the caller does not have
        // to clone keyframes for every bond.
        let block = self.decompress_block(bid)?;
        let parsed = self.get_or_parse_block(bid)?;
        let mut out = Vec::new();
        for cb in &parsed.bonds {
            let s = parsed.subject_table.get(cb.subject_id as usize)
                .ok_or_else(|| BrainError::Format(format!(
                    "SST block {} bond subject_id {} out of range ({} entries)",
                    bid, cb.subject_id, parsed.subject_table.len()
                )))?;
            if s == subject {
                let o = parsed.subject_table.get(cb.object_id as usize)
                    .ok_or_else(|| BrainError::Format(format!(
                        "SST block {} bond object_id {} out of range",
                        bid, cb.object_id
                    )))?;
                let (bt, dir, flags) = unpack_predicate_versioned(cb.predicate, self.header().version);
                let kfs = slice_keyframes(
                    &parsed.keyframe_pool, cb.keyframe_offset, cb.keyframe_count,
                ).map_err(|e| BrainError::Format(format!(
                    "SST block {} bond keyframe range: {}", bid, e
                )))?;
                let subject_node_record = parsed.node_table
                    .get(cb.subject_id as usize)
                    .and_then(|n| n.clone());
                let object_node_record = parsed.node_table
                    .get(cb.object_id as usize)
                    .and_then(|n| n.clone());
                out.push(BondView {
                    subject: *s,
                    object: *o,
                    bond_type: bt,
                    direction: dir,
                    flags,
                    weight: dequantize_weight(cb.weight),
                    confidence: u8_to_confidence(cb.confidence),
                    raw_confidence: cb.confidence,
                    keyframes: kfs,
                    subject_node_record,
                    object_node_record,
                    _block: block.clone(),
                    _phantom: std::marker::PhantomData,
                });
            }
        }
        Ok(out)
    }

    /// Iterate every bond in the SST in block-then-disk order. Used by
    /// Phase 6 compaction. Holds each block alive while iterating it,
    /// drops it before moving on.
    pub fn iter_bonds(&self) -> SSTBondIter<'_> {
        SSTBondIter {
            sst: self,
            block_id: 0,
            current_block: None,
            cursor: 0,
        }
    }

    /// Phase 10: iterate every `(Seed, KnowledgeNode)` pair carried by
    /// any block in this SST. Used by the boot-time keyword index
    /// rebuild (so `lu4d_ask` finds subjects post-restart) and by the
    /// migration verify pass. Blocks where the slot is `None` are
    /// skipped (those Seeds were referenced as objects only).
    pub fn iter_nodes(&self) -> Result<Vec<(Seed, crate::node::KnowledgeNode)>, BrainError> {
        let mut out: Vec<(Seed, crate::node::KnowledgeNode)> = Vec::new();
        for block_id in 0..(self.block_index.len() as u32) {
            let parsed = self.get_or_parse_block(block_id)?;
            for (i, slot) in parsed.node_table.iter().enumerate() {
                if let Some(n) = slot {
                    let seed = match parsed.subject_table.get(i) {
                        Some(s) => *s,
                        None => continue,
                    };
                    out.push((seed, n.clone()));
                }
            }
        }
        Ok(out)
    }
}

unsafe impl Send for SST {}
unsafe impl Sync for SST {}

// ---------------------------------------------------------------------
// Iterator
// ---------------------------------------------------------------------

pub struct SSTBondIter<'a> {
    sst: &'a SST,
    block_id: u32,
    current_block: Option<(Arc<Vec<u8>>, ParsedBlock)>,
    cursor: usize,
}

impl<'a> Iterator for SSTBondIter<'a> {
    type Item = Result<BondView<'a>, BrainError>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some((block_arc, parsed)) = &self.current_block {
                if self.cursor < parsed.bonds.len() {
                    let cb = parsed.bonds[self.cursor];
                    self.cursor += 1;
                    let s = match parsed.subject_table.get(cb.subject_id as usize) {
                        Some(v) => *v,
                        None => return Some(Err(BrainError::Format(format!(
                            "iter bond subject_id {} oob", cb.subject_id
                        )))),
                    };
                    let o = match parsed.subject_table.get(cb.object_id as usize) {
                        Some(v) => *v,
                        None => return Some(Err(BrainError::Format(format!(
                            "iter bond object_id {} oob", cb.object_id
                        )))),
                    };
                    let (bt, dir, flags) = unpack_predicate_versioned(cb.predicate, self.sst.header().version);
                    let kfs = match slice_keyframes(
                        &parsed.keyframe_pool, cb.keyframe_offset, cb.keyframe_count,
                    ) {
                        Ok(v) => v,
                        Err(e) => return Some(Err(BrainError::Format(format!(
                            "iter bond keyframe range: {}", e
                        )))),
                    };
                    let subject_node_record = parsed.node_table
                        .get(cb.subject_id as usize)
                        .and_then(|n| n.clone());
                    let object_node_record = parsed.node_table
                        .get(cb.object_id as usize)
                        .and_then(|n| n.clone());
                    return Some(Ok(BondView {
                        subject: s,
                        object: o,
                        bond_type: bt,
                        direction: dir,
                        flags,
                        weight: dequantize_weight(cb.weight),
                        confidence: u8_to_confidence(cb.confidence),
                        raw_confidence: cb.confidence,
                        keyframes: kfs,
                        subject_node_record,
                        object_node_record,
                        _block: block_arc.clone(),
                        _phantom: std::marker::PhantomData,
                    }));
                }
                self.current_block = None;
                self.cursor = 0;
                self.block_id += 1;
            }
            if (self.block_id as usize) >= self.sst.block_index.len() {
                return None;
            }
            let block = match self.sst.decompress_block(self.block_id) {
                Ok(b) => b,
                Err(e) => return Some(Err(e)),
            };
            let parsed = match parse_block(&block) {
                Ok(p) => p,
                Err(e) => return Some(Err(e)),
            };
            self.current_block = Some((block, parsed));
            self.cursor = 0;
        }
    }
}

// ---------------------------------------------------------------------
// Block payload parsing
// ---------------------------------------------------------------------

/// Decoded fixed-size bond row from a block payload. **Phase 2.1**:
/// the single `keyframe_idx: u16` field is replaced by
/// `keyframe_offset: u16 + keyframe_count: u8`, addressing a
/// contiguous slice in the per-block keyframe pool.
#[derive(Clone, Copy, Debug)]
pub struct CompactBondRaw {
    pub subject_id: u32,
    pub object_id: u32,
    pub predicate: u8,
    pub weight: u8,
    pub keyframe_offset: u16,
    pub keyframe_count: u8,
    pub confidence: u8,
}

pub struct ParsedBlock {
    pub subject_table: Vec<Seed>,
    /// Phase 10: per-slot optional KnowledgeNode. `node_table[i]`
    /// pairs with `subject_table[i]`; a `None` slot means the SST
    /// holds the bond reference but no metadata payload (object-
    /// only reference -- the subject side's SST owns the metadata).
    pub node_table: Vec<Option<crate::node::KnowledgeNode>>,
    pub keyframe_pool: Vec<TemporalKeyframe>,
    pub bonds: Vec<CompactBondRaw>,
}

pub fn parse_block(buf: &[u8]) -> Result<ParsedBlock, BrainError> {
    if buf.len() < BLOCK_PAYLOAD_HEADER_BYTES {
        return Err(BrainError::Format("SST block payload header truncated".to_string()));
    }
    let _subject_min_idx = u32::from_le_bytes(buf[0..4].try_into().unwrap());
    let _subject_max_idx = u32::from_le_bytes(buf[4..8].try_into().unwrap());
    let bond_count = u32::from_le_bytes(buf[8..12].try_into().unwrap()) as usize;
    let keyframe_pool_size = u32::from_le_bytes(buf[12..16].try_into().unwrap()) as usize;
    let subject_table_size = u32::from_le_bytes(buf[16..20].try_into().unwrap()) as usize;
    // Phase 10: was `_reserved`; now node_table_size. Must equal
    // subject_table_size; otherwise the block is malformed.
    let node_table_size = u32::from_le_bytes(buf[20..24].try_into().unwrap()) as usize;
    if node_table_size != subject_table_size {
        return Err(BrainError::Format(format!(
            "SST v3 block: node_table_size {} != subject_table_size {}",
            node_table_size, subject_table_size
        )));
    }

    let mut cursor = BLOCK_PAYLOAD_HEADER_BYTES;

    // Subject table.
    let st_bytes = subject_table_size * 32;
    if cursor + st_bytes > buf.len() {
        return Err(BrainError::Format("SST subject_table truncated".to_string()));
    }
    let mut subject_table = Vec::with_capacity(subject_table_size);
    for _ in 0..subject_table_size {
        let mut s = [0u8; 32];
        s.copy_from_slice(&buf[cursor..cursor + 32]);
        subject_table.push(Seed::from_bytes(s));
        cursor += 32;
    }

    // Phase 10: Node table. One Option<KnowledgeNode> per
    // subject_table slot; zero-length payload means None.
    let mut node_table: Vec<Option<crate::node::KnowledgeNode>> =
        Vec::with_capacity(node_table_size);
    for _ in 0..node_table_size {
        if cursor + 4 > buf.len() {
            return Err(BrainError::Format("SST node_table length truncated".to_string()));
        }
        let l = u32::from_le_bytes(buf[cursor..cursor + 4].try_into().unwrap()) as usize;
        cursor += 4;
        if cursor + l > buf.len() {
            return Err(BrainError::Format("SST node_table entry truncated".to_string()));
        }
        if l == 0 {
            node_table.push(None);
        } else {
            match crate::node::deserialize_knowledge_node(&buf[cursor..cursor + l]) {
                Some(n) => node_table.push(Some(n)),
                None => return Err(BrainError::Format(
                    "SST node_table entry bincode decode failed".to_string()
                )),
            }
            cursor += l;
        }
    }

    // Keyframe pool: each entry is u32 length-prefix + bincode bytes.
    let mut keyframe_pool = Vec::with_capacity(keyframe_pool_size);
    for _ in 0..keyframe_pool_size {
        if cursor + 4 > buf.len() {
            return Err(BrainError::Format("SST keyframe_pool length truncated".to_string()));
        }
        let l = u32::from_le_bytes(buf[cursor..cursor + 4].try_into().unwrap()) as usize;
        cursor += 4;
        if cursor + l > buf.len() {
            return Err(BrainError::Format("SST keyframe_pool entry truncated".to_string()));
        }
        let kf: TemporalKeyframe = bincode::deserialize(&buf[cursor..cursor + l])
            .map_err(|e| BrainError::Format(format!(
                "SST keyframe bincode decode: {}", e
            )))?;
        keyframe_pool.push(kf);
        cursor += l;
    }

    // Bonds (Phase 2.1: 14 bytes — u32 + u32 + u8 + u8 + u16 + u8 + u8).
    if cursor + bond_count * COMPACT_BOND_BYTES > buf.len() {
        return Err(BrainError::Format("SST bonds section truncated".to_string()));
    }
    let mut bonds = Vec::with_capacity(bond_count);
    for _ in 0..bond_count {
        let subject_id      = u32::from_le_bytes(buf[cursor..cursor + 4].try_into().unwrap());
        let object_id       = u32::from_le_bytes(buf[cursor + 4..cursor + 8].try_into().unwrap());
        let predicate       = buf[cursor + 8];
        let weight          = buf[cursor + 9];
        let keyframe_offset = u16::from_le_bytes(buf[cursor + 10..cursor + 12].try_into().unwrap());
        let keyframe_count  = buf[cursor + 12];
        let confidence      = buf[cursor + 13];
        bonds.push(CompactBondRaw {
            subject_id, object_id, predicate, weight,
            keyframe_offset, keyframe_count, confidence,
        });
        cursor += COMPACT_BOND_BYTES;
    }

    Ok(ParsedBlock { subject_table, node_table, keyframe_pool, bonds })
}

/// Materialise a bond's keyframe slice from its per-block pool range.
/// Phase 2.1 helper shared by `get_bonds_for_subject` and `iter_bonds`.
pub fn slice_keyframes(
    pool: &[TemporalKeyframe],
    offset: u16,
    count: u8,
) -> Result<Vec<TemporalKeyframe>, String> {
    if count == 0 {
        return Ok(Vec::new());
    }
    let start = offset as usize;
    let end = start + count as usize;
    if end > pool.len() {
        return Err(format!(
            "keyframe slice {}..{} out of pool bounds (len {})",
            start, end, pool.len()
        ));
    }
    Ok(pool[start..end].to_vec())
}

// ---------------------------------------------------------------------
// Helpers shared with sst_writer
// ---------------------------------------------------------------------

/// Derive the keystream index used by `xor_in_place` for block `id` in
/// SST `manifest_id`. The IV bytes are mixed in via XOR so two SSTs
/// with the same manifest_id (theoretically impossible but defensively
/// handled) still produce distinct keystreams.
pub fn derive_block_keystream_index(
    manifest_id: u64,
    block_id: u32,
    iv: &[u8; 16],
) -> u64 {
    let mut iv_mix: u64 = 0;
    for chunk in iv.chunks(8) {
        let mut tmp = [0u8; 8];
        tmp[..chunk.len()].copy_from_slice(chunk);
        iv_mix ^= u64::from_le_bytes(tmp);
    }
    manifest_id
        .wrapping_mul(0x9E37_79B9_7F4A_7C15) // splitmix64 multiplier
        .wrapping_add(block_id as u64)
        ^ iv_mix
}

pub fn confidence_to_u8(c: Confidence) -> u8 {
    match c {
        Confidence::Terminal => 0,
        Confidence::Verified => 1,
        Confidence::Unverified => 2,
        Confidence::Inferred => 3,
        Confidence::Contested => 4,
    }
}

pub fn u8_to_confidence(b: u8) -> Confidence {
    match b {
        0 => Confidence::Terminal,
        1 => Confidence::Verified,
        2 => Confidence::Unverified,
        3 => Confidence::Inferred,
        4 => Confidence::Contested,
        _ => Confidence::Unverified,
    }
}

/// Serialize a `TemporalKeyframe` to a length-prefixed bincode blob.
/// Public so the writer (`sst_writer`) and the reader share the
/// format. The length is `u32` LE; the bincode payload follows.
pub fn encode_keyframe(kf: &TemporalKeyframe, out: &mut Vec<u8>) {
    let bytes = bincode::serialize(kf).expect("keyframe bincode serialize");
    out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(&bytes);
}

// ---------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bond::{BondType, KeyframeSource, TemporalKeyframe};
    use std::collections::HashMap;
    use crate::encrypt::MasterKey;
    use crate::sst_writer::SSTWriter;
    use rand::{Rng, SeedableRng};
    use rand::rngs::StdRng;
    use std::sync::Arc;

    fn rand_seed(rng: &mut StdRng) -> Seed {
        let mut b = [0u8; 32];
        rng.fill(&mut b);
        Seed::from_bytes(b)
    }

    fn test_key() -> MasterKey {
        MasterKey::from_bytes([0xa5u8; 32])
    }

    fn make_kf(ts: i64, w: f32) -> TemporalKeyframe {
        TemporalKeyframe { timestamp: ts, weight: w, source: KeyframeSource::Baseline }
    }

    #[test]
    fn compact_bond_size_is_14_bytes() {
        // Load-bearing for Phase 2.1: the byte arithmetic in
        // parse_block and the writer's `approx_bytes` depend on this.
        assert_eq!(COMPACT_BOND_BYTES, 14);
    }

    #[test]
    fn predicate_pack_round_trip_all_variants() {
        // v4 SST predicate is a full u8: every BondType discriminant (0..=43,
        // headroom to 255) round-trips. dir/flags are no longer stored (zero).
        for bt in BondType::all() {
            let p = pack_predicate(*bt, false, 0);
            let (b2, d2, f2) = unpack_predicate(p);
            assert_eq!(b2, *bt, "v4 predicate round-trip broke for {:?}", bt);
            assert_eq!(d2, false);
            assert_eq!(f2, 0);
        }
    }

    #[test]
    fn predicate_v3_legacy_decode_still_reads() {
        // The 94.5M-bond canonical brain has v3 SSTs (5-bit predicate). Confirm
        // the versioned decoder still reads them for the 32 v3-representable
        // discriminants after the v4 cutover.
        for bt in BondType::all() {
            let d = crate::memtable::predicate_discriminant(*bt);
            if d >= 32 { continue; }
            let (b2, _, _) = unpack_predicate_versioned(d, SST_VERSION_LEGACY_5BIT);
            assert_eq!(b2, *bt, "v3 decode broke for {:?}", bt);
        }
    }

    #[test]
    fn weight_quantize_round_trip_within_1_over_255() {
        for steps in 0..=255u32 {
            let f = steps as f32 / 255.0;
            let q = quantize_weight(f);
            let f2 = dequantize_weight(q);
            assert!((f - f2).abs() <= 1.0 / 255.0 + 1e-6, "f={} f2={}", f, f2);
        }
        assert_eq!(quantize_weight(f32::NAN), 0);
        assert_eq!(quantize_weight(-1.0), 0);
        assert_eq!(quantize_weight(2.0), 255);
    }

    #[test]
    fn header_round_trip() {
        let h = SSTHeader {
            magic: SST_MAGIC,
            version: SST_VERSION,
            level: 2,
            bond_count: 12345,
            block_count: 7,
            manifest_id: 0xDEAD_BEEF_CAFE_BABE,
            index_offset: 1024,
            index_length: 7 * SST_INDEX_ENTRY_BYTES as u64,
            encryption_iv: [0u8; 16],
            subject_min_seed: Seed::from_label("min"),
            subject_max_seed: Seed::from_label("max"),
        };
        let mut buf = vec![0u8; SST_HEADER_BYTES];
        h.write_to(&mut buf);
        let h2 = SSTHeader::read_from(&buf).unwrap();
        assert_eq!(h.level, h2.level);
        assert_eq!(h.bond_count, h2.bond_count);
        assert_eq!(h.block_count, h2.block_count);
        assert_eq!(h.manifest_id, h2.manifest_id);
        assert_eq!(h.subject_min_seed, h2.subject_min_seed);
        assert_eq!(h.subject_max_seed, h2.subject_max_seed);
    }

    #[test]
    fn header_refuses_v1_sst_clearly() {
        // Build a header buffer with version = 1 (Phase 2 fixture) and
        // confirm the reader errors with a clear message naming
        // "unsupported SST version". This is the Phase 2.1 cut-over
        // contract: no v1 production data exists, so we refuse rather
        // than carry conversion code.
        let h = SSTHeader {
            magic: SST_MAGIC,
            version: 1, // forge a Phase 2 v1 header
            level: 0,
            bond_count: 0,
            block_count: 0,
            manifest_id: 0,
            index_offset: 0,
            index_length: 0,
            encryption_iv: [0u8; 16],
            subject_min_seed: Seed::from_bytes([0u8; 32]),
            subject_max_seed: Seed::from_bytes([0u8; 32]),
        };
        let mut buf = vec![0u8; SST_HEADER_BYTES];
        h.write_to(&mut buf);
        let err = SSTHeader::read_from(&buf).unwrap_err();
        let msg = format!("{}", err);
        assert!(
            msg.contains("unsupported SST version"),
            "expected 'unsupported SST version' in error, got: {}", msg
        );
        assert!(
            msg.contains("got 1"),
            "expected 'got 1' in error, got: {}", msg
        );
    }

    fn write_test_sst(
        path: &Path,
        bonds: &[(Seed, Seed, BondType, f32, Vec<TemporalKeyframe>)],
        key: Option<MasterKey>,
    ) -> Vec<(Seed, Seed, BondType, f32, Vec<TemporalKeyframe>)> {
        let mut sorted = bonds.to_vec();
        sorted.sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));
        let mut w = SSTWriter::create(path, 0, key.clone()).unwrap();
        for (s, o, bt, wt, kfs) in &sorted {
            w.push_bond(s, o, *bt, *wt, Confidence::Verified, kfs).unwrap();
        }
        w.finalize().unwrap();
        sorted
    }

    #[test]
    fn round_trip_10k_bonds_plaintext() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("sst_round_trip_10k_{}.sst", std::process::id()));
        let _ = std::fs::remove_file(&path);

        let mut rng = StdRng::seed_from_u64(0xC0FFEE);
        let mut bonds = Vec::with_capacity(10_000);
        for i in 0..10_000usize {
            let s = rand_seed(&mut rng);
            let o = rand_seed(&mut rng);
            let bt = BondType::all()[i % BondType::all().len()];
            let w = rng.gen::<f32>();
            let kfs = if i % 3 == 0 { vec![make_kf(i as i64, w)] } else { vec![] };
            bonds.push((s, o, bt, w, kfs));
        }
        let sorted = write_test_sst(&path, &bonds, None);

        let cache = Arc::new(BlockCache::new(DEFAULT_SST_BLOCK_CACHE_BYTES));
        let sst = SST::open(&path, None, cache).unwrap();
        assert_eq!(sst.header().bond_count, 10_000);
        assert!(sst.block_count() >= 1);

        // Verify each subject's bonds are present.
        let mut by_subject: HashMap<Seed, Vec<(Seed, BondType, f32)>> = HashMap::new();
        for (s, o, bt, w, _kfs) in &sorted {
            by_subject.entry(*s).or_default().push((*o, *bt, *w));
        }
        for (subject, expected) in by_subject.iter().take(500) {
            let views = sst.get_bonds_for_subject(subject).unwrap();
            assert_eq!(views.len(), expected.len(),
                "subject {:?}: got {} bonds, expected {}",
                subject, views.len(), expected.len());
            let mut got: Vec<(Seed, BondType)> = views.iter()
                .map(|v| (v.object, v.bond_type)).collect();
            let mut exp: Vec<(Seed, BondType)> = expected.iter()
                .map(|(o, bt, _)| (*o, *bt)).collect();
            got.sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));
            exp.sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));
            assert_eq!(got, exp);
        }

        // iter_bonds visits exactly bond_count bonds.
        let total: usize = sst.iter_bonds().map(|r| r.map(|_| 1).unwrap_or(0)).sum();
        assert_eq!(total, 10_000);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn round_trip_5_keyframes_per_bond_byte_exact() {
        // Phase 2.1 NEW: a single bond carrying 5 keyframes must come
        // back with exactly those 5 keyframes, byte-exact.
        let dir = std::env::temp_dir();
        let path = dir.join(format!("sst_5kf_{}.sst", std::process::id()));
        let _ = std::fs::remove_file(&path);

        let s = Seed::from_label("alpha");
        let o = Seed::from_label("beta");
        let kfs: Vec<TemporalKeyframe> = (0..5)
            .map(|i| TemporalKeyframe {
                timestamp: 1_000_000 + i as i64,
                weight: 0.10 + (i as f32) * 0.15,
                source: match i % 4 {
                    0 => KeyframeSource::Baseline,
                    1 => KeyframeSource::Hebbian,
                    2 => KeyframeSource::Session(42),
                    _ => KeyframeSource::Manual,
                },
            })
            .collect();

        let mut w = SSTWriter::create(&path, 0, None).unwrap();
        w.push_bond(&s, &o, BondType::IsA, 0.7, Confidence::Verified, &kfs).unwrap();
        w.finalize().unwrap();

        let cache = Arc::new(BlockCache::new(DEFAULT_SST_BLOCK_CACHE_BYTES));
        let sst = SST::open(&path, None, cache).unwrap();
        let views = sst.get_bonds_for_subject(&s).unwrap();
        assert_eq!(views.len(), 1, "expected one bond, got {}", views.len());
        let v = &views[0];
        assert_eq!(v.keyframes.len(), 5, "expected 5 keyframes");
        // Byte-exact comparison: bincode round-trip both sides and
        // compare. KeyframeSource doesn't implement PartialEq directly
        // for some variants in all paths, so compare on the bincode
        // bytes which are the on-disk canonical form anyway.
        for (i, (got, exp)) in v.keyframes.iter().zip(kfs.iter()).enumerate() {
            let got_b = bincode::serialize(got).unwrap();
            let exp_b = bincode::serialize(exp).unwrap();
            assert_eq!(got_b, exp_b, "keyframe {} mismatch", i);
        }
        // latest_keyframe convenience
        let latest = v.latest_keyframe().expect("latest_keyframe");
        assert_eq!(latest.timestamp, 1_000_004);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn round_trip_1000_bonds_random_keyframe_history() {
        // Phase 2.1 NEW: 1000 bonds each with 1–10 keyframes; verify
        // total keyframe count and per-bond shape are preserved.
        let dir = std::env::temp_dir();
        let path = dir.join(format!("sst_1k_multi_kf_{}.sst", std::process::id()));
        let _ = std::fs::remove_file(&path);

        let mut rng = StdRng::seed_from_u64(0xDEC0_DE);
        let mut bonds: Vec<(Seed, Seed, BondType, f32, Vec<TemporalKeyframe>)> = Vec::with_capacity(1000);
        let mut expected_total_kfs: usize = 0;
        for i in 0..1000usize {
            let s = rand_seed(&mut rng);
            let o = rand_seed(&mut rng);
            let bt = BondType::all()[i % BondType::all().len()];
            let w = rng.gen::<f32>();
            let n_kf = rng.gen_range(1..=10usize);
            expected_total_kfs += n_kf;
            let kfs: Vec<TemporalKeyframe> = (0..n_kf)
                .map(|k| TemporalKeyframe {
                    timestamp: (i as i64) * 100 + k as i64,
                    weight: ((i * 7 + k * 3) % 256) as f32 / 255.0,
                    source: match k % 4 {
                        0 => KeyframeSource::Baseline,
                        1 => KeyframeSource::Hebbian,
                        2 => KeyframeSource::Session((i * 31 + k) as u64),
                        _ => KeyframeSource::Manual,
                    },
                })
                .collect();
            bonds.push((s, o, bt, w, kfs));
        }
        // Map subject → expected keyframe count(s) (by object).
        let mut expected: HashMap<Seed, Vec<(Seed, usize, i64)>> = HashMap::new();
        for (s, o, _, _, kfs) in &bonds {
            expected.entry(*s).or_default().push((*o, kfs.len(), kfs[0].timestamp));
        }

        let sorted = write_test_sst(&path, &bonds, None);
        assert_eq!(sorted.len(), 1000);

        let cache = Arc::new(BlockCache::new(DEFAULT_SST_BLOCK_CACHE_BYTES));
        let sst = SST::open(&path, None, cache).unwrap();

        // Sum keyframe counts across the full SST.
        let mut got_total = 0usize;
        let mut bonds_seen = 0usize;
        for r in sst.iter_bonds() {
            let v = r.unwrap();
            got_total += v.keyframes.len();
            bonds_seen += 1;
            // Per-bond shape sanity: at least 1 keyframe and timestamps
            // align with the original bond.
            assert!(v.keyframes.len() >= 1 && v.keyframes.len() <= 10);
            // Verify each subject's entry exists in `expected` with
            // matching count for at least ONE of the recorded bonds at
            // this subject (we don't enforce object order since the SST
            // may interleave when subjects collide).
            let entry = expected.get(&v.subject).expect("subject in expected");
            let match_found = entry.iter().any(|(o, n, ts)| {
                *o == v.object && *n == v.keyframes.len() && *ts == v.keyframes[0].timestamp
            });
            assert!(match_found,
                "no expected entry matched subject {:?} object {:?} kf_count {} first_ts {}",
                v.subject, v.object, v.keyframes.len(), v.keyframes[0].timestamp);
        }
        assert_eq!(bonds_seen, 1000);
        assert_eq!(got_total, expected_total_kfs,
            "total keyframe count mismatch: got {}, expected {}",
            got_total, expected_total_kfs);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn encryption_parity_and_key_required() {
        let dir = std::env::temp_dir();
        let plain_path = dir.join(format!("sst_parity_plain_{}.sst", std::process::id()));
        let enc_path = dir.join(format!("sst_parity_enc_{}.sst", std::process::id()));
        let _ = std::fs::remove_file(&plain_path);
        let _ = std::fs::remove_file(&enc_path);

        let mut rng = StdRng::seed_from_u64(0xFEED);
        let mut bonds = Vec::with_capacity(2_000);
        for i in 0..2_000usize {
            bonds.push((
                rand_seed(&mut rng),
                rand_seed(&mut rng),
                BondType::all()[i % BondType::all().len()],
                rng.gen::<f32>(),
                Vec::<TemporalKeyframe>::new(),
            ));
        }
        let sorted_plain = write_test_sst(&plain_path, &bonds, None);
        let sorted_enc = write_test_sst(&enc_path, &bonds, Some(test_key()));

        let cache = Arc::new(BlockCache::new(DEFAULT_SST_BLOCK_CACHE_BYTES));
        let sst_plain = SST::open(&plain_path, None, cache.clone()).unwrap();
        let sst_enc = SST::open(&enc_path, Some(test_key()), cache.clone()).unwrap();

        for (subj, _, _, _, _) in sorted_plain.iter().step_by(50).take(40) {
            let a = sst_plain.get_bonds_for_subject(subj).unwrap();
            let b = sst_enc.get_bonds_for_subject(subj).unwrap();
            assert_eq!(a.len(), b.len());
        }
        let _ = sorted_enc; // silence unused

        let cache2 = Arc::new(BlockCache::new(DEFAULT_SST_BLOCK_CACHE_BYTES));
        let sst_wrong = SST::open(&enc_path, None, cache2).unwrap();
        let mut mismatch = false;
        for (subj, _, _, _, _) in sorted_plain.iter().take(30) {
            match sst_wrong.get_bonds_for_subject(subj) {
                Err(_) => { mismatch = true; break; }
                Ok(v) if v.is_empty() => continue,
                Ok(_v) => {
                    let plain = sst_plain.get_bonds_for_subject(subj).unwrap();
                    if plain.len() != _v.len() { mismatch = true; break; }
                }
            }
        }
        assert!(mismatch, "encrypted SST read without key looked identical to plaintext");

        let _ = std::fs::remove_file(&plain_path);
        let _ = std::fs::remove_file(&enc_path);
    }

    #[test]
    fn empty_sst_clean_handling() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("sst_empty_{}.sst", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let w = SSTWriter::create(&path, 0, None).unwrap();
        w.finalize().unwrap();
        let cache = Arc::new(BlockCache::new(DEFAULT_SST_BLOCK_CACHE_BYTES));
        let sst = SST::open(&path, None, cache).unwrap();
        assert_eq!(sst.header().bond_count, 0);
        assert_eq!(sst.block_count(), 0);
        let any = sst.get_bonds_for_subject(&Seed::from_label("x")).unwrap();
        assert!(any.is_empty());
        let total: usize = sst.iter_bonds().map(|r| r.map(|_| 1).unwrap_or(0)).sum();
        assert_eq!(total, 0);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn concurrent_reads_no_data_race() {
        use std::sync::Arc;
        use std::thread;
        let dir = std::env::temp_dir();
        let path = dir.join(format!("sst_concurrent_{}.sst", std::process::id()));
        let _ = std::fs::remove_file(&path);

        let mut rng = StdRng::seed_from_u64(0xBA5E);
        let mut bonds = Vec::with_capacity(5_000);
        for i in 0..5_000usize {
            bonds.push((
                rand_seed(&mut rng),
                rand_seed(&mut rng),
                BondType::all()[i % BondType::all().len()],
                rng.gen::<f32>(),
                if i % 4 == 0 { vec![make_kf(i as i64, 0.5)] } else { vec![] },
            ));
        }
        let sorted = write_test_sst(&path, &bonds, None);
        let cache = Arc::new(BlockCache::new(8 * 1024 * 1024)); // small cap, force eviction
        let sst = Arc::new(SST::open(&path, None, cache.clone()).unwrap());

        let subjects: Arc<Vec<Seed>> = Arc::new(sorted.iter().map(|t| t.0).collect());

        let mut handles = Vec::new();
        for t in 0..4u32 {
            let sst_c = sst.clone();
            let subs = subjects.clone();
            handles.push(thread::spawn(move || {
                let mut rng = StdRng::seed_from_u64(0xBEEF_0000 + t as u64);
                let mut found = 0usize;
                for _ in 0..1_000 {
                    let idx = rng.gen_range(0..subs.len());
                    let v = sst_c.get_bonds_for_subject(&subs[idx]).unwrap();
                    assert!(!v.is_empty());
                    found += v.len();
                }
                found
            }));
        }
        for h in handles { let _ = h.join().unwrap(); }
        assert!(cache.resident_bytes() <= cache.cap_bytes());

        let _ = std::fs::remove_file(&path);
    }
}
