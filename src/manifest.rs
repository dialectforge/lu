//! OP-BRAIN-V4 Phase 4 -- Manifest.
//!
//! On-disk metadata that names every live SST in the brain. Updated
//! atomically by the Phase 4 flush worker (and, in later phases, the
//! compactor); read by Brain at startup to discover the SST set.
//!
//! ## On-disk format
//!
//! `MANIFEST` is a self-contained file with the following layout:
//!
//! ```text
//! offset  size  field
//! 0       4     magic            (u32, "LMAN" little-endian)
//! 4       4     schema_version   (u32, current = MANIFEST_SCHEMA_VERSION)
//! 8       4     payload_len      (u32, length of the rkyv body)
//! 12      4     reserved         (zero -- pads payload to 8-byte alignment
//!                                 so rkyv::check_archived_root passes)
//! 16      N     payload          (rkyv-archived ManifestPayload{V1,} per
//!                                 schema_version)
//! 16+N    4     crc32            (u32 over bytes [0 .. 16+N])
//! ```
//!
//! The 12-byte fixed prefix lets the loader cheaply reject mis-typed
//! files (`magic` check) and bound-check the rkyv body (`payload_len`)
//! before invoking the rkyv validator. The trailing CRC32 catches
//! torn writes that survived the rename -- the rkyv archive itself
//! has no checksum.
//!
//! ## Atomic update protocol
//!
//! 1. Compose the new bytes in memory.
//! 2. Open `<dir>/MANIFEST.tmp.<pid>.<nonce>`, write the bytes,
//!    `fsync` the file.
//! 3. `rename` the previous `MANIFEST` to `MANIFEST.prev` (if any).
//! 4. `rename` the `.tmp` to `MANIFEST`.
//! 5. `fsync` the parent directory so the renames are durable.
//!
//! Atomic rename means concurrent readers see either the previous
//! manifest or the new one -- never a torn half-write. The retained
//! `MANIFEST.prev` is the rollback target if the new file later fails
//! validation (CRC or rkyv).
//!
//! ## Concurrency
//!
//! Writes happen on the flush worker thread only (Phase 4) or the
//! compactor (Phase 6). Reads on other threads take a snapshot via
//! `Arc<Manifest>` -- Brain holds `RwLock<Arc<Manifest>>`; writers
//! swap the inner Arc under the write lock, readers clone the Arc
//! under a read lock and release the lock immediately.
//!
//! ## Schema v1 -> v2 (OP-BRAIN-V5 Phase 2.5, Q7)
//!
//! v2 adds:
//!
//! - `node_levels: Vec<NodeLevel>` -- per-LSM-level NodeStore SST
//!   tracking, mirroring the bond-side `levels`. Each `NodeLevel`
//!   carries `Vec<NodeSSTEntry>`.
//! - `next_node_sst_uid: u64` -- monotonic counter for allocating
//!   NodeSST `file_uid` values (the same `u64` the Phase 2.4 writer
//!   stamps into the NodeSST header at bytes 20..28).
//! - `next_epoch: u64` -- monotonic counter for the Q7 epoch-barrier
//!   compaction. Every new SST (bond or node) is stamped with the
//!   current `next_epoch` at create time. Tombstone GC waits for
//!   every potentially-referencing SST to compact past that epoch.
//! - `SSTEntry.epoch: u64` -- per-SST epoch on the existing bond
//!   SSTEntry. Load-bearing for Q7: without per-bond-SST epoch the
//!   barrier cannot order bond and node stores against each other.
//!
//! v1 manifests are still readable (Phase 8 cutover relies on this).
//! Read path: on schema_version=1 the rkyv body is parsed as the
//! frozen `ManifestPayloadV1` type (which mirrors the v1 wire shape
//! exactly), then converted into the v2 in-memory representation
//! with `node_levels = vec![]`, `next_node_sst_uid = 0`,
//! `next_epoch = 0`, and `epoch = 0` on every existing SSTEntry. The
//! next `save_atomic` writes v2.
//!
//! Anything outside `[MANIFEST_MIN_READ_SCHEMA_VERSION,
//! MANIFEST_SCHEMA_VERSION]` -- currently `[1, 5]` after the 6.1
//! bump (v3: `bond_v5_levels` + `next_bond_v5_sst_uid`) and the Phase
//! 3.5 bump (v4: `BondSSTEntry.reverse_file_uid`) -- is a hard error
//! (`ManifestError::UnsupportedSchemaVersion`) -- no silent fallback.
//!
//! Because rkyv 0.7 does not natively support adding fields to an
//! existing archived struct without a wire break, the v1 archetype
//! types (`SSTEntryV1`, `LevelEntryV1`, `ManifestPayloadV1`) are
//! frozen below; they are deserialized read-only by the v1 branch of
//! `load_one` and converted into the live v2 types.

use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};

use rkyv::{Archive, Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::brain::BrainError;
use crate::encrypt::MasterKey;
use crate::node_intern::NodeId;
use crate::seed::Seed;

/// Magic bytes at the head of every manifest file ("LMAN" little-endian).
pub const MANIFEST_MAGIC: u32 = 0x4E_41_4D_4C;
/// Schema version this implementation writes. Bumped if the
/// `ManifestPayload` rkyv layout changes incompatibly.
///
/// History:
/// - 1: OP-BRAIN-V4 Phase 4 original shape.
/// - 2: OP-BRAIN-V5 Phase 2.5 -- adds `SSTEntry.epoch`, `node_levels`,
///   `next_node_sst_uid`, `next_epoch` for Q7 epoch-barrier compaction.
/// - 3: OP-BRAIN-V5 Phase 3.3 -- adds `bond_v5_levels: Vec<BondV5Level>`
///   and `next_bond_v5_sst_uid: u64` for the v5 BondStore substrate.
///   The v4 bond store keeps using `levels: Vec<LevelEntry>` until
///   Phase 3.9 retires it; v5 entries live in a sibling field so v4
///   and v5 coexist through the cutover window without a path-style
///   discriminator inside `SSTEntry`.
/// - 4: OP-BRAIN-V5 Phase 3.5 (Q3.0.B) -- adds
///   `BondSSTEntry.reverse_file_uid: Option<u64>`, pairing each v5
///   bond SST with its `.bsst.reverse` reverse-index file. `None`
///   until Phase 3.6 builds the reverse index at compaction time.
/// - 5: OP-BRAIN-V5 6.1 -- adds `ManifestPayload.format_tag: u8`, the
///   explicit brain-format discriminator (`BRAIN_FORMAT_TAG_V5` marks a
///   v5 node_store/bond_store brain). schema<=4 promotes in-memory with
///   `format_tag = BRAIN_FORMAT_TAG_V4`.
pub const MANIFEST_SCHEMA_VERSION: u32 = 5;
/// Lowest schema_version this implementation accepts on load. Anything
/// outside `[MANIFEST_MIN_READ_SCHEMA_VERSION, MANIFEST_SCHEMA_VERSION]`
/// is rejected with `ManifestError::UnsupportedSchemaVersion`.
pub const MANIFEST_MIN_READ_SCHEMA_VERSION: u32 = 1;
/// Size of the fixed prefix (magic + schema_version + payload_len).
pub const MANIFEST_PREFIX_BYTES: usize = 16;
/// Size of the trailing CRC32.
pub const MANIFEST_TRAILER_BYTES: usize = 4;
/// Filename used for the live manifest within the brain directory.
pub const MANIFEST_FILE_NAME: &str = "MANIFEST";

/// OP-BRAIN-V5 6.1 -- explicit brain-format tag stored in the manifest
/// payload (schema_version 5+). Distinguishes a v5 brain directory
/// (node_store/ + bond_store/ substrate) from a v4 directory PURELY by
/// manifest contents, per the locked loader-cutover decision (explicit
/// tag, NOT directory-shape sniffing). Pre-6.1 manifests (schema <= 4)
/// carry no tag and promote to `BRAIN_FORMAT_TAG_V4` on load.
pub const BRAIN_FORMAT_TAG_V4: u8 = 0;
/// See [`BRAIN_FORMAT_TAG_V4`]. Stamped by the v3->v5 migration tools.
pub const BRAIN_FORMAT_TAG_V5: u8 = 5;
/// Filename used for the rolled-back-to-able previous manifest.
pub const MANIFEST_PREV_FILE_NAME: &str = "MANIFEST.prev";

// =========================================================================
// ManifestError -- typed errors for the Phase 2.5 helpers and the
// schema-version gate. Kept separate from `BrainError` because
// the helpers below operate purely on the in-memory representation
// and the typed surface helps Phase 2.6/2.7 wiring.
//
// `BrainError` conversion is provided so the existing `load` /
// `save_atomic` signatures (which return `BrainError`) stay
// source-compatible.
// =========================================================================

/// Errors raised by manifest mutation helpers and the schema-version gate.
#[derive(Debug)]
pub enum ManifestError {
    /// Schema-version on disk is outside
    /// `[MANIFEST_MIN_READ_SCHEMA_VERSION, MANIFEST_SCHEMA_VERSION]`.
    /// Carries the offending value so the operator can see what they
    /// pointed at. NOT a silent fallback -- the load aborts.
    UnsupportedSchemaVersion {
        found: u32,
        min_supported: u32,
        max_supported: u32,
    },
    /// `add_node_sst` called with `level >= node_levels.len()`. We
    /// require levels to be created in order (level N requires levels
    /// 0..N-1 to already exist) so iteration order is deterministic
    /// and L0->L1 promotion can't accidentally land in a hole.
    InvalidLevel { requested: usize, current_len: usize },
    /// `retire_node_ssts` called with a `file_uid` that does not
    /// appear in any node level. Returned together with the list of
    /// successfully-retired entries up to the failure point would be
    /// ambiguous; we instead make retire all-or-nothing: validate
    /// presence first, then perform the removal.
    UnknownNodeSstUid(u64),
    /// `add_node_sst` called twice with the same `file_uid`. The
    /// manifest is the authority on which `file_uid`s are live, so
    /// the flush worker must never reuse a uid (it should always
    /// call `allocate_node_sst_uid` immediately before constructing
    /// the NodeSSTEntry).
    DuplicateNodeSstUid(u64),
    /// Phase 3.3: `retire_bond_v5_ssts` called with a `file_uid`
    /// that does not appear in any v5 bond level. Symmetric to
    /// `UnknownNodeSstUid`; the retire is all-or-nothing.
    UnknownBondV5SstUid(u64),
    /// Phase 3.3: `add_bond_v5_sst` called twice with the same
    /// `file_uid`. The manifest is the authority on which v5 bond
    /// `file_uid`s are live; the BondFlushWorker must always call
    /// `allocate_bond_v5_sst_uid` immediately before constructing
    /// the BondSSTEntry.
    DuplicateBondV5SstUid(u64),
}

impl std::fmt::Display for ManifestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ManifestError::UnsupportedSchemaVersion { found, min_supported, max_supported } => {
                write!(
                    f,
                    "manifest schema version {} is outside supported range [{}, {}]",
                    found, min_supported, max_supported
                )
            }
            ManifestError::InvalidLevel { requested, current_len } => {
                write!(
                    f,
                    "add_node_sst into level {} but node_levels has length {}; create lower levels first",
                    requested, current_len
                )
            }
            ManifestError::UnknownNodeSstUid(uid) => {
                write!(f, "retire_node_ssts: file_uid {} not present in any node level", uid)
            }
            ManifestError::DuplicateNodeSstUid(uid) => {
                write!(f, "add_node_sst: file_uid {} already live in node_levels", uid)
            }
            ManifestError::UnknownBondV5SstUid(uid) => {
                write!(f, "retire_bond_v5_ssts: file_uid {} not present in any v5 bond level", uid)
            }
            ManifestError::DuplicateBondV5SstUid(uid) => {
                write!(f, "add_bond_v5_sst: file_uid {} already live in bond_v5_levels", uid)
            }
        }
    }
}

impl std::error::Error for ManifestError {}

impl From<ManifestError> for BrainError {
    fn from(e: ManifestError) -> Self {
        BrainError::Format(format!("manifest: {}", e))
    }
}

/// 16-byte fingerprint of the master key used to encrypt the brain.
///
/// SHA-256 of the master-key bytes, truncated to 16 bytes. Stored in
/// the manifest so a load can refuse a brain whose key doesn't match.
#[derive(Archive, Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[archive(check_bytes)]
pub struct KeyFingerprint(pub [u8; 16]);

impl KeyFingerprint {
    /// Compute the 16-byte fingerprint of a master key.
    pub fn of(key: &MasterKey) -> Self {
        let mut h = Sha256::new();
        h.update(key.as_bytes());
        let digest = h.finalize();
        let mut out = [0u8; 16];
        out.copy_from_slice(&digest[..16]);
        KeyFingerprint(out)
    }

    /// Special "no key" fingerprint -- used for unencrypted brains.
    /// All zero bytes; an attacker who happens to derive a key whose
    /// SHA-256 starts with 16 zero bytes does not get a free decrypt
    /// because the brain itself is plaintext.
    pub fn none() -> Self {
        KeyFingerprint([0u8; 16])
    }
}

// =========================================================================
// Live (schema v2) types: SSTEntry, LevelEntry, NodeSSTEntry,
// NodeLevel, ManifestPayload.
//
// `SSTEntry` is the on-disk metadata for a bond-store SST. Phase 2.5
// added the trailing `epoch: u64` field; v1 manifests load with
// `epoch=0` and the next save writes v2 with the explicit field.
// =========================================================================

/// One bond-store SST file's metadata as recorded in the manifest.
///
/// **Schema v2 (OP-BRAIN-V5 Phase 2.5) added `epoch: u64`** -- a
/// monotonic counter shared with the node store (via
/// `ManifestPayload.next_epoch`) so the Q7 epoch-barrier compaction
/// can order operations across stores. v1 manifests deserialize with
/// `epoch=0`; the next save promotes them to v2.
#[derive(Archive, Serialize, Deserialize, Clone, Debug)]
#[archive(check_bytes)]
pub struct SSTEntry {
    /// Manifest-allocated unique id. Monotonic across the brain's
    /// lifetime (see `Manifest::next_sst_id`).
    pub id: u64,
    /// Relative path of the SST file from the manifest's directory.
    /// Stored as a `String` (not `PathBuf`) so rkyv handles it
    /// cleanly and so the on-disk form is filesystem-portable.
    pub file: String,
    /// First subject Seed in the file (inclusive).
    pub subject_min: [u8; 32],
    /// Last subject Seed in the file (inclusive).
    pub subject_max: [u8; 32],
    /// File size in bytes at flush time. Diagnostic only.
    pub size: u64,
    /// File mtime (seconds since epoch) at flush time. Diagnostic only.
    pub mtime: i64,
    /// Number of bonds in the SST. Cumulated for level stats.
    pub bond_count: u64,
    /// Manifest-allocated epoch stamp (Phase 2.5, Q7). Allocated at
    /// SST create time via `Manifest::allocate_epoch`. Compaction
    /// outputs inherit MAX(input epochs)+1 so the barrier can detect
    /// "all bond SSTs referencing tombstone E have been compacted".
    /// v1 manifests deserialize this as 0; production code should
    /// treat epoch=0 as "pre-Phase-2.5 record, no epoch claimed".
    pub epoch: u64,
}

impl SSTEntry {
    /// Convenience: return `subject_min` as a `Seed`.
    pub fn subject_min_seed(&self) -> Seed {
        Seed::from_bytes(self.subject_min)
    }
    /// Convenience: return `subject_max` as a `Seed`.
    pub fn subject_max_seed(&self) -> Seed {
        Seed::from_bytes(self.subject_max)
    }
}

/// One LSM level worth of bond-store SSTs.
#[derive(Archive, Serialize, Deserialize, Clone, Debug)]
#[archive(check_bytes)]
pub struct LevelEntry {
    /// Level number: 0 = recent flushes, 1+ = compacted.
    pub level: u8,
    /// SSTs currently live at this level. For L0 they may have
    /// overlapping subject ranges; for L1+ Phase 6 guarantees
    /// disjoint subject ranges.
    pub ssts: Vec<SSTEntry>,
}

/// One NodeStore SST file's metadata as recorded in the manifest.
///
/// Phase 2.5 (Q7) introduces this as the node-side analogue of
/// `SSTEntry`. The `file_uid` is the same `u64` that the Phase 2.4
/// `NodeSstWriter` stamps into the NodeSST header at bytes 20..28
/// (see `crate::node_store::sst_writer::NodeSstWriteOpts::file_uid`
/// and `crate::node_store::sst_reader::NodeSstReader::file_uid`).
/// The manifest is the authority on which `file_uid`s are live.
///
/// Per audit S5.3, the node-store entry tracks `node_count` rather
/// than `bond_count` (the two stores have distinct row semantics);
/// the file path is derived from the `file_uid` by the eventual
/// flush worker so it is not stored here (subject to revisit in
/// Phase 2.6 if the flush worker wants to choose its own filename).
#[derive(Archive, Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
#[archive(check_bytes)]
pub struct NodeSSTEntry {
    /// Stable file identifier stamped into the NodeSST header by the
    /// Phase 2.4 writer at `bytes 20..28`. Allocated by
    /// `Manifest::allocate_node_sst_uid` immediately before the
    /// flush worker constructs the writer.
    pub file_uid: u64,
    /// Lowest `NodeId` in the file (inclusive). NodeSST is sorted by
    /// NodeId ascending; this is the first key in the block index.
    /// Stored as raw `u64` so the rkyv archive stays self-contained
    /// (NodeId does not derive `Archive`); use `first_node_id_typed()`
    /// for the typed surface.
    pub first_node_id: u64,
    /// Highest `NodeId` in the file (inclusive). This is the last
    /// key in the block index. Stored as raw `u64`; use
    /// `last_node_id_typed()` for the typed surface.
    pub last_node_id: u64,
    /// Number of entries (live + tombstone) in the NodeSST.
    /// Tombstone GC keys off the epoch barrier, not on this counter.
    pub entry_count: u64,
    /// File size in bytes at finish time. Diagnostic only.
    pub file_size: u64,
    /// Manifest-allocated epoch stamp (Q7). Shares the counter with
    /// the bond-side `SSTEntry.epoch`. Compaction outputs use
    /// MAX(input epochs)+1 so tombstone GC can detect when every
    /// bond SST that could reference a tombstoned NodeId has been
    /// compacted past the tombstone's epoch.
    pub epoch: u64,
}

impl NodeSSTEntry {
    /// Typed view of `first_node_id`. Mirrors the `subject_min_seed`
    /// pattern on `SSTEntry` -- the field is stored as raw `u64` so
    /// the rkyv archive stays self-contained; this accessor returns
    /// the typed `NodeId` for callers.
    pub fn first_node_id_typed(&self) -> NodeId {
        NodeId(self.first_node_id)
    }
    /// Typed view of `last_node_id`. See `first_node_id_typed`.
    pub fn last_node_id_typed(&self) -> NodeId {
        NodeId(self.last_node_id)
    }
}

/// One LSM level worth of NodeStore SSTs.
///
/// Phase 2.5 ships the wrapper shape; the flush worker (Phase 2.6)
/// and the compactor (Phase 2.7) populate the levels. Disjointness
/// guarantees for L1+ live in the node-store compaction module --
/// the manifest carries the entries but does not validate cross-SST
/// invariants here (mirrors the bond-side pattern where
/// `levels_ge_1_disjoint` is queryable but not enforced on insert).
#[derive(Archive, Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
#[archive(check_bytes)]
pub struct NodeLevel {
    /// Level number: 0 = recent flushes, 1+ = compacted.
    pub level: u8,
    /// NodeSSTs currently live at this level.
    pub ssts: Vec<NodeSSTEntry>,
}

/// One v5 BondStore SST file's metadata as recorded in the manifest.
///
/// OP-BRAIN-V5 Phase 3.3. Sibling to [`NodeSSTEntry`]: the v5 bond
/// store gets its own LSM levels (`bond_v5_levels`) and its own
/// allocator (`next_bond_v5_sst_uid`) so the v4 bond store
/// (`levels: Vec<LevelEntry>` with `SSTEntry { id, file, ... }`) can
/// coexist with the v5 bond store through the Phase 3.3 -> 3.9
/// cutover window without any path-style discriminator on the v4
/// `SSTEntry`. The flush worker (Phase 3.3) writes both a v4
/// `<level>_<id>.sst` AND a v5 `<file_uid>.bsst` per flush; the
/// manifest carries the v4 entry under `levels` and the v5 entry
/// under `bond_v5_levels` -- two independent fields, two
/// independent retiring schedules.
///
/// The `file_uid` is the same `u64` that the Phase 3.1 `BondSstWriter`
/// stamps into the BondSST header at bytes 20..28 (see
/// `crate::bond_store::BondSstWriteOpts::file_uid` and
/// `crate::bond_store::BondSstReader::file_uid`). The manifest is
/// the authority on which `file_uid`s are live.
///
/// Phase 3.5 (Q3.0.B) added `reverse_file_uid` for the secondary
/// reverse-index file (see Phase 3.0 audit S7.2 (B)); see the
/// field doc for its `None` semantics.
#[derive(Archive, Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
#[archive(check_bytes)]
pub struct BondSSTEntry {
    /// Stable file identifier stamped into the BondSST header by
    /// the Phase 3.1 writer at `bytes 20..28`. Allocated by
    /// `Manifest::allocate_bond_v5_sst_uid` immediately before the
    /// flush worker constructs the writer.
    pub file_uid: u64,
    /// Lowest `NodeId` subject in the file (inclusive). BondSST is
    /// sorted by `(subject, predicate, object)` ascending; this is
    /// the first subject in the block index. Stored as raw `u64` so
    /// the rkyv archive stays self-contained.
    pub first_subject: u64,
    /// Highest `NodeId` subject in the file (inclusive). This is the
    /// last subject in the block index. Stored as raw `u64`.
    pub last_subject: u64,
    /// Number of bond entries (live + tombstone) in the BondSST.
    /// Tombstone GC keys off the epoch barrier, not on this counter.
    pub entry_count: u64,
    /// File size in bytes at finish time. Diagnostic only.
    pub file_size: u64,
    /// Manifest-allocated epoch stamp (Q7). Shares the counter with
    /// `SSTEntry.epoch` and `NodeSSTEntry.epoch` via
    /// `Manifest::allocate_epoch` -- single counter so cross-store
    /// ordering is `==` on `epoch: u64`.
    pub epoch: u64,
    /// Phase 3.5 (Q3.0.B): `file_uid` of the paired `.bsst.reverse`
    /// file holding the reverse-index projection of this bond SST,
    /// at `<bond_store_dir>/<uid>.bsst.reverse`. `Some(uid)` means
    /// that file exists and is referenced; `None` means no reverse
    /// index has been built for this bond SST yet -- the case for
    /// L0 SSTs (reverse not built at flush, Phase 3.3) and for any
    /// SST whose reverse index hasn't been built. Phase 3.6 will
    /// populate this lazily during compaction; Phase 3.5 only
    /// carries the field.
    pub reverse_file_uid: Option<u64>,
}

impl BondSSTEntry {
    /// Typed view of `first_subject`. Mirrors `NodeSSTEntry`'s
    /// `first_node_id_typed`.
    pub fn first_subject_typed(&self) -> NodeId {
        NodeId(self.first_subject)
    }
    /// Typed view of `last_subject`. See `first_subject_typed`.
    pub fn last_subject_typed(&self) -> NodeId {
        NodeId(self.last_subject)
    }
}

/// One LSM level worth of v5 BondStore SSTs.
///
/// Phase 3.3 ships the wrapper shape; the flush worker populates L0
/// and the (future) Phase 3.4 bond compactor will populate L1+.
/// Disjointness guarantees for L1+ live in the bond-store compaction
/// module -- the manifest carries the entries but does not validate
/// cross-SST invariants here (mirrors the bond-side v4 pattern where
/// `levels_ge_1_disjoint` is queryable but not enforced on insert).
#[derive(Archive, Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
#[archive(check_bytes)]
pub struct BondV5Level {
    /// Level number: 0 = recent flushes, 1+ = compacted.
    pub level: u8,
    /// BondSSTs currently live at this level.
    pub ssts: Vec<BondSSTEntry>,
}

/// rkyv-serializable body of the manifest (schema version 3).
///
/// Phase 2.5 grew the v1 shape with `node_levels`, `next_node_sst_uid`,
/// and `next_epoch`. Phase 3.3 grows v2 with `bond_v5_levels` and
/// `next_bond_v5_sst_uid` for the v5 BondStore substrate. Adding
/// fields to an existing rkyv archive is a wire break in rkyv 0.7,
/// which is why the v1 + v2 archetype types (`ManifestPayloadV1`
/// and `ManifestPayloadV2` et al.) below are kept frozen for the
/// backwards-compat read branches in `load_one`.
#[derive(Archive, Serialize, Deserialize, Clone, Debug)]
#[archive(check_bytes)]
pub struct ManifestPayload {
    pub master_key_id: KeyFingerprint,
    pub levels: Vec<LevelEntry>,
    /// Path of the WAL segment paired with the live MemTable. Empty
    /// string when there is no WAL.
    pub memtable_wal: String,
    /// Unix-seconds timestamp of the last compaction event, or 0 if
    /// no compaction has happened yet.
    pub last_compaction: i64,
    /// Monotonic counter for `next_sst_id`. Stored explicitly so a
    /// crash-and-restart cycle does not reuse ids assigned by the
    /// previous run.
    pub next_sst_id: u64,
    /// Node-store LSM levels (Phase 2.5, Q7). Empty on v1->v2 load;
    /// populated by the Phase 2.6 flush worker thereafter.
    pub node_levels: Vec<NodeLevel>,
    /// Monotonic counter for `Manifest::allocate_node_sst_uid`.
    /// Stamped onto NodeSST headers; the manifest is the authority
    /// on which `file_uid`s are live.
    pub next_node_sst_uid: u64,
    /// Monotonic counter for `Manifest::allocate_epoch`. Shared
    /// across bond and node stores -- single counter means cross-
    /// store ordering is `==` on `epoch: u64` (not a translated
    /// comparison).
    pub next_epoch: u64,
    /// v5 BondStore LSM levels (Phase 3.3, Q3/Q7). Empty on v1/v2 ->
    /// v3 load; populated by the Phase 3.3 BondFlushWorker thereafter.
    /// Sibling of `levels` (the v4 bond store) during the Phase 3.3
    /// -> 3.9 cutover window; v4 is retired in Phase 3.9 along with
    /// the `levels` field accessors.
    pub bond_v5_levels: Vec<BondV5Level>,
    /// Monotonic counter for `Manifest::allocate_bond_v5_sst_uid`.
    /// Stamped onto BondSST headers; the manifest is the authority
    /// on which v5 bond `file_uid`s are live. Independent of
    /// `next_node_sst_uid` so the two substrates allocate from
    /// disjoint id spaces -- prevents cross-store uid collisions
    /// during the cutover window where both stores can write at
    /// the same flush.
    pub next_bond_v5_sst_uid: u64,
    /// OP-BRAIN-V5 6.1 -- explicit brain-format tag. `BRAIN_FORMAT_TAG_V5`
    /// for a v5 node_store/bond_store brain (stamped by the v5 migration
    /// tools); `BRAIN_FORMAT_TAG_V4` for everything older. Read by
    /// `Brain::detect_brain_layout` to route a directory to `load_v5`
    /// vs `load_v4` without sniffing directory shape.
    pub format_tag: u8,
}

// =========================================================================
// Frozen v1 archetype types. These mirror the wire shape of
// schema_version=1 exactly so the rkyv check_archived_root validator
// accepts pre-Phase-2.5 manifest bytes. They are deserialized only
// by `load_one` when schema_version=1, then converted to the v2
// `ManifestPayload` / `SSTEntry` / `LevelEntry` types above for
// in-memory representation.
//
// DO NOT MODIFY -- changing these is a wire break for every brain
// produced before Phase 2.5. If a v1.x patch becomes necessary, mint
// a v1b type instead.
// =========================================================================

#[derive(Archive, Serialize, Deserialize, Clone, Debug)]
#[archive(check_bytes)]
struct SSTEntryV1 {
    id: u64,
    file: String,
    subject_min: [u8; 32],
    subject_max: [u8; 32],
    size: u64,
    mtime: i64,
    bond_count: u64,
}

#[derive(Archive, Serialize, Deserialize, Clone, Debug)]
#[archive(check_bytes)]
struct LevelEntryV1 {
    level: u8,
    ssts: Vec<SSTEntryV1>,
}

#[derive(Archive, Serialize, Deserialize, Clone, Debug)]
#[archive(check_bytes)]
struct ManifestPayloadV1 {
    master_key_id: KeyFingerprint,
    levels: Vec<LevelEntryV1>,
    memtable_wal: String,
    last_compaction: i64,
    next_sst_id: u64,
}

impl From<SSTEntryV1> for SSTEntry {
    fn from(v1: SSTEntryV1) -> SSTEntry {
        SSTEntry {
            id: v1.id,
            file: v1.file,
            subject_min: v1.subject_min,
            subject_max: v1.subject_max,
            size: v1.size,
            mtime: v1.mtime,
            bond_count: v1.bond_count,
            // v1 had no epoch field. 0 means "pre-Phase-2.5";
            // tombstone GC in node-store compaction treats epoch=0
            // as "oldest" so a freshly migrated brain never
            // accidentally drops a tombstone.
            epoch: 0,
        }
    }
}

impl From<LevelEntryV1> for LevelEntry {
    fn from(v1: LevelEntryV1) -> LevelEntry {
        LevelEntry {
            level: v1.level,
            ssts: v1.ssts.into_iter().map(SSTEntry::from).collect(),
        }
    }
}

impl From<ManifestPayloadV1> for ManifestPayload {
    fn from(v1: ManifestPayloadV1) -> ManifestPayload {
        ManifestPayload {
            master_key_id: v1.master_key_id,
            levels: v1.levels.into_iter().map(LevelEntry::from).collect(),
            memtable_wal: v1.memtable_wal,
            last_compaction: v1.last_compaction,
            next_sst_id: v1.next_sst_id,
            // v1 -> v4 migration defaults (direct promotion --
            // the live ManifestPayload IS the v4 shape): empty node
            // levels and empty v5 bond levels; counters at 0. With
            // zero bond_v5_levels there are no BondSSTEntries, so
            // the Phase 3.5 reverse_file_uid default is vacuous
            // here. The next save_atomic writes schema v4.
            node_levels: Vec::new(),
            next_node_sst_uid: 0,
            next_epoch: 0,
            bond_v5_levels: Vec::new(),
            next_bond_v5_sst_uid: 0,
            // schema 1 predates the explicit tag: legacy => v4.
            format_tag: BRAIN_FORMAT_TAG_V4,
        }
    }
}

// =========================================================================
// Frozen v2 archetype types. These mirror the wire shape of
// schema_version=2 (Phase 2.5) exactly so the rkyv check_archived_root
// validator accepts pre-Phase-3.3 manifest bytes. They are
// deserialized only by `load_one` when schema_version=2, then
// converted to the current `ManifestPayload` for in-memory
// representation.
//
// Phase 3.5 note: `LevelEntryV2` and `NodeLevelV2` are additionally
// composed by the frozen `ManifestPayloadV3` below (the v2 and v3
// wire shapes of those fields are identical) -- one more reason
// these types must stay frozen.
//
// SSTEntryV2 == current SSTEntry shape (v4 bond store; carries `epoch`).
// LevelEntryV2 == current LevelEntry shape.
// NodeSSTEntryV2 == current NodeSSTEntry shape.
// NodeLevelV2 == current NodeLevel shape.
// All four are byte-identical to the live types; the v2 archetypes
// exist so the v2 `ManifestPayloadV2` can be expressed with a
// frozen-shape wrapper independent of whatever fields v3+
// `ManifestPayload` grows.
//
// DO NOT MODIFY -- changing these is a wire break for every brain
// produced between Phase 2.5 and Phase 3.3. If a v2.x patch becomes
// necessary, mint a v2b type instead.
// =========================================================================

#[derive(Archive, Serialize, Deserialize, Clone, Debug)]
#[archive(check_bytes)]
struct SSTEntryV2 {
    id: u64,
    file: String,
    subject_min: [u8; 32],
    subject_max: [u8; 32],
    size: u64,
    mtime: i64,
    bond_count: u64,
    epoch: u64,
}

#[derive(Archive, Serialize, Deserialize, Clone, Debug)]
#[archive(check_bytes)]
struct LevelEntryV2 {
    level: u8,
    ssts: Vec<SSTEntryV2>,
}

#[derive(Archive, Serialize, Deserialize, Clone, Debug)]
#[archive(check_bytes)]
struct NodeSSTEntryV2 {
    file_uid: u64,
    first_node_id: u64,
    last_node_id: u64,
    entry_count: u64,
    file_size: u64,
    epoch: u64,
}

#[derive(Archive, Serialize, Deserialize, Clone, Debug)]
#[archive(check_bytes)]
struct NodeLevelV2 {
    level: u8,
    ssts: Vec<NodeSSTEntryV2>,
}

#[derive(Archive, Serialize, Deserialize, Clone, Debug)]
#[archive(check_bytes)]
struct ManifestPayloadV2 {
    master_key_id: KeyFingerprint,
    levels: Vec<LevelEntryV2>,
    memtable_wal: String,
    last_compaction: i64,
    next_sst_id: u64,
    node_levels: Vec<NodeLevelV2>,
    next_node_sst_uid: u64,
    next_epoch: u64,
}

impl From<SSTEntryV2> for SSTEntry {
    fn from(v2: SSTEntryV2) -> SSTEntry {
        SSTEntry {
            id: v2.id,
            file: v2.file,
            subject_min: v2.subject_min,
            subject_max: v2.subject_max,
            size: v2.size,
            mtime: v2.mtime,
            bond_count: v2.bond_count,
            epoch: v2.epoch,
        }
    }
}

impl From<LevelEntryV2> for LevelEntry {
    fn from(v2: LevelEntryV2) -> LevelEntry {
        LevelEntry {
            level: v2.level,
            ssts: v2.ssts.into_iter().map(SSTEntry::from).collect(),
        }
    }
}

impl From<NodeSSTEntryV2> for NodeSSTEntry {
    fn from(v2: NodeSSTEntryV2) -> NodeSSTEntry {
        NodeSSTEntry {
            file_uid: v2.file_uid,
            first_node_id: v2.first_node_id,
            last_node_id: v2.last_node_id,
            entry_count: v2.entry_count,
            file_size: v2.file_size,
            epoch: v2.epoch,
        }
    }
}

impl From<NodeLevelV2> for NodeLevel {
    fn from(v2: NodeLevelV2) -> NodeLevel {
        NodeLevel {
            level: v2.level,
            ssts: v2.ssts.into_iter().map(NodeSSTEntry::from).collect(),
        }
    }
}

impl From<ManifestPayloadV2> for ManifestPayload {
    fn from(v2: ManifestPayloadV2) -> ManifestPayload {
        ManifestPayload {
            master_key_id: v2.master_key_id,
            levels: v2.levels.into_iter().map(LevelEntry::from).collect(),
            memtable_wal: v2.memtable_wal,
            last_compaction: v2.last_compaction,
            next_sst_id: v2.next_sst_id,
            node_levels: v2.node_levels.into_iter().map(NodeLevel::from).collect(),
            next_node_sst_uid: v2.next_node_sst_uid,
            next_epoch: v2.next_epoch,
            // v2 -> v4 migration defaults (DIRECT promotion, no
            // v3 intermediate: the live ManifestPayload IS the v4
            // shape and every v3-introduced field also defaults
            // here): empty v5 bond levels; counter at 0. With zero
            // bond_v5_levels there are no BondSSTEntries, so the
            // Phase 3.5 reverse_file_uid default is vacuous here.
            // The next save_atomic writes schema v4.
            bond_v5_levels: Vec::new(),
            next_bond_v5_sst_uid: 0,
            // schema 2 predates the explicit tag: legacy => v4.
            format_tag: BRAIN_FORMAT_TAG_V4,
        }
    }
}

// =========================================================================
// Frozen v3 archetype types. These mirror the wire shape of
// schema_version=3 (Phase 3.3) exactly so the rkyv check_archived_root
// validator accepts pre-Phase-3.5 manifest bytes. They are
// deserialized only by `load_one` when schema_version=3, then
// converted to the v4 `ManifestPayload` for in-memory representation.
//
// BondSSTEntryV3 == the Phase 3.3 BondSSTEntry shape (no
// `reverse_file_uid`).
// BondV5LevelV3 == the Phase 3.3 BondV5Level shape (carries
// Vec<BondSSTEntryV3>).
// ManifestPayloadV3 mirrors the v3 ManifestPayload, composing the
// already-frozen `LevelEntryV2` / `NodeLevelV2` archetypes for the
// fields whose wire shape did not change between v2 and v3 -- those
// types are pinned ("DO NOT MODIFY") so reusing them keeps the v3
// wire shape frozen without minting byte-identical duplicates.
//
// DO NOT MODIFY -- changing these is a wire break for every brain
// produced between Phase 3.3 and Phase 3.5. If a v3.x patch becomes
// necessary, mint a v3b type instead.
// =========================================================================

#[derive(Archive, Serialize, Deserialize, Clone, Debug)]
#[archive(check_bytes)]
struct BondSSTEntryV3 {
    file_uid: u64,
    first_subject: u64,
    last_subject: u64,
    entry_count: u64,
    file_size: u64,
    epoch: u64,
}

#[derive(Archive, Serialize, Deserialize, Clone, Debug)]
#[archive(check_bytes)]
struct BondV5LevelV3 {
    level: u8,
    ssts: Vec<BondSSTEntryV3>,
}

#[derive(Archive, Serialize, Deserialize, Clone, Debug)]
#[archive(check_bytes)]
struct ManifestPayloadV3 {
    master_key_id: KeyFingerprint,
    levels: Vec<LevelEntryV2>,
    memtable_wal: String,
    last_compaction: i64,
    next_sst_id: u64,
    node_levels: Vec<NodeLevelV2>,
    next_node_sst_uid: u64,
    next_epoch: u64,
    bond_v5_levels: Vec<BondV5LevelV3>,
    next_bond_v5_sst_uid: u64,
}

impl From<BondSSTEntryV3> for BondSSTEntry {
    fn from(v3: BondSSTEntryV3) -> BondSSTEntry {
        BondSSTEntry {
            file_uid: v3.file_uid,
            first_subject: v3.first_subject,
            last_subject: v3.last_subject,
            entry_count: v3.entry_count,
            file_size: v3.file_size,
            epoch: v3.epoch,
            // v3 had no reverse_file_uid field. None means "no
            // reverse index built for this bond SST yet" -- Phase
            // 3.6 builds reverse indexes lazily at compaction time,
            // so a freshly migrated brain starts with zero reverse
            // files and earns them as compactions run.
            reverse_file_uid: None,
        }
    }
}

impl From<BondV5LevelV3> for BondV5Level {
    fn from(v3: BondV5LevelV3) -> BondV5Level {
        BondV5Level {
            level: v3.level,
            ssts: v3.ssts.into_iter().map(BondSSTEntry::from).collect(),
        }
    }
}

impl From<ManifestPayloadV3> for ManifestPayload {
    fn from(v3: ManifestPayloadV3) -> ManifestPayload {
        ManifestPayload {
            master_key_id: v3.master_key_id,
            levels: v3.levels.into_iter().map(LevelEntry::from).collect(),
            memtable_wal: v3.memtable_wal,
            last_compaction: v3.last_compaction,
            next_sst_id: v3.next_sst_id,
            node_levels: v3.node_levels.into_iter().map(NodeLevel::from).collect(),
            next_node_sst_uid: v3.next_node_sst_uid,
            next_epoch: v3.next_epoch,
            // v3 -> v4 migration: every promoted BondSSTEntry gets
            // `reverse_file_uid = None` (see From<BondSSTEntryV3>).
            // The next save_atomic writes schema v4.
            bond_v5_levels: v3.bond_v5_levels.into_iter().map(BondV5Level::from).collect(),
            next_bond_v5_sst_uid: v3.next_bond_v5_sst_uid,
            // schema 3 predates the explicit tag: legacy => v4.
            format_tag: BRAIN_FORMAT_TAG_V4,
        }
    }
}

// =========================================================================
// Frozen v4 archetype (schema_version=4, Phase 3.5 .. 6.1). Mirrors the
// schema-4 ManifestPayload wire shape -- i.e. the current payload MINUS
// the 6.1 `format_tag` field. Deserialized only by `load_one` when
// schema_version=4, then promoted to the current ManifestPayload with
// `format_tag = BRAIN_FORMAT_TAG_V4` (legacy brains predate the tag and
// are by definition not v5). The shared sub-types (LevelEntry, NodeLevel,
// BondV5Level, ...) did NOT change between schema 4 and 5, so this
// archetype composes them directly rather than minting byte-identical
// duplicates.
//
// DO NOT MODIFY -- changing this is a wire break for every brain produced
// between Phase 3.5 and 6.1.
// =========================================================================
#[derive(Archive, Serialize, Deserialize, Clone, Debug)]
#[archive(check_bytes)]
struct ManifestPayloadV4 {
    master_key_id: KeyFingerprint,
    levels: Vec<LevelEntry>,
    memtable_wal: String,
    last_compaction: i64,
    next_sst_id: u64,
    node_levels: Vec<NodeLevel>,
    next_node_sst_uid: u64,
    next_epoch: u64,
    bond_v5_levels: Vec<BondV5Level>,
    next_bond_v5_sst_uid: u64,
}

impl From<ManifestPayloadV4> for ManifestPayload {
    fn from(v4: ManifestPayloadV4) -> ManifestPayload {
        ManifestPayload {
            master_key_id: v4.master_key_id,
            levels: v4.levels,
            memtable_wal: v4.memtable_wal,
            last_compaction: v4.last_compaction,
            next_sst_id: v4.next_sst_id,
            node_levels: v4.node_levels,
            next_node_sst_uid: v4.next_node_sst_uid,
            next_epoch: v4.next_epoch,
            bond_v5_levels: v4.bond_v5_levels,
            next_bond_v5_sst_uid: v4.next_bond_v5_sst_uid,
            // schema 4 predates the explicit tag: legacy => v4.
            format_tag: BRAIN_FORMAT_TAG_V4,
        }
    }
}

// =========================================================================
// In-memory Manifest -- the live, mutated representation. Cheap to
// clone; serialized to disk via `save_atomic` as a v4 payload
// regardless of which schema the source file had.
// =========================================================================

/// In-memory manifest snapshot.
///
/// Cheap to clone (Vec<LevelEntry> deep-copies; usually tens to
/// hundreds of SSTs total). Phase 4's flush worker mutates a clone
/// and atomically swaps it into the Brain's live slot; readers see
/// either the pre-flush or post-flush manifest, never a partial one.
#[derive(Clone, Debug)]
pub struct Manifest {
    pub schema_version: u32,
    pub master_key_id: KeyFingerprint,
    pub levels: Vec<LevelEntry>,
    pub memtable_wal: Option<PathBuf>,
    pub last_compaction: Option<i64>,
    next_sst_id: u64,
    /// Phase 2.5: node-store LSM levels. Populated by the Phase 2.6
    /// flush worker; empty in pre-Phase-2.5 brains.
    pub node_levels: Vec<NodeLevel>,
    /// Phase 2.5: next NodeSST `file_uid` to hand out via
    /// `allocate_node_sst_uid`.
    next_node_sst_uid: u64,
    /// Phase 2.5: next epoch to hand out via `allocate_epoch`.
    /// Shared across both stores.
    next_epoch: u64,
    /// Phase 3.3: v5 BondStore LSM levels. Populated by the
    /// `BondFlushWorker` (Phase 3.3) at L0 and by the (future)
    /// `BondCompactionWorker` (Phase 3.4) at L1+. Empty in
    /// pre-Phase-3.3 brains. Lives ALONGSIDE `levels` (the v4
    /// bond store) until Phase 3.9 retires v4; each flush produces
    /// one entry in each field during the cutover.
    pub bond_v5_levels: Vec<BondV5Level>,
    /// Phase 3.3: next v5 BondSST `file_uid` to hand out via
    /// `allocate_bond_v5_sst_uid`. Independent of `next_node_sst_uid`
    /// so each substrate allocates from its own monotonic id space.
    next_bond_v5_sst_uid: u64,
    /// OP-BRAIN-V5 6.1 -- explicit brain-format tag (see ManifestPayload).
    pub format_tag: u8,
}

impl Manifest {
    /// Create an empty manifest for a fresh brain.
    pub fn new(master_key_id: KeyFingerprint) -> Self {
        Manifest {
            schema_version: MANIFEST_SCHEMA_VERSION,
            master_key_id,
            levels: Vec::new(),
            memtable_wal: None,
            last_compaction: None,
            next_sst_id: 1, // 0 reserved for "unset"
            node_levels: Vec::new(),
            // Phase 2.5: counters start at 0; allocate_* hands out 0
            // first. The brief mandates 0..99 for the first 100 calls.
            next_node_sst_uid: 0,
            next_epoch: 0,
            // Phase 3.3: v5 bond store starts empty, allocator at 0.
            bond_v5_levels: Vec::new(),
            next_bond_v5_sst_uid: 0,
            // OP-BRAIN-V5 6.1: default to the v4 tag; the v5 migration
            // tools call set_brain_format_v5() to mark a real v5 brain.
            format_tag: BRAIN_FORMAT_TAG_V4,
        }
    }

    /// OP-BRAIN-V5 6.1 -- stamp this manifest as a v5 brain so
    /// `Brain::detect_brain_layout` routes the directory to `load_v5`.
    pub fn set_brain_format_v5(&mut self) {
        self.format_tag = BRAIN_FORMAT_TAG_V5;
    }

    /// True iff this manifest carries the explicit v5 format tag.
    pub fn is_v5_format(&self) -> bool {
        self.format_tag == BRAIN_FORMAT_TAG_V5
    }

    /// Append an SST entry to the appropriate level, creating the
    /// level if missing.
    pub fn add_sst(&mut self, level: u8, entry: SSTEntry) {
        for lvl in &mut self.levels {
            if lvl.level == level {
                lvl.ssts.push(entry);
                return;
            }
        }
        self.levels.push(LevelEntry {
            level,
            ssts: vec![entry],
        });
        // Keep levels sorted so iteration is predictable.
        self.levels.sort_by_key(|l| l.level);
    }

    /// Set the path of the live WAL segment.
    pub fn set_memtable_wal(&mut self, p: Option<PathBuf>) {
        self.memtable_wal = p;
    }

    /// Allocate a fresh monotonic SST id. Bumps the internal counter
    /// in place -- call once per SST being written; the id is then
    /// recorded on the corresponding `SSTEntry`.
    pub fn next_sst_id(&mut self) -> u64 {
        let id = self.next_sst_id;
        self.next_sst_id = self.next_sst_id.checked_add(1)
            .expect("Manifest next_sst_id u64 overflow");
        id
    }

    /// Read-only view of the next id without bumping the counter.
    pub fn peek_next_sst_id(&self) -> u64 {
        self.next_sst_id
    }

    // ---------------------------------------------------------------------
    // Phase 2.5 helpers (Q7) -- node-store + epoch allocators and
    // node-level mutation. All operate on the in-memory representation;
    // callers `save_atomic` after the mutation completes.
    // ---------------------------------------------------------------------

    /// Phase 2.5: allocate a fresh NodeSST `file_uid`. Returns the
    /// current value then bumps the counter. The flush worker
    /// (Phase 2.6) calls this immediately before constructing the
    /// `NodeSstWriter` so the writer's header carries the manifest-
    /// owned identifier.
    ///
    /// First call on a fresh manifest returns 0; the brief mandates
    /// `0..99` for the first 100 calls.
    pub fn allocate_node_sst_uid(&mut self) -> u64 {
        let uid = self.next_node_sst_uid;
        self.next_node_sst_uid = self.next_node_sst_uid.checked_add(1)
            .expect("Manifest next_node_sst_uid u64 overflow");
        uid
    }

    /// Read-only view of the next NodeSST uid without bumping.
    pub fn peek_next_node_sst_uid(&self) -> u64 {
        self.next_node_sst_uid
    }

    /// Phase 2.5: allocate a fresh epoch. Returns the current value
    /// then bumps the counter. Shared across bond and node stores --
    /// callers stamp the returned value onto every newly-created
    /// SSTEntry / NodeSSTEntry. Compaction outputs use a higher
    /// epoch than their inputs so the tombstone-GC barrier can
    /// advance.
    pub fn allocate_epoch(&mut self) -> u64 {
        let e = self.next_epoch;
        self.next_epoch = self.next_epoch.checked_add(1)
            .expect("Manifest next_epoch u64 overflow");
        e
    }

    /// Read-only view of the next epoch without bumping.
    pub fn peek_next_epoch(&self) -> u64 {
        self.next_epoch
    }

    /// Phase 2.5: insert a freshly-flushed NodeSST into the named
    /// level. Used by the Phase 2.6 flush worker to commit a new
    /// L0 NodeSST after `NodeSstWriter::finish`.
    ///
    /// Errors:
    /// - `InvalidLevel` if `level > node_levels.len()`. Levels must
    ///   be created in order -- analogous to the bond-side `add_sst`
    ///   pattern of "missing level is implicitly created" except we
    ///   require the caller to grow levels monotonically (level N
    ///   requires 0..N-1 to exist). `level == node_levels.len()` is
    ///   accepted and creates the new level.
    /// - `DuplicateNodeSstUid` if `entry.file_uid` is already live in
    ///   any node level. The manifest is the authority on live
    ///   `file_uid`s, so duplicate insertion is a bug, not a data
    ///   condition the system can recover from silently.
    pub fn add_node_sst(
        &mut self,
        level: usize,
        entry: NodeSSTEntry,
    ) -> Result<(), ManifestError> {
        if level > self.node_levels.len() {
            return Err(ManifestError::InvalidLevel {
                requested: level,
                current_len: self.node_levels.len(),
            });
        }
        // Validate uid uniqueness across ALL node levels.
        for lvl in &self.node_levels {
            for e in &lvl.ssts {
                if e.file_uid == entry.file_uid {
                    return Err(ManifestError::DuplicateNodeSstUid(entry.file_uid));
                }
            }
        }
        if level == self.node_levels.len() {
            // Grow with a single new level at the requested index.
            self.node_levels.push(NodeLevel {
                level: u8::try_from(level).unwrap_or(u8::MAX),
                ssts: vec![entry],
            });
        } else {
            self.node_levels[level].ssts.push(entry);
        }
        Ok(())
    }

    /// Phase 2.5: remove NodeSST entries by `file_uid`. Used by
    /// the Phase 2.7 compactor after merging inputs into outputs.
    /// All-or-nothing: if any requested uid is absent, no entries
    /// are removed and `UnknownNodeSstUid` is returned. The caller
    /// is expected to insert the compaction outputs (via
    /// `add_node_sst`) before persisting and swapping the live
    /// `Arc<Manifest>` so readers never see a transient gap.
    ///
    /// Returns the retired entries in the order they were requested
    /// in `file_uids`.
    pub fn retire_node_ssts(
        &mut self,
        file_uids: &[u64],
    ) -> Result<Vec<NodeSSTEntry>, ManifestError> {
        // Phase 1: validate every uid is present. We collect
        // (level_idx, ssts_idx) pairs so the actual removal in
        // phase 2 is a single pass with no second lookup.
        let mut locations: Vec<(usize, usize)> = Vec::with_capacity(file_uids.len());
        for &uid in file_uids {
            let mut found = false;
            for (li, lvl) in self.node_levels.iter().enumerate() {
                if let Some(si) = lvl.ssts.iter().position(|e| e.file_uid == uid) {
                    locations.push((li, si));
                    found = true;
                    break;
                }
            }
            if !found {
                return Err(ManifestError::UnknownNodeSstUid(uid));
            }
        }
        // Phase 2: remove. Build output in the request order. We
        // remove highest-index-first within each level so earlier
        // indices stay valid; but locations was captured against
        // the pre-removal arrangement, so build a per-level
        // sorted-desc removal plan and resolve outputs after.
        let mut output: Vec<Option<NodeSSTEntry>> = vec![None; file_uids.len()];
        // Group output_indices by level so we can remove in
        // descending-ssts-index order per level.
        use std::collections::HashMap;
        let mut per_level: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
        for (req_idx, &(li, si)) in locations.iter().enumerate() {
            per_level.entry(li).or_default().push((si, req_idx));
        }
        for (li, mut plan) in per_level.into_iter() {
            plan.sort_by(|a, b| b.0.cmp(&a.0));
            for (si, req_idx) in plan {
                let removed = self.node_levels[li].ssts.remove(si);
                output[req_idx] = Some(removed);
            }
        }
        // Drop now-empty trailing node levels to keep the vector
        // tidy. Only trailing because intermediate empty levels are
        // meaningful (L1 empty while L2 has SSTs means "nothing
        // promoted to L1 yet", not "L2 has no parent").
        while self.node_levels.last().map(|l| l.ssts.is_empty()).unwrap_or(false) {
            self.node_levels.pop();
        }
        Ok(output.into_iter().map(|e| e.expect("all locations resolved")).collect())
    }

    /// Phase 2.5: enumerate every live NodeSST `file_uid` across all
    /// node levels, sorted ascending. Used by the restart-time
    /// file-cleanup pass (delete NodeSST files on disk whose `file_uid`
    /// the manifest does not reference).
    pub fn live_node_sst_uids(&self) -> Vec<u64> {
        let mut out: Vec<u64> = self
            .node_levels
            .iter()
            .flat_map(|l| l.ssts.iter().map(|e| e.file_uid))
            .collect();
        out.sort_unstable();
        out
    }

    /// Phase 2.5: total number of NodeSSTs across all node levels.
    pub fn total_node_sst_count(&self) -> usize {
        self.node_levels.iter().map(|l| l.ssts.len()).sum()
    }

    /// Phase 2.5: iterate every (level_idx, &NodeSSTEntry) pair.
    pub fn iter_node_ssts(&self) -> impl Iterator<Item = (u8, &NodeSSTEntry)> {
        self.node_levels
            .iter()
            .flat_map(|l| l.ssts.iter().map(move |e| (l.level, e)))
    }

    /// Phase 2.7 (Q7): minimum `epoch` across every live bond-store
    /// [`SSTEntry`] in `self.levels`. Returns `None` when zero bond
    /// SSTs exist.
    ///
    /// Load-bearing for node-side tombstone GC: a tombstoned NodeId
    /// can be dropped ONLY when no SST whose `epoch <` the tombstone"s
    /// own source-SST epoch still references it. Since both stores
    /// share the same `next_epoch` counter (via
    /// [`Manifest::allocate_epoch`]) the bond minimum is a valid
    /// lower bound on which tombstones are safe.
    ///
    /// The query is `O(total bond SSTs)`; the manifest never has more
    /// than tens to hundreds of bond SSTs in practice, so a linear
    /// scan is the right shape.
    pub fn min_bond_sst_epoch(&self) -> Option<u64> {
        self.levels
            .iter()
            .flat_map(|l| l.ssts.iter().map(|e| e.epoch))
            .min()
    }

    /// Phase 2.7 (Q7): minimum `epoch` across every live
    /// [`NodeSSTEntry`] in `self.node_levels` whose `file_uid` is
    /// NOT in `exclude_uids`. Returns `None` when no node SST
    /// remains after exclusion.
    ///
    /// The compactor calls this with `exclude_uids = plan.source_uids`
    /// so the inputs being merged do NOT raise their own barrier
    /// (otherwise a one-input compaction would never GC anything).
    ///
    /// `exclude_uids` is treated as a small (typically <= 16) set;
    /// the inner `contains` is O(k) per node SST, which is the right
    /// trade-off for manifest scale.
    pub fn min_node_sst_epoch_excluding(&self, exclude_uids: &[u64]) -> Option<u64> {
        self.node_levels
            .iter()
            .flat_map(|l| l.ssts.iter())
            .filter(|e| !exclude_uids.contains(&e.file_uid))
            .map(|e| e.epoch)
            .min()
    }

    // ---------------------------------------------------------------------
    // Phase 3.3 v5 BondStore helpers. Mirror the node-store surface
    // exactly (`allocate_*`, `peek_next_*`, `add_*`, `retire_*`,
    // `live_*_uids`, `total_*_count`, `iter_*`). The two substrates
    // are deliberately structurally symmetric so flush + cleanup +
    // compaction can share discipline patterns.
    // ---------------------------------------------------------------------

    /// Phase 3.3: allocate a fresh monotonic v5 bond SST `file_uid`.
    /// Bumps `next_bond_v5_sst_uid` in place; the id is then stamped
    /// onto the [`BondSSTEntry`] the caller commits via
    /// [`Manifest::add_bond_v5_sst`].
    ///
    /// Independent of [`Manifest::allocate_node_sst_uid`]: each
    /// substrate owns its own monotonic id space so cross-store uid
    /// collisions are impossible during the cutover window where
    /// both stores write at the same flush.
    pub fn allocate_bond_v5_sst_uid(&mut self) -> u64 {
        let uid = self.next_bond_v5_sst_uid;
        self.next_bond_v5_sst_uid = self.next_bond_v5_sst_uid.checked_add(1)
            .expect("Manifest next_bond_v5_sst_uid u64 overflow");
        uid
    }

    /// Phase 3.3: read-only view of the next v5 bond SST uid without
    /// bumping the counter.
    pub fn peek_next_bond_v5_sst_uid(&self) -> u64 {
        self.next_bond_v5_sst_uid
    }

    /// Phase 3.3: insert a freshly-flushed v5 BondSST into the named
    /// level. Used by the Phase 3.3 BondFlushWorker to commit a new
    /// L0 BondSST after `BondSstWriter::finish`.
    ///
    /// Errors mirror `add_node_sst`:
    /// - `InvalidLevel` if `level > bond_v5_levels.len()`.
    /// - `DuplicateBondV5SstUid` if `entry.file_uid` is already live.
    pub fn add_bond_v5_sst(
        &mut self,
        level: usize,
        entry: BondSSTEntry,
    ) -> Result<(), ManifestError> {
        if level > self.bond_v5_levels.len() {
            return Err(ManifestError::InvalidLevel {
                requested: level,
                current_len: self.bond_v5_levels.len(),
            });
        }
        for lvl in &self.bond_v5_levels {
            for e in &lvl.ssts {
                if e.file_uid == entry.file_uid {
                    return Err(ManifestError::DuplicateBondV5SstUid(entry.file_uid));
                }
            }
        }
        if level == self.bond_v5_levels.len() {
            self.bond_v5_levels.push(BondV5Level {
                level: u8::try_from(level).unwrap_or(u8::MAX),
                ssts: vec![entry],
            });
        } else {
            self.bond_v5_levels[level].ssts.push(entry);
        }
        Ok(())
    }

    /// Phase 3.3: remove v5 BondSST entries by `file_uid`. Used by
    /// the (future) Phase 3.4 bond compactor after merging inputs
    /// into outputs. All-or-nothing: if any requested uid is absent,
    /// no entries are removed and `UnknownBondV5SstUid` is returned.
    pub fn retire_bond_v5_ssts(
        &mut self,
        file_uids: &[u64],
    ) -> Result<Vec<BondSSTEntry>, ManifestError> {
        let mut locations: Vec<(usize, usize)> = Vec::with_capacity(file_uids.len());
        for &uid in file_uids {
            let mut found = false;
            for (li, lvl) in self.bond_v5_levels.iter().enumerate() {
                if let Some(si) = lvl.ssts.iter().position(|e| e.file_uid == uid) {
                    locations.push((li, si));
                    found = true;
                    break;
                }
            }
            if !found {
                return Err(ManifestError::UnknownBondV5SstUid(uid));
            }
        }
        let mut output: Vec<Option<BondSSTEntry>> = vec![None; file_uids.len()];
        use std::collections::HashMap;
        let mut per_level: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
        for (req_idx, &(li, si)) in locations.iter().enumerate() {
            per_level.entry(li).or_default().push((si, req_idx));
        }
        for (li, mut plan) in per_level.into_iter() {
            plan.sort_by(|a, b| b.0.cmp(&a.0));
            for (si, req_idx) in plan {
                let removed = self.bond_v5_levels[li].ssts.remove(si);
                output[req_idx] = Some(removed);
            }
        }
        while self.bond_v5_levels.last().map(|l| l.ssts.is_empty()).unwrap_or(false) {
            self.bond_v5_levels.pop();
        }
        Ok(output.into_iter().map(|e| e.expect("all locations resolved")).collect())
    }

    /// Phase 3.3: enumerate every live v5 BondSST `file_uid` across
    /// all v5 bond levels, sorted ascending. Used by the restart-time
    /// `cleanup_orphan_bond_ssts` pass to classify on-disk `.bsst`
    /// files as live vs. orphan.
    pub fn live_bond_v5_sst_uids(&self) -> Vec<u64> {
        let mut out: Vec<u64> = self
            .bond_v5_levels
            .iter()
            .flat_map(|l| l.ssts.iter().map(|e| e.file_uid))
            .collect();
        out.sort_unstable();
        out
    }

    /// Phase 3.6 (Q3.0.B): enumerate every `reverse_file_uid` across
    /// all live v5 [`BondSSTEntry`]s that have a paired reverse-index
    /// file (`reverse_file_uid == Some(_)`), sorted ascending.
    /// Entries with `None` (L0 flush-born SSTs and any SST whose
    /// reverse index has not been built yet) are excluded. Used by
    /// the restart-time `cleanup_orphan_bond_ssts` pass to classify
    /// on-disk `<uid>.bsst.reverse` files as live vs. orphan,
    /// mirroring [`Manifest::live_bond_v5_sst_uids`] for primaries.
    pub fn live_bond_v5_reverse_uids(&self) -> Vec<u64> {
        let mut out: Vec<u64> = self
            .bond_v5_levels
            .iter()
            .flat_map(|l| l.ssts.iter().filter_map(|e| e.reverse_file_uid))
            .collect();
        out.sort_unstable();
        out
    }

    /// Phase 3.3: total number of v5 BondSSTs across all v5 bond
    /// levels.
    pub fn total_bond_v5_sst_count(&self) -> usize {
        self.bond_v5_levels.iter().map(|l| l.ssts.len()).sum()
    }

    /// Phase 3.3: iterate every (level_idx, &BondSSTEntry) pair.
    pub fn iter_bond_v5_ssts(&self) -> impl Iterator<Item = (u8, &BondSSTEntry)> {
        self.bond_v5_levels
            .iter()
            .flat_map(|l| l.ssts.iter().map(move |e| (l.level, e)))
    }


    /// Phase 3.4 (Q7): minimum `epoch` across every live v5
    /// [`BondSSTEntry`] in `self.bond_v5_levels` whose `file_uid` is
    /// NOT in `exclude_uids`. Returns `None` when no v5 bond SST
    /// remains after exclusion.
    ///
    /// The bond compactor calls this with `exclude_uids =
    /// plan.source_uids` so the inputs being merged do NOT raise
    /// their own barrier (otherwise a one-input compaction would
    /// never GC anything). Mirrors
    /// [`Manifest::min_node_sst_epoch_excluding`] on the node side;
    /// both feed the cross-store `gc_barrier` (the two stores share
    /// one epoch counter via [`Manifest::allocate_epoch`], so the
    /// cross-store comparison is `==` on `epoch: u64`).
    ///
    /// `exclude_uids` is treated as a small (typically <= 16) set;
    /// the inner `contains` is O(k) per bond SST, which is the right
    /// trade-off for manifest scale.
    pub fn min_bond_v5_sst_epoch_excluding(&self, exclude_uids: &[u64]) -> Option<u64> {
        self.bond_v5_levels
            .iter()
            .flat_map(|l| l.ssts.iter())
            .filter(|e| !exclude_uids.contains(&e.file_uid))
            .map(|e| e.epoch)
            .min()
    }

    // ---------------------------------------------------------------------
    // Existing bond-side helpers, unchanged from v1.
    // ---------------------------------------------------------------------

    /// Total number of SSTs across all levels.
    pub fn total_sst_count(&self) -> usize {
        self.levels.iter().map(|l| l.ssts.len()).sum()
    }

    /// Iterate every (level, SSTEntry) pair.
    pub fn iter_ssts(&self) -> impl Iterator<Item = (u8, &SSTEntry)> {
        self.levels.iter().flat_map(|l| l.ssts.iter().map(move |e| (l.level, e)))
    }

    /// Return the set of paths currently referenced by the manifest,
    /// resolved relative to `base_dir`.
    pub fn referenced_paths(&self, base_dir: &Path) -> Vec<PathBuf> {
        self.iter_ssts()
            .map(|(_, e)| base_dir.join(&e.file))
            .collect()
    }

    /// Phase 6: returns true iff every level >= 1 has SSTs whose
    /// subject ranges are pairwise disjoint (i.e. for any two SSTs A
    /// and B in the same level, `A.subject_max < B.subject_min` or
    /// vice versa). L0 may have overlapping ranges; we do not check
    /// it. This is a load-bearing invariant of leveled compaction:
    /// each read at level N consults exactly one SST via binary
    /// search on the subject range. Used by `swap_sst_set` in debug
    /// assertions and by tests.
    pub fn levels_ge_1_disjoint(&self) -> bool {
        for lvl in &self.levels {
            if lvl.level == 0 {
                continue;
            }
            let mut ranges: Vec<([u8; 32], [u8; 32])> = lvl.ssts.iter()
                .map(|e| (e.subject_min, e.subject_max))
                .collect();
            ranges.sort_by(|a, b| a.0.cmp(&b.0));
            for w in ranges.windows(2) {
                // Sorted by min; require prev.max < next.min.
                if w[0].1 >= w[1].0 {
                    return false;
                }
            }
        }
        true
    }

    /// Phase 6: return references to every SST in `level` whose
    /// `[subject_min, subject_max]` range overlaps the query range
    /// `[range_min, range_max]`. Used by compaction to find
    /// overlapping SSTs at the destination level when promoting
    /// L0 -> L1 (or N -> N+1). Returns an empty Vec if the level
    /// is absent or empty.
    pub fn find_ssts_in_range<'a>(
        &'a self,
        level: u8,
        range_min: &Seed,
        range_max: &Seed,
    ) -> Vec<&'a SSTEntry> {
        let mut out: Vec<&'a SSTEntry> = Vec::new();
        for lvl in &self.levels {
            if lvl.level != level {
                continue;
            }
            for e in &lvl.ssts {
                // Ranges overlap iff each side's max >= the other's min.
                let a_min = &e.subject_min;
                let a_max = &e.subject_max;
                let b_min = range_min.as_bytes();
                let b_max = range_max.as_bytes();
                if a_max.as_slice() >= b_min.as_slice() && a_min.as_slice() <= b_max.as_slice() {
                    out.push(e);
                }
            }
        }
        out
    }

    /// Phase 6: build a NEW manifest where every SST in `old_ids`
    /// is removed and every entry in `new` is added at its
    /// declared level. Used by compaction for the atomic-swap step.
    ///
    /// Returns an owned new `Manifest` (immutable update); the
    /// caller `save_atomic`s it and then swaps the live `Arc`.
    ///
    /// Debug-asserts the level >= 1 disjointness invariant on the
    /// returned manifest.
    pub fn swap_sst_set(
        &self,
        old_ids: &[u64],
        new: &[(u8, SSTEntry)],
    ) -> Manifest {
        let old_set: std::collections::HashSet<u64> = old_ids.iter().copied().collect();
        let mut next = self.clone();
        // Remove old SSTs across all levels.
        for lvl in &mut next.levels {
            lvl.ssts.retain(|e| !old_set.contains(&e.id));
        }
        // Drop now-empty levels (keeps the level vector tidy).
        next.levels.retain(|l| !l.ssts.is_empty());
        // Add new SSTs at their declared levels.
        for (level, entry) in new {
            next.add_sst(*level, entry.clone());
        }
        // Sort SSTs within each level by subject_min so reads can
        // binary-search; required for the disjointness invariant
        // check below to be meaningful.
        for lvl in &mut next.levels {
            lvl.ssts.sort_by(|a, b| a.subject_min.cmp(&b.subject_min));
        }
        debug_assert!(
            next.levels_ge_1_disjoint(),
            "swap_sst_set produced overlapping ranges at level >= 1"
        );
        next
    }

    /// Phase 6: record a compaction event by stamping `last_compaction`
    /// to the supplied timestamp. Compaction writes the new manifest
    /// via `save_atomic` after calling this.
    pub fn record_compaction(&mut self, ts: i64) {
        self.last_compaction = Some(ts);
    }

    /// Phase 6: return SST entries at `level`, sorted by id.
    pub fn ssts_at_level(&self, level: u8) -> Vec<SSTEntry> {
        for lvl in &self.levels {
            if lvl.level == level {
                let mut v = lvl.ssts.clone();
                v.sort_by_key(|e| e.id);
                return v;
            }
        }
        Vec::new()
    }

    /// Phase 6: total size in bytes across `level`.
    pub fn level_size_bytes(&self, level: u8) -> u64 {
        for lvl in &self.levels {
            if lvl.level == level {
                return lvl.ssts.iter().map(|e| e.size).sum();
            }
        }
        0
    }

    /// Phase 6: number of SSTs at `level`.
    pub fn level_sst_count(&self, level: u8) -> usize {
        for lvl in &self.levels {
            if lvl.level == level {
                return lvl.ssts.len();
            }
        }
        0
    }

    /// Phase 6: highest level present in the manifest, or 0 if empty.
    pub fn max_level(&self) -> u8 {
        self.levels.iter().map(|l| l.level).max().unwrap_or(0)
    }

    // ---------------------------------------------------------------------
    // Load / save_atomic. Schema v1, v2, and v3 are all readable;
    // save always writes the current schema (v4).
    // ---------------------------------------------------------------------

    /// Load the manifest at `path`. On checksum mismatch, magic
    /// mismatch, or rkyv validation failure, fall back to
    /// `MANIFEST_PREV_FILE_NAME` in the same directory.
    ///
    /// Returns `Ok(None)` if neither file exists (fresh brain).
    ///
    /// Schema versions in `[1, MANIFEST_SCHEMA_VERSION]` are
    /// accepted; anything else is rejected via
    /// `ManifestError::UnsupportedSchemaVersion` (wrapped into
    /// `BrainError::Format`). Older-schema manifests (v1/v2/v3) are
    /// converted in-memory to the current representation via the
    /// frozen-archetype `From` impls.
    pub fn load(path: &Path) -> Result<Option<Manifest>, BrainError> {
        match Self::load_one(path) {
            Ok(Some(m)) => Ok(Some(m)),
            Ok(None) => {
                // Live manifest missing -- try the .prev fallback.
                let prev = sibling(path, MANIFEST_PREV_FILE_NAME);
                Self::load_one(&prev)
            }
            Err(e) => {
                eprintln!(
                    "[manifest] live manifest at {} failed validation ({}); \
                     attempting MANIFEST.prev rollback",
                    path.display(), e
                );
                let prev = sibling(path, MANIFEST_PREV_FILE_NAME);
                match Self::load_one(&prev) {
                    Ok(Some(m)) => {
                        eprintln!(
                            "[manifest] rolled back to MANIFEST.prev ({} SSTs)",
                            m.total_sst_count()
                        );
                        Ok(Some(m))
                    }
                    Ok(None) => Err(BrainError::Format(format!(
                        "manifest at {} failed validation and MANIFEST.prev does not exist: {}",
                        path.display(), e
                    ))),
                    Err(e2) => Err(BrainError::Format(format!(
                        "manifest at {} failed validation ({}); MANIFEST.prev also failed: {}",
                        path.display(), e, e2
                    ))),
                }
            }
        }
    }

    /// Load a single manifest file. Returns `Ok(None)` if the file
    /// does not exist; bubbles up validation errors otherwise.
    fn load_one(path: &Path) -> Result<Option<Manifest>, BrainError> {
        let mut f = match File::open(path) {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(BrainError::Io(e)),
        };
        let mut all = Vec::new();
        f.read_to_end(&mut all).map_err(BrainError::Io)?;
        if all.len() < MANIFEST_PREFIX_BYTES + MANIFEST_TRAILER_BYTES {
            return Err(BrainError::Format(format!(
                "manifest {} too short ({} bytes)",
                path.display(), all.len()
            )));
        }
        let magic = u32::from_le_bytes(all[0..4].try_into().unwrap());
        if magic != MANIFEST_MAGIC {
            return Err(BrainError::Format(format!(
                "manifest {} magic mismatch: got 0x{:08x}, expected 0x{:08x}",
                path.display(), magic, MANIFEST_MAGIC
            )));
        }
        let schema = u32::from_le_bytes(all[4..8].try_into().unwrap());
        if schema < MANIFEST_MIN_READ_SCHEMA_VERSION || schema > MANIFEST_SCHEMA_VERSION {
            // Hard error -- no silent fallback. The brief is
            // explicit on this: "NOT a silent fallback."
            return Err(BrainError::from(ManifestError::UnsupportedSchemaVersion {
                found: schema,
                min_supported: MANIFEST_MIN_READ_SCHEMA_VERSION,
                max_supported: MANIFEST_SCHEMA_VERSION,
            }));
        }
        let payload_len = u32::from_le_bytes(all[8..12].try_into().unwrap()) as usize;
        let body_end = MANIFEST_PREFIX_BYTES + payload_len;
        if all.len() != body_end + MANIFEST_TRAILER_BYTES {
            return Err(BrainError::Format(format!(
                "manifest {} length mismatch: prefix says payload={} (total={}), file is {}",
                path.display(), payload_len, body_end + MANIFEST_TRAILER_BYTES, all.len()
            )));
        }
        let stored_crc = u32::from_le_bytes(all[body_end..body_end + 4].try_into().unwrap());
        let actual_crc = crc32fast::hash(&all[..body_end]);
        if stored_crc != actual_crc {
            return Err(BrainError::Format(format!(
                "manifest {} CRC mismatch: stored 0x{:08x}, actual 0x{:08x}",
                path.display(), stored_crc, actual_crc
            )));
        }
        // rkyv::check_archived_root requires the slice to be
        // 8-byte aligned in memory. `Vec<u8>` allocations are not
        // guaranteed aligned beyond `align_of::<u8>() == 1`, even
        // when the file offset is 16-aligned. Copy into rkyv's
        // `AlignedVec` which DOES guarantee alignment.
        let mut aligned = rkyv::AlignedVec::with_capacity(payload_len);
        aligned.extend_from_slice(&all[MANIFEST_PREFIX_BYTES..body_end]);
        let body: &[u8] = aligned.as_slice();

        // Dispatch on schema_version. v1/v2/v3 promotion happens
        // entirely in-memory after the rkyv read; the on-disk file
        // remains untouched until the next save_atomic.
        let payload: ManifestPayload = match schema {
            1 => {
                let archived = rkyv::check_archived_root::<ManifestPayloadV1>(body).map_err(|e| {
                    BrainError::Format(format!(
                        "manifest {} v1 rkyv validation failed: {}",
                        path.display(), e
                    ))
                })?;
                let v1: ManifestPayloadV1 = archived
                    .deserialize(&mut rkyv::Infallible)
                    .map_err(|e| BrainError::Format(format!(
                        "manifest {} v1 rkyv deserialize failed: {:?}",
                        path.display(), e
                    )))?;
                ManifestPayload::from(v1)
            }
            2 => {
                // Phase 3.3: schema v2 is now a frozen archetype.
                // Read via `ManifestPayloadV2`, convert in-memory
                // to v3 `ManifestPayload`. The on-disk file is
                // left untouched until the next save_atomic.
                let archived = rkyv::check_archived_root::<ManifestPayloadV2>(body).map_err(|e| {
                    BrainError::Format(format!(
                        "manifest {} v2 rkyv validation failed: {}",
                        path.display(), e
                    ))
                })?;
                let v2: ManifestPayloadV2 = archived
                    .deserialize(&mut rkyv::Infallible)
                    .map_err(|e| BrainError::Format(format!(
                        "manifest {} v2 rkyv deserialize failed: {:?}",
                        path.display(), e
                    )))?;
                ManifestPayload::from(v2)
            }
            3 => {
                // Phase 3.5: schema v3 is now a frozen archetype.
                // Read via `ManifestPayloadV3`, convert in-memory
                // to v4 `ManifestPayload` (reverse_file_uid = None
                // on every promoted BondSSTEntry). The on-disk
                // file is left untouched until the next save_atomic.
                let archived = rkyv::check_archived_root::<ManifestPayloadV3>(body).map_err(|e| {
                    BrainError::Format(format!(
                        "manifest {} v3 rkyv validation failed: {}",
                        path.display(), e
                    ))
                })?;
                let v3: ManifestPayloadV3 = archived
                    .deserialize(&mut rkyv::Infallible)
                    .map_err(|e| BrainError::Format(format!(
                        "manifest {} v3 rkyv deserialize failed: {:?}",
                        path.display(), e
                    )))?;
                ManifestPayload::from(v3)
            }
            4 => {
                // OP-BRAIN-V5 6.1: schema v4 is now a frozen archetype.
                // Read via `ManifestPayloadV4` (no format_tag) and promote
                // to the current ManifestPayload with format_tag = v4
                // (legacy brains predate the explicit tag). The on-disk
                // file is left untouched until the next save_atomic.
                let archived = rkyv::check_archived_root::<ManifestPayloadV4>(body).map_err(|e| {
                    BrainError::Format(format!(
                        "manifest {} v4 rkyv validation failed: {}",
                        path.display(), e
                    ))
                })?;
                let v4: ManifestPayloadV4 = archived
                    .deserialize(&mut rkyv::Infallible)
                    .map_err(|e| BrainError::Format(format!(
                        "manifest {} v4 rkyv deserialize failed: {:?}",
                        path.display(), e
                    )))?;
                ManifestPayload::from(v4)
            }
            5 => {
                let archived = rkyv::check_archived_root::<ManifestPayload>(body).map_err(|e| {
                    BrainError::Format(format!(
                        "manifest {} v5 rkyv validation failed: {}",
                        path.display(), e
                    ))
                })?;
                archived
                    .deserialize(&mut rkyv::Infallible)
                    .map_err(|e| BrainError::Format(format!(
                        "manifest {} v5 rkyv deserialize failed: {:?}",
                        path.display(), e
                    )))?
            }
            // The gate above already restricts schema; this arm is
            // belt-and-suspenders and structurally unreachable.
            other => unreachable!(
                "manifest schema {} passed the gate but isn't handled", other
            ),
        };

        let memtable_wal = if payload.memtable_wal.is_empty() {
            None
        } else {
            Some(PathBuf::from(payload.memtable_wal))
        };
        let last_compaction = if payload.last_compaction == 0 {
            None
        } else {
            Some(payload.last_compaction)
        };
        Ok(Some(Manifest {
            // Preserve the on-disk schema in the snapshot so test
            // code can inspect which version the file carried. The
            // next save_atomic always writes MANIFEST_SCHEMA_VERSION.
            schema_version: schema,
            master_key_id: payload.master_key_id,
            levels: payload.levels,
            memtable_wal,
            last_compaction,
            next_sst_id: payload.next_sst_id.max(1),
            node_levels: payload.node_levels,
            next_node_sst_uid: payload.next_node_sst_uid,
            next_epoch: payload.next_epoch,
            bond_v5_levels: payload.bond_v5_levels,
            next_bond_v5_sst_uid: payload.next_bond_v5_sst_uid,
            format_tag: payload.format_tag,
        }))
    }

    /// Atomically write `self` to `path`. Steps:
    ///
    /// 1. Write to `<dir>/MANIFEST.tmp.<pid>.<nonce>`, fsync.
    /// 2. Rename existing `MANIFEST` to `MANIFEST.prev` (if any).
    /// 3. Rename the `.tmp` to `MANIFEST`.
    /// 4. Fsync the parent directory.
    ///
    /// Returns only after step 4 completes. ALWAYS writes the
    /// current schema_version (`MANIFEST_SCHEMA_VERSION`) regardless
    /// of which schema the source file had -- older-schema manifests
    /// are auto-migrated to the current schema on the first save
    /// after load.
    pub fn save_atomic(&self, path: &Path) -> Result<(), BrainError> {
        let payload = ManifestPayload {
            master_key_id: self.master_key_id,
            levels: self.levels.clone(),
            memtable_wal: self
                .memtable_wal
                .as_ref()
                .map(|p| p.to_string_lossy().into_owned())
                .unwrap_or_default(),
            last_compaction: self.last_compaction.unwrap_or(0),
            next_sst_id: self.next_sst_id,
            node_levels: self.node_levels.clone(),
            next_node_sst_uid: self.next_node_sst_uid,
            next_epoch: self.next_epoch,
            bond_v5_levels: self.bond_v5_levels.clone(),
            next_bond_v5_sst_uid: self.next_bond_v5_sst_uid,
            format_tag: self.format_tag,
        };
        let payload_bytes = rkyv::to_bytes::<_, 4096>(&payload).map_err(|e| {
            BrainError::Format(format!("manifest rkyv serialize: {}", e))
        })?;
        let payload_len = payload_bytes.len();
        if payload_len > u32::MAX as usize {
            return Err(BrainError::Format(format!(
                "manifest payload too large: {} > u32::MAX", payload_len
            )));
        }
        let total = MANIFEST_PREFIX_BYTES + payload_len + MANIFEST_TRAILER_BYTES;
        let mut buf = Vec::with_capacity(total);
        buf.extend_from_slice(&MANIFEST_MAGIC.to_le_bytes());
        buf.extend_from_slice(&MANIFEST_SCHEMA_VERSION.to_le_bytes());
        buf.extend_from_slice(&(payload_len as u32).to_le_bytes());
        // 4-byte reserved padding so the rkyv payload that follows
        // starts at offset 16 (8-byte aligned), satisfying
        // `rkyv::check_archived_root`'s alignment requirement.
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&payload_bytes);
        let crc = crc32fast::hash(&buf);
        buf.extend_from_slice(&crc.to_le_bytes());
        debug_assert_eq!(buf.len(), total);

        // Choose a tmp path co-located with the destination so the
        // rename stays on the same filesystem (cross-fs rename = EXDEV).
        let nonce = TMP_NONCE.fetch_add(1, Ordering::Relaxed);
        let pid = std::process::id();
        let tmp_name = format!("MANIFEST.tmp.{}.{}", pid, nonce);
        let parent = path.parent().ok_or_else(|| BrainError::Format(format!(
            "manifest path {} has no parent directory", path.display()
        )))?;
        let tmp_path = parent.join(&tmp_name);

        // 1. Write the tmp, fsync.
        {
            let mut f = OpenOptions::new()
                .create(true).write(true).truncate(true)
                .open(&tmp_path).map_err(BrainError::Io)?;
            f.write_all(&buf).map_err(BrainError::Io)?;
            f.sync_all().map_err(BrainError::Io)?;
        }

        // 2. Rotate existing MANIFEST -> MANIFEST.prev.
        let prev_path = sibling(path, MANIFEST_PREV_FILE_NAME);
        match fs::metadata(path) {
            Ok(_) => {
                // Remove a stale .prev if one exists (rename refuses
                // to overwrite on some filesystems).
                let _ = fs::remove_file(&prev_path);
                fs::rename(path, &prev_path).map_err(BrainError::Io)?;
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(BrainError::Io(e)),
        }

        // 3. Promote tmp -> MANIFEST.
        fs::rename(&tmp_path, path).map_err(|e| {
            BrainError::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("manifest rename {} -> {}: {}",
                    tmp_path.display(), path.display(), e),
            ))
        })?;

        // 4. Fsync the parent directory so the rename(s) are durable.
        fsync_dir(parent)?;
        Ok(())
    }
}

/// Process-local nonce so concurrent `save_atomic` calls on the same
/// process never collide on the tmp filename. (In Phase 4 only one
/// thread writes at a time; this is belt-and-suspenders.)
static TMP_NONCE: AtomicU32 = AtomicU32::new(0);

fn sibling(path: &Path, name: &str) -> PathBuf {
    let parent = path.parent().map(|p| p.to_path_buf()).unwrap_or_else(|| PathBuf::from("."));
    parent.join(name)
}

/// fsync a directory so any renames inside it are durable. On
/// non-POSIX targets `sync_data` on a directory handle may not be
/// meaningful; treat the error as best-effort.
pub(crate) fn fsync_dir(dir: &Path) -> Result<(), BrainError> {
    match File::open(dir) {
        Ok(d) => {
            // sync_all returns EINVAL on some filesystems for
            // directory handles even though the rename IS durable;
            // treat as best-effort.
            let _ = d.sync_all();
            Ok(())
        }
        Err(e) => Err(BrainError::Io(e)),
    }
}

/// Clean up orphan SST files in `sst_dir` not referenced by `manifest`.
///
/// Walks the directory looking for `L*.sst` files; any whose name
/// (relative to `sst_dir`) is not present in the manifest's SSTEntry
/// set is renamed to `<name>.orphan-<timestamp>` and a log line is
/// emitted. We rename rather than delete so the operator can
/// inspect orphans before GC'ing them.
///
/// Returns the list of orphan paths (post-rename).
pub fn cleanup_orphan_ssts(
    sst_dir: &Path,
    manifest: &Manifest,
) -> Result<Vec<PathBuf>, BrainError> {
    let referenced: std::collections::HashSet<String> = manifest
        .iter_ssts()
        .map(|(_, e)| e.file.clone())
        .collect();
    let mut orphans = Vec::new();
    let rd = match fs::read_dir(sst_dir) {
        Ok(r) => r,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(orphans),
        Err(e) => return Err(BrainError::Io(e)),
    };
    for entry in rd {
        let entry = entry.map_err(BrainError::Io)?;
        let path = entry.path();
        let name = match path.file_name().and_then(|n| n.to_str()) {
            Some(n) => n.to_string(),
            None => continue,
        };
        // Match L<digits>-<digits>.sst per the Phase 4 naming
        // convention. Other files (manifests, WAL, .orphan, .tmp)
        // are left alone.
        if !is_sst_filename(&name) {
            continue;
        }
        if referenced.contains(&name) {
            continue;
        }
        // Orphan -- rename for inspection.
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let orphan_name = format!("{}.orphan-{}", name, ts);
        let orphan_path = sst_dir.join(&orphan_name);
        fs::rename(&path, &orphan_path).map_err(BrainError::Io)?;
        eprintln!(
            "[manifest] orphan SST {} (not in manifest after restart) renamed to {}",
            name, orphan_name
        );
        orphans.push(orphan_path);
    }
    Ok(orphans)
}

/// True iff `name` looks like a flush-produced SST filename:
/// `L<digits>-<digits>.sst`.
fn is_sst_filename(name: &str) -> bool {
    if !name.ends_with(".sst") {
        return false;
    }
    let stem = &name[..name.len() - 4];
    let bytes = stem.as_bytes();
    if bytes.is_empty() || bytes[0] != b'L' {
        return false;
    }
    let rest = &stem[1..];
    let dash = match rest.find('-') {
        Some(i) => i,
        None => return false,
    };
    let level_part = &rest[..dash];
    let id_part = &rest[dash + 1..];
    if level_part.is_empty() || id_part.is_empty() {
        return false;
    }
    level_part.chars().all(|c| c.is_ascii_digit())
        && id_part.chars().all(|c| c.is_ascii_digit())
}

/// Construct the conventional filename for an SST at the given
/// level + id: `L<level>-<id>.sst`.
pub fn sst_filename(level: u8, id: u64) -> String {
    format!("L{}-{}.sst", level, id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn entry(id: u64, level: u8, smin: u8, smax: u8, bond_count: u64) -> SSTEntry {
        SSTEntry {
            id,
            file: sst_filename(level, id),
            subject_min: [smin; 32],
            subject_max: [smax; 32],
            size: 1024,
            mtime: 1_700_000_000,
            bond_count,
            // Phase 2.5: tests that don't care about epoch use 0.
            epoch: 0,
        }
    }

    fn node_entry(file_uid: u64, first: u64, last: u64, count: u64, size: u64, epoch: u64) -> NodeSSTEntry {
        NodeSSTEntry {
            file_uid,
            first_node_id: first,
            last_node_id: last,
            entry_count: count,
            file_size: size,
            epoch,
        }
    }

    #[test]
    fn round_trip_empty() {
        let dir = tempdir().unwrap();
        let path = dir.path().join(MANIFEST_FILE_NAME);
        let m = Manifest::new(KeyFingerprint::none());
        m.save_atomic(&path).unwrap();
        let loaded = Manifest::load(&path).unwrap().expect("loaded");
        assert_eq!(loaded.total_sst_count(), 0);
        assert_eq!(loaded.schema_version, MANIFEST_SCHEMA_VERSION);
        assert_eq!(loaded.master_key_id, KeyFingerprint::none());
        // Phase 2.5: fresh manifest has empty node levels and zeroed counters.
        assert_eq!(loaded.total_node_sst_count(), 0);
        assert_eq!(loaded.peek_next_node_sst_uid(), 0);
        assert_eq!(loaded.peek_next_epoch(), 0);
    }

    #[test]
    fn round_trip_with_ssts() {
        let dir = tempdir().unwrap();
        let path = dir.path().join(MANIFEST_FILE_NAME);
        let mut m = Manifest::new(KeyFingerprint::none());
        let id1 = m.next_sst_id();
        let id2 = m.next_sst_id();
        m.add_sst(0, entry(id1, 0, 0x10, 0x20, 100));
        m.add_sst(0, entry(id2, 0, 0x30, 0x40, 200));
        m.save_atomic(&path).unwrap();
        let loaded = Manifest::load(&path).unwrap().expect("loaded");
        assert_eq!(loaded.total_sst_count(), 2);
        let all: Vec<_> = loaded.iter_ssts().collect();
        assert_eq!(all[0].1.bond_count, 100);
        assert_eq!(all[1].1.bond_count, 200);
        // next_sst_id preserved across the round-trip.
        assert_eq!(loaded.peek_next_sst_id(), 3);
    }

    #[test]
    fn save_rotates_prev() {
        let dir = tempdir().unwrap();
        let path = dir.path().join(MANIFEST_FILE_NAME);
        let prev = dir.path().join(MANIFEST_PREV_FILE_NAME);
        let m1 = Manifest::new(KeyFingerprint::none());
        m1.save_atomic(&path).unwrap();
        assert!(path.exists());
        assert!(!prev.exists());

        let mut m2 = Manifest::new(KeyFingerprint::none());
        let id = m2.next_sst_id();
        m2.add_sst(0, entry(id, 0, 0, 0xFF, 1));
        m2.save_atomic(&path).unwrap();
        assert!(path.exists());
        assert!(prev.exists());

        // Loading the live file returns the m2 view.
        let live = Manifest::load(&path).unwrap().expect("live");
        assert_eq!(live.total_sst_count(), 1);
    }

    #[test]
    fn corrupt_live_falls_back_to_prev() {
        let dir = tempdir().unwrap();
        let path = dir.path().join(MANIFEST_FILE_NAME);
        let m1 = Manifest::new(KeyFingerprint::none());
        m1.save_atomic(&path).unwrap();
        let mut m2 = Manifest::new(KeyFingerprint::none());
        let id = m2.next_sst_id();
        m2.add_sst(0, entry(id, 0, 0, 0xFF, 1));
        m2.save_atomic(&path).unwrap();
        // Corrupt the live MANIFEST by flipping a CRC byte.
        let mut bad = std::fs::read(&path).unwrap();
        let last_idx = bad.len() - 1;
        bad[last_idx] ^= 0xFF;
        std::fs::write(&path, &bad).unwrap();
        // Load should fall back to MANIFEST.prev (m1 = 0 SSTs).
        let loaded = Manifest::load(&path).unwrap().expect("prev");
        assert_eq!(loaded.total_sst_count(), 0);
    }

    #[test]
    fn orphan_cleanup_renames_unknown_ssts() {
        let dir = tempdir().unwrap();
        let sst_dir = dir.path();
        let mut m = Manifest::new(KeyFingerprint::none());
        let id = m.next_sst_id();
        m.add_sst(0, entry(id, 0, 0, 1, 1));
        // Create the referenced file + an orphan.
        std::fs::write(sst_dir.join(sst_filename(0, id)), b"referenced").unwrap();
        std::fs::write(sst_dir.join(sst_filename(0, 999)), b"orphan").unwrap();
        // Also drop a non-SST file that must be left alone.
        std::fs::write(sst_dir.join("MANIFEST"), b"not an sst").unwrap();
        let orphans = cleanup_orphan_ssts(sst_dir, &m).unwrap();
        assert_eq!(orphans.len(), 1);
        assert!(sst_dir.join(sst_filename(0, id)).exists());
        assert!(!sst_dir.join(sst_filename(0, 999)).exists());
        assert!(sst_dir.join("MANIFEST").exists());
    }

    #[test]
    fn sst_filename_pattern_matches() {
        assert!(is_sst_filename("L0-1.sst"));
        assert!(is_sst_filename("L12-9999.sst"));
        assert!(!is_sst_filename("L0-1.sst.orphan-123"));
        assert!(!is_sst_filename("MANIFEST"));
        assert!(!is_sst_filename("foo.sst"));
        assert!(!is_sst_filename("L-1.sst"));
        assert!(!is_sst_filename("L0-.sst"));
    }

    #[test]
    fn add_sst_creates_levels_in_order() {
        let mut m = Manifest::new(KeyFingerprint::none());
        m.add_sst(2, entry(1, 2, 0, 0, 1));
        m.add_sst(0, entry(2, 0, 0, 0, 1));
        m.add_sst(1, entry(3, 1, 0, 0, 1));
        let levels: Vec<u8> = m.levels.iter().map(|l| l.level).collect();
        assert_eq!(levels, vec![0, 1, 2]);
    }

    #[test]
    fn key_fingerprint_changes_with_key() {
        let k1 = MasterKey::from_bytes([0u8; 32]);
        let k2 = MasterKey::from_bytes([1u8; 32]);
        assert_ne!(KeyFingerprint::of(&k1), KeyFingerprint::of(&k2));
        assert_ne!(KeyFingerprint::of(&k1), KeyFingerprint::none());
    }

    // ---------------------------------------------------------------------
    // OP-BRAIN-V5 Phase 2.5 (Q7) tests.
    // ---------------------------------------------------------------------

    /// Helper: write a manifest file with the v1 wire shape (uses
    /// the frozen `ManifestPayloadV1` archetypes). Mirrors the
    /// `save_atomic` byte layout but with schema_version=1.
    fn write_v1_manifest_file(
        path: &Path,
        payload: ManifestPayloadV1,
    ) -> std::io::Result<()> {
        let payload_bytes = rkyv::to_bytes::<_, 4096>(&payload)
            .expect("v1 rkyv serialize");
        let payload_len = payload_bytes.len();
        let total = MANIFEST_PREFIX_BYTES + payload_len + MANIFEST_TRAILER_BYTES;
        let mut buf = Vec::with_capacity(total);
        buf.extend_from_slice(&MANIFEST_MAGIC.to_le_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes()); // schema_version = 1
        buf.extend_from_slice(&(payload_len as u32).to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes()); // reserved
        buf.extend_from_slice(&payload_bytes);
        let crc = crc32fast::hash(&buf);
        buf.extend_from_slice(&crc.to_le_bytes());
        std::fs::write(path, &buf)
    }

    /// Phase 2.5 Test A. v1 -> v2 backwards-compat read.
    ///
    /// Build a v1-shape manifest on disk (schema_version baked at 1,
    /// no node_levels / next_node_sst_uid / next_epoch in the rkyv
    /// blob), load it via the v2 reader, assert:
    ///   - load succeeds,
    ///   - in-memory `schema_version` reflects the on-disk value (1),
    ///   - node_levels == empty, next_node_sst_uid == 0,
    ///     next_epoch == 0,
    ///   - every promoted SSTEntry has `epoch == 0`,
    /// Then save_atomic and assert the next file is v2.
    #[test]
    fn phase25_v1_backwards_compat_read() {
        let dir = tempdir().unwrap();
        let path = dir.path().join(MANIFEST_FILE_NAME);

        // Construct a v1 payload with two SSTs.
        let v1 = ManifestPayloadV1 {
            master_key_id: KeyFingerprint::none(),
            levels: vec![LevelEntryV1 {
                level: 0,
                ssts: vec![
                    SSTEntryV1 {
                        id: 1,
                        file: sst_filename(0, 1),
                        subject_min: [0x10; 32],
                        subject_max: [0x20; 32],
                        size: 1024,
                        mtime: 1_700_000_000,
                        bond_count: 42,
                    },
                    SSTEntryV1 {
                        id: 2,
                        file: sst_filename(0, 2),
                        subject_min: [0x30; 32],
                        subject_max: [0x40; 32],
                        size: 2048,
                        mtime: 1_700_000_500,
                        bond_count: 84,
                    },
                ],
            }],
            memtable_wal: String::new(),
            last_compaction: 0,
            next_sst_id: 3,
        };
        write_v1_manifest_file(&path, v1).unwrap();

        // Sanity: the on-disk schema byte is 1.
        let raw = std::fs::read(&path).unwrap();
        let on_disk_schema = u32::from_le_bytes(raw[4..8].try_into().unwrap());
        assert_eq!(on_disk_schema, 1);

        // Load via the current (v2) reader. Must succeed.
        let loaded = Manifest::load(&path).unwrap().expect("v1 manifest loads");
        assert_eq!(loaded.schema_version, 1, "schema_version snapshot reflects on-disk version");
        assert_eq!(loaded.total_sst_count(), 2);
        assert_eq!(loaded.node_levels.len(), 0);
        assert_eq!(loaded.peek_next_node_sst_uid(), 0);
        assert_eq!(loaded.peek_next_epoch(), 0);
        // Every promoted SSTEntry has epoch=0 (v1 default).
        for (_lvl, e) in loaded.iter_ssts() {
            assert_eq!(e.epoch, 0, "v1 -> v2 default epoch must be 0");
        }

        // Next save_atomic writes v2.
        loaded.save_atomic(&path).unwrap();
        let raw2 = std::fs::read(&path).unwrap();
        let new_schema = u32::from_le_bytes(raw2[4..8].try_into().unwrap());
        assert_eq!(new_schema, MANIFEST_SCHEMA_VERSION,
            "first save after v1 load must promote on-disk schema to v2");

        // And the freshly-saved current-schema manifest round-trips back cleanly.
        let reloaded = Manifest::load(&path).unwrap().expect("current-schema round-trip");
        assert_eq!(reloaded.schema_version, MANIFEST_SCHEMA_VERSION);
        assert_eq!(reloaded.total_sst_count(), 2);
        for (_lvl, e) in reloaded.iter_ssts() {
            assert_eq!(e.epoch, 0);
        }
    }

    /// Phase 2.5 Test B. v2 roundtrip.
    ///
    /// Build a v2 payload with 2 node_levels, 3 NodeSSTEntries
    /// (varied epochs), 2 existing bond SSTEntries (also varied
    /// epochs). save_atomic, load, assert byte-identical
    /// reconstruction (field-by-field).
    #[test]
    fn phase25_v2_roundtrip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join(MANIFEST_FILE_NAME);
        let mut m = Manifest::new(KeyFingerprint::none());

        // Bond side: two SSTs at different epochs.
        let mut e0 = entry(m.next_sst_id(), 0, 0x10, 0x20, 100);
        e0.epoch = 7;
        m.add_sst(0, e0);
        let mut e1 = entry(m.next_sst_id(), 1, 0x30, 0x40, 250);
        e1.epoch = 11;
        m.add_sst(1, e1);

        // Advance the next_epoch counter so the roundtrip has
        // something non-default to preserve.
        for _ in 0..15 { let _ = m.allocate_epoch(); }

        // Node side: two levels, three NodeSSTEntries.
        let n0 = node_entry(m.allocate_node_sst_uid(), 0, 99, 100, 4096, 3);
        m.add_node_sst(0, n0.clone()).unwrap();
        let n1 = node_entry(m.allocate_node_sst_uid(), 100, 199, 100, 4096, 5);
        m.add_node_sst(0, n1.clone()).unwrap();
        let n2 = node_entry(m.allocate_node_sst_uid(), 0, 199, 200, 8192, 12);
        m.add_node_sst(1, n2.clone()).unwrap();

        let next_uid_before = m.peek_next_node_sst_uid();
        let next_epoch_before = m.peek_next_epoch();
        let next_sst_id_before = m.peek_next_sst_id();

        m.save_atomic(&path).unwrap();
        let reloaded = Manifest::load(&path).unwrap().expect("v2 roundtrip");

        assert_eq!(reloaded.schema_version, MANIFEST_SCHEMA_VERSION);
        assert_eq!(reloaded.total_sst_count(), 2);
        assert_eq!(reloaded.total_node_sst_count(), 3);
        assert_eq!(reloaded.node_levels.len(), 2);
        assert_eq!(reloaded.peek_next_node_sst_uid(), next_uid_before);
        assert_eq!(reloaded.peek_next_epoch(), next_epoch_before);
        assert_eq!(reloaded.peek_next_sst_id(), next_sst_id_before);

        // Bond SSTs preserve their epochs.
        let bond_epochs: Vec<u64> = reloaded.iter_ssts().map(|(_, e)| e.epoch).collect();
        assert!(bond_epochs.contains(&7));
        assert!(bond_epochs.contains(&11));

        // Node SSTs preserve field-by-field. iter_node_ssts is
        // ordered by level ASC then by insertion order within level.
        let observed: Vec<NodeSSTEntry> = reloaded.iter_node_ssts().map(|(_, e)| e.clone()).collect();
        assert_eq!(observed.len(), 3);
        assert!(observed.contains(&n0));
        assert!(observed.contains(&n1));
        assert!(observed.contains(&n2));
    }

    /// Phase 2.5 Test C. add_node_sst / retire_node_ssts pair.
    ///
    /// Add 3 NodeSST entries via the helper, save, reload, retire 1
    /// by file_uid. Assert that file_uid is gone, the other 2
    /// remain, retire returns the removed NodeSSTEntry. retire on a
    /// non-existent file_uid returns Err.
    #[test]
    fn phase25_add_and_retire_node_sst() {
        let dir = tempdir().unwrap();
        let path = dir.path().join(MANIFEST_FILE_NAME);
        let mut m = Manifest::new(KeyFingerprint::none());

        let e0 = node_entry(m.allocate_node_sst_uid(), 0, 99, 100, 1024, m.allocate_epoch());
        let e1 = node_entry(m.allocate_node_sst_uid(), 100, 199, 100, 1024, m.allocate_epoch());
        let e2 = node_entry(m.allocate_node_sst_uid(), 200, 299, 100, 1024, m.allocate_epoch());
        let target_uid = e1.file_uid;

        m.add_node_sst(0, e0.clone()).unwrap();
        m.add_node_sst(0, e1.clone()).unwrap();
        m.add_node_sst(0, e2.clone()).unwrap();

        m.save_atomic(&path).unwrap();
        let mut reloaded = Manifest::load(&path).unwrap().expect("reload");
        assert_eq!(reloaded.total_node_sst_count(), 3);

        let retired = reloaded.retire_node_ssts(&[target_uid]).unwrap();
        assert_eq!(retired.len(), 1);
        assert_eq!(retired[0].file_uid, target_uid);
        assert_eq!(retired[0], e1);

        // The other 2 entries remain.
        let remaining_uids: Vec<u64> = reloaded.live_node_sst_uids();
        assert_eq!(remaining_uids, vec![e0.file_uid, e2.file_uid]);

        // retire on a non-existent file_uid is Err.
        let bogus = 0xDEAD_BEEF_u64;
        let err = reloaded.retire_node_ssts(&[bogus]).unwrap_err();
        match err {
            ManifestError::UnknownNodeSstUid(u) => assert_eq!(u, bogus),
            other => panic!("expected UnknownNodeSstUid, got {:?}", other),
        }
        // Failed retire is all-or-nothing -- remaining entries unchanged.
        assert_eq!(reloaded.live_node_sst_uids(), vec![e0.file_uid, e2.file_uid]);

        // Duplicate add returns Err and leaves state unchanged.
        let dup = NodeSSTEntry { file_uid: e0.file_uid, ..e0.clone() };
        let err = reloaded.add_node_sst(0, dup).unwrap_err();
        match err {
            ManifestError::DuplicateNodeSstUid(u) => assert_eq!(u, e0.file_uid),
            other => panic!("expected DuplicateNodeSstUid, got {:?}", other),
        }
        assert_eq!(reloaded.total_node_sst_count(), 2);
    }

    /// Phase 2.5 Test D. allocate_node_sst_uid + allocate_epoch monotonicity.
    ///
    /// 100 calls each. Assert returned values are 0..99 and strictly
    /// increasing. next_node_sst_uid and next_epoch end at 100.
    #[test]
    fn phase25_allocator_monotonicity() {
        let mut m = Manifest::new(KeyFingerprint::none());
        let mut uids = Vec::with_capacity(100);
        let mut epochs = Vec::with_capacity(100);
        for _ in 0..100 {
            uids.push(m.allocate_node_sst_uid());
            epochs.push(m.allocate_epoch());
        }
        let expected: Vec<u64> = (0..100u64).collect();
        assert_eq!(uids, expected, "uids must be 0..99");
        assert_eq!(epochs, expected, "epochs must be 0..99");
        // Strictly increasing.
        for w in uids.windows(2) { assert!(w[1] > w[0]); }
        for w in epochs.windows(2) { assert!(w[1] > w[0]); }
        assert_eq!(m.peek_next_node_sst_uid(), 100);
        assert_eq!(m.peek_next_epoch(), 100);
    }

    /// Phase 2.5 Test E. Unsupported schema version is a hard error.
    ///
    /// Hand-craft a manifest with schema_version=99 and assert load()
    /// returns an error mentioning UnsupportedSchemaVersion. NOT a
    /// silent fallback.
    #[test]
    fn format_tag_v5_round_trips() {
        // OP-BRAIN-V5 6.1: set_brain_format_v5 -> save_atomic -> load
        // preserves the explicit v5 tag, and the on-disk schema byte is
        // bumped to 5.
        let dir = tempdir().unwrap();
        let path = dir.path().join(MANIFEST_FILE_NAME);
        let mut m = Manifest::new(KeyFingerprint::none());
        assert!(!m.is_v5_format(), "fresh manifest defaults to the v4 tag");
        m.set_brain_format_v5();
        m.save_atomic(&path).unwrap();
        let raw = std::fs::read(&path).unwrap();
        assert_eq!(
            u32::from_le_bytes(raw[4..8].try_into().unwrap()),
            MANIFEST_SCHEMA_VERSION
        );
        assert_eq!(MANIFEST_SCHEMA_VERSION, 5);
        let loaded = Manifest::load(&path).unwrap().expect("manifest loads");
        assert!(loaded.is_v5_format(), "v5 tag survives save+load");
        assert_eq!(loaded.format_tag, BRAIN_FORMAT_TAG_V5);
    }

    #[test]
    fn phase25_unsupported_schema_version_is_hard_error() {
        let dir = tempdir().unwrap();
        let path = dir.path().join(MANIFEST_FILE_NAME);

        // Write a manifest with a bogus schema_version. We use a
        // valid v2 body but stamp the schema_version byte to 99.
        let m = Manifest::new(KeyFingerprint::none());
        m.save_atomic(&path).unwrap();
        let mut bytes = std::fs::read(&path).unwrap();
        bytes[4..8].copy_from_slice(&99u32.to_le_bytes());
        // Recompute CRC over the prefix + payload so the load
        // doesn't fall through to a CRC-mismatch error first --
        // we want the load to specifically reject on schema, not on
        // checksum.
        let body_end = bytes.len() - 4;
        let crc = crc32fast::hash(&bytes[..body_end]);
        bytes[body_end..body_end + 4].copy_from_slice(&crc.to_le_bytes());
        std::fs::write(&path, &bytes).unwrap();

        // Load should fail. Live manifest fails the schema gate,
        // and the prev file does not exist, so the load surfaces an
        // error.
        let res = Manifest::load(&path);
        match res {
            Ok(_) => panic!("load of schema_version=99 must error"),
            Err(BrainError::Format(s)) => {
                assert!(
                    s.contains("99") || s.contains("schema version"),
                    "error message must mention the bad schema; got: {}", s
                );
            }
            Err(other) => panic!("expected Format error, got {:?}", other),
        }
    }

    /// Phase 2.5 Test F. add_node_sst into invalid level errors.
    ///
    /// Calling with level > node_levels.len() returns Err. (Equal is
    /// accepted because it creates the next level in sequence;
    /// strictly greater is not.)
    #[test]
    fn phase25_add_node_sst_invalid_level_errors() {
        let mut m = Manifest::new(KeyFingerprint::none());
        // node_levels is empty.
        let e0 = node_entry(m.allocate_node_sst_uid(), 0, 99, 100, 1024, 0);
        // level=1 with empty node_levels jumps past 0 -- must Err.
        let err = m.add_node_sst(1, e0.clone()).unwrap_err();
        match err {
            ManifestError::InvalidLevel { requested, current_len } => {
                assert_eq!(requested, 1);
                assert_eq!(current_len, 0);
            }
            other => panic!("expected InvalidLevel, got {:?}", other),
        }
        assert_eq!(m.total_node_sst_count(), 0);

        // level=0 is accepted -- creates L0.
        m.add_node_sst(0, e0).unwrap();
        assert_eq!(m.total_node_sst_count(), 1);

        // level=2 with one level present (len=1) is still > len, Err.
        let e1 = node_entry(m.allocate_node_sst_uid(), 100, 199, 100, 1024, 0);
        let err = m.add_node_sst(2, e1.clone()).unwrap_err();
        match err {
            ManifestError::InvalidLevel { requested, current_len } => {
                assert_eq!(requested, 2);
                assert_eq!(current_len, 1);
            }
            other => panic!("expected InvalidLevel, got {:?}", other),
        }

        // level=1 (==len) is accepted -- grows to L1.
        m.add_node_sst(1, e1).unwrap();
        assert_eq!(m.node_levels.len(), 2);
    }

    /// Phase 2.5 Test G. live_node_sst_uids round-trip.
    ///
    /// Add 3 entries with file_uids 1, 5, 7 (deliberately not the
    /// 0..2 allocator default). live_node_sst_uids returns sorted
    /// [1, 5, 7]. Save + reload preserves the same enumeration.
    #[test]
    fn phase25_live_node_sst_uids_roundtrip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join(MANIFEST_FILE_NAME);
        let mut m = Manifest::new(KeyFingerprint::none());

        // We don't go through allocate_node_sst_uid here -- the test
        // explicitly constructs uids 1, 5, 7 (sparse).
        let e1 = node_entry(1, 0, 0, 1, 0, 0);
        let e5 = node_entry(5, 10, 10, 1, 0, 0);
        let e7 = node_entry(7, 20, 20, 1, 0, 0);
        // Insert in non-sorted order to verify live_node_sst_uids sorts.
        m.add_node_sst(0, e7.clone()).unwrap();
        m.add_node_sst(0, e1.clone()).unwrap();
        m.add_node_sst(0, e5.clone()).unwrap();

        let uids = m.live_node_sst_uids();
        assert_eq!(uids, vec![1, 5, 7]);

        m.save_atomic(&path).unwrap();
        let reloaded = Manifest::load(&path).unwrap().expect("reload");
        assert_eq!(reloaded.live_node_sst_uids(), vec![1, 5, 7]);
    }

    /// Phase 2.5 Test H. retire_node_ssts is all-or-nothing.
    ///
    /// Two valid uids + one bogus uid in one call -> Err, NO
    /// removals applied. Defends the documented contract.
    #[test]
    fn phase25_retire_node_ssts_all_or_nothing() {
        let mut m = Manifest::new(KeyFingerprint::none());
        let e0 = node_entry(m.allocate_node_sst_uid(), 0, 9, 10, 0, 0);
        let e1 = node_entry(m.allocate_node_sst_uid(), 10, 19, 10, 0, 0);
        m.add_node_sst(0, e0.clone()).unwrap();
        m.add_node_sst(0, e1.clone()).unwrap();
        let before = m.live_node_sst_uids();

        let bogus = 0xDEAD_BEEF;
        let res = m.retire_node_ssts(&[e0.file_uid, bogus, e1.file_uid]);
        assert!(res.is_err(), "retire with one bogus uid must Err");
        // State unchanged -- defending all-or-nothing.
        assert_eq!(m.live_node_sst_uids(), before);
        assert_eq!(m.total_node_sst_count(), 2);
    }

    // ---------------------------------------------------------------------
    // OP-BRAIN-V5 Phase 3.5 (Q3.0.B) tests.
    // ---------------------------------------------------------------------

    /// Shared byte-assembly for the frozen-schema writer helpers
    /// below. Mirrors the `save_atomic` byte layout with a
    /// caller-chosen schema_version baked into the header.
    fn write_manifest_bytes(
        path: &Path,
        schema_version: u32,
        payload_bytes: &[u8],
    ) -> std::io::Result<()> {
        let payload_len = payload_bytes.len();
        let total = MANIFEST_PREFIX_BYTES + payload_len + MANIFEST_TRAILER_BYTES;
        let mut buf = Vec::with_capacity(total);
        buf.extend_from_slice(&MANIFEST_MAGIC.to_le_bytes());
        buf.extend_from_slice(&schema_version.to_le_bytes());
        buf.extend_from_slice(&(payload_len as u32).to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes()); // reserved
        buf.extend_from_slice(payload_bytes);
        let crc = crc32fast::hash(&buf);
        buf.extend_from_slice(&crc.to_le_bytes());
        std::fs::write(path, &buf)
    }

    /// Helper: write a manifest file with the v2 wire shape (frozen
    /// `ManifestPayloadV2` archetype), schema_version baked at 2.
    fn write_v2_manifest_file(
        path: &Path,
        payload: ManifestPayloadV2,
    ) -> std::io::Result<()> {
        let payload_bytes = rkyv::to_bytes::<_, 4096>(&payload)
            .expect("v2 rkyv serialize");
        write_manifest_bytes(path, 2, &payload_bytes)
    }

    /// Helper: write a manifest file with the v3 wire shape (frozen
    /// `ManifestPayloadV3` archetype), schema_version baked at 3.
    fn write_v3_manifest_file(
        path: &Path,
        payload: ManifestPayloadV3,
    ) -> std::io::Result<()> {
        let payload_bytes = rkyv::to_bytes::<_, 4096>(&payload)
            .expect("v3 rkyv serialize");
        write_manifest_bytes(path, 3, &payload_bytes)
    }

    fn bond_v3_entry(
        file_uid: u64,
        first: u64,
        last: u64,
        count: u64,
        size: u64,
        epoch: u64,
    ) -> BondSSTEntryV3 {
        BondSSTEntryV3 {
            file_uid,
            first_subject: first,
            last_subject: last,
            entry_count: count,
            file_size: size,
            epoch,
        }
    }

    /// Phase 3.5 Test A. v3 -> v4 backwards-compat read.
    ///
    /// Build a v3-shape manifest on disk (schema_version baked at 3,
    /// no reverse_file_uid in the rkyv blob), load it via the v4
    /// reader, assert: load succeeds, every promoted BondSSTEntry
    /// has `reverse_file_uid == None` with all other fields
    /// preserved, and the next save_atomic writes v4.
    #[test]
    fn phase35_v3_backwards_compat_read() {
        let dir = tempdir().unwrap();
        let path = dir.path().join(MANIFEST_FILE_NAME);

        let v3 = ManifestPayloadV3 {
            master_key_id: KeyFingerprint::none(),
            levels: vec![],
            memtable_wal: String::new(),
            last_compaction: 0,
            next_sst_id: 1,
            node_levels: vec![],
            next_node_sst_uid: 0,
            next_epoch: 14,
            bond_v5_levels: vec![
                BondV5LevelV3 {
                    level: 0,
                    ssts: vec![
                        bond_v3_entry(1, 0, 99, 100, 4096, 7),
                        bond_v3_entry(2, 100, 199, 50, 2048, 11),
                    ],
                },
                BondV5LevelV3 {
                    level: 1,
                    ssts: vec![bond_v3_entry(3, 0, 199, 150, 8192, 13)],
                },
            ],
            next_bond_v5_sst_uid: 4,
        };
        write_v3_manifest_file(&path, v3).unwrap();

        // Sanity: the on-disk schema byte is 3.
        let raw = std::fs::read(&path).unwrap();
        assert_eq!(u32::from_le_bytes(raw[4..8].try_into().unwrap()), 3);

        // Load via the current (v4) reader. Must succeed.
        let loaded = Manifest::load(&path).unwrap().expect("v3 manifest loads");
        assert_eq!(loaded.schema_version, 3, "schema_version snapshot reflects on-disk version");
        assert_eq!(loaded.bond_v5_levels.len(), 2);
        assert_eq!(loaded.total_bond_v5_sst_count(), 3);
        assert_eq!(loaded.peek_next_bond_v5_sst_uid(), 4);
        assert_eq!(loaded.peek_next_epoch(), 14);
        // Every promoted BondSSTEntry defaults reverse_file_uid=None.
        for (_lvl, e) in loaded.iter_bond_v5_ssts() {
            assert_eq!(
                e.reverse_file_uid, None,
                "v3 -> v4 default reverse_file_uid must be None"
            );
        }
        // Spot-check full field preservation on the L1 entry.
        let l1: Vec<&BondSSTEntry> = loaded
            .iter_bond_v5_ssts()
            .filter(|(_, e)| e.file_uid == 3)
            .map(|(_, e)| e)
            .collect();
        assert_eq!(l1.len(), 1);
        assert_eq!(l1[0].first_subject, 0);
        assert_eq!(l1[0].last_subject, 199);
        assert_eq!(l1[0].entry_count, 150);
        assert_eq!(l1[0].file_size, 8192);
        assert_eq!(l1[0].epoch, 13);

        // Next save_atomic writes v4.
        loaded.save_atomic(&path).unwrap();
        let raw2 = std::fs::read(&path).unwrap();
        assert_eq!(
            u32::from_le_bytes(raw2[4..8].try_into().unwrap()),
            MANIFEST_SCHEMA_VERSION,
            "first save after v3 load must promote on-disk schema to v4"
        );

        // And the freshly-saved manifest round-trips back cleanly.
        let reloaded = Manifest::load(&path).unwrap().expect("v4 round-trip");
        assert_eq!(reloaded.schema_version, MANIFEST_SCHEMA_VERSION);
        assert_eq!(reloaded.total_bond_v5_sst_count(), 3);
        for (_lvl, e) in reloaded.iter_bond_v5_ssts() {
            assert_eq!(e.reverse_file_uid, None);
        }
    }

    /// Phase 3.5 Test B. v4 round-trip with reverse_file_uid = Some(42).
    #[test]
    fn phase35_v4_roundtrip_reverse_file_uid_some() {
        let dir = tempdir().unwrap();
        let path = dir.path().join(MANIFEST_FILE_NAME);
        let mut m = Manifest::new(KeyFingerprint::none());
        let entry = BondSSTEntry {
            file_uid: m.allocate_bond_v5_sst_uid(),
            first_subject: 10,
            last_subject: 20,
            entry_count: 5,
            file_size: 512,
            epoch: m.allocate_epoch(),
            reverse_file_uid: Some(42),
        };
        m.add_bond_v5_sst(0, entry.clone()).unwrap();
        m.save_atomic(&path).unwrap();

        let reloaded = Manifest::load(&path).unwrap().expect("v4 roundtrip");
        assert_eq!(reloaded.schema_version, MANIFEST_SCHEMA_VERSION);
        assert_eq!(reloaded.total_bond_v5_sst_count(), 1);
        let observed: Vec<BondSSTEntry> = reloaded
            .iter_bond_v5_ssts()
            .map(|(_, e)| e.clone())
            .collect();
        assert_eq!(
            observed,
            vec![entry],
            "field-identical reconstruction including Some(42)"
        );
        assert_eq!(observed[0].reverse_file_uid, Some(42));
    }

    /// Phase 3.5 Test C. v4 round-trip with reverse_file_uid = None.
    /// Verifies the optional encoding works both ways.
    #[test]
    fn phase35_v4_roundtrip_reverse_file_uid_none() {
        let dir = tempdir().unwrap();
        let path = dir.path().join(MANIFEST_FILE_NAME);
        let mut m = Manifest::new(KeyFingerprint::none());
        let entry = BondSSTEntry {
            file_uid: m.allocate_bond_v5_sst_uid(),
            first_subject: 10,
            last_subject: 20,
            entry_count: 5,
            file_size: 512,
            epoch: m.allocate_epoch(),
            reverse_file_uid: None,
        };
        m.add_bond_v5_sst(0, entry.clone()).unwrap();
        m.save_atomic(&path).unwrap();

        let reloaded = Manifest::load(&path).unwrap().expect("v4 roundtrip");
        assert_eq!(reloaded.total_bond_v5_sst_count(), 1);
        let observed: Vec<BondSSTEntry> = reloaded
            .iter_bond_v5_ssts()
            .map(|(_, e)| e.clone())
            .collect();
        assert_eq!(observed, vec![entry]);
        assert_eq!(observed[0].reverse_file_uid, None);
    }

    /// Phase 3.5 Test D. Unsupported schema version (v=99) stays a
    /// hard error at the v4 gate -- NOT a silent fallback. Mirrors
    /// phase25_unsupported_schema_version_is_hard_error; re-asserted
    /// here because Phase 3.5 moved the gate's upper bound to 4.
    #[test]
    fn phase35_unsupported_schema_version_is_hard_error() {
        let dir = tempdir().unwrap();
        let path = dir.path().join(MANIFEST_FILE_NAME);

        let m = Manifest::new(KeyFingerprint::none());
        m.save_atomic(&path).unwrap();
        let mut bytes = std::fs::read(&path).unwrap();
        bytes[4..8].copy_from_slice(&99u32.to_le_bytes());
        // Recompute the CRC so the load rejects on schema, not
        // checksum.
        let body_end = bytes.len() - 4;
        let crc = crc32fast::hash(&bytes[..body_end]);
        bytes[body_end..body_end + 4].copy_from_slice(&crc.to_le_bytes());
        std::fs::write(&path, &bytes).unwrap();

        let res = Manifest::load(&path);
        match res {
            Ok(_) => panic!("load of schema_version=99 must error"),
            Err(BrainError::Format(s)) => {
                // ManifestError::UnsupportedSchemaVersion Display
                // shape: "manifest schema version 99 is outside
                // supported range [1, 5]".
                assert!(
                    s.contains("schema version 99"),
                    "error must mention the bad schema; got: {}", s
                );
                assert!(
                    s.contains("[1, 5]"),
                    "error must mention the supported range incl. max=5; got: {}", s
                );
            }
            Err(other) => panic!("expected Format error, got {:?}", other),
        }
    }

    /// Phase 3.5 Test E. v2 -> v4 chain.
    ///
    /// v2 manifests promote DIRECTLY to the v4 in-memory shape: the
    /// existing `From<ManifestPayloadV2>` impl targets the live
    /// `ManifestPayload`, whose v3-introduced fields default
    /// empty/0. reverse_file_uid is not applicable because zero
    /// bond_v5_levels exist on a v2 brain.
    #[test]
    fn phase35_v2_to_v4_chain() {
        let dir = tempdir().unwrap();
        let path = dir.path().join(MANIFEST_FILE_NAME);

        let v2 = ManifestPayloadV2 {
            master_key_id: KeyFingerprint::none(),
            levels: vec![LevelEntryV2 {
                level: 0,
                ssts: vec![SSTEntryV2 {
                    id: 1,
                    file: sst_filename(0, 1),
                    subject_min: [0x10; 32],
                    subject_max: [0x20; 32],
                    size: 1024,
                    mtime: 1_700_000_000,
                    bond_count: 42,
                    epoch: 9,
                }],
            }],
            memtable_wal: String::new(),
            last_compaction: 0,
            next_sst_id: 2,
            node_levels: vec![NodeLevelV2 {
                level: 0,
                ssts: vec![NodeSSTEntryV2 {
                    file_uid: 0,
                    first_node_id: 0,
                    last_node_id: 99,
                    entry_count: 100,
                    file_size: 4096,
                    epoch: 3,
                }],
            }],
            next_node_sst_uid: 1,
            next_epoch: 10,
        };
        write_v2_manifest_file(&path, v2).unwrap();

        // Sanity: the on-disk schema byte is 2.
        let raw = std::fs::read(&path).unwrap();
        assert_eq!(u32::from_le_bytes(raw[4..8].try_into().unwrap()), 2);

        // Load via the current (v4) reader. Must succeed.
        let loaded = Manifest::load(&path).unwrap().expect("v2 manifest loads");
        assert_eq!(loaded.schema_version, 2);
        // v2 data preserved.
        assert_eq!(loaded.total_sst_count(), 1);
        assert_eq!(loaded.total_node_sst_count(), 1);
        assert_eq!(loaded.peek_next_node_sst_uid(), 1);
        assert_eq!(loaded.peek_next_epoch(), 10);
        // v3-introduced fields default empty/0.
        assert_eq!(loaded.bond_v5_levels.len(), 0);
        assert_eq!(loaded.total_bond_v5_sst_count(), 0);
        assert_eq!(loaded.peek_next_bond_v5_sst_uid(), 0);

        // Next save_atomic writes v4 and round-trips.
        loaded.save_atomic(&path).unwrap();
        let raw2 = std::fs::read(&path).unwrap();
        assert_eq!(
            u32::from_le_bytes(raw2[4..8].try_into().unwrap()),
            MANIFEST_SCHEMA_VERSION,
            "first save after v2 load must promote on-disk schema to v4"
        );
        let reloaded = Manifest::load(&path).unwrap().expect("v4 round-trip");
        assert_eq!(reloaded.schema_version, MANIFEST_SCHEMA_VERSION);
        assert_eq!(reloaded.total_sst_count(), 1);
        assert_eq!(reloaded.total_node_sst_count(), 1);
        assert_eq!(reloaded.total_bond_v5_sst_count(), 0);
    }
}
