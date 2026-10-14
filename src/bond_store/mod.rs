//! Bond store -- separate storage substrate for bond rows.
//!
//! OP-BRAIN-V5 Phase 3.x. Sibling to `node_store/` (Phase 2.x):
//! both share the WAL and the manifest but own independent on-disk
//! SSTs and independent in-process flush + compaction workers.
//! Per Phase 3.0 audit S7 Recommendation E (Q3.0.E): module path
//! `src/bond_store/` mirroring `src/node_store/`, NOT a separate
//! `lu_bond_store` crate -- the `lu4d <-> lu_bond_store` circular
//! dep on `Seed`/`NodeIntern`/`NodeId`/`TemporalKeyframe` is the
//! same killer that ruled out a shared substrate crate for Phase 2.
//!
//! ## Module layout
//!
//! - `sst_writer.rs` -- streaming sort-by-(subject, predicate,
//!   object) writer (Phase 3.1).
//! - `sst_reader.rs` -- block-decoded reader with LRU cache and
//!   per-subject range-scan API (Phase 3.2, this commit).
//! - `flush.rs` -- `BondFlushWorker` + `cleanup_orphan_bond_ssts`
//!   (Phase 3.3).
//! - `compaction.rs` -- `BondCompactionWorker` streaming k-way merge
//!   + epoch-barrier tombstone GC (Phase 3.4) + paired reverse-index
//!   emission (Phase 3.6).
//! - `read_path.rs` -- `BondSstReaderCache` + `get_outgoing_bonds_v5`
//!   + `get_incoming_bonds_v5` (Phase 3.7, this commit): the v5
//!   fall-through read path. SST-only (the live MemTable bond half
//!   is NOT consulted -- see the module docs there); coexists with
//!   the v4 production read path until Phase 3.9/3.10.
//!
//! No `bond_store/memtable.rs`: the bond MemTable lives in
//! `src/memtable.rs::MemTableInner.entries: BTreeMap<NodeId,
//! Vec<MemBond>>` and stays there per Phase 3.0 audit S5
//! Recommendation F (Q3.0.F). No `bond_store/sidecar.rs`: bonds
//! derive everything from NodeIntern + WAL + SSTs.
//!
//! ## Phase 3.4 status
//!
//! This commit adds `compaction.rs` (`BondCompactionWorker` --
//! streaming k-way merge of N BondSSTs into one output at a higher
//! level, with cross-store epoch-barrier tombstone GC). No
//! production code calls `BondCompactionWorker::compact` yet --
//! tests construct `CompactionPlan` directly; the scheduler and the
//! Phase 3.7 read path wire the store into production.

pub mod compaction;
pub mod flush;
pub mod read_path;
pub mod sst_reader;
pub mod sst_writer;
pub mod scheduler;      // OP-BRAIN-V5 -- production bond-compaction scheduler (L0->L1 + reverse index)

pub use scheduler::{
    BondV5CompactionScheduler, DEFAULT_BOND_V5_L0_TRIGGER,
    DEFAULT_BOND_V5_POLL_INTERVAL_MS,
};
pub use compaction::{
    BondCompactionError, BondCompactionWorker, CompactionPlan as BondCompactionPlan,
    CompactionResult as BondCompactionResult, BOND_TOMBSTONE_CONFIDENCE,
};
pub use flush::{
    bsst_filename, bsst_reverse_filename, cleanup_orphan_bond_ssts, BondFlushError,
    BondFlushWorker, CleanupReport as BondCleanupReport, BSST_REVERSE_SUFFIX, BSST_SUFFIX,
};
pub use read_path::{
    get_incoming_bonds_v5, get_outgoing_bonds_v5, BondReadPathError, BondSstReaderCache,
};
pub use sst_reader::{
    BondSstRangeIter, BondSstReadStats, BondSstReader, DEFAULT_BLOCK_CACHE_CAP,
};
pub use sst_writer::{
    BondPayload, BondSstError, BondSstWriteOpts, BondSstWriteStats, BondSstWriter,
    CompressionKind, CrcScope, BSST_FORMAT_VERSION, BSST_MAGIC,
    DEFAULT_TARGET_ENTRIES_PER_BLOCK, MAX_TARGET_ENTRIES_PER_BLOCK,
};
