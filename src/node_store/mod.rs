//! Node store -- separate storage substrate for `KnowledgeNode` payloads.
//!
//! OP-BRAIN-V5 Phase 2.x. The bond store and the node store are
//! independent substrates sharing the WAL and the manifest; this
//! module owns the node-side memtable, flush, SST, compaction, and
//! sidecar primitives (per audit S7 Recommendation M: module path,
//! not separate crate -- the circular `lu4d <-> lu_node_store`
//! dependency on `Seed`/`NodeIntern`/`KnowledgeNode` is the killer).
//!
//! ## Module layout
//!
//! - `sidecar.rs` -- `nodeintern.bin` streamed I/O (Phase 2.1).
//! - `memtable.rs` -- `NodeMemTable` (Phase 2.3).
//! - `sst_writer.rs` -- streaming sort-by-NodeId writer (Phase 2.4).
//! - `sst_reader.rs` -- block-decoded reader with LRU cache (Phase 2.4).
//! - `flush.rs` -- `NodeFlushWorker` + `cleanup_orphan_node_ssts` (Phase 2.6).
//! - `compaction.rs` -- node-store compaction worker (Phase 2.7).
//! - `read_path.rs` -- `NodeSstReaderCache` + `get_node_v5` (Phase 2.8).
//!
//! Sub-port files are added one port at a time. Through Phase 2.7
//! the node-store flush worker is wired into the bond-side
//! `flush.rs::flush_once` integration point (sidecar -> NodeSST ->
//! bond SST -> manifest -> WAL truncate). Phase 2.8 adds the v5
//! fall-through read path (`Brain::get_node_v5`) coexisting with
//! the v4 production path; Phase 2.9 migration will cut production
//! over to the v5 path.

pub mod compaction;
pub mod degmap;          // OP-BRAIN-MapRAM -- persisted resident degree-map sidecar
pub mod flush;
pub mod memtable;
pub mod read_path;
pub mod sidecar;
pub mod sst_reader;
pub mod sst_writer;
pub mod scan;          // OP-022.b -- v5 live-node scan + keyword-index build
pub mod scheduler;      // OP-BRAIN-V5 -- production node-compaction scheduler (L0->L1)

pub use compaction::{
    CompactionPlan, CompactionResult, NodeCompactionError, NodeCompactionWorker,
};
pub use flush::{
    cleanup_orphan_node_ssts, nsst_filename, CleanupReport, NodeFlushError, NodeFlushWorker,
    NSST_SUFFIX,
};
pub use memtable::{NodeEntry, NodeMemTable, APPROX_NODE_BYTES};
pub use sidecar::{SidecarError, SidecarStats, SIDECAR_FORMAT_VERSION, SIDECAR_MAGIC};
pub use sst_reader::{NodeSstReader, NodeSstReadStats, DEFAULT_BLOCK_CACHE_CAP};
pub use read_path::{get_node_v5, NodeSstReaderCache, ReadPathError};
pub use scan::{build_v5_keyword_sidecar, scan_live_v5_nodes};
pub use scheduler::{
    NodeV5CompactionScheduler, DEFAULT_NODE_V5_L0_TRIGGER,
    DEFAULT_NODE_V5_POLL_INTERVAL_MS,
};
pub use sst_writer::{
    CompressionKind, CrcScope, NodeSstError, NodeSstWriteOpts, NodeSstWriteStats, NodeSstWriter,
    DEFAULT_TARGET_ENTRIES_PER_BLOCK, MAX_TARGET_ENTRIES_PER_BLOCK, NSST_FORMAT_VERSION, NSST_MAGIC,
};
