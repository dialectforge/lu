//! OP-BRAIN-V4 Phase 5 — canonical read path through MemTable + L0 SSTs +
//! v3 archive.
//!
//! ## Why a separate module
//!
//! Phase 4 wired the WRITE path (MemTable insert -> WAL append -> flush
//! worker -> L0 SST -> manifest swap). Phase 5 wires the matching READ
//! path: a single helper that consults the three layers in newest-first
//! order, dedups by `(subject, predicate, object)` taking latest-seq,
//! and re-derives confidence from keyframe-source provenance.
//!
//! The merge logic is non-trivial (three layers, two duplicate sources,
//! one confidence rewrite), so it lives in its own module to keep
//! `brain.rs` from growing further and to keep test surface tight.
//!
//! ## Read-path mode flag
//!
//! `ReadPathMode` is the runtime switch:
//!
//! - `V3Only` (the production default) — read via the existing v3 path
//!   only. Byte-equivalent to the pre-Phase-5 binary. Phase 8 flips the
//!   default; until then production stays on `V3Only`.
//! - `V4Only` — read via `read_outgoing_bonds_v4` only. Used by Phase 5
//!   tests for A/B equivalence on v3 brains (MemTable + L0 are empty,
//!   so v4 falls through to v3 internally).
//! - `MergedV3V4` — same as `V4Only` from an external point of view
//!   (`read_outgoing_bonds_v4` already merges v3 internally). The
//!   variant exists as a labelled alias so callers can express
//!   intent — Phase 6 compaction will give it different semantics
//!   when L0 -> L1+ promotion changes the read ordering.
//!
//! ## Dedup discipline
//!
//! The deduplication key is `(subject, predicate, object)`. The winner
//! is the entry with the highest `seq`:
//!
//! - MemTable bonds use the MemTable's monotonic `seq` (the same value
//!   the WAL record carries).
//! - L0 SST bonds use the SST's manifest id as the seq (assigned
//!   monotonically by `Manifest::next_sst_id`, so newer SSTs have
//!   higher ids).
//! - v3 archive bonds use `seq = 0` (oldest layer).
//!
//! This ordering is correct because the MemTable's seq counter and
//! the manifest's next_sst_id are both monotonic for the lifetime
//! of the brain — a bond that was in a MemTable that has since been
//! flushed to L0 carries an SST id strictly higher than the bonds
//! that came from any previous L0. Pre-existing v3 archive content
//! always pre-dates the v4 write path, so seq=0 is correct.
//!
//! The dedup catches the Phase 4 "crash between manifest update and
//! WAL truncate" window: after restart, the WAL is replayed into the
//! new MemTable AND the SST is referenced by the manifest, so the
//! same bond can appear in both. Without dedup, spread activation
//! would count its weight twice. With dedup, the MemTable's seq
//! exceeds the SST's manifest id (because the replay is the most
//! recent write activity), so the MemTable copy wins.
//!
//! ## Confidence re-derivation
//!
//! Phase 4 flushes every bond at `Confidence::Verified` because
//! `MemBond` does not carry a confidence band. The read path re-derives
//! the band from the bond's latest-timestamp `KeyframeSource` using
//! `bond::derive_confidence` (`bond::confidence_from_keyframe_source`).
//! See the module comment in `src/bond.rs` for the mapping table.

use crate::bond::{BondType, TemporalKeyframe};
use crate::node::Confidence;
use crate::seed::Seed;

/// Runtime switch for `Brain::read_outgoing_bonds`. Stored on Brain as
/// an `AtomicU8` so tests and (future) MCP tools can flip it without
/// taking a write lock on the Brain itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ReadPathMode {
    /// Production default. Only the existing v3 read path is
    /// consulted; MemTable + L0 SSTs are ignored even if present.
    /// Byte-equivalent to the pre-Phase-5 binary.
    V3Only = 0,
    /// Read via `read_outgoing_bonds_v4` only. Always consults
    /// MemTable + L0 + v3 (with dedup). Used by Phase 5 tests and
    /// Phase 8 cutover.
    V4Only = 1,
    /// Same merge as `V4Only` today (the v4 path already folds v3 in).
    /// Kept as a labelled alias so callers can express "I expect the
    /// merge, do not strip v3" — Phase 6 compaction will give this
    /// variant different semantics when L0 -> L1+ promotion changes
    /// the SST snapshot semantics.
    MergedV3V4 = 2,
    /// OP-BRAIN-V5 6.2 -- read exclusively from the v5 substrate. Node
    /// reads route through `Brain::get_node_v5` (NodeMemTable + NodeSST);
    /// bond/neighbor reads come from the boot-rehydrated in-memory
    /// adjacency (populated from the v5 BondSSTs by `rehydrate_adjacency`)
    /// and, NodeId-keyed, from `get_outgoing_bonds_v5` /
    /// `get_incoming_bonds_v5`. Set by `Brain::load_v5`. The v3 archive
    /// and v4 L0 SSTs are not consulted (a v5 brain has neither).
    V5Only = 3,
}

impl ReadPathMode {
    /// Round-trip with the AtomicU8 storage on Brain.
    pub fn from_u8(v: u8) -> Self {
        match v {
            1 => ReadPathMode::V4Only,
            2 => ReadPathMode::MergedV3V4,
            3 => ReadPathMode::V5Only,
            _ => ReadPathMode::V3Only,
        }
    }

    pub fn as_u8(self) -> u8 {
        self as u8
    }

    /// True iff the v3-only fast path is in effect. Spread activation
    /// uses this to short-circuit to the existing `view_bond` machinery
    /// without paying merge overhead.
    pub fn is_v3_only(self) -> bool {
        matches!(self, ReadPathMode::V3Only)
    }

    /// OP-BRAIN-V5 6.2 -- true iff the v5-only read path is in effect.
    pub fn is_v5_only(self) -> bool {
        matches!(self, ReadPathMode::V5Only)
    }
}

/// An owned, layer-source-tagged view of one bond returned by the
/// Phase 5 read path. Distinct from `sst::BondView` and
/// `view::BondView` because those two are tightly bound to the SST
/// block (Arc) and the heap/archive split respectively — neither
/// can carry a MemTable-resident bond without copying. `MergedBondView`
/// is the unified shape after the merge runs.
#[derive(Debug, Clone)]
pub struct MergedBondView {
    pub subject: Seed,
    pub object: Seed,
    pub predicate: BondType,
    pub weight: f32,
    /// Confidence band as it WOULD be observed by spread activation.
    /// For MemTable / L0 sources this is the re-derived band per
    /// `bond::derive_confidence`; for v3 archive bonds this is the
    /// stored band (v3 already records the band on disk).
    pub confidence: Confidence,
    /// Full keyframe history — every keyframe the underlying layer
    /// carried for this bond. Empty Vec is legal (a freshly
    /// inserted MemBond may have none).
    pub keyframes: Vec<TemporalKeyframe>,
    /// Dedup seq used by the merge step. Diagnostic; the merge
    /// already used it before returning the bond.
    pub seq: u64,
    /// Source layer this bond came from. Diagnostic — spread
    /// activation does not branch on this. Useful for telemetry
    /// and Phase 5 tests asserting "this bond came from MemTable".
    pub source_layer: BondSourceLayer,
    /// Phase 6: true iff this entry is a tombstone (deletion marker).
    /// The merge step keeps tombstone winners in the dedup map so
    /// they shadow older same-key bonds; the public read API drops
    /// tombstone winners before returning the result Vec.
    pub tombstone: bool,
}

/// Which layer the merge step picked the bond from. Newest-first:
/// MemTable > L0 > V3.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BondSourceLayer {
    /// In-memory write buffer. Highest seq.
    MemTable,
    /// On-disk Level-0 SST. Seq = the SST's manifest id.
    L0Sst,
    /// OP-BRAIN-V5 6.3 -- on-disk v5 BondSST (any level), served by
    /// `get_outgoing_bonds_v5`. Seq = the bond's monotonic MemTable seq
    /// carried unchanged through flush into `BondPayload.seq`.
    V5Sst,
    /// Pre-existing v3 archive / heap bond. Seq = 0.
    V3Archive,
}

impl BondSourceLayer {
    pub fn as_str(self) -> &'static str {
        match self {
            BondSourceLayer::MemTable => "memtable",
            BondSourceLayer::L0Sst => "l0_sst",
            BondSourceLayer::V5Sst => "v5_sst",
            BondSourceLayer::V3Archive => "v3_archive",
        }
    }
}

/// Stable dedup key derived from `(subject, predicate, object)`. The
/// merge step builds a `HashMap<DedupKey, MergedBondView>` keyed on
/// this, replacing whenever a higher-seq entry comes in.
///
/// We use this small key (rather than a tuple of `Seed`s) so the
/// `HashMap` overhead stays small on hub subjects — the underlying
/// bytes are 32 + 32 + 1 = 65 bytes and the hash is fixed-time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DedupKey {
    pub subject: [u8; 32],
    pub object: [u8; 32],
    pub predicate: u8,
}

impl DedupKey {
    pub fn new(subject: Seed, object: Seed, predicate: BondType) -> Self {
        DedupKey {
            subject: *subject.as_bytes(),
            object: *object.as_bytes(),
            predicate: predicate_discriminant(predicate),
        }
    }
}

/// Stable u8 discriminant for `BondType`. Delegates to
/// `crate::memtable::predicate_discriminant` so the encoding is
/// canonical across the v4 stack (MemTable + SST writer + read
/// dedup). Kept as a thin re-export so the call sites in this module
/// stay readable.
#[inline]
fn predicate_discriminant(b: BondType) -> u8 {
    crate::memtable::predicate_discriminant(b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_path_mode_round_trips() {
        for m in [ReadPathMode::V3Only, ReadPathMode::V4Only, ReadPathMode::MergedV3V4] {
            assert_eq!(ReadPathMode::from_u8(m.as_u8()), m);
        }
    }

    #[test]
    fn read_path_mode_default_is_v3only_on_zero() {
        assert_eq!(ReadPathMode::from_u8(0), ReadPathMode::V3Only);
        // Unknown values fall back to V3Only — the safest default.
        assert_eq!(ReadPathMode::from_u8(99), ReadPathMode::V3Only);
    }

    #[test]
    fn read_path_mode_is_v3_only_only_for_v3only() {
        assert!(ReadPathMode::V3Only.is_v3_only());
        assert!(!ReadPathMode::V4Only.is_v3_only());
        assert!(!ReadPathMode::MergedV3V4.is_v3_only());
    }

    #[test]
    fn dedup_key_distinguishes_by_predicate() {
        let s = Seed::from_label("S");
        let o = Seed::from_label("O");
        let k1 = DedupKey::new(s, o, BondType::IsA);
        let k2 = DedupKey::new(s, o, BondType::HasProperty);
        assert_ne!(k1, k2);
    }

    #[test]
    fn dedup_key_equal_for_same_triple() {
        let s = Seed::from_label("S");
        let o = Seed::from_label("O");
        let k1 = DedupKey::new(s, o, BondType::IsA);
        let k2 = DedupKey::new(s, o, BondType::IsA);
        assert_eq!(k1, k2);
    }

    #[test]
    fn predicate_discriminant_covers_all_29_variants() {
        // BondType has 37 variants today (Phase 5 audit + Phase-1 code variants).
        // Map each and confirm uniqueness so a future re-ordering
        // would surface as a missing-arm compile error in
        // `memtable::predicate_discriminant` AND as a runtime
        // length mismatch here.
        let all = [
            BondType::IsA, BondType::HasProperty, BondType::Causes,
            BondType::PartOf, BondType::UsedFor, BondType::RelatedTo,
            BondType::ContrastedWith, BondType::DerivedFrom,
            BondType::Requires, BondType::Produces, BondType::Opposes,
            BondType::BornIn, BondType::BornOn, BondType::DiedIn,
            BondType::DiedOn, BondType::DiscoveredBy, BondType::FoundedBy,
            BondType::FoundedIn, BondType::HasAltitude, BondType::HasAspect,
            BondType::HasCapital, BondType::HasOfficialLanguage,
            BondType::HasPopulation, BondType::KnownFor, BondType::LocatedIn,
            BondType::MeasuredIn, BondType::NamedAfter, BondType::WrittenBy,
            BondType::HasArea,
            BondType::HasMethod,
            BondType::HasAttribute,
            BondType::HasParameter,
            BondType::Returns,
            BondType::Inherits,
            BondType::Calls,
            BondType::HasType,
            BondType::Implements,
            BondType::DefinedAs, BondType::WorksBy, BondType::ComposedOf,
            BondType::PurposeOf, BondType::DiffersFrom, BondType::LeadsTo,
            BondType::ExampleOf,
        ];
        let mut discs: Vec<u8> = all.iter().map(|b| predicate_discriminant(*b)).collect();
        discs.sort_unstable();
        discs.dedup();
        assert_eq!(discs.len(), all.len(), "predicate_discriminant must be injective across all BondType variants");
    }

    #[test]
    fn bond_source_layer_as_str_round_trips() {
        assert_eq!(BondSourceLayer::MemTable.as_str(), "memtable");
        assert_eq!(BondSourceLayer::L0Sst.as_str(), "l0_sst");
        assert_eq!(BondSourceLayer::V3Archive.as_str(), "v3_archive");
    }
}
