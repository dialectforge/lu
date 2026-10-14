//! Storage v2 Phase 2.2 (Slice 2a) — v3 binary brain file format.
//!
//! This module defines the v3 on-disk layout. The bulk of the brain
//! (`NetworkSnapshot` + `ScarLog`) is now rkyv-archived rather than
//! bincoded, which is the on-disk groundwork the future zero-copy mmap
//! load path (Slice 2c) will build on. The deferred side tables
//! (`InterestProfile`, `MetaReflection`, research proposals,
//! `ResearchMisc`, `gap_seed_frequency`) remain bincode in v3 because
//! their rkyv derives are not wired in yet — that's queued for a
//! later slice if needed.
//!
//! Slice 2a load path:
//!   open file → mmap → validate magic + version + crc32 + offsets →
//!   `rkyv::check_archived_root` the two archived sections →
//!   `rkyv::Deserialize` them onto the heap → eager-deserialize the
//!   side tables via bincode → return an owned `BrainFile`.
//!
//! That is, in this slice the v3 reader still produces a heap-allocated
//! `BrainFile` shaped identically to what the v2 reader produces. RSS
//! reduction requires the API refactor + a real zero-copy reader, which
//! are queued as Slices 2b + 2c. What Slice 2a delivers is the on-disk
//! format, the migration tool, and proof-of-life for the rkyv archived
//! sections through `check_archived_root`.
//!
//! Layout
//! ------
//!
//! ```text
//! [0..8]    magic   = b"LU4BRAIN"      (same magic as v2; the version
//!                                       field at bytes 8..12 is what
//!                                       distinguishes v2 from v3)
//! [8..12]   version = u32 LE  (= 3)
//! [12..16]  flags   = u32 LE  (= 0; reserved bits identical to v2)
//! [16..24]  node_count u64 LE
//! [24..32]  bond_count u64 LE
//! [32..40]  keyframe_count u64 LE
//! [40..48]  scar_count u64 LE
//! [48..52]  section_count u32 LE
//! [52..64]  reserved (zeroed) — fixed header is 64 bytes, identical
//!                               in shape to v2.
//!
//! [64 .. 64 + 24*section_count]  offset table (same shape as v2):
//!     each entry is 24 bytes:
//!       u32 LE  section_id
//!       u32 LE  reserved (zero)
//!       u64 LE  offset (absolute byte offset within file)
//!       u64 LE  length (in bytes)
//!
//! Sections (numeric ids reuse v2's; the version byte gates the
//! interpretation of sections 1 and 4 — those are rkyv-archived in v3
//! rather than bincoded):
//!
//!   id 1  Network          rkyv-archived NetworkSnapshot   (16-byte aligned)
//!   id 2  InterestProfile  bincode (deferred type)
//!   id 3  MetaReflection   bincode (deferred type)
//!   id 4  ScarLog          rkyv-archived ScarLog           (16-byte aligned)
//!   id 5  ResearchProposals bincode
//!   id 6  ResearchMisc     bincode (ResearchConfig + next_id)
//!   id 7  GapSeedFrequency bincode (HashMap<Seed, u32>)
//!
//! Each rkyv section's absolute file offset is padded forward so that
//! the offset is a multiple of `RKYV_ALIGN` (= 16). Pad bytes inside
//! the file are zeros and are NOT counted in the section's `length` —
//! they sit between the previous section's end and the next section's
//! recorded offset. The trailing crc32 covers them like every other
//! byte in the data region, so corruption in pad bytes is still
//! detected.
//!
//! Trailer (last 16 bytes — identical to v2):
//!   [trailer_pos .. +8]   magic = b"LU4ENDOK"
//!   [trailer_pos+8 ..+12] crc32 (IEEE) of every byte before the trailer
//!   [trailer_pos+12..+16] reserved (zeroed)
//! ```
//!
//! Auto-detection: a file is v3 if it has the v2 magic at bytes 0..8
//! AND a version of 3 at bytes 8..12. See `brain_io::detect_format`.

use std::collections::HashMap;
use std::path::Path;

use memmap2::Mmap;
use rkyv::Deserialize;

use crate::brain::{BrainError, BrainFile};
use crate::brain_io::{
    ResearchMiscBlob, ENTRY_SIZE, HEADER_SIZE, MAGIC_HEADER, MAGIC_TRAILER, TRAILER_SIZE,
};
use crate::curiosity::{InterestProfile, WonderState};
use crate::network::NetworkSnapshot;
use crate::research::proposal::ResearchProposal;
use crate::scar::ScarLog;
use crate::seed::Seed;
use crate::self_reflection::MetaReflection;

/// On-disk format version for the v3 binary layout.
pub const FORMAT_VERSION_V3: u32 = 3;

/// rkyv-archived sections must start at offsets that are multiples of
/// this value within the file. mmap regions are page-aligned (≥ 4096),
/// so as long as the section's *absolute* file offset is a multiple of
/// 16, the mmap'd byte slice for that section satisfies rkyv's
/// alignment requirement for `NetworkSnapshot` / `ScarLog`.
pub const RKYV_ALIGN: u64 = 16;

/// Section identifiers (numeric values match v2's; see module-level docs
/// for the v2 vs v3 interpretation difference on ids 1 and 4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum SectionIdV3 {
    Network = 1,
    InterestProfile = 2,
    MetaReflection = 3,
    ScarLog = 4,
    ResearchProposals = 5,
    ResearchMisc = 6,
    GapSeedFrequency = 7,
    /// Port J.1: persisted wonder queue (Vec<CuriositySignal>) +
    /// gap label map (HashMap<Seed, String>). Optional on read so
    /// pre-J.1 brain files (section_count=7) load with empty state.
    WonderQueueState = 8,
    /// Port T.4: persisted PendingLog snapshot (Vec<PendingChange> +
    /// next_pending_id). Optional on read so pre-T.4 brain files
    /// (section_count=8) load with an empty PendingLog snapshot.
    /// The companion `<brain>.pending.wal` is replayed on top of
    /// this snapshot at load time -- the WAL is the source of
    /// truth for any mutations between the last successful save
    /// and the current load.
    PendingLog = 9,
    /// Port T.8: persisted rejection-pattern store snapshot
    /// (Vec<RejectionFingerprint> rejections + approvals,
    /// Vec<RejectionPattern> learned patterns,
    /// proposals_filtered_total counter, RejectionConfig).
    /// Optional on read so pre-T.8 brain files (section_count=9)
    /// load with an empty store. Lu uses this to learn from
    /// rejected pending entries so she stops proposing the same
    /// shape of low-quality bonds.
    RejectionPatterns = 10,
    /// Port LB: persisted Left Brain session memory store snapshot
    /// (Vec<SessionEntry> entries, next_entry_id allocator,
    /// archived_sessions list). Optional on read so pre-LB brain
    /// files (section_count=10) load with an empty store. The
    /// Left Brain layer holds operational/episodic memory
    /// (thinking traces, observations, hypotheses, …) that
    /// informs reasoning but never auto-promotes to fact bonds.
    SessionMemory = 11,
    /// Port CS — Conversational Style profile per user_key.
    ConversationalStyle = 12,
    /// Port Cat.4 — signed safety pack ledger (trusted publishers,
    /// installed packs, revoked packs). Optional on read so pre-Cat.4
    /// brain files (section_count=12) load with an empty ledger.
    PackLedger = 13,
    /// Port Cat.3c — RuntimeConfig section consolidating the legacy
    /// `<brain>.trust.json` (Cat.3a) and `<brain>.hub_degree.json`
    /// (Cat.3b) sidecars. Optional on read so pre-Cat.3c brain files
    /// (section_count=13) load with `RuntimeConfigSnapshot::default()`
    /// and the legacy sidecar load path takes over.
    RuntimeConfig = 14,
    /// Port Cat.3c.1 — RuntimeConfigV2: authoritative, future-
    /// extensible runtime-config payload. JSON-encoded so adding
    /// fields just works (serde_json + #[serde(default)] gives
    /// forward AND backward compat at the field level). Cat.3c.1+
    /// readers prefer this section; older binaries skip it via the
    /// J.1.1 unknown-section tolerance and read section #14 instead.
    RuntimeConfigV2 = 15,
    /// Port Source-Persist internal notes — per-bond source
    /// attribution overlay. Bincode-encoded `SourceOverlay`. Optional on
    /// read: pre-Source-Persist brain files (section_count<=15) load
    /// with an empty overlay (every bond defaults to
    /// `SourceKind::Unknown`). Post-Source-Persist binaries reading a
    /// pre-Source-Persist brain hydrate empty; pre-Source-Persist
    /// binaries reading a Source-Persist brain skip this section via
    /// the J.1.1 unknown-section tolerance and lose attribution
    /// silently (same as the pre-fix bug — no regression on this path).
    SourceOverlay = 16,
    /// Port Cat.5 — schema-only signed capability pack ledger
    /// (trusted publishers, installed packs, revoked pack_ids).
    /// Optional on read so pre-Cat.5 brain files (section_count<=16)
    /// load with an empty ledger; pre-Cat.5 binaries reading a
    /// Cat.5 brain skip this section via the J.1.1 unknown-section
    /// tolerance (no data loss for Cat.5 since the ledger is the
    /// only Cat.5 state). Cat.5 schema-only does NOT modify any
    /// bonds; the ledger rides v3enc as cleartext bincode of
    /// `CapabilityPackLedgerSnapshot`.
    CapabilityPackLedger = 17,
}

impl SectionIdV3 {
    #[allow(dead_code)]
    fn from_u32(v: u32) -> Option<Self> {
        match v {
            1 => Some(SectionIdV3::Network),
            2 => Some(SectionIdV3::InterestProfile),
            3 => Some(SectionIdV3::MetaReflection),
            4 => Some(SectionIdV3::ScarLog),
            5 => Some(SectionIdV3::ResearchProposals),
            6 => Some(SectionIdV3::ResearchMisc),
            7 => Some(SectionIdV3::GapSeedFrequency),
            8 => Some(SectionIdV3::WonderQueueState),
            9 => Some(SectionIdV3::PendingLog),
            10 => Some(SectionIdV3::RejectionPatterns),
            11 => Some(SectionIdV3::SessionMemory),
            12 => Some(SectionIdV3::ConversationalStyle),
            13 => Some(SectionIdV3::PackLedger),
            14 => Some(SectionIdV3::RuntimeConfig),
            15 => Some(SectionIdV3::RuntimeConfigV2),
            16 => Some(SectionIdV3::SourceOverlay),
            17 => Some(SectionIdV3::CapabilityPackLedger),
            _ => None,
        }
    }
}

/// Port Cat.5 bumped this from 16 to 17 (added CapabilityPackLedger).
/// Port Source-Persist bumped this from 15 to 16 (added SourceOverlay).
/// Port Cat.3c.1 bumped this from 14 to 15 (added RuntimeConfigV2).
/// Port Cat.3c bumped this from 13 to 14 (added RuntimeConfig).
/// Port Cat.4 bumped it from 12 to 13 (added PackLedger).
/// Port CS bumped it from 11 to 12 (added ConversationalStyle).
/// Port LB bumped it from 10 to 11 (added SessionMemory).
/// Port T.8 bumped this from 9 to 10 (added RejectionPatterns).
/// Port T.4 bumped it from 8 to 9 (added PendingLog).
/// Port J.1 had bumped it from 7 to 8 (added WonderQueueState).
/// Reader paths accept files with `header.section_count >=
/// MIN_SUPPORTED_SECTION_COUNT` (pre-J.1 = 7) by treating the
/// WonderQueueState (id 8), PendingLog (id 9), and RejectionPatterns
/// (id 10) sections as optional on load; see
/// `parse_layout`/`parse_layout_no_crc` for the relaxed check.
const SECTION_COUNT: u32 = 17;
/// Lowest section_count we still accept on load (pre-J.1 = 7).
const MIN_SUPPORTED_SECTION_COUNT: u32 = 7;

/// Round `value` up to the next multiple of `align`. `align` must be > 0.
fn align_up(value: u64, align: u64) -> u64 {
    debug_assert!(align > 0);
    let rem = value % align;
    if rem == 0 {
        value
    } else {
        value + (align - rem)
    }
}

/// Local copy of `OffsetEntry` since the v2 module keeps it private.
/// The on-disk shape is identical.
#[derive(Debug, Clone, Copy)]
struct OffsetEntry {
    section_id: u32,
    offset: u64,
    length: u64,
}

fn write_entry(out: &mut Vec<u8>, e: &OffsetEntry) {
    let mut buf = [0u8; ENTRY_SIZE];
    buf[0..4].copy_from_slice(&e.section_id.to_le_bytes());
    // bytes 4..8 reserved (zero)
    buf[8..16].copy_from_slice(&e.offset.to_le_bytes());
    buf[16..24].copy_from_slice(&e.length.to_le_bytes());
    out.extend_from_slice(&buf);
}

fn read_entry(bytes: &[u8]) -> Result<OffsetEntry, BrainError> {
    if bytes.len() < ENTRY_SIZE {
        return Err(BrainError::Format(format!(
            "v3 offset entry truncated: {} bytes",
            bytes.len()
        )));
    }
    Ok(OffsetEntry {
        section_id: u32::from_le_bytes(bytes[0..4].try_into().unwrap()),
        offset: u64::from_le_bytes(bytes[8..16].try_into().unwrap()),
        length: u64::from_le_bytes(bytes[16..24].try_into().unwrap()),
    })
}

fn write_header(
    out: &mut Vec<u8>,
    node_count: u64,
    bond_count: u64,
    keyframe_count: u64,
    scar_count: u64,
) {
    let mut buf = [0u8; HEADER_SIZE];
    buf[0..8].copy_from_slice(&MAGIC_HEADER);
    buf[8..12].copy_from_slice(&FORMAT_VERSION_V3.to_le_bytes());
    buf[12..16].copy_from_slice(&0u32.to_le_bytes()); // flags = 0
    buf[16..24].copy_from_slice(&node_count.to_le_bytes());
    buf[24..32].copy_from_slice(&bond_count.to_le_bytes());
    buf[32..40].copy_from_slice(&keyframe_count.to_le_bytes());
    buf[40..48].copy_from_slice(&scar_count.to_le_bytes());
    buf[48..52].copy_from_slice(&SECTION_COUNT.to_le_bytes());
    // bytes 52..64 reserved (zero)
    out.extend_from_slice(&buf);
}

/// Parsed view of the v3 fixed header.
#[derive(Debug, Clone, Copy)]
struct V3Header {
    version: u32,
    flags: u32,
    node_count: u64,
    bond_count: u64,
    keyframe_count: u64,
    scar_count: u64,
    section_count: u32,
}

fn read_header(bytes: &[u8]) -> Result<V3Header, BrainError> {
    if bytes.len() < HEADER_SIZE {
        return Err(BrainError::Format(format!(
            "v3 file too short for header: {} bytes",
            bytes.len()
        )));
    }
    if bytes[0..8] != MAGIC_HEADER {
        return Err(BrainError::Format(format!(
            "v3 header magic mismatch: got {:02x?}",
            &bytes[0..8]
        )));
    }
    let version = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
    let flags = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
    let node_count = u64::from_le_bytes(bytes[16..24].try_into().unwrap());
    let bond_count = u64::from_le_bytes(bytes[24..32].try_into().unwrap());
    let keyframe_count = u64::from_le_bytes(bytes[32..40].try_into().unwrap());
    let scar_count = u64::from_le_bytes(bytes[40..48].try_into().unwrap());
    let section_count = u32::from_le_bytes(bytes[48..52].try_into().unwrap());
    Ok(V3Header {
        version,
        flags,
        node_count,
        bond_count,
        keyframe_count,
        scar_count,
        section_count,
    })
}

/// Serialize a `BrainFile` into a v3 binary buffer.
///
/// Returns `Vec<u8>` for ergonomics with the existing save path
/// (`fs::write(path, &bytes)`). The buffer is allocated by the global
/// allocator and is therefore at least 8-byte aligned in practice on
/// every platform Lu cares about; for tests that round-trip without
/// going through the file system we copy into an `rkyv::AlignedVec`
/// before validation (see the module tests).
pub fn write_v3(file: &BrainFile) -> Result<Vec<u8>, BrainError> {
    // ---- pre-serialize sections ---------------------------------------
    let net_archived = rkyv::to_bytes::<_, 4096>(&file.network)
        .map_err(|e| BrainError::Format(format!("rkyv serialize NetworkSnapshot: {}", e)))?;
    let scar_archived = rkyv::to_bytes::<_, 1024>(&file.scar_log)
        .map_err(|e| BrainError::Format(format!("rkyv serialize ScarLog: {}", e)))?;

    let interest_bytes =
        bincode::serialize(&file.interest_profile).map_err(BrainError::bincode)?;
    let meta_bytes =
        bincode::serialize(&file.meta_reflection).map_err(BrainError::bincode)?;
    let prop_bytes =
        bincode::serialize(&file.research_proposals).map_err(BrainError::bincode)?;
    let misc_blob = ResearchMiscBlob {
        config: file.research_config.clone(),
        next_id: file.next_research_proposal_id,
    };
    let misc_bytes = bincode::serialize(&misc_blob).map_err(BrainError::bincode)?;
    let gap_bytes =
        bincode::serialize(&file.gap_seed_frequency).map_err(BrainError::bincode)?;
    // Port LB.2: section #8 (WonderQueueState) is now a sentinel-empty
    // slot. Wonder state is canonical in section #11 (LB session memory)
    // as `SessionEntryKind::WonderItem` entries; section #8's payload
    // bytes are no longer emitted (offset=0, length=0). Reading
    // section #8 is preserved as a one-shot legacy fallback for
    // pre-LB.2 brains (see `de_bincode_optional` + the loader).
    let wonder_bytes: Vec<u8> = Vec::new();
    // Port T.4: serialize PendingLog snapshot as the optional
    // PendingLog section (id 9).
    let pending_bytes =
        bincode::serialize(&file.pending_log).map_err(BrainError::bincode)?;
    // Port T.8: serialize RejectionPatternsSnapshot as the optional
    // RejectionPatterns section (id 10).
    let rejection_bytes =
        bincode::serialize(&file.rejection_patterns).map_err(BrainError::bincode)?;
    // Port LB: serialize Left Brain SessionMemorySnapshot as the optional
    // SessionMemory section (id 11).
    let lb_bytes =
        bincode::serialize(&file.lb_session).map_err(BrainError::bincode)?;
    // Port CS: serialize ConversationalStyleSnapshot as the optional
    // ConversationalStyle section (id 12).
    let cs_bytes =
        bincode::serialize(&file.cs_style).map_err(BrainError::bincode)?;
    // Port Cat.4: serialize PackLedgerSnapshot as the optional
    // PackLedger section (id 13).
    let pack_bytes =
        bincode::serialize(&file.pack_ledger).map_err(BrainError::bincode)?;
    // Port Cat.3c: serialize the FROZEN Cat.3c shape into bincode
    // section #14 (RuntimeConfig). Cat.3c.1 keeps writing this so
    // pre-Cat.3c.1 binaries can still read their historical fields
    // unchanged. New runtime-config fields go into section #15 below.
    let runtime_config_bytes = file
        .runtime_config
        .encode_v14_section()
        .map_err(|e| BrainError::Bincode(e))?;
    // Port Cat.3c.1: serialize the FULL runtime-config snapshot as
    // JSON into section #15 (RuntimeConfigV2). JSON's named fields
    // give forward+backward compat at the field level so future
    // ports can extend RuntimeConfigSnapshot freely without breaking
    // older binaries (and vice versa).
    let runtime_config_v2_bytes = file
        .runtime_config
        .encode_v15_section()
        .map_err(|e| BrainError::Bincode(e))?;
    // Port Source-Persist: SourceOverlay section (id 16). Bincode
    // serialization mirrors the InterestProfile / PendingLog pattern.
    let source_overlay_bytes =
        bincode::serialize(&file.source_overlay).map_err(BrainError::bincode)?;
    // Port Cat.5: CapabilityPackLedger section (id 17). Schema-only
    // signed capability pack ledger. Bincode-encoded; defaults to
    // empty for pre-Cat.5 brains via `de_bincode_optional`.
    let cap_pack_ledger_bytes =
        bincode::serialize(&file.cap_pack_ledger).map_err(BrainError::bincode)?;

    // ---- compute counts for the header --------------------------------
    let keyframe_count: u64 = file
        .network
        .bonds
        .iter()
        .map(|b| b.keyframes().len() as u64)
        .sum();

    // ---- decide the on-disk order and offsets -------------------------
    //
    // Layout, in file-byte order, is:
    //   [0..64)             header
    //   [64..64+216)        offset table (9 * 24 bytes post T.4)
    //   [pad to align 16)   pad
    //   network rkyv blob   (id 1, aligned)
    //   [pad to align 16)
    //   scar rkyv blob      (id 4, aligned)
    //   interest blob       (id 2)
    //   meta blob           (id 3)
    //   research proposals  (id 5)
    //   research misc       (id 6)
    //   gap seed freq       (id 7)
    //   wonder queue state  (id 8)  -- J.1
    //   pending log         (id 9)  -- T.4
    //   [16-byte trailer]
    let header_end = HEADER_SIZE as u64;
    let table_bytes = (SECTION_COUNT as u64) * (ENTRY_SIZE as u64);
    let table_end = header_end + table_bytes;

    // Network rkyv (id 1)
    let net_offset = align_up(table_end, RKYV_ALIGN);
    let net_len = net_archived.len() as u64;

    // Scar rkyv (id 4)
    let scar_offset = align_up(net_offset + net_len, RKYV_ALIGN);
    let scar_len = scar_archived.len() as u64;

    // Bincode side tables — no alignment requirement.
    let interest_offset = scar_offset + scar_len;
    let interest_len = interest_bytes.len() as u64;

    let meta_offset = interest_offset + interest_len;
    let meta_len = meta_bytes.len() as u64;

    let prop_offset = meta_offset + meta_len;
    let prop_len = prop_bytes.len() as u64;

    let misc_offset = prop_offset + prop_len;
    let misc_len = misc_bytes.len() as u64;

    let gap_offset = misc_offset + misc_len;
    let gap_len = gap_bytes.len() as u64;

    // Port LB.2: sentinel section #8 — offset 0, length 0.
    // The slot stays in the offset table for backward-compat with
    // pre-LB.2 binaries (which iterate header.section_count entries).
    let wonder_offset: u64 = 0;
    let wonder_len: u64 = 0;

    // Port T.4: PendingLog section (id 9). Port LB.2: chains directly
    // off gap_offset+gap_len since the section #8 slot is sentinel-empty.
    let pending_offset = gap_offset + gap_len;
    let pending_len = pending_bytes.len() as u64;

    // Port T.8: RejectionPatterns section (id 10) follows pending.
    let rejection_offset = pending_offset + pending_len;
    let rejection_len = rejection_bytes.len() as u64;

    // Port LB: SessionMemory section (id 11) follows rejection.
    let lb_offset = rejection_offset + rejection_len;
    let lb_len = lb_bytes.len() as u64;

    // Port CS: ConversationalStyle section (id 12) follows lb_session.
    let cs_offset = lb_offset + lb_len;
    let cs_len = cs_bytes.len() as u64;

    // Port Cat.4: PackLedger section (id 13) follows ConversationalStyle.
    let pack_offset = cs_offset + cs_len;
    let pack_len = pack_bytes.len() as u64;

    // Port Cat.3c: RuntimeConfig section (id 14) follows PackLedger.
    let runtime_config_offset = pack_offset + pack_len;
    let runtime_config_len = runtime_config_bytes.len() as u64;

    // Port Cat.3c.1: RuntimeConfigV2 section (id 15) follows RuntimeConfig.
    let runtime_config_v2_offset = runtime_config_offset + runtime_config_len;
    let runtime_config_v2_len = runtime_config_v2_bytes.len() as u64;

    // Port Source-Persist: SourceOverlay section (id 16) follows RuntimeConfigV2.
    let source_overlay_offset = runtime_config_v2_offset + runtime_config_v2_len;
    let source_overlay_len = source_overlay_bytes.len() as u64;

    // Port Cat.5: CapabilityPackLedger section (id 17) follows SourceOverlay.
    let cap_pack_ledger_offset = source_overlay_offset + source_overlay_len;
    let cap_pack_ledger_len = cap_pack_ledger_bytes.len() as u64;

    let data_end = cap_pack_ledger_offset + cap_pack_ledger_len;
    let total_size = data_end as usize + TRAILER_SIZE;

    // ---- allocate + emit ---------------------------------------------
    let mut out: Vec<u8> = Vec::with_capacity(total_size);

    // Header
    write_header(
        &mut out,
        file.network.nodes.len() as u64,
        file.network.bonds.len() as u64,
        keyframe_count,
        file.scar_log.scars.len() as u64,
    );
    debug_assert_eq!(out.len() as u64, header_end);

    // Offset table — written in section_id order so the on-disk table
    // is deterministic. Order doesn't matter for the reader (it builds
    // a HashMap), but determinism makes the file easier to diff.
    let entries: [OffsetEntry; SECTION_COUNT as usize] = [
        OffsetEntry { section_id: SectionIdV3::Network as u32,           offset: net_offset,      length: net_len },
        OffsetEntry { section_id: SectionIdV3::InterestProfile as u32,   offset: interest_offset, length: interest_len },
        OffsetEntry { section_id: SectionIdV3::MetaReflection as u32,    offset: meta_offset,     length: meta_len },
        OffsetEntry { section_id: SectionIdV3::ScarLog as u32,           offset: scar_offset,     length: scar_len },
        OffsetEntry { section_id: SectionIdV3::ResearchProposals as u32, offset: prop_offset,     length: prop_len },
        OffsetEntry { section_id: SectionIdV3::ResearchMisc as u32,      offset: misc_offset,     length: misc_len },
        OffsetEntry { section_id: SectionIdV3::GapSeedFrequency as u32,  offset: gap_offset,      length: gap_len },
        OffsetEntry { section_id: SectionIdV3::WonderQueueState as u32,  offset: wonder_offset,   length: wonder_len },
        // Port T.4: PendingLog section (id 9).
        OffsetEntry { section_id: SectionIdV3::PendingLog as u32,        offset: pending_offset,  length: pending_len },
        // Port T.8: RejectionPatterns section (id 10).
        OffsetEntry { section_id: SectionIdV3::RejectionPatterns as u32, offset: rejection_offset, length: rejection_len },
        // Port LB: SessionMemory section (id 11).
        OffsetEntry { section_id: SectionIdV3::SessionMemory as u32,     offset: lb_offset,        length: lb_len },
        // Port CS: ConversationalStyle section (id 12).
        OffsetEntry { section_id: SectionIdV3::ConversationalStyle as u32, offset: cs_offset, length: cs_len },
        // Port Cat.4: PackLedger section (id 13).
        OffsetEntry { section_id: SectionIdV3::PackLedger as u32,          offset: pack_offset, length: pack_len },
        // Port Cat.3c: RuntimeConfig section (id 14).
        OffsetEntry { section_id: SectionIdV3::RuntimeConfig as u32,       offset: runtime_config_offset, length: runtime_config_len },
        // Port Cat.3c.1: RuntimeConfigV2 section (id 15).
        OffsetEntry { section_id: SectionIdV3::RuntimeConfigV2 as u32,     offset: runtime_config_v2_offset, length: runtime_config_v2_len },
        // Port Source-Persist: SourceOverlay section (id 16).
        OffsetEntry { section_id: SectionIdV3::SourceOverlay as u32,       offset: source_overlay_offset,    length: source_overlay_len },
        // Port Cat.5: CapabilityPackLedger section (id 17).
        OffsetEntry { section_id: SectionIdV3::CapabilityPackLedger as u32, offset: cap_pack_ledger_offset, length: cap_pack_ledger_len },
    ];
    for e in &entries {
        write_entry(&mut out, e);
    }
    debug_assert_eq!(out.len() as u64, table_end);

    // Pad to align 16 before the network rkyv section.
    while (out.len() as u64) < net_offset {
        out.push(0);
    }
    debug_assert_eq!(out.len() as u64 % RKYV_ALIGN, 0);
    out.extend_from_slice(&net_archived);

    // Pad to align 16 before scar rkyv section.
    while (out.len() as u64) < scar_offset {
        out.push(0);
    }
    debug_assert_eq!(out.len() as u64 % RKYV_ALIGN, 0);
    out.extend_from_slice(&scar_archived);

    // Bincode side-tables — write in offset-order.
    debug_assert_eq!(out.len() as u64, interest_offset);
    out.extend_from_slice(&interest_bytes);
    debug_assert_eq!(out.len() as u64, meta_offset);
    out.extend_from_slice(&meta_bytes);
    debug_assert_eq!(out.len() as u64, prop_offset);
    out.extend_from_slice(&prop_bytes);
    debug_assert_eq!(out.len() as u64, misc_offset);
    out.extend_from_slice(&misc_bytes);
    debug_assert_eq!(out.len() as u64, gap_offset);
    out.extend_from_slice(&gap_bytes);
    // Port LB.2: section #8 is sentinel-empty — no payload to append.
    let _ = wonder_bytes;
    let _ = wonder_offset;
    // Port T.4: PendingLog section.
    debug_assert_eq!(out.len() as u64, pending_offset);
    out.extend_from_slice(&pending_bytes);
    // Port T.8: RejectionPatterns section.
    debug_assert_eq!(out.len() as u64, rejection_offset);
    out.extend_from_slice(&rejection_bytes);
    // Port LB: SessionMemory section.
    debug_assert_eq!(out.len() as u64, lb_offset);
    out.extend_from_slice(&lb_bytes);
    // Port CS: ConversationalStyle section.
    debug_assert_eq!(out.len() as u64, cs_offset);
    out.extend_from_slice(&cs_bytes);
    // Port Cat.4: PackLedger section.
    debug_assert_eq!(out.len() as u64, pack_offset);
    out.extend_from_slice(&pack_bytes);
    // Port Cat.3c: RuntimeConfig section.
    debug_assert_eq!(out.len() as u64, runtime_config_offset);
    out.extend_from_slice(&runtime_config_bytes);
    // Port Cat.3c.1: RuntimeConfigV2 section.
    debug_assert_eq!(out.len() as u64, runtime_config_v2_offset);
    out.extend_from_slice(&runtime_config_v2_bytes);
    // Port Source-Persist: SourceOverlay section.
    debug_assert_eq!(out.len() as u64, source_overlay_offset);
    out.extend_from_slice(&source_overlay_bytes);
    // Port Cat.5: CapabilityPackLedger section.
    debug_assert_eq!(out.len() as u64, cap_pack_ledger_offset);
    out.extend_from_slice(&cap_pack_ledger_bytes);
    debug_assert_eq!(out.len() as u64, data_end);

    // Trailer: magic + crc32 + reserved.
    let crc = crc32fast::hash(&out);
    out.extend_from_slice(&MAGIC_TRAILER);
    out.extend_from_slice(&crc.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    debug_assert_eq!(out.len(), total_size);

    Ok(out)
}

/// Internal: validate trailer, parse header + offset table, and return
/// the section map. Shared between the bytes and the path readers.
fn parse_layout<'a>(bytes: &'a [u8]) -> Result<(V3Header, HashMap<u32, &'a [u8]>), BrainError> {
    if bytes.len() < HEADER_SIZE + TRAILER_SIZE {
        return Err(BrainError::Format(format!(
            "v3 file too short for layout: {} bytes",
            bytes.len()
        )));
    }
    let trailer_pos = bytes.len() - TRAILER_SIZE;
    if bytes[trailer_pos..trailer_pos + 8] != MAGIC_TRAILER {
        return Err(BrainError::Format(
            "v3 trailer magic mismatch — file truncated or corrupted".into(),
        ));
    }
    let stored_crc =
        u32::from_le_bytes(bytes[trailer_pos + 8..trailer_pos + 12].try_into().unwrap());
    let actual_crc = crc32fast::hash(&bytes[..trailer_pos]);
    if stored_crc != actual_crc {
        return Err(BrainError::Format(format!(
            "v3 checksum mismatch: stored 0x{:08x}, computed 0x{:08x}",
            stored_crc, actual_crc
        )));
    }

    let header = read_header(bytes)?;
    if header.version != FORMAT_VERSION_V3 {
        return Err(BrainError::VersionMismatch(header.version));
    }
    // Phase 3: bit 0 of `flags` selects the per-block compressed
    // network section. Any other bit being set is still a future-format
    // signal we don't understand.
    let known_flags: u32 = crate::compress::FORMAT_FLAG_COMPRESSED
        | crate::encrypt::FORMAT_FLAG_ENCRYPTED;
    if (header.flags & !known_flags) != 0 {
        return Err(BrainError::Format(format!(
            "v3 flags=0x{:08x} not understood — newer format from a future port?",
            header.flags
        )));
    }
    // Port J.1.1: only enforce a *lower* bound on section_count. A future
    // format port may add new sections (e.g. J.1 added WonderQueueState=8,
    // bumping the count from 7 to 8); pre-J.1 readers rejected that on the
    // upper bound, locking us into one-way upgrades. Now any file with at
    // least the floor of known required sections loads, and any unknown
    // section IDs encountered while walking the offset table are skipped
    // (see the loop below). Older files (count=7) are still tolerated
    // because WonderQueueState is read via `de_bincode_optional`.
    if header.section_count < MIN_SUPPORTED_SECTION_COUNT {
        return Err(BrainError::Format(format!(
            "v3 unexpected section_count={} (must be >= {} for the required known sections)",
            header.section_count, MIN_SUPPORTED_SECTION_COUNT
        )));
    }

    let table_start = HEADER_SIZE;
    let table_end = table_start + (header.section_count as usize) * ENTRY_SIZE;
    if bytes.len() < table_end + TRAILER_SIZE {
        return Err(BrainError::Format(
            "v3 file too short for offset table".into(),
        ));
    }
    let mut sections: HashMap<u32, &[u8]> = HashMap::with_capacity(header.section_count as usize);
    for i in 0..(header.section_count as usize) {
        let entry_start = table_start + i * ENTRY_SIZE;
        let entry = read_entry(&bytes[entry_start..entry_start + ENTRY_SIZE])?;
        let start = entry.offset as usize;
        let end = start.saturating_add(entry.length as usize);
        if end > trailer_pos {
            return Err(BrainError::Format(format!(
                "v3 section {} runs past data region: offset={} len={} trailer_pos={}",
                entry.section_id, entry.offset, entry.length, trailer_pos
            )));
        }
        // rkyv sections must be 16-aligned within the file. The reader
        // doesn't fundamentally need to enforce this (the bytes are
        // validated by check_archived_root anyway), but a misaligned
        // archived section is undefined behavior on some targets, so
        // we reject early with a clear error.
        let needs_rkyv_align = if header.flags & crate::compress::FORMAT_FLAG_COMPRESSED != 0 {
            // Phase 3: the compressed network payload is byte-addressed
            // and does NOT need to satisfy rkyv's 16-byte alignment;
            // each per-entity rkyv archive lives inside a decompressed
            // AlignedVec instead. The ScarLog section is still
            // uncompressed rkyv, so it still needs alignment.
            entry.section_id == SectionIdV3::ScarLog as u32
        } else {
            entry.section_id == SectionIdV3::Network as u32
                || entry.section_id == SectionIdV3::ScarLog as u32
        };
        if needs_rkyv_align && entry.offset % RKYV_ALIGN != 0 {
            return Err(BrainError::Format(format!(
                "v3 rkyv section {} at offset {} is not {}-byte aligned",
                entry.section_id, entry.offset, RKYV_ALIGN
            )));
        }
        // Port J.1.1: forward-compat tolerance. If the section id is not
        // one this build knows about, we still record its bytes in the
        // section map, but no `require()` call will look it up — so the
        // unknown payload is effectively skipped. Set LU_BRAIN_DEBUG=1 in
        // the environment to surface a one-line note per unknown section.
        if SectionIdV3::from_u32(entry.section_id).is_none()
            && std::env::var_os("LU_BRAIN_DEBUG").is_some()
        {
            eprintln!(
                "v3 reader (J.1.1): skipping unknown section id={} offset={} length={}",
                entry.section_id, entry.offset, entry.length
            );
        }
        sections.insert(entry.section_id, &bytes[start..end]);
    }

    Ok((header, sections))
}

fn require<'a>(
    sections: &HashMap<u32, &'a [u8]>,
    id: SectionIdV3,
) -> Result<&'a [u8], BrainError> {
    sections
        .get(&(id as u32))
        .copied()
        .ok_or_else(|| BrainError::Format(format!("v3 file missing required section {:?}", id)))
}


/// Extract the rkyv-archived NetworkSnapshot section from a v3 byte
/// slice. Validates the trailer/CRC and the offset table; returns the
/// raw section bytes (caller is responsible for `rkyv::check_archived_root`).
///
/// Used by `crate::archive::NetworkArchive::from_mmap` to find the
/// network blob inside an mmap'd v3 file without re-parsing the
/// whole file structure.
pub fn extract_archived_network_blob<'a>(bytes: &'a [u8]) -> Result<&'a [u8], BrainError> {
    let (_header, sections) = parse_layout(bytes)?;
    require(&sections, SectionIdV3::Network)
}

/// Extract the rkyv-archived ScarLog section from a v3 byte slice.
/// Same contract as `extract_archived_network_blob`.
pub fn extract_archived_scar_blob<'a>(bytes: &'a [u8]) -> Result<&'a [u8], BrainError> {
    let (_header, sections) = parse_layout(bytes)?;
    require(&sections, SectionIdV3::ScarLog)
}

/// Storage v2 Slice 2c — parse the v3 layout WITHOUT running the
/// CRC32 over the data region. The mmap-zero-copy load path uses
/// this so opening a brain file does not page-in every byte for the
/// CRC sweep — the bulk of the 424 MB file stays cold until
/// individual queries fault in only what they touch.
///
/// We still verify:
///   - file is large enough for header + offset table + trailer
///   - trailer magic bytes match
///   - header magic + version match
///   - offset table fits the data region
///   - rkyv sections are 16-byte aligned
///
/// We do NOT verify the CRC32 — for trusted local brain files this
/// is acceptable, and the rkyv-archived sections themselves are
/// validated structurally on first access via
/// `rkyv::check_archived_root` (only when explicitly opted into via
/// the `_validated` extractors below).
fn parse_layout_no_crc<'a>(
    bytes: &'a [u8],
) -> Result<(V3Header, HashMap<u32, &'a [u8]>), BrainError> {
    if bytes.len() < HEADER_SIZE + TRAILER_SIZE {
        return Err(BrainError::Format(format!(
            "v3 file too short for layout: {} bytes",
            bytes.len()
        )));
    }
    let trailer_pos = bytes.len() - TRAILER_SIZE;
    if bytes[trailer_pos..trailer_pos + 8] != MAGIC_TRAILER {
        return Err(BrainError::Format(
            "v3 trailer magic mismatch — file truncated or corrupted".into(),
        ));
    }
    // (CRC32 over data region INTENTIONALLY skipped here — see
    // function-level docstring.)
    let header = read_header(bytes)?;
    if header.version != FORMAT_VERSION_V3 {
        return Err(BrainError::VersionMismatch(header.version));
    }
    // Phase 3: bit 0 of `flags` selects the per-block compressed
    // network section. Any other bit being set is still a future-format
    // signal we don't understand.
    let known_flags: u32 = crate::compress::FORMAT_FLAG_COMPRESSED
        | crate::encrypt::FORMAT_FLAG_ENCRYPTED;
    if (header.flags & !known_flags) != 0 {
        return Err(BrainError::Format(format!(
            "v3 flags=0x{:08x} not understood — newer format from a future port?",
            header.flags
        )));
    }
    // Port J.1.1: only enforce a *lower* bound on section_count (mirrors
    // `parse_layout` above). Unknown section IDs in the offset table are
    // skipped further down so a J binary can still load a J.1 brain, and
    // a J.1.1 binary can load a brain produced by a hypothetical future
    // port that adds section IDs > 8.
    if header.section_count < MIN_SUPPORTED_SECTION_COUNT {
        return Err(BrainError::Format(format!(
            "v3 unexpected section_count={} (must be >= {} for the required known sections)",
            header.section_count, MIN_SUPPORTED_SECTION_COUNT
        )));
    }
    let table_start = HEADER_SIZE;
    let table_end = table_start + (header.section_count as usize) * ENTRY_SIZE;
    if bytes.len() < table_end + TRAILER_SIZE {
        return Err(BrainError::Format(
            "v3 file too short for offset table".into(),
        ));
    }
    let mut sections: HashMap<u32, &[u8]> = HashMap::with_capacity(header.section_count as usize);
    for i in 0..(header.section_count as usize) {
        let entry_start = table_start + i * ENTRY_SIZE;
        let entry = read_entry(&bytes[entry_start..entry_start + ENTRY_SIZE])?;
        let start = entry.offset as usize;
        let end = start.saturating_add(entry.length as usize);
        if end > trailer_pos {
            return Err(BrainError::Format(format!(
                "v3 section {} runs past data region: offset={} len={} trailer_pos={}",
                entry.section_id, entry.offset, entry.length, trailer_pos
            )));
        }
        let needs_rkyv_align = if header.flags & crate::compress::FORMAT_FLAG_COMPRESSED != 0 {
            // Phase 3: the compressed network payload is byte-addressed
            // and does NOT need to satisfy rkyv's 16-byte alignment;
            // each per-entity rkyv archive lives inside a decompressed
            // AlignedVec instead. The ScarLog section is still
            // uncompressed rkyv, so it still needs alignment.
            entry.section_id == SectionIdV3::ScarLog as u32
        } else {
            entry.section_id == SectionIdV3::Network as u32
                || entry.section_id == SectionIdV3::ScarLog as u32
        };
        if needs_rkyv_align && entry.offset % RKYV_ALIGN != 0 {
            return Err(BrainError::Format(format!(
                "v3 rkyv section {} at offset {} is not {}-byte aligned",
                entry.section_id, entry.offset, RKYV_ALIGN
            )));
        }
        // Port J.1.1: forward-compat tolerance. If the section id is not
        // one this build knows about, we still record its bytes in the
        // section map, but no `require()` call will look it up — so the
        // unknown payload is effectively skipped. Set LU_BRAIN_DEBUG=1 in
        // the environment to surface a one-line note per unknown section.
        if SectionIdV3::from_u32(entry.section_id).is_none()
            && std::env::var_os("LU_BRAIN_DEBUG").is_some()
        {
            eprintln!(
                "v3 reader (J.1.1): skipping unknown section id={} offset={} length={}",
                entry.section_id, entry.offset, entry.length
            );
        }
        sections.insert(entry.section_id, &bytes[start..end]);
    }
    Ok((header, sections))
}

/// Storage v2 Slice 2c — extract the network blob WITHOUT a CRC32
/// sweep over the file. Used by `NetworkArchive::from_mmap` so the
/// zero-copy load path keeps the bulk of the file cold.
pub fn extract_archived_network_blob_no_crc<'a>(bytes: &'a [u8]) -> Result<&'a [u8], BrainError> {
    let (_header, sections) = parse_layout_no_crc(bytes)?;
    require(&sections, SectionIdV3::Network)
}

/// Same contract as `extract_archived_network_blob_no_crc` for the
/// ScarLog section.
pub fn extract_archived_scar_blob_no_crc<'a>(bytes: &'a [u8]) -> Result<&'a [u8], BrainError> {
    let (_header, sections) = parse_layout_no_crc(bytes)?;
    require(&sections, SectionIdV3::ScarLog)
}

/// Storage v2 Slice 2c — eager-deserialize side tables WITHOUT a
/// CRC32 sweep over the network section. Side-table sections are
/// small (KB to single-digit MB range), so we still touch every byte
/// of those — but the network section's 400+ MB stays cold.
pub(crate) fn read_v3_side_tables_no_crc<'a>(
    bytes: &'a [u8],
) -> Result<(InterestProfile, MetaReflection, Vec<ResearchProposal>, ResearchMiscBlob, HashMap<Seed, u32>, WonderState, crate::pending::PendingLogSnapshot, crate::rejection::RejectionPatternsSnapshot, crate::lb_session::SessionMemorySnapshot, crate::style::ConversationalStyleSnapshot, crate::pack::PackLedgerSnapshot, crate::runtime_config::RuntimeConfigSnapshot, bool, crate::source::SourceOverlay, crate::capability_pack::CapabilityPackLedgerSnapshot), BrainError> {
    let (_header, sections) = parse_layout_no_crc(bytes)?;
    let interest: InterestProfile = de_bincode(&sections, SectionIdV3::InterestProfile)?;
    let meta: MetaReflection = de_bincode(&sections, SectionIdV3::MetaReflection)?;
    let props: Vec<ResearchProposal> = de_bincode(&sections, SectionIdV3::ResearchProposals)?;
    let misc: ResearchMiscBlob = de_bincode(&sections, SectionIdV3::ResearchMisc)?;
    let gap: HashMap<Seed, u32> = de_bincode(&sections, SectionIdV3::GapSeedFrequency)?;
    // Port J.1: optional read — pre-J.1 files (section_count=7) come up empty.
    let wonder: WonderState = de_bincode_optional(&sections, SectionIdV3::WonderQueueState)?;
    // Port T.4: optional read — pre-T.4 files (section_count<=8) come up empty.
    let pending_log: crate::pending::PendingLogSnapshot =
        de_bincode_optional(&sections, SectionIdV3::PendingLog)?;
    // Port T.8: optional read -- pre-T.8 files (section_count<=9) come up empty.
    let rejection_patterns: crate::rejection::RejectionPatternsSnapshot =
        de_bincode_optional(&sections, SectionIdV3::RejectionPatterns)?;
    // Port LB: optional read -- pre-LB files (section_count<=10) come up empty.
    let lb_session: crate::lb_session::SessionMemorySnapshot =
        de_bincode_optional(&sections, SectionIdV3::SessionMemory)?;
    let cs_style: crate::style::ConversationalStyleSnapshot =
        de_bincode_optional(&sections, SectionIdV3::ConversationalStyle)?;
    // Port Cat.4: optional PackLedger snapshot (default empty for pre-Cat.4 files).
    let pack_ledger: crate::pack::PackLedgerSnapshot =
        de_bincode_optional(&sections, SectionIdV3::PackLedger)?;
    // Port Cat.3c.1: dual-section RuntimeConfig read.
    //   - Section #15 (RuntimeConfigV2, JSON) is the authoritative,
    //     forward-compat payload. Use if present.
    //   - Section #14 (RuntimeConfig, bincode, frozen Cat.3c shape)
    //     is the legacy payload. Fall back if #15 is absent.
    //   - Neither present => default (pre-Cat.3c brain — sidecar
    //     migration runs at the brain.rs layer).
    let runtime_config_v2_present = sections
        .contains_key(&(SectionIdV3::RuntimeConfigV2 as u32));
    let runtime_config_present = sections
        .contains_key(&(SectionIdV3::RuntimeConfig as u32))
        || runtime_config_v2_present;
    let runtime_config: crate::runtime_config::RuntimeConfigSnapshot =
        if let Some(v2_bytes) = sections.get(&(SectionIdV3::RuntimeConfigV2 as u32)) {
            crate::runtime_config::RuntimeConfigSnapshot::decode_v15_section(v2_bytes)
                .map_err(|e| BrainError::Bincode(format!(
                    "v3 section RuntimeConfigV2 (id 15, {} bytes): {}",
                    v2_bytes.len(), e,
                )))?
        } else if let Some(v1_bytes) = sections.get(&(SectionIdV3::RuntimeConfig as u32)) {
            crate::runtime_config::RuntimeConfigSnapshot::decode_v14_section(v1_bytes)
                .map_err(|e| BrainError::Bincode(format!(
                    "v3 section RuntimeConfig (id 14, {} bytes): {}",
                    v1_bytes.len(), e,
                )))?
        } else {
            crate::runtime_config::RuntimeConfigSnapshot::default()
        };
    // Port Source-Persist: optional SourceOverlay (default empty for
    // pre-Source-Persist files via `de_bincode_optional`).
    let source_overlay: crate::source::SourceOverlay =
        de_bincode_optional(&sections, SectionIdV3::SourceOverlay)?;
    // Port Cat.5: optional CapabilityPackLedger snapshot (default empty
    // for pre-Cat.5 files via `de_bincode_optional`).
    let cap_pack_ledger: crate::capability_pack::CapabilityPackLedgerSnapshot =
        de_bincode_optional(&sections, SectionIdV3::CapabilityPackLedger)?;
    Ok((interest, meta, props, misc, gap, wonder, pending_log, rejection_patterns, lb_session, cs_style, pack_ledger, runtime_config, runtime_config_present, source_overlay, cap_pack_ledger))
}

/// Eager-deserialize the bincoded side tables from a v3 byte slice.
/// The zero-copy load path uses this to populate the heap-resident
/// fields of `BrainFile` (everything except the network section).
#[allow(dead_code)] // Audit fix internal notes: retained for future Phase-3.5
// incremental-load work; documented as intentionally unreferenced.
pub(crate) fn read_v3_side_tables<'a>(
    bytes: &'a [u8],
) -> Result<(InterestProfile, MetaReflection, Vec<ResearchProposal>, ResearchMiscBlob, HashMap<Seed, u32>, WonderState, crate::pending::PendingLogSnapshot, crate::rejection::RejectionPatternsSnapshot, crate::lb_session::SessionMemorySnapshot, crate::style::ConversationalStyleSnapshot, crate::pack::PackLedgerSnapshot, crate::runtime_config::RuntimeConfigSnapshot, bool, crate::source::SourceOverlay, crate::capability_pack::CapabilityPackLedgerSnapshot), BrainError> {
    let (_header, sections) = parse_layout(bytes)?;
    let interest: InterestProfile = de_bincode(&sections, SectionIdV3::InterestProfile)?;
    let meta: MetaReflection = de_bincode(&sections, SectionIdV3::MetaReflection)?;
    let props: Vec<ResearchProposal> = de_bincode(&sections, SectionIdV3::ResearchProposals)?;
    let misc: ResearchMiscBlob = de_bincode(&sections, SectionIdV3::ResearchMisc)?;
    let gap: HashMap<Seed, u32> = de_bincode(&sections, SectionIdV3::GapSeedFrequency)?;
    let wonder: WonderState = de_bincode_optional(&sections, SectionIdV3::WonderQueueState)?;
    // Port T.4: optional PendingLog snapshot, default empty for pre-T.4 brains.
    let pending_log: crate::pending::PendingLogSnapshot =
        de_bincode_optional(&sections, SectionIdV3::PendingLog)?;
    // Port T.8: optional read -- pre-T.8 files (section_count<=9) come up empty.
    let rejection_patterns: crate::rejection::RejectionPatternsSnapshot =
        de_bincode_optional(&sections, SectionIdV3::RejectionPatterns)?;
    // Port LB: optional read -- pre-LB files (section_count<=10) come up empty.
    let lb_session: crate::lb_session::SessionMemorySnapshot =
        de_bincode_optional(&sections, SectionIdV3::SessionMemory)?;
    let cs_style: crate::style::ConversationalStyleSnapshot =
        de_bincode_optional(&sections, SectionIdV3::ConversationalStyle)?;
    // Port Cat.4: optional PackLedger snapshot (default empty for pre-Cat.4 files).
    let pack_ledger: crate::pack::PackLedgerSnapshot =
        de_bincode_optional(&sections, SectionIdV3::PackLedger)?;
    // Port Cat.3c.1: dual-section RuntimeConfig read.
    //   - Section #15 (RuntimeConfigV2, JSON) is the authoritative,
    //     forward-compat payload. Use if present.
    //   - Section #14 (RuntimeConfig, bincode, frozen Cat.3c shape)
    //     is the legacy payload. Fall back if #15 is absent.
    //   - Neither present => default (pre-Cat.3c brain — sidecar
    //     migration runs at the brain.rs layer).
    let runtime_config_v2_present = sections
        .contains_key(&(SectionIdV3::RuntimeConfigV2 as u32));
    let runtime_config_present = sections
        .contains_key(&(SectionIdV3::RuntimeConfig as u32))
        || runtime_config_v2_present;
    let runtime_config: crate::runtime_config::RuntimeConfigSnapshot =
        if let Some(v2_bytes) = sections.get(&(SectionIdV3::RuntimeConfigV2 as u32)) {
            crate::runtime_config::RuntimeConfigSnapshot::decode_v15_section(v2_bytes)
                .map_err(|e| BrainError::Bincode(format!(
                    "v3 section RuntimeConfigV2 (id 15, {} bytes): {}",
                    v2_bytes.len(), e,
                )))?
        } else if let Some(v1_bytes) = sections.get(&(SectionIdV3::RuntimeConfig as u32)) {
            crate::runtime_config::RuntimeConfigSnapshot::decode_v14_section(v1_bytes)
                .map_err(|e| BrainError::Bincode(format!(
                    "v3 section RuntimeConfig (id 14, {} bytes): {}",
                    v1_bytes.len(), e,
                )))?
        } else {
            crate::runtime_config::RuntimeConfigSnapshot::default()
        };
    // Port Source-Persist: optional SourceOverlay (default empty for
    // pre-Source-Persist files via `de_bincode_optional`).
    let source_overlay: crate::source::SourceOverlay =
        de_bincode_optional(&sections, SectionIdV3::SourceOverlay)?;
    // Port Cat.5: optional CapabilityPackLedger snapshot (default empty
    // for pre-Cat.5 files via `de_bincode_optional`).
    let cap_pack_ledger: crate::capability_pack::CapabilityPackLedgerSnapshot =
        de_bincode_optional(&sections, SectionIdV3::CapabilityPackLedger)?;
    Ok((interest, meta, props, misc, gap, wonder, pending_log, rejection_patterns, lb_session, cs_style, pack_ledger, runtime_config, runtime_config_present, source_overlay, cap_pack_ledger))
}

fn de_bincode<'a, T: serde::de::DeserializeOwned>(
    sections: &HashMap<u32, &'a [u8]>,
    id: SectionIdV3,
) -> Result<T, BrainError> {
    let blob = require(sections, id)?;
    bincode::deserialize::<T>(blob).map_err(|e| {
        BrainError::Bincode(format!(
            "v3 section {:?} (id {}, {} bytes): {}",
            id,
            id as u32,
            blob.len(),
            e
        ))
    })
}

/// Port J.1: optional bincode read. Returns `Default::default()` if
/// the section is absent — used for `WonderQueueState` so old files
/// (section_count=7) load with empty state.
fn de_bincode_optional<'a, T: serde::de::DeserializeOwned + Default>(
    sections: &HashMap<u32, &'a [u8]>,
    id: SectionIdV3,
) -> Result<T, BrainError> {
    match sections.get(&(id as u32)) {
        None => Ok(T::default()),
        // Port LB.2: a sentinel-empty section (offset=0, length=0)
        // also deserializes to default — used for `WonderQueueState`
        // post-LB.2 so the LB.2-saved brain's empty section #8 loads
        // as `WonderState::default()`.
        Some(blob) if blob.is_empty() => Ok(T::default()),
        Some(blob) => bincode::deserialize::<T>(blob).map_err(|e| {
            BrainError::Bincode(format!(
                "v3 section {:?} (id {}, {} bytes): {}",
                id,
                id as u32,
                blob.len(),
                e
            ))
        }),
    }
}

/// Parse a v3 byte slice back into a `BrainFile`.
///
/// Caller is responsible for ensuring the slice has at least 16-byte
/// alignment so the rkyv archived sections inside it can be validated
/// without UB. mmap'd files (≥4096-byte aligned) and `rkyv::AlignedVec`
/// buffers both qualify; a plain `Vec<u8>` *usually* qualifies in
/// practice but is not guaranteed by the language. Use
/// `read_v3_path` for production loading from disk; this entry point
/// exists for tests and for in-memory uses where alignment is already
/// known to hold.
pub fn read_v3_bytes(bytes: &[u8]) -> Result<BrainFile, BrainError> {
    let (header, sections) = parse_layout(bytes)?;

    // ---- archived sections via rkyv -----------------------------------
    let net_blob = require(&sections, SectionIdV3::Network)?;
    let archived_net = rkyv::check_archived_root::<NetworkSnapshot>(net_blob).map_err(|e| {
        BrainError::Format(format!(
            "v3 NetworkSnapshot validation failed ({} bytes): {}",
            net_blob.len(),
            e
        ))
    })?;
    // The `Infallible` deserializer's error type is statically
    // uninhabited, so `expect` here is genuinely unreachable rather
    // than a hopeful guess.
    let network: NetworkSnapshot = archived_net
        .deserialize(&mut rkyv::Infallible)
        .expect("rkyv Infallible deserializer cannot fail");

    let scar_blob = require(&sections, SectionIdV3::ScarLog)?;
    let archived_scar = rkyv::check_archived_root::<ScarLog>(scar_blob).map_err(|e| {
        BrainError::Format(format!(
            "v3 ScarLog validation failed ({} bytes): {}",
            scar_blob.len(),
            e
        ))
    })?;
    let scar_log: ScarLog = archived_scar
        .deserialize(&mut rkyv::Infallible)
        .expect("rkyv Infallible deserializer cannot fail");

    // ---- bincode side-tables ------------------------------------------
    let interest_profile: InterestProfile = de_bincode(&sections, SectionIdV3::InterestProfile)?;
    let meta_reflection: MetaReflection = de_bincode(&sections, SectionIdV3::MetaReflection)?;
    let research_proposals: Vec<ResearchProposal> =
        de_bincode(&sections, SectionIdV3::ResearchProposals)?;
    let misc: ResearchMiscBlob = de_bincode(&sections, SectionIdV3::ResearchMisc)?;
    let gap_seed_frequency: HashMap<Seed, u32> =
        de_bincode(&sections, SectionIdV3::GapSeedFrequency)?;
    // Port J.1: optional WonderQueueState (default empty for pre-J.1 files).
    let wonder_state: WonderState =
        de_bincode_optional(&sections, SectionIdV3::WonderQueueState)?;
    // Port T.4: optional PendingLog snapshot (default empty for pre-T.4 files).
    let pending_log: crate::pending::PendingLogSnapshot =
        de_bincode_optional(&sections, SectionIdV3::PendingLog)?;
    // Port T.8: optional RejectionPatterns snapshot (default empty for pre-T.8 files).
    let rejection_patterns: crate::rejection::RejectionPatternsSnapshot =
        de_bincode_optional(&sections, SectionIdV3::RejectionPatterns)?;
    // Port LB: optional SessionMemory snapshot (default empty for pre-LB files).
    let lb_session: crate::lb_session::SessionMemorySnapshot =
        de_bincode_optional(&sections, SectionIdV3::SessionMemory)?;
    // Port CS: optional ConversationalStyle snapshot (default empty for pre-CS files).
    let cs_style: crate::style::ConversationalStyleSnapshot =
        de_bincode_optional(&sections, SectionIdV3::ConversationalStyle)?;
    // Port Cat.4: optional PackLedger snapshot (default empty for pre-Cat.4 files).
    let pack_ledger: crate::pack::PackLedgerSnapshot =
        de_bincode_optional(&sections, SectionIdV3::PackLedger)?;
    // Port Cat.3c.1: dual-section RuntimeConfig read.
    //   - Section #15 (RuntimeConfigV2, JSON) is the authoritative,
    //     forward-compat payload. Use if present.
    //   - Section #14 (RuntimeConfig, bincode, frozen Cat.3c shape)
    //     is the legacy payload. Fall back if #15 is absent.
    //   - Neither present => default (pre-Cat.3c brain — sidecar
    //     migration runs at the brain.rs layer).
    let runtime_config_v2_present = sections
        .contains_key(&(SectionIdV3::RuntimeConfigV2 as u32));
    let runtime_config_present = sections
        .contains_key(&(SectionIdV3::RuntimeConfig as u32))
        || runtime_config_v2_present;
    let runtime_config: crate::runtime_config::RuntimeConfigSnapshot =
        if let Some(v2_bytes) = sections.get(&(SectionIdV3::RuntimeConfigV2 as u32)) {
            crate::runtime_config::RuntimeConfigSnapshot::decode_v15_section(v2_bytes)
                .map_err(|e| BrainError::Bincode(format!(
                    "v3 section RuntimeConfigV2 (id 15, {} bytes): {}",
                    v2_bytes.len(), e,
                )))?
        } else if let Some(v1_bytes) = sections.get(&(SectionIdV3::RuntimeConfig as u32)) {
            crate::runtime_config::RuntimeConfigSnapshot::decode_v14_section(v1_bytes)
                .map_err(|e| BrainError::Bincode(format!(
                    "v3 section RuntimeConfig (id 14, {} bytes): {}",
                    v1_bytes.len(), e,
                )))?
        } else {
            crate::runtime_config::RuntimeConfigSnapshot::default()
        };

    // Port Source-Persist: optional SourceOverlay (default empty for
    // pre-Source-Persist files via `de_bincode_optional`).
    let source_overlay: crate::source::SourceOverlay =
        de_bincode_optional(&sections, SectionIdV3::SourceOverlay)?;
    // Port Cat.5: optional CapabilityPackLedger snapshot (default empty
    // for pre-Cat.5 files via `de_bincode_optional`).
    let cap_pack_ledger: crate::capability_pack::CapabilityPackLedgerSnapshot =
        de_bincode_optional(&sections, SectionIdV3::CapabilityPackLedger)?;

    // ---- sanity-check on-disk counts vs reconstructed structures ------
    if header.node_count != network.nodes.len() as u64 {
        return Err(BrainError::Format(format!(
            "v3 header node_count={} disagrees with network section ({} nodes)",
            header.node_count,
            network.nodes.len()
        )));
    }
    if header.bond_count != network.bonds.len() as u64 {
        return Err(BrainError::Format(format!(
            "v3 header bond_count={} disagrees with network section ({} bonds)",
            header.bond_count,
            network.bonds.len()
        )));
    }
    let kf: u64 = network.bonds.iter().map(|b| b.keyframes().len() as u64).sum();
    if header.keyframe_count != kf {
        return Err(BrainError::Format(format!(
            "v3 header keyframe_count={} disagrees with bonds ({} keyframes)",
            header.keyframe_count, kf
        )));
    }
    if header.scar_count != scar_log.scars.len() as u64 {
        return Err(BrainError::Format(format!(
            "v3 header scar_count={} disagrees with scar_log ({} scars)",
            header.scar_count,
            scar_log.scars.len()
        )));
    }

    Ok(BrainFile {
        format_version: FORMAT_VERSION_V3,
        network,
        interest_profile,
        meta_reflection,
        scar_log,
        research_proposals,
        research_config: misc.config,
        next_research_proposal_id: misc.next_id,
        gap_seed_frequency,
        wonder_state,
        negation_overlay: crate::negation::NegationOverlay::new(),
        source_overlay,
        pending_log,
        rejection_patterns,
        lb_session,
        cs_style,
        pack_ledger,
        runtime_config,
        runtime_config_present,
        cap_pack_ledger,
    })
}

/// Production load entry point: mmap the file, then delegate to
/// `read_v3_bytes`. Memory-mapped regions are page-aligned, so the
/// archived sections satisfy rkyv's alignment requirement directly.
///
/// The mmap is dropped at function return — Slice 2a eagerly
/// deserializes everything onto the heap, so the file can be unmapped
/// once the `BrainFile` is constructed. The future zero-copy reader
/// (Slice 2c) will hold the mmap alive for the lifetime of the brain.
pub fn read_v3_path<P: AsRef<Path>>(path: P) -> Result<BrainFile, BrainError> {
    let f = std::fs::File::open(path.as_ref())?;
    // SAFETY: Mmap::map is unsafe because mutating the underlying file
    // while the mapping exists is undefined behavior. Production usage
    // mmaps a brain file owned by lu4d and not concurrently mutated;
    // tests use temporary files exclusive to the test.
    let mmap = unsafe { Mmap::map(&f) }
        .map_err(|e| BrainError::Format(format!("mmap failed: {}", e)))?;
    read_v3_bytes(&mmap[..])
}

/// `brain_io::detect_format` only sniffs the v2 vs v1 distinction by
/// the leading magic. v3 reuses v2's magic, so the version field at
/// bytes 8..12 is what tells the two apart. This helper centralises
/// that check so callers can branch without re-reading the file header
/// themselves.
pub fn is_v3_path<P: AsRef<Path>>(path: P) -> Result<bool, BrainError> {
    let mut f = std::fs::File::open(path.as_ref())?;
    let mut buf = [0u8; 12];
    use std::io::Read;
    let n = f.read(&mut buf)?;
    if n < 12 {
        return Ok(false);
    }
    if buf[0..8] != MAGIC_HEADER {
        return Ok(false);
    }
    let version = u32::from_le_bytes(buf[8..12].try_into().unwrap());
    Ok(version == FORMAT_VERSION_V3)
}

/// Port LB.2 — test-only helper: take an LB.2-saved brain (section #8
/// is sentinel-empty) and splice in a legacy section #8 payload, so
/// tests can synthesize "pre-LB.2" and "dual-write era" brain bytes.
///
/// Patches the section #8 OffsetEntry to point at the appended
/// payload and recomputes the trailer CRC.
#[cfg(test)]
pub(crate) fn synthesize_section_8_payload(
    base: &[u8],
    wonder: &crate::curiosity::WonderState,
) -> Vec<u8> {
    let payload = bincode::serialize(wonder).expect("WonderState serializes");
    let trailer_pos = base.len() - TRAILER_SIZE;
    let mut out = Vec::with_capacity(base.len() + payload.len());
    out.extend_from_slice(&base[..trailer_pos]);
    let payload_offset = out.len() as u64;
    out.extend_from_slice(&payload);
    out.extend_from_slice(&base[trailer_pos..]);

    // Patch the OffsetEntry whose section_id == 8.
    let header = read_header(&out).expect("header parse");
    for i in 0..(header.section_count as usize) {
        let entry_start = HEADER_SIZE + i * ENTRY_SIZE;
        let section_id = u32::from_le_bytes(
            out[entry_start..entry_start + 4].try_into().unwrap(),
        );
        if section_id == SectionIdV3::WonderQueueState as u32 {
            out[entry_start + 8..entry_start + 16]
                .copy_from_slice(&payload_offset.to_le_bytes());
            out[entry_start + 16..entry_start + 24]
                .copy_from_slice(&(payload.len() as u64).to_le_bytes());
            break;
        }
    }

    // Recompute trailer CRC over [..new_trailer_pos].
    let new_trailer_pos = out.len() - TRAILER_SIZE;
    let crc = crc32fast::hash(&out[..new_trailer_pos]);
    out[new_trailer_pos + 8..new_trailer_pos + 12]
        .copy_from_slice(&crc.to_le_bytes());
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bond::BondType;
    use crate::brain::{Brain, BrainSaveFormat};
    use crate::node::Confidence;

    /// Tempdir helper mirroring brain_io.rs — same trick so we don't
    /// pull in tempfile as a dev-dep.
    mod tempfile {
        use std::path::PathBuf;
        pub struct TempDir {
            path: PathBuf,
        }
        impl TempDir {
            pub fn path(&self) -> &std::path::Path {
                &self.path
            }
        }
        impl Drop for TempDir {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.path);
            }
        }
        pub fn tempdir() -> std::io::Result<TempDir> {
            use std::sync::atomic::{AtomicU64, Ordering};
            static COUNTER: AtomicU64 = AtomicU64::new(0);
            let n = COUNTER.fetch_add(1, Ordering::SeqCst);
            let mut p = std::env::temp_dir();
            p.push(format!("lu4d-brain-io-v3-test-{}-{}", std::process::id(), n));
            std::fs::create_dir_all(&p)?;
            Ok(TempDir { path: p })
        }
    }

    fn small_brain() -> Brain {
        let mut b = Brain::new();
        b.teach("cell", BondType::IsA, "unit_of_life", 0.9, Confidence::Verified);
        b.teach("cell", BondType::HasProperty, "membrane", 0.8, Confidence::Verified);
        b.teach("mitosis", BondType::IsA, "cell_division", 0.95, Confidence::Verified);
        b.teach("mitosis", BondType::PartOf, "cell_cycle", 0.85, Confidence::Verified);
        b.teach("cell", BondType::Produces, "mitosis", 0.7, Confidence::Verified);
        b
    }

    fn brain_to_file(b: &Brain) -> BrainFile {
        b.to_brain_file(FORMAT_VERSION_V3)
    }

    /// Bytes returned by `write_v3` come from a `Vec<u8>` whose
    /// allocator alignment is implementation-defined, but in practice
    /// every modern allocator gives us at least 16-byte alignment. To
    /// remove that dependency entirely from tests, copy into an
    /// `rkyv::AlignedVec` (which guarantees 16-byte alignment) before
    /// calling `read_v3_bytes`.
    fn aligned_copy(bytes: &[u8]) -> rkyv::AlignedVec {
        let mut v = rkyv::AlignedVec::with_capacity(bytes.len());
        v.extend_from_slice(bytes);
        v
    }

    #[test]
    fn v3_round_trip_preserves_counts() {
        let b = small_brain();
        let file = brain_to_file(&b);
        let bytes = write_v3(&file).expect("write_v3 should succeed");

        // Magic at the start, trailer at the end — same shape as v2.
        assert_eq!(&bytes[0..8], &MAGIC_HEADER);
        let trailer_pos = bytes.len() - TRAILER_SIZE;
        assert_eq!(&bytes[trailer_pos..trailer_pos + 8], &MAGIC_TRAILER);

        // Version byte must be 3.
        let version = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        assert_eq!(version, FORMAT_VERSION_V3);

        let aligned = aligned_copy(&bytes);
        let parsed = read_v3_bytes(&aligned[..]).expect("read_v3_bytes should succeed");
        assert_eq!(parsed.format_version, FORMAT_VERSION_V3);
        assert_eq!(parsed.network.nodes.len(), file.network.nodes.len());
        assert_eq!(parsed.network.bonds.len(), file.network.bonds.len());
        assert_eq!(parsed.scar_log.scars.len(), file.scar_log.scars.len());
        assert_eq!(
            parsed.next_research_proposal_id,
            file.next_research_proposal_id
        );
        assert_eq!(
            parsed.gap_seed_frequency.len(),
            file.gap_seed_frequency.len()
        );
    }

    #[test]
    fn v3_rkyv_sections_are_aligned() {
        // Walk the offset table and verify the network + scar entries
        // start on 16-byte boundaries. This is a writer self-check and
        // a regression guard against future format edits forgetting the
        // padding step.
        let b = small_brain();
        let bytes = write_v3(&brain_to_file(&b)).unwrap();

        for i in 0..(SECTION_COUNT as usize) {
            let entry_start = HEADER_SIZE + i * ENTRY_SIZE;
            let entry = read_entry(&bytes[entry_start..entry_start + ENTRY_SIZE]).unwrap();
            if entry.section_id == SectionIdV3::Network as u32
                || entry.section_id == SectionIdV3::ScarLog as u32
            {
                assert_eq!(
                    entry.offset % RKYV_ALIGN,
                    0,
                    "section {} offset {} not aligned to {}",
                    entry.section_id,
                    entry.offset,
                    RKYV_ALIGN
                );
            }
        }
    }

    #[test]
    fn v3_detects_corruption() {
        let b = small_brain();
        let mut bytes = write_v3(&brain_to_file(&b)).unwrap();
        // Flip a byte in the middle of the data region (after the
        // offset table, well before the trailer).
        let i = HEADER_SIZE + ENTRY_SIZE * (SECTION_COUNT as usize) + 32;
        bytes[i] ^= 0xff;
        let aligned = aligned_copy(&bytes);
        let err = read_v3_bytes(&aligned[..]).expect_err("corrupt file must not parse");
        match err {
            BrainError::Format(msg) => {
                assert!(
                    msg.contains("checksum") || msg.contains("validation") || msg.contains("section"),
                    "unexpected error message: {}",
                    msg
                );
            }
            other => panic!("expected Format error, got {:?}", other),
        }
    }

    #[test]
    fn v3_detects_version_mismatch() {
        let b = small_brain();
        let mut bytes = write_v3(&brain_to_file(&b)).unwrap();
        // Tamper with the version byte to mark this as a future format.
        bytes[8..12].copy_from_slice(&99u32.to_le_bytes());
        // The crc covers bytes[..trailer_pos] including the version, so
        // we have to recompute the crc to make the version mismatch the
        // error we surface.
        let trailer_pos = bytes.len() - TRAILER_SIZE;
        let crc = crc32fast::hash(&bytes[..trailer_pos]);
        bytes[trailer_pos + 8..trailer_pos + 12].copy_from_slice(&crc.to_le_bytes());

        let aligned = aligned_copy(&bytes);
        let err = read_v3_bytes(&aligned[..]).expect_err("future version must not parse");
        match err {
            BrainError::VersionMismatch(99) => {}
            other => panic!("expected VersionMismatch(99), got {:?}", other),
        }
    }

    #[test]
    fn v2_to_v3_migration_preserves_counts_and_content() {
        let dir = tempfile::tempdir().unwrap();
        let v2_path = dir.path().join("src.lbr");
        let v3_path = dir.path().join("dst.lbr");

        // Build a v2 file directly via the existing v2 writer.
        let b = small_brain();
        let v2_file = b.to_brain_file(crate::brain_io::FORMAT_VERSION_V2);
        let v2_bytes = crate::brain_io::write_v2(&v2_file).expect("write_v2");
        std::fs::write(&v2_path, v2_bytes).unwrap();

        // Migrate via Brain::load_or_new + Brain::save_to_format.
        let mut migrated = Brain::load_or_new(&v2_path).expect("load v2");
        migrated
            .save_to_format(&v3_path, BrainSaveFormat::V3Binary)
            .expect("write v3");

        // Source must be untouched.
        let raw_src = std::fs::read(&v2_path).unwrap();
        assert_eq!(&raw_src[0..8], &MAGIC_HEADER);
        let src_version = u32::from_le_bytes(raw_src[8..12].try_into().unwrap());
        assert_eq!(src_version, crate::brain_io::FORMAT_VERSION_V2);

        // Destination must be v3 binary.
        let raw_dst = std::fs::read(&v3_path).unwrap();
        assert_eq!(&raw_dst[0..8], &MAGIC_HEADER);
        let dst_version = u32::from_le_bytes(raw_dst[8..12].try_into().unwrap());
        assert_eq!(dst_version, FORMAT_VERSION_V3);

        // Counts round-trip on reload via the production path
        // (Brain::load_or_new auto-detects v3 by the version byte).
        let reloaded = Brain::load_or_new(&v3_path).expect("load v3");
        assert_eq!(reloaded.network.node_count(), b.network.node_count());
        assert_eq!(reloaded.network.bond_count(), b.network.bond_count());
        let kf_src: usize = b.network.bonds().values().map(|x| x.keyframes().len()).sum();
        let kf_dst: usize = reloaded
            .network
            .bonds()
            .values()
            .map(|x| x.keyframes().len())
            .sum();
        assert_eq!(kf_src, kf_dst);
        assert_eq!(reloaded.scar_log.scars.len(), b.scar_log.scars.len());
    }

    #[test]
    fn v3_save_via_brain_with_format_flag_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("v3.lbr");

        let mut b = small_brain();
        b.path = Some(path.clone());
        b.save_format = BrainSaveFormat::V3Binary;
        b.save().expect("v3 save via Brain::save");

        let raw = std::fs::read(&path).unwrap();
        assert_eq!(&raw[0..8], &MAGIC_HEADER);
        let v = u32::from_le_bytes(raw[8..12].try_into().unwrap());
        assert_eq!(v, FORMAT_VERSION_V3);

        let reloaded = Brain::load_or_new(&path).expect("v3 load");
        assert_eq!(reloaded.network.node_count(), b.network.node_count());
        assert_eq!(reloaded.network.bond_count(), b.network.bond_count());
    }

    #[test]
    fn v3_path_load_via_mmap() {
        // Exercise the production path: write to a real file, load via
        // mmap, verify counts.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mmap_v3.lbr");

        let b = small_brain();
        let bytes = write_v3(&brain_to_file(&b)).unwrap();
        std::fs::write(&path, &bytes).unwrap();

        let parsed = read_v3_path(&path).expect("read_v3_path");
        assert_eq!(parsed.format_version, FORMAT_VERSION_V3);
        assert_eq!(parsed.network.nodes.len(), b.network.node_count());
        assert_eq!(parsed.network.bonds.len(), b.network.bond_count());
    }

    #[test]
    fn v3_is_v3_path_distinguishes_v2_and_v3() {
        let dir = tempfile::tempdir().unwrap();
        let v2_path = dir.path().join("v2.lbr");
        let v3_path = dir.path().join("v3.lbr");

        let b = small_brain();
        std::fs::write(&v2_path, crate::brain_io::write_v2(&b.to_brain_file(2)).unwrap()).unwrap();
        std::fs::write(&v3_path, write_v3(&brain_to_file(&b)).unwrap()).unwrap();

        assert!(!is_v3_path(&v2_path).unwrap());
        assert!(is_v3_path(&v3_path).unwrap());
    }

    // ================================================================
    // Port J.1.1 — v3 brain format reader tolerance for unknown sections.
    //
    // Goal: a J binary (which only knew sections 1..=7, count==7) was
    // unable to load a J.1 brain (count==8, adds WonderQueueState=8).
    // J.1.1 makes the reader skip-on-unknown rather than reject, so a
    // future port adding section 9, 10, ... can still be loaded by an
    // older binary running J.1.1+ logic. The four tests below pin that
    // contract: the BEFORE behavior (Phase 1), the round-trip safety of
    // the new optional section through the tolerant path (Phase 3), and
    // the forward simulation against a hypothetical section 9 (Phase 4).
    // ================================================================

    /// Phase 1 — simulate the pre-J.1 reader: section_count must be exactly 7
    /// and every entry id must fall in 1..=7. This documents the BEFORE
    /// state that Port J.1.1 fixes forward.
    fn pre_j1_simulated_parse(bytes: &[u8]) -> Result<(), BrainError> {
        if bytes.len() < HEADER_SIZE + TRAILER_SIZE {
            return Err(BrainError::Format("too short".into()));
        }
        let header = read_header(bytes)?;
        if header.version != FORMAT_VERSION_V3 {
            return Err(BrainError::VersionMismatch(header.version));
        }
        // Pre-J.1: exact count, no upper-bound tolerance.
        if header.section_count != 7 {
            return Err(BrainError::Format(format!(
                "pre-J.1 v3 reader: unexpected section_count={} (expected 7)",
                header.section_count
            )));
        }
        // Pre-J.1 also rejected any unknown id in the offset table.
        let table_start = HEADER_SIZE;
        for i in 0..(header.section_count as usize) {
            let entry_start = table_start + i * ENTRY_SIZE;
            let entry = read_entry(&bytes[entry_start..entry_start + ENTRY_SIZE])?;
            if entry.section_id < 1 || entry.section_id > 7 {
                return Err(BrainError::Format(format!(
                    "pre-J.1 v3 reader: unknown section id={}",
                    entry.section_id
                )));
            }
        }
        Ok(())
    }

    #[test]
    fn j11_phase1_pre_j1_reader_rejects_j1_brain() {
        // A J.1 brain (count==8, includes WonderQueueState) must be REJECTED
        // by a hypothetical pre-J.1 reader. This is the ops risk the J.1.1
        // port closes — once any J.1 binary saved canonical, the J binary
        // could not roll back.
        let b = small_brain();
        let bytes = write_v3(&brain_to_file(&b)).expect("write_v3");
        let header = read_header(&bytes).unwrap();
        // Port Source-Persist bumped the canonical writer count from 15 to 16.
        assert_eq!(header.section_count, 17, "writer must emit count=17 post-Cat.5");

        let err = pre_j1_simulated_parse(&bytes)
            .expect_err("pre-J.1 reader MUST reject a J.1 brain");
        match err {
            BrainError::Format(msg) => {
                assert!(
                    msg.contains("section_count=8") || msg.contains("expected 7"),
                    "unexpected pre-J.1 error: {msg}"
                );
            }
            other => panic!("expected Format error, got {other:?}"),
        }
    }

    #[test]
    fn j11_phase1_tolerant_reader_accepts_same_j1_brain() {
        // The flip side of Phase 1: the SAME J.1 brain bytes that the
        // pre-J.1 simulator just rejected MUST load successfully through
        // the tolerant reader. (This is implicitly the AFTER half of the
        // before/after pair.)
        let b = small_brain();
        let bytes = write_v3(&brain_to_file(&b)).unwrap();
        let aligned = aligned_copy(&bytes);
        let parsed = read_v3_bytes(&aligned[..])
            .expect("tolerant J.1.1 reader must accept the J.1 brain");
        assert_eq!(parsed.format_version, FORMAT_VERSION_V3);
    }

    #[test]
    fn j11_phase3_round_trip_preserves_wonder_queue_state() {
        // Phase 3 — round-trip safety, REVISED for Port LB.2.
        //
        // Pre-LB.2: section #8 (WonderQueueState) was the canonical home
        // for wonder signals + gap_label_map; this test asserted full
        // signal/label round-trip through write_v3/read_v3_bytes.
        //
        // Post-LB.2: section #8 is sentinel-empty. Wonder state is
        // canonical in section #11 (LB session memory) as `WonderItem`
        // entries. The BrainFile-boundary view of `wonder_state`
        // therefore deserializes to default-empty regardless of what
        // was written, and full wonder round-trip is exercised by the
        // `lb2_*` tests further down (Brain-level save/load via
        // `Brain::load_or_new`).
        //
        // This test is preserved as a regression guard for the
        // sentinel-empty contract: even when `file.wonder_state` is
        // assigned a non-empty value at write time, the parsed file
        // sees `WonderState::default()` after read.
        use crate::curiosity::{CuriositySignal, SignalType, WonderState};
        use crate::scope::TemporalScope;

        let b = small_brain();
        let mut file = brain_to_file(&b);
        let seed_a = Seed::from_label("persistent_gap_a");
        let seed_b = Seed::from_label("persistent_gap_b");
        file.wonder_state = WonderState {
            signals: vec![
                CuriositySignal {
                    seed: seed_a,
                    signal_type: SignalType::KnowledgeDesert,
                    intensity: 0.81,
                    scope: TemporalScope::Current,
                    timestamp_ms: 1_700_000_000_000,
                },
                CuriositySignal {
                    seed: seed_b,
                    signal_type: SignalType::DeadEnd,
                    intensity: 0.42,
                    scope: TemporalScope::Current,
                    timestamp_ms: 1_700_000_000_001,
                },
            ],
            gap_label_map: {
                let mut m = std::collections::HashMap::new();
                m.insert(seed_a, "how do you write a for loop in python".to_string());
                m.insert(seed_b, "what is photosynthesis".to_string());
                m
            },
        };

        // Save #1
        let bytes1 = write_v3(&file).expect("save #1");
        let aligned1 = aligned_copy(&bytes1);
        let parsed1 = read_v3_bytes(&aligned1[..]).expect("load #1");
        // Port LB.2: section #8 is sentinel-empty even when the writer
        // input had a non-empty wonder_state. Both fields default-decode.
        assert!(parsed1.wonder_state.signals.is_empty(),
            "LB.2: writer drops section #8 payload, parsed signals are empty");
        assert!(parsed1.wonder_state.gap_label_map.is_empty(),
            "LB.2: writer drops section #8 payload, parsed gap_label_map is empty");
        let _ = seed_a; // kept for symmetry with the historical pre-LB.2 form
        let _ = seed_b;

        // Reconstruct a BrainFile from the parsed result and save #2
        let file2 = BrainFile {
            format_version: FORMAT_VERSION_V3,
            network: parsed1.network,
            interest_profile: parsed1.interest_profile,
            meta_reflection: parsed1.meta_reflection,
            scar_log: parsed1.scar_log,
            research_proposals: parsed1.research_proposals,
            research_config: parsed1.research_config,
            next_research_proposal_id: parsed1.next_research_proposal_id,
            gap_seed_frequency: parsed1.gap_seed_frequency,
            wonder_state: parsed1.wonder_state,
            negation_overlay: crate::negation::NegationOverlay::new(),
            source_overlay: crate::source::SourceOverlay::new(),
            // Port T.4: pending log snapshot (default empty for pre-T.4 brains).
            pending_log: Default::default(),
            rejection_patterns: Default::default(),
            lb_session: Default::default(),
            cs_style: Default::default(),
            pack_ledger: Default::default(),
            cap_pack_ledger: Default::default(),
            // Port Cat.3c: runtime config defaults round-trip.
            runtime_config: Default::default(),
            runtime_config_present: false,
        };
        let bytes2 = write_v3(&file2).expect("save #2");
        let aligned2 = aligned_copy(&bytes2);
        let parsed2 = read_v3_bytes(&aligned2[..]).expect("load #2");

        // After two full save/load cycles through the tolerant reader the
        // Port LB.2: same sentinel-empty contract on the second pass.
        assert!(parsed2.wonder_state.signals.is_empty(),
            "LB.2: second-pass wonder_state is also sentinel-empty");
        assert!(parsed2.wonder_state.gap_label_map.is_empty(),
            "LB.2: second-pass gap_label_map is also sentinel-empty");
        // Port LB.2: s0/s1 inspection retired. The intensities and
        // labels round-trip via section #11 (LB session memory) instead;
        // see the lb2_* tests for the new contract.
    }

    /// Phase 4 — forward simulation. Construct a brain file with a
    /// HYPOTHETICAL section id that this build does NOT know about, and
    /// confirm the J.1.1 reader skips it cleanly (no panic, no error).
    /// This is the load-bearing forward-compat proof.
    ///
    /// Port T.4 update: the original phase-4 test used id 9 as the
    /// unknown future section. Successive ports have claimed each
    /// id up to 15 (Cat.3c.1's RuntimeConfigV2). Port Cat.3c.1 bumps the
    /// unknown-future id to 99 — far above anything we plan to add.
    /// The baseline writer emits 15 sections post-Cat.3c.1.
    #[test]
    fn j11_phase4_unknown_future_section_id_is_skipped() {
        let b = small_brain();
        let bytes = write_v3(&brain_to_file(&b)).expect("baseline write_v3");

        // Post-Cat.3c.1 the baseline file has section_count=15. We splice
        // in a 16th entry (section id 99) that points at a tiny synthetic
        // payload appended just before the trailer. The reader must walk
        // past it without rejecting.
        const TRAILER: usize = TRAILER_SIZE;
        let trailer_pos = bytes.len() - TRAILER;
        let saved_trailer = bytes[trailer_pos..].to_vec();

        // Synthetic payload: 8 bytes of a hypothetical FUTURE section.
        let future_payload: [u8; 8] = [0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x02, 0x03, 0x04];
        let future_offset = trailer_pos as u64;
        let future_length = future_payload.len() as u64;

        // We need to grow the offset table by one ENTRY_SIZE (24B), but
        // ENTRY_SIZE is not a multiple of RKYV_ALIGN (16), so a naive +24
        // shift would break the network/scar 16-byte alignment invariant.
        // To keep every original section's RKYV alignment we shift by
        // align_up(ENTRY_SIZE, RKYV_ALIGN) = 32 instead and pad the gap
        // between the new (10-entry) offset table and the start of the
        // original data region with zero bytes.
        let table_start = HEADER_SIZE;
        let old_table_end = table_start + 15 * ENTRY_SIZE;
        let old_data = &bytes[old_table_end..trailer_pos];

        let shift: u64 = align_up(ENTRY_SIZE as u64, RKYV_ALIGN);
        let pad_after_table: usize = (shift as usize) - ENTRY_SIZE;

        let mut entries: Vec<OffsetEntry> = (0..15)
            .map(|i| {
                let s = table_start + i * ENTRY_SIZE;
                read_entry(&bytes[s..s + ENTRY_SIZE]).unwrap()
            })
            .collect();
        // Shift every existing entry's offset by `shift` so the rkyv
        // sections stay 16-aligned in the rebuilt file.
        for e in entries.iter_mut() {
            e.offset += shift;
        }
        // Append the future-section entry. Section id 99 is intentionally
        // unknown to this build (SectionIdV3::from_u32(99) == None). It
        // sits at the very end of the data region.
        let new_data_end = (old_table_end as u64)
            + shift
            + (old_data.len() as u64);
        let future_entry = OffsetEntry {
            section_id: 99,
            offset: new_data_end,
            length: future_length,
        };
        entries.push(future_entry);
        let new_future_offset_in_file = new_data_end;
        let _ = future_offset; // (offset variable preserved for clarity)

        // Rebuild file bytes.
        let mut out: Vec<u8> = Vec::new();
        // Header — copy then rewrite section_count.
        out.extend_from_slice(&bytes[..HEADER_SIZE]);
        out[48..52].copy_from_slice(&16u32.to_le_bytes());
        // Offset table — 15 entries plus the synthetic future one.
        for e in &entries {
            write_entry(&mut out, e);
        }
        // Pad to keep the original data region 16-aligned post-shift.
        out.extend_from_slice(&vec![0u8; pad_after_table]);
        // Original data region (verbatim).
        out.extend_from_slice(old_data);
        // Synthetic future-section payload.
        out.extend_from_slice(&future_payload);
        debug_assert_eq!(out.len() as u64, new_future_offset_in_file + future_length);
        // Trailer — magic + recomputed CRC + reserved 4 bytes.
        let crc = crc32fast::hash(&out);
        out.extend_from_slice(&MAGIC_TRAILER);
        out.extend_from_slice(&crc.to_le_bytes());
        out.extend_from_slice(&saved_trailer[saved_trailer.len() - 4..]);

        // The J.1.1 reader MUST load this file successfully and skip
        // section id 10.
        let aligned = aligned_copy(&out);
        let parsed = read_v3_bytes(&aligned[..])
            .expect("J.1.1 reader must skip unknown section 99 and load the rest");
        assert_eq!(parsed.format_version, FORMAT_VERSION_V3);
        assert_eq!(parsed.network.nodes.len(), b.network.node_count());
        assert_eq!(parsed.network.bonds.len(), b.network.bond_count());

        // Confirm bytes really included a count==16 (we constructed it).
        let header = read_header(&out).unwrap();
        assert_eq!(header.section_count, 16);
        // And confirm the unknown-id is genuinely unknown to this build.
        assert!(SectionIdV3::from_u32(99).is_none());

        // Drop unused warnings.
        let _ = bytes;
    }

    /// Port LB.2 — saving a brain that has wonder items in
    /// `wonder_state` MUST emit section #8 as a sentinel-empty entry
    /// (offset 0, length 0). The slot stays in the offset table for
    /// section_count compat with pre-LB.2 binaries; only the payload
    /// goes away.
    #[test]
    fn lb2_save_does_not_write_section_8_payload() {
        let mut b = small_brain();
        // Push two non-trivial wonder signals into the live queue so
        // `to_brain_file` would (pre-LB.2) snapshot a non-empty
        // `WonderState` into section #8.
        let s1 = crate::curiosity::CuriositySignal {
            seed: crate::seed::Seed::from_label("lb2_seed_a"),
            signal_type: crate::curiosity::SignalType::HubBoundary,
            intensity: 0.7,
            scope: crate::scope::TemporalScope::Current,
            timestamp_ms: 1_700_000_000_000,
        };
        let s2 = crate::curiosity::CuriositySignal {
            seed: crate::seed::Seed::from_label("lb2_seed_b"),
            signal_type: crate::curiosity::SignalType::DeadEnd,
            intensity: 0.5,
            scope: crate::scope::TemporalScope::Current,
            timestamp_ms: 1_700_000_000_001,
        };
        b.wonder_queue.push(s1);
        b.wonder_queue.push(s2);
        b.gap_label_map.insert(
            crate::seed::Seed::from_label("lb2_seed_a"),
            "what is lb2_seed_a".to_string(),
        );

        let file = brain_to_file(&b);
        let bytes = write_v3(&file).expect("LB.2 write_v3 must succeed");

        // Walk the offset table looking for section_id == 8.
        let header = read_header(&bytes).expect("header parse");
        assert_eq!(header.section_count, SECTION_COUNT,
            "LB.2 keeps SECTION_COUNT stable so pre-LB.2 readers can still parse");
        let mut found_section_8 = false;
        for i in 0..(header.section_count as usize) {
            let entry_start = HEADER_SIZE + i * ENTRY_SIZE;
            let entry = read_entry(&bytes[entry_start..entry_start + ENTRY_SIZE])
                .expect("entry parse");
            if entry.section_id == SectionIdV3::WonderQueueState as u32 {
                found_section_8 = true;
                assert_eq!(entry.offset, 0,
                    "LB.2 sentinel: section #8 offset must be 0");
                assert_eq!(entry.length, 0,
                    "LB.2 sentinel: section #8 length must be 0 (no payload bytes)");
            }
        }
        assert!(found_section_8,
            "LB.2 keeps the section #8 offset-table slot for backward compat");

        // And the parser must come back with a default (empty) WonderState
        // for the section, since `de_bincode_optional` treats an empty
        // slice the same as an absent section.
        let parsed = read_v3_bytes(&bytes).expect("LB.2 read_v3_bytes must succeed");
        assert!(parsed.wonder_state.signals.is_empty(),
            "LB.2 sentinel section #8 deserializes to default-empty WonderState.signals");
        assert!(parsed.wonder_state.gap_label_map.is_empty(),
            "LB.2 sentinel section #8 deserializes to default-empty WonderState.gap_label_map");
    }

    /// Port LB.2 — a pre-LB.2 brain has section #8 populated and no
    /// WonderItem entries in section #11 (LB). On load the wonder
    /// queue MUST be hydrated from the section #8 payload (legacy
    /// fallback path), and the section #8 wonders must mirror into
    /// LB session memory (the existing LB.1 migration step).
    #[test]
    fn lb2_load_pre_lb2_brain_with_section_8_hydrates_correctly() {
        let b = small_brain();
        // Make sure LB has zero WonderItems — this is a pre-LB.1 era
        // brain shape (section #8 only).
        assert_eq!(
            b.lb_session
                .counts_by_kind()
                .get("wonder_item")
                .copied()
                .unwrap_or(0),
            0
        );
        // Save with LB.2 binary (section #8 will be empty), then
        // splice in a legacy section #8 payload to simulate a brain
        // that was last saved by a pre-LB.2 binary.
        let bytes = write_v3(&brain_to_file(&b)).expect("baseline write_v3");
        let legacy_wonder = crate::curiosity::WonderState {
            signals: vec![
                crate::curiosity::CuriositySignal {
                    seed: crate::seed::Seed::from_label("pre_lb2_q1"),
                    signal_type: crate::curiosity::SignalType::HubBoundary,
                    intensity: 0.6,
                    scope: crate::scope::TemporalScope::Current,
                    timestamp_ms: 1_700_000_000_000,
                },
                crate::curiosity::CuriositySignal {
                    seed: crate::seed::Seed::from_label("pre_lb2_q2"),
                    signal_type: crate::curiosity::SignalType::IsolatedNode,
                    intensity: 0.4,
                    scope: crate::scope::TemporalScope::Current,
                    timestamp_ms: 1_700_000_000_001,
                },
            ],
            gap_label_map: {
                let mut m = std::collections::HashMap::new();
                m.insert(
                    crate::seed::Seed::from_label("pre_lb2_q1"),
                    "pre_lb2 question one".to_string(),
                );
                m
            },
        };
        let synthesized = synthesize_section_8_payload(&bytes, &legacy_wonder);

        // Round-trip the synthesized bytes through read_v3_bytes to
        // confirm the wonder_state we injected is what comes back at
        // the BrainFile boundary.
        let parsed = read_v3_bytes(&synthesized).expect("synthesized parse");
        assert_eq!(parsed.wonder_state.signals.len(), 2,
            "section #8 payload survives the synthesis round-trip");
        // And the LB snapshot from this brain has zero WonderItem
        // entries (this is the pre-LB.2 shape we synthesized).
        let wonder_count_in_lb = parsed.lb_session.entries.iter()
            .filter(|e| matches!(e.kind, crate::lb_session::SessionEntryKind::WonderItem))
            .count();
        assert_eq!(wonder_count_in_lb, 0,
            "pre-LB.2 brain shape: section #11 has no WonderItem entries");
    }

    /// Port LB.2 — a dual-write era (LB.1) brain has BOTH section #8
    /// AND section #11 populated. After load, the wonder queue must
    /// reflect section #11 (LB WonderItems), not section #8 — section
    /// #11 is the canonical source under LB.2.
    #[test]
    fn lb2_load_lb_era_dual_write_brain_prefers_section_11() {
        let mut b = small_brain();
        // Push WonderItems into LB ONLY (no wonder_queue push), so
        // section #11 has these signals at save time.
        let now = 1_700_000_500_000_u64;
        let lb_signals: Vec<crate::curiosity::CuriositySignal> = (0..3)
            .map(|i| crate::curiosity::CuriositySignal {
                seed: crate::seed::Seed::from_label(&format!("lb_only_{}", i)),
                signal_type: crate::curiosity::SignalType::HubBoundary,
                intensity: 0.3 + 0.1 * (i as f32),
                scope: crate::scope::TemporalScope::Current,
                timestamp_ms: now,
            })
            .collect();
        for s in &lb_signals {
            let _ = b.lb_session.push_internal(
                crate::lb_session::SessionEntry::new(
                    0,
                    now,
                    crate::lb_session::SessionEntryKind::WonderItem,
                    format!("label_for_{}", s.seed),
                    serde_json::to_string(s).unwrap_or_default(),
                    s.intensity,
                ),
            );
        }
        // Confirm LB has 3 WonderItems.
        assert_eq!(
            b.lb_session
                .counts_by_kind()
                .get("wonder_item")
                .copied()
                .unwrap_or(0),
            3
        );

        let bytes = write_v3(&brain_to_file(&b)).expect("write_v3");
        // Now splice section #8 with DIFFERENT signals (the legacy
        // section #8 view from before the dual-write era cleanup).
        let section_8_signals = crate::curiosity::WonderState {
            signals: vec![crate::curiosity::CuriositySignal {
                seed: crate::seed::Seed::from_label("section8_phantom"),
                signal_type: crate::curiosity::SignalType::DeadEnd,
                intensity: 0.99,
                scope: crate::scope::TemporalScope::Current,
                timestamp_ms: now,
            }],
            gap_label_map: std::collections::HashMap::new(),
        };
        let synthesized = synthesize_section_8_payload(&bytes, &section_8_signals);
        let parsed = read_v3_bytes(&synthesized).expect("parse");

        // Sanity at BrainFile boundary: section #8 has the phantom,
        // section #11 has the 3 LB WonderItems.
        assert_eq!(parsed.wonder_state.signals.len(), 1,
            "section #8 payload (1 phantom) survives synthesis");
        let parsed_lb_wonder_count = parsed.lb_session.entries.iter()
            .filter(|e| matches!(e.kind, crate::lb_session::SessionEntryKind::WonderItem))
            .count();
        assert_eq!(parsed_lb_wonder_count, 3,
            "section #11 has 3 WonderItem entries");

        // Now exercise the full Brain hydration path. After load, the
        // wonder_queue should reflect the 3 LB signals (NOT the
        // section #8 phantom). Round-trip through Brain::load_or_new.
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("lb2_dual_write.lbr");
        std::fs::write(&p, &synthesized).unwrap();
        let brain = Brain::load_or_new(&p).expect("Brain::load_or_new");
        assert_eq!(
            brain.wonder_queue.len(),
            3,
            "LB.2: dual-write era brain hydrates wonder_queue from section #11 (3), NOT section #8 (1)"
        );
        // None of the in-queue seeds match the section #8 phantom.
        let in_queue: Vec<_> = brain.wonder_queue.top_n(10);
        let phantom_seed = crate::seed::Seed::from_label("section8_phantom");
        assert!(!in_queue.iter().any(|s| s.seed == phantom_seed),
            "section #8 phantom seed must NOT appear in wonder_queue");
    }

    /// Port LB.2 — an LB.2-saved brain has a sentinel-empty section
    /// #8 and a populated section #11. On load the wonder queue MUST
    /// be hydrated from section #11 (LB WonderItem entries).
    #[test]
    fn lb2_load_lb2_saved_brain_reads_from_section_11() {
        let mut b = small_brain();
        let now = 1_700_000_900_000_u64;
        let signals: Vec<_> = (0..2)
            .map(|i| crate::curiosity::CuriositySignal {
                seed: crate::seed::Seed::from_label(&format!("lb2_sig_{}", i)),
                signal_type: crate::curiosity::SignalType::HubBoundary,
                intensity: 0.5 + 0.1 * (i as f32),
                scope: crate::scope::TemporalScope::Current,
                timestamp_ms: now,
            })
            .collect();
        // Dual-write live: push to BOTH wonder_queue and LB (mirror
        // the LB.1 production dual-write).
        for s in &signals {
            let _ = b.lb_session.push_internal(
                crate::lb_session::SessionEntry::new(
                    0,
                    now,
                    crate::lb_session::SessionEntryKind::WonderItem,
                    format!("why_{}", s.seed),
                    serde_json::to_string(s).unwrap_or_default(),
                    s.intensity,
                ),
            );
            b.wonder_queue.push(s.clone());
        }

        let bytes = write_v3(&brain_to_file(&b)).expect("write_v3");

        // Section #8 in the LB.2-saved bytes is sentinel-empty.
        let parsed = read_v3_bytes(&bytes).expect("parse");
        assert!(parsed.wonder_state.signals.is_empty(),
            "LB.2-saved brain: section #8 is empty");
        // Round-trip through Brain::load_or_new and verify the wonder
        // queue is populated via section #11.
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("lb2_saved.lbr");
        std::fs::write(&p, &bytes).unwrap();
        let brain = Brain::load_or_new(&p).expect("Brain::load_or_new");
        assert_eq!(
            brain.wonder_queue.len(),
            2,
            "LB.2: section #11 hydrates the wonder_queue"
        );
    }

    /// Port LB.2 — wonder_queue state must be equivalent across all
    /// three brain eras: pre-LB.2 (section #8 only), dual-write era
    /// (both sections), and LB.2-saved (section #11 only). Uses the
    /// same set of source signals for each era's bytes.
    #[test]
    fn lb2_wonder_state_consistent_across_three_brain_eras() {
        let now = 1_700_001_000_000_u64;
        let canonical_signals: Vec<crate::curiosity::CuriositySignal> = (0..3)
            .map(|i| crate::curiosity::CuriositySignal {
                seed: crate::seed::Seed::from_label(&format!("era_sig_{}", i)),
                signal_type: crate::curiosity::SignalType::HubBoundary,
                intensity: 0.4 + 0.1 * (i as f32),
                scope: crate::scope::TemporalScope::Current,
                timestamp_ms: now,
            })
            .collect();

        // Era 3 (LB.2): dual-write live, LB.2 binary writes section
        // #8 sentinel-empty, section #11 populated.
        let mut b3 = small_brain();
        for s in &canonical_signals {
            let _ = b3.lb_session.push_internal(
                crate::lb_session::SessionEntry::new(
                    0,
                    now,
                    crate::lb_session::SessionEntryKind::WonderItem,
                    format!("label_{}", s.seed),
                    serde_json::to_string(s).unwrap_or_default(),
                    s.intensity,
                ),
            );
            b3.wonder_queue.push(s.clone());
        }
        let era3_bytes = write_v3(&brain_to_file(&b3)).expect("era3 write");

        // Era 2 (LB.1 dual-write era): take era3 bytes and splice in
        // a matching section #8 payload (same signals as section #11).
        let era2_bytes = synthesize_section_8_payload(
            &era3_bytes,
            &crate::curiosity::WonderState {
                signals: canonical_signals.clone(),
                gap_label_map: std::collections::HashMap::new(),
            },
        );

        // Era 1 (pre-LB.1): start from a brain with NO LB WonderItems,
        // splice in section #8 payload.
        let b1 = small_brain();
        let era1_base = write_v3(&brain_to_file(&b1)).expect("era1 base");
        let era1_bytes = synthesize_section_8_payload(
            &era1_base,
            &crate::curiosity::WonderState {
                signals: canonical_signals.clone(),
                gap_label_map: std::collections::HashMap::new(),
            },
        );

        let dir = tempfile::tempdir().unwrap();
        let p1 = dir.path().join("era1.lbr");
        let p2 = dir.path().join("era2.lbr");
        let p3 = dir.path().join("era3.lbr");
        std::fs::write(&p1, &era1_bytes).unwrap();
        std::fs::write(&p2, &era2_bytes).unwrap();
        std::fs::write(&p3, &era3_bytes).unwrap();

        let brain1 = Brain::load_or_new(&p1).expect("era1 load");
        let brain2 = Brain::load_or_new(&p2).expect("era2 load");
        let brain3 = Brain::load_or_new(&p3).expect("era3 load");

        let n1 = brain1.wonder_queue.len();
        let n2 = brain2.wonder_queue.len();
        let n3 = brain3.wonder_queue.len();
        assert_eq!(n1, n2, "era1 vs era2 wonder_queue length");
        assert_eq!(n2, n3, "era2 vs era3 wonder_queue length");
        assert_eq!(n3, 3, "all three eras hydrate to 3 wonder_queue entries");

        // Compare seed sets across the three eras.
        let seeds = |b: &Brain| -> std::collections::HashSet<crate::seed::Seed> {
            b.wonder_queue.top_n(100).into_iter().map(|s| s.seed).collect()
        };
        let s1 = seeds(&brain1);
        let s2 = seeds(&brain2);
        let s3 = seeds(&brain3);
        assert_eq!(s1, s2, "era1 and era2 must hydrate to the same wonder seeds");
        assert_eq!(s2, s3, "era2 and era3 must hydrate to the same wonder seeds");
    }

}


/// Read just the `flags` field out of a v3 file without parsing the
/// rest of the layout. Used by `NetworkArchive::from_mmap` to pick the
/// uncompressed vs per-block-compressed code path before constructing
/// the index maps. Returns the file's `flags` field on success.
pub fn read_v3_flags(bytes: &[u8]) -> Result<u32, BrainError> {
    let header = read_header(bytes)?;
    if header.version != FORMAT_VERSION_V3 {
        return Err(BrainError::VersionMismatch(header.version));
    }
    Ok(header.flags)
}

/// Locate the network section inside a v3 byte slice and return its
/// (offset, length) WITHOUT decompressing or rkyv-validating anything.
/// Used by `NetworkArchive::from_mmap` in compressed mode so the
/// `CompressedNetwork` reader can hold its own slice into the mmap.
pub fn locate_network_section_no_crc(bytes: &[u8]) -> Result<(usize, usize), BrainError> {
    let (_header, sections) = parse_layout_no_crc(bytes)?;
    let blob = require(&sections, SectionIdV3::Network)?;
    let base = bytes.as_ptr() as usize;
    let blob_base = blob.as_ptr() as usize;
    let offset = blob_base
        .checked_sub(base)
        .ok_or_else(|| BrainError::Format("network section pointer before base".into()))?;
    Ok((offset, blob.len()))
}


/// Storage v2 Phase 3 — write a v3 brain file with the network section
/// in the per-block-compressed layout (see `crate::compress`). The
/// `flags` field of the file header is set to
/// `crate::compress::FORMAT_FLAG_COMPRESSED` so readers know to dispatch
/// through the compressed reader for section 1.
///
/// The ScarLog section and bincoded side tables are written unchanged —
/// they are tiny on the production brain (~3 KB combined) so
/// compressing them is not worth the read-side complexity.
pub fn write_v3_compressed(
    file: &BrainFile,
    block_size: usize,
    zstd_level: i32,
) -> Result<Vec<u8>, BrainError> {
    // Compressed network payload (section 1).
    let net_compressed = crate::compress::write_compressed_network_payload(
        &file.network,
        block_size,
        zstd_level,
        None,
    )?;
    // Other sections — same encoding as the uncompressed writer.
    let scar_archived = rkyv::to_bytes::<_, 1024>(&file.scar_log)
        .map_err(|e| BrainError::Format(format!("rkyv serialize ScarLog: {}", e)))?;
    let interest_bytes =
        bincode::serialize(&file.interest_profile).map_err(BrainError::bincode)?;
    let meta_bytes =
        bincode::serialize(&file.meta_reflection).map_err(BrainError::bincode)?;
    let prop_bytes =
        bincode::serialize(&file.research_proposals).map_err(BrainError::bincode)?;
    let misc_blob = ResearchMiscBlob {
        config: file.research_config.clone(),
        next_id: file.next_research_proposal_id,
    };
    let misc_bytes = bincode::serialize(&misc_blob).map_err(BrainError::bincode)?;
    let gap_bytes =
        bincode::serialize(&file.gap_seed_frequency).map_err(BrainError::bincode)?;
    // Port LB.2: section #8 (WonderQueueState) is now a sentinel-empty
    // slot. Wonder state is canonical in section #11 (LB session memory)
    // as `SessionEntryKind::WonderItem` entries; section #8's payload
    // bytes are no longer emitted (offset=0, length=0). Reading
    // section #8 is preserved as a one-shot legacy fallback for
    // pre-LB.2 brains (see `de_bincode_optional` + the loader).
    let wonder_bytes: Vec<u8> = Vec::new();
    // Port T.4: serialize PendingLog snapshot as the optional
    // PendingLog section (id 9).
    let pending_bytes =
        bincode::serialize(&file.pending_log).map_err(BrainError::bincode)?;
    // Port T.8: serialize RejectionPatternsSnapshot as the optional
    // RejectionPatterns section (id 10).
    let rejection_bytes =
        bincode::serialize(&file.rejection_patterns).map_err(BrainError::bincode)?;
    // Port LB: serialize Left Brain SessionMemorySnapshot as the optional
    // SessionMemory section (id 11).
    let lb_bytes =
        bincode::serialize(&file.lb_session).map_err(BrainError::bincode)?;
    // Port CS: serialize ConversationalStyleSnapshot as the optional
    // ConversationalStyle section (id 12).
    let cs_bytes =
        bincode::serialize(&file.cs_style).map_err(BrainError::bincode)?;
    // Port Cat.4: serialize PackLedgerSnapshot as the optional
    // PackLedger section (id 13).
    let pack_bytes =
        bincode::serialize(&file.pack_ledger).map_err(BrainError::bincode)?;
    // Port Cat.3c: serialize the FROZEN Cat.3c shape into bincode
    // section #14 (RuntimeConfig). Cat.3c.1 keeps writing this so
    // pre-Cat.3c.1 binaries can still read their historical fields
    // unchanged. New runtime-config fields go into section #15 below.
    let runtime_config_bytes = file
        .runtime_config
        .encode_v14_section()
        .map_err(|e| BrainError::Bincode(e))?;
    // Port Cat.3c.1: serialize the FULL runtime-config snapshot as
    // JSON into section #15 (RuntimeConfigV2). JSON's named fields
    // give forward+backward compat at the field level so future
    // ports can extend RuntimeConfigSnapshot freely without breaking
    // older binaries (and vice versa).
    let runtime_config_v2_bytes = file
        .runtime_config
        .encode_v15_section()
        .map_err(|e| BrainError::Bincode(e))?;
    // Port Source-Persist: SourceOverlay section (id 16). Bincode
    // serialization mirrors the InterestProfile / PendingLog pattern.
    let source_overlay_bytes =
        bincode::serialize(&file.source_overlay).map_err(BrainError::bincode)?;
    // Port Cat.5: CapabilityPackLedger section (id 17). Schema-only
    // signed capability pack ledger. Bincode-encoded; defaults to
    // empty for pre-Cat.5 brains via `de_bincode_optional`.
    let cap_pack_ledger_bytes =
        bincode::serialize(&file.cap_pack_ledger).map_err(BrainError::bincode)?;

    let keyframe_count: u64 = file
        .network
        .bonds
        .iter()
        .map(|b| b.keyframes().len() as u64)
        .sum();

    // Layout, in file-byte order:
    //   header
    //   offset table
    //   network (compressed, byte-aligned)
    //   pad to align 16
    //   scar rkyv (id 4, 16-aligned)
    //   bincode side tables (ids 2, 3, 5, 6, 7)
    //   trailer
    let header_end = HEADER_SIZE as u64;
    let table_bytes = (SECTION_COUNT as u64) * (ENTRY_SIZE as u64);
    let table_end = header_end + table_bytes;

    let net_offset = table_end; // compressed payload is byte-aligned
    let net_len = net_compressed.len() as u64;

    let scar_offset = align_up(net_offset + net_len, RKYV_ALIGN);
    let scar_len = scar_archived.len() as u64;

    let interest_offset = scar_offset + scar_len;
    let interest_len = interest_bytes.len() as u64;
    let meta_offset = interest_offset + interest_len;
    let meta_len = meta_bytes.len() as u64;
    let prop_offset = meta_offset + meta_len;
    let prop_len = prop_bytes.len() as u64;
    let misc_offset = prop_offset + prop_len;
    let misc_len = misc_bytes.len() as u64;
    let gap_offset = misc_offset + misc_len;
    let gap_len = gap_bytes.len() as u64;

    // Port LB.2: sentinel section #8 — offset 0, length 0.
    // The slot stays in the offset table for backward-compat with
    // pre-LB.2 binaries (which iterate header.section_count entries).
    let wonder_offset: u64 = 0;
    let wonder_len: u64 = 0;

    // Port T.4: PendingLog section (id 9). Port LB.2: chains directly
    // off gap_offset+gap_len since the section #8 slot is sentinel-empty.
    let pending_offset = gap_offset + gap_len;
    let pending_len = pending_bytes.len() as u64;

    // Port T.8: RejectionPatterns section (id 10) follows pending.
    let rejection_offset = pending_offset + pending_len;
    let rejection_len = rejection_bytes.len() as u64;

    // Port LB: SessionMemory section (id 11) follows rejection.
    let lb_offset = rejection_offset + rejection_len;
    let lb_len = lb_bytes.len() as u64;

    // Port CS: ConversationalStyle section (id 12) follows lb_session.
    let cs_offset = lb_offset + lb_len;
    let cs_len = cs_bytes.len() as u64;

    // Port Cat.4: PackLedger section (id 13) follows ConversationalStyle.
    let pack_offset = cs_offset + cs_len;
    let pack_len = pack_bytes.len() as u64;

    // Port Cat.3c: RuntimeConfig section (id 14) follows PackLedger.
    let runtime_config_offset = pack_offset + pack_len;
    let runtime_config_len = runtime_config_bytes.len() as u64;

    // Port Cat.3c.1: RuntimeConfigV2 section (id 15) follows RuntimeConfig.
    let runtime_config_v2_offset = runtime_config_offset + runtime_config_len;
    let runtime_config_v2_len = runtime_config_v2_bytes.len() as u64;

    // Port Source-Persist: SourceOverlay section (id 16) follows RuntimeConfigV2.
    let source_overlay_offset = runtime_config_v2_offset + runtime_config_v2_len;
    let source_overlay_len = source_overlay_bytes.len() as u64;

    // Port Cat.5: CapabilityPackLedger section (id 17) follows SourceOverlay.
    let cap_pack_ledger_offset = source_overlay_offset + source_overlay_len;
    let cap_pack_ledger_len = cap_pack_ledger_bytes.len() as u64;

    let data_end = cap_pack_ledger_offset + cap_pack_ledger_len;
    let total_size = data_end as usize + TRAILER_SIZE;

    let mut out: Vec<u8> = Vec::with_capacity(total_size);

    // Header — identical to the uncompressed writer except the
    // `flags` field carries `FORMAT_FLAG_COMPRESSED` (bit 0).
    {
        let mut buf = [0u8; HEADER_SIZE];
        buf[0..8].copy_from_slice(&MAGIC_HEADER);
        buf[8..12].copy_from_slice(&FORMAT_VERSION_V3.to_le_bytes());
        buf[12..16].copy_from_slice(
            &crate::compress::FORMAT_FLAG_COMPRESSED.to_le_bytes(),
        );
        buf[16..24].copy_from_slice(&(file.network.nodes.len() as u64).to_le_bytes());
        buf[24..32].copy_from_slice(&(file.network.bonds.len() as u64).to_le_bytes());
        buf[32..40].copy_from_slice(&keyframe_count.to_le_bytes());
        buf[40..48].copy_from_slice(&(file.scar_log.scars.len() as u64).to_le_bytes());
        buf[48..52].copy_from_slice(&SECTION_COUNT.to_le_bytes());
        out.extend_from_slice(&buf);
    }

    let entries: [OffsetEntry; SECTION_COUNT as usize] = [
        OffsetEntry { section_id: SectionIdV3::Network as u32,           offset: net_offset,      length: net_len },
        OffsetEntry { section_id: SectionIdV3::InterestProfile as u32,   offset: interest_offset, length: interest_len },
        OffsetEntry { section_id: SectionIdV3::MetaReflection as u32,    offset: meta_offset,     length: meta_len },
        OffsetEntry { section_id: SectionIdV3::ScarLog as u32,           offset: scar_offset,     length: scar_len },
        OffsetEntry { section_id: SectionIdV3::ResearchProposals as u32, offset: prop_offset,     length: prop_len },
        OffsetEntry { section_id: SectionIdV3::ResearchMisc as u32,      offset: misc_offset,     length: misc_len },
        OffsetEntry { section_id: SectionIdV3::GapSeedFrequency as u32,  offset: gap_offset,      length: gap_len },
        OffsetEntry { section_id: SectionIdV3::WonderQueueState as u32,  offset: wonder_offset,   length: wonder_len },
        // Port T.4: PendingLog section (id 9).
        OffsetEntry { section_id: SectionIdV3::PendingLog as u32,        offset: pending_offset,  length: pending_len },
        // Port T.8: RejectionPatterns section (id 10).
        OffsetEntry { section_id: SectionIdV3::RejectionPatterns as u32, offset: rejection_offset, length: rejection_len },
        // Port LB: SessionMemory section (id 11).
        OffsetEntry { section_id: SectionIdV3::SessionMemory as u32,     offset: lb_offset,        length: lb_len },
        // Port CS: ConversationalStyle section (id 12).
        OffsetEntry { section_id: SectionIdV3::ConversationalStyle as u32, offset: cs_offset, length: cs_len },
        // Port Cat.4: PackLedger section (id 13).
        OffsetEntry { section_id: SectionIdV3::PackLedger as u32,          offset: pack_offset, length: pack_len },
        // Port Cat.3c: RuntimeConfig section (id 14).
        OffsetEntry { section_id: SectionIdV3::RuntimeConfig as u32,       offset: runtime_config_offset, length: runtime_config_len },
        // Port Cat.3c.1: RuntimeConfigV2 section (id 15).
        OffsetEntry { section_id: SectionIdV3::RuntimeConfigV2 as u32,     offset: runtime_config_v2_offset, length: runtime_config_v2_len },
        // Port Source-Persist: SourceOverlay section (id 16).
        OffsetEntry { section_id: SectionIdV3::SourceOverlay as u32,       offset: source_overlay_offset,    length: source_overlay_len },
        // Port Cat.5: CapabilityPackLedger section (id 17).
        OffsetEntry { section_id: SectionIdV3::CapabilityPackLedger as u32, offset: cap_pack_ledger_offset, length: cap_pack_ledger_len },
    ];
    for e in &entries {
        write_entry(&mut out, e);
    }

    // Section payloads — write at the recorded offsets, padding with
    // zeros so each section's pre-recorded `offset` is exact.
    debug_assert_eq!(out.len() as u64, net_offset);
    out.extend_from_slice(&net_compressed);

    while (out.len() as u64) < scar_offset {
        out.push(0);
    }
    debug_assert_eq!(out.len() as u64 % RKYV_ALIGN, 0);
    out.extend_from_slice(&scar_archived);

    debug_assert_eq!(out.len() as u64, interest_offset);
    out.extend_from_slice(&interest_bytes);
    debug_assert_eq!(out.len() as u64, meta_offset);
    out.extend_from_slice(&meta_bytes);
    debug_assert_eq!(out.len() as u64, prop_offset);
    out.extend_from_slice(&prop_bytes);
    debug_assert_eq!(out.len() as u64, misc_offset);
    out.extend_from_slice(&misc_bytes);
    debug_assert_eq!(out.len() as u64, gap_offset);
    out.extend_from_slice(&gap_bytes);
    // Port LB.2: section #8 is sentinel-empty — no payload to append.
    let _ = wonder_bytes;
    let _ = wonder_offset;
    // Port T.4: PendingLog section.
    debug_assert_eq!(out.len() as u64, pending_offset);
    out.extend_from_slice(&pending_bytes);
    // Port T.8: RejectionPatterns section.
    debug_assert_eq!(out.len() as u64, rejection_offset);
    out.extend_from_slice(&rejection_bytes);
    // Port LB: SessionMemory section.
    debug_assert_eq!(out.len() as u64, lb_offset);
    out.extend_from_slice(&lb_bytes);
    // Port CS: ConversationalStyle section.
    debug_assert_eq!(out.len() as u64, cs_offset);
    out.extend_from_slice(&cs_bytes);
    // Port Cat.4: PackLedger section.
    debug_assert_eq!(out.len() as u64, pack_offset);
    out.extend_from_slice(&pack_bytes);
    // Port Cat.3c: RuntimeConfig section.
    debug_assert_eq!(out.len() as u64, runtime_config_offset);
    out.extend_from_slice(&runtime_config_bytes);
    // Port Cat.3c.1: RuntimeConfigV2 section.
    debug_assert_eq!(out.len() as u64, runtime_config_v2_offset);
    out.extend_from_slice(&runtime_config_v2_bytes);
    // Port Source-Persist: SourceOverlay section.
    debug_assert_eq!(out.len() as u64, source_overlay_offset);
    out.extend_from_slice(&source_overlay_bytes);
    // Port Cat.5: CapabilityPackLedger section.
    debug_assert_eq!(out.len() as u64, cap_pack_ledger_offset);
    out.extend_from_slice(&cap_pack_ledger_bytes);
    debug_assert_eq!(out.len() as u64, data_end);

    let crc = crc32fast::hash(&out);
    out.extend_from_slice(&MAGIC_TRAILER);
    out.extend_from_slice(&crc.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    debug_assert_eq!(out.len(), total_size);
    Ok(out)
}
/// Storage v2 Phase 4 — write a v3 brain file with the network section
/// in the per-block-compressed-AND-encrypted layout. Each compressed
/// block payload is XOR-encrypted under `(master_key, block_index)`.
/// Header, offset table, block index, and block-frame headers stay
/// cleartext so the loader can find sections without the key.
///
/// The `flags` field carries `FORMAT_FLAG_COMPRESSED | FORMAT_FLAG_
/// ENCRYPTED`; the unencrypted-compressed reader will reject the file
/// (it does not have a key), and the ScarLog/side-table sections stay
/// in the clear (they are tiny and live next to the encrypted network).
pub fn write_v3_compressed_encrypted(
    file: &BrainFile,
    block_size: usize,
    zstd_level: i32,
    encryption_key: &crate::encrypt::MasterKey,
) -> Result<Vec<u8>, BrainError> {
    // OP-017 — thin Cursor-wrapper around the streaming writer.
    // The production save path (Brain::save_to_format) bypasses this
    // wrapper and calls write_v3_compressed_encrypted_to directly
    // against the tmp file to avoid materialising the full output on
    // the heap. This wrapper exists for backward-compat callers
    // (tests, migration tool) that still want a `Vec<u8>`.
    use std::io::Cursor;
    let mut cursor = Cursor::new(Vec::new());
    write_v3_compressed_encrypted_to(
        &file.network,
        file,
        block_size,
        zstd_level,
        encryption_key,
        &mut cursor,
    )?;
    Ok(cursor.into_inner())
}

/// Write an `ENTRY_SIZE`-byte offset-table entry to `w`. Streaming-writer
/// analogue of `write_entry` (which appends into a `Vec<u8>`).
fn write_entry_to<W: std::io::Write>(
    w: &mut W,
    e: &OffsetEntry,
) -> std::io::Result<()> {
    let mut buf = [0u8; ENTRY_SIZE];
    buf[0..4].copy_from_slice(&e.section_id.to_le_bytes());
    // bytes 4..8 reserved (zero)
    buf[8..16].copy_from_slice(&e.offset.to_le_bytes());
    buf[16..24].copy_from_slice(&e.length.to_le_bytes());
    w.write_all(&buf)
}

/// OP-017 streaming writer for v3-compressed-encrypted brains.
///
/// Layout is byte-identical to `write_v3_compressed_encrypted`. The
/// difference is purely in HOW the bytes get to disk:
///
///   1. Reserve a zero-filled placeholder at file offsets `[0, HEADER_SIZE + section_count*ENTRY_SIZE)` for the outer header + offset table.
///   2. Stream the compressed-encrypted network section directly via
///      `compress::write_compressed_network_payload_to` (which uses its
///      own seek-back trick for the section-internal header / indices /
///      block_offset_table).
///   3. Pad to RKYV_ALIGN for the scar_log section.
///   4. Serialize-then-write-then-drop each scar/bincode side table
///      one at a time; never hold more than one section's bytes alive
///      simultaneously.
///   5. Walk the network's bonds once more (via the NetworkSource
///      iterator — single allocation per archived bond, dropped
///      between iterations) to sum `keyframe_count` for the outer
///      header. Cheaper than another full rkyv pass.
///   6. Seek to file offset 0, write the real header + offset table.
///   7. Stream-read the file back from offset 0 to data_end through a
///      256 KiB buffer to compute the file-wide CRC32. The OS page
///      cache holds the bytes we just wrote, so this is essentially a
///      memory-speed scan (~3-5 s for a 2.3 GiB brain).
///   8. Seek to data_end, write the trailer (MAGIC + crc + reserved).
///
/// `file.network` is IGNORED — the caller supplies the network via
/// the explicit `net: &N` parameter. This lets the production save
/// path pass `&self.network` (an `&AssociativeNetwork`) so the
/// `to_snapshot` → `Vec<KnowledgeNode>` heap materialisation is
/// never performed.
pub fn write_v3_compressed_encrypted_to<N, RW>(
    net: &N,
    file: &BrainFile,
    block_size: usize,
    zstd_level: i32,
    encryption_key: &crate::encrypt::MasterKey,
    rw: &mut RW,
) -> Result<u64, BrainError>
where
    N: crate::network::NetworkSource + ?Sized,
    RW: std::io::Read + std::io::Write + std::io::Seek + Send,
{
    use std::io::SeekFrom;

    // ---- (1) Seek to 0, reserve header + offset table placeholder. ----
    rw.seek(SeekFrom::Start(0)).map_err(BrainError::Io)?;

    let header_end = HEADER_SIZE as u64;
    let table_bytes = (SECTION_COUNT as u64) * (ENTRY_SIZE as u64);
    let table_end = header_end + table_bytes;
    let placeholder_zeros = vec![0u8; table_end as usize];
    rw.write_all(&placeholder_zeros).map_err(BrainError::Io)?;
    drop(placeholder_zeros);
    debug_assert_eq!(
        rw.stream_position().map_err(BrainError::Io)?,
        table_end
    );

    // ---- (2) Stream the compressed-encrypted network section. ----
    let net_offset = table_end; // compressed payload is byte-aligned (matches legacy)
    let net_len = crate::compress::write_compressed_network_payload_to(
        net,
        block_size,
        zstd_level,
        Some(encryption_key),
        rw,
    )?;
    debug_assert_eq!(
        rw.stream_position().map_err(BrainError::Io)?,
        net_offset + net_len
    );

    // ---- (3) Pad to RKYV_ALIGN before the scar section. ----
    let scar_offset = align_up(net_offset + net_len, RKYV_ALIGN);
    let pad = scar_offset - (net_offset + net_len);
    if pad > 0 {
        let pad_zeros = vec![0u8; pad as usize];
        rw.write_all(&pad_zeros).map_err(BrainError::Io)?;
    }

    // ---- (4) scar_log: rkyv-archived, small (~few MB at most). ----
    let scar_len: u64;
    {
        let scar_archived = rkyv::to_bytes::<_, 1024>(&file.scar_log).map_err(|e| {
            BrainError::Format(format!("rkyv serialize ScarLog: {}", e))
        })?;
        scar_len = scar_archived.len() as u64;
        rw.write_all(&scar_archived).map_err(BrainError::Io)?;
        // scar_archived dropped at end of scope.
    }

    // Helper: serialize a bincode-able value, write, drop, return its length.
    macro_rules! emit_bincode {
        ($v:expr) => {{
            let bytes = bincode::serialize($v).map_err(BrainError::bincode)?;
            let n = bytes.len() as u64;
            rw.write_all(&bytes).map_err(BrainError::Io)?;
            drop(bytes);
            n
        }};
    }
    macro_rules! emit_section {
        ($bytes:expr) => {{
            let n = $bytes.len() as u64;
            rw.write_all(&$bytes).map_err(BrainError::Io)?;
            n
        }};
    }

    // ---- (5) Side-table sections, in the same on-disk order as the
    //         legacy writer (preserve byte-identical layout). ----
    let interest_offset = scar_offset + scar_len;
    let interest_len = emit_bincode!(&file.interest_profile);

    let meta_offset = interest_offset + interest_len;
    let meta_len = emit_bincode!(&file.meta_reflection);

    let prop_offset = meta_offset + meta_len;
    let prop_len = emit_bincode!(&file.research_proposals);

    let misc_blob = ResearchMiscBlob {
        config: file.research_config.clone(),
        next_id: file.next_research_proposal_id,
    };
    let misc_offset = prop_offset + prop_len;
    let misc_len = emit_bincode!(&misc_blob);
    drop(misc_blob);

    let gap_offset = misc_offset + misc_len;
    let gap_len = emit_bincode!(&file.gap_seed_frequency);

    // Port LB.2: section #8 sentinel — offset 0, length 0. No bytes.
    let wonder_offset: u64 = 0;
    let wonder_len: u64 = 0;

    let pending_offset = gap_offset + gap_len;
    let pending_len = emit_bincode!(&file.pending_log);

    let rejection_offset = pending_offset + pending_len;
    let rejection_len = emit_bincode!(&file.rejection_patterns);

    let lb_offset = rejection_offset + rejection_len;
    let lb_len = emit_bincode!(&file.lb_session);

    let cs_offset = lb_offset + lb_len;
    let cs_len = emit_bincode!(&file.cs_style);

    let pack_offset = cs_offset + cs_len;
    let pack_len = emit_bincode!(&file.pack_ledger);

    let runtime_config_bytes = file
        .runtime_config
        .encode_v14_section()
        .map_err(|e| BrainError::Bincode(e))?;
    let runtime_config_offset = pack_offset + pack_len;
    let runtime_config_len = emit_section!(runtime_config_bytes);
    drop(runtime_config_bytes);

    let runtime_config_v2_bytes = file
        .runtime_config
        .encode_v15_section()
        .map_err(|e| BrainError::Bincode(e))?;
    let runtime_config_v2_offset = runtime_config_offset + runtime_config_len;
    let runtime_config_v2_len = emit_section!(runtime_config_v2_bytes);
    drop(runtime_config_v2_bytes);

    let source_overlay_offset = runtime_config_v2_offset + runtime_config_v2_len;
    let source_overlay_len = emit_bincode!(&file.source_overlay);

    let cap_pack_ledger_offset = source_overlay_offset + source_overlay_len;
    let cap_pack_ledger_len = emit_bincode!(&file.cap_pack_ledger);

    let data_end = cap_pack_ledger_offset + cap_pack_ledger_len;
    debug_assert_eq!(
        rw.stream_position().map_err(BrainError::Io)?,
        data_end
    );

    // ---- (6) keyframe_count: walk bonds once more (cheap — no rkyv). ----
    let mut keyframe_count: u64 = 0;
    net.for_each_bond(|b| {
        keyframe_count = keyframe_count.saturating_add(b.keyframes().len() as u64);
        Ok(())
    })?;

    // ---- (7) Seek to 0, write real header + offset table. ----
    rw.seek(SeekFrom::Start(0)).map_err(BrainError::Io)?;

    let mut hdr = [0u8; HEADER_SIZE];
    hdr[0..8].copy_from_slice(&MAGIC_HEADER);
    hdr[8..12].copy_from_slice(&FORMAT_VERSION_V3.to_le_bytes());
    let flags = crate::compress::FORMAT_FLAG_COMPRESSED
        | crate::encrypt::FORMAT_FLAG_ENCRYPTED;
    hdr[12..16].copy_from_slice(&flags.to_le_bytes());
    hdr[16..24].copy_from_slice(&(net.node_count() as u64).to_le_bytes());
    hdr[24..32].copy_from_slice(&(net.bond_count() as u64).to_le_bytes());
    hdr[32..40].copy_from_slice(&keyframe_count.to_le_bytes());
    hdr[40..48].copy_from_slice(&(file.scar_log.scars.len() as u64).to_le_bytes());
    hdr[48..52].copy_from_slice(&SECTION_COUNT.to_le_bytes());
    // bytes 52..64 reserved (zero)
    rw.write_all(&hdr).map_err(BrainError::Io)?;

    let entries: [OffsetEntry; SECTION_COUNT as usize] = [
        OffsetEntry { section_id: SectionIdV3::Network as u32,            offset: net_offset,            length: net_len },
        OffsetEntry { section_id: SectionIdV3::InterestProfile as u32,    offset: interest_offset,       length: interest_len },
        OffsetEntry { section_id: SectionIdV3::MetaReflection as u32,     offset: meta_offset,           length: meta_len },
        OffsetEntry { section_id: SectionIdV3::ScarLog as u32,            offset: scar_offset,           length: scar_len },
        OffsetEntry { section_id: SectionIdV3::ResearchProposals as u32,  offset: prop_offset,           length: prop_len },
        OffsetEntry { section_id: SectionIdV3::ResearchMisc as u32,       offset: misc_offset,           length: misc_len },
        OffsetEntry { section_id: SectionIdV3::GapSeedFrequency as u32,   offset: gap_offset,            length: gap_len },
        OffsetEntry { section_id: SectionIdV3::WonderQueueState as u32,   offset: wonder_offset,         length: wonder_len },
        OffsetEntry { section_id: SectionIdV3::PendingLog as u32,         offset: pending_offset,        length: pending_len },
        OffsetEntry { section_id: SectionIdV3::RejectionPatterns as u32,  offset: rejection_offset,      length: rejection_len },
        OffsetEntry { section_id: SectionIdV3::SessionMemory as u32,      offset: lb_offset,             length: lb_len },
        OffsetEntry { section_id: SectionIdV3::ConversationalStyle as u32,offset: cs_offset,             length: cs_len },
        OffsetEntry { section_id: SectionIdV3::PackLedger as u32,         offset: pack_offset,           length: pack_len },
        OffsetEntry { section_id: SectionIdV3::RuntimeConfig as u32,      offset: runtime_config_offset, length: runtime_config_len },
        OffsetEntry { section_id: SectionIdV3::RuntimeConfigV2 as u32,    offset: runtime_config_v2_offset, length: runtime_config_v2_len },
        OffsetEntry { section_id: SectionIdV3::SourceOverlay as u32,      offset: source_overlay_offset, length: source_overlay_len },
        OffsetEntry { section_id: SectionIdV3::CapabilityPackLedger as u32, offset: cap_pack_ledger_offset, length: cap_pack_ledger_len },
    ];
    for e in &entries {
        write_entry_to(rw, e).map_err(BrainError::Io)?;
    }

    debug_assert_eq!(
        rw.stream_position().map_err(BrainError::Io)?,
        table_end
    );

    // ---- (8) CRC pass: read [0, data_end), hash with crc32fast. ----
    rw.seek(SeekFrom::Start(0)).map_err(BrainError::Io)?;
    let mut hasher = crc32fast::Hasher::new();
    let mut buf = vec![0u8; 256 * 1024];
    let mut remaining = data_end;
    while remaining > 0 {
        let n = remaining.min(buf.len() as u64) as usize;
        rw.read_exact(&mut buf[..n]).map_err(BrainError::Io)?;
        hasher.update(&buf[..n]);
        remaining -= n as u64;
    }
    drop(buf);
    let crc = hasher.finalize();

    // ---- (9) Seek to data_end, write trailer. ----
    rw.seek(SeekFrom::Start(data_end)).map_err(BrainError::Io)?;
    rw.write_all(&MAGIC_TRAILER).map_err(BrainError::Io)?;
    rw.write_all(&crc.to_le_bytes()).map_err(BrainError::Io)?;
    rw.write_all(&0u32.to_le_bytes()).map_err(BrainError::Io)?;

    Ok(data_end + TRAILER_SIZE as u64)
}

// =====================================================================
// OP-022.ef — borrowed-side-table streaming save path.
//
// Eliminates the `to_brain_file_no_network` clone that the legacy
// `write_v3_compressed_encrypted_to` required as input. Combined with
// `bincode::serialize_into` (no intermediate Vec per section), the
// peak heap during save drops by ~3.45 GiB (side-table clone) plus
// ~2.5 GiB (bincode buffer) measured via `op022_save_spike_diag`.
//
// Byte-identical output to the legacy path is enforced by the new
// regression test in this module.
// =====================================================================

/// Source of side-table bytes for the V3enc save path.
///
/// Each method writes the bytes for ONE section to the supplied writer
/// and returns the number of bytes written. Implementations should
/// call `bincode::serialize_into(&mut *w, &value)` for bincode-encoded
/// sections (no intermediate `Vec<u8>` is materialised), and snapshot
/// underlying stores ONLY at the point of use (and let the snapshot
/// drop at the end of the method), so peak heap during save holds at
/// most one section's snapshot+buffer at a time.
/// OP-022.s.v5 — combined Write + Seek bound, used by SideTableSource
/// generic methods so `bincode::serialize_into` monomorphises per call
/// site instead of paying virtual-dispatch cost on every primitive
/// write (the OP-022.ef regression).
///
/// `stream_position()` is the load-bearing reason Seek is required:
/// section length is captured from pre/post write positions, which
/// over the BufWriter costs one flush per section boundary instead of
/// an extra structure walk via `bincode::serialized_size`.
pub trait WriteSeek: std::io::Write + std::io::Seek {}
impl<T: std::io::Write + std::io::Seek + ?Sized> WriteSeek for T {}

/// OP-022.s.v5 — methods are generic over `W: WriteSeek` so bincode’s
/// per-primitive writes inline into the concrete BufWriter, restoring
/// the OLD-binary fast path while keeping OP-022.ef’s zero-Vec-clone
/// streaming model. `SideTableSource` is no longer object-safe; the
/// caller in `write_v3_compressed_encrypted_to_borrowed` is the only
/// site (verified via grep), so this is a non-issue.
pub trait SideTableSource {
    fn scar_log_scars_count(&self) -> u64;
    fn write_scar_log<W: WriteSeek + ?Sized>(&self, w: &mut W) -> Result<u64, BrainError>;
    fn write_interest_profile<W: WriteSeek + ?Sized>(&self, w: &mut W) -> Result<u64, BrainError>;
    fn write_meta_reflection<W: WriteSeek + ?Sized>(&self, w: &mut W) -> Result<u64, BrainError>;
    fn write_research_proposals<W: WriteSeek + ?Sized>(&self, w: &mut W) -> Result<u64, BrainError>;
    fn write_research_misc<W: WriteSeek + ?Sized>(&self, w: &mut W) -> Result<u64, BrainError>;
    fn write_gap_seed_frequency<W: WriteSeek + ?Sized>(&self, w: &mut W) -> Result<u64, BrainError>;
    fn write_pending_log<W: WriteSeek + ?Sized>(&self, w: &mut W) -> Result<u64, BrainError>;
    fn write_rejection_patterns<W: WriteSeek + ?Sized>(&self, w: &mut W) -> Result<u64, BrainError>;
    fn write_lb_session<W: WriteSeek + ?Sized>(&self, w: &mut W) -> Result<u64, BrainError>;
    fn write_cs_style<W: WriteSeek + ?Sized>(&self, w: &mut W) -> Result<u64, BrainError>;
    fn write_pack_ledger<W: WriteSeek + ?Sized>(&self, w: &mut W) -> Result<u64, BrainError>;
    fn write_runtime_config_v14<W: WriteSeek + ?Sized>(&self, w: &mut W) -> Result<u64, BrainError>;
    fn write_runtime_config_v15<W: WriteSeek + ?Sized>(&self, w: &mut W) -> Result<u64, BrainError>;
    fn write_source_overlay<W: WriteSeek + ?Sized>(&self, w: &mut W) -> Result<u64, BrainError>;
    fn write_cap_pack_ledger<W: WriteSeek + ?Sized>(&self, w: &mut W) -> Result<u64, BrainError>;
}

/// OP-022.ef — borrowed-side-table streaming writer for V3 encrypted brains.
///
/// Byte-identical to `write_v3_compressed_encrypted_to` but takes a
/// `&S: SideTableSource` instead of a pre-cloned `&BrainFile`. The
/// caller never builds a `BrainFile`, so the ~3.45 GiB side-table clone
/// (measured via op022_save_spike_diag against the production brain)
/// never happens.
/// OP-022.s.v4 — bundle of partial-passthrough inputs. When `Some`,
/// the writer prefers the partial-passthrough path (archive blocks
/// pass through verbatim, overlay encoded into appended blocks) over
/// re-encoding the full network. `passthrough_source` (the v3
/// full-passthrough hint) still wins when overlay is empty AND params
/// match — caller selects which path by passing only one of the two
/// args.
pub struct PartialPassthroughArgs<'a> {
    pub source: &'a crate::compress::CompressedNetwork,
    pub tombstoned_nodes: &'a std::collections::HashSet<crate::seed::Seed>,
    pub tombstoned_bonds: &'a std::collections::HashSet<(crate::seed::Seed, crate::seed::Seed)>,
    pub overlay_nodes: &'a dyn crate::compress::OverlayNodeSource,
    pub overlay_bonds: &'a dyn crate::compress::OverlayBondSource,
}

pub fn write_v3_compressed_encrypted_to_borrowed<N, S, RW>(
    net: &N,
    side_tables: &S,
    block_size: usize,
    zstd_level: i32,
    encryption_key: &crate::encrypt::MasterKey,
    rw: &mut RW,
    passthrough_source: Option<&crate::compress::CompressedNetwork>,
    partial_passthrough: Option<&PartialPassthroughArgs<'_>>,
) -> Result<u64, BrainError>
where
    N: crate::network::NetworkSource + ?Sized,
    S: SideTableSource + ?Sized,
    RW: std::io::Read + std::io::Write + std::io::Seek + Send,
{
    use std::io::{SeekFrom, Seek, Write};

    // OP-022.s.v5 — full-save BufWriter at the atomic_write layer.
    //
    // The save path writes ~115M tiny primitive `write_all()` calls
    // (8.3M nodes × 4 + 16M bonds × 5 + side-tables) directly to the
    // underlying tmp file, hitting the kernel one syscall per primitive
    // and pinning ~98% of save CPU in `vfs_write` / `fault_in_readable`
    // / `copy_page_from_iter_atomic` (flamegraph diag, internal notes.
    //
    // A 1 MiB BufWriter collapses those ~115M syscalls into ~6K
    // page-aligned writes for the entire 6.2 GB on-disk payload. The
    // wrapper spans the full write phase: header placeholder (1),
    // network section (2), pad (3), scar (4), all 14 side-tables (5),
    // header re-write at offset 0 (7), and the offset-entry table.
    //
    // Phase (8) is a CRC pass that reads the whole file back, so the
    // BufWriter is flushed via `into_inner` before (8) runs against
    // raw `rw` (which retains Read). Trailer write (9) is 16 bytes
    // after the CRC read — also raw `rw`, no buffering needed.
    //
    // The brief framed this as "wrap at the atomic_write_with closure",
    // but the inner CRC pass needs Read which BufWriter doesn’t impl;
    // wrapping here is the equivalent single-buffer span without
    // breaking the read-back step.
    let mut bw = std::io::BufWriter::with_capacity(1 << 20, &mut *rw);

    // (1) Seek to 0, reserve header + offset table placeholder.
    bw.seek(SeekFrom::Start(0)).map_err(BrainError::Io)?;
    let header_end = HEADER_SIZE as u64;
    let table_bytes = (SECTION_COUNT as u64) * (ENTRY_SIZE as u64);
    let table_end = header_end + table_bytes;
    let placeholder_zeros = vec![0u8; table_end as usize];
    bw.write_all(&placeholder_zeros).map_err(BrainError::Io)?;
    drop(placeholder_zeros);

    // (2) Stream the compressed-encrypted network section.
    // OP-022.s.v3: when a passthrough_source is provided AND its on-
    // disk parameters match this save’s parameters, copy compressed
    // block frames verbatim from the source mmap into the output —
    // no zstd decompress, no XOR decrypt/re-encrypt, no rkyv decode.
    // Falls through to the slow path silently when params differ.
    let net_offset = table_end;
    let save_params = crate::compress::CompressedSaveParams {
        block_size_target: block_size,
        zstd_level,
        encryption_key: Some(encryption_key),
    };
    // OP-022.s.v4 three-way dispatch:
    //   (a) full passthrough (overlay empty + params match)  — fastest
    //   (b) partial passthrough (overlay non-empty + params match) — v4 path
    //   (c) slow re-encode (params mismatch OR no source) — fallback
    let net_len = if let Some(src) = passthrough_source {
        if crate::compress::can_passthrough_compressed_blocks(src, &save_params) {
            crate::compress::write_compressed_network_payload_passthrough_to(
                src,
                &save_params,
                &mut bw,
            )?
        } else {
            crate::compress::write_compressed_network_payload_to(
                net,
                block_size,
                zstd_level,
                Some(encryption_key),
                &mut bw,
            )?
        }
    } else if let Some(pp) = partial_passthrough {
        let overlay_upper = pp.overlay_nodes.count() + pp.overlay_bonds.count();
        if crate::compress::can_passthrough_blocks_with_overlay(
            pp.source,
            &save_params,
            overlay_upper,
        ) {
            crate::compress::write_compressed_network_payload_partial_passthrough_to(
                pp.source,
                &save_params,
                pp.tombstoned_nodes,
                pp.tombstoned_bonds,
                pp.overlay_nodes,
                pp.overlay_bonds,
                &mut bw,
            )?
        } else {
            crate::compress::write_compressed_network_payload_to(
                net,
                block_size,
                zstd_level,
                Some(encryption_key),
                &mut bw,
            )?
        }
    } else {
        crate::compress::write_compressed_network_payload_to(
            net,
            block_size,
            zstd_level,
            Some(encryption_key),
            &mut bw,
        )?
    };

    // (3) Pad to RKYV_ALIGN.
    let scar_offset = align_up(net_offset + net_len, RKYV_ALIGN);
    let pad = scar_offset - (net_offset + net_len);
    if pad > 0 {
        let pad_zeros = vec![0u8; pad as usize];
        bw.write_all(&pad_zeros).map_err(BrainError::Io)?;
    }

    // (4) scar_log via SideTableSource (still rkyv internally).
    let scar_len = side_tables.write_scar_log(&mut bw)?;

    // (5) Side-table sections, in the SAME on-disk order as the
    //     legacy writer (preserve byte-identical layout). Each section
    //     is written via SideTableSource which uses
    //     bincode::serialize_into directly (no Vec<u8> per section).
    //     OP-022.s.v5: the SideTableSource methods are now generic
    //     over `W: WriteSeek`, so `bincode::serialize_into` inlines
    //     into the concrete BufWriter (no per-primitive vtable).
    let interest_offset = scar_offset + scar_len;
    let interest_len = side_tables.write_interest_profile(&mut bw)?;

    let meta_offset = interest_offset + interest_len;
    let meta_len = side_tables.write_meta_reflection(&mut bw)?;

    let prop_offset = meta_offset + meta_len;
    let prop_len = side_tables.write_research_proposals(&mut bw)?;

    let misc_offset = prop_offset + prop_len;
    let misc_len = side_tables.write_research_misc(&mut bw)?;

    let gap_offset = misc_offset + misc_len;
    let gap_len = side_tables.write_gap_seed_frequency(&mut bw)?;

    let wonder_offset: u64 = 0;
    let wonder_len: u64 = 0;

    let pending_offset = gap_offset + gap_len;
    let pending_len = side_tables.write_pending_log(&mut bw)?;

    let rejection_offset = pending_offset + pending_len;
    let rejection_len = side_tables.write_rejection_patterns(&mut bw)?;

    let lb_offset = rejection_offset + rejection_len;
    let lb_len = side_tables.write_lb_session(&mut bw)?;

    let cs_offset = lb_offset + lb_len;
    let cs_len = side_tables.write_cs_style(&mut bw)?;

    let pack_offset = cs_offset + cs_len;
    let pack_len = side_tables.write_pack_ledger(&mut bw)?;

    let runtime_config_offset = pack_offset + pack_len;
    let runtime_config_len = side_tables.write_runtime_config_v14(&mut bw)?;

    let runtime_config_v2_offset = runtime_config_offset + runtime_config_len;
    let runtime_config_v2_len = side_tables.write_runtime_config_v15(&mut bw)?;

    let source_overlay_offset = runtime_config_v2_offset + runtime_config_v2_len;
    let source_overlay_len = side_tables.write_source_overlay(&mut bw)?;

    let cap_pack_ledger_offset = source_overlay_offset + source_overlay_len;
    let cap_pack_ledger_len = side_tables.write_cap_pack_ledger(&mut bw)?;

    let data_end = cap_pack_ledger_offset + cap_pack_ledger_len;

    // (6) keyframe_count: walk bonds once more.
    let mut keyframe_count: u64 = 0;
    net.for_each_bond(|b| {
        keyframe_count = keyframe_count.saturating_add(b.keyframes().len() as u64);
        Ok(())
    })?;

    // (7) Seek to 0, write real header + offset table.
    // BufWriter<W: Write+Seek> impls Seek by flushing first, so this
    // seek backwards is safe — the data buffered behind us has already
    // been pushed to the underlying file by the seek’s flush.
    bw.seek(SeekFrom::Start(0)).map_err(BrainError::Io)?;
    let mut hdr = [0u8; HEADER_SIZE];
    hdr[0..8].copy_from_slice(&MAGIC_HEADER);
    hdr[8..12].copy_from_slice(&FORMAT_VERSION_V3.to_le_bytes());
    let flags = crate::compress::FORMAT_FLAG_COMPRESSED
        | crate::encrypt::FORMAT_FLAG_ENCRYPTED;
    hdr[12..16].copy_from_slice(&flags.to_le_bytes());
    hdr[16..24].copy_from_slice(&(net.node_count() as u64).to_le_bytes());
    hdr[24..32].copy_from_slice(&(net.bond_count() as u64).to_le_bytes());
    hdr[32..40].copy_from_slice(&keyframe_count.to_le_bytes());
    hdr[40..48].copy_from_slice(&side_tables.scar_log_scars_count().to_le_bytes());
    hdr[48..52].copy_from_slice(&SECTION_COUNT.to_le_bytes());
    bw.write_all(&hdr).map_err(BrainError::Io)?;

    let entries: [OffsetEntry; SECTION_COUNT as usize] = [
        OffsetEntry { section_id: SectionIdV3::Network as u32,            offset: net_offset,            length: net_len },
        OffsetEntry { section_id: SectionIdV3::InterestProfile as u32,    offset: interest_offset,       length: interest_len },
        OffsetEntry { section_id: SectionIdV3::MetaReflection as u32,     offset: meta_offset,           length: meta_len },
        OffsetEntry { section_id: SectionIdV3::ScarLog as u32,            offset: scar_offset,           length: scar_len },
        OffsetEntry { section_id: SectionIdV3::ResearchProposals as u32,  offset: prop_offset,           length: prop_len },
        OffsetEntry { section_id: SectionIdV3::ResearchMisc as u32,       offset: misc_offset,           length: misc_len },
        OffsetEntry { section_id: SectionIdV3::GapSeedFrequency as u32,   offset: gap_offset,            length: gap_len },
        OffsetEntry { section_id: SectionIdV3::WonderQueueState as u32,   offset: wonder_offset,         length: wonder_len },
        OffsetEntry { section_id: SectionIdV3::PendingLog as u32,         offset: pending_offset,        length: pending_len },
        OffsetEntry { section_id: SectionIdV3::RejectionPatterns as u32,  offset: rejection_offset,      length: rejection_len },
        OffsetEntry { section_id: SectionIdV3::SessionMemory as u32,      offset: lb_offset,             length: lb_len },
        OffsetEntry { section_id: SectionIdV3::ConversationalStyle as u32,offset: cs_offset,             length: cs_len },
        OffsetEntry { section_id: SectionIdV3::PackLedger as u32,         offset: pack_offset,           length: pack_len },
        OffsetEntry { section_id: SectionIdV3::RuntimeConfig as u32,      offset: runtime_config_offset, length: runtime_config_len },
        OffsetEntry { section_id: SectionIdV3::RuntimeConfigV2 as u32,    offset: runtime_config_v2_offset, length: runtime_config_v2_len },
        OffsetEntry { section_id: SectionIdV3::SourceOverlay as u32,      offset: source_overlay_offset, length: source_overlay_len },
        OffsetEntry { section_id: SectionIdV3::CapabilityPackLedger as u32, offset: cap_pack_ledger_offset, length: cap_pack_ledger_len },
    ];
    for e in &entries {
        write_entry_to(&mut bw, e).map_err(BrainError::Io)?;
    }

    // OP-022.s.v5 — flush BufWriter so all buffered bytes hit the
    // underlying tmp file BEFORE the CRC read pass (8) reads them back
    // via raw `rw`. `into_inner` flushes implicitly and returns the
    // borrowed inner writer; the explicit `flush()` first surfaces any
    // I/O error cleanly (into_inner only returns IntoInnerError on
    // flush failure, which is awkward to unwrap).
    std::io::Write::flush(&mut bw).map_err(BrainError::Io)?;
    let _ = bw.into_inner().map_err(|e| BrainError::Io(e.into_error()))?;

    // (8) CRC pass: read [0, data_end).
    rw.seek(SeekFrom::Start(0)).map_err(BrainError::Io)?;
    let mut hasher = crc32fast::Hasher::new();
    let mut buf = vec![0u8; 256 * 1024];
    let mut remaining = data_end;
    while remaining > 0 {
        let n = remaining.min(buf.len() as u64) as usize;
        rw.read_exact(&mut buf[..n]).map_err(BrainError::Io)?;
        hasher.update(&buf[..n]);
        remaining -= n as u64;
    }
    drop(buf);
    let crc = hasher.finalize();

    // (9) Seek to data_end, write trailer.
    rw.seek(SeekFrom::Start(data_end)).map_err(BrainError::Io)?;
    rw.write_all(&MAGIC_TRAILER).map_err(BrainError::Io)?;
    rw.write_all(&crc.to_le_bytes()).map_err(BrainError::Io)?;
    rw.write_all(&0u32.to_le_bytes()).map_err(BrainError::Io)?;

    Ok(data_end + TRAILER_SIZE as u64)
}
