//! Storage v2 — binary brain file format.
//!
//! Phase 1 of the Storage v2 plan. This module defines a binary on-disk
//! layout for `BrainFile` that replaces the JSON-pretty encoding used by
//! `BRAIN_FORMAT_VERSION = 1`. The v1 JSON path remains intact in
//! `brain.rs`; this module only adds the v2 capability. `Brain::load_or_new`
//! auto-detects which format a given path holds by sniffing the first 8
//! bytes (the v2 magic). `Brain::save_to_format` dispatches between the two.
//!
//! Layout
//! ------
//!
//! ```text
//! [0..8]    magic   = b"LU4BRAIN"
//! [8..12]   version = u32 LE  (= 2)
//! [12..16]  flags   = u32 LE  (bit 0: compressed - reserved Phase 3,
//!                              bit 1: encrypted - reserved Phase 4;
//!                              both 0 in Phase 1)
//! [16..24]  node_count u64 LE
//! [24..32]  bond_count u64 LE
//! [32..40]  keyframe_count u64 LE
//! [40..48]  scar_count u64 LE
//! [48..52]  section_count u32 LE
//! [52..64]  reserved (zeroed) -> fixed header is 64 bytes
//!
//! [64 .. 64 + 24*section_count]  offset table
//!     each entry is 24 bytes:
//!       u32 LE  section_id
//!       u32 LE  reserved (zero)
//!       u64 LE  offset (absolute byte offset within file)
//!       u64 LE  length (in bytes)
//!
//! Sections (each is an opaque bincode-1.x serialized blob of the
//! corresponding sub-structure of BrainFile):
//!
//!   Network          (NetworkSnapshot)
//!   InterestProfile  (InterestProfile)
//!   MetaReflection   (MetaReflection)
//!   ScarLog          (ScarLog)
//!   ResearchProposals (Vec<ResearchProposal>)
//!   ResearchMisc     ((ResearchConfig, u64) tuple — config + next_id)
//!   GapSeedFrequency (HashMap<Seed, u32>)
//!
//! Trailer (last 16 bytes of the file):
//!   [trailer_pos .. +8]  magic = b"LU4ENDOK"
//!   [trailer_pos+8 ..+12] crc32 (IEEE) of every byte before the trailer
//!   [trailer_pos+12..+16] reserved (zeroed)
//! ```
//!
//! All multi-byte integers are little-endian. The offset table is the
//! handle that future Phase 2 (mmap zero-copy load) will use to locate a
//! section without parsing it; Phase 3 (per-block zstd) will set the
//! `flags` compressed bit and replace each section blob with a
//! length-prefixed compressed stream; Phase 4 will set the encrypted bit
//! and wrap the whole post-header byte range in a DF XOR transform. None
//! of those are implemented here — Phase 1 only adds the binary capability
//! and the migration tool.

use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom, Write};

use serde::{Deserialize, Serialize};

use crate::brain::{BrainError, BrainFile};
use crate::curiosity::InterestProfile;
use crate::network::NetworkSnapshot;
use crate::research::proposal::ResearchProposal;
use crate::research::router::ResearchConfig;
use crate::scar::ScarLog;
use crate::seed::Seed;
use crate::self_reflection::MetaReflection;

/// Magic bytes at the start of a v2 binary brain file.
pub const MAGIC_HEADER: [u8; 8] = *b"LU4BRAIN";
/// Magic bytes at the start of the trailer.
pub const MAGIC_TRAILER: [u8; 8] = *b"LU4ENDOK";
/// On-disk format version for the binary layout.
pub const FORMAT_VERSION_V2: u32 = 2;
/// Fixed header size, in bytes.
pub const HEADER_SIZE: usize = 64;
/// Per-section offset table entry size, in bytes.
pub const ENTRY_SIZE: usize = 24;
/// Trailer size, in bytes.
pub const TRAILER_SIZE: usize = 16;

/// Section identifiers. The numeric values are persisted on disk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum SectionId {
    Network = 1,
    InterestProfile = 2,
    MetaReflection = 3,
    ScarLog = 4,
    ResearchProposals = 5,
    ResearchMisc = 6,
    GapSeedFrequency = 7,
}

impl SectionId {
    #[allow(dead_code)]
    fn from_u32(v: u32) -> Option<Self> {
        match v {
            1 => Some(SectionId::Network),
            2 => Some(SectionId::InterestProfile),
            3 => Some(SectionId::MetaReflection),
            4 => Some(SectionId::ScarLog),
            5 => Some(SectionId::ResearchProposals),
            6 => Some(SectionId::ResearchMisc),
            7 => Some(SectionId::GapSeedFrequency),
            _ => None,
        }
    }
}

/// Detected on-disk format for a brain file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BrainFormat {
    /// v1 JSON-pretty (the historic format).
    V1Json,
    /// v2 binary (this module).
    V2Binary,
    /// v3 binary (Storage v2 Phase 2.2; defined in `brain_io_v3`).
    /// Shares v2's leading magic; the version byte at [8..12]
    /// distinguishes the two.
    V3Binary,
}

/// Sniff the leading 12 bytes of a brain file and decide which format
/// it is.
///
/// v2 and v3 share the same `MAGIC_HEADER`; the version field at bytes
/// 8..12 is what tells them apart. Files that don't start with the
/// magic are assumed to be v1 JSON (which historically always starts
/// with `{`). The reader is left at the start (we seek back) so the
/// caller can pass it on to the appropriate parser.
pub fn detect_format<R: Read + Seek>(reader: &mut R) -> Result<BrainFormat, BrainError> {
    let mut buf = [0u8; 12];
    let n = reader.read(&mut buf)?;
    reader.seek(SeekFrom::Start(0))?;
    if n >= 8 && buf[0..8] == MAGIC_HEADER {
        // We have the magic. If we also got the version bytes, branch
        // on the version. A truncated read that contains the magic but
        // not the full version is treated as v2 — the v2 reader will
        // surface a clear "file too short" error rather than us
        // surfacing an ambiguous one here.
        if n >= 12 {
            let version = u32::from_le_bytes(buf[8..12].try_into().unwrap());
            if version == 3 {
                return Ok(BrainFormat::V3Binary);
            }
        }
        Ok(BrainFormat::V2Binary)
    } else {
        Ok(BrainFormat::V1Json)
    }
}

/// Convenience: detect the format of a path on disk.
pub fn detect_format_path<P: AsRef<std::path::Path>>(
    path: P,
) -> Result<BrainFormat, BrainError> {
    let mut f = std::fs::File::open(path)?;
    detect_format(&mut f)
}

/// The v2 fixed header.
#[derive(Debug, Clone, Copy)]
pub struct V2Header {
    pub version: u32,
    pub flags: u32,
    pub node_count: u64,
    pub bond_count: u64,
    pub keyframe_count: u64,
    pub scar_count: u64,
    pub section_count: u32,
}

impl V2Header {
    fn write<W: Write>(&self, w: &mut W) -> std::io::Result<()> {
        let mut buf = [0u8; HEADER_SIZE];
        buf[0..8].copy_from_slice(&MAGIC_HEADER);
        buf[8..12].copy_from_slice(&self.version.to_le_bytes());
        buf[12..16].copy_from_slice(&self.flags.to_le_bytes());
        buf[16..24].copy_from_slice(&self.node_count.to_le_bytes());
        buf[24..32].copy_from_slice(&self.bond_count.to_le_bytes());
        buf[32..40].copy_from_slice(&self.keyframe_count.to_le_bytes());
        buf[40..48].copy_from_slice(&self.scar_count.to_le_bytes());
        buf[48..52].copy_from_slice(&self.section_count.to_le_bytes());
        // bytes 52..64 are reserved (already zero from init)
        w.write_all(&buf)
    }

    fn read<R: Read>(r: &mut R) -> Result<Self, BrainError> {
        let mut buf = [0u8; HEADER_SIZE];
        r.read_exact(&mut buf)?;
        if buf[0..8] != MAGIC_HEADER {
            return Err(BrainError::Format(format!(
                "v2 header magic mismatch: got {:02x?}",
                &buf[0..8]
            )));
        }
        let version = u32::from_le_bytes(buf[8..12].try_into().unwrap());
        let flags = u32::from_le_bytes(buf[12..16].try_into().unwrap());
        let node_count = u64::from_le_bytes(buf[16..24].try_into().unwrap());
        let bond_count = u64::from_le_bytes(buf[24..32].try_into().unwrap());
        let keyframe_count = u64::from_le_bytes(buf[32..40].try_into().unwrap());
        let scar_count = u64::from_le_bytes(buf[40..48].try_into().unwrap());
        let section_count = u32::from_le_bytes(buf[48..52].try_into().unwrap());
        Ok(V2Header {
            version,
            flags,
            node_count,
            bond_count,
            keyframe_count,
            scar_count,
            section_count,
        })
    }
}

/// One entry in the offset table.
#[derive(Debug, Clone, Copy)]
struct OffsetEntry {
    section_id: u32,
    offset: u64,
    length: u64,
}

fn write_entry<W: Write>(w: &mut W, e: &OffsetEntry) -> std::io::Result<()> {
    let mut buf = [0u8; ENTRY_SIZE];
    buf[0..4].copy_from_slice(&e.section_id.to_le_bytes());
    // bytes 4..8 reserved (zero)
    buf[8..16].copy_from_slice(&e.offset.to_le_bytes());
    buf[16..24].copy_from_slice(&e.length.to_le_bytes());
    w.write_all(&buf)
}

fn read_entry<R: Read>(r: &mut R) -> Result<OffsetEntry, BrainError> {
    let mut buf = [0u8; ENTRY_SIZE];
    r.read_exact(&mut buf)?;
    Ok(OffsetEntry {
        section_id: u32::from_le_bytes(buf[0..4].try_into().unwrap()),
        offset: u64::from_le_bytes(buf[8..16].try_into().unwrap()),
        length: u64::from_le_bytes(buf[16..24].try_into().unwrap()),
    })
}

/// Helper used internally to bundle ResearchConfig + next_id into one
/// section. Tuple bincode would also work; this gives the field names a
/// place to live.
///
/// Visible at `pub(crate)` because v3 (`brain_io_v3`) reuses the exact
/// same on-disk shape for this side-table and we want a single source
/// of truth. Both formats serialize this with bincode-1.x.
#[derive(Serialize, Deserialize)]
pub(crate) struct ResearchMiscBlob {
    pub(crate) config: ResearchConfig,
    pub(crate) next_id: u64,
}

/// Serialize a `BrainFile` into a v2 binary stream.
///
/// The writer must be a `Write + Seek` because we patch the offset table
/// in place after each section is written. We use an in-memory buffer
/// rather than requiring a real seekable file, so this works for plain
/// `Vec<u8>` writers too.
pub fn write_v2(file: &BrainFile) -> Result<Vec<u8>, BrainError> {
    // Total keyframes across all bonds — exposed in the header so a
    // future status command can read it without parsing the network
    // section.
    let keyframe_count: u64 = file
        .network
        .bonds
        .iter()
        .map(|b| b.keyframes().len() as u64)
        .sum();
    let header = V2Header {
        version: FORMAT_VERSION_V2,
        flags: 0,
        node_count: file.network.nodes.len() as u64,
        bond_count: file.network.bonds.len() as u64,
        keyframe_count,
        scar_count: file.scar_log.scars.len() as u64,
        section_count: 7,
    };

    // Pre-serialize all sections into byte buffers so we know lengths
    // up front and can build the offset table in one pass.
    let net_bytes = bincode::serialize(&file.network).map_err(BrainError::bincode)?;
    let interest_bytes = bincode::serialize(&file.interest_profile).map_err(BrainError::bincode)?;
    let meta_bytes = bincode::serialize(&file.meta_reflection).map_err(BrainError::bincode)?;
    let scar_bytes = bincode::serialize(&file.scar_log).map_err(BrainError::bincode)?;
    let prop_bytes = bincode::serialize(&file.research_proposals).map_err(BrainError::bincode)?;
    let misc_blob = ResearchMiscBlob {
        config: file.research_config.clone(),
        next_id: file.next_research_proposal_id,
    };
    let misc_bytes = bincode::serialize(&misc_blob).map_err(BrainError::bincode)?;
    let gap_bytes = bincode::serialize(&file.gap_seed_frequency).map_err(BrainError::bincode)?;

    let sections: [(SectionId, &[u8]); 7] = [
        (SectionId::Network, &net_bytes),
        (SectionId::InterestProfile, &interest_bytes),
        (SectionId::MetaReflection, &meta_bytes),
        (SectionId::ScarLog, &scar_bytes),
        (SectionId::ResearchProposals, &prop_bytes),
        (SectionId::ResearchMisc, &misc_bytes),
        (SectionId::GapSeedFrequency, &gap_bytes),
    ];

    let table_bytes = (sections.len() * ENTRY_SIZE) as u64;
    let mut cursor = HEADER_SIZE as u64 + table_bytes;
    let mut entries: Vec<OffsetEntry> = Vec::with_capacity(sections.len());
    for (id, blob) in &sections {
        entries.push(OffsetEntry {
            section_id: *id as u32,
            offset: cursor,
            length: blob.len() as u64,
        });
        cursor += blob.len() as u64;
    }

    let mut out: Vec<u8> = Vec::with_capacity(cursor as usize + TRAILER_SIZE);
    header.write(&mut out)?;
    for e in &entries {
        write_entry(&mut out, e)?;
    }
    for (_id, blob) in &sections {
        out.write_all(blob)?;
    }

    // Trailer: magic, crc32 of everything-so-far, reserved.
    let crc = crc32fast::hash(&out);
    out.write_all(&MAGIC_TRAILER)?;
    out.write_all(&crc.to_le_bytes())?;
    out.write_all(&0u32.to_le_bytes())?;

    Ok(out)
}

/// Parse a v2 binary stream back into a `BrainFile`.
pub fn read_v2(bytes: &[u8]) -> Result<BrainFile, BrainError> {
    if bytes.len() < HEADER_SIZE + TRAILER_SIZE {
        return Err(BrainError::Format(format!(
            "file too short for v2 layout: {} bytes",
            bytes.len()
        )));
    }

    // Verify trailer first so corruption is caught before we trust
    // anything else.
    let trailer_pos = bytes.len() - TRAILER_SIZE;
    if bytes[trailer_pos..trailer_pos + 8] != MAGIC_TRAILER {
        return Err(BrainError::Format(
            "v2 trailer magic mismatch — file truncated or corrupted".into(),
        ));
    }
    let stored_crc = u32::from_le_bytes(
        bytes[trailer_pos + 8..trailer_pos + 12].try_into().unwrap(),
    );
    let actual_crc = crc32fast::hash(&bytes[..trailer_pos]);
    if stored_crc != actual_crc {
        return Err(BrainError::Format(format!(
            "v2 checksum mismatch: stored 0x{:08x}, computed 0x{:08x}",
            stored_crc, actual_crc
        )));
    }

    let mut cursor = std::io::Cursor::new(bytes);
    let header = V2Header::read(&mut cursor)?;
    if header.version != FORMAT_VERSION_V2 {
        return Err(BrainError::VersionMismatch(header.version));
    }
    if header.flags != 0 {
        return Err(BrainError::Format(format!(
            "v2 flags=0x{:08x} not understood — newer format from a future port?",
            header.flags
        )));
    }
    let mut entries = Vec::with_capacity(header.section_count as usize);
    for _ in 0..header.section_count {
        entries.push(read_entry(&mut cursor)?);
    }

    // Build a (id -> &[u8]) lookup so we don't depend on any particular
    // section ordering on disk.
    let mut sections: HashMap<u32, &[u8]> = HashMap::new();
    for e in &entries {
        let start = e.offset as usize;
        let end = start.saturating_add(e.length as usize);
        if end > trailer_pos {
            return Err(BrainError::Format(format!(
                "section {} runs past data region: offset={} len={} trailer_pos={}",
                e.section_id, e.offset, e.length, trailer_pos
            )));
        }
        sections.insert(e.section_id, &bytes[start..end]);
    }

    fn require<'a>(
        sections: &HashMap<u32, &'a [u8]>,
        id: SectionId,
    ) -> Result<&'a [u8], BrainError> {
        sections.get(&(id as u32)).copied().ok_or_else(|| {
            BrainError::Format(format!("v2 file missing required section {:?}", id))
        })
    }

    fn de<'a, T: serde::de::DeserializeOwned>(
        sections: &HashMap<u32, &'a [u8]>,
        id: SectionId,
    ) -> Result<T, BrainError> {
        let blob = require(sections, id)?;
        bincode::deserialize::<T>(blob).map_err(|e| {
            BrainError::Bincode(format!(
                "section {:?} (id {}, {} bytes): {}",
                id, id as u32, blob.len(), e
            ))
        })
    }
    let network: NetworkSnapshot = de(&sections, SectionId::Network)?;
    let interest_profile: InterestProfile = de(&sections, SectionId::InterestProfile)?;
    let meta_reflection: MetaReflection = de(&sections, SectionId::MetaReflection)?;
    let scar_log: ScarLog = de(&sections, SectionId::ScarLog)?;
    let research_proposals: Vec<ResearchProposal> = de(&sections, SectionId::ResearchProposals)?;
    let misc: ResearchMiscBlob = de(&sections, SectionId::ResearchMisc)?;
    let gap_seed_frequency: HashMap<Seed, u32> = de(&sections, SectionId::GapSeedFrequency)?;

    // Sanity check the on-disk counts against the rebuilt structures.
    // A mismatch means the file was tampered with or there's a writer bug.
    if header.node_count != network.nodes.len() as u64 {
        return Err(BrainError::Format(format!(
            "v2 header node_count={} disagrees with network section ({} nodes)",
            header.node_count,
            network.nodes.len()
        )));
    }
    if header.bond_count != network.bonds.len() as u64 {
        return Err(BrainError::Format(format!(
            "v2 header bond_count={} disagrees with network section ({} bonds)",
            header.bond_count,
            network.bonds.len()
        )));
    }
    let kf: u64 = network.bonds.iter().map(|b| b.keyframes().len() as u64).sum();
    if header.keyframe_count != kf {
        return Err(BrainError::Format(format!(
            "v2 header keyframe_count={} disagrees with bonds ({} keyframes)",
            header.keyframe_count, kf
        )));
    }
    if header.scar_count != scar_log.scars.len() as u64 {
        return Err(BrainError::Format(format!(
            "v2 header scar_count={} disagrees with scar_log ({} scars)",
            header.scar_count,
            scar_log.scars.len()
        )));
    }

    Ok(BrainFile {
        format_version: FORMAT_VERSION_V2,
        network,
        interest_profile,
        meta_reflection,
        scar_log,
        research_proposals,
        research_config: misc.config,
        next_research_proposal_id: misc.next_id,
        gap_seed_frequency,
        // Port J.1: v2 has no wonder_state on disk; load with empty.
        wonder_state: crate::curiosity::WonderState::default(),
        negation_overlay: crate::negation::NegationOverlay::new(),
        source_overlay: crate::source::SourceOverlay::new(),
        // Port T.4: pending log snapshot (default empty for pre-T.4 brains).
        pending_log: Default::default(),
        // Port T.8: rejection patterns snapshot (default empty for pre-T.8 brains).
        rejection_patterns: Default::default(),
        lb_session: Default::default(),
        cs_style: Default::default(),
        pack_ledger: Default::default(),
        cap_pack_ledger: Default::default(),
        // Port Cat.3c: runtime config defaults to absent for v2 brains;
        // the load path then falls back to the legacy sidecars.
        runtime_config: Default::default(),
        runtime_config_present: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brain::Brain;
    use crate::node::Confidence;
    use crate::BondType;

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
        BrainFile {
            format_version: FORMAT_VERSION_V2,
            network: b.network.to_snapshot(),
            interest_profile: b.interest_profile.clone(),
            meta_reflection: b.meta_reflection.clone(),
            scar_log: b.scar_log.clone(),
            research_proposals: b.research_proposals.clone(),
            research_config: b.research_config.clone(),
            next_research_proposal_id: b.next_research_proposal_id,
            gap_seed_frequency: b.gap_seed_frequency.clone(),
            wonder_state: Default::default(),
            negation_overlay: crate::negation::NegationOverlay::new(),
            source_overlay: crate::source::SourceOverlay::new(),
            // Port T.4: pending log snapshot (default empty for pre-T.4 brains).
            pending_log: Default::default(),
            rejection_patterns: Default::default(),
        lb_session: Default::default(),
        cs_style: Default::default(),
        pack_ledger: Default::default(),
        cap_pack_ledger: Default::default(),
        // Port Cat.3c: runtime config defaults to absent for v2 brains;
        // the load path then falls back to the legacy sidecars.
        runtime_config: Default::default(),
        runtime_config_present: false,
        }
    }

    #[test]
    fn v2_round_trip_preserves_counts() {
        let b = small_brain();
        let file = brain_to_file(&b);
        let bytes = write_v2(&file).expect("write_v2 should succeed");

        // Magic at the start; trailer at the end.
        assert_eq!(&bytes[0..8], &MAGIC_HEADER);
        let trailer_pos = bytes.len() - TRAILER_SIZE;
        assert_eq!(&bytes[trailer_pos..trailer_pos + 8], &MAGIC_TRAILER);

        let parsed = read_v2(&bytes).expect("read_v2 should succeed");
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
    fn v2_detect_format_distinguishes_v1_and_v2() {
        // v2 sample
        let b = small_brain();
        let v2 = write_v2(&brain_to_file(&b)).unwrap();
        let mut v2_cur = std::io::Cursor::new(&v2);
        assert_eq!(detect_format(&mut v2_cur).unwrap(), BrainFormat::V2Binary);

        // v1 JSON sample (anything not starting with magic)
        let v1 = b"{\"format_version\":1}".to_vec();
        let mut v1_cur = std::io::Cursor::new(&v1);
        assert_eq!(detect_format(&mut v1_cur).unwrap(), BrainFormat::V1Json);
    }

    #[test]
    fn v2_detects_corruption() {
        let b = small_brain();
        let mut bytes = write_v2(&brain_to_file(&b)).unwrap();
        // Flip a byte in the middle of the data region.
        let i = HEADER_SIZE + ENTRY_SIZE * 7 + 4;
        bytes[i] ^= 0xff;
        let err = read_v2(&bytes).expect_err("corrupt file must not parse");
        match err {
            BrainError::Format(msg) => assert!(msg.contains("checksum") || msg.contains("section")),
            other => panic!("expected Format error, got {:?}", other),
        }
    }
    // ----- v1-load + migration tests -----
    //
    // We mirror brain.rs's inline tempdir helper so this test module
    // doesn't need a dev-dep.

    mod tempfile {
        use std::path::PathBuf;
        pub struct TempDir { path: PathBuf }
        impl TempDir { pub fn path(&self) -> &std::path::Path { &self.path } }
        impl Drop for TempDir { fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.path); } }
        pub fn tempdir() -> std::io::Result<TempDir> {
            use std::sync::atomic::{AtomicU64, Ordering};
            static COUNTER: AtomicU64 = AtomicU64::new(0);
            let n = COUNTER.fetch_add(1, Ordering::SeqCst);
            let mut p = std::env::temp_dir();
            p.push(format!("lu4d-brain-io-test-{}-{}", std::process::id(), n));
            std::fs::create_dir_all(&p)?;
            Ok(TempDir { path: p })
        }
    }

    use crate::brain::{BrainSaveFormat, V1_JSON_VERSION};

    #[test]
    fn v1_load_still_works_after_v2_landed() {
        // v1 read path must continue working so legacy brains can be
        // migrated via --migrate-from-v1. Build the v1 file directly
        // via serde_json (NOT via Brain::save, which now refuses v1)
        // and verify Brain::load_or_new auto-detects and parses it.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("v1_brain.lbr");

        let b = small_brain();
        let file = BrainFile {
            format_version: V1_JSON_VERSION,
            network: b.network.to_snapshot(),
            interest_profile: b.interest_profile.clone(),
            meta_reflection: b.meta_reflection.clone(),
            scar_log: b.scar_log.clone(),
            research_proposals: b.research_proposals.clone(),
            research_config: b.research_config.clone(),
            next_research_proposal_id: b.next_research_proposal_id,
            gap_seed_frequency: b.gap_seed_frequency.clone(),
            wonder_state: Default::default(),
            negation_overlay: crate::negation::NegationOverlay::new(),
            source_overlay: crate::source::SourceOverlay::new(),
            // Port T.4: pending log snapshot (default empty for pre-T.4 brains).
            pending_log: Default::default(),
            rejection_patterns: Default::default(),
        lb_session: Default::default(),
        cs_style: Default::default(),
        pack_ledger: Default::default(),
        cap_pack_ledger: Default::default(),
        // Port Cat.3c: runtime config defaults to absent for v2 brains;
        // the load path then falls back to the legacy sidecars.
        runtime_config: Default::default(),
        runtime_config_present: false,
        };
        std::fs::write(&path, serde_json::to_vec_pretty(&file).unwrap()).unwrap();

        // First byte of a JSON-pretty file is `{`, not the v2 magic.
        let raw = std::fs::read(&path).unwrap();
        assert_eq!(raw[0], b'{');

        // Audit fix: load_or_new no longer reads v1; use the migration-only loader.
        let reloaded = Brain::load_v1_for_migration_only(&path).expect("v1 load");
        assert_eq!(reloaded.network.node_count(), b.network.node_count());
        assert_eq!(reloaded.network.bond_count(), b.network.bond_count());
    }

    #[test]
    fn v1_to_v2_migration_preserves_counts_and_content() {
        let dir = tempfile::tempdir().unwrap();
        let v1_path = dir.path().join("src.lbr");
        let v2_path = dir.path().join("dst.lbr");

        // Build a v1 file directly via serde_json so the test is
        // independent of the save path.
        let b = small_brain();
        let file = BrainFile {
            format_version: V1_JSON_VERSION,
            network: b.network.to_snapshot(),
            interest_profile: b.interest_profile.clone(),
            meta_reflection: b.meta_reflection.clone(),
            scar_log: b.scar_log.clone(),
            research_proposals: b.research_proposals.clone(),
            research_config: b.research_config.clone(),
            next_research_proposal_id: b.next_research_proposal_id,
            gap_seed_frequency: b.gap_seed_frequency.clone(),
            wonder_state: Default::default(),
            negation_overlay: crate::negation::NegationOverlay::new(),
            source_overlay: crate::source::SourceOverlay::new(),
            // Port T.4: pending log snapshot (default empty for pre-T.4 brains).
            pending_log: Default::default(),
            rejection_patterns: Default::default(),
        lb_session: Default::default(),
        cs_style: Default::default(),
        pack_ledger: Default::default(),
        cap_pack_ledger: Default::default(),
        // Port Cat.3c: runtime config defaults to absent for v2 brains;
        // the load path then falls back to the legacy sidecars.
        runtime_config: Default::default(),
        runtime_config_present: false,
        };
        std::fs::write(&v1_path, serde_json::to_vec_pretty(&file).unwrap()).unwrap();

        // Migrate via load_or_new + save_to_format.
        // Audit fix: load_or_new no longer reads v1; use the migration-only loader.
        let mut migrated = Brain::load_v1_for_migration_only(&v1_path).expect("load v1");
        migrated.save_to_format(&v2_path, BrainSaveFormat::V2Binary).expect("write v2");

        // Source must be untouched.
        let raw_src = std::fs::read(&v1_path).unwrap();
        assert_eq!(raw_src[0], b'{', "v1 source must remain JSON-pretty");

        // Destination must be v2 binary.
        let raw_dst = std::fs::read(&v2_path).unwrap();
        assert_eq!(&raw_dst[0..8], &MAGIC_HEADER);

        // Counts round-trip on reload.
        let reloaded = Brain::load_or_new(&v2_path).expect("load v2");
        assert_eq!(reloaded.network.node_count(), b.network.node_count());
        assert_eq!(reloaded.network.bond_count(), b.network.bond_count());
        let kf_src: usize = b.network.bonds().values().map(|x| x.keyframes().len()).sum();
        let kf_dst: usize = reloaded.network.bonds().values().map(|x| x.keyframes().len()).sum();
        assert_eq!(kf_src, kf_dst);
        assert_eq!(reloaded.scar_log.scars.len(), b.scar_log.scars.len());
    }

    #[test]
    fn v2_save_via_brain_with_format_flag_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("v2.lbr");

        let mut b = small_brain();
        b.path = Some(path.clone());
        b.save_format = BrainSaveFormat::V2Binary;
        b.save().expect("v2 save via Brain::save");

        let raw = std::fs::read(&path).unwrap();
        assert_eq!(&raw[0..8], &MAGIC_HEADER);

        let reloaded = Brain::load_or_new(&path).expect("v2 load");
        assert_eq!(reloaded.network.node_count(), b.network.node_count());
        assert_eq!(reloaded.network.bond_count(), b.network.bond_count());
    }

}
