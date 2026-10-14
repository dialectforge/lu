//! `nodeintern.bin` sidecar -- streamed length-prefixed records of
//! `(Seed, NodeId)` bindings, sorted by NodeId ascending.
//!
//! OP-BRAIN-V5 Phase 2.1 (Q3 / Q6 of the design doc). The sidecar is
//! the snapshot half of the "sidecar = snapshot, WAL = delta"
//! discipline (audit Section 4.3). On daemon boot we load the
//! sidecar verbatim, then replay WAL `REC_TYPE_NODE_INTERN` records
//! with their stored NodeIds (Phase 2.2 wires this side). The
//! sidecar removes the v4 quirk where post-restart NodeIds match the
//! WAL's stored values only by coincidence of replay order matching
//! original insertion order (current `brain.rs:9941-9967` comment).
//!
//! ## Wire format
//!
//! ```text
//! Header (16 B):
//!   magic        u32 LE  = 0x4E494E54  ("NINT" -- NodeIntern)
//!   format_ver   u16 LE  = SIDECAR_FORMAT_VERSION (=1)
//!   reserved     u16 LE  = 0
//!   record_count u64 LE  = N
//!
//! Records (variable length, repeated N times):
//!   seed         [u8; 32]    raw 32-byte seed (BLAKE3 of label)
//!   node_id      varint u64  unsigned LEB128 (Q1: low 32 bits load-bearing)
//!
//! Trailer (4 B):
//!   crc32        u32 LE      CRC-32 (IEEE polynomial) over header + records
//! ```
//!
//! Records are SORTED BY NodeId ASCENDING. The audit (S4.2) recommends
//! this because `NodeIntern.reverse: Vec<Seed>` is already in NodeId
//! order, so the natural emit order requires no sort key tracking --
//! BUT this commit cannot rely on that invariant (the audit assumes
//! the post-Phase-2.3 shape where reverse[] IS NodeId-indexed; today
//! reverse[] is *insertion*-indexed, which only coincides with NodeId
//! order when `intern_or_get` is the sole writer -- not when
//! `insert_with_id` is used for boot-replay restoration). Save
//! therefore collects the forward HashMap and sorts explicitly. See
//! the `save_to` doc-comment for the bounded-RSS note.
//!
//! The sidecar-absent fallback (load the pre-Phase-2 way via
//! `intern_or_get`) is owned by `Brain::enable_memtable` (Phase 2.2).
//! This module's responsibility ends at the file boundary.
//!
//! ## CRC scope
//!
//! CRC32 is computed over the 16-byte header AND every record byte
//! that follows, in write order. The trailer itself is NOT included.
//! Implementation: a single `crc32fast::Hasher` is updated as bytes
//! are written; the final digest is appended as 4 LE bytes. Load
//! reverses: a `Hasher` accumulates the bytes as they are consumed,
//! and the final value is compared against the trailer.
//!
//! ## Honest scope notes
//!
//! - `save_to` collects a `Vec<(NodeId, Seed)>` of `NodeIntern.len()`
//!   entries for the sort. At 100M nodes this is `100M * 40 B = ~4 GB`
//!   resident during write. The sort is in-place after the collect.
//!   This is NOT zero-overhead; the trade-off is documented on
//!   `NodeIntern::save_to` itself.
//! - `load_from` streams records and grows `reverse: Vec<Seed>` on
//!   demand. Peak load-side RSS is the final `NodeIntern` size plus
//!   a 64 KiB `BufReader`.
//! - The record-count upper bound on load (`MAX_RECORD_COUNT`) rejects
//!   structurally impossible files (header claiming > 1 billion
//!   entries -- 32 GB sidecar -- in a single load). Tunable later if
//!   AX41-NVMe headroom ever justifies it; the design doc's
//!   100M-nodes-at-target ceiling fits with 10x margin.

use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::time::Instant;

use crate::node_intern::NodeId;
use crate::seed::Seed;
use crate::encrypt::{xor_in_place, MasterKey};
use rand::RngCore;

/// Magic bytes at the start of every sidecar file ("NINT" -- NodeIntern).
/// Little-endian u32. Mismatch on load => [`SidecarError::MagicMismatch`].
pub const SIDECAR_MAGIC: u32 = 0x4E494E54;

/// Current sidecar wire-format version. Bumped on any breaking change
/// to the record encoding. Loader rejects unknown versions with
/// [`SidecarError::FormatVersionUnsupported`] -- forward compatibility
/// is NOT promised at the sidecar layer (the WAL + node-store SSTs
/// are the durable durable-state; the sidecar is rebuildable).
pub const SIDECAR_FORMAT_VERSION: u16 = 1;

/// Header size in bytes: 4 magic + 2 format_ver + 2 reserved + 8 record_count.
const HEADER_BYTES: usize = 16;

/// Trailer size in bytes: 4-byte CRC32 over header+records.
const TRAILER_BYTES: usize = 4;

/// Bytes per Seed.
const SEED_BYTES: usize = 32;

/// Maximum LEB128 varint length for u64 (10 * 7 = 70 bits >= 64).
const MAX_VARINT_U64_LEN: usize = 10;

/// Hard upper bound on the header-declared record count (1 billion).
/// Rejects structurally impossible files at the header boundary,
/// before any bytes-of-records are read. Sized to comfortably exceed
/// the design doc's 100M-node ceiling.
const MAX_RECORD_COUNT: u64 = 1_000_000_000;

/// BufReader / BufWriter capacity. Matches the std default; the
/// audit notes 64 KiB is sufficient for streamed-record workloads.
const BUF_CAPACITY: usize = 64 * 1024;

/// OP-BRAIN-V5 6.8: encrypted-capable sidecar format version. A v1
/// (plaintext) file keeps [`SIDECAR_FORMAT_VERSION`]; a file written
/// with a key uses this version, a 32-byte header carrying a flags
/// byte + 16-byte IV, and an XOR-keystream-encrypted records region.
pub const SIDECAR_FORMAT_VERSION_ENC: u16 = 2;

/// v2 header flags bit 0: the records region is XOR-keystream encrypted.
pub const SIDECAR_FLAG_ENCRYPTED: u8 = 0x01;

/// v2 header size: 4 magic + 2 ver + 1 flags + 1 reserved + 8 count + 16 iv.
const HEADER_V2_BYTES: usize = 32;

/// OP-BRAIN-V5 6.8: keystream index for the (single-buffer) records
/// region, derived from the per-file IV. Mirrors the SST/WAL index
/// derivation (splitmix64 multiplier + IV XOR-fold); the random IV
/// makes the keystream unique per sidecar file. The actual keystream
/// generator is the trusted `crate::encrypt::xor_in_place`.
fn derive_sidecar_keystream_index(iv: &[u8; 16]) -> u64 {
    const SIDECAR_KS_DOMAIN: u64 = 0x5344_4543_4152_3032; // "SDECAR02"
    let mut iv_mix: u64 = 0;
    for chunk in iv.chunks(8) {
        let mut tmp = [0u8; 8];
        tmp[..chunk.len()].copy_from_slice(chunk);
        iv_mix ^= u64::from_le_bytes(tmp);
    }
    SIDECAR_KS_DOMAIN.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ iv_mix
}

/// Append an unsigned-LEB128 varint to `out` (no hasher; used by the
/// 6.8 buffered encrypted writer).
fn append_varint_u64(out: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 {
        out.push(((v as u8) & 0x7F) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

/// Decode an unsigned-LEB128 varint from the front of `buf`; returns
/// `(value, bytes_consumed)`. Used by the 6.8 decrypted-buffer parse.
fn decode_varint_slice(buf: &[u8]) -> Result<(u64, usize), SidecarError> {
    let mut result: u64 = 0;
    let mut shift: u32 = 0;
    for i in 0..MAX_VARINT_U64_LEN {
        if i >= buf.len() {
            return Err(SidecarError::UnexpectedEof);
        }
        let byte = buf[i];
        let chunk = (byte & 0x7F) as u64;
        if i == MAX_VARINT_U64_LEN - 1 && byte > 0x01 {
            return Err(SidecarError::VarintOverflow);
        }
        match chunk.checked_shl(shift) {
            Some(sh) => result |= sh,
            None => return Err(SidecarError::VarintOverflow),
        }
        if byte & 0x80 == 0 {
            return Ok((result, i + 1));
        }
        shift += 7;
    }
    Err(SidecarError::VarintOverflow)
}

/// OP-BRAIN-V5 6.8: at-rest encryption status of a nodeintern sidecar,
/// used by the `--require-encrypted` guard.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SidecarEncStatus {
    /// File does not exist or is zero-length (nothing to leak).
    Absent,
    /// File exists but is not a 6.8 encrypted sidecar (v1 plaintext,
    /// or v2 with FLAG_ENCRYPTED clear).
    Plaintext,
    /// File exists as a v2 sidecar carrying FLAG_ENCRYPTED.
    Encrypted,
}

/// Inspect a sidecar's header and classify its at-rest encryption.
pub fn sidecar_encryption_status(path: &Path) -> Result<SidecarEncStatus, SidecarError> {
    let mut f = match File::open(path) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Ok(SidecarEncStatus::Absent)
        }
        Err(e) => return Err(SidecarError::Io(e)),
    };
    let mut hdr = [0u8; HEADER_V2_BYTES];
    let n = f.read(&mut hdr)?;
    if n == 0 {
        return Ok(SidecarEncStatus::Absent);
    }
    if n >= 6 {
        let magic = u32::from_le_bytes([hdr[0], hdr[1], hdr[2], hdr[3]]);
        let ver = u16::from_le_bytes([hdr[4], hdr[5]]);
        if magic == SIDECAR_MAGIC && ver == SIDECAR_FORMAT_VERSION_ENC
            && n >= 7 && hdr[6] & SIDECAR_FLAG_ENCRYPTED != 0
        {
            return Ok(SidecarEncStatus::Encrypted);
        }
    }
    Ok(SidecarEncStatus::Plaintext)
}

/// Sanity stats returned by [`crate::node_intern::NodeIntern::save_to`].
///
/// Logged by callers and surfaced in tests so a future audit can see
/// what the sidecar layer observed. `body_crc32` is the same CRC32
/// written to the trailer; emitting it lets diagnostic tools compare
/// two sidecar files without re-reading the trailer separately.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SidecarStats {
    /// Number of `(Seed, NodeId)` records written (== `NodeIntern::len()`).
    pub record_count: u64,
    /// On-disk file size in bytes (header + records + trailer).
    pub file_size_bytes: u64,
    /// CRC32 (IEEE polynomial) over header + records (NOT trailer).
    pub body_crc32: u32,
    /// Wall-clock elapsed for the save (collect + sort + stream).
    pub elapsed: std::time::Duration,
}

/// Errors surfaced by the sidecar I/O layer.
///
/// Hand-rolled `Display` + `error::Error` impls (matches the codebase
/// pattern in `WalError`, `BrainError`). The `thiserror` crate is not
/// in the workspace dep graph; pulling it in just for one enum was
/// rejected as scope creep -- the manual impls are <20 LOC and read
/// straight.
#[derive(Debug)]
pub enum SidecarError {
    /// `std::io::Error` propagated from file open / read / write / fsync.
    Io(std::io::Error),
    /// Header magic was not [`SIDECAR_MAGIC`]. File is not a node-intern
    /// sidecar (or is corrupted in the first 4 bytes).
    MagicMismatch { observed: u32 },
    /// Header `format_ver` is not [`SIDECAR_FORMAT_VERSION`]. The
    /// loader does not support cross-version reads.
    FormatVersionUnsupported { observed: u16 },
    /// Trailer CRC32 did not match the CRC computed over header +
    /// records. The file is corrupted; callers should fall back to
    /// the sidecar-absent boot path (per audit S4.3).
    CrcMismatch { expected: u32, observed: u32 },
    /// Header-declared record count exceeded [`MAX_RECORD_COUNT`]. The
    /// file is either corrupted or the wire format was changed without
    /// bumping the version. Guards against corruption-induced OOM.
    RecordCountImplausible { observed: u64, max: u64 },
    /// LEB128 varint at the tail of a record overflowed u64 or ran
    /// past EOF. Indicates record-level corruption.
    VarintOverflow,
    /// Record stream ended before the header-declared `record_count`
    /// was reached. Sidecar truncated mid-record.
    UnexpectedEof,
    /// [`crate::node_intern::NodeIntern::insert_with_id`] surfaced a
    /// collision: the same `Seed` was inserted twice with different
    /// `NodeId` values. Indicates sidecar corruption or a misuse of
    /// `insert_with_id` outside the bulk-load contract.
    InsertWithIdCollision { seed: Seed, existing: NodeId, attempted: NodeId },
    /// OP-BRAIN-V5 6.8: the sidecar header marks the records region
    /// encrypted but no master key was supplied to decrypt it.
    EncryptedButNoKey,
}

impl std::fmt::Display for SidecarError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SidecarError::Io(e) => write!(f, "sidecar IO: {}", e),
            SidecarError::MagicMismatch { observed } => write!(
                f,
                "sidecar magic mismatch: expected {:#x}, observed {:#x}",
                SIDECAR_MAGIC, observed
            ),
            SidecarError::FormatVersionUnsupported { observed } => write!(
                f,
                "sidecar format version unsupported: expected {}, observed {}",
                SIDECAR_FORMAT_VERSION, observed
            ),
            SidecarError::CrcMismatch { expected, observed } => write!(
                f,
                "sidecar CRC32 mismatch: expected {:#010x}, observed {:#010x}",
                expected, observed
            ),
            SidecarError::RecordCountImplausible { observed, max } => write!(
                f,
                "sidecar record_count {} exceeds upper bound {}",
                observed, max
            ),
            SidecarError::VarintOverflow => {
                write!(f, "sidecar varint overflow / truncated")
            }
            SidecarError::UnexpectedEof => write!(f, "sidecar truncated mid-record"),
            SidecarError::InsertWithIdCollision { seed, existing, attempted } => write!(
                f,
                "sidecar insert_with_id collision: seed {} already bound to NodeId({}), attempted NodeId({})",
                seed, existing.0, attempted.0
            ),
            SidecarError::EncryptedButNoKey => write!(
                f, "sidecar header marks the records region encrypted but no key was supplied"
            ),
        }
    }
}

impl std::error::Error for SidecarError {}

impl From<std::io::Error> for SidecarError {
    fn from(e: std::io::Error) -> Self { SidecarError::Io(e) }
}

/// Append unsigned-LEB128 (varint) encoding of `v` to `out`. 7 bits
/// per byte, MSB = continuation. The zero value still emits one byte.
/// Standard Protobuf wire-compatible. Mirrors the helper in `wal.rs`
/// (private to that module); duplicating ~10 LOC here avoids exposing
/// a shared utility surface before we know the codec actually
/// stabilises. If a third caller appears, extract to a `varint`
/// module then. This copy also updates a `crc32fast::Hasher` with
/// the written bytes so the body-CRC stream stays consistent with
/// the read side (the WAL has its own per-record CRC and doesn't
/// thread a hasher through the writer).
#[inline]
fn write_varint_u64<W: Write>(
    out: &mut W,
    hasher: &mut crc32fast::Hasher,
    mut v: u64,
) -> std::io::Result<usize> {
    let mut buf = [0u8; MAX_VARINT_U64_LEN];
    let mut n = 0usize;
    while v >= 0x80 {
        buf[n] = ((v as u8) & 0x7F) | 0x80;
        n += 1;
        v >>= 7;
    }
    buf[n] = v as u8;
    n += 1;
    out.write_all(&buf[..n])?;
    hasher.update(&buf[..n]);
    Ok(n)
}

/// Read an unsigned-LEB128 varint from a stream. Returns the decoded
/// `u64` and the number of bytes consumed (1..=10). Returns
/// [`SidecarError::VarintOverflow`] on >10 bytes or u64 overflow,
/// [`SidecarError::UnexpectedEof`] on premature end of stream.
#[inline]
fn read_varint_u64<R: Read>(r: &mut R, hasher: &mut crc32fast::Hasher) -> Result<(u64, usize), SidecarError> {
    let mut result: u64 = 0;
    let mut shift: u32 = 0;
    let mut buf = [0u8; 1];
    for i in 0..MAX_VARINT_U64_LEN {
        match r.read(&mut buf)? {
            0 => return Err(SidecarError::UnexpectedEof),
            _ => {}
        }
        hasher.update(&buf);
        let byte = buf[0];
        let chunk = (byte & 0x7F) as u64;
        if i == MAX_VARINT_U64_LEN - 1 && byte > 0x01 {
            // 10th byte may carry only 1 payload bit before u64 overflows.
            return Err(SidecarError::VarintOverflow);
        }
        if let Some(shifted) = chunk.checked_shl(shift) {
            result |= shifted;
        } else {
            return Err(SidecarError::VarintOverflow);
        }
        if (byte & 0x80) == 0 {
            return Ok((result, i + 1));
        }
        shift += 7;
    }
    // Loop exited without a terminating byte -- 10 continuation bytes.
    Err(SidecarError::VarintOverflow)
}

/// Write the 16-byte sidecar header. The header is fed into the same
/// `crc32fast::Hasher` that will accumulate the records' bytes.
fn write_header<W: Write>(
    w: &mut W,
    hasher: &mut crc32fast::Hasher,
    record_count: u64,
) -> std::io::Result<()> {
    let mut buf = [0u8; HEADER_BYTES];
    buf[0..4].copy_from_slice(&SIDECAR_MAGIC.to_le_bytes());
    buf[4..6].copy_from_slice(&SIDECAR_FORMAT_VERSION.to_le_bytes());
    // bytes 6..8 reserved = 0 (already zeroed)
    buf[8..16].copy_from_slice(&record_count.to_le_bytes());
    w.write_all(&buf)?;
    hasher.update(&buf);
    Ok(())
}

/// Read and validate the header. Returns the declared `record_count`
/// on success. Updates `hasher` so the body CRC remains in sync.
#[allow(dead_code)]
fn read_header<R: Read>(
    r: &mut R,
    hasher: &mut crc32fast::Hasher,
) -> Result<u64, SidecarError> {
    let mut buf = [0u8; HEADER_BYTES];
    r.read_exact(&mut buf)?;
    hasher.update(&buf);

    let magic = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
    if magic != SIDECAR_MAGIC {
        return Err(SidecarError::MagicMismatch { observed: magic });
    }
    let format_ver = u16::from_le_bytes([buf[4], buf[5]]);
    if format_ver != SIDECAR_FORMAT_VERSION {
        return Err(SidecarError::FormatVersionUnsupported { observed: format_ver });
    }
    // bytes 6..8 reserved -- accept any value (future-proofing window)
    let record_count = u64::from_le_bytes([
        buf[8], buf[9], buf[10], buf[11], buf[12], buf[13], buf[14], buf[15],
    ]);
    if record_count > MAX_RECORD_COUNT {
        return Err(SidecarError::RecordCountImplausible {
            observed: record_count,
            max: MAX_RECORD_COUNT,
        });
    }
    Ok(record_count)
}

/// Write one record (`seed` + `node_id`). Updates `hasher` with the
/// bytes written, so the body CRC stays in sync. Returns the number
/// of bytes written (32 + varint length).
fn write_record<W: Write>(
    w: &mut W,
    hasher: &mut crc32fast::Hasher,
    seed: &Seed,
    node_id: NodeId,
) -> std::io::Result<usize> {
    w.write_all(seed.as_bytes())?;
    hasher.update(seed.as_bytes());
    let varint_len = write_varint_u64(w, hasher, node_id.0)?;
    Ok(SEED_BYTES + varint_len)
}

/// Read one record (`seed` + `node_id`). Updates `hasher` with the
/// bytes consumed.
fn read_record<R: Read>(
    r: &mut R,
    hasher: &mut crc32fast::Hasher,
) -> Result<(Seed, NodeId), SidecarError> {
    let mut seed_bytes = [0u8; SEED_BYTES];
    r.read_exact(&mut seed_bytes)?;
    hasher.update(&seed_bytes);
    let (raw_id, _consumed) = read_varint_u64(r, hasher)?;
    Ok((Seed::from_bytes(seed_bytes), NodeId(raw_id)))
}

/// Stream a sorted sequence of `(NodeId, Seed)` records to `path` as
/// a sidecar file. Caller is responsible for the sort -- this fn
/// assumes the iterator is already in NodeId-ascending order (the
/// caller has more context to enforce / verify that invariant
/// cheaply). On Ok returns sanity stats (record count + size + crc +
/// elapsed).
///
/// The file is created with `OpenOptions::create(true).truncate(true)`,
/// streamed through a [`BufWriter`], and fsync'd before close to make
/// the durability story honest. Atomicity is the caller's job (they
/// can write to `path.tmp` and rename); this fn is the I/O primitive.
pub(crate) fn write_sidecar<I>(
    path: &Path,
    records_sorted_by_node_id: I,
    record_count: u64,
    key: Option<MasterKey>,
) -> Result<SidecarStats, SidecarError>
where
    I: IntoIterator<Item = (NodeId, Seed)>,
{
    let started = Instant::now();
    let f = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(path)?;
    let mut w = BufWriter::with_capacity(BUF_CAPACITY, f);

    match key {
        None => {
            // ---- pre-6.8 plaintext v1 path (byte-identical) ----
            let mut hasher = crc32fast::Hasher::new();
            write_header(&mut w, &mut hasher, record_count)?;
            let mut bytes_written: u64 = HEADER_BYTES as u64;
            let mut observed_count: u64 = 0;
            for (node_id, seed) in records_sorted_by_node_id {
                let n = write_record(&mut w, &mut hasher, &seed, node_id)?;
                bytes_written += n as u64;
                observed_count += 1;
            }
            debug_assert_eq!(observed_count, record_count, "declared record_count != emitted");
            let body_crc32 = hasher.finalize();
            w.write_all(&body_crc32.to_le_bytes())?;
            bytes_written += TRAILER_BYTES as u64;
            let mut f = w.into_inner().map_err(|e| SidecarError::Io(e.into_error()))?;
            f.flush()?;
            f.sync_all()?;
            Ok(SidecarStats {
                record_count,
                file_size_bytes: bytes_written,
                body_crc32,
                elapsed: started.elapsed(),
            })
        }
        Some(k) => {
            // ---- OP-BRAIN-V5 6.8 encrypted v2 path ----
            // Materialise the records region, XOR-encrypt it whole
            // under one keystream (mirrors the SST per-block xor), then
            // write a v2 header (carrying the IV) + ciphertext + CRC.
            // The buffer is bounded by NodeIntern::len() -- the same
            // ceiling the sort already pays; documented on save_to.
            let mut iv = [0u8; 16];
            rand::thread_rng().fill_bytes(&mut iv);
            let mut records_buf: Vec<u8> = Vec::new();
            let mut observed_count: u64 = 0;
            for (node_id, seed) in records_sorted_by_node_id {
                records_buf.extend_from_slice(seed.as_bytes());
                append_varint_u64(&mut records_buf, node_id.0);
                observed_count += 1;
            }
            debug_assert_eq!(observed_count, record_count, "declared record_count != emitted");

            let mut header = [0u8; HEADER_V2_BYTES];
            header[0..4].copy_from_slice(&SIDECAR_MAGIC.to_le_bytes());
            header[4..6].copy_from_slice(&SIDECAR_FORMAT_VERSION_ENC.to_le_bytes());
            header[6] = SIDECAR_FLAG_ENCRYPTED;
            // byte 7 reserved = 0
            header[8..16].copy_from_slice(&record_count.to_le_bytes());
            header[16..32].copy_from_slice(&iv);

            // CRC over header || PLAINTEXT records, computed BEFORE
            // encryption. Load decrypts then re-checks this CRC, so a
            // wrong key (garbage plaintext) fails cleanly with
            // CrcMismatch instead of being parsed into junk NodeIds.
            let mut hasher = crc32fast::Hasher::new();
            hasher.update(&header);
            hasher.update(&records_buf);
            let body_crc32 = hasher.finalize();

            let ksi = derive_sidecar_keystream_index(&iv);
            xor_in_place(&k, ksi, &mut records_buf);

            w.write_all(&header)?;
            w.write_all(&records_buf)?;
            w.write_all(&body_crc32.to_le_bytes())?;
            let file_size =
                HEADER_V2_BYTES as u64 + records_buf.len() as u64 + TRAILER_BYTES as u64;
            let mut f = w.into_inner().map_err(|e| SidecarError::Io(e.into_error()))?;
            f.flush()?;
            f.sync_all()?;
            Ok(SidecarStats {
                record_count,
                file_size_bytes: file_size,
                body_crc32,
                elapsed: started.elapsed(),
            })
        }
    }
}

/// Stream sidecar records back. Calls `accept(seed, node_id)` for
/// each record in file order; the caller threads them into a fresh
/// `NodeIntern` via `insert_with_id`. Validates header + CRC32 +
/// record count; any mismatch surfaces as a typed [`SidecarError`].
///
/// The CRC32 over header + records is validated at the END of the
/// stream against the 4-byte trailer. A corrupted body is only
/// reported AFTER the records have been delivered to `accept`; the
/// caller's `NodeIntern` is therefore not guaranteed to be
/// well-formed on `Err` return. This matches the audit's
/// recommendation (S4.3): a CRC-failed sidecar is treated as absent,
/// and the caller falls back to the WAL-replay path with a fresh
/// `NodeIntern`.
pub(crate) fn read_sidecar<F>(
    path: &Path,
    key: Option<MasterKey>,
    mut accept: F,
) -> Result<u64, SidecarError>
where
    F: FnMut(Seed, NodeId) -> Result<(), SidecarError>,
{
    let f = File::open(path)?;
    let mut r = BufReader::with_capacity(BUF_CAPACITY, f);

    // Read the 16-byte common header prefix. v1 (plaintext) and v2
    // (encrypted-capable) share byte positions 0..16; v2 carries an
    // extra 16-byte IV afterwards.
    let mut hdr16 = [0u8; HEADER_BYTES];
    r.read_exact(&mut hdr16)?;
    let magic = u32::from_le_bytes([hdr16[0], hdr16[1], hdr16[2], hdr16[3]]);
    if magic != SIDECAR_MAGIC {
        return Err(SidecarError::MagicMismatch { observed: magic });
    }
    let ver = u16::from_le_bytes([hdr16[4], hdr16[5]]);
    let record_count = u64::from_le_bytes([
        hdr16[8], hdr16[9], hdr16[10], hdr16[11], hdr16[12], hdr16[13], hdr16[14], hdr16[15],
    ]);
    if record_count > MAX_RECORD_COUNT {
        return Err(SidecarError::RecordCountImplausible {
            observed: record_count,
            max: MAX_RECORD_COUNT,
        });
    }

    if ver == SIDECAR_FORMAT_VERSION {
        // ---- pre-6.8 plaintext v1 streaming path ----
        let mut hasher = crc32fast::Hasher::new();
        hasher.update(&hdr16);
        for _ in 0..record_count {
            let (seed, node_id) = read_record(&mut r, &mut hasher)?;
            accept(seed, node_id)?;
        }
        let mut trailer = [0u8; TRAILER_BYTES];
        r.read_exact(&mut trailer)?;
        let expected = hasher.finalize();
        let observed = u32::from_le_bytes(trailer);
        if observed != expected {
            return Err(SidecarError::CrcMismatch { expected, observed });
        }
        Ok(record_count)
    } else if ver == SIDECAR_FORMAT_VERSION_ENC {
        // ---- OP-BRAIN-V5 6.8 encrypted-capable v2 path ----
        let flags = hdr16[6];
        let mut iv = [0u8; 16];
        r.read_exact(&mut iv)?;
        // Remaining bytes = ciphertext records region + 4-byte trailer.
        let mut remainder: Vec<u8> = Vec::new();
        r.read_to_end(&mut remainder)?;
        if remainder.len() < TRAILER_BYTES {
            return Err(SidecarError::UnexpectedEof);
        }
        let split = remainder.len() - TRAILER_BYTES;
        let observed = u32::from_le_bytes([
            remainder[split], remainder[split + 1], remainder[split + 2], remainder[split + 3],
        ]);
        let mut body = remainder[..split].to_vec();

        // Decrypt FIRST (if encrypted), then verify the CRC over
        // header (incl. IV) || PLAINTEXT records. A wrong key yields
        // garbage plaintext -> CRC mismatch -> clean CrcMismatch error
        // (rather than parsing junk NodeIds and OOMing on resize).
        if flags & SIDECAR_FLAG_ENCRYPTED != 0 {
            let k = key.ok_or(SidecarError::EncryptedButNoKey)?;
            let ksi = derive_sidecar_keystream_index(&iv);
            xor_in_place(&k, ksi, &mut body);
        }
        let mut header = [0u8; HEADER_V2_BYTES];
        header[..HEADER_BYTES].copy_from_slice(&hdr16);
        header[HEADER_BYTES..].copy_from_slice(&iv);
        let mut hasher = crc32fast::Hasher::new();
        hasher.update(&header);
        hasher.update(&body);
        let expected = hasher.finalize();
        if observed != expected {
            return Err(SidecarError::CrcMismatch { expected, observed });
        }

        // Parse records from the (now plaintext) buffer.
        let mut cur: &[u8] = &body;
        for _ in 0..record_count {
            if cur.len() < SEED_BYTES {
                return Err(SidecarError::UnexpectedEof);
            }
            let mut seed_bytes = [0u8; SEED_BYTES];
            seed_bytes.copy_from_slice(&cur[..SEED_BYTES]);
            cur = &cur[SEED_BYTES..];
            let (raw_id, consumed) = decode_varint_slice(cur)?;
            cur = &cur[consumed..];
            accept(Seed::from_bytes(seed_bytes), NodeId(raw_id))?;
        }
        Ok(record_count)
    } else {
        Err(SidecarError::FormatVersionUnsupported { observed: ver })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node_intern::NodeIntern;
    // (std::io::Write is already in scope at module level)

    /// Helper: a TempDir-scoped path for one test file.
    fn tmp_sidecar(name: &str) -> (tempfile::TempDir, std::path::PathBuf) {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join(name);
        (tmp, p)
    }

    /// A. Empty roundtrip: save an empty NodeIntern, load it back.
    /// File body is just the 16-byte header; CRC over the header is
    /// well-defined (crc32fast over a fixed 16-byte stream).
    #[test]
    fn sidecar_empty_roundtrip() {
        let intern = NodeIntern::new();
        let (_dir, p) = tmp_sidecar("empty.bin");
        let stats = intern.save_to(&p).unwrap();
        assert_eq!(stats.record_count, 0);
        assert_eq!(stats.file_size_bytes, (HEADER_BYTES + TRAILER_BYTES) as u64);

        let loaded = NodeIntern::load_from(&p).unwrap();
        assert_eq!(loaded.len(), 0);
        assert!(loaded.is_empty());
    }

    /// B. Single-entry roundtrip: one Seed, NodeId(42). The reverse
    /// Vec is sparse from 0..42 (filled with sentinel zeros).
    #[test]
    fn sidecar_single_entry_roundtrip() {
        let intern = NodeIntern::new();
        let s = Seed::from_label("answer");
        intern.insert_with_id(s, NodeId(42)).unwrap();

        let (_dir, p) = tmp_sidecar("one.bin");
        let stats = intern.save_to(&p).unwrap();
        assert_eq!(stats.record_count, 1);

        let loaded = NodeIntern::load_from(&p).unwrap();
        assert_eq!(loaded.get(&s), Some(NodeId(42)));
        assert_eq!(loaded.reverse_lookup(NodeId(42)), Some(s));
        // Sparse Vec must report len() = 43 (indices 0..=42).
        assert_eq!(loaded.len(), 43);
    }

    /// C. N=100K entries roundtrip. Random Seeds, sequential NodeIds.
    /// Validates: (i) save returns matching stats; (ii) load returns
    /// a NodeIntern with all 100K mappings; (iii) file size matches
    /// the wire-format math.
    #[test]
    fn sidecar_100k_roundtrip() {
        const N: u64 = 100_000;
        let intern = NodeIntern::new();
        let mut seeds: Vec<Seed> = Vec::with_capacity(N as usize);
        for i in 0..N {
            let s = Seed::from_label(&format!("node_{}", i));
            seeds.push(s);
            intern.insert_with_id(s, NodeId(i)).unwrap();
        }

        let (_dir, p) = tmp_sidecar("100k.bin");
        let stats = intern.save_to(&p).unwrap();
        assert_eq!(stats.record_count, N);

        // Wire-format math:
        // - Each NodeId < 128 takes 1 varint byte.
        // - NodeId in [128, 16384) takes 2 varint bytes.
        // - NodeId in [16384, 2097152) takes 3 varint bytes.
        // For N=100_000: 128 records at 1B + (16384-128) at 2B + (100000-16384) at 3B varint.
        let v1 = 128u64;
        let v2 = 16384u64 - 128;
        let v3 = N - 16384;
        let varint_total = v1 * 1 + v2 * 2 + v3 * 3;
        let expected_size =
            HEADER_BYTES as u64 + N * SEED_BYTES as u64 + varint_total + TRAILER_BYTES as u64;
        assert_eq!(
            stats.file_size_bytes, expected_size,
            "file size {} != expected {}",
            stats.file_size_bytes, expected_size
        );

        let loaded = NodeIntern::load_from(&p).unwrap();
        assert_eq!(loaded.len(), N as usize);
        for (i, s) in seeds.iter().enumerate() {
            assert_eq!(loaded.get(s), Some(NodeId(i as u64)));
            assert_eq!(loaded.reverse_lookup(NodeId(i as u64)), Some(*s));
        }
    }

    /// D. After save, NodeIds in the on-disk file are monotonically
    /// increasing. Reads the file directly, decodes records, asserts
    /// the sort invariant. Probes the format AND the sort path.
    #[test]
    fn sidecar_sorted_by_node_id_invariant() {
        let intern = NodeIntern::new();
        // Insert in NON-sequential order so the sort actually has work.
        let pairs: Vec<(NodeId, Seed)> = vec![
            (NodeId(7), Seed::from_label("seven")),
            (NodeId(3), Seed::from_label("three")),
            (NodeId(99), Seed::from_label("ninety-nine")),
            (NodeId(1), Seed::from_label("one")),
            (NodeId(42), Seed::from_label("answer")),
        ];
        for (id, s) in &pairs {
            intern.insert_with_id(*s, *id).unwrap();
        }

        let (_dir, p) = tmp_sidecar("sorted.bin");
        let _stats = intern.save_to(&p).unwrap();

        // Read file, decode records, assert sort.
        let bytes = std::fs::read(&p).unwrap();
        // Skip 16-byte header.
        let mut cursor = &bytes[HEADER_BYTES..bytes.len() - TRAILER_BYTES];
        let mut last_id: Option<u64> = None;
        let mut seen: Vec<u64> = Vec::new();
        while !cursor.is_empty() {
            // 32-byte seed + varint node_id.
            cursor = &cursor[SEED_BYTES..];
            // Decode varint inline.
            let mut val: u64 = 0;
            let mut shift: u32 = 0;
            let mut consumed = 0usize;
            for (i, b) in cursor.iter().enumerate() {
                let chunk = (*b & 0x7F) as u64;
                val |= chunk << shift;
                consumed = i + 1;
                if (*b & 0x80) == 0 {
                    break;
                }
                shift += 7;
            }
            cursor = &cursor[consumed..];
            if let Some(prev) = last_id {
                assert!(val > prev, "NodeIds not monotonic: {} then {}", prev, val);
            }
            last_id = Some(val);
            seen.push(val);
        }
        assert_eq!(seen, vec![1, 3, 7, 42, 99]);
    }

    /// E. CRC mismatch is rejected: save a file, corrupt one byte in
    /// the middle of the body, attempt to load -- must error with
    /// [`SidecarError::CrcMismatch`].
    #[test]
    fn sidecar_crc_mismatch_rejected() {
        let intern = NodeIntern::new();
        for i in 0..10u64 {
            intern
                .insert_with_id(Seed::from_label(&format!("n_{}", i)), NodeId(i))
                .unwrap();
        }
        let (_dir, p) = tmp_sidecar("corrupted.bin");
        intern.save_to(&p).unwrap();

        // Corrupt one byte well inside the body (skip header).
        let mut bytes = std::fs::read(&p).unwrap();
        let target = HEADER_BYTES + 8; // 8 bytes into the first seed
        bytes[target] ^= 0xFF;
        let mut f = std::fs::OpenOptions::new()
            .write(true)
            .truncate(true)
            .open(&p)
            .unwrap();
        f.write_all(&bytes).unwrap();
        f.sync_all().unwrap();
        drop(f);

        let res = NodeIntern::load_from(&p);
        match res {
            Err(SidecarError::CrcMismatch { .. }) => {}
            Err(other) => panic!("expected CrcMismatch, got {:?}", other),
            Ok(_) => panic!("expected CrcMismatch, got Ok"),
        }
    }

    /// F. `insert_with_id` collision is rejected: same Seed, two
    /// different NodeIds.
    #[test]
    fn insert_with_id_collision_rejected() {
        let intern = NodeIntern::new();
        let s = Seed::from_label("collide");
        intern.insert_with_id(s, NodeId(1)).unwrap();
        let err = intern.insert_with_id(s, NodeId(2)).unwrap_err();
        match err {
            SidecarError::InsertWithIdCollision { existing, attempted, .. } => {
                assert_eq!(existing, NodeId(1));
                assert_eq!(attempted, NodeId(2));
            }
            other => panic!("expected InsertWithIdCollision, got {:?}", other),
        }
        // Idempotent re-insertion with the SAME NodeId must succeed.
        intern.insert_with_id(s, NodeId(1)).unwrap();
    }

    /// G. `insert_with_id` out-of-order is OK: insert NodeId(7) then
    /// NodeId(3) for different seeds. Sparse Vec must correctly
    /// resolve both NodeIds back to their Seeds.
    #[test]
    fn insert_with_id_out_of_order_resolves_both() {
        let intern = NodeIntern::new();
        let s7 = Seed::from_label("seven");
        let s3 = Seed::from_label("three");
        intern.insert_with_id(s7, NodeId(7)).unwrap();
        intern.insert_with_id(s3, NodeId(3)).unwrap();

        assert_eq!(intern.get(&s7), Some(NodeId(7)));
        assert_eq!(intern.get(&s3), Some(NodeId(3)));
        assert_eq!(intern.reverse_lookup(NodeId(7)), Some(s7));
        assert_eq!(intern.reverse_lookup(NodeId(3)), Some(s3));
        // Index between (3, 7), e.g. 5, is a sentinel hole and must
        // NOT alias either real seed.
        let hole = intern.reverse_lookup(NodeId(5)).unwrap();
        assert_ne!(hole, s7);
        assert_ne!(hole, s3);
        // len() reflects the largest-NodeId + 1 (sparse-Vec convention).
        assert_eq!(intern.len(), 8);
    }

    // ---------------------------------------------------------------
    // OP-BRAIN-V5 6.8 -- encrypted nodeintern sidecar at rest.
    // ---------------------------------------------------------------

    /// (d) Encrypted sidecar round-trips and leaks no plaintext seed.
    #[test]
    fn enc68_sidecar_encrypted_round_trip() {
        let intern = NodeIntern::new();
        let mut seeds = Vec::new();
        for i in 0..200u64 {
            let s = Seed::from_label(&format!("enc_node_{}", i));
            seeds.push(s);
            intern.insert_with_id(s, NodeId(i)).unwrap();
        }
        let (_dir, p) = tmp_sidecar("enc.bin");
        let key = crate::encrypt::MasterKey::from_bytes([0x6Bu8; 32]);
        let stats = intern.save_to_with_key(&p, Some(key.clone())).unwrap();
        assert_eq!(stats.record_count, 200);
        assert_eq!(sidecar_encryption_status(&p).unwrap(), SidecarEncStatus::Encrypted);
        let raw = std::fs::read(&p).unwrap();
        assert!(!raw.windows(32).any(|w| w == seeds[0].as_bytes()),
            "encrypted sidecar must not leak a plaintext seed on disk");
        let loaded = NodeIntern::load_from_with_key(&p, Some(key.clone())).unwrap();
        for (i, s) in seeds.iter().enumerate() {
            assert_eq!(loaded.get(s), Some(NodeId(i as u64)));
        }
    }

    /// (c-sidecar) Wrong / no key fails cleanly (no panic, no OOM) on the
    /// sidecar.
    #[test]
    fn enc68_sidecar_wrong_or_no_key_fails_cleanly() {
        let intern = NodeIntern::new();
        for i in 0..50u64 {
            intern.insert_with_id(Seed::from_label(&format!("wk_{}", i)), NodeId(i)).unwrap();
        }
        let (_dir, p) = tmp_sidecar("wrongkey.bin");
        let key = crate::encrypt::MasterKey::from_bytes([0x33u8; 32]);
        intern.save_to_with_key(&p, Some(key.clone())).unwrap();
        // No key -> clean EncryptedButNoKey.
        match NodeIntern::load_from_with_key(&p, None) {
            Err(SidecarError::EncryptedButNoKey) => {}
            Err(e) => panic!("no-key load must error EncryptedButNoKey, got {:?}", e),
            Ok(_) => panic!("no-key load must fail, got Ok"),
        }
        // Wrong key -> decrypt garbage -> plaintext CRC mismatch (NOT an
        // OOM from junk NodeIds resizing the reverse Vec).
        let wrong = crate::encrypt::MasterKey::from_bytes([0x44u8; 32]);
        match NodeIntern::load_from_with_key(&p, Some(wrong)) {
            Err(SidecarError::CrcMismatch { .. }) => {}
            Err(e) => panic!("wrong-key load must error CrcMismatch, got {:?}", e),
            Ok(_) => panic!("wrong-key load must fail, got Ok"),
        }
    }
}
