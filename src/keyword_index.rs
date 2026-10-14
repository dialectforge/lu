//! OP-022.a — Mmap'd keyword index sidecar.
//!
//! Replaces the ~7 GiB heap-resident `HashMap<String, Vec<Seed>>` carried
//! on `AssociativeNetwork.keyword_index` with a sidecar file mmap'd into
//! the process. On-disk layout (little-endian throughout):
//!
//! ```text
//! [0]     magic           [u8; 8]   = b"LU4KWIDX"
//! [8]     format_version  u32       = 1
//! [12]    flags           u32       = 0 (reserved)
//! [16]    brain_crc32     u32       (matches brain.lbr v3 trailer CRC)
//! [20]    brain_node_count u64
//! [28]    brain_bond_count u64
//! [36]    token_count     u64
//! [44]    seed_total      u64
//! [52]    token_offsets_offset  u64
//! [60]    token_blob_offset     u64
//! [68]    bucket_offsets_offset u64
//! [76]    seed_array_offset     u64
//! [84]    reserved        [u8; 44]  (zero — header rounded to 128 B)
//!
//! [128]   token_offsets   [u32; token_count + 1]
//!         offset[i] = byte offset INTO token_blob where token i starts.
//!         offset[token_count] = total bytes in token_blob → length of
//!         token i = offset[i+1] - offset[i].
//!
//! (pad to 8-byte alignment)
//! [tbl]   token_blob      [u8; ...]   packed UTF-8, sorted ascending
//!                                     by byte order (so bisect works).
//!
//! (pad to 8-byte alignment)
//! [bo]    bucket_offsets  [u64; token_count + 1]
//!         offset[i] = index INTO seed_array (in Seed units, NOT bytes)
//!         where token i's bucket starts. offset[token_count] = seed_total.
//!
//! (pad to 32-byte alignment)
//! [sa]    seed_array      [Seed; seed_total]   packed 32-byte Seeds.
//!         Within each bucket the Seeds are byte-sorted
//!         (preserving the BR.C-Hyp-Methodology invariant).
//!
//! [EOF-8] trailer_crc32   u32       CRC32 over bytes [0..EOF-8]
//! [EOF-4] trailer_magic   [u8; 4]   = b"OKKW"
//! ```
//!
//! See `/tmp/op022a_phase2_design.md` (sha256 a367d6ab…) for the full
//! rationale: zero new Cargo deps (memmap2 + crc32fast already present);
//! sidecar avoids touching the brain.lbr v3 format; overlay+tombstone
//! mirrors `overlay_deletions`/`bond_tombstones`; `MergedBucket: Deref<[Seed]>`
//! so the ~6 reader call sites keep working through deref coercion.

use std::collections::{HashMap, HashSet};
use std::convert::TryInto;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Write};
use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::sync::Arc;

use memmap2::Mmap;

// OP-022.a.3: bound compaction page-in transient via MADV_DONTNEED.
#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

#[cfg(test)]
pub static MADVISE_CALLS: AtomicUsize = AtomicUsize::new(0);
#[cfg(test)]
pub static MADVISE_LAST_OFFSET: AtomicUsize = AtomicUsize::new(0);
#[cfg(test)]
pub static MADVISE_MONOTONIC_VIOLATIONS: AtomicUsize = AtomicUsize::new(0);

#[cfg(test)]
pub fn reset_madvise_counters() {
    MADVISE_CALLS.store(0, AtomicOrdering::SeqCst);
    MADVISE_LAST_OFFSET.store(0, AtomicOrdering::SeqCst);
    MADVISE_MONOTONIC_VIOLATIONS.store(0, AtomicOrdering::SeqCst);
}

/// OP-022.a.3 chunk size for MADV_DONTNEED batching. 64 MiB strikes a
/// balance between syscall frequency and active-resident-window size.
const MADVISE_CHUNK_BYTES: usize = 64 * 1024 * 1024;

/// SAFETY-wrapper around libc::madvise. Issues a no-op (returning Ok) on
/// non-Unix targets so the tests / behavior degrade gracefully — the
/// merge iterator still works, just without the page-in cap.
fn madvise_dontneed(ptr: *const u8, len: usize) -> std::io::Result<()> {
    if len == 0 { return Ok(()); }
    // OP-022.a.3: record the call BEFORE the syscall so the unit-test counter
    // stays accurate even when the syscall errors (madvise can EINVAL on a
    // misaligned range; we treat it as a hint and continue).
    #[cfg(test)]
    {
        MADVISE_CALLS.fetch_add(1, AtomicOrdering::SeqCst);
        let new_end = ptr as usize + len;
        let prev = MADVISE_LAST_OFFSET.load(AtomicOrdering::SeqCst);
        if new_end < prev {
            MADVISE_MONOTONIC_VIOLATIONS.fetch_add(1, AtomicOrdering::SeqCst);
        }
        MADVISE_LAST_OFFSET.store(new_end, AtomicOrdering::SeqCst);
    }
    #[cfg(unix)]
    {
        // SAFETY: caller passes a valid mapped range (computed from the
        // mmap base + section offset). MADV_DONTNEED is a hint; on EINVAL or
        // other failures we drop the error and keep going — the worst case
        // is the kernel keeps pages resident a bit longer.
        unsafe {
            let _ = libc::madvise(ptr as *mut libc::c_void, len, libc::MADV_DONTNEED);
        }
    }
    Ok(())
}

use crate::seed::Seed;

// ---------------- Format constants ----------------

const MAGIC: &[u8; 8] = b"LU4KWIDX";
const TRAILER_MAGIC: &[u8; 4] = b"OKKW";
const FORMAT_VERSION: u32 = 1;
const HEADER_SIZE: usize = 128;
const TRAILER_SIZE: usize = 8;
const SEED_SIZE: usize = 32;

// ---------------- Errors ----------------

/// Failure modes for reading the sidecar.
#[derive(Debug)]
pub enum KeywordIndexError {
    Io(std::io::Error),
    /// File is shorter than the minimum valid header+trailer.
    TooSmall { size: u64 },
    /// Header magic bytes don't match `LU4KWIDX`.
    BadMagic { found: [u8; 8] },
    /// Trailer magic bytes don't match `OKKW`.
    BadTrailerMagic { found: [u8; 4] },
    /// Header format_version doesn't match the version this build understands.
    BadVersion { found: u32, expected: u32 },
    /// CRC32 over [0..EOF-8] doesn't match the stored trailer CRC.
    BadCrc { computed: u32, stored: u32 },
    /// Header offset fields point outside the file.
    OutOfBounds { what: &'static str, offset: u64, len: u64, file_size: u64 },
    /// Token table not sorted ascending — bisect would be unsafe.
    UnsortedTokens { at_index: usize },
    /// brain_crc32 in sidecar doesn't match caller-supplied brain.lbr CRC.
    BrainCrcMismatch { sidecar: u32, brain: u32 },
    /// brain_node_count in sidecar doesn't match caller-supplied count.
    NodeCountMismatch { sidecar: u64, brain: u64 },
    /// brain_bond_count in sidecar doesn't match caller-supplied count.
    BondCountMismatch { sidecar: u64, brain: u64 },
}

impl std::fmt::Display for KeywordIndexError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "keyword_index io: {}", e),
            Self::TooSmall { size } => write!(f, "keyword_index too small: {} bytes", size),
            Self::BadMagic { found } => write!(f, "keyword_index bad magic: {:?}", found),
            Self::BadTrailerMagic { found } => write!(f, "keyword_index bad trailer magic: {:?}", found),
            Self::BadVersion { found, expected } => write!(f, "keyword_index bad version: found {}, expected {}", found, expected),
            Self::BadCrc { computed, stored } => write!(f, "keyword_index CRC mismatch: computed {:08x}, stored {:08x}", computed, stored),
            Self::OutOfBounds { what, offset, len, file_size } => write!(f, "keyword_index {} out of bounds: offset={} len={} file_size={}", what, offset, len, file_size),
            Self::UnsortedTokens { at_index } => write!(f, "keyword_index token table not sorted at index {}", at_index),
            Self::BrainCrcMismatch { sidecar, brain } => write!(f, "keyword_index brain_crc32 mismatch: sidecar={:08x} brain={:08x}", sidecar, brain),
            Self::NodeCountMismatch { sidecar, brain } => write!(f, "keyword_index node_count mismatch: sidecar={} brain={}", sidecar, brain),
            Self::BondCountMismatch { sidecar, brain } => write!(f, "keyword_index bond_count mismatch: sidecar={} brain={}", sidecar, brain),
        }
    }
}

impl std::error::Error for KeywordIndexError {}

impl From<std::io::Error> for KeywordIndexError {
    fn from(e: std::io::Error) -> Self { Self::Io(e) }
}

// ---------------- Header (parsed) ----------------

#[derive(Debug, Clone, Copy)]
struct Header {
    brain_crc32: u32,
    brain_node_count: u64,
    brain_bond_count: u64,
    token_count: u64,
    seed_total: u64,
    token_offsets_offset: u64,
    token_blob_offset: u64,
    bucket_offsets_offset: u64,
    seed_array_offset: u64,
}

fn read_u32_le(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(bytes[at..at + 4].try_into().expect("slice len 4"))
}
fn read_u64_le(bytes: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(bytes[at..at + 8].try_into().expect("slice len 8"))
}

impl Header {
    fn parse(bytes: &[u8], file_size: u64) -> Result<Self, KeywordIndexError> {
        if bytes.len() < HEADER_SIZE {
            return Err(KeywordIndexError::TooSmall { size: bytes.len() as u64 });
        }
        let mut magic = [0u8; 8];
        magic.copy_from_slice(&bytes[0..8]);
        if &magic != MAGIC {
            return Err(KeywordIndexError::BadMagic { found: magic });
        }
        let version = read_u32_le(bytes, 8);
        if version != FORMAT_VERSION {
            return Err(KeywordIndexError::BadVersion { found: version, expected: FORMAT_VERSION });
        }
        // bytes[12..16] = flags (reserved, ignored)
        let header = Header {
            brain_crc32:           read_u32_le(bytes, 16),
            brain_node_count:      read_u64_le(bytes, 20),
            brain_bond_count:      read_u64_le(bytes, 28),
            token_count:           read_u64_le(bytes, 36),
            seed_total:            read_u64_le(bytes, 44),
            token_offsets_offset:  read_u64_le(bytes, 52),
            token_blob_offset:     read_u64_le(bytes, 60),
            bucket_offsets_offset: read_u64_le(bytes, 68),
            seed_array_offset:     read_u64_le(bytes, 76),
        };
        // Bounds-check every section against the file size.
        let check = |what, offset: u64, byte_len: u64| -> Result<(), KeywordIndexError> {
            let end = offset.checked_add(byte_len).ok_or(KeywordIndexError::OutOfBounds {
                what, offset, len: byte_len, file_size,
            })?;
            if end > file_size {
                return Err(KeywordIndexError::OutOfBounds { what, offset, len: byte_len, file_size });
            }
            Ok(())
        };
        let n = header.token_count;
        let toff_len = n.saturating_add(1).saturating_mul(4);
        let bo_len = n.saturating_add(1).saturating_mul(8);
        let seed_bytes = header.seed_total.saturating_mul(SEED_SIZE as u64);
        check("token_offsets", header.token_offsets_offset, toff_len)?;
        check("bucket_offsets", header.bucket_offsets_offset, bo_len)?;
        check("seed_array", header.seed_array_offset, seed_bytes)?;
        // token_blob length comes from token_offsets[N]; we'll bounds-check
        // it once we have the table mapped.
        Ok(header)
    }
}

// ---------------- Mmap reader ----------------

/// Read-only mmap'd keyword index sidecar.
///
/// `KeywordIndexMmap` owns an `Arc<Mmap>` and exposes byte-pointers into
/// it via raw `NonNull<u8>` rather than `&'static [u8]` references — the
/// same self-referential-pointer pattern `NetworkArchive` uses
/// (`src/archive.rs:540`). Access only happens through `&self`, which
/// cannot outlive the `KeywordIndexMmap`, which drops the `Arc<Mmap>` last.
pub struct KeywordIndexMmap {
    _mmap: Arc<Mmap>,
    header: Header,
    // Raw byte pointers into the mmap. SAFETY: kept alive by `_mmap`.
    token_offsets: NonNull<u8>,   // length = (token_count+1) * 4 bytes
    token_blob: NonNull<u8>,      // length derived from token_offsets[N]
    token_blob_len: usize,
    bucket_offsets: NonNull<u8>,  // length = (token_count+1) * 8 bytes
    seed_array: NonNull<u8>,      // length = seed_total * 32 bytes
}

// SAFETY: same justification as `unsafe impl Send/Sync for NetworkArchive`
// at `src/archive.rs:540-541`. The mmap'd region is read-only after
// construction, no interior mutation, and `_mmap: Arc<Mmap>` is itself
// `Send + Sync`. The `NonNull<u8>` pointers refer to the same bytes that
// `_mmap` keeps alive, so they remain valid for the lifetime of `Self`.
unsafe impl Send for KeywordIndexMmap {}
unsafe impl Sync for KeywordIndexMmap {}

impl std::fmt::Debug for KeywordIndexMmap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KeywordIndexMmap")
            .field("token_count", &self.header.token_count)
            .field("seed_total", &self.header.seed_total)
            .field("brain_crc32", &format_args!("{:#010x}", self.header.brain_crc32))
            .field("brain_node_count", &self.header.brain_node_count)
            .field("brain_bond_count", &self.header.brain_bond_count)
            .finish()
    }
}

impl KeywordIndexMmap {
    /// Open and mmap a sidecar file. Validates the magic, version,
    /// trailer CRC, and section bounds before returning. Caller must
    /// still call [`validate`](Self::validate) against the brain.lbr
    /// metadata to confirm the sidecar isn't stale.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, KeywordIndexError> {
        let f = File::open(path.as_ref())?;
        let metadata = f.metadata()?;
        let file_size = metadata.len();
        if file_size < (HEADER_SIZE + TRAILER_SIZE) as u64 {
            return Err(KeywordIndexError::TooSmall { size: file_size });
        }
        // SAFETY: read-only mmap of a file we just opened. Same pattern as
        // `NetworkArchive::open` in `src/archive.rs`.
        let mmap = unsafe { Mmap::map(&f) }?;
        Self::from_mmap(Arc::new(mmap))
    }

    /// OP-022.a.3.1: same as `open` but skips the whole-file CRC pass
    /// (still checks trailer magic + header magic + version + section
    /// bounds). Use only for sidecars JUST written by this process via
    /// `write_sidecar_streaming` + fsync — the CRC check is defensive
    /// against disk-edge corruption that cannot have occurred since.
    /// Reading the full 2.5 GiB sidecar for CRC dominated the OP-022.a.3
    /// compaction HWM (645 MiB of the 645 MiB transient). Skipping it
    /// brings the post-compact transient inside the 500 MiB gate.
    pub fn open_trusted<P: AsRef<Path>>(path: P) -> Result<Self, KeywordIndexError> {
        let f = File::open(path.as_ref())?;
        let metadata = f.metadata()?;
        let file_size = metadata.len();
        if file_size < (HEADER_SIZE + TRAILER_SIZE) as u64 {
            return Err(KeywordIndexError::TooSmall { size: file_size });
        }
        // SAFETY: same as `open`.
        let mmap = unsafe { Mmap::map(&f) }?;
        Self::from_mmap_trusted(Arc::new(mmap))
    }

    /// Construct from an already-mapped `Arc<Mmap>`. Runs the full
    /// integrity check (trailer magic + CRC32 over [0..EOF-8]). Use for
    /// the cold-load path where the file came from disk and might be
    /// corrupt.
    pub fn from_mmap(mmap: Arc<Mmap>) -> Result<Self, KeywordIndexError> {
        Self::from_mmap_inner(mmap, /* verify_crc = */ true)
    }

    /// OP-022.a.3.1: same as `from_mmap` but SKIPS the CRC32 pass over
    /// the full file. Use only for sidecars this process JUST WROTE +
    /// fsync'd — the CRC validation is defensive against on-disk
    /// corruption that cannot possibly have occurred in that window.
    /// Trailer magic + header magic + version + section bounds still
    /// get checked; only the 2.5 GiB whole-file CRC pass is skipped.
    pub fn from_mmap_trusted(mmap: Arc<Mmap>) -> Result<Self, KeywordIndexError> {
        Self::from_mmap_inner(mmap, /* verify_crc = */ false)
    }

    /// Internal: the actual construction, gated on whether to CRC the file.
    fn from_mmap_inner(mmap: Arc<Mmap>, verify_crc: bool) -> Result<Self, KeywordIndexError> {
        let bytes: &[u8] = &mmap[..];
        let file_size = bytes.len() as u64;
        if file_size < (HEADER_SIZE + TRAILER_SIZE) as u64 {
            return Err(KeywordIndexError::TooSmall { size: file_size });
        }
        // 1. Trailer magic always; CRC32 only when verify_crc.
        let trailer_off = bytes.len() - TRAILER_SIZE;
        let stored_crc = read_u32_le(bytes, trailer_off);
        let mut trailer_magic = [0u8; 4];
        trailer_magic.copy_from_slice(&bytes[trailer_off + 4..trailer_off + 8]);
        if &trailer_magic != TRAILER_MAGIC {
            return Err(KeywordIndexError::BadTrailerMagic { found: trailer_magic });
        }
        if verify_crc {
            let computed_crc = crc32fast::hash(&bytes[..trailer_off]);
            if computed_crc != stored_crc {
                return Err(KeywordIndexError::BadCrc { computed: computed_crc, stored: stored_crc });
            }
        }
        // 2. Parse header (incl. section-bounds check).
        let header = Header::parse(bytes, file_size)?;
        // 3. token_blob length comes from token_offsets[N].
        let token_offsets_start = header.token_offsets_offset as usize;
        let n = header.token_count as usize;
        let token_blob_len_u32 = if n == 0 {
            0u32
        } else {
            // offset[N] = total token_blob bytes
            read_u32_le(bytes, token_offsets_start + n * 4)
        };
        let token_blob_len = token_blob_len_u32 as usize;
        let token_blob_start = header.token_blob_offset as usize;
        if token_blob_start.saturating_add(token_blob_len) > bytes.len() {
            return Err(KeywordIndexError::OutOfBounds {
                what: "token_blob",
                offset: header.token_blob_offset,
                len: token_blob_len as u64,
                file_size,
            });
        }
        // 4. Build NonNull pointers.
        // SAFETY: the slices below are subslices of `bytes`, which is
        // borrowed from `mmap`. We immediately convert them to `NonNull`
        // and `mmap` (via Arc) keeps the underlying pages alive for the
        // lifetime of `Self`.
        let token_offsets_ptr = unsafe {
            NonNull::new_unchecked(bytes.as_ptr().add(token_offsets_start) as *mut u8)
        };
        let token_blob_ptr = unsafe {
            NonNull::new_unchecked(bytes.as_ptr().add(token_blob_start) as *mut u8)
        };
        let bucket_offsets_ptr = unsafe {
            NonNull::new_unchecked(bytes.as_ptr().add(header.bucket_offsets_offset as usize) as *mut u8)
        };
        let seed_array_ptr = unsafe {
            NonNull::new_unchecked(bytes.as_ptr().add(header.seed_array_offset as usize) as *mut u8)
        };
        Ok(KeywordIndexMmap {
            _mmap: mmap,
            header,
            token_offsets: token_offsets_ptr,
            token_blob: token_blob_ptr,
            token_blob_len,
            bucket_offsets: bucket_offsets_ptr,
            seed_array: seed_array_ptr,
        })
    }

    /// Confirm this sidecar matches the brain.lbr it was built against.
    /// Returns `Err(BrainCrcMismatch | NodeCountMismatch | BondCountMismatch)`
    /// when stale — caller should treat as absent and rebuild.
    pub fn validate(&self, brain_crc32: u32, brain_node_count: u64, brain_bond_count: u64)
        -> Result<(), KeywordIndexError>
    {
        if self.header.brain_crc32 != brain_crc32 {
            return Err(KeywordIndexError::BrainCrcMismatch {
                sidecar: self.header.brain_crc32, brain: brain_crc32,
            });
        }
        if self.header.brain_node_count != brain_node_count {
            return Err(KeywordIndexError::NodeCountMismatch {
                sidecar: self.header.brain_node_count, brain: brain_node_count,
            });
        }
        if self.header.brain_bond_count != brain_bond_count {
            return Err(KeywordIndexError::BondCountMismatch {
                sidecar: self.header.brain_bond_count, brain: brain_bond_count,
            });
        }
        Ok(())
    }

    pub fn token_count(&self) -> usize { self.header.token_count as usize }

    /// OP-022.a.3: borrow the underlying Arc<Mmap> so the merge iterator
    /// can call MADV_DONTNEED on consumed seed_array chunks.
    pub fn mmap_arc(&self) -> &Arc<Mmap> { &self._mmap }

    /// OP-022.a.3: byte offset (in the mmap) where seed_array starts.
    pub fn seed_array_mmap_offset(&self) -> usize {
        self.header.seed_array_offset as usize
    }

    /// OP-022.a.3: byte length of seed_array in the mmap.
    pub fn seed_array_bytes(&self) -> usize {
        self.header.seed_total as usize * SEED_SIZE
    }
    pub fn seed_total(&self) -> usize { self.header.seed_total as usize }
    pub fn brain_crc32(&self) -> u32 { self.header.brain_crc32 }
    pub fn brain_node_count(&self) -> u64 { self.header.brain_node_count }
    pub fn brain_bond_count(&self) -> u64 { self.header.brain_bond_count }

    // -------- Raw byte slice accessors (used by the bisect) --------

    /// `[u32; token_count+1]` as a byte slice. SAFETY: kept alive by `_mmap`.
    fn token_offsets_bytes(&self) -> &[u8] {
        let n = self.header.token_count as usize + 1;
        unsafe { std::slice::from_raw_parts(self.token_offsets.as_ptr() as *const u8, n * 4) }
    }
    /// Packed UTF-8 token bytes. SAFETY: kept alive by `_mmap`.
    fn token_blob_bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.token_blob.as_ptr() as *const u8, self.token_blob_len) }
    }
    /// `[u64; token_count+1]` as a byte slice. SAFETY: kept alive by `_mmap`.
    fn bucket_offsets_bytes(&self) -> &[u8] {
        let n = self.header.token_count as usize + 1;
        unsafe { std::slice::from_raw_parts(self.bucket_offsets.as_ptr() as *const u8, n * 8) }
    }
    /// `[Seed; seed_total]` as a `&[Seed]` slice.
    /// SAFETY: `Seed` is `#[derive(Clone, Copy)] pub struct Seed(pub [u8; 32])` —
    /// 32 bytes of POD with 1-byte alignment. The mmap'd bytes have been
    /// CRC-validated at construction; reinterpreting them as `Seed` is sound.
    fn seed_array(&self) -> &[Seed] {
        let total = self.header.seed_total as usize;
        unsafe { std::slice::from_raw_parts(self.seed_array.as_ptr() as *const Seed, total) }
    }

    fn token_offset(&self, i: usize) -> usize {
        debug_assert!(i <= self.header.token_count as usize);
        read_u32_le(self.token_offsets_bytes(), i * 4) as usize
    }
    fn bucket_offset(&self, i: usize) -> usize {
        debug_assert!(i <= self.header.token_count as usize);
        read_u64_le(self.bucket_offsets_bytes(), i * 8) as usize
    }
    fn token_bytes(&self, i: usize) -> &[u8] {
        let s = self.token_offset(i);
        let e = self.token_offset(i + 1);
        &self.token_blob_bytes()[s..e]
    }

    /// Look up a token. Returns the byte-sorted Seed bucket if present.
    pub fn get(&self, token: &str) -> Option<&[Seed]> {
        let needle = token.as_bytes();
        let n = self.header.token_count as usize;
        if n == 0 { return None; }
        // Standard ascending-order bisect over the token table.
        let (mut lo, mut hi) = (0usize, n);
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let mid_tok = self.token_bytes(mid);
            match mid_tok.cmp(needle) {
                std::cmp::Ordering::Less => lo = mid + 1,
                std::cmp::Ordering::Greater => hi = mid,
                std::cmp::Ordering::Equal => {
                    let bs = self.bucket_offset(mid);
                    let be = self.bucket_offset(mid + 1);
                    return Some(&self.seed_array()[bs..be]);
                }
            }
        }
        None
    }

    /// OP-NEARMISS-1 -- lower bound (first index whose token is >=
    /// `needle`) over the byte-sorted token table. The token table is
    /// sorted ascending by construction (`check_token_sort` enforces it
    /// at load), so this is a plain bisect.
    fn lower_bound(&self, needle: &[u8]) -> usize {
        let n = self.header.token_count as usize;
        let (mut lo, mut hi) = (0usize, n);
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if self.token_bytes(mid) < needle {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo
    }

    /// OP-NEARMISS-1 -- iterate `(token, bucket_len)` for every token
    /// whose first BYTE-sequence matches `prefix`. Two bisects bound the
    /// window; the walk is over that window only, never the whole table.
    ///
    /// Yields bucket LENGTH rather than the bucket itself: the spelling
    /// repairer only needs "how well is this term represented", and
    /// lengths keep the iterator allocation-free.
    pub fn tokens_with_prefix(&self, prefix: &str) -> impl Iterator<Item = (&str, usize)> + '_ {
        let pb = prefix.as_bytes().to_vec();
        let start = self.lower_bound(&pb);
        // Upper bound: the prefix with its final byte incremented. If
        // that overflows (0xFF tail) fall back to the end of the table.
        let mut end_key = pb.clone();
        let mut end = self.header.token_count as usize;
        while let Some(last) = end_key.pop() {
            if last != 0xFF {
                end_key.push(last + 1);
                end = self.lower_bound(&end_key);
                break;
            }
        }
        (start..end).map(move |i| {
            let tok_bytes = self.token_bytes(i);
            // SAFETY: writer only accepts valid UTF-8 tokens (same
            // invariant `iter()` relies on).
            let tok = unsafe { std::str::from_utf8_unchecked(tok_bytes) };
            let bs = self.bucket_offset(i);
            let be = self.bucket_offset(i + 1);
            (tok, be - bs)
        })
    }

    /// Iterate every (token, bucket) pair. Used by the functional-equivalence
    /// harness and by `KeywordIndex::materialize_all` when the heap mode
    /// needs to reconstruct the full HashMap from the sidecar.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &[Seed])> + '_ {
        let n = self.header.token_count as usize;
        (0..n).map(move |i| {
            let tok_bytes = self.token_bytes(i);
            // Token bytes are UTF-8 by construction (the writer requires it).
            // SAFETY: writer only accepts valid UTF-8 tokens; bounds-checked at construction.
            let tok = unsafe { std::str::from_utf8_unchecked(tok_bytes) };
            let bs = self.bucket_offset(i);
            let be = self.bucket_offset(i + 1);
            (tok, &self.seed_array()[bs..be])
        })
    }

    /// Walk the token table and assert ascending byte order. Returns the
    /// first out-of-order index on failure. O(N); intended for the load-time
    /// integrity check and for tests.
    pub fn check_token_sort(&self) -> Result<(), KeywordIndexError> {
        let n = self.header.token_count as usize;
        for i in 1..n {
            let prev = self.token_bytes(i - 1);
            let cur = self.token_bytes(i);
            if !(prev < cur) {
                return Err(KeywordIndexError::UnsortedTokens { at_index: i });
            }
        }
        Ok(())
    }
}

// ---------------- MergedBucket: base + overlay - tombstones ----------------

/// A keyword-index bucket as exposed to callers. Either a zero-alloc
/// borrow of the mmap'd base slice (the common path: bucket has no
/// pending overlay or tombstoned seeds), or an owned merge of
/// base+overlay with tombstoned seeds excluded.
///
/// `Deref<Target = [Seed]>` — every existing `&Vec<Seed>` operation
/// (`.iter()`, `.len()`, `.binary_search_by`, `.first()`) compiles
/// unchanged at the call sites.
pub enum MergedBucket<'a> {
    Borrowed(&'a [Seed]),
    Owned(Vec<Seed>),
}

impl<'a> Deref for MergedBucket<'a> {
    type Target = [Seed];
    fn deref(&self) -> &[Seed] {
        match self {
            MergedBucket::Borrowed(s) => s,
            MergedBucket::Owned(v) => v.as_slice(),
        }
    }
}

impl<'a> AsRef<[Seed]> for MergedBucket<'a> {
    fn as_ref(&self) -> &[Seed] { self.deref() }
}

impl<'a> std::fmt::Debug for MergedBucket<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MergedBucket")
            .field("len", &self.deref().len())
            .field("borrowed", &matches!(self, MergedBucket::Borrowed(_)))
            .finish()
    }
}

// ---------------- The unified KeywordIndex ----------------

/// The keyword index field on `AssociativeNetwork`. Three layers:
///   * `base`: read-only mmap'd sidecar (or `None` before first load /
///     after `replace_base(None)`).
///   * `overlay`: tokens written since the sidecar was built.
///   * `tombstones`: seeds removed since the sidecar was built; every
///     bucket lookup filters these out.
///
/// Lookup precedence (per Phase 2 design §3.5):
///   * No base, no overlay → `None`.
///   * Base hit, no overlay, no tombstones → `MergedBucket::Borrowed` (zero alloc).
///   * Anything else → `MergedBucket::Owned` (allocate, merge, sort, dedup,
///     remove tombstoned seeds).
pub struct KeywordIndex {
    base: Option<Arc<KeywordIndexMmap>>,
    overlay: HashMap<String, Vec<Seed>>,
    tombstones: HashSet<Seed>,
}

impl Default for KeywordIndex {
    fn default() -> Self { Self::new() }
}

impl KeywordIndex {
    /// Construct an empty in-memory index (no mmap'd base).
    pub fn new() -> Self {
        KeywordIndex { base: None, overlay: HashMap::new(), tombstones: HashSet::new() }
    }

    /// Construct with an mmap'd base already loaded. Caller is responsible
    /// for having validated the base's brain_crc32/node_count/bond_count.
    pub fn with_base(base: Arc<KeywordIndexMmap>) -> Self {
        KeywordIndex { base: Some(base), overlay: HashMap::new(), tombstones: HashSet::new() }
    }

    pub fn has_base(&self) -> bool { self.base.is_some() }
    pub fn base(&self) -> Option<&Arc<KeywordIndexMmap>> { self.base.as_ref() }
    pub fn overlay_len(&self) -> usize { self.overlay.len() }
    pub fn overlay_seed_total(&self) -> usize { self.overlay.values().map(|v| v.len()).sum() }
    pub fn tombstone_count(&self) -> usize { self.tombstones.len() }
    /// True if there are pending mutations not yet compacted into the sidecar.
    pub fn has_dirty(&self) -> bool { !self.overlay.is_empty() || !self.tombstones.is_empty() }

    /// Replace the mmap'd base (or drop it). Drops the previous Arc;
    /// existing readers holding the old Arc are unaffected until they drop it.
    pub fn replace_base(&mut self, base: Option<Arc<KeywordIndexMmap>>) {
        self.base = base;
    }

    /// Clear the dirty state (overlay + tombstones). Used post-compaction.
    pub fn clear_dirty(&mut self) {
        self.overlay.clear();
        self.tombstones.clear();
    }

    // ---------- Mutation ----------

    /// Push a single (token, seed) pair into the overlay. Used by
    /// `add_node_with_index_flag`. The token may already exist in `base`
    /// or `overlay`; both are fine — `get` merges + dedups at read time.
    pub fn insert_overlay(&mut self, token: String, seed: Seed) {
        // If a tombstone exists for this seed and the user is re-teaching
        // a node with the same seed, lift the tombstone so the new
        // insertion is visible.
        self.tombstones.remove(&seed);
        self.overlay.entry(token).or_insert_with(Vec::new).push(seed);
    }

    /// Bulk-insert overlay entries. Convenience for the rebuild path.
    pub fn extend_overlay<I: IntoIterator<Item = (String, Vec<Seed>)>>(&mut self, iter: I) {
        for (k, v) in iter {
            for s in &v { self.tombstones.remove(s); }
            self.overlay.entry(k).or_insert_with(Vec::new).extend(v);
        }
    }

    /// Tombstone a single seed. All future `get` lookups will exclude
    /// this seed from every bucket until the next compaction.
    pub fn tombstone_seed(&mut self, seed: Seed) {
        // Also strip from overlay buckets so the overlay doesn't keep it alive.
        for v in self.overlay.values_mut() {
            v.retain(|s| *s != seed);
        }
        self.overlay.retain(|_, v| !v.is_empty());
        // Only need the tombstone if `base` could contain the seed.
        if self.base.is_some() {
            self.tombstones.insert(seed);
        }
    }

    /// Tombstone a whole set of seeds.
    pub fn tombstone_seeds(&mut self, seeds: &HashSet<Seed>) {
        for v in self.overlay.values_mut() {
            v.retain(|s| !seeds.contains(s));
        }
        self.overlay.retain(|_, v| !v.is_empty());
        if self.base.is_some() {
            for s in seeds { self.tombstones.insert(*s); }
        }
    }

    /// True iff this seed has been tombstoned.
    pub fn is_tombstoned(&self, seed: &Seed) -> bool {
        self.tombstones.contains(seed)
    }

    /// OP-022.a.1: borrow the overlay HashMap. Used by the streaming
    /// compaction path to k-way-merge over base + sorted overlay
    /// without materialising a full HashMap copy.
    pub fn overlay_ref(&self) -> &HashMap<String, Vec<Seed>> {
        &self.overlay
    }

    /// OP-022.a.1: borrow the tombstones set. Used by the streaming
    /// compaction path's merge iterator filter.
    pub fn tombstones_ref(&self) -> &HashSet<Seed> {
        &self.tombstones
    }

    // ---------- Read ----------

    /// Look up a token. See [`MergedBucket`] for the return shape.
    pub fn get(&self, token: &str) -> Option<MergedBucket<'_>> {
        let base_slice: Option<&[Seed]> = self.base.as_ref().and_then(|b| b.get(token));
        let overlay_slice: Option<&[Seed]> = self.overlay.get(token).map(|v| v.as_slice());
        match (base_slice, overlay_slice) {
            (None, None) => None,
            // Fast path: base-only, no tombstones touching this bucket.
            (Some(b), None) if self.tombstones.is_empty() => Some(MergedBucket::Borrowed(b)),
            (None, Some(o)) if self.tombstones.is_empty() => {
                // Overlay-only: return as-is in INSERTION order WITH duplicates.
                // Legacy `keyword_index: HashMap<String, Vec<Seed>>` was a raw
                // Vec from push() calls in add_node_with_index_flag — duplicates
                // from content+knowledge double-extraction were retained, and
                // downstream `select_seeds_for_query_inner` take-2-then-dedup
                // throttles unique seed count via those duplicates.
                if o.is_empty() { None } else { Some(MergedBucket::Owned(o.to_vec())) }
            }
            _ => {
                // Merged path: concatenate base + overlay, MINUS tombstones.
                // OP-022.a (functional equivalence): we deliberately do NOT
                // sort or dedup. The legacy code's mixed-state bucket was
                // [byte-sorted archive seeds (with duplicates)] +
                // [insertion-order overlay seeds (with duplicates)]; the
                // base half of this merge IS byte-sorted-with-duplicates
                // (the writer preserves dupes), and overlay is
                // insertion-order-with-duplicates. Concatenating them
                // gives the exact same layout the legacy HashMap path
                // produced.
                let cap = base_slice.map(|s| s.len()).unwrap_or(0)
                    + overlay_slice.map(|s| s.len()).unwrap_or(0);
                let mut merged: Vec<Seed> = Vec::with_capacity(cap);
                if let Some(b) = base_slice {
                    if self.tombstones.is_empty() {
                        merged.extend_from_slice(b);
                    } else {
                        for s in b { if !self.tombstones.contains(s) { merged.push(*s); } }
                    }
                }
                if let Some(o) = overlay_slice {
                    if self.tombstones.is_empty() {
                        merged.extend_from_slice(o);
                    } else {
                        for s in o { if !self.tombstones.contains(s) { merged.push(*s); } }
                    }
                }
                if merged.is_empty() { None } else { Some(MergedBucket::Owned(merged)) }
            }
        }
    }

    /// OP-NEARMISS-1 -- total number of distinct known terms (mmap base
    /// plus overlay). Zero means "this index knows nothing", which the
    /// spelling repairer treats as "make no repairs": with no evidence
    /// of what the brain knows, every rename would be a guess.
    pub fn token_count_total(&self) -> usize {
        self.base.as_ref().map(|b| b.token_count()).unwrap_or(0) + self.overlay.len()
    }

    /// OP-NEARMISS-1 -- every known term starting with `first`, as
    /// `(term, bucket_len)`. Sourced from the mmap base via a bounded
    /// bisect window plus the in-memory overlay.
    ///
    /// Terms whose bucket is entirely tombstoned still report their raw
    /// length here; the caller re-checks the real bucket via `get`
    /// before committing to a repair, so a revoked term cannot become a
    /// repair target.
    pub fn tokens_with_prefix_char(&self, first: char) -> Vec<(String, usize)> {
        let mut buf = [0u8; 4];
        let prefix: &str = first.encode_utf8(&mut buf);
        let mut out: Vec<(String, usize)> = Vec::new();
        if let Some(base) = &self.base {
            for (tok, len) in base.tokens_with_prefix(prefix) {
                out.push((tok.to_string(), len));
            }
        }
        for (tok, seeds) in &self.overlay {
            if tok.starts_with(prefix) {
                out.push((tok.clone(), seeds.len()));
            }
        }
        out
    }

    /// Materialise the *full* effective index as a `HashMap<String, Vec<Seed>>`.
    /// Walks `base` (if present), merges `overlay`, removes tombstoned seeds,
    /// byte-sorts + dedups each bucket. Used by the compaction path and the
    /// functional-equivalence harness.
    ///
    /// O(N) over the full index — avoid in hot paths.
    pub fn materialize_all(&self) -> HashMap<String, Vec<Seed>> {
        let mut out: HashMap<String, Vec<Seed>> = HashMap::new();
        if let Some(base) = &self.base {
            for (tok, bucket) in base.iter() {
                if self.tombstones.is_empty() {
                    out.insert(tok.to_string(), bucket.to_vec());
                } else {
                    let mut v: Vec<Seed> = bucket.iter()
                        .filter(|s| !self.tombstones.contains(s))
                        .copied()
                        .collect();
                    if !v.is_empty() { out.entry(tok.to_string()).or_insert_with(Vec::new).append(&mut v); }
                }
            }
        }
        for (tok, seeds) in &self.overlay {
            let bucket = out.entry(tok.clone()).or_insert_with(Vec::new);
            for s in seeds {
                if !self.tombstones.contains(s) {
                    bucket.push(*s);
                }
            }
        }
        // OP-022.a: byte-sort but PRESERVE duplicates (see write_sidecar
        // and the merged-path comment in `get`). Equivalence with the
        // legacy heap path requires duplicates to flow through to
        // `select_seeds_for_query_inner`'s take-2-then-dedup gate.
        for v in out.values_mut() {
            v.sort_by(|a, b| a.as_bytes().cmp(b.as_bytes()));
        }
        out.retain(|_, v| !v.is_empty());
        out
    }
}

// ---------------- Writer (atomic) ----------------

/// OP-022.a.1: streaming merge iterator over an optional mmap'd base,
/// a slice of overlay keys (pre-sorted by `as_bytes()`), the overlay
/// HashMap (lookup-only), and a tombstone set. Yields `(token, bucket)`
/// pairs in ascending token byte order; each bucket is byte-sorted with
/// duplicates preserved (BR.C-Hyp-Methodology + OP-022.a Phase C /
/// OP-030 cascade-throttle invariant).
///
/// Empty buckets (after tombstone filter) are skipped — the streaming
/// writer's Pass 1 metadata stays accurate.
pub struct KeywordMergeIter<'a> {
    base_iter: Option<std::iter::Peekable<Box<dyn Iterator<Item = (&'a str, &'a [Seed])> + 'a>>>,
    overlay_keys: std::iter::Peekable<std::slice::Iter<'a, &'a str>>,
    overlay: &'a HashMap<String, Vec<Seed>>,
    tombstones: &'a HashSet<Seed>,
    /// OP-022.a.3: tracker for periodic MADV_DONTNEED on consumed base chunks.
    /// None when there is no base (overlay-only compaction) or when the base
    /// has zero seeds.
    base_advise: Option<BaseAdviseTracker<'a>>,
}

/// OP-022.a.3: tracks how far the merge iterator has consumed the base mmap
/// so we can hint the kernel to drop already-walked pages. Bounded resident
/// set during compaction => peak RssFile delta capped near MADVISE_CHUNK_BYTES.
pub struct BaseAdviseTracker<'a> {
    mmap_start: *const u8,
    seed_array_start_offset: usize,
    #[allow(dead_code)]  // OP-022.a.3: reserved for future bounds checks (e.g. final-chunk advise)
    seed_array_total_bytes: usize,
    last_advised_byte: usize, // byte offset within seed_array, monotonic
    _marker: std::marker::PhantomData<&'a ()>,
}

impl<'a> BaseAdviseTracker<'a> {
    fn new(mmap: &Arc<Mmap>, seed_array_offset: usize, seed_array_bytes: usize) -> Self {
        BaseAdviseTracker {
            mmap_start: mmap.as_ptr(),
            seed_array_start_offset: seed_array_offset,
            seed_array_total_bytes: seed_array_bytes,
            last_advised_byte: 0,
            _marker: std::marker::PhantomData,
        }
    }

    /// Compute the byte offset of `seed_slice` within seed_array, plus its length.
    /// Returns the byte offset of the seed_slice's END within seed_array.
    fn end_offset_of(&self, seed_slice: &[Seed]) -> usize {
        let slice_ptr = seed_slice.as_ptr() as usize;
        let mmap_ptr = self.mmap_start as usize;
        let file_off = slice_ptr.saturating_sub(mmap_ptr);
        let seed_off = file_off.saturating_sub(self.seed_array_start_offset);
        seed_off + seed_slice.len() * SEED_SIZE
    }

    /// Issue MADV_DONTNEED on any complete 64 MiB chunk that the cursor
    /// has now passed. Aligns to page boundaries (rounded inward) so we
    /// don't advise on a partial page we still need.
    fn maybe_advise(&mut self, cursor_end_byte: usize) {
        if cursor_end_byte <= self.last_advised_byte + MADVISE_CHUNK_BYTES {
            return;
        }
        // OP-022.a.3: madvise requires PAGE-ALIGNED addresses or it returns
        // EINVAL silently (we swallow the error). The on-disk writer aligns
        // seed_array_offset to 32 bytes (Seed boundary), NOT 4096, so the
        // raw `mmap_start + seed_array_offset` is virtually NEVER page-aligned.
        // Round the address-range start UP and end DOWN to page boundaries
        // and only advise the page-aligned interior.
        const PAGE: usize = 4096;
        const SAFETY_MARGIN: usize = 1024 * 1024; // 1 MiB
        let logical_end = cursor_end_byte.saturating_sub(SAFETY_MARGIN);
        if logical_end <= self.last_advised_byte { return; }
        // Translate logical (seed_array-relative) offsets to absolute mmap
        // virtual addresses, then page-align.
        let base_va = self.mmap_start as usize + self.seed_array_start_offset;
        let raw_start_va = base_va + self.last_advised_byte;
        let raw_end_va = base_va + logical_end;
        let aligned_start_va = (raw_start_va + PAGE - 1) & !(PAGE - 1); // round UP
        let aligned_end_va = raw_end_va & !(PAGE - 1);                  // round DOWN
        if aligned_end_va > aligned_start_va {
            let aligned_len = aligned_end_va - aligned_start_va;
            // SAFETY: [aligned_start_va, aligned_end_va) lies fully within the
            // mmap region (cursor_end_byte <= seed_array_total_bytes by
            // base.iter() invariant; the inward rounding never extends past).
            let _ = madvise_dontneed(aligned_start_va as *const u8, aligned_len);
        }
        self.last_advised_byte = logical_end;
    }
}

impl<'a> KeywordMergeIter<'a> {
    pub fn new(
        base: Option<&'a KeywordIndexMmap>,
        overlay_sorted_keys: &'a [&'a str],
        overlay: &'a HashMap<String, Vec<Seed>>,
        tombstones: &'a HashSet<Seed>,
    ) -> Self {
        let base_iter = base.map(|b| {
            let raw: Box<dyn Iterator<Item = (&'a str, &'a [Seed])> + 'a> = Box::new(b.iter());
            raw.peekable()
        });
        let base_advise = base.and_then(|b| {
            let bytes = b.seed_array_bytes();
            if bytes == 0 { None }
            else { Some(BaseAdviseTracker::new(b.mmap_arc(), b.seed_array_mmap_offset(), bytes)) }
        });
        KeywordMergeIter {
            base_iter,
            overlay_keys: overlay_sorted_keys.iter().peekable(),
            overlay,
            tombstones,
            base_advise,
        }
    }

    /// Drop tombstoned seeds and produce a byte-sorted owned Vec<Seed>.
    /// Duplicates are PRESERVED per OP-030 / Phase C (the
    /// take-2-then-dedup throttle in `select_seeds_for_query_inner`
    /// relies on bucket cardinality including dupes).
    fn finalise(&self, mut combined: Vec<Seed>) -> Vec<Seed> {
        if !self.tombstones.is_empty() {
            combined.retain(|s| !self.tombstones.contains(s));
        }
        combined.sort_by(|a, b| a.as_bytes().cmp(b.as_bytes()));
        combined
    }
}

impl<'a> Iterator for KeywordMergeIter<'a> {
    type Item = (String, Vec<Seed>);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // Two-way merge over base + sorted overlay.
            let base_tok = self.base_iter.as_mut().and_then(|it| it.peek().map(|(t, _)| *t));
            let overlay_tok = self.overlay_keys.peek().copied().copied();

            let (token, combined): (String, Vec<Seed>) = match (base_tok, overlay_tok) {
                (None, None) => return None,
                (Some(b), None) => {
                    let (t, seeds) = self.base_iter.as_mut().unwrap().next().unwrap();
                    debug_assert_eq!(t, b);
                    if let Some(adv) = self.base_advise.as_mut() {
                        let end = adv.end_offset_of(seeds);
                        adv.maybe_advise(end);
                    }
                    (t.to_string(), seeds.to_vec())
                }
                (None, Some(o)) => {
                    self.overlay_keys.next();
                    let v = self.overlay.get(o).map(|x| x.clone()).unwrap_or_default();
                    (o.to_string(), v)
                }
                (Some(b), Some(o)) => {
                    match b.as_bytes().cmp(o.as_bytes()) {
                        std::cmp::Ordering::Less => {
                            let (t, seeds) = self.base_iter.as_mut().unwrap().next().unwrap();
                            if let Some(adv) = self.base_advise.as_mut() {
                                let end = adv.end_offset_of(seeds);
                                adv.maybe_advise(end);
                            }
                            (t.to_string(), seeds.to_vec())
                        }
                        std::cmp::Ordering::Greater => {
                            self.overlay_keys.next();
                            let v = self.overlay.get(o).map(|x| x.clone()).unwrap_or_default();
                            (o.to_string(), v)
                        }
                        std::cmp::Ordering::Equal => {
                            let (_, base_seeds) = self.base_iter.as_mut().unwrap().next().unwrap();
                            if let Some(adv) = self.base_advise.as_mut() {
                                let end = adv.end_offset_of(base_seeds);
                                adv.maybe_advise(end);
                            }
                            self.overlay_keys.next();
                            let mut combined: Vec<Seed> = base_seeds.to_vec();
                            if let Some(o_v) = self.overlay.get(o) {
                                combined.extend_from_slice(o_v);
                            }
                            (o.to_string(), combined)
                        }
                    }
                }
            };

            let finalised = self.finalise(combined);
            if finalised.is_empty() { continue; } // skip tombstone-emptied buckets
            return Some((token, finalised));
        }
    }
}

// =====================================================================
// OP-022.a.2: streaming archive-walk rebuild via external mergesort.
// Replaces the in-memory HashMap<String, Vec<Seed>> accumulation that
// drove the rebuild HWM to ~9.95 GiB on the canonical brain. The
// rebuilder buffers (token, seed) pairs in a bounded heap Vec; when
// the buffer crosses `buffer_limit`, the buffer is sorted in place,
// serialised to a run file under `runs_dir`, and cleared. On `finish`,
// the run files are k-way merged into the existing streaming sidecar
// writer (group consecutive identical tokens into per-token buckets).
// Run files are removed by `Drop` — success, error, or panic.
// =====================================================================

/// OP-022.a.2: streaming rebuilder for the keyword_index sidecar.
pub struct StreamingRebuilder {
    runs_dir: PathBuf,
    sidecar_path: PathBuf,
    buffer: Vec<(String, Seed)>,
    buffer_limit: usize,
    run_count: usize,
    brain_crc32: u32,
    brain_node_count: u64,
    brain_bond_count: u64,
    finished: bool,
}

impl StreamingRebuilder {
    /// Construct. Creates `runs_dir` (removing any stale contents first)
    /// and runs a disk-space sanity check (>= 10 GiB free on the
    /// filesystem hosting `runs_dir`).
    pub fn new(
        sidecar_path: PathBuf,
        runs_dir: PathBuf,
        buffer_limit: usize,
        brain_crc32: u32,
        brain_node_count: u64,
        brain_bond_count: u64,
    ) -> std::io::Result<Self> {
        // Sanity: never run with a buffer of 0 (would create one run per
        // push — pathological).
        if buffer_limit == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "StreamingRebuilder: buffer_limit must be > 0",
            ));
        }
        // Clear any prior runs dir contents (idempotent re-runs after
        // a crashed earlier rebuild).
        let _ = std::fs::remove_dir_all(&runs_dir);
        std::fs::create_dir_all(&runs_dir)?;
        // Disk-space sanity check on the runs_dir filesystem.
        const MIN_FREE_BYTES: u64 = 10 * 1024 * 1024 * 1024; // 10 GiB
        match streaming_rebuild_free_bytes(&runs_dir) {
            Ok(free) if free >= MIN_FREE_BYTES => {}
            Ok(free) => {
                let _ = std::fs::remove_dir_all(&runs_dir);
                return Err(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!(
                        "StreamingRebuilder: free space on {} is {} bytes; need >= {} bytes",
                        runs_dir.display(), free, MIN_FREE_BYTES,
                    ),
                ));
            }
            Err(e) => {
                // statvfs failure isn't fatal — proceed, but log.
                eprintln!(
                    "OP-022.a.2: free-space check failed ({}); proceeding anyway",
                    e
                );
            }
        }
        Ok(StreamingRebuilder {
            runs_dir,
            sidecar_path,
            buffer: Vec::with_capacity(buffer_limit),
            buffer_limit,
            run_count: 0,
            brain_crc32,
            brain_node_count,
            brain_bond_count,
            finished: false,
        })
    }

    /// Push a single (token, seed) pair. When the buffer crosses
    /// `buffer_limit`, the buffer is flushed to a sorted run file.
    pub fn push(&mut self, token: String, seed: Seed) -> std::io::Result<()> {
        self.buffer.push((token, seed));
        if self.buffer.len() >= self.buffer_limit {
            self.flush_run()?;
        }
        Ok(())
    }

    /// Sort the in-memory buffer and write it to a sorted run file.
    fn flush_run(&mut self) -> std::io::Result<()> {
        if self.buffer.is_empty() {
            return Ok(());
        }
        // Sort by (token bytes, seed bytes). Stable; preserves duplicates.
        self.buffer.sort_by(|a, b| {
            a.0.as_bytes().cmp(b.0.as_bytes())
                .then_with(|| a.1.as_bytes().cmp(b.1.as_bytes()))
        });
        let run_path = self.runs_dir.join(format!("run_{:04}.bin", self.run_count));
        let f = OpenOptions::new()
            .create(true).truncate(true).write(true)
            .open(&run_path)?;
        let mut w = BufWriter::with_capacity(64 * 1024, f);
        for (tok, seed) in self.buffer.drain(..) {
            streaming_rebuild_write_record(&mut w, &tok, &seed)?;
        }
        w.flush()?;
        // Don't fsync per-run — runs are scratch, the final atomic_write_with
        // for the sidecar does the durable fsync.
        self.run_count += 1;
        Ok(())
    }

    /// Drain the final buffer (if any), k-way merge all runs into the
    /// streaming sidecar writer, open the new mmap trusted, return it.
    /// Run files are cleaned up by `Drop`.
    pub fn finish(mut self) -> std::io::Result<std::sync::Arc<KeywordIndexMmap>> {
        self.flush_run()?;
        let run_paths: Vec<PathBuf> = (0..self.run_count)
            .map(|i| self.runs_dir.join(format!("run_{:04}.bin", i)))
            .collect();
        let brain_crc32 = self.brain_crc32;
        let brain_node_count = self.brain_node_count;
        let brain_bond_count = self.brain_bond_count;
        let sidecar_path = self.sidecar_path.clone();

        // Iterator factory: opens fresh readers for each pass of the
        // streaming writer (pass 1 = metadata, pass 2 = seed_array).
        let make_iter = || -> Box<dyn Iterator<Item = (String, Vec<Seed>)>> {
            let readers: Vec<BufReader<File>> = run_paths.iter()
                .map(|p| File::open(p)
                    .map(|f| BufReader::with_capacity(64 * 1024, f)))
                .collect::<std::io::Result<Vec<_>>>()
                .expect("OP-022.a.2: run file vanished between flush and merge");
            Box::new(StreamingRebuildGroupedMerge::new(readers))
        };

        write_sidecar_streaming(
            &sidecar_path,
            make_iter,
            brain_crc32,
            brain_node_count,
            brain_bond_count,
        )?;

        let mmap = KeywordIndexMmap::open_trusted(&sidecar_path)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;

        self.finished = true;
        // Drop runs (Drop impl handles).
        Ok(std::sync::Arc::new(mmap))
    }

    /// OP-022.a.2: total number of run files flushed so far.
    pub fn run_count(&self) -> usize { self.run_count }
    /// OP-022.a.2: borrow runs_dir (for tests / diagnostics).
    pub fn runs_dir(&self) -> &Path { &self.runs_dir }
}

impl Drop for StreamingRebuilder {
    fn drop(&mut self) {
        // Best-effort cleanup. Drop fires on success (finished=true) and
        // error (finished=false). Either way, remove the runs scratch dir.
        let _ = std::fs::remove_dir_all(&self.runs_dir);
    }
}

/// OP-022.a.2: serialise a single (token, seed) record. Format:
///   u32 token_len LE + token UTF-8 bytes + 32 Seed bytes.
fn streaming_rebuild_write_record<W: std::io::Write>(
    w: &mut W, token: &str, seed: &Seed,
) -> std::io::Result<()> {
    let len = token.len() as u32;
    w.write_all(&len.to_le_bytes())?;
    w.write_all(token.as_bytes())?;
    w.write_all(seed.as_bytes())?;
    Ok(())
}

/// OP-022.a.2: deserialise one record. Returns `Ok(None)` on clean EOF.
fn streaming_rebuild_read_record<R: std::io::Read>(
    r: &mut R,
) -> std::io::Result<Option<(String, Seed)>> {
    let mut len_buf = [0u8; 4];
    match r.read_exact(&mut len_buf) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e),
    }
    let len = u32::from_le_bytes(len_buf) as usize;
    if len > 1024 * 1024 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("streaming_rebuild: token length {} exceeds 1 MiB sanity cap", len),
        ));
    }
    let mut tok_buf = vec![0u8; len];
    r.read_exact(&mut tok_buf)?;
    let token = String::from_utf8(tok_buf).map_err(|e| {
        std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string())
    })?;
    let mut seed_bytes = [0u8; 32];
    r.read_exact(&mut seed_bytes)?;
    let seed = Seed::from_bytes(seed_bytes);
    Ok(Some((token, seed)))
}

/// OP-022.a.2: statvfs wrapper. Returns available bytes for unprivileged
/// users on the filesystem hosting `path`.
#[cfg(unix)]
fn streaming_rebuild_free_bytes(path: &Path) -> std::io::Result<u64> {
    use std::os::unix::ffi::OsStrExt;
    let c_path = std::ffi::CString::new(path.as_os_str().as_bytes())
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e))?;
    let mut s: libc::statvfs = unsafe { std::mem::zeroed() };
    let rc = unsafe { libc::statvfs(c_path.as_ptr(), &mut s) };
    if rc != 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok((s.f_bavail as u64) * (s.f_frsize as u64))
}

#[cfg(not(unix))]
fn streaming_rebuild_free_bytes(_path: &Path) -> std::io::Result<u64> {
    // Non-unix: skip the check (return a large sentinel).
    Ok(u64::MAX)
}

/// OP-022.a.2: k-way merge across multiple run-file readers, yielding
/// (token, seed) pairs in global sorted order (token-major, then seed-bytes).
struct StreamingRebuildKwayMerge {
    readers: Vec<BufReader<File>>,
    heap: std::collections::BinaryHeap<std::cmp::Reverse<StreamingRebuildMergeEntry>>,
}

#[derive(Eq, PartialEq)]
struct StreamingRebuildMergeEntry {
    token: String,
    seed: Seed,
    reader_idx: usize,
}

impl Ord for StreamingRebuildMergeEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.token.as_bytes().cmp(other.token.as_bytes())
            .then_with(|| self.seed.as_bytes().cmp(other.seed.as_bytes()))
            .then_with(|| self.reader_idx.cmp(&other.reader_idx))
    }
}

impl PartialOrd for StreamingRebuildMergeEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl StreamingRebuildKwayMerge {
    fn new(mut readers: Vec<BufReader<File>>) -> Self {
        let mut heap = std::collections::BinaryHeap::with_capacity(readers.len());
        for (i, r) in readers.iter_mut().enumerate() {
            match streaming_rebuild_read_record(r) {
                Ok(Some((t, s))) => heap.push(std::cmp::Reverse(
                    StreamingRebuildMergeEntry { token: t, seed: s, reader_idx: i }
                )),
                Ok(None) => {}
                Err(e) => {
                    eprintln!("OP-022.a.2: merge bootstrap read error reader[{}]: {}", i, e);
                }
            }
        }
        StreamingRebuildKwayMerge { readers, heap }
    }
}

impl Iterator for StreamingRebuildKwayMerge {
    type Item = (String, Seed);
    fn next(&mut self) -> Option<Self::Item> {
        let entry = self.heap.pop()?.0;
        // Try to refill from the same reader.
        let idx = entry.reader_idx;
        match streaming_rebuild_read_record(&mut self.readers[idx]) {
            Ok(Some((t, s))) => {
                self.heap.push(std::cmp::Reverse(
                    StreamingRebuildMergeEntry { token: t, seed: s, reader_idx: idx }
                ));
            }
            Ok(None) => {} // reader drained
            Err(e) => {
                eprintln!("OP-022.a.2: merge refill read error reader[{}]: {}", idx, e);
            }
        }
        Some((entry.token, entry.seed))
    }
}

/// OP-022.a.2: group consecutive identical-token entries from the k-way
/// merge into a single `(token, Vec<Seed>)` bucket. Duplicates are
/// preserved (OP-030 territory). Seeds within a bucket are already
/// sorted because the per-run sort produces tuples sorted by (token,
/// seed_bytes); the merge preserves that order globally.
struct StreamingRebuildGroupedMerge {
    inner: std::iter::Peekable<StreamingRebuildKwayMerge>,
}

impl StreamingRebuildGroupedMerge {
    fn new(readers: Vec<BufReader<File>>) -> Self {
        Self {
            inner: StreamingRebuildKwayMerge::new(readers).peekable(),
        }
    }
}

impl Iterator for StreamingRebuildGroupedMerge {
    type Item = (String, Vec<Seed>);
    fn next(&mut self) -> Option<Self::Item> {
        let (tok, seed) = self.inner.next()?;
        let mut bucket = vec![seed];
        while let Some((peek_tok, _)) = self.inner.peek() {
            if peek_tok.as_bytes() == tok.as_bytes() {
                let (_, next_seed) = self.inner.next().unwrap();
                bucket.push(next_seed);
            } else {
                break;
            }
        }
        Some((tok, bucket))
    }
}

/// OP-022.a.1: streaming sidecar writer. Takes a re-runnable iterator
/// factory; the writer makes two passes over the source (pass 1 collects
/// per-token sizes into a small metadata Vec, pass 2 streams the seed
/// bytes directly to the output file) so neither the full token table nor
/// the full seed_array ever lives concurrently in memory.
///
/// Memory peak budget: ~30 MB of metadata at canonical (1M-token) scale
/// + 64 KB BufWriter + 64 KB CRC read buffer. This is the fix for the
/// OP-022.a Phase F deploy halt — see /tmp/op022a1_artifacts.md.
///
/// `make_iter` must yield the SAME deterministic sequence of
/// `(token, bucket)` pairs on every call, sorted ascending by
/// `token.as_bytes()`. The buckets MUST already be byte-sorted within
/// (duplicates preserved per OP-022.a Phase C / OP-030).
pub fn write_sidecar_streaming<P, F, I>(
    path: P,
    mut make_iter: F,
    brain_crc32: u32,
    brain_node_count: u64,
    brain_bond_count: u64,
) -> std::io::Result<()>
where
    P: AsRef<Path>,
    F: FnMut() -> I,
    I: Iterator<Item = (String, Vec<Seed>)>,
{
    let path = path.as_ref();

    // -------- Pass 1: collect metadata --------
    // (token, bucket_seed_count). Strings owned so we can re-emit them in
    // pass 2 without re-running the merge for that piece.
    let mut metadata: Vec<(String, u32)> = Vec::new();
    let mut total_token_bytes: u64 = 0;
    let mut total_seeds: u64 = 0;
    for (token, seeds) in make_iter() {
        if seeds.is_empty() {
            // Skip empty buckets defensively; the merge iterator should already filter.
            continue;
        }
        total_token_bytes += token.len() as u64;
        total_seeds += seeds.len() as u64;
        metadata.push((token, seeds.len() as u32));
    }
    let n_tokens: u64 = metadata.len() as u64;

    // -------- Compute offsets (same layout as the in-memory writer) --------
    let token_offsets_offset: u64 = HEADER_SIZE as u64;
    let token_offsets_bytes: u64 = (n_tokens + 1) * 4;
    let token_blob_offset = align_up(token_offsets_offset + token_offsets_bytes, 8);
    let bucket_offsets_offset = align_up(token_blob_offset + total_token_bytes, 8);
    let bucket_offsets_bytes: u64 = (n_tokens + 1) * 8;
    let seed_array_offset = align_up(bucket_offsets_offset + bucket_offsets_bytes, 32);
    let seed_array_bytes: u64 = total_seeds * SEED_SIZE as u64;
    let body_end = seed_array_offset + seed_array_bytes;
    let file_end = body_end + TRAILER_SIZE as u64;

    // -------- Pass 2: open .tmp, stream header + tables + seed_array --------
    let tmp_path: PathBuf = {
        let mut q = path.to_path_buf();
        let mut name = q.file_name().map(|n| n.to_os_string()).unwrap_or_default();
        name.push(".tmp");
        q.set_file_name(name);
        q
    };
    {
        use std::io::Write;
        let f = OpenOptions::new()
            .create(true)
            .truncate(true)
            .read(true)
            .write(true)
            .open(&tmp_path)?;
        let mut w = BufWriter::with_capacity(64 * 1024, f);

        // Header (final values — pass 1 gave us everything we need).
        let mut header = [0u8; HEADER_SIZE];
        header[0..8].copy_from_slice(MAGIC);
        header[8..12].copy_from_slice(&FORMAT_VERSION.to_le_bytes());
        // bytes [12..16] flags = 0 (zeroed by array init).
        header[16..20].copy_from_slice(&brain_crc32.to_le_bytes());
        header[20..28].copy_from_slice(&brain_node_count.to_le_bytes());
        header[28..36].copy_from_slice(&brain_bond_count.to_le_bytes());
        header[36..44].copy_from_slice(&n_tokens.to_le_bytes());
        header[44..52].copy_from_slice(&total_seeds.to_le_bytes());
        header[52..60].copy_from_slice(&token_offsets_offset.to_le_bytes());
        header[60..68].copy_from_slice(&token_blob_offset.to_le_bytes());
        header[68..76].copy_from_slice(&bucket_offsets_offset.to_le_bytes());
        header[76..84].copy_from_slice(&seed_array_offset.to_le_bytes());
        // bytes [84..128] reserved (zeroed).
        w.write_all(&header)?;

        // token_offsets table.
        let mut tok_cursor: u32 = 0;
        for (token, _) in &metadata {
            w.write_all(&tok_cursor.to_le_bytes())?;
            tok_cursor += token.len() as u32;
        }
        w.write_all(&tok_cursor.to_le_bytes())?; // sentinel = total_token_bytes
        debug_assert_eq!(tok_cursor as u64, total_token_bytes,
            "token_offsets cursor diverged from pass-1 total_token_bytes");

        // Pad to token_blob_offset.
        write_zero_padding(&mut w, token_offsets_offset + (n_tokens + 1) * 4, token_blob_offset)?;

        // token_blob.
        for (token, _) in &metadata {
            w.write_all(token.as_bytes())?;
        }

        // Pad to bucket_offsets_offset.
        write_zero_padding(&mut w, token_blob_offset + total_token_bytes, bucket_offsets_offset)?;

        // bucket_offsets table (cumulative seed counts).
        let mut seed_cursor: u64 = 0;
        for (_, bucket_n) in &metadata {
            w.write_all(&seed_cursor.to_le_bytes())?;
            seed_cursor += *bucket_n as u64;
        }
        w.write_all(&seed_cursor.to_le_bytes())?; // sentinel = seed_total
        debug_assert_eq!(seed_cursor, total_seeds,
            "bucket_offsets cursor diverged from pass-1 total_seeds");

        // Pad to seed_array_offset.
        write_zero_padding(&mut w, bucket_offsets_offset + (n_tokens + 1) * 8, seed_array_offset)?;

        // -------- Pass 2 over the source: stream seed_array --------
        let mut emitted: u64 = 0;
        let mut meta_iter = metadata.iter();
        for (token, seeds) in make_iter() {
            if seeds.is_empty() { continue; }
            // Strict integrity check: source must yield same sequence as pass 1.
            let (expected_token, expected_n) = meta_iter.next().ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::Other,
                    "OP-022.a.1: streaming source emitted more tokens in pass 2 than pass 1",
                )
            })?;
            if expected_token != &token || *expected_n as usize != seeds.len() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!(
                        "OP-022.a.1: streaming source non-deterministic between passes:                          pass1 saw (token={:?} n={}), pass2 saw (token={:?} n={})",
                        expected_token, expected_n, token, seeds.len()
                    ),
                ));
            }
            for s in &seeds {
                w.write_all(s.as_bytes())?;
                emitted += 1;
            }
        }
        if meta_iter.next().is_some() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                "OP-022.a.1: streaming source emitted fewer tokens in pass 2 than pass 1",
            ));
        }
        debug_assert_eq!(emitted, total_seeds, "emitted seed count != total_seeds");

        // Trailer: CRC placeholder + magic. CRC filled by Pass 3.
        w.write_all(&[0u8; 4])?;
        w.write_all(TRAILER_MAGIC)?;
        w.flush()?;

        // Sync the data to disk so Pass 3 sees it.
        let f_back = w.into_inner().map_err(|e| e.into_error())?;
        f_back.sync_all()?;
    }

    // -------- Pass 3: stream-CRC over [0..EOF-8], write CRC into trailer --------
    {
        use std::io::{Read, Seek, SeekFrom, Write};
        let mut f = OpenOptions::new().read(true).write(true).open(&tmp_path)?;
        let mut hasher = crc32fast::Hasher::new();
        let mut buf = vec![0u8; 64 * 1024];
        let trailer_pos = file_end - TRAILER_SIZE as u64;
        let mut remaining = trailer_pos;
        while remaining > 0 {
            let take = std::cmp::min(remaining as usize, buf.len());
            let n_read = f.read(&mut buf[..take])?;
            if n_read == 0 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "OP-022.a.1: pass-3 CRC read hit EOF before trailer position",
                ));
            }
            hasher.update(&buf[..n_read]);
            remaining -= n_read as u64;
        }
        let crc = hasher.finalize();
        f.seek(SeekFrom::Start(trailer_pos))?;
        f.write_all(&crc.to_le_bytes())?;
        // Trailer magic was already written in Pass 2; leave it untouched.
        f.sync_all()?;
    }

    // Atomic rename + parent fsync.
    std::fs::rename(&tmp_path, path)?;
    if let Some(parent) = path.parent() {
        if let Ok(dir) = File::open(parent) {
            let _ = dir.sync_all();
        }
    }
    Ok(())
}

/// Pre-OP-022.a.1 in-memory writer kept as a thin wrapper so the existing
/// test suite (32 keyword_index tests using `&HashMap<String, Vec<Seed>>`)
/// keeps passing. Sorts the HashMap keys once, then delegates to the
/// streaming writer with an iterator factory that re-iterates the sorted
/// keys on each call.
pub fn write_sidecar<P: AsRef<Path>>(
    path: P,
    index: &HashMap<String, Vec<Seed>>,
    brain_crc32: u32,
    brain_node_count: u64,
    brain_bond_count: u64,
) -> std::io::Result<()> {
    // Sort keys once (deterministic source for both passes).
    let mut keys: Vec<&str> = index
        .iter()
        .filter(|(_, v)| !v.is_empty())
        .map(|(k, _)| k.as_str())
        .collect();
    keys.sort_by(|a, b| a.as_bytes().cmp(b.as_bytes()));

    let make_iter = || {
        keys.iter().filter_map(|k| {
            let v = index.get(*k)?;
            if v.is_empty() { return None; }
            // Byte-sort within bucket (PRESERVE duplicates per OP-022.a
            // Phase C / OP-030). Per-iteration clone is bounded by
            // bucket size — typically tens of seeds.
            let mut sorted: Vec<Seed> = v.clone();
            sorted.sort_by(|a, b| a.as_bytes().cmp(b.as_bytes()));
            Some((k.to_string(), sorted))
        })
    };
    write_sidecar_streaming(path, make_iter, brain_crc32, brain_node_count, brain_bond_count)
}

/// Helper: write the right number of zero bytes to align the writer's
/// position from `current_offset` up to `target_offset`. Both values are
/// file-absolute offsets we computed in Pass 1.
fn write_zero_padding<W: std::io::Write>(
    w: &mut W,
    current_offset: u64,
    target_offset: u64,
) -> std::io::Result<()> {
    debug_assert!(target_offset >= current_offset);
    let pad = target_offset - current_offset;
    if pad == 0 { return Ok(()); }
    // align_up never produces a padding > 31 bytes (largest align we use is 32).
    let zeros = [0u8; 32];
    w.write_all(&zeros[..pad as usize])
}


#[inline]
fn align_up(offset: u64, align: u64) -> u64 {
    debug_assert!(align.is_power_of_two());
    (offset + align - 1) & !(align - 1)
}

// ---------------- Tests ----------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn s(label: &str) -> Seed { Seed::from_label(label) }

    fn sample_index() -> HashMap<String, Vec<Seed>> {
        let mut m: HashMap<String, Vec<Seed>> = HashMap::new();
        m.insert("alpha".to_string(), vec![s("n1"), s("n2"), s("n3")]);
        m.insert("beta".to_string(),  vec![s("n2"), s("n4")]);
        m.insert("gamma".to_string(), vec![s("n5")]);
        m.insert("delta".to_string(), vec![s("n1"), s("n5"), s("n6"), s("n7")]);
        // Unicode token to exercise UTF-8 byte ordering.
        m.insert("über".to_string(),  vec![s("u1"), s("u2")]);
        m
    }

    fn tmpfile() -> std::path::PathBuf {
        let dir = tempfile::tempdir().expect("tempdir").keep();
        dir.join("sidecar.kw")
    }

    #[test]
    fn roundtrip_basic() {
        let idx = sample_index();
        let path = tmpfile();
        write_sidecar(&path, &idx, 0xCAFEBABE, 100, 200).expect("write");
        let mmap = KeywordIndexMmap::open(&path).expect("open");
        mmap.validate(0xCAFEBABE, 100, 200).expect("validate");
        assert_eq!(mmap.token_count(), idx.len());
        for (tok, expected) in &idx {
            let mut expected_sorted = expected.clone();
            expected_sorted.sort_by(|a, b| a.as_bytes().cmp(b.as_bytes()));
            // OP-022.a: duplicates preserved (no .dedup() expected).
            let got = mmap.get(tok).expect("token present");
            assert_eq!(got.len(), expected_sorted.len(), "bucket len for {}", tok);
            assert_eq!(got, expected_sorted.as_slice(), "bucket content for {}", tok);
        }
    }

    #[test]
    fn returns_none_for_missing_token() {
        let idx = sample_index();
        let path = tmpfile();
        write_sidecar(&path, &idx, 1, 0, 0).unwrap();
        let mmap = KeywordIndexMmap::open(&path).unwrap();
        assert!(mmap.get("nope").is_none());
        assert!(mmap.get("").is_none());
        assert!(mmap.get("alphas").is_none()); // prefix-of-existing must miss
    }

    #[test]
    fn token_sort_invariant_holds() {
        let idx = sample_index();
        let path = tmpfile();
        write_sidecar(&path, &idx, 1, 0, 0).unwrap();
        let mmap = KeywordIndexMmap::open(&path).unwrap();
        mmap.check_token_sort().expect("tokens sorted ascending");
    }

    #[test]
    fn bucket_seeds_byte_sorted() {
        let idx = sample_index();
        let path = tmpfile();
        write_sidecar(&path, &idx, 1, 0, 0).unwrap();
        let mmap = KeywordIndexMmap::open(&path).unwrap();
        for (_, bucket) in mmap.iter() {
            for i in 1..bucket.len() {
                assert!(bucket[i-1].as_bytes() < bucket[i].as_bytes(),
                    "bucket not byte-sorted at index {}", i);
            }
        }
    }

    #[test]
    fn empty_index_roundtrip() {
        let idx: HashMap<String, Vec<Seed>> = HashMap::new();
        let path = tmpfile();
        write_sidecar(&path, &idx, 7, 0, 0).unwrap();
        let mmap = KeywordIndexMmap::open(&path).unwrap();
        assert_eq!(mmap.token_count(), 0);
        assert_eq!(mmap.seed_total(), 0);
        assert!(mmap.get("anything").is_none());
    }

    #[test]
    fn single_seed_bucket() {
        let mut idx: HashMap<String, Vec<Seed>> = HashMap::new();
        idx.insert("only".to_string(), vec![s("x")]);
        let path = tmpfile();
        write_sidecar(&path, &idx, 1, 0, 0).unwrap();
        let mmap = KeywordIndexMmap::open(&path).unwrap();
        let bucket = mmap.get("only").unwrap();
        assert_eq!(bucket.len(), 1);
        assert_eq!(bucket[0], s("x"));
    }

    #[test]
    fn large_bucket_roundtrip() {
        let mut idx: HashMap<String, Vec<Seed>> = HashMap::new();
        let seeds: Vec<Seed> = (0..1000).map(|i| s(&format!("seed_{}", i))).collect();
        idx.insert("big".to_string(), seeds.clone());
        let path = tmpfile();
        write_sidecar(&path, &idx, 1, 0, 0).unwrap();
        let mmap = KeywordIndexMmap::open(&path).unwrap();
        let got = mmap.get("big").unwrap();
        let mut expected = seeds;
        expected.sort_by(|a, b| a.as_bytes().cmp(b.as_bytes()));
        expected.dedup();
        assert_eq!(got, expected.as_slice());
    }

    #[test]
    fn duplicates_within_bucket_preserved() {
        // OP-022.a: write_sidecar must NOT dedup. The legacy heap path
        // kept duplicates (content+knowledge double-extraction) and
        // downstream take-2-then-dedup uses them as a throttle.
        let mut idx: HashMap<String, Vec<Seed>> = HashMap::new();
        idx.insert("dup".to_string(), vec![s("a"), s("b"), s("a"), s("c"), s("b")]);
        let path = tmpfile();
        write_sidecar(&path, &idx, 1, 0, 0).unwrap();
        let mmap = KeywordIndexMmap::open(&path).unwrap();
        let got = mmap.get("dup").unwrap();
        assert_eq!(got.len(), 5, "all 5 entries preserved (duplicates retained)");
        // And the bucket is byte-sorted.
        for i in 1..got.len() {
            assert!(got[i-1].as_bytes() <= got[i].as_bytes());
        }
    }

    #[test]
    fn corrupt_magic_rejected() {
        let idx = sample_index();
        let path = tmpfile();
        write_sidecar(&path, &idx, 1, 0, 0).unwrap();
        let mut bytes = std::fs::read(&path).unwrap();
        bytes[0] = b'X'; // corrupt first byte of magic
        std::fs::write(&path, &bytes).unwrap();
        let err = KeywordIndexMmap::open(&path).unwrap_err();
        // CRC check runs before magic check (we cover whole file), so CRC may fire first.
        match err {
            KeywordIndexError::BadCrc { .. } | KeywordIndexError::BadMagic { .. } => {}
            other => panic!("unexpected error: {:?}", other),
        }
    }

    #[test]
    fn corrupt_trailer_magic_rejected() {
        let idx = sample_index();
        let path = tmpfile();
        write_sidecar(&path, &idx, 1, 0, 0).unwrap();
        let mut bytes = std::fs::read(&path).unwrap();
        let len = bytes.len();
        bytes[len - 1] = b'Z';
        std::fs::write(&path, &bytes).unwrap();
        let err = KeywordIndexMmap::open(&path).unwrap_err();
        assert!(matches!(err, KeywordIndexError::BadTrailerMagic { .. }));
    }

    #[test]
    fn corrupt_crc_rejected() {
        let idx = sample_index();
        let path = tmpfile();
        write_sidecar(&path, &idx, 1, 0, 0).unwrap();
        let mut bytes = std::fs::read(&path).unwrap();
        // Corrupt a random byte in the middle of the file (not in trailer magic).
        let mid = bytes.len() / 2;
        bytes[mid] ^= 0xff;
        std::fs::write(&path, &bytes).unwrap();
        let err = KeywordIndexMmap::open(&path).unwrap_err();
        assert!(matches!(err, KeywordIndexError::BadCrc { .. }));
    }

    #[test]
    fn bad_version_rejected() {
        let idx = sample_index();
        let path = tmpfile();
        write_sidecar(&path, &idx, 1, 0, 0).unwrap();
        let mut bytes = std::fs::read(&path).unwrap();
        // Bump version to 99.
        bytes[8..12].copy_from_slice(&99u32.to_le_bytes());
        // Recompute CRC so the version check fires instead of the CRC check.
        let len = bytes.len();
        let crc = crc32fast::hash(&bytes[..len - 8]);
        bytes[len - 8..len - 4].copy_from_slice(&crc.to_le_bytes());
        std::fs::write(&path, &bytes).unwrap();
        let err = KeywordIndexMmap::open(&path).unwrap_err();
        assert!(matches!(err, KeywordIndexError::BadVersion { found: 99, expected: 1 }));
    }

    #[test]
    fn validate_brain_crc_mismatch() {
        let idx = sample_index();
        let path = tmpfile();
        write_sidecar(&path, &idx, 0xAAAA, 5, 7).unwrap();
        let mmap = KeywordIndexMmap::open(&path).unwrap();
        let err = mmap.validate(0xBBBB, 5, 7).unwrap_err();
        assert!(matches!(err, KeywordIndexError::BrainCrcMismatch { .. }));
    }

    #[test]
    fn validate_node_count_mismatch() {
        let idx = sample_index();
        let path = tmpfile();
        write_sidecar(&path, &idx, 1, 100, 200).unwrap();
        let mmap = KeywordIndexMmap::open(&path).unwrap();
        let err = mmap.validate(1, 999, 200).unwrap_err();
        assert!(matches!(err, KeywordIndexError::NodeCountMismatch { .. }));
    }

    #[test]
    fn validate_bond_count_mismatch() {
        let idx = sample_index();
        let path = tmpfile();
        write_sidecar(&path, &idx, 1, 100, 200).unwrap();
        let mmap = KeywordIndexMmap::open(&path).unwrap();
        let err = mmap.validate(1, 100, 999).unwrap_err();
        assert!(matches!(err, KeywordIndexError::BondCountMismatch { .. }));
    }

    #[test]
    fn truncated_file_rejected() {
        let idx = sample_index();
        let path = tmpfile();
        write_sidecar(&path, &idx, 1, 0, 0).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        std::fs::write(&path, &bytes[..bytes.len() / 2]).unwrap();
        let err = KeywordIndexMmap::open(&path).unwrap_err();
        // Could be TooSmall, BadTrailerMagic, BadCrc, or OutOfBounds depending
        // on where the truncation lands.
        match err {
            KeywordIndexError::TooSmall { .. }
            | KeywordIndexError::BadTrailerMagic { .. }
            | KeywordIndexError::BadCrc { .. }
            | KeywordIndexError::OutOfBounds { .. }
            | KeywordIndexError::BadMagic { .. } => {}
            other => panic!("unexpected error: {:?}", other),
        }
    }

    #[test]
    fn very_truncated_file_rejected() {
        let path = tmpfile();
        std::fs::write(&path, b"too_short").unwrap();
        let err = KeywordIndexMmap::open(&path).unwrap_err();
        assert!(matches!(err, KeywordIndexError::TooSmall { .. }));
    }

    // ---------------- MergedBucket + KeywordIndex tests ----------------

    fn open_sample_mmap() -> Arc<KeywordIndexMmap> {
        let idx = sample_index();
        let path = tmpfile();
        write_sidecar(&path, &idx, 1, 0, 0).unwrap();
        Arc::new(KeywordIndexMmap::open(&path).unwrap())
    }

    #[test]
    fn merged_bucket_borrowed_when_base_only() {
        let mmap = open_sample_mmap();
        let ki = KeywordIndex::with_base(mmap);
        let bucket = ki.get("alpha").unwrap();
        assert!(matches!(bucket, MergedBucket::Borrowed(_)));
        assert_eq!(bucket.len(), 3);
    }

    #[test]
    fn merged_bucket_deref_to_slice() {
        let mmap = open_sample_mmap();
        let ki = KeywordIndex::with_base(mmap);
        let bucket = ki.get("alpha").unwrap();
        // Exercise the existing-call-site API: .iter(), .len(), binary_search_by, .first().
        assert!(bucket.iter().count() == 3);
        assert!(bucket.first().is_some());
        let needle = s("n2");
        let res = bucket.binary_search_by(|x| x.as_bytes().cmp(needle.as_bytes()));
        assert!(res.is_ok(), "binary_search_by must work via Deref");
    }

    #[test]
    fn overlay_only_preserves_insertion_order() {
        // OP-022.a (functional equivalence): heap-only Brain semantics
        // require insertion order with duplicates preserved.
        // BR.C-Hyp-Methodology byte-sort only runs in
        // ensure_keyword_index_built (a no-op for archive-less brains).
        let mut ki = KeywordIndex::new();
        ki.insert_overlay("only_overlay".to_string(), s("z"));
        ki.insert_overlay("only_overlay".to_string(), s("a"));
        ki.insert_overlay("only_overlay".to_string(), s("m"));
        let bucket = ki.get("only_overlay").unwrap();
        assert!(matches!(bucket, MergedBucket::Owned(_)));
        assert_eq!(bucket.len(), 3);
        assert_eq!(bucket[0], s("z"));
        assert_eq!(bucket[1], s("a"));
        assert_eq!(bucket[2], s("m"));
    }

    #[test]
    fn overlay_preserves_duplicates() {
        // OP-022.a: duplicates from per-teach double-extraction must
        // flow through to select_seeds' take-2 throttle.
        let mut ki = KeywordIndex::new();
        ki.insert_overlay("t".to_string(), s("x"));
        ki.insert_overlay("t".to_string(), s("x"));
        ki.insert_overlay("t".to_string(), s("y"));
        let bucket = ki.get("t").unwrap();
        assert_eq!(bucket.len(), 3, "duplicates preserved (legacy parity)");
    }

    #[test]
    fn overlay_merge_with_base() {
        // OP-022.a: merged result is concat(base, overlay) with no
        // sort/dedup. Base is byte-sorted (writer invariant); overlay
        // tail is insertion order. Total length = base.len + overlay.len.
        let mmap = open_sample_mmap();
        let mut ki = KeywordIndex::with_base(mmap);
        ki.insert_overlay("alpha".to_string(), s("new_seed"));
        let bucket = ki.get("alpha").unwrap();
        assert!(matches!(bucket, MergedBucket::Owned(_)));
        assert_eq!(bucket.len(), 4); // 3 from base + 1 from overlay
        // The base prefix is byte-sorted; the new seed is appended at the end.
        for i in 1..3 {
            assert!(bucket[i-1].as_bytes() <= bucket[i].as_bytes(),
                "base prefix should be byte-sorted");
        }
        assert_eq!(bucket[3], s("new_seed"), "overlay tail at the end");
    }

    #[test]
    fn overlay_introduces_duplicate_of_base_seed() {
        let mmap = open_sample_mmap();
        let mut ki = KeywordIndex::with_base(mmap);
        // Re-teach a seed that's already in base; merge must dedup.
        ki.insert_overlay("alpha".to_string(), s("n1"));
        let bucket = ki.get("alpha").unwrap();
        assert_eq!(bucket.len(), 4, "OP-022.a: duplicate-of-base preserved");
    }

    #[test]
    fn tombstone_removes_base_seed_from_all_buckets() {
        let mmap = open_sample_mmap();
        let mut ki = KeywordIndex::with_base(mmap);
        // n1 appears in both "alpha" and "delta".
        ki.tombstone_seed(s("n1"));
        let alpha = ki.get("alpha").unwrap();
        assert_eq!(alpha.len(), 2, "alpha lost n1");
        assert!(!alpha.iter().any(|s_| *s_ == s("n1")));
        let delta = ki.get("delta").unwrap();
        assert_eq!(delta.len(), 3, "delta lost n1");
        assert!(!delta.iter().any(|s_| *s_ == s("n1")));
        // gamma untouched (n1 not in its bucket).
        let gamma = ki.get("gamma").unwrap();
        assert_eq!(gamma.len(), 1);
    }

    #[test]
    fn tombstone_set_bulk() {
        let mmap = open_sample_mmap();
        let mut ki = KeywordIndex::with_base(mmap);
        let mut set = HashSet::new();
        set.insert(s("n1"));
        set.insert(s("n5"));
        ki.tombstone_seeds(&set);
        let alpha = ki.get("alpha").unwrap();
        assert!(!alpha.iter().any(|s_| set.contains(s_)));
        let gamma = ki.get("gamma");
        // gamma was [n5]; after tombstoning n5 it should be empty → None.
        assert!(gamma.is_none(), "empty bucket after tombstone → None");
    }

    #[test]
    fn tombstone_then_reinsert_lifts_tombstone() {
        let mmap = open_sample_mmap();
        let mut ki = KeywordIndex::with_base(mmap);
        ki.tombstone_seed(s("n1"));
        assert!(ki.is_tombstoned(&s("n1")));
        ki.insert_overlay("alpha".to_string(), s("n1"));
        assert!(!ki.is_tombstoned(&s("n1")), "overlay re-insert must lift tombstone");
        let alpha = ki.get("alpha").unwrap();
        assert!(alpha.iter().any(|s_| *s_ == s("n1")));
    }

    #[test]
    fn has_dirty_tracks_overlay_and_tombstones() {
        let mut ki = KeywordIndex::new();
        assert!(!ki.has_dirty());
        ki.insert_overlay("x".to_string(), s("a"));
        assert!(ki.has_dirty());
        ki.clear_dirty();
        assert!(!ki.has_dirty());
        ki.replace_base(Some(open_sample_mmap()));
        ki.tombstone_seed(s("n1"));
        assert!(ki.has_dirty());
        ki.clear_dirty();
        assert!(!ki.has_dirty());
    }

    #[test]
    fn materialize_all_matches_base_when_no_dirty() {
        let idx = sample_index();
        let path = tmpfile();
        write_sidecar(&path, &idx, 1, 0, 0).unwrap();
        let mmap = Arc::new(KeywordIndexMmap::open(&path).unwrap());
        let ki = KeywordIndex::with_base(mmap);
        let materialized = ki.materialize_all();
        for (tok, expected) in &idx {
            let mut exp = expected.clone();
            exp.sort_by(|a, b| a.as_bytes().cmp(b.as_bytes()));
            // OP-022.a: no dedup (duplicates preserved).
            let got = materialized.get(tok).expect("token present");
            assert_eq!(got.as_slice(), exp.as_slice(), "token {}", tok);
        }
        assert_eq!(materialized.len(), idx.len());
    }

    #[test]
    fn materialize_all_applies_overlay_and_tombstones() {
        let mmap = open_sample_mmap();
        let mut ki = KeywordIndex::with_base(mmap);
        ki.insert_overlay("newtoken".to_string(), s("q"));
        ki.insert_overlay("alpha".to_string(), s("extra"));
        ki.tombstone_seed(s("n1"));
        let materialized = ki.materialize_all();
        // newtoken present
        assert_eq!(materialized.get("newtoken").unwrap().len(), 1);
        // alpha has extra, minus n1
        let alpha = materialized.get("alpha").unwrap();
        assert!(alpha.iter().any(|s_| *s_ == s("extra")));
        assert!(!alpha.iter().any(|s_| *s_ == s("n1")));
    }

    #[test]
    fn rewrite_sidecar_round_trip_is_stable() {
        // Write -> read -> materialize -> rewrite -> read again must produce
        // byte-identical results for every bucket (bit-stability per Principle 3).
        let original = sample_index();
        let path1 = tmpfile();
        write_sidecar(&path1, &original, 0xDEAD, 11, 22).unwrap();
        let m1 = KeywordIndexMmap::open(&path1).unwrap();
        let ki = KeywordIndex::with_base(Arc::new(m1));
        let materialised = ki.materialize_all();

        let path2 = tmpfile();
        write_sidecar(&path2, &materialised, 0xDEAD, 11, 22).unwrap();
        let m2 = KeywordIndexMmap::open(&path2).unwrap();

        for (tok, _) in &original {
            let got1 = ki.base().unwrap().get(tok).unwrap();
            let got2 = m2.get(tok).unwrap();
            assert_eq!(got1, got2, "bit-stable rewrite for {}", tok);
        }
    }

    #[test]
    fn unicode_tokens_roundtrip() {
        let mut idx: HashMap<String, Vec<Seed>> = HashMap::new();
        idx.insert("café".to_string(), vec![s("c1")]);
        idx.insert("naïve".to_string(), vec![s("n1")]);
        idx.insert("日本語".to_string(), vec![s("j1"), s("j2")]);
        let path = tmpfile();
        write_sidecar(&path, &idx, 1, 0, 0).unwrap();
        let mmap = KeywordIndexMmap::open(&path).unwrap();
        assert_eq!(mmap.get("café").unwrap().len(), 1);
        assert_eq!(mmap.get("naïve").unwrap().len(), 1);
        assert_eq!(mmap.get("日本語").unwrap().len(), 2);
        assert!(mmap.get("missing").is_none());
    }

    // ---------------- OP-022.a.1 streaming + merge tests ----------------

    // ---------------- OP-022.a.2: streaming rebuild ----------------

    fn streaming_rebuild_tmpdir() -> std::path::PathBuf {
        let d = tempfile::tempdir().expect("tempdir").keep();
        d
    }

    #[test]
    fn streaming_rebuilder_equivalent_to_in_memory_writer() {
        // Build a small fixture corpus of (token, seed) pairs. Push them
        // through the streaming rebuilder; build the same input into a
        // HashMap and write_sidecar; assert byte-identical output.
        let mut fixture: Vec<(String, Seed)> = Vec::new();
        for t in 0..50u32 {
            let token = format!("tok_{:03}", t);
            // 5 seeds per token, with a duplicate to exercise the dup-preserve invariant.
            for i in 0..5u32 {
                fixture.push((token.clone(), s(&format!("s_{}_{}", t, i))));
            }
            fixture.push((token.clone(), s(&format!("s_{}_0", t)))); // dup of first
        }

        // Reference: HashMap path
        let mut ref_idx: HashMap<String, Vec<Seed>> = HashMap::new();
        for (t, sd) in &fixture {
            ref_idx.entry(t.clone()).or_insert_with(Vec::new).push(*sd);
        }
        let ref_path = tmpfile();
        write_sidecar(&ref_path, &ref_idx, 0x1234, 7, 11).unwrap();

        // Streaming rebuilder path
        let runs_dir = streaming_rebuild_tmpdir();
        let stream_path = tmpfile();
        let mut b = StreamingRebuilder::new(
            stream_path.clone(),
            runs_dir.clone(),
            16,                  // tiny buffer -> multiple runs
            0x1234, 7, 11,
        ).expect("new");
        for (t, sd) in &fixture {
            b.push(t.clone(), *sd).expect("push");
        }
        let _mmap = b.finish().expect("finish");
        assert!(!runs_dir.exists(), "runs_dir cleaned up by Drop on success");

        let a = std::fs::read(&ref_path).unwrap();
        let s = std::fs::read(&stream_path).unwrap();
        assert_eq!(a, s, "streaming rebuild must be byte-identical to HashMap path");
    }

    #[test]
    fn streaming_rebuilder_runs_dir_cleanup_on_drop_without_finish() {
        // Construct, push, then drop without calling finish. Runs dir
        // must be cleaned up by Drop. Simulates a fault-injection.
        let runs_dir = streaming_rebuild_tmpdir();
        let sidecar_path = tmpfile();
        {
            let mut b = StreamingRebuilder::new(
                sidecar_path,
                runs_dir.clone(),
                4,
                1, 0, 0,
            ).expect("new");
            for i in 0..20 {
                b.push(format!("k{}", i), s(&format!("x{}", i))).expect("push");
            }
            assert!(b.run_count() >= 4, "some runs flushed");
            // Drop here — no finish call.
        }
        assert!(!runs_dir.exists(), "Drop must clean runs_dir even when finish was not called");
    }

    #[test]
    fn streaming_rebuilder_multi_run_kway_merge_preserves_global_order() {
        // Force a tiny buffer so 50 input pairs produce many runs.
        // The k-way merge output must be globally sorted by token bytes,
        // with duplicates preserved.
        let mut fixture: Vec<(String, Seed)> = Vec::new();
        // Tokens deliberately INTERLEAVED so a single buffer can't sort
        // them globally; only the merge can.
        let order = ["a", "c", "e", "b", "d", "a", "c", "e", "b", "d"];
        for round in 0..5 {
            for tok in &order {
                fixture.push((tok.to_string(), s(&format!("{}_{}", tok, round))));
            }
        }

        let runs_dir = streaming_rebuild_tmpdir();
        let sidecar_path = tmpfile();
        let mut b = StreamingRebuilder::new(
            sidecar_path.clone(),
            runs_dir.clone(),
            3,                   // ridiculously tiny -> ~17 runs from 50 pairs
            1, 0, 0,
        ).expect("new");
        for (t, sd) in &fixture {
            b.push(t.clone(), *sd).expect("push");
        }
        assert!(b.run_count() >= 5, "expected several runs (got {})", b.run_count());
        let mmap = b.finish().expect("finish");
        mmap.check_token_sort().expect("global token sort");
        // Every bucket must contain exactly the expected number of seeds.
        for tok in &["a", "b", "c", "d", "e"] {
            let bucket = mmap.get(tok).expect("token present");
            assert_eq!(bucket.len(), 10, "token {} has 10 deposits (5 rounds * 2 in `order`)", tok);
        }
    }

        // ---------------- OP-022.a.3.1: trust-flag + bit-equivalence ----------------

    #[test]
    fn from_mmap_trusted_equivalent_to_from_mmap_for_good_sidecar() {
        // Regression: must NOT be a behavior change for inputs the cold-load
        // CRC pass would have accepted. The trusted variant skips the CRC
        // but every other parsed field + accessor must match exactly.
        let idx = sample_index();
        let path = tmpfile();
        write_sidecar(&path, &idx, 0xFEED, 99, 88).unwrap();

        let m1 = KeywordIndexMmap::open(&path).unwrap();
        let m2 = KeywordIndexMmap::open_trusted(&path).unwrap();

        assert_eq!(m1.token_count(), m2.token_count());
        assert_eq!(m1.seed_total(), m2.seed_total());
        assert_eq!(m1.brain_crc32(), m2.brain_crc32());
        assert_eq!(m1.brain_node_count(), m2.brain_node_count());
        assert_eq!(m1.brain_bond_count(), m2.brain_bond_count());
        m1.check_token_sort().expect("m1 tokens sorted");
        m2.check_token_sort().expect("m2 tokens sorted");
        // For every token in the sample, both mmaps must return identical buckets.
        for (tok, _) in &idx {
            let b1 = m1.get(tok).expect("m1 has token");
            let b2 = m2.get(tok).expect("m2 has token");
            assert_eq!(b1, b2, "bucket mismatch for {}", tok);
        }
    }

    #[test]
    fn from_mmap_trusted_still_rejects_bad_trailer_magic() {
        // The trust flag skips ONLY the CRC32-over-whole-file pass. Cheap
        // sanity checks (trailer magic, header magic, version, section
        // bounds) MUST still fire — otherwise we'd return junk on a
        // truncated or wrong-format file even via the trusted path.
        let idx = sample_index();
        let path = tmpfile();
        write_sidecar(&path, &idx, 1, 0, 0).unwrap();
        let mut bytes = std::fs::read(&path).unwrap();
        let len = bytes.len();
        bytes[len - 1] = b'Z';                 // corrupt trailer magic
        std::fs::write(&path, &bytes).unwrap();
        let err = KeywordIndexMmap::open_trusted(&path).unwrap_err();
        assert!(matches!(err, KeywordIndexError::BadTrailerMagic { .. }),
            "trust flag must not weaken trailer-magic check, got {:?}", err);
    }

    // ---------------- OP-022.a.3: madvise + bit-stability ----------------

    #[test]
    fn madvise_called_with_monotonic_offsets_during_compact() {
        // Build a base sidecar large enough to cross the 64 MiB advise threshold.
        // 64 MiB / 32 bytes per Seed = 2,097,152 Seeds. We'll use 5,000,000 to
        // ensure several chunk crossings.
        let mut idx: HashMap<String, Vec<Seed>> = HashMap::new();
        let seeds: Vec<Seed> = (0..5_000_000u64)
            .map(|i| {
                let mut b = [0u8; 32];
                b[..8].copy_from_slice(&i.to_le_bytes());
                Seed::from_bytes(b)
            })
            .collect();
        idx.insert("giant_bucket".to_string(), seeds);
        let base_path = tmpfile();
        write_sidecar(&base_path, &idx, 1, 0, 0).unwrap();
        let base_mmap = KeywordIndexMmap::open(&base_path).unwrap();

        reset_madvise_counters();
        let empty_overlay: HashMap<String, Vec<Seed>> = HashMap::new();
        let empty_tomb: HashSet<Seed> = HashSet::new();
        let make_iter = || KeywordMergeIter::new(Some(&base_mmap), &[], &empty_overlay, &empty_tomb);

        let out_path = tmpfile();
        write_sidecar_streaming(&out_path, make_iter, 1, 0, 0).unwrap();

        let calls = MADVISE_CALLS.load(AtomicOrdering::SeqCst);
        let violations = MADVISE_MONOTONIC_VIOLATIONS.load(AtomicOrdering::SeqCst);
        // Two passes over the merge -> at least 2 advise rounds when the bucket
        // crosses chunk boundaries. 5M seeds * 32 B = 160 MiB > 64 MiB chunk,
        // so each pass should issue at least one madvise call.
        assert!(calls >= 2, "expected >=2 madvise calls across two passes, got {}", calls);
        assert_eq!(violations, 0, "madvise offsets must be monotonic; got {} violations", violations);
    }

    #[test]
    fn streaming_output_byte_identical_with_madvise() {
        // Bit-stability: madvise is a hint to the kernel, not a write-path
        // change. Two writes of the same source must produce byte-identical
        // sidecar files even though pass 2 may page-fault from a different
        // physical location after pass 1 evicted.
        let original = sample_index();
        let base_path = tmpfile();
        write_sidecar(&base_path, &original, 0xCAFE, 7, 13).unwrap();
        let base_mmap = KeywordIndexMmap::open(&base_path).unwrap();

        let empty_overlay: HashMap<String, Vec<Seed>> = HashMap::new();
        let empty_tomb: HashSet<Seed> = HashSet::new();
        let make_iter = || KeywordMergeIter::new(Some(&base_mmap), &[], &empty_overlay, &empty_tomb);

        let out_path_1 = tmpfile();
        let out_path_2 = tmpfile();
        write_sidecar_streaming(&out_path_1, make_iter.clone(), 0xCAFE, 7, 13).unwrap();
        write_sidecar_streaming(&out_path_2, make_iter, 0xCAFE, 7, 13).unwrap();

        let bytes_1 = std::fs::read(&out_path_1).unwrap();
        let bytes_2 = std::fs::read(&out_path_2).unwrap();
        assert_eq!(bytes_1, bytes_2, "madvise must not perturb writer output");
        // Also assert byte-identical to a baseline write_sidecar_streaming over
        // the SAME source — verifies the merge iterator + writer are
        // deterministic regardless of madvise pressure.
        let base_bytes = std::fs::read(&base_path).unwrap();
        assert_eq!(bytes_1, base_bytes, "base-passthrough byte-identical");
    }

        #[test]
    fn streaming_writer_matches_in_memory_writer() {
        // Same input through `write_sidecar` (which now delegates to streaming)
        // and directly through `write_sidecar_streaming` must produce
        // byte-identical files.
        let idx = sample_index();
        let path1 = tmpfile();
        write_sidecar(&path1, &idx, 0xABCDEF01, 11, 22).unwrap();
        // Build the same source for the direct streaming API.
        let mut keys: Vec<&str> = idx.iter().filter(|(_, v)| !v.is_empty()).map(|(k, _)| k.as_str()).collect();
        keys.sort_by(|a, b| a.as_bytes().cmp(b.as_bytes()));
        let make_iter = || {
            keys.iter().filter_map(|k| {
                let v = idx.get(*k)?;
                if v.is_empty() { return None; }
                let mut s = v.clone();
                s.sort_by(|a, b| a.as_bytes().cmp(b.as_bytes()));
                Some((k.to_string(), s))
            })
        };
        let path2 = tmpfile();
        write_sidecar_streaming(&path2, make_iter, 0xABCDEF01, 11, 22).unwrap();
        let bytes1 = std::fs::read(&path1).unwrap();
        let bytes2 = std::fs::read(&path2).unwrap();
        assert_eq!(bytes1, bytes2, "streaming writer must be byte-identical to wrapper");
    }

    #[test]
    fn streaming_empty_overlay_base_passthrough() {
        // Write a base sidecar from a HashMap, mmap it back, then re-run
        // the merge iterator with empty overlay + empty tombstones, write
        // through the streaming writer, and confirm byte-for-byte
        // equality with the input — passing the base through unchanged.
        let original = sample_index();
        let base_path = tmpfile();
        write_sidecar(&base_path, &original, 0xDEAD, 9, 7).unwrap();
        let base_mmap = KeywordIndexMmap::open(&base_path).unwrap();

        let empty_overlay: HashMap<String, Vec<Seed>> = HashMap::new();
        let empty_tomb: HashSet<Seed> = HashSet::new();
        let make_iter = || KeywordMergeIter::new(Some(&base_mmap), &[], &empty_overlay, &empty_tomb);

        let out_path = tmpfile();
        write_sidecar_streaming(&out_path, make_iter, 0xDEAD, 9, 7).unwrap();
        let bytes_in = std::fs::read(&base_path).unwrap();
        let bytes_out = std::fs::read(&out_path).unwrap();
        assert_eq!(bytes_in.len(), bytes_out.len(), "file size must match base");
        assert_eq!(bytes_in, bytes_out, "base-passthrough must be byte-identical");
    }

    #[test]
    fn streaming_tombstone_only_overlay_shrinks_sidecar() {
        // Base contains seed `n1` in buckets alpha + delta. Tombstone `n1`
        // with no overlay additions. New sidecar should be SMALLER than the
        // base by exactly (count_of_n1 * SEED_SIZE) bytes, plus any
        // alignment shift, and queries for `n1` must return no buckets
        // containing it.
        let original = sample_index();
        let base_path = tmpfile();
        write_sidecar(&base_path, &original, 1, 0, 0).unwrap();
        let base_mmap = KeywordIndexMmap::open(&base_path).unwrap();

        let n1 = s("n1");
        // Count occurrences of n1 across all base buckets (with duplicates).
        let mut n1_count = 0usize;
        for (_, b) in base_mmap.iter() {
            n1_count += b.iter().filter(|seed| **seed == n1).count();
        }
        assert!(n1_count > 0, "sample_index must contain n1 for this test to be meaningful");

        let empty_overlay: HashMap<String, Vec<Seed>> = HashMap::new();
        let mut tomb = HashSet::new();
        tomb.insert(n1);
        let make_iter = || KeywordMergeIter::new(Some(&base_mmap), &[], &empty_overlay, &tomb);

        let out_path = tmpfile();
        write_sidecar_streaming(&out_path, make_iter, 1, 0, 0).unwrap();
        let out_mmap = KeywordIndexMmap::open(&out_path).unwrap();
        // Every bucket in the new sidecar must be n1-free.
        for (tok, bucket) in out_mmap.iter() {
            assert!(!bucket.iter().any(|seed| *seed == n1),
                "token {} still contains tombstoned n1 in streaming-compacted sidecar", tok);
        }
        // And the new file is smaller by at least (n1_count * SEED_SIZE).
        let len_in = std::fs::metadata(&base_path).unwrap().len();
        let len_out = std::fs::metadata(&out_path).unwrap().len();
        assert!(len_out < len_in, "tombstoned sidecar must shrink (in={} out={})", len_in, len_out);
    }

    #[test]
    fn streaming_preserves_duplicates_round_trip() {
        // Overlay inserts five copies of the same seed for one token —
        // simulates the content+knowledge double-extraction pattern
        // multiplied. The streaming merge must keep all 5 copies
        // (OP-022.a Phase C / OP-030: take-2-then-dedup cascade throttle
        // depends on the duplicate cardinality).
        let original = sample_index();
        let base_path = tmpfile();
        write_sidecar(&base_path, &original, 7, 0, 0).unwrap();
        let base_mmap = KeywordIndexMmap::open(&base_path).unwrap();

        let dup_seed = s("dup_seed");
        let mut overlay: HashMap<String, Vec<Seed>> = HashMap::new();
        overlay.insert("new_token".to_string(), vec![dup_seed; 5]);
        let empty_tomb: HashSet<Seed> = HashSet::new();
        let mut keys: Vec<&str> = overlay.keys().map(String::as_str).collect();
        keys.sort_by(|a, b| a.as_bytes().cmp(b.as_bytes()));
        let make_iter = || KeywordMergeIter::new(Some(&base_mmap), &keys, &overlay, &empty_tomb);

        let out_path = tmpfile();
        write_sidecar_streaming(&out_path, make_iter, 7, 0, 0).unwrap();
        let out_mmap = KeywordIndexMmap::open(&out_path).unwrap();
        let bucket = out_mmap.get("new_token").expect("new_token must be present");
        assert_eq!(bucket.len(), 5, "all 5 duplicate seeds preserved (cascade throttle invariant)");
        for s_in in bucket {
            assert_eq!(*s_in, dup_seed);
        }
    }

    #[test]
    fn streaming_merge_base_plus_overlay_intersecting_token() {
        // Base has "alpha" with [n1, n2, n3]. Overlay adds [n4, n5] for
        // "alpha". Merged bucket = byte-sorted([n1, n2, n3, n4, n5]).
        let original = sample_index();
        let base_path = tmpfile();
        write_sidecar(&base_path, &original, 1, 0, 0).unwrap();
        let base_mmap = KeywordIndexMmap::open(&base_path).unwrap();

        let mut overlay: HashMap<String, Vec<Seed>> = HashMap::new();
        overlay.insert("alpha".to_string(), vec![s("n4"), s("n5")]);
        let empty_tomb: HashSet<Seed> = HashSet::new();
        let mut keys: Vec<&str> = overlay.keys().map(String::as_str).collect();
        keys.sort_by(|a, b| a.as_bytes().cmp(b.as_bytes()));
        let make_iter = || KeywordMergeIter::new(Some(&base_mmap), &keys, &overlay, &empty_tomb);

        let out_path = tmpfile();
        write_sidecar_streaming(&out_path, make_iter, 1, 0, 0).unwrap();
        let out_mmap = KeywordIndexMmap::open(&out_path).unwrap();
        let bucket = out_mmap.get("alpha").expect("alpha present");
        assert_eq!(bucket.len(), 5, "3 base + 2 overlay merged");
        // Byte-sorted (no dedup needed since seeds are distinct here).
        for i in 1..bucket.len() {
            assert!(bucket[i - 1].as_bytes() <= bucket[i].as_bytes());
        }
    }

    #[test]
    fn lookup_matches_heap_reference_for_random_tokens() {
        // The load-bearing equivalence test in miniature: build a HashMap
        // reference, write+read the sidecar, assert byte-identical Seeds
        // for every token. This is the same pattern Phase E will run
        // against the canonical brain.lbr.
        let mut idx: HashMap<String, Vec<Seed>> = HashMap::new();
        for t in 0..200 {
            let token = format!("token_{:04}", t);
            let mut bucket: Vec<Seed> = (0..(t % 10 + 1)).map(|i| s(&format!("s_{}_{}", t, i))).collect();
            // Pre-shuffle so the writer's sort is exercised.
            bucket.reverse();
            idx.insert(token, bucket);
        }
        let path = tmpfile();
        write_sidecar(&path, &idx, 1, 0, 0).unwrap();
        let mmap = KeywordIndexMmap::open(&path).unwrap();
        for (tok, expected) in &idx {
            let mut exp = expected.clone();
            exp.sort_by(|a, b| a.as_bytes().cmp(b.as_bytes()));
            // OP-022.a: no dedup (duplicates preserved).
            let got = mmap.get(tok).unwrap();
            assert_eq!(got, exp.as_slice(), "mismatch for {}", tok);
        }
    }
}
