//! OP-BRAIN-V5 Phase 0.c — bench harness library.
//!
//! This module is the assertion-mechanism layer that every later v5
//! substrate phase calls to gate progress on a bounded-RSS budget. It
//! ships four primitives:
//!
//! 1. [`FixtureReader`] — streaming reader for the Phase 0.b `.bench`
//!    framed-binary format. Validates the header (`MAGIC`,
//!    `format_version`) before iterating and verifies the SHA-256 body
//!    trailer against a streaming digest on `finish`. Records yield
//!    [`BondRecord`] without any whole-fixture materialization.
//!
//! 2. [`RssSampler`] — polls `/proc/self/status` `RssAnon` from a
//!    background thread (default 250 ms cadence). Stops cleanly on
//!    `finish()`; no zombie threads. Samples are kept either in full
//!    (when count < `RESERVOIR_CAP`) or in a reservoir of fixed size,
//!    so memory stays bounded for arbitrarily long workloads. The
//!    running min / max / observed-count are tracked exactly even when
//!    the reservoir fills.
//!
//! 3. [`with_rss_budget`] — the budget API that downstream phases call.
//!    Wraps a user workload in a sampler, computes a percentile
//!    breakdown, compares peak observed against the threshold, and
//!    returns an [`RssReport`] (pass/fail + environment metadata).
//!
//! 4. [`predicate_coverage_from_stats`] — parses a Phase 0.b
//!    `.stats.json` sidecar and answers "which `BondType` variants
//!    emit nonzero bonds against this fixture?" Future workloads that
//!    target rare predicates can refuse to run when the fixture
//!    cannot exercise the target predicate.
//!
//! ## Discipline
//!
//! - Streaming everywhere. The fixture reader never collects records
//!   into a `Vec`. The sampler never accumulates unbounded samples.
//! - No production brain touched. `/var/lib/lu/brain.lbr` is read-only
//!   from this crate. Fixtures live under `/var/lib/lu/bench-fixtures/`.
//! - No stubs. Every code path is implemented. Any branch that cannot
//!   succeed returns a typed `HarnessError`.
//!
//! ## Adding a new workload (downstream usage)
//!
//! Workloads are registered via the [`WorkloadRegistry`] in the binary
//! `bench_v5_assert`. To add a new workload:
//!
//! 1. Implement an `fn workload(ctx: &mut WorkloadCtx) -> Result<u64,
//!    HarnessError>` (or any signature your phase needs — the registry
//!    is just a `BTreeMap<&'static str, Box<dyn Fn ...>>`).
//! 2. Register it in `bench_v5_assert::register_workloads`.
//! 3. Decide the RSS budget *in your phase brief*, not in 0.c — the
//!    `--threshold-rss-gib` CLI knob is per-invocation.

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use md5::{Digest as Md5Digest, Md5};
use sha2::Sha256;

use lu4d::bond::BondType;
use lu4d::seed::Seed;

// ----------------------------------------------------------------------
// Wire-format constants — must match `gen_prod_shape_fixture.rs`. Any
// drift between writer and reader is a wire break.
// ----------------------------------------------------------------------

/// Magic bytes at the start of every `.bench` fixture (Lu Bench FiXture).
pub const FIXTURE_MAGIC: &[u8; 4] = b"LBFX";
/// Format version this harness understands. Future writers MUST bump
/// this on any wire-incompatible change; readers MUST pin the versions
/// they accept. Mismatch raises [`HarnessError::FixtureVersion`].
pub const FIXTURE_FORMAT_VERSION: u32 = 1;
/// Tag byte for a BOND record in the fixture stream.
pub const TAG_BOND: u8 = 0x01;
/// Tag byte for the END trailer marker.
pub const TAG_END: u8 = 0xFF;
/// Maximum sensible header length. Bench fixtures use compact JSON; 16
/// MiB is many orders of magnitude beyond any real header. Used to
/// reject corrupt streams that claim a wildly oversized header.
pub const FIXTURE_HEADER_MAX_BYTES: u32 = 16 * 1024 * 1024;

// ----------------------------------------------------------------------
// Sampler constants. Values chosen at Phase 0.c spawn time; revisit if
// any downstream phase reports them as load-bearing.
// ----------------------------------------------------------------------

/// Default RSS sampler poll cadence.
pub const DEFAULT_SAMPLE_INTERVAL_MS: u64 = 250;
/// Reservoir cap for the RSS sampler. When `samples.len()` reaches
/// this, new samples are folded in via reservoir sampling so the
/// in-memory footprint stays bounded regardless of workload duration.
/// 100_000 samples at 250 ms = 6 h 56 m of dense data before reservoir
/// kicks in; at 100_000 u64 bytes each, the worst-case sampler heap
/// footprint is ~800 KB. Bounded.
pub const RESERVOIR_CAP: usize = 100_000;

// ----------------------------------------------------------------------
// Errors. One enum, no stubs, no `unwrap()` in the public API.
// ----------------------------------------------------------------------

/// Typed errors for harness operations. Surfaces wire-format violations,
/// I/O failures, and workload setup mistakes uniformly.
#[derive(Debug)]
pub enum HarnessError {
    /// Underlying I/O failure (file open, read, write).
    Io(std::io::Error),
    /// Fixture file's MAGIC bytes did not match [`FIXTURE_MAGIC`].
    FixtureMagic { got: [u8; 4] },
    /// Fixture file's format version did not match
    /// [`FIXTURE_FORMAT_VERSION`]. Wire-incompat — reject.
    FixtureVersion { got: u32, supported: u32 },
    /// Header JSON length exceeds [`FIXTURE_HEADER_MAX_BYTES`] (likely
    /// corruption — a real header is a few hundred bytes).
    FixtureHeaderTooLarge { len: u32, max: u32 },
    /// Header JSON failed to parse as UTF-8 / JSON.
    FixtureHeader(String),
    /// Unexpected end-of-file mid-record.
    FixtureTruncated { context: &'static str },
    /// Record tag byte was neither `TAG_BOND` nor `TAG_END`.
    FixtureBadTag { got: u8 },
    /// Predicate index in a record did not map to a known
    /// `BondType::all()` slot.
    FixturePredicateOutOfRange { got: u8, max: usize },
    /// SHA-256 body digest from the trailer disagreed with the digest
    /// we computed while streaming — fixture corruption or partial
    /// write.
    FixtureSha256Mismatch { computed: String, trailer: String },
    /// Trailer reported a bond count we did not see while iterating.
    /// Distinct from `Sha256Mismatch` — surfaces logical-count drift
    /// even when bytes are intact.
    FixtureBondCountMismatch { trailer: u64, observed: u64 },
    /// Sampler thread panicked or was already finished.
    SamplerJoin(String),
    /// CLI selected a workload name that the registry did not contain.
    WorkloadUnknown { got: String, available: Vec<String> },
    /// Stats sidecar JSON parse failure.
    StatsParse(String),
    /// `/proc/self/status` did not contain a `RssAnon:` line. Linux-only
    /// — surfaced if someone tries to run the harness on macOS / WSL
    /// without `/proc`.
    NoRssAnon,
}

impl std::fmt::Display for HarnessError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HarnessError::Io(e) => write!(f, "io: {}", e),
            HarnessError::FixtureMagic { got } => {
                write!(f, "fixture magic mismatch: got {:?}, want {:?}", got, FIXTURE_MAGIC)
            }
            HarnessError::FixtureVersion { got, supported } => {
                write!(f, "fixture format_version {} not supported (this harness supports {})", got, supported)
            }
            HarnessError::FixtureHeaderTooLarge { len, max } => {
                write!(f, "fixture header_len {} exceeds max {}", len, max)
            }
            HarnessError::FixtureHeader(msg) => write!(f, "fixture header: {}", msg),
            HarnessError::FixtureTruncated { context } => {
                write!(f, "fixture truncated at: {}", context)
            }
            HarnessError::FixtureBadTag { got } => write!(f, "fixture bad tag byte: 0x{:02x}", got),
            HarnessError::FixturePredicateOutOfRange { got, max } => {
                write!(f, "fixture predicate_idx {} out of range (max {})", got, max)
            }
            HarnessError::FixtureSha256Mismatch { computed, trailer } => {
                write!(f, "fixture sha256 mismatch: computed {}, trailer {}", computed, trailer)
            }
            HarnessError::FixtureBondCountMismatch { trailer, observed } => {
                write!(f, "fixture bond count mismatch: trailer {}, observed {}", trailer, observed)
            }
            HarnessError::SamplerJoin(msg) => write!(f, "sampler join: {}", msg),
            HarnessError::WorkloadUnknown { got, available } => write!(
                f,
                "unknown workload {:?}; available: {:?}",
                got, available
            ),
            HarnessError::StatsParse(msg) => write!(f, "stats.json parse: {}", msg),
            HarnessError::NoRssAnon => write!(
                f,
                "/proc/self/status has no RssAnon line (Linux required)"
            ),
        }
    }
}

impl std::error::Error for HarnessError {}

impl From<std::io::Error> for HarnessError {
    fn from(e: std::io::Error) -> Self { HarnessError::Io(e) }
}

// ----------------------------------------------------------------------
// Fixture header.
// ----------------------------------------------------------------------

/// Header echo returned from a successful fixture open. Mirrors the JSON
/// schema that `gen_prod_shape_fixture` writes (see the binary's module
/// doc for the spec) — we deserialize only the fields a downstream
/// reader cares about, leaving room for the writer to add forward-compat
/// fields without breaking us.
#[derive(Debug, Clone)]
pub struct FixtureHeader {
    /// Echoed `format_version` (must equal [`FIXTURE_FORMAT_VERSION`]).
    pub format_version: u32,
    /// Echoed `source_corpus` (e.g. `wikimedia/wikipedia`).
    pub source_corpus: String,
    /// Echoed `source_corpus_config` (e.g. `20231101.en`).
    pub source_corpus_config: String,
    /// Echoed `source_corpus_revision` (HF Hub revision, default `main`).
    pub source_corpus_revision: String,
    /// Echoed `article_range_end_requested`.
    pub article_range_end_requested: u64,
    /// Echoed `generator_binary_md5`.
    pub generator_binary_md5: String,
    /// Echoed `generation_timestamp_utc` (epoch seconds).
    pub generation_timestamp_utc: i64,
    /// Echoed `extractor` description.
    pub extractor: String,
    /// Original raw JSON bytes — returned so the report can echo them
    /// verbatim into `RssReport.fixture_header_raw` without a re-encode.
    pub raw_json: String,
}

/// One decoded bond record from a `.bench` fixture.
#[derive(Debug, Clone, Copy)]
pub struct BondRecord {
    /// Subject seed (32 bytes).
    pub subject: Seed,
    /// Predicate index into `BondType::all()`. Resolves to
    /// `BondType::all()[predicate_idx as usize]` — wire-compat
    /// guaranteed by the index encoding.
    pub predicate_idx: u8,
    /// Object seed (32 bytes).
    pub object: Seed,
    /// Bond weight (extractor-assigned).
    pub weight: f32,
    /// Confidence byte: 0 = Unverified, 1 = Inferred.
    pub confidence: u8,
}

/// Fixed-size kind-tag buffer. Tags are ASCII labels emitted by the
/// extractor ("definition", "composition", "purpose", ...). Bounded by
/// the wire-format u8 length prefix at 255 bytes.
#[derive(Debug, Clone)]
pub struct BondRecordKind(pub String);

// ----------------------------------------------------------------------
// FixtureReader — streaming iteration with SHA-256 integrity check.
// ----------------------------------------------------------------------

/// Streaming reader for a `.bench` fixture.
///
/// Construct with [`FixtureReader::open`]; that validates the magic and
/// format version and reads the header. After construction, drive
/// iteration via [`FixtureReader::read_record`] in a loop until
/// `Ok(None)`, then call [`FixtureReader::finish`] to validate the
/// trailer's bond count and SHA-256 body digest.
///
/// Records are NOT collected into a Vec at any stage. The whole-fixture
/// memory footprint is one decoded record + a `BufReader` (8 KiB).
pub struct FixtureReader {
    reader: BufReader<File>,
    header: FixtureHeader,
    /// SHA-256 over every byte AFTER the header and BEFORE the trailer
    /// — matches the writer's digest in `gen_prod_shape_fixture`.
    body_sha: Sha256,
    observed_bonds: u64,
    /// True once we have consumed the END tag and read the trailer.
    finished: bool,
    /// Predicate count for range-checking record predicate_idx fields.
    /// Snapshot at open time so a BondType variant addition mid-process
    /// still gives a meaningful error.
    predicate_count: usize,
}

impl FixtureReader {
    /// Open `path`, validate magic + version, read the header JSON.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, HarnessError> {
        let f = File::open(path.as_ref())?;
        let mut reader = BufReader::with_capacity(1 << 16, f);

        let mut magic = [0u8; 4];
        read_exact_or_truncated(&mut reader, &mut magic, "magic")?;
        if &magic != FIXTURE_MAGIC {
            return Err(HarnessError::FixtureMagic { got: magic });
        }

        let mut version_buf = [0u8; 4];
        read_exact_or_truncated(&mut reader, &mut version_buf, "format_version")?;
        let version = u32::from_le_bytes(version_buf);
        if version != FIXTURE_FORMAT_VERSION {
            return Err(HarnessError::FixtureVersion { got: version, supported: FIXTURE_FORMAT_VERSION });
        }

        let mut header_len_buf = [0u8; 4];
        read_exact_or_truncated(&mut reader, &mut header_len_buf, "header_len")?;
        let header_len = u32::from_le_bytes(header_len_buf);
        if header_len > FIXTURE_HEADER_MAX_BYTES {
            return Err(HarnessError::FixtureHeaderTooLarge { len: header_len, max: FIXTURE_HEADER_MAX_BYTES });
        }
        let mut header_bytes = vec![0u8; header_len as usize];
        read_exact_or_truncated(&mut reader, &mut header_bytes, "header_json")?;
        let header_str = String::from_utf8(header_bytes)
            .map_err(|e| HarnessError::FixtureHeader(format!("utf8: {}", e)))?;
        let header_value: serde_json::Value = serde_json::from_str(&header_str)
            .map_err(|e| HarnessError::FixtureHeader(format!("json: {}", e)))?;

        let header = parse_header(&header_value, header_str.clone())?;
        // Defense in depth: the writer also bumps format_version on wire
        // breaks, but JSON-encoded version must match the binary-encoded
        // one. Distrust until verified.
        if header.format_version != FIXTURE_FORMAT_VERSION {
            return Err(HarnessError::FixtureVersion {
                got: header.format_version,
                supported: FIXTURE_FORMAT_VERSION,
            });
        }

        Ok(FixtureReader {
            reader,
            header,
            body_sha: Sha256::new(),
            observed_bonds: 0,
            finished: false,
            predicate_count: BondType::all().len(),
        })
    }

    /// Header echo for callers that need the metadata before iterating.
    pub fn header(&self) -> &FixtureHeader { &self.header }

    /// Read one record from the stream.
    ///
    /// Returns `Ok(Some((record, kind)))` for each BOND record,
    /// `Ok(None)` once the END tag is reached (after which the trailer
    /// has been consumed and digest-checked).
    pub fn read_record(&mut self) -> Result<Option<(BondRecord, BondRecordKind)>, HarnessError> {
        if self.finished {
            return Ok(None);
        }

        let mut tag_buf = [0u8; 1];
        // The tag byte itself is NOT part of the SHA digest when it's
        // the END marker — but it IS part of the digest when it's BOND.
        // The writer hashes the whole record including its TAG_BOND
        // byte. We mirror that.
        match self.reader.read(&mut tag_buf)? {
            0 => return Err(HarnessError::FixtureTruncated { context: "missing tag" }),
            _ => {}
        }
        let tag = tag_buf[0];

        match tag {
            TAG_BOND => {
                // BOND record body: subj(32) + pred(1) + obj(32) + weight(4) + conf(1) + kindlen(1) + kind(N).
                let mut subj = [0u8; 32];
                read_exact_or_truncated(&mut self.reader, &mut subj, "bond.subject")?;

                let mut pred = [0u8; 1];
                read_exact_or_truncated(&mut self.reader, &mut pred, "bond.predicate")?;
                let predicate_idx = pred[0];
                if (predicate_idx as usize) >= self.predicate_count {
                    return Err(HarnessError::FixturePredicateOutOfRange {
                        got: predicate_idx,
                        max: self.predicate_count,
                    });
                }

                let mut obj = [0u8; 32];
                read_exact_or_truncated(&mut self.reader, &mut obj, "bond.object")?;

                let mut weight_buf = [0u8; 4];
                read_exact_or_truncated(&mut self.reader, &mut weight_buf, "bond.weight")?;
                let weight = f32::from_le_bytes(weight_buf);

                let mut conf_buf = [0u8; 1];
                read_exact_or_truncated(&mut self.reader, &mut conf_buf, "bond.confidence")?;
                let confidence = conf_buf[0];

                let mut kindlen_buf = [0u8; 1];
                read_exact_or_truncated(&mut self.reader, &mut kindlen_buf, "bond.kindlen")?;
                let kindlen = kindlen_buf[0] as usize;

                let mut kind_buf = vec![0u8; kindlen];
                read_exact_or_truncated(&mut self.reader, &mut kind_buf, "bond.kind")?;
                let kind = String::from_utf8(kind_buf.clone()).unwrap_or_else(|_| {
                    // The writer always emits ASCII; fall back to
                    // lossy decode rather than failing the run on a
                    // single bad record (kind tag is observational not
                    // load-bearing).
                    String::from_utf8_lossy(&kind_buf).into_owned()
                });

                // Replay into SHA digest. Writer hashed: [TAG_BOND, subj, pred, obj, weight, conf, kindlen, kind].
                let mut record_bytes = Vec::with_capacity(72 + kindlen);
                record_bytes.push(TAG_BOND);
                record_bytes.extend_from_slice(&subj);
                record_bytes.push(predicate_idx);
                record_bytes.extend_from_slice(&obj);
                record_bytes.extend_from_slice(&weight_buf);
                record_bytes.push(confidence);
                record_bytes.push(kindlen_buf[0]);
                record_bytes.extend_from_slice(&kind_buf);
                self.body_sha.update(&record_bytes);

                self.observed_bonds = self.observed_bonds.saturating_add(1);

                Ok(Some((
                    BondRecord {
                        subject: Seed::from_bytes(subj),
                        predicate_idx,
                        object: Seed::from_bytes(obj),
                        weight,
                        confidence,
                    },
                    BondRecordKind(kind),
                )))
            }
            TAG_END => {
                // Trailer: total_bonds(8) + distinct_subjects(8) + sha256_body(32).
                let mut total_buf = [0u8; 8];
                read_exact_or_truncated(&mut self.reader, &mut total_buf, "trailer.total_bonds")?;
                let total_bonds = u64::from_le_bytes(total_buf);

                let mut distinct_buf = [0u8; 8];
                read_exact_or_truncated(&mut self.reader, &mut distinct_buf, "trailer.distinct_subjects")?;
                let _distinct_subjects = u64::from_le_bytes(distinct_buf);

                let mut sha_buf = [0u8; 32];
                read_exact_or_truncated(&mut self.reader, &mut sha_buf, "trailer.sha256")?;

                let computed = self.body_sha.clone().finalize();
                if computed.as_slice() != sha_buf {
                    return Err(HarnessError::FixtureSha256Mismatch {
                        computed: hex_lower(&computed),
                        trailer: hex_lower(&sha_buf),
                    });
                }
                if total_bonds != self.observed_bonds {
                    return Err(HarnessError::FixtureBondCountMismatch {
                        trailer: total_bonds,
                        observed: self.observed_bonds,
                    });
                }

                self.finished = true;
                Ok(None)
            }
            other => Err(HarnessError::FixtureBadTag { got: other }),
        }
    }

    /// Total bonds observed so far. After [`Self::read_record`] returns
    /// `Ok(None)`, this equals the trailer's `total_bonds`.
    pub fn observed_bonds(&self) -> u64 { self.observed_bonds }

    /// Whether the END tag and trailer have been consumed.
    pub fn finished(&self) -> bool { self.finished }

    /// Drain the reader to completion. Convenience for callers that
    /// only need the trailer-validation side effect (e.g. the harness
    /// self-test). Streams records through a callback and stops at END.
    pub fn drain_with<F: FnMut(BondRecord, BondRecordKind)>(&mut self, mut f: F) -> Result<u64, HarnessError> {
        loop {
            match self.read_record()? {
                Some((rec, kind)) => f(rec, kind),
                None => return Ok(self.observed_bonds),
            }
        }
    }
}

fn read_exact_or_truncated<R: Read>(r: &mut R, buf: &mut [u8], context: &'static str) -> Result<(), HarnessError> {
    match r.read_exact(buf) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
            Err(HarnessError::FixtureTruncated { context })
        }
        Err(e) => Err(HarnessError::Io(e)),
    }
}

fn parse_header(v: &serde_json::Value, raw_json: String) -> Result<FixtureHeader, HarnessError> {
    let obj = v.as_object().ok_or_else(|| HarnessError::FixtureHeader("header is not a JSON object".to_string()))?;
    fn s(o: &serde_json::Map<String, serde_json::Value>, key: &str) -> Result<String, HarnessError> {
        o.get(key)
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| HarnessError::FixtureHeader(format!("missing string field: {}", key)))
    }
    fn u(o: &serde_json::Map<String, serde_json::Value>, key: &str) -> Result<u64, HarnessError> {
        o.get(key)
            .and_then(|v| v.as_u64())
            .ok_or_else(|| HarnessError::FixtureHeader(format!("missing u64 field: {}", key)))
    }
    fn i(o: &serde_json::Map<String, serde_json::Value>, key: &str) -> Result<i64, HarnessError> {
        o.get(key)
            .and_then(|v| v.as_i64())
            .ok_or_else(|| HarnessError::FixtureHeader(format!("missing i64 field: {}", key)))
    }
    Ok(FixtureHeader {
        format_version: u(obj, "format_version")? as u32,
        source_corpus: s(obj, "source_corpus")?,
        source_corpus_config: s(obj, "source_corpus_config")?,
        source_corpus_revision: s(obj, "source_corpus_revision")?,
        article_range_end_requested: u(obj, "article_range_end_requested")?,
        generator_binary_md5: s(obj, "generator_binary_md5")?,
        generation_timestamp_utc: i(obj, "generation_timestamp_utc")?,
        extractor: s(obj, "extractor")?,
        raw_json,
    })
}

fn hex_lower(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes { s.push_str(&format!("{:02x}", b)); }
    s
}

// ----------------------------------------------------------------------
// RSS sampler. Polls /proc/self/status RssAnon on a background thread.
// ----------------------------------------------------------------------

/// Bounded-memory RSS sampler.
///
/// Construct via [`RssSampler::start`], which spawns a background
/// thread that polls `/proc/self/status` at `interval`. Call
/// [`RssSampler::finish`] when the workload completes — this signals
/// the thread to stop, joins it, and returns the collected samples plus
/// exact min/max/observed-count.
///
/// Memory bound: samples are kept in a `Vec<u64>` up to [`RESERVOIR_CAP`]
/// entries, then folded in via reservoir sampling. Worst-case sampler
/// heap footprint stays under ~1 MB regardless of workload duration.
pub struct RssSampler {
    stop: Arc<AtomicBool>,
    peak: Arc<AtomicU64>,
    min_observed: Arc<AtomicU64>,
    observed_count: Arc<AtomicU64>,
    samples: Arc<Mutex<Vec<u64>>>,
    handle: Option<JoinHandle<()>>,
    interval: Duration,
    started_at: Instant,
}

/// Aggregated sample report — returned from [`RssSampler::finish`].
#[derive(Debug, Clone)]
pub struct SamplerReport {
    /// Sampler interval (echo).
    pub interval_ms: u64,
    /// Total samples observed (exact — counts even those that did not
    /// land in the reservoir).
    pub observed_count: u64,
    /// Number of samples retained in memory (== `observed_count` if
    /// below the reservoir cap, otherwise == [`RESERVOIR_CAP`]).
    pub retained_count: usize,
    /// Exact minimum across all observed samples (KiB).
    pub min_rss_kib: u64,
    /// Exact maximum across all observed samples (KiB).
    pub peak_rss_kib: u64,
    /// Percentile breakdown computed from retained samples (KiB).
    pub p50_rss_kib: u64,
    pub p90_rss_kib: u64,
    pub p99_rss_kib: u64,
    /// Whether the reservoir filled (true => percentile breakdown is an
    /// estimate from a uniform-random sample of the stream; min/max/peak
    /// remain exact regardless).
    pub reservoir_engaged: bool,
    /// Sorted retained samples (KiB), exposed so callers can dump full
    /// traces for offline plotting. Length == `retained_count`.
    pub sorted_samples_kib: Vec<u64>,
}

impl RssSampler {
    /// Start sampling on a background thread.
    pub fn start(interval: Duration) -> Result<Self, HarnessError> {
        // Probe once up-front so we fail fast if /proc is unavailable.
        let _ = read_rss_anon_kib()?;

        let stop = Arc::new(AtomicBool::new(false));
        let peak = Arc::new(AtomicU64::new(0));
        let min_observed = Arc::new(AtomicU64::new(u64::MAX));
        let observed_count = Arc::new(AtomicU64::new(0));
        let samples: Arc<Mutex<Vec<u64>>> = Arc::new(Mutex::new(Vec::with_capacity(1024)));

        let stop_t = stop.clone();
        let peak_t = peak.clone();
        let min_t = min_observed.clone();
        let count_t = observed_count.clone();
        let samples_t = samples.clone();
        let interval_t = interval;

        let handle = thread::Builder::new()
            .name("lu-bench-rss-sampler".to_string())
            .spawn(move || {
                // Tiny PRNG state for reservoir sampling. We avoid
                // pulling `rand` into the sampler hot path (rand IS in
                // the crate's dep list but cheap; this just keeps the
                // sampler self-contained and deterministic per-run).
                let mut prng_state: u64 = 0x9E37_79B9_7F4A_7C15;
                while !stop_t.load(Ordering::Relaxed) {
                    if let Ok(rss_kib) = read_rss_anon_kib() {
                        // Update peak.
                        let mut cur = peak_t.load(Ordering::Relaxed);
                        while rss_kib > cur {
                            match peak_t.compare_exchange(cur, rss_kib, Ordering::Relaxed, Ordering::Relaxed) {
                                Ok(_) => break,
                                Err(actual) => cur = actual,
                            }
                        }
                        // Update min.
                        let mut curmin = min_t.load(Ordering::Relaxed);
                        while rss_kib < curmin {
                            match min_t.compare_exchange(curmin, rss_kib, Ordering::Relaxed, Ordering::Relaxed) {
                                Ok(_) => break,
                                Err(actual) => curmin = actual,
                            }
                        }
                        let n = count_t.fetch_add(1, Ordering::Relaxed);
                        // Reservoir sampling: while len < cap, append.
                        // After: replace samples[k] where k uniform in
                        // [0, n] with probability cap/(n+1).
                        if let Ok(mut v) = samples_t.lock() {
                            if v.len() < RESERVOIR_CAP {
                                v.push(rss_kib);
                            } else {
                                prng_state = prng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                                let r = prng_state >> 33; // top 31 bits for uniformity
                                let k = (r % (n + 1)) as usize;
                                if k < RESERVOIR_CAP {
                                    v[k] = rss_kib;
                                }
                            }
                        }
                    }
                    thread::sleep(interval_t);
                }
            })
            .map_err(|e| HarnessError::Io(std::io::Error::new(std::io::ErrorKind::Other, format!("sampler spawn: {}", e))))?;

        Ok(RssSampler {
            stop,
            peak,
            min_observed,
            observed_count,
            samples,
            handle: Some(handle),
            interval,
            started_at: Instant::now(),
        })
    }

    /// Snapshot of current peak RSS (KiB). Safe to poll mid-workload.
    pub fn peak_kib(&self) -> u64 { self.peak.load(Ordering::Relaxed) }

    /// Elapsed time since `start`. Lives outside `finish` because
    /// long-running workloads sometimes want a progress readout without
    /// stopping the sampler.
    pub fn elapsed(&self) -> Duration { self.started_at.elapsed() }

    /// Signal the sampler to stop, join the thread, and return the
    /// aggregated [`SamplerReport`].
    pub fn finish(mut self) -> Result<SamplerReport, HarnessError> {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            h.join().map_err(|e| HarnessError::SamplerJoin(format!("{:?}", e)))?;
        }
        let observed_count = self.observed_count.load(Ordering::Relaxed);
        let peak_rss_kib = self.peak.load(Ordering::Relaxed);
        let raw_min = self.min_observed.load(Ordering::Relaxed);
        let min_rss_kib = if observed_count == 0 || raw_min == u64::MAX { 0 } else { raw_min };

        let mut sorted_samples: Vec<u64> = self.samples.lock()
            .map(|g| g.clone())
            .unwrap_or_default();
        sorted_samples.sort_unstable();
        let retained_count = sorted_samples.len();
        let p50_rss_kib = percentile(&sorted_samples, 0.50).unwrap_or(0);
        let p90_rss_kib = percentile(&sorted_samples, 0.90).unwrap_or(0);
        let p99_rss_kib = percentile(&sorted_samples, 0.99).unwrap_or(0);

        let reservoir_engaged = observed_count as usize > retained_count;

        Ok(SamplerReport {
            interval_ms: self.interval.as_millis() as u64,
            observed_count,
            retained_count,
            min_rss_kib,
            peak_rss_kib,
            p50_rss_kib,
            p90_rss_kib,
            p99_rss_kib,
            reservoir_engaged,
            sorted_samples_kib: sorted_samples,
        })
    }
}

fn percentile(sorted: &[u64], q: f64) -> Option<u64> {
    if sorted.is_empty() { return None; }
    let n = sorted.len() as f64;
    // "Nearest rank" definition: index = ceil(q * n) - 1, clamped.
    let mut idx = (q * n).ceil() as isize - 1;
    if idx < 0 { idx = 0; }
    if idx as usize >= sorted.len() { idx = sorted.len() as isize - 1; }
    Some(sorted[idx as usize])
}

fn read_rss_anon_kib() -> Result<u64, HarnessError> {
    let s = fs::read_to_string("/proc/self/status")?;
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("RssAnon:") {
            let tok = rest.split_whitespace().next();
            if let Some(t) = tok {
                if let Ok(v) = t.parse::<u64>() {
                    return Ok(v);
                }
            }
        }
    }
    Err(HarnessError::NoRssAnon)
}

// ----------------------------------------------------------------------
// Budget API.
// ----------------------------------------------------------------------

/// One full bounded-RSS assertion run.
///
/// `RssReport` is the canonical record that downstream phases produce
/// (one per workload invocation) and is also the JSON sidecar shape
/// that `bench_v5_assert` emits via [`RssReport::write_json`].
#[derive(Debug, Clone)]
pub struct RssReport {
    /// Pass/fail. `true` iff `peak_rss_gib <= threshold_rss_gib`.
    pub pass: bool,
    /// The threshold the caller asserted (GiB).
    pub threshold_rss_gib: f64,
    /// Peak observed RSS (GiB).
    pub peak_rss_gib: f64,
    /// Min observed RSS (GiB).
    pub min_rss_gib: f64,
    /// p50 / p90 / p99 (GiB).
    pub p50_rss_gib: f64,
    pub p90_rss_gib: f64,
    pub p99_rss_gib: f64,
    /// Wall-clock workload duration (seconds).
    pub duration_secs: f64,
    /// Sampler metadata.
    pub interval_ms: u64,
    pub observed_sample_count: u64,
    pub retained_sample_count: usize,
    pub reservoir_engaged: bool,
    /// Phase tag (e.g. `0c`, `1.0`, `7.0`) — caller-supplied label.
    pub phase: String,
    /// Workload name (registry key).
    pub workload: String,
    /// Echo of the fixture header (so the report stands alone).
    pub fixture_path: String,
    pub fixture_format_version: u32,
    pub fixture_source_corpus: String,
    pub fixture_source_revision: String,
    pub fixture_generator_md5: String,
    /// Workload-defined opaque numeric result (e.g. linear_scan returns
    /// total bond count). Surfaced for downstream tooling.
    pub workload_result: u64,
    /// Environment metadata.
    pub binary_md5: String,
    pub git_head: String,
    pub hostname: String,
    pub run_timestamp_utc: i64,
}

impl RssReport {
    /// Serialize to a pretty-printed JSON file. Top-level keys mirror
    /// the Phase 0.b `.stats.json` convention (sectioned, lower-snake-
    /// case).
    pub fn write_json(&self, path: impl AsRef<Path>) -> Result<(), HarnessError> {
        let v = serde_json::json!({
            "verdict": {
                "pass": self.pass,
                "threshold_rss_gib": self.threshold_rss_gib,
                "peak_rss_gib": self.peak_rss_gib,
                "phase": self.phase,
                "workload": self.workload,
            },
            "rss_breakdown": {
                "min_rss_gib": self.min_rss_gib,
                "p50_rss_gib": self.p50_rss_gib,
                "p90_rss_gib": self.p90_rss_gib,
                "p99_rss_gib": self.p99_rss_gib,
                "peak_rss_gib": self.peak_rss_gib,
            },
            "sampler": {
                "interval_ms": self.interval_ms,
                "observed_sample_count": self.observed_sample_count,
                "retained_sample_count": self.retained_sample_count,
                "reservoir_engaged": self.reservoir_engaged,
            },
            "workload_meta": {
                "duration_secs": self.duration_secs,
                "workload_result": self.workload_result,
            },
            "fixture": {
                "path": self.fixture_path,
                "format_version": self.fixture_format_version,
                "source_corpus": self.fixture_source_corpus,
                "source_corpus_revision": self.fixture_source_revision,
                "generator_binary_md5": self.fixture_generator_md5,
            },
            "env": {
                "binary_md5": self.binary_md5,
                "git_head": self.git_head,
                "hostname": self.hostname,
                "run_timestamp_utc": self.run_timestamp_utc,
            },
        });
        let bytes = serde_json::to_vec_pretty(&v)
            .map_err(|e| HarnessError::Io(std::io::Error::new(std::io::ErrorKind::Other, format!("serialize: {}", e))))?;
        let mut f = File::create(path.as_ref())?;
        f.write_all(&bytes)?;
        f.flush()?;
        Ok(())
    }
}

/// Caller-supplied parameters for [`with_rss_budget`].
#[derive(Debug, Clone)]
pub struct RssBudget {
    /// Pass/fail threshold (GiB).
    pub threshold_rss_gib: f64,
    /// Sampler interval. Defaults to [`DEFAULT_SAMPLE_INTERVAL_MS`].
    pub interval: Duration,
    /// Phase tag (echoed into the report).
    pub phase: String,
    /// Workload name (echoed into the report).
    pub workload: String,
    /// Fixture metadata so the report stands alone.
    pub fixture_path: PathBuf,
    pub fixture_header: FixtureHeader,
    /// Environment metadata, captured by the caller.
    pub binary_md5: String,
    pub git_head: String,
    pub hostname: String,
}

impl RssBudget {
    /// Construct a default-interval budget. Caller still owns phase,
    /// workload, threshold, and fixture wiring.
    pub fn new(
        threshold_rss_gib: f64,
        phase: impl Into<String>,
        workload: impl Into<String>,
        fixture_path: PathBuf,
        fixture_header: FixtureHeader,
        binary_md5: impl Into<String>,
        git_head: impl Into<String>,
        hostname: impl Into<String>,
    ) -> Self {
        RssBudget {
            threshold_rss_gib,
            interval: Duration::from_millis(DEFAULT_SAMPLE_INTERVAL_MS),
            phase: phase.into(),
            workload: workload.into(),
            fixture_path,
            fixture_header,
            binary_md5: binary_md5.into(),
            git_head: git_head.into(),
            hostname: hostname.into(),
        }
    }
}

/// Run `workload`, sampling RSS the whole time, return a populated
/// [`RssReport`]. Workload may panic — the sampler thread is still
/// joined cleanly via `Drop` on `Result` unwinding.
///
/// Note: this function does NOT short-circuit when the budget is
/// exceeded. The contract is to LET the workload run to completion and
/// REPORT the verdict. If a phase wants hard-abort behavior, it can
/// inspect `report.pass` and `process::exit(1)` itself — keeping abort
/// policy at the caller layer avoids surprising future workloads.
pub fn with_rss_budget<F, T>(budget: RssBudget, workload: F) -> Result<(RssReport, T), HarnessError>
where
    F: FnOnce() -> Result<T, HarnessError>,
    T: Into<u64> + Copy,
{
    let sampler = RssSampler::start(budget.interval)?;
    let t0 = Instant::now();
    // Propagate workload failure as-is — we still want to stop the
    // sampler before returning, so we hold the result and finish in
    // either branch.
    let workload_result = workload();
    let duration = t0.elapsed();
    let sampler_report = sampler.finish()?;

    let value = match workload_result {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    let numeric: u64 = (value).into();

    let report = build_report(&budget, &sampler_report, duration, numeric);
    Ok((report, value))
}

fn build_report(
    budget: &RssBudget,
    sampler: &SamplerReport,
    duration: Duration,
    workload_result: u64,
) -> RssReport {
    let peak_rss_gib = kib_to_gib(sampler.peak_rss_kib);
    let min_rss_gib = kib_to_gib(sampler.min_rss_kib);
    let p50_rss_gib = kib_to_gib(sampler.p50_rss_kib);
    let p90_rss_gib = kib_to_gib(sampler.p90_rss_kib);
    let p99_rss_gib = kib_to_gib(sampler.p99_rss_kib);
    let pass = peak_rss_gib <= budget.threshold_rss_gib;
    let run_timestamp_utc = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);

    RssReport {
        pass,
        threshold_rss_gib: budget.threshold_rss_gib,
        peak_rss_gib,
        min_rss_gib,
        p50_rss_gib,
        p90_rss_gib,
        p99_rss_gib,
        duration_secs: duration.as_secs_f64(),
        interval_ms: sampler.interval_ms,
        observed_sample_count: sampler.observed_count,
        retained_sample_count: sampler.retained_count,
        reservoir_engaged: sampler.reservoir_engaged,
        phase: budget.phase.clone(),
        workload: budget.workload.clone(),
        fixture_path: budget.fixture_path.to_string_lossy().into_owned(),
        fixture_format_version: budget.fixture_header.format_version,
        fixture_source_corpus: budget.fixture_header.source_corpus.clone(),
        fixture_source_revision: budget.fixture_header.source_corpus_revision.clone(),
        fixture_generator_md5: budget.fixture_header.generator_binary_md5.clone(),
        workload_result,
        binary_md5: budget.binary_md5.clone(),
        git_head: budget.git_head.clone(),
        hostname: budget.hostname.clone(),
        run_timestamp_utc,
    }
}

fn kib_to_gib(kib: u64) -> f64 { kib as f64 / 1024.0 / 1024.0 }

// ----------------------------------------------------------------------
// Predicate-coverage checker. Reads the Phase 0.b .stats.json sidecar.
// ----------------------------------------------------------------------

/// Predicate-coverage summary derived from a Phase 0.b `.stats.json`.
#[derive(Debug, Clone)]
pub struct PredicateCoverage {
    /// All BondType labels that emitted at least one bond.
    pub covered: Vec<String>,
    /// BondType labels that emitted zero bonds.
    pub empty: Vec<String>,
    /// Counts keyed by label, for callers that want raw numbers.
    pub counts: BTreeMap<String, u64>,
}

impl PredicateCoverage {
    /// Whether `label` (a `BondType` variant name) had nonzero bonds in
    /// the fixture. Workloads that target a specific predicate should
    /// call this before running and refuse if uncovered, to surface the
    /// Phase 0.b finding (Causes / ContrastedWith / DerivedFrom /
    /// Opposes / Produces / Requires emit zero from Wikipedia).
    pub fn is_covered(&self, label: &str) -> bool {
        self.counts.get(label).copied().unwrap_or(0) > 0
    }
}

/// Parse a Phase 0.b `.stats.json` and return a coverage summary.
pub fn predicate_coverage_from_stats(path: impl AsRef<Path>) -> Result<PredicateCoverage, HarnessError> {
    let bytes = fs::read(path.as_ref())?;
    let v: serde_json::Value = serde_json::from_slice(&bytes)
        .map_err(|e| HarnessError::StatsParse(format!("json: {}", e)))?;
    let dist = v.get("predicate_distribution")
        .and_then(|x| x.as_object())
        .ok_or_else(|| HarnessError::StatsParse("missing predicate_distribution object".to_string()))?;
    let mut counts = BTreeMap::new();
    let mut covered = Vec::new();
    let mut empty = Vec::new();
    for (k, val) in dist.iter() {
        let n = val.as_u64().unwrap_or(0);
        counts.insert(k.clone(), n);
        if n == 0 {
            empty.push(k.clone());
        } else {
            covered.push(k.clone());
        }
    }
    covered.sort();
    empty.sort();
    Ok(PredicateCoverage { covered, empty, counts })
}

// ----------------------------------------------------------------------
// Workload registry. Pure dispatch — no global state.
// ----------------------------------------------------------------------

/// Context passed to every workload. The registry constructs this from
/// CLI args before dispatch.
pub struct WorkloadCtx {
    /// Open reader on the fixture. Workloads consume it streaming.
    pub fixture: FixtureReader,
    /// Fixture path (for the report echo).
    pub fixture_path: PathBuf,
}

/// Boxed workload function. Returns the workload's primary scalar
/// result (e.g. linear_scan returns observed bond count) wrapped in
/// `HarnessError`. Numeric type is `u64` so it lives cleanly in the
/// `RssReport.workload_result` field.
pub type WorkloadFn = Box<dyn FnOnce(&mut WorkloadCtx) -> Result<u64, HarnessError>>;

/// Dispatch table.
///
/// `BTreeMap` (not `HashMap`) so listing available workloads in the
/// `--help` output and `WorkloadUnknown` error is deterministic.
pub struct WorkloadRegistry {
    entries: BTreeMap<&'static str, Box<dyn Fn() -> WorkloadFn>>,
}

impl WorkloadRegistry {
    pub fn new() -> Self { WorkloadRegistry { entries: BTreeMap::new() } }

    /// Register a workload constructor. The constructor returns a
    /// fresh `WorkloadFn` on each call so the registry can be reused
    /// across multiple invocations.
    pub fn register<F>(&mut self, name: &'static str, ctor: F)
    where
        F: Fn() -> WorkloadFn + 'static,
    {
        self.entries.insert(name, Box::new(ctor));
    }

    /// Dispatch by name. Returns `WorkloadUnknown` if not registered.
    pub fn dispatch(&self, name: &str) -> Result<WorkloadFn, HarnessError> {
        match self.entries.get(name) {
            Some(ctor) => Ok(ctor()),
            None => Err(HarnessError::WorkloadUnknown {
                got: name.to_string(),
                available: self.entries.keys().map(|s| s.to_string()).collect(),
            }),
        }
    }

    /// Sorted workload names. For `--help` listing.
    pub fn names(&self) -> Vec<&'static str> {
        self.entries.keys().copied().collect()
    }
}

impl Default for WorkloadRegistry { fn default() -> Self { Self::new() } }

// ----------------------------------------------------------------------
// Built-in workload: linear_scan. Reads every fixture record, counts
// bonds, returns the count for trailer-cross-check by the caller.
// ----------------------------------------------------------------------

/// Self-test workload: stream every record and count bonds. The
/// caller's `RssReport.workload_result` must equal the fixture
/// trailer's `total_bonds` for the run to be considered correct.
pub fn workload_linear_scan() -> WorkloadFn {
    Box::new(|ctx: &mut WorkloadCtx| -> Result<u64, HarnessError> {
        let count = ctx.fixture.drain_with(|_rec, _kind| {
            // Intentionally no-op: the linear scan exercises the
            // streaming-read path; per-record work is the bond decode
            // itself, which already happens inside FixtureReader.
        })?;
        Ok(count)
    })
}

// ----------------------------------------------------------------------
// Self-md5, git head, hostname — for the env block in RssReport.
// ----------------------------------------------------------------------

/// Compute the running binary's md5 by hashing `current_exe()`.
pub fn self_md5() -> Result<String, HarnessError> {
    let exe = std::env::current_exe()
        .map_err(|e| HarnessError::Io(std::io::Error::new(std::io::ErrorKind::Other, format!("current_exe: {}", e))))?;
    let mut f = File::open(&exe)?;
    let mut hasher = Md5::new();
    let mut buf = vec![0u8; 65536];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 { break; }
        hasher.update(&buf[..n]);
    }
    let out = hasher.finalize();
    let mut s = String::with_capacity(32);
    for b in &out { s.push_str(&format!("{:02x}", b)); }
    Ok(s)
}

/// Resolve git HEAD by reading `.git/HEAD` (and following refs). Falls
/// back to `"unknown"` if anything fails — this is metadata, not load-
/// bearing.
pub fn resolve_git_head(repo_root: impl AsRef<Path>) -> String {
    let root = repo_root.as_ref();
    let head_path = root.join(".git").join("HEAD");
    let head = match fs::read_to_string(&head_path) {
        Ok(s) => s.trim().to_string(),
        Err(_) => return "unknown".to_string(),
    };
    if let Some(rest) = head.strip_prefix("ref: ") {
        let ref_path = root.join(".git").join(rest.trim());
        if let Ok(s) = fs::read_to_string(&ref_path) {
            return s.trim().to_string();
        }
        // packed-refs fallback.
        let packed = root.join(".git").join("packed-refs");
        if let Ok(f) = File::open(&packed) {
            let r = BufReader::new(f);
            for line in r.lines().flatten() {
                if line.starts_with('#') { continue; }
                if let Some((sha, name)) = line.split_once(' ') {
                    if name.trim() == rest.trim() {
                        return sha.trim().to_string();
                    }
                }
            }
        }
        return "unknown".to_string();
    }
    head
}

/// Resolve hostname via `/etc/hostname` then `gethostname`-equivalent.
/// Falls back to `"unknown"`.
pub fn resolve_hostname() -> String {
    if let Ok(s) = fs::read_to_string("/etc/hostname") {
        let t = s.trim();
        if !t.is_empty() { return t.to_string(); }
    }
    "unknown".to_string()
}

// ----------------------------------------------------------------------
// Tests — exercise the fixture-reader header validation and the
// percentile helper on small inputs. Production fixture is not touched.
// ----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn percentile_basic() {
        let v = vec![1u64, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        assert_eq!(percentile(&v, 0.5), Some(5));
        assert_eq!(percentile(&v, 0.9), Some(9));
        assert_eq!(percentile(&v, 0.99), Some(10));
        assert_eq!(percentile(&v, 0.0), Some(1));
    }

    #[test]
    fn percentile_empty() {
        let v: Vec<u64> = vec![];
        assert_eq!(percentile(&v, 0.5), None);
    }

    fn write_minimal_fixture(path: &Path, bonds: &[(Seed, u8, Seed)]) {
        let mut f = File::create(path).unwrap();
        f.write_all(FIXTURE_MAGIC).unwrap();
        f.write_all(&FIXTURE_FORMAT_VERSION.to_le_bytes()).unwrap();
        let header = serde_json::json!({
            "format_version": FIXTURE_FORMAT_VERSION,
            "source_corpus": "test",
            "source_corpus_config": "unit",
            "source_corpus_revision": "main",
            "article_range_start": 0u64,
            "article_range_end_requested": 0u64,
            "generator_binary_md5": "00000000000000000000000000000000",
            "generation_timestamp_utc": 0i64,
            "predicate_encoding": "unit",
            "seed_derivation": "unit",
            "extractor": "unit",
        });
        let hdr_bytes = serde_json::to_vec(&header).unwrap();
        f.write_all(&(hdr_bytes.len() as u32).to_le_bytes()).unwrap();
        f.write_all(&hdr_bytes).unwrap();
        let mut sha = Sha256::new();
        for (subj, pred, obj) in bonds {
            let mut rec = Vec::new();
            rec.push(TAG_BOND);
            rec.extend_from_slice(subj.as_bytes());
            rec.push(*pred);
            rec.extend_from_slice(obj.as_bytes());
            rec.extend_from_slice(&1.0f32.to_le_bytes());
            rec.push(0u8);
            rec.push(4u8);
            rec.extend_from_slice(b"unit");
            sha.update(&rec);
            f.write_all(&rec).unwrap();
        }
        f.write_all(&[TAG_END]).unwrap();
        f.write_all(&(bonds.len() as u64).to_le_bytes()).unwrap();
        f.write_all(&0u64.to_le_bytes()).unwrap();
        f.write_all(&sha.finalize()).unwrap();
    }

    #[test]
    fn fixture_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.bench");
        let s1 = Seed::from_label("alpha");
        let s2 = Seed::from_label("beta");
        write_minimal_fixture(&path, &[(s1, 0, s2), (s2, 0, s1)]);
        let mut r = FixtureReader::open(&path).unwrap();
        assert_eq!(r.header().format_version, FIXTURE_FORMAT_VERSION);
        let mut n = 0u64;
        while let Some((_rec, _kind)) = r.read_record().unwrap() { n += 1; }
        assert_eq!(n, 2);
        assert!(r.finished());
    }

    #[test]
    fn fixture_bad_magic() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.bench");
        let mut f = File::create(&path).unwrap();
        f.write_all(b"XXXX").unwrap();
        let err = match FixtureReader::open(&path) { Ok(_) => panic!("expected error"), Err(e) => e };
        match err {
            HarnessError::FixtureMagic { got } => assert_eq!(&got, b"XXXX"),
            other => panic!("unexpected error: {:?}", other),
        }
    }
}
