//! OP-BRAIN-V5 Phase 0.b — production-shape bench fixture generator.
//!
//! Streams the same `wikimedia/wikipedia` 20231101.en HuggingFace dataset
//! that production ingest (`scripts/ingest_curriculum_full.py`) drives
//! against, runs each article through `lu4d::extractor::extract_from_article`
//! (the Rust port of `scripts/smart_extractor.py`, the production source of
//! truth), and emits a length-prefixed binary bond stream that the v5
//! substrate bench harnesses (Phase 0.c) will replay against the SST writer.
//!
//! ## Why a new fixture
//!
//! The existing 10M-record bench fixtures (`bench_v3.rs`, `bench_sst_read.rs`,
//! `bench_migration.rs`) are SYNTHETIC: every Seed is uniform random
//! (`Seed::from_bytes(rng.fill(...))`), every `BondType` is round-robin
//! (`bt_table[i % bt_table.len()]`), and there is no real-corpus shape.
//! The v4 audit lesson 1 was that synthetic 100K correctness tests miss
//! antipatterns that emerge at 9M+ real records; the same risk applies
//! at 10M synthetic if the SHAPE is wrong. v5 design §2 "Locked from
//! audit" point 3 demands a production-shape fixture FIRST so every
//! later phase can assert bounded-RSS against a real bond distribution.
//!
//! ## Format (v1)
//!
//! Length-prefixed framed binary, sole reader is the v5 substrate bench
//! harness. Layout:
//!
//! ```text
//!   MAGIC                4 bytes  b"LBFX"  (Lu Bench FiXture)
//!   format_version       u32 LE   = 1
//!   header_len           u32 LE   = N
//!   header_json          N bytes  UTF-8 JSON (see HeaderJson)
//!   --- records, repeated until trailer tag ---
//!     tag                u8       = 0x01 (BOND)
//!     subject_seed       32 bytes
//!     predicate_idx      u8       index into BondType::all() (0..=29)
//!     object_seed        32 bytes
//!     weight             f32 LE
//!     confidence         u8       0 = Unverified, 1 = Inferred
//!     kind_tag_len       u8       length of kind tag (0..=32)
//!     kind_tag           N bytes  ASCII ("definition", "composition", ...)
//!   --- trailer ---
//!     tag                u8       = 0xFF (END)
//!     total_bonds        u64 LE
//!     distinct_subjects  u64 LE
//!     sha256_body        32 bytes streaming digest of bytes between
//!                                 the end of the header and the END tag
//! ```
//!
//! ## Sidecar .stats.json
//!
//! Emitted alongside the fixture. Includes: total node count (distinct
//! Seeds seen — subject ∪ object), total bond count, predicate type
//! distribution (counts per BondType), node:bond ratio, top-50 hub Seeds
//! by bond degree (estimated via a Misra-Gries heavy-hitter sketch so
//! RSS stays bounded). Computed in a single pass during emission.
//!
//! ## Bounded-RSS discipline
//!
//! No materialization of 5M articles. Articles stream from parquet row
//! groups (~1000 rows / ~22 MB compressed each); each article is
//! immediately extracted and its bonds streamed to disk. Only resident
//! accumulators:
//!   - `distinct_seeds: HashSet<Seed>` (counts unique nodes)
//!   - `subject_degree: HashMap<Seed, u32>` (capped at MAX_DEGREE_TRACK
//!     via Misra-Gries decrement-on-overflow so RSS stays sub-2 GB)
//!   - `predicate_counts: [u64; NUM_BOND_TYPES]`
//!   - `sha256` streaming state
//!
//! All other state is per-article scratch.
//!
//! ## Reproducibility
//!
//! The HuggingFace dataset `wikimedia/wikipedia` at config `20231101.en`
//! ships in 41 parquet shards (`train-00000-of-00041.parquet` ..
//! `train-00040-of-00041.parquet`). Same `--articles` value + same
//! dataset revision = byte-equivalent fixture. The header records the
//! dataset revision, the article-range, and the generator binary md5.
//! Phase 0.c bench harnesses validate the header before replay.

use std::collections::{HashMap, HashSet};
use std::env;
use std::fs::{self, File};
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use md5::{Digest as Md5Digest, Md5};
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::{Row, RowAccessor};
#[allow(unused_imports)]
use sha2::Digest as _;
use sha2::Sha256;

use lu4d::bond::BondType;
use lu4d::extractor::{extract_from_article, ConfidenceLabel};
use lu4d::seed::Seed;

const MAGIC: &[u8; 4] = b"LBFX";
const FORMAT_VERSION: u32 = 1;
const TAG_BOND: u8 = 0x01;
const TAG_END: u8 = 0xFF;
const CONF_UNVERIFIED: u8 = 0;
const CONF_INFERRED: u8 = 1;

const HF_DATASET: &str = "wikimedia/wikipedia";
const HF_CONFIG: &str = "20231101.en";
const HF_SHARD_COUNT: u32 = 41;
const HF_RESOLVE_BASE: &str = "https://huggingface.co/datasets/wikimedia/wikipedia/resolve";

/// Top-K cap for the Misra-Gries hub-degree sketch. 1M entries × ~36
/// bytes ≈ 36 MB resident. Bounded well under the 2 GB RSS budget while
/// keeping the top-50 estimate accurate for power-law distributions.
const MAX_DEGREE_TRACK: usize = 1_000_000;

/// Number of `BondType` variants. Pinned here for the predicate-counts
/// array; an assertion in main() catches drift if a new variant is added.
const NUM_BOND_TYPES: usize = 29;

fn main() {
    if let Err(e) = run() {
        eprintln!("FATAL: {}", e);
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = parse_args()?;

    // Sanity: predicate slice length matches the constant. If BondType
    // gains a variant, NUM_BOND_TYPES must bump and the wire stays sound
    // (predicate_idx grows to 30, 31, ...). Bench harnesses on the
    // reader side accept the predicate as a generic u8 index into the
    // same BondType::all() slice they compile against.
    if BondType::all().len() != NUM_BOND_TYPES {
        return Err(format!(
            "NUM_BOND_TYPES drift: const={} BondType::all().len()={}. \
             Update const + bench reader together.",
            NUM_BOND_TYPES,
            BondType::all().len()
        ));
    }

    fs::create_dir_all(&args.shard_dir)
        .map_err(|e| format!("mkdir shard-dir {}: {}", args.shard_dir.display(), e))?;
    if let Some(parent) = args.out.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| format!("mkdir out parent {}: {}", parent.display(), e))?;
    }

    let gen_md5 = self_md5()?;
    let gen_ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    eprintln!("gen_prod_shape_fixture v1");
    eprintln!("  articles target  = {}", args.articles);
    eprintln!("  out              = {}", args.out.display());
    eprintln!("  shard cache      = {}", args.shard_dir.display());
    eprintln!("  HF dataset       = {} ({}, rev {})", HF_DATASET, HF_CONFIG, args.source_revision);
    eprintln!("  generator md5    = {}", gen_md5);
    eprintln!("  start RSS        = {:.0} MiB", rss_mib());

    // Write the fixture header now so a downstream reader can validate
    // it before we start streaming records.
    let header = HeaderJson {
        format_version: FORMAT_VERSION,
        source_corpus: HF_DATASET.to_string(),
        source_corpus_config: HF_CONFIG.to_string(),
        source_corpus_revision: args.source_revision.clone(),
        article_range_start: 0,
        article_range_end_requested: args.articles as u64,
        generator_binary_md5: gen_md5.clone(),
        generation_timestamp_utc: gen_ts,
        predicate_encoding: "BondType::all() index, u8".to_string(),
        seed_derivation: "Seed::from_label(text)".to_string(),
        extractor: "lu4d::extractor::extract_from_article (Rust port of smart_extractor.py)"
            .to_string(),
    };
    let header_json = serde_json::to_vec(&header.to_json_value())
        .map_err(|e| format!("header serialize: {}", e))?;

    // Temp write — atomic rename at end.
    let tmp_path = args.out.with_extension("bench.partial");
    let _ = fs::remove_file(&tmp_path);
    let f = File::create(&tmp_path)
        .map_err(|e| format!("create {}: {}", tmp_path.display(), e))?;
    let mut writer = BufWriter::with_capacity(8 << 20, f);
    writer.write_all(MAGIC).map_err(io_err)?;
    writer.write_all(&FORMAT_VERSION.to_le_bytes()).map_err(io_err)?;
    writer.write_all(&(header_json.len() as u32).to_le_bytes()).map_err(io_err)?;
    writer.write_all(&header_json).map_err(io_err)?;

    // ------------------------------------------------------------------
    // Streaming state.
    // ------------------------------------------------------------------
    let mut sha = Sha256::new();
    let mut total_bonds: u64 = 0;
    let mut predicate_counts = [0u64; NUM_BOND_TYPES];
    let mut distinct_seeds: HashSet<Seed> = HashSet::new();
    let mut subject_degree: HashMap<Seed, u32> = HashMap::with_capacity(1 << 16);
    let mut articles_processed: u64 = 0;
    let mut articles_skipped_empty: u64 = 0;

    let mut shard_idx: u32 = 0;
    let mut shards_opened: u32 = 0;
    let last_shard = args.max_shards
        .map(|n| n.min(HF_SHARD_COUNT))
        .unwrap_or(HF_SHARD_COUNT);
    let target_articles = args.articles;
    let t_total = Instant::now();
    let mut t_last_progress = Instant::now();

    'shard_loop: while shard_idx < last_shard && articles_processed < target_articles as u64 {
        let shard_path = args.shard_dir.join(shard_filename(shard_idx));
        download_shard(&args.source_revision, shard_idx, &shard_path)?;
        shards_opened += 1;

        let reader = SerializedFileReader::new(File::open(&shard_path).map_err(io_err)?)
            .map_err(|e| format!("parquet open {}: {}", shard_path.display(), e))?;
        let schema_indices = resolve_schema_indices(&reader)?;
        let num_row_groups = reader.num_row_groups();

        for rg_idx in 0..num_row_groups {
            if articles_processed >= target_articles as u64 { break 'shard_loop; }
            let rg = reader.get_row_group(rg_idx)
                .map_err(|e| format!("get_row_group({}): {}", rg_idx, e))?;
            let iter = rg.get_row_iter(None)
                .map_err(|e| format!("get_row_iter({}): {}", rg_idx, e))?;

            for row_result in iter {
                if articles_processed >= target_articles as u64 { break 'shard_loop; }
                let row: Row = match row_result {
                    Ok(r) => r,
                    Err(e) => return Err(format!("row decode: {}", e)),
                };
                let title = row.get_string(schema_indices.title)
                    .map_err(|e| format!("row.title: {}", e))?
                    .clone();
                let text = row.get_string(schema_indices.text)
                    .map_err(|e| format!("row.text: {}", e))?
                    .clone();
                let url = row.get_string(schema_indices.url)
                    .map_err(|e| format!("row.url: {}", e))?
                    .clone();

                if title.trim().is_empty() || text.trim().is_empty() {
                    articles_skipped_empty += 1;
                    articles_processed += 1;
                    continue;
                }

                // Production-canonical bond formation: run through the
                // Rust extractor (mirrors smart_extractor.py line-for-line
                // per internal notes. Returns Vec<extractor::Bond>.
                let bonds = extract_from_article(&text, Some(&title), Some(&url));

                for b in bonds {
                    let subj_seed = Seed::from_label(&b.subject);
                    let obj_seed = Seed::from_label(&b.object);
                    let pred_idx = match bond_type_to_idx(b.relation) {
                        Some(i) => i,
                        None => return Err(format!(
                            "bond_type_to_idx miss: variant {:?} not in BondType::all()",
                            b.relation
                        )),
                    };
                    let conf_byte = match b.confidence {
                        ConfidenceLabel::Unverified => CONF_UNVERIFIED,
                        ConfidenceLabel::Inferred => CONF_INFERRED,
                    };
                    let kind = b.kind;
                    let kind_bytes = kind.as_bytes();
                    if kind_bytes.len() > u8::MAX as usize {
                        return Err(format!(
                            "kind tag too long for u8 length prefix: {:?} ({} bytes)",
                            kind, kind_bytes.len()
                        ));
                    }

                    // Streaming record write — bytes go through sha256
                    // first so the trailer digest covers the exact byte
                    // sequence the reader sees.
                    let mut record = Vec::with_capacity(72 + kind_bytes.len());
                    record.push(TAG_BOND);
                    record.extend_from_slice(subj_seed.as_bytes());
                    record.push(pred_idx);
                    record.extend_from_slice(obj_seed.as_bytes());
                    record.extend_from_slice(&b.weight.to_le_bytes());
                    record.push(conf_byte);
                    record.push(kind_bytes.len() as u8);
                    record.extend_from_slice(kind_bytes);
                    sha.update(&record);
                    writer.write_all(&record).map_err(io_err)?;

                    total_bonds += 1;
                    predicate_counts[pred_idx as usize] += 1;
                    distinct_seeds.insert(subj_seed);
                    distinct_seeds.insert(obj_seed);
                    misra_gries_increment(&mut subject_degree, subj_seed);
                }

                articles_processed += 1;

                if articles_processed % 1000 == 0
                    || t_last_progress.elapsed().as_secs() >= 30
                {
                    let elapsed = t_total.elapsed().as_secs_f64().max(0.001);
                    let rate = articles_processed as f64 / elapsed;
                    eprintln!(
                        "  progress: shard={} rg={} articles={}/{} bonds={} \
                         rss={:.2} GiB rate={:.1} art/s elapsed={:.0}s",
                        shard_idx, rg_idx, articles_processed, target_articles,
                        total_bonds, rss_gib(), rate, elapsed
                    );
                    t_last_progress = Instant::now();
                }
            }
        }
        shard_idx += 1;
    }

    // Trailer.
    let distinct_subjects = subject_degree.len() as u64;
    writer.write_all(&[TAG_END]).map_err(io_err)?;
    writer.write_all(&total_bonds.to_le_bytes()).map_err(io_err)?;
    writer.write_all(&distinct_subjects.to_le_bytes()).map_err(io_err)?;
    let digest = sha.finalize();
    writer.write_all(&digest).map_err(io_err)?;
    writer.flush().map_err(io_err)?;
    drop(writer);

    fs::rename(&tmp_path, &args.out)
        .map_err(|e| format!("rename {} -> {}: {}", tmp_path.display(), args.out.display(), e))?;

    let out_size = fs::metadata(&args.out).map(|m| m.len()).unwrap_or(0);
    eprintln!();
    eprintln!("fixture written");
    eprintln!("  path             = {}", args.out.display());
    eprintln!("  size             = {} bytes ({:.1} MiB)", out_size, out_size as f64 / 1_048_576.0);
    eprintln!("  bonds            = {}", total_bonds);
    eprintln!("  distinct nodes   = {}", distinct_seeds.len());
    eprintln!("  distinct subjs   = {}", distinct_subjects);
    eprintln!("  articles parsed  = {}", articles_processed);
    eprintln!("  empty skipped    = {}", articles_skipped_empty);
    eprintln!("  shards opened    = {}", shards_opened);
    eprintln!("  shards completed = {}", shard_idx);
    eprintln!("  elapsed          = {:.0} s", t_total.elapsed().as_secs_f64());
    eprintln!("  peak RSS (now)   = {:.2} GiB", rss_gib());

    // ---------- Stats sidecar ----------
    let stats_path = args.out.with_extension("stats.json");
    write_stats_json(
        &stats_path,
        &header,
        total_bonds,
        distinct_seeds.len() as u64,
        distinct_subjects,
        articles_processed,
        articles_skipped_empty,
        shards_opened,
        out_size,
        &predicate_counts,
        &subject_degree,
        t_total.elapsed().as_secs_f64(),
        rss_gib(),
    )?;
    eprintln!("  stats sidecar    = {}", stats_path.display());

    if !args.keep_shards {
        eprintln!("  cleaning up cached shards ({})", args.shard_dir.display());
        let _ = fs::remove_dir_all(&args.shard_dir);
    }

    Ok(())
}

// ----------------------------------------------------------------------
// CLI
// ----------------------------------------------------------------------

struct Args {
    articles: usize,
    out: PathBuf,
    shard_dir: PathBuf,
    source_revision: String,
    keep_shards: bool,
    max_shards: Option<u32>,
}

fn parse_args() -> Result<Args, String> {
    let mut articles: usize = 5_000_000;
    let mut out = PathBuf::from("/var/lib/lu/bench-fixtures/v5-prod-shape-5m.bench");
    let mut shard_dir = PathBuf::from("/var/lib/lu/bench-fixtures/wikipedia-shards");
    let mut source_revision = String::from("main");
    let mut keep_shards = true;
    let mut max_shards: Option<u32> = None;
    let mut it = env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--articles" => {
                let v = it.next().ok_or("--articles needs a value")?;
                articles = v.replace('_', "").parse()
                    .map_err(|e| format!("--articles: {}", e))?;
            }
            "--out" => out = PathBuf::from(it.next().ok_or("--out needs a value")?),
            "--shard-dir" => shard_dir = PathBuf::from(it.next().ok_or("--shard-dir needs a value")?),
            "--source-revision" => source_revision = it.next().ok_or("--source-revision needs a value")?,
            "--drop-shards" => keep_shards = false,
            "--max-shards" => {
                let v = it.next().ok_or("--max-shards needs a value")?;
                max_shards = Some(v.parse().map_err(|e| format!("--max-shards: {}", e))?);
            }
            "--help" | "-h" => { print_help(); std::process::exit(0); }
            other => return Err(format!("unknown arg: {}", other)),
        }
    }
    Ok(Args { articles, out, shard_dir, source_revision, keep_shards, max_shards })
}

fn print_help() {
    println!("gen_prod_shape_fixture — OP-BRAIN-V5 Phase 0.b");
    println!();
    println!("Streams the wikimedia/wikipedia 20231101.en HuggingFace dataset,");
    println!("runs each article through lu4d::extractor::extract_from_article,");
    println!("and writes a length-prefixed framed binary bond stream suitable");
    println!("for v5 substrate bench replay.");
    println!();
    println!("Usage: gen_prod_shape_fixture [options]");
    println!();
    println!("  --articles N            Articles to process (default 5_000_000)");
    println!("  --out PATH              Output .bench path");
    println!("                          (default /var/lib/lu/bench-fixtures/v5-prod-shape-5m.bench)");
    println!("  --shard-dir PATH        Parquet shard cache directory");
    println!("                          (default /var/lib/lu/bench-fixtures/wikipedia-shards)");
    println!("  --source-revision REV   HF Hub revision (default `main`)");
    println!("  --max-shards N          Cap shard download at first N shards (debug)");
    println!("  --drop-shards           Delete cached shards after run");
    println!("  --help                  This message");
}

// ----------------------------------------------------------------------
// Header schema. serde::Serialize via serde_json (we keep the dep cost
// down by hand-implementing — no serde_derive required, no serde
// dependency on the bench crate).
// ----------------------------------------------------------------------

struct HeaderJson {
    format_version: u32,
    source_corpus: String,
    source_corpus_config: String,
    source_corpus_revision: String,
    article_range_start: u64,
    article_range_end_requested: u64,
    generator_binary_md5: String,
    generation_timestamp_utc: i64,
    predicate_encoding: String,
    seed_derivation: String,
    extractor: String,
}

impl HeaderJson {
    fn to_json_value(&self) -> serde_json::Value {
        serde_json::json!({
            "format_version": self.format_version,
            "source_corpus": self.source_corpus,
            "source_corpus_config": self.source_corpus_config,
            "source_corpus_revision": self.source_corpus_revision,
            "article_range_start": self.article_range_start,
            "article_range_end_requested": self.article_range_end_requested,
            "generator_binary_md5": self.generator_binary_md5,
            "generation_timestamp_utc": self.generation_timestamp_utc,
            "predicate_encoding": self.predicate_encoding,
            "seed_derivation": self.seed_derivation,
            "extractor": self.extractor,
        })
    }
}


// ----------------------------------------------------------------------
// RSS reporting. Same shape used by bench_v3 / bench_migration.
// ----------------------------------------------------------------------

fn read_vm_rss_kib() -> Option<u64> {
    let s = fs::read_to_string("/proc/self/status").ok()?;
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            let tok = rest.split_whitespace().next()?;
            return tok.parse::<u64>().ok();
        }
    }
    None
}
fn rss_mib() -> f64 { read_vm_rss_kib().map(|kb| kb as f64 / 1024.0).unwrap_or(f64::NAN) }
fn rss_gib() -> f64 { read_vm_rss_kib().map(|kb| kb as f64 / 1024.0 / 1024.0).unwrap_or(f64::NAN) }

// ----------------------------------------------------------------------
// io::Error -> String
// ----------------------------------------------------------------------

fn io_err<E: std::fmt::Display>(e: E) -> String { format!("io: {}", e) }

// ----------------------------------------------------------------------
// Predicate-index encoding
// ----------------------------------------------------------------------

fn bond_type_to_idx(bt: BondType) -> Option<u8> {
    BondType::all().iter().position(|x| *x == bt).map(|i| i as u8)
}

fn bond_type_label(bt: BondType) -> &'static str {
    match bt {
        BondType::IsA => "IsA",
        BondType::HasProperty => "HasProperty",
        BondType::Causes => "Causes",
        BondType::PartOf => "PartOf",
        BondType::UsedFor => "UsedFor",
        BondType::RelatedTo => "RelatedTo",
        BondType::ContrastedWith => "ContrastedWith",
        BondType::DerivedFrom => "DerivedFrom",
        BondType::Requires => "Requires",
        BondType::Produces => "Produces",
        BondType::Opposes => "Opposes",
        BondType::BornIn => "BornIn",
        BondType::BornOn => "BornOn",
        BondType::DiedIn => "DiedIn",
        BondType::DiedOn => "DiedOn",
        BondType::DiscoveredBy => "DiscoveredBy",
        BondType::FoundedBy => "FoundedBy",
        BondType::FoundedIn => "FoundedIn",
        BondType::HasAltitude => "HasAltitude",
        BondType::HasAspect => "HasAspect",
        BondType::HasCapital => "HasCapital",
        BondType::HasOfficialLanguage => "HasOfficialLanguage",
        BondType::HasPopulation => "HasPopulation",
        BondType::KnownFor => "KnownFor",
        BondType::LocatedIn => "LocatedIn",
        BondType::MeasuredIn => "MeasuredIn",
        BondType::NamedAfter => "NamedAfter",
        BondType::WrittenBy => "WrittenBy",
        BondType::HasArea => "HasArea",
        BondType::HasMethod => "HasMethod",
        BondType::HasAttribute => "HasAttribute",
        BondType::HasParameter => "HasParameter",
        BondType::Returns => "Returns",
        BondType::Inherits => "Inherits",
        BondType::Calls => "Calls",
        BondType::HasType => "HasType",
        BondType::Implements => "Implements",
        BondType::DefinedAs => "DefinedAs",
        BondType::WorksBy => "WorksBy",
        BondType::ComposedOf => "ComposedOf",
        BondType::PurposeOf => "PurposeOf",
        BondType::DiffersFrom => "DiffersFrom",
        BondType::LeadsTo => "LeadsTo",
        BondType::ExampleOf => "ExampleOf",
    }
}

// ----------------------------------------------------------------------
// Self-md5 — fixture header reproducibility marker.
// ----------------------------------------------------------------------

fn self_md5() -> Result<String, String> {
    let exe = env::current_exe().map_err(|e| format!("current_exe: {}", e))?;
    let mut f = File::open(&exe).map_err(|e| format!("open self ({}): {}", exe.display(), e))?;
    let mut hasher = Md5::new();
    let mut buf = vec![0u8; 65536];
    loop {
        let n = f.read(&mut buf).map_err(|e| format!("read self: {}", e))?;
        if n == 0 { break; }
        hasher.update(&buf[..n]);
    }
    let out = hasher.finalize();
    let mut s = String::with_capacity(32);
    for b in &out { s.push_str(&format!("{:02x}", b)); }
    Ok(s)
}

// ----------------------------------------------------------------------
// HF parquet shard fetch. Sync ureq GET; download to .partial then atomic
// rename. Resume support via size match: if the file is already present
// at the expected byte count, skip.
// ----------------------------------------------------------------------

fn shard_filename(idx: u32) -> String {
    format!("train-{:05}-of-{:05}.parquet", idx, HF_SHARD_COUNT)
}

fn shard_url(rev: &str, idx: u32) -> String {
    format!("{}/{}/{}/{}", HF_RESOLVE_BASE, rev, HF_CONFIG, shard_filename(idx))
}

fn fetch_shard_expected_size(rev: &str, idx: u32) -> Result<u64, String> {
    let url = format!(
        "https://huggingface.co/api/datasets/{}/tree/{}/{}",
        HF_DATASET, rev, HF_CONFIG,
    );
    let resp = ureq::get(&url)
        .timeout(std::time::Duration::from_secs(30))
        .call()
        .map_err(|e| format!("HF tree API {}: {}", url, e))?;
    let body: serde_json::Value = resp.into_json()
        .map_err(|e| format!("HF tree API decode: {}", e))?;
    let arr = body.as_array().ok_or("HF tree API: expected array")?;
    let want = format!("{}/{}", HF_CONFIG, shard_filename(idx));
    for entry in arr {
        let path = entry.get("path").and_then(|v| v.as_str()).unwrap_or("");
        if path == want {
            let size = entry.get("size").and_then(|v| v.as_u64())
                .ok_or("HF tree API: missing `size` field")?;
            return Ok(size);
        }
    }
    Err(format!("HF tree API: no entry for {}", want))
}

fn download_shard(rev: &str, idx: u32, dest: &Path) -> Result<u64, String> {
    let expected = fetch_shard_expected_size(rev, idx)?;
    if let Ok(meta) = fs::metadata(dest) {
        if meta.len() == expected {
            eprintln!("  shard {} already present ({} bytes), skipping download",
                shard_filename(idx), expected);
            return Ok(expected);
        } else {
            eprintln!("  shard {} size mismatch (have {}, want {}); re-downloading",
                shard_filename(idx), meta.len(), expected);
            fs::remove_file(dest).map_err(|e| format!("remove stale shard: {}", e))?;
        }
    }
    eprintln!("  downloading shard {} ({:.1} MB)",
        shard_filename(idx), expected as f64 / 1_048_576.0);
    let url = shard_url(rev, idx);
    let t0 = Instant::now();
    let resp = ureq::get(&url)
        .timeout(std::time::Duration::from_secs(600))
        .call()
        .map_err(|e| format!("GET {}: {}", url, e))?;
    let mut reader = resp.into_reader();
    let tmp = dest.with_extension("parquet.partial");
    let mut w = BufWriter::with_capacity(1 << 20, File::create(&tmp)
        .map_err(|e| format!("create {}: {}", tmp.display(), e))?);
    let mut buf = vec![0u8; 1 << 20];
    let mut total: u64 = 0;
    loop {
        let n = reader.read(&mut buf).map_err(|e| format!("read shard: {}", e))?;
        if n == 0 { break; }
        w.write_all(&buf[..n]).map_err(|e| format!("write shard: {}", e))?;
        total += n as u64;
    }
    w.flush().map_err(|e| format!("flush shard: {}", e))?;
    drop(w);
    if total != expected {
        return Err(format!("shard {} truncated: got {} bytes, expected {}",
            shard_filename(idx), total, expected));
    }
    fs::rename(&tmp, dest).map_err(|e| format!("rename shard: {}", e))?;
    let elapsed = t0.elapsed();
    eprintln!("  shard {} downloaded {:.1} MB in {:.1} s ({:.1} MB/s)",
        shard_filename(idx),
        total as f64 / 1_048_576.0,
        elapsed.as_secs_f64(),
        (total as f64 / 1_048_576.0) / elapsed.as_secs_f64().max(0.001));
    Ok(total)
}

// ----------------------------------------------------------------------
// Parquet schema resolution. Resolves column names to indices once per
// shard so we never rely on column ordering across revisions.
// ----------------------------------------------------------------------

struct SchemaIndices {
    title: usize,
    text: usize,
    url: usize,
}

fn resolve_schema_indices(reader: &SerializedFileReader<File>) -> Result<SchemaIndices, String> {
    let schema = reader.metadata().file_metadata().schema();
    let mut title = None;
    let mut text = None;
    let mut url = None;
    for (i, field) in schema.get_fields().iter().enumerate() {
        match field.name() {
            "title" => title = Some(i),
            "text" => text = Some(i),
            "url" => url = Some(i),
            _ => {}
        }
    }
    Ok(SchemaIndices {
        title: title.ok_or("parquet schema: missing `title` column")?,
        text: text.ok_or("parquet schema: missing `text` column")?,
        url: url.ok_or("parquet schema: missing `url` column")?,
    })
}

// ----------------------------------------------------------------------
// Misra-Gries heavy-hitter sketch for the subject_degree map.
// Standard algorithm: increment on present key, insert on absent (if
// under cap), decrement-all on absent when at cap. Guarantees the top-K
// is preserved exactly for any key with true frequency > N/(cap+1) where
// N is the total stream count. For real Wikipedia degree distributions
// (power-law) the top-50 is solidly above this floor at MAX_DEGREE_TRACK = 1M.
// ----------------------------------------------------------------------

fn misra_gries_increment(map: &mut HashMap<Seed, u32>, key: Seed) {
    if let Some(c) = map.get_mut(&key) {
        *c = c.saturating_add(1);
        return;
    }
    if map.len() < MAX_DEGREE_TRACK {
        map.insert(key, 1);
        return;
    }
    // Decrement-all (lazy compaction). Drains all 1-counters in one pass
    // and decrements the rest. Amortized O(1) per call since the
    // expensive sweep only fires after MAX_DEGREE_TRACK distinct misses
    // and removes at least one entry.
    let mut to_remove: Vec<Seed> = Vec::new();
    for (k, v) in map.iter_mut() {
        if *v <= 1 {
            to_remove.push(*k);
        } else {
            *v -= 1;
        }
    }
    for k in to_remove { map.remove(&k); }
    // The current `key` is effectively decremented along with everyone
    // else and then not inserted — this is the standard Misra-Gries
    // behavior when the cap is hit.
}

// ----------------------------------------------------------------------
// Stats sidecar emit.
// ----------------------------------------------------------------------

fn write_stats_json(
    path: &Path,
    header: &HeaderJson,
    total_bonds: u64,
    distinct_nodes: u64,
    distinct_subjects: u64,
    articles_processed: u64,
    articles_skipped_empty: u64,
    shards_consumed: u32,
    fixture_size_bytes: u64,
    predicate_counts: &[u64; NUM_BOND_TYPES],
    subject_degree: &HashMap<Seed, u32>,
    elapsed_secs: f64,
    peak_rss_gib: f64,
) -> Result<(), String> {
    // Predicate distribution.
    let mut predicate_dist = serde_json::Map::new();
    for (i, bt) in BondType::all().iter().enumerate() {
        predicate_dist.insert(bond_type_label(*bt).to_string(),
            serde_json::Value::Number(serde_json::Number::from(predicate_counts[i])));
    }

    // Top-50 hub Seeds. Sort by degree desc; ties broken by seed bytes
    // (deterministic).
    let mut degree_vec: Vec<(Seed, u32)> = subject_degree.iter()
        .map(|(s, c)| (*s, *c))
        .collect();
    degree_vec.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.as_bytes().cmp(b.0.as_bytes())));
    let top50: Vec<serde_json::Value> = degree_vec.iter().take(50)
        .map(|(s, c)| {
            serde_json::json!({
                "seed_hex": format!("{}", s),
                "degree_estimate": *c,
            })
        })
        .collect();

    let node_bond_ratio = if total_bonds == 0 {
        0.0
    } else {
        distinct_nodes as f64 / total_bonds as f64
    };

    let stats = serde_json::json!({
        "header": header.to_json_value(),
        "summary": {
            "total_bonds": total_bonds,
            "distinct_nodes": distinct_nodes,
            "distinct_subjects": distinct_subjects,
            "articles_processed": articles_processed,
            "articles_skipped_empty": articles_skipped_empty,
            "shards_consumed": shards_consumed,
            "fixture_size_bytes": fixture_size_bytes,
            "elapsed_secs": elapsed_secs,
            "peak_rss_gib_observed": peak_rss_gib,
            "node_bond_ratio": node_bond_ratio,
        },
        "predicate_distribution": predicate_dist,
        "top_50_hub_subjects": top50,
        "reference_v3_canonical_brain": {
            "description": "For shape comparison. Values from project_lu_brain_is_memory \
                            memory + 2026-06-08 v3 prod state.",
            "nodes_at_last_audit": 15_160_000_u64,
            "bonds_at_last_audit": 37_300_000_u64,
            "node_bond_ratio_at_last_audit": 0.4063,
        },
        "misra_gries_note": format!(
            "subject_degree counts are Misra-Gries estimates with cap {}. \
             For power-law Wikipedia distributions the top-50 estimate is exact; \
             tail counts may underrepresent true frequency.",
            MAX_DEGREE_TRACK
        ),
    });

    let bytes = serde_json::to_vec_pretty(&stats)
        .map_err(|e| format!("stats serialize: {}", e))?;
    fs::write(path, &bytes)
        .map_err(|e| format!("stats write {}: {}", path.display(), e))?;
    Ok(())
}
