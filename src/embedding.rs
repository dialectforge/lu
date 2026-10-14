//! OP-OFFNESS-1 — semantic embedding integration seam (Cattle Dog KEYSTONE port).
//!
//! Cattle Dog's keystone finding internal notes: replacing a lexical
//! bag-of-words stub with a real BGE-small-en-v1.5 embedder flipped
//! true-paraphrase cosine 0.00 -> 0.75 and off-ness false-alarms ~100% -> 0.
//! That work lived only on the local `cattledog-semantic-scar` branch of the
//! v2 codebase and was never merged into canonical v4.
//!
//! This module lands the INTEGRATION SEAM in canonical v4:
//!   * an `Embedder` trait (the swappable comparison surface),
//!   * a `cosine` similarity primitive,
//!   * an always-available deterministic `LexicalEmbedder` fallback, and
//!   * a real `BgeEmbedder` behind the OFF-BY-DEFAULT `bge` cargo feature.
//!
//! IMPORTANT (no-contention build): the default build does NOT compile the
//! ONNX/fastembed stack and does NOT download the 133 MB model. Nothing here
//! re-embeds the canonical brain — that is the deferred write step, to be run
//! in a quiet ingest window under John's go. The off-ness detector consumes
//! this trait, so the comparison logic is fully testable on sample vectors
//! today with the lexical fallback.

/// Cosine similarity of two equal-length vectors. Returns 0.0 ("no signal")
/// for length-mismatched, empty, or zero-norm inputs — a safe default that
/// makes the off-ness detector fail *closed* (treats unknowable as ungrounded).
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

/// The swappable semantic comparison surface. The off-ness detector and the
/// (future) semantic scar set depend only on this trait, so the lexical
/// fallback and the real BGE model are drop-in interchangeable.
pub trait Embedder: Send + Sync {
    /// Embed `text` into a fixed-dimension vector.
    fn embed(&self, text: &str) -> Vec<f32>;
    /// Embedding dimensionality.
    fn dim(&self) -> usize;
    /// Stable identifier for logging / audit (which embedder graded a run).
    fn name(&self) -> &'static str;
    /// Convenience: cosine similarity between two texts under this embedder.
    fn similarity(&self, a: &str, b: &str) -> f32 {
        cosine(&self.embed(a), &self.embed(b))
    }
}

/// FNV-1a 64-bit hash — deterministic, dependency-free, used to bucket tokens.
fn fnv1a(s: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Deterministic lexical fallback embedder: hashed bag-of-words projected into
/// a fixed-dimension, L2-normalized vector. This is the SAME class of stub
/// Cattle Dog started from (vocabulary-overlap, not true paraphrase) — kept as
/// the always-available fallback so the off-ness seam works with zero heavy
/// dependencies. Default dim = 384 to match BGE-small so the two are
/// vector-compatible.
pub struct LexicalEmbedder {
    pub dim: usize,
}

impl LexicalEmbedder {
    pub fn new() -> Self {
        Self { dim: 384 }
    }
    pub fn with_dim(dim: usize) -> Self {
        Self { dim: dim.max(1) }
    }
}

impl Default for LexicalEmbedder {
    fn default() -> Self {
        Self::new()
    }
}

impl Embedder for LexicalEmbedder {
    fn embed(&self, text: &str) -> Vec<f32> {
        let mut v = vec![0f32; self.dim];
        let lower = text.to_lowercase();
        for tok in lower
            .split(|c: char| !c.is_alphanumeric())
            .filter(|t| t.len() >= 2)
        {
            let idx = (fnv1a(tok) % self.dim as u64) as usize;
            v[idx] += 1.0;
        }
        let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if n > 0.0 {
            for x in v.iter_mut() {
                *x /= n;
            }
        }
        v
    }
    fn dim(&self) -> usize {
        self.dim
    }
    fn name(&self) -> &'static str {
        "lexical-fnv"
    }
}

/// A test/fixture embedder that returns caller-supplied vectors for exact
/// strings (falling back to zeros). Lets harness + unit tests drive the
/// off-ness comparison with hand-crafted "sample vectors" deterministically,
/// independent of any model.
pub struct FixedEmbedder {
    pub dim: usize,
    map: std::collections::HashMap<String, Vec<f32>>,
}

impl FixedEmbedder {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            map: std::collections::HashMap::new(),
        }
    }
    pub fn insert(&mut self, text: &str, vec: Vec<f32>) {
        self.map.insert(text.to_string(), vec);
    }
}

impl Embedder for FixedEmbedder {
    fn embed(&self, text: &str) -> Vec<f32> {
        self.map
            .get(text)
            .cloned()
            .unwrap_or_else(|| vec![0f32; self.dim])
    }
    fn dim(&self) -> usize {
        self.dim
    }
    fn name(&self) -> &'static str {
        "fixed-sample"
    }
}

// ---------------------------------------------------------------------------
// Real BGE-small-en-v1.5 embedder — the Cattle Dog KEYSTONE.
//
// Gated behind the off-by-default `bge` cargo feature so that:
//   (a) the default build + the production `lu_4_mcp` binary are byte-identical
//       and pull NO ONNX/fastembed dependency tree, and
//   (b) the heavy compile + 133 MB model download only happen at the deferred
//       cutover step, never under the running StarCoderData ingest.
//
// The `fastembed` API binding is verified/adjusted at cutover when the feature
// is first compiled. The off-ness comparison seam it plugs into is already
// exercised on sample vectors by the lexical + fixed embedders above.
// ---------------------------------------------------------------------------
#[cfg(feature = "bge")]
pub mod bge {
    use super::Embedder;
    use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};

    pub struct BgeEmbedder {
        model: TextEmbedding,
    }

    impl BgeEmbedder {
        /// Load BGE-small-en-v1.5. Downloads the model on first use (cutover
        /// step only). Returns Err if the model cannot be initialized so the
        /// factory can fall back to the lexical embedder.
        pub fn new() -> Result<Self, String> {
            let model = TextEmbedding::try_new(
                InitOptions::new(EmbeddingModel::BGESmallENV15).with_show_download_progress(false),
            )
            .map_err(|e| e.to_string())?;
            Ok(Self { model })
        }
    }

    impl Embedder for BgeEmbedder {
        fn embed(&self, text: &str) -> Vec<f32> {
            self.model
                .embed(vec![text.to_string()], None)
                .ok()
                .and_then(|mut v| v.pop())
                .unwrap_or_default()
        }
        fn dim(&self) -> usize {
            384
        }
        fn name(&self) -> &'static str {
            "bge-small-en-v1.5"
        }
    }
}

/// Factory mirroring Cattle Dog's `semantic_embed()`: use the real BGE model
/// when the `bge` feature is compiled AND the model initializes, otherwise the
/// deterministic lexical fallback. Default builds always return lexical.
pub fn default_embedder() -> Box<dyn Embedder> {
    #[cfg(feature = "bge")]
    {
        if let Ok(e) = bge::BgeEmbedder::new() {
            return Box::new(e);
        }
    }
    Box::new(LexicalEmbedder::new())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_identical_orthogonal_opposite() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let c = vec![0.0, 1.0, 0.0];
        let d = vec![-1.0, 0.0, 0.0];
        assert!((cosine(&a, &b) - 1.0).abs() < 1e-6);
        assert!(cosine(&a, &c).abs() < 1e-6);
        assert!((cosine(&a, &d) + 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_length_mismatch_and_zero_norm_are_zero() {
        assert_eq!(cosine(&[1.0, 2.0], &[1.0]), 0.0);
        assert_eq!(cosine(&[0.0, 0.0], &[0.0, 0.0]), 0.0);
        assert_eq!(cosine(&[], &[]), 0.0);
    }

    #[test]
    fn lexical_identical_text_is_max_similar() {
        let e = LexicalEmbedder::new();
        let s = e.similarity("the capital of france is paris", "the capital of france is paris");
        assert!((s - 1.0).abs() < 1e-6, "identical text sim was {s}");
    }

    #[test]
    fn lexical_overlap_beats_disjoint() {
        let e = LexicalEmbedder::new();
        let related = e.similarity("paris is the capital of france", "the capital of france is paris");
        let unrelated = e.similarity("paris is the capital of france", "photosynthesis converts sunlight");
        assert!(related > unrelated, "related {related} !> unrelated {unrelated}");
        assert!(related > 0.5, "related sim too low: {related}");
        assert!(unrelated < 0.3, "unrelated sim too high: {unrelated}");
    }

    #[test]
    fn lexical_is_deterministic_and_fixed_dim() {
        let e = LexicalEmbedder::new();
        let v1 = e.embed("determinism check");
        let v2 = e.embed("determinism check");
        assert_eq!(v1, v2);
        assert_eq!(v1.len(), 384);
        assert_eq!(e.dim(), 384);
    }

    #[test]
    fn fixed_embedder_returns_sample_vectors() {
        // Grounds the off-ness comparison on hand-crafted "sample vectors":
        // an answer vector strongly aligned with its evidence, weakly with a
        // distractor — exactly the shape the detector keys on.
        let mut fe = FixedEmbedder::new(3);
        fe.insert("answer", vec![1.0, 0.0, 0.0]);
        fe.insert("grounded_evidence", vec![0.95, 0.31, 0.0]);
        fe.insert("distractor", vec![0.0, 1.0, 0.0]);
        let grounded = fe.similarity("answer", "grounded_evidence");
        let off = fe.similarity("answer", "distractor");
        assert!(grounded > 0.9, "grounded sim {grounded}");
        assert!(off < 0.1, "off sim {off}");
    }

    #[test]
    fn default_embedder_is_usable() {
        let e = default_embedder();
        assert!(e.dim() >= 1);
        let s = e.similarity("hello world", "hello world");
        assert!((s - 1.0).abs() < 1e-6);
    }
}
