//! Knowledge nodes — the vertices of the Lu 4D graph.
//!
//! Each node has a cryptographic seed identity, content, a typed
//! knowledge unit, a confidence level, and a 384-dimensional embedding.

use crate::seed::Seed;
use serde::{Deserialize, Serialize};

/// Embedding dimensionality (matches common sentence-transformer models).
pub const EMBEDDING_DIM: usize = 384;

/// Confidence levels for knowledge assertions.
///
/// These map to numeric values used in activation weighting.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
#[archive(check_bytes)]
pub enum Confidence {
    /// Axiomatically true, cannot be overridden.
    Terminal,
    /// Verified by multiple corroborating sources.
    Verified,
    /// Single-source, not yet corroborated.
    Unverified,
    /// Derived through inference or reasoning.
    Inferred,
    /// Contradicted by other evidence.
    Contested,
}

impl Confidence {
    /// Numeric value of this confidence level, used as a multiplier
    /// during spread activation.
    pub fn value(&self) -> f32 {
        match self {
            Confidence::Terminal => 1.0,
            Confidence::Verified => 0.9,
            Confidence::Unverified => 0.6,
            Confidence::Inferred => 0.5,
            Confidence::Contested => 0.2,
        }
    }
}

/// Typed content within a knowledge unit.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
#[archive(check_bytes)]
pub enum KnowledgeContent {
    /// A factual assertion.
    Fact(String),
    /// A concept or definition.
    Concept(String),
    /// A procedural instruction or rule.
    Procedure(String),
    /// An episodic memory or experience.
    Episode(String),
    /// A preference or value judgment.
    Preference(String),
    /// Raw untyped content.
    Raw(String),
    /// Programming knowledge or code snippets.
    Code(String),
    /// Visual knowledge or imagery descriptions.
    Image(String),
    /// Factual definitions.
    Definition(String),
    /// Explanations of how things work.
    Mechanism(String),
    /// Descriptions of what things are composed of.
    Composition(String),
    /// What things are for or their purpose.
    Purpose(String),
    /// Differences between things.
    Contrast(String),
    /// Cause-and-effect relationships.
    Causal(String),
    /// Concrete examples.
    Example(String),
    /// Meta-level node. Used by the Thinking Layer to mark goal nodes
    /// that carry a query rather than a knowledge assertion.
    Meta(String),
}

impl KnowledgeContent {
    /// Extract the inner text regardless of type.
    pub fn text(&self) -> &str {
        match self {
            KnowledgeContent::Fact(s) => s,
            KnowledgeContent::Concept(s) => s,
            KnowledgeContent::Procedure(s) => s,
            KnowledgeContent::Episode(s) => s,
            KnowledgeContent::Preference(s) => s,
            KnowledgeContent::Raw(s) => s,
            KnowledgeContent::Code(s) => s,
            KnowledgeContent::Image(s) => s,
            KnowledgeContent::Definition(s) => s,
            KnowledgeContent::Mechanism(s) => s,
            KnowledgeContent::Composition(s) => s,
            KnowledgeContent::Purpose(s) => s,
            KnowledgeContent::Contrast(s) => s,
            KnowledgeContent::Causal(s) => s,
            KnowledgeContent::Example(s) => s,
            KnowledgeContent::Meta(s) => s,
        }
    }
}

/// Q.1 — typed-KU plumbing. Returns the lowercase variant tag
/// ("fact", "definition", "mechanism", ...) so reasoning prose and
/// the MCP API can surface KnowledgeContent type info without a
/// match arm at every call site. `Raw`, `Concept`, and `Meta` are
/// considered "common-case" and intentionally returned bare so the
/// narration layer can choose to suppress their tag.
impl KnowledgeContent {
    pub fn kind_str(&self) -> &'static str {
        match self {
            KnowledgeContent::Fact(_) => "fact",
            KnowledgeContent::Concept(_) => "concept",
            KnowledgeContent::Procedure(_) => "procedure",
            KnowledgeContent::Episode(_) => "episode",
            KnowledgeContent::Preference(_) => "preference",
            KnowledgeContent::Raw(_) => "raw",
            KnowledgeContent::Code(_) => "code",
            KnowledgeContent::Image(_) => "image",
            KnowledgeContent::Definition(_) => "definition",
            KnowledgeContent::Mechanism(_) => "mechanism",
            KnowledgeContent::Composition(_) => "composition",
            KnowledgeContent::Purpose(_) => "purpose",
            KnowledgeContent::Contrast(_) => "contrast",
            KnowledgeContent::Causal(_) => "causal",
            KnowledgeContent::Example(_) => "example",
            KnowledgeContent::Meta(_) => "meta",
        }
    }

    /// True if this variant should surface as a (kind) tag in narration
    /// prose. The bare-default variants (`Raw`, `Concept`, `Meta`) are
    /// excluded so the common case stays untagged.
    pub fn is_narration_tagged(&self) -> bool {
        !matches!(
            self,
            KnowledgeContent::Raw(_) | KnowledgeContent::Concept(_) | KnowledgeContent::Meta(_)
        )
    }

    /// Build a `KnowledgeContent` from a kind tag string and payload.
    /// Accepts the canonical lowercase tag ("definition") plus a small
    /// set of common spelling/case variants. Returns `None` for an
    /// unknown kind so callers can surface a 4xx-style error.
    pub fn from_kind_str(kind: &str, payload: String) -> Option<Self> {
        let norm: String = kind
            .chars()
            .filter(|c| c.is_alphanumeric())
            .collect::<String>()
            .to_lowercase();
        match norm.as_str() {
            "fact" | "facts" => Some(KnowledgeContent::Fact(payload)),
            "concept" | "concepts" => Some(KnowledgeContent::Concept(payload)),
            "procedure" | "proc" | "howto" => Some(KnowledgeContent::Procedure(payload)),
            "episode" | "episodic" => Some(KnowledgeContent::Episode(payload)),
            "preference" | "pref" => Some(KnowledgeContent::Preference(payload)),
            "raw" | "untyped" => Some(KnowledgeContent::Raw(payload)),
            "code" | "snippet" | "source" => Some(KnowledgeContent::Code(payload)),
            "image" | "img" | "visual" => Some(KnowledgeContent::Image(payload)),
            "definition" | "def" | "defined" | "definedas" => {
                Some(KnowledgeContent::Definition(payload))
            }
            "mechanism" | "mech" | "worksby" | "howworks" => {
                Some(KnowledgeContent::Mechanism(payload))
            }
            "composition" | "composed" | "composedof" | "madeof" | "consistsof" => {
                Some(KnowledgeContent::Composition(payload))
            }
            "purpose" | "isfor" | "usedfor" | "forwhat" => {
                Some(KnowledgeContent::Purpose(payload))
            }
            "contrast" | "unlike" | "differs" | "differsfrom" | "contrastedwith" => {
                Some(KnowledgeContent::Contrast(payload))
            }
            "causal" | "cause" | "leadsto" | "resultsin" => {
                Some(KnowledgeContent::Causal(payload))
            }
            "example" | "eg" | "egexample" | "suchas" | "forexample" => {
                Some(KnowledgeContent::Example(payload))
            }
            "meta" | "goal" | "query" => Some(KnowledgeContent::Meta(payload)),
            _ => None,
        }
    }

    /// Convenience: copy a payload, type it as the named kind, return
    /// the original (Concept) shape if the kind is unknown so callers
    /// have a single safe call site.
    pub fn from_kind_or_concept(kind: Option<&str>, payload: String) -> KnowledgeContent {
        match kind {
            Some(k) => KnowledgeContent::from_kind_str(k, payload.clone())
                .unwrap_or(KnowledgeContent::Concept(payload)),
            None => KnowledgeContent::Concept(payload),
        }
    }
}

/// Q.1 — relation-string -> KU kind tag inference. Returns the tag
/// name (matching `KnowledgeContent::from_kind_str`) when the
/// relation alias carries semantic intent the typed-KU layer should
/// honor; `None` when the relation is generic (RelatedTo) and the
/// default Concept variant is appropriate. BR.3 widened the table to
/// cover all 23 smart_extractor relation aliases (IsA -> definition,
/// PartOf/ComposedOf -> composition, UsedFor -> purpose, every
/// spatial/event/numeric/notability attribute -> fact).
pub fn infer_kind_from_relation(relation: &str) -> Option<&'static str> {
    let norm: String = relation
        .chars()
        .filter(|c| c.is_alphanumeric())
        .collect::<String>()
        .to_lowercase();
    match norm.as_str() {
        // Pre-existing semantic aliases (Q.1 originals).
        "definedas" | "means" | "definition" | "isdefinedas" => Some("definition"),
        "worksby" | "operatesvia" | "functionsthrough" | "mechanism" => Some("mechanism"),
        "isfor" | "purposeof" | "purpose" => Some("purpose"),
        "unlike" | "differsfrom" | "contrast" | "incontrastto" => Some("contrast"),
        "leadsto" | "resultsin" | "causal" => Some("causal"),
        "exampleof" | "suchas" | "forexample" | "eg" | "example" => Some("example"),
        // BR.3: smart_extractor relation aliases (23 strings, L1-L5).
        // RelatedTo intentionally absent -- generic association, falls back
        // to Concept (the default) so the L1/L4 high-volume layers don't
        // skew the kind histogram toward a single misleading variant.
        "isa" => Some("definition"),
        "partof" | "composedof" | "madeof" | "consistsof" | "composition" => Some("composition"),
        "usedfor" => Some("purpose"),
        // Spatial / event / origin / numeric / notability attributes all
        // surface as Fact -- they assert a factual property of the subject
        // anchored in the corpus.
        "hasaspect" | "hasproperty" | "knownfor" => Some("fact"),
        "locatedin" | "foundedby" | "foundedin"
        | "discoveredby" | "namedafter" | "writtenby" => Some("fact"),
        "bornin" | "diedin" | "bornon" | "diedon" => Some("fact"),
        "measuredin" | "haspopulation" | "hasaltitude" | "hasarea"
        | "hasofficiallanguage" | "hascapital" => Some("fact"),
        _ => None,
    }
}

/// Q.1 — text-shape -> KU kind tag inference. Returns `"code"` when
/// the payload looks like a code snippet (fenced block or recognizable
/// language signature). Other text patterns (e.g. "is composed of")
/// are routed via `infer_kind_from_relation` instead — the relation
/// alias is the cleaner signal where it exists.
pub fn infer_kind_from_text(text: &str) -> Option<&'static str> {
    if text.contains("```") {
        return Some("code");
    }
    let lower = text.to_lowercase();
    let code_signals = [
        "def ",
        "function ",
        "class ",
        "import ",
        "#include",
        "fn ",
        "public static",
        "=> {",
    ];
    for sig in code_signals.iter() {
        if lower.contains(sig) {
            return Some("code");
        }
    }
    None
}

/// Q.1 — combined inference. Relation alias takes priority; text shape
/// fills in the gap (currently only fires for code snippets).
pub fn infer_kind(
    relation: &str,
    subject_text: &str,
    object_text: &str,
) -> Option<&'static str> {
    if let Some(k) = infer_kind_from_relation(relation) {
        return Some(k);
    }
    if let Some(k) = infer_kind_from_text(subject_text) {
        return Some(k);
    }
    if let Some(k) = infer_kind_from_text(object_text) {
        return Some(k);
    }
    None
}

/// A typed knowledge unit — the semantic payload of a node.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
#[archive(check_bytes)]
pub struct KnowledgeUnit {
    /// Typed content.
    pub content: KnowledgeContent,
    /// Confidence level of this knowledge.
    pub confidence: Confidence,
    /// Timestamp when this knowledge was first recorded (epoch millis).
    pub created_at: i64,
    /// Timestamp of the last update (epoch millis).
    pub updated_at: i64,
}

/// A node in the Lu 4D knowledge graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
#[archive(check_bytes)]
pub struct KnowledgeNode {
    /// Unique cryptographic identity.
    pub seed: Seed,
    /// Human-readable content string (summary / label).
    pub content: String,
    /// Structured knowledge unit.
    pub knowledge: KnowledgeUnit,
    /// 384-dimensional embedding vector for semantic similarity.
    pub embedding: Vec<f32>,
    /// Optional parent seeds for nodes derived from another node
    /// (Thinking Layer sub-goal steps point at their goal node).
    /// `#[serde(default)]` so v1 brain files written before this field
    /// existed load with an empty Vec. The previous
    /// `skip_serializing_if = "Vec::is_empty"` was dropped in Storage
    /// v2 Phase 1: the v2 binary path uses bincode's positional layout
    /// where field omission corrupts every following field.
    #[serde(default)]
    pub parent_seeds: Vec<Seed>,
    /// Cached fraction of this node's outgoing bonds whose `BondType`
    /// is something other than `RelatedTo`. Used by the narration
    /// HubSelector / Walker to favour richly-typed neighbours over
    /// bag-of-association ones (see Port #18b). Recomputed at brain
    /// load (`AssociativeNetwork::from_snapshot`) and at every
    /// consolidation pass (alongside `degree_p90`); never persisted
    /// state in the sense of "authoritative on disk" — it's pure
    /// derived data, but it lives in the format so reload doesn't
    /// pay the recompute cost on a 105K-node brain. `#[serde(default)]`
    /// so v1 brain files load with 0.0 and the load-time recompute
    /// fills it in before any walker runs.
    #[serde(default)]
    pub typed_ratio: f32,
}

impl KnowledgeNode {
    /// View-style accessor: human-readable content string.
    /// Mirrors `NodeView::content`. Slice 2b adds these so call sites
    /// can use the same method surface whether they hold a heap node
    /// or a `NodeView`.
    pub fn content(&self) -> &str { self.content.as_str() }
    /// View-style accessor: structured knowledge unit.
    pub fn knowledge(&self) -> &KnowledgeUnit { &self.knowledge }
    /// View-style accessor: confidence on this node.
    pub fn confidence(&self) -> Confidence { self.knowledge.confidence }
    /// View-style accessor: the typed knowledge content.
    pub fn knowledge_content(&self) -> &KnowledgeContent { &self.knowledge.content }
    /// View-style accessor: 384-dim embedding.
    pub fn embedding(&self) -> &[f32] { self.embedding.as_slice() }
    /// View-style accessor: parent seeds (Thinking Layer step links).
    pub fn parent_seeds(&self) -> &[Seed] { self.parent_seeds.as_slice() }
    /// View-style accessor: cached typed-bond ratio.
    pub fn typed_ratio(&self) -> f32 { self.typed_ratio }
    /// View-style accessor: seed identity. (Field-access also works.)
    pub fn seed(&self) -> Seed { self.seed }

    /// Create a new knowledge node with a generated seed and zero embedding.
    pub fn new(content: String, knowledge_content: KnowledgeContent, confidence: Confidence, now: i64) -> Self {
        KnowledgeNode {
            seed: Seed::generate(),
            content,
            knowledge: KnowledgeUnit {
                content: knowledge_content,
                confidence,
                created_at: now,
                updated_at: now,
            },
            embedding: vec![0.0; EMBEDDING_DIM],
            parent_seeds: Vec::new(),
            typed_ratio: 0.0,
        }
    }

    /// Create a node with a specific seed (for deterministic construction).
    pub fn with_seed(
        seed: Seed,
        content: String,
        knowledge_content: KnowledgeContent,
        confidence: Confidence,
        now: i64,
    ) -> Self {
        KnowledgeNode {
            seed,
            content,
            knowledge: KnowledgeUnit {
                content: knowledge_content,
                confidence,
                created_at: now,
                updated_at: now,
            },
            embedding: vec![0.0; EMBEDDING_DIM],
            parent_seeds: Vec::new(),
            typed_ratio: 0.0,
        }
    }

    /// Set the embedding vector. Panics if the length isn't EMBEDDING_DIM.
    pub fn set_embedding(&mut self, embedding: Vec<f32>) {
        assert_eq!(
            embedding.len(),
            EMBEDDING_DIM,
            "Embedding must be {EMBEDDING_DIM}-dimensional"
        );
        self.embedding = embedding;
    }

    /// Cosine similarity with another node's embedding.
    pub fn cosine_similarity(&self, other: &KnowledgeNode) -> f32 {
        let dot: f32 = self
            .embedding
            .iter()
            .zip(other.embedding.iter())
            .map(|(a, b)| a * b)
            .sum();

        let mag_a: f32 = self.embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        let mag_b: f32 = other.embedding.iter().map(|x| x * x).sum::<f32>().sqrt();

        if mag_a < 1e-10 || mag_b < 1e-10 {
            return 0.0;
        }

        dot / (mag_a * mag_b)
    }
}

// ---------------------------------------------------------------------
// OP-BRAIN-V4 Phase 10 -- KnowledgeNode on-disk serialization helpers.
//
// Used by the WAL (record-type discriminator byte 1), the SST per-
// block node table (SST v3 format), and the MemTable nodes map.
// Bincode is the same codec the existing TemporalKeyframe round-trip
// uses in sst::encode_keyframe, so the format-stability story is
// identical: a future on-disk migration is a fresh SST-format bump,
// never a silent layout change.
// ---------------------------------------------------------------------

/// Serialize a KnowledgeNode for on-disk persistence (WAL node record,
/// SST per-block node table, MemTable spill).
pub fn serialize_knowledge_node(node: &KnowledgeNode) -> Result<Vec<u8>, String> {
    bincode::serialize(node).map_err(|e| format!("serialize KnowledgeNode: {}", e))
}

/// Inverse of serialize_knowledge_node. Returns None on any decode
/// failure -- the caller logs and treats the node as missing rather
/// than poisoning the read with a partial decode.
pub fn deserialize_knowledge_node(bytes: &[u8]) -> Option<KnowledgeNode> {
    bincode::deserialize(bytes).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_confidence_values() {
        assert_eq!(Confidence::Terminal.value(), 1.0);
        assert_eq!(Confidence::Verified.value(), 0.9);
        assert_eq!(Confidence::Contested.value(), 0.2);
    }

    #[test]
    fn test_node_creation() {
        let node = KnowledgeNode::new(
            "Rust is a systems language".into(),
            KnowledgeContent::Fact("Rust is a systems programming language".into()),
            Confidence::Verified,
            1000,
        );
        assert_eq!(node.embedding.len(), EMBEDDING_DIM);
        assert_eq!(node.knowledge.confidence.value(), 0.9);
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let mut a = KnowledgeNode::new("A".into(), KnowledgeContent::Raw("A".into()), Confidence::Verified, 0);
        let mut b = KnowledgeNode::new("B".into(), KnowledgeContent::Raw("B".into()), Confidence::Verified, 0);
        let emb = vec![1.0; EMBEDDING_DIM];
        a.set_embedding(emb.clone());
        b.set_embedding(emb);
        let sim = a.cosine_similarity(&b);
        assert!((sim - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_cosine_similarity_zero_embeddings() {
        let a = KnowledgeNode::new("A".into(), KnowledgeContent::Raw("A".into()), Confidence::Verified, 0);
        let b = KnowledgeNode::new("B".into(), KnowledgeContent::Raw("B".into()), Confidence::Verified, 0);
        assert_eq!(a.cosine_similarity(&b), 0.0);
    }

    // ===== Q.1 typed-KU tests =====================================
    #[test]
    fn q1_kind_str_round_trip_all_variants() {
        let cases = [
            ("fact", KnowledgeContent::Fact("x".into())),
            ("concept", KnowledgeContent::Concept("x".into())),
            ("procedure", KnowledgeContent::Procedure("x".into())),
            ("episode", KnowledgeContent::Episode("x".into())),
            ("preference", KnowledgeContent::Preference("x".into())),
            ("raw", KnowledgeContent::Raw("x".into())),
            ("code", KnowledgeContent::Code("x".into())),
            ("image", KnowledgeContent::Image("x".into())),
            ("definition", KnowledgeContent::Definition("x".into())),
            ("mechanism", KnowledgeContent::Mechanism("x".into())),
            ("composition", KnowledgeContent::Composition("x".into())),
            ("purpose", KnowledgeContent::Purpose("x".into())),
            ("contrast", KnowledgeContent::Contrast("x".into())),
            ("causal", KnowledgeContent::Causal("x".into())),
            ("example", KnowledgeContent::Example("x".into())),
            ("meta", KnowledgeContent::Meta("x".into())),
        ];
        for (tag, kc) in cases.iter() {
            assert_eq!(kc.kind_str(), *tag, "kind_str mismatch for {:?}", kc);
            let rebuilt = KnowledgeContent::from_kind_str(tag, "x".into()).unwrap();
            assert_eq!(rebuilt.kind_str(), *tag);
            assert_eq!(rebuilt.text(), "x");
        }
    }

    #[test]
    fn q1_from_kind_str_case_insensitive_and_aliases() {
        for s in ["Definition", "DEFINITION", "def", "defined", "definedAs"] {
            let kc = KnowledgeContent::from_kind_str(s, "x".into()).unwrap();
            assert!(matches!(kc, KnowledgeContent::Definition(_)), "failed: {}", s);
        }
        for s in ["Mechanism", "mech", "WorksBy", "howWorks"] {
            let kc = KnowledgeContent::from_kind_str(s, "x".into()).unwrap();
            assert!(matches!(kc, KnowledgeContent::Mechanism(_)), "failed: {}", s);
        }
        for s in ["composition", "composedOf", "madeOf", "consistsOf"] {
            let kc = KnowledgeContent::from_kind_str(s, "x".into()).unwrap();
            assert!(matches!(kc, KnowledgeContent::Composition(_)), "failed: {}", s);
        }
        assert!(KnowledgeContent::from_kind_str("nonsense_variant", "x".into()).is_none());
        assert!(KnowledgeContent::from_kind_str("", "x".into()).is_none());
    }

    #[test]
    fn q1_from_kind_or_concept_falls_back_safely() {
        let kc = KnowledgeContent::from_kind_or_concept(Some("definition"), "democracy".into());
        assert!(matches!(kc, KnowledgeContent::Definition(_)));
        let kc2 = KnowledgeContent::from_kind_or_concept(None, "x".into());
        assert!(matches!(kc2, KnowledgeContent::Concept(_)));
        // Unknown kind falls back to Concept rather than panicking.
        let kc3 = KnowledgeContent::from_kind_or_concept(Some("nope"), "x".into());
        assert!(matches!(kc3, KnowledgeContent::Concept(_)));
    }

    #[test]
    fn q1_is_narration_tagged_excludes_common_case() {
        assert!(!KnowledgeContent::Concept("x".into()).is_narration_tagged());
        assert!(!KnowledgeContent::Raw("x".into()).is_narration_tagged());
        assert!(!KnowledgeContent::Meta("x".into()).is_narration_tagged());
        assert!(KnowledgeContent::Definition("x".into()).is_narration_tagged());
        assert!(KnowledgeContent::Mechanism("x".into()).is_narration_tagged());
        assert!(KnowledgeContent::Composition("x".into()).is_narration_tagged());
        assert!(KnowledgeContent::Purpose("x".into()).is_narration_tagged());
        assert!(KnowledgeContent::Contrast("x".into()).is_narration_tagged());
        assert!(KnowledgeContent::Causal("x".into()).is_narration_tagged());
        assert!(KnowledgeContent::Example("x".into()).is_narration_tagged());
        assert!(KnowledgeContent::Code("x".into()).is_narration_tagged());
    }

    #[test]
    fn q1_infer_kind_from_relation_canonical() {
        assert_eq!(infer_kind_from_relation("DefinedAs"), Some("definition"));
        assert_eq!(infer_kind_from_relation("defined_as"), Some("definition"));
        assert_eq!(infer_kind_from_relation("means"), Some("definition"));
        assert_eq!(infer_kind_from_relation("WorksBy"), Some("mechanism"));
        assert_eq!(infer_kind_from_relation("OperatesVia"), Some("mechanism"));
        assert_eq!(infer_kind_from_relation("ComposedOf"), Some("composition"));
        assert_eq!(infer_kind_from_relation("made of"), Some("composition"));
        assert_eq!(infer_kind_from_relation("IsFor"), Some("purpose"));
        assert_eq!(infer_kind_from_relation("Unlike"), Some("contrast"));
        assert_eq!(infer_kind_from_relation("LeadsTo"), Some("causal"));
        assert_eq!(infer_kind_from_relation("SuchAs"), Some("example"));
        // RelatedTo stays generic -- it dominates L1/L4 volume and routing
        // it onto a single kind would skew the whole histogram. Falls back
        // to Concept.
        assert_eq!(infer_kind_from_relation("RelatedTo"), None);
    }

    /// BR.3 -- smart_extractor relation aliases must all resolve to a kind
    /// (with the sole exception of RelatedTo above). 22 of the 23 emitted
    /// relation strings carry semantic intent; this test pins the mapping
    /// so the kind histogram in preflight A.3 stays predictable.
    #[test]
    fn br3_extractor_relations_map_to_kind() {
        // Identity / category.
        assert_eq!(infer_kind_from_relation("IsA"), Some("definition"));
        // Composition (PartOf is wider in semantics than "composedof" but
        // we collapse them onto the composition kind for histogram purposes).
        assert_eq!(infer_kind_from_relation("PartOf"), Some("composition"));
        assert_eq!(infer_kind_from_relation("ComposedOf"), Some("composition"));
        // Purpose.
        assert_eq!(infer_kind_from_relation("UsedFor"), Some("purpose"));
        // Notability / aspect / property -> fact.
        assert_eq!(infer_kind_from_relation("HasAspect"), Some("fact"));
        assert_eq!(infer_kind_from_relation("HasProperty"), Some("fact"));
        assert_eq!(infer_kind_from_relation("KnownFor"), Some("fact"));
        // Spatial / event facts.
        assert_eq!(infer_kind_from_relation("LocatedIn"), Some("fact"));
        assert_eq!(infer_kind_from_relation("FoundedIn"), Some("fact"));
        assert_eq!(infer_kind_from_relation("FoundedBy"), Some("fact"));
        assert_eq!(infer_kind_from_relation("DiscoveredBy"), Some("fact"));
        assert_eq!(infer_kind_from_relation("NamedAfter"), Some("fact"));
        assert_eq!(infer_kind_from_relation("WrittenBy"), Some("fact"));
        assert_eq!(infer_kind_from_relation("BornIn"), Some("fact"));
        assert_eq!(infer_kind_from_relation("DiedIn"), Some("fact"));
        assert_eq!(infer_kind_from_relation("BornOn"), Some("fact"));
        assert_eq!(infer_kind_from_relation("DiedOn"), Some("fact"));
        // Numeric / linguistic attributes.
        assert_eq!(infer_kind_from_relation("MeasuredIn"), Some("fact"));
        assert_eq!(infer_kind_from_relation("HasPopulation"), Some("fact"));
        assert_eq!(infer_kind_from_relation("HasAltitude"), Some("fact"));
        assert_eq!(infer_kind_from_relation("HasArea"), Some("fact"));
        assert_eq!(infer_kind_from_relation("HasOfficialLanguage"), Some("fact"));
        assert_eq!(infer_kind_from_relation("HasCapital"), Some("fact"));
    }

    #[test]
    fn q1_infer_kind_from_text_recognizes_code() {
        assert_eq!(infer_kind_from_text("```py\nprint(1)\n```"), Some("code"));
        assert_eq!(infer_kind_from_text("def foo(x):"), Some("code"));
        assert_eq!(infer_kind_from_text("class Bar:"), Some("code"));
        assert_eq!(infer_kind_from_text("import json"), Some("code"));
        assert_eq!(infer_kind_from_text("function f() {}"), Some("code"));
        assert_eq!(infer_kind_from_text("fn main()"), Some("code"));
        assert_eq!(infer_kind_from_text("#include <stdio.h>"), Some("code"));
        assert_eq!(infer_kind_from_text("hello world"), None);
        assert_eq!(infer_kind_from_text(""), None);
    }

    #[test]
    fn q1_infer_kind_combined_priority() {
        // Relation hint wins over text-shape (subject is generic).
        assert_eq!(infer_kind("DefinedAs", "x", "y"), Some("definition"));
        // Generic relation but code in subject -> code.
        assert_eq!(infer_kind("RelatedTo", "def main()", "y"), Some("code"));
        // Generic relation, code in object only -> code.
        assert_eq!(infer_kind("RelatedTo", "x", "```rust"), Some("code"));
        // Generic everything -> None.
        assert_eq!(infer_kind("RelatedTo", "x", "y"), None);
        assert_eq!(infer_kind("", "", ""), None);
    }
}
