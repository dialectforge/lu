//! Storage v2 Slice 2c — archive-side accessors and the mmap holder.
//!
//! Slice 2a moved the on-disk format to rkyv but eagerly deserialized
//! everything onto the heap on load. Slice 2b introduced abstract
//! views (`NodeView`, `BondView`) so callers stop reaching for
//! `&KnowledgeNode` directly. Slice 2c (this module) closes the loop:
//! the brain can keep an mmap'd v3 file alive and serve nodes/bonds
//! directly from the archived bytes — zero-copy reads, copy-on-write
//! mutations.
//!
//! Two responsibilities live here:
//!   1. Conversion helpers from rkyv `Archived*` types to the
//!      heap-shaped enums/structs that the rest of the codebase
//!      already speaks (`Confidence`, `BondType`, `KeyframeSource`,
//!      `KnowledgeContent`, `Seed`). All allocation-free for the
//!      `Copy` types; `KnowledgeContent` clones the inner string.
//!   2. A `NetworkArchive` owner that holds an `Arc<Mmap>` alive,
//!      validates the v3 network section through
//!      `rkyv::check_archived_root`, and exposes seed-keyed lookups
//!      backed by two HashMaps (`node_index`, `bond_index`) that index
//!      into the archived `Vec`s. The archived blob itself is referred
//!      to via a `NonNull<ArchivedNetworkSnapshot>` raw pointer so the
//!      type system doesn't trip on the self-referential Arc->Mmap->ref
//!      pattern. SAFETY is preserved by making the NonNull access only
//!      through `&self`, which can't outlive the NetworkArchive, which
//!      drops the `Arc<Mmap>` last.
//!
//! The archive-side `historical_weight_at` / `historical_weight_in_scope`
//! helpers mirror `TemporalBond`'s heap-side logic byte-for-byte. They
//! exist as free functions over `&ArchivedTemporalBond` because the
//! heap path's logic is private and works on `&self: &TemporalBond`;
//! duplicating ~80 lines of decay/scope arithmetic is preferable to
//! shoehorning a trait that bridges the two reps.

use std::collections::HashMap;
use std::path::Path;
use std::ptr::NonNull;
use std::sync::Arc;

use memmap2::Mmap;

use crate::bond::{
    ArchivedBondType, ArchivedKeyframeSource, ArchivedTemporalBond, BondType, KeyframeSource,
    TemporalBond, TemporalKeyframe,
};
use crate::brain::BrainError;
use crate::brain_io_v3::{extract_archived_network_blob_no_crc, extract_archived_scar_blob_no_crc};
use crate::network::{ArchivedNetworkSnapshot, NetworkSnapshot};
use crate::node::{
    ArchivedConfidence, ArchivedKnowledgeContent, ArchivedKnowledgeNode, ArchivedKnowledgeUnit,
    Confidence, KnowledgeContent, KnowledgeNode, KnowledgeUnit,
};
use crate::scar::{self, ScarAwareReading, ScarLog};
use crate::scope::TemporalScope;
use crate::seed::{ArchivedSeed, Seed};

// ---------- Trivially-Copy conversions (no allocation) ----------

#[inline]
pub fn seed_from_archived(a: &ArchivedSeed) -> Seed {
    // Seed is `pub struct Seed(pub [u8; 32])` and ArchivedSeed has the
    // identical tuple layout (rkyv 0.7's tuple-struct archived form).
    // Direct field access is allocation-free.
    Seed(a.0)
}

#[inline]
pub fn confidence_from_archived(c: &ArchivedConfidence) -> Confidence {
    match c {
        ArchivedConfidence::Terminal => Confidence::Terminal,
        ArchivedConfidence::Verified => Confidence::Verified,
        ArchivedConfidence::Unverified => Confidence::Unverified,
        ArchivedConfidence::Inferred => Confidence::Inferred,
        ArchivedConfidence::Contested => Confidence::Contested,
    }
}

#[inline]
pub fn bond_type_from_archived(b: &ArchivedBondType) -> BondType {
    match b {
        ArchivedBondType::IsA => BondType::IsA,
        ArchivedBondType::DefinedAs => BondType::DefinedAs,
        ArchivedBondType::WorksBy => BondType::WorksBy,
        ArchivedBondType::ComposedOf => BondType::ComposedOf,
        ArchivedBondType::PurposeOf => BondType::PurposeOf,
        ArchivedBondType::DiffersFrom => BondType::DiffersFrom,
        ArchivedBondType::LeadsTo => BondType::LeadsTo,
        ArchivedBondType::ExampleOf => BondType::ExampleOf,
        ArchivedBondType::HasProperty => BondType::HasProperty,
        ArchivedBondType::Causes => BondType::Causes,
        ArchivedBondType::PartOf => BondType::PartOf,
        ArchivedBondType::UsedFor => BondType::UsedFor,
        ArchivedBondType::RelatedTo => BondType::RelatedTo,
        ArchivedBondType::ContrastedWith => BondType::ContrastedWith,
        ArchivedBondType::DerivedFrom => BondType::DerivedFrom,
        ArchivedBondType::Requires => BondType::Requires,
        ArchivedBondType::Produces => BondType::Produces,
        ArchivedBondType::Opposes => BondType::Opposes,
        ArchivedBondType::BornIn => BondType::BornIn,
        ArchivedBondType::BornOn => BondType::BornOn,
        ArchivedBondType::DiedIn => BondType::DiedIn,
        ArchivedBondType::DiedOn => BondType::DiedOn,
        ArchivedBondType::DiscoveredBy => BondType::DiscoveredBy,
        ArchivedBondType::FoundedBy => BondType::FoundedBy,
        ArchivedBondType::FoundedIn => BondType::FoundedIn,
        ArchivedBondType::HasAltitude => BondType::HasAltitude,
        ArchivedBondType::HasAspect => BondType::HasAspect,
        ArchivedBondType::HasCapital => BondType::HasCapital,
        ArchivedBondType::HasOfficialLanguage => BondType::HasOfficialLanguage,
        ArchivedBondType::HasPopulation => BondType::HasPopulation,
        ArchivedBondType::KnownFor => BondType::KnownFor,
        ArchivedBondType::LocatedIn => BondType::LocatedIn,
        ArchivedBondType::MeasuredIn => BondType::MeasuredIn,
        ArchivedBondType::NamedAfter => BondType::NamedAfter,
        ArchivedBondType::WrittenBy => BondType::WrittenBy,
        ArchivedBondType::HasArea => BondType::HasArea,
        ArchivedBondType::HasMethod => BondType::HasMethod,
        ArchivedBondType::HasAttribute => BondType::HasAttribute,
        ArchivedBondType::HasParameter => BondType::HasParameter,
        ArchivedBondType::Returns => BondType::Returns,
        ArchivedBondType::Inherits => BondType::Inherits,
        ArchivedBondType::Calls => BondType::Calls,
        ArchivedBondType::HasType => BondType::HasType,
        ArchivedBondType::Implements => BondType::Implements,
        ArchivedBondType::HasCurrency => BondType::HasCurrency,
        ArchivedBondType::AuthoredBy => BondType::AuthoredBy,
    }
}

/// Get the inner text of a `KnowledgeContent` payload without
/// allocating. Mirrors `KnowledgeContent::text` on the heap side.
#[inline]
pub fn knowledge_content_text<'a>(kc: &'a ArchivedKnowledgeContent) -> &'a str {
    match kc {
        ArchivedKnowledgeContent::Fact(s) => s.as_str(),
        ArchivedKnowledgeContent::Concept(s) => s.as_str(),
        ArchivedKnowledgeContent::Procedure(s) => s.as_str(),
        ArchivedKnowledgeContent::Episode(s) => s.as_str(),
        ArchivedKnowledgeContent::Preference(s) => s.as_str(),
        ArchivedKnowledgeContent::Raw(s) => s.as_str(),
        ArchivedKnowledgeContent::Code(s) => s.as_str(),
        ArchivedKnowledgeContent::Image(s) => s.as_str(),
        ArchivedKnowledgeContent::Definition(s) => s.as_str(),
        ArchivedKnowledgeContent::Mechanism(s) => s.as_str(),
        ArchivedKnowledgeContent::Composition(s) => s.as_str(),
        ArchivedKnowledgeContent::Purpose(s) => s.as_str(),
        ArchivedKnowledgeContent::Contrast(s) => s.as_str(),
        ArchivedKnowledgeContent::Causal(s) => s.as_str(),
        ArchivedKnowledgeContent::Example(s) => s.as_str(),
        ArchivedKnowledgeContent::Meta(s) => s.as_str(),
    }
}

/// Q.1 — archive-side `kind_str` mirror of `KnowledgeContent::kind_str`.
/// Lets the narration / MCP layers read the variant tag from a v3
/// archive-backed node without going through CoW promotion.
#[inline]
pub fn archived_knowledge_kind_str(kc: &ArchivedKnowledgeContent) -> &'static str {
    match kc {
        ArchivedKnowledgeContent::Fact(_) => "fact",
        ArchivedKnowledgeContent::Concept(_) => "concept",
        ArchivedKnowledgeContent::Procedure(_) => "procedure",
        ArchivedKnowledgeContent::Episode(_) => "episode",
        ArchivedKnowledgeContent::Preference(_) => "preference",
        ArchivedKnowledgeContent::Raw(_) => "raw",
        ArchivedKnowledgeContent::Code(_) => "code",
        ArchivedKnowledgeContent::Image(_) => "image",
        ArchivedKnowledgeContent::Definition(_) => "definition",
        ArchivedKnowledgeContent::Mechanism(_) => "mechanism",
        ArchivedKnowledgeContent::Composition(_) => "composition",
        ArchivedKnowledgeContent::Purpose(_) => "purpose",
        ArchivedKnowledgeContent::Contrast(_) => "contrast",
        ArchivedKnowledgeContent::Causal(_) => "causal",
        ArchivedKnowledgeContent::Example(_) => "example",
        ArchivedKnowledgeContent::Meta(_) => "meta",
    }
}

/// Materialize a heap `KnowledgeContent` from its archived form. Used
/// on the CoW promotion path in `AssociativeNetwork::get_node_mut` and
/// when serializing a fresh v3 file in `to_snapshot`.
pub fn knowledge_content_from_archived(kc: &ArchivedKnowledgeContent) -> KnowledgeContent {
    match kc {
        ArchivedKnowledgeContent::Fact(s) => KnowledgeContent::Fact(s.as_str().to_string()),
        ArchivedKnowledgeContent::Concept(s) => KnowledgeContent::Concept(s.as_str().to_string()),
        ArchivedKnowledgeContent::Procedure(s) => {
            KnowledgeContent::Procedure(s.as_str().to_string())
        }
        ArchivedKnowledgeContent::Episode(s) => KnowledgeContent::Episode(s.as_str().to_string()),
        ArchivedKnowledgeContent::Preference(s) => {
            KnowledgeContent::Preference(s.as_str().to_string())
        }
        ArchivedKnowledgeContent::Raw(s) => KnowledgeContent::Raw(s.as_str().to_string()),
        ArchivedKnowledgeContent::Code(s) => KnowledgeContent::Code(s.as_str().to_string()),
        ArchivedKnowledgeContent::Image(s) => KnowledgeContent::Image(s.as_str().to_string()),
        ArchivedKnowledgeContent::Definition(s) => {
            KnowledgeContent::Definition(s.as_str().to_string())
        }
        ArchivedKnowledgeContent::Mechanism(s) => {
            KnowledgeContent::Mechanism(s.as_str().to_string())
        }
        ArchivedKnowledgeContent::Composition(s) => {
            KnowledgeContent::Composition(s.as_str().to_string())
        }
        ArchivedKnowledgeContent::Purpose(s) => KnowledgeContent::Purpose(s.as_str().to_string()),
        ArchivedKnowledgeContent::Contrast(s) => KnowledgeContent::Contrast(s.as_str().to_string()),
        ArchivedKnowledgeContent::Causal(s) => KnowledgeContent::Causal(s.as_str().to_string()),
        ArchivedKnowledgeContent::Example(s) => KnowledgeContent::Example(s.as_str().to_string()),
        ArchivedKnowledgeContent::Meta(s) => KnowledgeContent::Meta(s.as_str().to_string()),
    }
}

/// Materialize a heap `KnowledgeUnit` from its archived form.
pub fn knowledge_unit_from_archived(ku: &ArchivedKnowledgeUnit) -> KnowledgeUnit {
    KnowledgeUnit {
        content: knowledge_content_from_archived(&ku.content),
        confidence: confidence_from_archived(&ku.confidence),
        created_at: ku.created_at.into(),
        updated_at: ku.updated_at.into(),
    }
}

/// Materialize a heap `KeyframeSource` from its archived form. The
/// `usize`-typed `step` field on `Hypothesis`/`ConfirmedHypothesis`
/// archives as `ArchivedUsize` (a wrapper that derefs to the underlying
/// integer); we cast back through the wrapper's `Into<u64>` impl and
/// then truncate to `usize`. Sound for the platforms Lu cares about
/// (x86_64/aarch64) where `usize == u64`.
pub fn keyframe_source_from_archived(s: &ArchivedKeyframeSource) -> KeyframeSource {
    match s {
        ArchivedKeyframeSource::Baseline => KeyframeSource::Baseline,
        ArchivedKeyframeSource::Session(id) => KeyframeSource::Session((*id).into()),
        ArchivedKeyframeSource::Hebbian => KeyframeSource::Hebbian,
        ArchivedKeyframeSource::Manual => KeyframeSource::Manual,
        ArchivedKeyframeSource::Synthesis { from_path } => KeyframeSource::Synthesis {
            from_path: [
                seed_from_archived(&from_path[0]),
                seed_from_archived(&from_path[1]),
                seed_from_archived(&from_path[2]),
            ],
        },
        ArchivedKeyframeSource::Hypothesis {
            goal,
            step,
            session,
        } => KeyframeSource::Hypothesis {
            goal: seed_from_archived(goal),
            step: u64::from(*step) as usize,
            session: session.as_ref().map(|s| (*s).into()),
        },
        ArchivedKeyframeSource::ConfirmedHypothesis {
            goal,
            step,
            session,
        } => KeyframeSource::ConfirmedHypothesis {
            goal: seed_from_archived(goal),
            step: u64::from(*step) as usize,
            session: session.as_ref().map(|s| (*s).into()),
        },
        ArchivedKeyframeSource::Variant { id, goal, strategy } => KeyframeSource::Variant {
            id: (*id).into(),
            goal: seed_from_archived(goal),
            strategy: *strategy,
        },
        ArchivedKeyframeSource::PromotedFromVariant {
            variant_id,
            original_strategy,
        } => KeyframeSource::PromotedFromVariant {
            variant_id: (*variant_id).into(),
            original_strategy: *original_strategy,
        },
        ArchivedKeyframeSource::Agent {
            role,
            multi_think_id,
            step,
        } => KeyframeSource::Agent {
            role: *role,
            multi_think_id: (*multi_think_id).into(),
            step: (*step).into(),
        },
        ArchivedKeyframeSource::ConcludedFrom { rule, cited_edges } => {
            KeyframeSource::ConcludedFrom {
                rule: derivation_rule_from_archived(rule),
                cited_edges: cited_edges.iter().map(premise_edge_from_archived).collect(),
            }
        }
    }
}

/// Convert an archived `DerivationRule` (fieldless enum) to its heap form.
#[inline]
pub fn derivation_rule_from_archived(
    r: &crate::bond::ArchivedDerivationRule,
) -> crate::bond::DerivationRule {
    use crate::bond::{ArchivedDerivationRule as A, DerivationRule as D};
    match r {
        A::Transitivity => D::Transitivity,
        A::PropertyInheritance => D::PropertyInheritance,
        A::CausalChain => D::CausalChain,
        A::Meronymy => D::Meronymy,
        A::Corroboration => D::Corroboration,
        A::Other => D::Other,
    }
}

/// Convert an archived `PremiseEdge` (lineage premise) to its heap form.
#[inline]
pub fn premise_edge_from_archived(e: &crate::bond::ArchivedPremiseEdge) -> crate::bond::PremiseEdge {
    crate::bond::PremiseEdge {
        subject: seed_from_archived(&e.subject),
        predicate: bond_type_from_archived(&e.predicate),
        object: seed_from_archived(&e.object),
    }
}

/// Materialize a heap `TemporalKeyframe` from its archived form.
pub fn temporal_keyframe_from_archived(
    kf: &crate::bond::ArchivedTemporalKeyframe,
) -> TemporalKeyframe {
    TemporalKeyframe {
        timestamp: kf.timestamp.into(),
        weight: kf.weight.into(),
        source: keyframe_source_from_archived(&kf.source),
    }
}

/// Materialize a heap `TemporalBond` from its archived form. Used on
/// the CoW promotion path inside `AssociativeNetwork::get_bond_mut`.
pub fn temporal_bond_from_archived(b: &ArchivedTemporalBond) -> TemporalBond {
    let keyframes: Vec<TemporalKeyframe> = b
        .keyframes
        .iter()
        .map(temporal_keyframe_from_archived)
        .collect();
    TemporalBond {
        from: seed_from_archived(&b.from),
        to: seed_from_archived(&b.to),
        bond_type: bond_type_from_archived(&b.bond_type),
        keyframes,
        half_life_ms: b.half_life_ms.into(),
    }
}

/// Materialize a heap `KnowledgeNode` from its archived form. Used on
/// the CoW promotion path inside `AssociativeNetwork::get_node_mut`,
/// and when iterating archived nodes for a fresh-v3 save.
pub fn knowledge_node_from_archived(n: &ArchivedKnowledgeNode) -> KnowledgeNode {
    let parent_seeds: Vec<Seed> = n.parent_seeds.iter().map(seed_from_archived).collect();
    let embedding: Vec<f32> = n.embedding.iter().map(|x| (*x).into()).collect();
    KnowledgeNode {
        seed: seed_from_archived(&n.seed),
        content: n.content.as_str().to_string(),
        knowledge: knowledge_unit_from_archived(&n.knowledge),
        embedding,
        parent_seeds,
        typed_ratio: n.typed_ratio.into(),
    }
}

// ---------- Archive-side weight computation ----------

/// Archive-side mirror of `TemporalBond::historical_weight_at`.
///
/// The behavior must match the heap path byte-for-byte:
///   1. Skip Variant / Agent keyframes (scope-isolated).
///   2. For `t == i64::MAX` (Current scope), return latest visible
///      keyframe weight * bond-type base weight.
///   3. Otherwise find the most recent visible keyframe `<= t`,
///      apply exponential decay over the elapsed window, return
///      decayed weight * bond-type base weight.
pub fn historical_weight_at_archived(b: &ArchivedTemporalBond, t: i64) -> f32 {
    if b.keyframes.is_empty() {
        return 0.0;
    }

    let bt = bond_type_from_archived(&b.bond_type);
    let half_life_ms: f64 = b.half_life_ms.into();

    // Latest visible keyframe at or before `t`. We walk in timestamp
    // order (the archive preserves the heap path's insertion-sorted
    // invariant) so we can short-circuit once we step past `t`.
    let mut last: Option<(i64, f32)> = None;
    for kf in b.keyframes.iter() {
        // Variant / Agent keyframes are scope-isolated.
        match &kf.source {
            ArchivedKeyframeSource::Variant { .. } | ArchivedKeyframeSource::Agent { .. } => {
                continue;
            }
            _ => {}
        }
        let kf_ts: i64 = kf.timestamp.into();
        if t != i64::MAX && kf_ts > t {
            break;
        }
        let kf_w: f32 = kf.weight.into();
        last = Some((kf_ts, kf_w));
    }

    let (ts, w) = match last {
        Some(x) => x,
        None => return 0.0,
    };

    if t == i64::MAX {
        return w * bt.base_weight();
    }
    let elapsed_ms = (t - ts) as f64;
    let decay = (-elapsed_ms / half_life_ms).exp2();
    let decayed = (w as f64) * decay;
    (decayed as f32) * bt.base_weight()
}

/// Archive-side mirror of `TemporalBond::historical_weight_in_scope`.
pub fn historical_weight_in_scope_archived(
    b: &ArchivedTemporalBond,
    scope: &TemporalScope,
) -> f32 {
    let bt = bond_type_from_archived(&b.bond_type);
    match scope {
        TemporalScope::Session { id, .. } => {
            // Latest Baseline OR Session(id) keyframe.
            let mut latest: Option<f32> = None;
            for kf in b.keyframes.iter() {
                let matches = match &kf.source {
                    ArchivedKeyframeSource::Baseline => true,
                    ArchivedKeyframeSource::Session(sid) => u64::from(*sid) == *id,
                    _ => false,
                };
                if matches {
                    latest = Some(kf.weight.into());
                }
            }
            if let Some(w) = latest {
                w * bt.base_weight()
            } else {
                historical_weight_at_archived(b, scope.timestamp())
            }
        }
        TemporalScope::Variant { id, parent } => {
            let mut latest: Option<f32> = None;
            for kf in b.keyframes.iter() {
                if let ArchivedKeyframeSource::Variant { id: vid, .. } = &kf.source {
                    if u64::from(*vid) == *id {
                        latest = Some(kf.weight.into());
                    }
                }
            }
            if let Some(w) = latest {
                w * bt.base_weight()
            } else {
                historical_weight_in_scope_archived(b, parent)
            }
        }
        TemporalScope::Agent {
            role,
            multi_think_id,
            parent,
        } => {
            let role_u8 = role.as_u8();
            let mut latest: Option<f32> = None;
            for kf in b.keyframes.iter() {
                if let ArchivedKeyframeSource::Agent {
                    role: r,
                    multi_think_id: m,
                    ..
                } = &kf.source
                {
                    if *r == role_u8 && u64::from(*m) == *multi_think_id {
                        latest = Some(kf.weight.into());
                    }
                }
            }
            if let Some(w) = latest {
                w * bt.base_weight()
            } else {
                historical_weight_in_scope_archived(b, parent)
            }
        }
        _ => historical_weight_at_archived(b, scope.timestamp()),
    }
}

/// Archive-side mirror of `TemporalBond::weight_at` (scar-aware).
pub fn weight_at_archived(
    b: &ArchivedTemporalBond,
    t: i64,
    scar_log: &ScarLog,
) -> ScarAwareReading {
    let from = seed_from_archived(&b.from);
    let to = seed_from_archived(&b.to);
    let historical = historical_weight_at_archived(b, t);
    let active: Vec<crate::scar::Scar> = scar_log
        .active_scars(from, to, scar::now_ms())
        .into_iter()
        .cloned()
        .collect();
    let effective = if active.is_empty() { historical } else { 0.0 };
    ScarAwareReading {
        historical_weight: historical,
        effective_weight: effective,
        active_scars: active,
    }
}

/// Archive-side mirror of `TemporalBond::weight_in_scope` (scar-aware).
pub fn weight_in_scope_archived(
    b: &ArchivedTemporalBond,
    scope: &TemporalScope,
    scar_log: &ScarLog,
) -> ScarAwareReading {
    let from = seed_from_archived(&b.from);
    let to = seed_from_archived(&b.to);
    let historical = historical_weight_in_scope_archived(b, scope);
    let active: Vec<crate::scar::Scar> = scar_log
        .active_scars(from, to, scar::now_ms())
        .into_iter()
        .cloned()
        .collect();
    let effective = if active.is_empty() { historical } else { 0.0 };
    ScarAwareReading {
        historical_weight: historical,
        effective_weight: effective,
        active_scars: active,
    }
}

/// Latest raw keyframe weight for an archived bond — no decay, no
/// bond-type multiplier. Mirrors `TemporalBond::latest_raw_weight`.
pub fn latest_raw_weight_archived(b: &ArchivedTemporalBond) -> f32 {
    b.keyframes
        .last()
        .map(|kf| kf.weight.into())
        .unwrap_or(0.0)
}

/// Distinct session count over an archived bond's keyframes. Mirrors
/// `TemporalBond::distinct_session_count`.
pub fn distinct_session_count_archived(b: &ArchivedTemporalBond) -> usize {
    let mut sessions = std::collections::HashSet::new();
    for kf in b.keyframes.iter() {
        if let ArchivedKeyframeSource::Session(id) = &kf.source {
            sessions.insert(u64::from(*id));
        }
    }
    sessions.len()
}

// ---------- NetworkArchive: the Mmap holder + index maps ----------

/// An mmap'd, validated, indexed view over a v3 brain file's network
/// section. Held alive for the lifetime of an `AssociativeNetwork`
/// running in zero-copy mode.
///
/// Phase 3 added a second backing: `Compressed`, which holds a
/// `CompressedNetwork` reader over a per-block-zstd-encoded section
/// (see `crate::compress`). Both backings share the same public API;
/// the caller never needs to know which one it is.
pub struct NetworkArchive {
    inner: NetworkArchiveInner,
}

enum NetworkArchiveInner {
    /// Uncompressed v3 (Phase 2.2 layout): rkyv-archived NetworkSnapshot
    /// directly mapped into the file. Lookups are pointer chases inside
    /// the mmap.
    Uncompressed {
        /// Pointer into the mmap'd bytes. SAFETY: backed by `_mmap`;
        /// the pointer remains valid for as long as `_mmap` is held.
        archived: NonNull<ArchivedNetworkSnapshot>,
        /// seed → index into `archived().nodes`.
        node_index: HashMap<Seed, u32>,
        /// (from, to) → index into `archived().bonds`.
        bond_index: HashMap<(Seed, Seed), u32>,
        /// Owned mmap; kept alive so `archived` stays valid.
        _mmap: Arc<Mmap>,
    },
    /// Per-block compressed (Phase 3 layout): each KnowledgeNode and
    /// TemporalBond is its own self-contained rkyv archive packed into
    /// zstd-compressed ~256 KB blocks. Lookups decompress one block on
    /// first touch.
    Compressed {
        compressed: crate::compress::CompressedNetwork,
    },
}

// SAFETY: `archived` is a `NonNull` into `_mmap`'s read-only mapped
// memory. Reading the bytes via shared `&self` is thread-safe because
// the mapping is read-only. `_mmap: Arc<Mmap>` is itself `Send + Sync`.
unsafe impl Send for NetworkArchive {}
unsafe impl Sync for NetworkArchive {}

/// OP-016: handle returned by `NetworkArchive::node_by_seed`. Holds
/// either a direct borrow into an mmap'd `ArchivedNetworkSnapshot`
/// (Uncompressed backing) or an `ArcArchive<KnowledgeNode>` that owns
/// an Arc clone of a decompressed block (Compressed backing). The
/// Arc keeps the block alive even if the LRU evicts the entry from
/// its map, so the `&ArchivedKnowledgeNode` returned by
/// `as_archived()` is valid for the lifetime of the handle.
#[derive(Clone)]
pub enum NodeArchive<'a> {
    Borrowed(&'a ArchivedKnowledgeNode),
    Owned(crate::compress::ArcArchive<crate::node::KnowledgeNode>),
}

impl<'a> NodeArchive<'a> {
    /// Borrow the underlying archived node. Lifetime ties to `&self`.
    pub fn as_archived(&self) -> &ArchivedKnowledgeNode {
        match self {
            NodeArchive::Borrowed(r) => r,
            NodeArchive::Owned(h) => h.archived(),
        }
    }
}

/// OP-016: handle returned by `NetworkArchive::bond_by_pair`. Mirror
/// of `NodeArchive` for bonds.
#[derive(Clone)]
pub enum BondArchive<'a> {
    Borrowed(&'a ArchivedTemporalBond),
    Owned(crate::compress::ArcArchive<crate::bond::TemporalBond>),
}

impl<'a> BondArchive<'a> {
    pub fn as_archived(&self) -> &ArchivedTemporalBond {
        match self {
            BondArchive::Borrowed(r) => r,
            BondArchive::Owned(h) => h.archived(),
        }
    }
}

impl NetworkArchive {
    /// Open a v3 file, mmap it, validate the network section, and
    /// build seed → index lookup tables. The returned `NetworkArchive`
    /// owns the mmap; dropping it unmaps the file.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, BrainError> {
        let f = std::fs::File::open(path.as_ref())?;
        // SAFETY: see brain_io_v3::read_v3_path — same justification.
        let mmap = unsafe { Mmap::map(&f) }
            .map_err(|e| BrainError::Format(format!("mmap failed: {}", e)))?;
        Self::from_mmap(Arc::new(mmap))
    }

    /// Construct from an already-mapped, already-`Arc`-wrapped Mmap.
    /// Auto-detects the on-disk flavor (uncompressed vs Phase 3
    /// per-block compressed) by reading the v3 header's `flags` field.
    /// Encrypted brains require the keyfile-supplied master key --
    /// callers that may load encrypted files should use
    /// `from_mmap_with_key` instead.
    pub fn from_mmap(mmap: Arc<Mmap>) -> Result<Self, BrainError> {
        Self::from_mmap_with_key(mmap, None)
    }

    /// Phase 4 entrypoint: like `from_mmap`, but accepts an optional
    /// master key so encrypted Phase 4 brains can be loaded. If the
    /// file's `flags` advertises encryption and `encryption_key` is
    /// `None`, returns a clear error rather than silently producing
    /// garbage.
    pub fn from_mmap_with_key(
        mmap: Arc<Mmap>,
        encryption_key: Option<crate::encrypt::MasterKey>,
    ) -> Result<Self, BrainError> {
        let flags = crate::brain_io_v3::read_v3_flags(&mmap[..])?;
        let is_compressed = flags & crate::compress::FORMAT_FLAG_COMPRESSED != 0;
        let is_encrypted = crate::encrypt::is_encrypted(flags);
        if is_encrypted && !is_compressed {
            return Err(BrainError::Format(
                "v3 flags: encrypted bit set without compressed bit -- invalid combination"
                    .into(),
            ));
        }
        if is_encrypted && encryption_key.is_none() {
            return Err(BrainError::Format(format!(
                "encrypted brain requires keyfile -- pass --keyfile or place one at {}",
                crate::encrypt::DEFAULT_KEYFILE_PATH
            )));
        }
        if !is_encrypted && encryption_key.is_some() {
            // A key was supplied but the file is unencrypted. Don't
            // hard-fail -- the load is still safe, the key is just
            // unused -- but warn so operators notice the mismatch.
            eprintln!(
                "warning: --keyfile provided but brain is not encrypted; ignoring key"
            );
        }
        if is_compressed {
            let (off, len) =
                crate::brain_io_v3::locate_network_section_no_crc(&mmap[..])?;
            let key_for_reader = if is_encrypted { encryption_key } else { None };
            let compressed = crate::compress::CompressedNetwork::from_mmap_with_key(
                mmap, off, len, key_for_reader,
            )?;
            return Ok(NetworkArchive {
                inner: NetworkArchiveInner::Compressed { compressed },
            });
        }
        let net_blob = extract_archived_network_blob_no_crc(&mmap[..])?;
        // Skip rkyv::check_archived_root() validation here: the v3
        // file's CRC32 trailer (verified upstream by parse_layout) and
        // its rkyv-section alignment guarantees suffice for a trusted
        // local file. Walking check_archived_root would touch every
        // page in the 424 MB network blob, defeating the zero-copy
        // RSS win. SAFETY: net_blob came from a CRC-validated v3 file
        // produced by this same crate, so the bytes are
        // well-formed by construction.
        let archived: &ArchivedNetworkSnapshot =
            unsafe { rkyv::archived_root::<NetworkSnapshot>(net_blob) };
        // Build seed → index. ~105K entries: a 4 MB index map for the
        // production brain.
        let mut node_index: HashMap<Seed, u32> = HashMap::with_capacity(archived.nodes.len());
        for (i, n) in archived.nodes.iter().enumerate() {
            node_index.insert(seed_from_archived(&n.seed), i as u32);
        }
        let mut bond_index: HashMap<(Seed, Seed), u32> =
            HashMap::with_capacity(archived.bonds.len());
        for (i, b) in archived.bonds.iter().enumerate() {
            let from = seed_from_archived(&b.from);
            let to = seed_from_archived(&b.to);
            bond_index.insert((from, to), i as u32);
        }
        let archived_ptr = NonNull::from(archived);
        Ok(NetworkArchive {
            inner: NetworkArchiveInner::Uncompressed {
                archived: archived_ptr,
                node_index,
                bond_index,
                _mmap: mmap,
            },
        })
    }

    /// True if this archive is backed by the per-block compressed
    /// Phase 3 network layout.
    pub fn is_compressed(&self) -> bool {
        matches!(self.inner, NetworkArchiveInner::Compressed { .. })
    }

    /// Lookup an archived node by seed. Dispatches to the underlying
    /// uncompressed-mmap or compressed-block backing.
    /// (OP-016: returns `NodeArchive<'a>` so the Compressed backing
    /// can carry an Arc to the decompressed block.)
    pub fn node_by_seed<'a>(&'a self, seed: &Seed) -> Option<NodeArchive<'a>> {
        match &self.inner {
            NetworkArchiveInner::Uncompressed {
                archived, node_index, ..
            } => {
                let i = *node_index.get(seed)?;
                // SAFETY: pointer points into the mmap; lifetime tied to &self.
                let arch = unsafe { archived.as_ref() };
                Some(NodeArchive::Borrowed(&arch.nodes[i as usize]))
            }
            NetworkArchiveInner::Compressed { compressed } => {
                compressed.node_by_seed(seed).map(NodeArchive::Owned)
            }
        }
    }

    /// Lookup an archived bond by (from, to).
    /// (OP-016: returns `BondArchive<'a>`; see `node_by_seed`.)
    pub fn bond_by_pair<'a>(&'a self, key: &(Seed, Seed)) -> Option<BondArchive<'a>> {
        match &self.inner {
            NetworkArchiveInner::Uncompressed {
                archived, bond_index, ..
            } => {
                let i = *bond_index.get(key)?;
                let arch = unsafe { archived.as_ref() };
                Some(BondArchive::Borrowed(&arch.bonds[i as usize]))
            }
            NetworkArchiveInner::Compressed { compressed } => {
                compressed.bond_by_pair(key).map(BondArchive::Owned)
            }
        }
    }

    /// True if the archive contains a node with this seed.
    pub fn has_node(&self, seed: &Seed) -> bool {
        match &self.inner {
            NetworkArchiveInner::Uncompressed { node_index, .. } => node_index.contains_key(seed),
            NetworkArchiveInner::Compressed { compressed } => compressed.has_node(seed),
        }
    }

    /// True if the archive contains a bond with this (from, to).
    pub fn has_bond(&self, key: &(Seed, Seed)) -> bool {
        match &self.inner {
            NetworkArchiveInner::Uncompressed { bond_index, .. } => bond_index.contains_key(key),
            NetworkArchiveInner::Compressed { compressed } => compressed.has_bond(key),
        }
    }

    /// Number of archived nodes.
    pub fn node_count(&self) -> usize {
        match &self.inner {
            NetworkArchiveInner::Uncompressed { node_index, .. } => node_index.len(),
            NetworkArchiveInner::Compressed { compressed } => compressed.node_count(),
        }
    }

    /// Number of archived bonds.
    pub fn bond_count(&self) -> usize {
        match &self.inner {
            NetworkArchiveInner::Uncompressed { bond_index, .. } => bond_index.len(),
            NetworkArchiveInner::Compressed { compressed } => compressed.bond_count(),
        }
    }

    /// Iterate `(seed, NodeArchive<'a>)` pairs. In compressed mode
    /// this triggers a block decompression for every block touched —
    /// the keyword-index build hits this on the first query.
    /// (OP-016: yields `NodeArchive` so the Compressed path can carry
    /// the per-block Arc.)
    pub fn iter_nodes<'a>(
        &'a self,
    ) -> Box<dyn Iterator<Item = (Seed, NodeArchive<'a>)> + 'a> {
        match &self.inner {
            NetworkArchiveInner::Uncompressed {
                archived, node_index, ..
            } => {
                let arch: &'a ArchivedNetworkSnapshot = unsafe { archived.as_ref() };
                Box::new(node_index.iter().map(move |(s, &i)| {
                    (*s, NodeArchive::Borrowed(&arch.nodes[i as usize]))
                }))
            }
            NetworkArchiveInner::Compressed { compressed } => {
                Box::new(compressed.iter_nodes().map(|(s, h)| (s, NodeArchive::Owned(h))))
            }
        }
    }

    /// Iterate `((from, to), BondArchive<'a>)` pairs. (OP-016: yields
    /// `BondArchive` so the Compressed path can carry the per-block
    /// Arc.)
    pub fn iter_bonds<'a>(
        &'a self,
    ) -> Box<dyn Iterator<Item = ((Seed, Seed), BondArchive<'a>)> + 'a> {
        match &self.inner {
            NetworkArchiveInner::Uncompressed {
                archived, bond_index, ..
            } => {
                let arch: &'a ArchivedNetworkSnapshot = unsafe { archived.as_ref() };
                Box::new(bond_index.iter().map(move |(k, &i)| {
                    (*k, BondArchive::Borrowed(&arch.bonds[i as usize]))
                }))
            }
            NetworkArchiveInner::Compressed { compressed } => {
                Box::new(compressed.iter_bonds().map(|(k, h)| (k, BondArchive::Owned(h))))
            }
        }
    }

    /// Iterate just the bond keys without ever dereferencing archived
    /// bond data. Used by `AssociativeNetwork::from_archive` to build
    /// adjacency at load time without page-faulting bond pages
    /// (uncompressed) / decompressing blocks (compressed).
    pub fn iter_bond_keys<'a>(&'a self) -> Box<dyn Iterator<Item = (Seed, Seed)> + 'a> {
        match &self.inner {
            NetworkArchiveInner::Uncompressed { bond_index, .. } => {
                Box::new(bond_index.keys().copied())
            }
            NetworkArchiveInner::Compressed { compressed } => {
                Box::new(compressed.iter_bond_keys())
            }
        }
    }

    /// Iterate just the node seeds — same page-fault story as
    /// `iter_bond_keys`.
    pub fn iter_node_seeds<'a>(&'a self) -> Box<dyn Iterator<Item = Seed> + 'a> {
        match &self.inner {
            NetworkArchiveInner::Uncompressed { node_index, .. } => {
                Box::new(node_index.keys().copied())
            }
            NetworkArchiveInner::Compressed { compressed } => {
                Box::new(compressed.iter_node_seeds())
            }
        }
    }

    /// Borrow the underlying `Arc<Mmap>`. Used by the load path so the
    /// eager scar/side-table reader can share the same mapping.
    pub fn mmap(&self) -> &Arc<Mmap> {
        match &self.inner {
            NetworkArchiveInner::Uncompressed { _mmap, .. } => _mmap,
            NetworkArchiveInner::Compressed { compressed } => compressed.mmap(),
        }
    }

    /// For diagnostics / RSS regression tests: how many compressed
    /// blocks have been faulted into the per-block arena. Returns 0
    /// for the uncompressed backing (which has no arena).
    pub fn blocks_resident(&self) -> usize {
        match &self.inner {
            NetworkArchiveInner::Uncompressed { .. } => 0,
            NetworkArchiveInner::Compressed { compressed } => compressed.blocks_resident(),
        }
    }

    /// For diagnostics / tests: total compressed-block count. Returns 0
    /// for the uncompressed backing.
    pub fn block_count(&self) -> usize {
        match &self.inner {
            NetworkArchiveInner::Uncompressed { .. } => 0,
            NetworkArchiveInner::Compressed { compressed } => compressed.block_count(),
        }
    }

    /// OP-022.s.v3: borrow the inner `CompressedNetwork` if this
    /// archive uses the per-block compressed (Phase 3+4) backing.
    /// Returns `None` for the Uncompressed (Phase 2.2) layout.
    /// Used by the save-side block-passthrough fast path to read
    /// source parameters + compressed block frames verbatim.
    pub fn compressed_network(&self) -> Option<&crate::compress::CompressedNetwork> {
        match &self.inner {
            NetworkArchiveInner::Uncompressed { .. } => None,
            NetworkArchiveInner::Compressed { compressed } => Some(compressed),
        }
    }
}

/// Read the archived `ScarLog` directly from a shared `Arc<Mmap>` of
/// a v3 brain file. The returned `ScarLog` is a fresh heap-resident
/// copy — Slice 2c does not extend zero-copy to the scar log because
/// the scar log is small (KB scale) and active-scar lookups want the
/// heap HashSet/HashMap structure rather than archived form.
pub fn read_scar_log_from_mmap(mmap: &Arc<Mmap>) -> Result<ScarLog, BrainError> {
    use rkyv::Deserialize;
    let blob = extract_archived_scar_blob_no_crc(&mmap[..])?;
    let archived = rkyv::check_archived_root::<ScarLog>(blob).map_err(|e| {
        BrainError::Format(format!(
            "v3 ScarLog validation failed ({} bytes): {}",
            blob.len(),
            e
        ))
    })?;
    let log: ScarLog = archived
        .deserialize(&mut rkyv::Infallible)
        .expect("rkyv Infallible deserializer cannot fail");
    Ok(log)
}
#[cfg(test)]
mod tests {
    
    use crate::bond::{BondType, KeyframeSource};
    use crate::brain::{Brain, BrainSaveFormat, BrainLoadMode};
    use crate::brain_io_v3;
    use crate::node::Confidence;
    use crate::seed::Seed;
    use std::path::PathBuf;

    /// Local tempdir helper (mirrors the trick used by brain_io.rs and
    /// brain_io_v3.rs so we don't pull in the `tempfile` dev-dep).
    fn temp_path(stem: &str) -> PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::SeqCst);
        let mut p = std::env::temp_dir();
        p.push(format!(
            "lu4d-archive-test-{}-{}-{}.lbr",
            std::process::id(),
            n,
            stem
        ));
        p
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

    /// End-to-end smoke test for the mmap-zero-copy path:
    ///   build heap brain → save v3 → load via NetworkArchive →
    ///   AssociativeNetwork::from_archive → look up nodes/bonds.
    #[test]
    fn archive_load_roundtrip_basic_lookups() {
        let path = temp_path("basic");
        let mut b = small_brain();
        b.save_format = BrainSaveFormat::V3Binary;
        b.save_to_format(&path, BrainSaveFormat::V3Binary)
            .expect("v3 save");

        // Open via the production zero-copy path.
        let reloaded = Brain::load_or_new_with_mode(&path, BrainLoadMode::MmapZeroCopy)
            .expect("mmap-zerocopy load");

        // Network is archive-backed.
        assert!(reloaded.network.is_archive_backed());
        assert_eq!(reloaded.network.node_count(), b.network.node_count());
        assert_eq!(reloaded.network.bond_count(), b.network.bond_count());

        // Node lookup serves NodeView::Archive.
        let cell_seed = Seed::from_label("cell");
        let view = reloaded
            .network
            .get_node(&cell_seed)
            .expect("cell node visible via archive");
        assert!(view.is_archive());
        assert_eq!(view.content(), "cell");
        assert_eq!(view.confidence(), Confidence::Verified);
        assert_eq!(view.embedding().len(), crate::node::EMBEDDING_DIM);

        // Bond lookup serves BondView::Archive.
        let bond_view = reloaded
            .network
            .get_bond(&cell_seed, &Seed::from_label("unit_of_life"))
            .expect("cell→unit_of_life bond visible via archive");
        assert!(bond_view.is_archive());
        assert_eq!(bond_view.bond_type(), BondType::IsA);
        assert!(bond_view.keyframe_count() >= 1);
        let _ = std::fs::remove_file(&path);
    }

    /// CoW promotion test: get_node_mut on an archive-resident seed
    /// must materialize a heap copy in the overlay. The mutation
    /// is then visible to subsequent reads.
    #[test]
    fn archive_get_node_mut_cow_promotes_and_persists() {
        let path = temp_path("cow_node");
        let mut b = small_brain();
        b.save_to_format(&path, BrainSaveFormat::V3Binary)
            .expect("v3 save");

        let mut reloaded =
            Brain::load_or_new_with_mode(&path, BrainLoadMode::MmapZeroCopy).unwrap();
        let seed = Seed::from_label("cell");

        // Pre-mutation read returns archive-backed view.
        assert!(reloaded.network.get_node(&seed).unwrap().is_archive());

        // Mutate via NodeMutHandle (CoW promotion happens here).
        {
            let mut h = reloaded
                .network
                .get_node_mut(&seed)
                .expect("cell get_node_mut");
            h.as_mut().typed_ratio = 0.42;
        }

        // Subsequent read returns the heap-promoted copy with the
        // mutation visible.
        let post = reloaded.network.get_node(&seed).unwrap();
        assert!(!post.is_archive(), "post-mutation read must hit overlay");
        assert!((post.typed_ratio() - 0.42).abs() < 1e-6);

        let _ = std::fs::remove_file(&path);
    }

    /// CoW promotion for bonds: mutation through get_bond_mut must
    /// materialize a heap copy.
    #[test]
    fn archive_get_bond_mut_cow_promotes() {
        let path = temp_path("cow_bond");
        let mut b = small_brain();
        b.save_to_format(&path, BrainSaveFormat::V3Binary)
            .expect("v3 save");

        let mut reloaded =
            Brain::load_or_new_with_mode(&path, BrainLoadMode::MmapZeroCopy).unwrap();
        let from = Seed::from_label("cell");
        let to = Seed::from_label("membrane");

        assert!(reloaded.network.get_bond(&from, &to).unwrap().is_archive());

        {
            let mut h = reloaded
                .network
                .get_bond_mut(&from, &to)
                .expect("cell→membrane get_bond_mut");
            // Append a Hebbian keyframe — this is the kind of write
            // the spread activation hot path performs.
            h.add_keyframe(99_999_999_999_999, 0.999, KeyframeSource::Hebbian);
        }

        let post = reloaded.network.get_bond(&from, &to).unwrap();
        assert!(!post.is_archive(), "post-mutation bond must be heap-resident");
        // Confirm the appended keyframe is visible.
        let last_w = post.latest_raw_weight();
        assert!((last_w - 0.999).abs() < 1e-6);

        let _ = std::fs::remove_file(&path);
    }

    /// Equivalence test: archive-backed reads should produce the same
    /// confidence / content / weight values as eager-deserialized
    /// heap reads. Modulo the storage path, the data is identical.
    #[test]
    fn archive_reads_match_heap_reads_for_same_brain() {
        let path = temp_path("equiv");
        let mut b = small_brain();
        b.save_to_format(&path, BrainSaveFormat::V3Binary)
            .expect("v3 save");

        let heap_brain = Brain::load_or_new_with_mode(&path, BrainLoadMode::Eager).unwrap();
        let mmap_brain =
            Brain::load_or_new_with_mode(&path, BrainLoadMode::MmapZeroCopy).unwrap();
        assert!(!heap_brain.network.is_archive_backed());
        assert!(mmap_brain.network.is_archive_backed());

        // Walk every node in the heap brain and confirm the mmap
        // brain returns matching values for content/confidence/
        // typed_ratio/embedding-length.
        for seed in heap_brain.network.all_seeds() {
            let h = heap_brain.network.get_node(&seed).unwrap();
            let m = mmap_brain.network.get_node(&seed).unwrap();
            assert_eq!(h.content(), m.content());
            assert_eq!(h.confidence(), m.confidence());
            assert_eq!(h.embedding().len(), m.embedding().len());
            assert!((h.typed_ratio() - m.typed_ratio()).abs() < 1e-6);
        }

        // Same for bonds.
        let scar_log = crate::scar::ScarLog::default();
        for (key, hbond) in heap_brain.network.bonds().iter() {
            let mbond = mmap_brain
                .network
                .bonds()
                .get(key)
                .expect("bond visible in mmap brain");
            assert_eq!(hbond.bond_type(), mbond.bond_type());
            let hw = hbond.weight_at(i64::MAX, &scar_log).effective_weight;
            let mw = mbond.weight_at(i64::MAX, &scar_log).effective_weight;
            assert!((hw - mw).abs() < 1e-5, "weight mismatch: heap={} mmap={}", hw, mw);
        }

        let _ = std::fs::remove_file(&path);
    }

    /// Save path: a network that is mmap-backed must round-trip cleanly
    /// to a fresh v3 file via to_snapshot.
    #[test]
    fn archive_to_snapshot_materializes_full_graph() {
        let path1 = temp_path("snap1");
        let path2 = temp_path("snap2");
        let mut b = small_brain();
        b.save_to_format(&path1, BrainSaveFormat::V3Binary).unwrap();

        let mmap_brain =
            Brain::load_or_new_with_mode(&path1, BrainLoadMode::MmapZeroCopy).unwrap();
        // Materialize a fresh snapshot from the archive-backed network.
        let snapshot = mmap_brain.network.to_snapshot();
        assert_eq!(snapshot.nodes.len(), b.network.node_count());
        assert_eq!(snapshot.bonds.len(), b.network.bond_count());

        // Write it back as v3 and reload.
        let file = mmap_brain.to_brain_file(brain_io_v3::FORMAT_VERSION_V3);
        let bytes = brain_io_v3::write_v3(&file).unwrap();
        std::fs::write(&path2, &bytes).unwrap();
        let reloaded =
            Brain::load_or_new_with_mode(&path2, BrainLoadMode::Eager).unwrap();
        assert_eq!(reloaded.network.node_count(), b.network.node_count());
        assert_eq!(reloaded.network.bond_count(), b.network.bond_count());

        let _ = std::fs::remove_file(&path1);
        let _ = std::fs::remove_file(&path2);
    }

    /// Skip-CRC reader: extract_archived_network_blob_no_crc must
    /// succeed on a well-formed v3 file.
    #[test]
    fn archive_no_crc_extractor_succeeds_on_well_formed_file() {
        let b = small_brain();
        let bytes = brain_io_v3::write_v3(&b.to_brain_file(brain_io_v3::FORMAT_VERSION_V3))
            .unwrap();
        let mut aligned = rkyv::AlignedVec::with_capacity(bytes.len());
        aligned.extend_from_slice(&bytes);
        let net_blob = brain_io_v3::extract_archived_network_blob_no_crc(&aligned[..]).unwrap();
        let scar_blob = brain_io_v3::extract_archived_scar_blob_no_crc(&aligned[..]).unwrap();
        assert!(!net_blob.is_empty());
        assert!(!scar_blob.is_empty());
        let _ = b; // silence unused mut warning
    }

    /// Slice 2c hand-back regression test: teaching a new bond out of
    /// an archive-resident node must CoW-promote the source node and
    /// refresh its cached `typed_ratio` to reflect the new outgoing
    /// edge. Pre-fix `update_typed_ratio_for` walked through heap-only
    /// `bond_raw` / `self.nodes.get_mut`, so the write silently missed
    /// the archive copy and the stale ratio remained visible to the
    /// narration HubSelector.
    #[test]
    fn archive_update_typed_ratio_for_promotes_and_refreshes() {
        let path = temp_path("typed_ratio_cow");
        let mut b = small_brain();
        // small_brain gives `cell` three outgoing edges, all typed:
        //   cell --IsA-->        unit_of_life
        //   cell --HasProperty-> membrane
        //   cell --Produces-->   mitosis
        // so typed_ratio(cell) == 1.0 at save time.
        b.save_to_format(&path, BrainSaveFormat::V3Binary)
            .expect("v3 save");

        let mut reloaded =
            Brain::load_or_new_with_mode(&path, BrainLoadMode::MmapZeroCopy).unwrap();
        let cell = Seed::from_label("cell");

        // Pre-mutation invariant: cell is archive-resident and the
        // archived typed_ratio reads as ~1.0.
        let pre = reloaded.network.get_node(&cell).unwrap();
        assert!(pre.is_archive(), "pre-test: cell must start archive-resident");
        assert!((pre.typed_ratio() - 1.0).abs() < 1e-6,
            "pre-test: archived typed_ratio should be 1.0, got {}", pre.typed_ratio());

        // Add a RelatedTo edge from cell to a brand-new node. teach
        // ends up calling network.add_bond → update_typed_ratio_for
        // for the source seed.
        let _ = reloaded.teach(
            "cell",
            BondType::RelatedTo,
            "misc_filler_node",
            0.5,
            Confidence::Verified,
        );

        // Post-condition: cell must now be heap-resident (CoW
        // promoted) and its typed_ratio must reflect 3 typed / 4
        // total = 0.75.
        let post = reloaded.network.get_node(&cell).unwrap();
        assert!(!post.is_archive(),
            "post-fix: teach should CoW-promote `cell` to overlay");
        let ratio = post.typed_ratio();
        assert!((ratio - 0.75).abs() < 1e-6,
            "post-fix: typed_ratio should drop to 0.75, got {}", ratio);

        let _ = std::fs::remove_file(&path);
    }
}
