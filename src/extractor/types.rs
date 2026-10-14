//! Text-extractor bond shape.
//!
//! Mirrors the Python `smart_extractor`/`ingest_curriculum_full`
//! bond dict but uses Lu's existing `BondType` enum from
//! `crate::bond` so callers can map straight onto `PendingChange::
//! AddBond` without an intermediate string-relation step.
//!
//! `kind` is the KnowledgeContent tag string ("definition" /
//! "composition" / "purpose" / "fact" / "concept") returned by
//! `crate::node::infer_kind_from_relation`. The extractor stamps it
//! at emit time so downstream code can route the bond onto a typed
//! KU variant without re-running inference.

use crate::bond::BondType;

/// Confidence label as emitted by the extractor. Mirrors the
/// Python strings ("Inferred" / "Unverified"). The MCP tool maps
/// these to a numeric confidence at queue time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfidenceLabel {
    Inferred,
    Unverified,
}

impl ConfidenceLabel {
    pub fn as_str(&self) -> &'static str {
        match self {
            ConfidenceLabel::Inferred => "Inferred",
            ConfidenceLabel::Unverified => "Unverified",
        }
    }

    /// Numeric score used by the MCP wrapper when creating the
    /// PendingChange entry. Mirrors the values used by
    /// `mcp_file_io::tool_file_extract_to_pending` (0.65 for
    /// Inferred, 0.5 for Unverified).
    pub fn score(&self) -> f32 {
        match self {
            ConfidenceLabel::Inferred => 0.65,
            ConfidenceLabel::Unverified => 0.5,
        }
    }
}

/// One extracted bond. `layer` records which extractor layer
/// produced it so callers can build a histogram ("L1: 4 bonds,
/// L4: 73 bonds, ..."). `source_url` propagates the article URL
/// into the bond's source attribution.
#[derive(Debug, Clone)]
pub struct Bond {
    pub subject: String,
    pub relation: BondType,
    pub object: String,
    pub kind: &'static str,
    pub weight: f32,
    pub confidence: ConfidenceLabel,
    pub layer: &'static str,
    pub source_url: Option<String>,
    /// Frequency annotation for L4 body co-mention bonds; None for
    /// every other layer. Surfaced through the MCP response sample
    /// so the caller can see why a bond got the weight it did.
    pub frequency: Option<u32>,
    /// Set by the multi-sentence extractor: "subject" (Path 1) or
    /// "anchor" (Path 2). None for layers L1-L5.
    pub subject_origin: Option<&'static str>,
}

impl Bond {
    /// Dedup key: lowercase subject + raw relation tag + lowercase
    /// object. Mirrors the Python
    /// `(subject.lower(), relation, object.lower())` tuple used
    /// in `extract_bonds_from_wikipedia_article`.
    pub fn dedup_key(&self) -> (String, BondType, String) {
        (self.subject.to_lowercase(), self.relation, self.object.to_lowercase())
    }
}

/// Map a relation tag string (as found in the Python source of
/// truth) onto Lu's `BondType` enum. Unknown strings fall back to
/// `RelatedTo` — the same defensive default the Python
/// `_kind_for` helper uses.
pub fn bondtype_from_relation_str(rel: &str) -> BondType {
    match rel {
        "IsA" => BondType::IsA,
        "HasProperty" => BondType::HasProperty,
        "Causes" => BondType::Causes,
        "PartOf" => BondType::PartOf,
        "UsedFor" => BondType::UsedFor,
        "RelatedTo" => BondType::RelatedTo,
        "ContrastedWith" => BondType::ContrastedWith,
        "DerivedFrom" => BondType::DerivedFrom,
        "Requires" => BondType::Requires,
        "Produces" => BondType::Produces,
        "Opposes" => BondType::Opposes,
        "BornIn" => BondType::BornIn,
        "BornOn" => BondType::BornOn,
        "DiedIn" => BondType::DiedIn,
        "DiedOn" => BondType::DiedOn,
        "DiscoveredBy" => BondType::DiscoveredBy,
        "FoundedBy" => BondType::FoundedBy,
        "FoundedIn" => BondType::FoundedIn,
        "HasAltitude" => BondType::HasAltitude,
        "HasAspect" => BondType::HasAspect,
        "HasCapital" => BondType::HasCapital,
        "HasOfficialLanguage" => BondType::HasOfficialLanguage,
        "HasPopulation" => BondType::HasPopulation,
        "KnownFor" => BondType::KnownFor,
        "LocatedIn" => BondType::LocatedIn,
        "MeasuredIn" => BondType::MeasuredIn,
        "NamedAfter" => BondType::NamedAfter,
        "WrittenBy" => BondType::WrittenBy,
        "HasArea" => BondType::HasArea,
        "HasMethod" => BondType::HasMethod,
        "HasAttribute" => BondType::HasAttribute,
        "HasParameter" => BondType::HasParameter,
        "Returns" => BondType::Returns,
        "Inherits" => BondType::Inherits,
        "Calls" => BondType::Calls,
        "HasType" => BondType::HasType,
        "Implements" => BondType::Implements,
        "HasCurrency" => BondType::HasCurrency,
        "AuthoredBy" => BondType::AuthoredBy,
        "ComposedOf" => BondType::PartOf,
        _ => BondType::RelatedTo,
    }
}

/// Look up the static kind tag string used by `Bond.kind`.
/// Centralised so tests can pin the string set without sprinkling
/// match arms across the layer files.
pub fn kind_for(rel: BondType) -> &'static str {
    use BondType::*;
    match rel {
        IsA => "definition",
        PartOf => "composition",
        UsedFor => "purpose",
        HasAspect | HasProperty | KnownFor => "fact",
        LocatedIn | FoundedBy | FoundedIn | DiscoveredBy | NamedAfter | WrittenBy => "fact",
        HasCurrency | AuthoredBy => "fact",
        BornIn | DiedIn | BornOn | DiedOn => "fact",
        MeasuredIn | HasPopulation | HasAltitude | HasArea | HasOfficialLanguage | HasCapital => "fact",
        DefinedAs => "definition",
        WorksBy => "mechanism",
        ComposedOf => "composition",
        PurposeOf => "purpose",
        DiffersFrom => "contrast",
        LeadsTo => "causal",
        ExampleOf => "example",
        // Generic association layers (L1, L4 RelatedTo) intentionally
        // route to the default "concept" tag — mirrors the Python
        // source-of-truth `_RELATION_TO_KIND` comment.
        _ => "concept",
    }
}


/// OP-EXTRACT-WIDEN -- relation specificity for pair-level reconciliation.
/// The brain dedups bonds by (from, to) PAIR ONLY (not bond_type), keeping the
/// first-seen type. Layer order emits generic RelatedTo (L1/L4) and IsA (L2b)
/// before the typed L3/L5 relations, so without reconciliation the typed
/// relation loses the race and the pair lands generic internal notes.
/// `extract_from_article` keeps, per (subject, object) pair, the bond whose
/// relation has the highest priority here. Lockstep with `_RELATION_PRIORITY`
/// in scripts/smart_extractor.py.
pub fn relation_priority(bt: BondType) -> u8 {
    use BondType::*;
    match bt {
        // IsA tied with RelatedTo (0): on a same-pair collision IsA is often
        // a noisy see-also / disambiguation typing; promoting a generic
        // RelatedTo to it manufactures false taxonomy. The precise relations
        // below still win.
        RelatedTo | IsA => 0,
        HasAspect | HasProperty | KnownFor | DefinedAs => 1,
        LocatedIn | FoundedBy | FoundedIn | DiscoveredBy | NamedAfter | WrittenBy
        | BornIn | BornOn | DiedIn | DiedOn | MeasuredIn | HasPopulation | HasArea
        | HasAltitude | HasOfficialLanguage | HasCapital => 3,
        // Every other typed relation (PartOf, UsedFor, Causes, ContrastedWith,
        // DerivedFrom, Requires, Produces, Opposes, ...) is structural/functional:
        // more specific than the generic RelatedTo, below the factual attributes.
        _ => 2,
    }
}

#[cfg(test)]
mod op_extract_widen_tests {
    use super::*;
    #[test]
    fn relation_priority_orders_generic_below_typed() {
        // IsA is tied with RelatedTo (both generic); typed relations beat both.
        assert_eq!(relation_priority(BondType::RelatedTo), relation_priority(BondType::IsA));
        assert!(relation_priority(BondType::IsA) < relation_priority(BondType::LocatedIn));
        assert!(relation_priority(BondType::HasAspect) > relation_priority(BondType::IsA));
        assert!(relation_priority(BondType::HasProperty) < relation_priority(BondType::HasPopulation));
        assert!(relation_priority(BondType::PartOf) > relation_priority(BondType::RelatedTo));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kind_for_canonical() {
        assert_eq!(kind_for(BondType::IsA), "definition");
        assert_eq!(kind_for(BondType::PartOf), "composition");
        assert_eq!(kind_for(BondType::PartOf), "composition");
        assert_eq!(kind_for(BondType::UsedFor), "purpose");
        assert_eq!(kind_for(BondType::HasAspect), "fact");
        assert_eq!(kind_for(BondType::HasPopulation), "fact");
        assert_eq!(kind_for(BondType::BornIn), "fact");
        assert_eq!(kind_for(BondType::RelatedTo), "concept");
    }

    #[test]
    fn bondtype_from_relation_str_known_and_unknown() {
        assert_eq!(bondtype_from_relation_str("IsA"), BondType::IsA);
        assert_eq!(bondtype_from_relation_str("HasPopulation"), BondType::HasPopulation);
        assert_eq!(bondtype_from_relation_str("BogusRelation"), BondType::RelatedTo);
    }
}
