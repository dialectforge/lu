//! Port R input — user-facing image ingest.
//!
//! Lu accepts images from users and emits scene-graph triplets through
//! the existing teach_corpus / user_statement pipes. The actual scene-
//! graph extraction is pluggable behind the `SceneGraphExtractor` trait
//! defined in `extractor.rs`. This port ships:
//!
//!   * `StubExtractor`        - placeholder for development. Returns
//!                              a single ("image", "uploaded_at", ts)
//!                              triplet. NOT for production use.
//!   * `SubprocessExtractor`  - shells out to a configured external
//!                              CV pipeline. Wired but inactive by
//!                              default; operators opt in via
//!                              runtime_config section #15.
//!
//! The future `CapabilityPackExtractor` (Cat.5.1) — once signed
//! capability packs can carry executable extractors — is documented
//! in `extractor.rs::CapabilityPackExtractor` but NOT implemented
//! here. That lands as a Cat.5.1 follow-up.
//!
//! Articulator-side: no brain schema changes. The triplets returned
//! by an extractor flow through existing brain APIs (teach_corpus or
//! lb_session UserStatement). C5 invariant: stored bond weights are
//! never mutated; we only ever ADD new bonds via the existing paths.

pub mod extractor;
pub mod geo_perceive;

pub use extractor::{
    ExtractError, ExtractHints, SceneGraphExtractor, StubExtractor,
    SubprocessExtractor, Triplet,
};

pub mod geo_perceive_p4;
pub mod geo_primitives;
pub mod geo_labels;
pub mod geo_constraints;
