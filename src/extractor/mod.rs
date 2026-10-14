//! Rust port of Lu's Python smart_extractor + multi-sentence
//! response-bond extractor.
//!
//! Layered design mirrors the Python source-of-truth files:
//!   * scripts/smart_extractor.py — L1 / L2 / L2b / L3 / L4 / L5
//!   * scripts/ingest_curriculum_full.py — LM multi-sentence (Path 1
//!     subject, Path 2 anchor)
//!
//! Top-level entry point is `extract_from_article`. The MCP tool
//! `lu4d_learn_text` wraps it with a knowledge-aware filter and
//! queues every surviving bond as a `PendingChangeKind::AddBond` on
//! the T.3 PendingLog.

pub mod aliases;
pub mod dispatch;
pub mod labels;
pub mod patterns;
pub mod sentence;
pub mod tail_strip;
pub mod types;

pub mod layers {
    pub mod shared_noise;
    pub mod l1_see_also;
    pub mod l2_section_headings;
    pub mod l2b_category_tail;
    pub mod l3_lead_patterns;
    pub mod l4_body_comentions;
    pub mod l5_numerical_attrs;
    pub mod lm_multi_sentence;
}

pub use dispatch::extract_from_article;
pub use types::{kind_for, Bond, ConfidenceLabel};
