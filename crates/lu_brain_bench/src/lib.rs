//! OP-BRAIN-V5 Phase 0.c — bench harness library.
//!
//! This crate is primarily a collection of bench binaries (one per
//! `[[bin]]` entry in `Cargo.toml`). The library target exists solely to
//! host shared infrastructure that those binaries — and the v5-substrate
//! per-phase RSS gates downstream of Phase 0.c — call into.
//!
//! ## Exported modules
//!
//! - [`harness`] — Phase 0.c deliverable. Streaming reader for the
//!   `.bench` fixture format written by `gen_prod_shape_fixture`, an
//!   RSS sampler that polls `/proc/self/status` on a background
//!   thread, a budget API (`with_rss_budget`), an `RssReport`
//!   serializer, a predicate-coverage checker, and a workload
//!   registry.
//!
//! See `docs/OP-BRAIN-V5-Design.md` Section 4 for the phase plan and
//! gate 2 discipline ("each phase must have a bounded-RSS assertion
//! target before any port spawn"). Phase 0.c is the assertion
//! *mechanism*; per-phase thresholds are set by Phase 1+ briefs, not
//! here.

pub mod harness;
