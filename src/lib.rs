#![recursion_limit = "512"]

//! Lu 4D — Temporal Knowledge Graph.

pub mod seed;
pub mod node_intern;
pub mod bond;
pub mod node;
pub mod scope;
pub mod bridge;
pub mod network;
pub mod context;
pub mod session;
pub mod consolidation;
pub mod timetravel;
pub mod brain;
pub mod brain_io;
pub mod brain_io_v3;
pub mod thinking;
pub mod curiosity;
pub mod variants;
pub mod self_reflection;
pub mod sense;
pub mod mcp_sense;
pub mod agents;
pub mod near_miss;      // OP-NEARMISS-1 -- spelling repair bounded by the keyword index
pub mod narration;
pub mod reason_loop;  // OP-REASON-LOOP -- prompt-reasoning-loop ContextSet accumulator
pub mod synthesis;  // OP-CONCLUDE -- conclusion synthesizer (Tier 1+)
pub mod epistemic;  // OP-EPISTEMIC (Phase 3) -- confidence -> epistemic-language realizer
pub mod fluent;    // OP-FLUENT (Phase 4) -- grounded grammar/aggregation realizer (answer-side)
pub mod identity;   // OP-IDENTITY -- locked, protected identity / provenance core
pub mod query_router;
pub mod canonical;  // OP-CANON-1 -- ingest label canonicalisation (unwired; gated with prune)
pub mod canon_merge; // OP-CANON-1 -- offline fold/bond-redirect planner (pure; gated with prune)
pub mod scar;
pub mod view;
pub mod archive;
pub mod compress;
pub mod encrypt;
pub mod rkyv_format;
pub mod research;
pub mod reasoning;
pub mod math;
pub mod session_memory;
pub mod idk_gate;
pub mod greet_refusal;
pub mod gap_signal;
pub mod wonder_sidecar;  // OP-CURIOSITY-1 -- durable cross-process wonder queue
pub mod confidence;
pub mod negation;
pub mod source;
pub mod overlay_sidecar;   // OP-REVOKE-V5 -- encrypted source/negation overlay sidecars
pub mod time;
pub mod idle;
pub mod pending;
pub mod mcp_t3;
pub mod mcp_t5;
pub mod mcp_t6;
pub mod decay;
pub mod mcp_envelope;
#[cfg(feature = "admin_mcp")]
pub mod mcp;
pub mod rejection;
pub mod mcp_t8;
pub mod mcp_t11;
pub mod lb_session;
pub mod mcp_lb;
pub mod style;
pub mod mcp_cs;
pub mod pack;
pub mod mcp_pack;
pub mod capability_pack;
pub mod capability_runtime;
pub mod code_exec;        // OP-CODE-EXEC Phase 1 — native X1 sandbox invocation
pub mod mcp_cat5;
pub mod mcp_cap_runtime;
pub mod trust;
pub mod mcp_cat3a;
pub mod hub_degree;
pub mod runtime_config;
pub mod mcp_cat3b;
pub mod file_io;
pub mod mcp_file_io;
pub mod vision;
pub mod mcp_vision;
pub mod mcp_runtime_setters;
pub mod keyword_index;  // OP-022.a: mmap'd keyword index sidecar
pub mod save_async;  // OP-SAVE-ASYNC: non-blocking background save
pub mod extractor;  // Text-extractor port (2026-05-31)
pub mod embedding;  // OP-OFFNESS-1 port -- lexical embedder (BgeEmbedder behind off-by-default bge feature)
pub mod offness;    // OP-OFFNESS-1 port -- post-answer off-ness self-check (John Cattle Dog), inert unless LU_OFFNESS_ENABLED
pub mod mcp_extractor;  // lu4d_learn_text MCP tool
pub mod sst;          // OP-BRAIN-V4 Phase 2 — SSTable reader
pub mod writeback;      // OP-WRITEBACK Phase 5 — compounding writeback + immune system
pub mod sst_writer;   // OP-BRAIN-V4 Phase 2 — SSTable writer
pub mod memtable;     // OP-BRAIN-V4 Phase 3 — MemTable
pub mod wal;          // OP-BRAIN-V4 Phase 3 — Write-Ahead Log
pub mod manifest;     // OP-BRAIN-V4 Phase 4 — Manifest
pub mod flush;        // OP-BRAIN-V4 Phase 4 — Flush worker
pub mod read_path;    // OP-BRAIN-V4 Phase 5 — canonical merged read path
pub mod node_store;   // OP-BRAIN-V5 Phase 2 — node store substrate (sidecar I/O Phase 2.1)
pub mod bond_store;   // OP-BRAIN-V5 Phase 3 -- bond store substrate (sst_writer scaffold Phase 3.1)
pub mod map_ram;      // OP-BRAIN-MapRAM P0 -- resident MAP (NodeId -> degree), brain on disk
pub mod cache_adjacency; // OP-BRAIN-MapRAM P1 -- cache-backed adjacency (map->cache->v5 reader)
pub mod read_only_reader; // OP-007 -- read-only guest reader (live-serving foundation)

pub use seed::Seed;
pub use bond::{BondType, TemporalBond, TemporalKeyframe, KeyframeSource};
pub use bridge::{BridgeCandidate, BridgeCollector, BridgeConfig, BridgeEvent};
pub use node::{KnowledgeNode, KnowledgeContent, KnowledgeUnit, Confidence};
pub use scope::{AgentRole, TemporalScope};
pub use network::{
    AssociativeNetwork, RecallResult, ActivatedNode, NetworkError, NetworkSnapshot,
};
pub use context::{ContextSelector, ContextResult};
pub use session::{TemporalSession, SessionManager};
pub use consolidation::{ConsolidationEngine, ConsolidationConfig, ConsolidationStats};
pub use timetravel::{
    time_travel_query, time_travel_with_seeds, TemporalDiff, NodeDiff, ChangeType,
};
pub use brain::{BrainLoadMode,
    Brain, BrainError, BrainFile, BrainFormatOverride, BrainLayout, BrainSaveFormat, BrainStatus,
    TeachResult, AskResult,
    parse_bond_type, parse_confidence, now_ms, BRAIN_FORMAT_VERSION, V1_JSON_VERSION,
};
pub use brain_io::{BrainFormat, FORMAT_VERSION_V2};
pub use brain_io_v3::FORMAT_VERSION_V3;
pub use thinking::{
    think, Goal, HypothesisCandidate, Observation, Plan, Reflection,
    ThinkConfig, ThinkResult, ThinkStep, ThoughtTrace,
    MAX_HYPOTHESIS_BONDS_PER_GOAL,
};
pub use curiosity::{
    classify_curiosity_candidates, CuriosityCandidate, CuriosityCollector,
    CuriosityConfig, CuriositySignal, InterestKeyframe, InterestProfile,
    RankedSignal, SignalType, WonderQueue, DEFAULT_INTEREST_HALFLIFE_MS,
};
pub use self_reflection::{
    analyze as analyze_self_reflection, AdjustmentDirection, AdjustmentProposal,
    AdjustmentTarget, CurrentKnobs, MetaReflection, PerformanceMonitor,
    ProposalStatus, QueryMetric, SelfReflectionConfig,
};
pub use variants::{
    PerturbationStrategy, VariantArena, VariantConfig, VariantScore, VariantSpec,
};
pub use agents::{
    multi_think, AgentStep, Belief, MultiAgentConfig, MultiAgentTrace, RecalledItem,
};
pub use narration::{
    narrate, IntentClassifier, ModulationFlags, NarrationConfig, NarrationResult,
    QueryIntent, Renderer, ResponseShape, ShapeSelector, ShapeSlots,
};
pub use reasoning::{reason as reason_fn, reason_with_config, reason_from_recall, articulate_bond, articulate_bond_typed, EvidenceKind, ReasoningConfig, ReasoningStep, ReasoningTrace};
pub use scar::{Scar, ScarAwareReading, ScarLog, UnscarRecord};
pub use view::{NodeView, BondView, BondsView, NodeMutHandle, BondMutHandle};
pub use session_memory::{Anchor, BackrefKind, SessionBelief, SessionMemory, SessionMemoryRegistry, Turn, TurnTraceSummary, classify_backreference, contains_bare_pronoun, is_continuation, is_elaboration, substitute_pronoun, ANCHOR_DECAY_FACTOR, DEFAULT_IDLE_TIMEOUT_MS, MAX_TURN_HISTORY, PRONOUN_RESOLUTION_THRESHOLD};
pub mod compaction;     // OP-BRAIN-V4 Phase 6 -- Leveled compaction
pub mod migration_v3_v4;  // OP-BRAIN-V4 Phase 7 -- v3 -> v4 migration tool
pub mod migration_v3_v5_node;  // OP-BRAIN-V5 Phase 2.9 -- v3 -> v5 node-side migration tool
pub mod migration_v3_v5_bond;  // OP-BRAIN-V5 Phase 3.10 -- v3 -> v5 bond-side migration tool
pub mod migration_verify_v5;  // OP-BRAIN-V5 6.7 -- offline v3<->v5 migration verifier
