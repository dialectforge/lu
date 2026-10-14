//! OP-007 / Port U privilege-separation property test.
//!
//! The whole point of the two-binary split is that the admin tool
//! handlers are PHYSICALLY ABSENT from the public guest binary — compiled
//! out via crate/feature boundaries, not merely permission-checked at
//! runtime. This test verifies that property empirically by scanning the
//! built `lu_guest_http` binary for admin tool-name strings.
//!
//! Mechanism under test: the guest crate depends on `lu4d` with
//! `default-features = false`, which drops the `admin_mcp` feature, so the
//! `lu4d::mcp` module (whose `tool_definitions()` embeds every admin tool
//! name) is never compiled into this binary.
//!
//! `CARGO_BIN_EXE_lu_guest_http` is set by Cargo for integration tests and
//! points at the freshly built guest binary.

use std::fs;

/// Admin tool names whose ONLY definition site is the gated `lu4d::mcp`
/// module (`tool_definitions()`). None of these may appear in the guest
/// binary.
const ADMIN_TOOLS: &[&str] = &[
    "lu4d_teach",
    "lu4d_configure",
    "lu4d_approve_pending",
    "lu4d_reject_pending",
    "lu4d_revoke_by_source",
    "lu4d_trust_set",
    "lu4d_modify_pending",
    "lu4d_prune_legacy_goals",
    // Brain-mutation tools whose handler code is linked (but unreachable)
    // yet whose tool-NAME literals live only in the gated
    // mcp::tool_definitions(). None may surface on the guest. Extends the
    // mutation-boundary check internal notes guest-capability spec).
    "lu4d_image_ingest",
    "lu4d_file_extract_to_pending",
    "lu4d_scar",
];

/// Guest tools that MUST be present — proves the scan is meaningful (i.e.
/// we are scanning a real binary that contains the strings we can see).
const GUEST_TOOLS: &[&str] = &["lu4d_ask", "lu4d_session_start", "lu4d_session_end", "lu4d_image_describe", "lu4d_file_read"];

fn contains(haystack: &[u8], needle: &[u8]) -> bool {
    if needle.is_empty() || haystack.len() < needle.len() {
        return false;
    }
    haystack.windows(needle.len()).any(|w| w == needle)
}

#[test]
fn admin_tools_physically_absent_from_guest_binary() {
    let exe = env!("CARGO_BIN_EXE_lu_guest_http");
    let bytes = fs::read(exe).unwrap_or_else(|e| panic!("read guest binary {}: {}", exe, e));

    // Positive control: the check only means something if the guest tool
    // strings ARE present in the same binary.
    for tool in GUEST_TOOLS {
        assert!(
            contains(&bytes, tool.as_bytes()),
            "guest tool string {:?} missing from {} — the absence check would be vacuous",
            tool,
            exe
        );
    }

    // The property: NO admin tool string is linked in.
    let mut leaked = Vec::new();
    for tool in ADMIN_TOOLS {
        if contains(&bytes, tool.as_bytes()) {
            leaked.push(*tool);
        }
    }
    assert!(
        leaked.is_empty(),
        "PRIVSEP VIOLATION: admin tool string(s) {:?} found in guest binary {} — admin handlers are NOT physically absent",
        leaked,
        exe
    );
}

// OP-REASON-LOOP Phase 1.5 -- runtime privilege-separation property: the
// session-scoped context hold accumulates across turns but writes NOTHING
// to the canonical graph. Builds a real v5 brain, runs the exact guest
// accumulation path (narrate -> get_or_create -> record_turn ->
// session_context.merge_prompt/evict) over a multi-turn session, and
// asserts node_count/bond_count are unchanged, that the hold accumulated,
// and that session_end + evict_idle drop it.
#[test]
fn session_context_accumulation_writes_no_nodes_or_bonds() {
    use lu4d::bond::BondType;
    use lu4d::node::Confidence;
    use lu4d::Brain;

    let tmp = std::env::temp_dir().join(format!("lu_privsep_sctx_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(tmp.join("node_store")).unwrap();
    std::fs::create_dir_all(tmp.join("bond_store")).unwrap();
    std::fs::create_dir_all(tmp.join("wal")).unwrap();
    let fp = lu4d::manifest::KeyFingerprint::none();
    let mut m = lu4d::manifest::Manifest::new(fp);
    m.set_brain_format_v5();
    m.save_atomic(&tmp.join(lu4d::manifest::MANIFEST_FILE_NAME)).unwrap();

    let mut brain = Brain::load_v5(&tmp, None).expect("v5 writer must boot");
    brain.teach("rust", BondType::IsA, "programming language", 0.95, Confidence::Verified);
    brain.teach("rust", BondType::RelatedTo, "memory safety", 0.9, Confidence::Verified);
    brain.teach("python", BondType::IsA, "programming language", 0.95, Confidence::Verified);
    brain.flush_now().expect("flush_now must seal an L0 SST");
    brain.build_keyword_index_v5().expect("writer must build the v5 keyword sidecar");

    let n0 = brain.status().node_count;
    let b0 = brain.status().bond_count;

    let sid = 4242u64;
    for q in ["rust", "python", "rust"] {
        let cfg = lu4d::narration::NarrationConfig { idk_gate_enabled: false, ..Default::default() };
        let nr = brain.narrate_with_config(q, 3, 10, false, &cfg);
        let now = lu4d::brain::now_ms();
        let mem = brain.session_memories.get_or_create(sid, now);
        mem.record_turn(
            q.to_string(),
            q.to_string(),
            nr.answer.clone(),
            Vec::new(),
            lu4d::session_memory::TurnTraceSummary::empty(),
            now,
        );
        if let Some(cs) = nr.context_set.as_ref() {
            let turn = mem.turn_history.len() as u64;
            mem.session_context.merge_prompt(cs, turn);
            mem.session_context.evict(turn);
        }
    }

    // Deterministic accumulation proof, independent of fixture recall
    // strength: merge a hand-built non-empty ContextSet into the same
    // session. This exercises the merge/evict path with real content and,
    // together with the count assertions below, proves that merging content
    // into the session hold writes NOTHING to the graph.
    {
        let now = lu4d::brain::now_ms();
        let ask = lu4d::brain::AskResult {
            query: "synthetic".to_string(),
            seed_hits: Vec::new(),
            recall: lu4d::network::RecallResult {
                activated: vec![
                    lu4d::network::ActivatedNode {
                        seed: lu4d::seed::Seed::from_bytes([7u8; 32]),
                        content: "held_a".to_string(),
                        activation: 0.9,
                        confidence: lu4d::node::Confidence::Verified,
                        path: Vec::new(),
                        depth: 0,
                    },
                    lu4d::network::ActivatedNode {
                        seed: lu4d::seed::Seed::from_bytes([8u8; 32]),
                        content: "held_b".to_string(),
                        activation: 0.5,
                        confidence: lu4d::node::Confidence::Verified,
                        path: Vec::new(),
                        depth: 1,
                    },
                ],
                query_seeds: Vec::new(),
                scope: lu4d::scope::TemporalScope::Current,
                nodes_visited: 0,
                bonds_evaluated: 0,
                activation_truncated: false,
                status: lu4d::network::SpreadStatus::Complete,
            },
            hebbian_strengthened: 0,
            session_id: None,
            bridges: Vec::new(),
            curiosity_signals: Vec::new(),
            scars: Vec::new(),
            status: lu4d::brain::AskStatus::Complete,
        };
        let cs = lu4d::reason_loop::ContextSet::from_ask(&ask, 9, 0, lu4d::reason_loop::DEFAULT_READ_BUDGET);
        let mem = brain.session_memories.get_or_create(sid, now);
        mem.session_context.merge_prompt(&cs, 9);
        mem.session_context.evict(9);
    }

    // (1) The session-context path wrote no nodes and no bonds.
    assert_eq!(brain.status().node_count, n0, "node_count changed under session-context accumulation");
    assert_eq!(brain.status().bond_count, b0, "bond_count changed under session-context accumulation");

    // (2) The hold accumulated context across the session.
    {
        let mem = brain.session_memories.get(sid).expect("session should exist");
        assert!(mem.session_context.contains_label("held_a"), "warm-start held");
        assert!(!mem.session_context.is_empty(), "session context accumulated nothing across turns");
        assert!(mem.session_context.len() <= mem.session_context.cap(), "hold exceeded its cap");
    }

    // (3) session_end drops the whole hold.
    brain.session_memories.end(sid);
    assert!(brain.session_memories.get(sid).is_none(), "session_end did not drop the session");

    // (4) Idle eviction drops an idle session too.
    let now = lu4d::brain::now_ms();
    let stale = now - lu4d::session_memory::DEFAULT_IDLE_TIMEOUT_MS - 1;
    let _ = brain.session_memories.get_or_create(sid, stale);
    brain.session_memories.evict_idle(now);
    assert!(brain.session_memories.get(sid).is_none(), "evict_idle did not drop the idle session");

    // Counts still pristine after the whole lifecycle.
    assert_eq!(brain.status().node_count, n0);
    assert_eq!(brain.status().bond_count, b0);

    let _ = std::fs::remove_dir_all(&tmp);
}
// OP-REASON-LOOP Phase 2 (FLAG D) -- runtime privilege-separation property:
// read-only reasoning (ThinkConfig.no_write / MultiAgentConfig.no_write)
// writes NOTHING to the graph, so repeated think/multi_think passes on a
// shared brain cannot mutate the in-RAM network or bleed Hypothesis/Agent
// bonds into another request's spread. The write-mode contrast proves the
// test is not vacuous (reasoning DOES mutate the graph without the flag).
//
// Uses an ephemeral heap Brain (Brain::new) -- the same construction the
// lu4d think/agents unit tests use -- where recall is reliable and
// status().bond_count reflects the heap graph directly (a v5 fixture would
// report 0 heap bonds post-flush, since committed bonds live in SSTs).
#[test]
fn read_only_reasoning_writes_no_bonds_no_cross_request_bleed() {
    use lu4d::bond::BondType;
    use lu4d::node::Confidence;
    use lu4d::Brain;

    let mut brain = Brain::new();
    // A 4-sibling star: recall("science") surfaces physics/biology/chemistry/
    // geology with NO cross-bonds, so think()'s PLAN proposes fresh Hypothesis
    // bonds between them -- write-mode reliably mutates the graph.
    brain.teach("science", BondType::IsA, "physics", 0.9, Confidence::Verified);
    brain.teach("science", BondType::IsA, "biology", 0.9, Confidence::Verified);
    brain.teach("science", BondType::IsA, "chemistry", 0.9, Confidence::Verified);
    brain.teach("science", BondType::IsA, "geology", 0.9, Confidence::Verified);
    // A 2-hop chain for multi_think synthesis (rust -> prog language -> safety).
    brain.teach("rust", BondType::IsA, "programming language", 0.95, Confidence::Verified);
    brain.teach("programming language", BondType::HasProperty, "type safety", 0.9, Confidence::Verified);
    brain.teach("python", BondType::IsA, "programming language", 0.95, Confidence::Verified);

    let b0 = brain.status().bond_count;
    let n0 = brain.status().node_count;
    assert!(b0 > 0, "fixture must have bonds for the test to be meaningful");

    // (A) N read-only think passes -> zero graph writes.
    for _ in 0..4 {
        let cfg = lu4d::thinking::ThinkConfig { no_write: true, max_wall_clock: None, ..Default::default() };
        let _ = lu4d::thinking::think(&mut brain, "science", cfg);
    }
    assert_eq!(brain.status().bond_count, b0, "no_write think() added bonds");
    assert_eq!(brain.status().node_count, n0, "no_write think() added nodes");

    // (B) N read-only multi_think passes -> zero graph writes.
    for _ in 0..4 {
        let mut mcfg = brain.multi_agent_config.clone();
        mcfg.no_write = true;
        let _ = brain.multi_think_with_config("rust", &mcfg);
    }
    assert_eq!(brain.status().bond_count, b0, "no_write multi_think added bonds");
    assert_eq!(brain.status().node_count, n0, "no_write multi_think added nodes");

    // (C) No cross-request bleed: after all the read-only reasoning above the
    // shared network still has exactly b0 bonds -- there is no Hypothesis/Agent
    // bond (and no leaked goal/step node) for a subsequent request to traverse.
    let recall = brain.ask_with_config("science", lu4d::brain::BrainAskConfig::new(3, 10));
    for a in &recall.recall.activated {
        assert!(!a.content.starts_with("goal:"), "leaked think goal/step node into a spread: {}", a.content);
    }
    assert_eq!(brain.status().bond_count, b0, "a spread after read-only reasoning must not have materialised bonds");

    // (D) MEANINGFULNESS CONTRAST: the same think WITHOUT no_write DOES write,
    // proving the graph is writable and the no_write assertions are not vacuous.
    let cfg_w = lu4d::thinking::ThinkConfig { no_write: false, max_wall_clock: None, ..Default::default() };
    let _ = lu4d::thinking::think(&mut brain, "science", cfg_w);
    assert!(
        brain.status().bond_count > b0,
        "write-mode think() must add bonds (else the no_write check is vacuous): b0={}, now={}",
        b0,
        brain.status().bond_count
    );
}
// OP-REASON-LOOP Phase 3 -- the reason<->retrieve loop reaches 2nd-hop nodes
// that a single shallow pass misses (two-pass recall), stays within the
// ContextSet read budget, and writes NO nodes or bonds. Uses an ephemeral heap
// Brain (reliable recall; heap bond_count reflects writes directly).
#[test]
fn reason_loop_two_pass_reaches_second_hop_without_writes() {
    use lu4d::bond::BondType;
    use lu4d::node::Confidence;
    use lu4d::Brain;

    let mut brain = Brain::new();
    // sky --HasProperty--> blue --CausedBy--> scattering --RelatedTo--> wavelength
    // (scattering also -> rayleigh; wavelength -> light). Pass 1 at depth 1 from
    // {sky, blue} reaches scattering (1 hop from blue) but NOT wavelength (2
    // hops). The loop reasons, picks a reasoning-derived topic seed, and
    // re-retrieves to reach the 2nd hop.
    brain.teach("sky", BondType::HasProperty, "blue", 0.9, Confidence::Verified);
    brain.teach("blue", BondType::RelatedTo, "scattering", 0.9, Confidence::Verified);
    brain.teach("scattering", BondType::RelatedTo, "wavelength", 0.9, Confidence::Verified);
    brain.teach("scattering", BondType::RelatedTo, "rayleigh", 0.9, Confidence::Verified);
    brain.teach("wavelength", BondType::RelatedTo, "light", 0.9, Confidence::Verified);

    let b0 = brain.status().bond_count;
    let n0 = brain.status().node_count;

    // Pass 1: shallow recall (depth 1) -- should NOT reach the 2nd hop.
    let ask = brain.ask_with_config("why is the sky blue", lu4d::brain::BrainAskConfig::new(1, 10));
    let mut cs = lu4d::reason_loop::ContextSet::from_ask(&ask, 0, 0, lu4d::reason_loop::DEFAULT_READ_BUDGET);
    let reached_before = cs.contains_label("wavelength");

    // (Gate 1) Degrade: iters=1 -> loop is a no-op, no enrichment.
    let mut cfg1 = lu4d::narration::NarrationConfig::default();
    cfg1.max_context_iters = 1;
    let s1 = lu4d::reason_loop::run_reason_loop(&mut brain, "why is the sky blue", &ask, &mut cs, 2, 10, &cfg1);
    assert_eq!(s1.iters_ran, 0, "iters=1 must not run the loop");
    assert_eq!(cs.contains_label("wavelength"), reached_before, "iters=1 must not enrich the context");

    // (Gate 2) Two-pass: iters=2 -> reason, re-seed, reach the 2nd hop.
    let mut cfg2 = lu4d::narration::NarrationConfig::default();
    cfg2.max_context_iters = 2;
    let s2 = lu4d::reason_loop::run_reason_loop(&mut brain, "why is the sky blue", &ask, &mut cs, 2, 10, &cfg2);

    // (Gate 5) No node/bond writes across the whole loop.
    assert_eq!(brain.status().bond_count, b0, "reason loop wrote bonds");
    assert_eq!(brain.status().node_count, n0, "reason loop wrote nodes");

    // (Gate 2) two-pass reached the 2nd-hop node the shallow pass 1 missed.
    assert!(!reached_before, "pass 1 (depth 1) should not have reached wavelength");
    assert!(s2.iters_ran >= 1, "iters=2 must run at least one retrieval iteration");
    assert!(cs.contains_label("wavelength"), "two-pass loop must reach the 2nd-hop wavelength via re-retrieval");
    assert!(!s2.seeds_used.is_empty(), "loop should have chosen reasoning-derived re-seed concepts");

    // (Gate 3) read-amplification bounded: held nodes within the ContextSet budget.
    assert!(cs.len() <= lu4d::reason_loop::DEFAULT_READ_BUDGET, "ContextSet exceeded its read budget");
}
// OP-REASON-LOOP Phase 4 -- termination + exit idk floor. A well-anchored
// query grounds at pass 1 and the loop stops immediately (Grounded, no
// iterations burned); an ungroundable query does NOT clear the exit floor
// (classify_context != Pass -> narrate would refuse); and none of it writes
// nodes or bonds. Ephemeral heap Brain (reliable recall + heap counts).
#[test]
fn reason_loop_termination_grounded_and_exit_floor() {
    use lu4d::bond::BondType;
    use lu4d::idk_gate::IdkVerdict;
    use lu4d::node::Confidence;
    use lu4d::reason_loop::{classify_context, run_reason_loop, ContextSet, TerminationReason, DEFAULT_READ_BUDGET};
    use lu4d::Brain;

    let mut brain = Brain::new();
    brain.teach("photosynthesis", BondType::RelatedTo, "chlorophyll", 0.9, Confidence::Verified);
    brain.teach("photosynthesis", BondType::RelatedTo, "sunlight", 0.9, Confidence::Verified);
    brain.teach("photosynthesis", BondType::RelatedTo, "carbon dioxide", 0.9, Confidence::Verified);
    brain.teach("chlorophyll", BondType::RelatedTo, "chloroplast", 0.9, Confidence::Verified);
    brain.teach("sunlight", BondType::RelatedTo, "energy", 0.9, Confidence::Verified);

    let b0 = brain.status().bond_count;
    let n0 = brain.status().node_count;

    let mut cfg = lu4d::narration::NarrationConfig::default();
    cfg.max_context_iters = 3;

    // (Grounded early-stop) A well-anchored query grounds at pass 1 -> the loop
    // stops on Grounded and burns ZERO iterations.
    let ask_g = brain.ask_with_config("photosynthesis", lu4d::brain::BrainAskConfig::new(3, 10));
    let mut cs_g = ContextSet::from_ask(&ask_g, 0, 0, DEFAULT_READ_BUDGET);
    assert_eq!(
        classify_context(&mut brain, "photosynthesis", &ask_g, &cs_g).0,
        IdkVerdict::Pass,
        "a well-anchored query should ground at pass 1"
    );
    let sg = run_reason_loop(&mut brain, "photosynthesis", &ask_g, &mut cs_g, 3, 10, &cfg);
    assert_eq!(sg.terminated, TerminationReason::Grounded, "grounded query must stop on Grounded");
    assert!(
        sg.iters_ran < cfg.max_context_iters - 1,
        "grounded query must stop EARLY (before the iteration budget): iters_ran={}",
        sg.iters_ran
    );

    // (Exit floor) An ungroundable query: tokens with no brain coverage ->
    // classify_context refuses (non-Pass) -> narrate refuses at the exit gate.
    let uq = "florble wimzat quxzy";
    let ask_u = brain.ask_with_config(uq, lu4d::brain::BrainAskConfig::new(3, 10));
    let cs_u = ContextSet::from_ask(&ask_u, 0, 0, DEFAULT_READ_BUDGET);
    assert_ne!(
        classify_context(&mut brain, uq, &ask_u, &cs_u).0,
        IdkVerdict::Pass,
        "an ungroundable query must NOT clear the exit floor"
    );

    // (No writes) the loop + exit-floor checks wrote no nodes or bonds.
    assert_eq!(brain.status().bond_count, b0, "phase 4 wrote bonds");
    assert_eq!(brain.status().node_count, n0, "phase 4 wrote nodes");
}
