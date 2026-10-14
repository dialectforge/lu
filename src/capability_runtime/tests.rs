//! Port Cat.5.1 — capability-pack EXECUTION runtime tests.
//!
//! These are REAL end-to-end tests: they build signed `.lupack5`
//! files, write them to disk, run an actual subprocess pack through
//! the registry, and assert on the security model. Nothing is mocked.
//!
//! Validation coverage (mirrors the four required Cat.5.1 checks):
//!   1. Load a signed pack from a trusted publisher -> it registers
//!      its tool(s).
//!   2. Permission/trust enforcement: an action outside granted perms
//!      is DENIED; an untrusted/unsigned pack is refused at load; a
//!      tampered pack fails re-verify-on-execute.
//!   3. Execute an ALLOWED action via Subprocess -> correct result;
//!      resource caps (wall/output/cpu) honored.
//!   4. Orthogonality: bond_count unchanged by install/load/call;
//!      a brain_write-granted pack mutates ONLY via PendingLog.

use super::*;

use std::path::Path;
use std::sync::Arc;

use crate::capability_pack::{
    build_capability_pack, hex_encode, read_capability_pack_file, signing_key_from_bytes,
    write_capability_pack_file, CapabilityPack, CapabilityPackBuildSpec,
    CapabilityPackLedgerStore, CapabilityPackManifestSeed, PackPermissions, PackRuntimeKind,
    PackToolDecl,
};

fn clock() -> Arc<dyn crate::time::ClockSource> {
    Arc::new(crate::time::SystemClockSource::new()) as Arc<dyn crate::time::ClockSource>
}

fn det_key(seed: u8) -> ed25519_dalek::SigningKey {
    let mut bytes = [0u8; 32];
    for (i, b) in bytes.iter_mut().enumerate() {
        *b = (i as u8).wrapping_mul(31).wrapping_add(seed);
    }
    signing_key_from_bytes(&bytes)
}

fn one_tool(name: &str) -> Vec<PackToolDecl> {
    vec![PackToolDecl {
        tool_name: name.into(),
        description: format!("test tool {}", name),
        args_schema: serde_json::json!({"type": "object"}),
    }]
}

fn make_pack(
    sk: &ed25519_dalek::SigningKey,
    pack_id: &str,
    tools: Vec<PackToolDecl>,
    perms: PackPermissions,
    script: &str,
) -> CapabilityPack {
    let seed = CapabilityPackManifestSeed {
        pack_id: pack_id.into(),
        pack_version: "0.1.0".into(),
        runtime_kind: PackRuntimeKind::Subprocess,
        tools,
        permissions: perms,
        description: "cat5.1 test pack".into(),
    };
    build_capability_pack(CapabilityPackBuildSpec {
        manifest_seed: seed,
        content: script.as_bytes().to_vec(),
        signing_key: sk,
        now_ms: 1_700_000_000_000,
    })
    .expect("build pack")
}

fn write_pack(dir: &Path, fname: &str, pack: &CapabilityPack) -> String {
    let p = dir.join(fname);
    write_capability_pack_file(pack, &p).expect("write pack file");
    p.to_string_lossy().into_owned()
}

fn trusted_ledger(sk: &ed25519_dalek::SigningKey) -> CapabilityPackLedgerStore {
    let mut l = CapabilityPackLedgerStore::new();
    let pk = hex_encode(sk.verifying_key().as_bytes());
    l.add_publisher(&pk, "Test Publisher", None, 1_000).unwrap();
    l
}

fn enabled_registry() -> LoadedPackRegistry {
    let mut r = LoadedPackRegistry::new();
    r.set_enabled(true);
    r
}

// Scripts implementing the host<->pack ABI. Single-quoted printf keeps
// the JSON literal intact without shell escaping.
const SCRIPT_ECHO: &str = r#"#!/bin/sh
req=$(cat)
case "$req" in
  *NEEDLE*) printf '{"ok":true,"result":{"found":true,"echo":"pong"}}' ;;
  *)        printf '{"ok":true,"result":{"found":false,"echo":"pong"}}' ;;
esac
"#;

const SCRIPT_WRITER: &str = r#"#!/bin/sh
cat >/dev/null
printf '{"ok":true,"result":{"proposed":1},"brain_writes":[{"subject":"alpha","object":"beta","bond_type":"RelatedTo","weight":0.5}]}'
"#;

const SCRIPT_SLOW: &str = r#"#!/bin/sh
cat >/dev/null
sleep 5
printf '{"ok":true,"result":{}}'
"#;

const SCRIPT_BIG: &str = r#"#!/bin/sh
cat >/dev/null
head -c 5000000 /dev/zero | tr '\0' 'A'
"#;

const SCRIPT_CPU: &str = r#"#!/bin/sh
cat >/dev/null
awk 'BEGIN{x=0;while(1){x=x+1}}'
"#;

const SCRIPT_SCENE: &str = r#"#!/bin/sh
cat >/dev/null
printf '{"ok":true,"result":{"triplets":[{"subject":"dog","predicate":"has","object":"tail","confidence":0.9}]}}'
"#;

// ---------------------------------------------------------------------
// VALIDATION 1 — load a signed trusted pack -> it registers its tools
// ---------------------------------------------------------------------

#[test]
fn v1_load_trusted_pack_registers_tools() {
    let sk = det_key(1);
    let ledger = trusted_ledger(&sk);
    let dir = tempfile::tempdir().unwrap();
    let pack = make_pack(&sk, "df/echo", one_tool("echo"), PackPermissions::default(), SCRIPT_ECHO);
    let src = write_pack(dir.path(), "echo.lupack5", &pack);
    let (pack, bytes) = read_capability_pack_file(&src).unwrap();

    let mut reg = enabled_registry();
    let summary = reg.load(&pack, &bytes, &src, &ledger, clock()).expect("load");

    let nsname = namespaced_tool_name("df/echo", "echo");
    assert_eq!(summary.tool_names, vec![nsname.clone()]);
    assert!(reg.has_tool(&nsname), "tool must be registered");
    assert!(reg.tool_decls().iter().any(|d| d["name"] == nsname));
    assert_eq!(reg.loaded_pack_ids(), vec!["df/echo".to_string()]);
}

#[test]
fn v1_disabled_registry_refuses_load() {
    let sk = det_key(1);
    let ledger = trusted_ledger(&sk);
    let dir = tempfile::tempdir().unwrap();
    let pack = make_pack(&sk, "df/echo", one_tool("echo"), PackPermissions::default(), SCRIPT_ECHO);
    let src = write_pack(dir.path(), "echo.lupack5", &pack);
    let (pack, bytes) = read_capability_pack_file(&src).unwrap();

    let mut reg = LoadedPackRegistry::new(); // disabled by default
    let err = reg.load(&pack, &bytes, &src, &ledger, clock()).unwrap_err();
    assert!(matches!(err, PackRuntimeError::Disabled));
}

// ---------------------------------------------------------------------
// VALIDATION 2 — permission / trust enforcement
// ---------------------------------------------------------------------

#[test]
fn v2_untrusted_publisher_refused_at_load() {
    let trusted = det_key(1);
    let attacker = det_key(99);
    let ledger = trusted_ledger(&trusted); // attacker NOT trusted
    let dir = tempfile::tempdir().unwrap();
    let pack = make_pack(&attacker, "evil/pack", one_tool("x"), PackPermissions::default(), SCRIPT_ECHO);
    let src = write_pack(dir.path(), "evil.lupack5", &pack);
    let (pack, bytes) = read_capability_pack_file(&src).unwrap();

    let mut reg = enabled_registry();
    let err = reg.load(&pack, &bytes, &src, &ledger, clock()).unwrap_err();
    match err {
        PackRuntimeError::ReVerify(m) => assert!(m.contains("untrusted"), "got: {}", m),
        other => panic!("expected ReVerify(untrusted), got {:?}", other),
    }
}

#[test]
fn v2_tampered_pack_refused_at_load() {
    let sk = det_key(1);
    let ledger = trusted_ledger(&sk);
    let dir = tempfile::tempdir().unwrap();
    let pack = make_pack(&sk, "df/echo", one_tool("echo"), PackPermissions::default(), SCRIPT_ECHO);
    let src = write_pack(dir.path(), "echo.lupack5", &pack);

    // Flip a content byte on disk (corrupts the signed payload).
    let mut raw = std::fs::read(&src).unwrap();
    let idx = raw.len() - crate::capability_pack::PACK5_SIG_LEN - 1;
    raw[idx] ^= 0xFF;
    std::fs::write(&src, &raw).unwrap();

    let (pack, bytes) = read_capability_pack_file(&src).unwrap();
    let mut reg = enabled_registry();
    let err = reg.load(&pack, &bytes, &src, &ledger, clock()).unwrap_err();
    assert!(matches!(err, PackRuntimeError::ReVerify(_)), "got {:?}", err);
}

#[test]
fn v2_tampered_after_load_fails_reverify_on_execute() {
    let sk = det_key(1);
    let ledger = trusted_ledger(&sk);
    let dir = tempfile::tempdir().unwrap();
    let pack = make_pack(&sk, "df/echo", one_tool("echo"), PackPermissions::default(), SCRIPT_ECHO);
    let src = write_pack(dir.path(), "echo.lupack5", &pack);
    let (pack, bytes) = read_capability_pack_file(&src).unwrap();

    let mut reg = enabled_registry();
    reg.load(&pack, &bytes, &src, &ledger, clock()).expect("load");
    let nsname = namespaced_tool_name("df/echo", "echo");

    // Sanity: works before tampering.
    reg.execute(&nsname, &serde_json::json!({"q": "NEEDLE"}), &ledger)
        .expect("pre-tamper execute");

    // Now tamper the on-disk file AFTER load.
    let mut raw = std::fs::read(&src).unwrap();
    let idx = raw.len() - crate::capability_pack::PACK5_SIG_LEN - 1;
    raw[idx] ^= 0xFF;
    std::fs::write(&src, &raw).unwrap();

    let err = reg
        .execute(&nsname, &serde_json::json!({"q": "NEEDLE"}), &ledger)
        .unwrap_err();
    assert!(matches!(err, PackRuntimeError::ReVerify(_)), "got {:?}", err);
}

#[test]
fn v2_revoked_after_load_fails_reverify_on_execute() {
    let sk = det_key(1);
    let mut ledger = trusted_ledger(&sk);
    let dir = tempfile::tempdir().unwrap();
    let pack = make_pack(&sk, "df/echo", one_tool("echo"), PackPermissions::default(), SCRIPT_ECHO);
    let src = write_pack(dir.path(), "echo.lupack5", &pack);
    let (pack, bytes) = read_capability_pack_file(&src).unwrap();

    let mut reg = enabled_registry();
    reg.load(&pack, &bytes, &src, &ledger, clock()).expect("load");
    let nsname = namespaced_tool_name("df/echo", "echo");

    ledger.revoke("df/echo"); // operator revokes after load
    let err = reg
        .execute(&nsname, &serde_json::json!({}), &ledger)
        .unwrap_err();
    match err {
        PackRuntimeError::ReVerify(m) => assert!(m.contains("revoked"), "got {}", m),
        other => panic!("expected ReVerify(revoked), got {:?}", other),
    }
}

#[test]
fn v2_unsandboxable_permission_refused_at_load() {
    let sk = det_key(1);
    let ledger = trusted_ledger(&sk);
    let dir = tempfile::tempdir().unwrap();
    let perms = PackPermissions {
        network_egress: vec!["api.example.com".into()],
        ..Default::default()
    };
    let pack = make_pack(&sk, "df/net", one_tool("x"), perms, SCRIPT_ECHO);
    let src = write_pack(dir.path(), "net.lupack5", &pack);
    let (pack, bytes) = read_capability_pack_file(&src).unwrap();

    let mut reg = enabled_registry(); // allow_unsandboxed = false
    let err = reg.load(&pack, &bytes, &src, &ledger, clock()).unwrap_err();
    match err {
        PackRuntimeError::PermissionUnsupported(p) => assert_eq!(p, "network_egress"),
        other => panic!("expected PermissionUnsupported, got {:?}", other),
    }
}

#[test]
fn v2_wasm_runtime_refused_sharedlib_rejected() {
    // build_pack_tool must reject Wasm (deferred) and SharedLib (rejected).
    let r = build_pack_tool(
        PackRuntimeKind::Wasm,
        "pack__x__y".into(),
        one_tool("y").remove(0),
        PackPermissions::default(),
        ResourceCaps::default(),
        std::path::PathBuf::from("/nonexistent"),
        clock(),
    );
    assert!(matches!(r, Err(PackRuntimeError::RuntimeUnsupported(PackRuntimeKind::Wasm))));

    let r = build_pack_tool(
        PackRuntimeKind::SharedLib,
        "pack__x__y".into(),
        one_tool("y").remove(0),
        PackPermissions::default(),
        ResourceCaps::default(),
        std::path::PathBuf::from("/nonexistent"),
        clock(),
    );
    assert!(matches!(r, Err(PackRuntimeError::RuntimeUnsupported(PackRuntimeKind::SharedLib))));
}

// ---------------------------------------------------------------------
// VALIDATION 3 — execute allowed action; resource caps honored
// ---------------------------------------------------------------------

#[test]
fn v3_allowed_execute_returns_correct_result() {
    let sk = det_key(1);
    let ledger = trusted_ledger(&sk);
    let dir = tempfile::tempdir().unwrap();
    let pack = make_pack(&sk, "df/echo", one_tool("echo"), PackPermissions::default(), SCRIPT_ECHO);
    let src = write_pack(dir.path(), "echo.lupack5", &pack);
    let (pack, bytes) = read_capability_pack_file(&src).unwrap();

    let mut reg = enabled_registry();
    reg.load(&pack, &bytes, &src, &ledger, clock()).expect("load");
    let nsname = namespaced_tool_name("df/echo", "echo");

    // args carrying the NEEDLE prove args reach the subprocess.
    let out = reg
        .execute(&nsname, &serde_json::json!({"q": "has NEEDLE inside"}), &ledger)
        .expect("execute");
    assert_eq!(out.result["found"], serde_json::json!(true));
    assert_eq!(out.result["echo"], serde_json::json!("pong"));
    assert!(out.brain_writes.is_empty());

    // Without the needle the branch flips -> proves real arg passing.
    let out2 = reg
        .execute(&nsname, &serde_json::json!({"q": "nothing here"}), &ledger)
        .expect("execute2");
    assert_eq!(out2.result["found"], serde_json::json!(false));
}

#[test]
fn v3_wall_clock_cap_honored() {
    let sk = det_key(1);
    let ledger = trusted_ledger(&sk);
    let dir = tempfile::tempdir().unwrap();
    let pack = make_pack(&sk, "df/slow", one_tool("slow"), PackPermissions::default(), SCRIPT_SLOW);
    let src = write_pack(dir.path(), "slow.lupack5", &pack);
    let (pack, bytes) = read_capability_pack_file(&src).unwrap();

    let mut reg = enabled_registry();
    reg.set_caps(ResourceCaps { wall_ms: 300, ..Default::default() });
    reg.load(&pack, &bytes, &src, &ledger, clock()).expect("load");
    let nsname = namespaced_tool_name("df/slow", "slow");

    let err = reg.execute(&nsname, &serde_json::json!({}), &ledger).unwrap_err();
    match err {
        PackRuntimeError::Timeout { ms } => assert_eq!(ms, 300),
        other => panic!("expected Timeout, got {:?}", other),
    }
}

#[test]
fn v3_output_size_cap_honored() {
    let sk = det_key(1);
    let ledger = trusted_ledger(&sk);
    let dir = tempfile::tempdir().unwrap();
    let pack = make_pack(&sk, "df/big", one_tool("big"), PackPermissions::default(), SCRIPT_BIG);
    let src = write_pack(dir.path(), "big.lupack5", &pack);
    let (pack, bytes) = read_capability_pack_file(&src).unwrap();

    let mut reg = enabled_registry();
    reg.set_caps(ResourceCaps { output_bytes: 64 * 1024, wall_ms: 10_000, ..Default::default() });
    reg.load(&pack, &bytes, &src, &ledger, clock()).expect("load");
    let nsname = namespaced_tool_name("df/big", "big");

    let err = reg.execute(&nsname, &serde_json::json!({}), &ledger).unwrap_err();
    match err {
        PackRuntimeError::OutputTooLarge { max } => assert_eq!(max, 64 * 1024),
        other => panic!("expected OutputTooLarge, got {:?}", other),
    }
}

#[test]
fn v3_cpu_cap_honored() {
    let sk = det_key(1);
    let ledger = trusted_ledger(&sk);
    let dir = tempfile::tempdir().unwrap();
    let pack = make_pack(&sk, "df/cpu", one_tool("cpu"), PackPermissions::default(), SCRIPT_CPU);
    let src = write_pack(dir.path(), "cpu.lupack5", &pack);
    let (pack, bytes) = read_capability_pack_file(&src).unwrap();

    let mut reg = enabled_registry();
    // 1s CPU budget, generous 30s wall -> CPU limit (SIGXCPU) fires
    // first, killing the busy loop well before the wall deadline.
    reg.set_caps(ResourceCaps { cpu_ms: 1_000, wall_ms: 30_000, ..Default::default() });
    reg.load(&pack, &bytes, &src, &ledger, clock()).expect("load");
    let nsname = namespaced_tool_name("df/cpu", "cpu");

    let err = reg.execute(&nsname, &serde_json::json!({}), &ledger).unwrap_err();
    // Killed by SIGXCPU -> non-success exit (NOT a wall Timeout).
    assert!(
        !matches!(err, PackRuntimeError::Timeout { .. }),
        "cpu cap should fire before wall timeout, got {:?}",
        err
    );
    assert!(matches!(err, PackRuntimeError::Exit { .. }), "got {:?}", err);
}

// ---------------------------------------------------------------------
// VALIDATION 4 — orthogonality + brain_write only via PendingLog
// ---------------------------------------------------------------------

fn load_into_brain(
    brain: &mut crate::brain::Brain,
    sk: &ed25519_dalek::SigningKey,
    pack_id: &str,
    perms: PackPermissions,
    script: &str,
    dir: &Path,
    tool: &str,
) -> String {
    let pk = hex_encode(sk.verifying_key().as_bytes());
    brain.cap_pack_ledger.add_publisher(&pk, "Test", None, 1_000).unwrap();
    brain.pack_registry.set_enabled(true);
    let pack = make_pack(sk, pack_id, one_tool(tool), perms, script);
    let src = write_pack(dir, &format!("{}.lupack5", tool), &pack);
    let (pack, bytes) = read_capability_pack_file(&src).unwrap();
    let clk = brain.clock.clone();
    brain
        .pack_registry
        .load(&pack, &bytes, &src, &brain.cap_pack_ledger, clk)
        .expect("load into brain");
    namespaced_tool_name(pack_id, tool)
}

#[test]
fn v4_orthogonality_call_leaves_bond_count_unchanged() {
    let mut brain = crate::brain::Brain::new();
    let sk = det_key(1);
    let dir = tempfile::tempdir().unwrap();
    let bonds_before = brain.network.bond_count();
    let pending_before = brain.pending.total_count();

    let nsname = load_into_brain(
        &mut brain, &sk, "df/echo", PackPermissions::default(), SCRIPT_ECHO, dir.path(), "echo",
    );
    // load left bond_count unchanged
    assert_eq!(brain.network.bond_count(), bonds_before, "load must not change bonds");

    let resp = crate::mcp_cap_runtime::dispatch_pack_tool(
        &mut brain,
        &nsname,
        serde_json::json!({"q": "NEEDLE"}),
    )
    .expect("dispatch");
    assert_eq!(resp["structuredContent"]["result"]["found"], serde_json::json!(true));

    // call left BOTH bond_count and pending unchanged (no brain_write).
    assert_eq!(brain.network.bond_count(), bonds_before, "call must not change bonds");
    assert_eq!(brain.pending.total_count(), pending_before, "non-writer must not queue pending");
}

#[test]
fn v4_brain_write_denied_without_grant() {
    let mut brain = crate::brain::Brain::new();
    let sk = det_key(1);
    let dir = tempfile::tempdir().unwrap();

    // writer script, but brain_write NOT granted
    let nsname = load_into_brain(
        &mut brain, &sk, "df/writer", PackPermissions::default(), SCRIPT_WRITER, dir.path(), "writer",
    );
    let bonds_before = brain.network.bond_count();
    let pending_before = brain.pending.total_count();

    let err = crate::mcp_cap_runtime::dispatch_pack_tool(
        &mut brain,
        &nsname,
        serde_json::json!({}),
    )
    .unwrap_err();
    assert!(err.1.contains("permission denied"), "got {:?}", err);
    // Nothing leaked through.
    assert_eq!(brain.network.bond_count(), bonds_before);
    assert_eq!(brain.pending.total_count(), pending_before);
}

#[test]
fn v4_brain_write_granted_routes_via_pendinglog_not_bonds() {
    let mut brain = crate::brain::Brain::new();
    let sk = det_key(1);
    let dir = tempfile::tempdir().unwrap();

    let perms = PackPermissions { brain_write: true, ..Default::default() };
    let nsname = load_into_brain(
        &mut brain, &sk, "df/writer", perms, SCRIPT_WRITER, dir.path(), "writer",
    );
    let bonds_before = brain.network.bond_count();
    let pending_before = brain.pending.total_count();

    let resp = crate::mcp_cap_runtime::dispatch_pack_tool(
        &mut brain,
        &nsname,
        serde_json::json!({}),
    )
    .expect("dispatch writer");
    assert_eq!(resp["structuredContent"]["queued_brain_writes"], serde_json::json!(1));

    // The exception case: a granted brain_write mutates ONLY via the
    // PendingLog (status Pending). bond_count is STILL unchanged.
    assert_eq!(
        brain.network.bond_count(),
        bonds_before,
        "granted brain_write must NOT touch bonds directly"
    );
    assert_eq!(
        brain.pending.total_count(),
        pending_before + 1,
        "granted brain_write must queue exactly one PendingChange"
    );
    assert_eq!(brain.pending.pending_count(), 1, "the queued change is Pending");
}

// ---------------------------------------------------------------------
// Vision extractor hook — generalised SceneGraphExtractor path
// ---------------------------------------------------------------------

#[test]
fn vision_capability_pack_extractor_runs() {
    use crate::vision::extractor::{ExtractHints, SceneGraphExtractor};
    let sk = det_key(1);
    let ledger = trusted_ledger(&sk);
    let dir = tempfile::tempdir().unwrap();
    let pack = make_pack(
        &sk,
        "df/vision",
        one_tool(CapabilityPackExtractor::EXTRACTOR_TOOL),
        PackPermissions::default(),
        SCRIPT_SCENE,
    );
    let src = write_pack(dir.path(), "vision.lupack5", &pack);
    let (pack, bytes) = read_capability_pack_file(&src).unwrap();

    let mut reg = enabled_registry();
    reg.load(&pack, &bytes, &src, &ledger, clock()).expect("load");
    reg.reverify("df/vision", &ledger).expect("reverify");
    let nsname = namespaced_tool_name("df/vision", CapabilityPackExtractor::EXTRACTOR_TOOL);
    let tool = reg.get_tool(&nsname).expect("tool");

    let ex = CapabilityPackExtractor::new("df/vision".into(), tool);
    assert_eq!(ex.name(), "capability_pack:df/vision");
    let triplets = ex
        .extract(b"\x89PNG\r\n\x1a\n", ExtractHints::default())
        .expect("extract");
    assert_eq!(triplets.len(), 1);
    assert_eq!(triplets[0].subject, "dog");
    assert_eq!(triplets[0].predicate, "has");
    assert_eq!(triplets[0].object, "tail");
}

// ---------------------------------------------------------------------
// unload wipes tools
// ---------------------------------------------------------------------

#[test]
fn unload_removes_tools() {
    let sk = det_key(1);
    let ledger = trusted_ledger(&sk);
    let dir = tempfile::tempdir().unwrap();
    let pack = make_pack(&sk, "df/echo", one_tool("echo"), PackPermissions::default(), SCRIPT_ECHO);
    let src = write_pack(dir.path(), "echo.lupack5", &pack);
    let (pack, bytes) = read_capability_pack_file(&src).unwrap();

    let mut reg = enabled_registry();
    reg.load(&pack, &bytes, &src, &ledger, clock()).expect("load");
    let nsname = namespaced_tool_name("df/echo", "echo");
    assert!(reg.has_tool(&nsname));
    let n = reg.unload("df/echo").expect("unload");
    assert_eq!(n, 1);
    assert!(!reg.has_tool(&nsname));
    assert!(reg.loaded_pack_ids().is_empty());
}
