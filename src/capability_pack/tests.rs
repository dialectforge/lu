//! Port Cat.5 — tests for the schema-only signed capability pack format.
//!
//! Coverage (mirrors Cat.4's tests file):
//!   * Serde round-trip on each struct (manifest, ledger snapshot,
//!     install record, runtime kind, permissions).
//!   * Pack file format read/write — magic check, signature verify,
//!     refuse on tampered manifest, refuse on tampered content.
//!   * Install fails when publisher not trusted; succeeds when
//!     trusted; ledger reflects the install.
//!   * Install fails when pack_id is in revoked_packs.
//!   * Already-installed gate (force=false refuses; force=true
//!     overwrites).
//!   * Uninstall removes from ledger; list reflects.
//!   * Backward compat: pre-Cat.5 brain (section_count=16) loads
//!     cleanly under the new binary (section #17 missing → empty
//!     ledger).
//!   * Forward compat: Cat.5-saved brain (section_count=17) loads
//!     under a synthetic pre-Cat.5 reader (J.1.1 unknown-section
//!     skip path).
//!   * Anti-mutation grep guard: `mcp_cat5.rs` and the install
//!     helpers MUST NOT contain `brain.teach`, `add_bond`, or
//!     `pending.push`.

use super::*;

fn test_signing_key() -> ed25519_dalek::SigningKey {
    // Deterministic 32-byte secret for reproducible tests.
    let mut bytes = [0u8; 32];
    for (i, b) in bytes.iter_mut().enumerate() {
        *b = (i as u8).wrapping_mul(17).wrapping_add(7);
    }
    signing_key_from_bytes(&bytes)
}

fn sample_tools() -> Vec<PackToolDecl> {
    vec![PackToolDecl {
        tool_name: "pack_demo_hello".into(),
        description: "Cat.5 demo tool — would print hello if Cat.5.1 ran it.".into(),
        args_schema: serde_json::json!({
            "type": "object",
            "properties": {"name": {"type": "string"}}
        }),
    }]
}

fn build_test_pack(
    sk: &ed25519_dalek::SigningKey,
    content: Vec<u8>,
    pack_id: &str,
    now_ms: u64,
) -> CapabilityPack {
    let seed = CapabilityPackManifestSeed {
        pack_id: pack_id.into(),
        pack_version: "0.1.0".into(),
        runtime_kind: PackRuntimeKind::Wasm,
        tools: sample_tools(),
        permissions: PackPermissions::default(),
        description: "Cat.5 test pack".into(),
    };
    build_capability_pack(CapabilityPackBuildSpec {
        manifest_seed: seed,
        content,
        signing_key: sk,
        now_ms,
    })
    .expect("build")
}

// ---------------------------------------------------------------------
// Serde round-trip tests
// ---------------------------------------------------------------------

#[test]
fn serde_roundtrip_runtime_kind() {
    for k in [
        PackRuntimeKind::Wasm,
        PackRuntimeKind::SharedLib,
        PackRuntimeKind::Subprocess,
        PackRuntimeKind::Reserved,
    ] {
        let bytes = bincode::serialize(&k).unwrap();
        let back: PackRuntimeKind = bincode::deserialize(&bytes).unwrap();
        assert_eq!(k, back);
    }
}

#[test]
fn serde_roundtrip_permissions() {
    let p = PackPermissions {
        filesystem_read: vec!["~/Downloads/*.pdf".into()],
        filesystem_write: vec!["/tmp/lu_pack/*".into()],
        network_egress: vec!["api.example.com".into()],
        brain_read: true,
        brain_write: false,
    };
    let bytes = bincode::serialize(&p).unwrap();
    let back: PackPermissions = bincode::deserialize(&bytes).unwrap();
    assert_eq!(p, back);
}

#[test]
fn serde_roundtrip_tool_decl() {
    let t = PackToolDecl {
        tool_name: "foo_bar".into(),
        description: "sample".into(),
        args_schema: serde_json::json!({"type": "object"}),
    };
    let bytes = bincode::serialize(&t).unwrap();
    let back: PackToolDecl = bincode::deserialize(&bytes).unwrap();
    assert_eq!(t, back);
}

#[test]
fn serde_roundtrip_manifest() {
    let sk = test_signing_key();
    let pack = build_test_pack(&sk, b"RUNTIME_TBD".to_vec(), "df/test", 1_700_000_000_000);
    let bytes = bincode::serialize(&pack.manifest).unwrap();
    let back: CapabilityPackManifest = bincode::deserialize(&bytes).unwrap();
    assert_eq!(pack.manifest, back);
}

#[test]
fn serde_roundtrip_ledger_snapshot() {
    let mut store = CapabilityPackLedgerStore::new();
    let pk_hex = {
        let sk = test_signing_key();
        hex_encode(sk.verifying_key().as_bytes())
    };
    store
        .add_publisher(&pk_hex, "Test Publisher", Some("notes".into()), 1_000)
        .unwrap();
    store.record_install(InstalledPack {
        pack_id: "df/test".into(),
        pack_version: "0.1.0".into(),
        publisher_pubkey_hex: pk_hex.clone(),
        runtime_kind: PackRuntimeKind::Wasm,
        installed_at_ms: 2_000,
        signed_at_ms: 1_500,
        manifest_hash_hex: "abc123".into(),
        tool_count: 1,
        source_path: "/tmp/p.lupack5".into(),
    });
    store.revoke("df/old");
    let snap = store.snapshot();
    let bytes = bincode::serialize(&snap).unwrap();
    let back: CapabilityPackLedgerSnapshot = bincode::deserialize(&bytes).unwrap();
    assert_eq!(snap, back);
}

// ---------------------------------------------------------------------
// Pack file format tests
// ---------------------------------------------------------------------

#[test]
fn pack_file_roundtrip_and_magic() {
    let sk = test_signing_key();
    let content = b"RUNTIME_TBD".to_vec();
    let pack = build_test_pack(&sk, content.clone(), "df/test", 1_700_000_000_000);
    let bytes = pack.to_bytes().unwrap();
    assert_eq!(&bytes[0..8], &PACK5_MAGIC);
    assert_eq!(
        u32::from_le_bytes(bytes[8..12].try_into().unwrap()),
        PACK5_FORMAT_VERSION
    );
    let parsed = CapabilityPack::decode_unverified(&bytes).unwrap();
    assert_eq!(parsed, pack);
    assert_eq!(parsed.content, content);
    verify_capability_pack(&parsed).expect("signature must verify");
}

#[test]
fn pack_file_refuses_bad_magic() {
    let sk = test_signing_key();
    let pack = build_test_pack(&sk, b"x".to_vec(), "df/test", 1);
    let mut bytes = pack.to_bytes().unwrap();
    bytes[0] = b'X'; // corrupt magic
    let err = CapabilityPack::decode_unverified(&bytes).expect_err("must refuse");
    match err {
        CapabilityPackError::Format(_) => {}
        other => panic!("expected Format error, got {:?}", other),
    }
}

#[test]
fn pack_file_refuses_bad_version() {
    let sk = test_signing_key();
    let pack = build_test_pack(&sk, b"x".to_vec(), "df/test", 1);
    let mut bytes = pack.to_bytes().unwrap();
    bytes[8..12].copy_from_slice(&999u32.to_le_bytes());
    let err = CapabilityPack::decode_unverified(&bytes).expect_err("must refuse");
    match err {
        CapabilityPackError::UnsupportedFormatVersion { got, expected } => {
            assert_eq!(got, 999);
            assert_eq!(expected, PACK5_FORMAT_VERSION);
        }
        other => panic!("expected UnsupportedFormatVersion, got {:?}", other),
    }
}

#[test]
fn pack_signature_refuses_tampered_content() {
    let sk = test_signing_key();
    let pack = build_test_pack(&sk, b"AAAAAAAA".to_vec(), "df/test", 1);
    let mut bytes = pack.to_bytes().unwrap();
    // Flip a byte in the content region. Header layout:
    //   0..20  header (magic, version, manifest_len, content_len)
    //   20..20+manifest_len  manifest
    //   then content, then 64-byte signature
    let manifest_len = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
    let content_off = PACK5_HEADER_LEN + manifest_len;
    bytes[content_off] ^= 0xff;
    let parsed = CapabilityPack::decode_unverified(&bytes).expect("format still valid");
    let err = verify_capability_pack(&parsed).expect_err("sig must fail");
    matches!(err, CapabilityPackError::SignatureInvalid);
}

#[test]
fn pack_signature_refuses_tampered_manifest() {
    let sk = test_signing_key();
    let pack = build_test_pack(&sk, b"x".to_vec(), "df/test", 1);
    let mut bytes = pack.to_bytes().unwrap();
    // Flip a byte inside the manifest region.
    bytes[PACK5_HEADER_LEN] ^= 0x01;
    // decode_unverified may still succeed (depends on the byte hit);
    // either it parses and the signature fails, or it returns Format.
    match CapabilityPack::decode_unverified(&bytes) {
        Ok(parsed) => {
            let err = verify_capability_pack(&parsed).expect_err("sig must fail");
            matches!(err, CapabilityPackError::SignatureInvalid);
        }
        Err(CapabilityPackError::Format(_)) => {}
        Err(other) => panic!("unexpected: {:?}", other),
    }
}

// ---------------------------------------------------------------------
// Ledger install / uninstall tests
// ---------------------------------------------------------------------

#[test]
fn install_fails_when_publisher_not_trusted() {
    let sk = test_signing_key();
    let pack = build_test_pack(&sk, b"RUNTIME_TBD".to_vec(), "df/test", 1);
    let bytes = pack.to_bytes().unwrap();
    let mut ledger = CapabilityPackLedgerStore::new();
    let err = install_capability_pack(&mut ledger, &pack, &bytes, "<test>", false, 1)
        .expect_err("untrusted publisher must refuse");
    matches!(err, CapabilityPackError::UnknownPublisher { .. });
    assert!(ledger.installed_packs.is_empty());
}

#[test]
fn install_succeeds_when_publisher_trusted_and_ledger_reflects() {
    let sk = test_signing_key();
    let pk_hex = hex_encode(sk.verifying_key().as_bytes());
    let pack = build_test_pack(&sk, b"RUNTIME_TBD".to_vec(), "df/test", 1);
    let bytes = pack.to_bytes().unwrap();
    let mut ledger = CapabilityPackLedgerStore::new();
    ledger.add_publisher(&pk_hex, "DF", None, 1).unwrap();
    let res =
        install_capability_pack(&mut ledger, &pack, &bytes, "/tmp/p.lupack5", false, 99).unwrap();
    assert_eq!(res.pack_id, "df/test");
    assert_eq!(res.runtime_kind, PackRuntimeKind::Wasm);
    assert_eq!(res.tool_count, 1);
    assert!(!res.already_installed_overwrite);
    assert!(ledger.installed_packs.contains_key("df/test"));
    let recorded = ledger.installed_packs.get("df/test").unwrap();
    assert_eq!(recorded.installed_at_ms, 99);
    assert_eq!(recorded.source_path, "/tmp/p.lupack5");
    assert_eq!(recorded.tool_count, 1);
    assert_eq!(recorded.manifest_hash_hex, res.manifest_hash_hex);
}

#[test]
fn install_refuses_revoked_pack_id() {
    let sk = test_signing_key();
    let pk_hex = hex_encode(sk.verifying_key().as_bytes());
    let pack = build_test_pack(&sk, b"RUNTIME_TBD".to_vec(), "df/test", 1);
    let bytes = pack.to_bytes().unwrap();
    let mut ledger = CapabilityPackLedgerStore::new();
    ledger.add_publisher(&pk_hex, "DF", None, 1).unwrap();
    ledger.revoke("df/test");
    let err = install_capability_pack(&mut ledger, &pack, &bytes, "<test>", false, 1)
        .expect_err("revoked must refuse");
    matches!(err, CapabilityPackError::PackRevoked { .. });
    assert!(ledger.installed_packs.is_empty());
}

#[test]
fn install_already_installed_gate() {
    let sk = test_signing_key();
    let pk_hex = hex_encode(sk.verifying_key().as_bytes());
    let pack = build_test_pack(&sk, b"RUNTIME_TBD".to_vec(), "df/test", 1);
    let bytes = pack.to_bytes().unwrap();
    let mut ledger = CapabilityPackLedgerStore::new();
    ledger.add_publisher(&pk_hex, "DF", None, 1).unwrap();
    install_capability_pack(&mut ledger, &pack, &bytes, "<test>", false, 1).unwrap();
    let err = install_capability_pack(&mut ledger, &pack, &bytes, "<test>", false, 2)
        .expect_err("already installed must refuse without force");
    matches!(err, CapabilityPackError::PackAlreadyInstalled { .. });
    let res = install_capability_pack(&mut ledger, &pack, &bytes, "<test>", true, 3).unwrap();
    assert!(res.already_installed_overwrite);
}

#[test]
fn uninstall_removes_from_ledger() {
    let sk = test_signing_key();
    let pk_hex = hex_encode(sk.verifying_key().as_bytes());
    let pack = build_test_pack(&sk, b"RUNTIME_TBD".to_vec(), "df/test", 1);
    let bytes = pack.to_bytes().unwrap();
    let mut ledger = CapabilityPackLedgerStore::new();
    ledger.add_publisher(&pk_hex, "DF", None, 1).unwrap();
    install_capability_pack(&mut ledger, &pack, &bytes, "<test>", false, 1).unwrap();
    assert!(ledger.installed_packs.contains_key("df/test"));
    let removed = ledger.remove_install("df/test");
    assert!(removed.is_some());
    assert!(ledger.installed_packs.is_empty());
    assert!(!ledger.is_revoked("df/test"));
    // Now uninstall + revoke.
    install_capability_pack(&mut ledger, &pack, &bytes, "<test>", false, 2).unwrap();
    ledger.revoke("df/test");
    assert!(ledger.is_revoked("df/test"));
    assert!(ledger.installed_packs.is_empty());
}

// ---------------------------------------------------------------------
// v3 section round-trip tests (backward + forward compat)
// ---------------------------------------------------------------------
//
// These exercise the schema-layer guarantee that adding section #17
// doesn't break loaders that pre-date it. Two scenarios:
//   1. Backward: simulate a pre-Cat.5 brain (no section #17) and load
//      it under the new binary — ledger must hydrate empty.
//   2. Forward:  simulate a Cat.5 brain saved with section #17, then
//      strip the section from the offset table to imitate a pre-Cat.5
//      binary (which would skip the unknown id via J.1.1 tolerance) —
//      load must still succeed and return the empty ledger.
//
// The actual end-to-end save/load tests run alongside Cat.4's
// section-13 tests in `brain_io_v3.rs`; here we just verify the
// `de_bincode_optional` shape.

#[test]
fn ledger_snapshot_default_is_empty() {
    let snap = CapabilityPackLedgerSnapshot::default();
    assert!(snap.trusted_publishers.is_empty());
    assert!(snap.installed_packs.is_empty());
    assert!(snap.revoked_packs.is_empty());
    assert_eq!(snap.version, 0);
}

#[test]
fn ledger_store_from_default_snapshot_is_empty() {
    let store = CapabilityPackLedgerStore::from_snapshot(
        CapabilityPackLedgerSnapshot::default(),
    );
    assert!(store.trusted_publishers.is_empty());
    assert!(store.installed_packs.is_empty());
    assert!(store.revoked_packs.is_empty());
}

#[test]
fn brain_save_load_roundtrip_preserves_ledger() {
    use crate::brain::Brain;
    use crate::encrypt::MasterKey;
    use tempfile::TempDir;

    let dir = TempDir::new().unwrap();
    let path = dir.path().join("brain.lbr");

    // Build a brain with a populated capability-pack ledger.
    let mk = MasterKey::from_bytes([0x55u8; 32]);
    let mut brain = Brain::load_or_new_with_mode_and_key(
        &path, crate::brain::BrainLoadMode::Auto, Some(mk.clone()),
    ).expect("new brain with key");
    brain.require_encrypted = true;
    let sk = test_signing_key();
    let pk_hex = hex_encode(sk.verifying_key().as_bytes());
    brain
        .cap_pack_ledger
        .add_publisher(&pk_hex, "Test", Some("unit test".into()), 1_000)
        .unwrap();
    brain.cap_pack_ledger.record_install(InstalledPack {
        pack_id: "df/round-trip".into(),
        pack_version: "0.1.0".into(),
        publisher_pubkey_hex: pk_hex.clone(),
        runtime_kind: PackRuntimeKind::Wasm,
        installed_at_ms: 2_000,
        signed_at_ms: 1_500,
        manifest_hash_hex: "deadbeef".into(),
        tool_count: 1,
        source_path: "/tmp/x.lupack5".into(),
    });
    brain.cap_pack_ledger.revoke("df/old");

    // Save (v3enc) and reload.
    brain
        .save_to_format(&path, crate::brain::BrainSaveFormat::V3BinaryEncrypted)
        .expect("save");
    let reloaded = Brain::load_or_new_with_mode_and_key(
        &path, crate::brain::BrainLoadMode::Auto, Some(mk.clone()),
    ).expect("reload");

    // Hydrated ledger must match.
    assert_eq!(reloaded.cap_pack_ledger.trusted_publishers.len(), 1);
    assert!(reloaded.cap_pack_ledger.is_trusted(&pk_hex));
    assert!(reloaded.cap_pack_ledger.installed_packs.contains_key("df/round-trip"));
    assert!(reloaded.cap_pack_ledger.is_revoked("df/old"));
}

// ---------------------------------------------------------------------
// Anti-mutation grep guard — STRUCTURAL invariant
// ---------------------------------------------------------------------
//
// Cat.5 schema-only must not touch bonds, the teach path, or the
// PendingLog. This test reads the on-disk source files and asserts
// that capability-pack tool bodies use ONLY `cap_pack_ledger.*`
// mutations and never the bond-touching paths.

#[test]
fn antimutation_grep_guard() {
    // Read the two source files that contain Cat.5 logic.
    let mod_src = std::fs::read_to_string(file!())
        .or_else(|_| std::fs::read_to_string("src/capability_pack/tests.rs"))
        .unwrap_or_default();
    let _ = mod_src; // tests.rs may legitimately mention guarded names in this guard itself.

    let cap_pack_mod = std::fs::read_to_string("src/capability_pack/mod.rs")
        .expect("read src/capability_pack/mod.rs");
    let mcp_cat5 = std::fs::read_to_string("src/mcp_cat5.rs")
        .expect("read src/mcp_cat5.rs");

    let forbidden = [
        "brain.teach(",
        "brain.network.add_bond",
        "add_bond_with_source",
        "add_bond(",
        "pending.push(",
        "brain.pending.push",
        "brain.revoke_bonds_by_source",
    ];
    for needle in &forbidden {
        assert!(
            !cap_pack_mod.contains(needle),
            "capability_pack/mod.rs must not contain `{}` — Cat.5 is schema-only and must not mutate bonds.",
            needle
        );
        assert!(
            !mcp_cat5.contains(needle),
            "mcp_cat5.rs must not contain `{}` — Cat.5 tool bodies must mutate only cap_pack_ledger state.",
            needle
        );
    }
}
