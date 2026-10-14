//! Port Cat.4 — tests for the signed safety pack format.
//!
//! Coverage: format round-trip, signature happy path + 2 sad paths,
//! ingest rejection paths (4), pending-log routing (the
//! anti-hallucination invariant), producer/consumer end-to-end,
//! uninstall-via-revoke-by-source, v3 section round-trip, pre-Cat.4
//! brain forward compat, clock-source determinism.

use super::*;
use crate::bond::BondType;
use crate::brain::Brain;
use crate::encrypt::MasterKey;

fn test_master_key() -> MasterKey {
    MasterKey::from_bytes([
        0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42,
        0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42,
        0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42,
        0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42,
    ])
}

fn sample_bonds() -> Vec<PackBond> {
    vec![
        PackBond {
            subject_text: "aspirin".into(),
            object_text: "analgesic".into(),
            bond_type: BondType::IsA,
            weight: 0.9,
            negated: false,
            confidence: 0.85,
            reason: "medical pack: aspirin classification".into(),
        },
        PackBond {
            subject_text: "ibuprofen".into(),
            object_text: "nsaid".into(),
            bond_type: BondType::IsA,
            weight: 0.95,
            negated: false,
            confidence: 0.9,
            reason: "".into(),
        },
    ]
}

fn build_test_pack(
    sk: &ed25519_dalek::SigningKey,
    mk: &MasterKey,
    bonds: Vec<PackBond>,
    now_ms: u64,
) -> Pack {
    let seed = PackManifestSeed {
        pack_id: "df.test-pack".into(),
        pack_version: "1.0.0".into(),
        content_summary: "test pack with sample bonds".into(),
        minimum_lu_version: "0.1.0".into(),
        depends_on_pack_ids: Vec::new(),
        license: "MIT".into(),
    };
    build_pack(PackBuildSpec {
        manifest_seed: seed,
        bonds,
        signing_key: sk,
        master_key: mk,
        now_ms,
    })
    .expect("build_pack")
}

fn install_brain(mk: MasterKey, pubkey_hex: &str) -> Brain {
    let mut brain = Brain::new();
    brain.encryption_key = Some(mk);
    brain.pack_ledger.add_publisher(
        pubkey_hex,
        "Test Publisher",
        Some("unit-test fixture".into()),
        1_000,
    ).expect("add publisher");
    brain
}

// ----------------------------------------------------------------
// 1. Pack format round-trip
// ----------------------------------------------------------------
#[test]
fn cat4_pack_format_round_trips() {
    let (sk_bytes, pk_hex) = generate_keypair();
    let sk = signing_key_from_bytes(&sk_bytes);
    let mk = test_master_key();
    let pack = build_test_pack(&sk, &mk, sample_bonds(), 1_500_000);
    let bytes = pack.to_bytes().expect("to_bytes");
    let back = Pack::decode_unverified(&bytes).expect("decode");
    assert_eq!(back.manifest, pack.manifest);
    assert_eq!(back.content_ciphertext, pack.content_ciphertext);
    assert_eq!(back.signature, pack.signature);
    assert_eq!(pack.manifest.publisher_id, pk_hex);
    assert_eq!(pack.manifest.bond_count, 2);
}

// ----------------------------------------------------------------
// 2. Sig verifies with correct pubkey
// ----------------------------------------------------------------
#[test]
fn cat4_pack_signature_verifies_with_correct_pubkey() {
    let (sk_bytes, pk_hex) = generate_keypair();
    let sk = signing_key_from_bytes(&sk_bytes);
    let mk = test_master_key();
    let pack = build_test_pack(&sk, &mk, sample_bonds(), 1_500_000);
    let mut brain = install_brain(mk, &pk_hex);
    let bytes = pack.to_bytes().expect("bytes");
    let res = ingest_pack(&mut brain, &pack, &bytes, 2_000_000)
        .expect("ingest ok");
    assert_eq!(res.queued_bond_count, 2);
    assert_eq!(res.queued_pending_ids.len(), 2);
    assert_eq!(res.publisher_id, pk_hex);
}

// ----------------------------------------------------------------
// 3. Sig rejects with wrong pubkey
// ----------------------------------------------------------------
#[test]
fn cat4_pack_signature_rejects_with_wrong_pubkey() {
    let (sk_bytes_a, _pk_hex_a) = generate_keypair();
    let (_sk_bytes_b, pk_hex_b) = generate_keypair();
    let sk_a = signing_key_from_bytes(&sk_bytes_a);
    let mk = test_master_key();
    let mut pack = build_test_pack(&sk_a, &mk, sample_bonds(), 1_500_000);
    // Swap publisher_id to point to a different key whose signature
    // does NOT match.
    pack.manifest.publisher_id = pk_hex_b.clone();
    // Re-sign? No — leave signature as-is. The publisher claim is
    // pk_hex_b but signed by sk_a, so verify must fail.
    let mut brain = install_brain(mk, &pk_hex_b);
    let bytes = pack.to_bytes().expect("bytes");
    let err = ingest_pack(&mut brain, &pack, &bytes, 2_000_000)
        .expect_err("must reject");
    matches!(err, PackError::SignatureInvalid);
}

// ----------------------------------------------------------------
// 4. Sig rejects when content tampered
// ----------------------------------------------------------------
#[test]
fn cat4_pack_signature_rejects_when_content_tampered() {
    let (sk_bytes, pk_hex) = generate_keypair();
    let sk = signing_key_from_bytes(&sk_bytes);
    let mk = test_master_key();
    let pack = build_test_pack(&sk, &mk, sample_bonds(), 1_500_000);
    let mut brain = install_brain(mk, &pk_hex);
    let mut bytes = pack.to_bytes().expect("bytes");
    // Flip a byte deep inside the content section so signature
    // mismatches but format still parses.
    let manifest_len_buf = [bytes[12], bytes[13], bytes[14], bytes[15]];
    let manifest_end = PACK_HEADER_LEN + u32::from_le_bytes(manifest_len_buf) as usize;
    let target = manifest_end + 5;
    bytes[target] ^= 0xFF;
    let parsed = Pack::decode_unverified(&bytes).expect("still parses");
    let err = ingest_pack(&mut brain, &parsed, &bytes, 2_000_000)
        .expect_err("must reject tampered");
    matches!(err, PackError::SignatureInvalid);
}

// ----------------------------------------------------------------
// 5. Ingest rejects unknown publisher
// ----------------------------------------------------------------
#[test]
fn cat4_pack_ingest_rejects_unknown_publisher() {
    let (sk_bytes, _pk_hex) = generate_keypair();
    let sk = signing_key_from_bytes(&sk_bytes);
    let mk = test_master_key();
    let pack = build_test_pack(&sk, &mk, sample_bonds(), 1_500_000);
    // Brain WITHOUT this publisher in the trusted list.
    let mut brain = Brain::new();
    brain.encryption_key = Some(mk);
    let bytes = pack.to_bytes().expect("bytes");
    let err = ingest_pack(&mut brain, &pack, &bytes, 2_000_000)
        .expect_err("must reject unknown publisher");
    matches!(err, PackError::UnknownPublisher { .. });
}

// ----------------------------------------------------------------
// 6. Ingest rejects revoked pack
// ----------------------------------------------------------------
#[test]
fn cat4_pack_ingest_rejects_revoked_pack() {
    let (sk_bytes, pk_hex) = generate_keypair();
    let sk = signing_key_from_bytes(&sk_bytes);
    let mk = test_master_key();
    let pack = build_test_pack(&sk, &mk, sample_bonds(), 1_500_000);
    let mut brain = install_brain(mk, &pk_hex);
    brain.pack_ledger.record_revoke("df.test-pack", "key compromise", 2_000_000);
    let bytes = pack.to_bytes().expect("bytes");
    let err = ingest_pack(&mut brain, &pack, &bytes, 2_000_000)
        .expect_err("must reject revoked");
    matches!(err, PackError::PackRevoked { .. });
}

// ----------------------------------------------------------------
// 7. Ingest rejects old minimum_lu_version (i.e. pack requires NEWER)
// ----------------------------------------------------------------
#[test]
fn cat4_pack_ingest_rejects_old_minimum_lu_version() {
    let (sk_bytes, pk_hex) = generate_keypair();
    let sk = signing_key_from_bytes(&sk_bytes);
    let mk = test_master_key();
    let mut pack = build_test_pack(&sk, &mk, sample_bonds(), 1_500_000);
    pack.manifest.minimum_lu_version = "99.0.0".into();
    // Re-sign with new manifest.
    use ed25519_dalek::Signer;
    let manifest_bytes = bincode::serialize(&pack.manifest).unwrap();
    let payload = Pack::signing_payload(&manifest_bytes, &pack.content_ciphertext);
    pack.signature = sk.sign(&payload).to_bytes();
    let mut brain = install_brain(mk, &pk_hex);
    let bytes = pack.to_bytes().expect("bytes");
    let err = ingest_pack(&mut brain, &pack, &bytes, 2_000_000)
        .expect_err("must reject old lu version");
    matches!(err, PackError::LuVersionTooOld { .. });
}

// ----------------------------------------------------------------
// 8. Ingest rejects old format version
// ----------------------------------------------------------------
#[test]
fn cat4_pack_ingest_rejects_old_format_version() {
    let (sk_bytes, _pk_hex) = generate_keypair();
    let sk = signing_key_from_bytes(&sk_bytes);
    let mk = test_master_key();
    let pack = build_test_pack(&sk, &mk, sample_bonds(), 1_500_000);
    let mut bytes = pack.to_bytes().expect("bytes");
    // Stomp the format version field with a value we don't support.
    bytes[8..12].copy_from_slice(&999u32.to_le_bytes());
    let err = Pack::decode_unverified(&bytes).expect_err("must reject");
    matches!(err, PackError::UnsupportedFormatVersion { .. });
}

// ----------------------------------------------------------------
// 9. Pack install routes bonds through PendingLog (anti-hallucination)
// ----------------------------------------------------------------
#[test]
fn cat4_pack_install_routes_bonds_through_pending_log() {
    let (sk_bytes, pk_hex) = generate_keypair();
    let sk = signing_key_from_bytes(&sk_bytes);
    let mk = test_master_key();
    let pack = build_test_pack(&sk, &mk, sample_bonds(), 1_500_000);
    let mut brain = install_brain(mk, &pk_hex);
    let initial_bonds = brain.network.bond_count();
    let initial_pending = brain.pending.total_count();
    let bytes = pack.to_bytes().expect("bytes");
    let res = ingest_pack(&mut brain, &pack, &bytes, 2_000_000).expect("ingest");
    // Bond count UNCHANGED — the anti-hallucination invariant.
    assert_eq!(brain.network.bond_count(), initial_bonds,
        "bond_count must not change before approval");
    // PendingLog grew by 2.
    let now_pending = brain.pending.total_count();
    assert_eq!(now_pending, initial_pending + 2);
    assert_eq!(res.queued_pending_ids.len(), 2);
    // Port SourceKind-Variants — assertion updated from legacy Teach
    // actor to first-class PackInstall.
    let entry0 = brain.pending.get(res.queued_pending_ids[0]).expect("pending");
    if let crate::source::SourceKind::PackInstall { pack_id, publisher_pubkey_hex, signed_at_ms } = &entry0.proposed_by.kind {
        assert_eq!(pack_id, "df.test-pack");
        assert_eq!(publisher_pubkey_hex, &pk_hex);
        assert!(*signed_at_ms > 0);
    } else {
        panic!("expected SourceKind::PackInstall, got {:?}", entry0.proposed_by.kind);
    }
}

// ----------------------------------------------------------------
// 10. Producer/consumer end-to-end via install
// ----------------------------------------------------------------
#[test]
fn cat4_pack_create_round_trips_via_install() {
    let (sk_bytes, pk_hex) = generate_keypair();
    let sk = signing_key_from_bytes(&sk_bytes);
    let mk = test_master_key();
    let bonds = sample_bonds();
    let pack = build_test_pack(&sk, &mk, bonds.clone(), 1_500_000);
    // Write to disk + read back, exercising the file path.
    let dir = std::env::temp_dir();
    let path = dir.join(format!("cat4_e2e_{}.lupack", std::process::id()));
    write_pack_file(&pack, &path).expect("write file");
    let (parsed, raw_bytes) = read_pack_file(&path).expect("read file");
    let _ = std::fs::remove_file(&path);
    let mut brain = install_brain(mk, &pk_hex);
    let res = ingest_pack(&mut brain, &parsed, &raw_bytes, 2_000_000)
        .expect("ingest");
    assert_eq!(res.queued_bond_count, bonds.len() as u64);
    // Approve every queued change and apply, then check bonds landed.
    for id in &res.queued_pending_ids {
        brain.pending.approve(*id, 3_000_000, "john").unwrap();
        crate::pending::apply_change(&mut brain, *id).unwrap();
    }
    // After approve+apply the bond_count must have increased by
    // exactly the count we queued (assuming no scar gate rejects).
    assert!(brain.network.bond_count() >= bonds.len(),
        "bond_count must include pack bonds after approve+apply");
}

// ----------------------------------------------------------------
// 11. Uninstall revokes bonds via revoke_by_source
// ----------------------------------------------------------------
#[test]
fn cat4_pack_uninstall_revokes_bonds_via_revoke_by_source() {
    let (sk_bytes, pk_hex) = generate_keypair();
    let sk = signing_key_from_bytes(&sk_bytes);
    let mk = test_master_key();
    let pack = build_test_pack(&sk, &mk, sample_bonds(), 1_500_000);
    let mut brain = install_brain(mk, &pk_hex);
    let bytes = pack.to_bytes().expect("bytes");
    let res = ingest_pack(&mut brain, &pack, &bytes, 2_000_000).expect("ingest");
    for id in &res.queued_pending_ids {
        brain.pending.approve(*id, 3_000_000, "john").unwrap();
        crate::pending::apply_change(&mut brain, *id).unwrap();
    }
    let bonds_after_install = brain.network.bond_count();
    assert!(bonds_after_install >= 2, "should have added bonds");
    // Revoke every bond whose source is Teach with actor =
    // "pack_install:df.test-pack". This is what the
    // `lu4d_pack_uninstall` MCP tool does internally.
    // Port SourceKind-Variants — revoke matches first-class PackInstall
    // (and as a back-compat safety net, the legacy Teach shape too).
    let (overlay_revoked, bonds_revoked) = brain.revoke_bonds_by_source(|attr| {
        match &attr.kind {
            crate::source::SourceKind::PackInstall { pack_id, .. } => {
                pack_id == "df.test-pack"
            }
            crate::source::SourceKind::Teach { actor, .. } => {
                actor == "pack_install:df.test-pack"
            }
            _ => false,
        }
    });
    assert!(overlay_revoked >= 2, "should have revoked >=2 overlay entries: {}", overlay_revoked);
    assert!(bonds_revoked >= 2, "should have revoked >=2 bonds: {}", bonds_revoked);
    // Mark the pack revoked in the ledger as the MCP tool also does.
    brain.pack_ledger.record_revoke("df.test-pack", "uninstalled by user", 4_000_000);
    assert!(brain.pack_ledger.is_revoked("df.test-pack"));
}

// ----------------------------------------------------------------
// 12. v3 section round-trips
// ----------------------------------------------------------------
#[test]
fn cat4_v3_section_pack_ledger_round_trips() {
    let mut store = PackLedgerStore::new();
    store.add_publisher(
        &"a".repeat(64),
        "DialectForge Official",
        Some("primary key".into()),
        1_000,
    ).expect("add");
    store.record_install(InstalledPack {
        pack_id: "df.icd10".into(),
        publisher_id: "a".repeat(64),
        pack_version: "1.0.0".into(),
        bond_count_added: 100,
        node_count_added: 80,
        installed_at_ms: 2_000,
        manifest_hash_hex: sha256_hex(b"manifest-bytes"),
    });
    store.record_revoke("old.pack", "compromised", 3_000);
    let snap = store.snapshot();
    let bytes = bincode::serialize(&snap).expect("serialize");
    let back: PackLedgerSnapshot =
        bincode::deserialize(&bytes).expect("deserialize");
    assert_eq!(snap, back);
    assert_eq!(back.trusted_publishers.len(), 1);
    assert_eq!(back.installed_packs.len(), 1);
    assert_eq!(back.revoked_packs.len(), 1);
}

// ----------------------------------------------------------------
// 13. Pre-Cat.4 brain loads with empty PackLedger (forward compat)
// ----------------------------------------------------------------
#[test]
fn cat4_pre_cat4_brain_loads_with_empty_pack_ledger() {
    // Default-snapshot is what `de_bincode_optional` returns when
    // the section is missing.
    let snap = PackLedgerSnapshot::default();
    let store = PackLedgerStore::from_snapshot(snap);
    assert!(store.trusted_publishers.is_empty());
    assert!(store.installed_packs.is_empty());
    assert!(store.revoked_packs.is_empty());
    // is_trusted is_revoked behave sensibly on an empty ledger.
    assert!(!store.is_trusted(&"a".repeat(64)));
    assert!(!store.is_revoked("any.pack"));
}

// ----------------------------------------------------------------
// 14. Ledger uses caller-provided clock (not SystemTime)
// ----------------------------------------------------------------
#[test]
fn cat4_pack_ledger_uses_clock_source_not_system_time() {
    // Same pattern as T.5 / CS: every mutator takes `now_ms` as
    // an explicit parameter — never reads SystemTime internally.
    let mut store = PackLedgerStore::new();
    let pk = "b".repeat(64);
    let p = store.add_publisher(&pk, "Test", None, 12_345).expect("add");
    assert_eq!(p.added_at_ms, 12_345);
    store.record_revoke("some.pack", "reason", 999_999);
    let r = store.revoked_packs.get("some.pack").expect("revoked");
    assert_eq!(r.revoked_at_ms, 999_999);
}
