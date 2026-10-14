//! Storage v2 Phase 2.1 — rkyv-compatibility surface.
//!
//! This module is the integration point Slice 2 will build the
//! mmap/zero-copy load path on top of. Slice 1 (this commit) only adds
//! the type-level machinery: `KnowledgeNode`, `TemporalBond`,
//! `NetworkSnapshot`, `Scar`, `ScarLog`, and their sub-types now derive
//! `rkyv::{Archive, Serialize, Deserialize}`. Production load is still
//! bincode v2 onto the heap via `brain_io.rs`.
//!
//! The single round-trip test below proves that the derives are wired up
//! correctly: a small synthetic `NetworkSnapshot` survives a round trip
//! through `rkyv::to_bytes` + `rkyv::check_archived_root` with node and
//! bond counts intact. That is the contract Slice 2 relies on.
//!
//! No public functions yet — the read path lives here in Slice 2.

// Intentionally empty at the type level; the derives live on the data
// types in their own modules.

#[cfg(test)]
mod tests {
    use crate::bond::{BondType, KeyframeSource, TemporalBond};
    use crate::network::NetworkSnapshot;
    use crate::node::{Confidence, KnowledgeContent, KnowledgeNode};
    use crate::scar::{Scar, ScarLog};
    use crate::seed::Seed;

    /// Construct a tiny synthetic snapshot — 10 nodes, 20 bonds, 1 scar —
    /// serialize via rkyv, validate the byte buffer with
    /// `check_archived_root`, and confirm node/bond/scar counts come
    /// back intact. This is the slice-1 acceptance test for Storage v2
    /// Phase 2.1.
    #[test]
    fn rkyv_roundtrip_small_snapshot_and_scar_log() {
        // 10 nodes, deterministic seeds so the test is reproducible.
        let seeds: Vec<Seed> = (0..10)
            .map(|i| Seed::from_label(&format!("node_{i}")))
            .collect();
        let nodes: Vec<KnowledgeNode> = seeds
            .iter()
            .enumerate()
            .map(|(i, s)| {
                KnowledgeNode::with_seed(
                    *s,
                    format!("content {i}"),
                    KnowledgeContent::Fact(format!("fact {i}")),
                    Confidence::Verified,
                    1_000 + i as i64,
                )
            })
            .collect();

        // 20 bonds, deterministic shape — every consecutive pair plus a
        // wrap-around pass — so we exercise the keyframe Vec, BondType
        // enum, and Seed identity through rkyv.
        let mut bonds: Vec<TemporalBond> = Vec::with_capacity(20);
        for i in 0..10 {
            let a = seeds[i];
            let b = seeds[(i + 1) % 10];
            let mut bond = TemporalBond::new(a, b, BondType::IsA, 0.5, 100 + i as i64);
            // Add a second keyframe with a non-Baseline source to exercise
            // the KeyframeSource enums associated-data variants.
            bond.add_keyframe(
                200 + i as i64,
                0.7,
                KeyframeSource::Hypothesis {
                    goal: seeds[(i + 2) % 10],
                    step: i,
                    session: Some(42),
                },
            );
            bonds.push(bond);
        }
        for i in 0..10 {
            let a = seeds[i];
            let b = seeds[(i + 3) % 10];
            bonds.push(TemporalBond::new(
                a,
                b,
                BondType::RelatedTo,
                0.25,
                500 + i as i64,
            ));
        }
        assert_eq!(nodes.len(), 10);
        assert_eq!(bonds.len(), 20);

        let snapshot = NetworkSnapshot { nodes, bonds };

        // === Snapshot round trip ===========================================
        // rkyv::to_bytes uses a fixed-size scratch arena; an AlignedVec
        // is returned. Slice 2 will write this to disk.
        let snap_bytes = rkyv::to_bytes::<_, 4096>(&snapshot)
            .expect("rkyv serialize NetworkSnapshot");
        assert!(!snap_bytes.is_empty(), "snapshot serialized to zero bytes");

        // check_archived_root validates the bytes with bytecheck before
        // exposing the archived view. This is the API Slice 2 will use
        // on mmap files (untrusted input must validate).
        let archived_snap =
            rkyv::check_archived_root::<NetworkSnapshot>(&snap_bytes[..])
                .expect("rkyv check_archived_root NetworkSnapshot");
        assert_eq!(archived_snap.nodes.len(), 10);
        assert_eq!(archived_snap.bonds.len(), 20);

        // Spot-check a single archived node: the embedding length and
        // the seed bytes should round-trip exactly. `archived_snap.nodes`
        // is an `ArchivedVec<ArchivedKnowledgeNode>`, so indexing works
        // without allocating.
        let archived_node_0 = &archived_snap.nodes[0];
        assert_eq!(
            archived_node_0.embedding.len(),
            crate::node::EMBEDDING_DIM,
        );
        assert_eq!(archived_node_0.seed.0, seeds[0].0);

        // Spot-check an archived bond: bond_type and from-seed should
        // round-trip exactly, and the second keyframe should still have
        // two entries (Baseline + Hypothesis).
        let archived_bond_0 = &archived_snap.bonds[0];
        assert_eq!(archived_bond_0.from.0, seeds[0].0);
        assert_eq!(archived_bond_0.keyframes.len(), 2);

        // === ScarLog round trip ============================================
        // ScarLog includes a HashSet<Seed> which is the trickiest of the
        // rkyv-derived types — exercise it explicitly.
        let mut log = ScarLog::new();
        let _: Scar = log.scar_bond(
            seeds[0],
            seeds[1],
            "slice 1 test".to_string(),
            Vec::new(),
            None,
            12_345,
        );
        log.scar_seed(seeds[2]);

        let scar_bytes = rkyv::to_bytes::<_, 1024>(&log)
            .expect("rkyv serialize ScarLog");
        let archived_log =
            rkyv::check_archived_root::<ScarLog>(&scar_bytes[..])
                .expect("rkyv check_archived_root ScarLog");
        assert_eq!(archived_log.scars.len(), 1);
        assert_eq!(archived_log.scarred_seeds.len(), 1);
        assert_eq!(archived_log.scars[0].reason.as_str(), "slice 1 test");
    }
}
