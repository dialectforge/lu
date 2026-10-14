//! OP-PARZSTD fixture-save sha256 equivalence harness (2026-06-01).
//!
//! Constructs a deterministic NetworkSnapshot from a seed, calls
//! `lu4d::compress::write_compressed_network_payload`, and prints
//! `sha256(payload) len=N` for both the unencrypted and encrypted
//! save paths.
//!
//! The same source compiles against OLD (pre-OP-PARZSTD) and NEW
//! (parallel) code; sha256 + len must match exactly between the two
//! builds for the change to be byte-identical.
//!
//! usage: parzstd_fixture <num_nodes> <bonds_per_node> [seed]

use lu4d::bond::{BondType, KeyframeSource, TemporalBond, TemporalKeyframe};
use lu4d::compress::write_compressed_network_payload;
use lu4d::encrypt::MasterKey;
use lu4d::network::NetworkSnapshot;
use lu4d::node::{Confidence, KnowledgeContent, KnowledgeNode, KnowledgeUnit, EMBEDDING_DIM};
use lu4d::seed::Seed;
use sha2::{Digest, Sha256};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: {} <num_nodes> <bonds_per_node> [seed]", args[0]);
        std::process::exit(2);
    }
    let num_nodes: usize = args[1].parse().unwrap();
    let bonds_per_node: usize = args[2].parse().unwrap();
    let seed_u64: u64 = args
        .get(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0xDEAD_BEEF);

    // Deterministic LCG. (Doesn't matter what algorithm — only matters
    // that both binaries see the same sequence.)
    let mut state = seed_u64;
    let mut rng = || -> u64 {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        state
    };

    let mut nodes: Vec<KnowledgeNode> = Vec::with_capacity(num_nodes);
    for i in 0..num_nodes {
        let r = rng();
        let label = format!("node-{:016x}-{}", r, i);
        let mut emb = vec![0.0_f32; EMBEDDING_DIM];
        for (j, s) in emb.iter_mut().enumerate() {
            *s = ((j as f32) * 0.001 + label.len() as f32 * 0.01).sin();
        }
        nodes.push(KnowledgeNode {
            seed: Seed::from_label(&label),
            content: label.clone(),
            knowledge: KnowledgeUnit {
                content: KnowledgeContent::Concept(format!("concept of {}", label)),
                confidence: Confidence::Verified,
                created_at: 1_700_000_000_000,
                updated_at: 1_700_000_000_000,
            },
            embedding: emb,
            parent_seeds: Vec::new(),
            typed_ratio: 1.0,
        });
    }

    let mut bonds: Vec<TemporalBond> = Vec::with_capacity(num_nodes * bonds_per_node);
    for i in 0..num_nodes {
        for _ in 0..bonds_per_node {
            let j = (rng() as usize) % num_nodes;
            if j == i {
                continue;
            }
            let bt = match rng() & 3 {
                0 => BondType::IsA,
                1 => BondType::HasProperty,
                2 => BondType::Produces,
                _ => BondType::Causes,
            };
            let w = ((rng() & 0xFF) as f32) / 255.0;
            let ts_off = (rng() & 0xFFFF) as i64;
            bonds.push(TemporalBond {
                from: nodes[i].seed,
                to: nodes[j].seed,
                bond_type: bt,
                keyframes: vec![TemporalKeyframe {
                    timestamp: 1_700_000_000_000 + ts_off,
                    weight: w,
                    source: KeyframeSource::Baseline,
                }],
                half_life_ms: 30.0 * 24.0 * 3600.0 * 1000.0,
            });
        }
    }

    let snap = NetworkSnapshot { nodes, bonds };
    eprintln!("snapshot: nodes={} bonds={}", snap.nodes.len(), snap.bonds.len());

    let key = MasterKey::from_bytes([0x37u8; 32]);

    let payload_unenc =
        write_compressed_network_payload(&snap, 256 * 1024, 3, None).expect("unenc");
    let mut h = Sha256::new();
    h.update(&payload_unenc);
    let sha = h.finalize();
    println!("UNENC len={} sha256={}", payload_unenc.len(), hex(&sha));

    let payload_enc =
        write_compressed_network_payload(&snap, 256 * 1024, 3, Some(&key)).expect("enc");
    let mut h2 = Sha256::new();
    h2.update(&payload_enc);
    let sha2 = h2.finalize();
    println!("ENC   len={} sha256={}", payload_enc.len(), hex(&sha2));
}

fn hex(b: &[u8]) -> String {
    let mut s = String::with_capacity(b.len() * 2);
    for x in b {
        s.push_str(&format!("{:02x}", x));
    }
    s
}
