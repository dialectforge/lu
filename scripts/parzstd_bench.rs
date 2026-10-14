//! OP-PARZSTD throughput benchmark (2026-06-01).
//!
//! Builds a synthetic NetworkSnapshot, then runs N save passes and
//! prints wall-clock per pass plus throughput in MB/s of compressed
//! output. Used to compare OLD vs NEW (parallel) save throughput on
//! identical inputs.
//!
//! usage: parzstd_bench <num_nodes> <bonds_per_node> <iters>

use lu4d::bond::{BondType, KeyframeSource, TemporalBond, TemporalKeyframe};
use lu4d::compress::write_compressed_network_payload;
use lu4d::encrypt::MasterKey;
use lu4d::network::NetworkSnapshot;
use lu4d::node::{Confidence, KnowledgeContent, KnowledgeNode, KnowledgeUnit, EMBEDDING_DIM};
use lu4d::seed::Seed;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("usage: {} <num_nodes> <bonds_per_node> <iters>", args[0]);
        std::process::exit(2);
    }
    let num_nodes: usize = args[1].parse().unwrap();
    let bonds_per_node: usize = args[2].parse().unwrap();
    let iters: usize = args[3].parse().unwrap();

    let mut state: u64 = 0xC0FFEE_DEAD_BEEF_u64;
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
            if j == i { continue; }
            let bt = match rng() & 3 {
                0 => BondType::IsA,
                1 => BondType::HasProperty,
                2 => BondType::Produces,
                _ => BondType::Causes,
            };
            let w = ((rng() & 0xFF) as f32) / 255.0;
            bonds.push(TemporalBond {
                from: nodes[i].seed,
                to: nodes[j].seed,
                bond_type: bt,
                keyframes: vec![TemporalKeyframe {
                    timestamp: 1_700_000_000_000,
                    weight: w,
                    source: KeyframeSource::Baseline,
                }],
                half_life_ms: 30.0 * 24.0 * 3600.0 * 1000.0,
            });
        }
    }
    let snap = NetworkSnapshot { nodes, bonds };
    eprintln!("snapshot: nodes={} bonds={}", snap.nodes.len(), snap.bonds.len());

    let key = MasterKey::from_bytes([0x55u8; 32]);

    let warm =
        write_compressed_network_payload(&snap, 256 * 1024, 3, Some(&key)).expect("warm");
    eprintln!("warm: {} bytes", warm.len());
    drop(warm);

    let mut total_ms: f64 = 0.0;
    let mut last_len: usize = 0;
    for i in 0..iters {
        let t0 = Instant::now();
        let payload =
            write_compressed_network_payload(&snap, 256 * 1024, 3, Some(&key)).expect("save");
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        total_ms += ms;
        last_len = payload.len();
        let mbs = (payload.len() as f64) / 1_048_576.0 / (ms / 1000.0);
        eprintln!(
            "iter {}: {} bytes in {:.1} ms ({:.1} MB/s)",
            i, payload.len(), ms, mbs
        );
        drop(payload);
    }
    let avg_ms = total_ms / iters as f64;
    let mbs = (last_len as f64) / 1_048_576.0 / (avg_ms / 1000.0);
    println!(
        "AVG: {} bytes in {:.1} ms over {} iters ({:.1} MB/s)",
        last_len, avg_ms, iters, mbs
    );
}
