//! OP-WRITEBACK Phase-5 live demonstration (fresh in-memory brain).
//! Shows: guest privsep, serving-ephemeral, Stage-0 gate, Inferred+lineage
//! (distinguishable from ingested), corroboration promotion, a deliberately
//! WRONG conclusion caught + surgically retracted with cascade (ingested
//! survive), and per-edge confidence sharpening. Prints a verbatim transcript.
use lu4d::bond::{BondType, DerivationRule, PremiseEdge};
use lu4d::brain::Brain;
use lu4d::node::Confidence;
use lu4d::writeback::{
    edge_confidence, edge_is_concluded, Actor, Contradiction, DerivedConclusion, EmitMode,
    Writeback, WritebackDecision,
};

fn main() {
    let mut b = Brain::new();
    let mut wb = Writeback::new();
    // Ingested ground facts.
    let r1 = b.teach("penguin", BondType::IsA, "bird", 0.9, Confidence::Verified);
    let r2 = b.teach("bird", BondType::HasProperty, "flies", 0.9, Confidence::Verified);
    let (penguin, bird, flies) = (r1.subject_seed, r1.object_seed, r2.object_seed);
    let premises = vec![
        PremiseEdge { subject: penguin, predicate: BondType::IsA, object: bird },
        PremiseEdge { subject: r2.subject_seed, predicate: BondType::HasProperty, object: flies },
    ];
    let dc = DerivedConclusion {
        subject: penguin, bond_type: BondType::HasProperty, object: flies,
        rule: DerivationRule::PropertyInheritance, premises: premises.clone(), weight: 0.7,
        subject_label: "penguin".into(), object_label: "flies".into(),
    };

    println!("== (e) PRIVSEP: guest cannot writeback ==");
    let g = wb.emit(&mut b, Actor::Guest, EmitMode::OwnerWindow, &dc, 1000);
    println!("guest emit -> {:?}; pending_count={}", g, b.pending.pending_count());

    println!("\n== (d) SERVING conclusions are EPHEMERAL ==");
    let bonds0 = b.network.bond_count();
    let s = wb.emit(&mut b, Actor::Owner, EmitMode::Serving, &dc, 1000);
    println!("serving emit -> {:?}", s);
    println!("persisted? bond(penguin->flies)={:?}; bond_count {}->{} (unchanged); pending={}",
        b.network.get_bond(&penguin, &flies).is_some(), bonds0, b.network.bond_count(), b.pending.pending_count());

    println!("\n== (d) STAGE-0 gate + (b) writeback distinguishable ==");
    let id = match wb.emit(&mut b, Actor::Owner, EmitMode::OwnerWindow, &dc, 1000) {
        WritebackDecision::Queued { pending_id } => pending_id, _ => unreachable!() };
    println!("owner emit -> Queued(pending_id={}); durable yet? {}", id, b.network.get_bond(&penguin, &flies).is_some());
    let t = std::time::Instant::now();
    let key = wb.approve_and_persist(&mut b, id, "john", 1001).unwrap();
    let persist_us = t.elapsed().as_micros();
    println!("approve+persist -> durable={}; belief={:?}; is_concluded(derived)={}; is_concluded(ingested penguin->bird)={}",
        b.network.get_bond(&penguin, &flies).is_some(), wb.belief_state(&b, &key),
        edge_is_concluded(&b, &penguin, &flies), edge_is_concluded(&b, &penguin, &bird));

    println!("\n== (f) per-edge confidence sharpens ==");
    println!("ingested  penguin-IsA->bird     : edge_confidence = {:?}", edge_confidence(&b, &penguin, &bird, Confidence::Verified));
    println!("derived   penguin-HasProp->flies: edge_confidence = {:?} (tentative)", edge_confidence(&b, &penguin, &flies, Confidence::Verified));
    let independent = vec![PremiseEdge { subject: penguin, predicate: BondType::IsA, object: bird }];
    wb.corroborate(&mut b, key, DerivationRule::Transitivity, independent, 1002);
    println!("after INDEPENDENT corroboration : belief={:?}; edge_confidence = {:?} (established)",
        wb.belief_state(&b, &key), edge_confidence(&b, &penguin, &flies, Confidence::Verified));

    println!("\n== (c) IMMUNE SYSTEM: plant a WRONG conclusion, catch + surgical retract + cascade ==");
    // derived2 built ON derived1 (penguin IsA flyer <- penguin HasProperty flies)
    let rf = b.teach("bird", BondType::RelatedTo, "flyer", 0.5, Confidence::Unverified);
    let flyer = rf.object_seed;
    let dc2 = DerivedConclusion {
        subject: penguin, bond_type: BondType::IsA, object: flyer, rule: DerivationRule::Transitivity,
        premises: vec![PremiseEdge { subject: penguin, predicate: BondType::HasProperty, object: flies }],
        weight: 0.6, subject_label: "penguin".into(), object_label: "flyer".into(),
    };
    let id2 = match wb.emit(&mut b, Actor::Owner, EmitMode::OwnerWindow, &dc2, 1003) {
        WritebackDecision::Queued { pending_id } => pending_id, _ => unreachable!() };
    let key2 = wb.approve_and_persist(&mut b, id2, "john", 1004).unwrap();
    println!("derived1 penguin->flies present={}, derived2 penguin->flyer present={}",
        b.network.get_bond(&penguin, &flies).is_some(), b.network.get_bond(&penguin, &flyer).is_some());
    // Owner learns the truth: penguins do NOT fly.
    b.set_bond_negated(penguin, flies, BondType::HasProperty, true);
    let sig = wb.detect_contradiction(&b, &key, BondType::HasProperty, false);
    println!("contradiction signal on penguin->flies = {:?}", sig);
    assert_eq!(sig, Contradiction::Negated);
    let tr = std::time::Instant::now();
    let retracted = wb.retract(&mut b, key);
    let retract_us = tr.elapsed().as_micros();
    println!("retracted (surgical, via lineage) = {} edges", retracted.len());
    println!("  derived1 penguin->flies removed?  {}", b.network.get_bond(&penguin, &flies).is_none());
    println!("  derived2 penguin->flyer cascaded? {}", b.network.get_bond(&penguin, &flyer).is_none());
    println!("  INGESTED penguin->bird survives?  {}", b.network.get_bond(&penguin, &bird).is_some());
    println!("  INGESTED bird->flies survives?    {}", b.network.get_bond(&bird, &flies).is_some());

    println!("\n== (h) latency (single-op, in-memory) ==");
    println!("approve+persist = {} us ; surgical retract+cascade = {} us", persist_us, retract_us);
    println!("\nALL PHASE-5 INVARIANTS DEMONSTRATED.");
}
