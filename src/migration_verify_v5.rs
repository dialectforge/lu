//! OP-BRAIN-V5 6.7 -- offline migration verifier.
//!
//! Given a v3 brain (`.lbr`) and the v5 directory it was migrated into,
//! this module loads BOTH and proves the v5 brain is a faithful,
//! safe-to-serve replacement for v3 BEFORE any production cutover. It is
//! the offline GO/NO-GO gate referenced in the design doc §7 step-5
//! runbook -- it is NEVER run against a live brain in place; it loads
//! read-only copies.
//!
//! Three independent checks, each reported with detail (not a bare
//! pass/fail) so a NO-GO tells the operator *what* diverged:
//!
//!   1. RECALL PARITY -- for a sample of subject nodes, the set of
//!      outgoing bonds `(object, predicate)` served by the v5 read path
//!      (`V5Only`) must equal the set served by v3 (`V3Only`). Reports a
//!      match rate and the first diverging subjects.
//!   2. ENCRYPTION-AT-REST -- every NodeSST/BondSST on disk must carry
//!      the `FLAG_ENCRYPTED` header bit (reusing the 6.5
//!      `scan_sst_encryption` marker mechanism) AND the MANIFEST must
//!      carry a real `master_key_id`. Any plaintext SST is a hard fail.
//!   3. COUNT PARITY -- v3 node/bond cardinality must equal v5's (the
//!      v5 figures are the in-RAM totals the brain actually serves after
//!      `rehydrate_adjacency` / the node intern).
//!
//! `VerifyReport::passed()` is the GO signal: all three must hold.

use std::collections::HashSet;
use std::path::PathBuf;

use crate::bond::BondType;
use crate::brain::{Brain, BrainError, BrainLoadMode};
use crate::encrypt::MasterKey;
use crate::read_path::ReadPathMode;
use crate::seed::Seed;

/// Inputs to a verification run.
#[derive(Debug, Clone)]
pub struct VerifyConfig {
    /// The source v3 `.lbr` brain (read-only; never mutated).
    pub v3_path: PathBuf,
    /// The migrated v5 directory (`node_store/` + `bond_store/` + MANIFEST).
    pub v5_dir: PathBuf,
    /// Master key. Required to read an encrypted v3 source and/or an
    /// encrypted v5 store. `None` only for the plaintext-fixture path.
    pub key: Option<MasterKey>,
    /// Max subjects to deep-compare for recall parity. `0` = all nodes.
    /// On the production-scale brain pass a bounded sample.
    pub sample_limit: usize,
}

impl VerifyConfig {
    pub fn new(v3_path: PathBuf, v5_dir: PathBuf) -> Self {
        Self { v3_path, v5_dir, key: None, sample_limit: 0 }
    }
}

/// One subject whose v3 and v5 outgoing-bond sets diverged.
#[derive(Debug, Clone)]
pub struct RecallMismatch {
    pub subject: Seed,
    /// `(object, predicate)` present in v3 but missing from v5.
    pub only_in_v3: Vec<(Seed, BondType)>,
    /// `(object, predicate)` present in v5 but missing from v3.
    pub only_in_v5: Vec<(Seed, BondType)>,
}

/// Full verification report. Every field is observable so a NO-GO is
/// actionable.
#[derive(Debug, Clone)]
pub struct VerifyReport {
    // ---- count parity ----
    pub v3_node_count: usize,
    pub v5_node_count: usize,
    pub node_count_match: bool,
    pub v3_bond_count: usize,
    pub v5_bond_count: usize,
    pub bond_count_match: bool,
    /// v3 node seeds that could NOT be resolved + read back in v5
    /// (intern miss or `get_node_v5` miss). Empty on a clean migration.
    pub missing_nodes_in_v5: usize,

    // ---- recall parity ----
    pub recall_subjects_sampled: usize,
    pub recall_subjects_matched: usize,
    /// Total v3 outgoing bonds seen across the sampled subjects.
    pub recall_objects_v3_total: usize,
    /// Of those, how many were also served (same object+predicate) by v5.
    pub recall_objects_matched: usize,
    /// First diverging subjects (capped; the counts above are exhaustive
    /// over the sample).
    pub recall_mismatches: Vec<RecallMismatch>,

    // ---- encryption-at-rest ----
    pub node_ssts_total: usize,
    pub node_ssts_encrypted: usize,
    pub bond_ssts_total: usize,
    pub bond_ssts_encrypted: usize,
    /// Recognised SSTs whose FLAG_ENCRYPTED bit is clear. Non-empty => fail.
    pub plaintext_ssts: Vec<PathBuf>,
    /// MANIFEST `master_key_id` is a real fingerprint (not `none()`).
    pub manifest_master_key_set: bool,
}

/// Cap on stored mismatch detail (the rate counters stay exhaustive).
const MAX_RECORDED_MISMATCHES: usize = 50;

impl VerifyReport {
    /// Fraction of sampled subjects whose recall set matched exactly.
    pub fn recall_match_rate(&self) -> f64 {
        if self.recall_subjects_sampled == 0 {
            return 1.0;
        }
        self.recall_subjects_matched as f64 / self.recall_subjects_sampled as f64
    }

    /// Fraction of sampled v3 outgoing bonds also served by v5.
    pub fn recall_object_match_rate(&self) -> f64 {
        if self.recall_objects_v3_total == 0 {
            return 1.0;
        }
        self.recall_objects_matched as f64 / self.recall_objects_v3_total as f64
    }

    /// True iff every on-disk SST is encrypted AND the MANIFEST carries a
    /// real key fingerprint AND at least one SST was actually present
    /// (an empty store can't prove encryption-at-rest).
    pub fn encryption_ok(&self) -> bool {
        let total = self.node_ssts_total + self.bond_ssts_total;
        self.plaintext_ssts.is_empty()
            && self.manifest_master_key_set
            && total > 0
            && (self.node_ssts_encrypted + self.bond_ssts_encrypted) == total
    }

    /// GO signal: counts match, every node round-trips, recall parity is
    /// perfect over the sample, and the brain is encrypted at rest.
    pub fn passed(&self) -> bool {
        self.node_count_match
            && self.bond_count_match
            && self.missing_nodes_in_v5 == 0
            && self.recall_mismatches.is_empty()
            && (self.recall_subjects_matched == self.recall_subjects_sampled)
            && self.encryption_ok()
    }

    /// Human-readable GO/NO-GO report.
    pub fn render(&self) -> String {
        let mut s = String::new();
        let verdict = if self.passed() { "GO  (PASS)" } else { "NO-GO  (FAIL)" };
        s.push_str("==== OP-BRAIN-V5 migration verifier ====\n");
        s.push_str(&format!("VERDICT: {}\n\n", verdict));

        s.push_str("-- count parity --\n");
        s.push_str(&format!(
            "  nodes : v3={} v5={} match={}\n",
            self.v3_node_count, self.v5_node_count, self.node_count_match
        ));
        s.push_str(&format!(
            "  bonds : v3={} v5={} match={}\n",
            self.v3_bond_count, self.v5_bond_count, self.bond_count_match
        ));
        s.push_str(&format!(
            "  nodes missing in v5: {}\n\n",
            self.missing_nodes_in_v5
        ));

        s.push_str("-- recall parity (v5 V5Only vs v3 V3Only) --\n");
        s.push_str(&format!(
            "  subjects sampled : {}\n  subjects matched : {}  ({:.2}%)\n",
            self.recall_subjects_sampled,
            self.recall_subjects_matched,
            self.recall_match_rate() * 100.0
        ));
        s.push_str(&format!(
            "  bonds matched    : {}/{}  ({:.2}%)\n",
            self.recall_objects_matched,
            self.recall_objects_v3_total,
            self.recall_object_match_rate() * 100.0
        ));
        if self.recall_mismatches.is_empty() {
            s.push_str("  mismatches       : none\n\n");
        } else {
            s.push_str(&format!(
                "  mismatches       : {} (showing up to {})\n",
                self.recall_mismatches.len(),
                MAX_RECORDED_MISMATCHES
            ));
            for m in &self.recall_mismatches {
                s.push_str(&format!(
                    "    subject {:?}: only_in_v3={} only_in_v5={}\n",
                    m.subject,
                    m.only_in_v3.len(),
                    m.only_in_v5.len()
                ));
            }
            s.push('\n');
        }

        s.push_str("-- encryption at rest --\n");
        s.push_str(&format!(
            "  node_store SSTs : {}/{} encrypted\n",
            self.node_ssts_encrypted, self.node_ssts_total
        ));
        s.push_str(&format!(
            "  bond_store SSTs : {}/{} encrypted\n",
            self.bond_ssts_encrypted, self.bond_ssts_total
        ));
        s.push_str(&format!(
            "  MANIFEST master_key_id set : {}\n",
            self.manifest_master_key_set
        ));
        if self.plaintext_ssts.is_empty() {
            s.push_str("  plaintext SSTs  : none\n");
        } else {
            s.push_str(&format!(
                "  plaintext SSTs  : {} -- AT-REST ENCRYPTION VIOLATION\n",
                self.plaintext_ssts.len()
            ));
            for p in &self.plaintext_ssts {
                s.push_str(&format!("      {}\n", p.display()));
            }
        }
        s.push_str(&format!("  encryption_ok   : {}\n", self.encryption_ok()));
        s
    }
}

/// Build the `(object, predicate)` set of outgoing bonds a brain serves
/// for `subject` through its CURRENT read-path mode.
fn outgoing_set(brain: &Brain, subject: &Seed) -> Result<HashSet<(Seed, BondType)>, BrainError> {
    let bonds = brain.read_outgoing_bonds(subject)?;
    Ok(bonds.into_iter().map(|b| (b.object, b.predicate)).collect())
}

/// Run the full verification. Loads the v3 source (read-only) and the
/// v5 directory (`load_v5`, `V5Only`), then runs the three checks.
pub fn verify(cfg: &VerifyConfig) -> Result<VerifyReport, BrainError> {
    // --- load v3 (read-only; never written) ---
    let v3 = Brain::load_or_new_with_mode_and_key(
        &cfg.v3_path,
        BrainLoadMode::Auto,
        cfg.key.clone(),
    )?;

    // --- load v5 via the production loader (V5Only) ---
    let mut v5 = Brain::load_v5(&cfg.v5_dir, cfg.key.clone())?;
    // OP-BRAIN-MapRAM: the production loader no longer builds the
    // resident adjacency by default (cache-backed). The verifier needs
    // the full in-RAM adjacency to count bonds + compare recall, so it
    // rehydrates explicitly here (offline migration check, not the
    // RAM-sensitive serving path).
    v5.rehydrate_adjacency()?;
    debug_assert_eq!(v5.read_path_mode(), ReadPathMode::V5Only);

    // ---- count parity ----
    let v3_node_count = v3.network.node_count();
    let v5_node_count = v5.network.node_intern().len();
    let v3_bond_count = v3.network.bond_count();
    // After rehydrate_adjacency, the v5 network holds every live bond the
    // brain will serve -- the authoritative in-RAM total.
    let v5_bond_count = v5.network.bond_count();

    // ---- per-node round-trip + recall parity ----
    let intern = v5.network.node_intern();
    let all_seeds = v3.network.all_seeds();
    let mut missing_nodes_in_v5 = 0usize;

    let mut recall_subjects_sampled = 0usize;
    let mut recall_subjects_matched = 0usize;
    let mut recall_objects_v3_total = 0usize;
    let mut recall_objects_matched = 0usize;
    let mut recall_mismatches: Vec<RecallMismatch> = Vec::new();

    let limit = if cfg.sample_limit == 0 {
        all_seeds.len()
    } else {
        cfg.sample_limit.min(all_seeds.len())
    };

    for s in all_seeds.iter().take(limit) {
        // Node round-trip: present in the v5 intern AND readable from the
        // v5 node store with the same seed.
        match intern.get(s) {
            Some(nid) => match v5.get_node_v5(nid) {
                Ok(Some(node)) if node.seed == *s => {}
                _ => missing_nodes_in_v5 += 1,
            },
            None => missing_nodes_in_v5 += 1,
        }

        // Recall parity.
        let v3_set = outgoing_set(&v3, s)?;
        let v5_set = outgoing_set(&v5, s)?;
        recall_subjects_sampled += 1;
        recall_objects_v3_total += v3_set.len();
        recall_objects_matched += v3_set.intersection(&v5_set).count();
        if v3_set == v5_set {
            recall_subjects_matched += 1;
        } else if recall_mismatches.len() < MAX_RECORDED_MISMATCHES {
            let only_in_v3: Vec<(Seed, BondType)> =
                v3_set.difference(&v5_set).cloned().collect();
            let only_in_v5: Vec<(Seed, BondType)> =
                v5_set.difference(&v3_set).cloned().collect();
            recall_mismatches.push(RecallMismatch {
                subject: *s,
                only_in_v3,
                only_in_v5,
            });
        }
    }

    // ---- encryption-at-rest ----
    let node_scan = crate::brain::scan_sst_encryption(&cfg.v5_dir.join("node_store"))?;
    let bond_scan = crate::brain::scan_sst_encryption(&cfg.v5_dir.join("bond_store"))?;
    let mut plaintext_ssts = Vec::new();
    plaintext_ssts.extend(node_scan.plaintext.iter().cloned());
    plaintext_ssts.extend(bond_scan.plaintext.iter().cloned());

    let manifest_master_key_set = {
        let mpath = cfg.v5_dir.join(crate::manifest::MANIFEST_FILE_NAME);
        match crate::manifest::Manifest::load(&mpath)? {
            Some(m) => m.master_key_id != crate::manifest::KeyFingerprint::none(),
            None => false,
        }
    };

    Ok(VerifyReport {
        v3_node_count,
        v5_node_count,
        node_count_match: v3_node_count == v5_node_count,
        v3_bond_count,
        v5_bond_count,
        bond_count_match: v3_bond_count == v5_bond_count,
        missing_nodes_in_v5,
        recall_subjects_sampled,
        recall_subjects_matched,
        recall_objects_v3_total,
        recall_objects_matched,
        recall_mismatches,
        node_ssts_total: node_scan.total,
        node_ssts_encrypted: node_scan.encrypted,
        bond_ssts_total: bond_scan.total,
        bond_ssts_encrypted: bond_scan.encrypted,
        plaintext_ssts,
        manifest_master_key_set,
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;
    use crate::bond::{KeyframeSource, TemporalBond, TemporalKeyframe};
    use crate::brain::BrainSaveFormat;
    use crate::migration_v3_v5_bond as bond_mig;
    use crate::migration_v3_v5_node as node_mig;
    use crate::node::{Confidence, KnowledgeContent, KnowledgeNode};
    use tempfile::tempdir;

    const BTS: [BondType; 4] = [
        BondType::IsA,
        BondType::HasProperty,
        BondType::RelatedTo,
        BondType::Causes,
    ];

    /// Deterministic 60-node graph: i -> i+1, i+2, i+3 (mod n). When
    /// `extra_bond` is set, add ONE perturbing edge (0 -> 30 RelatedTo)
    /// that the base graph does NOT contain, to force a divergence.
    /// Encrypts when `key` is `Some`.
    fn build_v3(
        path: &Path,
        key: Option<&MasterKey>,
        fmt: BrainSaveFormat,
        extra_bond: bool,
    ) -> Vec<Seed> {
        let mut brain = Brain::new();
        let n = 60usize;
        let mut seeds = Vec::with_capacity(n);
        for i in 0..n {
            let label = format!("verify{:04}", i);
            let mut node = KnowledgeNode::new(
                label.clone(),
                KnowledgeContent::Concept(label.clone()),
                Confidence::Verified,
                0,
            );
            node.seed = Seed::from_label(&label);
            seeds.push(node.seed);
            brain.network.add_node(node);
        }
        for i in 0..n {
            for step in 1..=3usize {
                let j = (i + step) % n;
                if i == j {
                    continue;
                }
                let bt = BTS[(i + step) % BTS.len()];
                brain
                    .network
                    .add_bond(TemporalBond {
                        from: seeds[i],
                        to: seeds[j],
                        bond_type: bt,
                        keyframes: vec![TemporalKeyframe {
                            timestamp: (i * 8 + step) as i64,
                            weight: 0.5,
                            source: KeyframeSource::Baseline,
                        }],
                        half_life_ms: 86_400_000.0,
                    })
                    .unwrap();
            }
        }
        if extra_bond {
            brain
                .network
                .add_bond(TemporalBond {
                    from: seeds[0],
                    to: seeds[30],
                    bond_type: BondType::RelatedTo,
                    keyframes: vec![TemporalKeyframe {
                        timestamp: 999,
                        weight: 0.5,
                        source: KeyframeSource::Baseline,
                    }],
                    half_life_ms: 86_400_000.0,
                })
                .unwrap();
        }
        if let Some(k) = key {
            brain.encryption_key = Some(k.clone());
        }
        brain.save_to_format(path, fmt).unwrap();
        seeds
    }

    /// Run BOTH migration tools (node then bond) with `key`; assert each
    /// tool's own GO/NO-GO gate passes.
    fn migrate(v3: &Path, v5: &Path, key: Option<&MasterKey>) {
        let mut ncfg = node_mig::MigrationConfig::new(v3.to_path_buf(), v5.to_path_buf());
        ncfg.key = key.cloned();
        let nstats = node_mig::migrate(&ncfg).expect("node migration succeeded");
        assert!(
            nstats.verification.expect("node verify ran").is_ok(),
            "node migration verify must pass"
        );
        let mut bcfg = bond_mig::MigrationConfig::new(v3.to_path_buf(), v5.to_path_buf());
        bcfg.key = key.cloned();
        let bstats = bond_mig::migrate(&bcfg).expect("bond migration succeeded");
        assert!(
            bstats.verification.expect("bond verify ran").is_ok(),
            "bond migration verify must pass"
        );
    }

    /// (1) A correctly-migrated ENCRYPTED brain PASSES every gate.
    #[test]
    fn verify_passes_on_good_encrypted_migration() {
        let tmp = tempdir().unwrap();
        let v3 = tmp.path().join("brain.lbr");
        let v5 = tmp.path().join("brain-v5");
        let key = MasterKey::from_bytes([0x33u8; 32]);
        build_v3(&v3, Some(&key), BrainSaveFormat::V3BinaryEncrypted, false);
        migrate(&v3, &v5, Some(&key));

        let mut cfg = VerifyConfig::new(v3, v5);
        cfg.key = Some(key);
        let r = verify(&cfg).expect("verify ran");
        assert!(r.passed(), "clean migration must PASS:\n{}", r.render());
        assert!(r.node_count_match);
        assert!(r.bond_count_match);
        assert_eq!(r.missing_nodes_in_v5, 0);
        assert_eq!(r.recall_subjects_matched, r.recall_subjects_sampled);
        assert!(r.recall_subjects_sampled >= 50, "sampled enough subjects");
        assert!((r.recall_match_rate() - 1.0).abs() < 1e-9);
        assert!(r.recall_mismatches.is_empty());
        assert!(r.encryption_ok());
        assert!(r.plaintext_ssts.is_empty());
        assert!(r.manifest_master_key_set);
        assert!(r.node_ssts_total >= 1 && r.bond_ssts_total >= 1);
        assert_eq!(r.node_ssts_encrypted, r.node_ssts_total);
        assert_eq!(r.bond_ssts_encrypted, r.bond_ssts_total);
    }

    /// (2) The verifier CATCHES a data-divergent migration: migrate the
    /// GOOD graph, then verify the v5 against a v3 carrying ONE extra
    /// bond. Must NO-GO via bond-count + recall mismatch (encryption is
    /// still fine -- proving the failure is data parity, not at-rest).
    #[test]
    fn verify_fails_on_bond_mismatch() {
        let tmp = tempdir().unwrap();
        let v3_good = tmp.path().join("good.lbr");
        let v3_bad = tmp.path().join("bad.lbr");
        let v5 = tmp.path().join("brain-v5");
        let key = MasterKey::from_bytes([0x44u8; 32]);
        build_v3(&v3_good, Some(&key), BrainSaveFormat::V3BinaryEncrypted, false);
        migrate(&v3_good, &v5, Some(&key));
        build_v3(&v3_bad, Some(&key), BrainSaveFormat::V3BinaryEncrypted, true);

        let mut cfg = VerifyConfig::new(v3_bad, v5);
        cfg.key = Some(key);
        let r = verify(&cfg).expect("verify ran");
        assert!(
            !r.passed(),
            "a mismatched migration must NO-GO:\n{}",
            r.render()
        );
        assert!(!r.bond_count_match, "bond count must diverge");
        assert_eq!(
            r.v3_bond_count,
            r.v5_bond_count + 1,
            "v3 has exactly the one extra bond"
        );
        assert!(
            !r.recall_mismatches.is_empty(),
            "recall parity must catch the extra bond"
        );
        assert!(
            r.recall_mismatches.iter().any(|m| !m.only_in_v3.is_empty()),
            "the extra bond is present in v3 only"
        );
        // node set is identical -> the only failure is data parity.
        assert!(r.node_count_match);
        assert_eq!(r.missing_nodes_in_v5, 0);
        assert!(r.encryption_ok(), "encryption is unaffected");
    }

    /// (3) The verifier CATCHES a plaintext-at-rest migration: migrate
    /// with NO key -> plaintext v5. Data is intact (counts + recall
    /// match) but the at-rest gate must NO-GO and name the plaintext
    /// SSTs -- proving the verifier fails for the RIGHT reason.
    #[test]
    fn verify_fails_on_plaintext_v5() {
        let tmp = tempdir().unwrap();
        let v3 = tmp.path().join("brain.lbr");
        let v5 = tmp.path().join("brain-v5");
        build_v3(&v3, None, BrainSaveFormat::V3BinaryCompressed, false);
        migrate(&v3, &v5, None);

        let cfg = VerifyConfig::new(v3, v5); // key = None
        let r = verify(&cfg).expect("verify ran");
        assert!(
            !r.passed(),
            "a plaintext v5 must NO-GO the at-rest gate:\n{}",
            r.render()
        );
        assert!(!r.encryption_ok());
        assert!(
            !r.plaintext_ssts.is_empty(),
            "plaintext SSTs must be listed"
        );
        assert!(
            !r.manifest_master_key_set,
            "a keyless migration leaves master_key_id unset"
        );
        // data is intact -- isolates the failure to encryption.
        assert!(r.node_count_match);
        assert!(r.bond_count_match);
        assert_eq!(r.missing_nodes_in_v5, 0);
        assert!(r.recall_mismatches.is_empty());
    }
}
#[cfg(test)]
mod banner_tests_6_6 {
    use crate::brain::{Brain, BrainSaveFormat};
    use crate::encrypt::MasterKey;
    use crate::migration_v3_v5_bond as bond_mig;
    use crate::migration_v3_v5_node as node_mig;
    use crate::read_path::ReadPathMode;
    use std::path::Path;
    use tempfile::tempdir;

    /// 6.6 (unit): a brain whose read-path is V5Only labels itself "v5";
    /// holding a master key makes it report encrypted-at-rest; the banner
    /// string carries format + encryption + require-encrypted at a glance.
    /// Also proves the PLAINTEXT/off rendering for the no-key case.
    #[test]
    fn banner_reports_v5_and_encryption_state() {
        let mut b = Brain::new();
        b.set_read_path_mode(ReadPathMode::V5Only);
        b.encryption_key = Some(MasterKey::from_bytes([0x77u8; 32]));
        assert_eq!(b.brain_format_label(), "v5");
        assert!(b.encrypted_at_rest());
        let line = b.operator_status_line(true);
        assert!(line.contains("format=v5"), "{}", line);
        assert!(line.contains("encryption=encrypted-at-rest"), "{}", line);
        assert!(line.contains("require_encrypted=on"), "{}", line);

        let mut p = Brain::new();
        p.set_read_path_mode(ReadPathMode::V5Only);
        assert!(!p.encrypted_at_rest());
        let pl = p.operator_status_line(false);
        assert!(pl.contains("format=v5"), "{}", pl);
        assert!(pl.contains("encryption=PLAINTEXT"), "{}", pl);
        assert!(pl.contains("require_encrypted=off"), "{}", pl);
    }

    fn build_v3_enc(path: &Path, key: &MasterKey) {
        use crate::bond::{KeyframeSource, TemporalBond, TemporalKeyframe, BondType};
        use crate::node::{Confidence, KnowledgeContent, KnowledgeNode};
        use crate::seed::Seed;
        let mut brain = Brain::new();
        let n = 40usize;
        let mut seeds = Vec::with_capacity(n);
        for i in 0..n {
            let label = format!("banner{:04}", i);
            let mut node = KnowledgeNode::new(
                label.clone(),
                KnowledgeContent::Concept(label.clone()),
                Confidence::Verified,
                0,
            );
            node.seed = Seed::from_label(&label);
            seeds.push(node.seed);
            brain.network.add_node(node);
        }
        for i in 0..n {
            let j = (i + 1) % n;
            brain
                .network
                .add_bond(TemporalBond {
                    from: seeds[i],
                    to: seeds[j],
                    bond_type: BondType::RelatedTo,
                    keyframes: vec![TemporalKeyframe {
                        timestamp: i as i64,
                        weight: 0.5,
                        source: KeyframeSource::Baseline,
                    }],
                    half_life_ms: 86_400_000.0,
                })
                .unwrap();
        }
        brain.encryption_key = Some(key.clone());
        brain
            .save_to_format(path, BrainSaveFormat::V3BinaryEncrypted)
            .unwrap();
    }

    /// 6.6 (real load): a genuinely migrated, encrypted v5 brain booted
    /// through `Brain::load_v5` banners format=v5 + encrypted-at-rest --
    /// the exact line an operator sees at production startup.
    #[test]
    fn real_v5_load_banner_reports_v5_encrypted() {
        let tmp = tempdir().unwrap();
        let v3 = tmp.path().join("brain.lbr");
        let v5 = tmp.path().join("brain-v5");
        let key = MasterKey::from_bytes([0x5Eu8; 32]);
        build_v3_enc(&v3, &key);
        let mut ncfg = node_mig::MigrationConfig::new(v3.clone(), v5.clone());
        ncfg.key = Some(key.clone());
        assert!(node_mig::migrate(&ncfg).unwrap().verification.unwrap().is_ok());
        let mut bcfg = bond_mig::MigrationConfig::new(v3.clone(), v5.clone());
        bcfg.key = Some(key.clone());
        assert!(bond_mig::migrate(&bcfg).unwrap().verification.unwrap().is_ok());

        let brain = Brain::load_v5(&v5, Some(key)).expect("load_v5 boots encrypted v5");
        assert_eq!(brain.brain_format_label(), "v5");
        assert!(brain.encrypted_at_rest());
        let line = brain.operator_status_line(true);
        assert!(
            line.contains("format=v5") && line.contains("encryption=encrypted-at-rest"),
            "banner must report v5 + encrypted: {}",
            line
        );
    }
}
