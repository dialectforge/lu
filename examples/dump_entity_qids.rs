//! OP-FAME: enumerate the QIDs Lu actually holds (entity shelf).
//! Read-only. Iterates every interned node, extracts a trailing
//! "#Q<digits>" from its label, emits one "qid<TAB>label" line per
//! distinct QID (first-seen label).
//!
//! usage: dump_entity_qids <brain_dir> <keyfile> <out_tsv>
use lu4d::encrypt::MasterKey;
use lu4d::node_intern::NodeId;
use lu4d::Brain;
use std::collections::HashMap;
use std::io::Write;
use std::path::Path;

fn qid_of(content: &str) -> Option<u64> {
    let h = content.rfind('#')?;
    let tail = &content[h + 1..];
    if tail.len() > 1
        && (tail.as_bytes()[0] == b'Q' || tail.as_bytes()[0] == b'q')
        && tail[1..].bytes().all(|b| b.is_ascii_digit())
    {
        tail[1..].parse::<u64>().ok()
    } else {
        None
    }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let brain_dir = Path::new(&a[1]);
    let keyfile = Path::new(&a[2]);
    let out = &a[3];
    let key = MasterKey::load_from_keyfile(keyfile).expect("load key");
    let brain = Brain::load_readonly(brain_dir, Some(key)).expect("load_readonly");
    assert!(brain.is_read_only_reader());
    let n = brain.network.node_intern().len();
    eprintln!("interned_nodes={}", n);
    let mut map: HashMap<u64, String> = HashMap::new();
    let mut scanned = 0u64;
    for id in 0..n as u64 {
        if let Some(seed) = brain.network.node_intern().reverse_lookup(NodeId(id)) {
            scanned += 1;
            if let Some(label) = brain.network.node_label(&seed) {
                if let Some(q) = qid_of(&label) {
                    let lp = match label.rfind('#') { Some(h) => label[..h].to_string(), None => label.clone() };
                    map.entry(q).or_insert(lp);
                }
            }
        }
        if id % 100000 == 0 && id > 0 { eprintln!("  progress id={} qids={}", id, map.len()); }
    }
    let mut f = std::fs::File::create(out).expect("create out");
    let mut keys: Vec<u64> = map.keys().copied().collect();
    keys.sort_unstable();
    for q in &keys {
        writeln!(f, "{}\t{}", q, map[q]).unwrap();
    }
    eprintln!("scanned={} distinct_qids={}", scanned, keys.len());
}
