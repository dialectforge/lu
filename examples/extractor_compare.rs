//! Read an article from a file path (arg 1) with title (arg 2),
//! run the Rust extractor, dump per-layer bond counts + every bond
//! as TSV on stdout. Used by the text-extractor v2 comparison
//! harness to cross-check parity against the Python source of
//! truth on the full Anarchism article.

use lu4d::extractor::extract_from_article;
use std::collections::BTreeMap;
use std::io::Read;

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("path arg required");
    let title = args.next().unwrap_or_else(|| {
        std::path::Path::new(&path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("Article")
            .to_string()
    });
    let mut text = String::new();
    if path == "-" {
        std::io::stdin().read_to_string(&mut text).expect("read stdin");
    } else {
        text = std::fs::read_to_string(&path).expect("read file");
    }
    let bonds = extract_from_article(&text, Some(&title), None);
    let mut by_layer: BTreeMap<String, usize> = BTreeMap::new();
    for b in &bonds {
        *by_layer.entry(b.layer.to_string()).or_insert(0) += 1;
    }
    println!("# total {}", bonds.len());
    for (layer, n) in &by_layer {
        println!("# layer\t{}\t{}", layer, n);
    }
    println!("# ---bonds---");
    for b in &bonds {
        println!(
            "{layer}\t{subj}\t{rel:?}\t{obj}",
            layer = b.layer,
            subj = b.subject,
            rel = b.relation,
            obj = b.object,
        );
    }
}
