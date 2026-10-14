//! End-to-end perception probe: read a PNG file + optional integer side, run
//! the REAL perceiver (public API only), print the outcome as JSON. Used to
//! demonstrate that pixels -> scene -> exact answer flows for images the code
//! has never seen (no filename/hash lookup exists on the path).
//!
//!   cargo run -p lu4d --example geo_probe -- <png-path> [side]
use lu4d::vision::geo_perceive::perceive_red_leaf_png;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: geo_probe <png-path> [side]");
        std::process::exit(2);
    }
    let path = &args[1];
    let side: Option<i64> = args.get(2).and_then(|s| s.parse().ok());
    let bytes = std::fs::read(path).unwrap_or_else(|e| {
        eprintln!("read {}: {}", path, e);
        std::process::exit(2);
    });
    let out = perceive_red_leaf_png(&bytes, side);
    println!("INPUT: {}  side_hint={:?}  bytes={}", path, side, bytes.len());
    println!("{}", serde_json::to_string_pretty(&out).unwrap());
}
