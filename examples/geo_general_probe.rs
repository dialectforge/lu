//! Full general-chain probe: decode -> P-B primitives -> P-C labels -> P-D
//! constraints -> algsolve. Prints the primitive census and the solve outcome.
//!   cargo run -p lu4d --example geo_general_probe -- <img-path>
use lu4d::vision::geo_primitives::{detect_primitives, DetectResult};
use lu4d::math::geometry_tool::solve_image_bytes;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 { eprintln!("usage: geo_general_probe <img-path>"); std::process::exit(2); }
    let bytes = std::fs::read(&args[1]).expect("read");
    println!("INPUT {} bytes={}", args[1], bytes.len());
    match detect_primitives(&bytes) {
        DetectResult::Primitives(p) => {
            println!("P-B CENSUS: {}", p.census());
            println!("P-B rectified={} mean_conf={:.3} items={}", p.rectified, p.mean_confidence, p.items.len());
        }
        DetectResult::Incomplete{stage,reason} => println!("P-B INCOMPLETE stage={} reason={}", stage, reason),
    }
    let out = solve_image_bytes(&bytes);
    println!("SOLVE OUTCOME:\n{}", serde_json::to_string_pretty(&out).unwrap());
}
