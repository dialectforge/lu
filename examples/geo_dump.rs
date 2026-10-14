//! Dump detected primitives (raw raster) with coordinates.
use lu4d::vision::geo_primitives::{detect_on_raster, DetectResult, Primitive};
use lu4d::vision::geo_perceive_p4::decode_image_rgb;
use lu4d::vision::geo_labels::read_tokens_on_raster;

fn main() {
    let path = std::env::args().nth(1).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    let img = decode_image_rgb(&bytes).expect("decode");
    println!("decoded {}x{}", img.w, img.h);
    match detect_on_raster(&img, false) {
        DetectResult::Primitives(p) => {
            println!("CENSUS: {}", p.census());
            for (i, it) in p.items.iter().enumerate() {
                match it {
                    Primitive::Segment { a, b, confidence, support } => println!(
                        "  [{}] SEG ({:.3},{:.3})-({:.3},{:.3}) len={:.3} conf={:.2} sup={}",
                        i, a.0, a.1, b.0, b.1,
                        ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt(), confidence, support),
                    Primitive::Polygon { verts, confidence } => println!(
                        "  [{}] POLY n={} conf={:.2} verts={:?}", i, verts.len(), confidence,
                        verts.iter().map(|v| ((v.0*1000.0).round()/1000.0, (v.1*1000.0).round()/1000.0)).collect::<Vec<_>>()),
                    Primitive::Arc { center, r, confidence, support, .. } => println!(
                        "  [{}] ARC c=({:.3},{:.3}) r={:.3} conf={:.2} sup={}", i, center.0, center.1, r, confidence, support),
                    Primitive::Circle { center, r, confidence, support } => println!(
                        "  [{}] CIRCLE c=({:.3},{:.3}) r={:.3} conf={:.2} sup={}", i, center.0, center.1, r, confidence, support),
                    Primitive::Point { p, confidence } => println!("  [{}] PT ({:.3},{:.3}) conf={:.2}", i, p.0, p.1, confidence),
                    Primitive::RightAngle { at, confidence } => println!("  [{}] RA ({:.3},{:.3}) conf={:.2}", i, at.0, at.1, confidence),
                }
            }
        }
        DetectResult::Incomplete { stage, reason } => println!("INCOMPLETE {} {}", stage, reason),
    }
    println!("--- OCR tokens ---");
    for t in read_tokens_on_raster(&img) {
        println!("  {:?}", t);
    }
}
