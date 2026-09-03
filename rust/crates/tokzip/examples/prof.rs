//! Per-step timing on the small documents of the bench split (codec iteration aid).
use std::path::Path;
use std::time::Instant;
fn main() {
    let mut docs = Vec::new();
    for arg in std::env::args().skip(1) {
        let (root, lang) = arg.split_once(':').expect("<corpus root>:<language>");
        docs.extend(tokzip::train::bench_docs(Path::new(root), lang).filter(|d| d.len() <= 1024 && !d.is_empty()));
    }
    let n = docs.len() as f64;
    let time = |f: &dyn Fn(&[u8])| {
        let t = Instant::now();
        for d in &docs { f(d); }
        1e6 * t.elapsed().as_secs_f64() / n
    };
    println!("{} docs", docs.len());
    println!("segments   {:>8.1} us", time(&|d| { tokzip::segments(d); }));
    println!("parse(0)   {:>8.1} us", time(&|d| { tokzip::frame_len_with_language(d, 0); }));
    println!("compress   {:>8.1} us", time(&|d| { tokzip::compress(d); }));
    let frames: Vec<Vec<u8>> = docs.iter().map(|d| tokzip::compress(d)).collect();
    let t = Instant::now();
    for f in &frames { tokzip::decompress(f, usize::MAX).unwrap(); }
    println!("decompress {:>8.1} us", 1e6 * t.elapsed().as_secs_f64() / n);
}
