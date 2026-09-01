//! Native corpus harness for codec iteration: ratio per language / size bucket and
//! throughput over the bench split of the sibling tokzip-corpus checkout.
//!
//!   cargo run --release --features train --example corpus [-- <corpus dir>]

use std::path::{Path, PathBuf};
use std::time::Instant;

const LANGS: [&str; 9] = [
    "text",
    "en-US",
    "ja-JP",
    "html",
    "css",
    "javascript",
    "typescript",
    "python",
    "java",
];

fn main() {
    let corpus = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../../tokzip-corpus/corpus")
        });
    let mut rows: Vec<(String, usize, usize, usize)> = Vec::new(); // key, docs, raw, packed
    let mut total_raw = 0usize;
    let mut comp_time = 0f64;
    let mut decomp_time = 0f64;
    let mut docs = Vec::new();
    for lang in LANGS {
        docs.extend(tokzip::train::bench_docs(&corpus, lang).map(|content| (lang, content)));
    }
    for (lang, content) in &docs {
        let t = Instant::now();
        let frame = tokzip::compress(content, false);
        comp_time += t.elapsed().as_secs_f64();
        let t = Instant::now();
        let (restored, _) = tokzip::decompress(&frame, usize::MAX).expect("decode");
        decomp_time += t.elapsed().as_secs_f64();
        assert_eq!(&restored, content);
        let bucket = match content.len() {
            0..=1024 => "bucket <=1K",
            1025..=4096 => "bucket 1-4K",
            4097..=16384 => "bucket 4-16K",
            _ => "bucket >16K",
        };
        total_raw += content.len();
        for key in [lang.to_string(), bucket.to_string(), "ALL".to_string()] {
            match rows.iter_mut().find(|r| r.0 == key) {
                Some(r) => {
                    r.1 += 1;
                    r.2 += content.len();
                    r.3 += frame.len();
                }
                None => rows.push((key, 1, content.len(), frame.len())),
            }
        }
    }
    rows.sort();
    for (key, n, raw, packed) in &rows {
        println!(
            "{key:<14}{n:>6}{raw:>10}  {:>6.1}%",
            100.0 * *packed as f64 / *raw as f64
        );
    }
    let mb = total_raw as f64 / 1e6;
    println!(
        "speed MB/s: compress {:.1}  decompress {:.1}",
        mb / comp_time,
        mb / decomp_time
    );
}
