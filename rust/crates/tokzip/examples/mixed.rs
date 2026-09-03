//! Mixed-language documents (prose + fenced code + HTML, assembled from corpus bench docs):
//! automatic segmentation vs. an oracle that codes each part with its own language.
//!
//!   cargo run --release --features train --example mixed [-- <corpus dir>]

use std::path::{Path, PathBuf};
use std::time::Instant;

fn bench_docs(corpus: &Path, lang: &str, max: usize) -> Vec<Vec<u8>> {
    tokzip::train::bench_docs(corpus, lang)
        .filter(|d| d.len() >= 300 && d.len() <= 6000)
        .take(max)
        .collect()
}

fn main() {
    let corpus = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../../tokzip-corpus/corpus")
        });
    let ja = bench_docs(&corpus, "ja-JP", 60);
    let js = bench_docs(&corpus, "javascript", 60);
    let html = bench_docs(&corpus, "html", 60);
    let en = bench_docs(&corpus, "en-US", 60);
    let mut raw = 0usize;
    let mut auto = 0usize;
    let mut oracle = 0usize;
    let mut single = 0usize;
    let n = ja.len().min(js.len()).min(html.len()).min(en.len());
    for i in 0..n {
        // LLM-answer shape: prose, a labeled code fence, an HTML block, closing prose.
        let parts: [(&[u8], usize); 5] = [
            (&ja[i], 2),
            (b"\n```js\n", 5),
            (&js[i], 5),
            (b"\n```\n\n", 2),
            (&en[i], 1),
        ];
        let mut doc = Vec::new();
        let mut oracle_frames = 0usize;
        for (bytes, lang) in parts {
            doc.extend_from_slice(bytes);
            oracle_frames += tokzip::frame_len_with_language(bytes, lang);
        }
        let mut doc2 = doc.clone();
        doc2.extend_from_slice(&html[i]);
        oracle_frames += tokzip::frame_len_with_language(&html[i], 3);
        raw += doc2.len();
        auto += tokzip::compress(&doc2).len();
        oracle += oracle_frames;
        single += (0..7)
            .map(|lang| tokzip::frame_len_with_language(&doc2, lang))
            .min()
            .unwrap();
    }
    println!("mixed docs: {n}, raw {raw} B");
    println!("auto segments: {:.1}%", 100.0 * auto as f64 / raw as f64);
    println!(
        "best single language: {:.1}%",
        100.0 * single as f64 / raw as f64
    );
    println!(
        "oracle (parts coded separately, framing counted per part): {:.1}%",
        100.0 * oracle as f64 / raw as f64
    );

    // Compression time and ratio by input size: every input is an equal share of each
    // language's bench documents, so a size never measures one language only.
    const TOTAL: usize = 1 << 20;
    let langs = [
        "html",
        "javascript",
        "en-US",
        "text",
        "ja-JP",
        "python",
        "typescript",
    ];
    let shares: Vec<Vec<u8>> = langs
        .iter()
        .map(|lang| {
            let mut share = Vec::new();
            for doc in tokzip::train::bench_docs(&corpus, lang) {
                if share.len() >= TOTAL / langs.len() {
                    break;
                }
                share.extend_from_slice(&doc);
            }
            share.truncate(TOTAL / langs.len());
            share
        })
        .collect();
    for size in [1024usize, 4096, 16384, 65536, 262_144, 1 << 20] {
        let doc: Vec<u8> = shares
            .iter()
            .flat_map(|share| &share[..(size / langs.len()).min(share.len())])
            .copied()
            .collect();
        let doc = doc.as_slice();
        let t = Instant::now();
        let iters = (2_000_000 / size).max(3);
        let mut coded = 0;
        for _ in 0..iters {
            coded = std::hint::black_box(tokzip::compress(doc)).len();
        }
        let per = t.elapsed().as_secs_f64() * 1000.0 / iters as f64;
        println!(
            "{:>7} B: {:.2} ms/doc ({:.1} MB/s) {:.2}%",
            doc.len(),
            per,
            doc.len() as f64 / per / 1000.0,
            100.0 * coded as f64 / doc.len() as f64
        );
    }
}
