//! Ratio and speed per corpus language directory (public or private checkout), for codec
//! iteration: every argument is a `<corpus root>:<language>` pair.
//!
//!   cargo run --release --features train --example eval -- ../tokzip-corpus/corpus:ja-JP ...

use std::path::Path;
use std::time::Instant;

fn main() {
    if std::env::var_os("TOKZIP_COST").is_some() {
        tokzip::train::enable_cost_report();
    }
    let mut grand = (0usize, 0usize, 0f64, 0f64);
    let mut small = (0usize, 0f64);
    for arg in std::env::args().skip(1) {
        let (root, lang) = arg.split_once(':').expect("<corpus root>:<language>");
        let mut rows = [(0usize, 0usize, 0usize); 4];
        let (mut raw, mut packed, mut comp, mut decomp) = (0usize, 0usize, 0f64, 0f64);
        for content in tokzip::train::bench_docs(Path::new(root), lang) {
            let t = Instant::now();
            let frame = tokzip::compress(&content);
            let c = t.elapsed().as_secs_f64();
            let t = Instant::now();
            let restored = tokzip::decompress(&frame, usize::MAX).expect("decode");
            decomp += t.elapsed().as_secs_f64();
            assert_eq!(restored, content);
            comp += c;
            if content.len() <= 1024 {
                small.0 += 1;
                small.1 += c;
            }
            let bucket = match content.len() {
                0..=1024 => 0,
                1025..=4096 => 1,
                4097..=16384 => 2,
                _ => 3,
            };
            rows[bucket].0 += 1;
            rows[bucket].1 += content.len();
            rows[bucket].2 += frame.len();
            raw += content.len();
            packed += frame.len();
        }
        let pct = |p: usize, r: usize| {
            if r == 0 {
                0.0
            } else {
                100.0 * p as f64 / r as f64
            }
        };
        println!(
            "{lang:<12} {:>7.2}%  <=1K {:>5.1}%  1-4K {:>5.1}%  4-16K {:>5.1}%  >16K {:>5.1}%  ({} docs, {} B)",
            pct(packed, raw),
            pct(rows[0].2, rows[0].1),
            pct(rows[1].2, rows[1].1),
            pct(rows[2].2, rows[2].1),
            pct(rows[3].2, rows[3].1),
            rows.iter().map(|r| r.0).sum::<usize>(),
            raw
        );
        grand.0 += raw;
        grand.1 += packed;
        grand.2 += comp;
        grand.3 += decomp;
    }
    if std::env::var_os("TOKZIP_COST").is_some() {
        let report = tokzip::train::cost_report();
        let names = [
            "flags",
            "len",
            "rep_len",
            "dict_len",
            "hist_dist",
            "dict_off",
            "lit",
            "lit_matched",
            "direct",
        ];
        let total: f64 = report[..9].iter().sum();
        let line: Vec<String> = names
            .iter()
            .zip(report.iter())
            .map(|(n, c)| format!("{n} {:.1}%", 100.0 * c / total))
            .collect();
        println!(
            "cost share over every parse attempted ({:.1}% of the parsed input): {}",
            100.0 * total / report[9],
            line.join("  ")
        );
    }
    let mb = grand.0 as f64 / 1e6;
    println!(
        "ALL          {:>7.2}%  compress {:.1} MB/s  decompress {:.1} MB/s  <=1K docs {:.3} ms each",
        100.0 * grand.1 as f64 / grand.0 as f64,
        mb / grand.2,
        mb / grand.3,
        1000.0 * small.1 / small.0.max(1) as f64
    );
}
