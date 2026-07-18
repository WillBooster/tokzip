//! Corpus benchmark: tokzip-rs vs zstd -19 with a `zstd --train` dictionary.
//!
//! Mirrors the TS-side `.tmp/benchZstdDict.ts` methodology: per corpus language,
//! train (or reuse) a zstd dictionary on the train split, then compare compressed
//! sizes on the bench split, bucketed by raw size (<=1KB / <=4KB / all).
//! Every method must round-trip losslessly or the run fails.
//!
//! Usage: cargo run --release -p bench -- [lang ...]

use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Deserialize)]
struct ManifestEntry {
    file: String,
    split: String,
    #[serde(default)]
    trainable: bool,
}

const METHODS: [&str; 3] = ["tokzip-rs", "zstd19+dict", "zstd19"];
const BUCKETS: [(&str, usize); 3] = [("<=1KB", 1024), ("<=4KB", 4096), ("all", usize::MAX)];

#[derive(Default, Clone)]
struct Acc {
    raw: u64,
    docs: u64,
    sizes: BTreeMap<&'static str, u64>,
    wins: BTreeMap<&'static str, u64>,
}

fn main() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..");
    let corpus_dir = std::env::var("TOKZIP_CORPUS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| repo_root.join("../tokzip-corpus/corpus"));
    let dict_dir = repo_root.join(".tmp/zstd-bench");
    std::fs::create_dir_all(&dict_dir).expect("create dict dir");

    let requested: Vec<String> = std::env::args().skip(1).collect();
    let mut languages: Vec<String> = std::fs::read_dir(&corpus_dir)
        .expect("corpus dir (set TOKZIP_CORPUS_DIR or clone ../tokzip-corpus)")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|l| requested.is_empty() || requested.contains(l))
        .collect();
    languages.sort();

    let mut totals: BTreeMap<&str, Acc> =
        BUCKETS.iter().map(|(n, _)| (*n, Acc::default())).collect();

    for lang in &languages {
        let manifest_path = corpus_dir.join(lang).join("manifest.jsonl");
        let Ok(manifest) = std::fs::read_to_string(&manifest_path) else {
            continue;
        };
        let entries: Vec<ManifestEntry> = manifest
            .lines()
            .filter(|l| !l.is_empty())
            .map(|l| serde_json::from_str(l).expect("manifest entry"))
            .collect();

        let dict_path = dict_dir.join(format!("{lang}.dict"));
        if !dict_path.exists() {
            let train_files: Vec<PathBuf> = entries
                .iter()
                .filter(|e| e.split == "train" && e.trainable)
                .map(|e| corpus_dir.join(lang).join(&e.file))
                .collect();
            let status = Command::new("zstd")
                .arg("--train")
                .args(&train_files)
                .arg("-o")
                .arg(&dict_path)
                .arg("-q")
                .status()
                .expect("run zstd --train");
            assert!(status.success(), "zstd --train failed for {lang}");
        }
        let dict = std::fs::read(&dict_path).expect("read dictionary");

        let mut lang_acc = Acc::default();
        for entry in entries.iter().filter(|e| e.split == "bench") {
            let raw =
                std::fs::read(corpus_dir.join(lang).join(&entry.file)).expect("read bench doc");

            let mut sizes: BTreeMap<&'static str, u64> = BTreeMap::new();

            let frame = tokzip::compress(&raw, Some(&dict));
            assert_eq!(
                tokzip::decompress(&frame, Some(&dict)).expect("tokzip decode"),
                raw
            );
            sizes.insert("tokzip-rs", frame.len() as u64);

            sizes.insert("zstd19+dict", zstd_round_trip(&raw, Some(&dict)));
            sizes.insert("zstd19", zstd_round_trip(&raw, None));

            let best = *sizes.values().min().unwrap();
            let raw_len = raw.len();
            let matching_totals = BUCKETS
                .iter()
                .filter(|(_, max)| raw_len <= *max)
                .map(|(n, _)| *n)
                .collect::<Vec<_>>();
            for acc in std::iter::once(&mut lang_acc).chain(
                totals
                    .iter_mut()
                    .filter(|(n, _)| matching_totals.contains(n))
                    .map(|(_, a)| a),
            ) {
                acc.raw += raw_len as u64;
                acc.docs += 1;
                for (m, s) in &sizes {
                    *acc.sizes.entry(m).or_default() += s;
                    if *s == best {
                        *acc.wins.entry(m).or_default() += 1;
                    }
                }
            }
        }
        if lang_acc.docs == 0 {
            continue;
        }
        let cells: Vec<String> = METHODS
            .iter()
            .map(|m| {
                format!(
                    "{m} {:.1}%",
                    100.0 * lang_acc.sizes[m] as f64 / lang_acc.raw as f64
                )
            })
            .collect();
        println!(
            "{lang:<11} docs={:>3} raw={:>5}KB  {}",
            lang_acc.docs,
            lang_acc.raw / 1024,
            cells.join("  ")
        );
    }

    println!("\n=== Totals (compressed/raw %, lower is better) ===");
    for (name, _) in BUCKETS {
        let acc = &totals[name];
        println!("\n[{name}] docs={} raw={}KB", acc.docs, acc.raw / 1024);
        for m in METHODS {
            println!(
                "  {m:<13} {:>6.2}%  wins={}",
                100.0 * acc.sizes[m] as f64 / acc.raw as f64,
                acc.wins.get(m).copied().unwrap_or(0)
            );
        }
    }
}

fn zstd_round_trip(raw: &[u8], dict: Option<&[u8]>) -> u64 {
    let compressed = match dict {
        Some(d) => zstd::bulk::Compressor::with_dictionary(19, d)
            .expect("zstd compressor")
            .compress(raw)
            .expect("zstd compress"),
        None => zstd::bulk::compress(raw, 19).expect("zstd compress"),
    };
    let restored = match dict {
        Some(d) => zstd::bulk::Decompressor::with_dictionary(d)
            .expect("zstd decompressor")
            .decompress(&compressed, raw.len() + 1)
            .expect("zstd decompress"),
        None => zstd::bulk::decompress(&compressed, raw.len() + 1).expect("zstd decompress"),
    };
    assert_eq!(restored, raw, "zstd round-trip mismatch");
    compressed.len() as u64
}
