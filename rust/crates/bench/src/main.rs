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
    #[serde(default)]
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

    // A corpus language is a directory with a manifest; other directories (docs,
    // scratch output, …) are not benchmark targets and are skipped silently.
    let mut languages: Vec<String> = std::fs::read_dir(&corpus_dir)
        .unwrap_or_else(|e| {
            panic!(
                "cannot read corpus dir {} (set TOKZIP_CORPUS_DIR or clone ../tokzip-corpus): {e}",
                corpus_dir.display()
            )
        })
        .map(|e| e.expect("read corpus dir entry"))
        .filter(|e| e.path().join("manifest.jsonl").is_file())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    languages.sort();

    let requested: Vec<String> = std::env::args().skip(1).collect();
    let unknown: Vec<&String> = requested
        .iter()
        .filter(|r| !languages.contains(r))
        .collect();
    assert!(
        unknown.is_empty(),
        "unknown language(s) {unknown:?}; available: {languages:?}"
    );
    if !requested.is_empty() {
        languages.retain(|l| requested.contains(l));
    }

    let mut totals: BTreeMap<&str, Acc> =
        BUCKETS.iter().map(|(n, _)| (*n, Acc::default())).collect();

    for lang in &languages {
        let manifest_path = corpus_dir.join(lang).join("manifest.jsonl");
        let manifest = std::fs::read_to_string(&manifest_path)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", manifest_path.display()));
        let entries: Vec<ManifestEntry> = manifest
            .lines()
            .filter(|l| !l.is_empty())
            .map(|l| serde_json::from_str(l).expect("manifest entry"))
            .collect();

        let train_files: Vec<PathBuf> = entries
            .iter()
            .filter(|e| e.split == "train" && e.trainable)
            .map(|e| corpus_dir.join(lang).join(&e.file))
            .collect();
        let dict_path = dict_dir.join(format!(
            "{lang}-{}.dict",
            train_fingerprint(&corpus_dir, &train_files)
        ));
        if !dict_path.exists() {
            let status = Command::new("zstd")
                .arg("--train")
                .args(&train_files)
                .arg("-o")
                .arg(&dict_path)
                .arg("-q")
                .status()
                .expect("run zstd --train (is the zstd CLI installed?)");
            assert!(status.success(), "zstd --train failed for {lang}");
        }
        let dict = std::fs::read(&dict_path).expect("read dictionary");
        // Digesting a level-19 dictionary costs ~5ms; per-document construction would
        // dominate the whole run, so build the contexts once per language.
        let mut dict_compressor =
            zstd::bulk::Compressor::with_dictionary(19, &dict).expect("zstd compressor");
        let mut dict_decompressor =
            zstd::bulk::Decompressor::with_dictionary(&dict).expect("zstd decompressor");

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

            let with_dict = dict_compressor.compress(&raw).expect("zstd compress");
            let restored = dict_decompressor
                .decompress(&with_dict, raw.len() + 1)
                .expect("zstd decompress");
            assert_eq!(restored, raw, "zstd+dict round-trip mismatch");
            sizes.insert("zstd19+dict", with_dict.len() as u64);

            sizes.insert("zstd19", zstd_round_trip_nodict(&raw));

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
            .map(|m| format!("{m} {}", ratio(&lang_acc, m)))
            .collect();
        println!(
            "{lang:<11} docs={:>3} raw={:>5}KB dict={}KB  {}",
            lang_acc.docs,
            lang_acc.raw / 1024,
            dict.len() / 1024,
            cells.join("  ")
        );
    }

    println!("\n=== Totals (compressed/raw %, lower is better) ===");
    for (name, _) in BUCKETS {
        let acc = &totals[name];
        println!("\n[{name}] docs={} raw={}KB", acc.docs, acc.raw / 1024);
        for m in METHODS {
            println!(
                "  {m:<13} {:>6}  wins={}",
                ratio(acc, m),
                acc.wins.get(m).copied().unwrap_or(0)
            );
        }
    }
}

/// Compressed/raw percentage, or `n/a` for an empty accumulator.
fn ratio(acc: &Acc, method: &str) -> String {
    if acc.raw == 0 {
        return "n/a".to_string();
    }
    let size = acc.sizes.get(method).copied().unwrap_or(0);
    format!("{:.2}%", 100.0 * size as f64 / acc.raw as f64)
}

/// Cache key for a trained dictionary: covers the corpus location and the exact
/// train-file contents, so switching `TOKZIP_CORPUS_DIR` or updating the corpus
/// retrains instead of silently reusing a dictionary from different data.
fn train_fingerprint(corpus_dir: &Path, train_files: &[PathBuf]) -> String {
    let mut hasher = crc32fast::Hasher::new();
    let canonical = corpus_dir
        .canonicalize()
        .unwrap_or_else(|_| corpus_dir.to_path_buf());
    hasher.update(canonical.to_string_lossy().as_bytes());
    for file in train_files {
        hasher.update(file.to_string_lossy().as_bytes());
        hasher.update(&std::fs::read(file).expect("read train doc"));
    }
    format!("{:08x}", hasher.finalize())
}

fn zstd_round_trip_nodict(raw: &[u8]) -> u64 {
    let compressed = zstd::bulk::compress(raw, 19).expect("zstd compress");
    let restored = zstd::bulk::decompress(&compressed, raw.len() + 1).expect("zstd decompress");
    assert_eq!(restored, raw, "zstd round-trip mismatch");
    compressed.len() as u64
}
