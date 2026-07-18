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

    // The dictionary is built by the external CLI (the in-process ZDICT builder takes
    // no compression level, so it cannot reproduce the -19 tuning); its version is
    // part of the cache identity so a CLI upgrade retrains instead of reusing.
    let zstd_version_output = Command::new("zstd")
        .arg("--version")
        .output()
        .expect("run zstd --version (is the zstd CLI installed?)");
    let zstd_version = String::from_utf8_lossy(&zstd_version_output.stdout)
        .trim()
        .to_string();

    let mut totals: BTreeMap<&str, Acc> =
        BUCKETS.iter().map(|(n, _)| (*n, Acc::default())).collect();

    for lang in &languages {
        let manifest_path = corpus_dir.join(lang).join("manifest.jsonl");
        let manifest = std::fs::read_to_string(&manifest_path)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", manifest_path.display()));
        let entries: Vec<ManifestEntry> = manifest
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| serde_json::from_str(l).expect("manifest entry"))
            .collect();

        let train_files: Vec<PathBuf> = entries
            .iter()
            .filter(|e| e.split == "train" && e.trainable)
            .map(|e| corpus_dir.join(lang).join(&e.file))
            .collect();
        // `l19` in the cache key: dictionary statistics are tuned for the evaluated
        // compression level (passing `-19` below measurably improves the baseline),
        // so the level is part of the cache identity, as is the trainer's version.
        let dict_path = dict_dir.join(format!(
            "{lang}-l19-{}.dict",
            train_fingerprint(&corpus_dir, &train_files, &zstd_version)
        ));
        if !dict_path.exists() {
            // Train to a process-unique temp path and rename atomically so neither an
            // interrupted run nor two concurrent runs of the same language can leave a
            // partial file at the cache path (a shared temp name would interleave
            // writes and even corrupt the cache after a successful rename).
            let tmp_path = dict_path.with_extension(format!("{}.tmp", std::process::id()));
            let output = Command::new("zstd")
                .arg("--train")
                .args(&train_files)
                .arg("-19")
                .arg("-o")
                .arg(&tmp_path)
                .arg("-f")
                .output()
                .expect("run zstd --train (is the zstd CLI installed?)");
            // No `-q`: it would also silence zstd's dictionary-quality warnings (e.g.
            // "size(source)/size(dictionary) should be >= 10"), which flag languages
            // whose baseline dictionary is knowingly under-trained. Re-emit only those.
            for line in String::from_utf8_lossy(&output.stderr).lines() {
                if line.contains("WARNING") || line.trim_start().starts_with('!') {
                    eprintln!("{lang}: {}", line.trim());
                }
            }
            if !output.status.success() {
                let _ = std::fs::remove_file(&tmp_path);
                panic!("zstd --train failed for {lang}");
            }
            std::fs::rename(&tmp_path, &dict_path).expect("move dictionary into cache");
            // The cache is content-addressed, so old generations accumulate forever;
            // prune this language's superseded entries (only on the train path, so a
            // concurrent cache-hit run never has its dictionary deleted mid-read).
            // `{lang}.dict` (no dash) belongs to the TS harness and is left alone.
            for entry in std::fs::read_dir(&dict_dir)
                .expect("read dict dir")
                .flatten()
            {
                let name = entry.file_name().to_string_lossy().into_owned();
                if name.starts_with(&format!("{lang}-"))
                    && name.ends_with(".dict")
                    && entry.path() != dict_path
                {
                    let _ = std::fs::remove_file(entry.path());
                }
            }
        }
        let dict = std::fs::read(&dict_path).expect("read dictionary");
        // libzstd silently accepts arbitrary bytes as a raw content dictionary, so a
        // corrupt cache entry would degrade ratios without any error — check the magic.
        assert!(
            dict.starts_with(&[0x37, 0xA4, 0x30, 0xEC]),
            "{} is not a trained zstd dictionary; delete it and rerun",
            dict_path.display()
        );
        // Digesting a level-19 dictionary costs ~5ms; per-document construction would
        // dominate the whole run, so build the contexts once per language. Framing is
        // equalized against tokzip's frame: both zstd arms carry a content checksum
        // (tokzip frames always pay their CRC-32), drop the 4-byte dictionary ID (a
        // single-dictionary at-rest deployment would strip it), and drop the frame
        // content size (the tokzip v0 frame encodes no decompressed length either).
        let mut dict_compressor =
            zstd::bulk::Compressor::with_dictionary(19, &dict).expect("zstd compressor");
        let mut plain_compressor = zstd::bulk::Compressor::new(19).expect("zstd compressor");
        for compressor in [&mut dict_compressor, &mut plain_compressor] {
            compressor
                .set_parameter(zstd::zstd_safe::CParameter::ChecksumFlag(true))
                .expect("enable zstd checksum");
            compressor
                .set_parameter(zstd::zstd_safe::CParameter::DictIdFlag(false))
                .expect("disable zstd dict id");
            compressor
                .set_parameter(zstd::zstd_safe::CParameter::ContentSizeFlag(false))
                .expect("disable zstd content size");
        }
        let mut dict_decompressor =
            zstd::bulk::Decompressor::with_dictionary(&dict).expect("zstd decompressor");
        let mut plain_decompressor = zstd::bulk::Decompressor::new().expect("zstd decompressor");

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

            sizes.insert(
                "zstd19+dict",
                zstd_round_trip(&raw, &mut dict_compressor, &mut dict_decompressor),
            );
            sizes.insert(
                "zstd19",
                zstd_round_trip(&raw, &mut plain_compressor, &mut plain_decompressor),
            );

            // Credit the win to the first method in METHODS order that hits the
            // minimum, so `wins` partitions `docs` even when two methods tie.
            let best = *sizes.values().min().unwrap();
            let winner = *METHODS.iter().find(|m| sizes[*m] == best).unwrap();
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
                }
                *acc.wins.entry(winner).or_default() += 1;
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

    assert!(
        totals["all"].docs > 0,
        "no bench documents were benchmarked — check the corpus at {}",
        corpus_dir.display()
    );

    println!("\n=== Totals (compressed/raw %, lower is better) ===");
    for (name, _) in BUCKETS {
        let acc = &totals[name];
        println!("\n[{name}] docs={} raw={}KB", acc.docs, acc.raw / 1024);
        for m in METHODS {
            println!(
                "  {m:<13} {:>7}  wins={}",
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

/// Cache key for a trained dictionary: covers the trainer version, the corpus
/// location, and the exact train-file contents, so a CLI upgrade, switching
/// `TOKZIP_CORPUS_DIR`, or updating the corpus retrains instead of silently
/// reusing a dictionary from different data.
/// Train files are hashed relative to the canonicalized corpus dir, so the same
/// corpus reached through a symlink or differently-spelled path fingerprints
/// identically. Every field is length-prefixed so boundary shifts between path
/// and content cannot collide; 32 bits of identity is plenty for a local cache
/// of ~21 entries.
fn train_fingerprint(corpus_dir: &Path, train_files: &[PathBuf], zstd_version: &str) -> String {
    let mut hasher = crc32fast::Hasher::new();
    let canonical = corpus_dir
        .canonicalize()
        .unwrap_or_else(|_| corpus_dir.to_path_buf());
    let mut update = |bytes: &[u8]| {
        hasher.update(&(bytes.len() as u64).to_le_bytes());
        hasher.update(bytes);
    };
    update(zstd_version.as_bytes());
    update(canonical.to_string_lossy().as_bytes());
    for file in train_files {
        let relative = file.strip_prefix(corpus_dir).unwrap_or(file);
        update(relative.to_string_lossy().as_bytes());
        update(&std::fs::read(file).expect("read train doc"));
    }
    format!("{:08x}", hasher.finalize())
}

fn zstd_round_trip(
    raw: &[u8],
    compressor: &mut zstd::bulk::Compressor,
    decompressor: &mut zstd::bulk::Decompressor,
) -> u64 {
    let compressed = compressor.compress(raw).expect("zstd compress");
    let restored = decompressor
        .decompress(&compressed, raw.len() + 1)
        .expect("zstd decompress");
    assert_eq!(restored, raw, "zstd round-trip mismatch");
    compressed.len() as u64
}
