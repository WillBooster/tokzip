//! Trains `priors/<language>.bin` for every embedded language from the corpus train split.
//!
//!   cargo run --release --features train --bin train-priors [-- <corpus dir>]

use std::path::{Path, PathBuf};

/// Per-language training input bound; keeps a full run to a few seconds per language.
const MAX_TRAIN_BYTES: usize = 6 * 1024 * 1024;

fn main() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..");
    let corpus = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| root.join("../tokzip-corpus/corpus"));
    for (lang, name) in tokzip::train::languages().into_iter().enumerate() {
        let mut docs = Vec::new();
        let mut total = 0usize;
        for content in tokzip::train::train_docs(&corpus, name) {
            if total >= MAX_TRAIN_BYTES {
                break;
            }
            total += content.len();
            docs.push(content);
        }
        let priors = tokzip::train::train_priors(lang as u8, &docs);
        let out = root.join("priors").join(format!("{name}.bin"));
        std::fs::write(&out, &priors).expect("write priors");
        println!(
            "{name}: {} docs, {total} bytes -> {}",
            docs.len(),
            out.display()
        );
    }
}
