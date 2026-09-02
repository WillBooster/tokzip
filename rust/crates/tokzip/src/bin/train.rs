//! Trains `dict/<language>.bin` and `priors/<language>.bin` for every embedded language from
//! the corpus train split; `dict/wrapper.bin` must already exist (scripts/train/train.ts
//! writes it before running this).
//!
//!   cargo run --release --features train --bin train [-- <corpus dir>]

use std::path::{Path, PathBuf};

/// Per-language priors-training input bound; keeps a full run to tens of seconds per language.
const MAX_PRIORS_TRAIN_BYTES: usize = 12 * 1024 * 1024;
/// Priors rounds: the first parses with dictionary-primed models, each later one with the
/// previous round's priors.
const PRIORS_ROUNDS: usize = 3;

fn main() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..");
    let corpus = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| root.join("../tokzip-corpus/corpus"));
    let wrapper = std::fs::read(root.join("dict/wrapper.bin")).expect("dict/wrapper.bin");
    for name in tokzip::train::languages() {
        let docs: Vec<Vec<u8>> = tokzip::train::train_docs(&corpus, name).collect();
        assert!(
            !docs.is_empty(),
            "no train-split corpus for {name} under {}",
            corpus.display()
        );
        let suffix =
            tokzip::train::train_dictionary(&docs, tokzip::train::DICTIONARY_BUDGET, &wrapper);
        std::fs::write(root.join("dict").join(format!("{name}.bin")), &suffix).expect("write dict");
        let mut dict = wrapper.clone();
        dict.extend_from_slice(&suffix);
        let mut total = 0usize;
        let priors_docs: Vec<Vec<u8>> = docs
            .iter()
            .take_while(|doc| {
                let within = total < MAX_PRIORS_TRAIN_BYTES;
                total += doc.len();
                within
            })
            .cloned()
            .collect();
        let mut priors: Option<Vec<u8>> = None;
        for _ in 0..PRIORS_ROUNDS {
            priors = Some(tokzip::train::train_priors(
                dict.clone(),
                &priors_docs,
                priors.as_deref(),
            ));
        }
        std::fs::write(
            root.join("priors").join(format!("{name}.bin")),
            priors.unwrap(),
        )
        .expect("write priors");
        println!(
            "{name}: dictionary {} B from {} docs; priors from {} docs ({total} B)",
            suffix.len(),
            docs.len(),
            priors_docs.len()
        );
    }
}
