//! Trains `dict/<language>.bin` and `priors/<language>.bin` for every embedded language from
//! the corpus train split; `dict/wrapper.bin` must already exist (scripts/train/train.ts
//! writes it before running this).
//!
//!   cargo run --release --features train --bin train [-- <corpus dir>]

use std::path::{Path, PathBuf};

/// Per-language priors-training input bound; keeps a full run to tens of seconds per language.
const MAX_PRIORS_TRAIN_BYTES: usize = 12 * 1024 * 1024;

fn main() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..");
    let corpus = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| root.join("../tokzip-corpus/corpus"));
    let wrapper = std::fs::read(root.join("dict/wrapper.bin")).expect("dict/wrapper.bin");
    let mut trainees = Vec::new();
    for &(name, group) in tokzip::train::languages() {
        let docs: Vec<Vec<u8>> = tokzip::train::train_docs(&corpus, name).collect();
        assert!(
            !docs.is_empty(),
            "no train-split corpus for {name} under {}",
            corpus.display()
        );
        let suffix = tokzip::train::train_dictionary(&docs, group.dictionary_budget(), &wrapper);
        std::fs::write(root.join("dict").join(format!("{name}.bin")), &suffix).expect("write dict");
        let mut dict = wrapper.clone();
        dict.extend_from_slice(&suffix);
        let mut total = 0usize;
        let mut priors_docs: Vec<Vec<u8>> = Vec::new();
        for doc in &docs {
            if total >= MAX_PRIORS_TRAIN_BYTES {
                break;
            }
            let take = doc.len().min(MAX_PRIORS_TRAIN_BYTES - total);
            priors_docs.push(doc[..take].to_vec());
            total += take;
        }
        println!(
            "{name}: dictionary {} B from {} docs; priors from {} docs ({total} B)",
            suffix.len(),
            docs.len(),
            priors_docs.len()
        );
        trainees.push(tokzip::train::Trainee {
            name: name.to_string(),
            group,
            dict,
            docs: priors_docs,
        });
    }
    for (trainee, priors) in trainees.iter().zip(tokzip::train::train_priors(&trainees)) {
        std::fs::write(
            root.join("priors").join(format!("{}.bin", trainee.name)),
            priors,
        )
        .expect("write priors");
    }
}
