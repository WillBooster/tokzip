//! Trains, from the corpus train split, `dict/<group>.bin` (the shared dictionary part of every
//! model group that has one), `dict/<language>.bin` (every language's dictionary suffix),
//! `priors/<group>.bin` (every group's literal priors, packed) and `priors/<language>.bin`
//! (every language's own model nodes); `dict/wrapper.bin` must already exist
//! (scripts/train/train.ts writes it before running this).
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
    let languages = tokzip::train::languages();
    let docs_by_language: Vec<Vec<Vec<u8>>> = languages
        .iter()
        .map(|&(name, _)| {
            let docs: Vec<Vec<u8>> = tokzip::train::train_docs(&corpus, name).collect();
            assert!(
                !docs.is_empty(),
                "no train-split corpus for {name} under {}",
                corpus.display()
            );
            docs
        })
        .collect();
    // A group's shared part is trained on its languages' documents interleaved, so every
    // language weighs in before the trainer's input bound cuts the pool.
    let mut shared: Vec<Vec<u8>> = Vec::new();
    for group in tokzip::train::Group::ALL {
        let members: Vec<usize> = (0..languages.len())
            .filter(|&i| languages[i].1 == group)
            .collect();
        let mut part = Vec::new();
        if group.shared_budget() > 0 && !members.is_empty() {
            let longest = members
                .iter()
                .map(|&i| docs_by_language[i].len())
                .max()
                .unwrap();
            let mut pooled: Vec<Vec<u8>> = Vec::new();
            // The member order rotates per round so that the trainer's fixed-stride held-out
            // selection does not land on the same members every round; the pool stops at the
            // trainer's input bound, which is all it reads.
            let mut pooled_bytes = 0usize;
            'pool: for round in 0..longest {
                for m in 0..members.len() {
                    let i = members[(round + m) % members.len()];
                    if let Some(doc) = docs_by_language[i].get(round) {
                        pooled.push(doc.clone());
                        pooled_bytes += doc.len();
                        if pooled_bytes >= tokzip::train::MAX_DICT_TRAIN_BYTES {
                            break 'pool;
                        }
                    }
                }
            }
            part = tokzip::train::train_dictionary(&pooled, group.shared_budget(), &wrapper);
            println!(
                "{}: shared dictionary part {} B from {} docs",
                group.name(),
                part.len(),
                pooled.len()
            );
        }
        let path = root.join("dict").join(format!("{}.bin", group.name()));
        if part.is_empty() {
            let _ = std::fs::remove_file(path);
        } else {
            std::fs::write(path, &part).expect("write shared dictionary part");
        }
        shared.push(part);
    }
    let mut trainees = Vec::new();
    for (i, &(name, group)) in languages.iter().enumerate() {
        let docs = &docs_by_language[i];
        let mut prefix = wrapper.clone();
        prefix.extend_from_slice(&shared[group as usize]);
        let suffix = tokzip::train::train_dictionary(docs, group.dictionary_budget(), &prefix);
        std::fs::write(root.join("dict").join(format!("{name}.bin")), &suffix).expect("write dict");
        let mut dict = prefix;
        dict.extend_from_slice(&suffix);
        let mut total = 0usize;
        let mut priors_docs: Vec<Vec<u8>> = Vec::new();
        for doc in docs {
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
    let priors = tokzip::train::train_priors(&trainees);
    for (trainee, raw) in trainees.iter().zip(&priors) {
        std::fs::write(
            root.join("priors").join(format!("{}.bin", trainee.name)),
            tokzip::train::language_priors(raw),
        )
        .expect("write priors");
    }
    for group in tokzip::train::Group::ALL {
        // Every language of a group gets the same literal part; the first one's is written.
        let raw = trainees
            .iter()
            .zip(&priors)
            .find(|(t, _)| t.group == group)
            .map(|(_, raw)| raw.as_slice());
        let part = raw.map_or_else(Vec::new, tokzip::train::group_priors);
        std::fs::write(
            root.join("priors").join(format!("{}.bin", group.name())),
            part,
        )
        .expect("write group priors");
    }
}
