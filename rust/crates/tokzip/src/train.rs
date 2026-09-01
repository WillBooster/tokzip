//! Priors trainer (cargo feature `train`): clusters previous-byte values into literal context
//! classes by their next-byte statistics, then counts the bits coded at every model node while
//! compressing a language's training documents with its dictionary, and turns the counts into
//! the initial probabilities shipped in `priors/<language>.bin`.

use crate::lz::{encode_doc_with_stats, Primed, Segment, LIT_CLASSES, MODEL_SIZE};
use crate::rc::PROB_BITS;
use std::path::Path;

/// Language names, in id order.
pub fn languages() -> Vec<&'static str> {
    crate::lang::LANGUAGES
        .iter()
        .map(|(name, _, _)| *name)
        .collect()
}

/// Documents of `lang`'s train split from a tokzip-corpus checkout, in manifest order, skipping
/// entries marked `"trainable": false` — the same selection as `loadTrainDocs` in
/// scripts/train/train.ts, which trains the dictionaries.
pub fn train_docs<'a>(corpus: &'a Path, lang: &str) -> impl Iterator<Item = Vec<u8>> + 'a {
    corpus_docs(corpus, lang, "train", true)
}

/// Documents of `lang`'s bench split from a tokzip-corpus checkout, in manifest order.
pub fn bench_docs<'a>(corpus: &'a Path, lang: &str) -> impl Iterator<Item = Vec<u8>> + 'a {
    corpus_docs(corpus, lang, "bench", false)
}

fn corpus_docs<'a>(
    corpus: &'a Path,
    lang: &str,
    split: &'a str,
    skip_untrainable: bool,
) -> impl Iterator<Item = Vec<u8>> + 'a {
    let dir = corpus.join(lang);
    let manifest = std::fs::read_to_string(dir.join("manifest.jsonl")).expect("manifest");
    manifest
        .lines()
        .filter(move |line| {
            manifest_value(line, "split") == Some(split)
                && !(skip_untrainable && manifest_value(line, "trainable") == Some("false"))
        })
        .map(|line| {
            manifest_value(line, "file")
                .expect("manifest entry without file")
                .to_string()
        })
        .collect::<Vec<_>>()
        .into_iter()
        .map(move |file| std::fs::read(dir.join(file)).expect("doc"))
}

/// Value of top-level `key` in one flat manifest.jsonl object, tolerant of spacing and key
/// order: a string's content (manifest strings carry no escapes) or a bare literal's text.
fn manifest_value<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    let quoted = format!("\"{key}\"");
    let rest = line[line.find(&quoted)? + quoted.len()..]
        .trim_start()
        .strip_prefix(':')?
        .trim_start();
    match rest.strip_prefix('"') {
        Some(string) => string.split('"').next(),
        None => rest.split([',', '}']).next().map(str::trim),
    }
}

/// Trains the literal class table and priors for language `lang` from `docs` (each compressed
/// as one segment of that language, starting from the dictionary-primed state). Returns the
/// serialized model (`PRIORS_SIZE` bytes).
pub fn train_priors(lang: u8, docs: &[Vec<u8>]) -> Vec<u8> {
    let lit_class = train_lit_classes(docs);
    let primed: &'static Primed = Box::leak(Box::new(Primed::new(
        crate::lang::dictionary(lang),
        None,
        lit_class,
    )));
    let lookup = |_: u8| primed;
    let mut stats = vec![0u32; 2 * MODEL_SIZE];
    for doc in docs {
        if doc.is_empty() {
            continue;
        }
        let segments = [Segment {
            end: doc.len(),
            lang: 0,
        }];
        let (_, counted) = encode_doc_with_stats(&lookup, doc, &segments, Some(stats));
        stats = counted.expect("stats");
    }
    let scale = f64::from(1u32 << PROB_BITS);
    let mut out = lit_class.to_vec();
    out.extend((0..MODEL_SIZE).map(|node| {
        let (n0, n1) = (f64::from(stats[2 * node]), f64::from(stats[2 * node + 1]));
        let prob = if n0 + n1 == 0.0 {
            f64::from(primed.models.probs[node])
        } else {
            // Laplace-smoothed P(bit = 0), clamped away from the coder's extremes.
            ((n0 + 0.5) / (n0 + n1 + 1.0) * scale).clamp(31.0, scale - 31.0)
        };
        (prob.round() as u32 >> 3) as u8
    }));
    out
}

/// K-means over previous-byte values: each byte value joins the class whose pooled next-byte
/// distribution predicts its successors best (minimum cross-entropy), seeded with the default
/// lexical table.
fn train_lit_classes(docs: &[Vec<u8>]) -> [u8; 256] {
    let mut counts = vec![[0u32; 256]; 256];
    for doc in docs {
        for pair in doc.windows(2) {
            counts[pair[0] as usize][pair[1] as usize] += 1;
        }
    }
    // Seed: the most frequent previous-byte values each get their own class (they carry the
    // most coding decisions); everything else starts in the last class.
    let mut order: Vec<usize> = (0..256).collect();
    order.sort_by_key(|&prev| {
        std::cmp::Reverse(counts[prev].iter().map(|&c| u64::from(c)).sum::<u64>())
    });
    let mut classes = [(LIT_CLASSES - 1) as u8; 256];
    for (class, &prev) in order.iter().take(LIT_CLASSES - 1).enumerate() {
        classes[prev] = class as u8;
    }
    for _ in 0..20 {
        // Pooled, smoothed log-probabilities per class.
        let mut pooled = vec![[0f64; 256]; LIT_CLASSES];
        for prev in 0..256 {
            for next in 0..256 {
                pooled[classes[prev] as usize][next] += f64::from(counts[prev][next]);
            }
        }
        let log_probs: Vec<[f64; 256]> = pooled
            .iter()
            .map(|row| {
                let total: f64 = row.iter().sum::<f64>() + 256.0 * 0.05;
                let mut out = [0f64; 256];
                for next in 0..256 {
                    out[next] = ((row[next] + 0.05) / total).ln();
                }
                out
            })
            .collect();
        let mut changed = false;
        for prev in 0..256 {
            if counts[prev].iter().all(|&c| c == 0) {
                continue;
            }
            let mut best = classes[prev] as usize;
            let mut best_score = f64::NEG_INFINITY;
            for (class, log_prob) in log_probs.iter().enumerate() {
                let score: f64 = (0..256)
                    .map(|next| f64::from(counts[prev][next]) * log_prob[next])
                    .sum();
                if score > best_score {
                    best_score = score;
                    best = class;
                }
            }
            if best != classes[prev] as usize {
                classes[prev] = best as u8;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    classes
}
