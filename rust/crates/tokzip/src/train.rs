//! Offline trainer (cargo feature `train`): builds each language's dictionary suffix
//! (`dict/<language>.bin`) by COVER-style segment selection, clusters context bytes into
//! literal classes by their next-byte statistics, then counts the bits coded at every model
//! node while compressing the language's training documents with its dictionary and turns the
//! counts into the initial probabilities shipped in `priors/<language>.bin`.

use crate::lz::{
    encode_doc_with_stats, LitClasses, Models, Primed, Segment, LIT_CLASSES, LIT_CLASSES2,
    MODEL_SIZE,
};
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
/// entries marked `"trainable": false`.
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
        .filter(|line| !line.trim().is_empty())
        .filter_map(|line| {
            let entry: serde_json::Value = serde_json::from_str(line).expect("manifest entry");
            let selected = entry["split"].as_str() == Some(split)
                && !(skip_untrainable && entry["trainable"].as_bool() == Some(false));
            selected.then(|| {
                entry["file"]
                    .as_str()
                    .expect("manifest entry without file")
                    .to_string()
            })
        })
        .collect::<Vec<_>>()
        .into_iter()
        .map(move |file| std::fs::read(dir.join(file)).expect("doc"))
}

/// Dictionary suffix budget per language. Ratio keeps improving with budget, but every
/// language ships in the wasm module, so the budget is bounded by the module size target.
pub const DICTIONARY_BUDGET: usize = 128 * 1024;
/// Bound on the dictionary-training input per language.
const MAX_DICT_TRAIN_BYTES: usize = 32 * 1024 * 1024;
/// Every held-out document used to pick the segment size, bounded so the sweep stays fast.
const MAX_VALIDATION_BYTES: usize = 1024 * 1024;
/// Dmer length: the shortest fragment worth a dictionary reference (the coder's minimum match
/// is 2, but references that short rarely beat literals).
const DMER: usize = 8;
const FREQ_BITS: u32 = 20;
/// Candidate segment sizes; the one whose dictionary codes the held-out documents smallest wins.
const SEGMENT_SIZES: [usize; 5] = [64, 128, 256, 512, 1024];

/// Trains a language's dictionary suffix from `docs`: COVER-style greedy selection of the
/// segments whose dmers occur most often across the documents, with the segment size chosen by
/// coding held-out documents with each candidate dictionary (`wrapper` precedes the suffix).
pub fn train_dictionary(docs: &[Vec<u8>], budget: usize, wrapper: &[u8]) -> Vec<u8> {
    let mut bounded: Vec<&[u8]> = Vec::new();
    let mut total = 0usize;
    for doc in docs {
        if total >= MAX_DICT_TRAIN_BYTES {
            break;
        }
        let take = doc.len().min(MAX_DICT_TRAIN_BYTES - total);
        bounded.push(&doc[..take]);
        total += take;
    }
    let mut validation = Vec::new();
    let mut validation_bytes = 0usize;
    let mut fit: Vec<u8> = Vec::new();
    for (i, doc) in bounded.iter().enumerate() {
        if i % 10 == 0 && validation_bytes < MAX_VALIDATION_BYTES {
            let take = doc.len().min(MAX_VALIDATION_BYTES - validation_bytes);
            validation.push(doc[..take].to_vec());
            validation_bytes += take;
        } else {
            fit.extend_from_slice(doc);
        }
    }
    let lit_classes = train_lit_classes(&validation);
    let mut best: Option<(usize, usize)> = None; // (cost, k)
    for k in SEGMENT_SIZES {
        let suffix = cover(&fit, budget, k);
        let mut dict = wrapper.to_vec();
        dict.extend_from_slice(&suffix);
        let cost = coded_size(dict, lit_classes, &validation);
        if best.is_none_or(|(c, _)| cost < c) {
            best = Some((cost, k));
        }
    }
    let k = best.expect("segment sizes").1;
    let all: Vec<u8> = bounded.concat();
    cover(&all, budget, k)
}

/// Greedy COVER selection over `data`: each of the `epochs` slices of `data` contributes its
/// highest-scoring `k`-byte segment (score = total frequency of the dmers it contains; the
/// dmers of a selected segment are then zeroed so later segments do not repeat its content),
/// and the segments are laid out best first, so the most valuable fragments have the lowest
/// dictionary offsets, until `budget` is filled.
fn cover(data: &[u8], budget: usize, k: usize) -> Vec<u8> {
    if data.len() < k.max(DMER) {
        return data.to_vec();
    }
    let dmers = data.len() - DMER + 1;
    let mut freqs = vec![0u32; 1 << FREQ_BITS];
    for pos in 0..dmers {
        freqs[dmer_hash(data, pos)] += 1;
    }
    // Twice the budget's worth of segments are ranked so the best fill the budget.
    let epochs = (2 * budget / k).clamp(1, (dmers / k).max(1));
    let epoch_size = dmers / epochs;
    let per_segment = k - DMER + 1;
    let mut segments: Vec<(u64, &[u8])> = Vec::new();
    for epoch in 0..epochs {
        let start = epoch * epoch_size;
        let end = (start + epoch_size).min(dmers);
        if end - start < per_segment {
            continue;
        }
        let mut score: u64 = (start..start + per_segment)
            .map(|p| u64::from(freqs[dmer_hash(data, p)]))
            .sum();
        let (mut best_score, mut best_start) = (score, start);
        for p in start + per_segment..end {
            score += u64::from(freqs[dmer_hash(data, p)]);
            score -= u64::from(freqs[dmer_hash(data, p - per_segment)]);
            if score > best_score {
                best_score = score;
                best_start = p + 1 - per_segment;
            }
        }
        if best_score == 0 {
            continue;
        }
        // Trim to the dmers that still count, then retire them.
        let live: Vec<usize> = (best_start..best_start + per_segment)
            .filter(|&p| freqs[dmer_hash(data, p)] > 0)
            .collect();
        let (first, last) = (live[0], *live.last().unwrap());
        for p in first..=last {
            freqs[dmer_hash(data, p)] = 0;
        }
        segments.push((best_score, &data[first..last + DMER]));
    }
    segments.sort_by_key(|&(score, _)| std::cmp::Reverse(score));
    let mut out = Vec::with_capacity(budget);
    for (_, segment) in segments {
        let room = budget - out.len();
        out.extend_from_slice(&segment[..segment.len().min(room)]);
        if out.len() == budget {
            break;
        }
    }
    out
}

#[inline]
fn dmer_hash(data: &[u8], pos: usize) -> usize {
    let v = u64::from_le_bytes(data[pos..pos + DMER].try_into().unwrap());
    (v.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> (64 - FREQ_BITS)) as usize
}

/// Total coded body size of `docs`, each as one segment against `dict` with the models the
/// trainer starts from (no priors), so candidate dictionaries compare on the codec's own cost.
fn coded_size(dict: Vec<u8>, lit_classes: LitClasses, docs: &[Vec<u8>]) -> usize {
    let primed = Primed::self_primed(dict, lit_classes);
    let lookup = |_: u8| &primed;
    docs.iter()
        .filter(|doc| !doc.is_empty())
        .map(|doc| {
            let segments = [Segment {
                end: doc.len(),
                lang: 0,
            }];
            encode_doc_with_stats(&lookup, doc, &segments, None).0.len()
        })
        .sum()
}

/// Trains the literal class tables and priors for a language whose full dictionary is `dict`
/// from `docs` (each compressed as one segment). The models start from `init` priors when
/// given — a second round parses the documents the way the shipped models will, so its counts
/// match the tokens the encoder actually picks — and from the dictionary-primed state
/// otherwise. Returns the raw serialized model (`PRIORS_SIZE` bytes).
pub fn train_priors(dict: Vec<u8>, docs: &[Vec<u8>], init: Option<&[u8]>) -> Vec<u8> {
    let primed = match init {
        Some(priors) => Primed::new(dict, Models::from_raw_priors(priors)),
        None => Primed::self_primed(dict, train_lit_classes(docs)),
    };
    let lit_classes = primed.models.lit_classes;
    let lookup = |_: u8| &primed;
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
    let mut out = lit_classes.0.to_vec();
    out.extend_from_slice(&lit_classes.1);
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

/// Clusters the previous byte into `LIT_CLASSES` classes and the byte before it into
/// `LIT_CLASSES2` classes, each by how well the class's pooled next-byte distribution predicts
/// the successors of the byte value.
fn train_lit_classes(docs: &[Vec<u8>]) -> LitClasses {
    let mut counts1 = vec![[0u32; 256]; 256];
    let mut counts2 = vec![[0u32; 256]; 256];
    for doc in docs {
        for pair in doc.windows(2) {
            counts1[pair[0] as usize][pair[1] as usize] += 1;
        }
        for triple in doc.windows(3) {
            counts2[triple[0] as usize][triple[2] as usize] += 1;
        }
    }
    (
        cluster_contexts(&counts1, LIT_CLASSES),
        cluster_contexts(&counts2, LIT_CLASSES2),
    )
}

/// K-means over context byte values: each value joins the class whose pooled next-byte
/// distribution predicts its successors best (minimum cross-entropy), seeded by frequency.
fn cluster_contexts(counts: &[[u32; 256]], classes_count: usize) -> [u8; 256] {
    // Seed: the most frequent previous-byte values each get their own class (they carry the
    // most coding decisions); everything else starts in the last class.
    let mut order: Vec<usize> = (0..256).collect();
    order.sort_by_key(|&prev| {
        std::cmp::Reverse(counts[prev].iter().map(|&c| u64::from(c)).sum::<u64>())
    });
    let mut classes = [(classes_count - 1) as u8; 256];
    for (class, &prev) in order.iter().take(classes_count - 1).enumerate() {
        classes[prev] = class as u8;
    }
    for _ in 0..20 {
        // Pooled, smoothed log-probabilities per class.
        let mut pooled = vec![[0f64; 256]; classes_count];
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
