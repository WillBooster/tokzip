//! Offline trainer (cargo feature `train`): builds the dictionary parts (a group's shared part,
//! `dict/<group>.bin`, and each language's suffix, `dict/<language>.bin`) by COVER-style
//! segment selection, clusters context bytes into literal classes by their next-byte
//! statistics, then counts the bits coded at every model node while compressing each
//! language's training documents with its dictionary and turns the counts into the initial
//! probabilities shipped as each group's packed literal priors (`priors/<group>.bin`) and each
//! language's own nodes (`priors/<language>.bin`).

pub use crate::languages::Group;
use crate::lz::{
    encode_doc_with_stats, LitClasses, Models, Primed, Segment, LIT, LIT_CLASSES, LIT_CLASSES2,
    MODEL_SIZE,
};
use crate::rc::{PROB_BITS, PROB_INIT};
use std::path::Path;

/// The embedded languages with their groups, in id order.
pub fn languages() -> &'static [(&'static str, Group)] {
    &crate::languages::LANGUAGES
}

/// The language's own part of a raw trained model, as `priors/<language>.bin` holds it.
pub fn language_priors(raw: &[u8]) -> Vec<u8> {
    crate::pack::language_part(raw)
}

/// The group's literal part of a raw trained model, packed as `priors/<group>.bin` holds it.
pub fn group_priors(raw: &[u8]) -> Vec<u8> {
    crate::pack::literal_part(raw)
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

/// Bound on the dictionary-training input per language (`train_dictionary` reads no more).
pub const MAX_DICT_TRAIN_BYTES: usize = 32 * 1024 * 1024;
/// Every held-out document used to pick the segment size, bounded so the sweep stays fast.
const MAX_VALIDATION_BYTES: usize = 1024 * 1024;
/// Dmer length: the shortest fragment worth a dictionary reference (the coder's minimum match
/// is 2, but references that short rarely beat literals).
const DMER: usize = 8;
const FREQ_BITS: u32 = 20;
/// Candidate segment sizes; the one whose dictionary codes the held-out documents smallest wins.
const SEGMENT_SIZES: [usize; 5] = [64, 128, 256, 512, 1024];

/// Trains a dictionary part from `docs`: COVER-style greedy selection of the segments whose
/// dmers occur in the most documents, with the segment size chosen by coding held-out
/// documents with each candidate dictionary. `prefix` is the dictionary content that precedes
/// the part (the wrapper, plus the group's shared part for a language suffix): its fragments
/// are not selected again.
pub fn train_dictionary(docs: &[Vec<u8>], budget: usize, prefix: &[u8]) -> Vec<u8> {
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
    let mut fit_lens: Vec<usize> = Vec::new();
    for (i, doc) in bounded.iter().enumerate() {
        if i % 10 == 0 && validation_bytes < MAX_VALIDATION_BYTES {
            let take = doc.len().min(MAX_VALIDATION_BYTES - validation_bytes);
            validation.push(doc[..take].to_vec());
            validation_bytes += take;
        } else {
            fit.extend_from_slice(doc);
            fit_lens.push(doc.len());
        }
    }
    let lit_classes = train_lit_classes(validation.iter().map(Vec::as_slice));
    let mut best: Option<(usize, usize)> = None; // (cost, k)
    for k in SEGMENT_SIZES {
        let suffix = cover(&fit, &fit_lens, prefix, budget, k);
        let mut dict = prefix.to_vec();
        dict.extend_from_slice(&suffix);
        let cost = coded_size(dict, lit_classes, &validation);
        if best.is_none_or(|(c, _)| cost < c) {
            best = Some((cost, k));
        }
    }
    let k = best.expect("segment sizes").1;
    let all: Vec<u8> = bounded.concat();
    let lens: Vec<usize> = bounded.iter().map(|d| d.len()).collect();
    cover(&all, &lens, prefix, budget, k)
}

/// Greedy COVER selection over `data` (the concatenation of documents of `doc_lens`): each of
/// the `epochs` slices of `data` contributes its highest-scoring `k`-byte segment (score =
/// total document frequency of the dmers it contains; the
/// dmers of a selected segment are then zeroed so later segments do not repeat its content),
/// and the segments are laid out best first, so the most valuable fragments have the lowest
/// dictionary offsets, until `budget` is filled.
fn cover(data: &[u8], doc_lens: &[usize], prefix: &[u8], budget: usize, k: usize) -> Vec<u8> {
    if data.len() < k.max(DMER) {
        return data.to_vec();
    }
    let dmers = data.len() - DMER + 1;
    // Document frequency: a dmer counts once per document, so a document repeating a fragment
    // many times does not outvote the fragments many documents share.
    let mut freqs = vec![0u32; 1 << FREQ_BITS];
    let mut seen = vec![u32::MAX; 1 << FREQ_BITS];
    let mut doc_start = 0usize;
    for (id, &len) in doc_lens.iter().enumerate() {
        let end = (doc_start + len).min(dmers + DMER - 1);
        for pos in doc_start..end.saturating_sub(DMER - 1) {
            let h = dmer_hash(data, pos);
            if seen[h] != id as u32 {
                seen[h] = id as u32;
                freqs[h] += 1;
            }
        }
        doc_start += len;
    }
    // Fragments the preceding dictionary content already holds are not worth selecting again.
    for pos in 0..prefix.len().saturating_sub(DMER - 1) {
        freqs[dmer_hash(prefix, pos)] = 0;
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

/// A language being trained: its full dictionary (wrapper + group part + suffix) and its
/// priors documents.
pub struct Trainee {
    pub name: String,
    pub group: Group,
    pub dict: Vec<u8>,
    pub docs: Vec<Vec<u8>>,
}

/// Priors rounds: the first parses with dictionary-primed models, each later one with the
/// previous round's priors, so the counts match the tokens the shipped encoder picks.
const PRIORS_ROUNDS: usize = 3;
/// A literal node whose trained value would have saved fewer bits than this over a flat node on
/// the training bits stays flat: such nodes are many (deep tree nodes of rare contexts) and
/// worth little, and every flat subtree is skipped by the packed priors (`pack.rs`).
const PRIORS_MIN_GAIN: f64 = 6.0;

/// Trains the priors of every trainee (each compressed as one segment against its own
/// dictionary): the literal class tables and literal-tree priors are shared by every language
/// of a group and trained on the group's pooled literal statistics; the other nodes are per
/// language. Returns each trainee's raw serialized model (`PRIORS_SIZE` bytes), in order
/// (`language_priors` / `group_priors` split it into the parts the repository holds).
pub fn train_priors(trainees: &[Trainee]) -> Vec<Vec<u8>> {
    let groups: Vec<Group> = Group::ALL
        .into_iter()
        .filter(|g| trainees.iter().any(|t| t.group == *g))
        .collect();
    let lit_classes: Vec<LitClasses> = groups
        .iter()
        .map(|g| {
            train_lit_classes(
                trainees
                    .iter()
                    .filter(|t| t.group == *g)
                    .flat_map(|t| t.docs.iter().map(Vec::as_slice)),
            )
        })
        .collect();
    let group_index = |t: &Trainee| groups.iter().position(|g| *g == t.group).unwrap();
    let scale = f64::from(1u32 << PROB_BITS);
    let mut priors: Vec<Vec<u8>> = Vec::new();
    for round in 0..PRIORS_ROUNDS {
        let mut stats: Vec<Vec<u32>> = Vec::new();
        let mut fallback: Vec<Vec<u16>> = Vec::new();
        for (i, t) in trainees.iter().enumerate() {
            let primed = if round == 0 {
                Primed::self_primed(t.dict.clone(), lit_classes[group_index(t)])
            } else {
                Primed::new(t.dict.clone(), Models::from_raw_priors(&priors[i]))
            };
            let lookup = |_: u8| &primed;
            let mut counts = vec![0u32; 2 * MODEL_SIZE];
            for doc in t.docs.iter().filter(|doc| !doc.is_empty()) {
                let segments = [Segment {
                    end: doc.len(),
                    lang: 0,
                }];
                counts = encode_doc_with_stats(&lookup, doc, &segments, Some(counts))
                    .1
                    .expect("stats");
            }
            stats.push(counts);
            fallback.push(primed.models.probs.clone());
        }
        // Pooled literal statistics per group.
        let mut pooled: Vec<Vec<u64>> = vec![vec![0u64; 2 * (MODEL_SIZE - LIT)]; groups.len()];
        for (t, counts) in trainees.iter().zip(&stats) {
            for (p, &c) in pooled[group_index(t)].iter_mut().zip(&counts[2 * LIT..]) {
                *p += u64::from(c);
            }
        }
        let quantize = |prob: f64| (prob.round() as u32 >> 3) as u8;
        // Laplace-smoothed P(bit = 0), clamped away from the coder's extremes.
        let trained =
            |n0: f64, n1: f64| ((n0 + 0.5) / (n0 + n1 + 1.0) * scale).clamp(31.0, scale - 31.0);
        priors = trainees
            .iter()
            .enumerate()
            .map(|(i, t)| {
                let g = group_index(t);
                let mut out = lit_classes[g].0.to_vec();
                out.extend_from_slice(&lit_classes[g].1);
                out.extend((0..LIT).map(|node| {
                    let (n0, n1) = (
                        f64::from(stats[i][2 * node]),
                        f64::from(stats[i][2 * node + 1]),
                    );
                    quantize(if n0 + n1 == 0.0 {
                        f64::from(fallback[i][node])
                    } else {
                        trained(n0, n1)
                    })
                }));
                out.extend((0..MODEL_SIZE - LIT).map(|node| {
                    let (n0, n1) = (pooled[g][2 * node] as f64, pooled[g][2 * node + 1] as f64);
                    if n0 + n1 == 0.0 {
                        return quantize(f64::from(PROB_INIT));
                    }
                    let p0 = trained(n0, n1) / scale;
                    // Bits the trained value saves over a flat node on the training bits.
                    let gain = n0 * (p0 / 0.5).log2() + n1 * ((1.0 - p0) / 0.5).log2();
                    quantize(if gain < PRIORS_MIN_GAIN {
                        f64::from(PROB_INIT)
                    } else {
                        p0 * scale
                    })
                }));
                out
            })
            .collect();
    }
    priors
}

/// Clusters the previous byte into `LIT_CLASSES` classes and the byte before it into
/// `LIT_CLASSES2` classes, each by how well the class's pooled next-byte distribution predicts
/// the successors of the byte value.
fn train_lit_classes<'a>(docs: impl Iterator<Item = &'a [u8]>) -> LitClasses {
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

/// Turns on the per-node cost accounting behind `cost_report`.
pub fn enable_cost_report() {
    crate::lz::COST_ENABLED.store(true, std::sync::atomic::Ordering::Relaxed);
}

/// Bits coded per model group so far, over every parse `encode_doc` ran — including the
/// alternative segmentations `compress` tries and discards (see `lz::COST_REPORT`).
pub fn cost_report() -> [f64; 10] {
    *crate::lz::COST_REPORT.lock().unwrap()
}
