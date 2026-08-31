//! Priors trainer (cargo feature `train`): clusters previous-byte values into literal context
//! classes by their next-byte statistics, then counts the bits coded at every model node while
//! compressing a language's training documents with its dictionary, and turns the counts into
//! the initial probabilities shipped in `priors/<language>.bin`.

use crate::lz::{encode_doc_with_stats, Primed, Segment, LIT_CLASSES, MODEL_SIZE};
use crate::rc::PROB_BITS;

/// Language names, in id order.
pub fn languages() -> Vec<&'static str> {
    crate::lang::LANGUAGES
        .iter()
        .map(|(name, _, _)| *name)
        .collect()
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
