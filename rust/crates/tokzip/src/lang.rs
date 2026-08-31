//! Embedded language dictionaries and automatic per-segment language detection.
//!
//! Every language's dictionary is the shared wrapper (Markdown/JSON scaffolding, generic
//! prose) followed by its trained suffix. Detection needs no parser: each language keeps a
//! bitset of the 4-grams in its suffix, the document is scored per 64-byte window by how many
//! positions hit each language's set (labeled code fences add a hint for their language), and
//! a Viterbi pass with a switch penalty turns the window scores into segments. The decoder
//! never detects anything — it reads the segment table from the frame.

use crate::lz::{Primed, Segment};
use std::sync::OnceLock;

const WRAPPER: &[u8] = include_bytes!("../../../../dict/wrapper.bin");

/// Language ids are frame-format identity (v0: any change is a format change). Each entry:
/// name, trained dictionary suffix, trained model priors.
pub const LANGUAGES: [(&str, &[u8], &[u8]); 7] = [
    (
        "text",
        include_bytes!("../../../../dict/text.bin"),
        include_bytes!("../../../../priors/text.bin"),
    ),
    (
        "en-US",
        include_bytes!("../../../../dict/en-US.bin"),
        include_bytes!("../../../../priors/en-US.bin"),
    ),
    (
        "ja-JP",
        include_bytes!("../../../../dict/ja-JP.bin"),
        include_bytes!("../../../../priors/ja-JP.bin"),
    ),
    (
        "html",
        include_bytes!("../../../../dict/html.bin"),
        include_bytes!("../../../../priors/html.bin"),
    ),
    (
        "css",
        include_bytes!("../../../../dict/css.bin"),
        include_bytes!("../../../../priors/css.bin"),
    ),
    (
        "javascript",
        include_bytes!("../../../../dict/javascript.bin"),
        include_bytes!("../../../../priors/javascript.bin"),
    ),
    (
        "typescript",
        include_bytes!("../../../../dict/typescript.bin"),
        include_bytes!("../../../../priors/typescript.bin"),
    ),
];
pub const LANGUAGE_COUNT: usize = LANGUAGES.len();
const LANG_TEXT: u8 = 0;
const LANG_HTML: u8 = 3;
const LANG_CSS: u8 = 4;
const LANG_JAVASCRIPT: u8 = 5;
const LANG_TYPESCRIPT: u8 = 6;

static PRIMED: [OnceLock<Primed>; LANGUAGE_COUNT] = [const { OnceLock::new() }; LANGUAGE_COUNT];

/// The language's dictionary (wrapper + suffix) with chains and trained priors, built on first
/// use and cached for the process.
pub fn primed(lang: u8) -> &'static Primed {
    PRIMED[lang as usize]
        .get_or_init(|| Primed::new(dictionary(lang), Some(LANGUAGES[lang as usize].2), [0; 256]))
}

/// The full dictionary bytes of a language: the shared wrapper followed by its suffix.
pub fn dictionary(lang: u8) -> Vec<u8> {
    let suffix = LANGUAGES[lang as usize].1;
    let mut bytes = Vec::with_capacity(WRAPPER.len() + suffix.len());
    bytes.extend_from_slice(WRAPPER);
    bytes.extend_from_slice(suffix);
    bytes
}

// ---------------------------------------------------------------------------
// Detection
// ---------------------------------------------------------------------------

const GRAM_BITS: u32 = 17;
const WINDOW: usize = 64;
/// Cost of switching languages between windows, in window-score units (a window scores at
/// most `WINDOW` gram hits plus `WINDOW` of fence hint).
const SWITCH_PENALTY: i32 = 48;

struct GramSet {
    bits: Vec<u64>,
}

impl GramSet {
    fn new(bytes: &[u8]) -> Self {
        let mut bits = vec![0u64; (1usize << GRAM_BITS) / 64];
        for pos in 0..bytes.len().saturating_sub(3) {
            let h = gram_hash(bytes, pos);
            bits[h >> 6] |= 1u64 << (h & 63);
        }
        Self { bits }
    }

    #[inline]
    fn contains(&self, h: usize) -> bool {
        (self.bits[h >> 6] >> (h & 63)) & 1 != 0
    }
}

#[inline]
fn gram_hash(bytes: &[u8], pos: usize) -> usize {
    let v = u32::from_le_bytes([bytes[pos], bytes[pos + 1], bytes[pos + 2], bytes[pos + 3]]);
    (v.wrapping_mul(0x9E37_79B1) >> (32 - GRAM_BITS)) as usize
}

fn gram_sets() -> &'static [GramSet] {
    static SETS: OnceLock<Vec<GramSet>> = OnceLock::new();
    SETS.get_or_init(|| {
        LANGUAGES
            .iter()
            .map(|(_, suffix, _)| GramSet::new(suffix))
            .collect()
    })
}

/// Splits `doc` into language segments (contiguous, covering the whole document).
pub fn segment(doc: &[u8]) -> Vec<Segment> {
    if doc.len() < WINDOW * 2 {
        return vec![Segment {
            end: doc.len(),
            lang: best_single(doc),
        }];
    }
    let sets = gram_sets();
    let windows = doc.len().div_ceil(WINDOW);
    let mut scores = vec![[0i32; LANGUAGE_COUNT]; windows];
    for pos in 0..doc.len() - 3 {
        let h = gram_hash(doc, pos);
        let row = &mut scores[pos / WINDOW];
        for (lang, set) in sets.iter().enumerate() {
            row[lang] += set.contains(h) as i32;
        }
    }
    apply_fence_hints(doc, &mut scores);

    // Viterbi: best cumulative score ending in each language per window.
    let mut best = [0i32; LANGUAGE_COUNT];
    let mut back = vec![[0u8; LANGUAGE_COUNT]; windows];
    for (w, row) in scores.iter().enumerate() {
        let mut next = [0i32; LANGUAGE_COUNT];
        let (top_lang, top) =
            best.iter()
                .copied()
                .enumerate()
                .fold(
                    (0, i32::MIN),
                    |acc, (l, s)| if s > acc.1 { (l, s) } else { acc },
                );
        for lang in 0..LANGUAGE_COUNT {
            let stay = best[lang];
            let switch = top - SWITCH_PENALTY;
            if w == 0 || stay >= switch {
                next[lang] = stay + row[lang];
                back[w][lang] = lang as u8;
            } else {
                next[lang] = switch + row[lang];
                back[w][lang] = top_lang as u8;
            }
        }
        best = next;
    }
    let mut lang = 0usize;
    for l in 1..LANGUAGE_COUNT {
        if best[l] > best[lang] {
            lang = l;
        }
    }
    let mut labels = vec![0u8; windows];
    for w in (0..windows).rev() {
        labels[w] = lang as u8;
        lang = back[w][lang] as usize;
    }
    let mut segments = Vec::new();
    for (w, &label) in labels.iter().enumerate() {
        let end = ((w + 1) * WINDOW).min(doc.len());
        match segments.last_mut() {
            Some(Segment {
                end: last_end,
                lang,
            }) if *lang == label => *last_end = end,
            _ => segments.push(Segment { end, lang: label }),
        }
    }
    segments
}

fn best_single(doc: &[u8]) -> u8 {
    if doc.len() < 4 {
        return LANG_TEXT;
    }
    let sets = gram_sets();
    let mut scores = [0i32; LANGUAGE_COUNT];
    for pos in 0..doc.len() - 3 {
        let h = gram_hash(doc, pos);
        for (lang, set) in sets.iter().enumerate() {
            scores[lang] += set.contains(h) as i32;
        }
    }
    let mut hinted = vec![[0i32; LANGUAGE_COUNT]; 1];
    apply_fence_hints(doc, &mut hinted);
    for lang in 0..LANGUAGE_COUNT {
        scores[lang] += hinted[0][lang];
    }
    let mut best = 0usize;
    for lang in 1..LANGUAGE_COUNT {
        if scores[lang] > scores[best] {
            best = lang;
        }
    }
    best as u8
}

/// Adds one point per byte inside a labeled ```` ``` ```` fence to the label's language.
fn apply_fence_hints(doc: &[u8], scores: &mut [[i32; LANGUAGE_COUNT]]) {
    let window = if scores.len() == 1 {
        doc.len().max(1)
    } else {
        WINDOW
    };
    let mut line_start = 0usize;
    let mut open: Option<(u8, usize)> = None;
    while line_start < doc.len() {
        let line_end = doc[line_start..]
            .iter()
            .position(|&b| b == b'\n')
            .map_or(doc.len(), |i| line_start + i);
        let line = &doc[line_start..line_end];
        let trimmed = line.strip_suffix(b"\r").unwrap_or(line);
        if let Some(rest) = trimmed.strip_prefix(b"```") {
            let label_end = rest
                .iter()
                .position(|&b| b == b' ' || b == b'\t')
                .unwrap_or(rest.len());
            let label_lang = fence_language(&rest[..label_end]);
            // Close the open fence, if any. A line that also carries a language label opens a
            // new fence in the same step, so back-to-back labeled blocks each keep their hint.
            if let Some((lang, from)) = open.take() {
                for pos in from..line_start {
                    scores[pos / window][lang as usize] += 1;
                }
            }
            if let Some(lang) = label_lang {
                open = Some((lang, line_end + 1));
            }
        }
        line_start = line_end + 1;
    }
    if let Some((lang, from)) = open {
        for pos in from..doc.len() {
            scores[pos / window][lang as usize] += 1;
        }
    }
}

fn fence_language(label: &[u8]) -> Option<u8> {
    let mut lower = [0u8; 12];
    if label.is_empty() || label.len() > lower.len() {
        return None;
    }
    for (i, &b) in label.iter().enumerate() {
        lower[i] = b.to_ascii_lowercase();
    }
    match &lower[..label.len()] {
        b"js" | b"javascript" | b"jsx" | b"mjs" | b"cjs" => Some(LANG_JAVASCRIPT),
        b"ts" | b"typescript" | b"tsx" | b"mts" => Some(LANG_TYPESCRIPT),
        b"html" | b"htm" | b"xml" | b"svg" | b"vue" | b"svelte" => Some(LANG_HTML),
        b"css" | b"scss" | b"less" => Some(LANG_CSS),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_fenced_code_inside_prose() {
        let mut doc = "以下の要件を満たすブロック崩しゲームを作成してください。パドルは左右矢印キーで移動します。\n".repeat(3);
        doc.push_str("```js\n");
        doc.push_str(&"const canvas = document.getElementById('game');\nconst ctx = canvas.getContext('2d');\nfunction draw() { ctx.clearRect(0, 0, canvas.width, canvas.height); }\n".repeat(3));
        doc.push_str("```\n");
        doc.push_str(&"全てのブロックを消すとクリア画面を表示してください。\n".repeat(3));
        let segments = segment(doc.as_bytes());
        let langs: Vec<u8> = segments.iter().map(|s| s.lang).collect();
        assert!(langs.contains(&LANG_JAVASCRIPT), "{segments:?}");
        assert_eq!(segments.last().unwrap().end, doc.len());
        assert!(segments.windows(2).all(|w| w[0].end < w[1].end));
    }

    #[test]
    fn short_documents_get_one_segment() {
        assert_eq!(segment(b"hi").len(), 1);
        assert_eq!(segment(b"").len(), 1);
    }
}
