//! Embedded language dictionaries and automatic per-segment language detection.
//!
//! Every language's dictionary is the shared wrapper (Markdown/JSON scaffolding, generic
//! prose) followed by its trained suffix. Detection needs no parser: one table maps every
//! 4-gram hash to a bit mask of the languages whose suffix has it (at most 32 languages), the
//! document is scored per 64-byte window by how many positions hit each language (labeled
//! code fences add a hint for their language), and a Viterbi pass with a switch penalty turns
//! the window scores into segments. The decoder never detects anything — it reads the segment
//! table from the frame.

use crate::grams::{gram_hash, GRAM_BITS, GRAM_SET_BYTES};
use crate::languages::LANGUAGES;
pub use crate::languages::LANGUAGE_COUNT;
use crate::lz::{decode_doc, Models, Primed, Segment};
use crate::varint::read_varint;
use std::sync::OnceLock;

const WRAPPER: &[u8] = include_bytes!("../../../../dict/wrapper.bin");

/// A language's embedded assets as packed by the build script (`build.rs`, see `pack.rs`).
struct Assets {
    /// The trained dictionary suffix coded by the codec itself.
    packed_suffix: &'static [u8],
    /// The language's own model nodes; the literal part comes from `GROUP_PRIORS`.
    priors: &'static [u8],
    /// The bitset of the 4-gram hashes of the suffix (`grams.rs`), so detection needs no
    /// decoded dictionary.
    grams: &'static [u8],
}

// `ASSETS` (per language, in id order) and `GROUP_PRIORS` (per `Group`, in `Group::ALL` order).
include!(concat!(env!("OUT_DIR"), "/assets.rs"));

const LANG_HTML: u8 = 3;
const LANG_CSS: u8 = 4;
const LANG_JAVASCRIPT: u8 = 5;
const LANG_TYPESCRIPT: u8 = 6;
const LANG_C: u8 = 7;
const LANG_CPP: u8 = 8;
const LANG_CSHARP: u8 = 9;
const LANG_DART: u8 = 10;
const LANG_HASKELL: u8 = 11;
const LANG_JAVA: u8 = 12;
const LANG_JSP: u8 = 13;
const LANG_PHP: u8 = 14;
const LANG_PYTHON: u8 = 15;
const LANG_RUBY: u8 = 16;
const LANG_RUST: u8 = 17;
const LANG_ZIG: u8 = 18;

static PRIMED: [OnceLock<Primed>; LANGUAGE_COUNT] = [const { OnceLock::new() }; LANGUAGE_COUNT];

/// The language's dictionary (wrapper + suffix) with its trained priors, built on first use and
/// cached for the process; the encoder's match index is built on first encode.
pub fn primed(lang: u8) -> &'static Primed {
    PRIMED[lang as usize].get_or_init(|| {
        let (_, group) = LANGUAGES[lang as usize];
        let assets = &ASSETS[lang as usize];
        let models = Models::from_packed(assets.priors, GROUP_PRIORS[group as usize]);
        let (len, body) = read_varint(assets.packed_suffix).expect("packed dictionary length");
        let mut bytes = Vec::with_capacity(WRAPPER.len() + len as usize);
        bytes.extend_from_slice(WRAPPER);
        // The suffix was coded with the language's models and no dictionary (`pack.rs`).
        let empty = Primed::new(Vec::new(), models.clone());
        let segments = [Segment {
            end: len as usize,
            lang: 0,
        }];
        decode_doc(&|_| &empty, body, len as usize, &segments, &mut bytes)
            .expect("packed dictionary decodes");
        Primed::new(bytes, models)
    })
}

// ---------------------------------------------------------------------------
// Detection
// ---------------------------------------------------------------------------

const WINDOW: usize = 64;
/// Cost of switching languages between windows, in window-score units (a window scores at
/// most `WINDOW` gram hits plus `WINDOW` of fence hint).
const SWITCH_PENALTY: i32 = 48;

/// For every 4-gram hash, the set of languages whose dictionary contains a gram with that
/// hash, as a bit per language id: one lookup scores a position for every language. Built
/// from the precomputed per-language bitsets, so the first `compress` decodes no dictionary
/// for it.
struct GramTable {
    masks: Vec<u32>,
}

const _: () = assert!(LANGUAGE_COUNT <= 32);

impl GramTable {
    fn new() -> Self {
        let mut masks = vec![0u32; 1 << GRAM_BITS];
        for (lang, assets) in ASSETS.iter().enumerate() {
            assert_eq!(assets.grams.len(), GRAM_SET_BYTES, "gram set size mismatch");
            for (byte, &bits) in assets.grams.iter().enumerate() {
                let mut bits = bits;
                while bits != 0 {
                    masks[byte << 3 | bits.trailing_zeros() as usize] |= 1 << lang;
                    bits &= bits - 1;
                }
            }
        }
        Self { masks }
    }
}

fn gram_table() -> &'static GramTable {
    static TABLE: OnceLock<GramTable> = OnceLock::new();
    TABLE.get_or_init(GramTable::new)
}

/// Splits `doc` into language segments (contiguous, covering the whole document).
pub fn segment(doc: &[u8]) -> Vec<Segment> {
    analyze(doc).0
}

/// The language with the most dictionary 4-gram overlap (no fence hint), from pre-computed
/// whole-document `scores`.
pub fn top_language(scores: &[i32; LANGUAGE_COUNT]) -> u8 {
    (0..LANGUAGE_COUNT as u8)
        .max_by_key(|&lang| (scores[lang as usize], std::cmp::Reverse(lang)))
        .unwrap()
}

/// Splits `doc` into language segments and returns the whole-document gram totals (pre-fence)
/// the split was derived from, so the caller can rank candidate languages without re-scanning.
pub fn analyze(doc: &[u8]) -> (Vec<Segment>, [i32; LANGUAGE_COUNT]) {
    // Short documents have no windows, so they score in one pass; longer ones score per window
    // below and the whole-document totals are the column sums of those rows (no second scan).
    if doc.len() < WINDOW * 2 {
        let totals = gram_scores(doc);
        return (
            vec![Segment {
                end: doc.len(),
                lang: best_single_from(doc, totals),
            }],
            totals,
        );
    }
    let windows = doc.len().div_ceil(WINDOW);
    let mut scores = vec![[0i32; LANGUAGE_COUNT]; windows];
    add_gram_hits(doc, &mut scores, |pos| pos / WINDOW);
    let mut totals = [0i32; LANGUAGE_COUNT];
    for row in &scores {
        for (lang, &score) in row.iter().enumerate() {
            totals[lang] += score;
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
    let mut segments: Vec<Segment> = Vec::new();
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
    // Windows are language evidence, not boundaries: a switch lands on the nearest line start,
    // where the language actually changes, instead of in the middle of a line.
    for i in 0..segments.len().saturating_sub(1) {
        let low = segments[i]
            .end
            .saturating_sub(WINDOW / 2)
            .max(i.checked_sub(1).map_or(1, |p| segments[p].end + 1));
        let high = (segments[i].end + WINDOW / 2).min(segments[i + 1].end - 1);
        segments[i].end = nearest_line_start(doc, segments[i].end, low, high);
    }
    (segments, totals)
}

/// The line start within `low..=high` closest to `at` (`at` itself when none).
fn nearest_line_start(doc: &[u8], at: usize, low: usize, high: usize) -> usize {
    let (mut best, mut best_dist) = (at, usize::MAX);
    for pos in low..=high.min(doc.len()) {
        let dist = pos.abs_diff(at);
        if pos > 0 && doc[pos - 1] == b'\n' && dist < best_dist {
            best = pos;
            best_dist = dist;
        }
    }
    best
}

/// Whole-document 4-gram overlap of `doc` against each language's dictionary (no fence hint),
/// for documents too short to have windows.
fn gram_scores(doc: &[u8]) -> [i32; LANGUAGE_COUNT] {
    let mut scores = [[0i32; LANGUAGE_COUNT]];
    add_gram_hits(doc, &mut scores, |_| 0);
    scores[0]
}

/// The one gram-scoring pass: for every 4-gram position of `doc`, counts a hit for each
/// language whose dictionary contains that gram, in the row `row_of(position)` selects.
fn add_gram_hits(doc: &[u8], rows: &mut [[i32; LANGUAGE_COUNT]], row_of: impl Fn(usize) -> usize) {
    if doc.len() < 4 {
        return;
    }
    let table = gram_table();
    for pos in 0..doc.len() - 3 {
        let mut mask = table.masks[gram_hash(doc, pos)];
        let row = &mut rows[row_of(pos)];
        while mask != 0 {
            row[mask.trailing_zeros() as usize] += 1;
            mask &= mask - 1;
        }
    }
}

/// Argmax language of `doc` from pre-computed gram `scores` plus fence hints (short-doc path).
fn best_single_from(doc: &[u8], mut scores: [i32; LANGUAGE_COUNT]) -> u8 {
    let mut hinted = [[0i32; LANGUAGE_COUNT]; 1];
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
            // Close the open fence, if any.
            let closed = open.take();
            if let Some((lang, from)) = closed {
                for pos in from..line_start {
                    scores[pos / window][lang as usize] += 1;
                }
            }
            // A labeled line re-opens only when it hands off to a *different* language (so
            // back-to-back `js`/`ts` blocks each keep their hint); a same-label labeled line
            // (```js … ```js) is just a closing fence and must not re-open the hint.
            if let Some(lang) = label_lang {
                if closed.map(|(closed_lang, _)| closed_lang) != Some(lang) {
                    open = Some((lang, line_end + 1));
                }
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
        b"c" | b"h" => Some(LANG_C),
        b"cpp" | b"c++" | b"cc" | b"cxx" | b"hpp" => Some(LANG_CPP),
        b"cs" | b"csharp" | b"c#" => Some(LANG_CSHARP),
        b"dart" => Some(LANG_DART),
        b"hs" | b"haskell" => Some(LANG_HASKELL),
        b"java" => Some(LANG_JAVA),
        b"jsp" => Some(LANG_JSP),
        b"php" => Some(LANG_PHP),
        b"py" | b"python" | b"python3" => Some(LANG_PYTHON),
        b"rb" | b"ruby" => Some(LANG_RUBY),
        b"rs" | b"rust" => Some(LANG_RUST),
        b"zig" => Some(LANG_ZIG),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The packed assets the build embeds must restore exactly the trained values: the packer
    /// and the unpacker are symmetric, so a lossy pack would still round-trip frames and show
    /// up only as a worse ratio.
    #[test]
    fn packed_assets_restore_the_trained_assets() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..");
        for (lang, (name, _)) in LANGUAGES.iter().enumerate() {
            let raw = std::fs::read(root.join(format!("priors/{name}.bin"))).expect("raw priors");
            let expected = Models::from_raw_priors(&raw);
            let primed = primed(lang as u8);
            assert_eq!(primed.models.lit_classes, expected.lit_classes, "{name}");
            assert_eq!(primed.models.probs, expected.probs, "{name}");
            let dict = std::fs::read(root.join(format!("dict/{name}.bin"))).expect("dictionary");
            assert_eq!(&primed.bytes[WRAPPER.len()..], dict, "{name}");
            assert_eq!(ASSETS[lang].grams, crate::grams::gram_set(&dict), "{name}");
        }
    }

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
    fn segment_boundaries_land_on_line_starts() {
        let ja = "ゲームの仕様を以下に示します。\n".repeat(20);
        let js = "const ctx = canvas.getContext('2d');\n".repeat(20);
        let doc = format!("{ja}{js}");
        let segments = segment(doc.as_bytes());
        assert!(segments.len() >= 2, "{segments:?}");
        for boundary in &segments[..segments.len() - 1] {
            assert_eq!(doc.as_bytes()[boundary.end - 1], b'\n', "{segments:?}");
        }
        assert_eq!(nearest_line_start(b"abc\ndefghij\nklm", 6, 1, 14), 4);
        assert_eq!(nearest_line_start(b"abc\ndefghij\nklm", 4, 1, 14), 4);
        assert_eq!(nearest_line_start(b"abcdefghijklm", 6, 1, 12), 6);
    }

    #[test]
    fn short_documents_get_one_segment() {
        assert_eq!(segment(b"hi").len(), 1);
        assert_eq!(segment(b"").len(), 1);
    }
}
