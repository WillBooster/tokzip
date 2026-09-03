//! Embedded language dictionaries and automatic per-segment language detection.
//!
//! Every language's dictionary is the shared wrapper (Markdown/JSON scaffolding, generic
//! prose), then its model group's shared part (programming languages share one), then its
//! trained suffix. Detection needs no parser: one table, computed by the build script, maps
//! every 4-gram hash to a bit mask of the languages whose trained dictionary (shared part and
//! suffix) has it (at most 24 languages), the
//! document is scored per 64-byte window by how many positions hit each language (labeled
//! code fences add a hint for their language), and a Viterbi pass with a switch penalty turns
//! the window scores into segments. The decoder never detects anything — it reads the segment
//! table from the frame.

use crate::grams::{gram_hash, gram_mask, GRAM_TABLE_BYTES};
pub use crate::languages::LANGUAGE_COUNT;
use crate::languages::{Group, LANGUAGES};
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
}

// `ASSETS` (per language, in id order), `GROUP_PRIORS` and `GROUP_DICTS` (per `Group`, in
// `Group::ALL` order; a group's shared dictionary part is empty for groups without one), and
// `GRAM_TABLE` (`grams.rs`).
include!(concat!(env!("OUT_DIR"), "/assets.rs"));
const _: () = assert!(GRAM_TABLE.len() == GRAM_TABLE_BYTES);

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
static SHARED: [OnceLock<Vec<u8>>; Group::ALL.len()] =
    [const { OnceLock::new() }; Group::ALL.len()];

/// The language's dictionary (wrapper + group part + suffix) with its trained priors, built on
/// first use and cached for the process; the encoder's match index is built on first encode.
pub fn primed(lang: u8) -> &'static Primed {
    PRIMED[lang as usize].get_or_init(|| {
        let (_, group) = LANGUAGES[lang as usize];
        let models = language_models(lang);
        let shared = SHARED[group as usize].get_or_init(|| {
            // Packed with the models of the group's first language (`build.rs`).
            let first = LANGUAGES.iter().position(|(_, g)| *g == group).unwrap();
            let models = if first == lang as usize {
                models.clone()
            } else {
                language_models(first as u8)
            };
            unpack_dictionary(GROUP_DICTS[group as usize], &models, Vec::new())
        });
        let mut bytes = Vec::with_capacity(
            WRAPPER.len() + shared.len() + ASSETS[lang as usize].packed_suffix.len() * 4,
        );
        bytes.extend_from_slice(WRAPPER);
        bytes.extend_from_slice(shared);
        let bytes = unpack_dictionary(ASSETS[lang as usize].packed_suffix, &models, bytes);
        Primed::new(bytes, models)
    })
}

fn language_models(lang: u8) -> Models {
    let (_, group) = LANGUAGES[lang as usize];
    Models::from_packed(ASSETS[lang as usize].priors, GROUP_PRIORS[group as usize])
}

/// Appends a dictionary part coded by the codec with `models` and no dictionary (`pack.rs`) to
/// `bytes`; an empty `packed` is an empty part.
fn unpack_dictionary(packed: &[u8], models: &Models, mut bytes: Vec<u8>) -> Vec<u8> {
    if packed.is_empty() {
        return bytes;
    }
    let (len, body) = read_varint(packed).expect("packed dictionary length");
    let empty = Primed::new(Vec::new(), models.clone());
    let segments = [Segment {
        end: len as usize,
        lang: 0,
    }];
    decode_doc(&|_| &empty, body, len as usize, &segments, &mut bytes)
        .expect("packed dictionary decodes");
    bytes
}

// ---------------------------------------------------------------------------
// Detection
// ---------------------------------------------------------------------------

const WINDOW: usize = 64;
/// Cost of switching languages between windows, in window-score units (a window scores at
/// most `WINDOW` gram hits plus `WINDOW` of fence hint).
const SWITCH_PENALTY: i32 = 48;

const _: () = assert!(LANGUAGE_COUNT <= 24);

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
    for pos in 0..doc.len() - 3 {
        let mut mask = gram_mask(GRAM_TABLE, gram_hash(doc, pos));
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

    /// The module must embed exactly the committed assets — the build script substitutes flat
    /// priors for stale ones, which would only show as a worse ratio — restore the
    /// dictionaries it codes byte for byte, and detect with the table of those dictionaries.
    #[test]
    fn embedded_assets_match_the_committed_assets() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..");
        let read = |path: String| std::fs::read(root.join(path)).unwrap_or_default();
        for group in Group::ALL {
            assert_eq!(
                GROUP_PRIORS[group as usize],
                read(format!("priors/{}.bin", group.name())),
                "{group:?}"
            );
        }
        let mut dictionaries = Vec::new();
        for (lang, (name, group)) in LANGUAGES.iter().enumerate() {
            assert_eq!(
                ASSETS[lang].priors,
                read(format!("priors/{name}.bin")),
                "{name}"
            );
            let mut trained = read(format!("dict/{}.bin", group.name()));
            trained.extend(read(format!("dict/{name}.bin")));
            assert!(!trained.is_empty(), "{name}");
            let primed = primed(lang as u8);
            assert_eq!(&primed.bytes[..WRAPPER.len()], WRAPPER, "{name}");
            assert_eq!(&primed.bytes[WRAPPER.len()..], trained, "{name}");
            dictionaries.push(trained);
        }
        assert_eq!(GRAM_TABLE, crate::grams::gram_table(&dictionaries));
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
