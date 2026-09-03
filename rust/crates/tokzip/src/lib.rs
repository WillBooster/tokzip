//! tokzip: at-rest compressor for prompts, LLM outputs, and source code. No options: the
//! encoder detects the language(s) of the input itself and picks the best of its embedded
//! dictionaries per segment.
//!
//! Frame layout (format v1; every change to the codec — algorithm, dictionaries, or priors —
//! is a new version, and a decoder reads every version released before it):
//!   [0]    header byte: high nibble 0b1101 (magic), bits 3–2 = version (1; 3 is reserved for
//!          an extension byte), bits 1–0 = layout (0 single-segment, 1 multi-segment,
//!          2 blocked, 3 stored)
//!   varint decompressed length
//!   CRC-32 (little-endian) of the decompressed content
//!   stored: the content. Single segment: u8 language, then the range-coded body.
//!   Multi-segment: varint segment count, (u8 language, varint length)*, then the body.
//!   Blocked (content above `BLOCK_LEN`): per block a layout byte (0, 1, or 3 as above), a
//!   varint payload length, and the payload — the block's content when stored, otherwise its
//!   segment table and body laid out as above and coded independently of the other blocks.
//!
//! `compress` verifies that every coded block decodes back to its input before returning the
//! frame and falls back to storing the block (or the whole content) otherwise, so a persisted
//! frame is provably recoverable.

mod error;
mod grams;
mod lang;
mod languages;
mod lz;
mod rc;
#[cfg(feature = "train")]
pub mod train;
mod varint;

pub use error::DecodeError;
use lz::Segment;
use varint::{push_varint, read_varint, varint_len};

const MAGIC: u8 = 0xD0;
const VERSION: u8 = 1;
/// The header byte of a version-1 frame with `layout`.
const fn header(layout: u8) -> u8 {
    MAGIC | (VERSION << 2) | layout
}
const LAYOUT_SINGLE: u8 = 0;
const LAYOUT_MULTI: u8 = 1;
const LAYOUT_BLOCKED: u8 = 2;
const LAYOUT_STORED: u8 = 3;
/// Content longer than this is coded as independent blocks of this size, which bounds the
/// coder's working set whatever the document length: coding keeps a 4-byte match-chain entry
/// per block byte plus the self-check's decoded copy of the block, and decoding appends each
/// block straight into the output — measured in Bun, coding a 4 MiB block adds ~24 MB of wasm
/// memory (which never shrinks), leaving a 128 MiB Cloudflare Workers isolate room for the
/// caller. Splitting only loses matches across block boundaries, negligible next to the
/// dictionaries and the local context.
const BLOCK_LEN: usize = 4 * 1024 * 1024;
/// In a blocked frame, a block longer than this is probed first: when this much of its start
/// does not shrink when coded on its own, the block is stored without coding the rest.
/// Incompressible content (already-compressed or encrypted blobs) would otherwise pay the full
/// parse (~0.4 s/MiB) only to be stored. Only blocked frames probe, so content up to `BLOCK_LEN`
/// is coded exactly as before and the probe costs a blocked frame ~6% per full 4 MiB block —
/// up to roughly twice that for a frame whose final block is little longer than the probe (a
/// final block up to this length is not probed).
const PROBE_LEN: usize = 256 * 1024;
/// Upper bound on how much a coded body may expand: the codec tops out near 7,000× on
/// degenerate runs, so a declared length beyond this is a forged header and is rejected
/// before any output is allocated. It is the only bound the format itself puts on the output
/// of a blocked frame (a valid frame of highly repetitive content legitimately reaches it), so
/// callers decoding untrusted frames pass `decompress` a length limit.
const MAX_EXPANSION: usize = 8192;
const MAX_SEGMENTS: u64 = 1 << 20;

/// Compresses `content` into a self-describing frame.
pub fn compress(content: &[u8]) -> Vec<u8> {
    let crc = crc32fast::hash(content);
    if let Some(frame) = coded_frame(content, crc) {
        return frame;
    }
    let mut frame = Vec::with_capacity(stored_frame_len(content.len()));
    frame.push(header(LAYOUT_STORED));
    push_varint(&mut frame, content.len() as u64);
    frame.extend_from_slice(&crc.to_le_bytes());
    frame.extend_from_slice(content);
    frame
}

/// The coded frame for `content`, or `None` when it would not be strictly smaller than the
/// stored frame. A block that does not shrink or fails its decode check is stored inside a
/// blocked frame; single-block content falls back to a stored frame instead.
fn coded_frame(content: &[u8], crc: u32) -> Option<Vec<u8>> {
    if content.is_empty() {
        return None;
    }
    let mut frame = vec![header(LAYOUT_SINGLE)];
    push_varint(&mut frame, content.len() as u64);
    frame.extend_from_slice(&crc.to_le_bytes());
    if content.len() <= BLOCK_LEN {
        let (layout, payload) = coded_block(content, false)?;
        frame[0] = header(layout);
        frame.extend_from_slice(&payload);
        return Some(frame);
    }
    // Every block is coded before the frame is laid out, so the frame is allocated once at its
    // exact length — no growth, which would leave holes in the (never shrinking) wasm heap —
    // and a block that stays stored is only a slice of the input until the frame is known to
    // beat the stored frame. The encoder therefore peaks at the input plus the coded payloads
    // plus the frame (at most twice the frame) plus one block's coding working set.
    let blocks: Vec<_> = content
        .chunks(BLOCK_LEN)
        .map(|block| (block, coded_block(block, true)))
        .collect();
    let frame_len = frame.len()
        + blocks
            .iter()
            .map(|(block, coded)| {
                let payload_len = coded.as_ref().map_or(block.len(), |(_, p)| p.len());
                1 + varint_len(payload_len as u64) + payload_len
            })
            .sum::<usize>();
    if frame_len >= stored_frame_len(content.len()) {
        return None;
    }
    frame[0] = header(LAYOUT_BLOCKED);
    frame.reserve_exact(frame_len - frame.len());
    for (block, coded) in blocks {
        let (layout, payload) = match &coded {
            Some((layout, payload)) => (*layout, payload.as_slice()),
            None => (LAYOUT_STORED, block),
        };
        frame.push(layout);
        push_varint(&mut frame, payload.len() as u64);
        frame.extend_from_slice(payload);
    }
    Some(frame)
}

/// Codes one block and returns its payload with its layout, or `None` when the
/// block is to be stored: the payload is not strictly smaller than the block or does not decode
/// back to it — or, with `probe`, the block's first `PROBE_LEN` bytes do not shrink on their own.
fn coded_block(block: &[u8], probe: bool) -> Option<(u8, Vec<u8>)> {
    if probe && block.len() > PROBE_LEN && coded_payload(&block[..PROBE_LEN]).1.len() >= PROBE_LEN {
        return None;
    }
    let (layout, payload) = coded_payload(block);
    // The expansion bound is the one decoder rule construction alone does not satisfy, so the
    // encoder asserts it too: a frame `compress` returns is decodable by construction and check.
    let mut decoded = Vec::new();
    let recoverable = payload.len() < block.len()
        && block.len() <= payload.len().saturating_mul(MAX_EXPANSION)
        && decode_block(layout == LAYOUT_MULTI, &payload, block.len(), &mut decoded).is_ok()
        && decoded == block;
    recoverable.then_some((layout, payload))
}

/// The segment table followed by the range-coded body of `block`, with its layout (single- or
/// multi-segment).
fn coded_payload(block: &[u8]) -> (u8, Vec<u8>) {
    let (segments, body) = best_segmentation(block);
    let mut payload = Vec::with_capacity(segment_table_len(&segments) + body.len());
    let layout = if segments.len() > 1 {
        LAYOUT_MULTI
    } else {
        LAYOUT_SINGLE
    };
    if segments.len() == 1 {
        payload.push(segments[0].lang);
    } else {
        push_varint(&mut payload, segments.len() as u64);
        let mut start = 0usize;
        for seg in &segments {
            payload.push(seg.lang);
            push_varint(&mut payload, (seg.end - start) as u64);
            start = seg.end;
        }
    }
    payload.extend_from_slice(&body);
    (layout, payload)
}

/// Frame length the content would have when coded as one segment of `lang` (diagnostic use;
/// the public API detects the language).
#[doc(hidden)]
pub fn frame_len_with_language(content: &[u8], lang: usize) -> usize {
    if lang >= lang::LANGUAGE_COUNT {
        return usize::MAX;
    }
    let segments = [Segment {
        end: content.len(),
        lang: lang as u8,
    }];
    let body = lz::encode_doc(&lang::primed, content, &segments);
    body.len() + 1 + varint_len(content.len() as u64) + 4 + 1
}

/// Embedded language names, in id order (diagnostic use).
#[doc(hidden)]
pub fn language_names() -> Vec<&'static str> {
    languages::LANGUAGES.iter().map(|(name, _)| *name).collect()
}

/// Detected segments as `(end, language id)` pairs (diagnostic use).
#[doc(hidden)]
pub fn segments(content: &[u8]) -> Vec<(usize, u8)> {
    lang::segment(content)
        .into_iter()
        .map(|s| (s.end, s.lang))
        .collect()
}

/// Codes `content` with the detected per-segment languages and returns the smallest whole
/// frame. When detection is uncertain (see the gate below) it also codes the input as a single
/// segment of the strongest gram-match language (up to 64 KiB) and keeps the smaller,
/// comparing full payload length (body plus the segment table, which differs between single-
/// and multi-segment frames). A confident single-language detection is coded as detected
/// without trying alternatives.
fn best_segmentation(content: &[u8]) -> (Vec<Segment>, Vec<u8>) {
    let (mut best_segments, gram_scores) = lang::analyze(content);
    let mut best_body = lz::encode_doc(&lang::primed, content, &best_segments);
    let best_cost = best_body.len() + segment_table_len(&best_segments);
    // The extra parse costs as much as the first (~0.35 ms at 4 KiB, ~2.8 ms at 18 KiB,
    // ~70 ms at 1 MiB), so it runs only up to 64 KiB. It matters at LLM-answer sizes:
    // multi-segment documents of 5-30 KB coded 1-4% larger as detected than as the best single
    // language (an 18 KB answer whose ```html fence pins a long `<script>` to the html
    // dictionary: 5,145 vs 4,977 bytes as `text`), and the winner was the top gram candidate;
    // trying further candidates changed nothing measurable. The gain does not stop at 64 KiB
    // (a 90 KB document of that shape still codes ~2.7% larger than its best single language),
    // but above it the extra parse costs more CPU (~15 ms at 90 KB, ~70 ms at 1 MiB, doubling
    // compression time) than a few percent is worth here.
    const CANDIDATE_MAX: usize = 64 * 1024;
    let top = lang::top_language(&gram_scores);
    let detected_single = (best_segments.len() == 1).then(|| best_segments[0].lang);
    // Search only when detection is uncertain — a multi-segment split, or a single language
    // that is not the strongest dictionary match (the fence-hint-overwhelm case). A confident
    // single-language detection (its segment is the top gram candidate) is trusted as-is: it
    // carries no never-worse-than-single guarantee, trading a few bytes on rare ties for not
    // running an extra parse on the common pure-single-language document.
    if content.len() <= CANDIDATE_MAX && detected_single != Some(top) {
        let single = vec![Segment {
            end: content.len(),
            lang: top,
        }];
        let body = lz::encode_doc(&lang::primed, content, &single);
        if body.len() + segment_table_len(&single) < best_cost {
            best_body = body;
            best_segments = single;
        }
    }
    (best_segments, best_body)
}

/// Bytes the segment table occupies in a payload: one language byte for a single segment, else
/// the segment-count varint plus a language byte and length varint per segment (see
/// `coded_payload`).
fn segment_table_len(segments: &[Segment]) -> usize {
    if segments.len() == 1 {
        return 1;
    }
    let mut len = varint_len(segments.len() as u64);
    let mut start = 0;
    for segment in segments {
        len += 1 + varint_len((segment.end - start) as u64);
        start = segment.end;
    }
    len
}

fn stored_frame_len(content_len: usize) -> usize {
    1 + varint_len(content_len as u64) + 4 + content_len
}

/// Decompresses a frame produced by [`compress`], verifying the content CRC-32. A frame
/// declaring more than `max_len` bytes of content is rejected before anything is allocated:
/// the format bounds the output only relative to the frame length (`MAX_EXPANSION`), so this
/// is the caller's guard against decompression bombs.
pub fn decompress(frame: &[u8], max_len: usize) -> Result<Vec<u8>, DecodeError> {
    if frame.len() < 6 {
        return Err(DecodeError::Truncated);
    }
    if frame[0] & 0xF0 != MAGIC {
        return Err(DecodeError::BadMagic);
    }
    if (frame[0] >> 2) & 3 != VERSION {
        return Err(DecodeError::UnsupportedVersion);
    }
    let layout = frame[0] & 3;
    let (out_len, rest) = read_varint(&frame[1..])?;
    if rest.len() < 4 {
        return Err(DecodeError::Truncated);
    }
    let expected_crc = u32::from_le_bytes([rest[0], rest[1], rest[2], rest[3]]);
    let body = &rest[4..];
    if out_len > max_len as u64 {
        return Err(DecodeError::TooLarge);
    }
    if layout == LAYOUT_STORED {
        // Compared as u64: a huge declared length must not wrap when narrowed to usize.
        if (body.len() as u64) < out_len {
            return Err(DecodeError::Truncated);
        }
        if body.len() as u64 > out_len {
            return Err(DecodeError::Corrupt);
        }
        if crc32fast::hash(body) != expected_crc {
            return Err(DecodeError::ChecksumMismatch);
        }
        let mut content = Vec::new();
        content
            .try_reserve_exact(body.len())
            .map_err(|_| DecodeError::TooLarge)?;
        content.extend_from_slice(body);
        return Ok(content);
    }
    if out_len == 0 || out_len > body.len().saturating_mul(MAX_EXPANSION) as u64 {
        return Err(DecodeError::Corrupt);
    }
    let out_len = out_len as usize;
    let mut content = Vec::new();
    if layout == LAYOUT_BLOCKED {
        decode_blocks(body, out_len, &mut content)?;
    } else if out_len > BLOCK_LEN {
        return Err(DecodeError::Corrupt);
    } else {
        decode_block(layout == LAYOUT_MULTI, body, out_len, &mut content)?;
    }
    if crc32fast::hash(&content) != expected_crc {
        return Err(DecodeError::ChecksumMismatch);
    }
    Ok(content)
}

/// Decodes the blocks of a blocked frame body, appending `out_len` bytes to `content`.
fn decode_blocks(
    mut body: &[u8],
    out_len: usize,
    content: &mut Vec<u8>,
) -> Result<(), DecodeError> {
    // Content that fits one block has the single-block layout; here it is a structural error.
    if out_len <= BLOCK_LEN {
        return Err(DecodeError::Corrupt);
    }
    // Grown one block at a time, so a forged length costs no allocation beyond the blocks that
    // decode, and a frame the memory cannot hold (no caller limit) fails instead of aborting the
    // module. Blocks decode straight into `content`, which therefore stays the heap's top
    // allocation and extends in place rather than being copied.
    while content.len() < out_len {
        let block_len = BLOCK_LEN.min(out_len - content.len());
        content
            .try_reserve_exact(block_len)
            .map_err(|_| DecodeError::TooLarge)?;
        let (&layout, rest) = body.split_first().ok_or(DecodeError::Truncated)?;
        if layout == LAYOUT_BLOCKED || layout > LAYOUT_STORED {
            return Err(DecodeError::Corrupt);
        }
        let (payload_len, rest) = read_varint(rest)?;
        if payload_len > rest.len() as u64 {
            return Err(DecodeError::Truncated);
        }
        let (payload, rest) = rest.split_at(payload_len as usize);
        if layout == LAYOUT_STORED {
            if payload.len() != block_len {
                return Err(DecodeError::Corrupt);
            }
            content.extend_from_slice(payload);
        } else {
            decode_block(layout == LAYOUT_MULTI, payload, block_len, content)?;
        }
        body = rest;
    }
    if !body.is_empty() {
        return Err(DecodeError::Corrupt);
    }
    Ok(())
}

/// Decodes one block's payload — its segment table followed by the range-coded body —
/// appending `block_len` bytes to `out`.
fn decode_block(
    multi: bool,
    payload: &[u8],
    block_len: usize,
    out: &mut Vec<u8>,
) -> Result<(), DecodeError> {
    let (segment_count, mut rest) = if multi {
        read_varint(payload)?
    } else {
        (1, payload)
    };
    if segment_count == 0 || segment_count > MAX_SEGMENTS {
        return Err(DecodeError::Corrupt);
    }
    let mut segments = Vec::with_capacity(segment_count.min(64) as usize);
    let mut end = 0usize;
    if !multi {
        let (&lang, after) = rest.split_first().ok_or(DecodeError::Truncated)?;
        if usize::from(lang) >= lang::LANGUAGE_COUNT {
            return Err(DecodeError::Corrupt);
        }
        segments.push(Segment {
            end: block_len,
            lang,
        });
        end = block_len;
        rest = after;
    }
    for _ in 0..if multi { segment_count } else { 0 } {
        let (&lang, after) = rest.split_first().ok_or(DecodeError::Truncated)?;
        if usize::from(lang) >= lang::LANGUAGE_COUNT {
            return Err(DecodeError::Corrupt);
        }
        let (len, after) = read_varint(after)?;
        if len == 0 || len > (block_len - end) as u64 {
            return Err(DecodeError::Corrupt);
        }
        end += len as usize;
        segments.push(Segment { end, lang });
        rest = after;
    }
    if end != block_len {
        return Err(DecodeError::Corrupt);
    }
    lz::decode_doc(&lang::primed, rest, block_len, &segments, out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip(content: &[u8]) -> Vec<u8> {
        let frame = compress(content);
        let restored = decompress(&frame, usize::MAX).expect("decode");
        assert_eq!(restored, content);
        frame
    }

    #[test]
    fn round_trips_and_compresses() {
        assert_eq!(round_trip(b"").len(), 6);
        round_trip(b"a");
        let prompt = "以下の要件を満たすブロック崩しゲームを作成してください。\n- キャンバスサイズは 800x600 とし、背景は暗い青にしてください。\n- パドルは左右矢印キーで移動し、画面端で止まります。\n".repeat(2);
        let frame = round_trip(prompt.as_bytes());
        assert!(
            frame.len() * 2 < prompt.len(),
            "{} -> {}",
            prompt.len(),
            frame.len()
        );
        let code = "export function compress(input: string | Uint8Array): Uint8Array {\n  const bytes = typeof input === 'string' ? new TextEncoder().encode(input) : input;\n  return call(bytes);\n}\n";
        let frame = round_trip(code.as_bytes());
        assert!(
            frame.len() * 4 < code.len() * 3,
            "{} -> {}",
            code.len(),
            frame.len()
        );
    }

    fn noise(len: usize) -> Vec<u8> {
        let mut x = 0x2545_F491u32;
        (0..len)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                (x >> 24) as u8
            })
            .collect()
    }

    #[test]
    fn incompressible_input_is_stored() {
        let noise = noise(300);
        let frame = round_trip(&noise);
        assert_eq!(frame.len(), stored_frame_len(noise.len()));
        assert_eq!(frame[0], header(LAYOUT_STORED));
    }

    #[test]
    fn content_above_a_block_is_coded_per_block() {
        // A compressible block, an incompressible one (stored inside the frame), and a short
        // compressible tail block.
        let text = "const answer = 42; // the answer\nゲームの仕様を以下に示します。\n"
            .repeat(BLOCK_LEN / 60);
        let mut content = text.as_bytes()[..BLOCK_LEN].to_vec();
        content.extend_from_slice(&noise(BLOCK_LEN));
        content.extend_from_slice(&text.as_bytes()[..1000]);
        let frame = round_trip(&content);
        assert_eq!(frame[0], header(LAYOUT_BLOCKED));
        // The stored block costs its own length; both coded blocks shrink to almost nothing.
        assert!(frame.len() < BLOCK_LEN + 20_000, "{}", frame.len());
        // The last block's declared payload length must match its bytes.
        assert_eq!(
            decompress(&frame[..frame.len() - 1], usize::MAX),
            Err(DecodeError::Truncated)
        );
        let mut extra = frame.clone();
        extra.push(0);
        assert_eq!(decompress(&extra, usize::MAX), Err(DecodeError::Corrupt));
        // A blocked flag on content that fits one block is a structural error.
        let small = compress(&content[..1000]);
        let mut flagged = small.clone();
        flagged[0] = header(LAYOUT_BLOCKED);
        assert_eq!(decompress(&flagged, usize::MAX), Err(DecodeError::Corrupt));
        // The length limit is checked before any block is decoded.
        assert_eq!(
            decompress(&frame, content.len() - 1),
            Err(DecodeError::TooLarge)
        );
    }

    /// Every decode of damaged or random input must return an error or the exact original —
    /// never panic (a panic aborts the wasm module) and never other content.
    #[test]
    fn damaged_and_random_frames_never_panic_or_misdecode() {
        let mut x = 0x9E37_79B9u32;
        let mut next = || {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            x
        };
        let mut docs: Vec<Vec<u8>> = vec![
            "以下の要件を満たすゲームを作成してください。"
                .repeat(30)
                .into_bytes(),
            b"const answer = 42; // the answer\n".repeat(40),
            noise(3000),
        ];
        let mut mixed = "# 仕様\n\n```js\n".to_string().into_bytes();
        mixed.extend_from_slice(&b"function f(a) { return a + 1; }\n".repeat(60));
        mixed.extend_from_slice("```\n以上です。\n".repeat(20).as_bytes());
        docs.push(mixed);
        for doc in &docs {
            let frame = compress(doc);
            for _ in 0..300 {
                let mut mutated = frame.clone();
                for _ in 0..1 + (next() % 4) {
                    let i = next() as usize % mutated.len();
                    mutated[i] ^= (next() % 255 + 1) as u8;
                }
                if next() % 4 == 0 {
                    mutated.truncate(next() as usize % mutated.len());
                }
                if let Ok(out) = decompress(&mutated, usize::MAX) {
                    assert_eq!(&out, doc);
                }
            }
        }
        for len in 0..200 {
            let mut garbage: Vec<u8> = (0..len).map(|_| next() as u8).collect();
            if let Some(first) = garbage.first_mut() {
                *first = header(next() as u8 & 3);
            }
            assert!(decompress(&garbage, usize::MAX).is_err());
        }
    }

    #[test]
    fn corrupt_frames_are_rejected() {
        let frame = compress("hello hello hello hello hello world".repeat(4).as_bytes());
        assert_eq!(decompress(&[], usize::MAX), Err(DecodeError::Truncated));
        assert!(decompress(&frame[..frame.len() / 2], usize::MAX).is_err());
        let mut bad = frame.clone();
        for version in (0..4).filter(|&v| v != VERSION) {
            bad[0] = MAGIC | (version << 2);
            assert_eq!(
                decompress(&bad, usize::MAX),
                Err(DecodeError::UnsupportedVersion)
            );
        }
        bad[0] = 0x42;
        assert_eq!(decompress(&bad, usize::MAX), Err(DecodeError::BadMagic));
        // A stored empty frame whose length varint encodes 2^64 must not wrap to 0 and decode.
        let empty = compress(b"");
        let mut overflowing = vec![header(LAYOUT_STORED)];
        overflowing.extend_from_slice(&[0x80; 9]);
        overflowing.push(0x02);
        overflowing.extend_from_slice(&empty[2..]);
        assert_eq!(
            decompress(&overflowing, usize::MAX),
            Err(DecodeError::Corrupt)
        );
        for i in 1..frame.len() {
            let mut mutated = frame.clone();
            mutated[i] ^= 0x55;
            if let Ok(out) = decompress(&mutated, usize::MAX) {
                assert_eq!(
                    out,
                    decompress(&frame, usize::MAX).unwrap(),
                    "mutation at {i} decoded to different content"
                );
            }
        }
    }
}
