//! tokzip: at-rest compressor for prompts, LLM outputs, and source code. No options: the
//! encoder detects the language(s) of the input itself and picks the best of its embedded
//! dictionaries per segment.
//!
//! Frame layout (format v0 — pre-release, changes freely without compatibility):
//!   [0]    magic/version 0xD0 (high nibble 0b1101, low nibble = version 0)
//!   [1]    flags: bit 0 = content is bytes (0 = UTF-8 string), bit 1 = stored,
//!          bit 2 = multi-segment, bit 3 = blocked
//!   varint decompressed length
//!   CRC-32 (little-endian) of the decompressed content followed by the type flag byte
//!   stored: the content. Single segment: u8 language, then the range-coded body.
//!   Multi-segment: varint segment count, (u8 language, varint length)*, then the body.
//!   Blocked (content above `BLOCK_LEN`): per block a flags byte (stored or multi-segment), a
//!   varint payload length, and the payload — the block's content when stored, otherwise its
//!   segment table and body laid out as above and coded independently of the other blocks.
//!
//! `compress` verifies that every coded block decodes back to its input before returning the
//! frame and falls back to storing the block (or the whole content) otherwise, so a persisted
//! frame is provably recoverable.

mod lang;
mod lz;
mod rc;
#[cfg(feature = "train")]
pub mod train;

use lz::Segment;

pub const MAGIC_VERSION: u8 = 0xD0;
const FLAG_BYTES: u8 = 0b1;
const FLAG_STORED: u8 = 0b10;
const FLAG_MULTI: u8 = 0b100;
const FLAG_BLOCKED: u8 = 0b1000;
/// Content longer than this is coded as independent blocks of this size, which bounds the
/// coder's working set whatever the document length: coding keeps a 4-byte match-chain entry
/// per block byte plus the self-check's decoded copy of the block, and decoding builds one block
/// before appending it to the output — measured in Bun, coding a 4 MiB block adds ~24 MB of wasm
/// memory (which never shrinks), leaving a 128 MiB Cloudflare Workers isolate room for the
/// caller. Splitting only loses matches across block boundaries, negligible next to the
/// dictionaries and the local context.
const BLOCK_LEN: usize = 4 * 1024 * 1024;
/// In a blocked frame, a block longer than this is probed first: when this much of its start
/// does not shrink when coded on its own, the block is stored without coding the rest.
/// Incompressible content (already-compressed or encrypted blobs) would otherwise pay the full
/// parse (~0.4 s/MiB) only to be stored. Only blocked frames probe, so content up to `BLOCK_LEN`
/// is coded exactly as before and the probe costs a blocked frame at most ~6% (one probe per
/// 4 MiB block; a short final block is probed only past this length).
const PROBE_LEN: usize = 256 * 1024;
/// Upper bound on how much a coded body may expand: the codec tops out near 7,000× on
/// degenerate runs, so a declared length beyond this is a forged header and is rejected
/// before any output is allocated. It is the only bound the format itself puts on the output
/// of a blocked frame (a valid frame of highly repetitive content legitimately reaches it), so
/// callers decoding untrusted frames pass `decompress` a length limit.
const MAX_EXPANSION: usize = 8192;
const MAX_SEGMENTS: u64 = 1 << 20;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeError {
    Truncated,
    BadMagic,
    UnsupportedVersion,
    ChecksumMismatch,
    Corrupt,
    /// The declared content length exceeds the caller's limit, or the output cannot be allocated.
    TooLarge,
}

impl DecodeError {
    pub fn code(self) -> u32 {
        match self {
            Self::Truncated => 1,
            Self::BadMagic => 2,
            Self::UnsupportedVersion => 3,
            Self::ChecksumMismatch => 4,
            Self::Corrupt => 5,
            Self::TooLarge => 6,
        }
    }
}

impl std::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Truncated => write!(f, "frame truncated"),
            Self::BadMagic => write!(f, "bad magic byte"),
            Self::UnsupportedVersion => write!(f, "unsupported format version"),
            Self::ChecksumMismatch => write!(f, "content checksum mismatch"),
            Self::Corrupt => write!(f, "corrupt compressed body"),
            Self::TooLarge => write!(f, "content too large for the length limit or memory"),
        }
    }
}

impl std::error::Error for DecodeError {}

/// Compresses `content` into a self-describing frame. `is_bytes` records whether the caller
/// passed raw bytes (true) or a UTF-8 string (false); `decompress` reports it back.
pub fn compress(content: &[u8], is_bytes: bool) -> Vec<u8> {
    let type_flag = if is_bytes { FLAG_BYTES } else { 0 };
    let crc = content_crc(content, is_bytes);
    if let Some(frame) = coded_frame(content, type_flag, crc) {
        return frame;
    }
    let mut frame = Vec::with_capacity(stored_frame_len(content.len()));
    frame.push(MAGIC_VERSION);
    frame.push(type_flag | FLAG_STORED);
    push_varint(&mut frame, content.len() as u64);
    frame.extend_from_slice(&crc.to_le_bytes());
    frame.extend_from_slice(content);
    frame
}

/// The coded frame for `content`, or `None` when it would not be strictly smaller than the
/// stored frame. A block that does not shrink or fails its decode check is stored inside a
/// blocked frame; single-block content falls back to a stored frame instead.
fn coded_frame(content: &[u8], type_flag: u8, crc: u32) -> Option<Vec<u8>> {
    if content.is_empty() {
        return None;
    }
    let mut frame = vec![MAGIC_VERSION, type_flag];
    push_varint(&mut frame, content.len() as u64);
    frame.extend_from_slice(&crc.to_le_bytes());
    if content.len() <= BLOCK_LEN {
        let (multi, payload) = coded_block(content, false)?;
        frame[1] |= multi;
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
    frame[1] |= FLAG_BLOCKED;
    frame.reserve_exact(frame_len - frame.len());
    for (block, coded) in blocks {
        let (flags, payload) = match &coded {
            Some((multi, payload)) => (*multi, payload.as_slice()),
            None => (FLAG_STORED, block),
        };
        frame.push(flags);
        push_varint(&mut frame, payload.len() as u64);
        frame.extend_from_slice(payload);
    }
    Some(frame)
}

/// Codes one block and returns its payload with its multi-segment flag, or `None` when the
/// block is to be stored: the payload is not strictly smaller than the block or does not decode
/// back to it — or, with `probe`, the block's first `PROBE_LEN` bytes do not shrink on their own.
fn coded_block(block: &[u8], probe: bool) -> Option<(u8, Vec<u8>)> {
    if probe && block.len() > PROBE_LEN && coded_payload(&block[..PROBE_LEN]).1.len() >= PROBE_LEN {
        return None;
    }
    let (multi, payload) = coded_payload(block);
    let mut decoded = Vec::new();
    let recoverable = payload.len() < block.len()
        && decode_block(multi != 0, &payload, block.len(), &mut decoded).is_ok()
        && decoded == block;
    recoverable.then_some((multi, payload))
}

/// The segment table followed by the range-coded body of `block`, with the multi-segment flag.
fn coded_payload(block: &[u8]) -> (u8, Vec<u8>) {
    let (segments, body) = best_segmentation(block);
    let mut payload = Vec::with_capacity(segment_table_len(&segments) + body.len());
    let multi = if segments.len() > 1 { FLAG_MULTI } else { 0 };
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
    (multi, payload)
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
    body.len() + 2 + varint_len(content.len() as u64) + 4 + 1
}

/// Detected segments as `(end, language id)` pairs (diagnostic use).
#[doc(hidden)]
pub fn segments(content: &[u8]) -> Vec<(usize, u8)> {
    lang::segment(content)
        .into_iter()
        .map(|s| (s.end, s.lang))
        .collect()
}

/// CRC-32 over the content followed by its type flag, so a frame cannot be silently retyped.
fn content_crc(content: &[u8], is_bytes: bool) -> u32 {
    let mut hasher = crc32fast::Hasher::new();
    hasher.update(content);
    hasher.update(&[u8::from(is_bytes)]);
    hasher.finalize()
}

/// Codes `content` with the detected per-segment languages and returns the smallest whole
/// frame. When detection is uncertain (see the gate below) it also codes the input as a single
/// segment of each top candidate language — all of them up to 4 KiB, only the strongest gram
/// match up to 64 KiB, none above — and keeps the smallest, comparing full frame length (body plus the segment
/// table, which differs between single- and multi-segment frames). A confident single-language
/// detection is coded as detected without trying alternatives.
fn best_segmentation(content: &[u8]) -> (Vec<Segment>, Vec<u8>) {
    let (mut best_segments, gram_scores) = lang::analyze(content);
    let mut best_body = lz::encode_doc(&lang::primed, content, &best_segments);
    let mut best_cost = best_body.len() + segment_table_len(&best_segments);
    // Every candidate costs a full extra optimal parse (~0.35 ms at 4 KiB, ~2.8 ms at 18 KiB,
    // ~70 ms at 1 MiB), so up to 4 KiB all top candidates are tried, up to 64 KiB only the
    // strongest gram match, and above that none. The single parse matters at LLM-answer sizes:
    // multi-segment documents of 5-30 KB coded 1-4% larger as detected than as the best single
    // language (an 18 KB answer whose ```html fence pins a long `<script>` to the html
    // dictionary: 5,145 vs 4,977 bytes as `text`), and the winner was the top gram candidate.
    // The gain does not stop at 64 KiB (a 90 KB document of that shape still codes ~2.7% larger
    // than its best single language), but above it the extra parse costs more CPU (~15 ms at
    // 90 KB, ~70 ms at 1 MiB, doubling compression time) than a few percent is worth here.
    const FULL_SEARCH_MAX: usize = 4 * 1024;
    const SINGLE_CANDIDATE_MAX: usize = 64 * 1024;
    {
        let mut candidates = lang::top_languages(&gram_scores);
        if content.len() > SINGLE_CANDIDATE_MAX {
            candidates.clear();
        } else if content.len() > FULL_SEARCH_MAX {
            candidates.truncate(1);
        }
        let detected_single = (best_segments.len() == 1).then(|| best_segments[0].lang);
        // Search only when detection is uncertain — a multi-segment split, or a single language
        // that is not the strongest dictionary match (the fence-hint-overwhelm case). A
        // confident single-language detection (its segment is the top gram candidate) is
        // trusted as-is: it carries no never-worse-than-single guarantee, trading a few bytes on
        // rare ties for not running extra parses on the common pure-single-language document.
        if detected_single != candidates.first().copied() {
            for lang in candidates {
                if Some(lang) == detected_single {
                    continue;
                }
                let single = vec![Segment {
                    end: content.len(),
                    lang,
                }];
                let body = lz::encode_doc(&lang::primed, content, &single);
                let cost = body.len() + segment_table_len(&single);
                if cost < best_cost {
                    best_cost = cost;
                    best_body = body;
                    best_segments = single;
                }
            }
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
    2 + varint_len(content_len as u64) + 4 + content_len
}

/// Decompresses a frame produced by [`compress`], verifying the content CRC-32. Returns the
/// content and whether it was compressed from raw bytes (true) or a UTF-8 string (false).
/// A frame declaring more than `max_len` bytes of content is rejected before anything is
/// allocated: the format bounds the output only relative to the frame length
/// (`MAX_EXPANSION`), so this is the caller's guard against decompression bombs.
pub fn decompress(frame: &[u8], max_len: usize) -> Result<(Vec<u8>, bool), DecodeError> {
    if frame.len() < 7 {
        return Err(DecodeError::Truncated);
    }
    if frame[0] != MAGIC_VERSION {
        return Err(if frame[0] & 0xF0 == MAGIC_VERSION & 0xF0 {
            DecodeError::UnsupportedVersion
        } else {
            DecodeError::BadMagic
        });
    }
    let flags = frame[1];
    if flags & !(FLAG_BYTES | FLAG_STORED | FLAG_MULTI | FLAG_BLOCKED) != 0
        || (flags & FLAG_STORED != 0 && flags & (FLAG_MULTI | FLAG_BLOCKED) != 0)
        || flags & (FLAG_MULTI | FLAG_BLOCKED) == FLAG_MULTI | FLAG_BLOCKED
    {
        return Err(DecodeError::Corrupt);
    }
    let is_bytes = flags & FLAG_BYTES != 0;
    let (out_len, rest) = read_varint(&frame[2..])?;
    if rest.len() < 4 {
        return Err(DecodeError::Truncated);
    }
    let expected_crc = u32::from_le_bytes([rest[0], rest[1], rest[2], rest[3]]);
    let body = &rest[4..];
    if out_len > max_len as u64 {
        return Err(DecodeError::TooLarge);
    }
    if flags & FLAG_STORED != 0 {
        // Compared as u64: a huge declared length must not wrap when narrowed to usize.
        if (body.len() as u64) < out_len {
            return Err(DecodeError::Truncated);
        }
        if body.len() as u64 > out_len {
            return Err(DecodeError::Corrupt);
        }
        if content_crc(body, is_bytes) != expected_crc {
            return Err(DecodeError::ChecksumMismatch);
        }
        return Ok((body.to_vec(), is_bytes));
    }
    if out_len == 0 || out_len > body.len().saturating_mul(MAX_EXPANSION) as u64 {
        return Err(DecodeError::Corrupt);
    }
    let out_len = out_len as usize;
    let mut content = Vec::new();
    if flags & FLAG_BLOCKED != 0 {
        decode_blocks(body, out_len, &mut content)?;
    } else if out_len > BLOCK_LEN {
        return Err(DecodeError::Corrupt);
    } else {
        decode_block(flags & FLAG_MULTI != 0, body, out_len, &mut content)?;
    }
    if content_crc(&content, is_bytes) != expected_crc {
        return Err(DecodeError::ChecksumMismatch);
    }
    Ok((content, is_bytes))
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
        let (&flags, rest) = body.split_first().ok_or(DecodeError::Truncated)?;
        if flags & !(FLAG_STORED | FLAG_MULTI) != 0
            || flags & (FLAG_STORED | FLAG_MULTI) == FLAG_STORED | FLAG_MULTI
        {
            return Err(DecodeError::Corrupt);
        }
        let (payload_len, rest) = read_varint(rest)?;
        if payload_len > rest.len() as u64 {
            return Err(DecodeError::Truncated);
        }
        let (payload, rest) = rest.split_at(payload_len as usize);
        if flags & FLAG_STORED != 0 {
            if payload.len() != block_len {
                return Err(DecodeError::Corrupt);
            }
            content.extend_from_slice(payload);
        } else {
            decode_block(flags & FLAG_MULTI != 0, payload, block_len, content)?;
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

fn varint_len(mut v: u64) -> usize {
    let mut n = 1;
    while v >= 0x80 {
        v >>= 7;
        n += 1;
    }
    n
}

fn push_varint(out: &mut Vec<u8>, mut v: u64) {
    loop {
        let byte = (v & 0x7F) as u8;
        v >>= 7;
        if v == 0 {
            out.push(byte);
            break;
        }
        out.push(byte | 0x80);
    }
}

fn read_varint(buf: &[u8]) -> Result<(u64, &[u8]), DecodeError> {
    let mut v = 0u64;
    for (i, &byte) in buf.iter().enumerate().take(10) {
        // The tenth group holds only bit 63; anything above it would be shifted out silently.
        if i == 9 && byte > 1 {
            return Err(DecodeError::Corrupt);
        }
        v |= u64::from(byte & 0x7F) << (7 * i);
        if byte & 0x80 == 0 {
            // Canonical form: a multi-byte varint never ends in a zero group.
            if i > 0 && byte == 0 {
                return Err(DecodeError::Corrupt);
            }
            return Ok((v, &buf[i + 1..]));
        }
    }
    Err(if buf.len() < 10 {
        DecodeError::Truncated
    } else {
        DecodeError::Corrupt
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip(content: &[u8], is_bytes: bool) -> Vec<u8> {
        let frame = compress(content, is_bytes);
        let (restored, restored_bytes) = decompress(&frame, usize::MAX).expect("decode");
        assert_eq!(restored, content);
        assert_eq!(restored_bytes, is_bytes);
        frame
    }

    #[test]
    fn round_trips_and_compresses() {
        assert_eq!(round_trip(b"", false).len(), 7);
        round_trip(b"a", true);
        let prompt = "以下の要件を満たすブロック崩しゲームを作成してください。\n- キャンバスサイズは 800x600 とし、背景は暗い青にしてください。\n- パドルは左右矢印キーで移動し、画面端で止まります。\n".repeat(2);
        let frame = round_trip(prompt.as_bytes(), false);
        assert!(
            frame.len() * 2 < prompt.len(),
            "{} -> {}",
            prompt.len(),
            frame.len()
        );
        let code = "export function compress(input: string | Uint8Array): Uint8Array {\n  const bytes = typeof input === 'string' ? new TextEncoder().encode(input) : input;\n  return call(bytes);\n}\n";
        let frame = round_trip(code.as_bytes(), false);
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
        let frame = round_trip(&noise, true);
        assert_eq!(frame.len(), stored_frame_len(noise.len()));
        assert_eq!(frame[1] & FLAG_STORED, FLAG_STORED);
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
        let frame = round_trip(&content, true);
        assert_eq!(frame[1], FLAG_BYTES | FLAG_BLOCKED);
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
        let small = compress(&content[..1000], true);
        let mut flagged = small.clone();
        flagged[1] |= FLAG_BLOCKED;
        assert_eq!(decompress(&flagged, usize::MAX), Err(DecodeError::Corrupt));
        // The length limit is checked before any block is decoded.
        assert_eq!(
            decompress(&frame, content.len() - 1),
            Err(DecodeError::TooLarge)
        );
    }

    #[test]
    fn corrupt_frames_are_rejected() {
        let frame = compress(
            "hello hello hello hello hello world".repeat(4).as_bytes(),
            false,
        );
        assert_eq!(decompress(&[], usize::MAX), Err(DecodeError::Truncated));
        assert!(decompress(&frame[..frame.len() / 2], usize::MAX).is_err());
        let mut bad = frame.clone();
        bad[0] = 0xD1;
        assert_eq!(
            decompress(&bad, usize::MAX),
            Err(DecodeError::UnsupportedVersion)
        );
        bad[0] = 0x42;
        assert_eq!(decompress(&bad, usize::MAX), Err(DecodeError::BadMagic));
        // A stored empty frame whose length varint encodes 2^64 must not wrap to 0 and decode.
        let empty = compress(b"", false);
        let mut overflowing = vec![MAGIC_VERSION, FLAG_STORED];
        overflowing.extend_from_slice(&[0x80; 9]);
        overflowing.push(0x02);
        overflowing.extend_from_slice(&empty[3..]);
        assert_eq!(
            decompress(&overflowing, usize::MAX),
            Err(DecodeError::Corrupt)
        );
        for i in 1..frame.len() {
            let mut mutated = frame.clone();
            mutated[i] ^= 0x55;
            if let Ok((out, _)) = decompress(&mutated, usize::MAX) {
                assert_eq!(
                    out,
                    decompress(&frame, usize::MAX).unwrap().0,
                    "mutation at {i} decoded to different content"
                );
            }
        }
    }
}
