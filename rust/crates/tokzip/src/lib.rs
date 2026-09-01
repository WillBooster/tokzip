//! tokzip: at-rest compressor for prompts, LLM outputs, and source code. No options: the
//! encoder detects the language(s) of the input itself and picks the best of its embedded
//! dictionaries per segment.
//!
//! Frame layout (format v0 — pre-release, changes freely without compatibility):
//!   [0]    magic/version 0xD0 (high nibble 0b1101, low nibble = version 0)
//!   [1]    flags: bit 0 = content is bytes (0 = UTF-8 string), bit 1 = stored,
//!          bit 2 = multi-segment
//!   varint decompressed length
//!   CRC-32 (little-endian) of the decompressed content followed by the type flag byte
//!   stored: the content. Single segment: u8 language, then the range-coded body.
//!   Multi-segment: varint segment count, (u8 language, varint length)*, then the body.
//!
//! `compress` verifies that its own output decodes back to the input before returning it and
//! falls back to a stored frame otherwise, so a persisted frame is provably recoverable.

mod lang;
mod lz;
mod rc;
#[cfg(feature = "train")]
pub mod train;

use lz::Segment;

pub const MAGIC_VERSION: u8 = 0xD0;
const FLAG_BYTES: u8 = 0b01;
const FLAG_STORED: u8 = 0b10;
const FLAG_MULTI: u8 = 0b100;
/// Upper bound on a coded frame's declared decompressed length, so a corrupt or forged header
/// cannot drive a large allocation, and the ceiling above which `compress` skips coding (see
/// there) and stores the content verbatim. Both paths cost a multiple of the content: coding
/// keeps the input copy, a 4-byte match-chain entry per input byte, the body, and the
/// self-check's decoded copy live at once, and decoding hands the caller a copy of the output
/// (plus a decoded string) on top of the wasm-side buffer — measured in Bun, a 4 MiB document
/// adds ~60 MB to the process. Sized for the intended at-rest payloads (prompts, LLM outputs,
/// cache entries) to leave a 128 MiB Cloudflare Workers isolate room for the caller.
const MAX_DECOMPRESSED_LEN: u64 = 4 * 1024 * 1024;
/// Upper bound on how much a coded body may expand: the codec tops out near 7,000× on
/// degenerate runs, so a declared length beyond this is a forged header and is rejected
/// before any output is allocated (a corrupt varint cannot force a large allocation).
const MAX_EXPANSION: usize = 8192;
const MAX_SEGMENTS: u64 = 1 << 20;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeError {
    Truncated,
    BadMagic,
    UnsupportedVersion,
    ChecksumMismatch,
    Corrupt,
}

impl DecodeError {
    pub fn code(self) -> u32 {
        match self {
            Self::Truncated => 1,
            Self::BadMagic => 2,
            Self::UnsupportedVersion => 3,
            Self::ChecksumMismatch => 4,
            Self::Corrupt => 5,
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
        }
    }
}

impl std::error::Error for DecodeError {}

/// Compresses `content` into a self-describing frame. `is_bytes` records whether the caller
/// passed raw bytes (true) or a UTF-8 string (false); `decompress` reports it back.
pub fn compress(content: &[u8], is_bytes: bool) -> Vec<u8> {
    let type_flag = if is_bytes { FLAG_BYTES } else { 0 };
    let crc = content_crc(content, is_bytes);
    // Content above the coded-frame cap can only ship as a stored frame, so it skips coding
    // entirely — the parser would otherwise allocate match-finder chains several times the
    // input size before the self-check fell back to stored anyway.
    if !content.is_empty() && content.len() as u64 <= MAX_DECOMPRESSED_LEN {
        let (segments, body) = best_segmentation(content);
        let mut frame = Vec::with_capacity(16 + segments.len() * 4 + body.len());
        frame.push(MAGIC_VERSION);
        frame.push(type_flag | if segments.len() > 1 { FLAG_MULTI } else { 0 });
        push_varint(&mut frame, content.len() as u64);
        frame.extend_from_slice(&crc.to_le_bytes());
        if segments.len() == 1 {
            frame.push(segments[0].lang);
        } else {
            push_varint(&mut frame, segments.len() as u64);
            let mut start = 0usize;
            for seg in &segments {
                frame.push(seg.lang);
                push_varint(&mut frame, (seg.end - start) as u64);
                start = seg.end;
            }
        }
        frame.extend_from_slice(&body);
        let stored_len = stored_frame_len(content.len());
        if frame.len() < stored_len
            && decompress(&frame)
                .map(|(out, b)| out == content && b == is_bytes)
                .unwrap_or(false)
        {
            return frame;
        }
    }
    let mut frame = Vec::with_capacity(stored_frame_len(content.len()));
    frame.push(MAGIC_VERSION);
    frame.push(type_flag | FLAG_STORED);
    push_varint(&mut frame, content.len() as u64);
    frame.extend_from_slice(&crc.to_le_bytes());
    frame.extend_from_slice(content);
    frame
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
/// frame. For a small input whose detection is uncertain (see the gate below) it also codes
/// the input as each top candidate language as a single segment and keeps the smallest,
/// comparing full frame length (body plus the segment table, which differs between single- and
/// multi-segment frames). A confident single-language detection is coded as detected without
/// trying alternatives.
fn best_segmentation(content: &[u8]) -> (Vec<Segment>, Vec<u8>) {
    let (mut best_segments, gram_scores) = lang::analyze(content);
    let mut best_body = lz::encode_doc(&lang::primed, content, &best_segments);
    let mut best_cost = best_body.len() + segment_table_len(&best_segments);
    // Every candidate costs a full extra optimal parse (~0.35 ms at 4 KiB, ~2.8 ms at 18 KiB), so
    // up to 4 KiB all top candidates are tried and above it only the strongest gram match is.
    // That one parse matters: on multi-segment documents above 4 KiB the detected split coded
    // 1-4% larger than the best single language (an 18 KB LLM answer whose ```html fence pins a
    // long `<script>` to the html dictionary: 5,145 vs 4,977 bytes as `text`), and the winner
    // was the top gram candidate.
    const FULL_SEARCH_MAX: usize = 4 * 1024;
    {
        let mut candidates = lang::top_languages(&gram_scores);
        if content.len() > FULL_SEARCH_MAX {
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

/// Bytes the segment table occupies in a frame: one language byte for a single segment, else
/// the segment-count varint plus a language byte and length varint per segment (see `compress`).
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
pub fn decompress(frame: &[u8]) -> Result<(Vec<u8>, bool), DecodeError> {
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
    if flags & !(FLAG_BYTES | FLAG_STORED | FLAG_MULTI) != 0
        || flags & (FLAG_STORED | FLAG_MULTI) == FLAG_STORED | FLAG_MULTI
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
    if out_len == 0
        || out_len > MAX_DECOMPRESSED_LEN
        || out_len > body.len().saturating_mul(MAX_EXPANSION) as u64
    {
        return Err(DecodeError::Corrupt);
    }
    let out_len = out_len as usize;
    let (segment_count, mut rest) = if flags & FLAG_MULTI != 0 {
        read_varint(body)?
    } else {
        (1, body)
    };
    if segment_count == 0 || segment_count > MAX_SEGMENTS {
        return Err(DecodeError::Corrupt);
    }
    let mut segments = Vec::with_capacity(segment_count.min(64) as usize);
    let mut end = 0usize;
    if flags & FLAG_MULTI == 0 {
        let (&lang, after) = rest.split_first().ok_or(DecodeError::Truncated)?;
        if usize::from(lang) >= lang::LANGUAGE_COUNT {
            return Err(DecodeError::Corrupt);
        }
        segments.push(Segment { end: out_len, lang });
        end = out_len;
        rest = after;
    }
    for _ in 0..if flags & FLAG_MULTI != 0 {
        segment_count
    } else {
        0
    } {
        let (&lang, after) = rest.split_first().ok_or(DecodeError::Truncated)?;
        if usize::from(lang) >= lang::LANGUAGE_COUNT {
            return Err(DecodeError::Corrupt);
        }
        let (len, after) = read_varint(after)?;
        if len == 0 || len > (out_len - end) as u64 {
            return Err(DecodeError::Corrupt);
        }
        end += len as usize;
        segments.push(Segment { end, lang });
        rest = after;
    }
    if end != out_len {
        return Err(DecodeError::Corrupt);
    }
    let content = lz::decode_doc(&lang::primed, rest, out_len, &segments)?;
    if content_crc(&content, is_bytes) != expected_crc {
        return Err(DecodeError::ChecksumMismatch);
    }
    Ok((content, is_bytes))
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
        let (restored, restored_bytes) = decompress(&frame).expect("decode");
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

    #[test]
    fn incompressible_input_is_stored() {
        let mut x = 0x2545_F491u32;
        let noise: Vec<u8> = (0..300)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                (x >> 24) as u8
            })
            .collect();
        let frame = round_trip(&noise, true);
        assert_eq!(frame.len(), stored_frame_len(noise.len()));
        assert_eq!(frame[1] & FLAG_STORED, FLAG_STORED);
    }

    #[test]
    fn corrupt_frames_are_rejected() {
        let frame = compress(
            "hello hello hello hello hello world".repeat(4).as_bytes(),
            false,
        );
        assert_eq!(decompress(&[]), Err(DecodeError::Truncated));
        assert!(decompress(&frame[..frame.len() / 2]).is_err());
        let mut bad = frame.clone();
        bad[0] = 0xD1;
        assert_eq!(decompress(&bad), Err(DecodeError::UnsupportedVersion));
        bad[0] = 0x42;
        assert_eq!(decompress(&bad), Err(DecodeError::BadMagic));
        // A stored empty frame whose length varint encodes 2^64 must not wrap to 0 and decode.
        let empty = compress(b"", false);
        let mut overflowing = vec![MAGIC_VERSION, FLAG_STORED];
        overflowing.extend_from_slice(&[0x80; 9]);
        overflowing.push(0x02);
        overflowing.extend_from_slice(&empty[3..]);
        assert_eq!(decompress(&overflowing), Err(DecodeError::Corrupt));
        for i in 1..frame.len() {
            let mut mutated = frame.clone();
            mutated[i] ^= 0x55;
            if let Ok((out, _)) = decompress(&mutated) {
                assert_eq!(
                    out,
                    decompress(&frame).unwrap().0,
                    "mutation at {i} decoded to different content"
                );
            }
        }
    }
}
