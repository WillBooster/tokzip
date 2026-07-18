//! tokzip-rs: server-side at-rest compressor for short code / LLM-output documents.
//!
//! Frame layout (format v0, pre-release):
//!   [0] magic 0xC2 (outside 0xB0-0xBF, which the TS format reserves for its
//!       binary frames (0b10110xxx, src/container.ts) and streams (0b10111xxx,
//!       src/stream.ts) with the low 3 bits as a version field)
//!   [1] version (0)
//!   [2] method (0 = stored, 1 = lzrc)
//!   [3..7] CRC-32 (little-endian) of the decompressed content
//!   [7..] method-specific body
//!
//! Method 1 (lzrc) body: LEB128 varint of the decompressed length, then the
//! range-coder stream (see [`lzrc`]). Compression falls back to `stored`
//! whenever lzrc would not be smaller, so a frame never expands beyond
//! `content + 8` bytes.

mod lzrc;
mod rc;

pub use lzrc::Dictionary;

pub const MAGIC: u8 = 0xC2;
pub const VERSION: u8 = 0;
const HEADER_LEN: usize = 7;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Method {
    Stored = 0,
    LzRc = 1,
}

#[derive(Debug)]
pub enum DecodeError {
    Truncated,
    BadMagic,
    UnsupportedVersion(u8),
    UnknownMethod(u8),
    ChecksumMismatch,
    Corrupt,
}

impl std::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Truncated => write!(f, "frame truncated"),
            Self::BadMagic => write!(f, "bad magic byte"),
            Self::UnsupportedVersion(v) => write!(f, "unsupported format version {v}"),
            Self::UnknownMethod(m) => write!(f, "unknown method {m}"),
            Self::ChecksumMismatch => write!(f, "content checksum mismatch"),
            Self::Corrupt => write!(f, "corrupt compressed body"),
        }
    }
}

impl std::error::Error for DecodeError {}

/// Compresses `content` into a self-describing frame. Passing the same
/// [`Dictionary`] to [`decompress`] is required to restore lzrc frames; the
/// `stored` fallback is dictionary-independent.
pub fn compress(content: &[u8], dictionary: Option<&Dictionary>) -> Vec<u8> {
    let empty;
    let dict = match dictionary {
        Some(d) => d,
        None => {
            empty = Dictionary::new(&[]);
            &empty
        }
    };
    let mut frame = Vec::with_capacity(HEADER_LEN + content.len());
    frame.push(MAGIC);
    frame.push(VERSION);
    frame.push(Method::LzRc as u8);
    frame.extend_from_slice(&crc32fast::hash(content).to_le_bytes());
    let body_start = frame.len();
    push_varint(&mut frame, content.len() as u64);
    frame.extend_from_slice(&lzrc::encode_doc(dict, content));
    if frame.len() - body_start >= content.len() {
        frame.truncate(body_start);
        frame[2] = Method::Stored as u8;
        frame.extend_from_slice(content);
    }
    frame
}

/// Decompresses a frame produced by [`compress`], verifying the content CRC-32.
pub fn decompress(frame: &[u8], dictionary: Option<&Dictionary>) -> Result<Vec<u8>, DecodeError> {
    if frame.len() < HEADER_LEN {
        return Err(DecodeError::Truncated);
    }
    if frame[0] != MAGIC {
        return Err(DecodeError::BadMagic);
    }
    if frame[1] != VERSION {
        return Err(DecodeError::UnsupportedVersion(frame[1]));
    }
    let expected_crc = u32::from_le_bytes([frame[3], frame[4], frame[5], frame[6]]);
    let body = &frame[HEADER_LEN..];
    let content = match frame[2] {
        m if m == Method::Stored as u8 => {
            // Stored content can be CRC-checked in place, before the output allocation.
            if crc32fast::hash(body) != expected_crc {
                return Err(DecodeError::ChecksumMismatch);
            }
            return Ok(body.to_vec());
        }
        m if m == Method::LzRc as u8 => {
            let (out_len, rc_body) = read_varint(body)?;
            let empty;
            let dict = match dictionary {
                Some(d) => d,
                None => {
                    empty = Dictionary::new(&[]);
                    &empty
                }
            };
            lzrc::decode_doc(dict, rc_body, out_len as usize)?
        }
        m => return Err(DecodeError::UnknownMethod(m)),
    };
    if crc32fast::hash(&content) != expected_crc {
        return Err(DecodeError::ChecksumMismatch);
    }
    Ok(content)
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
        v |= u64::from(byte & 0x7F) << (7 * i);
        if byte & 0x80 == 0 {
            return Ok((v, &buf[i + 1..]));
        }
    }
    Err(DecodeError::Corrupt)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips() {
        let input = "const answer = 42; // 日本語もOK".as_bytes();
        let frame = compress(input, None);
        assert_eq!(decompress(&frame, None).unwrap(), input);
    }

    #[test]
    fn round_trips_with_dictionary() {
        let dict = Dictionary::new(
            "const answer = 42; // dictionary text\n"
                .repeat(30)
                .as_bytes(),
        );
        let input = "const answer = 42; // 日本語もOK".as_bytes();
        let frame = compress(input, Some(&dict));
        assert_eq!(decompress(&frame, Some(&dict)).unwrap(), input);
        assert!(frame.len() < input.len());
    }

    #[test]
    fn incompressible_input_falls_back_to_stored() {
        let input: Vec<u8> = (0u32..300)
            .map(|i| (i.wrapping_mul(2_654_435_761) >> 13) as u8)
            .collect();
        let frame = compress(&input, None);
        assert!(frame.len() <= input.len() + HEADER_LEN + 1);
        assert_eq!(decompress(&frame, None).unwrap(), input);
    }

    #[test]
    fn rejects_corruption() {
        let mut frame = compress(b"hello world hello world hello world", None);
        // Flip a byte in the middle of the body: the final range-coder bytes can
        // be flush padding the decoder never reads, so corruption there is
        // legitimately unobservable.
        let mid = HEADER_LEN + (frame.len() - HEADER_LEN) / 2;
        frame[mid] ^= 0xFF;
        assert!(decompress(&frame, None).is_err());
    }

    #[test]
    fn rejects_truncation_and_garbage() {
        assert!(decompress(&[], None).is_err());
        assert!(decompress(&[0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06], None).is_err());
    }
}
