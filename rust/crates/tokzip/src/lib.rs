//! tokzip-rs: server-side at-rest compressor for short code / LLM-output documents.
//!
//! Frame layout (format v0, pre-release):
//!   [0] magic 0xC2 (outside 0xB0-0xBF, which the TS format reserves for its
//!       binary frames (0b10110xxx, src/container.ts) and streams (0b10111xxx,
//!       src/stream.ts) with the low 3 bits as a version field)
//!   [1] version (0)
//!   [2] method (0 = stored)
//!   [3..7] CRC-32 (little-endian) of the decompressed content
//!   [7..] method-specific body
//!
//! v0 ships only the `stored` method so the frame envelope, integrity check, and
//! benchmark plumbing are exercised end-to-end; compression methods land next.

pub const MAGIC: u8 = 0xC2;
pub const VERSION: u8 = 0;
const HEADER_LEN: usize = 7;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Method {
    Stored = 0,
}

#[derive(Debug)]
pub enum DecodeError {
    Truncated,
    BadMagic,
    UnsupportedVersion(u8),
    UnknownMethod(u8),
    ChecksumMismatch,
}

impl std::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Truncated => write!(f, "frame truncated"),
            Self::BadMagic => write!(f, "bad magic byte"),
            Self::UnsupportedVersion(v) => write!(f, "unsupported format version {v}"),
            Self::UnknownMethod(m) => write!(f, "unknown method {m}"),
            Self::ChecksumMismatch => write!(f, "content checksum mismatch"),
        }
    }
}

impl std::error::Error for DecodeError {}

/// Compresses `content` into a self-describing frame. `dictionary` is reserved
/// until real methods land (v0 always emits a stored frame).
pub fn compress(content: &[u8], _dictionary: Option<&[u8]>) -> Vec<u8> {
    let mut frame = Vec::with_capacity(HEADER_LEN + content.len());
    frame.push(MAGIC);
    frame.push(VERSION);
    frame.push(Method::Stored as u8);
    frame.extend_from_slice(&crc32fast::hash(content).to_le_bytes());
    frame.extend_from_slice(content);
    frame
}

/// Decompresses a frame produced by [`compress`], verifying the content CRC-32.
pub fn decompress(frame: &[u8], _dictionary: Option<&[u8]>) -> Result<Vec<u8>, DecodeError> {
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
    match frame[2] {
        // Stored content can be CRC-checked in place, before the output allocation.
        m if m == Method::Stored as u8 => {
            let body = &frame[HEADER_LEN..];
            if crc32fast::hash(body) != expected_crc {
                return Err(DecodeError::ChecksumMismatch);
            }
            Ok(body.to_vec())
        }
        m => Err(DecodeError::UnknownMethod(m)),
    }
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
    fn rejects_corruption() {
        let mut frame = compress(b"hello world", None);
        let last = frame.len() - 1;
        frame[last] ^= 0xFF;
        assert!(matches!(
            decompress(&frame, None),
            Err(DecodeError::ChecksumMismatch)
        ));
    }

    #[test]
    fn rejects_truncation_and_garbage() {
        assert!(decompress(&[], None).is_err());
        assert!(decompress(&[0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06], None).is_err());
    }
}
