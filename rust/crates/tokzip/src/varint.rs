//! LEB128 varints (7 payload bits per byte, bit 7 = continue, canonical: a multi-byte varint
//! never ends in a zero byte).

use crate::DecodeError;

pub fn varint_len(mut v: u64) -> usize {
    let mut n = 1;
    while v >= 0x80 {
        v >>= 7;
        n += 1;
    }
    n
}

pub fn push_varint(out: &mut Vec<u8>, mut v: u64) {
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

pub fn read_varint(buf: &[u8]) -> Result<(u64, &[u8]), DecodeError> {
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
