//! The 4-gram hash the language detector scores documents with (`lang.rs`) and the bitset of
//! a dictionary's grams the build script precomputes for it (`build.rs`), so the first
//! `compress` never has to decode every dictionary just to build the detection table.

pub const GRAM_BITS: u32 = 17;
/// Bytes of a dictionary's gram bitset: one bit per hash value.
pub const GRAM_SET_BYTES: usize = (1 << GRAM_BITS) / 8;

#[inline]
pub fn gram_hash(bytes: &[u8], pos: usize) -> usize {
    let v = u32::from_le_bytes([bytes[pos], bytes[pos + 1], bytes[pos + 2], bytes[pos + 3]]);
    (v.wrapping_mul(0x9E37_79B1) >> (32 - GRAM_BITS)) as usize
}

/// The bitset of the 4-gram hashes of `bytes`.
// The library only reads the sets the build script wrote; its tests recompute them.
#[cfg_attr(not(test), allow(dead_code))]
pub fn gram_set(bytes: &[u8]) -> Vec<u8> {
    let mut set = vec![0u8; GRAM_SET_BYTES];
    for pos in 0..bytes.len().saturating_sub(3) {
        let h = gram_hash(bytes, pos);
        set[h >> 3] |= 1 << (h & 7);
    }
    set
}
