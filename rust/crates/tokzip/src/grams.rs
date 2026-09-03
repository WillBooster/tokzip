//! The 4-gram hash the language detector scores documents with (`lang.rs`) and the detection
//! table the build script precomputes for it (`build.rs`): for every hash value, the set of
//! languages whose trained dictionary contains a gram with that hash, as a bit per language id
//! in three little-endian bytes — so the first `compress` builds nothing and decodes no
//! dictionary for detection.

pub const GRAM_BITS: u32 = 17;
/// Bytes per table entry (room for 24 language bits).
pub const GRAM_ENTRY_BYTES: usize = 3;
/// Bytes of the detection table.
pub const GRAM_TABLE_BYTES: usize = (1 << GRAM_BITS) * GRAM_ENTRY_BYTES;

#[inline]
pub fn gram_hash(bytes: &[u8], pos: usize) -> usize {
    let v = u32::from_le_bytes([bytes[pos], bytes[pos + 1], bytes[pos + 2], bytes[pos + 3]]);
    (v.wrapping_mul(0x9E37_79B1) >> (32 - GRAM_BITS)) as usize
}

/// The language mask of hash value `h` in a detection table.
#[inline]
pub fn gram_mask(table: &[u8], h: usize) -> u32 {
    let at = h * GRAM_ENTRY_BYTES;
    u32::from_le_bytes([table[at], table[at + 1], table[at + 2], 0])
}

/// The detection table of the given dictionaries, one per language id (at most
/// `8 * GRAM_ENTRY_BYTES`; `lang.rs` bounds the language count at compile time).
// The library only reads the table the build script wrote; its tests recompute it.
#[cfg_attr(not(test), allow(dead_code))]
pub fn gram_table(dictionaries: &[Vec<u8>]) -> Vec<u8> {
    let mut table = vec![0u8; GRAM_TABLE_BYTES];
    for (lang, bytes) in dictionaries.iter().enumerate() {
        for pos in 0..bytes.len().saturating_sub(3) {
            let hash = gram_hash(bytes, pos);
            let at = hash * GRAM_ENTRY_BYTES;
            let mask = gram_mask(&table, hash) | 1 << lang;
            table[at..at + GRAM_ENTRY_BYTES]
                .copy_from_slice(&mask.to_le_bytes()[..GRAM_ENTRY_BYTES]);
        }
    }
    table
}
