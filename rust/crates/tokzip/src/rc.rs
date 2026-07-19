//! Binary range coder with adaptive 11-bit probabilities (LZMA-compatible
//! renormalization). All probability state lives in the caller's model arrays so
//! the same models can be primed once per dictionary and cloned per document.

pub const PROB_BITS: u32 = 11;
pub const PROB_INIT: u16 = 1 << (PROB_BITS - 1);
const ADAPT_SHIFT: u32 = 5;
const TOP: u32 = 1 << 24;

pub struct Encoder {
    low: u64,
    range: u32,
    cache: u8,
    cache_size: u64,
    out: Vec<u8>,
}

impl Encoder {
    pub fn new() -> Self {
        Self {
            low: 0,
            range: u32::MAX,
            cache: 0,
            cache_size: 1,
            out: Vec::new(),
        }
    }

    pub fn encode_bit(&mut self, prob: &mut u16, bit: u32) {
        let bound = (self.range >> PROB_BITS) * u32::from(*prob);
        if bit == 0 {
            self.range = bound;
            *prob += ((1 << PROB_BITS) - *prob) >> ADAPT_SHIFT;
        } else {
            self.low += u64::from(bound);
            self.range -= bound;
            *prob -= *prob >> ADAPT_SHIFT;
        }
        while self.range < TOP {
            self.range <<= 8;
            self.shift_low();
        }
    }

    pub fn encode_direct_bits(&mut self, value: u32, count: u32) {
        for i in (0..count).rev() {
            self.range >>= 1;
            if (value >> i) & 1 != 0 {
                self.low += u64::from(self.range);
            }
            while self.range < TOP {
                self.range <<= 8;
                self.shift_low();
            }
        }
    }

    /// MSB-first bit tree; `probs.len()` must be `1 << bits`.
    pub fn encode_tree(&mut self, probs: &mut [u16], bits: u32, symbol: u32) {
        let mut m = 1usize;
        for i in (0..bits).rev() {
            let bit = (symbol >> i) & 1;
            self.encode_bit(&mut probs[m], bit);
            m = (m << 1) | bit as usize;
        }
    }

    /// LSB-first bit tree over `probs[m - 1]` (m is the 1-based tree index), so a
    /// slice of exactly `(1 << bits) - 1` entries suffices.
    pub fn encode_tree_reverse(&mut self, probs: &mut [u16], bits: u32, symbol: u32) {
        let mut m = 1usize;
        for i in 0..bits {
            let bit = (symbol >> i) & 1;
            self.encode_bit(&mut probs[m - 1], bit);
            m = (m << 1) | bit as usize;
        }
    }

    pub fn finish(mut self) -> Vec<u8> {
        for _ in 0..5 {
            self.shift_low();
        }
        self.out
    }

    fn shift_low(&mut self) {
        if self.low < 0xFF00_0000 || self.low > 0xFFFF_FFFF {
            let carry = (self.low >> 32) as u8;
            self.out.push(self.cache.wrapping_add(carry));
            for _ in 1..self.cache_size {
                self.out.push(0xFFu8.wrapping_add(carry));
            }
            self.cache = (self.low >> 24) as u8;
            self.cache_size = 0;
        }
        self.cache_size += 1;
        self.low = (self.low << 8) & 0xFFFF_FFFF;
    }
}

pub struct Decoder<'a> {
    code: u32,
    range: u32,
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Decoder<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        let mut d = Self {
            code: 0,
            range: u32::MAX,
            buf,
            pos: 1, // first encoder byte is always 0 padding from the initial cache
        };
        for _ in 0..4 {
            d.code = (d.code << 8) | u32::from(d.next_byte());
        }
        d
    }

    pub fn decode_bit(&mut self, prob: &mut u16) -> u32 {
        let bound = (self.range >> PROB_BITS) * u32::from(*prob);
        let bit;
        if self.code < bound {
            self.range = bound;
            *prob += ((1 << PROB_BITS) - *prob) >> ADAPT_SHIFT;
            bit = 0;
        } else {
            self.code -= bound;
            self.range -= bound;
            *prob -= *prob >> ADAPT_SHIFT;
            bit = 1;
        }
        if self.range < TOP {
            self.range <<= 8;
            self.code = (self.code << 8) | u32::from(self.next_byte());
        }
        bit
    }

    pub fn decode_direct_bits(&mut self, count: u32) -> u32 {
        let mut result = 0u32;
        for _ in 0..count {
            self.range >>= 1;
            let bit = if self.code >= self.range {
                self.code -= self.range;
                1
            } else {
                0
            };
            result = (result << 1) | bit;
            if self.range < TOP {
                self.range <<= 8;
                self.code = (self.code << 8) | u32::from(self.next_byte());
            }
        }
        result
    }

    pub fn decode_tree(&mut self, probs: &mut [u16], bits: u32) -> u32 {
        let mut m = 1usize;
        for _ in 0..bits {
            m = (m << 1) | self.decode_bit(&mut probs[m]) as usize;
        }
        m as u32 - (1 << bits)
    }

    pub fn decode_tree_reverse(&mut self, probs: &mut [u16], bits: u32) -> u32 {
        let mut m = 1usize;
        let mut symbol = 0u32;
        for i in 0..bits {
            let bit = self.decode_bit(&mut probs[m - 1]);
            m = (m << 1) | bit as usize;
            symbol |= bit << i;
        }
        symbol
    }

    fn next_byte(&mut self) -> u8 {
        // Corrupt/truncated input decodes to garbage that the frame CRC rejects;
        // feeding zeros past the end keeps the hot path branch-light.
        let b = self.buf.get(self.pos).copied().unwrap_or(0);
        self.pos += 1;
        b
    }
}
