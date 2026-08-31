//! LZ77 over the document history plus the active language dictionary, entropy-coded by an
//! adaptive binary range coder (LZMA-style symbol layout).
//!
//! Every match is addressed by a distance `d >= 1` from the current document position `p`:
//! `d <= p` reaches back into the document, larger distances continue into the active
//! dictionary as if it preceded the document (`dict[D - (d - p)]`). Segments (see
//! `lang.rs`) swap the active dictionary and the language's adaptive models at token
//! boundaries; the document history, the coder state, and the rep distances are shared.
//!
//! Each language's models start *primed*: the dictionary is compressed against itself once
//! and the resulting probabilities seed every document, so short documents start from
//! statistics learned on the dictionary rather than from flat models.

use crate::rc::{Decoder, Encoder, PROB_INIT};
use crate::DecodeError;
use std::sync::OnceLock;

pub const MATCH_MAX: usize = 273;
const NUM_STATES: usize = 12;
const NUM_LEN_TO_POS: usize = 4;
const START_POS_MODEL: u32 = 4;
const END_POS_MODEL: u32 = 14;
const NUM_SPEC_POS: usize = 114;
const ALIGN_BITS: u32 = 4;

const HASH4_MAX_BITS: u32 = 17;
const HASH4_MIN_BITS: u32 = 10;
const HASH3_BITS: u32 = 15;
// A length-3 match further away than this costs more than three modeled literals.
const MAX_DIST_LEN3: usize = 1 << 14;
const SEARCH_DEPTH: usize = 32;
const DICT_SEARCH_DEPTH: usize = 32;
const NICE_LEN: usize = 64;
/// The optimal parse relaxes every match length up to this bound plus the full length.
const RELAX_LEN_CAP: usize = 16;
const EMPTY: i32 = -1;

// ---------------------------------------------------------------------------
// Probability models (one flat array; the layout is format identity)
// ---------------------------------------------------------------------------

/// Literal context class of the previous byte (see `lit_class`).
pub const LIT_CLASSES: usize = 32;
const IS_MATCH: usize = 0;
const IS_REP: usize = IS_MATCH + NUM_STATES;
const IS_REP_G0: usize = IS_REP + NUM_STATES;
const IS_REP_G1: usize = IS_REP_G0 + NUM_STATES;
const IS_REP_G2: usize = IS_REP_G1 + NUM_STATES;
const IS_REP0_LONG: usize = IS_REP_G2 + NUM_STATES;
const IS_DICT: usize = IS_REP0_LONG + NUM_STATES;
const LEN: usize = IS_DICT + NUM_STATES;
const LEN_MODEL_SIZE: usize = 2 + 8 + 8 + 256;
const REP_LEN: usize = LEN + LEN_MODEL_SIZE;
const DICT_LEN: usize = REP_LEN + LEN_MODEL_SIZE;
const DIST_MODEL_SIZE: usize = NUM_LEN_TO_POS * 64 + NUM_SPEC_POS + (1 << ALIGN_BITS);
/// History distances (`dist − 1`, LZMA layout: slot trees per length state, then the
/// specialized low-slot trees and the align tree).
const HIST_DIST: usize = DICT_LEN + LEN_MODEL_SIZE;
/// Dictionary offsets (absolute index into the active dictionary), same layout as
/// `HIST_DIST` but separately modeled: the same dictionary fragment costs the same wherever
/// it is referenced, so the trained priors capture the dictionary's value ordering.
const DICT_OFF: usize = HIST_DIST + DIST_MODEL_SIZE;
/// Plain literal trees, 256 nodes per literal class.
const LIT: usize = DICT_OFF + DIST_MODEL_SIZE;
/// Matched-literal trees (first literals after a match, predicted by the byte at rep0), 512
/// nodes shared by all classes: `match_bit << 8 | node`.
const LIT_MATCHED: usize = LIT + LIT_CLASSES * 256;
pub const MODEL_SIZE: usize = LIT_MATCHED + 512;

#[inline]
fn dist_group(base: usize) -> (usize, usize, usize) {
    (
        base,
        base + NUM_LEN_TO_POS * 64,
        base + NUM_LEN_TO_POS * 64 + NUM_SPEC_POS,
    )
}

/// Serialized model: the literal class table followed by the quantized probabilities.
pub const PRIORS_SIZE: usize = 256 + MODEL_SIZE;

#[derive(Clone)]
pub struct Models {
    /// Literal context class per previous-byte value (values < `LIT_CLASSES`).
    pub lit_class: [u8; 256],
    pub probs: Vec<u16>,
}

impl Models {
    pub fn new(lit_class: [u8; 256]) -> Self {
        Self {
            lit_class,
            probs: vec![PROB_INIT; MODEL_SIZE],
        }
    }

    /// Restores trained priors: 256 class bytes, then each 11-bit probability quantized to 8 bits.
    pub fn from_priors(priors: &[u8]) -> Self {
        assert_eq!(priors.len(), PRIORS_SIZE, "priors size mismatch");
        let mut lit_class = [0u8; 256];
        lit_class.copy_from_slice(&priors[..256]);
        assert!(
            lit_class.iter().all(|&c| (c as usize) < LIT_CLASSES),
            "invalid literal class"
        );
        Self {
            lit_class,
            probs: priors[256..]
                .iter()
                .map(|&q| (u16::from(q) << 3) | 4)
                .collect(),
        }
    }

    #[inline]
    fn lit_block(&self, prev: u8) -> usize {
        LIT + self.lit_class[prev as usize] as usize * 256
    }
}

fn encode_len(rc: &mut Encoder, probs: &mut [u16], base: usize, len: usize) {
    let l = (len - 2) as u32;
    if l < 8 {
        rc.encode_bit(probs, base, 0);
        rc.encode_tree(probs, base + 2, 3, l);
    } else if l < 16 {
        rc.encode_bit(probs, base, 1);
        rc.encode_bit(probs, base + 1, 0);
        rc.encode_tree(probs, base + 10, 3, l - 8);
    } else {
        rc.encode_bit(probs, base, 1);
        rc.encode_bit(probs, base + 1, 1);
        rc.encode_tree(probs, base + 18, 8, l - 16);
    }
}

fn decode_len(rc: &mut Decoder, probs: &mut [u16], base: usize) -> usize {
    let l = if rc.decode_bit(probs, base) == 0 {
        rc.decode_tree(probs, base + 2, 3)
    } else if rc.decode_bit(probs, base + 1) == 0 {
        8 + rc.decode_tree(probs, base + 10, 3)
    } else {
        16 + rc.decode_tree(probs, base + 18, 8)
    };
    l as usize + 2
}

fn price_len(probs: &[u16], base: usize, len: usize) -> u32 {
    let l = (len - 2) as u32;
    if l < 8 {
        price_bit(probs[base], 0) + price_tree(probs, base + 2, 3, l)
    } else if l < 16 {
        price_bit(probs[base], 1)
            + price_bit(probs[base + 1], 0)
            + price_tree(probs, base + 10, 3, l - 8)
    } else {
        price_bit(probs[base], 1)
            + price_bit(probs[base + 1], 1)
            + price_tree(probs, base + 18, 8, l - 16)
    }
}

// ---------------------------------------------------------------------------
// Bit prices (1/16-bit units) for the optimal parse
// ---------------------------------------------------------------------------

static PROB_PRICES: [u32; 128] = build_prob_prices();

/// `PROB_PRICES[p >> 4]` ≈ −16·log2(p / 2048), the cost of the 0-branch of probability `p`.
const fn build_prob_prices() -> [u32; 128] {
    // Integer log2 via a 16-entry fractional table (no float math in const context).
    const FRAC: [u32; 16] = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15];
    let mut t = [0u32; 128];
    let mut i = 0;
    while i < 128 {
        let p = i as u32 * 16 + 8; // out of 2048
                                   // −16·log2(p/2048) = 16·(11 − log2 p)
        let mut lg = 0u32;
        while (p >> (lg + 1)) != 0 {
            lg += 1;
        }
        let frac_idx = ((p << 4) >> lg) & 15;
        let log16 = lg * 16 + FRAC[frac_idx as usize];
        t[i] = 11 * 16 - log16;
        i += 1;
    }
    t
}

#[inline]
fn price_bit(prob: u16, bit: u32) -> u32 {
    if bit == 0 {
        PROB_PRICES[(prob >> 4) as usize]
    } else {
        PROB_PRICES[(((1u32 << 11) - u32::from(prob)) >> 4) as usize]
    }
}

fn price_tree(probs: &[u16], base: usize, bits: u32, symbol: u32) -> u32 {
    let mut price = 0;
    let mut m = 1usize;
    for i in (0..bits).rev() {
        let bit = (symbol >> i) & 1;
        price += price_bit(probs[base + m], bit);
        m = (m << 1) | bit as usize;
    }
    price
}

fn price_tree_reverse(probs: &[u16], base: usize, bits: u32, symbol: u32) -> u32 {
    let mut price = 0;
    let mut m = 1usize;
    for i in 0..bits {
        let bit = (symbol >> i) & 1;
        price += price_bit(probs[base + m - 1], bit);
        m = (m << 1) | bit as usize;
    }
    price
}

impl Models {
    /// Price of the literal at `pos` (excluding the `is_match` decision bit).
    fn price_literal(&self, win: &Window, pos: usize, state: usize, reps: &[u32; 4]) -> u32 {
        let block = self.lit_block(win.prev_byte(pos));
        let probs = &self.probs;
        let symbol = u32::from(win.doc[pos]);
        let mut price = 0;
        let mut m = 1usize;
        let mut i = 8u32;
        if state >= 7 {
            let match_byte = u32::from(win.byte_at(pos, reps[0] as usize + 1));
            while i > 0 {
                i -= 1;
                let mb = (match_byte >> i) & 1;
                let bit = (symbol >> i) & 1;
                price += price_bit(probs[LIT_MATCHED + ((mb as usize) << 8) + m], bit);
                m = (m << 1) | bit as usize;
                if mb != bit {
                    break;
                }
            }
        }
        while i > 0 {
            i -= 1;
            let bit = (symbol >> i) & 1;
            price += price_bit(probs[block + m], bit);
            m = (m << 1) | bit as usize;
        }
        price
    }

    /// Price of a distance-like value (history `dist − 1` or dictionary offset) in `group`.
    fn price_dist(&self, group: usize, len_state: usize, value: u32) -> u32 {
        let (slot_base, spec_base, align_base) = dist_group(group);
        let slot = dist_slot(value);
        let mut price = price_tree(&self.probs, slot_base + len_state * 64, 6, slot);
        if slot >= START_POS_MODEL {
            let footer = (slot >> 1) - 1;
            let base = (2 | (slot & 1)) << footer;
            let reduced = value - base;
            if slot < END_POS_MODEL {
                price += price_tree_reverse(
                    &self.probs,
                    spec_base + (base - slot) as usize,
                    footer,
                    reduced,
                );
            } else {
                price += (footer - ALIGN_BITS) << 4;
                price += price_tree_reverse(&self.probs, align_base, ALIGN_BITS, reduced & 0xF);
            }
        }
        price
    }

    fn price_rep_choice(&self, state: usize, idx: usize) -> u32 {
        let p = &self.probs;
        if idx == 0 {
            price_bit(p[IS_REP_G0 + state], 0) + price_bit(p[IS_REP0_LONG + state], 1)
        } else {
            let mut price = price_bit(p[IS_REP_G0 + state], 1);
            if idx == 1 {
                price += price_bit(p[IS_REP_G1 + state], 0);
            } else {
                price += price_bit(p[IS_REP_G1 + state], 1)
                    + price_bit(p[IS_REP_G2 + state], (idx == 3) as u32);
            }
            price
        }
    }
}

/// Distance prices frozen for one parse chunk: full prices for values below `FULL_DIST`
/// (slots with specialized trees) and slot + align prices beyond, per length state.
struct DistPrices {
    full: Vec<[u32; FULL_DIST]>, // [len_state][value]
    slot: [[u32; 64]; NUM_LEN_TO_POS],
    align: [u32; 1 << ALIGN_BITS],
}

const FULL_DIST: usize = 1 << (END_POS_MODEL as usize >> 1);

impl DistPrices {
    fn new(models: &Models, group: usize) -> Self {
        let (slot_base, _, align_base) = dist_group(group);
        let mut full = vec![[0u32; FULL_DIST]; NUM_LEN_TO_POS];
        let mut slot = [[0u32; 64]; NUM_LEN_TO_POS];
        for ls in 0..NUM_LEN_TO_POS {
            for (value, price) in full[ls].iter_mut().enumerate() {
                *price = models.price_dist(group, ls, value as u32);
            }
            for (sl, price) in slot[ls].iter_mut().enumerate() {
                *price = price_tree(&models.probs, slot_base + ls * 64, 6, sl as u32);
            }
        }
        let mut align = [0u32; 1 << ALIGN_BITS];
        for (v, price) in align.iter_mut().enumerate() {
            *price = price_tree_reverse(&models.probs, align_base, ALIGN_BITS, v as u32);
        }
        Self { full, slot, align }
    }

    #[inline]
    fn price(&self, len_state: usize, value: u32) -> u32 {
        if (value as usize) < FULL_DIST {
            return self.full[len_state][value as usize];
        }
        let slot = dist_slot(value);
        let footer = (slot >> 1) - 1;
        self.slot[len_state][slot as usize]
            + ((footer - ALIGN_BITS) << 4)
            + self.align[(value & 0xF) as usize]
    }
}

/// Coder state carried across symbols: the LZMA 12-state machine plus the four most recent
/// match distances (stored as distance − 1).
#[derive(Clone, Copy)]
pub struct CoderState {
    state: usize,
    reps: [u32; 4],
}

impl CoderState {
    pub fn new() -> Self {
        Self {
            state: 0,
            reps: [0; 4],
        }
    }

    fn prev_was_match(&self) -> bool {
        self.state >= 7
    }
}

fn state_after_literal(s: usize) -> usize {
    match s {
        0..=3 => 0,
        4..=9 => s - 3,
        _ => s - 6,
    }
}

fn state_after_match(s: usize) -> usize {
    if s < 7 {
        7
    } else {
        10
    }
}

fn state_after_rep(s: usize) -> usize {
    if s < 7 {
        8
    } else {
        11
    }
}

fn dist_slot(dist_m1: u32) -> u32 {
    if dist_m1 < 4 {
        return dist_m1;
    }
    let n = 31 - dist_m1.leading_zeros();
    (n << 1) | ((dist_m1 >> (n - 1)) & 1)
}

// ---------------------------------------------------------------------------
// Window & match finder
// ---------------------------------------------------------------------------

/// The document plus the active dictionary that virtually precedes it.
#[derive(Clone, Copy)]
pub struct Window<'a> {
    pub doc: &'a [u8],
    pub dict: &'a [u8],
}

impl Window<'_> {
    /// Byte `dist` positions before `pos` (`dist >= 1`, reaching into the dictionary past
    /// the document start). Callers guarantee `dist <= pos + dict.len()`.
    #[inline]
    fn byte_at(&self, pos: usize, dist: usize) -> u8 {
        if dist <= pos {
            self.doc[pos - dist]
        } else {
            self.dict[self.dict.len() - (dist - pos)]
        }
    }

    #[inline]
    fn prev_byte(&self, pos: usize) -> u8 {
        if pos > 0 {
            self.doc[pos - 1]
        } else if let Some(&b) = self.dict.last() {
            b
        } else {
            0
        }
    }

    /// Length of the match at `pos` against the source `dist` back, capped at `max`.
    /// `max` never exceeds the document end.
    #[inline]
    fn common_len(&self, pos: usize, dist: usize, max: usize) -> usize {
        let target = &self.doc[pos..pos + max];
        if dist <= pos {
            // Overlapping sources are fine: the encoder holds the whole document, and the
            // bytes an overlapping copy would produce are exactly the document's bytes.
            return common_prefix(&self.doc[pos - dist..], target);
        }
        // The source starts inside the dictionary and may run on into the document.
        let start = self.dict.len() - (dist - pos);
        let l = common_prefix(&self.dict[start..], target);
        if l < max && start + l == self.dict.len() {
            return l + common_prefix(self.doc, &target[l..]);
        }
        l
    }
}

/// Length of the common prefix of `a` and `b`, eight bytes at a time.
#[inline]
fn common_prefix(a: &[u8], b: &[u8]) -> usize {
    let n = a.len().min(b.len());
    let mut i = 0;
    while i + 8 <= n {
        let x = u64::from_le_bytes(a[i..i + 8].try_into().unwrap())
            ^ u64::from_le_bytes(b[i..i + 8].try_into().unwrap());
        if x != 0 {
            return i + (x.trailing_zeros() / 8) as usize;
        }
        i += 8;
    }
    while i < n && a[i] == b[i] {
        i += 1;
    }
    i
}

#[inline]
fn hash4(bytes: &[u8], pos: usize, bits: u32) -> usize {
    let v = u32::from_le_bytes([bytes[pos], bytes[pos + 1], bytes[pos + 2], bytes[pos + 3]]);
    (v.wrapping_mul(0x9E37_79B1) >> (32 - bits)) as usize
}

#[inline]
fn hash3(bytes: &[u8], pos: usize) -> usize {
    let v =
        u32::from(bytes[pos]) | u32::from(bytes[pos + 1]) << 8 | u32::from(bytes[pos + 2]) << 16;
    (v.wrapping_mul(0x9E37_79B1) >> (32 - HASH3_BITS)) as usize
}

/// Hash chains over a byte sequence; built once per dictionary, incrementally per document.
pub struct Chains {
    bits: u32,
    head4: Vec<i32>,
    head3: Vec<i32>,
    prev4: Vec<i32>,
}

impl Chains {
    fn new(len: usize) -> Self {
        let mut bits = HASH4_MIN_BITS;
        while bits < HASH4_MAX_BITS && (1usize << bits) < len * 2 {
            bits += 1;
        }
        Self {
            bits,
            head4: vec![EMPTY; 1 << bits],
            head3: vec![EMPTY; 1 << HASH3_BITS],
            prev4: vec![EMPTY; len],
        }
    }

    #[inline]
    fn insert(&mut self, bytes: &[u8], pos: usize) {
        if pos + 3 <= bytes.len() {
            self.head3[hash3(bytes, pos)] = pos as i32;
        }
        if pos + 4 <= bytes.len() {
            let h = hash4(bytes, pos, self.bits);
            self.prev4[pos] = self.head4[h];
            self.head4[h] = pos as i32;
        }
    }
}

/// A language dictionary with its primed model state and, for the encoder, its match-finder
/// chains. Built once per language; the chains are built on first encode only, so a
/// decompress-only process never pays for them.
pub struct Primed {
    pub bytes: Vec<u8>,
    chains: OnceLock<Chains>,
    pub models: Models,
}

impl Primed {
    /// Builds the dictionary's chains and its initial models: trained `priors` when given,
    /// otherwise the state left by compressing the dictionary against itself (the trainer's
    /// starting point).
    pub fn new(bytes: Vec<u8>, priors: Option<&[u8]>, lit_class: [u8; 256]) -> Self {
        if let Some(priors) = priors {
            return Self {
                bytes,
                chains: OnceLock::new(),
                models: Models::from_priors(priors),
            };
        }
        let mut models = Models::new(lit_class);
        let mut state = CoderState::new();
        let empty = Primed {
            bytes: Vec::new(),
            chains: OnceLock::from(Chains::new(0)),
            models: Models::new(lit_class),
        };
        let mut mf = MatchFinder {
            win: Window {
                doc: &bytes,
                dict: &[],
            },
            chains: Chains::new(bytes.len()),
            dict: DictRef {
                bytes: &empty.bytes,
                chains: empty.chains.get().unwrap(),
            },
        };
        // Output is discarded: only the adapted models and the chains matter.
        let mut rc = Encoder::new();
        let mut inserted = 0usize;
        run_encode_optimal(
            &mut rc,
            &mut models,
            &mut state,
            &mut mf,
            0,
            bytes.len(),
            &mut inserted,
        );
        let chains = OnceLock::from(mf.chains);
        Self {
            bytes,
            chains,
            models,
        }
    }

    fn chains(&self) -> &Chains {
        self.chains.get_or_init(|| {
            let mut chains = Chains::new(self.bytes.len());
            for pos in 0..self.bytes.len() {
                chains.insert(&self.bytes, pos);
            }
            chains
        })
    }
}

/// The active dictionary as the match finder sees it: its bytes and its chains.
#[derive(Clone, Copy)]
struct DictRef<'a> {
    bytes: &'a [u8],
    chains: &'a Chains,
}

struct MatchFinder<'a> {
    win: Window<'a>,
    chains: Chains,
    dict: DictRef<'a>,
}

impl MatchFinder<'_> {
    #[inline]
    fn insert(&mut self, pos: usize) {
        self.chains.insert(self.win.doc, pos);
    }

    /// Useful matches at `pos` as `(len, dist)` with strictly increasing lengths; each carries
    /// the nearest distance found for that length.
    fn find_pairs(&self, pos: usize, max_len: usize, pairs: &mut Vec<(u32, u32)>) {
        pairs.clear();
        let doc = self.win.doc;
        let dlen = self.dict.bytes.len();
        let mut best_len = 0usize;
        if max_len >= 3 && pos + 3 <= doc.len() {
            let cand = self.chains.head3[hash3(doc, pos)];
            let mut dist = 0usize;
            if cand >= 0 && pos - cand as usize <= MAX_DIST_LEN3 {
                dist = pos - cand as usize;
            } else if dlen >= 3 {
                let cand = self.dict.chains.head3[hash3(doc, pos)];
                if cand >= 0 {
                    dist = pos + dlen - cand as usize;
                }
            }
            if dist > 0 {
                let l = self.win.common_len(pos, dist, max_len);
                if l >= 3 {
                    best_len = l;
                    pairs.push((l as u32, dist as u32));
                }
            }
        }
        if max_len < 4 || pos + 4 > doc.len() {
            return;
        }
        let mut budget = SEARCH_DEPTH;
        let mut cand = self.chains.head4[hash4(doc, pos, self.chains.bits)];
        while cand >= 0 && budget > 0 {
            budget -= 1;
            let c = cand as usize;
            let dist = pos - c;
            if best_len == 0
                || (pos + best_len < doc.len() && doc[c + best_len] == doc[pos + best_len])
            {
                let l = self.win.common_len(pos, dist, max_len);
                if l > best_len && l >= 4 {
                    best_len = l;
                    pairs.push((l as u32, dist as u32));
                    if l >= NICE_LEN || l == max_len {
                        return;
                    }
                }
            }
            cand = self.chains.prev4[c];
        }
        if dlen < 4 {
            return;
        }
        let dict = &self.dict.bytes;
        let mut cand = self.dict.chains.head4[hash4(doc, pos, self.dict.chains.bits)];
        let mut budget = DICT_SEARCH_DEPTH;
        while cand >= 0 && budget > 0 {
            budget -= 1;
            let c = cand as usize;
            let dist = pos + dlen - c;
            let probe_ok = best_len == 0
                || (pos + best_len < doc.len()
                    && (c + best_len >= dlen || dict[c + best_len] == doc[pos + best_len]));
            if probe_ok {
                let l = self.win.common_len(pos, dist, max_len);
                if l > best_len && l >= 4 {
                    best_len = l;
                    pairs.push((l as u32, dist as u32));
                    if l >= NICE_LEN || l == max_len {
                        return;
                    }
                }
            }
            cand = self.dict.chains.prev4[c];
        }
    }

    /// Match length at `pos` against the source `dist` back (0 when the distance underflows).
    #[inline]
    fn len_at(&self, pos: usize, dist: usize, max_len: usize) -> usize {
        if dist == 0 || dist > pos + self.dict.bytes.len() {
            return 0;
        }
        self.win.common_len(pos, dist, max_len)
    }
}

// ---------------------------------------------------------------------------
// Segments
// ---------------------------------------------------------------------------

/// A run of the document coded with one language's dictionary and models.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Segment {
    pub end: usize,
    pub lang: u8,
}

/// Per-document language state: each language's models are cloned from the primed state on
/// first use and keep adapting across that language's segments.
struct LangModels<'a> {
    lookup: &'a dyn Fn(u8) -> &'static Primed,
    models: Vec<Option<Models>>,
}

impl<'a> LangModels<'a> {
    fn new(lookup: &'a dyn Fn(u8) -> &'static Primed) -> Self {
        Self {
            lookup,
            models: vec![None; 256],
        }
    }

    fn take(&mut self, lang: u8) -> Models {
        self.models[lang as usize]
            .take()
            .unwrap_or_else(|| (self.lookup)(lang).models.clone())
    }

    fn put(&mut self, lang: u8, models: Models) {
        self.models[lang as usize] = Some(models);
    }
}

// ---------------------------------------------------------------------------
// Encoder
// ---------------------------------------------------------------------------

pub fn encode_doc(
    lookup: &dyn Fn(u8) -> &'static Primed,
    doc: &[u8],
    segments: &[Segment],
) -> Vec<u8> {
    encode_doc_with_stats(lookup, doc, segments, None).0
}

/// [`encode_doc`] that can also count the bits coded at every model node (trainer support).
pub fn encode_doc_with_stats(
    lookup: &dyn Fn(u8) -> &'static Primed,
    doc: &[u8],
    segments: &[Segment],
    stats: Option<Vec<u32>>,
) -> (Vec<u8>, Option<Vec<u32>>) {
    let mut rc = Encoder::new();
    rc.stats = stats;
    let mut cs = CoderState::new();
    let mut lang_models = LangModels::new(lookup);
    let mut chains = Chains::new(doc.len());
    let mut inserted = 0usize;
    let mut start = 0usize;
    for seg in segments {
        let dict = lookup(seg.lang);
        clamp_reps(&mut cs, start, dict.bytes.len());
        let mut mf = MatchFinder {
            win: Window {
                doc,
                dict: &dict.bytes,
            },
            chains,
            dict: DictRef {
                bytes: &dict.bytes,
                chains: dict.chains(),
            },
        };
        let mut models = lang_models.take(seg.lang);
        run_encode_optimal(
            &mut rc,
            &mut models,
            &mut cs,
            &mut mf,
            start,
            seg.end,
            &mut inserted,
        );
        // Positions skipped by the parse (inside matches) still join the chains.
        while inserted < seg.end {
            mf.insert(inserted);
            inserted += 1;
        }
        lang_models.put(seg.lang, models);
        chains = mf.chains;
        start = seg.end;
    }
    let stats = rc.stats.take();
    (rc.finish(), stats)
}

/// Normative at every segment start: rep distances the new segment's window cannot reach
/// (the previous segment's dictionary was longer) reset to distance 1, so a matched literal
/// or rep match after the boundary never addresses a byte that does not exist.
fn clamp_reps(cs: &mut CoderState, pos: usize, dlen: usize) {
    for rep in &mut cs.reps {
        // Width-independent: `*rep as usize + 1` would wrap at u32::MAX on 32-bit targets.
        if u64::from(*rep) >= (pos + dlen) as u64 {
            *rep = 0;
        }
    }
}

#[derive(Clone, Copy)]
enum Step {
    Literal,
    Match { len: u32, dist_m1: u32 },
    Rep { idx: u8, len: u32 },
}

#[derive(Clone, Copy)]
struct Node {
    price: u32,
    prev: u32,
    step: Step,
    state: u8,
    reps: [u32; 4],
}

const INF: u32 = u32::MAX;

/// Price-based optimal parse of `[start, end)` — the only parse: a forward shortest path over
/// positions using exact model bit prices, refreshed every `CHUNK` positions (each chunk's
/// chosen path is replayed through the adaptive coder before the next chunk is parsed).
fn run_encode_optimal(
    rc: &mut Encoder,
    models: &mut Models,
    cs: &mut CoderState,
    mf: &mut MatchFinder,
    start: usize,
    end: usize,
    inserted: &mut usize,
) {
    const CHUNK: usize = 4096;
    let mut pos = start;
    let mut pairs: Vec<(u32, u32)> = Vec::new();
    let mut steps: Vec<Step> = Vec::new();
    let mut nodes: Vec<Node> = Vec::new();
    while pos < end {
        let end_target = (pos + CHUNK).min(end);
        let max_reach = (end_target + MATCH_MAX).min(end);
        let n = max_reach - pos;
        nodes.clear();
        nodes.resize(
            n + 1,
            Node {
                price: INF,
                prev: 0,
                step: Step::Literal,
                state: 0,
                reps: [0; 4],
            },
        );
        nodes[0].price = 0;
        nodes[0].state = cs.state as u8;
        nodes[0].reps = cs.reps;

        let mut match_len_price = [0u32; MATCH_MAX + 1];
        let mut rep_len_price = [0u32; MATCH_MAX + 1];
        let mut dict_len_price = [0u32; MATCH_MAX + 1];
        for len in 2..=MATCH_MAX {
            match_len_price[len] = price_len(&models.probs, LEN, len);
            rep_len_price[len] = price_len(&models.probs, REP_LEN, len);
            dict_len_price[len] = price_len(&models.probs, DICT_LEN, len);
        }
        let dlen = mf.dict.bytes.len();
        let hist_prices = DistPrices::new(models, HIST_DIST);
        let dict_prices = DistPrices::new(models, DICT_OFF);

        for i in 0..(end_target - pos) {
            if nodes[i].price == INF {
                continue;
            }
            let gpos = pos + i;
            while *inserted < gpos {
                mf.insert(*inserted);
                *inserted += 1;
            }
            let state = nodes[i].state as usize;
            let reps = nodes[i].reps;
            let base = nodes[i].price;
            let max_len = MATCH_MAX.min(end - gpos);

            let lit_price = base
                + price_bit(models.probs[IS_MATCH + state], 0)
                + models.price_literal(&mf.win, gpos, state, &reps);
            relax(
                &mut nodes,
                i,
                1,
                lit_price,
                Step::Literal,
                state_after_literal(state) as u8,
                reps,
            );

            let match_bit = price_bit(models.probs[IS_MATCH + state], 1);
            let rep_bit = match_bit + price_bit(models.probs[IS_REP + state], 1);

            let mut long_rep = false;
            for idx in 0..4usize {
                let l = mf.len_at(gpos, reps[idx] as usize + 1, max_len);
                long_rep |= l >= NICE_LEN;
                if idx == 0 && l >= 1 {
                    let price = base
                        + rep_bit
                        + price_bit(models.probs[IS_REP_G0 + state], 0)
                        + price_bit(models.probs[IS_REP0_LONG + state], 0);
                    let st = if state < 7 { 9 } else { 11 };
                    relax(
                        &mut nodes,
                        i,
                        1,
                        price,
                        Step::Rep { idx: 0, len: 1 },
                        st,
                        reps,
                    );
                }
                if l < 2 {
                    continue;
                }
                let choice = base + rep_bit + models.price_rep_choice(state, idx);
                let mut new_reps = reps;
                let rep = new_reps[idx];
                new_reps.copy_within(0..idx, 1);
                new_reps[0] = rep;
                let st = state_after_rep(state) as u8;
                // Relax the short lengths (where a shorter match can enable a better
                // continuation) and the full length; intermediate lengths rarely win.
                let from = if l >= NICE_LEN { l } else { 2 };
                for len in (from..=l.min(RELAX_LEN_CAP))
                    .chain((l > RELAX_LEN_CAP.max(from - 1)).then_some(l))
                {
                    relax(
                        &mut nodes,
                        i,
                        len,
                        choice + rep_len_price[len],
                        Step::Rep {
                            idx: idx as u8,
                            len: len as u32,
                        },
                        st,
                        new_reps,
                    );
                }
            }

            // A rep match already at nice length is taken as is (no explicit match can beat
            // a rep of the same length), so the chain walk is skipped.
            if long_rep {
                continue;
            }
            mf.find_pairs(gpos, max_len, &mut pairs);
            let mat_bit = match_bit + price_bit(models.probs[IS_REP + state], 0);
            let hist_bit = mat_bit + price_bit(models.probs[IS_DICT + state], 0);
            let dict_bit = mat_bit + price_bit(models.probs[IS_DICT + state], 1);
            let st = state_after_match(state) as u8;
            let mut len_from = 2usize;
            for &(plen, pdist) in &pairs {
                let plen = plen as usize;
                let dist_m1 = pdist - 1;
                let new_reps = [dist_m1, reps[0], reps[1], reps[2]];
                let is_dict = pdist as usize > gpos;
                let (prices, value, kind_bit, len_price): (
                    &DistPrices,
                    u32,
                    u32,
                    &[u32; MATCH_MAX + 1],
                ) = if is_dict {
                    (
                        &dict_prices,
                        (dlen - (pdist as usize - gpos)) as u32,
                        dict_bit,
                        &dict_len_price,
                    )
                } else {
                    (&hist_prices, dist_m1, hist_bit, &match_len_price)
                };
                let mut dist_price = [0u32; NUM_LEN_TO_POS];
                for (ls, p) in dist_price.iter_mut().enumerate() {
                    *p = prices.price(ls, value);
                }
                let from = if plen >= NICE_LEN { plen } else { len_from };
                for len in (from..=plen.min(RELAX_LEN_CAP))
                    .chain((plen > RELAX_LEN_CAP.max(from - 1)).then_some(plen))
                {
                    let len_state = (len - 2).min(NUM_LEN_TO_POS - 1);
                    relax(
                        &mut nodes,
                        i,
                        len,
                        base + kind_bit + len_price[len] + dist_price[len_state],
                        Step::Match {
                            len: len as u32,
                            dist_m1,
                        },
                        st,
                        new_reps,
                    );
                }
                len_from = plen + 1;
            }
        }

        let mut best = end_target - pos;
        for j in (end_target - pos)..=n {
            if nodes[j].price < nodes[best].price {
                best = j;
            }
        }
        steps.clear();
        let mut cur = best;
        while cur != 0 {
            steps.push(nodes[cur].step);
            cur = nodes[cur].prev as usize;
        }
        for step in steps.iter().rev() {
            match *step {
                Step::Literal => {
                    emit_literal(rc, models, cs, &mf.win, pos);
                    pos += 1;
                }
                Step::Match { len, dist_m1 } => {
                    emit_match(rc, models, cs, pos, dlen, len as usize, dist_m1);
                    pos += len as usize;
                }
                Step::Rep { idx, len } => {
                    emit_rep(rc, models, cs, idx as usize, len as usize);
                    pos += len as usize;
                }
            }
        }
    }
}

#[inline]
fn relax(
    nodes: &mut [Node],
    from: usize,
    adv: usize,
    price: u32,
    step: Step,
    state: u8,
    reps: [u32; 4],
) {
    let to = from + adv;
    if to < nodes.len() && price < nodes[to].price {
        nodes[to] = Node {
            price,
            prev: from as u32,
            step,
            state,
            reps,
        };
    }
}

fn emit_literal(
    rc: &mut Encoder,
    models: &mut Models,
    cs: &mut CoderState,
    win: &Window,
    pos: usize,
) {
    rc.encode_bit(&mut models.probs, IS_MATCH + cs.state, 0);
    let block = models.lit_block(win.prev_byte(pos));
    let probs = &mut models.probs;
    let symbol = u32::from(win.doc[pos]);
    let mut m = 1usize;
    let mut i = 8u32;
    if cs.prev_was_match() {
        let match_byte = u32::from(win.byte_at(pos, cs.reps[0] as usize + 1));
        while i > 0 {
            i -= 1;
            let mb = (match_byte >> i) & 1;
            let bit = (symbol >> i) & 1;
            rc.encode_bit(probs, LIT_MATCHED + ((mb as usize) << 8) + m, bit);
            m = (m << 1) | bit as usize;
            if mb != bit {
                break;
            }
        }
    }
    while i > 0 {
        i -= 1;
        let bit = (symbol >> i) & 1;
        rc.encode_bit(probs, block + m, bit);
        m = (m << 1) | bit as usize;
    }
    cs.state = state_after_literal(cs.state);
}

/// Emits an explicit match: a history match (`dist <= pos`) codes `dist − 1`; a match
/// reaching into the dictionary codes the absolute dictionary offset instead.
fn emit_match(
    rc: &mut Encoder,
    models: &mut Models,
    cs: &mut CoderState,
    pos: usize,
    dlen: usize,
    len: usize,
    dist_m1: u32,
) {
    let probs = &mut models.probs;
    rc.encode_bit(probs, IS_MATCH + cs.state, 1);
    rc.encode_bit(probs, IS_REP + cs.state, 0);
    let dist = dist_m1 as usize + 1;
    let len_state = (len - 2).min(NUM_LEN_TO_POS - 1);
    if dist > pos {
        rc.encode_bit(probs, IS_DICT + cs.state, 1);
        encode_len(rc, probs, DICT_LEN, len);
        encode_dist(rc, probs, DICT_OFF, len_state, (dlen - (dist - pos)) as u32);
    } else {
        rc.encode_bit(probs, IS_DICT + cs.state, 0);
        encode_len(rc, probs, LEN, len);
        encode_dist(rc, probs, HIST_DIST, len_state, dist_m1);
    }
    cs.reps = [dist_m1, cs.reps[0], cs.reps[1], cs.reps[2]];
    cs.state = state_after_match(cs.state);
}

fn encode_dist(rc: &mut Encoder, probs: &mut [u16], group: usize, len_state: usize, value: u32) {
    let (slot_base, spec_base, align_base) = dist_group(group);
    let slot = dist_slot(value);
    rc.encode_tree(probs, slot_base + len_state * 64, 6, slot);
    if slot >= START_POS_MODEL {
        let footer = (slot >> 1) - 1;
        let base = (2 | (slot & 1)) << footer;
        let reduced = value - base;
        if slot < END_POS_MODEL {
            rc.encode_tree_reverse(probs, spec_base + (base - slot) as usize, footer, reduced);
        } else {
            rc.encode_direct_bits(reduced >> ALIGN_BITS, footer - ALIGN_BITS);
            rc.encode_tree_reverse(probs, align_base, ALIGN_BITS, reduced & 0xF);
        }
    }
}

fn decode_dist(rc: &mut Decoder, probs: &mut [u16], group: usize, len_state: usize) -> u32 {
    let (slot_base, spec_base, align_base) = dist_group(group);
    let slot = rc.decode_tree(probs, slot_base + len_state * 64, 6);
    if slot < START_POS_MODEL {
        return slot;
    }
    let footer = (slot >> 1) - 1;
    let base = (2 | (slot & 1)) << footer;
    let reduced = if slot < END_POS_MODEL {
        rc.decode_tree_reverse(probs, spec_base + (base - slot) as usize, footer)
    } else {
        (rc.decode_direct_bits(footer - ALIGN_BITS) << ALIGN_BITS)
            | rc.decode_tree_reverse(probs, align_base, ALIGN_BITS)
    };
    base + reduced
}

fn emit_rep(rc: &mut Encoder, models: &mut Models, cs: &mut CoderState, idx: usize, len: usize) {
    let probs = &mut models.probs;
    rc.encode_bit(probs, IS_MATCH + cs.state, 1);
    rc.encode_bit(probs, IS_REP + cs.state, 1);
    if idx == 0 {
        rc.encode_bit(probs, IS_REP_G0 + cs.state, 0);
        if len == 1 {
            rc.encode_bit(probs, IS_REP0_LONG + cs.state, 0);
            cs.state = if cs.state < 7 { 9 } else { 11 };
            return;
        }
        rc.encode_bit(probs, IS_REP0_LONG + cs.state, 1);
    } else {
        rc.encode_bit(probs, IS_REP_G0 + cs.state, 1);
        if idx == 1 {
            rc.encode_bit(probs, IS_REP_G1 + cs.state, 0);
        } else {
            rc.encode_bit(probs, IS_REP_G1 + cs.state, 1);
            rc.encode_bit(probs, IS_REP_G2 + cs.state, (idx == 3) as u32);
        }
        let rep = cs.reps[idx];
        cs.reps.copy_within(0..idx, 1);
        cs.reps[0] = rep;
    }
    encode_len(rc, probs, REP_LEN, len);
    cs.state = state_after_rep(cs.state);
}

// ---------------------------------------------------------------------------
// Decoder
// ---------------------------------------------------------------------------

pub fn decode_doc(
    lookup: &dyn Fn(u8) -> &'static Primed,
    body: &[u8],
    out_len: usize,
    segments: &[Segment],
) -> Result<Vec<u8>, DecodeError> {
    // The caller bounds `out_len` by the body size; the reservation is bounded tighter still so
    // a forged header cannot cost a large allocation before the CRC rejects it.
    let mut out: Vec<u8> = Vec::with_capacity(out_len.min(body.len() * 64 + 4096));
    let mut rc = Decoder::new(body);
    let mut cs = CoderState::new();
    let mut lang_models = LangModels::new(lookup);
    for seg in segments {
        if seg.end > out_len || seg.end < out.len() {
            return Err(DecodeError::Corrupt);
        }
        let dict: &[u8] = &lookup(seg.lang).bytes;
        let dlen = dict.len();
        clamp_reps(&mut cs, out.len(), dlen);
        let mut models = lang_models.take(seg.lang);
        let lit_class = models.lit_class;
        while out.len() < seg.end {
            if rc.overran() {
                return Err(DecodeError::Corrupt);
            }
            let pos = out.len();
            let probs = &mut models.probs;
            if rc.decode_bit(probs, IS_MATCH + cs.state) == 0 {
                let prev = if pos > 0 {
                    out[pos - 1]
                } else {
                    dict.last().copied().unwrap_or(0)
                };
                let block = LIT + lit_class[prev as usize] as usize * 256;
                let mut m = 1usize;
                if cs.prev_was_match() {
                    // u64 compare: `as usize + 1` would wrap at u32::MAX on 32-bit targets.
                    if u64::from(cs.reps[0]) >= (pos + dlen) as u64 {
                        return Err(DecodeError::Corrupt);
                    }
                    let dist = cs.reps[0] as usize + 1;
                    let match_byte = u32::from(source_byte(&out, dict, pos, dist));
                    let mut i = 8u32;
                    while i > 0 {
                        i -= 1;
                        let mb = (match_byte >> i) & 1;
                        let bit = rc.decode_bit(probs, LIT_MATCHED + ((mb as usize) << 8) + m);
                        m = (m << 1) | bit as usize;
                        if mb != bit {
                            break;
                        }
                    }
                }
                while m < 0x100 {
                    m = (m << 1) | rc.decode_bit(probs, block + m) as usize;
                }
                out.push((m & 0xFF) as u8);
                cs.state = state_after_literal(cs.state);
                continue;
            }
            let len;
            if rc.decode_bit(probs, IS_REP + cs.state) == 0 {
                let dist_m1 = if rc.decode_bit(probs, IS_DICT + cs.state) == 1 {
                    len = decode_len(&mut rc, probs, DICT_LEN);
                    let len_state = (len - 2).min(NUM_LEN_TO_POS - 1);
                    let off = decode_dist(&mut rc, probs, DICT_OFF, len_state) as usize;
                    if off >= dlen {
                        return Err(DecodeError::Corrupt);
                    }
                    (pos + dlen - off - 1) as u32
                } else {
                    len = decode_len(&mut rc, probs, LEN);
                    let len_state = (len - 2).min(NUM_LEN_TO_POS - 1);
                    decode_dist(&mut rc, probs, HIST_DIST, len_state)
                };
                cs.reps = [dist_m1, cs.reps[0], cs.reps[1], cs.reps[2]];
                cs.state = state_after_match(cs.state);
            } else {
                if rc.decode_bit(probs, IS_REP_G0 + cs.state) == 0 {
                    if rc.decode_bit(probs, IS_REP0_LONG + cs.state) == 0 {
                        // u64 compare: `as usize + 1` would wrap at u32::MAX on 32-bit targets.
                        if u64::from(cs.reps[0]) >= (pos + dlen) as u64 {
                            return Err(DecodeError::Corrupt);
                        }
                        let dist = cs.reps[0] as usize + 1;
                        let b = source_byte(&out, dict, pos, dist);
                        out.push(b);
                        cs.state = if cs.state < 7 { 9 } else { 11 };
                        continue;
                    }
                } else {
                    let idx = if rc.decode_bit(probs, IS_REP_G1 + cs.state) == 0 {
                        1
                    } else if rc.decode_bit(probs, IS_REP_G2 + cs.state) == 0 {
                        2
                    } else {
                        3
                    };
                    let rep = cs.reps[idx];
                    cs.reps.copy_within(0..idx, 1);
                    cs.reps[0] = rep;
                }
                len = decode_len(&mut rc, probs, REP_LEN);
                cs.state = state_after_rep(cs.state);
            }
            // u64 compare: `as usize + 1` would wrap at u32::MAX on 32-bit targets.
            if u64::from(cs.reps[0]) >= (pos + dlen) as u64 || pos + len > seg.end {
                return Err(DecodeError::Corrupt);
            }
            let dist = cs.reps[0] as usize + 1;
            for _ in 0..len {
                let p = out.len();
                let b = source_byte(&out, dict, p, dist);
                out.push(b);
            }
        }
        lang_models.put(seg.lang, models);
    }
    // Canonical bodies: the encoder trims trailing zero bytes, so every body byte must have
    // been consumed — trailing bytes beyond what the coder read are a structural error.
    if out.len() != out_len || rc.consumed() < body.len() {
        return Err(DecodeError::Corrupt);
    }
    Ok(out)
}

#[inline]
fn source_byte(out: &[u8], dict: &[u8], pos: usize, dist: usize) -> u8 {
    if dist <= pos {
        out[pos - dist]
    } else {
        dict[dict.len() - (dist - pos)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip(primed: &[&'static Primed], doc: &[u8], segments: &[Segment]) {
        let lookup = |lang: u8| primed[lang as usize];
        let body = encode_doc(&lookup, doc, segments);
        let restored = decode_doc(&lookup, &body, doc.len(), segments).expect("decode");
        assert_eq!(restored, doc, "round-trip mismatch (doc len {})", doc.len());
    }

    fn leak(bytes: Vec<u8>) -> &'static Primed {
        Box::leak(Box::new(Primed::new(bytes, None, [0; 256])))
    }

    fn single(len: usize, lang: u8) -> Vec<Segment> {
        vec![Segment { end: len, lang }]
    }

    #[test]
    fn round_trips_without_dictionary() {
        let primed = [leak(Vec::new())];
        for doc in [
            &b""[..],
            b"a",
            b"hello hello hello hello hello hello",
            "日本語テキストの繰り返し。日本語テキストの繰り返し。".as_bytes(),
            &[0u8; 5000],
        ] {
            round_trip(&primed, doc, &single(doc.len(), 0));
        }
    }

    #[test]
    fn round_trips_with_dictionary_and_segments() {
        let dict_text: Vec<u8> =
            b"function add(a, b) { return a + b; }\nconst result = add(1, 2);\nconsole.log(result);\n".repeat(50);
        let js = leak(dict_text.clone());
        let prose = leak(
            "これは日本語の辞書です。ゲームの仕様を以下に示します。"
                .repeat(40)
                .into_bytes(),
        );
        let primed = [prose, js];
        let doc = b"function add(a, b) { return a + b; }\nconsole.log(add(3, 4));\n";
        round_trip(&primed, doc, &single(doc.len(), 1));
        let unrelated = b"completely unrelated content with no dictionary overlap at all";
        round_trip(&primed, unrelated, &single(unrelated.len(), 0));
        round_trip(&primed, &dict_text[100..400], &single(300, 1));
        let mut mixed = "ゲームの仕様を以下に示します。\n```js\n"
            .to_string()
            .into_bytes();
        mixed.extend_from_slice(&dict_text[..200]);
        mixed.extend_from_slice("```\n以上がゲームの仕様です。".as_bytes());
        let segs = vec![
            Segment { end: 50, lang: 0 },
            Segment { end: 250, lang: 1 },
            Segment {
                end: mixed.len(),
                lang: 0,
            },
        ];
        round_trip(&primed, &mixed, &segs);
        let lookup = |lang: u8| primed[lang as usize];
        let body_mixed = encode_doc(&lookup, &mixed, &segs);
        let body_single = encode_doc(&lookup, &mixed, &single(mixed.len(), 0));
        assert!(
            body_mixed.len() < body_single.len(),
            "segments should help: {} vs {}",
            body_mixed.len(),
            body_single.len()
        );
    }

    #[test]
    fn round_trips_pseudo_random() {
        let mut x = 0x1234_5678u32;
        let mut doc = Vec::new();
        for i in 0..20_000 {
            x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            if i % 7 == 0 {
                doc.extend(std::iter::repeat_n((x >> 24) as u8, (x % 30) as usize));
            } else {
                doc.push((x >> 24) as u8);
            }
        }
        round_trip(
            &[leak(doc[..4096].to_vec())],
            &doc[4096..],
            &single(doc.len() - 4096, 0),
        );
        round_trip(&[leak(Vec::new())], &doc, &single(doc.len(), 0));
    }
}
