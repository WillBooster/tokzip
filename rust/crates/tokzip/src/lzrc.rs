//! LZ77 over a dictionary-prefix window + adaptive binary range coding.
//!
//! The design follows the two levers the TS experiments identified as zstd's
//! format-level blind spots (issue #19):
//!
//! 1. Adaptive, context-conditioned entropy coding: every decision bit and
//!    literal bit is coded with an adaptive probability, literals are
//!    conditioned on the previous byte's high bits (order-1) and, after a
//!    match, on the byte the match would have predicted.
//! 2. Unbounded dictionaries: the dictionary is simply the prefix of the LZ
//!    window, so matches address it directly and its size has no format limit.
//!
//! On top of that, the model state is *primed*: [`Dictionary::new`] simulates
//! compressing the dictionary against itself and keeps the resulting adaptive
//! probabilities, so a 300-byte document starts from statistics trained on the
//! dictionary instead of flat 50/50 models. Priming is a pure function of the
//! dictionary bytes, so encoder and decoder derive identical state.
//!
//! The symbol alphabet (literal / match / 4 recent-distance "rep" matches,
//! length and distance-slot coding) follows the well-studied LZMA layout.

use crate::rc::{Decoder, Encoder, PROB_INIT};
use crate::DecodeError;

pub(crate) const MATCH_MAX: usize = 273;
const NUM_STATES: usize = 12;
const LC: u32 = 3; // literal context: high bits of the previous byte
const NUM_LEN_TO_POS: usize = 4;
const START_POS_MODEL: u32 = 4;
const END_POS_MODEL: u32 = 14;
const NUM_SPEC_POS: usize = 114; // (2|1)<<5 (base of slot 13) - 13 + (1<<5) - 1
const ALIGN_BITS: u32 = 4;

const HASH4_BITS: u32 = 17;
const HASH3_BITS: u32 = 15;
// A length-3 match further away than this costs more than three modeled
// literals in practice, so the finder never proposes one.
const MAX_DIST_LEN3: u32 = 1 << 14;
const SEARCH_DEPTH: usize = 512;
// Stop searching (and skip lazy evaluation) once a match is at least this long.
const NICE_LEN: usize = 273;
const EMPTY: i32 = -1;

// ---------------------------------------------------------------------------
// Probability models
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct LenModel {
    choice: u16,
    choice2: u16,
    low: [u16; 8],
    mid: [u16; 8],
    high: [u16; 256],
}

impl LenModel {
    fn new() -> Self {
        Self {
            choice: PROB_INIT,
            choice2: PROB_INIT,
            low: [PROB_INIT; 8],
            mid: [PROB_INIT; 8],
            high: [PROB_INIT; 256],
        }
    }

    fn encode(&mut self, rc: &mut Encoder, len: usize) {
        let l = (len - 2) as u32;
        if l < 8 {
            rc.encode_bit(&mut self.choice, 0);
            rc.encode_tree(&mut self.low, 3, l);
        } else if l < 16 {
            rc.encode_bit(&mut self.choice, 1);
            rc.encode_bit(&mut self.choice2, 0);
            rc.encode_tree(&mut self.mid, 3, l - 8);
        } else {
            rc.encode_bit(&mut self.choice, 1);
            rc.encode_bit(&mut self.choice2, 1);
            rc.encode_tree(&mut self.high, 8, l - 16);
        }
    }

    fn decode(&mut self, rc: &mut Decoder) -> usize {
        let l = if rc.decode_bit(&mut self.choice) == 0 {
            rc.decode_tree(&mut self.low, 3)
        } else if rc.decode_bit(&mut self.choice2) == 0 {
            8 + rc.decode_tree(&mut self.mid, 3)
        } else {
            16 + rc.decode_tree(&mut self.high, 8)
        };
        l as usize + 2
    }
}

#[derive(Clone)]
pub(crate) struct Models {
    is_match: [u16; NUM_STATES],
    is_rep: [u16; NUM_STATES],
    is_rep_g0: [u16; NUM_STATES],
    is_rep_g1: [u16; NUM_STATES],
    is_rep_g2: [u16; NUM_STATES],
    is_rep0_long: [u16; NUM_STATES],
    lit: Vec<u16>, // (1 << LC) contexts x 0x300 tree probabilities
    len: LenModel,
    rep_len: LenModel,
    dist_slot: [[u16; 64]; NUM_LEN_TO_POS],
    spec_pos: [u16; NUM_SPEC_POS],
    align: [u16; 1 << ALIGN_BITS],
}

impl Models {
    fn new() -> Self {
        Self {
            is_match: [PROB_INIT; NUM_STATES],
            is_rep: [PROB_INIT; NUM_STATES],
            is_rep_g0: [PROB_INIT; NUM_STATES],
            is_rep_g1: [PROB_INIT; NUM_STATES],
            is_rep_g2: [PROB_INIT; NUM_STATES],
            is_rep0_long: [PROB_INIT; NUM_STATES],
            lit: vec![PROB_INIT; (1 << LC) * 0x300],
            len: LenModel::new(),
            rep_len: LenModel::new(),
            dist_slot: [[PROB_INIT; 64]; NUM_LEN_TO_POS],
            spec_pos: [PROB_INIT; NUM_SPEC_POS],
            align: [PROB_INIT; 1 << ALIGN_BITS],
        }
    }
}

// ---------------------------------------------------------------------------
// Bit prices (in 1/16-bit units) for the optimal parse
// ---------------------------------------------------------------------------

/// `PROB_PRICES[p >> 4]` ~= -16 * log2(p / 2048): the cost of coding the
/// 0-branch of a probability `p`, quantized to 128 entries.
fn prob_prices() -> &'static [u32; 128] {
    static PRICES: std::sync::OnceLock<[u32; 128]> = std::sync::OnceLock::new();
    PRICES.get_or_init(|| {
        let mut t = [0u32; 128];
        for (i, slot) in t.iter_mut().enumerate() {
            let p = (i as f64 * 16.0 + 8.0) / 2048.0;
            *slot = (-p.log2() * 16.0).round() as u32;
        }
        t
    })
}

fn price_bit(prob: u16, bit: u32) -> u32 {
    let prices = prob_prices();
    if bit == 0 {
        prices[(prob >> 4) as usize]
    } else {
        prices[(((1u32 << 11) - u32::from(prob)) >> 4) as usize]
    }
}

fn price_tree(probs: &[u16], bits: u32, symbol: u32) -> u32 {
    let mut price = 0;
    let mut m = 1usize;
    for i in (0..bits).rev() {
        let bit = (symbol >> i) & 1;
        price += price_bit(probs[m], bit);
        m = (m << 1) | bit as usize;
    }
    price
}

fn price_tree_reverse(probs: &[u16], bits: u32, symbol: u32) -> u32 {
    let mut price = 0;
    let mut m = 1usize;
    for i in 0..bits {
        let bit = (symbol >> i) & 1;
        price += price_bit(probs[m - 1], bit);
        m = (m << 1) | bit as usize;
    }
    price
}

impl LenModel {
    fn price(&self, len: usize) -> u32 {
        let l = (len - 2) as u32;
        if l < 8 {
            price_bit(self.choice, 0) + price_tree(&self.low, 3, l)
        } else if l < 16 {
            price_bit(self.choice, 1) + price_bit(self.choice2, 0) + price_tree(&self.mid, 3, l - 8)
        } else {
            price_bit(self.choice, 1)
                + price_bit(self.choice2, 1)
                + price_tree(&self.high, 8, l - 16)
        }
    }
}

impl Models {
    /// Price of the literal at `pos` (excluding the `is_match` decision bit).
    fn price_literal(&self, win: &Window, pos: usize, state: usize, reps: &[u32; 4]) -> u32 {
        let prev = if pos > 0 { win.get(pos - 1) } else { 0 };
        let ctx = (prev >> (8 - LC)) as usize;
        let probs = &self.lit[ctx * 0x300..(ctx + 1) * 0x300];
        let symbol = u32::from(win.get(pos));
        let mut price = 0;
        let mut m = 1usize;
        let mut i = 8u32;
        if state >= 7 {
            let match_byte = u32::from(win.get(pos - (reps[0] as usize + 1)));
            while i > 0 {
                i -= 1;
                let mb = (match_byte >> i) & 1;
                let bit = (symbol >> i) & 1;
                price += price_bit(probs[((1 + mb as usize) << 8) + m], bit);
                m = (m << 1) | bit as usize;
                if mb != bit {
                    break;
                }
            }
        }
        while i > 0 {
            i -= 1;
            let bit = (symbol >> i) & 1;
            price += price_bit(probs[m], bit);
            m = (m << 1) | bit as usize;
        }
        price
    }

    /// Price of the distance part of a match (slot + tail), given the length
    /// context. The length itself is priced by [`LenModel::price`].
    fn price_dist(&self, len_state: usize, dist_m1: u32) -> u32 {
        let slot = dist_slot(dist_m1);
        let mut price = price_tree(&self.dist_slot[len_state], 6, slot);
        if slot >= START_POS_MODEL {
            let footer = (slot >> 1) - 1;
            let base = (2 | (slot & 1)) << footer;
            let reduced = dist_m1 - base;
            if slot < END_POS_MODEL {
                price +=
                    price_tree_reverse(&self.spec_pos[(base - slot) as usize..], footer, reduced);
            } else {
                price += (footer - ALIGN_BITS) << 4;
                price += price_tree_reverse(&self.align, ALIGN_BITS, reduced & 0xF);
            }
        }
        price
    }

    /// Price of selecting rep index `idx` (the `is_rep_g*` cascade after
    /// `is_match`/`is_rep`), excluding the length.
    fn price_rep_choice(&self, state: usize, idx: usize) -> u32 {
        if idx == 0 {
            price_bit(self.is_rep_g0[state], 0) + price_bit(self.is_rep0_long[state], 1)
        } else {
            let mut price = price_bit(self.is_rep_g0[state], 1);
            if idx == 1 {
                price += price_bit(self.is_rep_g1[state], 0);
            } else {
                price += price_bit(self.is_rep_g1[state], 1)
                    + price_bit(self.is_rep_g2[state], (idx == 3) as u32);
            }
            price
        }
    }
}

/// Coder state carried across symbols: the LZMA 12-state machine plus the four
/// most recent match distances (stored as distance - 1).
#[derive(Clone)]
pub(crate) struct CoderState {
    state: usize,
    reps: [u32; 4],
}

impl CoderState {
    fn new() -> Self {
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

// ---------------------------------------------------------------------------
// Window & match finder
// ---------------------------------------------------------------------------

/// The LZ window: dictionary prefix followed by the document, addressed by one
/// global position without materializing the concatenation per call.
#[derive(Clone, Copy)]
struct Window<'a> {
    dict: &'a [u8],
    doc: &'a [u8],
}

impl<'a> Window<'a> {
    fn len(&self) -> usize {
        self.dict.len() + self.doc.len()
    }

    #[inline]
    fn get(&self, pos: usize) -> u8 {
        if pos < self.dict.len() {
            self.dict[pos]
        } else {
            self.doc[pos - self.dict.len()]
        }
    }

    /// Longest common extension of `a` and `b` (b > a), capped at `max`.
    /// Overlap (b inside the copied region) is fine: comparisons only look at
    /// already-defined window bytes.
    fn common_len(&self, a: usize, b: usize, max: usize) -> usize {
        let mut l = 0;
        while l < max && self.get(a + l) == self.get(b + l) {
            l += 1;
        }
        l
    }
}

fn hash4(w: &Window, pos: usize) -> usize {
    let v = u32::from(w.get(pos))
        | u32::from(w.get(pos + 1)) << 8
        | u32::from(w.get(pos + 2)) << 16
        | u32::from(w.get(pos + 3)) << 24;
    (v.wrapping_mul(0x9E37_79B1) >> (32 - HASH4_BITS)) as usize
}

fn hash3(w: &Window, pos: usize) -> usize {
    let v =
        u32::from(w.get(pos)) | u32::from(w.get(pos + 1)) << 8 | u32::from(w.get(pos + 2)) << 16;
    (v.wrapping_mul(0x9E37_79B1) >> (32 - HASH3_BITS)) as usize
}

/// Hash-chain match finder. Chain links for dictionary positions are built once
/// during priming and shared immutably; per-document runs clone only the hash
/// heads and grow their own links, so the heavy per-dictionary state is reused.
struct MatchFinder<'a> {
    win: Window<'a>,
    head4: Vec<i32>,
    head3: Vec<i32>,
    dict_prev: &'a [i32],
    doc_prev: Vec<i32>,
}

impl<'a> MatchFinder<'a> {
    fn prev(&self, pos: usize) -> i32 {
        let dlen = self.win.dict.len();
        if pos < dlen {
            self.dict_prev[pos]
        } else {
            self.doc_prev[pos - dlen]
        }
    }

    fn insert(&mut self, pos: usize) {
        let total = self.win.len();
        if pos + 3 <= total {
            // head3 keeps only the most recent occurrence: length-3 matches are
            // only worthwhile nearby, so a chain would add cost without gain.
            self.head3[hash3(&self.win, pos)] = pos as i32;
        }
        if pos + 4 > total {
            return;
        }
        let h = hash4(&self.win, pos);
        let link = self.head4[h];
        let dlen = self.win.dict.len();
        if pos >= dlen {
            self.doc_prev[pos - dlen] = link;
        }
        self.head4[h] = pos as i32;
    }

    /// All useful matches at `pos` as `(len, dist)` pairs with strictly
    /// increasing lengths; each pair carries the nearest distance found for
    /// that length (chains are walked nearest-first). Feeds the optimal parse.
    fn find_pairs(&self, pos: usize, max_len: usize, pairs: &mut Vec<(u32, u32)>) {
        pairs.clear();
        let total = self.win.len();
        let mut best_len = 0usize;
        if pos + 3 <= total && max_len >= 3 {
            let cand = self.head3[hash3(&self.win, pos)];
            if cand >= 0 {
                let c = cand as usize;
                let dist = (pos - c) as u32;
                if dist <= MAX_DIST_LEN3 {
                    let l = self.win.common_len(c, pos, max_len);
                    if l >= 3 {
                        best_len = l;
                        pairs.push((l as u32, dist));
                    }
                }
            }
        }
        if pos + 4 <= total && max_len >= 4 {
            let mut cand = self.head4[hash4(&self.win, pos)];
            for _ in 0..SEARCH_DEPTH {
                if cand < 0 {
                    break;
                }
                let c = cand as usize;
                if best_len == 0
                    || (pos + best_len < total
                        && self.win.get(c + best_len) == self.win.get(pos + best_len))
                {
                    let l = self.win.common_len(c, pos, max_len);
                    if l >= 4 && l > best_len {
                        best_len = l;
                        pairs.push((l as u32, (pos - c) as u32));
                        if l >= NICE_LEN || l == max_len {
                            return;
                        }
                    }
                }
                cand = self.prev(c);
            }
        }
    }

    /// Longest match ending the chain walk early at `NICE_LEN`. Returns
    /// `(len, dist)` with actual distance, or `(0, 0)`. Chains are walked from
    /// nearest to farthest, so on equal length the nearest (cheapest) wins.
    fn find(&self, pos: usize, max_len: usize) -> (usize, u32) {
        let total = self.win.len();
        let mut best_len = 0usize;
        let mut best_dist = 0u32;
        if pos + 4 <= total && max_len >= 4 {
            let mut cand = self.head4[hash4(&self.win, pos)];
            for _ in 0..SEARCH_DEPTH {
                if cand < 0 {
                    break;
                }
                let c = cand as usize;
                debug_assert!(c < pos);
                // Cheap pre-check: a candidate can only improve on best_len if
                // it also matches at offset best_len.
                if best_len == 0
                    || (pos + best_len < total
                        && self.win.get(c + best_len) == self.win.get(pos + best_len))
                {
                    let l = self.win.common_len(c, pos, max_len);
                    if l >= 4 && l > best_len {
                        best_len = l;
                        best_dist = (pos - c) as u32;
                        if l >= NICE_LEN || l == max_len {
                            return (best_len, best_dist);
                        }
                    }
                }
                cand = self.prev(c);
            }
        }
        if best_len < 3 && pos + 3 <= total && max_len >= 3 {
            let cand = self.head3[hash3(&self.win, pos)];
            if cand >= 0 {
                let c = cand as usize;
                let dist = (pos - c) as u32;
                if dist <= MAX_DIST_LEN3 && self.win.common_len(c, pos, max_len) >= 3 {
                    return (3, dist);
                }
            }
        }
        (best_len, best_dist)
    }

    /// Match length at `pos` against the source `dist` bytes back (0 if the
    /// distance underflows the window).
    fn len_at(&self, pos: usize, dist: u32, max_len: usize) -> usize {
        let dist = dist as usize;
        if dist == 0 || dist > pos {
            return 0;
        }
        self.win.common_len(pos - dist, pos, max_len)
    }
}

// ---------------------------------------------------------------------------
// Dictionary (shared, primed state)
// ---------------------------------------------------------------------------

/// A prepared compression dictionary: the raw bytes, the match-finder chains
/// over them, and the adaptive model state primed by simulating compression of
/// the dictionary against itself. Build once, share across documents.
pub struct Dictionary {
    bytes: Vec<u8>,
    head4: Vec<i32>,
    head3: Vec<i32>,
    prev: Vec<i32>,
    models: Models,
    state: CoderState,
}

impl Dictionary {
    pub fn new(bytes: &[u8]) -> Self {
        let mut models = Models::new();
        let mut state = CoderState::new();
        let mut mf = MatchFinder {
            win: Window {
                dict: &[],
                doc: bytes,
            },
            head4: vec![EMPTY; 1 << HASH4_BITS],
            head3: vec![EMPTY; 1 << HASH3_BITS],
            dict_prev: &[],
            doc_prev: vec![EMPTY; bytes.len()],
        };
        // The range-coder output is discarded; only the model adaptation and
        // the chains it builds matter. The parse is deterministic, so the
        // decoder side reconstructs the identical state from the same bytes.
        let mut rc = Encoder::new();
        run_encode(&mut rc, &mut models, &mut state, &mut mf, 0);
        Self {
            bytes: bytes.to_vec(),
            head4: mf.head4,
            head3: mf.head3,
            prev: mf.doc_prev,
            models,
            state,
        }
    }

    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Encoder
// ---------------------------------------------------------------------------

pub(crate) fn encode_doc(dict: &Dictionary, doc: &[u8]) -> Vec<u8> {
    let mut models = dict.models.clone();
    let mut state = dict.state.clone();
    let mut mf = MatchFinder {
        win: Window {
            dict: &dict.bytes,
            doc,
        },
        head4: dict.head4.clone(),
        head3: dict.head3.clone(),
        dict_prev: &dict.prev,
        doc_prev: vec![EMPTY; doc.len()],
    };
    let mut rc = Encoder::new();
    run_encode_optimal(&mut rc, &mut models, &mut state, &mut mf, dict.bytes.len());
    rc.finish()
}

#[derive(Clone, Copy)]
enum Step {
    Literal,
    Match { len: u32, dist_m1: u32 },
    Rep { idx: u8, len: u32 },
}

/// One node of the shortest-path parse: cheapest known way to reach this
/// position, with the (path-dependent) coder state it arrives in.
#[derive(Clone, Copy)]
struct Node {
    price: u32,
    prev: u32,
    step: Step,
    state: u8,
    reps: [u32; 4],
}

const INF: u32 = u32::MAX;

/// Price-based optimal parse: a forward shortest-path over positions, using
/// exact model bit prices. Prices are refreshed every `CHUNK` positions — the
/// chosen path for a chunk is replayed through the adaptive coder before the
/// next chunk is parsed, so drift stays bounded. Reachability note: a literal
/// edge always advances one position, so every node is reachable.
fn run_encode_optimal(
    rc: &mut Encoder,
    models: &mut Models,
    cs: &mut CoderState,
    mf: &mut MatchFinder,
    start: usize,
) {
    const CHUNK: usize = 4096;
    let total = mf.win.len();
    let mut pos = start;
    let mut inserted = start;
    let mut pairs: Vec<(u32, u32)> = Vec::new();
    let mut steps: Vec<Step> = Vec::new();
    let mut nodes: Vec<Node> = Vec::new();
    while pos < total {
        let end_target = (pos + CHUNK).min(total);
        let max_reach = (end_target + MATCH_MAX).min(total);
        let n = max_reach - pos;
        // Reuse the allocation across chunks; `resize` refills every slot, so
        // stale prices from the previous chunk never leak in.
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

        // Per-chunk length price tables (models are frozen within a chunk).
        let mut match_len_price = [0u32; MATCH_MAX + 1];
        let mut rep_len_price = [0u32; MATCH_MAX + 1];
        for len in 2..=MATCH_MAX {
            match_len_price[len] = models.len.price(len);
            rep_len_price[len] = models.rep_len.price(len);
        }

        for i in 0..(end_target - pos) {
            if nodes[i].price == INF {
                continue;
            }
            let gpos = pos + i;
            while inserted < gpos {
                mf.insert(inserted);
                inserted += 1;
            }
            let state = nodes[i].state as usize;
            let reps = nodes[i].reps;
            let base = nodes[i].price;
            let max_len = MATCH_MAX.min(total - gpos);

            let lit_price = base
                + price_bit(models.is_match[state], 0)
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

            let match_bit = price_bit(models.is_match[state], 1);
            let rep_bit = match_bit + price_bit(models.is_rep[state], 1);

            for idx in 0..4usize {
                let l = mf.len_at(gpos, reps[idx] + 1, max_len);
                if idx == 0 && l >= 1 {
                    // Short rep: one byte at rep0.
                    let price = base
                        + rep_bit
                        + price_bit(models.is_rep_g0[state], 0)
                        + price_bit(models.is_rep0_long[state], 0);
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
                let from = if l >= NICE_LEN { l } else { 2 };
                // The range IS the iteration domain here; an enumerate() over the
                // price table would obscure the len semantics.
                #[expect(clippy::needless_range_loop)]
                for len in from..=l {
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

            mf.find_pairs(gpos, max_len, &mut pairs);
            let mat_bit = match_bit + price_bit(models.is_rep[state], 0);
            let st = state_after_match(state) as u8;
            let mut len_from = 2usize;
            for &(plen, pdist) in &pairs {
                let plen = plen as usize;
                let dist_m1 = pdist - 1;
                let new_reps = [dist_m1, reps[0], reps[1], reps[2]];
                // Distance price only depends on the length via its saturated
                // context, so compute it once per context, not per length.
                let mut dist_price = [0u32; NUM_LEN_TO_POS];
                for (ls, p) in dist_price.iter_mut().enumerate() {
                    *p = models.price_dist(ls, dist_m1);
                }
                let from = if plen >= NICE_LEN { plen } else { len_from };
                // Same as above: `len` is the semantic loop variable.
                #[expect(clippy::needless_range_loop)]
                for len in from..=plen {
                    let len_state = (len - 2).min(NUM_LEN_TO_POS - 1);
                    relax(
                        &mut nodes,
                        i,
                        len,
                        base + mat_bit + match_len_price[len] + dist_price[len_state],
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

        // Cheapest node at or past the chunk boundary ends this chunk's path.
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
                    emit_match(rc, models, cs, len as usize, dist_m1);
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

/// One-step-lazy greedy parse (LZMA "fast mode" heuristics): prefer rep
/// matches when they are nearly as long as the main match, and defer to a
/// literal when the next position offers a clearly better match.
fn run_encode(
    rc: &mut Encoder,
    models: &mut Models,
    cs: &mut CoderState,
    mf: &mut MatchFinder,
    start: usize,
) {
    let total = mf.win.len();
    let mut pos = start;
    let mut inserted = start;
    while pos < total {
        while inserted < pos {
            mf.insert(inserted);
            inserted += 1;
        }
        let max_len = MATCH_MAX.min(total - pos);

        let mut rep_len = 0usize;
        let mut rep_idx = 0usize;
        for (i, &rep) in cs.reps.iter().enumerate() {
            let l = mf.len_at(pos, rep + 1, max_len);
            if l > rep_len {
                rep_len = l;
                rep_idx = i;
            }
        }
        let (main_len, main_dist) = mf.find(pos, max_len);

        let use_rep = rep_len >= 2
            && (rep_len + 1 >= main_len
                || (rep_len + 2 >= main_len && main_dist >= (1 << 9))
                || (rep_len + 3 >= main_len && main_dist >= (1 << 15)));
        if use_rep {
            emit_rep(rc, models, cs, rep_idx, rep_len);
            pos += rep_len;
            continue;
        }
        if main_len < 2 {
            emit_literal(rc, models, cs, &mf.win, pos);
            pos += 1;
            continue;
        }

        // Lazy step: if pos+1 has a strictly better main or rep match, emit a
        // literal now and take the better match next iteration.
        let mut defer = false;
        if main_len < NICE_LEN && pos + 1 < total {
            mf.insert(pos);
            inserted = pos + 1;
            let next_max = MATCH_MAX.min(total - pos - 1);
            let (next_len, next_dist) = mf.find(pos + 1, next_max);
            if next_len >= 2 {
                defer = (next_len >= main_len && next_dist < main_dist)
                    || (next_len == main_len + 1 && !change_pair(main_dist, next_dist))
                    || next_len > main_len + 1
                    || (next_len + 1 >= main_len
                        && main_len >= 3
                        && change_pair(next_dist, main_dist));
            }
            if !defer {
                for &rep in &cs.reps {
                    if mf.len_at(pos + 1, rep + 1, next_max) + 1 >= main_len {
                        defer = true;
                        break;
                    }
                }
            }
        }
        if defer {
            emit_literal(rc, models, cs, &mf.win, pos);
            pos += 1;
        } else {
            emit_match(rc, models, cs, main_len, main_dist - 1);
            pos += main_len;
        }
    }
}

/// True when `big_dist` is disproportionately farther than `small_dist`, i.e.
/// switching to it is only worth one extra length unit if it stays within ~128x.
fn change_pair(small_dist: u32, big_dist: u32) -> bool {
    (big_dist >> 7) > small_dist
}

fn emit_literal(
    rc: &mut Encoder,
    models: &mut Models,
    cs: &mut CoderState,
    win: &Window,
    pos: usize,
) {
    rc.encode_bit(&mut models.is_match[cs.state], 0);
    let prev = if pos > 0 { win.get(pos - 1) } else { 0 };
    let ctx = (prev >> (8 - LC)) as usize;
    let probs = &mut models.lit[ctx * 0x300..(ctx + 1) * 0x300];
    let symbol = u32::from(win.get(pos));
    let mut m = 1usize;
    let mut i = 8u32;
    if cs.prev_was_match() {
        // Matched-literal mode: condition on the byte the last match distance
        // predicts, until the prediction first diverges.
        let match_byte = u32::from(win.get(pos - (cs.reps[0] as usize + 1)));
        while i > 0 {
            i -= 1;
            let mb = (match_byte >> i) & 1;
            let bit = (symbol >> i) & 1;
            rc.encode_bit(&mut probs[((1 + mb as usize) << 8) + m], bit);
            m = (m << 1) | bit as usize;
            if mb != bit {
                break;
            }
        }
    }
    while i > 0 {
        i -= 1;
        let bit = (symbol >> i) & 1;
        rc.encode_bit(&mut probs[m], bit);
        m = (m << 1) | bit as usize;
    }
    cs.state = state_after_literal(cs.state);
}

fn emit_match(
    rc: &mut Encoder,
    models: &mut Models,
    cs: &mut CoderState,
    len: usize,
    dist_m1: u32,
) {
    rc.encode_bit(&mut models.is_match[cs.state], 1);
    rc.encode_bit(&mut models.is_rep[cs.state], 0);
    models.len.encode(rc, len);
    let len_state = (len - 2).min(NUM_LEN_TO_POS - 1);
    let slot = dist_slot(dist_m1);
    rc.encode_tree(&mut models.dist_slot[len_state], 6, slot);
    if slot >= START_POS_MODEL {
        let footer = (slot >> 1) - 1;
        let base = (2 | (slot & 1)) << footer;
        let reduced = dist_m1 - base;
        if slot < END_POS_MODEL {
            rc.encode_tree_reverse(
                &mut models.spec_pos[(base - slot) as usize..],
                footer,
                reduced,
            );
        } else {
            rc.encode_direct_bits(reduced >> ALIGN_BITS, footer - ALIGN_BITS);
            rc.encode_tree_reverse(&mut models.align, ALIGN_BITS, reduced & 0xF);
        }
    }
    cs.reps = [dist_m1, cs.reps[0], cs.reps[1], cs.reps[2]];
    cs.state = state_after_match(cs.state);
}

fn emit_rep(rc: &mut Encoder, models: &mut Models, cs: &mut CoderState, idx: usize, len: usize) {
    debug_assert!(len >= 2 || idx == 0);
    rc.encode_bit(&mut models.is_match[cs.state], 1);
    rc.encode_bit(&mut models.is_rep[cs.state], 1);
    if idx == 0 {
        rc.encode_bit(&mut models.is_rep_g0[cs.state], 0);
        if len == 1 {
            // Short rep: single byte at rep0.
            rc.encode_bit(&mut models.is_rep0_long[cs.state], 0);
            cs.state = if cs.state < 7 { 9 } else { 11 };
            return;
        }
        rc.encode_bit(&mut models.is_rep0_long[cs.state], 1);
    } else {
        rc.encode_bit(&mut models.is_rep_g0[cs.state], 1);
        if idx == 1 {
            rc.encode_bit(&mut models.is_rep_g1[cs.state], 0);
        } else {
            rc.encode_bit(&mut models.is_rep_g1[cs.state], 1);
            rc.encode_bit(&mut models.is_rep_g2[cs.state], (idx == 3) as u32);
        }
        let rep = cs.reps[idx];
        cs.reps.copy_within(0..idx, 1);
        cs.reps[0] = rep;
    }
    models.rep_len.encode(rc, len);
    cs.state = state_after_rep(cs.state);
}

fn dist_slot(dist_m1: u32) -> u32 {
    if dist_m1 < START_POS_MODEL {
        dist_m1
    } else {
        let n = 31 - dist_m1.leading_zeros();
        (n << 1) | ((dist_m1 >> (n - 1)) & 1)
    }
}

// ---------------------------------------------------------------------------
// Decoder
// ---------------------------------------------------------------------------

pub(crate) fn decode_doc(
    dict: &Dictionary,
    body: &[u8],
    out_len: usize,
) -> Result<Vec<u8>, DecodeError> {
    let mut models = dict.models.clone();
    let mut cs = dict.state.clone();
    let dlen = dict.bytes.len();
    let mut out: Vec<u8> = Vec::with_capacity(out_len);
    let mut rc = Decoder::new(body);
    // Matches copy byte-by-byte across the dict/output boundary; overlapping
    // copies (dist < len) rely on this order.
    let byte_at = |out: &Vec<u8>, gpos: usize| -> u8 {
        if gpos < dlen {
            dict.bytes[gpos]
        } else {
            out[gpos - dlen]
        }
    };
    while out.len() < out_len {
        let gpos = dlen + out.len();
        if rc.decode_bit(&mut models.is_match[cs.state]) == 0 {
            let prev = if gpos > 0 { byte_at(&out, gpos - 1) } else { 0 };
            let ctx = (prev >> (8 - LC)) as usize;
            let probs = &mut models.lit[ctx * 0x300..(ctx + 1) * 0x300];
            let mut m = 1usize;
            if cs.prev_was_match() {
                let dist = cs.reps[0] as usize + 1;
                if dist > gpos {
                    return Err(DecodeError::Corrupt);
                }
                let match_byte = u32::from(byte_at(&out, gpos - dist));
                let mut i = 8u32;
                while i > 0 {
                    i -= 1;
                    let mb = (match_byte >> i) & 1;
                    let bit = rc.decode_bit(&mut probs[((1 + mb as usize) << 8) + m]);
                    m = (m << 1) | bit as usize;
                    if mb != bit {
                        break;
                    }
                }
            }
            while m < 0x100 {
                m = (m << 1) | rc.decode_bit(&mut probs[m]) as usize;
            }
            out.push((m & 0xFF) as u8);
            cs.state = state_after_literal(cs.state);
            continue;
        }
        let len;
        if rc.decode_bit(&mut models.is_rep[cs.state]) == 0 {
            len = models.len.decode(&mut rc);
            let len_state = (len - 2).min(NUM_LEN_TO_POS - 1);
            let slot = rc.decode_tree(&mut models.dist_slot[len_state], 6);
            let dist_m1 = if slot < START_POS_MODEL {
                slot
            } else {
                let footer = (slot >> 1) - 1;
                let base = (2 | (slot & 1)) << footer;
                let reduced = if slot < END_POS_MODEL {
                    rc.decode_tree_reverse(&mut models.spec_pos[(base - slot) as usize..], footer)
                } else {
                    (rc.decode_direct_bits(footer - ALIGN_BITS) << ALIGN_BITS)
                        | rc.decode_tree_reverse(&mut models.align, ALIGN_BITS)
                };
                base + reduced
            };
            cs.reps = [dist_m1, cs.reps[0], cs.reps[1], cs.reps[2]];
            cs.state = state_after_match(cs.state);
        } else {
            if rc.decode_bit(&mut models.is_rep_g0[cs.state]) == 0 {
                if rc.decode_bit(&mut models.is_rep0_long[cs.state]) == 0 {
                    // Short rep: single byte at rep0.
                    let dist = cs.reps[0] as usize + 1;
                    if dist > gpos {
                        return Err(DecodeError::Corrupt);
                    }
                    out.push(byte_at(&out, gpos - dist));
                    cs.state = if cs.state < 7 { 9 } else { 11 };
                    continue;
                }
            } else {
                let idx = if rc.decode_bit(&mut models.is_rep_g1[cs.state]) == 0 {
                    1
                } else if rc.decode_bit(&mut models.is_rep_g2[cs.state]) == 0 {
                    2
                } else {
                    3
                };
                let rep = cs.reps[idx];
                cs.reps.copy_within(0..idx, 1);
                cs.reps[0] = rep;
            }
            len = models.rep_len.decode(&mut rc);
            cs.state = state_after_rep(cs.state);
        }
        let dist = cs.reps[0] as usize + 1;
        if dist > dlen + out.len() || out.len() + len > out_len {
            return Err(DecodeError::Corrupt);
        }
        for _ in 0..len {
            let gpos = dlen + out.len();
            out.push(byte_at(&out, gpos - dist));
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip(dict: &Dictionary, doc: &[u8]) {
        let body = encode_doc(dict, doc);
        let restored = decode_doc(dict, &body, doc.len()).expect("decode");
        assert_eq!(restored, doc, "round-trip mismatch (doc len {})", doc.len());
    }

    #[test]
    fn round_trips_without_dictionary() {
        let dict = Dictionary::new(&[]);
        round_trip(&dict, b"");
        round_trip(&dict, b"a");
        round_trip(&dict, b"hello hello hello hello hello hello");
        round_trip(
            &dict,
            "日本語テキストの繰り返し。日本語テキストの繰り返し。".as_bytes(),
        );
        round_trip(&dict, &[0u8; 5000]);
    }

    #[test]
    fn round_trips_with_dictionary() {
        let dict_text: Vec<u8> = "function add(a, b) { return a + b; }\nconst result = add(1, 2);\nconsole.log(result);\n"
            .as_bytes()
            .repeat(50);
        let dict = Dictionary::new(&dict_text);
        round_trip(&dict, b"");
        round_trip(
            &dict,
            b"function add(a, b) { return a + b; }\nconsole.log(add(3, 4));\n",
        );
        round_trip(
            &dict,
            b"completely unrelated content with no dictionary overlap at all",
        );
        round_trip(&dict, &dict_text[100..400]);
    }

    #[test]
    fn round_trips_pseudo_random() {
        // Deterministic LCG; mixes incompressible bytes with runs.
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
        let dict = Dictionary::new(&doc[..4096]);
        round_trip(&dict, &doc[4096..]);
        round_trip(&Dictionary::new(&[]), &doc);
    }

    #[test]
    fn dictionary_matches_compress_small_docs_well() {
        let sample = "import { useState } from 'react';\nexport function Counter() {\n  const [count, setCount] = useState(0);\n  return <button onClick={() => setCount(count + 1)}>{count}</button>;\n}\n";
        let dict = Dictionary::new(sample.repeat(20).as_bytes());
        let body = encode_doc(&dict, sample.as_bytes());
        // The document is fully present in the dictionary, so it should
        // compress to a tiny fraction of its size.
        assert!(
            body.len() * 8 < sample.len(),
            "expected strong dictionary compression, got {} -> {}",
            sample.len(),
            body.len()
        );
    }
}
