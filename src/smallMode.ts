// oxlint-disable unicorn/prefer-math-trunc -- `>>> 0` coerces to unsigned 32-bit throughout the inlined range decoder; Math.trunc would keep the sign
import type { LanguageModel, RegisteredLanguage } from './dictionary.ts';
import { allocateDecodeBuffer, TokzipDecodeError } from './errors.ts';
import { copyExtendedDictMatch, FenceTracker } from './fences.ts';
import {
  INITIAL_REPS,
  LEN_GROUP_DICT,
  LEN_GROUP_HISTORY,
  LEN_GROUP_REP,
  LITERAL_BLOCK_SIZE,
  MATCH_LEN_CAP,
  MIN_LEN_REP,
  MODEL_IS_DICT,
  MODEL_IS_MATCH,
  MODEL_IS_REP,
  MODEL_LEN_TREE,
  MODEL_LITERAL,
  MODEL_OFF_TREE,
  MODEL_REP_TREE,
  OFF_GROUP_DICT,
  OFF_GROUP_HISTORY,
  offLenBucketOf,
  SLOT_TREE_BITS,
  SMALL_WINDOW,
  TOKEN_KIND_COUNT,
  TOKEN_KIND_DICT,
  TOKEN_KIND_HISTORY,
  TOKEN_KIND_LIT,
  TOKEN_KIND_REP0,
} from './format.ts';
import { dictMatcherIfUsable, type ParsePricing, type SlotPricing, type Token } from './lz.ts';
import {
  ADAPT_SHIFT,
  bitPrice,
  encodeTree,
  PROB_BITS,
  PROB_SCALE,
  RangeEncoder,
  TOP,
  treePrice,
} from './rangeCoder.ts';
import { bytesFromWords, decodeRadix85 } from './radix85.ts';
import { extraBitsOf, extraValueOf, LENGTH_SLOT_COUNT, OFFSET_SLOT_COUNT, slotOf, valueOfSlot } from './slots.ts';

// Scratch prefix buffer reused across calls (compress is synchronous).
let smallPrefixScratch = new Float64Array(0);

/** Builds the pricing model driving the shared LZ parser (see {@link smallTablesFor}). */
export function smallPricing(bytes: Uint8Array, language: RegisteredLanguage): ParsePricing {
  const tables = smallTablesFor(language.model);
  const { litContext } = language.model;
  const { litBits, parser } = tables;

  if (smallPrefixScratch.length < bytes.length + 1) {
    smallPrefixScratch = new Float64Array(Math.max(bytes.length + 1, smallPrefixScratch.length * 2, 4096));
  }
  const litCostPrefix = smallPrefixScratch;
  let acc = 0;
  let prev = 0;
  for (let i = 0; i < bytes.length; i++) {
    const byte = bytes[i]!;
    acc += litBits[litContext[prev]! * 256 + byte]!;
    litCostPrefix[i + 1] = acc;
    prev = byte;
  }

  return {
    litCostPrefix,
    repCost: (repIndex, len) => tables.avgRepSlotBits[repIndex * LENGTH_SLOT_COUNT + slotOf(len - MIN_LEN_REP)]!,
    historyCost: (dist, len) =>
      tables.avgHistSlotBits[slotOf(len - MIN_LEN_REP)]! + parser.histOffsetSlotBits[slotOf(dist - 1)]!,
    dictCost: (start, len) =>
      tables.avgDictSlotBits[slotOf(len - MIN_LEN_REP)]! + parser.dictOffsetSlotBits[slotOf(start)]!,
    lazy: true,
    window: SMALL_WINDOW,
    maxDictStart: SMALL_WINDOW,
    optimal: parser,
    dictMatcher: dictMatcherIfUsable(language, bytes.length),
  };
}

/**
 * Static bit-price tables derived from a language's priors, cached per language. The
 * optimal parse prices with the priors (the standard adaptive-coder approximation): exact
 * for short documents, slightly stale for long ones, and always consistent between the
 * parser's objective and the emitted stream's starting state.
 */
interface SmallTables {
  parser: SlotPricing;
  /** Literal bits (byte tree + the literal-continuation isMatch bit), indexed class*256+byte. */
  litBits: Float64Array;
  avgHistSlotBits: Float64Array;
  avgDictSlotBits: Float64Array;
  avgRepSlotBits: Float64Array;
}

const smallTablesCache = new WeakMap<LanguageModel, SmallTables>();

function smallTablesFor(model: LanguageModel): SmallTables {
  let cached = smallTablesCache.get(model);
  if (cached) return cached;
  const p = model.priors;
  const litClassCount = model.litClassCount;

  const isMatch0 = new Float64Array(TOKEN_KIND_COUNT);
  const isMatch1 = new Float64Array(TOKEN_KIND_COUNT);
  const isRep0 = new Float64Array(TOKEN_KIND_COUNT);
  const isRep1 = new Float64Array(TOKEN_KIND_COUNT);
  const isDict0 = new Float64Array(TOKEN_KIND_COUNT);
  const isDict1 = new Float64Array(TOKEN_KIND_COUNT);
  for (let ctx = 0; ctx < TOKEN_KIND_COUNT; ctx++) {
    isMatch0[ctx] = bitPrice(p[MODEL_IS_MATCH + ctx]!, 0);
    isMatch1[ctx] = bitPrice(p[MODEL_IS_MATCH + ctx]!, 1);
    isRep0[ctx] = bitPrice(p[MODEL_IS_REP + ctx]!, 0);
    isRep1[ctx] = bitPrice(p[MODEL_IS_REP + ctx]!, 1);
    isDict0[ctx] = bitPrice(p[MODEL_IS_DICT + ctx]!, 0);
    isDict1[ctx] = bitPrice(p[MODEL_IS_DICT + ctx]!, 1);
  }
  const lenBits = (group: number, slot: number): number =>
    treePrice(p, MODEL_LEN_TREE + group * 63, SLOT_TREE_BITS, slot) + extraBitsOf(slot);

  const litRunStartBits = new Float64Array(TOKEN_KIND_COUNT);
  const histSlotBits = new Float64Array(TOKEN_KIND_COUNT * LENGTH_SLOT_COUNT);
  const dictSlotBits = new Float64Array(TOKEN_KIND_COUNT * LENGTH_SLOT_COUNT);
  const repSlotBits = new Float64Array(TOKEN_KIND_COUNT * 4 * LENGTH_SLOT_COUNT);
  for (let ctx = 0; ctx < TOKEN_KIND_COUNT; ctx++) {
    // Literal continuation (context LIT) is folded into litBits; opening a run from another
    // context pays the difference (floored at zero — the DP requires non-negative steps).
    litRunStartBits[ctx] = Math.max(0, isMatch0[ctx]! - isMatch0[TOKEN_KIND_LIT]!);
    for (let s = 0; s < LENGTH_SLOT_COUNT; s++) {
      histSlotBits[ctx * LENGTH_SLOT_COUNT + s] =
        isMatch1[ctx]! + isRep0[ctx]! + isDict0[ctx]! + lenBits(LEN_GROUP_HISTORY, s);
      dictSlotBits[ctx * LENGTH_SLOT_COUNT + s] =
        isMatch1[ctx]! + isRep0[ctx]! + isDict1[ctx]! + lenBits(LEN_GROUP_DICT, s);
      const repLen = lenBits(LEN_GROUP_REP, s);
      for (let r = 0; r < 4; r++) {
        repSlotBits[(ctx * 4 + r) * LENGTH_SLOT_COUNT + s] =
          isMatch1[ctx]! + isRep1[ctx]! + treePrice(p, MODEL_REP_TREE + ctx * 3, 2, r) + repLen;
      }
    }
  }
  // Offsets are priced with the ≥ 5 length bucket (the dominant one for slot-varying
  // lengths); per-bucket exactness matters little to the parse and would quadruple tables.
  const histOffsetSlotBits = new Float64Array(OFFSET_SLOT_COUNT);
  const dictOffsetSlotBits = new Float64Array(OFFSET_SLOT_COUNT);
  for (let s = 0; s < OFFSET_SLOT_COUNT; s++) {
    histOffsetSlotBits[s] =
      treePrice(p, MODEL_OFF_TREE + (OFF_GROUP_HISTORY * 4 + 3) * 63, SLOT_TREE_BITS, s) + extraBitsOf(s);
    dictOffsetSlotBits[s] =
      treePrice(p, MODEL_OFF_TREE + (OFF_GROUP_DICT * 4 + 3) * 63, SLOT_TREE_BITS, s) + extraBitsOf(s);
  }

  const litBits = new Float64Array(litClassCount * 256);
  const litContinue = isMatch0[TOKEN_KIND_LIT]!;
  for (let cls = 0; cls < litClassCount; cls++) {
    const base = MODEL_LITERAL + cls * LITERAL_BLOCK_SIZE;
    for (let byte = 0; byte < 256; byte++) {
      litBits[cls * 256 + byte] = treePrice(p, base, 8, byte) + litContinue;
    }
  }

  const avgHistSlotBits = new Float64Array(LENGTH_SLOT_COUNT);
  const avgDictSlotBits = new Float64Array(LENGTH_SLOT_COUNT);
  const avgRepSlotBits = new Float64Array(4 * LENGTH_SLOT_COUNT);
  for (let s = 0; s < LENGTH_SLOT_COUNT; s++) {
    let hist = 0;
    let dict = 0;
    for (let ctx = 0; ctx < TOKEN_KIND_COUNT; ctx++) {
      hist += histSlotBits[ctx * LENGTH_SLOT_COUNT + s]!;
      dict += dictSlotBits[ctx * LENGTH_SLOT_COUNT + s]!;
    }
    avgHistSlotBits[s] = hist / TOKEN_KIND_COUNT;
    avgDictSlotBits[s] = dict / TOKEN_KIND_COUNT;
    for (let r = 0; r < 4; r++) {
      let rep = 0;
      for (let ctx = 0; ctx < TOKEN_KIND_COUNT; ctx++) {
        rep += repSlotBits[(ctx * 4 + r) * LENGTH_SLOT_COUNT + s]!;
      }
      avgRepSlotBits[r * LENGTH_SLOT_COUNT + s] = rep / TOKEN_KIND_COUNT;
    }
  }

  cached = {
    parser: { litRunStartBits, histSlotBits, dictSlotBits, repSlotBits, histOffsetSlotBits, dictOffsetSlotBits },
    litBits,
    avgHistSlotBits,
    avgDictSlotBits,
    avgRepSlotBits,
  };
  smallTablesCache.set(model, cached);
  return cached;
}

/**
 * Serializes a token list as a range-coded `small` body. `parseStart` marks pre-seeded
 * history (streaming blocks): tokens cover bytes[parseStart, n) and the literal context
 * chains from the last history byte.
 */
export function encodeSmallBody(
  tokens: Token[],
  bytes: Uint8Array,
  language: RegisteredLanguage,
  parseStart = 0
): Uint8Array {
  const { litContext, priors } = language.model;
  const state = new Uint16Array(priors);
  const encoder = new RangeEncoder();
  let prevKind = TOKEN_KIND_LIT;
  let prevByte = parseStart > 0 ? bytes[parseStart - 1]! : 0;
  let pos = parseStart;

  // rep0 replay (identical to the decoder's full rep cache, whose other entries the
  // encoder never reads) so matched literals can read the byte at rep0 distance back.
  let rep0 = INITIAL_REPS[0]!;

  const encodeLength = (group: number, len: number): void => {
    const value = len - MIN_LEN_REP;
    const slot = slotOf(value);
    encodeTree(encoder, state, MODEL_LEN_TREE + group * 63, SLOT_TREE_BITS, slot);
    const extra = extraBitsOf(slot);
    if (extra > 0) encoder.encodeDirect(extraValueOf(value, slot), extra);
  };
  const encodeOffset = (group: number, len: number, value: number): void => {
    const slot = slotOf(value);
    encodeTree(encoder, state, MODEL_OFF_TREE + (group * 4 + offLenBucketOf(len)) * 63, SLOT_TREE_BITS, slot);
    const extra = extraBitsOf(slot);
    if (extra > 0) encoder.encodeDirect(extraValueOf(value, slot), extra);
  };

  for (const token of tokens) {
    if (token.type === 'lit') {
      for (let i = token.start; i < token.end; i++) {
        const byte = bytes[i]!;
        encoder.encodeBit(state, MODEL_IS_MATCH + prevKind, 0);
        const base = MODEL_LITERAL + litContext[prevByte]! * LITERAL_BLOCK_SIZE;
        if (prevKind === TOKEN_KIND_LIT || rep0 > i) {
          encodeTree(encoder, state, base, 8, byte);
        } else {
          // Matched literal: the byte at rep0 distance predicts each bit until the first
          // mismatch (then the plain tree nodes take over).
          const matchByte = bytes[i - rep0]!;
          let node = 1;
          let matched = true;
          for (let shift = 7; shift >= 0; shift--) {
            const bit = (byte >>> shift) & 1;
            if (matched) {
              const matchBit = (matchByte >>> shift) & 1;
              encoder.encodeBit(state, base + (((1 + matchBit) << 8) | node) - 1, bit);
              if (matchBit !== bit) matched = false;
            } else {
              encoder.encodeBit(state, base + node - 1, bit);
            }
            node = node * 2 + bit;
          }
        }
        prevKind = TOKEN_KIND_LIT;
        prevByte = byte;
      }
      pos = token.end;
      continue;
    }
    encoder.encodeBit(state, MODEL_IS_MATCH + prevKind, 1);
    if (token.type === 'history' && token.rep >= 0) {
      encoder.encodeBit(state, MODEL_IS_REP + prevKind, 1);
      encodeTree(encoder, state, MODEL_REP_TREE + prevKind * 3, 2, token.rep);
      encodeLength(LEN_GROUP_REP, token.len);
      prevKind = TOKEN_KIND_REP0 + token.rep;
      rep0 = token.dist;
    } else {
      encoder.encodeBit(state, MODEL_IS_REP + prevKind, 0);
      if (token.type === 'history') {
        encoder.encodeBit(state, MODEL_IS_DICT + prevKind, 0);
        encodeLength(LEN_GROUP_HISTORY, token.len);
        encodeOffset(OFF_GROUP_HISTORY, token.len, token.dist - 1);
        prevKind = TOKEN_KIND_HISTORY;
        rep0 = token.dist;
      } else {
        encoder.encodeBit(state, MODEL_IS_DICT + prevKind, 1);
        encodeLength(LEN_GROUP_DICT, token.len);
        encodeOffset(OFF_GROUP_DICT, token.len, token.start);
        prevKind = TOKEN_KIND_DICT;
      }
    }
    pos += token.len;
    prevByte = bytes[pos - 1]!;
  }
  return encoder.finish();
}

/** Decodes a text-frame `small` body in data[pos, end) into exactly `outputSize` bytes. */
export function decodeSmallBody(
  data: string,
  pos: number,
  end: number,
  outputSize: number,
  language: RegisteredLanguage,
  fenced = false
): Uint8Array {
  const body = bytesFromWords(decodeRadix85(data, pos, end));
  return decodeSmallCore(body, 0, body.length, outputSize, language, fenced, undefined, 3);
}

/** Decodes a binary-frame `small` body in body[pos, end) into exactly `outputSize` bytes. */
export function decodeSmallBodyBinary(
  body: Uint8Array,
  pos: number,
  end: number,
  outputSize: number,
  language: RegisteredLanguage,
  fenced = false,
  history?: Uint8Array
): Uint8Array {
  return decodeSmallCore(body, pos, end, outputSize, language, fenced, history, 0);
}

/**
 * Range-coded `small` decode core. `padAllowance` is the number of trailing zero padding
 * bytes the channel may append (3 for radix-85 text frames, whose payload is padded to a
 * 32-bit word; 0 for binary frames): the decoder must consume every byte before the
 * padding, and the padding itself must be zero (canonical frames).
 *
 * The range decoder is inlined here as local variables instead of using {@link RangeDecoder}
 * (which stays the reference implementation, property-tested against the encoder): V8 — the
 * production runtime (Cloudflare Workers, Node, Chrome) — does not inline the class methods
 * into this loop, and keeping `range`/`code`/`bpos` in locals with every bit decode expanded
 * in place nearly doubles decode throughput there (~22 → ~41 MB/s on the bench corpus,
 * Node 24). JSC (Bun, Safari) inlines the class methods fine and loses ~15% from the large
 * function body instead — accepted, since Bun is only the development runtime.
 */
function decodeSmallCore(
  body: Uint8Array,
  pos: number,
  end: number,
  outputSize: number,
  language: RegisteredLanguage,
  fenced: boolean,
  history: Uint8Array | undefined,
  padAllowance: number
): Uint8Array {
  // Structural output bound, checked before allocating: a literal costs ≥ 9 model decisions
  // and a match ≥ 10, each consuming ≥ 1/46 bit even at the adaptive probability clamp, so
  // a declared size beyond bodyBits × 5 × MATCH_LEN_CAP cannot be produced by this body
  // (this stops forged huge-size frames from forcing enormous allocations under
  // `maxOutputSize: Infinity`).
  if (outputSize > (end - pos) * 8 * 5 * MATCH_LEN_CAP) {
    throw new TokzipDecodeError('declared size exceeds body capacity');
  }
  const { litContext, priors } = language.model;
  const state = new Uint16Array(priors);

  let range = 0xFF_FF_FF_FF >>> 0;
  let code = 0;
  let bpos = pos;
  // The first byte is always 0 (encoder cacheSize starts at 1 with cache 0); rejecting
  // other values keeps frames canonical.
  if (bpos >= end || body[bpos++] !== 0) throw new TokzipDecodeError('invalid range-coder header');
  for (let i = 0; i < 4; i++) {
    if (bpos >= end) throw new TokzipDecodeError('truncated range-coded body');
    code = ((code << 8) | body[bpos++]!) >>> 0;
  }

  const historyLength = history?.length ?? 0;
  const target = historyLength + outputSize;
  const out = allocateDecodeBuffer(target);
  if (history) out.set(history);
  const { dictionary } = language;
  const tracker = fenced ? new FenceTracker(language.id) : undefined;
  let rep0 = INITIAL_REPS[0]!;
  let rep1 = INITIAL_REPS[1]!;
  let rep2 = INITIAL_REPS[2]!;
  let rep3 = INITIAL_REPS[3]!;
  let produced = historyLength;
  let prevKind = TOKEN_KIND_LIT;
  let prevByte = historyLength > 0 ? history![historyLength - 1]! : 0;

  // Working registers for the repeated inlined bit decodes below. Each `index = …` followed
  // by the probability/bound/renormalization block is one textual copy of
  // `RangeDecoder.decodeBit` — see the JSDoc above for why it is expanded by hand.
  let index = 0;
  let prob = 0;
  let bound = 0;
  let bit = 0;

  while (produced < target) {
    // isMatch bit: 0 = literal, 1 = match.
    index = MODEL_IS_MATCH + prevKind;
    prob = state[index]!;
    bound = (range >>> PROB_BITS) * prob;
    // Unsigned compare: both sides are in [0, 2^32).
    if (code >>> 0 < bound) {
      range = bound >>> 0;
      state[index] = prob + ((PROB_SCALE - prob) >> ADAPT_SHIFT);
      bit = 0;
    } else {
      code = (code - bound) >>> 0;
      range = (range - bound) >>> 0;
      state[index] = prob - (prob >> ADAPT_SHIFT);
      bit = 1;
    }
    // One renormalization step always suffices: probabilities are clamped to
    // [PROB_MIN, PROB_MAX], so a decode keeps range ≥ (TOP >> PROB_BITS) × PROB_MIN > 2^16.
    if (range < TOP) {
      if (bpos >= end) throw new TokzipDecodeError('truncated range-coded body');
      code = ((code << 8) | body[bpos++]!) >>> 0;
      range = (range << 8) >>> 0;
    }

    if (bit === 0) {
      const base = MODEL_LITERAL + litContext[prevByte]! * LITERAL_BLOCK_SIZE;
      let node = 1;
      if (prevKind === TOKEN_KIND_LIT || rep0 > produced) {
        // Plain literal: an 8-bit tree walk — the single hottest spot in the whole decoder.
        for (let i = 0; i < 8; i++) {
          index = base + node - 1;
          prob = state[index]!;
          bound = (range >>> PROB_BITS) * prob;
          if (code >>> 0 < bound) {
            range = bound >>> 0;
            state[index] = prob + ((PROB_SCALE - prob) >> ADAPT_SHIFT);
            node = node << 1;
          } else {
            code = (code - bound) >>> 0;
            range = (range - bound) >>> 0;
            state[index] = prob - (prob >> ADAPT_SHIFT);
            node = (node << 1) | 1;
          }
          if (range < TOP) {
            if (bpos >= end) throw new TokzipDecodeError('truncated range-coded body');
            code = ((code << 8) | body[bpos++]!) >>> 0;
            range = (range << 8) >>> 0;
          }
        }
      } else {
        // Matched literal (mirrors the encoder): the byte at rep0 distance predicts each
        // bit until the first mismatch.
        const matchByte = out[produced - rep0]!;
        let matched = 1;
        for (let shift = 7; shift >= 0; shift--) {
          const matchBit = (matchByte >>> shift) & 1;
          index = matched !== 0 ? base + (((1 + matchBit) << 8) | node) - 1 : base + node - 1;
          prob = state[index]!;
          bound = (range >>> PROB_BITS) * prob;
          if (code >>> 0 < bound) {
            range = bound >>> 0;
            state[index] = prob + ((PROB_SCALE - prob) >> ADAPT_SHIFT);
            bit = 0;
          } else {
            code = (code - bound) >>> 0;
            range = (range - bound) >>> 0;
            state[index] = prob - (prob >> ADAPT_SHIFT);
            bit = 1;
          }
          if (range < TOP) {
            if (bpos >= end) throw new TokzipDecodeError('truncated range-coded body');
            code = ((code << 8) | body[bpos++]!) >>> 0;
            range = (range << 8) >>> 0;
          }
          if (matchBit !== bit) matched = 0;
          node = node * 2 + bit;
        }
      }
      const byte = node - 256;
      out[produced++] = byte;
      prevByte = byte;
      prevKind = TOKEN_KIND_LIT;
      continue;
    }

    // Match: resolve the token kind (rep / explicit history / dictionary), then read the
    // length slot and — for explicit matches — the offset slot through the shared inlined
    // slot readers below.
    index = MODEL_IS_REP + prevKind;
    prob = state[index]!;
    bound = (range >>> PROB_BITS) * prob;
    if (code >>> 0 < bound) {
      range = bound >>> 0;
      state[index] = prob + ((PROB_SCALE - prob) >> ADAPT_SHIFT);
      bit = 0;
    } else {
      code = (code - bound) >>> 0;
      range = (range - bound) >>> 0;
      state[index] = prob - (prob >> ADAPT_SHIFT);
      bit = 1;
    }
    if (range < TOP) {
      if (bpos >= end) throw new TokzipDecodeError('truncated range-coded body');
      code = ((code << 8) | body[bpos++]!) >>> 0;
      range = (range << 8) >>> 0;
    }

    let repIndex = -1;
    let lenTreeBase: number;
    let offGroup = -1;
    if (bit !== 0) {
      let node = 1;
      for (let i = 0; i < 2; i++) {
        index = MODEL_REP_TREE + prevKind * 3 + node - 1;
        prob = state[index]!;
        bound = (range >>> PROB_BITS) * prob;
        if (code >>> 0 < bound) {
          range = bound >>> 0;
          state[index] = prob + ((PROB_SCALE - prob) >> ADAPT_SHIFT);
          node = node << 1;
        } else {
          code = (code - bound) >>> 0;
          range = (range - bound) >>> 0;
          state[index] = prob - (prob >> ADAPT_SHIFT);
          node = (node << 1) | 1;
        }
        if (range < TOP) {
          if (bpos >= end) throw new TokzipDecodeError('truncated range-coded body');
          code = ((code << 8) | body[bpos++]!) >>> 0;
          range = (range << 8) >>> 0;
        }
      }
      repIndex = node - 4;
      lenTreeBase = MODEL_LEN_TREE + LEN_GROUP_REP * 63;
    } else {
      index = MODEL_IS_DICT + prevKind;
      prob = state[index]!;
      bound = (range >>> PROB_BITS) * prob;
      if (code >>> 0 < bound) {
        range = bound >>> 0;
        state[index] = prob + ((PROB_SCALE - prob) >> ADAPT_SHIFT);
        bit = 0;
      } else {
        code = (code - bound) >>> 0;
        range = (range - bound) >>> 0;
        state[index] = prob - (prob >> ADAPT_SHIFT);
        bit = 1;
      }
      if (range < TOP) {
        if (bpos >= end) throw new TokzipDecodeError('truncated range-coded body');
        code = ((code << 8) | body[bpos++]!) >>> 0;
        range = (range << 8) >>> 0;
      }
      if (bit === 0) {
        lenTreeBase = MODEL_LEN_TREE + LEN_GROUP_HISTORY * 63;
        offGroup = OFF_GROUP_HISTORY;
      } else {
        lenTreeBase = MODEL_LEN_TREE + LEN_GROUP_DICT * 63;
        offGroup = OFF_GROUP_DICT;
      }
    }

    // Length slot: a SLOT_TREE_BITS tree walk plus raw extra bits.
    let node = 1;
    for (let i = 0; i < SLOT_TREE_BITS; i++) {
      index = lenTreeBase + node - 1;
      prob = state[index]!;
      bound = (range >>> PROB_BITS) * prob;
      if (code >>> 0 < bound) {
        range = bound >>> 0;
        state[index] = prob + ((PROB_SCALE - prob) >> ADAPT_SHIFT);
        node = node << 1;
      } else {
        code = (code - bound) >>> 0;
        range = (range - bound) >>> 0;
        state[index] = prob - (prob >> ADAPT_SHIFT);
        node = (node << 1) | 1;
      }
      if (range < TOP) {
        if (bpos >= end) throw new TokzipDecodeError('truncated range-coded body');
        code = ((code << 8) | body[bpos++]!) >>> 0;
        range = (range << 8) >>> 0;
      }
    }
    const lenSlot = node - (1 << SLOT_TREE_BITS);
    if (lenSlot >= LENGTH_SLOT_COUNT) throw new TokzipDecodeError('invalid symbol');
    let extra = extraBitsOf(lenSlot);
    let extraValue = 0;
    for (let i = 0; i < extra; i++) {
      range = range >>> 1;
      if (code >>> 0 >= range) {
        code = (code - range) >>> 0;
        extraValue = extraValue * 2 + 1;
      } else {
        extraValue = extraValue * 2;
      }
      if (range < TOP) {
        if (bpos >= end) throw new TokzipDecodeError('truncated range-coded body');
        code = ((code << 8) | body[bpos++]!) >>> 0;
        range = (range << 8) >>> 0;
      }
    }
    const length = valueOfSlot(lenSlot, extraValue) + MIN_LEN_REP;

    // Offset slot (explicit history/dictionary matches only), same shape as the length slot.
    let offValue = 0;
    if (offGroup >= 0) {
      const offTreeBase = MODEL_OFF_TREE + (offGroup * 4 + offLenBucketOf(length)) * 63;
      node = 1;
      for (let i = 0; i < SLOT_TREE_BITS; i++) {
        index = offTreeBase + node - 1;
        prob = state[index]!;
        bound = (range >>> PROB_BITS) * prob;
        if (code >>> 0 < bound) {
          range = bound >>> 0;
          state[index] = prob + ((PROB_SCALE - prob) >> ADAPT_SHIFT);
          node = node << 1;
        } else {
          code = (code - bound) >>> 0;
          range = (range - bound) >>> 0;
          state[index] = prob - (prob >> ADAPT_SHIFT);
          node = (node << 1) | 1;
        }
        if (range < TOP) {
          if (bpos >= end) throw new TokzipDecodeError('truncated range-coded body');
          code = ((code << 8) | body[bpos++]!) >>> 0;
          range = (range << 8) >>> 0;
        }
      }
      const offSlot = node - (1 << SLOT_TREE_BITS);
      if (offSlot >= OFFSET_SLOT_COUNT) throw new TokzipDecodeError('invalid symbol');
      extra = extraBitsOf(offSlot);
      extraValue = 0;
      for (let i = 0; i < extra; i++) {
        range = range >>> 1;
        if (code >>> 0 >= range) {
          code = (code - range) >>> 0;
          extraValue = extraValue * 2 + 1;
        } else {
          extraValue = extraValue * 2;
        }
        if (range < TOP) {
          if (bpos >= end) throw new TokzipDecodeError('truncated range-coded body');
          code = ((code << 8) | body[bpos++]!) >>> 0;
          range = (range << 8) >>> 0;
        }
      }
      offValue = valueOfSlot(offSlot, extraValue);
    }

    if (produced + length > target) throw new TokzipDecodeError('declared size exceeded');
    if (repIndex >= 0) {
      let dist: number;
      if (repIndex === 0) dist = rep0;
      else if (repIndex === 1) {
        dist = rep1;
        rep1 = rep0;
        rep0 = dist;
      } else if (repIndex === 2) {
        dist = rep2;
        rep2 = rep1;
        rep1 = rep0;
        rep0 = dist;
      } else {
        dist = rep3;
        rep3 = rep2;
        rep2 = rep1;
        rep1 = rep0;
        rep0 = dist;
      }
      copyHistory(out, produced, dist, length);
      prevKind = TOKEN_KIND_REP0 + repIndex;
    } else if (offGroup === OFF_GROUP_HISTORY) {
      const dist = offValue + 1;
      rep3 = rep2;
      rep2 = rep1;
      rep1 = rep0;
      rep0 = dist;
      copyHistory(out, produced, dist, length);
      prevKind = TOKEN_KIND_HISTORY;
    } else {
      const start = offValue;
      if (start + length <= dictionary.length) {
        if (length <= 16) {
          // Avoid the per-match subarray allocation for the dominant short matches.
          for (let i = 0; i < length; i++) out[produced + i] = dictionary[start + i]!;
        } else {
          out.set(dictionary.subarray(start, start + length), produced);
        }
      } else if (tracker) {
        copyExtendedDictMatch(out, produced, start, length, language, tracker);
      } else {
        throw new TokzipDecodeError('dictionary match out of bounds');
      }
      prevKind = TOKEN_KIND_DICT;
    }
    produced += length;
    prevByte = out[produced - 1]!;
  }
  // Canonical framing: the decoder must have consumed the whole payload up to the channel's
  // zero padding, and the padding itself must be zero.
  const consumed = bpos;
  if (end - consumed > padAllowance) throw new TokzipDecodeError('trailing characters after payload');
  for (let i = consumed; i < end; i++) {
    if (body[i] !== 0) throw new TokzipDecodeError('non-zero padding bits');
  }
  // slice, not subarray: a view would keep the whole history+output allocation alive for as
  // long as the caller retains the chunk, multiplying resident memory per decoded block.
  return historyLength > 0 ? out.slice(historyLength) : out;
}

function copyHistory(buffer: Uint8Array, at: number, dist: number, length: number): void {
  if (dist < 1 || dist > at) throw new TokzipDecodeError('history match out of bounds');
  const from = at - dist;
  if (length <= 16) {
    // Short matches dominate; a plain forward byte loop beats copyWithin's call overhead
    // and handles overlap (dist < length) correctly by construction.
    for (let i = 0; i < length; i++) buffer[at + i] = buffer[from + i]!;
  } else if (dist >= length) {
    buffer.copyWithin(at, from, from + length);
  } else {
    // Overlapping long match: extend the dist-periodic pattern by doubling. Each step's
    // source [from, from + n) ends at or before the destination `at + copied`, so
    // copyWithin's memmove semantics match the required forward-copy semantics.
    let copied = 0;
    while (copied < length) {
      const n = Math.min(dist + copied, length - copied);
      buffer.copyWithin(at + copied, from, from + n);
      copied += n;
    }
  }
}
