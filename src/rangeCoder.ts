// oxlint-disable unicorn/prefer-math-trunc -- `>>> 0` coerces to unsigned 32-bit throughout; Math.trunc would keep the sign
import { TokzipDecodeError } from './errors.ts';

/**
 * Adaptive binary range coder (LZMA-style): 32-bit range, 11-bit probabilities with
 * shift-based adaptation, byte-wise renormalization with carry propagation. Probabilities
 * start from trained per-language priors (see train.ts), so short documents behave like the
 * static model while long documents adapt to their own statistics.
 */

/** Probability scale: P(bit = 0) in [1, PROB_SCALE - 1] out of PROB_SCALE. */
export const PROB_BITS = 11;
export const PROB_SCALE = 1 << PROB_BITS;
/** Adaptation rate: the update moves 1/2^ADAPT_SHIFT of the distance to the extreme. */
const ADAPT_SHIFT = 5;
/** Priors are clamped inside this margin so both symbols always stay codable. */
export const PROB_MIN = 31;
export const PROB_MAX = PROB_SCALE - PROB_MIN;

const TOP = 1 << 24;

/** Exact-enough bit prices for the parser: PRICE_TABLE[p >> 4] ≈ -log2(p / PROB_SCALE). */
const PRICE_TABLE = new Float64Array(PROB_SCALE >> 4);
for (let i = 0; i < PRICE_TABLE.length; i++) {
  const p = (i << 4) + 8;
  PRICE_TABLE[i] = -Math.log2(p / PROB_SCALE);
}

/** Price in bits of coding `bit` with P(0) = prob / PROB_SCALE. */
export function bitPrice(prob: number, bit: number): number {
  return PRICE_TABLE[(bit === 0 ? prob : PROB_SCALE - prob) >> 4]!;
}

export class RangeEncoder {
  private low = 0; // Up to 33 bits; stays exact in a double.
  private range = 0xFF_FF_FF_FF >>> 0;
  private cache = 0;
  private cacheSize = 1;
  private out: Uint8Array = new Uint8Array(4096);
  private length = 0;

  /** Encodes `bit` with the adaptive probability at probs[index] and updates it. */
  encodeBit(probs: Uint16Array, index: number, bit: number): void {
    const prob = probs[index]!;
    // range is kept as an unsigned 32-bit value via >>> 0 after every update.
    const bound = (this.range >>> PROB_BITS) * prob;
    if (bit === 0) {
      this.range = bound >>> 0;
      probs[index] = prob + ((PROB_SCALE - prob) >> ADAPT_SHIFT);
    } else {
      this.low += bound;
      this.range = (this.range - bound) >>> 0;
      probs[index] = prob - (prob >> ADAPT_SHIFT);
    }
    while (this.range < TOP) {
      this.shiftLow();
      this.range = (this.range * 256) >>> 0;
    }
  }

  /** Encodes `count` raw bits of `value` (MSB first) at fixed probability 1/2. */
  encodeDirect(value: number, count: number): void {
    for (let shift = count - 1; shift >= 0; shift--) {
      this.range = (this.range >>> 1) >>> 0;
      if (((value >>> shift) & 1) !== 0) this.low += this.range;
      while (this.range < TOP) {
        this.shiftLow();
        this.range = (this.range * 256) >>> 0;
      }
    }
  }

  private shiftLow(): void {
    if (this.low < 0xFF_00_00_00 || this.low > 0xFF_FF_FF_FF) {
      const carry = this.low > 0xFF_FF_FF_FF ? 1 : 0;
      let temp = this.cache;
      do {
        this.push((temp + carry) & 0xFF);
        temp = 0xFF;
      } while (--this.cacheSize !== 0);
      this.cache = (this.low >>> 24) & 0xFF;
    }
    this.cacheSize++;
    this.low = (this.low % 0x1_00_00_00) * 256;
  }

  /** Flushes the pending bytes; the encoder must not be used afterwards. */
  finish(): Uint8Array {
    for (let i = 0; i < 5; i++) this.shiftLow();
    return this.out.subarray(0, this.length);
  }

  private push(byte: number): void {
    if (this.length === this.out.length) {
      const next = new Uint8Array(this.out.length * 2);
      next.set(this.out);
      this.out = next;
    }
    this.out[this.length++] = byte;
  }
}

export class RangeDecoder {
  private readonly data: Uint8Array;
  private readonly end: number;
  private pos: number;
  private range = 0xFF_FF_FF_FF >>> 0;
  private code = 0;

  constructor(data: Uint8Array, pos: number, end: number) {
    this.data = data;
    this.end = end;
    this.pos = pos;
    // The first byte is always 0 (encoder cacheSize starts at 1 with cache 0); rejecting
    // other values keeps frames canonical.
    if (this.readByte() !== 0) throw new TokzipDecodeError('invalid range-coder header');
    for (let i = 0; i < 4; i++) this.code = (this.code * 256 + this.readByte()) >>> 0;
  }

  /** Decodes one bit with the adaptive probability at probs[index] and updates it. */
  decodeBit(probs: Uint16Array, index: number): number {
    const prob = probs[index]!;
    const bound = (this.range >>> PROB_BITS) * prob;
    let bit: number;
    // Unsigned compare: both sides are in [0, 2^32).
    if (this.code >>> 0 < bound) {
      this.range = bound >>> 0;
      probs[index] = prob + ((PROB_SCALE - prob) >> ADAPT_SHIFT);
      bit = 0;
    } else {
      this.code = (this.code - bound) >>> 0;
      this.range = (this.range - bound) >>> 0;
      probs[index] = prob - (prob >> ADAPT_SHIFT);
      bit = 1;
    }
    while (this.range < TOP) {
      this.code = (this.code * 256 + this.readByte()) >>> 0;
      this.range = (this.range * 256) >>> 0;
    }
    return bit;
  }

  /** Decodes `count` raw bits (MSB first) at fixed probability 1/2. */
  decodeDirect(count: number): number {
    let value = 0;
    for (let i = 0; i < count; i++) {
      this.range = (this.range >>> 1) >>> 0;
      let bit = 0;
      if (this.code >>> 0 >= this.range) {
        this.code = (this.code - this.range) >>> 0;
        bit = 1;
      }
      value = value * 2 + bit;
      while (this.range < TOP) {
        this.code = (this.code * 256 + this.readByte()) >>> 0;
        this.range = (this.range * 256) >>> 0;
      }
    }
    return value;
  }

  /** Bytes consumed so far (for exact-length canonicality checks). */
  get position(): number {
    return this.pos;
  }

  private readByte(): number {
    if (this.pos >= this.end) throw new TokzipDecodeError('truncated range-coded body');
    return this.data[this.pos++]!;
  }
}

/**
 * Encodes `symbol` (< 2^bits) through a bit tree rooted at probs[base]: node indices follow
 * the canonical 1-based heap layout, so a tree of depth `bits` uses 2^bits - 1 slots.
 */
export function encodeTree(
  encoder: RangeEncoder,
  probs: Uint16Array,
  base: number,
  bits: number,
  symbol: number
): void {
  let node = 1;
  for (let shift = bits - 1; shift >= 0; shift--) {
    const bit = (symbol >>> shift) & 1;
    encoder.encodeBit(probs, base + node - 1, bit);
    node = node * 2 + bit;
  }
}

export function decodeTree(decoder: RangeDecoder, probs: Uint16Array, base: number, bits: number): number {
  let node = 1;
  for (let i = 0; i < bits; i++) node = node * 2 + decoder.decodeBit(probs, base + node - 1);
  return node - (1 << bits);
}

/** Total price in bits of coding `symbol` through the tree at probs[base] (no adaptation). */
export function treePrice(probs: Uint16Array, base: number, bits: number, symbol: number): number {
  let node = 1;
  let price = 0;
  for (let shift = bits - 1; shift >= 0; shift--) {
    const bit = (symbol >>> shift) & 1;
    price += bitPrice(probs[base + node - 1]!, bit);
    node = node * 2 + bit;
  }
  return price;
}
