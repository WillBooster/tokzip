import { TokzipDecodeError } from './errors.ts';
import { asciiCodeAt } from './radix64.ts';

/**
 * Radix-85 alphabet used by `small` mode: printable ASCII (0x21–0x7E) excluding the nine
 * unsafe characters `"` `\` `` ` `` `$` `<` `>` `&` `'` `%`, leaving exactly 85 JSON- and
 * template-literal-safe characters.
 */
export const RADIX85_ALPHABET = '!#()*+,-./0123456789:;=?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[]^_abcdefghijklmnopqrstuvwxyz{|}~';

const RADIX85_CODES = new Uint8Array(85);
for (let i = 0; i < 85; i++) RADIX85_CODES[i] = RADIX85_ALPHABET.codePointAt(i)!;

const RADIX85_VALUES = new Int8Array(128).fill(-1);
for (let i = 0; i < 85; i++) RADIX85_VALUES[RADIX85_ALPHABET.codePointAt(i)!] = i;

const asciiDecoder = new TextDecoder();

/**
 * MSB-first bit writer whose flush path is the fused Z85-style block emitter:
 * each full 32-bit word becomes 5 radix-85 chars (25% text tax), single pass, no
 * intermediate binary buffer. The final partial word is zero-padded.
 */
export class BitWriter {
  private words: Uint32Array = new Uint32Array(256);
  private wordCount = 0;
  private acc = 0;
  private accBits = 0;
  /** Total number of bits written so far. */
  bitLength = 0;

  private pushWord(word: number): void {
    if (this.wordCount >= this.words.length) {
      const grown = new Uint32Array(this.words.length * 2);
      grown.set(this.words);
      this.words = grown;
    }
    this.words[this.wordCount++] = word;
  }

  /** Writes the low `count` bits of `value` (0 ≤ count ≤ 24), MSB-first. Callers must pass `value < 2**count`; high bits are not masked. */
  writeBits(value: number, count: number): void {
    this.bitLength += count;
    let bits = this.accBits + count;
    if (bits < 32) {
      // oxlint-disable-next-line unicorn/prefer-math-trunc -- >>> 0 coerces to uint32, Math.trunc does not
      this.acc = ((this.acc << count) | value) >>> 0;
      this.accBits = bits;
      return;
    }
    bits -= 32;
    // oxlint-disable-next-line unicorn/prefer-math-trunc -- >>> 0 coerces to uint32, Math.trunc does not
    this.pushWord(((this.acc << (count - bits)) | (value >>> bits)) >>> 0);
    this.acc = bits === 0 ? 0 : value & ((1 << bits) - 1);
    this.accBits = bits;
  }

  /** Writes an unsigned value of arbitrary magnitude as 8-bit groups: 7 payload bits + continue bit. */
  writeVarint(value: number): void {
    while (value > 127) {
      this.writeBits((value & 127) | 128, 8);
      value = Math.floor(value / 128);
    }
    this.writeBits(value, 8);
  }

  /** Flushes to radix-85 text: zero-pads to a 32-bit boundary, then 5 chars per word. */
  toText(): string {
    const wordCount = this.wordCount + (this.accBits > 0 ? 1 : 0);
    const codes = new Uint8Array(wordCount * 5);
    let at = 0;
    for (let w = 0; w < wordCount; w++) {
      // oxlint-disable-next-line unicorn/prefer-math-trunc -- >>> 0 coerces to uint32, Math.trunc does not
      let word = w < this.wordCount ? this.words[w]! : (this.acc << (32 - this.accBits)) >>> 0;
      // Successive division emits the 5 digits least-significant first.
      codes[at + 4] = RADIX85_CODES[word % 85]!;
      word = Math.trunc(word / 85);
      codes[at + 3] = RADIX85_CODES[word % 85]!;
      word = Math.trunc(word / 85);
      codes[at + 2] = RADIX85_CODES[word % 85]!;
      word = Math.trunc(word / 85);
      codes[at + 1] = RADIX85_CODES[word % 85]!;
      codes[at] = RADIX85_CODES[Math.trunc(word / 85)]!;
      at += 5;
    }
    return asciiDecoder.decode(codes);
  }
}

/** Decodes a radix-85 payload back to its 32-bit words. */
export function decodeRadix85(data: string, start: number, end: number): Uint32Array {
  const length = end - start;
  if (length % 5 !== 0) throw new TokzipDecodeError('radix-85 body length is not a multiple of 5');
  const values = RADIX85_VALUES;
  const words = new Uint32Array(length / 5);
  for (let w = 0, i = start; i < end; w++, i += 5) {
    const c0 = asciiCodeAt(data, i);
    const c1 = asciiCodeAt(data, i + 1);
    const c2 = asciiCodeAt(data, i + 2);
    const c3 = asciiCodeAt(data, i + 3);
    const c4 = asciiCodeAt(data, i + 4);
    if ((c0 | c1 | c2 | c3 | c4) >= 128) throwNonAlphabet(data, i);
    const v0 = values[c0]!;
    const v1 = values[c1]!;
    const v2 = values[c2]!;
    const v3 = values[c3]!;
    const v4 = values[c4]!;
    if ((v0 | v1 | v2 | v3 | v4) < 0) throwNonAlphabet(data, i);
    const word = (((v0 * 85 + v1) * 85 + v2) * 85 + v3) * 85 + v4;
    if (word > 0xFF_FF_FF_FF) throw new TokzipDecodeError('radix-85 group out of range');
    words[w] = word;
  }
  return words;
}

function throwNonAlphabet(data: string, groupStart: number): never {
  for (let d = 0; d < 5; d++) {
    const code = asciiCodeAt(data, groupStart + d);
    if (code >= 128 || RADIX85_VALUES[code]! < 0) {
      throw new TokzipDecodeError(`non-alphabet character at position ${groupStart + d}`);
    }
  }
  throw new TokzipDecodeError(`non-alphabet character at position ${groupStart}`);
}

/** MSB-first bit reader over decoded words; independent readers over the same words act as stream cursors. */
export class BitReader {
  private readonly words: Uint32Array;
  private pos: number;
  /** Total bit capacity (multiple of 32; includes final zero padding). */
  readonly bitCapacity: number;

  constructor(words: Uint32Array, bitPos = 0) {
    this.words = words;
    this.bitCapacity = words.length * 32;
    if (bitPos < 0 || bitPos > this.bitCapacity) throw new TokzipDecodeError('bit position out of range');
    this.pos = bitPos;
  }

  get bitPosition(): number {
    return this.pos;
  }

  /** Reads `count` bits (0 ≤ count ≤ 24), MSB-first. */
  readBits(count: number): number {
    const pos = this.pos;
    if (pos + count > this.bitCapacity) throw new TokzipDecodeError('truncated bitstream');
    this.pos = pos + count;
    const offset = pos & 31;
    const spill = offset + count - 32;
    // A 32-bit shift is a no-op in JS (shift counts are mod 32); count 0 must yield 0.
    const first = count === 0 ? 0 : (this.words[pos >>> 5]! << offset) >>> (32 - count);
    if (spill <= 0) return first;
    // `first` already has zeros in its low `spill` bits (they were shifted out of word 1).
    return first + (this.words[(pos >>> 5) + 1]! >>> (32 - spill));
  }

  /** Peeks `count` bits (0 ≤ count ≤ 24) without consuming, zero-padded beyond capacity. */
  peekBits(count: number): number {
    const saved = this.pos;
    const available = Math.min(count, this.bitCapacity - this.pos);
    const value = available > 0 ? this.readBits(available) : 0;
    this.pos = saved;
    return value * 2 ** (count - available);
  }

  /** Consumes `count` bits previously peeked. */
  advance(count: number): void {
    if (this.pos + count > this.bitCapacity) throw new TokzipDecodeError('truncated bitstream');
    this.pos += count;
  }

  /** Reads a varint written by {@link BitWriter.writeVarint}. */
  readVarint(): number {
    let value = 0;
    let factor = 1;
    for (let i = 0; i < 8; i++) {
      const group = this.readBits(8);
      value += (group & 127) * factor;
      if ((group & 128) === 0) return value;
      factor *= 128;
    }
    throw new TokzipDecodeError('varint exceeds bound');
  }
}
