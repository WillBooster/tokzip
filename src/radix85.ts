import { TokzipDecodeError } from './errors.ts';

/**
 * Radix-85 alphabet used by `small` mode: printable ASCII (0x21–0x7E) excluding the nine
 * unsafe characters `"` `\` `` ` `` `$` `<` `>` `&` `'` `%`, leaving exactly 85 JSON- and
 * template-literal-safe characters.
 */
export const RADIX85_ALPHABET = '!#()*+,-./0123456789:;=?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[]^_abcdefghijklmnopqrstuvwxyz{|}~';

const RADIX85_VALUES = new Int8Array(128).fill(-1);
for (let i = 0; i < 85; i++) RADIX85_VALUES[RADIX85_ALPHABET.codePointAt(i)!] = i;

const POW85 = [1, 85, 85 * 85, 85 * 85 * 85, 85 * 85 * 85 * 85];

/**
 * MSB-first bit writer whose flush path is the fused Z85-style block emitter:
 * each full 32-bit word becomes 5 radix-85 chars (25% text tax), single pass, no
 * intermediate binary buffer. The final partial word is zero-padded.
 */
export class BitWriter {
  private words: number[] = [];
  private acc = 0;
  private accBits = 0;
  /** Total number of bits written so far. */
  bitLength = 0;

  /** Writes the low `count` bits of `value` (0 ≤ count ≤ 24), MSB-first. */
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
    this.words.push(((this.acc << (count - bits)) | (value >>> bits)) >>> 0);
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
    const words = [...this.words];
    // oxlint-disable-next-line unicorn/prefer-math-trunc -- >>> 0 coerces to uint32, Math.trunc does not
    if (this.accBits > 0) words.push((this.acc << (32 - this.accBits)) >>> 0);
    const chars: string[] = [];
    for (const word of words) {
      for (let d = 4; d >= 0; d--) chars.push(RADIX85_ALPHABET[Math.floor(word / POW85[d]!) % 85]!);
    }
    return chars.join('');
  }
}

/** Decodes a radix-85 payload back to its 32-bit words. */
export function decodeRadix85(data: string, start: number, end: number): Uint32Array {
  const length = end - start;
  if (length % 5 !== 0) throw new TokzipDecodeError('radix-85 body length is not a multiple of 5');
  const words = new Uint32Array(length / 5);
  for (let w = 0, i = start; i < end; w++, i += 5) {
    let word = 0;
    for (let d = 0; d < 5; d++) {
      const code = data.codePointAt(i + d)!;
      const value = code < 128 ? RADIX85_VALUES[code]! : -1;
      if (value < 0) throw new TokzipDecodeError(`non-alphabet character at position ${i + d}`);
      word = word * 85 + value;
    }
    if (word > 0xFF_FF_FF_FF) throw new TokzipDecodeError('radix-85 group out of range');
    words[w] = word;
  }
  return words;
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
    if (this.pos + count > this.bitCapacity) throw new TokzipDecodeError('truncated bitstream');
    let result = 0;
    let remaining = count;
    while (remaining > 0) {
      const wordIndex = this.pos >>> 5;
      const offset = this.pos & 31;
      const take = Math.min(remaining, 32 - offset);
      const chunk = (this.words[wordIndex]! << offset) >>> (32 - take);
      result = result * 2 ** take + chunk;
      this.pos += take;
      remaining -= take;
    }
    return result;
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
