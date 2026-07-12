import { TokzipDecodeError } from './errors.ts';

/**
 * Radix-64 alphabet (base64url set) used by the container header and `fast` mode.
 * Every field is a whole number of 6-bit characters — encode/decode is table lookup only.
 */
export const RADIX64_ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_';

/** Char code of each radix-64 value — emission writes bytes, not strings. */
export const RADIX64_CODES = new Uint8Array(64);
for (let i = 0; i < 64; i++) RADIX64_CODES[i] = RADIX64_ALPHABET.codePointAt(i)!;

/**
 * UTF-16 unit read for decode hot loops. All tokzip alphabets are ASCII, so any surrogate half
 * is ≥ 128 and rejected by the caller — `charCodeAt` skips `codePointAt`'s pair handling.
 */
// oxlint-disable-next-line unicorn/prefer-code-point -- hot path; ASCII-only semantics wanted
export const asciiCodeAt = (data: string, pos: number): number => data.charCodeAt(pos);

/** Maps char code → 6-bit value, or -1 for characters outside the alphabet. */
export const RADIX64_VALUES = new Int8Array(128).fill(-1);
for (let i = 0; i < 64; i++) RADIX64_VALUES[RADIX64_ALPHABET.codePointAt(i)!] = i;

const asciiDecoder = new TextDecoder();

/**
 * Growable byte buffer of ASCII char codes with a single-pass string conversion — emission
 * writes one byte per output character instead of paying string-array push/join costs.
 */
export class TextSink {
  buffer: Uint8Array;
  length = 0;

  constructor(capacity = 64) {
    this.buffer = new Uint8Array(capacity);
  }

  /** Ensures room for `extra` more bytes and returns the backing buffer. */
  reserve(extra: number): Uint8Array {
    const needed = this.length + extra;
    if (needed > this.buffer.length) {
      const grown = new Uint8Array(Math.max(needed, this.buffer.length * 2));
      grown.set(this.buffer.subarray(0, this.length));
      this.buffer = grown;
    }
    return this.buffer;
  }

  push(code: number): void {
    if (this.length >= this.buffer.length) this.reserve(1);
    this.buffer[this.length++] = code;
  }

  toString(): string {
    return asciiDecoder.decode(this.buffer.subarray(0, this.length));
  }
}

/** Reads one radix-64 char at `pos`, throwing a structural error on non-alphabet chars or truncation. */
export function readRadix64(data: string, pos: number): number {
  if (pos >= data.length) throw new TokzipDecodeError('truncated payload');
  const code = asciiCodeAt(data, pos);
  const value = code < 128 ? RADIX64_VALUES[code]! : -1;
  if (value < 0) throw new TokzipDecodeError(`non-alphabet character at position ${pos}`);
  return value;
}

const VARINT_MAX_CHARS = 7; // 7 × 5 payload bits = 35 bits, bounding declared sizes below 2^35.

/**
 * Appends a canonical radix-64 varint: little-endian 5-bit groups, bit 5 set on all but the
 * last char. Canonical means minimal length (no redundant trailing zero groups).
 */
export function pushVarint64(out: TextSink, value: number): void {
  if (value < 0 || !Number.isSafeInteger(value)) throw new RangeError(`invalid varint value: ${value}`);
  do {
    const group = value % 32;
    value = Math.floor(value / 32);
    out.push(RADIX64_CODES[value > 0 ? group | 32 : group]!);
  } while (value > 0);
}

/** Number of chars {@link pushVarint64} emits for `value`. */
export function varint64Length(value: number): number {
  let length = 1;
  for (let rest = Math.floor(value / 32); rest > 0; rest = Math.floor(rest / 32)) length++;
  return length;
}

/** Reads a canonical radix-64 varint; returns the value and the position after it. */
export function readVarint64(data: string, pos: number): { value: number; pos: number } {
  let value = 0;
  let shift = 1;
  for (let i = 0; i < VARINT_MAX_CHARS; i++) {
    const chunk = readRadix64(data, pos++);
    value += (chunk & 31) * shift;
    if ((chunk & 32) === 0) {
      // Canonical form: a multi-char varint must not end in a zero group.
      if (i > 0 && (chunk & 31) === 0) throw new TokzipDecodeError('non-canonical varint');
      return { value, pos };
    }
    shift *= 32;
  }
  throw new TokzipDecodeError('varint exceeds bound');
}

/** Number of chars needed to bit-pack `byteLength` raw bytes (3 bytes → 4 chars, pinned tail rule). */
export function packedRawLength(byteLength: number): number {
  const groups = Math.floor(byteLength / 3);
  const tail = byteLength % 3; // 1 trailing byte → 2 chars, 2 → 3 chars.
  return groups * 4 + (tail === 0 ? 0 : tail + 1);
}

/**
 * Bit-packs raw bytes at the normative fixed alignment: 3 bytes → 4 chars, big-endian bit order;
 * a 1-byte tail emits 2 chars and a 2-byte tail emits 3 chars, padding bits zero.
 */
export function pushPackedRaw(out: TextSink, bytes: Uint8Array, start: number, end: number): void {
  const buffer = out.reserve(packedRawLength(end - start));
  let at = out.length;
  let i = start;
  for (; i + 3 <= end; i += 3) {
    const bits = (bytes[i]! << 16) | (bytes[i + 1]! << 8) | bytes[i + 2]!;
    buffer[at] = RADIX64_CODES[(bits >>> 18) & 63]!;
    buffer[at + 1] = RADIX64_CODES[(bits >>> 12) & 63]!;
    buffer[at + 2] = RADIX64_CODES[(bits >>> 6) & 63]!;
    buffer[at + 3] = RADIX64_CODES[bits & 63]!;
    at += 4;
  }
  const tail = end - i;
  if (tail === 1) {
    const bits = bytes[i]! << 4;
    buffer[at] = RADIX64_CODES[(bits >>> 6) & 63]!;
    buffer[at + 1] = RADIX64_CODES[bits & 63]!;
    at += 2;
  } else if (tail === 2) {
    const bits = (bytes[i]! << 10) | (bytes[i + 1]! << 2);
    buffer[at] = RADIX64_CODES[(bits >>> 12) & 63]!;
    buffer[at + 1] = RADIX64_CODES[(bits >>> 6) & 63]!;
    buffer[at + 2] = RADIX64_CODES[bits & 63]!;
    at += 3;
  }
  out.length = at;
}

/**
 * Decodes {@link pushPackedRaw} output into `target` at `offset`; returns the position after the
 * consumed chars. Padding bits are ignored on decode per the pinned tail rule.
 */
export function readPackedRaw(
  data: string,
  pos: number,
  target: Uint8Array,
  offset: number,
  byteLength: number
): number {
  const values = RADIX64_VALUES;
  const end = offset + byteLength;
  let i = offset;
  for (; i + 3 <= end; i += 3) {
    // Bounds were checked by the caller against packedRawLength; validity folds into one test.
    const c0 = asciiCodeAt(data, pos);
    const c1 = asciiCodeAt(data, pos + 1);
    const c2 = asciiCodeAt(data, pos + 2);
    const c3 = asciiCodeAt(data, pos + 3);
    const v0 = (c0 | c1 | c2 | c3) < 128 ? values[c0]! : -1;
    const v1 = values[c1]!;
    const v2 = values[c2]!;
    const v3 = values[c3]!;
    if ((v0 | v1 | v2 | v3) < 0) {
      throw new TokzipDecodeError(`non-alphabet character at position ${pos}`);
    }
    const bits = (v0 << 18) | (v1 << 12) | (v2 << 6) | v3;
    pos += 4;
    target[i] = (bits >>> 16) & 255;
    target[i + 1] = (bits >>> 8) & 255;
    target[i + 2] = bits & 255;
  }
  const tail = end - i;
  if (tail === 1) {
    const bits = (readRadix64(data, pos) << 6) | readRadix64(data, pos + 1);
    pos += 2;
    target[i] = (bits >>> 4) & 255;
  } else if (tail === 2) {
    const bits = (readRadix64(data, pos) << 12) | (readRadix64(data, pos + 1) << 6) | readRadix64(data, pos + 2);
    pos += 3;
    target[i] = (bits >>> 10) & 255;
    target[i + 1] = (bits >>> 2) & 255;
  }
  return pos;
}
