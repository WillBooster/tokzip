import { TokzipDecodeError } from './errors.ts';

/**
 * Radix-64 alphabet (base64url set) used by the container header and `fast` mode.
 * Every field is a whole number of 6-bit characters — encode/decode is table lookup only.
 */
export const RADIX64_ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_';

// oxlint-disable-next-line no-misused-spread -- the alphabet is pure ASCII
export const RADIX64_CHARS: string[] = [...RADIX64_ALPHABET];

/** Maps char code → 6-bit value, or -1 for characters outside the alphabet. */
export const RADIX64_VALUES = new Int8Array(128).fill(-1);
for (let i = 0; i < 64; i++) RADIX64_VALUES[RADIX64_ALPHABET.codePointAt(i)!] = i;

/** Reads one radix-64 char at `pos`, throwing a structural error on non-alphabet chars or truncation. */
export function readRadix64(data: string, pos: number): number {
  if (pos >= data.length) throw new TokzipDecodeError('truncated payload');
  const code = data.codePointAt(pos)!;
  const value = code < 128 ? RADIX64_VALUES[code]! : -1;
  if (value < 0) throw new TokzipDecodeError(`non-alphabet character at position ${pos}`);
  return value;
}

const VARINT_MAX_CHARS = 7; // 7 × 5 payload bits = 35 bits, bounding declared sizes below 2^35.

/**
 * Appends a canonical radix-64 varint: little-endian 5-bit groups, bit 5 set on all but the
 * last char. Canonical means minimal length (no redundant trailing zero groups).
 */
export function pushVarint64(out: string[], value: number): void {
  if (value < 0 || !Number.isSafeInteger(value)) throw new RangeError(`invalid varint value: ${value}`);
  do {
    const group = value % 32;
    value = Math.floor(value / 32);
    out.push(RADIX64_CHARS[value > 0 ? group | 32 : group]!);
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
export function pushPackedRaw(out: string[], bytes: Uint8Array, start: number, end: number): void {
  let i = start;
  for (; i + 3 <= end; i += 3) {
    const bits = (bytes[i]! << 16) | (bytes[i + 1]! << 8) | bytes[i + 2]!;
    out.push(
      RADIX64_CHARS[(bits >>> 18) & 63]!,
      RADIX64_CHARS[(bits >>> 12) & 63]!,
      RADIX64_CHARS[(bits >>> 6) & 63]!,
      RADIX64_CHARS[bits & 63]!
    );
  }
  const tail = end - i;
  if (tail === 1) {
    const bits = bytes[i]! << 4;
    out.push(RADIX64_CHARS[(bits >>> 6) & 63]!, RADIX64_CHARS[bits & 63]!);
  } else if (tail === 2) {
    const bits = (bytes[i]! << 10) | (bytes[i + 1]! << 2);
    out.push(RADIX64_CHARS[(bits >>> 12) & 63]!, RADIX64_CHARS[(bits >>> 6) & 63]!, RADIX64_CHARS[bits & 63]!);
  }
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
  const end = offset + byteLength;
  let i = offset;
  for (; i + 3 <= end; i += 3) {
    const bits =
      (readRadix64(data, pos) << 18) |
      (readRadix64(data, pos + 1) << 12) |
      (readRadix64(data, pos + 2) << 6) |
      readRadix64(data, pos + 3);
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
