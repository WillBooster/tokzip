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

/** Unpacks decoded words back into their big-endian byte payload (including any zero padding). */
export function bytesFromWords(words: Uint32Array): Uint8Array {
  const bytes = new Uint8Array(words.length * 4);
  for (let w = 0, at = 0; w < words.length; w++, at += 4) {
    const word = words[w]!;
    bytes[at] = word >>> 24;
    bytes[at + 1] = (word >>> 16) & 255;
    bytes[at + 2] = (word >>> 8) & 255;
    bytes[at + 3] = word & 255;
  }
  return bytes;
}

/** Encodes a byte payload as radix-85 text: zero-padded to a 32-bit word, 5 chars per word. */
export function radix85FromBytes(bytes: Uint8Array): string {
  const wordCount = Math.ceil(bytes.length / 4);
  const codes = new Uint8Array(wordCount * 5);
  let at = 0;
  for (let w = 0; w < wordCount; w++) {
    const i = w * 4;
    const packed = (bytes[i]! << 24) | ((bytes[i + 1] ?? 0) << 16) | ((bytes[i + 2] ?? 0) << 8) | (bytes[i + 3] ?? 0);
    // oxlint-disable-next-line unicorn/prefer-math-trunc -- >>> 0 coerces to uint32, Math.trunc does not
    let word = packed >>> 0;
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

/** Exact char count {@link radix85FromBytes} produces for `byteLength` bytes. */
export function radix85Length(byteLength: number): number {
  return Math.ceil(byteLength / 4) * 5;
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

/** Validates a radix-85 body (length, alphabet, group range) without allocating the words. */
export function scanRadix85Body(data: string, start: number, end: number): void {
  const length = end - start;
  if (length % 5 !== 0) throw new TokzipDecodeError('radix-85 body length is not a multiple of 5');
  const values = RADIX85_VALUES;
  for (let i = start; i < end; i += 5) {
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
    if ((((v0 * 85 + v1) * 85 + v2) * 85 + v3) * 85 + v4 > 0xFF_FF_FF_FF) {
      throw new TokzipDecodeError('radix-85 group out of range');
    }
  }
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
