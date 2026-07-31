import { describe, expect, test } from 'bun:test';
import { frameChecksum } from '../../src/checksum.ts';
import { compress, decompress, TokzipDecodeError } from '../../src/index.ts';
import { MODE_SMALL, MODE_STORED } from '../../src/format.ts';
import { readByteVarint } from '../../src/container.ts';
import { RADIX64_ALPHABET, TextSink, pushVarint64, RADIX64_CODES } from '../../src/radix64.ts';
import { RADIX85_ALPHABET } from '../../src/radix85.ts';

/** Shipped mode from a frame's flags char (header char 2). */
function shippedMode(frame: string): number {
  return RADIX64_ALPHABET.indexOf(frame[2]!) & 3;
}

function expectDecodeError(frame: string, message: string | RegExp): void {
  expect(() => decompress(frame)).toThrow(TokzipDecodeError);
  expect(() => decompress(frame)).toThrow(message);
}

/** Builds a text frame header (magic, language 0, flags, size varint, zero CRC). */
function forgeHeader(flags: number, declaredSize: number): string {
  const out = new TextSink(24);
  out.push(RADIX64_CODES[0b11_0010]!);
  out.push(RADIX64_CODES[0]!);
  out.push(RADIX64_CODES[flags]!);
  pushVarint64(out, declaredSize);
  for (let i = 0; i < 6; i++) out.push(RADIX64_CODES[0]!);
  return out.toString();
}

describe('container vectors', () => {
  test('empty input is the exact 10-char stored frame', () => {
    const frame = compress('');
    // Header (3) + size varint (1) + CRC-32 of empty content + type byte 0x00 (6 chars).
    expect(frame).toBe('yAAAN-uASD');
    expect(decompress(frame)).toBe('');
  });

  test('empty input is the exact 8-byte binary stored frame', () => {
    const frame = compress('', { output: 'binary' });
    expect(frame).toEqual(new Uint8Array([0xB2, 0x00, 0x00, 0x00, 0x8D, 0xEF, 0x02, 0xD2]));
    expect(decompress(frame)).toBe('');
  });

  test('tiny stored frame overhead is exactly header + size varint + checksum', () => {
    const frame = compress('a');
    // 3 header chars + 1 varint char + 6 CRC chars + packed body (1 byte → 2 chars).
    expect(frame.length).toBe(3 + 1 + 6 + 2);
    expect(shippedMode(frame)).toBe(MODE_STORED);
  });

  test('stored frames carry language id 0', () => {
    const incompressible = 'qwZ7#kP9@mX2vL5';
    const frame = compress(incompressible);
    expect(shippedMode(frame)).toBe(MODE_STORED);
    expect(frame[1]).toBe('A');
  });

  test('stored frame with nonzero language id still decodes', () => {
    const frame = compress('qwZ7#kP9@mX2vL5');
    expect(shippedMode(frame)).toBe(MODE_STORED);
    const patched = frame[0]! + '9' + frame.slice(2); // Unregistered id 61.
    expect(decompress(patched)).toBe('qwZ7#kP9@mX2vL5');
  });

  test('unknown language id on a non-stored frame throws', () => {
    const frame = compress('abcabcabcabcabcabcabcabc');
    expect(shippedMode(frame)).toBe(MODE_SMALL);
    expectDecodeError(frame[0]! + '9' + frame.slice(2), /unknown language id/);
  });

  test('bad magic and unknown version', () => {
    const frame = compress('hello');
    expectDecodeError('A' + frame.slice(1), /bad magic/);
    // Same 3-bit magic, version 1 (the retired format) and version 3.
    expectDecodeError(RADIX64_ALPHABET[0b11_0001]! + frame.slice(1), /unknown version/);
    expectDecodeError(RADIX64_ALPHABET[0b11_0011]! + frame.slice(1), /unknown version/);
  });

  test('invalid mode and reserved flag bits', () => {
    const frame = compress('hello');
    expectDecodeError(frame.slice(0, 2) + 'D' + frame.slice(3), /invalid mode/); // Mode bits = 3.
    expectDecodeError(frame.slice(0, 2) + 'B' + frame.slice(3), /invalid mode/); // Mode bits = 1 (retired fast).
    expectDecodeError(frame.slice(0, 2) + 'Q' + frame.slice(3), /reserved flag bits/); // Bit 4 set.
  });

  test('non-canonical size varint', () => {
    // Varint 'gA' encodes value 0 with a redundant continuation group.
    expectDecodeError('yAAgA', /non-canonical varint/);
  });

  test('non-alphabet character', () => {
    expectDecodeError('y"AA', /non-alphabet character/);
  });

  test('truncated header and truncated payload', () => {
    const frame = compress('The quick brown fox jumps over the lazy dog.');
    expectDecodeError('', /truncated/);
    expectDecodeError(frame.slice(0, 2), /truncated/);
    expectDecodeError(frame.slice(0, -1), /truncated|declared size|stream|multiple of 5|stored/);
  });

  test('trailing characters after payload', () => {
    const source = 'function f() { return 42; } function g() { return f() + f(); }'.repeat(4);
    const frame = compress(source);
    expect(shippedMode(frame)).toBe(MODE_SMALL);
    expectDecodeError(frame + 'AAAAA', /trailing|truncated|stream|multiple of 5|invalid|size|checksum|stored/);
  });

  test('maxOutputSize is enforced before allocation', () => {
    const frame = compress('x'.repeat(100_000));
    expect(() => decompress(frame, { maxOutputSize: 1024 })).toThrow(/maxOutputSize/);
  });

  test('a declared size beyond the body capacity is rejected before allocation', () => {
    // A small frame declaring 2^34 - 1 bytes with a 4-byte body: structurally unproducible,
    // and must throw a typed error (not an engine out-of-memory RangeError) even under the
    // explicit "no cap" setting.
    const forged = forgeHeader(MODE_SMALL, 2 ** 34 - 1) + '!!!!!';
    expect(() => decompress(forged, { maxOutputSize: Number.POSITIVE_INFINITY })).toThrow(TokzipDecodeError);
    expect(() => decompress(forged, { maxOutputSize: Number.POSITIVE_INFINITY })).toThrow(/body capacity|allocatable/);
  });

  test('NaN or negative maxOutputSize is rejected instead of disabling the cap', () => {
    const frame = compress('x'.repeat(1000));
    expect(() => decompress(frame, { maxOutputSize: Number.NaN })).toThrow(RangeError);
    expect(() => decompress(frame, { maxOutputSize: -1 })).toThrow(RangeError);
    expect(decompress(frame, { maxOutputSize: Number.POSITIVE_INFINITY })).toBe('x'.repeat(1000)); // Explicit "no cap".
  });

  test('a small frame for size 0 is non-canonical and rejected', () => {
    // The canonical empty frame is the stored 'yAAAN-uASD'; a size-0 small body can never be
    // smaller than the (empty) stored body.
    expectDecodeError(forgeHeader(MODE_SMALL, 0) + '!!!!!', /non-canonical|stored/);
  });

  test('non-stored bodies at least as large as the stored body are rejected', () => {
    const frame = compress('abcabcabcabcabcabcabcabcabc');
    expect(shippedMode(frame)).toBe(MODE_SMALL);
    // Pad the small body with valid alphabet chars beyond the stored bound.
    expectDecodeError(frame + '!!!!!'.repeat(13), /non-canonical|stored/);
  });

  test('non-zero padding in a small frame is a structural error', () => {
    // Find a fixture whose range-coded body does not fill its last radix-85 word, so the
    // text frame provably carries a zero padding byte; then set that byte to 1. (The range
    // coder's own flush bytes are not canonicalized — see FORMAT.md §4.2 — so mutating
    // arbitrary trailing chars is not guaranteed to fail; the padding bytes are.)
    for (let repeats = 3; repeats < 24; repeats++) {
      const source = 'export function greet(name: string): string {\n  return name;\n}\n'.repeat(repeats);
      const binary = compress(source, { output: 'binary' });
      const text = compress(source);
      if (shippedMode(text) !== MODE_SMALL) continue;
      const bodyStart = readByteVarint(binary, 3).pos + 4;
      const rcLength = binary.length - bodyStart;
      if (rcLength % 4 === 0) continue; // No padding byte in the text frame's last word.
      let word = 0;
      for (const c of text.slice(-5)) word = word * 85 + RADIX85_ALPHABET.indexOf(c);
      word += 1; // The word's last byte is padding and canonically zero.
      let patchedGroup = '';
      for (let i = 0; i < 5; i++) {
        patchedGroup = RADIX85_ALPHABET[word % 85]! + patchedGroup;
        word = Math.floor(word / 85);
      }
      expectDecodeError(text.slice(0, -5) + patchedGroup, /non-zero padding/);
      return;
    }
    throw new Error('no small fixture with a padded last word found');
  });

  test('a corrupted body that still parses fails the content checksum', () => {
    const incompressible = 'qwZ7#kP9@mX2vL5%rT8&nB4';
    const frame = compress(incompressible);
    expect(shippedMode(frame)).toBe(MODE_STORED);
    // Flip one packed-raw body char to another alphabet char: the frame stays structurally
    // valid, so only the CRC-32 catches the corruption.
    const at = frame.length - 8;
    const original = frame[at]!;
    const flipped = original === 'A' ? 'B' : 'A';
    expectDecodeError(frame.slice(0, at) + flipped + frame.slice(at + 1), /checksum mismatch/);
  });

  test('a corrupted checksum field is a checksum mismatch', () => {
    const frame = compress('hello');
    // CRC chars sit right after the 3-char header + 1-char size varint.
    const at = 4;
    const flipped = frame[at] === 'A' ? 'B' : 'A';
    expectDecodeError(frame.slice(0, at) + flipped + frame.slice(at + 1), /checksum mismatch/);
  });

  test('binary frames verify the content checksum too', () => {
    const frame = compress('qwZ7#kP9@mX2vL5%rT8&nB4', { output: 'binary' });
    const patched = Uint8Array.from(frame);
    patched[patched.length - 1] = frame.at(-1)! ^ 0xFF;
    expect(() => decompress(patched)).toThrow(/checksum mismatch/);
  });

  test('a flipped input-type flag alone is a checksum mismatch', () => {
    // The type byte is part of the checksum domain, so retyping byte-identical content
    // (bytes frame → string frame) must fail the CRC, not silently change the return type.
    const frame = compress(new Uint8Array([0x41, 0x42, 0x43]));
    expect(frame[2]).toBe('E');
    expectDecodeError(frame.slice(0, 2) + 'A' + frame.slice(3), /checksum mismatch/);
  });

  test('invalid UTF-8 in a string-typed frame throws', () => {
    const invalid = new Uint8Array([0xFF, 0xFE, 0x41]);
    const frame = compress(invalid);
    // Flip the input-type flag from bytes to string (stored mode: flags 'E' → 'A') and
    // recompute the CRC field for the string type, so the UTF-8 check is what fails.
    expect(frame[2]).toBe('E');
    const crc = frameChecksum(invalid, false);
    let crcChars = '';
    for (let i = 0; i < 6; i++) crcChars += RADIX64_ALPHABET[(crc >>> (i * 6)) & 63]!;
    const patched = frame.slice(0, 2) + 'A' + frame[3]! + crcChars + frame.slice(10);
    expect(() => decompress(patched)).toThrow(/invalid UTF-8/);
  });
});

describe('token vectors', () => {
  test('history, rep, and overlap-copy matches round-trip', () => {
    const overlap = 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab'; // rep0 dist 1 overlap-copy
    const repeated = 'pattern-x pattern-y pattern-x pattern-y pattern-x';
    expect(decompress(compress(overlap))).toBe(overlap);
    expect(decompress(compress(repeated))).toBe(repeated);
  });

  test('long-distance matches round-trip', () => {
    // A match whose distance exceeds 4096 exercises the large offset slots.
    const unit = 'unique-marker-block-' + 'abcdefghij'.repeat(2);
    const filler = Array.from({ length: 400 }, (_, i) => `filler ${i} ${(i * 7919).toString(36)}`).join('\n');
    const input = unit + filler + unit;
    expect(input.length).toBeGreaterThan(4096 + unit.length);
    expect(decompress(compress(input))).toBe(input);
  });

  test('non-ASCII literal tails round-trip', () => {
    for (const tail of ['\u0080', '\u0080\u0081', '\u0080\u0081\u0082']) {
      const input = 'plain ascii text then raw bytes: ' + tail;
      expect(decompress(compress(input))).toBe(input);
    }
  });

  test('dictionary matches round-trip (wrapper dictionary idioms)', () => {
    const input = '```typescript\nexport function demo(): void {}\n```\n';
    expect(decompress(compress(input))).toBe(input);
  });

  test('downgrade determinism: identical inputs ship identical frames', () => {
    const inputs = ['', 'a', 'abcabcabc', 'x'.repeat(500), JSON.stringify({ k: 'v'.repeat(100) })];
    for (const input of inputs) {
      expect(compress(input)).toBe(compress(input));
    }
  });
});
