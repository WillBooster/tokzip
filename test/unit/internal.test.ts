import { expect, test } from 'bun:test';
import '../../src/index.ts';
import { languageByName } from '../../src/dictionary.ts';
import { dictIndexFor, parse } from '../../src/lz.ts';
import { BitReader, BitWriter, decodeRadix85, RADIX85_ALPHABET } from '../../src/radix85.ts';
import { decodeSmallBodyBinary, encodeSmallBody, smallPricing } from '../../src/smallMode.ts';
import { extraBitsOf, extraValueOf, slotOf, valueOfSlot } from '../../src/slots.ts';

const SAMPLES = [
  'const x = 1;\n'.repeat(40),
  'The quick brown fox jumps over the lazy dog. '.repeat(30),
  'mixed \u{0000}\u{00FF} bytes and ascii text '.repeat(15),
  JSON.stringify({ deep: { nested: ['a', 'b', 'c'] } }).repeat(10),
];

test('encoded small bodies decode back to the exact token-covered bytes', () => {
  const language = languageByName('none')!;
  const encoder = new TextEncoder();
  for (const sample of SAMPLES) {
    const bytes = encoder.encode(sample);
    const tokens = parse(bytes, language.dictionary, dictIndexFor(language), smallPricing(bytes, language));
    const body = encodeSmallBody(tokens, bytes, language);
    const decoded = decodeSmallBodyBinary(body, 0, body.length, bytes.length, language);
    expect(decoded).toEqual(bytes);
  }
});

test('BitWriter/BitReader round-trip across word boundaries', () => {
  const writer = new BitWriter();
  const values: [number, number][] = [];
  let seed = 42;
  for (let i = 0; i < 500; i++) {
    seed = (Math.imul(seed, 48_271) % 2_147_483_647) & 0x7F_FF_FF_FF || 1;
    const bits = (seed % 24) + 1;
    const value = seed % 2 ** bits;
    values.push([value, bits]);
    writer.writeBits(value, bits);
  }
  const text = writer.toText();
  expect(text.length % 5).toBe(0);
  for (const c of text) expect(RADIX85_ALPHABET.includes(c)).toBe(true);
  const reader = new BitReader(decodeRadix85(text, 0, text.length));
  for (const [value, bits] of values) expect(reader.readBits(bits)).toBe(value);
});

test('slot codec round-trips every value shape', () => {
  for (const value of [0, 1, 2, 3, 4, 5, 7, 8, 15, 16, 100, 4095, 4096, 262_143, 1_048_575]) {
    const slot = slotOf(value);
    expect(valueOfSlot(slot, extraValueOf(value, slot))).toBe(value);
    expect(extraValueOf(value, slot)).toBeLessThan(2 ** extraBitsOf(slot));
  }
});
