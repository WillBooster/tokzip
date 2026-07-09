import { expect, test } from 'bun:test';
import '../src/index.ts';
import { languageByName } from '../src/dictionary.ts';
import { encodeFastBody, fastBodyCost, fastPricing } from '../src/fastMode.ts';
import { buildDecoder, buildEncoder, buildLengths, isCompleteCode } from '../src/huffman.ts';
import { dictIndexFor, parse } from '../src/lz.ts';
import { BitReader, BitWriter, decodeRadix85, RADIX85_ALPHABET } from '../src/radix85.ts';
import { emitSmallBody, planSmallBody, smallPricing } from '../src/smallMode.ts';
import { extraBitsOf, extraValueOf, slotOf, valueOfSlot } from '../src/slots.ts';

const SAMPLES = [
  'const x = 1;\n'.repeat(40),
  'The quick brown fox jumps over the lazy dog. '.repeat(30),
  'mixed \u{0000}\u{00FF} bytes and ascii text '.repeat(15),
  JSON.stringify({ deep: { nested: ['a', 'b', 'c'] } }).repeat(10),
];

test('analytic fast cost equals the emitted fast body length (downgrade exactness)', () => {
  const language = languageByName('none')!;
  const encoder = new TextEncoder();
  for (const sample of SAMPLES) {
    const bytes = encoder.encode(sample);
    for (const pricing of [fastPricing(bytes, language), smallPricing(bytes, language)]) {
      const tokens = parse(bytes, language.dictionary, dictIndexFor(language), pricing);
      const cost = fastBodyCost(tokens, bytes, language);
      expect(cost).toBe(encodeFastBody(tokens, bytes, language).length);
    }
  }
});

test('planned small char cost equals the emitted small body length', () => {
  const language = languageByName('none')!;
  const encoder = new TextEncoder();
  for (const sample of SAMPLES) {
    const bytes = encoder.encode(sample);
    const tokens = parse(bytes, language.dictionary, dictIndexFor(language), smallPricing(bytes, language));
    const plan = planSmallBody(tokens, bytes, language);
    expect(plan.charCost).toBe(emitSmallBody(plan, bytes, language).length);
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

test('sparse complete tables (unused length-0 symbols) round-trip through encoder/decoder', () => {
  // RFC 1951-style assignment: unused symbols must not shift next_code.
  const lengths = new Uint8Array(8);
  lengths[0] = lengths[1] = lengths[2] = lengths[3] = 2;
  expect(isCompleteCode(lengths)).toBe(true);
  const { codes } = buildEncoder(lengths);
  expect([codes[0], codes[1], codes[2], codes[3]]).toEqual([0, 1, 2, 3]);
  const decoder = buildDecoder(lengths);
  for (let symbol = 0; symbol < 4; symbol++) {
    const writer = new BitWriter();
    writer.writeBits(codes[symbol]!, 2);
    const text = writer.toText();
    const reader = new BitReader(decodeRadix85(text, 0, text.length));
    const entry = decoder[reader.peekBits(12)]!;
    expect(entry >>> 4).toBe(symbol);
    expect(entry & 15).toBe(2);
  }
});

test('package-merge lengths form complete canonical codes', () => {
  const freqs = new Float64Array(256).map((_, i) => (i % 7 === 0 ? 1000 : i + 1));
  const lengths = buildLengths(freqs);
  expect(isCompleteCode(lengths)).toBe(true);
  const decoder = buildDecoder(lengths);
  const { codes } = buildEncoder(lengths);
  for (let symbol = 0; symbol < 256; symbol++) {
    const length = lengths[symbol]!;
    const entry = decoder[codes[symbol]! << (12 - length)]!;
    expect(entry >>> 4).toBe(symbol);
    expect(entry & 15).toBe(length);
  }
});
