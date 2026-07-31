import { expect, test } from 'bun:test';
import '../../src/index.ts';
import { languageByName } from '../../src/dictionary.ts';
import { dictIndexFor, parse } from '../../src/lz.ts';
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

test('slot codec round-trips every value shape', () => {
  for (const value of [0, 1, 2, 3, 4, 5, 7, 8, 15, 16, 100, 4095, 4096, 262_143, 1_048_575]) {
    const slot = slotOf(value);
    expect(valueOfSlot(slot, extraValueOf(value, slot))).toBe(value);
    expect(extraValueOf(value, slot)).toBeLessThan(2 ** extraBitsOf(slot));
  }
});
