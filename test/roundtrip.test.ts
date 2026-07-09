import { describe, expect, test } from 'bun:test';
import { compress, decompress } from '../src/index.ts';
import { RADIX64_ALPHABET } from '../src/radix64.ts';
import { RADIX85_ALPHABET } from '../src/radix85.ts';

// oxlint-disable-next-line no-misused-spread -- both alphabets are pure ASCII
const SAFE_CHARS = new Set([...RADIX64_ALPHABET, ...RADIX85_ALPHABET]);

const MODES: ('fast' | 'small')[] = ['fast', 'small'];

const STRING_CASES: Record<string, string> = {
  empty: '',
  'single char': 'a',
  short: 'hello world',
  'repetitive prose': 'The quick brown fox jumps over the lazy dog. '.repeat(40),
  'typescript source': 'export function greet(name: string): string {\n  return `Hello, ${name}!`;\n}\n'.repeat(8),
  json: JSON.stringify({ name: 'tokzip', versions: [1, 2, 3], nested: { deep: 'value'.repeat(50) } }),
  markdown: '# Title\n\n- item one\n- item two\n\n```typescript\nconst x = 1;\n```\n'.repeat(6),
  japanese: '圧縮アルゴリズムは辞書とエントロピー符号化を組み合わせます。'.repeat(25),
  chinese: '压缩算法结合了字典和熵编码。繁體中文也一樣。'.repeat(25),
  emoji: '😀🎉🚀 mixed emoji with text 😀🎉🚀 '.repeat(12),
  'control chars': 'line1\nline2\r\nline3\ttab\u0000null\u007F',
  'whole alphabet': Array.from({ length: 0x2_00 }, (_, i) => String.fromCodePoint(i)).join(''),
  'long single-char run': 'ab'.repeat(50_000) + 'x'.repeat(100_000),
};

describe.each(MODES)('%s mode round-trip', (mode) => {
  for (const [label, original] of Object.entries(STRING_CASES)) {
    test(label, () => {
      expect(decompress(compress(original, { mode }))).toBe(original);
    });
  }

  test('bytes input round-trips and returns Uint8Array', () => {
    const bytes = new Uint8Array(4096).map((_, i) => (i * 31 + (i >> 3)) & 255);
    const restored = decompress(compress(bytes, { mode }));
    expect(restored).toBeInstanceOf(Uint8Array);
    expect([...(restored as Uint8Array)]).toEqual([...bytes]);
  });

  test('random incompressible bytes never expand beyond header + stored body', () => {
    let seed = 0x12_34_56_78;
    const bytes = new Uint8Array(10_000).map(() => {
      // oxlint-disable-next-line unicorn/prefer-math-trunc -- >>> 0 coerces to uint32, Math.trunc does not
      seed = (Math.imul(seed, 1_103_515_245) + 12_345) >>> 0;
      return seed & 255;
    });
    const packed = compress(bytes, { mode });
    // Header (3) + size varint (4 for 10000) + packed stored body.
    expect(packed.length).toBeLessThanOrEqual(3 + 4 + Math.ceil((10_000 * 4) / 3) + 1);
    expect([...(decompress(packed) as Uint8Array)]).toEqual([...bytes]);
  });

  test('deterministic output', () => {
    const input = STRING_CASES['typescript source']!;
    expect(compress(input, { mode })).toBe(compress(input, { mode }));
  });

  test('output stays in the safe-ASCII alphabet', () => {
    for (const original of Object.values(STRING_CASES)) {
      const packed = compress(original, { mode });
      // oxlint-disable-next-line no-misused-spread -- payloads are pure ASCII
      expect([...packed].every((c) => SAFE_CHARS.has(c))).toBe(true);
      // JSON-safety: embedding the payload in JSON must survive verbatim.
      // oxlint-disable-next-line unicorn/prefer-structured-clone -- intentionally exercises a JSON round-trip
      expect(JSON.parse(JSON.stringify(packed))).toBe(packed);
      expect(packed.includes('"')).toBe(false);
      expect(packed.includes('\\')).toBe(false);
      expect(packed.includes('`')).toBe(false);
      expect(packed.includes('$')).toBe(false);
    }
  });
});

test('lone surrogates are replaced with U+FFFD (WHATWG TextEncoder semantics)', () => {
  const input = 'broken \u{D800} surrogate';
  expect(decompress(compress(input))).toBe('broken \u{FFFD} surrogate');
});

test('small mode is never larger than fast mode', () => {
  for (const original of Object.values(STRING_CASES)) {
    expect(compress(original, { mode: 'small' }).length).toBeLessThanOrEqual(
      compress(original, { mode: 'fast' }).length
    );
  }
});

test('unregistered language throws on compress', () => {
  expect(() => compress('x', { language: 'klingon' })).toThrow(RangeError);
});
