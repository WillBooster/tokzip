import { describe, expect, test } from 'bun:test';
import { compress, decompress, LANGUAGE_IDS } from '../src/index.ts';
// Importing the barrel registers every trained module, which re-runs table validation
// (registerLanguage throws on incomplete codes) before any round-trip below.
import '../src/languages/index.ts';
import { languageByName } from '../src/dictionary.ts';
import { fromBase64 } from '../src/moduleData.ts';

const SAMPLES: Record<string, string> = {
  code: 'export async function fetchUser(id: string): Promise<User> {\n  const response = await fetch(`/api/users/${id}`);\n  return (await response.json()) as User;\n}\n'.repeat(
    3
  ),
  markdown: '# Guide\n\nUse the following snippet:\n\n```python\nprint("hello")\n```\n\n- fast\n- small\n'.repeat(4),
  japanese: '辞書とエントロピー符号化を組み合わせた可逆圧縮の実装です。'.repeat(12),
  chinese: '这是一个结合字典和熵编码的无损压缩实现。壓縮與解壓縮都必須可逆。'.repeat(12),
};

// Exercises the dictionary-match, offset-slot, and trained-Huffman decode paths that the
// id-0 tests never reach; a trainer regression that ships a broken module fails here.
describe('every trained language module round-trips', () => {
  const names = Object.keys(LANGUAGE_IDS).filter((name) => languageByName(name));
  test('all v1 languages are registered', () => {
    expect(names.length).toBe(Object.keys(LANGUAGE_IDS).length);
  });
  describe.each(names)('%s', (name) => {
    test.each(['fast', 'small'] as const)('%s mode', (mode) => {
      for (const sample of Object.values(SAMPLES)) {
        expect(decompress(compress(sample, { language: name, mode }))).toBe(sample);
      }
    });
  });
});

test('fromBase64 rejects non-ASCII instead of silently decoding it as 0', () => {
  expect(() => fromBase64('AAあA')).toThrow(RangeError);
  expect(() => fromBase64('AA!A')).toThrow(RangeError);
});
