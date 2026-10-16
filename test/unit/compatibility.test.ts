import { expect, test } from 'bun:test';
import { readFileSync } from 'node:fs';
import { z } from 'zod';
import { compress, decompress, TokzipDecodeError } from '../../src/index.ts';

const fixtures = z
  .object({
    cases: z.array(
      z.object({
        name: z.string(),
        content: z.string(),
        frame: z.string(),
        repeat: z.number().int().positive().optional(),
      })
    ),
  })
  .parse(JSON.parse(readFileSync(new URL('../fixtures/v1Frames.json', import.meta.url), 'utf8')));

for (const fixture of fixtures.cases) {
  test(`reads v1.0.0 ${fixture.name} frames`, () => {
    const frame = Buffer.from(fixture.frame, 'base64');
    expect(decompress(frame)).toBe(fixture.content.repeat(fixture.repeat ?? 1));
  });
}

test('applies the output limit to released frames', () => {
  const fixture = fixtures.cases.find((entry) => entry.name === 'blocked')!;
  expect(() => decompress(Buffer.from(fixture.frame, 'base64'), { maxLength: 1024 })).toThrow(TokzipDecodeError);
});

test('keeps released and current models independent across calls', () => {
  const fixture = fixtures.cases.find((entry) => entry.name === 'ja-JP')!;
  const legacy = Buffer.from(fixture.frame, 'base64');
  const content = `${fixture.content}\nPreserve this content exactly.`;
  const current = compress(content);
  for (let round = 0; round < 3; round++) {
    expect(decompress(legacy)).toBe(fixture.content);
    expect(decompress(current)).toBe(content);
    expect(compress(content)).toEqual(current);
  }
});
