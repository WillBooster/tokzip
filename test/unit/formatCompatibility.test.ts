import { expect, test } from 'bun:test';
import { decompress } from '../../src/index.ts';
import fixtures from '../fixtures/format2.json';

for (const fixture of fixtures) {
  test(`reads a frozen format-2 frame: ${fixture.name}`, () => {
    const expected = fixture.text.repeat(fixture.repeat);
    const frame = Buffer.from(fixture.frame, 'base64');
    expect(decompress(frame, { maxLength: Buffer.byteLength(expected) })).toBe(expected);
  });
}
