import { expect, test } from 'bun:test';

import { decompress } from '../../src/index.ts';
import fixtures from '../fixtures/format2.json';
import languages from '../fixtures/format2Languages.json';

interface ReferenceFrame {
  name: string;
  text: string;
  repeat: number;
  frame: string;
  id?: number;
  suffix?: string;
}

const referenceFrames: ReferenceFrame[] = [...fixtures, ...languages];

test('keeps one reference frame for every format-2 language ID', () => {
  expect(languages.map(({ id }) => id).toSorted((a, b) => a - b)).toEqual(Array.from({ length: 21 }, (_, id) => id));
});

for (const fixture of referenceFrames) {
  test(`reads a frozen format-2 frame: ${fixture.name}`, () => {
    const expected = fixture.text.repeat(fixture.repeat) + (fixture.suffix ?? '');
    const frame = Buffer.from(fixture.frame, 'base64');
    if (fixture.id !== undefined) {
      expect(frame[0]).toBe(0xd8);
      const sizeEnd = frame.subarray(1).findIndex((byte) => (byte & 0x80) === 0) + 1;
      expect(sizeEnd).toBeGreaterThan(0);
      expect(frame[sizeEnd + 5]).toBe(fixture.id);
    }
    expect(decompress(frame, { maxLength: Buffer.byteLength(expected) })).toBe(expected);
  });
}
