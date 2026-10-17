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
}

const referenceFrames: ReferenceFrame[] = [...fixtures, ...languages];

for (const fixture of referenceFrames) {
  test(`reads a frozen format-2 frame: ${fixture.name}`, () => {
    const expected = fixture.text.repeat(fixture.repeat);
    const frame = Buffer.from(fixture.frame, 'base64');
    if (fixture.id !== undefined) {
      expect(frame[0]).toBe(0xD8);
      const sizeEnd = frame.subarray(1).findIndex((byte) => (byte & 0x80) === 0) + 1;
      expect(sizeEnd).toBeGreaterThan(0);
      expect(frame[sizeEnd + 5]).toBe(fixture.id);
    }
    expect(decompress(frame, { maxLength: Buffer.byteLength(expected) })).toBe(expected);
  });
}
