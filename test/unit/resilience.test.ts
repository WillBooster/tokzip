import { describe, expect, test } from 'bun:test';
import { compress, decompress, TokzipDecodeError } from '../../src/index.ts';

const SAMPLE = 'const answer = 42; // the answer\n'.repeat(20) + 'ゲームの仕様を以下に示します。'.repeat(10);

describe('malformed frames', () => {
  test('garbage and truncated frames throw TokzipDecodeError', () => {
    expect(() => decompress(new Uint8Array())).toThrow(TokzipDecodeError);
    expect(() => decompress(new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8]))).toThrow(TokzipDecodeError);
    const frame = compress(SAMPLE);
    for (const cut of [1, 5, 8, frame.length >> 1]) {
      expect(() => decompress(frame.subarray(0, cut))).toThrow(TokzipDecodeError);
    }
    // The range coder's final bytes carry slack, so a short truncation may still decode —
    // but only ever to the exact original (the CRC catches everything else).
    for (const cut of [frame.length - 1, frame.length - 2]) {
      let decoded: string | Uint8Array | undefined;
      try {
        decoded = decompress(frame.subarray(0, cut));
      } catch (error) {
        expect(error).toBeInstanceOf(TokzipDecodeError);
        continue;
      }
      expect(decoded).toBe(SAMPLE);
    }
    expect(() => decompress(new Uint8Array([...frame, 0]))).toThrow(TokzipDecodeError);
  });

  test('other format versions are rejected, never misdecoded', () => {
    const frame = compress(SAMPLE);
    const bumped = Uint8Array.from(frame);
    bumped[0] = (bumped[0]! & 0xF0) | 1;
    expect(() => decompress(bumped)).toThrow(/unsupported format version/);
  });

  test('every single-byte mutation either throws or decodes to the original', () => {
    const frame = compress(SAMPLE);
    for (let i = 0; i < frame.length; i++) {
      const mutated = Uint8Array.from(frame);
      mutated[i] = mutated[i]! ^ 0x5A;
      let decoded: string | Uint8Array | undefined;
      try {
        decoded = decompress(mutated);
      } catch (error) {
        expect(error).toBeInstanceOf(TokzipDecodeError);
        continue;
      }
      expect(decoded).toBe(SAMPLE);
    }
  });
});
