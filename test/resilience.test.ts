import { describe, expect, test } from 'bun:test';
import {
  compress,
  compressForStorage,
  decompress,
  inspectFrame,
  TokzipCompressionStream,
  TokzipDecodeError,
  TokzipDecompressionStream,
} from '../src/index.ts';
import '../src/languages/typescript.ts';

/** Deterministic PRNG (mulberry32) so fuzz failures reproduce exactly. */
// oxlint-disable unicorn/prefer-math-trunc -- >>> 0 converts to unsigned 32-bit; Math.trunc would keep the sign
function seededRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6D_2B_79_F5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4_294_967_296;
  };
}
// oxlint-enable unicorn/prefer-math-trunc

const SEED_DOCS = [
  'export function greet(name: string): string {\n  return `Hello, ${name}!`;\n}\n'.repeat(8),
  '# Notes\n\nSome prose with a fence:\n\n```ts\nconst x: number = 42;\nconsole.log(x);\n```\n'.repeat(4),
  'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab',
  JSON.stringify({ items: Array.from({ length: 40 }, (_, i) => ({ id: i, name: `item-${i}` })) }),
];

/** Decode must either throw TokzipDecodeError or return the exact original — never anything else. */
function expectSafeDecode(frame: string | Uint8Array, original: string): void {
  let decoded: string | Uint8Array;
  try {
    decoded = decompress(frame);
  } catch (error) {
    expect(error).toBeInstanceOf(TokzipDecodeError);
    return;
  }
  // Surviving a mutation is fine only if the content is untouched (CRC-32 guards the rest).
  expect(decoded).toBe(original);
}

describe('decoder fuzzing', () => {
  test('mutated text frames never crash, hang, or decode silently wrong', () => {
    const random = seededRandom(0xC0_FF_EE);
    for (const [docIndex, doc] of SEED_DOCS.entries()) {
      for (const mode of ['fast', 'small'] as const) {
        const frame = compress(doc, { language: 'typescript', mode });
        for (let round = 0; round < 150; round++) {
          const kind = random();
          let mutated: string;
          if (kind < 0.4) {
            // Flip one char to a random printable ASCII char.
            const at = Math.floor(random() * frame.length);
            const replacement = String.fromCodePoint(33 + Math.floor(random() * 94));
            mutated = frame.slice(0, at) + replacement + frame.slice(at + 1);
          } else if (kind < 0.7) {
            mutated = frame.slice(0, Math.floor(random() * frame.length));
          } else {
            const at = Math.floor(random() * frame.length);
            const insertion = String.fromCodePoint(33 + Math.floor(random() * 94));
            mutated = frame.slice(0, at) + insertion + frame.slice(at);
          }
          if (mutated === frame) continue;
          expectSafeDecode(mutated, doc);
          // docIndex keeps the loop observable so no round is optimized away.
          expect(docIndex).toBeLessThan(SEED_DOCS.length);
        }
      }
    }
  });

  test('mutated binary frames never crash, hang, or decode silently wrong', () => {
    const random = seededRandom(0xBE_EF);
    for (const doc of SEED_DOCS) {
      for (const mode of ['fast', 'small'] as const) {
        const frame = compress(doc, { language: 'typescript', mode, output: 'binary' });
        for (let round = 0; round < 150; round++) {
          const mutated = Uint8Array.from(frame);
          const kind = random();
          if (kind < 0.5) {
            const at = Math.floor(random() * mutated.length);
            mutated[at] = mutated[at]! ^ (1 << Math.floor(random() * 8));
            expectSafeDecode(mutated, doc);
          } else {
            expectSafeDecode(mutated.subarray(0, Math.floor(random() * mutated.length)), doc);
          }
        }
      }
    }
  });

  test('random garbage inputs throw typed decode errors', () => {
    // Every sample of this deterministic seed currently throws, so the assertion is
    // unconditional — a regression that starts accepting malformed input must fail here.
    const random = seededRandom(0xDE_AD);
    for (let round = 0; round < 300; round++) {
      const length = Math.floor(random() * 64);
      const text = Array.from({ length }, () => String.fromCodePoint(32 + Math.floor(random() * 95))).join('');
      expect(() => decompress(text)).toThrow(TokzipDecodeError);
      const bytes = Uint8Array.from({ length }, () => Math.floor(random() * 256));
      expect(() => decompress(bytes)).toThrow(TokzipDecodeError);
    }
  });

  test('corrupted streams throw typed decode errors', async () => {
    const doc = SEED_DOCS[0]!.repeat(8);
    const compressed = await pumpStream(new TokzipCompressionStream({ language: 'typescript' }), doc);
    const random = seededRandom(0x5E_ED);
    for (let round = 0; round < 40; round++) {
      const mutated = Uint8Array.from(compressed);
      const at = Math.floor(random() * mutated.length);
      mutated[at] = mutated[at]! ^ (1 << Math.floor(random() * 8));
      let output: Uint8Array;
      try {
        output = await pumpStream(new TokzipDecompressionStream(), mutated);
      } catch (error) {
        expect(error).toBeInstanceOf(TokzipDecodeError);
        continue;
      }
      expect(new TextDecoder().decode(output)).toBe(doc);
    }
  });
});

/** Parses an encoded stream into [start, end) ranges: header, blocks…, terminator. */
function streamRanges(encoded: Uint8Array): [number, number][] {
  const readVarint = (pos: number): [number, number] => {
    let value = 0;
    let shift = 1;
    for (;;) {
      const group = encoded[pos++]!;
      value += (group & 127) * shift;
      if ((group & 128) === 0) return [value, pos];
      shift *= 128;
    }
  };
  const ranges: [number, number][] = [[0, 3]];
  let pos = 3;
  for (;;) {
    const start = pos;
    const [bodyLength, afterLength] = readVarint(pos);
    if (bodyLength === 0) {
      ranges.push([start, encoded.length]);
      return ranges;
    }
    const [, afterRaw] = readVarint(afterLength + 1);
    pos = afterRaw + 4 + bodyLength;
    ranges.push([start, pos]);
  }
}

function concatBytes(parts: Uint8Array[]): Uint8Array {
  const out = new Uint8Array(parts.reduce((sum, part) => sum + part.length, 0));
  let at = 0;
  for (const part of parts) {
    out.set(part, at);
    at += part.length;
  }
  return out;
}

describe('stream block tampering', () => {
  async function expectStreamRejected(tampered: Uint8Array): Promise<void> {
    await expect(pumpStream(new TokzipDecompressionStream(), tampered)).rejects.toThrow(TokzipDecodeError);
  }

  test('deleting, reordering, or dropping trailing blocks fails the chained checksum', async () => {
    const input = Uint8Array.from({ length: 3072 }, (_, i) => (i * 7) % 251);
    const encoded = await pumpStream(new TokzipCompressionStream({ blockSize: 1024, carryWindow: false }), input);
    const ranges = streamRanges(encoded);
    expect(ranges.length).toBe(5); // Header + 3 blocks + terminator.
    const slice = ([start, end]: [number, number]): Uint8Array => encoded.subarray(start, end);
    const [header, block1, block2, block3, terminator] = ranges;
    // Deleting a middle block breaks the next block's chained CRC.
    await expectStreamRejected(concatBytes([slice(header!), slice(block1!), slice(block3!), slice(terminator!)]));
    // Reordering intact blocks breaks the chain too.
    await expectStreamRejected(
      concatBytes([slice(header!), slice(block2!), slice(block1!), slice(block3!), slice(terminator!)])
    );
    // Dropping trailing blocks passes every per-block check; the terminator catches it.
    await expectStreamRejected(concatBytes([slice(header!), slice(block1!), slice(block2!), slice(terminator!)]));
    // The untampered stream still round-trips.
    const restored = await pumpStream(new TokzipDecompressionStream(), encoded);
    expect(restored).toEqual(input);
  });
});

describe('compressForStorage', () => {
  test('verified frames round-trip and match plain compress output', () => {
    for (const doc of SEED_DOCS) {
      const frame = compressForStorage(doc, { language: 'typescript', mode: 'small' });
      expect(frame).toBe(compress(doc, { language: 'typescript', mode: 'small' }));
      expect(decompress(frame)).toBe(doc);
    }
  });

  test('byte inputs verify byte-exactly on both channels', () => {
    const bytes = Uint8Array.from({ length: 500 }, (_, i) => (i * 37) % 256);
    for (const output of ['text', 'binary'] as const) {
      const frame = compressForStorage(bytes, { mode: 'small', output });
      expect(decompress(frame)).toEqual(bytes);
    }
  });
});

describe('inspectFrame', () => {
  test('reports header facts without registered languages', () => {
    const doc = SEED_DOCS[0]!;
    const frame = compress(doc, { language: 'typescript', mode: 'small' });
    const info = inspectFrame(frame);
    expect(info.version).toBe(1);
    expect(info.container).toBe('text');
    expect(info.contentType).toBe('string');
    expect(info.contentBytes).toBe(Buffer.byteLength(doc));
    expect(['fast', 'small', 'stored']).toContain(info.mode);

    const binaryInfo = inspectFrame(compress(doc, { language: 'typescript', mode: 'small', output: 'binary' }));
    expect(binaryInfo.container).toBe('binary');
    expect(binaryInfo.contentBytes).toBe(Buffer.byteLength(doc));
    expect(binaryInfo.checksum).toBe(info.checksum);
  });

  test('inspects non-stored frames whose language id is not registered anywhere', () => {
    // This process has typescript registered, so patch the frame to an unallocated id:
    // inspection must still succeed (it never resolves languages — the server property).
    const frame = compress(SEED_DOCS[0]!, { language: 'typescript', mode: 'small' });
    expect(inspectFrame(frame).mode).not.toBe('stored');
    const patchedText = frame[0]! + '9' + frame.slice(2); // Radix-64 value 61: unregistered.
    expect(inspectFrame(patchedText).languageId).toBe(61);

    const binaryFrame = compress(SEED_DOCS[0]!, { language: 'typescript', mode: 'small', output: 'binary' });
    const patchedBinary = Uint8Array.from(binaryFrame);
    patchedBinary[1] = 200; // Far outside any allocation.
    expect(inspectFrame(patchedBinary).languageId).toBe(200);
  });

  test('rejects structural violations', () => {
    const frame = compress('hello world hello world hello world');
    expect(() => inspectFrame('A' + frame.slice(1))).toThrow(/bad magic/);
    expect(() => inspectFrame('z' + frame.slice(1))).toThrow(/unknown version/);
    expect(() => inspectFrame(frame.slice(0, 5))).toThrow(TokzipDecodeError);
    expect(() => inspectFrame(frame + 'AAAA'.repeat(20))).toThrow(TokzipDecodeError);
  });

  test('rejects declared sizes beyond the body capacity on both channels', () => {
    // The repro from review: a fast frame truncated to its first body char but declaring
    // more content than one char can produce must fail inspection, matching decompress.
    const oversized = compress('x'.repeat(300_000), { language: 'none', mode: 'fast' });
    // Keep the header (3 chars + 4-char size varint + 6-char CRC) plus one body char: a
    // single fast token cannot produce the declared 300,000 bytes.
    expect(() => inspectFrame(oversized.slice(0, 14))).toThrow(/body capacity|truncated/);
    const oversizedBinary = compress('x'.repeat(300_000), { language: 'none', mode: 'fast', output: 'binary' });
    expect(() => inspectFrame(oversizedBinary.subarray(0, 11))).toThrow(/body capacity|truncated/);
  });

  test('rejects malformed text bodies (alphabet and radix-85 word boundary)', () => {
    // Findings from review: these are decidable without decoding, so a pass-through server
    // must reject them like decompress does.
    const doc = 'export const value = { answer: 42, name: "example" };\n'.repeat(20);
    const small = compress(doc, { language: 'typescript', mode: 'small' });
    expect(inspectFrame(small).mode).toBe('small');
    expect(() => inspectFrame(small.slice(0, -1))).toThrow(/multiple of 5|non-canonical|truncated/);
    const bodyAt = small.length - 3;
    expect(() => inspectFrame(small.slice(0, bodyAt) + '%' + small.slice(bodyAt + 1))).toThrow(/non-alphabet/);
    expect(() => inspectFrame(small.slice(0, - 5) + '~~~~~')).toThrow(/out of range|non-alphabet/);

    const fast = compress(doc, { language: 'typescript', mode: 'fast' });
    expect(inspectFrame(fast).mode).toBe('fast');
    const fastBodyAt = fast.length - 2;
    expect(() => inspectFrame(fast.slice(0, fastBodyAt) + '%' + fast.slice(fastBodyAt + 1))).toThrow(/non-alphabet/);
  });

  test('accepts every fuzz-seed frame on both channels', () => {
    for (const doc of SEED_DOCS) {
      for (const mode of ['fast', 'small'] as const) {
        inspectFrame(compress(doc, { language: 'typescript', mode }));
        inspectFrame(compress(doc, { language: 'typescript', mode, output: 'binary' }));
      }
    }
  });
});

async function pumpStream(
  stream: TransformStream<Uint8Array | string, Uint8Array>,
  input: string | Uint8Array
): Promise<Uint8Array> {
  const writer = stream.writable.getWriter();
  const written = (async () => {
    await writer.write(input);
    await writer.close();
    // A decode error surfaces through the read loop; the mirrored writable rejection would
    // otherwise be an unhandled rejection that fails the test run.
  })().catch(() => {});
  const chunks: Uint8Array[] = [];
  let total = 0;
  const reader = stream.readable.getReader();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    total += value.length;
  }
  await written;
  const out = new Uint8Array(total);
  let at = 0;
  for (const chunk of chunks) {
    out.set(chunk, at);
    at += chunk.length;
  }
  return out;
}
