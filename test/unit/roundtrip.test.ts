import { describe, expect, test } from 'bun:test';
import { compress, decompress, TokzipDecodeError } from '../../src/index.ts';

const JAPANESE_PROMPT = `以下の要件を満たすブロック崩しゲームを作成してください。
- キャンバスサイズは 800x600 とし、背景は暗い青にしてください。
- パドルは左右矢印キーで移動し、画面端で止まります。
- ブロックは 5 行 × 10 列に配置し、色は行ごとに変えてください。
`;

const MIXED_ANSWER = `${JAPANESE_PROMPT}
\`\`\`html
<!doctype html>
<html><head><style>body { margin: 0; background: #0a1a3a; } canvas { display: block; }</style></head>
<body><canvas id="game" width="800" height="600"></canvas>
<script>
const canvas = document.getElementById('game');
const ctx = canvas.getContext('2d');
let score = 0;
function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = '#0a1a3a';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  requestAnimationFrame(draw);
}
draw();
</script></body></html>
\`\`\`

## 概要
パドルとボールを描画し、ブロックを配置しました。全てのブロックを消すとクリア画面を表示します。
`;

describe('compress/decompress', () => {
  test('round-trips strings, including empty and non-ASCII content', () => {
    // Astral characters at odd byte offsets: the UTF-8 length must be computed exactly.
    const inputs = [
      '',
      'a',
      'hello',
      JAPANESE_PROMPT,
      MIXED_ANSWER,
      '﻿bom prefixed',
      '🎮'.repeat(100),
      'あ𝄞',
      '🎮あ',
      'a🎮',
    ];
    for (const input of inputs) {
      const frame = compress(input);
      expect(frame).toBeInstanceOf(Uint8Array);
      expect(decompress(frame)).toBe(input);
    }
  });

  test('refuses strings with lone surrogates instead of altering them', () => {
    expect(() => compress('a\uD83C')).toThrow(RangeError);
  });

  test('compresses prompts and mixed LLM answers well below their size', () => {
    const prompt = compress(JAPANESE_PROMPT.repeat(3));
    expect(prompt.length * 2).toBeLessThan(Buffer.byteLength(JAPANESE_PROMPT.repeat(3)));
    const answer = compress(MIXED_ANSWER);
    expect(answer.length * 2).toBeLessThan(Buffer.byteLength(MIXED_ANSWER));
  });

  test('never expands incompressible input beyond the stored frame', () => {
    let x = 0x25_45_F4_91;
    const randomText = (length: number): string => {
      let text = '';
      for (let i = 0; i < length; i++) {
        x ^= x << 13;
        x ^= x >>> 17;
        x ^= x << 5;
        text += String.fromCodePoint(0x21 + ((x >>> 0) % 94));
      }
      return text;
    };
    // Short random text cannot be coded smaller than it is stored: the frame is the stored
    // layout (header bits 0b11) and costs exactly the 6-byte header.
    for (let i = 0; i < 20; i++) {
      const noise = randomText(64);
      const frame = compress(noise);
      expect(frame[0]! & 0b11).toBe(0b11);
      expect(frame.length).toBe(Buffer.byteLength(noise) + 6);
      expect(decompress(frame)).toBe(noise);
    }
    // Longer random text is coded (its byte structure is compressible), never expanded.
    const long = randomText(4096);
    expect(compress(long).length).toBeLessThanOrEqual(Buffer.byteLength(long) + 7);
    expect(decompress(compress(long))).toBe(long);
  });

  test('handles inputs spanning many parse chunks', () => {
    const large = MIXED_ANSWER.repeat(200); // ~180 KB
    expect(decompress(compress(large))).toBe(large);
  });

  test('codes content above one 4 MiB block instead of storing it', () => {
    const huge = MIXED_ANSWER.repeat(7000); // ~6.7 MiB, spanning two blocks
    expect(Buffer.byteLength(huge)).toBeGreaterThan(4 * 1024 * 1024);
    const frame = compress(huge);
    expect(frame[0]! & 0b11).toBe(0b10); // blocked layout
    expect(frame.length * 10).toBeLessThan(Buffer.byteLength(huge));
    expect(decompress(frame)).toBe(huge);
  });

  test('rejects frames declaring more content than maxLength', () => {
    const frame = compress(MIXED_ANSWER);
    expect(() => decompress(frame, { maxLength: 10 })).toThrow(TokzipDecodeError);
    expect(() => decompress(frame, { maxLength: -1 })).toThrow(RangeError);
    expect(decompress(frame, { maxLength: Buffer.byteLength(MIXED_ANSWER) })).toBe(MIXED_ANSWER);
  });
});
