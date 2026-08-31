import { describe, expect, test } from 'bun:test';
import { compress, decompress } from '../../src/index.ts';

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
    for (const input of ['', 'a', 'hello', JAPANESE_PROMPT, MIXED_ANSWER, '﻿bom prefixed', '🎮'.repeat(100)]) {
      const frame = compress(input);
      expect(frame).toBeInstanceOf(Uint8Array);
      expect(decompress(frame)).toBe(input);
    }
  });

  test('refuses strings with lone surrogates instead of altering them', () => {
    expect(() => compress('a\uD83C')).toThrow(RangeError);
  });

  test('round-trips bytes and preserves the string/bytes distinction', () => {
    const bytes = new Uint8Array(1000).map((_, i) => (i * 7919) & 0xFF);
    const restored = decompress(compress(bytes));
    expect(restored).toBeInstanceOf(Uint8Array);
    expect(restored).toEqual(bytes);
    const text = decompress(compress('plain text'));
    expect(typeof text).toBe('string');
  });

  test('compresses prompts and mixed LLM answers well below their size', () => {
    const prompt = compress(JAPANESE_PROMPT.repeat(3));
    expect(prompt.length * 2).toBeLessThan(Buffer.byteLength(JAPANESE_PROMPT.repeat(3)));
    const answer = compress(MIXED_ANSWER);
    expect(answer.length * 2).toBeLessThan(Buffer.byteLength(MIXED_ANSWER));
  });

  test('never expands incompressible input beyond the stored frame', () => {
    const noise = new Uint8Array(4096);
    let x = 0x25_45_F4_91;
    for (let i = 0; i < noise.length; i++) {
      x ^= x << 13;
      x ^= x >>> 17;
      x ^= x << 5;
      noise[i] = x >>> 24;
    }
    expect(compress(noise).length).toBeLessThanOrEqual(noise.length + 8);
    expect(decompress(compress(noise))).toEqual(noise);
  });

  test('handles inputs spanning many parse chunks', () => {
    const large = MIXED_ANSWER.repeat(200); // ~180 KB
    expect(decompress(compress(large))).toBe(large);
  });
});
