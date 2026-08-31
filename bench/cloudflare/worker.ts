/**
 * Cloudflare Workers benchmark for the wasm build. Workers freeze the clock during CPU work,
 * so this worker only does the work; times come from `wrangler tail --format json` (per-request
 * `cpuTime`) and from the client's time-to-first-byte.
 *
 *   /noop                       nothing (baseline)
 *   /load                       import the module and run one compress/decompress
 *   /compress/<sample>?iters=N  compress the sample N times
 *   /decompress/<sample>?iters=N
 *
 *   wrangler deploy --config bench/cloudflare/wrangler.jsonc
 *   bun bench/cloudflare/measure.ts
 */
import samples from './samples.json';

const SAMPLES: Record<string, string> = samples;

export default {
  async fetch(request: Request): Promise<Response> {
    const url = new URL(request.url);
    const [route, sample = ''] = url.pathname.split('/').filter(Boolean);
    if (route === undefined || route === 'noop') return new Response('ok');
    const codec = await import('../../src/index.ts');
    if (route === 'load') {
      const frame = codec.compress('warm-up: 最初の圧縮でモジュール初期化と辞書索引の構築が走る。');
      return new Response(`loaded: ${frame.length}`);
    }
    const text = SAMPLES[sample];
    if (!text) return new Response('unknown sample', { status: 404 });
    const iterations = Number(url.searchParams.get('iters') ?? '1');
    if (route === 'compress') {
      let size = 0;
      for (let i = 0; i < iterations; i++) size = codec.compress(text).length;
      return new Response(`${sample}: ${text.length} chars -> ${size} bytes x${iterations}`);
    }
    if (route === 'decompress') {
      const frame = codec.compress(text);
      let ok = true;
      for (let i = 0; i < iterations; i++) ok &&= codec.decompress(frame) === text;
      return new Response(`${sample}: ${ok ? 'round-trip ok' : 'MISMATCH'} x${iterations}`);
    }
    return new Response('unknown route', { status: 404 });
  },
};
