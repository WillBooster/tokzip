/**
 * Cloudflare Workers benchmark for the wasm build. Workers freeze the clock during CPU work,
 * so this worker only does the work; times come from `wrangler tail --format json` (per-request
 * `cpuTime`) and from the client's time-to-first-byte.
 *
 *   /noop                       nothing (baseline)
 *   /load                       import the module and pre-compress every sample
 *   /compress/<sample>?iters=N  compress the sample N times
 *   /decompress/<sample>?iters=N
 *
 *   wrangler deploy --config bench/cloudflare/wrangler.jsonc
 *   bun bench/cloudflare/measure.ts
 */
import samples from './samples.json';

const SAMPLES: Record<string, string> = samples;
/** Frames pre-compressed by /load so the /decompress route never times a compression. */
const FRAMES = new Map<string, Uint8Array>();
const MAX_ITERATIONS = 1000;

export default {
  async fetch(request: Request): Promise<Response> {
    const url = new URL(request.url);
    const [route, sample = ''] = url.pathname.split('/').filter(Boolean);
    if (route === undefined || route === 'noop') return new Response('ok');
    const codec = await import('../../src/index.ts');
    if (route === 'load') {
      for (const [name, text] of Object.entries(SAMPLES)) FRAMES.set(name, codec.compress(text));
      return new Response(`loaded: ${FRAMES.size} samples`);
    }
    const text = SAMPLES[sample];
    if (!text) return new Response('unknown sample', { status: 404 });
    // Untrusted, so clamped: an unbounded iteration count would run until the CPU limit.
    const iterations = Math.min(Math.max(1, Math.trunc(Number(url.searchParams.get('iters')) || 1)), MAX_ITERATIONS);
    if (route === 'compress') {
      let size = 0;
      for (let i = 0; i < iterations; i++) size = codec.compress(text).length;
      return new Response(`${sample}: ${text.length} chars -> ${size} bytes x${iterations}`);
    }
    if (route === 'decompress') {
      // Pre-compressed by /load (which the driver calls first) so this request times only
      // decompression; the fallback keeps the route usable when hit directly.
      const frame = FRAMES.get(sample) ?? codec.compress(text);
      let ok = true;
      for (let i = 0; i < iterations; i++) ok &&= codec.decompress(frame) === text;
      return new Response(`${sample}: ${ok ? 'round-trip ok' : 'MISMATCH'} x${iterations}`);
    }
    return new Response('unknown route', { status: 404 });
  },
};
