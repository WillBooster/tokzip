import { Buffer } from 'node:buffer';
import { createCodec, type Codec } from '../../src/core.ts';
import wasmModule from '../../wasm/tokzip.wasm';
import { SAMPLES } from './samples.ts';

const codec = createCodec(() => wasmModule);
const warmedSamples = new Set<string>();
const MAX_ITERATIONS = 1000;

export default {
  fetch(request: Request): Response {
    const url = new URL(request.url);
    const [route, sample = ''] = url.pathname.split('/').filter(Boolean);
    if (!route || route === 'noop') return Response.json({ ok: true });
    const text = SAMPLES[sample];
    if (!text) return new Response('unknown sample', { status: 404 });
    const iterations = Math.min(Math.max(1, Math.trunc(Number(url.searchParams.get('iters')) || 1)), MAX_ITERATIONS);
    const fresh = route.startsWith('fresh-');
    const operation = route.replace('fresh-', '');
    if (operation !== 'compress' && operation !== 'decompress') return new Response('unknown route', { status: 404 });
    const warmed = warmedSamples.has(sample);
    const active = fresh ? createCodec(() => wasmModule) : codec;
    const count = fresh ? 1 : iterations;
    const input = operation === 'decompress' ? Buffer.from(url.searchParams.get('frame') ?? '', 'base64') : undefined;
    const { frame, ok } = measure(active, text, input, operation, count);
    if (!fresh) warmedSamples.add(sample);
    return Response.json({
      ok,
      warmed,
      fresh,
      iterations: count,
      bytes: Buffer.byteLength(text),
      packed: frame.length,
    });
  },
};

function measure(
  active: Codec,
  text: string,
  cached: Uint8Array | undefined,
  operation: 'compress' | 'decompress',
  count: number
): { frame: Uint8Array; ok: boolean } {
  let frame = cached ?? new Uint8Array();
  let ok = true;
  for (let i = 0; i < count; i++) {
    if (operation === 'compress') frame = active.compress(text);
    else ok &&= active.decompress(frame, { maxLength: Buffer.byteLength(text) }) === text;
  }
  return { frame, ok };
}
