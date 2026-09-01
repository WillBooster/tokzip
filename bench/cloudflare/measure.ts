/**
 * Drives a deployed bench worker and reports per-request CPU time from `wrangler tail`
 * alongside the client-side time-to-first-byte.
 *
 *   bun bench/cloudflare/measure.ts <worker-name> [<wrangler command>]
 */
import { spawn } from 'node:child_process';
import { createInterface } from 'node:readline';

const name = process.argv[2] ?? 'tokzip-bench-wasm';
const wrangler = process.argv[3] ?? 'wrangler';
const base = `https://${name}.willbooster.workers.dev`;
const ITERATIONS: Record<string, number> = { 'ja-prompt-1k': 200, 'ts-source-4k': 50, 'mixed-answer-20k': 10 };

let done = false;
const tail = spawn(wrangler, ['tail', name, '--format', 'json'], { stdio: ['ignore', 'pipe', 'inherit'] });
tail.on('error', (error) => {
  console.error(`failed to start \`${wrangler} tail\`: ${error.message}`);
  process.exit(1);
});
tail.on('exit', (code) => {
  if (!done) {
    console.error(`\`${wrangler} tail\` exited early with code ${code}`);
    process.exit(1);
  }
});
const events: { url: string; cpuMs: number }[] = [];
// `wrangler tail --format json` prints one pretty-printed object per event; objects are
// reassembled from lines (stream chunks are not line-aligned) and parsed at each top-level `}`.
let pending = '';
createInterface({ input: tail.stdout }).on('line', (line) => {
  pending += line;
  if (line !== '}') return;
  // Only optional fields are read, with defaults: an unexpected event shape yields a row without
  // a match (rejected below), so no schema validation is needed.
  const event = JSON.parse(pending) as { event?: { request?: { url?: string } }; cpuTime?: number };
  events.push({ url: event.event?.request?.url ?? '?', cpuMs: event.cpuTime ?? Number.NaN });
  pending = '';
});
await Bun.sleep(4000);

async function hit(path: string): Promise<{ ttfb: number; body: string }> {
  const started = performance.now();
  const response = await fetch(`${base}${path}`);
  // Time to first byte: measured when headers arrive, before the body is drained.
  const ttfb = performance.now() - started;
  const body = await response.text();
  // `fetch` resolves on 4xx/5xx, and the /decompress route reports a failed round-trip in its
  // body, so both are rejected here — a broken deployment must not print as a bench row.
  if (!response.ok || body.includes('MISMATCH')) {
    done = true;
    tail.kill();
    throw new Error(`${path}: HTTP ${response.status} ${body}`);
  }
  return { ttfb, body };
}

interface Row {
  label: string;
  ttfb: number;
  contaminated: boolean;
}
const rows: Row[] = [];
/** The first /load pays the module import + wasm instantiate and then compresses every sample. */
const COLD_LOAD = '/load (cold: import + compress all samples)';
const record = async (label: string, path: string): Promise<void> => {
  const { ttfb, body } = await hit(path);
  // Discard rows contaminated by a cold isolate: `cached=false` (a /decompress that missed the
  // /load-warmed isolate compressed once) or `warm=false` (this request paid the module import
  // + wasm instantiate). The first /load is meant to be cold, so it is never flagged.
  const contaminated = label !== COLD_LOAD && (body.includes('cached=false') || body.includes('warm=false'));
  rows.push({ label, ttfb, contaminated });
};
await record('/noop', '/noop');
await record(COLD_LOAD, '/load');
await record('/load (warm: compress all samples)', '/load');
for (const [sample, iterations] of Object.entries(ITERATIONS)) {
  await record(`/compress/${sample} x${iterations}`, `/compress/${sample}?iters=${iterations}`);
  await record(`/decompress/${sample} x${iterations}`, `/decompress/${sample}?iters=${iterations}`);
}
// Tail events trail their responses; wait (bounded) for one per request instead of a fixed delay.
const deadline = Date.now() + 15_000;
while (events.length < rows.length && Date.now() < deadline) await Bun.sleep(200);
done = true;
tail.kill();
const labelWidth = Math.max(...rows.map((row) => row.label.length));
console.log(
  `${name}: ${'request'.padEnd(labelWidth)} ${'ttfb ms'.padStart(9)} ${'cpu ms'.padStart(9)} ${'cpu ms/iter'.padStart(12)}`
);
for (const { label, ttfb, contaminated } of rows) {
  const path = label.split(' ')[0]!;
  const event = events.find((e) => e.url.includes(path) && !e.url.includes('#used'));
  if (!event) throw new Error(`no \`wrangler tail\` event for ${label}; the CPU column would be a guess`);
  event.url += '#used';
  const iterations = Number(/x(\d+)/.exec(label)?.[1] ?? 1);
  const cpu = event.cpuMs;
  const note = contaminated ? '  DISCARD (cold isolate: includes one compress)' : '';
  console.log(
    `  ${label.padEnd(labelWidth)} ${ttfb.toFixed(0).padStart(9)} ${cpu.toFixed(1).padStart(9)} ${(cpu / iterations).toFixed(3).padStart(12)}${note}`
  );
}
