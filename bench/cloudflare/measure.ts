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
  const event = JSON.parse(pending) as { event?: { request?: { url?: string } }; cpuTime?: number };
  events.push({ url: event.event?.request?.url ?? '?', cpuMs: event.cpuTime ?? Number.NaN });
  pending = '';
});
await Bun.sleep(4000);

async function hit(path: string): Promise<number> {
  const started = performance.now();
  const response = await fetch(`${base}${path}`);
  // Time to first byte: measured when headers arrive, before the body is drained.
  const ttfb = performance.now() - started;
  await response.text();
  return ttfb;
}

const rows: [string, number][] = [
  ['/noop', await hit('/noop')],
  ['/load (first request)', await hit('/load')],
  ['/load (warm)', await hit('/load')],
];
for (const [sample, iterations] of Object.entries(ITERATIONS)) {
  rows.push([`/compress/${sample} x${iterations}`, await hit(`/compress/${sample}?iters=${iterations}`)]);
  rows.push([`/decompress/${sample} x${iterations}`, await hit(`/decompress/${sample}?iters=${iterations}`)]);
}
await Bun.sleep(3000);
done = true;
tail.kill();
console.log(
  `${name}: ${'request'.padEnd(36)} ${'ttfb ms'.padStart(9)} ${'cpu ms'.padStart(9)} ${'cpu ms/iter'.padStart(12)}`
);
for (const [label, ttfb] of rows) {
  const path = label.split(' ')[0]!;
  const event = events.find((e) => e.url.includes(path.replace(/ .*/, '')) && !e.url.includes('#used'));
  if (event) event.url += '#used';
  const iterations = Number(/x(\d+)/.exec(label)?.[1] ?? 1);
  const cpu = event?.cpuMs ?? Number.NaN;
  console.log(
    `  ${label.padEnd(36)} ${ttfb.toFixed(0).padStart(9)} ${cpu.toFixed(1).padStart(9)} ${(cpu / iterations).toFixed(3).padStart(12)}`
  );
}
