import { type ChildProcess, spawn } from 'node:child_process';
import { createInterface } from 'node:readline';
import { setTimeout } from 'node:timers/promises';
import { z } from 'zod';
import { compress } from '../../src/index.ts';
import { SAMPLES } from './samples.ts';

const base = process.argv[2] ?? 'https://tokzip-bench-wasm.willbooster.workers.dev';
const wrangler = process.argv[3] ?? 'node_modules/wrangler/bin/wrangler.js';
const name = new URL(base).hostname.split('.')[0]!;
const frames = new Map(
  Object.entries(SAMPLES).map(([sample, text]) => [sample, Buffer.from(compress(text)).toString('base64')])
);
const resultSchema = z.object({
  ok: z.boolean(),
  isolateWarm: z.boolean(),
  warmed: z.boolean(),
  fresh: z.boolean(),
  iterations: z.number().int().positive(),
});
const tailEventSchema = z.object({
  event: z.object({ request: z.object({ url: z.string() }) }),
  cpuTime: z.number(),
});
interface TailSession {
  process: ChildProcess;
  events: Map<string, number>;
  stopping: boolean;
  failure?: Error;
}

class MissingCpuEvent extends Error {}

let sequence = 0;
let session = openTail();

try {
  // A uniquely identified probe confirms that tail is subscribed before any measured work.
  await subscribe();
  for (const sample of Object.keys(SAMPLES)) {
    for (const operation of ['compress', 'decompress', 'fresh-compress', 'fresh-decompress']) {
      const bytes = Buffer.byteLength(SAMPLES[sample]!);
      const iterations = operation.startsWith('fresh-') ? 1 : Math.max(1, Math.min(200, Math.floor(200_000 / bytes)));
      await measure(`/${operation}/${sample}?iters=${iterations}`);
    }
  }
} finally {
  closeTail();
}

async function subscribe(): Promise<void> {
  const deadline = Date.now() + 60_000;
  const probes = new Set<string>();
  while (Date.now() < deadline) {
    const url = urlFor('/noop');
    probes.add(new URL(url).pathname);
    await fetch(url).then((response) => response.arrayBuffer());
    await setTimeout(200);
    if (session.failure) throw session.failure;
    if ([...probes].some((path) => session.events.has(path))) return;
  }
  throw new Error('wrangler tail did not receive a probe');
}

async function measure(path: string): Promise<void> {
  for (let attempt = 0; attempt < 5; attempt++) {
    const url = urlFor(path);
    const start = performance.now();
    const response = await fetch(url);
    const ttfbMs = performance.now() - start;
    if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`);
    const result = resultSchema.parse(await response.json());
    if (!result.ok) throw new Error(`${path}: worker reported a failed round trip`);
    if (!result.isolateWarm || (!result.warmed && !result.fresh)) {
      continue;
    }
    let cpuMs: number;
    try {
      cpuMs = await cpuTime(url);
    } catch (error) {
      if (!(error instanceof MissingCpuEvent)) throw error;
      console.error(`${path}: missing tail event; reconnecting before retry`);
      closeTail();
      session = openTail();
      await subscribe();
      continue;
    }
    console.log(JSON.stringify({ path, cpuMs, cpuMsPerIteration: cpuMs / result.iterations, ttfbMs, ...result }));
    return;
  }
  throw new Error(`${path}: could not collect a prepared CPU measurement after five attempts`);
}

function urlFor(path: string): string {
  const url = new URL(path, base);
  const [operation, sample] = url.pathname.split('/').filter(Boolean);
  if (operation?.endsWith('decompress')) url.searchParams.set('frame', frames.get(sample!)!);
  // Tail redacts query values and UUID path segments; short sequence IDs survive.
  url.pathname += `/m${process.pid.toString(36)}-${sequence++}`;
  return url.href;
}

async function cpuTime(url: string): Promise<number> {
  const key = new URL(url).pathname;
  const deadline = Date.now() + 15_000;
  while (Date.now() < deadline) {
    if (session.failure) throw session.failure;
    const cpuMs = session.events.get(key);
    if (cpuMs !== undefined) {
      session.events.delete(key);
      return cpuMs;
    }
    await setTimeout(100);
  }
  throw new MissingCpuEvent(`no CPU event for ${url}`);
}

function openTail(): TailSession {
  const child = spawn('node', [wrangler, 'tail', name, '--format', 'json'], { stdio: ['ignore', 'pipe', 'inherit'] });
  const current: TailSession = { process: child, events: new Map(), stopping: false };
  child.on('error', (error) => {
    current.failure = error;
  });
  child.on('exit', (code) => {
    if (!current.stopping) current.failure = new Error(`wrangler tail exited with code ${code}`);
  });
  let pending = '';
  createInterface({ input: child.stdout }).on('line', (line) => {
    pending = line.startsWith('{') ? line : pending + line;
    let value: unknown;
    try {
      value = JSON.parse(pending);
    } catch {
      return;
    }
    pending = '';
    const event = tailEventSchema.safeParse(value);
    if (event.success) current.events.set(new URL(event.data.event.request.url).pathname, event.data.cpuTime);
  });
  return current;
}

function closeTail(): void {
  session.stopping = true;
  session.process.kill();
}
