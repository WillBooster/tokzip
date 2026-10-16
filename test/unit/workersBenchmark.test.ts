import { expect, test } from 'bun:test';
import { appendFileSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { z } from 'zod';

// Live Workers placement and tail delivery cannot deterministically produce these failure cases.
test('names a failed round-trip row and stops without reporting it as a measurement', async () => {
  const result = await runDriver('mismatch');
  expect(result.code).not.toBe(0);
  expect(result.stderr).toContain('/decompress/ts-source-4k');
  expect(result.stderr).toContain('failed round trip');
  expect(result.stdout).not.toContain('"path":"/decompress/ts-source-4k');
}, 15_000);

test('reports fresh-instance CPU only after isolate startup has been excluded', async () => {
  const result = await runDriver('cold');
  expect(result.code, result.stderr).toBe(0);
  const rows = z.array(z.object({ path: z.string(), cpuMs: z.number() })).parse(
    result.stdout
      .trim()
      .split('\n')
      .map((line) => JSON.parse(line))
  );
  expect(rows).toHaveLength(20);
  const fresh = rows.find((row) => row.path.startsWith('/fresh-compress/ja-prompt-1k'));
  expect(fresh?.cpuMs).toBe(2.5);
  expect(rows.some((row) => row.cpuMs === 1000)).toBe(false);
}, 15_000);

async function runDriver(mode: 'mismatch' | 'cold'): Promise<{ code: number; stdout: string; stderr: string }> {
  mkdirSync('.tmp', { recursive: true });
  const directory = mkdtempSync('.tmp/workers-benchmark-');
  const eventFile = join(directory, 'events.jsonl');
  writeFileSync(eventFile, '');
  let firstFresh = true;
  const server = Bun.serve({
    port: 0,
    hostname: '127.0.0.1',
    fetch(request) {
      const url = new URL(request.url);
      const [operation = '', sample] = url.pathname.split('/').filter(Boolean);
      const isolateWarm = !(mode === 'cold' && operation === 'fresh-compress' && firstFresh);
      if (operation === 'fresh-compress') firstFresh = false;
      appendFileSync(eventFile, JSON.stringify({ url: request.url, cpuMs: isolateWarm ? 2.5 : 1000 }) + '\n');
      return Response.json({
        ok: !(mode === 'mismatch' && operation === 'decompress' && sample === 'ts-source-4k'),
        isolateWarm,
        warmed: true,
        fresh: operation.startsWith('fresh-'),
        iterations: Number(url.searchParams.get('iters') ?? 1),
      });
    },
  });
  const child = Bun.spawn(['node', 'bench/cloudflare/measure.ts', server.url.href, 'test/fixtures/benchmarkTail.mjs'], {
    env: { ...process.env, TOKZIP_BENCHMARK_EVENTS: eventFile },
    stdout: 'pipe',
    stderr: 'pipe',
  });
  try {
    const [code, stdout, stderr] = await Promise.all([
      child.exited,
      new Response(child.stdout).text(),
      new Response(child.stderr).text(),
    ]);
    return { code, stdout, stderr };
  } finally {
    child.kill();
    await server.stop(true);
    rmSync(directory, { recursive: true, force: true });
  }
}
