import { expect, test } from 'bun:test';
import { spawnSync } from 'node:child_process';
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

import { z } from 'zod';

const percentiles = z.object({
  p50: z.number().nonnegative(),
  p95: z.number().nonnegative(),
  p99: z.number().nonnegative(),
  max: z.number().nonnegative(),
});
const groupSchema = z.object({
  docs: z.number(),
  raw: z.number(),
  packed: z.object({ tokzip: z.number() }),
  speed: z.object({
    compressMBps: z.number().positive(),
    decompressMBps: z.number().positive(),
    compressMs: percentiles,
    decompressMs: percentiles,
  }),
});

test('benchmarks JSONL records and identifies invalid input without copying content', () => {
  mkdirSync('.tmp', { recursive: true });
  const directory = mkdtempSync('.tmp/benchmark-');
  try {
    const documents = [
      '{"text":"large JSON exponent stays byte-for-byte intact","value":1e999}',
      JSON.stringify({ text: '日本語の回答です。'.repeat(20), reasoning: 'A reason.\n'.repeat(20) }),
      JSON.stringify({
        text: 'export const answer = 42;\n'.repeat(200),
        toolCalls: [{ name: 'read_file', input: { path: 'example.ts' } }],
      }),
    ];
    const input = join(directory, 'input.jsonl');
    const output = join(directory, 'report.json');
    writeFileSync(input, documents.join('\r\n') + '\r\n');
    const result = spawnSync(
      process.execPath,
      ['scripts/bench/bench.ts', '--codec-only', '--speed', '--repeat', '2', '--input', input, '--json', output],
      { encoding: 'utf8' }
    );
    expect(result.status, result.stderr).toBe(0);
    const report = z
      .object({ groups: z.record(z.string(), groupSchema), corpusFingerprint: z.string() })
      .parse(JSON.parse(readFileSync(output, 'utf8')));
    const total = report.groups['all']!;
    expect(total.docs).toBe(documents.length);
    expect(total.raw).toBe(documents.reduce((sum, document) => sum + Buffer.byteLength(document), 0));
    expect(total.packed.tokzip).toBeLessThan(total.raw);
    expect(report.groups['corpus input']).toEqual(total);
    expect(
      Object.entries(report.groups)
        .filter(([key]) => key.startsWith('bucket '))
        .reduce((sum, [, group]) => sum + group.docs, 0)
    ).toBe(total.docs);
    for (const latency of [total.speed.compressMs, total.speed.decompressMs]) {
      expect(latency.p50).toBeLessThanOrEqual(latency.p95);
      expect(latency.p95).toBeLessThanOrEqual(latency.p99);
      expect(latency.p99).toBeLessThanOrEqual(latency.max);
    }
    const serialized = readFileSync(output, 'utf8');
    expect(serialized).not.toContain('read_file');
    expect(report.corpusFingerprint).toMatch(/^[a-f\d]{64}$/);
    writeFileSync(input, `${documents[0]}\r\n\r\n{"text":"DO_NOT_INCLUDE_RECORD"\r\n`);
    const failure = spawnSync(process.execPath, ['scripts/bench/bench.ts', '--codec-only', '--input', input], {
      encoding: 'utf8',
    });
    expect(failure.status).not.toBe(0);
    expect(failure.stderr).toContain(`${input}:3`);
    expect(failure.stderr).not.toContain('DO_NOT_INCLUDE_RECORD');
    expect(failure.stdout).not.toContain('round-trip:');
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
});
