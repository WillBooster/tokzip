/**
 * Compression benchmark on the corpus bench split: tokzip (the committed wasm build) against
 * brotli -11, zstd -19, and gzip -9, as size ratios per language and per size bucket, with a
 * lossless round-trip check of every tokzip frame.
 *
 *   bun scripts/bench/bench.ts [--speed] [--codec-only] [--repeat <count>] [--input <documents.jsonl>] [--json <out.json>]
 */
import { execSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { parseArgs } from 'node:util';
import { brotliCompressSync, constants as zlibConstants, gzipSync } from 'node:zlib';
import { compress, decompress } from '../../src/index.ts';
import { type Doc, loadBenchDocs } from './documents.ts';
import { summarizeTimings, type Timings } from './timings.ts';

interface Method {
  name: string;
  compress(content: string): Uint8Array;
}

interface Totals {
  docs: number;
  raw: number;
  packed: Record<string, number>;
  timings: Timings;
}

// oxlint-disable-next-line no-explicit-any -- zstd is only in newer Node/Bun typings
const zlib = (await import('node:zlib')) as any;
const zstdCompressSync = zlib.zstdCompressSync as
  | ((data: Uint8Array, options: { params: Record<number, number> }) => Uint8Array)
  | undefined;

const METHODS: Method[] = [
  {
    name: 'brotli -11',
    compress: (content) => brotliCompressSync(content, { params: { [zlibConstants.BROTLI_PARAM_QUALITY]: 11 } }),
  },
  ...(zstdCompressSync
    ? [
        {
          name: 'zstd -19',
          compress: (content: string) =>
            zstdCompressSync(new TextEncoder().encode(content), {
              params: { [zlib.constants.ZSTD_c_compressionLevel]: 19 },
            }),
        },
      ]
    : []),
  { name: 'gzip -9', compress: (content) => gzipSync(content, { level: 9 }) },
];

function main(): void {
  const { values } = parseArgs({
    args: process.argv.slice(2),
    options: {
      speed: { type: 'boolean' },
      'codec-only': { type: 'boolean' },
      repeat: { type: 'string', default: '1' },
      json: { type: 'string' },
      input: { type: 'string' },
    },
  });
  const speed = values.speed;
  const methods = values['codec-only'] ? [] : METHODS;
  const repeat = Number(values.repeat);
  if (!Number.isSafeInteger(repeat) || repeat < 1 || repeat > 100) {
    throw new Error('--repeat must be an integer between 1 and 100');
  }
  const jsonPath = values.json;
  const docs = loadBenchDocs(values.input);
  if (docs.length === 0) throw new Error('no benchmark documents found');
  const groups = measureDocuments(docs, methods, repeat);
  const report = printReport(docs, groups, ['tokzip', ...methods.map((method) => method.name)], repeat, speed ?? false);
  if (jsonPath) {
    mkdirSync(dirname(jsonPath), { recursive: true });
    writeFileSync(jsonPath, JSON.stringify(report, undefined, 2));
  }
}

function measureDocuments(docs: Doc[], methods: Method[], repeat: number): Map<string, Totals> {
  const groups = new Map<string, Totals>();
  // Repeated measurements exclude first-use initialization; the default still includes it.
  if (repeat > 1) for (const doc of docs) compress(doc.content);
  for (const doc of docs) {
    let frame: Uint8Array = new Uint8Array();
    const timings: Timings = { compress: [], decompress: [] };
    for (let round = 0; round < repeat; round++) {
      const started = performance.now();
      frame = compress(doc.content);
      const compressed = performance.now();
      const restored = decompress(frame);
      timings.decompress.push(performance.now() - compressed);
      timings.compress.push(compressed - started);
      if (restored !== doc.content) throw new Error(`tokzip round-trip mismatch at ${doc.location}`);
    }
    const sizes: Record<string, number> = { tokzip: frame.length };
    for (const method of methods) sizes[method.name] = method.compress(doc.content).length;
    for (const key of [`corpus ${doc.corpus}`, `language ${doc.language}`, `bucket ${doc.bucket}`, 'all']) {
      addDocument(groups, key, doc, sizes, timings);
    }
  }

  return groups;
}

function printReport(
  docs: Doc[],
  groups: Map<string, Totals>,
  names: string[],
  repeat: number,
  speed: boolean
): Record<string, unknown> {
  console.log(
    ['group'.padEnd(22), 'docs'.padStart(6), 'bytes'.padStart(10), ...names.map((n) => n.padStart(11))].join(' ')
  );
  const fingerprint = createHash('sha256');
  for (const doc of docs) fingerprint.update(JSON.stringify([doc.corpus, doc.language, doc.content]));
  const report: Record<string, unknown> = {
    commit: gitCommit(),
    ...wasmInfo(),
    corpusFingerprint: fingerprint.digest('hex'),
    runtime: process.versions,
    repeat,
    groups: {},
  };
  for (const [key, totals] of [...groups.entries()].toSorted(([a], [b]) => a.localeCompare(b))) {
    const ratios = Object.fromEntries(names.map((n) => [n, totals.packed[n]! / totals.raw]));
    (report['groups'] as Record<string, unknown>)[key] = {
      docs: totals.docs,
      raw: totals.raw,
      packed: totals.packed,
      ratios,
      ...(speed ? { speed: summarizeTimings(totals.raw * repeat, totals.timings) } : {}),
    };
    console.log(
      [
        key.padEnd(22),
        String(totals.docs).padStart(6),
        String(totals.raw).padStart(10),
        ...names.map((n) => `${(100 * ratios[n]!).toFixed(1)}%`.padStart(11)),
      ].join(' ')
    );
  }
  if (speed) {
    const total = groups.get('all')!;
    const speeds = summarizeTimings(total.raw * repeat, total.timings);
    report['speed'] = speeds;
    console.log(
      `\ntokzip speed: compress ${speeds.compressMBps.toFixed(1)} MB/s, decompress ${speeds.decompressMBps.toFixed(1)} MB/s`
    );
  }
  console.log(`\nround-trip: ${docs.length * repeat} tokzip frames verified; wasm module ${report['wasmBytes']} bytes`);
  return report;
}

function addDocument(
  groups: Map<string, Totals>,
  key: string,
  doc: Doc,
  sizes: Record<string, number>,
  timings: Timings
): void {
  const totals = groups.get(key) ?? { docs: 0, raw: 0, packed: {}, timings: { compress: [], decompress: [] } };
  totals.docs++;
  totals.raw += doc.bytes;
  for (const [name, size] of Object.entries(sizes)) totals.packed[name] = (totals.packed[name] ?? 0) + size;
  totals.timings.compress.push(...timings.compress);
  totals.timings.decompress.push(...timings.decompress);
  groups.set(key, totals);
}

function gitCommit(): string {
  return execSync('git rev-parse HEAD', { encoding: 'utf8' }).trim();
}

function wasmInfo(): { wasmBytes: number; wasmSha256: string } {
  const wasm = readFileSync(join(import.meta.dir, '../../wasm/tokzip.wasm'));
  return { wasmBytes: wasm.length, wasmSha256: createHash('sha256').update(wasm).digest('hex') };
}

main();
