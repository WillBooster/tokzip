/**
 * Compression benchmark on the corpus bench split: tokzip (the committed wasm build) against
 * brotli -11, zstd -19, and gzip -9, as size ratios per language and per size bucket, with a
 * lossless round-trip check of every tokzip frame.
 *
 *   bun scripts/bench/bench.ts [--speed] [--json <out.json>]
 */
import { execSync } from 'node:child_process';
import { existsSync, mkdirSync, readdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { brotliCompressSync, constants as zlibConstants, gzipSync } from 'node:zlib';
import { compress, decompress } from '../../src/index.ts';
import { corpusDirs, type ManifestEntry } from '../corpus.ts';

interface Doc {
  language: string;
  bucket: string;
  content: string;
  bytes: number;
}

interface Method {
  name: string;
  compress(content: string): Uint8Array;
}

interface Totals {
  docs: number;
  raw: number;
  packed: Record<string, number>;
}

const BUCKETS: [number, string][] = [
  [1024, '≤1K'],
  [4096, '≤4K'],
  [16_384, '≤16K'],
  [Number.POSITIVE_INFINITY, '>16K'],
];

// oxlint-disable-next-line no-explicit-any -- zstd is only in newer Node/Bun typings
const zlib = (await import('node:zlib')) as any;
const zstdCompressSync = zlib.zstdCompressSync as
  | ((data: Uint8Array, options: { params: Record<number, number> }) => Uint8Array)
  | undefined;

const METHODS: Method[] = [
  { name: 'tokzip', compress: (content) => compress(content) },
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
  const args = process.argv.slice(2);
  const speed = args.includes('--speed');
  const jsonIndex = args.indexOf('--json');
  const jsonPath = jsonIndex === -1 ? undefined : args[jsonIndex + 1];

  const docs = loadBenchDocs();
  if (docs.length === 0) throw new Error('no bench documents found (check the corpus checkout)');
  const groups = new Map<string, Totals>();
  const add = (key: string, doc: Doc, sizes: Record<string, number>): void => {
    const totals = groups.get(key) ?? { docs: 0, raw: 0, packed: {} };
    totals.docs++;
    totals.raw += doc.bytes;
    for (const [name, size] of Object.entries(sizes)) totals.packed[name] = (totals.packed[name] ?? 0) + size;
    groups.set(key, totals);
  };
  let compressMs = 0;
  let decompressMs = 0;
  for (const doc of docs) {
    const sizes: Record<string, number> = {};
    for (const method of METHODS) sizes[method.name] = method.compress(doc.content).length;
    const started = performance.now();
    const frame = compress(doc.content);
    const compressed = performance.now();
    const restored = decompress(frame);
    decompressMs += performance.now() - compressed;
    compressMs += compressed - started;
    if (restored !== doc.content) throw new Error(`tokzip round-trip mismatch on a ${doc.language} document`);
    add(`language ${doc.language}`, doc, sizes);
    add(`bucket ${doc.bucket}`, doc, sizes);
    add('all', doc, sizes);
  }

  const names = METHODS.map((method) => method.name);
  console.log(
    ['group'.padEnd(22), 'docs'.padStart(6), 'bytes'.padStart(10), ...names.map((n) => n.padStart(11))].join(' ')
  );
  const report: Record<string, unknown> = { commit: gitCommit(), wasmBytes: wasmSize(), groups: {} };
  for (const [key, totals] of [...groups.entries()].toSorted(([a], [b]) => a.localeCompare(b))) {
    const ratios = Object.fromEntries(names.map((n) => [n, totals.packed[n]! / totals.raw]));
    (report['groups'] as Record<string, unknown>)[key] = { ...totals, ratios };
    console.log(
      [
        key.padEnd(22),
        String(totals.docs).padStart(6),
        String(totals.raw).padStart(10),
        ...names.map((n) => `${(100 * ratios[n]!).toFixed(1)}%`.padStart(11)),
      ].join(' ')
    );
  }
  const totalMb = groups.get('all')!.raw / 1e6;
  if (speed) {
    const speeds = { compressMBps: totalMb / (compressMs / 1000), decompressMBps: totalMb / (decompressMs / 1000) };
    report['speed'] = speeds;
    console.log(
      `\ntokzip speed: compress ${speeds.compressMBps.toFixed(1)} MB/s, decompress ${speeds.decompressMBps.toFixed(1)} MB/s`
    );
  }
  console.log(`\nround-trip: ${docs.length} tokzip frames verified; wasm module ${report['wasmBytes']} bytes`);
  if (jsonPath) {
    mkdirSync(dirname(jsonPath), { recursive: true });
    writeFileSync(jsonPath, JSON.stringify(report, undefined, 2));
  }
}

function loadBenchDocs(): Doc[] {
  const docs: Doc[] = [];
  for (const corpusDir of corpusDirs()) {
    for (const language of readdirLanguages(corpusDir)) {
      const dir = join(corpusDir, language);
      const manifestPath = join(dir, 'manifest.jsonl');
      if (!existsSync(manifestPath)) continue;
      for (const line of readFileSync(manifestPath, 'utf8').split('\n')) {
        if (!line.trim()) continue;
        const entry = JSON.parse(line) as ManifestEntry;
        if (entry.split !== 'bench') continue;
        const content = readFileSync(join(dir, entry.file), 'utf8');
        const bytes = Buffer.byteLength(content);
        docs.push({ language, bucket: BUCKETS.find(([limit]) => bytes <= limit)![1], content, bytes });
      }
    }
  }
  return docs;
}

function readdirLanguages(corpusDir: string): string[] {
  return readdirSync(corpusDir).filter((name) => statSync(join(corpusDir, name)).isDirectory());
}

function gitCommit(): string {
  return execSync('git rev-parse HEAD', { encoding: 'utf8' }).trim();
}

function wasmSize(): number {
  return readFileSync(join(import.meta.dir, '../../wasm/tokzip.wasm')).length;
}

main();
