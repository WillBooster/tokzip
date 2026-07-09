/**
 * Benchmark harness: size on the text channel and end-to-end speed, tokzip vs
 * base64(brotli) / base64(gzip) / base64(zstd when available), across size buckets, on the
 * frozen benchmark split (`bench-v1`). Competitors' text output is base64(binary) — the +33%
 * text tax the design targets; tokzip's own output is already text.
 *
 * Usage: bun scripts/bench/bench.ts [--speed] [<language> ...]
 */
import { brotliCompressSync, constants as zlibConstants, gzipSync } from 'node:zlib';
import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { compress, decompress } from '../../src/index.ts';
import '../../src/languages/index.ts';
import { CORPUS_DIR, type ManifestEntry } from '../corpus/shared.ts';
import { languageByName } from '../../src/dictionary.ts';

interface Competitor {
  name: string;
  encode(bytes: Uint8Array): number;
}

// oxlint-disable-next-line no-explicit-any -- zstd is only in newer Node/Bun typings
const zstdCompressSync = (await import('node:zlib').then((m) => (m as any).zstdCompressSync)) as
  | ((data: Uint8Array, options?: unknown) => Uint8Array)
  | undefined;

const base64Length = (byteLength: number): number => Math.ceil(byteLength / 3) * 4;

const competitors: Competitor[] = [
  {
    name: 'b64(brotli q11)',
    encode: (bytes) =>
      base64Length(brotliCompressSync(bytes, { params: { [zlibConstants.BROTLI_PARAM_QUALITY]: 11 } }).length),
  },
  {
    name: 'b64(brotli q5)',
    encode: (bytes) =>
      base64Length(brotliCompressSync(bytes, { params: { [zlibConstants.BROTLI_PARAM_QUALITY]: 5 } }).length),
  },
  { name: 'b64(gzip -6)', encode: (bytes) => base64Length(gzipSync(bytes, { level: 6 }).length) },
];
if (zstdCompressSync) {
  // oxlint-disable-next-line no-explicit-any -- ZSTD_c_compressionLevel is missing from Bun's constants typings
  const levelKey = (zlibConstants as any).ZSTD_c_compressionLevel as number;
  const zstd = (bytes: Uint8Array, level: number): number =>
    base64Length(zstdCompressSync(bytes, { params: { [levelKey]: level } }).length);
  competitors.push(
    { name: 'b64(zstd -19)', encode: (bytes) => zstd(bytes, 19) },
    { name: 'b64(zstd -3)', encode: (bytes) => zstd(bytes, 3) }
  );
} else {
  console.error('note: node:zlib zstd not available in this runtime; skipping zstd baselines');
}

const BUCKETS = ['0.5k', '2k', '8k', '24k'] as const;

function benchDocsOf(language: string): { content: string; bucket: string }[] {
  const dir = join(CORPUS_DIR, language);
  const manifestPath = join(dir, 'manifest.jsonl');
  if (!existsSync(manifestPath)) return [];
  return readFileSync(manifestPath, 'utf8')
    .split('\n')
    .filter((line) => line.trim())
    .map((line) => JSON.parse(line) as ManifestEntry)
    .filter((entry) => entry.split === 'bench')
    .map((entry) => ({ content: readFileSync(join(dir, entry.file), 'utf8'), bucket: entry.sizeBucket }));
}

function formatRatio(outputChars: number, inputBytes: number): string {
  return ((outputChars / inputBytes) * 100).toFixed(1).padStart(6) + '%';
}

function benchLanguage(language: string): void {
  const docs = benchDocsOf(language);
  if (docs.length === 0) {
    console.log(`\n${language}: no bench split (fetch + split the corpus first)`);
    return;
  }
  const registered = languageByName(language) !== undefined;
  console.log(`\n=== ${language} (${docs.length} bench docs${registered ? '' : ', id-0 fallback'}) ===`);
  const header = ['bucket', 'docs', 'input', 'tokzip fast', 'tokzip small', ...competitors.map((c) => c.name)];
  console.log(header.map((h) => h.padStart(15)).join(''));
  for (const bucket of BUCKETS) {
    const bucketDocs = docs.filter((doc) => doc.bucket === bucket);
    if (bucketDocs.length === 0) continue;
    let input = 0;
    let fastTotal = 0;
    let smallTotal = 0;
    const competitorTotals = competitors.map(() => 0);
    for (const doc of bucketDocs) {
      const bytes = new TextEncoder().encode(doc.content);
      input += bytes.length;
      fastTotal += compress(doc.content, { language: registered ? language : 'none', mode: 'fast' }).length;
      smallTotal += compress(doc.content, { language: registered ? language : 'none', mode: 'small' }).length;
      for (const [i, competitor] of competitors.entries()) competitorTotals[i]! += competitor.encode(bytes);
    }
    console.log(
      [
        bucket.padStart(15),
        String(bucketDocs.length).padStart(15),
        String(input).padStart(15),
        formatRatio(fastTotal, input).padStart(15),
        formatRatio(smallTotal, input).padStart(15),
        ...competitorTotals.map((total) => formatRatio(total, input).padStart(15)),
      ].join('')
    );
  }
}

function benchSpeed(language: string): void {
  const docs = benchDocsOf(language);
  if (docs.length === 0) return;
  const payload = docs.map((d) => d.content).join('\n');
  const bytes = new TextEncoder().encode(payload);
  const registered = languageByName(language) ? language : 'none';
  for (const mode of ['fast', 'small'] as const) {
    const started = performance.now();
    let packed = '';
    const iterations = Math.max(1, Math.ceil(20_000_000 / bytes.length));
    for (let i = 0; i < iterations; i++) packed = compress(payload, { language: registered, mode });
    const encodeMs = performance.now() - started;
    const decodeStarted = performance.now();
    for (let i = 0; i < iterations; i++) decompress(packed);
    const decodeMs = performance.now() - decodeStarted;
    const mb = (bytes.length * iterations) / 1_048_576;
    console.log(
      `${language} ${mode}: compress ${(mb / (encodeMs / 1000)).toFixed(1)} MB/s, decompress ${(mb / (decodeMs / 1000)).toFixed(1)} MB/s (${mb.toFixed(1)} MB)`
    );
  }
}

const args = process.argv.slice(2);
const speed = args.includes('--speed');
const requested = args.filter((a) => !a.startsWith('--'));
const languages =
  requested.length > 0
    ? requested
    : readdirSync(CORPUS_DIR, { withFileTypes: true })
        .filter((e) => e.isDirectory() && !e.name.startsWith('.'))
        .map((e) => e.name);
for (const language of languages) {
  benchLanguage(language);
  if (speed) benchSpeed(language);
}
