/**
 * Text-channel compression benchmark on the seeded `bench-v1` corpus split.
 *
 * Size, lossless round-trip, and (with --speed) end-to-end per-document throughput are
 * measured for tokzip and every competitor. Binary codecs include base64url encode/decode
 * work and use unpadded URL-safe output, matching tokzip's intended transport.
 *
 * Usage: bun scripts/bench/bench.ts [--speed] [--json <path>] [<language> ...]
 */
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { languageByName } from '../../src/dictionary.ts';
import { compress, decompress } from '../../src/index.ts';
import '../../src/languages/index.ts';
import { CORPUS_DIRS, type ManifestEntry } from '../corpus.ts';
import { competitors } from './competitors.ts';

const BUCKETS = ['0.5k', '2k', '8k', '24k'] as const;
const SPEED_SAMPLE_COUNT = 3;
const SPEED_TARGET_BYTES = 8_000_000;

interface BenchDoc {
  file: string;
  content: string;
  bucket: string;
}

interface LoadedDoc extends BenchDoc {
  language: string;
  registered: boolean;
  inputBytes: number;
}

interface BenchMethod {
  name: string;
  compress(doc: LoadedDoc): string;
  decompress(encoded: string): string;
}

interface SizeTotals {
  docs: number;
  inputBytes: number;
  outputChars: number[];
}

interface SpeedResult {
  compressMBps: number;
  decompressMBps: number;
  compressKops: number;
  decompressKops: number;
  megabytesPerSample: number;
  samples: number;
}

interface LanguageReport {
  docs: number;
  registered: boolean;
  buckets: Record<string, { docs: number; inputBytes: number; ratios: Record<string, number> }>;
  total: { inputBytes: number; ratios: Record<string, number> };
}

interface BenchReport {
  schemaVersion: 2;
  commit: string;
  commitTimestamp: string;
  timestamp: string;
  runtime: string;
  methods: string[];
  corpus: { split: 'bench-v1'; sha256: string };
  roundTrip: { docs: number; methods: number; checks: number; failures: string[] };
  languages: Record<string, LanguageReport>;
  total: { docs: number; inputBytes: number; ratios: Record<string, number> };
  speed?: Record<string, SpeedResult>;
}

const METHODS: BenchMethod[] = [
  tokzipMethod('fast'),
  tokzipMethod('small'),
  ...competitors.map((competitor) => ({
    name: competitor.name,
    compress: (doc: LoadedDoc) => competitor.compress(doc.content),
    decompress: (encoded: string) => competitor.decompress(encoded),
  })),
];
const METHOD_NAMES = METHODS.map((method) => method.name);

function main(): void {
  const { speed, jsonPath, languages } = parseArgs(process.argv.slice(2));
  const report: BenchReport = {
    schemaVersion: 2,
    commit: process.env['GITHUB_SHA'] ?? gitOutput(['rev-parse', 'HEAD']) ?? 'unknown',
    commitTimestamp: new Date(gitOutput(['show', '-s', '--format=%cI', 'HEAD']) ?? Date.now()).toISOString(),
    timestamp: new Date().toISOString(),
    runtime: `bun ${Bun.version}`,
    methods: METHOD_NAMES,
    corpus: { split: 'bench-v1', sha256: '' },
    roundTrip: { docs: 0, methods: METHODS.length, checks: 0, failures: [] },
    languages: {},
    total: { docs: 0, inputBytes: 0, ratios: {} },
  };
  const grandTotals = emptyTotals();
  const loadedDocs: LoadedDoc[] = [];

  for (const language of languages) benchLanguage(language, report, grandTotals, loadedDocs);
  if (grandTotals.docs === 0) {
    console.error('error: no bench documents found (fetch + split the corpus first, or check the language name)');
    process.exit(1);
  }

  report.total = { docs: grandTotals.docs, inputBytes: grandTotals.inputBytes, ratios: ratiosOf(grandTotals) };
  report.corpus.sha256 = corpusHash(loadedDocs);
  printTotals(report);
  if (speed) report.speed = benchSpeed(loadedDocs);
  printRoundTrip(report);
  if (jsonPath) writeReport(jsonPath, report);
  if (report.roundTrip.failures.length > 0) process.exitCode = 1;
}

function parseArgs(args: string[]): { speed: boolean; jsonPath?: string; languages: string[] } {
  const speed = args.includes('--speed');
  const jsonIndex = args.indexOf('--json');
  const jsonPath = jsonIndex === -1 ? undefined : args[jsonIndex + 1];
  if (jsonIndex !== -1 && !jsonPath) {
    console.error('error: --json requires a path');
    process.exit(1);
  }
  const requested = args.filter((arg, index) => {
    return !arg.startsWith('--') && (jsonIndex === -1 || index !== jsonIndex + 1);
  });
  const languages =
    requested.length > 0
      ? requested
      : [
          ...new Set(
            CORPUS_DIRS.filter((corpusDir) => existsSync(corpusDir)).flatMap((corpusDir) =>
              readdirSync(corpusDir, { withFileTypes: true })
                .filter((entry) => entry.isDirectory() && !entry.name.startsWith('.'))
                .map((entry) => entry.name)
            )
          ),
        ].toSorted();
  return { speed, jsonPath, languages };
}

function benchLanguage(language: string, report: BenchReport, grandTotals: SizeTotals, allDocs: LoadedDoc[]): void {
  const docs = loadBenchDocs(language);
  if (docs.length === 0) {
    console.log(`\n${language}: no bench split (fetch + split the corpus first)`);
    return;
  }
  const registered = languageByName(language) !== undefined;
  const loaded = docs.map((doc) => ({
    ...doc,
    language,
    registered,
    inputBytes: Buffer.byteLength(doc.content),
  }));
  allDocs.push(...loaded);
  console.log(`\n=== ${language} (${docs.length} bench docs${registered ? '' : ', id-0 fallback'}) ===`);
  printHeader('bucket');

  const languageReport: LanguageReport = {
    docs: docs.length,
    registered,
    buckets: {},
    total: { inputBytes: 0, ratios: {} },
  };
  const languageTotals = emptyTotals();
  for (const bucket of BUCKETS) {
    const bucketDocs = loaded.filter((doc) => doc.bucket === bucket);
    if (bucketDocs.length === 0) continue;
    const totals = emptyTotals();
    for (const doc of bucketDocs) benchDoc(doc, totals, report.roundTrip);
    accumulate(languageTotals, totals);
    languageReport.buckets[bucket] = { docs: totals.docs, inputBytes: totals.inputBytes, ratios: ratiosOf(totals) };
    printRow(bucket, totals);
  }
  accumulate(grandTotals, languageTotals);
  languageReport.total = { inputBytes: languageTotals.inputBytes, ratios: ratiosOf(languageTotals) };
  printRow('all', languageTotals);
  report.languages[language] = languageReport;
}

function benchDoc(doc: LoadedDoc, totals: SizeTotals, roundTrip: BenchReport['roundTrip']): void {
  totals.docs += 1;
  totals.inputBytes += doc.inputBytes;
  roundTrip.docs += 1;
  for (const [index, method] of METHODS.entries()) {
    let failure: string | undefined;
    try {
      const encoded = method.compress(doc);
      totals.outputChars[index]! += encoded.length;
      roundTrip.checks += 1;
      if (method.decompress(encoded) !== doc.content) failure = `${doc.language}/${doc.file} (${method.name})`;
    } catch (error) {
      failure = `${doc.language}/${doc.file} (${method.name}): ${error}`;
    }
    if (failure !== undefined) {
      roundTrip.failures.push(failure);
      console.error(`ROUND-TRIP FAILURE: ${failure}`);
    }
  }
}

function benchSpeed(docs: LoadedDoc[]): Record<string, SpeedResult> {
  console.log(`\n=== END-TO-END SPEED (${SPEED_SAMPLE_COUNT} median samples, per-document framing) ===`);
  const inputBytes = docs.reduce((sum, doc) => sum + doc.inputBytes, 0);
  const iterations = Math.max(1, Math.ceil(SPEED_TARGET_BYTES / inputBytes));
  const processedBytes = inputBytes * iterations;
  const operations = docs.length * iterations;
  const result: Record<string, SpeedResult> = {};

  for (const method of METHODS) {
    const encoded = docs.map((doc) => method.compress(doc));
    // Warm both code paths without adding another expensive full q11 corpus pass.
    for (let index = 0; index < Math.min(32, docs.length); index++) method.decompress(encoded[index]!);
    const compressSamples = sampleTimes(() => {
      let chars = 0;
      for (let iteration = 0; iteration < iterations; iteration++) {
        for (const doc of docs) chars += method.compress(doc).length;
      }
      return chars;
    });
    const decompressSamples = sampleTimes(() => {
      let chars = 0;
      for (let iteration = 0; iteration < iterations; iteration++) {
        for (const value of encoded) chars += method.decompress(value).length;
      }
      return chars;
    });
    const compressMs = median(compressSamples);
    const decompressMs = median(decompressSamples);
    const speed = {
      compressMBps: round1(processedBytes / 1_048_576 / (compressMs / 1000)),
      decompressMBps: round1(processedBytes / 1_048_576 / (decompressMs / 1000)),
      compressKops: round1(operations / compressMs),
      decompressKops: round1(operations / decompressMs),
      megabytesPerSample: round1(processedBytes / 1_048_576),
      samples: SPEED_SAMPLE_COUNT,
    };
    result[method.name] = speed;
    printSpeed(method.name, speed);
  }
  return result;
}

function sampleTimes(operation: () => number): number[] {
  const times: number[] = [];
  let checksum = 0;
  for (let sample = 0; sample < SPEED_SAMPLE_COUNT; sample++) {
    const started = performance.now();
    checksum ^= operation();
    times.push(performance.now() - started);
  }
  // Retain an observable dependency on every operation result so engines cannot discard
  // the work while still keeping benchmark output deterministic.
  if (checksum === Number.MIN_SAFE_INTEGER) console.log(checksum);
  return times;
}

function loadBenchDocs(language: string): BenchDoc[] {
  return CORPUS_DIRS.flatMap((corpusDir) => {
    const dir = join(corpusDir, language);
    const manifestPath = join(dir, 'manifest.jsonl');
    if (!existsSync(manifestPath)) return [];
    return readFileSync(manifestPath, 'utf8')
      .split('\n')
      .filter((line) => line.trim())
      .map((line) => JSON.parse(line) as ManifestEntry)
      .filter((entry) => entry.split === 'bench')
      .map((entry) => ({
        file: entry.file,
        content: readFileSync(join(dir, entry.file), 'utf8'),
        bucket: entry.sizeBucket,
      }));
  });
}

function tokzipMethod(mode: 'fast' | 'small'): BenchMethod {
  return {
    name: `tokzip ${mode}`,
    compress: (doc) => compress(doc.content, { language: doc.registered ? doc.language : 'none', mode }),
    decompress: (encoded) => decompress(encoded) as string,
  };
}

function corpusHash(docs: LoadedDoc[]): string {
  const hash = createHash('sha256');
  for (const doc of docs) {
    hash.update(`${doc.language}\0${doc.file}\0${doc.bucket}\0${doc.inputBytes}\0`);
    hash.update(doc.content);
    hash.update('\0');
  }
  return hash.digest('hex');
}

function writeReport(path: string, report: BenchReport): void {
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, JSON.stringify(report) + '\n');
  console.log(`\nwrote ${path}`);
}

function printTotals(report: BenchReport): void {
  console.log(`\n=== TOTAL (${report.total.docs} docs, ${Object.keys(report.languages).length} languages) ===`);
  printHeader('');
  printRow('all', {
    docs: report.total.docs,
    inputBytes: report.total.inputBytes,
    outputChars: METHOD_NAMES.map((method) => report.total.ratios[method]! * report.total.inputBytes),
  });
}

function printHeader(firstColumn: string): void {
  console.log(
    [firstColumn, 'docs', 'input', ...METHOD_NAMES].map((value) => value.padStart(columnWidth(value))).join('')
  );
}

function printRow(label: string, totals: SizeTotals): void {
  console.log(
    [
      label.padStart(columnWidth(label)),
      String(totals.docs).padStart(columnWidth('docs')),
      String(totals.inputBytes).padStart(columnWidth('input')),
      ...totals.outputChars.map((chars, index) =>
        formatRatio(chars, totals.inputBytes).padStart(columnWidth(METHOD_NAMES[index]!))
      ),
    ].join('')
  );
}

function printSpeed(label: string, speed: SpeedResult): void {
  console.log(
    `${label.padEnd(Math.max(...METHOD_NAMES.map((name) => name.length)))}  ` +
      `compress ${speed.compressMBps.toFixed(1)} MB/s (${speed.compressKops.toFixed(1)} kdoc/s), ` +
      `decompress ${speed.decompressMBps.toFixed(1)} MB/s (${speed.decompressKops.toFixed(1)} kdoc/s)`
  );
}

function printRoundTrip(report: BenchReport): void {
  const { docs, methods, checks, failures } = report.roundTrip;
  console.log(
    failures.length === 0
      ? `\nround-trip: ${checks} checks passed (${docs} docs × ${methods} methods)`
      : `\nround-trip: ${failures.length} FAILURES from ${checks} checks:\n  ${failures.join('\n  ')}`
  );
}

function emptyTotals(): SizeTotals {
  return { docs: 0, inputBytes: 0, outputChars: METHODS.map(() => 0) };
}

function accumulate(target: SizeTotals, source: SizeTotals): void {
  target.docs += source.docs;
  target.inputBytes += source.inputBytes;
  for (const [index, chars] of source.outputChars.entries()) target.outputChars[index]! += chars;
}

function ratiosOf(totals: SizeTotals): Record<string, number> {
  return Object.fromEntries(
    METHOD_NAMES.map((method, index) => [method, round4(totals.outputChars[index]! / totals.inputBytes)])
  );
}

function columnWidth(value: string): number {
  return Math.max(10, value.length + 2);
}

function formatRatio(outputChars: number, inputBytes: number): string {
  return ((outputChars / inputBytes) * 100).toFixed(1) + '%';
}

function median(values: number[]): number {
  return values.toSorted((left, right) => left - right)[Math.floor(values.length / 2)]!;
}

function gitOutput(args: string[]): string | undefined {
  const result = spawnSync('git', args, { encoding: 'utf8' });
  return result.status === 0 ? result.stdout.trim() : undefined;
}

const round1 = (value: number): number => Math.round(value * 10) / 10;
const round4 = (value: number): number => Math.round(value * 10_000) / 10_000;

main();
