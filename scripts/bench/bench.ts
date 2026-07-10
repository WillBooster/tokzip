/**
 * Benchmark harness: size on the text channel, end-to-end speed, and lossless round-trip
 * verification, tokzip vs base64(brotli) / base64(gzip) / base64(zstd when available),
 * across size buckets, on the frozen benchmark split (`bench-v1`). Competitors' text
 * output is base64(binary) — the +33% text tax the design targets; tokzip's own output
 * is already text.
 *
 * Every benchmarked document is decompressed and compared with the original in both
 * modes, so a benchmark run doubles as a corpus-wide losslessness check: any mismatch is
 * reported and the process exits non-zero.
 *
 * Usage: bun scripts/bench/bench.ts [--speed] [--json <path>] [<language> ...]
 *   --speed        also measure compress/decompress throughput (MB/s) per language
 *   --json <path>  write a machine-readable report (consumed by CI and the dashboard)
 */
import { spawnSync } from 'node:child_process';
import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { compress, decompress } from '../../src/index.ts';
import '../../src/languages/index.ts';
import { CORPUS_DIR, type ManifestEntry } from '../corpus/shared.ts';
import { languageByName } from '../../src/dictionary.ts';
import { competitors } from './competitors.ts';

const MODES = ['fast', 'small'] as const;
type Mode = (typeof MODES)[number];

const BUCKETS = ['0.5k', '2k', '8k', '24k'] as const;

/** Column order shared by the console tables and the JSON report. */
const METHODS = ['tokzip fast', 'tokzip small', ...competitors.map((c) => c.name)];

const SPEED_TARGET_BYTES = 20_000_000;

interface BenchDoc {
  file: string;
  content: string;
  bucket: string;
}

/** Output chars per method, plus the input bytes they compressed. */
interface SizeTotals {
  docs: number;
  inputBytes: number;
  outputChars: number[];
}

interface SpeedResult {
  compressMBps: number;
  decompressMBps: number;
  megabytes: number;
}

interface LanguageReport {
  docs: number;
  registered: boolean;
  buckets: Record<string, { docs: number; inputBytes: number; ratios: Record<string, number> }>;
  total: { inputBytes: number; ratios: Record<string, number> };
  speed?: Record<Mode, SpeedResult>;
}

interface BenchReport {
  schemaVersion: 1;
  commit: string;
  timestamp: string;
  runtime: string;
  methods: string[];
  roundTrip: { docs: number; failures: string[] };
  languages: Record<string, LanguageReport>;
  /** Byte-weighted aggregate across all benchmarked languages. */
  total: { docs: number; inputBytes: number; ratios: Record<string, number> };
  speed?: Record<Mode, SpeedResult>;
}

function main(): void {
  const args = process.argv.slice(2);
  const speed = args.includes('--speed');
  const jsonIndex = args.indexOf('--json');
  const jsonPath = jsonIndex !== -1 ? args[jsonIndex + 1] : undefined;
  if (jsonIndex !== -1 && !jsonPath) {
    console.error('error: --json requires a path');
    process.exit(1);
  }
  const requested = args.filter((a, i) => !a.startsWith('--') && i !== jsonIndex + 1);
  const languages =
    requested.length > 0
      ? requested
      : readdirSync(CORPUS_DIR, { withFileTypes: true })
          .filter((e) => e.isDirectory() && !e.name.startsWith('.'))
          .map((e) => e.name)
          .toSorted();

  const report: BenchReport = {
    schemaVersion: 1,
    commit: process.env['GITHUB_SHA'] ?? gitOutput(['rev-parse', 'HEAD']) ?? 'unknown',
    timestamp: new Date().toISOString(),
    runtime: `bun ${Bun.version}`,
    methods: METHODS,
    roundTrip: { docs: 0, failures: [] },
    languages: {},
    total: { docs: 0, inputBytes: 0, ratios: {} },
  };
  const grandTotals = emptyTotals();
  const speedAccumulator = { fast: emptySpeedAccumulator(), small: emptySpeedAccumulator() };

  for (const language of languages) {
    benchLanguage(language, report, grandTotals);
    if (speed && report.languages[language]) benchSpeed(language, report, speedAccumulator);
  }

  report.total = { docs: grandTotals.docs, inputBytes: grandTotals.inputBytes, ratios: ratiosOf(grandTotals) };
  printTotals(report);
  if (speed) {
    report.speed = { fast: finishSpeed(speedAccumulator.fast), small: finishSpeed(speedAccumulator.small) };
    for (const mode of MODES) printSpeed(`ALL ${mode}`, report.speed[mode]);
  }
  printRoundTrip(report);
  if (jsonPath) {
    mkdirSync(dirname(jsonPath), { recursive: true });
    writeFileSync(jsonPath, JSON.stringify(report) + '\n');
    console.log(`\nwrote ${jsonPath}`);
  }
  if (report.roundTrip.failures.length > 0) process.exitCode = 1;
}

function benchLanguage(language: string, report: BenchReport, grandTotals: SizeTotals): void {
  const docs = benchDocsOf(language);
  if (docs.length === 0) {
    console.log(`\n${language}: no bench split (fetch + split the corpus first)`);
    return;
  }
  const registered = languageByName(language) !== undefined;
  console.log(`\n=== ${language} (${docs.length} bench docs${registered ? '' : ', id-0 fallback'}) ===`);
  const header = ['bucket', 'docs', 'input', ...METHODS];
  console.log(header.map((h) => h.padStart(15)).join(''));

  const languageReport: LanguageReport = {
    docs: docs.length,
    registered,
    buckets: {},
    total: { inputBytes: 0, ratios: {} },
  };
  const languageTotals = emptyTotals();
  for (const bucket of BUCKETS) {
    const bucketDocs = docs.filter((doc) => doc.bucket === bucket);
    if (bucketDocs.length === 0) continue;
    const totals = emptyTotals();
    for (const doc of bucketDocs) {
      benchDoc(language, registered, doc, totals, report.roundTrip);
    }
    accumulate(languageTotals, totals);
    languageReport.buckets[bucket] = { docs: totals.docs, inputBytes: totals.inputBytes, ratios: ratiosOf(totals) };
    printRow(bucket, totals);
  }
  accumulate(grandTotals, languageTotals);
  languageReport.total = { inputBytes: languageTotals.inputBytes, ratios: ratiosOf(languageTotals) };
  printRow('all', languageTotals);
  report.languages[language] = languageReport;
}

/** Compresses one document with every method, verifying tokzip output round-trips. */
function benchDoc(
  language: string,
  registered: boolean,
  doc: BenchDoc,
  totals: SizeTotals,
  roundTrip: BenchReport['roundTrip']
): void {
  const bytes = new TextEncoder().encode(doc.content);
  totals.docs += 1;
  totals.inputBytes += bytes.length;
  roundTrip.docs += 1;
  for (const [modeIndex, mode] of MODES.entries()) {
    const packed = compress(doc.content, { language: registered ? language : 'none', mode });
    totals.outputChars[modeIndex]! += packed.length;
    if (decompress(packed) !== doc.content) {
      roundTrip.failures.push(`${language}/${doc.file} (${mode})`);
      console.error(`ROUND-TRIP FAILURE: ${language}/${doc.file} (${mode})`);
    }
  }
  for (const [i, competitor] of competitors.entries()) {
    totals.outputChars[MODES.length + i]! += competitor.encodedLength(bytes);
  }
}

function benchSpeed(
  language: string,
  report: BenchReport,
  accumulator: Record<Mode, { megabytes: number; compressMs: number; decompressMs: number }>
): void {
  const docs = benchDocsOf(language);
  const payload = docs.map((d) => d.content).join('\n');
  const bytes = new TextEncoder().encode(payload);
  const registered = languageByName(language) ? language : 'none';
  const speed = {} as Record<Mode, SpeedResult>;
  for (const mode of MODES) {
    const iterations = Math.max(1, Math.ceil(SPEED_TARGET_BYTES / bytes.length));
    const started = performance.now();
    let packed = '';
    for (let i = 0; i < iterations; i++) packed = compress(payload, { language: registered, mode });
    const compressMs = performance.now() - started;
    const decodeStarted = performance.now();
    for (let i = 0; i < iterations; i++) decompress(packed);
    const decompressMs = performance.now() - decodeStarted;
    const megabytes = (bytes.length * iterations) / 1_048_576;
    speed[mode] = {
      compressMBps: round1(megabytes / (compressMs / 1000)),
      decompressMBps: round1(megabytes / (decompressMs / 1000)),
      megabytes: round1(megabytes),
    };
    accumulator[mode].megabytes += megabytes;
    accumulator[mode].compressMs += compressMs;
    accumulator[mode].decompressMs += decompressMs;
    printSpeed(`${language} ${mode}`, speed[mode]);
  }
  report.languages[language]!.speed = speed;
}

function benchDocsOf(language: string): BenchDoc[] {
  const dir = join(CORPUS_DIR, language);
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
}

function printTotals(report: BenchReport): void {
  console.log(`\n=== TOTAL (${report.total.docs} docs, ${Object.keys(report.languages).length} languages) ===`);
  const header = ['', 'docs', 'input', ...METHODS];
  console.log(header.map((h) => h.padStart(15)).join(''));
  printRow('all', {
    docs: report.total.docs,
    inputBytes: report.total.inputBytes,
    outputChars: METHODS.map((method) => report.total.ratios[method]! * report.total.inputBytes),
  });
}

function printRow(label: string, totals: SizeTotals): void {
  console.log(
    [
      label.padStart(15),
      String(totals.docs).padStart(15),
      String(totals.inputBytes).padStart(15),
      ...totals.outputChars.map((chars) => formatRatio(chars, totals.inputBytes).padStart(15)),
    ].join('')
  );
}

function printSpeed(label: string, speed: SpeedResult): void {
  console.log(
    `${label}: compress ${speed.compressMBps.toFixed(1)} MB/s, ` +
      `decompress ${speed.decompressMBps.toFixed(1)} MB/s (${speed.megabytes.toFixed(1)} MB)`
  );
}

function printRoundTrip(report: BenchReport): void {
  const { docs, failures } = report.roundTrip;
  console.log(
    failures.length === 0
      ? `\nround-trip: all ${docs} docs restored losslessly in both modes`
      : `\nround-trip: ${failures.length} FAILURES out of ${docs} docs:\n  ${failures.join('\n  ')}`
  );
}

function emptyTotals(): SizeTotals {
  return { docs: 0, inputBytes: 0, outputChars: METHODS.map(() => 0) };
}

function accumulate(target: SizeTotals, source: SizeTotals): void {
  target.docs += source.docs;
  target.inputBytes += source.inputBytes;
  for (const [i, chars] of source.outputChars.entries()) target.outputChars[i]! += chars;
}

function ratiosOf(totals: SizeTotals): Record<string, number> {
  return Object.fromEntries(METHODS.map((method, i) => [method, round4(totals.outputChars[i]! / totals.inputBytes)]));
}

function emptySpeedAccumulator(): { megabytes: number; compressMs: number; decompressMs: number } {
  return { megabytes: 0, compressMs: 0, decompressMs: 0 };
}

function finishSpeed(accumulated: { megabytes: number; compressMs: number; decompressMs: number }): SpeedResult {
  return {
    compressMBps: round1(accumulated.megabytes / (accumulated.compressMs / 1000)),
    decompressMBps: round1(accumulated.megabytes / (accumulated.decompressMs / 1000)),
    megabytes: round1(accumulated.megabytes),
  };
}

function formatRatio(outputChars: number, inputBytes: number): string {
  return ((outputChars / inputBytes) * 100).toFixed(1).padStart(6) + '%';
}

function gitOutput(args: string[]): string | undefined {
  const result = spawnSync('git', args, { encoding: 'utf8' });
  return result.status === 0 ? result.stdout.trim() : undefined;
}

const round1 = (value: number): number => Math.round(value * 10) / 10;
const round4 = (value: number): number => Math.round(value * 10_000) / 10_000;

main();
