/**
 * Compression benchmark on the seeded `bench-v2` corpus split.
 *
 * The primary metric is the **session-amortized, dictionary-inclusive ratio**: each
 * language's bench docs are treated as one client session, and tokzip's per-language
 * dictionary module is charged once per session at its brotli-compressed transfer size
 * (what a CDN actually ships; competitors carry no dictionary). Short documents (≤ 4 KB,
 * the primary workload) are additionally reported as their own session. The classic
 * dictionary-free ratio stays as a secondary metric.
 *
 * Size, lossless round-trip, and (with --speed) end-to-end per-document throughput are
 * measured for tokzip and every competitor on one of two channels:
 * - text (default): tokzip text frames vs binary codecs behind unpadded base64url — every
 *   method pays its complete binary-to-text framing cost, matching a text transport.
 * - binary (--binary): tokzip binary frames vs the raw codec bytes, no text framing.
 *
 * Usage: bun scripts/bench/bench.ts [--binary] [--speed] [--json <path>] [<language> ...]
 * The corpus roots come from corpusDirs(); set TOKZIP_CORPUS_DIR to benchmark one corpus.
 */
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { brotliCompressSync, constants as zlibConstants } from 'node:zlib';
import { languageByName } from '../../src/dictionary.ts';
import { FenceTracker } from '../../src/fences.ts';
import { FLAG_FENCED } from '../../src/format.ts';
import { compress, decompress } from '../../src/index.ts';
import { LANGUAGE_IDS } from '../../src/languageIds.ts';
import { RADIX64_ALPHABET } from '../../src/radix64.ts';
import '../../src/languages/index.ts';
import { corpusDirs, type ManifestEntry } from '../corpus.ts';
import { binaryCompetitors, competitors } from './competitors.ts';

const BUCKETS = ['0.25k', '0.5k', '2k', '8k', '24k'] as const;
/** Buckets forming the short-document (≤ 4 KB) primary workload. */
const SHORT_BUCKETS: ReadonlySet<string> = new Set(['0.25k', '0.5k', '2k']);
/** The browser-native competitor the breakeven analysis is computed against. */
const REFERENCE_METHOD_TEXT = 'b64url(cs gzip)';
const REFERENCE_METHOD_BINARY = 'cs gzip';
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
  /** Encoded output; `.length` is the measured size (chars on text, bytes on binary). */
  compress(doc: LoadedDoc): string | Uint8Array | Promise<string | Uint8Array>;
  decompress(encoded: string | Uint8Array): string | Promise<string>;
  /** True for tokzip modes: the session-amortized metric charges them the dictionary. */
  usesDictionary?: boolean;
  /** Excluded from the speed benchmark (see {@link Competitor.speedExempt}). */
  speedExempt?: boolean;
}

interface SizeTotals {
  docs: number;
  inputBytes: number;
  /** Output units per method: chars on the text channel, bytes on the binary channel. */
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

interface SessionView {
  docs: number;
  inputBytes: number;
  /** Classic per-document ratio, no dictionary cost (secondary metric). */
  ratios: Record<string, number>;
  /** Session-amortized ratio: dictionary transfer charged once per session (primary metric). */
  amortizedRatios: Record<string, number>;
}

interface LanguageReport {
  docs: number;
  registered: boolean;
  /**
   * Per-method brotli-compressed dictionary transfer charged to the full session (own
   * module + fenced dependencies of that method's frames); keyed by method name.
   */
  dictTransferBytes: Record<string, number>;
  /** Same as dictTransferBytes but scoped to the short (≤ 4 KB) session's documents. */
  shortDictTransferBytes: Record<string, number>;
  buckets: Record<string, { docs: number; inputBytes: number; ratios: Record<string, number> }>;
  total: { inputBytes: number; ratios: Record<string, number> };
  session: SessionView;
  /** The ≤ 4 KB documents as their own session (the primary workload). */
  shortSession: SessionView;
  /**
   * Cumulative input bytes after which each tokzip mode's dictionary pays for itself against
   * the browser-native reference codec; undefined when it never does.
   */
  breakevenBytes: Record<string, number | undefined>;
}

interface BenchReport {
  schemaVersion: 3;
  channel: 'text' | 'binary';
  commit: string;
  commitTimestamp: string;
  timestamp: string;
  runtime: string;
  methods: string[];
  referenceMethod: string;
  corpus: { split: 'bench-v2'; sha256: string };
  roundTrip: { docs: number; methods: number; checks: number; failures: string[] };
  languages: Record<string, LanguageReport>;
  total: { docs: number; inputBytes: number; ratios: Record<string, number> };
  session: SessionView;
  shortSession: SessionView;
  speed?: Record<string, SpeedResult>;
}

const { binary: BINARY_CHANNEL } = parseChannel(process.argv.slice(2));

const METHODS: BenchMethod[] = BINARY_CHANNEL
  ? [
      tokzipMethod('fast', 'binary'),
      tokzipMethod('small', 'binary'),
      ...binaryCompetitors.map(
        (competitor): BenchMethod => ({
          name: competitor.name,
          compress: (doc) => competitor.compress(doc.content),
          decompress: (encoded) => competitor.decompress(encoded as Uint8Array),
          speedExempt: competitor.speedExempt,
        })
      ),
    ]
  : [
      tokzipMethod('fast', 'text'),
      tokzipMethod('small', 'text'),
      ...competitors.map(
        (competitor): BenchMethod => ({
          name: competitor.name,
          compress: (doc) => competitor.compress(doc.content),
          decompress: (encoded) => competitor.decompress(encoded as string),
          speedExempt: competitor.speedExempt,
        })
      ),
    ];
const METHOD_NAMES = METHODS.map((method) => method.name);
const REFERENCE_METHOD = BINARY_CHANNEL ? REFERENCE_METHOD_BINARY : REFERENCE_METHOD_TEXT;

async function main(): Promise<void> {
  const { speed, jsonPath, languages } = parseArgs(process.argv.slice(2));
  const report: BenchReport = {
    schemaVersion: 3,
    channel: BINARY_CHANNEL ? 'binary' : 'text',
    commit: process.env['GITHUB_SHA'] ?? gitOutput(['rev-parse', 'HEAD']) ?? 'unknown',
    commitTimestamp: new Date(gitOutput(['show', '-s', '--format=%cI', 'HEAD']) ?? Date.now()).toISOString(),
    timestamp: new Date().toISOString(),
    runtime: `bun ${Bun.version}`,
    methods: METHOD_NAMES,
    referenceMethod: REFERENCE_METHOD,
    corpus: { split: 'bench-v2', sha256: '' },
    roundTrip: { docs: 0, methods: METHODS.length, checks: 0, failures: [] },
    languages: {},
    total: { docs: 0, inputBytes: 0, ratios: {} },
    session: emptySession(),
    shortSession: emptySession(),
  };
  const grandTotals = emptyTotals();
  const grandShortTotals = emptyTotals();
  const grandDictBytes: Record<string, number> = {};
  const grandShortDictBytes: Record<string, number> = {};
  const loadedDocs: LoadedDoc[] = [];

  for (const language of languages) {
    const result = await benchLanguage(language, report, grandTotals, loadedDocs);
    if (!result) continue;
    accumulate(grandShortTotals, result.shortTotals);
    for (const [method, bytes] of Object.entries(result.dictBytes)) {
      grandDictBytes[method] = (grandDictBytes[method] ?? 0) + bytes;
    }
    for (const [method, bytes] of Object.entries(result.shortDictBytes)) {
      grandShortDictBytes[method] = (grandShortDictBytes[method] ?? 0) + bytes;
    }
  }
  if (grandTotals.docs === 0) {
    console.error('error: no bench documents found (fetch + split the corpus first, or check the language name)');
    process.exit(1);
  }

  report.total = { docs: grandTotals.docs, inputBytes: grandTotals.inputBytes, ratios: ratiosOf(grandTotals) };
  report.session = sessionOf(grandTotals, grandDictBytes);
  report.shortSession = sessionOf(grandShortTotals, grandShortDictBytes);
  report.corpus.sha256 = corpusHash(loadedDocs);
  printTotals({
    report,
    grand: grandTotals,
    short: grandShortTotals,
    dictBytes: grandDictBytes,
    shortDictBytes: grandShortDictBytes,
  });
  if (speed) report.speed = await benchSpeed(loadedDocs);
  printRoundTrip(report);
  if (jsonPath) writeReport(jsonPath, report);
  if (report.roundTrip.failures.length > 0) process.exitCode = 1;
}

function parseChannel(args: string[]): { binary: boolean } {
  return { binary: args.includes('--binary') };
}

function parseArgs(args: string[]): { speed: boolean; jsonPath?: string; languages: string[] } {
  const speed = args.includes('--speed');
  const jsonIndex = args.indexOf('--json');
  const jsonPath = jsonIndex === -1 ? undefined : args[jsonIndex + 1];
  if (jsonIndex !== -1 && (!jsonPath || jsonPath.startsWith('--'))) {
    console.error('error: --json requires a path');
    process.exit(1);
  }
  // Flags (--speed, --binary, --json) are excluded by the '--' prefix check; the --json
  // value is excluded by position.
  const requested = args.filter((arg, index) => {
    return !arg.startsWith('--') && (jsonIndex === -1 || index !== jsonIndex + 1);
  });
  const languages =
    requested.length > 0
      ? requested
      : [
          ...new Set(
            corpusDirs()
              .filter((corpusDir) => existsSync(corpusDir))
              .flatMap((corpusDir) =>
                readdirSync(corpusDir, { withFileTypes: true })
                  .filter((entry) => entry.isDirectory() && !entry.name.startsWith('.'))
                  .map((entry) => entry.name)
              )
          ),
        ].toSorted();
  return { speed, jsonPath, languages };
}

async function benchLanguage(
  language: string,
  report: BenchReport,
  grandTotals: SizeTotals,
  allDocs: LoadedDoc[]
): Promise<
  { shortTotals: SizeTotals; dictBytes: Record<string, number>; shortDictBytes: Record<string, number> } | undefined
> {
  const docs = loadBenchDocs(language);
  if (docs.length === 0) {
    console.log(`\n${language}: no bench split (fetch + split the corpus first)`);
    return undefined;
  }
  const registered = languageByName(language) !== undefined;
  const dictBytes = registered ? dictionaryTransferBytes(language) : 0;
  const loaded = docs.map((doc) => ({
    ...doc,
    language,
    registered,
    inputBytes: Buffer.byteLength(doc.content),
  }));
  allDocs.push(...loaded);
  console.log(
    `\n=== ${language} (${docs.length} bench docs${registered ? `, dict ${formatKb(dictBytes)} brotli` : ', id-0 fallback'}) ===`
  );
  printHeader('bucket');

  const languageTotals = emptyTotals();
  const shortTotals = emptyTotals();
  const buckets: LanguageReport['buckets'] = {};
  // Fence-aware frames (FLAG_FENCED) additionally require each referenced block language's
  // module on the decoding side, so those transfers are charged to the session too.
  const fencedCollector = makeFencedCollector();
  for (const bucket of BUCKETS) {
    const bucketDocs = loaded.filter((doc) => doc.bucket === bucket);
    if (bucketDocs.length === 0) continue;
    const totals = emptyTotals();
    for (const doc of bucketDocs) await benchDoc(doc, totals, report.roundTrip, fencedCollector.collect);
    accumulate(languageTotals, totals);
    if (SHORT_BUCKETS.has(bucket)) accumulate(shortTotals, totals);
    buckets[bucket] = { docs: totals.docs, inputBytes: totals.inputBytes, ratios: ratiosOf(totals) };
    printRow(bucket, totals);
  }
  accumulate(grandTotals, languageTotals);
  const sessionDictBytes: Record<string, number> = {};
  const shortDictBytes: Record<string, number> = {};
  for (const method of METHODS) {
    if (!method.usesDictionary) continue;
    sessionDictBytes[method.name] = dictBytes + fencedCollector.sessionExtraBytes(method.name);
    shortDictBytes[method.name] = shortTotals.docs > 0 ? dictBytes + fencedCollector.shortExtraBytes(method.name) : 0;
  }
  const session = sessionOf(languageTotals, sessionDictBytes);
  const shortSession = sessionOf(shortTotals, shortDictBytes);
  printRow('all', languageTotals);
  printSessionRow('all+dict', languageTotals, sessionDictBytes);
  printSessionRow('sh+dict', shortTotals, shortDictBytes);
  const breakeven = breakevenOf(languageTotals, sessionDictBytes);
  if (dictBytes > 0) {
    const parts = Object.entries(breakeven).map(
      ([method, bytes]) => `${method}: ${bytes === undefined ? 'never' : formatKb(bytes)}`
    );
    console.log(`  dictionary breakeven vs ${REFERENCE_METHOD} — ${parts.join(', ')}`);
  }
  report.languages[language] = {
    docs: docs.length,
    registered,
    dictTransferBytes: sessionDictBytes,
    shortDictTransferBytes: shortDictBytes,
    buckets,
    total: { inputBytes: languageTotals.inputBytes, ratios: ratiosOf(languageTotals) },
    session,
    shortSession,
    breakevenBytes: breakeven,
  };
  return { shortTotals, dictBytes: sessionDictBytes, shortDictBytes };
}

async function benchDoc(
  doc: LoadedDoc,
  totals: SizeTotals,
  roundTrip: BenchReport['roundTrip'],
  onTokzipFrame?: (doc: LoadedDoc, methodName: string, encoded: string | Uint8Array) => void
): Promise<void> {
  totals.docs += 1;
  totals.inputBytes += doc.inputBytes;
  roundTrip.docs += 1;
  for (const [index, method] of METHODS.entries()) {
    let failure: string | undefined;
    try {
      const encoded = await method.compress(doc);
      if (method.usesDictionary) onTokzipFrame?.(doc, method.name, encoded);
      totals.outputChars[index]! += encoded.length;
      roundTrip.checks += 1;
      if ((await method.decompress(encoded)) !== doc.content) failure = `${doc.language}/${doc.file} (${method.name})`;
    } catch (error) {
      failure = `${doc.language}/${doc.file} (${method.name}): ${error}`;
    }
    if (failure !== undefined) {
      roundTrip.failures.push(failure);
      console.error(`ROUND-TRIP FAILURE: ${failure}`);
    }
  }
}

const LANGUAGE_NAME_BY_ID = new Map(Object.entries(LANGUAGE_IDS).map(([name, id]) => [id, name]));

function addFencedIds(map: Map<string, Set<number>>, methodName: string, ids: number[]): void {
  let set = map.get(methodName);
  if (!set) map.set(methodName, (set = new Set()));
  for (const id of ids) set.add(id);
}

interface FencedCollector {
  collect(doc: LoadedDoc, methodName: string, encoded: string | Uint8Array): void;
  sessionExtraBytes(methodName: string): number;
  shortExtraBytes(methodName: string): number;
}

/**
 * Charges fence-extended dictionary dependencies: when a shipped tokzip frame carries
 * FLAG_FENCED, every registered block language named by the document's fences must also be
 * delivered to the decoding client, so those modules' transfers join the session cost.
 * Conservative: fences are charged when any fenced frame ships for the document, without
 * proving which specific matches reached each extension.
 */
function makeFencedCollector(): FencedCollector {
  const encoder = new TextEncoder();
  // FLAG_FENCED is per encoded frame, so fast and small can depend on different modules —
  // dependencies are tracked per method, while the per-document fence scan is cached.
  const sessionIds = new Map<string, Set<number>>();
  const shortIds = new Map<string, Set<number>>();
  const docFenceIds = new Map<string, number[]>();
  const fenceIdsOf = (doc: LoadedDoc): number[] => {
    const cached = docFenceIds.get(doc.file);
    if (cached) return cached;
    const frameId = languageByName(doc.registered ? doc.language : 'none')!.id;
    const bytes = encoder.encode(doc.content);
    const tracker = new FenceTracker(frameId);
    const ids = new Set<number>();
    for (let i = 0; i < bytes.length; i++) {
      if (bytes[i] !== 10) continue;
      const id = tracker.languageIdAt(bytes, i + 1);
      if (id !== frameId) ids.add(id);
    }
    const list = [...ids];
    docFenceIds.set(doc.file, list);
    return list;
  };
  const transferOf = (map: Map<string, Set<number>>, methodName: string): number => {
    let bytes = 0;
    for (const id of map.get(methodName) ?? []) bytes += dictionaryTransferBytes(LANGUAGE_NAME_BY_ID.get(id) ?? '');
    return bytes;
  };
  return {
    collect(doc, methodName, encoded) {
      const flags = typeof encoded === 'string' ? RADIX64_ALPHABET.indexOf(encoded[2]!) : encoded[2]!;
      if ((flags & FLAG_FENCED) === 0) return;
      const ids = fenceIdsOf(doc);
      addFencedIds(sessionIds, methodName, ids);
      if (SHORT_BUCKETS.has(doc.bucket)) addFencedIds(shortIds, methodName, ids);
    },
    sessionExtraBytes: (methodName) => transferOf(sessionIds, methodName),
    shortExtraBytes: (methodName) => transferOf(shortIds, methodName),
  };
}

/**
 * Brotli-compressed transfer size of the generated dictionary module — what a CDN actually
 * sends to a client that needs this language (competitors ship nothing). Cached per language.
 */
const dictTransferCache = new Map<string, number>();
function dictionaryTransferBytes(language: string): number {
  const cached = dictTransferCache.get(language);
  if (cached !== undefined) return cached;
  // Corpus locale dirs are kebab-case (en-US); generated modules are camelCase (enUs).
  const parts = language.split('-');
  const moduleName =
    parts[0]! +
    parts
      .slice(1)
      .map((part) => part[0]! + part.slice(1).toLowerCase())
      .join('');
  const modulePath = join(import.meta.dirname, '..', '..', 'src', 'generated', `${moduleName}.ts`);
  const bytes = existsSync(modulePath)
    ? brotliCompressSync(readFileSync(modulePath), {
        params: { [zlibConstants.BROTLI_PARAM_QUALITY]: 11 },
      }).length
    : 0;
  dictTransferCache.set(language, bytes);
  return bytes;
}

function sessionOf(totals: SizeTotals, dictBytes: Record<string, number>): SessionView {
  // A corpus can legitimately have no docs in a session (e.g. no short-bucket documents
  // under TOKZIP_CORPUS_DIR); ratiosOf would divide by zero and serialize NaN as null.
  if (totals.inputBytes === 0) return { docs: totals.docs, inputBytes: 0, ratios: {}, amortizedRatios: {} };
  const amortizedRatios = Object.fromEntries(
    METHODS.map((method, index) => [
      method.name,
      round4((totals.outputChars[index]! + (dictBytes[method.name] ?? 0)) / totals.inputBytes),
    ])
  );
  return { docs: totals.docs, inputBytes: totals.inputBytes, ratios: ratiosOf(totals), amortizedRatios };
}

/**
 * Cumulative input bytes at which a dictionary-carrying method's total transfer (output +
 * dictionary) drops below the reference codec's, assuming the language's average ratios.
 */
function breakevenOf(totals: SizeTotals, dictBytes: Record<string, number>): Record<string, number | undefined> {
  const referenceIndex = METHOD_NAMES.indexOf(REFERENCE_METHOD);
  const result: Record<string, number | undefined> = {};
  if (referenceIndex === -1 || totals.inputBytes === 0) return result;
  const referenceRatio = totals.outputChars[referenceIndex]! / totals.inputBytes;
  for (const [index, method] of METHODS.entries()) {
    if (!method.usesDictionary) continue;
    const ratio = totals.outputChars[index]! / totals.inputBytes;
    result[method.name] =
      ratio < referenceRatio ? Math.ceil((dictBytes[method.name] ?? 0) / (referenceRatio - ratio)) : undefined;
  }
  return result;
}

async function benchSpeed(docs: LoadedDoc[]): Promise<Record<string, SpeedResult>> {
  console.log(`\n=== END-TO-END SPEED (${SPEED_SAMPLE_COUNT} median samples, per-document framing) ===`);
  const inputBytes = docs.reduce((sum, doc) => sum + doc.inputBytes, 0);
  const iterations = Math.max(1, Math.ceil(SPEED_TARGET_BYTES / inputBytes));
  const processedBytes = inputBytes * iterations;
  const operations = docs.length * iterations;
  const result: Record<string, SpeedResult> = {};

  for (const method of METHODS) {
    if (method.speedExempt) continue;
    const encoded: (string | Uint8Array)[] = [];
    for (const doc of docs) encoded.push(await method.compress(doc));
    // Warm both code paths without adding another expensive full q11 corpus pass.
    for (let index = 0; index < Math.min(32, docs.length); index++) await method.decompress(encoded[index]!);
    const compressSamples = await sampleTimes(async () => {
      let chars = 0;
      for (let iteration = 0; iteration < iterations; iteration++) {
        for (const doc of docs) {
          const out = await method.compress(doc);
          chars += out.length;
        }
      }
      return chars;
    });
    const decompressSamples = await sampleTimes(async () => {
      let chars = 0;
      for (let iteration = 0; iteration < iterations; iteration++) {
        for (const value of encoded) {
          const out = await method.decompress(value);
          chars += out.length;
        }
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

async function sampleTimes(operation: () => Promise<number>): Promise<number[]> {
  const times: number[] = [];
  let checksum = 0;
  for (let sample = 0; sample < SPEED_SAMPLE_COUNT; sample++) {
    const started = performance.now();
    checksum ^= await operation();
    times.push(performance.now() - started);
  }
  // Retain an observable dependency on every operation result so engines cannot discard
  // the work while still keeping benchmark output deterministic.
  if (checksum === Number.MIN_SAFE_INTEGER) console.log(checksum);
  return times;
}

function loadBenchDocs(language: string): BenchDoc[] {
  return corpusDirs().flatMap((corpusDir) => {
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

function tokzipMethod(mode: 'fast' | 'small', output: 'text' | 'binary'): BenchMethod {
  return {
    name: `tokzip ${mode}`,
    compress: (doc) => {
      const language = doc.registered ? doc.language : 'none';
      return output === 'binary'
        ? compress(doc.content, { language, mode, output: 'binary' })
        : compress(doc.content, { language, mode });
    },
    decompress: (encoded) => decompress(encoded) as string,
    usesDictionary: true,
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

interface TotalsForPrint {
  report: BenchReport;
  grand: SizeTotals;
  short: SizeTotals;
  dictBytes: Record<string, number>;
  shortDictBytes: Record<string, number>;
}

function printTotals({ report, grand, short, dictBytes, shortDictBytes }: TotalsForPrint): void {
  console.log(
    `\n=== TOTAL (${report.total.docs} docs, ${Object.keys(report.languages).length} languages, ${report.channel} channel) ===`
  );
  printHeader('');
  // Raw totals, not the report's rounded ratios: reconstructing output sizes from
  // 4-decimal ratios double-rounds and can contradict the per-language rows by 0.1 pt.
  printRow('all', grand);
  printSessionRow('all+dict', grand, dictBytes);
  printSessionRow('sh+dict', short, shortDictBytes);
  console.log(
    `\nPRIMARY metric: sh+dict = session-amortized ratio on ≤4 KB docs, tokzip charged each\n` +
      `language's brotli-compressed dictionary once per session; reference codec: ${report.referenceMethod}.`
  );
}

/** Session-amortized row computed from raw totals (single rounding at display time). */
function printSessionRow(label: string, totals: SizeTotals, dictBytes: Record<string, number>): void {
  if (totals.docs === 0 || totals.inputBytes === 0) return;
  console.log(
    [
      label.padStart(columnWidth(label)),
      String(totals.docs).padStart(columnWidth('docs')),
      String(totals.inputBytes).padStart(columnWidth('input')),
      ...METHODS.map((method, index) =>
        `${(((totals.outputChars[index]! + (dictBytes[method.name] ?? 0)) / totals.inputBytes) * 100).toFixed(1)}%`.padStart(
          columnWidth(method.name)
        )
      ),
    ].join('')
  );
}

function emptySession(): SessionView {
  return { docs: 0, inputBytes: 0, ratios: {}, amortizedRatios: {} };
}

function formatKb(bytes: number): string {
  return `${(bytes / 1024).toFixed(1)} KB`;
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

await main();
