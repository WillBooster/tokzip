/**
 * Streaming benchmark: tokzip streams vs one-shot binary frames on the seeded `bench-v2`
 * corpus split, per language, with the corpus documents concatenated into one large input
 * (the streaming use case: many documents flowing through one pipe).
 *
 * Every configuration is round-trip verified; ratio is compressed bytes / input bytes, and
 * speed is end-to-end through the TransformStream pipe (best of SAMPLES runs).
 *
 * Usage: bun scripts/bench/streamBench.ts [--fast-only|--small-only] [<language> ...]
 */
import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { languageByName } from '../../src/dictionary.ts';
import { compress } from '../../src/index.ts';
import '../../src/languages/index.ts';
import { TokzipCompressionStream, TokzipDecompressionStream } from '../../src/stream.ts';
import { corpusDirs, type ManifestEntry } from '../corpus.ts';

const DEFAULT_LANGUAGES = ['typescript', 'python', 'text', 'ja-JP'];
const BLOCK_SIZES = [16 * 1024, 64 * 1024, 256 * 1024];
const CHUNK_SIZE = 8192; // Feed size, emulating a network/file stream.
const SAMPLES = 3;

interface StreamResult {
  name: string;
  compressedBytes: number;
  compressSeconds: number;
  decompressSeconds: number;
}

function loadConcatenatedInput(language: string): Uint8Array | undefined {
  const parts = corpusDirs().flatMap((corpusDir) => {
    const dir = join(corpusDir, language);
    const manifestPath = join(dir, 'manifest.jsonl');
    if (!existsSync(manifestPath)) return [];
    return readFileSync(manifestPath, 'utf8')
      .split('\n')
      .filter((line) => line.trim())
      .map((line) => JSON.parse(line) as ManifestEntry)
      .filter((entry) => entry.split === 'bench')
      .map((entry) => readFileSync(join(dir, entry.file), 'utf8'));
  });
  if (parts.length === 0) return undefined;
  return new TextEncoder().encode(parts.join('\n'));
}

async function pipeThrough(
  stream: TransformStream<Uint8Array, Uint8Array>,
  input: Uint8Array,
  chunkSize: number
): Promise<Uint8Array> {
  const writer = stream.writable.getWriter();
  const parts: Uint8Array[] = [];
  const readAll = (async () => {
    const reader = stream.readable.getReader();
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      parts.push(value);
    }
  })();
  for (let at = 0; at < input.length; at += chunkSize) {
    await writer.write(input.subarray(at, Math.min(at + chunkSize, input.length)));
  }
  await writer.close();
  await readAll;
  const total = parts.reduce((sum, part) => sum + part.length, 0);
  const out = new Uint8Array(total);
  let at = 0;
  for (const part of parts) {
    out.set(part, at);
    at += part.length;
  }
  return out;
}

function equalBytes(a: Uint8Array, b: Uint8Array): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

async function benchStream(
  name: string,
  input: Uint8Array,
  language: string,
  mode: 'fast' | 'small',
  blockSize: number,
  carryWindow: boolean
): Promise<StreamResult> {
  let compressedBytes = 0;
  let compressSeconds = Number.POSITIVE_INFINITY;
  let decompressSeconds = Number.POSITIVE_INFINITY;
  for (let sample = 0; sample < SAMPLES; sample++) {
    const compressStart = performance.now();
    const compressed = await pipeThrough(
      new TokzipCompressionStream({ language, mode, blockSize, carryWindow }),
      input,
      CHUNK_SIZE
    );
    compressSeconds = Math.min(compressSeconds, (performance.now() - compressStart) / 1000);
    compressedBytes = compressed.length;
    const decompressStart = performance.now();
    const output = await pipeThrough(new TokzipDecompressionStream(), compressed, CHUNK_SIZE);
    decompressSeconds = Math.min(decompressSeconds, (performance.now() - decompressStart) / 1000);
    if (!equalBytes(input, output)) throw new Error(`round-trip mismatch: ${name}`);
  }
  return { name, compressedBytes, compressSeconds, decompressSeconds };
}

function benchOneShot(name: string, input: Uint8Array, language: string, mode: 'fast' | 'small'): StreamResult {
  let compressedBytes = 0;
  let compressSeconds = Number.POSITIVE_INFINITY;
  for (let sample = 0; sample < SAMPLES; sample++) {
    const start = performance.now();
    compressedBytes = compress(input, { language, mode, output: 'binary' }).length;
    compressSeconds = Math.min(compressSeconds, (performance.now() - start) / 1000);
  }
  return { name, compressedBytes, compressSeconds, decompressSeconds: Number.NaN };
}

function printResult(result: StreamResult, inputBytes: number, baseline: number): void {
  const ratio = ((result.compressedBytes / inputBytes) * 100).toFixed(2).padStart(6);
  const delta = (((result.compressedBytes - baseline) / baseline) * 100).toFixed(2).padStart(7);
  const compressMBps = (inputBytes / 1e6 / result.compressSeconds).toFixed(1).padStart(7);
  const decompressMBps = Number.isNaN(result.decompressSeconds)
    ? '      —'
    : (inputBytes / 1e6 / result.decompressSeconds).toFixed(1).padStart(7);
  console.log(
    `${result.name.padEnd(34)} ${String(result.compressedBytes).padStart(9)} ${ratio}% ${delta}% ${compressMBps} ${decompressMBps}`
  );
}

async function main(): Promise<void> {
  const args = process.argv.slice(2);
  const modes: ('fast' | 'small')[] = args.includes('--fast-only')
    ? ['fast']
    : args.includes('--small-only')
      ? ['small']
      : ['fast', 'small'];
  const languages = args.filter((arg) => !arg.startsWith('--'));
  const targets = languages.length > 0 ? languages : DEFAULT_LANGUAGES;

  for (const language of targets) {
    const input = loadConcatenatedInput(language);
    if (!input) {
      console.log(`\n=== ${language}: no bench split, skipping ===`);
      continue;
    }
    const registered = languageByName(language) ? language : 'none';
    console.log(`\n=== ${language} (${(input.length / 1e6).toFixed(2)} MB concatenated bench docs) ===`);
    console.log(`${'config'.padEnd(34)} ${'bytes'.padStart(9)}  ratio   Δbase   cMB/s   dMB/s`);
    for (const mode of modes) {
      const oneShot = benchOneShot(`one-shot ${mode}`, input, registered, mode);
      printResult(oneShot, input.length, oneShot.compressedBytes);
      for (const carryWindow of [true, false]) {
        for (const blockSize of BLOCK_SIZES) {
          const name = `stream ${mode} ${blockSize / 1024}K carry=${carryWindow ? 'on' : 'off'}`;
          const result = await benchStream(name, input, registered, mode, blockSize, carryWindow);
          printResult(result, input.length, oneShot.compressedBytes);
        }
      }
    }
  }
}

await main();
