/**
 * Deterministic, seeded train/benchmark split (~85/15 by document) with near-duplicate
 * detection across splits (shingled n-gram hashing), rewriting each language's manifest.
 * License policy: non-trainable (copyleft/share-alike) samples are forced into the benchmark
 * split, which also guarantees they can never leak into shipped dictionaries.
 *
 * The resulting benchmark split is versioned (`bench-v1`): re-running with the same corpus
 * and seed reproduces it exactly.
 *
 * Usage: bun scripts/corpus/split.ts [<language> ...]
 */
import { existsSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { CORPUS_DIR, seededRandom, type ManifestEntry } from './shared.ts';

const SPLIT_SEED = 0xBE_9C_11;
const BENCH_RATIO = 0.15;
const SHINGLE_LENGTH = 32;
const SHINGLES_PER_DOC = 64;
/** Documents sharing at least this fraction of sampled shingles are near-duplicates. */
const NEAR_DUP_THRESHOLD = 0.5;

function shinglesOf(content: string): Set<number> {
  const shingles = new Set<number>();
  if (content.length < SHINGLE_LENGTH) return shingles;
  const step = Math.max(1, Math.floor((content.length - SHINGLE_LENGTH) / SHINGLES_PER_DOC));
  for (let i = 0; i + SHINGLE_LENGTH <= content.length && shingles.size < SHINGLES_PER_DOC; i += step) {
    let hash = 0x81_1C_9D_C5;
    for (let j = 0; j < SHINGLE_LENGTH; j++) hash = Math.imul(hash ^ content.codePointAt(i + j)!, 0x01_00_01_93);
    // oxlint-disable-next-line unicorn/prefer-math-trunc -- >>> 0 coerces to uint32, Math.trunc does not
    shingles.add(hash >>> 0);
  }
  return shingles;
}

function overlap(a: Set<number>, b: Set<number>): number {
  let shared = 0;
  for (const hash of a) if (b.has(hash)) shared++;
  return shared / Math.max(1, Math.min(a.size, b.size));
}

function splitLanguage(language: string): void {
  const dir = join(CORPUS_DIR, language);
  const manifestPath = join(dir, 'manifest.jsonl');
  if (!existsSync(manifestPath)) return;
  const entries: ManifestEntry[] = readFileSync(manifestPath, 'utf8')
    .split('\n')
    .filter((line) => line.trim())
    .map((line) => JSON.parse(line) as ManifestEntry);

  const random = seededRandom(SPLIT_SEED);
  const shingleSets: Set<number>[] = entries.map((entry) => shinglesOf(readFileSync(join(dir, entry.file), 'utf8')));

  // Near-duplicate clusters: union-find over pairwise-overlapping documents, so a whole
  // cluster always lands in one split (a bench doc must never have a near-copy in training).
  const parent = entries.map((_, i) => i);
  const find = (i: number): number => (parent[i] === i ? i : (parent[i] = find(parent[i]!)));
  for (let i = 0; i < entries.length; i++) {
    for (let j = i + 1; j < entries.length; j++) {
      if (shingleSets[i]!.size === 0 || shingleSets[j]!.size === 0) continue;
      if (overlap(shingleSets[i]!, shingleSets[j]!) >= NEAR_DUP_THRESHOLD) parent[find(j)] = find(i);
    }
  }

  const clusterSplit = new Map<number, 'train' | 'bench'>();
  for (let i = 0; i < entries.length; i++) {
    const root = find(i);
    if (!clusterSplit.has(root)) clusterSplit.set(root, random() < BENCH_RATIO ? 'bench' : 'train');
    // License policy: non-trainable sources are benchmark-only (and drag their cluster along).
    if (!entries[i]!.trainable) clusterSplit.set(root, 'bench');
  }
  for (let i = 0; i < entries.length; i++) entries[i]!.split = clusterSplit.get(find(i))!;
  const trainCount = entries.filter((entry) => entry.split === 'train').length;
  const benchCount = entries.length - trainCount;

  writeFileSync(manifestPath, entries.map((entry) => JSON.stringify(entry)).join('\n') + '\n');
  console.log(`${language}: ${trainCount} train / ${benchCount} bench (bench-v1, seed ${SPLIT_SEED})`);
}

const requested = process.argv.slice(2);
const languages =
  requested.length > 0
    ? requested
    : readdirSync(CORPUS_DIR, { withFileTypes: true })
        .filter((e) => e.isDirectory() && !e.name.startsWith('.'))
        .map((e) => e.name);
for (const language of languages) splitLanguage(language);
