import { appendFileSync, mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

export const CORPUS_DIR = join(import.meta.dir, '../../.corpus');

/** One manifest line per sample; split.ts fills in the `split` field. */
export interface ManifestEntry {
  file: string;
  lang: string;
  origin: 'human' | 'llm';
  source: string;
  license: string;
  sizeBucket: string;
  /** false = copyleft/share-alike: benchmark split only, never dictionary training. */
  trainable: boolean;
  split?: 'train' | 'bench';
}

export function sizeBucketOf(bytes: number): string {
  if (bytes <= 1024) return '0.5k';
  if (bytes <= 4096) return '2k';
  if (bytes <= 16 * 1024) return '8k';
  return '24k';
}

export function writeSample(language: string, origin: 'human' | 'llm', name: string, content: string): void {
  const dir = join(CORPUS_DIR, language, origin);
  mkdirSync(dir, { recursive: true });
  writeFileSync(join(dir, name), content);
}

export function appendManifest(language: string, entry: ManifestEntry): void {
  mkdirSync(join(CORPUS_DIR, language), { recursive: true });
  appendFileSync(join(CORPUS_DIR, language, 'manifest.jsonl'), JSON.stringify(entry) + '\n');
}

/** Deterministic seeded RNG (mulberry32) shared by split and generation sampling. */
// oxlint-disable unicorn/prefer-math-trunc -- >>> 0 coerces to uint32, Math.trunc does not
export function seededRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6D_2B_79_F5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4_294_967_296;
  };
}
