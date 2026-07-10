import { resolve } from 'node:path';

export interface ManifestEntry {
  file: string;
  lang: string;
  origin: 'human' | 'llm';
  source: string;
  license: string;
  sizeBucket: string;
  trainable: boolean;
  split?: 'train' | 'bench';
}

const defaultCorpusDir = resolve(import.meta.dir, '../../tokzip-corpus/corpus');

/** Corpus checkout used by offline training and benchmarks. */
export const CORPUS_DIR = resolve(process.env['TOKZIP_CORPUS_DIR'] ?? defaultCorpusDir);
