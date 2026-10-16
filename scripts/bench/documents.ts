import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs';
import { basename, dirname, join } from 'node:path';
import { corpusDirs, manifestEntrySchema } from '../corpus.ts';

export interface Doc {
  corpus: string;
  language: string;
  bucket: string;
  content: string;
  bytes: number;
}

const BUCKETS: [number, string][] = [
  [1024, '≤1K'],
  [4096, '1–4K'],
  [16_384, '4–16K'],
  [Number.POSITIVE_INFINITY, '>16K'],
];

export function loadBenchDocs(inputPath?: string): Doc[] {
  if (inputPath) {
    return readFileSync(inputPath, 'utf8')
      .split(/\r?\n/)
      .filter((line) => line.trim())
      .map((content) => {
        JSON.parse(content);
        return document(content, 'input', 'json');
      });
  }
  return corpusDirs().flatMap((root) => readdirLanguages(root).flatMap((language) => languageDocs(root, language)));
}

function languageDocs(root: string, language: string): Doc[] {
  const dir = join(root, language);
  const manifestPath = join(dir, 'manifest.jsonl');
  if (!existsSync(manifestPath)) return [];
  const docs: Doc[] = [];
  for (const line of readFileSync(manifestPath, 'utf8').split('\n')) {
    if (!line.trim()) continue;
    const entry = manifestEntrySchema.parse(JSON.parse(line));
    if (entry.split === 'bench') {
      docs.push(document(readFileSync(join(dir, entry.file), 'utf8'), basename(dirname(root)), language));
    }
  }
  return docs;
}

function document(content: string, corpus: string, language: string): Doc {
  const bytes = Buffer.byteLength(content);
  return { content, corpus, language, bytes, bucket: BUCKETS.find(([limit]) => bytes <= limit)![1] };
}

function readdirLanguages(corpusDir: string): string[] {
  return readdirSync(corpusDir)
    .filter((name) => statSync(join(corpusDir, name)).isDirectory())
    .toSorted();
}
