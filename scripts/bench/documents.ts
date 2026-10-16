import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs';
import { basename, dirname, join } from 'node:path';
import { corpusDirs, manifestEntrySchema } from '../corpus.ts';

export interface Doc {
  corpus: string;
  language: string;
  bucket: string;
  content: string;
  location: string;
  bytes: number;
}

const BUCKETS: [number, string][] = [
  [1024, '≤1K'],
  [4096, '1–4K'],
  [16_384, '4–16K'],
  [Number.POSITIVE_INFINITY, '>16K'],
];

export function loadBenchDocs(inputPath?: string): Doc[] {
  if (inputPath) return inputDocs(inputPath);
  return corpusDirs().flatMap((root) => readdirLanguages(root).flatMap((language) => languageDocs(root, language)));
}

function inputDocs(file: string): Doc[] {
  const docs: Doc[] = [];
  for (const [index, content] of readFileSync(file, 'utf8').split(/\r?\n/).entries()) {
    if (!content.trim()) continue;
    const location = `${file}:${index + 1}`;
    try {
      JSON.parse(content);
    } catch {
      throw new Error(`${location}: expected one complete JSON value`);
    }
    docs.push(document(content, 'input', 'json', location));
  }
  return docs;
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
      const file = join(dir, entry.file);
      docs.push(document(readFileSync(file, 'utf8'), basename(dirname(root)), language, file));
    }
  }
  return docs;
}

function document(content: string, corpus: string, language: string, location: string): Doc {
  const bytes = Buffer.byteLength(content);
  return { content, corpus, language, location, bytes, bucket: BUCKETS.find(([limit]) => bytes <= limit)![1] };
}

function readdirLanguages(corpusDir: string): string[] {
  return readdirSync(corpusDir)
    .filter((name) => statSync(join(corpusDir, name)).isDirectory())
    .toSorted();
}
