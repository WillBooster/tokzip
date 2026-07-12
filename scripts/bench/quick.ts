/** Quick tokzip-only harness: total ratio + compress/decompress MB/s on the bench split. */
import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { languageByName } from '../../src/dictionary.ts';
import { compress, decompress } from '../../src/index.ts';
import '../../src/languages/index.ts';
import { corpusDirs, type ManifestEntry } from '../corpus.ts';

const mode = (process.argv[2] ?? 'small') as 'fast' | 'small';
const docs: { content: string; language: string; bytes: number }[] = [];
for (const corpusDir of corpusDirs()) {
  if (!existsSync(corpusDir)) continue;
  for (const entry of readdirSync(corpusDir, { withFileTypes: true })) {
    if (!entry.isDirectory() || entry.name.startsWith('.')) continue;
    const manifestPath = join(corpusDir, entry.name, 'manifest.jsonl');
    if (!existsSync(manifestPath)) continue;
    for (const line of readFileSync(manifestPath, 'utf8').split('\n')) {
      if (!line.trim()) continue;
      const row = JSON.parse(line) as ManifestEntry;
      if (row.split !== 'bench') continue;
      const content = readFileSync(join(corpusDir, entry.name, row.file), 'utf8');
      docs.push({
        content,
        language: languageByName(entry.name) ? entry.name : 'none',
        bytes: Buffer.byteLength(content),
      });
    }
  }
}

const inputBytes = docs.reduce((a, d) => a + d.bytes, 0);
let outChars = 0;
const encoded: string[] = [];
for (const d of docs) {
  const e = compress(d.content, { language: d.language, mode });
  outChars += e.length;
  encoded.push(e);
}
for (let i = 0; i < docs.length; i++) {
  if (decompress(encoded[i]!) !== docs[i]!.content) throw new Error('round-trip failure');
}

const iterations = Math.max(1, Math.ceil(8_000_000 / inputBytes));
const SAMPLES = 7;
const timeIt = (op: () => void): number => {
  const samples: number[] = [];
  for (let s = 0; s < SAMPLES; s++) {
    const t0 = performance.now();
    for (let it = 0; it < iterations; it++) op();
    samples.push(performance.now() - t0);
  }
  const median = samples.toSorted((a, b) => a - b)[Math.floor(SAMPLES / 2)]!;
  return (inputBytes * iterations) / 1_048_576 / (median / 1000);
};
const cMBps = timeIt(() => {
  for (const d of docs) compress(d.content, { language: d.language, mode });
});
const dMBps = timeIt(() => {
  for (const e of encoded) decompress(e);
});
console.log(
  `${mode}: ratio ${((outChars / inputBytes) * 100).toFixed(2)}% ` +
    `compress ${cMBps.toFixed(1)} MB/s decompress ${dMBps.toFixed(1)} MB/s (${docs.length} docs)`
);
