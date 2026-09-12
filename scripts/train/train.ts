/**
 * Offline trainer: writes the shared wrapper dictionary (`dict/wrapper.bin`), then runs the
 * Rust trainer, which builds the code group's shared dictionary part (`dict/code.bin`), every
 * language's dictionary suffix (`dict/<language>.bin`), every group's literal classes plus
 * literal priors (`priors/<group>.bin`), and every language's own model priors
 * (`priors/<language>.bin`) from the corpus train split. All are embedded into the wasm module
 * by the Rust build.
 *
 *   bun scripts/train/train.ts
 */
import { spawnSync } from 'node:child_process';
import { mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { CORPUS_DIR, corpusDirs } from '../corpus.ts';
import { buildWrapperDictionary } from './wrapperContent.ts';

const ROOT = join(import.meta.dir, '../..');

function main(): void {
  mkdirSync(join(ROOT, 'dict'), { recursive: true });
  const wrapper = buildWrapperDictionary();
  writeFileSync(join(ROOT, 'dict', 'wrapper.bin'), wrapper);
  console.log(`wrapper: ${wrapper.length} B`);
  // Dictionary content comes from the public corpus only: generated dictionaries embed literal
  // fragments of their training documents and are committed to this public repository. The
  // private corpus, when checked out beside it, only scores — which public fragments to
  // select, in what order, and the model priors — for the languages it holds enough of.
  const [, privateCorpusDir] = corpusDirs();
  const scoring = privateCorpusDir ? ['--scoring', privateCorpusDir] : [];
  console.log(
    privateCorpusDir ? `scoring corpus: ${privateCorpusDir}` : 'scoring corpus: none (public statistics only)'
  );
  const trainer = spawnSync(
    'cargo',
    ['run', '--release', '--features', 'train', '--bin', 'train', '--', CORPUS_DIR, ...scoring],
    { cwd: join(ROOT, 'rust'), stdio: 'inherit' }
  );
  if (trainer.status !== 0) throw new Error('training failed');
}

main();
