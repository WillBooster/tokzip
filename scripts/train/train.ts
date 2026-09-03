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
import { CORPUS_DIR } from '../corpus.ts';
import { buildWrapperDictionary } from './wrapperContent.ts';

const ROOT = join(import.meta.dir, '../..');

function main(): void {
  mkdirSync(join(ROOT, 'dict'), { recursive: true });
  const wrapper = buildWrapperDictionary();
  writeFileSync(join(ROOT, 'dict', 'wrapper.bin'), wrapper);
  console.log(`wrapper: ${wrapper.length} B`);
  // Training reads only the public corpus: generated dictionaries embed literal fragments of
  // their training documents and are committed to this public repository.
  const trainer = spawnSync('cargo', ['run', '--release', '--features', 'train', '--bin', 'train', '--', CORPUS_DIR], {
    cwd: join(ROOT, 'rust'),
    stdio: 'inherit',
  });
  if (trainer.status !== 0) throw new Error('training failed');
}

main();
