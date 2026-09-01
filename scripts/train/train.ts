/**
 * Offline trainer: builds the shared wrapper dictionary and every language's dictionary suffix
 * from the corpus train split (`dict/*.bin`), then runs the Rust priors trainer, which derives
 * each language's literal context classes and initial model probabilities (`priors/*.bin`).
 * Both outputs are embedded into the wasm module by the Rust build.
 *
 *   bun scripts/train/train.ts
 */
import { spawnSync } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { CORPUS_DIR, manifestEntrySchema } from '../corpus.ts';
import { trainDictionary } from './trainDictionary.ts';
import { buildWrapperDictionary } from './wrapperContent.ts';

/** Embedded languages, in id order (must match `LANGUAGES` in rust/crates/tokzip/src/lang.rs). */
const LANGUAGES = ['text', 'en-US', 'ja-JP', 'html', 'css', 'javascript', 'typescript'];

/**
 * Dictionary-suffix budget per language. Ratio improves with budget, but every language ships
 * in the wasm module and its chains are built at first use, so the budget is bounded by the
 * module size target (≤ 500 KB) and load time rather than by ratio alone.
 */
const DICTIONARY_BUDGET_BYTES = 40 * 1024;

const ROOT = join(import.meta.dir, '../..');
const DICT_DIR = join(ROOT, 'dict');

function main(): void {
  mkdirSync(DICT_DIR, { recursive: true });
  const wrapper = buildWrapperDictionary();
  writeFileSync(join(DICT_DIR, 'wrapper.bin'), wrapper);
  console.log(`wrapper: ${wrapper.length} B`);
  const wrapperText = new TextDecoder().decode(wrapper);
  for (const language of LANGUAGES) {
    const docs = loadTrainDocs(language);
    if (docs.length === 0) throw new Error(`no train-split corpus for ${language} under ${CORPUS_DIR}/${language}`);
    const suffix = trainDictionary(docs, DICTIONARY_BUDGET_BYTES, wrapperText);
    writeFileSync(join(DICT_DIR, `${language}.bin`), suffix);
    console.log(`${language}: dictionary ${suffix.length} B from ${docs.length} docs`);
  }
  const priors = spawnSync(
    'cargo',
    ['run', '--release', '--features', 'train', '--bin', 'train-priors', '--', CORPUS_DIR],
    {
      cwd: join(ROOT, 'rust'),
      stdio: 'inherit',
    }
  );
  if (priors.status !== 0) throw new Error('priors training failed');
}

/**
 * Training reads only the public corpus: generated dictionaries embed literal fragments of
 * their training documents and are committed to this public repository.
 */
function loadTrainDocs(language: string): string[] {
  const dir = join(CORPUS_DIR, language);
  const manifestPath = join(dir, 'manifest.jsonl');
  if (!existsSync(manifestPath)) return [];
  const docs: string[] = [];
  for (const line of readFileSync(manifestPath, 'utf8').split('\n')) {
    if (!line.trim()) continue;
    const entry = manifestEntrySchema.parse(JSON.parse(line));
    if (entry.split !== 'train' || entry.trainable === false) continue;
    const path = join(dir, entry.file);
    if (existsSync(path)) docs.push(readFileSync(path, 'utf8'));
  }
  return docs;
}

main();
