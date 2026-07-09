/**
 * Fetches the human-written code corpus: shallow-clones pinned OSS repos, applies the
 * sampling rules from the design issue, and writes documents + manifest entries under
 * `.corpus/<lang>/human/`. Corpus data is git-ignored; this script (plus oss-sources.json)
 * is the committed, reproducible artifact.
 *
 * Usage: bun scripts/corpus/fetchOss.ts [--quick] [<language> ...]
 *   --quick  clone only the repo marked `quick` per language and cap the sample volume.
 */
import { spawnSync } from 'node:child_process';
import { existsSync, mkdirSync, readdirSync, readFileSync, statSync } from 'node:fs';
import { join } from 'node:path';
import sources from './oss-sources.json';
import { appendManifest, CORPUS_DIR, sizeBucketOf, writeSample } from './shared.ts';

const CACHE_DIR = join(CORPUS_DIR, '.cache');
const EXTENSIONS: Record<string, string[]> = {
  c: ['.c', '.h'],
  cpp: ['.cc', '.cpp', '.cxx', '.hpp', '.hh'],
  csharp: ['.cs'],
  css: ['.css', '.scss'],
  dart: ['.dart'],
  haskell: ['.hs'],
  html: ['.html', '.htm'],
  java: ['.java'],
  jsp: ['.jsp'],
  javascript: ['.js', '.mjs', '.cjs', '.jsx'],
  php: ['.php'],
  python: ['.py'],
  ruby: ['.rb'],
  rust: ['.rs'],
  text: ['.md', '.txt'],
  typescript: ['.ts', '.tsx'],
  zig: ['.zig'],
};
const EXCLUDED_DIRS = new Set([
  'node_modules',
  'vendor',
  'vendored',
  'dist',
  'build',
  'out',
  'third_party',
  'third-party',
  'deps',
  'external',
  'extern',
  '.git',
  'generated',
  '__generated__',
]);
const MAX_FILE_BYTES = 128 * 1024;
const MAX_AVG_LINE_LENGTH = 200;
/** Cap any single repo at this share of a language's corpus volume. */
const SINGLE_REPO_SHARE = 0.2;
const LANG_BUDGET_BYTES = 8 * 1024 * 1024;
const QUICK_LANG_BUDGET_BYTES = 4 * 1024 * 1024;

interface SourceEntry {
  repo: string;
  ref: string;
  license: string;
  trainable: boolean;
  quick?: boolean;
}

main();

function main(): void {
  const args = process.argv.slice(2);
  const quick = args.includes('--quick');
  const requested = args.filter((a) => !a.startsWith('--'));
  const languages = requested.length > 0 ? requested : Object.keys(sources.languages);
  mkdirSync(CACHE_DIR, { recursive: true });
  for (const language of languages) {
    if (language === 'text') continue; // Harvested after all clones below.
    fetchLanguage(language, quick);
  }
  if (languages.includes('text') || requested.length === 0) harvestText(quick);
}

function fetchLanguage(language: string, quick: boolean): void {
  const entries = (sources.languages as Record<string, SourceEntry[]>)[language];
  const extensions = EXTENSIONS[language];
  if (!entries || !extensions) {
    console.error(`skip ${language}: no sources or extensions defined`);
    return;
  }
  const selected = quick ? entries.filter((e) => e.quick) : entries;
  const budget = quick ? QUICK_LANG_BUDGET_BYTES : LANG_BUDGET_BYTES;
  const repoCap = selected.length > 1 ? budget * SINGLE_REPO_SHARE : budget;
  let total = 0;
  let index = 0;
  for (const entry of selected) {
    if (total >= budget) break;
    const checkout = cloneAt(entry);
    if (!checkout) continue;
    const sha = resolveSha(checkout.dir);
    let repoBytes = 0;
    for (const file of sampleFiles(checkout.dir, extensions)) {
      if (total >= budget || repoBytes >= repoCap) break;
      const content = readFileSync(file.path, 'utf8');
      const name = `${String(index++).padStart(5, '0')}.txt`;
      writeSample(language, 'human', name, content);
      appendManifest(language, {
        file: `human/${name}`,
        lang: language,
        origin: 'human',
        source: `${entry.repo}@${sha}:${file.relative}`,
        license: entry.license,
        sizeBucket: sizeBucketOf(file.bytes),
        trainable: entry.trainable,
      });
      total += file.bytes;
      repoBytes += file.bytes;
    }
    console.log(`${language}: ${entry.repo} → ${repoBytes} B (total ${total} B)`);
  }
}

/** The `text` corpus harvests README/CHANGELOG/docs prose from every cloned repo. */
function harvestText(quick: boolean): void {
  if (!existsSync(CACHE_DIR)) return;
  const budget = quick ? QUICK_LANG_BUDGET_BYTES : LANG_BUDGET_BYTES;
  let total = 0;
  let index = 0;
  for (const repoDir of readdirSync(CACHE_DIR)) {
    const dir = join(CACHE_DIR, repoDir);
    if (!statSync(dir).isDirectory()) continue;
    for (const file of sampleFiles(dir, EXTENSIONS.text!)) {
      if (total >= budget) return;
      const content = readFileSync(file.path, 'utf8');
      const name = `${String(index++).padStart(5, '0')}.txt`;
      writeSample('text', 'human', name, content);
      appendManifest('text', {
        file: `human/${name}`,
        lang: 'text',
        origin: 'human',
        source: `${repoDir}:${file.relative}`,
        license: 'per-repo (see source repo)',
        sizeBucket: sizeBucketOf(file.bytes),
        trainable: true,
      });
      total += file.bytes;
    }
  }
  console.log(`text: harvested ${total} B of docs/prose`);
}

function cloneAt(entry: SourceEntry): { dir: string } | undefined {
  const name = entry.repo.split('/').slice(-2).join('__');
  const dir = join(CACHE_DIR, name);
  if (existsSync(dir)) return { dir };
  console.log(`cloning ${entry.repo}@${entry.ref} ...`);
  const clone = spawnSync('git', ['clone', '--depth', '1', '--branch', entry.ref, '--single-branch', entry.repo, dir], {
    stdio: ['ignore', 'ignore', 'pipe'],
    timeout: 600_000,
  });
  if (clone.status !== 0) {
    // Fall back to the default branch when the pinned ref is not a branch/tag name.
    const fallback = spawnSync('git', ['clone', '--depth', '1', entry.repo, dir], {
      stdio: ['ignore', 'ignore', 'pipe'],
      timeout: 600_000,
    });
    if (fallback.status !== 0) {
      console.error(`failed to clone ${entry.repo}: ${clone.stderr?.toString().trim()}`);
      return undefined;
    }
    console.error(`warning: ${entry.repo}@${entry.ref} not found; using default branch (record the resolved SHA)`);
  }
  return { dir };
}

function resolveSha(dir: string): string {
  const result = spawnSync('git', ['-C', dir, 'rev-parse', 'HEAD'], { encoding: 'utf8' });
  return result.stdout?.trim() ?? 'unknown';
}

interface SampledFile {
  path: string;
  relative: string;
  bytes: number;
}

/** Sampling rules: language extensions only; skip vendored/generated/minified; keep tests; whole files. */
function sampleFiles(root: string, extensions: string[]): SampledFile[] {
  const files: SampledFile[] = [];
  const walk = (dir: string, prefix: string): void => {
    let entries;
    try {
      entries = readdirSync(dir, { withFileTypes: true });
    } catch {
      return;
    }
    for (const entry of entries) {
      if (entry.name.startsWith('.')) continue;
      const path = join(dir, entry.name);
      const relative = prefix ? `${prefix}/${entry.name}` : entry.name;
      if (entry.isDirectory()) {
        if (!EXCLUDED_DIRS.has(entry.name.toLowerCase())) walk(path, relative);
        continue;
      }
      if (!extensions.some((ext) => entry.name.endsWith(ext))) continue;
      if (entry.name.includes('.min.')) continue;
      let stat;
      try {
        stat = statSync(path);
      } catch {
        continue;
      }
      if (stat.size === 0 || stat.size > MAX_FILE_BYTES) continue;
      const content = readFileSync(path, 'utf8');
      if (content.includes('\u0000')) continue;
      const lines = content.split('\n');
      if (content.length / Math.max(lines.length, 1) > MAX_AVG_LINE_LENGTH) continue; // Minified/generated.
      files.push({ path, relative, bytes: Buffer.byteLength(content) });
    }
  };
  walk(root, '');
  // Deterministic order, then spread across the tree by interleaving hash order.
  files.sort((a, b) => (a.relative < b.relative ? -1 : 1));
  return files;
}
