import { spawnSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import { join, resolve } from 'node:path';

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

let detectedCorpusDirs: string[] | undefined;

/**
 * Every corpus root benchmarks read, in a stable order: the public corpus plus the sibling
 * `tokzip-corpus-private` checkout when one exists. Detection is automatic and freshens the
 * private checkout with `git pull` so local benchmarks always see its latest committed
 * samples. An explicit `TOKZIP_CORPUS_DIR` means "use exactly this corpus", so it disables
 * the detection; CI without the private checkout is unaffected either way.
 *
 * Detection (and its `git pull` side effect) runs lazily on first call, so importing
 * `CORPUS_DIR` alone — as training does — never touches the private checkout. Training
 * intentionally stays on `CORPUS_DIR` only: generated dictionaries embed literal fragments
 * of their training documents and are committed to this public repository, so private
 * production content must never flow into them.
 */
export function corpusDirs(): string[] {
  detectedCorpusDirs ??= detectCorpusDirs();
  return detectedCorpusDirs;
}

function detectCorpusDirs(): string[] {
  if (process.env['TOKZIP_CORPUS_DIR']) return [CORPUS_DIR];
  const privateRepoDir = resolve(import.meta.dir, '../../tokzip-corpus-private');
  const privateCorpusDir = join(privateRepoDir, 'corpus');
  if (!existsSync(privateCorpusDir)) return [CORPUS_DIR];
  // A stale private corpus would silently skew benchmark fingerprints; a failed pull
  // (offline, diverged branch) only degrades to the existing checkout. Credential prompts
  // must fail fast too: git asks on /dev/tty even with piped stdio, which would otherwise
  // block the benchmark until the timeout.
  const pull = spawnSync('git', ['-C', privateRepoDir, 'pull', '--ff-only'], {
    encoding: 'utf8',
    timeout: 60_000,
    env: { ...process.env, GIT_TERMINAL_PROMPT: '0', GIT_SSH_COMMAND: 'ssh -o BatchMode=yes' },
  });
  if (pull.status === 0) {
    console.error(`private corpus: ${privateCorpusDir} (pulled)`);
  } else {
    // spawnSync reports a failed launch (missing git, timeout) via `error` with null stderr.
    // git itself buries the reason under fetch progress lines, so prefer the fatal: line.
    const stderrLines = (pull.stderr ?? '').split('\n').filter((line) => line.trim());
    const reason =
      pull.error?.message ??
      stderrLines.findLast((line) => line.startsWith('fatal:')) ??
      stderrLines.at(-1) ??
      'unknown error';
    console.error(`private corpus: ${privateCorpusDir} (git pull failed, using existing checkout: ${reason.trim()})`);
  }
  return [CORPUS_DIR, privateCorpusDir];
}
