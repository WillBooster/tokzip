/**
 * Fetches the human-written natural-language corpus per nl-sources.json:
 * - Project Gutenberg plain text (public domain → trainable),
 * - Aozora Bunko via the aozorabunko GitHub mirror (public domain → trainable),
 * - Wikipedia extracts via the MediaWiki action API (CC-BY-SA → benchmark split only).
 * Documents are chunked to realistic payload sizes across the size buckets.
 *
 * Usage: bun scripts/corpus/fetchNl.ts [<locale> ...]
 */
import sources from './nl-sources.json';
import { appendManifest, CORPUS_DIR, sizeBucketOf, writeSample } from './shared.ts';

const CHUNK_TARGETS = [512, 2048, 8192, 24_576];
const MAX_DOC_BYTES = 512 * 1024;

interface WikipediaSource {
  host: string;
  variant?: string;
  titles: string[];
}
interface GitDocsSource {
  repo: string;
  ref: string;
  license: string;
  trainable: boolean;
}
interface LocaleSources {
  gutenberg?: number[];
  aozoraGithub?: string[];
  gitDocs?: GitDocsSource[];
  wikipedia?: WikipediaSource;
}

const counters = new Map<string, number>();

function nextName(locale: string): string {
  const n = counters.get(locale) ?? 0;
  counters.set(locale, n + 1);
  return `${String(n).padStart(5, '0')}.txt`;
}

function saveChunks(locale: string, text: string, source: string, license: string, trainable: boolean): number {
  let saved = 0;
  let offset = 0;
  let chunkIndex = 0;
  while (offset < text.length) {
    // Rotate through the size buckets so every bucket is represented.
    const target = CHUNK_TARGETS[chunkIndex % CHUNK_TARGETS.length]!;
    let end = Math.min(offset + target, text.length);
    const paragraphBreak = text.indexOf('\n\n', end);
    if (paragraphBreak !== -1 && paragraphBreak - end < target) end = paragraphBreak + 2;
    const chunk = text.slice(offset, end).trim();
    offset = end;
    chunkIndex++;
    if (chunk.length < 200) continue;
    const name = nextName(locale);
    writeSample(locale, 'human', name, chunk);
    appendManifest(locale, {
      file: `human/${name}`,
      lang: locale,
      origin: 'human',
      source: `${source}#chunk${chunkIndex}`,
      license,
      sizeBucket: sizeBucketOf(Buffer.byteLength(chunk)),
      trainable,
    });
    saved += Buffer.byteLength(chunk);
  }
  return saved;
}

const sleep = (ms: number): Promise<void> => new Promise((resolve) => setTimeout(resolve, ms));

async function fetchText(url: string, encoding = 'utf8'): Promise<string | undefined> {
  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      const response = await fetch(url, { headers: { 'user-agent': 'tokzip-corpus/1.0 (corpus build script)' } });
      if (response.status === 429 || response.status >= 500) {
        console.error(`  ${url} → HTTP ${response.status}, retrying`);
        await sleep(15_000 * (attempt + 1));
        continue;
      }
      if (!response.ok) {
        console.error(`  ${url} → HTTP ${response.status}`);
        return undefined;
      }
      const buffer = await response.arrayBuffer();
      // Bun types TextDecoder's label narrowly, but shift_jis is supported at runtime.
      // oxlint-disable-next-line unicorn/text-encoding-identifier-case -- cast target must match the TextDecoder label type
      return new TextDecoder(encoding as 'utf-8').decode(buffer).slice(0, MAX_DOC_BYTES);
    } catch (error) {
      console.error(`  ${url} → ${String(error)}`);
      await sleep(5000);
    }
  }
  return undefined;
}

async function fetchGutenberg(locale: string, ids: number[]): Promise<void> {
  for (const id of ids) {
    const text = await fetchText(`https://www.gutenberg.org/cache/epub/${id}/pg${id}.txt`);
    if (!text) continue;
    // Strip the Gutenberg header/footer boilerplate.
    const start = text.indexOf('*** START');
    const end = text.indexOf('*** END');
    const body = text.slice(start !== -1 ? text.indexOf('\n', start) + 1 : 0, end !== -1 ? end : undefined);
    const bytes = saveChunks(locale, body, `gutenberg:pg${id}`, 'Public domain', true);
    console.log(`${locale}: gutenberg pg${id} → ${bytes} B`);
  }
}

/** Aozora Bunko XHTML → plain text (strip ruby annotations and tags). */
function aozoraToText(html: string): string {
  const mainMatch = html.match(/<div class="main_text">([\s\S]*?)<\/div>/);
  const body = mainMatch ? mainMatch[1]! : html;
  return body
    .replaceAll(/<rp>[\s\S]*?<\/rp>/g, '')
    .replaceAll(/<rt>[\s\S]*?<\/rt>/g, '')
    .replaceAll(/<br\s*\/?>/g, '\n')
    .replaceAll(/<[^>]+>/g, '')
    .replaceAll(/&[a-z]+;/g, ' ');
}

async function fetchAozora(locale: string, paths: string[]): Promise<void> {
  for (const path of paths) {
    const url = `https://raw.githubusercontent.com/aozorabunko/aozorabunko/master/${path}`;
    // Aozora Bunko HTML files are Shift_JIS-encoded.
    const html = await fetchText(url, 'shift_jis');
    if (!html) continue;
    const text = aozoraToText(html);
    const bytes = saveChunks(locale, text, `aozora:${path}`, 'Public domain', true);
    console.log(`${locale}: aozora ${path} → ${bytes} B`);
  }
}

async function fetchWikipedia(locale: string, source: WikipediaSource): Promise<void> {
  for (const title of source.titles) {
    await sleep(3000); // Politeness delay: the API rate-limits bursts hard.
    const params = new URLSearchParams({
      action: 'query',
      prop: 'extracts',
      explaintext: '1',
      format: 'json',
      redirects: '1',
      titles: title,
    });
    if (source.variant) params.set('variant', source.variant);
    const raw = await fetchText(`https://${source.host}/w/api.php?${params}`);
    if (!raw) continue;
    let extract = '';
    try {
      const pages = (JSON.parse(raw) as { query: { pages: Record<string, { extract?: string }> } }).query.pages;
      extract = Object.values(pages)[0]?.extract ?? '';
    } catch {
      continue;
    }
    if (!extract) continue;
    // CC-BY-SA (share-alike): benchmark split only, never dictionary training.
    const bytes = saveChunks(locale, extract, `wikipedia:${source.host}/${title}`, 'CC-BY-SA-4.0', false);
    console.log(`${locale}: wikipedia ${title} → ${bytes} B`);
  }
}

/** Modern technical prose from permissively licensed docs-translation repos. */
async function fetchGitDocs(locale: string, entries: GitDocsSource[]): Promise<void> {
  const { spawnSync } = await import('node:child_process');
  const { existsSync, mkdirSync, readdirSync, readFileSync, statSync } = await import('node:fs');
  const { join } = await import('node:path');
  const cacheDir = join(CORPUS_DIR, '.cache');
  mkdirSync(cacheDir, { recursive: true });
  for (const entry of entries) {
    const dir = join(cacheDir, entry.repo.split('/').slice(-2).join('__'));
    if (!existsSync(dir)) {
      const clone = spawnSync('git', ['clone', '--depth', '1', '--branch', entry.ref, entry.repo, dir], {
        stdio: ['ignore', 'ignore', 'pipe'],
        timeout: 600_000,
      });
      if (clone.status !== 0) {
        const fallback = spawnSync('git', ['clone', '--depth', '1', entry.repo, dir], {
          stdio: ['ignore', 'ignore', 'pipe'],
          timeout: 600_000,
        });
        if (fallback.status !== 0) {
          console.error(`  failed to clone ${entry.repo}`);
          continue;
        }
      }
    }
    let bytes = 0;
    const walk = (current: string): void => {
      for (const dirent of readdirSync(current, { withFileTypes: true })) {
        if (dirent.name.startsWith('.') || dirent.name === 'node_modules') continue;
        const path = join(current, dirent.name);
        if (dirent.isDirectory()) {
          walk(path);
          continue;
        }
        if (!dirent.name.endsWith('.md') || statSync(path).size < 500) continue;
        bytes += saveChunks(
          locale,
          readFileSync(path, 'utf8'),
          `${entry.repo}:${dirent.name}`,
          entry.license,
          entry.trainable
        );
      }
    };
    walk(dir);
    console.log(`${locale}: ${entry.repo} docs → ${bytes} B`);
  }
}

const requested = process.argv.slice(2);
const locales = requested.length > 0 ? requested : Object.keys(sources.locales);
for (const locale of locales) {
  const localeSources = (sources.locales as Record<string, LocaleSources>)[locale];
  if (!localeSources) {
    console.error(`unknown locale: ${locale}`);
    continue;
  }
  if (localeSources.gutenberg) await fetchGutenberg(locale, localeSources.gutenberg);
  if (localeSources.aozoraGithub) await fetchAozora(locale, localeSources.aozoraGithub);
  if (localeSources.gitDocs) await fetchGitDocs(locale, localeSources.gitDocs);
  if (localeSources.wikipedia) await fetchWikipedia(locale, localeSources.wikipedia);
}
