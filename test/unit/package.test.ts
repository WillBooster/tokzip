import { expect, test } from 'bun:test';
import { cpSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';

import { z } from 'zod';

const root = resolve(import.meta.dir, '../..');
const legalFiles = [
  'LICENSE',
  'NOTICE',
  'FORMAT.md',
  ...readdirSync(join(root, 'THIRD_PARTY_NOTICES'), { recursive: true, encoding: 'utf8' })
    .filter((path) => statSync(join(root, 'THIRD_PARTY_NOTICES', path)).isFile())
    .map((path) => `THIRD_PARTY_NOTICES/${path}`),
];

test('ships a usable package with its license material and excludes adjacent private artifacts', async () => {
  mkdirSync(join(root, '.tmp'), { recursive: true });
  const directory = mkdtempSync(join(root, '.tmp/package-'));
  const staging = join(directory, 'staging');
  try {
    stagePackage(staging);
    const packed = await run(['npm', 'pack', '--ignore-scripts', '--json', '--pack-destination', directory], staging);
    const [archive] = z
      .tuple([z.object({ filename: z.string(), files: z.array(z.object({ path: z.string() })) })])
      .parse(JSON.parse(packed));
    expect(archive.files.some(({ path }) => path.includes('publicationCanary'))).toBe(false);
    for (const runtime of ['node', process.execPath]) {
      await verifyConsumer(
        join(directory, archive.filename),
        join(directory, runtime === 'node' ? 'node' : 'bun'),
        runtime
      );
    }
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
}, 120_000);

async function verifyConsumer(archive: string, directory: string, runtime: string): Promise<void> {
  mkdirSync(directory);
  writeFileSync(
    join(directory, 'package.json'),
    JSON.stringify({ private: true, dependencies: { tokzip: `file:${archive}` } })
  );
  const install =
    runtime === 'node'
      ? ['npm', 'install', '--ignore-scripts', '--omit=dev', '--no-audit', '--no-fund']
      : [runtime, 'install', '--ignore-scripts', '--production'];
  await run(install, directory);
  for (const path of legalFiles) {
    expect(
      readFileSync(join(directory, 'node_modules/tokzip', path)),
      `Missing or changed license material: ${path}`
    ).toEqual(readFileSync(join(root, path)));
  }
  const text = 'A public package must preserve this Unicode record. 日本語 ✅\n'.repeat(100);
  const output = await run(
    [
      runtime,
      '--input-type=module',
      '--eval',
      `
    import { compress, decompress } from 'tokzip';
    const text = ${JSON.stringify(text)};
    process.stdout.write(decompress(compress(text)));
  `,
    ],
    directory
  );
  expect(output).toBe(text);
}

function stagePackage(directory: string): void {
  mkdirSync(directory);
  for (const path of [
    'package.json',
    'README.md',
    'dist',
    'src',
    'wasm',
    'THIRD_PARTY_NOTICES',
    'LICENSE',
    'NOTICE',
    'FORMAT.md',
  ]) {
    cpSync(join(root, path), join(directory, path), { recursive: true });
  }
  for (const path of [
    'publicationCanary.jsonl',
    'src/publicationCanary.ts',
    'dist/publicationCanary.js',
    'wasm/publicationCanary.jsonl',
  ]) {
    writeFileSync(join(directory, path), 'Synthetic private-artifact canary; never publish.');
  }
}

async function run(command: string[], cwd: string): Promise<string> {
  const child = Bun.spawn(command, { cwd, stdout: 'pipe', stderr: 'pipe' });
  const [status, stdout, stderr] = await Promise.all([
    child.exited,
    new Response(child.stdout).text(),
    new Response(child.stderr).text(),
  ]);
  expect(status, `${command[0]} failed: ${stderr}`).toBe(0);
  return stdout;
}
