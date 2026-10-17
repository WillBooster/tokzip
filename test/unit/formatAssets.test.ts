import { expect, test } from 'bun:test';
import { createHash } from 'node:crypto';
import { readdirSync, readFileSync } from 'node:fs';
import assets from '../fixtures/format2Assets.json';

const root = new URL('../../', import.meta.url);

test('preserves every format-2 dictionary and trained prior', () => {
  const paths = ['dict', 'priors'].flatMap((directory) =>
    readdirSync(new URL(directory, root))
      .filter((name) => name.endsWith('.bin'))
      .map((name) => `${directory}/${name}`)
  );
  expect(paths.toSorted()).toEqual(Object.keys(assets.sha256).toSorted());
  for (const [path, expected] of Object.entries(assets.sha256)) {
    const actual = createHash('sha256')
      .update(readFileSync(new URL(path, root)))
      .digest('hex');
    expect(actual, path).toBe(expected);
  }
});
