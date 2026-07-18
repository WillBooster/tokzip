import { expect, test } from 'bun:test';
import { compress, decompress, TokzipDecodeError } from '../src/index.ts';
import '../src/languages/index.ts';
import { FLAG_FENCED } from '../src/format.ts';
import { typescriptModule } from '../src/generated/typescript.ts';

const RADIX64 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_';
const MODES = ['fast', 'small'] as const;

function flagsOf(frame: string): number {
  return RADIX64.indexOf(frame[2]!);
}

function isFenced(frame: string): boolean {
  return (flagsOf(frame) & FLAG_FENCED) !== 0;
}

const TS_CODE = `export function greet(name: string): string {
  const message = \`Hello, \${name}!\`;
  console.log(message);
  return message;
}
export interface Config {
  readonly enabled: boolean;
  readonly timeout: number;
}
`;
const PY_CODE = `import os\nfrom typing import Optional\n\ndef resolve(path: str) -> Optional[str]:\n    return os.path.abspath(path) if os.path.exists(path) else None\n`;
// Fence content that is guaranteed to be coverable by extended dictionary matches
// regardless of how the dictionary was trained: a literal printable-ASCII run taken from
// the shipped typescript dictionary itself. Hand-written sample code would silently stop
// matching whenever the dictionary is retrained (e.g. at a different --budget).
const DICT_TS_CODE = `${TS_CODE}${dictionarySnippet()}`;

function dictionarySnippet(): string {
  const dictText = new TextDecoder('utf-8', { fatal: false }).decode(typescriptModule.dictionarySuffix);
  // Longest run of printable ASCII / newlines, so the snippet is honest UTF-8 text.
  let bestStart = 0;
  let bestLength = 0;
  let runStart = 0;
  for (let i = 0; i <= dictText.length; i++) {
    const code = i < dictText.length ? dictText.codePointAt(i)! : -1;
    if (code === 10 || (code >= 32 && code < 127)) continue;
    if (i - runStart > bestLength) {
      bestStart = runStart;
      bestLength = i - runStart;
    }
    runStart = i + 1;
  }
  if (bestLength < 64) throw new Error('typescript dictionary has no usable ASCII run for the fence tests');
  return `${dictText.slice(bestStart, bestStart + Math.min(bestLength, 400))}\n`;
}

function docWith(label: string, code = TS_CODE): string {
  return `# Usage\n\nInstall the package and call the function as follows.\n\n\`\`\`${label}\n${code}\`\`\`\n\nThe function returns the formatted message.\n`;
}

test('fenced round-trips with extended matches in both modes and frame languages', () => {
  const doc = `${docWith('ts')}\nAnd in Python:\n\n\`\`\`python\n${PY_CODE}\`\`\`\n`;
  for (const mode of MODES) {
    for (const language of ['none', 'en-US', 'text']) {
      const frame = compress(doc, { language, mode });
      expect(decompress(frame)).toBe(doc);
    }
    const byteFrame = compress(new TextEncoder().encode(doc), { language: 'none', mode });
    expect(decompress(byteFrame)).toEqual(new TextEncoder().encode(doc));
  }
});

test('extended matches set flag bit 3 and shrink output versus an unknown label', () => {
  for (const mode of MODES) {
    const fenced = compress(docWith('ts', DICT_TS_CODE), { language: 'none', mode });
    const unknown = compress(docWith('mystery', DICT_TS_CODE), { language: 'none', mode });
    expect(isFenced(fenced)).toBe(true);
    expect(isFenced(unknown)).toBe(false);
    expect(fenced.length).toBeLessThan(unknown.length);
  }
});

test('label aliases and ASCII case both resolve', () => {
  for (const label of ['TS', 'TypeScript', 'tsx', 'typescript']) {
    const frame = compress(docWith(label), { language: 'none', mode: 'fast' });
    expect(isFenced(frame)).toBe(true);
    expect(decompress(frame)).toBe(docWith(label));
  }
});

test('a block labeled with the frame language stays a plain unfenced frame', () => {
  const frame = compress(docWith('ts'), { language: 'typescript', mode: 'small' });
  expect(isFenced(frame)).toBe(false);
  expect(decompress(frame)).toBe(docWith('ts'));
});

test('CRLF fence lines resolve the label and round-trip', () => {
  const doc = docWith('ts').replaceAll('\n', '\r\n');
  for (const mode of MODES) {
    const frame = compress(doc, { language: 'none', mode });
    expect(isFenced(frame)).toBe(true);
    expect(decompress(frame)).toBe(doc);
  }
});

test('longer fences nest plain-fence content; unclosed blocks extend to the end', () => {
  const nested = `\`\`\`\`ts\n${DICT_TS_CODE}\`\`\`ts\ninner fence line is content\n\`\`\`\`\n`;
  const unclosed = `Intro line.\n\`\`\`ts\n${DICT_TS_CODE}`;
  for (const doc of [nested, unclosed]) {
    for (const mode of MODES) {
      const frame = compress(doc, { language: 'none', mode });
      expect(isFenced(frame)).toBe(true);
      expect(decompress(frame)).toBe(doc);
    }
  }
});

test('a lone CR is label content, not a separator (only trailing CRs are trimmed)', () => {
  for (const opener of ['```\rts', '```ts\rfoo']) {
    const doc = `${opener}\n${TS_CODE}\`\`\`\n`;
    const frame = compress(doc, { language: 'none', mode: 'small' });
    expect(isFenced(frame)).toBe(false);
    expect(decompress(frame)).toBe(doc);
  }
});

test('blocks below the extension-content threshold stay plain unfenced frames', () => {
  const doc = 'Intro.\n```ts\nconst tiny = 1;\n```\n';
  for (const mode of MODES) {
    const frame = compress(doc, { language: 'none', mode });
    expect(isFenced(frame)).toBe(false);
    expect(decompress(frame)).toBe(doc);
  }
});

test('indented fences and info strings containing backticks do not extend', () => {
  for (const opener of ['  ```ts', '```ts `tick`']) {
    const doc = `${opener}\n${TS_CODE}\`\`\`\n`;
    const frame = compress(doc, { language: 'none', mode: 'small' });
    expect(isFenced(frame)).toBe(false);
    expect(decompress(frame)).toBe(doc);
  }
});

test('flag bit 3 on a frame without extended matches decodes identically', () => {
  const source = 'const value = JSON.stringify({ answer: 42 });\n'.repeat(3);
  for (const mode of MODES) {
    const frame = compress(source, { language: 'typescript', mode });
    expect(isFenced(frame)).toBe(false);
    const flipped = frame.slice(0, 2) + RADIX64[flagsOf(frame) | FLAG_FENCED]! + frame.slice(3);
    expect(decompress(flipped)).toBe(source);
  }
});

test('decoding an extended match without the block language registered throws', () => {
  const frame = compress(docWith('ts'), { language: 'none', mode: 'fast' });
  expect(isFenced(frame)).toBe(true);
  // A fresh process registering only core (id 0) must reject the typescript extension.
  const probe = `import { decompress, TokzipDecodeError } from '${import.meta.dir}/../src/index.ts';
try {
  decompress(process.argv.at(-1));
  console.log('NO_ERROR');
} catch (error) {
  console.log(error instanceof TokzipDecodeError ? 'DECODE_ERROR:' + error.message : 'OTHER');
}`;
  const result = Bun.spawnSync(['bun', '-e', probe, frame]);
  expect(result.stdout.toString().trim()).toBe('DECODE_ERROR:unknown language id: 16');
});

test('TokzipDecodeError stays typed for in-process extended-match bounds violations', () => {
  // Flag flipped on a plain frame plus a forged oversized dict offset is covered by the
  // fenced round-trip suite; here the plain out-of-bounds path must still throw when the
  // flag is clear (extended offsets are invalid without it).
  const frame = compress(docWith('ts'), { language: 'none', mode: 'fast' });
  const cleared = frame.slice(0, 2) + RADIX64[flagsOf(frame) & ~FLAG_FENCED]! + frame.slice(3);
  expect(() => decompress(cleared)).toThrow(TokzipDecodeError);
});
