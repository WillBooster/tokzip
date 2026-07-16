/**
 * Competitor codecs for both benchmark channels: the text channel wraps each binary codec in
 * unpadded base64url (its complete binary-to-text framing cost), while the binary channel
 * measures the raw codec bytes.
 */
import { spawnSync } from 'node:child_process';
import { brotliCompressSync, brotliDecompressSync, constants as zlibConstants, gunzipSync, gzipSync } from 'node:zlib';
import { compressToEncodedURIComponent, decompressFromEncodedURIComponent } from 'lz-string';

export interface Competitor<Encoded extends string | Uint8Array = string> {
  name: string;
  compress(input: string): Encoded;
  decompress(encoded: Encoded): string;
  /**
   * Excluded from the speed benchmark (size and round-trip only). Used for CLI-backed
   * ratio references whose per-document process-spawn overhead would not measure the codec.
   */
  speedExempt?: boolean;
}

interface BinaryCodec {
  name: string;
  compress(bytes: Uint8Array): Uint8Array;
  decompress(bytes: Uint8Array): Uint8Array;
  speedExempt?: boolean;
}

// oxlint-disable-next-line no-explicit-any -- zstd is only in newer Node/Bun typings
const zstdModule = (await import('node:zlib')) as any;
const zstdCompressSync = zstdModule.zstdCompressSync as
  | ((data: Uint8Array, options?: unknown) => Uint8Array)
  | undefined;
const zstdDecompressSync = zstdModule.zstdDecompressSync as ((data: Uint8Array) => Uint8Array) | undefined;

const encoder = new TextEncoder();
// ignoreBOM so the baselines are measured on the same lossless contract as tokzip: without it a
// leading U+FEFF is swallowed on decode and every codec "fails" the round-trip check.
const decoder = new TextDecoder('utf-8', { fatal: true, ignoreBOM: true });

const binaryCodecs: BinaryCodec[] = [
  {
    name: 'brotli q11',
    compress: (bytes) => brotliCompressSync(bytes, { params: { [zlibConstants.BROTLI_PARAM_QUALITY]: 11 } }),
    decompress: brotliDecompressSync,
  },
  {
    name: 'brotli q5',
    compress: (bytes) => brotliCompressSync(bytes, { params: { [zlibConstants.BROTLI_PARAM_QUALITY]: 5 } }),
    decompress: brotliDecompressSync,
  },
  {
    name: 'gzip -6',
    compress: (bytes) => gzipSync(bytes, { level: 6 }),
    decompress: gunzipSync,
  },
];

if (zstdCompressSync && zstdDecompressSync) {
  // oxlint-disable-next-line no-explicit-any -- ZSTD_c_compressionLevel is missing from Bun's constants typings
  const levelKey = (zlibConstants as any).ZSTD_c_compressionLevel as number;
  binaryCodecs.push(
    {
      name: 'zstd -19',
      compress: (bytes) => zstdCompressSync(bytes, { params: { [levelKey]: 19 } }),
      decompress: zstdDecompressSync,
    },
    {
      name: 'zstd -3',
      compress: (bytes) => zstdCompressSync(bytes, { params: { [levelKey]: 3 } }),
      decompress: zstdDecompressSync,
    }
  );
} else {
  console.error('note: node:zlib zstd not available in this runtime; skipping zstd baselines');
}

// xz (LZMA2) is the strongest widely deployed general-purpose ratio reference. It has no
// JS-runtime implementation, so it runs via the system CLI and is size/round-trip only.
const xz = spawnSync('xz', ['--version'], { encoding: 'utf8' });
if (xz.status === 0) {
  binaryCodecs.push({
    name: 'xz -9e',
    compress: (bytes) => runPipe('xz', ['-9e', '--format=xz', '-T1', '-c'], bytes),
    decompress: (bytes) => runPipe('xz', ['-d', '-c'], bytes),
    speedExempt: true,
  });
} else {
  console.error('note: xz CLI not available; skipping the xz -9e ratio reference');
}

/** Text-channel competitors: binary codecs behind unpadded base64url, plus lz-string. */
export const competitors: Competitor[] = [
  ...binaryCodecs.map(
    (codec): Competitor => ({
      name: `b64url(${codec.name})`,
      // Unpadded base64url is the shortest standard URL-safe framing and is therefore a
      // stronger baseline than the padded base64 previously used by this benchmark.
      compress: (input) => Buffer.from(codec.compress(encoder.encode(input))).toString('base64url'),
      decompress: (encoded) => decoder.decode(codec.decompress(Buffer.from(encoded, 'base64url'))),
      speedExempt: codec.speedExempt,
    })
  ),
  {
    name: 'lz-string URI',
    compress: compressToEncodedURIComponent,
    decompress: decompressFromEncodedURIComponent,
  },
];

/** Binary-channel competitors: the raw codec bytes with no text framing. */
export const binaryCompetitors: Competitor<Uint8Array>[] = binaryCodecs.map((codec) => ({
  name: codec.name,
  compress: (input) => codec.compress(encoder.encode(input)),
  decompress: (encoded) => decoder.decode(codec.decompress(encoded)),
  speedExempt: codec.speedExempt,
}));

function runPipe(command: string, args: string[], input: Uint8Array): Uint8Array {
  const result = spawnSync(command, args, { input, maxBuffer: 1 << 28 });
  if (result.status !== 0) throw new Error(`${command} exited with ${result.status}: ${result.stderr}`);
  return new Uint8Array(result.stdout);
}
