/**
 * Competitor codecs for both benchmark channels: the text channel wraps each binary codec in
 * unpadded base64url (its complete binary-to-text framing cost), while the binary channel
 * measures the raw codec bytes.
 *
 * The primary competitors are the browser-native `CompressionStream` formats (gzip,
 * deflate-raw), measured through the real Web Streams API — exactly what a client-side
 * deployment can use without shipping a codec. brotli/zstd/xz are server-side or CLI-only
 * references.
 */
import { spawnSync } from 'node:child_process';
import { brotliCompressSync, brotliDecompressSync, constants as zlibConstants } from 'node:zlib';
import { compressToEncodedURIComponent, decompressFromEncodedURIComponent } from 'lz-string';

export interface Competitor<Encoded extends string | Uint8Array = string> {
  name: string;
  compress(input: string): Encoded | Promise<Encoded>;
  decompress(encoded: Encoded): string | Promise<string>;
  /**
   * Excluded from the speed benchmark (size and round-trip only). Used for CLI-backed
   * ratio references whose per-document process-spawn overhead would not measure the codec.
   */
  speedExempt?: boolean;
}

interface BinaryCodec {
  name: string;
  compress(bytes: Uint8Array): Uint8Array | Promise<Uint8Array>;
  decompress(bytes: Uint8Array): Uint8Array | Promise<Uint8Array>;
  speedExempt?: boolean;
}

/** Pumps one buffer through a TransformStream (the browser CompressionStream usage). */
async function transformBytes(
  // Wide input side: CompressionStream/DecompressionStream accept any BufferSource.
  stream: TransformStream<ArrayBuffer | ArrayBufferView, Uint8Array>,
  input: Uint8Array
): Promise<Uint8Array> {
  const writer = stream.writable.getWriter();
  const written = (async () => {
    await writer.write(input);
    await writer.close();
    // A stream failure rejects both sides; the read loop below surfaces it, so the
    // mirrored write-side rejection must not become an unhandled rejection.
  })().catch(() => {});
  const chunks: Uint8Array[] = [];
  let total = 0;
  const reader = stream.readable.getReader();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    total += value.length;
  }
  await written;
  const out = new Uint8Array(total);
  let at = 0;
  for (const chunk of chunks) {
    out.set(chunk, at);
    at += chunk.length;
  }
  return out;
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
  // Browser-native primary competitors: the only codecs a client-side deployment gets for
  // free, measured through the actual Web Streams API (per-document stream construction is
  // part of the measured cost, as it is in a real browser).
  {
    name: 'cs gzip',
    compress: (bytes) => transformBytes(new CompressionStream('gzip'), bytes),
    decompress: (bytes) => transformBytes(new DecompressionStream('gzip'), bytes),
  },
  {
    name: 'cs deflate-raw',
    compress: (bytes) => transformBytes(new CompressionStream('deflate-raw'), bytes),
    decompress: (bytes) => transformBytes(new DecompressionStream('deflate-raw'), bytes),
  },
  // Server-side references (not available to browser clients without shipping a codec).
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
      compress: (input) => {
        const out = codec.compress(encoder.encode(input));
        return out instanceof Promise
          ? out.then((bytes) => Buffer.from(bytes).toString('base64url'))
          : Buffer.from(out).toString('base64url');
      },
      decompress: (encoded) => {
        const out = codec.decompress(Buffer.from(encoded, 'base64url'));
        return out instanceof Promise ? out.then((bytes) => decoder.decode(bytes)) : decoder.decode(out);
      },
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
  decompress: (encoded) => {
    const out = codec.decompress(encoded);
    return out instanceof Promise ? out.then((bytes) => decoder.decode(bytes)) : decoder.decode(out);
  },
  speedExempt: codec.speedExempt,
}));

function runPipe(command: string, args: string[], input: Uint8Array): Uint8Array {
  const result = spawnSync(command, args, { input, maxBuffer: 1 << 28 });
  if (result.status !== 0) throw new Error(`${command} exited with ${result.status}: ${result.stderr}`);
  return new Uint8Array(result.stdout);
}
