/** Text-channel competitors, including their complete binary-to-text framing cost. */
import { brotliCompressSync, brotliDecompressSync, constants as zlibConstants, gunzipSync, gzipSync } from 'node:zlib';
import { compressToEncodedURIComponent, decompressFromEncodedURIComponent } from 'lz-string';

export interface Competitor {
  name: string;
  compress(input: string): string;
  decompress(encoded: string): string;
}

interface BinaryCodec {
  compress(bytes: Uint8Array): Uint8Array;
  decompress(bytes: Uint8Array): Uint8Array;
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

export const competitors: Competitor[] = [
  binaryCompetitor('b64url(brotli q11)', {
    compress: (bytes) => brotliCompressSync(bytes, { params: { [zlibConstants.BROTLI_PARAM_QUALITY]: 11 } }),
    decompress: brotliDecompressSync,
  }),
  binaryCompetitor('b64url(brotli q5)', {
    compress: (bytes) => brotliCompressSync(bytes, { params: { [zlibConstants.BROTLI_PARAM_QUALITY]: 5 } }),
    decompress: brotliDecompressSync,
  }),
  binaryCompetitor('b64url(gzip -6)', {
    compress: (bytes) => gzipSync(bytes, { level: 6 }),
    decompress: gunzipSync,
  }),
  {
    name: 'lz-string URI',
    compress: compressToEncodedURIComponent,
    decompress: decompressFromEncodedURIComponent,
  },
];

if (zstdCompressSync && zstdDecompressSync) {
  // oxlint-disable-next-line no-explicit-any -- ZSTD_c_compressionLevel is missing from Bun's constants typings
  const levelKey = (zlibConstants as any).ZSTD_c_compressionLevel as number;
  competitors.push(
    binaryCompetitor('b64url(zstd -19)', {
      compress: (bytes) => zstdCompressSync(bytes, { params: { [levelKey]: 19 } }),
      decompress: zstdDecompressSync,
    }),
    binaryCompetitor('b64url(zstd -3)', {
      compress: (bytes) => zstdCompressSync(bytes, { params: { [levelKey]: 3 } }),
      decompress: zstdDecompressSync,
    })
  );
} else {
  console.error('note: node:zlib zstd not available in this runtime; skipping zstd baselines');
}

function binaryCompetitor(name: string, codec: BinaryCodec): Competitor {
  return {
    name,
    // Unpadded base64url is the shortest standard URL-safe framing and is therefore a
    // stronger baseline than the padded base64 previously used by this benchmark.
    compress: (input) => Buffer.from(codec.compress(encoder.encode(input))).toString('base64url'),
    decompress: (encoded) => decoder.decode(codec.decompress(Buffer.from(encoded, 'base64url'))),
  };
}
