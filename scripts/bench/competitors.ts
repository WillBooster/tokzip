/**
 * Text-channel competitor codecs for the benchmark. Each general-purpose binary
 * compressor is charged the base64 cost its binary output would pay to travel on a text
 * channel — the +33% tax tokzip avoids by emitting safe ASCII directly.
 */
import { brotliCompressSync, constants as zlibConstants, gzipSync } from 'node:zlib';

export interface Competitor {
  name: string;
  /** Length in characters of the base64 text frame carrying the compressed `bytes`. */
  encodedLength(bytes: Uint8Array): number;
}

// oxlint-disable-next-line no-explicit-any -- zstd is only in newer Node/Bun typings
const zstdCompressSync = (await import('node:zlib').then((m) => (m as any).zstdCompressSync)) as
  | ((data: Uint8Array, options?: unknown) => Uint8Array)
  | undefined;

const base64Length = (byteLength: number): number => Math.ceil(byteLength / 3) * 4;

const brotli = (bytes: Uint8Array, quality: number): number =>
  base64Length(brotliCompressSync(bytes, { params: { [zlibConstants.BROTLI_PARAM_QUALITY]: quality } }).length);

export const competitors: Competitor[] = [
  { name: 'b64(brotli q11)', encodedLength: (bytes) => brotli(bytes, 11) },
  { name: 'b64(brotli q5)', encodedLength: (bytes) => brotli(bytes, 5) },
  { name: 'b64(gzip -6)', encodedLength: (bytes) => base64Length(gzipSync(bytes, { level: 6 }).length) },
];

if (zstdCompressSync) {
  // oxlint-disable-next-line no-explicit-any -- ZSTD_c_compressionLevel is missing from Bun's constants typings
  const levelKey = (zlibConstants as any).ZSTD_c_compressionLevel as number;
  const zstd = (bytes: Uint8Array, level: number): number =>
    base64Length(zstdCompressSync(bytes, { params: { [levelKey]: level } }).length);
  competitors.push(
    { name: 'b64(zstd -19)', encodedLength: (bytes) => zstd(bytes, 19) },
    { name: 'b64(zstd -3)', encodedLength: (bytes) => zstd(bytes, 3) }
  );
} else {
  console.error('note: node:zlib zstd not available in this runtime; skipping zstd baselines');
}
