// Node and Bun entry: the wasm module is compiled from the file shipped beside this package
// (bundlers that trace `new URL(..., import.meta.url)` carry the file along). Cloudflare
// Workers resolve the `workerd` export condition to `workers.ts` instead, which imports the
// module the way Workers require.
/// <reference lib="es2024.string" />
import { readFileSync } from 'node:fs';
import { createCodec, type DecompressOptions } from './core.ts';

export { type Codec, type DecompressOptions, FORMAT_VERSION, TokzipDecodeError } from './core.ts';

const codec = createCodec(() => new WebAssembly.Module(readFileSync(new URL('../wasm/tokzip.wasm', import.meta.url))));

/**
 * Compresses a string (stored as UTF-8; must be well-formed UTF-16) into a self-describing
 * binary frame. The first call in a process builds the detection table and decodes the
 * dictionaries it uses (~10 ms); later calls do not.
 */
export function compress(text: string): Uint8Array {
  return codec.compress(text);
}

/** Decompresses a frame produced by `compress` back to the original string. */
export function decompress(frame: Uint8Array, options?: DecompressOptions): string {
  return codec.decompress(frame, options);
}
