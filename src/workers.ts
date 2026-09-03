// Cloudflare Workers entry (the `workerd` export condition): Workers forbid compiling wasm
// from bytes at runtime, so the module is imported, which wrangler and the Cloudflare Vite
// plugin turn into a compiled `WebAssembly.Module`.
/// <reference lib="es2024.string" />
// oxlint-disable-next-line typescript/triple-slash-reference -- an import cannot hand consumers the ambient `*.wasm` module declaration
/// <reference path="./wasm.d.ts" />
import wasmModule from '../wasm/tokzip.wasm';
import { createCodec, type DecompressOptions } from './core.ts';

export { type Codec, type DecompressOptions, FORMAT_VERSION, TokzipDecodeError } from './core.ts';

const codec = createCodec(() => wasmModule);

/**
 * Compresses a string (stored as UTF-8; must be well-formed UTF-16) into a self-describing
 * binary frame. The first call in a isolate decodes the language dictionaries (tens of
 * milliseconds of CPU time); later calls do not.
 */
export function compress(text: string): Uint8Array {
  return codec.compress(text);
}

/** Decompresses a frame produced by `compress` back to the original string. */
export function decompress(frame: Uint8Array, options?: DecompressOptions): string {
  return codec.decompress(frame, options);
}
