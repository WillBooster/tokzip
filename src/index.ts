import wasmModuleOrPath from '../wasm/tokzip.wasm';

/** Thrown when a frame is truncated, corrupt, from another format version, or fails its CRC. */
export class TokzipDecodeError extends Error {
  constructor(readonly code: number) {
    super(`tokzip: ${DECODE_ERROR_MESSAGES[code] ?? `decode error ${code}`}`);
    this.name = 'TokzipDecodeError';
  }
}

const DECODE_ERROR_MESSAGES: Record<number, string> = {
  1: 'frame truncated',
  2: 'bad magic byte',
  3: 'unsupported format version',
  4: 'content checksum mismatch',
  5: 'corrupt compressed body',
};

interface Exports {
  memory: WebAssembly.Memory;
  tokzip_alloc(len: number): number;
  tokzip_free(ptr: number, len: number): void;
  tokzip_compress(ptr: number, len: number, isBytes: number): number;
  tokzip_decompress(ptr: number, len: number): number;
  tokzip_out_is_bytes(): number;
  tokzip_out_ptr(): number;
  tokzip_out_len(): number;
}

const textEncoder = new TextEncoder();
// Fatal decoding: a string frame whose content is not valid UTF-8 is corrupt, never U+FFFD.
const textDecoder = new TextDecoder('utf-8', { fatal: true, ignoreBOM: true });
const wasm = new WebAssembly.Instance(loadModule()).exports as unknown as Exports;

/**
 * Compresses a string (stored as UTF-8; must be well-formed UTF-16) or raw bytes into a
 * self-describing binary frame.
 */
export function compress(input: string | Uint8Array): Uint8Array {
  const isString = typeof input === 'string';
  // A lone surrogate cannot round-trip through UTF-8 (TextEncoder would substitute U+FFFD),
  // so it is refused rather than silently altered.
  if (isString && !input.isWellFormed()) {
    throw new RangeError('tokzip: string contains a lone surrogate and cannot be stored losslessly');
  }
  const bytes = isString ? textEncoder.encode(input) : input;
  return withInput(bytes, (ptr) => {
    wasm.tokzip_compress(ptr, bytes.length, isString ? 0 : 1);
    return takeOutput();
  });
}

/** Decompresses a frame; returns a string or bytes according to what was compressed. */
export function decompress(frame: Uint8Array): string | Uint8Array {
  return withInput(frame, (ptr) => {
    const code = wasm.tokzip_decompress(ptr, frame.length);
    if (code !== 0) throw new TokzipDecodeError(code);
    const out = takeOutput();
    if (wasm.tokzip_out_is_bytes() !== 0) return out;
    try {
      return textDecoder.decode(out);
    } catch {
      throw new TokzipDecodeError(5);
    }
  });
}

function withInput<T>(bytes: Uint8Array, run: (ptr: number) => T): T {
  const ptr = wasm.tokzip_alloc(bytes.length);
  try {
    new Uint8Array(wasm.memory.buffer, ptr, bytes.length).set(bytes);
    return run(ptr);
  } finally {
    wasm.tokzip_free(ptr, bytes.length);
  }
}

/** Copies the module-owned output buffer out of wasm memory (it is reused by the next call). */
function takeOutput(): Uint8Array {
  return new Uint8Array(wasm.memory.buffer, wasm.tokzip_out_ptr(), wasm.tokzip_out_len()).slice();
}

function loadModule(): WebAssembly.Module {
  if (typeof wasmModuleOrPath !== 'string') return wasmModuleOrPath;
  // Bun resolves a .wasm import to its file path; Cloudflare Workers and bundlers hand over a
  // compiled module instead (they forbid compiling wasm from bytes at runtime).
  const fs = (process as { getBuiltinModule?: (name: string) => unknown }).getBuiltinModule?.('node:fs') as
    | { readFileSync(path: string): Uint8Array }
    | undefined;
  if (!fs) throw new Error('tokzip: cannot load tokzip.wasm in this runtime');
  return new WebAssembly.Module(fs.readFileSync(wasmModuleOrPath));
}
