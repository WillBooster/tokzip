// This file is the package entry, so a consumer's type-check compiles it under the consumer's
// own `lib`/`types`: the references below supply what it needs (the `.wasm` module declaration
// and ES2024's `String.prototype.isWellFormed`) instead of relying on this repository's tsconfig.
// oxlint-disable-next-line typescript/triple-slash-reference -- an import cannot hand consumers the ambient `*.wasm` module declaration
/// <reference path="./wasm.d.ts" />
/// <reference lib="es2024.string" />
import wasmModuleOrPath from '../wasm/tokzip.wasm';

/**
 * Thrown when a frame is truncated, corrupt, from another format version, fails its CRC, or
 * declares more content than `maxLength` allows.
 */
export class TokzipDecodeError extends Error {
  readonly code: number;

  constructor(code: number) {
    super(`tokzip: ${DECODE_ERROR_MESSAGES[code] ?? `decode error ${code}`}`);
    this.name = 'TokzipDecodeError';
    this.code = code;
  }
}

const DECODE_ERROR_MESSAGES: Record<number, string> = {
  1: 'frame truncated',
  2: 'bad magic byte',
  3: 'unsupported format version',
  4: 'content checksum mismatch',
  5: 'corrupt compressed body',
  6: 'content too large for maxLength or memory',
};

export interface DecompressOptions {
  /**
   * Upper bound on the decompressed length in bytes; a frame declaring more is rejected before
   * anything is allocated. Unlimited by default — set it when decompressing frames from an
   * untrusted source, since a small frame of repetitive content can legitimately expand
   * thousands of times (without a limit, a frame the memory cannot hold still fails with this
   * error rather than trapping).
   */
  maxLength?: number;
}

interface Exports {
  memory: WebAssembly.Memory;
  tokzip_alloc(len: number): number;
  tokzip_free(ptr: number, len: number): void;
  tokzip_compress(ptr: number, len: number): void;
  tokzip_decompress(ptr: number, len: number, maxLen: number): number;
  tokzip_out_ptr(): number;
  tokzip_out_len(): number;
}

const textEncoder = new TextEncoder();
// Fatal decoding: a frame whose content is not valid UTF-8 is corrupt, never U+FFFD.
const textDecoder = new TextDecoder('utf-8', { fatal: true, ignoreBOM: true });
const wasmModule = loadModule();
let wasm = instantiate();

/** Compresses a string (stored as UTF-8; must be well-formed UTF-16) into a self-describing binary frame. */
export function compress(text: string): Uint8Array {
  // A lone surrogate cannot round-trip through UTF-8 (TextEncoder would substitute U+FFFD),
  // so it is refused rather than silently altered.
  if (!text.isWellFormed()) {
    throw new RangeError('tokzip: string contains a lone surrogate and cannot be stored losslessly');
  }
  return withInput(text, (ptr, len) => {
    wasm.tokzip_compress(ptr, len);
    return takeOutput();
  });
}

/** Decompresses a frame produced by `compress` back to the original string. */
export function decompress(frame: Uint8Array, { maxLength = Infinity }: DecompressOptions = {}): string {
  // A negative limit would wrap to a huge unsigned length at the wasm boundary (NaN fails too).
  if (!(maxLength >= 0)) throw new RangeError(`tokzip: maxLength must be a non-negative number, got ${maxLength}`);
  // wasm32 lengths are 32-bit; anything above cannot be a frame's length anyway.
  const maxLen = Math.min(maxLength, 0xFF_FF_FF_FF);
  return withInput(frame, (ptr, len) => {
    const code = wasm.tokzip_decompress(ptr, len, maxLen);
    if (code !== 0) throw new TokzipDecodeError(code);
    try {
      return textDecoder.decode(takeOutput());
    } catch {
      throw new TokzipDecodeError(5);
    }
  });
}

/**
 * Places `input` in wasm memory for the duration of `run`. A string is encoded as UTF-8 straight
 * into an exactly sized module buffer, so no intermediate copy of a large document is held on
 * the JS heap. (Exact sizing also matters for correctness: Bun 1.3 `encodeInto` writes U+FFFD
 * for a 4-byte character when only 3 bytes of room remain instead of stopping before it, so
 * the buffer is never filled in several passes.)
 *
 * A trap (the module ran out of memory) leaves the instance's heap in an unknown state, so the
 * instance is replaced before the error propagates and later calls start from a fresh one.
 */
function withInput<T>(input: string | Uint8Array, run: (ptr: number, len: number) => T): T {
  const len = typeof input === 'string' ? utf8Length(input) : input.length;
  let trapped = false;
  let ptr: number | undefined;
  try {
    ptr = wasm.tokzip_alloc(len);
    const view = new Uint8Array(wasm.memory.buffer, ptr, len);
    if (typeof input === 'string') textEncoder.encodeInto(input, view);
    else view.set(input);
    return run(ptr, len);
  } catch (error) {
    if (error instanceof WebAssembly.RuntimeError) {
      trapped = true;
      wasm = instantiate();
    }
    throw error;
  } finally {
    if (!trapped && ptr !== undefined) wasm.tokzip_free(ptr, len);
  }
}

/** UTF-8 length of a well-formed string. */
function utf8Length(text: string): number {
  let length = 0;
  for (let i = 0; i < text.length; i++) {
    const codePoint = text.codePointAt(i) ?? 0;
    if (codePoint < 0x80) length += 1;
    else if (codePoint < 0x8_00) length += 2;
    else if (codePoint < 0x1_00_00) length += 3;
    else {
      // A supplementary code point occupies two UTF-16 units.
      length += 4;
      i++;
    }
  }
  return length;
}

/** Copies the module-owned output buffer out of wasm memory (it is reused by the next call). */
function takeOutput(): Uint8Array {
  return new Uint8Array(wasm.memory.buffer, wasm.tokzip_out_ptr(), wasm.tokzip_out_len()).slice();
}

function instantiate(): Exports {
  return new WebAssembly.Instance(wasmModule).exports as unknown as Exports;
}

function loadModule(): WebAssembly.Module {
  if (typeof wasmModuleOrPath !== 'string') return wasmModuleOrPath;
  // Bun resolves a .wasm import to its file path; Cloudflare Workers and bundlers hand over a
  // compiled module instead (they forbid compiling wasm from bytes at runtime).
  // Looked up on `globalThis` so neither the type-check nor the runtime needs a `process` global.
  const process = (globalThis as { process?: { getBuiltinModule?: (name: string) => unknown } }).process;
  const fs = process?.getBuiltinModule?.('node:fs') as
    | { readFileSync(path: string): Uint8Array<ArrayBuffer> }
    | undefined;
  if (!fs) throw new Error('tokzip: cannot load tokzip.wasm in this runtime');
  return new WebAssembly.Module(fs.readFileSync(wasmModuleOrPath));
}
