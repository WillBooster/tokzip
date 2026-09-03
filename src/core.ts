// The codec over a compiled wasm module; the package entries only differ in how they obtain
// the module (`index.ts` reads the file, `workers.ts` imports it as a module).

/** The frame format version `compress` writes; `decompress` reads this version only. */
export const FORMAT_VERSION = 1;

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

export interface Codec {
  /**
   * Compresses a string (stored as UTF-8; must be well-formed UTF-16) into a self-describing
   * binary frame. The first call in a process builds the detection table and decodes the
   * dictionaries it uses (~10 ms); later calls do not.
   */
  compress: (text: string) => Uint8Array;
  /** Decompresses a frame produced by `compress` back to the original string. */
  decompress: (frame: Uint8Array, options?: DecompressOptions) => string;
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

/** The codec over `loadModule()`, which is called once, on the first compress or decompress. */
export function createCodec(loadModule: () => WebAssembly.Module): Codec {
  let wasmModule: WebAssembly.Module | undefined;
  /** The live instance; `undefined` after a trap until the next call instantiates a fresh one. */
  let wasm: Exports | undefined;

  function instantiate(): Exports {
    wasmModule ??= loadModule();
    return new WebAssembly.Instance(wasmModule).exports as unknown as Exports;
  }

  /**
   * Places `input` in wasm memory for the duration of `run`. A string is encoded as UTF-8
   * straight into an exactly sized module buffer, so no intermediate copy of a large document
   * is held on the JS heap. (Exact sizing also matters for correctness: Bun 1.3 `encodeInto`
   * writes U+FFFD for a 4-byte character when only 3 bytes of room remain instead of stopping
   * before it, so the buffer is never filled in several passes.)
   *
   * A trap (the module ran out of memory) leaves the instance's heap in an unknown state, so
   * the instance is dropped before the error propagates and the next call instantiates a
   * fresh one.
   */
  function withInput<T>(input: string | Uint8Array, run: (exports: Exports, ptr: number, len: number) => T): T {
    const exports = (wasm ??= instantiate());
    const len = typeof input === 'string' ? utf8Length(input) : input.length;
    let trapped = false;
    let ptr: number | undefined;
    try {
      ptr = exports.tokzip_alloc(len);
      const view = new Uint8Array(exports.memory.buffer, ptr, len);
      if (typeof input === 'string') textEncoder.encodeInto(input, view);
      else view.set(input);
      return run(exports, ptr, len);
    } catch (error) {
      if (error instanceof WebAssembly.RuntimeError) {
        trapped = true;
        wasm = undefined;
      }
      throw error;
    } finally {
      if (!trapped && ptr !== undefined) exports.tokzip_free(ptr, len);
    }
  }

  return {
    compress(text) {
      // A lone surrogate cannot round-trip through UTF-8 (TextEncoder would substitute
      // U+FFFD), so it is refused rather than silently altered.
      if (!text.isWellFormed()) {
        throw new RangeError('tokzip: string contains a lone surrogate and cannot be stored losslessly');
      }
      return withInput(text, (exports, ptr, len) => {
        exports.tokzip_compress(ptr, len);
        return takeOutput(exports);
      });
    },
    decompress(frame, { maxLength = Number.POSITIVE_INFINITY } = {}) {
      // A negative limit would wrap to a huge unsigned length at the wasm boundary (NaN fails too).
      if (!(maxLength >= 0)) throw new RangeError(`tokzip: maxLength must be a non-negative number, got ${maxLength}`);
      // wasm32 lengths are 32-bit; anything above cannot be a frame's length anyway.
      const maxLen = Math.min(maxLength, 0xFF_FF_FF_FF);
      return withInput(frame, (exports, ptr, len) => {
        const code = exports.tokzip_decompress(ptr, len, maxLen);
        if (code !== 0) throw new TokzipDecodeError(code);
        const out = takeOutput(exports);
        try {
          return textDecoder.decode(out);
        } catch {
          throw new TokzipDecodeError(5);
        }
      });
    },
  };
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
function takeOutput(exports: Exports): Uint8Array {
  return new Uint8Array(exports.memory.buffer, exports.tokzip_out_ptr(), exports.tokzip_out_len()).slice();
}
