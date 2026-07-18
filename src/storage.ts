import { compress, compressStored, decompress, type CompressOptions } from './container.ts';

/**
 * Storage-grade compression: compress, verify the frame round-trips to the exact input, and
 * fall back to a plain stored frame when anything goes wrong. Persisted user data must never
 * depend on the compressor being bug-free — a frame this function returns has already been
 * decoded back to the original content once, so the stored payload is provably recoverable
 * by the same library version.
 */
export function compressForStorage(input: string | Uint8Array, options?: CompressOptions & { output?: 'text' }): string;
export function compressForStorage(
  input: string | Uint8Array,
  options: CompressOptions & { output: 'binary' }
): Uint8Array;
/** Fallback for options whose `output` is not statically known (e.g. a `CompressOptions` variable). */
export function compressForStorage(input: string | Uint8Array, options?: CompressOptions): string | Uint8Array;
export function compressForStorage(input: string | Uint8Array, options?: CompressOptions): string | Uint8Array {
  // Lone surrogates cannot round-trip through UTF-8 (TextEncoder replaces them with
  // U+FFFD), so the exact-input guarantee is unsatisfiable — fail fast with direction
  // instead of dying later in the fallback's own verification.
  if (typeof input === 'string' && !isWellFormedString(input)) {
    throw new RangeError(
      'tokzip: input contains lone surrogates, which cannot round-trip through UTF-8; ' +
        'pass a well-formed string (String.prototype.toWellFormed is lossy but safe) or byte-exact Uint8Array data'
    );
  }
  const output = options?.output ?? 'text';
  try {
    const frame = compress(input, options);
    if (roundTrips(frame, input)) return frame;
  } catch (error) {
    // RangeError is compress() rejecting the caller's options (bad mode/output/language) —
    // rethrow it: silently "recovering" from a typo by ignoring the requested options
    // would e.g. hand a text frame to a binary storage path. Everything else is an
    // internal compression failure, which falls through to the stored fallback so a
    // compressor bug never loses data.
    if (error instanceof RangeError) throw error;
  }
  const fallback = compressStored(input, output === 'binary' ? 'binary' : 'text');
  if (!roundTrips(fallback, input)) throw new Error('tokzip: stored fallback failed round-trip verification');
  return fallback;
}

/**
 * Manual surrogate-pair scan instead of String.prototype.isWellFormed: the builtin only
 * exists from V8 11.3 (Node 20), while this library supports Node 18.
 */
// oxlint-disable unicorn/prefer-code-point -- surrogate detection needs raw UTF-16 units; codePointAt would combine valid pairs
function isWellFormedString(text: string): boolean {
  for (let i = 0; i < text.length; i++) {
    const unit = text.charCodeAt(i);
    if (unit < 0xD8_00 || unit > 0xDF_FF) continue;
    // A high surrogate must be followed by a low surrogate; a bare low surrogate is lone.
    if (unit > 0xDB_FF) return false;
    const next = i + 1 < text.length ? text.charCodeAt(i + 1) : 0;
    if (next < 0xDC_00 || next > 0xDF_FF) return false;
    i++;
  }
  return true;
}
// oxlint-enable unicorn/prefer-code-point

function roundTrips(frame: string | Uint8Array, input: string | Uint8Array): boolean {
  try {
    const decoded = decompress(frame, { maxOutputSize: Number.POSITIVE_INFINITY });
    if (typeof input === 'string') return decoded === input;
    if (typeof decoded === 'string') return false;
    if (decoded.length !== input.length) return false;
    for (let i = 0; i < input.length; i++) if (decoded[i] !== input[i]) return false;
    return true;
  } catch {
    return false;
  }
}
