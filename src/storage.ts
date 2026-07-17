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
  const output = options?.output ?? 'text';
  try {
    const frame = compress(input, options);
    if (roundTrips(frame, input)) return frame;
  } catch {
    // Fall through to the stored fallback: a compression failure must never lose data.
  }
  const fallback = compressStored(input, output === 'binary' ? 'binary' : 'text');
  if (!roundTrips(fallback, input)) throw new Error('tokzip: stored fallback failed round-trip verification');
  return fallback;
}

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
