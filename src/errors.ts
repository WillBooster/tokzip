/** Thrown when a tokzip payload is structurally invalid (corrupt, truncated, or malformed). */
export class TokzipDecodeError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'TokzipDecodeError';
  }
}

/**
 * Allocates a decode output buffer, translating engine allocation failures (e.g. a declared
 * size beyond available memory under `maxOutputSize: Infinity`) into typed decode errors so
 * callers never see a bare RangeError from a hostile frame.
 */
export function allocateDecodeBuffer(size: number): Uint8Array {
  try {
    return new Uint8Array(size);
  } catch {
    throw new TokzipDecodeError('declared size exceeds allocatable memory');
  }
}
