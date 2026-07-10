/** Thrown when a tokzip payload is structurally invalid (corrupt, truncated, or malformed). */
export class TokzipDecodeError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'TokzipDecodeError';
  }
}
