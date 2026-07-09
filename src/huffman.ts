import { TokzipDecodeError } from './errors.ts';

/** Normative maximum canonical Huffman code length for all `small`-mode streams. */
export const MAX_CODE_LENGTH = 12;

const DECODE_TABLE_SIZE = 1 << MAX_CODE_LENGTH;

export interface HuffmanEncoder {
  /** Canonical code for each symbol, MSB-aligned to its length (0 for unused symbols). */
  codes: Uint32Array;
  lengths: Uint8Array;
}

/**
 * Validates that `lengths` forms a complete canonical code (Kraft sum exactly 1) over the used
 * alphabet with every length ≤ {@link MAX_CODE_LENGTH}. Shipped language tables must satisfy this.
 */
export function isCompleteCode(lengths: Uint8Array): boolean {
  let kraft = 0;
  let used = 0;
  for (const length of lengths) {
    if (length === 0) continue;
    if (length > MAX_CODE_LENGTH) return false;
    kraft += DECODE_TABLE_SIZE >> length;
    used++;
  }
  return used >= 2 && kraft === DECODE_TABLE_SIZE;
}

/** Assigns canonical codes (increasing length, then symbol order). */
export function buildEncoder(lengths: Uint8Array): HuffmanEncoder {
  const countPerLength = new Uint32Array(MAX_CODE_LENGTH + 1);
  // Unused symbols (length 0) must not shift next_code, or sparse complete tables mis-assign.
  for (const length of lengths) if (length > 0) countPerLength[length]!++;
  const nextCode = new Uint32Array(MAX_CODE_LENGTH + 1);
  let code = 0;
  for (let length = 1; length <= MAX_CODE_LENGTH; length++) {
    code = (code + countPerLength[length - 1]!) << 1;
    nextCode[length] = code;
  }
  const codes = new Uint32Array(lengths.length);
  for (let symbol = 0; symbol < lengths.length; symbol++) {
    const length = lengths[symbol]!;
    if (length > 0) codes[symbol] = nextCode[length]!++;
  }
  return { codes, lengths };
}

/**
 * Builds a single-lookup decode table: index by the next 12 bits of the stream, entry packs
 * `symbol << 4 | codeLength`. Throws on incomplete codes — decoders must reject invalid tables.
 */
export function buildDecoder(lengths: Uint8Array): Uint16Array {
  if (!isCompleteCode(lengths)) throw new TokzipDecodeError('invalid Huffman table (incomplete code)');
  const { codes } = buildEncoder(lengths);
  const table = new Uint16Array(DECODE_TABLE_SIZE);
  for (let symbol = 0; symbol < lengths.length; symbol++) {
    const length = lengths[symbol]!;
    if (length === 0) continue;
    const first = codes[symbol]! << (MAX_CODE_LENGTH - length);
    const count = 1 << (MAX_CODE_LENGTH - length);
    table.fill((symbol << 4) | length, first, first + count);
  }
  return table;
}

/**
 * Computes length-limited Huffman code lengths from symbol frequencies via package-merge
 * (used by the offline trainer; exactness matters because encode prices derive from these).
 * Frequencies of 0 yield length 0 (unused symbol); at least two symbols must be used.
 */
export function buildLengths(freqs: ArrayLike<number>, maxLength = MAX_CODE_LENGTH): Uint8Array {
  const used: number[] = [];
  for (let symbol = 0; symbol < freqs.length; symbol++) {
    if (freqs[symbol]! > 0) used.push(symbol);
  }
  const lengths = new Uint8Array(freqs.length);
  if (used.length === 1) {
    lengths[used[0]!] = 1;
    return lengths;
  }
  if (used.length === 0) return lengths;
  if (2 ** maxLength < used.length) throw new RangeError('alphabet too large for the length limit');

  // Package-merge: at each level, pair up the cheapest items; a leaf's inclusion count across
  // the first (maxLength) levels equals its code length.
  interface Item {
    weight: number;
    leaves: number[];
  }
  let packages: Item[] = [];
  const leaves: Item[] = used
    .map((symbol) => ({ weight: freqs[symbol]!, leaves: [symbol] }))
    .toSorted((a, b) => a.weight - b.weight);
  for (let level = 0; level < maxLength; level++) {
    const merged: Item[] = [];
    let li = 0;
    let pi = 0;
    while (li < leaves.length || pi < packages.length) {
      const takeLeaf = pi >= packages.length || (li < leaves.length && leaves[li]!.weight <= packages[pi]!.weight);
      merged.push(takeLeaf ? leaves[li++]! : packages[pi++]!);
    }
    packages = [];
    for (let i = 0; i + 1 < merged.length; i += 2) {
      packages.push({
        weight: merged[i]!.weight + merged[i + 1]!.weight,
        leaves: [...merged[i]!.leaves, ...merged[i + 1]!.leaves],
      });
    }
  }
  // The optimal length-limited code takes the cheapest (n - 1) packages from the final level.
  for (const item of packages.slice(0, used.length - 1)) {
    for (const symbol of item.leaves) lengths[symbol]!++;
  }
  return lengths;
}
