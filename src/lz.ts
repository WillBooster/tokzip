import type { DictIndex, RegisteredLanguage } from './dictionary.ts';
import { INITIAL_REPS, MATCH_LEN_CAP, MIN_LEN_EXPLICIT, MIN_LEN_REP } from './format.ts';

/** Literal run: a raw byte range of the input. */
export interface LiteralToken {
  type: 'lit';
  start: number;
  end: number;
}
/** History match: `dist` bytes back in the produced output; `rep >= 0` marks a rep-cache hit. */
export interface HistoryToken {
  type: 'history';
  len: number;
  dist: number;
  rep: number;
}
/** Dictionary match: absolute `start` within the assembled preset dictionary. */
export interface DictToken {
  type: 'dict';
  len: number;
  start: number;
}
export type Token = LiteralToken | HistoryToken | DictToken;

/** Mode-specific pricing and limits driving the shared parser. */
export interface ParsePricing {
  /** litCostPrefix[i] = exact cost of bytes[0..i) encoded as literals, in output units. */
  litCostPrefix: Float64Array;
  repCost(repIndex: number, len: number): number;
  historyCost(dist: number, len: number): number;
  dictCost(start: number, len: number): number;
  /** Enables bounded price-aware lazy matching (`small` mode). */
  lazy: boolean;
  window: number;
  /** Exclusive bound on representable dictionary start offsets. */
  maxDictStart: number;
}

const BUCKET_SLOTS = 4;
const HASH_MULTIPLIER = 0x9E_37_79_B1;

function hash4(bytes: Uint8Array, i: number, shift: number): number {
  const x = bytes[i]! | (bytes[i + 1]! << 8) | (bytes[i + 2]! << 16) | (bytes[i + 3]! << 24);
  return Math.imul(x, HASH_MULTIPLIER) >>> shift;
}

function hashBitsFor(length: number): number {
  let bits = 8;
  while (bits < 16 && 1 << bits < length) bits++;
  return bits;
}

/**
 * Builds (or returns the cached) multi-probe hash index over a language's assembled dictionary.
 * Built lazily on first compress() per language, cached per process; idempotent.
 */
export function dictIndexFor(language: RegisteredLanguage): DictIndex | undefined {
  const dict = language.dictionary;
  if (dict.length < 4) return undefined;
  if (language.dictIndex) return language.dictIndex;
  const bits = hashBitsFor(dict.length);
  const shift = 32 - bits;
  const table = new Int32Array((1 << bits) * BUCKET_SLOTS).fill(-1);
  const counts = new Uint8Array(1 << bits);
  for (let i = 0; i + 4 <= dict.length; i++) {
    const bucket = hash4(dict, i, shift);
    table[bucket * BUCKET_SLOTS + (counts[bucket]! & (BUCKET_SLOTS - 1))] = i;
    counts[bucket] = counts[bucket]! + 1;
  }
  language.dictIndex = { hashShift: shift, table };
  return language.dictIndex;
}

function matchLength(a: Uint8Array, ai: number, b: Uint8Array, bi: number, cap: number): number {
  let len = 0;
  while (len < cap && a[ai + len] === b[bi + len]) len++;
  return len;
}

interface Candidate {
  savings: number;
  cost: number;
  len: number;
  token: Token;
}

/**
 * Shared LZ pass: greedy (or price-aware lazy) parse over the input against the sliding history
 * window, the rep-offset cache, and the preset dictionary. Emits the token list both wire
 * formats serialize; rep-cache updates are replayed identically by decoders.
 */
export function parse(
  bytes: Uint8Array,
  dictionary: Uint8Array,
  dictIndex: DictIndex | undefined,
  pricing: ParsePricing
): Token[] {
  const n = bytes.length;
  const tokens: Token[] = [];
  if (n === 0) return tokens;

  const bits = hashBitsFor(n);
  const shift = 32 - bits;
  const table = new Int32Array((1 << bits) * BUCKET_SLOTS).fill(-1);
  const counts = new Uint8Array(1 << bits);
  const insertHash = (i: number): void => {
    if (i + 4 > n) return;
    const bucket = hash4(bytes, i, shift);
    table[bucket * BUCKET_SLOTS + (counts[bucket]! & (BUCKET_SLOTS - 1))] = i;
    counts[bucket] = counts[bucket]! + 1;
  };

  const reps = [...INITIAL_REPS];
  const { litCostPrefix } = pricing;
  const litCost = (start: number, end: number): number => litCostPrefix[end]! - litCostPrefix[start]!;

  const findBest = (pos: number): Candidate | undefined => {
    const cap = Math.min(n - pos, MATCH_LEN_CAP);
    if (cap < MIN_LEN_REP) return undefined;
    let best: Candidate | undefined;
    const consider = (cost: number, len: number, token: Token): void => {
      const savings = litCost(pos, pos + len) - cost;
      if (savings > 0 && (!best || savings > best.savings)) best = { savings, cost, len, token };
    };

    for (let r = 0; r < 4; r++) {
      const dist = reps[r]!;
      if (dist > pos) continue;
      const len = matchLength(bytes, pos, bytes, pos - dist, cap);
      if (len >= MIN_LEN_REP) consider(pricing.repCost(r, len), len, { type: 'history', len, dist, rep: r });
    }
    if (cap >= 4 && pos + 4 <= n) {
      const bucket = hash4(bytes, pos, shift) * BUCKET_SLOTS;
      for (let s = 0; s < BUCKET_SLOTS; s++) {
        const cand = table[bucket + s]!;
        if (cand < 0 || cand >= pos) continue;
        const dist = pos - cand;
        if (dist > pricing.window) continue;
        const len = matchLength(bytes, pos, bytes, cand, cap);
        if (len < MIN_LEN_EXPLICIT) continue;
        const repIndex = reps.indexOf(dist);
        if (repIndex !== -1)
          consider(pricing.repCost(repIndex, len), len, { type: 'history', len, dist, rep: repIndex });
        else consider(pricing.historyCost(dist, len), len, { type: 'history', len, dist, rep: -1 });
      }
      if (dictIndex) {
        const dictBucket = hash4(bytes, pos, dictIndex.hashShift) * BUCKET_SLOTS;
        for (let s = 0; s < BUCKET_SLOTS; s++) {
          const cand = dictIndex.table[dictBucket + s]!;
          if (cand < 0 || cand >= pricing.maxDictStart) continue;
          // Dictionary matches lie entirely within dictionary space.
          const len = matchLength(bytes, pos, dictionary, cand, Math.min(cap, dictionary.length - cand));
          if (len < MIN_LEN_EXPLICIT) continue;
          consider(pricing.dictCost(cand, len), len, { type: 'dict', len, start: cand });
        }
      }
    }
    return best;
  };

  const applyReps = (token: Token): void => {
    if (token.type !== 'history') return;
    if (token.rep >= 0) {
      const [dist] = reps.splice(token.rep, 1);
      reps.unshift(dist!);
    } else {
      reps.pop();
      reps.unshift(token.dist);
    }
  };

  let pos = 0;
  let litStart = 0;
  while (pos < n) {
    const best = findBest(pos);
    let accepted = best;
    let posHashed = false;
    if (best && pricing.lazy && pos + 1 < n) {
      // Bounded price-aware lazy step: prefer deferring when a literal plus the next match
      // covers bytes at a strictly better price density.
      insertHash(pos);
      posHashed = true;
      const next = findBest(pos + 1);
      if (next && (litCost(pos, pos + 1) + next.cost) / (1 + next.len) < best.cost / best.len) {
        accepted = undefined;
      }
    }
    if (accepted) {
      if (litStart < pos) tokens.push({ type: 'lit', start: litStart, end: pos });
      tokens.push(accepted.token);
      applyReps(accepted.token);
      const end = pos + accepted.len;
      const stride = accepted.len > 1024 ? 8 : 1;
      for (let i = posHashed ? pos + stride : pos; i < end; i += stride) insertHash(i);
      pos = end;
      litStart = pos;
    } else {
      if (!posHashed) insertHash(pos);
      pos++;
    }
  }
  if (litStart < n) tokens.push({ type: 'lit', start: litStart, end: n });
  return tokens;
}
