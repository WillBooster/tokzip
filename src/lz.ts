import type { DictIndex, RegisteredLanguage } from './dictionary.ts';
import { INITIAL_REPS, MATCH_LEN_CAP, MIN_LEN_EXPLICIT, MIN_LEN_REP } from './format.ts';
import { LENGTH_SLOT_COUNT, maxSlotValue, slotOf } from './slots.ts';

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

/**
 * Exact `small`-mode bit prices in slot-table form. Enables the optimal (shortest-path) parse:
 * every price is a plain array lookup, so the DP inner loop stays allocation- and call-free.
 */
export interface SlotPricing {
  /** Exact literal bit price per byte value (raw-mode fallback substituted for codeless bytes). */
  litBits: Float64Array;
  /** History token bits per length slot (token symbol + length extra bits). */
  histSlotBits: Float64Array;
  /** Dictionary token bits per length slot. */
  dictSlotBits: Float64Array;
  /** Rep token bits, indexed `repIndex * LENGTH_SLOT_COUNT + lengthSlot`. */
  repSlotBits: Float64Array;
  /** Offset bits per offset slot (symbol + offset extra bits). */
  offsetSlotBits: Float64Array;
}

/** Mode-specific pricing and limits driving the shared parser. */
export interface ParsePricing {
  /** litCostPrefix[i] = exact cost of bytes[0..i) encoded as literals, in output units. */
  litCostPrefix: Float64Array;
  repCost(repIndex: number, len: number): number;
  historyCost(dist: number, len: number): number;
  dictCost(start: number, len: number): number;
  /** Enables bounded price-aware lazy matching in the greedy parser. */
  lazy: boolean;
  window: number;
  /** Exclusive bound on representable dictionary start offsets. */
  maxDictStart: number;
  /** When set, inputs up to {@link OPTIMAL_MAX_INPUT} take the optimal parse instead. */
  optimal?: SlotPricing;
}

const HASH_MULTIPLIER = 0x9E_37_79_B1;

/** Inputs beyond this take the greedy-lazy parser (the DP costs ~40 bytes of scratch per byte). */
const OPTIMAL_MAX_INPUT = 1 << 19;

/** Match-finder search depths (chain links walked per position). */
const GREEDY_DEPTH = 16;
const GREEDY_DICT_DEPTH = 12;
/** Shallow 4-byte-hash walks (short matches) + deep selective 6-byte-hash walks (long matches). */
const OPTIMAL_DEPTH_SHORT = 12;
const OPTIMAL_DEPTH = 96;
const OPTIMAL_DICT_DEPTH_SHORT = 8;
const OPTIMAL_DICT_DEPTH = 64;

/** The DP evaluates every match length up to this bound, then only slot-boundary lengths. */
const DENSE_LEN_BOUND = 48;

/**
 * Matches at least this long are taken whole and the DP jumps past them (zstd-style immediate
 * encoding). Splitting such a match is essentially never profitable, and the jump keeps
 * degenerate inputs (long byte runs) linear instead of quadratic.
 */
const CUT_LEN = 128;

/** Chain walks stop once a match this long is found (zstd-style sufficient length). */
const SUFFICIENT_LEN = 64;

/** Internal DP arrival kinds (3 + r encodes rep r). */
const DP_LIT = 0;
const DP_HISTORY = 1;
const DP_DICT = 2;
const DP_REP0 = 3;

/** Largest length value covered by each length slot (index = slot). */
const SLOT_MAX_VALUE = new Int32Array(LENGTH_SLOT_COUNT);
for (let s = 0; s < LENGTH_SLOT_COUNT; s++) SLOT_MAX_VALUE[s] = maxSlotValue(s + 1);

function hash4(bytes: Uint8Array, i: number, shift: number): number {
  const x = bytes[i]! | (bytes[i + 1]! << 8) | (bytes[i + 2]! << 16) | (bytes[i + 3]! << 24);
  return Math.imul(x, HASH_MULTIPLIER) >>> shift;
}

function hash6(bytes: Uint8Array, i: number, shift: number): number {
  const lo = bytes[i]! | (bytes[i + 1]! << 8) | (bytes[i + 2]! << 16) | (bytes[i + 3]! << 24);
  const mixed = Math.imul(lo, HASH_MULTIPLIER) ^ Math.imul(bytes[i + 4]! | (bytes[i + 5]! << 8), 0x85_EB_CA_6B);
  return mixed >>> shift;
}

function hashBitsFor(length: number): number {
  let bits = 8;
  while (bits < 17 && 1 << bits < length) bits++;
  return bits;
}

/**
 * Builds (or returns the cached) hash-chain index over a language's assembled dictionary.
 * Built lazily on first compress() per language, cached per process; idempotent.
 */
export function dictIndexFor(language: RegisteredLanguage): DictIndex | undefined {
  const dict = language.dictionary;
  if (dict.length < 4) return undefined;
  if (language.dictIndex) return language.dictIndex;
  const bits = hashBitsFor(dict.length);
  const shift = 32 - bits;
  const head = new Int32Array(1 << bits).fill(-1);
  const prev = new Int32Array(dict.length);
  for (let i = 0; i + 4 <= dict.length; i++) {
    const bucket = hash4(dict, i, shift);
    prev[i] = head[bucket]!;
    head[bucket] = i;
  }
  const head6 = new Int32Array(1 << bits).fill(-1);
  const prev6 = new Int32Array(dict.length);
  for (let i = 0; i + 6 <= dict.length; i++) {
    const bucket = hash6(dict, i, shift);
    prev6[i] = head6[bucket]!;
    head6[bucket] = i;
  }
  language.dictIndex = { hashShift: shift, head, prev, head6, prev6 };
  return language.dictIndex;
}

function matchLength(a: Uint8Array, ai: number, b: Uint8Array, bi: number, cap: number): number {
  let len = 0;
  while (len < cap && a[ai + len] === b[bi + len]) len++;
  return len;
}

/** Indexes position `i` in both the 4-byte- and 6-byte-hash chains (optimal parse). */
function insertChains(
  bytes: Uint8Array,
  i: number,
  n: number,
  shift: number,
  head: Int32Array,
  prev: Int32Array,
  head6: Int32Array,
  prev6: Int32Array
): void {
  if (i + 4 > n) return;
  const bucket = hash4(bytes, i, shift);
  prev[i] = head[bucket]!;
  head[bucket] = i;
  if (i + 6 > n) return;
  const bucket6 = hash6(bytes, i, shift);
  prev6[i] = head6[bucket6]!;
  head6[bucket6] = i;
}

/**
 * Shared LZ pass: parses the input against the sliding history window, the rep-offset cache,
 * and the preset dictionary into the token list both wire formats serialize. `small` pricing
 * (with {@link ParsePricing.optimal}) takes an exact-price shortest-path parse; `fast` pricing
 * takes the greedy parser. Rep-cache updates are replayed identically by decoders.
 */
export function parse(
  bytes: Uint8Array,
  dictionary: Uint8Array,
  dictIndex: DictIndex | undefined,
  pricing: ParsePricing
): Token[] {
  if (bytes.length === 0) return [];
  if (pricing.optimal && bytes.length <= OPTIMAL_MAX_INPUT) {
    return parseOptimal(bytes, dictionary, dictIndex, pricing.window, pricing.maxDictStart, pricing.optimal);
  }
  return parseGreedy(bytes, dictionary, dictIndex, pricing);
}

// Scratch buffers reused across calls (compress is synchronous; JS is single-threaded).
const headPool = new Map<number, Int32Array>();
const headPool6 = new Map<number, Int32Array>();
let chainPrev = new Int32Array(0);
let chainPrev6 = new Int32Array(0);
let dpCost = new Float64Array(0);
let dpSrc = new Int32Array(0);
let dpKind = new Uint8Array(0);
let dpDist = new Int32Array(0);
let dpReps = new Int32Array(0);
// Pareto match candidates collected per position (length strictly ascending).
const candDist = new Int32Array(OPTIMAL_DEPTH_SHORT + OPTIMAL_DEPTH);
const candLen = new Int32Array(OPTIMAL_DEPTH_SHORT + OPTIMAL_DEPTH);

function headFor(pool: Map<number, Int32Array>, bits: number): Int32Array {
  let head = pool.get(bits);
  if (!head) {
    head = new Int32Array(1 << bits);
    pool.set(bits, head);
  }
  head.fill(-1);
  return head;
}

function prevFor(length: number): Int32Array {
  if (chainPrev.length < length) chainPrev = new Int32Array(Math.max(length, chainPrev.length * 2, 4096));
  return chainPrev;
}

function prevFor6(length: number): Int32Array {
  if (chainPrev6.length < length) chainPrev6 = new Int32Array(Math.max(length, chainPrev6.length * 2, 4096));
  return chainPrev6;
}

/**
 * Optimal parse: forward shortest-path DP over exact static-table bit prices. The rep cache is
 * path-dependent, so each position stores the rep state of its best arrival (the standard
 * zstd/LZMA-style approximation). Literal-run token overhead is ignored during the DP (like the
 * greedy parser's pricing) and recovered exactly by the frame-level plan afterwards.
 */
function parseOptimal(
  bytes: Uint8Array,
  dictionary: Uint8Array,
  dictIndex: DictIndex | undefined,
  window: number,
  maxDictStart: number,
  prices: SlotPricing
): Token[] {
  const n = bytes.length;
  const { litBits, histSlotBits, dictSlotBits, repSlotBits, offsetSlotBits } = prices;

  const bits = hashBitsFor(n);
  const shift = 32 - bits;
  const head = headFor(headPool, bits);
  const prev = prevFor(n);
  const head6 = headFor(headPool6, bits);
  const prev6 = prevFor6(n);

  if (dpCost.length < n + 1) {
    const size = Math.max(n + 1, dpCost.length * 2, 4096);
    dpCost = new Float64Array(size);
    dpSrc = new Int32Array(size);
    dpKind = new Uint8Array(size);
    dpDist = new Int32Array(size);
    dpReps = new Int32Array(size * 4);
  }
  const cost = dpCost;
  const src = dpSrc;
  const kind = dpKind;
  const dist = dpDist;
  const reps = dpReps;
  cost.fill(Number.POSITIVE_INFINITY, 0, n + 1);
  cost[0] = 0;
  reps[0] = INITIAL_REPS[0]!;
  reps[1] = INITIAL_REPS[1]!;
  reps[2] = INITIAL_REPS[2]!;
  reps[3] = INITIAL_REPS[3]!;

  for (let i = 0; i < n; i++) {
    const base = cost[i]!;
    const ri = i * 4;
    const rep0 = reps[ri]!;
    const rep1 = reps[ri + 1]!;
    const rep2 = reps[ri + 2]!;
    const rep3 = reps[ri + 3]!;

    // Literal step (always available; keeps every position reachable).
    {
      const c = base + litBits[bytes[i]!]!;
      if (c < cost[i + 1]!) {
        cost[i + 1] = c;
        src[i + 1] = i;
        kind[i + 1] = DP_LIT;
        const rj = ri + 4;
        reps[rj] = rep0;
        reps[rj + 1] = rep1;
        reps[rj + 2] = rep2;
        reps[rj + 3] = rep3;
      }
    }

    const cap = n - i < MATCH_LEN_CAP ? n - i : MATCH_LEN_CAP;
    if (cap >= MIN_LEN_REP) {
      // Rep matches (min length 2): a single joint pass over lengths. For each length only the
      // cheapest rep covering it can win, so the per-length work is one cost compare; the
      // arg-min rep is recomputed only when the slot advances or a rep's match length runs out.
      const m0 = rep0 <= i ? matchLength(bytes, i, bytes, i - rep0, cap) : 0;
      const m1 = rep1 <= i && rep1 !== rep0 ? matchLength(bytes, i, bytes, i - rep1, cap) : 0;
      const m2 = rep2 <= i && rep2 !== rep1 && rep2 !== rep0 ? matchLength(bytes, i, bytes, i - rep2, cap) : 0;
      const m3 =
        rep3 <= i && rep3 !== rep2 && rep3 !== rep1 && rep3 !== rep0 ? matchLength(bytes, i, bytes, i - rep3, cap) : 0;
      let maxM = m0 > m1 ? m0 : m1;
      if (m2 > maxM) maxM = m2;
      if (m3 > maxM) maxM = m3;
      if (maxM >= CUT_LEN) {
        // Immediate encoding: take the longest rep whole and jump past it.
        const r = maxM === m0 ? 0 : maxM === m1 ? 1 : maxM === m2 ? 2 : 3;
        const d = r === 0 ? rep0 : r === 1 ? rep1 : r === 2 ? rep2 : rep3;
        const c = base + repSlotBits[r * LENGTH_SLOT_COUNT + slotOf(maxM - MIN_LEN_REP)]!;
        const j = i + maxM;
        if (c < cost[j]!) updateRep(cost, src, kind, dist, reps, i, j, c, r, d, rep0, rep1, rep2, rep3);
        insertChains(bytes, i, n, shift, head, prev, head6, prev6);
        i = j - 1;
        continue;
      }
      if (maxM >= MIN_LEN_REP) {
        const denseEnd = maxM < DENSE_LEN_BOUND ? maxM : DENSE_LEN_BOUND;
        let s = 0;
        let bestR = -1;
        let bestBits = 0;
        let stale = true;
        for (let len = MIN_LEN_REP; len <= denseEnd; len++) {
          if (len - MIN_LEN_REP > SLOT_MAX_VALUE[s]!) {
            s++;
            stale = true;
          }
          if (len === m0 + 1 || len === m1 + 1 || len === m2 + 1 || len === m3 + 1) stale = true;
          if (stale) {
            stale = false;
            bestR = -1;
            bestBits = Number.POSITIVE_INFINITY;
            if (m0 >= len && repSlotBits[s]! < bestBits) {
              bestR = 0;
              bestBits = repSlotBits[s]!;
            }
            if (m1 >= len && repSlotBits[LENGTH_SLOT_COUNT + s]! < bestBits) {
              bestR = 1;
              bestBits = repSlotBits[LENGTH_SLOT_COUNT + s]!;
            }
            if (m2 >= len && repSlotBits[2 * LENGTH_SLOT_COUNT + s]! < bestBits) {
              bestR = 2;
              bestBits = repSlotBits[2 * LENGTH_SLOT_COUNT + s]!;
            }
            if (m3 >= len && repSlotBits[3 * LENGTH_SLOT_COUNT + s]! < bestBits) {
              bestR = 3;
              bestBits = repSlotBits[3 * LENGTH_SLOT_COUNT + s]!;
            }
          }
          const c = base + bestBits;
          const j = i + len;
          if (c < cost[j]!) {
            const d = bestR === 0 ? rep0 : bestR === 1 ? rep1 : bestR === 2 ? rep2 : rep3;
            updateRep(cost, src, kind, dist, reps, i, j, c, bestR, d, rep0, rep1, rep2, rep3);
          }
        }
        if (maxM > DENSE_LEN_BOUND) {
          for (let s2 = slotOf(DENSE_LEN_BOUND - MIN_LEN_REP); s2 < LENGTH_SLOT_COUNT; s2++) {
            const len = Math.min(maxM, SLOT_MAX_VALUE[s2]! + MIN_LEN_REP);
            if (len <= DENSE_LEN_BOUND) continue;
            const slot = slotOf(len - MIN_LEN_REP);
            let bestR2 = -1;
            let bestBits2 = Number.POSITIVE_INFINITY;
            if (m0 >= len && repSlotBits[slot]! < bestBits2) {
              bestR2 = 0;
              bestBits2 = repSlotBits[slot]!;
            }
            if (m1 >= len && repSlotBits[LENGTH_SLOT_COUNT + slot]! < bestBits2) {
              bestR2 = 1;
              bestBits2 = repSlotBits[LENGTH_SLOT_COUNT + slot]!;
            }
            if (m2 >= len && repSlotBits[2 * LENGTH_SLOT_COUNT + slot]! < bestBits2) {
              bestR2 = 2;
              bestBits2 = repSlotBits[2 * LENGTH_SLOT_COUNT + slot]!;
            }
            if (m3 >= len && repSlotBits[3 * LENGTH_SLOT_COUNT + slot]! < bestBits2) {
              bestR2 = 3;
              bestBits2 = repSlotBits[3 * LENGTH_SLOT_COUNT + slot]!;
            }
            const c = base + bestBits2;
            const j = i + len;
            if (c < cost[j]!) {
              const d = bestR2 === 0 ? rep0 : bestR2 === 1 ? rep1 : bestR2 === 2 ? rep2 : rep3;
              updateRep(cost, src, kind, dist, reps, i, j, c, bestR2, d, rep0, rep1, rep2, rep3);
            }
            if (len === maxM) break;
          }
        }
      }

      if (cap >= MIN_LEN_EXPLICIT && i + 4 <= n) {
        // Explicit history matches: walk the chains collecting the Pareto set (nearer candidates
        // first, so a farther candidate is kept only when it extends the match length). The
        // shallow 4-byte-hash walk covers short matches; the selective 6-byte-hash chain is
        // then walked deep for long matches.
        let candCount = 0;
        {
          let bestM = MIN_LEN_EXPLICIT - 1;
          const minPos = i - window;
          let cand = head[hash4(bytes, i, shift)]!;
          let depth = OPTIMAL_DEPTH_SHORT;
          while (cand >= 0 && cand >= minPos && depth-- > 0) {
            if (bytes[cand + bestM] === bytes[i + bestM]) {
              const m = matchLength(bytes, i, bytes, cand, cap);
              if (m > bestM) {
                const d = i - cand;
                // Distances already in the rep cache were priced above at every length.
                if (d !== rep0 && d !== rep1 && d !== rep2 && d !== rep3) {
                  candDist[candCount] = d;
                  candLen[candCount] = m;
                  candCount++;
                }
                bestM = m;
                if (m === cap || m >= SUFFICIENT_LEN) break;
              }
            }
            cand = prev[cand]!;
          }
          if (bestM < cap && i + 6 <= n) {
            let cand6 = head6[hash6(bytes, i, shift)]!;
            let depth6 = OPTIMAL_DEPTH;
            while (cand6 >= 0 && cand6 >= minPos && depth6-- > 0) {
              if (bytes[cand6 + bestM] === bytes[i + bestM]) {
                const m = matchLength(bytes, i, bytes, cand6, cap);
                if (m > bestM) {
                  const d = i - cand6;
                  if (d !== rep0 && d !== rep1 && d !== rep2 && d !== rep3) {
                    candDist[candCount] = d;
                    candLen[candCount] = m;
                    candCount++;
                  }
                  bestM = m;
                  if (m === cap || m >= SUFFICIENT_LEN) break;
                }
              }
              cand6 = prev6[cand6]!;
            }
          }
        }
        if (candCount > 0 && candLen[candCount - 1]! >= CUT_LEN) {
          // Immediate encoding: take the longest match whole and jump past it.
          const d = candDist[candCount - 1]!;
          const m = candLen[candCount - 1]!;
          const c = base + histSlotBits[slotOf(m - MIN_LEN_REP)]! + offsetSlotBits[slotOf(d - 1)]!;
          const j = i + m;
          if (c < cost[j]!) updateHistory(cost, src, kind, dist, reps, i, j, c, d, rep0, rep1, rep2);
          insertChains(bytes, i, n, shift, head, prev, head6, prev6);
          i = j - 1;
          continue;
        }
        let lo = MIN_LEN_EXPLICIT;
        for (let c0 = 0; c0 < candCount; c0++) {
          const d = candDist[c0]!;
          const m = candLen[c0]!;
          const offBits = offsetSlotBits[slotOf(d - 1)]!;
          const denseEnd = m < DENSE_LEN_BOUND ? m : DENSE_LEN_BOUND;
          for (let len = lo; len <= denseEnd; len++) {
            const c = base + histSlotBits[slotOf(len - MIN_LEN_REP)]! + offBits;
            const j = i + len;
            if (c < cost[j]!) updateHistory(cost, src, kind, dist, reps, i, j, c, d, rep0, rep1, rep2);
          }
          if (m > DENSE_LEN_BOUND) {
            for (let s = slotOf(Math.max(lo, DENSE_LEN_BOUND) - MIN_LEN_REP); s < LENGTH_SLOT_COUNT; s++) {
              const len = Math.min(m, SLOT_MAX_VALUE[s]! + MIN_LEN_REP);
              if (len <= DENSE_LEN_BOUND || len < lo) continue;
              const c = base + histSlotBits[slotOf(len - MIN_LEN_REP)]! + offBits;
              const j = i + len;
              if (c < cost[j]!) updateHistory(cost, src, kind, dist, reps, i, j, c, d, rep0, rep1, rep2);
              if (len === m) break;
            }
          }
          lo = m + 1;
        }

        // Dictionary matches (no rep-cache interaction). Two-tier search: a shallow 4-byte-hash
        // walk covers short matches, then the selective 6-byte-hash chain is walked deep for
        // long matches without paying for the dictionary's dense 4-gram collisions.
        if (dictIndex) {
          let candCountD = 0;
          {
            let cand = dictIndex.head[hash4(bytes, i, dictIndex.hashShift)]!;
            let bestM = MIN_LEN_EXPLICIT - 1;
            let depth = OPTIMAL_DICT_DEPTH_SHORT;
            while (cand >= 0 && depth-- > 0) {
              if (cand < maxDictStart && dictionary[cand + bestM] === bytes[i + bestM]) {
                const dcap = dictionary.length - cand < cap ? dictionary.length - cand : cap;
                const m = matchLength(bytes, i, dictionary, cand, dcap);
                if (m > bestM) {
                  candDist[candCountD] = cand;
                  candLen[candCountD] = m;
                  candCountD++;
                  bestM = m;
                  if (m === cap || m >= SUFFICIENT_LEN) break;
                }
              }
              cand = dictIndex.prev[cand]!;
            }
            if (bestM < cap && i + 6 <= n) {
              let cand6 = dictIndex.head6[hash6(bytes, i, dictIndex.hashShift)]!;
              let depth6 = OPTIMAL_DICT_DEPTH;
              while (cand6 >= 0 && depth6-- > 0) {
                if (cand6 < maxDictStart && dictionary[cand6 + bestM] === bytes[i + bestM]) {
                  const dcap = dictionary.length - cand6 < cap ? dictionary.length - cand6 : cap;
                  const m = matchLength(bytes, i, dictionary, cand6, dcap);
                  if (m > bestM) {
                    candDist[candCountD] = cand6;
                    candLen[candCountD] = m;
                    candCountD++;
                    bestM = m;
                    if (m === cap || m >= SUFFICIENT_LEN) break;
                  }
                }
                cand6 = dictIndex.prev6[cand6]!;
              }
            }
          }
          if (candCountD > 0 && candLen[candCountD - 1]! >= CUT_LEN) {
            const start = candDist[candCountD - 1]!;
            const m = candLen[candCountD - 1]!;
            const c = base + dictSlotBits[slotOf(m - MIN_LEN_REP)]! + offsetSlotBits[slotOf(start)]!;
            const j = i + m;
            if (c < cost[j]!) updateDict(cost, src, kind, dist, reps, i, j, c, start, rep0, rep1, rep2, rep3);
            insertChains(bytes, i, n, shift, head, prev, head6, prev6);
            i = j - 1;
            continue;
          }
          let dlo = MIN_LEN_EXPLICIT;
          for (let c0 = 0; c0 < candCountD; c0++) {
            const start = candDist[c0]!;
            const m = candLen[c0]!;
            const offBits = offsetSlotBits[slotOf(start)]!;
            const denseEnd = m < DENSE_LEN_BOUND ? m : DENSE_LEN_BOUND;
            for (let len = dlo; len <= denseEnd; len++) {
              const c = base + dictSlotBits[slotOf(len - MIN_LEN_REP)]! + offBits;
              const j = i + len;
              if (c < cost[j]!) updateDict(cost, src, kind, dist, reps, i, j, c, start, rep0, rep1, rep2, rep3);
            }
            if (m > DENSE_LEN_BOUND) {
              for (let s = slotOf(Math.max(dlo, DENSE_LEN_BOUND) - MIN_LEN_REP); s < LENGTH_SLOT_COUNT; s++) {
                const len = Math.min(m, SLOT_MAX_VALUE[s]! + MIN_LEN_REP);
                if (len <= DENSE_LEN_BOUND || len < dlo) continue;
                const c = base + dictSlotBits[slotOf(len - MIN_LEN_REP)]! + offBits;
                const j = i + len;
                if (c < cost[j]!) updateDict(cost, src, kind, dist, reps, i, j, c, start, rep0, rep1, rep2, rep3);
                if (len === m) break;
              }
            }
            dlo = m + 1;
          }
        }
      }
    }

    insertChains(bytes, i, n, shift, head, prev, head6, prev6);
  }

  // Backtrack: walk arrivals from the end, merging consecutive literal steps into runs.
  const tokens: Token[] = [];
  let j = n;
  while (j > 0) {
    const k = kind[j]!;
    if (k === DP_LIT) {
      let start = j - 1;
      while (start > 0 && kind[start]! === DP_LIT) start--;
      tokens.push({ type: 'lit', start, end: j });
      j = start;
    } else {
      const i = src[j]!;
      if (k === DP_HISTORY) tokens.push({ type: 'history', len: j - i, dist: dist[j]!, rep: -1 });
      else if (k === DP_DICT) tokens.push({ type: 'dict', len: j - i, start: dist[j]! });
      else tokens.push({ type: 'history', len: j - i, dist: dist[j]!, rep: k - DP_REP0 });
      j = i;
    }
  }
  tokens.reverse();
  return tokens;
}

function updateRep(
  cost: Float64Array,
  src: Int32Array,
  kind: Uint8Array,
  dist: Int32Array,
  reps: Int32Array,
  i: number,
  j: number,
  c: number,
  r: number,
  d: number,
  rep0: number,
  rep1: number,
  rep2: number,
  rep3: number
): void {
  cost[j] = c;
  src[j] = i;
  kind[j] = DP_REP0 + r;
  dist[j] = d;
  const rj = j * 4;
  // Move rep r to the front of the cache.
  reps[rj] = d;
  reps[rj + 1] = r === 0 ? rep1 : rep0;
  reps[rj + 2] = r <= 1 ? rep2 : rep1;
  reps[rj + 3] = r <= 2 ? rep3 : rep2;
}

function updateHistory(
  cost: Float64Array,
  src: Int32Array,
  kind: Uint8Array,
  dist: Int32Array,
  reps: Int32Array,
  i: number,
  j: number,
  c: number,
  d: number,
  rep0: number,
  rep1: number,
  rep2: number
): void {
  cost[j] = c;
  src[j] = i;
  kind[j] = DP_HISTORY;
  dist[j] = d;
  const rj = j * 4;
  // Insert the explicit distance at the front; the last entry drops out.
  reps[rj] = d;
  reps[rj + 1] = rep0;
  reps[rj + 2] = rep1;
  reps[rj + 3] = rep2;
}

function updateDict(
  cost: Float64Array,
  src: Int32Array,
  kind: Uint8Array,
  dist: Int32Array,
  reps: Int32Array,
  i: number,
  j: number,
  c: number,
  start: number,
  rep0: number,
  rep1: number,
  rep2: number,
  rep3: number
): void {
  cost[j] = c;
  src[j] = i;
  kind[j] = DP_DICT;
  dist[j] = start;
  const rj = j * 4;
  // Dictionary matches do not modify the rep cache.
  reps[rj] = rep0;
  reps[rj + 1] = rep1;
  reps[rj + 2] = rep2;
  reps[rj + 3] = rep3;
}

/**
 * Greedy (optionally 1-step price-aware lazy) parse used by `fast` mode and by `small` mode
 * beyond the optimal-parse input bound.
 */
function parseGreedy(
  bytes: Uint8Array,
  dictionary: Uint8Array,
  dictIndex: DictIndex | undefined,
  pricing: ParsePricing
): Token[] {
  const n = bytes.length;
  const tokens: Token[] = [];
  const { litCostPrefix, window, maxDictStart } = pricing;

  const bits = hashBitsFor(n);
  const shift = 32 - bits;
  const head = headFor(headPool, bits);
  const prev = prevFor(n);

  let rep0 = INITIAL_REPS[0]!;
  let rep1 = INITIAL_REPS[1]!;
  let rep2 = INITIAL_REPS[2]!;
  let rep3 = INITIAL_REPS[3]!;

  // Best candidate at the probed position (scalar slots; no per-candidate allocation).
  let bestSavings = 0;
  let bestCost = 0;
  let bestLen = 0;
  let bestKind = 0; // 0 none, 1 history/rep, 2 dict.
  let bestDist = 0;
  let bestRep = -1;
  let bestStart = 0;

  const findBest = (pos: number): boolean => {
    const cap = n - pos < MATCH_LEN_CAP ? n - pos : MATCH_LEN_CAP;
    bestKind = 0;
    bestSavings = 0;
    if (cap < MIN_LEN_REP) return false;
    const litBase = litCostPrefix[pos]!;

    for (let r = 0; r < 4; r++) {
      const d = r === 0 ? rep0 : r === 1 ? rep1 : r === 2 ? rep2 : rep3;
      if (d > pos) continue;
      const len = matchLength(bytes, pos, bytes, pos - d, cap);
      if (len < MIN_LEN_REP) continue;
      const cost = pricing.repCost(r, len);
      const savings = litCostPrefix[pos + len]! - litBase - cost;
      if (savings > 0 && savings > bestSavings) {
        bestSavings = savings;
        bestCost = cost;
        bestLen = len;
        bestKind = 1;
        bestDist = d;
        bestRep = r;
      }
    }
    if (cap >= MIN_LEN_EXPLICIT && pos + 4 <= n) {
      let cand = head[hash4(bytes, pos, shift)]!;
      // The current position may already be indexed (the lazy probe inserts ahead); skip it.
      if (cand === pos) cand = prev[pos]!;
      const minPos = pos - window;
      let depth = GREEDY_DEPTH;
      let bestM = MIN_LEN_EXPLICIT - 1;
      while (cand >= 0 && cand >= minPos && depth-- > 0) {
        if (bytes[cand + bestM] === bytes[pos + bestM]) {
          const len = matchLength(bytes, pos, bytes, cand, cap);
          if (len > bestM) {
            bestM = len;
            const d = pos - cand;
            const r = d === rep0 ? 0 : d === rep1 ? 1 : d === rep2 ? 2 : d === rep3 ? 3 : -1;
            const cost = r >= 0 ? pricing.repCost(r, len) : pricing.historyCost(d, len);
            const savings = litCostPrefix[pos + len]! - litBase - cost;
            if (savings > 0 && savings > bestSavings) {
              bestSavings = savings;
              bestCost = cost;
              bestLen = len;
              bestKind = 1;
              bestDist = d;
              bestRep = r;
            }
            if (len === cap) break;
          }
        }
        cand = prev[cand]!;
      }
      if (dictIndex) {
        let dcand = dictIndex.head[hash4(bytes, pos, dictIndex.hashShift)]!;
        let depthD = GREEDY_DICT_DEPTH;
        let bestMD = MIN_LEN_EXPLICIT - 1;
        while (dcand >= 0 && depthD-- > 0) {
          if (dcand < maxDictStart && dictionary[dcand + bestMD] === bytes[pos + bestMD]) {
            const dcap = dictionary.length - dcand < cap ? dictionary.length - dcand : cap;
            const len = matchLength(bytes, pos, dictionary, dcand, dcap);
            if (len > bestMD) {
              bestMD = len;
              const cost = pricing.dictCost(dcand, len);
              const savings = litCostPrefix[pos + len]! - litBase - cost;
              if (savings > 0 && savings > bestSavings) {
                bestSavings = savings;
                bestCost = cost;
                bestLen = len;
                bestKind = 2;
                bestStart = dcand;
              }
              if (len === cap) break;
            }
          }
          dcand = dictIndex.prev[dcand]!;
        }
      }
    }
    return bestKind !== 0;
  };

  // Watermark keeping every position indexed exactly once, in increasing order (chains stay
  // sorted by recency, which the window early-exit and Pareto walks rely on).
  let insertedUpTo = 0;
  const insertUpTo = (limit: number): void => {
    const stop = limit < n - 3 ? limit : n - 3;
    while (insertedUpTo < stop) {
      const bucket = hash4(bytes, insertedUpTo, shift);
      prev[insertedUpTo] = head[bucket]!;
      head[bucket] = insertedUpTo;
      insertedUpTo++;
    }
    if (insertedUpTo < limit) insertedUpTo = limit;
  };

  const lazy = pricing.lazy;
  let pos = 0;
  let litStart = 0;
  while (pos < n) {
    insertUpTo(pos + 1);
    if (!findBest(pos)) {
      pos++;
      continue;
    }
    if (lazy && pos + 1 < n) {
      // Bounded price-aware lazy step: prefer deferring when a literal plus the next match
      // covers bytes at a strictly better price density.
      const curCost = bestCost;
      const curLen = bestLen;
      const curKind = bestKind;
      const curDist = bestDist;
      const curRep = bestRep;
      const curStart = bestStart;
      const litOne = litCostPrefix[pos + 1]! - litCostPrefix[pos]!;
      insertUpTo(pos + 2);
      if (findBest(pos + 1) && (litOne + bestCost) / (1 + bestLen) < curCost / curLen) {
        pos++;
        continue;
      }
      bestLen = curLen;
      bestKind = curKind;
      bestDist = curDist;
      bestRep = curRep;
      bestStart = curStart;
    }
    if (litStart < pos) tokens.push({ type: 'lit', start: litStart, end: pos });
    tokens.push(
      bestKind === 1
        ? { type: 'history', len: bestLen, dist: bestDist, rep: bestRep }
        : { type: 'dict', len: bestLen, start: bestStart }
    );
    if (bestKind === 1) {
      const d = bestDist;
      if (bestRep === 0) {
        // Rep0 hit: the cache is already fronted.
      } else if (bestRep === 1) {
        rep1 = rep0;
        rep0 = d;
      } else if (bestRep === 2) {
        rep2 = rep1;
        rep1 = rep0;
        rep0 = d;
      } else {
        // Explicit match and rep3 both shift the remaining three entries down.
        rep3 = rep2;
        rep2 = rep1;
        rep1 = rep0;
        rep0 = d;
      }
    }
    pos += bestLen;
    insertUpTo(pos);
    litStart = pos;
  }
  if (litStart < n) tokens.push({ type: 'lit', start: litStart, end: n });
  return tokens;
}
