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
const OPTIMAL_DEPTH = 96;
const OPTIMAL_DICT_DEPTH = 64;

/** The DP evaluates every match length up to this bound, then only slot-boundary lengths. */
const DENSE_LEN_BOUND = 66;

/**
 * Matches at least this long are taken whole and the DP jumps past them (zstd-style immediate
 * encoding). Splitting such a match is essentially never profitable, and the jump keeps
 * degenerate inputs (long byte runs) linear instead of quadratic.
 */
const CUT_LEN = 128;

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
  language.dictIndex = { hashShift: shift, head, prev };
  return language.dictIndex;
}

function matchLength(a: Uint8Array, ai: number, b: Uint8Array, bi: number, cap: number): number {
  let len = 0;
  while (len < cap && a[ai + len] === b[bi + len]) len++;
  return len;
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
let chainPrev = new Int32Array(0);
let dpCost = new Float64Array(0);
let dpSrc = new Int32Array(0);
let dpKind = new Uint8Array(0);
let dpDist = new Int32Array(0);
let dpReps = new Int32Array(0);
// Pareto match candidates collected per position (distance/start ascending in cost, length ascending).
const candDist = new Int32Array(OPTIMAL_DEPTH);
const candLen = new Int32Array(OPTIMAL_DEPTH);

function headFor(bits: number): Int32Array {
  let head = headPool.get(bits);
  if (!head) {
    head = new Int32Array(1 << bits);
    headPool.set(bits, head);
  }
  head.fill(-1);
  return head;
}

function prevFor(length: number): Int32Array {
  if (chainPrev.length < length) chainPrev = new Int32Array(Math.max(length, chainPrev.length * 2, 4096));
  return chainPrev;
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
  const head = headFor(bits);
  const prev = prevFor(n);

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
      // Rep matches (min length 2).
      let cutHappened = false;
      for (let r = 0; r < 4; r++) {
        const d = r === 0 ? rep0 : r === 1 ? rep1 : r === 2 ? rep2 : rep3;
        if (d > i) continue;
        const m = matchLength(bytes, i, bytes, i - d, cap);
        if (m < MIN_LEN_REP) continue;
        const rowBase = r * LENGTH_SLOT_COUNT;
        if (m >= CUT_LEN) {
          const c = base + repSlotBits[rowBase + slotOf(m - MIN_LEN_REP)]!;
          const j = i + m;
          if (c < cost[j]!) updateRep(cost, src, kind, dist, reps, i, j, c, r, d, rep0, rep1, rep2, rep3);
          if (i + 4 <= n) {
            const bucket = hash4(bytes, i, shift);
            prev[i] = head[bucket]!;
            head[bucket] = i;
          }
          i = j - 1;
          cutHappened = true;
          break;
        }
        const denseEnd = m < DENSE_LEN_BOUND ? m : DENSE_LEN_BOUND;
        for (let len = MIN_LEN_REP; len <= denseEnd; len++) {
          const c = base + repSlotBits[rowBase + slotOf(len - MIN_LEN_REP)]!;
          const j = i + len;
          if (c < cost[j]!) updateRep(cost, src, kind, dist, reps, i, j, c, r, d, rep0, rep1, rep2, rep3);
        }
        if (m > DENSE_LEN_BOUND) {
          for (let s = slotOf(DENSE_LEN_BOUND - MIN_LEN_REP); s < LENGTH_SLOT_COUNT; s++) {
            const len = Math.min(m, SLOT_MAX_VALUE[s]! + MIN_LEN_REP);
            if (len <= DENSE_LEN_BOUND) continue;
            const c = base + repSlotBits[rowBase + slotOf(len - MIN_LEN_REP)]!;
            const j = i + len;
            if (c < cost[j]!) updateRep(cost, src, kind, dist, reps, i, j, c, r, d, rep0, rep1, rep2, rep3);
            if (len === m) break;
          }
        }
      }
      if (cutHappened) continue;

      if (cap >= MIN_LEN_EXPLICIT && i + 4 <= n) {
        // Explicit history matches: walk the chain collecting the Pareto set (nearer candidates
        // first, so a farther candidate is kept only when it extends the match length).
        let candCount = 0;
        {
          let cand = head[hash4(bytes, i, shift)]!;
          let bestM = MIN_LEN_EXPLICIT - 1;
          const minPos = i - window;
          let depth = OPTIMAL_DEPTH;
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
                if (m === cap) break;
              }
            }
            cand = prev[cand]!;
          }
        }
        if (candCount > 0 && candLen[candCount - 1]! >= CUT_LEN) {
          // Immediate encoding: take the longest match whole and jump past it.
          const d = candDist[candCount - 1]!;
          const m = candLen[candCount - 1]!;
          const c = base + histSlotBits[slotOf(m - MIN_LEN_REP)]! + offsetSlotBits[slotOf(d - 1)]!;
          const j = i + m;
          if (c < cost[j]!) updateHistory(cost, src, kind, dist, reps, i, j, c, d, rep0, rep1, rep2);
          const bucket = hash4(bytes, i, shift);
          prev[i] = head[bucket]!;
          head[bucket] = i;
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

        // Dictionary matches (no rep-cache interaction).
        if (dictIndex) {
          let candCountD = 0;
          {
            let cand = dictIndex.head[hash4(bytes, i, dictIndex.hashShift)]!;
            let bestM = MIN_LEN_EXPLICIT - 1;
            let depth = OPTIMAL_DICT_DEPTH;
            while (cand >= 0 && depth-- > 0) {
              if (cand < maxDictStart && dictionary[cand + bestM] === bytes[i + bestM]) {
                const dcap = dictionary.length - cand < cap ? dictionary.length - cand : cap;
                const m = matchLength(bytes, i, dictionary, cand, dcap);
                if (m > bestM) {
                  candDist[candCountD] = cand;
                  candLen[candCountD] = m;
                  candCountD++;
                  bestM = m;
                  if (m === cap) break;
                }
              }
              cand = dictIndex.prev[cand]!;
            }
          }
          if (candCountD > 0 && candLen[candCountD - 1]! >= CUT_LEN) {
            const start = candDist[candCountD - 1]!;
            const m = candLen[candCountD - 1]!;
            const c = base + dictSlotBits[slotOf(m - MIN_LEN_REP)]! + offsetSlotBits[slotOf(start)]!;
            const j = i + m;
            if (c < cost[j]!) updateDict(cost, src, kind, dist, reps, i, j, c, start, rep0, rep1, rep2, rep3);
            const bucket = hash4(bytes, i, shift);
            prev[i] = head[bucket]!;
            head[bucket] = i;
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

    if (i + 4 <= n) {
      const bucket = hash4(bytes, i, shift);
      prev[i] = head[bucket]!;
      head[bucket] = i;
    }
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
  const head = headFor(bits);
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
