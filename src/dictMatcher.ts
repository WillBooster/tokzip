/**
 * Exact preset-dictionary matcher: a suffix automaton over the reversed dictionary, built
 * once per language and cached. One backward pass over the input (matching statistics)
 * yields, for every input position, the longest dictionary match starting there plus the
 * automaton state whose suffix-link chain enumerates the exact Pareto front — for each
 * shorter length, the lowest (cheapest-slot) dictionary offset. Replaces the bounded
 * hash-chain walks for frame-dictionary matches in the optimal parse: unbounded effective
 * search depth at O(1) amortized cost per input byte, independent of dictionary size.
 */

import { slotOf } from './slots.ts';

const HASH_MULTIPLIER = 0x9E_37_79_B1;

export interface DictMatcher {
  /** Longest string length per state. */
  stateLen: Int32Array;
  /** Suffix link per state (-1 for the root). */
  stateLink: Int32Array;
  /**
   * Lowest dictionary start offset over the state's occurrences (identical for every
   * length in the state's range: the occurrence end in reversed coordinates fixes the
   * forward start).
   */
  stateMinStart: Int32Array;
  /**
   * Nearest suffix-link ancestor whose min-start offset slot is strictly smaller (-1 when
   * none). Offset slots weakly decrease toward the root (occurrence sets grow), so this
   * chain *is* the slot-merged Pareto front: equal-slot ancestors are redundant against a
   * deeper candidate (same bit price, and the deeper start supports every shorter length).
   */
  paretoParent: Int32Array;
  /** Open-addressing transition table: key = state * 256 + byte + 1, 0 = empty. */
  transKeys: Int32Array;
  transVals: Int32Array;
  transMask: number;
}

/** Builds the matcher for an assembled dictionary (wrapper + suffix). */
export function buildDictMatcher(dictionary: Uint8Array): DictMatcher {
  const n = dictionary.length;
  const maxStates = 2 * n + 4;
  const stateLen = new Int32Array(maxStates);
  const stateLink = new Int32Array(maxStates).fill(-1);
  const maxEnd = new Int32Array(maxStates).fill(-1);
  const trans = new Map<number, number>();
  // Per-state transition byte lists so a clone copies only its source's actual out-edges
  // (a fixed 256-probe sweep per clone would dominate construction on large dictionaries).
  const listHead = new Int32Array(maxStates).fill(-1);
  let listNext = new Int32Array(4 * n + 8);
  let listByte = new Uint8Array(4 * n + 8);
  let listCount = 0;
  const addTransition = (state: number, byte: number, to: number): void => {
    const key = state * 256 + byte;
    if (!trans.has(key)) {
      if (listCount === listNext.length) {
        const nextNext = new Int32Array(listNext.length * 2);
        nextNext.set(listNext);
        listNext = nextNext;
        const nextByte = new Uint8Array(listByte.length * 2);
        nextByte.set(listByte);
        listByte = nextByte;
      }
      listByte[listCount] = byte;
      listNext[listCount] = listHead[state]!;
      listHead[state] = listCount++;
    }
    trans.set(key, to);
  };
  let last = 0;
  let size = 1;
  // Standard online construction over the reversed dictionary: appending R[e] = dict[n-1-e]
  // makes an occurrence ending at reversed position e a forward match starting at n-1-e.
  for (let e = 0; e < n; e++) {
    const c = dictionary[n - 1 - e]!;
    const cur = size++;
    stateLen[cur] = stateLen[last]! + 1;
    maxEnd[cur] = e;
    let p = last;
    while (p !== -1 && !trans.has(p * 256 + c)) {
      addTransition(p, c, cur);
      p = stateLink[p]!;
    }
    if (p === -1) stateLink[cur] = 0;
    else {
      const q = trans.get(p * 256 + c)!;
      if (stateLen[p]! + 1 === stateLen[q]!) stateLink[cur] = q;
      else {
        const clone = size++;
        stateLen[clone] = stateLen[p]! + 1;
        stateLink[clone] = stateLink[q]!;
        for (let node = listHead[q]!; node !== -1; node = listNext[node]!) {
          const b = listByte[node]!;
          addTransition(clone, b, trans.get(q * 256 + b)!);
        }
        while (p !== -1 && trans.get(p * 256 + c) === q) {
          trans.set(p * 256 + c, clone);
          p = stateLink[p]!;
        }
        stateLink[q] = clone;
        stateLink[cur] = clone;
      }
    }
    last = cur;
  }

  // Propagate occurrence ends up the suffix-link tree (a state's occurrences are the union
  // of its link-tree children's), processing states by decreasing length via counting sort.
  const lenCount = new Int32Array(n + 2);
  for (let s = 0; s < size; s++) lenCount[stateLen[s]!]!++;
  for (let l = 1; l <= n; l++) lenCount[l]! += lenCount[l - 1]!;
  const byLen = new Int32Array(size);
  for (let s = 0; s < size; s++) byLen[--lenCount[stateLen[s]!]!] = s;
  for (let at = size - 1; at > 0; at--) {
    const s = byLen[at]!;
    const parent = stateLink[s]!;
    if (parent >= 0 && maxEnd[s]! > maxEnd[parent]!) maxEnd[parent] = maxEnd[s]!;
  }
  // Forward start offsets: an occurrence ending at reversed position e starts at n - 1 - e
  // regardless of match length, so the max end is the min start.
  const stateMinStart = maxEnd;
  for (let s = 0; s < size; s++) stateMinStart[s] = n - 1 - maxEnd[s]!;

  // Pareto-parent chain (see the interface doc). byLen orders parents before children, so
  // each state can extend its link's chain in O(1).
  const paretoParent = new Int32Array(size).fill(-1);
  for (let at = 1; at < size; at++) {
    const s = byLen[at]!;
    const parent = stateLink[s]!;
    if (parent <= 0) continue;
    paretoParent[s] = slotOf(stateMinStart[parent]!) < slotOf(stateMinStart[s]!) ? parent : paretoParent[parent]!;
  }

  // Freeze the transitions into an open-addressing table for the hot matching-statistics walk.
  // Sized for a ~0.75 load factor: linear probing stays short there, and the next power of
  // two would quadruple the table (measured 8 MB of an 11 MB matcher on a 128 KB dictionary).
  let capacity = 8;
  while (capacity * 3 < trans.size * 4) capacity <<= 1;
  const transKeys = new Int32Array(capacity);
  const transVals = new Int32Array(capacity);
  const transMask = capacity - 1;
  for (const [key, value] of trans) {
    let idx = Math.imul(key + 1, HASH_MULTIPLIER) & transMask;
    while (transKeys[idx] !== 0) idx = (idx + 1) & transMask;
    transKeys[idx] = key + 1;
    transVals[idx] = value;
  }
  return {
    // slice, not subarray: views would retain the whole maxStates-sized build buffers
    // (~25% dead memory) for the lifetime of the cached matcher.
    stateLen: stateLen.slice(0, size),
    stateLink: stateLink.slice(0, size),
    stateMinStart: stateMinStart.slice(0, size),
    paretoParent,
    transKeys,
    transVals,
    transMask,
  };
}

/**
 * Matching statistics over `bytes[parseStart, n)`, walked backward so that `msLen[i]` is the
 * longest dictionary match starting at `i` and `msState[i]` its automaton state (0 when none).
 */
export function computeMatchingStatistics(
  bytes: Uint8Array,
  parseStart: number,
  n: number,
  matcher: DictMatcher,
  msLen: Int32Array,
  msState: Int32Array
): void {
  const { stateLen, stateLink } = matcher;
  let state = 0;
  let length = 0;
  for (let i = n - 1; i >= parseStart; i--) {
    const c = bytes[i]!;
    for (;;) {
      const t = lookupTransition(matcher, state, c);
      if (t >= 0) {
        state = t;
        length++;
        break;
      }
      if (state === 0) {
        length = 0;
        break;
      }
      state = stateLink[state]!;
      length = stateLen[state]!;
    }
    msLen[i] = length;
    msState[i] = state;
  }
}

function lookupTransition(matcher: DictMatcher, state: number, byte: number): number {
  const key = state * 256 + byte + 1;
  const { transKeys, transMask } = matcher;
  let idx = Math.imul(key, HASH_MULTIPLIER) & transMask;
  while (true) {
    const k = transKeys[idx]!;
    if (k === key) return matcher.transVals[idx]!;
    if (k === 0) return -1;
    idx = (idx + 1) & transMask;
  }
}
