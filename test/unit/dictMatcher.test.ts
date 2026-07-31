import { expect, test } from 'bun:test';
import { buildDictMatcher, computeMatchingStatistics } from '../../src/dictMatcher.ts';
import { slotOf } from '../../src/slots.ts';

/** Deterministic LCG so failures reproduce. */
function makeRandom(seed: number): () => number {
  let state = seed;
  return () => {
    state = (Math.imul(state, 1_103_515_245) + 12_345) & 0x7F_FF_FF_FF;
    return state;
  };
}

/** Longest match of input[i..] inside dict, by brute force. */
function bruteLongest(dict: Uint8Array, input: Uint8Array, i: number): number {
  let best = 0;
  for (let p = 0; p < dict.length; p++) {
    let l = 0;
    while (i + l < input.length && p + l < dict.length && dict[p + l] === input[i + l]) l++;
    if (l > best) best = l;
  }
  return best;
}

/** Lowest dict start offset of a length-l match of input[i..i+l), by brute force (-1 if none). */
function bruteMinStart(dict: Uint8Array, input: Uint8Array, i: number, l: number): number {
  for (let p = 0; p + l <= dict.length; p++) {
    let k = 0;
    while (k < l && dict[p + k] === input[i + k]) k++;
    if (k === l) return p;
  }
  return -1;
}

test('matching statistics agree with brute force on longest match and per-length min offset', () => {
  const random = makeRandom(0xC0_FF_EE);
  for (let round = 0; round < 6; round++) {
    // Small alphabets force dense repetition (and thus suffix-automaton clones).
    const alphabet = 3 + (round % 4);
    const dictLen = 200 + (random() % 600);
    const dict = new Uint8Array(dictLen);
    for (let p = 0; p < dictLen; p++) dict[p] = 97 + (random() % alphabet);
    const inputLen = 300;
    const input = new Uint8Array(inputLen);
    for (let i = 0; i < inputLen; i++) {
      if (random() % 3 === 0) {
        // Splice a dictionary fragment so long matches exist.
        const from = random() % dictLen;
        const take = Math.min(2 + (random() % 24), dictLen - from, inputLen - i);
        input.set(dict.subarray(from, from + take), i);
        i += take - 1;
      } else {
        input[i] = 97 + (random() % (alphabet + 1));
      }
    }

    const matcher = buildDictMatcher(dict);
    const msLen = new Int32Array(inputLen);
    const msState = new Int32Array(inputLen);
    computeMatchingStatistics(input, 0, inputLen, matcher, msLen, msState);
    const { stateLen, stateLink, stateMinStart, paretoParent } = matcher;

    // Offset slots weakly decrease toward the root, and paretoParent is the nearest
    // strictly-smaller-slot ancestor (the invariants the parse's Pareto walk relies on).
    for (let s = 1; s < stateLen.length; s++) {
      const parent = stateLink[s]!;
      if (parent > 0) expect(slotOf(stateMinStart[parent]!)).toBeLessThanOrEqual(slotOf(stateMinStart[s]!));
      let expected = parent;
      while (expected > 0 && slotOf(stateMinStart[expected]!) >= slotOf(stateMinStart[s]!)) {
        expected = stateLink[expected]!;
      }
      expect(paretoParent[s]).toBe(expected > 0 ? expected : -1);
    }
    for (let i = 0; i < inputLen; i++) {
      expect(msLen[i]).toBe(bruteLongest(dict, input, i));
      // For every length, the state covering it on the suffix-link chain must report the
      // exact lowest start offset (the Pareto front the optimal parse prices).
      for (let l = msLen[i]!; l >= 1; l--) {
        let s = msState[i]!;
        while (s !== 0 && stateLen[stateLink[s]!]! >= l) s = stateLink[s]!;
        expect(stateLen[s]).toBeGreaterThanOrEqual(l);
        expect(stateMinStart[s]).toBe(bruteMinStart(dict, input, i, l));
      }
    }
  }
});
