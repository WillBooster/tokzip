import { expect, test } from 'bun:test';
import {
  decodeTree,
  encodeTree,
  PROB_MAX,
  PROB_MIN,
  PROB_SCALE,
  RangeDecoder,
  RangeEncoder,
} from '../../src/rangeCoder.ts';

function makeRandom(seed: number): () => number {
  let state = seed;
  return () => {
    state = (Math.imul(state, 1_103_515_245) + 12_345) & 0x7F_FF_FF_FF;
    return state;
  };
}

test('bit, direct-bit, and tree streams round-trip with adaptive probabilities', () => {
  const random = makeRandom(0xBE_EF);
  for (let round = 0; round < 20; round++) {
    // Skewed priors exercise carry propagation (long runs of near-certain bits).
    const prior = round % 3 === 0 ? PROB_MIN : round % 3 === 1 ? PROB_MAX : PROB_SCALE >> 1;
    const encProbs = new Uint16Array(64).fill(prior);
    const decProbs = new Uint16Array(64).fill(prior);
    const treeEncProbs = new Uint16Array(63).fill(PROB_SCALE >> 1);
    const treeDecProbs = new Uint16Array(63).fill(PROB_SCALE >> 1);

    interface Op {
      kind: 'bit' | 'direct' | 'tree';
      value: number;
      extra: number;
    }
    const ops: Op[] = [];
    const count = 500 + (random() % 500);
    for (let i = 0; i < count; i++) {
      const pick = random() % 3;
      if (pick === 0) ops.push({ kind: 'bit', value: random() % 100 < 90 ? 0 : 1, extra: random() % 64 });
      else if (pick === 1) {
        const bits = 1 + (random() % 20);
        ops.push({ kind: 'direct', value: random() % 2 ** bits, extra: bits });
      } else ops.push({ kind: 'tree', value: random() % 64, extra: 6 });
    }

    const encoder = new RangeEncoder();
    for (const op of ops) {
      if (op.kind === 'bit') encoder.encodeBit(encProbs, op.extra, op.value);
      else if (op.kind === 'direct') encoder.encodeDirect(op.value, op.extra);
      else encodeTree(encoder, treeEncProbs, 0, op.extra, op.value);
    }
    const bytes = encoder.finish();

    const decoder = new RangeDecoder(bytes, 0, bytes.length);
    for (const op of ops) {
      const value =
        op.kind === 'bit'
          ? decoder.decodeBit(decProbs, op.extra)
          : op.kind === 'direct'
            ? decoder.decodeDirect(op.extra)
            : decodeTree(decoder, treeDecProbs, 0, op.extra);
      expect(value).toBe(op.value);
    }
    // Encoder and decoder must agree on the adapted state and consume the exact bytes.
    expect(decProbs).toEqual(encProbs);
    expect(treeDecProbs).toEqual(treeEncProbs);
    expect(decoder.position).toBe(bytes.length);
  }
});
