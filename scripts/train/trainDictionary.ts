/**
 * COVER-like dictionary trainer, scored with the codec cost model rather than raw
 * bytes × frequency: a segment's value is the output chars it saves per occurrence
 * (literal cost of the covered bytes minus tag + offset cost of a dictionary match),
 * so the trainer's objective matches what benchmarks measure.
 */

const SEGMENT_LENGTHS = [128, 96, 64, 48, 32, 24, 16, 12, 8, 6, 4] as const;
/** Approximate fast-mode cost of a dictionary reference (tag + 2–3 offset chars). */
const MATCH_OVERHEAD_CHARS = 3.5;
/** Bound on dictionary-training input (chars) so n-gram counting stays tractable. */
const MAX_TRAINING_CHARS = 10_000_000;
const MAX_SELECTED_CANDIDATES = 150_000;

interface Candidate {
  segment: string;
  score: number;
}

function countNgrams(docs: string[], length: number, cap: number): Map<string, number> {
  const counts = new Map<string, number>();
  const stride = length >= 48 ? 2 : 1;
  for (const doc of docs) {
    for (let i = 0; i + length <= doc.length; i += stride) {
      const gram = doc.slice(i, i + length);
      const current = counts.get(gram);
      if (current !== undefined) counts.set(gram, current + 1);
      else if (counts.size < cap) counts.set(gram, 1);
    }
  }
  return counts;
}

/**
 * Greedy cost-scored packing: rank segments by saved chars per dictionary byte, then append
 * highest-density segments (skipping ones already contained) until the budget is filled.
 * Appending reuses the longest dictionary tail that prefixes the segment (suffix–prefix
 * packing), so the budget buys strictly more coverage than plain concatenation.
 * Most valuable segments land at the lowest offsets, where references are cheapest.
 */
export function trainDictionary(docs: string[], budgetBytes: number, alreadyCovered: string): Uint8Array {
  const bounded: string[] = [];
  let total = 0;
  for (const doc of docs) {
    if (total >= MAX_TRAINING_CHARS) break;
    const take = doc.slice(0, MAX_TRAINING_CHARS - total);
    bounded.push(take);
    total += take.length;
  }

  const candidates: Candidate[] = [];
  for (const length of SEGMENT_LENGTHS) {
    const cap = length >= 16 ? 400_000 : 800_000;
    for (const [segment, freq] of countNgrams(bounded, length, cap)) {
      if (freq < 4) continue;
      const savedPerOccurrence = length - MATCH_OVERHEAD_CHARS;
      if (savedPerOccurrence <= 0) continue;
      // Density: chars saved across occurrences per dictionary byte spent.
      candidates.push({ segment, score: (freq * savedPerOccurrence) / length });
    }
  }
  candidates.sort((a, b) => b.score - a.score);

  const encoder = new TextEncoder();
  let packed = '';
  let packedBytes = 0;
  let coveredProbe = alreadyCovered;
  for (const { segment } of candidates.slice(0, MAX_SELECTED_CANDIDATES)) {
    if (coveredProbe.includes(segment)) continue;
    const overlap = tailOverlap(packed, segment);
    const addition = segment.slice(overlap);
    const additionBytes = encoder.encode(addition).length;
    if (packedBytes + additionBytes > budgetBytes) continue;
    packed += addition;
    packedBytes += additionBytes;
    coveredProbe = alreadyCovered + packed;
    if (packedBytes >= budgetBytes - 4) break;
  }
  return encoder.encode(packed);
}

/** Longest `packed` suffix that is also a prefix of `segment` (< segment length). */
function tailOverlap(packed: string, segment: string): number {
  const max = Math.min(packed.length, segment.length - 1);
  for (let overlap = max; overlap > 0; overlap--) {
    if (packed.endsWith(segment.slice(0, overlap))) return overlap;
  }
  return 0;
}
