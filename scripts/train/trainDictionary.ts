/**
 * COVER-like dictionary trainer. Segments are scored with a fixed-cost heuristic — a
 * segment's value is the bytes it covers per occurrence minus a flat per-reference
 * overhead — not with the v2 range coder's true (context-, prior-, and slot-dependent)
 * prices; modeling those exactly here would couple dictionary packing to a model that is
 * itself trained on the packed dictionary. The flat overhead is a proxy for a typical
 * dictionary-token cost.
 */

const SEGMENT_LENGTHS = [128, 96, 64, 48, 32, 24, 16, 12, 8, 6, 4] as const;
/** Flat per-reference overhead (bytes a dictionary match roughly costs to encode). */
const MATCH_OVERHEAD_CHARS = 3.5;
/** Bound on dictionary-training input (chars) so n-gram counting stays tractable. */
const MAX_TRAINING_CHARS = 24_000_000;
const MAX_SELECTED_CANDIDATES = 400_000;

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
    const cap = length >= 16 ? 800_000 : 1_600_000;
    for (const [segment, freq] of countNgrams(bounded, length, cap)) {
      if (freq < 3) continue;
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
  // Containment prefilter: a segment contained in coveredProbe must have every 4-gram of its
  // prefix present, so an absent leading 4-gram skips the (linear-scan) includes() probe.
  // Rebuilt incrementally as packed grows; keeps large budgets tractable.
  const gramSet = new Set<string>();
  let gramIndexed = 0;
  const indexGramsUpTo = (probe: string): void => {
    for (; gramIndexed + 4 <= probe.length; gramIndexed++) gramSet.add(probe.slice(gramIndexed, gramIndexed + 4));
  };
  indexGramsUpTo(coveredProbe);
  for (const { segment } of candidates.slice(0, MAX_SELECTED_CANDIDATES)) {
    const mayBeCovered = segment.length < 4 || gramSet.has(segment.slice(0, 4));
    if (mayBeCovered && coveredProbe.includes(segment)) continue;
    const overlap = tailOverlap(packed, segment);
    const addition = segment.slice(overlap);
    const additionBytes = encoder.encode(addition).length;
    if (packedBytes + additionBytes > budgetBytes) continue;
    packed += addition;
    packedBytes += additionBytes;
    coveredProbe = alreadyCovered + packed;
    indexGramsUpTo(coveredProbe);
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
