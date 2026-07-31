import { LENGTH_SLOT_COUNT, maxSlotValue } from './slots.ts';

/**
 * First payload char of every tokzip frame: magic 0b110 in the high 3 bits, version 2 in the
 * low 3. The format is still evolving pre-release: a version bump invalidates all previously
 * written frames (decoders reject other versions as 'unknown version', never misdecode).
 * v2 replaced the fast/small mode pair and the static-Huffman small body with a single
 * range-coded mode over adaptive models seeded from trained per-language priors.
 */
export const MAGIC_VERSION = 0b11_0010;

/**
 * First byte of every binary tokzip frame: bit 7 set (never a safe-ASCII text frame, whose
 * chars are all < 0x80) over the same 6-bit magic/version value as the text container.
 */
export const BINARY_MAGIC_VERSION = 0b1000_0000 | MAGIC_VERSION;

/** Shipped-mode values in the flags char (bits 1:0). Values 1 and 3 are invalid in v2. */
export const MODE_STORED = 0;
export const MODE_SMALL = 2;

/** Input-type flag (bit 2 of the flags char): 0 = string (UTF-8), 1 = bytes. */
export const FLAG_BYTES = 0b100;
/**
 * Fenced-dictionary flag (bit 3): inside a labeled code fence the dictionary space is
 * extended with that language's dictionary suffix above the frame dictionary (see fences.ts
 * and FORMAT.md §6.1).
 */
export const FLAG_FENCED = 0b1000;
/** Reserved flag bits (5:4): encoders write 0, decoders reject non-zero. */
export const RESERVED_FLAG_MASK = 0b11_0000;

/**
 * Fixed width of the header's CRC-32 field: 6 radix-64 chars in text frames (little-endian
 * 6-bit groups; the top 4 bits of the last group are zero and decoders reject non-zero),
 * 4 little-endian bytes in binary frames and stream blocks.
 */
export const CRC_TEXT_CHARS = 6;
export const CRC_BINARY_BYTES = 4;

/** Default `maxOutputSize` (64 MiB). */
export const DEFAULT_MAX_OUTPUT_SIZE = 64 * 1024 * 1024;

/** History window (normative): the farthest back a history match may reach. */
export const SMALL_WINDOW = 1 << 20; // 1 MB via offset slots.

/** Minimum encodable lengths (bases of the length coding) per kind. */
export const MIN_LEN_REP = 2;
export const MIN_LEN_EXPLICIT = 4; // History and dictionary matches.

/** Matches longer than this are split by encoders (bound of the length-slot alphabet). */
export const MATCH_LEN_CAP = maxSlotValue(LENGTH_SLOT_COUNT) + MIN_LEN_REP;

/** Initial rep-offset cache (history distances), most recent first. */
export const INITIAL_REPS: readonly number[] = [1, 2, 3, 4];

/**
 * Token kinds; also the previous-token-kind context ids for the adaptive models (the
 * context at the start of a body is LIT).
 */
export const TOKEN_KIND_LIT = 0;
export const TOKEN_KIND_HISTORY = 1;
export const TOKEN_KIND_DICT = 2;
export const TOKEN_KIND_REP0 = 3; // Kinds 3–6 are rep0–rep3.
export const TOKEN_KIND_COUNT = 7;

/** Upper bound on trained literal context classes (`litClassCount`). */
export const LIT_CLASS_MAX = 64;

/**
 * Adaptive-model layout (a single Uint16Array of 11-bit probabilities, P(bit = 0)).
 * Each compress/decompress call copies the language's trained priors and adapts the copy as
 * it codes, so short documents behave like the static model while long documents converge
 * to their own statistics. The layout is normative: encoder and decoder walk the same
 * indices in the same order.
 */
/** Offset length-bucket count: bucket = min(len − MIN_LEN_REP, OFF_LEN_BUCKETS − 1). */
export const OFF_LEN_BUCKETS = 4;

export const MODEL_IS_MATCH = 0; // TOKEN_KIND_COUNT bits: 0 = literal, 1 = match.
export const MODEL_IS_REP = MODEL_IS_MATCH + TOKEN_KIND_COUNT; // 0 = explicit, 1 = rep.
export const MODEL_IS_DICT = MODEL_IS_REP + TOKEN_KIND_COUNT; // 0 = history, 1 = dictionary.
export const MODEL_REP_TREE = MODEL_IS_DICT + TOKEN_KIND_COUNT; // 2-bit tree per context.
export const MODEL_LEN_TREE = MODEL_REP_TREE + TOKEN_KIND_COUNT * 3; // 3 groups × 63 nodes.
/** Offset slot trees: 2 groups × 4 length buckets (min(len − 2, 3)) × 63 nodes. */
export const MODEL_OFF_TREE = MODEL_LEN_TREE + 3 * 63;
export const MODEL_LITERAL = MODEL_OFF_TREE + 2 * OFF_LEN_BUCKETS * 63;
export const MODEL_BASE_SIZE = MODEL_LITERAL;

/**
 * Per-class literal block: a plain 8-bit tree (nodes 1–255 at [0, 255)) plus the two
 * matched subtrees ((1 + matchBit) · 256 + node − 1 at [256, 767)) used for the first
 * literals after a match, where the byte at rep0 distance predicts each bit until the
 * first mismatch (LZMA-style matched literals).
 */
export const LITERAL_BLOCK_SIZE = 768;

/** Length-model groups (second index of MODEL_LEN_TREE). */
export const LEN_GROUP_REP = 0;
export const LEN_GROUP_HISTORY = 1;
export const LEN_GROUP_DICT = 2;
/** Offset-model groups (first index of MODEL_OFF_TREE). */
export const OFF_GROUP_HISTORY = 0;
export const OFF_GROUP_DICT = 1;
/** Offset length bucket: min(len − MIN_LEN_REP, OFF_LEN_BUCKETS − 1). */
export function offLenBucketOf(len: number): number {
  const value = len - MIN_LEN_REP;
  return value < OFF_LEN_BUCKETS ? value : OFF_LEN_BUCKETS - 1;
}

/** Bit-tree depth: 64 leaves cover the 36 length slots and 40 offset slots. */
export const SLOT_TREE_BITS = 6;

export function modelSizeFor(litClassCount: number): number {
  return MODEL_BASE_SIZE + litClassCount * LITERAL_BLOCK_SIZE;
}
