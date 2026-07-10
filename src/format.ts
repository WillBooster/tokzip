import { LENGTH_SLOT_COUNT, maxSlotValue } from './slots.ts';

/** First payload char of every tokzip frame: magic 0b110 in the high 3 bits, version 1 in the low 3. */
export const MAGIC_VERSION = 0b11_0001;

/** Shipped-mode values in the flags char (bits 1:0). Value 3 is invalid. */
export const MODE_STORED = 0;
export const MODE_FAST = 1;
export const MODE_SMALL = 2;

/** Input-type flag (bit 2 of the flags char): 0 = string (UTF-8), 1 = bytes. */
export const FLAG_BYTES = 0b100;
/** Reserved flag bits (5:3): encoders write 0, decoders reject non-zero. */
export const RESERVED_FLAG_MASK = 0b11_1000;

/** Default `maxOutputSize` (64 MiB). */
export const DEFAULT_MAX_OUTPUT_SIZE = 64 * 1024 * 1024;

/** `fast`-mode tag kinds (3 high bits of the tag char). All 8 values are allocated. */
export const KIND_LIT64 = 0;
export const KIND_LITRAW = 1;
export const KIND_HISTORY = 2;
export const KIND_DICT = 3;
export const KIND_REP0 = 4; // Kinds 4–7 are rep0–rep3.

/** History windows (normative per mode). */
export const FAST_WINDOW = 1 << 18; // 256 KB (18-bit offset field holds distance - 1).
export const SMALL_WINDOW = 1 << 20; // 1 MB via offset slots.

/** Offset-field widths in `fast` mode: 2 chars (12 bits) or 3 chars (18 bits). */
export const SHORT_OFFSET_LIMIT = 1 << 12;

/** Minimum encodable lengths (bases of the length coding) per kind. */
export const MIN_LEN_REP = 2;
export const MIN_LEN_EXPLICIT = 4; // History and dictionary matches.

/** Matches longer than this are split by encoders (bound of the `small` length-slot alphabet). */
export const MATCH_LEN_CAP = maxSlotValue(LENGTH_SLOT_COUNT) + MIN_LEN_REP;

/** Initial rep-offset cache (history distances), most recent first. */
export const INITIAL_REPS: readonly number[] = [1, 2, 3, 4];

/** `small`-mode token-stream alphabet: 7 kinds × 36 length slots. */
export const TOKEN_KIND_LITRUN = 0;
export const TOKEN_KIND_HISTORY = 1;
export const TOKEN_KIND_DICT = 2;
export const TOKEN_KIND_REP0 = 3; // Kinds 3–6 are rep0–rep3.
export const TOKEN_KIND_COUNT = 7;
export const TOKEN_ALPHABET_SIZE = TOKEN_KIND_COUNT * LENGTH_SLOT_COUNT; // 252

/** Raw-mode (non-Huffman) fixed widths per `small` stream. */
export const RAW_LITERAL_BITS = 8;
export const RAW_TOKEN_BITS = 8;
export const RAW_OFFSET_SLOT_BITS = 6;
