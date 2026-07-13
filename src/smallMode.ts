import type { EntropyTables, RegisteredLanguage } from './dictionary.ts';
import { TokzipDecodeError } from './errors.ts';
import {
  INITIAL_REPS,
  MIN_LEN_REP,
  OFFSET_CONTEXT_COUNT,
  OFFSET_CONTEXT_DICT,
  OFFSET_CONTEXT_HISTORY,
  RAW_LITERAL_BITS,
  RAW_OFFSET_SLOT_BITS,
  RAW_TOKEN_BITS,
  SMALL_WINDOW,
  TOKEN_ALPHABET_SIZE,
  TOKEN_CONTEXT_COUNT,
  TOKEN_KIND_DICT,
  TOKEN_KIND_HISTORY,
  TOKEN_KIND_LITRUN,
  TOKEN_KIND_REP0,
} from './format.ts';
import { buildDecoder, buildEncoder, type HuffmanEncoder, MAX_CODE_LENGTH } from './huffman.ts';
import type { ParsePricing, SlotPricing, Token } from './lz.ts';
import { BitReader, BitWriter, decodeRadix85 } from './radix85.ts';
import {
  extraBitsOf,
  extraValueOf,
  LENGTH_SLOT_COUNT,
  maxSlotValue,
  OFFSET_SLOT_COUNT,
  slotOf,
  valueOfSlot,
} from './slots.ts';

/** Longest literal run one litrun token can carry (runs beyond this split). */
const MAX_RUN_LENGTH = maxSlotValue(LENGTH_SLOT_COUNT) + 1;

/** Decode-side lookups replacing a division/modulo per token symbol. */
const SYMBOL_KIND = new Uint8Array(TOKEN_ALPHABET_SIZE);
const SYMBOL_SLOT = new Uint8Array(TOKEN_ALPHABET_SIZE);
for (let symbol = 0; symbol < TOKEN_ALPHABET_SIZE; symbol++) {
  SYMBOL_KIND[symbol] = Math.trunc(symbol / LENGTH_SLOT_COUNT);
  SYMBOL_SLOT[symbol] = symbol % LENGTH_SLOT_COUNT;
}

function bitVarintLength(value: number): number {
  let length = 8;
  for (let rest = Math.floor(value / 128); rest > 0; rest = Math.floor(rest / 128)) length += 8;
  return length;
}

/**
 * Exact bit-price tables derived from a language's static context tables, cached per language:
 * the context-indexed slot tables driving the optimal parse, the per-(class, byte) literal
 * prices, and context-averaged token prices for the greedy (large-input) parser, whose
 * stateless pricing callbacks cannot know the previous token kind.
 */
interface SmallTables {
  parser: SlotPricing;
  /** Exact literal bit price, indexed `contextClass * 256 + byte`. */
  litBits: Float64Array;
  avgHistSlotBits: Float64Array;
  avgDictSlotBits: Float64Array;
  avgRepSlotBits: Float64Array;
}

const smallTablesCache = new WeakMap<EntropyTables, SmallTables>();

function smallTablesFor(tables: EntropyTables): SmallTables {
  let cached = smallTablesCache.get(tables);
  if (cached) return cached;
  const { literal, token, offset, litClassCount } = tables;
  const litBits = new Float64Array(litClassCount * 256);
  for (let i = 0; i < litBits.length; i++) litBits[i] = literal[i]! || RAW_LITERAL_BITS;
  const tokenBits = (ctx: number, symbol: number): number =>
    token[ctx * TOKEN_ALPHABET_SIZE + symbol]! || RAW_TOKEN_BITS;

  const histSlotBits = new Float64Array(TOKEN_CONTEXT_COUNT * LENGTH_SLOT_COUNT);
  const dictSlotBits = new Float64Array(TOKEN_CONTEXT_COUNT * LENGTH_SLOT_COUNT);
  const repSlotBits = new Float64Array(TOKEN_CONTEXT_COUNT * 4 * LENGTH_SLOT_COUNT);
  const litRunStartBits = new Float64Array(TOKEN_CONTEXT_COUNT);
  for (let ctx = 0; ctx < TOKEN_CONTEXT_COUNT; ctx++) {
    litRunStartBits[ctx] = tokenBits(ctx, TOKEN_KIND_LITRUN * LENGTH_SLOT_COUNT);
    for (let s = 0; s < LENGTH_SLOT_COUNT; s++) {
      const extra = extraBitsOf(s);
      histSlotBits[ctx * LENGTH_SLOT_COUNT + s] = tokenBits(ctx, TOKEN_KIND_HISTORY * LENGTH_SLOT_COUNT + s) + extra;
      dictSlotBits[ctx * LENGTH_SLOT_COUNT + s] = tokenBits(ctx, TOKEN_KIND_DICT * LENGTH_SLOT_COUNT + s) + extra;
      for (let r = 0; r < 4; r++) {
        repSlotBits[(ctx * 4 + r) * LENGTH_SLOT_COUNT + s] =
          tokenBits(ctx, (TOKEN_KIND_REP0 + r) * LENGTH_SLOT_COUNT + s) + extra;
      }
    }
  }
  const histOffsetSlotBits = new Float64Array(OFFSET_SLOT_COUNT);
  const dictOffsetSlotBits = new Float64Array(OFFSET_SLOT_COUNT);
  for (let s = 0; s < OFFSET_SLOT_COUNT; s++) {
    const extra = extraBitsOf(s);
    histOffsetSlotBits[s] = (offset[OFFSET_CONTEXT_HISTORY * OFFSET_SLOT_COUNT + s]! || RAW_OFFSET_SLOT_BITS) + extra;
    dictOffsetSlotBits[s] = (offset[OFFSET_CONTEXT_DICT * OFFSET_SLOT_COUNT + s]! || RAW_OFFSET_SLOT_BITS) + extra;
  }

  const avgHistSlotBits = new Float64Array(LENGTH_SLOT_COUNT);
  const avgDictSlotBits = new Float64Array(LENGTH_SLOT_COUNT);
  const avgRepSlotBits = new Float64Array(4 * LENGTH_SLOT_COUNT);
  for (let s = 0; s < LENGTH_SLOT_COUNT; s++) {
    let hist = 0;
    let dict = 0;
    for (let ctx = 0; ctx < TOKEN_CONTEXT_COUNT; ctx++) {
      hist += histSlotBits[ctx * LENGTH_SLOT_COUNT + s]!;
      dict += dictSlotBits[ctx * LENGTH_SLOT_COUNT + s]!;
    }
    avgHistSlotBits[s] = hist / TOKEN_CONTEXT_COUNT;
    avgDictSlotBits[s] = dict / TOKEN_CONTEXT_COUNT;
    for (let r = 0; r < 4; r++) {
      let rep = 0;
      for (let ctx = 0; ctx < TOKEN_CONTEXT_COUNT; ctx++) {
        rep += repSlotBits[(ctx * 4 + r) * LENGTH_SLOT_COUNT + s]!;
      }
      avgRepSlotBits[r * LENGTH_SLOT_COUNT + s] = rep / TOKEN_CONTEXT_COUNT;
    }
  }

  cached = {
    parser: { litRunStartBits, histSlotBits, dictSlotBits, repSlotBits, histOffsetSlotBits, dictOffsetSlotBits },
    litBits,
    avgHistSlotBits,
    avgDictSlotBits,
    avgRepSlotBits,
  };
  smallTablesCache.set(tables, cached);
  return cached;
}

// Scratch prefix buffer reused across calls (compress is synchronous; each pricing's prefix
// is only read while its own parse runs, and fastMode keeps a separate scratch).
let smallPrefixScratch = new Float64Array(0);

/**
 * Builds the `small`-mode pricing model: exact output-bit prices from the static per-language
 * context tables. The attached slot tables let the parser run its exact-price optimal parse.
 */
export function smallPricing(bytes: Uint8Array, language: RegisteredLanguage): ParsePricing {
  const tables = smallTablesFor(language.tables);
  const { litContext } = language.tables;
  const { litBits, parser } = tables;

  // Literal prices are context-exact even inside the DP: the context is the previous input
  // byte, which does not depend on the tokenization path.
  if (smallPrefixScratch.length < bytes.length + 1) {
    smallPrefixScratch = new Float64Array(Math.max(bytes.length + 1, smallPrefixScratch.length * 2, 4096));
  }
  const litCostPrefix = smallPrefixScratch;
  let acc = 0;
  let prev = 0;
  for (let i = 0; i < bytes.length; i++) {
    const byte = bytes[i]!;
    acc += litBits[litContext[prev]! * 256 + byte]!;
    litCostPrefix[i + 1] = acc;
    prev = byte;
  }

  return {
    litCostPrefix,
    repCost: (repIndex, len) => tables.avgRepSlotBits[repIndex * LENGTH_SLOT_COUNT + slotOf(len - MIN_LEN_REP)]!,
    historyCost: (dist, len) =>
      tables.avgHistSlotBits[slotOf(len - MIN_LEN_REP)]! + parser.histOffsetSlotBits[slotOf(dist - 1)]!,
    dictCost: (start, len) =>
      tables.avgDictSlotBits[slotOf(len - MIN_LEN_REP)]! + parser.dictOffsetSlotBits[slotOf(start)]!,
    lazy: true,
    window: SMALL_WINDOW,
    maxDictStart: SMALL_WINDOW,
    optimal: parser,
  };
}

interface StreamPlan {
  huffman: boolean;
  bitLength: number;
}

/** A fully priced `small` body: sizes are exact, so the frame comparison never needs emission. */
export interface SmallPlan {
  literals: StreamPlan;
  tokenStream: StreamPlan;
  offsets: StreamPlan;
  tokenCount: number;
  totalBits: number;
  /** Exact char count of the emitted radix-85 body. */
  charCost: number;
  collected: CollectedStreams;
}

interface CollectedStreams {
  literalBytes: Uint8Array;
  /** Literal context class per literal (the trained class of the previous input byte). */
  literalCtxs: Uint8Array;
  literalCount: number;
  tokenSyms: Uint8Array;
  /** Token context per token (previous token kind; litrun at the start). */
  tokenCtxs: Uint8Array;
  tokenExtraBits: Uint8Array;
  tokenExtraValues: Int32Array;
  tokenCount: number;
  offsetSlots: Uint8Array;
  /** Offset context per offset (history or dict). */
  offsetCtxs: Uint8Array;
  offsetExtraBits: Uint8Array;
  offsetExtraValues: Int32Array;
  offsetCount: number;
  tokenExtraTotal: number;
  offsetExtraTotal: number;
}

function collectStreams(tokens: Token[], bytes: Uint8Array, tables: EntropyTables): CollectedStreams {
  const { litContext } = tables;
  // Every capacity is exact or a safe upper bound: literal runs longer than MAX_RUN_LENGTH
  // split into extra litrun tokens.
  const tokenCap = tokens.length + Math.ceil(bytes.length / MAX_RUN_LENGTH) + 1;
  const s: CollectedStreams = {
    literalBytes: new Uint8Array(bytes.length),
    literalCtxs: new Uint8Array(bytes.length),
    literalCount: 0,
    tokenSyms: new Uint8Array(tokenCap),
    tokenCtxs: new Uint8Array(tokenCap),
    tokenExtraBits: new Uint8Array(tokenCap),
    tokenExtraValues: new Int32Array(tokenCap),
    tokenCount: 0,
    offsetSlots: new Uint8Array(tokens.length),
    offsetCtxs: new Uint8Array(tokens.length),
    offsetExtraBits: new Uint8Array(tokens.length),
    offsetExtraValues: new Int32Array(tokens.length),
    offsetCount: 0,
    tokenExtraTotal: 0,
    offsetExtraTotal: 0,
  };
  let prevKind = TOKEN_KIND_LITRUN;
  const pushToken = (kind: number, lenValue: number): void => {
    const slot = slotOf(lenValue);
    const extra = extraBitsOf(slot);
    const at = s.tokenCount++;
    s.tokenSyms[at] = kind * LENGTH_SLOT_COUNT + slot;
    s.tokenCtxs[at] = prevKind;
    s.tokenExtraBits[at] = extra;
    s.tokenExtraValues[at] = extraValueOf(lenValue, slot);
    s.tokenExtraTotal += extra;
    prevKind = kind;
  };
  const pushOffset = (value: number, ctx: number): void => {
    const slot = slotOf(value);
    const extra = extraBitsOf(slot);
    const at = s.offsetCount++;
    s.offsetSlots[at] = slot;
    s.offsetCtxs[at] = ctx;
    s.offsetExtraBits[at] = extra;
    s.offsetExtraValues[at] = extraValueOf(value, slot);
    s.offsetExtraTotal += extra;
  };
  for (const token of tokens) {
    if (token.type === 'lit') {
      // Runs beyond the length-slot alphabet are split into consecutive litrun tokens.
      for (let start = token.start; start < token.end; start += MAX_RUN_LENGTH) {
        const end = Math.min(start + MAX_RUN_LENGTH, token.end);
        pushToken(TOKEN_KIND_LITRUN, end - start - 1);
        let prevByte = start > 0 ? bytes[start - 1]! : 0;
        for (let i = start; i < end; i++) {
          const byte = bytes[i]!;
          const at = s.literalCount++;
          s.literalBytes[at] = byte;
          s.literalCtxs[at] = litContext[prevByte]!;
          prevByte = byte;
        }
      }
    } else if (token.type === 'history') {
      if (token.rep >= 0) pushToken(TOKEN_KIND_REP0 + token.rep, token.len - MIN_LEN_REP);
      else {
        pushToken(TOKEN_KIND_HISTORY, token.len - MIN_LEN_REP);
        pushOffset(token.dist - 1, OFFSET_CONTEXT_HISTORY);
      }
    } else {
      pushToken(TOKEN_KIND_DICT, token.len - MIN_LEN_REP);
      pushOffset(token.start, OFFSET_CONTEXT_DICT);
    }
  }
  return s;
}

function planStream(
  syms: Uint8Array,
  ctxs: Uint8Array,
  count: number,
  lengths: Uint8Array,
  alphabetSize: number,
  rawBits: number,
  extraBitsTotal: number
): StreamPlan {
  let huffmanBits = 0;
  let huffmanUsable = true;
  for (let i = 0; i < count; i++) {
    const length = lengths[ctxs[i]! * alphabetSize + syms[i]!]!;
    if (length === 0) {
      huffmanUsable = false;
      break;
    }
    huffmanBits += length;
  }
  const rawTotal = count * rawBits + extraBitsTotal;
  if (huffmanUsable && huffmanBits + extraBitsTotal <= rawTotal) {
    return { huffman: true, bitLength: huffmanBits + extraBitsTotal };
  }
  return { huffman: false, bitLength: rawTotal };
}

/** Prices the complete `small` body for a token list without emitting anything. */
export function planSmallBody(tokens: Token[], bytes: Uint8Array, language: RegisteredLanguage): SmallPlan {
  const tables = language.tables;
  const collected = collectStreams(tokens, bytes, tables);
  const literals = planStream(
    collected.literalBytes,
    collected.literalCtxs,
    collected.literalCount,
    tables.literal,
    256,
    RAW_LITERAL_BITS,
    0
  );
  const tokenStream = planStream(
    collected.tokenSyms,
    collected.tokenCtxs,
    collected.tokenCount,
    tables.token,
    TOKEN_ALPHABET_SIZE,
    RAW_TOKEN_BITS,
    collected.tokenExtraTotal
  );
  const offsets = planStream(
    collected.offsetSlots,
    collected.offsetCtxs,
    collected.offsetCount,
    tables.offset,
    OFFSET_SLOT_COUNT,
    RAW_OFFSET_SLOT_BITS,
    collected.offsetExtraTotal
  );
  const tokenCount = collected.tokenCount;
  const totalBits =
    3 +
    bitVarintLength(tokenCount) +
    bitVarintLength(literals.bitLength) +
    bitVarintLength(tokenStream.bitLength) +
    literals.bitLength +
    tokenStream.bitLength +
    offsets.bitLength;
  return {
    literals,
    tokenStream,
    offsets,
    tokenCount,
    totalBits,
    charCost: Math.ceil(totalBits / 32) * 5,
    collected,
  };
}

interface ContextEncoders {
  literal: HuffmanEncoder[];
  token: HuffmanEncoder[];
  offset: HuffmanEncoder[];
}

const encoderCache = new WeakMap<EntropyTables, ContextEncoders>();

function buildEncoders(lengths: Uint8Array, alphabetSize: number): HuffmanEncoder[] {
  const encoders: HuffmanEncoder[] = [];
  for (let base = 0; base < lengths.length; base += alphabetSize) {
    encoders.push(buildEncoder(lengths.subarray(base, base + alphabetSize)));
  }
  return encoders;
}

function encodersFor(tables: EntropyTables): ContextEncoders {
  let encoders = encoderCache.get(tables);
  if (!encoders) {
    encoders = {
      literal: buildEncoders(tables.literal, 256),
      token: buildEncoders(tables.token, TOKEN_ALPHABET_SIZE),
      offset: buildEncoders(tables.offset, OFFSET_SLOT_COUNT),
    };
    encoderCache.set(tables, encoders);
  }
  return encoders;
}

/** Serializes a planned `small` body through the fused radix-85 writer (single pass). */
export function emitSmallBody(plan: SmallPlan, language: RegisteredLanguage): string {
  const collected = plan.collected;
  const encoders = encodersFor(language.tables);
  const writer = new BitWriter();
  writer.writeBits(
    (plan.literals.huffman ? 4 : 0) | (plan.tokenStream.huffman ? 2 : 0) | (plan.offsets.huffman ? 1 : 0),
    3
  );
  writer.writeVarint(plan.tokenCount);
  writer.writeVarint(plan.literals.bitLength);
  writer.writeVarint(plan.tokenStream.bitLength);

  if (plan.literals.huffman) {
    const literalEncoders = encoders.literal;
    for (let i = 0; i < collected.literalCount; i++) {
      const { codes, lengths } = literalEncoders[collected.literalCtxs[i]!]!;
      const byte = collected.literalBytes[i]!;
      writer.writeBits(codes[byte]!, lengths[byte]!);
    }
  } else {
    for (let i = 0; i < collected.literalCount; i++) writer.writeBits(collected.literalBytes[i]!, RAW_LITERAL_BITS);
  }
  {
    const tokenEncoders = encoders.token;
    const huffman = plan.tokenStream.huffman;
    for (let i = 0; i < collected.tokenCount; i++) {
      const symbol = collected.tokenSyms[i]!;
      if (huffman) {
        const { codes, lengths } = tokenEncoders[collected.tokenCtxs[i]!]!;
        writer.writeBits(codes[symbol]!, lengths[symbol]!);
      } else writer.writeBits(symbol, RAW_TOKEN_BITS);
      const extra = collected.tokenExtraBits[i]!;
      if (extra > 0) writer.writeBits(collected.tokenExtraValues[i]!, extra);
    }
  }
  {
    const offsetEncoders = encoders.offset;
    const huffman = plan.offsets.huffman;
    for (let i = 0; i < collected.offsetCount; i++) {
      const slot = collected.offsetSlots[i]!;
      if (huffman) {
        const { codes, lengths } = offsetEncoders[collected.offsetCtxs[i]!]!;
        writer.writeBits(codes[slot]!, lengths[slot]!);
      } else writer.writeBits(slot, RAW_OFFSET_SLOT_BITS);
      const extra = collected.offsetExtraBits[i]!;
      if (extra > 0) writer.writeBits(collected.offsetExtraValues[i]!, extra);
    }
  }
  return writer.toText();
}

function readSymbol(cursor: BitReader, table: Uint16Array): number {
  const entry = table[cursor.peekBits(MAX_CODE_LENGTH)]!;
  const length = entry & 15;
  if (length === 0) throw new TokzipDecodeError('invalid symbol');
  cursor.advance(length);
  return entry >>> 4;
}

/** Per-context single-lookup decode tables, built lazily per context and cached. */
class LazyDecoders {
  private readonly lengths: Uint8Array;
  private readonly alphabetSize: number;
  private readonly tables: (Uint16Array | undefined)[];

  constructor(lengths: Uint8Array, alphabetSize: number, contextCount: number) {
    this.lengths = lengths;
    this.alphabetSize = alphabetSize;
    this.tables = Array.from({ length: contextCount });
  }

  get(ctx: number): Uint16Array {
    return (this.tables[ctx] ??= buildDecoder(
      this.lengths.subarray(ctx * this.alphabetSize, (ctx + 1) * this.alphabetSize)
    ));
  }
}

interface ContextDecoders {
  literal: LazyDecoders;
  token: LazyDecoders;
  offset: LazyDecoders;
}

const decoderCache = new WeakMap<EntropyTables, ContextDecoders>();

function decodersFor(tables: EntropyTables): ContextDecoders {
  let decoders = decoderCache.get(tables);
  if (!decoders) {
    decoders = {
      literal: new LazyDecoders(tables.literal, 256, tables.litClassCount),
      token: new LazyDecoders(tables.token, TOKEN_ALPHABET_SIZE, TOKEN_CONTEXT_COUNT),
      offset: new LazyDecoders(tables.offset, OFFSET_SLOT_COUNT, OFFSET_CONTEXT_COUNT),
    };
    decoderCache.set(tables, decoders);
  }
  return decoders;
}

/** Decodes a `small` body in data[pos, end) into exactly `outputSize` bytes. */
export function decodeSmallBody(
  data: string,
  pos: number,
  end: number,
  outputSize: number,
  language: RegisteredLanguage
): Uint8Array {
  const words = decodeRadix85(data, pos, end);
  const header = new BitReader(words);
  const modes = header.readBits(3);
  const tokenCount = header.readVarint();
  const litBitLength = header.readVarint();
  const tokenBitLength = header.readVarint();

  const litStart = header.bitPosition;
  const tokenStart = litStart + litBitLength;
  const offsetStart = tokenStart + tokenBitLength;
  if (offsetStart > words.length * 32) throw new TokzipDecodeError('stream lengths exceed payload');
  const litCursor = new BitReader(words, litStart);
  const tokenCursor = new BitReader(words, tokenStart);
  const offsetCursor = new BitReader(words, offsetStart);

  const decoders = decodersFor(language.tables);
  const litHuffman = (modes & 4) !== 0;
  const tokenHuffman = (modes & 2) !== 0;
  const offsetHuffman = (modes & 1) !== 0;

  const readOffsetValue = (ctx: number): number => {
    const slot = offsetHuffman
      ? readSymbol(offsetCursor, decoders.offset.get(ctx))
      : offsetCursor.readBits(RAW_OFFSET_SLOT_BITS);
    if (slot >= OFFSET_SLOT_COUNT) throw new TokzipDecodeError('invalid symbol');
    const extraBits = extraBitsOf(slot);
    return valueOfSlot(slot, extraBits > 0 ? offsetCursor.readBits(extraBits) : 0);
  };

  const out = new Uint8Array(outputSize);
  const { dictionary } = language;
  const { litContext } = language.tables;
  let rep0 = INITIAL_REPS[0]!;
  let rep1 = INITIAL_REPS[1]!;
  let rep2 = INITIAL_REPS[2]!;
  let rep3 = INITIAL_REPS[3]!;
  let produced = 0;
  let prevKind = TOKEN_KIND_LITRUN;
  let prevByte = 0;
  for (let t = 0; t < tokenCount; t++) {
    const symbol = tokenHuffman
      ? readSymbol(tokenCursor, decoders.token.get(prevKind))
      : tokenCursor.readBits(RAW_TOKEN_BITS);
    if (symbol >= TOKEN_ALPHABET_SIZE) throw new TokzipDecodeError('invalid symbol');
    const kind = SYMBOL_KIND[symbol]!;
    const slot = SYMBOL_SLOT[symbol]!;
    const extraBits = extraBitsOf(slot);
    const slotValue = valueOfSlot(slot, extraBits > 0 ? tokenCursor.readBits(extraBits) : 0);
    prevKind = kind;

    if (kind === TOKEN_KIND_LITRUN) {
      const length = slotValue + 1;
      if (produced + length > outputSize) throw new TokzipDecodeError('declared size exceeded');
      const runEnd = produced + length;
      if (litHuffman) {
        const litDecoders = decoders.literal;
        for (let i = produced; i < runEnd; i++) {
          const byte = readSymbol(litCursor, litDecoders.get(litContext[prevByte]!));
          out[i] = byte;
          prevByte = byte;
        }
      } else {
        for (let i = produced; i < runEnd; i++) out[i] = litCursor.readBits(RAW_LITERAL_BITS);
        prevByte = out[runEnd - 1]!;
      }
      produced = runEnd;
      continue;
    }
    const length = slotValue + MIN_LEN_REP;
    if (produced + length > outputSize) throw new TokzipDecodeError('declared size exceeded');
    if (kind === TOKEN_KIND_DICT) {
      const start = readOffsetValue(OFFSET_CONTEXT_DICT);
      if (start + length > dictionary.length) throw new TokzipDecodeError('dictionary match out of bounds');
      out.set(dictionary.subarray(start, start + length), produced);
    } else {
      let dist: number;
      if (kind === TOKEN_KIND_HISTORY) {
        dist = readOffsetValue(OFFSET_CONTEXT_HISTORY) + 1;
        rep3 = rep2;
        rep2 = rep1;
        rep1 = rep0;
        rep0 = dist;
      } else {
        const repIndex = kind - TOKEN_KIND_REP0;
        if (repIndex === 0) dist = rep0;
        else if (repIndex === 1) {
          dist = rep1;
          rep1 = rep0;
          rep0 = dist;
        } else if (repIndex === 2) {
          dist = rep2;
          rep2 = rep1;
          rep1 = rep0;
          rep0 = dist;
        } else {
          dist = rep3;
          rep3 = rep2;
          rep2 = rep1;
          rep1 = rep0;
          rep0 = dist;
        }
      }
      if (dist < 1 || dist > produced) throw new TokzipDecodeError('history match out of bounds');
      if (dist >= length) {
        // Non-overlapping: block copy.
        out.copyWithin(produced, produced - dist, produced - dist + length);
      } else {
        const from = produced - dist;
        for (let i = 0; i < length; i++) out[produced + i] = out[from + i]!;
      }
    }
    produced += length;
    prevByte = out[produced - 1]!;
  }
  if (produced !== outputSize) throw new TokzipDecodeError('declared size mismatch');
  if (litCursor.bitPosition !== tokenStart || tokenCursor.bitPosition !== offsetStart) {
    throw new TokzipDecodeError('stream length mismatch');
  }
  // Trailing characters are a structural error: only the final word's zero padding may remain.
  if (Math.ceil(Math.max(offsetCursor.bitPosition, 1) / 32) !== words.length) {
    throw new TokzipDecodeError('trailing characters after payload');
  }
  // The padding itself must be zero (canonical frames; catches tail corruption).
  for (let remaining = words.length * 32 - offsetCursor.bitPosition; remaining > 0;) {
    const take = Math.min(24, remaining);
    if (offsetCursor.readBits(take) !== 0) throw new TokzipDecodeError('non-zero padding bits');
    remaining -= take;
  }
  return out;
}
