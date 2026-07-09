import type { EntropyTables, RegisteredLanguage } from './dictionary.ts';
import { TokzipDecodeError } from './errors.ts';
import {
  INITIAL_REPS,
  MIN_LEN_REP,
  RAW_LITERAL_BITS,
  RAW_OFFSET_SLOT_BITS,
  RAW_TOKEN_BITS,
  SMALL_WINDOW,
  TOKEN_ALPHABET_SIZE,
  TOKEN_KIND_DICT,
  TOKEN_KIND_HISTORY,
  TOKEN_KIND_LITRUN,
  TOKEN_KIND_REP0,
} from './format.ts';
import { buildDecoder, buildEncoder, MAX_CODE_LENGTH } from './huffman.ts';
import type { ParsePricing, Token } from './lz.ts';
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

function tokenSymbol(kind: number, lenSlot: number): number {
  return kind * LENGTH_SLOT_COUNT + lenSlot;
}

function bitVarintLength(value: number): number {
  let length = 8;
  for (let rest = Math.floor(value / 128); rest > 0; rest = Math.floor(rest / 128)) length += 8;
  return length;
}

/**
 * Builds the `small`-mode pricing model: exact output-bit prices from the static per-language
 * tables (this is what lets the bounded lazy parser make true cost-based decisions).
 */
export function smallPricing(bytes: Uint8Array, language: RegisteredLanguage): ParsePricing {
  const { literal, token, offset } = language.tables;
  const litBits = (byte: number): number => literal[byte]! || RAW_LITERAL_BITS;
  const tokenBits = (symbol: number): number => token[symbol]! || RAW_TOKEN_BITS;
  const offsetBits = (slot: number): number => offset[slot]! || RAW_OFFSET_SLOT_BITS;

  const litCostPrefix = new Float64Array(bytes.length + 1);
  for (let i = 0; i < bytes.length; i++) litCostPrefix[i + 1] = litCostPrefix[i]! + litBits(bytes[i]!);

  const matchTokenBits = (kind: number, len: number): number => {
    const slot = slotOf(len - MIN_LEN_REP);
    return tokenBits(tokenSymbol(kind, slot)) + extraBitsOf(slot);
  };
  return {
    litCostPrefix,
    repCost: (repIndex, len) => matchTokenBits(TOKEN_KIND_REP0 + repIndex, len),
    historyCost: (dist, len) => {
      const offSlot = slotOf(dist - 1);
      return matchTokenBits(TOKEN_KIND_HISTORY, len) + offsetBits(offSlot) + extraBitsOf(offSlot);
    },
    dictCost: (start, len) => {
      const offSlot = slotOf(start);
      return matchTokenBits(TOKEN_KIND_DICT, len) + offsetBits(offSlot) + extraBitsOf(offSlot);
    },
    lazy: true,
    window: SMALL_WINDOW,
    maxDictStart: SMALL_WINDOW,
  };
}

interface StreamPlan {
  huffman: boolean;
  bitLength: number;
}

/** A fully priced `small` body: sizes are exact, so the frame comparison never needs emission. */
export interface SmallPlan {
  tokens: Token[];
  literals: StreamPlan;
  tokenStream: StreamPlan;
  offsets: StreamPlan;
  tokenCount: number;
  totalBits: number;
  /** Exact char count of the emitted radix-85 body. */
  charCost: number;
}

interface CollectedStreams {
  literalBytes: number[];
  tokenSyms: number[];
  tokenExtraBits: number[];
  tokenExtraValues: number[];
  offsetSlots: number[];
  offsetExtraBits: number[];
  offsetExtraValues: number[];
}

function collectStreams(tokens: Token[], bytes: Uint8Array): CollectedStreams {
  const collected: CollectedStreams = {
    literalBytes: [],
    tokenSyms: [],
    tokenExtraBits: [],
    tokenExtraValues: [],
    offsetSlots: [],
    offsetExtraBits: [],
    offsetExtraValues: [],
  };
  const pushToken = (kind: number, lenValue: number): void => {
    const slot = slotOf(lenValue);
    collected.tokenSyms.push(tokenSymbol(kind, slot));
    collected.tokenExtraBits.push(extraBitsOf(slot));
    collected.tokenExtraValues.push(extraValueOf(lenValue, slot));
  };
  const pushOffset = (value: number): void => {
    const slot = slotOf(value);
    collected.offsetSlots.push(slot);
    collected.offsetExtraBits.push(extraBitsOf(slot));
    collected.offsetExtraValues.push(extraValueOf(value, slot));
  };
  const maxRunLength = maxSlotValue(LENGTH_SLOT_COUNT) + 1;
  for (const token of tokens) {
    if (token.type === 'lit') {
      // Runs beyond the length-slot alphabet are split into consecutive litrun tokens.
      for (let start = token.start; start < token.end; start += maxRunLength) {
        const end = Math.min(start + maxRunLength, token.end);
        pushToken(TOKEN_KIND_LITRUN, end - start - 1);
        for (let i = start; i < end; i++) collected.literalBytes.push(bytes[i]!);
      }
    } else if (token.type === 'history') {
      if (token.rep >= 0) pushToken(TOKEN_KIND_REP0 + token.rep, token.len - MIN_LEN_REP);
      else {
        pushToken(TOKEN_KIND_HISTORY, token.len - MIN_LEN_REP);
        pushOffset(token.dist - 1);
      }
    } else {
      pushToken(TOKEN_KIND_DICT, token.len - MIN_LEN_REP);
      pushOffset(token.start);
    }
  }
  return collected;
}

function planStream(symbols: number[], lengths: Uint8Array, rawBits: number, extraBitsTotal: number): StreamPlan {
  let huffmanBits = 0;
  let huffmanUsable = true;
  for (const symbol of symbols) {
    const length = lengths[symbol]!;
    if (length === 0) {
      huffmanUsable = false;
      break;
    }
    huffmanBits += length;
  }
  const rawTotal = symbols.length * rawBits + extraBitsTotal;
  if (huffmanUsable && huffmanBits + extraBitsTotal <= rawTotal) {
    return { huffman: true, bitLength: huffmanBits + extraBitsTotal };
  }
  return { huffman: false, bitLength: rawTotal };
}

const sum = (values: number[]): number => values.reduce((a, b) => a + b, 0);

/** Prices the complete `small` body for a token list without emitting anything. */
export function planSmallBody(tokens: Token[], bytes: Uint8Array, language: RegisteredLanguage): SmallPlan {
  const collected = collectStreams(tokens, bytes);
  const tables = language.tables;
  const literals = planStream(collected.literalBytes, tables.literal, RAW_LITERAL_BITS, 0);
  const tokenStream = planStream(collected.tokenSyms, tables.token, RAW_TOKEN_BITS, sum(collected.tokenExtraBits));
  const offsets = planStream(
    collected.offsetSlots,
    tables.offset,
    RAW_OFFSET_SLOT_BITS,
    sum(collected.offsetExtraBits)
  );
  const tokenCount = collected.tokenSyms.length;
  const totalBits =
    3 +
    bitVarintLength(tokenCount) +
    bitVarintLength(literals.bitLength) +
    bitVarintLength(tokenStream.bitLength) +
    literals.bitLength +
    tokenStream.bitLength +
    offsets.bitLength;
  return { tokens, literals, tokenStream, offsets, tokenCount, totalBits, charCost: Math.ceil(totalBits / 32) * 5 };
}

/** Serializes a planned `small` body through the fused radix-85 writer (single pass). */
export function emitSmallBody(plan: SmallPlan, bytes: Uint8Array, language: RegisteredLanguage): string {
  const collected = collectStreams(plan.tokens, bytes);
  const tables = language.tables;
  const writer = new BitWriter();
  writer.writeBits(
    (plan.literals.huffman ? 4 : 0) | (plan.tokenStream.huffman ? 2 : 0) | (plan.offsets.huffman ? 1 : 0),
    3
  );
  writer.writeVarint(plan.tokenCount);
  writer.writeVarint(plan.literals.bitLength);
  writer.writeVarint(plan.tokenStream.bitLength);

  const litEncoder = plan.literals.huffman ? buildEncoder(tables.literal) : undefined;
  for (const byte of collected.literalBytes) {
    if (litEncoder) writer.writeBits(litEncoder.codes[byte]!, litEncoder.lengths[byte]!);
    else writer.writeBits(byte, RAW_LITERAL_BITS);
  }
  const tokenEncoder = plan.tokenStream.huffman ? buildEncoder(tables.token) : undefined;
  for (let i = 0; i < collected.tokenSyms.length; i++) {
    const symbol = collected.tokenSyms[i]!;
    if (tokenEncoder) writer.writeBits(tokenEncoder.codes[symbol]!, tokenEncoder.lengths[symbol]!);
    else writer.writeBits(symbol, RAW_TOKEN_BITS);
    if (collected.tokenExtraBits[i]! > 0)
      writer.writeBits(collected.tokenExtraValues[i]!, collected.tokenExtraBits[i]!);
  }
  const offsetEncoder = plan.offsets.huffman ? buildEncoder(tables.offset) : undefined;
  for (let i = 0; i < collected.offsetSlots.length; i++) {
    const slot = collected.offsetSlots[i]!;
    if (offsetEncoder) writer.writeBits(offsetEncoder.codes[slot]!, offsetEncoder.lengths[slot]!);
    else writer.writeBits(slot, RAW_OFFSET_SLOT_BITS);
    if (collected.offsetExtraBits[i]! > 0)
      writer.writeBits(collected.offsetExtraValues[i]!, collected.offsetExtraBits[i]!);
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

const decoderCache = new WeakMap<EntropyTables, { literal: Uint16Array; token: Uint16Array; offset: Uint16Array }>();

function decodersFor(tables: EntropyTables): { literal: Uint16Array; token: Uint16Array; offset: Uint16Array } {
  let decoders = decoderCache.get(tables);
  if (!decoders) {
    decoders = {
      literal: buildDecoder(tables.literal),
      token: buildDecoder(tables.token),
      offset: buildDecoder(tables.offset),
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

  const readLiteral = (): number =>
    litHuffman ? readSymbol(litCursor, decoders.literal) : litCursor.readBits(RAW_LITERAL_BITS);
  const readToken = (): number => {
    const symbol = tokenHuffman ? readSymbol(tokenCursor, decoders.token) : tokenCursor.readBits(RAW_TOKEN_BITS);
    if (symbol >= TOKEN_ALPHABET_SIZE) throw new TokzipDecodeError('invalid symbol');
    return symbol;
  };
  const readOffsetValue = (): number => {
    const slot = offsetHuffman
      ? readSymbol(offsetCursor, decoders.offset)
      : offsetCursor.readBits(RAW_OFFSET_SLOT_BITS);
    if (slot >= OFFSET_SLOT_COUNT) throw new TokzipDecodeError('invalid symbol');
    const extraBits = extraBitsOf(slot);
    return valueOfSlot(slot, extraBits > 0 ? offsetCursor.readBits(extraBits) : 0);
  };

  const out = new Uint8Array(outputSize);
  const { dictionary } = language;
  const reps = [...INITIAL_REPS];
  let produced = 0;
  for (let t = 0; t < tokenCount; t++) {
    const symbol = readToken();
    const kind = Math.floor(symbol / LENGTH_SLOT_COUNT);
    const slot = symbol % LENGTH_SLOT_COUNT;
    const extraBits = extraBitsOf(slot);
    const slotValue = valueOfSlot(slot, extraBits > 0 ? tokenCursor.readBits(extraBits) : 0);

    if (kind === TOKEN_KIND_LITRUN) {
      const length = slotValue + 1;
      if (produced + length > outputSize) throw new TokzipDecodeError('declared size exceeded');
      for (let i = 0; i < length; i++) out[produced + i] = readLiteral();
      produced += length;
      continue;
    }
    const length = slotValue + MIN_LEN_REP;
    if (produced + length > outputSize) throw new TokzipDecodeError('declared size exceeded');
    if (kind === TOKEN_KIND_DICT) {
      const start = readOffsetValue();
      if (start + length > dictionary.length) throw new TokzipDecodeError('dictionary match out of bounds');
      out.set(dictionary.subarray(start, start + length), produced);
    } else {
      let dist: number;
      if (kind === TOKEN_KIND_HISTORY) {
        dist = readOffsetValue() + 1;
        reps.pop();
        reps.unshift(dist);
      } else {
        const repIndex = kind - TOKEN_KIND_REP0;
        if (repIndex < 0 || repIndex > 3) throw new TokzipDecodeError('invalid symbol');
        dist = reps[repIndex]!;
        reps.splice(repIndex, 1);
        reps.unshift(dist);
      }
      if (dist < 1 || dist > produced) throw new TokzipDecodeError('history match out of bounds');
      for (let i = 0; i < length; i++) out[produced + i] = out[produced - dist + i]!;
    }
    produced += length;
  }
  if (produced !== outputSize) throw new TokzipDecodeError('declared size mismatch');
  if (litCursor.bitPosition !== tokenStart || tokenCursor.bitPosition !== offsetStart) {
    throw new TokzipDecodeError('stream length mismatch');
  }
  // Trailing characters are a structural error: only the final word's zero padding may remain.
  if (Math.ceil(Math.max(offsetCursor.bitPosition, 1) / 32) !== words.length) {
    throw new TokzipDecodeError('trailing characters after payload');
  }
  return out;
}
