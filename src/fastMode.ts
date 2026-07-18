import type { RegisteredLanguage } from './dictionary.ts';
import { allocateDecodeBuffer, TokzipDecodeError } from './errors.ts';
import { copyExtendedDictMatch, FenceTracker } from './fences.ts';
import {
  FAST_WINDOW,
  INITIAL_REPS,
  MATCH_LEN_CAP,
  KIND_DICT,
  KIND_HISTORY,
  KIND_LIT64,
  KIND_LITRAW,
  KIND_REP0,
  MIN_LEN_EXPLICIT,
  MIN_LEN_REP,
  SHORT_OFFSET_LIMIT,
} from './format.ts';
import type { ParsePricing, Token } from './lz.ts';
import {
  asciiCodeAt,
  packedRawLength,
  pushPackedRaw,
  pushVarint64,
  RADIX64_CODES,
  RADIX64_VALUES,
  readPackedRaw,
  readRadix64,
  readVarint64,
  TextSink,
  varint64Length,
} from './radix64.ts';

/** Length-coding bases: payload values at which a varint extension follows the tag. */
const LIT_EXT_LEN = 8; // Literal runs: payload 0–6 → len 1–7; 7 → len 8 + varint.
const EXPLICIT_EXT_LEN = MIN_LEN_EXPLICIT + 3; // Matches: lencode 0–2 → len 4–6; 3 → len 7 + varint.
const REP_EXT_LEN = MIN_LEN_REP + 7; // Reps: payload 0–6 → len 2–8; 7 → len 9 + varint.

/** Exact char cost of one `fast` match token (tag + offset field + length extension). */
function matchCharCost(kind: 'history' | 'dict' | 'rep', value: number, len: number): number {
  if (kind === 'rep') return 1 + (len >= REP_EXT_LEN ? varint64Length(len - REP_EXT_LEN) : 0);
  const offsetChars = value < SHORT_OFFSET_LIMIT ? 2 : 3;
  return 1 + offsetChars + (len >= EXPLICIT_EXT_LEN ? varint64Length(len - EXPLICIT_EXT_LEN) : 0);
}

// Scratch prefix buffer reused across calls (compress is synchronous; each pricing's prefix
// is only read while its own parse runs, and smallMode keeps a separate scratch).
let fastPrefixScratch = new Float64Array(0);

/** Builds the `fast`-mode pricing model for the shared LZ parser. */
export function fastPricing(bytes: Uint8Array, language: RegisteredLanguage): ParsePricing {
  const { top64Index } = language;
  if (fastPrefixScratch.length < bytes.length + 1) {
    fastPrefixScratch = new Float64Array(Math.max(bytes.length + 1, fastPrefixScratch.length * 2, 4096));
  }
  const litCostPrefix = fastPrefixScratch;
  // In-charset literals cost exactly 1 char; others amortize raw bit-packing (3 bytes → 4 chars).
  // The running sum stays in a local so the loop carries no load-after-store dependency.
  let acc = 0;
  for (let i = 0; i < bytes.length; i++) {
    acc += top64Index[bytes[i]!]! >= 0 ? 1 : 4 / 3;
    litCostPrefix[i + 1] = acc;
  }
  return {
    litCostPrefix,
    repCost: (_r, len) => matchCharCost('rep', 0, len),
    historyCost: (dist, len) => matchCharCost('history', dist - 1, len),
    dictCost: (start, len) => matchCharCost('dict', start, len),
    // Encoder-side policy (format-compatible): the bounded price-aware lazy step benches
    // 1–2% smaller for a modest compression-speed cost, which the storage workload
    // (KB-scale documents compressed client-side) happily pays.
    lazy: true,
    window: FAST_WINDOW,
    maxDictStart: FAST_WINDOW,
  };
}

interface LiteralSegment {
  raw: boolean;
  start: number;
  end: number;
}

/**
 * Splits a literal run between the two literal kinds. Encoder policy (not normative): short
 * in-charset islands are absorbed into surrounding raw runs to save switch tags.
 */
function segmentLiterals(bytes: Uint8Array, start: number, end: number, top64Index: Int8Array): LiteralSegment[] {
  const runs: LiteralSegment[] = [];
  let runStart = start;
  let runRaw = top64Index[bytes[start]!]! < 0;
  for (let i = start + 1; i < end; i++) {
    const raw = top64Index[bytes[i]!]! < 0;
    if (raw !== runRaw) {
      runs.push({ raw: runRaw, start: runStart, end: i });
      runStart = i;
      runRaw = raw;
    }
  }
  runs.push({ raw: runRaw, start: runStart, end });
  if (runs.length === 1) return runs;

  const segments: LiteralSegment[] = [];
  const pushRaw = (segStart: number, segEnd: number): void => {
    const last = segments.at(-1);
    if (last?.raw && last.end === segStart) last.end = segEnd;
    else segments.push({ raw: true, start: segStart, end: segEnd });
  };
  for (let r = 0; r < runs.length; r++) {
    const run = runs[r]!;
    if (run.raw) {
      pushRaw(run.start, run.end);
      continue;
    }
    const length = run.end - run.start;
    const isEdge = r === 0 || r === runs.length - 1;
    if (length >= (isEdge ? 3 : 7)) segments.push(run);
    else pushRaw(run.start, run.end);
  }
  return segments;
}

function literalRunCost(raw: boolean, length: number): number {
  const body = raw ? packedRawLength(length) : length;
  return 1 + body + (length >= LIT_EXT_LEN ? varint64Length(length - LIT_EXT_LEN) : 0);
}

/**
 * Exact char cost of serializing `tokens` in `fast` mode — used by the analytic auto-downgrade
 * (no emission) and kept in lockstep with {@link encodeFastBody}. Returns undefined when a token
 * exceeds `fast`'s representable ranges (the frame comparison must then skip the fast candidate).
 */
export function fastBodyCost(tokens: Token[], bytes: Uint8Array, language: RegisteredLanguage): number | undefined {
  let cost = 0;
  for (const token of tokens) {
    if (token.type === 'lit') {
      for (const segment of segmentLiterals(bytes, token.start, token.end, language.top64Index)) {
        cost += literalRunCost(segment.raw, segment.end - segment.start);
      }
    } else if (token.type === 'history') {
      if (token.rep >= 0) cost += matchCharCost('rep', 0, token.len);
      else if (token.dist > FAST_WINDOW || token.len < MIN_LEN_EXPLICIT) return undefined;
      else cost += matchCharCost('history', token.dist - 1, token.len);
    } else {
      if (token.start >= FAST_WINDOW || token.len < MIN_LEN_EXPLICIT) return undefined;
      cost += matchCharCost('dict', token.start, token.len);
    }
  }
  return cost;
}

function pushTag(out: TextSink, kind: number, payload: number): void {
  out.push(RADIX64_CODES[(kind << 3) | payload]!);
}

function pushOffset(out: TextSink, value: number): void {
  if (value < SHORT_OFFSET_LIMIT) {
    out.push(RADIX64_CODES[(value >>> 6) & 63]!);
    out.push(RADIX64_CODES[value & 63]!);
  } else {
    out.push(RADIX64_CODES[(value >>> 12) & 63]!);
    out.push(RADIX64_CODES[(value >>> 6) & 63]!);
    out.push(RADIX64_CODES[value & 63]!);
  }
}

function pushLiteralSegment(out: TextSink, bytes: Uint8Array, segment: LiteralSegment, top64Index: Int8Array): void {
  const length = segment.end - segment.start;
  pushTag(out, segment.raw ? KIND_LITRAW : KIND_LIT64, Math.min(length - 1, 7));
  // The length varint precedes the body so decoding stays single-pass (body size depends on it).
  if (length >= LIT_EXT_LEN) pushVarint64(out, length - LIT_EXT_LEN);
  if (segment.raw) pushPackedRaw(out, bytes, segment.start, segment.end);
  else {
    const buffer = out.reserve(length);
    let at = out.length;
    for (let i = segment.start; i < segment.end; i++) buffer[at++] = RADIX64_CODES[top64Index[bytes[i]!]!]!;
    out.length = at;
  }
}

/** Serializes the token list into `out` as the `fast`-mode radix-64 stream (tag, offset field, length varint). */
export function emitFastBody(out: TextSink, tokens: Token[], bytes: Uint8Array, language: RegisteredLanguage): void {
  for (const token of tokens) {
    if (token.type === 'lit') {
      for (const segment of segmentLiterals(bytes, token.start, token.end, language.top64Index)) {
        pushLiteralSegment(out, bytes, segment, language.top64Index);
      }
    } else if (token.type === 'history' && token.rep >= 0) {
      pushTag(out, KIND_REP0 + token.rep, Math.min(token.len - MIN_LEN_REP, 7));
      if (token.len >= REP_EXT_LEN) pushVarint64(out, token.len - REP_EXT_LEN);
    } else {
      const value = token.type === 'history' ? token.dist - 1 : token.start;
      const width = value < SHORT_OFFSET_LIMIT ? 0 : 1;
      pushTag(
        out,
        token.type === 'history' ? KIND_HISTORY : KIND_DICT,
        (width << 2) | Math.min(token.len - MIN_LEN_EXPLICIT, 3)
      );
      pushOffset(out, value);
      if (token.len >= EXPLICIT_EXT_LEN) pushVarint64(out, token.len - EXPLICIT_EXT_LEN);
    }
  }
}

/** Serializes the token list as the `fast`-mode radix-64 stream (string form; see {@link emitFastBody}). */
export function encodeFastBody(tokens: Token[], bytes: Uint8Array, language: RegisteredLanguage): string {
  const out = new TextSink(bytes.length + 64);
  emitFastBody(out, tokens, bytes, language);
  return out.toString();
}

/**
 * Decodes a text-frame `fast` body in data[pos, end) into exactly `outputSize` bytes, throwing
 * {@link TokzipDecodeError} on any structural violation.
 */
export function decodeFastBody(
  data: string,
  pos: number,
  end: number,
  outputSize: number,
  language: RegisteredLanguage,
  fenced = false
): Uint8Array {
  const result = decodeFastBodyCore(data, pos, end, outputSize, language, fenced);
  if (result.pos !== end) throw new TokzipDecodeError('trailing characters after payload');
  return result.out;
}

const asciiDecoder = new TextDecoder();

/**
 * Decodes a binary-frame `fast` body in body[pos, end) into exactly `outputSize` bytes.
 * The body is the radix-64 char stream bit-packed at 6 bits per char, MSB-first, zero-padded
 * to a byte boundary; the byte length must be exactly ceil(6·chars/8) (canonical framing).
 */
export function decodeFastBodyBinary(
  body: Uint8Array,
  pos: number,
  end: number,
  outputSize: number,
  language: RegisteredLanguage,
  fenced = false,
  history?: Uint8Array
): Uint8Array {
  const byteLength = end - pos;
  const charCount = Math.floor((byteLength * 8) / 6);
  const codes = new Uint8Array(charCount);
  let acc = 0;
  let accBits = 0;
  let at = 0;
  for (let i = pos; i < end; i++) {
    acc = (acc << 8) | body[i]!;
    accBits += 8;
    while (accBits >= 6) {
      accBits -= 6;
      codes[at++] = RADIX64_CODES[(acc >>> accBits) & 63]!;
    }
    acc &= (1 << accBits) - 1;
  }
  const result = decodeFastBodyCore(asciiDecoder.decode(codes), 0, charCount, outputSize, language, fenced, history);
  // Canonical framing: exactly the consumed chars' bytes, with zero padding bits (at most 7,
  // so the whole padding sits in the final body byte).
  if (Math.ceil((result.pos * 6) / 8) !== byteLength) throw new TokzipDecodeError('trailing characters after payload');
  const padBits = byteLength * 8 - result.pos * 6;
  if (padBits > 0 && (body[end - 1]! & ((1 << padBits) - 1)) !== 0) {
    throw new TokzipDecodeError('non-zero padding bits');
  }
  return result.out;
}

/** Bit-packs an emitted `fast` radix-64 char-code stream into the binary-frame body bytes. */
export function packFastCodes(codes: Uint8Array, count: number): Uint8Array {
  // Math.ceil, not (x + 7) >> 3: count * 6 can exceed 2^31, where 32-bit ops truncate.
  const out = new Uint8Array(Math.ceil((count * 6) / 8));
  const values = RADIX64_VALUES;
  let acc = 0;
  let accBits = 0;
  let at = 0;
  for (let i = 0; i < count; i++) {
    acc = (acc << 6) | values[codes[i]!]!;
    accBits += 6;
    if (accBits >= 8) {
      accBits -= 8;
      out[at++] = (acc >>> accBits) & 255;
    }
  }
  if (accBits > 0) out[at] = (acc << (8 - accBits)) & 255;
  return out;
}

/**
 * Shared `fast` decode loop; returns the output and the position after the consumed chars.
 * A `history` prefix (streaming blocks) is seeded as already-produced output: history matches
 * may reach into it, and only the newly produced `outputSize` bytes are returned.
 */
function decodeFastBodyCore(
  data: string,
  pos: number,
  end: number,
  outputSize: number,
  language: RegisteredLanguage,
  fenced: boolean,
  history?: Uint8Array
): { out: Uint8Array; pos: number } {
  // Structural output bound, checked before allocating: every token consumes at least one
  // char and produces at most MATCH_LEN_CAP bytes, so a declared size beyond that cannot be
  // produced by this body (this also stops forged huge-size frames from forcing enormous
  // allocations under `maxOutputSize: Infinity`).
  if (outputSize > (end - pos) * MATCH_LEN_CAP) {
    throw new TokzipDecodeError('declared size exceeds body capacity');
  }
  const historyLength = history?.length ?? 0;
  const target = historyLength + outputSize;
  const out = allocateDecodeBuffer(target);
  if (history) out.set(history);
  const { dictionary, top64 } = language;
  const tracker = fenced ? new FenceTracker(language.id) : undefined;
  let rep0 = INITIAL_REPS[0]!;
  let rep1 = INITIAL_REPS[1]!;
  let rep2 = INITIAL_REPS[2]!;
  let rep3 = INITIAL_REPS[3]!;
  let produced = historyLength;

  const readOffset = (width: number): number => {
    let value = (readRadix64(data, pos) << 6) | readRadix64(data, pos + 1);
    pos += 2;
    if (width === 1) {
      value = ((value << 6) | readRadix64(data, pos)) & (FAST_WINDOW - 1);
      pos++;
    }
    return value;
  };

  while (produced < target) {
    if (pos >= end) throw new TokzipDecodeError('truncated token stream');
    const tag = readRadix64(data, pos++);
    const kind = tag >>> 3;
    const payload = tag & 7;
    if (kind === KIND_LIT64 || kind === KIND_LITRAW) {
      let length = payload + 1;
      if (payload === 7) {
        // Extended run: the length varint precedes the body (the body size depends on it).
        const result = readVarint64(data, pos);
        length = LIT_EXT_LEN + result.value;
        pos = result.pos;
      }
      const bodyPos = pos;
      if (produced + length > target) throw new TokzipDecodeError('declared size exceeded');
      if (kind === KIND_LIT64) {
        if (bodyPos + length > end) throw new TokzipDecodeError('truncated literal run');
        const values = RADIX64_VALUES;
        for (let i = 0; i < length; i++) {
          const code = asciiCodeAt(data, bodyPos + i);
          const value = code < 128 ? values[code]! : -1;
          if (value < 0) throw new TokzipDecodeError(`non-alphabet character at position ${bodyPos + i}`);
          out[produced + i] = top64[value]!;
        }
        pos = bodyPos + length;
      } else {
        if (bodyPos + packedRawLength(length) > end) throw new TokzipDecodeError('truncated literal run');
        pos = readPackedRaw(data, bodyPos, out, produced, length);
      }
      produced += length;
      continue;
    }

    let length: number;
    let sourceIsDict = false;
    let dist = 0;
    let dictStart = 0;
    if (kind === KIND_HISTORY || kind === KIND_DICT) {
      const width = payload >>> 2;
      const lenCode = payload & 3;
      const value = readOffset(width);
      if (lenCode < 3) length = MIN_LEN_EXPLICIT + lenCode;
      else {
        const result = readVarint64(data, pos);
        length = EXPLICIT_EXT_LEN + result.value;
        pos = result.pos;
      }
      if (kind === KIND_HISTORY) {
        dist = value + 1;
        rep3 = rep2;
        rep2 = rep1;
        rep1 = rep0;
        rep0 = dist;
      } else {
        sourceIsDict = true;
        dictStart = value;
      }
    } else {
      const repIndex = kind - KIND_REP0;
      if (payload < 7) length = MIN_LEN_REP + payload;
      else {
        const result = readVarint64(data, pos);
        length = REP_EXT_LEN + result.value;
        pos = result.pos;
      }
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

    // Encoders MUST split longer matches (format-wide cap for cross-mode token compatibility;
    // the pre-allocation capacity bound also relies on it), so a longer one is structural.
    if (length > MATCH_LEN_CAP) throw new TokzipDecodeError('match length exceeds cap');
    if (produced + length > target) throw new TokzipDecodeError('declared size exceeded');
    if (sourceIsDict) {
      if (dictStart + length <= dictionary.length) {
        out.set(dictionary.subarray(dictStart, dictStart + length), produced);
      } else if (tracker) {
        copyExtendedDictMatch(out, produced, dictStart, length, language, tracker);
      } else {
        throw new TokzipDecodeError('dictionary match out of bounds');
      }
    } else {
      if (dist < 1 || dist > produced) throw new TokzipDecodeError('history match out of bounds');
      if (dist >= length) {
        out.copyWithin(produced, produced - dist, produced - dist + length);
      } else {
        // Overlap-copy: history matches may copy bytes produced by the same match.
        const from = produced - dist;
        for (let i = 0; i < length; i++) out[produced + i] = out[from + i]!;
      }
    }
    produced += length;
  }
  // slice, not subarray: a view would keep the whole history+output allocation alive for as
  // long as the caller retains the chunk, multiplying resident memory per decoded block.
  return { out: historyLength > 0 ? out.slice(historyLength) : out, pos };
}
