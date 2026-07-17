import { crc32 } from './checksum.ts';
import { languageByName, requireLanguageById, type RegisteredLanguage } from './dictionary.ts';
import { TokzipDecodeError } from './errors.ts';
import {
  decodeFastBody,
  decodeFastBodyBinary,
  emitFastBody,
  fastBodyCost,
  fastPricing,
  packFastCodes,
} from './fastMode.ts';
import { computeDictSegments, usesExtendedDictionary } from './fences.ts';
import {
  BINARY_MAGIC_VERSION,
  CRC_BINARY_BYTES,
  CRC_TEXT_CHARS,
  DEFAULT_MAX_OUTPUT_SIZE,
  FAST_WINDOW,
  FLAG_BYTES,
  FLAG_FENCED,
  MAGIC_VERSION,
  MODE_FAST,
  MODE_SMALL,
  MODE_STORED,
  RESERVED_FLAG_MASK,
  SMALL_WINDOW,
} from './format.ts';
import { dictIndexFor, parse, type Token } from './lz.ts';
import {
  packedRawLength,
  pushPackedRaw,
  pushVarint64,
  RADIX64_CODES,
  readPackedRaw,
  readRadix64,
  readVarint64,
  TextSink,
} from './radix64.ts';
import {
  decodeSmallBody,
  decodeSmallBodyBinary,
  emitSmallBody,
  planSmallBody,
  type SmallPlan,
  smallPricing,
} from './smallMode.ts';

export interface CompressOptions {
  /** Language dictionary to use; default 'none' (id 0, wrapper dictionary only). */
  language?: string;
  /** Optimization target; both modes are lossless. Default 'fast'. */
  mode?: 'fast' | 'small';
  /**
   * Output channel; default 'text'. 'text' emits a safe-ASCII frame (JSON- and
   * template-literal-safe); 'binary' emits the same streams packed at 8 bits per byte —
   * about 25% smaller for `fast` frames and 20% smaller for `small` frames.
   */
  output?: 'text' | 'binary';
}

export interface DecompressOptions {
  /** Refuses to allocate more than this many output bytes (default 64 MiB). */
  maxOutputSize?: number;
}

const textEncoder = new TextEncoder();
// Fatal decoding: invalid UTF-8 in a string-typed frame throws, never U+FFFD insertion.
// ignoreBOM keeps a leading U+FEFF as a character instead of eating it, so round-tripping a
// BOM-prefixed string returns it intact — decoding is lossless or it throws, never silently lossy.
const fatalDecoder = new TextDecoder('utf-8', { fatal: true, ignoreBOM: true });

// Binary-frame byte varints mirror the text container's radix-64 varints: little-endian
// 7-bit groups, continue bit 7, canonical (minimal) length, 5 groups = 35 bits max.
const BYTE_VARINT_MAX_BYTES = 5;

/** Compresses a string (UTF-8) or raw bytes into a safe-ASCII text frame. */
export function compress(input: string | Uint8Array, options?: CompressOptions & { output?: 'text' }): string;
/** Compresses a string (UTF-8) or raw bytes into a dense binary frame. */
export function compress(input: string | Uint8Array, options: CompressOptions & { output: 'binary' }): Uint8Array;
/** Fallback for options whose `output` is not statically known (e.g. a `CompressOptions` variable). */
export function compress(input: string | Uint8Array, options?: CompressOptions): string | Uint8Array;
export function compress(input: string | Uint8Array, options?: CompressOptions): string | Uint8Array {
  const isString = typeof input === 'string';
  const bytes = isString ? textEncoder.encode(input) : input;
  const languageName = options?.language ?? 'none';
  const language = languageByName(languageName);
  if (!language) throw new RangeError(`unregistered language: ${languageName}`);
  const mode = options?.mode ?? 'fast';
  // Untyped callers must not silently fall through to the (expensive) small path on typos.
  if (mode !== 'fast' && mode !== 'small') throw new RangeError(`invalid mode: ${String(mode)}`);
  const output = options?.output ?? 'text';
  if (output !== 'text' && output !== 'binary') throw new RangeError(`invalid output: ${String(output)}`);
  const binary = output === 'binary';

  // Fenced dictionary extension: labeled code fences extend the searchable dictionary space
  // with the block language's suffix (undefined when the input has no such fence, or when
  // the mode's offset bound cannot address the extension at all).
  const segments = computeDictSegments(bytes, language, mode === 'fast' ? FAST_WINDOW : SMALL_WINDOW);
  const dictIndex = dictIndexFor(language);

  // The auto-downgrade compares body costs in output units — chars for text frames, bytes
  // for binary frames (fast bodies pack 6 bits per char; small bodies byte-pad their bit
  // stream) — so each channel independently ships its smallest frame.
  // Math.ceil, not (x + 7) >> 3: bit counts can exceed 2^31 on large inputs, where 32-bit
  // bitwise ops silently truncate.
  const fastOutCost = (chars: number): number => (binary ? Math.ceil((chars * 6) / 8) : chars);
  const storedCost = binary ? bytes.length : packedRawLength(bytes.length);
  let shippedMode = MODE_STORED;
  let fastTokensToShip: Token[] | undefined;
  let smallTokensToShip: Token[] | undefined;
  let smallPlanToShip: SmallPlan | undefined;
  if (mode === 'fast') {
    const pricing = fastPricing(bytes, language);
    let tokens = parse(bytes, language.dictionary, dictIndex, pricing, segments);
    let fastCost = fastBodyCost(tokens, bytes, language)!;
    if (segments) {
      // The greedy parse is approximate, so the extended search space can occasionally ship a
      // larger body; compare against the plain parse exactly and prefer plain on ties (the
      // frame then stays bit-identical to the plain unfenced frame).
      const plainTokens = parse(bytes, language.dictionary, dictIndex, pricing);
      const plainCost = fastBodyCost(plainTokens, bytes, language)!;
      if (fastOutCost(plainCost) <= fastOutCost(fastCost)) {
        tokens = plainTokens;
        fastCost = plainCost;
      }
    }
    if (fastOutCost(fastCost) < storedCost) {
      shippedMode = MODE_FAST;
      fastTokensToShip = tokens;
    }
  } else {
    // Auto-downgrade (normative, emission-free): compare complete frames analytically —
    // small vs fast vs stored — smallest wins, ties choose the simpler encoding. The fast
    // candidate is the cheaper of two token lists: the small (optimal) parse re-priced in fast
    // chars, and a pure fast parse — the optimal parse minimizes bits, so alone it could ship a
    // fast frame larger than mode 'fast' would produce for the same input.
    // Plan sizes are compared in output units — exact chars for text, exact bytes for binary —
    // so ties at the shipped-frame granularity keep the documented preference for the plainer
    // encoding (e.g. a byte-tied fenced plan must not add a needless registration dependency).
    const planCost = (plan: SmallPlan): number => (binary ? Math.ceil(plan.totalBits / 8) : plan.charCost);
    const pricing = smallPricing(bytes, language);
    let smallTokens = parse(bytes, language.dictionary, dictIndex, pricing, segments);
    let plan = planSmallBody(smallTokens, bytes, language);
    if (segments) {
      // The DP's run-floor and path-carried rep state are approximations, so mirror the fast
      // path: price the plain parse exactly and prefer it on ties.
      const plainTokens = parse(bytes, language.dictionary, dictIndex, pricing);
      const plainPlan = planSmallBody(plainTokens, bytes, language);
      if (planCost(plainPlan) <= planCost(plan)) {
        smallTokens = plainTokens;
        plan = plainPlan;
      }
    }
    let planTokens = smallTokens;
    if (bytes.length > 0) {
      // The DP charges literal runs only a slot-0 floor, so on rare short inputs a match-bearing
      // parse can lose to plain literals; the all-literal plan is O(n) to price, so compare it.
      const allLiteralTokens: Token[] = [{ type: 'lit', start: 0, end: bytes.length }];
      const allLiteralPlan = planSmallBody(allLiteralTokens, bytes, language);
      if (planCost(allLiteralPlan) < planCost(plan)) {
        plan = allLiteralPlan;
        planTokens = allLiteralTokens;
      }
    }
    const lazyFastCost = fastBodyCost(smallTokens, bytes, language);
    const fastPricingModel = fastPricing(bytes, language);
    let fastTokens = parse(bytes, language.dictionary, dictIndex, fastPricingModel, segments);
    let pureFastCost = fastBodyCost(fastTokens, bytes, language)!;
    if (segments) {
      const plainFastTokens = parse(bytes, language.dictionary, dictIndex, fastPricingModel);
      const plainFastCost = fastBodyCost(plainFastTokens, bytes, language)!;
      if (fastOutCost(plainFastCost) <= fastOutCost(pureFastCost)) {
        fastTokens = plainFastTokens;
        pureFastCost = plainFastCost;
      }
    }
    // Output-unit comparison with pure-fast preferred on ties: the lazy (small-parse) tokens
    // may reach the extended dictionary, so a byte-tied win must not add a fenced dependency.
    const useLazyTokensForFast = lazyFastCost !== undefined && fastOutCost(lazyFastCost) < fastOutCost(pureFastCost);
    const fastCost = useLazyTokensForFast ? lazyFastCost : pureFastCost;
    const smallCost = planCost(plan);
    // Pick the smallest complete frame; on ties the simpler encoding wins (stored, fast, small).
    let bestCost = storedCost;
    if (fastOutCost(fastCost) < bestCost) {
      shippedMode = MODE_FAST;
      bestCost = fastOutCost(fastCost);
    }
    if (smallCost < bestCost) shippedMode = MODE_SMALL;
    if (shippedMode === MODE_FAST) fastTokensToShip = useLazyTokensForFast ? smallTokens : fastTokens;
    else if (shippedMode === MODE_SMALL) {
      smallTokensToShip = planTokens;
      smallPlanToShip = plan;
    }
  }

  // Normative: the flag is set iff a shipped dict token reaches above the frame dictionary,
  // so frames whose matches all stay inside it remain bit-identical to plain unfenced frames.
  const shippedTokens = fastTokensToShip ?? smallTokensToShip;
  const fenced = shippedTokens !== undefined && usesExtendedDictionary(shippedTokens, language.dictionary.length);
  const flags = shippedMode | (isString ? 0 : FLAG_BYTES) | (fenced ? FLAG_FENCED : 0);
  // Stored frames always carry language id 0 (decoders ignore it).
  const languageId = shippedMode === MODE_STORED ? 0 : language.id;

  const checksum = crc32(bytes);

  if (binary) {
    // Exact-capacity sink, no growth: the small body is ceil(totalBits / 8) bytes, and every
    // other body is smaller than the input (fast auto-downgrades to stored otherwise).
    const outCapacity =
      8 + CRC_BINARY_BYTES + (smallPlanToShip ? Math.ceil(smallPlanToShip.totalBits / 8) : bytes.length);
    const out = new TextSink(outCapacity);
    out.push(BINARY_MAGIC_VERSION);
    out.push(languageId);
    out.push(flags);
    pushByteVarint(out, bytes.length);
    pushCrc32Binary(out, checksum);
    if (shippedMode === MODE_STORED) out.append(bytes);
    else if (fastTokensToShip) {
      const body = new TextSink(bytes.length + 64);
      emitFastBody(body, fastTokensToShip, bytes, language);
      out.append(packFastCodes(body.buffer, body.length));
    } else out.append(emitSmallBody(smallPlanToShip!, language).toBytes());
    return out.toBytes();
  }

  const out = new TextSink(shippedMode === MODE_SMALL ? 24 : packedRawLength(bytes.length) + 24);
  out.push(RADIX64_CODES[MAGIC_VERSION]!);
  out.push(RADIX64_CODES[languageId]!);
  out.push(RADIX64_CODES[flags]!);
  pushVarint64(out, bytes.length);
  pushCrc32Text(out, checksum);
  if (shippedMode === MODE_STORED) pushPackedRaw(out, bytes, 0, bytes.length);
  else if (fastTokensToShip) emitFastBody(out, fastTokensToShip, bytes, language);
  return out.toString() + (smallPlanToShip ? emitSmallBody(smallPlanToShip, language).toText() : '');
}

/**
 * Decompresses a tokzip frame — a text frame when given a string, a binary frame when given
 * bytes; the return type follows the header's input-type flag.
 */
export function decompress(data: string | Uint8Array, options?: DecompressOptions): string | Uint8Array {
  const maxOutputSize = options?.maxOutputSize ?? DEFAULT_MAX_OUTPUT_SIZE;
  // NaN or a non-number (e.g. '10MB' from an untyped caller) would make the size guard below
  // always pass, silently disabling the allocation cap. Infinity is allowed as an explicit
  // "no cap".
  if (typeof maxOutputSize !== 'number' || Number.isNaN(maxOutputSize) || maxOutputSize < 0) {
    throw new RangeError(`invalid maxOutputSize: ${maxOutputSize}`);
  }
  const { flags, bytes } =
    typeof data === 'string' ? decompressText(data, maxOutputSize) : decompressBinary(data, maxOutputSize);
  if ((flags & FLAG_BYTES) !== 0) return bytes;
  try {
    return fatalDecoder.decode(bytes);
  } catch {
    throw new TokzipDecodeError('invalid UTF-8 in string frame');
  }
}

function decompressText(data: string, maxOutputSize: number): { flags: number; bytes: Uint8Array } {
  const magicVersion = readRadix64(data, 0);
  if (magicVersion !== MAGIC_VERSION) {
    if (magicVersion >>> 3 === MAGIC_VERSION >>> 3) throw new TokzipDecodeError('unknown version');
    throw new TokzipDecodeError('bad magic');
  }
  const languageId = readRadix64(data, 1);
  const flags = readRadix64(data, 2);
  if ((flags & RESERVED_FLAG_MASK) !== 0) throw new TokzipDecodeError('reserved flag bits set');
  const mode = flags & 3;
  const fenced = (flags & FLAG_FENCED) !== 0;
  const { value: outputSize, pos: crcStart } = readVarint64(data, 3);
  if (outputSize > maxOutputSize) throw new TokzipDecodeError('declared size exceeds maxOutputSize');
  const declaredCrc = readCrc32Text(data, crcStart);
  const bodyStart = crcStart + CRC_TEXT_CHARS;

  let bytes: Uint8Array;
  if (mode === MODE_STORED) {
    // Stored frames decode under any language id (zero registration needed).
    const bodyLength = packedRawLength(outputSize);
    if (data.length !== bodyStart + bodyLength) {
      if (data.length < bodyStart + bodyLength) throw new TokzipDecodeError('truncated payload');
      throw new TokzipDecodeError('trailing characters after payload');
    }
    bytes = new Uint8Array(outputSize);
    readPackedRaw(data, bodyStart, bytes, 0, outputSize);
  } else if (mode === MODE_FAST || mode === MODE_SMALL) {
    // The normative auto-downgrade means a conforming non-stored body is strictly smaller
    // than the stored body of the same size; rejecting the rest keeps frames canonical and
    // bounds decode-side allocations by the declared output size.
    if (data.length - bodyStart >= packedRawLength(outputSize)) {
      throw new TokzipDecodeError('non-canonical frame: body not smaller than stored');
    }
    const language: RegisteredLanguage = requireLanguageById(languageId);
    bytes =
      mode === MODE_FAST
        ? decodeFastBody(data, bodyStart, data.length, outputSize, language, fenced)
        : decodeSmallBody(data, bodyStart, data.length, outputSize, language, fenced);
  } else {
    throw new TokzipDecodeError('invalid mode');
  }
  if (crc32(bytes) !== declaredCrc) throw new TokzipDecodeError('checksum mismatch');
  return { flags, bytes };
}

function decompressBinary(data: Uint8Array, maxOutputSize: number): { flags: number; bytes: Uint8Array } {
  if (data.length < 3) throw new TokzipDecodeError('truncated payload');
  const magicVersion = data[0]!;
  if (magicVersion !== BINARY_MAGIC_VERSION) {
    if ((magicVersion & 0b1111_1000) === (BINARY_MAGIC_VERSION & 0b1111_1000)) {
      throw new TokzipDecodeError('unknown version');
    }
    throw new TokzipDecodeError('bad magic');
  }
  const languageId = data[1]!;
  const flags = data[2]!;
  // The binary flags byte reserves bits 7:4 on top of the text container's reserved bits.
  if ((flags & (0b1111_0000 | RESERVED_FLAG_MASK)) !== 0) throw new TokzipDecodeError('reserved flag bits set');
  const mode = flags & 3;
  const fenced = (flags & FLAG_FENCED) !== 0;
  const { value: outputSize, pos: crcStart } = readByteVarint(data, 3);
  if (outputSize > maxOutputSize) throw new TokzipDecodeError('declared size exceeds maxOutputSize');
  const declaredCrc = readCrc32Binary(data, crcStart);
  const bodyStart = crcStart + CRC_BINARY_BYTES;

  let bytes: Uint8Array;
  if (mode === MODE_STORED) {
    // Stored frames decode under any language id (zero registration needed).
    if (data.length !== bodyStart + outputSize) {
      if (data.length < bodyStart + outputSize) throw new TokzipDecodeError('truncated payload');
      throw new TokzipDecodeError('trailing characters after payload');
    }
    // Explicit copy, not data.slice(): callers may pass a Buffer, whose slice() returns a
    // view over the input frame's memory instead of an independent copy.
    bytes = new Uint8Array(outputSize);
    bytes.set(data.subarray(bodyStart, bodyStart + outputSize));
  } else if (mode === MODE_FAST || mode === MODE_SMALL) {
    // Mirrors the text container: a conforming non-stored body is strictly smaller than the
    // stored body (here the raw byte count), keeping frames canonical and allocations bounded.
    if (data.length - bodyStart >= outputSize) {
      throw new TokzipDecodeError('non-canonical frame: body not smaller than stored');
    }
    const language: RegisteredLanguage = requireLanguageById(languageId);
    bytes =
      mode === MODE_FAST
        ? decodeFastBodyBinary(data, bodyStart, data.length, outputSize, language, fenced)
        : decodeSmallBodyBinary(data, bodyStart, data.length, outputSize, language, fenced);
  } else {
    throw new TokzipDecodeError('invalid mode');
  }
  if (crc32(bytes) !== declaredCrc) throw new TokzipDecodeError('checksum mismatch');
  return { flags, bytes };
}

/**
 * Builds an unconditional stored frame — no tokenizer, no entropy coder, no dictionary.
 * The last-resort fallback of `compressForStorage`: even if every compression path is
 * broken, this depends only on the header writers and the raw packing.
 */
export function compressStored(input: string | Uint8Array, output: 'text' | 'binary'): string | Uint8Array {
  const isString = typeof input === 'string';
  const bytes = isString ? textEncoder.encode(input) : input;
  const flags = MODE_STORED | (isString ? 0 : FLAG_BYTES);
  const checksum = crc32(bytes);
  if (output === 'binary') {
    const out = new TextSink(8 + CRC_BINARY_BYTES + bytes.length);
    out.push(BINARY_MAGIC_VERSION);
    out.push(0);
    out.push(flags);
    pushByteVarint(out, bytes.length);
    pushCrc32Binary(out, checksum);
    out.append(bytes);
    return out.toBytes();
  }
  const out = new TextSink(packedRawLength(bytes.length) + 24);
  out.push(RADIX64_CODES[MAGIC_VERSION]!);
  out.push(RADIX64_CODES[0]!);
  out.push(RADIX64_CODES[flags]!);
  pushVarint64(out, bytes.length);
  pushCrc32Text(out, checksum);
  pushPackedRaw(out, bytes, 0, bytes.length);
  return out.toString();
}

/** Emits a CRC-32 as 6 radix-64 chars: little-endian 6-bit groups, top 4 bits zero. */
function pushCrc32Text(out: TextSink, crc: number): void {
  for (let i = 0; i < CRC_TEXT_CHARS; i++) out.push(RADIX64_CODES[(crc >>> (i * 6)) & 63]!);
}

export function readCrc32Text(data: string, pos: number): number {
  let crc = 0;
  for (let i = 0; i < CRC_TEXT_CHARS; i++) {
    const group = readRadix64(data, pos + i);
    // Canonical: bits above 31 do not exist, so the last group must fit in 2 bits.
    if (i === CRC_TEXT_CHARS - 1 && group > 3) throw new TokzipDecodeError('non-canonical checksum');
    crc |= group << (i * 6);
  }
  // oxlint-disable-next-line unicorn/prefer-math-trunc -- >>> 0 converts to unsigned; Math.trunc would keep the sign
  return crc >>> 0;
}

/** Emits a CRC-32 as 4 little-endian bytes. */
export function pushCrc32Binary(out: TextSink, crc: number): void {
  for (let i = 0; i < CRC_BINARY_BYTES; i++) out.push((crc >>> (i * 8)) & 0xFF);
}

export function readCrc32Binary(data: Uint8Array, pos: number): number {
  if (pos + CRC_BINARY_BYTES > data.length) throw new TokzipDecodeError('truncated payload');
  let crc = 0;
  for (let i = 0; i < CRC_BINARY_BYTES; i++) crc |= data[pos + i]! << (i * 8);
  // oxlint-disable-next-line unicorn/prefer-math-trunc -- >>> 0 converts to unsigned; Math.trunc would keep the sign
  return crc >>> 0;
}

export function pushByteVarint(out: TextSink, value: number): void {
  if (value < 0 || !Number.isSafeInteger(value)) throw new RangeError(`invalid varint value: ${value}`);
  do {
    // Arithmetic, not & / >>>: varint values span 35 bits, beyond 32-bit bitwise range.
    const group = value % 128;
    value = Math.floor(value / 128);
    out.push(value > 0 ? group | 128 : group);
  } while (value > 0);
}

export function readByteVarint(data: Uint8Array, pos: number): { value: number; pos: number } {
  let value = 0;
  let shift = 1;
  for (let i = 0; i < BYTE_VARINT_MAX_BYTES; i++) {
    if (pos >= data.length) throw new TokzipDecodeError('truncated payload');
    const group = data[pos++]!;
    value += (group & 127) * shift;
    if ((group & 128) === 0) {
      // Canonical form: a multi-byte varint must not end in a zero group.
      if (i > 0 && (group & 127) === 0) throw new TokzipDecodeError('non-canonical varint');
      return { value, pos };
    }
    shift *= 128;
  }
  throw new TokzipDecodeError('varint exceeds bound');
}
