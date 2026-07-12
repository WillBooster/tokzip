import { languageByName, requireLanguageById, type RegisteredLanguage } from './dictionary.ts';
import { TokzipDecodeError } from './errors.ts';
import { decodeFastBody, encodeFastBody, fastBodyCost, fastPricing } from './fastMode.ts';
import {
  DEFAULT_MAX_OUTPUT_SIZE,
  FLAG_BYTES,
  MAGIC_VERSION,
  MODE_FAST,
  MODE_SMALL,
  MODE_STORED,
  RESERVED_FLAG_MASK,
} from './format.ts';
import { dictIndexFor, parse } from './lz.ts';
import {
  packedRawLength,
  pushPackedRaw,
  pushVarint64,
  RADIX64_CHARS,
  readPackedRaw,
  readRadix64,
  readVarint64,
} from './radix64.ts';
import { decodeSmallBody, emitSmallBody, planSmallBody, smallPricing } from './smallMode.ts';

export interface CompressOptions {
  /** Language dictionary to use; default 'none' (id 0, wrapper dictionary only). */
  language?: string;
  /** Optimization target; both modes are lossless. Default 'fast'. */
  mode?: 'fast' | 'small';
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

/** Compresses a string (UTF-8) or raw bytes into a safe-ASCII text frame. */
export function compress(input: string | Uint8Array, options?: CompressOptions): string {
  const isString = typeof input === 'string';
  const bytes = isString ? textEncoder.encode(input) : input;
  const languageName = options?.language ?? 'none';
  const language = languageByName(languageName);
  if (!language) throw new RangeError(`unregistered language: ${languageName}`);
  const mode = options?.mode ?? 'fast';
  // Untyped callers must not silently fall through to the (expensive) small path on typos.
  if (mode !== 'fast' && mode !== 'small') throw new RangeError(`invalid mode: ${String(mode)}`);

  const storedCost = packedRawLength(bytes.length);
  let shippedMode = MODE_STORED;
  let body = '';
  if (mode === 'fast') {
    const tokens = parse(bytes, language.dictionary, dictIndexFor(language), fastPricing(bytes, language));
    const fastCost = fastBodyCost(tokens, bytes, language)!;
    if (fastCost < storedCost) {
      shippedMode = MODE_FAST;
      body = encodeFastBody(tokens, bytes, language);
    }
  } else {
    // Auto-downgrade (normative, emission-free): compare complete frames analytically —
    // small vs fast vs stored — smallest wins, ties choose the simpler encoding. The fast
    // candidate is the cheaper of two token lists: the small (lazy) parse re-priced in fast
    // chars, and a pure fast parse — the lazy parse optimizes bits, so alone it could ship a
    // fast frame larger than mode 'fast' would produce for the same input.
    const smallTokens = parse(bytes, language.dictionary, dictIndexFor(language), smallPricing(bytes, language));
    const plan = planSmallBody(smallTokens, bytes, language);
    const lazyFastCost = fastBodyCost(smallTokens, bytes, language);
    const fastTokens = parse(bytes, language.dictionary, dictIndexFor(language), fastPricing(bytes, language));
    const pureFastCost = fastBodyCost(fastTokens, bytes, language)!;
    const useLazyTokensForFast = lazyFastCost !== undefined && lazyFastCost < pureFastCost;
    const fastCost = useLazyTokensForFast ? lazyFastCost : pureFastCost;
    // Pick the smallest complete frame; on ties the simpler encoding wins (stored, fast, small).
    let bestCost = storedCost;
    if (fastCost < bestCost) {
      shippedMode = MODE_FAST;
      bestCost = fastCost;
    }
    if (plan.charCost < bestCost) shippedMode = MODE_SMALL;
    if (shippedMode === MODE_FAST)
      body = encodeFastBody(useLazyTokensForFast ? smallTokens : fastTokens, bytes, language);
    else if (shippedMode === MODE_SMALL) body = emitSmallBody(plan, bytes, language);
  }

  const out: string[] = [
    RADIX64_CHARS[MAGIC_VERSION]!,
    // Stored frames always carry language id 0 (decoders ignore it).
    RADIX64_CHARS[shippedMode === MODE_STORED ? 0 : language.id]!,
    RADIX64_CHARS[shippedMode | (isString ? 0 : FLAG_BYTES)]!,
  ];
  pushVarint64(out, bytes.length);
  if (shippedMode === MODE_STORED) pushPackedRaw(out, bytes, 0, bytes.length);
  return out.join('') + body;
}

/** Decompresses a tokzip text frame; the return type follows the header's input-type flag. */
export function decompress(data: string, options?: DecompressOptions): string | Uint8Array {
  const maxOutputSize = options?.maxOutputSize ?? DEFAULT_MAX_OUTPUT_SIZE;
  // NaN would make the size guard below always pass, silently disabling the allocation cap.
  // Infinity is allowed as an explicit "no cap".
  if (Number.isNaN(maxOutputSize) || maxOutputSize < 0) {
    throw new RangeError(`invalid maxOutputSize: ${maxOutputSize}`);
  }
  const magicVersion = readRadix64(data, 0);
  if (magicVersion !== MAGIC_VERSION) {
    if (magicVersion >>> 3 === MAGIC_VERSION >>> 3) throw new TokzipDecodeError('unknown version');
    throw new TokzipDecodeError('bad magic');
  }
  const languageId = readRadix64(data, 1);
  const flags = readRadix64(data, 2);
  if ((flags & RESERVED_FLAG_MASK) !== 0) throw new TokzipDecodeError('reserved flag bits set');
  const mode = flags & 3;
  const isBytes = (flags & FLAG_BYTES) !== 0;
  const { value: outputSize, pos: bodyStart } = readVarint64(data, 3);
  if (outputSize > maxOutputSize) throw new TokzipDecodeError('declared size exceeds maxOutputSize');

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
        ? decodeFastBody(data, bodyStart, data.length, outputSize, language)
        : decodeSmallBody(data, bodyStart, data.length, outputSize, language);
  } else {
    throw new TokzipDecodeError('invalid mode');
  }

  if (isBytes) return bytes;
  try {
    return fatalDecoder.decode(bytes);
  } catch {
    throw new TokzipDecodeError('invalid UTF-8 in string frame');
  }
}
