import { frameChecksum } from './checksum.ts';
import { languageByName, requireLanguageById, type RegisteredLanguage } from './dictionary.ts';
import { TokzipDecodeError } from './errors.ts';
import { computeDictSegments, usesExtendedDictionary } from './fences.ts';
import {
  BINARY_MAGIC_VERSION,
  CRC_BINARY_BYTES,
  CRC_TEXT_CHARS,
  DEFAULT_MAX_OUTPUT_SIZE,
  FLAG_BYTES,
  FLAG_FENCED,
  MAGIC_VERSION,
  MODE_SMALL,
  MODE_STORED,
  RESERVED_FLAG_MASK,
  SMALL_WINDOW,
} from './format.ts';
import { dictIndexIfNeeded, parse, type Token } from './lz.ts';
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
import { radix85FromBytes, radix85Length } from './radix85.ts';
import { decodeSmallBody, decodeSmallBodyBinary, encodeSmallBody, smallPricing } from './smallMode.ts';

export interface CompressOptions {
  /** Language dictionary to use; default 'none' (id 0, wrapper dictionary only). */
  language?: string;
  /**
   * Output channel; default 'text'. 'text' emits a safe-ASCII frame (JSON- and
   * template-literal-safe); 'binary' emits the range-coded stream at 8 bits per byte,
   * saving the 25% radix-85 text tax on the body (each channel independently downgrades
   * to stored, so whole-frame ratios vary slightly).
   */
  output?: 'text' | 'binary';
}

export interface DecompressOptions {
  /** Refuses to allocate more than this many output bytes (default 64 MiB). */
  maxOutputSize?: number;
}

/**
 * Inputs up to this size also price the all-literal candidate. The DP's mispricing only
 * materializes on short high-entropy inputs — measured on 60k synthetic payloads it never
 * fired above ~150 bytes (and never on real corpus documents) — and the extra literal-only
 * range encode measurably slows compression when run on every document, so the bound stays
 * comfortably above the observed ceiling while keeping the common path single-encode.
 */
export const ALL_LITERAL_CANDIDATE_MAX = 512;

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
  // Lone surrogates encode as U+FFFD per WHATWG TextEncoder (see FORMAT.md §3) — the same
  // behavior as the platform CompressionStream pipeline. compressForStorage rejects such
  // strings up front for callers that need the byte-exact guarantee.
  const bytes = isString ? textEncoder.encode(input) : input;
  const languageName = options?.language ?? 'none';
  const language = languageByName(languageName);
  if (!language) throw new RangeError(`unregistered language: ${languageName}`);
  const output = options?.output ?? 'text';
  if (output !== 'text' && output !== 'binary') throw new RangeError(`invalid output: ${String(output)}`);
  const binary = output === 'binary';

  const storedCost = binary ? bytes.length : packedRawLength(bytes.length);
  let shippedMode = MODE_STORED;
  let bodyToShip: Uint8Array | undefined;
  let fenced = false;
  // Index building stays inside the non-empty branch: empty input ships the fixed stored
  // frame without parsing, so it must not pay a first-compress matcher build.
  if (bytes.length > 0) {
    // Fenced dictionary extension: labeled code fences extend the searchable dictionary
    // space with the block language's suffix (undefined when the input has no such fence).
    const segments = computeDictSegments(bytes, language, SMALL_WINDOW);
    const dictIndex = dictIndexIfNeeded(language, bytes.length);
    // Candidate bodies are compared in output units — bytes on the binary channel, radix-85
    // chars on the text channel (which quantizes to 4-byte words, so 1–3 body bytes often
    // cost zero shipped chars): a candidate must not win a side effect for zero savings.
    const outCost = (candidate: Uint8Array): number => (binary ? candidate.length : radix85Length(candidate.length));
    const pricing = smallPricing(bytes, language);
    let tokens = parse(bytes, language.dictionary, dictIndex, pricing, segments);
    let body = encodeSmallBody(tokens, bytes, language);
    if (segments) {
      // The parse is guided by approximate prices, so the extended search space can
      // occasionally ship a larger body; compare against the plain parse exactly and prefer
      // plain on ties (the frame then stays identical to the plain unfenced frame, adding no
      // registration dependency).
      const plainTokens = parse(bytes, language.dictionary, dictIndex, pricing);
      const plainBody = encodeSmallBody(plainTokens, bytes, language);
      if (outCost(plainBody) <= outCost(body)) {
        tokens = plainTokens;
        body = plainBody;
      }
    }
    if (bytes.length <= ALL_LITERAL_CANDIDATE_MAX) {
      // The DP prices with static priors and a literal-run floor, so on short high-entropy
      // inputs a match-bearing parse can lose to plain literals; the all-literal body is
      // O(n) to emit and only ever wins there, so larger inputs skip the extra pass (the
      // bound keys off the input length alone, keeping compression deterministic).
      const allLiteralTokens: Token[] = [{ type: 'lit', start: 0, end: bytes.length }];
      const allLiteralBody = encodeSmallBody(allLiteralTokens, bytes, language);
      if (outCost(allLiteralBody) < outCost(body)) {
        tokens = allLiteralTokens;
        body = allLiteralBody;
      }
    }
    // Auto-downgrade (normative): the frame never expands beyond the stored body; ties
    // choose the simpler stored encoding.
    const bodyCost = outCost(body);
    if (bodyCost < storedCost) {
      shippedMode = MODE_SMALL;
      bodyToShip = body;
      // Normative: the flag is set iff a shipped dict token reaches above the frame
      // dictionary, so frames whose matches all stay inside it remain identical to plain
      // unfenced frames.
      fenced = usesExtendedDictionary(tokens, language.dictionary.length);
    }
  }

  const flags = shippedMode | (isString ? 0 : FLAG_BYTES) | (fenced ? FLAG_FENCED : 0);
  // Stored frames always carry language id 0 (decoders ignore it).
  const languageId = shippedMode === MODE_STORED ? 0 : language.id;
  const checksum = frameChecksum(bytes, !isString);

  if (binary) {
    const outCapacity = 8 + CRC_BINARY_BYTES + (bodyToShip ? bodyToShip.length : bytes.length);
    const out = new TextSink(outCapacity);
    out.push(BINARY_MAGIC_VERSION);
    out.push(languageId);
    out.push(flags);
    pushByteVarint(out, bytes.length);
    pushCrc32Binary(out, checksum);
    out.append(bodyToShip ?? bytes);
    return out.toBytes();
  }

  const out = new TextSink(shippedMode === MODE_SMALL ? 24 : packedRawLength(bytes.length) + 24);
  out.push(RADIX64_CODES[MAGIC_VERSION]!);
  out.push(RADIX64_CODES[languageId]!);
  out.push(RADIX64_CODES[flags]!);
  pushVarint64(out, bytes.length);
  pushCrc32Text(out, checksum);
  if (shippedMode === MODE_STORED) pushPackedRaw(out, bytes, 0, bytes.length);
  return out.toString() + (bodyToShip ? radix85FromBytes(bodyToShip) : '');
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
  } else if (mode === MODE_SMALL) {
    // The normative auto-downgrade means a conforming non-stored body is strictly smaller
    // than the stored body of the same size; rejecting the rest keeps frames canonical and
    // bounds decode-side allocations by the declared output size.
    if (data.length - bodyStart >= packedRawLength(outputSize)) {
      throw new TokzipDecodeError('non-canonical frame: body not smaller than stored');
    }
    const language: RegisteredLanguage = requireLanguageById(languageId);
    bytes = decodeSmallBody(data, bodyStart, data.length, outputSize, language, fenced);
  } else {
    throw new TokzipDecodeError('invalid mode');
  }
  if (frameChecksum(bytes, (flags & FLAG_BYTES) !== 0) !== declaredCrc)
    throw new TokzipDecodeError('checksum mismatch');
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
  } else if (mode === MODE_SMALL) {
    // Mirrors the text container: a conforming non-stored body is strictly smaller than the
    // stored body (here the raw byte count), keeping frames canonical and allocations bounded.
    if (data.length - bodyStart >= outputSize) {
      throw new TokzipDecodeError('non-canonical frame: body not smaller than stored');
    }
    const language: RegisteredLanguage = requireLanguageById(languageId);
    bytes = decodeSmallBodyBinary(data, bodyStart, data.length, outputSize, language, fenced);
  } else {
    throw new TokzipDecodeError('invalid mode');
  }
  if (frameChecksum(bytes, (flags & FLAG_BYTES) !== 0) !== declaredCrc)
    throw new TokzipDecodeError('checksum mismatch');
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
  const checksum = frameChecksum(bytes, !isString);
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
