import { pushByteVarint } from './container.ts';
import { languageByName, requireLanguageById, type RegisteredLanguage } from './dictionary.ts';
import { TokzipDecodeError } from './errors.ts';
import { decodeFastBodyBinary, emitFastBody, fastBodyCost, fastPricing, packFastCodes } from './fastMode.ts';
import { DEFAULT_MAX_OUTPUT_SIZE, FAST_WINDOW, MODE_FAST, MODE_SMALL, MODE_STORED, SMALL_WINDOW } from './format.ts';
import { dictIndexFor, OPTIMAL_MAX_INPUT, parse } from './lz.ts';
import { TextSink } from './radix64.ts';
import { decodeSmallBodyBinary, emitSmallBody, planSmallBody, smallPricing } from './smallMode.ts';

/**
 * First byte of every tokzip stream: bit 7 set (binary channel) over low-6 magic 0b111 and
 * stream-format version 0 — disjoint from every frame magic (low-6 magic 0b110) so streams
 * and one-shot frames can never be confused.
 */
const STREAM_MAGIC_VERSION = 0b1011_1000;

/**
 * Stream flags byte: bits 1:0 carry the stream mode (fast/small); bit 2 marks window
 * carry-over (blocks are decoded with the previous blocks' output seeded as history — the
 * small-mode literal context chains across the block boundary, so decoders must know);
 * the rest are reserved.
 */
const STREAM_FLAG_CARRY = 0b100;
const STREAM_RESERVED_FLAG_MASK = 0b1111_1000;

const BYTE_VARINT_MAX_BYTES = 5;

const DEFAULT_BLOCK_SIZE = 1 << 18; // 256 KB (matches the fast-mode window).
const MIN_BLOCK_SIZE = 1 << 10;

const textEncoder = new TextEncoder();

export interface CompressionStreamOptions {
  /** Language dictionary to use; default 'none' (id 0, wrapper dictionary only). */
  language?: string;
  /** Optimization target; both modes are lossless. Default 'fast'. */
  mode?: 'fast' | 'small';
  /** Raw bytes per compressed block (larger blocks trade latency/memory for ratio). Default 256 KB. */
  blockSize?: number;
  /**
   * Carries the LZ window across block boundaries (matches may reference the previous blocks'
   * output), recovering most of the ratio lost to chunking at some compression-speed cost.
   * Default true.
   */
  carryWindow?: boolean;
  /**
   * Upper bound on the carried history in bytes (clamped to the mode's window). Carried
   * history is re-priced and re-indexed every block, so small blocks with a full window pay
   * a large speed multiplier; a tighter limit trades ratio for compression speed. Defaults
   * to the mode's window (small mode also stays inside the optimal-parse input bound).
   */
  historyLimit?: number;
}

export interface DecompressionStreamOptions {
  /** Refuses blocks declaring more than this many output bytes (default 64 MiB). */
  maxBlockSize?: number;
}

/**
 * Compresses a byte (or string-chunk) stream into a tokzip stream: a 3-byte stream header
 * followed by independent length-prefixed blocks and a terminator. Works on Web Streams, so
 * it can be piped in Node.js (18+) and browsers alike; the whole compression pipeline —
 * blocking, window carry-over, per-block stored/fast/small selection — is hidden inside.
 */
export class TokzipCompressionStream extends TransformStream<Uint8Array | string, Uint8Array> {
  constructor(options?: CompressionStreamOptions) {
    const encoder = new BlockEncoder(options);
    super({
      transform(chunk, controller) {
        encoder.push(chunk, controller);
      },
      flush(controller) {
        encoder.finish(controller);
      },
    });
  }
}

/** Decompresses a tokzip stream produced by {@link TokzipCompressionStream}. */
export class TokzipDecompressionStream extends TransformStream<Uint8Array, Uint8Array> {
  constructor(options?: DecompressionStreamOptions) {
    const decoder = new BlockDecoder(options);
    super({
      transform(chunk, controller) {
        decoder.push(chunk, controller);
      },
      flush() {
        decoder.finish();
      },
    });
  }
}

type ByteController = TransformStreamDefaultController<Uint8Array>;

class BlockEncoder {
  private readonly language: RegisteredLanguage;
  private readonly mode: 'fast' | 'small';
  private readonly blockSize: number;
  private readonly carryWindow: boolean;
  /** Longest carried history; keeps small-mode combined inputs inside the optimal-parse bound. */
  private readonly historyLimit: number;
  private history: Uint8Array = new Uint8Array(0);
  private readonly pending: Uint8Array[] = [];
  private pendingLength = 0;
  private headerWritten = false;

  constructor(options?: CompressionStreamOptions) {
    const languageName = options?.language ?? 'none';
    const language = languageByName(languageName);
    if (!language) throw new RangeError(`unregistered language: ${languageName}`);
    this.language = language;
    const mode = options?.mode ?? 'fast';
    if (mode !== 'fast' && mode !== 'small') throw new RangeError(`invalid mode: ${String(mode)}`);
    this.mode = mode;
    const blockSize = options?.blockSize ?? DEFAULT_BLOCK_SIZE;
    if (!Number.isSafeInteger(blockSize) || blockSize < MIN_BLOCK_SIZE) {
      throw new RangeError(`invalid blockSize: ${blockSize}`);
    }
    this.blockSize = blockSize;
    const window = mode === 'fast' ? FAST_WINDOW : SMALL_WINDOW;
    const defaultLimit = mode === 'small' ? Math.min(window, Math.max(0, OPTIMAL_MAX_INPUT - blockSize)) : window;
    const historyLimit = options?.historyLimit ?? defaultLimit;
    if (!Number.isSafeInteger(historyLimit) || historyLimit < 0) {
      throw new RangeError(`invalid historyLimit: ${historyLimit}`);
    }
    this.historyLimit = Math.min(historyLimit, window);
    // A zero history budget (huge small-mode blocks) degenerates to carry-less blocks; the
    // header flag must say so, or decoders would seed history the encoder never used.
    this.carryWindow = (options?.carryWindow ?? true) && this.historyLimit > 0;
  }

  push(chunk: Uint8Array | string, controller: ByteController): void {
    const bytes = typeof chunk === 'string' ? textEncoder.encode(chunk) : chunk;
    if (!(bytes instanceof Uint8Array)) throw new TypeError('chunk must be a Uint8Array or string');
    if (bytes.length === 0) return;
    this.pending.push(bytes);
    this.pendingLength += bytes.length;
    while (this.pendingLength >= this.blockSize) this.emitBlock(this.takeBlock(this.blockSize), controller);
  }

  finish(controller: ByteController): void {
    if (this.pendingLength > 0) this.emitBlock(this.takeBlock(this.pendingLength), controller);
    const out = new TextSink(4);
    if (!this.headerWritten) this.writeHeader(out);
    out.push(0); // End-of-stream marker (a zero block-length varint).
    controller.enqueue(out.toBytes());
  }

  private writeHeader(out: TextSink): void {
    out.push(STREAM_MAGIC_VERSION);
    out.push(this.language.id);
    out.push((this.mode === 'fast' ? MODE_FAST : MODE_SMALL) | (this.carryWindow ? STREAM_FLAG_CARRY : 0));
    this.headerWritten = true;
  }

  private takeBlock(size: number): Uint8Array {
    const block = new Uint8Array(size);
    let at = 0;
    while (at < size) {
      const head = this.pending[0]!;
      const take = Math.min(head.length, size - at);
      block.set(head.subarray(0, take), at);
      at += take;
      if (take === head.length) this.pending.shift();
      else this.pending[0] = head.subarray(take);
    }
    this.pendingLength -= size;
    return block;
  }

  private emitBlock(block: Uint8Array, controller: ByteController): void {
    const language = this.language;
    const historyLength = this.carryWindow ? this.history.length : 0;
    let input = block;
    if (historyLength > 0) {
      input = new Uint8Array(historyLength + block.length);
      input.set(this.history);
      input.set(block, historyLength);
    }
    const dictIndex = dictIndexFor(language);

    // Per-block frame selection mirrors the one-shot auto-downgrade: smallest body wins,
    // ties prefer the simpler encoding (stored, then fast, then small).
    let mode = MODE_STORED;
    let body: Uint8Array | undefined;
    if (this.mode === 'fast') {
      const pricing = fastPricing(input, language);
      // Encoder-side policy (format-compatible): price-aware lazy matching benches 1–2%
      // smaller than the plain greedy parse at stream-comparable speeds, so streams — which
      // already amortize per-block work — take the better parse unconditionally.
      pricing.lazy = true;
      const tokens = parse(input, language.dictionary, dictIndex, pricing, undefined, historyLength);
      // Parse bounds (window, maxDictStart, length caps) keep every token representable.
      const chars = fastBodyCost(tokens, input, language)!;
      if (Math.ceil((chars * 6) / 8) < block.length) {
        mode = MODE_FAST;
        const sink = new TextSink(chars);
        emitFastBody(sink, tokens, input, language);
        body = packFastCodes(sink.buffer, sink.length);
      }
    } else {
      const pricing = smallPricing(input, language);
      const tokens = parse(input, language.dictionary, dictIndex, pricing, undefined, historyLength);
      const plan = planSmallBody(tokens, input, language);
      const smallBytes = Math.ceil(plan.totalBits / 8);
      // The small parse re-priced in fast units (undefined when a token exceeds fast's
      // offset range); a cheaper fast body ships instead of a marginal small one.
      const fastChars = fastBodyCost(tokens, input, language);
      const fastBytes = fastChars === undefined ? Number.POSITIVE_INFINITY : Math.ceil((fastChars * 6) / 8);
      const best = Math.min(block.length, fastBytes, smallBytes);
      if (fastBytes === best && fastBytes < block.length) {
        mode = MODE_FAST;
        const sink = new TextSink(fastChars!);
        emitFastBody(sink, tokens, input, language);
        body = packFastCodes(sink.buffer, sink.length);
      } else if (smallBytes === best && smallBytes < block.length) {
        mode = MODE_SMALL;
        body = emitSmallBody(plan, language).toBytes();
      }
    }
    if (mode === MODE_STORED) body = block;

    const out = new TextSink(body!.length + 16);
    if (!this.headerWritten) this.writeHeader(out);
    pushByteVarint(out, body!.length);
    out.push(mode);
    pushByteVarint(out, block.length);
    out.append(body!);
    controller.enqueue(out.toBytes());

    if (this.carryWindow) {
      // `input` is history ⧺ block, so its tail is exactly the next block's history.
      this.history = input.slice(Math.max(0, input.length - this.historyLimit));
    }
  }
}

class BlockDecoder {
  private readonly maxBlockSize: number;
  private buffer: Uint8Array = new Uint8Array(0);
  private offset = 0;
  private language: RegisteredLanguage | undefined;
  private carry = false;
  private window = 0;
  private history: Uint8Array = new Uint8Array(0);
  private done = false;

  constructor(options?: DecompressionStreamOptions) {
    const maxBlockSize = options?.maxBlockSize ?? DEFAULT_MAX_OUTPUT_SIZE;
    if (Number.isNaN(maxBlockSize) || maxBlockSize < 0) throw new RangeError(`invalid maxBlockSize: ${maxBlockSize}`);
    this.maxBlockSize = maxBlockSize;
  }

  push(chunk: Uint8Array, controller: ByteController): void {
    if (!(chunk instanceof Uint8Array)) throw new TypeError('chunk must be a Uint8Array');
    if (chunk.length === 0) return;
    if (this.done) throw new TokzipDecodeError('trailing bytes after end of stream');
    this.append(chunk);
    while (this.decodeNext(controller)) {
      // Each iteration consumes one complete item from the buffer.
    }
  }

  finish(): void {
    if (!this.done || this.available() > 0) throw new TokzipDecodeError('truncated stream');
  }

  private available(): number {
    return this.buffer.length - this.offset;
  }

  private append(chunk: Uint8Array): void {
    const remaining = this.buffer.subarray(this.offset);
    const merged = new Uint8Array(remaining.length + chunk.length);
    merged.set(remaining);
    merged.set(chunk, remaining.length);
    this.buffer = merged;
    this.offset = 0;
  }

  /** Reads a canonical byte varint at `pos`, or undefined when more input is needed. */
  private tryReadVarint(pos: number): { value: number; pos: number } | undefined {
    let value = 0;
    let shift = 1;
    for (let i = 0; i < BYTE_VARINT_MAX_BYTES; i++) {
      if (pos >= this.buffer.length) return undefined;
      const group = this.buffer[pos++]!;
      value += (group & 127) * shift;
      if ((group & 128) === 0) {
        if (i > 0 && (group & 127) === 0) throw new TokzipDecodeError('non-canonical varint');
        return { value, pos };
      }
      shift *= 128;
    }
    throw new TokzipDecodeError('varint exceeds bound');
  }

  /** Decodes one header/block/terminator if fully buffered; false means "need more input". */
  private decodeNext(controller: ByteController): boolean {
    if (this.done) {
      if (this.available() > 0) throw new TokzipDecodeError('trailing bytes after end of stream');
      return false;
    }
    if (!this.language) {
      if (this.available() < 3) return false;
      const magic = this.buffer[this.offset]!;
      if (magic !== STREAM_MAGIC_VERSION) {
        if ((magic & 0b1111_1000) === (STREAM_MAGIC_VERSION & 0b1111_1000)) {
          throw new TokzipDecodeError('unknown version');
        }
        throw new TokzipDecodeError('bad magic');
      }
      const languageId = this.buffer[this.offset + 1]!;
      const flags = this.buffer[this.offset + 2]!;
      if ((flags & STREAM_RESERVED_FLAG_MASK) !== 0) throw new TokzipDecodeError('reserved flag bits set');
      const mode = flags & 3;
      if (mode !== MODE_FAST && mode !== MODE_SMALL) throw new TokzipDecodeError('invalid mode');
      this.language = requireLanguageById(languageId);
      this.carry = (flags & STREAM_FLAG_CARRY) !== 0;
      this.window = mode === MODE_FAST ? FAST_WINDOW : SMALL_WINDOW;
      this.offset += 3;
      return true;
    }

    const lengthField = this.tryReadVarint(this.offset);
    if (!lengthField) return false;
    const bodyLength = lengthField.value;
    if (bodyLength === 0) {
      this.offset = lengthField.pos;
      this.done = true;
      if (this.available() > 0) throw new TokzipDecodeError('trailing bytes after end of stream');
      return false;
    }
    if (lengthField.pos >= this.buffer.length) return false;
    const mode = this.buffer[lengthField.pos]!;
    const rawField = this.tryReadVarint(lengthField.pos + 1);
    if (!rawField) return false;
    const rawLength = rawField.value;
    const bodyStart = rawField.pos;
    if (this.buffer.length - bodyStart < bodyLength) return false;

    if (rawLength === 0) throw new TokzipDecodeError('empty block');
    if (rawLength > this.maxBlockSize) throw new TokzipDecodeError('declared size exceeds maxBlockSize');
    const bodyEnd = bodyStart + bodyLength;
    let out: Uint8Array;
    if (mode === MODE_STORED) {
      if (bodyLength !== rawLength) throw new TokzipDecodeError('stored block length mismatch');
      out = this.buffer.slice(bodyStart, bodyEnd);
    } else if (mode === MODE_FAST || mode === MODE_SMALL) {
      // Mirrors the frame containers: a conforming non-stored body is strictly smaller than
      // the stored body, keeping blocks canonical and allocations bounded.
      if (bodyLength >= rawLength) throw new TokzipDecodeError('non-canonical block: body not smaller than stored');
      const history = this.carry && this.history.length > 0 ? this.history : undefined;
      out =
        mode === MODE_FAST
          ? decodeFastBodyBinary(this.buffer, bodyStart, bodyEnd, rawLength, this.language, false, history)
          : decodeSmallBodyBinary(this.buffer, bodyStart, bodyEnd, rawLength, this.language, false, history);
    } else {
      throw new TokzipDecodeError('invalid mode');
    }
    this.offset = bodyEnd;
    // Keep the window's worth of produced output as the next block's history.
    if (this.carry) {
      if (out.length >= this.window) this.history = out.slice(out.length - this.window);
      else {
        const keep = Math.min(this.window, this.history.length + out.length);
        const merged = new Uint8Array(keep);
        const fromHistory = keep - out.length;
        if (fromHistory > 0) merged.set(this.history.subarray(this.history.length - fromHistory));
        merged.set(out, fromHistory);
        this.history = merged;
      }
    }
    controller.enqueue(out);
    return true;
  }
}
